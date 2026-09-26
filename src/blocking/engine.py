"""Vectorised candidate generation (scipy sparse products instead of pair loops).

For one country:
  * vocabulary per field = tokens whose document frequency (over S1+S2+S3 of that
    country) is at most ``max_df``; IDF = log((N + 1) / df);
  * ``CT[f]`` = binary incidence matrix (token x candidate) of all S2/S3 records;
  * for a shard of S1 rows, ``W[f]`` holds the S1 tokens' IDF weights and
    ``W[f] @ CT[f]`` gives, for every candidate sharing at least one such token,
    the summed IDF of the shared tokens. The union over fields is the candidate
    pool (blocking) and the weighted sum is the ranking score.

Re-ranking: tokens above ``max_df`` (city, state, "road", ...) are too common to
block on, but they still separate a true match from a same-name business in another
city. For each row's ``rerank_depth`` best candidates (by combined and by name
score) the exact score over *all* tokens is computed by adding the shared common
tokens' IDF (sorted-key membership tests, no dense products).

Top-K per S1 row: ``round(top_k * name_quota)`` slots go to the best candidates by
name evidence (fields n, p, pb), ``round(top_k * sparse_quota)`` to the best
candidates flagged as non-Latin name / empty address (their scores are structurally
low), the rest to the best combined score. Phonetic evidence of non-Latin
candidate names is multiplied by ``nonlatin_boost`` (they cannot match on ``n``).
Ties are broken by candidate index, so output is fully deterministic.

Optional character-trigram fallback (field ``g``): only for S1 rows whose
combined pool has fewer than ``fallback_below`` candidates; trigram candidates are
merged into the pool with their own (scaled) score before Top-K.
"""

import json
import os
from collections import namedtuple
from dataclasses import asdict, dataclass, field

import numpy as np
import scipy.sparse as sp

from .features import ADDRESS_FIELDS, FLAG_NONLATIN_NAME, NAME_FIELDS

DEFAULT_WEIGHTS = (("n", 1.0), ("p", 0.5), ("pb", 0.5), ("a", 1.5), ("ab", 1.0))
SCORE_FIELDS = NAME_FIELDS + ADDRESS_FIELDS
PHONETIC_FIELDS = ("p", "pb")

# pool: blocked pairs (rare-token scores); combined/name: scores used for Top-K
ShardScores = namedtuple("ShardScores", "pool combined name")


@dataclass(frozen=True)
class CandidateConfig:
    """Defaults = recommended configuration (docs/blocking_handoff.md)."""

    top_k: int = 100
    name_quota: float = 0.3
    sparse_quota: float = 0.15  # share of top_k for best candidates with non-Latin names / empty addresses
    # A non-Latin candidate name can only match through phonetic keys (p, pb); their
    # weight is multiplied by this factor for such candidates.
    nonlatin_boost: float = 3.0
    max_df: int = 20000
    weights: tuple = DEFAULT_WEIGHTS
    fallback_below: int = 0  # 0 disables the trigram fallback
    fallback_max_df: int = 2000
    fallback_weight: float = 0.5
    rerank_depth: int = 500  # 0 disables exact re-ranking
    shard_size: int = 1000

    @property
    def weight_map(self):
        return dict(self.weights)

    @property
    def fields(self):
        f = [k for k, w in self.weights if w > 0]
        return tuple(f) + (("g",) if self.fallback_below > 0 else ())

    def to_dict(self):
        d = asdict(self)
        d["weights"] = [list(x) for x in self.weights]
        return d

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        d["weights"] = tuple(tuple(x) for x in d["weights"])
        return cls(**d)

    def index_key(self):
        """Parameters that change the country index (for caching)."""
        g = self.fallback_max_df if self.fallback_below else 0
        return (f"df{self.max_df}-g{g}-r{int(self.rerank_depth > 0)}-b{self.nonlatin_boost:g}-"
                + "-".join(self.fields))


def _lookup(vocab, toks):
    """Column index of each token in the sorted ``vocab`` (-1 if absent)."""
    if len(vocab) == 0 or len(toks) == 0:
        return np.full(len(toks), -1, dtype=np.int64)
    col = np.searchsorted(vocab, toks)
    col[col >= len(vocab)] = 0
    col[vocab[col] != toks] = -1
    return col


@dataclass
class CountryIndex:
    country: str
    n_records: int
    vocab: dict = field(default_factory=dict)  # field -> sorted int64 hashes
    idf: dict = field(default_factory=dict)  # field -> float32
    ct: dict = field(default_factory=dict)  # field -> csr (V x n_candidates)
    cand_source: np.ndarray = None  # int8 (2 or 3)
    cand_row: np.ndarray = None  # int32 row in that source file
    cand_ids: np.ndarray = None  # bytes
    cand_flags: np.ndarray = None  # uint8 (features.FLAG_*)
    # tokens above max_df (used only for re-ranking)
    cvocab: dict = field(default_factory=dict)  # field -> sorted hashes
    cidf: dict = field(default_factory=dict)  # field -> float32
    ckeys: dict = field(default_factory=dict)  # field -> sorted int64 (candidate * V + column)

    @classmethod
    def build(cls, enc, country, config):
        rows = {s: enc[s].rows_of_country(country) for s in (1, 2, 3)}
        n_records = sum(len(r) for r in rows.values())
        idx = cls(country=country, n_records=n_records)
        idx.cand_source = np.concatenate([np.full(len(rows[s]), s, dtype=np.int8) for s in (2, 3)])
        idx.cand_row = np.concatenate([rows[2], rows[3]]).astype(np.int32)
        idx.cand_ids = np.concatenate([enc[2].ids[rows[2]], enc[3].ids[rows[3]]])
        idx.cand_flags = np.concatenate([enc[2].flags[rows[2]], enc[3].flags[rows[3]]])
        n_cand = len(idx.cand_row)
        for f in config.fields:
            max_df = config.fallback_max_df if f == "g" else config.max_df
            per_source = {s: enc[s].row_tokens(f, rows[s]) for s in (1, 2, 3)}
            all_toks = np.concatenate([per_source[s][1] for s in (1, 2, 3)])
            uniq, df = np.unique(all_toks, return_counts=True)
            keep = df <= max_df
            idx.vocab[f] = uniq[keep]
            idx.idf[f] = np.log((n_records + 1) / df[keep]).astype(np.float32)
            lengths = np.concatenate([per_source[2][0], per_source[3][0]])
            toks = np.concatenate([per_source[2][1], per_source[3][1]])
            cand = np.repeat(np.arange(n_cand, dtype=np.int64), lengths)
            col = _lookup(idx.vocab[f], toks)
            ok = col >= 0
            vals = np.ones(int(ok.sum()), dtype=np.float32)
            if f in PHONETIC_FIELDS and config.nonlatin_boost != 1.0:
                nonlatin = (idx.cand_flags[cand[ok]] & FLAG_NONLATIN_NAME) != 0
                vals[nonlatin] = config.nonlatin_boost
            idx.ct[f] = sp.csr_matrix(
                (vals, (col[ok], cand[ok])),
                shape=(len(idx.vocab[f]), n_cand),
            )
            if f in SCORE_FIELDS and config.rerank_depth > 0:
                common = ~keep
                idx.cvocab[f] = uniq[common]
                idx.cidf[f] = np.log((n_records + 1) / df[common]).astype(np.float32)
                ccol = _lookup(idx.cvocab[f], toks)
                cok = ccol >= 0
                idx.ckeys[f] = np.sort(cand[cok] * max(1, len(idx.cvocab[f])) + ccol[cok])
        return idx

    # ---------------------------------------------------------- persistence
    def _arrays(self):
        arrays = {"cand_source": self.cand_source, "cand_row": self.cand_row, "cand_ids": self.cand_ids,
                  "cand_flags": self.cand_flags}
        for f in self.cvocab:
            arrays[f"cvocab_{f}"], arrays[f"cidf_{f}"], arrays[f"ckeys_{f}"] = self.cvocab[f], self.cidf[f], self.ckeys[f]
        for f in self.vocab:
            m = self.ct[f]
            arrays[f"vocab_{f}"], arrays[f"idf_{f}"] = self.vocab[f], self.idf[f]
            arrays[f"ct_{f}_data"], arrays[f"ct_{f}_indices"], arrays[f"ct_{f}_indptr"] = m.data, m.indices, m.indptr
        return arrays

    def save(self, directory):
        """One ``.npy`` per array so workers can memory-map (and share) the index."""
        os.makedirs(directory, exist_ok=True)
        for name, arr in self._arrays().items():
            tmp = os.path.join(directory, f"{name}.tmp.npy")
            np.save(tmp, np.ascontiguousarray(arr))
            os.replace(tmp, os.path.join(directory, f"{name}.npy"))
        meta = {"country": self.country, "n_records": self.n_records, "fields": list(self.vocab),
                "common_fields": list(self.cvocab),
                "shapes": {f: list(self.ct[f].shape) for f in self.vocab}}
        tmp = os.path.join(directory, "index.tmp.json")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(meta, fh)
        os.replace(tmp, os.path.join(directory, "index.json"))  # written last: marks completion

    @classmethod
    def load(cls, directory, mmap=True):
        with open(os.path.join(directory, "index.json"), encoding="utf-8") as fh:
            meta = json.load(fh)
        mode = "r" if mmap else None

        def arr(name):
            return np.load(os.path.join(directory, f"{name}.npy"), mmap_mode=mode)

        idx = cls(country=meta["country"], n_records=meta["n_records"])
        idx.cand_source, idx.cand_row, idx.cand_ids = arr("cand_source"), arr("cand_row"), arr("cand_ids")
        idx.cand_flags = arr("cand_flags")
        for f in meta["fields"]:
            idx.vocab[f], idx.idf[f] = arr(f"vocab_{f}"), arr(f"idf_{f}")
            idx.ct[f] = sp.csr_matrix((arr(f"ct_{f}_data"), arr(f"ct_{f}_indices"), arr(f"ct_{f}_indptr")),
                                      shape=tuple(meta["shapes"][f]), copy=False)
        for f in meta.get("common_fields", []):
            idx.cvocab[f], idx.cidf[f], idx.ckeys[f] = arr(f"cvocab_{f}"), arr(f"cidf_{f}"), arr(f"ckeys_{f}")
        return idx

    @staticmethod
    def exists(directory):
        return os.path.exists(os.path.join(directory, "index.json"))

    @property
    def n_candidates(self):
        return len(self.cand_row)

    # ---------------------------------------------------------- scoring
    def weight_matrix(self, f, lengths, toks, weight=1.0):
        rows = np.repeat(np.arange(len(lengths), dtype=np.int64), lengths)
        col = _lookup(self.vocab[f], toks)
        ok = col >= 0
        vals = self.idf[f][col[ok]] * np.float32(weight)
        return sp.csr_matrix((vals, (rows[ok], col[ok])), shape=(len(lengths), len(self.vocab[f])))

    def score(self, s1_tokens, config):
        """``ShardScores(pool, combined, name)`` csr matrices (S1 rows x candidates).

        ``s1_tokens``: {field: (lengths, tokens)} for the shard's S1 rows.
        """
        wmap = config.weight_map
        n_rows = len(next(iter(s1_tokens.values()))[0])
        shape = (n_rows, self.n_candidates)
        name = sp.csr_matrix(shape, dtype=np.float32)
        addr = sp.csr_matrix(shape, dtype=np.float32)
        for f in NAME_FIELDS + ADDRESS_FIELDS:
            w = wmap.get(f, 0.0)
            if w <= 0 or f not in self.vocab:
                continue
            part = self.weight_matrix(f, *s1_tokens[f], weight=w) @ self.ct[f]
            if f in NAME_FIELDS:
                name = name + part
            else:
                addr = addr + part
        combined = (name + addr).tocsr()
        if config.fallback_below > 0 and "g" in self.vocab:
            combined = self._with_fallback(combined, s1_tokens, config)
        combined.sort_indices()
        name = name.tocsr()
        name.sort_indices()
        if config.rerank_depth > 0 and self.cvocab:
            rc, rn = self._rerank(combined, name, s1_tokens, config, self.sparse_mask(config))
            return ShardScores(combined, rc, rn)
        return ShardScores(combined, combined, name)

    def sparse_mask(self, config):
        """Candidates with structurally low scores (non-Latin name / empty address)."""
        if config.sparse_quota <= 0:
            return None
        if getattr(self, "_sparse_mask", None) is None:
            self._sparse_mask = np.asarray(self.cand_flags) != 0
        return self._sparse_mask

    def _rerank(self, combined, name, s1_tokens, config, sparse=None):
        """Exact all-token scores for each row's shortlist (see module docstring).

        The shortlist holds the ``rerank_depth`` best candidates by combined score and
        by name score, plus (if ``sparse`` is given) the best flagged candidates, so
        the sparse quota can reach them.
        """
        depth = config.rerank_depth
        sparse_depth = max(1, int(round(config.top_k * config.sparse_quota))) * 5
        rr, jj, cv, nv = [], [], [], []
        for r in range(combined.shape[0]):
            a, b = combined.indptr[r], combined.indptr[r + 1]
            cols, vals = combined.indices[a:b], combined.data[a:b]
            na, nb = name.indptr[r], name.indptr[r + 1]
            ncols, nvals = name.indices[na:nb], name.data[na:nb]
            short = np.union1d(cols[_top_n(vals, cols, depth)], ncols[_top_n(nvals, ncols, depth)])
            if sparse is not None:
                m = sparse[cols]
                short = np.union1d(short, cols[m][_top_n(vals[m], cols[m], sparse_depth)])
            rr.append(np.full(len(short), r, dtype=np.int64))
            jj.append(short.astype(np.int64))
            cv.append(_gather(cols, vals, short))
            nv.append(_gather(ncols, nvals, short))
        R, J = np.concatenate(rr), np.concatenate(jj)
        c_vals, n_vals = np.concatenate(cv).astype(np.float64), np.concatenate(nv).astype(np.float64)
        wmap = config.weight_map
        for f in SCORE_FIELDS:
            w = wmap.get(f, 0.0)
            if w <= 0 or f not in self.cvocab or len(self.cvocab[f]) == 0 or len(R) == 0:
                continue
            extra = self._common_overlap(f, s1_tokens[f], R, J) * w
            if f in PHONETIC_FIELDS and config.nonlatin_boost != 1.0:
                nonlatin = (np.asarray(self.cand_flags)[J] & FLAG_NONLATIN_NAME) != 0
                extra = np.where(nonlatin, extra * config.nonlatin_boost, extra)
            c_vals += extra
            if f in NAME_FIELDS:
                n_vals += extra
        shape = combined.shape
        rc = sp.csr_matrix((c_vals.astype(np.float32), (R, J)), shape=shape)
        pos = n_vals > 0
        rn = sp.csr_matrix((n_vals[pos].astype(np.float32), (R[pos], J[pos])), shape=shape)
        rc.sort_indices()
        rn.sort_indices()
        return rc, rn

    def _common_overlap(self, f, row_tokens, R, J):
        """Summed IDF of common tokens shared by S1 row ``R[i]`` and candidate ``J[i]``."""
        lengths, toks = row_tokens
        col = _lookup(self.cvocab[f], toks)
        row_of_tok = np.repeat(np.arange(len(lengths)), lengths)
        ok = col >= 0
        row_of_tok, col = row_of_tok[ok], col[ok]
        counts = np.bincount(row_of_tok, minlength=len(lengths))
        starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
        per_pair = counts[R]
        if per_pair.sum() == 0:
            return np.zeros(len(R))
        pair = np.repeat(np.arange(len(R)), per_pair)
        offs = np.arange(per_pair.sum()) - np.repeat(np.cumsum(per_pair) - per_pair, per_pair)
        tcol = col[starts[R][pair] + offs]
        keys = J[pair] * max(1, len(self.cvocab[f])) + tcol
        ck = self.ckeys[f]
        if len(ck) == 0:
            return np.zeros(len(R))
        loc = np.minimum(np.searchsorted(ck, keys), len(ck) - 1)
        found = ck[loc] == keys
        return np.bincount(pair, weights=self.cidf[f][tcol] * found, minlength=len(R))

    def _with_fallback(self, combined, s1_tokens, config):
        pool = np.diff(combined.indptr)
        rows = np.nonzero(pool < config.fallback_below)[0]
        if len(rows) == 0:
            return combined
        lengths, toks = s1_tokens["g"]
        ptr = np.concatenate(([0], np.cumsum(lengths)))
        sub_len = lengths[rows]
        sub_tok = np.concatenate([toks[ptr[r]:ptr[r + 1]] for r in rows]) if sub_len.sum() else np.zeros(0, np.int64)
        tri = self.weight_matrix("g", sub_len, sub_tok, weight=config.fallback_weight) @ self.ct["g"]
        expand = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, np.arange(len(rows)))),
                               shape=(combined.shape[0], len(rows)))
        return (combined + expand @ tri).tocsr()


# ---------------------------------------------------------------- Top-K
def _top_n(vals, cols, n):
    """Positions of the ``n`` best entries by (-value, column): exact and deterministic."""
    if n <= 0 or len(vals) == 0:
        return np.zeros(0, dtype=np.int64)
    if len(vals) > n:
        part = np.argpartition(-vals, n - 1)[:n]
        sel = np.nonzero(vals >= vals[part].min())[0]  # keep every tie at the boundary
    else:
        sel = np.arange(len(vals))
    order = np.lexsort((cols[sel], -vals[sel]))[:n]
    return sel[order]


def select_top_k(combined, name, top_k, name_quota, sparse=None, sparse_quota=0.0):
    """Per S1 row: ``(candidate indices, combined scores, name scores)``, best first.

    Slots: ``name_quota`` share by name score, then ``sparse_quota`` share by combined
    score among candidates flagged in the boolean ``sparse`` array (indexed by
    candidate), then the rest by combined score; no candidate is taken twice.
    """
    n_slots = int(round(top_k * name_quota))
    s_slots = int(round(top_k * sparse_quota)) if sparse is not None else 0
    out = []
    for r in range(combined.shape[0]):
        a, b = combined.indptr[r], combined.indptr[r + 1]
        cols, vals = combined.indices[a:b], combined.data[a:b]
        na, nb = name.indptr[r], name.indptr[r + 1]
        ncols, nvals = name.indices[na:nb], name.data[na:nb]
        picked = ncols[_top_n(nvals, ncols, n_slots)] if n_slots else np.zeros(0, dtype=cols.dtype)
        if s_slots:
            m = sparse[cols] & ~np.isin(cols, picked)
            picked = np.concatenate([picked, cols[m][_top_n(vals[m], cols[m], s_slots)]])
        best = cols[_top_n(vals, cols, top_k)]
        rest = best[~np.isin(best, picked)][: max(0, top_k - len(picked))]
        chosen = np.concatenate([picked, rest])
        c_scores = _gather(cols, vals, chosen)
        n_scores = _gather(ncols, nvals, chosen)
        order = np.lexsort((chosen, -c_scores))
        out.append((chosen[order], c_scores[order], n_scores[order]))
    return out


def select(index, scores, config):
    """Top-K for a scored shard with the config's quotas (see ``select_top_k``)."""
    return select_top_k(scores.combined, scores.name, config.top_k, config.name_quota,
                        index.sparse_mask(config), config.sparse_quota)


def _gather(sorted_cols, vals, query):
    """Values at ``query`` columns of one sparse row (0 where absent)."""
    if len(sorted_cols) == 0 or len(query) == 0:
        return np.zeros(len(query), dtype=np.float32)
    pos = np.minimum(np.searchsorted(sorted_cols, query), len(sorted_cols) - 1)
    return np.where(sorted_cols[pos] == query, vals[pos], 0).astype(np.float32)
