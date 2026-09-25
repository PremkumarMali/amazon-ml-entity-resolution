"""P3 — pairwise matching: features, model, threshold, final matches.

Contract:
  records: DataFrame with entity_id, business_name, business_address, country
           (S1+S2+S3 stacked; P1 may overwrite name/address with cleaned text)
  pairs:   DataFrame with s1_id, cand_id   (from P2 blocking)
  truth:   {s1_id: set(matched ids)} for EVERY S1 entity (empty set = singleton)
"""
import re
import unicodedata

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

_NUM = re.compile(r"\d+")
# legal forms / honorifics / junk seen in the data; French forms included for the unseen test country
_STOP = set("""the sri shri dr mr mrs ms m s co company corp corporation inc incorporated llc llp ltd limited
pvt private pc lp plc group formerly dba com www and null nan sarl sas sa sasu eurl sci""".split())


def _norm(s):
    # ponytail: fallback cleaning only; P1's normalizer replaces this upstream
    if pd.isna(s):
        return []
    s = "".join(c for c in unicodedata.normalize("NFKD", str(s).lower()) if not unicodedata.combining(c))
    return [t.lstrip("0") or "0" if t.isdigit() else t for t in re.findall(r"\w+", s)]


def _jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 0.0


def _nonascii(s):
    return sum(ord(c) > 127 for c in s) / max(len(s), 1)


def _tfidf_cos(left, right):
    # each S1 text repeats once per candidate: vectorize unique texts once, then index
    codes, uniq = pd.factorize(pd.Series(left + right, dtype=object))
    M = normalize(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, dtype=np.float32)
                  .fit_transform(uniq))
    L, R = M[codes[:len(left)]], M[codes[len(left):]]
    return np.asarray(L.multiply(R).sum(axis=1)).ravel()


def _pair(scorer, x, y, scale=100):
    return process.cpdist(x, y, scorer=scorer, workers=-1) / scale


def build_features(pairs, records):
    rec = records.set_index("entity_id")
    a, b = rec.loc[pairs["s1_id"]], rec.loc[pairs["cand_id"]]
    f = pd.DataFrame(index=pairs.index)
    toks = {}
    for col, key in (("business_name", "name"), ("business_address", "addr")):
        va, vb = a[col].fillna("").tolist(), b[col].fillna("").tolist()
        cache = {v: _norm(v) for v in set(va) | set(vb)}
        wa, wb = [cache[x] for x in va], [cache[x] for x in vb]
        toks[key] = (wa, wb)
        ta, tb = [" ".join(w) for w in wa], [" ".join(w) for w in wb]
        f[f"{key}_ratio"] = _pair(fuzz.ratio, ta, tb)
        f[f"{key}_jw"] = _pair(JaroWinkler.similarity, ta, tb, 1)
        f[f"{key}_tsort"] = _pair(fuzz.token_sort_ratio, ta, tb)
        f[f"{key}_tset"] = _pair(fuzz.token_set_ratio, ta, tb)
        f[f"{key}_partial"] = _pair(fuzz.partial_ratio, ta, tb)
        f[f"{key}_jacc"] = [_jaccard(x, y) for x, y in zip(wa, wb)]
        f[f"{key}_tfidf"] = _tfidf_cos(ta, tb)
        f[f"{key}_lendiff"] = [abs(len(x) - len(y)) / max(len(x), len(y), 1) for x, y in zip(ta, tb)]
        f[f"{key}_missing"] = [int(not x or not y) for x, y in zip(ta, tb)]
        f[f"{key}_nonascii"] = [max(_nonascii(x), _nonascii(y)) for x, y in zip(ta, tb)]

    # name without legal forms/honorifics, and joined (catches "bigfoods.com")
    ca = [[t for t in w if t not in _STOP] for w in toks["name"][0]]
    cb = [[t for t in w if t not in _STOP] for w in toks["name"][1]]
    f["core_tsort"] = _pair(fuzz.token_sort_ratio, [" ".join(w) for w in ca], [" ".join(w) for w in cb])
    f["core_jacc"] = [_jaccard(x, y) for x, y in zip(ca, cb)]
    f["joined_ratio"] = _pair(fuzz.ratio, ["".join(w) for w in ca], ["".join(w) for w in cb])
    f["joined_partial"] = _pair(fuzz.partial_ratio, ["".join(w) for w in ca], ["".join(w) for w in cb])

    # numbers in address: house no / PIN / ZIP agreement (true matches sometimes differ, so soft features)
    na = [[t for t in w if t.isdigit()] for w in toks["addr"][0]]
    nb = [[t for t in w if t.isdigit()] for w in toks["addr"][1]]
    f["num_jacc"] = [_jaccard(x, y) for x, y in zip(na, nb)]
    f["num_conflict"] = [int(bool(x) and bool(y) and not set(x) & set(y)) for x, y in zip(na, nb)]
    f["first_num_eq"] = [int(x[0] == y[0]) if x and y else -1 for x, y in zip(na, nb)]
    f["same_country"] = (a["country"].values == b["country"].values).astype(int)  # never one-hot: France is unseen
    f["is_s3"] = pairs["cand_id"].str.startswith("S3-").astype(int).values
    if "block_score" in pairs:
        f["block_score"] = pairs["block_score"].values

    # rank features: how this candidate compares to its S1's other candidates
    g = pairs["s1_id"].values
    for c in ("name_tfidf", "name_tsort", "core_tsort", "addr_tfidf") + (("block_score",) if "block_score" in f else ()):
        mx = f.groupby(g)[c].transform("max")
        f[f"{c}_gap"] = mx - f[c]
        f[f"{c}_rank"] = f.groupby(g)[c].rank(ascending=False, method="min")
    f["n_cands"] = f.groupby(g)["name_ratio"].transform("size")
    return f


def label_pairs(pairs, truth):
    return np.array([c in truth.get(s, ()) for s, c in zip(pairs["s1_id"], pairs["cand_id"])], dtype=int)


def macro_f05(pred, truth):
    """pred/truth: {s1_id: set}. Averaged over ALL S1 in truth, singletons included."""
    scores = []
    for s, t in truth.items():
        p = pred.get(s, set())
        if not t:
            scores.append(1.0 if not p else 0.0)
            continue
        tp = len(p & t)
        if tp == 0:
            scores.append(0.0)
            continue
        pr, rc = tp / len(p), tp / len(t)
        scores.append(1.25 * pr * rc / (0.25 * pr + rc))
    return float(np.mean(scores))


def exclusive(pairs, prob):
    """Each S2/S3 record belongs to at most one S1 (true in train GT): only its best-scoring S1 may keep it.
    Only bites when competing S1s are scored together, i.e. full test inference, not a random train sample."""
    best = pd.Series(prob).groupby(pairs["cand_id"].to_numpy()).transform("max").to_numpy()
    return np.where(prob >= best, prob, 0.0)


def to_matches(pairs, prob, threshold, s1_ids):
    keep = pairs[prob >= threshold]
    out = {s: set() for s in s1_ids}
    for s, c in zip(keep["s1_id"], keep["cand_id"]):
        out[s].add(c)
    return out


def train(X, y, **kw):
    params = dict(max_iter=500, learning_rate=0.05, max_leaf_nodes=31, early_stopping=True, random_state=0)
    return HistGradientBoostingClassifier(**{**params, **kw}).fit(X, y)


def best_threshold(pairs, prob, truth):
    grid = np.round(np.arange(0.2, 0.96, 0.01), 2)
    scores = [macro_f05(to_matches(pairs, prob, t, truth), truth) for t in grid]
    i = int(np.argmax(scores))
    return float(grid[i]), scores[i]


def cross_validate(pairs, records, truth, n_splits=5):
    """GroupKFold by S1 entity -> out-of-fold probs, best threshold, macro-F0.5."""
    from sklearn.model_selection import GroupKFold

    X, y = build_features(pairs, records), label_pairs(pairs, truth)
    oof = np.zeros(len(pairs))
    for tr, va in GroupKFold(n_splits).split(X, y, pairs["s1_id"]):
        oof[va] = train(X.iloc[tr], y[tr]).predict_proba(X.iloc[va])[:, 1]
    t, score = best_threshold(pairs, oof, truth)
    return oof, t, score


def write_matching_results(matches, path, col="matched_entity_ids"):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(f"source1_entity_id\t{col}\n")
        for s, ids in matches.items():
            fh.write(f"{s}\t{','.join(sorted(ids))}\n")


if __name__ == "__main__":
    # self-check on a tiny synthetic set
    assert abs(macro_f05({"a": {"x", "y", "z"}}, {"a": {"x", "z"}}) - 0.714) < 1e-3
    assert macro_f05({"a": set()}, {"a": set()}) == 1.0
    assert macro_f05({"a": {"x"}}, {"a": set()}) == 0.0
    pp = pd.DataFrame({"s1_id": ["a", "b", "b"], "cand_id": ["x", "x", "y"]})
    assert list(exclusive(pp, np.array([0.9, 0.7, 0.8]))) == [0.9, 0.0, 0.8]

    rng = np.random.default_rng(0)
    names = ["abc technologies", "sharma traders", "blue ocean cafe", "delta motors", "zenith pharma",
             "orbit logistics", "green leaf foods", "nova textiles", "apex builders", "sun electricals"]
    rows, truth, pairs = [], {}, []
    for i, n in enumerate(names * 6):
        s1 = f"S1-{i:05d}"
        rows.append((s1, n.upper() + " PVT LTD", f"{i} MG Road Bengaluru", "India"))
        truth[s1] = set()
        if i % 3:  # 2/3 have a true match with noise
            m = f"S2-{i:05d}"
            rows.append((m, n.replace("technologies", "tech") + " ltd", f"{i}, M.G. Rd, Bangalore", "India"))
            truth[s1].add(m)
            pairs.append((s1, m))
        d = f"S3-{i:05d}"  # hard negative: other name, same street
        rows.append((d, names[(i + 3) % 10], f"{int(rng.integers(500, 900))} MG Road", "India"))
        pairs.append((s1, d))
    records = pd.DataFrame(rows, columns=["entity_id", "business_name", "business_address", "country"])
    pairs = pd.DataFrame(pairs, columns=["s1_id", "cand_id"])
    oof, t, score = cross_validate(pairs, records, truth, n_splits=3)
    print(f"synthetic CV macro-F0.5={score:.3f} at threshold={t}")
    assert score > 0.9
    print("ok")
