"""S1-driven rare-token blocking with scored Top-K pruning.

For every Source-1 record:
  1. keys = its ``name_keys`` rarest name tokens and ``address_keys`` rarest address
     tokens (rarity = country-specific document frequency), skipping any token whose
     block (document frequency) exceeds ``max_block_size``;
  2. candidates = every S2/S3 record of the *same country* that contains a key token
     in the same field;
  3. each candidate is scored from the IDF weight of the name and address tokens it
     shares with the S1 record, and only the ``top_k`` best are kept. Scoring modes:
       * ``idf_sum``  - summed IDF of shared tokens over both fields;
       * ``coverage`` - per field, shared IDF / S1 total IDF (0..1), summed over the
         fields; a field the candidate leaves empty takes the other field's value,
         so a name-only record is not outranked by neighbours at the same address.
  4. optionally, part of the Top-K budget is reserved for the best candidates by
     name evidence alone (``name_quota``) and by address evidence alone
     (``address_quota``); the rest is filled by the combined score.

Candidates are consumed as a stream, so S2/S3 never need to be held in memory: only
the S1 key index and one bounded heap per S1 record.
"""

import heapq
import math
from dataclasses import dataclass

from .token_stats import ADDRESS, FIELDS, NAME, record_token_sets

SCORE_MODES = ("idf_sum", "coverage")
VIEWS = ("combined", "name", "address")


@dataclass(frozen=True)
class BlockingConfig:
    """Defaults are the recommended baseline (docs/blocking_data_analysis.md, section 9)."""

    name_keys: int = 2
    address_keys: int = 2
    max_block_size: int = 20000
    top_k: int = 100
    name_weight: float = 1.0
    address_weight: float = 1.0
    name_quota: float = 0.3  # share of top_k reserved for the best name-only matches
    address_quota: float = 0.0  # share of top_k reserved for the best address-only matches
    # If no key survives ``max_block_size``, keep the single rarest name or address
    # token as long as its block is at most this size (0 disables the fallback).
    fallback_max_block_size: int = 50000
    score_mode: str = "idf_sum"  # "idf_sum" or "coverage"

    def keys_for(self, field):
        return self.name_keys if field == NAME else self.address_keys


class S1Profile:
    """Pre-computed blocking keys and token weights for one S1 record."""

    __slots__ = ("entity_id", "country", "weights", "totals", "field_weight", "score_mode", "keys")

    def __init__(self, record, stats, config):
        self.entity_id = record.entity_id
        self.country = record.country
        toks = record_token_sets(record)
        if config.score_mode not in SCORE_MODES:
            raise ValueError(f"unknown score_mode {config.score_mode!r}")
        self.score_mode = config.score_mode
        self.field_weight = {NAME: config.name_weight, ADDRESS: config.address_weight}
        self.weights = {f: {t: stats.idf(f, record.country, t) for t in toks[f]} for f in FIELDS}
        self.totals = {f: math.fsum(self.weights[f].values()) for f in FIELDS}
        self.keys = [
            (f, t)
            for f in FIELDS
            for t, df in stats.rarest(f, record.country, toks[f], config.keys_for(f))
            if df <= config.max_block_size
        ]
        if not self.keys and config.fallback_max_block_size:
            rarest = [(df, f, t) for f in FIELDS for t, df in stats.rarest(f, record.country, toks[f], 1)]
            if rarest:
                df, f, t = min(rarest)
                if df <= config.fallback_max_block_size:
                    self.keys = [(f, t)]


def build_profiles(s1_records, stats, config):
    return [S1Profile(r, stats, config) for r in s1_records]


def build_key_index(profiles):
    """``(field, country, token) -> [profile index]``."""
    index = {}
    for i, p in enumerate(profiles):
        for f, t in p.keys:
            index.setdefault((f, p.country, t), []).append(i)
    return index


def score(profile, cand_tokens):
    """Combined similarity of a candidate to the S1 profile (see module docstring)."""
    return score_views(profile, cand_tokens)[0]


def score_views(profile, cand_tokens):
    """``(combined, name evidence, address evidence)`` for one candidate."""
    shared = {}
    for f in FIELDS:
        w = profile.weights[f]
        # fsum is exactly rounded, so the result does not depend on set iteration
        # order (which varies between processes) and Top-K ties stay reproducible.
        shared[f] = math.fsum(w[t] for t in cand_tokens[f] if t in w)
    fw = profile.field_weight
    if profile.score_mode == "idf_sum":
        combined = fw[NAME] * shared[NAME] + fw[ADDRESS] * shared[ADDRESS]
    else:
        sim = {f: (shared[f] / profile.totals[f] if profile.totals[f] else 0.0) for f in FIELDS}
        if not cand_tokens[ADDRESS]:
            sim[ADDRESS] = sim[NAME]
        elif not cand_tokens[NAME]:
            sim[NAME] = sim[ADDRESS]
        combined = fw[NAME] * sim[NAME] + fw[ADDRESS] * sim[ADDRESS]
    return combined, shared[NAME], shared[ADDRESS]


def blocked_profiles(index, country, cand_tokens):
    """Indices of S1 profiles that share at least one key with the candidate."""
    hits = set()
    for f in FIELDS:
        for t in cand_tokens[f]:
            lst = index.get((f, country, t))
            if lst:
                hits.update(lst)
    return hits


class TopKCollector:
    """Bounded min-heaps per S1 profile, one per view, of ``(view score, combined, id)``."""

    def __init__(self, n_profiles, capacity):
        self.capacity = capacity
        self.heaps = [{v: [] for v in VIEWS} for _ in range(n_profiles)]
        self.pool_sizes = [0] * n_profiles

    def push(self, i, scores, entity_id):
        """``scores`` is ``(combined, name, address)`` as returned by ``score_views``."""
        self.pool_sizes[i] += 1
        combined = scores[0]
        for view, s in zip(VIEWS, scores):
            if s <= 0:
                continue
            h = self.heaps[i][view]
            item = (s, combined, entity_id)
            if len(h) < self.capacity:
                heapq.heappush(h, item)
            elif item > h[0]:
                heapq.heapreplace(h, item)

    def merge(self, other):
        for i, (mine, theirs) in enumerate(zip(self.heaps, other.heaps)):
            self.pool_sizes[i] += other.pool_sizes[i]
            for v in VIEWS:
                if theirs[v]:
                    merged = heapq.nlargest(self.capacity, mine[v] + theirs[v])
                    heapq.heapify(merged)
                    mine[v] = merged

    def ranked(self, i, view="combined"):
        """Retained candidates of profile ``i`` for ``view``, best first."""
        return sorted(self.heaps[i][view], reverse=True)

    def select(self, i, top_k, name_quota=0.0, address_quota=0.0):
        """Final candidate IDs for profile ``i``: reserved name / address slots first,
        remaining slots by combined score. Returned best-combined first."""
        chosen = {}

        def take(view, n):
            added = 0
            for _, combined, eid in self.ranked(i, view):
                if added >= n:
                    break
                if eid not in chosen:
                    chosen[eid] = combined
                    added += 1

        take("name", int(round(top_k * name_quota)))
        take("address", int(round(top_k * address_quota)))
        take("combined", top_k - len(chosen))
        return [eid for eid, _ in sorted(chosen.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)]


def process_candidates(profiles, index, candidates, collector, on_pair=None):
    """Stream S2/S3 ``Record`` objects through the index into ``collector``.

    ``on_pair(profile_index, record, score)`` is called for every blocked pair
    (used by evaluation to trace true matches).
    """
    for rec in candidates:
        toks = record_token_sets(rec)
        hits = blocked_profiles(index, rec.country, toks)
        if not hits:
            continue
        for i in hits:
            scores = score_views(profiles[i], toks)
            collector.push(i, scores, rec.entity_id)
            if on_pair is not None:
                on_pair(i, rec, scores[0])


def generate_candidates(s1_records, candidate_records, stats, config):
    """In-memory convenience wrapper: ``{s1_id: [candidate ids]}`` after Top-K."""
    profiles = build_profiles(s1_records, stats, config)
    index = build_key_index(profiles)
    collector = TopKCollector(len(profiles), max(1, config.top_k))
    process_candidates(profiles, index, candidate_records, collector)
    return {
        p.entity_id: collector.select(i, config.top_k, config.name_quota, config.address_quota)
        for i, p in enumerate(profiles)
    }
