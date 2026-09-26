"""Country-specific token document frequencies for names and addresses.

Frequencies are computed over all three sources of a split (no labels used), so the
same procedure applies unchanged to the test split. Tokens seen only once are not
stored: an absent token has document frequency 1.
"""

import math
import os
import pickle
from collections import Counter
from multiprocessing import Pool

from .data_io import chunk_ranges, iter_records, source_path
from .normalize import address_tokens, name_tokens

NAME, ADDRESS = "n", "a"
FIELDS = (NAME, ADDRESS)


def record_token_sets(record):
    return {NAME: set(name_tokens(record.name)), ADDRESS: set(address_tokens(record.address))}


class TokenStats:
    """Document frequencies keyed by ``(country, token)`` per field."""

    def __init__(self, df, country_counts, min_df_stored=2):
        self.df = df  # {field: {(country, token): count}}
        self.country_counts = country_counts  # {country: records across sources}
        self.min_df_stored = min_df_stored

    @classmethod
    def from_records(cls, records, min_df_stored=1):
        df = {f: Counter() for f in FIELDS}
        countries = Counter()
        _accumulate(records, df, countries)
        return cls._finalise(df, countries, min_df_stored)

    @classmethod
    def _finalise(cls, df, countries, min_df_stored):
        pruned = {f: {k: v for k, v in df[f].items() if v >= min_df_stored} for f in FIELDS}
        return cls(pruned, dict(countries), min_df_stored)

    def doc_freq(self, field, country, token):
        return self.df[field].get((country, token), 1)

    def idf(self, field, country, token):
        n = self.country_counts.get(country, 1)
        return math.log((n + 1) / self.doc_freq(field, country, token))

    def rarest(self, field, country, tokens, m):
        """The ``m`` rarest tokens as ``(token, df)``; ties broken by token text."""
        ranked = sorted((self.doc_freq(field, country, t), t) for t in tokens)
        return [(t, d) for d, t in ranked[:m]]

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"df": self.df, "country_counts": self.country_counts,
                         "min_df_stored": self.min_df_stored}, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            d = pickle.load(f)
        return cls(d["df"], d["country_counts"], d["min_df_stored"])


def _accumulate(records, df, countries):
    for r in records:
        countries[r.country] += 1
        toks = record_token_sets(r)
        for f in FIELDS:
            for t in toks[f]:
                df[f][r.country, t] += 1


def _count_chunk(task):
    path, start, end = task
    df = {f: Counter() for f in FIELDS}
    countries = Counter()
    _accumulate(iter_records(path, start, end), df, countries)
    return df, countries


def build_token_stats(data_dir, split="train", workers=6, chunks_per_file=4, min_df_stored=2):
    tasks = []
    for s in (1, 2, 3):
        path = source_path(data_dir, split, s)
        tasks += [(path, a, b) for a, b in chunk_ranges(path, chunks_per_file)]
    df = {f: Counter() for f in FIELDS}
    countries = Counter()
    with Pool(workers) as pool:
        for part_df, part_countries in pool.imap_unordered(_count_chunk, tasks):
            for f in FIELDS:
                df[f].update(part_df[f])
            countries.update(part_countries)
    return TokenStats._finalise(df, countries, min_df_stored)


def load_or_build(cache_path, data_dir, split="train", workers=6):
    if os.path.exists(cache_path):
        return TokenStats.load(cache_path)
    stats = build_token_stats(data_dir, split, workers)
    stats.save(cache_path)
    return stats
