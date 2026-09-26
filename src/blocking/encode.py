"""Tokenise a split once into hashed integer tokens (a reusable, cached index).

For every source file of a split this stores, in file order:
  ids        entity ids (bytes)
  country    int16 country code (codes listed in ``meta.json``)
  flags      uint8 record flags (``features.FLAG_*``)
  <f>_ptr    CSR row pointers for field ``f`` (int64, n+1)
  <f>_tok    64-bit token hashes for field ``f`` (int64), unique within a row

Hashing is stable (blake2b) so encodings are identical across runs and machines.
"""

import hashlib
import json
import os
from multiprocessing import Pool

import numpy as np

from .data_io import chunk_ranges, iter_records, source_path
from .features import ALL_FIELDS, FEATURE_VERSION, record_features, record_flags

ENCODING_VERSION = f"f{FEATURE_VERSION}-e1"


def token_hash(field, token):
    digest = hashlib.blake2b(f"{field}\x1f{token}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=True)


def encode_records(records, fields=ALL_FIELDS, cache=None):
    """Encode ``Record``s into (ids, countries, flags, {field: (lengths, tokens)})."""
    cache = {} if cache is None else cache
    ids, countries, flags = [], [], []
    lengths = {f: [] for f in fields}
    tokens = {f: [] for f in fields}
    for r in records:
        ids.append(r.entity_id)
        countries.append(r.country)
        flags.append(record_flags(r.name, r.address))
        feats = record_features(r.name, r.address, with_fallback="g" in fields)
        for f in fields:
            hs = []
            for t in feats[f]:
                key = (f, t)
                h = cache.get(key)
                if h is None:
                    h = cache[key] = token_hash(f, t)
                hs.append(h)
            hs.sort()
            lengths[f].append(len(hs))
            tokens[f].extend(hs)
    return ids, countries, flags, {
        f: (np.asarray(lengths[f], dtype=np.int32), np.asarray(tokens[f], dtype=np.int64)) for f in fields
    }


def _encode_chunk(task):
    path, start, end, fields = task
    return encode_records(iter_records(path, start, end), fields)


class EncodedSource:
    """In-memory view of one encoded source file."""

    def __init__(self, arrays, country_names):
        self.ids = arrays["ids"]
        self.country = arrays["country"]
        self.flags = arrays["flags"]
        self.country_names = country_names
        self.fields = {f: (arrays[f"{f}_ptr"], arrays[f"{f}_tok"]) for f in ALL_FIELDS if f"{f}_ptr" in arrays}

    def __len__(self):
        return len(self.ids)

    def rows_of_country(self, name):
        if name not in self.country_names:
            return np.zeros(0, dtype=np.int64)
        return np.nonzero(self.country == self.country_names.index(name))[0]

    def row_tokens(self, field, rows):
        """(lengths, concatenated tokens) of ``rows`` for ``field``, in the given row order."""
        ptr, tok = self.fields[field]
        rows = np.asarray(rows, dtype=np.int64)
        starts, ends = ptr[rows], ptr[rows + 1]
        lengths = ends - starts
        if lengths.sum() == 0:
            return lengths, np.zeros(0, dtype=np.int64)
        offsets = np.repeat(starts - np.concatenate(([0], np.cumsum(lengths)[:-1])), lengths)
        return lengths, tok[np.arange(lengths.sum()) + offsets]


def cache_dir(root, split):
    return os.path.join(root, split, f"encoded_{ENCODING_VERSION}")


def encode_split(data_dir, split, root, workers=8, chunks_per_file=16, fields=ALL_FIELDS):
    """Encode all three sources of ``split`` (cached). Returns {source: EncodedSource}."""
    out_dir = cache_dir(root, split)
    meta_path = os.path.join(out_dir, "meta.json")
    if os.path.exists(meta_path):
        return load_split(root, split)
    os.makedirs(out_dir, exist_ok=True)
    country_names = []
    with Pool(workers) as pool:
        for s in (1, 2, 3):
            path = source_path(data_dir, split, s)
            tasks = [(path, a, b, fields) for a, b in chunk_ranges(path, chunks_per_file)]
            ids, countries, flags = [], [], []
            parts = {f: ([], []) for f in fields}
            for c_ids, c_countries, c_flags, c_fields in pool.imap(_encode_chunk, tasks):  # ordered
                ids.extend(c_ids)
                countries.extend(c_countries)
                flags.extend(c_flags)
                for f in fields:
                    parts[f][0].append(c_fields[f][0])
                    parts[f][1].append(c_fields[f][1])
            for c in countries:
                if c not in country_names:
                    country_names.append(c)
            code = {c: i for i, c in enumerate(country_names)}
            arrays = {
                "ids": np.asarray(ids, dtype="S"),
                "country": np.asarray([code[c] for c in countries], dtype=np.int16),
                "flags": np.asarray(flags, dtype=np.uint8),
            }
            for f in fields:
                lengths = np.concatenate(parts[f][0])
                arrays[f"{f}_ptr"] = np.concatenate(([0], np.cumsum(lengths, dtype=np.int64)))
                arrays[f"{f}_tok"] = np.concatenate(parts[f][1])
            tmp = os.path.join(out_dir, f"source{s}.tmp.npz")
            np.savez(tmp, **arrays)
            os.replace(tmp, os.path.join(out_dir, f"source{s}.npz"))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"version": ENCODING_VERSION, "split": split, "countries": country_names,
                   "fields": list(fields)}, f, indent=2)
    return load_split(root, split)


def load_split(root, split):
    out_dir = cache_dir(root, split)
    with open(os.path.join(out_dir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    result = {}
    for s in (1, 2, 3):
        with np.load(os.path.join(out_dir, f"source{s}.npz")) as z:
            result[s] = EncodedSource({k: z[k] for k in z.files}, meta["countries"])
    return result
