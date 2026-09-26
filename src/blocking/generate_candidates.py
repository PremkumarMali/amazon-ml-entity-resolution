"""Generate blocking candidates for a whole split (scalable, resumable).

Usage (from the repository root, Python with requirements.txt installed):

    python -m src.blocking.generate_candidates --split test  --top-k 100
    python -m src.blocking.generate_candidates --split train --top-k 100

Pipeline
  1. encode the split once into hashed tokens (cached under --cache-dir);
  2. per country, build (or reuse) a memory-mapped candidate index;
  3. split that country's S1 rows (file order) into fixed shards and score them in
     a bounded process pool; every shard is written atomically to
     ``<out-dir>/<split>/shards/<country>/<shard>.parquet``; existing shards are
     skipped, so an interrupted run resumes where it stopped;
  4. stream the shards back in S1 file order into ``candidate_pairs.tsv``
     (``source1_entity_id<TAB>comma-separated candidate ids``, one row per S1
     entity, empty when there is no candidate).

Only S1 entities are ever held per shard; S2/S3 live in the shared index. Output
is deterministic for a given config (see docs/blocking_handoff.md).
"""

import argparse
import json
import os
import shutil
import time
from multiprocessing import Pool

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .encode import ENCODING_VERSION, cache_dir, encode_split
from .engine import CandidateConfig, CountryIndex, select

SHARD_SCHEMA = pa.schema([
    ("s1_row", pa.int32()),
    ("s1_entity_id", pa.string()),
    ("rank", pa.int16()),
    ("candidate_entity_id", pa.string()),
    ("candidate_source", pa.int8()),
    ("candidate_row", pa.int32()),
    ("score", pa.float32()),
    ("name_score", pa.float32()),
])


def default_candidate_pairs_path(out_dir, split):
    # The official submission file lives at output/candidate_pairs.tsv (test split).
    if split == "test":
        return os.path.join(os.path.dirname(os.path.normpath(out_dir)) or ".", "candidate_pairs.tsv")
    return os.path.join(out_dir, split, "candidate_pairs.tsv")


def plan_shards(enc, config, s1_limit=None):
    """Deterministic shard plan: ``[(country, shard_name, s1_rows)]``."""
    s1 = enc[1]
    allowed = None if s1_limit is None else np.arange(min(s1_limit, len(s1)))
    plan = []
    for country in sorted(s1.country_names):
        rows = s1.rows_of_country(country)
        if allowed is not None:
            rows = rows[rows < len(allowed)]
        for i, a in enumerate(range(0, len(rows), config.shard_size)):
            plan.append((country, f"{i:05d}", rows[a:a + config.shard_size]))
    return plan


def shard_path(run_dir, country, name):
    return os.path.join(run_dir, "shards", country, f"{name}.parquet")


# ---------------------------------------------------------------- workers
_W = {}


def _init_worker(index_dir, config_dict):
    _W["index"] = CountryIndex.load(index_dir, mmap=True)
    _W["config"] = CandidateConfig.from_dict(config_dict)


def score_shard(index, config, s1_rows, s1_ids, s1_tokens):
    """Top-K candidates for one shard as a pyarrow Table (SHARD_SCHEMA)."""
    sc = index.score(s1_tokens, config)
    chosen = select(index, sc, config)
    counts = np.asarray([len(c[0]) for c in chosen], dtype=np.int64)
    cand = np.concatenate([c[0] for c in chosen]).astype(np.int64) if counts.sum() else np.zeros(0, np.int64)
    ranks = np.concatenate([np.arange(1, n + 1) for n in counts]) if counts.sum() else np.zeros(0, np.int64)
    table = pa.table({
        "s1_row": np.repeat(np.asarray(s1_rows, dtype=np.int32), counts),
        "s1_entity_id": np.repeat(np.asarray(s1_ids).astype(str), counts),
        "rank": ranks.astype(np.int16),
        "candidate_entity_id": np.asarray(index.cand_ids[cand]).astype(str),
        "candidate_source": np.asarray(index.cand_source[cand], dtype=np.int8),
        "candidate_row": np.asarray(index.cand_row[cand], dtype=np.int32),
        "score": np.concatenate([c[1] for c in chosen]).astype(np.float32) if counts.sum() else np.zeros(0, np.float32),
        "name_score": np.concatenate([c[2] for c in chosen]).astype(np.float32) if counts.sum() else np.zeros(0, np.float32),
    }, schema=SHARD_SCHEMA)
    pool = np.diff(sc.pool.indptr)
    return table, {"s1": len(s1_rows), "pairs": int(counts.sum()), "pool_mean": float(pool.mean()) if len(pool) else 0.0,
                   "zero_candidates": int((counts == 0).sum())}


def _run_shard(task):
    out_path, s1_rows, s1_ids, s1_tokens = task
    t0 = time.perf_counter()
    table, stats = score_shard(_W["index"], _W["config"], s1_rows, s1_ids, s1_tokens)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + ".tmp"
    pq.write_table(table, tmp)
    os.replace(tmp, out_path)
    stats["seconds"] = time.perf_counter() - t0
    try:
        import psutil

        info = psutil.Process().memory_info()
        stats["peak_mb"] = getattr(info, "peak_wset", info.rss) / 2 ** 20
    except Exception:
        stats["peak_mb"] = None
    return stats


# ---------------------------------------------------------------- assembly
def iter_s1_candidates(run_dir, enc, config, s1_limit=None):
    """Yield ``(s1_entity_id, [candidate ids])`` for every planned S1 row, in S1 file order."""
    plan = plan_shards(enc, config, s1_limit)
    by_country = {}
    for country, name, rows in plan:
        by_country.setdefault(country, []).append((name, rows))

    def country_stream(country):
        for name, rows in by_country[country]:
            t = pq.read_table(shard_path(run_dir, country, name), columns=["s1_row", "candidate_entity_id"])
            s1r = t.column("s1_row").to_numpy()
            cids = t.column("candidate_entity_id").to_pylist()
            bounds = np.searchsorted(s1r, rows, side="left"), np.searchsorted(s1r, rows, side="right")
            for r, a, b in zip(rows, *bounds):
                yield int(r), cids[a:b]

    streams = {c: country_stream(c) for c in by_country}
    order = np.sort(np.concatenate([rows for _, _, rows in plan])) if plan else np.zeros(0, np.int64)
    names = enc[1].country_names
    ids = enc[1].ids
    for r in order:
        got_row, cands = next(streams[names[enc[1].country[r]]])
        assert got_row == r, (got_row, r)
        yield ids[r].decode(), cands


def write_candidate_pairs(path, rows):
    """Write the official TSV from ``(s1_id, [candidate ids])`` pairs; returns stats."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    n = empty = total = 0
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id, cands in rows:
            if len(set(cands)) != len(cands) or any(not c.startswith(("S2-", "S3-")) for c in cands):
                raise ValueError(f"invalid candidate list for {s1_id}")
            f.write(f"{s1_id}\t{','.join(cands)}\n")
            n += 1
            empty += not cands
            total += len(cands)
    os.replace(tmp, path)
    return {"rows": n, "empty_rows": empty, "candidates": total}


# ---------------------------------------------------------------- driver
def run(split, data_dir, cache_root, out_dir, config, workers=6, s1_limit=None, overwrite=False,
        candidate_pairs=None, write_tsv=True, log=print):
    t0 = time.perf_counter()
    enc = encode_split(data_dir, split, cache_root, workers=max(workers, 4))
    run_dir = os.path.join(out_dir, split)
    run_meta = {"split": split, "encoding": ENCODING_VERSION, "config": config.to_dict(), "s1_limit": s1_limit}
    meta_path = os.path.join(run_dir, "run.json")
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            old = json.load(f)
        if old != run_meta:
            if not overwrite:
                raise SystemExit(f"{run_dir} holds a run with a different config; use --overwrite to replace it")
            shutil.rmtree(run_dir)
    os.makedirs(run_dir, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    plan = plan_shards(enc, config, s1_limit)
    totals = {"s1": 0, "pairs": 0, "zero_candidates": 0, "shards_done": 0, "shards_skipped": 0}
    shard_seconds, peaks = [], []
    for country in sorted({c for c, _, _ in plan}):
        index_dir = os.path.join(cache_dir(cache_root, split), f"index_{config.index_key()}", country)
        if not CountryIndex.exists(index_dir):
            t = time.perf_counter()
            CountryIndex.build(enc, country, config).save(index_dir)
            log(f"[{country}] index built in {time.perf_counter() - t:.0f}s")
        todo = [(c, n, r) for c, n, r in plan if c == country and not os.path.exists(shard_path(run_dir, c, n))]
        totals["shards_skipped"] += sum(1 for c, n, _ in plan if c == country) - len(todo)
        if not todo:
            continue

        def tasks():
            for c, n, rows in todo:
                toks = {f: enc[1].row_tokens(f, rows) for f in config.fields}
                yield shard_path(run_dir, c, n), rows, enc[1].ids[rows], toks

        log(f"[{country}] scoring {len(todo)} shards with {workers} workers")
        with Pool(workers, initializer=_init_worker, initargs=(index_dir, config.to_dict())) as pool:
            for i, st in enumerate(pool.imap_unordered(_run_shard, tasks()), 1):
                totals["s1"] += st["s1"]
                totals["pairs"] += st["pairs"]
                totals["zero_candidates"] += st["zero_candidates"]
                totals["shards_done"] += 1
                shard_seconds.append(st["seconds"])
                if st["peak_mb"]:
                    peaks.append(st["peak_mb"])
                if i % 50 == 0 or i == len(todo):
                    log(f"[{country}] {i}/{len(todo)} shards, {time.perf_counter() - t0:.0f}s elapsed")
    summary = {**totals, "seconds": round(time.perf_counter() - t0, 1),
               "mean_shard_seconds": round(float(np.mean(shard_seconds)), 2) if shard_seconds else None,
               "worker_peak_mb": round(max(peaks)) if peaks else None}
    if write_tsv:
        path = candidate_pairs or default_candidate_pairs_path(out_dir, split)
        summary["candidate_pairs"] = {"path": path, **write_candidate_pairs(
            path, iter_s1_candidates(run_dir, enc, config, s1_limit))}
    summary["total_seconds"] = round(time.perf_counter() - t0, 1)
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log(json.dumps(summary, indent=2))
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--data-dir", default=None, help="default: data/raw/<split>")
    ap.add_argument("--cache-dir", default="data/processed/blocking")
    ap.add_argument("--out-dir", default="output/candidates")
    ap.add_argument("--candidate-pairs", default=None,
                    help="TSV path (default: output/candidate_pairs.tsv for test, "
                         "output/candidates/train/candidate_pairs.tsv for train)")
    ap.add_argument("--top-k", type=int, default=CandidateConfig.top_k)
    ap.add_argument("--name-quota", type=float, default=CandidateConfig.name_quota)
    ap.add_argument("--shard-size", type=int, default=CandidateConfig.shard_size)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--s1-limit", type=int, default=None, help="only the first N S1 rows (smoke tests)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no-tsv", action="store_true", help="only write shards")
    args = ap.parse_args(argv)
    config = CandidateConfig.from_dict({**CandidateConfig().to_dict(), "top_k": args.top_k,
                                        "name_quota": args.name_quota, "shard_size": args.shard_size})
    run(args.split, args.data_dir or os.path.join("data", "raw", args.split), args.cache_dir, args.out_dir, config,
        workers=args.workers, s1_limit=args.s1_limit, overwrite=args.overwrite,
        candidate_pairs=args.candidate_pairs, write_tsv=not args.no_tsv)


if __name__ == "__main__":
    main()
