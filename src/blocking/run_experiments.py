"""Blocking experiments on a reproducible validation sample of the training data.

Usage (from the repository root):

    python -m src.blocking.run_experiments
    python -m src.blocking.run_experiments --name-keys 2 --address-keys 2 \
        --max-block-sizes 5000,20000 --top-k 25,50,100,200

Only the sampled S1 entities are blocked (against the *full* S2/S3 files), so the
numbers are exact for that sample. Writes a JSON summary to ``--out-dir``; it does
not write ``candidate_pairs.tsv``.
"""

import argparse
import itertools
import json
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool

from .blocker import BlockingConfig, TopKCollector, build_key_index, build_profiles, process_candidates
from .data_io import chunk_ranges, ground_truth_path, iter_ground_truth, iter_records, source_path
from .normalize import address_tokens, has_nonlatin, name_tokens
from .sampling import DEFAULT_SALT, in_sample
from .token_stats import load_or_build


def peak_memory_mb():
    """Peak resident memory of the current process in MB (best effort)."""
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class PMC(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

            pmc = PMC()
            pmc.cb = ctypes.sizeof(PMC)
            psapi = ctypes.WinDLL("psapi")
            kernel = ctypes.WinDLL("kernel32")
            kernel.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
            psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
            return pmc.PeakWorkingSetSize / 2 ** 20
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except Exception:
        return float("nan")


# ---------------------------------------------------------------- sample
def load_sample(data_dir, rate, salt):
    truth = {}
    for s1, ids in iter_ground_truth(ground_truth_path(data_dir)):
        if in_sample(s1, rate, salt):
            truth[s1] = ids
    s1_records = [r for r in iter_records(source_path(data_dir, "train", 1)) if r.entity_id in truth]
    s1_records.sort(key=lambda r: r.entity_id)
    return s1_records, truth


# ---------------------------------------------------------------- workers
_W = {}


def _init_worker(profiles, index, capacity, true_pairs):
    _W.update(profiles=profiles, index=index, capacity=capacity, true_pairs=true_pairs,
              true_ids={cid for _, cid in true_pairs})


def _run_chunk(task):
    path, start, end = task
    collector = TopKCollector(len(_W["profiles"]), _W["capacity"])
    true_scores, true_texts = {}, {}
    true_pairs, true_ids = _W["true_pairs"], _W["true_ids"]
    profiles = _W["profiles"]

    def on_pair(i, rec, s):
        if (i, rec.entity_id) in true_pairs:
            true_scores[i, rec.entity_id] = s

    def records():
        for rec in iter_records(path, start, end):
            if rec.entity_id in true_ids:
                true_texts[rec.entity_id] = (rec.name, rec.address, rec.country)
            yield rec

    process_candidates(profiles, _W["index"], records(), collector, on_pair)
    return collector, true_scores, true_texts, peak_memory_mb()


# ---------------------------------------------------------------- metrics
def pct(values, q):
    if not values:
        return 0
    s = sorted(values)
    return s[min(len(s) - 1, int(q * len(s)))]


def summarize(values):
    return {"mean": round(statistics.fmean(values), 1) if values else 0,
            "median": pct(values, 0.5), "p95": pct(values, 0.95), "max": max(values) if values else 0}


def evaluate_selection(profiles, truth, collector, top_k, name_quota=0.0, address_quota=0.0):
    hits = Counter()
    totals = Counter()
    sizes = []
    entity_full = entity_n = 0
    kept = {}
    for i, p in enumerate(profiles):
        chosen = set(collector.select(i, top_k, name_quota, address_quota))
        kept[i] = chosen
        sizes.append(len(chosen))
        ids = truth[p.entity_id]
        if ids:
            entity_n += 1
            entity_full += all(c in chosen for c in ids)
        for c in ids:
            for key in ("all", p.country, c[:2]):
                totals[key] += 1
                hits[key] += c in chosen
    recall = {k: round(100 * hits[k] / totals[k], 2) for k in totals}
    return {"recall": recall, "entity_all_matches_kept": round(100 * entity_full / max(1, entity_n), 2),
            "candidates": summarize(sizes)}, kept


def rank_bucket(ranked_ids, cand):
    """Where a pruned true match sat in the retained ranking."""
    if cand not in ranked_ids:
        return "beyond_retained"
    r = ranked_ids.index(cand) + 1
    for b in (25, 50, 100, 200):
        if r <= b:
            return f"rank<={b}"
    return "rank>200"


def loss_analysis(profiles, truth, s1_by_id, true_scores, true_texts, collector, kept, capacity):
    cats = Counter()
    by_stage = Counter()
    ranks = Counter()
    for i, p in enumerate(profiles):
        s1 = s1_by_id[p.entity_id]
        s1_name, s1_addr = set(name_tokens(s1.name)), set(address_tokens(s1.address))
        ranked_ids = [eid for _, _, eid in collector.ranked(i)]
        for c in truth[p.entity_id]:
            if c in kept[i]:
                continue
            name, addr, country = true_texts[c]
            stage = "pruned_by_top_k" if (i, c) in true_scores else "not_blocked"
            by_stage[stage, country] += 1
            flags = {
                "cand_name_nonlatin": has_nonlatin(name),
                "cand_address_empty": not address_tokens(addr),
                "no_shared_name_token": not (s1_name & set(name_tokens(name))),
                "no_shared_address_token": not (s1_addr & set(address_tokens(addr))),
                "s1_has_no_usable_key": not p.keys,
            }
            for k, v in flags.items():
                if v:
                    cats[stage, k] += 1
            if stage == "pruned_by_top_k":
                ranks[rank_bucket(ranked_ids, c)] += 1
    return {"by_stage": {f"{s}|{c}": n for (s, c), n in sorted(by_stage.items())},
            "flags": {f"{s}|{k}": n for (s, k), n in sorted(cats.items())},
            "pruned_rank_buckets": dict(ranks), "rank_capacity": capacity}


# ---------------------------------------------------------------- driver
def run_config(config, s1_records, truth, data_dir, workers, chunks_per_file, capacity, stats):
    t0 = time.perf_counter()
    profiles = build_profiles(s1_records, stats, config)
    index = build_key_index(profiles)
    pos = {p.entity_id: i for i, p in enumerate(profiles)}
    true_pairs = {(pos[s1], c) for s1, ids in truth.items() for c in ids}
    tasks = []
    for s in (2, 3):
        path = source_path(data_dir, "train", s)
        tasks += [(path, a, b) for a, b in chunk_ranges(path, chunks_per_file)]
    collector = TopKCollector(len(profiles), capacity)
    true_scores, true_texts = {}, {}
    worker_peaks = []
    with Pool(workers, initializer=_init_worker, initargs=(profiles, index, capacity, true_pairs)) as pool:
        for part, ts, tt, peak in pool.imap_unordered(_run_chunk, tasks):
            collector.merge(part)
            true_scores.update(ts)
            true_texts.update(tt)
            worker_peaks.append(peak)
    elapsed = time.perf_counter() - t0
    return profiles, collector, true_scores, true_texts, elapsed, max(worker_peaks)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data/raw/train")
    ap.add_argument("--cache-dir", default="data/processed/blocking")
    ap.add_argument("--out-dir", default="output/blocking_experiments")
    ap.add_argument("--sample-rate", type=float, default=0.0025, help="fraction of S1 entities (~5.5k)")
    ap.add_argument("--salt", default=DEFAULT_SALT)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--chunks-per-file", type=int, default=6)
    ap.add_argument("--name-keys", default="2")
    ap.add_argument("--address-keys", default="2")
    ap.add_argument("--max-block-sizes", default="20000")
    ap.add_argument("--fallback-max-block-sizes", default="50000", help="0 disables the fallback key")
    ap.add_argument("--score-modes", default="idf_sum", help="comma list of idf_sum,coverage")
    ap.add_argument("--top-k", default="25,50,100,200")
    ap.add_argument("--name-quotas", default="0,0.3", help="shares of K reserved for best name-only matches")
    ap.add_argument("--address-quotas", default="0", help="shares of K reserved for best address-only matches")
    ap.add_argument("--address-weight", type=float, default=1.0)
    ap.add_argument("--name-weight", type=float, default=1.0)
    args = ap.parse_args(argv)
    ints = lambda s: [int(x) for x in s.split(",")]
    floats = lambda s: [float(x) for x in s.split(",")]
    quotas = [(nq, aq) for nq in floats(args.name_quotas) for aq in floats(args.address_quotas) if nq + aq < 1]
    top_ks = ints(args.top_k)
    capacity = max(top_ks)

    t = time.perf_counter()
    stats = load_or_build(os.path.join(args.cache_dir, "token_stats_train.pkl"), args.data_dir, "train", args.workers)
    print(f"token stats ready in {time.perf_counter() - t:.0f}s "
          f"(name vocab {len(stats.df['n']):,}, address vocab {len(stats.df['a']):,})", flush=True)
    s1_records, truth = load_sample(args.data_dir, args.sample_rate, args.salt)
    s1_by_id = {r.entity_id: r for r in s1_records}
    n_pairs = sum(len(v) for v in truth.values())
    print(f"sample: {len(s1_records):,} S1 entities "
          f"({sum(1 for v in truth.values() if not v):,} singletons), {n_pairs:,} true pairs", flush=True)

    results = {"sample": {"rate": args.sample_rate, "salt": args.salt, "s1": len(s1_records), "true_pairs": n_pairs},
               "runs": []}
    grid = itertools.product(ints(args.name_keys), ints(args.address_keys), ints(args.max_block_sizes),
                             ints(args.fallback_max_block_sizes), args.score_modes.split(","))
    for nk, ak, cap, fb, mode in grid:
        cfg = BlockingConfig(name_keys=nk, address_keys=ak, max_block_size=cap, top_k=capacity,
                             name_weight=args.name_weight, address_weight=args.address_weight,
                             fallback_max_block_size=fb, score_mode=mode)
        profiles, collector, true_scores, true_texts, elapsed, wpeak = run_config(
            cfg, s1_records, truth, args.data_dir, args.workers, args.chunks_per_file, capacity, stats)
        pool = collector.pool_sizes
        blocked_recall = 100 * len(true_scores) / max(1, n_pairs)
        run = {"name_keys": nk, "address_keys": ak, "max_block_size": cap, "fallback_max_block_size": fb, "score_mode": mode,
               "name_weight": args.name_weight, "address_weight": args.address_weight,
               "seconds": round(elapsed, 1), "worker_peak_mb": round(wpeak), "parent_peak_mb": round(peak_memory_mb()),
               "before_pruning": {"recall": round(blocked_recall, 2), "candidates": summarize(pool),
                                  "s1_without_keys": sum(1 for p in profiles if not p.keys)},
               "top_k": {}}
        print(f"\n== name_keys={nk} address_keys={ak} max_block={cap} fallback={fb} score={mode}: {elapsed:.0f}s, "
              f"worker peak {wpeak:.0f} MB | before pruning: recall {blocked_recall:.2f}%, "
              f"pool {summarize(pool)}", flush=True)
        for k in top_ks:
            for nq, aq in quotas:
                m, kept = evaluate_selection(profiles, truth, collector, k, nq, aq)
                m["loss"] = loss_analysis(profiles, truth, s1_by_id, true_scores, true_texts,
                                          collector, kept, capacity)
                run["top_k"][f"K={k} name_quota={nq} address_quota={aq}"] = m
                r = m["recall"]
                print(f"  K={k:<4d} nq={nq:<4} aq={aq:<4} recall {r['all']:6.2f}% (US {r.get('US', 0):6.2f}, "
                      f"India {r.get('India', 0):6.2f}, S2 {r.get('S2', 0):6.2f}, S3 {r.get('S3', 0):6.2f}) "
                      f"entity-all-kept {m['entity_all_matches_kept']:6.2f}% cands {m['candidates']}", flush=True)
        results["runs"].append(run)

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, time.strftime("results_%Y%m%d_%H%M%S.json"))
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
