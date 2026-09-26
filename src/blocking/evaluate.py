"""Evaluate the vectorised candidate generator on the fixed validation sample.

Usage (from the repository root):

    python -m src.blocking.evaluate                                   # final config, K=100
    python -m src.blocking.evaluate --variant final --top-k 25,50,100,200
    python -m src.blocking.evaluate --variant final_no_rerank         # ablations: see VARIANTS
    python -m src.blocking.evaluate --weights n=1,p=0.5,pb=0.5,a=1.5,ab=1 --sparse-quotas 0 --nonlatin-boosts 1

Uses the same deterministic sample as ``run_experiments`` (hash rate 0.25%, salt
``blocking-validation-v1``: 5,509 S1 entities / 19,108 true pairs), blocked
against the full S2/S3 training files. Writes JSON to ``--out-dir``.
"""

import argparse
import json
import os
import time
from collections import Counter

import numpy as np

from .data_io import iter_records, source_path
from .encode import encode_split
from .engine import CandidateConfig, CountryIndex, select
from .normalize import address_tokens, has_nonlatin, name_tokens
from .run_experiments import load_sample, summarize
from .sampling import DEFAULT_SALT

W_TOKENS = (("n", 1.0), ("p", 0.0), ("pb", 0.0), ("a", 1.5), ("ab", 0.0))
_D = CandidateConfig()

VARIANTS = {
    # ablations of the recommended default: one component removed at a time
    "final": _D,
    "final_no_phonetic_or_pair_fields": CandidateConfig(weights=W_TOKENS),
    "final_no_rerank": CandidateConfig(rerank_depth=0),
    "final_no_sparse_quota": CandidateConfig(sparse_quota=0.0),
    "final_no_nonlatin_boost": CandidateConfig(nonlatin_boost=1.0),
    "final_with_trigram_fallback": CandidateConfig(fallback_below=1000),
}


def peak_memory_mb():
    try:
        import psutil

        info = psutil.Process().memory_info()
        return round(getattr(info, "peak_wset", info.rss) / 2 ** 20)
    except Exception:
        return None


def load_true_texts(data_dir, ids):
    texts = {}
    for s in (2, 3):
        for r in iter_records(source_path(data_dir, "train", s)):
            if r.entity_id in ids:
                texts[r.entity_id] = r
    return texts


def evaluate_config(config, enc, s1_by_id, truth, texts, index_cache=None):
    t0 = time.perf_counter()
    index_cache = {} if index_cache is None else index_cache
    ids1 = enc[1].ids
    sample_ids = np.asarray(sorted(truth), dtype="S")
    rows = np.nonzero(np.isin(ids1, sample_ids))[0]
    hits, totals = Counter(), Counter()
    sizes, pools = [], []
    lost = Counter()
    entity_full = entity_n = 0
    kept_all = {}
    for country in enc[1].country_names:
        crow = rows[enc[1].country[rows] == enc[1].country_names.index(country)]
        if len(crow) == 0:
            continue
        key = (country, config.index_key())
        if key not in index_cache:
            index_cache[key] = CountryIndex.build(enc, country, config)
        index = index_cache[key]
        true_ids = np.asarray(sorted({c for r in crow for c in truth[ids1[r].decode()]}), dtype="S")
        pos = np.nonzero(np.isin(index.cand_ids, true_ids))[0]
        cand_pos = {index.cand_ids[p].decode(): int(p) for p in pos}
        for a in range(0, len(crow), config.shard_size):
            shard = crow[a:a + config.shard_size]
            toks = {f: enc[1].row_tokens(f, shard) for f in config.fields}
            sc = index.score(toks, config)
            chosen = select(index, sc, config)
            for i, r in enumerate(shard):
                s1_id = ids1[r].decode()
                cands = set(index.cand_ids[chosen[i][0]].astype(str))
                kept_all[s1_id] = cands
                sizes.append(len(cands))
                pools.append(int(sc.pool.indptr[i + 1] - sc.pool.indptr[i]))
                row_cols = sc.pool.indices[sc.pool.indptr[i]:sc.pool.indptr[i + 1]]
                ids = truth[s1_id]
                if ids:
                    entity_n += 1
                    entity_full += all(c in cands for c in ids)
                for c in ids:
                    t = texts[c]
                    groups = ["all", country, c[:2]] + (["nonlatin_name"] if has_nonlatin(t.name) else [])
                    ok = c in cands
                    for g in groups:
                        totals[g] += 1
                        hits[g] += ok
                    if not ok:
                        in_pool = c in cand_pos and bool((row_cols == cand_pos[c]).any())
                        stage = "pruned_by_top_k" if in_pool else "not_blocked"
                        lost[stage] += 1
                        lost[f"{stage}|{country}"] += 1
                        s1 = s1_by_id[s1_id]
                        flags = {
                            "cand_name_nonlatin": has_nonlatin(t.name),
                            "cand_address_empty": not address_tokens(t.address),
                            "no_shared_name_token": not set(name_tokens(s1.name)) & set(name_tokens(t.name)),
                            "no_shared_address_token": not set(address_tokens(s1.address)) & set(address_tokens(t.address)),
                        }
                        for k, v in flags.items():
                            lost[f"{stage}|{k}"] += v
    recall = {g: round(100 * hits[g] / totals[g], 2) for g in totals}
    return {
        "config": config.to_dict(),
        "recall": recall,
        "true_pairs": dict(totals),
        "entity_all_matches_kept": round(100 * entity_full / max(1, entity_n), 2),
        "candidates": summarize(sizes),
        "pool": summarize(pools),
        "s1_with_zero_candidates": sum(1 for x in sizes if x == 0),
        "lost": dict(sorted(lost.items())),
        "seconds": round(time.perf_counter() - t0, 1),
        "peak_memory_mb": peak_memory_mb(),
    }, kept_all


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data/raw/train")
    ap.add_argument("--cache-dir", default="data/processed/blocking")
    ap.add_argument("--out-dir", default="output/blocking_experiments")
    ap.add_argument("--sample-rate", type=float, default=0.0025)
    ap.add_argument("--salt", default=DEFAULT_SALT)
    ap.add_argument("--variant", action="append", help=f"one of {sorted(VARIANTS)} (default: final)")
    ap.add_argument("--weights", action="append", help="extra variant, e.g. n=1,p=0.5,pb=0.5,a=1,ab=1")
    ap.add_argument("--rerank-depth", type=int, default=None)
    ap.add_argument("--fallback-below", default=None, help="comma list; >0 enables the trigram fallback")
    ap.add_argument("--nonlatin-boosts", default=None, help="comma list (default: variant's own)")
    ap.add_argument("--top-k", default="100")
    ap.add_argument("--name-quotas", default=None, help="comma list (default: variant's own)")
    ap.add_argument("--sparse-quotas", default=None, help="comma list (default: variant's own)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)

    enc = encode_split(args.data_dir, "train", args.cache_dir, workers=args.workers)
    s1_records, truth = load_sample(args.data_dir, args.sample_rate, args.salt)
    s1_by_id = {r.entity_id: r for r in s1_records}
    texts = load_true_texts(args.data_dir, {c for ids in truth.values() for c in ids})
    print(f"sample: {len(s1_records):,} S1, {sum(len(v) for v in truth.values()):,} true pairs", flush=True)

    for w in args.weights or []:
        pairs = tuple((k, float(v)) for k, v in (x.split("=") for x in w.split(",")))
        VARIANTS[w] = CandidateConfig(weights=pairs)
    names = (args.variant or []) + (args.weights or []) or ["final"]
    index_cache = {}
    results = []
    for name in names:
        base = VARIANTS[name]
        for k in [int(x) for x in args.top_k.split(",")]:
            def opts(arg, own, cast):
                return [cast(x) for x in arg.split(",")] if arg else [own]

            grid = [(a, b, c, d) for a in opts(args.name_quotas, base.name_quota, float)
                    for b in opts(args.sparse_quotas, base.sparse_quota, float)
                    for c in opts(args.fallback_below, base.fallback_below, int)
                    for d in opts(args.nonlatin_boosts, base.nonlatin_boost, float)]
            for q, sq, fb, boost in grid:
                over = {"top_k": k, "name_quota": q, "sparse_quota": sq, "fallback_below": fb,
                        "nonlatin_boost": boost}
                if args.rerank_depth is not None:
                    over["rerank_depth"] = args.rerank_depth
                cfg = CandidateConfig.from_dict({**base.to_dict(), **over})
                m, _ = evaluate_config(cfg, enc, s1_by_id, truth, texts, index_cache)
                m["variant"] = name
                results.append(m)
                r = m["recall"]
                print(f"{name:24s} K={k:<4d} q={q:<4} sq={sq:<4} fb={fb:<5} boost={boost:<3} recall {r['all']:6.2f}% US {r.get('US', 0):6.2f} "
                      f"India {r.get('India', 0):6.2f} nonLatin {r.get('nonlatin_name', 0):6.2f} | "
                      f"cands {m['candidates']} pool mean {m['pool']['mean']} | "
                      f"not_blocked {m['lost'].get('not_blocked', 0)} pruned {m['lost'].get('pruned_by_top_k', 0)} "
                      f"| {m['seconds']}s peak {m['peak_memory_mb']} MB", flush=True)
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, time.strftime("engine_%Y%m%d_%H%M%S.json"))
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
