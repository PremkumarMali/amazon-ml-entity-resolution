"""Test inference: blocking -> features -> model -> exclusivity -> threshold -> both submission files.
Two passes: pass 1 blocks every chunk and records each candidate's top-2 cross-entity scores over ALL S1s,
pass 2 builds features with them, so a competing S1 in another chunk still counts (as it does in training).

python -m src.matching.predict --model data/interim/matcher.pkl
python -m src.matching.predict --limit 3000      # quick smoke run on the first 3000 test S1
"""
import argparse
import os
import pickle
import time

import numpy as np
import pandas as pd

from src.blocking.baseline_tokens import iter_by_country, read
from src.matching.matcher import build_features, exclusive, to_matches, top2, write_matching_results, xtop

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="data/interim/matcher.pkl")
ap.add_argument("--split", default="test")
ap.add_argument("--limit", type=int, help="only the first N S1 (smoke test)")
ap.add_argument("--chunk", type=int, default=20000, help="S1 per feature batch; lower if RAM is tight")
ap.add_argument("--out", default="output")
a = ap.parse_args()
t0 = time.time()

with open(a.model, "rb") as fh:
    m = pickle.load(fh)
d = f"data/raw/{a.split}"
s1 = read(f"{d}/{a.split}_source1.tsv")
if a.limit:
    s1 = s1.head(a.limit)
others = pd.concat([read(f"{d}/{a.split}_source2.tsv"), read(f"{d}/{a.split}_source3.tsv")], ignore_index=True)
print(f"loaded {time.time() - t0:.0f}s: {len(s1)} S1, {len(others)} S2+S3", flush=True)

tf = m.get("tfidf")  # None: pickle from before the TF-IDF fix (per-chunk fit, no cross-entity features)
chunks, tops, done = [], [], 0
for pairs, records in iter_by_country(s1, others, chunk=a.chunk):
    chunks.append(pairs)
    if tf is not None:
        tops.append(xtop(pairs, records, tf))
    done += pairs["s1_id"].nunique()
    print(f"  pass 1: {done}/{len(s1)} S1 blocked, {time.time() - t0:.0f}s", flush=True)
xt = None
if tops:
    xt = {}
    for c in tops[0]:
        d = pd.concat(t[c] for t in tops)
        xt[c] = top2(d["s"], d["c"])
del tops

# keep only compact columns across chunks: exclusivity needs every pair's prob before deciding
s1i, others = s1.set_index("entity_id"), others.set_index("entity_id")
parts, done = [], 0
while chunks:
    pairs = chunks.pop(0)
    records = pd.concat([s1i.loc[pairs["s1_id"].unique()], others.loc[pairs["cand_id"].unique()]]).reset_index()
    X = build_features(pairs, records, tf, xt)[m["features"]]
    parts.append(pairs[["s1_id", "cand_id"]].assign(prob=m["model"].predict_proba(X)[:, 1].astype(np.float32)))
    done += pairs["s1_id"].nunique()
    print(f"  pass 2: {done}/{len(s1)} S1 scored, {time.time() - t0:.0f}s", flush=True)
scored = pd.concat(parts, ignore_index=True)
del parts

prob = exclusive(scored, scored["prob"].to_numpy())
matches = to_matches(scored, prob, m["threshold"], s1["entity_id"])

os.makedirs(a.out, exist_ok=True)
write_matching_results(matches, f"{a.out}/matching_results.tsv")
cands = scored.groupby("s1_id")["cand_id"].agg(set).to_dict()
write_matching_results({s: cands.get(s, set()) for s in s1["entity_id"]}, f"{a.out}/candidate_pairs.tsv",
                       col="candidate_entity_ids")
print(f"wrote {a.out}/ in {time.time() - t0:.0f}s | S1 with a match: "
      f"{np.mean([bool(v) for v in matches.values()]):.3f}")
