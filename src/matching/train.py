"""Train + validate the matcher on blocked train pairs.

python -m src.matching.train --pairs data/interim/train_pairs.parquet data/interim/city_pairs.parquet
Several pair files are merged. Reads the S1 list blocking saved next to each (<pairs>_s1.parquet), so
S1 entities that got zero candidates still count in the macro-F0.5.
"""
import argparse
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.blocking.baseline_tokens import read
from src.matching.matcher import (best_threshold, build_features, exclusive, label_pairs, macro_f05, to_matches,
                                  train)

ap = argparse.ArgumentParser()
ap.add_argument("--pairs", nargs="+", default=["data/interim/train_pairs.parquet"])
ap.add_argument("--model", default="data/interim/matcher.pkl")
a = ap.parse_args()
t0 = time.time()

pairs = pd.concat(map(pd.read_parquet, a.pairs)).drop_duplicates(["s1_id", "cand_id"]).reset_index(drop=True)
ids = pd.concat(pd.read_parquet(p.replace(".parquet", "_s1.parquet")) for p in a.pairs).entity_id
s1 = read("data/raw/train/train_source1.tsv")
s1 = s1[s1.entity_id.isin(ids)]
want = set(pairs["cand_id"])
others = pd.concat(ch[ch.entity_id.isin(want)] for f in ("train_source2", "train_source3")
                   for ch in pd.read_csv(f"data/raw/train/{f}.tsv", sep="\t", dtype=str, chunksize=1_000_000))
records = pd.concat([s1, others], ignore_index=True)
gt = pd.read_csv("data/raw/train/train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
gt = gt[gt.source1_entity_id.isin(s1.entity_id)]
truth = {s: set(filter(None, m.split(","))) for s, m in zip(gt.source1_entity_id, gt.matched_entity_ids)}
print(f"loaded {time.time() - t0:.0f}s: {len(pairs)} pairs, {len(truth)} S1")

X, y = build_features(pairs, records), label_pairs(pairs, truth)
print(f"features {time.time() - t0:.0f}s: {X.shape}, positive rate {y.mean():.3f}")

from sklearn.model_selection import GroupKFold
oof = np.zeros(len(pairs))
for tr, va in GroupKFold(5).split(X, y, pairs["s1_id"]):
    oof[va] = train(X.iloc[tr], y[tr]).predict_proba(X.iloc[va])[:, 1]
t, score = best_threshold(pairs, oof, truth)
ceiling = macro_f05(to_matches(pairs, y.astype(float), 0.5, truth), truth)
print(f"CV {time.time() - t0:.0f}s | pair AUC {roc_auc_score(y, oof):.4f}")
print(f"macro-F0.5 {score:.4f} @ threshold {t} | blocking ceiling {ceiling:.4f}")

ex = exclusive(pairs, oof)
te, score_ex = best_threshold(pairs, ex, truth)
print(f"with exclusivity: macro-F0.5 {score_ex:.4f} @ threshold {te} ({(ex < oof).sum()} pairs zeroed)")

pred = to_matches(pairs, oof, t, truth)
single = {s for s, v in truth.items() if not v}
print(f"singletons: {np.mean([not pred[s] for s in single]):.3f} correctly empty ({len(single)})")
print(f"matched S1: F0.5 {macro_f05(pred, {s: v for s, v in truth.items() if v}):.4f}")

pairs.assign(prob=oof, label=y).to_parquet(a.pairs[0].replace(".parquet", "_oof.parquet"))  # for P4 error analysis
with open(a.model, "wb") as fh:
    pickle.dump({"model": train(X, y), "threshold": te,  # predict.py always applies exclusive()
                 "features": list(X.columns)}, fh)
print(f"saved model + oof, total {time.time() - t0:.0f}s")
