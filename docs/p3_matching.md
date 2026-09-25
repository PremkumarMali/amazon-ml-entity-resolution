# P3: Matching Model (input for Documentation_template.md, sections 2.1, 4, 5)

Code: `src/matching/matcher.py` (features, model, decision rule, metric), `src/matching/train.py` (train and validate), `src/matching/predict.py` (test inference to both output files).

## 2.1 EDA findings that shaped the matcher

- Train has 2.2M S1 records and about 10.3M S2+S3 records. 5.6% of S1 are singletons; the mean is about 3.5 matches per S1, and the maximum is 11.
- Each S2/S3 record belongs to at most one S1. About 26% of S2/S3 records match no S1 at all, so they only act as distractors.
- Names:
  - native-script names: Hindi, Kannada, Telugu, Punjabi (9–11% of S2 records)
  - accents, and prefixes such as `***`, `The`, `Sri`, `Dr`
  - `formerly X`, `DBA: X`, brackets, domain-style names (`nelsonsmetrohealth.com`)
  - OCR-like typos (`Appare1s`) and word-order shuffles
- Addresses:
  - state names in native script, and state abbreviations (`MH` / `Maharashtra`)
  - reordered components, 3% missing or `NULL`
  - house numbers that differ even between true matches (`5425` vs `1425`)
- France appears only in test, so no feature one-hots `country`.

## 4. Matching model

**Unit:** one (S1, candidate) pair from blocking, labelled 1 if the candidate is in that S1's ground-truth list. Training uses only blocked candidates, so the training distribution matches inference.

**Features (43):**
- Name and address, each: Levenshtein ratio, Jaro-Winkler, token-sort, token-set and partial ratio, token Jaccard, character 2–4-gram TF-IDF cosine (vectorizers fitted once on the training texts and stored in `matcher.pkl`, so a pair scores the same in training and in any test chunk), length difference, missing flag, non-Latin script share.
- Core name, after removing legal forms and honorifics (including French SARL/SAS): token-sort ratio, Jaccard, and ratio and partial ratio of the joined name (catches domain-style names).
- Address numbers (leading zeros stripped): Jaccard, "both have numbers but none shared" flag, first-number equality.
- Context: same country, S2 vs S3, blocking score.
- Rank within the S1's candidates: gap to the best score and rank position, for name TF-IDF, name token-sort, core token-sort, address TF-IDF and blocking score. This lets the model reject the weaker of two lookalike candidates.
- Cross-entity, for name TF-IDF and blocking score: this S1's score minus the best score any *other* S1 gives the same candidate (0 if no other S1 has it). This targets the largest false-positive group, a different business at the same address. `predict.py` computes it in two passes (pass 1 keeps each candidate's top-2 scores over all test S1), so competing S1 entities in different 20k-S1 chunks still count, as they do in training. Checked on 119,658 dense-city pairs: two-pass values equal single-batch values exactly, while a naive per-chunk computation got 10% of rows wrong.

**Model:** sklearn `HistGradientBoostingClassifier` (BSD-3; gradient-boosted trees, same family as LightGBM). Settings: 500 iterations, learning rate 0.05, 31 leaves, early stopping.

**Decision rule:**
1. **Exclusivity:** each S2/S3 record may go only to the S1 that scores it highest (it belongs to at most one S1 in the ground truth).
2. **Threshold:** a pair counts as a match if its probability is at least 0.64. The threshold was chosen by maximising macro-F0.5 on out-of-fold predictions, after exclusivity.

A per-entity rule (keep candidates within x% of the entity's best score, with a separate "has any match" gate) was tested and gave no gain (+0.0001), so it was dropped.

**Validation:** 5-fold GroupKFold by S1 entity, so no S1 appears in both train and validation folds. The metric is the official macro-F0.5 over all sampled S1, including singletons and S1 with zero candidates. The sample is 64,692 S1 (1.94M pairs): 30k random S1 plus every S1 in Jaipur, Columbus, Tucson and Memphis. The dense city part keeps competing S1 together, so exclusivity can be measured.

## 5. Results (stand-in token blocking, K=30)

| | macro-F0.5 |
|---|---|
| Final model + exclusivity, threshold 0.64 | **0.8877** |
| Without exclusivity, threshold 0.64 | 0.8872 |
| Blocking ceiling (perfect matcher on these candidates) | 0.9276 |

- Pair AUC is 0.9992, and 91.2% of singletons are correctly left empty.
- TF-IDF fitted once instead of per batch: 0.8877 → 0.8868 (−0.0009, within run-to-run noise). CV cannot show the gain, because the whole CV sample is one batch; the fix removes a train/test skew in `predict.py`'s 20k-S1 chunks. Measured on the 30k sample, the old per-batch fit moved a pair's name TF-IDF cosine by a mean of 0.023 (p99 0.127, max 0.21) in a 500-S1 batch, and by 0.005 (p99 0.029) in a 20k-S1 batch. Small batches happen at test time: each country's last chunk, and France.
- Exclusivity measured on the dense-city sample alone: 0.8808 → 0.8825.
- Cross-entity features: dense-city sample (35,170 S1, the criterion for keeping the feature) 0.8804 → 0.8829 without exclusivity and 0.8823 → **0.8835** with it. Full sample with exclusivity: 0.8868 → 0.8877. Most of the gain overlaps with exclusivity, which already removes the weaker claim on a shared record.

**Where the remaining score goes** (30k random sample, before exclusivity):

| Cause | Points lost |
|---|---|
| True match never reaches candidates (blocking) | 0.063 |
| Model misses true pairs | 0.027 |
| Model accepts wrong pairs | 0.014 |
| Singletons wrongly given a match | 0.005 |

**Common false positives:**
- about 39% are records that belong to a different S1: a different business at the same address, or the same brand at another branch
- names whose address is missing (`nan`), where the model can only judge the name
- native-script names where only the address can decide

**Common false negatives:** mostly outside the model's reach: native-script names with a changed address, or a missing address together with a typo in the name.

## Cost of a smaller K (for P2)

The organizers rank a smaller candidate set per S1 higher, so we measured what cutting K costs the matcher. Each S1 keeps its top-K candidates by `block_score` from the stand-in blocking, and the matcher is retrained and cross-validated for each K. Sample: 30k random S1 (`train_pairs`, the same pairs as `train_pairs_oof.parquet`), 5-fold GroupKFold, exclusivity applied.

| K | Mean candidates per S1 | Blocking ceiling | Model macro-F0.5 |
|---|---|---|---|
| 3 | 3.00 | 0.8063 | 0.7806 |
| 5 | 5.00 | 0.8633 | 0.8307 |
| 10 | 10.00 | 0.8994 | 0.8607 |
| 15 | 14.99 | 0.9161 | 0.8745 |
| 20 | 19.99 | 0.9267 | 0.8836 |
| 30 | 29.97 | 0.9370 | **0.8921** |

**Recommended K: 30.** No smaller K comes within 0.002 of the best: K=20 already costs 0.0085, and K=10 costs 0.031. The model tracks the ceiling at a nearly constant gap (0.03–0.045), so almost all of the loss comes from true matches that the cut throws away.

Notes:
- The curve is still rising at K=30, because the stand-in `block_score` is a plain IDF token overlap that ranks many true matches at positions 21–30. Ground truth has a mean of 3.5 matches per S1 (max 11), so a sharper ranking should reach the same ceiling with a much smaller K. P2 should improve the candidate ranking first, then cut K.
- Re-run this sweep on P2's `block_score` before choosing K: `python -m src.matching.train --pairs <P2 pairs>.parquet --topk K` (experiment only; it saves nothing).
- Not tested: an adaptive per-S1 cut (drop candidates far below that S1's best `block_score`) could lower the mean candidate count without a fixed K.

## Reproduce

```
python -m src.blocking.baseline_tokens --n 30000 --out data/interim/train_pairs.parquet
python -m src.blocking.baseline_tokens --cities jaipur,columbus,tucson,memphis --out data/interim/city_pairs.parquet
python -m src.matching.train --pairs data/interim/train_pairs.parquet data/interim/city_pairs.parquet --model data/interim/matcher.pkl
python -m src.matching.predict --model data/interim/matcher.pkl --out output
python resources/utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir data/raw/test
```

On 16 GB RAM with 32 threads: blocking plus training takes about 25 min, and test inference about 2.5 h. When P2's blocking replaces `baseline_tokens`, retrain the matcher on P2's candidates. Its features and threshold depend on the candidate distribution.
