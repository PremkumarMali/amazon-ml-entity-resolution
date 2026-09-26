# Blocking → Matching handoff (P2 → P3)

P2 produces, for every Source-1 record, a ranked list of at most **K = 100** Source-2/Source-3
records that could be the same business. The matcher only needs to classify these pairs;
**final matches must be a subset of these candidates**. Measured results and method details are in
[blocking_data_analysis.md](blocking_data_analysis.md) (§10).

## 1. Run it

Python 3.13, `pip install -r requirements.txt` (pandas, numpy, scipy, pyarrow; all BSD/Apache).
Raw data must be under `data/raw/<split>/` (see `data/README.md`).

```bash
python -m src.blocking.generate_candidates --split test  --top-k 100   # -> output/candidate_pairs.tsv
python -m src.blocking.generate_candidates --split train --top-k 100   # -> output/candidates/train/candidate_pairs.tsv
```

Useful options:

| Option | Default | Meaning |
|---|---|---|
| `--workers` | 6 | parallel processes (about 1.8 GB each at shard size 1,000) |
| `--s1-limit N` | all | only the first N S1 rows (smoke tests) |
| `--candidate-pairs PATH` | see above | where to write the TSV |
| `--no-tsv` | off | only write the per-shard parquet files |
| `--overwrite` | off | replace a run that used a different config |

**Re-runs are safe.** Every shard is written atomically. If a run stops, starting it again only
computes the missing shards and then rewrites the TSV. Output is deterministic for a given config.

**Runtime.** Measured on this machine (20 cores):
- **Encoding:** about 2.5 min per split, cached under `data/processed/blocking/`.
- **Country indexes:** under 1 min each.
- **Scoring:** 200,000 test S1 rows took 12 min with 6 workers.
- **Full test split:** 1.73M S1 records took 87 min end to end with 8 workers (1.8 GB each).
  The output passed the organiser validator (see §10.6 of the analysis document).
- **Full training split:** estimated at about 1.8 h with 8 workers.

All generated data (caches, shards, TSVs) is git-ignored.

## 2. Outputs (contract)

### `candidate_pairs.tsv` (official format)

```
source1_entity_id<TAB>candidate_entity_ids
S1-...<TAB>S2-...,S3-...,S2-...
```

- One row for **every** S1 entity, in the S1 file order.
- Candidate lists are comma-separated, best first.
- Lists contain S2/S3 IDs only, with no duplicates.
- A list is empty when there is no candidate. This happened for 0 of 5,509 S1 records on the validation sample
  and for 2 of 1,732,544 on the full test split.
- For the test split this is the file that goes into the submission package (`output/candidate_pairs.tsv`).

### Per-shard parquet (rich format, for training / inference)

These files live in `output/candidates/<split>/shards/<country>/<shard>.parquet`. Each shard
holds 1,000 S1 rows and one row per (S1, candidate) pair:

| Column | Type | Meaning |
|---|---|---|
| `s1_entity_id` | str | Source-1 ID |
| `candidate_entity_id` | str | S2-/S3- ID |
| `rank` | int16 | 1 = best blocking score |
| `score` | float32 | blocking score (weighted IDF of shared name/address tokens; see analysis §10.1) |
| `name_score` | float32 | the name part of `score` |
| `candidate_source` | int8 | 2 or 3 |
| `candidate_row` / `s1_row` | int32 | 0-based data row in the source TSV (fast joins) |

`score` and `rank` are useful features, but they are **not** match probabilities.

## 3. Python API (recommended)

```python
from src.blocking.handoff import CandidateStore, read_candidate_pairs

store = CandidateStore("train")        # after generate_candidates --split train
for df in store.iter_frames(with_records=True, with_labels=True):
    # one pandas DataFrame per shard (~100k rows); columns:
    # s1_entity_id, candidate_entity_id, rank, score, name_score, candidate_source, candidate_row, s1_row,
    # s1_name, s1_address, s1_country, candidate_name, candidate_address, candidate_country,
    # label (train only: 1 if the pair is in train_ground_truth.tsv)
    ...

for s1_id, candidate_ids in read_candidate_pairs("output/candidate_pairs.tsv"):
    ...
```

- **Streaming:** frames are streamed shard by shard.
- **Source tables:** with `with_records=True` they are loaded once (about 30 s and 1–2 GB for a split).
- **Raw frames:** `store.iter_shards()` gives the frames without the joins.

## 4. How to use the training candidates

- **Training pairs.** Train the matcher on the candidate pairs (`label` from ground truth), not
  on random pairs. The negatives are then the hard, realistic ones the matcher will see at test
  time.
- **Missing positives.** About 3% of true training pairs are not among the candidates (the blocking
  recall ceiling). They cannot be predicted at inference, so leave them out of pair-level training
  metrics. Do count them when estimating end-to-end F0.5: they are unrecoverable false negatives.
- **Singletons.** S1 entities with no true match still get up to 100 candidates. The matcher must
  be able to reject all of them, because an empty prediction scores 1.0 for a singleton.
- **Validation split.** The blocking hyper-parameters (weights, quotas) were tuned on the hash
  sample `md5("blocking-validation-v1:" + s1_id) < 0.0025` (see `src/blocking/sampling.py`). To
  get an unbiased final estimate, use a different salt for P3's hold-out.
- **Subset rule.** Every predicted match must come from the candidate list. The organiser
  validator warns otherwise:
  `python resources/utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir data/raw/test`.

## 5. Default configuration

`src/blocking/engine.py::CandidateConfig` (all measured in the analysis document, §10):

| Setting | Value |
|---|---|
| `top_k` | 100 |
| blocking | S2/S3 records of the same country sharing ≥1 token with document frequency ≤ 20,000 in any field |
| fields (weight) | name tokens (1.0), phonetic keys after offline transliteration (0.5), phonetic key pairs (0.5), address tokens (1.5), address pairs with a number (1.0) |
| re-ranking | exact all-token score for each S1's 500 best candidates |
| reserved slots | 30% best by name evidence, 15% best among non-Latin-name / empty-address candidates |
| non-Latin boost | phonetic evidence ×3 for candidates whose name is in an Indic script |
| character-trigram fallback | implemented, **off** (no measured gain) |

## 6. Known limitations

- **Recall ceiling (K = 100) on the validation sample:** 97.05% of true pairs overall; US 98.6%,
  India 94.7%, non-Latin candidate names 91.7%. Missed pairs are mostly Indian records with partial
  addresses and aliases/rebrands that share no name word.
- **France appears only in test.** The method is country-agnostic (per-country frequencies, no
  country-specific rules), but the weights were tuned on US/India only.
- **Transliteration** is a small rule-based Indic→Latin mapper (no external library or data). It
  is good enough to match phonetic consonant skeletons, not to produce readable text.
- **Blocking uses no labels**, so there is no leakage from ground truth. Only the hyper-parameters
  saw the validation sample.
- **Speed:** about 18–24 ms per S1 record per worker, dominated by per-row Python in re-ranking and
  Top-K. The full test split took 87 min with 8 workers; the training split is estimated at about 1.8 h.
