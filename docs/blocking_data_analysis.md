# Blocking Data Analysis (Training Set)

Owner: Person 2 — Candidate Generation / Blocking
Scope: exploratory analysis (§1–8), the first candidate-generation baseline (§9) and the final scalable
candidate generator (§10). No ML matcher.

Reproduce:
- §1–4: `python -m src.blocking.analyze_data`
- §9: `python -m src.blocking.run_experiments`
- §10: `python -m src.blocking.evaluate`
- Candidates for a split: `python -m src.blocking.generate_candidates --split test`. See
  [blocking_handoff.md](blocking_handoff.md).

## Method

- All four training TSVs were streamed line by line (stdlib only, read-only); no file was modified.
- Column statistics and block-size counters were computed over the **full** training data.
- Pair-level patterns and blocking recall were measured on a seeded random sample of
  **5,000 matched S1 entities → 18,367 true (S1, S2/S3) pairs**, plus 1,000 singletons.
  Recall figures carry roughly ±0.3 pp sampling error.
- Normalisation used for tokens: lowercase, strip Latin accents (Indic vowel signs kept),
  punctuation → space, and a small stop-list of legal/generic words
  (`inc llc ltd limited pvt private corp co company lp llp pc the of and services center partners dba fka mr shri smt …`
  plus Indic-script `प्राइवेट / लिमिटेड`).
- Examples below are synthetic illustrations of observed patterns, not real records.

## 1. Schema and basic statistics

All files are tab-separated, UTF-8, with the documented headers. No malformed rows, no duplicate IDs.

| File | Rows | US | India | Avg / max name len | Avg / max address len |
|---|---|---|---|---|---|
| train_source1 | 2,206,821 | 1,323,633 | 883,188 | 24.0 / 105 | 52.1 / 256 |
| train_source2 | 5,034,616 | 3,016,817 | 2,017,799 | 25.1 / 104 | 46.2 / 249 |
| train_source3 | 5,285,603 | 3,170,056 | 2,115,547 | 25.2 / 123 | 46.7 / 240 |
| train_ground_truth | 2,206,821 | – | – | – | – |

Null / placeholder rates:

| | S1 | S2 | S3 |
|---|---|---|---|
| empty `business_name` | 0% | 0% | 0% |
| empty `business_address` | 0% | 3.36% | 3.33% |
| address contains `<NULL>` | 0% | 0.87% | 0.82% |
| address contains `N/A` | 0% | 0.87% | 0.83% |
| no house number in address | 12.0% | 16.9% | 17.3% |
| empty `country` | 0% | 0% | 0% |

- **No postal-code field, and addresses essentially contain no ZIP/PIN.** India 6-digit hits are 0.1–0.5%. US 5-digit hits
  (~6%) are consistent with 5-digit house numbers in spot checks. Postal-code blocking is therefore **not available**.
- US addresses are typically `number street, city, STATE`; India addresses are longer and multi-part.
- **S1 is clean**: Latin script only, no accents, no placeholders, mostly full words (`Road`, `Street`, `Limited`).
  S2/S3 carry the noise.

## 2. Name and address patterns (full data)

| Pattern | S1 | S2 | S3 |
|---|---|---|---|
| name in non-Latin script | 0% | 9.3% (Devanagari 5.4%, Telugu, Kannada, Tamil, Gujarati, Bengali, Malayalam, Oriya) | 5.2% |
| address contains non-Latin script (mostly state names) | 0% | ~9% | ~9% |
| name ALL CAPS | – | 18.9% | 3.0% |
| address ALL CAPS | 0% | 63.4% | 0.02% |
| Latin accents injected (`é`, `í`) | 0% | 5.8% | 6.2% |
| double spaces in name | – | 11.0% | 10.9% |
| hyphen / bracket / paren in name | 0.6 / 0 / 2.1% | 5.3 / 2.9 / 5.0% | 5.6 / 3.1 / 5.1% |
| digit in name (e.g. `0` for `o`, phone numbers) | 1.6% | 5.1% | 5.1% |
| domain-style name (`xyzcorp.com`) | – | 4.0% | 4.0% |
| alias marker (`dba`, `fka`, `formerly known as`) | 0% | 0% | 1.3% |
| `#` in address | 0.8% | 7.6% | 10.3% |

Abbreviations:
- S1 uses mostly full forms, e.g. `road` 467k vs `rd` 20k, and `limited` 522k vs `ltd` 149k.
- S2 and S3 mix forms roughly 50/50, e.g. `street` 338k vs `st` 335k.
- `incorporated` appears 8 times in S1 but about 25k times in each of S2 and S3.

## 3. Transformations between S1 and its true S2/S3 matches (sample, 18,367 pairs)

| Property of a true pair | All | US | India |
|---|---|---|---|
| same `country` | **100%** | 100% | 100% |
| name identical (raw) | 5.0% | 6.5% | 2.7% |
| name identical after case/accent/punctuation folding | 26.1% | 31.0% | 18.7% |
| name identical after also dropping legal/generic words | 53.2% | 55.6% | 49.6% |
| share ≥1 informative name token | 85.7% | 92.2% | 75.8% |
| first informative name token equal | 79.5% | 86.0% | 69.6% |
| S2/S3 name in non-Latin script | 7.1% | 0% | **17.9%** |
| S2/S3 address empty | 4.2% | 4.7% | 3.5% |
| house number equal | 65.7% | 76.7% | 49.1% |
| share ≥2 address tokens | **95.2%** | 94.5% | 96.3% |
| **no** name-token overlap but ≥2 address tokens shared | 14.2% | 7.7% | 24.1% |
| no name **and** no address token overlap | ~0% | ~0% | ~0% |

Observed transformation types (synthetic examples):
- **Case/spacing/punctuation:** `Acme Tools Inc` → `ACME  TOOLS INC`, `Acme-Tools [Inc]`, `Acme Tools Inc.`
- **Legal suffix add/drop/swap:** `Private Limited` ↔ `Pvt Ltd` ↔ `Private` ↔ `(Limited)`; `LLC` added or removed.
  The most frequently *removed* tokens are `limited, private, llc, inc, ltd, pvt, llp`.
- **Generic words added:** `Center`, `Services`, `Partners`, `LP`, honorifics `Mr / Shri / Smt / Sri`.
- **Word order:** `Acme Private Tools Limited`, `Private Acme Tools Limited`.
- **Typos / character injection / OCR-like swaps:** `Tols`, `Toxols`, `T0ols`, `lnc`; token duplication (`Acme Acme Tools`).
- **Truncation:** `Bureau of Finance` → `Bureau of`.
- **Domain form:** `acmetools.com` (tokens concatenated, so token blocking fails).
- **Aliases / rebrands:** `NewName dba Acme Tools`, `NewName fka …`, and sometimes a **completely different name** at the same address.
- **Transliteration:** Latin name → Devanagari or other Indic script, fully or partially (`Acme Tools प्राइवेट लिमिटेड`).
- **Address:**
  - abbreviations (`Blvd`/`Boulevard`, `Rd`/`Road`); state as code, full name or native script (`TN` / `Tamil Nadu` / Tamil script)
  - component reordering (`STATE, number street, city`)
  - missing components (street number, street, whole address)
  - placeholders (`<NULL>`, `N/A`); house-number variants (`007285`, `#529`, `8908.`, `1017-`, `12584-12586`)
  - added prefixes (`Block C-75`, `Door No 582`); `CITY`/`CDP` suffixes; county or neighbouring town instead of city

## 4. Singletons vs matched entities

| | Count | Share |
|---|---|---|
| S1 entities | 2,206,821 | 100% |
| singletons (no match) | 123,247 | 5.6% |
| matched | 2,083,574 | 94.4% |
| links to S2 / S3 | 3,693,619 / 3,944,746 | 7,638,365 total |

- Matches per S1: 1 → 119k, 2 → 375k, 3 → 531k, 4 → 484k, 5 → 322k, 6 → 165k, 7 → 64k, 8 → 19k, 9+ → 4.8k
  (max 11; mean 3.67 per matched entity).
- **Every linked S2/S3 record belongs to exactly one S1**, so clusters are disjoint.
- 26.6% of S2 and 25.4% of S3 records are linked to no S1. These are distractors, which matters for precision.
- Typical matched S1 has 1–3 records in **each** of S2 and S3 (multiple near-duplicates per source).

## 5. Why brute force is impossible

| | Pairs |
|---|---|
| Train, all S1 × (S2 + S3) | 2.2M × 10.3M = **2.28 × 10¹³** |
| Train, restricted to same country | **1.18 × 10¹³** |
| Test, all S1 × (S2 + S3) | 1.73M × 9.97M ≈ 1.73 × 10¹³ |
| Test, same country (US / India / France) | ≈ 6.7 × 10¹² |

- At an optimistic 10⁶ string comparisons per second per core, 1.2 × 10¹³ pairs is about 140 core-days.
- Storing only the ID pairs (16 B each) would need about 190 TB.
- Only about 7.6M of those pairs are true matches (1 in ~1.5 million), so even country partitioning leaves the
  space about 5 orders of magnitude too large.

## 6. Blocking keys measured

Measurement setup:
- Recall is measured on the 18,367 sampled true pairs.
- Pairs are counted on the full training set as S1 × (S2+S3) comparisons.
- For multi-key schemes the pair count is an **upper bound** (a pair sharing several keys is counted more than once).
- "Cap" means blocks whose total size across S1+S2+S3 exceeds the cap are skipped.

| Scheme | Recall | Pairs | Candidates / S1 |
|---|---|---|---|
| country only | 100% | 1.18e13 | 5.4M |
| country + first informative name token | 79.5% | 1.28e10 | 5.8k |
| country + house number | 65.7% | 2.70e10 | 12k |
| country + house number + first name token | 52.8% | 3.3e7 | 15 |
| first name token **OR** house number | 92.4% | 4.0e10 | 18k |
| all name tokens, cap 10k | 62.6% | 3.9e9 | 1.8k |
| all address tokens, cap 10k | 90.8% | 1.25e10 | 5.7k |
| top-2 rarest address tokens, no cap | 89.4% | 3.0e9 | 1.4k |
| top-2 rarest name tokens, no cap | 84.8% | 3.3e10 | 15k |
| **top-2 rarest name OR top-2 rarest address, cap 5k** | **95.4%** | **4.8e9** | **~2.2k** |
| top-2 rarest name OR top-2 rarest address, cap 20k | 97.7% | 1.9e10 | 8.5k |
| top-2 rarest name OR top-2 rarest address, no cap | 98.2% | 3.6e10 | 16.5k |
| top-3 rarest name OR top-3 rarest address, cap 20k | 98.3% | 2.8e10 | 12.5k |

Findings:
- **Country is a safe hard partition** (100% agreement in true pairs).
- **Address tokens are the strongest single signal.** They survive name aliasing, transliteration and rebrands.
  Name tokens alone plateau around 85%.
- **Name-token blocks degrade quickly under caps.** A typo inside the distinctive token makes it unique, so the only
  shared token left is a common one (e.g. a surname), and that block is huge.
- Naive token blocks explode: Indian `no` appears in 2.19M records, `road` in 982k, `nagar` in 711k; US `street` in 840k.
  Frequency-based (rarest-k) selection or a block-size cap is mandatory.
- **Remaining misses** (top-2, cap 5k; 840 pairs, about 4.6%):
  - 64% are India
  - 31% have a non-Latin S2/S3 name
  - 27% have an empty or placeholder S2/S3 address

## 7. Biggest matching difficulties for blocking

1. **Transliteration:** 18% of India true pairs have an Indic-script name on the S2/S3 side, with zero token overlap with S1.
2. **Aliases, rebrands and domain names:** 14% of true pairs share no informative name token at all.
3. **Typos inside rare tokens,** which defeat exact rare-token keys. Character-level keys are needed.
4. **Missing or partial addresses:** 4% empty, and house numbers are equal in only 66% of pairs.
5. **Very common tokens** make naive blocks explode (see §6).
6. **One-to-many, disjoint clusters with ~26% distractor records.** Blocking must keep several candidates per source
   per S1 without flooding the matcher.
7. **Unseen country (France) in test.** Do not hard-code English/Indian stop-lists. Derive frequent-token stop-lists
   from token frequencies per country (test data included), and keep `country` as an open label.

## 8. Strategies to test next (simplest → most advanced)

1. **Standard key blocking (reference baseline).** Within country, the union of *first informative name token* and
   *house number*, plus the tight key *house number + first name token*. Easy to implement; it sets a reference
   (≈92% recall, but ~4e10 pairs, far too many to use as-is).
2. **Frequency-aware token blocking — recommended first baseline.** Per country, compute token document frequency
   over S1+S2+S3. Key each record on its **2 rarest name tokens and 2 rarest address tokens**, and drop blocks larger
   than a cap (sweep 2k–20k). Measured: **~95–98% recall at ~2–9k candidates per S1**. Streaming-friendly: an
   inverted index keyed by `(country, token)`.
3. **Meta-blocking / top-K pruning.** Score each candidate from (2) by IDF-weighted shared tokens (name and address)
   and keep the top-K per S1 per source (e.g. K = 20–50). This targets ≤10⁸ pairs, sized for the matcher, while
   keeping most of the recall. Needs measurement.
4. **Character n-gram TF-IDF + sparse top-K nearest neighbours** on normalised name and address (char 3–4-grams, per
   country). Handles typos, character injection, concatenated domain names and truncation that token keys miss.
   Combine with (2) as a union.
5. **Script-robust / learned blocking.**
   - Offline Indic → Latin transliteration (rule/character mapping) before tokenising.
   - Alternatively, a small multilingual character/sentence encoder (MIT/Apache, ≤8B) with an ANN index (FAISS),
     per country.
   - Also exploit that near-duplicate S2 (or S3) records of the same entity often share an identical address, so
     they can be grouped before blocking.
   - No external lookups or geocoding: challenge rule.

## Initial recommendation (superseded by §9)

The first plan was symmetric rarest-token blocking (strategy 2) followed by Top-K pruning (strategy 3).
§9 implements an S1-driven variant of that plan and replaces this recommendation.

## 9. Candidate-generation baseline (experiments)

Code: `src/blocking/`

| File | Role |
|---|---|
| `normalize.py` | normalisation and tokenisation |
| `token_stats.py` | country-specific document frequencies (DF) |
| `blocker.py` | keys, index, scoring, Top-K |
| `run_experiments.py` | evaluation |

Tests: `tests/test_blocking.py`, `tests/test_analyze_data.py`

### 9.1 Method

1. **Token statistics.** Name and address token DF are computed per `country` over S1+S2+S3 of the split. No labels
   are used, so the identical procedure runs on test (France included). Tokens seen once are not stored (DF=1).
   Build time is 56 s with 6 processes, and the cache goes in `data/processed/blocking/` (git-ignored).
2. **Keys (S1-driven).** Each S1 record takes its 2 rarest name tokens and 2 rarest address tokens. A token is
   dropped if its DF exceeds `max_block_size`. If nothing survives, a single fallback key (the rarest token) is
   kept when its DF ≤ 50,000.
3. **Blocking.** Candidates are all S2/S3 records of the **same country** that contain a key token in the same
   field. S2/S3 are streamed in byte-range chunks against an in-memory S1 key index. A true match is not lost
   when the *candidate* has a typo in a different token, because only the S1 side chooses the keys.
4. **Scoring.** Each candidate gets the summed IDF of all name and address tokens it shares with the S1 record
   (`idf_sum`). Name and address evidence are also kept separately.
5. **Top-K pruning.** Keep K candidates per S1:
   - `name_quota` × K slots go to the best candidates by name evidence alone
   - `address_quota` × K slots go to the best by address evidence alone
   - the remaining slots go to the best combined score
6. **Evaluation sample.** A deterministic hash sample of S1 entities (`md5(salt:id)`, rate 0.25%, salt
   `blocking-validation-v1`): 5,509 S1 entities (301 singletons) with **19,108 true pairs**, blocked against the
   **full** S2/S3 files. The sampling error on recall is roughly ±0.3 pp. Runs are bit-for-bit reproducible
   (scores use `math.fsum`, so they don't depend on hash-randomised set order).

### 9.2 Key count and block-size cap (combined score only, no quotas)

| Name keys | Address keys | Max block | Recall before pruning | Pool / S1 (mean / p95) | Recall @K=50 | @K=100 | @K=200 |
|---|---|---|---|---|---|---|---|
| 2 | 2 | 5,000 | 93.14% | 2,165 / 5,667 | 91.15% | 91.86% | 92.32% |
| 2 | 3 | 5,000 | 93.79% | 2,831 / 6,857 | – | 91.94% | 92.61% |
| 3 | 3 | 5,000 | 93.79% | 2,924 / 7,021 | – | 91.94% | – |
| 2 | 2 | 20,000 | 97.18% | 8,132 / 21,781 | 93.66% | 94.55% | 95.24% |
| 2 | 3 | 20,000 | 97.86% | 11,899 / 27,542 | – | 94.37% | 95.20% |
| 3 | 3 | 20,000 | 97.86% | 12,547 / 28,800 | – | 94.37% | – |

Pool = distinct candidates per S1 before pruning.

- A third name key changes nothing. A third address key raises pool recall but not Top-K recall.
- Adding the fallback key raises cap-5k recall by about 0.9 pp and has no effect at cap 20k.
- At cap 20k the **ranking**, not blocking, becomes the bottleneck: 97.2% of true pairs are in the pool, but
  only 94.6% survive K=100.

### 9.3 Scoring / pruning variants (2+2 keys, cap 20,000)

| Variant | @K=25 | @K=50 | @K=100 | @K=200 | India @K=100 |
|---|---|---|---|---|---|
| `idf_sum`, no quotas | 92.52% | 93.66% | 94.55% | 95.24% | 90.27% |
| `coverage` (per-field share, empty field imputed) | 84.79% | 86.21% | 88.35% | 89.24% | 79.48% |
| `idf_sum` + name_quota 0.1 | – | – | 95.39% | – | 92.14% |
| `idf_sum` + name_quota 0.2 | 93.88% | 95.00% | 95.50% | 96.00% | 92.32% |
| **`idf_sum` + name_quota 0.3** | **94.16%** | **95.09%** | **95.60%** | **96.05%** | **92.50%** |
| `idf_sum` + name_quota 0.3 + address_quota 0.1 | – | – | 95.64% | – | 92.50% |

- **Why the name quota helps.** Pruned true matches were mostly records with an **empty or very short address**.
  Their only evidence is the name, so neighbouring businesses at the same building outscored them on shared
  address tokens. This is common for long Indian addresses.
- **Coverage scoring was worse.** Imputing the missing address let many empty-address records with only a common
  word in the name jump ahead. It is kept as an option only to document the negative result.
- **Address quota adds ≤0.05 pp.**

### 9.4 Recommended configuration — full metrics

Configuration: 2 name keys + 2 address keys, `max_block_size=20000`, fallback 50,000, `idf_sum`, `name_quota=0.3`.
These are the defaults in `BlockingConfig` and `run_experiments`.

| K | Recall | US | India | S2 | S3 | S1 with *all* matches kept | Cands / S1 mean | median | p95 | max |
|---|---|---|---|---|---|---|---|---|---|---|
| 25 | 94.16% | 96.58% | 90.54% | 94.42% | 93.92% | 83.18% | 25.0 | 25 | 25 | 25 |
| 50 | 95.09% | 97.29% | 91.78% | 95.17% | 95.01% | 85.98% | 49.9 | 50 | 50 | 50 |
| **100** | **95.60%** | **97.68%** | **92.50%** | **95.64%** | **95.57%** | **87.60%** | **99.6** | **100** | **100** | **100** |
| 200 | 96.05% | 98.06% | 93.05% | 95.96% | 96.14% | 88.86% | 197.7 | 200 | 200 | 200 |

Before pruning:
- recall 97.18%
- pool per S1: mean 8,132, median 6,688, p95 21,781, max 39,997
- 0 S1 records without keys

Candidates per S1 is below K only when the pool is smaller than K.

Runtime and memory for the sample (6 worker processes, stdlib Python):
- about 135 s wall per configuration, including streaming and tokenising all 10.3M S2/S3 records
- 612 MB peak per worker
- 3.5 GB parent peak, holding the retained heaps for every K, plus loss analysis

### 9.5 What is still lost (recommended config, K=100: 840 of 19,108 true pairs, 4.4%)

| Stage | Pairs | India / US | Dominant traits (a pair can have several) |
|---|---|---|---|
| **Not blocked** (no shared key token) | 539 (2.8%) | 393 / 146 | no shared informative name token 380; S2/S3 name in Indic script 271; empty address 93 |
| **Pruned by Top-K** | 301 (1.6%) | 181 / 120 | empty candidate address 162; no shared address token 165; no shared name token 96 |

- 230 of the 301 pruned pairs rank below the top 200 of the combined ranking, so raising K is an inefficient fix:
  K=200 recovers only 0.45 pp.
- **Transliteration is the largest single remaining blocking gap**, mostly in India. So are aliases/rebrands with
  no shared name word and a partial address.
- Pruning losses are mainly name-only records whose name words are common.

### 9.6 Recommendation (superseded by §10)

- **Default for the matcher: K=100** at 95.6% pair recall (India 92.5%), exactly ≤100 candidates per S1.
  Projected size: ≈2.2 × 10⁸ pairs for the training S1 set and ≈1.7 × 10⁸ for test.
- **Cheaper option: K=50** at 95.1% (India 91.8%), with half the matcher workload.
- The cap-5,000 variant is 3.6× cheaper to block, but tops out at 93.3% @K=100.

### 9.7 Scaling to the full dataset (done in §10)

Scoring cost grows with pool size:
- mean pool 8.1k × 2.2M S1 ≈ 1.8 × 10¹⁰ scored pairs (train), ≈1.4 × 10¹⁰ (test)
- an estimated ~8 × 10⁴ pairs per second per worker (from the sample run) implies about 10 h on 6 processes in pure Python

Retaining Top-K heaps for all S1 at once would also need tens of GB. Before full inference:
- **Shard S1** (by country and hash, about 20–40 shards) so each shard's heaps fit in memory; S2/S3 are re-streamed per shard.
- **Reduce retained state** to exactly K per view.
- **Vectorise scoring** (sparse token-incidence matrices with numpy/scipy), or cap the pool per S1 by scoring only
  the rarest key blocks first.

### 9.8 Next steps for blocking (items 1, 2 and 4 are done in §10)

1. **Offline transliteration** of Indic-script names/addresses to Latin before tokenising. This targets the
   largest remaining gap (271 of 539 unblocked pairs involve Indic-script names).
2. **Character n-gram keys/scoring** (TF-IDF char 3–4-grams) for typos, concatenated domain names and truncation.
3. **Better pruning features:** house-number agreement, and "same address, different name" demotion.
4. Produce `candidate_pairs.tsv` on the full training data once the scaling work (§9.7) is done, and hand the
   candidate set to the matcher (Person 3).

## 10. Final candidate generator (supersedes §9.6–9.8)

Code is in `src/blocking/`:

| File | Role |
|---|---|
| `transliterate.py` | offline Indic→Latin transliteration + phonetic keys |
| `features.py` | per-record token fields and flags |
| `encode.py` | cached hashed-token encoding of a split |
| `engine.py` | per-country sparse index, scoring, re-ranking, Top-K |
| `generate_candidates.py` | sharded, resumable CLI; writes `candidate_pairs.tsv` |
| `handoff.py` | P3 consumer API |
| `evaluate.py` | evaluation on the fixed sample |

Consumer documentation: [blocking_handoff.md](blocking_handoff.md).

All numbers below come from the **same fixed validation sample** as §9: 5,509 S1 entities, 19,108 true pairs, of
which 1,425 have a candidate name in an Indic script ("non-Latin"). Blocking is against the full training S2/S3.
Reproduce with `python -m src.blocking.evaluate` (add `--variant final --top-k 25,50,100,200` for the full table).

### 10.1 Method

1. **Fields per record** (`features.py`):

   | Field | Content |
   |---|---|
   | `n` | name tokens |
   | `p` | phonetic consonant skeletons of the name after offline transliteration ("पायोनियर" → "payoniyar" → `pnr` = "Pioneer") |
   | `pb` | adjacent pairs of phonetic keys (rare even when each word is common, e.g. "maple trust") |
   | `a` | address tokens, with leading zeros of numbers removed ("0052" = "52") |
   | `ab` | adjacent address pairs that contain a number ("3 55", "52 no") |
   | `g` | character trigrams (fallback only) |

   **Transliteration** is a single offset table shared by the nine Brahmic scripts (their Unicode blocks are
   parallel). It is deterministic, offline, and uses no library or external data.
2. **Encoding** (`encode.py`): each split is tokenised once into 64-bit token hashes. The CSR arrays are cached
   under `data/processed/blocking/`.
3. **Index per country** (`engine.py`): document frequencies are computed over S1+S2+S3 of that country. Tokens
   with DF ≤ 20,000 form a binary token × candidate matrix.
4. **Blocking and first score.** The candidate pool of an S1 record is every same-country S2/S3 record sharing
   at least one such token in any field. The weighted summed IDF of the shared tokens comes out of one scipy
   sparse product per field for a whole shard of S1 rows. There are no Python pair loops.
5. **Exact re-rank.** Very common tokens (city, state, "road") are not used for blocking, but they separate a
   true match from a same-name business elsewhere. For each S1 record's 500 best candidates (by combined, by
   name, and among flagged candidates), the shared common tokens' IDF is added through sorted-key membership
   tests. The resulting score is exactly the summed IDF over all tokens; a unit test checks it against a brute
   force.
6. **Top-K = 100** is filled in this order:
   - 30% of slots: best by name evidence (`n`, `p`, `pb`)
   - 15%: best among candidates flagged *non-Latin name* or *empty address* (structurally low scores)
   - the rest: best combined score

   Candidates are deduplicated. Ties are broken by candidate index, so output is deterministic.
7. **Weights:**
   - `n` 1.0, `p` 0.5, `pb` 0.5, `a` 1.5, `ab` 1.0
   - phonetic evidence ×3 for candidates whose name is non-Latin, since such a name can only match
     phonetically

### 10.2 Development path (K=100, fixed sample)

| Step | Recall | US | India | Non-Latin | Never blocked | Lost in Top-K |
|---|---|---|---|---|---|---|
| §9 baseline: 2+2 rarest keys, cap 20k, name quota 0.3 | 95.60% | 97.68% | 92.50% | 77.5%* | 539 | 301 |
| vectorised: all name+address tokens with DF ≤ 20k as keys, rare-token score only | 92.50% | 95.31% | 88.30% | 73.05% | 280 | 1,153 |
| + fields `p`, `pb`, `ab` + exact re-rank (all weights 1) | 96.00% | 96.57% | 95.15% | 89.19% | 90 | 674 |
| weights `p`=`pb`=0.5, `a`=1.5 | 96.47% | 98.04% | 94.12% | 87.02% | 90 | 584 |
| + 15% sparse slots | 96.79% | 98.59% | 94.10% | 87.16% | 90 | 523 |
| **+ non-Latin phonetic boost ×3 (final)** | **97.05%** | **98.59%** | **94.74%** | **91.65%** | **90** | **474** |

\* Derived from the §9 loss analysis on the same sample: 271 unblocked + 49 pruned non-Latin pairs of 1,425.

Reproduce rows 2–5 with `python -m src.blocking.evaluate`:

| Row | Flags |
|---|---|
| 2 | `--weights n=1,p=0,pb=0,a=1,ab=0 --rerank-depth 0 --sparse-quotas 0` |
| 3 | `--weights n=1,p=1,pb=1,a=1,ab=1 --sparse-quotas 0 --nonlatin-boosts 1` |
| 4 | `--weights n=1,p=0.5,pb=0.5,a=1.5,ab=1 --sparse-quotas 0 --nonlatin-boosts 1` |
| 5 | as row 4 with `--sparse-quotas 0.15` |

The final row is the default.

Other measured settings:
- **Weights:** `p`=`pb`=0.3 gave 96.40%; `p`=`pb`=0.2 with `ab`=0.5 gave 96.31%.
- **Non-Latin boost:** ×2 gave 97.02%.
- **Sparse slots:** 10% gave 96.73%.
- **Name slots:** 20% instead of 30% gave 96.46% (without sparse slots).
- **Re-rank depth of 500 is justified.** Without re-ranking, the rare-token ranking keeps 96.85% / 97.45% /
  97.91% of true pairs in its top 300 / 500 / 1,000.

### 10.3 Ablation of the final configuration (K=100, one component removed)

| Configuration | Recall | US | India | Non-Latin | Lost in Top-K |
|---|---|---|---|---|---|
| final | 97.05% | 98.59% | 94.74% | 91.65% | 474 |
| without `p`, `pb`, `ab` (no transliteration/phonetic/pair fields) | 94.24% | 97.04% | 90.06% | 75.93% | 820 (+280 never blocked) |
| without exact re-rank | 96.37% | 98.09% | 93.81% | 88.77% | 603 |
| without name slots | 96.49% | 98.64% | 93.29% | 91.58% | 580 |
| without sparse slots | 96.69% | 98.04% | 94.67% | 91.16% | 542 |
| without non-Latin boost | 96.79% | 98.59% | 94.10% | 87.16% | 523 |
| with character-trigram fallback (pool < 1,000) | 97.05% | 98.59% | 94.74% | 91.65% | 474 |

- Every component except the trigram fallback helps.
- **The trigram fallback is implemented and tested but off by default.** It changed nothing on the sample: the
  mean pool grew by 1.6 candidates, and peak memory rose by about 3 GB. After the phonetic and pair fields,
  only 0.47% of true pairs are unblocked, which leaves it little to find.

### 10.4 Final configuration — full metrics

| K | Recall | US | India | S2 | S3 | Non-Latin | S1 with all matches kept | Cands / S1 (mean / median / p95 / max) |
|---|---|---|---|---|---|---|---|---|
| 25 | 95.22% | 97.25% | 92.19% | 96.64% | 93.91% | 88.28% | 86.14% | 25 / 25 / 25 / 25 |
| 50 | 96.39% | 98.12% | 93.81% | 97.38% | 95.48% | 90.81% | 89.46% | 50 / 50 / 50 / 50 |
| **100** | **97.05%** | **98.59%** | **94.74%** | **97.71%** | **96.43%** | **91.65%** | **91.42%** | **100 / 100 / 100 / 100** |
| 200 | 97.55% | 98.79% | 95.69% | 98.08% | 97.05% | 93.19% | 92.76% | 200 / 200 / 200 / 200 |

- Pool before Top-K: mean 20,696, median 19,270, p95 42,541, max 92,077.
- No S1 record has zero candidates.

**Against the §9 baseline at K=100:**
- Recall 95.60% → **97.05%**
- US 97.68% → 98.59%; India 92.50% → 94.74%; non-Latin ≈77.5% → 91.65%
- S1 with all matches kept 87.60% → 91.42%
- Never-blocked true pairs 539 → 90
- **Candidate budget unchanged:** exactly ≤100 per S1

K=50 (96.39%) already beats the old K=100.

### 10.5 What is still lost (final, K=100: 564 of 19,108 true pairs, 2.95%)

| Stage | Pairs | India / US | Traits (a pair can have several) |
|---|---|---|---|
| never blocked | 90 (0.47%) | 56 / 34 | no shared name token 58; non-Latin name 32; empty address 29 |
| lost in Top-K | 474 (2.48%) | 347 / 127 | no shared name token 215; no shared address token 117; empty address 115; non-Latin name 87 |

The remaining misses are dominated by Indian records that combine a **different name**
(alias/rebrand/transliteration mismatch) with a **short or partial address**. Their scores are genuinely
indistinguishable from neighbouring businesses at the same building.

### 10.6 Runtime, memory and full-scale runs

| Step | Measurement |
|---|---|
| encode train / test (12 processes) | 133 s / 142 s; cache 3.6 GB / 3.4 GB |
| country index build (test) | France 5 s, US 30 s, India 48 s; 1.6 GB on disk, memory-mapped and shared by workers |
| sample evaluation (5,509 S1, one process) | 75–180 s per configuration; peak 8–12 GB (both country indexes in one process) |
| smoke test: first 200,000 test S1, 6 workers, shards of 1,000 | 724 s; 19,998,579 pairs; 1.78 GB peak per worker; 0 empty lists; all IDs valid test S2/S3, S1 file order, no duplicates |
| **full test split**: 1,732,544 S1, 8 workers, 1,734 shards | **85 min** scoring (5,113 s), 87 min end to end including the TSV; 173,241,710 pairs; 1.80 GB peak per worker; `output/candidate_pairs.tsv` is 2.26 GB |

**Checks on the full test output:**
- **Streaming check of every row:** each S1 appears exactly once, in file order; no duplicates; every one of the
  173.2M IDs exists in test S2/S3; at most 100 per S1.
- **Empty lists:** 2 S1 rows (1 India, 1 France) share no token under the block-size limit with any candidate.
- **Official validator:** `resources/utils/validate_submission.py --check-ids` **passes** on the first
  250,000 rows. Its full-file candidate cross-check holds every ID in memory (more than 15 GB), which did not fit
  alongside the other applications on this machine.

**Estimated training run:** about 1.3× the test split, so roughly 1.8 h with 8 workers.

**Scalability properties:**
- **Sharding:** S1 is processed in fixed shards. Only one shard's candidates are in memory per worker.
- **Shared index:** S2/S3 live in a memory-mapped per-country index shared through the OS page cache.
- **Resumable output:** shards are written atomically. Re-runs skip finished shards, and changing the config
  requires `--overwrite`.
- **Deterministic:** output is identical for any shard size (tested).
- **Cost:** about 18 ms per S1 record per worker, mostly per-row Python in re-ranking and Top-K selection.
