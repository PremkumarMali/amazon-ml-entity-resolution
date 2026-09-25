# Data

Official Amazon ML Challenge 2026 data lives **locally** under `data/raw/`.
Raw challenge data is intentionally excluded from Git (see `.gitignore`); the repository is public.

## Setup

Each team member must download the official student resource package themselves, extract it,
and copy the dataset files into place without renaming or modifying them:

```
data/raw/train/train_source1.tsv
data/raw/train/train_source2.tsv
data/raw/train/train_source3.tsv
data/raw/train/train_ground_truth.tsv

data/raw/test/test_source1.tsv
data/raw/test/test_source2.tsv
data/raw/test/test_source3.tsv
```

All files are tab-separated. Read them with an explicit separator, e.g. `pd.read_csv(path, sep="\t")`.

## Rules

- Do not commit any file under `data/raw/` or any derived dataset.
- Do not add external datasets to this project.
- The challenge strictly prohibits external business-identity lookup and external data augmentation
  (entity resolution APIs, business registries, geocoding APIs, internet data). Use only the provided data.
