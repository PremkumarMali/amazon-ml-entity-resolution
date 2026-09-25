"""Streaming readers for the challenge TSV files (never loads a whole file)."""

import os
from dataclasses import dataclass

SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
GROUND_TRUTH_COLUMNS = ["source1_entity_id", "matched_entity_ids"]


@dataclass(frozen=True)
class Record:
    entity_id: str
    name: str
    address: str
    country: str


def source_path(data_dir, split, source):
    return os.path.join(data_dir, f"{split}_source{source}.tsv")


def ground_truth_path(data_dir, split="train"):
    return os.path.join(data_dir, f"{split}_ground_truth.tsv")


def _check_header(path, expected):
    with open(path, encoding="utf-8") as f:
        header = f.readline().rstrip("\r\n").split("\t")
    if header != expected:
        raise ValueError(f"{path}: unexpected header {header}, expected {expected}")


def chunk_ranges(path, n_chunks):
    """Split ``path`` (excluding its header) into ``n_chunks`` byte ranges."""
    with open(path, "rb") as f:
        f.readline()
        start = f.tell()
    end = os.path.getsize(path)
    step = max(1, (end - start) // max(1, n_chunks))
    bounds = [start + i * step for i in range(n_chunks)] + [end]
    return [(bounds[i], bounds[i + 1]) for i in range(n_chunks) if bounds[i] < bounds[i + 1]]


def iter_records(path, start=None, end=None):
    """Yield ``Record`` objects for lines that *start* inside ``[start, end)``.

    With no range the whole file (minus header) is read. Byte ranges from
    ``chunk_ranges`` cover every line exactly once.
    """
    _check_header(path, SOURCE_COLUMNS)
    with open(path, "rb") as f:
        header_end = len(f.readline())
        if start is None:
            start, end = header_end, os.path.getsize(path)
        if start > header_end:
            f.seek(start - 1)
            f.readline()  # finish the line that straddles ``start``
        else:
            f.seek(header_end)
        while f.tell() < end:
            raw = f.readline()
            if not raw:
                break
            parts = raw.decode("utf-8").rstrip("\r\n").split("\t")
            if len(parts) != 4:
                raise ValueError(f"{path}: malformed row near byte {f.tell()}")
            yield Record(*parts)


def iter_ground_truth(path):
    """Yield ``(source1_entity_id, [matched ids])``."""
    _check_header(path, GROUND_TRUTH_COLUMNS)
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            s1, ids = line.rstrip("\r\n").split("\t")
            yield s1, (ids.split(",") if ids else [])
