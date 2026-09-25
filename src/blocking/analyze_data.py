"""Reusable dataset analysis behind docs/blocking_data_analysis.md.

Usage (from the repository root):

    python -m src.blocking.analyze_data                      # train split
    python -m src.blocking.analyze_data --pair-sample-rate 0.003

Streams every source file (read-only, chunked, in parallel) and reports:
  * per-source row counts, null/placeholder rates, string lengths, country mix,
    scripts, punctuation / casing / abbreviation patterns;
  * ground-truth summary (singletons, matches per S1, distractor share) when a
    ground-truth file exists;
  * how true (S1, S2/S3) pairs differ, on a deterministic hash sample of S1 entities.

Results are printed and written as JSON to ``--out``.
"""

import argparse
import json
import os
import re
import unicodedata
from collections import Counter
from multiprocessing import Pool

from .data_io import chunk_ranges, ground_truth_path, iter_ground_truth, iter_records, source_path
from .normalize import address_tokens, fold, has_nonlatin, house_number, is_nonlatin_letter, name_tokens
from .sampling import DEFAULT_SALT, in_sample

DOMAIN = re.compile(r"\.(com|in|net|org|co|fr)\b", re.I)
ALIAS = re.compile(r"\b(dba|d\.b\.a\.?|fka|formerly known as|aka)\b", re.I)
PUNCT = {"&": "amp", "-": "hyphen", "(": "paren", "[": "bracket", ".": "dot", ",": "comma", "'": "apos", "/": "slash"}
ADDR_ABBR = {"st": "street", "rd": "road", "ave": "avenue", "blvd": "boulevard", "dr": "drive", "ln": "lane"}
NAME_ABBR = {"pvt": "private", "ltd": "limited", "inc": "incorporated", "corp": "corporation", "co": "company"}


def script_of(text):
    for c in text:
        if is_nonlatin_letter(c):
            try:
                return unicodedata.name(c).split()[0]
            except ValueError:
                return "OTHER"
    return None


# ---------------------------------------------------------------- per-source profile
def _profile_chunk(task):
    path, start, end = task
    c = Counter()
    for r in iter_records(path, start, end):
        name, addr = r.name, r.address
        c["rows"] += 1
        c["country=" + r.country] += 1
        c["len_name"] += len(name)
        c["len_addr"] += len(addr)
        c["max_len_name"] = max(c["max_len_name"], len(name))
        c["max_len_addr"] = max(c["max_len_addr"], len(addr))
        c["empty_name"] += not name.strip()
        c["empty_addr"] += not addr.strip()
        c["empty_country"] += not r.country.strip()
        c["addr_<NULL>"] += "<NULL>" in addr
        c["addr_N/A"] += "N/A" in addr
        c["addr_no_house_number"] += not house_number(addr)
        c["addr_all_upper"] += addr.isupper()
        c["addr_hash"] += "#" in addr
        c["name_all_upper"] += name.isupper()
        c["name_all_lower"] += name.islower()
        c["name_domain_like"] += bool(DOMAIN.search(name))
        c["name_alias_marker"] += bool(ALIAS.search(name))
        c["name_latin_accent"] += any(0xC0 <= ord(ch) <= 0x24F for ch in name)
        c["name_double_space"] += "  " in name
        c["name_digit"] += any(ch.isdigit() for ch in name)
        c["name_no_informative_token"] += not name_tokens(name)
        for ch, lab in PUNCT.items():
            c["name_" + lab] += ch in name
        s = script_of(name)
        if s:
            c["name_script=" + s] += 1
        s = script_of(addr)
        if s:
            c["addr_script=" + s] += 1
        for t in fold(addr):
            if t in ADDR_ABBR:
                c[f"addr_abbr_{t}"] += 1
            elif t in ADDR_ABBR.values():
                c[f"addr_full_{t}"] += 1
        for t in fold(name):
            if t in NAME_ABBR:
                c[f"name_abbr_{t}"] += 1
            elif t in NAME_ABBR.values():
                c[f"name_full_{t}"] += 1
    return path, c


def profile_sources(data_dir, split, workers, chunks_per_file):
    tasks = []
    for s in (1, 2, 3):
        p = source_path(data_dir, split, s)
        tasks += [(p, a, b) for a, b in chunk_ranges(p, chunks_per_file)]
    totals = {}
    with Pool(workers) as pool:
        for path, c in pool.imap_unordered(_profile_chunk, tasks):
            t = totals.setdefault(os.path.basename(path), Counter())
            for k, v in c.items():
                t[k] = max(t[k], v) if k.startswith("max_") else t[k] + v
    out = {}
    for fname, c in sorted(totals.items()):
        n = c["rows"]
        out[fname] = {
            "rows": n,
            "countries": {k.split("=", 1)[1]: v for k, v in c.items() if k.startswith("country=")},
            "avg_len_name": round(c["len_name"] / n, 1), "max_len_name": c["max_len_name"],
            "avg_len_addr": round(c["len_addr"] / n, 1), "max_len_addr": c["max_len_addr"],
            "rates_pct": {k: round(100 * v / n, 2) for k, v in sorted(c.items())
                          if not k.startswith(("rows", "country=", "len_", "max_", "addr_abbr", "addr_full",
                                               "name_abbr", "name_full"))},
            "abbreviations": {k: v for k, v in sorted(c.items()) if "_abbr_" in k or "_full_" in k},
        }
    return out


# ---------------------------------------------------------------- ground truth
def ground_truth_summary(path, source_rows):
    per = Counter()
    links = Counter()
    seen = set()
    multi = 0
    for _, ids in iter_ground_truth(path):
        per[len(ids)] += 1
        for i in ids:
            links[i[:2]] += 1
            multi += i in seen
            seen.add(i)
    n = sum(per.values())
    matched = n - per[0]
    distinct = Counter(i[:2] for i in seen)
    return {
        "s1_rows": n, "singletons": per[0], "matched": matched,
        "singleton_pct": round(100 * per[0] / n, 2),
        "matches_per_s1": dict(sorted(per.items())),
        "links": dict(links), "mean_links_per_matched": round(sum(links.values()) / max(1, matched), 2),
        "ids_linked_to_multiple_s1": multi,
        "unlinked_pct": {s: round(100 * (1 - distinct[s] / source_rows[s]), 1) for s in ("S2", "S3")},
    }


# ---------------------------------------------------------------- pair patterns
def pair_patterns(data_dir, rate, salt):
    truth = {s1: ids for s1, ids in iter_ground_truth(ground_truth_path(data_dir)) if ids and in_sample(s1, rate, salt)}
    need = set(truth) | {i for ids in truth.values() for i in ids}
    recs = {}
    for s in (1, 2, 3):
        for r in iter_records(source_path(data_dir, "train", s)):
            if r.entity_id in need:
                recs[r.entity_id] = r
    c = Counter()
    added, removed = Counter(), Counter()
    for s1, ids in truth.items():
        a = recs[s1]
        a_fold, a_name = fold(a.name), name_tokens(a.name)
        a_addr, a_hn = set(address_tokens(a.address)), house_number(a.address)
        for i in ids:
            b = recs[i]
            b_name, b_addr = name_tokens(b.name), set(address_tokens(b.address))
            b_hn = house_number(b.address)
            flags = {
                "same_country": a.country == b.country,
                "name_equal_raw": a.name == b.name,
                "name_equal_folded": a_fold == fold(b.name),
                "name_equal_without_stopwords": a_name == b_name,
                "share_name_token": bool(set(a_name) & set(b_name)),
                "first_name_token_equal": bool(a_name and b_name and a_name[0] == b_name[0]),
                "cand_name_nonlatin": has_nonlatin(b.name),
                "cand_name_domain": bool(DOMAIN.search(b.name)),
                "cand_name_alias": bool(ALIAS.search(b.name)),
                "cand_addr_empty": not b_addr,
                "cand_addr_nonlatin": has_nonlatin(b.address),
                "house_number_equal": bool(a_hn and a_hn == b_hn),
                "share_2plus_addr_tokens": len(a_addr & b_addr) >= 2,
                "no_name_overlap_but_addr_overlap": not (set(a_name) & set(b_name)) and len(a_addr & b_addr) >= 2,
            }
            for group in ("all", a.country, i[:2]):
                c[group, "pairs"] += 1
                for k, v in flags.items():
                    c[group, k] += v
            added.update(set(fold(b.name)) - set(a_fold))
            removed.update(set(a_fold) - set(fold(b.name)))
    groups = sorted({g for g, _ in c})
    return {
        "s1_sampled": len(truth), "pairs": {g: c[g, "pairs"] for g in groups},
        "rates_pct": {g: {k: round(100 * c[g, k] / c[g, "pairs"], 1) for gg, k in c if gg == g and k != "pairs"}
                      for g in groups},
        "top_tokens_added": added.most_common(25), "top_tokens_removed": removed.most_common(25),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data/raw/train")
    ap.add_argument("--split", default="train")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--chunks-per-file", type=int, default=4)
    ap.add_argument("--pair-sample-rate", type=float, default=0.0025)
    ap.add_argument("--salt", default=DEFAULT_SALT)
    ap.add_argument("--out", default="output/blocking_analysis.json")
    args = ap.parse_args(argv)

    result = {"sources": profile_sources(args.data_dir, args.split, args.workers, args.chunks_per_file)}
    gt = ground_truth_path(args.data_dir, args.split)
    if os.path.exists(gt):
        rows = {f"S{s}": result["sources"][f"{args.split}_source{s}.tsv"]["rows"] for s in (2, 3)}
        result["ground_truth"] = ground_truth_summary(gt, rows)
        result["pair_patterns"] = pair_patterns(args.data_dir, args.pair_sample_rate, args.salt)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
