"""Smoke test for src/blocking/analyze_data.py on a tiny synthetic split."""

import json
import os
import tempfile
import unittest

from src.blocking import analyze_data

HEADER = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"


def write(path, rows, header=HEADER):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(header)
        for r in rows:
            f.write("\t".join(r) + "\n")


class AnalyzeDataTests(unittest.TestCase):
    def test_end_to_end_on_synthetic_split(self):
        with tempfile.TemporaryDirectory() as d:
            write(os.path.join(d, "train_source1.tsv"), [
                ("S1-1", "Acme Tools Inc", "12 Oak Street, Dayton, OH", "US"),
                ("S1-2", "Lonely Shop", "3 Pine Road, Pune", "India"),
            ])
            write(os.path.join(d, "train_source2.tsv"), [
                ("S2-1", "ACME TOOLS", "12 OAK ST, DAYTON, OH", "US"),
                ("S2-2", "Other", "", "US"),
            ])
            write(os.path.join(d, "train_source3.tsv"), [
                ("S3-1", "acmetools.com", "#12 Oak Street, Dayton, Ohio", "US"),
            ])
            write(os.path.join(d, "train_ground_truth.tsv"), [("S1-1", "S2-1,S3-1"), ("S1-2", "")],
                  header="source1_entity_id\tmatched_entity_ids\n")
            out = os.path.join(d, "out.json")
            analyze_data.main(["--data-dir", d, "--workers", "1", "--chunks-per-file", "2",
                               "--pair-sample-rate", "1.0", "--out", out])
            with open(out, encoding="utf-8") as f:
                res = json.load(f)
        self.assertEqual(res["sources"]["train_source2.tsv"]["rows"], 2)
        self.assertEqual(res["sources"]["train_source2.tsv"]["rates_pct"]["empty_addr"], 50.0)
        gt = res["ground_truth"]
        self.assertEqual((gt["singletons"], gt["matched"]), (1, 1))
        self.assertEqual(gt["unlinked_pct"]["S2"], 50.0)
        pp = res["pair_patterns"]
        self.assertEqual(pp["pairs"]["all"], 2)
        self.assertEqual(pp["rates_pct"]["all"]["same_country"], 100.0)
        self.assertEqual(pp["rates_pct"]["all"]["cand_name_domain"], 50.0)


if __name__ == "__main__":
    unittest.main()
