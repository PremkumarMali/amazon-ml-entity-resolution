"""Tests for the scalable candidate generator (synthetic data only).

Requires numpy, scipy, pandas and pyarrow (see requirements.txt).
Run from the repository root:  python -m unittest discover -s tests -t .
"""

import math
import os
import shutil
import tempfile
import unittest
from collections import Counter

import numpy as np

from src.blocking.data_io import Record
from src.blocking.encode import encode_records, token_hash
from src.blocking.engine import CandidateConfig, CountryIndex, _top_n, select, select_top_k
from src.blocking.features import record_features, trigrams
from src.blocking.normalize import has_nonlatin
from src.blocking.transliterate import phonetic_key, to_latin

def read_text(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


HEADER = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"


class TransliterationTests(unittest.TestCase):
    def test_devanagari_matches_english_phonetically(self):
        self.assertEqual(to_latin("पायोनियर बेकर्स"), "paayoniyar bekars")
        for native, english in [("पायोनियर", "pioneer"), ("कंसल्टेंट्स", "consultants"),
                                ("हॉस्पिटैलिटी", "hospitality"), ("मैनेजमेंट", "management")]:
            self.assertEqual(phonetic_key(to_latin(native)), phonetic_key(english), native)

    def test_other_scripts_share_the_table(self):
        self.assertEqual(phonetic_key(to_latin("ಗ್ಲೋಬಲ್")), phonetic_key("global"))       # Kannada
        self.assertEqual(phonetic_key(to_latin("శివం")), phonetic_key("shivam"))          # Telugu
        self.assertEqual(to_latin("പ്രൈവറ്റ് ലിമിറ്റഡ്"), "praivatt limittad")            # Malayalam tt
        self.assertEqual(phonetic_key(to_latin("भारत")), phonetic_key("bharat"))

    def test_non_brahmic_text_passes_through(self):
        self.assertEqual(to_latin("Acme Tools #12"), "Acme Tools #12")

    def test_phonetic_key_rules(self):
        self.assertEqual(phonetic_key("m0lecular"), phonetic_key("molecular"))  # OCR digit
        self.assertEqual(phonetic_key("technologies")[:4], "tknl")
        self.assertEqual(phonetic_key("12345"), "")
        self.assertEqual(phonetic_key("aditya"), "adt")


class FeatureTests(unittest.TestCase):
    def test_fields(self):
        f = record_features("Maple Trust LLC", "House No. 0052, Sector 9")
        self.assertEqual(f["n"], {"maple", "trust"})
        self.assertEqual(f["pb"], {"mpl trst"})
        self.assertIn("52", f["a"])  # leading zeros removed
        self.assertTrue(all(any(c.isdigit() for c in p) for p in f["ab"]))
        self.assertIn("52 no", f["ab"])

    def test_record_flags(self):
        from src.blocking.features import FLAG_EMPTY_ADDRESS, FLAG_NONLATIN_NAME, record_flags
        self.assertEqual(record_flags("Acme", "12 Oak St"), 0)
        self.assertEqual(record_flags("राम", "<NULL>, N/A"), FLAG_NONLATIN_NAME | FLAG_EMPTY_ADDRESS)

    def test_indic_name_gets_latin_phonetic_keys(self):
        f = record_features("ग्रीन कंसल्टेंट्स प्राइवेट लिमिटेड", "")
        self.assertEqual(f["p"], {"jrn", "knsltnts"})  # legal words dropped after transliteration
        self.assertEqual(f["a"], set())

    def test_trigrams(self):
        self.assertEqual(trigrams("ab cd"), {"abc", "bcd"})
        self.assertEqual(trigrams(""), set())

    def test_encoding_is_stable_and_sorted(self):
        recs = [Record("S1-1", "Acme Tools", "12 Oak St", "US")]
        _, _, fa, a = encode_records(recs)
        _, _, _, b = encode_records(recs)
        self.assertEqual(fa, [0])
        for f in a:
            np.testing.assert_array_equal(a[f][1], b[f][1])
            self.assertTrue(np.all(np.diff(a[f][1]) >= 0))
        self.assertEqual(token_hash("n", "acme"), token_hash("n", "acme"))
        self.assertNotEqual(token_hash("n", "acme"), token_hash("a", "acme"))


# ---------------------------------------------------------------- synthetic split
def synthetic_split():
    s1 = [
        ("S1-1", "Zephyrine Bakery LLC", "12 Quillon Street, Dayton, OH", "US"),
        ("S1-2", "Acme Tools Inc", "5 Main Street, Dayton, OH", "US"),
        ("S1-3", "Pioneer Bakers Private Limited", "Plot 7, Gaya Road, Bihar", "India"),
        ("S1-4", "Lonely Shop", "", "US"),                                  # no match anywhere
        ("S1-5", "Harbor Kite Dental Studio", "4410 Elmwood Lane, Springfield, IL", "US"),
        ("S1-6", "Orphan Record", "Nowhere", "France"),                     # country with no candidates
    ]
    s2 = [
        ("S2-1", "ZEPHYRINE BAKERY", "12 QUILLON ST, DAYTON, OH", "US"),
        ("S2-2", "Acme Tols", "5 Main St, Dayton, OH", "US"),
        ("S2-3", "पायोनियर बेकर्स प्राइवेट लिमिटेड", "GAYA ROAD, Bihar", "India"),
        ("S2-4", "Zephyrine Bakery", "12 Quillon Street", "India"),  # other country
    ]
    s3 = [
        ("S3-1", "zephyrine.com", "12 Quillon Street, Dayton, Ohio", "US"),
        ("S3-2", "harborkitedental.com", "", "US"),
    ]
    filler = [(f"S3-f{i}", f"Filler {i} Shop", f"{i} Main Street, Dayton, OH", "US") for i in range(30)]
    truth = {"S1-1": ["S2-1", "S3-1"], "S1-2": ["S2-2"], "S1-3": ["S2-3"], "S1-4": [],
             "S1-5": ["S3-2"], "S1-6": []}
    return s1, s2, s3 + filler, truth


def write_split(d, split="train"):
    s1, s2, s3, truth = synthetic_split()
    os.makedirs(d, exist_ok=True)
    for i, rows in ((1, s1), (2, s2), (3, s3)):
        with open(os.path.join(d, f"{split}_source{i}.tsv"), "w", encoding="utf-8", newline="\n") as f:
            f.write(HEADER + "".join("\t".join(r) + "\n" for r in rows))
    if split == "train":
        with open(os.path.join(d, "train_ground_truth.tsv"), "w", encoding="utf-8", newline="\n") as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")
            f.write("".join(f"{k}\t{','.join(v)}\n" for k, v in truth.items()))
    return truth


class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.data = os.path.join(cls.tmp, "raw")
        cls.truth = write_split(cls.data)
        from src.blocking.encode import encode_split
        cls.enc = encode_split(cls.data, "train", os.path.join(cls.tmp, "cache"), workers=1, chunks_per_file=2)
        cls.cfg = CandidateConfig(top_k=5, max_df=6, shard_size=2)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def run_country(self, country, cfg=None):
        cfg = cfg or self.cfg
        idx = CountryIndex.build(self.enc, country, cfg)
        rows = self.enc[1].rows_of_country(country)
        toks = {f: self.enc[1].row_tokens(f, rows) for f in cfg.fields}
        sc = idx.score(toks, cfg)
        chosen = select(idx, sc, cfg)
        ids = [self.enc[1].ids[r].decode() for r in rows]
        return idx, sc, {i: list(idx.cand_ids[c[0]].astype(str)) for i, c in zip(ids, chosen)}, chosen

    def test_true_matches_found_and_country_respected(self):
        _, _, us, _ = self.run_country("US")
        self.assertIn("S2-1", us["S1-1"])
        self.assertIn("S2-2", us["S1-2"])
        self.assertNotIn("S2-4", us["S1-1"])  # India record never a US candidate
        _, _, india, _ = self.run_country("India")
        self.assertEqual(india["S1-3"], ["S2-3"])  # Devanagari name via transliteration

    def test_top_k_bound_and_empty_lists(self):
        _, _, us, _ = self.run_country("US")
        self.assertTrue(all(len(v) <= 5 for v in us.values()))
        self.assertEqual(us["S1-4"], [])
        idx, _, fr, _ = self.run_country("France")
        self.assertEqual(idx.n_candidates, 0)
        self.assertEqual(fr["S1-6"], [])

    def test_scores_equal_brute_force_all_token_idf(self):
        # Default weights / non-Latin boost, re-ranking deep enough to cover every candidate.
        for country in ("US", "India"):
            cfg = CandidateConfig(top_k=50, max_df=6, rerank_depth=50)
            idx, sc, _, chosen = self.run_country(country, cfg)
            # Brute force: weighted summed IDF (df over S1+S2+S3 of the country) of shared
            # tokens over all fields; phonetic fields boosted for non-Latin candidate names.
            s1, s2, s3, _ = synthetic_split()
            recs = [r for r in s1 + s2 + s3 if r[3] == country]
            feats = {r[0]: record_features(r[1], r[2], with_fallback=False) for r in recs}
            nonlatin = {r[0]: has_nonlatin(r[1]) for r in recs}
            df = Counter((f, t) for r in recs for f, ts in feats[r[0]].items() for t in ts)
            n = len(recs)
            w = cfg.weight_map
            cand_ids = list(idx.cand_ids.astype(str))
            s1_rows = [r[0] for r in s1 if r[3] == country]
            checked = 0
            for i, s1_id in enumerate(s1_rows):
                for c, score in zip(chosen[i][0], chosen[i][1]):
                    cid = cand_ids[c]
                    expect = sum(
                        w[f] * (cfg.nonlatin_boost if f in ("p", "pb") and nonlatin[cid] else 1.0)
                        * math.log((n + 1) / df[f, t])
                        for f in ("n", "p", "pb", "a", "ab") for t in feats[s1_id][f] & feats[cid][f])
                    self.assertAlmostEqual(float(score), expect, places=3, msg=(s1_id, cid))
                    checked += 1
            self.assertGreater(checked, 0)

    def test_deterministic_and_order_independent(self):
        a = self.run_country("US")[2]
        b = self.run_country("US")[2]
        self.assertEqual(a, b)

    def test_trigram_fallback_recovers_domain_name(self):
        base = CandidateConfig(top_k=5, max_df=6, weights=(("n", 1.0), ("p", 0.0), ("pb", 0.0),
                                                            ("a", 1.0), ("ab", 0.0)))
        _, _, plain, _ = self.run_country("US", base)
        self.assertNotIn("S3-2", plain["S1-5"])
        fb = CandidateConfig.from_dict({**base.to_dict(), "fallback_below": 3, "fallback_max_df": 50})
        _, _, with_fb, _ = self.run_country("US", fb)
        self.assertIn("S3-2", with_fb["S1-5"])

    def test_index_save_load_roundtrip(self):
        idx = CountryIndex.build(self.enc, "US", self.cfg)
        d = os.path.join(self.tmp, "idx_us")
        idx.save(d)
        loaded = CountryIndex.load(d, mmap=True)
        rows = self.enc[1].rows_of_country("US")
        toks = {f: self.enc[1].row_tokens(f, rows) for f in self.cfg.fields}
        a, b = idx.score(toks, self.cfg), loaded.score(toks, self.cfg)
        self.assertEqual((a.combined != b.combined).nnz, 0)
        del loaded


class TopKTests(unittest.TestCase):
    def test_top_n_ties_broken_by_column(self):
        vals = np.array([1.0, 3.0, 3.0, 2.0, 3.0], dtype=np.float32)
        cols = np.array([10, 40, 20, 30, 50])
        self.assertEqual(list(cols[_top_n(vals, cols, 2)]), [20, 40])
        self.assertEqual(len(_top_n(vals, cols, 0)), 0)

    def test_sparse_quota_reserves_flagged_candidates(self):
        import scipy.sparse as sp
        combined = sp.csr_matrix(np.array([[9.0, 8.0, 7.0, 1.0]], dtype=np.float32))
        sparse = np.array([False, False, False, True])
        (c, _, _), = select_top_k(combined, combined, 2, 0.0)
        self.assertEqual(c.tolist(), [0, 1])
        (c, _, _), = select_top_k(combined, combined, 2, 0.0, sparse, 0.5)
        self.assertEqual(c.tolist(), [0, 3])

    def test_quota_union_dedup(self):
        import scipy.sparse as sp
        combined = sp.csr_matrix(np.array([[9.0, 8.0, 7.0, 1.0]], dtype=np.float32))
        name = sp.csr_matrix(np.array([[0.0, 0.0, 0.0, 5.0]], dtype=np.float32))
        (c, s, n), = select_top_k(combined, name, 2, 0.5)
        self.assertEqual(sorted(c.tolist()), [0, 3])  # one name slot + best combined, no duplicates
        (c, _, _), = select_top_k(combined, combined, 3, 0.5)
        self.assertEqual(c.tolist(), [0, 1, 2])


class GenerateCandidatesTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.data = os.path.join(self.tmp, "raw")
        self.truth = write_split(self.data)
        self.cache = os.path.join(self.tmp, "cache")
        self.out = os.path.join(self.tmp, "output", "candidates")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_gen(self, shard_size=2, **kw):
        from src.blocking.generate_candidates import run
        cfg = CandidateConfig(top_k=5, max_df=6, shard_size=shard_size)
        return run("train", self.data, self.cache, self.out, cfg, workers=1, log=lambda *a: None, **kw)

    def read_tsv(self, path):
        from src.blocking.handoff import read_candidate_pairs
        return list(read_candidate_pairs(path))

    def test_candidate_pairs_format_and_resume(self):
        summary = self.run_gen()
        path = summary["candidate_pairs"]["path"]
        rows = self.read_tsv(path)
        s1_ids = [r[0] for r in synthetic_split()[0]]
        self.assertEqual([r[0] for r in rows], s1_ids)  # every S1, file order
        for _, cands in rows:
            self.assertEqual(len(cands), len(set(cands)))
            self.assertTrue(all(c.startswith(("S2-", "S3-")) for c in cands))
            self.assertLessEqual(len(cands), 5)
        self.assertEqual(dict(rows)["S1-4"], [])
        with open(path, encoding="utf-8") as f:
            self.assertEqual(f.readline(), "source1_entity_id\tcandidate_entity_ids\n")
            self.assertIn("S1-4\t\n", f.read())
        first = read_text(path)
        again = self.run_gen()  # resume: nothing recomputed, identical output
        self.assertEqual(again["shards_done"], 0)
        self.assertGreater(again["shards_skipped"], 0)
        self.assertEqual(read_text(path), first)

    def test_shard_size_does_not_change_output(self):
        a = self.read_tsv(self.run_gen(shard_size=2)["candidate_pairs"]["path"])
        b = self.read_tsv(self.run_gen(shard_size=3, overwrite=True)["candidate_pairs"]["path"])
        self.assertEqual(a, b)

    def test_config_change_requires_overwrite(self):
        self.run_gen(shard_size=2)
        with self.assertRaises(SystemExit):
            self.run_gen(shard_size=3)

    def test_handoff_frames_with_records_and_labels(self):
        self.run_gen()
        from src.blocking.handoff import CandidateStore
        store = CandidateStore("train", out_dir=self.out, data_dir=self.data)
        import pandas as pd
        df = pd.concat(list(store.iter_frames(with_records=True, with_labels=True)))
        for col in ("s1_entity_id", "candidate_entity_id", "rank", "score", "s1_name", "candidate_name",
                    "candidate_address", "label"):
            self.assertIn(col, df.columns)
        row = df[(df.s1_entity_id == "S1-3") & (df.candidate_entity_id == "S2-3")].iloc[0]
        self.assertEqual(row.label, 1)
        self.assertEqual(row.candidate_name, "पायोनियर बेकर्स प्राइवेट लिमिटेड")
        self.assertEqual(row.s1_address, "Plot 7, Gaya Road, Bihar")
        self.assertTrue((df.groupby("s1_entity_id")["rank"].min() == 1).all())
        self.assertEqual(set(df.label.unique()) <= {0, 1}, True)

    def test_missing_run_is_explained(self):
        from src.blocking.handoff import CandidateStore
        with self.assertRaises(FileNotFoundError):
            CandidateStore("test", out_dir=self.out, data_dir=self.data)


if __name__ == "__main__":
    unittest.main()
