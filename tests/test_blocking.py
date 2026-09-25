"""Unit tests for src/blocking (synthetic data only; no challenge records).

Run from the repository root:  python -m unittest discover -s tests -t .
"""

import os
import tempfile
import unittest

from src.blocking.blocker import (
    BlockingConfig,
    S1Profile,
    TopKCollector,
    build_key_index,
    generate_candidates,
    score,
    score_views,
)
from src.blocking.data_io import Record, chunk_ranges, iter_records
from src.blocking.normalize import address_tokens, fold, house_number, name_tokens
from src.blocking.sampling import in_sample
from src.blocking.token_stats import ADDRESS, NAME, TokenStats, record_token_sets


def rec(eid, name, addr, country="US"):
    return Record(eid, name, addr, country)


class NormalizeTests(unittest.TestCase):
    def test_fold_case_accents_punctuation(self):
        self.assertEqual(fold("Acmé-Tools [INC.]"), ["acme", "tools", "inc"])

    def test_fold_keeps_indic_words_whole(self):
        self.assertEqual(fold("राम मार्केटिंग"), ["राम", "मार्केटिंग"])

    def test_name_tokens_drop_legal_and_noise_words(self):
        self.assertEqual(name_tokens("The Acme Tools Pvt. Ltd Services"), ["acme", "tools"])
        self.assertEqual(name_tokens("Acme प्राइवेट लिमिटेड"), ["acme"])

    def test_address_tokens_drop_placeholders(self):
        self.assertEqual(address_tokens("12 Oak St, <NULL>, N/A, Springfield"), ["12", "oak", "st", "springfield"])

    def test_house_number_variants(self):
        self.assertEqual(house_number("#529 Church St"), "529")
        self.assertEqual(house_number("007285 BRANTFORD RD"), "7285")
        self.assertEqual(house_number("8908. BRENNAN RD"), "8908")
        self.assertEqual(house_number("Main Street, Springfield"), "")


class SamplingTests(unittest.TestCase):
    def test_deterministic_and_roughly_rate(self):
        ids = [f"S1-{i}" for i in range(20000)]
        a = [i for i in ids if in_sample(i, 0.1)]
        b = [i for i in ids if in_sample(i, 0.1)]
        self.assertEqual(a, b)
        self.assertTrue(1700 < len(a) < 2300)
        self.assertNotEqual(a, [i for i in ids if in_sample(i, 0.1, salt="other")])


class DataIOTests(unittest.TestCase):
    def test_chunks_cover_every_line_once(self):
        lines = [f"S2-{i}\tNäme {i} राम\t{i} Road, City\tIndia" for i in range(500)]
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "x.tsv")
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
                f.write("\n".join(lines) + "\n")
            for n in (1, 3, 7, 64):
                got = [r.entity_id for a, b in chunk_ranges(path, n) for r in iter_records(path, a, b)]
                self.assertEqual(got, [f"S2-{i}" for i in range(500)], n)

    def test_rejects_wrong_header(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "x.tsv")
            with open(path, "w", encoding="utf-8") as f:
                f.write("a,b,c,d\n")
            with self.assertRaises(ValueError):
                list(iter_records(path))


def toy_corpus():
    s1 = [
        rec("S1-1", "Zephyrine Bakery LLC", "12 Quillon Street, Dayton, OH"),
        rec("S1-2", "Acme Tools Inc", "5 Main Street, Dayton, OH"),
        rec("S1-3", "Zephyrine Bakery", "12 Quillon Street, Pune", country="India"),
    ]
    others = [
        rec("S2-1", "ZEPHYRINE BAKERY", "12 QUILLON ST, DAYTON, OH"),        # match S1-1
        rec("S3-1", "zephyrine.com", "12 Quillon Street, Dayton, Ohio"),       # match S1-1 via address
        rec("S2-2", "Acme Tols", "5 Main St, Dayton, OH"),                     # match S1-2
        rec("S3-2", "Zephyrine Bakery", "12 Quillon Street, Pune", "India"),  # match S1-3 only
    ]
    # Filler records make "main", "street", "dayton" common, "zephyrine"/"quillon" rare.
    filler = [rec(f"S2-f{i}", f"Filler {i} Shop", f"{i} Main Street, Dayton, OH") for i in range(40)]
    return s1, others + filler


class TokenStatsTests(unittest.TestCase):
    def setUp(self):
        s1, others = toy_corpus()
        self.stats = TokenStats.from_records(s1 + others)

    def test_country_specific_frequencies(self):
        us = self.stats.doc_freq(NAME, "US", "zephyrine")
        india = self.stats.doc_freq(NAME, "India", "zephyrine")
        self.assertEqual(india, 2)
        self.assertGreater(us, india)
        self.assertEqual(self.stats.doc_freq(NAME, "US", "never-seen"), 1)

    def test_rarest_orders_by_frequency(self):
        toks = record_token_sets(rec("x", "", "12 Quillon Street, Dayton, OH"))[ADDRESS]
        rare = [t for t, _ in self.stats.rarest(ADDRESS, "US", toks, 2)]
        self.assertIn("quillon", rare)
        self.assertNotIn("dayton", rare)

    def test_idf_monotonic(self):
        self.assertGreater(self.stats.idf(ADDRESS, "US", "quillon"), self.stats.idf(ADDRESS, "US", "dayton"))


class BlockerTests(unittest.TestCase):
    def setUp(self):
        self.s1, self.others = toy_corpus()
        self.stats = TokenStats.from_records(self.s1 + self.others)

    def test_block_size_cap_drops_common_keys(self):
        cfg = BlockingConfig(name_keys=2, address_keys=5, max_block_size=10)
        p = S1Profile(self.s1[1], self.stats, cfg)  # Acme Tools, 5 Main Street, Dayton
        keyed = {t for _, t in p.keys}
        self.assertNotIn("main", keyed)
        self.assertNotIn("dayton", keyed)
        self.assertIn("acme", keyed)

    def test_fallback_key_when_every_token_is_common(self):
        common = rec("S1-9", "Filler Shop", "Main Street, Dayton, OH")
        strict = S1Profile(common, self.stats, BlockingConfig(max_block_size=5, fallback_max_block_size=0))
        self.assertEqual(strict.keys, [])
        loose = S1Profile(common, self.stats, BlockingConfig(max_block_size=5, fallback_max_block_size=1000))
        self.assertEqual(len(loose.keys), 1)
        capped = S1Profile(common, self.stats, BlockingConfig(max_block_size=5, fallback_max_block_size=10))
        self.assertEqual(capped.keys, [])

    def test_score_prefers_rare_shared_tokens(self):
        p = S1Profile(self.s1[0], self.stats, BlockingConfig())
        rare = record_token_sets(rec("c", "Zephyrine", ""))
        common = record_token_sets(rec("c", "", "Dayton"))
        self.assertGreater(score(p, rare), score(p, common))
        self.assertEqual(score(p, record_token_sets(rec("c", "Unrelated", "Nowhere"))), 0.0)

    def test_coverage_mode_keeps_name_only_match_above_address_neighbour(self):
        name_only = record_token_sets(rec("c1", "Zephyrine Bakery", ""))
        neighbour = record_token_sets(rec("c2", "Other Shop", "12 Quillon Street, Dayton, OH"))
        cov = S1Profile(self.s1[0], self.stats, BlockingConfig(score_mode="coverage"))
        self.assertGreater(score(cov, name_only), score(cov, neighbour))
        self.assertAlmostEqual(score(cov, name_only), 2.0)
        full = record_token_sets(self.s1[0])
        self.assertAlmostEqual(score(cov, full), 2.0)

    def test_unknown_score_mode_rejected(self):
        with self.assertRaises(ValueError):
            S1Profile(self.s1[0], self.stats, BlockingConfig(score_mode="nope"))

    def test_score_views_split_name_and_address(self):
        p = S1Profile(self.s1[0], self.stats, BlockingConfig())
        combined, name, addr = score_views(p, record_token_sets(rec("c", "Zephyrine", "Dayton")))
        self.assertGreater(name, 0)
        self.assertGreater(addr, 0)
        self.assertAlmostEqual(combined, name + addr)

    def test_generate_candidates_finds_matches_and_respects_country(self):
        cfg = BlockingConfig(name_keys=2, address_keys=2, max_block_size=20, top_k=5)
        out = generate_candidates(self.s1, self.others, self.stats, cfg)
        self.assertIn("S2-1", out["S1-1"])
        self.assertIn("S3-1", out["S1-1"])  # name differs, rare address token blocks it
        self.assertIn("S2-2", out["S1-2"])  # typo in name, address still shares tokens
        self.assertNotIn("S3-2", out["S1-1"])  # India record never a US candidate
        self.assertEqual(out["S1-3"], ["S3-2"])
        self.assertTrue(all(len(v) <= 5 for v in out.values()))

    def test_top_k_limits_and_orders(self):
        cfg = BlockingConfig(name_keys=2, address_keys=3, max_block_size=100, top_k=3)
        out = generate_candidates(self.s1, self.others, self.stats, cfg)
        self.assertEqual(len(out["S1-2"]), 3)
        self.assertEqual(out["S1-2"][0], "S2-2")  # best-scoring candidate first

    def test_key_index_is_country_scoped(self):
        cfg = BlockingConfig()
        idx = build_key_index([S1Profile(r, self.stats, cfg) for r in self.s1])
        self.assertTrue(all(country in ("US", "India") for _, country, _ in idx))
        self.assertNotEqual(idx.get((NAME, "US", "zephyrine")), idx.get((NAME, "India", "zephyrine")))


class TopKCollectorTests(unittest.TestCase):
    def test_keeps_best_and_merges(self):
        a, b = TopKCollector(1, 3), TopKCollector(1, 3)
        for s, eid in [(1.0, "S2-a"), (5.0, "S2-b"), (3.0, "S3-c")]:
            a.push(0, (s, s, 0.0), eid)
        for s, eid in [(4.0, "S2-d"), (0.5, "S3-e"), (6.0, "S3-f")]:
            b.push(0, (s, s, 0.0), eid)
        a.merge(b)
        self.assertEqual(a.pool_sizes[0], 6)
        self.assertEqual(a.select(0, 3), ["S3-f", "S2-b", "S2-d"])

    def test_capacity_bound(self):
        c = TopKCollector(1, 2)
        for i in range(10):
            c.push(0, (float(i + 1), 0.0, float(i + 1)), f"S2-{i}")
        self.assertEqual(len(c.heaps[0]["combined"]), 2)
        self.assertEqual(c.heaps[0]["name"], [])  # zero evidence is never kept
        self.assertEqual(c.select(0, 5), ["S2-9", "S2-8"])

    def test_quota_reserves_name_only_slots(self):
        c = TopKCollector(1, 10)
        # Neighbours: strong address evidence, no name evidence.
        for i in range(5):
            c.push(0, (10.0 + i, 0.0, 10.0 + i), f"S2-n{i}")
        # Name-only record: weaker combined score but the best name evidence.
        c.push(0, (6.0, 6.0, 0.0), "S3-true")
        self.assertNotIn("S3-true", c.select(0, 3))
        picked = c.select(0, 3, name_quota=0.34)
        self.assertIn("S3-true", picked)
        self.assertEqual(len(picked), 3)
        self.assertEqual(picked[-1], "S3-true")  # output ordered by combined score

    def test_quota_never_exceeds_k_and_dedupes(self):
        c = TopKCollector(1, 10)
        for i in range(6):
            c.push(0, (float(10 - i), float(10 - i), float(10 - i)), f"S2-{i}")
        picked = c.select(0, 4, name_quota=0.5, address_quota=0.5)
        self.assertEqual(picked, ["S2-0", "S2-1", "S2-2", "S2-3"])


if __name__ == "__main__":
    unittest.main()
