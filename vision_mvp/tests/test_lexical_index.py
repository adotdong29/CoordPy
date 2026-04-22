"""Tests for the BM25 lexical index — Phase 20."""

from __future__ import annotations

import unittest

from vision_mvp.core.lexical_index import (
    LexicalIndex, reciprocal_rank_fusion, tokenize, _ascii_fold,
)


class TestTokenize(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(tokenize("hello world"), ["hello", "world"])

    def test_punctuation_split(self):
        self.assertEqual(tokenize("incident OS-2026-0001 in Frankfurt."),
                         ["incident", "os-2026-0001", "frankfurt"])

    def test_stopwords_dropped(self):
        toks = tokenize("the vendor is in the building")
        self.assertNotIn("the", toks)
        self.assertNotIn("is", toks)
        self.assertNotIn("in", toks)
        self.assertIn("vendor", toks)

    def test_unicode(self):
        # São Paulo and Reykjavík must not be lost.
        toks = tokenize("incident in São Paulo and Reykjavík")
        self.assertIn("são", toks)
        self.assertIn("reykjavík", toks)

    def test_ascii_fold(self):
        self.assertEqual(_ascii_fold("são"), "sao")
        self.assertEqual(_ascii_fold("reykjavík"), "reykjavik")


class TestLexicalIndexBasic(unittest.TestCase):
    def setUp(self):
        self.idx = LexicalIndex()
        self.idx.add("c1", "incident OS-2026-0001 in Frankfurt with NordAxis")
        self.idx.add("c2", "incident OS-2026-0002 in Singapore with HelixQ")
        self.idx.add("c3", "billing failure in Cape Town tracking REL-12345")

    def test_search_by_id_token(self):
        # An exact ID token should rank its document first.
        hits = self.idx.knn("OS-2026-0002", k=3)
        self.assertGreater(len(hits), 0)
        self.assertEqual(hits[0][0], "c2")

    def test_search_by_ticket(self):
        hits = self.idx.knn("REL-12345", k=3)
        self.assertEqual(hits[0][0], "c3")

    def test_search_by_city(self):
        hits = self.idx.knn("Frankfurt vendor", k=3)
        self.assertEqual(hits[0][0], "c1")

    def test_no_match_returns_empty(self):
        hits = self.idx.knn("xyzunrelated", k=5)
        self.assertEqual(hits, [])

    def test_remove_drops_doc(self):
        self.idx.remove("c1")
        hits = self.idx.knn("Frankfurt", k=3)
        # c1 was the only Frankfurt doc; no other doc mentions it.
        self.assertEqual(hits, [])

    def test_avgdl_positive(self):
        self.assertGreater(self.idx.avgdl, 0.0)


class TestUnicodeQuery(unittest.TestCase):
    def test_unaccented_query_matches_accented_corpus(self):
        idx = LexicalIndex()
        idx.add("c1", "Sao Paulo did NOT happen here")    # ASCII
        idx.add("c2", "São Paulo region had the outage")  # accented
        # Query without diacritics must still pull c2.
        hits = idx.knn("São Paulo outage", k=2)
        cids = {h[0] for h in hits}
        self.assertIn("c2", cids)
        # And vice versa: ASCII query "Sao Paulo" should also find c2 via
        # the accent-folded token added at indexing time.
        hits2 = idx.knn("Sao Paulo outage", k=2)
        self.assertIn("c2", {h[0] for h in hits2})


class TestRRF(unittest.TestCase):
    def test_basic_fusion_consensus_wins(self):
        # ranking 1 likes A best, C second, B third
        # ranking 2 likes C best, B second, A third
        # C is the consensus pick: rank 2 in r1 AND rank 1 in r2.
        r1 = [("A", 0.9), ("C", 0.5), ("B", 0.1)]
        r2 = [("C", 0.9), ("B", 0.6), ("A", 0.2)]
        fused = reciprocal_rank_fusion([r1, r2], top_k=3)
        cids = [c for c, _ in fused]
        # C should win — best aggregate rank.
        self.assertEqual(cids[0], "C")
        self.assertEqual(set(cids), {"A", "B", "C"})

    def test_handles_disjoint_rankings(self):
        # If only one ranking has a doc, it still gets a score.
        r1 = [("A", 0.9), ("B", 0.5)]
        r2 = [("C", 0.9), ("D", 0.5)]
        fused = reciprocal_rank_fusion([r1, r2], top_k=4)
        self.assertEqual(len(fused), 4)
        self.assertEqual(set(c for c, _ in fused), {"A", "B", "C", "D"})


class TestUpdateSemantics(unittest.TestCase):
    def test_overwrite_changes_index(self):
        idx = LexicalIndex()
        idx.add("c1", "alpha")
        idx.add("c1", "beta")     # overwrite
        # Only "beta" should match.
        self.assertEqual(idx.knn("alpha", k=2), [])
        self.assertEqual(idx.knn("beta", k=2)[0][0], "c1")

    def test_remove_idempotent(self):
        idx = LexicalIndex()
        idx.add("c1", "alpha")
        idx.remove("c1")
        idx.remove("c1")    # no error


if __name__ == "__main__":
    unittest.main()
