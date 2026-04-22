"""Tests for the Phase-19 lossless context substrate."""

from __future__ import annotations

import unittest

import numpy as np

from vision_mvp.core.context_ledger import (
    ContextLedger, Handle, hash_embedding,
)


def _embed(text: str) -> np.ndarray:
    return hash_embedding(text, dim=32)


class TestPutAndIdempotency(unittest.TestCase):
    def setUp(self):
        self.l = ContextLedger(embed_dim=32, embed_fn=_embed)

    def test_put_returns_handle_with_cid(self):
        h = self.l.put("hello world", metadata={"doc_id": "x"})
        self.assertIsInstance(h, Handle)
        self.assertEqual(len(h.cid), 64)  # SHA-256 hex

    def test_idempotent_on_same_body_meta(self):
        h1 = self.l.put("alpha", metadata={"k": "v"})
        h2 = self.l.put("alpha", metadata={"k": "v"})
        self.assertEqual(h1.cid, h2.cid)
        # Only one entry stored.
        self.assertEqual(len(self.l), 1)

    def test_different_meta_gives_different_cid(self):
        h1 = self.l.put("alpha", metadata={"k": "v"})
        h2 = self.l.put("alpha", metadata={"k": "different"})
        self.assertNotEqual(h1.cid, h2.cid)
        self.assertEqual(len(self.l), 2)


class TestFetchExactness(unittest.TestCase):
    def setUp(self):
        self.l = ContextLedger(embed_dim=32, embed_fn=_embed)
        # A specific body with edge characters: leading whitespace, unicode,
        # punctuation. The substrate must return these byte-equal.
        self.body = "  Lorem\n  ipsum — dolor «sit» amet, consectetur 12345."
        self.h = self.l.put(self.body, metadata={"doc_id": "lorem"})

    def test_fetch_returns_exact_bytes(self):
        text = self.l.fetch(self.h)
        self.assertEqual(text, self.body)

    def test_fetch_with_explicit_span(self):
        prefix = self.l.fetch(self.h, span=(0, 11))
        self.assertEqual(prefix, self.body[:11])

    def test_fetch_full_after_partial_unchanged(self):
        # Two fetches don't perturb the stored body.
        _ = self.l.fetch(self.h, span=(0, 5))
        text = self.l.fetch(self.h)
        self.assertEqual(text, self.body)

    def test_fetch_unknown_cid_raises(self):
        bogus = Handle(cid="0" * 64, span=None, fingerprint="", metadata=())
        with self.assertRaises(KeyError):
            self.l.fetch(bogus)

    def test_stats_increment_on_fetch(self):
        before = self.l.stats_dict()["bytes_fetched"]
        text = self.l.fetch(self.h)
        after = self.l.stats_dict()["bytes_fetched"]
        self.assertEqual(after - before, len(text))


class TestSearch(unittest.TestCase):
    def setUp(self):
        self.l = ContextLedger(embed_dim=32, embed_fn=_embed)
        # Three artifacts about distinct topics.
        self.h1 = self.l.put("Frankfurt incident with NordAxis vendor",
                             metadata={"section": 0})
        self.h2 = self.l.put("Singapore latency spike on HelixQ",
                             metadata={"section": 1})
        self.h3 = self.l.put("Cape Town billing mis-ledger",
                             metadata={"section": 2})

    def test_search_returns_handles_only(self):
        hits = self.l.search("Frankfurt vendor", top_k=2)
        self.assertEqual(len(hits), 2)
        for handle, sim in hits:
            self.assertIsInstance(handle, Handle)
            self.assertIsInstance(sim, float)

    def test_search_does_not_load_bodies(self):
        # The handle never carries the body — only fingerprint.
        before_fetches = self.l.stats_dict()["n_fetch"]
        hits = self.l.search("Frankfurt", top_k=3)
        after_fetches = self.l.stats_dict()["n_fetch"]
        self.assertEqual(before_fetches, after_fetches)
        # Fingerprint is the first non-blank line, NOT the full body.
        for h, _ in hits:
            self.assertLessEqual(len(h.fingerprint), 80)

    def test_top_hit_is_relevant(self):
        # Hash-based embedding isn't semantic, but with these specific
        # strings the top-1 for "Frankfurt" should be h1 (most overlapping
        # 3-grams). We don't assert identity, only that h1 is in top-2.
        hits = self.l.search("Frankfurt", top_k=2)
        cids = {h.cid for h, _ in hits}
        self.assertIn(self.h1.cid, cids)


class TestProvenance(unittest.TestCase):
    def setUp(self):
        self.l = ContextLedger(embed_dim=32, embed_fn=_embed)
        # Build a small derivation chain: source → digest → answer.
        self.source = self.l.put("source text", metadata={"kind": "source"})
        self.digest = self.l.put("a digest of the source",
                                 parent_cids=[self.source.cid],
                                 metadata={"kind": "digest"})
        self.answer = self.l.put("final answer derived from the digest",
                                 parent_cids=[self.digest.cid],
                                 metadata={"kind": "answer"})

    def test_parents_returns_immediate(self):
        ps = self.l.parents(self.answer)
        self.assertEqual(len(ps), 1)
        self.assertEqual(ps[0].cid, self.digest.cid)

    def test_lineage_walks_to_root(self):
        line = self.l.lineage(self.answer)
        cids = [h.cid for h in line]
        self.assertIn(self.digest.cid, cids)
        self.assertIn(self.source.cid, cids)
        # Root (source) must be after the immediate parent (digest) in BFS.
        self.assertLess(cids.index(self.digest.cid), cids.index(self.source.cid))

    def test_unknown_parent_raises(self):
        with self.assertRaises(KeyError):
            self.l.put("orphan",
                       parent_cids=["deadbeef" * 8],
                       metadata={"kind": "orphan"})

    def test_children_lookup(self):
        kids = self.l.children(self.source)
        self.assertEqual(len(kids), 1)
        self.assertEqual(kids[0].cid, self.digest.cid)


class TestMerkleRoot(unittest.TestCase):
    def setUp(self):
        self.l = ContextLedger(embed_dim=32, embed_fn=_embed)
        for s in ("a", "b", "c", "d"):
            self.l.put(s, metadata={"k": s})

    def test_root_is_deterministic(self):
        r1, _ = self.l.merkle_root()
        r2, _ = self.l.merkle_root()
        self.assertEqual(r1, r2)

    def test_root_changes_with_new_artifact(self):
        r1, _ = self.l.merkle_root()
        self.l.put("e", metadata={"k": "e"})
        r2, _ = self.l.merkle_root()
        self.assertNotEqual(r1, r2)


class TestStats(unittest.TestCase):
    def test_counters_accurate(self):
        l = ContextLedger(embed_dim=32, embed_fn=_embed)
        h = l.put("aaa")
        h2 = l.put("bbb")
        l.search("aaa", top_k=2)
        l.fetch(h)
        l.fetch(h2, span=(0, 2))
        s = l.stats_dict()
        self.assertEqual(s["n_artifacts"], 2)
        self.assertEqual(s["n_put"], 2)
        self.assertEqual(s["n_search"], 1)
        self.assertEqual(s["n_fetch"], 2)
        self.assertEqual(s["bytes_fetched"], 3 + 2)


if __name__ == "__main__":
    unittest.main()
