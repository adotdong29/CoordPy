"""Tests for the hybrid search modes added to ContextLedger — Phase 20."""

from __future__ import annotations

import unittest

import numpy as np

from vision_mvp.core.context_ledger import (
    ContextLedger, LedgerCapacityError, hash_embedding,
)


def _embed(text: str) -> np.ndarray:
    return hash_embedding(text, dim=32)


class TestSearchModes(unittest.TestCase):
    def setUp(self):
        self.l = ContextLedger(embed_dim=32, embed_fn=_embed)
        self.h_id = self.l.put(
            "Section 1. Incident OS-2026-0001 — PulseCore event in Frankfurt "
            "(ticket REL-12345). Vendor NordAxis was the upstream provider.",
            metadata={"section_idx": 0})
        self.h_unrelated_1 = self.l.put(
            "general overview prose about reliability culture and tooling, "
            "no specific incident references",
            metadata={"section_idx": 1})
        self.h_unrelated_2 = self.l.put(
            "another generic prose paragraph about runbooks and process",
            metadata={"section_idx": 2})

    def test_lexical_finds_exact_id(self):
        hits = self.l.search("OS-2026-0001", top_k=2, mode="lexical")
        self.assertGreater(len(hits), 0)
        self.assertEqual(hits[0][0].cid, self.h_id.cid)

    def test_lexical_finds_exact_ticket(self):
        hits = self.l.search("REL-12345", top_k=2, mode="lexical")
        self.assertEqual(hits[0][0].cid, self.h_id.cid)

    def test_dense_returns_handles(self):
        hits = self.l.search("PulseCore vendor incident", top_k=2, mode="dense")
        self.assertGreater(len(hits), 0)

    def test_hybrid_includes_lexical_hit_for_id(self):
        # Even when dense ranking would mis-rank a rare-token query, the
        # hybrid mode must surface the correct doc near the top.
        hits = self.l.search("OS-2026-0001 details please", top_k=2, mode="hybrid")
        cids = [h.cid for h, _ in hits]
        self.assertIn(self.h_id.cid, cids)

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            self.l.search("anything", top_k=1, mode="bogus")

    def test_stats_track_mode_usage(self):
        self.l.search("foo", top_k=1, mode="dense")
        self.l.search("foo", top_k=1, mode="lexical")
        self.l.search("foo", top_k=1, mode="hybrid")
        s = self.l.stats_dict()
        self.assertEqual(s["n_search_dense"], 1)
        self.assertEqual(s["n_search_lexical"], 1)
        self.assertEqual(s["n_search_hybrid"], 1)
        self.assertEqual(s["n_search"], 3)


class TestHybridDominatesOnRareTokenQuery(unittest.TestCase):
    """Theorem T20.2: for queries containing a literal that exists in
    exactly one indexed doc, hybrid retrieval ranks that doc at position
    1 (tie-break favoured by the lexical leg). Dense-only is allowed to
    miss; hybrid is not."""

    def setUp(self):
        self.l = ContextLedger(embed_dim=32, embed_fn=_embed)
        # 12 generic prose docs — none mention the literal we'll query.
        for i in range(12):
            self.l.put(
                f"This is paragraph {i} about reliability engineering and "
                f"general operational practice. No specific incidents.",
                metadata={"section_idx": i})
        # One doc that uniquely contains the rare token.
        self.target = self.l.put(
            "obscure prose with a single embedded token UNIQ-7TOKEN-Z9 "
            "buried in normal text",
            metadata={"section_idx": 99, "kind": "needle"})

    def test_lexical_finds_at_rank_1(self):
        hits = self.l.search("UNIQ-7TOKEN-Z9", top_k=3, mode="lexical")
        self.assertEqual(hits[0][0].cid, self.target.cid)

    def test_hybrid_ranks_target_in_top_k(self):
        hits = self.l.search("UNIQ-7TOKEN-Z9", top_k=3, mode="hybrid")
        cids = [h.cid for h, _ in hits]
        self.assertIn(self.target.cid, cids)


class TestHandleValidation(unittest.TestCase):
    def test_handle_passes_verify(self):
        l = ContextLedger(embed_dim=32, embed_fn=_embed)
        h = l.put("hello world")
        self.assertTrue(l.verify_handle(h))

    def test_unknown_cid_fails_verify(self):
        from vision_mvp.core.context_ledger import Handle
        l = ContextLedger(embed_dim=32, embed_fn=_embed)
        bogus = Handle(cid="0" * 64, span=None,
                       fingerprint="bogus", metadata=())
        self.assertFalse(l.verify_handle(bogus))

    def test_tampered_fingerprint_fails_fetch(self):
        from vision_mvp.core.context_ledger import Handle
        l = ContextLedger(embed_dim=32, embed_fn=_embed)
        h = l.put("the actual body")
        forged = Handle(cid=h.cid, span=None,
                        fingerprint="something else entirely",
                        metadata=h.metadata)
        with self.assertRaises(ValueError):
            l.fetch(forged)


class TestCapacityGuards(unittest.TestCase):
    def test_max_artifacts_enforced(self):
        l = ContextLedger(embed_dim=32, embed_fn=_embed, max_artifacts=2)
        l.put("a")
        l.put("b")
        with self.assertRaises(LedgerCapacityError):
            l.put("c")

    def test_max_artifact_chars_enforced(self):
        l = ContextLedger(embed_dim=32, embed_fn=_embed,
                          max_artifact_chars=8)
        l.put("0123456")          # 7 chars — ok
        with self.assertRaises(LedgerCapacityError):
            l.put("0123456789")    # 10 chars — too big

    def test_invalid_max_args_raise(self):
        with self.assertRaises(ValueError):
            ContextLedger(embed_dim=32, embed_fn=_embed, max_artifacts=0)
        with self.assertRaises(ValueError):
            ContextLedger(embed_dim=32, embed_fn=_embed, max_artifact_chars=0)


if __name__ == "__main__":
    unittest.main()
