"""Tests for the Phase-20 multi-hop BoundedRetrievalWorker."""

from __future__ import annotations

import unittest

import numpy as np

from vision_mvp.core.bounded_worker import (
    BoundedRetrievalWorker, extract_references,
)
from vision_mvp.core.context_ledger import ContextLedger, hash_embedding


def _embed(text: str) -> np.ndarray:
    return hash_embedding(text, dim=32)


def _identity_llm(prompt: str) -> str:
    """Returns the full prompt — lets us inspect what the worker fed in."""
    return prompt


class TestExtractReferences(unittest.TestCase):
    def test_incident_id(self):
        refs = extract_references("see also OS-2026-0042 for details")
        self.assertIn("OS-2026-0042", refs)

    def test_ticket_id(self):
        refs = extract_references("tracked as REL-12345 historical")
        self.assertIn("REL-12345", refs)

    def test_section_ref(self):
        refs = extract_references("compare with Section 7 above")
        self.assertIn("Section 7", refs)

    def test_dedup(self):
        refs = extract_references("OS-2026-0001 OS-2026-0001 OS-2026-0001")
        self.assertEqual(refs.count("OS-2026-0001"), 1)

    def test_empty(self):
        self.assertEqual(extract_references("no references here"), [])


class TestSingleHopUnchanged(unittest.TestCase):
    """The default (max_hops=1) worker behaves identically to Phase 19:
    one search, fetched bytes go in the prompt, single LLM call."""

    def setUp(self):
        self.l = ContextLedger(embed_dim=32, embed_fn=_embed)
        self.h = self.l.put("Section 1. Incident OS-2026-0001 — Frankfurt")
        self.worker = BoundedRetrievalWorker(
            ledger=self.l, llm_call=_identity_llm,
            prompt_budget_chars=400, top_k=2, fetch_chars_per_handle=120,
        )

    def test_one_hop_recorded(self):
        r = self.worker.answer("Frankfurt incident")
        self.assertEqual(len(r.hops), 1)
        self.assertEqual(r.llm_calls, 1)


class TestMultiHopExpansion(unittest.TestCase):
    """When hop 1 retrieves a body containing a structured reference, hop
    2 must search for that reference and add the referenced doc to the
    prompt — without an extra LLM call."""

    def setUp(self):
        self.l = ContextLedger(embed_dim=32, embed_fn=_embed)
        self.h_start = self.l.put(
            "Section 1. Incident OS-2026-0001 — Frankfurt event. "
            "Vendor NordAxis. Related: OS-2026-0042 (see ticket REL-99999).",
            metadata={"section_idx": 0})
        self.h_target = self.l.put(
            "Section 42. Incident OS-2026-0042 — Singapore event. "
            "Vendor HelixQ-Special. Tracked under ticket REL-99999.",
            metadata={"section_idx": 42})

    def test_multi_hop_pulls_referenced_doc(self):
        worker = BoundedRetrievalWorker(
            ledger=self.l, llm_call=_identity_llm,
            prompt_budget_chars=2000, top_k=3, fetch_chars_per_handle=300,
            search_mode="hybrid", max_hops=3,
        )
        r = worker.answer("vendor for the related incident referenced in "
                          "section 1 about OS-2026-0001")
        # Hop 1 must surface h_start; hop 2 must surface h_target via the
        # extracted OS-2026-0042 reference.
        self.assertIn(self.h_start.cid, r.cited_cids)
        self.assertIn(self.h_target.cid, r.cited_cids)
        self.assertEqual(r.llm_calls, 1, "multi-hop must NOT add LLM calls")
        self.assertGreaterEqual(len(r.hops), 2)
        # The hop-2 query string must contain the extracted reference.
        self.assertTrue(any(
            "OS-2026-0042" in h.query for h in r.hops[1:]),
            f"hop 2+ should query for the extracted reference; got "
            f"{[h.query for h in r.hops]}")

    def test_multi_hop_stops_when_no_new_refs(self):
        # If the start doc has no extractable refs, hop 2 should be empty
        # / skipped — verify by using a single doc with no IDs.
        l = ContextLedger(embed_dim=32, embed_fn=_embed)
        l.put("plain prose about reliability with no structured ids")
        worker = BoundedRetrievalWorker(
            ledger=l, llm_call=_identity_llm, prompt_budget_chars=2000,
            top_k=3, fetch_chars_per_handle=200, max_hops=4,
        )
        r = worker.answer("reliability prose")
        # At most one hop should have produced a non-empty candidate set.
        non_empty = [h for h in r.hops if h.candidate_cids]
        self.assertLessEqual(len(non_empty), 1)


class TestHybridWorker(unittest.TestCase):
    def test_hybrid_mode_propagates(self):
        l = ContextLedger(embed_dim=32, embed_fn=_embed)
        l.put("doc1 with token UNIQ-XYZ-1")
        l.put("doc2 unrelated prose")
        worker = BoundedRetrievalWorker(
            ledger=l, llm_call=_identity_llm, prompt_budget_chars=600,
            top_k=2, fetch_chars_per_handle=120, search_mode="hybrid",
        )
        r = worker.answer("UNIQ-XYZ-1")
        self.assertEqual(r.search_mode, "hybrid")
        self.assertGreater(len(r.cited_cids), 0)


if __name__ == "__main__":
    unittest.main()
