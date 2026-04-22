"""Tests for the Phase-19 bounded retrieval worker."""

from __future__ import annotations

import unittest

import numpy as np

from vision_mvp.core.context_ledger import ContextLedger, hash_embedding
from vision_mvp.core.bounded_worker import BoundedRetrievalWorker


def _embed(text: str) -> np.ndarray:
    return hash_embedding(text, dim=32)


def _echo_llm(prompt: str) -> str:
    """Mock LLM: returns a deterministic answer that echoes the first
    [CID] tag found in the prompt. Lets us confirm the worker put the
    right excerpt into the prompt without needing a real model."""
    if "[" not in prompt or "]" not in prompt:
        return "no excerpts seen"
    start = prompt.index("[")
    end = prompt.index("]", start)
    return f"answer-from-{prompt[start:end+1]}"


class TestBoundedWorker(unittest.TestCase):
    def setUp(self):
        self.ledger = ContextLedger(embed_dim=32, embed_fn=_embed)
        # Seed three artifacts.
        self.h1 = self.ledger.put(
            "Frankfurt vendor NordAxis incident OS-2026-0001 ticket REL-12345",
            metadata={"section": 0})
        self.h2 = self.ledger.put(
            "Singapore HelixQ latency spike incident OS-2026-0002",
            metadata={"section": 1})
        self.h3 = self.ledger.put(
            "Cape Town billing mis-ledger incident OS-2026-0003",
            metadata={"section": 2})
        self.worker = BoundedRetrievalWorker(
            ledger=self.ledger,
            llm_call=_echo_llm,
            prompt_budget_chars=400,
            top_k=3,
            fetch_chars_per_handle=80,
        )

    def test_answer_returns_worker_result(self):
        r = self.worker.answer("Frankfurt vendor")
        self.assertGreater(len(r.cited_cids), 0)
        self.assertEqual(r.exact_input, True)
        self.assertEqual(r.llm_calls, 1)

    def test_prompt_size_bounded_by_budget(self):
        r = self.worker.answer("anything")
        self.assertLessEqual(r.prompt_chars, 400)

    def test_excerpt_bytes_are_exact(self):
        r = self.worker.answer("Frankfurt vendor NordAxis")
        # The cited CIDs must all exist in the ledger (provenance check).
        for cid in r.cited_cids:
            self.assertIn(cid, self.ledger)
        # The prompt contains the literal CID tag.
        for cid in r.cited_cids:
            self.assertIn(f"[{cid[:8]}]", r.prompt)
        # The bytes after the tag are byte-equal to the ledger artifact's
        # prefix (up to fetch_chars_per_handle).
        for cid in r.cited_cids:
            stored = self.ledger.get_body(cid)
            tag = f"[{cid[:8]}] "
            start = r.prompt.index(tag) + len(tag)
            # The excerpt continues until the next "\n\n[" boundary or end.
            nxt = r.prompt.find("\n\n[", start)
            end = nxt if nxt != -1 else len(r.prompt)
            excerpt = r.prompt[start:end]
            # excerpt is a prefix of stored body, length ≤ fetch budget.
            self.assertEqual(stored[:len(excerpt)], excerpt)
            self.assertLessEqual(len(excerpt), 80)

    def test_empty_question_returns_empty(self):
        r = self.worker.answer("")
        self.assertEqual(r.answer, "")
        self.assertEqual(r.cited_cids, [])
        self.assertEqual(r.fetch_count, 0)

    def test_prefilter_drops_handles(self):
        # Drop everything → zero excerpts cited.
        r = self.worker.answer("Frankfurt", prefilter=lambda h, s: False)
        self.assertEqual(r.cited_cids, [])

    def test_fetch_count_accounted(self):
        r = self.worker.answer("Frankfurt vendor")
        self.assertGreaterEqual(r.fetch_count, 1)
        self.assertLessEqual(r.fetch_count, 3)


class TestBudgetTooSmallForExcerpts(unittest.TestCase):
    def test_question_only_budget(self):
        ledger = ContextLedger(embed_dim=32, embed_fn=_embed)
        ledger.put("some artifact body that's long enough to overflow")
        worker = BoundedRetrievalWorker(
            ledger=ledger, llm_call=_echo_llm,
            prompt_budget_chars=20,    # too small for any excerpt
            top_k=2, fetch_chars_per_handle=200,
        )
        r = worker.answer("anything")
        # No excerpts could fit; cited_cids should be empty; LLM still called.
        self.assertEqual(r.cited_cids, [])
        self.assertLessEqual(r.prompt_chars, 1000)


if __name__ == "__main__":
    unittest.main()
