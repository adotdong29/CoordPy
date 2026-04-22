"""Smoke test for the Phase-19 lossless benchmark — runs in mock mode.

This is a regression check: the benchmark harness must run end-to-end
against the deterministic mock LLM and produce the expected qualitative
result (lossless > map-reduce on the needle questions). It does NOT
require Ollama and finishes in under a second.
"""

from __future__ import annotations

import unittest

from vision_mvp.experiments.phase19_lossless import (
    MockLLM, run_lossless, run_map_reduce, run_oracle,
)
from vision_mvp.core.context_ledger import hash_embedding
from vision_mvp.tasks.needle_corpus import NeedleCorpus


class TestPhase19Smoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Phase-19 smoke: single-fact questions only. Aggregation
        # (Phase-21) golds aren't substrings of the document, so the
        # mock extractor cannot answer them perfectly even from oracle.
        cls.corpus = NeedleCorpus(n_sections=12, seed=19,
                                   include_aggregation=False)
        cls.corpus.build()
        cls.embed = lambda t: hash_embedding(t, dim=32)

    def setUp(self):
        self.llm = MockLLM()

    def test_oracle_perfect_under_mock(self):
        rep = run_oracle(self.corpus, self.llm, prompt_budget_chars=200_000)
        a = rep.aggregate()
        self.assertEqual(a["exact_correct"], a["n_questions"],
                         f"oracle should be perfect: {a}")

    def test_lossless_beats_or_matches_map_reduce(self):
        llm_mr = MockLLM()
        llm_l = MockLLM()
        rep_mr = run_map_reduce(
            self.corpus, llm_mr,
            summary_prompt_chars_max=4000,
            answer_prompt_chars_max=4000,
            progress=lambda _m: None,
        )
        rep_l = run_lossless(
            self.corpus, llm_l,
            embed_fn=type(self).embed, embed_dim=32,
            prompt_budget_chars=4000, top_k=5, fetch_chars_per_handle=600,
            progress=lambda _m: None,
        )
        a_mr = rep_mr.aggregate()
        a_l = rep_l.aggregate()
        self.assertGreaterEqual(
            a_l["exact_correct_rate"], a_mr["exact_correct_rate"],
            f"lossless ({a_l}) should be ≥ map_reduce ({a_mr}) on needle Qs")

    def test_lossless_exactness_diagnostic(self):
        """When the mock extractor is perfect, exact_correct == fact_in_input
        in lossless mode. This validates Conjecture L3 (no extraction noise
        confounds substrate diagnostics)."""
        rep = run_lossless(
            self.corpus, self.llm,
            embed_fn=type(self).embed, embed_dim=32,
            prompt_budget_chars=4000, top_k=8, fetch_chars_per_handle=600,
            progress=lambda _m: None,
        )
        for q in rep.questions:
            self.assertEqual(
                q.exact_correct, q.fact_in_input,
                f"L3 violated for {q.kind!r} q={q.question!r}: "
                f"exact={q.exact_correct} fact_in={q.fact_in_input}")

    def test_lossless_prompt_within_budget(self):
        rep = run_lossless(
            self.corpus, self.llm,
            embed_fn=type(self).embed, embed_dim=32,
            prompt_budget_chars=4000, top_k=8, fetch_chars_per_handle=600,
            progress=lambda _m: None,
        )
        for q in rep.questions:
            self.assertLessEqual(
                q.prompt_chars, 4000,
                f"prompt overran budget on {q.kind!r}: {q.prompt_chars}")


if __name__ == "__main__":
    unittest.main()
