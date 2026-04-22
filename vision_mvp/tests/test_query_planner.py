"""Tests for the natural-language → operator planner (Phase 21)."""

from __future__ import annotations

import unittest

import numpy as np

from vision_mvp.core.context_ledger import ContextLedger, hash_embedding
from vision_mvp.core.exact_ops import StageGroups, StageList, StageScalar
from vision_mvp.core.query_planner import QueryPlanner
from vision_mvp.tasks.needle_corpus import NeedleCorpus


def _embed(text: str) -> np.ndarray:
    return hash_embedding(text, dim=32)


class TestPatternRecognition(unittest.TestCase):
    def setUp(self):
        self.p = QueryPlanner()

    def test_count_distinct_vendors(self):
        r = self.p.plan("How many distinct vendors are mentioned?")
        self.assertEqual(r.pattern, "count_distinct_field")
        self.assertEqual(r.matched_groups["field"], "vendor")
        self.assertIsNotNone(r.plan)

    def test_count_distinct_products(self):
        r = self.p.plan("how many unique products are listed in the corpus")
        self.assertEqual(r.pattern, "count_distinct_field")
        self.assertEqual(r.matched_groups["field"], "product")

    def test_count_sev_filter(self):
        r = self.p.plan("How many Sev-1 incidents were there?")
        self.assertEqual(r.pattern, "count_filter")
        self.assertEqual(r.matched_groups["sev"], "Sev-1")

    def test_count_in_city(self):
        r = self.p.plan("How many incidents were recorded in Lyon?")
        self.assertEqual(r.pattern, "count_filter")
        self.assertEqual(r.matched_groups["city"], "Lyon")

    def test_list_in_city(self):
        r = self.p.plan("List all incidents in Frankfurt.")
        self.assertEqual(r.pattern, "list_filter")
        self.assertEqual(r.matched_groups["field"], "incident_id")
        self.assertEqual(r.matched_groups["city"], "Frankfurt")

    def test_top_vendor(self):
        r = self.p.plan("Which vendor appears most often?")
        self.assertEqual(r.pattern, "top_group")
        self.assertEqual(r.matched_groups["field"], "vendor")
        self.assertTrue(r.matched_groups["most"])

    def test_min_max(self):
        r = self.p.plan("What is the largest MTTD hours across all incidents?")
        self.assertEqual(r.pattern, "min_max_field")
        self.assertEqual(r.matched_groups["field"], "mttd_hours")
        self.assertTrue(r.matched_groups["max"])

    def test_sum_for_product(self):
        r = self.p.plan("What is the total MTTD hours across all PulseCore incidents?")
        self.assertEqual(r.pattern, "sum_field")
        self.assertEqual(r.matched_groups["field"], "mttd_hours")
        self.assertEqual(r.matched_groups["product"], "PulseCore")

    def test_join_via_ref(self):
        r = self.p.plan("For incident OS-2026-0001, what is the related vendor?")
        self.assertEqual(r.pattern, "join_via_ref")
        self.assertEqual(r.matched_groups["start"], "OS-2026-0001")
        self.assertEqual(r.matched_groups["field"], "vendor")

    def test_unmatched(self):
        r = self.p.plan("Tell me a story about a vendor and a city.")
        self.assertIsNone(r.plan)
        self.assertEqual(r.pattern, "(unmatched)")


class TestPlannerExecution(unittest.TestCase):
    """End-to-end: planner emits a plan, plan executes against a ledger
    seeded from the needle corpus."""

    @classmethod
    def setUpClass(cls):
        cls.corpus = NeedleCorpus(n_sections=12, seed=21,
                                   include_aggregation=True)
        cls.corpus.build()
        cls.ledger = ContextLedger(embed_dim=32, embed_fn=_embed)
        for i, sec in enumerate(cls.corpus.sections):
            cls.ledger.put(sec, metadata=cls.corpus.section_meta[i])
        cls.planner = QueryPlanner()

    def test_count_distinct_vendors_correct(self):
        r = self.planner.plan("How many distinct vendors are mentioned?")
        result, _ = r.plan.execute(self.ledger)
        self.assertIsInstance(result, StageScalar)
        gold = len({m["vendor"] for m in self.corpus.section_meta})
        self.assertEqual(result.value, gold)

    def test_max_mttd_correct(self):
        r = self.planner.plan("What is the largest MTTD hours across all incidents?")
        result, _ = r.plan.execute(self.ledger)
        gold = max(m["mttd_hours"] for m in self.corpus.section_meta)
        self.assertEqual(result.value, gold)

    def test_min_sla_correct(self):
        r = self.planner.plan("What is the smallest SLA minutes across all incidents?")
        result, _ = r.plan.execute(self.ledger)
        gold = min(m["sla_minutes"] for m in self.corpus.section_meta)
        self.assertEqual(result.value, gold)

    def test_top_vendor_correct(self):
        from collections import Counter
        r = self.planner.plan("Which vendor appears most often?")
        result, _ = r.plan.execute(self.ledger)
        self.assertIsInstance(result, StageGroups)
        # Check top entry matches Counter's most_common(1).
        top_actual = Counter(m["vendor"] for m in self.corpus.section_meta).most_common(1)[0]
        top_planned = next(iter(result.groups.items()))
        self.assertEqual(top_planned[0], top_actual[0])
        self.assertEqual(top_planned[1], top_actual[1])

    def test_planner_is_llm_free(self):
        """The planner pipeline must not invoke any LLM, embedding, or
        body fetch when the operators are metadata-only."""
        before = self.ledger.stats_dict()
        r = self.planner.plan("How many distinct vendors are mentioned?")
        result, _ = r.plan.execute(self.ledger)
        after = self.ledger.stats_dict()
        self.assertEqual(after["n_fetch"], before["n_fetch"])
        self.assertEqual(after["n_search"], before["n_search"])


if __name__ == "__main__":
    unittest.main()
