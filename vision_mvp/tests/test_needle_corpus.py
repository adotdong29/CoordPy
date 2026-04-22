"""Tests for the Phase-19 needle corpus."""

from __future__ import annotations

import unittest

from vision_mvp.tasks.needle_corpus import NeedleCorpus, NeedleQuestion


class TestNeedleCorpus(unittest.TestCase):
    def setUp(self):
        self.corp = NeedleCorpus(n_sections=12, target_words_per_section=120,
                                 seed=7)
        self.corp.build()

    def test_section_count(self):
        self.assertEqual(len(self.corp.sections), 12)
        self.assertEqual(len(self.corp.section_meta), 12)

    def test_each_section_self_contains_meta(self):
        # Every section's incident_id, vendor, city, ticket_id appear
        # literally in the section text. (The whole point is exact facts.)
        for s, m in zip(self.corp.sections, self.corp.section_meta):
            self.assertIn(m['incident_id'], s)
            self.assertIn(m['vendor'], s)
            self.assertIn(m['city'], s)
            self.assertIn(m['ticket_id'], s)

    def test_questions_built(self):
        self.assertGreater(len(self.corp.questions), 0)
        for q in self.corp.questions:
            self.assertIsInstance(q, NeedleQuestion)
            self.assertTrue(q.gold)
            self.assertTrue(q.question)
            if q.kind in self.corp._AGG_KINDS:
                # Aggregation gold is computed FROM metadata across many
                # sections — it doesn't necessarily appear verbatim in
                # any one section's body. Check that the source_section
                # tuple is non-empty instead.
                self.assertGreater(len(q.source_section), 0)
            else:
                # Single-hop / multi-hop: gold appears verbatim in the
                # terminal source section.
                terminal_idx = q.source_section[-1]
                self.assertIn(q.gold, self.corp.sections[terminal_idx])

    def test_multi_hop_chain_addressable(self):
        """For every multi-hop question, the start section must contain
        the related-incident reference token. This is what makes the
        multi-hop benchmark winnable: a worker can pattern-match the
        cross-reference out of the start section's body."""
        for q in self.corp.multi_hop_questions():
            start_idx, related_idx = q.source_section[0], q.source_section[1]
            related_id = self.corp.section_meta[related_idx]['incident_id']
            self.assertIn(related_id, self.corp.sections[start_idx],
                          f"hop-1 → hop-2 link absent for {q.kind!r}")

    def test_split_sets_partition_questions(self):
        s = self.corp.single_hop_questions()
        m = self.corp.multi_hop_questions()
        a = self.corp.aggregation_questions()
        self.assertGreater(len(s), 0)
        self.assertGreater(len(m), 0)
        self.assertGreater(len(a), 0)
        self.assertEqual(len(s) + len(m) + len(a), len(self.corp.questions))

    def test_aggregation_golds_match_metadata(self):
        """Spot-check that aggregation golds are correctly computed."""
        meta = self.corp.section_meta
        for q in self.corp.aggregation_questions():
            if q.kind == "count_distinct_vendors":
                self.assertEqual(int(q.gold), len({m["vendor"] for m in meta}))
            elif q.kind == "count_distinct_products":
                self.assertEqual(int(q.gold), len({m["product"] for m in meta}))
            elif q.kind == "max_mttd":
                self.assertEqual(int(q.gold),
                                 max(m["mttd_hours"] for m in meta))
            elif q.kind == "min_sla":
                self.assertEqual(int(q.gold),
                                 min(m["sla_minutes"] for m in meta))

    def test_score_exact_recognizes_gold(self):
        for q in self.corp.questions:
            self.assertTrue(self.corp.score_exact(
                f"the answer is {q.gold} ok", q))
            self.assertFalse(self.corp.score_exact("totally unrelated", q))

    def test_score_exact_case_insensitive(self):
        for q in self.corp.questions:
            self.assertTrue(self.corp.score_exact(q.gold.lower(), q))
            self.assertTrue(self.corp.score_exact(q.gold.upper(), q))

    def test_chunks_match_sections(self):
        self.assertEqual(self.corp.chunks(), self.corp.sections)


if __name__ == "__main__":
    unittest.main()
