"""Tests for long_corpus task generator."""
from __future__ import annotations
import unittest

from vision_mvp.tasks.long_corpus import LongCorpusTask


class TestLongCorpus(unittest.TestCase):
    def test_word_count_exceeds_default_context(self):
        t = LongCorpusTask(n_sections=40, seed=13)
        t.generate()
        # 4096 tokens ≈ 3000 words; doc should be way past that
        self.assertGreater(t.word_count, 4000)

    def test_chunks_partition_document(self):
        t = LongCorpusTask(n_sections=20, seed=1)
        t.generate()
        chunks = t.chunk(10)
        total = sum(len(c.split()) for c in chunks)
        self.assertAlmostEqual(total, t.word_count, delta=20)

    def test_scoring_detects_keywords(self):
        t = LongCorpusTask(seed=1)
        t.generate()
        perfect_answer = ("Top risks: NordAxis vendor concentration, "
                          "stale runbooks across incidents, slow "
                          "detection time on multiple events.")
        s = t.score(perfect_answer)
        self.assertEqual(s["n_risks_identified"], 3)

    def test_empty_answer_scores_zero(self):
        t = LongCorpusTask(seed=1)
        t.generate()
        self.assertEqual(t.score("")["n_risks_identified"], 0)

    def test_deterministic_given_seed(self):
        t1 = LongCorpusTask(n_sections=10, seed=99)
        t2 = LongCorpusTask(n_sections=10, seed=99)
        t1.generate()
        t2.generate()
        self.assertEqual(t1.document, t2.document)


if __name__ == "__main__":
    unittest.main()
