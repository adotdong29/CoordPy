"""Tests for the distributed-summary task and chunking."""
from __future__ import annotations
import unittest

from vision_mvp.tasks.distributed_summary import (
    DistributedSummaryTask, split_into_chunks, ORION_DOCUMENT,
)


class TestChunking(unittest.TestCase):
    def test_chunks_cover_whole_document(self):
        # Union of all chunk words should equal the document's words (up to
        # internal whitespace normalisation).
        chunks = split_into_chunks(ORION_DOCUMENT, n_chunks=8)
        chunk_words = []
        for c in chunks:
            chunk_words.extend(c.split())
        doc_words = ORION_DOCUMENT.split()
        self.assertEqual(sorted(chunk_words), sorted(doc_words))

    def test_chunk_count(self):
        for n in (4, 8, 16):
            chunks = split_into_chunks(ORION_DOCUMENT, n_chunks=n)
            self.assertLessEqual(len(chunks), n)
            self.assertGreater(len(chunks), 0)


class TestScoring(unittest.TestCase):
    def test_perfect_answer_scores_3(self):
        task = DistributedSummaryTask()
        answer = ("The top risks are: vendor concentration around NordAxis, "
                  "stale runbooks across multiple incidents, and slow "
                  "detection-to-mitigation times up to 21 days.")
        score = task.score(answer)
        self.assertEqual(score["n_risks_identified"], 3)

    def test_empty_answer_scores_0(self):
        task = DistributedSummaryTask()
        score = task.score("")
        self.assertEqual(score["n_risks_identified"], 0)

    def test_partial_answer(self):
        task = DistributedSummaryTask()
        ans = "NordAxis supplier concentration is a major problem."
        score = task.score(ans)
        self.assertTrue(score["supplier_concentration"])
        self.assertFalse(score["documentation_gaps"])

    def test_runbook_keyword(self):
        task = DistributedSummaryTask()
        ans = "Documentation runbook issues recur."
        score = task.score(ans)
        self.assertTrue(score["documentation_gaps"])


if __name__ == "__main__":
    unittest.main()
