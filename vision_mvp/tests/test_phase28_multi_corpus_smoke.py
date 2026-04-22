"""Phase 28 smoke test — multi-corpus runtime calibration runs and
produces structurally valid output on a trivial corpus.

We keep this test small: a single synthetic corpus placed in a
temp directory, one seed, a 10ms budget. The goal is to catch
regressions in the benchmark's aggregation code, NOT to reproduce
the Phase-28 numbers. Full-corpus runs are reproduced via the CLI
commands in ``RESULTS_PHASE28.md``.
"""

from __future__ import annotations

import os
import sys
import tempfile
import textwrap
import unittest

from vision_mvp.core.code_corpus_runtime import (
    calibrate_corpus, summarise_corpus_calibration,
)


PHASE28_SMOKE_PREDICATES = (
    "may_raise", "may_raise_explicit", "may_raise_implicit",
)


class TestPhase28MultiCorpusSmoke(unittest.TestCase):
    def test_multi_corpus_run_returns_rows_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as corpus_a, \
                tempfile.TemporaryDirectory() as corpus_b:
            # Corpus A — one file with a no-arg function that raises
            # implicitly, one that raises explicitly, one benign.
            self._write_corpus(corpus_a, "mod_a.py", """
                def implicit():
                    return 1 / 0

                def explicit():
                    raise ValueError('x')

                def benign():
                    return 0
            """)
            # Corpus B — one benign file.
            self._write_corpus(corpus_b, "mod_b.py", """
                def benign_b():
                    return 1
            """)

            # Make both corpora importable.
            for path in (corpus_a, corpus_b):
                if path not in sys.path:
                    sys.path.insert(0, path)
            try:
                rows_a, cov_a = calibrate_corpus(
                    "a", corpus_a, corpus_package=None,
                    predicates=PHASE28_SMOKE_PREDICATES,
                    seeds=(0,), budget_s=0.1,
                )
                rows_b, cov_b = calibrate_corpus(
                    "b", corpus_b, corpus_package=None,
                    predicates=PHASE28_SMOKE_PREDICATES,
                    seeds=(0,), budget_s=0.1,
                )
            finally:
                for path in (corpus_a, corpus_b):
                    try:
                        sys.path.remove(path)
                    except ValueError:
                        pass

            self.assertGreaterEqual(cov_a.n_total, 3)
            self.assertGreaterEqual(cov_b.n_total, 1)
            self.assertEqual(cov_a.status_ready_no_args, 3)
            self.assertEqual(cov_b.status_ready_no_args, 1)

            metrics_a = summarise_corpus_calibration(
                rows_a, predicates=PHASE28_SMOKE_PREDICATES)
            self.assertIn("may_raise", metrics_a)
            self.assertIn("may_raise_explicit", metrics_a)
            self.assertIn("may_raise_implicit", metrics_a)
            # At least one explicit triggered (the `explicit`
            # function) and one implicit triggered (the `implicit`
            # function).
            self.assertGreaterEqual(
                metrics_a["may_raise_explicit"].n_runtime_true, 1)
            self.assertGreaterEqual(
                metrics_a["may_raise_implicit"].n_runtime_true, 1)

    def _write_corpus(self, root: str, filename: str, source: str) -> None:
        path = os.path.join(root, filename)
        with open(path, "w") as f:
            f.write(textwrap.dedent(source))


if __name__ == "__main__":
    unittest.main()
