"""Tests for the code-aware planner — Phase 22."""

from __future__ import annotations

import os
import tempfile
import textwrap
import unittest

import numpy as np

from vision_mvp.core.code_index import CodeIndexer
from vision_mvp.core.code_planner import CodeQueryPlanner
from vision_mvp.core.context_ledger import ContextLedger, hash_embedding
from vision_mvp.core.exact_ops import StageList, StageScalar


def _embed(text: str) -> np.ndarray:
    return hash_embedding(text, dim=32)


def _make_repo() -> tempfile.TemporaryDirectory:
    """Build a tiny in-memory Python repo with known structural facts."""
    tmp = tempfile.TemporaryDirectory()
    root = tmp.name
    files = {
        "a.py": textwrap.dedent("""\
            import os
            def foo(): return 1
            def bar(): pass
            class A: pass
        """),
        "b.py": textwrap.dedent("""\
            import os
            import sys
            from collections import Counter
            def baz() -> None: pass
            class B: pass
            class C: pass
        """),
        "tests/test_x.py": textwrap.dedent("""\
            def test_one(): pass
            def test_two(): pass
        """),
    }
    for rel, content in files.items():
        full = os.path.join(root, rel)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w") as f:
            f.write(content)
    return tmp


class TestCodePlanner(unittest.TestCase):
    def setUp(self):
        self.tmp = _make_repo()
        self.root = self.tmp.name
        self.ledger = ContextLedger(embed_dim=32, embed_fn=_embed)
        CodeIndexer(root=self.root).index_into(self.ledger)
        self.planner = CodeQueryPlanner()

    def tearDown(self):
        self.tmp.cleanup()

    # ---- pattern recognition ----

    def test_count_files_pattern(self):
        r = self.planner.plan("How many Python files are in the corpus?")
        self.assertEqual(r.pattern, "code_count_files")
        self.assertIsNotNone(r.plan)

    def test_count_functions_pattern(self):
        r = self.planner.plan("How many functions are defined in total?")
        self.assertEqual(r.pattern, "code_count_functions_total")

    def test_count_classes_pattern(self):
        r = self.planner.plan("How many classes are defined in the corpus?")
        self.assertEqual(r.pattern, "code_count_classes_total")

    def test_count_test_files_pattern(self):
        r = self.planner.plan("How many test files are there?")
        self.assertEqual(r.pattern, "code_count_test_files")

    def test_distinct_imports_pattern(self):
        r = self.planner.plan("How many distinct modules are imported?")
        self.assertEqual(r.pattern, "code_distinct_imports")

    def test_files_importing_pattern(self):
        r = self.planner.plan("List the Python files importing os.")
        self.assertEqual(r.pattern, "code_files_importing")
        self.assertEqual(r.matched_groups["target"], "os")

    def test_functions_returning_none_pattern(self):
        r = self.planner.plan("List functions in the corpus that return None.")
        self.assertEqual(r.pattern, "code_functions_returning_none")

    def test_top_file_pattern(self):
        r = self.planner.plan("Which file has the most functions defined?")
        self.assertEqual(r.pattern, "code_top_file_by_functions")

    def test_largest_file_pattern(self):
        r = self.planner.plan("What is the largest file by line count?")
        self.assertEqual(r.pattern, "code_largest_file")

    def test_unmatched_falls_through(self):
        r = self.planner.plan("Tell me about software engineering.")
        self.assertIsNone(r.plan)
        self.assertEqual(r.pattern, "(unmatched)")

    # ---- end-to-end execution ----

    def test_count_files_executes(self):
        r = self.planner.plan("How many Python files are in the corpus?")
        result, _ = r.plan.execute(self.ledger)
        self.assertEqual(result.value, 3)   # a.py, b.py, tests/test_x.py

    def test_count_functions_total_executes(self):
        r = self.planner.plan("How many functions are defined in total?")
        result, _ = r.plan.execute(self.ledger)
        # foo, bar (a.py), baz (b.py), test_one, test_two (tests) = 5
        self.assertEqual(result.value, 5)

    def test_count_classes_total_executes(self):
        r = self.planner.plan("How many classes are defined in the corpus?")
        result, _ = r.plan.execute(self.ledger)
        # A, B, C = 3
        self.assertEqual(result.value, 3)

    def test_count_test_files_executes(self):
        r = self.planner.plan("How many test files are there?")
        result, _ = r.plan.execute(self.ledger)
        self.assertEqual(result.value, 1)

    def test_files_importing_os_executes(self):
        r = self.planner.plan("List files importing os.")
        result, _ = r.plan.execute(self.ledger)
        self.assertIsInstance(result, StageList)
        rendered = r.plan.render(result)
        self.assertIn("a.py", rendered)
        self.assertIn("b.py", rendered)

    def test_top_file_by_functions_executes(self):
        r = self.planner.plan("Which file has the most functions?")
        result, _ = r.plan.execute(self.ledger)
        rendered = r.plan.render(result)
        # tests/test_x.py has 2 functions; b.py has 1; a.py has 2.
        # max(a.py=2, b.py=1, test_x.py=2) — tied at 2; deterministic
        # tie-break is the file_path comparison on the underlying dict.
        # We only assert it's one of the 2-function files.
        self.assertTrue(rendered.endswith("a.py") or rendered.endswith("test_x.py"))

    def test_distinct_imports_executes(self):
        r = self.planner.plan("How many distinct modules are imported?")
        result, _ = r.plan.execute(self.ledger)
        # a.py: os
        # b.py: os, sys, collections, collections.Counter
        # test_x.py: ()
        # distinct: {os, sys, collections, collections.Counter} = 4
        self.assertEqual(result.value, 4)

    def test_functions_returning_none_executes(self):
        r = self.planner.plan("List functions in the corpus that return None.")
        result, _ = r.plan.execute(self.ledger)
        # bar (a.py - no value return), baz (b.py - annotated None),
        # test_one and test_two (no value return). 4 total.
        self.assertEqual(len(result.items), 4)

    def test_planner_is_llm_free(self):
        before = self.ledger.stats_dict()
        r = self.planner.plan("How many functions are defined in total?")
        result, _ = r.plan.execute(self.ledger)
        after = self.ledger.stats_dict()
        # No fetches and no embedding searches in the operator pipeline.
        self.assertEqual(after["n_fetch"], before["n_fetch"])
        self.assertEqual(after["n_search"], before["n_search"])


class TestPhase23Patterns(unittest.TestCase):
    """Phase-23 planner additions."""

    def setUp(self):
        self.tmp = _make_repo()
        self.root = self.tmp.name
        self.ledger = ContextLedger(embed_dim=32, embed_fn=_embed)
        CodeIndexer(root=self.root).index_into(self.ledger)
        self.planner = CodeQueryPlanner()

    def tearDown(self):
        self.tmp.cleanup()

    # ---- pattern recognition ----

    def test_count_methods_pattern(self):
        r = self.planner.plan("How many methods are defined in the corpus?")
        self.assertEqual(r.pattern, "code_count_methods_total")

    def test_count_methods_pattern_variations(self):
        for q in ("how many methods are there",
                  "How many methods are defined?"):
            r = self.planner.plan(q)
            self.assertEqual(r.pattern, "code_count_methods_total",
                             f"failed on {q!r}")

    def test_count_methods_does_not_match_return_none(self):
        # This should fall through to another pattern (or unmatched) —
        # NOT count_methods_total, since the intent is different.
        r = self.planner.plan("how many methods return None")
        self.assertNotEqual(r.pattern, "code_count_methods_total")

    def test_count_files_with_docstrings_pattern(self):
        for q in ("How many files have docstrings?",
                  "how many modules have a docstring",
                  "how many files contain docstrings"):
            r = self.planner.plan(q)
            self.assertEqual(r.pattern, "code_count_files_with_docstrings",
                             f"failed on {q!r}")

    def test_most_imported_module_pattern(self):
        for q in ("Which module is imported most often?",
                  "What is the most imported module?",
                  "which import appears most often",
                  "what module is most frequently imported"):
            r = self.planner.plan(q)
            self.assertEqual(r.pattern, "code_most_imported_module",
                             f"failed on {q!r}")

    def test_most_imported_does_not_shadow_files_importing(self):
        r = self.planner.plan("List files importing os.")
        self.assertEqual(r.pattern, "code_files_importing")

    # ---- end-to-end execution ----

    def test_count_methods_executes(self):
        r = self.planner.plan("How many methods are defined?")
        result, _ = r.plan.execute(self.ledger)
        # a.py: 0 methods (class A: pass, no methods)
        # b.py: 0 methods (class B, class C, no methods)
        # tests/test_x.py: no classes
        self.assertIsInstance(result, StageScalar)
        self.assertEqual(result.value, 0)

    def test_count_files_with_docstrings_executes(self):
        r = self.planner.plan("How many files have docstrings?")
        result, _ = r.plan.execute(self.ledger)
        # None of the fixture files have module-level docstrings.
        self.assertEqual(result.value, 0)

    def test_most_imported_module_executes(self):
        r = self.planner.plan("which module is imported most often?")
        result, _ = r.plan.execute(self.ledger)
        rendered = r.plan.render(result)
        # a.py + b.py both import os → os is top (ties with 2 uses)
        self.assertIn("os", rendered)


class TestPhase23PatternsRichRepo(unittest.TestCase):
    """Exercises the Phase-23 patterns against a repo where the answers
    are non-trivial — at least one method, at least one docstring."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        files = {
            "rich_a.py": textwrap.dedent('''\
                """Module A with a docstring."""
                import os
                import sys

                def helper(): pass

                class Thing:
                    def method_one(self): pass
                    def method_two(self): pass
                '''),
            "rich_b.py": textwrap.dedent('''\
                """Another module."""
                import os
                from collections import Counter

                class Store:
                    def put(self, x): pass
                    def get(self, k): return None
                '''),
            "stub.py": "",   # trivial
        }
        for rel, content in files.items():
            with open(os.path.join(self.root, rel), "w") as f:
                f.write(content)

        self.ledger = ContextLedger(embed_dim=32, embed_fn=_embed)
        CodeIndexer(root=self.root).index_into(self.ledger)
        self.planner = CodeQueryPlanner()

    def tearDown(self):
        self.tmp.cleanup()

    def test_count_methods_real(self):
        r = self.planner.plan("How many methods are defined?")
        result, _ = r.plan.execute(self.ledger)
        self.assertEqual(result.value, 4)   # 2 in Thing + 2 in Store

    def test_count_files_with_docstrings_real(self):
        r = self.planner.plan("How many files have docstrings?")
        result, _ = r.plan.execute(self.ledger)
        self.assertEqual(result.value, 2)   # rich_a + rich_b

    def test_most_imported_module_real(self):
        r = self.planner.plan("which module is imported most often?")
        result, _ = r.plan.execute(self.ledger)
        rendered = r.plan.render(result)
        # os appears in both rich_a (import os) and rich_b (import os),
        # so it ties with __future__-style multi-import modules for
        # the top slot, but since we have no __future__ here, os wins.
        self.assertIn("os", rendered)


if __name__ == "__main__":
    unittest.main()
