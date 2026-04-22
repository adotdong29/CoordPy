"""Tests for the Phase-24 semantic patterns added to CodeQueryPlanner.

These verify:
  (i)   the planner recognises the natural-language shape of each
        semantic question and emits a plan with the expected `pattern`.
  (ii)  the emitted plan, when executed against a small hand-built
        ledger whose handles carry the semantic metadata fields,
        returns the right count / list for each predicate.
  (iii) unmatched questions still fall through to the Phase-22/23
        patterns or return `plan=None`.
"""

from __future__ import annotations

import textwrap
import unittest

from vision_mvp.core.code_index import CodeMetadata, extract_metadata
from vision_mvp.core.code_planner import CodeQueryPlanner
from vision_mvp.core.context_ledger import ContextLedger, hash_embedding


# =============================================================================
# Helpers
# =============================================================================


def _mk_ledger(files: dict[str, str]) -> ContextLedger:
    """Build an in-memory ledger from `{filename: source}` pairs, running
    `extract_metadata` on each so every handle carries real Phase-24
    semantic tuples."""
    ledger = ContextLedger(
        embed_dim=32, embed_fn=lambda t: hash_embedding(t, dim=32),
        max_artifacts=1000,
    )
    for name, src in files.items():
        md = extract_metadata(f"/tmp/{name}", textwrap.dedent(src), root="/tmp")
        ledger.put(textwrap.dedent(src), metadata=md.as_dict())
    return ledger


def _run_plan(planner: CodeQueryPlanner, ledger: ContextLedger,
              question: str):
    r = planner.plan(question)
    if r.plan is None:
        return r, None
    stage, _trace = r.plan.execute(ledger)
    ans = r.plan.render(stage)
    return r, ans


# =============================================================================
# Pattern recognition
# =============================================================================


class TestSemanticPatternRecognition(unittest.TestCase):
    def setUp(self) -> None:
        self.planner = CodeQueryPlanner()

    def test_count_may_raise_recognised(self) -> None:
        r = self.planner.plan("How many functions may raise?")
        self.assertEqual(r.pattern, "code_count_may_raise")

    def test_count_recursive_recognised(self) -> None:
        r = self.planner.plan("How many recursive functions are defined?")
        self.assertEqual(r.pattern, "code_count_is_recursive")

    def test_count_may_write_global_recognised(self) -> None:
        r = self.planner.plan("How many functions may write to global state?")
        self.assertEqual(r.pattern, "code_count_may_write_global")

    def test_count_subprocess_recognised(self) -> None:
        r = self.planner.plan("How many functions call subprocess?")
        self.assertEqual(r.pattern, "code_count_calls_subprocess")

    def test_count_filesystem_recognised(self) -> None:
        r = self.planner.plan("How many functions touch the filesystem?")
        self.assertEqual(r.pattern, "code_count_calls_filesystem")

    def test_count_network_recognised(self) -> None:
        r = self.planner.plan("How many functions make network calls?")
        self.assertEqual(r.pattern, "code_count_calls_network")

    def test_count_external_io_recognised(self) -> None:
        r = self.planner.plan(
            "How many functions have external side effects?")
        self.assertEqual(r.pattern, "code_count_calls_external_io")

    def test_list_may_raise_recognised(self) -> None:
        r = self.planner.plan("List functions that may raise exceptions.")
        self.assertEqual(r.pattern, "code_list_may_raise")

    def test_list_recursive_recognised(self) -> None:
        r = self.planner.plan("Which functions are recursive?")
        self.assertEqual(r.pattern, "code_list_is_recursive")

    def test_list_filesystem_recognised(self) -> None:
        r = self.planner.plan("List functions that touch the filesystem.")
        self.assertEqual(r.pattern, "code_list_calls_filesystem")

    def test_list_subprocess_recognised(self) -> None:
        r = self.planner.plan("List functions that call subprocess.")
        self.assertEqual(r.pattern, "code_list_calls_subprocess")

    def test_list_network_recognised(self) -> None:
        r = self.planner.plan("List functions that make HTTP calls.")
        self.assertEqual(r.pattern, "code_list_calls_network")

    def test_phase22_patterns_not_hijacked(self) -> None:
        # The generic "how many functions" still routes to the
        # Phase-22 count pattern.
        r = self.planner.plan(
            "How many functions are defined in total in the corpus?")
        self.assertEqual(r.pattern, "code_count_functions_total")

    def test_unmatched_returns_none_plan(self) -> None:
        r = self.planner.plan("explain the architecture of this file")
        self.assertIsNone(r.plan)


# =============================================================================
# Execution against a small ledger
# =============================================================================


_FILES_FOR_EXEC = {
    "a.py": """
        import subprocess
        def alpha(x):
            if x < 0:
                raise ValueError("bad")
            return x
        def beta(cmd):
            subprocess.run(cmd)
        def gamma(n):
            if n <= 0:
                return 0
            return gamma(n - 1)
    """,
    "b.py": """
        import requests
        STATE = []
        def fetch(u):
            return requests.get(u)
        def record(x):
            STATE.append(x)
        def read_file(p):
            with open(p) as f:
                return f.read()
    """,
    "c.py": """
        # purely pure module, no semantics
        def add(a, b):
            return a + b
        def mul(a, b):
            return a * b
    """,
}


class TestSemanticPlanExecution(unittest.TestCase):
    def setUp(self) -> None:
        self.ledger = _mk_ledger(_FILES_FOR_EXEC)
        self.planner = CodeQueryPlanner()

    def test_count_may_raise_returns_one(self) -> None:
        r, ans = _run_plan(self.planner, self.ledger,
                           "How many functions may raise?")
        self.assertEqual(r.pattern, "code_count_may_raise")
        self.assertEqual(ans, "1")

    def test_count_recursive_returns_one(self) -> None:
        _r, ans = _run_plan(self.planner, self.ledger,
                            "How many functions are recursive?")
        self.assertEqual(ans, "1")

    def test_count_may_write_global_returns_one(self) -> None:
        _r, ans = _run_plan(self.planner, self.ledger,
                            "How many functions may write to globals?")
        self.assertEqual(ans, "1")

    def test_count_subprocess_returns_one(self) -> None:
        _r, ans = _run_plan(self.planner, self.ledger,
                            "How many functions call subprocess?")
        self.assertEqual(ans, "1")

    def test_count_filesystem_returns_one(self) -> None:
        # Only `read_file` calls open; subprocess.run is NOT filesystem.
        _r, ans = _run_plan(self.planner, self.ledger,
                            "How many functions touch the filesystem?")
        self.assertEqual(ans, "1")

    def test_count_network_returns_one(self) -> None:
        _r, ans = _run_plan(self.planner, self.ledger,
                            "How many functions make network calls?")
        self.assertEqual(ans, "1")

    def test_count_external_io_is_union(self) -> None:
        # 1 subprocess + 1 filesystem + 1 network = 3 io-touching fns
        # (all distinct functions in this corpus).
        _r, ans = _run_plan(self.planner, self.ledger,
                            "How many functions have external side effects?")
        self.assertEqual(ans, "3")

    def test_list_recursive_contains_gamma(self) -> None:
        _r, ans = _run_plan(self.planner, self.ledger,
                            "List recursive functions.")
        self.assertIn("gamma", ans)

    def test_list_may_raise_contains_alpha(self) -> None:
        _r, ans = _run_plan(self.planner, self.ledger,
                            "List functions that may raise exceptions.")
        self.assertIn("alpha", ans)

    def test_list_calls_subprocess_contains_beta(self) -> None:
        _r, ans = _run_plan(self.planner, self.ledger,
                            "List functions that call subprocess.")
        self.assertIn("beta", ans)

    def test_list_calls_filesystem_contains_read_file(self) -> None:
        _r, ans = _run_plan(self.planner, self.ledger,
                            "Which functions touch the filesystem?")
        self.assertIn("read_file", ans)

    def test_list_calls_network_contains_fetch(self) -> None:
        _r, ans = _run_plan(self.planner, self.ledger,
                            "List functions that make network calls.")
        self.assertIn("fetch", ans)

    def test_pure_module_is_silent_on_semantic_questions(self) -> None:
        # With only c.py we'd have 0/0 semantic flags; here we just
        # confirm that a pure-module function (`add`) appears in
        # NO semantic listing.
        _r, ans = _run_plan(self.planner, self.ledger,
                            "List functions that may raise exceptions.")
        self.assertNotIn(".add", ans)
        self.assertNotIn(".mul", ans)


if __name__ == "__main__":
    unittest.main()
