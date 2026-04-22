"""Tests for the Phase-25 interprocedural patterns added to
`CodeQueryPlanner`.

These pin three kinds of behaviour:

  (i)   the planner recognises the natural-language shape of each
        interprocedural question and emits a plan tagged
        `code_count_<topic>` / `code_list_<topic>`.
  (ii)  intraprocedural phrasings ("how many functions call
        subprocess") continue to route to the Phase-24 patterns —
        no regression.
  (iii) the emitted plan, when executed against a small built
        ledger whose metadata carries Phase-25 trans-tuples,
        returns the right aggregate count / listing.

The interprocedural triggers all require a *transitivity marker*
("transitively" / "indirectly" / "through a helper" / "cycle" /
"mutual") so the Phase-24 phrasings stay disambiguated. Special
cases: `participates_in_cycle` fires for "mutual recursion" / "call
cycle"; `has_unresolved_callees` fires for "unresolved helpers"
phrasings.
"""

from __future__ import annotations

import os
import tempfile
import textwrap
import unittest

from vision_mvp.core.code_index import CodeIndexer
from vision_mvp.core.code_planner import CodeQueryPlanner
from vision_mvp.core.context_ledger import ContextLedger, hash_embedding


def _build_ledger(files: dict[str, str]) -> ContextLedger:
    """Build a ledger by actually running CodeIndexer over a temp
    directory — so the Phase-25 post-pass actually fires."""
    with tempfile.TemporaryDirectory() as root:
        for name, src in files.items():
            with open(os.path.join(root, name), "w") as f:
                f.write(textwrap.dedent(src))
        indexer = CodeIndexer(root=root)
        ledger = ContextLedger(
            embed_dim=32, embed_fn=lambda t: hash_embedding(t, dim=32),
            max_artifacts=1000,
        )
        indexer.index_into(ledger)
        return ledger


def _run(planner: CodeQueryPlanner, ledger: ContextLedger, q: str):
    r = planner.plan(q)
    if r.plan is None:
        return r, None
    stage, _ = r.plan.execute(ledger)
    return r, r.plan.render(stage)


# =============================================================================
# Pattern recognition
# =============================================================================


class TestInterprocPatternRecognition(unittest.TestCase):
    def setUp(self) -> None:
        self.p = CodeQueryPlanner()

    def test_trans_may_raise_count(self) -> None:
        r = self.p.plan(
            "How many functions may transitively raise an exception "
            "through a helper?")
        self.assertEqual(r.pattern, "code_count_trans_may_raise")

    def test_trans_may_raise_list(self) -> None:
        r = self.p.plan(
            "List functions that may transitively raise an exception.")
        self.assertEqual(r.pattern, "code_list_trans_may_raise")

    def test_trans_subprocess_count(self) -> None:
        r = self.p.plan(
            "How many functions transitively invoke subprocess "
            "through a helper?")
        self.assertEqual(r.pattern, "code_count_trans_calls_subprocess")

    def test_trans_subprocess_list(self) -> None:
        r = self.p.plan(
            "List functions that transitively invoke subprocess.")
        self.assertEqual(r.pattern, "code_list_trans_calls_subprocess")

    def test_trans_filesystem_count(self) -> None:
        r = self.p.plan(
            "How many functions transitively touch the filesystem?")
        self.assertEqual(r.pattern, "code_count_trans_calls_filesystem")

    def test_trans_network_count(self) -> None:
        r = self.p.plan(
            "How many functions transitively make network calls?")
        self.assertEqual(r.pattern, "code_count_trans_calls_network")

    def test_trans_may_write_global_count(self) -> None:
        r = self.p.plan(
            "How many functions transitively mutate module globals "
            "through a helper?")
        self.assertEqual(r.pattern, "code_count_trans_may_write_global")

    def test_trans_external_io_count(self) -> None:
        r = self.p.plan(
            "How many functions have transitive external side effects?")
        self.assertEqual(r.pattern, "code_count_trans_calls_external_io")

    def test_participates_in_cycle_count(self) -> None:
        r = self.p.plan(
            "How many functions participate in a recursion cycle?")
        self.assertEqual(r.pattern, "code_count_participates_in_cycle")

    def test_participates_in_cycle_list_via_mutual(self) -> None:
        r = self.p.plan("Which functions are in a mutual-recursion cycle?")
        self.assertEqual(r.pattern, "code_list_participates_in_cycle")

    def test_has_unresolved_callees_count(self) -> None:
        r = self.p.plan("How many functions call into unresolved helpers?")
        self.assertEqual(r.pattern, "code_count_has_unresolved_callees")

    def test_has_unresolved_callees_list(self) -> None:
        r = self.p.plan("List functions with unresolved callees.")
        self.assertEqual(r.pattern, "code_list_has_unresolved_callees")


# =============================================================================
# Phase-24 intraprocedural phrasings still route to Phase-24 (no regression)
# =============================================================================


class TestNoPhase24Regression(unittest.TestCase):
    def setUp(self) -> None:
        self.p = CodeQueryPlanner()

    def test_intra_subprocess_still_routes_to_phase_24(self) -> None:
        r = self.p.plan("How many functions call subprocess?")
        self.assertEqual(r.pattern, "code_count_calls_subprocess")

    def test_intra_recursive_still_routes_to_phase_24(self) -> None:
        r = self.p.plan("How many functions are recursive?")
        self.assertEqual(r.pattern, "code_count_is_recursive")

    def test_intra_may_raise_still_routes_to_phase_24(self) -> None:
        r = self.p.plan("How many functions may raise?")
        self.assertEqual(r.pattern, "code_count_may_raise")

    def test_intra_filesystem_still_routes_to_phase_24(self) -> None:
        r = self.p.plan("How many functions touch the filesystem?")
        self.assertEqual(r.pattern, "code_count_calls_filesystem")


# =============================================================================
# Execution correctness — end-to-end from source → planner → count/list
# =============================================================================


class TestInterprocExecution(unittest.TestCase):
    """End-to-end: build a real ledger via CodeIndexer, ask the
    planner, verify the rendered answer matches the gold that the
    corpus's own aggregates compute. `direct-exact` with no LLM in
    the loop."""

    def setUp(self) -> None:
        self.p = CodeQueryPlanner()
        self.ledger = _build_ledger({
            "helpers.py": """
                import subprocess
                def run_cmd(c):
                    return subprocess.run(c, shell=True)
            """,
            "callers.py": """
                from helpers import run_cmd
                def wrapper(c):
                    return run_cmd(c)
                def outer(c):
                    return wrapper(c)
                def f(x):
                    return g(x)
                def g(x):
                    return f(x)
                def pure(x):
                    return x + 1
            """,
        })

    def test_count_trans_subprocess_is_three(self) -> None:
        """run_cmd (direct) + wrapper (1 hop) + outer (2 hops) = 3."""
        _, ans = _run(self.p, self.ledger,
                      "How many functions transitively invoke "
                      "subprocess through a helper?")
        self.assertEqual(ans, "3")

    def test_list_trans_subprocess_contains_wrapper_and_outer(self) -> None:
        _, ans = _run(self.p, self.ledger,
                      "List functions that transitively invoke subprocess.")
        for name in ("run_cmd", "wrapper", "outer"):
            self.assertIn(name, ans,
                          f"{name} should appear in the transitive list")

    def test_count_participates_in_cycle_is_two(self) -> None:
        """f ↔ g is a 2-SCC → 2 functions."""
        _, ans = _run(self.p, self.ledger,
                      "How many functions participate in a recursion cycle?")
        self.assertEqual(ans, "2")

    def test_list_participates_in_cycle_contains_f_and_g(self) -> None:
        _, ans = _run(self.p, self.ledger,
                      "Which functions are in a mutual-recursion cycle?")
        self.assertIn("f", ans)
        self.assertIn("g", ans)

    def test_pure_function_is_not_in_any_trans_list(self) -> None:
        _, ans = _run(self.p, self.ledger,
                      "List functions that transitively invoke subprocess.")
        self.assertNotIn(".pure", ans, "`pure` has no IO transitively")


class TestUnresolvedExecution(unittest.TestCase):
    def test_count_unresolved_callees_reports_caller(self) -> None:
        """A function that calls a name outside the corpus must raise
        the `has_unresolved_callees` count by 1."""
        p = CodeQueryPlanner()
        ledger = _build_ledger({
            "caller.py": """
                def caller(x):
                    return external_lib.query(x)
                def pure(x):
                    return x + 1
            """,
        })
        _, ans = _run(p, ledger,
                      "How many functions call into unresolved helpers?")
        self.assertEqual(ans, "1")


# =============================================================================
# Phase-25 count strictly ≥ Phase-24 count when trivially true
# =============================================================================


class TestTransSupersetsIntraAtCorpusLevel(unittest.TestCase):
    def test_trans_subprocess_count_at_least_intra(self) -> None:
        """Corpus-level: n_trans_calls_subprocess ≥ n_calls_subprocess."""
        p = CodeQueryPlanner()
        ledger = _build_ledger({
            "helpers.py": """
                import subprocess
                def run_cmd(c):
                    subprocess.run(c, shell=True)
            """,
            "callers.py": """
                from helpers import run_cmd
                def w(c):
                    run_cmd(c)
            """,
        })
        _, intra = _run(p, ledger, "How many functions call subprocess?")
        _, trans = _run(p, ledger,
                        "How many functions transitively invoke "
                        "subprocess through a helper?")
        self.assertGreaterEqual(int(trans), int(intra),
                                "trans_calls_subprocess ⊇ calls_subprocess")


if __name__ == "__main__":
    unittest.main()
