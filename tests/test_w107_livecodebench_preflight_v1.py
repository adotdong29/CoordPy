"""W107-β — LiveCodeBench next-battlefield preflight scaffolding tests.

Covers the FUNCTIONAL-form executor cleanness (gold top-level PASS,
gold Solution-method PASS, wrong FAIL, infinite-loop TIMEOUT), the
loader schema-shape refuse-on-mismatch guard (the W102 silent-
degeneration discipline), release-pin enforcement, functional-subset
filtering, and the preflight selection-verdict shape (LiveCodeBench
primary / APPS backup; no pivot).  All NIM-free.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from coordpy.livecodebench_executor_v1 import (  # noqa: E402
    W107_LIVECODEBENCH_EXECUTOR_V1_SCHEMA_VERSION,
    run_livecodebench_executor_v1,
)
from coordpy.livecodebench_loader_v1 import (  # noqa: E402
    LIVECODEBENCH_KNOWN_RELEASES,
    LiveCodeBenchCorpusError,
    assert_release_pinned,
    is_functional_row,
    parse_functional_subset,
    validate_row_schema,
)

_TESTS = [
    {"input": json.dumps([2, 3]), "output": json.dumps(5)},
    {"input": json.dumps([-1, 1]), "output": json.dumps(0)},
]


class TestExecutorCleanness(unittest.TestCase):
    def test_gold_toplevel_passes(self):
        r = run_livecodebench_executor_v1(
            question_id="q", func_name="add", tests=_TESTS,
            candidate_code="def add(a, b):\n    return a + b\n")
        self.assertTrue(r.passed)
        self.assertEqual(r.returncode, 0)

    def test_gold_solution_method_passes(self):
        r = run_livecodebench_executor_v1(
            question_id="q", func_name="add", tests=_TESTS,
            candidate_code=(
                "class Solution:\n"
                "    def add(self, a, b):\n"
                "        return a + b\n"))
        self.assertTrue(r.passed)

    def test_wrong_fails(self):
        r = run_livecodebench_executor_v1(
            question_id="q", func_name="add", tests=_TESTS,
            candidate_code="def add(a, b):\n    return a - b\n")
        self.assertFalse(r.passed)
        self.assertEqual(r.returncode, 1)
        self.assertIn("CASE_FAIL", r.stderr_tail)

    def test_missing_entry_fails(self):
        r = run_livecodebench_executor_v1(
            question_id="q", func_name="add", tests=_TESTS,
            candidate_code="def subtract(a, b):\n    return a - b\n")
        self.assertFalse(r.passed)
        self.assertIn("ENTRY_NOT_FOUND", r.stderr_tail)

    def test_infinite_loop_times_out(self):
        r = run_livecodebench_executor_v1(
            question_id="q", func_name="add", tests=_TESTS,
            candidate_code=(
                "def add(a, b):\n    while True:\n        pass\n"),
            timeout_s=1.0, kill_after_s=2.0)
        self.assertTrue(r.timed_out)
        self.assertFalse(r.passed)

    def test_candidate_code_cid_deterministic(self):
        # The result cid() intentionally includes wall_ms (per-execution
        # timing), mirroring humaneval_plus_executor_v1 — so it is NOT
        # stable across runs.  The candidate_code_cid IS the stable
        # content hash and must be deterministic.
        kw = dict(
            question_id="q", func_name="add", tests=_TESTS,
            candidate_code="def add(a, b):\n    return a + b\n")
        self.assertEqual(
            run_livecodebench_executor_v1(**kw).candidate_code_cid,
            run_livecodebench_executor_v1(**kw).candidate_code_cid)

    def test_schema_version_tag(self):
        r = run_livecodebench_executor_v1(
            question_id="q", func_name="add", tests=_TESTS,
            candidate_code="def add(a, b):\n    return a + b\n")
        self.assertEqual(
            r.schema,
            W107_LIVECODEBENCH_EXECUTOR_V1_SCHEMA_VERSION)


class TestLoaderSchemaGuard(unittest.TestCase):
    def _func_row(self):
        return {
            "question_id": "lc/1",
            "question_content": "implement add",
            "starter_code": (
                "class Solution:\n    def add(self, a, b):"),
            "public_test_cases": json.dumps([
                {"input": "[1,2]", "output": "3",
                 "testtype": "functional"}]),
            "metadata": {"func_name": "add"},
        }

    def test_valid_functional_row_accepted(self):
        ok, _ = validate_row_schema(self._func_row())
        self.assertTrue(ok)

    def test_missing_required_field_refused(self):
        row = self._func_row()
        del row["question_content"]
        ok, reason = validate_row_schema(row)
        self.assertFalse(ok)
        self.assertIn("question_content", reason)

    def test_stdin_row_is_not_functional(self):
        row = self._func_row()
        row["starter_code"] = ""
        self.assertFalse(is_functional_row(row))
        ok, _ = validate_row_schema(row)
        self.assertTrue(ok)  # valid row, just filtered later

    def test_parse_refuses_on_schema_mismatch(self):
        bad = json.dumps({"question_id": "x",
                          "starter_code": "def f():"})  # no content
        with self.assertRaises(LiveCodeBenchCorpusError):
            parse_functional_subset(bad.encode("utf-8"))

    def test_parse_extracts_functional_only(self):
        func = self._func_row()
        stdin = self._func_row()
        stdin["starter_code"] = ""
        stdin["question_id"] = "lc/2"
        raw = (json.dumps(func) + "\n" + json.dumps(stdin)).encode(
            "utf-8")
        problems = parse_functional_subset(raw)
        self.assertEqual(len(problems), 1)
        self.assertEqual(problems[0].func_name, "add")
        self.assertEqual(len(problems[0].tests), 1)


class TestReleasePin(unittest.TestCase):
    def test_unknown_release_refused(self):
        with self.assertRaises(LiveCodeBenchCorpusError):
            assert_release_pinned(
                release="release_v99", expected_sha256="deadbeef")

    def test_unpinned_release_refused(self):
        with self.assertRaises(LiveCodeBenchCorpusError):
            assert_release_pinned(
                release=LIVECODEBENCH_KNOWN_RELEASES[0],
                expected_sha256=None)

    def test_pinned_release_ok(self):
        # known release + explicit pin => no raise
        assert_release_pinned(
            release=LIVECODEBENCH_KNOWN_RELEASES[-1],
            expected_sha256="0" * 64)


class TestPreflightVerdict(unittest.TestCase):
    def test_verdict_artifact_shape(self):
        path = (
            ROOT / "results" / "w107" / "livecodebench_preflight"
            / "preflight_verdict.json")
        if not path.exists():
            self.skipTest("preflight not yet run")
        v = json.loads(path.read_text())
        self.assertEqual(v["battlefield_lead"], "LiveCodeBench")
        self.assertEqual(v["battlefield_backup"], "APPS")
        self.assertTrue(v["offline_probes_pass"])
        self.assertFalse(v["selection_verdict"]["pivot_triggered"])
        self.assertTrue(
            v["P2_executor_self_test"]["all_pass"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
