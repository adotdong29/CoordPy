"""W108 / COO-9 — APPS BACKUP-lane scaffolding tests.

Locks the APPS call-based loader + executor machinery (the structural-pivot
backup; RUNBOOK_W108 § 6).  APPS is NOT the active battlefield (LiveCodeBench
passed real-data soundness), but the scaffolding must be REAL so a pivot is
possible in-milestone — these tests prove the machinery is clean offline.
"""
from __future__ import annotations

import json

import pytest

from coordpy.apps_loader_v1 import (
    AppsCorpusError,
    assert_apps_pinned,
    is_call_based_row,
    parse_call_based_subset,
    validate_row_schema,
)
from coordpy.apps_executor_v1 import run_apps_executor_v1

GOLD = "def add(a, b):\n    return a + b\n"
GOLD_SOLUTION = (
    "class Solution:\n    def add(self, a, b):\n        return a + b\n")


def _call_based_row():
    return {
        "problem_id": "apps/1", "question": "implement add", "starter_code": "",
        "input_output": json.dumps(
            {"fn_name": "add", "inputs": [[2, 3], [10, 20]],
             "outputs": [5, [30]]}),  # 2nd output is the 1-element wrapper
        "difficulty": "introductory", "url": "http://x",
    }


def test_loader_parses_call_based_input_output_string():
    (p,) = parse_call_based_subset(
        (json.dumps(_call_based_row()) + "\n").encode("utf-8"))
    assert p.fn_name == "add"
    assert len(p.tests) == 2
    assert p.tests[0].args_repr == "[2, 3]"


def test_loader_filters_stdin_rows():
    stdin_row = {
        "problem_id": "apps/2", "question": "read ints",
        "input_output": json.dumps({"inputs": ["1 2\n"], "outputs": ["3\n"]}),
    }
    assert is_call_based_row(stdin_row) is False
    assert parse_call_based_subset(
        (json.dumps(stdin_row) + "\n").encode("utf-8")) == ()


def test_loader_refuses_missing_required_field():
    bad = {"problem_id": "apps/3"}  # no question / input_output
    ok, _ = validate_row_schema(bad)
    assert ok is False
    with pytest.raises(AppsCorpusError):
        parse_call_based_subset((json.dumps(bad) + "\n").encode("utf-8"))


def test_loader_refuses_unpinned():
    with pytest.raises(AppsCorpusError):
        assert_apps_pinned(expected_sha256=None)


def test_executor_gold_toplevel_and_solution_pass():
    tests = [{"args": json.dumps([2, 3]), "output": json.dumps(5)},
             {"args": json.dumps([-1, 1]), "output": json.dumps([0])}]
    for code in (GOLD, GOLD_SOLUTION):
        r = run_apps_executor_v1(
            problem_id="t", func_name="add", tests=tests, candidate_code=code)
        assert r.passed is True
        assert r.returncode == 0


def test_executor_output_wrapper_tolerance():
    """APPS often wraps a single expected return as [expected]; both forms
    must match a correct candidate."""
    tests = [{"args": json.dumps([2, 3]), "output": json.dumps([5])}]
    r = run_apps_executor_v1(
        problem_id="t", func_name="add", tests=tests, candidate_code=GOLD)
    assert r.passed is True


def test_executor_wrong_fails_and_empty_fn_is_entry_not_found():
    tests = [{"args": json.dumps([2, 3]), "output": json.dumps(5)}]
    wrong = run_apps_executor_v1(
        problem_id="t", func_name="add", tests=tests,
        candidate_code="def add(a, b):\n    return a - b\n")
    assert wrong.passed is False and wrong.returncode == 1
    no_entry = run_apps_executor_v1(
        problem_id="t", func_name="", tests=tests, candidate_code=GOLD)
    assert no_entry.passed is False and no_entry.returncode == 3
