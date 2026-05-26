"""W102 — HumanEval+ V1 infrastructure unit tests (NIM-free)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from coordpy import (
    humaneval_plus_executor_v1,
    humaneval_plus_loader_v1,
    humaneval_plus_preflight_v1,
    humaneval_plus_reflexion_bench_v1,
)


# ---------------------------------------------------------------
# Loader
# ---------------------------------------------------------------


def test_loader_schema_version():
    assert (
        humaneval_plus_loader_v1
        .W102_HUMANEVAL_PLUS_LOADER_V1_SCHEMA_VERSION
        == "coordpy.humaneval_plus_loader_v1.v1")


def test_loader_constants_documented():
    assert (
        "huggingface.co/datasets/evalplus/humanevalplus"
        in humaneval_plus_loader_v1
        .HUMANEVAL_PLUS_HF_CANONICAL_URL_V021)
    assert len(
        humaneval_plus_loader_v1
        .HUMANEVAL_PLUS_HF_EXPECTED_SHA256_V021) == 64
    assert (
        humaneval_plus_loader_v1
        .HUMANEVAL_PLUS_EXPECTED_PROBLEM_COUNT == 164)


def test_loader_parse_jsonl_synthetic_too_few_rejects():
    rows = [
        {
            "task_id": f"HumanEval/{i}",
            "prompt": "def f(): pass",
            "canonical_solution": "    return 0",
            "entry_point": "f",
            "test": "def check(c): c()",
        }
        for i in range(10)
    ]
    payload = "\n".join(json.dumps(r) for r in rows).encode("utf-8")
    with pytest.raises(
            humaneval_plus_loader_v1.HumanEvalPlusCorpusError):
        humaneval_plus_loader_v1.parse_humaneval_plus_jsonl(
            payload)


def test_loader_is_cached_false_on_missing(tmp_path):
    path = tmp_path / "nonexistent.jsonl"
    assert not (
        humaneval_plus_loader_v1.is_humaneval_plus_cached(
            cache_path=str(path)))


# ---------------------------------------------------------------
# Executor (lightweight; canonical solutions only)
# ---------------------------------------------------------------


def test_executor_passes_on_canonical_solution_simple():
    p = humaneval_plus_loader_v1.HumanEvalPlusProblemV1(
        task_id="HumanEval/test",
        prompt=(
            "def add(a: int, b: int) -> int:\n"
            "    \"\"\"Add two ints.\"\"\"\n"),
        canonical_solution=(
            "    return a + b\n"),
        test=(
            "def check(candidate):\n"
            "    assert candidate(1, 2) == 3\n"
            "    assert candidate(10, -5) == 5\n"),
        entry_point="add",
    )
    cand = p.prompt + p.canonical_solution
    res = (
        humaneval_plus_executor_v1
        .run_humaneval_plus_executor_v1(
            problem=p, candidate_code=cand))
    assert res.passed
    assert res.returncode == 0


def test_executor_fails_on_buggy_candidate():
    p = humaneval_plus_loader_v1.HumanEvalPlusProblemV1(
        task_id="HumanEval/test",
        prompt=(
            "def add(a: int, b: int) -> int:\n"
            "    \"\"\"Add two ints.\"\"\"\n"),
        canonical_solution="    return a + 3\n",  # bug
        test=(
            "def check(candidate):\n"
            "    assert candidate(1, 2) == 3\n"
            "    assert candidate(10, -5) == 5\n"),
        entry_point="add",
    )
    cand = p.prompt + p.canonical_solution
    res = (
        humaneval_plus_executor_v1
        .run_humaneval_plus_executor_v1(
            problem=p, candidate_code=cand))
    assert not res.passed


# ---------------------------------------------------------------
# Bench subset
# ---------------------------------------------------------------


def test_bench_subset_deterministic():
    corpus = tuple(
        humaneval_plus_loader_v1.HumanEvalPlusProblemV1(
            task_id=f"HumanEval/{i}",
            prompt=f"def f_{i}(): pass\n",
            canonical_solution="    return 0\n",
            test=(
                "def check(c):\n"
                "    assert c() == 0\n"),
            entry_point=f"f_{i}",
        )
        for i in range(50))
    s1 = (
        humaneval_plus_reflexion_bench_v1
        .select_humaneval_plus_subset_v1(
            corpus=corpus, n_problems=10, seed=102_001))
    s2 = (
        humaneval_plus_reflexion_bench_v1
        .select_humaneval_plus_subset_v1(
            corpus=corpus, n_problems=10, seed=102_001))
    s3 = (
        humaneval_plus_reflexion_bench_v1
        .select_humaneval_plus_subset_v1(
            corpus=corpus, n_problems=10, seed=102_002))
    assert s1 == s2
    assert s1 != s3
    assert len(s1) == 10


# ---------------------------------------------------------------
# Preflight probes
# ---------------------------------------------------------------


def test_preflight_p3_residual_estimate():
    p = (
        humaneval_plus_preflight_v1
        .probe_humaneval_plus_a1_residual_v1())
    assert p.passed
    assert (
        p.evidence["a1_predicted_humaneval_plus_pp"] < 90.0)


def test_preflight_p4_decomposition_argument_long_enough():
    p = (
        humaneval_plus_preflight_v1
        .probe_humaneval_plus_decomposition_v1())
    assert p.passed
    assert p.evidence["argument_length_chars"] >= 800


def test_preflight_anti_pattern_guard_clean(tmp_path):
    clean = tmp_path / "clean.py"
    clean.write_text(
        "from coordpy.humaneval_plus_executor_v1 import "
        "run_humaneval_plus_executor_v1\n")
    p = (
        humaneval_plus_preflight_v1
        .probe_humaneval_plus_anti_pattern_guard_v1(
            bench_module_path=clean))
    assert p.passed


def test_preflight_anti_pattern_guard_dirty(tmp_path):
    dirty = tmp_path / "dirty.py"
    dirty.write_text(
        "from coordpy.bounded_window_baseline_v1 import x\n")
    p = (
        humaneval_plus_preflight_v1
        .probe_humaneval_plus_anti_pattern_guard_v1(
            bench_module_path=dirty))
    assert not p.passed


def test_preflight_w89_rescue_prior():
    p = (
        humaneval_plus_preflight_v1
        .probe_humaneval_plus_w89_rescue_prior_v1())
    assert p.passed
    assert (
        p.evidence["w89_rescue_fraction"]
        >= p.evidence["min_required_fraction"])


def test_preflight_run_end_to_end_with_cache(tmp_path):
    if not (
            humaneval_plus_loader_v1
            .is_humaneval_plus_cached()):
        pytest.skip(
            "HumanEval+ cache absent on this machine")
    bench_module = Path(
        humaneval_plus_reflexion_bench_v1.__file__)
    verdict = (
        humaneval_plus_preflight_v1
        .run_humaneval_plus_preflight_v1(
            bench_module_path=bench_module,
            run_executor_self_test=False))
    assert len(verdict.probes) == 7
    # 6 of 7 PASS minimum when executor self-test is skipped
    # (P2 is the skipped one).
    n_pass = sum(1 for p in verdict.probes if p.passed)
    assert n_pass >= 6
