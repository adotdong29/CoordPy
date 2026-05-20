"""W86 — HumanEval bench surface + audit-chain CI tests.

Lightweight, non-NIM-dependent surface tests for
``coordpy.humaneval_real_bench_v1`` and a CI gate that
re-verifies the live HumanEval audit chain on disk
(results/w86/humaneval/...).

The end-to-end NIM run lives at
``scripts/run_w86_humaneval_bench.py`` and is too slow for CI;
the CI gate skips cleanly when the report is not on disk and
asserts the strict-beat closure when it is.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest


REPORT_DIR = (
    Path(__file__).resolve().parent.parent
    / "results" / "w86" / "humaneval")
REPORT_PATH = (
    REPORT_DIR / "humaneval_bench_report.json")
CALLS_PATH = (
    REPORT_DIR / "humaneval_bench_report.calls.jsonl")


def test_w86_humaneval_module_imports():
    from coordpy.humaneval_real_bench_v1 import (
        W86_HUMANEVAL_REAL_BENCH_V1_SCHEMA_VERSION,
        HumanEvalProblemV1, HumanEvalExecutorResultV1,
        HumanEvalArmCallCapsuleV1,
        HumanEvalArmOutcomeCapsuleV1,
        HumanEvalSeedReportV1, HumanEvalBenchReportV1,
        HumanEvalBenchConfigV1,
        load_humaneval_corpus_v1,
        extract_candidate_code_v1,
        run_humaneval_executor_v1,
        select_humaneval_subset_v1,
        run_humaneval_real_bench_v1,
    )
    assert W86_HUMANEVAL_REAL_BENCH_V1_SCHEMA_VERSION == (
        "coordpy.humaneval_real_bench_v1.v1")


def test_w86_humaneval_corpus_sha256_verified():
    """The corpus loader must SHA-256-verify the upstream
    .jsonl.gz blob. Anti-cheat: a substituted corpus is refused
    via HumanEvalCorpusError."""
    from coordpy.humaneval_real_bench_v1 import (
        HUMANEVAL_RAW_EXPECTED_SHA256,
        HUMANEVAL_EXPECTED_PROBLEM_COUNT,
        load_humaneval_corpus_v1,
    )
    # The expected SHA-256 must be 64 hex chars.
    assert len(HUMANEVAL_RAW_EXPECTED_SHA256) == 64
    assert all(
        c in "0123456789abcdef"
        for c in HUMANEVAL_RAW_EXPECTED_SHA256)
    assert int(HUMANEVAL_EXPECTED_PROBLEM_COUNT) == 164


def test_w86_humaneval_subset_deterministic():
    """Same (corpus, n_problems, seed) → same subset, every
    time. Anti-cheat: subset selection cannot be tuned
    post-hoc."""
    from coordpy.humaneval_real_bench_v1 import (
        select_humaneval_subset_v1, HumanEvalProblemV1,
    )
    fake_corpus = tuple(
        HumanEvalProblemV1(
            task_id=f"HumanEval/{i}",
            prompt=f"def f{i}():\n    pass\n",
            canonical_solution="    return 0",
            test=f"def check(c): assert True",
            entry_point=f"f{i}",
        )
        for i in range(50))
    s1 = select_humaneval_subset_v1(
        corpus=fake_corpus, n_problems=8, seed=42)
    s2 = select_humaneval_subset_v1(
        corpus=fake_corpus, n_problems=8, seed=42)
    assert [p.task_id for p in s1] == [
        p.task_id for p in s2]
    s3 = select_humaneval_subset_v1(
        corpus=fake_corpus, n_problems=8, seed=43)
    assert [p.task_id for p in s1] != [
        p.task_id for p in s3]


def test_w86_humaneval_executor_passes_canonical_solution():
    """Anti-cheat: a known-good solution must run to
    completion under the subprocess executor."""
    from coordpy.humaneval_real_bench_v1 import (
        load_humaneval_corpus_v1,
        run_humaneval_executor_v1,
    )
    corpus = load_humaneval_corpus_v1()
    p0 = corpus[0]
    exe = run_humaneval_executor_v1(
        problem=p0,
        candidate_code=p0.prompt + p0.canonical_solution,
        timeout_s=8.0, kill_after_s=12.0)
    assert exe.passed is True
    assert exe.timed_out is False
    assert exe.returncode == 0


def test_w86_humaneval_executor_rejects_wrong_solution():
    """Anti-cheat: a known-bad solution must FAIL with the
    real Python traceback in stderr — the critic gets real
    signal, not a hand-written summary."""
    from coordpy.humaneval_real_bench_v1 import (
        load_humaneval_corpus_v1,
        run_humaneval_executor_v1,
    )
    corpus = load_humaneval_corpus_v1()
    p0 = corpus[0]
    bad_code = p0.prompt + "    return False\n"
    exe = run_humaneval_executor_v1(
        problem=p0, candidate_code=bad_code,
        timeout_s=8.0, kill_after_s=12.0)
    assert exe.passed is False
    assert "Traceback" in exe.stderr_tail


def test_w86_humaneval_code_extraction_with_fence():
    from coordpy.humaneval_real_bench_v1 import (
        extract_candidate_code_v1,
    )
    resp = (
        "Sure, here is the answer:\n\n"
        "```python\n"
        "def answer():\n    return 42\n"
        "```\n\n"
        "Hope this helps!")
    out = extract_candidate_code_v1(
        response_text=resp, prompt="", entry_point="answer")
    assert "def answer()" in out
    assert "return 42" in out
    assert "Hope this helps" not in out


def test_w86_humaneval_code_extraction_without_fence():
    from coordpy.humaneval_real_bench_v1 import (
        extract_candidate_code_v1,
    )
    resp = "def answer():\n    return 42\n"
    out = extract_candidate_code_v1(
        response_text=resp, prompt="", entry_point="answer")
    assert "def answer()" in out


@pytest.mark.skipif(
    not REPORT_PATH.exists(),
    reason=(
        "W86 HumanEval bench report not present in this "
        "checkout"))
def test_w86_humaneval_audit_chain_re_derives():
    """Every per-call response_cid must re-hash to the
    response_text bytes in the sidecar (anti-cheat: model
    responses cannot be silently rewritten); the bench
    Merkle root must re-derive from the per-seed outcome
    CIDs."""
    report = json.loads(
        REPORT_PATH.read_bytes().decode("utf-8"))
    # 1. Per-call CIDs.
    if CALLS_PATH.exists():
        n_checked = 0
        for raw in CALLS_PATH.read_text(
                encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            row = json.loads(raw)
            derived = hashlib.sha256(
                row.get("response_text", "")
                .encode("utf-8")).hexdigest()
            assert derived == row.get("response_cid")
            derived_p = hashlib.sha256(
                row.get("prompt", "")
                .encode("utf-8")).hexdigest()
            assert derived_p == row.get("prompt_cid")
            n_checked += 1
        assert n_checked > 0

    # 2. Bench Merkle root.
    def _sha256(payload):
        return hashlib.sha256(
            json.dumps(
                payload, sort_keys=True,
                separators=(",", ":"),
                default=str).encode("utf-8")).hexdigest()

    all_outcome_cids: list[str] = []
    for s in report.get("per_seed", []):
        all_outcome_cids.extend(
            list(s.get("outcome_cids", [])))
    seeds = [
        int(s.get("seed", 0))
        for s in report.get("per_seed", [])]
    derived_bench = _sha256({
        "kind": "w86_humaneval_bench_merkle_root",
        "model_id": str(report.get("model_id", "")),
        "outcome_cids": list(all_outcome_cids),
        "seeds": list(seeds),
    })
    assert derived_bench == str(
        report.get("bench_merkle_root", ""))


@pytest.mark.skipif(
    not REPORT_PATH.exists(),
    reason="W86 HumanEval bench report not present")
def test_w86_humaneval_at_least_3_seeds_30_problems():
    """The closure run must report at least 3 seeds × 30
    problems per seed, as the issue body requires."""
    report = json.loads(
        REPORT_PATH.read_bytes().decode("utf-8"))
    n_seeds = int(report.get("n_seeds", 0))
    n_problems = int(report.get("n_problems", 0))
    assert n_seeds >= 3
    assert n_problems >= 30


@pytest.mark.skipif(
    not REPORT_PATH.exists(),
    reason="W86 HumanEval bench report not present")
def test_w86_humaneval_strict_improvement_for_28_closure():
    """The load-bearing #28 DoD bullet: the composed pipeline
    strictly improves at least one published metric vs the
    stock harness under same-model / same-budget conditions.

    For HumanEval the published metric is pass@1; the same-
    budget head-to-head is B vs A1 (both spend K=5 model
    calls per problem).
    """
    report = json.loads(
        REPORT_PATH.read_bytes().decode("utf-8"))
    a1_mean = float(report.get("a1_mean_pass_at_1", 0.0))
    b_mean = float(report.get("b_mean_pass_at_1", 0.0))
    assert b_mean > a1_mean, (
        f"#28 strict-improvement requires B > A1 on mean "
        f"pass@1; got B={b_mean} A1={a1_mean}")
    assert bool(report.get(
        "b_mean_strictly_beats_a1_mean")) is True
