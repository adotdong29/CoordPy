"""W93 — preflight harness + failure miner CI tests."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from coordpy.cross_modal_preflight_harness_v1 import (
    W93_CROSS_MODAL_PREFLIGHT_HARNESS_V1_SCHEMA_VERSION,
    gate_budget_accounting,
    gate_hypothesis_written,
    gate_benchmark_justification,
    gate_sidecar_evidence,
    gate_ablation,
    run_preflight,
)
from coordpy.failure_cluster_miner_v1 import (
    W93_FAILURE_CLUSTER_MINER_V1_SCHEMA_VERSION,
    discover_runs,
    mine_all_runs,
)


def test_w93_preflight_schema_version():
    assert (
        W93_CROSS_MODAL_PREFLIGHT_HARNESS_V1_SCHEMA_VERSION
        == "coordpy.cross_modal_preflight_harness_v1.v1")


def test_w93_miner_schema_version():
    assert (
        W93_FAILURE_CLUSTER_MINER_V1_SCHEMA_VERSION
        == "coordpy.failure_cluster_miner_v1.v1")


def test_w93_gate_budget_accounting_pass():
    g = gate_budget_accounting(
        candidate_id="test", n_model_calls_per_problem=5,
        target_K=5)
    assert g.passed
    assert "matches" in g.evidence_summary


def test_w93_gate_budget_accounting_fail():
    g = gate_budget_accounting(
        candidate_id="test", n_model_calls_per_problem=7,
        target_K=5)
    assert not g.passed
    assert "MISMATCH" in g.evidence_summary


def test_w93_gate_hypothesis_written_passes_for_long_text():
    g = gate_hypothesis_written(
        candidate_id="test",
        hypothesis=(
            "This is a sufficiently long hypothesis "
            "that explains why this candidate is expected to "
            "beat A1 at fair K=5 budget with structural "
            "details."))
    assert g.passed


def test_w93_gate_hypothesis_written_fails_for_short_text():
    g = gate_hypothesis_written(
        candidate_id="test", hypothesis="short")
    assert not g.passed


def test_w93_gate_benchmark_justification_pass():
    g = gate_benchmark_justification(
        candidate_id="test",
        chosen_benchmark="MathVista",
        why_better_than_humaneval_visual=(
            "MathVista is a multimodal math reasoning "
            "benchmark where unified VLM at K=5 does NOT "
            "approach ceiling (typically 50-70% pass rate), "
            "leaving room for team-based decomposition."))
    assert g.passed


def test_w93_gate_benchmark_justification_fail_empty():
    g = gate_benchmark_justification(
        candidate_id="test",
        chosen_benchmark="some-benchmark",
        why_better_than_humaneval_visual="")
    assert not g.passed


def test_w93_run_preflight_full_pass_when_all_gates_pass():
    """All 5 gates pass: hypothesis long, evidence True,
    ablation True, budget matches, benchmark justified."""
    def evidence_ok():
        return (True, "ok", {})

    def ablation_ok():
        return (True, "ok", {})

    verdict = run_preflight(
        candidate_id="W93-T-fake-strong",
        candidate_hypothesis=(
            "A hypothesis that is sufficiently long to pass "
            "the length check, mentions structural feature "
            "and explains why it should beat A1 in fair budget "
            "comparison."),
        n_model_calls_per_problem=5,
        target_K=5,
        evidence_check_fn=evidence_ok,
        ablation_check_fn=ablation_ok,
        chosen_benchmark="MathVista",
        why_better=(
            "A sufficiently long justification of why this is "
            "a better battlefield than HumanEval-Visual K=5."))
    assert verdict.overall_passes
    assert len(verdict.gates) == 5
    assert all(g.passed for g in verdict.gates)


def test_w93_run_preflight_kills_on_short_hypothesis():
    """Hypothesis too short → G1 fails → overall_passes False."""
    def evidence_ok():
        return (True, "ok", {})

    def ablation_ok():
        return (True, "ok", {})

    verdict = run_preflight(
        candidate_id="W93-T-short-hypothesis",
        candidate_hypothesis="too short",
        n_model_calls_per_problem=5,
        target_K=5,
        evidence_check_fn=evidence_ok,
        ablation_check_fn=ablation_ok,
        chosen_benchmark="MathVista",
        why_better="long enough justification text " * 5)
    assert not verdict.overall_passes
    # G1 fails
    g1 = verdict.gates[0]
    assert g1.gate_id == "G1_hypothesis_written"
    assert not g1.passed


def test_w93_run_preflight_kills_on_evidence_fail():
    """Evidence check returns False → G2 fails → overall fails."""
    def evidence_fail():
        return (False, "no evidence", {})

    def ablation_ok():
        return (True, "ok", {})

    verdict = run_preflight(
        candidate_id="W93-T-no-evidence",
        candidate_hypothesis=(
            "A sufficiently long hypothesis that explains "
            "the candidate architecture and why it should "
            "beat A1 in fair comparison."),
        n_model_calls_per_problem=5,
        target_K=5,
        evidence_check_fn=evidence_fail,
        ablation_check_fn=ablation_ok,
        chosen_benchmark="MathVista",
        why_better="long enough justification text " * 5)
    assert not verdict.overall_passes


def test_w93_run_preflight_kills_on_budget_mismatch():
    """Budget doesn't match K → G4 fails."""
    def evidence_ok():
        return (True, "ok", {})

    def ablation_ok():
        return (True, "ok", {})

    verdict = run_preflight(
        candidate_id="W93-T-budget-mismatch",
        candidate_hypothesis="A " * 60,
        n_model_calls_per_problem=7,   # mismatch
        target_K=5,
        evidence_check_fn=evidence_ok,
        ablation_check_fn=ablation_ok,
        chosen_benchmark="MathVista",
        why_better="A" * 60)
    assert not verdict.overall_passes


def test_w93_run_preflight_verdict_is_content_addressed():
    """The verdict_cid is a stable content hash of the
    verdict dict."""
    def evidence_ok():
        return (True, "ok", {})

    def ablation_ok():
        return (True, "ok", {})

    v1 = run_preflight(
        candidate_id="W93-T-cid-stable",
        candidate_hypothesis="A " * 60,
        n_model_calls_per_problem=5, target_K=5,
        evidence_check_fn=evidence_ok,
        ablation_check_fn=ablation_ok,
        chosen_benchmark="MathVista",
        why_better="A" * 60)
    v2 = run_preflight(
        candidate_id="W93-T-cid-stable",
        candidate_hypothesis="A " * 60,
        n_model_calls_per_problem=5, target_K=5,
        evidence_check_fn=evidence_ok,
        ablation_check_fn=ablation_ok,
        chosen_benchmark="MathVista",
        why_better="A" * 60)
    assert v1.verdict_cid == v2.verdict_cid
    assert v1.verdict_cid != ""


def test_w93_miner_discovers_existing_runs():
    """Miner discovers at least the W88/W89/W90/W91/W92 bench
    reports under results/."""
    found = discover_runs(ROOT / "results")
    # We expect at least 8 reports (W88 + W89 + W90 + W91 + W92).
    assert len(found) >= 8


def test_w93_miner_runs_without_error():
    """End-to-end miner run produces a valid report."""
    report = mine_all_runs(ROOT / "results")
    assert (
        report["schema"]
        == "coordpy.failure_cluster_miner_v1.v1")
    assert int(report["cross_run_patterns"]["n_runs_total"]) >= 8


def test_w93_modules_explicit_import_only():
    import importlib
    mod_a = importlib.import_module(
        "coordpy.cross_modal_preflight_harness_v1")
    mod_b = importlib.import_module(
        "coordpy.failure_cluster_miner_v1")
    assert hasattr(
        mod_a,
        "W93_CROSS_MODAL_PREFLIGHT_HARNESS_V1_SCHEMA_VERSION")
    assert hasattr(
        mod_b,
        "W93_FAILURE_CLUSTER_MINER_V1_SCHEMA_VERSION")
    import coordpy
    assert not hasattr(coordpy, "run_preflight")
    assert not hasattr(coordpy, "mine_all_runs")
