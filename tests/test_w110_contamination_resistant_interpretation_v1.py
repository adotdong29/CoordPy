"""W110 — Lane β contamination-resistant interpretation rule tests."""
from __future__ import annotations

from coordpy.contamination_resistant_interpretation_v1 import (
    evaluate_phase2_gates_v1,
    interpret_second_resistant_result_v1,
)


def _gates(*, a0, a1, b, b_not_worse, invoked, rescued, n=30):
    return evaluate_phase2_gates_v1(
        n_problems=n, a0_pass_rate=a0, a1_pass_rate=a1, b_pass_rate=b,
        per_problem_b_not_worse_count=b_not_worse,
        reflexion_invoked_count=invoked, reflexion_rescued_count=rescued,
        slice_pre_committed=True, budget_byte_exact=True,
        audit_chain_ok=True, executor_clean=True)


def test_gate_evaluator_pass_mechanism_driven():
    # +16.67pp margin, MLB-1 50%, MLB-2 50% -> PASS_MECHANISM_DRIVEN
    r = _gates(a0=0.50, a1=0.60, b=0.7667, b_not_worse=30,
               invoked=15, rescued=8)
    assert r.gates["G3_b_gt_a1"] and r.gates["G4_margin_ge_5pp"]
    assert r.verdict_label == "PASS_MECHANISM_DRIVEN"


def test_gate_evaluator_pass_non_mechanism_driven():
    # 9/9 gates pass but MLB-1 invocation only 7/30 = 23% (< 33%) -> NON
    r = _gates(a0=0.7333, a1=0.7333, b=0.90, b_not_worse=30,
               invoked=7, rescued=4)
    assert r.verdict_label == "PASS_NON_MECHANISM_DRIVEN"
    assert r.mlb2_rescue_rate > 0.33 and r.mlb1_invocation_rate < 0.33


def test_gate_evaluator_fail_negative_margin():
    # B < A1 (the W108 LCB shape): -3.33pp -> FAIL
    r = _gates(a0=0.4333, a1=0.6333, b=0.60, b_not_worse=25,
               invoked=16, rescued=4)
    assert not r.gates["G3_b_gt_a1"] and not r.gates["G4_margin_ge_5pp"]
    assert r.verdict_label == "FAIL"


def test_interpretation_fail_strengthens_confound():
    d = interpret_second_resistant_result_v1(
        second_resistant_benchmark="BigCodeBench",
        verdict_label="FAIL", b_minus_a1_pp=-3.33, mlb2_rescue_rate=0.25)
    assert d.confound_direction == "STRENGTHENS"
    assert not d.earns_phase3_retirement_bench
    assert "EXPOSED-specific" in d.boundary_after_w110


def test_interpretation_pass_mech_weakens_confound_earns_phase3():
    d = interpret_second_resistant_result_v1(
        second_resistant_benchmark="BigCodeBench",
        verdict_label="PASS_MECHANISM_DRIVEN", b_minus_a1_pp=12.0,
        mlb2_rescue_rate=0.45)
    assert d.confound_direction == "WEAKENS"
    assert d.earns_phase3_retirement_bench
    assert "LCB-SPECIFIC" in d.boundary_after_w110
    assert "Phase-3" in d.w111_branch


def test_interpretation_pass_non_mech_unchanged():
    d = interpret_second_resistant_result_v1(
        second_resistant_benchmark="BigCodeBench",
        verdict_label="PASS_NON_MECHANISM_DRIVEN", b_minus_a1_pp=10.0,
        mlb2_rescue_rate=0.5)
    assert d.confound_direction == "UNCHANGED"
    assert not d.earns_phase3_retirement_bench


def test_decision_cid_stable():
    d1 = interpret_second_resistant_result_v1(
        second_resistant_benchmark="BigCodeBench", verdict_label="FAIL",
        b_minus_a1_pp=-3.33, mlb2_rescue_rate=0.25)
    d2 = interpret_second_resistant_result_v1(
        second_resistant_benchmark="BigCodeBench", verdict_label="FAIL",
        b_minus_a1_pp=-3.33, mlb2_rescue_rate=0.25)
    assert d1.cid() == d2.cid()
