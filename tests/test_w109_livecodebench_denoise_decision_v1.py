"""W109-β — LiveCodeBench multi-seed de-noise decision-rule tests.

Locks the falsifiable two-gate rule that decides whether spending more LCB NIM
to de-noise the W108 single-seed FAIL is warranted.  The rule must say NO to
the W108 shape (negative margin + weak mechanism) and YES only to a marginal
POSITIVE miss with a load-bearing mechanism (the W105-Llama-3.1 shape) —
without re-opening the closed rescue-concentrated branch.
"""
from __future__ import annotations

from coordpy.livecodebench_denoise_decision_v1 import (
    W108_LCB_RESULT,
    LcbPhase2ResultV1,
    decide_livecodebench_denoise_v1,
)


def test_w108_result_is_not_warranted():
    d = decide_livecodebench_denoise_v1(W108_LCB_RESULT)
    assert d.warranted is False
    assert d.gate1_marginal_positive_miss is False  # margin negative
    assert d.gate2_mechanism_load_bearing is False  # MLB-2 25% < 33%
    assert abs(d.required_mean_shift_pp - 8.33) < 1e-6
    assert d.recommended_followup is None
    assert "NOT" in d.w110_implication or "APPS" in d.w110_implication


def test_marginal_positive_miss_with_healthy_mechanism_is_warranted():
    # The W105-Llama-3.1 shape: +2.33 pp, MLB-2 healthy.
    r = LcbPhase2ResultV1(b_minus_a1_pp=2.33, mlb2_rescue_rate=0.50,
                          n_seeds=1, n_problems=30, a1_pct=80.0)
    d = decide_livecodebench_denoise_v1(r)
    assert d.gate1_marginal_positive_miss is True
    assert d.gate2_mechanism_load_bearing is True
    assert d.warranted is True
    assert d.recommended_followup is not None
    # forbids cross-class + rescue-concentrated re-runs
    assert "same" in d.recommended_followup["model_class"].lower()
    assert "NOT a rescue-concentrated" in d.recommended_followup["slice"]
    assert d.recommended_followup["n_seeds"] == 3


def test_positive_miss_but_weak_mechanism_not_warranted():
    r = LcbPhase2ResultV1(b_minus_a1_pp=3.0, mlb2_rescue_rate=0.20,
                          n_seeds=1, n_problems=30, a1_pct=70.0)
    d = decide_livecodebench_denoise_v1(r)
    assert d.gate1_marginal_positive_miss is True
    assert d.gate2_mechanism_load_bearing is False
    assert d.warranted is False


def test_negative_margin_with_healthy_mechanism_not_warranted():
    r = LcbPhase2ResultV1(b_minus_a1_pp=-2.0, mlb2_rescue_rate=0.60,
                          n_seeds=1, n_problems=30, a1_pct=65.0)
    d = decide_livecodebench_denoise_v1(r)
    assert d.gate1_marginal_positive_miss is False
    assert d.warranted is False
    assert d.required_mean_shift_pp == 7.0


def test_decision_is_deterministic():
    a = decide_livecodebench_denoise_v1(W108_LCB_RESULT)
    b = decide_livecodebench_denoise_v1(W108_LCB_RESULT)
    assert a.cid() == b.cid()
