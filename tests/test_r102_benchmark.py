"""R-102 benchmark family smoke test.

Tests that each H1-H12 family meets its pre-committed bar.
Runs each family across seeds (1, 2, 3) and checks the mean
against the threshold.
"""

from __future__ import annotations

import pytest

from coordpy.r102_benchmark import (
    R102_BASELINE_ARM,
    R102_W52_ARM,
    run_family,
)


def _mean(comparison, arm: str) -> float:
    a = comparison.get(arm)
    assert a is not None, f"missing arm {arm}"
    return a.mean


def test_h1_trivial_passthrough() -> None:
    cmp = run_family(
        "family_trivial_w52_passthrough", seeds=(1, 2, 3))
    assert _mean(cmp, R102_W52_ARM) == 1.0


def test_h2_persistent_v4_long_horizon_gain() -> None:
    cmp = run_family(
        "family_persistent_v4_long_horizon_gain",
        seeds=(1, 2, 3))
    delta = (
        _mean(cmp, R102_W52_ARM)
        - _mean(cmp, R102_BASELINE_ARM))
    assert delta >= 0.15


def test_h3_multi_hop_quad_transitivity() -> None:
    cmp = run_family(
        "family_multi_hop_quad_transitivity", seeds=(1, 2, 3))
    assert _mean(cmp, R102_W52_ARM) >= 0.70


def test_h4_disagreement_weighted_arbitration() -> None:
    cmp = run_family(
        "family_disagreement_weighted_arbitration",
        seeds=(1, 2, 3))
    delta = (
        _mean(cmp, R102_W52_ARM)
        - _mean(cmp, R102_BASELINE_ARM))
    assert delta >= 0.05


def test_h5_deep_stack_v3_depth_strict_gain() -> None:
    cmp = run_family(
        "family_deep_stack_v3_depth_strict_gain",
        seeds=(1, 2, 3))
    w52_mean = _mean(cmp, R102_W52_ARM)
    delta = w52_mean - _mean(cmp, R102_BASELINE_ARM)
    assert w52_mean >= 0.55
    assert delta >= -0.05


def test_h6_role_graph_transfer_gain() -> None:
    cmp = run_family(
        "family_role_graph_transfer_gain", seeds=(1, 2, 3))
    delta = (
        _mean(cmp, R102_W52_ARM)
        - _mean(cmp, R102_BASELINE_ARM))
    assert delta >= 0.05


def test_h7_transcript_vs_shared_state() -> None:
    cmp = run_family(
        "family_transcript_vs_shared_state", seeds=(1, 2, 3))
    delta = (
        _mean(cmp, R102_W52_ARM)
        - _mean(cmp, R102_BASELINE_ARM))
    assert delta >= 0.10


def test_h8_multi_hop_realism_probe_skip_ok() -> None:
    cmp = run_family(
        "family_multi_hop_realism_probe", seeds=(1, 2, 3))
    assert _mean(cmp, R102_BASELINE_ARM) == 1.0


def test_h9_envelope_verifier() -> None:
    cmp = run_family(
        "family_w52_envelope_verifier", seeds=(1, 2, 3))
    assert _mean(cmp, R102_W52_ARM) == 1.0


def test_h10_replay_determinism() -> None:
    cmp = run_family(
        "family_w52_replay_determinism", seeds=(1, 2, 3))
    assert _mean(cmp, R102_W52_ARM) == 1.0


def test_h11_multi_hop_translator_compromise_cap() -> None:
    cmp = run_family(
        "family_multi_hop_translator_compromise_cap",
        seeds=(1, 2, 3))
    assert _mean(cmp, R102_W52_ARM) >= 0.40


def test_h12_role_graph_distribution_cap() -> None:
    cmp = run_family(
        "family_role_graph_distribution_cap", seeds=(1, 2, 3))
    assert _mean(cmp, R102_W52_ARM) >= 0.60
