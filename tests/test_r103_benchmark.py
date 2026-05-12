"""R-103 benchmark family smoke test.

Tests that each H13-H22 family meets its pre-committed bar.
"""

from __future__ import annotations

import pytest

from coordpy.r103_benchmark import (
    R103_BASELINE_ARM,
    R103_W52_ARM,
    run_family,
)


def _mean(comparison, arm: str) -> float:
    a = comparison.get(arm)
    assert a is not None, f"missing arm {arm}"
    return a.mean


def test_h13_long_horizon_v4_20turn() -> None:
    cmp = run_family(
        "family_long_horizon_v4_retention_20turn",
        seeds=(1, 2, 3))
    w52_mean = _mean(cmp, R103_W52_ARM)
    delta = w52_mean - _mean(cmp, R103_BASELINE_ARM)
    assert w52_mean >= 0.40
    assert delta >= 0.15


def test_h14_long_horizon_v4_24turn_stretch() -> None:
    cmp = run_family(
        "family_long_horizon_v4_retention_24turn_stretch",
        seeds=(1, 2, 3))
    assert _mean(cmp, R103_W52_ARM) >= 0.25


def test_h15_reconstruction_v4_k8() -> None:
    cmp = run_family(
        "family_reconstruction_v4_recovers_t_minus_8",
        seeds=(1, 2, 3))
    assert _mean(cmp, R103_W52_ARM) <= 0.55


def test_h16_reconstruction_v4_k12_stretch() -> None:
    cmp = run_family(
        "family_reconstruction_v4_k12_stretch",
        seeds=(1, 2, 3))
    assert _mean(cmp, R103_W52_ARM) <= 0.70


def test_h17_quantised_14_bits() -> None:
    cmp = run_family(
        "family_quantised_compression_14bits",
        seeds=(1, 2, 3))
    assert _mean(cmp, R103_W52_ARM) >= 14.0


def test_h18_quantised_degradation_curve_min_bits() -> None:
    cmp = run_family(
        "family_quantised_degradation_curve",
        seeds=(1, 2, 3))
    assert _mean(cmp, R103_W52_ARM) >= 5.0


def test_h19_branch_cycle_memory_v2_merge_gain() -> None:
    cmp = run_family(
        "family_branch_cycle_memory_v2_merge_gain",
        seeds=(1, 2, 3))
    delta = (
        _mean(cmp, R103_W52_ARM)
        - _mean(cmp, R103_BASELINE_ARM))
    assert delta >= 0.10


def test_h20_w52_distribution_cap() -> None:
    cmp = run_family(
        "family_w52_distribution_cap", seeds=(1, 2, 3))
    assert _mean(cmp, R103_W52_ARM) >= 0.70


def test_h21_deep_stack_v3_overdepth_cap() -> None:
    cmp = run_family(
        "family_deep_stack_v3_overdepth_cap",
        seeds=(1, 2, 3))
    # L=8 - L=6 ≤ +0.05 (overdepth doesn't strictly improve)
    assert _mean(cmp, R103_W52_ARM) <= 0.05


def test_h22_quantised_rate_floor_falsifier() -> None:
    cmp = run_family(
        "family_quantised_rate_floor_falsifier",
        seeds=(1, 2, 3))
    assert _mean(cmp, R103_W52_ARM) == 1.0
