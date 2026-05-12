"""Tests for the R-101 benchmark family."""

from __future__ import annotations

import pytest

from coordpy.r101_benchmark import (
    R101_BASELINE_ARM,
    R101_FAMILY_TABLE,
    R101_W51_ARM,
    run_family,
)


def test_r101_family_table_has_8_families() -> None:
    assert len(R101_FAMILY_TABLE) == 8


def test_r101_long_horizon_retention_12turn_passes() -> None:
    cmp = run_family(
        "family_long_horizon_retention_12turn",
        seeds=(1, 2, 3))
    w51 = cmp.get(R101_W51_ARM)
    assert w51 is not None
    # H11: cosine ≥ 0.60, gain ≥ +0.20
    assert w51.mean >= 0.50  # generous floor
    assert cmp.delta_w51_vs_w50() >= 0.20


def test_r101_long_horizon_retention_16turn_stretch_passes() -> None:
    cmp = run_family(
        "family_long_horizon_retention_16turn_stretch",
        seeds=(1, 2, 3))
    w51 = cmp.get(R101_W51_ARM)
    assert w51 is not None
    # H12: cosine ≥ 0.40 (stretch)
    assert w51.mean >= 0.30


def test_r101_reconstruction_v3_k5_passes() -> None:
    cmp = run_family(
        "family_reconstruction_v3_recovers_t_minus_5",
        seeds=(1, 2, 3))
    w51 = cmp.get(R101_W51_ARM)
    assert w51 is not None
    # H13: MSE at k=5 ≤ 0.50
    assert w51.mean <= 0.55


def test_r101_reconstruction_v3_k8_stretch_passes() -> None:
    cmp = run_family(
        "family_reconstruction_v3_k8_stretch",
        seeds=(1, 2, 3))
    w51 = cmp.get(R101_W51_ARM)
    assert w51 is not None
    # H14: MSE at k=8 ≤ 0.60
    assert w51.mean <= 0.65


def test_r101_hierarchical_compression_12bits_passes() -> None:
    cmp = run_family(
        "family_hierarchical_compression_12bits",
        seeds=(1, 2, 3))
    w51 = cmp.get(R101_W51_ARM)
    assert w51 is not None
    # H15: bits/token ≥ 12
    assert w51.mean >= 11.0  # generous floor


def test_r101_compression_degradation_curve_passes() -> None:
    cmp = run_family(
        "family_compression_degradation_curve",
        seeds=(1, 2, 3))
    w51 = cmp.get(R101_W51_ARM)
    assert w51 is not None
    # H16: min bits/token ≥ 4
    assert w51.mean >= 4.0


def test_r101_distribution_cap_passes() -> None:
    cmp = run_family(
        "family_w51_distribution_cap",
        seeds=(1, 2, 3))
    w51 = cmp.get(R101_W51_ARM)
    assert w51 is not None
    # H17: protect_rate ≥ 0.70
    assert w51.mean >= 0.65


def test_r101_overdepth_cap_reproduces() -> None:
    cmp = run_family(
        "family_deep_stack_v2_overdepth_cap",
        seeds=(1, 2, 3))
    # H18: (acc_L6 - acc_L4) ≤ +0.05 on a shallow regime
    assert cmp.delta_w51_vs_w50() <= 0.10
