"""Tests for the R-99 retention/reconstruction/compression benchmark."""

from __future__ import annotations

import pytest

from coordpy.r99_benchmark import (
    R99_BASELINE_ARM,
    R99_FAMILY_TABLE,
    R99_W50_ARM,
    R99AggregateResult,
    R99FamilyComparison,
    R99SeedResult,
    run_all_families,
    run_family,
)


def test_r99_has_seven_families() -> None:
    assert len(R99_FAMILY_TABLE) == 7


def test_r99_run_family_returns_comparison() -> None:
    cmp = run_family(
        "family_long_horizon_retention_8turn", seeds=(1,))
    assert isinstance(cmp, R99FamilyComparison)
    assert cmp.get(R99_BASELINE_ARM) is not None
    assert cmp.get(R99_W50_ARM) is not None


def test_r99_run_unknown_family_raises() -> None:
    with pytest.raises(ValueError):
        run_family("family_does_not_exist", seeds=(1,))


def test_r99_h6_long_horizon_retention_8turn() -> None:
    cmp = run_family(
        "family_long_horizon_retention_8turn", seeds=(1, 2, 3))
    w50 = cmp.get(R99_W50_ARM)
    base = cmp.get(R99_BASELINE_ARM)
    assert w50 is not None
    assert base is not None
    # H6: w50 mean cosine ≥ 0.90
    assert w50.mean >= 0.90, (
        f"H6 missed: w50 cosine {w50.mean}")
    # W50 strictly better than W49 baseline
    assert w50.mean > base.mean


def test_r99_h7_long_horizon_retention_12turn_stretch() -> None:
    cmp = run_family(
        "family_long_horizon_retention_12turn_stretch",
        seeds=(1, 2, 3))
    w50 = cmp.get(R99_W50_ARM)
    assert w50 is not None
    # H7: cos ≥ 0.70 (stretch — honest about drop-off)
    assert w50.mean >= 0.70, (
        f"H7 missed: w50 cosine {w50.mean}")


def test_r99_h8_reconstruction_v2_recovers_prior_turn() -> None:
    cmp = run_family(
        "family_reconstruction_v2_recovers_prior_turn",
        seeds=(1, 2, 3))
    w50 = cmp.get(R99_W50_ARM)
    assert w50 is not None
    # H8: MSE ≤ 0.25 at k=3 (random-baseline MSE = 0.333)
    assert w50.mean <= 0.25, (
        f"H8 missed: w50 MSE {w50.mean}")


def test_r99_h9_adaptive_compression_8bits() -> None:
    cmp = run_family(
        "family_adaptive_compression_8bits", seeds=(1, 2, 3))
    w50 = cmp.get(R99_W50_ARM)
    base = cmp.get(R99_BASELINE_ARM)
    assert w50 is not None
    assert base is not None
    # H9: bits/visible-token ≥ 8.0
    assert w50.mean >= 8.0, (
        f"H9 missed: bits/token {w50.mean}")
    # Strictly better than W49 baseline (5.0)
    assert w50.mean > 5.0


def test_r99_h14_adaptive_compression_rate_falsifier() -> None:
    cmp = run_family(
        "family_adaptive_compression_rate_falsifier",
        seeds=(1, 2, 3))
    w50 = cmp.get(R99_W50_ARM)
    assert w50 is not None
    # H14: falsifier reproduces (target_missed = True)
    assert w50.mean == 1.0


def test_r99_aggressive_compression_recovery_v2() -> None:
    cmp = run_family(
        "family_aggressive_compression_recovery_v2",
        seeds=(1, 2, 3))
    w50 = cmp.get(R99_W50_ARM)
    assert w50 is not None
    # Recovery rate ≥ 0.60
    assert w50.mean >= 0.60, (
        f"recovery rate {w50.mean}")


def test_r99_h15_w50_distribution_cap() -> None:
    cmp = run_family(
        "family_w50_distribution_cap", seeds=(1, 2, 3))
    w50 = cmp.get(R99_W50_ARM)
    assert w50 is not None
    # H15: protect rate ≥ 0.7
    assert w50.mean >= 0.7, (
        f"H15 missed: protect rate {w50.mean}")


def test_r99_run_all_families_returns_full_map() -> None:
    out = run_all_families(seeds=(1,))
    assert len(out) == 7
    for name, cmp in out.items():
        assert cmp.family == name


def test_r99_seed_result_deterministic_across_runs() -> None:
    cmp1 = run_family(
        "family_long_horizon_retention_8turn", seeds=(1,))
    cmp2 = run_family(
        "family_long_horizon_retention_8turn", seeds=(1,))
    a1 = cmp1.get(R99_W50_ARM)
    a2 = cmp2.get(R99_W50_ARM)
    assert a1 is not None and a2 is not None
    assert a1.values == a2.values


def test_r99_aggregate_serialises_to_dict() -> None:
    cmp = run_family(
        "family_adaptive_compression_rate_falsifier",
        seeds=(1, 2, 3))
    d = cmp.to_dict()
    assert d["family"] == (
        "family_adaptive_compression_rate_falsifier")
    assert len(d["aggregates"]) == 2
