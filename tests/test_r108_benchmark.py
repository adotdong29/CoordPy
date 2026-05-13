"""R-108 benchmark family-level smoke tests."""

from __future__ import annotations

from coordpy.r108_benchmark import (
    R108_FAMILY_TABLE,
    R108_W54_ARM,
    run_family,
)


def test_r108_family_registry_has_10_families() -> None:
    assert len(R108_FAMILY_TABLE) == 10


def test_r108_persistent_v6_36turn_soundness() -> None:
    c = run_family(
        "family_persistent_v6_36turn", seeds=(1, 2))
    assert c.get(R108_W54_ARM).mean == 1.0


def test_r108_persistent_v6_40turn_stretch_soundness() -> None:
    c = run_family(
        "family_persistent_v6_40turn_stretch",
        seeds=(1, 2))
    assert c.get(R108_W54_ARM).mean == 1.0


def test_r108_lhr_v6_recovers_t_minus_18() -> None:
    c = run_family(
        "family_lhr_v6_recovers_t_minus_18", seeds=(1,))
    # MSE-style metric: lower is better; bar ≤ 0.70.
    assert c.get(R108_W54_ARM).mean <= 0.70


def test_r108_lhr_v6_k24_stretch() -> None:
    c = run_family(
        "family_lhr_v6_k24_stretch", seeds=(1, 2))
    # Stretch bar ≤ 1.50 mean across seeds.
    assert c.get(R108_W54_ARM).mean <= 1.50


def test_r108_ecc_v6_compression_16_bits() -> None:
    c = run_family(
        "family_ecc_v6_compression_16_bits",
        seeds=(1, 2))
    assert c.get(R108_W54_ARM).mean >= 16.0


def test_r108_lhr_v6_degradation_curve() -> None:
    c = run_family(
        "family_lhr_v6_degradation_curve", seeds=(1,))
    assert c.get(R108_W54_ARM).mean <= 1.0


def test_r108_w54_distribution_cap() -> None:
    c = run_family(
        "family_w54_distribution_cap", seeds=(1, 2))
    assert c.get(R108_W54_ARM).mean >= 0.5


def test_r108_deep_v5_overdepth_cap() -> None:
    c = run_family(
        "family_deep_v5_overdepth_cap", seeds=(1,))
    assert c.get(R108_W54_ARM).mean == 1.0


def test_r108_ecc_v6_rate_floor_falsifier() -> None:
    c = run_family(
        "family_ecc_v6_rate_floor_falsifier", seeds=(1, 2))
    assert c.get(R108_W54_ARM).mean == 1.0


def test_r108_tvs_arbiter_v3_oracle_dominance() -> None:
    c = run_family(
        "family_tvs_arbiter_v3_oracle_dominance",
        seeds=(1, 2))
    assert c.get(R108_W54_ARM).mean >= 0.5
