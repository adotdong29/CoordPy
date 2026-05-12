"""R-105 benchmark family-level smoke tests."""

from __future__ import annotations

from coordpy.r105_benchmark import (
    R105_BASELINE_ARM,
    R105_FAMILY_TABLE,
    R105_W53_ARM,
    run_family,
)


def test_r105_family_registry_has_10_families() -> None:
    assert len(R105_FAMILY_TABLE) == 10


def test_r105_ecc_compression_meets_14p5_bits() -> None:
    c = run_family(
        "family_ecc_compression_14p5_bits", seeds=(1,))
    w53_mean = c.get(R105_W53_ARM).mean
    w52_mean = c.get(R105_BASELINE_ARM).mean
    assert w53_mean >= 14.5
    assert w53_mean >= w52_mean


def test_r105_ecc_rate_floor_falsifier_misses() -> None:
    c = run_family(
        "family_ecc_rate_floor_falsifier", seeds=(1, 2))
    assert c.get(R105_W53_ARM).mean == 1.0


def test_r105_arbiter_strict_dominance_oracle() -> None:
    c = run_family(
        "family_arbiter_strict_dominance", seeds=(1, 2))
    assert c.get(R105_W53_ARM).mean >= 0.5


def test_r105_lhr_v5_degradation_curve_within_range() -> None:
    c = run_family(
        "family_lhr_v5_degradation_curve", seeds=(1,))
    assert c.get(R105_W53_ARM).mean <= 1.0
