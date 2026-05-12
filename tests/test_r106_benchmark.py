"""R-106 benchmark family-level smoke tests."""

from __future__ import annotations

from coordpy.r106_benchmark import (
    R106_BASELINE_ARM,
    R106_FAMILY_TABLE,
    R106_W53_ARM,
    run_family,
)


def test_r106_family_registry_has_12_families() -> None:
    assert len(R106_FAMILY_TABLE) == 12


def test_r106_single_bit_detect_rate_high() -> None:
    c = run_family(
        "family_single_bit_detect_rate", seeds=(1, 2))
    assert c.get(R106_W53_ARM).mean >= 0.8


def test_r106_single_bit_correction_rate_above_floor() -> None:
    c = run_family(
        "family_single_bit_correction_rate", seeds=(1, 2))
    assert c.get(R106_W53_ARM).mean >= 0.3


def test_r106_two_bit_graceful_degrade() -> None:
    c = run_family(
        "family_two_bit_graceful_degrade", seeds=(1, 2))
    assert c.get(R106_W53_ARM).mean >= 0.5


def test_r106_consensus_recall_kof2() -> None:
    c = run_family(
        "family_consensus_recall_kof2", seeds=(1, 2))
    assert c.get(R106_W53_ARM).mean >= 0.7


def test_r106_consensus_abstain_when_disagreed() -> None:
    c = run_family(
        "family_consensus_abstain_when_disagreed",
        seeds=(1, 2))
    assert c.get(R106_W53_ARM).mean == 1.0


def test_r106_mlsc_merge_replay_determinism() -> None:
    c = run_family(
        "family_mlsc_merge_replay_determinism", seeds=(1, 2))
    assert c.get(R106_W53_ARM).mean == 1.0


def test_r106_corruption_robust_carrier_safety() -> None:
    c = run_family(
        "family_corruption_robust_carrier_safety",
        seeds=(1, 2))
    assert c.get(R106_W53_ARM).mean >= 0.5


def test_r106_uncertainty_calibration_under_noise() -> None:
    c = run_family(
        "family_uncertainty_calibration_under_noise",
        seeds=(1, 2))
    assert c.get(R106_W53_ARM).mean >= 0.5


def test_r106_persistent_v5_chain_walk_depth() -> None:
    c = run_family(
        "family_persistent_v5_chain_walk_depth", seeds=(1,))
    assert c.get(R106_W53_ARM).mean == 1.0


def test_r106_w53_integration_envelope() -> None:
    c = run_family(
        "family_w53_integration_envelope", seeds=(1,))
    assert c.get(R106_W53_ARM).mean == 1.0
