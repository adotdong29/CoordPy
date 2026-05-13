"""R-109 benchmark family-level smoke tests."""

from __future__ import annotations

from coordpy.r109_benchmark import (
    R109_FAMILY_TABLE,
    R109_W54_ARM,
    run_family,
)


def test_r109_family_registry_has_14_families() -> None:
    assert len(R109_FAMILY_TABLE) == 14


def test_r109_hamming_single_bit_correct() -> None:
    c = run_family(
        "family_hamming_single_bit_correct", seeds=(1, 2))
    assert c.get(R109_W54_ARM).mean >= 0.95


def test_r109_hamming_two_bit_detect() -> None:
    c = run_family(
        "family_hamming_two_bit_detect", seeds=(1, 2, 3))
    assert c.get(R109_W54_ARM).mean >= 0.65


def test_r109_crc_v2_silent_failure_floor() -> None:
    c = run_family(
        "family_crc_v2_silent_failure_floor", seeds=(1, 2))
    assert c.get(R109_W54_ARM).mean == 1.0


def test_r109_consensus_controller_recall() -> None:
    c = run_family(
        "family_consensus_controller_recall", seeds=(1, 2))
    assert c.get(R109_W54_ARM).mean >= 0.7


def test_r109_consensus_controller_abstain_fallback() -> None:
    c = run_family(
        "family_consensus_controller_abstain_fallback",
        seeds=(1, 2, 3))
    assert c.get(R109_W54_ARM).mean == 1.0


def test_r109_mlsc_v2_trust_signature_weights() -> None:
    c = run_family(
        "family_mlsc_v2_trust_signature_weights",
        seeds=(1, 2))
    assert c.get(R109_W54_ARM).mean == 1.0


def test_r109_disagreement_arbiter_uncertainty_rises() -> None:
    c = run_family(
        "family_disagreement_arbiter_uncertainty_rises",
        seeds=(1, 2))
    assert c.get(R109_W54_ARM).mean >= 0.5


def test_r109_compromise_v6_persistent_state() -> None:
    c = run_family(
        "family_compromise_v6_persistent_state",
        seeds=(1, 2))
    assert c.get(R109_W54_ARM).mean >= 0.5


def test_r109_corruption_robust_carrier_v2_safety() -> None:
    c = run_family(
        "family_corruption_robust_carrier_v2_safety",
        seeds=(1, 2))
    assert c.get(R109_W54_ARM).mean == 1.0


def test_r109_uncertainty_v2_disagreement_downweight() -> None:
    c = run_family(
        "family_uncertainty_v2_disagreement_downweight",
        seeds=(1, 2))
    assert c.get(R109_W54_ARM).mean == 1.0


def test_r109_persistent_v6_chain_walk_depth() -> None:
    c = run_family(
        "family_persistent_v6_chain_walk_depth",
        seeds=(1,))
    assert c.get(R109_W54_ARM).mean == 1.0


def test_r109_w54_integration_envelope() -> None:
    c = run_family(
        "family_w54_integration_envelope", seeds=(1,))
    assert c.get(R109_W54_ARM).mean == 1.0


def test_r109_arbiter_v3_abstain_with_fallback_invariant() -> None:
    c = run_family(
        "family_arbiter_v3_abstain_with_fallback_invariant",
        seeds=(1, 2))
    assert c.get(R109_W54_ARM).mean == 1.0


def test_r109_deep_v5_disagreement_head_soundness() -> None:
    c = run_family(
        "family_deep_v5_disagreement_head_soundness",
        seeds=(1, 2))
    assert c.get(R109_W54_ARM).mean == 1.0
