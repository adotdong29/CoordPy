"""R-107 benchmark family-level smoke tests."""

from __future__ import annotations

from coordpy.r107_benchmark import (
    R107_BASELINE_ARM,
    R107_FAMILY_TABLE,
    R107_W54_ARM,
    run_family,
)


def test_r107_family_registry_has_12_families() -> None:
    assert len(R107_FAMILY_TABLE) == 12


def test_r107_trivial_passthrough() -> None:
    c = run_family(
        "family_trivial_w54_passthrough", seeds=(1, 2))
    assert c.get(R107_W54_ARM).mean == 1.0


def test_r107_persistent_v6_dual_skip_gain() -> None:
    c = run_family(
        "family_persistent_v6_dual_skip_gain",
        seeds=(1, 2, 3))
    assert c.get(R107_W54_ARM).mean >= 0.5


def test_r107_hex_chain_len5_transitivity() -> None:
    c = run_family(
        "family_hex_chain_len5_transitivity",
        seeds=(1, 2))
    assert c.get(R107_W54_ARM).mean >= 0.6


def test_r107_disagreement_compromise_arbiter() -> None:
    c = run_family(
        "family_disagreement_compromise_arbiter",
        seeds=(1, 2))
    assert c.get(R107_W54_ARM).mean == 1.0


def test_r107_mlsc_v2_disagreement_metadata() -> None:
    c = run_family(
        "family_mlsc_v2_disagreement_metadata",
        seeds=(1, 2))
    assert c.get(R107_W54_ARM).mean == 1.0


def test_r107_deep_v5_abstain_short_circuit() -> None:
    c = run_family(
        "family_deep_v5_abstain_short_circuit",
        seeds=(1, 2))
    assert c.get(R107_W54_ARM).mean == 1.0


def test_r107_w54_envelope_verifier() -> None:
    c = run_family(
        "family_w54_envelope_verifier", seeds=(1, 2))
    assert c.get(R107_W54_ARM).mean == 1.0


def test_r107_w54_replay_determinism() -> None:
    c = run_family(
        "family_w54_replay_determinism", seeds=(1, 2))
    assert c.get(R107_W54_ARM).mean == 1.0


def test_r107_uncertainty_layer_v2_noise_calibration() -> None:
    c = run_family(
        "family_uncertainty_layer_v2_noise_calibration",
        seeds=(1, 2))
    assert c.get(R107_W54_ARM).mean >= 0.5


def test_r107_mlsc_v2_provenance_walk() -> None:
    c = run_family(
        "family_mlsc_v2_provenance_walk", seeds=(1, 2))
    assert c.get(R107_W54_ARM).mean == 1.0


def test_r107_consensus_controller_kof_n_audit() -> None:
    c = run_family(
        "family_consensus_controller_kof_n_audit",
        seeds=(1, 2))
    assert c.get(R107_W54_ARM).mean == 1.0
