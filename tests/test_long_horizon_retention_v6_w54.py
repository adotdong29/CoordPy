"""W54 M7 — Long-Horizon Reconstruction V6 tests."""

from __future__ import annotations

from coordpy.long_horizon_retention_v4 import (
    synthesize_long_horizon_v4_training_set,
)
from coordpy.long_horizon_retention_v6 import (
    LongHorizonReconstructionV6Head,
    W54_DEFAULT_LHR_V6_FLAT_FEATURE_DIM,
    W54_DEFAULT_LHR_V6_HIDDEN_DIM,
    W54_DEFAULT_LHR_V6_MAX_K,
    W54_LHR_V6_VERIFIER_FAILURE_MODES,
    emit_lhr_v6_witness,
    evaluate_v6_degradation_curve,
    verify_lhr_v6_witness,
)


def test_lhr_v6_head_has_max_k_24() -> None:
    head = LongHorizonReconstructionV6Head.init(
        carrier_dim=24 * W54_DEFAULT_LHR_V6_FLAT_FEATURE_DIM,
        hidden_dim=W54_DEFAULT_LHR_V6_HIDDEN_DIM,
        out_dim=W54_DEFAULT_LHR_V6_FLAT_FEATURE_DIM,
        max_k=W54_DEFAULT_LHR_V6_MAX_K,
        seed=1)
    assert head.max_k == 24


def test_lhr_v6_forward_value_returns_4_tuples() -> None:
    head = LongHorizonReconstructionV6Head.init(
        carrier_dim=24 * 4, hidden_dim=8, out_dim=4,
        max_k=24, n_branches=2, n_cycles=2,
        n_merge_pairs=2, n_roles=2, seed=1)
    carrier = [0.1] * (24 * 4)
    v4, merge, role, deg = head.forward_value(
        carrier=carrier, k=8,
        branch_index=0, cycle_index=0,
        merge_pair_index=0, role_index=0)
    assert len(v4) == 4
    assert len(merge) == 4
    assert len(role) == 4
    assert len(deg) == 4
    for d in deg:
        assert d >= 0.0


def test_lhr_v6_degradation_curve_to_k48() -> None:
    head = LongHorizonReconstructionV6Head.init(
        carrier_dim=24 * 4, hidden_dim=8, out_dim=4,
        max_k=24, n_branches=2, n_cycles=2,
        n_merge_pairs=2, n_roles=2, seed=1)
    ts = synthesize_long_horizon_v4_training_set(
        n_sequences=4, sequence_length=20,
        out_dim=4, max_k=12,
        seed=1, n_branches=2, n_cycles=2)
    curve = evaluate_v6_degradation_curve(
        head, ts.examples[:4], k_max=24)
    assert len(curve) == 24
    for p in curve[:24]:
        assert p.mse >= 0.0


def test_w54_lhr_v6_verifier_rejects_wrong_head() -> None:
    head = LongHorizonReconstructionV6Head.init(
        carrier_dim=24 * 4, hidden_dim=8, out_dim=4,
        max_k=24, n_branches=2, n_cycles=2,
        n_merge_pairs=2, n_roles=2, seed=1)
    ts = synthesize_long_horizon_v4_training_set(
        n_sequences=4, sequence_length=20,
        out_dim=4, max_k=12,
        seed=1, n_branches=2, n_cycles=2)
    w = emit_lhr_v6_witness(
        head=head, examples=ts.examples[:4],
        k_max_for_degradation=8)
    v_bad = verify_lhr_v6_witness(
        w, expected_head_cid="ff" * 32,
        min_max_k=24, min_n_roles=2)
    assert not v_bad["ok"]
    assert "w54_lhr_v6_head_cid_mismatch" in v_bad["failures"]


def test_w54_lhr_v6_verifier_failure_modes_count() -> None:
    assert len(W54_LHR_V6_VERIFIER_FAILURE_MODES) == 6
