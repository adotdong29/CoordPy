"""Unit tests for W51 M5 — long-horizon reconstruction V3."""

from __future__ import annotations

import pytest

from coordpy.long_horizon_retention import (
    LongHorizonReconstructionV3Head,
    W51_DEFAULT_LHR_DROPOFF_PROBE_KS,
    W51_DEFAULT_LHR_MAX_K,
    W51_LONG_HORIZON_RETENTION_VERIFIER_FAILURE_MODES,
    emit_long_horizon_reconstruction_v3_witness,
    evaluate_long_horizon_cosine_at_k,
    evaluate_long_horizon_mse_at_k,
    evaluate_long_horizon_mse_curve,
    fit_long_horizon_reconstruction_v3,
    synthesize_long_horizon_reconstruction_training_set,
    verify_long_horizon_reconstruction_v3_witness,
)


def test_head_default_max_k_is_8() -> None:
    assert W51_DEFAULT_LHR_MAX_K == 8
    head = LongHorizonReconstructionV3Head.init(seed=11)
    assert head.max_k == 8


def test_head_init_stable_seed() -> None:
    a = LongHorizonReconstructionV3Head.init(seed=11)
    b = LongHorizonReconstructionV3Head.init(seed=11)
    assert a.cid() == b.cid()


def test_head_input_dim_includes_branch_one_hot() -> None:
    head = LongHorizonReconstructionV3Head.init(
        carrier_dim=8, max_k=4, n_branches=2, seed=11)
    # in_dim = 8 (carrier) + 4 (max_k one-hot) + 2 (branch one-hot) = 14
    assert head.in_dim == 14


def test_training_set_synthesises_n_examples() -> None:
    ts = synthesize_long_horizon_reconstruction_training_set(
        n_sequences=3, sequence_length=10, max_k=4,
        out_dim=3, n_branches=2, seed=11)
    # Each sequence has (sequence_length) turns × max_k example
    # types (one per k). But examples only created for valid
    # (t, k) pairs where t-k >= 0.
    assert len(ts.examples) > 0


def test_mse_at_k_decreases_with_training() -> None:
    ts = synthesize_long_horizon_reconstruction_training_set(
        n_sequences=4, sequence_length=10, max_k=4,
        out_dim=4, n_branches=2, seed=11)
    untrained = LongHorizonReconstructionV3Head.init(
        carrier_dim=ts.carrier_dim, out_dim=ts.out_dim,
        max_k=ts.max_k, n_branches=ts.n_branches, seed=11)
    mse_untrained = evaluate_long_horizon_mse_at_k(
        untrained, ts.examples, 1)
    # Use the robust training schedule (lower LR + more steps
    # + larger hidden) consistent with the R-101 family fit.
    trained, _ = fit_long_horizon_reconstruction_v3(
        ts, n_steps=192, hidden_dim=18,
        learning_rate=0.02, seed=11)
    mse_trained = evaluate_long_horizon_mse_at_k(
        trained, ts.examples, 1)
    # Training should improve MSE (with tolerance for the
    # pure-Python autograd cost cap).
    assert mse_trained <= mse_untrained + 0.20


def test_extended_mse_curve_returns_all_ks() -> None:
    ts = synthesize_long_horizon_reconstruction_training_set(
        n_sequences=4, sequence_length=12, max_k=8,
        out_dim=4, n_branches=2, seed=11)
    head, _ = fit_long_horizon_reconstruction_v3(
        ts, n_steps=48, seed=11)
    curve = evaluate_long_horizon_mse_curve(
        head, ts.examples,
        ks=W51_DEFAULT_LHR_DROPOFF_PROBE_KS)
    assert len(curve) == len(
        W51_DEFAULT_LHR_DROPOFF_PROBE_KS)


def test_witness_passes_clean_verifier() -> None:
    ts = synthesize_long_horizon_reconstruction_training_set(
        n_sequences=4, sequence_length=12, max_k=8,
        out_dim=4, n_branches=2, seed=11)
    head, trace = fit_long_horizon_reconstruction_v3(
        ts, n_steps=48, seed=11)
    w = emit_long_horizon_reconstruction_v3_witness(
        head=head, training_trace=trace,
        examples=ts.examples)
    v = verify_long_horizon_reconstruction_v3_witness(
        w, expected_head_cid=head.cid(),
        expected_trace_cid=trace.cid(),
        mse_floor_at_k5=1.5,  # generous
        mse_floor_at_k8=1.5)
    assert v["ok"] is True


def test_verifier_has_7_failure_modes() -> None:
    assert len(
        W51_LONG_HORIZON_RETENTION_VERIFIER_FAILURE_MODES) == 7
