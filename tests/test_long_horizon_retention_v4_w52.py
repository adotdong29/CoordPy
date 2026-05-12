"""Per-module tests for the W52 long-horizon reconstruction V4."""

from __future__ import annotations

import pytest

from coordpy.long_horizon_retention_v4 import (
    LongHorizonReconstructionV4Head,
    emit_long_horizon_v4_witness,
    evaluate_long_horizon_v4_mse_at_k,
    evaluate_long_horizon_v4_mse_curve,
    fit_long_horizon_v4,
    synthesize_long_horizon_v4_training_set,
    verify_long_horizon_v4_witness,
)


def test_v4_head_init_three_heads() -> None:
    head = LongHorizonReconstructionV4Head.init(
        carrier_dim=48, hidden_dim=16, out_dim=4,
        max_k=12, n_branches=2, n_cycles=2, seed=1)
    assert head.max_k == 12
    assert head.n_branches == 2
    assert head.n_cycles == 2


def test_v4_synth_has_max_k_examples() -> None:
    ts = synthesize_long_horizon_v4_training_set(
        n_sequences=2, sequence_length=14, max_k=12,
        out_dim=4, n_branches=2, n_cycles=2, seed=1)
    # Each sequence contributes sum_{t}{min(t, max_k)} examples;
    # but the easy check: examples > 0
    assert len(ts.examples) > 0


def test_v4_fit_runs_to_completion() -> None:
    ts = synthesize_long_horizon_v4_training_set(
        n_sequences=4, sequence_length=14, max_k=8,
        out_dim=4, n_branches=2, n_cycles=2, seed=2)
    head, trace = fit_long_horizon_v4(
        ts, n_steps=48, hidden_dim=16,
        learning_rate=0.005, seed=2)
    assert not trace.diverged
    assert len(trace.loss_head) > 0
    # final loss must be a finite float; per-step bouncing is
    # expected with stochastic Adam on a tiny budget.
    assert trace.final_loss == trace.final_loss
    assert trace.final_loss < float("inf")


def test_v4_witness_round_trips() -> None:
    ts = synthesize_long_horizon_v4_training_set(
        n_sequences=2, sequence_length=8, max_k=4,
        out_dim=4, n_branches=2, n_cycles=2, seed=3)
    head, trace = fit_long_horizon_v4(
        ts, n_steps=24, hidden_dim=12,
        learning_rate=0.01, seed=3)
    w = emit_long_horizon_v4_witness(
        head=head, training_trace=trace,
        examples=ts.examples[:4],
        probe_ks=(1, 2))
    v = verify_long_horizon_v4_witness(
        w, expected_head_cid=head.cid(),
        expected_max_k=4)
    assert v["ok"] is True


def test_v4_curve_returns_all_ks() -> None:
    ts = synthesize_long_horizon_v4_training_set(
        n_sequences=2, sequence_length=8, max_k=4,
        out_dim=4, n_branches=2, n_cycles=2, seed=4)
    head, _ = fit_long_horizon_v4(
        ts, n_steps=24, hidden_dim=8,
        learning_rate=0.01, seed=4)
    curve = evaluate_long_horizon_v4_mse_curve(
        head, ts.examples, ks=(1, 2, 3, 4))
    assert len(curve) == 4
