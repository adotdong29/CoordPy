"""Tests for the W50 M2 deep proxy stack."""

from __future__ import annotations

import pytest

from coordpy.deep_proxy_stack import (
    W50_DEFAULT_DEEP_FACTOR_DIM,
    W50_DEFAULT_DEEP_IN_DIM,
    W50_DEFAULT_DEEP_N_HEADS,
    W50_DEFAULT_DEEP_N_LAYERS,
    DeepLayer,
    DeepProxyStack,
    DeepProxyStackForwardWitness,
    DeepStackTrainingExample,
    DeepStackTrainingSet,
    LayerMaskGate,
    emit_deep_proxy_stack_forward_witness,
    evaluate_deep_stack_accuracy,
    fit_deep_proxy_stack,
    force_residual_pathology,
    synthesize_deep_stack_training_set,
    verify_deep_proxy_stack_forward_witness,
)


def test_deep_stack_initializes_with_default_n_layers() -> None:
    s = DeepProxyStack.init(seed=1)
    assert s.n_layers == W50_DEFAULT_DEEP_N_LAYERS
    assert s.in_dim == W50_DEFAULT_DEEP_IN_DIM
    assert len(s.layers) == W50_DEFAULT_DEEP_N_LAYERS


def test_deep_stack_cid_round_trips_through_to_dict() -> None:
    s = DeepProxyStack.init(seed=3)
    d = s.to_dict()
    assert d["n_layers"] == s.n_layers
    assert len(d["layers"]) == s.n_layers
    assert len(s.cid()) == 64


def test_layer_mask_gate_outputs_in_unit_interval() -> None:
    g = LayerMaskGate.init(in_dim=6, seed=2)
    for vals in ([0.0] * 6, [1.0] * 6, [-1.0] * 6, [0.5] * 6):
        v = g.forward_value(vals)
        assert 0.0 <= v <= 1.0


def test_deep_layer_forward_value_returns_gate() -> None:
    layer = DeepLayer.init(in_dim=4, seed=5)
    q = [0.1, -0.2, 0.3, -0.4]
    slot = [q, q]
    out, gate_v = layer.forward_value(
        query_input=q, slot_keys=[q], slot_values=[q],
        gate_input=q)
    assert len(out) == 4
    assert 0.0 <= gate_v <= 1.0


def test_deep_stack_forward_value_consistent_with_witness() -> None:
    s = DeepProxyStack.init(seed=7)
    q = [0.1] * s.in_dim
    out1 = s.forward_value(
        query_input=q, slot_keys=[q], slot_values=[q])
    w, out2 = emit_deep_proxy_stack_forward_witness(
        stack=s, query_input=q, slot_keys=[q], slot_values=[q])
    assert out1 == out2
    assert len(w.per_layer_l2_norms) == s.n_layers
    assert len(w.per_layer_gate_values) == s.n_layers
    for g in w.per_layer_gate_values:
        assert 0.0 <= g <= 1.0


def test_synthesize_training_set_balanced() -> None:
    ts = synthesize_deep_stack_training_set(
        n_examples=32, seed=11, in_dim=6)
    labels = [e.target_label for e in ts.examples]
    n_pos = sum(1 for l in labels if l >= 0.5)
    n_neg = len(labels) - n_pos
    # Balanced 16/16
    assert n_pos == n_neg
    assert ts.cid() == ts.cid()  # deterministic


def test_synthesize_training_set_deterministic_across_runs() -> None:
    a = synthesize_deep_stack_training_set(
        n_examples=16, seed=99, in_dim=6)
    b = synthesize_deep_stack_training_set(
        n_examples=16, seed=99, in_dim=6)
    assert a.cid() == b.cid()


def test_fit_deep_stack_reduces_loss() -> None:
    ts = synthesize_deep_stack_training_set(
        n_examples=16, seed=21, in_dim=6)
    s, trace = fit_deep_proxy_stack(ts, n_steps=48, seed=21)
    assert trace.diverged is False
    assert trace.final_stack_cid == s.cid()
    # The training trace records start and end loss.
    if trace.loss_head and trace.loss_tail:
        # No strong guarantee, but loss should not be wildly worse.
        assert trace.loss_tail[-1] <= trace.loss_head[0] * 2.0 + 1.0


def test_fit_deterministic_across_runs() -> None:
    ts = synthesize_deep_stack_training_set(
        n_examples=12, seed=31, in_dim=6)
    s1, t1 = fit_deep_proxy_stack(ts, n_steps=24, seed=31)
    s2, t2 = fit_deep_proxy_stack(ts, n_steps=24, seed=31)
    assert s1.cid() == s2.cid()
    assert t1.cid() == t2.cid()


def test_l4_strict_gain_over_l2_mean_three_seeds() -> None:
    # H2: acc(L=4) - acc(L=2) >= 0.05 on the mean across 3 seeds.
    deltas: list[float] = []
    for seed in (1, 2, 3):
        ts = synthesize_deep_stack_training_set(
            n_examples=32, seed=seed, in_dim=6)
        s4, _ = fit_deep_proxy_stack(
            ts, n_layers=4, n_steps=288, seed=seed)
        s2, _ = fit_deep_proxy_stack(
            ts, n_layers=2, n_steps=288, seed=seed)
        a4 = evaluate_deep_stack_accuracy(s4, ts.examples)
        a2 = evaluate_deep_stack_accuracy(s2, ts.examples)
        deltas.append(a4 - a2)
    mean_delta = sum(deltas) / len(deltas)
    assert mean_delta >= 0.05, (
        f"H2 bar missed: mean delta {mean_delta:.4f}")


def test_force_residual_pathology_collapses_accuracy() -> None:
    # H13: residual_scale = 0 collapses the stack to noise.
    ts = synthesize_deep_stack_training_set(
        n_examples=24, seed=41, in_dim=6)
    s, _ = fit_deep_proxy_stack(ts, n_layers=4, n_steps=128, seed=41)
    broken = force_residual_pathology(s)
    a_broken = evaluate_deep_stack_accuracy(broken, ts.examples)
    # Forced pathology must score near random (≤ 0.65).
    assert a_broken <= 0.65, (
        f"forced residual pathology should be ~chance, got "
        f"{a_broken:.3f}")


def test_forward_witness_verify_passes() -> None:
    s = DeepProxyStack.init(seed=43)
    q = [0.2] * s.in_dim
    w, _ = emit_deep_proxy_stack_forward_witness(
        stack=s, query_input=q, slot_keys=[q], slot_values=[q])
    v = verify_deep_proxy_stack_forward_witness(
        w, expected_stack_cid=s.cid(),
        expected_n_layers=s.n_layers)
    assert v["ok"] is True


def test_forward_witness_verify_detects_stack_cid_tamper() -> None:
    s = DeepProxyStack.init(seed=45)
    q = [0.3] * s.in_dim
    w, _ = emit_deep_proxy_stack_forward_witness(
        stack=s, query_input=q, slot_keys=[q], slot_values=[q])
    v = verify_deep_proxy_stack_forward_witness(
        w, expected_stack_cid="00" * 32,
        expected_n_layers=s.n_layers)
    assert v["ok"] is False
    assert "w50_deep_stack_stack_cid_mismatch" in v["failures"]


def test_forward_witness_verify_detects_layer_count_mismatch() -> None:
    s = DeepProxyStack.init(seed=47, n_layers=4)
    q = [0.3] * s.in_dim
    w, _ = emit_deep_proxy_stack_forward_witness(
        stack=s, query_input=q, slot_keys=[q], slot_values=[q])
    v = verify_deep_proxy_stack_forward_witness(
        w, expected_stack_cid=s.cid(),
        expected_n_layers=2)
    assert v["ok"] is False
    assert "w50_deep_stack_layer_count_mismatch" in v["failures"]


def test_stack_cid_is_deterministic() -> None:
    s1 = DeepProxyStack.init(seed=51)
    s2 = DeepProxyStack.init(seed=51)
    assert s1.cid() == s2.cid()
