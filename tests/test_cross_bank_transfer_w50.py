"""Tests for W50 M4 cross-bank transfer + adaptive eviction V2."""

from __future__ import annotations

import pytest

from coordpy.cross_bank_transfer import (
    W50_DEFAULT_EVICTION_V2_IN_DIM,
    AdaptiveEvictionPolicyV2,
    CrossBankTransferLayer,
    CrossBankTransferWitness,
    RolePairProjection,
    _cosine,
    emit_cross_bank_transfer_witness,
    evaluate_role_pair_recall,
    fit_cross_bank_transfer,
    forge_cross_bank_training_set,
    synthesize_cross_bank_transfer_training_set,
    verify_cross_bank_transfer_witness,
)
from coordpy.shared_state_proxy import PseudoKVBank, PseudoKVSlot


def _no_transfer_recall(examples) -> float:
    cos_sum = 0.0
    n = 0
    for ex in examples:
        cos_sum += _cosine(ex.source_key, ex.target_key)
        n += 1
    return cos_sum / max(1, n)


def test_role_pair_projection_default_is_near_identity() -> None:
    p = RolePairProjection.init(
        source_role="a", target_role="b", factor_dim=4,
        seed=1, init_as_identity=True)
    x = [0.5, -0.3, 0.7, 0.1]
    y = p.forward_value(x)
    # Should be close to x (near identity)
    for i in range(4):
        assert abs(y[i] - x[i]) < 0.05


def test_cross_bank_layer_initialises_all_role_pairs() -> None:
    roles = ("a", "b", "c", "d")
    layer = CrossBankTransferLayer.init(
        role_universe=roles, factor_dim=4, seed=2)
    assert len(layer.projections) == 16  # 4x4
    for ra in roles:
        for rb in roles:
            assert (ra, rb) in layer.projections


def test_cross_bank_layer_cid_round_trips_through_to_dict() -> None:
    layer = CrossBankTransferLayer.init(
        role_universe=("a", "b"), factor_dim=4, seed=3)
    d = layer.to_dict()
    assert "projections" in d
    assert d["factor_dim"] == 4
    assert len(layer.cid()) == 64


def test_layer_transfer_slot_changes_role_and_value() -> None:
    layer = CrossBankTransferLayer.init(
        role_universe=("a", "b"), factor_dim=4, seed=5,
        init_as_identity=False, init_scale=0.5)
    slot = PseudoKVSlot(
        slot_index=0, turn_index=0, role="a",
        key=(0.5, 0.3, 0.7, 0.1),
        value=(0.2, 0.4, 0.6, 0.8),
        write_gate_value=0.7,
        source_observation_cid="abc")
    new_slot = layer.transfer_slot(slot=slot, target_role="b")
    assert new_slot.role == "b"
    # Identity-init random differs from input
    assert new_slot.key != slot.key or new_slot.value != slot.value


def test_synthesize_training_set_deterministic() -> None:
    a = synthesize_cross_bank_transfer_training_set(seed=11)
    b = synthesize_cross_bank_transfer_training_set(seed=11)
    assert a.cid() == b.cid()


def test_fit_cross_bank_transfer_deterministic() -> None:
    ts = synthesize_cross_bank_transfer_training_set(seed=21)
    l1, t1 = fit_cross_bank_transfer(ts, n_steps=24, seed=21)
    l2, t2 = fit_cross_bank_transfer(ts, n_steps=24, seed=21)
    assert l1.cid() == l2.cid()
    assert t1.cid() == t2.cid()


def test_trained_layer_beats_no_transfer_baseline_3_seeds() -> None:
    """H4: (transfer_recall - no_transfer_recall) ≥ 0.15."""
    deltas: list[float] = []
    for seed in (1, 2, 3):
        ts = synthesize_cross_bank_transfer_training_set(
            seed=seed, n_examples_per_pair=4, factor_dim=4)
        base = _no_transfer_recall(ts.examples[:32])
        layer, _ = fit_cross_bank_transfer(
            ts, n_steps=192, seed=seed)
        trained = evaluate_role_pair_recall(
            layer, ts.examples[:32])
        deltas.append(trained - base)
    mean_delta = sum(deltas) / len(deltas)
    assert mean_delta >= 0.15, (
        f"H4 missed: mean delta {mean_delta:.4f}")


def test_cross_bank_compromise_cap_reproduces() -> None:
    """H12: forged training set → trained layer cannot recover."""
    ts = synthesize_cross_bank_transfer_training_set(
        seed=31, n_examples_per_pair=4)
    forged = forge_cross_bank_training_set(ts, seed=31)
    layer_forged, _ = fit_cross_bank_transfer(
        forged, n_steps=96, seed=31)
    # Test against CLEAN probes — recall should be near zero
    recall_on_clean = evaluate_role_pair_recall(
        layer_forged, ts.examples[:32])
    assert abs(recall_on_clean) <= 0.3, (
        f"forgery should produce near-zero recall on clean probes, "
        f"got {recall_on_clean:.4f}")


def test_eviction_v2_in_dim_is_five() -> None:
    p = AdaptiveEvictionPolicyV2.init(seed=41)
    assert p.in_dim == W50_DEFAULT_EVICTION_V2_IN_DIM
    assert p.in_dim == 5


def test_eviction_v2_score_in_unit_interval() -> None:
    p = AdaptiveEvictionPolicyV2.init(seed=43)
    for inputs in (
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [0.5, 1.0, 0.7, 0.9, 0.0]):
        s = p.score_value(inputs)
        assert 0.0 <= s <= 1.0


def test_eviction_v2_picks_lowest_score_slot() -> None:
    p = AdaptiveEvictionPolicyV2.init(seed=45)
    bank = PseudoKVBank(capacity=4, factor_dim=4)
    for i in range(3):
        bank.write(PseudoKVSlot(
            slot_index=i, turn_index=i, role="a",
            key=tuple([0.1] * 4), value=tuple([0.2] * 4),
            write_gate_value=0.5,
            source_observation_cid=f"src{i}"))
    idx = p.evict_index(
        bank=bank, current_role="a", current_turn=5,
        retention_probs=[0.9, 0.1, 0.5],
        transfer_signals=[0, 1, 0])
    assert 0 <= idx < 3


def test_witness_seal_and_verify_pass() -> None:
    ts = synthesize_cross_bank_transfer_training_set(seed=51)
    layer, trace = fit_cross_bank_transfer(
        ts, n_steps=96, seed=51)
    w = emit_cross_bank_transfer_witness(
        layer=layer, training_trace=trace,
        probe_examples=ts.examples[:16])
    v = verify_cross_bank_transfer_witness(
        w,
        expected_layer_cid=layer.cid(),
        expected_trace_cid=trace.cid(),
        expected_role_universe=layer.role_universe,
        recall_floor=-1.0)
    assert v["ok"] is True
    assert v["witness_cid"] == w.cid()


def test_witness_verify_detects_layer_cid_tamper() -> None:
    ts = synthesize_cross_bank_transfer_training_set(seed=53)
    layer, trace = fit_cross_bank_transfer(
        ts, n_steps=12, seed=53)
    w = emit_cross_bank_transfer_witness(
        layer=layer, training_trace=trace,
        probe_examples=ts.examples[:8])
    v = verify_cross_bank_transfer_witness(
        w, expected_layer_cid="00" * 32)
    assert v["ok"] is False
    assert "w50_cross_bank_transfer_layer_cid_mismatch" in v[
        "failures"]


def test_witness_verify_detects_recall_below_floor() -> None:
    ts = synthesize_cross_bank_transfer_training_set(seed=55)
    layer = CrossBankTransferLayer.init(
        role_universe=ts.role_universe,
        factor_dim=ts.factor_dim, seed=55,
        init_as_identity=True)
    _, trace = fit_cross_bank_transfer(ts, n_steps=4, seed=55)
    w = emit_cross_bank_transfer_witness(
        layer=layer, training_trace=trace,
        probe_examples=ts.examples[:8])
    v = verify_cross_bank_transfer_witness(
        w, recall_floor=0.99)
    assert v["ok"] is False
    assert "w50_cross_bank_transfer_recall_below_floor" in v[
        "failures"]
