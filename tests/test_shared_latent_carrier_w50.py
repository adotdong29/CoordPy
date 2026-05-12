"""Tests for W50 M5 shared latent carrier V2 + reconstruction V2."""

from __future__ import annotations

import pytest

from coordpy.shared_latent_carrier import (
    W50_DEFAULT_FLAT_FEATURE_DIM,
    W50_DEFAULT_MAX_K_RECONSTRUCTION,
    W50_DEFAULT_RECONSTRUCTION_HIDDEN_DIM,
    ReconstructionV2Example,
    ReconstructionV2Head,
    ReconstructionV2TrainingSet,
    RoleReuseMap,
    SharedLatentCarrierChain,
    SharedLatentCarrierV2,
    emit_reconstruction_v2_witness,
    emit_shared_latent_carrier_witness,
    evaluate_reconstruction_v2_mse_at_k,
    fit_reconstruction_v2,
    synthesize_reconstruction_v2_training_set,
    verify_reconstruction_v2_witness,
    verify_shared_latent_carrier_witness,
)


def test_carrier_v2_cid_is_deterministic() -> None:
    c1 = SharedLatentCarrierV2(
        turn_index=1, role="a", carrier_dim=6,
        values=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
        parent_carrier_cid="", role_reuse_map_cid="m")
    c2 = SharedLatentCarrierV2(
        turn_index=1, role="a", carrier_dim=6,
        values=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
        parent_carrier_cid="", role_reuse_map_cid="m")
    assert c1.cid() == c2.cid()


def test_role_reuse_map_identity_init_returns_near_input() -> None:
    rm = RoleReuseMap.init(
        role_universe=("a", "b", "c"), carrier_dim=4, seed=2)
    x = [0.1, 0.2, 0.3, 0.4]
    for role in ("a", "b", "c"):
        y = rm.project_value(role=role, carrier=x)
        for j in range(4):
            assert abs(y[j] - x[j]) < 0.05


def test_role_reuse_map_unknown_role_falls_back_to_first() -> None:
    rm = RoleReuseMap.init(
        role_universe=("a", "b"), carrier_dim=4, seed=5)
    # Unknown role should not crash; uses role index 0
    y = rm.project_value(role="zzz", carrier=[0.1] * 4)
    assert len(y) == 4


def test_reconstruction_head_init_input_dim_matches_carrier_plus_k() -> None:
    head = ReconstructionV2Head.init(
        carrier_dim=8, max_k=3, out_dim=4, seed=7)
    assert head.in_dim == 8 + 3
    assert head.carrier_dim == 8
    assert head.max_k == 3


def test_reconstruction_head_forward_returns_out_dim() -> None:
    head = ReconstructionV2Head.init(
        carrier_dim=8, max_k=3, out_dim=4, seed=9)
    y = head.forward_value(carrier=[0.1] * 8, k=2)
    assert len(y) == 4


def test_synthesize_training_set_carrier_dim_matches_max_k() -> None:
    ts = synthesize_reconstruction_v2_training_set(
        seed=11, n_sequences=4, out_dim=4, max_k=3)
    # carrier_dim = max_k * out_dim = 12
    assert ts.carrier_dim == 12


def test_synthesize_training_set_deterministic() -> None:
    a = synthesize_reconstruction_v2_training_set(seed=13)
    b = synthesize_reconstruction_v2_training_set(seed=13)
    assert a.cid() == b.cid()


def test_fit_reconstruction_v2_deterministic() -> None:
    ts = synthesize_reconstruction_v2_training_set(
        seed=21, n_sequences=4)
    h1, t1 = fit_reconstruction_v2(
        ts, n_steps=24, seed=21)
    h2, t2 = fit_reconstruction_v2(
        ts, n_steps=24, seed=21)
    assert h1.cid() == h2.cid()
    assert t1.cid() == t2.cid()


def test_reconstruction_mse_below_0_25_for_k_3_three_seeds() -> None:
    """H8: trained reconstruction head recovers t-k for k ≤ 3 with
    MSE ≤ 0.25 (honest bar under W47 pure-Python autograd cap).
    """
    mses_at_k3: list[float] = []
    for seed in (1, 2, 3):
        ts = synthesize_reconstruction_v2_training_set(
            seed=seed, n_sequences=8, out_dim=4)
        head, _ = fit_reconstruction_v2(
            ts, n_steps=480, hidden_dim=14, seed=seed,
            learning_rate=0.01, init_scale=0.05)
        mse = evaluate_reconstruction_v2_mse_at_k(
            head, ts.examples, k=3)
        mses_at_k3.append(mse)
    mean_mse = sum(mses_at_k3) / len(mses_at_k3)
    assert mean_mse <= 0.25, (
        f"H8 missed: mean MSE at k=3 {mean_mse:.4f}")


def test_chain_walker_recovers_ancestors() -> None:
    chain = SharedLatentCarrierChain.empty()
    c1 = SharedLatentCarrierV2(
        turn_index=1, role="a", carrier_dim=4,
        values=(0.1, 0.2, 0.3, 0.4),
        parent_carrier_cid="", role_reuse_map_cid="m")
    chain.add(c1)
    c2 = SharedLatentCarrierV2(
        turn_index=2, role="b", carrier_dim=4,
        values=(0.2, 0.3, 0.4, 0.5),
        parent_carrier_cid=c1.cid(),
        role_reuse_map_cid="m")
    chain.add(c2)
    c3 = SharedLatentCarrierV2(
        turn_index=3, role="a", carrier_dim=4,
        values=(0.3, 0.4, 0.5, 0.6),
        parent_carrier_cid=c2.cid(),
        role_reuse_map_cid="m")
    chain.add(c3)
    walk = chain.walk_from(c3.cid())
    assert len(walk) == 3
    assert walk[0].cid() == c3.cid()
    assert walk[1].cid() == c2.cid()
    assert walk[2].cid() == c1.cid()


def test_chain_walker_terminates_on_missing_parent() -> None:
    chain = SharedLatentCarrierChain.empty()
    c1 = SharedLatentCarrierV2(
        turn_index=1, role="a", carrier_dim=4,
        values=(0.1, 0.2, 0.3, 0.4),
        parent_carrier_cid="missing_parent",
        role_reuse_map_cid="m")
    chain.add(c1)
    walk = chain.walk_from(c1.cid())
    assert len(walk) == 1
    assert walk[0].cid() == c1.cid()


def test_shared_latent_carrier_witness_verify_pass() -> None:
    chain = SharedLatentCarrierChain.empty()
    c = SharedLatentCarrierV2(
        turn_index=1, role="a", carrier_dim=4,
        values=(0.1, 0.2, 0.3, 0.4),
        parent_carrier_cid="", role_reuse_map_cid="map_cid")
    chain.add(c)
    w = emit_shared_latent_carrier_witness(
        carrier=c, chain=chain)
    v = verify_shared_latent_carrier_witness(
        w, expected_carrier_cid=c.cid(),
        expected_role_reuse_map_cid="map_cid",
        min_chain_walk_depth=1)
    assert v["ok"] is True


def test_shared_latent_carrier_witness_detects_cid_tamper() -> None:
    chain = SharedLatentCarrierChain.empty()
    c = SharedLatentCarrierV2(
        turn_index=1, role="a", carrier_dim=4,
        values=(0.1, 0.2, 0.3, 0.4),
        parent_carrier_cid="", role_reuse_map_cid="map_cid")
    chain.add(c)
    w = emit_shared_latent_carrier_witness(
        carrier=c, chain=chain)
    v = verify_shared_latent_carrier_witness(
        w, expected_carrier_cid="00" * 32)
    assert v["ok"] is False
    assert "w50_shared_latent_carrier_cid_mismatch" in v[
        "failures"]


def test_reconstruction_v2_witness_verify_pass() -> None:
    ts = synthesize_reconstruction_v2_training_set(
        seed=33, n_sequences=4)
    head, trace = fit_reconstruction_v2(ts, n_steps=24, seed=33)
    w = emit_reconstruction_v2_witness(
        head=head, training_trace=trace,
        examples=ts.examples[:24])
    v = verify_reconstruction_v2_witness(
        w, expected_head_cid=head.cid(),
        expected_trace_cid=trace.cid())
    assert v["ok"] is True


def test_reconstruction_v2_witness_detects_mse_above_floor() -> None:
    ts = synthesize_reconstruction_v2_training_set(
        seed=37, n_sequences=4)
    head, trace = fit_reconstruction_v2(ts, n_steps=8, seed=37)
    w = emit_reconstruction_v2_witness(
        head=head, training_trace=trace,
        examples=ts.examples[:12])
    v = verify_reconstruction_v2_witness(
        w, mse_floor=0.0001)
    assert v["ok"] is False
    assert "w50_reconstruction_v2_mse_above_floor" in v[
        "failures"]
