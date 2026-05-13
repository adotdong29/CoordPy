"""W56 module unit tests."""

from __future__ import annotations

import numpy as np

from coordpy.consensus_fallback_controller_v2 import (
    ConsensusFallbackControllerV2,
    W56_CONSENSUS_V2_STAGES,
    W56_CONSENSUS_V2_STAGE_SUBSTRATE,
    emit_consensus_v2_witness,
)
from coordpy.corruption_robust_carrier_v4 import (
    CorruptionRobustCarrierV4,
    bch_31_16_decode,
    bch_31_16_encode,
    interleave_2d,
    deinterleave_2d,
)
from coordpy.deep_proxy_stack_v6 import DeepProxyStackV6
from coordpy.deep_substrate_hybrid import (
    DeepSubstrateHybrid,
    deep_substrate_hybrid_forward,
)
from coordpy.disagreement_algebra import AlgebraTrace
from coordpy.disagreement_algebra_v2 import (
    emit_disagreement_algebra_v2_witness,
)
from coordpy.ecc_codebook_v8 import (
    ECCCodebookV8,
    probe_ecc_v8_rate_floor_falsifier,
)
from coordpy.kv_bridge import (
    KVBridgeProjection,
    bridge_carrier_and_measure,
    inject_carrier_into_kv_cache,
)
from coordpy.long_horizon_retention_v8 import (
    LongHorizonReconstructionV8Head,
    evaluate_lhr_v8_substrate_vs_proxy,
)
from coordpy.mergeable_latent_capsule_v3 import (
    make_root_capsule_v3,
)
from coordpy.mergeable_latent_capsule_v4 import (
    MergeOperatorV4,
    W56_MLSC_V4_ALGEBRA_MERGE,
    emit_mlsc_v4_witness,
    wrap_v3_as_v4,
)
from coordpy.multi_hop_translator_v6 import (
    OctBackendChainPath,
    evaluate_oct_chain_len7_fidelity,
    substrate_trust_weighted_arbitration,
)
from coordpy.persistent_latent_v8 import (
    PersistentLatentStateV8Chain,
    V8StackedCell,
    emit_persistent_v8_witness,
    step_persistent_state_v8,
)
from coordpy.substrate_adapter import (
    SUBSTRATE_TIER_SUBSTRATE_FULL,
    SUBSTRATE_TIER_TEXT_ONLY,
    probe_synthetic_adapter,
    probe_tiny_substrate_adapter,
)
from coordpy.tiny_substrate import (
    TinyKVCache,
    build_default_tiny_substrate,
    decode_greedy_tiny_substrate,
    detokenize_bytes,
    forward_tiny_substrate,
    tokenize_bytes,
)
from coordpy.transcript_vs_shared_arbiter_v5 import (
    W56_TVS_V5_ARMS,
    emit_tvs_arbiter_v5_witness,
    six_arm_compare,
)
from coordpy.uncertainty_layer_v4 import (
    compose_uncertainty_report_v4,
)


# ---------------------------------------------------------------------------
# Tiny substrate
# ---------------------------------------------------------------------------


def test_tiny_substrate_forward_determinism() -> None:
    p = build_default_tiny_substrate(seed=42)
    ids = tokenize_bytes("hello", max_len=12)
    t1 = forward_tiny_substrate(p, ids)
    t2 = forward_tiny_substrate(p, ids)
    assert np.array_equal(t1.logits, t2.logits)
    assert all(
        np.array_equal(a, b)
        for a, b in zip(t1.hidden_states, t2.hidden_states))
    assert t1.kv_cache.cid() == t2.kv_cache.cid()


def test_tiny_substrate_kv_cache_reuse() -> None:
    p = build_default_tiny_substrate(seed=42)
    ids = tokenize_bytes("kv-reuse-test", max_len=20)
    t_full = forward_tiny_substrate(p, ids)
    half = len(ids) // 2
    t_a = forward_tiny_substrate(p, ids[:half])
    t_b = forward_tiny_substrate(
        p, ids[half:], kv_cache=t_a.kv_cache)
    # Last-position logits must match within float64 precision.
    diff = float(np.max(np.abs(
        t_full.logits[-1] - t_b.logits[-1])))
    assert diff < 1e-9


def test_tiny_substrate_causal_mask_soundness() -> None:
    p = build_default_tiny_substrate(seed=42)
    ids = tokenize_bytes("causal-test", max_len=12)
    t = forward_tiny_substrate(p, ids, return_attention=True)
    for attn in t.attn_weights_per_layer:
        n_heads, T_new, T_all = attn.shape
        for i in range(T_new):
            for j in range(i + 1, T_all):
                assert float(np.max(attn[:, i, j])) < 1e-9


def test_tiny_substrate_decode_greedy() -> None:
    p = build_default_tiny_substrate(seed=42)
    ids = tokenize_bytes("decode", max_len=8)
    gen, trace = decode_greedy_tiny_substrate(
        p, ids, max_new_tokens=4)
    assert len(gen) <= 4
    assert all(0 <= t < p.config.vocab_size for t in gen)


def test_tiny_substrate_tokenize_round_trip() -> None:
    text = "hello, world!"
    ids = tokenize_bytes(text, max_len=64)
    back = detokenize_bytes(ids)
    assert back == text


# ---------------------------------------------------------------------------
# Substrate adapter
# ---------------------------------------------------------------------------


def test_substrate_adapter_tiers() -> None:
    tiny = probe_tiny_substrate_adapter()
    synth = probe_synthetic_adapter()
    assert tiny.tier == SUBSTRATE_TIER_SUBSTRATE_FULL
    assert synth.tier == SUBSTRATE_TIER_TEXT_ONLY


# ---------------------------------------------------------------------------
# KV bridge
# ---------------------------------------------------------------------------


def test_kv_bridge_injection_perturbs_logits() -> None:
    p = build_default_tiny_substrate(seed=42)
    proj = KVBridgeProjection.init(
        n_layers=int(p.config.n_layers),
        n_inject_tokens=2,
        carrier_dim=16,
        d_model=int(p.config.d_model),
        seed=99)
    carrier = list(np.linspace(-1.0, 1.0, 16))
    ids = tokenize_bytes("bridge", max_len=8)
    w = bridge_carrier_and_measure(
        params=p, carrier=carrier, projection=proj,
        follow_up_token_ids=ids)
    assert w.max_abs_logit_perturbation > 1e-6


def test_kv_bridge_replay_determinism() -> None:
    p = build_default_tiny_substrate(seed=42)
    proj = KVBridgeProjection.init(
        n_layers=int(p.config.n_layers),
        n_inject_tokens=2,
        carrier_dim=16,
        d_model=int(p.config.d_model),
        seed=99)
    carrier = list(np.linspace(-1.0, 1.0, 16))
    ids = tokenize_bytes("replay", max_len=8)
    w1 = bridge_carrier_and_measure(
        params=p, carrier=carrier, projection=proj,
        follow_up_token_ids=ids)
    w2 = bridge_carrier_and_measure(
        params=p, carrier=carrier, projection=proj,
        follow_up_token_ids=ids)
    assert w1.cid() == w2.cid()


def test_kv_bridge_inject_grows_cache() -> None:
    p = build_default_tiny_substrate(seed=42)
    proj = KVBridgeProjection.init(
        n_layers=int(p.config.n_layers),
        n_inject_tokens=2,
        carrier_dim=8,
        d_model=int(p.config.d_model),
        seed=99)
    cache = TinyKVCache.empty(int(p.config.n_layers))
    assert cache.n_tokens() == 0
    new_cache, record = inject_carrier_into_kv_cache(
        carrier=[0.5] * 8,
        projection=proj,
        kv_cache=cache)
    assert new_cache.n_tokens() == 2


# ---------------------------------------------------------------------------
# V8 persistent
# ---------------------------------------------------------------------------


def test_v8_persistent_chain_walk() -> None:
    cell = V8StackedCell.init(seed=42)
    chain = PersistentLatentStateV8Chain.empty()
    prev = None
    for t in range(10):
        prev = step_persistent_state_v8(
            cell=cell, prev_state=prev,
            carrier_values=[0.1 * t] * cell.state_dim,
            turn_index=t, role="r0")
        chain.add(prev)
    walks = chain.walk_from(prev.cid())
    assert len(walks) == 10


def test_v8_persistent_witness_cid_stable() -> None:
    cell = V8StackedCell.init(seed=42)
    chain = PersistentLatentStateV8Chain.empty()
    s = step_persistent_state_v8(
        cell=cell, prev_state=None,
        carrier_values=[0.5] * cell.state_dim,
        turn_index=0, role="r0")
    chain.add(s)
    w1 = emit_persistent_v8_witness(
        cell=cell, state=s, chain=chain)
    w2 = emit_persistent_v8_witness(
        cell=cell, state=s, chain=chain)
    assert w1.cid() == w2.cid()


# ---------------------------------------------------------------------------
# MLSC V4
# ---------------------------------------------------------------------------


def test_mlsc_v4_round_trip() -> None:
    c1 = make_root_capsule_v3(
        branch_id="a", payload=[0.5] * 6,
        fact_tags=("x",), confidence=0.9,
        trust=0.9, turn_index=0)
    c2 = make_root_capsule_v3(
        branch_id="b", payload=[0.4] * 6,
        fact_tags=("y",), confidence=0.8,
        trust=0.8, turn_index=0)
    v4a = wrap_v3_as_v4(c1, substrate_witness_cid="aa" * 32)
    v4b = wrap_v3_as_v4(c2)
    op = MergeOperatorV4(factor_dim=6)
    merged = op.merge(
        [v4a, v4b], substrate_witness_cid="cc" * 32,
        algebra_signature=W56_MLSC_V4_ALGEBRA_MERGE)
    assert merged.substrate_witness_cid == "cc" * 32
    assert merged.algebra_signature == W56_MLSC_V4_ALGEBRA_MERGE
    w = emit_mlsc_v4_witness(
        capsule=merged, v3_witness_cid="bb" * 32)
    assert w.deepest_provenance_chain >= 1


# ---------------------------------------------------------------------------
# Consensus V2
# ---------------------------------------------------------------------------


def test_consensus_v2_6_stages_defined() -> None:
    assert len(W56_CONSENSUS_V2_STAGES) == 6
    assert (W56_CONSENSUS_V2_STAGE_SUBSTRATE
            in W56_CONSENSUS_V2_STAGES)


def test_consensus_v2_substrate_oracle_picks_stage() -> None:
    def oracle(payloads, qdir):
        return 0
    ctrl = ConsensusFallbackControllerV2(
        k_required=3, cosine_floor=0.99,
        trust_threshold=2.0, substrate_oracle=oracle)
    res = ctrl.decide(
        parent_payloads=[[0.5] * 6, [0.4] * 6],
        parent_trusts=[0.5, 0.4],
        query_direction=[0.5] * 6,
        transcript_payload=[0.0] * 6)
    assert res["stage"] == W56_CONSENSUS_V2_STAGE_SUBSTRATE


# ---------------------------------------------------------------------------
# BCH(31,16) + CRC V4
# ---------------------------------------------------------------------------


def test_bch_31_16_encode_decode_clean() -> None:
    data = 0x1234
    cw = bch_31_16_encode(data)
    assert len(cw) == 31
    data_back, dist, corr = bch_31_16_decode(cw)
    assert data_back == data
    assert dist == 0
    assert corr is True


def test_bch_31_16_corrects_3_bit_flip() -> None:
    data = 0xABCD
    cw = list(bch_31_16_encode(data))
    cw[0] ^= 1
    cw[5] ^= 1
    cw[15] ^= 1
    data_back, dist, corr = bch_31_16_decode(cw)
    assert data_back == data
    assert corr is True


def test_2d_interleave_round_trip() -> None:
    bits = [1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0]
    out = interleave_2d(bits, n_rows=4, n_cols=4)
    back = deinterleave_2d(out, n_rows=4, n_cols=4)
    assert list(back) == list(bits)


# ---------------------------------------------------------------------------
# ECC V8
# ---------------------------------------------------------------------------


def test_ecc_v8_rate_floor_falsifier_reproduces() -> None:
    f = probe_ecc_v8_rate_floor_falsifier(
        target_bits_per_token=128.0, seed=0)
    assert f["rate_target_missed"] is True


# ---------------------------------------------------------------------------
# Deep substrate hybrid
# ---------------------------------------------------------------------------


def test_deep_substrate_hybrid_forward_kv_grows() -> None:
    deep = DeepProxyStackV6.init(seed=42)
    sub = build_default_tiny_substrate(seed=42)
    hyb = DeepSubstrateHybrid.init(
        deep_v6=deep, substrate=sub)
    in_dim = deep.in_dim
    fd = deep.inner_v5.inner_v4.factor_dim
    q = [0.1] * in_dim
    k = [[0.05] * fd for _ in range(2)]
    v = [[0.07] * fd for _ in range(2)]
    _, w, cache = deep_substrate_hybrid_forward(
        hybrid=hyb,
        query_input=q, slot_keys=k, slot_values=v)
    assert cache.n_tokens() >= 1
    assert w.ablation_perturbation_l2 >= 0.0


# ---------------------------------------------------------------------------
# TVS V5
# ---------------------------------------------------------------------------


def test_tvs_v5_six_arms() -> None:
    assert len(W56_TVS_V5_ARMS) == 6


def test_tvs_v5_pick_rates_sum_to_one() -> None:
    res = six_arm_compare(
        per_turn_confidences=[0.5, 0.6, 0.7],
        per_turn_trust_scores=[0.5, 0.6, 0.7],
        per_turn_merge_retentions=[0.5, 0.6, 0.7],
        per_turn_tw_retentions=[0.5, 0.6, 0.7],
        per_turn_substrate_fidelities=[0.6, 0.7, 0.8],
        budget_tokens=4)
    assert abs(float(sum(res.pick_rates.values())) - 1.0) < 1e-9
    w = emit_tvs_arbiter_v5_witness(result=res)
    assert w.n_turns == 3


# ---------------------------------------------------------------------------
# Uncertainty V4
# ---------------------------------------------------------------------------


def test_uncertainty_v4_low_substrate_aware() -> None:
    r_high = compose_uncertainty_report_v4(
        component_confidences={"a": 0.9, "b": 0.8},
        trust_weights={"a": 1.0, "b": 1.0},
        substrate_fidelities={"a": 1.0, "b": 1.0})
    r_low = compose_uncertainty_report_v4(
        component_confidences={"a": 0.9, "b": 0.8},
        trust_weights={"a": 1.0, "b": 1.0},
        substrate_fidelities={"a": 1.0, "b": 0.1})
    assert not r_high.substrate_aware
    assert r_low.substrate_aware


# ---------------------------------------------------------------------------
# Disagreement algebra V2
# ---------------------------------------------------------------------------


def test_disagreement_algebra_v2_identities() -> None:
    trace = AlgebraTrace.empty()
    w = emit_disagreement_algebra_v2_witness(
        trace=trace,
        probe_a=[0.5, 0.3, -0.1, 0.0],
        probe_b=[0.5, 0.3, -0.1, 0.0],
        probe_c=[0.1, 0.2, -0.3, 0.4])
    assert w.idempotent_ok
    assert w.self_cancel_ok


# ---------------------------------------------------------------------------
# Multi-hop V6
# ---------------------------------------------------------------------------


def test_multi_hop_v6_arbiter_abstains_when_no_eligibility() -> None:
    paths = [
        OctBackendChainPath(
            chain=("A", "B"),
            payload=tuple([0.5] * 6),
            confidence=0.9,
            declared_trust=0.5,
            substrate_trust=0.0),
    ]
    pred, info = substrate_trust_weighted_arbitration(
        paths=paths, substrate_trust_floor=0.1)
    assert info["kind"] == "abstain"


def test_multi_hop_v6_oct_chain_evaluator_returns_value() -> None:
    r = evaluate_oct_chain_len7_fidelity(
        n_probes=3, feature_dim=8, seed=42)
    assert 0.0 <= float(r["chain_len_fidelity_mean"]) <= 1.5
