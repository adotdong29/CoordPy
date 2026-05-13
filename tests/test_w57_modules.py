"""W57 module unit tests."""

from __future__ import annotations

import numpy as np

from coordpy.attention_steering_bridge import (
    AttentionSteeringProjection,
    steer_attention_and_measure,
)
from coordpy.consensus_fallback_controller_v3 import (
    ConsensusFallbackControllerV3,
    W57_CONSENSUS_V3_STAGES,
    W57_CONSENSUS_V3_STAGE_LOGIT_LENS,
)
from coordpy.corruption_robust_carrier_v5 import (
    CorruptionRobustCarrierV5,
    deinterleave_3d,
    detect_kv_corruption,
    emit_corruption_robustness_v5_witness,
    interleave_3d,
    kv_cache_fingerprint,
    majority_9_of_13_decode,
    majority_9_of_13_encode,
)
from coordpy.deep_proxy_stack_v6 import DeepProxyStackV6
from coordpy.deep_substrate_hybrid_v2 import (
    DeepSubstrateHybridV2,
    deep_substrate_hybrid_v2_forward,
)
from coordpy.disagreement_algebra import AlgebraTrace
from coordpy.disagreement_algebra_v3 import (
    emit_disagreement_algebra_v3_witness,
)
from coordpy.ecc_codebook_v9 import (
    ECCCodebookV9,
    compress_carrier_ecc_v9,
    probe_ecc_v9_rate_floor_falsifier,
)
from coordpy.hidden_state_bridge import (
    HiddenStateBridgeProjection,
    bridge_hidden_state_and_measure,
)
from coordpy.kv_bridge_v2 import (
    KVBridgeV2Projection,
    bridge_carrier_and_measure_v2,
)
from coordpy.long_horizon_retention_v9 import (
    LongHorizonReconstructionV9Head,
    evaluate_lhr_v9_three_way,
)
from coordpy.mergeable_latent_capsule_v3 import (
    make_root_capsule_v3,
)
from coordpy.mergeable_latent_capsule_v4 import wrap_v3_as_v4
from coordpy.mergeable_latent_capsule_v5 import (
    MergeOperatorV5,
    W57_MLSC_V5_ALGEBRA_SUBSTRATE_PROJECT,
    W57_MLSC_V5_KNOWN_ALGEBRA_SIGNATURES,
    wrap_v4_as_v5,
)
from coordpy.multi_hop_translator_v7 import (
    DecBackendChainPath,
    W57_DEFAULT_MH_V7_BACKENDS,
    W57_DEFAULT_MH_V7_CHAIN_LEN,
    evaluate_dec_chain_len9_fidelity,
    substrate_hidden_trust_arbitration,
)
from coordpy.persistent_latent_v9 import (
    PersistentLatentStateV9Chain,
    V9StackedCell,
    step_persistent_state_v9,
    W57_DEFAULT_V9_N_LAYERS,
    W57_DEFAULT_V9_MAX_CHAIN_WALK_DEPTH,
)
from coordpy.prefix_state_bridge import (
    bridge_prefix_state_and_measure,
)
from coordpy.quantised_compression import QuantisedBudgetGate
from coordpy.ecc_codebook_v5 import (
    W53_DEFAULT_ECC_CODE_DIM,
    W53_DEFAULT_ECC_EMIT_MASK_LEN,
)
from coordpy.substrate_adapter_v2 import (
    W57_SUBSTRATE_TIER_SUBSTRATE_V2_FULL,
    probe_all_v2_adapters,
    probe_synthetic_v2_adapter,
    probe_tiny_substrate_v2_adapter,
)
from coordpy.tiny_substrate_v2 import (
    TinyV2KVCache,
    build_default_tiny_substrate_v2,
    detokenize_bytes_v2,
    extract_prefix_state,
    forward_tiny_substrate_v2,
    tokenize_bytes_v2,
)
from coordpy.transcript_vs_shared_arbiter_v6 import (
    W57_TVS_V6_ARMS,
    seven_arm_compare,
)
from coordpy.uncertainty_layer_v5 import (
    compose_uncertainty_report_v5,
)


# ---------------------------------------------------------------------------
# Tiny substrate V2
# ---------------------------------------------------------------------------


def test_tiny_substrate_v2_forward_determinism() -> None:
    p = build_default_tiny_substrate_v2(seed=42)
    ids = tokenize_bytes_v2("hello-w57", max_len=12)
    t1 = forward_tiny_substrate_v2(p, ids)
    t2 = forward_tiny_substrate_v2(p, ids)
    assert np.array_equal(t1.logits, t2.logits)
    assert t1.kv_cache.cid() == t2.kv_cache.cid()
    assert t1.cid() == t2.cid()


def test_tiny_substrate_v2_kv_cache_reuse() -> None:
    p = build_default_tiny_substrate_v2(seed=42)
    ids = tokenize_bytes_v2("kv-reuse-w57", max_len=20)
    t_full = forward_tiny_substrate_v2(p, ids)
    half = len(ids) // 2
    t_a = forward_tiny_substrate_v2(p, ids[:half])
    t_b = forward_tiny_substrate_v2(
        p, ids[half:], kv_cache=t_a.kv_cache)
    diff = float(np.max(np.abs(
        t_full.logits[-1] - t_b.logits[-1])))
    assert diff < 1e-9


def test_tiny_substrate_v2_causal_mask() -> None:
    p = build_default_tiny_substrate_v2(seed=42)
    ids = tokenize_bytes_v2("causal-w57", max_len=12)
    t = forward_tiny_substrate_v2(p, ids, return_attention=True)
    for attn in t.attn_weights_per_layer:
        H, T_new, T_all = attn.shape
        for i in range(T_new):
            for j in range(i + 1, T_all):
                assert float(np.max(attn[:, i, j])) < 1e-9


def test_tiny_substrate_v2_logit_lens() -> None:
    p = build_default_tiny_substrate_v2(seed=42)
    ids = tokenize_bytes_v2("lens-w57", max_len=8)
    t = forward_tiny_substrate_v2(p, ids)
    assert len(t.per_layer_logit_lens) == p.config.n_layers + 1
    for lens in t.per_layer_logit_lens:
        assert lens.shape == (len(ids), p.config.vocab_size)


def test_tiny_substrate_v2_prefix_state() -> None:
    p = build_default_tiny_substrate_v2(seed=42)
    ids = tokenize_bytes_v2("prefix-w57", max_len=12)
    t = forward_tiny_substrate_v2(p, ids)
    ps = extract_prefix_state(
        t.kv_cache, prefix_len=4,
        source_params_cid=p.cid())
    assert ps.prefix_len == 4
    cache = ps.to_cache()
    assert cache.n_tokens() == 4


def test_tiny_substrate_v2_cache_eviction_lru() -> None:
    p = build_default_tiny_substrate_v2(seed=42)
    ids = tokenize_bytes_v2("evict-lru", max_len=20)
    t = forward_tiny_substrate_v2(p, ids)
    pruned = t.kv_cache.evict_lru(2)
    assert pruned.n_tokens() == t.kv_cache.n_tokens() - 2


def test_tiny_substrate_v2_cache_eviction_weighted() -> None:
    p = build_default_tiny_substrate_v2(seed=42)
    ids = tokenize_bytes_v2("evict-weighted", max_len=20)
    t = forward_tiny_substrate_v2(p, ids)
    n0 = t.kv_cache.n_tokens()
    weights = list(np.linspace(0.0, 1.0, n0))
    pruned = t.kv_cache.evict_weighted(weights, keep=3)
    assert pruned.n_tokens() == 3


def test_tiny_substrate_v2_tokenize_round_trip() -> None:
    text = "hello, world!"
    ids = tokenize_bytes_v2(text, max_len=64)
    back = detokenize_bytes_v2(ids)
    assert back == text


# ---------------------------------------------------------------------------
# Substrate adapter V2
# ---------------------------------------------------------------------------


def test_substrate_adapter_v2_tiers() -> None:
    tiny = probe_tiny_substrate_v2_adapter()
    synth = probe_synthetic_v2_adapter()
    assert tiny.tier == W57_SUBSTRATE_TIER_SUBSTRATE_V2_FULL
    assert synth.tier == "text_only"


def test_substrate_adapter_v2_matrix() -> None:
    m = probe_all_v2_adapters(
        probe_ollama=False, probe_openai=False)
    assert m.has_v2_full() is True


# ---------------------------------------------------------------------------
# KV Bridge V2
# ---------------------------------------------------------------------------


def test_kv_bridge_v2_perturbation() -> None:
    p = build_default_tiny_substrate_v2(seed=42)
    proj = KVBridgeV2Projection.init(
        n_layers=p.config.n_layers,
        n_heads=p.config.n_heads,
        n_inject_tokens=3,
        carrier_dim=12,
        d_head=p.config.d_model // p.config.n_heads)
    rng = np.random.default_rng(42)
    carrier = list(rng.standard_normal(12))
    w = bridge_carrier_and_measure_v2(
        params=p, carrier=carrier, projection=proj,
        follow_up_token_ids=[257])
    assert w.last_logit_l2_perturbation > 1e-6


# ---------------------------------------------------------------------------
# Hidden-State Bridge
# ---------------------------------------------------------------------------


def test_hidden_state_bridge_perturbation() -> None:
    p = build_default_tiny_substrate_v2(seed=42)
    proj = HiddenStateBridgeProjection.init(
        n_layers=p.config.n_layers,
        n_tokens=4,
        carrier_dim=12,
        d_model=p.config.d_model)
    rng = np.random.default_rng(42)
    carrier = list(rng.standard_normal(12))
    w = bridge_hidden_state_and_measure(
        params=p, carrier=carrier, projection=proj,
        target_layer=1,
        token_ids=[257, 104])
    assert w.last_logit_l2_perturbation > 1e-6


# ---------------------------------------------------------------------------
# Prefix-State Bridge
# ---------------------------------------------------------------------------


def test_prefix_state_bridge_reuse_matches_recompute() -> None:
    p = build_default_tiny_substrate_v2(seed=42)
    prompt = tokenize_bytes_v2("prompt", max_len=12)
    follow = [104, 105]
    w = bridge_prefix_state_and_measure(
        params=p,
        prompt_token_ids=prompt,
        follow_up_token_ids=follow)
    assert w.reuse_matches_recompute is True
    assert w.max_abs_reuse_recompute_diff < 1e-9


def test_prefix_state_bridge_corruption_detected() -> None:
    p = build_default_tiny_substrate_v2(seed=42)
    prompt = tokenize_bytes_v2("p2", max_len=12)
    w = bridge_prefix_state_and_measure(
        params=p,
        prompt_token_ids=prompt,
        follow_up_token_ids=[104],
        corrupt_after_save=True,
        corruption_layer=0,
        corruption_position=0,
        corruption_magnitude=2.0)
    assert w.corruption_detected is True


# ---------------------------------------------------------------------------
# Attention Steering Bridge
# ---------------------------------------------------------------------------


def test_attention_steering_kl_positive() -> None:
    p = build_default_tiny_substrate_v2(seed=42)
    ids = tokenize_bytes_v2("attn", max_len=8)
    proj = AttentionSteeringProjection.init(
        n_layers=p.config.n_layers,
        n_heads=p.config.n_heads,
        n_query=len(ids), n_key=len(ids),
        carrier_dim=12)
    rng = np.random.default_rng(42)
    carrier = list(rng.standard_normal(12))
    w = steer_attention_and_measure(
        params=p, carrier=carrier, projection=proj,
        token_ids=ids)
    for kl in w.mean_kl_per_layer:
        assert float(kl) > 0.0
    assert w.attention_pattern_shifted is True


# ---------------------------------------------------------------------------
# Persistent Latent V9
# ---------------------------------------------------------------------------


def test_v9_chain_walk_depth() -> None:
    cell = V9StackedCell.init(seed=57)
    assert cell.n_layers == W57_DEFAULT_V9_N_LAYERS
    chain = PersistentLatentStateV9Chain.empty()
    prev = None
    for i in range(20):
        carrier = [0.1 * (i + j)
                    for j in range(cell.state_dim)]
        st = step_persistent_state_v9(
            cell=cell, prev_state=prev,
            carrier_values=carrier, turn_index=i, role="r0")
        chain.add(st)
        prev = st
    assert len(chain.walk_from(prev.cid())) == 20


def test_v9_carrier_round_trip_deterministic() -> None:
    cell = V9StackedCell.init(seed=57)
    carrier = [0.1] * cell.state_dim
    s1 = step_persistent_state_v9(
        cell=cell, prev_state=None,
        carrier_values=carrier, turn_index=0, role="r0")
    s2 = step_persistent_state_v9(
        cell=cell, prev_state=None,
        carrier_values=carrier, turn_index=0, role="r0")
    assert s1.cid() == s2.cid()


def test_v9_max_chain_walk_depth_constant() -> None:
    assert W57_DEFAULT_V9_MAX_CHAIN_WALK_DEPTH == 384


# ---------------------------------------------------------------------------
# Multi-Hop V7
# ---------------------------------------------------------------------------


def test_multi_hop_v7_constants() -> None:
    assert len(W57_DEFAULT_MH_V7_BACKENDS) == 10
    assert W57_DEFAULT_MH_V7_CHAIN_LEN == 9


def test_multi_hop_v7_chain_len9_fidelity_runs() -> None:
    r = evaluate_dec_chain_len9_fidelity(
        backends=W57_DEFAULT_MH_V7_BACKENDS, feature_dim=8)
    assert r["chain_length"] == 9


def test_multi_hop_v7_arbitration_abstain() -> None:
    out, info = substrate_hidden_trust_arbitration(paths=[])
    assert info["kind"] == "abstain"


def test_multi_hop_v7_arbitration_agreeing_subset() -> None:
    paths = [
        DecBackendChainPath(
            chain=("A", "B"), payload=(1.0, 0.0, 0.0),
            confidence=0.9, substrate_trust=0.9,
            hidden_trust=0.9),
        DecBackendChainPath(
            chain=("A", "C"), payload=(1.0, 0.0, 0.0),
            confidence=0.9, substrate_trust=0.9,
            hidden_trust=0.9),
    ]
    out, info = substrate_hidden_trust_arbitration(
        paths=paths)
    assert info["n_chosen"] == 2


# ---------------------------------------------------------------------------
# MLSC V5
# ---------------------------------------------------------------------------


def test_mlsc_v5_known_signatures() -> None:
    assert W57_MLSC_V5_ALGEBRA_SUBSTRATE_PROJECT in (
        W57_MLSC_V5_KNOWN_ALGEBRA_SIGNATURES)


def test_mlsc_v5_merge_chain_inheritance() -> None:
    op = MergeOperatorV5(factor_dim=6)
    c1 = make_root_capsule_v3(
        branch_id="b1", payload=(0.1,) * 6,
        fact_tags=("t",), confidence=0.9, trust=0.9,
        turn_index=0)
    c2 = make_root_capsule_v3(
        branch_id="b2", payload=(0.2,) * 6,
        fact_tags=("t",), confidence=0.9, trust=0.9,
        turn_index=0)
    v5a = wrap_v4_as_v5(
        wrap_v3_as_v4(c1),
        hidden_state_witness_chain=("h1", "h2"))
    v5b = wrap_v4_as_v5(
        wrap_v3_as_v4(c2),
        hidden_state_witness_chain=("h2", "h3"))
    merged = op.merge([v5a, v5b])
    chain = list(merged.hidden_state_witness_chain)
    assert "h1" in chain
    assert "h2" in chain
    assert "h3" in chain


# ---------------------------------------------------------------------------
# Consensus V3
# ---------------------------------------------------------------------------


def test_consensus_v3_seven_stages() -> None:
    assert len(W57_CONSENSUS_V3_STAGES) == 7
    assert W57_CONSENSUS_V3_STAGE_LOGIT_LENS in (
        W57_CONSENSUS_V3_STAGES)


def test_consensus_v3_logit_lens_fires() -> None:
    ctrl = ConsensusFallbackControllerV3(
        k_required=2, cosine_floor=0.99, trust_threshold=10.0)
    p1 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    p2 = [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6]
    q = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    ctrl.logit_lens_oracle = lambda payloads, qd: 1
    res = ctrl.decide(
        parent_payloads=[p1, p2],
        parent_trusts=[0.1, 0.1],
        query_direction=q,
        transcript_payload=[0.0] * 6)
    assert res["decision_stage"] == W57_CONSENSUS_V3_STAGE_LOGIT_LENS


# ---------------------------------------------------------------------------
# CRC V5
# ---------------------------------------------------------------------------


def test_crc_v5_3d_interleave_round_trip() -> None:
    bits = [1, 0, 1, 0] * 16  # 64 bits
    rt = deinterleave_3d(interleave_3d(bits))
    assert list(rt) == bits


def test_crc_v5_majority_9_of_13() -> None:
    enc = list(majority_9_of_13_encode(1))
    # Flip 4 bits: still > 9 zeros + 4 flipped = need to think.
    # encode(1) -> 13 ones; flip 4 -> 9 ones, 4 zeros.
    # decode: count 1s = 9, threshold 9 → returns 1.
    enc[0] ^= 1
    enc[1] ^= 1
    enc[2] ^= 1
    enc[3] ^= 1
    assert majority_9_of_13_decode(enc) == 1


def test_crc_v5_kv_corruption_detect() -> None:
    a = bytes([1, 2, 3, 4] * 32)
    b = bytes([5, 6, 7, 8] * 32)
    pre = kv_cache_fingerprint(a, b)
    bb = bytearray(b)
    bb[0] ^= 0x37
    post = kv_cache_fingerprint(a, bytes(bb))
    assert detect_kv_corruption(pre, post) is True


def test_crc_v5_witness_runs() -> None:
    v5 = CorruptionRobustCarrierV5()
    w = emit_corruption_robustness_v5_witness(
        crc_v5=v5, n_probes=8, seed=57)
    assert w.three_d_interleave_round_trip_ok is True


# ---------------------------------------------------------------------------
# LHR V9
# ---------------------------------------------------------------------------


def test_lhr_v9_three_way_runs() -> None:
    head = LongHorizonReconstructionV9Head.init(seed=57)
    cd = head.inner_v8.inner_v7.carrier_dim
    od = head.out_dim
    rng = np.random.default_rng(57)
    cs = [list(rng.standard_normal(cd)) for _ in range(3)]
    ts = [list(rng.standard_normal(od)) for _ in range(3)]
    sub = [list(rng.standard_normal(head.hidden_dim))
            for _ in range(3)]
    hid = [list(rng.standard_normal(head.hidden_dim))
            for _ in range(3)]
    r = evaluate_lhr_v9_three_way(
        head,
        carrier_examples=cs, target_examples=ts,
        substrate_states=sub, hidden_states=hid, k=8)
    assert r["n"] == 3


# ---------------------------------------------------------------------------
# ECC V9
# ---------------------------------------------------------------------------


def test_ecc_v9_20_bits_per_token() -> None:
    cb = ECCCodebookV9.init(seed=57)
    gate = QuantisedBudgetGate.init(
        in_dim=W53_DEFAULT_ECC_CODE_DIM,
        emit_mask_len=W53_DEFAULT_ECC_EMIT_MASK_LEN,
        seed=57)
    gate.importance_threshold = 0.0
    gate.w_emit.values = [1.0] * len(gate.w_emit.values)
    rng = np.random.default_rng(57)
    carrier = list(rng.standard_normal(
        W53_DEFAULT_ECC_CODE_DIM))
    comp = compress_carrier_ecc_v9(
        carrier, codebook=cb, gate=gate)
    assert comp["bits_per_visible_token"] >= 20.0


def test_ecc_v9_rate_floor_falsifier() -> None:
    cb = ECCCodebookV9.init(seed=57)
    f = probe_ecc_v9_rate_floor_falsifier(
        codebook=cb, target_bits_per_token=256.0)
    assert f["target_above_info_bound"] is True


# ---------------------------------------------------------------------------
# TVS V6
# ---------------------------------------------------------------------------


def test_tvs_v6_seven_arms() -> None:
    assert len(W57_TVS_V6_ARMS) == 7


def test_tvs_v6_pick_rates_sum_to_1() -> None:
    r = seven_arm_compare(
        per_turn_confidences=[0.7, 0.6, 0.5],
        per_turn_trust_scores=[0.8, 0.7, 0.6],
        per_turn_merge_retentions=[0.6, 0.5, 0.4],
        per_turn_tw_retentions=[0.5, 0.4, 0.3],
        per_turn_substrate_fidelities=[0.6, 0.7, 0.8],
        per_turn_hidden_fidelities=[0.7, 0.6, 0.5],
        budget_tokens=4)
    s = float(sum(r.pick_rates.values()))
    assert abs(s - 1.0) < 1e-9


def test_tvs_v6_hidden_inject_preferred_when_hf_high() -> None:
    r = seven_arm_compare(
        per_turn_confidences=[0.3],
        per_turn_trust_scores=[0.3],
        per_turn_merge_retentions=[0.3],
        per_turn_tw_retentions=[0.3],
        per_turn_substrate_fidelities=[0.4],
        per_turn_hidden_fidelities=[0.9],
        budget_tokens=4)
    assert r.hidden_inject_used is True


# ---------------------------------------------------------------------------
# Uncertainty V5
# ---------------------------------------------------------------------------


def test_uncertainty_v5_bracket() -> None:
    r = compose_uncertainty_report_v5(
        component_confidences={"a": 0.7, "b": 0.6},
        trust_weights={"a": 0.8, "b": 0.7},
        substrate_fidelities={"a": 0.9, "b": 0.85},
        hidden_state_fidelities={"a": 0.85, "b": 0.8},
        adversarial_radius=0.05)
    assert (
        r.pessimistic_composite
        <= r.weighted_composite + 1e-9
        <= r.optimistic_composite + 1e-9)


# ---------------------------------------------------------------------------
# Disagreement Algebra V3
# ---------------------------------------------------------------------------


def test_disagreement_algebra_v3_witness_runs() -> None:
    w = emit_disagreement_algebra_v3_witness(
        trace=AlgebraTrace.empty(),
        probe_a=[0.1, 0.2, 0.3, 0.4],
        probe_b=[0.5, 0.6, 0.7, 0.8],
        probe_c=[0.9, 1.0, 1.1, 1.2],
        hidden_state_projector=lambda x: list(x))
    assert w.merge_idempotent_ok is True
    assert w.diff_self_cancel_ok is True
    assert w.hidden_projection_ok is True


# ---------------------------------------------------------------------------
# Deep substrate hybrid V2
# ---------------------------------------------------------------------------


def test_deep_substrate_hybrid_v2_bidirectional() -> None:
    p = build_default_tiny_substrate_v2(seed=57)
    deep = DeepProxyStackV6.init(seed=571)
    hyb = DeepSubstrateHybridV2.init(
        deep_v6=deep, substrate=p,
        n_inject_tokens=3,
        carrier_dim=int(deep.inner_v5.inner_v4.factor_dim),
        substrate_back_inject_weight=0.10)
    in_dim = int(deep.in_dim)
    fd = int(deep.inner_v5.inner_v4.factor_dim)
    q = [0.1] * in_dim
    k = [[0.05] * fd, [0.07] * fd]
    v = [[0.06] * fd, [0.08] * fd]
    _, w, _ = deep_substrate_hybrid_v2_forward(
        hybrid=hyb, query_input=q,
        slot_keys=k, slot_values=v)
    assert w.bidirectional is True
    assert w.substrate_back_l2 > 0.0
    assert w.ablation_perturbation_l2 > 0.0
