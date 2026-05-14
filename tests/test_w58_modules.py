"""W58 module unit tests."""

from __future__ import annotations

import numpy as np

from coordpy.attention_steering_bridge_v2 import (
    AttentionSteeringV2Projection,
    steer_attention_and_measure_v2,
)
from coordpy.cache_controller import (
    CacheController, apply_cache_controller_and_measure,
    W58_CACHE_POLICY_IMPORTANCE, W58_CACHE_POLICY_UNIFORM,
    fit_learned_cache_controller,
)
from coordpy.consensus_fallback_controller_v4 import (
    ConsensusFallbackControllerV4,
    W58_CONSENSUS_V4_STAGES,
    W58_CONSENSUS_V4_STAGE_CACHE_REUSE,
)
from coordpy.corruption_robust_carrier_v6 import (
    CorruptionRobustCarrierV6,
    emit_corruption_robustness_v6_witness,
)
from coordpy.deep_proxy_stack_v6 import DeepProxyStackV6
from coordpy.deep_substrate_hybrid_v3 import (
    DeepSubstrateHybridV3,
    deep_substrate_hybrid_v3_forward,
)
from coordpy.disagreement_algebra import AlgebraTrace
from coordpy.disagreement_algebra_v4 import (
    emit_disagreement_algebra_v4_witness,
)
from coordpy.ecc_codebook_v10 import (
    ECCCodebookV10, compress_carrier_ecc_v10,
    probe_ecc_v10_rate_floor_falsifier,
)
from coordpy.ecc_codebook_v5 import (
    W53_DEFAULT_ECC_CODE_DIM,
)
from coordpy.hidden_state_bridge_v2 import (
    HiddenStateBridgeV2Projection,
    bridge_hidden_state_and_measure_v2,
)
from coordpy.kv_bridge_v3 import (
    KVBridgeV3Projection, bridge_carrier_and_measure_v3,
)
from coordpy.long_horizon_retention_v10 import (
    LongHorizonReconstructionV10Head,
    evaluate_lhr_v10_four_way,
)
from coordpy.mergeable_latent_capsule_v3 import (
    make_root_capsule_v3,
)
from coordpy.mergeable_latent_capsule_v4 import wrap_v3_as_v4
from coordpy.mergeable_latent_capsule_v5 import wrap_v4_as_v5
from coordpy.mergeable_latent_capsule_v6 import (
    MergeOperatorV6, emit_mlsc_v6_witness, wrap_v5_as_v6,
)
from coordpy.multi_hop_translator_v8 import (
    W58_DEFAULT_MH_V8_BACKENDS,
    W58_DEFAULT_MH_V8_CHAIN_LEN,
    evaluate_dec_chain_len11_fidelity,
)
from coordpy.persistent_latent_v10 import (
    PersistentLatentStateV10Chain, V10StackedCell,
    W58_DEFAULT_V10_MAX_CHAIN_WALK_DEPTH,
    step_persistent_state_v10,
)
from coordpy.prefix_state_bridge_v2 import (
    bridge_prefix_state_and_measure_v2,
)
from coordpy.quantised_compression import QuantisedBudgetGate
from coordpy.substrate_adapter_v3 import (
    W58_SUBSTRATE_TIER_SUBSTRATE_V3_FULL,
    probe_all_v3_adapters,
    probe_synthetic_v3_adapter,
    probe_tiny_substrate_v3_adapter,
)
from coordpy.tiny_substrate_v3 import (
    TinyV3KVCache, build_default_tiny_substrate_v3,
    detokenize_bytes_v3, extract_prefix_state_v3,
    forward_tiny_substrate_v3, tokenize_bytes_v3,
    partial_forward_layers_l_to_end_v3,
)
from coordpy.transcript_vs_shared_arbiter_v7 import (
    W58_TVS_V7_ARMS, eight_arm_compare,
)
from coordpy.uncertainty_layer_v6 import (
    compose_uncertainty_report_v6,
)
from coordpy.w58_team import (
    W58Params, W58_ENVELOPE_VERIFIER_FAILURE_MODES,
    build_w58_team, build_trivial_w58_envelope,
    verify_w58_handoff,
)


# ---------------------------------------------------------------------------
# Tiny substrate V3
# ---------------------------------------------------------------------------


def test_tiny_substrate_v3_forward_determinism() -> None:
    p = build_default_tiny_substrate_v3(seed=42)
    ids = tokenize_bytes_v3("hello-w58", max_len=12)
    t1 = forward_tiny_substrate_v3(p, ids)
    t2 = forward_tiny_substrate_v3(p, ids)
    assert np.array_equal(t1.logits, t2.logits)
    assert t1.kv_cache.cid() == t2.kv_cache.cid()
    assert t1.cid() == t2.cid()


def test_tiny_substrate_v3_kv_cache_reuse() -> None:
    p = build_default_tiny_substrate_v3(seed=42)
    ids = tokenize_bytes_v3("kv-reuse-w58", max_len=20)
    t_full = forward_tiny_substrate_v3(p, ids)
    half = len(ids) // 2
    t_a = forward_tiny_substrate_v3(p, ids[:half])
    t_b = forward_tiny_substrate_v3(
        p, ids[half:], kv_cache=t_a.kv_cache)
    diff = float(np.max(np.abs(
        t_full.logits[-1] - t_b.logits[-1])))
    assert diff < 1e-9


def test_tiny_substrate_v3_causal_mask() -> None:
    p = build_default_tiny_substrate_v3(seed=42)
    ids = tokenize_bytes_v3("causal-w58", max_len=10)
    t = forward_tiny_substrate_v3(p, ids, return_attention=True)
    for attn in t.attn_weights_per_layer:
        H, T_new, T_all = attn.shape
        for i in range(T_new):
            for j in range(i + 1, T_all):
                assert float(np.max(attn[:, i, j])) < 1e-9


def test_tiny_substrate_v3_gqa_cache_smaller() -> None:
    p = build_default_tiny_substrate_v3(seed=42)
    cfg = p.config
    d_head = int(cfg.d_model) // int(cfg.n_heads)
    d_kv = int(cfg.n_kv_heads) * int(d_head)
    assert d_kv < int(cfg.d_model)


def test_tiny_substrate_v3_logit_lens() -> None:
    p = build_default_tiny_substrate_v3(seed=42)
    ids = tokenize_bytes_v3("lens-w58", max_len=8)
    t = forward_tiny_substrate_v3(p, ids)
    assert len(t.per_layer_logit_lens) == p.config.n_layers + 1


def test_tiny_substrate_v3_flop_count() -> None:
    p = build_default_tiny_substrate_v3(seed=42)
    ids = tokenize_bytes_v3("flops-w58", max_len=10)
    t = forward_tiny_substrate_v3(p, ids)
    assert t.flop_count > 0
    assert sum(t.flop_count_per_layer) <= t.flop_count


def test_tiny_substrate_v3_importance_vectors_per_layer() -> None:
    p = build_default_tiny_substrate_v3(seed=42)
    ids = tokenize_bytes_v3("imp-w58", max_len=10)
    t = forward_tiny_substrate_v3(p, ids)
    assert len(t.kv_importance_per_layer) == p.config.n_layers
    for imp in t.kv_importance_per_layer:
        assert imp.shape[0] == len(ids)


def test_tiny_substrate_v3_partial_forward() -> None:
    p = build_default_tiny_substrate_v3(seed=42)
    ids = tokenize_bytes_v3("partial", max_len=8)
    t_full = forward_tiny_substrate_v3(p, ids)
    h0 = t_full.hidden_states[1]
    positions = np.arange(len(ids), dtype=np.int64)
    t_suffix = partial_forward_layers_l_to_end_v3(
        p, residual_at_layer_l=h0,
        start_layer=1, positions=positions)
    assert t_suffix.logits.shape == t_full.logits.shape


def test_tiny_substrate_v3_prefix_state_extract() -> None:
    p = build_default_tiny_substrate_v3(seed=42)
    ids = tokenize_bytes_v3("prefix", max_len=12)
    t = forward_tiny_substrate_v3(p, ids)
    ps = extract_prefix_state_v3(
        t.kv_cache, prefix_len=len(ids),
        source_params_cid=str(p.cid()))
    assert ps.prefix_len == len(ids)


# ---------------------------------------------------------------------------
# KV bridge V3
# ---------------------------------------------------------------------------


def test_kv_bridge_v3_perturbs() -> None:
    p = build_default_tiny_substrate_v3(seed=42)
    proj = KVBridgeV3Projection.init(
        n_layers=p.config.n_layers,
        n_heads=p.config.n_heads,
        n_kv_heads=p.config.n_kv_heads,
        n_inject_tokens=3,
        carrier_dim=16,
        d_head=p.config.d_model // p.config.n_heads,
        seed=12345)
    rng = np.random.default_rng(7)
    carrier = list(rng.standard_normal(16).tolist())
    ids = tokenize_bytes_v3("w58-kv", max_len=10)
    w = bridge_carrier_and_measure_v3(
        params=p, carrier=carrier, projection=proj,
        follow_up_token_ids=ids)
    assert w.last_logit_l2_perturbation > 0.01


def test_kv_bridge_v3_fits_target() -> None:
    p = build_default_tiny_substrate_v3(seed=42)
    proj = KVBridgeV3Projection.init(
        n_layers=p.config.n_layers,
        n_heads=p.config.n_heads,
        n_kv_heads=p.config.n_kv_heads,
        n_inject_tokens=3, carrier_dim=16,
        d_head=p.config.d_model // p.config.n_heads,
        seed=12345)
    rng = np.random.default_rng(7)
    carrier = list(rng.standard_normal(16).tolist())
    ids = tokenize_bytes_v3("w58-kv", max_len=10)
    w_un = bridge_carrier_and_measure_v3(
        params=p, carrier=carrier, projection=proj,
        follow_up_token_ids=ids)
    target = 5.0
    w_fi = bridge_carrier_and_measure_v3(
        params=p, carrier=carrier, projection=proj,
        follow_up_token_ids=ids,
        target_l2_perturbation=target)
    err_un = abs(w_un.last_logit_l2_perturbation - target)
    err_fi = abs(w_fi.last_logit_l2_perturbation - target)
    assert err_fi <= err_un + 1e-9
    assert w_fi.fitted_scale_used


def test_kv_bridge_v3_banks_distinct() -> None:
    p = build_default_tiny_substrate_v3(seed=42)
    proj = KVBridgeV3Projection.init(
        n_layers=p.config.n_layers,
        n_heads=p.config.n_heads,
        n_kv_heads=p.config.n_kv_heads,
        n_inject_tokens=3, carrier_dim=16,
        d_head=p.config.d_model // p.config.n_heads,
        seed=12345)
    rng = np.random.default_rng(7)
    carrier = list(rng.standard_normal(16).tolist())
    ids = tokenize_bytes_v3("w58-kv", max_len=10)
    w_a = bridge_carrier_and_measure_v3(
        params=p, carrier=carrier, projection=proj,
        follow_up_token_ids=ids, role="bank_a")
    w_b = bridge_carrier_and_measure_v3(
        params=p, carrier=carrier, projection=proj,
        follow_up_token_ids=ids, role="bank_b")
    assert (abs(w_a.last_logit_l2_perturbation
                 - w_b.last_logit_l2_perturbation) > 1e-3)


# ---------------------------------------------------------------------------
# HSB V2
# ---------------------------------------------------------------------------


def test_hsb_v2_multi_layer_perturbs() -> None:
    p = build_default_tiny_substrate_v3(seed=42)
    ids = tokenize_bytes_v3("hsb-w58", max_len=10)
    proj = HiddenStateBridgeV2Projection.init(
        target_layers=(1, 3),
        n_tokens=len(ids),
        carrier_dim=16,
        d_model=p.config.d_model,
        seed=9999)
    rng = np.random.default_rng(7)
    carrier = list(rng.standard_normal(16).tolist())
    w = bridge_hidden_state_and_measure_v2(
        params=p, carrier=carrier, projection=proj,
        token_ids=ids)
    assert w.last_logit_l2_perturbation > 0.01
    assert len(w.per_layer_residual_delta_l2) == (
        p.config.n_layers + 1)


# ---------------------------------------------------------------------------
# Prefix state V2
# ---------------------------------------------------------------------------


def test_prefix_state_v2_reuse_byte_identical() -> None:
    p = build_default_tiny_substrate_v3(seed=42)
    prompt = tokenize_bytes_v3("prefix-w58", max_len=18)
    follow = tokenize_bytes_v3("follow", max_len=10)
    w = bridge_prefix_state_and_measure_v2(
        params=p, prompt_token_ids=prompt,
        follow_up_token_ids=follow)
    assert w.reuse_matches_recompute
    assert w.max_abs_reuse_recompute_diff < 1e-9


def test_prefix_state_v2_flop_saved_positive() -> None:
    p = build_default_tiny_substrate_v3(seed=42)
    prompt = tokenize_bytes_v3("prefix-w58", max_len=20)
    follow = tokenize_bytes_v3("follow", max_len=12)
    w = bridge_prefix_state_and_measure_v2(
        params=p, prompt_token_ids=prompt,
        follow_up_token_ids=follow)
    assert w.flop_saved > 0
    assert w.flop_savings_ratio > 0.0


def test_prefix_state_v2_corruption_detected() -> None:
    p = build_default_tiny_substrate_v3(seed=42)
    prompt = tokenize_bytes_v3("corrupt-prefix", max_len=16)
    follow = tokenize_bytes_v3("fu", max_len=8)
    w = bridge_prefix_state_and_measure_v2(
        params=p, prompt_token_ids=prompt,
        follow_up_token_ids=follow,
        corrupt_after_save=True,
        corruption_layer=1, corruption_position=2,
        corruption_magnitude=1.0)
    assert w.corruption_detected


# ---------------------------------------------------------------------------
# Attention steering V2
# ---------------------------------------------------------------------------


def test_attn_v2_kl_budget_enforced() -> None:
    p = build_default_tiny_substrate_v3(seed=42)
    ids = tokenize_bytes_v3("attn-w58", max_len=10)
    proj = AttentionSteeringV2Projection.init(
        n_layers=p.config.n_layers, n_heads=p.config.n_heads,
        n_query=len(ids), n_key=len(ids), carrier_dim=16,
        seed=12345)
    rng = np.random.default_rng(7)
    carrier = list(rng.standard_normal(16).tolist())
    w = steer_attention_and_measure_v2(
        params=p, carrier=carrier, projection=proj,
        token_ids=ids, kl_budget=1.0)
    assert w.kl_budget_enforced
    assert max(w.mean_kl_per_layer) <= 1.0 * 1.001


# ---------------------------------------------------------------------------
# Cache controller
# ---------------------------------------------------------------------------


def test_cache_controller_uniform_argmax_at_full_retention() -> None:
    p = build_default_tiny_substrate_v3(seed=42)
    prompt = tokenize_bytes_v3("cc", max_len=20)
    follow = tokenize_bytes_v3("fu", max_len=8)
    ctrl = CacheController.init(
        policy=W58_CACHE_POLICY_IMPORTANCE,
        d_model=p.config.d_model)
    w = apply_cache_controller_and_measure(
        p, ctrl, prompt_token_ids=prompt,
        follow_up_token_ids=follow,
        retention_ratio=1.0)
    assert w.argmax_preserved


def test_cache_controller_flop_savings_at_low_retention() -> None:
    p = build_default_tiny_substrate_v3(seed=42)
    prompt = tokenize_bytes_v3("cc", max_len=20)
    follow = tokenize_bytes_v3("fu", max_len=8)
    ctrl = CacheController.init(
        policy=W58_CACHE_POLICY_UNIFORM,
        d_model=p.config.d_model)
    w = apply_cache_controller_and_measure(
        p, ctrl, prompt_token_ids=prompt,
        follow_up_token_ids=follow,
        retention_ratio=0.5)
    assert w.flop_saved > 0


def test_cache_controller_learned_fit_completes() -> None:
    p = build_default_tiny_substrate_v3(seed=42)
    prompt = tokenize_bytes_v3("learn-cc", max_len=16)
    follow = tokenize_bytes_v3("fu", max_len=6)
    ctrl = fit_learned_cache_controller(
        p, prompt_token_ids=prompt,
        follow_up_token_ids=follow)
    assert ctrl.scoring_head is not None
    assert ctrl.scoring_head.shape == (p.config.d_model,)


# ---------------------------------------------------------------------------
# Persistent V10
# ---------------------------------------------------------------------------


def test_persistent_v10_512_chain_walk() -> None:
    cell = V10StackedCell.init(
        state_dim=8, input_dim=8, n_layers=8, seed=42)
    chain = PersistentLatentStateV10Chain.empty()
    prev = None
    for i in range(512):
        state = step_persistent_state_v10(
            cell=cell, prev_state=prev,
            carrier_values=[float(i % 7 - 3)] * 8,
            turn_index=i, role="r")
        chain.add(state)
        prev = state
    walked = chain.walk_from(prev.cid())
    assert len(walked) == 512
    assert cell.n_layers == 8
    assert (W58_DEFAULT_V10_MAX_CHAIN_WALK_DEPTH == 512)


def test_persistent_v10_round_trip_deterministic() -> None:
    cell = V10StackedCell.init(
        state_dim=8, input_dim=8, n_layers=8, seed=42)
    s1 = step_persistent_state_v10(
        cell=cell, prev_state=None,
        carrier_values=[0.2] * 8,
        turn_index=0, role="r")
    s2 = step_persistent_state_v10(
        cell=cell, prev_state=None,
        carrier_values=[0.2] * 8,
        turn_index=0, role="r")
    assert s1.cid() == s2.cid()


# ---------------------------------------------------------------------------
# Deep substrate hybrid V3
# ---------------------------------------------------------------------------


def test_deep_hybrid_v3_three_way() -> None:
    p = build_default_tiny_substrate_v3(seed=42)
    v6 = DeepProxyStackV6.init(seed=42)
    hybrid = DeepSubstrateHybridV3.init(
        deep_v6=v6, substrate=p,
        n_inject_tokens=3,
        carrier_dim=v6.inner_v5.inner_v4.factor_dim,
        substrate_back_inject_weight=0.1,
        cache_retention_ratio=0.6)
    rng = np.random.default_rng(42)
    qi = list(rng.standard_normal(v6.in_dim).tolist())
    sks = [list(rng.standard_normal(v6.in_dim).tolist())
           for _ in range(4)]
    svs = [list(rng.standard_normal(v6.in_dim).tolist())
           for _ in range(4)]
    out, w, cache = deep_substrate_hybrid_v3_forward(
        hybrid=hybrid, query_input=qi,
        slot_keys=sks, slot_values=svs)
    assert w.three_way
    assert w.substrate_back_l2 > 0.0
    assert w.ablation_perturbation_l2 > 0.0


# ---------------------------------------------------------------------------
# Multi-hop V8
# ---------------------------------------------------------------------------


def test_multi_hop_v8_chain_len_11() -> None:
    w = evaluate_dec_chain_len11_fidelity(seed=4242)
    assert w.chain_length == 11
    assert w.n_edges == 132
    assert len(W58_DEFAULT_MH_V8_BACKENDS) == 12


# ---------------------------------------------------------------------------
# MLSC V6
# ---------------------------------------------------------------------------


def test_mlsc_v6_attention_chain_inheritance() -> None:
    op = MergeOperatorV6(factor_dim=6)
    c1_v3 = make_root_capsule_v3(
        branch_id="b1", payload=(0.1,) * 6,
        fact_tags=("t",), confidence=0.9, trust=0.9,
        turn_index=0)
    c2_v3 = make_root_capsule_v3(
        branch_id="b2", payload=(0.2,) * 6,
        fact_tags=("t",), confidence=0.85, trust=0.85,
        turn_index=0)
    v4a = wrap_v3_as_v4(c1_v3)
    v4b = wrap_v3_as_v4(c2_v3)
    v5a = wrap_v4_as_v5(v4a)
    v5b = wrap_v4_as_v5(v4b)
    v6a = wrap_v5_as_v6(
        v5a, attention_witness_chain=("a1", "a2"))
    v6b = wrap_v5_as_v6(
        v5b, attention_witness_chain=("a2", "a3"))
    merged = op.merge([v6a, v6b])
    chain = list(merged.attention_witness_chain)
    assert "a1" in chain
    assert "a2" in chain
    assert "a3" in chain


# ---------------------------------------------------------------------------
# Consensus V4
# ---------------------------------------------------------------------------


def test_consensus_v4_8_stages() -> None:
    assert len(W58_CONSENSUS_V4_STAGES) == 8


def test_consensus_v4_cache_reuse_fires() -> None:
    ctrl = ConsensusFallbackControllerV4(
        k_required=2, cosine_floor=0.99,
        trust_threshold=10.0)
    rng = np.random.default_rng(42)
    p1 = rng.standard_normal(6).tolist()
    p2 = rng.standard_normal(6).tolist()
    q = rng.standard_normal(6).tolist()
    ctrl.cache_reuse_oracle = (
        lambda payloads, qd, fps: 1)
    res = ctrl.decide(
        parent_payloads=[p1, p2],
        parent_trusts=[0.1, 0.1],
        parent_cache_fingerprints=[(1, 2), (3, 4)],
        query_direction=q,
        transcript_payload=[0.0] * 6)
    assert (res["decision_stage"]
            == W58_CONSENSUS_V4_STAGE_CACHE_REUSE)


# ---------------------------------------------------------------------------
# CRC V6
# ---------------------------------------------------------------------------


def test_crc_v6_kv64_detect_rate_high() -> None:
    crc6 = CorruptionRobustCarrierV6()
    w = emit_corruption_robustness_v6_witness(
        crc_v6=crc6, n_probes=16, seed=42)
    assert w.kv64_corruption_detect_rate >= 0.95


# ---------------------------------------------------------------------------
# LHR V10
# ---------------------------------------------------------------------------


def test_lhr_v10_four_way_runs() -> None:
    head = LongHorizonReconstructionV10Head.init(seed=42)
    rng = np.random.default_rng(42)
    carriers = [rng.standard_normal(8).tolist()
                for _ in range(4)]
    targets = [rng.standard_normal(head.out_dim).tolist()
               for _ in range(4)]
    subs = [rng.standard_normal(8).tolist() for _ in range(4)]
    hids = [rng.standard_normal(8).tolist() for _ in range(4)]
    atts = [rng.standard_normal(8).tolist() for _ in range(4)]
    res = evaluate_lhr_v10_four_way(
        head, carrier_examples=carriers,
        target_examples=targets,
        substrate_states=subs, hidden_states=hids,
        attention_states=atts, k=4)
    assert res["n"] == 4
    assert head.max_k == 72


# ---------------------------------------------------------------------------
# ECC V10
# ---------------------------------------------------------------------------


def test_ecc_v10_bits_per_token_meets_target() -> None:
    cb = ECCCodebookV10.init(seed=42)
    gate = QuantisedBudgetGate.init()
    gate.importance_threshold = 0.0
    gate.w_emit.values = [1.0] * len(gate.w_emit.values)
    rng = np.random.default_rng(42)
    carrier = rng.standard_normal(W53_DEFAULT_ECC_CODE_DIM).tolist()
    comp = compress_carrier_ecc_v10(
        carrier, codebook=cb, gate=gate)
    assert comp["bits_per_visible_token"] >= 21.0


def test_ecc_v10_total_codes() -> None:
    cb = ECCCodebookV10.init(seed=42)
    assert cb.total_codes == 524288


def test_ecc_v10_rate_floor_falsifier() -> None:
    cb = ECCCodebookV10.init(seed=42)
    res = probe_ecc_v10_rate_floor_falsifier(
        codebook=cb, target_bits_per_token=1024.0)
    assert res["target_above_info_bound"]


# ---------------------------------------------------------------------------
# TVS V7
# ---------------------------------------------------------------------------


def test_tvs_v7_eight_arms() -> None:
    assert len(W58_TVS_V7_ARMS) == 8


def test_tvs_v7_cache_reuse_dominates_when_high() -> None:
    cmp = eight_arm_compare(
        per_turn_confidences=[0.5, 0.5, 0.5],
        per_turn_trust_scores=[0.5, 0.5, 0.5],
        per_turn_merge_retentions=[0.5, 0.5, 0.5],
        per_turn_tw_retentions=[0.5, 0.5, 0.5],
        per_turn_substrate_fidelities=[0.4, 0.3, 0.4],
        per_turn_hidden_fidelities=[0.3, 0.4, 0.3],
        per_turn_cache_fidelities=[0.9, 0.9, 0.9])
    assert cmp.pick_rates["cache_reuse_replay"] == 1.0


# ---------------------------------------------------------------------------
# Uncertainty V6
# ---------------------------------------------------------------------------


def test_uncertainty_v6_cache_aware() -> None:
    c = compose_uncertainty_report_v6(
        component_confidences={"a": 0.8, "b": 0.6},
        trust_weights={"a": 0.9, "b": 0.7},
        substrate_fidelities={"a": 0.7, "b": 0.5},
        hidden_state_fidelities={"a": 0.8, "b": 0.4},
        cache_reuse_fidelities={"a": 0.9, "b": 0.3})
    assert c.cache_aware


def test_uncertainty_v6_bracket_holds() -> None:
    c = compose_uncertainty_report_v6(
        component_confidences={"a": 0.7, "b": 0.5},
        trust_weights={"a": 0.9, "b": 0.7},
        substrate_fidelities={"a": 0.8, "b": 0.6},
        hidden_state_fidelities={"a": 0.7, "b": 0.5},
        cache_reuse_fidelities={"a": 0.9, "b": 0.6},
        adversarial_radius=0.1)
    assert (c.pessimistic_composite
            <= c.weighted_composite + 1e-9)
    assert (c.weighted_composite
            <= c.optimistic_composite + 1e-9)


# ---------------------------------------------------------------------------
# Disagreement Algebra V4
# ---------------------------------------------------------------------------


def test_disagreement_algebra_v4_cache_identity() -> None:
    trace = AlgebraTrace.empty()
    pa = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    pb = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    pc = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    w = emit_disagreement_algebra_v4_witness(
        trace=trace,
        probe_a=pa, probe_b=pb, probe_c=pc,
        cache_reuse_oracle=lambda: (0.0, True))
    assert w.cache_reuse_equiv_ok


# ---------------------------------------------------------------------------
# Substrate adapter V3
# ---------------------------------------------------------------------------


def test_substrate_adapter_v3_tiers() -> None:
    tv3 = probe_tiny_substrate_v3_adapter()
    syn = probe_synthetic_v3_adapter()
    assert tv3.tier == W58_SUBSTRATE_TIER_SUBSTRATE_V3_FULL
    assert syn.tier == "text_only"


def test_substrate_adapter_v3_matrix_has_v3_full() -> None:
    matrix = probe_all_v3_adapters(
        probe_ollama=False, probe_openai=False)
    assert matrix.has_v3_full()


# ---------------------------------------------------------------------------
# W58 team / envelope
# ---------------------------------------------------------------------------


def test_w58_envelope_failure_modes_disjoint_and_at_least_45(
) -> None:
    modes = W58_ENVELOPE_VERIFIER_FAILURE_MODES
    assert len(modes) >= 45
    assert len(modes) == len(set(modes))


def test_w58_envelope_clean_verify_ok() -> None:
    team = build_w58_team(seed=42)
    env = team.step(turn_index=0, role="r",
                     w57_outer_cid="abc123")
    ver = verify_w58_handoff(env)
    assert ver["ok"]
    assert ver["failures"] == []


def test_w58_envelope_outer_cid_stable_across_runs() -> None:
    team1 = build_w58_team(seed=42)
    team2 = build_w58_team(seed=42)
    env1 = team1.step(turn_index=0, role="r",
                       w57_outer_cid="abc")
    env2 = team2.step(turn_index=0, role="r",
                       w57_outer_cid="abc")
    assert env1.cid() == env2.cid()


def test_w58_trivial_passthrough_w57_outer_cid_carries() -> None:
    triv = build_trivial_w58_envelope(w57_outer_cid="w57_x")
    assert triv.w57_outer_cid == "w57_x"
    assert not triv.substrate_v3_used


def test_w58_substrate_v3_used_flag_true_in_real_run() -> None:
    team = build_w58_team(seed=42)
    env = team.step(turn_index=0, role="r",
                     w57_outer_cid="abc")
    assert env.substrate_v3_used


def test_w58_three_way_used_flag_true_in_real_run() -> None:
    team = build_w58_team(seed=42)
    env = team.step(turn_index=0, role="r",
                     w57_outer_cid="abc")
    assert env.three_way_used
