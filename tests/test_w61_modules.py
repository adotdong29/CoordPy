"""W61 module focused tests (post-W60 milestone).

Per-module sanity tests for the W61 line. The R-128/R-129/R-130
suites separately exercise the H-bars end-to-end.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_tiny_substrate_v6_determinism_and_axes():
    from coordpy.tiny_substrate_v6 import (
        build_default_tiny_substrate_v6,
        forward_tiny_substrate_v6,
        logit_jacobian_v6,
        record_hidden_write_v6,
        tokenize_bytes_v6,
        emit_tiny_substrate_v6_forward_witness,
    )
    p = build_default_tiny_substrate_v6(seed=611)
    ids = tokenize_bytes_v6("w61-test", max_len=12)
    t1, c1 = forward_tiny_substrate_v6(p, ids)
    t2, c2 = forward_tiny_substrate_v6(p, ids)
    assert t1.cid() == t2.cid()
    # Cache keys axis.
    d_key = p.config.d_key
    assert all(
        k.shape == (c1.n_tokens(), d_key)
        for k in c1.cache_keys)
    # Hidden write trace.
    pre = float(c1.hidden_write_trace.sum())
    record_hidden_write_v6(
        c1, layer_index=2, per_head_l2=[0.5] * p.config.n_heads)
    post = float(c1.hidden_write_trace.sum())
    assert post > pre + 1e-9
    # Replay age increments on second forward.
    t3, c3 = forward_tiny_substrate_v6(
        p, ids[:3], v6_kv_cache=c1)
    assert int(c3.forward_count) > int(c1.forward_count)
    # Logit Jacobian table shape.
    j = logit_jacobian_v6(t1, p, target_token=10)
    arr = np.asarray(j["per_layer_per_head"])
    assert arr.shape == (p.config.n_layers, p.config.n_heads)
    # Witness.
    w = emit_tiny_substrate_v6_forward_witness(t1, c1)
    assert w.n_layers == p.config.n_layers
    assert w.n_heads == p.config.n_heads


def test_kv_bridge_v6_multi_target_and_attention_pattern():
    from coordpy.kv_bridge_v3 import KVBridgeV3Projection
    from coordpy.kv_bridge_v4 import KVBridgeV4Projection
    from coordpy.kv_bridge_v5 import KVBridgeV5Projection
    from coordpy.kv_bridge_v6 import (
        KVBridgeV6Projection,
        bridge_carrier_and_measure_v6,
        fit_kv_bridge_v6_attention_pattern,
        fit_kv_bridge_v6_multi_target,
        v6_cache_key_fingerprint,
    )
    from coordpy.tiny_substrate_v6 import (
        build_default_tiny_substrate_v6,
        forward_tiny_substrate_v6, tokenize_bytes_v6,
    )
    p = build_default_tiny_substrate_v6(seed=612)
    v3p = p.v3_params
    proj_v3 = KVBridgeV3Projection.init(
        n_layers=p.config.n_layers, n_heads=p.config.n_heads,
        n_kv_heads=p.config.n_kv_heads,
        n_inject_tokens=3, carrier_dim=8,
        d_head=p.config.d_model // p.config.n_heads,
        seed=61200)
    proj_v4 = KVBridgeV4Projection.init_from_v3(
        proj_v3, seed_v4=61201)
    proj_v5 = KVBridgeV5Projection.init_from_v4(
        proj_v4, seed_v5=61202)
    proj_v6 = KVBridgeV6Projection.init_from_v5(
        proj_v5, seed_v6=61203)
    rng = np.random.default_rng(0)
    ids = tokenize_bytes_v6("kv6", max_len=4)
    carriers = [list(rng.standard_normal(8)) for _ in range(4)]
    t1 = np.zeros(p.config.vocab_size); t1[100] = 1.0
    t2 = np.zeros(p.config.vocab_size); t2[120] = 0.5
    fitted, report = fit_kv_bridge_v6_multi_target(
        params=v3p, projection=proj_v6,
        train_carriers=carriers,
        target_delta_logits_stack=[t1.tolist(), t2.tolist()],
        follow_up_token_ids=ids, n_directions=3)
    assert report.converged
    assert report.n_targets == 2
    # Attention-pattern fit.
    fitted2, report2 = fit_kv_bridge_v6_attention_pattern(
        params=v3p, projection=proj_v6,
        train_carriers=carriers,
        target_attention_last_row=[0.5, 0.2, 0.1, 0.1],
        follow_up_token_ids=ids, n_directions=2)
    assert report2.fit_kind == "attention_pattern"
    # Fingerprint.
    _, v6_cache = forward_tiny_substrate_v6(p, ids)
    fp = v6_cache_key_fingerprint(v6_cache)
    assert len(fp) == 128


def test_hidden_state_bridge_v5_multi_target():
    from coordpy.hidden_state_bridge_v2 import (
        HiddenStateBridgeV2Projection,
    )
    from coordpy.hidden_state_bridge_v3 import (
        HiddenStateBridgeV3Projection,
    )
    from coordpy.hidden_state_bridge_v4 import (
        HiddenStateBridgeV4Projection,
    )
    from coordpy.hidden_state_bridge_v5 import (
        HiddenStateBridgeV5Projection,
        bridge_hidden_state_and_measure_v5,
        fit_hsb_v5_multi_target,
        recover_hsb_v5_inject,
        write_hsb_v5_into_v6_cache,
    )
    from coordpy.tiny_substrate_v6 import (
        build_default_tiny_substrate_v6, tokenize_bytes_v6,
        TinyV6KVCache,
    )
    p = build_default_tiny_substrate_v6(seed=613)
    v3p = p.v3_params
    hsb_v2 = HiddenStateBridgeV2Projection.init(
        target_layers=(1, 3), n_tokens=4, carrier_dim=6,
        d_model=p.config.d_model, seed=61300)
    hsb_v3 = HiddenStateBridgeV3Projection.init_from_v2(
        hsb_v2, n_heads=p.config.n_heads, seed_v3=61301)
    hsb_v4 = HiddenStateBridgeV4Projection.init_from_v3(
        hsb_v3, seed_v4=61302)
    hsb_v5 = HiddenStateBridgeV5Projection.init_from_v4(
        hsb_v4, n_positions=3, seed_v5=61303)
    rng = np.random.default_rng(0)
    carriers = [list(rng.standard_normal(6)) for _ in range(3)]
    ids = tokenize_bytes_v6("hsb5", max_len=4)[:4]
    while len(ids) < 4: ids.append(0)
    target = np.zeros(p.config.vocab_size); target[100] = 1.0
    target2 = np.zeros(p.config.vocab_size); target2[120] = 0.7
    fitted, report = fit_hsb_v5_multi_target(
        params=v3p, projection=hsb_v5,
        train_carriers=carriers,
        target_delta_logits_stack=[
            target.tolist(), target2.tolist()],
        token_ids=ids)
    assert report.converged
    assert report.n_targets == 2
    assert report.n_positions_fitted == 3
    # Recovery.
    adv = rng.standard_normal((2, p.config.n_heads, 3)) * 0.5
    rec, rep = recover_hsb_v5_inject(
        params=v3p, projection=hsb_v5,
        carrier=carriers[0], token_ids=ids,
        target_delta_logits=target.tolist(),
        adversarial_per_head_pos=adv)
    assert rep.post_fit_residual <= rep.pre_fit_residual + 1e-9
    # Witness + write into V6 cache.
    w = bridge_hidden_state_and_measure_v5(
        params=v3p, carrier=carriers[0], projection=fitted,
        token_ids=ids, target_unit=target.tolist())
    v6_cache = TinyV6KVCache.empty(
        p.config.n_layers, n_heads=p.config.n_heads,
        d_key=p.config.d_key)
    write_hsb_v5_into_v6_cache(
        projection=fitted, v6_cache=v6_cache)
    assert float(v6_cache.hidden_write_trace.sum()) > 0.0


def test_attention_steering_v5_4d_budget_and_falsifiers():
    from coordpy.attention_steering_bridge_v2 import (
        AttentionSteeringV2Projection,
    )
    from coordpy.attention_steering_bridge_v5 import (
        steer_attention_and_measure_v5,
    )
    from coordpy.tiny_substrate_v6 import (
        build_default_tiny_substrate_v6, tokenize_bytes_v6,
    )
    p = build_default_tiny_substrate_v6(seed=614)
    ids = tokenize_bytes_v6("attn5", max_len=4)[:4]
    while len(ids) < 4: ids.append(0)
    proj = AttentionSteeringV2Projection.init(
        n_layers=p.config.n_layers, n_heads=p.config.n_heads,
        n_query=len(ids), n_key=len(ids), carrier_dim=6,
        seed=61400)
    rng = np.random.default_rng(0)
    carrier = list(rng.standard_normal(6))
    w = steer_attention_and_measure_v5(
        params=p.v3_params, carrier=carrier,
        projection=proj, token_ids=ids,
        kl_budget_per_key=0.4)
    assert w.attention_pattern_shifted
    # Negative-budget falsifier returns kl_max=0.
    w_neg = steer_attention_and_measure_v5(
        params=p.v3_params, carrier=carrier,
        projection=proj, token_ids=ids,
        kl_budget_per_key=0.4, negative_budget=True)
    assert w_neg.per_head_query_kl_post_max < 1e-9
    # Signed falsifier produces a non-zero correlation.
    w_sgn = steer_attention_and_measure_v5(
        params=p.v3_params, carrier=carrier,
        projection=proj, token_ids=ids,
        kl_budget_per_key=0.4, signed_falsifier=True)
    assert w_sgn.signed_falsifier_used


def test_cache_controller_v4_fits():
    from coordpy.cache_controller_v4 import (
        CacheControllerV4, W61_CACHE_POLICY_BILINEAR_RETRIEVAL_V6,
        fit_bilinear_retrieval_v6, fit_composite_v4,
        fit_trained_corruption_floor, fit_two_stage_threshold,
    )
    from coordpy.tiny_substrate_v6 import (
        build_default_tiny_substrate_v6, tokenize_bytes_v6,
    )
    p = build_default_tiny_substrate_v6(seed=615)
    c = CacheControllerV4.init(
        policy=W61_CACHE_POLICY_BILINEAR_RETRIEVAL_V6,
        d_model=p.config.d_model, d_key=p.config.d_key,
        fit_seed=61500)
    rng = np.random.default_rng(0)
    ids_list = [tokenize_bytes_v6(f"slot-{i}", max_len=4)
                 for i in range(6)]
    qvs = [list(rng.standard_normal(p.config.d_model))
           for _ in range(6)]
    rels = [1.0, 0.7, 0.3, 0.0, 0.9, 0.5]
    fitted, r = fit_bilinear_retrieval_v6(
        controller=c, params_v6=p,
        train_token_ids_per_slot=ids_list,
        train_query_vectors=qvs, train_relevance=rels)
    assert r.converged
    assert fitted.bilinear_retrieval_v6_matrix is not None
    fitted2, _ = fit_trained_corruption_floor(
        controller=fitted, params_v6=p,
        train_token_ids=tokenize_bytes_v6("x", max_len=8),
        train_corruption_counts=[0, 1, 2, 3, 4, 5],
        train_floor_targets=[0.0, -1.0, -2.5, -4.0, -6.0, -9.0])
    assert fitted2.corruption_floor_coefs is not None
    fitted3, _ = fit_two_stage_threshold(
        controller=fitted2,
        attention_receive_scores=[0.1, 0.2, 0.5, 0.7, 0.9, 0.3],
        drop_oracle_per_slot=[0.0, 0.0, 0.5, 1.0, 1.0, 0.0])
    heads = rng.standard_normal((10, 5))
    y = heads.sum(axis=1) + rng.standard_normal(10) * 0.1
    fitted4, _ = fit_composite_v4(
        controller=fitted3, head_scores=heads.tolist(),
        drop_oracle=y.tolist())
    assert fitted4.composite_v4_weights is not None
    assert fitted4.composite_v4_weights.size == 5


def test_replay_controller_v2_fit_and_decide():
    from coordpy.replay_controller import (
        ReplayCandidate, W60_REPLAY_DECISION_ABSTAIN,
        W60_REPLAY_DECISION_RECOMPUTE,
        W60_REPLAY_DECISION_REUSE,
    )
    from coordpy.replay_controller_v2 import (
        ReplayControllerV2, fit_replay_controller_v2,
        emit_replay_controller_v2_witness,
    )
    rc = ReplayControllerV2.init()
    cands = [
        ReplayCandidate(
            flop_reuse=100, flop_recompute=1000, flop_fallback=50,
            drift_l2_reuse=0.1, drift_l2_recompute=0.0,
            drift_l2_fallback=0.3,
            crc_passed=True, transcript_available=True,
            n_corruption_flags=0),
        ReplayCandidate(
            flop_reuse=900, flop_recompute=1000, flop_fallback=50,
            drift_l2_reuse=0.8, drift_l2_recompute=0.0,
            drift_l2_fallback=0.3,
            crc_passed=False, transcript_available=True,
            n_corruption_flags=3),
    ] * 4
    labels = [
        W60_REPLAY_DECISION_REUSE,
        W60_REPLAY_DECISION_RECOMPUTE] * 4
    fitted, rep = fit_replay_controller_v2(
        controller=rc,
        train_candidates=cands,
        train_optimal_decisions=labels)
    assert rep.post_fit_residual <= rep.pre_fit_residual + 1e-9
    d, conf = fitted.decide(cands[0])
    assert 0.0 <= conf <= 1.0
    # Hidden-write gate.
    fitted.hidden_write_cap = 0.5
    d2, conf2 = fitted.decide(
        cands[0], hidden_write_total_l2=10.0)
    assert d2.decision == W60_REPLAY_DECISION_ABSTAIN
    assert d2.rationale == "hidden_write_cap_exceeded"
    w = emit_replay_controller_v2_witness(fitted, cands)
    assert w.n_decisions > 0


def test_substrate_adapter_v6_tier():
    from coordpy.substrate_adapter_v5 import (
        probe_tiny_substrate_v5_adapter,
    )
    from coordpy.substrate_adapter_v6 import (
        W61_SUBSTRATE_TIER_SUBSTRATE_V6_FULL,
        probe_tiny_substrate_v6_adapter,
        probe_v5_substrate_adapter_as_v6,
    )
    v6 = probe_tiny_substrate_v6_adapter()
    v5_as_v6 = probe_v5_substrate_adapter_as_v6(
        probe_tiny_substrate_v5_adapter())
    assert v6.tier == W61_SUBSTRATE_TIER_SUBSTRATE_V6_FULL
    assert v5_as_v6.tier != W61_SUBSTRATE_TIER_SUBSTRATE_V6_FULL


def test_persistent_latent_v13_ninth_carrier():
    from coordpy.persistent_latent_v12 import V12StackedCell
    from coordpy.persistent_latent_v13 import (
        PersistentLatentStateV13Chain,
        emit_persistent_v13_witness,
        step_persistent_state_v13,
    )
    cell = V12StackedCell.init(seed=61600)
    chain = PersistentLatentStateV13Chain.empty()
    s = step_persistent_state_v13(
        cell=cell, prev_state=None,
        carrier_values=[0.1] * 16,
        turn_index=0, role="r",
        replay_confidence_skip=[0.5] * 16)
    chain.add(s)
    w = emit_persistent_v13_witness(chain, s.cid())
    assert w.nine_skip_present
    assert w.replay_confidence_carrier_l1_sum > 0.0


def test_mergeable_latent_capsule_v9_chains():
    from coordpy.mergeable_latent_capsule_v3 import (
        make_root_capsule_v3,
    )
    from coordpy.mergeable_latent_capsule_v4 import wrap_v3_as_v4
    from coordpy.mergeable_latent_capsule_v5 import wrap_v4_as_v5
    from coordpy.mergeable_latent_capsule_v6 import wrap_v5_as_v6
    from coordpy.mergeable_latent_capsule_v7 import wrap_v6_as_v7
    from coordpy.mergeable_latent_capsule_v8 import wrap_v7_as_v8
    from coordpy.mergeable_latent_capsule_v9 import (
        MergeOperatorV9,
        W61_MLSC_V9_ALGEBRA_ATTENTION_PATTERN_STEER,
        wrap_v8_as_v9,
    )
    v3 = make_root_capsule_v3(
        branch_id="t", payload=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        confidence=1.0, trust=1.0)
    v4 = wrap_v3_as_v4(v3); v5 = wrap_v4_as_v5(v4)
    v6 = wrap_v5_as_v6(v5); v7 = wrap_v6_as_v7(v6)
    v8 = wrap_v7_as_v8(v7)
    cap_a = wrap_v8_as_v9(
        v8,
        attention_pattern_witness_chain=("ap_a", "ap_b"),
        cache_retrieval_witness_chain=("cr_a",),
        per_layer_head_trust_matrix=((0, 0, 0.5),),
        algebra_signature_v7=(
            W61_MLSC_V9_ALGEBRA_ATTENTION_PATTERN_STEER))
    op = MergeOperatorV9()
    merged = op.merge(
        [cap_a],
        attention_pattern_witness_chain=("ap_c",),
        cache_retrieval_witness_chain=("cr_b",),
        per_layer_head_trust_matrix=((0, 0, 0.9),),
        algebra_signature_v7=(
            W61_MLSC_V9_ALGEBRA_ATTENTION_PATTERN_STEER))
    assert "ap_c" in merged.attention_pattern_witness_chain
    assert "cr_b" in merged.cache_retrieval_witness_chain
    tab = dict(
        ((int(l), int(h)), float(t))
        for l, h, t in merged.per_layer_head_trust_matrix)
    assert tab.get((0, 0), 0.0) == 0.9


def test_consensus_v7_stages():
    from coordpy.consensus_fallback_controller_v7 import (
        W61_CONSENSUS_V7_STAGE_ATTENTION_PATTERN,
        W61_CONSENSUS_V7_STAGES,
    )
    assert len(W61_CONSENSUS_V7_STAGES) == 11
    assert (
        W61_CONSENSUS_V7_STAGE_ATTENTION_PATTERN
        in W61_CONSENSUS_V7_STAGES)


def test_corruption_robust_carrier_v9_fingerprint():
    from coordpy.corruption_robust_carrier_v9 import (
        CorruptionRobustCarrierV9,
        emit_corruption_robustness_v9_witness,
        kv_cache_fingerprint_512,
        post_replay_topk_jaccard,
    )
    fp = kv_cache_fingerprint_512(
        b"hello", b"world-512-bucket")
    assert len(fp) == 512
    crc = CorruptionRobustCarrierV9()
    w = emit_corruption_robustness_v9_witness(
        crc_v9=crc, n_probes=8, seed=61700)
    assert 0.0 <= w.kv512_corruption_detect_rate <= 1.0
    # post-replay top-K Jaccard.
    j = post_replay_topk_jaccard(
        [0.1, 0.9, 0.4, 0.6], [0.1, 0.9, 0.4, 0.6], k=2)
    assert j == 1.0


def test_long_horizon_retention_v13_twelve_way():
    from coordpy.long_horizon_retention_v13 import (
        LongHorizonReconstructionV13Head,
        emit_lhr_v13_witness,
    )
    head = LongHorizonReconstructionV13Head.init(seed=61800)
    w = emit_lhr_v13_witness(
        head, carrier=[0.1] * 8, k=4,
        attention_top_k_indicator=[1.0] * 8)
    assert w.twelve_way_runs
    assert w.n_heads == 12


def test_ecc_codebook_v13_bits_and_codes():
    from coordpy.ecc_codebook_v13 import (
        ECCCodebookV13, compress_carrier_ecc_v13,
        probe_ecc_v13_rate_floor_falsifier,
    )
    from coordpy.ecc_codebook_v5 import (
        W53_DEFAULT_ECC_CODE_DIM, W53_DEFAULT_ECC_EMIT_MASK_LEN,
    )
    from coordpy.quantised_compression import (
        QuantisedBudgetGate,
    )
    cb = ECCCodebookV13.init(seed=61900)
    assert cb.total_codes == (1 << 22)
    gate = QuantisedBudgetGate.init(
        in_dim=W53_DEFAULT_ECC_CODE_DIM,
        emit_mask_len=W53_DEFAULT_ECC_EMIT_MASK_LEN,
        seed=61901)
    gate.importance_threshold = 0.0
    gate.w_emit.values = [1.0] * len(gate.w_emit.values)
    rng = np.random.default_rng(0)
    carrier = list(rng.standard_normal(
        W53_DEFAULT_ECC_CODE_DIM).tolist())
    out = compress_carrier_ecc_v13(
        carrier, codebook=cb, gate=gate)
    assert float(out["bits_per_visible_token"]) >= 24.0
    fal = probe_ecc_v13_rate_floor_falsifier(codebook=cb)
    assert fal["reproduces_cap"]


def test_uncertainty_v9_eight_axis():
    from coordpy.uncertainty_layer_v9 import (
        compose_uncertainty_report_v9,
        emit_uncertainty_v9_witness,
    )
    rep = compose_uncertainty_report_v9(
        confidences=[0.7, 0.5],
        trusts=[0.9, 0.8],
        substrate_fidelities=[0.95, 0.9],
        hidden_state_fidelities=[0.95, 0.9],
        cache_reuse_fidelities=[0.95, 0.9],
        retrieval_fidelities=[0.95, 0.9],
        replay_fidelities=[0.95, 0.9],
        attention_pattern_fidelities=[0.95, 0.7])
    assert rep.attention_pattern_aware
    w = emit_uncertainty_v9_witness(rep)
    assert w.composite_cid == rep.cid()


def test_disagreement_algebra_v7_identity():
    from coordpy.disagreement_algebra import AlgebraTrace
    from coordpy.disagreement_algebra_v7 import (
        emit_disagreement_algebra_v7_witness,
    )
    trace = AlgebraTrace(steps=[])
    probe = [0.1, 0.2, 0.3]
    w = emit_disagreement_algebra_v7_witness(
        trace=trace, probe_a=probe, probe_b=probe, probe_c=probe,
        attention_pattern_oracle=lambda: (True, 0.8))
    assert w.attention_pattern_equiv_ok


def test_tvs_arbiter_v10_eleven_arms():
    from coordpy.transcript_vs_shared_arbiter_v10 import (
        W61_TVS_V10_ARMS,
        eleven_arm_compare,
    )
    out = eleven_arm_compare(
        per_turn_confidences=[0.0],
        per_turn_trust_scores=[0.0],
        per_turn_merge_retentions=[0.0],
        per_turn_tw_retentions=[0.0],
        per_turn_substrate_fidelities=[0.0],
        per_turn_hidden_fidelities=[0.0],
        per_turn_cache_fidelities=[0.0],
        per_turn_retrieval_fidelities=[0.0],
        per_turn_replay_fidelities=[0.0],
        per_turn_attention_pattern_fidelities=[0.9])
    assert out.attention_pattern_used
    assert len(W61_TVS_V10_ARMS) == 11


def test_deep_substrate_hybrid_v6_six_way():
    from coordpy.attention_steering_bridge_v2 import (
        AttentionSteeringV2Projection,
    )
    from coordpy.attention_steering_bridge_v5 import (
        steer_attention_and_measure_v5,
    )
    from coordpy.cache_controller_v4 import (
        CacheControllerV4, W61_CACHE_POLICY_BILINEAR_RETRIEVAL_V6,
    )
    from coordpy.deep_substrate_hybrid_v5 import (
        DeepSubstrateHybridV5ForwardWitness,
    )
    from coordpy.deep_substrate_hybrid_v6 import (
        DeepSubstrateHybridV6, deep_substrate_hybrid_v6_forward,
    )
    from coordpy.replay_controller_v2 import (
        ReplayControllerV2, fit_replay_controller_v2,
    )
    from coordpy.replay_controller import (
        ReplayCandidate, W60_REPLAY_DECISION_REUSE,
    )
    from coordpy.tiny_substrate_v6 import (
        build_default_tiny_substrate_v6, tokenize_bytes_v6,
    )
    p = build_default_tiny_substrate_v6(seed=620)
    cc = CacheControllerV4.init(
        policy=W61_CACHE_POLICY_BILINEAR_RETRIEVAL_V6,
        d_model=64, d_key=8, fit_seed=62000)
    cc.bilinear_retrieval_v6_matrix = np.zeros(
        (64, 8), dtype=np.float64)
    rc = ReplayControllerV2.init()
    cand = ReplayCandidate(
        flop_reuse=100, flop_recompute=1000, flop_fallback=50,
        drift_l2_reuse=0.1, drift_l2_recompute=0.0,
        drift_l2_fallback=0.3,
        crc_passed=True, transcript_available=True,
        n_corruption_flags=0)
    rc, _ = fit_replay_controller_v2(
        controller=rc,
        train_candidates=[cand] * 4,
        train_optimal_decisions=[
            W60_REPLAY_DECISION_REUSE] * 4)
    rc.decide(cand)
    ids = tokenize_bytes_v6("attn5", max_len=4)[:4]
    while len(ids) < 4: ids.append(0)
    proj = AttentionSteeringV2Projection.init(
        n_layers=p.config.n_layers, n_heads=p.config.n_heads,
        n_query=len(ids), n_key=len(ids), carrier_dim=6,
        seed=62001)
    rng = np.random.default_rng(0)
    attn_w = steer_attention_and_measure_v5(
        params=p.v3_params,
        carrier=list(rng.standard_normal(6)),
        projection=proj, token_ids=ids,
        kl_budget_per_key=0.4)
    v5_witness = DeepSubstrateHybridV5ForwardWitness(
        schema="x", deep_v6_witness_cid="x",
        substrate_v5_forward_cid="x",
        bridge_v5_injection_cid="x",
        pre_inject_kv_cid="x", post_inject_kv_cid="x",
        post_evict_kv_cid="x",
        substrate_back_l2=0.0,
        ablation_perturbation_l2=0.0,
        cache_eviction_perturbation_l2=0.0,
        cache_retention_ratio=1.0,
        cache_policy="composite_v3",
        cache_flop_savings_ratio=0.5,
        retrieval_used=True, five_way=True,
        attention_steering_used=True,
        replay_decision="choose_reuse",
        replay_flop_chosen=100, replay_drift_chosen=0.05)
    hybrid = DeepSubstrateHybridV6(inner_v5=None)
    w = deep_substrate_hybrid_v6_forward(
        hybrid=hybrid, v5_witness=v5_witness,
        cache_controller_v4=cc, replay_controller_v2=rc,
        attention_steering_v5_witness=attn_w)
    assert w.six_way
