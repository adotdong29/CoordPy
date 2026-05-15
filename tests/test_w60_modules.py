"""W60 module focused tests (post-W59 milestone).

Per-module sanity tests for the W60 line. The R-125/R-126/R-127
suites separately exercise the H-bars end-to-end.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_tiny_substrate_v5_determinism_and_multi_segment_reuse():
    from coordpy.tiny_substrate_v5 import (
        W60_V5_SEGMENT_DROP, W60_V5_SEGMENT_RECOMPUTE,
        W60_V5_SEGMENT_REUSE,
        build_default_tiny_substrate_v5,
        extract_multi_segment_prefix_v5,
        forward_tiny_substrate_v5,
        forward_with_multi_segment_reuse_v5,
        logit_jacobian_v5,
        tokenize_bytes_v5,
    )
    p = build_default_tiny_substrate_v5(seed=600)
    ids = tokenize_bytes_v5("w60-determ", max_len=12)
    t1, c1 = forward_tiny_substrate_v5(p, ids)
    t2, c2 = forward_tiny_substrate_v5(p, ids)
    assert t1.cid() == t2.cid()
    assert all(a.shape == (p.config.n_heads, c1.n_tokens())
               for a in c1.attention_receive)
    # Multi-segment partial reuse runs.
    prompt = tokenize_bytes_v5("hello world", max_len=14)
    fu = tokenize_bytes_v5("a", max_len=2)
    segs = [
        (0, 4, W60_V5_SEGMENT_REUSE),
        (4, 10, W60_V5_SEGMENT_RECOMPUTE),
        (10, len(prompt), W60_V5_SEGMENT_DROP),
    ]
    prefix = extract_multi_segment_prefix_v5(
        p, prompt, segments=segs)
    ft, fc, split = forward_with_multi_segment_reuse_v5(
        p, prefix, fu)
    assert int(split["flop_recompute"]) > 0
    assert int(split["flop_drop_skipped"]) > 0
    # Logit Jacobian table shape.
    j = logit_jacobian_v5(t1, p, target_token=10)
    arr = np.asarray(j["per_layer_per_head"])
    assert arr.shape == (p.config.n_layers, p.config.n_heads)
    # Corruption flag channel.
    c1.set_corruption_flag(layer_index=0, position=2, flagged=True)
    assert c1.n_corrupted() == 1


def test_kv_bridge_v5_multi_dir_and_logit_dir_fits():
    from coordpy.kv_bridge_v3 import KVBridgeV3Projection
    from coordpy.kv_bridge_v4 import KVBridgeV4Projection
    from coordpy.kv_bridge_v5 import (
        KVBridgeV5Projection,
        bridge_carrier_and_measure_v5,
        fit_kv_bridge_v5_correction,
        fit_kv_bridge_v5_logit_direction,
    )
    from coordpy.tiny_substrate_v5 import (
        build_default_tiny_substrate_v5,
        tokenize_bytes_v5,
    )
    p = build_default_tiny_substrate_v5(seed=601)
    v3p = p.v3_params
    proj_v3 = KVBridgeV3Projection.init(
        n_layers=p.config.n_layers,
        n_heads=p.config.n_heads,
        n_kv_heads=p.config.n_kv_heads,
        n_inject_tokens=3, carrier_dim=8,
        d_head=p.config.d_model // p.config.n_heads,
        seed=60100)
    proj_v4 = KVBridgeV4Projection.init_from_v3(
        proj_v3, seed_v4=60101)
    proj_v5 = KVBridgeV5Projection.init_from_v4(
        proj_v4, seed_v5=60102)
    rng = np.random.default_rng(0)
    ids = tokenize_bytes_v5("kv5-test", max_len=4)
    carriers = [list(rng.standard_normal(8))
                 for _ in range(4)]
    targets = [3.0] * 4
    fitted, report = fit_kv_bridge_v5_correction(
        params=v3p, projection=proj_v5,
        train_carriers=carriers, train_target_l2=targets,
        follow_up_token_ids=ids, n_directions=3)
    assert report.converged
    assert report.n_directions == 3
    target_d = np.zeros(p.config.vocab_size)
    target_d[100] = 1.0
    fitted2, report2 = fit_kv_bridge_v5_logit_direction(
        params=v3p, projection=proj_v5,
        train_carriers=carriers,
        target_delta_logits=target_d.tolist(),
        follow_up_token_ids=ids, n_directions=2)
    assert report2.converged
    assert report2.fit_kind == "logit_direction"
    # Witness with reverse-extract.
    w = bridge_carrier_and_measure_v5(
        params=v3p, carrier=carriers[0],
        projection=fitted, follow_up_token_ids=ids)
    assert float(w.extract_carrier_residual_l2) < 1e-3


def test_hidden_state_bridge_v4_per_head_fit_and_compare_kv():
    from coordpy.hidden_state_bridge_v2 import (
        HiddenStateBridgeV2Projection,
    )
    from coordpy.hidden_state_bridge_v3 import (
        HiddenStateBridgeV3Projection,
    )
    from coordpy.hidden_state_bridge_v4 import (
        HiddenStateBridgeV4Projection,
        compare_hidden_vs_kv_injection_v4,
        fit_hsb_v4_per_head_target,
        recover_hsb_v4_inject,
    )
    from coordpy.kv_bridge_v3 import KVBridgeV3Projection
    from coordpy.kv_bridge_v4 import KVBridgeV4Projection
    from coordpy.tiny_substrate_v5 import (
        build_default_tiny_substrate_v5,
        tokenize_bytes_v5,
    )
    p = build_default_tiny_substrate_v5(seed=602)
    v3p = p.v3_params
    hsb_v2 = HiddenStateBridgeV2Projection.init(
        target_layers=(1, 3), n_tokens=4,
        carrier_dim=6, d_model=p.config.d_model,
        seed=60200)
    hsb_v3 = HiddenStateBridgeV3Projection.init_from_v2(
        hsb_v2, n_heads=p.config.n_heads, seed_v3=60201)
    hsb_v4 = HiddenStateBridgeV4Projection.init_from_v3(
        hsb_v3, seed_v4=60202)
    target = np.zeros(p.config.vocab_size); target[100] = 1.0
    rng = np.random.default_rng(0)
    carriers = [list(rng.standard_normal(6))
                 for _ in range(3)]
    ids = tokenize_bytes_v5("hsb4", max_len=4)[:4]
    while len(ids) < 4:
        ids.append(0)
    fitted, report = fit_hsb_v4_per_head_target(
        params=v3p, projection=hsb_v4,
        train_carriers=carriers,
        target_delta_logits=target, token_ids=ids)
    assert report.converged
    assert report.n_layers_fitted == 2
    assert report.n_heads_fitted == p.config.n_heads
    # Recovery
    adv = rng.standard_normal((2, p.config.n_heads)) * 0.5
    rec, rep = recover_hsb_v4_inject(
        params=v3p, projection=hsb_v4,
        carrier=carriers[0], token_ids=ids,
        target_delta_logits=target,
        adversarial_per_head=adv)
    assert rep.post_fit_residual <= rep.pre_fit_residual + 1e-9
    # Compare hidden vs KV.
    proj_v3 = KVBridgeV3Projection.init(
        n_layers=p.config.n_layers,
        n_heads=p.config.n_heads,
        n_kv_heads=p.config.n_kv_heads,
        n_inject_tokens=3, carrier_dim=6,
        d_head=p.config.d_model // p.config.n_heads,
        seed=60203)
    proj_v4 = KVBridgeV4Projection.init_from_v3(
        proj_v3, seed_v4=60204)
    cmp = compare_hidden_vs_kv_injection_v4(
        params=v3p, carrier=carriers[0], token_ids=ids,
        target_delta_logits=target,
        hsb_v4_projection=hsb_v4,
        kv_v4_projection=proj_v4,
        train_carriers=carriers)
    assert cmp.hidden_beats_kv or cmp.kv_beats_hidden or cmp.tie


def test_prefix_state_v4_multi_segment():
    from coordpy.prefix_state_bridge_v4 import (
        bridge_prefix_state_and_measure_v4,
    )
    from coordpy.tiny_substrate_v5 import (
        W60_V5_SEGMENT_DROP, W60_V5_SEGMENT_RECOMPUTE,
        W60_V5_SEGMENT_REUSE,
        build_default_tiny_substrate_v5,
        tokenize_bytes_v5,
    )
    p = build_default_tiny_substrate_v5(seed=603)
    prompt = tokenize_bytes_v5("multi-seg-prefix", max_len=14)
    fu1 = tokenize_bytes_v5("a", max_len=2)
    fu2 = tokenize_bytes_v5("b", max_len=2)
    segs = [
        (0, 4, W60_V5_SEGMENT_REUSE),
        (4, 10, W60_V5_SEGMENT_RECOMPUTE),
        (10, len(prompt), W60_V5_SEGMENT_DROP),
    ]
    w = bridge_prefix_state_and_measure_v4(
        params_v5=p, prompt_token_ids=prompt,
        follow_up_chain=[fu1, fu2], segments=segs)
    assert w.flop_saved_vs_recompute > 0
    assert len(w.chain_step_drifts_l2) == 2


def test_attention_steering_v4_per_query_budget():
    from coordpy.attention_steering_bridge_v2 import (
        AttentionSteeringV2Projection,
    )
    from coordpy.attention_steering_bridge_v4 import (
        steer_attention_and_measure_v4,
    )
    from coordpy.tiny_substrate_v5 import (
        build_default_tiny_substrate_v5,
        tokenize_bytes_v5,
    )
    p = build_default_tiny_substrate_v5(seed=605)
    v3p = p.v3_params
    proj = AttentionSteeringV2Projection.init(
        n_layers=p.config.n_layers,
        n_heads=p.config.n_heads,
        n_query=4, n_key=8, carrier_dim=6,
        seed=60500)
    carrier = list(np.random.default_rng(2).standard_normal(6))
    ids = tokenize_bytes_v5("attn4", max_len=4)[:4]
    while len(ids) < 4:
        ids.append(0)
    w = steer_attention_and_measure_v4(
        params=v3p, carrier=carrier, projection=proj,
        token_ids=ids, kl_budget_per_query=0.4)
    assert w.per_query_budget_enforced
    assert w.per_head_query_kl_post_max <= 0.4 + 1e-3
    # Negative budget falsifier.
    wn = steer_attention_and_measure_v4(
        params=v3p, carrier=carrier, projection=proj,
        token_ids=ids, negative_budget=True)
    assert wn.negative_budget_post_kl_max < 1e-6


def test_cache_controller_v3_trained_eviction_and_composite():
    from coordpy.cache_controller_v3 import (
        CacheControllerV3,
        W60_CACHE_POLICY_COMPOSITE_V3,
        W60_CACHE_POLICY_LEARNED_CORRUPTION_AWARE,
        apply_cache_controller_v3_and_measure,
        fit_composite_v3_controller,
        fit_learned_attention_receive_controller,
        fit_trained_eviction_controller,
    )
    from coordpy.tiny_substrate_v5 import (
        build_default_tiny_substrate_v5,
        forward_tiny_substrate_v5, tokenize_bytes_v5,
    )
    p = build_default_tiny_substrate_v5(seed=606)
    v3p = p.v3_params
    prompt = tokenize_bytes_v5("cc3", max_len=12)
    fu = tokenize_bytes_v5("fu", max_len=4)
    trace, cache = forward_tiny_substrate_v5(p, prompt)
    n = int(cache.n_tokens())
    ctrl_a, rep_a = fit_learned_attention_receive_controller(
        v3p, prompt_token_ids=prompt,
        follow_up_token_ids=fu,
        attention_receive=cache.attention_receive)
    assert rep_a.converged
    ctrl_e, rep_e = fit_trained_eviction_controller(
        v3p, prompt_token_ids=prompt,
        follow_up_token_ids=fu,
        attention_receive=cache.attention_receive)
    assert rep_e.converged
    rng = np.random.default_rng(0)
    imp = np.zeros((n,))
    for li in trace.v4_trace.kv_cache_v3.importance:
        if li.size and li.shape[0] == n:
            imp += li
    ctrl_c, rep_c = fit_composite_v3_controller(
        v3p, prompt_token_ids=prompt,
        follow_up_token_ids=fu,
        importance_vector=imp,
        learned_hidden_scores=rng.standard_normal(n),
        learned_retrieval_scores=rng.standard_normal(n),
        learned_attention_receive_scores=(
            rng.standard_normal(n)))
    assert rep_c.converged
    assert ctrl_c.composite_weights is not None
    assert len(ctrl_c.composite_weights) == 4
    # Corruption-aware.
    flags = [np.zeros(n, dtype=bool)
              for _ in range(p.config.n_layers)]
    flags[0][3] = True
    ctrl_corr = CacheControllerV3.init(
        policy=W60_CACHE_POLICY_LEARNED_CORRUPTION_AWARE,
        d_model=p.config.d_model)
    w_corr = apply_cache_controller_v3_and_measure(
        v3p, ctrl_corr, prompt_token_ids=prompt,
        follow_up_token_ids=fu, retention_ratio=0.5,
        corruption_flags=flags)
    assert w_corr.used_corruption_flags


def test_replay_controller_decisions():
    from coordpy.replay_controller import (
        ReplayCandidate, ReplayController,
        W60_REPLAY_DECISION_ABSTAIN,
        W60_REPLAY_DECISION_RECOMPUTE,
        W60_REPLAY_DECISION_REUSE,
        emit_replay_controller_witness,
    )
    rc = ReplayController()
    c1 = ReplayCandidate(
        flop_reuse=100, flop_recompute=1000,
        flop_fallback=10, drift_l2_reuse=0.05,
        drift_l2_recompute=0.0, drift_l2_fallback=2.0,
        crc_passed=True, transcript_available=True,
        n_corruption_flags=0)
    d1 = rc.decide(c1)
    assert d1.decision == W60_REPLAY_DECISION_REUSE
    c2 = ReplayCandidate(
        flop_reuse=100, flop_recompute=1000,
        flop_fallback=10, drift_l2_reuse=2.0,
        drift_l2_recompute=0.0, drift_l2_fallback=2.0,
        crc_passed=False, transcript_available=True,
        n_corruption_flags=3)
    d2 = rc.decide(c2)
    assert d2.decision == W60_REPLAY_DECISION_RECOMPUTE
    c3 = ReplayCandidate(
        flop_reuse=10000, flop_recompute=10000,
        flop_fallback=0, drift_l2_reuse=10.0,
        drift_l2_recompute=0.0, drift_l2_fallback=0.0,
        crc_passed=False, transcript_available=False,
        n_corruption_flags=10)
    d3 = rc.decide(c3)
    # All paths unavailable / unhelpful: ABSTAIN or RECOMPUTE.
    assert d3.decision in (
        W60_REPLAY_DECISION_ABSTAIN,
        W60_REPLAY_DECISION_RECOMPUTE)
    w = emit_replay_controller_witness(rc, [c1, c2, c3])
    assert w.n_decisions == 3
    assert w.reuse_count >= 1


def test_persistent_v12_chain_walk():
    from coordpy.persistent_latent_v12 import (
        PersistentLatentStateV12Chain,
        V12StackedCell,
        emit_persistent_v12_witness,
        step_persistent_state_v12,
    )
    cell = V12StackedCell.init(seed=608)
    chain = PersistentLatentStateV12Chain.empty()
    prev = None
    for t in range(8):
        carrier = [0.1 * (t + 1)] * cell.state_dim
        prev = step_persistent_state_v12(
            cell=cell, prev_state=prev,
            carrier_values=carrier,
            turn_index=t, role="r",
            replay_skip=carrier, replay_fidelity=0.9)
        chain.add(prev)
    w = emit_persistent_v12_witness(
        cell=cell, chain=chain, leaf_cid=prev.cid())
    assert w.achieved_chain_walk_depth == 8
    assert w.n_layers == 10
    assert w.distractor_basis_rank == 4


def test_multi_hop_v10_chain_length_and_axes():
    from coordpy.multi_hop_translator_v10 import (
        W60_DEFAULT_MH_V10_BACKENDS,
        W60_DEFAULT_MH_V10_CHAIN_LEN,
        emit_multi_hop_v10_witness,
    )
    w = emit_multi_hop_v10_witness(seed=609)
    assert w.chain_length == 15
    assert len(w.backends) == 16
    assert w.n_edges == 16 * 15
    assert w.replay_axis_used


def test_mlsc_v8_replay_chain_inheritance():
    from coordpy.mergeable_latent_capsule_v3 import (
        make_root_capsule_v3,
    )
    from coordpy.mergeable_latent_capsule_v4 import (
        wrap_v3_as_v4,
    )
    from coordpy.mergeable_latent_capsule_v5 import (
        wrap_v4_as_v5,
    )
    from coordpy.mergeable_latent_capsule_v6 import (
        wrap_v5_as_v6,
    )
    from coordpy.mergeable_latent_capsule_v7 import (
        wrap_v6_as_v7,
    )
    from coordpy.mergeable_latent_capsule_v8 import (
        MergeOperatorV8, wrap_v7_as_v8,
    )
    v3 = make_root_capsule_v3(
        branch_id="x", payload=tuple([0.1] * 6),
        fact_tags=("x",), confidence=0.9, trust=0.9,
        turn_index=0)
    v4 = wrap_v3_as_v4(v3)
    v5 = wrap_v4_as_v5(v4, attention_witness_cid="a")
    v6 = wrap_v5_as_v6(
        v5, attention_witness_chain=("a1",),
        cache_reuse_witness_cid="c")
    v7 = wrap_v6_as_v7(
        v6, retrieval_witness_chain=("r1",),
        controller_witness_cid="ctrl1")
    v8 = wrap_v7_as_v8(
        v7, replay_witness_chain=("rep1",),
        substrate_witness_chain=("sub1",),
        provenance_trust_table={"backend_a": 0.9})
    op = MergeOperatorV8(factor_dim=6)
    merged = op.merge(
        [v8], replay_witness_chain=("rep2",),
        substrate_witness_chain=("sub2",),
        provenance_trust_table={"backend_b": 0.85})
    assert "rep1" in merged.replay_witness_chain
    assert "rep2" in merged.replay_witness_chain
    assert "sub1" in merged.substrate_witness_chain
    assert "sub2" in merged.substrate_witness_chain


def test_consensus_v6_ten_stages():
    from coordpy.consensus_fallback_controller_v6 import (
        ConsensusFallbackControllerV6,
        W60_CONSENSUS_V6_STAGES,
        W60_CONSENSUS_V6_STAGE_REPLAY_CONTROLLER,
    )
    assert len(W60_CONSENSUS_V6_STAGES) == 10
    ctrl = ConsensusFallbackControllerV6.init(
        k_required=3, cosine_floor=0.95,
        trust_threshold=0.99)

    def replay_oracle(payloads, q, decisions):
        for i, d in enumerate(decisions):
            if str(d) == "choose_reuse":
                return i
        return 0
    ctrl.replay_controller_oracle = replay_oracle
    rng = np.random.default_rng(4)
    q = list(rng.standard_normal(6))
    p1 = list(rng.standard_normal(6))
    p2 = list(rng.standard_normal(6))
    res = ctrl.decide(
        parent_payloads=[p1, p2],
        parent_trusts=[0.1, 0.1],
        parent_cache_fingerprints=None,
        parent_retrieval_scores=None,
        parent_replay_decisions=[
            "choose_recompute", "choose_reuse"],
        query_direction=q, transcript_payload=[])
    assert (res["decision_stage"]
             == W60_CONSENSUS_V6_STAGE_REPLAY_CONTROLLER)


def test_crc_v8_kv256_and_post_replay():
    from coordpy.corruption_robust_carrier_v8 import (
        CorruptionRobustCarrierV8,
        emit_corruption_robustness_v8_witness,
    )
    crc = CorruptionRobustCarrierV8()
    w = emit_corruption_robustness_v8_witness(
        crc_v8=crc, n_probes=12, seed=610)
    assert float(w.kv256_corruption_detect_rate) >= 0.9
    assert float(
        w.cache_retrieval_post_replay_topk_agreement_rate) >= float(
        w.cache_retrieval_topk_agreement_rate)


def test_lhr_v12_six_way_and_two_layer_scorer_fit():
    from coordpy.long_horizon_retention_v12 import (
        LongHorizonReconstructionV12Head,
        evaluate_lhr_v12_six_way,
        fit_lhr_v12_two_layer_retention_scorer,
    )
    head = LongHorizonReconstructionV12Head.init(seed=611)
    rng = np.random.default_rng(5)
    carriers = [list(rng.standard_normal(8)) for _ in range(8)]
    targets = [list(rng.standard_normal(head.out_dim))
                for _ in range(8)]
    subs = [list(rng.standard_normal(32)) for _ in range(8)]
    hids = [list(rng.standard_normal(32)) for _ in range(8)]
    atts = [list(rng.standard_normal(32)) for _ in range(8)]
    rets = [list(rng.standard_normal(32)) for _ in range(8)]
    reps = [list(rng.standard_normal(16)) for _ in range(8)]
    rep = evaluate_lhr_v12_six_way(
        head, carrier_examples=carriers,
        target_examples=targets, substrate_states=subs,
        hidden_states=hids, attention_states=atts,
        retrieval_states=rets, replay_states=reps, k=16)
    assert rep["n"] == 8
    X = rng.standard_normal((16, 8))
    y = X @ rng.standard_normal(8)
    head, residual = fit_lhr_v12_two_layer_retention_scorer(
        head, train_carriers=X.tolist(),
        train_targets=y.tolist())
    assert head.scorer_layer2 is not None
    assert head.max_k == 96


def test_ecc_v12_bits_per_token_and_falsifier():
    from coordpy.ecc_codebook_v12 import (
        ECCCodebookV12,
        compress_carrier_ecc_v12,
        probe_ecc_v12_rate_floor_falsifier,
    )
    from coordpy.ecc_codebook_v5 import (
        W53_DEFAULT_ECC_CODE_DIM,
        W53_DEFAULT_ECC_EMIT_MASK_LEN,
    )
    from coordpy.quantised_compression import (
        QuantisedBudgetGate,
    )
    cb = ECCCodebookV12.init(seed=612)
    assert cb.total_codes == 2097152
    gate = QuantisedBudgetGate.init(
        in_dim=W53_DEFAULT_ECC_CODE_DIM,
        emit_mask_len=W53_DEFAULT_ECC_EMIT_MASK_LEN,
        seed=613)
    gate.importance_threshold = 0.0
    gate.w_emit.values = [1.0] * len(gate.w_emit.values)
    rng = np.random.default_rng(6)
    carrier = list(rng.standard_normal(
        W53_DEFAULT_ECC_CODE_DIM))
    comp = compress_carrier_ecc_v12(
        carrier, codebook=cb, gate=gate)
    assert float(comp["bits_per_visible_token"]) >= 23.0
    fal = probe_ecc_v12_rate_floor_falsifier(codebook=cb)
    assert fal["reproduces_cap"]


def test_tvs_v9_ten_arms_sum_to_one():
    from coordpy.transcript_vs_shared_arbiter_v9 import (
        emit_tvs_arbiter_v9_witness,
        ten_arm_compare,
    )
    n = 16
    rng = np.random.default_rng(7)
    res = ten_arm_compare(
        per_turn_confidences=rng.uniform(0, 1, n).tolist(),
        per_turn_trust_scores=rng.uniform(0, 1, n).tolist(),
        per_turn_merge_retentions=rng.uniform(
            0, 1, n).tolist(),
        per_turn_tw_retentions=rng.uniform(0, 1, n).tolist(),
        per_turn_substrate_fidelities=rng.uniform(
            0, 1, n).tolist(),
        per_turn_hidden_fidelities=rng.uniform(
            0, 1, n).tolist(),
        per_turn_cache_fidelities=rng.uniform(
            0, 1, n).tolist(),
        per_turn_retrieval_fidelities=rng.uniform(
            0, 1, n).tolist(),
        per_turn_replay_fidelities=rng.uniform(
            0, 1, n).tolist())
    w = emit_tvs_arbiter_v9_witness(result=res)
    assert w.n_arms == 10
    assert w.pick_rates_sum_to_one


def test_uncertainty_v8_brackets_and_replay_aware():
    from coordpy.uncertainty_layer_v8 import (
        compose_uncertainty_report_v8,
        emit_uncertainty_v8_witness,
    )
    comp = compose_uncertainty_report_v8(
        component_confidences={"a": 0.5, "b": 0.6},
        trust_weights={"a": 0.5, "b": 0.5},
        substrate_fidelities={"a": 0.5, "b": 0.5},
        hidden_state_fidelities={"a": 0.5, "b": 0.5},
        cache_reuse_fidelities={"a": 0.5, "b": 0.5},
        retrieval_fidelities={"a": 0.5, "b": 0.5},
        replay_fidelities={"a": 0.3, "b": 0.9})
    assert comp.replay_aware
    w = emit_uncertainty_v8_witness(composite=comp)
    assert w.pessimistic_le_weighted
    assert w.weighted_le_optimistic
    assert w.replay_aware


def test_disagreement_algebra_v6_replay():
    from coordpy.disagreement_algebra import AlgebraTrace
    from coordpy.disagreement_algebra_v6 import (
        emit_disagreement_algebra_v6_witness,
    )
    trace = AlgebraTrace.empty()
    pa = [0.1] * 6; pb = [0.2] * 6; pc = [0.3] * 6

    def ok_oracle():
        return (True, 0.0)
    w = emit_disagreement_algebra_v6_witness(
        trace=trace, probe_a=pa, probe_b=pb, probe_c=pc,
        replay_controller_oracle=ok_oracle)
    assert w.replay_controller_equiv_ok

    def bad_oracle():
        return (False, 10.0)
    w2 = emit_disagreement_algebra_v6_witness(
        trace=trace, probe_a=pa, probe_b=pb, probe_c=pc,
        replay_controller_oracle=bad_oracle)
    assert not w2.replay_controller_equiv_ok


def test_deep_substrate_hybrid_v5_five_way():
    from coordpy.cache_controller_v3 import (
        CacheControllerV3,
        W60_CACHE_POLICY_COMPOSITE_V3,
    )
    from coordpy.deep_proxy_stack_v6 import DeepProxyStackV6
    from coordpy.deep_substrate_hybrid_v5 import (
        DeepSubstrateHybridV5,
        deep_substrate_hybrid_v5_forward,
    )
    from coordpy.kv_bridge_v3 import KVBridgeV3Projection
    from coordpy.kv_bridge_v4 import KVBridgeV4Projection
    from coordpy.kv_bridge_v5 import KVBridgeV5Projection
    from coordpy.replay_controller import ReplayController
    from coordpy.tiny_substrate_v5 import (
        build_default_tiny_substrate_v5,
    )
    p = build_default_tiny_substrate_v5(seed=614)
    v6 = DeepProxyStackV6.init(seed=615)
    proj_v3 = KVBridgeV3Projection.init(
        n_layers=p.config.n_layers,
        n_heads=p.config.n_heads,
        n_kv_heads=p.config.n_kv_heads,
        n_inject_tokens=3,
        carrier_dim=v6.inner_v5.inner_v4.factor_dim,
        d_head=p.config.d_model // p.config.n_heads,
        seed=616)
    proj_v4 = KVBridgeV4Projection.init_from_v3(
        proj_v3, seed_v4=617)
    proj_v5 = KVBridgeV5Projection.init_from_v4(
        proj_v4, seed_v5=618)
    ctrl = CacheControllerV3.init(
        policy=W60_CACHE_POLICY_COMPOSITE_V3,
        d_model=p.config.d_model)
    ctrl.composite_weights = np.array(
        [0.4, 0.2, 0.2, 0.2], dtype=np.float64)
    ctrl.attention_receive_scorer = np.zeros(
        (p.config.n_layers * p.config.n_heads,),
        dtype=np.float64)
    rc = ReplayController()
    hybrid = DeepSubstrateHybridV5.init(
        deep_v6=v6, substrate_v5=p, bridge_v5=proj_v5,
        cache_controller_v3=ctrl, replay_controller=rc,
        substrate_back_inject_weight=0.1,
        cache_retention_ratio=0.5)
    rng = np.random.default_rng(8)
    qi = list(rng.standard_normal(v6.in_dim))
    sks = [list(rng.standard_normal(v6.in_dim))
           for _ in range(3)]
    svs = [list(rng.standard_normal(v6.in_dim))
           for _ in range(3)]
    out, w, cache = deep_substrate_hybrid_v5_forward(
        hybrid=hybrid, query_input=qi,
        slot_keys=sks, slot_values=svs)
    assert w.five_way


def test_substrate_adapter_v5_tier():
    from coordpy.substrate_adapter_v5 import (
        W60_SUBSTRATE_TIER_SUBSTRATE_V5_FULL,
        probe_all_v5_adapters,
        probe_tiny_substrate_v5_adapter,
    )
    cap = probe_tiny_substrate_v5_adapter()
    assert cap.tier == W60_SUBSTRATE_TIER_SUBSTRATE_V5_FULL
    mat = probe_all_v5_adapters(
        probe_ollama=False, probe_openai=False)
    assert mat.has_v5_full()


def test_w60_team_end_to_end_envelope():
    from coordpy.w60_team import (
        W60_ENVELOPE_VERIFIER_FAILURE_MODES,
        build_trivial_w60_envelope,
        build_w60_team,
        verify_w60_handoff,
    )
    assert len(W60_ENVELOPE_VERIFIER_FAILURE_MODES) >= 50
    team = build_w60_team(seed=60001)
    env = team.step(turn_index=1, w59_outer_cid="w59_x")
    res = verify_w60_handoff(env)
    assert res["ok"], f"failures: {res['failures']}"
    assert env.five_way_used
    assert env.substrate_v5_used
    triv = build_trivial_w60_envelope(w59_outer_cid="w59_y")
    triv_res = verify_w60_handoff(triv)
    assert not triv_res["ok"]
    assert len(triv_res["failures"]) >= 19
