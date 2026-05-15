"""W59 module focused tests (post-W58 milestone).

Per-module sanity tests for the W59 line. The R-122/R-123/R-124
suites separately exercise the H-bars end-to-end.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_tiny_substrate_v4_determinism_and_partial_reuse():
    from coordpy.tiny_substrate_v4 import (
        build_default_tiny_substrate_v4,
        extract_partial_prefix_v4,
        forward_tiny_substrate_v4,
        forward_with_partial_prefix_reuse_v4,
        tokenize_bytes_v4,
    )
    p = build_default_tiny_substrate_v4(seed=590)
    ids = tokenize_bytes_v4("w59-determ", max_len=12)
    t1, c1 = forward_tiny_substrate_v4(p, ids)
    t2, c2 = forward_tiny_substrate_v4(p, ids)
    assert t1.cid() == t2.cid()
    # Partial reuse matches full recompute.
    prompt = tokenize_bytes_v4("prefix-test", max_len=12)
    fu = tokenize_bytes_v4("fu", max_len=4)
    partial = extract_partial_prefix_v4(
        p, prompt, prefix_reuse_len=6)
    ft, _, split = forward_with_partial_prefix_reuse_v4(
        p, partial, fu)
    # Compare to a full recompute.
    full_trace, _ = forward_tiny_substrate_v4(
        p, list(prompt) + list(fu))
    full_last = full_trace.logits[-1]
    part_last = ft.logits[-1]
    assert float(np.max(np.abs(full_last - part_last))) < 1e-9
    assert int(split["flop_recompute_tail"]) >= 0


def test_kv_bridge_v4_banks_distinct_and_fit():
    from coordpy.kv_bridge_v3 import KVBridgeV3Projection
    from coordpy.kv_bridge_v4 import (
        KVBridgeV4Projection,
        bridge_carrier_and_measure_v4,
        fit_kv_bridge_v4_correction,
    )
    from coordpy.tiny_substrate_v4 import (
        build_default_tiny_substrate_v4,
        tokenize_bytes_v4,
    )
    p = build_default_tiny_substrate_v4(seed=591)
    v3p = p.v3_params
    proj_v3 = KVBridgeV3Projection.init(
        n_layers=p.config.n_layers,
        n_heads=p.config.n_heads,
        n_kv_heads=p.config.n_kv_heads,
        n_inject_tokens=3, carrier_dim=8,
        d_head=p.config.d_model // p.config.n_heads,
        seed=59100)
    proj_v4 = KVBridgeV4Projection.init_from_v3(
        proj_v3, seed_v4=59101)
    rng = np.random.default_rng(0)
    ids = tokenize_bytes_v4("kv-bridge", max_len=6)
    carrier = list(rng.standard_normal(8))
    wa = bridge_carrier_and_measure_v4(
        params=v3p, carrier=carrier, projection=proj_v4,
        follow_up_token_ids=ids, role="bank_a")
    wc = bridge_carrier_and_measure_v4(
        params=v3p, carrier=carrier, projection=proj_v4,
        follow_up_token_ids=ids, role="bank_c")
    assert abs(wa.last_logit_l2_perturbation
                - wc.last_logit_l2_perturbation) > 1e-3
    # Fit reduces mean residual.
    carriers = [
        list(rng.standard_normal(8)) for _ in range(4)]
    targets = [3.0] * 4
    fitted, report = fit_kv_bridge_v4_correction(
        params=v3p, projection=proj_v4,
        train_carriers=carriers, train_target_l2=targets,
        follow_up_token_ids=ids)
    assert report.converged
    assert report.fit_used_closed_form


def test_hidden_state_bridge_v3_target_fit():
    from coordpy.hidden_state_bridge_v2 import (
        HiddenStateBridgeV2Projection,
    )
    from coordpy.hidden_state_bridge_v3 import (
        HiddenStateBridgeV3Projection,
        bridge_hidden_state_and_measure_v3,
    )
    from coordpy.tiny_substrate_v4 import (
        build_default_tiny_substrate_v4,
        tokenize_bytes_v4,
    )
    p = build_default_tiny_substrate_v4(seed=592)
    v3p = p.v3_params
    hsb_v2 = HiddenStateBridgeV2Projection.init(
        target_layers=(1, 3), n_tokens=6,
        carrier_dim=6, d_model=p.config.d_model,
        seed=59200)
    hsb_v3 = HiddenStateBridgeV3Projection.init_from_v2(
        hsb_v2, n_heads=p.config.n_heads, seed_v3=59201)
    ids = tokenize_bytes_v4("hsb-target", max_len=6)[:6]
    while len(ids) < 6:
        ids.append(0)
    carrier = list(np.random.default_rng(1).standard_normal(6))
    target = np.zeros(p.config.vocab_size, dtype=np.float64)
    target[100] = 1.0
    w = bridge_hidden_state_and_measure_v3(
        params=v3p, carrier=carrier, projection=hsb_v3,
        token_ids=ids, target_delta_logits=target)
    assert w.fit_used_closed_form
    assert w.target_delta_used


def test_attention_steering_v3_per_head_clip():
    from coordpy.attention_steering_bridge_v2 import (
        AttentionSteeringV2Projection,
    )
    from coordpy.attention_steering_bridge_v3 import (
        steer_attention_and_measure_v3,
    )
    from coordpy.tiny_substrate_v4 import (
        build_default_tiny_substrate_v4,
        tokenize_bytes_v4,
    )
    p = build_default_tiny_substrate_v4(seed=593)
    v3p = p.v3_params
    proj = AttentionSteeringV2Projection.init(
        n_layers=p.config.n_layers,
        n_heads=p.config.n_heads,
        n_query=4, n_key=8, carrier_dim=6,
        seed=59300)
    carrier = list(np.random.default_rng(2).standard_normal(6))
    ids = tokenize_bytes_v4("attn-v3", max_len=4)[:4]
    while len(ids) < 4:
        ids.append(0)
    w = steer_attention_and_measure_v3(
        params=v3p, carrier=carrier, projection=proj,
        token_ids=ids, kl_budget_per_head=0.5,
        do_per_head_dominance=False)
    assert w.per_head_kl_budget_enforced
    # Every head must be ≤ budget.
    for row in w.mean_kl_per_layer_per_head_post:
        for k in row:
            assert float(k) <= 0.5 + 1e-3


def test_cache_controller_v2_learned_retrieval():
    from coordpy.cache_controller_v2 import (
        CacheControllerV2,
        W59_CACHE_POLICY_LEARNED_RETRIEVAL,
        apply_cache_controller_v2_and_measure,
        fit_learned_hidden_cache_controller,
        fit_learned_retrieval_cache_controller,
    )
    from coordpy.tiny_substrate_v4 import (
        build_default_tiny_substrate_v4,
        tokenize_bytes_v4,
    )
    p = build_default_tiny_substrate_v4(seed=594)
    v3p = p.v3_params
    prompt = tokenize_bytes_v4("cc2-test", max_len=10)
    follow = tokenize_bytes_v4("fu", max_len=4)
    # Learned hidden.
    ctrl_h, rep_h = fit_learned_hidden_cache_controller(
        v3p, prompt_token_ids=prompt,
        follow_up_token_ids=follow)
    assert rep_h.converged
    assert rep_h.post_fit_mean_residual <= (
        rep_h.pre_fit_mean_residual + 1e-9)
    # Learned retrieval.
    q = list(np.random.default_rng(3).standard_normal(
        p.config.d_model))
    ctrl_r, rep_r = fit_learned_retrieval_cache_controller(
        v3p, prompt_token_ids=prompt,
        follow_up_token_ids=follow, query_vector=q)
    assert rep_r.converged
    # Apply on a forward.
    w = apply_cache_controller_v2_and_measure(
        v3p, ctrl_r, prompt_token_ids=prompt,
        follow_up_token_ids=follow, retention_ratio=0.5,
        query_vector=q)
    assert w.used_retrieval_query
    assert w.policy == W59_CACHE_POLICY_LEARNED_RETRIEVAL


def test_prefix_state_v3_partial_reuse():
    from coordpy.prefix_state_bridge_v3 import (
        bridge_prefix_state_and_measure_v3,
    )
    from coordpy.tiny_substrate_v4 import (
        build_default_tiny_substrate_v4,
        tokenize_bytes_v4,
    )
    p = build_default_tiny_substrate_v4(seed=595)
    cross = [
        build_default_tiny_substrate_v4(seed=595 + 1000 + i)
        for i in range(2)]
    prompt = tokenize_bytes_v4("prefix-v3-test", max_len=14)
    follow = tokenize_bytes_v4("fu", max_len=4)
    w = bridge_prefix_state_and_measure_v3(
        params_v4=p, prompt_token_ids=prompt,
        follow_up_token_ids=follow,
        prefix_reuse_len=6,
        cross_seed_params_list=cross)
    assert w.partial_reuse_matches_recompute
    assert w.max_abs_partial_reuse_vs_recompute_diff < 1e-9


def test_persistent_v11_chain_walk():
    from coordpy.persistent_latent_v11 import (
        PersistentLatentStateV11Chain,
        V11StackedCell,
        emit_persistent_v11_witness,
        step_persistent_state_v11,
    )
    cell = V11StackedCell.init(seed=596)
    chain = PersistentLatentStateV11Chain.empty()
    prev = None
    for t in range(8):
        carrier = [0.1 * (t + 1)] * cell.state_dim
        prev = step_persistent_state_v11(
            cell=cell, prev_state=prev,
            carrier_values=carrier,
            turn_index=t, role="r",
            retrieval_skip=carrier,
            retrieval_fidelity=0.9)
        chain.add(prev)
    w = emit_persistent_v11_witness(
        cell=cell, chain=chain, leaf_cid=prev.cid())
    assert w.achieved_chain_walk_depth == 8
    assert w.n_layers == 9


def test_multi_hop_v9_chain_length_and_axes():
    from coordpy.multi_hop_translator_v9 import (
        W59_DEFAULT_MH_V9_BACKENDS,
        W59_DEFAULT_MH_V9_CHAIN_LEN,
        emit_multi_hop_v9_witness,
    )
    w = emit_multi_hop_v9_witness(seed=597)
    assert w.chain_length == 13
    assert len(w.backends) == 14
    assert w.n_edges == 14 * 13
    assert w.retrieval_axis_used


def test_mlsc_v7_retrieval_chain_inheritance():
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
        MergeOperatorV7, wrap_v6_as_v7,
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
    op = MergeOperatorV7(factor_dim=6)
    merged = op.merge(
        [v7], retrieval_witness_chain=("r2",),
        controller_witness_cid="ctrl2")
    assert "r1" in merged.retrieval_witness_chain
    assert "r2" in merged.retrieval_witness_chain


def test_consensus_v5_nine_stages():
    from coordpy.consensus_fallback_controller_v5 import (
        ConsensusFallbackControllerV5,
        W59_CONSENSUS_V5_STAGES,
        W59_CONSENSUS_V5_STAGE_RETRIEVAL,
    )
    assert len(W59_CONSENSUS_V5_STAGES) == 9
    ctrl = ConsensusFallbackControllerV5(
        k_required=3, cosine_floor=0.95,
        trust_threshold=0.99)

    def retrieval_oracle(payloads, q, scores):
        return int(np.argmax(scores))
    ctrl.retrieval_replay_oracle = retrieval_oracle
    rng = np.random.default_rng(4)
    q = list(rng.standard_normal(6))
    p1 = list(rng.standard_normal(6))
    p2 = list(rng.standard_normal(6))
    res = ctrl.decide(
        parent_payloads=[p1, p2],
        parent_trusts=[0.1, 0.1],
        parent_cache_fingerprints=[(1,), (2,)],
        parent_retrieval_scores=[0.3, 0.8],
        query_direction=q, transcript_payload=[])
    assert res["decision_stage"] == W59_CONSENSUS_V5_STAGE_RETRIEVAL


def test_crc_v7_kv128_and_retrieval_topk():
    from coordpy.corruption_robust_carrier_v7 import (
        CorruptionRobustCarrierV7,
        emit_corruption_robustness_v7_witness,
    )
    crc = CorruptionRobustCarrierV7()
    w = emit_corruption_robustness_v7_witness(
        crc_v7=crc, n_probes=12, seed=598)
    assert float(w.kv128_corruption_detect_rate) >= 0.9


def test_lhr_v11_five_way_and_scorer_fit():
    from coordpy.long_horizon_retention_v11 import (
        LongHorizonReconstructionV11Head,
        evaluate_lhr_v11_five_way,
        fit_lhr_v11_retention_scorer,
    )
    head = LongHorizonReconstructionV11Head.init(seed=599)
    rng = np.random.default_rng(5)
    carriers = [list(rng.standard_normal(8)) for _ in range(8)]
    targets = [list(rng.standard_normal(head.out_dim))
                for _ in range(8)]
    subs = [list(rng.standard_normal(32)) for _ in range(8)]
    hids = [list(rng.standard_normal(32)) for _ in range(8)]
    atts = [list(rng.standard_normal(32)) for _ in range(8)]
    rets = [list(rng.standard_normal(32)) for _ in range(8)]
    rep = evaluate_lhr_v11_five_way(
        head, carrier_examples=carriers,
        target_examples=targets, substrate_states=subs,
        hidden_states=hids, attention_states=atts,
        retrieval_states=rets, k=16)
    assert rep["n"] == 8
    # Fit the retention scorer.
    X = rng.standard_normal((16, 8))
    y = X @ rng.standard_normal(8)
    head, residual = fit_lhr_v11_retention_scorer(
        head, train_carriers=X.tolist(),
        train_targets=y.tolist())
    assert head.retention_scorer is not None
    assert head.max_k == 80


def test_ecc_v11_bits_per_token_and_falsifier():
    from coordpy.ecc_codebook_v11 import (
        ECCCodebookV11,
        compress_carrier_ecc_v11,
        probe_ecc_v11_rate_floor_falsifier,
    )
    from coordpy.ecc_codebook_v5 import (
        W53_DEFAULT_ECC_CODE_DIM,
        W53_DEFAULT_ECC_EMIT_MASK_LEN,
    )
    from coordpy.quantised_compression import (
        QuantisedBudgetGate,
    )
    cb = ECCCodebookV11.init(seed=600)
    assert cb.total_codes == 1048576
    gate = QuantisedBudgetGate.init(
        in_dim=W53_DEFAULT_ECC_CODE_DIM,
        emit_mask_len=W53_DEFAULT_ECC_EMIT_MASK_LEN,
        seed=601)
    gate.importance_threshold = 0.0
    gate.w_emit.values = [1.0] * len(gate.w_emit.values)
    rng = np.random.default_rng(6)
    carrier = list(rng.standard_normal(
        W53_DEFAULT_ECC_CODE_DIM))
    comp = compress_carrier_ecc_v11(
        carrier, codebook=cb, gate=gate)
    assert float(comp["bits_per_visible_token"]) >= 22.0
    fal = probe_ecc_v11_rate_floor_falsifier(codebook=cb)
    assert fal["reproduces_cap"]


def test_tvs_v8_nine_arms_sum_to_one():
    from coordpy.transcript_vs_shared_arbiter_v8 import (
        emit_tvs_arbiter_v8_witness,
        nine_arm_compare,
    )
    n = 16
    rng = np.random.default_rng(7)
    res = nine_arm_compare(
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
            0, 1, n).tolist())
    w = emit_tvs_arbiter_v8_witness(result=res)
    assert w.n_arms == 9
    assert w.pick_rates_sum_to_one


def test_uncertainty_v7_brackets_and_retrieval_aware():
    from coordpy.uncertainty_layer_v7 import (
        compose_uncertainty_report_v7,
        emit_uncertainty_v7_witness,
    )
    comp = compose_uncertainty_report_v7(
        component_confidences={"a": 0.5, "b": 0.6},
        trust_weights={"a": 0.5, "b": 0.5},
        substrate_fidelities={"a": 0.5, "b": 0.5},
        hidden_state_fidelities={"a": 0.5, "b": 0.5},
        cache_reuse_fidelities={"a": 0.5, "b": 0.5},
        retrieval_fidelities={"a": 0.3, "b": 0.9})
    assert comp.retrieval_aware
    w = emit_uncertainty_v7_witness(composite=comp)
    assert w.pessimistic_le_weighted
    assert w.weighted_le_optimistic
    assert w.retrieval_aware


def test_disagreement_algebra_v5_retrieval():
    from coordpy.disagreement_algebra import AlgebraTrace
    from coordpy.disagreement_algebra_v5 import (
        emit_disagreement_algebra_v5_witness,
    )
    trace = AlgebraTrace.empty()
    pa = [0.1] * 6
    pb = [0.2] * 6
    pc = [0.3] * 6

    def ok_oracle():
        return (True, 0.0)
    w = emit_disagreement_algebra_v5_witness(
        trace=trace, probe_a=pa, probe_b=pb, probe_c=pc,
        retrieval_replay_oracle=ok_oracle)
    assert w.retrieval_equiv_ok

    def bad_oracle():
        return (False, 10.0)
    w2 = emit_disagreement_algebra_v5_witness(
        trace=trace, probe_a=pa, probe_b=pb, probe_c=pc,
        retrieval_replay_oracle=bad_oracle)
    assert not w2.retrieval_equiv_ok


def test_deep_substrate_hybrid_v4_four_way():
    from coordpy.cache_controller_v2 import (
        CacheControllerV2,
        W59_CACHE_POLICY_LEARNED_RETRIEVAL,
    )
    from coordpy.deep_proxy_stack_v6 import DeepProxyStackV6
    from coordpy.deep_substrate_hybrid_v4 import (
        DeepSubstrateHybridV4,
        deep_substrate_hybrid_v4_forward,
    )
    from coordpy.kv_bridge_v3 import KVBridgeV3Projection
    from coordpy.kv_bridge_v4 import KVBridgeV4Projection
    from coordpy.tiny_substrate_v4 import (
        build_default_tiny_substrate_v4,
    )
    p = build_default_tiny_substrate_v4(seed=602)
    v6 = DeepProxyStackV6.init(seed=603)
    d_head = p.config.d_model // p.config.n_heads
    proj_v3 = KVBridgeV3Projection.init(
        n_layers=p.config.n_layers,
        n_heads=p.config.n_heads,
        n_kv_heads=p.config.n_kv_heads,
        n_inject_tokens=3,
        carrier_dim=v6.inner_v5.inner_v4.factor_dim,
        d_head=d_head, seed=604)
    proj_v4 = KVBridgeV4Projection.init_from_v3(
        proj_v3, seed_v4=605)
    ctrl = CacheControllerV2.init(
        policy=W59_CACHE_POLICY_LEARNED_RETRIEVAL,
        d_model=p.config.d_model)
    ctrl.retrieval_matrix = (
        0.1 * np.eye(p.config.d_model, dtype=np.float64))
    hybrid = DeepSubstrateHybridV4.init(
        deep_v6=v6, substrate_v4=p,
        bridge_v4=proj_v4, cache_controller_v2=ctrl,
        substrate_back_inject_weight=0.1,
        cache_retention_ratio=0.5)
    rng = np.random.default_rng(8)
    qi = list(rng.standard_normal(v6.in_dim))
    sks = [list(rng.standard_normal(v6.in_dim))
           for _ in range(3)]
    svs = [list(rng.standard_normal(v6.in_dim))
           for _ in range(3)]
    out, w, cache = deep_substrate_hybrid_v4_forward(
        hybrid=hybrid, query_input=qi,
        slot_keys=sks, slot_values=svs)
    assert w.four_way
    assert w.retrieval_used


def test_substrate_adapter_v4_tier():
    from coordpy.substrate_adapter_v4 import (
        W59_SUBSTRATE_TIER_SUBSTRATE_V4_FULL,
        probe_all_v4_adapters,
        probe_tiny_substrate_v4_adapter,
    )
    cap = probe_tiny_substrate_v4_adapter()
    assert cap.tier == W59_SUBSTRATE_TIER_SUBSTRATE_V4_FULL
    mat = probe_all_v4_adapters(
        probe_ollama=False, probe_openai=False)
    assert mat.has_v4_full()


def test_w59_team_end_to_end_envelope():
    from coordpy.w59_team import (
        W59_ENVELOPE_VERIFIER_FAILURE_MODES,
        build_trivial_w59_envelope,
        build_w59_team,
        verify_w59_handoff,
    )
    assert len(W59_ENVELOPE_VERIFIER_FAILURE_MODES) >= 48
    team = build_w59_team(seed=59000)
    env = team.step(turn_index=1, w58_outer_cid="w58_x")
    res = verify_w59_handoff(env)
    assert res["ok"], f"failures: {res['failures']}"
    assert env.four_way_used
    assert env.substrate_v4_used
    # Trivial envelope verifier exposes missing modes.
    triv = build_trivial_w59_envelope(w58_outer_cid="w58_y")
    triv_res = verify_w59_handoff(triv)
    assert not triv_res["ok"]
    assert len(triv_res["failures"]) >= 18
