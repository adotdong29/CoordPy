"""W64 module focused tests (post-W63 milestone).

Per-module sanity tests for the W64 line. R-137/R-138/R-139 suites
exercise the H-bars end-to-end.
"""

from __future__ import annotations

import numpy as np


def test_tiny_substrate_v9_determinism_and_axes():
    from coordpy.tiny_substrate_v9 import (
        build_default_tiny_substrate_v9,
        forward_tiny_substrate_v9,
        emit_tiny_substrate_v9_forward_witness,
        hidden_wins_primary_summary,
        is_hidden_wins_primary_regime,
        record_hidden_wins_primary_v9,
        record_replay_dominance_witness_v9,
        record_hidden_state_trust_decision_v9,
        tokenize_bytes_v9,
    )
    p = build_default_tiny_substrate_v9(seed=641)
    ids = tokenize_bytes_v9("w64-test", max_len=12)
    t1, c1 = forward_tiny_substrate_v9(p, ids)
    t2, c2 = forward_tiny_substrate_v9(p, ids)
    assert t1.cid() == t2.cid()
    assert c1.hidden_wins_primary.shape == (
        p.config.n_layers, p.config.n_heads, p.config.max_len)
    assert c1.replay_dominance_witness.shape == (
        p.config.n_layers, p.config.n_heads, p.config.max_len)
    assert c1.hidden_state_trust_ledger.shape == (
        p.config.n_layers, p.config.n_heads)
    assert t1.attention_entropy_per_layer.shape == (
        p.config.n_layers,)
    assert (
        c1.cache_similarity_matrix.shape[0]
        == p.config.n_layers)
    record_hidden_wins_primary_v9(
        c1, layer_index=0, head_index=0, slot=0,
        decision="hidden_wins")
    assert c1.hidden_wins_primary[0, 0, 0] == 1.0
    record_replay_dominance_witness_v9(
        c1, layer_index=0, head_index=0, slot=0,
        dominance=0.7)
    assert c1.replay_dominance_witness[0, 0, 0] == 0.7
    pre = float(c1.hidden_state_trust_ledger[0, 0])
    record_hidden_state_trust_decision_v9(
        c1, layer_index=0, head_index=0,
        decision="hidden_wins")
    assert float(c1.hidden_state_trust_ledger[0, 0]) > pre
    summary = hidden_wins_primary_summary(c1)
    assert "primary_margin" in summary
    w = emit_tiny_substrate_v9_forward_witness(t1, c1)
    assert w.n_layers == p.config.n_layers


def test_kv_bridge_v9_five_target():
    from coordpy.kv_bridge_v3 import KVBridgeV3Projection
    from coordpy.kv_bridge_v4 import KVBridgeV4Projection
    from coordpy.kv_bridge_v5 import KVBridgeV5Projection
    from coordpy.kv_bridge_v6 import KVBridgeV6Projection
    from coordpy.kv_bridge_v7 import KVBridgeV7Projection
    from coordpy.kv_bridge_v8 import KVBridgeV8Projection
    from coordpy.kv_bridge_v9 import (
        KVBridgeV9Projection,
        fit_kv_bridge_v9_five_target,
        probe_kv_bridge_v9_hidden_wins_primary_falsifier,
    )
    from coordpy.tiny_substrate_v9 import (
        build_default_tiny_substrate_v9, tokenize_bytes_v9,
    )
    p = build_default_tiny_substrate_v9(seed=642)
    proj_v3 = KVBridgeV3Projection.init(
        n_layers=p.config.n_layers, n_heads=p.config.n_heads,
        n_kv_heads=p.config.n_kv_heads,
        n_inject_tokens=3, carrier_dim=8,
        d_head=p.config.d_model // p.config.n_heads,
        seed=64200)
    proj_v4 = KVBridgeV4Projection.init_from_v3(
        proj_v3, seed_v4=64201)
    proj_v5 = KVBridgeV5Projection.init_from_v4(
        proj_v4, seed_v5=64202)
    proj_v6 = KVBridgeV6Projection.init_from_v5(
        proj_v5, seed_v6=64203)
    proj_v7 = KVBridgeV7Projection.init_from_v6(
        proj_v6, seed_v7=64204)
    proj_v8 = KVBridgeV8Projection.init_from_v7(
        proj_v7, seed_v8=64205)
    proj_v9 = KVBridgeV9Projection.init_from_v8(
        proj_v8, seed_v9=64206)
    rng = np.random.default_rng(0)
    ids = tokenize_bytes_v9("kv9", max_len=4)
    while len(ids) < 4: ids.append(0)
    carriers = [list(rng.standard_normal(8)) for _ in range(4)]
    targets = []
    for i in range(5):
        t = np.zeros(p.config.vocab_size)
        t[100 + 20 * i] = 1.0 / (i + 1)
        targets.append(t.tolist())
    fitted, report = fit_kv_bridge_v9_five_target(
        params=p, projection=proj_v9,
        train_carriers=carriers,
        target_delta_logits_stack=targets,
        follow_up_token_ids=ids, n_directions=3)
    assert report.n_targets == 5
    f = probe_kv_bridge_v9_hidden_wins_primary_falsifier(
        primary_flag=0.5)
    assert f.falsifier_score == 0.0


def test_hsb_v8_hidden_wins_primary_margin():
    from coordpy.hidden_state_bridge_v8 import (
        compute_hsb_v8_hidden_wins_primary_margin,
    )
    pos = compute_hsb_v8_hidden_wins_primary_margin(
        hidden_residual_l2=0.2,
        kv_residual_l2=0.5,
        prefix_residual_l2=0.4)
    neg = compute_hsb_v8_hidden_wins_primary_margin(
        hidden_residual_l2=0.6,
        kv_residual_l2=0.3,
        prefix_residual_l2=0.4)
    assert pos > 0.0
    assert neg < 0.0


def test_prefix_v8_three_way_decision():
    from coordpy.prefix_state_bridge_v8 import (
        compare_prefix_vs_hidden_vs_replay_v8,
    )
    w = compare_prefix_vs_hidden_vs_replay_v8(
        prefix_drift_curve=[0.05, 0.05],
        hidden_drift_curve=[0.5, 0.5],
        replay_drift_curve=[0.3, 0.3])
    assert w.decision == "prefix_wins"


def test_attention_v8_four_stage_zero_under_negative():
    from coordpy.attention_steering_bridge_v2 import (
        AttentionSteeringV2Projection,
    )
    from coordpy.attention_steering_bridge_v8 import (
        steer_attention_and_measure_v8,
    )
    from coordpy.tiny_substrate_v9 import (
        build_default_tiny_substrate_v9, tokenize_bytes_v9,
    )
    p = build_default_tiny_substrate_v9(seed=647)
    ids = tokenize_bytes_v9("attn", max_len=4)[:4]
    while len(ids) < 4: ids.append(0)
    proj = AttentionSteeringV2Projection.init(
        n_layers=p.config.n_layers, n_heads=p.config.n_heads,
        n_query=len(ids), n_key=len(ids), carrier_dim=6,
        seed=64700)
    rng = np.random.default_rng(64701)
    carrier = list(rng.standard_normal(6))
    w_neg = steer_attention_and_measure_v8(
        params=p.v3_params, carrier=carrier,
        projection=proj, token_ids=ids,
        hellinger_budget=-1.0)
    assert w_neg.attention_map_delta_l2 == 0.0


def test_cache_controller_v7_four_objective():
    from coordpy.cache_controller_v7 import (
        CacheControllerV7,
        fit_four_objective_ridge_v7,
    )
    rng = np.random.default_rng(64300)
    cc = CacheControllerV7.init(fit_seed=64300)
    X = rng.standard_normal((16, 4))
    y1 = X.sum(axis=-1)
    y2 = X[:, 0]
    y3 = X[:, 1] - X[:, 2]
    y4 = X[:, 3] * 0.5
    _, report = fit_four_objective_ridge_v7(
        controller=cc, train_features=X.tolist(),
        target_drop_oracle=y1.tolist(),
        target_retrieval_relevance=y2.tolist(),
        target_hidden_wins=y3.tolist(),
        target_replay_dominance=y4.tolist())
    assert report.n_objectives == 4
    assert report.converged


def test_replay_controller_v5_seven_regimes():
    from coordpy.replay_controller_v5 import (
        ReplayControllerV5,
        W64_REPLAY_REGIMES_V5,
        fit_replay_controller_v5_per_regime,
    )
    from coordpy.replay_controller import (
        ReplayCandidate, W60_REPLAY_DECISION_REUSE,
    )
    ctrl = ReplayControllerV5.init()
    cands = {
        r: [ReplayCandidate(
            100, 1000, 50, 0.1, 0.0, 0.3,
            True, True, 0)]
        for r in W64_REPLAY_REGIMES_V5}
    decs = {
        r: [W60_REPLAY_DECISION_REUSE]
        for r in W64_REPLAY_REGIMES_V5}
    _, report = fit_replay_controller_v5_per_regime(
        controller=ctrl, train_candidates_per_regime=cands,
        train_decisions_per_regime=decs)
    assert report.n_regimes == 7
    assert report.converged


def test_substrate_adapter_v9_full_tier():
    from coordpy.substrate_adapter_v9 import (
        W64_SUBSTRATE_TIER_SUBSTRATE_V9_FULL,
        probe_tiny_substrate_v9_adapter,
    )
    cap = probe_tiny_substrate_v9_adapter()
    assert cap.tier == W64_SUBSTRATE_TIER_SUBSTRATE_V9_FULL


def test_persistent_v16_thirteenth_skip():
    from coordpy.persistent_latent_v12 import V12StackedCell
    from coordpy.persistent_latent_v16 import (
        PersistentLatentStateV16Chain,
        emit_persistent_v16_witness,
        step_persistent_state_v16,
    )
    cell = V12StackedCell.init(seed=649)
    chain = PersistentLatentStateV16Chain.empty()
    carrier = [0.5] * int(cell.state_dim)
    state = step_persistent_state_v16(
        cell=cell, prev_state=None,
        carrier_values=carrier,
        turn_index=0, role="r",
        substrate_skip=carrier,
        hidden_wins_skip=carrier,
        prefix_reuse_skip=carrier,
        replay_dominance_witness_skip_v16=carrier)
    chain.add(state)
    w = emit_persistent_v16_witness(chain, state.cid())
    assert w.thirteenth_skip_present
    assert w.n_layers == 15


def test_multi_hop_v14_nine_axis_compromise():
    from coordpy.multi_hop_translator_v14 import (
        evaluate_dec_chain_len21_fidelity,
    )
    res = evaluate_dec_chain_len21_fidelity(seed=64800)
    assert 1 <= res["compromise_threshold"] <= 9
    assert "nine_axis" in res["kind"]


def test_consensus_v10_fourteen_stages_fires():
    from coordpy.consensus_fallback_controller_v10 import (
        ConsensusFallbackControllerV10,
        W64_CONSENSUS_V10_STAGE_REPLAY_DOMINANCE_PRIMARY_ARBITER,
        W64_CONSENSUS_V10_STAGES,
    )
    assert len(W64_CONSENSUS_V10_STAGES) == 14
    ctrl = ConsensusFallbackControllerV10.init()
    res = ctrl.decide_v10(
        payloads=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        trusts=[0.4, 0.4],
        replay_decisions=["choose_abstain", "choose_abstain"],
        transcript_available=False,
        corruption_detected_per_parent=[False, False],
        repair_amount=0.0,
        hidden_wins_margins_per_parent=[0.0, 0.0],
        three_way_predictions_per_parent=[
            "kv_wins", "kv_wins"],
        replay_dominance_primary_scores_per_parent=[
            0.5, 0.0],
        four_way_predictions_per_parent=[
            "replay_wins", "kv_wins"])
    assert (
        res["stage"]
        == W64_CONSENSUS_V10_STAGE_REPLAY_DOMINANCE_PRIMARY_ARBITER)


def test_corruption_v12_witness():
    from coordpy.corruption_robust_carrier_v12 import (
        CorruptionRobustCarrierV12,
        emit_corruption_robustness_v12_witness,
    )
    crc = CorruptionRobustCarrierV12()
    w = emit_corruption_robustness_v12_witness(
        crc_v12=crc, n_probes=8, seed=64900)
    assert w.kv4096_corruption_detect_rate >= 0.5
    assert 0.0 <= w.replay_dominance_recovery_ratio_mean <= 2.0


def test_lhr_v16_fifteen_way():
    from coordpy.long_horizon_retention_v16 import (
        LongHorizonReconstructionV16Head,
        emit_lhr_v16_witness,
    )
    head = LongHorizonReconstructionV16Head.init(seed=650)
    w = emit_lhr_v16_witness(
        head, carrier=[0.1] * 8, k=4,
        replay_dominance_indicator=[1.0] * 8,
        hidden_wins_indicator=[0.5] * 8,
        replay_dominance_primary_indicator=[0.7] * 8)
    assert w.fifteen_way_runs
    assert w.max_k == 192


def test_ecc_v16_total_codes_and_rate_floor():
    from coordpy.ecc_codebook_v16 import (
        ECCCodebookV16,
        probe_ecc_v16_rate_floor_falsifier,
    )
    book = ECCCodebookV16.init(seed=651)
    assert book.total_codes == (1 << 25)
    f = probe_ecc_v16_rate_floor_falsifier(codebook=book)
    assert f["target_exceeds_ceiling"]


def test_tvs_v13_fourteen_arms_sum_to_one():
    from coordpy.transcript_vs_shared_arbiter_v13 import (
        W64_TVS_V13_ARMS, fourteen_arm_compare,
    )
    res = fourteen_arm_compare(
        per_turn_replay_dominance_primary_fidelities=[0.5],
        per_turn_hidden_wins_fidelities=[0.5],
        per_turn_replay_dominance_fidelities=[0.6],
        per_turn_confidences=[0.8],
        per_turn_trust_scores=[0.7],
        per_turn_merge_retentions=[0.6],
        per_turn_tw_retentions=[0.6],
        per_turn_substrate_fidelities=[0.5],
        per_turn_hidden_fidelities=[0.4],
        per_turn_cache_fidelities=[0.5],
        per_turn_retrieval_fidelities=[0.6],
        per_turn_replay_fidelities=[0.7],
        per_turn_attention_pattern_fidelities=[0.9],
        budget_tokens=6)
    s = sum(res.pick_rates.values())
    assert abs(s - 1.0) < 1e-6
    assert len(W64_TVS_V13_ARMS) == 14


def test_uncertainty_v12_eleven_axis():
    from coordpy.uncertainty_layer_v12 import (
        compose_uncertainty_report_v12,
    )
    res = compose_uncertainty_report_v12(
        confidences=[0.7], trusts=[0.9],
        substrate_fidelities=[0.95],
        hidden_state_fidelities=[0.95],
        cache_reuse_fidelities=[0.95],
        retrieval_fidelities=[0.95],
        replay_fidelities=[0.92],
        hidden_wins_fidelities=[0.5],
        replay_dominance_primary_fidelities=[0.5])
    assert res.n_axes == 11
    assert res.replay_dominance_primary_aware


def test_disagreement_v10_tv_identity():
    from coordpy.disagreement_algebra_v10 import (
        check_tv_equivalence_identity,
        tv_equivalence_falsifier,
    )
    ok = check_tv_equivalence_identity(
        tv_oracle=lambda: (True, 0.1), tv_floor=0.30)
    assert ok
    fals = tv_equivalence_falsifier(
        tv_oracle=lambda: (False, 1.0), tv_floor=0.30)
    assert fals


def test_mlsc_v12_replay_dominance_primary_union():
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
        wrap_v7_as_v8,
    )
    from coordpy.mergeable_latent_capsule_v9 import (
        wrap_v8_as_v9,
    )
    from coordpy.mergeable_latent_capsule_v10 import (
        wrap_v9_as_v10,
    )
    from coordpy.mergeable_latent_capsule_v11 import (
        wrap_v10_as_v11,
    )
    from coordpy.mergeable_latent_capsule_v12 import (
        MergeOperatorV12, wrap_v11_as_v12,
    )
    op = MergeOperatorV12(factor_dim=6)
    def mk(branch, rdp, hst):
        v3 = make_root_capsule_v3(
            branch_id=branch, payload=tuple([0.1] * 6),
            fact_tags=("w64",), confidence=0.9, trust=0.9,
            turn_index=0)
        v4 = wrap_v3_as_v4(v3)
        v5 = wrap_v4_as_v5(v4, attention_witness_cid="a")
        v6 = wrap_v5_as_v6(
            v5, attention_witness_chain=("a_chain",),
            cache_reuse_witness_cid="c")
        v7 = wrap_v6_as_v7(
            v6, retrieval_witness_chain=("r_chain",),
            controller_witness_cid="ctrl")
        v8 = wrap_v7_as_v8(
            v7, replay_witness_chain=("rc",),
            substrate_witness_chain=("sc",),
            provenance_trust_table={"a": 0.9})
        v9 = wrap_v8_as_v9(
            v8,
            attention_pattern_witness_chain=("ap",),
            cache_retrieval_witness_chain=("cr",),
            per_layer_head_trust_matrix=((0, 0, 0.9),))
        v10 = wrap_v9_as_v10(
            v9, replay_dominance_witness_chain=("rd",),
            disagreement_wasserstein_distance=0.05)
        v11 = wrap_v10_as_v11(
            v10, hidden_wins_witness_chain=("hw",))
        return wrap_v11_as_v12(
            v11,
            replay_dominance_primary_witness_chain=(rdp,),
            hidden_state_trust_witness_chain=(hst,))
    a = mk("A", "rdp_a", "hst_a")
    b = mk("B", "rdp_b", "hst_b")
    merged = op.merge([a, b])
    assert "rdp_a" in merged.replay_dominance_primary_witness_chain
    assert "rdp_b" in merged.replay_dominance_primary_witness_chain
    assert "hst_a" in merged.hidden_state_trust_witness_chain
    assert "hst_b" in merged.hidden_state_trust_witness_chain
