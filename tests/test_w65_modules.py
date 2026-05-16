"""W65 module focused tests (post-W64 milestone).

Per-module sanity tests for the W65 line. R-143/R-144/R-145 suites
exercise the H-bars end-to-end.
"""

from __future__ import annotations

import numpy as np


def test_tiny_substrate_v10_determinism_and_axes():
    from coordpy.tiny_substrate_v10 import (
        build_default_tiny_substrate_v10,
        forward_tiny_substrate_v10,
        emit_tiny_substrate_v10_forward_witness,
        record_hidden_write_merit_v10,
        register_role_kv_bank_v10,
        substrate_checkpoint_v10,
        substrate_restore_v10,
        substrate_checkpoint_reuse_flops_v10,
        tokenize_bytes_v10,
        W65_DEFAULT_V10_MAX_ROLES,
    )
    p = build_default_tiny_substrate_v10(seed=651)
    ids = tokenize_bytes_v10("w65-test", max_len=12)
    t1, c1 = forward_tiny_substrate_v10(p, ids)
    t2, _ = forward_tiny_substrate_v10(p, ids)
    assert t1.cid() == t2.cid()
    assert p.config.v9.n_layers == 12
    assert c1.hidden_write_merit.shape == (
        p.config.v9.n_layers, p.config.v9.n_heads,
        p.config.v9.max_len)
    assert t1.v10_gate_score_per_layer.shape == (
        p.config.v9.n_layers,)
    record_hidden_write_merit_v10(
        c1, layer_index=0, head_index=0, slot=0, merit=0.7)
    assert c1.hidden_write_merit[0, 0, 0] == 0.7
    offset = np.zeros(
        (p.config.v9.n_layers, p.config.v9.n_heads, 4, 8))
    for i in range(W65_DEFAULT_V10_MAX_ROLES + 2):
        register_role_kv_bank_v10(
            c1, role=f"r_{i}", offset_matrix=offset)
    assert len(c1.role_kv_bank) == W65_DEFAULT_V10_MAX_ROLES
    ck = substrate_checkpoint_v10(c1)
    assert substrate_restore_v10(ck, c1)
    saving = substrate_checkpoint_reuse_flops_v10(n_tokens=128)
    assert saving["saving_ratio"] >= 0.5
    w = emit_tiny_substrate_v10_forward_witness(t1, c1)
    assert w.n_roles_in_bank == W65_DEFAULT_V10_MAX_ROLES


def test_kv_bridge_v10_six_target_and_falsifier():
    from coordpy.kv_bridge_v3 import KVBridgeV3Projection
    from coordpy.kv_bridge_v4 import KVBridgeV4Projection
    from coordpy.kv_bridge_v5 import KVBridgeV5Projection
    from coordpy.kv_bridge_v6 import KVBridgeV6Projection
    from coordpy.kv_bridge_v7 import KVBridgeV7Projection
    from coordpy.kv_bridge_v8 import KVBridgeV8Projection
    from coordpy.kv_bridge_v9 import KVBridgeV9Projection
    from coordpy.kv_bridge_v10 import (
        KVBridgeV10Projection,
        fit_kv_bridge_v10_six_target,
        probe_kv_bridge_v10_team_task_falsifier,
    )
    from coordpy.tiny_substrate_v10 import (
        build_default_tiny_substrate_v10, tokenize_bytes_v10,
    )
    p = build_default_tiny_substrate_v10(seed=652)
    proj = KVBridgeV10Projection.init_from_v9(
        KVBridgeV9Projection.init_from_v8(
            KVBridgeV8Projection.init_from_v7(
                KVBridgeV7Projection.init_from_v6(
                    KVBridgeV6Projection.init_from_v5(
                        KVBridgeV5Projection.init_from_v4(
                            KVBridgeV4Projection.init_from_v3(
                                KVBridgeV3Projection.init(
                                    n_layers=p.config.v9.n_layers,
                                    n_heads=p.config.v9.n_heads,
                                    n_kv_heads=p.config.v9.n_kv_heads,
                                    n_inject_tokens=3,
                                    carrier_dim=8,
                                    d_head=(p.config.v9.d_model
                                              // p.config.v9.n_heads),
                                    seed=65200),
                                seed_v4=65201),
                            seed_v5=65202),
                        seed_v6=65203),
                    seed_v7=65204),
                seed_v8=65205),
            seed_v9=65206),
        seed_v10=65207)
    rng = np.random.default_rng(0)
    ids = tokenize_bytes_v10("kv10", max_len=4)
    while len(ids) < 4:
        ids.append(0)
    carriers = [list(rng.standard_normal(8)) for _ in range(4)]
    targets = []
    for i in range(6):
        t = np.zeros(p.config.v9.vocab_size)
        t[100 + 20 * i] = 1.0 / (i + 1)
        targets.append(t.tolist())
    _, report = fit_kv_bridge_v10_six_target(
        params=p, projection=proj, train_carriers=carriers,
        target_delta_logits_stack=targets,
        follow_up_token_ids=ids, n_directions=3)
    assert report.n_targets == 6
    f = probe_kv_bridge_v10_team_task_falsifier(
        team_task_flag=0.3)
    assert f.falsifier_score == 0.0


def test_hsb_v9_six_target_and_hidden_wins_rate():
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
    )
    from coordpy.hidden_state_bridge_v6 import (
        HiddenStateBridgeV6Projection,
    )
    from coordpy.hidden_state_bridge_v7 import (
        HiddenStateBridgeV7Projection,
    )
    from coordpy.hidden_state_bridge_v8 import (
        HiddenStateBridgeV8Projection,
    )
    from coordpy.hidden_state_bridge_v9 import (
        HiddenStateBridgeV9Projection, fit_hsb_v9_six_target,
        probe_hsb_v9_hidden_wins_rate,
        compute_hsb_v9_team_coordination_margin,
    )
    from coordpy.tiny_substrate_v10 import (
        build_default_tiny_substrate_v10, tokenize_bytes_v10,
    )
    p = build_default_tiny_substrate_v10(seed=653)
    hsb9 = HiddenStateBridgeV9Projection.init_from_v8(
        HiddenStateBridgeV8Projection.init_from_v7(
            HiddenStateBridgeV7Projection.init_from_v6(
                HiddenStateBridgeV6Projection.init_from_v5(
                    HiddenStateBridgeV5Projection.init_from_v4(
                        HiddenStateBridgeV4Projection.init_from_v3(
                            HiddenStateBridgeV3Projection.init_from_v2(
                                HiddenStateBridgeV2Projection.init(
                                    target_layers=(1, 3),
                                    n_tokens=6, carrier_dim=6,
                                    d_model=int(p.config.v9.d_model),
                                    seed=65300),
                                n_heads=int(p.config.v9.n_heads),
                                seed_v3=65301),
                            seed_v4=65302),
                        n_positions=3, seed_v5=65303),
                    seed_v6=65304),
                seed_v7=65305),
            seed_v8=65306),
        seed_v9=65307)
    rng = np.random.default_rng(0)
    ids = tokenize_bytes_v10("hsb9", max_len=4)
    while len(ids) < 4:
        ids.append(0)
    carriers = [list(rng.standard_normal(6)) for _ in range(4)]
    targets = []
    for i in range(6):
        t = np.zeros(p.config.v9.vocab_size)
        t[200 + 10 * i] = 0.8 / (i + 1)
        targets.append(t.tolist())
    _, report = fit_hsb_v9_six_target(
        params=p.v3_params, projection=hsb9,
        train_carriers=carriers,
        target_delta_logits_stack=targets, token_ids=ids)
    assert report.n_targets == 6
    probe = probe_hsb_v9_hidden_wins_rate(
        n_layers=2, n_heads=2,
        hidden_residual_l2_per_lh=[[0.1, 0.5], [0.6, 0.2]],
        kv_residual_l2_per_lh=[[0.4, 0.3], [0.5, 0.6]])
    assert 0.0 <= probe["mean_win_rate"] <= 1.0
    margin = compute_hsb_v9_team_coordination_margin(
        hidden_residual_l2=0.1, kv_residual_l2=0.4,
        prefix_residual_l2=0.3, replay_residual_l2=0.2)
    assert margin > 0.0


def test_prefix_v9_k64_and_four_way_decision():
    from coordpy.prefix_state_bridge_v9 import (
        W65_DEFAULT_PREFIX_V9_K_STEPS,
        compute_role_task_fingerprint_v9,
        compare_prefix_vs_hidden_vs_replay_vs_team_v9,
        fit_prefix_drift_curve_predictor_v9,
    )
    from coordpy.tiny_substrate_v5 import (
        build_default_tiny_substrate_v5,
    )
    pv5 = build_default_tiny_substrate_v5(seed=654)
    pred = fit_prefix_drift_curve_predictor_v9(
        params_v5=pv5, prompt_token_ids=[10, 20, 30],
        train_segment_configs=[[(0, 1, "reuse"),
                                  (1, 2, "recompute"),
                                  (2, 3, "drop")]],
        train_chain=[[40, 50], [60, 70], [80, 90]],
        roles=["planner"])
    assert pred.k_steps_v9 == W65_DEFAULT_PREFIX_V9_K_STEPS
    curve = pred.predict_curve_v9(
        reuse_len=1, recompute_len=1, drop_len=1,
        follow_up_tokens=[40, 50], role="planner",
        task_name="abc")
    assert len(curve) == W65_DEFAULT_PREFIX_V9_K_STEPS
    fp = compute_role_task_fingerprint_v9(
        role="planner", task_name="abc")
    assert len(fp) == 20
    dec = compare_prefix_vs_hidden_vs_replay_vs_team_v9(
        prefix_drift_curve=[0.5, 0.5],
        hidden_drift_curve=[0.5, 0.5],
        replay_drift_curve=[0.5, 0.5],
        team_drift_curve=[0.05, 0.05])
    assert dec.decision == "team_wins"


def test_attention_v9_five_stage_and_negative():
    from coordpy.attention_steering_bridge_v2 import (
        AttentionSteeringV2Projection,
    )
    from coordpy.attention_steering_bridge_v9 import (
        steer_attention_and_measure_v9,
    )
    from coordpy.tiny_substrate_v10 import (
        build_default_tiny_substrate_v10, tokenize_bytes_v10,
    )
    p = build_default_tiny_substrate_v10(seed=655)
    ids = tokenize_bytes_v10("attn", max_len=4)[:4]
    while len(ids) < 4:
        ids.append(0)
    proj = AttentionSteeringV2Projection.init(
        n_layers=p.config.v9.n_layers,
        n_heads=p.config.v9.n_heads,
        n_query=len(ids), n_key=len(ids), carrier_dim=6,
        seed=65500)
    rng = np.random.default_rng(65501)
    carrier = list(rng.standard_normal(6))
    w_pos = steer_attention_and_measure_v9(
        params=p.v3_params, carrier=carrier,
        projection=proj, token_ids=ids,
        hellinger_budget=0.15)
    assert w_pos.five_stage_used
    assert w_pos.attention_map_fingerprint_cid != ""
    w_neg = steer_attention_and_measure_v9(
        params=p.v3_params, carrier=carrier,
        projection=proj, token_ids=ids,
        hellinger_budget=-1.0)
    assert w_neg.attention_map_delta_l2 == 0.0
    assert w_neg.attention_map_fingerprint_cid == ""


def test_cache_controller_v8_five_objective_per_role():
    from coordpy.cache_controller_v8 import (
        CacheControllerV8, fit_five_objective_ridge_v8,
        fit_per_role_eviction_head_v8,
        score_per_role_eviction_v8,
        emit_cache_controller_v8_witness,
    )
    rng = np.random.default_rng(656)
    cc = CacheControllerV8.init(fit_seed=656)
    X = rng.standard_normal((20, 4))
    cc, rep = fit_five_objective_ridge_v8(
        controller=cc, train_features=X.tolist(),
        target_drop_oracle=X.sum(axis=-1).tolist(),
        target_retrieval_relevance=X[:, 0].tolist(),
        target_hidden_wins=(X[:, 1] - X[:, 2]).tolist(),
        target_replay_dominance=(X[:, 3] * 0.5).tolist(),
        target_team_task_success=(
            X[:, 0] * 0.3 - X[:, 1] * 0.1).tolist())
    assert rep.n_objectives == 5
    X6 = rng.standard_normal((12, 6))
    cc, rep2 = fit_per_role_eviction_head_v8(
        controller=cc, role="planner",
        train_features=X6.tolist(),
        target_eviction_priorities=(
            X6[:, 0] * 0.4 + X6[:, 5] * 0.3).tolist())
    assert rep2.converged
    score = score_per_role_eviction_v8(
        cc, role="planner", flag_count=1, hidden_write=0.5,
        replay_age=2, attention_receive_l1=0.3,
        cache_key_norm=1.0, mean_similarity_to_others=0.4)
    assert isinstance(score, float)
    w = emit_cache_controller_v8_witness(controller=cc)
    assert "planner" in w.role_tags


def test_replay_controller_v6_team_regime_and_abstain():
    from coordpy.replay_controller import (
        ReplayCandidate, W60_REPLAY_DECISION_REUSE,
        W60_REPLAY_DECISION_RECOMPUTE,
        W60_REPLAY_DECISION_FALLBACK,
        W60_REPLAY_DECISION_ABSTAIN,
    )
    from coordpy.replay_controller_v6 import (
        ReplayControllerV6,
        W65_REPLAY_REGIME_TEAM_SUBSTRATE_COORDINATION,
        W65_REPLAY_REGIMES_V6,
        fit_replay_controller_v6_per_role,
        fit_replay_v6_multi_agent_abstain_head,
    )
    ctrl = ReplayControllerV6.init()
    cands = {
        r: [ReplayCandidate(100, 1000, 50, 0.1, 0.0, 0.3,
                              True, True, 0)]
        for r in W65_REPLAY_REGIMES_V6}
    decs = {r: [W60_REPLAY_DECISION_REUSE]
            for r in W65_REPLAY_REGIMES_V6}
    ctrl, _ = fit_replay_controller_v6_per_role(
        controller=ctrl, role="planner",
        train_candidates_per_regime=cands,
        train_decisions_per_regime=decs)
    assert (len(ctrl.per_role_per_regime_heads)
            == len(W65_REPLAY_REGIMES_V6))
    rng = np.random.default_rng(657)
    X = rng.standard_normal((30, 9))
    labs = []
    for i in range(30):
        if X[i, 0] > 0.5:
            labs.append(W60_REPLAY_DECISION_ABSTAIN)
        elif X[i, 1] > 0.0:
            labs.append(W60_REPLAY_DECISION_REUSE)
        elif X[i, 2] > 0.0:
            labs.append(W60_REPLAY_DECISION_FALLBACK)
        else:
            labs.append(W60_REPLAY_DECISION_RECOMPUTE)
    ctrl, rep = fit_replay_v6_multi_agent_abstain_head(
        controller=ctrl, train_team_features=X.tolist(),
        train_decisions=labs)
    assert rep.multi_agent_abstain_trained
    cand = ReplayCandidate(100, 1000, 50, 0.1, 0.0, 0.3,
                              True, True, 0)
    _dec, _conf, _dom, regime, _dpa, role_used = (
        ctrl.decide_v6(cand, role="planner",
                        team_coordination_flag=0.8,
                        substrate_fidelity=0.9))
    assert regime == W65_REPLAY_REGIME_TEAM_SUBSTRATE_COORDINATION
    assert role_used


def test_persistent_v17_fourteenth_skip():
    from coordpy.persistent_latent_v12 import V12StackedCell
    from coordpy.persistent_latent_v17 import (
        PersistentLatentStateV17Chain,
        emit_persistent_v17_witness,
        step_persistent_state_v17,
    )
    cell = V12StackedCell.init(seed=658)
    chain = PersistentLatentStateV17Chain.empty()
    carrier = [0.5] * cell.state_dim
    state = step_persistent_state_v17(
        cell=cell, prev_state=None,
        carrier_values=carrier,
        turn_index=0, role="r",
        substrate_skip=carrier,
        hidden_wins_skip=carrier,
        prefix_reuse_skip=carrier,
        replay_dominance_witness_skip_v16=carrier,
        team_task_success_skip_v17=carrier)
    chain.add(state)
    w = emit_persistent_v17_witness(chain, state.cid())
    assert w.fourteenth_skip_present
    assert w.n_layers == 16


def test_multi_hop_v15_thirty_five_backends():
    from coordpy.multi_hop_translator_v15 import (
        W65_DEFAULT_MH_V15_BACKENDS,
        W65_DEFAULT_MH_V15_CHAIN_LEN,
        evaluate_dec_chain_len25_fidelity,
    )
    n_b = len(W65_DEFAULT_MH_V15_BACKENDS)
    assert n_b == 35
    assert W65_DEFAULT_MH_V15_CHAIN_LEN == 25
    res = evaluate_dec_chain_len25_fidelity(seed=659)
    assert "ten_axis" in str(res["kind"])


def test_mlsc_v13_team_substrate_chain_inherits():
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
        wrap_v11_as_v12,
    )
    from coordpy.mergeable_latent_capsule_v13 import (
        MergeOperatorV13, wrap_v12_as_v13,
    )
    op = MergeOperatorV13(factor_dim=6)
    def mk(branch, ts, rc):
        v3 = make_root_capsule_v3(
            branch_id=branch, payload=tuple([0.1] * 6),
            fact_tags=("w65",), confidence=0.9, trust=0.9,
            turn_index=0)
        v4 = wrap_v3_as_v4(v3)
        v5 = wrap_v4_as_v5(v4, attention_witness_cid="a")
        v6 = wrap_v5_as_v6(
            v5, attention_witness_chain=("ac",),
            cache_reuse_witness_cid="c")
        v7 = wrap_v6_as_v7(
            v6, retrieval_witness_chain=("rc",),
            controller_witness_cid="ctrl")
        v8 = wrap_v7_as_v8(
            v7, replay_witness_chain=("rc",),
            substrate_witness_chain=("sc",),
            provenance_trust_table={"a": 0.9})
        v9 = wrap_v8_as_v9(
            v8, attention_pattern_witness_chain=("ap",),
            cache_retrieval_witness_chain=("cr",),
            per_layer_head_trust_matrix=((0, 0, 0.9),))
        v10 = wrap_v9_as_v10(
            v9, replay_dominance_witness_chain=("rd",),
            disagreement_wasserstein_distance=0.05)
        v11 = wrap_v10_as_v11(
            v10, hidden_wins_witness_chain=("hw",))
        v12 = wrap_v11_as_v12(
            v11,
            replay_dominance_primary_witness_chain=("rdp",),
            hidden_state_trust_witness_chain=("hst",))
        return wrap_v12_as_v13(
            v12, team_substrate_witness_chain=(ts,),
            role_conditioned_witness_chain=(rc,))
    a = mk("A", "ts_a", "rc_a")
    b = mk("B", "ts_b", "rc_b")
    merged = op.merge([a, b])
    assert "ts_a" in merged.team_substrate_witness_chain
    assert "ts_b" in merged.team_substrate_witness_chain
    assert "rc_a" in merged.role_conditioned_witness_chain
    assert "rc_b" in merged.role_conditioned_witness_chain


def test_consensus_v11_sixteen_stages_and_arbiters():
    from coordpy.consensus_fallback_controller_v11 import (
        ConsensusFallbackControllerV11,
        W65_CONSENSUS_V11_STAGES,
        W65_CONSENSUS_V11_STAGE_MULTI_AGENT_ABSTAIN_ARBITER,
        W65_CONSENSUS_V11_STAGE_TEAM_SUBSTRATE_COORDINATION_ARBITER,
    )
    assert len(W65_CONSENSUS_V11_STAGES) == 16
    ctrl = ConsensusFallbackControllerV11.init()
    res = ctrl.decide_v11(
        payloads=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        trusts=[0.4, 0.4],
        replay_decisions=["choose_abstain", "choose_abstain"],
        transcript_available=False,
        team_substrate_coordination_scores_per_parent=[
            0.7, 0.4],
        multi_agent_abstain_score=0.0,
        corruption_detected_per_parent=[False, False],
        repair_amount=0.0,
        hidden_wins_margins_per_parent=[0.0, 0.0],
        three_way_predictions_per_parent=[
            "kv_wins", "kv_wins"],
        replay_dominance_primary_scores_per_parent=[0.0, 0.0],
        four_way_predictions_per_parent=[
            "kv_wins", "kv_wins"])
    assert (res["stage"]
            == W65_CONSENSUS_V11_STAGE_TEAM_SUBSTRATE_COORDINATION_ARBITER)
    res2 = ctrl.decide_v11(
        payloads=[[0.1, 0.2, 0.3]],
        trusts=[0.4],
        replay_decisions=["choose_abstain"],
        transcript_available=False,
        team_substrate_coordination_scores_per_parent=[0.0],
        multi_agent_abstain_score=0.8,
        corruption_detected_per_parent=[False],
        repair_amount=0.0,
        hidden_wins_margins_per_parent=[0.0],
        three_way_predictions_per_parent=["kv_wins"],
        replay_dominance_primary_scores_per_parent=[0.0],
        four_way_predictions_per_parent=["kv_wins"])
    assert (res2["stage"]
            == W65_CONSENSUS_V11_STAGE_MULTI_AGENT_ABSTAIN_ARBITER)


def test_corruption_v13_8192_and_31bit():
    from coordpy.corruption_robust_carrier_v13 import (
        CorruptionRobustCarrierV13,
        emit_corruption_robustness_v13_witness,
    )
    crc = CorruptionRobustCarrierV13()
    w = emit_corruption_robustness_v13_witness(
        crc_v13=crc, n_probes=24, seed=65900)
    assert w.kv8192_corruption_detect_rate >= 0.95
    assert w.adversarial_31bit_burst_detect_rate >= 0.4


def test_lhr_v17_sixteen_way_and_max_k():
    from coordpy.long_horizon_retention_v17 import (
        LongHorizonReconstructionV17Head,
        emit_lhr_v17_witness,
    )
    head = LongHorizonReconstructionV17Head.init(seed=660)
    w = emit_lhr_v17_witness(
        head, carrier=[0.1] * 8, k=4,
        team_task_success_indicator=[0.5] * 8,
        replay_dominance_indicator=[0.5] * 8,
        hidden_wins_indicator=[0.5] * 8,
        replay_dominance_primary_indicator=[0.6] * 8)
    assert w.sixteen_way_runs
    assert w.max_k == 256


def test_ecc_v17_total_codes_and_bits_per_token():
    from coordpy.ecc_codebook_v5 import (
        W53_DEFAULT_ECC_CODE_DIM,
        W53_DEFAULT_ECC_EMIT_MASK_LEN,
    )
    from coordpy.ecc_codebook_v17 import (
        ECCCodebookV17, compress_carrier_ecc_v17,
        probe_ecc_v17_rate_floor_falsifier,
    )
    from coordpy.quantised_compression import QuantisedBudgetGate
    book = ECCCodebookV17.init(seed=661)
    assert book.total_codes == (1 << 27)
    gate = QuantisedBudgetGate.init(
        in_dim=W53_DEFAULT_ECC_CODE_DIM,
        emit_mask_len=W53_DEFAULT_ECC_EMIT_MASK_LEN,
        seed=662)
    gate.importance_threshold = 0.0
    gate.w_emit.values = [1.0] * len(gate.w_emit.values)
    rng = np.random.default_rng(663)
    carrier = list(rng.standard_normal(
        W53_DEFAULT_ECC_CODE_DIM))
    comp = compress_carrier_ecc_v17(
        carrier, codebook=book, gate=gate)
    assert comp["bits_per_visible_token"] >= 29.0
    f = probe_ecc_v17_rate_floor_falsifier(codebook=book)
    assert f["target_exceeds_ceiling"]


def test_uncertainty_v13_twelve_axis():
    from coordpy.uncertainty_layer_v13 import (
        compose_uncertainty_report_v13,
    )
    res = compose_uncertainty_report_v13(
        confidences=[0.7], trusts=[0.9],
        substrate_fidelities=[0.95],
        hidden_state_fidelities=[0.95],
        cache_reuse_fidelities=[0.95],
        retrieval_fidelities=[0.95],
        replay_fidelities=[0.92],
        team_coordination_fidelities=[0.5])
    assert res.n_axes == 12
    assert res.team_coordination_aware


def test_disagreement_v11_wasserstein():
    from coordpy.disagreement_algebra_v11 import (
        check_wasserstein_equivalence_identity,
        wasserstein_equivalence_falsifier,
        wasserstein_1d_distance,
    )
    assert check_wasserstein_equivalence_identity(
        wasserstein_oracle=lambda: (True, 0.1),
        wasserstein_floor=0.20)
    assert wasserstein_equivalence_falsifier(
        wasserstein_oracle=lambda: (False, 1.0),
        wasserstein_floor=0.20)
    d = wasserstein_1d_distance([1.0, 2.0], [1.0, 2.0])
    assert d == 0.0


def test_tvs_v14_fifteen_arms_sum_to_one():
    from coordpy.transcript_vs_shared_arbiter_v14 import (
        W65_TVS_V14_ARMS, fifteen_arm_compare,
    )
    res = fifteen_arm_compare(
        per_turn_team_substrate_coordination_fidelities=[0.5],
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
    assert len(W65_TVS_V14_ARMS) == 15
    assert res.team_substrate_coordination_used


def test_substrate_adapter_v10_full_tier():
    from coordpy.substrate_adapter_v10 import (
        W65_SUBSTRATE_TIER_SUBSTRATE_V10_FULL,
        probe_tiny_substrate_v10_adapter,
        probe_all_v10_adapters,
    )
    cap = probe_tiny_substrate_v10_adapter()
    assert cap.tier == W65_SUBSTRATE_TIER_SUBSTRATE_V10_FULL
    matrix = probe_all_v10_adapters()
    assert matrix.has_v10_full()


def test_multi_agent_coordinator_v10_beats_baselines():
    from coordpy.multi_agent_substrate_coordinator import (
        MultiAgentSubstrateCoordinator,
        W65_MASC_POLICY_SUBSTRATE_ROUTED_V10,
        W65_MASC_POLICY_TRANSCRIPT_ONLY,
    )
    masc = MultiAgentSubstrateCoordinator()
    _, agg = masc.run_batch(
        seeds=list(range(15)), n_agents=4, n_turns=12)
    v10_succ = agg.per_policy_success_rate[
        W65_MASC_POLICY_SUBSTRATE_ROUTED_V10]
    t_succ = agg.per_policy_success_rate[
        W65_MASC_POLICY_TRANSCRIPT_ONLY]
    assert v10_succ >= 0.9
    assert v10_succ >= t_succ + 0.1
    assert agg.v10_strictly_beats_rate >= 0.5
    assert agg.v10_visible_tokens_savings_vs_transcript >= 0.5


def test_deep_substrate_hybrid_v10_ten_way():
    from coordpy.cache_controller_v8 import CacheControllerV8
    from coordpy.deep_substrate_hybrid_v10 import (
        DeepSubstrateHybridV10,
        deep_substrate_hybrid_v10_forward,
    )
    from coordpy.deep_substrate_hybrid_v9 import (
        DeepSubstrateHybridV9,
        DeepSubstrateHybridV9ForwardWitness,
    )
    from coordpy.replay_controller_v6 import ReplayControllerV6
    import numpy as _np
    hybrid = DeepSubstrateHybridV10(
        inner_v9=DeepSubstrateHybridV9(inner_v8=None))
    cc = CacheControllerV8.init(fit_seed=664)
    cc.five_objective_head = _np.eye(4, 5)
    rcv6 = ReplayControllerV6.init()
    rcv6.per_role_per_regime_heads[("planner", "x")] = _np.zeros(
        (4, 10))
    rcv6.multi_agent_abstain_head = _np.zeros((4, 10))
    v9w = DeepSubstrateHybridV9ForwardWitness(
        schema="x", hybrid_cid="x",
        inner_v8_witness_cid="x",
        nine_way=True,
        cache_controller_v7_fired=True,
        replay_controller_v5_fired=True,
        four_way_bridge_classifier_fired=True,
        hidden_wins_primary_active=True,
        attention_v8_active=True,
        prefix_v8_drift_predictor_active=True,
        hidden_state_trust_active=True,
        hsb_v8_hidden_wins_primary_active=True,
        mean_replay_dominance=0.4,
        hidden_wins_primary_l1=1.0,
        attention_v8_hellinger_max=0.15,
        hidden_state_trust_ledger_l1=0.5,
        hsb_v8_hidden_wins_primary_margin=0.2)
    w = deep_substrate_hybrid_v10_forward(
        hybrid=hybrid, v9_witness=v9w,
        cache_controller_v8=cc,
        replay_controller_v6=rcv6,
        hidden_write_merit_l1=1.0,
        attention_v9_fingerprint_present=True,
        prefix_v9_predictor_present=True,
        hsb_v9_hidden_wins_rate_mean=0.6,
        n_roles_in_bank=2,
        n_team_invocations=3)
    assert w.ten_way

    # reduces-to-nine-way when one axis is off
    w2 = deep_substrate_hybrid_v10_forward(
        hybrid=hybrid, v9_witness=v9w,
        cache_controller_v8=cc,
        replay_controller_v6=rcv6,
        hidden_write_merit_l1=0.0,    # turn merit off
        attention_v9_fingerprint_present=True,
        prefix_v9_predictor_present=True,
        hsb_v9_hidden_wins_rate_mean=0.6,
        n_roles_in_bank=2,
        n_team_invocations=3)
    assert not w2.ten_way
