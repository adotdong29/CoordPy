"""W68 module focused tests (post-W67 milestone).

Per-module sanity tests for the W68 line. R-152/R-153/R-154/R-155
suites exercise the H-bars end-to-end.
"""

from __future__ import annotations


def test_tiny_substrate_v13_determinism_and_axes():
    from coordpy.tiny_substrate_v13 import (
        build_default_tiny_substrate_v13,
        forward_tiny_substrate_v13,
        emit_tiny_substrate_v13_forward_witness,
        record_partial_contradiction_witness_v13,
        trigger_agent_replacement_v13,
        record_prefix_reuse_v13,
        substrate_prefix_reuse_flops_v13,
        tokenize_bytes_v13,
    )
    p = build_default_tiny_substrate_v13(seed=681)
    ids = tokenize_bytes_v13("w68-test", max_len=12)
    t1, c1 = forward_tiny_substrate_v13(p, ids)
    t2, _ = forward_tiny_substrate_v13(p, ids)
    assert t1.cid() == t2.cid()
    assert p.config.v12.v11.v10.v9.n_layers == 15
    assert c1.partial_contradiction_witness.shape == (
        p.config.v12.v11.v10.v9.n_layers,
        p.config.v12.v11.v10.v9.n_heads,
        p.config.v12.v11.v10.v9.max_len)
    assert t1.v13_gate_score_per_layer.shape == (
        p.config.v12.v11.v10.v9.n_layers,)
    record_partial_contradiction_witness_v13(
        c1, layer_index=0, head_index=0, slot=0,
        witness=0.7)
    assert c1.partial_contradiction_witness[0, 0, 0] == 0.7
    trigger_agent_replacement_v13(
        c1, role="planner", replacement_index=1,
        warm_restart_window=3)
    assert c1.agent_replacement_flag["planner"]["replaced"]
    record_prefix_reuse_v13(c1, prefix_cid="cidX", turn=1)
    record_prefix_reuse_v13(c1, prefix_cid="cidX", turn=2)
    assert c1.prefix_reuse_counter["cidX"]["hits"] == 2
    saving = substrate_prefix_reuse_flops_v13(
        n_tokens=128, n_reuses=4)
    assert saving["saving_ratio"] >= 0.8
    w = emit_tiny_substrate_v13_forward_witness(t1, c1)
    assert w.agent_replacement_count == 1
    assert w.prefix_reuse_count >= 2


def test_kv_bridge_v13_nine_target_and_falsifier():
    import numpy as _np
    from coordpy.kv_bridge_v3 import KVBridgeV3Projection
    from coordpy.kv_bridge_v4 import KVBridgeV4Projection
    from coordpy.kv_bridge_v5 import KVBridgeV5Projection
    from coordpy.kv_bridge_v6 import KVBridgeV6Projection
    from coordpy.kv_bridge_v7 import KVBridgeV7Projection
    from coordpy.kv_bridge_v8 import KVBridgeV8Projection
    from coordpy.kv_bridge_v9 import KVBridgeV9Projection
    from coordpy.kv_bridge_v10 import KVBridgeV10Projection
    from coordpy.kv_bridge_v11 import KVBridgeV11Projection
    from coordpy.kv_bridge_v12 import KVBridgeV12Projection
    from coordpy.kv_bridge_v13 import (
        KVBridgeV13Projection,
        compute_agent_replacement_fingerprint_v13,
        fit_kv_bridge_v13_nine_target,
        probe_kv_bridge_v13_partial_contradiction_falsifier,
    )
    from coordpy.tiny_substrate_v13 import (
        build_default_tiny_substrate_v13, tokenize_bytes_v13,
    )
    p = build_default_tiny_substrate_v13(seed=682)
    cfg = p.config.v12.v11.v10.v9
    d_head = cfg.d_model // cfg.n_heads
    proj = KVBridgeV13Projection.init_from_v12(
        KVBridgeV12Projection.init_from_v11(
            KVBridgeV11Projection.init_from_v10(
                KVBridgeV10Projection.init_from_v9(
                    KVBridgeV9Projection.init_from_v8(
                        KVBridgeV8Projection.init_from_v7(
                            KVBridgeV7Projection.init_from_v6(
                                KVBridgeV6Projection.init_from_v5(
                                    KVBridgeV5Projection.init_from_v4(
                                        KVBridgeV4Projection.init_from_v3(
                                            KVBridgeV3Projection.init(
                                                n_layers=cfg.n_layers,
                                                n_heads=cfg.n_heads,
                                                n_kv_heads=cfg.n_kv_heads,
                                                n_inject_tokens=3,
                                                carrier_dim=8,
                                                d_head=d_head,
                                                seed=68200),
                                            seed_v4=68201),
                                        seed_v5=68202),
                                    seed_v6=68203),
                                seed_v7=68204),
                            seed_v8=68205),
                        seed_v9=68206),
                    seed_v10=68207),
                seed_v11=68208),
            seed_v12=68209),
        seed_v13=68210)
    rng = _np.random.default_rng(682)
    ids = tokenize_bytes_v13("kv13", max_len=4)
    while len(ids) < 4:
        ids.append(0)
    carriers = [list(rng.standard_normal(8)) for _ in range(4)]
    targets = []
    for i in range(9):
        t = _np.zeros(cfg.vocab_size)
        t[20 + 15 * i] = 1.0 / (i + 1)
        targets.append(t.tolist())
    new_proj, report = fit_kv_bridge_v13_nine_target(
        params=p, projection=proj,
        train_carriers=carriers,
        target_delta_logits_stack=targets,
        follow_up_token_ids=ids, n_directions=3)
    assert report.n_targets == 9
    assert (
        report.partial_contradiction_post
        <= report.partial_contradiction_pre + 1e-3)
    fp = compute_agent_replacement_fingerprint_v13(
        role="planner", replacement_index=1,
        task_id="t", team_id="x",
        branch_id="b1", warm_restart_window=2)
    assert len(fp) == 50
    fals = probe_kv_bridge_v13_partial_contradiction_falsifier(
        partial_contradiction_flag=0.5)
    assert fals.falsifier_score == 0.0


def test_hsb_v12_nine_target_and_probe():
    import numpy as _np
    from coordpy.hidden_state_bridge_v12 import (
        HiddenStateBridgeV12Projection,
        compute_hsb_v12_agent_replacement_margin,
        fit_hsb_v12_nine_target,
        probe_hsb_v12_hidden_vs_agent_replacement,
    )
    from coordpy.hidden_state_bridge_v11 import (
        HiddenStateBridgeV11Projection,
    )
    from coordpy.hidden_state_bridge_v10 import (
        HiddenStateBridgeV10Projection,
    )
    from coordpy.hidden_state_bridge_v9 import (
        HiddenStateBridgeV9Projection,
    )
    from coordpy.hidden_state_bridge_v8 import (
        HiddenStateBridgeV8Projection,
    )
    from coordpy.hidden_state_bridge_v7 import (
        HiddenStateBridgeV7Projection,
    )
    from coordpy.hidden_state_bridge_v6 import (
        HiddenStateBridgeV6Projection,
    )
    from coordpy.hidden_state_bridge_v5 import (
        HiddenStateBridgeV5Projection,
    )
    from coordpy.hidden_state_bridge_v4 import (
        HiddenStateBridgeV4Projection,
    )
    from coordpy.hidden_state_bridge_v3 import (
        HiddenStateBridgeV3Projection,
    )
    from coordpy.hidden_state_bridge_v2 import (
        HiddenStateBridgeV2Projection,
    )
    from coordpy.tiny_substrate_v13 import (
        build_default_tiny_substrate_v13,
    )
    p = build_default_tiny_substrate_v13(seed=683)
    cfg = p.config.v12.v11.v10.v9
    proj = HiddenStateBridgeV12Projection.init_from_v11(
        HiddenStateBridgeV11Projection.init_from_v10(
            HiddenStateBridgeV10Projection.init_from_v9(
                HiddenStateBridgeV9Projection.init_from_v8(
                    HiddenStateBridgeV8Projection.init_from_v7(
                        HiddenStateBridgeV7Projection.init_from_v6(
                            HiddenStateBridgeV6Projection.init_from_v5(
                                HiddenStateBridgeV5Projection.init_from_v4(
                                    HiddenStateBridgeV4Projection.init_from_v3(
                                        HiddenStateBridgeV3Projection.init_from_v2(
                                            HiddenStateBridgeV2Projection.init(
                                                target_layers=(1, 3),
                                                n_tokens=6, carrier_dim=6,
                                                d_model=cfg.d_model,
                                                seed=68300),
                                            n_heads=cfg.n_heads,
                                            seed_v3=68301),
                                        seed_v4=68302),
                                    n_positions=3, seed_v5=68303),
                                seed_v6=68304),
                            seed_v7=68305),
                        seed_v8=68306),
                    seed_v9=68307),
                seed_v10=68308),
            seed_v11=68309),
        seed_v12=68310)
    rng = _np.random.default_rng(683)
    ids = list(range(1, 5))
    carriers = [list(rng.standard_normal(6)) for _ in range(4)]
    targets = []
    for i in range(9):
        t = _np.zeros(cfg.vocab_size)
        t[20 + 15 * i] = 1.0 / (i + 1)
        targets.append(t.tolist())
    new_proj, report = fit_hsb_v12_nine_target(
        params=p.v3_params, projection=proj,
        train_carriers=carriers,
        target_delta_logits_stack=targets,
        follow_up_token_ids=ids, n_directions=3)
    assert report.n_targets == 9
    L, H = 4, 4
    hres = _np.full((L, H), 0.1)
    ares = _np.full((L, H), 0.5)
    r = probe_hsb_v12_hidden_vs_agent_replacement(
        n_layers=L, n_heads=H,
        hidden_residual_l2_grid=hres.tolist(),
        agent_replacement_residual_l2_grid=ares.tolist())
    assert 0.0 <= r["win_rate_mean"] <= 1.0
    margin = compute_hsb_v12_agent_replacement_margin(
        hidden_residual_l2=0.05, kv_residual_l2=0.2,
        prefix_residual_l2=0.2, replay_residual_l2=0.2,
        recover_residual_l2=0.2,
        branch_merge_residual_l2=0.2,
        agent_replacement_residual_l2=0.2)
    assert margin > 0.0


def test_replay_controller_v9_fourteen_regimes():
    import numpy as _np
    from coordpy.replay_controller import ReplayCandidate
    from coordpy.replay_controller_v9 import (
        ReplayControllerV9,
        W68_AGENT_REPLACEMENT_ROUTING_LABELS,
        W68_REPLAY_REGIMES_V9,
        fit_replay_controller_v9_per_role,
        fit_replay_v9_agent_replacement_routing_head,
    )
    assert len(W68_REPLAY_REGIMES_V9) == 14
    assert len(W68_AGENT_REPLACEMENT_ROUTING_LABELS) == 6
    ctrl = ReplayControllerV9.init()
    cands = {
        r: [ReplayCandidate(100, 1000, 50, 0.1, 0.0, 0.3,
                            True, True, 0)]
        for r in W68_REPLAY_REGIMES_V9}
    decs = {
        r: ["choose_reuse"] for r in W68_REPLAY_REGIMES_V9}
    ctrl, _ = fit_replay_controller_v9_per_role(
        controller=ctrl, role="planner",
        train_candidates_per_regime=cands,
        train_decisions_per_regime=decs)
    rng = _np.random.default_rng(684)
    X = rng.standard_normal((24, 10))
    labs: list[str] = []
    for i in range(24):
        labs.append(W68_AGENT_REPLACEMENT_ROUTING_LABELS[i % 6])
    ctrl, report = (
        fit_replay_v9_agent_replacement_routing_head(
            controller=ctrl, train_team_features=X.tolist(),
            train_routing_labels=labs))
    assert report.agent_replacement_routing_trained
    assert report.converged


def test_cache_controller_v11_eight_objective():
    import numpy as _np
    from coordpy.cache_controller_v11 import (
        CacheControllerV11,
        fit_eight_objective_ridge_v11,
        fit_per_role_agent_replacement_head_v11,
    )
    cc = CacheControllerV11.init()
    rng = _np.random.default_rng(685)
    X = rng.standard_normal((10, 4))
    cc, report = fit_eight_objective_ridge_v11(
        controller=cc, train_features=X.tolist(),
        target_drop_oracle=X.sum(axis=-1).tolist(),
        target_retrieval_relevance=X[:, 0].tolist(),
        target_hidden_wins=(X[:, 1] - X[:, 2]).tolist(),
        target_replay_dominance=(X[:, 3] * 0.5).tolist(),
        target_team_task_success=(
            X[:, 0] * 0.3 - X[:, 1] * 0.1).tolist(),
        target_team_failure_recovery=(
            X[:, 2] * 0.4 + X[:, 3] * 0.2).tolist(),
        target_branch_merge=(
            X[:, 0] * 0.2 + X[:, 2] * 0.5).tolist(),
        target_partial_contradiction=(
            X[:, 1] * 0.3 + X[:, 3] * 0.4).tolist())
    assert report.n_objectives == 8 and report.converged
    X9 = rng.standard_normal((8, 9))
    cc, rep2 = fit_per_role_agent_replacement_head_v11(
        controller=cc, role="planner",
        train_features=X9.tolist(),
        target_replacement_priorities=(
            X9[:, 0] * 0.4 + X9[:, 8] * 0.3).tolist())
    assert rep2.converged


def test_consensus_v14_twenty_two_stages():
    from coordpy.consensus_fallback_controller_v14 import (
        ConsensusFallbackControllerV14,
        W68_CONSENSUS_V14_STAGES,
    )
    c = ConsensusFallbackControllerV14.init()
    assert len(W68_CONSENSUS_V14_STAGES) >= 22


def test_crc_v16_sixty_five_thousand_buckets():
    from coordpy.corruption_robust_carrier_v16 import (
        CorruptionRobustCarrierV16,
        W68_CRC_V16_ADVERSARIAL_BURST_BITS,
        W68_CRC_V16_KV_FINGERPRINT_BUCKETS,
        emit_corruption_robustness_v16_witness,
    )
    assert W68_CRC_V16_KV_FINGERPRINT_BUCKETS == 65536
    assert W68_CRC_V16_ADVERSARIAL_BURST_BITS == 36
    crc = CorruptionRobustCarrierV16()
    w = emit_corruption_robustness_v16_witness(
        crc_v16=crc, n_probes=4)
    assert w.kv65536_corruption_detect_rate >= 0.0
    assert w.adversarial_36bit_burst_detect_rate >= 0.0


def test_lhr_v20_nineteen_heads():
    from coordpy.long_horizon_retention_v20 import (
        LongHorizonReconstructionV20Head,
        W68_DEFAULT_LHR_V20_MAX_K,
        emit_lhr_v20_witness,
    )
    assert W68_DEFAULT_LHR_V20_MAX_K == 448
    h = LongHorizonReconstructionV20Head.init()
    w = emit_lhr_v20_witness(
        h, carrier=[0.1] * 6, k=16,
        partial_contradiction_indicator=[0.5] * 8)
    assert w.n_heads == 19


def test_ecc_v20_two_to_thirtythree_codes():
    from coordpy.ecc_codebook_v20 import (
        ECCCodebookV20,
        probe_ecc_v20_rate_floor_falsifier,
    )
    cb = ECCCodebookV20.init()
    fal = probe_ecc_v20_rate_floor_falsifier(codebook=cb)
    # Total codes should exceed 2^32 (V20 = V19 * 4 = 2^31 * 4 = 2^33)
    assert int(fal["total_codes"]) >= (1 << 32)
    # The 65536-bit/visible-token target trivially exceeds the
    # log2(total_codes) ceiling.
    assert bool(fal["target_exceeds_ceiling"])


def test_tvs_v17_eighteen_arms():
    from coordpy.transcript_vs_shared_arbiter_v17 import (
        W68_TVS_V17_ARMS,
    )
    assert len(W68_TVS_V17_ARMS) == 18


def test_uncertainty_v16_fifteen_axes():
    from coordpy.uncertainty_layer_v16 import (
        compose_uncertainty_report_v16,
    )
    c = compose_uncertainty_report_v16(
        confidences=[0.7, 0.5],
        trusts=[0.9, 0.8],
        substrate_fidelities=[0.95, 0.9],
        hidden_state_fidelities=[0.95, 0.92],
        cache_reuse_fidelities=[0.95, 0.92],
        retrieval_fidelities=[0.95, 0.93],
        replay_fidelities=[0.92, 0.90],
        attention_pattern_fidelities=[0.95, 0.90],
        replay_dominance_fidelities=[0.88, 0.85],
        hidden_wins_fidelities=[0.86, 0.82],
        replay_dominance_primary_fidelities=[0.84, 0.80],
        team_coordination_fidelities=[0.80, 0.75],
        team_failure_recovery_fidelities=[0.78, 0.74],
        branch_merge_reconciliation_fidelities=[0.78, 0.74],
        partial_contradiction_resolution_fidelities=[0.76, 0.72])
    assert c.n_axes == 15
    assert c.partial_contradiction_aware


def test_disagreement_v14_agent_replacement_falsifier():
    from coordpy.disagreement_algebra import AlgebraTrace
    from coordpy.disagreement_algebra_v14 import (
        agent_replacement_equivalence_falsifier,
        check_agent_replacement_equivalence_identity,
        symmetric_kl,
    )
    # Honest oracle: argmax preserved, kl small, fingerprint matches.
    def honest_oracle():
        return True, 0.05, True
    def dishonest_oracle():
        return True, 0.5, True
    assert check_agent_replacement_equivalence_identity(
        ar_oracle=honest_oracle, ar_floor=0.2)
    assert agent_replacement_equivalence_falsifier(
        ar_oracle=dishonest_oracle, ar_floor=0.2)
    assert symmetric_kl([0.5, 0.5], [0.5, 0.5]) >= 0.0


def test_persistent_v20_nineteen_layers():
    from coordpy.persistent_latent_v20 import (
        W68_DEFAULT_V20_MAX_CHAIN_WALK_DEPTH,
        W68_DEFAULT_V20_N_LAYERS,
        W68_DEFAULT_V20_DISTRACTOR_RANK,
    )
    assert W68_DEFAULT_V20_N_LAYERS == 19
    assert W68_DEFAULT_V20_MAX_CHAIN_WALK_DEPTH == 32768
    assert W68_DEFAULT_V20_DISTRACTOR_RANK == 19


def test_multi_hop_v18_thirteen_axis():
    from coordpy.multi_hop_translator_v18 import (
        W68_DEFAULT_MH_V18_BACKENDS,
        W68_DEFAULT_MH_V18_CHAIN_LEN,
        emit_multi_hop_v18_witness,
    )
    assert len(W68_DEFAULT_MH_V18_BACKENDS) == 44
    assert W68_DEFAULT_MH_V18_CHAIN_LEN == 34
    w = emit_multi_hop_v18_witness(seed=10)
    assert w.thirteen_axis_used
    # n_edges = N * (N-1) = 44*43 = 1892
    assert w.n_edges == 44 * 43


def test_mlsc_v16_partial_contradiction_chain():
    from coordpy.mergeable_latent_capsule_v3 import (
        make_root_capsule_v3,
    )
    from coordpy.mergeable_latent_capsule_v4 import wrap_v3_as_v4
    from coordpy.mergeable_latent_capsule_v5 import wrap_v4_as_v5
    from coordpy.mergeable_latent_capsule_v6 import wrap_v5_as_v6
    from coordpy.mergeable_latent_capsule_v7 import wrap_v6_as_v7
    from coordpy.mergeable_latent_capsule_v8 import wrap_v7_as_v8
    from coordpy.mergeable_latent_capsule_v9 import wrap_v8_as_v9
    from coordpy.mergeable_latent_capsule_v10 import wrap_v9_as_v10
    from coordpy.mergeable_latent_capsule_v11 import wrap_v10_as_v11
    from coordpy.mergeable_latent_capsule_v12 import wrap_v11_as_v12
    from coordpy.mergeable_latent_capsule_v13 import wrap_v12_as_v13
    from coordpy.mergeable_latent_capsule_v14 import wrap_v13_as_v14
    from coordpy.mergeable_latent_capsule_v15 import wrap_v14_as_v15
    from coordpy.mergeable_latent_capsule_v16 import (
        W68_MLSC_V16_KNOWN_ALGEBRA_SIGNATURES,
        emit_mlsc_v16_witness,
        wrap_v15_as_v16,
    )
    v3 = make_root_capsule_v3(
        branch_id="w68_test",
        payload=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
        fact_tags=("w68",), confidence=0.9, trust=0.9,
        turn_index=0)
    v4 = wrap_v3_as_v4(v3)
    v5 = wrap_v4_as_v5(v4)
    v6 = wrap_v5_as_v6(v5)
    v7 = wrap_v6_as_v7(v6)
    v8 = wrap_v7_as_v8(v7)
    v9 = wrap_v8_as_v9(v8)
    v10 = wrap_v9_as_v10(v9)
    v11 = wrap_v10_as_v11(v10)
    v12 = wrap_v11_as_v12(v11)
    v13 = wrap_v12_as_v13(v12)
    v14 = wrap_v13_as_v14(v13)
    v15 = wrap_v14_as_v15(v14)
    v16 = wrap_v15_as_v16(
        v15,
        partial_contradiction_witness_chain=("pc_chain_test",),
        agent_replacement_witness_chain=("ar_chain_test",))
    w = emit_mlsc_v16_witness(v16)
    assert w.partial_contradiction_chain_depth >= 1
    assert w.agent_replacement_chain_depth >= 1
    assert len(W68_MLSC_V16_KNOWN_ALGEBRA_SIGNATURES) == 3


def test_substrate_adapter_v13_full_tier():
    from coordpy.substrate_adapter_v13 import (
        W68_SUBSTRATE_TIER_SUBSTRATE_V13_FULL,
        W68_SUBSTRATE_V13_NEW_AXES,
        probe_all_v13_adapters,
        probe_tiny_substrate_v13_adapter,
    )
    cap = probe_tiny_substrate_v13_adapter()
    assert cap.tier == W68_SUBSTRATE_TIER_SUBSTRATE_V13_FULL
    matrix = probe_all_v13_adapters()
    assert matrix.has_v13_full()
    assert len(W68_SUBSTRATE_V13_NEW_AXES) == 4


def test_masc_v4_v13_beats_v12_baseline():
    from coordpy.multi_agent_substrate_coordinator_v2 import (
        W66_MASC_V2_REGIME_BASELINE,
    )
    from coordpy.multi_agent_substrate_coordinator_v4 import (
        MultiAgentSubstrateCoordinatorV4,
        W68_MASC_V4_REGIMES,
    )
    assert len(W68_MASC_V4_REGIMES) == 7
    masc = MultiAgentSubstrateCoordinatorV4()
    _, agg = masc.run_batch(
        seeds=list(range(15)),
        regime=W66_MASC_V2_REGIME_BASELINE)
    assert agg.v13_beats_v12_rate >= 0.5
    assert agg.tsc_v13_beats_tsc_v12_rate >= 0.5


def test_tcc_v3_partial_contradiction_arbiter():
    from coordpy.multi_agent_substrate_coordinator_v4 import (
        W68_MASC_V4_REGIME_PARTIAL_CONTRADICTION,
        W68_MASC_V4_REGIME_AGENT_REPLACEMENT_WARM_RESTART,
    )
    from coordpy.team_consensus_controller_v3 import (
        TeamConsensusControllerV3,
        W68_TC_V3_DECISION_AGENT_REPLACEMENT_WARM_RESTART,
        W68_TC_V3_DECISION_PARTIAL_CONTRADICTION,
    )
    t = TeamConsensusControllerV3()
    out = t.decide_v3(
        regime=W68_MASC_V4_REGIME_PARTIAL_CONTRADICTION,
        agent_guesses=[1.0, -1.0, 0.5],
        agent_confidences=[0.8, 0.4, 0.7],
        substrate_replay_trust=0.7,
        contradiction_indicators=[0.0, 1.0, 0.0])
    assert out["decision"] == W68_TC_V3_DECISION_PARTIAL_CONTRADICTION
    out2 = t.decide_v3(
        regime=W68_MASC_V4_REGIME_AGENT_REPLACEMENT_WARM_RESTART,
        agent_guesses=[1.0, -1.0, 0.5],
        agent_confidences=[0.8, 0.4, 0.7],
        substrate_replay_trust=0.7,
        replaced_role_indices=[1])
    assert (
        out2["decision"]
        == W68_TC_V3_DECISION_AGENT_REPLACEMENT_WARM_RESTART)


def test_hosted_router_decision_is_deterministic():
    from coordpy.hosted_router_controller import (
        HostedRouterController,
        HostedRoutingRequest,
        default_hosted_registry,
    )
    reg = default_hosted_registry()
    rc1 = HostedRouterController(registry=reg)
    rc2 = HostedRouterController(registry=reg)
    req = HostedRoutingRequest(
        request_cid="abc", input_tokens=1000,
        expected_output_tokens=300,
        require_logprobs=True, require_prefix_cache=True,
        data_policy_required="no_log",
        max_latency_ms=2000.0, max_cost_usd=50.0)
    d1 = rc1.decide(req)
    d2 = rc2.decide(req)
    assert d1.chosen_provider == d2.chosen_provider
    assert d1.cid() == d2.cid()


def test_hosted_logprob_router_fuses_shared_top_k():
    from coordpy.hosted_logprob_router import (
        HostedLogprobRouter, TopKLogprobsPayload,
    )
    r = HostedLogprobRouter()
    res = r.fuse([
        TopKLogprobsPayload(
            "p1", (("a", -0.1), ("b", -0.5), ("c", -1.0))),
        TopKLogprobsPayload(
            "p2", (("a", -0.2), ("b", -0.4), ("c", -1.1))),
    ])
    assert res["fusion_kind"] == "weighted_mean_shared_top_k"
    assert res["fused_distribution"][0][0] == "a"


def test_hosted_cache_aware_savings_positive():
    from coordpy.hosted_cache_aware_planner import (
        HostedCacheAwarePlanner,
        hosted_cache_aware_savings_vs_recompute,
    )
    cp = HostedCacheAwarePlanner()
    planned, report = cp.plan(
        shared_prefix_text="The W68 team's shared prefix" * 5,
        role_blocks=["x", "y", "z"])
    assert report["input_token_savings_ratio"] > 0.5
    s = hosted_cache_aware_savings_vs_recompute(
        n_turns=10, prefix_tokens_per_turn=500,
        role_tokens_per_turn=100,
        hosted_cache_hit_rate=1.0)
    assert s["saving_ratio"] >= 0.5


def test_hosted_real_substrate_boundary_falsifier():
    from coordpy.hosted_real_substrate_boundary import (
        build_default_hosted_real_substrate_boundary,
        build_wall_report,
        probe_hosted_real_substrate_boundary_falsifier,
    )
    b = build_default_hosted_real_substrate_boundary()
    wr = build_wall_report(boundary=b)
    assert len(b.blocked_axes) >= 15
    assert len(wr.real_substrate_only_axes) >= 60
    honest = probe_hosted_real_substrate_boundary_falsifier(
        boundary=b, claimed_axis="hidden_state_read",
        claim_satisfied_at_hosted=False)
    dishonest = (
        probe_hosted_real_substrate_boundary_falsifier(
            boundary=b, claimed_axis="hidden_state_read",
            claim_satisfied_at_hosted=True))
    assert honest.falsifier_score == 0.0
    assert dishonest.falsifier_score == 1.0
