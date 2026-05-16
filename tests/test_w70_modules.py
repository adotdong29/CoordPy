"""W70 module sanity tests.

Focused tests for the W70 mechanism modules: V15 substrate, KV V15,
cache V13, replay V11, persistent V22, LHR V22, MLSC V18, consensus
V16, deep substrate hybrid V15, substrate adapter V15, MASC V6, TCC
V5, hosted V3 modules, and the budget-primary handoff V2 coordinator.
"""

from __future__ import annotations


def test_tiny_substrate_v15_determinism_and_axes():
    from coordpy.tiny_substrate_v15 import (
        W70_DEFAULT_V15_N_LAYERS,
        build_default_tiny_substrate_v15,
        forward_tiny_substrate_v15,
        tokenize_bytes_v15,
    )
    p = build_default_tiny_substrate_v15(seed=70100)
    ids = tokenize_bytes_v15("w70_test", max_len=8)
    t1, c1 = forward_tiny_substrate_v15(p, ids)
    t2, c2 = forward_tiny_substrate_v15(p, ids)
    assert t1.cid() == t2.cid()
    assert c1.cid() == c2.cid()
    assert (
        c1.repair_trajectory_cid == c2.repair_trajectory_cid
        and len(c1.repair_trajectory_cid) > 0)
    assert (
        c1.dominant_repair_per_layer.shape
        == (W70_DEFAULT_V15_N_LAYERS,))
    assert (
        t1.budget_primary_gate_per_layer.shape
        == (W70_DEFAULT_V15_N_LAYERS,))
    assert (
        t1.v15_gate_score_per_layer.shape
        == (W70_DEFAULT_V15_N_LAYERS,))
    assert W70_DEFAULT_V15_N_LAYERS == 17


def test_kv_bridge_v15_eleven_target_ridge_and_falsifier():
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
    from coordpy.kv_bridge_v13 import KVBridgeV13Projection
    from coordpy.kv_bridge_v14 import KVBridgeV14Projection
    from coordpy.kv_bridge_v15 import (
        KVBridgeV15Projection, fit_kv_bridge_v15_eleven_target,
        probe_kv_bridge_v15_repair_dominance_falsifier,
        compute_repair_trajectory_fingerprint_v15,
    )
    from coordpy.tiny_substrate_v15 import (
        build_default_tiny_substrate_v15, tokenize_bytes_v15,
    )
    p = build_default_tiny_substrate_v15(seed=70110)
    cfg = p.config.v14.v13.v12.v11.v10.v9
    d_head = cfg.d_model // cfg.n_heads
    b3 = KVBridgeV3Projection.init(
        n_layers=cfg.n_layers, n_heads=cfg.n_heads,
        n_kv_heads=cfg.n_kv_heads, n_inject_tokens=3,
        carrier_dim=8, d_head=d_head, seed=70300)
    b15 = KVBridgeV15Projection.init_from_v14(
        KVBridgeV14Projection.init_from_v13(
            KVBridgeV13Projection.init_from_v12(
                KVBridgeV12Projection.init_from_v11(
                    KVBridgeV11Projection.init_from_v10(
                        KVBridgeV10Projection.init_from_v9(
                            KVBridgeV9Projection.init_from_v8(
                                KVBridgeV8Projection.init_from_v7(
                                    KVBridgeV7Projection.init_from_v6(
                                        KVBridgeV6Projection.init_from_v5(
                                            KVBridgeV5Projection.init_from_v4(
                                                KVBridgeV4Projection.init_from_v3(
                                                    b3, seed_v4=70301),
                                                seed_v5=70302),
                                            seed_v6=70303),
                                        seed_v7=70304),
                                    seed_v8=70305),
                                seed_v9=70306),
                            seed_v10=70307),
                        seed_v11=70308),
                    seed_v12=70309),
                seed_v13=70310),
            seed_v14=70311),
        seed_v15=70312)
    rng = _np.random.default_rng(70313)
    ids = tokenize_bytes_v15("kvr", max_len=4)
    while len(ids) < 4:
        ids.append(0)
    carriers = [list(rng.standard_normal(8)) for _ in range(4)]
    targets = []
    for i in range(11):
        t = _np.zeros(cfg.vocab_size)
        t[20 + 15 * i] = 1.0 / (i + 1)
        targets.append(t.tolist())
    _, rep = fit_kv_bridge_v15_eleven_target(
        params=p, projection=b15,
        train_carriers=carriers,
        target_delta_logits_stack=targets,
        follow_up_token_ids=ids, n_directions=3)
    assert rep.n_targets == 11 and rep.converged
    honest = probe_kv_bridge_v15_repair_dominance_falsifier(
        dominant_repair_label=1)
    assert honest.falsifier_score == 0.0
    fp = compute_repair_trajectory_fingerprint_v15(
        role="planner", repair_trajectory_cid="abc",
        dominant_repair_label=2)
    assert len(fp) == 70


def test_cache_controller_v13_ten_objective_ridge():
    import numpy as _np
    from coordpy.cache_controller_v13 import (
        CacheControllerV13, fit_ten_objective_ridge_v13,
        fit_per_role_budget_primary_head_v13,
    )
    cc13 = CacheControllerV13.init(fit_seed=70123)
    rng = _np.random.default_rng(70124)
    X = rng.standard_normal((10, 4))
    cc13, rep = fit_ten_objective_ridge_v13(
        controller=cc13, train_features=X.tolist(),
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
            X[:, 1] * 0.3 + X[:, 3] * 0.4).tolist(),
        target_multi_branch_rejoin=(
            X[:, 0] * 0.5 + X[:, 1] * 0.2).tolist(),
        target_budget_primary=(
            X[:, 0] * 0.2 + X[:, 1] * 0.3 + X[:, 2] * 0.4
        ).tolist())
    assert rep.converged and rep.n_objectives == 10
    X11 = rng.standard_normal((8, 11))
    _, rep11 = fit_per_role_budget_primary_head_v13(
        controller=cc13, role="planner",
        train_features=X11.tolist(),
        target_budget_primary_priorities=(
            X11[:, 0] * 0.4 + X11[:, 10] * 0.3).tolist())
    assert rep11.converged


def test_replay_controller_v11_eighteen_regimes_and_routing():
    import numpy as _np
    from coordpy.replay_controller import ReplayCandidate
    from coordpy.replay_controller_v11 import (
        ReplayControllerV11,
        W70_BUDGET_PRIMARY_ROUTING_LABELS,
        W70_REPLAY_REGIMES_V11,
        fit_replay_controller_v11_per_role,
        fit_replay_v11_budget_primary_routing_head,
    )
    assert len(W70_REPLAY_REGIMES_V11) == 18
    rc11 = ReplayControllerV11.init()
    cands = {
        r: [ReplayCandidate(
            100, 1000, 50, 0.1, 0.0, 0.3, True, True, 0)]
        for r in W70_REPLAY_REGIMES_V11}
    decs = {r: ["choose_reuse"] for r in W70_REPLAY_REGIMES_V11}
    rc11, _ = fit_replay_controller_v11_per_role(
        controller=rc11, role="planner",
        train_candidates_per_regime=cands,
        train_decisions_per_regime=decs)
    rng = _np.random.default_rng(70130)
    X = rng.standard_normal((40, 11))
    labs = [
        W70_BUDGET_PRIMARY_ROUTING_LABELS[
            i % len(W70_BUDGET_PRIMARY_ROUTING_LABELS)]
        for i in range(40)]
    _, rep = fit_replay_v11_budget_primary_routing_head(
        controller=rc11, train_team_features=X.tolist(),
        train_routing_labels=labs)
    assert (
        rep.post_classification_acc
        >= rep.pre_classification_acc)


def test_persistent_v22_and_lhr_v22_dims():
    from coordpy.persistent_latent_v22 import (
        W70_DEFAULT_V22_MAX_CHAIN_WALK_DEPTH,
        W70_DEFAULT_V22_N_LAYERS,
    )
    from coordpy.long_horizon_retention_v22 import (
        LongHorizonReconstructionV22Head,
        W70_DEFAULT_LHR_V22_MAX_K,
    )
    assert W70_DEFAULT_V22_N_LAYERS == 21
    assert W70_DEFAULT_V22_MAX_CHAIN_WALK_DEPTH == 131072
    head = LongHorizonReconstructionV22Head.init(seed=70140)
    assert head.max_k == W70_DEFAULT_LHR_V22_MAX_K == 576


def test_mlsc_v18_repair_trajectory_chain():
    from coordpy.mergeable_latent_capsule_v3 import (
        make_root_capsule_v3)
    from coordpy.mergeable_latent_capsule_v4 import (
        wrap_v3_as_v4)
    from coordpy.mergeable_latent_capsule_v5 import (
        wrap_v4_as_v5)
    from coordpy.mergeable_latent_capsule_v6 import (
        wrap_v5_as_v6)
    from coordpy.mergeable_latent_capsule_v7 import (
        wrap_v6_as_v7)
    from coordpy.mergeable_latent_capsule_v8 import (
        wrap_v7_as_v8)
    from coordpy.mergeable_latent_capsule_v9 import (
        wrap_v8_as_v9)
    from coordpy.mergeable_latent_capsule_v10 import (
        wrap_v9_as_v10)
    from coordpy.mergeable_latent_capsule_v11 import (
        wrap_v10_as_v11)
    from coordpy.mergeable_latent_capsule_v12 import (
        wrap_v11_as_v12)
    from coordpy.mergeable_latent_capsule_v13 import (
        wrap_v12_as_v13)
    from coordpy.mergeable_latent_capsule_v14 import (
        wrap_v13_as_v14)
    from coordpy.mergeable_latent_capsule_v15 import (
        wrap_v14_as_v15)
    from coordpy.mergeable_latent_capsule_v16 import (
        wrap_v15_as_v16)
    from coordpy.mergeable_latent_capsule_v17 import (
        wrap_v16_as_v17)
    from coordpy.mergeable_latent_capsule_v18 import (
        emit_mlsc_v18_witness, wrap_v17_as_v18,
    )
    v3 = make_root_capsule_v3(
        branch_id="w70_mlsc",
        payload=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
        fact_tags=("w70",), confidence=0.9, trust=0.9,
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
    v16 = wrap_v15_as_v16(v15)
    v17 = wrap_v16_as_v17(v16)
    v18 = wrap_v17_as_v18(
        v17,
        repair_trajectory_chain=("rt0", "rt1"),
        budget_primary_chain=("bp0",))
    w = emit_mlsc_v18_witness(v18)
    assert w.repair_trajectory_chain_depth == 2
    assert w.budget_primary_chain_depth == 1


def test_consensus_v16_twenty_six_stages_and_arbiters():
    from coordpy.consensus_fallback_controller_v16 import (
        ConsensusFallbackControllerV16,
        W70_CONSENSUS_V16_STAGES,
        W70_CONSENSUS_V16_STAGE_BUDGET_PRIMARY_ARBITER,
        W70_CONSENSUS_V16_STAGE_REPAIR_DOMINANCE_ARBITER,
        emit_consensus_v16_witness,
    )
    assert len(W70_CONSENSUS_V16_STAGES) >= 26
    assert (
        W70_CONSENSUS_V16_STAGE_REPAIR_DOMINANCE_ARBITER
        in W70_CONSENSUS_V16_STAGES)
    assert (
        W70_CONSENSUS_V16_STAGE_BUDGET_PRIMARY_ARBITER
        in W70_CONSENSUS_V16_STAGES)
    c = ConsensusFallbackControllerV16.init()
    w = emit_consensus_v16_witness(c)
    assert w.stages == W70_CONSENSUS_V16_STAGES


def test_substrate_adapter_v15_tier():
    from coordpy.substrate_adapter_v15 import (
        W70_SUBSTRATE_TIER_SUBSTRATE_V15_FULL,
        probe_all_v15_adapters,
    )
    matrix = probe_all_v15_adapters()
    in_repo = matrix.by_name().get("tiny_substrate_v15")
    assert in_repo is not None
    assert (
        in_repo.tier
        == W70_SUBSTRATE_TIER_SUBSTRATE_V15_FULL)


def test_masc_v6_ten_regimes_and_v15_strictly_beats_v14_baseline():
    from coordpy.multi_agent_substrate_coordinator_v2 import (
        W66_MASC_V2_REGIME_BASELINE,
    )
    from coordpy.multi_agent_substrate_coordinator_v6 import (
        MultiAgentSubstrateCoordinatorV6,
        W70_MASC_V6_POLICIES,
        W70_MASC_V6_REGIMES,
    )
    assert len(W70_MASC_V6_POLICIES) == 14
    assert len(W70_MASC_V6_REGIMES) == 10
    masc = MultiAgentSubstrateCoordinatorV6()
    _, agg = masc.run_batch(
        seeds=list(range(15)),
        regime=W66_MASC_V2_REGIME_BASELINE)
    assert agg.v15_beats_v14_rate >= 0.5
    assert agg.tsc_v15_beats_tsc_v14_rate >= 0.5


def test_tcc_v5_repair_dominance_and_budget_primary_arbiters():
    from coordpy.multi_agent_substrate_coordinator_v2 import (
        W66_MASC_V2_REGIME_BASELINE)
    from coordpy.team_consensus_controller_v5 import (
        TeamConsensusControllerV5,
        W70_TC_V5_DECISION_BUDGET_PRIMARY,
        W70_TC_V5_DECISION_REPAIR_DOMINANCE,
        W70_TC_V5_DECISION_CONTRADICTION_THEN_REJOIN,
    )
    tcc = TeamConsensusControllerV5()
    out = tcc.decide_v5(
        regime=W66_MASC_V2_REGIME_BASELINE,
        agent_guesses=[1.0, -1.0, 0.5, 0.2],
        agent_confidences=[0.8, 0.6, 0.7, 0.7],
        substrate_replay_trust=0.7,
        visible_token_budget_ratio=0.9,
        dominant_repair_label=1,
        agent_repair_labels=[1, 1, 0, 0])
    assert (
        out["decision"]
        == W70_TC_V5_DECISION_REPAIR_DOMINANCE)
    out2 = tcc.decide_v5(
        regime=W66_MASC_V2_REGIME_BASELINE,
        agent_guesses=[0.5, 0.5, 0.4, 0.5],
        agent_confidences=[0.8, 0.6, 0.7, 0.7],
        substrate_replay_trust=0.7,
        visible_token_budget_ratio=0.3,
        dominant_repair_label=0)
    assert (
        out2["decision"]
        == W70_TC_V5_DECISION_BUDGET_PRIMARY)
    out3 = tcc.decide_v5(
        regime=(
            "contradiction_then_rejoin_under_budget"),
        agent_guesses=[1.0, -1.0, 0.5, 0.2],
        agent_confidences=[0.8, 0.6, 0.7, 0.7],
        substrate_replay_trust=0.7,
        visible_token_budget_ratio=0.3,
        dominant_repair_label=1,
        agent_repair_labels=[1, 1, 0, 0],
        branch_assignments=[0, 1, 2, 0])
    assert (
        out3["decision"]
        == W70_TC_V5_DECISION_CONTRADICTION_THEN_REJOIN)


def test_hosted_router_v3_budget_aware_and_logprob_v3_abstain():
    from coordpy.hosted_logprob_router import (
        TopKLogprobsPayload)
    from coordpy.hosted_logprob_router_v3 import (
        HostedLogprobRouterV3)
    from coordpy.hosted_router_controller import (
        HostedRoutingRequest, default_hosted_registry)
    from coordpy.hosted_router_controller_v2 import (
        HostedRoutingRequestV2)
    from coordpy.hosted_router_controller_v3 import (
        HostedRouterControllerV3, HostedRoutingRequestV3,
    )
    reg = default_hosted_registry()
    rc = HostedRouterControllerV3.init(reg, {
        "openrouter_paid": 0.85, "openai_paid": 0.92})
    req = HostedRoutingRequestV3(
        inner_v2=HostedRoutingRequestV2(
            inner_v1=HostedRoutingRequest(
                request_cid="t_router_v3",
                input_tokens=1000, expected_output_tokens=300,
                require_logprobs=True, require_prefix_cache=True,
                data_policy_required="no_log",
                max_latency_ms=2000.0, max_cost_usd=50.0)),
        visible_token_budget=128, baseline_token_cost=512,
        repair_dominance_label=1)
    d = rc.decide_v3(req)
    assert d.chosen_provider is not None
    assert d.budget_efficiency_score > 0.0
    lr = HostedLogprobRouterV3(abstain_entropy_floor=0.5)
    payloads = [
        TopKLogprobsPayload("p1", tuple(
            (chr(ord("a") + i), -0.1 - i * 0.01)
            for i in range(20))),
        TopKLogprobsPayload("p2", tuple(
            (chr(ord("a") + i), -0.1 - i * 0.005)
            for i in range(20))),
    ]
    res = lr.fuse_v3(
        payloads, visible_token_budget=256,
        baseline_token_cost=512)
    assert "abstain_v3" in str(res.get("fusion_kind", ""))


def test_hosted_cache_v3_savings_65pct():
    from coordpy.hosted_cache_aware_planner_v3 import (
        hosted_cache_aware_savings_v3_vs_recompute)
    sav = hosted_cache_aware_savings_v3_vs_recompute(
        n_roles=8, n_turns=8, hosted_cache_hit_rate=1.0)
    assert sav["saving_ratio"] >= 0.65


def test_hosted_cost_v3_abstain_when_budget_violated():
    from coordpy.hosted_cost_planner import HostedCostPlanSpec
    from coordpy.hosted_cost_planner_v2 import (
        HostedCostPlanSpecV2)
    from coordpy.hosted_cost_planner_v3 import (
        HostedCostPlanSpecV3, plan_hosted_cost_v3,
    )
    from coordpy.hosted_router_controller import (
        default_hosted_registry)
    reg = default_hosted_registry()
    qs = {p.name: 0.8 for p in reg.providers}
    spec_v3 = HostedCostPlanSpecV3(
        inner_v2=HostedCostPlanSpecV2(
            inner_v1=HostedCostPlanSpec(
                n_turns=5, input_tokens_per_turn=500,
                output_tokens_per_turn=200,
                max_cost_per_turn_usd=5.0,
                max_latency_per_turn_ms=2500.0,
                min_quality_score=0.5),
            allow_provider_rotation=True),
        visible_token_budget_per_turn=256,
        baseline_token_cost_per_turn=512,
        abstain_when_budget_violated=True)
    rep = plan_hosted_cost_v3(
        registry=reg, provider_quality_scores=qs,
        spec_v3=spec_v3)
    assert rep.budget_violated
    assert "abstain_v3" in str(rep.rationale_v3)


def test_boundary_v3_blocked_axes_and_falsifier():
    from coordpy.hosted_real_substrate_boundary_v3 import (
        build_default_hosted_real_substrate_boundary_v3,
        build_wall_report_v3,
        probe_hosted_real_substrate_boundary_v3_falsifier,
    )
    b = build_default_hosted_real_substrate_boundary_v3()
    assert len(b.blocked_axes) >= 22
    assert len(b.frontier_blocked_axes) >= 3
    wr = build_wall_report_v3(boundary=b)
    assert len(wr.real_substrate_only_axes) >= 60
    honest = probe_hosted_real_substrate_boundary_v3_falsifier(
        boundary=b,
        claimed_axis="repair_trajectory_cid",
        claim_satisfied_at_hosted=False)
    dishonest = (
        probe_hosted_real_substrate_boundary_v3_falsifier(
            boundary=b,
            claimed_axis="repair_trajectory_cid",
            claim_satisfied_at_hosted=True))
    assert honest.falsifier_score == 0.0
    assert dishonest.falsifier_score == 1.0


def test_handoff_v2_budget_primary_and_savings():
    from coordpy.hosted_real_handoff_coordinator import (
        HandoffRequest)
    from coordpy.hosted_real_handoff_coordinator_v2 import (
        HandoffRequestV2, HostedRealHandoffCoordinatorV2,
        W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY,
        hosted_real_handoff_v2_budget_primary_savings,
        probe_hosted_real_handoff_v2_repair_dominance_falsifier,
    )
    c = HostedRealHandoffCoordinatorV2()
    # Text-only + dominant_repair_label=1 → V2 promotes to
    # real_substrate_only.
    e = c.decide_v2(
        req_v2=HandoffRequestV2(
            inner_v1=HandoffRequest(
                request_cid="t_handoff",
                needs_text_only=True,
                needs_substrate_state_access=False),
            visible_token_budget=256,
            baseline_token_cost=512,
            dominant_repair_label=1))
    assert (
        str(e.decision_v2)
        == W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY)
    assert e.repair_dominance_alignment == 1.0
    sav = hosted_real_handoff_v2_budget_primary_savings(
        n_turns=12)
    assert sav["saving_ratio"] >= 0.5
    e2 = c.decide_v2(
        req_v2=HandoffRequestV2(
            inner_v1=HandoffRequest(
                request_cid="t_handoff2",
                needs_text_only=True,
                needs_substrate_state_access=False),
            visible_token_budget=256,
            baseline_token_cost=512,
            dominant_repair_label=0))
    honest = (
        probe_hosted_real_handoff_v2_repair_dominance_falsifier(
            envelope_v2=e2, claim_satisfied=False))
    dishonest = (
        probe_hosted_real_handoff_v2_repair_dominance_falsifier(
            envelope_v2=e2, claim_satisfied=True))
    assert honest.falsifier_score == 0.0
    assert dishonest.falsifier_score == 1.0
