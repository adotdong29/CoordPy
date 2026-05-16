"""W71 focused module tests.

Per-module checks for the W71 mechanism bundle: V16 substrate axes,
KV V16 12-target ridge, cache V14 + restart-priority head, replay
V12 + restart-aware routing head, persistent V23 / LHR V23 / MLSC
V19 / consensus V17, hosted V4 modules, handoff V3, MASC V7 + new
regime, TCC V6. Mirrors the W70 ``test_w70_modules.py`` style.
"""

from __future__ import annotations


def test_tiny_substrate_v16_determinism_and_axes():
    from coordpy.tiny_substrate_v16 import (
        build_default_tiny_substrate_v16,
        emit_tiny_substrate_v16_forward_witness,
        forward_tiny_substrate_v16,
        record_restart_event_v16, record_delay_window_v16,
        tokenize_bytes_v16,
        W71_DEFAULT_V16_N_LAYERS, W71_REPAIR_RESTART_DOMINANCE,
    )
    p = build_default_tiny_substrate_v16(seed=71000)
    ids = tokenize_bytes_v16("w71_modules_test", max_len=8)
    t1, c1 = forward_tiny_substrate_v16(p, ids)
    t2, c2 = forward_tiny_substrate_v16(p, ids)
    assert t1.cid() == t2.cid()
    assert (
        t1.restart_dominance_per_layer.shape
        == (int(W71_DEFAULT_V16_N_LAYERS),))
    assert (
        t1.delayed_repair_gate_per_layer.shape
        == (int(W71_DEFAULT_V16_N_LAYERS),))
    record_restart_event_v16(
        c1, turn=2, restart_kind="agent_restart",
        role="planner")
    record_delay_window_v16(
        c1, restart_turn=2, repair_turn=5,
        delay_turns=3, role="planner")
    t3, c3 = forward_tiny_substrate_v16(
        p, ids, v16_kv_cache=c1, restart_pressure=0.5)
    assert bool(t3.delayed_repair_trajectory_cid)
    rl = c3.restart_dominance_per_layer
    assert int(rl.shape[0]) == int(W71_DEFAULT_V16_N_LAYERS)
    # Some layers fire restart-dominance when both restart and
    # delay window > 0 are recorded.
    import numpy as _np
    assert int(_np.count_nonzero(
        rl == W71_REPAIR_RESTART_DOMINANCE)) > 0
    w = emit_tiny_substrate_v16_forward_witness(t3, c3)
    assert int(w.n_layers) == int(W71_DEFAULT_V16_N_LAYERS)


def test_kv_bridge_v16_twelve_target_ridge_converges():
    from coordpy.kv_bridge_v16 import (
        KVBridgeV16Projection, fit_kv_bridge_v16_twelve_target,
    )
    from coordpy.kv_bridge_v15 import KVBridgeV15Projection
    from coordpy.kv_bridge_v14 import KVBridgeV14Projection
    from coordpy.kv_bridge_v13 import KVBridgeV13Projection
    from coordpy.kv_bridge_v12 import KVBridgeV12Projection
    from coordpy.kv_bridge_v11 import KVBridgeV11Projection
    from coordpy.kv_bridge_v10 import KVBridgeV10Projection
    from coordpy.kv_bridge_v9 import KVBridgeV9Projection
    from coordpy.kv_bridge_v8 import KVBridgeV8Projection
    from coordpy.kv_bridge_v7 import KVBridgeV7Projection
    from coordpy.kv_bridge_v6 import KVBridgeV6Projection
    from coordpy.kv_bridge_v5 import KVBridgeV5Projection
    from coordpy.kv_bridge_v4 import KVBridgeV4Projection
    from coordpy.kv_bridge_v3 import KVBridgeV3Projection
    from coordpy.tiny_substrate_v16 import (
        build_default_tiny_substrate_v16,
    )
    import numpy as _np
    sub = build_default_tiny_substrate_v16(seed=71000)
    cfg = sub.config.v15.v14.v13.v12.v11.v10.v9
    d_head = cfg.d_model // cfg.n_heads
    kv_b3 = KVBridgeV3Projection.init(
        n_layers=cfg.n_layers, n_heads=cfg.n_heads,
        n_kv_heads=cfg.n_kv_heads, n_inject_tokens=3,
        carrier_dim=6, d_head=int(d_head), seed=71008)
    kv_b15 = KVBridgeV15Projection.init_from_v14(
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
                                                    kv_b3, seed_v4=71009),
                                                seed_v5=71010),
                                            seed_v6=71011),
                                        seed_v7=71012),
                                    seed_v8=71013),
                                seed_v9=71014),
                            seed_v10=71015),
                        seed_v11=71016),
                    seed_v12=71017),
                seed_v13=71018),
            seed_v14=71019),
        seed_v15=71020)
    proj = KVBridgeV16Projection.init_from_v15(
        kv_b15, seed_v16=71021)
    rng = _np.random.default_rng(71300)
    carriers = [
        list(rng.standard_normal(8)) for _ in range(4)]
    targets = []
    for i in range(12):
        t = _np.zeros(int(cfg.vocab_size))
        t[20 + 15 * i] = 1.0 / (i + 1)
        targets.append(t.tolist())
    _, rep = fit_kv_bridge_v16_twelve_target(
        params=sub, projection=proj,
        train_carriers=carriers,
        target_delta_logits_stack=targets,
        follow_up_token_ids=[1, 2, 3, 4],
        n_directions=3,
        restart_dominance_target_index=11)
    assert int(rep.n_targets) == 12


def test_kv_bridge_v16_restart_dominance_falsifier_honest():
    from coordpy.kv_bridge_v16 import (
        probe_kv_bridge_v16_restart_dominance_falsifier,
    )
    honest = probe_kv_bridge_v16_restart_dominance_falsifier(
        restart_dominance_flag=1)
    assert float(honest.falsifier_score) == 0.0


def test_cache_v14_eleven_objective_and_restart_head_converge():
    from coordpy.cache_controller_v14 import (
        CacheControllerV14, fit_eleven_objective_ridge_v14,
        fit_per_role_restart_priority_head_v14,
    )
    import numpy as _np
    cc = CacheControllerV14.init(fit_seed=71100)
    rng = _np.random.default_rng(71310)
    X = rng.standard_normal((10, 4))
    cc, rep = fit_eleven_objective_ridge_v14(
        controller=cc, train_features=X.tolist(),
        target_drop_oracle=X.sum(-1).tolist(),
        target_retrieval_relevance=X[:, 0].tolist(),
        target_hidden_wins=(
            X[:, 1] - X[:, 2]).tolist(),
        target_replay_dominance=(X[:, 3] * 0.5).tolist(),
        target_team_task_success=(X[:, 0] * 0.3).tolist(),
        target_team_failure_recovery=(X[:, 2] * 0.4).tolist(),
        target_branch_merge=(X[:, 0] * 0.2).tolist(),
        target_partial_contradiction=(X[:, 3] * 0.4).tolist(),
        target_multi_branch_rejoin=(X[:, 0] * 0.5).tolist(),
        target_budget_primary=(X[:, 2] * 0.3).tolist(),
        target_restart_dominance=(X[:, 3] * 0.2).tolist())
    assert rep.converged
    assert int(rep.n_objectives) == 11
    X12 = rng.standard_normal((10, 12))
    cc, rep2 = fit_per_role_restart_priority_head_v14(
        controller=cc, role="planner",
        train_features=X12.tolist(),
        target_restart_priorities=(
            X12[:, 0] * 0.3 + X12[:, 11] * 0.4).tolist())
    assert rep2.converged


def test_replay_v12_regimes_and_routing_head():
    from coordpy.replay_controller_v12 import (
        ReplayControllerV12,
        W71_REPLAY_REGIMES_V12,
        W71_RESTART_AWARE_ROUTING_LABELS,
        fit_replay_controller_v12_per_role,
        fit_replay_v12_restart_aware_routing_head,
    )
    from coordpy.replay_controller import ReplayCandidate
    import numpy as _np
    assert int(len(W71_REPLAY_REGIMES_V12)) == 19
    assert (
        "delayed_repair_after_restart_regime"
        in W71_REPLAY_REGIMES_V12)
    assert int(len(W71_RESTART_AWARE_ROUTING_LABELS)) == 9
    rc = ReplayControllerV12.init()
    cands = {
        r: [ReplayCandidate(
            100, 1000, 50, 0.1, 0.0, 0.3, True, True, 0)]
        for r in W71_REPLAY_REGIMES_V12}
    decs = {
        r: ["choose_reuse"]
        for r in W71_REPLAY_REGIMES_V12}
    rc, _ = fit_replay_controller_v12_per_role(
        controller=rc, role="planner",
        train_candidates_per_regime=cands,
        train_decisions_per_regime=decs)
    rng = _np.random.default_rng(71400)
    Xt = rng.standard_normal((45, 10))
    labs = [
        W71_RESTART_AWARE_ROUTING_LABELS[
            i % len(W71_RESTART_AWARE_ROUTING_LABELS)]
        for i in range(45)]
    rc, _ = fit_replay_v12_restart_aware_routing_head(
        controller=rc, train_team_features=Xt.tolist(),
        train_routing_labels=labs)
    assert rc.restart_aware_routing_head is not None
    lab, score = rc.decide_restart_aware_routing(
        team_features=Xt[0].tolist())
    assert lab in W71_RESTART_AWARE_ROUTING_LABELS


def test_persistent_v23_step_and_lhr_v23_fit():
    from coordpy.persistent_latent_v12 import V12StackedCell
    from coordpy.persistent_latent_v23 import (
        W71_DEFAULT_V23_DISTRACTOR_RANK,
        W71_DEFAULT_V23_MAX_CHAIN_WALK_DEPTH,
        W71_DEFAULT_V23_N_LAYERS,
        step_persistent_state_v23,
    )
    from coordpy.long_horizon_retention_v23 import (
        LongHorizonReconstructionV23Head,
        fit_lhr_v23_thirteen_layer_scorer,
        emit_lhr_v23_witness,
    )
    import numpy as _np
    assert int(W71_DEFAULT_V23_N_LAYERS) == 22
    assert int(W71_DEFAULT_V23_DISTRACTOR_RANK) == 22
    assert int(W71_DEFAULT_V23_MAX_CHAIN_WALK_DEPTH) >= 262144
    cell = V12StackedCell.init(state_dim=4, seed=71030)
    # First step with no prev_state: EMA from zero baseline lifts
    # carrier toward the skip value.
    s1 = step_persistent_state_v23(
        cell=cell, prev_state=None,
        carrier_values=[0.1] * 4, turn_index=0,
        role="planner",
        restart_dominance_skip_v23=[0.5] * 4,
        restart_dominance_ema_alpha=0.10)
    rdc = s1.restart_dominance_carrier
    assert all(0.0 < float(x) < 0.5 for x in rdc)
    s2 = step_persistent_state_v23(
        cell=cell, prev_state=s1,
        carrier_values=[0.2] * 4, turn_index=1,
        role="planner",
        restart_dominance_skip_v23=[1.0] * 4)
    assert s2.restart_dominance_carrier != s1.restart_dominance_carrier
    # LHR V23.
    head = LongHorizonReconstructionV23Head.init(seed=71200)
    X = _np.random.default_rng(71210).standard_normal((10, 36))
    head2, audit = fit_lhr_v23_thirteen_layer_scorer(
        head=head, train_features=X.tolist(),
        train_targets=(X[:, 0] + X[:, 5]).tolist())
    assert audit["converged"]
    w = emit_lhr_v23_witness(
        head2, carrier=[0.1] * 6, k=16,
        partial_contradiction_indicator=[0.5] * 8,
        multi_branch_rejoin_indicator=[0.5] * 8,
        repair_dominance_indicator=[0.5] * 7,
        restart_indicator=[0.5] * 8)
    assert int(w.n_heads) == 22


def test_mlsc_v19_chains_merge_union():
    from coordpy.mergeable_latent_capsule_v3 import (
        make_root_capsule_v3,
    )
    from coordpy.mergeable_latent_capsule_v4 import wrap_v3_as_v4
    from coordpy.mergeable_latent_capsule_v5 import wrap_v4_as_v5
    from coordpy.mergeable_latent_capsule_v6 import wrap_v5_as_v6
    from coordpy.mergeable_latent_capsule_v7 import wrap_v6_as_v7
    from coordpy.mergeable_latent_capsule_v8 import wrap_v7_as_v8
    from coordpy.mergeable_latent_capsule_v9 import wrap_v8_as_v9
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
        wrap_v12_as_v13,
    )
    from coordpy.mergeable_latent_capsule_v14 import (
        wrap_v13_as_v14,
    )
    from coordpy.mergeable_latent_capsule_v15 import (
        wrap_v14_as_v15,
    )
    from coordpy.mergeable_latent_capsule_v16 import (
        wrap_v15_as_v16,
    )
    from coordpy.mergeable_latent_capsule_v17 import (
        wrap_v16_as_v17,
    )
    from coordpy.mergeable_latent_capsule_v18 import (
        wrap_v17_as_v18,
    )
    from coordpy.mergeable_latent_capsule_v19 import (
        MergeOperatorV19, wrap_v18_as_v19,
    )
    v3 = make_root_capsule_v3(
        branch_id="test", payload=(0.1,) * 6,
        fact_tags=("w71",), confidence=0.9, trust=0.9,
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
    v18 = wrap_v17_as_v18(v17)
    v19_a = wrap_v18_as_v19(
        v18,
        delayed_repair_trajectory_chain=("drt_a",),
        restart_dominance_chain=("rst_a",))
    v19_b = wrap_v18_as_v19(
        v18,
        delayed_repair_trajectory_chain=("drt_b",),
        restart_dominance_chain=("rst_b",))
    merged = MergeOperatorV19(factor_dim=6).merge(
        [v19_a, v19_b])
    assert (
        "drt_a" in merged.delayed_repair_trajectory_chain
        and "drt_b" in merged.delayed_repair_trajectory_chain)
    assert (
        "rst_a" in merged.restart_dominance_chain
        and "rst_b" in merged.restart_dominance_chain)


def test_consensus_v17_28_stages_and_arbiters():
    from coordpy.consensus_fallback_controller_v17 import (
        ConsensusFallbackControllerV17,
        W71_CONSENSUS_V17_STAGES,
        emit_consensus_v17_witness,
    )
    assert len(W71_CONSENSUS_V17_STAGES) >= 28
    c = ConsensusFallbackControllerV17.init()
    out_dr = c.decide_v17(
        payloads=[[1.0], [2.0]], trusts=[0.6, 0.7],
        replay_decisions=["u", "u"],
        delayed_repair_scores_per_parent=[0.4, 0.9])
    assert (
        out_dr["stage"]
        == "delayed_repair_after_restart_arbiter")
    out_ra = c.decide_v17(
        payloads=[[1.0], [2.0]], trusts=[0.6, 0.7],
        replay_decisions=["u", "u"],
        restart_aware_scores_per_parent=[0.8, 0.4])
    assert out_ra["stage"] == "restart_aware_arbiter"
    w = emit_consensus_v17_witness(c)
    assert int(w.restart_aware_stage_fired) >= 1
    assert int(w.delayed_repair_stage_fired) >= 1


def test_hosted_router_v4_restart_pressure_score():
    from coordpy.hosted_router_controller import (
        HostedRoutingRequest, default_hosted_registry,
    )
    from coordpy.hosted_router_controller_v2 import (
        HostedRoutingRequestV2,
    )
    from coordpy.hosted_router_controller_v3 import (
        HostedRoutingRequestV3,
    )
    from coordpy.hosted_router_controller_v4 import (
        HostedRouterControllerV4, HostedRoutingRequestV4,
    )
    reg = default_hosted_registry()
    rc = HostedRouterControllerV4.init(reg, {
        "openrouter_paid": 0.85, "openai_paid": 0.92})
    req = HostedRoutingRequestV4(
        inner_v3=HostedRoutingRequestV3(
            inner_v2=HostedRoutingRequestV2(
                inner_v1=HostedRoutingRequest(
                    request_cid="t_router",
                    input_tokens=1000,
                    expected_output_tokens=300,
                    require_logprobs=True,
                    require_prefix_cache=True,
                    data_policy_required="no_log",
                    max_latency_ms=2000.0,
                    max_cost_usd=50.0),
                weight_cost=1.0, weight_latency=0.5,
                weight_success=0.3),
            visible_token_budget=128,
            baseline_token_cost=512,
            repair_dominance_label=1),
        restart_pressure=0.8,
        weight_restart_pressure=0.6,
        weight_delayed_repair_match=0.4)
    d = rc.decide_v4(req)
    assert d.restart_pressure_score > 0.0
    d2 = rc.decide_v4(req)
    assert d.cid() == d2.cid()


def test_hosted_logprob_v4_restart_aware_abstain_floor():
    from coordpy.hosted_logprob_router import (
        TopKLogprobsPayload,
    )
    from coordpy.hosted_logprob_router_v4 import (
        HostedLogprobRouterV4,
    )
    lr = HostedLogprobRouterV4(
        restart_pressure_floor=0.5,
        restart_abstain_floor_delta=1.5)
    payloads = [
        TopKLogprobsPayload("p1", tuple(
            (chr(ord("a") + i), -1.0 - i * 0.1)
            for i in range(8))),
        TopKLogprobsPayload("p2", tuple(
            (chr(ord("a") + i), -1.0 - i * 0.05)
            for i in range(8))),
    ]
    no = lr.fuse_v4(
        payloads, visible_token_budget=256,
        baseline_token_cost=512, restart_pressure=0.0)
    high = lr.fuse_v4(
        payloads, visible_token_budget=256,
        baseline_token_cost=512, restart_pressure=0.9)
    assert (
        high["effective_abstain_entropy_floor"]
        < no["effective_abstain_entropy_floor"])


def test_hosted_cache_v4_two_layer_savings_above_72_pct():
    from coordpy.hosted_cache_aware_planner_v4 import (
        hosted_cache_aware_savings_v4_vs_recompute,
    )
    sav = hosted_cache_aware_savings_v4_vs_recompute(
        n_roles=10, n_turns=8, hosted_cache_hit_rate=1.0)
    assert float(sav["saving_ratio"]) >= 0.72


def test_hosted_cost_v4_abstain_under_restart():
    from coordpy.hosted_cost_planner import HostedCostPlanSpec
    from coordpy.hosted_cost_planner_v2 import (
        HostedCostPlanSpecV2,
    )
    from coordpy.hosted_cost_planner_v3 import (
        HostedCostPlanSpecV3,
    )
    from coordpy.hosted_cost_planner_v4 import (
        HostedCostPlanSpecV4, plan_hosted_cost_v4,
    )
    from coordpy.hosted_router_controller import (
        default_hosted_registry,
    )
    reg = default_hosted_registry()
    qs = {p.name: 0.8 for p in reg.providers}
    spec = HostedCostPlanSpecV4(
        inner_v3=HostedCostPlanSpecV3(
            inner_v2=HostedCostPlanSpecV2(
                inner_v1=HostedCostPlanSpec(
                    n_turns=5, input_tokens_per_turn=200,
                    output_tokens_per_turn=100,
                    max_cost_per_turn_usd=5.0,
                    max_latency_per_turn_ms=2500.0,
                    min_quality_score=0.5),
                allow_provider_rotation=True),
            visible_token_budget_per_turn=512,
            baseline_token_cost_per_turn=512,
            abstain_when_budget_violated=True),
        restart_pressure=0.9, restart_pressure_cap=0.7,
        abstain_when_restart_violated=True)
    rep = plan_hosted_cost_v4(
        registry=reg, provider_quality_scores=qs,
        spec_v4=spec)
    assert rep.restart_violated
    assert "abstain_v4" in str(rep.rationale_v4)


def test_hosted_boundary_v4_25_blocked_axes_and_falsifier():
    from coordpy.hosted_real_substrate_boundary_v4 import (
        build_default_hosted_real_substrate_boundary_v4,
        probe_hosted_real_substrate_boundary_v4_falsifier,
    )
    b = build_default_hosted_real_substrate_boundary_v4()
    assert int(len(b.blocked_axes)) >= 25
    f_h = probe_hosted_real_substrate_boundary_v4_falsifier(
        boundary=b,
        claimed_axis="delayed_repair_trajectory_cid",
        claim_satisfied_at_hosted=False)
    assert float(f_h.falsifier_score) == 0.0
    f_d = probe_hosted_real_substrate_boundary_v4_falsifier(
        boundary=b,
        claimed_axis="delayed_repair_trajectory_cid",
        claim_satisfied_at_hosted=True)
    assert float(f_d.falsifier_score) == 1.0


def test_handoff_v3_restart_promotion_and_dr_fallback():
    from coordpy.hosted_real_handoff_coordinator import (
        HandoffRequest,
    )
    from coordpy.hosted_real_handoff_coordinator_v2 import (
        HandoffRequestV2,
        W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY,
    )
    from coordpy.hosted_real_handoff_coordinator_v3 import (
        HandoffRequestV3, HostedRealHandoffCoordinatorV3,
        W71_HANDOFF_DECISION_DELAYED_REPAIR_FALLBACK,
    )
    c = HostedRealHandoffCoordinatorV3()
    e_rest = c.decide_v3(
        req_v3=HandoffRequestV3(
            inner_v2=HandoffRequestV2(
                inner_v1=HandoffRequest(
                    request_cid="t_rest", needs_text_only=True,
                    needs_substrate_state_access=False),
                visible_token_budget=256,
                baseline_token_cost=512,
                dominant_repair_label=0),
            restart_pressure=0.8,
            expected_substrate_trust=0.7))
    assert (
        str(e_rest.decision_v3)
        == W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY)
    assert e_rest.restart_alignment == 1.0
    e_dr = c.decide_v3(
        req_v3=HandoffRequestV3(
            inner_v2=HandoffRequestV2(
                inner_v1=HandoffRequest(
                    request_cid="t_dr", needs_text_only=True,
                    needs_substrate_state_access=False),
                visible_token_budget=256,
                baseline_token_cost=512,
                dominant_repair_label=0),
            restart_pressure=0.0,
            delayed_repair_trajectory_cid="drt_xyz",
            delay_turns=3, expected_substrate_trust=0.7))
    assert (
        str(e_dr.decision_v3)
        == W71_HANDOFF_DECISION_DELAYED_REPAIR_FALLBACK)
    assert e_dr.delayed_repair_alignment == 1.0


def test_masc_v7_v16_beats_v15_on_baseline_and_new_regime():
    from coordpy.multi_agent_substrate_coordinator_v2 import (
        W66_MASC_V2_REGIME_BASELINE,
    )
    from coordpy.multi_agent_substrate_coordinator_v7 import (
        MultiAgentSubstrateCoordinatorV7,
        W71_MASC_V7_REGIME_DELAYED_REPAIR_AFTER_RESTART,
    )
    m = MultiAgentSubstrateCoordinatorV7()
    _, agg_b = m.run_batch(
        seeds=list(range(15)),
        regime=W66_MASC_V2_REGIME_BASELINE)
    assert agg_b.v16_beats_v15_rate >= 0.5
    assert agg_b.tsc_v16_beats_tsc_v15_rate >= 0.5
    _, agg_drar = m.run_batch(
        seeds=list(range(15)),
        regime=(
            W71_MASC_V7_REGIME_DELAYED_REPAIR_AFTER_RESTART))
    assert agg_drar.v16_beats_v15_rate >= 0.5
    assert agg_drar.tsc_v16_beats_tsc_v15_rate >= 0.5


def test_tcc_v6_restart_and_delayed_repair_arbiters():
    from coordpy.multi_agent_substrate_coordinator_v2 import (
        W66_MASC_V2_REGIME_BASELINE,
    )
    from coordpy.multi_agent_substrate_coordinator_v7 import (
        W71_MASC_V7_REGIME_DELAYED_REPAIR_AFTER_RESTART,
    )
    from coordpy.team_consensus_controller_v6 import (
        TeamConsensusControllerV6,
        W71_TC_V6_DECISION_DELAYED_REPAIR,
        W71_TC_V6_DECISION_RESTART_AWARE,
    )
    tcc = TeamConsensusControllerV6()
    out_r = tcc.decide_v6(
        regime=W66_MASC_V2_REGIME_BASELINE,
        agent_guesses=[1.0, -1.0, 0.5, 0.2],
        agent_confidences=[0.8, 0.6, 0.7, 0.7],
        substrate_replay_trust=0.7,
        visible_token_budget_ratio=0.9,
        dominant_repair_label=1,
        agent_repair_labels=[1, 1, 0, 0],
        restart_pressure=0.8,
        agent_restart_recovery_flags=[1, 0, 1, 0])
    assert (
        out_r["decision"]
        == W71_TC_V6_DECISION_RESTART_AWARE)
    out_d = tcc.decide_v6(
        regime=(
            W71_MASC_V7_REGIME_DELAYED_REPAIR_AFTER_RESTART),
        agent_guesses=[1.0, -1.0, 0.5, 0.2],
        agent_confidences=[0.8, 0.6, 0.7, 0.7],
        substrate_replay_trust=0.7,
        visible_token_budget_ratio=0.3,
        branch_assignments=[0, 1, 2, 0],
        delayed_repair_trajectory_cid="drt_abc",
        delay_turns=3,
        agent_delay_absorption_scores=[0.9, 0.5, 0.4, 0.3])
    assert (
        out_d["decision"]
        == W71_TC_V6_DECISION_DELAYED_REPAIR)


def test_substrate_adapter_v16_has_v16_full_tier():
    from coordpy.substrate_adapter_v16 import (
        W71_SUBSTRATE_TIER_SUBSTRATE_V16_FULL,
        probe_all_v16_adapters,
        probe_tiny_substrate_v16_adapter,
        probe_synthetic_v16_adapter,
    )
    cap = probe_tiny_substrate_v16_adapter()
    assert (
        str(cap.tier)
        == str(W71_SUBSTRATE_TIER_SUBSTRATE_V16_FULL))
    syn = probe_synthetic_v16_adapter()
    assert str(syn.tier) != str(
        W71_SUBSTRATE_TIER_SUBSTRATE_V16_FULL)
    matrix = probe_all_v16_adapters()
    assert matrix.has_v16_full()


def test_w71_no_version_bump():
    from coordpy import SDK_VERSION, __version__
    assert str(__version__) == "0.5.20"
    assert str(SDK_VERSION) == "coordpy.sdk.v3.43"
