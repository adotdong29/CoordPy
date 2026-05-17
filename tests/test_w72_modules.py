"""W72 focused module tests.

Per-module checks for the W72 mechanism bundle: V17 substrate axes,
KV V17 13-target ridge, cache V15 + rejoin-pressure head, replay
V13 + rejoin-aware routing head, persistent V24 / LHR V24 / MLSC
V20 / consensus V18, hosted V5 modules, handoff V4, MASC V8 + new
regime, TCC V7. Mirrors the W71 ``test_w71_modules.py`` style.
"""

from __future__ import annotations


def test_tiny_substrate_v17_determinism_and_axes():
    from coordpy.tiny_substrate_v17 import (
        build_default_tiny_substrate_v17,
        emit_tiny_substrate_v17_forward_witness,
        forward_tiny_substrate_v17,
        record_branch_pressure_window_v17,
        record_rejoin_event_v17,
        tokenize_bytes_v17,
        W72_DEFAULT_V17_N_LAYERS,
        W72_REPAIR_DELAYED_REJOIN_AFTER_RESTART,
    )
    p = build_default_tiny_substrate_v17(seed=72000)
    ids = tokenize_bytes_v17("w72_modules_test", max_len=8)
    t1, c1 = forward_tiny_substrate_v17(p, ids)
    t2, c2 = forward_tiny_substrate_v17(p, ids)
    assert t1.cid() == t2.cid()
    assert (
        t1.delayed_rejoin_after_restart_per_layer.shape
        == (int(W72_DEFAULT_V17_N_LAYERS),))
    assert (
        t1.rejoin_pressure_gate_per_layer.shape
        == (int(W72_DEFAULT_V17_N_LAYERS),))
    record_rejoin_event_v17(
        c1, turn=3, rejoin_kind="branch_rejoin",
        role="planner")
    record_branch_pressure_window_v17(
        c1, restart_turn=1, rejoin_turn=5,
        rejoin_lag_turns=4, role="planner")
    # The V17 cache also needs a V16 restart event to combine with
    # the rejoin event to fire the new label.
    from coordpy.tiny_substrate_v16 import (
        record_restart_event_v16,
    )
    record_restart_event_v16(
        c1.v16_cache, turn=1, restart_kind="agent_restart",
        role="planner")
    t3, c3 = forward_tiny_substrate_v17(
        p, ids, v17_kv_cache=c1, rejoin_pressure=0.5,
        restart_pressure=0.6)
    assert bool(t3.restart_repair_trajectory_cid)
    rl = c3.delayed_rejoin_after_restart_per_layer
    assert int(rl.shape[0]) == int(W72_DEFAULT_V17_N_LAYERS)
    import numpy as _np
    assert int(_np.count_nonzero(
        rl == W72_REPAIR_DELAYED_REJOIN_AFTER_RESTART)) > 0
    w = emit_tiny_substrate_v17_forward_witness(t3, c3)
    assert int(w.n_layers) == int(W72_DEFAULT_V17_N_LAYERS)


def test_kv_bridge_v17_thirteen_target_ridge_converges():
    from coordpy.kv_bridge_v17 import (
        KVBridgeV17Projection, fit_kv_bridge_v17_thirteen_target,
    )
    from coordpy.kv_bridge_v16 import KVBridgeV16Projection
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
    from coordpy.tiny_substrate_v17 import (
        build_default_tiny_substrate_v17,
    )
    import numpy as _np
    sub = build_default_tiny_substrate_v17(seed=72000)
    cfg = sub.config.v16.v15.v14.v13.v12.v11.v10.v9
    d_head = cfg.d_model // cfg.n_heads
    kv_b3 = KVBridgeV3Projection.init(
        n_layers=cfg.n_layers, n_heads=cfg.n_heads,
        n_kv_heads=cfg.n_kv_heads, n_inject_tokens=3,
        carrier_dim=6, d_head=int(d_head), seed=72008)
    kv_b16 = KVBridgeV16Projection.init_from_v15(
        KVBridgeV15Projection.init_from_v14(
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
                                                        kv_b3, seed_v4=72009),
                                                    seed_v5=72010),
                                                seed_v6=72011),
                                            seed_v7=72012),
                                        seed_v8=72013),
                                    seed_v9=72014),
                                seed_v10=72015),
                            seed_v11=72016),
                        seed_v12=72017),
                    seed_v13=72018),
                seed_v14=72019),
            seed_v15=72020),
        seed_v16=72021)
    proj = KVBridgeV17Projection.init_from_v16(
        kv_b16, seed_v17=72022)
    rng = _np.random.default_rng(72300)
    carriers = [
        list(rng.standard_normal(8)) for _ in range(4)]
    targets = []
    for i in range(13):
        t = _np.zeros(int(cfg.vocab_size))
        t[20 + 15 * i] = 1.0 / (i + 1)
        targets.append(t.tolist())
    _, rep = fit_kv_bridge_v17_thirteen_target(
        params=sub, projection=proj,
        train_carriers=carriers,
        target_delta_logits_stack=targets,
        follow_up_token_ids=[1, 2, 3, 4],
        n_directions=3,
        delayed_rejoin_target_index=12)
    assert int(rep.n_targets) == 13


def test_kv_bridge_v17_rejoin_pressure_falsifier_honest():
    from coordpy.kv_bridge_v17 import (
        probe_kv_bridge_v17_rejoin_pressure_falsifier,
    )
    honest = probe_kv_bridge_v17_rejoin_pressure_falsifier(
        rejoin_pressure_flag=1)
    assert float(honest.falsifier_score) == 0.0


def test_cache_v15_twelve_objective_and_rejoin_head_converge():
    from coordpy.cache_controller_v15 import (
        CacheControllerV15, fit_twelve_objective_ridge_v15,
        fit_per_role_rejoin_pressure_head_v15,
    )
    import numpy as _np
    cc = CacheControllerV15.init(fit_seed=72100)
    rng = _np.random.default_rng(72310)
    X = rng.standard_normal((10, 4))
    cc, rep = fit_twelve_objective_ridge_v15(
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
        target_restart_dominance=(X[:, 3] * 0.2).tolist(),
        target_delayed_rejoin_after_restart=(
            X[:, 1] * 0.3 + X[:, 2] * 0.2).tolist())
    assert rep.converged
    assert int(rep.n_objectives) == 12
    X13 = rng.standard_normal((10, 13))
    cc, rep2 = fit_per_role_rejoin_pressure_head_v15(
        controller=cc, role="planner",
        train_features=X13.tolist(),
        target_rejoin_priorities=(
            X13[:, 0] * 0.3 + X13[:, 12] * 0.4).tolist())
    assert rep2.converged


def test_replay_v13_regimes_and_routing_head():
    from coordpy.replay_controller_v13 import (
        ReplayControllerV13,
        W72_REPLAY_REGIMES_V13,
        W72_REJOIN_AWARE_ROUTING_LABELS,
        fit_replay_controller_v13_per_role,
        fit_replay_v13_rejoin_aware_routing_head,
    )
    from coordpy.replay_controller import ReplayCandidate
    import numpy as _np
    assert int(len(W72_REPLAY_REGIMES_V13)) == 20
    assert (
        "delayed_rejoin_after_restart_under_budget_regime"
        in W72_REPLAY_REGIMES_V13)
    assert int(len(W72_REJOIN_AWARE_ROUTING_LABELS)) == 10
    rc = ReplayControllerV13.init()
    cands = {
        r: [ReplayCandidate(
            100, 1000, 50, 0.1, 0.0, 0.3, True, True, 0)]
        for r in W72_REPLAY_REGIMES_V13}
    decs = {
        r: ["choose_reuse"]
        for r in W72_REPLAY_REGIMES_V13}
    rc, _ = fit_replay_controller_v13_per_role(
        controller=rc, role="planner",
        train_candidates_per_regime=cands,
        train_decisions_per_regime=decs)
    rng = _np.random.default_rng(72400)
    Xt = rng.standard_normal((50, 10))
    labs = [
        W72_REJOIN_AWARE_ROUTING_LABELS[
            i % len(W72_REJOIN_AWARE_ROUTING_LABELS)]
        for i in range(50)]
    rc, _ = fit_replay_v13_rejoin_aware_routing_head(
        controller=rc, train_team_features=Xt.tolist(),
        train_routing_labels=labs)
    assert rc.rejoin_aware_routing_head is not None
    lab, score = rc.decide_rejoin_aware_routing(
        team_features=Xt[0].tolist())
    assert lab in W72_REJOIN_AWARE_ROUTING_LABELS


def test_persistent_v24_step_and_lhr_v24_fit():
    from coordpy.persistent_latent_v12 import V12StackedCell
    from coordpy.persistent_latent_v24 import (
        W72_DEFAULT_V24_DISTRACTOR_RANK,
        W72_DEFAULT_V24_MAX_CHAIN_WALK_DEPTH,
        W72_DEFAULT_V24_N_LAYERS,
        step_persistent_state_v24,
    )
    from coordpy.long_horizon_retention_v24 import (
        LongHorizonReconstructionV24Head,
        fit_lhr_v24_fourteen_layer_scorer,
        emit_lhr_v24_witness,
    )
    import numpy as _np
    assert int(W72_DEFAULT_V24_N_LAYERS) == 23
    assert int(W72_DEFAULT_V24_DISTRACTOR_RANK) == 23
    assert int(W72_DEFAULT_V24_MAX_CHAIN_WALK_DEPTH) >= 524288
    cell = V12StackedCell.init(state_dim=4, seed=72030)
    s1 = step_persistent_state_v24(
        cell=cell, prev_state=None,
        carrier_values=[0.1] * 4, turn_index=0,
        role="planner",
        rejoin_pressure_skip_v24=[0.5] * 4,
        rejoin_pressure_ema_alpha=0.10)
    rjc = s1.rejoin_pressure_carrier
    assert all(0.0 < float(x) < 0.5 for x in rjc)
    s2 = step_persistent_state_v24(
        cell=cell, prev_state=s1,
        carrier_values=[0.2] * 4, turn_index=1,
        role="planner",
        rejoin_pressure_skip_v24=[1.0] * 4)
    assert s2.rejoin_pressure_carrier != s1.rejoin_pressure_carrier
    head = LongHorizonReconstructionV24Head.init(seed=72200)
    X = _np.random.default_rng(72210).standard_normal((10, 42))
    head2, audit = fit_lhr_v24_fourteen_layer_scorer(
        head=head, train_features=X.tolist(),
        train_targets=(X[:, 0] + X[:, 5]).tolist())
    assert audit["converged"]
    w = emit_lhr_v24_witness(
        head2, carrier=[0.1] * 6, k=16,
        partial_contradiction_indicator=[0.5] * 8,
        multi_branch_rejoin_indicator=[0.5] * 8,
        repair_dominance_indicator=[0.5] * 7,
        restart_indicator=[0.5] * 8,
        rejoin_indicator=[0.5] * 8)
    assert int(w.n_heads) == 23


def test_mlsc_v20_chains_merge_union():
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
        wrap_v18_as_v19,
    )
    from coordpy.mergeable_latent_capsule_v20 import (
        MergeOperatorV20, wrap_v19_as_v20,
    )
    v3 = make_root_capsule_v3(
        branch_id="test", payload=(0.1,) * 6,
        fact_tags=("w72",), confidence=0.9, trust=0.9,
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
    v19 = wrap_v18_as_v19(v18)
    v20_a = wrap_v19_as_v20(
        v19,
        restart_repair_trajectory_chain=("rrt_a",),
        rejoin_pressure_chain=("rj_a",))
    v20_b = wrap_v19_as_v20(
        v19,
        restart_repair_trajectory_chain=("rrt_b",),
        rejoin_pressure_chain=("rj_b",))
    merged = MergeOperatorV20(factor_dim=6).merge(
        [v20_a, v20_b])
    assert (
        "rrt_a" in merged.restart_repair_trajectory_chain
        and "rrt_b" in merged.restart_repair_trajectory_chain)
    assert (
        "rj_a" in merged.rejoin_pressure_chain
        and "rj_b" in merged.rejoin_pressure_chain)


def test_consensus_v18_30_stages_and_arbiters():
    from coordpy.consensus_fallback_controller_v18 import (
        ConsensusFallbackControllerV18,
        W72_CONSENSUS_V18_STAGES,
        emit_consensus_v18_witness,
    )
    assert len(W72_CONSENSUS_V18_STAGES) >= 30
    c = ConsensusFallbackControllerV18.init()
    out_rj = c.decide_v18(
        payloads=[[1.0], [2.0]], trusts=[0.6, 0.7],
        replay_decisions=["u", "u"],
        rejoin_pressure_scores_per_parent=[0.4, 0.9])
    assert (
        out_rj["stage"]
        == "rejoin_pressure_arbiter")
    out_drj = c.decide_v18(
        payloads=[[1.0], [2.0]], trusts=[0.6, 0.7],
        replay_decisions=["u", "u"],
        delayed_rejoin_scores_per_parent=[0.4, 0.9])
    assert (
        out_drj["stage"]
        == "delayed_rejoin_after_restart_arbiter")
    w = emit_consensus_v18_witness(c)
    assert int(w.rejoin_pressure_stage_fired) >= 1
    assert int(w.delayed_rejoin_stage_fired) >= 1


def test_hosted_router_v5_rejoin_pressure_score():
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
        HostedRoutingRequestV4,
    )
    from coordpy.hosted_router_controller_v5 import (
        HostedRouterControllerV5, HostedRoutingRequestV5,
    )
    reg = default_hosted_registry()
    rc = HostedRouterControllerV5.init(reg, {
        "openrouter_paid": 0.85, "openai_paid": 0.92})
    req = HostedRoutingRequestV5(
        inner_v4=HostedRoutingRequestV4(
            inner_v3=HostedRoutingRequestV3(
                inner_v2=HostedRoutingRequestV2(
                    inner_v1=HostedRoutingRequest(
                        request_cid="t_router_v5",
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
            restart_pressure=0.7,
            weight_restart_pressure=0.6,
            weight_delayed_repair_match=0.4),
        rejoin_pressure=0.8,
        weight_rejoin_pressure=0.6,
        weight_delayed_rejoin_match=0.4)
    d = rc.decide_v5(req)
    assert d.rejoin_pressure_score > 0.0
    d2 = rc.decide_v5(req)
    assert d.cid() == d2.cid()


def test_hosted_logprob_v5_rejoin_aware_abstain_floor():
    from coordpy.hosted_logprob_router import (
        TopKLogprobsPayload,
    )
    from coordpy.hosted_logprob_router_v5 import (
        HostedLogprobRouterV5,
    )
    lr = HostedLogprobRouterV5(
        rejoin_pressure_floor=0.5,
        rejoin_abstain_floor_delta=1.5)
    payloads = [
        TopKLogprobsPayload("p1", tuple(
            (chr(ord("a") + i), -1.0 - i * 0.1)
            for i in range(8))),
        TopKLogprobsPayload("p2", tuple(
            (chr(ord("a") + i), -1.0 - i * 0.05)
            for i in range(8))),
    ]
    no = lr.fuse_v5(
        payloads, visible_token_budget=256,
        baseline_token_cost=512,
        restart_pressure=0.0, rejoin_pressure=0.0)
    high = lr.fuse_v5(
        payloads, visible_token_budget=256,
        baseline_token_cost=512,
        restart_pressure=0.0, rejoin_pressure=0.9)
    assert (
        high["effective_abstain_entropy_floor"]
        < no["effective_abstain_entropy_floor"])


def test_hosted_cache_v5_three_layer_savings_above_80_pct():
    from coordpy.hosted_cache_aware_planner_v5 import (
        hosted_cache_aware_savings_v5_vs_recompute,
    )
    sav = hosted_cache_aware_savings_v5_vs_recompute(
        n_roles=12, n_turns=8, hosted_cache_hit_rate=1.0)
    assert float(sav["saving_ratio"]) >= 0.80


def test_hosted_cost_v5_abstain_under_rejoin():
    from coordpy.hosted_cost_planner import HostedCostPlanSpec
    from coordpy.hosted_cost_planner_v2 import (
        HostedCostPlanSpecV2,
    )
    from coordpy.hosted_cost_planner_v3 import (
        HostedCostPlanSpecV3,
    )
    from coordpy.hosted_cost_planner_v4 import (
        HostedCostPlanSpecV4,
    )
    from coordpy.hosted_cost_planner_v5 import (
        HostedCostPlanSpecV5, plan_hosted_cost_v5,
    )
    from coordpy.hosted_router_controller import (
        default_hosted_registry,
    )
    reg = default_hosted_registry()
    qs = {p.name: 0.8 for p in reg.providers}
    spec = HostedCostPlanSpecV5(
        inner_v4=HostedCostPlanSpecV4(
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
            restart_pressure=0.3,
            restart_pressure_cap=0.7,
            abstain_when_restart_violated=True),
        rejoin_pressure=0.9, rejoin_pressure_cap=0.7,
        abstain_when_rejoin_violated=True)
    rep = plan_hosted_cost_v5(
        registry=reg, provider_quality_scores=qs,
        spec_v5=spec)
    assert rep.rejoin_violated
    assert "abstain_v5" in str(rep.rationale_v5)


def test_hosted_boundary_v5_28_blocked_axes_and_falsifier():
    from coordpy.hosted_real_substrate_boundary_v5 import (
        build_default_hosted_real_substrate_boundary_v5,
        probe_hosted_real_substrate_boundary_v5_falsifier,
    )
    b = build_default_hosted_real_substrate_boundary_v5()
    assert int(len(b.blocked_axes)) >= 28
    f_h = probe_hosted_real_substrate_boundary_v5_falsifier(
        boundary=b,
        claimed_axis="restart_repair_trajectory_cid",
        claim_satisfied_at_hosted=False)
    assert float(f_h.falsifier_score) == 0.0
    f_d = probe_hosted_real_substrate_boundary_v5_falsifier(
        boundary=b,
        claimed_axis="restart_repair_trajectory_cid",
        claim_satisfied_at_hosted=True)
    assert float(f_d.falsifier_score) == 1.0


def test_handoff_v4_rejoin_promotion_and_drrj_fallback():
    from coordpy.hosted_real_handoff_coordinator import (
        HandoffRequest,
    )
    from coordpy.hosted_real_handoff_coordinator import (
        W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY,
    )
    from coordpy.hosted_real_handoff_coordinator_v2 import (
        HandoffRequestV2,
    )
    from coordpy.hosted_real_handoff_coordinator_v3 import (
        HandoffRequestV3,
    )
    from coordpy.hosted_real_handoff_coordinator_v4 import (
        HandoffRequestV4, HostedRealHandoffCoordinatorV4,
        W72_HANDOFF_DECISION_DELAYED_REJOIN_AFTER_RESTART_FALLBACK,
    )
    c = HostedRealHandoffCoordinatorV4()
    e_rj = c.decide_v4(
        req_v4=HandoffRequestV4(
            inner_v3=HandoffRequestV3(
                inner_v2=HandoffRequestV2(
                    inner_v1=HandoffRequest(
                        request_cid="t_rj", needs_text_only=True,
                        needs_substrate_state_access=False),
                    visible_token_budget=256,
                    baseline_token_cost=512,
                    dominant_repair_label=0),
                restart_pressure=0.0,
                delayed_repair_trajectory_cid="",
                delay_turns=0,
                expected_substrate_trust=0.7),
            rejoin_pressure=0.8,
            restart_repair_trajectory_cid="",
            rejoin_lag_turns=0,
            expected_substrate_trust_v4=0.7))
    assert (
        str(e_rj.decision_v4)
        == W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY)
    assert e_rj.rejoin_alignment == 1.0
    e_drrj = c.decide_v4(
        req_v4=HandoffRequestV4(
            inner_v3=HandoffRequestV3(
                inner_v2=HandoffRequestV2(
                    inner_v1=HandoffRequest(
                        request_cid="t_drrj",
                        needs_text_only=True,
                        needs_substrate_state_access=False),
                    visible_token_budget=256,
                    baseline_token_cost=512,
                    dominant_repair_label=0),
                restart_pressure=0.0,
                delayed_repair_trajectory_cid="",
                delay_turns=0,
                expected_substrate_trust=0.7),
            rejoin_pressure=0.0,
            restart_repair_trajectory_cid="rrt_xyz",
            rejoin_lag_turns=4,
            expected_substrate_trust_v4=0.7))
    assert (
        str(e_drrj.decision_v4)
        == W72_HANDOFF_DECISION_DELAYED_REJOIN_AFTER_RESTART_FALLBACK)
    assert e_drrj.delayed_rejoin_alignment == 1.0


def test_masc_v8_v17_beats_v16_on_baseline_and_new_regime():
    from coordpy.multi_agent_substrate_coordinator_v2 import (
        W66_MASC_V2_REGIME_BASELINE,
    )
    from coordpy.multi_agent_substrate_coordinator_v8 import (
        MultiAgentSubstrateCoordinatorV8,
        W72_MASC_V8_REGIME_DELAYED_REJOIN_AFTER_RESTART_UNDER_BUDGET,
    )
    m = MultiAgentSubstrateCoordinatorV8()
    _, agg_b = m.run_batch(
        seeds=list(range(15)),
        regime=W66_MASC_V2_REGIME_BASELINE)
    assert agg_b.v17_beats_v16_rate >= 0.5
    assert agg_b.tsc_v17_beats_tsc_v16_rate >= 0.5
    _, agg_drrjub = m.run_batch(
        seeds=list(range(15)),
        regime=(
            W72_MASC_V8_REGIME_DELAYED_REJOIN_AFTER_RESTART_UNDER_BUDGET))
    assert agg_drrjub.v17_beats_v16_rate >= 0.5
    assert agg_drrjub.tsc_v17_beats_tsc_v16_rate >= 0.5


def test_tcc_v7_rejoin_and_delayed_rejoin_arbiters():
    from coordpy.multi_agent_substrate_coordinator_v2 import (
        W66_MASC_V2_REGIME_BASELINE,
    )
    from coordpy.multi_agent_substrate_coordinator_v8 import (
        W72_MASC_V8_REGIME_DELAYED_REJOIN_AFTER_RESTART_UNDER_BUDGET,
    )
    from coordpy.team_consensus_controller_v7 import (
        TeamConsensusControllerV7,
        W72_TC_V7_DECISION_DELAYED_REJOIN,
        W72_TC_V7_DECISION_REJOIN_PRESSURE,
    )
    tcc = TeamConsensusControllerV7()
    out_r = tcc.decide_v7(
        regime=W66_MASC_V2_REGIME_BASELINE,
        agent_guesses=[1.0, -1.0, 0.5, 0.2],
        agent_confidences=[0.8, 0.6, 0.7, 0.7],
        substrate_replay_trust=0.7,
        visible_token_budget_ratio=0.9,
        dominant_repair_label=1,
        agent_repair_labels=[1, 1, 0, 0],
        rejoin_pressure=0.8,
        agent_rejoin_recovery_flags=[1, 0, 1, 0])
    assert (
        out_r["decision"]
        == W72_TC_V7_DECISION_REJOIN_PRESSURE)
    out_d = tcc.decide_v7(
        regime=(
            W72_MASC_V8_REGIME_DELAYED_REJOIN_AFTER_RESTART_UNDER_BUDGET),
        agent_guesses=[1.0, -1.0, 0.5, 0.2],
        agent_confidences=[0.8, 0.6, 0.7, 0.7],
        substrate_replay_trust=0.7,
        visible_token_budget_ratio=0.3,
        branch_assignments=[0, 1, 2, 0],
        restart_repair_trajectory_cid="rrt_abc",
        rejoin_lag_turns=4,
        agent_rejoin_absorption_scores=[0.9, 0.5, 0.4, 0.3])
    assert (
        out_d["decision"]
        == W72_TC_V7_DECISION_DELAYED_REJOIN)


def test_substrate_adapter_v17_has_v17_full_tier():
    from coordpy.substrate_adapter_v17 import (
        W72_SUBSTRATE_TIER_SUBSTRATE_V17_FULL,
        probe_all_v17_adapters,
        probe_tiny_substrate_v17_adapter,
        probe_synthetic_v17_adapter,
    )
    cap = probe_tiny_substrate_v17_adapter()
    assert (
        str(cap.tier)
        == str(W72_SUBSTRATE_TIER_SUBSTRATE_V17_FULL))
    syn = probe_synthetic_v17_adapter()
    assert str(syn.tier) != str(
        W72_SUBSTRATE_TIER_SUBSTRATE_V17_FULL)
    matrix = probe_all_v17_adapters()
    assert matrix.has_v17_full()


def test_w72_no_version_bump():
    from coordpy import SDK_VERSION, __version__
    assert str(__version__) == "0.5.20"
    assert str(SDK_VERSION) == "coordpy.sdk.v3.43"
