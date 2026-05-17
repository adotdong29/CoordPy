"""W73 focused module tests.

Per-module checks for the W73 mechanism bundle: V18 substrate
axes, KV V18 14-target ridge, cache V16 + replacement-pressure
head, replay V14 + replacement-aware routing head, persistent V25
/ LHR V25 / MLSC V21 / consensus V19, hosted V6 modules, handoff
V5, MASC V9 + new regime, TCC V8. Mirrors the W72
``test_w72_modules.py`` style.
"""

from __future__ import annotations


def test_tiny_substrate_v18_determinism_and_axes():
    from coordpy.tiny_substrate_v18 import (
        build_default_tiny_substrate_v18,
        emit_tiny_substrate_v18_forward_witness,
        forward_tiny_substrate_v18,
        record_contradiction_event_v18,
        record_replacement_event_v18,
        record_replacement_window_v18,
        tokenize_bytes_v18,
        W73_DEFAULT_V18_N_LAYERS,
        W73_REPAIR_REPLACEMENT_AFTER_CTR,
    )
    p = build_default_tiny_substrate_v18(seed=73000)
    ids = tokenize_bytes_v18("w73_modules_test", max_len=8)
    t1, c1 = forward_tiny_substrate_v18(p, ids)
    t2, _ = forward_tiny_substrate_v18(p, ids)
    assert t1.cid() == t2.cid()
    assert (
        t1.replacement_after_ctr_per_layer.shape
        == (int(W73_DEFAULT_V18_N_LAYERS),))
    assert (
        t1.replacement_pressure_gate_per_layer.shape
        == (int(W73_DEFAULT_V18_N_LAYERS),))
    record_contradiction_event_v18(
        c1, turn=2, role="planner")
    record_replacement_event_v18(
        c1, turn=3, role="planner")
    record_replacement_window_v18(
        c1, contradiction_turn=2, replacement_turn=3,
        rejoin_turn=8, replacement_lag_turns=5, role="planner")
    # The V18 cache also needs a rejoin event on the inner V17
    # cache to fire the new label.
    from coordpy.tiny_substrate_v17 import (
        record_rejoin_event_v17,
    )
    record_rejoin_event_v17(
        c1.v17_cache, turn=8, rejoin_kind="branch_rejoin",
        role="planner")
    t3, c3 = forward_tiny_substrate_v18(
        p, ids, v18_kv_cache=c1,
        replacement_pressure=0.5,
        contradiction_pressure=0.5)
    assert bool(t3.replacement_repair_trajectory_cid)
    rl = c3.replacement_after_ctr_per_layer
    assert int(rl.shape[0]) == int(W73_DEFAULT_V18_N_LAYERS)
    import numpy as _np
    assert int(_np.count_nonzero(
        rl == W73_REPAIR_REPLACEMENT_AFTER_CTR)) > 0
    w = emit_tiny_substrate_v18_forward_witness(t3, c3)
    assert int(w.n_layers) == int(W73_DEFAULT_V18_N_LAYERS)


def test_kv_bridge_v18_fourteen_target_ridge_converges():
    from coordpy.kv_bridge_v18 import (
        KVBridgeV18Projection,
        fit_kv_bridge_v18_fourteen_target,
    )
    from coordpy.kv_bridge_v17 import KVBridgeV17Projection
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
    from coordpy.tiny_substrate_v18 import (
        build_default_tiny_substrate_v18,
    )
    import numpy as _np
    sub = build_default_tiny_substrate_v18(seed=73000)
    cfg = sub.config.v17.v16.v15.v14.v13.v12.v11.v10.v9
    d_head = cfg.d_model // cfg.n_heads
    kv_b3 = KVBridgeV3Projection.init(
        n_layers=cfg.n_layers, n_heads=cfg.n_heads,
        n_kv_heads=cfg.n_kv_heads, n_inject_tokens=3,
        carrier_dim=6, d_head=int(d_head), seed=73008)
    kv_b17 = KVBridgeV17Projection.init_from_v16(
        KVBridgeV16Projection.init_from_v15(
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
                                                            kv_b3, seed_v4=73009),
                                                        seed_v5=73010),
                                                    seed_v6=73011),
                                                seed_v7=73012),
                                            seed_v8=73013),
                                        seed_v9=73014),
                                    seed_v10=73015),
                                seed_v11=73016),
                            seed_v12=73017),
                        seed_v13=73018),
                    seed_v14=73019),
                seed_v15=73020),
            seed_v16=73021),
        seed_v17=73022)
    proj = KVBridgeV18Projection.init_from_v17(
        kv_b17, seed_v18=73023)
    rng = _np.random.default_rng(73300)
    carriers = [
        list(rng.standard_normal(8)) for _ in range(4)]
    targets = []
    for i in range(14):
        t = _np.zeros(int(cfg.vocab_size))
        t[20 + 15 * i] = 1.0 / (i + 1)
        targets.append(t.tolist())
    _, rep = fit_kv_bridge_v18_fourteen_target(
        params=sub, projection=proj,
        train_carriers=carriers,
        target_delta_logits_stack=targets,
        follow_up_token_ids=[1, 2, 3, 4],
        n_directions=3,
        replacement_target_index=13)
    assert int(rep.n_targets) == 14


def test_kv_bridge_v18_replacement_pressure_falsifier_honest():
    from coordpy.kv_bridge_v18 import (
        probe_kv_bridge_v18_replacement_pressure_falsifier,
    )
    honest = probe_kv_bridge_v18_replacement_pressure_falsifier(
        replacement_pressure_flag=1)
    assert float(honest.falsifier_score) == 0.0


def test_cache_v16_thirteen_objective_and_rep_head_converge():
    from coordpy.cache_controller_v16 import (
        CacheControllerV16,
        fit_thirteen_objective_ridge_v16,
        fit_per_role_replacement_pressure_head_v16,
    )
    import numpy as _np
    cc = CacheControllerV16.init(fit_seed=73100)
    rng = _np.random.default_rng(73310)
    X = rng.standard_normal((10, 4))
    cc, rep = fit_thirteen_objective_ridge_v16(
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
            X[:, 1] * 0.3 + X[:, 2] * 0.2).tolist(),
        target_replacement_after_ctr=(
            X[:, 0] * 0.3 + X[:, 1] * 0.2
            + X[:, 2] * 0.3).tolist())
    assert rep.converged
    assert int(rep.n_objectives) == 13
    X14 = rng.standard_normal((10, 14))
    cc, rep2 = fit_per_role_replacement_pressure_head_v16(
        controller=cc, role="planner",
        train_features=X14.tolist(),
        target_replacement_priorities=(
            X14[:, 0] * 0.3 + X14[:, 13] * 0.4).tolist())
    assert rep2.converged


def test_replay_v14_regimes_and_routing_head():
    from coordpy.replay_controller_v14 import (
        ReplayControllerV14,
        W73_REPLAY_REGIMES_V14,
        W73_REPLACEMENT_AWARE_ROUTING_LABELS,
        fit_replay_controller_v14_per_role,
        fit_replay_v14_replacement_aware_routing_head,
    )
    from coordpy.replay_controller import ReplayCandidate
    import numpy as _np
    assert int(len(W73_REPLAY_REGIMES_V14)) == 21
    assert (
        "replacement_after_contradiction_then_rejoin_regime"
        in W73_REPLAY_REGIMES_V14)
    assert int(len(W73_REPLACEMENT_AWARE_ROUTING_LABELS)) == 11
    rc = ReplayControllerV14.init()
    cands = {
        r: [ReplayCandidate(
            100, 1000, 50, 0.1, 0.0, 0.3, True, True, 0)]
        for r in W73_REPLAY_REGIMES_V14}
    decs = {
        r: ["choose_reuse"]
        for r in W73_REPLAY_REGIMES_V14}
    rc, _ = fit_replay_controller_v14_per_role(
        controller=rc, role="planner",
        train_candidates_per_regime=cands,
        train_decisions_per_regime=decs)
    rng = _np.random.default_rng(73400)
    Xt = rng.standard_normal((50, 10))
    labs = [
        W73_REPLACEMENT_AWARE_ROUTING_LABELS[
            i % len(W73_REPLACEMENT_AWARE_ROUTING_LABELS)]
        for i in range(50)]
    rc, _ = fit_replay_v14_replacement_aware_routing_head(
        controller=rc, train_team_features=Xt.tolist(),
        train_routing_labels=labs)
    assert rc.replacement_aware_routing_head is not None
    lab, score = rc.decide_replacement_aware_routing(
        team_features=Xt[0].tolist())
    assert lab in W73_REPLACEMENT_AWARE_ROUTING_LABELS


def test_persistent_v25_step_and_lhr_v25_fit():
    from coordpy.persistent_latent_v12 import V12StackedCell
    from coordpy.persistent_latent_v25 import (
        W73_DEFAULT_V25_DISTRACTOR_RANK,
        W73_DEFAULT_V25_MAX_CHAIN_WALK_DEPTH,
        W73_DEFAULT_V25_N_LAYERS,
        step_persistent_state_v25,
    )
    from coordpy.long_horizon_retention_v25 import (
        LongHorizonReconstructionV25Head,
        fit_lhr_v25_fifteen_layer_scorer,
        emit_lhr_v25_witness,
    )
    import numpy as _np
    assert int(W73_DEFAULT_V25_N_LAYERS) == 24
    assert int(W73_DEFAULT_V25_DISTRACTOR_RANK) == 24
    assert int(W73_DEFAULT_V25_MAX_CHAIN_WALK_DEPTH) >= 1048576
    cell = V12StackedCell.init(state_dim=4, seed=73030)
    s1 = step_persistent_state_v25(
        cell=cell, prev_state=None,
        carrier_values=[0.1] * 4, turn_index=0,
        role="planner",
        replacement_pressure_skip_v25=[0.5] * 4,
        replacement_pressure_ema_alpha=0.10)
    rpc = s1.replacement_pressure_carrier
    assert all(0.0 < float(x) < 0.5 for x in rpc)
    s2 = step_persistent_state_v25(
        cell=cell, prev_state=s1,
        carrier_values=[0.2] * 4, turn_index=1,
        role="planner",
        replacement_pressure_skip_v25=[1.0] * 4)
    assert (
        s2.replacement_pressure_carrier
        != s1.replacement_pressure_carrier)
    head = LongHorizonReconstructionV25Head.init(seed=73200)
    X = _np.random.default_rng(73210).standard_normal((10, 48))
    head2, audit = fit_lhr_v25_fifteen_layer_scorer(
        head=head, train_features=X.tolist(),
        train_targets=(X[:, 0] + X[:, 5]).tolist())
    assert audit["converged"]
    w = emit_lhr_v25_witness(
        head2, carrier=[0.1] * 6, k=16,
        partial_contradiction_indicator=[0.5] * 8,
        multi_branch_rejoin_indicator=[0.5] * 8,
        repair_dominance_indicator=[0.5] * 7,
        restart_indicator=[0.5] * 8,
        rejoin_indicator=[0.5] * 8,
        replacement_indicator=[0.5] * 8)
    assert int(w.n_heads) == 24


def test_mlsc_v21_chains_merge_union():
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
        wrap_v19_as_v20,
    )
    from coordpy.mergeable_latent_capsule_v21 import (
        MergeOperatorV21, wrap_v20_as_v21,
    )
    v3 = make_root_capsule_v3(
        branch_id="test", payload=(0.1,) * 6,
        fact_tags=("w73",), confidence=0.9, trust=0.9,
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
    v20 = wrap_v19_as_v20(v19)
    v21_a = wrap_v20_as_v21(
        v20,
        replacement_repair_trajectory_chain=("rep_a",),
        contradiction_chain=("ct_a",))
    v21_b = wrap_v20_as_v21(
        v20,
        replacement_repair_trajectory_chain=("rep_b",),
        contradiction_chain=("ct_b",))
    merged = MergeOperatorV21(factor_dim=6).merge(
        [v21_a, v21_b])
    assert (
        "rep_a" in merged.replacement_repair_trajectory_chain
        and "rep_b"
        in merged.replacement_repair_trajectory_chain)
    assert (
        "ct_a" in merged.contradiction_chain
        and "ct_b" in merged.contradiction_chain)


def test_consensus_v19_32_stages_and_arbiters():
    from coordpy.consensus_fallback_controller_v19 import (
        ConsensusFallbackControllerV19,
        W73_CONSENSUS_V19_STAGES,
        emit_consensus_v19_witness,
    )
    assert len(W73_CONSENSUS_V19_STAGES) >= 32
    c = ConsensusFallbackControllerV19.init()
    out_rp = c.decide_v19(
        payloads=[[1.0], [2.0]], trusts=[0.6, 0.7],
        replay_decisions=["u", "u"],
        replacement_pressure_scores_per_parent=[0.4, 0.9])
    assert (
        out_rp["stage"]
        == "replacement_pressure_arbiter")
    out_rep = c.decide_v19(
        payloads=[[1.0], [2.0]], trusts=[0.6, 0.7],
        replay_decisions=["u", "u"],
        replacement_after_ctr_scores_per_parent=[0.4, 0.9])
    assert (
        out_rep["stage"]
        == "replacement_after_contradiction_then_rejoin_arbiter")
    w = emit_consensus_v19_witness(c)
    assert int(w.replacement_pressure_stage_fired) >= 1
    assert int(w.replacement_after_ctr_stage_fired) >= 1


def test_hosted_router_v6_replacement_pressure_score():
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
        HostedRoutingRequestV5,
    )
    from coordpy.hosted_router_controller_v6 import (
        HostedRouterControllerV6, HostedRoutingRequestV6,
    )
    reg = default_hosted_registry()
    rc = HostedRouterControllerV6.init(reg, {
        "openrouter_paid": 0.85, "openai_paid": 0.92})
    req = HostedRoutingRequestV6(
        inner_v5=HostedRoutingRequestV5(
            inner_v4=HostedRoutingRequestV4(
                inner_v3=HostedRoutingRequestV3(
                    inner_v2=HostedRoutingRequestV2(
                        inner_v1=HostedRoutingRequest(
                            request_cid="t_router_v6",
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
            rejoin_pressure=0.7,
            weight_rejoin_pressure=0.6,
            weight_delayed_rejoin_match=0.4),
        replacement_pressure=0.8,
        weight_replacement_pressure=0.6,
        weight_replacement_after_ctr_match=0.4)
    d = rc.decide_v6(req)
    assert d.replacement_pressure_score > 0.0
    d2 = rc.decide_v6(req)
    assert d.cid() == d2.cid()


def test_hosted_logprob_v6_replacement_aware_abstain_floor():
    from coordpy.hosted_logprob_router import (
        TopKLogprobsPayload,
    )
    from coordpy.hosted_logprob_router_v6 import (
        HostedLogprobRouterV6,
    )
    lr = HostedLogprobRouterV6(
        replacement_pressure_floor=0.5,
        replacement_abstain_floor_delta=1.8)
    payloads = [
        TopKLogprobsPayload("p1", tuple(
            (chr(ord("a") + i), -1.0 - i * 0.1)
            for i in range(8))),
        TopKLogprobsPayload("p2", tuple(
            (chr(ord("a") + i), -1.0 - i * 0.05)
            for i in range(8))),
    ]
    no = lr.fuse_v6(
        payloads, visible_token_budget=256,
        baseline_token_cost=512,
        restart_pressure=0.0, rejoin_pressure=0.0,
        replacement_pressure=0.0)
    high = lr.fuse_v6(
        payloads, visible_token_budget=256,
        baseline_token_cost=512,
        restart_pressure=0.0, rejoin_pressure=0.0,
        replacement_pressure=0.9)
    assert (
        high["effective_abstain_entropy_floor"]
        < no["effective_abstain_entropy_floor"])


def test_hosted_cache_v6_four_layer_savings_above_85_pct():
    from coordpy.hosted_cache_aware_planner_v6 import (
        hosted_cache_aware_savings_v6_vs_recompute,
    )
    sav = hosted_cache_aware_savings_v6_vs_recompute(
        n_roles=14, n_turns=8, hosted_cache_hit_rate=1.0)
    assert float(sav["saving_ratio"]) >= 0.85


def test_hosted_cost_v6_abstain_under_replacement():
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
        HostedCostPlanSpecV5,
    )
    from coordpy.hosted_cost_planner_v6 import (
        HostedCostPlanSpecV6, plan_hosted_cost_v6,
    )
    from coordpy.hosted_router_controller import (
        default_hosted_registry,
    )
    reg = default_hosted_registry()
    qs = {p.name: 0.8 for p in reg.providers}
    spec = HostedCostPlanSpecV6(
        inner_v5=HostedCostPlanSpecV5(
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
            rejoin_pressure=0.3,
            rejoin_pressure_cap=0.7,
            abstain_when_rejoin_violated=True),
        replacement_pressure=0.9, replacement_pressure_cap=0.7,
        abstain_when_replacement_violated=True)
    rep = plan_hosted_cost_v6(
        registry=reg, provider_quality_scores=qs,
        spec_v6=spec)
    assert rep.replacement_violated
    assert "abstain_v6" in str(rep.rationale_v6)


def test_hosted_boundary_v6_31_blocked_axes_and_falsifier():
    from coordpy.hosted_real_substrate_boundary_v6 import (
        build_default_hosted_real_substrate_boundary_v6,
        probe_hosted_real_substrate_boundary_v6_falsifier,
    )
    b = build_default_hosted_real_substrate_boundary_v6()
    assert int(len(b.blocked_axes)) >= 31
    f_h = probe_hosted_real_substrate_boundary_v6_falsifier(
        boundary=b,
        claimed_axis="replacement_repair_trajectory_cid",
        claim_satisfied_at_hosted=False)
    assert float(f_h.falsifier_score) == 0.0
    f_d = probe_hosted_real_substrate_boundary_v6_falsifier(
        boundary=b,
        claimed_axis="replacement_repair_trajectory_cid",
        claim_satisfied_at_hosted=True)
    assert float(f_d.falsifier_score) == 1.0


def test_handoff_v5_replacement_promotion_and_rep_fallback():
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
        HandoffRequestV4,
    )
    from coordpy.hosted_real_handoff_coordinator_v5 import (
        HandoffRequestV5, HostedRealHandoffCoordinatorV5,
        W73_HANDOFF_DECISION_REPLACEMENT_AFTER_CTR_FALLBACK,
    )
    c = HostedRealHandoffCoordinatorV5()
    e_rep = c.decide_v5(
        req_v5=HandoffRequestV5(
            inner_v4=HandoffRequestV4(
                inner_v3=HandoffRequestV3(
                    inner_v2=HandoffRequestV2(
                        inner_v1=HandoffRequest(
                            request_cid="t_rep",
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
                restart_repair_trajectory_cid="",
                rejoin_lag_turns=0,
                expected_substrate_trust_v4=0.7),
            replacement_pressure=0.8,
            replacement_repair_trajectory_cid="",
            replacement_lag_turns=0,
            expected_substrate_trust_v5=0.7))
    assert (
        str(e_rep.decision_v5)
        == W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY)
    assert e_rep.replacement_alignment == 1.0
    e_repf = c.decide_v5(
        req_v5=HandoffRequestV5(
            inner_v4=HandoffRequestV4(
                inner_v3=HandoffRequestV3(
                    inner_v2=HandoffRequestV2(
                        inner_v1=HandoffRequest(
                            request_cid="t_repf",
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
                restart_repair_trajectory_cid="",
                rejoin_lag_turns=0,
                expected_substrate_trust_v4=0.7),
            replacement_pressure=0.0,
            replacement_repair_trajectory_cid="rep_xyz",
            replacement_lag_turns=5,
            expected_substrate_trust_v5=0.7))
    assert (
        str(e_repf.decision_v5)
        == W73_HANDOFF_DECISION_REPLACEMENT_AFTER_CTR_FALLBACK)
    assert e_repf.replacement_after_ctr_alignment == 1.0


def test_masc_v9_v18_beats_v17_on_baseline_and_new_regime():
    from coordpy.multi_agent_substrate_coordinator_v2 import (
        W66_MASC_V2_REGIME_BASELINE,
    )
    from coordpy.multi_agent_substrate_coordinator_v9 import (
        MultiAgentSubstrateCoordinatorV9,
        W73_MASC_V9_REGIME_REPLACEMENT_AFTER_CTR,
    )
    m = MultiAgentSubstrateCoordinatorV9()
    _, agg_b = m.run_batch(
        seeds=list(range(15)),
        regime=W66_MASC_V2_REGIME_BASELINE)
    assert agg_b.v18_beats_v17_rate >= 0.5
    assert agg_b.tsc_v18_beats_tsc_v17_rate >= 0.5
    _, agg_rep = m.run_batch(
        seeds=list(range(15)),
        regime=(
            W73_MASC_V9_REGIME_REPLACEMENT_AFTER_CTR))
    assert agg_rep.v18_beats_v17_rate >= 0.5
    assert agg_rep.tsc_v18_beats_tsc_v17_rate >= 0.5


def test_tcc_v8_replacement_and_replacement_after_ctr_arbiters():
    from coordpy.multi_agent_substrate_coordinator_v2 import (
        W66_MASC_V2_REGIME_BASELINE,
    )
    from coordpy.multi_agent_substrate_coordinator_v9 import (
        W73_MASC_V9_REGIME_REPLACEMENT_AFTER_CTR,
    )
    from coordpy.team_consensus_controller_v8 import (
        TeamConsensusControllerV8,
        W73_TC_V8_DECISION_REPLACEMENT_AFTER_CTR,
        W73_TC_V8_DECISION_REPLACEMENT_PRESSURE,
    )
    tcc = TeamConsensusControllerV8()
    out_r = tcc.decide_v8(
        regime=W66_MASC_V2_REGIME_BASELINE,
        agent_guesses=[1.0, -1.0, 0.5, 0.2],
        agent_confidences=[0.8, 0.6, 0.7, 0.7],
        substrate_replay_trust=0.7,
        visible_token_budget_ratio=0.9,
        dominant_repair_label=1,
        agent_repair_labels=[1, 1, 0, 0],
        replacement_pressure=0.8,
        agent_replacement_recovery_flags=[1, 0, 1, 0])
    assert (
        out_r["decision"]
        == W73_TC_V8_DECISION_REPLACEMENT_PRESSURE)
    out_d = tcc.decide_v8(
        regime=(
            W73_MASC_V9_REGIME_REPLACEMENT_AFTER_CTR),
        agent_guesses=[1.0, -1.0, 0.5, 0.2],
        agent_confidences=[0.8, 0.6, 0.7, 0.7],
        substrate_replay_trust=0.7,
        visible_token_budget_ratio=0.3,
        branch_assignments=[0, 1, 2, 0],
        replacement_repair_trajectory_cid="rep_abc",
        replacement_lag_turns=5,
        agent_replacement_absorption_scores=[
            0.9, 0.5, 0.4, 0.3])
    assert (
        out_d["decision"]
        == W73_TC_V8_DECISION_REPLACEMENT_AFTER_CTR)


def test_substrate_adapter_v18_has_v18_full_tier():
    from coordpy.substrate_adapter_v18 import (
        W73_SUBSTRATE_TIER_SUBSTRATE_V18_FULL,
        probe_all_v18_adapters,
        probe_tiny_substrate_v18_adapter,
        probe_synthetic_v18_adapter,
    )
    cap = probe_tiny_substrate_v18_adapter()
    assert (
        str(cap.tier)
        == str(W73_SUBSTRATE_TIER_SUBSTRATE_V18_FULL))
    syn = probe_synthetic_v18_adapter()
    assert str(syn.tier) != str(
        W73_SUBSTRATE_TIER_SUBSTRATE_V18_FULL)
    matrix = probe_all_v18_adapters()
    assert matrix.has_v18_full()


def test_w73_no_version_bump():
    from coordpy import SDK_VERSION, __version__
    assert str(__version__) == "0.5.20"
    assert str(SDK_VERSION) == "coordpy.sdk.v3.43"
