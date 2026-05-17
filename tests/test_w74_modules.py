"""W74 focused module tests.

Per-module checks for the W74 mechanism bundle: V19 substrate
axes, KV V19 15-target ridge, cache V17 + compound-pressure head,
replay V15 + compound-aware routing head, persistent V26 / LHR
V26 / MLSC V22 / consensus V20, hosted V7 modules, handoff V6,
MASC V10 + new regime, TCC V9. Mirrors the W73
``test_w73_modules.py`` style.
"""

from __future__ import annotations


def test_tiny_substrate_v19_determinism_and_axes():
    from coordpy.tiny_substrate_v19 import (
        build_default_tiny_substrate_v19,
        emit_tiny_substrate_v19_forward_witness,
        forward_tiny_substrate_v19,
        record_delayed_repair_event_v19,
        record_compound_failure_window_v19,
        tokenize_bytes_v19,
        W74_DEFAULT_V19_N_LAYERS,
        W74_REPAIR_COMPOUND_REPAIR,
    )
    from coordpy.tiny_substrate_v18 import (
        record_contradiction_event_v18,
        record_replacement_event_v18,
        record_replacement_window_v18,
    )
    from coordpy.tiny_substrate_v17 import (
        record_rejoin_event_v17,
    )
    p = build_default_tiny_substrate_v19(seed=74000)
    ids = tokenize_bytes_v19("w74_modules_test", max_len=8)
    t1, c1 = forward_tiny_substrate_v19(p, ids)
    t2, _ = forward_tiny_substrate_v19(p, ids)
    assert t1.cid() == t2.cid()
    assert (
        t1.compound_repair_rate_per_layer.shape
        == (int(W74_DEFAULT_V19_N_LAYERS),))
    assert (
        t1.compound_pressure_gate_per_layer.shape
        == (int(W74_DEFAULT_V19_N_LAYERS),))
    record_contradiction_event_v18(
        c1.v18_cache, turn=2, role="planner")
    record_replacement_event_v18(
        c1.v18_cache, turn=3, role="planner")
    record_replacement_window_v18(
        c1.v18_cache, contradiction_turn=2,
        replacement_turn=3, rejoin_turn=8,
        replacement_lag_turns=5, role="planner")
    record_rejoin_event_v17(
        c1.v18_cache.v17_cache, turn=8,
        rejoin_kind="branch_rejoin", role="planner")
    record_delayed_repair_event_v19(
        c1, turn=2, role="planner")
    record_compound_failure_window_v19(
        c1, delayed_repair_turn=2, replacement_turn=5,
        rejoin_turn=12, compound_window_turns=10,
        role="planner")
    t3, c3 = forward_tiny_substrate_v19(
        p, ids, v19_kv_cache=c1,
        delayed_repair_pressure=0.5,
        compound_pressure=0.7)
    assert bool(t3.compound_repair_trajectory_cid)
    crr = c3.compound_repair_rate_per_layer
    assert int(crr.shape[0]) == int(W74_DEFAULT_V19_N_LAYERS)
    import numpy as _np
    assert int(_np.count_nonzero(
        crr == W74_REPAIR_COMPOUND_REPAIR)) > 0
    w = emit_tiny_substrate_v19_forward_witness(t3, c3)
    assert int(w.n_layers) == int(W74_DEFAULT_V19_N_LAYERS)


def test_kv_bridge_v19_fifteen_target_ridge_converges():
    from coordpy.kv_bridge_v19 import (
        KVBridgeV19Projection,
        fit_kv_bridge_v19_fifteen_target,
    )
    from coordpy.kv_bridge_v18 import KVBridgeV18Projection
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
    from coordpy.tiny_substrate_v19 import (
        build_default_tiny_substrate_v19,
        tokenize_bytes_v19,
    )
    import numpy as _np
    sub = build_default_tiny_substrate_v19(seed=74000)
    cfg = sub.config.v18.v17.v16.v15.v14.v13.v12.v11.v10.v9
    d_head = cfg.d_model // cfg.n_heads
    kv_b3 = KVBridgeV3Projection.init(
        n_layers=cfg.n_layers, n_heads=cfg.n_heads,
        n_kv_heads=cfg.n_kv_heads, n_inject_tokens=3,
        carrier_dim=6, d_head=int(d_head), seed=74008)
    kv_b18 = KVBridgeV18Projection.init_from_v17(
        KVBridgeV17Projection.init_from_v16(
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
                                                                kv_b3, seed_v4=74009),
                                                            seed_v5=74010),
                                                        seed_v6=74011),
                                                    seed_v7=74012),
                                                seed_v8=74013),
                                            seed_v9=74014),
                                        seed_v10=74015),
                                    seed_v11=74016),
                                seed_v12=74017),
                            seed_v13=74018),
                        seed_v14=74019),
                    seed_v15=74020),
                seed_v16=74021),
            seed_v17=74022),
        seed_v18=74023)
    kv_b19 = KVBridgeV19Projection.init_from_v18(
        kv_b18, seed_v19=74024)
    rng = _np.random.default_rng(74025)
    ids = tokenize_bytes_v19("w74", max_len=4)
    while len(ids) < 4:
        ids.append(0)
    carriers = [
        list(rng.standard_normal(8)) for _ in range(4)]
    targets = []
    for i in range(15):
        t = _np.zeros(cfg.vocab_size)
        t[20 + 15 * i] = 1.0 / (i + 1)
        targets.append(t.tolist())
    _, rep = fit_kv_bridge_v19_fifteen_target(
        params=sub, projection=kv_b19,
        train_carriers=carriers,
        target_delta_logits_stack=targets,
        follow_up_token_ids=ids, n_directions=3,
        compound_target_index=14)
    assert rep.n_targets == 15
    assert rep.converged


def test_cache_controller_v17_fourteen_objective_and_per_role():
    from coordpy.cache_controller_v17 import (
        CacheControllerV17,
        fit_fourteen_objective_ridge_v17,
        fit_per_role_compound_pressure_head_v17,
    )
    import numpy as _np
    cc = CacheControllerV17.init(fit_seed=74100)
    rng = _np.random.default_rng(74101)
    X = rng.standard_normal((10, 4))
    cc, rep = fit_fourteen_objective_ridge_v17(
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
            X[:, 1] * 0.3 + X[:, 3] * 0.4).tolist(),
        target_multi_branch_rejoin=(X[:, 0] * 0.5).tolist(),
        target_budget_primary=(X[:, 2] * 0.3).tolist(),
        target_restart_dominance=(
            X[:, 3] * 0.2 + X[:, 0] * 0.1).tolist(),
        target_delayed_rejoin_after_restart=(
            X[:, 1] * 0.3 + X[:, 2] * 0.3
            + X[:, 3] * 0.2).tolist(),
        target_replacement_after_ctr=(
            X[:, 0] * 0.3 + X[:, 1] * 0.3
            + X[:, 2] * 0.3).tolist(),
        target_compound_repair=(
            X[:, 0] * 0.25 + X[:, 1] * 0.25
            + X[:, 2] * 0.25 + X[:, 3] * 0.25).tolist())
    assert rep.converged and rep.n_objectives == 14
    X15 = rng.standard_normal((10, 15))
    cc, rep2 = fit_per_role_compound_pressure_head_v17(
        controller=cc, role="planner",
        train_features=X15.tolist(),
        target_compound_priorities=(
            X15[:, 0] * 0.3 + X15[:, 14] * 0.4).tolist())
    assert rep2.converged


def test_replay_controller_v15_22_regimes_and_compound_routing():
    from coordpy.replay_controller_v15 import (
        ReplayControllerV15,
        W74_COMPOUND_AWARE_ROUTING_LABELS,
        W74_REPLAY_REGIMES_V15,
        fit_replay_controller_v15_per_role,
        fit_replay_v15_compound_aware_routing_head,
    )
    from coordpy.replay_controller import ReplayCandidate
    import numpy as _np
    assert len(W74_REPLAY_REGIMES_V15) == 22
    assert (
        "compound_repair_after_delayed_repair_then_replacement"
        "_regime" in W74_REPLAY_REGIMES_V15)
    assert len(W74_COMPOUND_AWARE_ROUTING_LABELS) == 12
    assert "compound_route" in W74_COMPOUND_AWARE_ROUTING_LABELS
    rc = ReplayControllerV15.init()
    cands = {
        r: [ReplayCandidate(
            100, 1000, 50, 0.1, 0.0, 0.3, True, True, 0)]
        for r in W74_REPLAY_REGIMES_V15}
    decs = {
        r: ["choose_reuse"]
        for r in W74_REPLAY_REGIMES_V15}
    rc, _ = fit_replay_controller_v15_per_role(
        controller=rc, role="planner",
        train_candidates_per_regime=cands,
        train_decisions_per_regime=decs)
    rng = _np.random.default_rng(74200)
    X = rng.standard_normal((48, 10))
    labs = [
        W74_COMPOUND_AWARE_ROUTING_LABELS[
            i % len(W74_COMPOUND_AWARE_ROUTING_LABELS)]
        for i in range(48)]
    rc, rep = fit_replay_v15_compound_aware_routing_head(
        controller=rc, train_team_features=X.tolist(),
        train_routing_labels=labs)
    assert rep.n_classes == 12


def test_persistent_v26_axes():
    from coordpy.persistent_latent_v26 import (
        W74_DEFAULT_V26_DISTRACTOR_RANK,
        W74_DEFAULT_V26_MAX_CHAIN_WALK_DEPTH,
        W74_DEFAULT_V26_N_LAYERS,
    )
    assert W74_DEFAULT_V26_N_LAYERS == 25
    assert W74_DEFAULT_V26_DISTRACTOR_RANK == 25
    assert W74_DEFAULT_V26_MAX_CHAIN_WALK_DEPTH == 2097152


def test_lhr_v26_max_k_and_compound_dim():
    from coordpy.long_horizon_retention_v26 import (
        LongHorizonReconstructionV26Head,
        W74_DEFAULT_LHR_V26_MAX_K,
    )
    h = LongHorizonReconstructionV26Head.init(seed=74300)
    assert int(h.max_k) >= int(W74_DEFAULT_LHR_V26_MAX_K)
    assert int(h.compound_dim) == 8


def test_mlsc_v22_compound_chain_merge():
    from coordpy.mergeable_latent_capsule_v22 import (
        MergeOperatorV22, wrap_v21_as_v22,
    )
    from coordpy.mergeable_latent_capsule_v21 import (
        wrap_v20_as_v21,
    )
    from coordpy.mergeable_latent_capsule_v20 import (
        wrap_v19_as_v20,
    )
    from coordpy.mergeable_latent_capsule_v19 import (
        wrap_v18_as_v19,
    )
    from coordpy.mergeable_latent_capsule_v18 import (
        wrap_v17_as_v18,
    )
    from coordpy.mergeable_latent_capsule_v17 import (
        wrap_v16_as_v17,
    )
    from coordpy.mergeable_latent_capsule_v16 import (
        wrap_v15_as_v16,
    )
    from coordpy.mergeable_latent_capsule_v15 import (
        wrap_v14_as_v15,
    )
    from coordpy.mergeable_latent_capsule_v14 import (
        wrap_v13_as_v14,
    )
    from coordpy.mergeable_latent_capsule_v13 import (
        wrap_v12_as_v13,
    )
    from coordpy.mergeable_latent_capsule_v12 import (
        wrap_v11_as_v12,
    )
    from coordpy.mergeable_latent_capsule_v11 import (
        wrap_v10_as_v11,
    )
    from coordpy.mergeable_latent_capsule_v10 import (
        wrap_v9_as_v10,
    )
    from coordpy.mergeable_latent_capsule_v9 import wrap_v8_as_v9
    from coordpy.mergeable_latent_capsule_v8 import wrap_v7_as_v8
    from coordpy.mergeable_latent_capsule_v7 import wrap_v6_as_v7
    from coordpy.mergeable_latent_capsule_v6 import wrap_v5_as_v6
    from coordpy.mergeable_latent_capsule_v5 import wrap_v4_as_v5
    from coordpy.mergeable_latent_capsule_v4 import wrap_v3_as_v4
    from coordpy.mergeable_latent_capsule_v3 import (
        make_root_capsule_v3,
    )
    v3 = make_root_capsule_v3(
        branch_id="w74_mod", payload=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
        fact_tags=("w74",), confidence=0.9, trust=0.9,
        turn_index=0)
    v21 = wrap_v20_as_v21(wrap_v19_as_v20(wrap_v18_as_v19(
        wrap_v17_as_v18(wrap_v16_as_v17(wrap_v15_as_v16(
            wrap_v14_as_v15(wrap_v13_as_v14(wrap_v12_as_v13(
                wrap_v11_as_v12(wrap_v10_as_v11(wrap_v9_as_v10(
                    wrap_v8_as_v9(wrap_v7_as_v8(wrap_v6_as_v7(
                        wrap_v5_as_v6(wrap_v4_as_v5(
                            wrap_v3_as_v4(v3))))))))))))))))))
    a = wrap_v21_as_v22(
        v21,
        compound_repair_trajectory_chain=("cmp_a",),
        delayed_repair_chain=("dr_a",))
    b = wrap_v21_as_v22(
        v21,
        compound_repair_trajectory_chain=("cmp_b",),
        delayed_repair_chain=("dr_b",))
    merged = MergeOperatorV22(factor_dim=6).merge([a, b])
    assert "cmp_a" in merged.compound_repair_trajectory_chain
    assert "cmp_b" in merged.compound_repair_trajectory_chain
    assert "dr_a" in merged.delayed_repair_chain
    assert "dr_b" in merged.delayed_repair_chain


def test_consensus_v20_stages():
    from coordpy.consensus_fallback_controller_v20 import (
        W74_CONSENSUS_V20_STAGES,
    )
    assert len(W74_CONSENSUS_V20_STAGES) >= 34
    assert (
        "compound_repair_arbiter" in W74_CONSENSUS_V20_STAGES)
    assert (
        "compound_repair_after_delayed_repair_then_"
        "replacement_arbiter" in W74_CONSENSUS_V20_STAGES)


def test_substrate_adapter_v19_tier():
    from coordpy.substrate_adapter_v19 import (
        W74_SUBSTRATE_TIER_SUBSTRATE_V19_FULL,
        probe_tiny_substrate_v19_adapter,
    )
    cap = probe_tiny_substrate_v19_adapter()
    assert (
        cap.tier == W74_SUBSTRATE_TIER_SUBSTRATE_V19_FULL)


def test_hosted_router_v7_determinism():
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
        HostedRoutingRequestV6,
    )
    from coordpy.hosted_router_controller_v7 import (
        HostedRouterControllerV7, HostedRoutingRequestV7,
    )
    reg = default_hosted_registry()
    rc1 = HostedRouterControllerV7.init(
        reg, {"openrouter_paid": 0.85, "openai_paid": 0.92})
    rc2 = HostedRouterControllerV7.init(
        reg, {"openrouter_paid": 0.85, "openai_paid": 0.92})
    req = HostedRoutingRequestV7(
        inner_v6=HostedRoutingRequestV6(
            inner_v5=HostedRoutingRequestV5(
                inner_v4=HostedRoutingRequestV4(
                    inner_v3=HostedRoutingRequestV3(
                        inner_v2=HostedRoutingRequestV2(
                            inner_v1=HostedRoutingRequest(
                                request_cid="mod-test",
                                input_tokens=500,
                                expected_output_tokens=100,
                                require_logprobs=True,
                                require_prefix_cache=True,
                                data_policy_required="no_log",
                                max_latency_ms=2000.0,
                                max_cost_usd=50.0),
                            weight_cost=1.0,
                            weight_latency=0.5,
                            weight_success=0.3),
                        visible_token_budget=128,
                        baseline_token_cost=512,
                        repair_dominance_label=1),
                    restart_pressure=0.7),
                rejoin_pressure=0.7),
            replacement_pressure=0.7),
        compound_pressure=0.7)
    d1 = rc1.decide_v7(req)
    d2 = rc2.decide_v7(req)
    assert d1.cid() == d2.cid()


def test_handoff_v6_promotion_and_fallback():
    from coordpy.hosted_real_handoff_coordinator import (
        HandoffRequest,
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
        HandoffRequestV5,
    )
    from coordpy.hosted_real_handoff_coordinator_v6 import (
        HandoffRequestV6, HostedRealHandoffCoordinatorV6,
        W74_HANDOFF_DECISION_COMPOUND_REPAIR_FALLBACK,
    )
    c = HostedRealHandoffCoordinatorV6()
    base = HandoffRequest(
        request_cid="mod-test",
        needs_text_only=True,
        needs_substrate_state_access=False)
    req_v2 = HandoffRequestV2(
        inner_v1=base, visible_token_budget=256,
        baseline_token_cost=512, dominant_repair_label=0)
    req_v3 = HandoffRequestV3(
        inner_v2=req_v2, restart_pressure=0.0,
        delayed_repair_trajectory_cid="",
        delay_turns=0, expected_substrate_trust=0.7)
    req_v4 = HandoffRequestV4(
        inner_v3=req_v3, rejoin_pressure=0.0,
        restart_repair_trajectory_cid="",
        rejoin_lag_turns=0, expected_substrate_trust_v4=0.7)
    req_v5 = HandoffRequestV5(
        inner_v4=req_v4, replacement_pressure=0.0,
        replacement_repair_trajectory_cid="",
        replacement_lag_turns=0,
        expected_substrate_trust_v5=0.7)
    e_promote = c.decide_v6(
        req_v6=HandoffRequestV6(
            inner_v5=req_v5, compound_pressure=0.8,
            expected_substrate_trust_v6=0.7))
    assert (e_promote.decision_v6
            == W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY)
    e_fb = c.decide_v6(
        req_v6=HandoffRequestV6(
            inner_v5=req_v5,
            compound_repair_trajectory_cid="cmp_xyz",
            compound_window_turns=5,
            expected_substrate_trust_v6=0.7))
    assert (e_fb.decision_v6
            == W74_HANDOFF_DECISION_COMPOUND_REPAIR_FALLBACK)


def test_masc_v10_compound_regime_beats_v18():
    from coordpy.multi_agent_substrate_coordinator_v10 import (
        MultiAgentSubstrateCoordinatorV10,
        W74_MASC_V10_REGIME_REPLACEMENT_AFTER_DELAYED_REPAIR,
    )
    masc = MultiAgentSubstrateCoordinatorV10()
    _, agg = masc.run_batch(
        seeds=list(range(15)),
        regime=W74_MASC_V10_REGIME_REPLACEMENT_AFTER_DELAYED_REPAIR)
    assert agg.v19_beats_v18_rate >= 0.5
    assert agg.tsc_v19_beats_tsc_v18_rate >= 0.5


def test_tcc_v9_compound_arbiters():
    from coordpy.team_consensus_controller_v9 import (
        TeamConsensusControllerV9,
        W74_TC_V9_DECISION_COMPOUND_PRESSURE,
        W74_TC_V9_DECISION_COMPOUND_REPAIR_AFTER_DRTR,
    )
    tcc = TeamConsensusControllerV9()
    out_p = tcc.decide_v9(
        regime="baseline",
        agent_guesses=[1.0, -1.0, 0.5, 0.2],
        agent_confidences=[0.8, 0.6, 0.7, 0.7],
        substrate_replay_trust=0.7,
        compound_pressure=0.8,
        agent_compound_recovery_flags=[1, 0, 1, 0])
    assert (out_p["decision"]
            == W74_TC_V9_DECISION_COMPOUND_PRESSURE)
    out_d = tcc.decide_v9(
        regime=(
            "replacement_after_delayed_repair_under_budget"),
        agent_guesses=[1.0, -1.0, 0.5, 0.2],
        agent_confidences=[0.8, 0.6, 0.7, 0.7],
        substrate_replay_trust=0.7,
        compound_repair_trajectory_cid="cmp_abc",
        compound_window_turns=8,
        agent_compound_absorption_scores=[
            0.95, 0.5, 0.4, 0.3])
    assert (out_d["decision"]
            == W74_TC_V9_DECISION_COMPOUND_REPAIR_AFTER_DRTR)


def test_no_version_bump_invariant():
    from coordpy import SDK_VERSION, __version__
    assert str(__version__) == "0.5.20"
    assert str(SDK_VERSION) == "coordpy.sdk.v3.43"
