"""W69 module focused tests (post-W68 milestone).

Per-module sanity tests for the W69 line. R-156/R-157/R-158/
R-159/R-160 suites exercise the H-bars end-to-end.
"""

from __future__ import annotations


def test_tiny_substrate_v14_determinism_and_axes():
    from coordpy.tiny_substrate_v14 import (
        build_default_tiny_substrate_v14,
        forward_tiny_substrate_v14,
        emit_tiny_substrate_v14_forward_witness,
        record_multi_branch_rejoin_witness_v14,
        trigger_silent_corruption_v14,
        repair_silent_corruption_v14,
        substrate_self_checksum_v14,
        substrate_multi_branch_rejoin_flops_v14,
        substrate_silent_corruption_detect_v14,
        tokenize_bytes_v14,
    )
    p = build_default_tiny_substrate_v14(seed=691)
    ids = tokenize_bytes_v14("w69-test", max_len=12)
    t1, c1 = forward_tiny_substrate_v14(p, ids)
    t2, _ = forward_tiny_substrate_v14(p, ids)
    assert t1.cid() == t2.cid()
    assert p.config.v13.v12.v11.v10.v9.n_layers == 16
    assert c1.multi_branch_rejoin_witness.shape == (
        p.config.v13.v12.v11.v10.v9.n_layers,
        p.config.v13.v12.v11.v10.v9.n_heads,
        p.config.v13.v12.v11.v10.v9.max_len)
    assert t1.v14_gate_score_per_layer.shape == (
        p.config.v13.v12.v11.v10.v9.n_layers,)
    record_multi_branch_rejoin_witness_v14(
        c1, layer_index=0, head_index=0, slot=0,
        witness=0.7)
    assert c1.multi_branch_rejoin_witness[0, 0, 0] == 0.7
    trigger_silent_corruption_v14(
        c1, role="planner", corrupted_bytes=2,
        member_replaced=True, detect_turn=3)
    entry = c1.silent_corruption_witness["planner"]
    assert entry["member_replaced"]
    assert entry["corrupted_bytes"] == 2
    assert repair_silent_corruption_v14(
        c1, role="planner", repair_turn=5)
    cid1 = substrate_self_checksum_v14(c1)
    cid2 = substrate_self_checksum_v14(c1)
    assert cid1 == cid2
    sav = substrate_multi_branch_rejoin_flops_v14(
        n_tokens=128, n_branches=4)
    assert sav["saving_ratio"] >= 0.8
    det = substrate_silent_corruption_detect_v14(n_roles=6)
    assert det["passes_95pct_floor"]
    w = emit_tiny_substrate_v14_forward_witness(t1, c1)
    assert w.silent_corruption_count == 1
    assert w.substrate_self_checksum_cid != ""


def test_kv_bridge_v14_ten_target_and_falsifier():
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
    from coordpy.kv_bridge_v14 import (
        KVBridgeV14Projection,
        compute_silent_corruption_fingerprint_v14,
        fit_kv_bridge_v14_ten_target,
        probe_kv_bridge_v14_multi_branch_rejoin_falsifier,
    )
    from coordpy.tiny_substrate_v14 import (
        build_default_tiny_substrate_v14, tokenize_bytes_v14,
    )
    p = build_default_tiny_substrate_v14(seed=692)
    cfg = p.config.v13.v12.v11.v10.v9
    d_head = cfg.d_model // cfg.n_heads
    b3 = KVBridgeV3Projection.init(
        n_layers=cfg.n_layers, n_heads=cfg.n_heads,
        n_kv_heads=cfg.n_kv_heads, n_inject_tokens=3,
        carrier_dim=8, d_head=d_head, seed=69200)
    b14 = KVBridgeV14Projection.init_from_v13(
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
                                                b3, seed_v4=69201),
                                            seed_v5=69202),
                                        seed_v6=69203),
                                    seed_v7=69204),
                                seed_v8=69205),
                            seed_v9=69206),
                        seed_v10=69207),
                    seed_v11=69208),
                seed_v12=69209),
            seed_v13=69210),
        seed_v14=69211)
    rng = _np.random.default_rng(692)
    ids = tokenize_bytes_v14("kv14", max_len=4)
    while len(ids) < 4:
        ids.append(0)
    carriers = [list(rng.standard_normal(8)) for _ in range(4)]
    targets = []
    for i in range(10):
        t = _np.zeros(cfg.vocab_size)
        t[20 + 15 * i] = 1.0 / (i + 1)
        targets.append(t.tolist())
    _, rep = fit_kv_bridge_v14_ten_target(
        params=p, projection=b14, train_carriers=carriers,
        target_delta_logits_stack=targets,
        follow_up_token_ids=ids, n_directions=3)
    assert rep.n_targets == 10
    assert rep.converged
    fp = compute_silent_corruption_fingerprint_v14(
        role="planner", corrupted_bytes=2, member_replaced=True,
        task_id="t", team_id="team")
    assert len(fp) == 60
    fal = probe_kv_bridge_v14_multi_branch_rejoin_falsifier(
        multi_branch_rejoin_flag=0.5)
    assert fal.falsifier_score == 0.0


def test_replay_controller_v10_sixteen_regimes():
    from coordpy.replay_controller_v10 import (
        W69_REPLAY_REGIMES_V10,
        W69_MULTI_BRANCH_REJOIN_ROUTING_LABELS,
    )
    assert len(W69_REPLAY_REGIMES_V10) == 16
    assert len(W69_MULTI_BRANCH_REJOIN_ROUTING_LABELS) == 7


def test_consensus_v15_twenty_four_stages():
    from coordpy.consensus_fallback_controller_v15 import (
        W69_CONSENSUS_V15_STAGES,
    )
    assert len(W69_CONSENSUS_V15_STAGES) == 24


def test_tvs_v18_nineteen_arms_and_uncertainty_v17_sixteen_axes():
    from coordpy.transcript_vs_shared_arbiter_v18 import (
        W69_TVS_V18_ARMS,
    )
    assert len(W69_TVS_V18_ARMS) == 19
    from coordpy.uncertainty_layer_v17 import (
        compose_uncertainty_report_v17,
    )
    u = compose_uncertainty_report_v17(
        multi_branch_rejoin_resolution_fidelities=[0.8, 0.9],
        partial_contradiction_resolution_fidelities=[0.7, 0.8],
        confidences=[0.7, 0.8],
        trusts=[0.9, 0.85],
        substrate_fidelities=[0.95, 0.92],
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
        branch_merge_reconciliation_fidelities=[0.78, 0.74])
    assert u.n_axes == 16


def test_ecc_v21_total_codes_and_rate_ceiling():
    from coordpy.ecc_codebook_v21 import (
        ECCCodebookV21, probe_ecc_v21_rate_floor_falsifier,
    )
    cb = ECCCodebookV21.init(seed=69200)
    assert cb.total_codes >= 2 ** 35
    fal = probe_ecc_v21_rate_floor_falsifier(codebook=cb)
    assert fal["target_exceeds_ceiling"]


def test_crc_v17_buckets_and_burst_bits():
    from coordpy.corruption_robust_carrier_v17 import (
        W69_CRC_V17_KV_FINGERPRINT_BUCKETS,
        W69_CRC_V17_ADVERSARIAL_BURST_BITS,
    )
    assert W69_CRC_V17_KV_FINGERPRINT_BUCKETS == 131072
    assert W69_CRC_V17_ADVERSARIAL_BURST_BITS == 37


def test_persistent_v21_layers_and_chain_depth():
    from coordpy.persistent_latent_v21 import (
        W69_DEFAULT_V21_N_LAYERS,
        W69_DEFAULT_V21_MAX_CHAIN_WALK_DEPTH,
        W69_DEFAULT_V21_DISTRACTOR_RANK,
    )
    assert W69_DEFAULT_V21_N_LAYERS == 20
    assert W69_DEFAULT_V21_MAX_CHAIN_WALK_DEPTH == 65536
    assert W69_DEFAULT_V21_DISTRACTOR_RANK == 20


def test_multi_hop_v19_backends_and_chain_len():
    from coordpy.multi_hop_translator_v19 import (
        W69_DEFAULT_MH_V19_BACKENDS,
        W69_DEFAULT_MH_V19_CHAIN_LEN,
    )
    assert len(W69_DEFAULT_MH_V19_BACKENDS) == 48
    assert W69_DEFAULT_MH_V19_CHAIN_LEN == 38


def test_substrate_adapter_v14_full_tier_only_at_in_repo():
    from coordpy.substrate_adapter_v14 import (
        W69_SUBSTRATE_TIER_SUBSTRATE_V14_FULL,
        probe_all_v14_adapters,
    )
    matrix = probe_all_v14_adapters()
    in_repo = matrix.by_name().get("tiny_substrate_v14")
    synthetic = matrix.by_name().get("synthetic")
    assert in_repo is not None
    assert in_repo.tier == W69_SUBSTRATE_TIER_SUBSTRATE_V14_FULL
    assert synthetic is not None
    assert synthetic.tier != (
        W69_SUBSTRATE_TIER_SUBSTRATE_V14_FULL)


def test_masc_v5_nine_regimes_and_twelve_policies():
    from coordpy.multi_agent_substrate_coordinator_v5 import (
        MultiAgentSubstrateCoordinatorV5,
        W69_MASC_V5_POLICIES, W69_MASC_V5_REGIMES,
    )
    assert len(W69_MASC_V5_REGIMES) == 9
    assert len(W69_MASC_V5_POLICIES) == 12
    masc = MultiAgentSubstrateCoordinatorV5()
    _, agg = masc.run_batch(seeds=list(range(5)),
                            regime=W69_MASC_V5_REGIMES[0])
    assert agg.v14_beats_v13_rate >= 0.5


def test_tcc_v4_decisions():
    from coordpy.team_consensus_controller_v4 import (
        TeamConsensusControllerV4,
        W69_TC_V4_DECISION_MULTI_BRANCH_REJOIN,
        W69_TC_V4_DECISION_SILENT_CORRUPTION_PLUS_REPLACEMENT,
    )
    from coordpy.multi_agent_substrate_coordinator_v5 import (
        W69_MASC_V5_REGIME_MULTI_BRANCH_REJOIN,
        W69_MASC_V5_REGIME_SILENT_CORRUPTION_PLUS_REPLACEMENT,
    )
    tcc = TeamConsensusControllerV4()
    out = tcc.decide_v4(
        regime=W69_MASC_V5_REGIME_MULTI_BRANCH_REJOIN,
        agent_guesses=[1.0, -1.0, 0.5],
        agent_confidences=[0.8, 0.6, 0.7],
        substrate_replay_trust=0.7,
        transcript_available=False,
        branch_assignments=[0, 1, 2])
    assert out["decision"] == (
        W69_TC_V4_DECISION_MULTI_BRANCH_REJOIN)
    out2 = tcc.decide_v4(
        regime=W69_MASC_V5_REGIME_SILENT_CORRUPTION_PLUS_REPLACEMENT,
        agent_guesses=[1.0, 0.5, 0.7, 0.6],
        agent_confidences=[0.8, 0.6, 0.7, 0.7],
        substrate_replay_trust=0.7,
        transcript_available=False,
        corrupted_role_indices=[1])
    assert out2["decision"] == (
        W69_TC_V4_DECISION_SILENT_CORRUPTION_PLUS_REPLACEMENT)


def test_hosted_real_handoff_coordinator_routes_three_ways():
    from coordpy.hosted_real_handoff_coordinator import (
        HandoffRequest, HostedRealHandoffCoordinator,
        W69_HANDOFF_DECISION_HOSTED_ONLY,
        W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY,
        W69_HANDOFF_DECISION_HOSTED_WITH_REAL_AUDIT,
        probe_hosted_real_handoff_falsifier,
    )
    co = HostedRealHandoffCoordinator()
    e_text = co.decide(req=HandoffRequest(
        request_cid="t", needs_text_only=True,
        needs_substrate_state_access=False))
    e_sub = co.decide(req=HandoffRequest(
        request_cid="s", needs_text_only=False,
        needs_substrate_state_access=True))
    e_audit = co.decide(req=HandoffRequest(
        request_cid="b", needs_text_only=True,
        needs_substrate_state_access=True))
    assert e_text.decision == W69_HANDOFF_DECISION_HOSTED_ONLY
    assert e_sub.decision == (
        W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY)
    assert e_audit.decision == (
        W69_HANDOFF_DECISION_HOSTED_WITH_REAL_AUDIT)
    honest = probe_hosted_real_handoff_falsifier(
        envelope=e_text, claim_satisfied=False)
    dishonest = probe_hosted_real_handoff_falsifier(
        envelope=e_text, claim_satisfied=True)
    assert honest.falsifier_score == 0.0
    assert dishonest.falsifier_score == 1.0


def test_hosted_real_substrate_boundary_v2_blocked_axes():
    from coordpy.hosted_real_substrate_boundary_v2 import (
        W69_HOSTED_PLANE_BLOCKED_AXES_V2,
        build_default_hosted_real_substrate_boundary_v2,
        build_wall_report_v2,
    )
    b = build_default_hosted_real_substrate_boundary_v2()
    assert len(b.blocked_axes) >= 19
    wr = build_wall_report_v2(boundary=b)
    assert len(wr.real_substrate_only_axes) >= 60
    assert len(b.frontier_blocked_axes) >= 3


def test_deep_substrate_hybrid_v14_fourteen_way():
    from coordpy.deep_substrate_hybrid_v13 import (
        DeepSubstrateHybridV13ForwardWitness,
    )
    from coordpy.deep_substrate_hybrid_v14 import (
        DeepSubstrateHybridV14,
        deep_substrate_hybrid_v14_forward,
    )
    from coordpy.cache_controller_v12 import (
        CacheControllerV12, fit_nine_objective_ridge_v12,
    )
    from coordpy.replay_controller_v10 import (
        ReplayControllerV10, W69_REPLAY_REGIMES_V10,
        W69_MULTI_BRANCH_REJOIN_ROUTING_LABELS,
        fit_replay_controller_v10_per_role,
        fit_replay_v10_multi_branch_rejoin_routing_head,
    )
    from coordpy.replay_controller import ReplayCandidate
    import numpy as _np
    cc = CacheControllerV12.init(fit_seed=69100)
    rng = _np.random.default_rng(69100)
    X = rng.standard_normal((10, 4))
    cc, _ = fit_nine_objective_ridge_v12(
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
        target_multi_branch_rejoin=(
            X[:, 0] * 0.5 + X[:, 1] * 0.2).tolist())
    rc = ReplayControllerV10.init()
    cands = {
        r: [ReplayCandidate(
            100, 1000, 50, 0.1, 0.0, 0.3, True, True, 0)]
        for r in W69_REPLAY_REGIMES_V10}
    decs = {r: ["choose_reuse"] for r in W69_REPLAY_REGIMES_V10}
    rc, _ = fit_replay_controller_v10_per_role(
        controller=rc, role="planner",
        train_candidates_per_regime=cands,
        train_decisions_per_regime=decs)
    X_team = rng.standard_normal((40, 11))
    labs = [
        W69_MULTI_BRANCH_REJOIN_ROUTING_LABELS[i % 7]
        for i in range(40)]
    rc, _ = fit_replay_v10_multi_branch_rejoin_routing_head(
        controller=rc, train_team_features=X_team.tolist(),
        train_routing_labels=labs)
    v13_witness = DeepSubstrateHybridV13ForwardWitness(
        schema="coordpy.deep_substrate_hybrid_v13.v1",
        hybrid_cid="", inner_v12_witness_cid="",
        thirteen_way=True,
        cache_controller_v11_fired=True,
        replay_controller_v9_fired=True,
        partial_contradiction_witness_active=True,
        agent_replacement_active=True,
        prefix_reuse_active=True,
        team_consensus_controller_v3_active=True,
        partial_contradiction_witness_l1=0.5,
        agent_replacement_count=1,
        prefix_reuse_count=2)
    hybrid = DeepSubstrateHybridV14(inner_v13=None)
    w = deep_substrate_hybrid_v14_forward(
        hybrid=hybrid, v13_witness=v13_witness,
        cache_controller_v12=cc,
        replay_controller_v10=rc,
        multi_branch_rejoin_witness_l1=0.5,
        silent_corruption_count=1,
        substrate_self_checksum_cid="abc",
        n_team_consensus_v4_invocations=1)
    assert w.fourteen_way
