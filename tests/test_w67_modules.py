"""W67 module focused tests (post-W66 milestone).

Per-module sanity tests for the W67 line. R-149/R-150/R-151 suites
exercise the H-bars end-to-end.
"""

from __future__ import annotations

import numpy as np


def test_tiny_substrate_v12_determinism_and_axes():
    from coordpy.tiny_substrate_v12 import (
        build_default_tiny_substrate_v12,
        forward_tiny_substrate_v12,
        emit_tiny_substrate_v12_forward_witness,
        record_branch_merge_witness_v12,
        trigger_role_dropout_recovery_v12,
        substrate_snapshot_fork_v12,
        substrate_branch_merge_v12,
        substrate_branch_merge_flops_v12,
        tokenize_bytes_v12,
    )
    p = build_default_tiny_substrate_v12(seed=671)
    ids = tokenize_bytes_v12("w67-test", max_len=12)
    t1, c1 = forward_tiny_substrate_v12(p, ids)
    t2, _ = forward_tiny_substrate_v12(p, ids)
    assert t1.cid() == t2.cid()
    assert p.config.v11.v10.v9.n_layers == 14
    assert c1.branch_merge_witness.shape == (
        p.config.v11.v10.v9.n_layers,
        p.config.v11.v10.v9.n_heads,
        p.config.v11.v10.v9.max_len)
    assert t1.v12_gate_score_per_layer.shape == (
        p.config.v11.v10.v9.n_layers,)
    record_branch_merge_witness_v12(
        c1, layer_index=0, head_index=0, slot=0, witness=0.7)
    assert c1.branch_merge_witness[0, 0, 0] == 0.7
    trigger_role_dropout_recovery_v12(
        c1, absent_role="r0", covering_role="r1", window=2)
    fork = substrate_snapshot_fork_v12(
        c1, branch_ids=["b0", "b1"], fork_label="test_fork")
    assert len(fork.per_branch_cid) == 2
    merge = substrate_branch_merge_v12(
        c1, fork=fork,
        branch_payloads={
            "b0": [1.0, 2.0, 3.0], "b1": [1.1, 2.1, 3.1]})
    assert merge.n_branches_merged == 2
    saving = substrate_branch_merge_flops_v12(
        n_tokens=128, n_branches=4)
    assert saving["saving_ratio"] >= 0.6
    w = emit_tiny_substrate_v12_forward_witness(t1, c1)
    assert w.role_dropout_recovery_count == 1


def test_kv_bridge_v12_eight_target_and_falsifier():
    from coordpy.kv_bridge_v3 import KVBridgeV3Projection
    from coordpy.kv_bridge_v4 import KVBridgeV4Projection
    from coordpy.kv_bridge_v5 import KVBridgeV5Projection
    from coordpy.kv_bridge_v6 import KVBridgeV6Projection
    from coordpy.kv_bridge_v7 import KVBridgeV7Projection
    from coordpy.kv_bridge_v8 import KVBridgeV8Projection
    from coordpy.kv_bridge_v9 import KVBridgeV9Projection
    from coordpy.kv_bridge_v10 import KVBridgeV10Projection
    from coordpy.kv_bridge_v11 import KVBridgeV11Projection
    from coordpy.kv_bridge_v12 import (
        KVBridgeV12Projection,
        compute_role_pair_fingerprint_v12,
        fit_kv_bridge_v12_eight_target,
        probe_kv_bridge_v12_branch_merge_falsifier,
    )
    from coordpy.tiny_substrate_v12 import (
        build_default_tiny_substrate_v12, tokenize_bytes_v12,
    )
    p = build_default_tiny_substrate_v12(seed=672)
    cfg = p.config.v11.v10.v9
    d_head = cfg.d_model // cfg.n_heads
    proj = KVBridgeV12Projection.init_from_v11(
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
                                            seed=67200),
                                        seed_v4=67201),
                                    seed_v5=67202),
                                seed_v6=67203),
                            seed_v7=67204),
                        seed_v8=67205),
                    seed_v9=67206),
                seed_v10=67207),
            seed_v11=67208),
        seed_v12=67209)
    rng = np.random.default_rng(0)
    ids = tokenize_bytes_v12("kv12", max_len=4)
    while len(ids) < 4:
        ids.append(0)
    carriers = [list(rng.standard_normal(8)) for _ in range(4)]
    targets = []
    for i in range(8):
        t = np.zeros(cfg.vocab_size)
        t[100 + 20 * i] = 1.0 / (i + 1)
        targets.append(t.tolist())
    _, report = fit_kv_bridge_v12_eight_target(
        params=p, projection=proj,
        train_carriers=carriers,
        target_delta_logits_stack=targets,
        follow_up_token_ids=ids, n_directions=3)
    assert report.n_targets == 8
    assert report.branch_merge_post <= report.branch_merge_pre + 1e-3
    fp = compute_role_pair_fingerprint_v12(
        role_a="planner", role_b="analyst",
        task_id="t", team_id="x", branch_id="b1")
    assert len(fp) == 40
    fals = probe_kv_bridge_v12_branch_merge_falsifier(
        branch_merge_flag=0.5)
    assert fals.falsifier_score == 0.0


def test_hsb_v11_eight_target_and_probe():
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
        HiddenStateBridgeV9Projection,
    )
    from coordpy.hidden_state_bridge_v10 import (
        HiddenStateBridgeV10Projection,
    )
    from coordpy.hidden_state_bridge_v11 import (
        HiddenStateBridgeV11Projection,
        compute_hsb_v11_branch_merge_margin,
        fit_hsb_v11_eight_target,
        probe_hsb_v11_hidden_vs_branch_merge,
    )
    from coordpy.tiny_substrate_v11 import (
        build_default_tiny_substrate_v11,
    )
    p = build_default_tiny_substrate_v11(seed=673)
    cfg = p.config.v10.v9
    proj = HiddenStateBridgeV11Projection.init_from_v10(
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
                                            n_tokens=6,
                                            carrier_dim=6,
                                            d_model=cfg.d_model,
                                            seed=67300),
                                        n_heads=cfg.n_heads,
                                        seed_v3=67301),
                                    seed_v4=67302),
                                n_positions=3, seed_v5=67303),
                            seed_v6=67304),
                        seed_v7=67305),
                    seed_v8=67306),
                seed_v9=67307),
            seed_v10=67308),
        seed_v11=67309)
    rng = np.random.default_rng(0)
    ids = list(range(1, 5))
    carriers = [list(rng.standard_normal(6)) for _ in range(4)]
    targets = []
    for i in range(8):
        t = np.zeros(cfg.vocab_size)
        t[100 + 20 * i] = 1.0 / (i + 1)
        targets.append(t.tolist())
    _, report = fit_hsb_v11_eight_target(
        params=p.v3_params, projection=proj,
        train_carriers=carriers,
        target_delta_logits_stack=targets,
        follow_up_token_ids=ids, n_directions=3)
    assert report.n_targets == 8
    margin = compute_hsb_v11_branch_merge_margin(
        hidden_residual_l2=0.1, kv_residual_l2=0.5,
        prefix_residual_l2=0.4, replay_residual_l2=0.3,
        recover_residual_l2=0.35,
        branch_merge_residual_l2=0.45)
    assert margin > 0.0
    L, H = 3, 3
    h = np.full((L, H), 0.1)
    b = np.full((L, H), 0.5)
    res = probe_hsb_v11_hidden_vs_branch_merge(
        n_layers=L, n_heads=H,
        hidden_residual_l2_grid=h.tolist(),
        branch_merge_residual_l2_grid=b.tolist())
    assert res["win_rate_mean"] >= 0.99


def test_replay_controller_v8_twelve_regimes():
    from coordpy.replay_controller import ReplayCandidate
    from coordpy.replay_controller_v5 import ReplayControllerV5
    from coordpy.replay_controller_v6 import ReplayControllerV6
    from coordpy.replay_controller_v7 import ReplayControllerV7
    from coordpy.replay_controller_v8 import (
        ReplayControllerV8,
        W67_BRANCH_MERGE_ROUTING_LABELS,
        W67_REPLAY_REGIMES_V8,
        fit_replay_controller_v8_per_role,
        fit_replay_v8_branch_merge_routing_head,
    )
    assert len(W67_REPLAY_REGIMES_V8) == 12
    rcv5 = ReplayControllerV5.init()
    rcv6 = ReplayControllerV6.init(inner_v5=rcv5)
    rcv7 = ReplayControllerV7.init(inner_v6=rcv6)
    ctrl = ReplayControllerV8.init(inner_v7=rcv7)
    cands = {
        r: [ReplayCandidate(
            100, 1000, 50, 0.1, 0.0, 0.3, True, True, 0)]
        for r in W67_REPLAY_REGIMES_V8}
    decs = {
        r: ["choose_reuse"] for r in W67_REPLAY_REGIMES_V8}
    ctrl, report = fit_replay_controller_v8_per_role(
        controller=ctrl, role="planner",
        train_candidates_per_regime=cands,
        train_decisions_per_regime=decs)
    assert report.converged
    assert report.n_role_regime_heads_v8 == 12
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 10))
    labs = []
    for i in range(30):
        if X[i, 0] > 0.5:
            labs.append(W67_BRANCH_MERGE_ROUTING_LABELS[0])
        elif X[i, 1] > 0.0:
            labs.append(W67_BRANCH_MERGE_ROUTING_LABELS[1])
        elif X[i, 2] > 0.0:
            labs.append(W67_BRANCH_MERGE_ROUTING_LABELS[2])
        else:
            labs.append(W67_BRANCH_MERGE_ROUTING_LABELS[3])
    ctrl, rep2 = fit_replay_v8_branch_merge_routing_head(
        controller=ctrl, train_team_features=X.tolist(),
        train_routing_labels=labs)
    assert rep2.branch_merge_routing_trained


def test_masc_v3_five_regimes_wins():
    from coordpy.multi_agent_substrate_coordinator_v2 import (
        W66_MASC_V2_REGIME_BASELINE,
        W66_MASC_V2_REGIME_TEAM_CONSENSUS_UNDER_BUDGET,
        W66_MASC_V2_REGIME_TEAM_FAILURE_RECOVERY,
    )
    from coordpy.multi_agent_substrate_coordinator_v3 import (
        MultiAgentSubstrateCoordinatorV3,
        W67_MASC_V3_REGIME_BRANCH_MERGE_RECONCILIATION,
        W67_MASC_V3_REGIME_ROLE_DROPOUT,
    )
    masc = MultiAgentSubstrateCoordinatorV3()
    for regime in (
            W66_MASC_V2_REGIME_BASELINE,
            W66_MASC_V2_REGIME_TEAM_CONSENSUS_UNDER_BUDGET,
            W67_MASC_V3_REGIME_ROLE_DROPOUT,
            W67_MASC_V3_REGIME_BRANCH_MERGE_RECONCILIATION):
        _, agg = masc.run_batch(
            seeds=list(range(15)), regime=regime)
        assert agg.v12_beats_v11_rate >= 0.5, (
            f"v12 beats v11 below floor in {regime!r}: "
            f"{agg.v12_beats_v11_rate}")
    # TSC_V12 beats TSC_V11 only need to hold in tfr / baseline /
    # tcub — role_dropout already has TSC_V11 at ceiling.
    _, agg_tfr = masc.run_batch(
        seeds=list(range(15)),
        regime=W66_MASC_V2_REGIME_TEAM_FAILURE_RECOVERY)
    assert agg_tfr.v12_beats_v11_rate >= 0.5 or (
        agg_tfr.per_policy_success_rate.get(
            "substrate_routed_v12", 0.0)
        > agg_tfr.per_policy_success_rate.get(
            "substrate_routed_v11", 0.0))


def test_team_consensus_controller_v2_branch_merge_and_role_dropout():
    from coordpy.team_consensus_controller_v2 import (
        TeamConsensusControllerV2,
        W67_TC_V2_DECISION_BRANCH_MERGE_RECONCILIATION,
        W67_TC_V2_DECISION_ROLE_DROPOUT_REPAIR,
    )
    tcc2 = TeamConsensusControllerV2()
    bm = tcc2.decide_v2(
        regime="branch_merge_reconciliation",
        agent_guesses=[0.5, 0.6, 0.55],
        agent_confidences=[0.7, 0.8, 0.65],
        substrate_replay_trust=0.7,
        branch_payloads_per_branch={
            "b0": [0.5, 0.6], "b1": [0.45, 0.55]},
        branch_trusts={"b0": 0.6, "b1": 0.4})
    assert (
        bm["decision"]
        == W67_TC_V2_DECISION_BRANCH_MERGE_RECONCILIATION)
    rd = tcc2.decide_v2(
        regime="role_dropout",
        agent_guesses=[0.5, 0.6, 0.55],
        agent_confidences=[0.7, 0.8, 0.65],
        substrate_replay_trust=0.7,
        missing_role_indices=[0])
    assert (
        rd["decision"]
        == W67_TC_V2_DECISION_ROLE_DROPOUT_REPAIR)


def test_consensus_v13_twenty_stages():
    from coordpy.consensus_fallback_controller_v13 import (
        ConsensusFallbackControllerV13,
        W67_CONSENSUS_V13_STAGES,
        W67_CONSENSUS_V13_STAGE_BRANCH_MERGE_RECONCILIATION_ARBITER,
        W67_CONSENSUS_V13_STAGE_ROLE_DROPOUT_ARBITER,
    )
    assert len(W67_CONSENSUS_V13_STAGES) == 20
    assert (
        W67_CONSENSUS_V13_STAGE_ROLE_DROPOUT_ARBITER
        in W67_CONSENSUS_V13_STAGES)
    assert (
        W67_CONSENSUS_V13_STAGE_BRANCH_MERGE_RECONCILIATION_ARBITER
        in W67_CONSENSUS_V13_STAGES)
    ctrl = ConsensusFallbackControllerV13.init()
    res = ctrl.decide_v13(
        payloads=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        trusts=[0.2, 0.2],
        replay_decisions=["choose_abstain", "choose_abstain"],
        transcript_available=False,
        role_dropout_scores_per_parent=[0.6, 0.5],
        team_failure_recovery_scores_per_parent=[0.1, 0.1],
        team_consensus_under_budget_scores_per_parent=[0.1, 0.1],
        visible_token_budget_frac=0.3,
        team_substrate_coordination_scores_per_parent=[0.1, 0.1],
        multi_agent_abstain_score=0.4,
        corruption_detected_per_parent=[False, False],
        repair_amount=0.0,
        hidden_wins_margins_per_parent=[0.0, 0.0],
        three_way_predictions_per_parent=["kv_wins", "kv_wins"],
        replay_dominance_primary_scores_per_parent=[0.0, 0.0],
        four_way_predictions_per_parent=[
            "kv_wins", "kv_wins"])
    assert (
        res.get("stage")
        == W67_CONSENSUS_V13_STAGE_ROLE_DROPOUT_ARBITER)


def test_crc_v15_kv32768_and_35bit():
    from coordpy.corruption_robust_carrier_v15 import (
        CorruptionRobustCarrierV15,
        emit_corruption_robustness_v15_witness,
    )
    crc = CorruptionRobustCarrierV15()
    w = emit_corruption_robustness_v15_witness(
        crc_v15=crc, n_probes=32, seed=67510)
    assert w.kv32768_corruption_detect_rate >= 0.95
    assert w.adversarial_35bit_burst_detect_rate >= 0.4


def test_ecc_v19_bits_per_token():
    from coordpy.ecc_codebook_v19 import (
        ECCCodebookV19,
        compress_carrier_ecc_v19,
    )
    from coordpy.ecc_codebook_v5 import (
        W53_DEFAULT_ECC_CODE_DIM,
        W53_DEFAULT_ECC_EMIT_MASK_LEN,
    )
    from coordpy.quantised_compression import QuantisedBudgetGate
    book = ECCCodebookV19.init(seed=67500)
    assert book.total_codes == 2 ** 31
    gate = QuantisedBudgetGate.init(
        in_dim=W53_DEFAULT_ECC_CODE_DIM,
        emit_mask_len=W53_DEFAULT_ECC_EMIT_MASK_LEN,
        seed=67501)
    gate.importance_threshold = 0.0
    gate.w_emit.values = [1.0] * len(gate.w_emit.values)
    carrier = [0.1] * W53_DEFAULT_ECC_CODE_DIM
    comp = compress_carrier_ecc_v19(
        carrier, codebook=book, gate=gate)
    assert comp["bits_per_visible_token"] >= 33.0


def test_persistent_v19_sixteenth_skip_and_chain_walk():
    from coordpy.persistent_latent_v12 import V12StackedCell
    from coordpy.persistent_latent_v19 import (
        PersistentLatentStateV19Chain,
        W67_DEFAULT_V19_MAX_CHAIN_WALK_DEPTH,
        step_persistent_state_v19,
    )
    assert W67_DEFAULT_V19_MAX_CHAIN_WALK_DEPTH >= 16384
    cell = V12StackedCell.init(seed=67520)
    chain = PersistentLatentStateV19Chain.empty()
    state = step_persistent_state_v19(
        cell=cell, prev_state=None,
        carrier_values=[0.1] * int(cell.state_dim),
        turn_index=0, role="r",
        role_dropout_recovery_skip_v19=[0.5] * int(
            cell.state_dim))
    chain.add(state)
    assert sum(
        abs(float(v))
        for v in state.role_dropout_recovery_carrier) > 0.0


def test_deep_substrate_hybrid_v12_twelve_way():
    from coordpy.deep_substrate_hybrid_v11 import (
        DeepSubstrateHybridV11ForwardWitness,
    )
    from coordpy.deep_substrate_hybrid_v12 import (
        DeepSubstrateHybridV12,
        deep_substrate_hybrid_v12_forward,
    )
    hybrid = DeepSubstrateHybridV12(inner_v11=None)
    v11_w = DeepSubstrateHybridV11ForwardWitness(
        schema="x", hybrid_cid="x", inner_v10_witness_cid="x",
        eleven_way=True,
        cache_controller_v9_fired=True,
        replay_controller_v7_fired=True,
        replay_trust_active=True,
        team_failure_recovery_active=True,
        substrate_snapshot_diff_active=True,
        team_consensus_controller_active=True,
        replay_trust_l1=1.0,
        team_failure_recovery_count=1,
        n_team_consensus_invocations=1)
    from coordpy.cache_controller_v10 import (
        CacheControllerV10, fit_seven_objective_ridge_v10,
    )
    from coordpy.replay_controller_v8 import (
        ReplayControllerV8,
        W67_BRANCH_MERGE_ROUTING_LABELS,
        fit_replay_v8_branch_merge_routing_head,
    )
    rng = np.random.default_rng(0)
    cc = CacheControllerV10.init()
    X = rng.standard_normal((10, 4))
    cc, _ = fit_seven_objective_ridge_v10(
        controller=cc, train_features=X.tolist(),
        target_drop_oracle=X.sum(axis=-1).tolist(),
        target_retrieval_relevance=X[:, 0].tolist(),
        target_hidden_wins=X[:, 1].tolist(),
        target_replay_dominance=X[:, 2].tolist(),
        target_team_task_success=X[:, 3].tolist(),
        target_team_failure_recovery=X[:, 0].tolist(),
        target_branch_merge=X[:, 1].tolist())
    rcv8 = ReplayControllerV8.init()
    X2 = rng.standard_normal((30, 10))
    labs = [W67_BRANCH_MERGE_ROUTING_LABELS[i % 4]
            for i in range(30)]
    # Need the per_role_per_regime_heads_v8 populated for fired
    # check.
    from coordpy.replay_controller import ReplayCandidate
    from coordpy.replay_controller_v8 import (
        W67_REPLAY_REGIMES_V8,
        fit_replay_controller_v8_per_role,
    )
    cands = {
        r: [ReplayCandidate(100, 1000, 50, 0.1, 0.0, 0.3,
                            True, True, 0)]
        for r in W67_REPLAY_REGIMES_V8}
    decs = {r: ["choose_reuse"] for r in W67_REPLAY_REGIMES_V8}
    rcv8, _ = fit_replay_controller_v8_per_role(
        controller=rcv8, role="planner",
        train_candidates_per_regime=cands,
        train_decisions_per_regime=decs)
    rcv8, _ = fit_replay_v8_branch_merge_routing_head(
        controller=rcv8, train_team_features=X2.tolist(),
        train_routing_labels=labs)
    wh = deep_substrate_hybrid_v12_forward(
        hybrid=hybrid, v11_witness=v11_w,
        cache_controller_v10=cc, replay_controller_v8=rcv8,
        branch_merge_witness_l1=1.0,
        role_dropout_recovery_count=1,
        n_branches_active=2,
        n_team_consensus_v2_invocations=1)
    assert wh.twelve_way


def test_substrate_adapter_v12_full_tier():
    from coordpy.substrate_adapter_v12 import (
        W67_SUBSTRATE_TIER_SUBSTRATE_V12_FULL,
        probe_all_v12_adapters,
        probe_tiny_substrate_v12_adapter,
    )
    cap = probe_tiny_substrate_v12_adapter()
    assert cap.tier == W67_SUBSTRATE_TIER_SUBSTRATE_V12_FULL
    matrix = probe_all_v12_adapters()
    assert matrix.has_v12_full()
