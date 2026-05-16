"""W66 module focused tests (post-W65 milestone).

Per-module sanity tests for the W66 line. R-146/R-147/R-148 suites
exercise the H-bars end-to-end.
"""

from __future__ import annotations

import numpy as np


def test_tiny_substrate_v11_determinism_and_axes():
    from coordpy.tiny_substrate_v11 import (
        build_default_tiny_substrate_v11,
        forward_tiny_substrate_v11,
        emit_tiny_substrate_v11_forward_witness,
        record_replay_trust_v11,
        trigger_team_failure_recovery_v11,
        substrate_snapshot_diff_v11,
        substrate_snapshot_recover_flops_v11,
        tokenize_bytes_v11,
    )
    p = build_default_tiny_substrate_v11(seed=661)
    ids = tokenize_bytes_v11("w66-test", max_len=12)
    t1, c1 = forward_tiny_substrate_v11(p, ids)
    t2, _ = forward_tiny_substrate_v11(p, ids)
    assert t1.cid() == t2.cid()
    assert p.config.v10.v9.n_layers == 13
    assert c1.replay_trust_ledger.shape == (
        p.config.v10.v9.n_layers, p.config.v10.v9.n_heads,
        p.config.v10.v9.max_len)
    assert t1.v11_gate_score_per_layer.shape == (
        p.config.v10.v9.n_layers,)
    record_replay_trust_v11(
        c1, layer_index=0, head_index=0, slot=0, trust=0.7)
    assert c1.replay_trust_ledger[0, 0, 0] == 0.7
    before = c1.clone()
    trigger_team_failure_recovery_v11(
        c1, role="planner", reason="test")
    diff = substrate_snapshot_diff_v11(before, c1)
    assert "team_failure_recovery_flag" in diff.changed_axes
    saving = substrate_snapshot_recover_flops_v11(n_tokens=128)
    assert saving["saving_ratio"] >= 0.6
    w = emit_tiny_substrate_v11_forward_witness(t1, c1)
    assert w.team_failure_recovery_count == 1


def test_kv_bridge_v11_seven_target_and_falsifier():
    from coordpy.kv_bridge_v3 import KVBridgeV3Projection
    from coordpy.kv_bridge_v4 import KVBridgeV4Projection
    from coordpy.kv_bridge_v5 import KVBridgeV5Projection
    from coordpy.kv_bridge_v6 import KVBridgeV6Projection
    from coordpy.kv_bridge_v7 import KVBridgeV7Projection
    from coordpy.kv_bridge_v8 import KVBridgeV8Projection
    from coordpy.kv_bridge_v9 import KVBridgeV9Projection
    from coordpy.kv_bridge_v10 import KVBridgeV10Projection
    from coordpy.kv_bridge_v11 import (
        KVBridgeV11Projection,
        compute_multi_agent_task_fingerprint_v11,
        fit_kv_bridge_v11_seven_target,
        probe_kv_bridge_v11_team_failure_recovery_falsifier,
    )
    from coordpy.tiny_substrate_v11 import (
        build_default_tiny_substrate_v11, tokenize_bytes_v11,
    )
    p = build_default_tiny_substrate_v11(seed=662)
    cfg = p.config.v10.v9
    d_head = cfg.d_model // cfg.n_heads
    proj = KVBridgeV11Projection.init_from_v10(
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
                                        seed=66200),
                                    seed_v4=66201),
                                seed_v5=66202),
                            seed_v6=66203),
                        seed_v7=66204),
                    seed_v8=66205),
                seed_v9=66206),
            seed_v10=66207),
        seed_v11=66208)
    rng = np.random.default_rng(0)
    ids = tokenize_bytes_v11("kv11", max_len=4)
    while len(ids) < 4:
        ids.append(0)
    carriers = [list(rng.standard_normal(8)) for _ in range(4)]
    targets = []
    for i in range(7):
        t = np.zeros(cfg.vocab_size)
        t[100 + 20 * i] = 1.0 / (i + 1)
        targets.append(t.tolist())
    _, report = fit_kv_bridge_v11_seven_target(
        params=p, projection=proj,
        train_carriers=carriers,
        target_delta_logits_stack=targets,
        follow_up_token_ids=ids, n_directions=3)
    assert report.n_targets == 7
    assert (
        report.team_failure_recovery_post
        <= report.team_failure_recovery_pre + 1e-3)
    fp = compute_multi_agent_task_fingerprint_v11(
        role="planner", task_id="t", team_id="x")
    assert len(fp) == 30
    fals = probe_kv_bridge_v11_team_failure_recovery_falsifier(
        team_failure_recovery_flag=0.5)
    assert fals.falsifier_score == 0.0


def test_hsb_v10_seven_target_and_probe():
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
        fit_hsb_v10_seven_target,
        probe_hsb_v10_hidden_wins_vs_team_success,
    )
    from coordpy.tiny_substrate_v11 import (
        build_default_tiny_substrate_v11,
        tokenize_bytes_v11,
    )
    p = build_default_tiny_substrate_v11(seed=663)
    cfg = p.config.v10.v9
    hsb2 = HiddenStateBridgeV2Projection.init(
        target_layers=(1, 3), n_tokens=6, carrier_dim=6,
        d_model=cfg.d_model, seed=66301)
    hsb10 = HiddenStateBridgeV10Projection.init_from_v9(
        HiddenStateBridgeV9Projection.init_from_v8(
            HiddenStateBridgeV8Projection.init_from_v7(
                HiddenStateBridgeV7Projection.init_from_v6(
                    HiddenStateBridgeV6Projection.init_from_v5(
                        HiddenStateBridgeV5Projection.init_from_v4(
                            HiddenStateBridgeV4Projection.init_from_v3(
                                HiddenStateBridgeV3Projection.init_from_v2(
                                    hsb2,
                                    n_heads=cfg.n_heads,
                                    seed_v3=66302),
                                seed_v4=66303),
                            n_positions=3, seed_v5=66304),
                        seed_v6=66305),
                    seed_v7=66306),
                seed_v8=66307),
            seed_v9=66308),
        seed_v10=66309)
    rng = np.random.default_rng(0)
    carriers = rng.standard_normal((6, 6)).tolist()
    targets = [
        rng.standard_normal(cfg.vocab_size).tolist()
        for _ in range(7)]
    ids = tokenize_bytes_v11("hsb10", max_len=8)
    _, report = fit_hsb_v10_seven_target(
        params=p.v3_params, projection=hsb10,
        train_carriers=carriers,
        target_delta_logits_stack=targets,
        token_ids=ids)
    assert report.n_targets == 7
    assert report.converged
    res = probe_hsb_v10_hidden_wins_vs_team_success(
        n_layers=2, n_heads=2,
        hidden_residual_l2_per_lh=[[0.2, 0.3], [0.1, 0.4]],
        team_success_indicator_per_lh=[
            [1.0, 0.0], [1.0, 0.0]])
    assert 0.0 <= res["mean_joint_win_rate"] <= 1.0


def test_cache_controller_v9_six_objective():
    from coordpy.cache_controller_v9 import (
        CacheControllerV9, fit_six_objective_ridge_v9,
        fit_per_role_eviction_head_v9,
    )
    cc = CacheControllerV9.init(fit_seed=664)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 4))
    cc, report = fit_six_objective_ridge_v9(
        controller=cc, train_features=X.tolist(),
        target_drop_oracle=X.sum(axis=-1).tolist(),
        target_retrieval_relevance=X[:, 0].tolist(),
        target_hidden_wins=(X[:, 1] - X[:, 2]).tolist(),
        target_replay_dominance=(X[:, 3] * 0.5).tolist(),
        target_team_task_success=(
            X[:, 0] * 0.3 - X[:, 1] * 0.1).tolist(),
        target_team_failure_recovery=(
            X[:, 2] * 0.4 + X[:, 3] * 0.2).tolist())
    assert report.n_objectives == 6
    assert report.converged
    Y = rng.standard_normal((12, 7))
    y = Y[:, 0] * 0.4 + Y[:, 6] * 0.3
    cc, r2 = fit_per_role_eviction_head_v9(
        controller=cc, role="planner",
        train_features=Y.tolist(),
        target_eviction_priorities=y.tolist())
    assert r2.converged


def test_replay_controller_v7_regimes_and_routing():
    from coordpy.replay_controller import ReplayCandidate
    from coordpy.replay_controller_v5 import ReplayControllerV5
    from coordpy.replay_controller_v6 import ReplayControllerV6
    from coordpy.replay_controller_v7 import (
        ReplayControllerV7, W66_REPLAY_REGIMES_V7,
        W66_TEAM_SUBSTRATE_ROUTING_LABELS,
        fit_replay_controller_v7_per_role,
        fit_replay_v7_team_substrate_routing_head,
    )
    rcv5 = ReplayControllerV5.init()
    rcv6 = ReplayControllerV6.init(inner_v5=rcv5)
    ctrl = ReplayControllerV7.init(inner_v6=rcv6)
    cands = {
        r: [ReplayCandidate(
            100, 1000, 50, 0.1, 0.0, 0.3,
            True, True, 0)]
        for r in W66_REPLAY_REGIMES_V7}
    decs = {
        r: ["choose_reuse"]
        for r in W66_REPLAY_REGIMES_V7}
    ctrl, report = fit_replay_controller_v7_per_role(
        controller=ctrl, role="planner",
        train_candidates_per_regime=cands,
        train_decisions_per_regime=decs)
    assert report.converged
    assert report.n_role_regime_heads_v7 == 10
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 10)).tolist()
    labs = [W66_TEAM_SUBSTRATE_ROUTING_LABELS[i % 4]
             for i in range(30)]
    ctrl, r2 = fit_replay_v7_team_substrate_routing_head(
        controller=ctrl, train_team_features=X,
        train_routing_labels=labs)
    assert r2.team_substrate_routing_trained
    assert r2.converged


def test_masc_v2_per_regime_v11_beats():
    from coordpy.multi_agent_substrate_coordinator_v2 import (
        MultiAgentSubstrateCoordinatorV2,
        W66_MASC_V2_REGIMES,
    )
    c = MultiAgentSubstrateCoordinatorV2()
    aggs = c.run_all_regimes(seeds=list(range(15)))
    assert set(aggs.keys()) == set(W66_MASC_V2_REGIMES)
    # Baseline: V11 beats V10 majority.
    assert aggs["baseline"].v11_beats_v10_rate >= 0.5
    # Baseline: TSC_V11 beats V11 majority.
    assert aggs["baseline"].tsc_v11_beats_v11_rate >= 0.5
    # Failure recovery: TSC_V11 beats V11 majority.
    assert (aggs[
        "team_failure_recovery"
        ].tsc_v11_beats_v11_rate >= 0.5)


def test_team_consensus_controller_three_regimes():
    from coordpy.team_consensus_controller import (
        TeamConsensusController, W66_TC_DECISION_QUORUM,
        W66_TC_DECISION_SUBSTRATE_REPLAY,
        W66_TC_DECISION_ABSTAIN,
    )
    from coordpy.multi_agent_substrate_coordinator_v2 import (
        W66_MASC_V2_REGIME_BASELINE,
        W66_MASC_V2_REGIME_TEAM_CONSENSUS_UNDER_BUDGET,
        W66_MASC_V2_REGIME_TEAM_FAILURE_RECOVERY,
    )
    tc = TeamConsensusController()
    r = tc.decide(
        regime=W66_MASC_V2_REGIME_BASELINE,
        agent_guesses=[0.5, 0.6, 0.55],
        agent_confidences=[0.7, 0.8, 0.65])
    assert r["decision"] == W66_TC_DECISION_QUORUM
    r = tc.decide(
        regime=W66_MASC_V2_REGIME_TEAM_CONSENSUS_UNDER_BUDGET,
        agent_guesses=[0.5, 0.6, 0.55],
        agent_confidences=[0.1, 0.2, 0.15],
        substrate_replay_trust=0.7)
    assert r["decision"] == W66_TC_DECISION_SUBSTRATE_REPLAY
    r = tc.decide(
        regime=W66_MASC_V2_REGIME_TEAM_FAILURE_RECOVERY,
        agent_guesses=[0.5, 0.6],
        agent_confidences=[0.05, 0.10],
        substrate_replay_trust=0.1,
        transcript_available=False)
    assert r["decision"] == W66_TC_DECISION_ABSTAIN


def test_ecc_v18_bits_per_token_floor():
    from coordpy.ecc_codebook_v18 import (
        ECCCodebookV18, compress_carrier_ecc_v18,
    )
    from coordpy.ecc_codebook_v5 import (
        W53_DEFAULT_ECC_CODE_DIM,
        W53_DEFAULT_ECC_EMIT_MASK_LEN,
    )
    from coordpy.quantised_compression import QuantisedBudgetGate
    cb = ECCCodebookV18.init(seed=665)
    gate = QuantisedBudgetGate.init(
        in_dim=W53_DEFAULT_ECC_CODE_DIM,
        emit_mask_len=W53_DEFAULT_ECC_EMIT_MASK_LEN,
        seed=666)
    gate.importance_threshold = 0.0
    gate.w_emit.values = [1.0] * len(gate.w_emit.values)
    comp = compress_carrier_ecc_v18(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], codebook=cb, gate=gate)
    assert comp["bits_per_visible_token"] >= 31.0
    assert cb.total_codes == 2 ** 29


def test_disagreement_v12_js_identity_and_falsifier():
    from coordpy.disagreement_algebra import AlgebraTrace
    from coordpy.disagreement_algebra_v12 import (
        emit_disagreement_algebra_v12_witness,
        js_divergence_symmetric,
    )
    w = emit_disagreement_algebra_v12_witness(
        trace=AlgebraTrace(steps=[]),
        probe_a=[0.1, 0.2, 0.3],
        probe_b=[0.1, 0.2, 0.3],
        probe_c=[0.1, 0.2, 0.3],
        tv_oracle=lambda: (True, 0.05),
        tv_falsifier_oracle=lambda: (False, 1.0),
        wasserstein_oracle=lambda: (True, 0.1),
        wasserstein_falsifier_oracle=lambda: (False, 1.0),
        js_oracle=lambda: (True, 0.05),
        js_falsifier_oracle=lambda: (False, 1.0),
        attention_pattern_oracle=lambda: (True, 0.85))
    assert w.js_equiv_ok
    assert w.js_falsifier_ok
    # JS divergence is non-negative.
    j = js_divergence_symmetric([1.0, 2.0, 3.0], [1.0, 1.0, 1.0])
    assert j >= 0.0
