"""W63 module focused tests (post-W62 milestone).

Per-module sanity tests for the W63 line. R-134/R-135/R-136 suites
exercise the H-bars end-to-end.
"""

from __future__ import annotations

import numpy as np


def test_tiny_substrate_v8_determinism_and_axes():
    from coordpy.tiny_substrate_v8 import (
        build_default_tiny_substrate_v8,
        forward_tiny_substrate_v8,
        emit_tiny_substrate_v8_forward_witness,
        hidden_vs_kv_contention_summary,
        record_hidden_vs_kv_contention_v8,
        record_prefix_reuse_decision_v8,
        tokenize_bytes_v8,
    )
    p = build_default_tiny_substrate_v8(seed=631)
    ids = tokenize_bytes_v8("w63-test", max_len=12)
    t1, c1 = forward_tiny_substrate_v8(p, ids)
    t2, c2 = forward_tiny_substrate_v8(p, ids)
    assert t1.cid() == t2.cid()
    assert c1.hidden_vs_kv_contention.shape == (
        p.config.n_layers, p.config.n_heads, p.config.max_len)
    assert c1.prefix_reuse_trust.shape == (
        p.config.n_layers, p.config.n_heads)
    assert t1.hidden_state_confidence_per_layer.shape == (
        p.config.n_layers,)
    assert t1.cross_layer_head_coupling.shape == (
        p.config.n_layers, p.config.n_heads,
        p.config.n_layers, p.config.n_heads)
    pre = float(np.abs(c1.hidden_vs_kv_contention).sum())
    record_hidden_vs_kv_contention_v8(
        c1, layer_index=0, head_index=0, slot=0,
        hidden_write_abs=0.7, kv_write_abs=0.3)
    post = float(np.abs(c1.hidden_vs_kv_contention).sum())
    assert post > pre
    pre = float(c1.prefix_reuse_trust[0, 0])
    record_prefix_reuse_decision_v8(
        c1, layer_index=0, head_index=0,
        decision="prefix_reuse_success")
    assert float(c1.prefix_reuse_trust[0, 0]) > pre
    summary = hidden_vs_kv_contention_summary(c1)
    assert "hidden_dominant_l1" in summary
    w = emit_tiny_substrate_v8_forward_witness(t1, c1)
    assert w.n_layers == p.config.n_layers


def test_kv_bridge_v8_four_target():
    from coordpy.kv_bridge_v3 import KVBridgeV3Projection
    from coordpy.kv_bridge_v4 import KVBridgeV4Projection
    from coordpy.kv_bridge_v5 import KVBridgeV5Projection
    from coordpy.kv_bridge_v6 import KVBridgeV6Projection
    from coordpy.kv_bridge_v7 import KVBridgeV7Projection
    from coordpy.kv_bridge_v8 import (
        KVBridgeV8Projection,
        fit_kv_bridge_v8_four_target,
        probe_kv_bridge_v8_hidden_wins_falsifier,
    )
    from coordpy.tiny_substrate_v8 import (
        build_default_tiny_substrate_v8, tokenize_bytes_v8,
    )
    p = build_default_tiny_substrate_v8(seed=632)
    proj_v3 = KVBridgeV3Projection.init(
        n_layers=p.config.n_layers, n_heads=p.config.n_heads,
        n_kv_heads=p.config.n_kv_heads,
        n_inject_tokens=3, carrier_dim=8,
        d_head=p.config.d_model // p.config.n_heads,
        seed=63200)
    proj_v4 = KVBridgeV4Projection.init_from_v3(
        proj_v3, seed_v4=63201)
    proj_v5 = KVBridgeV5Projection.init_from_v4(
        proj_v4, seed_v5=63202)
    proj_v6 = KVBridgeV6Projection.init_from_v5(
        proj_v5, seed_v6=63203)
    proj_v7 = KVBridgeV7Projection.init_from_v6(
        proj_v6, seed_v7=63204)
    proj_v8 = KVBridgeV8Projection.init_from_v7(
        proj_v7, seed_v8=63205)
    rng = np.random.default_rng(0)
    ids = tokenize_bytes_v8("kv8", max_len=4)
    while len(ids) < 4: ids.append(0)
    carriers = [list(rng.standard_normal(8)) for _ in range(4)]
    t1 = np.zeros(p.config.vocab_size); t1[100] = 1.0
    t2 = np.zeros(p.config.vocab_size); t2[120] = 0.5
    t3 = np.zeros(p.config.vocab_size); t3[150] = 0.3
    t4 = np.zeros(p.config.vocab_size); t4[200] = 0.2
    fitted, report = fit_kv_bridge_v8_four_target(
        params=p, projection=proj_v8,
        train_carriers=carriers,
        target_delta_logits_stack=[
            t1.tolist(), t2.tolist(), t3.tolist(), t4.tolist()],
        follow_up_token_ids=ids, n_directions=3)
    assert report.n_targets == 4
    # Falsifier returns 0 under inversion.
    f = probe_kv_bridge_v8_hidden_wins_falsifier(
        hidden_residual_l2=0.3, kv_residual_l2=0.6)
    assert f.falsifier_score == 0.0


def test_replay_controller_v4_six_regimes():
    from coordpy.replay_controller_v4 import (
        ReplayControllerV4,
        W63_REPLAY_REGIMES_V4,
        fit_replay_controller_v4_per_regime,
    )
    from coordpy.replay_controller import (
        ReplayCandidate, W60_REPLAY_DECISION_REUSE,
    )
    ctrl = ReplayControllerV4.init()
    cands = {
        r: [ReplayCandidate(
            100, 1000, 50, 0.1, 0.0, 0.3,
            True, True, 0)]
        for r in W63_REPLAY_REGIMES_V4}
    decs = {
        r: [W60_REPLAY_DECISION_REUSE]
        for r in W63_REPLAY_REGIMES_V4}
    _, report = fit_replay_controller_v4_per_regime(
        controller=ctrl, train_candidates_per_regime=cands,
        train_decisions_per_regime=decs)
    assert report.n_regimes == 6
    assert report.converged


def test_cache_controller_v6_three_objective():
    from coordpy.cache_controller_v6 import (
        CacheControllerV6, fit_three_objective_ridge_v6,
    )
    rng = np.random.default_rng(63300)
    cc = CacheControllerV6.init(fit_seed=63300)
    X = rng.standard_normal((16, 4))
    y1 = X.sum(axis=-1)
    y2 = X[:, 0]
    y3 = X[:, 1] - X[:, 2]
    _, report = fit_three_objective_ridge_v6(
        controller=cc, train_features=X.tolist(),
        target_drop_oracle=y1.tolist(),
        target_retrieval_relevance=y2.tolist(),
        target_hidden_wins=y3.tolist())
    assert report.n_objectives == 3
    assert report.converged


def test_hsb_v7_hidden_wins_margin():
    from coordpy.hidden_state_bridge_v7 import (
        compute_hsb_v7_hidden_wins_margin,
    )
    pos = compute_hsb_v7_hidden_wins_margin(
        hidden_residual_l2=0.3, kv_residual_l2=0.6)
    neg = compute_hsb_v7_hidden_wins_margin(
        hidden_residual_l2=0.6, kv_residual_l2=0.3)
    assert pos > 0.0
    assert neg < 0.0


def test_prefix_v7_vs_hidden_decision():
    from coordpy.prefix_state_bridge_v7 import (
        compare_prefix_vs_hidden_v7,
    )
    w_prefix = compare_prefix_vs_hidden_v7(
        prefix_drift_curve=[0.1, 0.2],
        hidden_drift_curve=[0.5, 0.6])
    assert w_prefix.decision == "prefix_beats_hidden"


def test_attention_v7_three_stage_zero_under_negative_budget():
    from coordpy.attention_steering_bridge_v2 import (
        AttentionSteeringV2Projection,
    )
    from coordpy.attention_steering_bridge_v7 import (
        steer_attention_and_measure_v7,
    )
    from coordpy.tiny_substrate_v8 import (
        build_default_tiny_substrate_v8, tokenize_bytes_v8,
    )
    p = build_default_tiny_substrate_v8(seed=637)
    ids = tokenize_bytes_v8("attn", max_len=4)[:4]
    while len(ids) < 4: ids.append(0)
    proj = AttentionSteeringV2Projection.init(
        n_layers=p.config.n_layers, n_heads=p.config.n_heads,
        n_query=len(ids), n_key=len(ids), carrier_dim=6,
        seed=63700)
    rng = np.random.default_rng(63701)
    carrier = list(rng.standard_normal(6))
    w_neg = steer_attention_and_measure_v7(
        params=p.v3_params, carrier=carrier,
        projection=proj, token_ids=ids,
        js_budget=-1.0)
    assert w_neg.js_max_after_three_stage == 0.0


def test_substrate_adapter_v8_full_tier():
    from coordpy.substrate_adapter_v8 import (
        W63_SUBSTRATE_TIER_SUBSTRATE_V8_FULL,
        probe_tiny_substrate_v8_adapter,
    )
    cap = probe_tiny_substrate_v8_adapter()
    assert cap.tier == W63_SUBSTRATE_TIER_SUBSTRATE_V8_FULL


def test_persistent_v15_twelfth_skip():
    from coordpy.persistent_latent_v12 import V12StackedCell
    from coordpy.persistent_latent_v15 import (
        PersistentLatentStateV15Chain,
        emit_persistent_v15_witness,
        step_persistent_state_v15,
    )
    cell = V12StackedCell.init(seed=639)
    chain = PersistentLatentStateV15Chain.empty()
    carrier = [0.5] * int(cell.state_dim)
    state = step_persistent_state_v15(
        cell=cell, prev_state=None,
        carrier_values=carrier,
        turn_index=0, role="r",
        substrate_skip=carrier,
        hidden_wins_skip=carrier,
        prefix_reuse_skip=carrier)
    chain.add(state)
    w = emit_persistent_v15_witness(chain, state.cid())
    assert w.twelfth_skip_present
    assert w.n_layers == 14


def test_multi_hop_v13_eight_axis_compromise():
    from coordpy.multi_hop_translator_v13 import (
        W63_DEFAULT_MH_V13_BACKENDS,
        evaluate_dec_chain_len19_fidelity,
    )
    res = evaluate_dec_chain_len19_fidelity(seed=63800)
    assert 1 <= res["compromise_threshold"] <= 8
    assert "eight_axis" in res["kind"]


def test_consensus_v9_thirteen_stages_fires():
    from coordpy.consensus_fallback_controller_v9 import (
        ConsensusFallbackControllerV9,
        W63_CONSENSUS_V9_STAGE_HIDDEN_WINS_ARBITER,
        W63_CONSENSUS_V9_STAGES,
    )
    assert len(W63_CONSENSUS_V9_STAGES) == 13
    ctrl = ConsensusFallbackControllerV9.init()
    res = ctrl.decide_v9(
        payloads=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        trusts=[0.4, 0.4],
        replay_decisions=["choose_abstain", "choose_abstain"],
        transcript_available=False,
        corruption_detected_per_parent=[False, False],
        repair_amount=0.0,
        hidden_wins_margins_per_parent=[0.1, 0.0],
        three_way_predictions_per_parent=[
            "hidden_wins", "kv_wins"])
    assert (res["stage"]
            == W63_CONSENSUS_V9_STAGE_HIDDEN_WINS_ARBITER)


def test_corruption_v11_witness():
    from coordpy.corruption_robust_carrier_v11 import (
        CorruptionRobustCarrierV11,
        emit_corruption_robustness_v11_witness,
    )
    crc = CorruptionRobustCarrierV11()
    w = emit_corruption_robustness_v11_witness(
        crc_v11=crc, n_probes=8, seed=63900)
    assert w.kv2048_corruption_detect_rate >= 0.5
    assert 0.0 <= w.hidden_state_recovery_ratio_mean <= 1.0


def test_lhr_v15_fourteen_way():
    from coordpy.long_horizon_retention_v15 import (
        LongHorizonReconstructionV15Head,
        emit_lhr_v15_witness,
    )
    head = LongHorizonReconstructionV15Head.init(seed=640)
    w = emit_lhr_v15_witness(
        head, carrier=[0.1] * 8, k=4,
        replay_dominance_indicator=[1.0] * 8,
        hidden_wins_indicator=[0.5] * 8)
    assert w.fourteen_way_runs
    assert w.max_k == 160


def test_ecc_v15_total_codes_and_rate_floor():
    from coordpy.ecc_codebook_v15 import (
        ECCCodebookV15,
        probe_ecc_v15_rate_floor_falsifier,
    )
    book = ECCCodebookV15.init(seed=641)
    assert book.total_codes == (1 << 24)
    f = probe_ecc_v15_rate_floor_falsifier(codebook=book)
    assert f["target_exceeds_ceiling"]


def test_tvs_v12_thirteen_arms_sum_to_one():
    from coordpy.transcript_vs_shared_arbiter_v12 import (
        W63_TVS_V12_ARMS, thirteen_arm_compare,
    )
    res = thirteen_arm_compare(
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
    assert len(W63_TVS_V12_ARMS) == 13


def test_uncertainty_v11_ten_axis():
    from coordpy.uncertainty_layer_v11 import (
        compose_uncertainty_report_v11,
    )
    res = compose_uncertainty_report_v11(
        confidences=[0.7], trusts=[0.9],
        substrate_fidelities=[0.95],
        hidden_state_fidelities=[0.95],
        cache_reuse_fidelities=[0.95],
        retrieval_fidelities=[0.95],
        replay_fidelities=[0.92],
        hidden_wins_fidelities=[0.5])
    assert res.n_axes == 10
    assert res.hidden_wins_aware


def test_disagreement_v9_js_identity():
    from coordpy.disagreement_algebra_v9 import (
        check_js_equivalence_identity,
        js_equivalence_falsifier,
    )
    ok = check_js_equivalence_identity(
        js_oracle=lambda: (True, 0.1), js_floor=0.30)
    assert ok
    fals = js_equivalence_falsifier(
        js_oracle=lambda: (False, 1.0), js_floor=0.30)
    assert fals


def test_mlsc_v11_hidden_wins_union():
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
        MergeOperatorV11, wrap_v10_as_v11,
    )
    op = MergeOperatorV11(factor_dim=6)
    def mk(branch, hw):
        v3 = make_root_capsule_v3(
            branch_id=branch, payload=tuple([0.1] * 6),
            fact_tags=("w63",), confidence=0.9, trust=0.9,
            turn_index=0)
        v4 = wrap_v3_as_v4(v3)
        v5 = wrap_v4_as_v5(v4, attention_witness_cid="a")
        v6 = wrap_v5_as_v6(
            v5, attention_witness_chain=("a_chain",),
            cache_reuse_witness_cid="c")
        v7 = wrap_v6_as_v7(
            v6, retrieval_witness_chain=("r_chain",),
            controller_witness_cid="ctrl")
        v8 = wrap_v7_as_v8(
            v7, replay_witness_chain=("rc",),
            substrate_witness_chain=("sc",),
            provenance_trust_table={"a": 0.9})
        v9 = wrap_v8_as_v9(
            v8,
            attention_pattern_witness_chain=("ap",),
            cache_retrieval_witness_chain=("cr",),
            per_layer_head_trust_matrix=((0, 0, 0.9),))
        v10 = wrap_v9_as_v10(
            v9, replay_dominance_witness_chain=("rd",),
            disagreement_wasserstein_distance=0.05)
        return wrap_v10_as_v11(
            v10, hidden_wins_witness_chain=(hw,))
    a = mk("A", "hw_a")
    b = mk("B", "hw_b")
    merged = op.merge([a, b])
    assert "hw_a" in merged.hidden_wins_witness_chain
    assert "hw_b" in merged.hidden_wins_witness_chain
