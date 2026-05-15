"""W62 module focused tests (post-W61 milestone).

Per-module sanity tests for the W62 line. R-131/R-132/R-133 suites
exercise the H-bars end-to-end.
"""

from __future__ import annotations

import numpy as np


def test_tiny_substrate_v7_determinism_and_axes():
    from coordpy.tiny_substrate_v7 import (
        build_default_tiny_substrate_v7,
        forward_tiny_substrate_v7,
        emit_tiny_substrate_v7_forward_witness,
        logit_lens_entropy_per_layer,
        record_cache_write_v7,
        record_replay_decision_v7,
        tokenize_bytes_v7,
    )
    p = build_default_tiny_substrate_v7(seed=621)
    ids = tokenize_bytes_v7("w62-test", max_len=12)
    t1, c1 = forward_tiny_substrate_v7(p, ids)
    t2, c2 = forward_tiny_substrate_v7(p, ids)
    assert t1.cid() == t2.cid()
    # Cache-write ledger axis.
    assert c1.cache_write_ledger.shape == (
        p.config.n_layers, p.config.n_heads, p.config.max_len)
    # Replay-trust ledger.
    assert c1.replay_trust_ledger.shape == (
        p.config.n_layers, p.config.n_heads)
    # Logit-lens probe.
    assert t1.logit_lens_per_layer.shape == (
        p.config.n_layers, p.config.vocab_size)
    # Attention-receive delta.
    assert len(t1.attention_receive_delta_per_layer) == (
        p.config.n_layers)
    # Cache write recorded.
    pre = float(c1.cache_write_ledger.sum())
    record_cache_write_v7(
        c1, layer_index=2, head_index=3, slot=0, l2=0.5)
    assert float(c1.cache_write_ledger.sum()) > pre
    # Replay decision recorded.
    pre = float(c1.replay_trust_ledger[0, 0])
    record_replay_decision_v7(
        c1, layer_index=0, head_index=0,
        decision="choose_reuse")
    assert float(c1.replay_trust_ledger[0, 0]) > pre
    # Logit-lens entropy.
    ent = logit_lens_entropy_per_layer(t1)
    assert ent.shape == (p.config.n_layers,)
    # Witness.
    w = emit_tiny_substrate_v7_forward_witness(t1, c1)
    assert w.n_layers == p.config.n_layers


def test_kv_bridge_v7_three_target():
    from coordpy.kv_bridge_v3 import KVBridgeV3Projection
    from coordpy.kv_bridge_v4 import KVBridgeV4Projection
    from coordpy.kv_bridge_v5 import KVBridgeV5Projection
    from coordpy.kv_bridge_v6 import KVBridgeV6Projection
    from coordpy.kv_bridge_v7 import (
        KVBridgeV7Projection,
        compare_hidden_vs_kv_v7,
        fit_kv_bridge_v7_three_target,
        write_kv_bridge_v7_into_v7_cache_ledger,
    )
    from coordpy.tiny_substrate_v7 import (
        build_default_tiny_substrate_v7,
        forward_tiny_substrate_v7, tokenize_bytes_v7,
    )
    p = build_default_tiny_substrate_v7(seed=622)
    proj_v3 = KVBridgeV3Projection.init(
        n_layers=p.config.n_layers, n_heads=p.config.n_heads,
        n_kv_heads=p.config.n_kv_heads,
        n_inject_tokens=3, carrier_dim=8,
        d_head=p.config.d_model // p.config.n_heads,
        seed=62200)
    proj_v4 = KVBridgeV4Projection.init_from_v3(
        proj_v3, seed_v4=62201)
    proj_v5 = KVBridgeV5Projection.init_from_v4(
        proj_v4, seed_v5=62202)
    proj_v6 = KVBridgeV6Projection.init_from_v5(
        proj_v5, seed_v6=62203)
    proj_v7 = KVBridgeV7Projection.init_from_v6(
        proj_v6, seed_v7=62204)
    rng = np.random.default_rng(0)
    ids = tokenize_bytes_v7("kv7", max_len=4)
    carriers = [list(rng.standard_normal(8)) for _ in range(4)]
    t1 = np.zeros(p.config.vocab_size); t1[100] = 1.0
    t2 = np.zeros(p.config.vocab_size); t2[120] = 0.5
    t3 = np.zeros(p.config.vocab_size); t3[150] = 0.3
    fitted, report = fit_kv_bridge_v7_three_target(
        params=p, projection=proj_v7,
        train_carriers=carriers,
        target_delta_logits_stack=[
            t1.tolist(), t2.tolist(), t3.tolist()],
        follow_up_token_ids=ids, n_directions=3)
    assert report.n_targets == 3
    _, v7_cache = forward_tiny_substrate_v7(p, ids)
    info = write_kv_bridge_v7_into_v7_cache_ledger(
        projection=fitted, v7_cache=v7_cache)
    assert info["total_l2"] > 0.0
    dec = compare_hidden_vs_kv_v7(
        hidden_residual_l2=0.5, kv_residual_l2=0.8)
    assert dec.decision == "hidden_beats_kv"


def test_replay_controller_v3_per_regime_and_hidden_vs_kv():
    from coordpy.replay_controller import (
        ReplayCandidate,
        W60_REPLAY_DECISION_REUSE,
        W60_REPLAY_DECISION_RECOMPUTE,
        W60_REPLAY_DECISION_FALLBACK,
    )
    from coordpy.replay_controller_v2 import (
        ReplayControllerV2, fit_replay_controller_v2,
    )
    from coordpy.replay_controller_v3 import (
        ReplayControllerV3,
        W62_REPLAY_REGIME_CRC_PASSED_LOW_DRIFT,
        W62_REPLAY_REGIME_SYNTHETIC_CORRUPTION,
        W62_REPLAY_REGIME_HIDDEN_WRITE_HEAVY,
        W62_REPLAY_REGIME_TRANSCRIPT_ONLY,
        fit_hidden_vs_kv_regime_classifier,
        fit_replay_controller_v3_per_regime,
        predict_hidden_vs_kv_regime,
    )
    rcv2 = ReplayControllerV2.init()
    train = [
        ReplayCandidate(100, 1000, 50, 0.1, 0.0, 0.3,
                        True, True, 0),
        ReplayCandidate(900, 1000, 50, 0.8, 0.0, 0.3,
                        False, True, 3),
    ]
    rcv2, _ = fit_replay_controller_v2(
        controller=rcv2, train_candidates=train,
        train_optimal_decisions=[
            W60_REPLAY_DECISION_REUSE,
            W60_REPLAY_DECISION_RECOMPUTE])
    rcv3 = ReplayControllerV3.init(inner_v2=rcv2)
    regimes_cands = {
        W62_REPLAY_REGIME_SYNTHETIC_CORRUPTION: [
            ReplayCandidate(900, 1000, 50, 0.8, 0.0, 0.3,
                            False, True, 3),
        ],
        W62_REPLAY_REGIME_CRC_PASSED_LOW_DRIFT: [
            ReplayCandidate(100, 1000, 50, 0.1, 0.0, 0.3,
                            True, True, 0),
        ],
        W62_REPLAY_REGIME_HIDDEN_WRITE_HEAVY: [
            ReplayCandidate(300, 1000, 50, 0.4, 0.0, 0.3,
                            True, True, 0),
        ],
        W62_REPLAY_REGIME_TRANSCRIPT_ONLY: [
            ReplayCandidate(0, 1000, 50, 1.0, 0.0, 0.3,
                            False, False, 5),
        ],
    }
    regimes_decisions = {
        W62_REPLAY_REGIME_SYNTHETIC_CORRUPTION: [
            W60_REPLAY_DECISION_RECOMPUTE],
        W62_REPLAY_REGIME_CRC_PASSED_LOW_DRIFT: [
            W60_REPLAY_DECISION_REUSE],
        W62_REPLAY_REGIME_HIDDEN_WRITE_HEAVY: [
            W60_REPLAY_DECISION_RECOMPUTE],
        W62_REPLAY_REGIME_TRANSCRIPT_ONLY: [
            W60_REPLAY_DECISION_FALLBACK],
    }
    rcv3, report = fit_replay_controller_v3_per_regime(
        controller=rcv3,
        train_candidates_per_regime=regimes_cands,
        train_decisions_per_regime=regimes_decisions)
    assert report.converged
    assert report.n_regimes == 4
    # Decide.
    cand = ReplayCandidate(
        100, 1000, 50, 0.1, 0.0, 0.3, True, True, 0)
    dec, conf, dom, regime = rcv3.decide(cand)
    assert regime == W62_REPLAY_REGIME_CRC_PASSED_LOW_DRIFT
    # Hidden-vs-KV classifier.
    rng = np.random.default_rng(0)
    n = 60
    feats = rng.standard_normal((n, 5))
    feats[:, 0] = np.sign(feats[:, 0]) * (
        np.abs(feats[:, 0]) + 0.5)
    labs = []
    for i in range(n):
        if feats[i, 0] > 0.5:
            labs.append("hidden_beats_kv")
        elif feats[i, 0] < -0.5:
            labs.append("kv_beats_hidden")
        else:
            labs.append("tie")
    rcv3, audit = fit_hidden_vs_kv_regime_classifier(
        controller=rcv3,
        train_features=feats.tolist(), train_labels=labs)
    assert audit["accuracy_train"] >= 0.8
    assert predict_hidden_vs_kv_regime(
        rcv3, [1.0, 0, 0, 0, 0]) == "hidden_beats_kv"
    assert predict_hidden_vs_kv_regime(
        rcv3, [-1.0, 0, 0, 0, 0]) == "kv_beats_hidden"


def test_cache_controller_v5_two_objective_and_repair():
    from coordpy.cache_controller_v5 import (
        CacheControllerV5,
        emit_cache_controller_v5_witness,
        fit_composite_v5,
        fit_corruption_repair_head_v5,
        fit_two_objective_ridge_v5,
    )
    cc = CacheControllerV5.init(
        d_model=64, d_key=8, fit_seed=62500)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 4))
    cc, two_obj = fit_two_objective_ridge_v5(
        controller=cc, train_features=X.tolist(),
        target_drop_oracle=X.sum(axis=-1).tolist(),
        target_retrieval_relevance=(X[:, 0] * 2).tolist())
    assert two_obj.converged
    cc, repair = fit_corruption_repair_head_v5(
        controller=cc,
        train_flag_counts=X[:, 0].astype(int).tolist(),
        train_hidden_writes=X[:, 1].tolist(),
        train_replay_ages=X[:, 2].astype(int).tolist(),
        train_attention_receive_l1=X[:, 3].tolist(),
        target_repair_amounts=(X[:, 0] * 0.5).tolist())
    assert repair.converged
    heads = rng.standard_normal((20, 6))
    cc, comp = fit_composite_v5(
        controller=cc, head_scores=heads.tolist(),
        drop_oracle=heads.sum(axis=-1).tolist())
    assert comp.converged
    w = emit_cache_controller_v5_witness(controller=cc)
    assert w.two_objective_used
    assert w.repair_head_used
    assert w.composite_v5_used


def test_hidden_state_bridge_v6_three_target():
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
        fit_hsb_v6_three_target,
        recover_hsb_v6_inject_v2,
        write_hsb_v6_into_v7_cache_ledger,
    )
    from coordpy.tiny_substrate_v7 import (
        TinyV7KVCache,
        build_default_tiny_substrate_v7, tokenize_bytes_v7,
    )
    p = build_default_tiny_substrate_v7(seed=623)
    hsb_v2 = HiddenStateBridgeV2Projection.init(
        target_layers=(1, 3), n_tokens=4, carrier_dim=6,
        d_model=p.config.d_model, seed=62300)
    hsb_v3 = HiddenStateBridgeV3Projection.init_from_v2(
        hsb_v2, n_heads=p.config.n_heads, seed_v3=62301)
    hsb_v4 = HiddenStateBridgeV4Projection.init_from_v3(
        hsb_v3, seed_v4=62302)
    hsb_v5 = HiddenStateBridgeV5Projection.init_from_v4(
        hsb_v4, n_positions=3, seed_v5=62303)
    hsb_v6 = HiddenStateBridgeV6Projection.init_from_v5(
        hsb_v5, seed_v6=62304)
    rng = np.random.default_rng(0)
    carriers = [list(rng.standard_normal(6)) for _ in range(3)]
    ids = tokenize_bytes_v7("hsb6", max_len=4)[:4]
    while len(ids) < 4: ids.append(0)
    target1 = np.zeros(p.config.vocab_size); target1[100] = 1.0
    target2 = np.zeros(p.config.vocab_size); target2[120] = 0.7
    target3 = np.zeros(p.config.vocab_size); target3[150] = 0.5
    fitted, report = fit_hsb_v6_three_target(
        params=p.v3_params, projection=hsb_v6,
        train_carriers=carriers,
        target_delta_logits_stack=[
            target1.tolist(), target2.tolist(),
            target3.tolist()],
        token_ids=ids)
    assert report.converged
    assert report.n_targets == 3
    # Recovery.
    adv = rng.standard_normal((2, p.config.n_heads, 3)) * 0.3
    _, audit = recover_hsb_v6_inject_v2(
        params=p.v3_params, projection=fitted,
        carrier=carriers[0], token_ids=ids,
        target_delta_logits=target1.tolist(),
        adversarial_per_head_pos=adv)
    assert audit["post_recovery_margin_positive"]


def test_prefix_state_bridge_v6_drift_curve():
    from coordpy.prefix_state_bridge_v6 import (
        emit_prefix_state_bridge_v6_witness,
        fit_prefix_drift_curve_predictor,
    )
    from coordpy.tiny_substrate_v5 import (
        TinyV5SubstrateConfig, TinyV5SubstrateParams,
    )
    cfg = TinyV5SubstrateConfig()
    params = TinyV5SubstrateParams.init(cfg)
    prompt = list(range(10))
    chain = [[11, 12], [13, 14], [15, 16]]
    configs = [
        [(0, 5, "reuse"), (5, 8, "recompute"),
         (8, 10, "drop")],
        [(0, 3, "reuse"), (3, 7, "recompute"),
         (7, 10, "drop")],
        [(0, 7, "reuse"), (7, 10, "recompute")],
        [(0, 4, "reuse"), (4, 10, "drop")],
    ]
    pred = fit_prefix_drift_curve_predictor(
        params_v5=params, prompt_token_ids=prompt,
        train_segment_configs=configs, train_chain=chain)
    assert pred.converged
    assert pred.n_target_steps == 3
    pc = pred.predict_curve(
        reuse_len=5, recompute_len=3, drop_len=2)
    assert len(pc) == 3
    w = emit_prefix_state_bridge_v6_witness(
        predictor=pred, actual_curve=[0.1, 0.2, 0.3])
    assert w.converged


def test_attention_steering_v6_two_stage_clamp():
    from coordpy.attention_steering_bridge_v2 import (
        AttentionSteeringV2Projection,
    )
    from coordpy.attention_steering_bridge_v6 import (
        steer_attention_and_measure_v6,
    )
    from coordpy.tiny_substrate_v7 import (
        build_default_tiny_substrate_v7, tokenize_bytes_v7,
    )
    p = build_default_tiny_substrate_v7(seed=624)
    ids = tokenize_bytes_v7("attn6", max_len=4)[:4]
    while len(ids) < 4: ids.append(0)
    proj = AttentionSteeringV2Projection.init(
        n_layers=p.config.n_layers, n_heads=p.config.n_heads,
        n_query=len(ids), n_key=len(ids), carrier_dim=6,
        seed=62400)
    rng = np.random.default_rng(0)
    carrier = list(rng.standard_normal(6))
    w = steer_attention_and_measure_v6(
        params=p.v3_params, carrier=carrier,
        projection=proj, token_ids=ids,
        coarse_l1_budget=0.3, kl_budget_per_key=0.4)
    assert w.two_stage_used
    # Negative-budget falsifier.
    w_neg = steer_attention_and_measure_v6(
        params=p.v3_params, carrier=carrier,
        projection=proj, token_ids=ids,
        coarse_l1_budget=0.0, kl_budget_per_key=0.0,
        negative_budget=True)
    assert float(w_neg.fine_kl_max_after_coarse) < 1e-6


def test_substrate_adapter_v7_tier():
    from coordpy.substrate_adapter_v7 import (
        W62_SUBSTRATE_TIER_SUBSTRATE_V7_FULL,
        probe_all_v7_adapters,
        probe_tiny_substrate_v7_adapter,
    )
    matrix = probe_all_v7_adapters(
        probe_ollama=False, probe_openai=False)
    assert matrix.has_v7_full()
    tiny = probe_tiny_substrate_v7_adapter()
    assert tiny.tier == W62_SUBSTRATE_TIER_SUBSTRATE_V7_FULL


def test_deep_substrate_hybrid_v7_seven_way():
    from coordpy.cache_controller_v5 import CacheControllerV5
    from coordpy.deep_substrate_hybrid_v6 import (
        DeepSubstrateHybridV6,
        DeepSubstrateHybridV6ForwardWitness,
    )
    from coordpy.deep_substrate_hybrid_v7 import (
        DeepSubstrateHybridV7,
        deep_substrate_hybrid_v7_forward,
    )
    from coordpy.replay_controller_v3 import (
        ReplayControllerV3,
    )
    cc = CacheControllerV5.init(d_model=64, d_key=8)
    cc.two_objective_head = np.zeros((4, 2))
    cc.repair_head_coefs = np.zeros((4,))
    cc.composite_v5_weights = np.ones((6,)) * (1.0 / 6.0)
    rcv3 = ReplayControllerV3.init()
    rcv3.audit_v3.append({
        "stage": "v3_per_regime",
        "regime": "crc_passed_low_drift",
        "replay_dominance": 0.3, "confidence": 0.7,
        "decision": "choose_reuse"})
    rcv3.hidden_vs_kv_classifier = np.zeros((3, 5))
    hybrid = DeepSubstrateHybridV7(
        inner_v6=DeepSubstrateHybridV6(inner_v5=None))
    v6_witness = DeepSubstrateHybridV6ForwardWitness(
        schema="x", hybrid_cid="x",
        inner_v5_witness_cid="x", six_way=True,
        cache_controller_v4_fired=True,
        replay_controller_v2_fired=True,
        attention_steering_v5_fired=True,
        bilinear_retrieval_v6_used=True,
        hidden_write_gate_fired=False,
        decision_confidence_mean=0.6,
        attention_pattern_jaccard_mean=0.6,
        attention_pattern_l2_max=0.5)
    wh = deep_substrate_hybrid_v7_forward(
        hybrid=hybrid, v6_witness=v6_witness,
        cache_controller_v5=cc, replay_controller_v3=rcv3,
        cache_write_ledger_l2=1.0,
        attention_v6_coarse_l1_shift=0.5,
        prefix_v6_drift_predictor_trained=True)
    assert wh.seven_way


def test_persistent_v14_decuple_skip_and_chain_depth():
    from coordpy.persistent_latent_v12 import V12StackedCell
    from coordpy.persistent_latent_v14 import (
        PersistentLatentStateV14Chain,
        W62_DEFAULT_V14_DISTRACTOR_RANK,
        W62_DEFAULT_V14_MAX_CHAIN_WALK_DEPTH,
        emit_persistent_v14_witness,
        step_persistent_state_v14,
    )
    cell = V12StackedCell.init(seed=62000)
    sd = int(cell.state_dim)
    state = step_persistent_state_v14(
        cell=cell, prev_state=None,
        carrier_values=[0.1] * sd,
        turn_index=0, role="r",
        replay_dominance_skip=[1.0] * sd)
    chain = PersistentLatentStateV14Chain.empty()
    chain.add(state)
    w = emit_persistent_v14_witness(chain, state.cid())
    assert w.decuple_skip_present
    assert w.replay_dominance_carrier_l1_sum > 0.0
    assert W62_DEFAULT_V14_MAX_CHAIN_WALK_DEPTH >= 2048
    assert W62_DEFAULT_V14_DISTRACTOR_RANK >= 8


def test_multi_hop_v12_seven_axis_and_compromise():
    from coordpy.multi_hop_translator_v12 import (
        W62_DEFAULT_MH_V12_BACKENDS,
        W62_DEFAULT_MH_V12_CHAIN_LEN,
        emit_multi_hop_v12_witness,
    )
    w = emit_multi_hop_v12_witness(
        backends=W62_DEFAULT_MH_V12_BACKENDS,
        chain_length=W62_DEFAULT_MH_V12_CHAIN_LEN, seed=62100)
    assert w.n_backends == 20
    assert w.chain_length == 17
    assert w.n_edges == 20 * 19
    assert w.seven_axis_used
    assert 1 <= int(w.compromise_threshold) <= 7


def test_mergeable_v10_wasserstein_and_chain_inherit():
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
        MergeOperatorV10,
        W62_MLSC_V10_ALGEBRA_REPLAY_DOMINANCE_PROPAGATION,
        emit_mlsc_v10_witness, wrap_v9_as_v10,
    )
    op = MergeOperatorV10(factor_dim=6)
    def _mk(payload, tag):
        v3 = make_root_capsule_v3(
            branch_id=tag, payload=tuple(payload),
            fact_tags=("w62",),
            confidence=0.9, trust=0.9, turn_index=0)
        v4 = wrap_v3_as_v4(v3)
        v5 = wrap_v4_as_v5(v4, attention_witness_cid="a")
        v6 = wrap_v5_as_v6(v5,
            attention_witness_chain=("aw",),
            cache_reuse_witness_cid="cr")
        v7 = wrap_v6_as_v7(v6,
            retrieval_witness_chain=("rw",),
            controller_witness_cid="ctrl")
        v8 = wrap_v7_as_v8(v7,
            replay_witness_chain=("repl",),
            substrate_witness_chain=("sub",),
            provenance_trust_table={"b": 0.9})
        v9 = wrap_v8_as_v9(v8,
            attention_pattern_witness_chain=("ap",),
            cache_retrieval_witness_chain=("crv9",))
        v10 = wrap_v9_as_v10(v9,
            replay_dominance_witness_chain=(f"rd_{tag}",),
            algebra_signature_v10=(
                W62_MLSC_V10_ALGEBRA_REPLAY_DOMINANCE_PROPAGATION))
        return v10
    cap_a = _mk([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], "a")
    cap_b = _mk([1.0, 1.1, 1.2, 1.3, 1.4, 1.5], "b")
    merged = op.merge(
        [cap_a, cap_b],
        replay_dominance_witness_chain=("merge",),
        algebra_signature_v10=(
            W62_MLSC_V10_ALGEBRA_REPLAY_DOMINANCE_PROPAGATION))
    w = emit_mlsc_v10_witness(merged)
    assert w.replay_dominance_witness_chain_depth >= 3
    assert w.disagreement_wasserstein_distance > 0.0


def test_consensus_v8_repair_stage_fires():
    from coordpy.consensus_fallback_controller_v8 import (
        ConsensusFallbackControllerV8,
        W62_CONSENSUS_V8_STAGES,
        W62_CONSENSUS_V8_STAGE_TRAINED_REPAIR,
        emit_consensus_v8_witness,
    )
    rc = ConsensusFallbackControllerV8.init(
        k_required=3, cosine_floor=0.99,
        repair_amount_threshold=0.1)
    out = rc.decide_v8(
        payloads=[
            [1.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0]],
        trusts=[0.9, 0.9, 0.6],
        replay_decisions=[
            "choose_reuse", "choose_reuse", "choose_reuse"],
        corruption_detected_per_parent=[True, False, False],
        repair_amount=0.5,
        repaired_payload=[0.7, 0.2, 0.1, 0.0])
    assert out["stage"] == W62_CONSENSUS_V8_STAGE_TRAINED_REPAIR
    assert len(W62_CONSENSUS_V8_STAGES) == 12
    w = emit_consensus_v8_witness(rc)
    assert w.repair_stage_fired >= 1


def test_crc_v10_kv1024_and_17bit_burst():
    from coordpy.corruption_robust_carrier_v10 import (
        CorruptionRobustCarrierV10,
        emit_corruption_robustness_v10_witness,
        kv_cache_fingerprint_1024,
    )
    crc = CorruptionRobustCarrierV10()
    w = emit_corruption_robustness_v10_witness(
        crc_v10=crc, n_probes=32, seed=62000)
    assert w.kv1024_corruption_detect_rate >= 0.95
    assert w.adversarial_17bit_burst_detect_rate >= 0.95
    fp_a = kv_cache_fingerprint_1024(b"abc", b"def")
    fp_b = kv_cache_fingerprint_1024(b"abc", b"deg")
    assert fp_a != fp_b


def test_lhr_v14_thirteen_way_and_replay_dominance_head():
    from coordpy.long_horizon_retention_v14 import (
        LongHorizonReconstructionV14Head,
        emit_lhr_v14_witness,
        fit_lhr_v14_four_layer_scorer,
    )
    head = LongHorizonReconstructionV14Head.init(seed=62000)
    runs = head.thirteen_way_value(
        carrier=[0.1] * 8, k=4,
        replay_dominance_indicator=[1.0] * 8)
    assert runs.size > 0
    # Replay-dominance head produces non-trivial output.
    zero = head.thirteen_way_value(
        carrier=[0.1] * 8, k=4,
        replay_dominance_indicator=[0.0] * 8)
    delta = float(np.linalg.norm(
        np.asarray(zero) - np.asarray(runs)))
    assert delta > 1e-6
    # Four-layer scorer fits.
    rng = np.random.default_rng(0)
    feat_dim = head.inner_v13.tanh_proj_dim
    X = rng.standard_normal((20, feat_dim))
    y = X.sum(axis=-1)
    _, audit = fit_lhr_v14_four_layer_scorer(
        head=head, train_features=X.tolist(),
        train_targets=y.tolist())
    assert audit["converged"]
    w = emit_lhr_v14_witness(head, carrier=[0.1] * 8, k=4)
    assert w.n_heads == 13


def test_ecc_v14_total_codes_and_bits_per_token():
    from coordpy.ecc_codebook_v14 import (
        ECCCodebookV14, compress_carrier_ecc_v14,
        emit_ecc_v14_compression_witness,
        probe_ecc_v14_rate_floor_falsifier,
    )
    from coordpy.ecc_codebook_v5 import (
        W53_DEFAULT_ECC_CODE_DIM,
        W53_DEFAULT_ECC_EMIT_MASK_LEN,
    )
    from coordpy.quantised_compression import QuantisedBudgetGate
    ecc = ECCCodebookV14.init(seed=62000)
    assert ecc.total_codes == 2 ** 23
    gate = QuantisedBudgetGate.init(
        in_dim=W53_DEFAULT_ECC_CODE_DIM,
        emit_mask_len=W53_DEFAULT_ECC_EMIT_MASK_LEN, seed=62000)
    gate.importance_threshold = 0.0
    gate.w_emit.values = [1.0] * len(gate.w_emit.values)
    carrier = [0.1] * W53_DEFAULT_ECC_CODE_DIM
    comp = compress_carrier_ecc_v14(
        carrier, codebook=ecc, gate=gate)
    assert comp["bits_per_visible_token"] >= 25.0
    out = probe_ecc_v14_rate_floor_falsifier(codebook=ecc)
    assert out["target_exceeds_ceiling"]
    w = emit_ecc_v14_compression_witness(
        codebook=ecc, compression=comp)
    assert w.bits_per_visible_token >= 25.0


def test_tvs_v11_twelve_arms_sum_and_replay_dominance_arm():
    from coordpy.transcript_vs_shared_arbiter_v11 import (
        W62_TVS_V11_ARMS,
        W62_TVS_V11_ARM_REPLAY_DOMINANCE,
        emit_tvs_arbiter_v11_witness,
        twelve_arm_compare,
    )
    out = twelve_arm_compare(
        per_turn_replay_dominance_fidelities=[0.99],
        per_turn_confidences=[0.0],
        per_turn_trust_scores=[0.0],
        per_turn_merge_retentions=[0.0],
        per_turn_tw_retentions=[0.0],
        per_turn_substrate_fidelities=[0.0],
        per_turn_hidden_fidelities=[0.0],
        per_turn_cache_fidelities=[0.0],
        per_turn_retrieval_fidelities=[0.0],
        per_turn_replay_fidelities=[0.0],
        per_turn_attention_pattern_fidelities=[0.0])
    s = sum(out.pick_rates.values())
    assert abs(s - 1.0) < 1e-6
    assert len(W62_TVS_V11_ARMS) == 12
    assert out.pick_rates.get(
        W62_TVS_V11_ARM_REPLAY_DOMINANCE, 0.0) > 0.5
    w = emit_tvs_arbiter_v11_witness(out)
    assert w.n_arms == 12


def test_uncertainty_v10_nine_axis_and_aware():
    from coordpy.uncertainty_layer_v10 import (
        compose_uncertainty_report_v10,
        emit_uncertainty_v10_witness,
    )
    out = compose_uncertainty_report_v10(
        confidences=[0.7, 0.5],
        trusts=[0.9, 0.8],
        substrate_fidelities=[0.95, 0.9],
        hidden_state_fidelities=[0.95, 0.92],
        cache_reuse_fidelities=[0.95, 0.92],
        retrieval_fidelities=[0.95, 0.93],
        replay_fidelities=[0.92, 0.90],
        attention_pattern_fidelities=[0.95, 0.90],
        replay_dominance_fidelities=[0.88, 0.85])
    assert out.n_axes == 9
    assert out.replay_dominance_aware
    w = emit_uncertainty_v10_witness(out)
    assert w.replay_dominance_aware


def test_disagreement_algebra_v8_wasserstein_identity():
    from coordpy.disagreement_algebra import AlgebraTrace
    from coordpy.disagreement_algebra_v8 import (
        emit_disagreement_algebra_v8_witness,
        wasserstein_1_distance,
    )
    # Identity holds for matching probes.
    assert wasserstein_1_distance(
        [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0
    w = emit_disagreement_algebra_v8_witness(
        trace=AlgebraTrace(steps=[]),
        probe_a=[0.1, 0.2, 0.3],
        probe_b=[0.1, 0.2, 0.3],
        probe_c=[0.1, 0.2, 0.3],
        wasserstein_oracle=lambda: (True, 0.1))
    assert w.wasserstein_equiv_ok
    # Falsifier triggers on argmax mismatch / high Wasserstein.
    w2 = emit_disagreement_algebra_v8_witness(
        trace=AlgebraTrace(steps=[]),
        probe_a=[0.1, 0.2, 0.3],
        probe_b=[0.1, 0.2, 0.3],
        probe_c=[0.1, 0.2, 0.3],
        wasserstein_falsifier_oracle=lambda: (False, 1.0))
    assert w2.wasserstein_falsifier_ok
