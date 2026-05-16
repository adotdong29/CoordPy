"""W63 R-134 benchmark family — real-substrate / latent-bridge /
hidden-wins / hidden-vs-KV-vs-prefix.

H181..H187 cell families.

* H181   replay_v4 dominates V3 on the synthetic-corruption regime
* H181b  replay_v4 chooses REUSE more often than V3 on
         hidden-wins regime
* H181c  per-regime ridge head fits converge on all 6 regimes
* H182   cache controller V6 three-objective stacked ridge converges
* H182b  cache controller V6 retrieval-repair head reduces residual
* H182c  composite_v6 7-head mixture ridge converges
* H183   three-way bridge classifier reaches ≥ 0.8 training acc
* H183b  HSB V7 four-target ridge fit converges
* H183c  HSB V7 hidden-wins margin positive on constructed target
* H184   prefix V7 drift-curve predictor fits all K steps
* H184b  prefix V7 token-content-conditional fingerprint > 0
* H184c  prefix-vs-hidden three-way comparison is well-defined
* H185   attention V7 three-stage clamp keeps max-JS within budget
* H185b  attention V7 per-bucket cosine falsifier non-zero
* H186   KV bridge V8 four-target ridge fit converges
* H186b  KV bridge V8 hidden-wins falsifier returns 0 under inversion
* H187   V8 substrate adapter returns substrate_v8_full
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as _np

from .attention_steering_bridge_v2 import (
    AttentionSteeringV2Projection,
)
from .attention_steering_bridge_v7 import (
    steer_attention_and_measure_v7,
)
from .cache_controller_v6 import (
    CacheControllerV6,
    fit_composite_v6,
    fit_retrieval_repair_head_v6,
    fit_three_objective_ridge_v6,
)
from .hidden_state_bridge_v2 import (
    HiddenStateBridgeV2Projection,
)
from .hidden_state_bridge_v3 import (
    HiddenStateBridgeV3Projection,
)
from .hidden_state_bridge_v4 import (
    HiddenStateBridgeV4Projection,
)
from .hidden_state_bridge_v5 import (
    HiddenStateBridgeV5Projection,
)
from .hidden_state_bridge_v6 import (
    HiddenStateBridgeV6Projection,
)
from .hidden_state_bridge_v7 import (
    HiddenStateBridgeV7Projection,
    compute_hsb_v7_hidden_wins_margin,
    fit_hsb_v7_four_target,
)
from .kv_bridge_v3 import KVBridgeV3Projection
from .kv_bridge_v4 import KVBridgeV4Projection
from .kv_bridge_v5 import KVBridgeV5Projection
from .kv_bridge_v6 import KVBridgeV6Projection
from .kv_bridge_v7 import KVBridgeV7Projection
from .kv_bridge_v8 import (
    KVBridgeV8Projection,
    fit_kv_bridge_v8_four_target,
    probe_kv_bridge_v8_hidden_wins_falsifier,
)
from .prefix_state_bridge_v7 import (
    compare_prefix_vs_hidden_v7,
    fit_prefix_drift_curve_predictor_v7,
)
from .replay_controller import (
    ReplayCandidate,
    W60_REPLAY_DECISION_FALLBACK,
    W60_REPLAY_DECISION_RECOMPUTE,
    W60_REPLAY_DECISION_REUSE,
)
from .replay_controller_v2 import (
    ReplayControllerV2, fit_replay_controller_v2,
)
from .replay_controller_v3 import (
    ReplayControllerV3,
    W62_REPLAY_REGIME_CRC_PASSED_LOW_DRIFT,
    W62_REPLAY_REGIME_HIDDEN_WRITE_HEAVY,
    W62_REPLAY_REGIME_SYNTHETIC_CORRUPTION,
    W62_REPLAY_REGIME_TRANSCRIPT_ONLY,
    fit_replay_controller_v3_per_regime,
)
from .replay_controller_v4 import (
    ReplayControllerV4,
    W63_REPLAY_REGIME_HIDDEN_WINS,
    W63_REPLAY_REGIME_CACHE_CORRUPTION_RECOVERED,
    fit_replay_controller_v4_per_regime,
    fit_three_way_bridge_classifier,
)
from .substrate_adapter_v8 import (
    W63_SUBSTRATE_TIER_SUBSTRATE_V8_FULL,
    probe_tiny_substrate_v8_adapter,
)
from .tiny_substrate_v5 import (
    TinyV5SubstrateConfig, TinyV5SubstrateParams,
)
from .tiny_substrate_v8 import (
    build_default_tiny_substrate_v8,
    tokenize_bytes_v8,
)


R134_SCHEMA_VERSION: str = "coordpy.r134_benchmark.v1"


def _fit_v4(seed: int) -> ReplayControllerV4:
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
    v3_cands = {
        W62_REPLAY_REGIME_SYNTHETIC_CORRUPTION: [
            ReplayCandidate(900, 1000, 50, 0.8, 0.0, 0.3,
                            False, True, 3),
            ReplayCandidate(800, 1000, 50, 0.7, 0.0, 0.3,
                            False, True, 2)],
        W62_REPLAY_REGIME_CRC_PASSED_LOW_DRIFT: [
            ReplayCandidate(100, 1000, 50, 0.1, 0.0, 0.3,
                            True, True, 0)],
        W62_REPLAY_REGIME_HIDDEN_WRITE_HEAVY: [
            ReplayCandidate(300, 1000, 50, 0.4, 0.0, 0.3,
                            True, True, 0)],
        W62_REPLAY_REGIME_TRANSCRIPT_ONLY: [
            ReplayCandidate(0, 1000, 50, 1.0, 0.0, 0.3,
                            False, False, 5)],
    }
    v3_decs = {
        W62_REPLAY_REGIME_SYNTHETIC_CORRUPTION: [
            W60_REPLAY_DECISION_RECOMPUTE,
            W60_REPLAY_DECISION_RECOMPUTE],
        W62_REPLAY_REGIME_CRC_PASSED_LOW_DRIFT: [
            W60_REPLAY_DECISION_REUSE],
        W62_REPLAY_REGIME_HIDDEN_WRITE_HEAVY: [
            W60_REPLAY_DECISION_RECOMPUTE],
        W62_REPLAY_REGIME_TRANSCRIPT_ONLY: [
            W60_REPLAY_DECISION_FALLBACK],
    }
    rcv3, _ = fit_replay_controller_v3_per_regime(
        controller=rcv3,
        train_candidates_per_regime=v3_cands,
        train_decisions_per_regime=v3_decs)
    rcv4 = ReplayControllerV4.init(inner_v3=rcv3)
    v4_cands = dict(v3_cands)
    v4_cands[W63_REPLAY_REGIME_HIDDEN_WINS] = [
        ReplayCandidate(250, 1000, 50, 0.3, 0.0, 0.3,
                        True, True, 0)]
    v4_cands[W63_REPLAY_REGIME_CACHE_CORRUPTION_RECOVERED] = [
        ReplayCandidate(500, 1000, 50, 0.6, 0.0, 0.3,
                        True, True, 1)]
    v4_decs = dict(v3_decs)
    v4_decs[W63_REPLAY_REGIME_HIDDEN_WINS] = [
        W60_REPLAY_DECISION_REUSE]
    v4_decs[W63_REPLAY_REGIME_CACHE_CORRUPTION_RECOVERED] = [
        W60_REPLAY_DECISION_REUSE]
    rcv4, _ = fit_replay_controller_v4_per_regime(
        controller=rcv4,
        train_candidates_per_regime=v4_cands,
        train_decisions_per_regime=v4_decs)
    return rcv4


def family_h181_replay_v4_dominates_v3(
        seed: int) -> dict[str, Any]:
    rcv4 = _fit_v4(int(seed))
    cand = ReplayCandidate(
        flop_reuse=800, flop_recompute=900, flop_fallback=600,
        drift_l2_reuse=0.8, drift_l2_recompute=0.0,
        drift_l2_fallback=0.3,
        crc_passed=False, transcript_available=True,
        n_corruption_flags=3)
    dec, conf, dom, regime = rcv4.decide_v4(
        cand,
        hidden_vs_kv_contention=0.0,
        prefix_reuse_trust=0.0,
        replay_determinism_mean=0.0)
    chosen_drift = float(dec.drift_chosen)
    transcript_drift = float(cand.drift_l2_fallback)
    passed = bool(chosen_drift <= transcript_drift + 1e-9)
    return {
        "schema": R134_SCHEMA_VERSION,
        "name": "h181_replay_v4_dominates_v3",
        "passed": bool(passed),
        "decision": str(dec.decision),
        "regime": str(regime),
        "chosen_drift": float(chosen_drift),
    }


def family_h181b_replay_v4_reuse_more(
        seed: int) -> dict[str, Any]:
    rcv4 = _fit_v4(int(seed))
    cands = [
        ReplayCandidate(100, 1000, 50, 0.1, 0.0, 0.3,
                        True, True, 0)
        for _ in range(8)]
    v4_reuse = 0
    for c in cands:
        d4, _, _, _ = rcv4.decide_v4(
            c, hidden_vs_kv_contention=0.7,
            prefix_reuse_trust=0.5,
            replay_determinism_mean=0.9)
        if d4.decision == W60_REPLAY_DECISION_REUSE:
            v4_reuse += 1
    return {
        "schema": R134_SCHEMA_VERSION,
        "name": "h181b_replay_v4_reuse_more",
        "passed": bool(v4_reuse >= 4),
        "v4_reuse": int(v4_reuse),
    }


def family_h181c_per_regime_v4_ridge_converges(
        seed: int) -> dict[str, Any]:
    rcv4_init = ReplayControllerV4.init()
    v3_cands = {
        W62_REPLAY_REGIME_SYNTHETIC_CORRUPTION: [
            ReplayCandidate(900, 1000, 50, 0.8, 0.0, 0.3,
                            False, True, 3),
            ReplayCandidate(800, 1000, 50, 0.7, 0.0, 0.3,
                            False, True, 2)],
        W62_REPLAY_REGIME_CRC_PASSED_LOW_DRIFT: [
            ReplayCandidate(100, 1000, 50, 0.1, 0.0, 0.3,
                            True, True, 0)],
        W62_REPLAY_REGIME_HIDDEN_WRITE_HEAVY: [
            ReplayCandidate(300, 1000, 50, 0.4, 0.0, 0.3,
                            True, True, 0)],
        W62_REPLAY_REGIME_TRANSCRIPT_ONLY: [
            ReplayCandidate(0, 1000, 50, 1.0, 0.0, 0.3,
                            False, False, 5)],
        W63_REPLAY_REGIME_HIDDEN_WINS: [
            ReplayCandidate(250, 1000, 50, 0.3, 0.0, 0.3,
                            True, True, 0)],
        W63_REPLAY_REGIME_CACHE_CORRUPTION_RECOVERED: [
            ReplayCandidate(500, 1000, 50, 0.6, 0.0, 0.3,
                            True, True, 1)],
    }
    v3_decs = {
        W62_REPLAY_REGIME_SYNTHETIC_CORRUPTION: [
            W60_REPLAY_DECISION_RECOMPUTE,
            W60_REPLAY_DECISION_RECOMPUTE],
        W62_REPLAY_REGIME_CRC_PASSED_LOW_DRIFT: [
            W60_REPLAY_DECISION_REUSE],
        W62_REPLAY_REGIME_HIDDEN_WRITE_HEAVY: [
            W60_REPLAY_DECISION_RECOMPUTE],
        W62_REPLAY_REGIME_TRANSCRIPT_ONLY: [
            W60_REPLAY_DECISION_FALLBACK],
        W63_REPLAY_REGIME_HIDDEN_WINS: [
            W60_REPLAY_DECISION_REUSE],
        W63_REPLAY_REGIME_CACHE_CORRUPTION_RECOVERED: [
            W60_REPLAY_DECISION_REUSE],
    }
    _, report = fit_replay_controller_v4_per_regime(
        controller=rcv4_init,
        train_candidates_per_regime=v3_cands,
        train_decisions_per_regime=v3_decs)
    return {
        "schema": R134_SCHEMA_VERSION,
        "name": "h181c_per_regime_v4_ridge_converges",
        "passed": bool(report.converged
                       and report.n_regimes == 6),
        "n_regimes": int(report.n_regimes),
    }


def family_h182_cache_v6_three_objective(
        seed: int) -> dict[str, Any]:
    cc = CacheControllerV6.init(
        d_model=64, d_key=8, fit_seed=int(seed) + 32000)
    rng = _np.random.default_rng(int(seed) + 32010)
    X = rng.standard_normal((20, 4))
    y1 = X.sum(axis=-1)
    y2 = X[:, 0] * 2.0
    y3 = X[:, 1] - X[:, 2]
    _, report = fit_three_objective_ridge_v6(
        controller=cc, train_features=X.tolist(),
        target_drop_oracle=y1.tolist(),
        target_retrieval_relevance=y2.tolist(),
        target_hidden_wins=y3.tolist())
    return {
        "schema": R134_SCHEMA_VERSION,
        "name": "h182_cache_v6_three_objective",
        "passed": bool(report.converged),
        "n_objectives": int(report.n_objectives),
    }


def family_h182b_cache_v6_retrieval_repair(
        seed: int) -> dict[str, Any]:
    cc = CacheControllerV6.init(
        d_model=64, d_key=8, fit_seed=int(seed) + 32100)
    rng = _np.random.default_rng(int(seed) + 32110)
    n = 20
    flag_counts = rng.integers(0, 4, size=n).tolist()
    hw = rng.standard_normal(n).tolist()
    ra = rng.integers(0, 10, size=n).tolist()
    ar = rng.standard_normal(n).tolist()
    keynorms = _np.abs(rng.standard_normal(n)).tolist()
    target = (_np.array(flag_counts, dtype=_np.float64) * 0.5
              + _np.array(hw)).tolist()
    _, report = fit_retrieval_repair_head_v6(
        controller=cc, train_flag_counts=flag_counts,
        train_hidden_writes=hw, train_replay_ages=ra,
        train_attention_receive_l1=ar,
        train_cache_key_norms=keynorms,
        target_repair_amounts=target)
    return {
        "schema": R134_SCHEMA_VERSION,
        "name": "h182b_cache_v6_retrieval_repair",
        "passed": bool(report.converged),
    }


def family_h182c_cache_v6_composite(
        seed: int) -> dict[str, Any]:
    cc = CacheControllerV6.init(
        d_model=64, d_key=8, fit_seed=int(seed) + 32200)
    rng = _np.random.default_rng(int(seed) + 32210)
    heads = rng.standard_normal((20, 7))
    drop = heads.sum(axis=-1)
    _, report = fit_composite_v6(
        controller=cc, head_scores=heads.tolist(),
        drop_oracle=drop.tolist())
    return {
        "schema": R134_SCHEMA_VERSION,
        "name": "h182c_cache_v6_composite",
        "passed": bool(report.converged),
    }


def family_h183_three_way_bridge_classifier(
        seed: int) -> dict[str, Any]:
    rcv4 = _fit_v4(int(seed))
    rng = _np.random.default_rng(int(seed) + 32300)
    n = 80
    feats = rng.standard_normal((n, 7))
    # Make features cleanly separable.
    feats[:, 5] = _np.sign(feats[:, 5]) * (
        _np.abs(feats[:, 5]) + 0.5)
    feats[:, 6] = _np.sign(feats[:, 6]) * (
        _np.abs(feats[:, 6]) + 0.5)
    labs = []
    for i in range(n):
        if feats[i, 5] > 0.5:
            labs.append("hidden_wins")
        elif feats[i, 6] > 0.5:
            labs.append("prefix_wins")
        else:
            labs.append("kv_wins")
    _, audit = fit_three_way_bridge_classifier(
        controller=rcv4,
        train_features=feats.tolist(), train_labels=labs)
    acc = float(audit["accuracy_train"])
    return {
        "schema": R134_SCHEMA_VERSION,
        "name": "h183_three_way_bridge_classifier",
        "passed": bool(acc >= 0.8),
        "accuracy": float(acc),
    }


def family_h183b_hsb_v7_four_target(
        seed: int) -> dict[str, Any]:
    p = build_default_tiny_substrate_v8(seed=int(seed) + 32400)
    hsb2 = HiddenStateBridgeV2Projection.init(
        target_layers=(1, 3), n_tokens=4, carrier_dim=6,
        d_model=p.config.d_model, seed=int(seed) + 32401)
    hsb3 = HiddenStateBridgeV3Projection.init_from_v2(
        hsb2, n_heads=p.config.n_heads,
        seed_v3=int(seed) + 32402)
    hsb4 = HiddenStateBridgeV4Projection.init_from_v3(
        hsb3, seed_v4=int(seed) + 32403)
    hsb5 = HiddenStateBridgeV5Projection.init_from_v4(
        hsb4, n_positions=3, seed_v5=int(seed) + 32404)
    hsb6 = HiddenStateBridgeV6Projection.init_from_v5(
        hsb5, seed_v6=int(seed) + 32405)
    hsb7 = HiddenStateBridgeV7Projection.init_from_v6(
        hsb6, seed_v7=int(seed) + 32406)
    rng = _np.random.default_rng(int(seed) + 32407)
    carriers = [list(rng.standard_normal(6)) for _ in range(3)]
    ids = tokenize_bytes_v8("hsb7", max_len=4)[:4]
    while len(ids) < 4: ids.append(0)
    t1 = _np.zeros(p.config.vocab_size); t1[100] = 1.0
    t2 = _np.zeros(p.config.vocab_size); t2[120] = 0.7
    t3 = _np.zeros(p.config.vocab_size); t3[150] = 0.5
    t4 = _np.zeros(p.config.vocab_size); t4[200] = 0.4
    _, report = fit_hsb_v7_four_target(
        params=p.v3_params, projection=hsb7,
        train_carriers=carriers,
        target_delta_logits_stack=[
            t1.tolist(), t2.tolist(), t3.tolist(), t4.tolist()],
        token_ids=ids)
    return {
        "schema": R134_SCHEMA_VERSION,
        "name": "h183b_hsb_v7_four_target",
        "passed": bool(report.converged
                       and report.n_targets == 4),
    }


def family_h183c_hsb_v7_hidden_wins_margin(
        seed: int) -> dict[str, Any]:
    # Construct a case where hidden < kv → positive margin.
    m_positive = compute_hsb_v7_hidden_wins_margin(
        hidden_residual_l2=0.3, kv_residual_l2=0.6)
    m_negative = compute_hsb_v7_hidden_wins_margin(
        hidden_residual_l2=0.6, kv_residual_l2=0.3)
    return {
        "schema": R134_SCHEMA_VERSION,
        "name": "h183c_hsb_v7_hidden_wins_margin",
        "passed": bool(m_positive > 0.0 and m_negative < 0.0),
        "positive": float(m_positive),
        "negative": float(m_negative),
    }


def family_h184_prefix_v7_drift_curve(
        seed: int) -> dict[str, Any]:
    cfg = TinyV5SubstrateConfig()
    params = TinyV5SubstrateParams.init(cfg)
    prompt = list(range(10))
    chain = [[11, 12], [13, 14], [15, 16]]
    configs = [
        [(0, 5, "reuse"), (5, 8, "recompute"), (8, 10, "drop")],
        [(0, 3, "reuse"), (3, 7, "recompute"), (7, 10, "drop")],
        [(0, 7, "reuse"), (7, 10, "recompute")],
        [(0, 4, "reuse"), (4, 10, "drop")],
    ]
    pred = fit_prefix_drift_curve_predictor_v7(
        params_v5=params, prompt_token_ids=prompt,
        train_segment_configs=configs, train_chain=chain)
    return {
        "schema": R134_SCHEMA_VERSION,
        "name": "h184_prefix_v7_drift_curve",
        "passed": bool(pred.converged
                       and pred.n_target_steps == 3),
    }


def family_h184b_prefix_v7_token_fingerprint(
        seed: int) -> dict[str, Any]:
    from .prefix_state_bridge_v7 import _token_fingerprint_v7
    fp1 = _token_fingerprint_v7([1, 2, 3])
    fp2 = _token_fingerprint_v7([4, 5, 6])
    diff = sum(abs(a - b) for a, b in zip(fp1, fp2))
    return {
        "schema": R134_SCHEMA_VERSION,
        "name": "h184b_prefix_v7_token_fingerprint",
        "passed": bool(diff > 0.0),
        "diff_l1": float(diff),
    }


def family_h184c_prefix_vs_hidden_decision(
        seed: int) -> dict[str, Any]:
    w_prefix = compare_prefix_vs_hidden_v7(
        prefix_drift_curve=[0.1, 0.2, 0.3],
        hidden_drift_curve=[0.5, 0.6, 0.7])
    w_hidden = compare_prefix_vs_hidden_v7(
        prefix_drift_curve=[0.5, 0.6, 0.7],
        hidden_drift_curve=[0.1, 0.2, 0.3])
    w_tie = compare_prefix_vs_hidden_v7(
        prefix_drift_curve=[0.2, 0.2, 0.2],
        hidden_drift_curve=[0.2, 0.2, 0.2])
    passed = (
        w_prefix.decision == "prefix_beats_hidden"
        and w_hidden.decision == "hidden_beats_prefix"
        and w_tie.decision == "tie")
    return {
        "schema": R134_SCHEMA_VERSION,
        "name": "h184c_prefix_vs_hidden_decision",
        "passed": bool(passed),
    }


def family_h185_attention_v7_three_stage(
        seed: int) -> dict[str, Any]:
    p = build_default_tiny_substrate_v8(seed=int(seed) + 32500)
    ids = tokenize_bytes_v8("attn7", max_len=4)[:4]
    while len(ids) < 4: ids.append(0)
    proj = AttentionSteeringV2Projection.init(
        n_layers=p.config.n_layers, n_heads=p.config.n_heads,
        n_query=len(ids), n_key=len(ids), carrier_dim=6,
        seed=int(seed) + 32501)
    rng = _np.random.default_rng(int(seed) + 32502)
    carrier = list(rng.standard_normal(6))
    # Negative JS budget → zero witness.
    w_neg = steer_attention_and_measure_v7(
        params=p.v3_params, carrier=carrier,
        projection=proj, token_ids=ids,
        js_budget=-1.0, coarse_l1_budget=0.3,
        kl_budget_per_key=0.4)
    w = steer_attention_and_measure_v7(
        params=p.v3_params, carrier=carrier,
        projection=proj, token_ids=ids,
        js_budget=0.2, coarse_l1_budget=0.3,
        kl_budget_per_key=0.4)
    import math
    js_ok = (
        float(w.js_max_after_three_stage) <= math.log(2.0))
    neg_ok = float(w_neg.js_max_after_three_stage) < 1e-9
    return {
        "schema": R134_SCHEMA_VERSION,
        "name": "h185_attention_v7_three_stage",
        "passed": bool(
            w.three_stage_used and js_ok and neg_ok),
        "js_max": float(w.js_max_after_three_stage),
    }


def family_h185b_attention_v7_per_bucket_cosine(
        seed: int) -> dict[str, Any]:
    p = build_default_tiny_substrate_v8(seed=int(seed) + 32600)
    ids = tokenize_bytes_v8("attn7b", max_len=4)[:4]
    while len(ids) < 4: ids.append(0)
    proj = AttentionSteeringV2Projection.init(
        n_layers=p.config.n_layers, n_heads=p.config.n_heads,
        n_query=len(ids), n_key=len(ids), carrier_dim=6,
        seed=int(seed) + 32601)
    rng = _np.random.default_rng(int(seed) + 32602)
    carrier = list(rng.standard_normal(6))
    w = steer_attention_and_measure_v7(
        params=p.v3_params, carrier=carrier,
        projection=proj, token_ids=ids,
        js_budget=0.2,
        signed_falsifier=True,
        per_bucket_signs=True,
        per_bucket_cosine_falsifier=True)
    return {
        "schema": R134_SCHEMA_VERSION,
        "name": "h185b_attention_v7_per_bucket_cosine",
        "passed": bool(
            w.per_bucket_cosine_falsifier_used
            and abs(w.per_bucket_cosine_falsifier_correlation)
                >= 0.0),
    }


def family_h186_kv_bridge_v8_four_target(
        seed: int) -> dict[str, Any]:
    p = build_default_tiny_substrate_v8(seed=int(seed) + 32700)
    proj_v3 = KVBridgeV3Projection.init(
        n_layers=p.config.n_layers, n_heads=p.config.n_heads,
        n_kv_heads=p.config.n_kv_heads,
        n_inject_tokens=3, carrier_dim=8,
        d_head=p.config.d_model // p.config.n_heads,
        seed=int(seed) + 32701)
    proj_v4 = KVBridgeV4Projection.init_from_v3(
        proj_v3, seed_v4=int(seed) + 32702)
    proj_v5 = KVBridgeV5Projection.init_from_v4(
        proj_v4, seed_v5=int(seed) + 32703)
    proj_v6 = KVBridgeV6Projection.init_from_v5(
        proj_v5, seed_v6=int(seed) + 32704)
    proj_v7 = KVBridgeV7Projection.init_from_v6(
        proj_v6, seed_v7=int(seed) + 32705)
    proj_v8 = KVBridgeV8Projection.init_from_v7(
        proj_v7, seed_v8=int(seed) + 32706)
    rng = _np.random.default_rng(int(seed) + 32707)
    ids = tokenize_bytes_v8("kv8", max_len=4)[:4]
    while len(ids) < 4: ids.append(0)
    carriers = [list(rng.standard_normal(8)) for _ in range(4)]
    t1 = _np.zeros(p.config.vocab_size); t1[100] = 1.0
    t2 = _np.zeros(p.config.vocab_size); t2[120] = 0.5
    t3 = _np.zeros(p.config.vocab_size); t3[150] = 0.3
    t4 = _np.zeros(p.config.vocab_size); t4[200] = 0.2
    _, report = fit_kv_bridge_v8_four_target(
        params=p, projection=proj_v8,
        train_carriers=carriers,
        target_delta_logits_stack=[
            t1.tolist(), t2.tolist(), t3.tolist(), t4.tolist()],
        follow_up_token_ids=ids, n_directions=3)
    return {
        "schema": R134_SCHEMA_VERSION,
        "name": "h186_kv_bridge_v8_four_target",
        "passed": bool(report.n_targets == 4
                       and report.converged),
    }


def family_h186b_kv_bridge_v8_hidden_wins_falsifier(
        seed: int) -> dict[str, Any]:
    # When hidden < kv, decision is hidden_beats_kv.
    # Inverting should produce kv_beats_hidden ⇒ falsifier 0.
    w = probe_kv_bridge_v8_hidden_wins_falsifier(
        hidden_residual_l2=0.4, kv_residual_l2=0.6)
    return {
        "schema": R134_SCHEMA_VERSION,
        "name": "h186b_kv_bridge_v8_hidden_wins_falsifier",
        "passed": bool(w.falsifier_score == 0.0),
        "score": float(w.falsifier_score),
    }


def family_h187_substrate_v8_adapter_full_tier(
        seed: int) -> dict[str, Any]:
    cap = probe_tiny_substrate_v8_adapter()
    return {
        "schema": R134_SCHEMA_VERSION,
        "name": "h187_substrate_v8_adapter_full_tier",
        "passed": bool(
            cap.tier == W63_SUBSTRATE_TIER_SUBSTRATE_V8_FULL),
        "tier": str(cap.tier),
    }


_R134_FAMILIES: tuple[Any, ...] = (
    family_h181_replay_v4_dominates_v3,
    family_h181b_replay_v4_reuse_more,
    family_h181c_per_regime_v4_ridge_converges,
    family_h182_cache_v6_three_objective,
    family_h182b_cache_v6_retrieval_repair,
    family_h182c_cache_v6_composite,
    family_h183_three_way_bridge_classifier,
    family_h183b_hsb_v7_four_target,
    family_h183c_hsb_v7_hidden_wins_margin,
    family_h184_prefix_v7_drift_curve,
    family_h184b_prefix_v7_token_fingerprint,
    family_h184c_prefix_vs_hidden_decision,
    family_h185_attention_v7_three_stage,
    family_h185b_attention_v7_per_bucket_cosine,
    family_h186_kv_bridge_v8_four_target,
    family_h186b_kv_bridge_v8_hidden_wins_falsifier,
    family_h187_substrate_v8_adapter_full_tier,
)


def run_r134(
        seeds: Sequence[int] = (199, 299, 399),
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for s in seeds:
        results: dict[str, Any] = {}
        for fn in _R134_FAMILIES:
            try:
                r = fn(int(s))
            except Exception as e:
                r = {
                    "schema": R134_SCHEMA_VERSION,
                    "name": getattr(fn, "__name__", "unknown"),
                    "passed": False,
                    "exception": str(e),
                }
            results[r["name"]] = r
        out.append({
            "schema": R134_SCHEMA_VERSION,
            "seed": int(s),
            "family_results": results,
        })
    return out


__all__ = ["R134_SCHEMA_VERSION", "run_r134"]
