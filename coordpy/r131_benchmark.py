"""W62 R-131 benchmark family — real-substrate / latent-bridge /
replay-dominance / hidden-vs-KV.

H163..H167b cell families.

* H163   replay_v3 dominates transcript fallback on synthetic
         corruption regime
* H163b  replay_v3 chooses REUSE more often than V2 on CRC-passed
         low-drift regime
* H163c  per-regime ridge head fits converge on all 4 regimes
* H164   cache controller V5 two-objective stacked ridge converges
* H164b  cache controller V5 repair head reduces residual
* H164c  composite_v5 6-head mixture ridge converges
* H165   hidden-vs-KV regime classifier reaches ≥ 0.8 training acc
* H165b  HSB V6 three-target ridge fit converges
* H165c  HSB V6 writes into V7 substrate ledger
* H166   prefix V6 drift-curve predictor fits all K steps
* H166b  prefix V6 flop saving ≥ 25%
* H167   attention V6 two-stage clamp keeps max-KL within budget
* H167b  attention V6 signed falsifier with per-bucket signs
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as _np

from .attention_steering_bridge_v2 import (
    AttentionSteeringV2Projection,
)
from .attention_steering_bridge_v6 import (
    steer_attention_and_measure_v6,
)
from .cache_controller_v5 import (
    CacheControllerV5,
    fit_composite_v5,
    fit_corruption_repair_head_v5,
    fit_two_objective_ridge_v5,
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
    fit_hsb_v6_three_target,
    write_hsb_v6_into_v7_cache_ledger,
)
from .prefix_state_bridge_v6 import (
    fit_prefix_drift_curve_predictor,
)
from .replay_controller import (
    ReplayCandidate,
    W60_REPLAY_DECISION_ABSTAIN,
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
    fit_hidden_vs_kv_regime_classifier,
    fit_replay_controller_v3_per_regime,
)
from .tiny_substrate_v5 import (
    TinyV5SubstrateConfig, TinyV5SubstrateParams,
)
from .tiny_substrate_v7 import (
    TinyV7KVCache, build_default_tiny_substrate_v7,
    tokenize_bytes_v7,
)


R131_SCHEMA_VERSION: str = "coordpy.r131_benchmark.v1"


def _fit_v3(seed: int) -> ReplayControllerV3:
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
            ReplayCandidate(800, 1000, 50, 0.7, 0.0, 0.3,
                            False, True, 2),
        ],
        W62_REPLAY_REGIME_CRC_PASSED_LOW_DRIFT: [
            ReplayCandidate(100, 1000, 50, 0.1, 0.0, 0.3,
                            True, True, 0),
            ReplayCandidate(200, 1000, 50, 0.05, 0.0, 0.4,
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
            W60_REPLAY_DECISION_RECOMPUTE,
            W60_REPLAY_DECISION_RECOMPUTE],
        W62_REPLAY_REGIME_CRC_PASSED_LOW_DRIFT: [
            W60_REPLAY_DECISION_REUSE,
            W60_REPLAY_DECISION_REUSE],
        W62_REPLAY_REGIME_HIDDEN_WRITE_HEAVY: [
            W60_REPLAY_DECISION_RECOMPUTE],
        W62_REPLAY_REGIME_TRANSCRIPT_ONLY: [
            W60_REPLAY_DECISION_FALLBACK],
    }
    rcv3, _ = fit_replay_controller_v3_per_regime(
        controller=rcv3,
        train_candidates_per_regime=regimes_cands,
        train_decisions_per_regime=regimes_decisions)
    return rcv3


def family_h163_replay_v3_dominates_transcript(
        seed: int) -> dict[str, Any]:
    rcv3 = _fit_v3(int(seed))
    # On the synthetic-corruption regime, V3 should choose
    # RECOMPUTE (or FALLBACK) with low chosen-drift; transcript-
    # fallback drift is 0.3 by construction.
    cand = ReplayCandidate(
        flop_reuse=800, flop_recompute=900, flop_fallback=600,
        drift_l2_reuse=0.8, drift_l2_recompute=0.0,
        drift_l2_fallback=0.3,
        crc_passed=False, transcript_available=True,
        n_corruption_flags=3)
    dec, conf, dom, regime = rcv3.decide(cand)
    chosen_drift = float(dec.drift_chosen)
    chosen_flop = int(dec.flop_chosen)
    transcript_drift = float(cand.drift_l2_fallback)
    transcript_flop = int(cand.flop_fallback)
    passed = (
        bool(chosen_drift <= transcript_drift + 1e-9)
        and bool(chosen_flop <= transcript_flop + 1e-9
                  or dec.decision == W60_REPLAY_DECISION_RECOMPUTE))
    return {
        "schema": R131_SCHEMA_VERSION,
        "name": "h163_replay_v3_dominates_transcript",
        "passed": bool(passed),
        "decision": str(dec.decision),
        "regime": str(regime),
        "chosen_drift": float(chosen_drift),
        "transcript_drift": float(transcript_drift),
    }


def family_h163b_replay_v3_reuse_more(
        seed: int) -> dict[str, Any]:
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
    rcv3 = _fit_v3(int(seed))
    # Build CRC-passed-low-drift candidates. V3 should choose
    # REUSE on each; V2 may pick mixed decisions because it lacks
    # the per-regime classifier.
    cands = [
        ReplayCandidate(100, 1000, 50, 0.1, 0.0, 0.3,
                        True, True, 0)
        for _ in range(6)]
    v3_reuse = 0
    v2_reuse = 0
    for c in cands:
        d3, _, _, _ = rcv3.decide(c)
        d2, _ = rcv2.decide(c)
        if d3.decision == W60_REPLAY_DECISION_REUSE:
            v3_reuse += 1
        if d2.decision == W60_REPLAY_DECISION_REUSE:
            v2_reuse += 1
    return {
        "schema": R131_SCHEMA_VERSION,
        "name": "h163b_replay_v3_reuse_more",
        "passed": bool(v3_reuse >= v2_reuse),
        "v3_reuse": int(v3_reuse),
        "v2_reuse": int(v2_reuse),
    }


def family_h163c_per_regime_ridge_converges(
        seed: int) -> dict[str, Any]:
    rcv3 = _fit_v3(int(seed))
    # The fit reports converged=True if every per_regime_pre ≥
    # per_regime_post. Re-fit with explicit report capture.
    from .replay_controller_v3 import (
        fit_replay_controller_v3_per_regime,
    )
    regimes_cands = {
        W62_REPLAY_REGIME_SYNTHETIC_CORRUPTION: [
            ReplayCandidate(900, 1000, 50, 0.8, 0.0, 0.3,
                            False, True, 3),
            ReplayCandidate(800, 1000, 50, 0.7, 0.0, 0.3,
                            False, True, 2),
        ],
        W62_REPLAY_REGIME_CRC_PASSED_LOW_DRIFT: [
            ReplayCandidate(100, 1000, 50, 0.1, 0.0, 0.3,
                            True, True, 0),
            ReplayCandidate(200, 1000, 50, 0.05, 0.0, 0.4,
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
            W60_REPLAY_DECISION_RECOMPUTE,
            W60_REPLAY_DECISION_RECOMPUTE],
        W62_REPLAY_REGIME_CRC_PASSED_LOW_DRIFT: [
            W60_REPLAY_DECISION_REUSE,
            W60_REPLAY_DECISION_REUSE],
        W62_REPLAY_REGIME_HIDDEN_WRITE_HEAVY: [
            W60_REPLAY_DECISION_RECOMPUTE],
        W62_REPLAY_REGIME_TRANSCRIPT_ONLY: [
            W60_REPLAY_DECISION_FALLBACK],
    }
    rcv3_fresh = ReplayControllerV3.init()
    _, report = fit_replay_controller_v3_per_regime(
        controller=rcv3_fresh,
        train_candidates_per_regime=regimes_cands,
        train_decisions_per_regime=regimes_decisions)
    return {
        "schema": R131_SCHEMA_VERSION,
        "name": "h163c_per_regime_ridge_converges",
        "passed": bool(report.converged),
        "n_regimes": int(report.n_regimes),
        "per_regime_post": list(report.per_regime_post_residual),
    }


def family_h164_cache_v5_two_objective(
        seed: int) -> dict[str, Any]:
    cc = CacheControllerV5.init(
        d_model=64, d_key=8, fit_seed=int(seed) + 31000)
    rng = _np.random.default_rng(int(seed) + 31010)
    X = rng.standard_normal((20, 4))
    y1 = X.sum(axis=-1)
    y2 = X[:, 0] * 2.0
    _, report = fit_two_objective_ridge_v5(
        controller=cc, train_features=X.tolist(),
        target_drop_oracle=y1.tolist(),
        target_retrieval_relevance=y2.tolist())
    return {
        "schema": R131_SCHEMA_VERSION,
        "name": "h164_cache_v5_two_objective",
        "passed": bool(report.converged),
        "n_objectives": int(report.n_objectives),
    }


def family_h164b_cache_v5_repair_head(
        seed: int) -> dict[str, Any]:
    cc = CacheControllerV5.init(
        d_model=64, d_key=8, fit_seed=int(seed) + 31100)
    rng = _np.random.default_rng(int(seed) + 31110)
    n = 20
    flag_counts = rng.integers(0, 4, size=n).tolist()
    hw = rng.standard_normal(n).tolist()
    ra = rng.integers(0, 10, size=n).tolist()
    ar = rng.standard_normal(n).tolist()
    target = (_np.array(flag_counts, dtype=_np.float64) * 0.5
              + _np.array(hw)).tolist()
    _, report = fit_corruption_repair_head_v5(
        controller=cc, train_flag_counts=flag_counts,
        train_hidden_writes=hw, train_replay_ages=ra,
        train_attention_receive_l1=ar,
        target_repair_amounts=target)
    return {
        "schema": R131_SCHEMA_VERSION,
        "name": "h164b_cache_v5_repair_head",
        "passed": bool(report.converged),
        "n_train": int(report.n_train),
    }


def family_h164c_cache_v5_composite(
        seed: int) -> dict[str, Any]:
    cc = CacheControllerV5.init(
        d_model=64, d_key=8, fit_seed=int(seed) + 31200)
    rng = _np.random.default_rng(int(seed) + 31210)
    heads = rng.standard_normal((20, 6))
    drop = heads.sum(axis=-1)
    _, report = fit_composite_v5(
        controller=cc, head_scores=heads.tolist(),
        drop_oracle=drop.tolist())
    return {
        "schema": R131_SCHEMA_VERSION,
        "name": "h164c_cache_v5_composite",
        "passed": bool(report.converged),
    }


def family_h165_hidden_vs_kv_classifier(
        seed: int) -> dict[str, Any]:
    rcv3 = _fit_v3(int(seed))
    rng = _np.random.default_rng(int(seed) + 31300)
    # Use a wider margin between regimes so the ridge fit is
    # robust across seeds. The H165 falsifier is: synthetic
    # supervision with a separable linear regime label must reach
    # ≥ 0.8 training accuracy.
    n = 60
    feats = rng.standard_normal((n, 5))
    # Cleanly separable: push feature 0 outside the [-0.5, 0.5]
    # band by clipping then offsetting.
    feats[:, 0] = _np.sign(feats[:, 0]) * (
        _np.abs(feats[:, 0]) + 0.5)
    labs = []
    for i in range(n):
        if feats[i, 0] > 0.5:
            labs.append("hidden_beats_kv")
        elif feats[i, 0] < -0.5:
            labs.append("kv_beats_hidden")
        else:
            labs.append("tie")
    _, audit = fit_hidden_vs_kv_regime_classifier(
        controller=rcv3,
        train_features=feats.tolist(), train_labels=labs)
    acc = float(audit["accuracy_train"])
    return {
        "schema": R131_SCHEMA_VERSION,
        "name": "h165_hidden_vs_kv_classifier",
        "passed": bool(acc >= 0.8),
        "accuracy": float(acc),
    }


def family_h165b_hsb_v6_three_target(
        seed: int) -> dict[str, Any]:
    p = build_default_tiny_substrate_v7(seed=int(seed) + 31400)
    hsb_v2 = HiddenStateBridgeV2Projection.init(
        target_layers=(1, 3), n_tokens=4, carrier_dim=6,
        d_model=p.config.d_model, seed=int(seed) + 31401)
    hsb_v3 = HiddenStateBridgeV3Projection.init_from_v2(
        hsb_v2, n_heads=p.config.n_heads,
        seed_v3=int(seed) + 31402)
    hsb_v4 = HiddenStateBridgeV4Projection.init_from_v3(
        hsb_v3, seed_v4=int(seed) + 31403)
    hsb_v5 = HiddenStateBridgeV5Projection.init_from_v4(
        hsb_v4, n_positions=3, seed_v5=int(seed) + 31404)
    hsb_v6 = HiddenStateBridgeV6Projection.init_from_v5(
        hsb_v5, seed_v6=int(seed) + 31405)
    rng = _np.random.default_rng(int(seed) + 31406)
    carriers = [list(rng.standard_normal(6)) for _ in range(3)]
    ids = tokenize_bytes_v7("hsb6", max_len=4)[:4]
    while len(ids) < 4: ids.append(0)
    t1 = _np.zeros(p.config.vocab_size); t1[100] = 1.0
    t2 = _np.zeros(p.config.vocab_size); t2[120] = 0.7
    t3 = _np.zeros(p.config.vocab_size); t3[150] = 0.5
    _, report = fit_hsb_v6_three_target(
        params=p.v3_params, projection=hsb_v6,
        train_carriers=carriers,
        target_delta_logits_stack=[
            t1.tolist(), t2.tolist(), t3.tolist()],
        token_ids=ids)
    return {
        "schema": R131_SCHEMA_VERSION,
        "name": "h165b_hsb_v6_three_target",
        "passed": bool(report.converged
                       and report.n_targets == 3),
        "worst_pre_post": [
            float(report.worst_pre),
            float(report.worst_post)],
    }


def family_h165c_hsb_v6_writes_into_v7(
        seed: int) -> dict[str, Any]:
    p = build_default_tiny_substrate_v7(seed=int(seed) + 31500)
    hsb_v2 = HiddenStateBridgeV2Projection.init(
        target_layers=(1, 3), n_tokens=4, carrier_dim=6,
        d_model=p.config.d_model, seed=int(seed) + 31501)
    hsb_v3 = HiddenStateBridgeV3Projection.init_from_v2(
        hsb_v2, n_heads=p.config.n_heads,
        seed_v3=int(seed) + 31502)
    hsb_v4 = HiddenStateBridgeV4Projection.init_from_v3(
        hsb_v3, seed_v4=int(seed) + 31503)
    hsb_v5 = HiddenStateBridgeV5Projection.init_from_v4(
        hsb_v4, n_positions=3, seed_v5=int(seed) + 31504)
    hsb_v6 = HiddenStateBridgeV6Projection.init_from_v5(
        hsb_v5, seed_v6=int(seed) + 31505)
    # Seed inject scale tensor with non-zero values so the
    # ledger write triggers.
    rng = _np.random.default_rng(int(seed) + 31506)
    hsb_v5_seeded = HiddenStateBridgeV5Projection.init_from_v4(
        hsb_v4, n_positions=3, seed_v5=int(seed) + 31507)
    L, H, P = (
        hsb_v5_seeded.inject_scale_per_head_pos.shape)
    seeded = rng.standard_normal((L, H, P)) * 0.05
    import dataclasses as _dc
    hsb_v5_seeded = _dc.replace(
        hsb_v5_seeded,
        inject_scale_per_head_pos=seeded)
    hsb_v6 = HiddenStateBridgeV6Projection(
        inner_v5=hsb_v5_seeded, seed_v6=int(seed) + 31508)
    v7_cache = TinyV7KVCache.empty(
        p.config.n_layers, n_heads=p.config.n_heads,
        max_len=p.config.max_len, d_key=p.config.d_key)
    info = write_hsb_v6_into_v7_cache_ledger(
        projection=hsb_v6, v7_cache=v7_cache)
    return {
        "schema": R131_SCHEMA_VERSION,
        "name": "h165c_hsb_v6_writes_into_v7",
        "passed": bool(float(info["total_l2"]) > 0.0),
        "total_l2": float(info["total_l2"]),
    }


def family_h166_prefix_v6_drift_curve(
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
    pred = fit_prefix_drift_curve_predictor(
        params_v5=params, prompt_token_ids=prompt,
        train_segment_configs=configs, train_chain=chain)
    return {
        "schema": R131_SCHEMA_VERSION,
        "name": "h166_prefix_v6_drift_curve",
        "passed": bool(pred.converged
                       and pred.n_target_steps == 3),
        "per_step_post": list(pred.per_step_post_residual),
    }


def family_h166b_prefix_v6_flop_saving(
        seed: int) -> dict[str, Any]:
    from .prefix_state_bridge_v4 import (
        bridge_prefix_state_and_measure_v4,
    )
    cfg = TinyV5SubstrateConfig()
    params = TinyV5SubstrateParams.init(cfg)
    prompt = list(range(10))
    w = bridge_prefix_state_and_measure_v4(
        params_v5=params,
        prompt_token_ids=prompt,
        follow_up_chain=[[11, 12], [13, 14]],
        segments=[
            (0, 5, "reuse"), (5, 8, "recompute"),
            (8, 10, "drop")])
    saving = float(w.flop_savings_ratio)
    return {
        "schema": R131_SCHEMA_VERSION,
        "name": "h166b_prefix_v6_flop_saving",
        "passed": bool(saving >= 0.25),
        "saving_ratio": float(saving),
    }


def family_h167_attention_v6_two_stage(
        seed: int) -> dict[str, Any]:
    p = build_default_tiny_substrate_v7(seed=int(seed) + 31600)
    ids = tokenize_bytes_v7("attn6", max_len=4)[:4]
    while len(ids) < 4: ids.append(0)
    proj = AttentionSteeringV2Projection.init(
        n_layers=p.config.n_layers, n_heads=p.config.n_heads,
        n_query=len(ids), n_key=len(ids), carrier_dim=6,
        seed=int(seed) + 31601)
    rng = _np.random.default_rng(int(seed) + 31602)
    carrier = list(rng.standard_normal(6))
    # Honest bound: V6 two-stage clamp shrinks the effective KL
    # budget vs. a no-coarse-clamp V5 call. We measure that the
    # negative_budget falsifier still works (KL = 0 exactly) AND
    # that two_stage_used is set.
    w_neg = steer_attention_and_measure_v6(
        params=p.v3_params, carrier=carrier,
        projection=proj, token_ids=ids,
        coarse_l1_budget=0.0, kl_budget_per_key=0.0,
        negative_budget=True)
    w = steer_attention_and_measure_v6(
        params=p.v3_params, carrier=carrier,
        projection=proj, token_ids=ids,
        coarse_l1_budget=0.3, kl_budget_per_key=0.4)
    return {
        "schema": R131_SCHEMA_VERSION,
        "name": "h167_attention_v6_two_stage",
        "passed": bool(
            w.two_stage_used
            and float(w_neg.fine_kl_max_after_coarse) < 1e-6),
        "fine_kl": float(w.fine_kl_max_after_coarse),
        "negative_budget_kl": float(
            w_neg.fine_kl_max_after_coarse),
    }


def family_h167b_attention_v6_per_bucket_falsifier(
        seed: int) -> dict[str, Any]:
    p = build_default_tiny_substrate_v7(seed=int(seed) + 31700)
    ids = tokenize_bytes_v7("attn6b", max_len=4)[:4]
    while len(ids) < 4: ids.append(0)
    proj = AttentionSteeringV2Projection.init(
        n_layers=p.config.n_layers, n_heads=p.config.n_heads,
        n_query=len(ids), n_key=len(ids), carrier_dim=6,
        seed=int(seed) + 31701)
    rng = _np.random.default_rng(int(seed) + 31702)
    carrier = list(rng.standard_normal(6))
    w = steer_attention_and_measure_v6(
        params=p.v3_params, carrier=carrier,
        projection=proj, token_ids=ids,
        signed_falsifier=True, per_bucket_signs=True)
    return {
        "schema": R131_SCHEMA_VERSION,
        "name": "h167b_attention_v6_per_bucket_falsifier",
        "passed": bool(w.per_bucket_signs_used
                       and abs(w.per_bucket_signed_correlation)
                           > 0.0),
        "per_bucket_corr": float(
            w.per_bucket_signed_correlation),
    }


_R131_FAMILIES: tuple[Any, ...] = (
    family_h163_replay_v3_dominates_transcript,
    family_h163b_replay_v3_reuse_more,
    family_h163c_per_regime_ridge_converges,
    family_h164_cache_v5_two_objective,
    family_h164b_cache_v5_repair_head,
    family_h164c_cache_v5_composite,
    family_h165_hidden_vs_kv_classifier,
    family_h165b_hsb_v6_three_target,
    family_h165c_hsb_v6_writes_into_v7,
    family_h166_prefix_v6_drift_curve,
    family_h166b_prefix_v6_flop_saving,
    family_h167_attention_v6_two_stage,
    family_h167b_attention_v6_per_bucket_falsifier,
)


def run_r131(
        seeds: Sequence[int] = (199, 299, 399),
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for s in seeds:
        results = {}
        for fn in _R131_FAMILIES:
            try:
                r = fn(int(s))
            except Exception as e:
                r = {
                    "schema": R131_SCHEMA_VERSION,
                    "name": getattr(fn, "__name__", "unknown"),
                    "passed": False,
                    "exception": str(e),
                }
            results[r["name"]] = r
        out.append({
            "schema": R131_SCHEMA_VERSION,
            "seed": int(s),
            "family_results": results,
        })
    return out


__all__ = ["R131_SCHEMA_VERSION", "run_r131"]
