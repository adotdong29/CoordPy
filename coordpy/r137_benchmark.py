"""W64 R-137 benchmark family — replay-dominance / hidden-wins-primary
/ V9 substrate axes / four-way bridge / nine-way hybrid.

H203..H219 cell families (17 H-bars).

* H203   V9 substrate determinism + axes shape
* H203b  V9 hidden_wins_primary tensor shape (L, H, T)
* H203c  V9 replay_dominance_witness tensor shape (L, H, T)
* H203d  V9 attention_entropy probe shape (L,)
* H203e  V9 cache_similarity_matrix shape (L, H, T, T)
* H204   KV bridge V9 five-target ridge converges
* H204b  KV bridge V9 hidden_wins_primary falsifier returns 0
         under inversion
* H205   HSB V8 five-target ridge converges
* H205b  HSB V8 hidden_wins_primary margin positive when hidden
         dominates both KV and prefix
* H206   prefix V8 drift-curve predictor converges with role +
         token fingerprints
* H206b  prefix V8 three-way decision (prefix/hidden/replay) is
         valid
* H207   attention V8 four-stage clamp is bounded
* H207b  attention V8 returns zero under negative Hellinger budget
* H208   cache controller V7 four-objective ridge converges
* H208b  cache controller V7 similarity-aware eviction head
         converges
* H209   replay controller V5 per-regime head converges on all 7
         regimes
* H209b  replay controller V5 four-way bridge classifier ≥ 0.7
         training accuracy
* H209c  replay controller V5 dominance-primary head converges
* H210   substrate adapter V9 has matrix.has_v9_full=True
* H210b  hosted backends remain text-only at V9 tier
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as _np

from coordpy.attention_steering_bridge_v2 import (
    AttentionSteeringV2Projection,
)
from coordpy.attention_steering_bridge_v8 import (
    steer_attention_and_measure_v8,
)
from coordpy.cache_controller_v7 import (
    CacheControllerV7,
    fit_four_objective_ridge_v7,
    fit_similarity_aware_eviction_head_v7,
)
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
    fit_hsb_v8_five_target,
    compute_hsb_v8_hidden_wins_primary_margin,
)
from coordpy.kv_bridge_v3 import KVBridgeV3Projection
from coordpy.kv_bridge_v4 import KVBridgeV4Projection
from coordpy.kv_bridge_v5 import KVBridgeV5Projection
from coordpy.kv_bridge_v6 import KVBridgeV6Projection
from coordpy.kv_bridge_v7 import KVBridgeV7Projection
from coordpy.kv_bridge_v8 import KVBridgeV8Projection
from coordpy.kv_bridge_v9 import (
    KVBridgeV9Projection,
    fit_kv_bridge_v9_five_target,
    probe_kv_bridge_v9_hidden_wins_primary_falsifier,
)
from coordpy.prefix_state_bridge_v8 import (
    compare_prefix_vs_hidden_vs_replay_v8,
    fit_prefix_drift_curve_predictor_v8,
)
from coordpy.replay_controller import (
    ReplayCandidate, W60_REPLAY_DECISION_REUSE,
)
from coordpy.replay_controller_v5 import (
    ReplayControllerV5,
    W64_REPLAY_REGIMES_V5,
    fit_replay_controller_v5_per_regime,
    fit_four_way_bridge_classifier,
    fit_replay_dominance_primary_head_v5,
)
from coordpy.substrate_adapter_v9 import (
    W64_SUBSTRATE_TIER_SUBSTRATE_V9_FULL,
    probe_all_v9_adapters,
    probe_tiny_substrate_v9_adapter,
)
from coordpy.tiny_substrate_v9 import (
    build_default_tiny_substrate_v9,
    forward_tiny_substrate_v9,
    tokenize_bytes_v9,
)


R137_SCHEMA_VERSION: str = "coordpy.r137_benchmark.v1"


def family_h203_v9_substrate_determinism(
        seed: int) -> dict[str, Any]:
    p = build_default_tiny_substrate_v9(
        seed=int(seed) + 37000)
    ids = tokenize_bytes_v9("h203", max_len=12)
    t1, _ = forward_tiny_substrate_v9(p, ids)
    t2, _ = forward_tiny_substrate_v9(p, ids)
    return {
        "schema": R137_SCHEMA_VERSION,
        "name": "h203_v9_substrate_determinism",
        "passed": bool(t1.cid() == t2.cid()),
    }


def family_h203b_v9_hidden_wins_primary_shape(
        seed: int) -> dict[str, Any]:
    p = build_default_tiny_substrate_v9(
        seed=int(seed) + 37001)
    ids = tokenize_bytes_v9("h203b", max_len=12)
    _, c = forward_tiny_substrate_v9(p, ids)
    expected = (
        p.config.n_layers, p.config.n_heads, p.config.max_len)
    return {
        "schema": R137_SCHEMA_VERSION,
        "name": "h203b_v9_hidden_wins_primary_shape",
        "passed": bool(
            c.hidden_wins_primary.shape == expected),
        "shape": list(c.hidden_wins_primary.shape),
    }


def family_h203c_v9_replay_dominance_witness_shape(
        seed: int) -> dict[str, Any]:
    p = build_default_tiny_substrate_v9(
        seed=int(seed) + 37002)
    ids = tokenize_bytes_v9("h203c", max_len=12)
    _, c = forward_tiny_substrate_v9(p, ids)
    expected = (
        p.config.n_layers, p.config.n_heads, p.config.max_len)
    return {
        "schema": R137_SCHEMA_VERSION,
        "name": "h203c_v9_replay_dominance_witness_shape",
        "passed": bool(
            c.replay_dominance_witness.shape == expected),
        "shape": list(c.replay_dominance_witness.shape),
    }


def family_h203d_v9_attention_entropy_shape(
        seed: int) -> dict[str, Any]:
    p = build_default_tiny_substrate_v9(
        seed=int(seed) + 37003)
    ids = tokenize_bytes_v9("h203d", max_len=12)
    t, _ = forward_tiny_substrate_v9(p, ids)
    return {
        "schema": R137_SCHEMA_VERSION,
        "name": "h203d_v9_attention_entropy_shape",
        "passed": bool(
            t.attention_entropy_per_layer.shape
            == (p.config.n_layers,)),
        "shape": list(t.attention_entropy_per_layer.shape),
    }


def family_h203e_v9_cache_similarity_matrix_shape(
        seed: int) -> dict[str, Any]:
    p = build_default_tiny_substrate_v9(
        seed=int(seed) + 37004)
    ids = tokenize_bytes_v9("h203e", max_len=12)
    _, c = forward_tiny_substrate_v9(p, ids)
    s = c.cache_similarity_matrix.shape
    return {
        "schema": R137_SCHEMA_VERSION,
        "name": "h203e_v9_cache_similarity_matrix_shape",
        "passed": bool(
            len(s) == 4
            and s[0] == p.config.n_layers
            and s[1] == p.config.n_heads),
        "shape": list(s),
    }


def family_h204_kv_bridge_v9_five_target(
        seed: int) -> dict[str, Any]:
    p = build_default_tiny_substrate_v9(
        seed=int(seed) + 37005)
    proj_v3 = KVBridgeV3Projection.init(
        n_layers=p.config.n_layers, n_heads=p.config.n_heads,
        n_kv_heads=p.config.n_kv_heads,
        n_inject_tokens=3, carrier_dim=8,
        d_head=p.config.d_model // p.config.n_heads,
        seed=int(seed) + 37100)
    proj_v4 = KVBridgeV4Projection.init_from_v3(
        proj_v3, seed_v4=int(seed) + 37101)
    proj_v5 = KVBridgeV5Projection.init_from_v4(
        proj_v4, seed_v5=int(seed) + 37102)
    proj_v6 = KVBridgeV6Projection.init_from_v5(
        proj_v5, seed_v6=int(seed) + 37103)
    proj_v7 = KVBridgeV7Projection.init_from_v6(
        proj_v6, seed_v7=int(seed) + 37104)
    proj_v8 = KVBridgeV8Projection.init_from_v7(
        proj_v7, seed_v8=int(seed) + 37105)
    proj_v9 = KVBridgeV9Projection.init_from_v8(
        proj_v8, seed_v9=int(seed) + 37106)
    rng = _np.random.default_rng(int(seed) + 37107)
    ids = tokenize_bytes_v9("kv9", max_len=4)
    while len(ids) < 4: ids.append(0)
    carriers = [list(rng.standard_normal(8)) for _ in range(4)]
    targets = []
    for i in range(5):
        t = _np.zeros(p.config.vocab_size)
        t[100 + 20 * i] = 1.0 / (i + 1)
        targets.append(t.tolist())
    _, report = fit_kv_bridge_v9_five_target(
        params=p, projection=proj_v9,
        train_carriers=carriers,
        target_delta_logits_stack=targets,
        follow_up_token_ids=ids, n_directions=3)
    return {
        "schema": R137_SCHEMA_VERSION,
        "name": "h204_kv_bridge_v9_five_target",
        "passed": bool(report.n_targets == 5),
        "n_targets": int(report.n_targets),
    }


def family_h204b_kv_bridge_v9_hidden_wins_primary_falsifier(
        seed: int) -> dict[str, Any]:
    w = probe_kv_bridge_v9_hidden_wins_primary_falsifier(
        primary_flag=0.5)
    return {
        "schema": R137_SCHEMA_VERSION,
        "name": (
            "h204b_kv_bridge_v9_hidden_wins_primary_falsifier"),
        "passed": bool(w.falsifier_score == 0.0),
        "score": float(w.falsifier_score),
    }


def family_h205_hsb_v8_five_target(
        seed: int) -> dict[str, Any]:
    p = build_default_tiny_substrate_v9(
        seed=int(seed) + 37200)
    hsb2 = HiddenStateBridgeV2Projection.init(
        target_layers=(1, 3), n_tokens=6, carrier_dim=6,
        d_model=int(p.config.d_model),
        seed=int(seed) + 37201)
    hsb3 = HiddenStateBridgeV3Projection.init_from_v2(
        hsb2, n_heads=int(p.config.n_heads),
        seed_v3=int(seed) + 37202)
    hsb4 = HiddenStateBridgeV4Projection.init_from_v3(
        hsb3, seed_v4=int(seed) + 37203)
    hsb5 = HiddenStateBridgeV5Projection.init_from_v4(
        hsb4, n_positions=3, seed_v5=int(seed) + 37204)
    hsb6 = HiddenStateBridgeV6Projection.init_from_v5(
        hsb5, seed_v6=int(seed) + 37205)
    hsb7 = HiddenStateBridgeV7Projection.init_from_v6(
        hsb6, seed_v7=int(seed) + 37206)
    hsb8 = HiddenStateBridgeV8Projection.init_from_v7(
        hsb7, seed_v8=int(seed) + 37207)
    rng = _np.random.default_rng(int(seed) + 37208)
    ids = tokenize_bytes_v9("hsb", max_len=4)
    while len(ids) < 4: ids.append(0)
    carriers = [list(rng.standard_normal(6)) for _ in range(4)]
    targets = []
    for i in range(5):
        t = _np.zeros(p.config.vocab_size)
        t[200 + 10 * i] = 0.8 / (i + 1)
        targets.append(t.tolist())
    _, report = fit_hsb_v8_five_target(
        params=p.v3_params, projection=hsb8,
        train_carriers=carriers,
        target_delta_logits_stack=targets,
        token_ids=ids)
    return {
        "schema": R137_SCHEMA_VERSION,
        "name": "h205_hsb_v8_five_target",
        "passed": bool(report.n_targets == 5),
        "n_targets": int(report.n_targets),
    }


def family_h205b_hsb_v8_hidden_wins_primary_margin(
        seed: int) -> dict[str, Any]:
    pos = compute_hsb_v8_hidden_wins_primary_margin(
        hidden_residual_l2=0.2,
        kv_residual_l2=0.5,
        prefix_residual_l2=0.4)
    neg = compute_hsb_v8_hidden_wins_primary_margin(
        hidden_residual_l2=0.6,
        kv_residual_l2=0.3,
        prefix_residual_l2=0.4)
    return {
        "schema": R137_SCHEMA_VERSION,
        "name": "h205b_hsb_v8_hidden_wins_primary_margin",
        "passed": bool(pos > 0.0 and neg < 0.0),
        "positive": float(pos), "negative": float(neg),
    }


def family_h206_prefix_v8_drift_curve(
        seed: int) -> dict[str, Any]:
    from coordpy.tiny_substrate_v5 import (
        build_default_tiny_substrate_v5,
    )
    pv5 = build_default_tiny_substrate_v5(
        seed=int(seed) + 37300)
    prompt = [10, 20, 30]
    chain = [[40, 50], [60, 70], [80, 90]]
    segs = [
        [(0, 1, "reuse"), (1, 2, "recompute"),
         (2, 3, "drop")]]
    pred = fit_prefix_drift_curve_predictor_v8(
        params_v5=pv5, prompt_token_ids=prompt,
        train_segment_configs=segs, train_chain=chain,
        roles=["r"])
    return {
        "schema": R137_SCHEMA_VERSION,
        "name": "h206_prefix_v8_drift_curve",
        "passed": bool(
            pred.n_target_steps >= 1
            and pred.converged),
        "n_steps": int(pred.n_target_steps),
    }


def family_h206b_prefix_v8_three_way_decision(
        seed: int) -> dict[str, Any]:
    w_prefix = compare_prefix_vs_hidden_vs_replay_v8(
        prefix_drift_curve=[0.05, 0.05],
        hidden_drift_curve=[0.5, 0.5],
        replay_drift_curve=[0.3, 0.3])
    w_replay = compare_prefix_vs_hidden_vs_replay_v8(
        prefix_drift_curve=[0.5, 0.5],
        hidden_drift_curve=[0.6, 0.6],
        replay_drift_curve=[0.05, 0.05])
    w_hidden = compare_prefix_vs_hidden_vs_replay_v8(
        prefix_drift_curve=[0.5, 0.5],
        hidden_drift_curve=[0.05, 0.05],
        replay_drift_curve=[0.3, 0.3])
    return {
        "schema": R137_SCHEMA_VERSION,
        "name": "h206b_prefix_v8_three_way_decision",
        "passed": bool(
            w_prefix.decision == "prefix_wins"
            and w_replay.decision == "replay_wins"
            and w_hidden.decision == "hidden_wins"),
    }


def family_h207_attention_v8_four_stage(
        seed: int) -> dict[str, Any]:
    p = build_default_tiny_substrate_v9(
        seed=int(seed) + 37400)
    ids = tokenize_bytes_v9("attn", max_len=4)[:4]
    while len(ids) < 4: ids.append(0)
    proj = AttentionSteeringV2Projection.init(
        n_layers=p.config.n_layers, n_heads=p.config.n_heads,
        n_query=len(ids), n_key=len(ids), carrier_dim=6,
        seed=int(seed) + 37401)
    rng = _np.random.default_rng(int(seed) + 37402)
    carrier = list(rng.standard_normal(6))
    w = steer_attention_and_measure_v8(
        params=p.v3_params, carrier=carrier, projection=proj,
        token_ids=ids, hellinger_budget=0.15)
    return {
        "schema": R137_SCHEMA_VERSION,
        "name": "h207_attention_v8_four_stage",
        "passed": bool(
            w.four_stage_used
            and w.hellinger_max_after_four_stage <= 1.0),
        "h_max": float(w.hellinger_max_after_four_stage),
    }


def family_h207b_attention_v8_zero_under_negative(
        seed: int) -> dict[str, Any]:
    p = build_default_tiny_substrate_v9(
        seed=int(seed) + 37410)
    ids = tokenize_bytes_v9("attn-neg", max_len=4)[:4]
    while len(ids) < 4: ids.append(0)
    proj = AttentionSteeringV2Projection.init(
        n_layers=p.config.n_layers, n_heads=p.config.n_heads,
        n_query=len(ids), n_key=len(ids), carrier_dim=6,
        seed=int(seed) + 37411)
    rng = _np.random.default_rng(int(seed) + 37412)
    carrier = list(rng.standard_normal(6))
    w = steer_attention_and_measure_v8(
        params=p.v3_params, carrier=carrier, projection=proj,
        token_ids=ids, hellinger_budget=-1.0)
    return {
        "schema": R137_SCHEMA_VERSION,
        "name": "h207b_attention_v8_zero_under_negative",
        "passed": bool(
            w.attention_map_delta_l2 == 0.0
            and w.hellinger_max_after_four_stage == 0.0),
    }


def family_h208_cache_v7_four_objective(
        seed: int) -> dict[str, Any]:
    rng = _np.random.default_rng(int(seed) + 37500)
    cc = CacheControllerV7.init(fit_seed=int(seed) + 37500)
    X = rng.standard_normal((16, 4))
    y1 = X.sum(axis=-1)
    y2 = X[:, 0]
    y3 = X[:, 1] - X[:, 2]
    y4 = X[:, 3] * 0.5
    _, report = fit_four_objective_ridge_v7(
        controller=cc, train_features=X.tolist(),
        target_drop_oracle=y1.tolist(),
        target_retrieval_relevance=y2.tolist(),
        target_hidden_wins=y3.tolist(),
        target_replay_dominance=y4.tolist())
    return {
        "schema": R137_SCHEMA_VERSION,
        "name": "h208_cache_v7_four_objective",
        "passed": bool(
            report.n_objectives == 4 and report.converged),
    }


def family_h208b_cache_v7_similarity_eviction(
        seed: int) -> dict[str, Any]:
    rng = _np.random.default_rng(int(seed) + 37600)
    cc = CacheControllerV7.init(fit_seed=int(seed) + 37600)
    n = 12
    X = rng.standard_normal((n, 6))
    y = X[:, 0] * 0.4 + X[:, 5] * 0.3
    _, report = fit_similarity_aware_eviction_head_v7(
        controller=cc,
        train_flag_counts=X[:, 0].astype(int).tolist(),
        train_hidden_writes=X[:, 1].tolist(),
        train_replay_ages=X[:, 2].astype(int).tolist(),
        train_attention_receive_l1=X[:, 3].tolist(),
        train_cache_key_norms=_np.abs(X[:, 4]).tolist(),
        train_mean_similarities=_np.abs(X[:, 5]).tolist(),
        target_eviction_priorities=y.tolist())
    return {
        "schema": R137_SCHEMA_VERSION,
        "name": "h208b_cache_v7_similarity_eviction",
        "passed": bool(report.converged),
    }


def family_h209_replay_v5_seven_regimes(
        seed: int) -> dict[str, Any]:
    ctrl = ReplayControllerV5.init()
    cands = {
        r: [ReplayCandidate(
            100, 1000, 50, 0.1, 0.0, 0.3,
            True, True, 0)]
        for r in W64_REPLAY_REGIMES_V5}
    decs = {
        r: [W60_REPLAY_DECISION_REUSE]
        for r in W64_REPLAY_REGIMES_V5}
    _, report = fit_replay_controller_v5_per_regime(
        controller=ctrl, train_candidates_per_regime=cands,
        train_decisions_per_regime=decs)
    return {
        "schema": R137_SCHEMA_VERSION,
        "name": "h209_replay_v5_seven_regimes",
        "passed": bool(
            report.n_regimes == 7 and report.converged),
        "n_regimes": int(report.n_regimes),
    }


def family_h209b_replay_v5_four_way_classifier(
        seed: int) -> dict[str, Any]:
    ctrl = ReplayControllerV5.init()
    rng = _np.random.default_rng(int(seed) + 37700)
    n = 40
    X = rng.standard_normal((n, 9))
    labels = []
    for i in range(n):
        if X[i, 8] > 0.3:
            labels.append("replay_wins")
        elif X[i, 5] > 0.3:
            labels.append("hidden_wins")
        elif X[i, 6] > 0.3:
            labels.append("prefix_wins")
        else:
            labels.append("kv_wins")
    _, audit = fit_four_way_bridge_classifier(
        controller=ctrl, train_features=X.tolist(),
        train_labels=labels)
    return {
        "schema": R137_SCHEMA_VERSION,
        "name": "h209b_replay_v5_four_way_classifier",
        "passed": bool(audit["accuracy_train"] >= 0.7),
        "accuracy": float(audit["accuracy_train"]),
    }


def family_h209c_replay_v5_dominance_primary_head(
        seed: int) -> dict[str, Any]:
    ctrl = ReplayControllerV5.init()
    rng = _np.random.default_rng(int(seed) + 37800)
    n = 30
    X = rng.standard_normal((n, 9))
    decs = []
    for i in range(n):
        if X[i, 7] > 0.0:
            decs.append("choose_reuse")
        else:
            decs.append("choose_recompute")
    _, audit = fit_replay_dominance_primary_head_v5(
        controller=ctrl, train_features=X.tolist(),
        train_decisions=decs)
    return {
        "schema": R137_SCHEMA_VERSION,
        "name": "h209c_replay_v5_dominance_primary_head",
        "passed": bool(audit["converged"]),
    }


def family_h210_substrate_adapter_v9_full_tier(
        seed: int) -> dict[str, Any]:
    cap = probe_tiny_substrate_v9_adapter()
    return {
        "schema": R137_SCHEMA_VERSION,
        "name": "h210_substrate_adapter_v9_full_tier",
        "passed": bool(
            cap.tier == W64_SUBSTRATE_TIER_SUBSTRATE_V9_FULL),
        "tier": str(cap.tier),
    }


def family_h210b_hosted_backends_text_only_at_v9_tier(
        seed: int) -> dict[str, Any]:
    matrix = probe_all_v9_adapters(
        probe_ollama=False, probe_openai=False)
    return {
        "schema": R137_SCHEMA_VERSION,
        "name": "h210b_hosted_backends_text_only_at_v9_tier",
        "passed": bool(matrix.has_v9_full()),
        "n_caps": int(len(matrix.capabilities)),
    }


_R137_FAMILIES: tuple[Any, ...] = (
    family_h203_v9_substrate_determinism,
    family_h203b_v9_hidden_wins_primary_shape,
    family_h203c_v9_replay_dominance_witness_shape,
    family_h203d_v9_attention_entropy_shape,
    family_h203e_v9_cache_similarity_matrix_shape,
    family_h204_kv_bridge_v9_five_target,
    family_h204b_kv_bridge_v9_hidden_wins_primary_falsifier,
    family_h205_hsb_v8_five_target,
    family_h205b_hsb_v8_hidden_wins_primary_margin,
    family_h206_prefix_v8_drift_curve,
    family_h206b_prefix_v8_three_way_decision,
    family_h207_attention_v8_four_stage,
    family_h207b_attention_v8_zero_under_negative,
    family_h208_cache_v7_four_objective,
    family_h208b_cache_v7_similarity_eviction,
    family_h209_replay_v5_seven_regimes,
    family_h209b_replay_v5_four_way_classifier,
    family_h209c_replay_v5_dominance_primary_head,
    family_h210_substrate_adapter_v9_full_tier,
    family_h210b_hosted_backends_text_only_at_v9_tier,
)


def run_r137(
        seeds: Sequence[int] = (199, 299, 399),
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for s in seeds:
        results: dict[str, Any] = {}
        for fn in _R137_FAMILIES:
            try:
                r = fn(int(s))
            except Exception as e:
                r = {
                    "schema": R137_SCHEMA_VERSION,
                    "name": getattr(fn, "__name__", "unknown"),
                    "passed": False,
                    "exception": str(e),
                }
            results[r["name"]] = r
        out.append({
            "schema": R137_SCHEMA_VERSION,
            "seed": int(s),
            "family_results": results,
        })
    return out
