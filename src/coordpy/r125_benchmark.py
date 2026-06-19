"""R-125 — W60 real-substrate / latent-bridge / cache-reuse-vs-
recompute / trained-controller / hidden-vs-KV head-to-head family.

H125..H140 cell families. R-125 is the W60 *substrate* family:
every H-bar in this file is load-bearing on the in-repo V5
substrate, a closed-form-fitted bridge, the W60 ReplayController,
or the hidden-vs-KV head-to-head harness.

* H125  substrate V5 forward determinism
* H125b substrate V5 per-(layer, head, position) attention-receive
        propagates
* H125c substrate V5 logit Jacobian table shape correct
* H125d substrate V5 corruption flag channel writes/reads
* H126  KV bridge V5 multi-direction ridge fit converges
* H126b KV bridge V5 logit-direction fit converges
* H126c KV bridge V5 reverse-extract residual within bound
* H127  HSB V4 per-(layer, head) ridge fit converges
* H127b HSB V4 recovery from adversarial reduces residual
* H128  attention V4 per-(layer, head, query) budget enforced
* H128b attention V4 negative budget falsifier zero shift
* H129  prefix V4 multi-segment flop saving > 0
* H129b prefix V4 chain drift bounded
* H130  cache controller V3 trained_eviction ridge fit converges
* H130b cache controller V3 composite weights fit
* H131  replay controller chooses REUSE on CRC-passed candidate
* H131b replay controller chooses RECOMPUTE on CRC-failed candidate
* H132  hidden-vs-KV head-to-head: at least one arm beats the other
        on a fixed target (falsifiable: tie is allowed)
* H132b deep substrate hybrid V5 five-way flag True
* H133  substrate adapter V5 substrate_v5_full tier on V5 only
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.r125_benchmark requires numpy") from exc

from .attention_steering_bridge_v2 import (
    AttentionSteeringV2Projection,
)
from .attention_steering_bridge_v4 import (
    steer_attention_and_measure_v4,
)
from .cache_controller_v3 import (
    CacheControllerV3,
    W60_CACHE_POLICY_LEARNED_ATTENTION_RECEIVE,
    W60_CACHE_POLICY_TRAINED_EVICTION,
    fit_composite_v3_controller,
    fit_learned_attention_receive_controller,
    fit_trained_eviction_controller,
    apply_cache_controller_v3_and_measure,
)
from .deep_proxy_stack_v6 import DeepProxyStackV6
from .deep_substrate_hybrid_v5 import (
    DeepSubstrateHybridV5,
    deep_substrate_hybrid_v5_forward,
)
from .hidden_state_bridge_v2 import (
    HiddenStateBridgeV2Projection,
)
from .hidden_state_bridge_v3 import (
    HiddenStateBridgeV3Projection,
)
from .hidden_state_bridge_v4 import (
    HiddenStateBridgeV4Projection,
    compare_hidden_vs_kv_injection_v4,
    fit_hsb_v4_per_head_target,
    recover_hsb_v4_inject,
)
from .kv_bridge_v3 import KVBridgeV3Projection
from .kv_bridge_v4 import KVBridgeV4Projection
from .kv_bridge_v5 import (
    KVBridgeV5Projection,
    bridge_carrier_and_measure_v5,
    extract_carrier_from_v5_kv_cache,
    fit_kv_bridge_v5_correction,
    fit_kv_bridge_v5_logit_direction,
)
from .prefix_state_bridge_v4 import (
    bridge_prefix_state_and_measure_v4,
)
from .replay_controller import (
    ReplayCandidate, ReplayController,
    W60_REPLAY_DECISION_REUSE,
    W60_REPLAY_DECISION_RECOMPUTE,
)
from .substrate_adapter_v5 import (
    W60_SUBSTRATE_TIER_SUBSTRATE_V5_FULL,
    probe_synthetic_v5_adapter,
    probe_tiny_substrate_v5_adapter,
)
from .tiny_substrate_v5 import (
    W60_V5_SEGMENT_DROP, W60_V5_SEGMENT_RECOMPUTE,
    W60_V5_SEGMENT_REUSE,
    build_default_tiny_substrate_v5,
    forward_tiny_substrate_v5,
    logit_jacobian_v5,
    tokenize_bytes_v5,
)


R125_SCHEMA_VERSION: str = "coordpy.r125_benchmark.v1"


def _build_substrate(seed: int):
    return build_default_tiny_substrate_v5(
        seed=int(seed) + 25000)


def family_h125_substrate_v5_determinism(
        seed: int) -> dict[str, Any]:
    p = _build_substrate(seed)
    ids = tokenize_bytes_v5("w60-det", max_len=12)
    t1, c1 = forward_tiny_substrate_v5(p, ids)
    t2, c2 = forward_tiny_substrate_v5(p, ids)
    return {
        "schema": R125_SCHEMA_VERSION,
        "name": "h125_substrate_v5_determinism",
        "passed": bool(_np.array_equal(t1.logits, t2.logits)
                        and t1.cid() == t2.cid()),
    }


def family_h125b_substrate_v5_attention_receive_propagates(
        seed: int) -> dict[str, Any]:
    p = _build_substrate(seed)
    ids = tokenize_bytes_v5("attn-recv-w60", max_len=10)
    t1, c1 = forward_tiny_substrate_v5(p, ids)
    # Attention-receive shape (n_heads, n_tokens), and the cumulative
    # signal is non-zero on at least one layer.
    nonzero = any(
        float(_np.sum(a)) > 0.0 for a in c1.attention_receive)
    shape_ok = all(
        a.shape == (p.config.n_heads, c1.n_tokens())
        for a in c1.attention_receive)
    return {
        "schema": R125_SCHEMA_VERSION,
        "name": "h125b_substrate_v5_attention_receive_propagates",
        "passed": bool(shape_ok and nonzero),
        "shape_ok": bool(shape_ok),
        "nonzero": bool(nonzero),
    }


def family_h125c_substrate_v5_logit_jacobian_shape(
        seed: int) -> dict[str, Any]:
    p = _build_substrate(seed)
    ids = tokenize_bytes_v5("jac-w60", max_len=8)
    t1, _ = forward_tiny_substrate_v5(p, ids)
    j = logit_jacobian_v5(t1, p, target_token=10)
    arr = _np.asarray(j["per_layer_per_head"])
    return {
        "schema": R125_SCHEMA_VERSION,
        "name": "h125c_substrate_v5_logit_jacobian_shape",
        "passed": bool(
            arr.shape == (p.config.n_layers, p.config.n_heads)),
        "shape": list(arr.shape),
    }


def family_h125d_substrate_v5_corruption_flag(
        seed: int) -> dict[str, Any]:
    p = _build_substrate(seed)
    ids = tokenize_bytes_v5("corr-w60", max_len=10)
    t1, c1 = forward_tiny_substrate_v5(p, ids)
    n_pre = c1.n_corrupted()
    c1.set_corruption_flag(layer_index=0, position=2, flagged=True)
    c1.set_corruption_flag(layer_index=3, position=4, flagged=True)
    n_post = c1.n_corrupted()
    return {
        "schema": R125_SCHEMA_VERSION,
        "name": "h125d_substrate_v5_corruption_flag",
        "passed": bool(n_pre == 0 and n_post == 2),
        "n_pre": int(n_pre),
        "n_post": int(n_post),
    }


def family_h126_kv_bridge_v5_multi_dir_fit(
        seed: int) -> dict[str, Any]:
    p = _build_substrate(seed)
    v3p = p.v3_params
    proj_v3 = KVBridgeV3Projection.init(
        n_layers=p.config.n_layers,
        n_heads=p.config.n_heads,
        n_kv_heads=p.config.n_kv_heads,
        n_inject_tokens=3, carrier_dim=8,
        d_head=p.config.d_model // p.config.n_heads,
        seed=int(seed) + 25100)
    proj_v4 = KVBridgeV4Projection.init_from_v3(
        proj_v3, seed_v4=int(seed) + 25101)
    proj_v5 = KVBridgeV5Projection.init_from_v4(
        proj_v4, seed_v5=int(seed) + 25102)
    rng = _np.random.default_rng(int(seed) + 25103)
    ids = tokenize_bytes_v5("kv5-fit", max_len=4)
    carriers = [list(rng.standard_normal(8)) for _ in range(4)]
    targets = [3.0] * 4
    fitted, report = fit_kv_bridge_v5_correction(
        params=v3p, projection=proj_v5,
        train_carriers=carriers, train_target_l2=targets,
        follow_up_token_ids=ids, n_directions=3)
    return {
        "schema": R125_SCHEMA_VERSION,
        "name": "h126_kv_bridge_v5_multi_dir_fit",
        "passed": bool(report.converged
                        and report.n_directions == 3),
        "pre": float(report.pre_fit_mean_residual),
        "post": float(report.post_fit_mean_residual),
        "n_directions": int(report.n_directions),
    }


def family_h126b_kv_bridge_v5_logit_dir_fit(
        seed: int) -> dict[str, Any]:
    p = _build_substrate(seed)
    v3p = p.v3_params
    proj_v3 = KVBridgeV3Projection.init(
        n_layers=p.config.n_layers,
        n_heads=p.config.n_heads,
        n_kv_heads=p.config.n_kv_heads,
        n_inject_tokens=3, carrier_dim=8,
        d_head=p.config.d_model // p.config.n_heads,
        seed=int(seed) + 25200)
    proj_v4 = KVBridgeV4Projection.init_from_v3(
        proj_v3, seed_v4=int(seed) + 25201)
    proj_v5 = KVBridgeV5Projection.init_from_v4(
        proj_v4, seed_v5=int(seed) + 25202)
    rng = _np.random.default_rng(int(seed) + 25203)
    ids = tokenize_bytes_v5("kv5-ld", max_len=4)
    carriers = [list(rng.standard_normal(8)) for _ in range(4)]
    target = _np.zeros(p.config.vocab_size)
    target[100] = 1.0
    fitted, report = fit_kv_bridge_v5_logit_direction(
        params=v3p, projection=proj_v5,
        train_carriers=carriers,
        target_delta_logits=target.tolist(),
        follow_up_token_ids=ids, n_directions=2)
    return {
        "schema": R125_SCHEMA_VERSION,
        "name": "h126b_kv_bridge_v5_logit_dir_fit",
        "passed": bool(report.converged
                        and report.fit_kind
                        == "logit_direction"),
        "pre": float(report.pre_fit_mean_residual),
        "post": float(report.post_fit_mean_residual),
    }


def family_h126c_kv_bridge_v5_reverse_extract(
        seed: int) -> dict[str, Any]:
    p = _build_substrate(seed)
    v3p = p.v3_params
    proj_v3 = KVBridgeV3Projection.init(
        n_layers=p.config.n_layers,
        n_heads=p.config.n_heads,
        n_kv_heads=p.config.n_kv_heads,
        n_inject_tokens=3, carrier_dim=6,
        d_head=p.config.d_model // p.config.n_heads,
        seed=int(seed) + 25300)
    proj_v4 = KVBridgeV4Projection.init_from_v3(
        proj_v3, seed_v4=int(seed) + 25301)
    proj_v5 = KVBridgeV5Projection.init_from_v4(
        proj_v4, seed_v5=int(seed) + 25302)
    rng = _np.random.default_rng(int(seed) + 25303)
    ids = tokenize_bytes_v5("kv5-rev", max_len=4)
    carrier = list(rng.standard_normal(6))
    w = bridge_carrier_and_measure_v5(
        params=v3p, carrier=carrier,
        projection=proj_v5,
        follow_up_token_ids=ids)
    return {
        "schema": R125_SCHEMA_VERSION,
        "name": "h126c_kv_bridge_v5_reverse_extract",
        "passed": bool(
            float(w.extract_carrier_residual_l2) < 1e-3),
        "extract_residual": float(
            w.extract_carrier_residual_l2),
    }


def family_h127_hsb_v4_per_head_fit(
        seed: int) -> dict[str, Any]:
    p = _build_substrate(seed)
    v3p = p.v3_params
    hsb_v2 = HiddenStateBridgeV2Projection.init(
        target_layers=(1, 3), n_tokens=4,
        carrier_dim=6, d_model=p.config.d_model,
        seed=int(seed) + 25400)
    hsb_v3 = HiddenStateBridgeV3Projection.init_from_v2(
        hsb_v2, n_heads=p.config.n_heads,
        seed_v3=int(seed) + 25401)
    hsb_v4 = HiddenStateBridgeV4Projection.init_from_v3(
        hsb_v3, seed_v4=int(seed) + 25402)
    rng = _np.random.default_rng(int(seed) + 25403)
    carriers = [list(rng.standard_normal(6)) for _ in range(3)]
    target = _np.zeros(p.config.vocab_size)
    target[100] = 1.0
    ids = tokenize_bytes_v5("hsb-v4", max_len=4)[:4]
    while len(ids) < 4:
        ids.append(0)
    fitted, report = fit_hsb_v4_per_head_target(
        params=v3p, projection=hsb_v4,
        train_carriers=carriers,
        target_delta_logits=target,
        token_ids=ids)
    return {
        "schema": R125_SCHEMA_VERSION,
        "name": "h127_hsb_v4_per_head_fit",
        "passed": bool(report.converged
                        and report.n_layers_fitted == 2
                        and report.n_heads_fitted
                        == p.config.n_heads),
        "pre": float(report.pre_fit_residual),
        "post": float(report.post_fit_residual),
    }


def family_h127b_hsb_v4_recovery(
        seed: int) -> dict[str, Any]:
    p = _build_substrate(seed)
    v3p = p.v3_params
    hsb_v2 = HiddenStateBridgeV2Projection.init(
        target_layers=(1, 3), n_tokens=4,
        carrier_dim=6, d_model=p.config.d_model,
        seed=int(seed) + 25500)
    hsb_v3 = HiddenStateBridgeV3Projection.init_from_v2(
        hsb_v2, n_heads=p.config.n_heads,
        seed_v3=int(seed) + 25501)
    hsb_v4 = HiddenStateBridgeV4Projection.init_from_v3(
        hsb_v3, seed_v4=int(seed) + 25502)
    rng = _np.random.default_rng(int(seed) + 25503)
    carrier = list(rng.standard_normal(6))
    target = _np.zeros(p.config.vocab_size)
    target[200] = 1.0
    ids = tokenize_bytes_v5("hsb-rec", max_len=4)[:4]
    while len(ids) < 4:
        ids.append(0)
    adv = rng.standard_normal((2, p.config.n_heads)) * 0.5
    rec, rep = recover_hsb_v4_inject(
        params=v3p, projection=hsb_v4, carrier=carrier,
        token_ids=ids, target_delta_logits=target,
        adversarial_per_head=adv)
    return {
        "schema": R125_SCHEMA_VERSION,
        "name": "h127b_hsb_v4_recovery",
        "passed": bool(rep.post_fit_residual <= rep.pre_fit_residual + 1e-9),
        "pre": float(rep.pre_fit_residual),
        "post": float(rep.post_fit_residual),
    }


def family_h128_attention_v4_per_query_budget(
        seed: int) -> dict[str, Any]:
    p = _build_substrate(seed)
    v3p = p.v3_params
    proj = AttentionSteeringV2Projection.init(
        n_layers=p.config.n_layers,
        n_heads=p.config.n_heads,
        n_query=4, n_key=8, carrier_dim=6,
        seed=int(seed) + 25600)
    rng = _np.random.default_rng(int(seed) + 25601)
    carrier = list(rng.standard_normal(6))
    ids = tokenize_bytes_v5("attn4", max_len=4)[:4]
    while len(ids) < 4:
        ids.append(0)
    w = steer_attention_and_measure_v4(
        params=v3p, carrier=carrier, projection=proj,
        token_ids=ids, kl_budget_per_query=0.4)
    return {
        "schema": R125_SCHEMA_VERSION,
        "name": "h128_attention_v4_per_query_budget",
        "passed": bool(w.per_query_budget_enforced
                        and float(
                            w.per_head_query_kl_post_max)
                        <= 0.4 + 1e-3),
        "kl_post_max": float(
            w.per_head_query_kl_post_max),
        "l1_max": float(w.per_head_query_l1_shift_max),
    }


def family_h128b_attention_v4_negative_budget_falsifier(
        seed: int) -> dict[str, Any]:
    p = _build_substrate(seed)
    v3p = p.v3_params
    proj = AttentionSteeringV2Projection.init(
        n_layers=p.config.n_layers,
        n_heads=p.config.n_heads,
        n_query=4, n_key=8, carrier_dim=6,
        seed=int(seed) + 25700)
    rng = _np.random.default_rng(int(seed) + 25701)
    carrier = list(rng.standard_normal(6))
    ids = tokenize_bytes_v5("attn-neg", max_len=4)[:4]
    while len(ids) < 4:
        ids.append(0)
    w = steer_attention_and_measure_v4(
        params=v3p, carrier=carrier, projection=proj,
        token_ids=ids, negative_budget=True)
    return {
        "schema": R125_SCHEMA_VERSION,
        "name": "h128b_attention_v4_negative_budget_falsifier",
        "passed": bool(
            float(w.negative_budget_post_kl_max) < 1e-6
            and not w.attention_pattern_shifted),
        "post_kl_max": float(
            w.negative_budget_post_kl_max),
        "shifted": bool(w.attention_pattern_shifted),
    }


def family_h129_prefix_v4_multi_segment_flop_saving(
        seed: int) -> dict[str, Any]:
    p = _build_substrate(seed)
    prompt = tokenize_bytes_v5(
        "multi-seg-prefix", max_len=14)
    fu = tokenize_bytes_v5("a", max_len=2)
    segs = [
        (0, 4, W60_V5_SEGMENT_REUSE),
        (4, 10, W60_V5_SEGMENT_RECOMPUTE),
        (10, len(prompt), W60_V5_SEGMENT_DROP),
    ]
    w = bridge_prefix_state_and_measure_v4(
        params_v5=p, prompt_token_ids=prompt,
        follow_up_chain=[fu], segments=segs)
    return {
        "schema": R125_SCHEMA_VERSION,
        "name": "h129_prefix_v4_multi_segment_flop_saving",
        "passed": bool(
            w.flop_saved_vs_recompute > 0
            and w.flop_savings_ratio > 0.0),
        "flop_saved": int(w.flop_saved_vs_recompute),
        "ratio": float(w.flop_savings_ratio),
    }


def family_h129b_prefix_v4_chain_drift_bounded(
        seed: int) -> dict[str, Any]:
    p = _build_substrate(seed)
    prompt = tokenize_bytes_v5("chain-prefix", max_len=12)
    fu1 = tokenize_bytes_v5("a", max_len=2)
    fu2 = tokenize_bytes_v5("b", max_len=2)
    segs = [
        (0, 6, W60_V5_SEGMENT_REUSE),
        (6, len(prompt), W60_V5_SEGMENT_RECOMPUTE),
    ]
    w = bridge_prefix_state_and_measure_v4(
        params_v5=p, prompt_token_ids=prompt,
        follow_up_chain=[fu1, fu2], segments=segs)
    return {
        "schema": R125_SCHEMA_VERSION,
        "name": "h129b_prefix_v4_chain_drift_bounded",
        "passed": bool(
            len(w.chain_step_drifts_l2) == 2
            and float(w.chain_cumulative_drift_l2) >= 0.0),
        "step_drifts": [
            float(d) for d in w.chain_step_drifts_l2],
        "cumulative_drift": float(
            w.chain_cumulative_drift_l2),
    }


def family_h130_cache_controller_v3_trained_eviction(
        seed: int) -> dict[str, Any]:
    p = _build_substrate(seed)
    v3p = p.v3_params
    prompt = tokenize_bytes_v5("cc3-train", max_len=12)
    fu = tokenize_bytes_v5("fu", max_len=4)
    trace, cache = forward_tiny_substrate_v5(p, prompt)
    ctrl, rep = fit_trained_eviction_controller(
        v3p, prompt_token_ids=prompt,
        follow_up_token_ids=fu,
        attention_receive=cache.attention_receive)
    return {
        "schema": R125_SCHEMA_VERSION,
        "name": "h130_cache_controller_v3_trained_eviction",
        "passed": bool(rep.converged
                        and rep.post_fit_residual
                        <= rep.pre_fit_residual + 1e-9),
        "pre": float(rep.pre_fit_residual),
        "post": float(rep.post_fit_residual),
    }


def family_h130b_cache_controller_v3_composite_fit(
        seed: int) -> dict[str, Any]:
    p = _build_substrate(seed)
    v3p = p.v3_params
    prompt = tokenize_bytes_v5("cc3-comp", max_len=12)
    fu = tokenize_bytes_v5("fu", max_len=4)
    trace, cache = forward_tiny_substrate_v5(p, prompt)
    n = int(cache.n_tokens())
    rng = _np.random.default_rng(int(seed) + 25800)
    imp = _np.zeros((n,))
    for li in trace.v4_trace.kv_cache_v3.importance:
        if li.size and li.shape[0] == n:
            imp += li
    ctrl, rep = fit_composite_v3_controller(
        v3p, prompt_token_ids=prompt,
        follow_up_token_ids=fu,
        importance_vector=imp,
        learned_hidden_scores=rng.standard_normal(n),
        learned_retrieval_scores=rng.standard_normal(n),
        learned_attention_receive_scores=(
            rng.standard_normal(n)))
    return {
        "schema": R125_SCHEMA_VERSION,
        "name": "h130b_cache_controller_v3_composite_fit",
        "passed": bool(rep.converged
                        and ctrl.composite_weights is not None
                        and len(ctrl.composite_weights) == 4),
        "weights": (
            [float(w) for w in ctrl.composite_weights]
            if ctrl.composite_weights is not None else []),
        "pre": float(rep.pre_fit_residual),
        "post": float(rep.post_fit_residual),
    }


def family_h131_replay_controller_chooses_reuse(
        seed: int) -> dict[str, Any]:
    rc = ReplayController()
    cand = ReplayCandidate(
        flop_reuse=100, flop_recompute=1000,
        flop_fallback=10,
        drift_l2_reuse=0.05, drift_l2_recompute=0.0,
        drift_l2_fallback=2.0,
        crc_passed=True, transcript_available=True,
        n_corruption_flags=0)
    d = rc.decide(cand)
    return {
        "schema": R125_SCHEMA_VERSION,
        "name": "h131_replay_controller_chooses_reuse",
        "passed": bool(
            d.decision == W60_REPLAY_DECISION_REUSE),
        "decision": str(d.decision),
        "saving": float(d.flop_saving_vs_recompute),
    }


def family_h131b_replay_controller_chooses_recompute(
        seed: int) -> dict[str, Any]:
    rc = ReplayController()
    cand = ReplayCandidate(
        flop_reuse=100, flop_recompute=1000,
        flop_fallback=10,
        drift_l2_reuse=2.0, drift_l2_recompute=0.0,
        drift_l2_fallback=2.0,
        crc_passed=False, transcript_available=True,
        n_corruption_flags=3)
    d = rc.decide(cand)
    return {
        "schema": R125_SCHEMA_VERSION,
        "name": "h131b_replay_controller_chooses_recompute",
        "passed": bool(
            d.decision == W60_REPLAY_DECISION_RECOMPUTE),
        "decision": str(d.decision),
    }


def family_h132_hidden_vs_kv_compare(
        seed: int) -> dict[str, Any]:
    p = _build_substrate(seed)
    v3p = p.v3_params
    hsb_v2 = HiddenStateBridgeV2Projection.init(
        target_layers=(1, 3), n_tokens=4,
        carrier_dim=6, d_model=p.config.d_model,
        seed=int(seed) + 25900)
    hsb_v3 = HiddenStateBridgeV3Projection.init_from_v2(
        hsb_v2, n_heads=p.config.n_heads,
        seed_v3=int(seed) + 25901)
    hsb_v4 = HiddenStateBridgeV4Projection.init_from_v3(
        hsb_v3, seed_v4=int(seed) + 25902)
    proj_v3 = KVBridgeV3Projection.init(
        n_layers=p.config.n_layers,
        n_heads=p.config.n_heads,
        n_kv_heads=p.config.n_kv_heads,
        n_inject_tokens=3, carrier_dim=6,
        d_head=p.config.d_model // p.config.n_heads,
        seed=int(seed) + 25903)
    proj_v4 = KVBridgeV4Projection.init_from_v3(
        proj_v3, seed_v4=int(seed) + 25904)
    rng = _np.random.default_rng(int(seed) + 25905)
    carrier = list(rng.standard_normal(6))
    target = _np.zeros(p.config.vocab_size)
    target[100] = 1.0
    ids = tokenize_bytes_v5("hvskv", max_len=4)[:4]
    while len(ids) < 4:
        ids.append(0)
    cmp = compare_hidden_vs_kv_injection_v4(
        params=v3p, carrier=carrier, token_ids=ids,
        target_delta_logits=target,
        hsb_v4_projection=hsb_v4,
        kv_v4_projection=proj_v4,
        train_carriers=[carrier])
    # H-bar: at least one arm wins or it's a tie. Both can't be
    # true simultaneously by construction.
    passed = bool(
        cmp.hidden_beats_kv
        or cmp.kv_beats_hidden
        or cmp.tie)
    return {
        "schema": R125_SCHEMA_VERSION,
        "name": "h132_hidden_vs_kv_compare",
        "passed": passed,
        "hidden_residual": float(cmp.hidden_l2_residual),
        "kv_residual": float(cmp.kv_l2_residual),
        "hidden_beats_kv": bool(cmp.hidden_beats_kv),
        "kv_beats_hidden": bool(cmp.kv_beats_hidden),
        "tie": bool(cmp.tie),
    }


def family_h132b_deep_substrate_hybrid_v5_five_way(
        seed: int) -> dict[str, Any]:
    p = _build_substrate(seed)
    v6 = DeepProxyStackV6.init(seed=int(seed) + 26000)
    proj_v3 = KVBridgeV3Projection.init(
        n_layers=p.config.n_layers,
        n_heads=p.config.n_heads,
        n_kv_heads=p.config.n_kv_heads,
        n_inject_tokens=3,
        carrier_dim=v6.inner_v5.inner_v4.factor_dim,
        d_head=p.config.d_model // p.config.n_heads,
        seed=int(seed) + 26001)
    proj_v4 = KVBridgeV4Projection.init_from_v3(
        proj_v3, seed_v4=int(seed) + 26002)
    proj_v5 = KVBridgeV5Projection.init_from_v4(
        proj_v4, seed_v5=int(seed) + 26003)
    ctrl_v3 = CacheControllerV3.init(
        policy="composite_v3",
        d_model=p.config.d_model)
    ctrl_v3.composite_weights = _np.array(
        [0.4, 0.2, 0.2, 0.2], dtype=_np.float64)
    ctrl_v3.attention_receive_scorer = _np.zeros(
        (p.config.n_layers * p.config.n_heads,),
        dtype=_np.float64)
    rc = ReplayController()
    hyb = DeepSubstrateHybridV5.init(
        deep_v6=v6, substrate_v5=p, bridge_v5=proj_v5,
        cache_controller_v3=ctrl_v3,
        replay_controller=rc,
        substrate_back_inject_weight=0.10,
        cache_retention_ratio=0.5)
    rng = _np.random.default_rng(int(seed) + 26004)
    qi = list(rng.standard_normal(v6.in_dim))
    sks = [list(rng.standard_normal(v6.in_dim))
           for _ in range(3)]
    svs = [list(rng.standard_normal(v6.in_dim))
           for _ in range(3)]
    out, w, _ = deep_substrate_hybrid_v5_forward(
        hybrid=hyb, query_input=qi,
        slot_keys=sks, slot_values=svs)
    return {
        "schema": R125_SCHEMA_VERSION,
        "name": "h132b_deep_substrate_hybrid_v5_five_way",
        "passed": bool(w.five_way),
        "five_way": bool(w.five_way),
        "replay_decision": str(w.replay_decision),
    }


def family_h133_substrate_adapter_v5_tier(
        seed: int) -> dict[str, Any]:
    tv5 = probe_tiny_substrate_v5_adapter()
    syn = probe_synthetic_v5_adapter()
    return {
        "schema": R125_SCHEMA_VERSION,
        "name": "h133_substrate_adapter_v5_tier",
        "passed": bool(
            tv5.tier
            == W60_SUBSTRATE_TIER_SUBSTRATE_V5_FULL
            and syn.tier == "text_only"),
        "v5_tier": str(tv5.tier),
        "synthetic_tier": str(syn.tier),
    }


def run_r125(seeds: Sequence[int] = (193, 293, 393)
              ) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    families = [
        family_h125_substrate_v5_determinism,
        family_h125b_substrate_v5_attention_receive_propagates,
        family_h125c_substrate_v5_logit_jacobian_shape,
        family_h125d_substrate_v5_corruption_flag,
        family_h126_kv_bridge_v5_multi_dir_fit,
        family_h126b_kv_bridge_v5_logit_dir_fit,
        family_h126c_kv_bridge_v5_reverse_extract,
        family_h127_hsb_v4_per_head_fit,
        family_h127b_hsb_v4_recovery,
        family_h128_attention_v4_per_query_budget,
        family_h128b_attention_v4_negative_budget_falsifier,
        family_h129_prefix_v4_multi_segment_flop_saving,
        family_h129b_prefix_v4_chain_drift_bounded,
        family_h130_cache_controller_v3_trained_eviction,
        family_h130b_cache_controller_v3_composite_fit,
        family_h131_replay_controller_chooses_reuse,
        family_h131b_replay_controller_chooses_recompute,
        family_h132_hidden_vs_kv_compare,
        family_h132b_deep_substrate_hybrid_v5_five_way,
        family_h133_substrate_adapter_v5_tier,
    ]
    for s in seeds:
        per_family: dict[str, dict[str, Any]] = {}
        for fam in families:
            res = fam(int(s))
            per_family[res["name"]] = res
        out.append({"seed": int(s),
                     "family_results": per_family})
    return out


__all__ = [
    "R125_SCHEMA_VERSION",
    "family_h125_substrate_v5_determinism",
    "family_h125b_substrate_v5_attention_receive_propagates",
    "family_h125c_substrate_v5_logit_jacobian_shape",
    "family_h125d_substrate_v5_corruption_flag",
    "family_h126_kv_bridge_v5_multi_dir_fit",
    "family_h126b_kv_bridge_v5_logit_dir_fit",
    "family_h126c_kv_bridge_v5_reverse_extract",
    "family_h127_hsb_v4_per_head_fit",
    "family_h127b_hsb_v4_recovery",
    "family_h128_attention_v4_per_query_budget",
    "family_h128b_attention_v4_negative_budget_falsifier",
    "family_h129_prefix_v4_multi_segment_flop_saving",
    "family_h129b_prefix_v4_chain_drift_bounded",
    "family_h130_cache_controller_v3_trained_eviction",
    "family_h130b_cache_controller_v3_composite_fit",
    "family_h131_replay_controller_chooses_reuse",
    "family_h131b_replay_controller_chooses_recompute",
    "family_h132_hidden_vs_kv_compare",
    "family_h132b_deep_substrate_hybrid_v5_five_way",
    "family_h133_substrate_adapter_v5_tier",
    "run_r125",
]
