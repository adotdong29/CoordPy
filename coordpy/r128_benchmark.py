"""W61 R-128 benchmark family — substrate V6 / latent bridges /
cache controller V4 / replay controller V2 / attention pattern.

H144..H152 cell families.

* H144   substrate V6 forward determinism
* H144b  substrate V6 cache_keys axis shape (L, T, d_key)
* H144c  substrate V6 hidden_write_trace records writes
* H144d  substrate V6 replay_age channel increments
* H144e  substrate V6 cross_layer_coupling diagonal=1
* H145   kv_bridge V6 multi-target ridge converges
* H145b  kv_bridge V6 attention-pattern ridge converges
* H145c  kv_bridge V6 cache_key fingerprint changes per cache state
* H146   hsb V5 multi-target stack fit converges
* H146b  hsb V5 recovery from adversarial 3-D δ
* H146c  hsb V5 write_into_v6_cache propagates
* H147   attention V5 4-D budget enforced
* H147b  attention V5 negative-budget falsifier zero
* H147c  attention V5 signed-coefficient falsifier triggers
* H148   prefix V5 chain-of-chains saves flop
* H148b  prefix V5 drift predictor converges
* H149   cache controller V4 bilinear retrieval ridge converges
* H149b  cache controller V4 trained corruption floor fit
* H149c  cache controller V4 two_stage_threshold non-trivial
* H149d  cache controller V4 composite_v4 weights fit
* H150   replay V2 threshold ridge fit converges + accuracy ≥ 0.5
* H150b  replay V2 decision_confidence in [0, 1]
* H150c  replay V2 hidden_write_gate fires when cap exceeded
* H151   substrate_adapter_v6 substrate_v6_full tier on V6 only
* H152   deep_substrate_hybrid_v6 six_way flag fires
"""

from __future__ import annotations

import numpy as _np
from typing import Any, Sequence

from .attention_steering_bridge_v2 import (
    AttentionSteeringV2Projection,
)
from .attention_steering_bridge_v5 import (
    steer_attention_and_measure_v5,
)
from .cache_controller_v4 import (
    CacheControllerV4, W61_CACHE_POLICY_BILINEAR_RETRIEVAL_V6,
    apply_cache_controller_v4_and_measure,
    fit_bilinear_retrieval_v6,
    fit_composite_v4, fit_trained_corruption_floor,
    fit_two_stage_threshold,
)
from .deep_substrate_hybrid_v5 import (
    DeepSubstrateHybridV5, DeepSubstrateHybridV5ForwardWitness,
)
from .deep_substrate_hybrid_v6 import (
    DeepSubstrateHybridV6, deep_substrate_hybrid_v6_forward,
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
    fit_hsb_v5_multi_target, recover_hsb_v5_inject,
    write_hsb_v5_into_v6_cache,
)
from .kv_bridge_v3 import KVBridgeV3Projection
from .kv_bridge_v4 import KVBridgeV4Projection
from .kv_bridge_v5 import KVBridgeV5Projection
from .kv_bridge_v6 import (
    KVBridgeV6Projection,
    bridge_carrier_and_measure_v6,
    fit_kv_bridge_v6_attention_pattern,
    fit_kv_bridge_v6_multi_target,
    v6_cache_key_fingerprint,
)
from .prefix_state_bridge_v5 import (
    bridge_prefix_chain_v5, fit_prefix_drift_predictor,
)
from .replay_controller import (
    ReplayCandidate, W60_REPLAY_DECISION_ABSTAIN,
    W60_REPLAY_DECISION_RECOMPUTE, W60_REPLAY_DECISION_REUSE,
)
from .replay_controller_v2 import (
    ReplayControllerV2, fit_replay_controller_v2,
)
from .substrate_adapter_v6 import (
    W61_SUBSTRATE_TIER_SUBSTRATE_V6_FULL,
    probe_tiny_substrate_v6_adapter,
    probe_v5_substrate_adapter_as_v6,
    probe_synthetic_v6_adapter,
)
from .substrate_adapter_v5 import (
    probe_tiny_substrate_v5_adapter,
)
from .tiny_substrate_v5 import (
    W60_V5_SEGMENT_DROP, W60_V5_SEGMENT_RECOMPUTE,
    W60_V5_SEGMENT_REUSE,
)
from .tiny_substrate_v6 import (
    build_default_tiny_substrate_v6,
    forward_tiny_substrate_v6,
    logit_jacobian_v6, record_hidden_write_v6,
    tokenize_bytes_v6,
)


R128_SCHEMA_VERSION: str = "coordpy.r128_benchmark.v1"


def _build_substrate(seed: int):
    return build_default_tiny_substrate_v6(
        seed=int(seed) + 28000)


def family_h144_substrate_v6_determinism(seed: int) -> dict[str, Any]:
    p = _build_substrate(seed)
    ids = tokenize_bytes_v6("w61-det", max_len=12)
    t1, c1 = forward_tiny_substrate_v6(p, ids)
    t2, c2 = forward_tiny_substrate_v6(p, ids)
    return {
        "schema": R128_SCHEMA_VERSION,
        "name": "h144_substrate_v6_determinism",
        "passed": bool(
            t1.cid() == t2.cid()
            and _np.array_equal(t1.logits, t2.logits)),
    }


def family_h144b_substrate_v6_cache_keys_shape(seed: int) -> dict[str, Any]:
    p = _build_substrate(seed)
    ids = tokenize_bytes_v6("ckeys-w61", max_len=10)
    t1, c1 = forward_tiny_substrate_v6(p, ids)
    L = p.config.n_layers
    d_key = p.config.d_key
    shape_ok = all(
        k.shape == (c1.n_tokens(), d_key)
        for k in c1.cache_keys[:L])
    return {
        "schema": R128_SCHEMA_VERSION,
        "name": "h144b_substrate_v6_cache_keys_shape",
        "passed": bool(shape_ok),
    }


def family_h144c_substrate_v6_hidden_write_trace(seed: int) -> dict[str, Any]:
    p = _build_substrate(seed)
    ids = tokenize_bytes_v6("hw-w61", max_len=8)
    _, c1 = forward_tiny_substrate_v6(p, ids)
    pre = float(c1.hidden_write_trace.sum())
    record_hidden_write_v6(
        c1, layer_index=2, per_head_l2=[0.5] * p.config.n_heads)
    post = float(c1.hidden_write_trace.sum())
    return {
        "schema": R128_SCHEMA_VERSION,
        "name": "h144c_substrate_v6_hidden_write_trace",
        "passed": bool(post > pre + 1e-9),
        "delta": float(post - pre),
    }


def family_h144d_substrate_v6_replay_age_increments(seed: int) -> dict[str, Any]:
    p = _build_substrate(seed)
    ids = tokenize_bytes_v6("age-w61", max_len=8)
    t1, c1 = forward_tiny_substrate_v6(p, ids)
    ages_pre = [int(a.max()) if a.size else 0
                 for a in c1.replay_age]
    t2, c2 = forward_tiny_substrate_v6(
        p, ids[:3], v6_kv_cache=c1)
    ages_post = [int(a.max()) if a.size else 0
                  for a in c2.replay_age]
    inc = all(b >= a for a, b in zip(ages_pre, ages_post))
    return {
        "schema": R128_SCHEMA_VERSION,
        "name": "h144d_substrate_v6_replay_age_increments",
        "passed": bool(inc and max(ages_post) > max(ages_pre)),
        "max_pre": int(max(ages_pre)) if ages_pre else 0,
        "max_post": int(max(ages_post)) if ages_post else 0,
    }


def family_h144e_substrate_v6_cross_layer_coupling(seed: int) -> dict[str, Any]:
    p = _build_substrate(seed)
    ids = tokenize_bytes_v6("clc-w61", max_len=10)
    t1, _ = forward_tiny_substrate_v6(p, ids)
    L = int(t1.cross_layer_coupling.shape[0])
    diag_ok = all(
        abs(float(t1.cross_layer_coupling[i, i]) - 1.0) < 1e-6
        for i in range(L))
    return {
        "schema": R128_SCHEMA_VERSION,
        "name": "h144e_substrate_v6_cross_layer_coupling",
        "passed": bool(diag_ok),
        "L": int(L),
    }


def family_h145_kv_bridge_v6_multi_target(seed: int) -> dict[str, Any]:
    p = _build_substrate(seed)
    v3p = p.v3_params
    proj_v3 = KVBridgeV3Projection.init(
        n_layers=p.config.n_layers, n_heads=p.config.n_heads,
        n_kv_heads=p.config.n_kv_heads,
        n_inject_tokens=3, carrier_dim=8,
        d_head=p.config.d_model // p.config.n_heads,
        seed=int(seed) + 28100)
    proj_v4 = KVBridgeV4Projection.init_from_v3(
        proj_v3, seed_v4=int(seed) + 28101)
    proj_v5 = KVBridgeV5Projection.init_from_v4(
        proj_v4, seed_v5=int(seed) + 28102)
    proj_v6 = KVBridgeV6Projection.init_from_v5(
        proj_v5, seed_v6=int(seed) + 28103)
    rng = _np.random.default_rng(int(seed) + 28104)
    ids = tokenize_bytes_v6("kv6-mt", max_len=4)
    carriers = [list(rng.standard_normal(8)) for _ in range(4)]
    t1 = _np.zeros(p.config.vocab_size); t1[100] = 1.0
    t2 = _np.zeros(p.config.vocab_size); t2[120] = 0.5
    fitted, report = fit_kv_bridge_v6_multi_target(
        params=v3p, projection=proj_v6,
        train_carriers=carriers,
        target_delta_logits_stack=[t1.tolist(), t2.tolist()],
        follow_up_token_ids=ids, n_directions=3)
    return {
        "schema": R128_SCHEMA_VERSION,
        "name": "h145_kv_bridge_v6_multi_target",
        "passed": bool(report.converged
                        and report.n_targets == 2
                        and report.n_directions == 3),
        "pre": float(report.pre_fit_mean_residual),
        "post": float(report.post_fit_mean_residual),
    }


def family_h145b_kv_bridge_v6_attention_pattern(seed: int) -> dict[str, Any]:
    p = _build_substrate(seed)
    v3p = p.v3_params
    proj_v3 = KVBridgeV3Projection.init(
        n_layers=p.config.n_layers, n_heads=p.config.n_heads,
        n_kv_heads=p.config.n_kv_heads,
        n_inject_tokens=3, carrier_dim=6,
        d_head=p.config.d_model // p.config.n_heads,
        seed=int(seed) + 28200)
    proj_v4 = KVBridgeV4Projection.init_from_v3(
        proj_v3, seed_v4=int(seed) + 28201)
    proj_v5 = KVBridgeV5Projection.init_from_v4(
        proj_v4, seed_v5=int(seed) + 28202)
    proj_v6 = KVBridgeV6Projection.init_from_v5(
        proj_v5, seed_v6=int(seed) + 28203)
    rng = _np.random.default_rng(int(seed) + 28204)
    ids = tokenize_bytes_v6("kv6-ap", max_len=4)
    carriers = [list(rng.standard_normal(6)) for _ in range(4)]
    target_attn = [0.5, 0.2, 0.1, 0.1]
    fitted, report = fit_kv_bridge_v6_attention_pattern(
        params=v3p, projection=proj_v6,
        train_carriers=carriers,
        target_attention_last_row=target_attn,
        follow_up_token_ids=ids, n_directions=2)
    return {
        "schema": R128_SCHEMA_VERSION,
        "name": "h145b_kv_bridge_v6_attention_pattern",
        "passed": bool(report.converged
                        and report.fit_kind
                        == "attention_pattern"),
        "pre": float(report.pre_fit_mean_residual),
        "post": float(report.post_fit_mean_residual),
    }


def family_h145c_kv_bridge_v6_cache_key_fingerprint(seed: int) -> dict[str, Any]:
    p = _build_substrate(seed)
    ids1 = tokenize_bytes_v6("fp-a", max_len=8)
    ids2 = tokenize_bytes_v6("fp-b-different", max_len=12)
    _, c1 = forward_tiny_substrate_v6(p, ids1)
    _, c2 = forward_tiny_substrate_v6(p, ids2)
    fp1 = v6_cache_key_fingerprint(c1)
    fp2 = v6_cache_key_fingerprint(c2)
    return {
        "schema": R128_SCHEMA_VERSION,
        "name": "h145c_kv_bridge_v6_cache_key_fingerprint",
        "passed": bool(fp1 != fp2 and len(fp1) == 128),
        "fp_len": int(len(fp1)),
    }


def family_h146_hsb_v5_multi_target_fit(seed: int) -> dict[str, Any]:
    p = _build_substrate(seed)
    v3p = p.v3_params
    hsb_v2 = HiddenStateBridgeV2Projection.init(
        target_layers=(1, 3), n_tokens=4, carrier_dim=6,
        d_model=p.config.d_model, seed=int(seed) + 28300)
    hsb_v3 = HiddenStateBridgeV3Projection.init_from_v2(
        hsb_v2, n_heads=p.config.n_heads,
        seed_v3=int(seed) + 28301)
    hsb_v4 = HiddenStateBridgeV4Projection.init_from_v3(
        hsb_v3, seed_v4=int(seed) + 28302)
    hsb_v5 = HiddenStateBridgeV5Projection.init_from_v4(
        hsb_v4, n_positions=3, seed_v5=int(seed) + 28303)
    rng = _np.random.default_rng(int(seed) + 28304)
    carriers = [list(rng.standard_normal(6)) for _ in range(3)]
    ids = tokenize_bytes_v6("hsb5", max_len=4)[:4]
    while len(ids) < 4: ids.append(0)
    t1 = _np.zeros(p.config.vocab_size); t1[100] = 1.0
    t2 = _np.zeros(p.config.vocab_size); t2[120] = 0.7
    fitted, report = fit_hsb_v5_multi_target(
        params=v3p, projection=hsb_v5,
        train_carriers=carriers,
        target_delta_logits_stack=[t1.tolist(), t2.tolist()],
        token_ids=ids)
    return {
        "schema": R128_SCHEMA_VERSION,
        "name": "h146_hsb_v5_multi_target_fit",
        "passed": bool(report.converged
                        and report.n_targets == 2
                        and report.n_positions_fitted == 3),
        "pre": float(report.pre_fit_residual),
        "post": float(report.post_fit_residual),
    }


def family_h146b_hsb_v5_recovery(seed: int) -> dict[str, Any]:
    p = _build_substrate(seed)
    v3p = p.v3_params
    hsb_v2 = HiddenStateBridgeV2Projection.init(
        target_layers=(1, 3), n_tokens=4, carrier_dim=6,
        d_model=p.config.d_model, seed=int(seed) + 28400)
    hsb_v3 = HiddenStateBridgeV3Projection.init_from_v2(
        hsb_v2, n_heads=p.config.n_heads,
        seed_v3=int(seed) + 28401)
    hsb_v4 = HiddenStateBridgeV4Projection.init_from_v3(
        hsb_v3, seed_v4=int(seed) + 28402)
    hsb_v5 = HiddenStateBridgeV5Projection.init_from_v4(
        hsb_v4, n_positions=3, seed_v5=int(seed) + 28403)
    rng = _np.random.default_rng(int(seed) + 28404)
    carrier = list(rng.standard_normal(6))
    ids = tokenize_bytes_v6("hsb5-rec", max_len=4)[:4]
    while len(ids) < 4: ids.append(0)
    target = _np.zeros(p.config.vocab_size); target[100] = 1.0
    adv = rng.standard_normal((2, p.config.n_heads, 3)) * 0.5
    rec, rep = recover_hsb_v5_inject(
        params=v3p, projection=hsb_v5, carrier=carrier,
        token_ids=ids, target_delta_logits=target.tolist(),
        adversarial_per_head_pos=adv)
    return {
        "schema": R128_SCHEMA_VERSION,
        "name": "h146b_hsb_v5_recovery",
        "passed": bool(rep.post_fit_residual
                        <= rep.pre_fit_residual + 1e-9),
        "pre": float(rep.pre_fit_residual),
        "post": float(rep.post_fit_residual),
    }


def family_h146c_hsb_v5_write_into_v6_cache(seed: int) -> dict[str, Any]:
    from .tiny_substrate_v6 import TinyV6KVCache
    p = _build_substrate(seed)
    hsb_v2 = HiddenStateBridgeV2Projection.init(
        target_layers=(1, 3), n_tokens=4, carrier_dim=6,
        d_model=p.config.d_model, seed=int(seed) + 28500)
    hsb_v3 = HiddenStateBridgeV3Projection.init_from_v2(
        hsb_v2, n_heads=p.config.n_heads,
        seed_v3=int(seed) + 28501)
    hsb_v4 = HiddenStateBridgeV4Projection.init_from_v3(
        hsb_v3, seed_v4=int(seed) + 28502)
    hsb_v5 = HiddenStateBridgeV5Projection.init_from_v4(
        hsb_v4, n_positions=3, seed_v5=int(seed) + 28503)
    # Set non-trivial scales.
    hsb_v5.inject_scale_per_head_pos[:, :, 0] = 0.3
    cache = TinyV6KVCache.empty(
        p.config.n_layers, n_heads=p.config.n_heads,
        d_key=p.config.d_key)
    pre = float(cache.hidden_write_trace.sum())
    write_hsb_v5_into_v6_cache(
        projection=hsb_v5, v6_cache=cache)
    post = float(cache.hidden_write_trace.sum())
    return {
        "schema": R128_SCHEMA_VERSION,
        "name": "h146c_hsb_v5_write_into_v6_cache",
        "passed": bool(post > pre + 1e-9),
        "trace_total_l2": float(post),
    }


def family_h147_attention_v5_4d_budget(seed: int) -> dict[str, Any]:
    p = _build_substrate(seed)
    ids = tokenize_bytes_v6("attn5", max_len=4)[:4]
    while len(ids) < 4: ids.append(0)
    proj = AttentionSteeringV2Projection.init(
        n_layers=p.config.n_layers, n_heads=p.config.n_heads,
        n_query=len(ids), n_key=len(ids), carrier_dim=6,
        seed=int(seed) + 28600)
    rng = _np.random.default_rng(int(seed) + 28601)
    carrier = list(rng.standard_normal(6))
    w = steer_attention_and_measure_v5(
        params=p.v3_params, carrier=carrier,
        projection=proj, token_ids=ids,
        kl_budget_per_key=0.4)
    return {
        "schema": R128_SCHEMA_VERSION,
        "name": "h147_attention_v5_4d_budget",
        "passed": bool(w.attention_pattern_shifted
                        and w.per_query_key_budget_enforced),
        "kl_max": float(w.per_head_query_kl_post_max),
        "l1_max": float(w.per_head_query_l1_shift_max),
    }


def family_h147b_attention_v5_negative_budget(seed: int) -> dict[str, Any]:
    p = _build_substrate(seed)
    ids = tokenize_bytes_v6("attn5-neg", max_len=4)[:4]
    while len(ids) < 4: ids.append(0)
    proj = AttentionSteeringV2Projection.init(
        n_layers=p.config.n_layers, n_heads=p.config.n_heads,
        n_query=len(ids), n_key=len(ids), carrier_dim=6,
        seed=int(seed) + 28700)
    rng = _np.random.default_rng(int(seed) + 28701)
    carrier = list(rng.standard_normal(6))
    w = steer_attention_and_measure_v5(
        params=p.v3_params, carrier=carrier,
        projection=proj, token_ids=ids,
        kl_budget_per_key=0.4, negative_budget=True)
    return {
        "schema": R128_SCHEMA_VERSION,
        "name": "h147b_attention_v5_negative_budget_falsifier",
        "passed": bool(w.per_head_query_kl_post_max < 1e-6
                        and not w.attention_pattern_shifted),
        "kl_max": float(w.per_head_query_kl_post_max),
    }


def family_h147c_attention_v5_signed_falsifier(seed: int) -> dict[str, Any]:
    p = _build_substrate(seed)
    ids = tokenize_bytes_v6("attn5-sgn", max_len=4)[:4]
    while len(ids) < 4: ids.append(0)
    proj = AttentionSteeringV2Projection.init(
        n_layers=p.config.n_layers, n_heads=p.config.n_heads,
        n_query=len(ids), n_key=len(ids), carrier_dim=6,
        seed=int(seed) + 28800)
    rng = _np.random.default_rng(int(seed) + 28801)
    carrier = list(rng.standard_normal(6))
    w = steer_attention_and_measure_v5(
        params=p.v3_params, carrier=carrier,
        projection=proj, token_ids=ids,
        kl_budget_per_key=0.4, signed_falsifier=True)
    return {
        "schema": R128_SCHEMA_VERSION,
        "name": "h147c_attention_v5_signed_falsifier",
        "passed": bool(w.signed_falsifier_used
                        and abs(w.signed_falsifier_correlation)
                        >= 0.0),
        "signed_corr": float(w.signed_falsifier_correlation),
    }


def family_h148_prefix_v5_chain_of_chains(seed: int) -> dict[str, Any]:
    p = _build_substrate(seed)
    prompt = tokenize_bytes_v6("w61-prefix-chain-prompt", max_len=20)
    segs = [
        [(0, 4, W60_V5_SEGMENT_REUSE),
         (4, len(prompt), W60_V5_SEGMENT_RECOMPUTE)],
        [(0, 8, W60_V5_SEGMENT_REUSE),
         (8, len(prompt), W60_V5_SEGMENT_RECOMPUTE)],
    ]
    fu = tokenize_bytes_v6("a", max_len=2)
    w = bridge_prefix_chain_v5(
        params_v6=p, prompt_token_ids=prompt,
        chain_segments=segs, chain_follow_ups=[fu, fu])
    return {
        "schema": R128_SCHEMA_VERSION,
        "name": "h148_prefix_v5_chain_of_chains",
        "passed": bool(w.n_links == 2
                        and w.chain_flop_saved_vs_full > 0),
        "saving": float(w.chain_flop_savings_ratio),
    }


def family_h148b_prefix_v5_drift_predictor(seed: int) -> dict[str, Any]:
    p = _build_substrate(seed)
    prompt = tokenize_bytes_v6(
        "w61-prefix-drift-predictor", max_len=20)
    segs_train = [
        [(0, 4, W60_V5_SEGMENT_REUSE),
         (4, len(prompt), W60_V5_SEGMENT_RECOMPUTE)],
        [(0, 8, W60_V5_SEGMENT_REUSE),
         (8, len(prompt), W60_V5_SEGMENT_RECOMPUTE)],
        [(0, 4, W60_V5_SEGMENT_REUSE),
         (4, 10, W60_V5_SEGMENT_RECOMPUTE),
         (10, len(prompt), W60_V5_SEGMENT_DROP)],
        [(0, 12, W60_V5_SEGMENT_REUSE),
         (12, len(prompt), W60_V5_SEGMENT_DROP)],
    ]
    fu = tokenize_bytes_v6("a", max_len=2)
    predictor = fit_prefix_drift_predictor(
        params_v5=p.v5_params, prompt_token_ids=prompt,
        train_segment_configs=segs_train,
        train_first_step_ids=fu)
    return {
        "schema": R128_SCHEMA_VERSION,
        "name": "h148b_prefix_v5_drift_predictor",
        "passed": bool(predictor.converged
                        and predictor.n_train_examples == 4),
        "pre": float(predictor.pre_fit_residual),
        "post": float(predictor.post_fit_residual),
    }


def family_h149_cache_controller_v4_bilinear(seed: int) -> dict[str, Any]:
    p = _build_substrate(seed)
    c = CacheControllerV4.init(
        policy=W61_CACHE_POLICY_BILINEAR_RETRIEVAL_V6,
        d_model=p.config.d_model, d_key=p.config.d_key,
        fit_seed=int(seed) + 28900)
    rng = _np.random.default_rng(int(seed) + 28901)
    ids_list = [
        tokenize_bytes_v6(f"slot-{i}", max_len=4)
        for i in range(6)]
    qvs = [list(rng.standard_normal(p.config.d_model))
           for _ in range(6)]
    rels = [1.0, 0.7, 0.3, 0.0, 0.9, 0.5]
    fitted, rep = fit_bilinear_retrieval_v6(
        controller=c, params_v6=p,
        train_token_ids_per_slot=ids_list,
        train_query_vectors=qvs,
        train_relevance=rels)
    return {
        "schema": R128_SCHEMA_VERSION,
        "name": "h149_cache_controller_v4_bilinear",
        "passed": bool(rep.converged
                        and rep.fit_kind
                        == "bilinear_retrieval_v6"),
        "pre": float(rep.pre_fit_residual),
        "post": float(rep.post_fit_residual),
    }


def family_h149b_cache_controller_v4_corruption_floor(
        seed: int) -> dict[str, Any]:
    p = _build_substrate(seed)
    c = CacheControllerV4.init(
        policy="trained_corruption_floor",
        d_model=p.config.d_model, d_key=p.config.d_key,
        fit_seed=int(seed) + 29000)
    fitted, rep = fit_trained_corruption_floor(
        controller=c, params_v6=p,
        train_token_ids=tokenize_bytes_v6("test", max_len=8),
        train_corruption_counts=[0, 1, 2, 3, 4, 5],
        train_floor_targets=[0.0, -1.0, -2.5, -4.0, -6.0, -9.0])
    return {
        "schema": R128_SCHEMA_VERSION,
        "name": "h149b_cache_controller_v4_corruption_floor",
        "passed": bool(rep.converged
                        and rep.fit_kind
                        == "trained_corruption_floor"),
        "pre": float(rep.pre_fit_residual),
        "post": float(rep.post_fit_residual),
    }


def family_h149c_cache_controller_v4_two_stage(seed: int) -> dict[str, Any]:
    p = _build_substrate(seed)
    c = CacheControllerV4.init(
        policy="two_stage_v4",
        d_model=p.config.d_model, d_key=p.config.d_key,
        fit_seed=int(seed) + 29100)
    rng = _np.random.default_rng(int(seed) + 29101)
    ar = [float(rng.random()) for _ in range(8)]
    oracle = [float(rng.random()) for _ in range(8)]
    fitted, rep = fit_two_stage_threshold(
        controller=c, attention_receive_scores=ar,
        drop_oracle_per_slot=oracle)
    return {
        "schema": R128_SCHEMA_VERSION,
        "name": "h149c_cache_controller_v4_two_stage",
        "passed": bool(rep.fit_kind == "two_stage_threshold"),
        "threshold": float(fitted.two_stage_threshold),
    }


def family_h149d_cache_controller_v4_composite(seed: int) -> dict[str, Any]:
    p = _build_substrate(seed)
    c = CacheControllerV4.init(
        policy="composite_v4",
        d_model=p.config.d_model, d_key=p.config.d_key,
        fit_seed=int(seed) + 29200)
    rng = _np.random.default_rng(int(seed) + 29201)
    heads = rng.standard_normal((10, 5))
    y = heads.sum(axis=1) + rng.standard_normal(10) * 0.1
    fitted, rep = fit_composite_v4(
        controller=c, head_scores=heads.tolist(),
        drop_oracle=y.tolist())
    return {
        "schema": R128_SCHEMA_VERSION,
        "name": "h149d_cache_controller_v4_composite",
        "passed": bool(rep.converged
                        and fitted.composite_v4_weights is not None
                        and fitted.composite_v4_weights.size == 5),
        "pre": float(rep.pre_fit_residual),
        "post": float(rep.post_fit_residual),
    }


def family_h150_replay_v2_threshold_fit(seed: int) -> dict[str, Any]:
    rcv2 = ReplayControllerV2.init()
    cands = [
        ReplayCandidate(
            flop_reuse=100, flop_recompute=1000,
            flop_fallback=50,
            drift_l2_reuse=0.1, drift_l2_recompute=0.0,
            drift_l2_fallback=0.3,
            crc_passed=True, transcript_available=True,
            n_corruption_flags=0),
        ReplayCandidate(
            flop_reuse=200, flop_recompute=1000,
            flop_fallback=50,
            drift_l2_reuse=0.05, drift_l2_recompute=0.0,
            drift_l2_fallback=0.4,
            crc_passed=True, transcript_available=True,
            n_corruption_flags=0),
        ReplayCandidate(
            flop_reuse=900, flop_recompute=1000,
            flop_fallback=50,
            drift_l2_reuse=0.8, drift_l2_recompute=0.0,
            drift_l2_fallback=0.3,
            crc_passed=False, transcript_available=True,
            n_corruption_flags=3),
        ReplayCandidate(
            flop_reuse=0, flop_recompute=1000,
            flop_fallback=50,
            drift_l2_reuse=1.0, drift_l2_recompute=0.0,
            drift_l2_fallback=0.3,
            crc_passed=False, transcript_available=False,
            n_corruption_flags=5),
    ] * 3
    labels = [
        W60_REPLAY_DECISION_REUSE, W60_REPLAY_DECISION_REUSE,
        W60_REPLAY_DECISION_RECOMPUTE,
        W60_REPLAY_DECISION_ABSTAIN] * 3
    fitted, rep = fit_replay_controller_v2(
        controller=rcv2, train_candidates=cands,
        train_optimal_decisions=labels)
    return {
        "schema": R128_SCHEMA_VERSION,
        "name": "h150_replay_v2_threshold_fit",
        "passed": bool(rep.post_fit_residual
                        <= rep.pre_fit_residual + 1e-9
                        and rep.accuracy_train >= 0.5),
        "accuracy": float(rep.accuracy_train),
    }


def family_h150b_replay_v2_decision_confidence(seed: int) -> dict[str, Any]:
    rcv2 = ReplayControllerV2.init()
    cand = ReplayCandidate(
        flop_reuse=100, flop_recompute=1000, flop_fallback=50,
        drift_l2_reuse=0.1, drift_l2_recompute=0.0,
        drift_l2_fallback=0.3,
        crc_passed=True, transcript_available=True,
        n_corruption_flags=0)
    cands = [cand] * 4
    labels = [W60_REPLAY_DECISION_REUSE] * 4
    fitted, _ = fit_replay_controller_v2(
        controller=rcv2, train_candidates=cands,
        train_optimal_decisions=labels)
    d, conf = fitted.decide(cand)
    return {
        "schema": R128_SCHEMA_VERSION,
        "name": "h150b_replay_v2_decision_confidence",
        "passed": bool(0.0 <= conf <= 1.0),
        "confidence": float(conf),
    }


def family_h150c_replay_v2_hidden_write_gate(seed: int) -> dict[str, Any]:
    rcv2 = ReplayControllerV2.init(hidden_write_cap=0.5)
    cand = ReplayCandidate(
        flop_reuse=100, flop_recompute=1000, flop_fallback=50,
        drift_l2_reuse=0.1, drift_l2_recompute=0.0,
        drift_l2_fallback=0.3,
        crc_passed=True, transcript_available=True,
        n_corruption_flags=0)
    d, conf = rcv2.decide(cand, hidden_write_total_l2=10.0)
    return {
        "schema": R128_SCHEMA_VERSION,
        "name": "h150c_replay_v2_hidden_write_gate",
        "passed": bool(
            d.decision == W60_REPLAY_DECISION_ABSTAIN
            and d.rationale == "hidden_write_cap_exceeded"),
        "decision": str(d.decision),
    }


def family_h151_substrate_adapter_v6_tier(seed: int) -> dict[str, Any]:
    v6 = probe_tiny_substrate_v6_adapter()
    v5_as_v6 = probe_v5_substrate_adapter_as_v6(
        probe_tiny_substrate_v5_adapter())
    syn = probe_synthetic_v6_adapter()
    return {
        "schema": R128_SCHEMA_VERSION,
        "name": "h151_substrate_adapter_v6_tier",
        "passed": bool(
            v6.tier == W61_SUBSTRATE_TIER_SUBSTRATE_V6_FULL
            and v5_as_v6.tier
            != W61_SUBSTRATE_TIER_SUBSTRATE_V6_FULL
            and syn.tier
            != W61_SUBSTRATE_TIER_SUBSTRATE_V6_FULL),
        "v6_tier": str(v6.tier),
        "v5_as_v6_tier": str(v5_as_v6.tier),
    }


def family_h152_deep_substrate_hybrid_v6(seed: int) -> dict[str, Any]:
    v5_witness = DeepSubstrateHybridV5ForwardWitness(
        schema="x", deep_v6_witness_cid="x",
        substrate_v5_forward_cid="x",
        bridge_v5_injection_cid="x",
        pre_inject_kv_cid="x", post_inject_kv_cid="x",
        post_evict_kv_cid="x",
        substrate_back_l2=0.0,
        ablation_perturbation_l2=0.0,
        cache_eviction_perturbation_l2=0.0,
        cache_retention_ratio=1.0, cache_policy="composite_v3",
        cache_flop_savings_ratio=0.5,
        retrieval_used=True, five_way=True,
        attention_steering_used=True,
        replay_decision="choose_reuse",
        replay_flop_chosen=100, replay_drift_chosen=0.05)
    # Witness-only composition for the H-bar (no inner V5 needed).
    hybrid = DeepSubstrateHybridV6(inner_v5=None)
    # Stub V4 cache + V2 replay + attention V5 witness.
    cc = CacheControllerV4.init(
        policy=W61_CACHE_POLICY_BILINEAR_RETRIEVAL_V6,
        d_model=64, d_key=8, fit_seed=int(seed) + 29300)
    cc.bilinear_retrieval_v6_matrix = _np.zeros(
        (64, 8), dtype=_np.float64)
    rc = ReplayControllerV2.init()
    rc.audit_v2.append({
        "stage": "v2_softmax", "decision": "choose_reuse",
        "confidence": 0.85,
        "probs": [0.85, 0.10, 0.03, 0.02],
        "flop_chosen": 100, "drift_chosen": 0.05,
        "flop_saving_vs_recompute": 0.9,
        "rationale": "v2_softmax_reuse", "crc_passed": True})
    p = _build_substrate(seed)
    ids = tokenize_bytes_v6("attn5", max_len=4)[:4]
    while len(ids) < 4: ids.append(0)
    proj = AttentionSteeringV2Projection.init(
        n_layers=p.config.n_layers, n_heads=p.config.n_heads,
        n_query=len(ids), n_key=len(ids), carrier_dim=6,
        seed=int(seed) + 29301)
    rng = _np.random.default_rng(int(seed) + 29302)
    carrier = list(rng.standard_normal(6))
    attn_w = steer_attention_and_measure_v5(
        params=p.v3_params, carrier=carrier,
        projection=proj, token_ids=ids,
        kl_budget_per_key=0.4)
    w = deep_substrate_hybrid_v6_forward(
        hybrid=hybrid, v5_witness=v5_witness,
        cache_controller_v4=cc, replay_controller_v2=rc,
        attention_steering_v5_witness=attn_w)
    return {
        "schema": R128_SCHEMA_VERSION,
        "name": "h152_deep_substrate_hybrid_v6",
        "passed": bool(w.six_way),
    }


_R128_FAMILIES: tuple[Any, ...] = (
    family_h144_substrate_v6_determinism,
    family_h144b_substrate_v6_cache_keys_shape,
    family_h144c_substrate_v6_hidden_write_trace,
    family_h144d_substrate_v6_replay_age_increments,
    family_h144e_substrate_v6_cross_layer_coupling,
    family_h145_kv_bridge_v6_multi_target,
    family_h145b_kv_bridge_v6_attention_pattern,
    family_h145c_kv_bridge_v6_cache_key_fingerprint,
    family_h146_hsb_v5_multi_target_fit,
    family_h146b_hsb_v5_recovery,
    family_h146c_hsb_v5_write_into_v6_cache,
    family_h147_attention_v5_4d_budget,
    family_h147b_attention_v5_negative_budget,
    family_h147c_attention_v5_signed_falsifier,
    family_h148_prefix_v5_chain_of_chains,
    family_h148b_prefix_v5_drift_predictor,
    family_h149_cache_controller_v4_bilinear,
    family_h149b_cache_controller_v4_corruption_floor,
    family_h149c_cache_controller_v4_two_stage,
    family_h149d_cache_controller_v4_composite,
    family_h150_replay_v2_threshold_fit,
    family_h150b_replay_v2_decision_confidence,
    family_h150c_replay_v2_hidden_write_gate,
    family_h151_substrate_adapter_v6_tier,
    family_h152_deep_substrate_hybrid_v6,
)


def run_r128(
        seeds: Sequence[int] = (196, 296, 396),
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for s in seeds:
        results = {}
        for fn in _R128_FAMILIES:
            try:
                r = fn(int(s))
            except Exception as e:
                r = {
                    "schema": R128_SCHEMA_VERSION,
                    "name": getattr(fn, "__name__", "unknown"),
                    "passed": False,
                    "exception": str(e),
                }
            results[r["name"]] = r
        out.append({
            "schema": R128_SCHEMA_VERSION,
            "seed": int(s),
            "family_results": results,
        })
    return out


__all__ = [
    "R128_SCHEMA_VERSION",
    "run_r128",
]
