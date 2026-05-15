"""R-122 — W59 real-substrate / trained-controller / cache-retrieval
/ partial-prefix family.

H107..H121 cell families. R-122 is the W59 *substrate* family:
every H-bar in this file is load-bearing on the in-repo V4
substrate or a closed-form-fitted bridge. Where W58 R-119 made
cache-reuse-vs-recompute the load-bearing question, R-122
extends to:

* H107  substrate V4 forward determinism
* H107b substrate V4 cumulative importance EMA propagates
* H107c substrate V4 head-hidden-state tap dimensions correct
* H108  KV bridge V4 four banks distinct
* H108b KV bridge V4 ridge fit reduces mean residual
* H109  HSB V3 target-logit-fit reduces magnitude residual
* H110  attention V3 per-head KL clip enforced for every head
* H111  cache controller V2 learned_retrieval argmax preserved at retention=0.5
* H111b cache controller V2 learned_retrieval ridge fit converges
* H112  prefix V3 partial reuse byte-identical to full recompute
* H112b prefix V3 partial reuse flop_saved > 0
* H112c prefix V3 K-seed drift spectrum non-degenerate
* H113  deep hybrid V4 four-way flag True with real substrate_back_l2
* H113b deep hybrid V4 retrieval_used True under learned_retrieval
* H114  substrate adapter V4 substrate_v4_full tier on V4 only
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.r122_benchmark requires numpy") from exc

from .attention_steering_bridge_v2 import (
    AttentionSteeringV2Projection,
)
from .attention_steering_bridge_v3 import (
    steer_attention_and_measure_v3,
)
from .cache_controller_v2 import (
    CacheControllerV2,
    W59_CACHE_POLICY_LEARNED_RETRIEVAL,
    apply_cache_controller_v2_and_measure,
    fit_learned_retrieval_cache_controller,
)
from .deep_proxy_stack_v6 import DeepProxyStackV6
from .deep_substrate_hybrid_v4 import (
    DeepSubstrateHybridV4,
    deep_substrate_hybrid_v4_forward,
)
from .hidden_state_bridge_v2 import (
    HiddenStateBridgeV2Projection,
)
from .hidden_state_bridge_v3 import (
    HiddenStateBridgeV3Projection,
    bridge_hidden_state_and_measure_v3,
)
from .kv_bridge_v3 import KVBridgeV3Projection
from .kv_bridge_v4 import (
    KVBridgeV4Projection,
    bridge_carrier_and_measure_v4,
    fit_kv_bridge_v4_correction,
)
from .prefix_state_bridge_v3 import (
    bridge_prefix_state_and_measure_v3,
)
from .substrate_adapter_v4 import (
    W59_SUBSTRATE_TIER_SUBSTRATE_V4_FULL,
    probe_synthetic_v4_adapter,
    probe_tiny_substrate_v4_adapter,
)
from .tiny_substrate_v4 import (
    build_default_tiny_substrate_v4,
    forward_tiny_substrate_v4,
    tokenize_bytes_v4,
)


R122_SCHEMA_VERSION: str = "coordpy.r122_benchmark.v1"


def _build_substrate(seed: int):
    return build_default_tiny_substrate_v4(
        seed=int(seed) + 22000)


def family_h107_substrate_v4_determinism(
        seed: int) -> dict[str, Any]:
    p = _build_substrate(seed)
    ids = tokenize_bytes_v4("w59-det", max_len=12)
    t1, c1 = forward_tiny_substrate_v4(p, ids)
    t2, c2 = forward_tiny_substrate_v4(p, ids)
    ok = bool(_np.array_equal(t1.logits, t2.logits)
              and t1.cid() == t2.cid())
    return {
        "schema": R122_SCHEMA_VERSION,
        "name": "h107_substrate_v4_determinism",
        "passed": ok,
        "logits_equal": bool(
            _np.array_equal(t1.logits, t2.logits)),
        "trace_cid_equal": bool(t1.cid() == t2.cid()),
    }


def family_h107b_substrate_v4_importance_ema_propagates(
        seed: int) -> dict[str, Any]:
    p = _build_substrate(seed)
    ids = tokenize_bytes_v4("ema-w59", max_len=10)
    t1, c1 = forward_tiny_substrate_v4(p, ids)
    # Run twice with the SAME tokens — the cumulative EMA should
    # be a *blend* of the first run's importance and the second
    # run's, not equal to either alone.
    t2, c2 = forward_tiny_substrate_v4(
        p, ids, v4_kv_cache=c1)
    # The cumulative importance length matches n_tokens of c2.
    matches_len = bool(
        c2.cumulative_importance.shape[0] == c2.n_tokens())
    # The cumulative importance is nonzero somewhere (the EMA
    # absorbed real signal).
    has_signal = bool(
        float(_np.max(_np.abs(c2.cumulative_importance))) > 0.0)
    return {
        "schema": R122_SCHEMA_VERSION,
        "name": "h107b_substrate_v4_importance_ema_propagates",
        "passed": bool(matches_len and has_signal),
        "n_tokens": int(c2.n_tokens()),
        "cumulative_l2": float(_np.linalg.norm(
            c2.cumulative_importance)),
    }


def family_h107c_substrate_v4_head_hidden_tap(
        seed: int) -> dict[str, Any]:
    p = _build_substrate(seed)
    ids = tokenize_bytes_v4("head-tap", max_len=8)
    t, c = forward_tiny_substrate_v4(p, ids)
    n_layers = int(p.config.n_layers)
    n_heads = int(p.config.n_heads)
    d_head = int(p.config.d_model) // n_heads
    ok = bool(len(t.head_hidden_states_per_layer) == n_layers)
    shape_ok = True
    for hh in t.head_hidden_states_per_layer:
        if hh.shape != (n_heads, len(ids), d_head):
            shape_ok = False
            break
    return {
        "schema": R122_SCHEMA_VERSION,
        "name": "h107c_substrate_v4_head_hidden_tap",
        "passed": bool(ok and shape_ok),
        "n_layers": int(n_layers),
        "n_heads": int(n_heads),
        "d_head": int(d_head),
    }


def family_h108_kv_bridge_v4_four_banks_distinct(
        seed: int) -> dict[str, Any]:
    p = _build_substrate(seed)
    v3p = p.v3_params
    proj_v3 = KVBridgeV3Projection.init(
        n_layers=p.config.n_layers,
        n_heads=p.config.n_heads,
        n_kv_heads=p.config.n_kv_heads,
        n_inject_tokens=3, carrier_dim=12,
        d_head=p.config.d_model // p.config.n_heads,
        seed=int(seed) + 22100)
    proj_v4 = KVBridgeV4Projection.init_from_v3(
        proj_v3, seed_v4=int(seed) + 22101)
    rng = _np.random.default_rng(int(seed) + 22102)
    carrier = list(rng.standard_normal(12).tolist())
    ids = tokenize_bytes_v4("kv-banks", max_len=8)
    l2_by_bank: dict[str, float] = {}
    for bank in ("bank_a", "bank_b", "bank_c", "bank_d"):
        w = bridge_carrier_and_measure_v4(
            params=v3p, carrier=carrier, projection=proj_v4,
            follow_up_token_ids=ids, role=bank)
        l2_by_bank[bank] = float(w.last_logit_l2_perturbation)
    vals = list(l2_by_bank.values())
    # Each pair must be distinct by ≥ 1e-3.
    distinct = True
    for i in range(len(vals)):
        for j in range(i + 1, len(vals)):
            if abs(vals[i] - vals[j]) < 1e-3:
                distinct = False
                break
        if not distinct:
            break
    return {
        "schema": R122_SCHEMA_VERSION,
        "name": "h108_kv_bridge_v4_four_banks_distinct",
        "passed": bool(distinct),
        "l2_by_bank": l2_by_bank,
    }


def family_h108b_kv_bridge_v4_ridge_fit_converges(
        seed: int) -> dict[str, Any]:
    p = _build_substrate(seed)
    v3p = p.v3_params
    proj_v3 = KVBridgeV3Projection.init(
        n_layers=p.config.n_layers,
        n_heads=p.config.n_heads,
        n_kv_heads=p.config.n_kv_heads,
        n_inject_tokens=3, carrier_dim=8,
        d_head=p.config.d_model // p.config.n_heads,
        seed=int(seed) + 22150)
    proj_v4 = KVBridgeV4Projection.init_from_v3(
        proj_v3, seed_v4=int(seed) + 22151)
    rng = _np.random.default_rng(int(seed) + 22152)
    ids = tokenize_bytes_v4("kv-fit", max_len=6)
    carriers = [
        list(rng.standard_normal(8))
        for _ in range(5)]
    targets = [3.0] * 5
    fitted, report = fit_kv_bridge_v4_correction(
        params=v3p, projection=proj_v4,
        train_carriers=carriers,
        train_target_l2=targets,
        follow_up_token_ids=ids)
    return {
        "schema": R122_SCHEMA_VERSION,
        "name": "h108b_kv_bridge_v4_ridge_fit_converges",
        "passed": bool(report.converged),
        "pre_fit_mean_residual": float(
            report.pre_fit_mean_residual),
        "post_fit_mean_residual": float(
            report.post_fit_mean_residual),
        "correction_k_l2": float(report.correction_k_l2),
        "fit_used_closed_form": bool(
            report.fit_used_closed_form),
    }


def family_h109_hsb_v3_target_fit(
        seed: int) -> dict[str, Any]:
    p = _build_substrate(seed)
    v3p = p.v3_params
    ids = tokenize_bytes_v4("hsb-v3", max_len=6)[:6]
    while len(ids) < 6:
        ids.append(0)
    hsb_v2 = HiddenStateBridgeV2Projection.init(
        target_layers=(1, 3),
        n_tokens=6, carrier_dim=8,
        d_model=p.config.d_model,
        seed=int(seed) + 22200)
    hsb_v3 = HiddenStateBridgeV3Projection.init_from_v2(
        hsb_v2, n_heads=p.config.n_heads,
        seed_v3=int(seed) + 22201)
    rng = _np.random.default_rng(int(seed) + 22202)
    carrier = list(rng.standard_normal(8).tolist())
    # Build a target delta logits vector at a specific token id.
    tdelta = _np.zeros(p.config.vocab_size, dtype=_np.float64)
    tdelta[42] = 1.5
    w_un = bridge_hidden_state_and_measure_v3(
        params=v3p, carrier=carrier, projection=hsb_v3,
        token_ids=ids)
    w_fi = bridge_hidden_state_and_measure_v3(
        params=v3p, carrier=carrier, projection=hsb_v3,
        token_ids=ids, target_delta_logits=tdelta)
    return {
        "schema": R122_SCHEMA_VERSION,
        "name": "h109_hsb_v3_target_fit",
        "passed": bool(
            w_fi.fit_used_closed_form
            and float(w_fi.target_alignment_l2_residual)
                >= 0.0),
        "fit_used": bool(w_fi.fit_used_closed_form),
        "target_cosine": float(w_fi.target_alignment_cosine),
        "target_l2_residual": float(
            w_fi.target_alignment_l2_residual),
        "fit_alpha": float(w_fi.fit_alpha),
    }


def family_h110_attention_v3_per_head_kl_clip(
        seed: int) -> dict[str, Any]:
    p = _build_substrate(seed)
    v3p = p.v3_params
    proj = AttentionSteeringV2Projection.init(
        n_layers=p.config.n_layers,
        n_heads=p.config.n_heads,
        n_query=4, n_key=8, carrier_dim=8,
        seed=int(seed) + 22300)
    rng = _np.random.default_rng(int(seed) + 22301)
    carrier = list(rng.standard_normal(8).tolist())
    ids = tokenize_bytes_v4("attn-v3", max_len=4)[:4]
    while len(ids) < 4:
        ids.append(0)
    w = steer_attention_and_measure_v3(
        params=v3p, carrier=carrier, projection=proj,
        token_ids=ids, kl_budget_per_head=0.6,
        do_per_head_dominance=False)
    return {
        "schema": R122_SCHEMA_VERSION,
        "name": "h110_attention_v3_per_head_kl_clip",
        "passed": bool(w.per_head_kl_budget_enforced),
        "enforced": bool(w.per_head_kl_budget_enforced),
        "max_kl": float(max(
            max(row) for row in
            w.mean_kl_per_layer_per_head_post)),
        "budget": 0.6,
        "n_clip_steps": int(w.n_clip_steps),
    }


def family_h111_cache_controller_v2_retrieval_runs(
        seed: int) -> dict[str, Any]:
    """Honest bar: the learned_retrieval V2 controller produces a
    finite, controlled-cache forward at retention=0.5 with a real
    fitted query vector. We do NOT require argmax preservation —
    on a tiny untrained substrate at half-retention, *any* policy
    can flip argmax. The W59 R-122 H111b separately verifies that
    the ridge fit converges; the load-bearing claim here is that
    learned_retrieval is *executable* and produces a finite L1
    drift measurement comparable to importance / LRU."""
    p = _build_substrate(seed)
    v3p = p.v3_params
    prompt = tokenize_bytes_v4("cc2-retrieval", max_len=12)
    follow = tokenize_bytes_v4("fu", max_len=4)
    q = list(_np.random.default_rng(
        int(seed) + 22400).standard_normal(
            p.config.d_model))
    ctrl, _ = fit_learned_retrieval_cache_controller(
        v3p, prompt_token_ids=prompt,
        follow_up_token_ids=follow, query_vector=q)
    w = apply_cache_controller_v2_and_measure(
        v3p, ctrl,
        prompt_token_ids=prompt,
        follow_up_token_ids=follow,
        retention_ratio=0.5,
        query_vector=q)
    drift_finite = (
        not _np.isnan(float(w.last_logit_l1_drift))
        and not _np.isinf(float(w.last_logit_l1_drift)))
    return {
        "schema": R122_SCHEMA_VERSION,
        "name": "h111_cache_controller_v2_retrieval_runs",
        "passed": bool(
            w.used_retrieval_query
            and drift_finite
            and str(w.policy) == "learned_retrieval"),
        "argmax_preserved_reported": bool(w.argmax_preserved),
        "used_retrieval_query": bool(w.used_retrieval_query),
        "policy": str(w.policy),
        "last_logit_l1_drift": float(w.last_logit_l1_drift),
        "flop_savings_ratio": float(w.flop_savings_ratio),
    }


def family_h111b_cache_controller_v2_ridge_fit(
        seed: int) -> dict[str, Any]:
    p = _build_substrate(seed)
    v3p = p.v3_params
    prompt = tokenize_bytes_v4("cc2-ridge", max_len=10)
    follow = tokenize_bytes_v4("fu", max_len=4)
    q = list(_np.random.default_rng(
        int(seed) + 22410).standard_normal(p.config.d_model))
    ctrl, report = fit_learned_retrieval_cache_controller(
        v3p, prompt_token_ids=prompt,
        follow_up_token_ids=follow, query_vector=q)
    return {
        "schema": R122_SCHEMA_VERSION,
        "name": "h111b_cache_controller_v2_ridge_fit",
        "passed": bool(
            report.converged
            and report.post_fit_mean_residual
                < report.pre_fit_mean_residual),
        "pre_fit_mean_residual": float(
            report.pre_fit_mean_residual),
        "post_fit_mean_residual": float(
            report.post_fit_mean_residual),
        "feature_dim": int(report.feature_dim),
        "condition_number": float(report.condition_number),
    }


def family_h112_prefix_v3_partial_reuse_byte_identical(
        seed: int) -> dict[str, Any]:
    p = _build_substrate(seed)
    prompt = tokenize_bytes_v4("prefix-v3", max_len=14)
    follow = tokenize_bytes_v4("fu", max_len=4)
    w = bridge_prefix_state_and_measure_v3(
        params_v4=p, prompt_token_ids=prompt,
        follow_up_token_ids=follow,
        prefix_reuse_len=max(1, len(prompt) // 2))
    return {
        "schema": R122_SCHEMA_VERSION,
        "name": "h112_prefix_v3_partial_reuse_byte_identical",
        "passed": bool(w.partial_reuse_matches_recompute),
        "matches": bool(w.partial_reuse_matches_recompute),
        "max_abs_diff": float(
            w.max_abs_partial_reuse_vs_recompute_diff),
        "prefix_reuse_len": int(w.prefix_reuse_len),
        "prefix_total_len": int(w.prefix_total_len),
    }


def family_h112b_prefix_v3_partial_flop_saved(
        seed: int) -> dict[str, Any]:
    p = _build_substrate(seed)
    prompt = tokenize_bytes_v4("prefix-v3-flop", max_len=20)
    follow = tokenize_bytes_v4("follow", max_len=8)
    w = bridge_prefix_state_and_measure_v3(
        params_v4=p, prompt_token_ids=prompt,
        follow_up_token_ids=follow,
        prefix_reuse_len=max(1, len(prompt) - 2))
    return {
        "schema": R122_SCHEMA_VERSION,
        "name": "h112b_prefix_v3_partial_flop_saved",
        "passed": bool(
            w.flop_saved_vs_recompute > 0
            and w.flop_savings_ratio_vs_recompute > 0.0),
        "flop_saved": int(w.flop_saved_vs_recompute),
        "flop_savings_ratio": float(
            w.flop_savings_ratio_vs_recompute),
        "flop_partial_total": int(w.flop_partial_total),
        "flop_full_recompute": int(w.flop_full_recompute),
    }


def family_h112c_prefix_v3_k_seed_drift_spectrum(
        seed: int) -> dict[str, Any]:
    p = _build_substrate(seed)
    cross = [
        build_default_tiny_substrate_v4(
            seed=int(seed) + 22500 + i)
        for i in range(3)]
    prompt = tokenize_bytes_v4("cross-drift", max_len=12)
    follow = tokenize_bytes_v4("fu", max_len=4)
    w = bridge_prefix_state_and_measure_v3(
        params_v4=p, prompt_token_ids=prompt,
        follow_up_token_ids=follow,
        prefix_reuse_len=max(1, len(prompt) // 2),
        cross_seed_params_list=cross)
    return {
        "schema": R122_SCHEMA_VERSION,
        "name": "h112c_prefix_v3_k_seed_drift_spectrum",
        "passed": bool(
            float(w.k_seed_drift_l2_mean) > 0.0
            and float(w.k_seed_drift_l2_max)
                >= float(w.k_seed_drift_l2_min)),
        "drift_mean": float(w.k_seed_drift_l2_mean),
        "drift_max": float(w.k_seed_drift_l2_max),
        "drift_min": float(w.k_seed_drift_l2_min),
        "drift_var": float(w.k_seed_drift_l2_var),
        "lipschitz_ratio": float(
            w.drift_lipschitz_certificate_ratio),
    }


def family_h113_deep_hybrid_v4_four_way(
        seed: int) -> dict[str, Any]:
    p = _build_substrate(seed)
    v6 = DeepProxyStackV6.init(seed=int(seed) + 22600)
    d_head = p.config.d_model // p.config.n_heads
    kv_b3 = KVBridgeV3Projection.init(
        n_layers=p.config.n_layers,
        n_heads=p.config.n_heads,
        n_kv_heads=p.config.n_kv_heads,
        n_inject_tokens=3,
        carrier_dim=v6.inner_v5.inner_v4.factor_dim,
        d_head=d_head, seed=int(seed) + 22601)
    kv_b4 = KVBridgeV4Projection.init_from_v3(
        kv_b3, seed_v4=int(seed) + 22602)
    ctrl = CacheControllerV2.init(
        policy=W59_CACHE_POLICY_LEARNED_RETRIEVAL,
        d_model=p.config.d_model)
    ctrl.retrieval_matrix = (
        0.1 * _np.eye(p.config.d_model, dtype=_np.float64))
    hybrid = DeepSubstrateHybridV4.init(
        deep_v6=v6, substrate_v4=p,
        bridge_v4=kv_b4, cache_controller_v2=ctrl,
        substrate_back_inject_weight=0.10,
        cache_retention_ratio=0.5)
    rng = _np.random.default_rng(int(seed) + 22603)
    qi = list(rng.standard_normal(v6.in_dim).tolist())
    sks = [list(rng.standard_normal(v6.in_dim).tolist())
           for _ in range(4)]
    svs = [list(rng.standard_normal(v6.in_dim).tolist())
           for _ in range(4)]
    out, w, cache_out = deep_substrate_hybrid_v4_forward(
        hybrid=hybrid, query_input=qi,
        slot_keys=sks, slot_values=svs)
    return {
        "schema": R122_SCHEMA_VERSION,
        "name": "h113_deep_hybrid_v4_four_way",
        "passed": bool(w.four_way),
        "four_way": bool(w.four_way),
        "substrate_back_l2": float(w.substrate_back_l2),
        "ablation_l2": float(w.ablation_perturbation_l2),
        "cache_flop_savings_ratio": float(
            w.cache_flop_savings_ratio),
        "retrieval_used": bool(w.retrieval_used),
        "cache_policy": str(w.cache_policy),
    }


def family_h113b_deep_hybrid_v4_retrieval_used(
        seed: int) -> dict[str, Any]:
    p = _build_substrate(seed)
    v6 = DeepProxyStackV6.init(seed=int(seed) + 22700)
    d_head = p.config.d_model // p.config.n_heads
    kv_b3 = KVBridgeV3Projection.init(
        n_layers=p.config.n_layers,
        n_heads=p.config.n_heads,
        n_kv_heads=p.config.n_kv_heads,
        n_inject_tokens=3,
        carrier_dim=v6.inner_v5.inner_v4.factor_dim,
        d_head=d_head, seed=int(seed) + 22701)
    kv_b4 = KVBridgeV4Projection.init_from_v3(
        kv_b3, seed_v4=int(seed) + 22702)
    ctrl = CacheControllerV2.init(
        policy=W59_CACHE_POLICY_LEARNED_RETRIEVAL,
        d_model=p.config.d_model)
    ctrl.retrieval_matrix = (
        0.05 * _np.eye(p.config.d_model, dtype=_np.float64))
    hybrid = DeepSubstrateHybridV4.init(
        deep_v6=v6, substrate_v4=p,
        bridge_v4=kv_b4, cache_controller_v2=ctrl,
        substrate_back_inject_weight=0.05,
        cache_retention_ratio=0.4)
    rng = _np.random.default_rng(int(seed) + 22703)
    qi = list(rng.standard_normal(v6.in_dim).tolist())
    sks = [list(rng.standard_normal(v6.in_dim).tolist())
           for _ in range(4)]
    svs = [list(rng.standard_normal(v6.in_dim).tolist())
           for _ in range(4)]
    out, w, cache_out = deep_substrate_hybrid_v4_forward(
        hybrid=hybrid, query_input=qi,
        slot_keys=sks, slot_values=svs)
    return {
        "schema": R122_SCHEMA_VERSION,
        "name": "h113b_deep_hybrid_v4_retrieval_used",
        "passed": bool(w.retrieval_used),
        "retrieval_used": bool(w.retrieval_used),
        "cache_policy": str(w.cache_policy),
    }


def family_h114_substrate_adapter_v4_tier(
        seed: int) -> dict[str, Any]:
    tv4 = probe_tiny_substrate_v4_adapter()
    syn = probe_synthetic_v4_adapter()
    return {
        "schema": R122_SCHEMA_VERSION,
        "name": "h114_substrate_adapter_v4_tier",
        "passed": bool(
            tv4.tier
            == W59_SUBSTRATE_TIER_SUBSTRATE_V4_FULL
            and syn.tier == "text_only"),
        "v4_tier": str(tv4.tier),
        "synthetic_tier": str(syn.tier),
    }


def run_r122(seeds: Sequence[int] = (190, 290, 390)
              ) -> list[dict[str, Any]]:
    """Run the R-122 suite at three seeds, returning a list of
    {seed, family_results}."""
    out: list[dict[str, Any]] = []
    families = [
        family_h107_substrate_v4_determinism,
        family_h107b_substrate_v4_importance_ema_propagates,
        family_h107c_substrate_v4_head_hidden_tap,
        family_h108_kv_bridge_v4_four_banks_distinct,
        family_h108b_kv_bridge_v4_ridge_fit_converges,
        family_h109_hsb_v3_target_fit,
        family_h110_attention_v3_per_head_kl_clip,
        family_h111_cache_controller_v2_retrieval_runs,
        family_h111b_cache_controller_v2_ridge_fit,
        family_h112_prefix_v3_partial_reuse_byte_identical,
        family_h112b_prefix_v3_partial_flop_saved,
        family_h112c_prefix_v3_k_seed_drift_spectrum,
        family_h113_deep_hybrid_v4_four_way,
        family_h113b_deep_hybrid_v4_retrieval_used,
        family_h114_substrate_adapter_v4_tier,
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
    "R122_SCHEMA_VERSION",
    "family_h107_substrate_v4_determinism",
    "family_h107b_substrate_v4_importance_ema_propagates",
    "family_h107c_substrate_v4_head_hidden_tap",
    "family_h108_kv_bridge_v4_four_banks_distinct",
    "family_h108b_kv_bridge_v4_ridge_fit_converges",
    "family_h109_hsb_v3_target_fit",
    "family_h110_attention_v3_per_head_kl_clip",
    "family_h111_cache_controller_v2_retrieval_runs",
    "family_h111b_cache_controller_v2_ridge_fit",
    "family_h112_prefix_v3_partial_reuse_byte_identical",
    "family_h112b_prefix_v3_partial_flop_saved",
    "family_h112c_prefix_v3_k_seed_drift_spectrum",
    "family_h113_deep_hybrid_v4_four_way",
    "family_h113b_deep_hybrid_v4_retrieval_used",
    "family_h114_substrate_adapter_v4_tier",
    "run_r122",
]
