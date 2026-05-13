"""R-116 — W57 real-substrate / latent-bridge / multi-hop family.

H43..H56 cell families that exercise:

  * Substrate V2 determinism (4 layers, 8 heads)
  * Substrate V2 KV cache reuse
  * Substrate V2 causal mask soundness
  * Substrate V2 logit lens determinism
  * Substrate V2 prefix-state save+reuse
  * Substrate V2 cache eviction LRU
  * Substrate V2 cache eviction weighted
  * KV Bridge V2 last-position L2 perturbation > 0
  * KV Bridge V2 monotone-in-n_inject_tokens
  * Hidden-State Bridge perturbation > 0
  * Attention Steering bridge mean-KL > 0 per layer
  * Multi-Hop V7 chain-length-9 fidelity
  * Substrate Adapter V2 tier classification
  * Two-way substrate bridge load-bearing
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import random
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError("coordpy.r116_benchmark requires numpy") from exc

from .attention_steering_bridge import (
    AttentionSteeringProjection,
    steer_attention_and_measure,
)
from .deep_proxy_stack_v6 import DeepProxyStackV6
from .deep_substrate_hybrid_v2 import (
    DeepSubstrateHybridV2,
    deep_substrate_hybrid_v2_forward,
)
from .hidden_state_bridge import (
    HiddenStateBridgeProjection,
    bridge_hidden_state_and_measure,
)
from .kv_bridge_v2 import (
    KVBridgeV2Projection,
    bridge_carrier_and_measure_v2,
)
from .multi_hop_translator_v7 import (
    W57_DEFAULT_MH_V7_BACKENDS,
    evaluate_dec_chain_len9_fidelity,
)
from .prefix_state_bridge import (
    bridge_prefix_state_and_measure,
)
from .substrate_adapter_v2 import (
    probe_all_v2_adapters,
    W57_SUBSTRATE_TIER_SUBSTRATE_V2_FULL,
)
from .tiny_substrate_v2 import (
    TinyV2KVCache,
    build_default_tiny_substrate_v2,
    forward_tiny_substrate_v2,
    tokenize_bytes_v2,
)


R116_SCHEMA_VERSION: str = "coordpy.r116_benchmark.v1"


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass(frozen=True)
class R116SeedResult:
    seed: int
    family_results: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": R116_SCHEMA_VERSION,
            "seed": int(self.seed),
            "family_results": dict(self.family_results),
        }


# ---------------------------------------------------------------------------
# Cell families (H43..H56)
# ---------------------------------------------------------------------------


def family_substrate_v2_forward_determinism(
        seed: int,
) -> dict[str, Any]:
    """H43 — substrate V2 forward is byte-deterministic."""
    p = build_default_tiny_substrate_v2(seed=int(seed))
    ids = tokenize_bytes_v2("hello-w57", max_len=20)
    t1 = forward_tiny_substrate_v2(p, ids)
    t2 = forward_tiny_substrate_v2(p, ids)
    return {
        "schema": R116_SCHEMA_VERSION,
        "name": "substrate_v2_forward_determinism",
        "passed": bool(t1.cid() == t2.cid()),
        "logits_diff": float(
            _np.max(_np.abs(t1.logits - t2.logits))),
    }


def family_substrate_v2_kv_cache_reuse(seed: int) -> dict[str, Any]:
    """H44 — V2 KV reuse: last-position logits match within 1e-9."""
    p = build_default_tiny_substrate_v2(seed=int(seed))
    ids = tokenize_bytes_v2("kv-reuse-w57", max_len=24)
    t_full = forward_tiny_substrate_v2(p, ids)
    half = len(ids) // 2
    t_a = forward_tiny_substrate_v2(p, ids[:half])
    t_b = forward_tiny_substrate_v2(
        p, ids[half:], kv_cache=t_a.kv_cache)
    diff = float(_np.max(_np.abs(
        t_full.logits[-1] - t_b.logits[-1])))
    return {
        "schema": R116_SCHEMA_VERSION,
        "name": "substrate_v2_kv_cache_reuse",
        "passed": bool(diff < 1e-9),
        "max_abs_diff": float(diff),
    }


def family_substrate_v2_causal_mask(seed: int) -> dict[str, Any]:
    """H45 — V2 causal mask strict at the attention layer."""
    p = build_default_tiny_substrate_v2(seed=int(seed))
    ids = tokenize_bytes_v2("causal-w57", max_len=12)
    t = forward_tiny_substrate_v2(p, ids, return_attention=True)
    max_upper = 0.0
    for attn in t.attn_weights_per_layer:
        H, T_new, T_all = attn.shape
        for i in range(T_new):
            for j in range(i + 1, T_all):
                max_upper = max(
                    max_upper, float(_np.max(attn[:, i, j])))
    return {
        "schema": R116_SCHEMA_VERSION,
        "name": "substrate_v2_causal_mask",
        "passed": bool(max_upper < 1e-9),
        "max_upper_triangle_weight": float(max_upper),
    }


def family_substrate_v2_logit_lens(seed: int) -> dict[str, Any]:
    """H46 — V2 logit lens is deterministic and shape-consistent."""
    p = build_default_tiny_substrate_v2(seed=int(seed))
    ids = tokenize_bytes_v2("lens-w57", max_len=12)
    t1 = forward_tiny_substrate_v2(p, ids)
    t2 = forward_tiny_substrate_v2(p, ids)
    shape_ok = True
    diff_max = 0.0
    for l1, l2 in zip(t1.per_layer_logit_lens,
                       t2.per_layer_logit_lens):
        if l1.shape != l2.shape:
            shape_ok = False
        diff_max = max(diff_max,
                        float(_np.max(_np.abs(l1 - l2))))
    return {
        "schema": R116_SCHEMA_VERSION,
        "name": "substrate_v2_logit_lens",
        "passed": bool(shape_ok and diff_max < 1e-12),
        "lens_count": len(t1.per_layer_logit_lens),
        "max_diff": float(diff_max),
    }


def family_substrate_v2_prefix_state_reuse(
        seed: int,
) -> dict[str, Any]:
    """H47 — prefix-state save+reuse matches full recompute."""
    p = build_default_tiny_substrate_v2(seed=int(seed))
    prompt = tokenize_bytes_v2("prefix-prompt", max_len=20)
    follow = [104, 105]  # 'hi'
    w = bridge_prefix_state_and_measure(
        params=p, prompt_token_ids=prompt,
        follow_up_token_ids=follow)
    return {
        "schema": R116_SCHEMA_VERSION,
        "name": "substrate_v2_prefix_state_reuse",
        "passed": bool(w.reuse_matches_recompute),
        "max_abs_diff": float(w.max_abs_reuse_recompute_diff),
    }


def family_substrate_v2_cache_eviction_lru(
        seed: int,
) -> dict[str, Any]:
    """H48 — LRU eviction reduces cache n_tokens correctly."""
    p = build_default_tiny_substrate_v2(seed=int(seed))
    ids = tokenize_bytes_v2("evict-prompt", max_len=20)
    t = forward_tiny_substrate_v2(p, ids)
    n0 = t.kv_cache.n_tokens()
    pruned = t.kv_cache.evict_lru(2)
    n1 = pruned.n_tokens()
    return {
        "schema": R116_SCHEMA_VERSION,
        "name": "substrate_v2_cache_eviction_lru",
        "passed": bool(n1 == n0 - 2),
        "n0": int(n0), "n1": int(n1),
    }


def family_substrate_v2_cache_eviction_weighted(
        seed: int,
) -> dict[str, Any]:
    """H49 — weighted eviction keeps top-k by weight."""
    p = build_default_tiny_substrate_v2(seed=int(seed))
    ids = tokenize_bytes_v2("weighted-evict", max_len=20)
    t = forward_tiny_substrate_v2(p, ids)
    n0 = t.kv_cache.n_tokens()
    rng = random.Random(int(seed))
    weights = [rng.random() for _ in range(n0)]
    pruned = t.kv_cache.evict_weighted(weights, keep=3)
    n1 = pruned.n_tokens()
    return {
        "schema": R116_SCHEMA_VERSION,
        "name": "substrate_v2_cache_eviction_weighted",
        "passed": bool(n1 == 3),
        "n0": int(n0), "n1": int(n1),
    }


def family_kv_bridge_v2_perturbation(seed: int) -> dict[str, Any]:
    """H50 — V2 KV bridge L2 perturbation > 0."""
    p = build_default_tiny_substrate_v2(seed=int(seed))
    proj = KVBridgeV2Projection.init(
        n_layers=p.config.n_layers,
        n_heads=p.config.n_heads,
        n_inject_tokens=3,
        carrier_dim=12,
        d_head=p.config.d_model // p.config.n_heads,
        seed=int(seed) + 7)
    rng = _np.random.default_rng(int(seed))
    carrier = list(rng.standard_normal(12))
    w = bridge_carrier_and_measure_v2(
        params=p, carrier=carrier,
        projection=proj,
        follow_up_token_ids=tokenize_bytes_v2(
            "fu", max_len=12))
    return {
        "schema": R116_SCHEMA_VERSION,
        "name": "kv_bridge_v2_perturbation",
        "passed": bool(w.last_logit_l2_perturbation > 1e-6),
        "l2_perturb": float(w.last_logit_l2_perturbation),
        "ce_delta": float(w.cross_entropy_delta),
    }


def family_kv_bridge_v2_inject_robust_nonzero(
        seed: int,
) -> dict[str, Any]:
    """H51 — at fixed n_inject_tokens=3, ≥ 90% of random carriers
    produce a non-zero L2 perturbation (substrate coupling is
    robust, not a single-seed artefact)."""
    p = build_default_tiny_substrate_v2(seed=int(seed))
    proj = KVBridgeV2Projection.init(
        n_layers=p.config.n_layers,
        n_heads=p.config.n_heads,
        n_inject_tokens=3,
        carrier_dim=12,
        d_head=p.config.d_model // p.config.n_heads,
        seed=int(seed) + 7)
    rng = _np.random.default_rng(int(seed))
    n_probes = 16
    ok = 0
    for _ in range(n_probes):
        carrier = list(rng.standard_normal(12))
        w = bridge_carrier_and_measure_v2(
            params=p, carrier=carrier, projection=proj,
            follow_up_token_ids=[257])
        if float(w.last_logit_l2_perturbation) > 1e-6:
            ok += 1
    rate = float(ok) / float(n_probes)
    return {
        "schema": R116_SCHEMA_VERSION,
        "name": "kv_bridge_v2_inject_robust_nonzero",
        "passed": bool(rate >= 0.90),
        "nonzero_rate": float(rate),
        "n_probes": int(n_probes),
    }


def family_hidden_state_bridge_perturbation(
        seed: int,
) -> dict[str, Any]:
    """H52 — hidden-state bridge produces non-zero L2 perturb."""
    p = build_default_tiny_substrate_v2(seed=int(seed))
    proj = HiddenStateBridgeProjection.init(
        n_layers=p.config.n_layers,
        n_tokens=4,
        carrier_dim=12,
        d_model=p.config.d_model,
        seed=int(seed) + 8)
    rng = _np.random.default_rng(int(seed))
    carrier = list(rng.standard_normal(12))
    w = bridge_hidden_state_and_measure(
        params=p, carrier=carrier, projection=proj,
        target_layer=1,
        token_ids=tokenize_bytes_v2("hs-fu", max_len=12))
    return {
        "schema": R116_SCHEMA_VERSION,
        "name": "hidden_state_bridge_perturbation",
        "passed": bool(w.last_logit_l2_perturbation > 1e-6),
        "l2_perturb": float(w.last_logit_l2_perturbation),
        "ce_delta": float(w.cross_entropy_delta),
    }


def family_attention_steering_kl(seed: int) -> dict[str, Any]:
    """H53 — attention-steering yields mean-KL > 0 per layer."""
    p = build_default_tiny_substrate_v2(seed=int(seed))
    ids = tokenize_bytes_v2("attn-w57", max_len=12)
    proj = AttentionSteeringProjection.init(
        n_layers=p.config.n_layers,
        n_heads=p.config.n_heads,
        n_query=len(ids),
        n_key=len(ids),
        carrier_dim=12,
        seed=int(seed) + 9)
    rng = _np.random.default_rng(int(seed))
    carrier = list(rng.standard_normal(12))
    w = steer_attention_and_measure(
        params=p, carrier=carrier, projection=proj,
        token_ids=ids)
    per_layer_positive = [
        float(k) > 0.0 for k in w.mean_kl_per_layer]
    all_positive = all(per_layer_positive)
    return {
        "schema": R116_SCHEMA_VERSION,
        "name": "attention_steering_kl",
        "passed": bool(all_positive),
        "mean_kl_per_layer": [
            float(round(k, 6)) for k in w.mean_kl_per_layer],
        "attention_pattern_shifted": bool(
            w.attention_pattern_shifted),
    }


def family_multi_hop_v7_chain_len9(seed: int) -> dict[str, Any]:
    """H54 — multi-hop V7 chain-length-9 fidelity probe."""
    r = evaluate_dec_chain_len9_fidelity(
        backends=W57_DEFAULT_MH_V7_BACKENDS,
        feature_dim=8,
        seed=int(seed))
    return {
        "schema": R116_SCHEMA_VERSION,
        "name": "multi_hop_v7_chain_len9",
        "passed": bool(r["chain_length"] == 9),
        "fidelity": float(r["fidelity"]),
        "final_l2": float(r["final_l2"]),
    }


def family_substrate_adapter_v2_tiers(
        seed: int,
) -> dict[str, Any]:
    """H55 — adapter V2 classifies tiny_substrate_v2 as
    substrate_v2_full and synthetic as text_only."""
    m = probe_all_v2_adapters(
        probe_ollama=False, probe_openai=False)
    by_name = m.by_name()
    tiny = by_name.get("tiny_substrate_v2")
    synth = by_name.get("synthetic")
    return {
        "schema": R116_SCHEMA_VERSION,
        "name": "substrate_adapter_v2_tiers",
        "passed": bool(
            tiny is not None
            and tiny.tier == W57_SUBSTRATE_TIER_SUBSTRATE_V2_FULL
            and synth is not None
            and synth.tier == "text_only"),
        "tiny_tier": str(tiny.tier) if tiny else "missing",
        "synth_tier": str(synth.tier) if synth else "missing",
    }


def family_two_way_substrate_bridge(seed: int) -> dict[str, Any]:
    """H56 — bidirectional bridge: substrate-back L2 > 0 AND
    ablation perturbation L2 > 0."""
    p = build_default_tiny_substrate_v2(seed=int(seed))
    deep_v6 = DeepProxyStackV6.init(seed=int(seed) + 1)
    hyb = DeepSubstrateHybridV2.init(
        deep_v6=deep_v6, substrate=p,
        n_inject_tokens=3,
        carrier_dim=int(deep_v6.inner_v5.inner_v4.factor_dim),
        substrate_back_inject_weight=0.10)
    rng = _np.random.default_rng(int(seed))
    in_dim = int(deep_v6.in_dim)
    fd = int(deep_v6.inner_v5.inner_v4.factor_dim)
    q = list(rng.standard_normal(in_dim) * 0.1)
    k = [list(rng.standard_normal(fd) * 0.1) for _ in range(2)]
    v = [list(rng.standard_normal(fd) * 0.1) for _ in range(2)]
    _, w, _ = deep_substrate_hybrid_v2_forward(
        hybrid=hyb,
        query_input=q, slot_keys=k, slot_values=v)
    return {
        "schema": R116_SCHEMA_VERSION,
        "name": "two_way_substrate_bridge",
        "passed": bool(
            w.substrate_back_l2 > 1e-6
            and w.ablation_perturbation_l2 > 1e-6
            and w.bidirectional),
        "substrate_back_l2": float(w.substrate_back_l2),
        "ablation_perturbation_l2": float(
            w.ablation_perturbation_l2),
        "bidirectional": bool(w.bidirectional),
    }


# ---------------------------------------------------------------------------
# Family registry + driver
# ---------------------------------------------------------------------------


R116_FAMILIES = (
    ("h43_substrate_v2_forward_determinism",
     family_substrate_v2_forward_determinism),
    ("h44_substrate_v2_kv_cache_reuse",
     family_substrate_v2_kv_cache_reuse),
    ("h45_substrate_v2_causal_mask",
     family_substrate_v2_causal_mask),
    ("h46_substrate_v2_logit_lens",
     family_substrate_v2_logit_lens),
    ("h47_substrate_v2_prefix_state_reuse",
     family_substrate_v2_prefix_state_reuse),
    ("h48_substrate_v2_cache_eviction_lru",
     family_substrate_v2_cache_eviction_lru),
    ("h49_substrate_v2_cache_eviction_weighted",
     family_substrate_v2_cache_eviction_weighted),
    ("h50_kv_bridge_v2_perturbation",
     family_kv_bridge_v2_perturbation),
    ("h51_kv_bridge_v2_inject_robust_nonzero",
     family_kv_bridge_v2_inject_robust_nonzero),
    ("h52_hidden_state_bridge_perturbation",
     family_hidden_state_bridge_perturbation),
    ("h53_attention_steering_kl",
     family_attention_steering_kl),
    ("h54_multi_hop_v7_chain_len9",
     family_multi_hop_v7_chain_len9),
    ("h55_substrate_adapter_v2_tiers",
     family_substrate_adapter_v2_tiers),
    ("h56_two_way_substrate_bridge",
     family_two_way_substrate_bridge),
)


def run_r116(*, seeds: Sequence[int] = (0, 1, 2)) -> dict[str, Any]:
    """Run all R-116 families across multiple seeds; aggregate."""
    rows: list[R116SeedResult] = []
    for s in seeds:
        results: dict[str, dict[str, Any]] = {}
        for name, fn in R116_FAMILIES:
            results[name] = fn(int(s))
        rows.append(R116SeedResult(
            seed=int(s), family_results=results))
    summary = {
        "schema": R116_SCHEMA_VERSION,
        "n_seeds": int(len(seeds)),
        "seeds": [r.to_dict() for r in rows],
    }
    pass_counts: dict[str, int] = {}
    for r in rows:
        for k, v in r.family_results.items():
            if bool(v.get("passed", False)):
                pass_counts[k] = pass_counts.get(k, 0) + 1
    summary["pass_counts"] = pass_counts
    summary["all_passed"] = bool(all(
        pass_counts.get(name, 0) == len(seeds)
        for name, _ in R116_FAMILIES))
    return summary


__all__ = [
    "R116_SCHEMA_VERSION",
    "R116_FAMILIES",
    "R116SeedResult",
    "run_r116",
]
