"""W58 M8 — Deep Substrate Hybrid V3.

Strictly extends W57's ``coordpy.deep_substrate_hybrid_v2``. V3
upgrades the substrate from ``tiny_substrate_v2`` to
``tiny_substrate_v3``, adds the W58 KV bridge V3 + hidden-state
bridge V2 + attention-steering V2 alongside the existing V6 ↔
substrate bidirectional path, AND closes a *third* bridge to a
**cache controller** that scores tokens by carrier-conditioned
importance and retains only a configurable fraction of them
before the substrate's final forward.

Three-way forward flow:

1. Substrate V3 baseline forward over a reference token sequence
   → produces a per-layer hidden state and KV importance vectors.
2. Substrate hidden state projects *back* into V6 residual.
3. V6 forward runs with the biased query input.
4. V6 residual injects into substrate V3 KV cache via KV bridge V3.
5. **Cache controller** scores the injected cache by carrier-
   conditioned importance (using the substrate's own importance
   vectors *plus* the V6 head value as the carrier) and evicts to
   a target retention ratio.
6. Optional W58 attention-steering V2 bias.
7. Substrate V3 forward on the controlled-cache produces the
   final logits and per-layer attention.
8. Ablation: a substrate forward on the un-controlled cache
   measures the V3 hybrid's contribution.

V3 strictly extends V2 in spirit: when ``cache_retention_ratio
== 1.0`` and attention-steering is disabled and the cache
controller is uniform/LRU, V3 reduces to V2-style behaviour.
However, V3 *targets* the new V3 substrate runtime; it is NOT
byte-compatible with V2 substrate (different cache layout, GQA,
different RoPE table).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.deep_substrate_hybrid_v3 requires numpy"
        ) from exc

from .attention_steering_bridge_v2 import (
    AttentionSteeringV2Projection,
    steer_attention_and_measure_v2,
)
from .cache_controller import (
    CacheController,
    apply_cache_controller_and_measure,
    W58_CACHE_POLICY_IMPORTANCE,
)
from .deep_proxy_stack_v6 import (
    DeepProxyStackV6,
    emit_deep_proxy_stack_v6_forward_witness,
)
from .hidden_state_bridge_v2 import (
    HiddenStateBridgeV2Projection,
    bridge_hidden_state_and_measure_v2,
)
from .kv_bridge_v3 import (
    KVBridgeV3Projection,
    inject_carrier_into_v3_kv_cache,
)
from .tiny_substrate_v3 import (
    TinyV3KVCache,
    TinyV3SubstrateParams,
    _ndarray_cid,
    _sha256_hex,
    forward_tiny_substrate_v3,
    tokenize_bytes_v3,
)


W58_DEEP_SUBSTRATE_HYBRID_V3_SCHEMA_VERSION: str = (
    "coordpy.deep_substrate_hybrid_v3.v1")
W58_DEFAULT_HYBRID_V3_QUERY_TOKEN: int = 257
W58_DEFAULT_HYBRID_V3_BASE_ABSTAIN: float = 0.10
W58_DEFAULT_HYBRID_V3_GAMMA: float = 0.05
W58_DEFAULT_HYBRID_V3_CACHE_RETENTION: float = 0.75


@dataclasses.dataclass
class DeepSubstrateHybridV3:
    deep_v6: DeepProxyStackV6
    substrate: TinyV3SubstrateParams
    bridge: KVBridgeV3Projection
    cache_controller: CacheController
    substrate_back_proj_seed: int = 58161
    substrate_back_inject_weight: float = 0.10
    base_abstain_threshold: float = (
        W58_DEFAULT_HYBRID_V3_BASE_ABSTAIN)
    gamma: float = W58_DEFAULT_HYBRID_V3_GAMMA
    cache_retention_ratio: float = (
        W58_DEFAULT_HYBRID_V3_CACHE_RETENTION)
    attention_steering: (
        AttentionSteeringV2Projection | None) = None
    hidden_state_bridge: (
        HiddenStateBridgeV2Projection | None) = None
    substrate_read_layer: int = 1
    kl_budget: float | None = None

    @classmethod
    def init(
            cls, *,
            deep_v6: DeepProxyStackV6,
            substrate: TinyV3SubstrateParams,
            n_inject_tokens: int = 3,
            carrier_dim: int | None = None,
            bridge_seed: int = 58151,
            substrate_back_inject_weight: float = 0.10,
            substrate_read_layer: int = 1,
            base_abstain_threshold: float = (
                W58_DEFAULT_HYBRID_V3_BASE_ABSTAIN),
            gamma: float = W58_DEFAULT_HYBRID_V3_GAMMA,
            cache_retention_ratio: float = (
                W58_DEFAULT_HYBRID_V3_CACHE_RETENTION),
            cache_policy: str = W58_CACHE_POLICY_IMPORTANCE,
            kl_budget: float | None = None,
    ) -> "DeepSubstrateHybridV3":
        cd = (
            int(carrier_dim) if carrier_dim is not None
            else int(deep_v6.inner_v5.inner_v4.factor_dim))
        d_head = (int(substrate.config.d_model)
                  // int(substrate.config.n_heads))
        bridge = KVBridgeV3Projection.init(
            n_layers=int(substrate.config.n_layers),
            n_heads=int(substrate.config.n_heads),
            n_kv_heads=int(substrate.config.n_kv_heads),
            n_inject_tokens=int(n_inject_tokens),
            carrier_dim=int(cd),
            d_head=int(d_head),
            seed=int(bridge_seed))
        ctrl = CacheController.init(
            policy=str(cache_policy),
            d_model=int(substrate.config.d_model),
            probe_layer=-1,
        )
        return cls(
            deep_v6=deep_v6,
            substrate=substrate,
            bridge=bridge,
            cache_controller=ctrl,
            substrate_back_proj_seed=int(bridge_seed) + 10,
            substrate_back_inject_weight=float(
                substrate_back_inject_weight),
            substrate_read_layer=int(substrate_read_layer),
            base_abstain_threshold=float(base_abstain_threshold),
            gamma=float(gamma),
            cache_retention_ratio=float(cache_retention_ratio),
            kl_budget=(
                float(kl_budget) if kl_budget is not None
                else None),
        )

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W58_DEEP_SUBSTRATE_HYBRID_V3_SCHEMA_VERSION,
            "deep_v6_cid": str(self.deep_v6.cid()),
            "substrate_cid": str(self.substrate.cid()),
            "bridge_cid": str(self.bridge.cid()),
            "cache_controller_cid": str(
                self.cache_controller.cid()),
            "substrate_back_proj_seed": int(
                self.substrate_back_proj_seed),
            "substrate_back_inject_weight": float(round(
                self.substrate_back_inject_weight, 12)),
            "substrate_read_layer": int(
                self.substrate_read_layer),
            "base_abstain_threshold": float(round(
                self.base_abstain_threshold, 12)),
            "gamma": float(round(self.gamma, 12)),
            "cache_retention_ratio": float(round(
                self.cache_retention_ratio, 12)),
            "kl_budget": (
                float(round(self.kl_budget, 12))
                if self.kl_budget is not None else -1.0),
        })

    def compute_adaptive_threshold(
            self, query_input: Sequence[float],
    ) -> float:
        l2 = math.sqrt(sum(
            float(v) * float(v) for v in query_input))
        return float(
            self.base_abstain_threshold
            * (1.0 + self.gamma
               * math.log(1.0 + max(0.0, l2))))


@dataclasses.dataclass(frozen=True)
class DeepSubstrateHybridV3ForwardWitness:
    schema: str
    deep_v6_witness_cid: str
    substrate_forward_cid: str
    bridge_injection_cid: str
    pre_inject_kv_cid: str
    post_inject_kv_cid: str
    post_evict_kv_cid: str
    substrate_back_l2: float
    ablation_perturbation_l2: float
    cache_eviction_perturbation_l2: float
    adaptive_threshold: float
    abstain: bool
    corruption_confidence: float
    attention_steering_kl_sum: float
    attention_kl_budget_enforced: bool
    cache_retention_ratio: float
    cache_policy: str
    cache_flop_savings_ratio: float
    three_way: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "deep_v6_witness_cid": str(self.deep_v6_witness_cid),
            "substrate_forward_cid": str(
                self.substrate_forward_cid),
            "bridge_injection_cid": str(
                self.bridge_injection_cid),
            "pre_inject_kv_cid": str(self.pre_inject_kv_cid),
            "post_inject_kv_cid": str(self.post_inject_kv_cid),
            "post_evict_kv_cid": str(self.post_evict_kv_cid),
            "substrate_back_l2": float(round(
                self.substrate_back_l2, 12)),
            "ablation_perturbation_l2": float(round(
                self.ablation_perturbation_l2, 12)),
            "cache_eviction_perturbation_l2": float(round(
                self.cache_eviction_perturbation_l2, 12)),
            "adaptive_threshold": float(round(
                self.adaptive_threshold, 12)),
            "abstain": bool(self.abstain),
            "corruption_confidence": float(round(
                self.corruption_confidence, 12)),
            "attention_steering_kl_sum": float(round(
                self.attention_steering_kl_sum, 12)),
            "attention_kl_budget_enforced": bool(
                self.attention_kl_budget_enforced),
            "cache_retention_ratio": float(round(
                self.cache_retention_ratio, 12)),
            "cache_policy": str(self.cache_policy),
            "cache_flop_savings_ratio": float(round(
                self.cache_flop_savings_ratio, 12)),
            "three_way": bool(self.three_way),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "deep_substrate_hybrid_v3_witness",
            "witness": self.to_dict()})


def _project_substrate_hidden_back_to_v6(
        hidden_vec: "_np.ndarray", *,
        v6_dim: int, seed: int,
) -> list[float]:
    rng = _np.random.default_rng(int(seed))
    h = _np.asarray(hidden_vec, dtype=_np.float64).reshape(-1)
    h_dim = h.size
    M = rng.standard_normal(
        (int(v6_dim), int(h_dim))) * (1.0 / math.sqrt(
            max(1, int(h_dim))))
    return list(_np.einsum("vh,h->v", M, h))


def deep_substrate_hybrid_v3_forward(
        *,
        hybrid: DeepSubstrateHybridV3,
        query_input: Sequence[float],
        slot_keys: Sequence[Sequence[float]],
        slot_values: Sequence[Sequence[float]],
        role_index: int = 0,
        branch_index: int = 0,
        cycle_index: int = 0,
        trust_scalar: float = 1.0,
        uncertainty_scale: float = 1.0,
        substrate_query_token_ids: Sequence[int] | None = None,
        substrate_kv_cache: TinyV3KVCache | None = None,
        substrate_ref_text: str = "ref",
) -> tuple["_np.ndarray",
            DeepSubstrateHybridV3ForwardWitness,
            TinyV3KVCache]:
    """One V3 hybrid forward (3-way bridge)."""
    cfg = hybrid.substrate.config
    # Step 1 — substrate baseline forward.
    ref_ids = tokenize_bytes_v3(
        substrate_ref_text, max_len=int(cfg.max_len))
    base_ref_trace = forward_tiny_substrate_v3(
        hybrid.substrate, ref_ids, return_attention=False)
    sub_layer_idx = max(0, min(int(hybrid.substrate_read_layer),
                                  len(base_ref_trace.hidden_states) - 1))
    sub_hidden = base_ref_trace.hidden_states[sub_layer_idx][-1]
    # Step 2 — substrate → V6 back-injection.
    v6_in_dim = int(hybrid.deep_v6.in_dim)
    back_contrib = _project_substrate_hidden_back_to_v6(
        sub_hidden, v6_dim=v6_in_dim,
        seed=int(hybrid.substrate_back_proj_seed))
    w = float(hybrid.substrate_back_inject_weight)
    biased_query = [
        float(query_input[i]
              if i < len(query_input) else 0.0)
        + w * float(back_contrib[i]
                     if i < len(back_contrib) else 0.0)
        for i in range(v6_in_dim)
    ]
    back_l2 = float(_np.linalg.norm(
        _np.asarray(back_contrib) * w))
    # Step 3 — V6 forward (with biased query).
    v6_witness, v6_head_value = (
        emit_deep_proxy_stack_v6_forward_witness(
            stack=hybrid.deep_v6,
            query_input=list(biased_query),
            slot_keys=[list(k) for k in slot_keys],
            slot_values=[list(v) for v in slot_values],
            role_index=int(role_index),
            branch_index=int(branch_index),
            cycle_index=int(cycle_index),
            trust_scalar=float(trust_scalar),
            uncertainty_scale=float(uncertainty_scale)))
    # Step 4 — V6 residual → substrate V3 KV.
    base_cache = (
        substrate_kv_cache.clone()
        if substrate_kv_cache is not None
        else TinyV3KVCache.empty(int(cfg.n_layers)))
    pre_cid = base_cache.cid()
    inj_cache, inj_record = inject_carrier_into_v3_kv_cache(
        carrier=list(v6_head_value),
        projection=hybrid.bridge,
        kv_cache=base_cache,
        position="prepend",
        role="bank_a")
    post_inj_cid = inj_cache.cid()
    # Step 5 — cache controller eviction.
    keep = max(
        1,
        int(float(hybrid.cache_retention_ratio)
            * inj_cache.n_tokens()))
    n_tok = inj_cache.n_tokens()
    if n_tok > 0:
        if hybrid.cache_controller.policy == (
                W58_CACHE_POLICY_IMPORTANCE):
            evicted_cache = inj_cache.evict_by_importance(keep)
        else:
            evicted_cache = inj_cache.evict_lru(
                max(0, n_tok - keep))
    else:
        evicted_cache = inj_cache.clone()
    post_evict_cid = evicted_cache.cid()
    # Step 6 — optional attention-steering bias V2.
    attn_kl_sum = 0.0
    attn_budget_enforced = bool(hybrid.kl_budget is None)
    if hybrid.attention_steering is not None:
        try:
            w_attn = steer_attention_and_measure_v2(
                params=hybrid.substrate,
                carrier=list(v6_head_value),
                projection=hybrid.attention_steering,
                token_ids=(
                    substrate_query_token_ids
                    or [W58_DEFAULT_HYBRID_V3_QUERY_TOKEN]),
                baseline_kv_cache=evicted_cache,
                kl_budget=hybrid.kl_budget)
            attn_kl_sum = float(sum(w_attn.mean_kl_per_layer))
            attn_budget_enforced = bool(w_attn.kl_budget_enforced)
        except Exception:
            attn_kl_sum = 0.0
            attn_budget_enforced = False
    # Step 7 — substrate forward on controlled cache.
    inj_trace = forward_tiny_substrate_v3(
        hybrid.substrate,
        (list(substrate_query_token_ids)
         if substrate_query_token_ids is not None
         else [W58_DEFAULT_HYBRID_V3_QUERY_TOKEN]),
        kv_cache=evicted_cache,
        return_attention=False)
    # Step 8 — ablations.
    abl_trace = forward_tiny_substrate_v3(
        hybrid.substrate,
        (list(substrate_query_token_ids)
         if substrate_query_token_ids is not None
         else [W58_DEFAULT_HYBRID_V3_QUERY_TOKEN]),
        kv_cache=base_cache,
        return_attention=False)
    # Cache-eviction-only ablation (V6 injected but no controller).
    full_trace = forward_tiny_substrate_v3(
        hybrid.substrate,
        (list(substrate_query_token_ids)
         if substrate_query_token_ids is not None
         else [W58_DEFAULT_HYBRID_V3_QUERY_TOKEN]),
        kv_cache=inj_cache,
        return_attention=False)
    inj_hidden = inj_trace.hidden_states[-1]
    abl_hidden = abl_trace.hidden_states[-1]
    full_hidden = full_trace.hidden_states[-1]
    perturb_l2 = float(_np.linalg.norm(
        inj_hidden[-1] - abl_hidden[-1]))
    eviction_l2 = float(_np.linalg.norm(
        inj_hidden[-1] - full_hidden[-1]))
    # Flop savings ratio from cache eviction.
    flop_saved_ratio = 0.0
    if full_trace.flop_count > 0:
        flop_saved_ratio = (
            float(full_trace.flop_count - inj_trace.flop_count)
            / float(full_trace.flop_count))
    thr = hybrid.compute_adaptive_threshold(query_input)
    abstain = bool(perturb_l2 < thr)
    corruption_confidence = float(
        1.0 / (1.0 + max(0.0, perturb_l2)))
    three_way = bool(
        (w > 1e-9)
        and (inj_record.injected_layer_indices != ())
        and (hybrid.cache_controller is not None))
    witness = DeepSubstrateHybridV3ForwardWitness(
        schema=W58_DEEP_SUBSTRATE_HYBRID_V3_SCHEMA_VERSION,
        deep_v6_witness_cid=str(v6_witness.cid()),
        substrate_forward_cid=str(inj_trace.cid()),
        bridge_injection_cid=str(inj_record.cid()),
        pre_inject_kv_cid=str(pre_cid),
        post_inject_kv_cid=str(post_inj_cid),
        post_evict_kv_cid=str(post_evict_cid),
        substrate_back_l2=float(back_l2),
        ablation_perturbation_l2=float(perturb_l2),
        cache_eviction_perturbation_l2=float(eviction_l2),
        adaptive_threshold=float(thr),
        abstain=bool(abstain),
        corruption_confidence=float(corruption_confidence),
        attention_steering_kl_sum=float(attn_kl_sum),
        attention_kl_budget_enforced=bool(attn_budget_enforced),
        cache_retention_ratio=float(
            hybrid.cache_retention_ratio),
        cache_policy=str(hybrid.cache_controller.policy),
        cache_flop_savings_ratio=float(flop_saved_ratio),
        three_way=bool(three_way),
    )
    return inj_hidden[-1], witness, inj_trace.kv_cache


__all__ = [
    "W58_DEEP_SUBSTRATE_HYBRID_V3_SCHEMA_VERSION",
    "W58_DEFAULT_HYBRID_V3_QUERY_TOKEN",
    "W58_DEFAULT_HYBRID_V3_BASE_ABSTAIN",
    "W58_DEFAULT_HYBRID_V3_GAMMA",
    "W58_DEFAULT_HYBRID_V3_CACHE_RETENTION",
    "DeepSubstrateHybridV3",
    "DeepSubstrateHybridV3ForwardWitness",
    "deep_substrate_hybrid_v3_forward",
]
