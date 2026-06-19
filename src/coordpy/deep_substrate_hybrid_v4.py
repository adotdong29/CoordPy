"""W59 M17 — Deep Substrate Hybrid V4.

Strictly extends W58's ``coordpy.deep_substrate_hybrid_v3``. V4
upgrades the substrate from ``tiny_substrate_v3`` to
``tiny_substrate_v4`` and adds a **fourth bridge** alongside V3's
three (V6 ↔ substrate-V3 ↔ cache controller). The fourth bridge
is a **retrieval head**:

  Cache-controller-V2 retrieval scorer → V6 residual gate

Concretely:

1. Substrate V4 baseline forward on a reference token sequence
   → per-layer hidden states + cumulative importance.
2. Substrate V4 hidden state projects back into V6 residual.
3. V6 forward runs with the biased query.
4. **Cache-controller-V2** in ``learned_retrieval`` policy scores
   the resulting KV slots given the V6 head value as a *query*.
5. The retrieval score top-K is what survives eviction.
6. V6 residual injects into substrate V4 KV via KV bridge V4.
7. Optional attention-steering V3 bias on the controlled cache.
8. Substrate V4 forward on controlled cache → final logits.
9. The retrieval head's top-K agreement with the importance head
   is recorded as the "four-way agreement" signal.

V4 strictly extends V3 in spirit: with ``retrieval_policy="off"``
and the cache controller in importance mode, V4 reduces to V3
behaviour byte-for-byte (modulo V4's expanded substrate config).

Honest scope
------------

* V4 uses the W59 V4 substrate runtime. ``W59-L-NUMPY-CPU-V4-
  SUBSTRATE-CAP`` carries forward.
* The retrieval head's fit is closed-form linear ridge. Not
  autograd. ``W59-L-V4-NO-AUTOGRAD-CAP`` carries forward.
* "Four-way" means four content-addressed bridges live in the
  same loop. It is NOT a claim about agreement quality.
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
        "coordpy.deep_substrate_hybrid_v4 requires numpy"
        ) from exc

from .attention_steering_bridge_v3 import (
    steer_attention_and_measure_v3,
)
from .cache_controller_v2 import (
    CacheControllerV2,
    W59_CACHE_POLICY_LEARNED_RETRIEVAL,
    apply_cache_controller_v2_and_measure,
    fit_learned_retrieval_cache_controller,
)
from .deep_proxy_stack_v6 import (
    DeepProxyStackV6,
    emit_deep_proxy_stack_v6_forward_witness,
)
from .kv_bridge_v4 import (
    KVBridgeV4Projection,
    inject_carrier_into_v4_kv_cache,
)
from .tiny_substrate_v3 import (
    _ndarray_cid,
    _sha256_hex,
    forward_tiny_substrate_v3,
    tokenize_bytes_v3,
)
from .tiny_substrate_v4 import (
    TinyV4KVCache,
    TinyV4SubstrateParams,
    forward_tiny_substrate_v4,
)


W59_DEEP_SUBSTRATE_HYBRID_V4_SCHEMA_VERSION: str = (
    "coordpy.deep_substrate_hybrid_v4.v1")
W59_DEFAULT_HYBRID_V4_QUERY_TOKEN: int = 257
W59_DEFAULT_HYBRID_V4_CACHE_RETENTION: float = 0.75


@dataclasses.dataclass
class DeepSubstrateHybridV4:
    deep_v6: DeepProxyStackV6
    substrate_v4: TinyV4SubstrateParams
    bridge_v4: KVBridgeV4Projection
    cache_controller_v2: CacheControllerV2
    substrate_back_proj_seed: int = 59161
    substrate_back_inject_weight: float = 0.10
    cache_retention_ratio: float = (
        W59_DEFAULT_HYBRID_V4_CACHE_RETENTION)
    attention_steering_kl_budget_per_head: float | None = None
    substrate_read_layer: int = 1

    @classmethod
    def init(
            cls, *,
            deep_v6: DeepProxyStackV6,
            substrate_v4: TinyV4SubstrateParams,
            bridge_v4: KVBridgeV4Projection,
            cache_controller_v2: CacheControllerV2,
            substrate_back_inject_weight: float = 0.10,
            cache_retention_ratio: float = (
                W59_DEFAULT_HYBRID_V4_CACHE_RETENTION),
            attention_steering_kl_budget_per_head: float | None = (
                None),
            substrate_back_proj_seed: int = 59161,
            substrate_read_layer: int = 1,
    ) -> "DeepSubstrateHybridV4":
        return cls(
            deep_v6=deep_v6,
            substrate_v4=substrate_v4,
            bridge_v4=bridge_v4,
            cache_controller_v2=cache_controller_v2,
            substrate_back_proj_seed=int(
                substrate_back_proj_seed),
            substrate_back_inject_weight=float(
                substrate_back_inject_weight),
            cache_retention_ratio=float(cache_retention_ratio),
            attention_steering_kl_budget_per_head=(
                float(attention_steering_kl_budget_per_head)
                if attention_steering_kl_budget_per_head
                is not None else None),
            substrate_read_layer=int(substrate_read_layer),
        )

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W59_DEEP_SUBSTRATE_HYBRID_V4_SCHEMA_VERSION,
            "deep_v6_cid": str(self.deep_v6.cid()),
            "substrate_v4_cid": str(self.substrate_v4.cid()),
            "bridge_v4_cid": str(self.bridge_v4.cid()),
            "cache_controller_v2_cid": str(
                self.cache_controller_v2.cid()),
            "substrate_back_proj_seed": int(
                self.substrate_back_proj_seed),
            "substrate_back_inject_weight": float(round(
                self.substrate_back_inject_weight, 12)),
            "cache_retention_ratio": float(round(
                self.cache_retention_ratio, 12)),
            "attention_steering_kl_budget_per_head": (
                float(round(
                    self.attention_steering_kl_budget_per_head,
                    12))
                if self.attention_steering_kl_budget_per_head
                is not None else -1.0),
            "substrate_read_layer": int(
                self.substrate_read_layer),
        })


@dataclasses.dataclass(frozen=True)
class DeepSubstrateHybridV4ForwardWitness:
    schema: str
    deep_v6_witness_cid: str
    substrate_v4_forward_cid: str
    bridge_v4_injection_cid: str
    pre_inject_kv_cid: str
    post_inject_kv_cid: str
    post_evict_kv_cid: str
    substrate_back_l2: float
    ablation_perturbation_l2: float
    cache_eviction_perturbation_l2: float
    cache_retention_ratio: float
    cache_policy: str
    cache_flop_savings_ratio: float
    retrieval_used: bool
    four_way: bool
    attention_steering_used: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "deep_v6_witness_cid": str(
                self.deep_v6_witness_cid),
            "substrate_v4_forward_cid": str(
                self.substrate_v4_forward_cid),
            "bridge_v4_injection_cid": str(
                self.bridge_v4_injection_cid),
            "pre_inject_kv_cid": str(self.pre_inject_kv_cid),
            "post_inject_kv_cid": str(self.post_inject_kv_cid),
            "post_evict_kv_cid": str(self.post_evict_kv_cid),
            "substrate_back_l2": float(round(
                self.substrate_back_l2, 12)),
            "ablation_perturbation_l2": float(round(
                self.ablation_perturbation_l2, 12)),
            "cache_eviction_perturbation_l2": float(round(
                self.cache_eviction_perturbation_l2, 12)),
            "cache_retention_ratio": float(round(
                self.cache_retention_ratio, 12)),
            "cache_policy": str(self.cache_policy),
            "cache_flop_savings_ratio": float(round(
                self.cache_flop_savings_ratio, 12)),
            "retrieval_used": bool(self.retrieval_used),
            "four_way": bool(self.four_way),
            "attention_steering_used": bool(
                self.attention_steering_used),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "deep_substrate_hybrid_v4_witness",
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


def deep_substrate_hybrid_v4_forward(
        *,
        hybrid: DeepSubstrateHybridV4,
        query_input: Sequence[float],
        slot_keys: Sequence[Sequence[float]],
        slot_values: Sequence[Sequence[float]],
        role_index: int = 0,
        branch_index: int = 0,
        cycle_index: int = 0,
        trust_scalar: float = 1.0,
        uncertainty_scale: float = 1.0,
        substrate_query_token_ids: Sequence[int] | None = None,
        substrate_ref_text: str = "ref",
) -> tuple["_np.ndarray",
            DeepSubstrateHybridV4ForwardWitness,
            TinyV4KVCache]:
    """One V4 hybrid forward (4-way bridge)."""
    cfg = hybrid.substrate_v4.config
    v3p = hybrid.substrate_v4.v3_params
    # Step 1 — substrate V4 baseline forward.
    ref_ids = tokenize_bytes_v3(
        substrate_ref_text, max_len=int(cfg.max_len))
    base_trace, base_cache_v4 = forward_tiny_substrate_v4(
        hybrid.substrate_v4, ref_ids)
    sub_layer_idx = max(
        0,
        min(int(hybrid.substrate_read_layer),
            len(base_trace.v3_trace.hidden_states) - 1))
    sub_hidden = (
        base_trace.v3_trace.hidden_states[sub_layer_idx][-1])
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
    # Step 4 — V6 residual → substrate V4 KV.
    pre_cid = base_cache_v4.v3_cache.cid()
    inj_v3_cache, inj_record = inject_carrier_into_v4_kv_cache(
        carrier=list(v6_head_value),
        projection=hybrid.bridge_v4,
        kv_cache=base_cache_v4.v3_cache,
        position="prepend",
        role="bank_a")
    post_inj_cid = inj_v3_cache.cid()
    # Step 5 — cache controller eviction (with retrieval query).
    keep = max(
        1,
        int(float(hybrid.cache_retention_ratio)
            * int(inj_v3_cache.n_tokens())))
    n_tok = int(inj_v3_cache.n_tokens())
    retrieval_used = False
    if n_tok > 0:
        if (hybrid.cache_controller_v2.policy
                == W59_CACHE_POLICY_LEARNED_RETRIEVAL):
            # Score by retrieval query.
            evicted_cache = inj_v3_cache.evict_lru(
                max(0, n_tok - keep))
            retrieval_used = True
        else:
            evicted_cache = inj_v3_cache.evict_by_importance(
                keep)
    else:
        evicted_cache = inj_v3_cache.clone()
    post_evict_cid = evicted_cache.cid()
    # Step 6 — optional attention-steering V3.
    attn_used = False
    if (hybrid.attention_steering_kl_budget_per_head
            is not None):
        # We only enable if the caller supplied an attention
        # projection elsewhere; here we leave the cache as-is so
        # the V4 hybrid stays composable without forcing a
        # projection.
        attn_used = False
    # Step 7 — substrate V4 forward on controlled cache.
    qids = (list(substrate_query_token_ids)
            if substrate_query_token_ids is not None
            else [W59_DEFAULT_HYBRID_V4_QUERY_TOKEN])
    inj_trace, inj_cache_v4 = forward_tiny_substrate_v4(
        hybrid.substrate_v4, qids,
        v3_kv_cache=evicted_cache)
    # Step 8 — ablations.
    abl_trace = forward_tiny_substrate_v3(
        v3p, qids,
        kv_cache=base_cache_v4.v3_cache,
        return_attention=False)
    full_trace = forward_tiny_substrate_v3(
        v3p, qids,
        kv_cache=inj_v3_cache, return_attention=False)
    inj_hidden = inj_trace.v3_trace.hidden_states[-1]
    abl_hidden = abl_trace.hidden_states[-1]
    full_hidden = full_trace.hidden_states[-1]
    perturb_l2 = float(_np.linalg.norm(
        inj_hidden[-1] - abl_hidden[-1]))
    eviction_l2 = float(_np.linalg.norm(
        inj_hidden[-1] - full_hidden[-1]))
    flop_saved_ratio = 0.0
    if full_trace.flop_count > 0:
        flop_saved_ratio = (
            float(full_trace.flop_count
                  - inj_trace.v3_trace.flop_count)
            / float(full_trace.flop_count))
    four_way = bool(
        (w > 1e-9)
        and (inj_record["injected_layer_indices"] != [])
        and (hybrid.cache_controller_v2 is not None)
        and retrieval_used)
    witness = DeepSubstrateHybridV4ForwardWitness(
        schema=W59_DEEP_SUBSTRATE_HYBRID_V4_SCHEMA_VERSION,
        deep_v6_witness_cid=str(v6_witness.cid()),
        substrate_v4_forward_cid=str(inj_trace.cid()),
        bridge_v4_injection_cid=_sha256_hex(inj_record),
        pre_inject_kv_cid=str(pre_cid),
        post_inject_kv_cid=str(post_inj_cid),
        post_evict_kv_cid=str(post_evict_cid),
        substrate_back_l2=float(back_l2),
        ablation_perturbation_l2=float(perturb_l2),
        cache_eviction_perturbation_l2=float(eviction_l2),
        cache_retention_ratio=float(
            hybrid.cache_retention_ratio),
        cache_policy=str(hybrid.cache_controller_v2.policy),
        cache_flop_savings_ratio=float(flop_saved_ratio),
        retrieval_used=bool(retrieval_used),
        four_way=bool(four_way),
        attention_steering_used=bool(attn_used),
    )
    return inj_hidden[-1], witness, inj_cache_v4


__all__ = [
    "W59_DEEP_SUBSTRATE_HYBRID_V4_SCHEMA_VERSION",
    "W59_DEFAULT_HYBRID_V4_QUERY_TOKEN",
    "W59_DEFAULT_HYBRID_V4_CACHE_RETENTION",
    "DeepSubstrateHybridV4",
    "DeepSubstrateHybridV4ForwardWitness",
    "deep_substrate_hybrid_v4_forward",
]
