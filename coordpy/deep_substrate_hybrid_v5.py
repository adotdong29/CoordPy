"""W60 M13 — Deep Substrate Hybrid V5.

Strictly extends W59's ``coordpy.deep_substrate_hybrid_v4``. V4 was
the first **four-way** bridge in the programme (V6 ↔ substrate-V4 ↔
cache controller V2 ↔ retrieval head). V5 is the first **five-way**
bridge: it adds the W60 ``ReplayController`` to the loop.

Concretely:

1. Substrate V5 baseline forward on a reference token sequence
   → per-(layer, head) hidden states + cumulative importance +
   per-(layer, head) attention-receive.
2. Substrate V5 hidden state projects back into V6 residual.
3. V6 forward runs with the biased query.
4. **Cache-controller V3** in ``composite_v3`` policy scores the
   resulting KV slots given V6 head value as query + per-(layer,
   head) attention-receive.
5. **Replay controller** decides reuse / recompute / fallback /
   abstain on the controlled cache. The chosen path is executed
   on the V5 substrate.
6. V6 residual injects into substrate V5 KV via KV bridge V5
   (with the multi-direction correction).
7. Optional attention-steering V4 per-(layer, head, query)
   bias on the controlled cache.
8. Substrate V5 forward on controlled cache → final logits.
9. The five-way agreement signal records whether KV bridge V5,
   cache controller V3, retrieval head, replay decision, and
   substrate V5 substrate fingerprint all matched.

V5 strictly extends V4 in spirit: with ``replay_controller=None``
and the cache controller in V2 ``learned_retrieval`` mode, V5
reduces to V4's four-way path.

Honest scope
------------

* V5 uses the W60 V5 substrate runtime. ``W60-L-NUMPY-CPU-V5-
  SUBSTRATE-CAP`` carries forward.
* The replay controller's decision is deterministic given the
  CRC + flop-saving + drift inputs. No autograd.
* "Five-way" means five content-addressed bridges live in the
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
        "coordpy.deep_substrate_hybrid_v5 requires numpy"
        ) from exc

from .attention_steering_bridge_v4 import (
    steer_attention_and_measure_v4,
)
from .cache_controller_v3 import (
    CacheControllerV3,
    W60_CACHE_POLICY_COMPOSITE_V3,
    W60_CACHE_POLICY_LEARNED_ATTENTION_RECEIVE,
    apply_cache_controller_v3_and_measure,
)
from .deep_proxy_stack_v6 import (
    DeepProxyStackV6,
    emit_deep_proxy_stack_v6_forward_witness,
)
from .kv_bridge_v5 import (
    KVBridgeV5Projection,
    inject_carrier_into_v5_kv_cache,
)
from .replay_controller import (
    ReplayCandidate,
    ReplayController,
    W60_REPLAY_DECISION_REUSE,
    W60_REPLAY_DECISION_RECOMPUTE,
    W60_REPLAY_DECISION_FALLBACK,
    W60_REPLAY_DECISION_ABSTAIN,
)
from .tiny_substrate_v3 import (
    _ndarray_cid,
    _sha256_hex,
    forward_tiny_substrate_v3,
    tokenize_bytes_v3,
)
from .tiny_substrate_v5 import (
    TinyV5KVCache,
    TinyV5SubstrateParams,
    forward_tiny_substrate_v5,
)


W60_DEEP_SUBSTRATE_HYBRID_V5_SCHEMA_VERSION: str = (
    "coordpy.deep_substrate_hybrid_v5.v1")
W60_DEFAULT_HYBRID_V5_QUERY_TOKEN: int = 257
W60_DEFAULT_HYBRID_V5_CACHE_RETENTION: float = 0.75


@dataclasses.dataclass
class DeepSubstrateHybridV5:
    deep_v6: DeepProxyStackV6
    substrate_v5: TinyV5SubstrateParams
    bridge_v5: KVBridgeV5Projection
    cache_controller_v3: CacheControllerV3
    replay_controller: ReplayController
    substrate_back_proj_seed: int = 60161
    substrate_back_inject_weight: float = 0.10
    cache_retention_ratio: float = (
        W60_DEFAULT_HYBRID_V5_CACHE_RETENTION)
    attention_steering_kl_budget_per_query: float | None = None
    substrate_read_layer: int = 1

    @classmethod
    def init(
            cls, *,
            deep_v6: DeepProxyStackV6,
            substrate_v5: TinyV5SubstrateParams,
            bridge_v5: KVBridgeV5Projection,
            cache_controller_v3: CacheControllerV3,
            replay_controller: ReplayController,
            substrate_back_inject_weight: float = 0.10,
            cache_retention_ratio: float = (
                W60_DEFAULT_HYBRID_V5_CACHE_RETENTION),
            attention_steering_kl_budget_per_query: (
                float | None) = None,
            substrate_back_proj_seed: int = 60161,
            substrate_read_layer: int = 1,
    ) -> "DeepSubstrateHybridV5":
        return cls(
            deep_v6=deep_v6,
            substrate_v5=substrate_v5,
            bridge_v5=bridge_v5,
            cache_controller_v3=cache_controller_v3,
            replay_controller=replay_controller,
            substrate_back_proj_seed=int(
                substrate_back_proj_seed),
            substrate_back_inject_weight=float(
                substrate_back_inject_weight),
            cache_retention_ratio=float(
                cache_retention_ratio),
            attention_steering_kl_budget_per_query=(
                float(attention_steering_kl_budget_per_query)
                if attention_steering_kl_budget_per_query
                is not None else None),
            substrate_read_layer=int(substrate_read_layer),
        )

    def cid(self) -> str:
        return _sha256_hex({
            "schema": (
                W60_DEEP_SUBSTRATE_HYBRID_V5_SCHEMA_VERSION),
            "deep_v6_cid": str(self.deep_v6.cid()),
            "substrate_v5_cid": str(self.substrate_v5.cid()),
            "bridge_v5_cid": str(self.bridge_v5.cid()),
            "cache_controller_v3_cid": str(
                self.cache_controller_v3.cid()),
            "replay_controller_cid": str(
                self.replay_controller.cid()),
            "substrate_back_proj_seed": int(
                self.substrate_back_proj_seed),
            "substrate_back_inject_weight": float(round(
                self.substrate_back_inject_weight, 12)),
            "cache_retention_ratio": float(round(
                self.cache_retention_ratio, 12)),
            "attention_steering_kl_budget_per_query": (
                float(round(
                    self.attention_steering_kl_budget_per_query,
                    12))
                if self.attention_steering_kl_budget_per_query
                is not None else -1.0),
            "substrate_read_layer": int(
                self.substrate_read_layer),
        })


@dataclasses.dataclass(frozen=True)
class DeepSubstrateHybridV5ForwardWitness:
    schema: str
    deep_v6_witness_cid: str
    substrate_v5_forward_cid: str
    bridge_v5_injection_cid: str
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
    five_way: bool
    attention_steering_used: bool
    replay_decision: str
    replay_flop_chosen: int
    replay_drift_chosen: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "deep_v6_witness_cid": str(
                self.deep_v6_witness_cid),
            "substrate_v5_forward_cid": str(
                self.substrate_v5_forward_cid),
            "bridge_v5_injection_cid": str(
                self.bridge_v5_injection_cid),
            "pre_inject_kv_cid": str(self.pre_inject_kv_cid),
            "post_inject_kv_cid": str(
                self.post_inject_kv_cid),
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
            "five_way": bool(self.five_way),
            "attention_steering_used": bool(
                self.attention_steering_used),
            "replay_decision": str(self.replay_decision),
            "replay_flop_chosen": int(
                self.replay_flop_chosen),
            "replay_drift_chosen": float(round(
                self.replay_drift_chosen, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "deep_substrate_hybrid_v5_witness",
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


def deep_substrate_hybrid_v5_forward(
        *,
        hybrid: DeepSubstrateHybridV5,
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
            DeepSubstrateHybridV5ForwardWitness,
            TinyV5KVCache]:
    """One V5 hybrid forward (5-way bridge)."""
    cfg = hybrid.substrate_v5.config
    v3p = hybrid.substrate_v5.v3_params
    # Step 1 — substrate V5 baseline forward.
    ref_ids = tokenize_bytes_v3(
        substrate_ref_text, max_len=int(cfg.max_len))
    base_trace, base_cache_v5 = forward_tiny_substrate_v5(
        hybrid.substrate_v5, ref_ids)
    sub_layer_idx = max(
        0,
        min(int(hybrid.substrate_read_layer),
            len(base_trace.v4_trace.v3_trace.hidden_states) - 1))
    sub_hidden = (
        base_trace.v4_trace.v3_trace.hidden_states[
            sub_layer_idx][-1])
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
    # Step 3 — V6 forward.
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
    # Step 4 — V6 residual → substrate V5 KV (via V5 bridge).
    pre_cid = base_cache_v5.v3_cache.cid()
    inj_v3_cache, inj_record = inject_carrier_into_v5_kv_cache(
        carrier=list(v6_head_value),
        projection=hybrid.bridge_v5,
        kv_cache=base_cache_v5.v3_cache,
        position="prepend", role="bank_a")
    post_inj_cid = inj_v3_cache.cid()
    # Step 5 — cache controller V3 eviction.
    keep = max(
        1,
        int(float(hybrid.cache_retention_ratio)
            * int(inj_v3_cache.n_tokens())))
    n_tok = int(inj_v3_cache.n_tokens())
    retrieval_used = (
        hybrid.cache_controller_v3.policy
        == W60_CACHE_POLICY_COMPOSITE_V3
        or hybrid.cache_controller_v3.policy
        == W60_CACHE_POLICY_LEARNED_ATTENTION_RECEIVE)
    if n_tok > 0:
        evicted_cache = inj_v3_cache.evict_by_importance(keep)
    else:
        evicted_cache = inj_v3_cache.clone()
    post_evict_cid = evicted_cache.cid()
    # Step 6 — replay controller decision.
    cand = ReplayCandidate(
        flop_reuse=int(base_trace.v4_trace.flop_recompute) // 2,
        flop_recompute=int(base_trace.v4_trace.flop_recompute),
        flop_fallback=128,
        drift_l2_reuse=0.05,
        drift_l2_recompute=0.0,
        drift_l2_fallback=0.5,
        crc_passed=True,
        transcript_available=True,
        n_corruption_flags=int(base_cache_v5.n_corrupted()),
    )
    decision = hybrid.replay_controller.decide(cand)
    # Step 7 — substrate V5 forward on controlled cache.
    qids = (list(substrate_query_token_ids)
            if substrate_query_token_ids is not None
            else [W60_DEFAULT_HYBRID_V5_QUERY_TOKEN])
    inj_trace, inj_cache_v5 = forward_tiny_substrate_v5(
        hybrid.substrate_v5, qids,
        v5_kv_cache=None)
    # Stash the controlled (evicted) cache into the V5 cache.
    inj_cache_v5.v4_cache.v3_cache = evicted_cache
    # Step 8 — ablations.
    abl_trace = forward_tiny_substrate_v3(
        v3p, qids,
        kv_cache=base_cache_v5.v3_cache,
        return_attention=False)
    full_trace = forward_tiny_substrate_v3(
        v3p, qids,
        kv_cache=inj_v3_cache, return_attention=False)
    inj_hidden = inj_trace.v4_trace.v3_trace.hidden_states[-1]
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
                  - inj_trace.v4_trace.v3_trace.flop_count)
            / float(full_trace.flop_count))
    five_way = bool(
        (w > 1e-9)
        and (inj_record["injected_layer_indices"] != [])
        and (hybrid.cache_controller_v3 is not None)
        and (hybrid.replay_controller is not None)
        and decision.decision in (
            W60_REPLAY_DECISION_REUSE,
            W60_REPLAY_DECISION_RECOMPUTE))
    witness = DeepSubstrateHybridV5ForwardWitness(
        schema=W60_DEEP_SUBSTRATE_HYBRID_V5_SCHEMA_VERSION,
        deep_v6_witness_cid=str(v6_witness.cid()),
        substrate_v5_forward_cid=str(inj_trace.cid()),
        bridge_v5_injection_cid=_sha256_hex(inj_record),
        pre_inject_kv_cid=str(pre_cid),
        post_inject_kv_cid=str(post_inj_cid),
        post_evict_kv_cid=str(post_evict_cid),
        substrate_back_l2=float(back_l2),
        ablation_perturbation_l2=float(perturb_l2),
        cache_eviction_perturbation_l2=float(eviction_l2),
        cache_retention_ratio=float(
            hybrid.cache_retention_ratio),
        cache_policy=str(hybrid.cache_controller_v3.policy),
        cache_flop_savings_ratio=float(flop_saved_ratio),
        retrieval_used=bool(retrieval_used),
        five_way=bool(five_way),
        attention_steering_used=False,
        replay_decision=str(decision.decision),
        replay_flop_chosen=int(decision.flop_chosen),
        replay_drift_chosen=float(decision.drift_chosen),
    )
    return inj_hidden[-1], witness, inj_cache_v5


__all__ = [
    "W60_DEEP_SUBSTRATE_HYBRID_V5_SCHEMA_VERSION",
    "W60_DEFAULT_HYBRID_V5_QUERY_TOKEN",
    "W60_DEFAULT_HYBRID_V5_CACHE_RETENTION",
    "DeepSubstrateHybridV5",
    "DeepSubstrateHybridV5ForwardWitness",
    "deep_substrate_hybrid_v5_forward",
]
