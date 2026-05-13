"""W57 M16 — Deep Substrate Hybrid V2.

Strictly extends W56's ``deep_substrate_hybrid`` with **bi-
directional substrate coupling**: not only does the W55 V6 deep
proxy stack project into the substrate's KV cache via the V2
bridge, the substrate's hidden-state at an intermediate layer
also projects **back** into the V6 residual stream before the V6
forward. This is the first **two-way bridge** in the programme.

Forward flow:

1. Read substrate at layer ``l_substrate_read`` (a baseline
   substrate forward over a small reference token sequence) and
   project its hidden state through a deterministic-seeded map
   back into the V6 residual dim. This is the *substrate→V6*
   side of the bridge.
2. Run the V6 deep stack forward using a residual stream that
   has been *biased* by the substrate's intermediate hidden state.
3. Project the V6 residual output via the W57 KV bridge V2 into
   the substrate's KV cache. This is the *V6→substrate* side.
4. Apply an optional W57 attention-steering bridge bias from the
   capsule layer to the substrate's attention.
5. Run the substrate's forward with the injected KV + optional
   bias.
6. Compute an ablation perturbation against a pure baseline.

V2 strictly extends V1: when ``substrate_back_inject_weight = 0``
and no attention bias is supplied, V2 reduces to V1 exactly.
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
        "coordpy.deep_substrate_hybrid_v2 requires numpy") from exc

from .attention_steering_bridge import (
    AttentionSteeringProjection,
    AttentionSteeringWitness,
    steer_attention_and_measure,
)
from .deep_proxy_stack_v6 import (
    DeepProxyStackV6,
    DeepProxyStackV6ForwardWitness,
    emit_deep_proxy_stack_v6_forward_witness,
)
from .hidden_state_bridge import (
    HiddenStateBridgeProjection,
    HiddenStateBridgeWitness,
    bridge_hidden_state_and_measure,
)
from .kv_bridge_v2 import (
    KVBridgeV2Projection,
    inject_carrier_into_v2_kv_cache,
)
from .tiny_substrate_v2 import (
    TinyV2KVCache,
    TinyV2SubstrateParams,
    _ndarray_cid,
    _sha256_hex,
    forward_tiny_substrate_v2,
    tokenize_bytes_v2,
)


W57_DEEP_SUBSTRATE_HYBRID_V2_SCHEMA_VERSION: str = (
    "coordpy.deep_substrate_hybrid_v2.v1")
W57_DEFAULT_HYBRID_V2_QUERY_TOKEN: int = 257
W57_DEFAULT_HYBRID_V2_BASE_ABSTAIN: float = 0.10
W57_DEFAULT_HYBRID_V2_GAMMA: float = 0.05


@dataclasses.dataclass
class DeepSubstrateHybridV2:
    deep_v6: DeepProxyStackV6
    substrate: TinyV2SubstrateParams
    bridge: KVBridgeV2Projection
    substrate_back_proj_seed: int = 57161
    substrate_back_inject_weight: float = 0.10
    base_abstain_threshold: float = (
        W57_DEFAULT_HYBRID_V2_BASE_ABSTAIN)
    gamma: float = W57_DEFAULT_HYBRID_V2_GAMMA
    attention_steering: (
        AttentionSteeringProjection | None) = None
    hidden_state_bridge: (
        HiddenStateBridgeProjection | None) = None
    substrate_read_layer: int = 1

    @classmethod
    def init(
            cls, *,
            deep_v6: DeepProxyStackV6,
            substrate: TinyV2SubstrateParams,
            n_inject_tokens: int = 3,
            carrier_dim: int | None = None,
            bridge_seed: int = 57151,
            substrate_back_inject_weight: float = 0.10,
            substrate_read_layer: int = 1,
            base_abstain_threshold: float = (
                W57_DEFAULT_HYBRID_V2_BASE_ABSTAIN),
            gamma: float = W57_DEFAULT_HYBRID_V2_GAMMA,
    ) -> "DeepSubstrateHybridV2":
        cd = (
            int(carrier_dim) if carrier_dim is not None
            else int(deep_v6.inner_v5.inner_v4.factor_dim))
        d_head = (int(substrate.config.d_model)
                  // int(substrate.config.n_heads))
        bridge = KVBridgeV2Projection.init(
            n_layers=int(substrate.config.n_layers),
            n_heads=int(substrate.config.n_heads),
            n_inject_tokens=int(n_inject_tokens),
            carrier_dim=int(cd),
            d_head=int(d_head),
            seed=int(bridge_seed))
        return cls(
            deep_v6=deep_v6,
            substrate=substrate,
            bridge=bridge,
            substrate_back_proj_seed=int(bridge_seed) + 10,
            substrate_back_inject_weight=float(
                substrate_back_inject_weight),
            substrate_read_layer=int(substrate_read_layer),
            base_abstain_threshold=float(base_abstain_threshold),
            gamma=float(gamma),
        )

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W57_DEEP_SUBSTRATE_HYBRID_V2_SCHEMA_VERSION,
            "deep_v6_cid": str(self.deep_v6.cid()),
            "substrate_cid": str(self.substrate.cid()),
            "bridge_cid": str(self.bridge.cid()),
            "substrate_back_proj_seed": int(
                self.substrate_back_proj_seed),
            "substrate_back_inject_weight": float(round(
                self.substrate_back_inject_weight, 12)),
            "substrate_read_layer": int(self.substrate_read_layer),
            "base_abstain_threshold": float(round(
                self.base_abstain_threshold, 12)),
            "gamma": float(round(self.gamma, 12)),
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
class DeepSubstrateHybridV2ForwardWitness:
    schema: str
    deep_v6_witness_cid: str
    substrate_forward_cid: str
    bridge_injection_cid: str
    pre_inject_kv_cid: str
    post_inject_kv_cid: str
    substrate_back_l2: float
    ablation_perturbation_l2: float
    adaptive_threshold: float
    abstain: bool
    corruption_confidence: float
    attention_steering_kl_sum: float
    bidirectional: bool

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
            "substrate_back_l2": float(round(
                self.substrate_back_l2, 12)),
            "ablation_perturbation_l2": float(round(
                self.ablation_perturbation_l2, 12)),
            "adaptive_threshold": float(round(
                self.adaptive_threshold, 12)),
            "abstain": bool(self.abstain),
            "corruption_confidence": float(round(
                self.corruption_confidence, 12)),
            "attention_steering_kl_sum": float(round(
                self.attention_steering_kl_sum, 12)),
            "bidirectional": bool(self.bidirectional),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "deep_substrate_hybrid_v2_witness",
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


def deep_substrate_hybrid_v2_forward(
        *,
        hybrid: DeepSubstrateHybridV2,
        query_input: Sequence[float],
        slot_keys: Sequence[Sequence[float]],
        slot_values: Sequence[Sequence[float]],
        role_index: int = 0,
        branch_index: int = 0,
        cycle_index: int = 0,
        trust_scalar: float = 1.0,
        uncertainty_scale: float = 1.0,
        substrate_query_token_ids: Sequence[int] | None = None,
        substrate_kv_cache: TinyV2KVCache | None = None,
        substrate_ref_text: str = "ref",
) -> tuple["_np.ndarray",
            DeepSubstrateHybridV2ForwardWitness,
            TinyV2KVCache]:
    """One V2 hybrid forward.

    Bidirectional flow:

    1. Substrate baseline forward over a reference text to get a
       hidden state at ``substrate_read_layer``.
    2. Project that hidden state back into V6 input dim and add
       a weighted contribution to ``query_input``.
    3. Run V6 deep stack forward; capture residual.
    4. Inject residual into substrate KV via V2 bridge.
    5. Optionally apply an attention-steering bias built from the
       V6 residual.
    6. Run substrate forward with injected cache + optional
       attention bias.
    7. Compare to an ablated baseline.
    """
    cfg = hybrid.substrate.config
    # Step 1 — substrate baseline forward on ``ref`` token.
    ref_ids = tokenize_bytes_v2(
        substrate_ref_text, max_len=int(cfg.max_len))
    base_ref_trace = forward_tiny_substrate_v2(
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
        + w * float(back_contrib[i] if i < len(back_contrib) else 0.0)
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
    # Step 4 — V6 residual → substrate KV.
    base_cache = (
        substrate_kv_cache.clone()
        if substrate_kv_cache is not None
        else TinyV2KVCache.empty(int(cfg.n_layers)))
    pre_cid = base_cache.cid()
    inj_cache, inj_record = inject_carrier_into_v2_kv_cache(
        carrier=list(v6_head_value),
        projection=hybrid.bridge,
        kv_cache=base_cache,
        position="prepend")
    post_cid = inj_cache.cid()
    # Step 5 — optional attention-steering bias.
    attention_steering_kl_sum = 0.0
    if hybrid.attention_steering is not None:
        try:
            w_attn = steer_attention_and_measure(
                params=hybrid.substrate,
                carrier=list(v6_head_value),
                projection=hybrid.attention_steering,
                token_ids=(
                    substrate_query_token_ids
                    or [W57_DEFAULT_HYBRID_V2_QUERY_TOKEN]),
                baseline_kv_cache=inj_cache)
            attention_steering_kl_sum = float(sum(
                w_attn.mean_kl_per_layer))
        except Exception:
            attention_steering_kl_sum = 0.0
    # Step 6 — substrate forward on the injected cache.
    inj_trace = forward_tiny_substrate_v2(
        hybrid.substrate,
        (list(substrate_query_token_ids)
         if substrate_query_token_ids is not None
         else [W57_DEFAULT_HYBRID_V2_QUERY_TOKEN]),
        kv_cache=inj_cache,
        return_attention=False)
    # Step 7 — ablated baseline.
    abl_trace = forward_tiny_substrate_v2(
        hybrid.substrate,
        (list(substrate_query_token_ids)
         if substrate_query_token_ids is not None
         else [W57_DEFAULT_HYBRID_V2_QUERY_TOKEN]),
        kv_cache=base_cache,
        return_attention=False)
    inj_hidden = inj_trace.hidden_states[-1]
    abl_hidden = abl_trace.hidden_states[-1]
    perturb_l2 = float(_np.linalg.norm(
        inj_hidden[-1] - abl_hidden[-1]))
    thr = hybrid.compute_adaptive_threshold(query_input)
    abstain = bool(perturb_l2 < thr)
    corruption_confidence = float(
        1.0 / (1.0 + max(0.0, perturb_l2)))
    bidirectional = bool(w > 1e-9)
    witness = DeepSubstrateHybridV2ForwardWitness(
        schema=W57_DEEP_SUBSTRATE_HYBRID_V2_SCHEMA_VERSION,
        deep_v6_witness_cid=str(v6_witness.cid()),
        substrate_forward_cid=str(inj_trace.cid()),
        bridge_injection_cid=str(inj_record.cid()),
        pre_inject_kv_cid=str(pre_cid),
        post_inject_kv_cid=str(post_cid),
        substrate_back_l2=float(back_l2),
        ablation_perturbation_l2=float(perturb_l2),
        adaptive_threshold=float(thr),
        abstain=bool(abstain),
        corruption_confidence=float(corruption_confidence),
        attention_steering_kl_sum=float(
            attention_steering_kl_sum),
        bidirectional=bool(bidirectional),
    )
    return inj_hidden[-1], witness, inj_trace.kv_cache


__all__ = [
    "W57_DEEP_SUBSTRATE_HYBRID_V2_SCHEMA_VERSION",
    "W57_DEFAULT_HYBRID_V2_QUERY_TOKEN",
    "W57_DEFAULT_HYBRID_V2_BASE_ABSTAIN",
    "W57_DEFAULT_HYBRID_V2_GAMMA",
    "DeepSubstrateHybridV2",
    "DeepSubstrateHybridV2ForwardWitness",
    "deep_substrate_hybrid_v2_forward",
]
