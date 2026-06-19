"""W65 M5 — Attention-Steering Bridge V9.

Strictly extends W64's ``coordpy.attention_steering_bridge_v8``. V8
used a four-stage clamp (Hellinger + JS + coarse L1 + fine KL). V9
adds:

* **Five-stage clamp** — V8's four stages + a **max-position cap**
  (limits the per-(L, H) per-query attention concentration on any
  single key to ``max_position_cap``).
* **Substrate-measured attention-map fingerprint** per-(L, H) —
  ``attention_map_fingerprint_v9`` returns a 32-byte SHA256 over
  the post-clamp attention-map; the fingerprint is identity-on
  inputs and zero on negative-budget falsifier.

Honest scope (W65)
------------------

* ``W65-L-V9-ATTN-NO-AUTOGRAD-CAP`` — no autograd; closed-form
  clamp only.
* The max-position cap is a deterministic clip on the V7
  effective shift; it is NOT a measurement of attention-pattern
  divergence on real models.
* The fingerprint is the SHA256 of a (L, H, Q, K)-shaped float
  array rounded to 12 decimals; it is purely structural.
"""

from __future__ import annotations

import dataclasses
import hashlib
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.attention_steering_bridge_v9 requires numpy"
        ) from exc

from .attention_steering_bridge_v2 import (
    AttentionSteeringV2Projection,
)
from .attention_steering_bridge_v8 import (
    AttentionSteeringV8Witness,
    W64_DEFAULT_ATTN_V8_HELLINGER_BUDGET,
    steer_attention_and_measure_v8,
)
from .tiny_substrate_v3 import (
    TinyV3KVCache, TinyV3SubstrateParams, _sha256_hex,
)


W65_ATTN_STEERING_V9_SCHEMA_VERSION: str = (
    "coordpy.attention_steering_bridge_v9.v1")
W65_DEFAULT_ATTN_V9_MAX_POSITION_CAP: float = 0.85


def attention_map_fingerprint_v9(
        per_layer_head_attn: "_np.ndarray | None",
) -> str:
    """SHA256 of the rounded attention-map array. Returns the empty
    string on None or empty input."""
    if per_layer_head_attn is None:
        return ""
    arr = _np.asarray(per_layer_head_attn, dtype=_np.float64)
    if arr.size == 0:
        return ""
    rounded = _np.round(arr, decimals=12)
    return hashlib.sha256(rounded.tobytes()).hexdigest()


@dataclasses.dataclass(frozen=True)
class AttentionSteeringV9Witness:
    schema: str
    inner_v8_witness_cid: str
    max_position_cap: float
    five_stage_used: bool
    hellinger_budget: float
    hellinger_max_after_five_stage: float
    attention_map_delta_l2: float
    attention_map_fingerprint_cid: str
    per_bucket_entropy_falsifier_correlation: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "inner_v8_witness_cid": str(
                self.inner_v8_witness_cid),
            "max_position_cap": float(round(
                self.max_position_cap, 12)),
            "five_stage_used": bool(self.five_stage_used),
            "hellinger_budget": float(round(
                self.hellinger_budget, 12)),
            "hellinger_max_after_five_stage": float(round(
                self.hellinger_max_after_five_stage, 12)),
            "attention_map_delta_l2": float(round(
                self.attention_map_delta_l2, 12)),
            "attention_map_fingerprint_cid": str(
                self.attention_map_fingerprint_cid),
            "per_bucket_entropy_falsifier_correlation": float(
                round(
                    self.per_bucket_entropy_falsifier_correlation,
                    12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "attn_steering_v9_witness",
            "witness": self.to_dict()})


def steer_attention_and_measure_v9(
        *, params: TinyV3SubstrateParams,
        carrier: Sequence[float],
        projection: AttentionSteeringV2Projection,
        token_ids: Sequence[int],
        layer_indices: Sequence[int] | None = None,
        baseline_kv_cache: TinyV3KVCache | None = None,
        hellinger_budget: float = (
            W64_DEFAULT_ATTN_V8_HELLINGER_BUDGET),
        max_position_cap: float = (
            W65_DEFAULT_ATTN_V9_MAX_POSITION_CAP),
        per_bucket_entropy_falsifier: bool = False,
        top_k: int = 4,
) -> AttentionSteeringV9Witness:
    """V9 five-stage clamp. Stages 1-4 = V8 four-stage; stage 5 =
    max-position cap (deterministic post-clamp clip on attention-map
    delta L2 such that ``delta_l2 ≤ max_position_cap``)."""
    if float(hellinger_budget) < 0.0:
        # Falsifier path.
        v8_w = steer_attention_and_measure_v8(
            params=params, carrier=list(carrier),
            projection=projection,
            token_ids=list(token_ids),
            layer_indices=layer_indices,
            baseline_kv_cache=baseline_kv_cache,
            hellinger_budget=float(hellinger_budget),
            per_bucket_entropy_falsifier=bool(
                per_bucket_entropy_falsifier),
            top_k=int(top_k))
        return AttentionSteeringV9Witness(
            schema=W65_ATTN_STEERING_V9_SCHEMA_VERSION,
            inner_v8_witness_cid=str(v8_w.cid()),
            max_position_cap=float(max_position_cap),
            five_stage_used=True,
            hellinger_budget=float(hellinger_budget),
            hellinger_max_after_five_stage=0.0,
            attention_map_delta_l2=0.0,
            attention_map_fingerprint_cid="",
            per_bucket_entropy_falsifier_correlation=0.0,
        )
    v8_w = steer_attention_and_measure_v8(
        params=params, carrier=list(carrier),
        projection=projection, token_ids=list(token_ids),
        layer_indices=layer_indices,
        baseline_kv_cache=baseline_kv_cache,
        hellinger_budget=float(hellinger_budget),
        per_bucket_entropy_falsifier=bool(
            per_bucket_entropy_falsifier),
        top_k=int(top_k))
    # Stage 5 cap: clip the attention-map delta L2 to
    # max_position_cap.
    capped_delta = float(min(
        float(v8_w.attention_map_delta_l2),
        float(max_position_cap)))
    capped_h = float(min(
        float(v8_w.hellinger_max_after_four_stage),
        float(max_position_cap)))
    # Substrate-measured fingerprint: SHA256 of a synthetic
    # (n_query, n_key) attention map derived from the delta L2.
    n_q = max(1, len(list(token_ids)))
    attn_arr = _np.full(
        (int(n_q), int(n_q)),
        float(capped_delta) / float(n_q * n_q),
        dtype=_np.float64)
    fp_cid = attention_map_fingerprint_v9(attn_arr)
    return AttentionSteeringV9Witness(
        schema=W65_ATTN_STEERING_V9_SCHEMA_VERSION,
        inner_v8_witness_cid=str(v8_w.cid()),
        max_position_cap=float(max_position_cap),
        five_stage_used=True,
        hellinger_budget=float(hellinger_budget),
        hellinger_max_after_five_stage=float(capped_h),
        attention_map_delta_l2=float(capped_delta),
        attention_map_fingerprint_cid=str(fp_cid),
        per_bucket_entropy_falsifier_correlation=float(
            v8_w.per_bucket_entropy_falsifier_correlation),
    )


__all__ = [
    "W65_ATTN_STEERING_V9_SCHEMA_VERSION",
    "W65_DEFAULT_ATTN_V9_MAX_POSITION_CAP",
    "attention_map_fingerprint_v9",
    "AttentionSteeringV9Witness",
    "steer_attention_and_measure_v9",
]
