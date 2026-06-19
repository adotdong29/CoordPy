"""W60 M5 — Attention Steering Bridge V4.

Strictly extends W59's ``coordpy.attention_steering_bridge_v3``. V3
fit a per-(layer, head) clip vector by iterative shrink-per-head
under a uniform per-head budget. V4 adds:

* **Per-(layer, head, query) budgets** — V3's clip is a 2-D
  tensor; V4 supports a *3-D* per-(layer, head, query) budget
  tensor so different query positions can carry different KL
  ceilings. This is load-bearing for the W60 ``ReplayController``
  which wants to KL-cap retrieval-token positions tightly while
  letting filler tokens drift.
* **Measurable attention-map deltas** — V4 reports the per-(layer,
  head, query) **post-vs-pre attention probability mass shift**
  alongside the KL — the L1 distance between the post and pre
  attention rows. Lets the cache controller V3 score "how much
  this head's attention pattern actually moved" rather than just
  the KL.
* **Negative-budget falsifier** — V4 supports
  ``negative_budget=True`` which sets the budget tensor to all-
  zero. The expected behaviour is that V4 must converge to the
  baseline forward (clip → 0); the falsifier is that any
  measurable post-KL > 1e-6 is a real failure.

V4 strictly extends V3: with ``per_query_budget = None`` and
``negative_budget = False``, V4 reduces to V3's
``steer_attention_and_measure_v3`` byte-for-byte (modulo the V4
schema tag and the additional attention-shift fields).

Honest scope
------------

* Per-(layer, head, query) budget is a *measurement-and-clip
  loop* with one extra axis. Still no autograd.
  ``W60-L-V5-NO-AUTOGRAD-CAP`` carries forward.
* Attention-map shift is L1 on probability mass, not a substrate-
  derived gradient.
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
        "coordpy.attention_steering_bridge_v4 requires numpy"
        ) from exc

from .attention_steering_bridge_v2 import (
    AttentionSteeringV2Projection,
    _per_layer_biases_from_carrier,
)
from .attention_steering_bridge_v3 import (
    W59_DEFAULT_ATTN_V3_CLIP_STEPS,
    W59_DEFAULT_ATTN_V3_KL_BUDGET_PER_HEAD,
    _kl_per_layer_per_head,
    steer_attention_and_measure_v3,
)
from .tiny_substrate_v3 import (
    TinyV3KVCache,
    TinyV3SubstrateParams,
    _ndarray_cid,
    _sha256_hex,
    forward_tiny_substrate_v3,
)


W60_ATTN_STEERING_V4_SCHEMA_VERSION: str = (
    "coordpy.attention_steering_bridge_v4.v1")
W60_DEFAULT_ATTN_V4_CLIP_STEPS: int = 16
W60_DEFAULT_ATTN_V4_KL_BUDGET_PER_QUERY: float = 0.4


def _kl_per_layer_per_head_per_query(
        p: "_np.ndarray", q: "_np.ndarray",
) -> "_np.ndarray":
    """Per-(head, query) KL between two attention matrices of
    shape ``(H, Q, K)``. Returns ``(H, Q)``."""
    eps = 1e-30
    return _np.sum(
        p * (_np.log(p + eps) - _np.log(q + eps)), axis=-1)


def _attention_l1_shift(
        p: "_np.ndarray", q: "_np.ndarray",
) -> "_np.ndarray":
    """Per-(head, query) L1 attention shift on the K-axis."""
    return _np.sum(_np.abs(p - q), axis=-1)


def _apply_per_head_per_query_clip(
        biases: list["_np.ndarray | None"],
        per_head_query_clip: "_np.ndarray",
) -> list["_np.ndarray | None"]:
    """``per_head_query_clip`` shape ``(L, H, Q)``; if Q dim
    differs from a layer's bias Q axis, broadcast as needed."""
    out: list["_np.ndarray | None"] = []
    for l, b in enumerate(biases):
        if b is None:
            out.append(None)
            continue
        clip_lh = per_head_query_clip[int(l)]
        # b shape: (H, Q, K). clip_lh shape: (H, Q).
        scaled = b.copy()
        H = int(scaled.shape[0])
        Q = int(scaled.shape[1])
        Hc, Qc = clip_lh.shape
        for h in range(min(H, Hc)):
            for q in range(min(Q, Qc)):
                scaled[h, q] = (
                    scaled[h, q] * float(clip_lh[h, q]))
        out.append(scaled)
    return out


@dataclasses.dataclass(frozen=True)
class AttentionSteeringV4Witness:
    schema: str
    projection_cid: str
    carrier_cid: str
    baseline_forward_cid: str
    steered_forward_cid: str
    per_head_query_clip_cid: str
    per_head_query_kl_post_max: float
    per_head_query_kl_post_mean: float
    per_head_query_l1_shift_max: float
    per_head_query_l1_shift_mean: float
    attention_pattern_shifted: bool
    kl_budget_per_query: float
    per_query_budget_enforced: bool
    n_clip_steps: int
    negative_budget_used: bool
    negative_budget_post_kl_max: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "projection_cid": str(self.projection_cid),
            "carrier_cid": str(self.carrier_cid),
            "baseline_forward_cid": str(
                self.baseline_forward_cid),
            "steered_forward_cid": str(
                self.steered_forward_cid),
            "per_head_query_clip_cid": str(
                self.per_head_query_clip_cid),
            "per_head_query_kl_post_max": float(round(
                self.per_head_query_kl_post_max, 12)),
            "per_head_query_kl_post_mean": float(round(
                self.per_head_query_kl_post_mean, 12)),
            "per_head_query_l1_shift_max": float(round(
                self.per_head_query_l1_shift_max, 12)),
            "per_head_query_l1_shift_mean": float(round(
                self.per_head_query_l1_shift_mean, 12)),
            "attention_pattern_shifted": bool(
                self.attention_pattern_shifted),
            "kl_budget_per_query": float(round(
                self.kl_budget_per_query, 12)),
            "per_query_budget_enforced": bool(
                self.per_query_budget_enforced),
            "n_clip_steps": int(self.n_clip_steps),
            "negative_budget_used": bool(
                self.negative_budget_used),
            "negative_budget_post_kl_max": float(round(
                self.negative_budget_post_kl_max, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "attention_steering_v4_witness",
            "witness": self.to_dict()})


def steer_attention_and_measure_v4(
        *,
        params: TinyV3SubstrateParams,
        carrier: Sequence[float],
        projection: AttentionSteeringV2Projection,
        token_ids: Sequence[int],
        layer_indices: Sequence[int] | None = None,
        baseline_kv_cache: TinyV3KVCache | None = None,
        per_query_budget: "_np.ndarray | float | None" = None,
        kl_budget_per_query: float = (
            W60_DEFAULT_ATTN_V4_KL_BUDGET_PER_QUERY),
        negative_budget: bool = False,
) -> AttentionSteeringV4Witness:
    """V4 attention steering with per-(layer, head, query) clip.

    ``per_query_budget`` shape ``(L, H, Q)`` or scalar or None
    (defaults to a uniform budget = ``kl_budget_per_query``).
    ``negative_budget=True`` overrides the budget to all-zero
    (the controller must clip the bias to a no-op).
    """
    if layer_indices is None:
        layer_indices = tuple(range(int(projection.n_layers)))
    layer_indices = tuple(int(i) for i in layer_indices)
    n_new = len(list(token_ids))
    n_prev = (int(baseline_kv_cache.n_tokens())
              if baseline_kv_cache is not None else 0)
    n_all = n_prev + n_new
    L = int(projection.n_layers)
    H = int(projection.n_heads)
    Q = int(n_new)
    if per_query_budget is None:
        budget_arr = _np.full(
            (L, H, Q), float(kl_budget_per_query),
            dtype=_np.float64)
    elif isinstance(per_query_budget, (int, float)):
        budget_arr = _np.full(
            (L, H, Q), float(per_query_budget),
            dtype=_np.float64)
    else:
        budget_arr = _np.asarray(
            per_query_budget, dtype=_np.float64)
        # Broadcast to (L, H, Q) if smaller.
        if budget_arr.shape != (L, H, Q):
            tmp = _np.full(
                (L, H, Q),
                float(_np.mean(budget_arr)),
                dtype=_np.float64)
            tmp[:budget_arr.shape[0],
                :budget_arr.shape[1],
                :budget_arr.shape[2]] = budget_arr
            budget_arr = tmp
    if bool(negative_budget):
        budget_arr = _np.zeros_like(budget_arr)
    base = forward_tiny_substrate_v3(
        params, list(token_ids),
        kv_cache=baseline_kv_cache,
        return_attention=True)
    biases = _per_layer_biases_from_carrier(
        carrier, projection,
        layer_indices=layer_indices,
        clip=1.0, n_new=n_new, n_all=n_all)
    per_clip = _np.ones((L, H, Q), dtype=_np.float64)
    if bool(negative_budget):
        per_clip = _np.zeros_like(per_clip)
    tol = 1e-6
    n_steps = 0
    steered = base
    for step in range(int(W60_DEFAULT_ATTN_V4_CLIP_STEPS)):
        clipped = _apply_per_head_per_query_clip(
            biases, per_clip)
        steered = forward_tiny_substrate_v3(
            params, list(token_ids),
            kv_cache=baseline_kv_cache,
            return_attention=True,
            attention_bias_per_layer=clipped)
        any_violation = False
        for l, (ba, sa) in enumerate(zip(
                base.attn_weights_per_layer,
                steered.attn_weights_per_layer)):
            kls = _kl_per_layer_per_head_per_query(ba, sa)
            # kls: (H, Q)
            for h in range(int(kls.shape[0])):
                for q in range(int(kls.shape[1])):
                    k = float(kls[h, q])
                    bgt = float(budget_arr[
                        l, h, min(q, Q - 1)])
                    if k > bgt + tol:
                        if bgt <= 1e-12:
                            per_clip[l, h, q] = 0.0
                        else:
                            ratio = bgt / max(k, 1e-12)
                            damped = math.sqrt(
                                max(ratio, 1e-6))
                            damped = max(damped, 0.05)
                            per_clip[l, h, q] = (
                                per_clip[l, h, q] * damped)
                        any_violation = True
        n_steps = step + 1
        if not any_violation:
            break
    final_clipped = _apply_per_head_per_query_clip(
        biases, per_clip)
    steered = forward_tiny_substrate_v3(
        params, list(token_ids),
        kv_cache=baseline_kv_cache,
        return_attention=True,
        attention_bias_per_layer=final_clipped)
    # Post measurements.
    kl_max = 0.0
    kl_sum = 0.0
    kl_count = 0
    l1_max = 0.0
    l1_sum = 0.0
    l1_count = 0
    enforced = True
    for l, (ba, sa) in enumerate(zip(
            base.attn_weights_per_layer,
            steered.attn_weights_per_layer)):
        kls = _kl_per_layer_per_head_per_query(ba, sa)
        l1 = _attention_l1_shift(ba, sa)
        for h in range(int(kls.shape[0])):
            for q in range(int(kls.shape[1])):
                k = float(kls[h, q])
                if k > kl_max:
                    kl_max = k
                kl_sum += k
                kl_count += 1
                bgt = float(budget_arr[
                    l, h, min(q, Q - 1)])
                if k > bgt + 1e-3:
                    enforced = False
                lv = float(l1[h, q])
                if lv > l1_max:
                    l1_max = lv
                l1_sum += lv
                l1_count += 1
    kl_mean = kl_sum / max(1, kl_count)
    l1_mean = l1_sum / max(1, l1_count)
    shifted = bool(kl_sum > 1e-9 or l1_sum > 1e-9)
    carrier_cid = _ndarray_cid(
        _np.asarray(carrier, dtype=_np.float64).reshape(-1))
    return AttentionSteeringV4Witness(
        schema=W60_ATTN_STEERING_V4_SCHEMA_VERSION,
        projection_cid=str(projection.cid()),
        carrier_cid=str(carrier_cid),
        baseline_forward_cid=str(base.cid()),
        steered_forward_cid=str(steered.cid()),
        per_head_query_clip_cid=_ndarray_cid(per_clip),
        per_head_query_kl_post_max=float(kl_max),
        per_head_query_kl_post_mean=float(kl_mean),
        per_head_query_l1_shift_max=float(l1_max),
        per_head_query_l1_shift_mean=float(l1_mean),
        attention_pattern_shifted=bool(shifted),
        kl_budget_per_query=float(kl_budget_per_query),
        per_query_budget_enforced=bool(enforced),
        n_clip_steps=int(n_steps),
        negative_budget_used=bool(negative_budget),
        negative_budget_post_kl_max=(
            float(kl_max) if bool(negative_budget) else 0.0),
    )


__all__ = [
    "W60_ATTN_STEERING_V4_SCHEMA_VERSION",
    "W60_DEFAULT_ATTN_V4_CLIP_STEPS",
    "W60_DEFAULT_ATTN_V4_KL_BUDGET_PER_QUERY",
    "AttentionSteeringV4Witness",
    "steer_attention_and_measure_v4",
]
