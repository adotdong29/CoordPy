"""W59 M5 — Attention Steering Bridge V3.

Strictly extends W58's ``coordpy.attention_steering_bridge_v2``.
V2 enforced a single *global* clip scalar across all heads. V3
fits a **per-(layer, head)** clip vector so the per-head mean-KL
ceiling is honoured for every head individually — a finer
controller for attention steering.

Mechanism:

1. Start from a baseline forward (no steering).
2. Measure per-(layer, head) KL with bias clip = 1.0.
3. For each layer ``l`` and head ``h`` whose mean-KL is above the
   per-head budget ``budget_per_head``, shrink the corresponding
   bias slice by ``sqrt(budget / measured)``. This is the V3
   equivalent of V2's single-clip step, applied to each head.
4. Re-run; iterate up to ``W59_DEFAULT_ATTN_V3_CLIP_STEPS`` times.

V3 also produces a *per-head dominance* report: the per-head
contribution to the final logit shift, ranked by L2 magnitude.

V3 strictly extends V2: with the per-head budget set equal to V2's
global budget and clip restricted to a single global multiplier,
V3 reduces to V2's behaviour.

Honest scope
------------

* Per-head clipping is a *measurement-and-clip* loop, not
  back-propagation. ``W59-L-V4-NO-AUTOGRAD-CAP`` carries forward.
* Per-head dominance is a measured contribution, not a training
  signal.
* The substrate is the in-repo V3/V4 NumPy runtime.
  ``W59-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP`` carries forward.
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
        "coordpy.attention_steering_bridge_v3 requires numpy"
        ) from exc

from .attention_steering_bridge_v2 import (
    AttentionSteeringV2Projection,
    _kl_per_query,
    _per_layer_biases_from_carrier,
    _trim_bias,
)
from .tiny_substrate_v3 import (
    TinyV3KVCache,
    TinyV3SubstrateParams,
    _ndarray_cid,
    _sha256_hex,
    forward_tiny_substrate_v3,
)


W59_ATTN_STEERING_V3_SCHEMA_VERSION: str = (
    "coordpy.attention_steering_bridge_v3.v1")
W59_DEFAULT_ATTN_V3_CLIP_STEPS: int = 12
W59_DEFAULT_ATTN_V3_KL_BUDGET_PER_HEAD: float = 0.6


def _kl_per_layer_per_head(
        p: "_np.ndarray", q: "_np.ndarray",
) -> "_np.ndarray":
    """Compute per-head mean-KL between two attention matrices
    of shape ``(H, Q, K)``. Returns shape ``(H,)``."""
    eps = 1e-30
    kl = _np.sum(
        p * (_np.log(p + eps) - _np.log(q + eps)), axis=-1)
    # kl: (H, Q). Mean over queries.
    return _np.mean(kl, axis=-1)


@dataclasses.dataclass(frozen=True)
class AttentionSteeringV3Witness:
    schema: str
    projection_cid: str
    carrier_cid: str
    baseline_forward_cid: str
    steered_forward_cid: str
    per_head_clip: tuple[tuple[float, ...], ...]
    mean_kl_per_layer_per_head_post: tuple[
        tuple[float, ...], ...]
    mean_kl_per_layer_post: tuple[float, ...]
    attention_pattern_shifted: bool
    kl_budget_per_head: float
    per_head_kl_budget_enforced: bool
    n_clip_steps: int
    per_head_dominance_l2: tuple[float, ...]
    n_query: int
    n_key_min: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "projection_cid": str(self.projection_cid),
            "carrier_cid": str(self.carrier_cid),
            "baseline_forward_cid": str(
                self.baseline_forward_cid),
            "steered_forward_cid": str(
                self.steered_forward_cid),
            "per_head_clip": [
                [float(round(v, 12)) for v in row]
                for row in self.per_head_clip],
            "mean_kl_per_layer_per_head_post": [
                [float(round(v, 12)) for v in row]
                for row in self.mean_kl_per_layer_per_head_post],
            "mean_kl_per_layer_post": [
                float(round(v, 12))
                for v in self.mean_kl_per_layer_post],
            "attention_pattern_shifted": bool(
                self.attention_pattern_shifted),
            "kl_budget_per_head": float(round(
                self.kl_budget_per_head, 12)),
            "per_head_kl_budget_enforced": bool(
                self.per_head_kl_budget_enforced),
            "n_clip_steps": int(self.n_clip_steps),
            "per_head_dominance_l2": [
                float(round(v, 12))
                for v in self.per_head_dominance_l2],
            "n_query": int(self.n_query),
            "n_key_min": int(self.n_key_min),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "attention_steering_v3_witness",
            "witness": self.to_dict()})


def _apply_per_head_clip(
        biases: list["_np.ndarray | None"],
        per_head_clip: "_np.ndarray",
) -> list["_np.ndarray | None"]:
    """``per_head_clip`` shape ``(n_layers, n_heads)``."""
    out: list["_np.ndarray | None"] = []
    for l, b in enumerate(biases):
        if b is None:
            out.append(None)
            continue
        clip_row = per_head_clip[int(l)]
        scaled = b.copy()
        for h in range(int(scaled.shape[0])):
            scaled[h] = scaled[h] * float(clip_row[h])
        out.append(scaled)
    return out


def steer_attention_and_measure_v3(
        *,
        params: TinyV3SubstrateParams,
        carrier: Sequence[float],
        projection: AttentionSteeringV2Projection,
        token_ids: Sequence[int],
        layer_indices: Sequence[int] | None = None,
        baseline_kv_cache: TinyV3KVCache | None = None,
        kl_budget_per_head: float = (
            W59_DEFAULT_ATTN_V3_KL_BUDGET_PER_HEAD),
        do_per_head_dominance: bool = True,
) -> AttentionSteeringV3Witness:
    """V3 attention steering with per-(layer, head) KL clip fit."""
    if layer_indices is None:
        layer_indices = tuple(range(int(projection.n_layers)))
    layer_indices = tuple(int(i) for i in layer_indices)
    n_new = len(list(token_ids))
    n_prev = (int(baseline_kv_cache.n_tokens())
              if baseline_kv_cache is not None else 0)
    n_all = n_prev + n_new
    L = int(projection.n_layers)
    H = int(projection.n_heads)
    base = forward_tiny_substrate_v3(
        params, list(token_ids),
        kv_cache=baseline_kv_cache,
        return_attention=True)
    # Initial full-clip biases.
    biases = _per_layer_biases_from_carrier(
        carrier, projection,
        layer_indices=layer_indices,
        clip=1.0, n_new=n_new, n_all=n_all)
    per_head_clip = _np.ones((L, H), dtype=_np.float64)
    budget = float(kl_budget_per_head)
    tol = max(1e-6 * budget, 1e-12)
    n_steps = 0
    steered = base  # placeholder
    for step in range(int(W59_DEFAULT_ATTN_V3_CLIP_STEPS)):
        clipped = _apply_per_head_clip(
            biases, per_head_clip)
        steered = forward_tiny_substrate_v3(
            params, list(token_ids),
            kv_cache=baseline_kv_cache,
            return_attention=True,
            attention_bias_per_layer=clipped)
        # Per-(layer, head) KL.
        any_violation = False
        for l, (ba, sa) in enumerate(zip(
                base.attn_weights_per_layer,
                steered.attn_weights_per_layer)):
            kls = _kl_per_layer_per_head(ba, sa)
            for h in range(int(kls.shape[0])):
                k = float(kls[h])
                if k > budget + tol:
                    # KL scales like clip^2; damp.
                    ratio = float(budget / max(k, 1e-12))
                    damped = math.sqrt(max(ratio, 1e-6))
                    damped = max(damped, 0.05)
                    per_head_clip[l, h] = (
                        per_head_clip[l, h] * damped)
                    any_violation = True
        n_steps = step + 1
        if not any_violation:
            break
    # Final measurement.
    final_clipped = _apply_per_head_clip(
        biases, per_head_clip)
    steered = forward_tiny_substrate_v3(
        params, list(token_ids),
        kv_cache=baseline_kv_cache,
        return_attention=True,
        attention_bias_per_layer=final_clipped)
    per_layer_per_head_post: list[tuple[float, ...]] = []
    per_layer_post: list[float] = []
    n_key_min = 10**9
    enforced = True
    for l, (ba, sa) in enumerate(zip(
            base.attn_weights_per_layer,
            steered.attn_weights_per_layer)):
        kls = _kl_per_layer_per_head(ba, sa)
        per_layer_per_head_post.append(tuple(
            float(k) for k in kls))
        per_layer_post.append(float(_np.mean(kls)))
        if float(_np.max(kls)) > budget + tol:
            enforced = False
        n_key_min = min(n_key_min, int(ba.shape[-1]))

    shifted = bool(sum(per_layer_post) > 1e-9)
    per_head_dominance: list[float] = []
    if do_per_head_dominance:
        steered_logits = steered.logits[-1]
        for head_idx in range(int(H)):
            abl_biases = []
            for l in range(L):
                b = final_clipped[l]
                if b is None:
                    abl_biases.append(None)
                else:
                    ab = b.copy()
                    ab[head_idx] = 0.0
                    abl_biases.append(ab)
            abl_trace = forward_tiny_substrate_v3(
                params, list(token_ids),
                kv_cache=baseline_kv_cache,
                return_attention=False,
                attention_bias_per_layer=abl_biases)
            per_head_dominance.append(float(
                _np.linalg.norm(
                    steered_logits - abl_trace.logits[-1])))

    carrier_cid = _ndarray_cid(
        _np.asarray(carrier, dtype=_np.float64).reshape(-1))
    return AttentionSteeringV3Witness(
        schema=W59_ATTN_STEERING_V3_SCHEMA_VERSION,
        projection_cid=str(projection.cid()),
        carrier_cid=str(carrier_cid),
        baseline_forward_cid=str(base.cid()),
        steered_forward_cid=str(steered.cid()),
        per_head_clip=tuple(
            tuple(float(v) for v in row)
            for row in per_head_clip),
        mean_kl_per_layer_per_head_post=tuple(
            per_layer_per_head_post),
        mean_kl_per_layer_post=tuple(per_layer_post),
        attention_pattern_shifted=bool(shifted),
        kl_budget_per_head=float(kl_budget_per_head),
        per_head_kl_budget_enforced=bool(enforced),
        n_clip_steps=int(n_steps),
        per_head_dominance_l2=tuple(per_head_dominance),
        n_query=int(n_new),
        n_key_min=int(n_key_min),
    )


__all__ = [
    "W59_ATTN_STEERING_V3_SCHEMA_VERSION",
    "W59_DEFAULT_ATTN_V3_CLIP_STEPS",
    "W59_DEFAULT_ATTN_V3_KL_BUDGET_PER_HEAD",
    "AttentionSteeringV3Witness",
    "steer_attention_and_measure_v3",
]
