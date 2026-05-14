"""W58 M5 — Attention Steering Bridge V2.

Strictly extends W57's ``coordpy.attention_steering_bridge``. V2
adds **KL-budget enforcement** and **per-head ablation**:

1. **KL-budget enforcement**. The caller specifies
   ``kl_budget`` (per-layer mean-KL ceiling). V2 measures the
   un-clamped KL and, if any layer exceeds the budget, fits a
   single global scalar ``bias_clip`` ∈ (0, 1] by coordinate
   descent so that the per-layer mean-KL is ≤ budget on every
   layer. The clip is applied as a uniform multiplier on the bias
   tensor.
2. **Per-head ablation**. Given the steered run, V2 zeroes one
   query head at a time and re-runs the substrate. The witness
   reports the per-head L2 contribution to the final logits,
   ranked by magnitude. This is the substrate-side equivalent of
   "which head moved most under steering".

V2 strictly extends V1: with no KL budget and no ablation, V2
reduces to V1 exactly.

Honest scope
------------

* KL budget is enforced via a global scalar clip, NOT per-(layer,
  head) clipping. Per-head ablation is a *measurement*, not a
  training signal. ``W58-L-V3-NO-BACKPROP-CAP`` carries forward.
* The substrate runtime is the in-repo V3. Third-party model
  attention remains out of scope. ``W58-L-NO-THIRD-PARTY-
  SUBSTRATE-COUPLING-CAP`` carries forward.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.attention_steering_bridge_v2 requires numpy"
        ) from exc

from .tiny_substrate_v3 import (
    TinyV3KVCache,
    TinyV3SubstrateParams,
    _ndarray_cid,
    _sha256_hex,
    forward_tiny_substrate_v3,
)


W58_ATTN_STEERING_V2_SCHEMA_VERSION: str = (
    "coordpy.attention_steering_bridge_v2.v1")
W58_DEFAULT_ATTN_V2_PROJECTION_SEED: int = 58045045
W58_DEFAULT_ATTN_V2_BIAS_SCALE: float = 1.5
W58_DEFAULT_ATTN_V2_KL_BUDGET: float = 4.0
W58_DEFAULT_ATTN_V2_CLIP_STEPS: int = 12


@dataclasses.dataclass
class AttentionSteeringV2Projection:
    n_layers: int
    n_heads: int
    n_query: int
    n_key: int
    carrier_dim: int
    proj: "_np.ndarray"
    seed: int
    bias_scale: float

    @classmethod
    def init(
            cls, *,
            n_layers: int,
            n_heads: int,
            n_query: int,
            n_key: int,
            carrier_dim: int,
            seed: int = W58_DEFAULT_ATTN_V2_PROJECTION_SEED,
            bias_scale: float = W58_DEFAULT_ATTN_V2_BIAS_SCALE,
    ) -> "AttentionSteeringV2Projection":
        rng = _np.random.default_rng(int(seed))
        shape = (int(n_layers), int(n_heads), int(n_query),
                 int(n_key), int(carrier_dim))
        return cls(
            n_layers=int(n_layers),
            n_heads=int(n_heads),
            n_query=int(n_query),
            n_key=int(n_key),
            carrier_dim=int(carrier_dim),
            proj=rng.standard_normal(shape)
                * float(bias_scale),
            seed=int(seed),
            bias_scale=float(bias_scale),
        )

    def project(
            self, carrier: Sequence[float],
            *, clip: float = 1.0,
    ) -> list["_np.ndarray"]:
        x = _np.asarray(carrier, dtype=_np.float64).reshape(-1)
        if x.size < self.carrier_dim:
            x = _np.concatenate(
                [x, _np.zeros(self.carrier_dim - x.size,
                              dtype=_np.float64)])
        elif x.size > self.carrier_dim:
            x = x[: self.carrier_dim]
        c = float(clip)
        biases = []
        for l in range(int(self.n_layers)):
            b = _np.einsum("hqkc,c->hqk", self.proj[l], x) * c
            biases.append(b)
        return biases

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W58_ATTN_STEERING_V2_SCHEMA_VERSION,
            "kind": "attention_steering_v2_projection",
            "n_layers": int(self.n_layers),
            "n_heads": int(self.n_heads),
            "n_query": int(self.n_query),
            "n_key": int(self.n_key),
            "carrier_dim": int(self.carrier_dim),
            "seed": int(self.seed),
            "bias_scale": float(self.bias_scale),
            "proj_cid": _ndarray_cid(self.proj),
        })


@dataclasses.dataclass(frozen=True)
class AttentionSteeringV2Witness:
    schema: str
    projection_cid: str
    carrier_cid: str
    baseline_forward_cid: str
    steered_forward_cid: str
    mean_kl_per_layer: tuple[float, ...]
    max_abs_attention_shift_per_layer: tuple[float, ...]
    attention_pattern_shifted: bool
    kl_budget: float
    kl_budget_enforced: bool
    final_clip: float
    n_clip_steps: int
    per_head_ablation_l2: tuple[float, ...]
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
            "mean_kl_per_layer": [
                float(round(k, 12))
                for k in self.mean_kl_per_layer],
            "max_abs_attention_shift_per_layer": [
                float(round(s, 12))
                for s in self.max_abs_attention_shift_per_layer],
            "attention_pattern_shifted": bool(
                self.attention_pattern_shifted),
            "kl_budget": float(round(self.kl_budget, 12)),
            "kl_budget_enforced": bool(self.kl_budget_enforced),
            "final_clip": float(round(self.final_clip, 12)),
            "n_clip_steps": int(self.n_clip_steps),
            "per_head_ablation_l2": [
                float(round(v, 12))
                for v in self.per_head_ablation_l2],
            "n_query": int(self.n_query),
            "n_key_min": int(self.n_key_min),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "attention_steering_v2_witness",
            "witness": self.to_dict()})


def _kl_per_query(
        p: "_np.ndarray", q: "_np.ndarray",
) -> float:
    eps = 1e-30
    kl = _np.sum(p * (_np.log(p + eps) - _np.log(q + eps)),
                  axis=-1)
    return float(_np.mean(kl))


def _trim_bias(b: "_np.ndarray", n_new: int, n_all: int
                ) -> "_np.ndarray":
    """Trim/pad a ``(H, Q, K)`` bias to match ``(H, n_new, n_all)``."""
    b_trim_q = b[:, : n_new, :]
    b_trim_k = b_trim_q[:, :, : n_all]
    if b_trim_k.shape[2] < n_all:
        pad = _np.zeros(
            (b_trim_k.shape[0], b_trim_k.shape[1],
             n_all - b_trim_k.shape[2]),
            dtype=_np.float64)
        b_trim_k = _np.concatenate([b_trim_k, pad], axis=2)
    return b_trim_k


def _per_layer_biases_from_carrier(
        carrier: Sequence[float],
        projection: AttentionSteeringV2Projection,
        *,
        layer_indices: Sequence[int],
        clip: float,
        n_new: int,
        n_all: int,
) -> list["_np.ndarray | None"]:
    biases_full = projection.project(carrier, clip=float(clip))
    per_layer_bias: list["_np.ndarray | None"] = [
        None] * int(projection.n_layers)
    for l in layer_indices:
        per_layer_bias[int(l)] = _trim_bias(
            biases_full[int(l)], int(n_new), int(n_all))
    return per_layer_bias


def steer_attention_and_measure_v2(
        *,
        params: TinyV3SubstrateParams,
        carrier: Sequence[float],
        projection: AttentionSteeringV2Projection,
        token_ids: Sequence[int],
        layer_indices: Sequence[int] | None = None,
        baseline_kv_cache: TinyV3KVCache | None = None,
        kl_budget: float | None = None,
        do_per_head_ablation: bool = False,
) -> AttentionSteeringV2Witness:
    """End-to-end V2 attention steering with KL-budget + ablation."""
    if layer_indices is None:
        layer_indices = tuple(range(int(projection.n_layers)))
    layer_indices = tuple(int(i) for i in layer_indices)
    n_new = len(list(token_ids))
    n_prev = (int(baseline_kv_cache.n_tokens())
              if baseline_kv_cache is not None else 0)
    n_all = n_prev + n_new

    # Step 0: baseline forward.
    base_trace = forward_tiny_substrate_v3(
        params, list(token_ids),
        kv_cache=baseline_kv_cache,
        return_attention=True)

    # Step 1: full-scale steering.
    biases = _per_layer_biases_from_carrier(
        carrier, projection,
        layer_indices=layer_indices,
        clip=1.0,
        n_new=n_new, n_all=n_all)
    steer_trace = forward_tiny_substrate_v3(
        params, list(token_ids),
        kv_cache=baseline_kv_cache,
        return_attention=True,
        attention_bias_per_layer=biases)

    # KL budget fit (if requested).
    kl_enforced = False
    final_clip = 1.0
    n_clip_steps = 0
    if kl_budget is not None:
        budget = float(kl_budget)
        # Measure max KL across layers.
        max_kl = max(
            _kl_per_query(ba, sa)
            for ba, sa in zip(
                base_trace.attn_weights_per_layer,
                steer_trace.attn_weights_per_layer))
        clip = 1.0
        steps = 0
        # Allow a small numerical tolerance so that hitting
        # budget within 1e-6 relative counts as enforced.
        tol = max(1e-6 * budget, 1e-12)
        while max_kl > budget + tol and steps < int(
                W58_DEFAULT_ATTN_V2_CLIP_STEPS):
            ratio = float(budget / max(max_kl, 1e-12))
            # KL scales roughly with the *square* of the bias
            # magnitude, so apply sqrt(ratio) as a damped multiplier.
            damped = float(_np.sqrt(max(ratio, 1e-6)))
            damped = max(damped, 0.05)
            clip = clip * damped
            biases = _per_layer_biases_from_carrier(
                carrier, projection,
                layer_indices=layer_indices,
                clip=float(clip),
                n_new=n_new, n_all=n_all)
            steer_trace = forward_tiny_substrate_v3(
                params, list(token_ids),
                kv_cache=baseline_kv_cache,
                return_attention=True,
                attention_bias_per_layer=biases)
            max_kl = max(
                _kl_per_query(ba, sa)
                for ba, sa in zip(
                    base_trace.attn_weights_per_layer,
                    steer_trace.attn_weights_per_layer))
            steps += 1
            if max_kl <= budget + tol:
                break
        kl_enforced = bool(max_kl <= budget + tol)
        final_clip = float(clip)
        n_clip_steps = int(steps)

    # Final witness metrics.
    mean_kls: list[float] = []
    max_shifts: list[float] = []
    n_key_min = 10**9
    for ba, sa in zip(
            base_trace.attn_weights_per_layer,
            steer_trace.attn_weights_per_layer):
        mean_kls.append(_kl_per_query(ba, sa))
        max_shifts.append(float(
            _np.max(_np.abs(sa - ba))))
        n_key_min = min(n_key_min, int(ba.shape[-1]))
    shifted = bool(sum(mean_kls) > 1e-9)

    # Per-head ablation: zero one query head at a time on every
    # layer (same head index), re-run, measure L2 vs steered.
    per_head_ablation_l2: list[float] = []
    if do_per_head_ablation:
        n_heads = int(steer_trace.attn_weights_per_layer[0].shape[0])
        steered_logits = steer_trace.logits[-1]
        for head_idx in range(n_heads):
            ablated_biases: list["_np.ndarray | None"] = []
            for l in range(int(projection.n_layers)):
                b = biases[l]
                if b is None:
                    ablated_biases.append(None)
                else:
                    ab = b.copy()
                    # Drive this head's bias toward -inf for all
                    # (q, k), making softmax produce ~uniform.
                    # Cleaner: just zero this head's bias so it
                    # behaves like baseline for that head only.
                    ab[head_idx] = 0.0
                    ablated_biases.append(ab)
            abl_trace = forward_tiny_substrate_v3(
                params, list(token_ids),
                kv_cache=baseline_kv_cache,
                return_attention=False,
                attention_bias_per_layer=ablated_biases)
            per_head_ablation_l2.append(float(
                _np.linalg.norm(
                    steered_logits - abl_trace.logits[-1])))

    carrier_cid = _ndarray_cid(
        _np.asarray(carrier, dtype=_np.float64).reshape(-1))
    return AttentionSteeringV2Witness(
        schema=W58_ATTN_STEERING_V2_SCHEMA_VERSION,
        projection_cid=str(projection.cid()),
        carrier_cid=str(carrier_cid),
        baseline_forward_cid=str(base_trace.cid()),
        steered_forward_cid=str(steer_trace.cid()),
        mean_kl_per_layer=tuple(mean_kls),
        max_abs_attention_shift_per_layer=tuple(max_shifts),
        attention_pattern_shifted=bool(shifted),
        kl_budget=float(kl_budget if kl_budget is not None else 0.0),
        kl_budget_enforced=bool(kl_enforced),
        final_clip=float(final_clip),
        n_clip_steps=int(n_clip_steps),
        per_head_ablation_l2=tuple(per_head_ablation_l2),
        n_query=int(n_new),
        n_key_min=int(n_key_min),
    )


__all__ = [
    "W58_ATTN_STEERING_V2_SCHEMA_VERSION",
    "W58_DEFAULT_ATTN_V2_PROJECTION_SEED",
    "W58_DEFAULT_ATTN_V2_BIAS_SCALE",
    "W58_DEFAULT_ATTN_V2_KL_BUDGET",
    "AttentionSteeringV2Projection",
    "AttentionSteeringV2Witness",
    "steer_attention_and_measure_v2",
]
