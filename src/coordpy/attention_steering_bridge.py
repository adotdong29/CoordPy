"""W57 M5 — Attention Steering Bridge.

Writes a *pre-softmax attention bias tensor* into the substrate's
forward pass, biasing the attention distribution toward specific
key positions. Mechanism:

1. Project the carrier to a ``(n_heads, n_tokens, n_key_positions)``
   bias tensor for one or more layers.
2. Pass the per-layer biases into the substrate forward through
   the ``attention_bias_per_layer`` hook.
3. The substrate honours the bias byte-for-byte: it adds the
   bias to the scaled scores BEFORE the causal mask + softmax.
4. Measure the KL divergence between the un-biased and biased
   attention distributions per head and per query position.

This is the most invasive substrate-coupling mechanism in W57:
the attention pattern itself becomes a function of the capsule
carrier.

Honest scope:

* The bridge changes the attention distribution; whether that
  improves downstream behaviour depends on training. We do NOT
  claim "steering improves quality"; we claim "steering is
  load-bearing on the substrate's attention".
* Tiny in-repo substrate only. W56-L-NO-THIRD-PARTY-SUBSTRATE-
  COUPLING-CAP carries forward.
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
        "coordpy.attention_steering_bridge requires numpy") from exc

from .tiny_substrate_v2 import (
    TinyV2KVCache,
    TinyV2SubstrateParams,
    _ndarray_cid,
    _sha256_hex,
    forward_tiny_substrate_v2,
)


W57_ATTN_STEERING_SCHEMA_VERSION: str = (
    "coordpy.attention_steering_bridge.v1")
W57_DEFAULT_ATTN_PROJECTION_SEED: int = 57045045
W57_DEFAULT_ATTN_BIAS_SCALE: float = 1.5


@dataclasses.dataclass
class AttentionSteeringProjection:
    """Carrier → per-head attention bias tensor.

    Shapes: ``proj`` of shape
    ``(n_layers, n_heads, n_query, n_key, carrier_dim)``. The
    projection contracts the carrier into the last axis to
    produce the bias.
    """

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
            seed: int = W57_DEFAULT_ATTN_PROJECTION_SEED,
            bias_scale: float = W57_DEFAULT_ATTN_BIAS_SCALE,
    ) -> "AttentionSteeringProjection":
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
    ) -> list["_np.ndarray"]:
        """Return a list of length ``n_layers`` of
        ``(n_heads, n_query, n_key)`` bias tensors."""
        x = _np.asarray(carrier, dtype=_np.float64).reshape(-1)
        if x.size < self.carrier_dim:
            x = _np.concatenate(
                [x, _np.zeros(self.carrier_dim - x.size,
                              dtype=_np.float64)])
        elif x.size > self.carrier_dim:
            x = x[: self.carrier_dim]
        biases = []
        for l in range(int(self.n_layers)):
            b = _np.einsum("hqkc,c->hqk", self.proj[l], x)
            biases.append(b)
        return biases

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W57_ATTN_STEERING_SCHEMA_VERSION,
            "kind": "attention_steering_projection",
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
class AttentionSteeringWitness:
    schema: str
    projection_cid: str
    carrier_cid: str
    baseline_forward_cid: str
    steered_forward_cid: str
    mean_kl_per_layer: tuple[float, ...]
    max_abs_attention_shift_per_layer: tuple[float, ...]
    attention_pattern_shifted: bool
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
            "n_query": int(self.n_query),
            "n_key_min": int(self.n_key_min),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "attention_steering_witness",
            "witness": self.to_dict()})


def _kl_per_query(
        p: "_np.ndarray", q: "_np.ndarray",
) -> float:
    """Mean KL(p || q) over heads × query positions of a
    ``(H, Q, K)`` attention tensor pair."""
    eps = 1e-30
    kl = _np.sum(p * (_np.log(p + eps) - _np.log(q + eps)),
                  axis=-1)
    return float(_np.mean(kl))


def steer_attention_and_measure(
        *,
        params: TinyV2SubstrateParams,
        carrier: Sequence[float],
        projection: AttentionSteeringProjection,
        token_ids: Sequence[int],
        layer_indices: Sequence[int] | None = None,
        baseline_kv_cache: TinyV2KVCache | None = None,
) -> AttentionSteeringWitness:
    """End-to-end attention-steering measurement.

    Builds two forward passes: baseline (no bias) and steered (per-
    layer bias from the carrier). Compares attention distributions
    per head and reports KL + max-abs shift.
    """
    if layer_indices is None:
        layer_indices = tuple(range(int(projection.n_layers)))
    layer_indices = tuple(int(i) for i in layer_indices)
    biases_full = projection.project(carrier)
    n_new = len(list(token_ids))
    n_prev = (int(baseline_kv_cache.n_tokens())
              if baseline_kv_cache is not None else 0)
    n_all = n_prev + n_new
    # Trim bias to match the actual (n_query, n_key) of this run.
    per_layer_bias: list["_np.ndarray | None"] = [
        None] * int(projection.n_layers)
    for l in layer_indices:
        b = biases_full[l]  # (H, Q, K)
        b_trim_q = b[:, : n_new, :]
        b_trim_k = b_trim_q[:, :, : n_all]
        # If the actual run has more keys than the projection
        # configured for, zero-pad the unused tail with zeros.
        if b_trim_k.shape[2] < n_all:
            pad = _np.zeros(
                (b_trim_k.shape[0], b_trim_k.shape[1],
                 n_all - b_trim_k.shape[2]),
                dtype=_np.float64)
            b_trim_k = _np.concatenate(
                [b_trim_k, pad], axis=2)
        per_layer_bias[l] = b_trim_k
    base_trace = forward_tiny_substrate_v2(
        params, list(token_ids),
        kv_cache=baseline_kv_cache,
        return_attention=True)
    steer_trace = forward_tiny_substrate_v2(
        params, list(token_ids),
        kv_cache=baseline_kv_cache,
        return_attention=True,
        attention_bias_per_layer=per_layer_bias)
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
    carrier_cid = _ndarray_cid(
        _np.asarray(carrier, dtype=_np.float64).reshape(-1))
    return AttentionSteeringWitness(
        schema=W57_ATTN_STEERING_SCHEMA_VERSION,
        projection_cid=str(projection.cid()),
        carrier_cid=str(carrier_cid),
        baseline_forward_cid=str(base_trace.cid()),
        steered_forward_cid=str(steer_trace.cid()),
        mean_kl_per_layer=tuple(mean_kls),
        max_abs_attention_shift_per_layer=tuple(max_shifts),
        attention_pattern_shifted=bool(shifted),
        n_query=int(n_new),
        n_key_min=int(n_key_min),
    )


__all__ = [
    "W57_ATTN_STEERING_SCHEMA_VERSION",
    "W57_DEFAULT_ATTN_PROJECTION_SEED",
    "W57_DEFAULT_ATTN_BIAS_SCALE",
    "AttentionSteeringProjection",
    "AttentionSteeringWitness",
    "steer_attention_and_measure",
]
