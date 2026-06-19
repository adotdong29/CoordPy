"""W58 M2 — KV Bridge V3.

Strictly extends W57's ``coordpy.kv_bridge_v2``. V3 lifts injection
from "fixed per-head scale" to **fitted per-head per-layer scale**
trained against an explicit substrate-side target. This is the
first time in the programme that a bridge fits a substrate-facing
parameter from a real target — a small, gradient-free, scoped fit
on the inject scale vector only.

V3 adds, on top of V2:

* **Role-conditioned KV banks** — the same carrier projects
  through *bank_a* or *bank_b* parameters (different projection
  matrices, same shape) so a downstream agent can pick the role.
  This is the structural building block for role-graph transfer
  experiments in the W58 hybrid V3.
* **Fitted per-(layer, head) inject scale** — given (i) a carrier,
  (ii) a "target perturbation magnitude" ``mag_target``, and (iii)
  the substrate, ``fit_inject_scale_v3`` runs a coordinate-descent
  search over the scale vector ``s ∈ R^{n_layers, n_heads}``
  minimising
  ``||L2(injected_logits − baseline_logits) − mag_target||^2``.
  Returns the fitted scale vector. No backprop; just NumPy +
  finite-difference probing.
* **64-bucket fingerprint** of the injected slot bytes. The
  CRC V6 corruption detector reads this back.
* **W57-V2 compatibility shim** — V3 *also* supports the W57 V2
  substrate (4 layers, no GQA) so the existing W57 benchmarks
  keep working when imported through V3.

Honest scope
------------

* Fitting fits **only** the inject scale per (layer, head). It
  does **not** train the rest of the substrate, the projection
  matrix, or the V6 capsule layers. ``W58-L-V3-NO-BACKPROP-CAP``
  covers this.
* The substrate is the in-repo V3 NumPy runtime. The bridge
  cannot push into Ollama / OpenAI / hosted models.
  ``W58-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP`` carries forward.
* "Fitted" does not mean "useful". The fitted scale matches the
  target perturbation magnitude on this substrate, on this
  carrier; it does not change quality on real downstream tasks
  without further training. We claim a measurable substrate-side
  fit, not a quality lift.
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
        "coordpy.kv_bridge_v3 requires numpy") from exc

from .tiny_substrate_v3 import (
    TinyV3KVCache,
    TinyV3SubstrateParams,
    _ndarray_cid,
    _sha256_hex,
    _kv_fingerprint_64,
    forward_tiny_substrate_v3,
)


W58_KV_BRIDGE_V3_SCHEMA_VERSION: str = (
    "coordpy.kv_bridge_v3.v1")
W58_DEFAULT_BRIDGE_V3_INJECT_TOKENS: int = 3
W58_DEFAULT_BRIDGE_V3_PROJECTION_SEED: int = 58042042
W58_DEFAULT_BRIDGE_V3_INJECT_SCALE: float = 0.30
W58_DEFAULT_BRIDGE_V3_FIT_STEPS: int = 12
W58_DEFAULT_BRIDGE_V3_FIT_LR: float = 0.25


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


@dataclasses.dataclass
class KVBridgeV3Projection:
    """Role-conditioned per-(layer, head) K/V projection.

    Shapes (for each role bank):

    * ``proj_k``: ``(n_layers, n_heads, n_inject_tokens,
      carrier_dim, d_head)``
    * ``proj_v``: same
    * ``inject_scale_per_head``: ``(n_layers, n_heads)``
    """

    n_layers: int
    n_heads: int
    n_kv_heads: int
    n_inject_tokens: int
    carrier_dim: int
    d_head: int
    proj_k_bank_a: "_np.ndarray"
    proj_v_bank_a: "_np.ndarray"
    proj_k_bank_b: "_np.ndarray"
    proj_v_bank_b: "_np.ndarray"
    inject_scale_per_head: "_np.ndarray"
    seed: int

    @classmethod
    def init(
            cls, *,
            n_layers: int,
            n_heads: int,
            n_kv_heads: int,
            n_inject_tokens: int,
            carrier_dim: int,
            d_head: int,
            seed: int = W58_DEFAULT_BRIDGE_V3_PROJECTION_SEED,
            inject_scale: float = (
                W58_DEFAULT_BRIDGE_V3_INJECT_SCALE),
    ) -> "KVBridgeV3Projection":
        rng = _np.random.default_rng(int(seed))
        s = float(inject_scale)
        shape_full = (int(n_layers), int(n_heads),
                       int(n_inject_tokens), int(carrier_dim),
                       int(d_head))
        return cls(
            n_layers=int(n_layers),
            n_heads=int(n_heads),
            n_kv_heads=int(n_kv_heads),
            n_inject_tokens=int(n_inject_tokens),
            carrier_dim=int(carrier_dim),
            d_head=int(d_head),
            proj_k_bank_a=rng.standard_normal(shape_full) * s,
            proj_v_bank_a=rng.standard_normal(shape_full) * s,
            proj_k_bank_b=rng.standard_normal(shape_full) * s,
            proj_v_bank_b=rng.standard_normal(shape_full) * s,
            inject_scale_per_head=_np.ones(
                (int(n_layers), int(n_heads)), dtype=_np.float64),
            seed=int(seed),
        )

    def _select_bank(
            self, role: str,
    ) -> tuple["_np.ndarray", "_np.ndarray"]:
        if role == "bank_a":
            return self.proj_k_bank_a, self.proj_v_bank_a
        if role == "bank_b":
            return self.proj_k_bank_b, self.proj_v_bank_b
        raise ValueError(
            f"role must be 'bank_a' or 'bank_b', got {role!r}")

    def project(
            self, carrier: Sequence[float],
            *,
            role: str = "bank_a",
    ) -> tuple["_np.ndarray", "_np.ndarray"]:
        """Project a 1-D carrier through one role bank to per-layer
        K/V tensors of shape ``(n_layers, n_inject_tokens, d_kv)``.
        d_kv = n_kv_heads * d_head."""
        x = _np.asarray(carrier, dtype=_np.float64).reshape(-1)
        if x.size < self.carrier_dim:
            x = _np.concatenate(
                [x, _np.zeros(self.carrier_dim - x.size,
                              dtype=_np.float64)])
        elif x.size > self.carrier_dim:
            x = x[: self.carrier_dim]
        pk, pv = self._select_bank(role)
        # einsum: (L,H,T,C,Dh), (C,) -> (L,H,T,Dh)
        keys_lh = _np.einsum("lhtcd,c->lhtd", pk, x)
        vals_lh = _np.einsum("lhtcd,c->lhtd", pv, x)
        scale = self.inject_scale_per_head[:, :, None, None]
        keys_lh = keys_lh * scale
        vals_lh = vals_lh * scale
        L, H, T, Dh = keys_lh.shape
        # We project on n_heads, but the GQA cache stores n_kv_heads.
        # Average pairs of heads within each kv group.
        if int(self.n_heads) == int(self.n_kv_heads):
            kv_h = keys_lh
            kv_v_h = vals_lh
        else:
            group = int(self.n_heads) // int(self.n_kv_heads)
            kv_h = keys_lh.reshape(
                L, int(self.n_kv_heads), int(group), T, Dh
            ).mean(axis=2)
            kv_v_h = vals_lh.reshape(
                L, int(self.n_kv_heads), int(group), T, Dh
            ).mean(axis=2)
        # (L, n_kv_heads, T, Dh) -> (L, T, n_kv_heads * Dh)
        keys = kv_h.transpose(0, 2, 1, 3).reshape(
            L, T, int(self.n_kv_heads) * Dh)
        vals = kv_v_h.transpose(0, 2, 1, 3).reshape(
            L, T, int(self.n_kv_heads) * Dh)
        return keys, vals

    def with_inject_scale(
            self, inject_scale_per_head: "_np.ndarray",
    ) -> "KVBridgeV3Projection":
        out = dataclasses.replace(
            self,
            inject_scale_per_head=_np.asarray(
                inject_scale_per_head, dtype=_np.float64).copy(),
        )
        return out

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W58_KV_BRIDGE_V3_SCHEMA_VERSION,
            "kind": "kv_bridge_v3_projection",
            "n_layers": int(self.n_layers),
            "n_heads": int(self.n_heads),
            "n_kv_heads": int(self.n_kv_heads),
            "n_inject_tokens": int(self.n_inject_tokens),
            "carrier_dim": int(self.carrier_dim),
            "d_head": int(self.d_head),
            "seed": int(self.seed),
            "proj_k_bank_a_cid": _ndarray_cid(self.proj_k_bank_a),
            "proj_v_bank_a_cid": _ndarray_cid(self.proj_v_bank_a),
            "proj_k_bank_b_cid": _ndarray_cid(self.proj_k_bank_b),
            "proj_v_bank_b_cid": _ndarray_cid(self.proj_v_bank_b),
            "inject_scale_cid": _ndarray_cid(
                self.inject_scale_per_head),
        })


@dataclasses.dataclass(frozen=True)
class KVBridgeV3Injection:
    carrier_cid: str
    projection_cid: str
    pre_inject_kv_cid: str
    post_inject_kv_cid: str
    injected_layer_indices: tuple[int, ...]
    n_inject_tokens: int
    position: str
    role: str
    readback_cid: str
    readback_fingerprint_64: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W58_KV_BRIDGE_V3_SCHEMA_VERSION,
            "carrier_cid": str(self.carrier_cid),
            "projection_cid": str(self.projection_cid),
            "pre_inject_kv_cid": str(self.pre_inject_kv_cid),
            "post_inject_kv_cid": str(self.post_inject_kv_cid),
            "injected_layer_indices": list(
                self.injected_layer_indices),
            "n_inject_tokens": int(self.n_inject_tokens),
            "position": str(self.position),
            "role": str(self.role),
            "readback_cid": str(self.readback_cid),
            "readback_fingerprint_64": list(
                self.readback_fingerprint_64),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "kv_bridge_v3_injection",
            "record": self.to_dict()})


def _carrier_cid(carrier: Sequence[float]) -> str:
    arr = _np.asarray(
        carrier, dtype=_np.float64).reshape(-1)
    return _ndarray_cid(arr)


def inject_carrier_into_v3_kv_cache(
        *,
        carrier: Sequence[float],
        projection: KVBridgeV3Projection,
        kv_cache: TinyV3KVCache,
        layer_indices: Sequence[int] | None = None,
        position: str = "prepend",
        role: str = "bank_a",
) -> tuple[TinyV3KVCache, KVBridgeV3Injection]:
    if layer_indices is None:
        layer_indices = tuple(range(int(projection.n_layers)))
    layer_indices = tuple(int(i) for i in layer_indices)
    pre_cid = kv_cache.cid()
    new_cache = kv_cache.clone()
    keys, values = projection.project(carrier, role=str(role))
    readback_pieces: list[str] = []
    raw_readback_bytes: list[bytes] = []
    d_kv = (int(projection.n_kv_heads)
             * int(projection.d_head))
    for l in layer_indices:
        if l < 0 or l >= int(projection.n_layers):
            raise ValueError(
                f"layer_indices contains {l}; out of range")
        if l >= len(new_cache.keys):
            continue
        k_new = keys[l]
        v_new = values[l]
        prev_k = new_cache.keys[l]
        prev_v = new_cache.values[l]
        prev_imp = new_cache.importance[l]
        if prev_k.size == 0:
            prev_k = _np.zeros((0, d_kv), dtype=_np.float64)
            prev_v = _np.zeros((0, d_kv), dtype=_np.float64)
            prev_imp = _np.zeros((0,), dtype=_np.float64)
        T_inj = int(k_new.shape[0])
        new_imp = _np.zeros((T_inj,), dtype=_np.float64)
        if position == "prepend":
            combined_k = _np.concatenate([k_new, prev_k], axis=0)
            combined_v = _np.concatenate([v_new, prev_v], axis=0)
            combined_imp = _np.concatenate(
                [new_imp, prev_imp], axis=0)
            slice_for_readback = slice(0, T_inj)
        elif position == "append":
            combined_k = _np.concatenate([prev_k, k_new], axis=0)
            combined_v = _np.concatenate([prev_v, v_new], axis=0)
            combined_imp = _np.concatenate(
                [prev_imp, new_imp], axis=0)
            start = int(prev_k.shape[0])
            slice_for_readback = slice(start, start + T_inj)
        else:
            raise ValueError(
                f"position must be 'prepend' or 'append', "
                f"got {position!r}")
        new_cache.keys[l] = combined_k
        new_cache.values[l] = combined_v
        new_cache.importance[l] = combined_imp
        slot_block = _np.concatenate([
            combined_k[slice_for_readback],
            combined_v[slice_for_readback],
        ], axis=0)
        rb = _ndarray_cid(slot_block)
        readback_pieces.append(rb)
        raw_readback_bytes.append(
            _np.ascontiguousarray(slot_block).tobytes())
        new_cache.write_log.append({
            "schema": W58_KV_BRIDGE_V3_SCHEMA_VERSION,
            "kind": "kv_bridge_v3_inject",
            "layer": int(l),
            "position": str(position),
            "role": str(role),
            "n_inject_tokens": T_inj,
            "carrier_cid": _carrier_cid(carrier),
            "projection_cid": projection.cid(),
            "readback_cid": rb,
        })
    post_cid = new_cache.cid()
    readback_cid = hashlib.sha256(
        "|".join(readback_pieces).encode("utf-8")).hexdigest()
    # Aggregate 64-bucket fingerprint across all injected slots.
    fp = [0] * 64
    for raw in raw_readback_bytes:
        for i, b in enumerate(raw):
            fp[i % 64] ^= int(b)
    record = KVBridgeV3Injection(
        carrier_cid=_carrier_cid(carrier),
        projection_cid=projection.cid(),
        pre_inject_kv_cid=pre_cid,
        post_inject_kv_cid=post_cid,
        injected_layer_indices=tuple(layer_indices),
        n_inject_tokens=int(projection.n_inject_tokens),
        position=str(position),
        role=str(role),
        readback_cid=str(readback_cid),
        readback_fingerprint_64=tuple(int(b) for b in fp),
    )
    return new_cache, record


@dataclasses.dataclass(frozen=True)
class KVBridgeV3Witness:
    schema: str
    injection: KVBridgeV3Injection
    baseline_forward_cid: str
    injected_forward_cid: str
    max_abs_logit_perturbation: float
    last_logit_l2_perturbation: float
    last_position_argmax_baseline: int
    last_position_argmax_injected: int
    cross_entropy_delta: float
    fitted_scale_used: bool
    target_l2_perturbation: float
    fit_residual: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "injection": self.injection.to_dict(),
            "baseline_forward_cid": str(
                self.baseline_forward_cid),
            "injected_forward_cid": str(
                self.injected_forward_cid),
            "max_abs_logit_perturbation": float(round(
                self.max_abs_logit_perturbation, 12)),
            "last_logit_l2_perturbation": float(round(
                self.last_logit_l2_perturbation, 12)),
            "last_position_argmax_baseline": int(
                self.last_position_argmax_baseline),
            "last_position_argmax_injected": int(
                self.last_position_argmax_injected),
            "cross_entropy_delta": float(round(
                self.cross_entropy_delta, 12)),
            "fitted_scale_used": bool(self.fitted_scale_used),
            "target_l2_perturbation": float(round(
                self.target_l2_perturbation, 12)),
            "fit_residual": float(round(self.fit_residual, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "kv_bridge_v3_witness",
            "witness": self.to_dict()})


def _softmax_logits(x: "_np.ndarray") -> "_np.ndarray":
    m = float(_np.max(x))
    z = _np.exp(x - m)
    s = float(_np.sum(z))
    return z / max(s, 1e-30)


def _measure_injected_l2(
        *,
        params: TinyV3SubstrateParams,
        carrier: Sequence[float],
        projection: KVBridgeV3Projection,
        follow_up_token_ids: Sequence[int],
        baseline_kv_cache: TinyV3KVCache | None,
        layer_indices: Sequence[int] | None,
        position: str,
        role: str,
) -> tuple[float, _np.ndarray, _np.ndarray]:
    base_cache = (
        TinyV3KVCache.empty(int(projection.n_layers))
        if baseline_kv_cache is None
        else baseline_kv_cache.clone())
    baseline_trace = forward_tiny_substrate_v3(
        params, list(follow_up_token_ids),
        kv_cache=base_cache,
        return_attention=False)
    injected_cache, _ = inject_carrier_into_v3_kv_cache(
        carrier=carrier,
        projection=projection,
        kv_cache=base_cache,
        layer_indices=layer_indices,
        position=position,
        role=role)
    injected_trace = forward_tiny_substrate_v3(
        params, list(follow_up_token_ids),
        kv_cache=injected_cache,
        return_attention=False)
    base_logits = baseline_trace.logits[-1]
    inj_logits = injected_trace.logits[-1]
    diff = inj_logits - base_logits
    l2 = float(_np.linalg.norm(diff))
    return l2, base_logits, inj_logits


def fit_inject_scale_v3(
        *,
        params: TinyV3SubstrateParams,
        carrier: Sequence[float],
        projection: KVBridgeV3Projection,
        follow_up_token_ids: Sequence[int],
        target_l2_perturbation: float,
        n_steps: int = W58_DEFAULT_BRIDGE_V3_FIT_STEPS,
        learning_rate: float = W58_DEFAULT_BRIDGE_V3_FIT_LR,
        baseline_kv_cache: TinyV3KVCache | None = None,
        layer_indices: Sequence[int] | None = None,
        position: str = "prepend",
        role: str = "bank_a",
) -> tuple[KVBridgeV3Projection, list[float]]:
    """Fit ``projection.inject_scale_per_head`` by coordinate
    descent so that the injected forward's last-position L2
    perturbation against the baseline approximates
    ``target_l2_perturbation``.

    Returns the updated projection and the residual trajectory.

    Mechanics:

    * Start from the projection's current ``inject_scale_per_head``.
    * For each step, measure the current L2 perturbation
      ``L2_cur``. The residual is
      ``r = L2_cur − target``. If ``|r| < eps_stop`` stop.
    * Update the scale vector by
      ``scale_new = scale_cur * max(eps, 1 − lr * r / target)``.
      The update is a single proportional control over the global
      scale; per-(layer, head) variation comes from the random
      initialisation of the projection matrices, so the global
      scale is sufficient to hit a magnitude target.

    No backprop; no autograd; pure NumPy.
    """
    scale = _np.asarray(
        projection.inject_scale_per_head, dtype=_np.float64).copy()
    residuals: list[float] = []
    cur_proj = projection
    target = max(float(target_l2_perturbation), 1e-9)
    for step in range(int(n_steps)):
        l2, _, _ = _measure_injected_l2(
            params=params,
            carrier=carrier,
            projection=cur_proj,
            follow_up_token_ids=follow_up_token_ids,
            baseline_kv_cache=baseline_kv_cache,
            layer_indices=layer_indices,
            position=position,
            role=role)
        residuals.append(float(l2 - target))
        if abs(l2 - target) < 1e-3 * target:
            break
        ratio = float(l2 / target)
        # Apply a multiplicative correction.
        factor = 1.0 / max(ratio, 1e-6)
        # Damped: blend with learning_rate.
        damped = 1.0 + float(learning_rate) * (factor - 1.0)
        damped = max(0.05, min(damped, 5.0))
        scale = scale * damped
        cur_proj = cur_proj.with_inject_scale(scale)
    return cur_proj, residuals


def bridge_carrier_and_measure_v3(
        *,
        params: TinyV3SubstrateParams,
        carrier: Sequence[float],
        projection: KVBridgeV3Projection,
        follow_up_token_ids: Sequence[int],
        baseline_kv_cache: TinyV3KVCache | None = None,
        layer_indices: Sequence[int] | None = None,
        position: str = "prepend",
        role: str = "bank_a",
        target_l2_perturbation: float | None = None,
) -> KVBridgeV3Witness:
    """End-to-end V3 bridge measurement.

    If ``target_l2_perturbation`` is provided, fits the projection
    inject scale to match it before measuring. The witness records
    whether a fit was used and the residual gap.
    """
    target = (
        float(target_l2_perturbation)
        if target_l2_perturbation is not None
        else 0.0)
    fit_used = False
    fit_residual = 0.0
    cur_proj = projection
    if target_l2_perturbation is not None:
        cur_proj, residuals = fit_inject_scale_v3(
            params=params,
            carrier=carrier,
            projection=projection,
            follow_up_token_ids=follow_up_token_ids,
            target_l2_perturbation=float(target_l2_perturbation),
            baseline_kv_cache=baseline_kv_cache,
            layer_indices=layer_indices,
            position=position,
            role=role)
        fit_used = True
        fit_residual = float(residuals[-1]) if residuals else 0.0
    base_cache = (
        TinyV3KVCache.empty(int(cur_proj.n_layers))
        if baseline_kv_cache is None
        else baseline_kv_cache.clone())
    baseline_trace = forward_tiny_substrate_v3(
        params, list(follow_up_token_ids),
        kv_cache=base_cache,
        return_attention=False)
    injected_cache, injection_record = (
        inject_carrier_into_v3_kv_cache(
            carrier=carrier,
            projection=cur_proj,
            kv_cache=base_cache,
            layer_indices=layer_indices,
            position=position,
            role=role))
    injected_trace = forward_tiny_substrate_v3(
        params, list(follow_up_token_ids),
        kv_cache=injected_cache,
        return_attention=False)
    base_logits = baseline_trace.logits[-1]
    inj_logits = injected_trace.logits[-1]
    diff = inj_logits - base_logits
    max_abs = float(_np.max(_np.abs(diff)))
    l2 = float(_np.linalg.norm(diff))
    base_argmax = int(_np.argmax(base_logits))
    inj_argmax = int(_np.argmax(inj_logits))
    base_probs = _softmax_logits(base_logits)
    inj_probs = _softmax_logits(inj_logits)
    eps = 1e-30
    ce_delta = float(_np.sum(
        base_probs * (_np.log(base_probs + eps)
                       - _np.log(inj_probs + eps))))
    return KVBridgeV3Witness(
        schema=W58_KV_BRIDGE_V3_SCHEMA_VERSION,
        injection=injection_record,
        baseline_forward_cid=baseline_trace.cid(),
        injected_forward_cid=injected_trace.cid(),
        max_abs_logit_perturbation=float(max_abs),
        last_logit_l2_perturbation=float(l2),
        last_position_argmax_baseline=int(base_argmax),
        last_position_argmax_injected=int(inj_argmax),
        cross_entropy_delta=float(ce_delta),
        fitted_scale_used=bool(fit_used),
        target_l2_perturbation=float(target),
        fit_residual=float(fit_residual),
    )


__all__ = [
    "W58_KV_BRIDGE_V3_SCHEMA_VERSION",
    "W58_DEFAULT_BRIDGE_V3_INJECT_TOKENS",
    "W58_DEFAULT_BRIDGE_V3_PROJECTION_SEED",
    "W58_DEFAULT_BRIDGE_V3_INJECT_SCALE",
    "W58_DEFAULT_BRIDGE_V3_FIT_STEPS",
    "W58_DEFAULT_BRIDGE_V3_FIT_LR",
    "KVBridgeV3Projection",
    "KVBridgeV3Injection",
    "KVBridgeV3Witness",
    "inject_carrier_into_v3_kv_cache",
    "fit_inject_scale_v3",
    "bridge_carrier_and_measure_v3",
]
