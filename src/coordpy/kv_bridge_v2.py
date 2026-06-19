"""W57 M2 — KV Bridge V2.

Stronger than W56 M3:

* **Per-head projection**: the carrier projects to per-(layer, head)
  K and V slots rather than the W56 per-layer scheme. This lets
  the same carrier specialise different heads of the substrate
  for different roles.
* **Calibrated injection scale**: each layer-head pair carries an
  individual scale knob ``inject_scale_per_head[l, h]``. Callers
  can boost or suppress injection per head. Default is uniform.
* **Replay-deterministic readback**: after injection, the bridge
  re-extracts the injected slot positions from the cache and
  records their CID; this is the readback witness, used by the
  consensus controller V3 to detect cache corruption.
* **Compose-with-W56**: V2 is byte-compatible with the W56 KV
  cache layout when used against the W56 ``TinyKVCache``; the
  default V2 path targets the W57 ``TinyV2KVCache`` which has
  per-head depth implicit in the d_model split.
* **Perturbation accounting**: the bridge measures both max-abs
  logit perturbation and last-position cross-entropy delta versus
  a baseline forward; the witness records both.
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
    raise ImportError("coordpy.kv_bridge_v2 requires numpy") from exc

from .tiny_substrate_v2 import (
    TinyV2KVCache,
    TinyV2SubstrateParams,
    _ndarray_cid,
    _sha256_hex,
    forward_tiny_substrate_v2,
)


W57_KV_BRIDGE_V2_SCHEMA_VERSION: str = "coordpy.kv_bridge_v2.v1"
W57_DEFAULT_BRIDGE_V2_INJECT_TOKENS: int = 3
W57_DEFAULT_BRIDGE_V2_PROJECTION_SEED: int = 57042042
W57_DEFAULT_BRIDGE_V2_INJECT_SCALE: float = 0.30


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


@dataclasses.dataclass
class KVBridgeV2Projection:
    """Per-(layer, head) K/V projection from a fixed-dim carrier.

    Shapes:

    * ``proj_k``: ``(n_layers, n_heads, n_inject_tokens, carrier_dim, d_head)``
    * ``proj_v``: same
    * ``inject_scale_per_head``: ``(n_layers, n_heads)`` — multiplicative
      scale; defaults to a uniform ``inject_scale`` per init.

    The carrier is projected first to per-head ``d_head``-vectors,
    then concatenated across heads back into the cache's ``d_model``
    layout.
    """

    n_layers: int
    n_heads: int
    n_inject_tokens: int
    carrier_dim: int
    d_head: int
    proj_k: "_np.ndarray"
    proj_v: "_np.ndarray"
    inject_scale_per_head: "_np.ndarray"
    seed: int

    @classmethod
    def init(
            cls, *,
            n_layers: int,
            n_heads: int,
            n_inject_tokens: int,
            carrier_dim: int,
            d_head: int,
            seed: int = W57_DEFAULT_BRIDGE_V2_PROJECTION_SEED,
            inject_scale: float = (
                W57_DEFAULT_BRIDGE_V2_INJECT_SCALE),
    ) -> "KVBridgeV2Projection":
        rng = _np.random.default_rng(int(seed))
        s = float(inject_scale)
        shape = (int(n_layers), int(n_heads),
                 int(n_inject_tokens), int(carrier_dim),
                 int(d_head))
        return cls(
            n_layers=int(n_layers),
            n_heads=int(n_heads),
            n_inject_tokens=int(n_inject_tokens),
            carrier_dim=int(carrier_dim),
            d_head=int(d_head),
            proj_k=rng.standard_normal(shape) * s,
            proj_v=rng.standard_normal(shape) * s,
            inject_scale_per_head=_np.ones(
                (int(n_layers), int(n_heads)), dtype=_np.float64),
            seed=int(seed),
        )

    def project(
            self, carrier: Sequence[float],
    ) -> tuple["_np.ndarray", "_np.ndarray"]:
        """Project a 1-D carrier to per-layer K/V tensors of shape
        ``(n_layers, n_inject_tokens, d_model)`` (d_model =
        n_heads * d_head)."""
        x = _np.asarray(carrier, dtype=_np.float64).reshape(-1)
        if x.size < self.carrier_dim:
            x = _np.concatenate(
                [x, _np.zeros(self.carrier_dim - x.size,
                              dtype=_np.float64)])
        elif x.size > self.carrier_dim:
            x = x[: self.carrier_dim]
        # einsum: (L,H,T,C,Dh) , (C,) -> (L,H,T,Dh)
        keys_lh = _np.einsum("lhtcd,c->lhtd", self.proj_k, x)
        vals_lh = _np.einsum("lhtcd,c->lhtd", self.proj_v, x)
        scale = self.inject_scale_per_head[:, :, None, None]
        keys_lh = keys_lh * scale
        vals_lh = vals_lh * scale
        L, H, T, Dh = keys_lh.shape
        # merge heads: (L, H, T, Dh) -> (L, T, H*Dh)
        keys = keys_lh.transpose(0, 2, 1, 3).reshape(L, T, H * Dh)
        vals = vals_lh.transpose(0, 2, 1, 3).reshape(L, T, H * Dh)
        return keys, vals

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W57_KV_BRIDGE_V2_SCHEMA_VERSION,
            "kind": "kv_bridge_v2_projection",
            "n_layers": int(self.n_layers),
            "n_heads": int(self.n_heads),
            "n_inject_tokens": int(self.n_inject_tokens),
            "carrier_dim": int(self.carrier_dim),
            "d_head": int(self.d_head),
            "seed": int(self.seed),
            "proj_k_cid": _ndarray_cid(self.proj_k),
            "proj_v_cid": _ndarray_cid(self.proj_v),
            "inject_scale_cid": _ndarray_cid(
                self.inject_scale_per_head),
        })


@dataclasses.dataclass(frozen=True)
class KVBridgeV2Injection:
    carrier_cid: str
    projection_cid: str
    pre_inject_kv_cid: str
    post_inject_kv_cid: str
    injected_layer_indices: tuple[int, ...]
    n_inject_tokens: int
    position: str
    readback_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W57_KV_BRIDGE_V2_SCHEMA_VERSION,
            "carrier_cid": str(self.carrier_cid),
            "projection_cid": str(self.projection_cid),
            "pre_inject_kv_cid": str(self.pre_inject_kv_cid),
            "post_inject_kv_cid": str(self.post_inject_kv_cid),
            "injected_layer_indices": list(
                self.injected_layer_indices),
            "n_inject_tokens": int(self.n_inject_tokens),
            "position": str(self.position),
            "readback_cid": str(self.readback_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "kv_bridge_v2_injection",
            "record": self.to_dict()})


def _carrier_cid(carrier: Sequence[float]) -> str:
    arr = _np.asarray(
        carrier, dtype=_np.float64).reshape(-1)
    return _ndarray_cid(arr)


def inject_carrier_into_v2_kv_cache(
        *,
        carrier: Sequence[float],
        projection: KVBridgeV2Projection,
        kv_cache: TinyV2KVCache,
        layer_indices: Sequence[int] | None = None,
        position: str = "prepend",
) -> tuple[TinyV2KVCache, KVBridgeV2Injection]:
    """Inject a carrier into the V2 KV cache.

    Honest readback: after injection, the bridge re-extracts the
    inserted slot bytes and content-addresses them. The
    ``readback_cid`` lets downstream code detect cache corruption.
    """
    if layer_indices is None:
        layer_indices = tuple(range(int(projection.n_layers)))
    layer_indices = tuple(int(i) for i in layer_indices)
    pre_cid = kv_cache.cid()
    new_cache = kv_cache.clone()
    keys, values = projection.project(carrier)
    readback_pieces: list[str] = []
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
        d_model = int(projection.n_heads) * int(projection.d_head)
        if prev_k.size == 0:
            prev_k = _np.zeros((0, d_model), dtype=_np.float64)
            prev_v = _np.zeros((0, d_model), dtype=_np.float64)
        if position == "prepend":
            combined_k = _np.concatenate([k_new, prev_k], axis=0)
            combined_v = _np.concatenate([v_new, prev_v], axis=0)
            slice_for_readback = slice(0, int(k_new.shape[0]))
        elif position == "append":
            combined_k = _np.concatenate([prev_k, k_new], axis=0)
            combined_v = _np.concatenate([prev_v, v_new], axis=0)
            start = int(prev_k.shape[0])
            slice_for_readback = slice(start,
                                         start + int(k_new.shape[0]))
        else:
            raise ValueError(
                f"position must be 'prepend' or 'append', got {position!r}")
        new_cache.keys[l] = combined_k
        new_cache.values[l] = combined_v
        rb = _ndarray_cid(
            _np.concatenate([
                combined_k[slice_for_readback],
                combined_v[slice_for_readback],
            ], axis=0))
        readback_pieces.append(rb)
        new_cache.write_log.append({
            "schema": W57_KV_BRIDGE_V2_SCHEMA_VERSION,
            "kind": "kv_bridge_v2_inject",
            "layer": int(l),
            "position": str(position),
            "n_inject_tokens": int(k_new.shape[0]),
            "carrier_cid": _carrier_cid(carrier),
            "projection_cid": projection.cid(),
            "readback_cid": rb,
        })
    post_cid = new_cache.cid()
    readback_cid = hashlib.sha256(
        "|".join(readback_pieces).encode("utf-8")).hexdigest()
    record = KVBridgeV2Injection(
        carrier_cid=_carrier_cid(carrier),
        projection_cid=projection.cid(),
        pre_inject_kv_cid=pre_cid,
        post_inject_kv_cid=post_cid,
        injected_layer_indices=tuple(layer_indices),
        n_inject_tokens=int(projection.n_inject_tokens),
        position=str(position),
        readback_cid=str(readback_cid),
    )
    return new_cache, record


@dataclasses.dataclass(frozen=True)
class KVBridgeV2Witness:
    schema: str
    injection: KVBridgeV2Injection
    baseline_forward_cid: str
    injected_forward_cid: str
    max_abs_logit_perturbation: float
    last_logit_l2_perturbation: float
    last_position_argmax_baseline: int
    last_position_argmax_injected: int
    cross_entropy_delta: float

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
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "kv_bridge_v2_witness",
            "witness": self.to_dict()})


def _softmax_logits(x: "_np.ndarray") -> "_np.ndarray":
    m = float(_np.max(x))
    z = _np.exp(x - m)
    s = float(_np.sum(z))
    return z / max(s, 1e-30)


def bridge_carrier_and_measure_v2(
        *,
        params: TinyV2SubstrateParams,
        carrier: Sequence[float],
        projection: KVBridgeV2Projection,
        follow_up_token_ids: Sequence[int],
        baseline_kv_cache: TinyV2KVCache | None = None,
        layer_indices: Sequence[int] | None = None,
        position: str = "prepend",
) -> KVBridgeV2Witness:
    if baseline_kv_cache is None:
        base_cache = TinyV2KVCache.empty(
            int(projection.n_layers))
    else:
        base_cache = baseline_kv_cache.clone()
    baseline_trace = forward_tiny_substrate_v2(
        params, list(follow_up_token_ids),
        kv_cache=base_cache,
        return_attention=False)
    injected_cache, injection_record = (
        inject_carrier_into_v2_kv_cache(
            carrier=carrier,
            projection=projection,
            kv_cache=base_cache,
            layer_indices=layer_indices,
            position=position))
    injected_trace = forward_tiny_substrate_v2(
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
    return KVBridgeV2Witness(
        schema=W57_KV_BRIDGE_V2_SCHEMA_VERSION,
        injection=injection_record,
        baseline_forward_cid=baseline_trace.cid(),
        injected_forward_cid=injected_trace.cid(),
        max_abs_logit_perturbation=float(max_abs),
        last_logit_l2_perturbation=float(l2),
        last_position_argmax_baseline=int(base_argmax),
        last_position_argmax_injected=int(inj_argmax),
        cross_entropy_delta=float(ce_delta),
    )


__all__ = [
    "W57_KV_BRIDGE_V2_SCHEMA_VERSION",
    "W57_DEFAULT_BRIDGE_V2_INJECT_TOKENS",
    "W57_DEFAULT_BRIDGE_V2_PROJECTION_SEED",
    "W57_DEFAULT_BRIDGE_V2_INJECT_SCALE",
    "KVBridgeV2Projection",
    "KVBridgeV2Injection",
    "KVBridgeV2Witness",
    "inject_carrier_into_v2_kv_cache",
    "bridge_carrier_and_measure_v2",
]
