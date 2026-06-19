"""W56 M3 — KV bridge.

Bridges a capsule-layer latent carrier into the tiny substrate's
KV cache. This is the load-bearing test of "does a capsule
carrier actually change the model's output when injected as KV?"

The bridge does three things:

1. **Project** a fixed-dim latent carrier (e.g. an MLSC payload,
   a V8 persistent state, or a TVS arbiter selection) into
   per-layer (K, V) slot pairs of shape ``(n_inject_tokens,
   d_model)``.
2. **Inject** those slots into one or more layers of the tiny
   substrate's KV cache. The injection is *real*: the cache is
   modified in place (or in a cloned cache) and the next forward
   pass attends over the injected keys.
3. **Run** a forward pass on a follow-up prompt to measure
   whether the injection changed the output.

Injection points
----------------

* ``prepend`` — injected K/V appear *before* the prompt tokens.
  The attention causally attends to them as if they were prior
  tokens in the sequence.
* ``per_layer`` — different injection slots per layer (the
  default in the deep substrate hybrid path).

Determinism
-----------

The bridge is byte-deterministic given: substrate params CID,
injection payload bytes, projection matrix (seeded), and the
follow-up prompt. The bridge emits a content-addressed witness
binding all of these inputs plus the forward CID of the
post-injection pass.

Honest scope
------------

* This only bridges into the **tiny in-repo substrate**.
  ``W56-L-REAL-LLM-KV-BRIDGE-CAP`` documents that the same
  bridge cannot be invoked against Ollama / OpenAI / hosted
  backends — their HTTP surface does not accept KV injection.
* The projection is a deterministic-seeded linear map. There is
  no training claim. We are measuring "can a capsule carrier
  measurably perturb a real substrate's logits", not "can a
  capsule carrier *helpfully* steer a real substrate".
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover - numpy is required.
    raise ImportError("coordpy.kv_bridge requires numpy") from exc

from .tiny_substrate import (
    TinyForwardTrace,
    TinyKVCache,
    TinySubstrateParams,
    _ndarray_cid,
    _sha256_hex,
    forward_tiny_substrate,
)


W56_KV_BRIDGE_SCHEMA_VERSION: str = "coordpy.kv_bridge.v1"
W56_DEFAULT_BRIDGE_INJECT_TOKENS: int = 2
W56_DEFAULT_BRIDGE_PROJECTION_SEED: int = 56042042
W56_DEFAULT_BRIDGE_INJECT_SCALE: float = 0.25


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


@dataclasses.dataclass
class KVBridgeProjection:
    """A deterministic-seeded linear projection from a fixed-dim
    latent carrier vector to per-layer (K, V) slot tensors.

    The projection has shape
    ``(n_layers, 2, n_inject_tokens, carrier_dim, d_model)``:
    the ``(K, V)`` pair for each of ``n_layers`` layers, each with
    ``n_inject_tokens`` slots, each slot mapping the
    ``carrier_dim``-vector to a ``d_model``-vector.

    Initialisation is deterministic given a seed. There is no
    training in W56; the projection is fixed. The point is to
    test substrate coupling, not to optimise it.
    """

    n_layers: int
    n_inject_tokens: int
    carrier_dim: int
    d_model: int
    proj_k: "_np.ndarray"
    proj_v: "_np.ndarray"
    seed: int
    inject_scale: float

    @classmethod
    def init(
            cls, *,
            n_layers: int,
            n_inject_tokens: int,
            carrier_dim: int,
            d_model: int,
            seed: int = W56_DEFAULT_BRIDGE_PROJECTION_SEED,
            inject_scale: float = (
                W56_DEFAULT_BRIDGE_INJECT_SCALE),
    ) -> "KVBridgeProjection":
        rng = _np.random.default_rng(int(seed))
        scale = float(inject_scale)
        return cls(
            n_layers=int(n_layers),
            n_inject_tokens=int(n_inject_tokens),
            carrier_dim=int(carrier_dim),
            d_model=int(d_model),
            proj_k=rng.standard_normal(
                (int(n_layers), int(n_inject_tokens),
                 int(carrier_dim), int(d_model))) * scale,
            proj_v=rng.standard_normal(
                (int(n_layers), int(n_inject_tokens),
                 int(carrier_dim), int(d_model))) * scale,
            seed=int(seed),
            inject_scale=scale,
        )

    def project(
            self, carrier: Sequence[float],
    ) -> tuple["_np.ndarray", "_np.ndarray"]:
        """Project a 1-D carrier to per-layer K/V slot tensors.

        Returns ``(keys, values)`` of shape
        ``(n_layers, n_inject_tokens, d_model)``.
        """
        x = _np.asarray(carrier, dtype=_np.float64).reshape(-1)
        if x.size < self.carrier_dim:
            x = _np.concatenate(
                [x, _np.zeros(self.carrier_dim - x.size,
                              dtype=_np.float64)])
        elif x.size > self.carrier_dim:
            x = x[: self.carrier_dim]
        # einsum: (L, T, C, D), (C,) -> (L, T, D)
        keys = _np.einsum("ltcd,c->ltd", self.proj_k, x)
        values = _np.einsum("ltcd,c->ltd", self.proj_v, x)
        return keys, values

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W56_KV_BRIDGE_SCHEMA_VERSION,
            "kind": "kv_bridge_projection",
            "n_layers": int(self.n_layers),
            "n_inject_tokens": int(self.n_inject_tokens),
            "carrier_dim": int(self.carrier_dim),
            "d_model": int(self.d_model),
            "seed": int(self.seed),
            "inject_scale": float(self.inject_scale),
            "proj_k_cid": _ndarray_cid(self.proj_k),
            "proj_v_cid": _ndarray_cid(self.proj_v),
        })


@dataclasses.dataclass(frozen=True)
class KVBridgeInjection:
    """One bridge injection record.

    Binds the carrier hash, the projection CID, the pre-injection
    KV cache CID, the post-injection KV cache CID, and the
    layers that were injected into.
    """

    carrier_cid: str
    projection_cid: str
    pre_inject_kv_cid: str
    post_inject_kv_cid: str
    injected_layer_indices: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W56_KV_BRIDGE_SCHEMA_VERSION,
            "carrier_cid": str(self.carrier_cid),
            "projection_cid": str(self.projection_cid),
            "pre_inject_kv_cid": str(self.pre_inject_kv_cid),
            "post_inject_kv_cid": str(self.post_inject_kv_cid),
            "injected_layer_indices": list(
                self.injected_layer_indices),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "kv_bridge_injection",
            "record": self.to_dict()})


def _carrier_cid(carrier: Sequence[float]) -> str:
    arr = _np.asarray(
        carrier, dtype=_np.float64).reshape(-1)
    return _ndarray_cid(arr)


def inject_carrier_into_kv_cache(
        *,
        carrier: Sequence[float],
        projection: KVBridgeProjection,
        kv_cache: TinyKVCache,
        layer_indices: Sequence[int] | None = None,
        position: str = "prepend",
) -> tuple[TinyKVCache, KVBridgeInjection]:
    """Inject a projected carrier into one or more layers' KV.

    ``position`` is ``"prepend"`` (the default; injected slots
    appear *before* the existing cache) or ``"append"`` (slots
    appear after the existing cache, before any future token).
    Most W56 callers use ``"prepend"`` so the injection is fully
    in-context for any follow-up token.

    Returns a *new* KV cache (the original is not mutated) plus an
    ``KVBridgeInjection`` record describing what happened.
    """
    if layer_indices is None:
        layer_indices = tuple(range(int(projection.n_layers)))
    layer_indices = tuple(int(i) for i in layer_indices)
    pre_cid = kv_cache.cid()
    new_cache = kv_cache.clone()
    keys, values = projection.project(carrier)
    # keys[l] / values[l] : (n_inject_tokens, d_model)
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
        if prev_k.size == 0:
            prev_k = _np.zeros(
                (0, int(projection.d_model)), dtype=_np.float64)
            prev_v = _np.zeros(
                (0, int(projection.d_model)), dtype=_np.float64)
        if position == "prepend":
            combined_k = _np.concatenate([k_new, prev_k], axis=0)
            combined_v = _np.concatenate([v_new, prev_v], axis=0)
        elif position == "append":
            combined_k = _np.concatenate([prev_k, k_new], axis=0)
            combined_v = _np.concatenate([prev_v, v_new], axis=0)
        else:
            raise ValueError(
                f"position must be 'prepend' or 'append', got {position!r}")
        new_cache.keys[l] = combined_k
        new_cache.values[l] = combined_v
        new_cache.write_log.append({
            "schema": W56_KV_BRIDGE_SCHEMA_VERSION,
            "kind": "kv_bridge_inject",
            "layer": int(l),
            "position": str(position),
            "n_inject_tokens": int(k_new.shape[0]),
            "carrier_cid": _carrier_cid(carrier),
            "projection_cid": projection.cid(),
        })
    post_cid = new_cache.cid()
    record = KVBridgeInjection(
        carrier_cid=_carrier_cid(carrier),
        projection_cid=projection.cid(),
        pre_inject_kv_cid=pre_cid,
        post_inject_kv_cid=post_cid,
        injected_layer_indices=tuple(layer_indices),
    )
    return new_cache, record


@dataclasses.dataclass(frozen=True)
class KVBridgeWitness:
    """W56 KV-bridge witness.

    Records: carrier CID, projection CID, baseline forward CID
    (without injection), injected forward CID (with injection),
    measured logit perturbation (max abs diff of last-position
    logits, baseline vs injected).
    """

    schema: str
    injection: KVBridgeInjection
    baseline_forward_cid: str
    injected_forward_cid: str
    max_abs_logit_perturbation: float
    last_logit_l2_perturbation: float
    last_position_argmax_baseline: int
    last_position_argmax_injected: int

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
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "kv_bridge_witness",
            "witness": self.to_dict()})


def bridge_carrier_and_measure(
        *,
        params: TinySubstrateParams,
        carrier: Sequence[float],
        projection: KVBridgeProjection,
        follow_up_token_ids: Sequence[int],
        baseline_kv_cache: TinyKVCache | None = None,
        layer_indices: Sequence[int] | None = None,
        position: str = "prepend",
) -> KVBridgeWitness:
    """End-to-end bridge measurement.

    1. Build the baseline KV cache (either supplied or empty).
    2. Forward over ``follow_up_token_ids`` on the baseline cache.
    3. Inject the carrier into a *clone* of the baseline cache.
    4. Forward over ``follow_up_token_ids`` on the injected cache.
    5. Measure the max-abs and L2 perturbation of last-position
       logits and the argmax flip status.

    Returns a content-addressed ``KVBridgeWitness``.
    """
    if baseline_kv_cache is None:
        base_cache = TinyKVCache.empty(int(projection.n_layers))
    else:
        base_cache = baseline_kv_cache.clone()
    baseline_trace = forward_tiny_substrate(
        params, list(follow_up_token_ids),
        kv_cache=base_cache,
        return_attention=False)
    # The injection happens on the *baseline* cache (before the
    # follow-up tokens), so the follow-up tokens attend to the
    # injected K/V.
    injected_cache, injection_record = (
        inject_carrier_into_kv_cache(
            carrier=carrier,
            projection=projection,
            kv_cache=base_cache,
            layer_indices=layer_indices,
            position=position))
    injected_trace = forward_tiny_substrate(
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
    return KVBridgeWitness(
        schema=W56_KV_BRIDGE_SCHEMA_VERSION,
        injection=injection_record,
        baseline_forward_cid=baseline_trace.cid(),
        injected_forward_cid=injected_trace.cid(),
        max_abs_logit_perturbation=float(max_abs),
        last_logit_l2_perturbation=float(l2),
        last_position_argmax_baseline=int(base_argmax),
        last_position_argmax_injected=int(inj_argmax),
    )


__all__ = [
    "W56_KV_BRIDGE_SCHEMA_VERSION",
    "W56_DEFAULT_BRIDGE_INJECT_TOKENS",
    "W56_DEFAULT_BRIDGE_PROJECTION_SEED",
    "W56_DEFAULT_BRIDGE_INJECT_SCALE",
    "KVBridgeProjection",
    "KVBridgeInjection",
    "KVBridgeWitness",
    "inject_carrier_into_kv_cache",
    "bridge_carrier_and_measure",
]
