"""W57 M3 — Hidden-State Bridge.

W56's KV bridge writes into the substrate's KV cache; W57 adds a
**hidden-state bridge** that injects a capsule-layer latent
carrier as an *additive perturbation* to the residual stream at a
chosen layer. The injection happens *between* the layer's
attention output and FF input, so the layer's FF + downstream
layers see the perturbed residual and the logits change in a
measurable, content-addressed way.

Mechanism:

1. Pick a target layer ``l``.
2. Project the carrier to a ``(n_tokens, d_model)`` perturbation
   via a deterministic-seeded linear map.
3. Run a forward through layers ``[0..l]`` to obtain the residual
   stream just after layer ``l-1`` (or layer 0's embedding for
   ``l=0``).
4. Add the projected perturbation to the residual stream at the
   chosen token positions.
5. Run the remaining layers ``[l..L-1]`` from the perturbed
   residual.
6. Read out logits + per-layer hidden states; compare to
   baseline.

The implementation runs *two* forwards (baseline and injected)
and records the measurable perturbation (max-abs, L2,
cross-entropy delta) in a content-addressed witness.

Honest scope:

* The bridge is *behavioural at the substrate*: the substrate's
  logits change as a function of the carrier. The bridge does
  NOT claim the change is *helpful* unless paired with training
  or evaluation. (W56's H6 honest framing carries forward.)
* The substrate is still the tiny in-repo NumPy runtime. We do
  NOT bridge into third-party models' hidden states. The W56
  ``W56-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP`` cap carries
  forward unchanged.
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
        "coordpy.hidden_state_bridge requires numpy") from exc

from .tiny_substrate_v2 import (
    TinyV2KVCache,
    TinyV2SubstrateParams,
    _gelu,
    _layer_norm,
    _ndarray_cid,
    _sha256_hex,
    _apply_rope,
    _split_heads,
    _merge_heads,
    forward_tiny_substrate_v2,
)


W57_HIDDEN_STATE_BRIDGE_SCHEMA_VERSION: str = (
    "coordpy.hidden_state_bridge.v1")
W57_DEFAULT_HSB_PROJECTION_SEED: int = 57043043
W57_DEFAULT_HSB_INJECT_SCALE: float = 0.20


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


@dataclasses.dataclass
class HiddenStateBridgeProjection:
    """Project a carrier to a ``(n_tokens, d_model)`` perturbation."""

    n_layers: int
    n_tokens: int
    carrier_dim: int
    d_model: int
    proj: "_np.ndarray"  # (n_layers, n_tokens, carrier_dim, d_model)
    seed: int
    inject_scale: float

    @classmethod
    def init(
            cls, *,
            n_layers: int,
            n_tokens: int,
            carrier_dim: int,
            d_model: int,
            seed: int = W57_DEFAULT_HSB_PROJECTION_SEED,
            inject_scale: float = W57_DEFAULT_HSB_INJECT_SCALE,
    ) -> "HiddenStateBridgeProjection":
        rng = _np.random.default_rng(int(seed))
        s = float(inject_scale)
        shape = (int(n_layers), int(n_tokens), int(carrier_dim),
                 int(d_model))
        proj = rng.standard_normal(shape) * s
        return cls(
            n_layers=int(n_layers),
            n_tokens=int(n_tokens),
            carrier_dim=int(carrier_dim),
            d_model=int(d_model),
            proj=proj,
            seed=int(seed),
            inject_scale=float(inject_scale),
        )

    def project(
            self, carrier: Sequence[float], target_layer: int,
    ) -> "_np.ndarray":
        x = _np.asarray(carrier, dtype=_np.float64).reshape(-1)
        if x.size < self.carrier_dim:
            x = _np.concatenate(
                [x, _np.zeros(self.carrier_dim - x.size,
                              dtype=_np.float64)])
        elif x.size > self.carrier_dim:
            x = x[: self.carrier_dim]
        layer_proj = self.proj[int(target_layer)]
        return _np.einsum("tcd,c->td", layer_proj, x)

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W57_HIDDEN_STATE_BRIDGE_SCHEMA_VERSION,
            "kind": "hidden_state_bridge_projection",
            "n_layers": int(self.n_layers),
            "n_tokens": int(self.n_tokens),
            "carrier_dim": int(self.carrier_dim),
            "d_model": int(self.d_model),
            "seed": int(self.seed),
            "inject_scale": float(self.inject_scale),
            "proj_cid": _ndarray_cid(self.proj),
        })


def _attention_layer_forward_for_inject(
        x: "_np.ndarray",
        layer: Any,
        *,
        kv_keys_prev: "_np.ndarray",
        kv_values_prev: "_np.ndarray",
        positions_new: "_np.ndarray",
        rope_table: "_np.ndarray",
        use_rope: bool,
) -> tuple["_np.ndarray", "_np.ndarray", "_np.ndarray"]:
    n_tokens = int(x.shape[0])
    d_model = int(x.shape[1])
    n_heads = int(layer.attn.n_heads)
    d_head = d_model // n_heads
    q = x @ layer.attn.w_q
    k_new = x @ layer.attn.w_k
    v_new = x @ layer.attn.w_v
    q_h = _split_heads(q, n_heads)
    k_new_h = _split_heads(k_new, n_heads)
    v_new_h = _split_heads(v_new, n_heads)
    if use_rope:
        q_h = _apply_rope(q_h, positions_new, rope_table)
        k_new_h = _apply_rope(k_new_h, positions_new, rope_table)
    if kv_keys_prev.size == 0:
        k_all_h = k_new_h
        v_all_h = v_new_h
    else:
        prev_k_h = _split_heads(kv_keys_prev, n_heads)
        prev_v_h = _split_heads(kv_values_prev, n_heads)
        k_all_h = _np.concatenate([prev_k_h, k_new_h], axis=1)
        v_all_h = _np.concatenate([prev_v_h, v_new_h], axis=1)
    scores = _np.einsum(
        "htd,hsd->hts", q_h, k_all_h) / math.sqrt(float(d_head))
    n_prev = (int(kv_keys_prev.shape[0])
              if kv_keys_prev.size else 0)
    mask = _np.full(
        (n_tokens, n_prev + n_tokens), -1e9, dtype=_np.float64)
    for i in range(n_tokens):
        mask[i, : n_prev + i + 1] = 0.0
    scores = scores + mask[None, :, :]
    attn = _np.exp(scores - _np.max(scores, axis=-1,
                                       keepdims=True))
    attn = attn / _np.sum(attn, axis=-1, keepdims=True)
    out_h = _np.einsum("hts,hsd->htd", attn, v_all_h)
    out = _merge_heads(out_h) @ layer.attn.w_o
    k_all = _merge_heads(k_all_h)
    v_all = _merge_heads(v_all_h)
    return out, k_all, v_all


def forward_with_hidden_state_injection(
        params: TinyV2SubstrateParams,
        token_ids: Sequence[int],
        *,
        injection_target_layer: int,
        injection_perturbation: "_np.ndarray | None",
        kv_cache: TinyV2KVCache | None = None,
) -> dict[str, Any]:
    """Forward with hidden-state injection at one layer boundary.

    Injection point: after layer ``target_layer - 1`` (or
    embedding if ``target_layer == 0``), before layer
    ``target_layer``'s attention. The perturbation is added to
    the residual stream over the new tokens.

    Returns a dict with:
      * ``logits`` — final logits
      * ``last_position_logits`` — final logits at last new token
      * ``hidden_states`` — per-layer hidden states post-injection
      * ``kv_cache`` — post-forward cache
    """
    cfg = params.config
    n_tokens = len(token_ids)
    ids = _np.asarray(token_ids, dtype=_np.int64)
    n_prev = 0
    if kv_cache is not None and kv_cache.n_tokens() > 0:
        n_prev = int(kv_cache.n_tokens())
    pos = _np.arange(n_prev, n_prev + n_tokens, dtype=_np.int64)
    x = params.embed[ids] + params.pos_embed[pos]
    new_cache = (
        kv_cache.clone() if kv_cache is not None
        else TinyV2KVCache.empty(cfg.n_layers))
    if len(new_cache.keys) != cfg.n_layers:
        new_cache = TinyV2KVCache.empty(cfg.n_layers)
    hidden_states: list["_np.ndarray"] = [x.copy()]
    inj_done = False
    if (int(injection_target_layer) == 0
            and injection_perturbation is not None):
        x = x + _np.asarray(injection_perturbation,
                             dtype=_np.float64)
        inj_done = True
    for layer_idx, layer in enumerate(params.layers):
        if (int(injection_target_layer) == layer_idx
                and not inj_done
                and injection_perturbation is not None):
            x = x + _np.asarray(injection_perturbation,
                                 dtype=_np.float64)
            inj_done = True
        x_norm = _layer_norm(x, layer.ln1)
        kv_k = new_cache.keys[layer_idx]
        kv_v = new_cache.values[layer_idx]
        if kv_k.size == 0:
            kv_k = _np.zeros((0, cfg.d_model),
                              dtype=_np.float64)
            kv_v = _np.zeros((0, cfg.d_model),
                              dtype=_np.float64)
        attn_out, k_all, v_all = (
            _attention_layer_forward_for_inject(
                x_norm, layer,
                kv_keys_prev=kv_k,
                kv_values_prev=kv_v,
                positions_new=pos,
                rope_table=params.rope_table,
                use_rope=bool(cfg.use_rope)))
        new_cache.keys[layer_idx] = k_all
        new_cache.values[layer_idx] = v_all
        x = x + attn_out
        x_norm2 = _layer_norm(x, layer.ln2)
        ff_h = _gelu(x_norm2 @ layer.ff.w_1 + layer.ff.b_1)
        ff_out = ff_h @ layer.ff.w_2 + layer.ff.b_2
        x = x + ff_out
        hidden_states.append(x.copy())
    final = _layer_norm(x, params.ln_f)
    logits = final @ params.unembed
    return {
        "logits": logits,
        "last_position_logits": logits[-1],
        "hidden_states": hidden_states,
        "kv_cache": new_cache,
        "injection_done": bool(inj_done),
    }


@dataclasses.dataclass(frozen=True)
class HiddenStateBridgeWitness:
    schema: str
    projection_cid: str
    target_layer: int
    carrier_cid: str
    baseline_logits_cid: str
    injected_logits_cid: str
    baseline_last_argmax: int
    injected_last_argmax: int
    max_abs_logit_perturbation: float
    last_logit_l2_perturbation: float
    cross_entropy_delta: float
    pre_residual_l2: float
    post_residual_l2: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "projection_cid": str(self.projection_cid),
            "target_layer": int(self.target_layer),
            "carrier_cid": str(self.carrier_cid),
            "baseline_logits_cid": str(self.baseline_logits_cid),
            "injected_logits_cid": str(self.injected_logits_cid),
            "baseline_last_argmax": int(self.baseline_last_argmax),
            "injected_last_argmax": int(self.injected_last_argmax),
            "max_abs_logit_perturbation": float(round(
                self.max_abs_logit_perturbation, 12)),
            "last_logit_l2_perturbation": float(round(
                self.last_logit_l2_perturbation, 12)),
            "cross_entropy_delta": float(round(
                self.cross_entropy_delta, 12)),
            "pre_residual_l2": float(round(
                self.pre_residual_l2, 12)),
            "post_residual_l2": float(round(
                self.post_residual_l2, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hidden_state_bridge_witness",
            "witness": self.to_dict()})


def _softmax_logits(x: "_np.ndarray") -> "_np.ndarray":
    m = float(_np.max(x))
    z = _np.exp(x - m)
    s = float(_np.sum(z))
    return z / max(s, 1e-30)


def bridge_hidden_state_and_measure(
        *,
        params: TinyV2SubstrateParams,
        carrier: Sequence[float],
        projection: HiddenStateBridgeProjection,
        target_layer: int,
        token_ids: Sequence[int],
) -> HiddenStateBridgeWitness:
    """End-to-end hidden-state injection measurement."""
    perturb = projection.project(carrier, int(target_layer))
    n_new = len(token_ids)
    if perturb.shape[0] >= n_new:
        perturb = perturb[:n_new]
    else:
        pad = _np.zeros(
            (n_new - perturb.shape[0], perturb.shape[1]),
            dtype=_np.float64)
        perturb = _np.concatenate([perturb, pad], axis=0)
    base = forward_with_hidden_state_injection(
        params, list(token_ids),
        injection_target_layer=int(target_layer),
        injection_perturbation=None)
    inj = forward_with_hidden_state_injection(
        params, list(token_ids),
        injection_target_layer=int(target_layer),
        injection_perturbation=perturb)
    b = base["last_position_logits"]
    i = inj["last_position_logits"]
    diff = i - b
    max_abs = float(_np.max(_np.abs(diff)))
    l2 = float(_np.linalg.norm(diff))
    base_p = _softmax_logits(b)
    inj_p = _softmax_logits(i)
    eps = 1e-30
    ce = float(_np.sum(
        base_p * (_np.log(base_p + eps)
                   - _np.log(inj_p + eps))))
    pre_l2 = float(_np.linalg.norm(
        base["hidden_states"][int(target_layer)]))
    post_l2 = float(_np.linalg.norm(
        inj["hidden_states"][int(target_layer)]))
    carrier_cid = _ndarray_cid(
        _np.asarray(carrier, dtype=_np.float64).reshape(-1))
    return HiddenStateBridgeWitness(
        schema=W57_HIDDEN_STATE_BRIDGE_SCHEMA_VERSION,
        projection_cid=str(projection.cid()),
        target_layer=int(target_layer),
        carrier_cid=str(carrier_cid),
        baseline_logits_cid=_ndarray_cid(b),
        injected_logits_cid=_ndarray_cid(i),
        baseline_last_argmax=int(_np.argmax(b)),
        injected_last_argmax=int(_np.argmax(i)),
        max_abs_logit_perturbation=float(max_abs),
        last_logit_l2_perturbation=float(l2),
        cross_entropy_delta=float(ce),
        pre_residual_l2=float(pre_l2),
        post_residual_l2=float(post_l2),
    )


__all__ = [
    "W57_HIDDEN_STATE_BRIDGE_SCHEMA_VERSION",
    "W57_DEFAULT_HSB_PROJECTION_SEED",
    "W57_DEFAULT_HSB_INJECT_SCALE",
    "HiddenStateBridgeProjection",
    "HiddenStateBridgeWitness",
    "forward_with_hidden_state_injection",
    "bridge_hidden_state_and_measure",
]
