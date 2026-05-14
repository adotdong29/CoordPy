"""W58 M3 — Hidden-State Bridge V2.

Strictly extends W57's ``coordpy.hidden_state_bridge``. V2 lifts
hidden-state injection from "one layer at a time" to
**multi-layer simultaneous injection** with a fitted projection
that targets a specific logit shift.

Mechanism:

1. Pick a set of target layers ``L = {l_1, ..., l_k}``.
2. Project the carrier through *per-layer* projections to
   ``(n_tokens, d_model)`` perturbations, one per target layer.
3. Run a V3 forward where every layer in ``L`` receives its
   perturbation added to the residual stream at the input of that
   layer.
4. Measure logit perturbation + per-layer L2 + KL.
5. If a target L2 logit shift is provided, fit a global injection
   scale on the multi-layer projection via coordinate descent.

V2 strictly extends V1: a single-layer V2 reduces to V1 exactly
when ``target_layers=(l,)`` and no fit is used.

Honest scope
------------

* The bridge is *behavioural at the substrate V3*: logits change
  as a function of the carrier. The V2 bridge does NOT claim
  helpfulness without training; ``W58-L-V3-NO-BACKPROP-CAP``
  carries forward.
* Multi-layer perturbations DO interact non-linearly with the
  substrate; we measure the joint effect, not the sum-of-singles.
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
        "coordpy.hidden_state_bridge_v2 requires numpy") from exc

from .tiny_substrate_v3 import (
    TinyV3KVCache,
    TinyV3SubstrateParams,
    _ndarray_cid,
    _sha256_hex,
    _norm_v3,
    _swish,
    _gelu,
    _apply_rope_v3,
    _split_heads_q,
    _split_heads_kv,
    _merge_heads,
    _gqa_broadcast_k_v,
    forward_tiny_substrate_v3,
)


W58_HIDDEN_STATE_BRIDGE_V2_SCHEMA_VERSION: str = (
    "coordpy.hidden_state_bridge_v2.v1")
W58_DEFAULT_HSB2_PROJECTION_SEED: int = 58043043
W58_DEFAULT_HSB2_INJECT_SCALE: float = 0.20


@dataclasses.dataclass
class HiddenStateBridgeV2Projection:
    """Per-target-layer carrier → (n_tokens, d_model)
    perturbation projections.

    ``proj``: shape ``(n_target_layers, n_tokens, carrier_dim,
                       d_model)``.
    """

    n_target_layers: int
    n_tokens: int
    carrier_dim: int
    d_model: int
    target_layers: tuple[int, ...]
    proj: "_np.ndarray"
    inject_scale: float
    seed: int

    @classmethod
    def init(
            cls, *,
            target_layers: Sequence[int],
            n_tokens: int,
            carrier_dim: int,
            d_model: int,
            seed: int = W58_DEFAULT_HSB2_PROJECTION_SEED,
            inject_scale: float = W58_DEFAULT_HSB2_INJECT_SCALE,
    ) -> "HiddenStateBridgeV2Projection":
        tl = tuple(int(l) for l in target_layers)
        rng = _np.random.default_rng(int(seed))
        s = float(inject_scale)
        shape = (len(tl), int(n_tokens), int(carrier_dim),
                 int(d_model))
        proj = rng.standard_normal(shape) * s
        return cls(
            n_target_layers=int(len(tl)),
            n_tokens=int(n_tokens),
            carrier_dim=int(carrier_dim),
            d_model=int(d_model),
            target_layers=tl,
            proj=proj,
            inject_scale=float(inject_scale),
            seed=int(seed),
        )

    def project_all_layers(
            self, carrier: Sequence[float],
    ) -> dict[int, "_np.ndarray"]:
        """Return a ``{layer_idx -> (n_tokens, d_model)}`` map."""
        x = _np.asarray(carrier, dtype=_np.float64).reshape(-1)
        if x.size < self.carrier_dim:
            x = _np.concatenate(
                [x, _np.zeros(self.carrier_dim - x.size,
                              dtype=_np.float64)])
        elif x.size > self.carrier_dim:
            x = x[: self.carrier_dim]
        out: dict[int, "_np.ndarray"] = {}
        for i, l in enumerate(self.target_layers):
            layer_proj = self.proj[i]
            out[int(l)] = _np.einsum("tcd,c->td", layer_proj, x)
        return out

    def with_inject_scale(
            self, inject_scale: float,
    ) -> "HiddenStateBridgeV2Projection":
        new_proj = (self.proj
                     / max(float(self.inject_scale), 1e-9)
                     * float(inject_scale))
        return dataclasses.replace(
            self, inject_scale=float(inject_scale),
            proj=new_proj)

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W58_HIDDEN_STATE_BRIDGE_V2_SCHEMA_VERSION,
            "kind": "hidden_state_bridge_v2_projection",
            "n_target_layers": int(self.n_target_layers),
            "n_tokens": int(self.n_tokens),
            "carrier_dim": int(self.carrier_dim),
            "d_model": int(self.d_model),
            "target_layers": list(self.target_layers),
            "seed": int(self.seed),
            "inject_scale": float(self.inject_scale),
            "proj_cid": _ndarray_cid(self.proj),
        })


def _attention_layer_forward_v3_for_inject(
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
    n_kv_heads = int(layer.attn.n_kv_heads)
    d_head = d_model // n_heads
    d_kv = n_kv_heads * d_head
    q = x @ layer.attn.w_q
    k_new = x @ layer.attn.w_k
    v_new = x @ layer.attn.w_v
    q_h = _split_heads_q(q, n_heads)
    k_new_h = _split_heads_kv(k_new, n_kv_heads)
    v_new_h = _split_heads_kv(v_new, n_kv_heads)
    if use_rope:
        q_h = _apply_rope_v3(q_h, positions_new, rope_table)
        k_new_h = _apply_rope_v3(
            k_new_h, positions_new, rope_table)
    if kv_keys_prev.size == 0:
        k_all_h = k_new_h
        v_all_h = v_new_h
    else:
        prev_k_h = _split_heads_kv(kv_keys_prev, n_kv_heads)
        prev_v_h = _split_heads_kv(kv_values_prev, n_kv_heads)
        k_all_h = _np.concatenate([prev_k_h, k_new_h], axis=1)
        v_all_h = _np.concatenate([prev_v_h, v_new_h], axis=1)
    k_all_h_bcast = _gqa_broadcast_k_v(
        k_all_h, n_heads, n_kv_heads)
    v_all_h_bcast = _gqa_broadcast_k_v(
        v_all_h, n_heads, n_kv_heads)
    scores = _np.einsum(
        "htd,hsd->hts",
        q_h, k_all_h_bcast) / math.sqrt(float(d_head))
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
    out_h = _np.einsum(
        "hts,hsd->htd", attn, v_all_h_bcast)
    out = _merge_heads(out_h) @ layer.attn.w_o
    k_all = _merge_heads(k_all_h)
    v_all = _merge_heads(v_all_h)
    return out, k_all, v_all


def forward_with_hidden_state_injection_v2(
        params: TinyV3SubstrateParams,
        token_ids: Sequence[int],
        *,
        injections: dict[int, "_np.ndarray"],
        kv_cache: TinyV3KVCache | None = None,
) -> dict[str, Any]:
    """V3-substrate forward with multi-layer hidden-state injection.

    ``injections`` is ``{layer_idx -> (n_tokens, d_model)}`` map.
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
        else TinyV3KVCache.empty(cfg.n_layers))
    if len(new_cache.keys) != cfg.n_layers:
        new_cache = TinyV3KVCache.empty(cfg.n_layers)
    hidden_states: list["_np.ndarray"] = [x.copy()]
    injected_layers: list[int] = []
    d_kv = (int(cfg.d_model) // int(cfg.n_heads)
            * int(cfg.n_kv_heads))
    # Inject at embedding boundary if layer 0 is targeted.
    if 0 in injections:
        perturb = _np.asarray(
            injections[0], dtype=_np.float64)
        # Right-trim or pad.
        if perturb.shape[0] >= n_tokens:
            perturb = perturb[: n_tokens]
        else:
            pad = _np.zeros(
                (n_tokens - perturb.shape[0],
                 perturb.shape[1]), dtype=_np.float64)
            perturb = _np.concatenate([perturb, pad], axis=0)
        x = x + perturb
        injected_layers.append(0)
    for layer_idx, layer in enumerate(params.layers):
        if (layer_idx > 0
                and int(layer_idx) in injections):
            perturb = _np.asarray(
                injections[int(layer_idx)], dtype=_np.float64)
            if perturb.shape[0] >= n_tokens:
                perturb = perturb[: n_tokens]
            else:
                pad = _np.zeros(
                    (n_tokens - perturb.shape[0],
                     perturb.shape[1]), dtype=_np.float64)
                perturb = _np.concatenate([perturb, pad], axis=0)
            x = x + perturb
            injected_layers.append(int(layer_idx))
        x_norm = _norm_v3(x, layer.ln1)
        kv_k = new_cache.keys[layer_idx]
        kv_v = new_cache.values[layer_idx]
        if kv_k.size == 0:
            kv_k = _np.zeros((0, d_kv), dtype=_np.float64)
            kv_v = _np.zeros((0, d_kv), dtype=_np.float64)
        attn_out, k_all, v_all = (
            _attention_layer_forward_v3_for_inject(
                x_norm, layer,
                kv_keys_prev=kv_k,
                kv_values_prev=kv_v,
                positions_new=pos,
                rope_table=params.rope_table,
                use_rope=bool(cfg.use_rope)))
        new_cache.keys[layer_idx] = k_all
        new_cache.values[layer_idx] = v_all
        x = x + attn_out
        x_norm2 = _norm_v3(x, layer.ln2)
        if bool(layer.ff.use_swiglu):
            gate = _swish(x_norm2 @ layer.ff.w_gate)
            up = x_norm2 @ layer.ff.w_1 + layer.ff.b_1
            ff_h = gate * up
            ff_out = ff_h @ layer.ff.w_2 + layer.ff.b_2
        else:
            ff_h = _gelu(
                x_norm2 @ layer.ff.w_1 + layer.ff.b_1)
            ff_out = ff_h @ layer.ff.w_2 + layer.ff.b_2
        x = x + ff_out
        hidden_states.append(x.copy())
    final = _norm_v3(x, params.ln_f)
    logits = final @ params.unembed
    return {
        "logits": logits,
        "last_position_logits": logits[-1],
        "hidden_states": hidden_states,
        "kv_cache": new_cache,
        "injected_layers": list(injected_layers),
    }


@dataclasses.dataclass(frozen=True)
class HiddenStateBridgeV2Witness:
    schema: str
    projection_cid: str
    target_layers: tuple[int, ...]
    carrier_cid: str
    baseline_logits_cid: str
    injected_logits_cid: str
    baseline_last_argmax: int
    injected_last_argmax: int
    max_abs_logit_perturbation: float
    last_logit_l2_perturbation: float
    cross_entropy_delta: float
    per_layer_residual_delta_l2: tuple[float, ...]
    fitted_scale_used: bool
    target_l2_perturbation: float
    fit_residual: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "projection_cid": str(self.projection_cid),
            "target_layers": list(self.target_layers),
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
            "per_layer_residual_delta_l2": [
                float(round(v, 12))
                for v in self.per_layer_residual_delta_l2],
            "fitted_scale_used": bool(self.fitted_scale_used),
            "target_l2_perturbation": float(round(
                self.target_l2_perturbation, 12)),
            "fit_residual": float(round(self.fit_residual, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hidden_state_bridge_v2_witness",
            "witness": self.to_dict()})


def _softmax_logits(x: "_np.ndarray") -> "_np.ndarray":
    m = float(_np.max(x))
    z = _np.exp(x - m)
    s = float(_np.sum(z))
    return z / max(s, 1e-30)


def _measure_hsb_v2_l2(
        *,
        params: TinyV3SubstrateParams,
        carrier: Sequence[float],
        projection: HiddenStateBridgeV2Projection,
        token_ids: Sequence[int],
) -> float:
    injections = projection.project_all_layers(carrier)
    base = forward_with_hidden_state_injection_v2(
        params, list(token_ids), injections={})
    inj = forward_with_hidden_state_injection_v2(
        params, list(token_ids), injections=injections)
    return float(_np.linalg.norm(
        inj["last_position_logits"]
        - base["last_position_logits"]))


def fit_hsb_v2_inject_scale(
        *,
        params: TinyV3SubstrateParams,
        carrier: Sequence[float],
        projection: HiddenStateBridgeV2Projection,
        token_ids: Sequence[int],
        target_l2_perturbation: float,
        n_steps: int = 12,
        learning_rate: float = 0.25,
) -> tuple[HiddenStateBridgeV2Projection, list[float]]:
    residuals: list[float] = []
    cur = projection
    target = max(float(target_l2_perturbation), 1e-9)
    for step in range(int(n_steps)):
        l2 = _measure_hsb_v2_l2(
            params=params, carrier=carrier,
            projection=cur, token_ids=token_ids)
        residuals.append(float(l2 - target))
        if abs(l2 - target) < 1e-3 * target:
            break
        ratio = float(l2 / target)
        factor = 1.0 / max(ratio, 1e-6)
        damped = 1.0 + float(learning_rate) * (factor - 1.0)
        damped = max(0.05, min(damped, 5.0))
        cur = cur.with_inject_scale(
            float(cur.inject_scale) * damped)
    return cur, residuals


def bridge_hidden_state_and_measure_v2(
        *,
        params: TinyV3SubstrateParams,
        carrier: Sequence[float],
        projection: HiddenStateBridgeV2Projection,
        token_ids: Sequence[int],
        target_l2_perturbation: float | None = None,
) -> HiddenStateBridgeV2Witness:
    """End-to-end multi-layer hidden-state injection on V3 substrate."""
    fit_used = False
    fit_residual = 0.0
    cur_proj = projection
    if target_l2_perturbation is not None:
        cur_proj, residuals = fit_hsb_v2_inject_scale(
            params=params, carrier=carrier,
            projection=projection,
            token_ids=token_ids,
            target_l2_perturbation=float(target_l2_perturbation))
        fit_used = True
        fit_residual = float(residuals[-1]) if residuals else 0.0
    injections = cur_proj.project_all_layers(carrier)
    base = forward_with_hidden_state_injection_v2(
        params, list(token_ids), injections={})
    inj = forward_with_hidden_state_injection_v2(
        params, list(token_ids), injections=injections)
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
    per_layer_deltas: list[float] = []
    for hb, hi in zip(
            base["hidden_states"], inj["hidden_states"]):
        per_layer_deltas.append(
            float(_np.linalg.norm(hi - hb)))
    carrier_cid = _ndarray_cid(
        _np.asarray(carrier, dtype=_np.float64).reshape(-1))
    return HiddenStateBridgeV2Witness(
        schema=W58_HIDDEN_STATE_BRIDGE_V2_SCHEMA_VERSION,
        projection_cid=str(cur_proj.cid()),
        target_layers=cur_proj.target_layers,
        carrier_cid=str(carrier_cid),
        baseline_logits_cid=_ndarray_cid(b),
        injected_logits_cid=_ndarray_cid(i),
        baseline_last_argmax=int(_np.argmax(b)),
        injected_last_argmax=int(_np.argmax(i)),
        max_abs_logit_perturbation=float(max_abs),
        last_logit_l2_perturbation=float(l2),
        cross_entropy_delta=float(ce),
        per_layer_residual_delta_l2=tuple(per_layer_deltas),
        fitted_scale_used=bool(fit_used),
        target_l2_perturbation=float(
            target_l2_perturbation
            if target_l2_perturbation is not None else 0.0),
        fit_residual=float(fit_residual),
    )


__all__ = [
    "W58_HIDDEN_STATE_BRIDGE_V2_SCHEMA_VERSION",
    "W58_DEFAULT_HSB2_PROJECTION_SEED",
    "W58_DEFAULT_HSB2_INJECT_SCALE",
    "HiddenStateBridgeV2Projection",
    "HiddenStateBridgeV2Witness",
    "forward_with_hidden_state_injection_v2",
    "fit_hsb_v2_inject_scale",
    "bridge_hidden_state_and_measure_v2",
]
