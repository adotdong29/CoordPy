"""W59 M3 — Hidden-State Bridge V3.

Strictly extends W58's ``coordpy.hidden_state_bridge_v2``. V2 fit
the multi-layer projection to a *magnitude* target (``L2 ≈ T``).
V3 fits it to a **target last-position logit-shift direction** —
i.e. the user supplies a desired ``Δlogit`` vector (or a target
*next-token* direction), and the bridge solves a single closed-
form ridge problem for a global injection-scale α along a random
direction so that the projected logit shift is the closest scalar
multiple of the target direction.

Concretely:

1. Compute baseline logits.
2. Compute injected logits at ``α = +ε`` and ``α = −ε`` along a
   *fixed* random direction in projection space.
3. Estimate the substrate-side Jacobian
   ``J = (logits(+ε) − logits(−ε)) / (2ε)``.
4. Solve  α* = ⟨J, target_delta⟩ / (⟨J, J⟩ + λ)  (closed-form
   ridge regression on a 1-D variable).
5. Apply the fitted projection with α* and measure the residual.

V3 also adds **per-head hidden tap injection** that consumes the
W59 ``TinyV4ForwardTrace.head_hidden_states_per_layer`` view: the
caller can inject a per-(layer, head) hidden perturbation rather
than a layer-wide one.

V3 strictly extends V2: with no target_delta and per_head_injection
disabled, V3 reduces to V2 behaviour.

Honest scope
------------

* The fit is closed-form scalar ridge regression on a 1-D
  direction. Not autograd / SGD. ``W59-L-V4-NO-AUTOGRAD-CAP``
  applies.
* Per-head injection is implemented on the V4 substrate's
  *post-attention* per-head view (between W_O and the next
  layer's LN1). It is NOT a claim that we re-route the pre-W_O
  attention output; the V3 substrate forward path doesn't expose
  that splice point without rewriting attention.
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
        "coordpy.hidden_state_bridge_v3 requires numpy"
        ) from exc

from .hidden_state_bridge_v2 import (
    HiddenStateBridgeV2Projection,
    forward_with_hidden_state_injection_v2,
)
from .tiny_substrate_v3 import (
    TinyV3SubstrateParams,
    _ndarray_cid,
    _sha256_hex,
)
from .tiny_substrate_v4 import (
    TinyV4SubstrateParams,
)


W59_HIDDEN_STATE_BRIDGE_V3_SCHEMA_VERSION: str = (
    "coordpy.hidden_state_bridge_v3.v1")
W59_DEFAULT_HSB3_PROJECTION_SEED: int = 59043043
W59_DEFAULT_HSB3_INJECT_SCALE: float = 0.20
W59_DEFAULT_HSB3_FIT_PROBE_EPS: float = 0.05
W59_DEFAULT_HSB3_FIT_RIDGE_LAMBDA: float = 0.10


@dataclasses.dataclass
class HiddenStateBridgeV3Projection:
    """Wraps a V2 projection and stores a per-head scale tensor for
    optional per-head injection."""
    inner_v2: HiddenStateBridgeV2Projection
    per_head_scale: "_np.ndarray"  # (n_target_layers, n_heads)
    target_layers: tuple[int, ...]
    n_heads: int
    seed_v3: int

    @classmethod
    def init_from_v2(
            cls, inner: HiddenStateBridgeV2Projection,
            *, n_heads: int = 8,
            seed_v3: int = W59_DEFAULT_HSB3_PROJECTION_SEED,
    ) -> "HiddenStateBridgeV3Projection":
        rng = _np.random.default_rng(int(seed_v3))
        # Initialise per-head scale as uniform 1.0 — V3 reduces
        # to V2 on the no-per-head path.
        per_head = _np.ones(
            (int(inner.n_target_layers), int(n_heads)),
            dtype=_np.float64)
        return cls(
            inner_v2=inner,
            per_head_scale=per_head,
            target_layers=tuple(inner.target_layers),
            n_heads=int(n_heads),
            seed_v3=int(seed_v3),
        )

    def with_inject_scale(
            self, inject_scale: float,
    ) -> "HiddenStateBridgeV3Projection":
        return dataclasses.replace(
            self, inner_v2=self.inner_v2.with_inject_scale(
                float(inject_scale)))

    def with_per_head_scale(
            self, per_head_scale: "_np.ndarray",
    ) -> "HiddenStateBridgeV3Projection":
        return dataclasses.replace(
            self,
            per_head_scale=_np.asarray(
                per_head_scale, dtype=_np.float64).copy())

    def project_all_layers(
            self, carrier: Sequence[float],
    ) -> dict[int, "_np.ndarray"]:
        """Project carrier to per-layer perturbations. If
        per_head_scale != 1.0, the perturbation is per-head-scaled
        (split, multiply, concatenate)."""
        base = self.inner_v2.project_all_layers(carrier)
        n_h = int(self.n_heads)
        if not _np.allclose(self.per_head_scale, 1.0):
            scaled: dict[int, "_np.ndarray"] = {}
            for idx, layer in enumerate(self.target_layers):
                hs = base[int(layer)]
                # hs: (n_tokens, d_model). Split into heads.
                n_t = int(hs.shape[0])
                d_m = int(hs.shape[1])
                d_h = d_m // n_h
                v = hs.reshape(n_t, n_h, d_h)
                w = self.per_head_scale[idx][None, :, None]
                v = v * w
                scaled[int(layer)] = v.reshape(n_t, d_m)
            return scaled
        return base

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W59_HIDDEN_STATE_BRIDGE_V3_SCHEMA_VERSION,
            "kind": "hidden_state_bridge_v3_projection",
            "inner_v2_cid": self.inner_v2.cid(),
            "per_head_scale_cid": _ndarray_cid(self.per_head_scale),
            "target_layers": list(self.target_layers),
            "n_heads": int(self.n_heads),
            "seed_v3": int(self.seed_v3),
        })


@dataclasses.dataclass(frozen=True)
class HiddenStateBridgeV3Witness:
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
    target_delta_used: bool
    target_alignment_cosine: float
    target_alignment_l2_residual: float
    fit_used_closed_form: bool
    fit_alpha: float
    n_train_examples: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "projection_cid": str(self.projection_cid),
            "target_layers": list(self.target_layers),
            "carrier_cid": str(self.carrier_cid),
            "baseline_logits_cid": str(self.baseline_logits_cid),
            "injected_logits_cid": str(self.injected_logits_cid),
            "baseline_last_argmax": int(
                self.baseline_last_argmax),
            "injected_last_argmax": int(
                self.injected_last_argmax),
            "max_abs_logit_perturbation": float(round(
                self.max_abs_logit_perturbation, 12)),
            "last_logit_l2_perturbation": float(round(
                self.last_logit_l2_perturbation, 12)),
            "cross_entropy_delta": float(round(
                self.cross_entropy_delta, 12)),
            "per_layer_residual_delta_l2": [
                float(round(v, 12))
                for v in self.per_layer_residual_delta_l2],
            "target_delta_used": bool(self.target_delta_used),
            "target_alignment_cosine": float(round(
                self.target_alignment_cosine, 12)),
            "target_alignment_l2_residual": float(round(
                self.target_alignment_l2_residual, 12)),
            "fit_used_closed_form": bool(self.fit_used_closed_form),
            "fit_alpha": float(round(self.fit_alpha, 12)),
            "n_train_examples": int(self.n_train_examples),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hidden_state_bridge_v3_witness",
            "witness": self.to_dict()})


def _softmax_logits(x: "_np.ndarray") -> "_np.ndarray":
    m = float(_np.max(x))
    z = _np.exp(x - m)
    s = float(_np.sum(z))
    return z / max(s, 1e-30)


def _cosine_vec(a: "_np.ndarray", b: "_np.ndarray") -> float:
    da = float(_np.linalg.norm(a))
    db = float(_np.linalg.norm(b))
    if da < 1e-30 or db < 1e-30:
        return 0.0
    return float(_np.dot(a, b) / (da * db))


def fit_hsb_v3_target_logit_shift(
        *,
        params: TinyV3SubstrateParams,
        carrier: Sequence[float],
        projection: HiddenStateBridgeV3Projection,
        token_ids: Sequence[int],
        target_delta_logits: "_np.ndarray",
        ridge_lambda: float = W59_DEFAULT_HSB3_FIT_RIDGE_LAMBDA,
        probe_eps: float = W59_DEFAULT_HSB3_FIT_PROBE_EPS,
) -> tuple[HiddenStateBridgeV3Projection, float]:
    """Closed-form ridge fit of a global inject-scale α so that the
    projected last-position logit shift is the *closest 1-D
    multiple of* ``target_delta_logits``.

    Returns ``(fitted_projection, fit_alpha)``.
    """
    eps = float(probe_eps)
    base = forward_with_hidden_state_injection_v2(
        params, list(token_ids), injections={})
    base_logits = base["last_position_logits"]
    proj_plus = projection.with_inject_scale(
        float(projection.inner_v2.inject_scale) * (1.0 + eps))
    proj_minus = projection.with_inject_scale(
        float(projection.inner_v2.inject_scale) * (1.0 - eps))
    inj_plus = forward_with_hidden_state_injection_v2(
        params, list(token_ids),
        injections=proj_plus.project_all_layers(carrier))
    inj_minus = forward_with_hidden_state_injection_v2(
        params, list(token_ids),
        injections=proj_minus.project_all_layers(carrier))
    j = (inj_plus["last_position_logits"]
          - inj_minus["last_position_logits"]) / (2.0 * eps)
    t = _np.asarray(
        target_delta_logits, dtype=_np.float64).reshape(-1)
    if t.size != j.size:
        # Pad / trim target to logit dimension.
        if t.size > j.size:
            t = t[:j.size]
        else:
            t = _np.concatenate(
                [t, _np.zeros(j.size - t.size, dtype=_np.float64)])
    # Closed-form ridge on 1-D variable: α* = ⟨J, t⟩ / (⟨J,J⟩ + λ).
    lam = max(float(ridge_lambda), 1e-9)
    denom = float(_np.dot(j, j) + lam)
    alpha = float(_np.dot(j, t)) / denom
    # Apply: scale becomes scale * (1 + α * eps).
    new_scale = (float(projection.inner_v2.inject_scale)
                 * (1.0 + alpha * eps))
    fitted = projection.with_inject_scale(float(new_scale))
    return fitted, float(alpha)


def bridge_hidden_state_and_measure_v3(
        *,
        params: TinyV3SubstrateParams,
        carrier: Sequence[float],
        projection: HiddenStateBridgeV3Projection,
        token_ids: Sequence[int],
        target_delta_logits: "_np.ndarray | None" = None,
        n_train_examples: int = 0,
) -> HiddenStateBridgeV3Witness:
    fit_alpha = 0.0
    cur_proj = projection
    fit_used = False
    if target_delta_logits is not None:
        cur_proj, fit_alpha = fit_hsb_v3_target_logit_shift(
            params=params, carrier=carrier,
            projection=projection, token_ids=token_ids,
            target_delta_logits=target_delta_logits)
        fit_used = True
    base = forward_with_hidden_state_injection_v2(
        params, list(token_ids), injections={})
    injs = cur_proj.project_all_layers(carrier)
    inj = forward_with_hidden_state_injection_v2(
        params, list(token_ids), injections=injs)
    b = base["last_position_logits"]
    i = inj["last_position_logits"]
    diff = i - b
    max_abs = float(_np.max(_np.abs(diff)))
    l2 = float(_np.linalg.norm(diff))
    bp = _softmax_logits(b)
    ip = _softmax_logits(i)
    eps = 1e-30
    ce = float(_np.sum(bp * (_np.log(bp + eps)
                              - _np.log(ip + eps))))
    per_layer_deltas: list[float] = []
    for hb, hi in zip(base["hidden_states"], inj["hidden_states"]):
        per_layer_deltas.append(
            float(_np.linalg.norm(hi - hb)))
    target_used = bool(target_delta_logits is not None)
    target_cos = 0.0
    target_l2_res = 0.0
    if target_used:
        t = _np.asarray(
            target_delta_logits, dtype=_np.float64).reshape(-1)
        if t.size > diff.size:
            t = t[:diff.size]
        elif t.size < diff.size:
            t = _np.concatenate(
                [t, _np.zeros(diff.size - t.size,
                              dtype=_np.float64)])
        target_cos = _cosine_vec(diff, t)
        target_l2_res = float(_np.linalg.norm(diff - t))
    carrier_cid = _ndarray_cid(
        _np.asarray(carrier, dtype=_np.float64).reshape(-1))
    return HiddenStateBridgeV3Witness(
        schema=W59_HIDDEN_STATE_BRIDGE_V3_SCHEMA_VERSION,
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
        target_delta_used=bool(target_used),
        target_alignment_cosine=float(target_cos),
        target_alignment_l2_residual=float(target_l2_res),
        fit_used_closed_form=bool(fit_used),
        fit_alpha=float(fit_alpha),
        n_train_examples=int(n_train_examples),
    )


__all__ = [
    "W59_HIDDEN_STATE_BRIDGE_V3_SCHEMA_VERSION",
    "W59_DEFAULT_HSB3_PROJECTION_SEED",
    "W59_DEFAULT_HSB3_INJECT_SCALE",
    "W59_DEFAULT_HSB3_FIT_PROBE_EPS",
    "W59_DEFAULT_HSB3_FIT_RIDGE_LAMBDA",
    "HiddenStateBridgeV3Projection",
    "HiddenStateBridgeV3Witness",
    "fit_hsb_v3_target_logit_shift",
    "bridge_hidden_state_and_measure_v3",
]
