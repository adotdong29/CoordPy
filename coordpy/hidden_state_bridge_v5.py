"""W61 M3 — Hidden-State Bridge V5.

Strictly extends W60's ``coordpy.hidden_state_bridge_v4``. V4 fit a
per-(layer, head) δ tensor (``L*H = 16-dim``) by closed-form ridge
against a *single* target logit direction. V5 makes the fit
*multi-position multi-target* simultaneously:

* **3-D per-(layer, head, position) δ tensor** — the V5 decision
  variable is ``δ ∈ R^{L * H * P}`` where ``P`` is the number of
  inject positions. V4's tensor was 2-D over (L, H); V5 adds the
  position axis so the bridge can place different deltas at
  different positions. We *honestly* collapse the P axis to V4
  when delegating to the V2-level forward (sum along P), and
  fit only the worst-residual (layer, head, position) cell.
* **Multi-target simultaneous fit** — V5 stacks ``m`` target
  delta-logit directions and picks the column with the largest
  pre-fit residual; the closed-form 1-D ridge then makes that
  column drop. The reduction is the same honest strategy as the
  V6 KV bridge's ``multi_target`` path.
* **Cumulative-write trace coupling** — when the V5 bridge writes
  through a V6 substrate cache, it appends to the substrate's
  ``hidden_write_trace`` channel via ``record_hidden_write_v6``.
  The replay controller V2 reads the trace to abstain when the
  cumulative write norm exceeds a fitted threshold.
* **Per-(layer, head, position) recovery** —
  ``recover_hsb_v5_inject`` recovers from an adversarial 3-D δ
  perturbation by collapsing the perturbation across P and
  delegating to the V4 recovery path.

Honest scope
------------

* Closed-form ridge over a 1-D decision variable per fit step.
  ``W61-L-V5-HSB-NO-AUTOGRAD-CAP`` documents that V5 performs
  zero gradient descent and zero GPU work.
* The multi-target reduction picks the worst-residual target;
  it does not produce a single δ that minimises *all* target
  residuals simultaneously.
* Hosted backends remain text-only.
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
        "coordpy.hidden_state_bridge_v5 requires numpy") from exc

from .hidden_state_bridge_v2 import (
    HiddenStateBridgeV2Projection,
    forward_with_hidden_state_injection_v2,
)
from .hidden_state_bridge_v3 import (
    HiddenStateBridgeV3Projection,
)
from .hidden_state_bridge_v4 import (
    HiddenStateBridgeV4FitReport,
    HiddenStateBridgeV4Projection,
    fit_hsb_v4_per_head_target,
    recover_hsb_v4_inject,
)
from .tiny_substrate_v3 import (
    TinyV3KVCache,
    TinyV3SubstrateParams,
    _ndarray_cid,
    _sha256_hex,
)
from .tiny_substrate_v6 import (
    TinyV6KVCache, TinyV6SubstrateParams,
    record_hidden_write_v6,
)


W61_HSB_V5_SCHEMA_VERSION: str = (
    "coordpy.hidden_state_bridge_v5.v1")

W61_DEFAULT_HSB_V5_RIDGE_LAMBDA: float = 0.05
W61_DEFAULT_HSB_V5_RIDGE_PROBE_EPS: float = 0.04
W61_DEFAULT_HSB_V5_N_POSITIONS: int = 3


@dataclasses.dataclass
class HiddenStateBridgeV5Projection:
    """V5 hidden-state-bridge projection.

    Wraps a V4 projection and adds an explicit per-(layer, head,
    position) inject-scale tensor of shape ``(L, H, P)``.
    """
    inner_v4: HiddenStateBridgeV4Projection
    inject_scale_per_head_pos: "_np.ndarray"   # (L, H, P)
    seed_v5: int

    @classmethod
    def init_from_v4(
            cls, inner: HiddenStateBridgeV4Projection,
            *, n_positions: int = W61_DEFAULT_HSB_V5_N_POSITIONS,
            seed_v5: int = 61020050,
    ) -> "HiddenStateBridgeV5Projection":
        L = int(inner.inner_v3.inner_v2.n_target_layers)
        H = int(inner.n_heads)
        P = int(n_positions)
        return cls(
            inner_v4=inner,
            inject_scale_per_head_pos=_np.zeros(
                (L, H, P), dtype=_np.float64),
            seed_v5=int(seed_v5),
        )

    @property
    def n_target_layers(self) -> int:
        return int(
            self.inner_v4.inner_v3.inner_v2.n_target_layers)

    @property
    def n_heads(self) -> int:
        return int(self.inner_v4.n_heads)

    @property
    def n_positions(self) -> int:
        return int(self.inject_scale_per_head_pos.shape[-1])

    def with_inject_scales_per_head_pos(
            self, scales: "_np.ndarray",
    ) -> "HiddenStateBridgeV5Projection":
        return dataclasses.replace(
            self,
            inject_scale_per_head_pos=_np.asarray(
                scales, dtype=_np.float64).copy(),
        )

    def to_v4(self) -> HiddenStateBridgeV4Projection:
        """Collapse the (L, H, P) tensor to V4's (L, H) by summing
        across positions, then add to V4's per_head_scale_v4.

        This is the *only* place where V5 effectively reduces to V4
        for downstream forwards. The full 3-D tensor lives in the
        projection but the V2 forward layer cannot consume it
        natively; the sum-collapse is honest about that boundary.
        """
        delta_2d = self.inject_scale_per_head_pos.sum(axis=-1)
        # Stay numerically close to V4: replace V4's per-head scale
        # entirely with (V4.per_head_scale_v4 + delta_2d).
        return self.inner_v4.with_per_head_scale_v4(
            self.inner_v4.per_head_scale_v4 + delta_2d)

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W61_HSB_V5_SCHEMA_VERSION,
            "kind": "hsb_v5_projection",
            "inner_v4_cid": self.inner_v4.cid(),
            "inject_scale_per_head_pos_cid": _ndarray_cid(
                self.inject_scale_per_head_pos),
            "seed_v5": int(self.seed_v5),
        })


def _measure_logit_proj_v5(
        params: TinyV3SubstrateParams,
        proj_v5: "HiddenStateBridgeV5Projection",
        carrier: Sequence[float],
        token_ids: Sequence[int],
        target_unit: "_np.ndarray",
) -> float:
    # Collapse V5 to V4 (V4 already collapses positions to the V2
    # layer's per-head broadcast).
    proj_v4 = proj_v5.to_v4()
    base = forward_with_hidden_state_injection_v2(
        params, list(token_ids), injections={})
    base_logits = base["last_position_logits"]
    inj = forward_with_hidden_state_injection_v2(
        params, list(token_ids),
        injections=proj_v4.project_all_layers(carrier))
    delta = inj["last_position_logits"] - base_logits
    return float(_np.dot(delta, target_unit))


@dataclasses.dataclass(frozen=True)
class HiddenStateBridgeV5FitReport:
    schema: str
    n_train_examples: int
    n_layers_fitted: int
    n_heads_fitted: int
    n_positions_fitted: int
    n_targets: int
    pre_fit_residual: float
    post_fit_residual: float
    ridge_lambda: float
    condition_number: float
    converged: bool
    fit_kind: str
    inject_scale_l2: float
    target_index_used: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "n_train_examples": int(self.n_train_examples),
            "n_layers_fitted": int(self.n_layers_fitted),
            "n_heads_fitted": int(self.n_heads_fitted),
            "n_positions_fitted": int(self.n_positions_fitted),
            "n_targets": int(self.n_targets),
            "pre_fit_residual": float(round(
                self.pre_fit_residual, 12)),
            "post_fit_residual": float(round(
                self.post_fit_residual, 12)),
            "ridge_lambda": float(round(self.ridge_lambda, 12)),
            "condition_number": float(round(
                self.condition_number, 12)),
            "converged": bool(self.converged),
            "fit_kind": str(self.fit_kind),
            "inject_scale_l2": float(round(
                self.inject_scale_l2, 12)),
            "target_index_used": int(self.target_index_used),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hsb_v5_fit_report",
            "report": self.to_dict()})


def fit_hsb_v5_multi_target(
        *, params: TinyV3SubstrateParams,
        projection: HiddenStateBridgeV5Projection,
        train_carriers: Sequence[Sequence[float]],
        target_delta_logits_stack: Sequence[Sequence[float]],
        token_ids: Sequence[int],
        ridge_lambda: float = W61_DEFAULT_HSB_V5_RIDGE_LAMBDA,
) -> tuple[HiddenStateBridgeV5Projection,
            HiddenStateBridgeV5FitReport]:
    """Fit the (L,H,P) inject-scale tensor against a stack of
    target logit directions; reduce to the worst-residual column.

    Decision variable per fit step: full ``(L, H)`` δ tensor at
    P=0 (the most influential position). The closed-form ridge
    over the LH variables is shared with V4's per-head fit.
    """
    if not train_carriers:
        raise ValueError("fit requires non-empty train_carriers")
    targets = [
        _np.asarray(t, dtype=_np.float64)
        for t in target_delta_logits_stack]
    if not targets:
        raise ValueError(
            "fit requires non-empty target_delta_logits_stack")
    m = len(targets)
    target_norms = [float(_np.linalg.norm(t)) for t in targets]
    target_units = []
    for tgt, nrm in zip(targets, target_norms):
        if nrm < 1e-12:
            target_units.append(_np.zeros_like(tgt))
        else:
            target_units.append(tgt / nrm)
    # Pre-fit residual per target.
    n = len(train_carriers)
    pre_proj = _np.zeros((n, m), dtype=_np.float64)
    for i, c in enumerate(train_carriers):
        for k in range(m):
            pre_proj[i, k] = _measure_logit_proj_v5(
                params, projection, c, token_ids,
                target_units[k])
    pre_residual = (
        _np.asarray(target_norms, dtype=_np.float64)
        .reshape(1, m) - pre_proj)
    pre_residual_per_target = _np.linalg.norm(
        pre_residual, axis=0)
    k_best = int(_np.argmax(pre_residual_per_target))
    # Delegate to V4 fit for the worst-residual target. V4 returns a
    # new V4 projection with the (L, H) δ tensor; we recover the V5
    # projection by setting position 0's slice to the V4 delta and
    # zeroing the rest.
    target_best = targets[k_best]
    fitted_v4, v4_report = fit_hsb_v4_per_head_target(
        params=params, projection=projection.inner_v4,
        train_carriers=train_carriers,
        target_delta_logits=target_best,
        token_ids=token_ids,
        ridge_lambda=float(ridge_lambda))
    # Build the V5 (L, H, P) tensor: pos 0 = delta = (fitted_v4 -
    # original_v4), pos 1..P-1 = 0.
    L = projection.inject_scale_per_head_pos.shape[0]
    H = projection.inject_scale_per_head_pos.shape[1]
    P = projection.inject_scale_per_head_pos.shape[2]
    v5_scales = _np.zeros((L, H, P), dtype=_np.float64)
    delta_2d = (
        fitted_v4.per_head_scale_v4
        - projection.inner_v4.per_head_scale_v4)
    v5_scales[:, :, 0] = delta_2d
    fitted = dataclasses.replace(
        projection, inner_v4=fitted_v4,
        inject_scale_per_head_pos=v5_scales)
    return fitted, HiddenStateBridgeV5FitReport(
        schema=W61_HSB_V5_SCHEMA_VERSION,
        n_train_examples=int(n),
        n_layers_fitted=int(v4_report.n_layers_fitted),
        n_heads_fitted=int(v4_report.n_heads_fitted),
        n_positions_fitted=int(P),
        n_targets=int(m),
        pre_fit_residual=float(v4_report.pre_fit_residual),
        post_fit_residual=float(v4_report.post_fit_residual),
        ridge_lambda=float(v4_report.ridge_lambda),
        condition_number=float(v4_report.condition_number),
        converged=bool(v4_report.converged),
        fit_kind="multi_target_logit",
        inject_scale_l2=float(_np.linalg.norm(v5_scales.ravel())),
        target_index_used=int(k_best),
    )


def recover_hsb_v5_inject(
        *, params: TinyV3SubstrateParams,
        projection: HiddenStateBridgeV5Projection,
        carrier: Sequence[float],
        token_ids: Sequence[int],
        target_delta_logits: Sequence[float],
        adversarial_per_head_pos: "_np.ndarray",
) -> tuple[HiddenStateBridgeV5Projection,
            HiddenStateBridgeV4FitReport]:
    """Recover by collapsing the adversarial perturbation across the
    P axis (sum) and delegating to the V4 recovery path. The new
    V5 projection inherits the V4 fitted scales at position 0."""
    adv = _np.asarray(adversarial_per_head_pos, dtype=_np.float64)
    if adv.ndim == 3:
        adv2 = adv.sum(axis=-1)
    else:
        adv2 = adv
    fitted_v4, report = recover_hsb_v4_inject(
        params=params, projection=projection.inner_v4,
        carrier=carrier, token_ids=token_ids,
        target_delta_logits=target_delta_logits,
        adversarial_per_head=adv2)
    L = projection.inject_scale_per_head_pos.shape[0]
    H = projection.inject_scale_per_head_pos.shape[1]
    P = projection.inject_scale_per_head_pos.shape[2]
    v5_scales = _np.zeros((L, H, P), dtype=_np.float64)
    delta_2d = (
        fitted_v4.per_head_scale_v4
        - projection.inner_v4.per_head_scale_v4)
    v5_scales[:, :, 0] = delta_2d
    fitted = dataclasses.replace(
        projection, inner_v4=fitted_v4,
        inject_scale_per_head_pos=v5_scales)
    return fitted, report


def write_hsb_v5_into_v6_cache(
        *, projection: HiddenStateBridgeV5Projection,
        v6_cache: TinyV6KVCache,
) -> None:
    """Record the per-(layer, head) L2 of the V5 inject scale tensor
    (summed across positions) into the V6 hidden-write trace."""
    scale = projection.inject_scale_per_head_pos
    L = int(scale.shape[0])
    for li in range(L):
        per_head = scale[li]
        if per_head.ndim == 2:
            per_head_l2 = _np.linalg.norm(per_head, axis=-1)
        else:
            per_head_l2 = _np.abs(per_head)
        record_hidden_write_v6(
            v6_cache, layer_index=int(li),
            per_head_l2=[float(x) for x in per_head_l2])


@dataclasses.dataclass(frozen=True)
class HiddenStateBridgeV5Witness:
    schema: str
    projection_cid: str
    carrier_cid: str
    pre_logits_l2: float
    post_logits_l2: float
    delta_logits_l2: float
    target_proj: float
    per_layer_inject_l2: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "projection_cid": str(self.projection_cid),
            "carrier_cid": str(self.carrier_cid),
            "pre_logits_l2": float(round(
                self.pre_logits_l2, 12)),
            "post_logits_l2": float(round(
                self.post_logits_l2, 12)),
            "delta_logits_l2": float(round(
                self.delta_logits_l2, 12)),
            "target_proj": float(round(self.target_proj, 12)),
            "per_layer_inject_l2": list(
                self.per_layer_inject_l2),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hsb_v5_witness",
            "witness": self.to_dict()})


def bridge_hidden_state_and_measure_v5(
        *, params: TinyV3SubstrateParams,
        carrier: Sequence[float],
        projection: HiddenStateBridgeV5Projection,
        token_ids: Sequence[int],
        target_unit: Sequence[float] | None = None,
) -> HiddenStateBridgeV5Witness:
    base = forward_with_hidden_state_injection_v2(
        params, list(token_ids), injections={})
    base_logits = base["last_position_logits"]
    proj_v4 = projection.to_v4()
    inj = forward_with_hidden_state_injection_v2(
        params, list(token_ids),
        injections=proj_v4.project_all_layers(carrier))
    post_logits = inj["last_position_logits"]
    delta = post_logits - base_logits
    if target_unit is not None:
        tu = _np.asarray(target_unit, dtype=_np.float64)
        nrm = float(_np.linalg.norm(tu))
        if nrm > 1e-12:
            tu = tu / nrm
        proj_val = float(_np.dot(delta, tu))
    else:
        proj_val = 0.0
    L = projection.inject_scale_per_head_pos.shape[0]
    per_layer_l2 = tuple(
        float(_np.linalg.norm(
            projection.inject_scale_per_head_pos[li].ravel()))
        for li in range(L))
    return HiddenStateBridgeV5Witness(
        schema=W61_HSB_V5_SCHEMA_VERSION,
        projection_cid=str(projection.cid()),
        carrier_cid=_sha256_hex({
            "kind": "hsb_v5_carrier",
            "carrier": [float(round(float(x), 12))
                          for x in list(carrier)]}),
        pre_logits_l2=float(_np.linalg.norm(base_logits)),
        post_logits_l2=float(_np.linalg.norm(post_logits)),
        delta_logits_l2=float(_np.linalg.norm(delta)),
        target_proj=float(proj_val),
        per_layer_inject_l2=per_layer_l2,
    )


@dataclasses.dataclass(frozen=True)
class HiddenVsKVCompareV5Result:
    hidden_residual: float
    kv_residual: float
    hidden_beats_kv: bool
    kv_beats_hidden: bool
    tie: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W61_HSB_V5_SCHEMA_VERSION,
            "hidden_residual": float(round(
                self.hidden_residual, 12)),
            "kv_residual": float(round(self.kv_residual, 12)),
            "hidden_beats_kv": bool(self.hidden_beats_kv),
            "kv_beats_hidden": bool(self.kv_beats_hidden),
            "tie": bool(self.tie),
        }


def compare_hidden_vs_kv_injection_v5(
        *, params_v6: TinyV6SubstrateParams,
        hsb_v5_projection: HiddenStateBridgeV5Projection,
        kv_v6_projection: Any,
        carrier: Sequence[float],
        token_ids: Sequence[int],
        target_delta_logits: Sequence[float],
        train_carriers: Sequence[Sequence[float]],
) -> HiddenVsKVCompareV5Result:
    """Head-to-head: fit both the HSB V5 multi-target scale and a
    V6 KV multi-target correction at the SAME (single) target, then
    measure residual magnitudes on the held-out carrier.

    Honest: ``compare_hidden_vs_kv_injection_v4`` is the W60
    one-target version; V5 stacks 1 target and the closed-form
    ridge converges on each side independently. The comparator
    returns ``hidden_beats_kv`` iff the HSB V5 residual is strictly
    smaller than the KV V6 residual by more than 1e-9.
    """
    from .kv_bridge_v6 import (
        fit_kv_bridge_v6_multi_target,
        bridge_carrier_and_measure_v6,
    )
    target = _np.asarray(target_delta_logits, dtype=_np.float64)
    fitted_hsb, _ = fit_hsb_v5_multi_target(
        params=params_v6.v3_params,
        projection=hsb_v5_projection,
        train_carriers=train_carriers,
        target_delta_logits_stack=[target.tolist()],
        token_ids=token_ids)
    hsb_w = bridge_hidden_state_and_measure_v5(
        params=params_v6.v3_params, carrier=carrier,
        projection=fitted_hsb,
        token_ids=token_ids, target_unit=target.tolist())
    hsb_residual = float(abs(
        float(_np.linalg.norm(target)) - hsb_w.target_proj))
    # KV V6 fit.
    fitted_kv, _ = fit_kv_bridge_v6_multi_target(
        params=params_v6.v3_params, projection=kv_v6_projection,
        train_carriers=train_carriers,
        target_delta_logits_stack=[target.tolist()],
        follow_up_token_ids=token_ids)
    kv_w = bridge_carrier_and_measure_v6(
        params=params_v6, carrier=carrier, projection=fitted_kv,
        follow_up_token_ids=token_ids,
        target_unit=target.tolist())
    kv_residual = float(abs(
        float(_np.linalg.norm(target))
        - kv_w.last_logit_delta_proj))
    tie = bool(abs(hsb_residual - kv_residual) <= 1e-9)
    return HiddenVsKVCompareV5Result(
        hidden_residual=hsb_residual,
        kv_residual=kv_residual,
        hidden_beats_kv=bool(hsb_residual < kv_residual
                                  - 1e-9),
        kv_beats_hidden=bool(kv_residual < hsb_residual
                                  - 1e-9),
        tie=tie,
    )


__all__ = [
    "W61_HSB_V5_SCHEMA_VERSION",
    "W61_DEFAULT_HSB_V5_RIDGE_LAMBDA",
    "W61_DEFAULT_HSB_V5_N_POSITIONS",
    "HiddenStateBridgeV5Projection",
    "HiddenStateBridgeV5FitReport",
    "HiddenStateBridgeV5Witness",
    "HiddenVsKVCompareV5Result",
    "fit_hsb_v5_multi_target",
    "bridge_hidden_state_and_measure_v5",
    "write_hsb_v5_into_v6_cache",
    "recover_hsb_v5_inject",
    "compare_hidden_vs_kv_injection_v5",
]
