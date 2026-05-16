"""W63 M3 — Hidden-State Bridge V7.

Strictly extends W62's ``coordpy.hidden_state_bridge_v6``. V6 fit a
3-target stack against a (L, H, P) δ tensor. V7 adds:

* **Four-target stacked ridge fit** —
  ``fit_hsb_v7_four_target`` is V6's three-target fit with an
  added *hidden-wins target* column. The fourth target represents
  a δ that requires the hidden-state bridge to dominate the KV
  bridge (i.e. the hidden injection is the only path to reach
  that target logit direction).
* **V8 contention coupling** —
  ``write_hsb_v7_into_v8_contention`` records per-(layer, head,
  slot) ``|hidden|`` into the V8 hidden-vs-KV contention channel
  (positive side). The kv_bridge_v8 writes the negative side.
* **Recovery audit V3** — ``recover_hsb_v7_inject_v3`` is V6's
  recovery path with a *two-stage* recovery margin (post-recovery
  improvement + post-recovery basin width).
* **Hidden-wins margin** — ``compute_hsb_v7_hidden_wins_margin``
  returns a positive scalar when the hidden injection's residual
  is strictly less than the KV residual on the hidden-wins target.

Honest scope
------------

* All fits delegate to V6's closed-form ridge. No new gradient
  descent. ``W63-L-V7-HSB-NO-AUTOGRAD-CAP`` documents.
* The hidden-wins target is *constructed* — it is engineered such
  that only the hidden bridge can reach it. It is not a measured
  in-the-wild target.
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
        "coordpy.hidden_state_bridge_v7 requires numpy"
        ) from exc

from .hidden_state_bridge_v5 import (
    fit_hsb_v5_multi_target,
    recover_hsb_v5_inject,
)
from .hidden_state_bridge_v6 import (
    HiddenStateBridgeV6Projection,
    HiddenStateBridgeV6FitReport,
    fit_hsb_v6_three_target,
    recover_hsb_v6_inject_v2,
)
from .tiny_substrate_v3 import (
    TinyV3SubstrateParams, _ndarray_cid, _sha256_hex,
)
from .tiny_substrate_v8 import (
    TinyV8KVCache, TinyV8SubstrateParams,
    record_hidden_vs_kv_contention_v8,
)


W63_HSB_V7_SCHEMA_VERSION: str = (
    "coordpy.hidden_state_bridge_v7.v1")
W63_DEFAULT_HSB_V7_RIDGE_LAMBDA: float = 0.05


@dataclasses.dataclass
class HiddenStateBridgeV7Projection:
    inner_v6: HiddenStateBridgeV6Projection
    seed_v7: int

    @classmethod
    def init_from_v6(
            cls, inner: HiddenStateBridgeV6Projection,
            *, seed_v7: int = 630700,
    ) -> "HiddenStateBridgeV7Projection":
        return cls(inner_v6=inner, seed_v7=int(seed_v7))

    @property
    def carrier_dim(self) -> int:
        return int(self.inner_v6.carrier_dim)

    @property
    def n_layers(self) -> int:
        return int(self.inner_v6.n_layers)

    @property
    def n_heads(self) -> int:
        return int(self.inner_v6.n_heads)

    @property
    def n_positions(self) -> int:
        return int(self.inner_v6.n_positions)

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W63_HSB_V7_SCHEMA_VERSION,
            "kind": "hsb_v7_projection",
            "inner_v6_cid": self.inner_v6.cid(),
            "seed_v7": int(self.seed_v7),
        })


@dataclasses.dataclass(frozen=True)
class HiddenStateBridgeV7FitReport:
    schema: str
    n_targets: int
    per_target_pre_residual: tuple[float, ...]
    per_target_post_residual: tuple[float, ...]
    hidden_wins_target_index: int
    hidden_wins_pre: float
    hidden_wins_post: float
    worst_index: int
    worst_pre: float
    worst_post: float
    converged: bool
    ridge_lambda: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "n_targets": int(self.n_targets),
            "per_target_pre_residual": [
                float(round(float(x), 12))
                for x in self.per_target_pre_residual],
            "per_target_post_residual": [
                float(round(float(x), 12))
                for x in self.per_target_post_residual],
            "hidden_wins_target_index": int(
                self.hidden_wins_target_index),
            "hidden_wins_pre": float(round(
                self.hidden_wins_pre, 12)),
            "hidden_wins_post": float(round(
                self.hidden_wins_post, 12)),
            "worst_index": int(self.worst_index),
            "worst_pre": float(round(self.worst_pre, 12)),
            "worst_post": float(round(self.worst_post, 12)),
            "converged": bool(self.converged),
            "ridge_lambda": float(round(self.ridge_lambda, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hsb_v7_fit_report",
            "report": self.to_dict()})


def fit_hsb_v7_four_target(
        *, params: TinyV3SubstrateParams,
        projection: HiddenStateBridgeV7Projection,
        train_carriers: Sequence[Sequence[float]],
        target_delta_logits_stack: Sequence[Sequence[float]],
        token_ids: Sequence[int],
        hidden_wins_target_index: int = 3,
        ridge_lambda: float = W63_DEFAULT_HSB_V7_RIDGE_LAMBDA,
) -> tuple[HiddenStateBridgeV7Projection,
            HiddenStateBridgeV7FitReport]:
    """Four-target stacked ridge fit. Delegates to V6 for the first
    three targets, then a second V5 fit on the hidden-wins target.
    """
    n_targets = int(len(target_delta_logits_stack))
    if n_targets < 1:
        raise ValueError("must provide ≥ 1 target")
    primary = list(target_delta_logits_stack[:3])
    while len(primary) < 3:
        primary.append(primary[0] if primary
                       else [0.0] * 1)
    fitted_v6, v6_report = fit_hsb_v6_three_target(
        params=params, projection=projection.inner_v6,
        train_carriers=list(train_carriers),
        target_delta_logits_stack=primary,
        token_ids=list(token_ids),
        ridge_lambda=float(ridge_lambda))
    # Hidden-wins target second-pass V5 fit.
    if n_targets >= int(hidden_wins_target_index) + 1:
        hw_target = list(
            target_delta_logits_stack[
                int(hidden_wins_target_index)])
    else:
        hw_target = list(target_delta_logits_stack[-1])
    fitted_v5, v5_report = fit_hsb_v5_multi_target(
        params=params, projection=fitted_v6.inner_v5,
        train_carriers=list(train_carriers),
        target_delta_logits_stack=[hw_target],
        token_ids=list(token_ids),
        ridge_lambda=float(ridge_lambda))
    new_inner_v6 = dataclasses.replace(
        fitted_v6, inner_v5=fitted_v5)
    new_proj = dataclasses.replace(
        projection, inner_v6=new_inner_v6)
    per_pre_list = list(v6_report.per_target_pre_residual)[
        :3] + [float(v5_report.pre_fit_residual)]
    per_post_list = list(v6_report.per_target_post_residual)[
        :3] + [float(v5_report.post_fit_residual)]
    worst_idx = int(_np.argmax(per_pre_list))
    converged = bool(
        all(po <= pr + 1e-9
            for pr, po in zip(per_pre_list, per_post_list)))
    report = HiddenStateBridgeV7FitReport(
        schema=W63_HSB_V7_SCHEMA_VERSION,
        n_targets=int(n_targets),
        per_target_pre_residual=tuple(per_pre_list),
        per_target_post_residual=tuple(per_post_list),
        hidden_wins_target_index=int(hidden_wins_target_index),
        hidden_wins_pre=float(v5_report.pre_fit_residual),
        hidden_wins_post=float(v5_report.post_fit_residual),
        worst_index=int(worst_idx),
        worst_pre=float(per_pre_list[worst_idx]),
        worst_post=float(per_post_list[worst_idx]),
        converged=bool(converged),
        ridge_lambda=float(ridge_lambda),
    )
    return new_proj, report


def recover_hsb_v7_inject_v3(
        *, params: TinyV3SubstrateParams,
        projection: HiddenStateBridgeV7Projection,
        carrier: Sequence[float],
        token_ids: Sequence[int],
        target_delta_logits: Sequence[float],
        adversarial_per_head_pos: "_np.ndarray",
) -> tuple[HiddenStateBridgeV7Projection, dict[str, Any]]:
    """Two-stage recovery audit. Stage 1 = V6 recovery. Stage 2 =
    re-fit the V5 inner projection on the (corrupted) carrier and
    measure the post-recovery basin width.
    """
    new_v6, audit_v6 = recover_hsb_v6_inject_v2(
        params=params, projection=projection.inner_v6,
        carrier=list(carrier), token_ids=list(token_ids),
        target_delta_logits=list(target_delta_logits),
        adversarial_per_head_pos=adversarial_per_head_pos)
    # Stage 2: also run a V5 re-fit on the (recovered) inner V5
    # projection to confirm the basin width.
    rec_v5, v5_rep2 = recover_hsb_v5_inject(
        params=params,
        projection=new_v6.inner_v5,
        carrier=list(carrier), token_ids=list(token_ids),
        target_delta_logits=list(target_delta_logits),
        adversarial_per_head_pos=adversarial_per_head_pos)
    new_inner_v6 = dataclasses.replace(
        new_v6, inner_v5=rec_v5)
    new_proj = dataclasses.replace(
        projection, inner_v6=new_inner_v6)
    basin_width = float(
        v5_rep2.pre_fit_residual - v5_rep2.post_fit_residual)
    audit = {
        "schema": W63_HSB_V7_SCHEMA_VERSION,
        "kind": "hsb_v7_recovery_audit_v3",
        "stage1_audit": dict(audit_v6),
        "stage2_basin_width": float(round(basin_width, 12)),
        "stage2_pre": float(round(
            v5_rep2.pre_fit_residual, 12)),
        "stage2_post": float(round(
            v5_rep2.post_fit_residual, 12)),
        "two_stage_recovered": bool(
            audit_v6.get("post_recovery_margin_positive", False)
            and basin_width >= -1e-9),
    }
    return new_proj, audit


def write_hsb_v7_into_v8_contention(
        *, projection: HiddenStateBridgeV7Projection,
        v8_cache: TinyV8KVCache,
        kv_write_abs_per_slot: (
            Sequence[Sequence[Sequence[float]]] | None) = None,
) -> dict[str, Any]:
    """Record the V5 inject-scale tensor into the V8 hidden-vs-KV
    contention channel (positive side). If kv_write_abs_per_slot
    is provided, both sides are recorded simultaneously."""
    inj = _np.asarray(
        projection.inner_v6.inner_v5.inject_scale_per_head_pos,
        dtype=_np.float64)
    if inj.ndim != 3:
        return {
            "schema": W63_HSB_V7_SCHEMA_VERSION,
            "kind": "hsb_v7_contention_write",
            "n_writes": 0, "total_l2": 0.0}
    L, H, P = inj.shape
    total_l2 = 0.0
    n_writes = 0
    for li in range(L):
        for hi in range(H):
            for pi in range(P):
                hl2 = float(abs(inj[li, hi, pi]))
                kl2 = 0.0
                if kv_write_abs_per_slot is not None:
                    try:
                        kl2 = float(
                            kv_write_abs_per_slot[li][hi][pi])
                    except (IndexError, TypeError):
                        kl2 = 0.0
                if hl2 > 0.0 or kl2 > 0.0:
                    record_hidden_vs_kv_contention_v8(
                        v8_cache,
                        layer_index=int(li),
                        head_index=int(hi),
                        slot=int(pi),
                        hidden_write_abs=float(hl2),
                        kv_write_abs=float(kl2))
                    total_l2 += hl2
                    n_writes += 1
    return {
        "schema": W63_HSB_V7_SCHEMA_VERSION,
        "kind": "hsb_v7_contention_write",
        "n_writes": int(n_writes),
        "total_l2": float(round(total_l2, 12)),
    }


def compute_hsb_v7_hidden_wins_margin(
        *, hidden_residual_l2: float,
        kv_residual_l2: float,
) -> float:
    """Returns kv_residual - hidden_residual. Positive ⇒ hidden
    wins; negative ⇒ kv wins."""
    return float(kv_residual_l2) - float(hidden_residual_l2)


@dataclasses.dataclass(frozen=True)
class HiddenStateBridgeV7Witness:
    schema: str
    projection_cid: str
    fit_report_cid: str
    contention_l1: float
    hidden_wins_margin: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "projection_cid": str(self.projection_cid),
            "fit_report_cid": str(self.fit_report_cid),
            "contention_l1": float(round(
                self.contention_l1, 12)),
            "hidden_wins_margin": float(round(
                self.hidden_wins_margin, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hsb_v7_witness",
            "witness": self.to_dict()})


def emit_hsb_v7_witness(
        *, projection: HiddenStateBridgeV7Projection,
        fit_report: HiddenStateBridgeV7FitReport | None = None,
        contention_l1: float = 0.0,
        hidden_wins_margin: float = 0.0,
) -> HiddenStateBridgeV7Witness:
    return HiddenStateBridgeV7Witness(
        schema=W63_HSB_V7_SCHEMA_VERSION,
        projection_cid=str(projection.cid()),
        fit_report_cid=(
            fit_report.cid() if fit_report is not None else ""),
        contention_l1=float(contention_l1),
        hidden_wins_margin=float(hidden_wins_margin),
    )


__all__ = [
    "W63_HSB_V7_SCHEMA_VERSION",
    "W63_DEFAULT_HSB_V7_RIDGE_LAMBDA",
    "HiddenStateBridgeV7Projection",
    "HiddenStateBridgeV7FitReport",
    "fit_hsb_v7_four_target",
    "recover_hsb_v7_inject_v3",
    "write_hsb_v7_into_v8_contention",
    "compute_hsb_v7_hidden_wins_margin",
    "HiddenStateBridgeV7Witness",
    "emit_hsb_v7_witness",
]
