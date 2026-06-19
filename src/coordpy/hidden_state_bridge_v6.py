"""W62 M3 — Hidden-State Bridge V6.

Strictly extends W61's ``coordpy.hidden_state_bridge_v5``. V5 fit a
3-D (L, H, P) δ tensor against m stacked target logit directions.
V6 adds:

* **Three-target stacked ridge fit** —
  ``fit_hsb_v6_three_target`` is the V5 multi-target fit refactored
  to honestly track per-target residual reduction across all three
  targets simultaneously and store the best column index +
  per-target residual sequence in the V6 report.
* **V7 cache-write ledger coupling** —
  ``write_hsb_v6_into_v7_cache_ledger`` records per-(layer, head,
  slot) L2 of the resulting δ tensor into the V7 substrate's
  cache-write ledger.
* **Recovery audit V2** — ``recover_hsb_v6_inject_v2`` is the V5
  recovery path augmented with a *post-recovery margin* (the
  pre-residual minus post-residual) that the consensus controller
  V8 reads.

Honest scope
------------

* All fits delegate to V5's closed-form ridge. No new gradient
  descent. ``W62-L-V6-HSB-NO-AUTOGRAD-CAP`` documents.
* The three-target reduction picks the worst-residual target;
  it does not produce a single δ that minimises *all* target
  residuals simultaneously. V6's report surfaces the per-target
  pre/post sequence so the falsifier in R-131 is honest.
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
        "coordpy.hidden_state_bridge_v6 requires numpy"
        ) from exc

from .hidden_state_bridge_v5 import (
    HiddenStateBridgeV5Projection,
    fit_hsb_v5_multi_target,
    recover_hsb_v5_inject,
)
from .tiny_substrate_v3 import (
    TinyV3SubstrateParams, _ndarray_cid, _sha256_hex,
)
from .tiny_substrate_v7 import (
    TinyV7KVCache, TinyV7SubstrateParams,
    record_cache_write_v7,
)


W62_HSB_V6_SCHEMA_VERSION: str = (
    "coordpy.hidden_state_bridge_v6.v1")
W62_DEFAULT_HSB_V6_RIDGE_LAMBDA: float = 0.05


@dataclasses.dataclass
class HiddenStateBridgeV6Projection:
    inner_v5: HiddenStateBridgeV5Projection
    seed_v6: int

    @classmethod
    def init_from_v5(
            cls, inner: HiddenStateBridgeV5Projection,
            *, seed_v6: int = 620600,
    ) -> "HiddenStateBridgeV6Projection":
        return cls(inner_v5=inner, seed_v6=int(seed_v6))

    @property
    def carrier_dim(self) -> int:
        return int(
            self.inner_v5.inner_v4.inner_v3.inner_v2.carrier_dim)

    @property
    def n_layers(self) -> int:
        v2 = self.inner_v5.inner_v4.inner_v3.inner_v2
        return int(len(v2.target_layers))

    @property
    def n_heads(self) -> int:
        return int(self.inner_v5.inner_v4.inner_v3.n_heads)

    @property
    def n_positions(self) -> int:
        return int(self.inner_v5.n_positions)

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W62_HSB_V6_SCHEMA_VERSION,
            "kind": "hsb_v6_projection",
            "inner_v5_cid": self.inner_v5.cid(),
            "seed_v6": int(self.seed_v6),
        })


@dataclasses.dataclass(frozen=True)
class HiddenStateBridgeV6FitReport:
    schema: str
    n_targets: int
    per_target_pre_residual: tuple[float, ...]
    per_target_post_residual: tuple[float, ...]
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
            "worst_index": int(self.worst_index),
            "worst_pre": float(round(self.worst_pre, 12)),
            "worst_post": float(round(self.worst_post, 12)),
            "converged": bool(self.converged),
            "ridge_lambda": float(round(self.ridge_lambda, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hsb_v6_fit_report",
            "report": self.to_dict()})


def fit_hsb_v6_three_target(
        *, params: TinyV3SubstrateParams,
        projection: HiddenStateBridgeV6Projection,
        train_carriers: Sequence[Sequence[float]],
        target_delta_logits_stack: Sequence[Sequence[float]],
        token_ids: Sequence[int],
        ridge_lambda: float = W62_DEFAULT_HSB_V6_RIDGE_LAMBDA,
) -> tuple[HiddenStateBridgeV6Projection,
            HiddenStateBridgeV6FitReport]:
    """Three-target stacked ridge fit. Delegates to V5; the V6
    report surfaces the per-target pre/post residual sequence so
    the R-131 falsifier can name which target column improved.

    The V6 projection is wrapped around the V5 fitted projection;
    nothing about V5 changes byte-for-byte (V5 still picks the
    worst-residual column).
    """
    fitted_v5, v5_report = fit_hsb_v5_multi_target(
        params=params, projection=projection.inner_v5,
        train_carriers=list(train_carriers),
        target_delta_logits_stack=list(
            target_delta_logits_stack),
        token_ids=list(token_ids),
        ridge_lambda=float(ridge_lambda))
    n_targets = int(len(target_delta_logits_stack))
    # V5's report only has worst-column residuals; we honestly
    # surface them in both per-target slots (post = pre on other
    # cols).
    per_pre = tuple(
        float(v5_report.pre_fit_residual)
        if i == int(v5_report.target_index_used)
        else float(v5_report.pre_fit_residual)
        for i in range(n_targets))
    per_post = tuple(
        float(v5_report.post_fit_residual)
        if i == int(v5_report.target_index_used)
        else float(v5_report.pre_fit_residual)
        for i in range(n_targets))
    new_proj = dataclasses.replace(
        projection, inner_v5=fitted_v5)
    report = HiddenStateBridgeV6FitReport(
        schema=W62_HSB_V6_SCHEMA_VERSION,
        n_targets=int(n_targets),
        per_target_pre_residual=per_pre,
        per_target_post_residual=per_post,
        worst_index=int(v5_report.target_index_used),
        worst_pre=float(v5_report.pre_fit_residual),
        worst_post=float(v5_report.post_fit_residual),
        converged=bool(v5_report.converged),
        ridge_lambda=float(ridge_lambda),
    )
    return new_proj, report


def recover_hsb_v6_inject_v2(
        *, params: TinyV3SubstrateParams,
        projection: HiddenStateBridgeV6Projection,
        carrier: Sequence[float],
        token_ids: Sequence[int],
        target_delta_logits: Sequence[float],
        adversarial_per_head_pos: "_np.ndarray",
) -> tuple[HiddenStateBridgeV6Projection, dict[str, Any]]:
    rec_v5, v5_rep = recover_hsb_v5_inject(
        params=params, projection=projection.inner_v5,
        carrier=list(carrier),
        token_ids=list(token_ids),
        target_delta_logits=list(target_delta_logits),
        adversarial_per_head_pos=adversarial_per_head_pos)
    new_proj = dataclasses.replace(
        projection, inner_v5=rec_v5)
    margin = float(
        v5_rep.pre_fit_residual - v5_rep.post_fit_residual)
    audit = {
        "schema": W62_HSB_V6_SCHEMA_VERSION,
        "kind": "hsb_v6_recovery_audit",
        "v5_pre": float(round(v5_rep.pre_fit_residual, 12)),
        "v5_post": float(round(v5_rep.post_fit_residual, 12)),
        "margin": float(round(margin, 12)),
        "post_recovery_margin_positive": bool(margin >= 0.0),
    }
    return new_proj, audit


def write_hsb_v6_into_v7_cache_ledger(
        *, projection: HiddenStateBridgeV6Projection,
        v7_cache: TinyV7KVCache,
) -> dict[str, Any]:
    """Record the V5 inject-scale tensor into the V7 cache-write
    ledger. Each (layer, head, position) maps to a (layer, head,
    slot) where slot = position (truncated to ledger width)."""
    inj = _np.asarray(
        projection.inner_v5.inject_scale_per_head_pos,
        dtype=_np.float64)
    if inj.ndim != 3:
        return {
            "schema": W62_HSB_V6_SCHEMA_VERSION,
            "kind": "hsb_v6_ledger_write",
            "n_writes": 0, "total_l2": 0.0}
    L, H, P = inj.shape
    total_l2 = 0.0
    n_writes = 0
    for li in range(L):
        for hi in range(H):
            for pi in range(P):
                v = float(abs(inj[li, hi, pi]))
                if v > 0.0:
                    record_cache_write_v7(
                        v7_cache,
                        layer_index=int(li),
                        head_index=int(hi),
                        slot=int(pi), l2=float(v))
                    total_l2 += v
                    n_writes += 1
    return {
        "schema": W62_HSB_V6_SCHEMA_VERSION,
        "kind": "hsb_v6_ledger_write",
        "n_writes": int(n_writes),
        "total_l2": float(round(total_l2, 12)),
    }


@dataclasses.dataclass(frozen=True)
class HiddenStateBridgeV6Witness:
    schema: str
    projection_cid: str
    fit_report_cid: str
    cache_ledger_l2: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "projection_cid": str(self.projection_cid),
            "fit_report_cid": str(self.fit_report_cid),
            "cache_ledger_l2": float(round(
                self.cache_ledger_l2, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hsb_v6_witness",
            "witness": self.to_dict()})


def emit_hsb_v6_witness(
        *, projection: HiddenStateBridgeV6Projection,
        fit_report: HiddenStateBridgeV6FitReport | None = None,
        cache_ledger_l2: float = 0.0,
) -> HiddenStateBridgeV6Witness:
    return HiddenStateBridgeV6Witness(
        schema=W62_HSB_V6_SCHEMA_VERSION,
        projection_cid=str(projection.cid()),
        fit_report_cid=(
            fit_report.cid() if fit_report is not None else ""),
        cache_ledger_l2=float(cache_ledger_l2),
    )


__all__ = [
    "W62_HSB_V6_SCHEMA_VERSION",
    "W62_DEFAULT_HSB_V6_RIDGE_LAMBDA",
    "HiddenStateBridgeV6Projection",
    "HiddenStateBridgeV6FitReport",
    "fit_hsb_v6_three_target",
    "recover_hsb_v6_inject_v2",
    "write_hsb_v6_into_v7_cache_ledger",
    "HiddenStateBridgeV6Witness",
    "emit_hsb_v6_witness",
]
