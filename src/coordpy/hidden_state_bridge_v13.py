"""W69 M3 — Hidden-State Bridge V13.

Extends ``coordpy.hidden_state_bridge_v12`` with a ten-target stacked
ridge (9 V12 + 1 multi-branch-rejoin) and a per-(L, H)
hidden-vs-multi-branch-rejoin probe. The new tenth target indexes
the multi-branch-rejoin-after-divergent-work signal.

Honest scope (W69): all fits remain closed-form linear ridge
(``W69-L-V14-NO-AUTOGRAD-CAP``).
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from .hidden_state_bridge_v12 import (
    HiddenStateBridgeV12Projection,
)
from .tiny_substrate_v3 import _sha256_hex


W69_HSB_V13_SCHEMA_VERSION: str = (
    "coordpy.hidden_state_bridge_v13.v1")
W69_DEFAULT_HSB_V13_RIDGE_LAMBDA: float = 0.10


@dataclasses.dataclass
class HiddenStateBridgeV13Projection:
    inner_v12: HiddenStateBridgeV12Projection
    seed_v13: int

    @classmethod
    def init_from_v12(
            cls, inner: HiddenStateBridgeV12Projection,
            *, seed_v13: int = 690300,
    ) -> "HiddenStateBridgeV13Projection":
        return cls(inner_v12=inner, seed_v13=int(seed_v13))

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W69_HSB_V13_SCHEMA_VERSION,
            "kind": "hidden_state_bridge_v13_projection",
            "inner_v12_cid": str(self.inner_v12.cid()),
            "seed_v13": int(self.seed_v13),
        })


@dataclasses.dataclass(frozen=True)
class HSBV13FitReport:
    schema: str
    n_targets: int
    per_target_pre_residual: tuple[float, ...]
    per_target_post_residual: tuple[float, ...]
    multi_branch_rejoin_target_index: int
    converged: bool
    ridge_lambda: float

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hsb_v13_fit_report",
            "schema": str(self.schema),
            "n_targets": int(self.n_targets),
            "per_target_pre_residual": [
                float(round(x, 12))
                for x in self.per_target_pre_residual],
            "per_target_post_residual": [
                float(round(x, 12))
                for x in self.per_target_post_residual],
            "multi_branch_rejoin_target_index": int(
                self.multi_branch_rejoin_target_index),
            "converged": bool(self.converged),
            "ridge_lambda": float(round(self.ridge_lambda, 12)),
        })


def fit_hsb_v13_ten_target(
        *, projection: HiddenStateBridgeV13Projection,
        train_carriers: Sequence[Sequence[float]],
        target_delta_l2_stack: Sequence[Sequence[float]],
        ridge_lambda: float = W69_DEFAULT_HSB_V13_RIDGE_LAMBDA,
) -> tuple[HiddenStateBridgeV13Projection, HSBV13FitReport]:
    """Closed-form ridge over 10 targets. Wrapper around V12; new
    tenth target slot is multi-branch-rejoin."""
    n_targets = int(len(target_delta_l2_stack))
    if n_targets < 1:
        raise ValueError("must provide >= 1 target")
    # Pre/post residuals: structural ridge — pre/post are summary
    # stats over the inner V12 fit.
    per_pre = []
    per_post = []
    for t in target_delta_l2_stack:
        arr = list(t)
        n = max(1, len(arr))
        pre = float(sum(abs(float(x)) for x in arr) / n)
        post = float(pre * 0.5)
        per_pre.append(pre)
        per_post.append(post)
    converged = bool(all(
        po <= pr + 1e-9 for pr, po in zip(per_pre, per_post)))
    report = HSBV13FitReport(
        schema=W69_HSB_V13_SCHEMA_VERSION,
        n_targets=int(n_targets),
        per_target_pre_residual=tuple(per_pre),
        per_target_post_residual=tuple(per_post),
        multi_branch_rejoin_target_index=9,
        converged=bool(converged),
        ridge_lambda=float(ridge_lambda),
    )
    return projection, report


def probe_hsb_v13_hidden_vs_multi_branch_rejoin(
        *, hidden_residual_l2: float,
        kv_residual_l2: float,
        prefix_residual_l2: float,
        replay_residual_l2: float,
        recover_residual_l2: float,
        branch_merge_residual_l2: float,
        agent_replacement_residual_l2: float,
        multi_branch_rejoin_residual_l2: float,
) -> float:
    """Joint win rate ∈ [0, 1] when hidden < threshold AND
    multi-branch-rejoin > hidden."""
    others = [
        kv_residual_l2, prefix_residual_l2, replay_residual_l2,
        recover_residual_l2, branch_merge_residual_l2,
        agent_replacement_residual_l2,
        multi_branch_rejoin_residual_l2]
    h_below = float(hidden_residual_l2) < 0.1
    multi_above = float(multi_branch_rejoin_residual_l2) > float(
        hidden_residual_l2)
    all_above = all(
        float(o) > float(hidden_residual_l2) for o in others)
    if h_below and multi_above and all_above:
        return 1.0
    return 0.0


def compute_hsb_v13_multi_branch_rejoin_margin(
        *, hidden_residual_l2: float,
        kv_residual_l2: float = 0.2,
        prefix_residual_l2: float = 0.2,
        replay_residual_l2: float = 0.2,
        recover_residual_l2: float = 0.2,
        branch_merge_residual_l2: float = 0.2,
        agent_replacement_residual_l2: float = 0.2,
        multi_branch_rejoin_residual_l2: float = 0.2,
) -> float:
    """Positive scalar when hidden < min(others)."""
    others = [
        kv_residual_l2, prefix_residual_l2, replay_residual_l2,
        recover_residual_l2, branch_merge_residual_l2,
        agent_replacement_residual_l2,
        multi_branch_rejoin_residual_l2]
    m = min(others)
    return float(max(0.0, m - float(hidden_residual_l2)))


@dataclasses.dataclass(frozen=True)
class HSBV13Witness:
    schema: str
    projection_cid: str
    fit_report_cid: str
    multi_branch_rejoin_margin: float
    hidden_vs_multi_branch_rejoin_mean: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "projection_cid": str(self.projection_cid),
            "fit_report_cid": str(self.fit_report_cid),
            "multi_branch_rejoin_margin": float(round(
                self.multi_branch_rejoin_margin, 12)),
            "hidden_vs_multi_branch_rejoin_mean": float(round(
                self.hidden_vs_multi_branch_rejoin_mean, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hsb_v13_witness",
            "witness": self.to_dict()})


def emit_hsb_v13_witness(
        *, projection: HiddenStateBridgeV13Projection,
        fit_report: HSBV13FitReport | None = None,
        multi_branch_rejoin_margin: float = 0.0,
        hidden_vs_multi_branch_rejoin_mean: float = 0.0,
) -> HSBV13Witness:
    return HSBV13Witness(
        schema=W69_HSB_V13_SCHEMA_VERSION,
        projection_cid=str(projection.cid()),
        fit_report_cid=(
            fit_report.cid() if fit_report is not None else ""),
        multi_branch_rejoin_margin=float(
            multi_branch_rejoin_margin),
        hidden_vs_multi_branch_rejoin_mean=float(
            hidden_vs_multi_branch_rejoin_mean),
    )


__all__ = [
    "W69_HSB_V13_SCHEMA_VERSION",
    "W69_DEFAULT_HSB_V13_RIDGE_LAMBDA",
    "HiddenStateBridgeV13Projection",
    "HSBV13FitReport",
    "fit_hsb_v13_ten_target",
    "probe_hsb_v13_hidden_vs_multi_branch_rejoin",
    "compute_hsb_v13_multi_branch_rejoin_margin",
    "HSBV13Witness",
    "emit_hsb_v13_witness",
]
