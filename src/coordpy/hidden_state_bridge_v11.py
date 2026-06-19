"""W67 M3 — Hidden-State Bridge V11.

Strictly extends W66's ``coordpy.hidden_state_bridge_v10``. V10 fit a
7-target stack (6 V9 + 1 team-consensus-under-budget target). V11
adds:

* **Eight-target stacked ridge fit** —
  ``fit_hsb_v11_eight_target`` adds a *role-dropout* target column.
* **Substrate-measured per-(L, H) hidden-vs-branch-merge probe** —
  ``probe_hsb_v11_hidden_vs_branch_merge``.
* **Branch-merge margin** —
  ``compute_hsb_v11_branch_merge_margin`` returns the margin by
  which hidden injection beats KV/prefix/replay under branch-merge
  reconciliation.

Honest scope (W67)
------------------

* All fits remain closed-form linear (W67-L-V12-NO-AUTOGRAD-CAP).
* The eighth target is *constructed*.
* The hidden-vs-branch-merge probe is a finite synthetic grid.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.hidden_state_bridge_v11 requires numpy") from exc

from .hidden_state_bridge_v10 import (
    HiddenStateBridgeV10Projection,
    fit_hsb_v10_seven_target,
)
from .hidden_state_bridge_v5 import fit_hsb_v5_multi_target
from .tiny_substrate_v3 import (
    TinyV3SubstrateParams, _sha256_hex,
)


W67_HSB_V11_SCHEMA_VERSION: str = (
    "coordpy.hidden_state_bridge_v11.v1")
W67_DEFAULT_HSB_V11_RIDGE_LAMBDA: float = 0.05


@dataclasses.dataclass
class HiddenStateBridgeV11Projection:
    inner_v10: HiddenStateBridgeV10Projection
    seed_v11: int

    @classmethod
    def init_from_v10(
            cls, inner: HiddenStateBridgeV10Projection,
            *, seed_v11: int = 671000,
    ) -> "HiddenStateBridgeV11Projection":
        return cls(inner_v10=inner, seed_v11=int(seed_v11))

    @property
    def carrier_dim(self) -> int:
        return int(self.inner_v10.carrier_dim)

    @property
    def n_layers(self) -> int:
        return int(self.inner_v10.n_layers)

    @property
    def n_heads(self) -> int:
        return int(self.inner_v10.n_heads)

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W67_HSB_V11_SCHEMA_VERSION,
            "kind": "hsb_v11_projection",
            "inner_v10_cid": str(self.inner_v10.cid()),
            "seed_v11": int(self.seed_v11),
        })


@dataclasses.dataclass(frozen=True)
class HiddenStateBridgeV11FitReport:
    schema: str
    n_targets: int
    per_target_pre_residual: tuple[float, ...]
    per_target_post_residual: tuple[float, ...]
    role_dropout_target_index: int
    role_dropout_pre: float
    role_dropout_post: float
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
            "role_dropout_target_index": int(
                self.role_dropout_target_index),
            "role_dropout_pre": float(round(
                self.role_dropout_pre, 12)),
            "role_dropout_post": float(round(
                self.role_dropout_post, 12)),
            "converged": bool(self.converged),
            "ridge_lambda": float(round(self.ridge_lambda, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hsb_v11_fit_report",
            "report": self.to_dict()})


def fit_hsb_v11_eight_target(
        *, params: TinyV3SubstrateParams,
        projection: HiddenStateBridgeV11Projection,
        train_carriers: Sequence[Sequence[float]],
        target_delta_logits_stack: Sequence[Sequence[float]],
        follow_up_token_ids: Sequence[int],
        role_dropout_target_index: int = 7,
        n_directions: int = 3,
        ridge_lambda: float = W67_DEFAULT_HSB_V11_RIDGE_LAMBDA,
) -> tuple[
        HiddenStateBridgeV11Projection,
        HiddenStateBridgeV11FitReport]:
    """Eight-target stacked ridge: 7 V10 + 1 role-dropout."""
    n_targets = int(len(target_delta_logits_stack))
    if n_targets < 1:
        raise ValueError("must provide >= 1 target")
    primary = list(target_delta_logits_stack[:7])
    while len(primary) < 7:
        primary.append(primary[0] if primary else [0.0])
    v10_fit, v10_report = fit_hsb_v10_seven_target(
        params=params, projection=projection.inner_v10,
        train_carriers=list(train_carriers),
        target_delta_logits_stack=primary,
        token_ids=list(follow_up_token_ids),
        ridge_lambda=float(ridge_lambda))
    if n_targets >= int(role_dropout_target_index) + 1:
        role_dropout = list(target_delta_logits_stack[
            int(role_dropout_target_index)])
    else:
        role_dropout = list(target_delta_logits_stack[-1])
    inner_v5 = (
        v10_fit.inner_v9.inner_v8.inner_v7.inner_v6.inner_v5)
    v5_fit, v5_audit = fit_hsb_v5_multi_target(
        params=params, projection=inner_v5,
        train_carriers=list(train_carriers),
        target_delta_logits_stack=[role_dropout],
        token_ids=list(follow_up_token_ids),
        ridge_lambda=float(ridge_lambda))
    new_inner_v6 = dataclasses.replace(
        v10_fit.inner_v9.inner_v8.inner_v7.inner_v6,
        inner_v5=v5_fit)
    new_inner_v7 = dataclasses.replace(
        v10_fit.inner_v9.inner_v8.inner_v7, inner_v6=new_inner_v6)
    new_inner_v8 = dataclasses.replace(
        v10_fit.inner_v9.inner_v8, inner_v7=new_inner_v7)
    new_inner_v9 = dataclasses.replace(
        v10_fit.inner_v9, inner_v8=new_inner_v8)
    new_v10 = dataclasses.replace(v10_fit, inner_v9=new_inner_v9)
    new_proj = dataclasses.replace(projection, inner_v10=new_v10)
    pre8 = float(v5_audit.pre_fit_residual)
    post8 = float(v5_audit.post_fit_residual)
    per_pre = list(v10_report.per_target_pre_residual) + [pre8]
    per_post = list(v10_report.per_target_post_residual) + [post8]
    converged = bool(
        all(po <= pr + 1e-9
            for pr, po in zip(per_pre[:7], per_post[:7]))
        and per_post[7] <= per_pre[7] + 1e-3)
    report = HiddenStateBridgeV11FitReport(
        schema=W67_HSB_V11_SCHEMA_VERSION,
        n_targets=int(n_targets),
        per_target_pre_residual=tuple(per_pre),
        per_target_post_residual=tuple(per_post),
        role_dropout_target_index=int(role_dropout_target_index),
        role_dropout_pre=float(pre8),
        role_dropout_post=float(post8),
        converged=bool(converged),
        ridge_lambda=float(ridge_lambda),
    )
    return new_proj, report


def probe_hsb_v11_hidden_vs_branch_merge(
        *, n_layers: int, n_heads: int,
        hidden_residual_l2_grid: Sequence[Sequence[float]],
        branch_merge_residual_l2_grid: Sequence[Sequence[float]],
        win_threshold: float = 0.1,
) -> dict[str, Any]:
    """Per-(L, H) hidden-vs-branch-merge joint win-rate probe."""
    h = _np.asarray(hidden_residual_l2_grid, dtype=_np.float64)
    b = _np.asarray(branch_merge_residual_l2_grid, dtype=_np.float64)
    if h.shape != (int(n_layers), int(n_heads)) or h.shape != b.shape:
        raise ValueError("grids must be (n_layers, n_heads)")
    wins = ((h + float(win_threshold)) < b).astype(_np.float64)
    return {
        "schema": W67_HSB_V11_SCHEMA_VERSION,
        "kind": "hidden_vs_branch_merge",
        "per_layer_head_win_rate": [
            [float(round(float(x), 12)) for x in row]
            for row in wins.tolist()],
        "win_rate_mean": float(round(float(wins.mean()), 12)),
        "n_layers": int(n_layers),
        "n_heads": int(n_heads),
    }


def compute_hsb_v11_branch_merge_margin(
        *,
        hidden_residual_l2: float,
        kv_residual_l2: float,
        prefix_residual_l2: float,
        replay_residual_l2: float,
        recover_residual_l2: float,
        branch_merge_residual_l2: float,
) -> float:
    """Positive when hidden residual is strictly less than min of
    every alternative including branch-merge."""
    others = float(min(
        kv_residual_l2, prefix_residual_l2, replay_residual_l2,
        recover_residual_l2, branch_merge_residual_l2))
    return float(round(max(0.0, others - float(hidden_residual_l2)), 12))


@dataclasses.dataclass(frozen=True)
class HSBV11Witness:
    schema: str
    projection_cid: str
    fit_report_cid: str
    hidden_vs_branch_merge_probe_cid: str
    branch_merge_margin: float
    hidden_vs_branch_merge_mean: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "projection_cid": str(self.projection_cid),
            "fit_report_cid": str(self.fit_report_cid),
            "hidden_vs_branch_merge_probe_cid": str(
                self.hidden_vs_branch_merge_probe_cid),
            "branch_merge_margin": float(round(
                self.branch_merge_margin, 12)),
            "hidden_vs_branch_merge_mean": float(round(
                self.hidden_vs_branch_merge_mean, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hsb_v11_witness",
            "witness": self.to_dict()})


def emit_hsb_v11_witness(
        *, projection: HiddenStateBridgeV11Projection,
        fit_report: HiddenStateBridgeV11FitReport | None = None,
        hidden_vs_branch_merge_probe: (
            dict[str, Any] | None) = None,
        branch_merge_margin: float = 0.0,
        hidden_vs_branch_merge_mean: float = 0.0,
) -> HSBV11Witness:
    return HSBV11Witness(
        schema=W67_HSB_V11_SCHEMA_VERSION,
        projection_cid=str(projection.cid()),
        fit_report_cid=(
            fit_report.cid() if fit_report is not None else ""),
        hidden_vs_branch_merge_probe_cid=(
            _sha256_hex(hidden_vs_branch_merge_probe)
            if hidden_vs_branch_merge_probe is not None else ""),
        branch_merge_margin=float(branch_merge_margin),
        hidden_vs_branch_merge_mean=float(
            hidden_vs_branch_merge_mean),
    )


__all__ = [
    "W67_HSB_V11_SCHEMA_VERSION",
    "W67_DEFAULT_HSB_V11_RIDGE_LAMBDA",
    "HiddenStateBridgeV11Projection",
    "HiddenStateBridgeV11FitReport",
    "fit_hsb_v11_eight_target",
    "probe_hsb_v11_hidden_vs_branch_merge",
    "compute_hsb_v11_branch_merge_margin",
    "HSBV11Witness",
    "emit_hsb_v11_witness",
]
