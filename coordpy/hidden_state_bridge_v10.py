"""W66 M3 — Hidden-State Bridge V10.

Strictly extends W65's ``coordpy.hidden_state_bridge_v9``. V9 fit a
6-target stack (5 V8 + 1 team-coordination target). V10 adds:

* **Seven-target stacked ridge fit** —
  ``fit_hsb_v10_seven_target`` adds a *team-consensus-under-budget*
  target column.
* **Substrate-measured per-(L, H) hidden-wins-vs-team-success
  probe** — ``probe_hsb_v10_hidden_wins_vs_team_success``.
* **Team-consensus margin** —
  ``compute_hsb_v10_team_consensus_margin`` returns the margin by
  which hidden injection beats KV/prefix/replay under tight budget.

Honest scope (W66)
------------------

* All fits remain closed-form linear (W66-L-V11-NO-AUTOGRAD-CAP).
* The seventh target is *constructed*.
* The hidden-wins-vs-team-success probe is a finite synthetic grid.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.hidden_state_bridge_v10 requires numpy") from exc

from .hidden_state_bridge_v5 import fit_hsb_v5_multi_target
from .hidden_state_bridge_v9 import (
    HiddenStateBridgeV9Projection,
    fit_hsb_v9_six_target,
)
from .tiny_substrate_v3 import (
    TinyV3SubstrateParams, _sha256_hex,
)


W66_HSB_V10_SCHEMA_VERSION: str = (
    "coordpy.hidden_state_bridge_v10.v1")
W66_DEFAULT_HSB_V10_RIDGE_LAMBDA: float = 0.05


@dataclasses.dataclass
class HiddenStateBridgeV10Projection:
    inner_v9: HiddenStateBridgeV9Projection
    seed_v10: int

    @classmethod
    def init_from_v9(
            cls, inner: HiddenStateBridgeV9Projection,
            *, seed_v10: int = 661000,
    ) -> "HiddenStateBridgeV10Projection":
        return cls(inner_v9=inner, seed_v10=int(seed_v10))

    @property
    def carrier_dim(self) -> int:
        return int(self.inner_v9.carrier_dim)

    @property
    def n_layers(self) -> int:
        return int(self.inner_v9.n_layers)

    @property
    def n_heads(self) -> int:
        return int(self.inner_v9.n_heads)

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W66_HSB_V10_SCHEMA_VERSION,
            "kind": "hsb_v10_projection",
            "inner_v9_cid": str(self.inner_v9.cid()),
            "seed_v10": int(self.seed_v10),
        })


@dataclasses.dataclass(frozen=True)
class HiddenStateBridgeV10FitReport:
    schema: str
    n_targets: int
    per_target_pre_residual: tuple[float, ...]
    per_target_post_residual: tuple[float, ...]
    team_consensus_target_index: int
    team_consensus_pre: float
    team_consensus_post: float
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
            "team_consensus_target_index": int(
                self.team_consensus_target_index),
            "team_consensus_pre": float(round(
                self.team_consensus_pre, 12)),
            "team_consensus_post": float(round(
                self.team_consensus_post, 12)),
            "converged": bool(self.converged),
            "ridge_lambda": float(round(self.ridge_lambda, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hsb_v10_fit_report",
            "report": self.to_dict()})


def fit_hsb_v10_seven_target(
        *, params: TinyV3SubstrateParams,
        projection: HiddenStateBridgeV10Projection,
        train_carriers: Sequence[Sequence[float]],
        target_delta_logits_stack: Sequence[Sequence[float]],
        token_ids: Sequence[int],
        team_consensus_target_index: int = 6,
        ridge_lambda: float = W66_DEFAULT_HSB_V10_RIDGE_LAMBDA,
) -> tuple[HiddenStateBridgeV10Projection,
            HiddenStateBridgeV10FitReport]:
    n_targets = int(len(target_delta_logits_stack))
    if n_targets < 1:
        raise ValueError("must provide >= 1 target")
    primary = list(target_delta_logits_stack[:6])
    while len(primary) < 6:
        primary.append(primary[0] if primary else [0.0])
    v9_fit, v9_report = fit_hsb_v9_six_target(
        params=params, projection=projection.inner_v9,
        train_carriers=list(train_carriers),
        target_delta_logits_stack=primary,
        token_ids=list(token_ids),
        ridge_lambda=float(ridge_lambda))
    if n_targets >= int(team_consensus_target_index) + 1:
        team_target = list(
            target_delta_logits_stack[
                int(team_consensus_target_index)])
    else:
        team_target = list(target_delta_logits_stack[-1])
    inner_v5 = (
        v9_fit.inner_v8.inner_v7.inner_v6.inner_v5)
    fitted_v5, v5_report = fit_hsb_v5_multi_target(
        params=params, projection=inner_v5,
        train_carriers=list(train_carriers),
        target_delta_logits_stack=[team_target],
        token_ids=list(token_ids),
        ridge_lambda=float(ridge_lambda))
    new_inner_v6 = dataclasses.replace(
        v9_fit.inner_v8.inner_v7.inner_v6, inner_v5=fitted_v5)
    new_inner_v7 = dataclasses.replace(
        v9_fit.inner_v8.inner_v7, inner_v6=new_inner_v6)
    new_inner_v8 = dataclasses.replace(
        v9_fit.inner_v8, inner_v7=new_inner_v7)
    new_v9 = dataclasses.replace(
        v9_fit, inner_v8=new_inner_v8)
    new_proj = dataclasses.replace(
        projection, inner_v9=new_v9)
    per_pre = list(v9_report.per_target_pre_residual)[:6] + [
        float(v5_report.pre_fit_residual)]
    per_post = list(v9_report.per_target_post_residual)[:6] + [
        float(v5_report.post_fit_residual)]
    converged = bool(
        all(po <= pr + 1e-9
            for pr, po in zip(per_pre[:6], per_post[:6]))
        and per_post[6] <= per_pre[6] + 1e-3)
    return new_proj, HiddenStateBridgeV10FitReport(
        schema=W66_HSB_V10_SCHEMA_VERSION,
        n_targets=int(n_targets),
        per_target_pre_residual=tuple(per_pre),
        per_target_post_residual=tuple(per_post),
        team_consensus_target_index=int(
            team_consensus_target_index),
        team_consensus_pre=float(v5_report.pre_fit_residual),
        team_consensus_post=float(v5_report.post_fit_residual),
        converged=bool(converged),
        ridge_lambda=float(ridge_lambda),
    )


def probe_hsb_v10_hidden_wins_vs_team_success(
        *, n_layers: int, n_heads: int,
        hidden_residual_l2_per_lh: Sequence[Sequence[float]],
        team_success_indicator_per_lh: Sequence[Sequence[float]],
) -> dict[str, Any]:
    """Per-(L, H) joint hidden-wins-and-team-success rate."""
    win = _np.zeros(
        (int(n_layers), int(n_heads)), dtype=_np.float64)
    for li in range(int(n_layers)):
        for hi in range(int(n_heads)):
            h = float(hidden_residual_l2_per_lh[li][hi])
            ts = float(team_success_indicator_per_lh[li][hi])
            # Hidden wins when residual low AND team succeeded.
            win[li, hi] = (
                1.0 if (h < 0.5 and ts > 0.5) else 0.0)
    return {
        "schema": W66_HSB_V10_SCHEMA_VERSION,
        "kind": "hidden_wins_vs_team_success_probe",
        "n_layers": int(n_layers),
        "n_heads": int(n_heads),
        "mean_joint_win_rate": float(win.mean()),
        "max_joint_win_rate": float(win.max()),
    }


def compute_hsb_v10_team_consensus_margin(
        *, hidden_residual_l2: float,
        kv_residual_l2: float,
        prefix_residual_l2: float,
        replay_residual_l2: float,
        recover_residual_l2: float,
) -> float:
    """Returns min(kv, prefix, replay, recover) - hidden."""
    return float(min(
        float(kv_residual_l2),
        float(prefix_residual_l2),
        float(replay_residual_l2),
        float(recover_residual_l2)) - float(hidden_residual_l2))


@dataclasses.dataclass(frozen=True)
class HiddenStateBridgeV10Witness:
    schema: str
    projection_cid: str
    fit_report_cid: str
    team_consensus_margin: float
    hidden_wins_vs_team_success_mean: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "projection_cid": str(self.projection_cid),
            "fit_report_cid": str(self.fit_report_cid),
            "team_consensus_margin": float(round(
                self.team_consensus_margin, 12)),
            "hidden_wins_vs_team_success_mean": float(round(
                self.hidden_wins_vs_team_success_mean, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hsb_v10_witness",
            "witness": self.to_dict()})


def emit_hsb_v10_witness(
        *, projection: HiddenStateBridgeV10Projection,
        fit_report: HiddenStateBridgeV10FitReport | None = None,
        team_consensus_margin: float = 0.0,
        hidden_wins_vs_team_success_mean: float = 0.0,
) -> HiddenStateBridgeV10Witness:
    return HiddenStateBridgeV10Witness(
        schema=W66_HSB_V10_SCHEMA_VERSION,
        projection_cid=str(projection.cid()),
        fit_report_cid=(
            fit_report.cid() if fit_report is not None else ""),
        team_consensus_margin=float(team_consensus_margin),
        hidden_wins_vs_team_success_mean=float(
            hidden_wins_vs_team_success_mean),
    )


__all__ = [
    "W66_HSB_V10_SCHEMA_VERSION",
    "W66_DEFAULT_HSB_V10_RIDGE_LAMBDA",
    "HiddenStateBridgeV10Projection",
    "HiddenStateBridgeV10FitReport",
    "fit_hsb_v10_seven_target",
    "probe_hsb_v10_hidden_wins_vs_team_success",
    "compute_hsb_v10_team_consensus_margin",
    "HiddenStateBridgeV10Witness",
    "emit_hsb_v10_witness",
]
