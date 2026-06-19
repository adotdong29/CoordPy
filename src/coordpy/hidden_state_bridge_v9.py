"""W65 M3 — Hidden-State Bridge V9.

Strictly extends W64's ``coordpy.hidden_state_bridge_v8``. V8 fit
a 5-target stack with explicit hidden-wins-primary target. V9 adds:

* **Six-target stacked ridge fit** —
  ``fit_hsb_v9_six_target`` is V8's five-target fit with an added
  *team-coordination target* column.
* **Substrate-measured per-(L, H) hidden-wins-rate probe** —
  ``probe_hsb_v9_hidden_wins_rate`` evaluates how often the
  hidden residual beats the KV residual across (L, H) under a
  set of test deltas and returns a per-(L, H) win-rate matrix.
* **Team-coordination margin** —
  ``compute_hsb_v9_team_coordination_margin`` returns a positive
  scalar when the hidden injection's residual is strictly less
  than min(kv, prefix, replay).

Honest scope (W65)
------------------

* All fits delegate to V8/V5 closed-form ridge. No new gradient
  descent. ``W65-L-V9-HSB-NO-AUTOGRAD-CAP`` documents.
* The team-coordination target is *constructed*.
* The hidden-wins-rate probe is over a finite synthetic feature
  grid; it is NOT a measurement on real models.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.hidden_state_bridge_v9 requires numpy") from exc

from .hidden_state_bridge_v5 import fit_hsb_v5_multi_target
from .hidden_state_bridge_v8 import (
    HiddenStateBridgeV8Projection,
    fit_hsb_v8_five_target,
    compute_hsb_v8_hidden_wins_primary_margin,
)
from .tiny_substrate_v3 import (
    TinyV3SubstrateParams, _sha256_hex,
)


W65_HSB_V9_SCHEMA_VERSION: str = (
    "coordpy.hidden_state_bridge_v9.v1")
W65_DEFAULT_HSB_V9_RIDGE_LAMBDA: float = 0.05


@dataclasses.dataclass
class HiddenStateBridgeV9Projection:
    inner_v8: HiddenStateBridgeV8Projection
    seed_v9: int

    @classmethod
    def init_from_v8(
            cls, inner: HiddenStateBridgeV8Projection,
            *, seed_v9: int = 650900,
    ) -> "HiddenStateBridgeV9Projection":
        return cls(inner_v8=inner, seed_v9=int(seed_v9))

    @property
    def carrier_dim(self) -> int:
        return int(self.inner_v8.carrier_dim)

    @property
    def n_layers(self) -> int:
        return int(self.inner_v8.n_layers)

    @property
    def n_heads(self) -> int:
        return int(self.inner_v8.n_heads)

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W65_HSB_V9_SCHEMA_VERSION,
            "kind": "hsb_v9_projection",
            "inner_v8_cid": str(self.inner_v8.cid()),
            "seed_v9": int(self.seed_v9),
        })


@dataclasses.dataclass(frozen=True)
class HiddenStateBridgeV9FitReport:
    schema: str
    n_targets: int
    per_target_pre_residual: tuple[float, ...]
    per_target_post_residual: tuple[float, ...]
    team_coordination_target_index: int
    team_coordination_pre: float
    team_coordination_post: float
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
            "team_coordination_target_index": int(
                self.team_coordination_target_index),
            "team_coordination_pre": float(round(
                self.team_coordination_pre, 12)),
            "team_coordination_post": float(round(
                self.team_coordination_post, 12)),
            "worst_index": int(self.worst_index),
            "worst_pre": float(round(self.worst_pre, 12)),
            "worst_post": float(round(self.worst_post, 12)),
            "converged": bool(self.converged),
            "ridge_lambda": float(round(self.ridge_lambda, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hsb_v9_fit_report",
            "report": self.to_dict()})


def fit_hsb_v9_six_target(
        *, params: TinyV3SubstrateParams,
        projection: HiddenStateBridgeV9Projection,
        train_carriers: Sequence[Sequence[float]],
        target_delta_logits_stack: Sequence[Sequence[float]],
        token_ids: Sequence[int],
        team_coordination_target_index: int = 5,
        ridge_lambda: float = W65_DEFAULT_HSB_V9_RIDGE_LAMBDA,
) -> tuple[HiddenStateBridgeV9Projection,
            HiddenStateBridgeV9FitReport]:
    n_targets = int(len(target_delta_logits_stack))
    if n_targets < 1:
        raise ValueError("must provide >= 1 target")
    primary = list(target_delta_logits_stack[:5])
    while len(primary) < 5:
        primary.append(primary[0] if primary else [0.0] * 1)
    v8_fit, v8_report = fit_hsb_v8_five_target(
        params=params, projection=projection.inner_v8,
        train_carriers=list(train_carriers),
        target_delta_logits_stack=primary,
        token_ids=list(token_ids),
        ridge_lambda=float(ridge_lambda))
    if n_targets >= int(team_coordination_target_index) + 1:
        team_target = list(
            target_delta_logits_stack[
                int(team_coordination_target_index)])
    else:
        team_target = list(target_delta_logits_stack[-1])
    fitted_v5, v5_report = fit_hsb_v5_multi_target(
        params=params,
        projection=v8_fit.inner_v7.inner_v6.inner_v5,
        train_carriers=list(train_carriers),
        target_delta_logits_stack=[team_target],
        token_ids=list(token_ids),
        ridge_lambda=float(ridge_lambda))
    new_inner_v6 = dataclasses.replace(
        v8_fit.inner_v7.inner_v6, inner_v5=fitted_v5)
    new_inner_v7 = dataclasses.replace(
        v8_fit.inner_v7, inner_v6=new_inner_v6)
    new_inner_v8 = dataclasses.replace(
        v8_fit, inner_v7=new_inner_v7)
    new_proj = dataclasses.replace(
        projection, inner_v8=new_inner_v8)
    per_pre = list(v8_report.per_target_pre_residual)[:5] + [
        float(v5_report.pre_fit_residual)]
    per_post = list(v8_report.per_target_post_residual)[:5] + [
        float(v5_report.post_fit_residual)]
    worst = int(_np.argmax(per_pre))
    converged = bool(
        all(po <= pr + 1e-9
            for pr, po in zip(per_pre, per_post)))
    return new_proj, HiddenStateBridgeV9FitReport(
        schema=W65_HSB_V9_SCHEMA_VERSION,
        n_targets=int(n_targets),
        per_target_pre_residual=tuple(per_pre),
        per_target_post_residual=tuple(per_post),
        team_coordination_target_index=int(
            team_coordination_target_index),
        team_coordination_pre=float(v5_report.pre_fit_residual),
        team_coordination_post=float(v5_report.post_fit_residual),
        worst_index=int(worst),
        worst_pre=float(per_pre[worst]),
        worst_post=float(per_post[worst]),
        converged=bool(converged),
        ridge_lambda=float(ridge_lambda),
    )


def probe_hsb_v9_hidden_wins_rate(
        *, n_layers: int, n_heads: int,
        hidden_residual_l2_per_lh: Sequence[Sequence[float]],
        kv_residual_l2_per_lh: Sequence[Sequence[float]],
) -> dict[str, Any]:
    """Per-(L, H) win-rate matrix: 1.0 where hidden beats KV."""
    win = _np.zeros((int(n_layers), int(n_heads)), dtype=_np.float64)
    for li in range(int(n_layers)):
        for hi in range(int(n_heads)):
            h = float(hidden_residual_l2_per_lh[li][hi])
            k = float(kv_residual_l2_per_lh[li][hi])
            win[li, hi] = float(1.0 if h < k else 0.0)
    return {
        "schema": W65_HSB_V9_SCHEMA_VERSION,
        "kind": "hidden_wins_rate_probe",
        "n_layers": int(n_layers),
        "n_heads": int(n_heads),
        "mean_win_rate": float(win.mean()),
        "max_win_rate": float(win.max()),
        "argmax_lh": [int(_np.unravel_index(
            int(_np.argmax(win)), win.shape)[0]),
            int(_np.unravel_index(
                int(_np.argmax(win)), win.shape)[1])],
    }


def compute_hsb_v9_team_coordination_margin(
        *, hidden_residual_l2: float,
        kv_residual_l2: float,
        prefix_residual_l2: float,
        replay_residual_l2: float,
) -> float:
    """Returns min(kv, prefix, replay) - hidden. Positive ⇒ hidden
    is the best team-coordination bridge."""
    return float(min(
        float(kv_residual_l2),
        float(prefix_residual_l2),
        float(replay_residual_l2)) - float(hidden_residual_l2))


@dataclasses.dataclass(frozen=True)
class HiddenStateBridgeV9Witness:
    schema: str
    projection_cid: str
    fit_report_cid: str
    team_coordination_margin: float
    hidden_wins_rate_mean: float
    hidden_wins_primary_margin: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "projection_cid": str(self.projection_cid),
            "fit_report_cid": str(self.fit_report_cid),
            "team_coordination_margin": float(round(
                self.team_coordination_margin, 12)),
            "hidden_wins_rate_mean": float(round(
                self.hidden_wins_rate_mean, 12)),
            "hidden_wins_primary_margin": float(round(
                self.hidden_wins_primary_margin, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hsb_v9_witness",
            "witness": self.to_dict()})


def emit_hsb_v9_witness(
        *, projection: HiddenStateBridgeV9Projection,
        fit_report: HiddenStateBridgeV9FitReport | None = None,
        team_coordination_margin: float = 0.0,
        hidden_wins_rate_mean: float = 0.0,
        hidden_wins_primary_margin: float = 0.0,
) -> HiddenStateBridgeV9Witness:
    return HiddenStateBridgeV9Witness(
        schema=W65_HSB_V9_SCHEMA_VERSION,
        projection_cid=str(projection.cid()),
        fit_report_cid=(
            fit_report.cid() if fit_report is not None else ""),
        team_coordination_margin=float(team_coordination_margin),
        hidden_wins_rate_mean=float(hidden_wins_rate_mean),
        hidden_wins_primary_margin=float(hidden_wins_primary_margin),
    )


__all__ = [
    "W65_HSB_V9_SCHEMA_VERSION",
    "W65_DEFAULT_HSB_V9_RIDGE_LAMBDA",
    "HiddenStateBridgeV9Projection",
    "HiddenStateBridgeV9FitReport",
    "fit_hsb_v9_six_target",
    "probe_hsb_v9_hidden_wins_rate",
    "compute_hsb_v9_team_coordination_margin",
    "HiddenStateBridgeV9Witness",
    "emit_hsb_v9_witness",
    "compute_hsb_v8_hidden_wins_primary_margin",
]
