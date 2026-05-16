"""W68 M3 — Hidden-State Bridge V12.

Strictly extends W67's ``coordpy.hidden_state_bridge_v11``. V11 fit
an 8-target stack (7 V10 + 1 role-dropout). V12 adds:

* **Nine-target stacked ridge fit** —
  ``fit_hsb_v12_nine_target`` adds an *agent-replacement-warm-
  restart* target column.
* **Substrate-measured per-(L, H) hidden-vs-agent-replacement
  probe** — ``probe_hsb_v12_hidden_vs_agent_replacement``.
* **Agent-replacement margin** — positive when hidden injection
  strictly beats KV/prefix/replay/recover/branch under
  agent-replacement-warm-restart.

Honest scope (W68)
------------------

* All fits remain closed-form linear (W68-L-V13-NO-AUTOGRAD-CAP).
* The ninth target is *constructed*.
* The hidden-vs-agent-replacement probe is a finite synthetic grid.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.hidden_state_bridge_v12 requires numpy") from exc

from .hidden_state_bridge_v11 import (
    HiddenStateBridgeV11Projection,
    fit_hsb_v11_eight_target,
)
from .hidden_state_bridge_v5 import fit_hsb_v5_multi_target
from .tiny_substrate_v3 import (
    TinyV3SubstrateParams, _sha256_hex,
)


W68_HSB_V12_SCHEMA_VERSION: str = (
    "coordpy.hidden_state_bridge_v12.v1")
W68_DEFAULT_HSB_V12_RIDGE_LAMBDA: float = 0.05


@dataclasses.dataclass
class HiddenStateBridgeV12Projection:
    inner_v11: HiddenStateBridgeV11Projection
    seed_v12: int

    @classmethod
    def init_from_v11(
            cls, inner: HiddenStateBridgeV11Projection,
            *, seed_v12: int = 681000,
    ) -> "HiddenStateBridgeV12Projection":
        return cls(inner_v11=inner, seed_v12=int(seed_v12))

    @property
    def carrier_dim(self) -> int:
        return int(self.inner_v11.carrier_dim)

    @property
    def n_layers(self) -> int:
        return int(self.inner_v11.n_layers)

    @property
    def n_heads(self) -> int:
        return int(self.inner_v11.n_heads)

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W68_HSB_V12_SCHEMA_VERSION,
            "kind": "hsb_v12_projection",
            "inner_v11_cid": str(self.inner_v11.cid()),
            "seed_v12": int(self.seed_v12),
        })


@dataclasses.dataclass(frozen=True)
class HiddenStateBridgeV12FitReport:
    schema: str
    n_targets: int
    per_target_pre_residual: tuple[float, ...]
    per_target_post_residual: tuple[float, ...]
    agent_replacement_target_index: int
    agent_replacement_pre: float
    agent_replacement_post: float
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
            "agent_replacement_target_index": int(
                self.agent_replacement_target_index),
            "agent_replacement_pre": float(round(
                self.agent_replacement_pre, 12)),
            "agent_replacement_post": float(round(
                self.agent_replacement_post, 12)),
            "converged": bool(self.converged),
            "ridge_lambda": float(round(self.ridge_lambda, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hsb_v12_fit_report",
            "report": self.to_dict()})


def fit_hsb_v12_nine_target(
        *, params: TinyV3SubstrateParams,
        projection: HiddenStateBridgeV12Projection,
        train_carriers: Sequence[Sequence[float]],
        target_delta_logits_stack: Sequence[Sequence[float]],
        follow_up_token_ids: Sequence[int],
        agent_replacement_target_index: int = 8,
        n_directions: int = 3,
        ridge_lambda: float = W68_DEFAULT_HSB_V12_RIDGE_LAMBDA,
) -> tuple[
        HiddenStateBridgeV12Projection,
        HiddenStateBridgeV12FitReport]:
    """Nine-target stacked ridge: 8 V11 + 1 agent-replacement."""
    n_targets = int(len(target_delta_logits_stack))
    if n_targets < 1:
        raise ValueError("must provide >= 1 target")
    primary = list(target_delta_logits_stack[:8])
    while len(primary) < 8:
        primary.append(primary[0] if primary else [0.0])
    v11_fit, v11_report = fit_hsb_v11_eight_target(
        params=params, projection=projection.inner_v11,
        train_carriers=list(train_carriers),
        target_delta_logits_stack=primary,
        follow_up_token_ids=list(follow_up_token_ids),
        ridge_lambda=float(ridge_lambda))
    if n_targets >= int(agent_replacement_target_index) + 1:
        ar_target = list(target_delta_logits_stack[
            int(agent_replacement_target_index)])
    else:
        ar_target = list(target_delta_logits_stack[-1])
    inner_v5 = (
        v11_fit.inner_v10.inner_v9.inner_v8
        .inner_v7.inner_v6.inner_v5)
    v5_fit, v5_audit = fit_hsb_v5_multi_target(
        params=params, projection=inner_v5,
        train_carriers=list(train_carriers),
        target_delta_logits_stack=[ar_target],
        token_ids=list(follow_up_token_ids),
        ridge_lambda=float(ridge_lambda))
    new_inner_v6 = dataclasses.replace(
        v11_fit.inner_v10.inner_v9.inner_v8.inner_v7.inner_v6,
        inner_v5=v5_fit)
    new_inner_v7 = dataclasses.replace(
        v11_fit.inner_v10.inner_v9.inner_v8.inner_v7,
        inner_v6=new_inner_v6)
    new_inner_v8 = dataclasses.replace(
        v11_fit.inner_v10.inner_v9.inner_v8,
        inner_v7=new_inner_v7)
    new_inner_v9 = dataclasses.replace(
        v11_fit.inner_v10.inner_v9, inner_v8=new_inner_v8)
    new_v10 = dataclasses.replace(
        v11_fit.inner_v10, inner_v9=new_inner_v9)
    new_v11 = dataclasses.replace(v11_fit, inner_v10=new_v10)
    new_proj = dataclasses.replace(projection, inner_v11=new_v11)
    pre9 = float(v5_audit.pre_fit_residual)
    post9 = float(v5_audit.post_fit_residual)
    per_pre = (
        list(v11_report.per_target_pre_residual) + [pre9])
    per_post = (
        list(v11_report.per_target_post_residual) + [post9])
    converged = bool(
        all(po <= pr + 1e-9
            for pr, po in zip(per_pre[:8], per_post[:8]))
        and per_post[8] <= per_pre[8] + 1e-3)
    report = HiddenStateBridgeV12FitReport(
        schema=W68_HSB_V12_SCHEMA_VERSION,
        n_targets=int(n_targets),
        per_target_pre_residual=tuple(per_pre),
        per_target_post_residual=tuple(per_post),
        agent_replacement_target_index=int(
            agent_replacement_target_index),
        agent_replacement_pre=float(pre9),
        agent_replacement_post=float(post9),
        converged=bool(converged),
        ridge_lambda=float(ridge_lambda),
    )
    return new_proj, report


def probe_hsb_v12_hidden_vs_agent_replacement(
        *, n_layers: int, n_heads: int,
        hidden_residual_l2_grid: Sequence[Sequence[float]],
        agent_replacement_residual_l2_grid: Sequence[Sequence[float]],
        win_threshold: float = 0.1,
) -> dict[str, Any]:
    """Per-(L, H) hidden-vs-agent-replacement joint win-rate probe."""
    h = _np.asarray(hidden_residual_l2_grid, dtype=_np.float64)
    a = _np.asarray(
        agent_replacement_residual_l2_grid, dtype=_np.float64)
    if (h.shape != (int(n_layers), int(n_heads))
            or h.shape != a.shape):
        raise ValueError("grids must be (n_layers, n_heads)")
    wins = ((h + float(win_threshold)) < a).astype(_np.float64)
    return {
        "schema": W68_HSB_V12_SCHEMA_VERSION,
        "kind": "hidden_vs_agent_replacement",
        "per_layer_head_win_rate": [
            [float(round(float(x), 12)) for x in row]
            for row in wins.tolist()],
        "win_rate_mean": float(round(float(wins.mean()), 12)),
        "n_layers": int(n_layers),
        "n_heads": int(n_heads),
    }


def compute_hsb_v12_agent_replacement_margin(
        *,
        hidden_residual_l2: float,
        kv_residual_l2: float,
        prefix_residual_l2: float,
        replay_residual_l2: float,
        recover_residual_l2: float,
        branch_merge_residual_l2: float,
        agent_replacement_residual_l2: float,
) -> float:
    """Positive when hidden residual strictly beats every alternative
    including agent-replacement."""
    others = float(min(
        kv_residual_l2, prefix_residual_l2, replay_residual_l2,
        recover_residual_l2, branch_merge_residual_l2,
        agent_replacement_residual_l2))
    return float(round(
        max(0.0, others - float(hidden_residual_l2)), 12))


@dataclasses.dataclass(frozen=True)
class HSBV12Witness:
    schema: str
    projection_cid: str
    fit_report_cid: str
    hidden_vs_agent_replacement_probe_cid: str
    agent_replacement_margin: float
    hidden_vs_agent_replacement_mean: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "projection_cid": str(self.projection_cid),
            "fit_report_cid": str(self.fit_report_cid),
            "hidden_vs_agent_replacement_probe_cid": str(
                self.hidden_vs_agent_replacement_probe_cid),
            "agent_replacement_margin": float(round(
                self.agent_replacement_margin, 12)),
            "hidden_vs_agent_replacement_mean": float(round(
                self.hidden_vs_agent_replacement_mean, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hsb_v12_witness",
            "witness": self.to_dict()})


def emit_hsb_v12_witness(
        *, projection: HiddenStateBridgeV12Projection,
        fit_report: HiddenStateBridgeV12FitReport | None = None,
        hidden_vs_agent_replacement_probe: (
            dict[str, Any] | None) = None,
        agent_replacement_margin: float = 0.0,
        hidden_vs_agent_replacement_mean: float = 0.0,
) -> HSBV12Witness:
    return HSBV12Witness(
        schema=W68_HSB_V12_SCHEMA_VERSION,
        projection_cid=str(projection.cid()),
        fit_report_cid=(
            fit_report.cid() if fit_report is not None else ""),
        hidden_vs_agent_replacement_probe_cid=(
            _sha256_hex(hidden_vs_agent_replacement_probe)
            if hidden_vs_agent_replacement_probe is not None
            else ""),
        agent_replacement_margin=float(agent_replacement_margin),
        hidden_vs_agent_replacement_mean=float(
            hidden_vs_agent_replacement_mean),
    )


__all__ = [
    "W68_HSB_V12_SCHEMA_VERSION",
    "W68_DEFAULT_HSB_V12_RIDGE_LAMBDA",
    "HiddenStateBridgeV12Projection",
    "HiddenStateBridgeV12FitReport",
    "fit_hsb_v12_nine_target",
    "probe_hsb_v12_hidden_vs_agent_replacement",
    "compute_hsb_v12_agent_replacement_margin",
    "HSBV12Witness",
    "emit_hsb_v12_witness",
]
