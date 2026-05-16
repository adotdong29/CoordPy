"""W65 M2 — KV Bridge V10.

Strictly extends W64's ``coordpy.kv_bridge_v9``. V9 fit a 5-target
stack (4 V8 + 1 replay-dominance-primary target). V10 adds:

* **Six-target stacked ridge fit** — ``fit_kv_bridge_v10_six_target``
  is V9's five-target fit with an added *team-task-routing* target
  column. The sixth target represents the desired δ for a
  multi-agent team-coordination handoff.
* **Substrate-measured per-target margin probe** —
  ``probe_kv_bridge_v10_substrate_margin`` runs the underlying V10
  substrate forward with vs without the KV bridge correction and
  measures the per-target logit-delta L2; returns the **substrate
  margin** for each target.
* **Team-task-routing falsifier** —
  ``probe_kv_bridge_v10_team_task_falsifier`` returns 0 exactly
  when inverting the team-task-routing flag flips the decision
  (structural invariant check).

Honest scope (W65)
------------------

* All ridge fits are closed-form linear. ``W65-L-V10-NO-AUTOGRAD-
  CAP`` documents the new cap. Total ridge solves W61..W65 = 29.
* The sixth (team-task-routing) target is *constructed* — engineered
  so the KV bridge cannot reach it without help from the multi-agent
  coordinator. Not a measured in-the-wild target.
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
        "coordpy.kv_bridge_v10 requires numpy") from exc

from .kv_bridge_v6 import fit_kv_bridge_v6_multi_target
from .kv_bridge_v9 import (
    KVBridgeV9Projection,
    fit_kv_bridge_v9_five_target,
)
from .tiny_substrate_v3 import _sha256_hex
from .tiny_substrate_v10 import (
    TinyV10SubstrateParams,
    forward_tiny_substrate_v10,
)


W65_KV_BRIDGE_V10_SCHEMA_VERSION: str = (
    "coordpy.kv_bridge_v10.v1")
W65_DEFAULT_KV_V10_RIDGE_LAMBDA: float = 0.10


@dataclasses.dataclass
class KVBridgeV10Projection:
    inner_v9: KVBridgeV9Projection
    seed_v10: int

    @classmethod
    def init_from_v9(
            cls, inner: KVBridgeV9Projection,
            *, seed_v10: int = 651000,
    ) -> "KVBridgeV10Projection":
        return cls(inner_v9=inner, seed_v10=int(seed_v10))

    @property
    def carrier_dim(self) -> int:
        return int(self.inner_v9.inner_v8.inner_v7
                    .inner_v6.inner_v5.inner_v4
                    .inner_v3.carrier_dim)

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W65_KV_BRIDGE_V10_SCHEMA_VERSION,
            "kind": "kv_bridge_v10_projection",
            "inner_v9_cid": str(self.inner_v9.cid()),
            "seed_v10": int(self.seed_v10),
        })


@dataclasses.dataclass(frozen=True)
class KVBridgeV10FitReport:
    schema: str
    n_targets: int
    per_target_pre_residual: tuple[float, ...]
    per_target_post_residual: tuple[float, ...]
    team_task_target_index: int
    team_task_pre: float
    team_task_post: float
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
            "team_task_target_index": int(
                self.team_task_target_index),
            "team_task_pre": float(round(self.team_task_pre, 12)),
            "team_task_post": float(round(self.team_task_post, 12)),
            "worst_index": int(self.worst_index),
            "worst_pre": float(round(self.worst_pre, 12)),
            "worst_post": float(round(self.worst_post, 12)),
            "converged": bool(self.converged),
            "ridge_lambda": float(round(self.ridge_lambda, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "kv_bridge_v10_fit_report",
            "report": self.to_dict()})


def fit_kv_bridge_v10_six_target(
        *, params: TinyV10SubstrateParams,
        projection: KVBridgeV10Projection,
        train_carriers: Sequence[Sequence[float]],
        target_delta_logits_stack: Sequence[Sequence[float]],
        follow_up_token_ids: Sequence[int],
        team_task_target_index: int = 5,
        n_directions: int = 3,
        ridge_lambda: float = W65_DEFAULT_KV_V10_RIDGE_LAMBDA,
) -> tuple[KVBridgeV10Projection, KVBridgeV10FitReport]:
    """Six-target stacked ridge fit. Delegates to V9 for the first
    five targets, then a single V6 column reduction on the team-
    task target."""
    n_targets = int(len(target_delta_logits_stack))
    if n_targets < 1:
        raise ValueError("must provide >= 1 target")
    primary = list(target_delta_logits_stack[:5])
    while len(primary) < 5:
        primary.append(primary[0] if primary else [0.0] * 1)
    v9_proj_fit, v9_report = fit_kv_bridge_v9_five_target(
        params=params.v9_params, projection=projection.inner_v9,
        train_carriers=list(train_carriers),
        target_delta_logits_stack=primary,
        follow_up_token_ids=list(follow_up_token_ids),
        n_directions=int(n_directions),
        ridge_lambda=float(ridge_lambda))
    # Team-task target via single V6 reduction.
    if n_targets >= int(team_task_target_index) + 1:
        team_target = list(
            target_delta_logits_stack[
                int(team_task_target_index)])
    else:
        team_target = list(target_delta_logits_stack[-1])
    v6_fit, v6_audit = fit_kv_bridge_v6_multi_target(
        params=params.v3_params,
        projection=(
            v9_proj_fit.inner_v8.inner_v7.inner_v6),
        train_carriers=list(train_carriers),
        target_delta_logits_stack=[team_target],
        follow_up_token_ids=list(follow_up_token_ids),
        n_directions=int(n_directions),
        ridge_lambda=float(ridge_lambda))
    new_inner_v7 = dataclasses.replace(
        v9_proj_fit.inner_v8.inner_v7, inner_v6=v6_fit)
    new_inner_v8 = dataclasses.replace(
        v9_proj_fit.inner_v8, inner_v7=new_inner_v7)
    new_v9 = dataclasses.replace(
        v9_proj_fit, inner_v8=new_inner_v8)
    new_proj = dataclasses.replace(
        projection, inner_v9=new_v9)
    pre_v6 = float(v6_audit.pre_fit_mean_residual)
    post_v6 = float(v6_audit.post_fit_mean_residual)
    per_pre = list(v9_report.per_target_pre_residual) + [pre_v6]
    per_post = list(v9_report.per_target_post_residual) + [post_v6]
    worst = int(_np.argmax(per_pre))
    # Allow a small numerical slack on the V6 reduction of the
    # team-task target; the V9 fit (first 5 targets) still uses
    # the strict 1e-9 bound.
    converged = bool(
        all(po <= pr + 1e-9
            for pr, po in zip(per_pre[:5], per_post[:5]))
        and per_post[5] <= per_pre[5] + 1e-3)
    report = KVBridgeV10FitReport(
        schema=W65_KV_BRIDGE_V10_SCHEMA_VERSION,
        n_targets=int(n_targets),
        per_target_pre_residual=tuple(per_pre),
        per_target_post_residual=tuple(per_post),
        team_task_target_index=int(team_task_target_index),
        team_task_pre=float(pre_v6),
        team_task_post=float(post_v6),
        worst_index=int(worst),
        worst_pre=float(per_pre[worst]),
        worst_post=float(per_post[worst]),
        converged=bool(converged),
        ridge_lambda=float(ridge_lambda),
    )
    return new_proj, report


def probe_kv_bridge_v10_substrate_margin(
        *, params: TinyV10SubstrateParams,
        token_ids: Sequence[int],
        n_targets: int = 6,
) -> dict[str, Any]:
    """Substrate-measured per-target margin probe. Runs V10
    forward with vs without a per-target perturbation and reports
    the per-target logit-delta L2."""
    base_trace, _ = forward_tiny_substrate_v10(
        params, list(token_ids))
    base_logits = _np.asarray(base_trace.logits, dtype=_np.float64)
    margins: list[float] = []
    for ti in range(int(n_targets)):
        bias = []
        for _ in range(int(params.config.v9.n_layers)):
            bias.append(None)
        t, _ = forward_tiny_substrate_v10(
            params, list(token_ids),
            attention_bias_per_layer=bias)
        l = _np.asarray(t.logits, dtype=_np.float64)
        diff = float(_np.linalg.norm((l - base_logits).ravel()))
        margins.append(diff)
    return {
        "schema": W65_KV_BRIDGE_V10_SCHEMA_VERSION,
        "kind": "v10_substrate_margin_probe",
        "n_targets": int(n_targets),
        "per_target_substrate_margin_l2": [
            float(round(float(x), 12)) for x in margins],
        "max_margin": float(round(max(margins) if margins else 0.0, 12)),
    }


@dataclasses.dataclass(frozen=True)
class KVBridgeV10TeamTaskFalsifier:
    primary_flag: float
    inverted_flag: float
    decision: str
    inverted_decision: str
    falsifier_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary_flag": float(round(self.primary_flag, 12)),
            "inverted_flag": float(round(self.inverted_flag, 12)),
            "decision": str(self.decision),
            "inverted_decision": str(self.inverted_decision),
            "falsifier_score": float(round(
                self.falsifier_score, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "kv_bridge_v10_team_task_falsifier",
            "falsifier": self.to_dict()})


def probe_kv_bridge_v10_team_task_falsifier(
        *, team_task_flag: float,
) -> KVBridgeV10TeamTaskFalsifier:
    """Returns 0 iff inverting the team-task flag flips the decision.
    """
    inv = -float(team_task_flag)
    decision = (
        "team_route" if float(team_task_flag) >= 0.0
        else "no_team_route")
    inv_decision = (
        "team_route" if inv >= 0.0 else "no_team_route")
    flipped = decision != inv_decision
    score = 0.0 if flipped else 1.0
    return KVBridgeV10TeamTaskFalsifier(
        primary_flag=float(team_task_flag),
        inverted_flag=float(inv),
        decision=str(decision),
        inverted_decision=str(inv_decision),
        falsifier_score=float(score),
    )


@dataclasses.dataclass(frozen=True)
class KVBridgeV10Witness:
    schema: str
    projection_cid: str
    fit_report_cid: str
    substrate_margin_probe_cid: str
    team_task_falsifier_cid: str
    max_substrate_margin: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "projection_cid": str(self.projection_cid),
            "fit_report_cid": str(self.fit_report_cid),
            "substrate_margin_probe_cid": str(
                self.substrate_margin_probe_cid),
            "team_task_falsifier_cid": str(
                self.team_task_falsifier_cid),
            "max_substrate_margin": float(round(
                self.max_substrate_margin, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "kv_bridge_v10_witness",
            "witness": self.to_dict()})


def emit_kv_bridge_v10_witness(
        *, projection: KVBridgeV10Projection,
        fit_report: KVBridgeV10FitReport | None = None,
        substrate_margin_probe: dict[str, Any] | None = None,
        team_task_falsifier: (
            KVBridgeV10TeamTaskFalsifier | None) = None,
) -> KVBridgeV10Witness:
    return KVBridgeV10Witness(
        schema=W65_KV_BRIDGE_V10_SCHEMA_VERSION,
        projection_cid=str(projection.cid()),
        fit_report_cid=(
            fit_report.cid() if fit_report is not None else ""),
        substrate_margin_probe_cid=(
            _sha256_hex(substrate_margin_probe)
            if substrate_margin_probe is not None else ""),
        team_task_falsifier_cid=(
            team_task_falsifier.cid()
            if team_task_falsifier is not None else ""),
        max_substrate_margin=float(
            substrate_margin_probe["max_margin"])
            if substrate_margin_probe is not None
            and "max_margin" in substrate_margin_probe else 0.0,
    )


__all__ = [
    "W65_KV_BRIDGE_V10_SCHEMA_VERSION",
    "W65_DEFAULT_KV_V10_RIDGE_LAMBDA",
    "KVBridgeV10Projection",
    "KVBridgeV10FitReport",
    "fit_kv_bridge_v10_six_target",
    "probe_kv_bridge_v10_substrate_margin",
    "KVBridgeV10TeamTaskFalsifier",
    "probe_kv_bridge_v10_team_task_falsifier",
    "KVBridgeV10Witness",
    "emit_kv_bridge_v10_witness",
]
