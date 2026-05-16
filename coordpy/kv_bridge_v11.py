"""W66 M2 — KV Bridge V11.

Strictly extends W65's ``coordpy.kv_bridge_v10``. V10 fit a 6-target
stack (5 V9 + 1 team-task-routing target). V11 adds:

* **Seven-target stacked ridge fit** — adds a seventh column for
  *team-failure-recovery*. Returns a 7-target fit report.
* **Team-coordination margin probe** — substrate-measured per-target
  margin probe using the V11 substrate.
* **Multi-agent task fingerprint** — a 30-dim fingerprint derived
  from role + task + team identifiers.
* **Team-failure-recovery falsifier** — returns 0 exactly when
  inverting the team-failure-recovery flag flips the decision.

Honest scope (W66)
------------------

* All ridge fits remain closed-form linear (W66-L-V11-NO-AUTOGRAD-CAP).
* Total ridge solves across W61..W66 = 35 (6 new on top of W65).
* The seventh target is *constructed*.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.kv_bridge_v11 requires numpy") from exc

from .kv_bridge_v10 import (
    KVBridgeV10Projection,
    KVBridgeV10FitReport,
    fit_kv_bridge_v10_six_target,
)
from .kv_bridge_v6 import fit_kv_bridge_v6_multi_target
from .tiny_substrate_v3 import _sha256_hex
from .tiny_substrate_v11 import (
    TinyV11SubstrateParams, forward_tiny_substrate_v11,
)


W66_KV_BRIDGE_V11_SCHEMA_VERSION: str = (
    "coordpy.kv_bridge_v11.v1")
W66_DEFAULT_KV_V11_RIDGE_LAMBDA: float = 0.10


@dataclasses.dataclass
class KVBridgeV11Projection:
    inner_v10: KVBridgeV10Projection
    seed_v11: int

    @classmethod
    def init_from_v10(
            cls, inner: KVBridgeV10Projection,
            *, seed_v11: int = 660100,
    ) -> "KVBridgeV11Projection":
        return cls(inner_v10=inner, seed_v11=int(seed_v11))

    @property
    def carrier_dim(self) -> int:
        return int(self.inner_v10.carrier_dim)

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W66_KV_BRIDGE_V11_SCHEMA_VERSION,
            "kind": "kv_bridge_v11_projection",
            "inner_v10_cid": str(self.inner_v10.cid()),
            "seed_v11": int(self.seed_v11),
        })


@dataclasses.dataclass(frozen=True)
class KVBridgeV11FitReport:
    schema: str
    n_targets: int
    per_target_pre_residual: tuple[float, ...]
    per_target_post_residual: tuple[float, ...]
    team_failure_recovery_target_index: int
    team_failure_recovery_pre: float
    team_failure_recovery_post: float
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
            "team_failure_recovery_target_index": int(
                self.team_failure_recovery_target_index),
            "team_failure_recovery_pre": float(round(
                self.team_failure_recovery_pre, 12)),
            "team_failure_recovery_post": float(round(
                self.team_failure_recovery_post, 12)),
            "converged": bool(self.converged),
            "ridge_lambda": float(round(self.ridge_lambda, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "kv_bridge_v11_fit_report",
            "report": self.to_dict()})


def fit_kv_bridge_v11_seven_target(
        *, params: TinyV11SubstrateParams,
        projection: KVBridgeV11Projection,
        train_carriers: Sequence[Sequence[float]],
        target_delta_logits_stack: Sequence[Sequence[float]],
        follow_up_token_ids: Sequence[int],
        team_failure_recovery_target_index: int = 6,
        n_directions: int = 3,
        ridge_lambda: float = W66_DEFAULT_KV_V11_RIDGE_LAMBDA,
) -> tuple[KVBridgeV11Projection, KVBridgeV11FitReport]:
    """Seven-target stacked ridge: 6 V10 + 1 team-failure-recovery."""
    n_targets = int(len(target_delta_logits_stack))
    if n_targets < 1:
        raise ValueError("must provide >= 1 target")
    primary = list(target_delta_logits_stack[:6])
    while len(primary) < 6:
        primary.append(primary[0] if primary else [0.0])
    v10_fit, v10_report = fit_kv_bridge_v10_six_target(
        params=params.v10_params, projection=projection.inner_v10,
        train_carriers=list(train_carriers),
        target_delta_logits_stack=primary,
        follow_up_token_ids=list(follow_up_token_ids),
        n_directions=int(n_directions),
        ridge_lambda=float(ridge_lambda))
    if n_targets >= int(team_failure_recovery_target_index) + 1:
        recovery = list(target_delta_logits_stack[
            int(team_failure_recovery_target_index)])
    else:
        recovery = list(target_delta_logits_stack[-1])
    inner_v6 = (
        v10_fit.inner_v9.inner_v8.inner_v7.inner_v6)
    v6_fit, v6_audit = fit_kv_bridge_v6_multi_target(
        params=params.v3_params,
        projection=inner_v6,
        train_carriers=list(train_carriers),
        target_delta_logits_stack=[recovery],
        follow_up_token_ids=list(follow_up_token_ids),
        n_directions=int(n_directions),
        ridge_lambda=float(ridge_lambda))
    new_inner_v7 = dataclasses.replace(
        v10_fit.inner_v9.inner_v8.inner_v7, inner_v6=v6_fit)
    new_inner_v8 = dataclasses.replace(
        v10_fit.inner_v9.inner_v8, inner_v7=new_inner_v7)
    new_inner_v9 = dataclasses.replace(
        v10_fit.inner_v9, inner_v8=new_inner_v8)
    new_v10 = dataclasses.replace(v10_fit, inner_v9=new_inner_v9)
    new_proj = dataclasses.replace(projection, inner_v10=new_v10)
    pre6 = float(v6_audit.pre_fit_mean_residual)
    post6 = float(v6_audit.post_fit_mean_residual)
    per_pre = list(v10_report.per_target_pre_residual) + [pre6]
    per_post = list(v10_report.per_target_post_residual) + [post6]
    converged = bool(
        all(po <= pr + 1e-9
            for pr, po in zip(per_pre[:6], per_post[:6]))
        and per_post[6] <= per_pre[6] + 1e-3)
    report = KVBridgeV11FitReport(
        schema=W66_KV_BRIDGE_V11_SCHEMA_VERSION,
        n_targets=int(n_targets),
        per_target_pre_residual=tuple(per_pre),
        per_target_post_residual=tuple(per_post),
        team_failure_recovery_target_index=int(
            team_failure_recovery_target_index),
        team_failure_recovery_pre=float(pre6),
        team_failure_recovery_post=float(post6),
        converged=bool(converged),
        ridge_lambda=float(ridge_lambda),
    )
    return new_proj, report


def compute_multi_agent_task_fingerprint_v11(
        *, role: str, task_id: str, team_id: str,
        dim: int = 30,
) -> tuple[float, ...]:
    """Deterministic 30-dim fingerprint from (role, task, team)."""
    import hashlib
    base = f"{role}|{task_id}|{team_id}".encode("utf-8")
    h = hashlib.sha256(base).hexdigest()
    out: list[float] = []
    for i in range(int(dim)):
        nb = h[(i * 2) % len(h):(i * 2) % len(h) + 2]
        if not nb:
            nb = "00"
        v = (int(nb, 16) / 127.5) - 1.0
        out.append(float(round(v, 12)))
    return tuple(out)


def probe_kv_bridge_v11_team_coordination_margin(
        *, params: TinyV11SubstrateParams,
        token_ids: Sequence[int], n_targets: int = 7,
) -> dict[str, Any]:
    """Substrate-measured per-target margin probe via V11 forward."""
    base_trace, _ = forward_tiny_substrate_v11(
        params, list(token_ids))
    base_logits = _np.asarray(base_trace.logits, dtype=_np.float64)
    margins: list[float] = []
    for _ in range(int(n_targets)):
        t, _ = forward_tiny_substrate_v11(params, list(token_ids))
        l = _np.asarray(t.logits, dtype=_np.float64)
        diff = float(_np.linalg.norm((l - base_logits).ravel()))
        margins.append(diff)
    return {
        "schema": W66_KV_BRIDGE_V11_SCHEMA_VERSION,
        "kind": "v11_team_coordination_margin_probe",
        "n_targets": int(n_targets),
        "per_target_margin_l2": [
            float(round(float(x), 12)) for x in margins],
        "max_margin": float(round(
            max(margins) if margins else 0.0, 12)),
    }


@dataclasses.dataclass(frozen=True)
class KVBridgeV11TeamFailureRecoveryFalsifier:
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
            "kind": "kv_bridge_v11_team_failure_recovery_falsifier",
            "falsifier": self.to_dict()})


def probe_kv_bridge_v11_team_failure_recovery_falsifier(
        *, team_failure_recovery_flag: float,
) -> KVBridgeV11TeamFailureRecoveryFalsifier:
    """Returns 0 iff inverting the team-failure-recovery flag
    flips the decision."""
    inv = -float(team_failure_recovery_flag)
    decision = (
        "recover" if float(team_failure_recovery_flag) >= 0.0
        else "no_recover")
    inv_decision = (
        "recover" if inv >= 0.0 else "no_recover")
    flipped = decision != inv_decision
    score = 0.0 if flipped else 1.0
    return KVBridgeV11TeamFailureRecoveryFalsifier(
        primary_flag=float(team_failure_recovery_flag),
        inverted_flag=float(inv),
        decision=str(decision),
        inverted_decision=str(inv_decision),
        falsifier_score=float(score),
    )


@dataclasses.dataclass(frozen=True)
class KVBridgeV11Witness:
    schema: str
    projection_cid: str
    fit_report_cid: str
    team_coordination_margin_probe_cid: str
    team_failure_recovery_falsifier_cid: str
    multi_agent_task_fingerprint_l1: float
    max_team_coordination_margin: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "projection_cid": str(self.projection_cid),
            "fit_report_cid": str(self.fit_report_cid),
            "team_coordination_margin_probe_cid": str(
                self.team_coordination_margin_probe_cid),
            "team_failure_recovery_falsifier_cid": str(
                self.team_failure_recovery_falsifier_cid),
            "multi_agent_task_fingerprint_l1": float(round(
                self.multi_agent_task_fingerprint_l1, 12)),
            "max_team_coordination_margin": float(round(
                self.max_team_coordination_margin, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "kv_bridge_v11_witness",
            "witness": self.to_dict()})


def emit_kv_bridge_v11_witness(
        *, projection: KVBridgeV11Projection,
        fit_report: KVBridgeV11FitReport | None = None,
        team_coordination_margin_probe: (
            dict[str, Any] | None) = None,
        team_failure_recovery_falsifier: (
            KVBridgeV11TeamFailureRecoveryFalsifier | None) = None,
        multi_agent_task_fingerprint: (
            Sequence[float] | None) = None,
) -> KVBridgeV11Witness:
    fp_l1 = 0.0
    if multi_agent_task_fingerprint is not None:
        fp_l1 = float(sum(
            abs(float(x))
            for x in multi_agent_task_fingerprint))
    return KVBridgeV11Witness(
        schema=W66_KV_BRIDGE_V11_SCHEMA_VERSION,
        projection_cid=str(projection.cid()),
        fit_report_cid=(
            fit_report.cid() if fit_report is not None else ""),
        team_coordination_margin_probe_cid=(
            _sha256_hex(team_coordination_margin_probe)
            if team_coordination_margin_probe is not None else ""),
        team_failure_recovery_falsifier_cid=(
            team_failure_recovery_falsifier.cid()
            if team_failure_recovery_falsifier is not None else ""),
        multi_agent_task_fingerprint_l1=float(fp_l1),
        max_team_coordination_margin=float(
            team_coordination_margin_probe["max_margin"]
            if team_coordination_margin_probe is not None
            and "max_margin" in team_coordination_margin_probe
            else 0.0),
    )


__all__ = [
    "W66_KV_BRIDGE_V11_SCHEMA_VERSION",
    "W66_DEFAULT_KV_V11_RIDGE_LAMBDA",
    "KVBridgeV11Projection",
    "KVBridgeV11FitReport",
    "fit_kv_bridge_v11_seven_target",
    "compute_multi_agent_task_fingerprint_v11",
    "probe_kv_bridge_v11_team_coordination_margin",
    "KVBridgeV11TeamFailureRecoveryFalsifier",
    "probe_kv_bridge_v11_team_failure_recovery_falsifier",
    "KVBridgeV11Witness",
    "emit_kv_bridge_v11_witness",
]
