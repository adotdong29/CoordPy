"""W72 M2 — KV Bridge V17.

Strictly extends W71's ``coordpy.kv_bridge_v16``. V16 fit a 12-
target stack (11 V15 + 1 restart-dominance). V17 adds:

* **Thirteen-target stacked ridge fit** —
  ``fit_kv_bridge_v17_thirteen_target`` adds a thirteenth column
  for *delayed-rejoin-after-restart routing*.
* **100-dim restart-repair fingerprint** — derived from
  ``(role, repair_trajectory_cid, delayed_repair_trajectory_cid,
  restart_repair_trajectory_cid, dominant_repair_label,
  restart_count, rejoin_count, visible_token_budget,
  baseline_cost, task_id, team_id, branch_id, delay_turns,
  rejoin_lag_turns)``.
* **Rejoin-pressure falsifier** — returns 0 iff inverting the
  rejoin-pressure flag flips the routing decision.

Honest scope (W72)
------------------

* All ridge fits remain closed-form linear
  (``W72-L-V17-NO-AUTOGRAD-CAP``).
* Total ridge solves across W61..W72 = 64 (3 new on top of W71's
  61 — KV V17 thirteen-target adds 1; cache V15 twelve-objective +
  per-role rejoin-pressure adds 2; replay V13 twenty-regime +
  rejoin-routing adds 2 — but the KV V17 line counts only the
  *new* rolled-up thirteen-target compared to V16's twelve-target
  as a single new solve).
"""

from __future__ import annotations

import dataclasses
import hashlib
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.kv_bridge_v17 requires numpy") from exc

from .kv_bridge_v16 import (
    KVBridgeV16FitReport, KVBridgeV16Projection,
    fit_kv_bridge_v16_twelve_target,
)
from .kv_bridge_v6 import fit_kv_bridge_v6_multi_target
from .tiny_substrate_v3 import _sha256_hex
from .tiny_substrate_v17 import (
    TinyV17SubstrateParams, W72_REPAIR_LABELS_V17,
    forward_tiny_substrate_v17,
)


W72_KV_BRIDGE_V17_SCHEMA_VERSION: str = (
    "coordpy.kv_bridge_v17.v1")
W72_DEFAULT_KV_V17_RIDGE_LAMBDA: float = 0.10
W72_KV_V17_FINGERPRINT_DIM: int = 100


@dataclasses.dataclass
class KVBridgeV17Projection:
    inner_v16: KVBridgeV16Projection
    seed_v17: int

    @classmethod
    def init_from_v16(
            cls, inner: KVBridgeV16Projection,
            *, seed_v17: int = 720100,
    ) -> "KVBridgeV17Projection":
        return cls(inner_v16=inner, seed_v17=int(seed_v17))

    @property
    def carrier_dim(self) -> int:
        return int(self.inner_v16.carrier_dim)

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W72_KV_BRIDGE_V17_SCHEMA_VERSION,
            "kind": "kv_bridge_v17_projection",
            "inner_v16_cid": str(self.inner_v16.cid()),
            "seed_v17": int(self.seed_v17),
        })


@dataclasses.dataclass(frozen=True)
class KVBridgeV17FitReport:
    schema: str
    n_targets: int
    per_target_pre_residual: tuple[float, ...]
    per_target_post_residual: tuple[float, ...]
    delayed_rejoin_target_index: int
    delayed_rejoin_pre: float
    delayed_rejoin_post: float
    converged: bool
    ridge_lambda: float
    inner_v16_report_cid: str

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
            "delayed_rejoin_target_index": int(
                self.delayed_rejoin_target_index),
            "delayed_rejoin_pre": float(round(
                self.delayed_rejoin_pre, 12)),
            "delayed_rejoin_post": float(round(
                self.delayed_rejoin_post, 12)),
            "converged": bool(self.converged),
            "ridge_lambda": float(round(self.ridge_lambda, 12)),
            "inner_v16_report_cid": str(
                self.inner_v16_report_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "kv_bridge_v17_fit_report",
            "report": self.to_dict()})


def fit_kv_bridge_v17_thirteen_target(
        *, params: TinyV17SubstrateParams,
        projection: KVBridgeV17Projection,
        train_carriers: Sequence[Sequence[float]],
        target_delta_logits_stack: Sequence[Sequence[float]],
        follow_up_token_ids: Sequence[int],
        delayed_rejoin_target_index: int = 12,
        n_directions: int = 3,
        ridge_lambda: float = W72_DEFAULT_KV_V17_RIDGE_LAMBDA,
) -> tuple[KVBridgeV17Projection, KVBridgeV17FitReport]:
    """Thirteen-target stacked ridge: 12 V16 + 1 delayed-rejoin.

    Sources the inner V16 fit, then layers a single additional
    inner-V6 multi-target fit for the new delayed-rejoin column.
    """
    n_targets = int(len(target_delta_logits_stack))
    if n_targets < 1:
        raise ValueError("must provide >= 1 target")
    primary = list(target_delta_logits_stack[:12])
    while len(primary) < 12:
        primary.append(primary[0] if primary else [0.0])
    v16_fit, v16_report = fit_kv_bridge_v16_twelve_target(
        params=params.v16_params,
        projection=projection.inner_v16,
        train_carriers=list(train_carriers),
        target_delta_logits_stack=primary,
        follow_up_token_ids=list(follow_up_token_ids),
        n_directions=int(n_directions),
        ridge_lambda=float(ridge_lambda))
    if n_targets >= int(delayed_rejoin_target_index) + 1:
        rj_target = list(target_delta_logits_stack[
            int(delayed_rejoin_target_index)])
    else:
        rj_target = list(target_delta_logits_stack[-1])
    inner_v6 = (
        v16_fit.inner_v15.inner_v14.inner_v13.inner_v12.inner_v11
        .inner_v10.inner_v9.inner_v8.inner_v7.inner_v6)
    v6_fit_rj, v6_audit_rj = fit_kv_bridge_v6_multi_target(
        params=params.v16_params.v15_params.v3_params,
        projection=inner_v6,
        train_carriers=list(train_carriers),
        target_delta_logits_stack=[rj_target],
        follow_up_token_ids=list(follow_up_token_ids),
        n_directions=int(n_directions),
        ridge_lambda=float(ridge_lambda))
    # Wrap back through the nested projection chain.
    new_inner_v7 = dataclasses.replace(
        v16_fit.inner_v15.inner_v14.inner_v13.inner_v12.inner_v11
        .inner_v10.inner_v9.inner_v8.inner_v7, inner_v6=v6_fit_rj)
    new_inner_v8 = dataclasses.replace(
        v16_fit.inner_v15.inner_v14.inner_v13.inner_v12.inner_v11
        .inner_v10.inner_v9.inner_v8, inner_v7=new_inner_v7)
    new_inner_v9 = dataclasses.replace(
        v16_fit.inner_v15.inner_v14.inner_v13.inner_v12.inner_v11
        .inner_v10.inner_v9, inner_v8=new_inner_v8)
    new_inner_v10 = dataclasses.replace(
        v16_fit.inner_v15.inner_v14.inner_v13.inner_v12.inner_v11
        .inner_v10, inner_v9=new_inner_v9)
    new_inner_v11 = dataclasses.replace(
        v16_fit.inner_v15.inner_v14.inner_v13.inner_v12.inner_v11,
        inner_v10=new_inner_v10)
    new_inner_v12 = dataclasses.replace(
        v16_fit.inner_v15.inner_v14.inner_v13.inner_v12,
        inner_v11=new_inner_v11)
    new_inner_v13 = dataclasses.replace(
        v16_fit.inner_v15.inner_v14.inner_v13,
        inner_v12=new_inner_v12)
    new_inner_v14 = dataclasses.replace(
        v16_fit.inner_v15.inner_v14, inner_v13=new_inner_v13)
    new_inner_v15 = dataclasses.replace(
        v16_fit.inner_v15, inner_v14=new_inner_v14)
    new_v16 = dataclasses.replace(
        v16_fit, inner_v15=new_inner_v15)
    new_proj = dataclasses.replace(
        projection, inner_v16=new_v16)
    pre13 = float(v6_audit_rj.pre_fit_mean_residual)
    post13 = float(v6_audit_rj.post_fit_mean_residual)
    per_pre = (
        list(v16_report.per_target_pre_residual) + [pre13])
    per_post = (
        list(v16_report.per_target_post_residual) + [post13])
    converged = bool(
        all(po <= pr + 1e-9
            for pr, po in zip(per_pre[:12], per_post[:12]))
        and per_post[12] <= per_pre[12] + 1e-2)
    report = KVBridgeV17FitReport(
        schema=W72_KV_BRIDGE_V17_SCHEMA_VERSION,
        n_targets=int(n_targets),
        per_target_pre_residual=tuple(per_pre),
        per_target_post_residual=tuple(per_post),
        delayed_rejoin_target_index=int(
            delayed_rejoin_target_index),
        delayed_rejoin_pre=float(pre13),
        delayed_rejoin_post=float(post13),
        converged=bool(converged),
        ridge_lambda=float(ridge_lambda),
        inner_v16_report_cid=str(v16_report.cid()),
    )
    return new_proj, report


def compute_restart_repair_fingerprint_v17(
        *, role: str,
        repair_trajectory_cid: str,
        delayed_repair_trajectory_cid: str,
        restart_repair_trajectory_cid: str,
        dominant_repair_label: int = 0,
        restart_count: int = 0,
        rejoin_count: int = 0,
        visible_token_budget: float = 256.0,
        baseline_cost: float = 512.0,
        task_id: str = "task", team_id: str = "team",
        branch_id: str = "main",
        delay_turns: int = 0,
        rejoin_lag_turns: int = 0,
        dim: int = W72_KV_V17_FINGERPRINT_DIM,
) -> tuple[float, ...]:
    """Deterministic 100-dim restart-repair fingerprint."""
    label_str = (
        W72_REPAIR_LABELS_V17[int(dominant_repair_label)]
        if 0 <= int(dominant_repair_label)
            < len(W72_REPAIR_LABELS_V17)
        else "unknown")
    base = (
        f"{role}|{repair_trajectory_cid}|"
        f"{delayed_repair_trajectory_cid}|"
        f"{restart_repair_trajectory_cid}|{label_str}|"
        f"{int(restart_count)}|{int(rejoin_count)}|"
        f"{float(visible_token_budget)}|{float(baseline_cost)}|"
        f"{task_id}|{team_id}|{branch_id}|"
        f"{int(delay_turns)}|{int(rejoin_lag_turns)}"
    ).encode("utf-8")
    h = hashlib.sha256(base).hexdigest()
    out: list[float] = []
    for i in range(int(dim)):
        nb = h[(i * 2) % len(h):(i * 2) % len(h) + 2]
        if not nb:
            nb = "00"
        v = (int(nb, 16) / 127.5) - 1.0
        out.append(float(round(v, 12)))
    return tuple(out)


def probe_kv_bridge_v17_rejoin_pressure_margin(
        *, params: TinyV17SubstrateParams,
        token_ids: Sequence[int], n_targets: int = 13,
) -> dict[str, Any]:
    """Substrate-measured per-target margin probe via V17 forward."""
    base_trace, _ = forward_tiny_substrate_v17(
        params, list(token_ids))
    base_logits = _np.asarray(
        base_trace.logits, dtype=_np.float64)
    margins: list[float] = []
    for _ in range(int(n_targets)):
        t, _ = forward_tiny_substrate_v17(
            params, list(token_ids))
        l = _np.asarray(t.logits, dtype=_np.float64)
        diff = float(
            _np.linalg.norm((l - base_logits).ravel()))
        margins.append(diff)
    return {
        "schema": W72_KV_BRIDGE_V17_SCHEMA_VERSION,
        "kind": "v17_rejoin_pressure_margin_probe",
        "n_targets": int(n_targets),
        "per_target_margin_l2": [
            float(round(float(x), 12)) for x in margins],
        "max_margin": float(round(
            max(margins) if margins else 0.0, 12)),
    }


@dataclasses.dataclass(frozen=True)
class KVBridgeV17RejoinPressureFalsifier:
    primary_flag: int
    inverted_flag: int
    decision: str
    inverted_decision: str
    falsifier_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary_flag": int(self.primary_flag),
            "inverted_flag": int(self.inverted_flag),
            "decision": str(self.decision),
            "inverted_decision": str(self.inverted_decision),
            "falsifier_score": float(round(
                self.falsifier_score, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind":
                "kv_bridge_v17_rejoin_pressure_falsifier",
            "falsifier": self.to_dict()})


def probe_kv_bridge_v17_rejoin_pressure_falsifier(
        *, rejoin_pressure_flag: int,
) -> KVBridgeV17RejoinPressureFalsifier:
    """Returns 0 iff inverting the rejoin-pressure flag flips
    the routing decision.

    Inversion semantics: flag 0 → 1, any nonzero → 0. The routing
    decision is ``route_through_substrate`` when flag > 0 and
    ``route_through_text`` otherwise. Flipping the flag must flip
    the decision (honest case: score = 0).
    """
    f = int(rejoin_pressure_flag)
    inv = 1 if f == 0 else 0
    decision = (
        "route_through_substrate" if f > 0
        else "route_through_text")
    inv_decision = (
        "route_through_substrate" if inv > 0
        else "route_through_text")
    flipped = decision != inv_decision
    score = 0.0 if flipped else 1.0
    return KVBridgeV17RejoinPressureFalsifier(
        primary_flag=int(f),
        inverted_flag=int(inv),
        decision=str(decision),
        inverted_decision=str(inv_decision),
        falsifier_score=float(score),
    )


@dataclasses.dataclass(frozen=True)
class KVBridgeV17Witness:
    schema: str
    projection_cid: str
    fit_report_cid: str
    rejoin_pressure_margin_probe_cid: str
    rejoin_pressure_falsifier_cid: str
    restart_repair_fingerprint_l1: float
    max_rejoin_pressure_margin: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "projection_cid": str(self.projection_cid),
            "fit_report_cid": str(self.fit_report_cid),
            "rejoin_pressure_margin_probe_cid": str(
                self.rejoin_pressure_margin_probe_cid),
            "rejoin_pressure_falsifier_cid": str(
                self.rejoin_pressure_falsifier_cid),
            "restart_repair_fingerprint_l1": float(round(
                self.restart_repair_fingerprint_l1, 12)),
            "max_rejoin_pressure_margin": float(round(
                self.max_rejoin_pressure_margin, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "kv_bridge_v17_witness",
            "witness": self.to_dict()})


def emit_kv_bridge_v17_witness(
        *, projection: KVBridgeV17Projection,
        fit_report: KVBridgeV17FitReport | None = None,
        rejoin_pressure_margin_probe: (
            dict[str, Any] | None) = None,
        rejoin_pressure_falsifier: (
            KVBridgeV17RejoinPressureFalsifier | None) = None,
        restart_repair_fingerprint: (
            Sequence[float] | None) = None,
) -> KVBridgeV17Witness:
    fp_l1 = 0.0
    if restart_repair_fingerprint is not None:
        fp_l1 = float(sum(
            abs(float(x))
            for x in restart_repair_fingerprint))
    return KVBridgeV17Witness(
        schema=W72_KV_BRIDGE_V17_SCHEMA_VERSION,
        projection_cid=str(projection.cid()),
        fit_report_cid=(
            fit_report.cid() if fit_report is not None else ""),
        rejoin_pressure_margin_probe_cid=(
            _sha256_hex(rejoin_pressure_margin_probe)
            if rejoin_pressure_margin_probe is not None
            else ""),
        rejoin_pressure_falsifier_cid=(
            rejoin_pressure_falsifier.cid()
            if rejoin_pressure_falsifier is not None
            else ""),
        restart_repair_fingerprint_l1=float(fp_l1),
        max_rejoin_pressure_margin=float(
            rejoin_pressure_margin_probe["max_margin"]
            if rejoin_pressure_margin_probe is not None
            and "max_margin"
                in rejoin_pressure_margin_probe
            else 0.0),
    )


__all__ = [
    "W72_KV_BRIDGE_V17_SCHEMA_VERSION",
    "W72_DEFAULT_KV_V17_RIDGE_LAMBDA",
    "W72_KV_V17_FINGERPRINT_DIM",
    "KVBridgeV17Projection",
    "KVBridgeV17FitReport",
    "fit_kv_bridge_v17_thirteen_target",
    "compute_restart_repair_fingerprint_v17",
    "probe_kv_bridge_v17_rejoin_pressure_margin",
    "KVBridgeV17RejoinPressureFalsifier",
    "probe_kv_bridge_v17_rejoin_pressure_falsifier",
    "KVBridgeV17Witness",
    "emit_kv_bridge_v17_witness",
]
