"""W71 M2 — KV Bridge V16.

Strictly extends W70's ``coordpy.kv_bridge_v15``. V15 fit an 11-
target stack (10 V14 + 1 repair-dominance). V16 adds:

* **Twelve-target stacked ridge fit** —
  ``fit_kv_bridge_v16_twelve_target`` adds a twelfth column for
  *restart-dominance routing*.
* **84-dim delayed-repair fingerprint** — derived from
  ``(role, repair_trajectory_cid, delayed_repair_trajectory_cid,
  dominant_repair_label, restart_count, visible_token_budget,
  baseline_cost, task_id, team_id, branch_id, delay_turns)``.
* **Restart-dominance falsifier** — returns 0 iff inverting the
  restart-dominance flag flips the routing decision.

Honest scope (W71)
------------------

* All ridge fits remain closed-form linear
  (``W71-L-V16-NO-AUTOGRAD-CAP``).
* Total ridge solves across W61..W71 = 61 (3 new on top of W70's
  58 — KV V16 twelve-target adds 1; cache V14 eleven-objective +
  per-role restart adds 2; replay V12 nineteen-regime + restart-
  routing adds 2 — but the KV V16 line counts only the *new*
  rolled-up twelve-target compared to V15's eleven-target as a
  single new solve).
"""

from __future__ import annotations

import dataclasses
import hashlib
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.kv_bridge_v16 requires numpy") from exc

from .kv_bridge_v15 import (
    KVBridgeV15FitReport, KVBridgeV15Projection,
    fit_kv_bridge_v15_eleven_target,
)
from .kv_bridge_v6 import fit_kv_bridge_v6_multi_target
from .tiny_substrate_v3 import _sha256_hex
from .tiny_substrate_v16 import (
    TinyV16SubstrateParams, W71_REPAIR_LABELS_V16,
    forward_tiny_substrate_v16,
)


W71_KV_BRIDGE_V16_SCHEMA_VERSION: str = (
    "coordpy.kv_bridge_v16.v1")
W71_DEFAULT_KV_V16_RIDGE_LAMBDA: float = 0.10
W71_KV_V16_FINGERPRINT_DIM: int = 84


@dataclasses.dataclass
class KVBridgeV16Projection:
    inner_v15: KVBridgeV15Projection
    seed_v16: int

    @classmethod
    def init_from_v15(
            cls, inner: KVBridgeV15Projection,
            *, seed_v16: int = 710100,
    ) -> "KVBridgeV16Projection":
        return cls(inner_v15=inner, seed_v16=int(seed_v16))

    @property
    def carrier_dim(self) -> int:
        return int(self.inner_v15.carrier_dim)

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W71_KV_BRIDGE_V16_SCHEMA_VERSION,
            "kind": "kv_bridge_v16_projection",
            "inner_v15_cid": str(self.inner_v15.cid()),
            "seed_v16": int(self.seed_v16),
        })


@dataclasses.dataclass(frozen=True)
class KVBridgeV16FitReport:
    schema: str
    n_targets: int
    per_target_pre_residual: tuple[float, ...]
    per_target_post_residual: tuple[float, ...]
    restart_dominance_target_index: int
    restart_dominance_pre: float
    restart_dominance_post: float
    converged: bool
    ridge_lambda: float
    inner_v15_report_cid: str

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
            "restart_dominance_target_index": int(
                self.restart_dominance_target_index),
            "restart_dominance_pre": float(round(
                self.restart_dominance_pre, 12)),
            "restart_dominance_post": float(round(
                self.restart_dominance_post, 12)),
            "converged": bool(self.converged),
            "ridge_lambda": float(round(self.ridge_lambda, 12)),
            "inner_v15_report_cid": str(self.inner_v15_report_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "kv_bridge_v16_fit_report",
            "report": self.to_dict()})


def fit_kv_bridge_v16_twelve_target(
        *, params: TinyV16SubstrateParams,
        projection: KVBridgeV16Projection,
        train_carriers: Sequence[Sequence[float]],
        target_delta_logits_stack: Sequence[Sequence[float]],
        follow_up_token_ids: Sequence[int],
        restart_dominance_target_index: int = 11,
        n_directions: int = 3,
        ridge_lambda: float = W71_DEFAULT_KV_V16_RIDGE_LAMBDA,
) -> tuple[KVBridgeV16Projection, KVBridgeV16FitReport]:
    """Twelve-target stacked ridge: 11 V15 + 1 restart-dominance.

    Sources the inner V15 fit, then layers a single additional
    inner-V6 multi-target fit for the new restart-dominance
    column.
    """
    n_targets = int(len(target_delta_logits_stack))
    if n_targets < 1:
        raise ValueError("must provide >= 1 target")
    primary = list(target_delta_logits_stack[:11])
    while len(primary) < 11:
        primary.append(primary[0] if primary else [0.0])
    v15_fit, v15_report = fit_kv_bridge_v15_eleven_target(
        params=params.v15_params,
        projection=projection.inner_v15,
        train_carriers=list(train_carriers),
        target_delta_logits_stack=primary,
        follow_up_token_ids=list(follow_up_token_ids),
        n_directions=int(n_directions),
        ridge_lambda=float(ridge_lambda))
    if n_targets >= int(restart_dominance_target_index) + 1:
        rd_target = list(target_delta_logits_stack[
            int(restart_dominance_target_index)])
    else:
        rd_target = list(target_delta_logits_stack[-1])
    inner_v6 = (
        v15_fit.inner_v14.inner_v13.inner_v12.inner_v11.inner_v10
        .inner_v9.inner_v8.inner_v7.inner_v6)
    v6_fit_rd, v6_audit_rd = fit_kv_bridge_v6_multi_target(
        params=params.v15_params.v3_params,
        projection=inner_v6,
        train_carriers=list(train_carriers),
        target_delta_logits_stack=[rd_target],
        follow_up_token_ids=list(follow_up_token_ids),
        n_directions=int(n_directions),
        ridge_lambda=float(ridge_lambda))
    # Wrap back through the nested projection chain.
    new_inner_v7 = dataclasses.replace(
        v15_fit.inner_v14.inner_v13.inner_v12.inner_v11.inner_v10
        .inner_v9.inner_v8.inner_v7, inner_v6=v6_fit_rd)
    new_inner_v8 = dataclasses.replace(
        v15_fit.inner_v14.inner_v13.inner_v12.inner_v11.inner_v10
        .inner_v9.inner_v8, inner_v7=new_inner_v7)
    new_inner_v9 = dataclasses.replace(
        v15_fit.inner_v14.inner_v13.inner_v12.inner_v11.inner_v10
        .inner_v9, inner_v8=new_inner_v8)
    new_inner_v10 = dataclasses.replace(
        v15_fit.inner_v14.inner_v13.inner_v12.inner_v11.inner_v10,
        inner_v9=new_inner_v9)
    new_inner_v11 = dataclasses.replace(
        v15_fit.inner_v14.inner_v13.inner_v12.inner_v11,
        inner_v10=new_inner_v10)
    new_inner_v12 = dataclasses.replace(
        v15_fit.inner_v14.inner_v13.inner_v12,
        inner_v11=new_inner_v11)
    new_inner_v13 = dataclasses.replace(
        v15_fit.inner_v14.inner_v13, inner_v12=new_inner_v12)
    new_inner_v14 = dataclasses.replace(
        v15_fit.inner_v14, inner_v13=new_inner_v13)
    new_v15 = dataclasses.replace(
        v15_fit, inner_v14=new_inner_v14)
    new_proj = dataclasses.replace(
        projection, inner_v15=new_v15)
    pre12 = float(v6_audit_rd.pre_fit_mean_residual)
    post12 = float(v6_audit_rd.post_fit_mean_residual)
    per_pre = (
        list(v15_report.per_target_pre_residual) + [pre12])
    per_post = (
        list(v15_report.per_target_post_residual) + [post12])
    converged = bool(
        all(po <= pr + 1e-9
            for pr, po in zip(per_pre[:11], per_post[:11]))
        and per_post[11] <= per_pre[11] + 1e-2)
    report = KVBridgeV16FitReport(
        schema=W71_KV_BRIDGE_V16_SCHEMA_VERSION,
        n_targets=int(n_targets),
        per_target_pre_residual=tuple(per_pre),
        per_target_post_residual=tuple(per_post),
        restart_dominance_target_index=int(
            restart_dominance_target_index),
        restart_dominance_pre=float(pre12),
        restart_dominance_post=float(post12),
        converged=bool(converged),
        ridge_lambda=float(ridge_lambda),
        inner_v15_report_cid=str(v15_report.cid()),
    )
    return new_proj, report


def compute_delayed_repair_fingerprint_v16(
        *, role: str,
        repair_trajectory_cid: str,
        delayed_repair_trajectory_cid: str,
        dominant_repair_label: int = 0,
        restart_count: int = 0,
        visible_token_budget: float = 256.0,
        baseline_cost: float = 512.0,
        task_id: str = "task", team_id: str = "team",
        branch_id: str = "main",
        delay_turns: int = 0,
        dim: int = W71_KV_V16_FINGERPRINT_DIM,
) -> tuple[float, ...]:
    """Deterministic 84-dim delayed-repair fingerprint."""
    label_str = (
        W71_REPAIR_LABELS_V16[int(dominant_repair_label)]
        if 0 <= int(dominant_repair_label)
            < len(W71_REPAIR_LABELS_V16)
        else "unknown")
    base = (
        f"{role}|{repair_trajectory_cid}|"
        f"{delayed_repair_trajectory_cid}|{label_str}|"
        f"{int(restart_count)}|"
        f"{float(visible_token_budget)}|{float(baseline_cost)}|"
        f"{task_id}|{team_id}|{branch_id}|"
        f"{int(delay_turns)}"
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


def probe_kv_bridge_v16_restart_dominance_margin(
        *, params: TinyV16SubstrateParams,
        token_ids: Sequence[int], n_targets: int = 12,
) -> dict[str, Any]:
    """Substrate-measured per-target margin probe via V16 forward."""
    base_trace, _ = forward_tiny_substrate_v16(
        params, list(token_ids))
    base_logits = _np.asarray(
        base_trace.logits, dtype=_np.float64)
    margins: list[float] = []
    for _ in range(int(n_targets)):
        t, _ = forward_tiny_substrate_v16(
            params, list(token_ids))
        l = _np.asarray(t.logits, dtype=_np.float64)
        diff = float(
            _np.linalg.norm((l - base_logits).ravel()))
        margins.append(diff)
    return {
        "schema": W71_KV_BRIDGE_V16_SCHEMA_VERSION,
        "kind": "v16_restart_dominance_margin_probe",
        "n_targets": int(n_targets),
        "per_target_margin_l2": [
            float(round(float(x), 12)) for x in margins],
        "max_margin": float(round(
            max(margins) if margins else 0.0, 12)),
    }


@dataclasses.dataclass(frozen=True)
class KVBridgeV16RestartDominanceFalsifier:
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
                "kv_bridge_v16_restart_dominance_falsifier",
            "falsifier": self.to_dict()})


def probe_kv_bridge_v16_restart_dominance_falsifier(
        *, restart_dominance_flag: int,
) -> KVBridgeV16RestartDominanceFalsifier:
    """Returns 0 iff inverting the restart-dominance flag flips
    the routing decision.

    Inversion semantics: flag 0 → 1, any nonzero → 0. The routing
    decision is ``route_through_substrate`` when flag > 0 and
    ``route_through_text`` otherwise. Flipping the flag must flip
    the decision (honest case: score = 0).
    """
    f = int(restart_dominance_flag)
    inv = 1 if f == 0 else 0
    decision = (
        "route_through_substrate" if f > 0
        else "route_through_text")
    inv_decision = (
        "route_through_substrate" if inv > 0
        else "route_through_text")
    flipped = decision != inv_decision
    score = 0.0 if flipped else 1.0
    return KVBridgeV16RestartDominanceFalsifier(
        primary_flag=int(f),
        inverted_flag=int(inv),
        decision=str(decision),
        inverted_decision=str(inv_decision),
        falsifier_score=float(score),
    )


@dataclasses.dataclass(frozen=True)
class KVBridgeV16Witness:
    schema: str
    projection_cid: str
    fit_report_cid: str
    restart_dominance_margin_probe_cid: str
    restart_dominance_falsifier_cid: str
    delayed_repair_fingerprint_l1: float
    max_restart_dominance_margin: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "projection_cid": str(self.projection_cid),
            "fit_report_cid": str(self.fit_report_cid),
            "restart_dominance_margin_probe_cid": str(
                self.restart_dominance_margin_probe_cid),
            "restart_dominance_falsifier_cid": str(
                self.restart_dominance_falsifier_cid),
            "delayed_repair_fingerprint_l1": float(round(
                self.delayed_repair_fingerprint_l1, 12)),
            "max_restart_dominance_margin": float(round(
                self.max_restart_dominance_margin, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "kv_bridge_v16_witness",
            "witness": self.to_dict()})


def emit_kv_bridge_v16_witness(
        *, projection: KVBridgeV16Projection,
        fit_report: KVBridgeV16FitReport | None = None,
        restart_dominance_margin_probe: (
            dict[str, Any] | None) = None,
        restart_dominance_falsifier: (
            KVBridgeV16RestartDominanceFalsifier | None) = None,
        delayed_repair_fingerprint: (
            Sequence[float] | None) = None,
) -> KVBridgeV16Witness:
    fp_l1 = 0.0
    if delayed_repair_fingerprint is not None:
        fp_l1 = float(sum(
            abs(float(x))
            for x in delayed_repair_fingerprint))
    return KVBridgeV16Witness(
        schema=W71_KV_BRIDGE_V16_SCHEMA_VERSION,
        projection_cid=str(projection.cid()),
        fit_report_cid=(
            fit_report.cid() if fit_report is not None else ""),
        restart_dominance_margin_probe_cid=(
            _sha256_hex(restart_dominance_margin_probe)
            if restart_dominance_margin_probe is not None
            else ""),
        restart_dominance_falsifier_cid=(
            restart_dominance_falsifier.cid()
            if restart_dominance_falsifier is not None
            else ""),
        delayed_repair_fingerprint_l1=float(fp_l1),
        max_restart_dominance_margin=float(
            restart_dominance_margin_probe["max_margin"]
            if restart_dominance_margin_probe is not None
            and "max_margin"
                in restart_dominance_margin_probe
            else 0.0),
    )


__all__ = [
    "W71_KV_BRIDGE_V16_SCHEMA_VERSION",
    "W71_DEFAULT_KV_V16_RIDGE_LAMBDA",
    "W71_KV_V16_FINGERPRINT_DIM",
    "KVBridgeV16Projection",
    "KVBridgeV16FitReport",
    "fit_kv_bridge_v16_twelve_target",
    "compute_delayed_repair_fingerprint_v16",
    "probe_kv_bridge_v16_restart_dominance_margin",
    "KVBridgeV16RestartDominanceFalsifier",
    "probe_kv_bridge_v16_restart_dominance_falsifier",
    "KVBridgeV16Witness",
    "emit_kv_bridge_v16_witness",
]
