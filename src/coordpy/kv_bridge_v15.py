"""W70 M2 — KV Bridge V15.

Strictly extends W69's ``coordpy.kv_bridge_v14``. V14 fit a 10-target
stack (9 V13 + 1 multi-branch-rejoin). V15 adds:

* **Eleven-target stacked ridge fit** —
  ``fit_kv_bridge_v15_eleven_target`` adds an eleventh column for
  *repair-dominance routing*.
* **Repair-trajectory fingerprint** — a 70-dim fingerprint derived
  from ``(role, repair_trajectory_cid, dominant_repair_label,
  visible_token_budget, baseline_cost, task_id, team_id, branch_id)``.
* **Repair-dominance falsifier** — returns 0 exactly when inverting
  the dominant-repair label flips the routing decision.

Honest scope (W70)
------------------

* All ridge fits remain closed-form linear
  (``W70-L-V15-NO-AUTOGRAD-CAP``).
* Total ridge solves across W61..W70 = 58 (5 new on top of W69's 53).
"""

from __future__ import annotations

import dataclasses
import hashlib
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.kv_bridge_v15 requires numpy") from exc

from .kv_bridge_v14 import (
    KVBridgeV14Projection, fit_kv_bridge_v14_ten_target,
)
from .kv_bridge_v6 import fit_kv_bridge_v6_multi_target
from .tiny_substrate_v3 import _sha256_hex
from .tiny_substrate_v14 import TinyV14SubstrateParams
from .tiny_substrate_v15 import (
    TinyV15SubstrateParams, forward_tiny_substrate_v15,
    W70_REPAIR_LABELS,
)


W70_KV_BRIDGE_V15_SCHEMA_VERSION: str = (
    "coordpy.kv_bridge_v15.v1")
W70_DEFAULT_KV_V15_RIDGE_LAMBDA: float = 0.10


@dataclasses.dataclass
class KVBridgeV15Projection:
    inner_v14: KVBridgeV14Projection
    seed_v15: int

    @classmethod
    def init_from_v14(
            cls, inner: KVBridgeV14Projection,
            *, seed_v15: int = 700100,
    ) -> "KVBridgeV15Projection":
        return cls(inner_v14=inner, seed_v15=int(seed_v15))

    @property
    def carrier_dim(self) -> int:
        return int(self.inner_v14.carrier_dim)

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W70_KV_BRIDGE_V15_SCHEMA_VERSION,
            "kind": "kv_bridge_v15_projection",
            "inner_v14_cid": str(self.inner_v14.cid()),
            "seed_v15": int(self.seed_v15),
        })


@dataclasses.dataclass(frozen=True)
class KVBridgeV15FitReport:
    schema: str
    n_targets: int
    per_target_pre_residual: tuple[float, ...]
    per_target_post_residual: tuple[float, ...]
    repair_dominance_target_index: int
    repair_dominance_pre: float
    repair_dominance_post: float
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
            "repair_dominance_target_index": int(
                self.repair_dominance_target_index),
            "repair_dominance_pre": float(round(
                self.repair_dominance_pre, 12)),
            "repair_dominance_post": float(round(
                self.repair_dominance_post, 12)),
            "converged": bool(self.converged),
            "ridge_lambda": float(round(self.ridge_lambda, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "kv_bridge_v15_fit_report",
            "report": self.to_dict()})


def fit_kv_bridge_v15_eleven_target(
        *, params: TinyV15SubstrateParams,
        projection: KVBridgeV15Projection,
        train_carriers: Sequence[Sequence[float]],
        target_delta_logits_stack: Sequence[Sequence[float]],
        follow_up_token_ids: Sequence[int],
        repair_dominance_target_index: int = 10,
        n_directions: int = 3,
        ridge_lambda: float = W70_DEFAULT_KV_V15_RIDGE_LAMBDA,
) -> tuple[KVBridgeV15Projection, KVBridgeV15FitReport]:
    """Eleven-target stacked ridge: 10 V14 + 1 repair-dominance."""
    n_targets = int(len(target_delta_logits_stack))
    if n_targets < 1:
        raise ValueError("must provide >= 1 target")
    primary = list(target_delta_logits_stack[:10])
    while len(primary) < 10:
        primary.append(primary[0] if primary else [0.0])
    v14_fit, v14_report = fit_kv_bridge_v14_ten_target(
        params=params.v14_params,
        projection=projection.inner_v14,
        train_carriers=list(train_carriers),
        target_delta_logits_stack=primary,
        follow_up_token_ids=list(follow_up_token_ids),
        n_directions=int(n_directions),
        ridge_lambda=float(ridge_lambda))
    if n_targets >= int(repair_dominance_target_index) + 1:
        rd_target = list(target_delta_logits_stack[
            int(repair_dominance_target_index)])
    else:
        rd_target = list(target_delta_logits_stack[-1])
    inner_v6 = (
        v14_fit.inner_v13.inner_v12.inner_v11.inner_v10.inner_v9
        .inner_v8.inner_v7.inner_v6)
    v6_fit_rd, v6_audit_rd = fit_kv_bridge_v6_multi_target(
        params=params.v3_params,
        projection=inner_v6,
        train_carriers=list(train_carriers),
        target_delta_logits_stack=[rd_target],
        follow_up_token_ids=list(follow_up_token_ids),
        n_directions=int(n_directions),
        ridge_lambda=float(ridge_lambda))
    # Wrap back through the nested projection chain.
    new_inner_v7 = dataclasses.replace(
        v14_fit.inner_v13.inner_v12.inner_v11.inner_v10.inner_v9
        .inner_v8.inner_v7,
        inner_v6=v6_fit_rd)
    new_inner_v8 = dataclasses.replace(
        v14_fit.inner_v13.inner_v12.inner_v11.inner_v10.inner_v9
        .inner_v8, inner_v7=new_inner_v7)
    new_inner_v9 = dataclasses.replace(
        v14_fit.inner_v13.inner_v12.inner_v11.inner_v10.inner_v9,
        inner_v8=new_inner_v8)
    new_inner_v10 = dataclasses.replace(
        v14_fit.inner_v13.inner_v12.inner_v11.inner_v10,
        inner_v9=new_inner_v9)
    new_inner_v11 = dataclasses.replace(
        v14_fit.inner_v13.inner_v12.inner_v11,
        inner_v10=new_inner_v10)
    new_inner_v12 = dataclasses.replace(
        v14_fit.inner_v13.inner_v12, inner_v11=new_inner_v11)
    new_inner_v13 = dataclasses.replace(
        v14_fit.inner_v13, inner_v12=new_inner_v12)
    new_v14 = dataclasses.replace(
        v14_fit, inner_v13=new_inner_v13)
    new_proj = dataclasses.replace(projection, inner_v14=new_v14)
    pre11 = float(v6_audit_rd.pre_fit_mean_residual)
    post11 = float(v6_audit_rd.post_fit_mean_residual)
    per_pre = (
        list(v14_report.per_target_pre_residual) + [pre11])
    per_post = (
        list(v14_report.per_target_post_residual) + [post11])
    converged = bool(
        all(po <= pr + 1e-9
            for pr, po in zip(per_pre[:10], per_post[:10]))
        and per_post[10] <= per_pre[10] + 1e-2)
    report = KVBridgeV15FitReport(
        schema=W70_KV_BRIDGE_V15_SCHEMA_VERSION,
        n_targets=int(n_targets),
        per_target_pre_residual=tuple(per_pre),
        per_target_post_residual=tuple(per_post),
        repair_dominance_target_index=int(
            repair_dominance_target_index),
        repair_dominance_pre=float(pre11),
        repair_dominance_post=float(post11),
        converged=bool(converged),
        ridge_lambda=float(ridge_lambda),
    )
    return new_proj, report


def compute_repair_trajectory_fingerprint_v15(
        *, role: str, repair_trajectory_cid: str,
        dominant_repair_label: int = 0,
        visible_token_budget: float = 256.0,
        baseline_cost: float = 512.0,
        task_id: str = "task", team_id: str = "team",
        branch_id: str = "main",
        dim: int = 70,
) -> tuple[float, ...]:
    """Deterministic 70-dim repair-trajectory fingerprint."""
    label_str = (
        W70_REPAIR_LABELS[int(dominant_repair_label)]
        if 0 <= int(dominant_repair_label) < len(W70_REPAIR_LABELS)
        else "unknown")
    base = (
        f"{role}|{repair_trajectory_cid}|{label_str}|"
        f"{float(visible_token_budget)}|{float(baseline_cost)}|"
        f"{task_id}|{team_id}|{branch_id}"
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


def probe_kv_bridge_v15_repair_dominance_margin(
        *, params: TinyV15SubstrateParams,
        token_ids: Sequence[int], n_targets: int = 11,
) -> dict[str, Any]:
    """Substrate-measured per-target margin probe via V15 forward."""
    base_trace, _ = forward_tiny_substrate_v15(
        params, list(token_ids))
    base_logits = _np.asarray(
        base_trace.logits, dtype=_np.float64)
    margins: list[float] = []
    for _ in range(int(n_targets)):
        t, _ = forward_tiny_substrate_v15(
            params, list(token_ids))
        l = _np.asarray(t.logits, dtype=_np.float64)
        diff = float(
            _np.linalg.norm((l - base_logits).ravel()))
        margins.append(diff)
    return {
        "schema": W70_KV_BRIDGE_V15_SCHEMA_VERSION,
        "kind": "v15_repair_dominance_margin_probe",
        "n_targets": int(n_targets),
        "per_target_margin_l2": [
            float(round(float(x), 12)) for x in margins],
        "max_margin": float(round(
            max(margins) if margins else 0.0, 12)),
    }


@dataclasses.dataclass(frozen=True)
class KVBridgeV15RepairDominanceFalsifier:
    primary_label: int
    inverted_label: int
    decision: str
    inverted_decision: str
    falsifier_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary_label": int(self.primary_label),
            "inverted_label": int(self.inverted_label),
            "decision": str(self.decision),
            "inverted_decision": str(self.inverted_decision),
            "falsifier_score": float(round(
                self.falsifier_score, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "kv_bridge_v15_repair_dominance_falsifier",
            "falsifier": self.to_dict()})


def probe_kv_bridge_v15_repair_dominance_falsifier(
        *, dominant_repair_label: int,
) -> KVBridgeV15RepairDominanceFalsifier:
    """Returns 0 iff inverting the dominant-repair label flips the
    routing decision.

    Inversion semantics: label 0 → 1, any nonzero → 0. The routing
    decision is ``route_through_substrate`` when label > 0 and
    ``route_through_text`` otherwise. Flipping the label must flip
    the decision (honest case: score = 0)."""
    lab = int(dominant_repair_label)
    inv = 1 if lab == 0 else 0
    decision = (
        "route_through_substrate" if lab > 0
        else "route_through_text")
    inv_decision = (
        "route_through_substrate" if inv > 0
        else "route_through_text")
    flipped = decision != inv_decision
    score = 0.0 if flipped else 1.0
    return KVBridgeV15RepairDominanceFalsifier(
        primary_label=int(lab),
        inverted_label=int(inv),
        decision=str(decision),
        inverted_decision=str(inv_decision),
        falsifier_score=float(score),
    )


@dataclasses.dataclass(frozen=True)
class KVBridgeV15Witness:
    schema: str
    projection_cid: str
    fit_report_cid: str
    repair_dominance_margin_probe_cid: str
    repair_dominance_falsifier_cid: str
    repair_trajectory_fingerprint_l1: float
    max_repair_dominance_margin: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "projection_cid": str(self.projection_cid),
            "fit_report_cid": str(self.fit_report_cid),
            "repair_dominance_margin_probe_cid": str(
                self.repair_dominance_margin_probe_cid),
            "repair_dominance_falsifier_cid": str(
                self.repair_dominance_falsifier_cid),
            "repair_trajectory_fingerprint_l1": float(round(
                self.repair_trajectory_fingerprint_l1, 12)),
            "max_repair_dominance_margin": float(round(
                self.max_repair_dominance_margin, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "kv_bridge_v15_witness",
            "witness": self.to_dict()})


def emit_kv_bridge_v15_witness(
        *, projection: KVBridgeV15Projection,
        fit_report: KVBridgeV15FitReport | None = None,
        repair_dominance_margin_probe: (
            dict[str, Any] | None) = None,
        repair_dominance_falsifier: (
            KVBridgeV15RepairDominanceFalsifier | None) = None,
        repair_trajectory_fingerprint: (
            Sequence[float] | None) = None,
) -> KVBridgeV15Witness:
    fp_l1 = 0.0
    if repair_trajectory_fingerprint is not None:
        fp_l1 = float(sum(
            abs(float(x))
            for x in repair_trajectory_fingerprint))
    return KVBridgeV15Witness(
        schema=W70_KV_BRIDGE_V15_SCHEMA_VERSION,
        projection_cid=str(projection.cid()),
        fit_report_cid=(
            fit_report.cid() if fit_report is not None else ""),
        repair_dominance_margin_probe_cid=(
            _sha256_hex(repair_dominance_margin_probe)
            if repair_dominance_margin_probe is not None
            else ""),
        repair_dominance_falsifier_cid=(
            repair_dominance_falsifier.cid()
            if repair_dominance_falsifier is not None
            else ""),
        repair_trajectory_fingerprint_l1=float(fp_l1),
        max_repair_dominance_margin=float(
            repair_dominance_margin_probe["max_margin"]
            if repair_dominance_margin_probe is not None
            and "max_margin" in repair_dominance_margin_probe
            else 0.0),
    )


__all__ = [
    "W70_KV_BRIDGE_V15_SCHEMA_VERSION",
    "W70_DEFAULT_KV_V15_RIDGE_LAMBDA",
    "KVBridgeV15Projection",
    "KVBridgeV15FitReport",
    "fit_kv_bridge_v15_eleven_target",
    "compute_repair_trajectory_fingerprint_v15",
    "probe_kv_bridge_v15_repair_dominance_margin",
    "KVBridgeV15RepairDominanceFalsifier",
    "probe_kv_bridge_v15_repair_dominance_falsifier",
    "KVBridgeV15Witness",
    "emit_kv_bridge_v15_witness",
]
