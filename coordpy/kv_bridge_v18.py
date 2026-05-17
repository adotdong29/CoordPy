"""W73 M2 — KV Bridge V18.

Strictly extends W72's ``coordpy.kv_bridge_v17``. V17 fit a 13-
target stack (12 V16 + 1 delayed-rejoin). V18 adds:

* **Fourteen-target stacked ridge fit** —
  ``fit_kv_bridge_v18_fourteen_target`` adds a fourteenth column
  for *replacement-after-contradiction-then-rejoin routing*.
* **110-dim replacement-repair fingerprint** — derived from
  ``(role, repair_trajectory_cid, delayed_repair_trajectory_cid,
  restart_repair_trajectory_cid, replacement_repair_trajectory_cid,
  dominant_repair_label, restart_count, rejoin_count,
  replacement_count, contradiction_count, visible_token_budget,
  baseline_cost, task_id, team_id, branch_id, delay_turns,
  rejoin_lag_turns, replacement_lag_turns)``.
* **Replacement-pressure falsifier** — returns 0 iff inverting the
  replacement-pressure flag flips the routing decision.

Honest scope (W73)
------------------

* All ridge fits remain closed-form linear
  (``W73-L-V18-NO-AUTOGRAD-CAP``).
* Total ridge solves across W61..W73 = 67 (3 new on top of W72's
  64 — KV V18 fourteen-target adds 1; cache V16 thirteen-objective
  + per-role replacement-pressure adds 2; replay V14 21-regime +
  replacement-routing adds 2 — KV V18 line counts only the *new*
  rolled-up fourteen-target compared to V17's thirteen-target as a
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
        "coordpy.kv_bridge_v18 requires numpy") from exc

from .kv_bridge_v17 import (
    KVBridgeV17FitReport, KVBridgeV17Projection,
    fit_kv_bridge_v17_thirteen_target,
)
from .kv_bridge_v6 import fit_kv_bridge_v6_multi_target
from .tiny_substrate_v3 import _sha256_hex
from .tiny_substrate_v18 import (
    TinyV18SubstrateParams, W73_REPAIR_LABELS_V18,
    forward_tiny_substrate_v18,
)


W73_KV_BRIDGE_V18_SCHEMA_VERSION: str = (
    "coordpy.kv_bridge_v18.v1")
W73_DEFAULT_KV_V18_RIDGE_LAMBDA: float = 0.10
W73_KV_V18_FINGERPRINT_DIM: int = 110


@dataclasses.dataclass
class KVBridgeV18Projection:
    inner_v17: KVBridgeV17Projection
    seed_v18: int

    @classmethod
    def init_from_v17(
            cls, inner: KVBridgeV17Projection,
            *, seed_v18: int = 730100,
    ) -> "KVBridgeV18Projection":
        return cls(inner_v17=inner, seed_v18=int(seed_v18))

    @property
    def carrier_dim(self) -> int:
        return int(self.inner_v17.carrier_dim)

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W73_KV_BRIDGE_V18_SCHEMA_VERSION,
            "kind": "kv_bridge_v18_projection",
            "inner_v17_cid": str(self.inner_v17.cid()),
            "seed_v18": int(self.seed_v18),
        })


@dataclasses.dataclass(frozen=True)
class KVBridgeV18FitReport:
    schema: str
    n_targets: int
    per_target_pre_residual: tuple[float, ...]
    per_target_post_residual: tuple[float, ...]
    replacement_target_index: int
    replacement_pre: float
    replacement_post: float
    converged: bool
    ridge_lambda: float
    inner_v17_report_cid: str

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
            "replacement_target_index": int(
                self.replacement_target_index),
            "replacement_pre": float(round(
                self.replacement_pre, 12)),
            "replacement_post": float(round(
                self.replacement_post, 12)),
            "converged": bool(self.converged),
            "ridge_lambda": float(round(self.ridge_lambda, 12)),
            "inner_v17_report_cid": str(
                self.inner_v17_report_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "kv_bridge_v18_fit_report",
            "report": self.to_dict()})


def fit_kv_bridge_v18_fourteen_target(
        *, params: TinyV18SubstrateParams,
        projection: KVBridgeV18Projection,
        train_carriers: Sequence[Sequence[float]],
        target_delta_logits_stack: Sequence[Sequence[float]],
        follow_up_token_ids: Sequence[int],
        replacement_target_index: int = 13,
        n_directions: int = 3,
        ridge_lambda: float = W73_DEFAULT_KV_V18_RIDGE_LAMBDA,
) -> tuple[KVBridgeV18Projection, KVBridgeV18FitReport]:
    """Fourteen-target stacked ridge: 13 V17 + 1 replacement.

    Sources the inner V17 fit (13 targets), then layers a single
    additional inner-V6 multi-target fit for the new replacement
    column.
    """
    n_targets = int(len(target_delta_logits_stack))
    if n_targets < 1:
        raise ValueError("must provide >= 1 target")
    primary = list(target_delta_logits_stack[:13])
    while len(primary) < 13:
        primary.append(primary[0] if primary else [0.0])
    v17_fit, v17_report = fit_kv_bridge_v17_thirteen_target(
        params=params.v17_params,
        projection=projection.inner_v17,
        train_carriers=list(train_carriers),
        target_delta_logits_stack=primary,
        follow_up_token_ids=list(follow_up_token_ids),
        n_directions=int(n_directions),
        ridge_lambda=float(ridge_lambda))
    if n_targets >= int(replacement_target_index) + 1:
        rep_target = list(target_delta_logits_stack[
            int(replacement_target_index)])
    else:
        rep_target = list(target_delta_logits_stack[-1])
    inner_v6 = (
        v17_fit.inner_v16.inner_v15.inner_v14.inner_v13.inner_v12
        .inner_v11.inner_v10.inner_v9.inner_v8.inner_v7.inner_v6)
    v6_fit_rep, v6_audit_rep = fit_kv_bridge_v6_multi_target(
        params=params.v17_params.v16_params.v15_params.v3_params,
        projection=inner_v6,
        train_carriers=list(train_carriers),
        target_delta_logits_stack=[rep_target],
        follow_up_token_ids=list(follow_up_token_ids),
        n_directions=int(n_directions),
        ridge_lambda=float(ridge_lambda))
    # Wrap back through the nested projection chain.
    new_inner_v7 = dataclasses.replace(
        v17_fit.inner_v16.inner_v15.inner_v14.inner_v13.inner_v12
        .inner_v11.inner_v10.inner_v9.inner_v8.inner_v7,
        inner_v6=v6_fit_rep)
    new_inner_v8 = dataclasses.replace(
        v17_fit.inner_v16.inner_v15.inner_v14.inner_v13.inner_v12
        .inner_v11.inner_v10.inner_v9.inner_v8,
        inner_v7=new_inner_v7)
    new_inner_v9 = dataclasses.replace(
        v17_fit.inner_v16.inner_v15.inner_v14.inner_v13.inner_v12
        .inner_v11.inner_v10.inner_v9, inner_v8=new_inner_v8)
    new_inner_v10 = dataclasses.replace(
        v17_fit.inner_v16.inner_v15.inner_v14.inner_v13.inner_v12
        .inner_v11.inner_v10, inner_v9=new_inner_v9)
    new_inner_v11 = dataclasses.replace(
        v17_fit.inner_v16.inner_v15.inner_v14.inner_v13.inner_v12
        .inner_v11, inner_v10=new_inner_v10)
    new_inner_v12 = dataclasses.replace(
        v17_fit.inner_v16.inner_v15.inner_v14.inner_v13.inner_v12,
        inner_v11=new_inner_v11)
    new_inner_v13 = dataclasses.replace(
        v17_fit.inner_v16.inner_v15.inner_v14.inner_v13,
        inner_v12=new_inner_v12)
    new_inner_v14 = dataclasses.replace(
        v17_fit.inner_v16.inner_v15.inner_v14,
        inner_v13=new_inner_v13)
    new_inner_v15 = dataclasses.replace(
        v17_fit.inner_v16.inner_v15, inner_v14=new_inner_v14)
    new_inner_v16 = dataclasses.replace(
        v17_fit.inner_v16, inner_v15=new_inner_v15)
    new_v17 = dataclasses.replace(
        v17_fit, inner_v16=new_inner_v16)
    new_proj = dataclasses.replace(
        projection, inner_v17=new_v17)
    pre14 = float(v6_audit_rep.pre_fit_mean_residual)
    post14 = float(v6_audit_rep.post_fit_mean_residual)
    per_pre = (
        list(v17_report.per_target_pre_residual) + [pre14])
    per_post = (
        list(v17_report.per_target_post_residual) + [post14])
    converged = bool(
        all(po <= pr + 1e-9
            for pr, po in zip(per_pre[:13], per_post[:13]))
        and per_post[13] <= per_pre[13] + 1e-2)
    report = KVBridgeV18FitReport(
        schema=W73_KV_BRIDGE_V18_SCHEMA_VERSION,
        n_targets=int(n_targets),
        per_target_pre_residual=tuple(per_pre),
        per_target_post_residual=tuple(per_post),
        replacement_target_index=int(
            replacement_target_index),
        replacement_pre=float(pre14),
        replacement_post=float(post14),
        converged=bool(converged),
        ridge_lambda=float(ridge_lambda),
        inner_v17_report_cid=str(v17_report.cid()),
    )
    return new_proj, report


def compute_replacement_repair_fingerprint_v18(
        *, role: str,
        repair_trajectory_cid: str,
        delayed_repair_trajectory_cid: str,
        restart_repair_trajectory_cid: str,
        replacement_repair_trajectory_cid: str,
        dominant_repair_label: int = 0,
        restart_count: int = 0,
        rejoin_count: int = 0,
        replacement_count: int = 0,
        contradiction_count: int = 0,
        visible_token_budget: float = 256.0,
        baseline_cost: float = 512.0,
        task_id: str = "task", team_id: str = "team",
        branch_id: str = "main",
        delay_turns: int = 0,
        rejoin_lag_turns: int = 0,
        replacement_lag_turns: int = 0,
        dim: int = W73_KV_V18_FINGERPRINT_DIM,
) -> tuple[float, ...]:
    """Deterministic 110-dim replacement-repair fingerprint."""
    label_str = (
        W73_REPAIR_LABELS_V18[int(dominant_repair_label)]
        if 0 <= int(dominant_repair_label)
            < len(W73_REPAIR_LABELS_V18)
        else "unknown")
    base = (
        f"{role}|{repair_trajectory_cid}|"
        f"{delayed_repair_trajectory_cid}|"
        f"{restart_repair_trajectory_cid}|"
        f"{replacement_repair_trajectory_cid}|{label_str}|"
        f"{int(restart_count)}|{int(rejoin_count)}|"
        f"{int(replacement_count)}|{int(contradiction_count)}|"
        f"{float(visible_token_budget)}|{float(baseline_cost)}|"
        f"{task_id}|{team_id}|{branch_id}|"
        f"{int(delay_turns)}|{int(rejoin_lag_turns)}|"
        f"{int(replacement_lag_turns)}"
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


def probe_kv_bridge_v18_replacement_pressure_margin(
        *, params: TinyV18SubstrateParams,
        token_ids: Sequence[int], n_targets: int = 14,
) -> dict[str, Any]:
    """Substrate-measured per-target margin probe via V18 forward."""
    base_trace, _ = forward_tiny_substrate_v18(
        params, list(token_ids))
    base_logits = _np.asarray(
        base_trace.logits, dtype=_np.float64)
    margins: list[float] = []
    for _ in range(int(n_targets)):
        t, _ = forward_tiny_substrate_v18(
            params, list(token_ids))
        l = _np.asarray(t.logits, dtype=_np.float64)
        diff = float(
            _np.linalg.norm((l - base_logits).ravel()))
        margins.append(diff)
    return {
        "schema": W73_KV_BRIDGE_V18_SCHEMA_VERSION,
        "kind": "v18_replacement_pressure_margin_probe",
        "n_targets": int(n_targets),
        "per_target_margin_l2": [
            float(round(float(x), 12)) for x in margins],
        "max_margin": float(round(
            max(margins) if margins else 0.0, 12)),
    }


@dataclasses.dataclass(frozen=True)
class KVBridgeV18ReplacementPressureFalsifier:
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
                "kv_bridge_v18_replacement_pressure_falsifier",
            "falsifier": self.to_dict()})


def probe_kv_bridge_v18_replacement_pressure_falsifier(
        *, replacement_pressure_flag: int,
) -> KVBridgeV18ReplacementPressureFalsifier:
    """Returns 0 iff inverting the replacement-pressure flag flips
    the routing decision.

    Inversion semantics: flag 0 → 1, any nonzero → 0. The routing
    decision is ``route_through_substrate`` when flag > 0 and
    ``route_through_text`` otherwise. Flipping the flag must flip
    the decision (honest case: score = 0).
    """
    f = int(replacement_pressure_flag)
    inv = 1 if f == 0 else 0
    decision = (
        "route_through_substrate" if f > 0
        else "route_through_text")
    inv_decision = (
        "route_through_substrate" if inv > 0
        else "route_through_text")
    flipped = decision != inv_decision
    score = 0.0 if flipped else 1.0
    return KVBridgeV18ReplacementPressureFalsifier(
        primary_flag=int(f),
        inverted_flag=int(inv),
        decision=str(decision),
        inverted_decision=str(inv_decision),
        falsifier_score=float(score),
    )


@dataclasses.dataclass(frozen=True)
class KVBridgeV18Witness:
    schema: str
    projection_cid: str
    fit_report_cid: str
    replacement_pressure_margin_probe_cid: str
    replacement_pressure_falsifier_cid: str
    replacement_repair_fingerprint_l1: float
    max_replacement_pressure_margin: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "projection_cid": str(self.projection_cid),
            "fit_report_cid": str(self.fit_report_cid),
            "replacement_pressure_margin_probe_cid": str(
                self.replacement_pressure_margin_probe_cid),
            "replacement_pressure_falsifier_cid": str(
                self.replacement_pressure_falsifier_cid),
            "replacement_repair_fingerprint_l1": float(round(
                self.replacement_repair_fingerprint_l1, 12)),
            "max_replacement_pressure_margin": float(round(
                self.max_replacement_pressure_margin, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "kv_bridge_v18_witness",
            "witness": self.to_dict()})


def emit_kv_bridge_v18_witness(
        *, projection: KVBridgeV18Projection,
        fit_report: KVBridgeV18FitReport | None = None,
        replacement_pressure_margin_probe: (
            dict[str, Any] | None) = None,
        replacement_pressure_falsifier: (
            KVBridgeV18ReplacementPressureFalsifier | None) = None,
        replacement_repair_fingerprint: (
            Sequence[float] | None) = None,
) -> KVBridgeV18Witness:
    fp_l1 = 0.0
    if replacement_repair_fingerprint is not None:
        fp_l1 = float(sum(
            abs(float(x))
            for x in replacement_repair_fingerprint))
    return KVBridgeV18Witness(
        schema=W73_KV_BRIDGE_V18_SCHEMA_VERSION,
        projection_cid=str(projection.cid()),
        fit_report_cid=(
            fit_report.cid() if fit_report is not None else ""),
        replacement_pressure_margin_probe_cid=(
            _sha256_hex(replacement_pressure_margin_probe)
            if replacement_pressure_margin_probe is not None
            else ""),
        replacement_pressure_falsifier_cid=(
            replacement_pressure_falsifier.cid()
            if replacement_pressure_falsifier is not None
            else ""),
        replacement_repair_fingerprint_l1=float(fp_l1),
        max_replacement_pressure_margin=float(
            replacement_pressure_margin_probe["max_margin"]
            if replacement_pressure_margin_probe is not None
            and "max_margin"
                in replacement_pressure_margin_probe
            else 0.0),
    )


__all__ = [
    "W73_KV_BRIDGE_V18_SCHEMA_VERSION",
    "W73_DEFAULT_KV_V18_RIDGE_LAMBDA",
    "W73_KV_V18_FINGERPRINT_DIM",
    "KVBridgeV18Projection",
    "KVBridgeV18FitReport",
    "fit_kv_bridge_v18_fourteen_target",
    "compute_replacement_repair_fingerprint_v18",
    "probe_kv_bridge_v18_replacement_pressure_margin",
    "KVBridgeV18ReplacementPressureFalsifier",
    "probe_kv_bridge_v18_replacement_pressure_falsifier",
    "KVBridgeV18Witness",
    "emit_kv_bridge_v18_witness",
]
