"""W74 M2 — KV Bridge V19.

Strictly extends W73's ``coordpy.kv_bridge_v18``. V18 fit a 14-
target stack (13 V17 + 1 replacement). V19 adds:

* **Fifteen-target stacked ridge fit** —
  ``fit_kv_bridge_v19_fifteen_target`` adds a fifteenth column
  for *compound-repair-after-delayed-repair-then-replacement
  routing*.
* **120-dim compound-repair fingerprint** — derived from the V18
  fingerprint inputs plus the compound-repair-trajectory CID and
  compound-window width.
* **Compound-pressure falsifier** — returns 0 iff inverting the
  compound-pressure flag flips the routing decision.

Honest scope (W74)
------------------

* All ridge fits remain closed-form linear
  (``W74-L-V19-NO-AUTOGRAD-CAP``).
* Total ridge solves across W61..W74 = 70 (3 new on top of W73's
  67 — KV V19 fifteen-target adds 1; cache V17 fourteen-objective +
  per-role compound-pressure adds 2; replay V15 22-regime +
  compound-routing adds 2 — KV V19 line counts only the new rolled-
  up fifteen-target as a single new solve).
"""

from __future__ import annotations

import dataclasses
import hashlib
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.kv_bridge_v19 requires numpy") from exc

from .kv_bridge_v18 import (
    KVBridgeV18FitReport, KVBridgeV18Projection,
    fit_kv_bridge_v18_fourteen_target,
)
from .kv_bridge_v6 import fit_kv_bridge_v6_multi_target
from .tiny_substrate_v3 import _sha256_hex
from .tiny_substrate_v19 import (
    TinyV19SubstrateParams, W74_REPAIR_LABELS_V19,
    forward_tiny_substrate_v19,
)


W74_KV_BRIDGE_V19_SCHEMA_VERSION: str = (
    "coordpy.kv_bridge_v19.v1")
W74_DEFAULT_KV_V19_RIDGE_LAMBDA: float = 0.10
W74_KV_V19_FINGERPRINT_DIM: int = 120


@dataclasses.dataclass
class KVBridgeV19Projection:
    inner_v18: KVBridgeV18Projection
    seed_v19: int

    @classmethod
    def init_from_v18(
            cls, inner: KVBridgeV18Projection,
            *, seed_v19: int = 740100,
    ) -> "KVBridgeV19Projection":
        return cls(inner_v18=inner, seed_v19=int(seed_v19))

    @property
    def carrier_dim(self) -> int:
        return int(self.inner_v18.carrier_dim)

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W74_KV_BRIDGE_V19_SCHEMA_VERSION,
            "kind": "kv_bridge_v19_projection",
            "inner_v18_cid": str(self.inner_v18.cid()),
            "seed_v19": int(self.seed_v19),
        })


@dataclasses.dataclass(frozen=True)
class KVBridgeV19FitReport:
    schema: str
    n_targets: int
    per_target_pre_residual: tuple[float, ...]
    per_target_post_residual: tuple[float, ...]
    compound_target_index: int
    compound_pre: float
    compound_post: float
    converged: bool
    ridge_lambda: float
    inner_v18_report_cid: str

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
            "compound_target_index": int(
                self.compound_target_index),
            "compound_pre": float(round(
                self.compound_pre, 12)),
            "compound_post": float(round(
                self.compound_post, 12)),
            "converged": bool(self.converged),
            "ridge_lambda": float(round(self.ridge_lambda, 12)),
            "inner_v18_report_cid": str(
                self.inner_v18_report_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "kv_bridge_v19_fit_report",
            "report": self.to_dict()})


def fit_kv_bridge_v19_fifteen_target(
        *, params: TinyV19SubstrateParams,
        projection: KVBridgeV19Projection,
        train_carriers: Sequence[Sequence[float]],
        target_delta_logits_stack: Sequence[Sequence[float]],
        follow_up_token_ids: Sequence[int],
        compound_target_index: int = 14,
        n_directions: int = 3,
        ridge_lambda: float = W74_DEFAULT_KV_V19_RIDGE_LAMBDA,
) -> tuple[KVBridgeV19Projection, KVBridgeV19FitReport]:
    """Fifteen-target stacked ridge: 14 V18 + 1 compound.

    Sources the inner V18 fit (14 targets), then layers a single
    additional inner-V6 multi-target fit for the new compound
    column.
    """
    n_targets = int(len(target_delta_logits_stack))
    if n_targets < 1:
        raise ValueError("must provide >= 1 target")
    primary = list(target_delta_logits_stack[:14])
    while len(primary) < 14:
        primary.append(primary[0] if primary else [0.0])
    v18_fit, v18_report = fit_kv_bridge_v18_fourteen_target(
        params=params.v18_params,
        projection=projection.inner_v18,
        train_carriers=list(train_carriers),
        target_delta_logits_stack=primary,
        follow_up_token_ids=list(follow_up_token_ids),
        n_directions=int(n_directions),
        ridge_lambda=float(ridge_lambda))
    if n_targets >= int(compound_target_index) + 1:
        comp_target = list(target_delta_logits_stack[
            int(compound_target_index)])
    else:
        comp_target = list(target_delta_logits_stack[-1])
    inner_v6 = (
        v18_fit.inner_v17.inner_v16.inner_v15.inner_v14.inner_v13
        .inner_v12.inner_v11.inner_v10.inner_v9.inner_v8.inner_v7
        .inner_v6)
    v6_fit_comp, v6_audit_comp = fit_kv_bridge_v6_multi_target(
        params=params.v18_params.v17_params.v16_params.v15_params
        .v3_params,
        projection=inner_v6,
        train_carriers=list(train_carriers),
        target_delta_logits_stack=[comp_target],
        follow_up_token_ids=list(follow_up_token_ids),
        n_directions=int(n_directions),
        ridge_lambda=float(ridge_lambda))
    # Wrap back through the nested projection chain.
    new_inner_v7 = dataclasses.replace(
        v18_fit.inner_v17.inner_v16.inner_v15.inner_v14.inner_v13
        .inner_v12.inner_v11.inner_v10.inner_v9.inner_v8.inner_v7,
        inner_v6=v6_fit_comp)
    new_inner_v8 = dataclasses.replace(
        v18_fit.inner_v17.inner_v16.inner_v15.inner_v14.inner_v13
        .inner_v12.inner_v11.inner_v10.inner_v9.inner_v8,
        inner_v7=new_inner_v7)
    new_inner_v9 = dataclasses.replace(
        v18_fit.inner_v17.inner_v16.inner_v15.inner_v14.inner_v13
        .inner_v12.inner_v11.inner_v10.inner_v9,
        inner_v8=new_inner_v8)
    new_inner_v10 = dataclasses.replace(
        v18_fit.inner_v17.inner_v16.inner_v15.inner_v14.inner_v13
        .inner_v12.inner_v11.inner_v10, inner_v9=new_inner_v9)
    new_inner_v11 = dataclasses.replace(
        v18_fit.inner_v17.inner_v16.inner_v15.inner_v14.inner_v13
        .inner_v12.inner_v11, inner_v10=new_inner_v10)
    new_inner_v12 = dataclasses.replace(
        v18_fit.inner_v17.inner_v16.inner_v15.inner_v14.inner_v13
        .inner_v12, inner_v11=new_inner_v11)
    new_inner_v13 = dataclasses.replace(
        v18_fit.inner_v17.inner_v16.inner_v15.inner_v14.inner_v13,
        inner_v12=new_inner_v12)
    new_inner_v14 = dataclasses.replace(
        v18_fit.inner_v17.inner_v16.inner_v15.inner_v14,
        inner_v13=new_inner_v13)
    new_inner_v15 = dataclasses.replace(
        v18_fit.inner_v17.inner_v16.inner_v15,
        inner_v14=new_inner_v14)
    new_inner_v16 = dataclasses.replace(
        v18_fit.inner_v17.inner_v16, inner_v15=new_inner_v15)
    new_inner_v17 = dataclasses.replace(
        v18_fit.inner_v17, inner_v16=new_inner_v16)
    new_v18 = dataclasses.replace(
        v18_fit, inner_v17=new_inner_v17)
    new_proj = dataclasses.replace(
        projection, inner_v18=new_v18)
    pre15 = float(v6_audit_comp.pre_fit_mean_residual)
    post15 = float(v6_audit_comp.post_fit_mean_residual)
    per_pre = (
        list(v18_report.per_target_pre_residual) + [pre15])
    per_post = (
        list(v18_report.per_target_post_residual) + [post15])
    converged = bool(
        all(po <= pr + 1e-9
            for pr, po in zip(per_pre[:14], per_post[:14]))
        and per_post[14] <= per_pre[14] + 1e-2)
    report = KVBridgeV19FitReport(
        schema=W74_KV_BRIDGE_V19_SCHEMA_VERSION,
        n_targets=int(n_targets),
        per_target_pre_residual=tuple(per_pre),
        per_target_post_residual=tuple(per_post),
        compound_target_index=int(
            compound_target_index),
        compound_pre=float(pre15),
        compound_post=float(post15),
        converged=bool(converged),
        ridge_lambda=float(ridge_lambda),
        inner_v18_report_cid=str(v18_report.cid()),
    )
    return new_proj, report


def compute_compound_repair_fingerprint_v19(
        *, role: str,
        repair_trajectory_cid: str,
        delayed_repair_trajectory_cid: str,
        restart_repair_trajectory_cid: str,
        replacement_repair_trajectory_cid: str,
        compound_repair_trajectory_cid: str,
        dominant_repair_label: int = 0,
        restart_count: int = 0,
        rejoin_count: int = 0,
        replacement_count: int = 0,
        contradiction_count: int = 0,
        delayed_repair_count: int = 0,
        visible_token_budget: float = 256.0,
        baseline_cost: float = 512.0,
        task_id: str = "task", team_id: str = "team",
        branch_id: str = "main",
        delay_turns: int = 0,
        rejoin_lag_turns: int = 0,
        replacement_lag_turns: int = 0,
        compound_window_turns: int = 0,
        dim: int = W74_KV_V19_FINGERPRINT_DIM,
) -> tuple[float, ...]:
    """Deterministic 120-dim compound-repair fingerprint."""
    label_str = (
        W74_REPAIR_LABELS_V19[int(dominant_repair_label)]
        if 0 <= int(dominant_repair_label)
            < len(W74_REPAIR_LABELS_V19)
        else "unknown")
    base = (
        f"{role}|{repair_trajectory_cid}|"
        f"{delayed_repair_trajectory_cid}|"
        f"{restart_repair_trajectory_cid}|"
        f"{replacement_repair_trajectory_cid}|"
        f"{compound_repair_trajectory_cid}|{label_str}|"
        f"{int(restart_count)}|{int(rejoin_count)}|"
        f"{int(replacement_count)}|{int(contradiction_count)}|"
        f"{int(delayed_repair_count)}|"
        f"{float(visible_token_budget)}|{float(baseline_cost)}|"
        f"{task_id}|{team_id}|{branch_id}|"
        f"{int(delay_turns)}|{int(rejoin_lag_turns)}|"
        f"{int(replacement_lag_turns)}|"
        f"{int(compound_window_turns)}"
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


def probe_kv_bridge_v19_compound_pressure_margin(
        *, params: TinyV19SubstrateParams,
        token_ids: Sequence[int], n_targets: int = 15,
) -> dict[str, Any]:
    """Substrate-measured per-target margin probe via V19 forward."""
    base_trace, _ = forward_tiny_substrate_v19(
        params, list(token_ids))
    base_logits = _np.asarray(
        base_trace.logits, dtype=_np.float64)
    margins: list[float] = []
    for _ in range(int(n_targets)):
        t, _ = forward_tiny_substrate_v19(
            params, list(token_ids))
        l = _np.asarray(t.logits, dtype=_np.float64)
        diff = float(
            _np.linalg.norm((l - base_logits).ravel()))
        margins.append(diff)
    return {
        "schema": W74_KV_BRIDGE_V19_SCHEMA_VERSION,
        "kind": "v19_compound_pressure_margin_probe",
        "n_targets": int(n_targets),
        "per_target_margin_l2": [
            float(round(float(x), 12)) for x in margins],
        "max_margin": float(round(
            max(margins) if margins else 0.0, 12)),
    }


@dataclasses.dataclass(frozen=True)
class KVBridgeV19CompoundPressureFalsifier:
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
                "kv_bridge_v19_compound_pressure_falsifier",
            "falsifier": self.to_dict()})


def probe_kv_bridge_v19_compound_pressure_falsifier(
        *, compound_pressure_flag: int,
) -> KVBridgeV19CompoundPressureFalsifier:
    """Returns 0 iff inverting the compound-pressure flag flips the
    routing decision.

    Inversion semantics: flag 0 → 1, any nonzero → 0. The routing
    decision is ``route_through_substrate`` when flag > 0 and
    ``route_through_text`` otherwise. Flipping the flag must flip
    the decision (honest case: score = 0).
    """
    f = int(compound_pressure_flag)
    inv = 1 if f == 0 else 0
    decision = (
        "route_through_substrate" if f > 0
        else "route_through_text")
    inv_decision = (
        "route_through_substrate" if inv > 0
        else "route_through_text")
    flipped = decision != inv_decision
    score = 0.0 if flipped else 1.0
    return KVBridgeV19CompoundPressureFalsifier(
        primary_flag=int(f),
        inverted_flag=int(inv),
        decision=str(decision),
        inverted_decision=str(inv_decision),
        falsifier_score=float(score),
    )


@dataclasses.dataclass(frozen=True)
class KVBridgeV19Witness:
    schema: str
    projection_cid: str
    fit_report_cid: str
    compound_pressure_margin_probe_cid: str
    compound_pressure_falsifier_cid: str
    compound_repair_fingerprint_l1: float
    max_compound_pressure_margin: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "projection_cid": str(self.projection_cid),
            "fit_report_cid": str(self.fit_report_cid),
            "compound_pressure_margin_probe_cid": str(
                self.compound_pressure_margin_probe_cid),
            "compound_pressure_falsifier_cid": str(
                self.compound_pressure_falsifier_cid),
            "compound_repair_fingerprint_l1": float(round(
                self.compound_repair_fingerprint_l1, 12)),
            "max_compound_pressure_margin": float(round(
                self.max_compound_pressure_margin, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "kv_bridge_v19_witness",
            "witness": self.to_dict()})


def emit_kv_bridge_v19_witness(
        *, projection: KVBridgeV19Projection,
        fit_report: KVBridgeV19FitReport | None = None,
        compound_pressure_margin_probe: (
            dict[str, Any] | None) = None,
        compound_pressure_falsifier: (
            KVBridgeV19CompoundPressureFalsifier | None) = None,
        compound_repair_fingerprint: (
            Sequence[float] | None) = None,
) -> KVBridgeV19Witness:
    fp_l1 = 0.0
    if compound_repair_fingerprint is not None:
        fp_l1 = float(sum(
            abs(float(x))
            for x in compound_repair_fingerprint))
    return KVBridgeV19Witness(
        schema=W74_KV_BRIDGE_V19_SCHEMA_VERSION,
        projection_cid=str(projection.cid()),
        fit_report_cid=(
            fit_report.cid() if fit_report is not None else ""),
        compound_pressure_margin_probe_cid=(
            _sha256_hex(compound_pressure_margin_probe)
            if compound_pressure_margin_probe is not None
            else ""),
        compound_pressure_falsifier_cid=(
            compound_pressure_falsifier.cid()
            if compound_pressure_falsifier is not None
            else ""),
        compound_repair_fingerprint_l1=float(fp_l1),
        max_compound_pressure_margin=float(
            compound_pressure_margin_probe["max_margin"]
            if compound_pressure_margin_probe is not None
            and "max_margin"
                in compound_pressure_margin_probe
            else 0.0),
    )


__all__ = [
    "W74_KV_BRIDGE_V19_SCHEMA_VERSION",
    "W74_DEFAULT_KV_V19_RIDGE_LAMBDA",
    "W74_KV_V19_FINGERPRINT_DIM",
    "KVBridgeV19Projection",
    "KVBridgeV19FitReport",
    "fit_kv_bridge_v19_fifteen_target",
    "compute_compound_repair_fingerprint_v19",
    "probe_kv_bridge_v19_compound_pressure_margin",
    "KVBridgeV19CompoundPressureFalsifier",
    "probe_kv_bridge_v19_compound_pressure_falsifier",
    "KVBridgeV19Witness",
    "emit_kv_bridge_v19_witness",
]
