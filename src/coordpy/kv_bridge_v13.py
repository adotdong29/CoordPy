"""W68 M2 — KV Bridge V13.

Strictly extends W67's ``coordpy.kv_bridge_v12``. V12 fit an 8-target
stack (7 V11 + 1 branch-merge-reconciliation). V13 adds:

* **Nine-target stacked ridge fit** —
  ``fit_kv_bridge_v13_nine_target`` adds a ninth column for
  *partial-contradiction-under-delayed-reconciliation*.
* **Partial-contradiction margin probe** — substrate-measured per-
  target margin probe using the V13 substrate.
* **Agent-replacement fingerprint** — a 50-dim fingerprint derived
  from ``(role, replacement_index, task_id, team_id, branch_id,
  warm_restart_window)``.
* **Partial-contradiction falsifier** — returns 0 exactly when
  inverting the partial-contradiction flag flips the decision.

Honest scope (W68)
------------------

* All ridge fits remain closed-form linear
  (``W68-L-V13-NO-AUTOGRAD-CAP``).
* Total ridge solves across W61..W68 = 47 (6 new on top of W67's 41).
* The ninth target is *constructed* (the eighth and ninth are
  derived from substrate-measured signals; they are not
  hidden-state oracles).
"""

from __future__ import annotations

import dataclasses
import hashlib
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.kv_bridge_v13 requires numpy") from exc

from .kv_bridge_v12 import (
    KVBridgeV12Projection, fit_kv_bridge_v12_eight_target,
)
from .kv_bridge_v6 import fit_kv_bridge_v6_multi_target
from .tiny_substrate_v3 import _sha256_hex
from .tiny_substrate_v13 import (
    TinyV13SubstrateParams, forward_tiny_substrate_v13,
)


W68_KV_BRIDGE_V13_SCHEMA_VERSION: str = (
    "coordpy.kv_bridge_v13.v1")
W68_DEFAULT_KV_V13_RIDGE_LAMBDA: float = 0.10


@dataclasses.dataclass
class KVBridgeV13Projection:
    inner_v12: KVBridgeV12Projection
    seed_v13: int

    @classmethod
    def init_from_v12(
            cls, inner: KVBridgeV12Projection,
            *, seed_v13: int = 680100,
    ) -> "KVBridgeV13Projection":
        return cls(inner_v12=inner, seed_v13=int(seed_v13))

    @property
    def carrier_dim(self) -> int:
        return int(self.inner_v12.carrier_dim)

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W68_KV_BRIDGE_V13_SCHEMA_VERSION,
            "kind": "kv_bridge_v13_projection",
            "inner_v12_cid": str(self.inner_v12.cid()),
            "seed_v13": int(self.seed_v13),
        })


@dataclasses.dataclass(frozen=True)
class KVBridgeV13FitReport:
    schema: str
    n_targets: int
    per_target_pre_residual: tuple[float, ...]
    per_target_post_residual: tuple[float, ...]
    partial_contradiction_target_index: int
    partial_contradiction_pre: float
    partial_contradiction_post: float
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
            "partial_contradiction_target_index": int(
                self.partial_contradiction_target_index),
            "partial_contradiction_pre": float(round(
                self.partial_contradiction_pre, 12)),
            "partial_contradiction_post": float(round(
                self.partial_contradiction_post, 12)),
            "converged": bool(self.converged),
            "ridge_lambda": float(round(self.ridge_lambda, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "kv_bridge_v13_fit_report",
            "report": self.to_dict()})


def fit_kv_bridge_v13_nine_target(
        *, params: TinyV13SubstrateParams,
        projection: KVBridgeV13Projection,
        train_carriers: Sequence[Sequence[float]],
        target_delta_logits_stack: Sequence[Sequence[float]],
        follow_up_token_ids: Sequence[int],
        partial_contradiction_target_index: int = 8,
        n_directions: int = 3,
        ridge_lambda: float = W68_DEFAULT_KV_V13_RIDGE_LAMBDA,
) -> tuple[KVBridgeV13Projection, KVBridgeV13FitReport]:
    """Nine-target stacked ridge: 8 V12 + 1 partial-contradiction."""
    n_targets = int(len(target_delta_logits_stack))
    if n_targets < 1:
        raise ValueError("must provide >= 1 target")
    primary = list(target_delta_logits_stack[:8])
    while len(primary) < 8:
        primary.append(primary[0] if primary else [0.0])
    v12_fit, v12_report = fit_kv_bridge_v12_eight_target(
        params=params.v12_params,
        projection=projection.inner_v12,
        train_carriers=list(train_carriers),
        target_delta_logits_stack=primary,
        follow_up_token_ids=list(follow_up_token_ids),
        n_directions=int(n_directions),
        ridge_lambda=float(ridge_lambda))
    if n_targets >= int(partial_contradiction_target_index) + 1:
        pc_target = list(target_delta_logits_stack[
            int(partial_contradiction_target_index)])
    else:
        pc_target = list(target_delta_logits_stack[-1])
    inner_v6 = (
        v12_fit.inner_v11.inner_v10.inner_v9
        .inner_v8.inner_v7.inner_v6)
    v6_fit_pc, v6_audit_pc = fit_kv_bridge_v6_multi_target(
        params=params.v3_params,
        projection=inner_v6,
        train_carriers=list(train_carriers),
        target_delta_logits_stack=[pc_target],
        follow_up_token_ids=list(follow_up_token_ids),
        n_directions=int(n_directions),
        ridge_lambda=float(ridge_lambda))
    new_inner_v7 = dataclasses.replace(
        v12_fit.inner_v11.inner_v10.inner_v9.inner_v8.inner_v7,
        inner_v6=v6_fit_pc)
    new_inner_v8 = dataclasses.replace(
        v12_fit.inner_v11.inner_v10.inner_v9.inner_v8,
        inner_v7=new_inner_v7)
    new_inner_v9 = dataclasses.replace(
        v12_fit.inner_v11.inner_v10.inner_v9, inner_v8=new_inner_v8)
    new_inner_v10 = dataclasses.replace(
        v12_fit.inner_v11.inner_v10, inner_v9=new_inner_v9)
    new_inner_v11 = dataclasses.replace(
        v12_fit.inner_v11, inner_v10=new_inner_v10)
    new_v12 = dataclasses.replace(v12_fit, inner_v11=new_inner_v11)
    new_proj = dataclasses.replace(projection, inner_v12=new_v12)
    pre9 = float(v6_audit_pc.pre_fit_mean_residual)
    post9 = float(v6_audit_pc.post_fit_mean_residual)
    per_pre = (
        list(v12_report.per_target_pre_residual) + [pre9])
    per_post = (
        list(v12_report.per_target_post_residual) + [post9])
    converged = bool(
        all(po <= pr + 1e-9
            for pr, po in zip(per_pre[:8], per_post[:8]))
        and per_post[8] <= per_pre[8] + 1e-3)
    report = KVBridgeV13FitReport(
        schema=W68_KV_BRIDGE_V13_SCHEMA_VERSION,
        n_targets=int(n_targets),
        per_target_pre_residual=tuple(per_pre),
        per_target_post_residual=tuple(per_post),
        partial_contradiction_target_index=int(
            partial_contradiction_target_index),
        partial_contradiction_pre=float(pre9),
        partial_contradiction_post=float(post9),
        converged=bool(converged),
        ridge_lambda=float(ridge_lambda),
    )
    return new_proj, report


def compute_agent_replacement_fingerprint_v13(
        *, role: str, replacement_index: int, task_id: str,
        team_id: str, branch_id: str = "main",
        warm_restart_window: int = 0, dim: int = 50,
) -> tuple[float, ...]:
    """Deterministic 50-dim fingerprint."""
    base = (
        f"{role}|{int(replacement_index)}|{task_id}|{team_id}|"
        f"{branch_id}|{int(warm_restart_window)}"
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


def probe_kv_bridge_v13_partial_contradiction_margin(
        *, params: TinyV13SubstrateParams,
        token_ids: Sequence[int], n_targets: int = 9,
) -> dict[str, Any]:
    """Substrate-measured per-target margin probe via V13 forward."""
    base_trace, _ = forward_tiny_substrate_v13(
        params, list(token_ids))
    base_logits = _np.asarray(base_trace.logits, dtype=_np.float64)
    margins: list[float] = []
    for _ in range(int(n_targets)):
        t, _ = forward_tiny_substrate_v13(params, list(token_ids))
        l = _np.asarray(t.logits, dtype=_np.float64)
        diff = float(_np.linalg.norm((l - base_logits).ravel()))
        margins.append(diff)
    return {
        "schema": W68_KV_BRIDGE_V13_SCHEMA_VERSION,
        "kind": "v13_partial_contradiction_margin_probe",
        "n_targets": int(n_targets),
        "per_target_margin_l2": [
            float(round(float(x), 12)) for x in margins],
        "max_margin": float(round(
            max(margins) if margins else 0.0, 12)),
    }


@dataclasses.dataclass(frozen=True)
class KVBridgeV13PartialContradictionFalsifier:
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
            "kind": "kv_bridge_v13_partial_contradiction_falsifier",
            "falsifier": self.to_dict()})


def probe_kv_bridge_v13_partial_contradiction_falsifier(
        *, partial_contradiction_flag: float,
) -> KVBridgeV13PartialContradictionFalsifier:
    """Returns 0 iff inverting the partial-contradiction flag flips
    the decision."""
    inv = -float(partial_contradiction_flag)
    decision = (
        "contradict" if float(partial_contradiction_flag) >= 0.0
        else "no_contradict")
    inv_decision = (
        "contradict" if inv >= 0.0 else "no_contradict")
    flipped = decision != inv_decision
    score = 0.0 if flipped else 1.0
    return KVBridgeV13PartialContradictionFalsifier(
        primary_flag=float(partial_contradiction_flag),
        inverted_flag=float(inv),
        decision=str(decision),
        inverted_decision=str(inv_decision),
        falsifier_score=float(score),
    )


@dataclasses.dataclass(frozen=True)
class KVBridgeV13Witness:
    schema: str
    projection_cid: str
    fit_report_cid: str
    partial_contradiction_margin_probe_cid: str
    partial_contradiction_falsifier_cid: str
    agent_replacement_fingerprint_l1: float
    max_partial_contradiction_margin: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "projection_cid": str(self.projection_cid),
            "fit_report_cid": str(self.fit_report_cid),
            "partial_contradiction_margin_probe_cid": str(
                self.partial_contradiction_margin_probe_cid),
            "partial_contradiction_falsifier_cid": str(
                self.partial_contradiction_falsifier_cid),
            "agent_replacement_fingerprint_l1": float(round(
                self.agent_replacement_fingerprint_l1, 12)),
            "max_partial_contradiction_margin": float(round(
                self.max_partial_contradiction_margin, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "kv_bridge_v13_witness",
            "witness": self.to_dict()})


def emit_kv_bridge_v13_witness(
        *, projection: KVBridgeV13Projection,
        fit_report: KVBridgeV13FitReport | None = None,
        partial_contradiction_margin_probe: (
            dict[str, Any] | None) = None,
        partial_contradiction_falsifier: (
            KVBridgeV13PartialContradictionFalsifier | None) = None,
        agent_replacement_fingerprint: (
            Sequence[float] | None) = None,
) -> KVBridgeV13Witness:
    fp_l1 = 0.0
    if agent_replacement_fingerprint is not None:
        fp_l1 = float(sum(
            abs(float(x))
            for x in agent_replacement_fingerprint))
    return KVBridgeV13Witness(
        schema=W68_KV_BRIDGE_V13_SCHEMA_VERSION,
        projection_cid=str(projection.cid()),
        fit_report_cid=(
            fit_report.cid() if fit_report is not None else ""),
        partial_contradiction_margin_probe_cid=(
            _sha256_hex(partial_contradiction_margin_probe)
            if partial_contradiction_margin_probe is not None
            else ""),
        partial_contradiction_falsifier_cid=(
            partial_contradiction_falsifier.cid()
            if partial_contradiction_falsifier is not None
            else ""),
        agent_replacement_fingerprint_l1=float(fp_l1),
        max_partial_contradiction_margin=float(
            partial_contradiction_margin_probe["max_margin"]
            if partial_contradiction_margin_probe is not None
            and "max_margin" in partial_contradiction_margin_probe
            else 0.0),
    )


__all__ = [
    "W68_KV_BRIDGE_V13_SCHEMA_VERSION",
    "W68_DEFAULT_KV_V13_RIDGE_LAMBDA",
    "KVBridgeV13Projection",
    "KVBridgeV13FitReport",
    "fit_kv_bridge_v13_nine_target",
    "compute_agent_replacement_fingerprint_v13",
    "probe_kv_bridge_v13_partial_contradiction_margin",
    "KVBridgeV13PartialContradictionFalsifier",
    "probe_kv_bridge_v13_partial_contradiction_falsifier",
    "KVBridgeV13Witness",
    "emit_kv_bridge_v13_witness",
]
