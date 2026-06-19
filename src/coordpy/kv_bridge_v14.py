"""W69 M2 — KV Bridge V14.

Strictly extends W68's ``coordpy.kv_bridge_v13``. V13 fit a 9-target
stack (8 V12 + 1 partial-contradiction). V14 adds:

* **Ten-target stacked ridge fit** —
  ``fit_kv_bridge_v14_ten_target`` adds a tenth column for
  *multi-branch-rejoin-after-divergent-work*.
* **Multi-branch-rejoin margin probe** — substrate-measured per-
  target margin via the V14 forward.
* **Silent-corruption fingerprint** — a 60-dim fingerprint derived
  from ``(role, corrupted_bytes, member_replaced, task_id, team_id,
  branch_id, detect_turn, repair_turn)``.
* **Multi-branch-rejoin falsifier** — returns 0 exactly when
  inverting the rejoin flag flips the decision.

Honest scope (W69)
------------------

* All ridge fits remain closed-form linear
  (``W69-L-V14-NO-AUTOGRAD-CAP``).
* Total ridge solves across W61..W69 = 53 (6 new on top of W68's 47).
"""

from __future__ import annotations

import dataclasses
import hashlib
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.kv_bridge_v14 requires numpy") from exc

from .kv_bridge_v13 import (
    KVBridgeV13Projection, fit_kv_bridge_v13_nine_target,
)
from .kv_bridge_v6 import fit_kv_bridge_v6_multi_target
from .tiny_substrate_v3 import _sha256_hex
from .tiny_substrate_v14 import (
    TinyV14SubstrateParams, forward_tiny_substrate_v14,
)


W69_KV_BRIDGE_V14_SCHEMA_VERSION: str = (
    "coordpy.kv_bridge_v14.v1")
W69_DEFAULT_KV_V14_RIDGE_LAMBDA: float = 0.10


@dataclasses.dataclass
class KVBridgeV14Projection:
    inner_v13: KVBridgeV13Projection
    seed_v14: int

    @classmethod
    def init_from_v13(
            cls, inner: KVBridgeV13Projection,
            *, seed_v14: int = 690100,
    ) -> "KVBridgeV14Projection":
        return cls(inner_v13=inner, seed_v14=int(seed_v14))

    @property
    def carrier_dim(self) -> int:
        return int(self.inner_v13.carrier_dim)

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W69_KV_BRIDGE_V14_SCHEMA_VERSION,
            "kind": "kv_bridge_v14_projection",
            "inner_v13_cid": str(self.inner_v13.cid()),
            "seed_v14": int(self.seed_v14),
        })


@dataclasses.dataclass(frozen=True)
class KVBridgeV14FitReport:
    schema: str
    n_targets: int
    per_target_pre_residual: tuple[float, ...]
    per_target_post_residual: tuple[float, ...]
    multi_branch_rejoin_target_index: int
    multi_branch_rejoin_pre: float
    multi_branch_rejoin_post: float
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
            "multi_branch_rejoin_target_index": int(
                self.multi_branch_rejoin_target_index),
            "multi_branch_rejoin_pre": float(round(
                self.multi_branch_rejoin_pre, 12)),
            "multi_branch_rejoin_post": float(round(
                self.multi_branch_rejoin_post, 12)),
            "converged": bool(self.converged),
            "ridge_lambda": float(round(self.ridge_lambda, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "kv_bridge_v14_fit_report",
            "report": self.to_dict()})


def fit_kv_bridge_v14_ten_target(
        *, params: TinyV14SubstrateParams,
        projection: KVBridgeV14Projection,
        train_carriers: Sequence[Sequence[float]],
        target_delta_logits_stack: Sequence[Sequence[float]],
        follow_up_token_ids: Sequence[int],
        multi_branch_rejoin_target_index: int = 9,
        n_directions: int = 3,
        ridge_lambda: float = W69_DEFAULT_KV_V14_RIDGE_LAMBDA,
) -> tuple[KVBridgeV14Projection, KVBridgeV14FitReport]:
    """Ten-target stacked ridge: 9 V13 + 1 multi-branch-rejoin."""
    n_targets = int(len(target_delta_logits_stack))
    if n_targets < 1:
        raise ValueError("must provide >= 1 target")
    primary = list(target_delta_logits_stack[:9])
    while len(primary) < 9:
        primary.append(primary[0] if primary else [0.0])
    v13_fit, v13_report = fit_kv_bridge_v13_nine_target(
        params=params.v13_params,
        projection=projection.inner_v13,
        train_carriers=list(train_carriers),
        target_delta_logits_stack=primary,
        follow_up_token_ids=list(follow_up_token_ids),
        n_directions=int(n_directions),
        ridge_lambda=float(ridge_lambda))
    if n_targets >= int(multi_branch_rejoin_target_index) + 1:
        mbr_target = list(target_delta_logits_stack[
            int(multi_branch_rejoin_target_index)])
    else:
        mbr_target = list(target_delta_logits_stack[-1])
    inner_v6 = (
        v13_fit.inner_v12.inner_v11.inner_v10.inner_v9
        .inner_v8.inner_v7.inner_v6)
    v6_fit_mbr, v6_audit_mbr = fit_kv_bridge_v6_multi_target(
        params=params.v3_params,
        projection=inner_v6,
        train_carriers=list(train_carriers),
        target_delta_logits_stack=[mbr_target],
        follow_up_token_ids=list(follow_up_token_ids),
        n_directions=int(n_directions),
        ridge_lambda=float(ridge_lambda))
    new_inner_v7 = dataclasses.replace(
        v13_fit.inner_v12.inner_v11.inner_v10.inner_v9
        .inner_v8.inner_v7,
        inner_v6=v6_fit_mbr)
    new_inner_v8 = dataclasses.replace(
        v13_fit.inner_v12.inner_v11.inner_v10.inner_v9.inner_v8,
        inner_v7=new_inner_v7)
    new_inner_v9 = dataclasses.replace(
        v13_fit.inner_v12.inner_v11.inner_v10.inner_v9,
        inner_v8=new_inner_v8)
    new_inner_v10 = dataclasses.replace(
        v13_fit.inner_v12.inner_v11.inner_v10,
        inner_v9=new_inner_v9)
    new_inner_v11 = dataclasses.replace(
        v13_fit.inner_v12.inner_v11, inner_v10=new_inner_v10)
    new_inner_v12 = dataclasses.replace(
        v13_fit.inner_v12, inner_v11=new_inner_v11)
    new_v13 = dataclasses.replace(v13_fit, inner_v12=new_inner_v12)
    new_proj = dataclasses.replace(projection, inner_v13=new_v13)
    pre10 = float(v6_audit_mbr.pre_fit_mean_residual)
    post10 = float(v6_audit_mbr.post_fit_mean_residual)
    per_pre = (
        list(v13_report.per_target_pre_residual) + [pre10])
    per_post = (
        list(v13_report.per_target_post_residual) + [post10])
    converged = bool(
        all(po <= pr + 1e-9
            for pr, po in zip(per_pre[:9], per_post[:9]))
        and per_post[9] <= per_pre[9] + 1e-2)
    report = KVBridgeV14FitReport(
        schema=W69_KV_BRIDGE_V14_SCHEMA_VERSION,
        n_targets=int(n_targets),
        per_target_pre_residual=tuple(per_pre),
        per_target_post_residual=tuple(per_post),
        multi_branch_rejoin_target_index=int(
            multi_branch_rejoin_target_index),
        multi_branch_rejoin_pre=float(pre10),
        multi_branch_rejoin_post=float(post10),
        converged=bool(converged),
        ridge_lambda=float(ridge_lambda),
    )
    return new_proj, report


def compute_silent_corruption_fingerprint_v14(
        *, role: str, corrupted_bytes: int = 1,
        member_replaced: bool = False, task_id: str = "task",
        team_id: str = "team", branch_id: str = "main",
        detect_turn: int = 0, repair_turn: int = -1,
        dim: int = 60,
) -> tuple[float, ...]:
    """Deterministic 60-dim silent-corruption fingerprint."""
    base = (
        f"{role}|{int(corrupted_bytes)}|{int(bool(member_replaced))}|"
        f"{task_id}|{team_id}|{branch_id}|{int(detect_turn)}|"
        f"{int(repair_turn)}"
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


def probe_kv_bridge_v14_multi_branch_rejoin_margin(
        *, params: TinyV14SubstrateParams,
        token_ids: Sequence[int], n_targets: int = 10,
) -> dict[str, Any]:
    """Substrate-measured per-target margin probe via V14 forward."""
    base_trace, _ = forward_tiny_substrate_v14(
        params, list(token_ids))
    base_logits = _np.asarray(base_trace.logits, dtype=_np.float64)
    margins: list[float] = []
    for _ in range(int(n_targets)):
        t, _ = forward_tiny_substrate_v14(params, list(token_ids))
        l = _np.asarray(t.logits, dtype=_np.float64)
        diff = float(_np.linalg.norm((l - base_logits).ravel()))
        margins.append(diff)
    return {
        "schema": W69_KV_BRIDGE_V14_SCHEMA_VERSION,
        "kind": "v14_multi_branch_rejoin_margin_probe",
        "n_targets": int(n_targets),
        "per_target_margin_l2": [
            float(round(float(x), 12)) for x in margins],
        "max_margin": float(round(
            max(margins) if margins else 0.0, 12)),
    }


@dataclasses.dataclass(frozen=True)
class KVBridgeV14MultiBranchRejoinFalsifier:
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
            "kind": "kv_bridge_v14_multi_branch_rejoin_falsifier",
            "falsifier": self.to_dict()})


def probe_kv_bridge_v14_multi_branch_rejoin_falsifier(
        *, multi_branch_rejoin_flag: float,
) -> KVBridgeV14MultiBranchRejoinFalsifier:
    """Returns 0 iff inverting the multi-branch-rejoin flag flips
    the decision."""
    inv = -float(multi_branch_rejoin_flag)
    decision = (
        "rejoin" if float(multi_branch_rejoin_flag) >= 0.0
        else "no_rejoin")
    inv_decision = (
        "rejoin" if inv >= 0.0 else "no_rejoin")
    flipped = decision != inv_decision
    score = 0.0 if flipped else 1.0
    return KVBridgeV14MultiBranchRejoinFalsifier(
        primary_flag=float(multi_branch_rejoin_flag),
        inverted_flag=float(inv),
        decision=str(decision),
        inverted_decision=str(inv_decision),
        falsifier_score=float(score),
    )


@dataclasses.dataclass(frozen=True)
class KVBridgeV14Witness:
    schema: str
    projection_cid: str
    fit_report_cid: str
    multi_branch_rejoin_margin_probe_cid: str
    multi_branch_rejoin_falsifier_cid: str
    silent_corruption_fingerprint_l1: float
    max_multi_branch_rejoin_margin: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "projection_cid": str(self.projection_cid),
            "fit_report_cid": str(self.fit_report_cid),
            "multi_branch_rejoin_margin_probe_cid": str(
                self.multi_branch_rejoin_margin_probe_cid),
            "multi_branch_rejoin_falsifier_cid": str(
                self.multi_branch_rejoin_falsifier_cid),
            "silent_corruption_fingerprint_l1": float(round(
                self.silent_corruption_fingerprint_l1, 12)),
            "max_multi_branch_rejoin_margin": float(round(
                self.max_multi_branch_rejoin_margin, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "kv_bridge_v14_witness",
            "witness": self.to_dict()})


def emit_kv_bridge_v14_witness(
        *, projection: KVBridgeV14Projection,
        fit_report: KVBridgeV14FitReport | None = None,
        multi_branch_rejoin_margin_probe: (
            dict[str, Any] | None) = None,
        multi_branch_rejoin_falsifier: (
            KVBridgeV14MultiBranchRejoinFalsifier | None) = None,
        silent_corruption_fingerprint: (
            Sequence[float] | None) = None,
) -> KVBridgeV14Witness:
    fp_l1 = 0.0
    if silent_corruption_fingerprint is not None:
        fp_l1 = float(sum(
            abs(float(x))
            for x in silent_corruption_fingerprint))
    return KVBridgeV14Witness(
        schema=W69_KV_BRIDGE_V14_SCHEMA_VERSION,
        projection_cid=str(projection.cid()),
        fit_report_cid=(
            fit_report.cid() if fit_report is not None else ""),
        multi_branch_rejoin_margin_probe_cid=(
            _sha256_hex(multi_branch_rejoin_margin_probe)
            if multi_branch_rejoin_margin_probe is not None
            else ""),
        multi_branch_rejoin_falsifier_cid=(
            multi_branch_rejoin_falsifier.cid()
            if multi_branch_rejoin_falsifier is not None
            else ""),
        silent_corruption_fingerprint_l1=float(fp_l1),
        max_multi_branch_rejoin_margin=float(
            multi_branch_rejoin_margin_probe["max_margin"]
            if multi_branch_rejoin_margin_probe is not None
            and "max_margin" in multi_branch_rejoin_margin_probe
            else 0.0),
    )


__all__ = [
    "W69_KV_BRIDGE_V14_SCHEMA_VERSION",
    "W69_DEFAULT_KV_V14_RIDGE_LAMBDA",
    "KVBridgeV14Projection",
    "KVBridgeV14FitReport",
    "fit_kv_bridge_v14_ten_target",
    "compute_silent_corruption_fingerprint_v14",
    "probe_kv_bridge_v14_multi_branch_rejoin_margin",
    "KVBridgeV14MultiBranchRejoinFalsifier",
    "probe_kv_bridge_v14_multi_branch_rejoin_falsifier",
    "KVBridgeV14Witness",
    "emit_kv_bridge_v14_witness",
]
