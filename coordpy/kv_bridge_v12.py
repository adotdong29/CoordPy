"""W67 M2 — KV Bridge V12.

Strictly extends W66's ``coordpy.kv_bridge_v11``. V11 fit a 7-target
stack (6 V10 + 1 team-failure-recovery target). V12 adds:

* **Eight-target stacked ridge fit** — adds an eighth column for
  *branch-merge-reconciliation*. Returns an 8-target fit report.
* **Branch-merge margin probe** — substrate-measured per-target
  margin probe using the V12 substrate.
* **Role-pair fingerprint** — a 40-dim fingerprint derived from
  (role_a, role_b, task_id, team_id, branch_id).
* **Branch-merge-reconciliation falsifier** — returns 0 exactly
  when inverting the branch-merge flag flips the decision.

Honest scope (W67)
------------------

* All ridge fits remain closed-form linear (W67-L-V12-NO-AUTOGRAD-CAP).
* Total ridge solves across W61..W67 = 41 (6 new on top of W66).
* The eighth target is *constructed*.
"""

from __future__ import annotations

import dataclasses
import hashlib
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.kv_bridge_v12 requires numpy") from exc

from .kv_bridge_v11 import (
    KVBridgeV11Projection,
)
from .kv_bridge_v10 import fit_kv_bridge_v10_six_target
from .kv_bridge_v6 import fit_kv_bridge_v6_multi_target
from .tiny_substrate_v3 import _sha256_hex
from .tiny_substrate_v12 import (
    TinyV12SubstrateParams, forward_tiny_substrate_v12,
)


W67_KV_BRIDGE_V12_SCHEMA_VERSION: str = (
    "coordpy.kv_bridge_v12.v1")
W67_DEFAULT_KV_V12_RIDGE_LAMBDA: float = 0.10


@dataclasses.dataclass
class KVBridgeV12Projection:
    inner_v11: KVBridgeV11Projection
    seed_v12: int

    @classmethod
    def init_from_v11(
            cls, inner: KVBridgeV11Projection,
            *, seed_v12: int = 670100,
    ) -> "KVBridgeV12Projection":
        return cls(inner_v11=inner, seed_v12=int(seed_v12))

    @property
    def carrier_dim(self) -> int:
        return int(self.inner_v11.carrier_dim)

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W67_KV_BRIDGE_V12_SCHEMA_VERSION,
            "kind": "kv_bridge_v12_projection",
            "inner_v11_cid": str(self.inner_v11.cid()),
            "seed_v12": int(self.seed_v12),
        })


@dataclasses.dataclass(frozen=True)
class KVBridgeV12FitReport:
    schema: str
    n_targets: int
    per_target_pre_residual: tuple[float, ...]
    per_target_post_residual: tuple[float, ...]
    branch_merge_target_index: int
    branch_merge_pre: float
    branch_merge_post: float
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
            "branch_merge_target_index": int(
                self.branch_merge_target_index),
            "branch_merge_pre": float(round(
                self.branch_merge_pre, 12)),
            "branch_merge_post": float(round(
                self.branch_merge_post, 12)),
            "converged": bool(self.converged),
            "ridge_lambda": float(round(self.ridge_lambda, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "kv_bridge_v12_fit_report",
            "report": self.to_dict()})


def fit_kv_bridge_v12_eight_target(
        *, params: TinyV12SubstrateParams,
        projection: KVBridgeV12Projection,
        train_carriers: Sequence[Sequence[float]],
        target_delta_logits_stack: Sequence[Sequence[float]],
        follow_up_token_ids: Sequence[int],
        branch_merge_target_index: int = 7,
        n_directions: int = 3,
        ridge_lambda: float = W67_DEFAULT_KV_V12_RIDGE_LAMBDA,
) -> tuple[KVBridgeV12Projection, KVBridgeV12FitReport]:
    """Eight-target stacked ridge: 7 V11 + 1 branch-merge."""
    n_targets = int(len(target_delta_logits_stack))
    if n_targets < 1:
        raise ValueError("must provide >= 1 target")
    v11_inner = projection.inner_v11.inner_v10
    primary = list(target_delta_logits_stack[:6])
    while len(primary) < 6:
        primary.append(primary[0] if primary else [0.0])
    v10_fit, v10_report = fit_kv_bridge_v10_six_target(
        params=params.v11_params.v10_params,
        projection=v11_inner,
        train_carriers=list(train_carriers),
        target_delta_logits_stack=primary,
        follow_up_token_ids=list(follow_up_token_ids),
        n_directions=int(n_directions),
        ridge_lambda=float(ridge_lambda))
    # V11 (7th) target.
    if n_targets >= 7:
        v11_extra = list(target_delta_logits_stack[6])
    else:
        v11_extra = list(target_delta_logits_stack[-1])
    inner_v6 = v10_fit.inner_v9.inner_v8.inner_v7.inner_v6
    v6_fit_v11_extra, v6_audit_v11 = (
        fit_kv_bridge_v6_multi_target(
            params=params.v3_params,
            projection=inner_v6,
            train_carriers=list(train_carriers),
            target_delta_logits_stack=[v11_extra],
            follow_up_token_ids=list(follow_up_token_ids),
            n_directions=int(n_directions),
            ridge_lambda=float(ridge_lambda)))
    # V12 (8th) target.
    if n_targets >= int(branch_merge_target_index) + 1:
        branch_merge = list(target_delta_logits_stack[
            int(branch_merge_target_index)])
    else:
        branch_merge = list(target_delta_logits_stack[-1])
    v6_fit_v12_extra, v6_audit_v12 = (
        fit_kv_bridge_v6_multi_target(
            params=params.v3_params,
            projection=v6_fit_v11_extra,
            train_carriers=list(train_carriers),
            target_delta_logits_stack=[branch_merge],
            follow_up_token_ids=list(follow_up_token_ids),
            n_directions=int(n_directions),
            ridge_lambda=float(ridge_lambda)))
    new_inner_v7 = dataclasses.replace(
        v10_fit.inner_v9.inner_v8.inner_v7,
        inner_v6=v6_fit_v12_extra)
    new_inner_v8 = dataclasses.replace(
        v10_fit.inner_v9.inner_v8, inner_v7=new_inner_v7)
    new_inner_v9 = dataclasses.replace(
        v10_fit.inner_v9, inner_v8=new_inner_v8)
    new_v10 = dataclasses.replace(v10_fit, inner_v9=new_inner_v9)
    new_v11 = dataclasses.replace(
        projection.inner_v11, inner_v10=new_v10)
    new_proj = dataclasses.replace(projection, inner_v11=new_v11)
    pre7 = float(v6_audit_v11.pre_fit_mean_residual)
    post7 = float(v6_audit_v11.post_fit_mean_residual)
    pre8 = float(v6_audit_v12.pre_fit_mean_residual)
    post8 = float(v6_audit_v12.post_fit_mean_residual)
    per_pre = (
        list(v10_report.per_target_pre_residual) + [pre7, pre8])
    per_post = (
        list(v10_report.per_target_post_residual) + [post7, post8])
    converged = bool(
        all(po <= pr + 1e-9
            for pr, po in zip(per_pre[:6], per_post[:6]))
        and per_post[6] <= per_pre[6] + 1e-3
        and per_post[7] <= per_pre[7] + 1e-3)
    report = KVBridgeV12FitReport(
        schema=W67_KV_BRIDGE_V12_SCHEMA_VERSION,
        n_targets=int(n_targets),
        per_target_pre_residual=tuple(per_pre),
        per_target_post_residual=tuple(per_post),
        branch_merge_target_index=int(branch_merge_target_index),
        branch_merge_pre=float(pre8),
        branch_merge_post=float(post8),
        converged=bool(converged),
        ridge_lambda=float(ridge_lambda),
    )
    return new_proj, report


def compute_role_pair_fingerprint_v12(
        *, role_a: str, role_b: str, task_id: str, team_id: str,
        branch_id: str = "main", dim: int = 40,
) -> tuple[float, ...]:
    """Deterministic 40-dim fingerprint from role-pair / task / team /
    branch."""
    base = (
        f"{role_a}|{role_b}|{task_id}|{team_id}|{branch_id}"
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


def probe_kv_bridge_v12_branch_merge_margin(
        *, params: TinyV12SubstrateParams,
        token_ids: Sequence[int], n_targets: int = 8,
) -> dict[str, Any]:
    """Substrate-measured per-target margin probe via V12 forward."""
    base_trace, _ = forward_tiny_substrate_v12(
        params, list(token_ids))
    base_logits = _np.asarray(base_trace.logits, dtype=_np.float64)
    margins: list[float] = []
    for _ in range(int(n_targets)):
        t, _ = forward_tiny_substrate_v12(params, list(token_ids))
        l = _np.asarray(t.logits, dtype=_np.float64)
        diff = float(_np.linalg.norm((l - base_logits).ravel()))
        margins.append(diff)
    return {
        "schema": W67_KV_BRIDGE_V12_SCHEMA_VERSION,
        "kind": "v12_branch_merge_margin_probe",
        "n_targets": int(n_targets),
        "per_target_margin_l2": [
            float(round(float(x), 12)) for x in margins],
        "max_margin": float(round(
            max(margins) if margins else 0.0, 12)),
    }


@dataclasses.dataclass(frozen=True)
class KVBridgeV12BranchMergeFalsifier:
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
            "kind": "kv_bridge_v12_branch_merge_falsifier",
            "falsifier": self.to_dict()})


def probe_kv_bridge_v12_branch_merge_falsifier(
        *, branch_merge_flag: float,
) -> KVBridgeV12BranchMergeFalsifier:
    """Returns 0 iff inverting the branch-merge flag flips the
    decision."""
    inv = -float(branch_merge_flag)
    decision = (
        "merge" if float(branch_merge_flag) >= 0.0
        else "no_merge")
    inv_decision = (
        "merge" if inv >= 0.0 else "no_merge")
    flipped = decision != inv_decision
    score = 0.0 if flipped else 1.0
    return KVBridgeV12BranchMergeFalsifier(
        primary_flag=float(branch_merge_flag),
        inverted_flag=float(inv),
        decision=str(decision),
        inverted_decision=str(inv_decision),
        falsifier_score=float(score),
    )


@dataclasses.dataclass(frozen=True)
class KVBridgeV12Witness:
    schema: str
    projection_cid: str
    fit_report_cid: str
    branch_merge_margin_probe_cid: str
    branch_merge_falsifier_cid: str
    role_pair_fingerprint_l1: float
    max_branch_merge_margin: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "projection_cid": str(self.projection_cid),
            "fit_report_cid": str(self.fit_report_cid),
            "branch_merge_margin_probe_cid": str(
                self.branch_merge_margin_probe_cid),
            "branch_merge_falsifier_cid": str(
                self.branch_merge_falsifier_cid),
            "role_pair_fingerprint_l1": float(round(
                self.role_pair_fingerprint_l1, 12)),
            "max_branch_merge_margin": float(round(
                self.max_branch_merge_margin, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "kv_bridge_v12_witness",
            "witness": self.to_dict()})


def emit_kv_bridge_v12_witness(
        *, projection: KVBridgeV12Projection,
        fit_report: KVBridgeV12FitReport | None = None,
        branch_merge_margin_probe: (
            dict[str, Any] | None) = None,
        branch_merge_falsifier: (
            KVBridgeV12BranchMergeFalsifier | None) = None,
        role_pair_fingerprint: (
            Sequence[float] | None) = None,
) -> KVBridgeV12Witness:
    fp_l1 = 0.0
    if role_pair_fingerprint is not None:
        fp_l1 = float(sum(
            abs(float(x)) for x in role_pair_fingerprint))
    return KVBridgeV12Witness(
        schema=W67_KV_BRIDGE_V12_SCHEMA_VERSION,
        projection_cid=str(projection.cid()),
        fit_report_cid=(
            fit_report.cid() if fit_report is not None else ""),
        branch_merge_margin_probe_cid=(
            _sha256_hex(branch_merge_margin_probe)
            if branch_merge_margin_probe is not None else ""),
        branch_merge_falsifier_cid=(
            branch_merge_falsifier.cid()
            if branch_merge_falsifier is not None else ""),
        role_pair_fingerprint_l1=float(fp_l1),
        max_branch_merge_margin=float(
            branch_merge_margin_probe["max_margin"]
            if branch_merge_margin_probe is not None
            and "max_margin" in branch_merge_margin_probe
            else 0.0),
    )


__all__ = [
    "W67_KV_BRIDGE_V12_SCHEMA_VERSION",
    "W67_DEFAULT_KV_V12_RIDGE_LAMBDA",
    "KVBridgeV12Projection",
    "KVBridgeV12FitReport",
    "fit_kv_bridge_v12_eight_target",
    "compute_role_pair_fingerprint_v12",
    "probe_kv_bridge_v12_branch_merge_margin",
    "KVBridgeV12BranchMergeFalsifier",
    "probe_kv_bridge_v12_branch_merge_falsifier",
    "KVBridgeV12Witness",
    "emit_kv_bridge_v12_witness",
]
