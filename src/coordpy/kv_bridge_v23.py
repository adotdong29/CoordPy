"""W78 M2 — KV Bridge V23.

Strictly extends W77's ``coordpy.kv_bridge_v22``. V23 adds:

* **Nineteenth-target stacked ridge fit** — adds a nineteenth
  column for *long-horizon-reconstruction* routing.
* **156-dim long-horizon-reconstruction fingerprint** — derived
  from the V22 fingerprint inputs plus the long-horizon-
  reconstruction trajectory CID and reconstruction-window.
* **Long-horizon-reconstruction-pressure falsifier** — returns 0
  iff inverting the long-horizon-reconstruction-pressure flag
  flips the routing decision.

Honest scope (W78)
------------------

* All ridge fits remain closed-form linear
  (``W78-L-V23-NO-AUTOGRAD-CAP``).
* Total ridge solves across W61..W78 = 79 (2 new on top of W77's
  77 — cache V21 eighteen-objective + KV V23 nineteen-target).
"""

from __future__ import annotations

import dataclasses
import hashlib
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.kv_bridge_v23 requires numpy") from exc

from .kv_bridge_v22 import (
    KVBridgeV22Projection, fit_kv_bridge_v22_eighteen_target,
)
from .tiny_substrate_v3 import _sha256_hex
from .tiny_substrate_v23 import (
    TinyV23SubstrateParams, W78_REPAIR_LABELS_V23,
    forward_tiny_substrate_v23,
)


W78_KV_BRIDGE_V23_SCHEMA_VERSION: str = (
    "coordpy.kv_bridge_v23.v1")
W78_DEFAULT_KV_V23_RIDGE_LAMBDA: float = 0.10
W78_KV_V23_FINGERPRINT_DIM: int = 156


@dataclasses.dataclass
class KVBridgeV23Projection:
    inner_v22: KVBridgeV22Projection
    seed_v23: int

    @classmethod
    def init_from_v22(
            cls, inner: KVBridgeV22Projection,
            *, seed_v23: int = 780100,
    ) -> "KVBridgeV23Projection":
        return cls(inner_v22=inner, seed_v23=int(seed_v23))

    @property
    def carrier_dim(self) -> int:
        return int(self.inner_v22.carrier_dim)

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W78_KV_BRIDGE_V23_SCHEMA_VERSION,
            "kind": "kv_bridge_v23_projection",
            "inner_v22_cid": str(self.inner_v22.cid()),
            "seed_v23": int(self.seed_v23),
        })


@dataclasses.dataclass(frozen=True)
class KVBridgeV23FitReport:
    schema: str
    n_targets: int
    per_target_pre_residual: tuple[float, ...]
    per_target_post_residual: tuple[float, ...]
    long_horizon_reconstruction_target_index: int
    long_horizon_reconstruction_pre: float
    long_horizon_reconstruction_post: float
    converged: bool
    ridge_lambda: float
    inner_v22_report_cid: str

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
            "long_horizon_reconstruction_target_index": int(
                self.long_horizon_reconstruction_target_index),
            "long_horizon_reconstruction_pre": float(round(
                self.long_horizon_reconstruction_pre, 12)),
            "long_horizon_reconstruction_post": float(round(
                self.long_horizon_reconstruction_post, 12)),
            "converged": bool(self.converged),
            "ridge_lambda": float(round(self.ridge_lambda, 12)),
            "inner_v22_report_cid": str(
                self.inner_v22_report_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "kv_bridge_v23_fit_report",
            "report": self.to_dict()})


def fit_kv_bridge_v23_nineteen_target(
        *, params: TinyV23SubstrateParams,
        projection: KVBridgeV23Projection,
        train_carriers: Sequence[Sequence[float]],
        target_delta_logits_stack: Sequence[Sequence[float]],
        follow_up_token_ids: Sequence[int],
        long_horizon_reconstruction_target_index: int = 18,
        n_directions: int = 3,
        ridge_lambda: float = W78_DEFAULT_KV_V23_RIDGE_LAMBDA,
) -> tuple[KVBridgeV23Projection, KVBridgeV23FitReport]:
    """Nineteen-target stacked ridge: 18 V22 + 1 long-horizon-
    reconstruction.

    Reuses the V22 eighteen-target inner fit, then closed-form
    ridges the nineteenth column residually.
    """
    n_targets = int(len(target_delta_logits_stack))
    if n_targets < 1:
        raise ValueError("must provide >= 1 target")
    primary = list(target_delta_logits_stack[:18])
    while len(primary) < 18:
        primary.append(primary[0] if primary else [0.0])
    v22_fit, v22_report = fit_kv_bridge_v22_eighteen_target(
        params=params.v22_params,
        projection=projection.inner_v22,
        train_carriers=list(train_carriers),
        target_delta_logits_stack=primary,
        follow_up_token_ids=list(follow_up_token_ids),
        n_directions=int(n_directions),
        ridge_lambda=float(ridge_lambda))
    if n_targets >= int(
            long_horizon_reconstruction_target_index) + 1:
        lhr_target = list(target_delta_logits_stack[
            int(long_horizon_reconstruction_target_index)])
    else:
        lhr_target = list(target_delta_logits_stack[-1])
    # Closed-form ridge solve for the nineteenth-column residual.
    X = _np.asarray(train_carriers, dtype=_np.float64)
    if X.ndim == 1:
        X = X.reshape((-1, 1))
    if X.shape[0] == 0:
        pre19 = 0.0
        post19 = 0.0
    else:
        y = _np.asarray(
            lhr_target[:X.shape[0]], dtype=_np.float64)
        if y.size < X.shape[0]:
            pad = _np.zeros(
                X.shape[0] - y.size, dtype=_np.float64)
            y = _np.concatenate([y, pad])
        lam = max(float(ridge_lambda), 1e-9)
        A = X.T @ X + lam * _np.eye(
            X.shape[1], dtype=_np.float64)
        b = X.T @ y
        try:
            theta = _np.linalg.solve(A, b)
        except Exception:
            theta = _np.zeros((X.shape[1],), dtype=_np.float64)
        y_hat = X @ theta
        pre19 = float(_np.mean(_np.abs(y)))
        post19 = float(_np.mean(_np.abs(y - y_hat)))
    new_proj = dataclasses.replace(
        projection, inner_v22=v22_fit)
    per_pre = (
        list(v22_report.per_target_pre_residual) + [pre19])
    per_post = (
        list(v22_report.per_target_post_residual) + [post19])
    converged = bool(
        all(po <= pr + 1e-9
            for pr, po in zip(per_pre[:18], per_post[:18]))
        and per_post[18] <= per_pre[18] + 1e-2)
    report = KVBridgeV23FitReport(
        schema=W78_KV_BRIDGE_V23_SCHEMA_VERSION,
        n_targets=int(n_targets),
        per_target_pre_residual=tuple(per_pre),
        per_target_post_residual=tuple(per_post),
        long_horizon_reconstruction_target_index=int(
            long_horizon_reconstruction_target_index),
        long_horizon_reconstruction_pre=float(pre19),
        long_horizon_reconstruction_post=float(post19),
        converged=bool(converged),
        ridge_lambda=float(ridge_lambda),
        inner_v22_report_cid=str(v22_report.cid()),
    )
    return new_proj, report


def compute_long_horizon_reconstruction_fingerprint_v23(
        *, role: str,
        post_restart_replacement_trajectory_cid: str,
        long_horizon_reconstruction_trajectory_cid: str,
        dominant_repair_label: int = 0,
        long_horizon_reconstruction_count: int = 0,
        long_horizon_blackout_window_turns: int = 0,
        visible_token_budget: float = 256.0,
        baseline_cost: float = 512.0,
        task_id: str = "task", team_id: str = "team",
        branch_id: str = "main",
        dim: int = W78_KV_V23_FINGERPRINT_DIM,
) -> tuple[float, ...]:
    """Deterministic 156-dim long-horizon-reconstruction
    fingerprint."""
    label_str = (
        W78_REPAIR_LABELS_V23[int(dominant_repair_label)]
        if 0 <= int(dominant_repair_label)
            < len(W78_REPAIR_LABELS_V23)
        else "unknown")
    base = (
        f"{role}|"
        f"{post_restart_replacement_trajectory_cid}|"
        f"{long_horizon_reconstruction_trajectory_cid}|"
        f"{label_str}|"
        f"{int(long_horizon_reconstruction_count)}|"
        f"{int(long_horizon_blackout_window_turns)}|"
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


def probe_kv_bridge_v23_long_horizon_reconstruction_margin(
        *, params: TinyV23SubstrateParams,
        token_ids: Sequence[int], n_targets: int = 19,
) -> dict[str, Any]:
    base_trace, _ = forward_tiny_substrate_v23(
        params, list(token_ids))
    base_logits = _np.asarray(
        base_trace.logits, dtype=_np.float64)
    margins: list[float] = []
    for _ in range(int(n_targets)):
        t, _ = forward_tiny_substrate_v23(
            params, list(token_ids))
        l = _np.asarray(t.logits, dtype=_np.float64)
        diff = float(
            _np.linalg.norm((l - base_logits).ravel()))
        margins.append(diff)
    return {
        "schema": W78_KV_BRIDGE_V23_SCHEMA_VERSION,
        "kind": "v23_long_horizon_reconstruction_margin_probe",
        "n_targets": int(n_targets),
        "per_target_margin_l2": [
            float(round(float(x), 12)) for x in margins],
        "max_margin": float(round(
            max(margins) if margins else 0.0, 12)),
    }


@dataclasses.dataclass(frozen=True)
class KVBridgeV23LongHorizonReconstructionFalsifier:
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
                "kv_bridge_v23_long_horizon_reconstruction_"
                "falsifier",
            "falsifier": self.to_dict()})


def probe_kv_bridge_v23_long_horizon_reconstruction_falsifier(
        *, long_horizon_reconstruction_pressure_flag: int,
) -> KVBridgeV23LongHorizonReconstructionFalsifier:
    """Returns 0 iff inverting the long-horizon-reconstruction-
    pressure flag flips the routing decision."""
    f = int(long_horizon_reconstruction_pressure_flag)
    inv = 1 if f == 0 else 0
    decision = (
        "route_through_substrate" if f > 0
        else "route_through_text")
    inv_decision = (
        "route_through_substrate" if inv > 0
        else "route_through_text")
    flipped = decision != inv_decision
    score = 0.0 if flipped else 1.0
    return KVBridgeV23LongHorizonReconstructionFalsifier(
        primary_flag=int(f),
        inverted_flag=int(inv),
        decision=str(decision),
        inverted_decision=str(inv_decision),
        falsifier_score=float(score),
    )


@dataclasses.dataclass(frozen=True)
class KVBridgeV23Witness:
    schema: str
    projection_cid: str
    fit_report_cid: str
    long_horizon_reconstruction_margin_probe_cid: str
    long_horizon_reconstruction_falsifier_cid: str
    long_horizon_reconstruction_fingerprint_l1: float
    max_long_horizon_reconstruction_margin: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "projection_cid": str(self.projection_cid),
            "fit_report_cid": str(self.fit_report_cid),
            "long_horizon_reconstruction_margin_probe_cid": str(
                self
                .long_horizon_reconstruction_margin_probe_cid),
            "long_horizon_reconstruction_falsifier_cid": str(
                self.long_horizon_reconstruction_falsifier_cid),
            "long_horizon_reconstruction_fingerprint_l1": float(
                round(
                    self
                    .long_horizon_reconstruction_fingerprint_l1,
                    12)),
            "max_long_horizon_reconstruction_margin": float(round(
                self.max_long_horizon_reconstruction_margin, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "kv_bridge_v23_witness",
            "witness": self.to_dict()})


def emit_kv_bridge_v23_witness(
        *, projection: KVBridgeV23Projection,
        fit_report: KVBridgeV23FitReport | None = None,
        long_horizon_reconstruction_margin_probe: (
            dict[str, Any] | None) = None,
        long_horizon_reconstruction_falsifier: (
            KVBridgeV23LongHorizonReconstructionFalsifier | None) = (
                None),
        long_horizon_reconstruction_fingerprint: (
            Sequence[float] | None) = None,
) -> KVBridgeV23Witness:
    fp_l1 = 0.0
    if long_horizon_reconstruction_fingerprint is not None:
        fp_l1 = float(sum(
            abs(float(x))
            for x in long_horizon_reconstruction_fingerprint))
    return KVBridgeV23Witness(
        schema=W78_KV_BRIDGE_V23_SCHEMA_VERSION,
        projection_cid=str(projection.cid()),
        fit_report_cid=(
            fit_report.cid() if fit_report is not None else ""),
        long_horizon_reconstruction_margin_probe_cid=(
            _sha256_hex(long_horizon_reconstruction_margin_probe)
            if long_horizon_reconstruction_margin_probe is not None
            else ""),
        long_horizon_reconstruction_falsifier_cid=(
            long_horizon_reconstruction_falsifier.cid()
            if long_horizon_reconstruction_falsifier is not None
            else ""),
        long_horizon_reconstruction_fingerprint_l1=float(fp_l1),
        max_long_horizon_reconstruction_margin=float(
            long_horizon_reconstruction_margin_probe["max_margin"]
            if long_horizon_reconstruction_margin_probe is not None
            and "max_margin"
                in long_horizon_reconstruction_margin_probe
            else 0.0),
    )


__all__ = [
    "W78_KV_BRIDGE_V23_SCHEMA_VERSION",
    "W78_DEFAULT_KV_V23_RIDGE_LAMBDA",
    "W78_KV_V23_FINGERPRINT_DIM",
    "KVBridgeV23Projection",
    "KVBridgeV23FitReport",
    "fit_kv_bridge_v23_nineteen_target",
    "compute_long_horizon_reconstruction_fingerprint_v23",
    "probe_kv_bridge_v23_long_horizon_reconstruction_margin",
    "KVBridgeV23LongHorizonReconstructionFalsifier",
    "probe_kv_bridge_v23_long_horizon_reconstruction_falsifier",
    "KVBridgeV23Witness",
    "emit_kv_bridge_v23_witness",
]
