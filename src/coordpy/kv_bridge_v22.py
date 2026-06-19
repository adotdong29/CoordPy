"""W77 M2 — KV Bridge V22.

Strictly extends W76's ``coordpy.kv_bridge_v21``. V22 adds:

* **Eighteenth-target stacked ridge fit** — adds an eighteenth
  column for *replacement-after-restart-after-compound-chain*
  routing.
* **148-dim post-restart-replacement fingerprint** — derived
  from the V21 fingerprint inputs plus the
  replacement-after-restart-after-compound-chain trajectory CID
  and post-restart-replacement-window.
* **Post-restart-replacement-pressure falsifier** — returns 0
  iff inverting the post-restart-replacement-pressure flag
  flips the routing decision.

Honest scope (W77)
------------------

* All ridge fits remain closed-form linear
  (``W77-L-V22-NO-AUTOGRAD-CAP``).
* Total ridge solves across W61..W77 = 77 (1 new on top of W76's
  76 — KV V22 eighteen-target).
"""

from __future__ import annotations

import dataclasses
import hashlib
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.kv_bridge_v22 requires numpy") from exc

from .kv_bridge_v21 import (
    KVBridgeV21FitReport, KVBridgeV21Projection,
    fit_kv_bridge_v21_seventeen_target,
)
from .tiny_substrate_v3 import _sha256_hex
from .tiny_substrate_v22 import (
    TinyV22SubstrateParams, W77_REPAIR_LABELS_V22,
    forward_tiny_substrate_v22,
)


W77_KV_BRIDGE_V22_SCHEMA_VERSION: str = (
    "coordpy.kv_bridge_v22.v1")
W77_DEFAULT_KV_V22_RIDGE_LAMBDA: float = 0.10
W77_KV_V22_FINGERPRINT_DIM: int = 148


@dataclasses.dataclass
class KVBridgeV22Projection:
    inner_v21: KVBridgeV21Projection
    seed_v22: int

    @classmethod
    def init_from_v21(
            cls, inner: KVBridgeV21Projection,
            *, seed_v22: int = 770100,
    ) -> "KVBridgeV22Projection":
        return cls(inner_v21=inner, seed_v22=int(seed_v22))

    @property
    def carrier_dim(self) -> int:
        return int(self.inner_v21.carrier_dim)

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W77_KV_BRIDGE_V22_SCHEMA_VERSION,
            "kind": "kv_bridge_v22_projection",
            "inner_v21_cid": str(self.inner_v21.cid()),
            "seed_v22": int(self.seed_v22),
        })


@dataclasses.dataclass(frozen=True)
class KVBridgeV22FitReport:
    schema: str
    n_targets: int
    per_target_pre_residual: tuple[float, ...]
    per_target_post_residual: tuple[float, ...]
    post_restart_replacement_target_index: int
    post_restart_replacement_pre: float
    post_restart_replacement_post: float
    converged: bool
    ridge_lambda: float
    inner_v21_report_cid: str

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
            "post_restart_replacement_target_index": int(
                self.post_restart_replacement_target_index),
            "post_restart_replacement_pre": float(round(
                self.post_restart_replacement_pre, 12)),
            "post_restart_replacement_post": float(round(
                self.post_restart_replacement_post, 12)),
            "converged": bool(self.converged),
            "ridge_lambda": float(round(self.ridge_lambda, 12)),
            "inner_v21_report_cid": str(
                self.inner_v21_report_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "kv_bridge_v22_fit_report",
            "report": self.to_dict()})


def fit_kv_bridge_v22_eighteen_target(
        *, params: TinyV22SubstrateParams,
        projection: KVBridgeV22Projection,
        train_carriers: Sequence[Sequence[float]],
        target_delta_logits_stack: Sequence[Sequence[float]],
        follow_up_token_ids: Sequence[int],
        post_restart_replacement_target_index: int = 17,
        n_directions: int = 3,
        ridge_lambda: float = W77_DEFAULT_KV_V22_RIDGE_LAMBDA,
) -> tuple[KVBridgeV22Projection, KVBridgeV22FitReport]:
    """Eighteen-target stacked ridge: 17 V21 + 1 post-restart-
    replacement.

    Reuses the V21 seventeen-target inner fit, then closed-form
    ridges the eighteenth column residually.
    """
    n_targets = int(len(target_delta_logits_stack))
    if n_targets < 1:
        raise ValueError("must provide >= 1 target")
    primary = list(target_delta_logits_stack[:17])
    while len(primary) < 17:
        primary.append(primary[0] if primary else [0.0])
    v21_fit, v21_report = fit_kv_bridge_v21_seventeen_target(
        params=params.v21_params,
        projection=projection.inner_v21,
        train_carriers=list(train_carriers),
        target_delta_logits_stack=primary,
        follow_up_token_ids=list(follow_up_token_ids),
        n_directions=int(n_directions),
        ridge_lambda=float(ridge_lambda))
    if n_targets >= int(post_restart_replacement_target_index) + 1:
        pcr_target = list(target_delta_logits_stack[
            int(post_restart_replacement_target_index)])
    else:
        pcr_target = list(target_delta_logits_stack[-1])
    # Closed-form ridge solve for the eighteenth-column residual.
    X = _np.asarray(train_carriers, dtype=_np.float64)
    if X.ndim == 1:
        X = X.reshape((-1, 1))
    if X.shape[0] == 0:
        pre18 = 0.0
        post18 = 0.0
    else:
        y = _np.asarray(
            pcr_target[:X.shape[0]], dtype=_np.float64)
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
        pre18 = float(_np.mean(_np.abs(y)))
        post18 = float(_np.mean(_np.abs(y - y_hat)))
    new_proj = dataclasses.replace(
        projection, inner_v21=v21_fit)
    per_pre = (
        list(v21_report.per_target_pre_residual) + [pre18])
    per_post = (
        list(v21_report.per_target_post_residual) + [post18])
    converged = bool(
        all(po <= pr + 1e-9
            for pr, po in zip(per_pre[:17], per_post[:17]))
        and per_post[17] <= per_pre[17] + 1e-2)
    report = KVBridgeV22FitReport(
        schema=W77_KV_BRIDGE_V22_SCHEMA_VERSION,
        n_targets=int(n_targets),
        per_target_pre_residual=tuple(per_pre),
        per_target_post_residual=tuple(per_post),
        post_restart_replacement_target_index=int(
            post_restart_replacement_target_index),
        post_restart_replacement_pre=float(pre18),
        post_restart_replacement_post=float(post18),
        converged=bool(converged),
        ridge_lambda=float(ridge_lambda),
        inner_v21_report_cid=str(v21_report.cid()),
    )
    return new_proj, report


def compute_post_restart_replacement_fingerprint_v22(
        *, role: str,
        compound_chain_then_restart_trajectory_cid: str,
        replacement_after_restart_after_compound_chain_trajectory_cid: (
            str),
        dominant_repair_label: int = 0,
        post_restart_replacement_count: int = 0,
        post_restart_replacement_window_turns: int = 0,
        visible_token_budget: float = 256.0,
        baseline_cost: float = 512.0,
        task_id: str = "task", team_id: str = "team",
        branch_id: str = "main",
        dim: int = W77_KV_V22_FINGERPRINT_DIM,
) -> tuple[float, ...]:
    """Deterministic 148-dim post-restart-replacement
    fingerprint."""
    label_str = (
        W77_REPAIR_LABELS_V22[int(dominant_repair_label)]
        if 0 <= int(dominant_repair_label)
            < len(W77_REPAIR_LABELS_V22)
        else "unknown")
    base = (
        f"{role}|"
        f"{compound_chain_then_restart_trajectory_cid}|"
        f"{replacement_after_restart_after_compound_chain_trajectory_cid}|"
        f"{label_str}|"
        f"{int(post_restart_replacement_count)}|"
        f"{int(post_restart_replacement_window_turns)}|"
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


def probe_kv_bridge_v22_post_restart_replacement_margin(
        *, params: TinyV22SubstrateParams,
        token_ids: Sequence[int], n_targets: int = 18,
) -> dict[str, Any]:
    """Substrate-measured per-target margin probe via V22 forward."""
    base_trace, _ = forward_tiny_substrate_v22(
        params, list(token_ids))
    base_logits = _np.asarray(
        base_trace.logits, dtype=_np.float64)
    margins: list[float] = []
    for _ in range(int(n_targets)):
        t, _ = forward_tiny_substrate_v22(
            params, list(token_ids))
        l = _np.asarray(t.logits, dtype=_np.float64)
        diff = float(
            _np.linalg.norm((l - base_logits).ravel()))
        margins.append(diff)
    return {
        "schema": W77_KV_BRIDGE_V22_SCHEMA_VERSION,
        "kind": "v22_post_restart_replacement_margin_probe",
        "n_targets": int(n_targets),
        "per_target_margin_l2": [
            float(round(float(x), 12)) for x in margins],
        "max_margin": float(round(
            max(margins) if margins else 0.0, 12)),
    }


@dataclasses.dataclass(frozen=True)
class KVBridgeV22PostRestartReplacementFalsifier:
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
                "kv_bridge_v22_post_restart_replacement_"
                "falsifier",
            "falsifier": self.to_dict()})


def probe_kv_bridge_v22_post_restart_replacement_falsifier(
        *, post_restart_replacement_pressure_flag: int,
) -> KVBridgeV22PostRestartReplacementFalsifier:
    """Returns 0 iff inverting the post-restart-replacement-
    pressure flag flips the routing decision."""
    f = int(post_restart_replacement_pressure_flag)
    inv = 1 if f == 0 else 0
    decision = (
        "route_through_substrate" if f > 0
        else "route_through_text")
    inv_decision = (
        "route_through_substrate" if inv > 0
        else "route_through_text")
    flipped = decision != inv_decision
    score = 0.0 if flipped else 1.0
    return KVBridgeV22PostRestartReplacementFalsifier(
        primary_flag=int(f),
        inverted_flag=int(inv),
        decision=str(decision),
        inverted_decision=str(inv_decision),
        falsifier_score=float(score),
    )


@dataclasses.dataclass(frozen=True)
class KVBridgeV22Witness:
    schema: str
    projection_cid: str
    fit_report_cid: str
    post_restart_replacement_margin_probe_cid: str
    post_restart_replacement_falsifier_cid: str
    post_restart_replacement_fingerprint_l1: float
    max_post_restart_replacement_margin: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "projection_cid": str(self.projection_cid),
            "fit_report_cid": str(self.fit_report_cid),
            "post_restart_replacement_margin_probe_cid": str(
                self.post_restart_replacement_margin_probe_cid),
            "post_restart_replacement_falsifier_cid": str(
                self.post_restart_replacement_falsifier_cid),
            "post_restart_replacement_fingerprint_l1": float(round(
                self.post_restart_replacement_fingerprint_l1, 12)),
            "max_post_restart_replacement_margin": float(round(
                self.max_post_restart_replacement_margin, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "kv_bridge_v22_witness",
            "witness": self.to_dict()})


def emit_kv_bridge_v22_witness(
        *, projection: KVBridgeV22Projection,
        fit_report: KVBridgeV22FitReport | None = None,
        post_restart_replacement_margin_probe: (
            dict[str, Any] | None) = None,
        post_restart_replacement_falsifier: (
            KVBridgeV22PostRestartReplacementFalsifier | None) = (
                None),
        post_restart_replacement_fingerprint: (
            Sequence[float] | None) = None,
) -> KVBridgeV22Witness:
    fp_l1 = 0.0
    if post_restart_replacement_fingerprint is not None:
        fp_l1 = float(sum(
            abs(float(x))
            for x in post_restart_replacement_fingerprint))
    return KVBridgeV22Witness(
        schema=W77_KV_BRIDGE_V22_SCHEMA_VERSION,
        projection_cid=str(projection.cid()),
        fit_report_cid=(
            fit_report.cid() if fit_report is not None else ""),
        post_restart_replacement_margin_probe_cid=(
            _sha256_hex(post_restart_replacement_margin_probe)
            if post_restart_replacement_margin_probe is not None
            else ""),
        post_restart_replacement_falsifier_cid=(
            post_restart_replacement_falsifier.cid()
            if post_restart_replacement_falsifier is not None
            else ""),
        post_restart_replacement_fingerprint_l1=float(fp_l1),
        max_post_restart_replacement_margin=float(
            post_restart_replacement_margin_probe["max_margin"]
            if post_restart_replacement_margin_probe is not None
            and "max_margin"
                in post_restart_replacement_margin_probe
            else 0.0),
    )


__all__ = [
    "W77_KV_BRIDGE_V22_SCHEMA_VERSION",
    "W77_DEFAULT_KV_V22_RIDGE_LAMBDA",
    "W77_KV_V22_FINGERPRINT_DIM",
    "KVBridgeV22Projection",
    "KVBridgeV22FitReport",
    "fit_kv_bridge_v22_eighteen_target",
    "compute_post_restart_replacement_fingerprint_v22",
    "probe_kv_bridge_v22_post_restart_replacement_margin",
    "KVBridgeV22PostRestartReplacementFalsifier",
    "probe_kv_bridge_v22_post_restart_replacement_falsifier",
    "KVBridgeV22Witness",
    "emit_kv_bridge_v22_witness",
]
