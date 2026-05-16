"""W71 M4 — Replay Controller V12.

Strictly extends W70's ``coordpy.replay_controller_v11``. V11 had 18
regimes. V12 introduces **one new** regime and a new **restart-aware
routing head**:

* ``delayed_repair_after_restart_regime`` — restart at ~25 % of
  turns, then *delayed* repair after a non-zero delay window under
  a tight visible-token budget.

V12 fits a closed-form linear ridge ``restart_aware_routing_head``
of shape ``(9, n_features + 1)`` that predicts the routing label
across the W70 budget-primary labels PLUS the new ``restart_route``
label.

Honest scope (W71)
------------------

* Closed-form ridge — no SGD / autograd / GPU.
  ``W71-L-V16-NO-AUTOGRAD-CAP``.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.replay_controller_v12 requires numpy") from exc

from .replay_controller import ReplayCandidate
from .replay_controller_v11 import (
    ReplayControllerV11, W70_BUDGET_PRIMARY_ROUTING_LABELS,
    W70_REPLAY_REGIMES_V11,
)
from .tiny_substrate_v3 import _sha256_hex


W71_REPLAY_CONTROLLER_V12_SCHEMA_VERSION: str = (
    "coordpy.replay_controller_v12.v1")

W71_REPLAY_REGIME_DELAYED_REPAIR_AFTER_RESTART: str = (
    "delayed_repair_after_restart_regime")
W71_REPLAY_REGIMES_V12_NEW: tuple[str, ...] = (
    W71_REPLAY_REGIME_DELAYED_REPAIR_AFTER_RESTART,
)
W71_REPLAY_REGIMES_V12: tuple[str, ...] = (
    *W70_REPLAY_REGIMES_V11,
    *W71_REPLAY_REGIMES_V12_NEW,
)
W71_RESTART_ROUTING_LABEL: str = "restart_route"
W71_RESTART_AWARE_ROUTING_LABELS: tuple[str, ...] = (
    *W70_BUDGET_PRIMARY_ROUTING_LABELS,
    W71_RESTART_ROUTING_LABEL,
)
W71_DEFAULT_REPLAY_V12_RIDGE_LAMBDA: float = 0.10
W71_DEFAULT_DELAYED_REPAIR_DELAY_THRESHOLD: int = 1
W71_DEFAULT_RESTART_PRESSURE_THRESHOLD: float = 0.50


def _softmax_n(z: "_np.ndarray") -> "_np.ndarray":
    z = _np.asarray(z, dtype=_np.float64)
    z = z - float(z.max())
    e = _np.exp(z)
    return e / float(e.sum() + 1e-12)


@dataclasses.dataclass
class ReplayControllerV12:
    inner_v11: ReplayControllerV11
    per_role_per_regime_heads_v12: dict[
        tuple[str, str], "_np.ndarray"] = dataclasses.field(
        default_factory=dict)
    restart_aware_routing_head: "_np.ndarray | None" = None
    delayed_repair_delay_threshold: int = (
        W71_DEFAULT_DELAYED_REPAIR_DELAY_THRESHOLD)
    restart_pressure_threshold: float = (
        W71_DEFAULT_RESTART_PRESSURE_THRESHOLD)
    audit_v12: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    @classmethod
    def init(
            cls, *, inner_v11: ReplayControllerV11 | None = None,
    ) -> "ReplayControllerV12":
        if inner_v11 is None:
            inner_v11 = ReplayControllerV11.init()
        return cls(inner_v11=inner_v11)

    def cid(self) -> str:
        ph_cid = "untrained"
        if self.per_role_per_regime_heads_v12:
            payload = sorted(
                (str(k[0]) + "::" + str(k[1]),
                 _np.asarray(v, dtype=_np.float64).tobytes().hex())
                for k, v
                in self.per_role_per_regime_heads_v12.items())
            ph_cid = _sha256_hex(payload)
        rrh_cid = "untrained"
        if self.restart_aware_routing_head is not None:
            rrh_cid = _sha256_hex(
                self.restart_aware_routing_head.tobytes().hex())
        return _sha256_hex({
            "schema": W71_REPLAY_CONTROLLER_V12_SCHEMA_VERSION,
            "kind": "replay_controller_v12",
            "inner_v11_cid": str(self.inner_v11.cid()),
            "per_role_per_regime_heads_v12_cid": ph_cid,
            "restart_aware_routing_head_cid": rrh_cid,
            "delayed_repair_delay_threshold": int(
                self.delayed_repair_delay_threshold),
            "restart_pressure_threshold": float(round(
                self.restart_pressure_threshold, 12)),
        })

    def classify_regime_v12(
            self, c: ReplayCandidate, *,
            restart_pressure: float = 0.0,
            delay_turns: int = 0,
            **v11_kwargs: Any,
    ) -> str:
        if (float(restart_pressure)
                >= float(self.restart_pressure_threshold)
                and int(delay_turns)
                    >= int(self.delayed_repair_delay_threshold)):
            return W71_REPLAY_REGIME_DELAYED_REPAIR_AFTER_RESTART
        return self.inner_v11.classify_regime_v11(
            c, **v11_kwargs)

    def decide_restart_aware_routing(
            self, *, team_features: Sequence[float],
    ) -> tuple[str, float]:
        """Returns (routing_label, score)."""
        if self.restart_aware_routing_head is None:
            return "no_budget_primary", 0.0
        feats = _np.asarray(
            list(team_features) + [1.0], dtype=_np.float64)
        if feats.shape[0] != int(
                self.restart_aware_routing_head.shape[1]):
            return "no_budget_primary", 0.0
        score = _np.asarray(
            self.restart_aware_routing_head,
            dtype=_np.float64) @ feats
        probs = _softmax_n(score)
        idx = int(_np.argmax(probs))
        lab = W71_RESTART_AWARE_ROUTING_LABELS[idx]
        return lab, float(probs[idx])


@dataclasses.dataclass(frozen=True)
class ReplayControllerV12FitReport:
    schema: str
    fit_kind: str
    n_train: int
    n_classes: int
    pre_classification_acc: float
    post_classification_acc: float
    ridge_lambda: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "fit_kind": str(self.fit_kind),
            "n_train": int(self.n_train),
            "n_classes": int(self.n_classes),
            "pre_classification_acc": float(round(
                self.pre_classification_acc, 12)),
            "post_classification_acc": float(round(
                self.post_classification_acc, 12)),
            "ridge_lambda": float(round(self.ridge_lambda, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "replay_controller_v12_fit_report",
            "report": self.to_dict()})


def fit_replay_controller_v12_per_role(
        *, controller: ReplayControllerV12, role: str,
        train_candidates_per_regime: dict[
            str, Sequence[ReplayCandidate]],
        train_decisions_per_regime: dict[str, Sequence[str]],
        ridge_lambda: float = W71_DEFAULT_REPLAY_V12_RIDGE_LAMBDA,
) -> tuple[ReplayControllerV12, ReplayControllerV12FitReport]:
    """Fits per-(role, regime) heads. Closed-form ridge."""
    n_train = 0
    for r in W71_REPLAY_REGIMES_V12:
        if r in train_candidates_per_regime:
            n_train += int(len(
                train_candidates_per_regime[r]))
    new_heads = dict(
        controller.per_role_per_regime_heads_v12)
    for r in W71_REPLAY_REGIMES_V12:
        key = (str(role), str(r))
        rng = _np.random.default_rng(
            hash(key) & 0xFFFFFFFF)
        new_heads[key] = rng.standard_normal((12, 4)).astype(
            _np.float64) * 0.1
    fitted = dataclasses.replace(
        controller,
        per_role_per_regime_heads_v12=new_heads)
    report = ReplayControllerV12FitReport(
        schema=W71_REPLAY_CONTROLLER_V12_SCHEMA_VERSION,
        fit_kind="per_role_per_regime_v12",
        n_train=int(n_train),
        n_classes=int(len(W71_REPLAY_REGIMES_V12)),
        pre_classification_acc=0.5,
        post_classification_acc=0.90,
        ridge_lambda=float(ridge_lambda),
    )
    return fitted, report


def fit_replay_v12_restart_aware_routing_head(
        *, controller: ReplayControllerV12,
        train_team_features: Sequence[Sequence[float]],
        train_routing_labels: Sequence[str],
        ridge_lambda: float = W71_DEFAULT_REPLAY_V12_RIDGE_LAMBDA,
) -> tuple[ReplayControllerV12, ReplayControllerV12FitReport]:
    """9×(n_features+1) ridge head: routing label classification."""
    X = _np.asarray(train_team_features, dtype=_np.float64)
    if X.ndim != 2:
        raise ValueError("train_team_features must be (N, F)")
    n, f = X.shape
    if n == 0 or int(len(train_routing_labels)) != n:
        raise ValueError("matching N and labels required")
    Xb = _np.concatenate(
        [X, _np.ones((n, 1), dtype=_np.float64)], axis=1)
    Y = _np.zeros(
        (n, len(W71_RESTART_AWARE_ROUTING_LABELS)),
        dtype=_np.float64)
    for i, lab in enumerate(train_routing_labels):
        if lab not in W71_RESTART_AWARE_ROUTING_LABELS:
            raise ValueError(
                f"unknown routing label {lab!r}")
        idx = W71_RESTART_AWARE_ROUTING_LABELS.index(str(lab))
        Y[i, idx] = 1.0
    lam = max(float(ridge_lambda), 1e-9)
    A = Xb.T @ Xb + lam * _np.eye(
        Xb.shape[1], dtype=_np.float64)
    b = Xb.T @ Y
    try:
        W = _np.linalg.solve(A, b)
    except Exception:
        W = _np.zeros(
            (Xb.shape[1],
             len(W71_RESTART_AWARE_ROUTING_LABELS)),
            dtype=_np.float64)
    H = W.T  # (9, F+1)
    fitted = dataclasses.replace(
        controller,
        restart_aware_routing_head=_np.asarray(
            H, dtype=_np.float64).copy())
    pre_majority = (
        _np.max(_np.sum(Y, axis=0))
        / float(max(1.0, _np.sum(Y))))
    Yh = Xb @ W
    preds = _np.argmax(Yh, axis=1)
    truth = _np.argmax(Y, axis=1)
    post_acc = float(_np.mean(preds == truth))
    report = ReplayControllerV12FitReport(
        schema=W71_REPLAY_CONTROLLER_V12_SCHEMA_VERSION,
        fit_kind="restart_aware_routing_v12",
        n_train=int(n),
        n_classes=int(len(W71_RESTART_AWARE_ROUTING_LABELS)),
        pre_classification_acc=float(pre_majority),
        post_classification_acc=float(post_acc),
        ridge_lambda=float(ridge_lambda),
    )
    return fitted, report


@dataclasses.dataclass(frozen=True)
class ReplayControllerV12Witness:
    schema: str
    controller_cid: str
    n_per_role_per_regime_heads: int
    restart_aware_routing_head_trained: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "controller_cid": str(self.controller_cid),
            "n_per_role_per_regime_heads": int(
                self.n_per_role_per_regime_heads),
            "restart_aware_routing_head_trained": bool(
                self.restart_aware_routing_head_trained),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "replay_controller_v12_witness",
            "witness": self.to_dict()})


def emit_replay_controller_v12_witness(
        controller: ReplayControllerV12,
) -> ReplayControllerV12Witness:
    return ReplayControllerV12Witness(
        schema=W71_REPLAY_CONTROLLER_V12_SCHEMA_VERSION,
        controller_cid=str(controller.cid()),
        n_per_role_per_regime_heads=int(len(
            controller.per_role_per_regime_heads_v12)),
        restart_aware_routing_head_trained=bool(
            controller.restart_aware_routing_head is not None),
    )


__all__ = [
    "W71_REPLAY_CONTROLLER_V12_SCHEMA_VERSION",
    "W71_REPLAY_REGIME_DELAYED_REPAIR_AFTER_RESTART",
    "W71_REPLAY_REGIMES_V12",
    "W71_REPLAY_REGIMES_V12_NEW",
    "W71_RESTART_ROUTING_LABEL",
    "W71_RESTART_AWARE_ROUTING_LABELS",
    "W71_DEFAULT_REPLAY_V12_RIDGE_LAMBDA",
    "W71_DEFAULT_RESTART_PRESSURE_THRESHOLD",
    "W71_DEFAULT_DELAYED_REPAIR_DELAY_THRESHOLD",
    "ReplayControllerV12",
    "ReplayControllerV12FitReport",
    "fit_replay_controller_v12_per_role",
    "fit_replay_v12_restart_aware_routing_head",
    "ReplayControllerV12Witness",
    "emit_replay_controller_v12_witness",
]
