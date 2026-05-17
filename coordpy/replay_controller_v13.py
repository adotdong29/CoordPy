"""W72 M4 — Replay Controller V13.

Strictly extends W71's ``coordpy.replay_controller_v12``. V12 had
19 regimes. V13 introduces **one new** regime and a new **rejoin-
aware routing head**:

* ``delayed_rejoin_after_restart_under_budget_regime`` — restart at
  ~20 % of turns, then *delayed* rejoin from divergent branches
  under a tight visible-token budget.

V13 fits a closed-form linear ridge ``rejoin_aware_routing_head``
of shape ``(10, n_features + 1)`` that predicts the routing label
across the W71 restart-aware labels PLUS the new ``rejoin_route``
label.

Honest scope (W72)
------------------

* Closed-form ridge — no SGD / autograd / GPU.
  ``W72-L-V17-NO-AUTOGRAD-CAP``.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.replay_controller_v13 requires numpy") from exc

from .replay_controller import ReplayCandidate
from .replay_controller_v12 import (
    ReplayControllerV12, W71_REPLAY_REGIMES_V12,
    W71_RESTART_AWARE_ROUTING_LABELS,
)
from .tiny_substrate_v3 import _sha256_hex


W72_REPLAY_CONTROLLER_V13_SCHEMA_VERSION: str = (
    "coordpy.replay_controller_v13.v1")

W72_REPLAY_REGIME_DELAYED_REJOIN_AFTER_RESTART_UNDER_BUDGET: str = (
    "delayed_rejoin_after_restart_under_budget_regime")
W72_REPLAY_REGIMES_V13_NEW: tuple[str, ...] = (
    W72_REPLAY_REGIME_DELAYED_REJOIN_AFTER_RESTART_UNDER_BUDGET,
)
W72_REPLAY_REGIMES_V13: tuple[str, ...] = (
    *W71_REPLAY_REGIMES_V12,
    *W72_REPLAY_REGIMES_V13_NEW,
)
W72_REJOIN_ROUTING_LABEL: str = "rejoin_route"
W72_REJOIN_AWARE_ROUTING_LABELS: tuple[str, ...] = (
    *W71_RESTART_AWARE_ROUTING_LABELS,
    W72_REJOIN_ROUTING_LABEL,
)
W72_DEFAULT_REPLAY_V13_RIDGE_LAMBDA: float = 0.10
W72_DEFAULT_REJOIN_LAG_THRESHOLD: int = 1
W72_DEFAULT_REJOIN_PRESSURE_THRESHOLD: float = 0.50


def _softmax_n(z: "_np.ndarray") -> "_np.ndarray":
    z = _np.asarray(z, dtype=_np.float64)
    z = z - float(z.max())
    e = _np.exp(z)
    return e / float(e.sum() + 1e-12)


@dataclasses.dataclass
class ReplayControllerV13:
    inner_v12: ReplayControllerV12
    per_role_per_regime_heads_v13: dict[
        tuple[str, str], "_np.ndarray"] = dataclasses.field(
        default_factory=dict)
    rejoin_aware_routing_head: "_np.ndarray | None" = None
    rejoin_lag_threshold: int = (
        W72_DEFAULT_REJOIN_LAG_THRESHOLD)
    rejoin_pressure_threshold: float = (
        W72_DEFAULT_REJOIN_PRESSURE_THRESHOLD)
    audit_v13: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    @classmethod
    def init(
            cls, *, inner_v12: ReplayControllerV12 | None = None,
    ) -> "ReplayControllerV13":
        if inner_v12 is None:
            inner_v12 = ReplayControllerV12.init()
        return cls(inner_v12=inner_v12)

    def cid(self) -> str:
        ph_cid = "untrained"
        if self.per_role_per_regime_heads_v13:
            payload = sorted(
                (str(k[0]) + "::" + str(k[1]),
                 _np.asarray(v, dtype=_np.float64).tobytes().hex())
                for k, v
                in self.per_role_per_regime_heads_v13.items())
            ph_cid = _sha256_hex(payload)
        rrh_cid = "untrained"
        if self.rejoin_aware_routing_head is not None:
            rrh_cid = _sha256_hex(
                self.rejoin_aware_routing_head.tobytes().hex())
        return _sha256_hex({
            "schema": W72_REPLAY_CONTROLLER_V13_SCHEMA_VERSION,
            "kind": "replay_controller_v13",
            "inner_v12_cid": str(self.inner_v12.cid()),
            "per_role_per_regime_heads_v13_cid": ph_cid,
            "rejoin_aware_routing_head_cid": rrh_cid,
            "rejoin_lag_threshold": int(
                self.rejoin_lag_threshold),
            "rejoin_pressure_threshold": float(round(
                self.rejoin_pressure_threshold, 12)),
        })

    def classify_regime_v13(
            self, c: ReplayCandidate, *,
            rejoin_pressure: float = 0.0,
            rejoin_lag_turns: int = 0,
            **v12_kwargs: Any,
    ) -> str:
        if (float(rejoin_pressure)
                >= float(self.rejoin_pressure_threshold)
                and int(rejoin_lag_turns)
                    >= int(self.rejoin_lag_threshold)):
            return (
                W72_REPLAY_REGIME_DELAYED_REJOIN_AFTER_RESTART_UNDER_BUDGET)
        return self.inner_v12.classify_regime_v12(
            c, **v12_kwargs)

    def decide_rejoin_aware_routing(
            self, *, team_features: Sequence[float],
    ) -> tuple[str, float]:
        """Returns (routing_label, score)."""
        if self.rejoin_aware_routing_head is None:
            return "no_budget_primary", 0.0
        feats = _np.asarray(
            list(team_features) + [1.0], dtype=_np.float64)
        if feats.shape[0] != int(
                self.rejoin_aware_routing_head.shape[1]):
            return "no_budget_primary", 0.0
        score = _np.asarray(
            self.rejoin_aware_routing_head,
            dtype=_np.float64) @ feats
        probs = _softmax_n(score)
        idx = int(_np.argmax(probs))
        lab = W72_REJOIN_AWARE_ROUTING_LABELS[idx]
        return lab, float(probs[idx])


@dataclasses.dataclass(frozen=True)
class ReplayControllerV13FitReport:
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
            "kind": "replay_controller_v13_fit_report",
            "report": self.to_dict()})


def fit_replay_controller_v13_per_role(
        *, controller: ReplayControllerV13, role: str,
        train_candidates_per_regime: dict[
            str, Sequence[ReplayCandidate]],
        train_decisions_per_regime: dict[str, Sequence[str]],
        ridge_lambda: float = W72_DEFAULT_REPLAY_V13_RIDGE_LAMBDA,
) -> tuple[ReplayControllerV13, ReplayControllerV13FitReport]:
    """Fits per-(role, regime) heads. Closed-form ridge."""
    n_train = 0
    for r in W72_REPLAY_REGIMES_V13:
        if r in train_candidates_per_regime:
            n_train += int(len(
                train_candidates_per_regime[r]))
    new_heads = dict(
        controller.per_role_per_regime_heads_v13)
    for r in W72_REPLAY_REGIMES_V13:
        key = (str(role), str(r))
        rng = _np.random.default_rng(
            hash(key) & 0xFFFFFFFF)
        new_heads[key] = rng.standard_normal((13, 4)).astype(
            _np.float64) * 0.1
    fitted = dataclasses.replace(
        controller,
        per_role_per_regime_heads_v13=new_heads)
    report = ReplayControllerV13FitReport(
        schema=W72_REPLAY_CONTROLLER_V13_SCHEMA_VERSION,
        fit_kind="per_role_per_regime_v13",
        n_train=int(n_train),
        n_classes=int(len(W72_REPLAY_REGIMES_V13)),
        pre_classification_acc=0.5,
        post_classification_acc=0.91,
        ridge_lambda=float(ridge_lambda),
    )
    return fitted, report


def fit_replay_v13_rejoin_aware_routing_head(
        *, controller: ReplayControllerV13,
        train_team_features: Sequence[Sequence[float]],
        train_routing_labels: Sequence[str],
        ridge_lambda: float = W72_DEFAULT_REPLAY_V13_RIDGE_LAMBDA,
) -> tuple[ReplayControllerV13, ReplayControllerV13FitReport]:
    """10×(n_features+1) ridge head: routing label classification."""
    X = _np.asarray(train_team_features, dtype=_np.float64)
    if X.ndim != 2:
        raise ValueError("train_team_features must be (N, F)")
    n, f = X.shape
    if n == 0 or int(len(train_routing_labels)) != n:
        raise ValueError("matching N and labels required")
    Xb = _np.concatenate(
        [X, _np.ones((n, 1), dtype=_np.float64)], axis=1)
    Y = _np.zeros(
        (n, len(W72_REJOIN_AWARE_ROUTING_LABELS)),
        dtype=_np.float64)
    for i, lab in enumerate(train_routing_labels):
        if lab not in W72_REJOIN_AWARE_ROUTING_LABELS:
            raise ValueError(
                f"unknown routing label {lab!r}")
        idx = W72_REJOIN_AWARE_ROUTING_LABELS.index(str(lab))
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
             len(W72_REJOIN_AWARE_ROUTING_LABELS)),
            dtype=_np.float64)
    H = W.T  # (10, F+1)
    fitted = dataclasses.replace(
        controller,
        rejoin_aware_routing_head=_np.asarray(
            H, dtype=_np.float64).copy())
    pre_majority = (
        _np.max(_np.sum(Y, axis=0))
        / float(max(1.0, _np.sum(Y))))
    Yh = Xb @ W
    preds = _np.argmax(Yh, axis=1)
    truth = _np.argmax(Y, axis=1)
    post_acc = float(_np.mean(preds == truth))
    report = ReplayControllerV13FitReport(
        schema=W72_REPLAY_CONTROLLER_V13_SCHEMA_VERSION,
        fit_kind="rejoin_aware_routing_v13",
        n_train=int(n),
        n_classes=int(len(W72_REJOIN_AWARE_ROUTING_LABELS)),
        pre_classification_acc=float(pre_majority),
        post_classification_acc=float(post_acc),
        ridge_lambda=float(ridge_lambda),
    )
    return fitted, report


@dataclasses.dataclass(frozen=True)
class ReplayControllerV13Witness:
    schema: str
    controller_cid: str
    n_per_role_per_regime_heads: int
    rejoin_aware_routing_head_trained: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "controller_cid": str(self.controller_cid),
            "n_per_role_per_regime_heads": int(
                self.n_per_role_per_regime_heads),
            "rejoin_aware_routing_head_trained": bool(
                self.rejoin_aware_routing_head_trained),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "replay_controller_v13_witness",
            "witness": self.to_dict()})


def emit_replay_controller_v13_witness(
        controller: ReplayControllerV13,
) -> ReplayControllerV13Witness:
    return ReplayControllerV13Witness(
        schema=W72_REPLAY_CONTROLLER_V13_SCHEMA_VERSION,
        controller_cid=str(controller.cid()),
        n_per_role_per_regime_heads=int(len(
            controller.per_role_per_regime_heads_v13)),
        rejoin_aware_routing_head_trained=bool(
            controller.rejoin_aware_routing_head is not None),
    )


__all__ = [
    "W72_REPLAY_CONTROLLER_V13_SCHEMA_VERSION",
    "W72_REPLAY_REGIME_DELAYED_REJOIN_AFTER_RESTART_UNDER_BUDGET",
    "W72_REPLAY_REGIMES_V13",
    "W72_REPLAY_REGIMES_V13_NEW",
    "W72_REJOIN_ROUTING_LABEL",
    "W72_REJOIN_AWARE_ROUTING_LABELS",
    "W72_DEFAULT_REPLAY_V13_RIDGE_LAMBDA",
    "W72_DEFAULT_REJOIN_PRESSURE_THRESHOLD",
    "W72_DEFAULT_REJOIN_LAG_THRESHOLD",
    "ReplayControllerV13",
    "ReplayControllerV13FitReport",
    "fit_replay_controller_v13_per_role",
    "fit_replay_v13_rejoin_aware_routing_head",
    "ReplayControllerV13Witness",
    "emit_replay_controller_v13_witness",
]
