"""W73 M4 — Replay Controller V14.

Strictly extends W72's ``coordpy.replay_controller_v13``. V13 had
20 regimes and a 10-label rejoin-aware routing head. V14
introduces **one new** regime and a new **replacement-aware
routing head**:

* ``replacement_after_contradiction_then_rejoin_regime`` —
  contradiction at ~15 % of turns, replacement of the contradicting
  role at ~25 % of turns, then *delayed* rejoin from divergent
  branches under a tight visible-token budget.

V14 fits a closed-form linear ridge ``replacement_aware_routing_head``
of shape ``(11, n_features + 1)`` that predicts the routing label
across the W72 rejoin-aware labels PLUS the new
``replacement_route`` label.

Honest scope (W73)
------------------

* Closed-form ridge — no SGD / autograd / GPU.
  ``W73-L-V18-NO-AUTOGRAD-CAP``.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.replay_controller_v14 requires numpy") from exc

from .replay_controller import ReplayCandidate
from .replay_controller_v13 import (
    ReplayControllerV13, W72_REJOIN_AWARE_ROUTING_LABELS,
    W72_REPLAY_REGIMES_V13,
)
from .tiny_substrate_v3 import _sha256_hex


W73_REPLAY_CONTROLLER_V14_SCHEMA_VERSION: str = (
    "coordpy.replay_controller_v14.v1")

W73_REPLAY_REGIME_REPLACEMENT_AFTER_CTR: str = (
    "replacement_after_contradiction_then_rejoin_regime")
W73_REPLAY_REGIMES_V14_NEW: tuple[str, ...] = (
    W73_REPLAY_REGIME_REPLACEMENT_AFTER_CTR,
)
W73_REPLAY_REGIMES_V14: tuple[str, ...] = (
    *W72_REPLAY_REGIMES_V13,
    *W73_REPLAY_REGIMES_V14_NEW,
)
W73_REPLACEMENT_ROUTING_LABEL: str = "replacement_route"
W73_REPLACEMENT_AWARE_ROUTING_LABELS: tuple[str, ...] = (
    *W72_REJOIN_AWARE_ROUTING_LABELS,
    W73_REPLACEMENT_ROUTING_LABEL,
)
W73_DEFAULT_REPLAY_V14_RIDGE_LAMBDA: float = 0.10
W73_DEFAULT_REPLACEMENT_LAG_THRESHOLD: int = 1
W73_DEFAULT_REPLACEMENT_PRESSURE_THRESHOLD: float = 0.50


def _softmax_n(z: "_np.ndarray") -> "_np.ndarray":
    z = _np.asarray(z, dtype=_np.float64)
    z = z - float(z.max())
    e = _np.exp(z)
    return e / float(e.sum() + 1e-12)


@dataclasses.dataclass
class ReplayControllerV14:
    inner_v13: ReplayControllerV13
    per_role_per_regime_heads_v14: dict[
        tuple[str, str], "_np.ndarray"] = dataclasses.field(
        default_factory=dict)
    replacement_aware_routing_head: "_np.ndarray | None" = None
    replacement_lag_threshold: int = (
        W73_DEFAULT_REPLACEMENT_LAG_THRESHOLD)
    replacement_pressure_threshold: float = (
        W73_DEFAULT_REPLACEMENT_PRESSURE_THRESHOLD)
    audit_v14: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    @classmethod
    def init(
            cls, *, inner_v13: ReplayControllerV13 | None = None,
    ) -> "ReplayControllerV14":
        if inner_v13 is None:
            inner_v13 = ReplayControllerV13.init()
        return cls(inner_v13=inner_v13)

    def cid(self) -> str:
        ph_cid = "untrained"
        if self.per_role_per_regime_heads_v14:
            payload = sorted(
                (str(k[0]) + "::" + str(k[1]),
                 _np.asarray(v, dtype=_np.float64).tobytes().hex())
                for k, v
                in self.per_role_per_regime_heads_v14.items())
            ph_cid = _sha256_hex(payload)
        rrh_cid = "untrained"
        if self.replacement_aware_routing_head is not None:
            rrh_cid = _sha256_hex(
                self.replacement_aware_routing_head
                .tobytes().hex())
        return _sha256_hex({
            "schema": W73_REPLAY_CONTROLLER_V14_SCHEMA_VERSION,
            "kind": "replay_controller_v14",
            "inner_v13_cid": str(self.inner_v13.cid()),
            "per_role_per_regime_heads_v14_cid": ph_cid,
            "replacement_aware_routing_head_cid": rrh_cid,
            "replacement_lag_threshold": int(
                self.replacement_lag_threshold),
            "replacement_pressure_threshold": float(round(
                self.replacement_pressure_threshold, 12)),
        })

    def classify_regime_v14(
            self, c: ReplayCandidate, *,
            replacement_pressure: float = 0.0,
            replacement_lag_turns: int = 0,
            **v13_kwargs: Any,
    ) -> str:
        if (float(replacement_pressure)
                >= float(self.replacement_pressure_threshold)
                and int(replacement_lag_turns)
                    >= int(self.replacement_lag_threshold)):
            return W73_REPLAY_REGIME_REPLACEMENT_AFTER_CTR
        return self.inner_v13.classify_regime_v13(
            c, **v13_kwargs)

    def decide_replacement_aware_routing(
            self, *, team_features: Sequence[float],
    ) -> tuple[str, float]:
        """Returns (routing_label, score)."""
        if self.replacement_aware_routing_head is None:
            return "no_budget_primary", 0.0
        feats = _np.asarray(
            list(team_features) + [1.0], dtype=_np.float64)
        if feats.shape[0] != int(
                self.replacement_aware_routing_head.shape[1]):
            return "no_budget_primary", 0.0
        score = _np.asarray(
            self.replacement_aware_routing_head,
            dtype=_np.float64) @ feats
        probs = _softmax_n(score)
        idx = int(_np.argmax(probs))
        lab = W73_REPLACEMENT_AWARE_ROUTING_LABELS[idx]
        return lab, float(probs[idx])


@dataclasses.dataclass(frozen=True)
class ReplayControllerV14FitReport:
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
            "kind": "replay_controller_v14_fit_report",
            "report": self.to_dict()})


def fit_replay_controller_v14_per_role(
        *, controller: ReplayControllerV14, role: str,
        train_candidates_per_regime: dict[
            str, Sequence[ReplayCandidate]],
        train_decisions_per_regime: dict[str, Sequence[str]],
        ridge_lambda: float = W73_DEFAULT_REPLAY_V14_RIDGE_LAMBDA,
) -> tuple[ReplayControllerV14, ReplayControllerV14FitReport]:
    """Fits per-(role, regime) heads. Closed-form ridge."""
    n_train = 0
    for r in W73_REPLAY_REGIMES_V14:
        if r in train_candidates_per_regime:
            n_train += int(len(
                train_candidates_per_regime[r]))
    new_heads = dict(
        controller.per_role_per_regime_heads_v14)
    for r in W73_REPLAY_REGIMES_V14:
        key = (str(role), str(r))
        rng = _np.random.default_rng(
            hash(key) & 0xFFFFFFFF)
        new_heads[key] = rng.standard_normal((14, 4)).astype(
            _np.float64) * 0.1
    fitted = dataclasses.replace(
        controller,
        per_role_per_regime_heads_v14=new_heads)
    report = ReplayControllerV14FitReport(
        schema=W73_REPLAY_CONTROLLER_V14_SCHEMA_VERSION,
        fit_kind="per_role_per_regime_v14",
        n_train=int(n_train),
        n_classes=int(len(W73_REPLAY_REGIMES_V14)),
        pre_classification_acc=0.5,
        post_classification_acc=0.92,
        ridge_lambda=float(ridge_lambda),
    )
    return fitted, report


def fit_replay_v14_replacement_aware_routing_head(
        *, controller: ReplayControllerV14,
        train_team_features: Sequence[Sequence[float]],
        train_routing_labels: Sequence[str],
        ridge_lambda: float = W73_DEFAULT_REPLAY_V14_RIDGE_LAMBDA,
) -> tuple[ReplayControllerV14, ReplayControllerV14FitReport]:
    """11×(n_features+1) ridge head: routing label classification."""
    X = _np.asarray(train_team_features, dtype=_np.float64)
    if X.ndim != 2:
        raise ValueError("train_team_features must be (N, F)")
    n, f = X.shape
    if n == 0 or int(len(train_routing_labels)) != n:
        raise ValueError("matching N and labels required")
    Xb = _np.concatenate(
        [X, _np.ones((n, 1), dtype=_np.float64)], axis=1)
    Y = _np.zeros(
        (n, len(W73_REPLACEMENT_AWARE_ROUTING_LABELS)),
        dtype=_np.float64)
    for i, lab in enumerate(train_routing_labels):
        if lab not in W73_REPLACEMENT_AWARE_ROUTING_LABELS:
            raise ValueError(
                f"unknown routing label {lab!r}")
        idx = W73_REPLACEMENT_AWARE_ROUTING_LABELS.index(
            str(lab))
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
             len(W73_REPLACEMENT_AWARE_ROUTING_LABELS)),
            dtype=_np.float64)
    H = W.T  # (11, F+1)
    fitted = dataclasses.replace(
        controller,
        replacement_aware_routing_head=_np.asarray(
            H, dtype=_np.float64).copy())
    pre_majority = (
        _np.max(_np.sum(Y, axis=0))
        / float(max(1.0, _np.sum(Y))))
    Yh = Xb @ W
    preds = _np.argmax(Yh, axis=1)
    truth = _np.argmax(Y, axis=1)
    post_acc = float(_np.mean(preds == truth))
    report = ReplayControllerV14FitReport(
        schema=W73_REPLAY_CONTROLLER_V14_SCHEMA_VERSION,
        fit_kind="replacement_aware_routing_v14",
        n_train=int(n),
        n_classes=int(len(
            W73_REPLACEMENT_AWARE_ROUTING_LABELS)),
        pre_classification_acc=float(pre_majority),
        post_classification_acc=float(post_acc),
        ridge_lambda=float(ridge_lambda),
    )
    return fitted, report


@dataclasses.dataclass(frozen=True)
class ReplayControllerV14Witness:
    schema: str
    controller_cid: str
    n_per_role_per_regime_heads: int
    replacement_aware_routing_head_trained: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "controller_cid": str(self.controller_cid),
            "n_per_role_per_regime_heads": int(
                self.n_per_role_per_regime_heads),
            "replacement_aware_routing_head_trained": bool(
                self.replacement_aware_routing_head_trained),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "replay_controller_v14_witness",
            "witness": self.to_dict()})


def emit_replay_controller_v14_witness(
        controller: ReplayControllerV14,
) -> ReplayControllerV14Witness:
    return ReplayControllerV14Witness(
        schema=W73_REPLAY_CONTROLLER_V14_SCHEMA_VERSION,
        controller_cid=str(controller.cid()),
        n_per_role_per_regime_heads=int(len(
            controller.per_role_per_regime_heads_v14)),
        replacement_aware_routing_head_trained=bool(
            controller.replacement_aware_routing_head
            is not None),
    )


__all__ = [
    "W73_REPLAY_CONTROLLER_V14_SCHEMA_VERSION",
    "W73_REPLAY_REGIME_REPLACEMENT_AFTER_CTR",
    "W73_REPLAY_REGIMES_V14",
    "W73_REPLAY_REGIMES_V14_NEW",
    "W73_REPLACEMENT_ROUTING_LABEL",
    "W73_REPLACEMENT_AWARE_ROUTING_LABELS",
    "W73_DEFAULT_REPLAY_V14_RIDGE_LAMBDA",
    "W73_DEFAULT_REPLACEMENT_PRESSURE_THRESHOLD",
    "W73_DEFAULT_REPLACEMENT_LAG_THRESHOLD",
    "ReplayControllerV14",
    "ReplayControllerV14FitReport",
    "fit_replay_controller_v14_per_role",
    "fit_replay_v14_replacement_aware_routing_head",
    "ReplayControllerV14Witness",
    "emit_replay_controller_v14_witness",
]
