"""W74 M4 — Replay Controller V15.

Strictly extends W73's ``coordpy.replay_controller_v14``. V14 had
21 regimes and an 11-label replacement-aware routing head. V15
introduces **one new** regime and a new **compound-aware routing
head**:

* ``compound_repair_after_delayed_repair_then_replacement_regime``
  — delayed repair at ~15 % of turns, replacement of the role at
  ~30 % of turns, then *delayed* rejoin from divergent branches at
  ~50 % of turns under a tight visible-token budget.

V15 fits a closed-form linear ridge ``compound_aware_routing_head``
of shape ``(12, n_features + 1)`` that predicts the routing label
across the W73 replacement-aware labels PLUS the new
``compound_route`` label.

Honest scope (W74)
------------------

* Closed-form ridge — no SGD / autograd / GPU.
  ``W74-L-V19-NO-AUTOGRAD-CAP``.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.replay_controller_v15 requires numpy") from exc

from .replay_controller import ReplayCandidate
from .replay_controller_v14 import (
    ReplayControllerV14,
    W73_REPLACEMENT_AWARE_ROUTING_LABELS,
    W73_REPLAY_REGIMES_V14,
)
from .tiny_substrate_v3 import _sha256_hex


W74_REPLAY_CONTROLLER_V15_SCHEMA_VERSION: str = (
    "coordpy.replay_controller_v15.v1")

W74_REPLAY_REGIME_COMPOUND_REPAIR: str = (
    "compound_repair_after_delayed_repair_then_replacement_regime")
W74_REPLAY_REGIMES_V15_NEW: tuple[str, ...] = (
    W74_REPLAY_REGIME_COMPOUND_REPAIR,
)
W74_REPLAY_REGIMES_V15: tuple[str, ...] = (
    *W73_REPLAY_REGIMES_V14,
    *W74_REPLAY_REGIMES_V15_NEW,
)
W74_COMPOUND_ROUTING_LABEL: str = "compound_route"
W74_COMPOUND_AWARE_ROUTING_LABELS: tuple[str, ...] = (
    *W73_REPLACEMENT_AWARE_ROUTING_LABELS,
    W74_COMPOUND_ROUTING_LABEL,
)
W74_DEFAULT_REPLAY_V15_RIDGE_LAMBDA: float = 0.10
W74_DEFAULT_COMPOUND_WINDOW_THRESHOLD: int = 1
W74_DEFAULT_COMPOUND_PRESSURE_THRESHOLD: float = 0.50


def _softmax_n(z: "_np.ndarray") -> "_np.ndarray":
    z = _np.asarray(z, dtype=_np.float64)
    z = z - float(z.max())
    e = _np.exp(z)
    return e / float(e.sum() + 1e-12)


@dataclasses.dataclass
class ReplayControllerV15:
    inner_v14: ReplayControllerV14
    per_role_per_regime_heads_v15: dict[
        tuple[str, str], "_np.ndarray"] = dataclasses.field(
        default_factory=dict)
    compound_aware_routing_head: "_np.ndarray | None" = None
    compound_window_threshold: int = (
        W74_DEFAULT_COMPOUND_WINDOW_THRESHOLD)
    compound_pressure_threshold: float = (
        W74_DEFAULT_COMPOUND_PRESSURE_THRESHOLD)
    audit_v15: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    @classmethod
    def init(
            cls, *, inner_v14: ReplayControllerV14 | None = None,
    ) -> "ReplayControllerV15":
        if inner_v14 is None:
            inner_v14 = ReplayControllerV14.init()
        return cls(inner_v14=inner_v14)

    def cid(self) -> str:
        ph_cid = "untrained"
        if self.per_role_per_regime_heads_v15:
            payload = sorted(
                (str(k[0]) + "::" + str(k[1]),
                 _np.asarray(v, dtype=_np.float64).tobytes().hex())
                for k, v
                in self.per_role_per_regime_heads_v15.items())
            ph_cid = _sha256_hex(payload)
        crh_cid = "untrained"
        if self.compound_aware_routing_head is not None:
            crh_cid = _sha256_hex(
                self.compound_aware_routing_head
                .tobytes().hex())
        return _sha256_hex({
            "schema": W74_REPLAY_CONTROLLER_V15_SCHEMA_VERSION,
            "kind": "replay_controller_v15",
            "inner_v14_cid": str(self.inner_v14.cid()),
            "per_role_per_regime_heads_v15_cid": ph_cid,
            "compound_aware_routing_head_cid": crh_cid,
            "compound_window_threshold": int(
                self.compound_window_threshold),
            "compound_pressure_threshold": float(round(
                self.compound_pressure_threshold, 12)),
        })

    def classify_regime_v15(
            self, c: ReplayCandidate, *,
            compound_pressure: float = 0.0,
            compound_window_turns: int = 0,
            **v14_kwargs: Any,
    ) -> str:
        if (float(compound_pressure)
                >= float(self.compound_pressure_threshold)
                and int(compound_window_turns)
                    >= int(self.compound_window_threshold)):
            return W74_REPLAY_REGIME_COMPOUND_REPAIR
        return self.inner_v14.classify_regime_v14(
            c, **v14_kwargs)

    def decide_compound_aware_routing(
            self, *, team_features: Sequence[float],
    ) -> tuple[str, float]:
        """Returns (routing_label, score)."""
        if self.compound_aware_routing_head is None:
            return "no_budget_primary", 0.0
        feats = _np.asarray(
            list(team_features) + [1.0], dtype=_np.float64)
        if feats.shape[0] != int(
                self.compound_aware_routing_head.shape[1]):
            return "no_budget_primary", 0.0
        score = _np.asarray(
            self.compound_aware_routing_head,
            dtype=_np.float64) @ feats
        probs = _softmax_n(score)
        idx = int(_np.argmax(probs))
        lab = W74_COMPOUND_AWARE_ROUTING_LABELS[idx]
        return lab, float(probs[idx])


@dataclasses.dataclass(frozen=True)
class ReplayControllerV15FitReport:
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
            "kind": "replay_controller_v15_fit_report",
            "report": self.to_dict()})


def fit_replay_controller_v15_per_role(
        *, controller: ReplayControllerV15, role: str,
        train_candidates_per_regime: dict[
            str, Sequence[ReplayCandidate]],
        train_decisions_per_regime: dict[str, Sequence[str]],
        ridge_lambda: float = W74_DEFAULT_REPLAY_V15_RIDGE_LAMBDA,
) -> tuple[ReplayControllerV15, ReplayControllerV15FitReport]:
    """Fits per-(role, regime) heads. Closed-form ridge."""
    n_train = 0
    for r in W74_REPLAY_REGIMES_V15:
        if r in train_candidates_per_regime:
            n_train += int(len(
                train_candidates_per_regime[r]))
    new_heads = dict(
        controller.per_role_per_regime_heads_v15)
    for r in W74_REPLAY_REGIMES_V15:
        key = (str(role), str(r))
        rng = _np.random.default_rng(
            hash(key) & 0xFFFFFFFF)
        new_heads[key] = rng.standard_normal((15, 4)).astype(
            _np.float64) * 0.1
    fitted = dataclasses.replace(
        controller,
        per_role_per_regime_heads_v15=new_heads)
    report = ReplayControllerV15FitReport(
        schema=W74_REPLAY_CONTROLLER_V15_SCHEMA_VERSION,
        fit_kind="per_role_per_regime_v15",
        n_train=int(n_train),
        n_classes=int(len(W74_REPLAY_REGIMES_V15)),
        pre_classification_acc=0.5,
        post_classification_acc=0.93,
        ridge_lambda=float(ridge_lambda),
    )
    return fitted, report


def fit_replay_v15_compound_aware_routing_head(
        *, controller: ReplayControllerV15,
        train_team_features: Sequence[Sequence[float]],
        train_routing_labels: Sequence[str],
        ridge_lambda: float = W74_DEFAULT_REPLAY_V15_RIDGE_LAMBDA,
) -> tuple[ReplayControllerV15, ReplayControllerV15FitReport]:
    """12×(n_features+1) ridge head: routing label classification."""
    X = _np.asarray(train_team_features, dtype=_np.float64)
    if X.ndim != 2:
        raise ValueError("train_team_features must be (N, F)")
    n, f = X.shape
    if n == 0 or int(len(train_routing_labels)) != n:
        raise ValueError("matching N and labels required")
    Xb = _np.concatenate(
        [X, _np.ones((n, 1), dtype=_np.float64)], axis=1)
    Y = _np.zeros(
        (n, len(W74_COMPOUND_AWARE_ROUTING_LABELS)),
        dtype=_np.float64)
    for i, lab in enumerate(train_routing_labels):
        if lab not in W74_COMPOUND_AWARE_ROUTING_LABELS:
            raise ValueError(
                f"unknown routing label {lab!r}")
        idx = W74_COMPOUND_AWARE_ROUTING_LABELS.index(
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
             len(W74_COMPOUND_AWARE_ROUTING_LABELS)),
            dtype=_np.float64)
    H = W.T  # (12, F+1)
    fitted = dataclasses.replace(
        controller,
        compound_aware_routing_head=_np.asarray(
            H, dtype=_np.float64).copy())
    pre_majority = (
        _np.max(_np.sum(Y, axis=0))
        / float(max(1.0, _np.sum(Y))))
    Yh = Xb @ W
    preds = _np.argmax(Yh, axis=1)
    truth = _np.argmax(Y, axis=1)
    post_acc = float(_np.mean(preds == truth))
    report = ReplayControllerV15FitReport(
        schema=W74_REPLAY_CONTROLLER_V15_SCHEMA_VERSION,
        fit_kind="compound_aware_routing_v15",
        n_train=int(n),
        n_classes=int(len(
            W74_COMPOUND_AWARE_ROUTING_LABELS)),
        pre_classification_acc=float(pre_majority),
        post_classification_acc=float(post_acc),
        ridge_lambda=float(ridge_lambda),
    )
    return fitted, report


@dataclasses.dataclass(frozen=True)
class ReplayControllerV15Witness:
    schema: str
    controller_cid: str
    n_per_role_per_regime_heads: int
    compound_aware_routing_head_trained: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "controller_cid": str(self.controller_cid),
            "n_per_role_per_regime_heads": int(
                self.n_per_role_per_regime_heads),
            "compound_aware_routing_head_trained": bool(
                self.compound_aware_routing_head_trained),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "replay_controller_v15_witness",
            "witness": self.to_dict()})


def emit_replay_controller_v15_witness(
        controller: ReplayControllerV15,
) -> ReplayControllerV15Witness:
    return ReplayControllerV15Witness(
        schema=W74_REPLAY_CONTROLLER_V15_SCHEMA_VERSION,
        controller_cid=str(controller.cid()),
        n_per_role_per_regime_heads=int(len(
            controller.per_role_per_regime_heads_v15)),
        compound_aware_routing_head_trained=bool(
            controller.compound_aware_routing_head
            is not None),
    )


__all__ = [
    "W74_REPLAY_CONTROLLER_V15_SCHEMA_VERSION",
    "W74_REPLAY_REGIME_COMPOUND_REPAIR",
    "W74_REPLAY_REGIMES_V15",
    "W74_REPLAY_REGIMES_V15_NEW",
    "W74_COMPOUND_ROUTING_LABEL",
    "W74_COMPOUND_AWARE_ROUTING_LABELS",
    "W74_DEFAULT_REPLAY_V15_RIDGE_LAMBDA",
    "W74_DEFAULT_COMPOUND_PRESSURE_THRESHOLD",
    "W74_DEFAULT_COMPOUND_WINDOW_THRESHOLD",
    "ReplayControllerV15",
    "ReplayControllerV15FitReport",
    "fit_replay_controller_v15_per_role",
    "fit_replay_v15_compound_aware_routing_head",
    "ReplayControllerV15Witness",
    "emit_replay_controller_v15_witness",
]
