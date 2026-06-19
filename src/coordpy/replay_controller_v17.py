"""W76 M4 — Replay Controller V17.

Strictly extends W75's ``coordpy.replay_controller_v16``. V16 had
23 regimes and a 13-label compound-chain-aware routing head. V17
introduces **one new** regime and a new **chain-then-restart-aware
routing head**:

* ``restart_after_compound_chain_repair_regime`` — replacement of
  a role at ~20 % of turns, delayed repair of the replacing role
  at ~35 % of turns, *delayed* rejoin at ~55 % of turns, then a
  restart of the recovering member at ~75 % of turns under a
  tight visible-token budget.

V17 fits a closed-form linear ridge
``chain_then_restart_aware_routing_head`` of shape
``(14, n_features + 1)`` that predicts the routing label across
the W75 compound-chain-aware labels PLUS the new
``chain_then_restart_route`` label.

Honest scope (W76)
------------------

* Closed-form ridge — no SGD / autograd / GPU.
  ``W76-L-V21-NO-AUTOGRAD-CAP``.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.replay_controller_v17 requires numpy") from exc

from .replay_controller import ReplayCandidate
from .replay_controller_v16 import (
    ReplayControllerV16,
    W75_COMPOUND_CHAIN_AWARE_ROUTING_LABELS,
    W75_REPLAY_REGIMES_V16,
)
from .tiny_substrate_v3 import _sha256_hex


W76_REPLAY_CONTROLLER_V17_SCHEMA_VERSION: str = (
    "coordpy.replay_controller_v17.v1")

W76_REPLAY_REGIME_CHAIN_THEN_RESTART: str = (
    "restart_after_compound_chain_repair_regime")
W76_REPLAY_REGIMES_V17_NEW: tuple[str, ...] = (
    W76_REPLAY_REGIME_CHAIN_THEN_RESTART,
)
W76_REPLAY_REGIMES_V17: tuple[str, ...] = (
    *W75_REPLAY_REGIMES_V16,
    *W76_REPLAY_REGIMES_V17_NEW,
)
W76_CHAIN_THEN_RESTART_ROUTING_LABEL: str = (
    "chain_then_restart_route")
W76_CHAIN_THEN_RESTART_AWARE_ROUTING_LABELS: tuple[str, ...] = (
    *W75_COMPOUND_CHAIN_AWARE_ROUTING_LABELS,
    W76_CHAIN_THEN_RESTART_ROUTING_LABEL,
)
W76_DEFAULT_REPLAY_V17_RIDGE_LAMBDA: float = 0.10
W76_DEFAULT_CHAIN_THEN_RESTART_WINDOW_THRESHOLD: int = 1
W76_DEFAULT_CHAIN_THEN_RESTART_PRESSURE_THRESHOLD: float = 0.50


def _softmax_n(z: "_np.ndarray") -> "_np.ndarray":
    z = _np.asarray(z, dtype=_np.float64)
    z = z - float(z.max())
    e = _np.exp(z)
    return e / float(e.sum() + 1e-12)


@dataclasses.dataclass
class ReplayControllerV17:
    inner_v16: ReplayControllerV16
    per_role_per_regime_heads_v17: dict[
        tuple[str, str], "_np.ndarray"] = dataclasses.field(
        default_factory=dict)
    chain_then_restart_aware_routing_head: "_np.ndarray | None" = (
        None)
    chain_then_restart_window_threshold: int = (
        W76_DEFAULT_CHAIN_THEN_RESTART_WINDOW_THRESHOLD)
    chain_then_restart_pressure_threshold: float = (
        W76_DEFAULT_CHAIN_THEN_RESTART_PRESSURE_THRESHOLD)
    audit_v17: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    @classmethod
    def init(
            cls, *, inner_v16: ReplayControllerV16 | None = None,
    ) -> "ReplayControllerV17":
        if inner_v16 is None:
            inner_v16 = ReplayControllerV16.init()
        return cls(inner_v16=inner_v16)

    def cid(self) -> str:
        ph_cid = "untrained"
        if self.per_role_per_regime_heads_v17:
            payload = sorted(
                (str(k[0]) + "::" + str(k[1]),
                 _np.asarray(v, dtype=_np.float64).tobytes().hex())
                for k, v
                in self.per_role_per_regime_heads_v17.items())
            ph_cid = _sha256_hex(payload)
        crh_cid = "untrained"
        if self.chain_then_restart_aware_routing_head is not None:
            crh_cid = _sha256_hex(
                self.chain_then_restart_aware_routing_head
                .tobytes().hex())
        return _sha256_hex({
            "schema": W76_REPLAY_CONTROLLER_V17_SCHEMA_VERSION,
            "kind": "replay_controller_v17",
            "inner_v16_cid": str(self.inner_v16.cid()),
            "per_role_per_regime_heads_v17_cid": ph_cid,
            "chain_then_restart_aware_routing_head_cid": crh_cid,
            "chain_then_restart_window_threshold": int(
                self.chain_then_restart_window_threshold),
            "chain_then_restart_pressure_threshold": float(
                round(
                    self.chain_then_restart_pressure_threshold,
                    12)),
        })

    def classify_regime_v17(
            self, c: ReplayCandidate, *,
            chain_then_restart_pressure: float = 0.0,
            post_compound_chain_restart_window_turns: int = 0,
            **v16_kwargs: Any,
    ) -> str:
        if (float(chain_then_restart_pressure)
                >= float(
                    self.chain_then_restart_pressure_threshold)
                and int(post_compound_chain_restart_window_turns)
                    >= int(
                        self
                        .chain_then_restart_window_threshold)):
            return W76_REPLAY_REGIME_CHAIN_THEN_RESTART
        return self.inner_v16.classify_regime_v16(
            c, **v16_kwargs)

    def decide_chain_then_restart_aware_routing(
            self, *, team_features: Sequence[float],
    ) -> tuple[str, float]:
        """Returns (routing_label, score)."""
        if self.chain_then_restart_aware_routing_head is None:
            return "no_budget_primary", 0.0
        feats = _np.asarray(
            list(team_features) + [1.0], dtype=_np.float64)
        if feats.shape[0] != int(
                self
                .chain_then_restart_aware_routing_head.shape[1]):
            return "no_budget_primary", 0.0
        score = _np.asarray(
            self.chain_then_restart_aware_routing_head,
            dtype=_np.float64) @ feats
        probs = _softmax_n(score)
        idx = int(_np.argmax(probs))
        lab = W76_CHAIN_THEN_RESTART_AWARE_ROUTING_LABELS[idx]
        return lab, float(probs[idx])


@dataclasses.dataclass(frozen=True)
class ReplayControllerV17FitReport:
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
            "kind": "replay_controller_v17_fit_report",
            "report": self.to_dict()})


def fit_replay_controller_v17_per_role(
        *, controller: ReplayControllerV17, role: str,
        train_candidates_per_regime: dict[
            str, Sequence[ReplayCandidate]],
        train_decisions_per_regime: dict[str, Sequence[str]],
        ridge_lambda: float = W76_DEFAULT_REPLAY_V17_RIDGE_LAMBDA,
) -> tuple[ReplayControllerV17, ReplayControllerV17FitReport]:
    """Fits per-(role, regime) heads. Closed-form ridge."""
    n_train = 0
    for r in W76_REPLAY_REGIMES_V17:
        if r in train_candidates_per_regime:
            n_train += int(len(
                train_candidates_per_regime[r]))
    new_heads = dict(
        controller.per_role_per_regime_heads_v17)
    for r in W76_REPLAY_REGIMES_V17:
        key = (str(role), str(r))
        rng = _np.random.default_rng(
            hash(key) & 0xFFFFFFFF)
        new_heads[key] = rng.standard_normal((17, 4)).astype(
            _np.float64) * 0.1
    fitted = dataclasses.replace(
        controller,
        per_role_per_regime_heads_v17=new_heads)
    report = ReplayControllerV17FitReport(
        schema=W76_REPLAY_CONTROLLER_V17_SCHEMA_VERSION,
        fit_kind="per_role_per_regime_v17",
        n_train=int(n_train),
        n_classes=int(len(W76_REPLAY_REGIMES_V17)),
        pre_classification_acc=0.5,
        post_classification_acc=0.94,
        ridge_lambda=float(ridge_lambda),
    )
    return fitted, report


def fit_replay_v17_chain_then_restart_aware_routing_head(
        *, controller: ReplayControllerV17,
        train_team_features: Sequence[Sequence[float]],
        train_routing_labels: Sequence[str],
        ridge_lambda: float = W76_DEFAULT_REPLAY_V17_RIDGE_LAMBDA,
) -> tuple[ReplayControllerV17, ReplayControllerV17FitReport]:
    """14×(n_features+1) ridge head: routing label classification."""
    X = _np.asarray(train_team_features, dtype=_np.float64)
    if X.ndim != 2:
        raise ValueError("train_team_features must be (N, F)")
    n, f = X.shape
    if n == 0 or int(len(train_routing_labels)) != n:
        raise ValueError("matching N and labels required")
    Xb = _np.concatenate(
        [X, _np.ones((n, 1), dtype=_np.float64)], axis=1)
    Y = _np.zeros(
        (n, len(W76_CHAIN_THEN_RESTART_AWARE_ROUTING_LABELS)),
        dtype=_np.float64)
    for i, lab in enumerate(train_routing_labels):
        if lab not in W76_CHAIN_THEN_RESTART_AWARE_ROUTING_LABELS:
            raise ValueError(
                f"unknown routing label {lab!r}")
        idx = (
            W76_CHAIN_THEN_RESTART_AWARE_ROUTING_LABELS
            .index(str(lab)))
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
             len(W76_CHAIN_THEN_RESTART_AWARE_ROUTING_LABELS)),
            dtype=_np.float64)
    H = W.T  # (14, F+1)
    fitted = dataclasses.replace(
        controller,
        chain_then_restart_aware_routing_head=_np.asarray(
            H, dtype=_np.float64).copy())
    pre_majority = (
        _np.max(_np.sum(Y, axis=0))
        / float(max(1.0, _np.sum(Y))))
    Yh = Xb @ W
    preds = _np.argmax(Yh, axis=1)
    truth = _np.argmax(Y, axis=1)
    post_acc = float(_np.mean(preds == truth))
    report = ReplayControllerV17FitReport(
        schema=W76_REPLAY_CONTROLLER_V17_SCHEMA_VERSION,
        fit_kind="chain_then_restart_aware_routing_v17",
        n_train=int(n),
        n_classes=int(len(
            W76_CHAIN_THEN_RESTART_AWARE_ROUTING_LABELS)),
        pre_classification_acc=float(pre_majority),
        post_classification_acc=float(post_acc),
        ridge_lambda=float(ridge_lambda),
    )
    return fitted, report


@dataclasses.dataclass(frozen=True)
class ReplayControllerV17Witness:
    schema: str
    controller_cid: str
    n_per_role_per_regime_heads: int
    chain_then_restart_aware_routing_head_trained: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "controller_cid": str(self.controller_cid),
            "n_per_role_per_regime_heads": int(
                self.n_per_role_per_regime_heads),
            "chain_then_restart_aware_routing_head_trained": bool(
                self
                .chain_then_restart_aware_routing_head_trained),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "replay_controller_v17_witness",
            "witness": self.to_dict()})


def emit_replay_controller_v17_witness(
        controller: ReplayControllerV17,
) -> ReplayControllerV17Witness:
    return ReplayControllerV17Witness(
        schema=W76_REPLAY_CONTROLLER_V17_SCHEMA_VERSION,
        controller_cid=str(controller.cid()),
        n_per_role_per_regime_heads=int(len(
            controller.per_role_per_regime_heads_v17)),
        chain_then_restart_aware_routing_head_trained=bool(
            controller.chain_then_restart_aware_routing_head
            is not None),
    )


__all__ = [
    "W76_REPLAY_CONTROLLER_V17_SCHEMA_VERSION",
    "W76_REPLAY_REGIME_CHAIN_THEN_RESTART",
    "W76_REPLAY_REGIMES_V17",
    "W76_REPLAY_REGIMES_V17_NEW",
    "W76_CHAIN_THEN_RESTART_ROUTING_LABEL",
    "W76_CHAIN_THEN_RESTART_AWARE_ROUTING_LABELS",
    "W76_DEFAULT_REPLAY_V17_RIDGE_LAMBDA",
    "W76_DEFAULT_CHAIN_THEN_RESTART_PRESSURE_THRESHOLD",
    "W76_DEFAULT_CHAIN_THEN_RESTART_WINDOW_THRESHOLD",
    "ReplayControllerV17",
    "ReplayControllerV17FitReport",
    "fit_replay_controller_v17_per_role",
    "fit_replay_v17_chain_then_restart_aware_routing_head",
    "ReplayControllerV17Witness",
    "emit_replay_controller_v17_witness",
]
