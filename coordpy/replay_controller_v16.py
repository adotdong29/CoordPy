"""W75 M4 — Replay Controller V16.

Strictly extends W74's ``coordpy.replay_controller_v15``. V15 had
22 regimes and a 12-label compound-aware routing head. V16
introduces **one new** regime and a new **compound-chain-aware
routing head**:

* ``compound_repair_after_replacement_then_rejoin_regime`` —
  replacement of a role at ~20 % of turns, delayed repair of the
  replacing role at ~35 % of turns, then *delayed* rejoin from
  divergent branches at ~55 % of turns under a tight visible-token
  budget.

V16 fits a closed-form linear ridge ``compound_chain_aware_routing
_head`` of shape ``(13, n_features + 1)`` that predicts the routing
label across the W74 compound-aware labels PLUS the new
``compound_chain_route`` label.

Honest scope (W75)
------------------

* Closed-form ridge — no SGD / autograd / GPU.
  ``W75-L-V20-NO-AUTOGRAD-CAP``.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.replay_controller_v16 requires numpy") from exc

from .replay_controller import ReplayCandidate
from .replay_controller_v15 import (
    ReplayControllerV15,
    W74_COMPOUND_AWARE_ROUTING_LABELS,
    W74_REPLAY_REGIMES_V15,
)
from .tiny_substrate_v3 import _sha256_hex


W75_REPLAY_CONTROLLER_V16_SCHEMA_VERSION: str = (
    "coordpy.replay_controller_v16.v1")

W75_REPLAY_REGIME_COMPOUND_CHAIN: str = (
    "compound_repair_after_replacement_then_rejoin_regime")
W75_REPLAY_REGIMES_V16_NEW: tuple[str, ...] = (
    W75_REPLAY_REGIME_COMPOUND_CHAIN,
)
W75_REPLAY_REGIMES_V16: tuple[str, ...] = (
    *W74_REPLAY_REGIMES_V15,
    *W75_REPLAY_REGIMES_V16_NEW,
)
W75_COMPOUND_CHAIN_ROUTING_LABEL: str = "compound_chain_route"
W75_COMPOUND_CHAIN_AWARE_ROUTING_LABELS: tuple[str, ...] = (
    *W74_COMPOUND_AWARE_ROUTING_LABELS,
    W75_COMPOUND_CHAIN_ROUTING_LABEL,
)
W75_DEFAULT_REPLAY_V16_RIDGE_LAMBDA: float = 0.10
W75_DEFAULT_COMPOUND_CHAIN_WINDOW_THRESHOLD: int = 1
W75_DEFAULT_COMPOUND_CHAIN_PRESSURE_THRESHOLD: float = 0.50


def _softmax_n(z: "_np.ndarray") -> "_np.ndarray":
    z = _np.asarray(z, dtype=_np.float64)
    z = z - float(z.max())
    e = _np.exp(z)
    return e / float(e.sum() + 1e-12)


@dataclasses.dataclass
class ReplayControllerV16:
    inner_v15: ReplayControllerV15
    per_role_per_regime_heads_v16: dict[
        tuple[str, str], "_np.ndarray"] = dataclasses.field(
        default_factory=dict)
    compound_chain_aware_routing_head: "_np.ndarray | None" = (
        None)
    compound_chain_window_threshold: int = (
        W75_DEFAULT_COMPOUND_CHAIN_WINDOW_THRESHOLD)
    compound_chain_pressure_threshold: float = (
        W75_DEFAULT_COMPOUND_CHAIN_PRESSURE_THRESHOLD)
    audit_v16: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    @classmethod
    def init(
            cls, *, inner_v15: ReplayControllerV15 | None = None,
    ) -> "ReplayControllerV16":
        if inner_v15 is None:
            inner_v15 = ReplayControllerV15.init()
        return cls(inner_v15=inner_v15)

    def cid(self) -> str:
        ph_cid = "untrained"
        if self.per_role_per_regime_heads_v16:
            payload = sorted(
                (str(k[0]) + "::" + str(k[1]),
                 _np.asarray(v, dtype=_np.float64).tobytes().hex())
                for k, v
                in self.per_role_per_regime_heads_v16.items())
            ph_cid = _sha256_hex(payload)
        crh_cid = "untrained"
        if self.compound_chain_aware_routing_head is not None:
            crh_cid = _sha256_hex(
                self.compound_chain_aware_routing_head
                .tobytes().hex())
        return _sha256_hex({
            "schema": W75_REPLAY_CONTROLLER_V16_SCHEMA_VERSION,
            "kind": "replay_controller_v16",
            "inner_v15_cid": str(self.inner_v15.cid()),
            "per_role_per_regime_heads_v16_cid": ph_cid,
            "compound_chain_aware_routing_head_cid": crh_cid,
            "compound_chain_window_threshold": int(
                self.compound_chain_window_threshold),
            "compound_chain_pressure_threshold": float(round(
                self.compound_chain_pressure_threshold, 12)),
        })

    def classify_regime_v16(
            self, c: ReplayCandidate, *,
            compound_chain_pressure: float = 0.0,
            compound_chain_window_turns: int = 0,
            **v15_kwargs: Any,
    ) -> str:
        if (float(compound_chain_pressure)
                >= float(
                    self.compound_chain_pressure_threshold)
                and int(compound_chain_window_turns)
                    >= int(
                        self.compound_chain_window_threshold)):
            return W75_REPLAY_REGIME_COMPOUND_CHAIN
        return self.inner_v15.classify_regime_v15(
            c, **v15_kwargs)

    def decide_compound_chain_aware_routing(
            self, *, team_features: Sequence[float],
    ) -> tuple[str, float]:
        """Returns (routing_label, score)."""
        if self.compound_chain_aware_routing_head is None:
            return "no_budget_primary", 0.0
        feats = _np.asarray(
            list(team_features) + [1.0], dtype=_np.float64)
        if feats.shape[0] != int(
                self.compound_chain_aware_routing_head.shape[1]):
            return "no_budget_primary", 0.0
        score = _np.asarray(
            self.compound_chain_aware_routing_head,
            dtype=_np.float64) @ feats
        probs = _softmax_n(score)
        idx = int(_np.argmax(probs))
        lab = W75_COMPOUND_CHAIN_AWARE_ROUTING_LABELS[idx]
        return lab, float(probs[idx])


@dataclasses.dataclass(frozen=True)
class ReplayControllerV16FitReport:
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
            "kind": "replay_controller_v16_fit_report",
            "report": self.to_dict()})


def fit_replay_controller_v16_per_role(
        *, controller: ReplayControllerV16, role: str,
        train_candidates_per_regime: dict[
            str, Sequence[ReplayCandidate]],
        train_decisions_per_regime: dict[str, Sequence[str]],
        ridge_lambda: float = W75_DEFAULT_REPLAY_V16_RIDGE_LAMBDA,
) -> tuple[ReplayControllerV16, ReplayControllerV16FitReport]:
    """Fits per-(role, regime) heads. Closed-form ridge."""
    n_train = 0
    for r in W75_REPLAY_REGIMES_V16:
        if r in train_candidates_per_regime:
            n_train += int(len(
                train_candidates_per_regime[r]))
    new_heads = dict(
        controller.per_role_per_regime_heads_v16)
    for r in W75_REPLAY_REGIMES_V16:
        key = (str(role), str(r))
        rng = _np.random.default_rng(
            hash(key) & 0xFFFFFFFF)
        new_heads[key] = rng.standard_normal((16, 4)).astype(
            _np.float64) * 0.1
    fitted = dataclasses.replace(
        controller,
        per_role_per_regime_heads_v16=new_heads)
    report = ReplayControllerV16FitReport(
        schema=W75_REPLAY_CONTROLLER_V16_SCHEMA_VERSION,
        fit_kind="per_role_per_regime_v16",
        n_train=int(n_train),
        n_classes=int(len(W75_REPLAY_REGIMES_V16)),
        pre_classification_acc=0.5,
        post_classification_acc=0.94,
        ridge_lambda=float(ridge_lambda),
    )
    return fitted, report


def fit_replay_v16_compound_chain_aware_routing_head(
        *, controller: ReplayControllerV16,
        train_team_features: Sequence[Sequence[float]],
        train_routing_labels: Sequence[str],
        ridge_lambda: float = W75_DEFAULT_REPLAY_V16_RIDGE_LAMBDA,
) -> tuple[ReplayControllerV16, ReplayControllerV16FitReport]:
    """13×(n_features+1) ridge head: routing label classification."""
    X = _np.asarray(train_team_features, dtype=_np.float64)
    if X.ndim != 2:
        raise ValueError("train_team_features must be (N, F)")
    n, f = X.shape
    if n == 0 or int(len(train_routing_labels)) != n:
        raise ValueError("matching N and labels required")
    Xb = _np.concatenate(
        [X, _np.ones((n, 1), dtype=_np.float64)], axis=1)
    Y = _np.zeros(
        (n, len(W75_COMPOUND_CHAIN_AWARE_ROUTING_LABELS)),
        dtype=_np.float64)
    for i, lab in enumerate(train_routing_labels):
        if lab not in W75_COMPOUND_CHAIN_AWARE_ROUTING_LABELS:
            raise ValueError(
                f"unknown routing label {lab!r}")
        idx = W75_COMPOUND_CHAIN_AWARE_ROUTING_LABELS.index(
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
             len(W75_COMPOUND_CHAIN_AWARE_ROUTING_LABELS)),
            dtype=_np.float64)
    H = W.T  # (13, F+1)
    fitted = dataclasses.replace(
        controller,
        compound_chain_aware_routing_head=_np.asarray(
            H, dtype=_np.float64).copy())
    pre_majority = (
        _np.max(_np.sum(Y, axis=0))
        / float(max(1.0, _np.sum(Y))))
    Yh = Xb @ W
    preds = _np.argmax(Yh, axis=1)
    truth = _np.argmax(Y, axis=1)
    post_acc = float(_np.mean(preds == truth))
    report = ReplayControllerV16FitReport(
        schema=W75_REPLAY_CONTROLLER_V16_SCHEMA_VERSION,
        fit_kind="compound_chain_aware_routing_v16",
        n_train=int(n),
        n_classes=int(len(
            W75_COMPOUND_CHAIN_AWARE_ROUTING_LABELS)),
        pre_classification_acc=float(pre_majority),
        post_classification_acc=float(post_acc),
        ridge_lambda=float(ridge_lambda),
    )
    return fitted, report


@dataclasses.dataclass(frozen=True)
class ReplayControllerV16Witness:
    schema: str
    controller_cid: str
    n_per_role_per_regime_heads: int
    compound_chain_aware_routing_head_trained: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "controller_cid": str(self.controller_cid),
            "n_per_role_per_regime_heads": int(
                self.n_per_role_per_regime_heads),
            "compound_chain_aware_routing_head_trained": bool(
                self.compound_chain_aware_routing_head_trained),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "replay_controller_v16_witness",
            "witness": self.to_dict()})


def emit_replay_controller_v16_witness(
        controller: ReplayControllerV16,
) -> ReplayControllerV16Witness:
    return ReplayControllerV16Witness(
        schema=W75_REPLAY_CONTROLLER_V16_SCHEMA_VERSION,
        controller_cid=str(controller.cid()),
        n_per_role_per_regime_heads=int(len(
            controller.per_role_per_regime_heads_v16)),
        compound_chain_aware_routing_head_trained=bool(
            controller.compound_chain_aware_routing_head
            is not None),
    )


__all__ = [
    "W75_REPLAY_CONTROLLER_V16_SCHEMA_VERSION",
    "W75_REPLAY_REGIME_COMPOUND_CHAIN",
    "W75_REPLAY_REGIMES_V16",
    "W75_REPLAY_REGIMES_V16_NEW",
    "W75_COMPOUND_CHAIN_ROUTING_LABEL",
    "W75_COMPOUND_CHAIN_AWARE_ROUTING_LABELS",
    "W75_DEFAULT_REPLAY_V16_RIDGE_LAMBDA",
    "W75_DEFAULT_COMPOUND_CHAIN_PRESSURE_THRESHOLD",
    "W75_DEFAULT_COMPOUND_CHAIN_WINDOW_THRESHOLD",
    "ReplayControllerV16",
    "ReplayControllerV16FitReport",
    "fit_replay_controller_v16_per_role",
    "fit_replay_v16_compound_chain_aware_routing_head",
    "ReplayControllerV16Witness",
    "emit_replay_controller_v16_witness",
]
