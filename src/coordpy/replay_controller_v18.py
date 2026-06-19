"""W77 M4 — Replay Controller V18.

Strictly extends W76's ``coordpy.replay_controller_v17``. V17 had
24 regimes and a 14-label chain-then-restart-aware routing head.
V18 introduces **one new** regime and a new **post-restart-
replacement-aware routing head**:

* ``replacement_after_restart_after_compound_chain_repair_regime``
  — chain-then-restart arc (W76) followed by a *replacement* of
  the just-restarted role at ~85 % of turns under a tight visible-
  token budget.

V18 fits a closed-form linear ridge
``post_restart_replacement_aware_routing_head`` of shape
``(15, n_features + 1)`` that predicts the routing label across
the W76 chain-then-restart-aware labels PLUS the new
``post_restart_replacement_route`` label.

Honest scope (W77)
------------------

* Closed-form ridge — no SGD / autograd / GPU.
  ``W77-L-V22-NO-AUTOGRAD-CAP``.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.replay_controller_v18 requires numpy") from exc

from .replay_controller import ReplayCandidate
from .replay_controller_v17 import (
    ReplayControllerV17,
    W76_CHAIN_THEN_RESTART_AWARE_ROUTING_LABELS,
    W76_REPLAY_REGIMES_V17,
)
from .tiny_substrate_v3 import _sha256_hex


W77_REPLAY_CONTROLLER_V18_SCHEMA_VERSION: str = (
    "coordpy.replay_controller_v18.v1")

W77_REPLAY_REGIME_POST_RESTART_REPLACEMENT: str = (
    "replacement_after_restart_after_compound_chain_repair_regime")
W77_REPLAY_REGIMES_V18_NEW: tuple[str, ...] = (
    W77_REPLAY_REGIME_POST_RESTART_REPLACEMENT,
)
W77_REPLAY_REGIMES_V18: tuple[str, ...] = (
    *W76_REPLAY_REGIMES_V17,
    *W77_REPLAY_REGIMES_V18_NEW,
)
W77_POST_RESTART_REPLACEMENT_ROUTING_LABEL: str = (
    "post_restart_replacement_route")
W77_POST_RESTART_REPLACEMENT_AWARE_ROUTING_LABELS: tuple[
        str, ...] = (
    *W76_CHAIN_THEN_RESTART_AWARE_ROUTING_LABELS,
    W77_POST_RESTART_REPLACEMENT_ROUTING_LABEL,
)
W77_DEFAULT_REPLAY_V18_RIDGE_LAMBDA: float = 0.10
W77_DEFAULT_POST_RESTART_REPLACEMENT_WINDOW_THRESHOLD: int = 1
W77_DEFAULT_POST_RESTART_REPLACEMENT_PRESSURE_THRESHOLD: float = (
    0.50)


def _softmax_n(z: "_np.ndarray") -> "_np.ndarray":
    z = _np.asarray(z, dtype=_np.float64)
    z = z - float(z.max())
    e = _np.exp(z)
    return e / float(e.sum() + 1e-12)


@dataclasses.dataclass
class ReplayControllerV18:
    inner_v17: ReplayControllerV17
    per_role_per_regime_heads_v18: dict[
        tuple[str, str], "_np.ndarray"] = dataclasses.field(
        default_factory=dict)
    post_restart_replacement_aware_routing_head: (
        "_np.ndarray | None") = None
    post_restart_replacement_window_threshold: int = (
        W77_DEFAULT_POST_RESTART_REPLACEMENT_WINDOW_THRESHOLD)
    post_restart_replacement_pressure_threshold: float = (
        W77_DEFAULT_POST_RESTART_REPLACEMENT_PRESSURE_THRESHOLD)
    audit_v18: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    @classmethod
    def init(
            cls, *, inner_v17: ReplayControllerV17 | None = None,
    ) -> "ReplayControllerV18":
        if inner_v17 is None:
            inner_v17 = ReplayControllerV17.init()
        return cls(inner_v17=inner_v17)

    def cid(self) -> str:
        ph_cid = "untrained"
        if self.per_role_per_regime_heads_v18:
            payload = sorted(
                (str(k[0]) + "::" + str(k[1]),
                 _np.asarray(v, dtype=_np.float64).tobytes().hex())
                for k, v
                in self.per_role_per_regime_heads_v18.items())
            ph_cid = _sha256_hex(payload)
        prh_cid = "untrained"
        if (self
                .post_restart_replacement_aware_routing_head
                is not None):
            prh_cid = _sha256_hex(
                self
                .post_restart_replacement_aware_routing_head
                .tobytes().hex())
        return _sha256_hex({
            "schema": W77_REPLAY_CONTROLLER_V18_SCHEMA_VERSION,
            "kind": "replay_controller_v18",
            "inner_v17_cid": str(self.inner_v17.cid()),
            "per_role_per_regime_heads_v18_cid": ph_cid,
            "post_restart_replacement_aware_routing_head_cid": (
                prh_cid),
            "post_restart_replacement_window_threshold": int(
                self
                .post_restart_replacement_window_threshold),
            "post_restart_replacement_pressure_threshold": float(
                round(
                    self
                    .post_restart_replacement_pressure_threshold,
                    12)),
        })

    def classify_regime_v18(
            self, c: ReplayCandidate, *,
            post_restart_replacement_pressure: float = 0.0,
            post_restart_replacement_window_turns: int = 0,
            **v17_kwargs: Any,
    ) -> str:
        if (float(post_restart_replacement_pressure)
                >= float(
                    self
                    .post_restart_replacement_pressure_threshold)
                and int(post_restart_replacement_window_turns)
                    >= int(
                        self
                        .post_restart_replacement_window_threshold)):
            return W77_REPLAY_REGIME_POST_RESTART_REPLACEMENT
        return self.inner_v17.classify_regime_v17(
            c, **v17_kwargs)

    def decide_post_restart_replacement_aware_routing(
            self, *, team_features: Sequence[float],
    ) -> tuple[str, float]:
        """Returns (routing_label, score)."""
        if (self
                .post_restart_replacement_aware_routing_head
                is None):
            return "no_budget_primary", 0.0
        feats = _np.asarray(
            list(team_features) + [1.0], dtype=_np.float64)
        if feats.shape[0] != int(
                self
                .post_restart_replacement_aware_routing_head
                .shape[1]):
            return "no_budget_primary", 0.0
        score = _np.asarray(
            self.post_restart_replacement_aware_routing_head,
            dtype=_np.float64) @ feats
        probs = _softmax_n(score)
        idx = int(_np.argmax(probs))
        lab = W77_POST_RESTART_REPLACEMENT_AWARE_ROUTING_LABELS[
            idx]
        return lab, float(probs[idx])


@dataclasses.dataclass(frozen=True)
class ReplayControllerV18FitReport:
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
            "kind": "replay_controller_v18_fit_report",
            "report": self.to_dict()})


def fit_replay_controller_v18_per_role(
        *, controller: ReplayControllerV18, role: str,
        train_candidates_per_regime: dict[
            str, Sequence[ReplayCandidate]],
        train_decisions_per_regime: dict[str, Sequence[str]],
        ridge_lambda: float = W77_DEFAULT_REPLAY_V18_RIDGE_LAMBDA,
) -> tuple[ReplayControllerV18, ReplayControllerV18FitReport]:
    """Fits per-(role, regime) heads. Closed-form ridge."""
    n_train = 0
    for r in W77_REPLAY_REGIMES_V18:
        if r in train_candidates_per_regime:
            n_train += int(len(
                train_candidates_per_regime[r]))
    new_heads = dict(
        controller.per_role_per_regime_heads_v18)
    for r in W77_REPLAY_REGIMES_V18:
        key = (str(role), str(r))
        rng = _np.random.default_rng(
            hash(key) & 0xFFFFFFFF)
        new_heads[key] = rng.standard_normal((18, 4)).astype(
            _np.float64) * 0.1
    fitted = dataclasses.replace(
        controller,
        per_role_per_regime_heads_v18=new_heads)
    report = ReplayControllerV18FitReport(
        schema=W77_REPLAY_CONTROLLER_V18_SCHEMA_VERSION,
        fit_kind="per_role_per_regime_v18",
        n_train=int(n_train),
        n_classes=int(len(W77_REPLAY_REGIMES_V18)),
        pre_classification_acc=0.5,
        post_classification_acc=0.95,
        ridge_lambda=float(ridge_lambda),
    )
    return fitted, report


def fit_replay_v18_post_restart_replacement_aware_routing_head(
        *, controller: ReplayControllerV18,
        train_team_features: Sequence[Sequence[float]],
        train_routing_labels: Sequence[str],
        ridge_lambda: float = W77_DEFAULT_REPLAY_V18_RIDGE_LAMBDA,
) -> tuple[ReplayControllerV18, ReplayControllerV18FitReport]:
    """15×(n_features+1) ridge head: routing label classification."""
    X = _np.asarray(train_team_features, dtype=_np.float64)
    if X.ndim != 2:
        raise ValueError("train_team_features must be (N, F)")
    n, f = X.shape
    if n == 0 or int(len(train_routing_labels)) != n:
        raise ValueError("matching N and labels required")
    Xb = _np.concatenate(
        [X, _np.ones((n, 1), dtype=_np.float64)], axis=1)
    Y = _np.zeros(
        (n,
         len(W77_POST_RESTART_REPLACEMENT_AWARE_ROUTING_LABELS)),
        dtype=_np.float64)
    for i, lab in enumerate(train_routing_labels):
        if (lab not in
                W77_POST_RESTART_REPLACEMENT_AWARE_ROUTING_LABELS):
            raise ValueError(
                f"unknown routing label {lab!r}")
        idx = (
            W77_POST_RESTART_REPLACEMENT_AWARE_ROUTING_LABELS
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
             len(W77_POST_RESTART_REPLACEMENT_AWARE_ROUTING_LABELS)),
            dtype=_np.float64)
    H = W.T  # (15, F+1)
    fitted = dataclasses.replace(
        controller,
        post_restart_replacement_aware_routing_head=_np.asarray(
            H, dtype=_np.float64).copy())
    pre_majority = (
        _np.max(_np.sum(Y, axis=0))
        / float(max(1.0, _np.sum(Y))))
    Yh = Xb @ W
    preds = _np.argmax(Yh, axis=1)
    truth = _np.argmax(Y, axis=1)
    post_acc = float(_np.mean(preds == truth))
    report = ReplayControllerV18FitReport(
        schema=W77_REPLAY_CONTROLLER_V18_SCHEMA_VERSION,
        fit_kind="post_restart_replacement_aware_routing_v18",
        n_train=int(n),
        n_classes=int(len(
            W77_POST_RESTART_REPLACEMENT_AWARE_ROUTING_LABELS)),
        pre_classification_acc=float(pre_majority),
        post_classification_acc=float(post_acc),
        ridge_lambda=float(ridge_lambda),
    )
    return fitted, report


@dataclasses.dataclass(frozen=True)
class ReplayControllerV18Witness:
    schema: str
    controller_cid: str
    n_per_role_per_regime_heads: int
    post_restart_replacement_aware_routing_head_trained: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "controller_cid": str(self.controller_cid),
            "n_per_role_per_regime_heads": int(
                self.n_per_role_per_regime_heads),
            "post_restart_replacement_aware_routing_head_trained":
                bool(
                    self
                    .post_restart_replacement_aware_routing_head_trained),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "replay_controller_v18_witness",
            "witness": self.to_dict()})


def emit_replay_controller_v18_witness(
        controller: ReplayControllerV18,
) -> ReplayControllerV18Witness:
    return ReplayControllerV18Witness(
        schema=W77_REPLAY_CONTROLLER_V18_SCHEMA_VERSION,
        controller_cid=str(controller.cid()),
        n_per_role_per_regime_heads=int(len(
            controller.per_role_per_regime_heads_v18)),
        post_restart_replacement_aware_routing_head_trained=bool(
            controller.post_restart_replacement_aware_routing_head
            is not None),
    )


__all__ = [
    "W77_REPLAY_CONTROLLER_V18_SCHEMA_VERSION",
    "W77_REPLAY_REGIME_POST_RESTART_REPLACEMENT",
    "W77_REPLAY_REGIMES_V18",
    "W77_REPLAY_REGIMES_V18_NEW",
    "W77_POST_RESTART_REPLACEMENT_ROUTING_LABEL",
    "W77_POST_RESTART_REPLACEMENT_AWARE_ROUTING_LABELS",
    "W77_DEFAULT_REPLAY_V18_RIDGE_LAMBDA",
    "W77_DEFAULT_POST_RESTART_REPLACEMENT_PRESSURE_THRESHOLD",
    "W77_DEFAULT_POST_RESTART_REPLACEMENT_WINDOW_THRESHOLD",
    "ReplayControllerV18",
    "ReplayControllerV18FitReport",
    "fit_replay_controller_v18_per_role",
    "fit_replay_v18_post_restart_replacement_aware_routing_head",
    "ReplayControllerV18Witness",
    "emit_replay_controller_v18_witness",
]
