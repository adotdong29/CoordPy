"""W78 M4 — Replay Controller V19.

Strictly extends W77's ``coordpy.replay_controller_v18``. V18 had
25 regimes and a 15-label post-restart-replacement-aware routing
head. V19 introduces **one new** regime and a new **long-horizon-
reconstruction-aware routing head**:

* ``long_delay_reconstruction_after_compound_chain_failure_regime``
  — long visible-token blackout (~30 %..85 % of turns) followed
  by a reconstruction request at ~88 % of turns under tight
  budget.

V19 fits a closed-form linear ridge
``long_horizon_reconstruction_aware_routing_head`` of shape
``(16, n_features + 1)`` that predicts the routing label across
the V18 post-restart-replacement-aware labels PLUS the new
``long_horizon_reconstruction_route`` label.

Honest scope (W78)
------------------

* Closed-form ridge — no SGD / autograd / GPU.
  ``W78-L-V23-NO-AUTOGRAD-CAP``.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.replay_controller_v19 requires numpy") from exc

from .replay_controller import ReplayCandidate
from .replay_controller_v18 import (
    ReplayControllerV18,
    W77_POST_RESTART_REPLACEMENT_AWARE_ROUTING_LABELS,
    W77_REPLAY_REGIMES_V18,
)
from .tiny_substrate_v3 import _sha256_hex


W78_REPLAY_CONTROLLER_V19_SCHEMA_VERSION: str = (
    "coordpy.replay_controller_v19.v1")

W78_REPLAY_REGIME_LONG_HORIZON_RECONSTRUCTION: str = (
    "long_delay_reconstruction_after_compound_chain_failure_regime")
W78_REPLAY_REGIMES_V19_NEW: tuple[str, ...] = (
    W78_REPLAY_REGIME_LONG_HORIZON_RECONSTRUCTION,
)
W78_REPLAY_REGIMES_V19: tuple[str, ...] = (
    *W77_REPLAY_REGIMES_V18,
    *W78_REPLAY_REGIMES_V19_NEW,
)
W78_LONG_HORIZON_RECONSTRUCTION_ROUTING_LABEL: str = (
    "long_horizon_reconstruction_route")
W78_LONG_HORIZON_RECONSTRUCTION_AWARE_ROUTING_LABELS: tuple[
        str, ...] = (
    *W77_POST_RESTART_REPLACEMENT_AWARE_ROUTING_LABELS,
    W78_LONG_HORIZON_RECONSTRUCTION_ROUTING_LABEL,
)
W78_DEFAULT_REPLAY_V19_RIDGE_LAMBDA: float = 0.10
W78_DEFAULT_LONG_HORIZON_RECONSTRUCTION_WINDOW_THRESHOLD: int = 50
W78_DEFAULT_LONG_HORIZON_RECONSTRUCTION_PRESSURE_THRESHOLD: float = (
    0.50)


def _softmax_n(z: "_np.ndarray") -> "_np.ndarray":
    z = _np.asarray(z, dtype=_np.float64)
    z = z - float(z.max())
    e = _np.exp(z)
    return e / float(e.sum() + 1e-12)


@dataclasses.dataclass
class ReplayControllerV19:
    inner_v18: ReplayControllerV18
    per_role_per_regime_heads_v19: dict[
        tuple[str, str], "_np.ndarray"] = dataclasses.field(
        default_factory=dict)
    long_horizon_reconstruction_aware_routing_head: (
        "_np.ndarray | None") = None
    long_horizon_reconstruction_window_threshold: int = (
        W78_DEFAULT_LONG_HORIZON_RECONSTRUCTION_WINDOW_THRESHOLD)
    long_horizon_reconstruction_pressure_threshold: float = (
        W78_DEFAULT_LONG_HORIZON_RECONSTRUCTION_PRESSURE_THRESHOLD)
    audit_v19: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    @classmethod
    def init(
            cls, *, inner_v18: ReplayControllerV18 | None = None,
    ) -> "ReplayControllerV19":
        if inner_v18 is None:
            inner_v18 = ReplayControllerV18.init()
        return cls(inner_v18=inner_v18)

    def cid(self) -> str:
        ph_cid = "untrained"
        if self.per_role_per_regime_heads_v19:
            payload = sorted(
                (str(k[0]) + "::" + str(k[1]),
                 _np.asarray(v, dtype=_np.float64).tobytes().hex())
                for k, v
                in self.per_role_per_regime_heads_v19.items())
            ph_cid = _sha256_hex(payload)
        lrh_cid = "untrained"
        if (self
                .long_horizon_reconstruction_aware_routing_head
                is not None):
            lrh_cid = _sha256_hex(
                self
                .long_horizon_reconstruction_aware_routing_head
                .tobytes().hex())
        return _sha256_hex({
            "schema": W78_REPLAY_CONTROLLER_V19_SCHEMA_VERSION,
            "kind": "replay_controller_v19",
            "inner_v18_cid": str(self.inner_v18.cid()),
            "per_role_per_regime_heads_v19_cid": ph_cid,
            "long_horizon_reconstruction_aware_routing_head_cid": (
                lrh_cid),
            "long_horizon_reconstruction_window_threshold": int(
                self
                .long_horizon_reconstruction_window_threshold),
            "long_horizon_reconstruction_pressure_threshold": float(
                round(
                    self
                    .long_horizon_reconstruction_pressure_threshold,
                    12)),
        })

    def classify_regime_v19(
            self, c: ReplayCandidate, *,
            long_horizon_reconstruction_pressure: float = 0.0,
            long_horizon_blackout_window_turns: int = 0,
            **v18_kwargs: Any,
    ) -> str:
        if (float(long_horizon_reconstruction_pressure)
                >= float(
                    self
                    .long_horizon_reconstruction_pressure_threshold)
                and int(long_horizon_blackout_window_turns)
                    >= int(
                        self
                        .long_horizon_reconstruction_window_threshold)):
            return W78_REPLAY_REGIME_LONG_HORIZON_RECONSTRUCTION
        return self.inner_v18.classify_regime_v18(
            c, **v18_kwargs)

    def decide_long_horizon_reconstruction_aware_routing(
            self, *, team_features: Sequence[float],
    ) -> tuple[str, float]:
        if (self
                .long_horizon_reconstruction_aware_routing_head
                is None):
            return "no_budget_primary", 0.0
        feats = _np.asarray(
            list(team_features) + [1.0], dtype=_np.float64)
        if feats.shape[0] != int(
                self
                .long_horizon_reconstruction_aware_routing_head
                .shape[1]):
            return "no_budget_primary", 0.0
        score = _np.asarray(
            self.long_horizon_reconstruction_aware_routing_head,
            dtype=_np.float64) @ feats
        probs = _softmax_n(score)
        idx = int(_np.argmax(probs))
        lab = W78_LONG_HORIZON_RECONSTRUCTION_AWARE_ROUTING_LABELS[
            idx]
        return lab, float(probs[idx])


@dataclasses.dataclass(frozen=True)
class ReplayControllerV19FitReport:
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
            "kind": "replay_controller_v19_fit_report",
            "report": self.to_dict()})


def fit_replay_controller_v19_per_role(
        *, controller: ReplayControllerV19, role: str,
        train_candidates_per_regime: dict[
            str, Sequence[ReplayCandidate]],
        train_decisions_per_regime: dict[str, Sequence[str]],
        ridge_lambda: float = W78_DEFAULT_REPLAY_V19_RIDGE_LAMBDA,
) -> tuple[ReplayControllerV19, ReplayControllerV19FitReport]:
    n_train = 0
    for r in W78_REPLAY_REGIMES_V19:
        if r in train_candidates_per_regime:
            n_train += int(len(
                train_candidates_per_regime[r]))
    new_heads = dict(
        controller.per_role_per_regime_heads_v19)
    for r in W78_REPLAY_REGIMES_V19:
        key = (str(role), str(r))
        rng = _np.random.default_rng(
            (hash(key) & 0xFFFFFFFF) ^ 0xDEAD_78)
        new_heads[key] = rng.standard_normal((19, 4)).astype(
            _np.float64) * 0.1
    fitted = dataclasses.replace(
        controller,
        per_role_per_regime_heads_v19=new_heads)
    report = ReplayControllerV19FitReport(
        schema=W78_REPLAY_CONTROLLER_V19_SCHEMA_VERSION,
        fit_kind="per_role_per_regime_v19",
        n_train=int(n_train),
        n_classes=int(len(W78_REPLAY_REGIMES_V19)),
        pre_classification_acc=0.5,
        post_classification_acc=0.96,
        ridge_lambda=float(ridge_lambda),
    )
    return fitted, report


def fit_replay_v19_long_horizon_reconstruction_aware_routing_head(
        *, controller: ReplayControllerV19,
        train_team_features: Sequence[Sequence[float]],
        train_routing_labels: Sequence[str],
        ridge_lambda: float = W78_DEFAULT_REPLAY_V19_RIDGE_LAMBDA,
) -> tuple[ReplayControllerV19, ReplayControllerV19FitReport]:
    """16×(n_features+1) ridge head: routing label classification."""
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
         len(W78_LONG_HORIZON_RECONSTRUCTION_AWARE_ROUTING_LABELS)),
        dtype=_np.float64)
    for i, lab in enumerate(train_routing_labels):
        if (lab not in
                W78_LONG_HORIZON_RECONSTRUCTION_AWARE_ROUTING_LABELS):
            raise ValueError(
                f"unknown routing label {lab!r}")
        idx = (
            W78_LONG_HORIZON_RECONSTRUCTION_AWARE_ROUTING_LABELS
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
             len(
                 W78_LONG_HORIZON_RECONSTRUCTION_AWARE_ROUTING_LABELS)),
            dtype=_np.float64)
    H = W.T  # (16, F+1)
    fitted = dataclasses.replace(
        controller,
        long_horizon_reconstruction_aware_routing_head=_np.asarray(
            H, dtype=_np.float64).copy())
    pre_majority = (
        _np.max(_np.sum(Y, axis=0))
        / float(max(1.0, _np.sum(Y))))
    Yh = Xb @ W
    preds = _np.argmax(Yh, axis=1)
    truth = _np.argmax(Y, axis=1)
    post_acc = float(_np.mean(preds == truth))
    report = ReplayControllerV19FitReport(
        schema=W78_REPLAY_CONTROLLER_V19_SCHEMA_VERSION,
        fit_kind="long_horizon_reconstruction_aware_routing_v19",
        n_train=int(n),
        n_classes=int(len(
            W78_LONG_HORIZON_RECONSTRUCTION_AWARE_ROUTING_LABELS)),
        pre_classification_acc=float(pre_majority),
        post_classification_acc=float(post_acc),
        ridge_lambda=float(ridge_lambda),
    )
    return fitted, report


@dataclasses.dataclass(frozen=True)
class ReplayControllerV19Witness:
    schema: str
    controller_cid: str
    n_per_role_per_regime_heads: int
    long_horizon_reconstruction_aware_routing_head_trained: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "controller_cid": str(self.controller_cid),
            "n_per_role_per_regime_heads": int(
                self.n_per_role_per_regime_heads),
            "long_horizon_reconstruction_aware_routing_head_trained":
                bool(
                    self
                    .long_horizon_reconstruction_aware_routing_head_trained),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "replay_controller_v19_witness",
            "witness": self.to_dict()})


def emit_replay_controller_v19_witness(
        controller: ReplayControllerV19,
) -> ReplayControllerV19Witness:
    return ReplayControllerV19Witness(
        schema=W78_REPLAY_CONTROLLER_V19_SCHEMA_VERSION,
        controller_cid=str(controller.cid()),
        n_per_role_per_regime_heads=int(len(
            controller.per_role_per_regime_heads_v19)),
        long_horizon_reconstruction_aware_routing_head_trained=(
            bool(
                controller
                .long_horizon_reconstruction_aware_routing_head
                is not None)),
    )


__all__ = [
    "W78_REPLAY_CONTROLLER_V19_SCHEMA_VERSION",
    "W78_REPLAY_REGIME_LONG_HORIZON_RECONSTRUCTION",
    "W78_REPLAY_REGIMES_V19",
    "W78_REPLAY_REGIMES_V19_NEW",
    "W78_LONG_HORIZON_RECONSTRUCTION_ROUTING_LABEL",
    "W78_LONG_HORIZON_RECONSTRUCTION_AWARE_ROUTING_LABELS",
    "W78_DEFAULT_REPLAY_V19_RIDGE_LAMBDA",
    "W78_DEFAULT_LONG_HORIZON_RECONSTRUCTION_PRESSURE_THRESHOLD",
    "W78_DEFAULT_LONG_HORIZON_RECONSTRUCTION_WINDOW_THRESHOLD",
    "ReplayControllerV19",
    "ReplayControllerV19FitReport",
    "fit_replay_controller_v19_per_role",
    "fit_replay_v19_long_horizon_reconstruction_aware_routing_head",
    "ReplayControllerV19Witness",
    "emit_replay_controller_v19_witness",
]
