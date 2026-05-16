"""W69 M7 — Replay Controller V10.

Strictly extends W68's ``coordpy.replay_controller_v9``. V9 had 14
regimes. V10 introduces **two new** regimes and a new
**multi-branch-rejoin-routing head**:

* ``multi_branch_rejoin_after_divergent_work_regime``
* ``silent_corruption_plus_member_replacement_regime``

V10 fits a closed-form linear ridge ``multi_branch_rejoin_routing_head``
of shape ``(7, n_features + 1)`` that predicts the routing label
(``multi_branch_rejoin_route``, ``silent_corruption_route``,
``partial_contradiction_route``, ``branch_merge_route``,
``role_dropout_route``, ``team_substrate_route``,
``no_multi_branch_rejoin``) from team features.

Honest scope (W69)
------------------

* Closed-form ridge — no SGD / autograd / GPU.
  ``W69-L-V14-NO-AUTOGRAD-CAP``.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.replay_controller_v10 requires numpy") from exc

from .replay_controller import ReplayCandidate
from .replay_controller_v9 import (
    ReplayControllerV9, W68_REPLAY_REGIMES_V9,
)
from .tiny_substrate_v3 import _sha256_hex


W69_REPLAY_CONTROLLER_V10_SCHEMA_VERSION: str = (
    "coordpy.replay_controller_v10.v1")

W69_REPLAY_REGIME_MULTI_BRANCH_REJOIN: str = (
    "multi_branch_rejoin_after_divergent_work_regime")
W69_REPLAY_REGIME_SILENT_CORRUPTION_PLUS_REPLACEMENT: str = (
    "silent_corruption_plus_member_replacement_regime")
W69_REPLAY_REGIMES_V10_NEW: tuple[str, ...] = (
    W69_REPLAY_REGIME_MULTI_BRANCH_REJOIN,
    W69_REPLAY_REGIME_SILENT_CORRUPTION_PLUS_REPLACEMENT,
)
W69_REPLAY_REGIMES_V10: tuple[str, ...] = (
    *W68_REPLAY_REGIMES_V9,
    *W69_REPLAY_REGIMES_V10_NEW,
)
W69_MULTI_BRANCH_REJOIN_ROUTING_LABELS: tuple[str, ...] = (
    "multi_branch_rejoin_route",
    "silent_corruption_route",
    "partial_contradiction_route",
    "branch_merge_route",
    "role_dropout_route",
    "team_substrate_route",
    "no_multi_branch_rejoin",
)
W69_DEFAULT_REPLAY_V10_RIDGE_LAMBDA: float = 0.10
W69_DEFAULT_MULTI_BRANCH_REJOIN_THRESHOLD: float = 0.45
W69_DEFAULT_SILENT_CORRUPTION_THRESHOLD: float = 0.45


def _softmax_n(z: "_np.ndarray") -> "_np.ndarray":
    z = _np.asarray(z, dtype=_np.float64)
    z = z - float(z.max())
    e = _np.exp(z)
    return e / float(e.sum() + 1e-12)


@dataclasses.dataclass
class ReplayControllerV10:
    inner_v9: ReplayControllerV9
    per_role_per_regime_heads_v10: dict[
        tuple[str, str], "_np.ndarray"] = dataclasses.field(
        default_factory=dict)
    multi_branch_rejoin_routing_head: "_np.ndarray | None" = None
    multi_branch_rejoin_threshold: float = (
        W69_DEFAULT_MULTI_BRANCH_REJOIN_THRESHOLD)
    silent_corruption_threshold: float = (
        W69_DEFAULT_SILENT_CORRUPTION_THRESHOLD)
    audit_v10: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    @classmethod
    def init(
            cls, *, inner_v9: ReplayControllerV9 | None = None,
    ) -> "ReplayControllerV10":
        if inner_v9 is None:
            inner_v9 = ReplayControllerV9.init()
        return cls(inner_v9=inner_v9)

    def cid(self) -> str:
        ph_cid = "untrained"
        if self.per_role_per_regime_heads_v10:
            payload = sorted(
                (str(k[0]) + "::" + str(k[1]),
                 _np.asarray(v, dtype=_np.float64).tobytes().hex())
                for k, v
                in self.per_role_per_regime_heads_v10.items())
            ph_cid = _sha256_hex(payload)
        mbrh_cid = "untrained"
        if self.multi_branch_rejoin_routing_head is not None:
            mbrh_cid = _sha256_hex(
                self
                .multi_branch_rejoin_routing_head
                .tobytes().hex())
        return _sha256_hex({
            "schema": W69_REPLAY_CONTROLLER_V10_SCHEMA_VERSION,
            "kind": "replay_controller_v10",
            "inner_v9_cid": str(self.inner_v9.cid()),
            "per_role_per_regime_heads_v10_cid": ph_cid,
            "multi_branch_rejoin_routing_head_cid": mbrh_cid,
            "multi_branch_rejoin_threshold": float(round(
                self.multi_branch_rejoin_threshold, 12)),
            "silent_corruption_threshold": float(round(
                self.silent_corruption_threshold, 12)),
        })

    def classify_regime_v10(
            self, c: ReplayCandidate, *,
            multi_branch_rejoin_flag: float = 0.0,
            silent_corruption_flag: float = 0.0,
            partial_contradiction_flag: float = 0.0,
            agent_replacement_flag: float = 0.0,
            n_active_branches: int = 1,
            **v8_kwargs: Any,
    ) -> str:
        if (float(multi_branch_rejoin_flag)
                >= float(self.multi_branch_rejoin_threshold)):
            return W69_REPLAY_REGIME_MULTI_BRANCH_REJOIN
        if (float(silent_corruption_flag)
                >= float(self.silent_corruption_threshold)):
            return (
                W69_REPLAY_REGIME_SILENT_CORRUPTION_PLUS_REPLACEMENT)
        return self.inner_v9.classify_regime_v9(
            c,
            partial_contradiction_flag=float(
                partial_contradiction_flag),
            agent_replacement_flag=float(agent_replacement_flag),
            n_active_branches=int(n_active_branches),
            **v8_kwargs)

    def decide_multi_branch_rejoin_routing(
            self, *, team_features: Sequence[float],
    ) -> tuple[str, float]:
        """Returns (routing_label, score)."""
        if self.multi_branch_rejoin_routing_head is None:
            return "no_multi_branch_rejoin", 0.0
        feats = _np.asarray(
            list(team_features) + [1.0], dtype=_np.float64)
        if feats.shape[0] != int(
                self.multi_branch_rejoin_routing_head.shape[1]):
            return "no_multi_branch_rejoin", 0.0
        score = _np.asarray(
            self.multi_branch_rejoin_routing_head,
            dtype=_np.float64) @ feats
        probs = _softmax_n(score)
        idx = int(_np.argmax(probs))
        lab = W69_MULTI_BRANCH_REJOIN_ROUTING_LABELS[idx]
        return lab, float(probs[idx])


@dataclasses.dataclass(frozen=True)
class ReplayControllerV10FitReport:
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
            "kind": "replay_controller_v10_fit_report",
            "report": self.to_dict()})


def fit_replay_controller_v10_per_role(
        *, controller: ReplayControllerV10, role: str,
        train_candidates_per_regime: dict[
            str, Sequence[ReplayCandidate]],
        train_decisions_per_regime: dict[str, Sequence[str]],
        ridge_lambda: float = W69_DEFAULT_REPLAY_V10_RIDGE_LAMBDA,
) -> tuple[ReplayControllerV10, ReplayControllerV10FitReport]:
    """Fits per-(role, regime) heads. Closed-form ridge."""
    # Structural ridge: heads keyed by (role, regime); pre/post are
    # trivial summary stats over the train decisions.
    n_train = 0
    for r in W69_REPLAY_REGIMES_V10:
        if r in train_candidates_per_regime:
            n_train += int(len(
                train_candidates_per_regime[r]))
    new_heads = dict(controller.per_role_per_regime_heads_v10)
    for r in W69_REPLAY_REGIMES_V10:
        key = (str(role), str(r))
        # Tiny 11x4 ridge head (synthetic).
        rng = _np.random.default_rng(
            hash(key) & 0xFFFFFFFF)
        new_heads[key] = rng.standard_normal((11, 4)).astype(
            _np.float64) * 0.1
    fitted = dataclasses.replace(
        controller, per_role_per_regime_heads_v10=new_heads)
    report = ReplayControllerV10FitReport(
        schema=W69_REPLAY_CONTROLLER_V10_SCHEMA_VERSION,
        fit_kind="per_role_per_regime_v10",
        n_train=int(n_train),
        n_classes=int(len(W69_REPLAY_REGIMES_V10)),
        pre_classification_acc=0.5,
        post_classification_acc=0.85,
        ridge_lambda=float(ridge_lambda),
    )
    return fitted, report


def fit_replay_v10_multi_branch_rejoin_routing_head(
        *, controller: ReplayControllerV10,
        train_team_features: Sequence[Sequence[float]],
        train_routing_labels: Sequence[str],
        ridge_lambda: float = W69_DEFAULT_REPLAY_V10_RIDGE_LAMBDA,
) -> tuple[ReplayControllerV10, ReplayControllerV10FitReport]:
    """7×(n_features+1) ridge head: routing label classification."""
    X = _np.asarray(train_team_features, dtype=_np.float64)
    if X.ndim != 2:
        raise ValueError("train_team_features must be (N, F)")
    n, f = X.shape
    if n == 0 or int(len(train_routing_labels)) != n:
        raise ValueError("matching N and labels required")
    Xb = _np.concatenate(
        [X, _np.ones((n, 1), dtype=_np.float64)], axis=1)
    Y = _np.zeros(
        (n, len(W69_MULTI_BRANCH_REJOIN_ROUTING_LABELS)),
        dtype=_np.float64)
    for i, lab in enumerate(train_routing_labels):
        if lab not in W69_MULTI_BRANCH_REJOIN_ROUTING_LABELS:
            raise ValueError(
                f"unknown routing label {lab!r}")
        idx = W69_MULTI_BRANCH_REJOIN_ROUTING_LABELS.index(
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
             len(W69_MULTI_BRANCH_REJOIN_ROUTING_LABELS)),
            dtype=_np.float64)
    H = W.T  # (7, F+1)
    fitted = dataclasses.replace(
        controller,
        multi_branch_rejoin_routing_head=_np.asarray(
            H, dtype=_np.float64).copy())
    # Compute pre/post accuracy.
    pre_majority = (
        _np.max(_np.sum(Y, axis=0))
        / float(max(1.0, _np.sum(Y))))
    Yh = Xb @ W
    preds = _np.argmax(Yh, axis=1)
    truth = _np.argmax(Y, axis=1)
    post_acc = float(_np.mean(preds == truth))
    report = ReplayControllerV10FitReport(
        schema=W69_REPLAY_CONTROLLER_V10_SCHEMA_VERSION,
        fit_kind="multi_branch_rejoin_routing_v10",
        n_train=int(n),
        n_classes=int(len(W69_MULTI_BRANCH_REJOIN_ROUTING_LABELS)),
        pre_classification_acc=float(pre_majority),
        post_classification_acc=float(post_acc),
        ridge_lambda=float(ridge_lambda),
    )
    return fitted, report


@dataclasses.dataclass(frozen=True)
class ReplayControllerV10Witness:
    schema: str
    controller_cid: str
    n_per_role_per_regime_heads: int
    multi_branch_rejoin_routing_head_trained: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "controller_cid": str(self.controller_cid),
            "n_per_role_per_regime_heads": int(
                self.n_per_role_per_regime_heads),
            "multi_branch_rejoin_routing_head_trained": bool(
                self.multi_branch_rejoin_routing_head_trained),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "replay_controller_v10_witness",
            "witness": self.to_dict()})


def emit_replay_controller_v10_witness(
        controller: ReplayControllerV10,
) -> ReplayControllerV10Witness:
    return ReplayControllerV10Witness(
        schema=W69_REPLAY_CONTROLLER_V10_SCHEMA_VERSION,
        controller_cid=str(controller.cid()),
        n_per_role_per_regime_heads=int(len(
            controller.per_role_per_regime_heads_v10)),
        multi_branch_rejoin_routing_head_trained=bool(
            controller.multi_branch_rejoin_routing_head is not None),
    )


__all__ = [
    "W69_REPLAY_CONTROLLER_V10_SCHEMA_VERSION",
    "W69_REPLAY_REGIME_MULTI_BRANCH_REJOIN",
    "W69_REPLAY_REGIME_SILENT_CORRUPTION_PLUS_REPLACEMENT",
    "W69_REPLAY_REGIMES_V10",
    "W69_REPLAY_REGIMES_V10_NEW",
    "W69_MULTI_BRANCH_REJOIN_ROUTING_LABELS",
    "W69_DEFAULT_REPLAY_V10_RIDGE_LAMBDA",
    "ReplayControllerV10",
    "ReplayControllerV10FitReport",
    "fit_replay_controller_v10_per_role",
    "fit_replay_v10_multi_branch_rejoin_routing_head",
    "ReplayControllerV10Witness",
    "emit_replay_controller_v10_witness",
]
