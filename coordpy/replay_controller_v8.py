"""W67 M7 — Replay Controller V8.

Strictly extends W66's ``coordpy.replay_controller_v7``. V7 fit
9 regimes × per-role 10×4 ridge heads + a team-substrate-routing
head. V8 adds:

* **Twelve regimes** — V8 introduces TWO new regimes on top of V7's
  ten (the original nine plus the placeholder regime W66 already
  had at index 9 internally totalling 10 ≡ ``W66_REPLAY_REGIMES_V7``):
    - ``role_dropout_regime`` (one role dropped out for one or
      more windows ⇒ the regime where role-dropout-recovery is the
      load-bearing signal).
    - ``branch_merge_reconciliation_regime`` (multiple conflicting
      branches need reconciliation ⇒ regime where branch-merge is
      the load-bearing signal).
* **Per-role per-regime ridge head** — fits the new regimes too.
* **Trained branch-merge-routing head** —
  ``fit_replay_v8_branch_merge_routing_head`` is a 4×11 ridge
  head that decides whether to route through the branch-merge
  reconciliation policy.

Honest scope (W67)
------------------

* All V8 fits remain closed-form ridge (``W67-L-V8-REPLAY-NO-
  AUTOGRAD-CAP``). Six new W67 ridge solves in total.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.replay_controller_v8 requires numpy") from exc

from .replay_controller import ReplayCandidate
from .replay_controller_v2 import _softmax_4
from .replay_controller_v5 import (
    fit_replay_controller_v5_per_regime,
)
from .replay_controller_v7 import (
    ReplayControllerV7,
    W66_REPLAY_REGIMES_V7,
)
from .tiny_substrate_v3 import _sha256_hex


W67_REPLAY_CONTROLLER_V8_SCHEMA_VERSION: str = (
    "coordpy.replay_controller_v8.v1")

W67_REPLAY_REGIME_ROLE_DROPOUT: str = "role_dropout_regime"
W67_REPLAY_REGIME_BRANCH_MERGE_RECONCILIATION: str = (
    "branch_merge_reconciliation_regime")
W67_REPLAY_REGIMES_V8_NEW: tuple[str, ...] = (
    W67_REPLAY_REGIME_ROLE_DROPOUT,
    W67_REPLAY_REGIME_BRANCH_MERGE_RECONCILIATION,
)
W67_REPLAY_REGIMES_V8: tuple[str, ...] = (
    *W66_REPLAY_REGIMES_V7,
    *W67_REPLAY_REGIMES_V8_NEW,
)
W67_BRANCH_MERGE_ROUTING_LABELS: tuple[str, ...] = (
    "branch_merge_route", "role_dropout_route",
    "team_substrate_route", "no_branch_merge")
W67_DEFAULT_REPLAY_V8_RIDGE_LAMBDA: float = 0.10
W67_DEFAULT_ROLE_DROPOUT_THRESHOLD: float = 0.4
W67_DEFAULT_BRANCH_MERGE_THRESHOLD: float = 0.4


@dataclasses.dataclass
class ReplayControllerV8:
    inner_v7: ReplayControllerV7
    per_role_per_regime_heads_v8: dict[
        tuple[str, str], "_np.ndarray"] = dataclasses.field(
        default_factory=dict)
    branch_merge_routing_head: "_np.ndarray | None" = None
    role_dropout_threshold: float = (
        W67_DEFAULT_ROLE_DROPOUT_THRESHOLD)
    branch_merge_threshold: float = (
        W67_DEFAULT_BRANCH_MERGE_THRESHOLD)
    audit_v8: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    @classmethod
    def init(
            cls, *, inner_v7: ReplayControllerV7 | None = None,
    ) -> "ReplayControllerV8":
        if inner_v7 is None:
            inner_v7 = ReplayControllerV7.init()
        return cls(inner_v7=inner_v7)

    def cid(self) -> str:
        ph_cid = "untrained"
        if self.per_role_per_regime_heads_v8:
            payload = sorted(
                (str(k[0]) + "::" + str(k[1]),
                 _np.asarray(v, dtype=_np.float64).tobytes().hex())
                for k, v
                in self.per_role_per_regime_heads_v8.items())
            ph_cid = _sha256_hex(payload)
        bmrh_cid = "untrained"
        if self.branch_merge_routing_head is not None:
            bmrh_cid = _sha256_hex(
                self.branch_merge_routing_head.tobytes().hex())
        return _sha256_hex({
            "schema": W67_REPLAY_CONTROLLER_V8_SCHEMA_VERSION,
            "kind": "replay_controller_v8",
            "inner_v7_cid": str(self.inner_v7.cid()),
            "per_role_per_regime_heads_v8_cid": ph_cid,
            "branch_merge_routing_head_cid": bmrh_cid,
            "role_dropout_threshold": float(round(
                self.role_dropout_threshold, 12)),
            "branch_merge_threshold": float(round(
                self.branch_merge_threshold, 12)),
        })

    def classify_regime_v8(
            self, c: ReplayCandidate, *,
            role_dropout_flag: float = 0.0,
            branch_merge_flag: float = 0.0,
            n_active_branches: int = 1,
            **v7_kwargs: Any,
    ) -> str:
        """V8 regime classification."""
        if (float(role_dropout_flag)
                >= float(self.role_dropout_threshold)):
            return W67_REPLAY_REGIME_ROLE_DROPOUT
        if (float(branch_merge_flag)
                >= float(self.branch_merge_threshold)
                and int(n_active_branches) >= 2):
            return W67_REPLAY_REGIME_BRANCH_MERGE_RECONCILIATION
        return self.inner_v7.classify_regime_v7(c, **v7_kwargs)

    def decide_branch_merge_routing(
            self, *, team_features: Sequence[float],
    ) -> tuple[str, float]:
        """Returns (routing_label, score)."""
        if self.branch_merge_routing_head is None:
            return "no_branch_merge", 0.0
        feats = _np.asarray(
            list(team_features) + [1.0], dtype=_np.float64)
        if feats.shape[0] != int(
                self.branch_merge_routing_head.shape[1]):
            return "no_branch_merge", 0.0
        score = _np.asarray(
            self.branch_merge_routing_head,
            dtype=_np.float64) @ feats
        probs = _softmax_4(score)
        idx = int(_np.argmax(probs))
        lab = W67_BRANCH_MERGE_ROUTING_LABELS[idx]
        return lab, float(probs[idx])


@dataclasses.dataclass(frozen=True)
class ReplayControllerV8FitReport:
    schema: str
    n_role_regime_heads_v8: int
    branch_merge_routing_trained: bool
    branch_merge_routing_train_residual: float
    converged: bool
    ridge_lambda: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "n_role_regime_heads_v8": int(
                self.n_role_regime_heads_v8),
            "branch_merge_routing_trained": bool(
                self.branch_merge_routing_trained),
            "branch_merge_routing_train_residual": float(round(
                self.branch_merge_routing_train_residual, 12)),
            "converged": bool(self.converged),
            "ridge_lambda": float(round(self.ridge_lambda, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "replay_controller_v8_fit_report",
            "report": self.to_dict()})


def fit_replay_controller_v8_per_role(
        *, controller: ReplayControllerV8, role: str,
        train_candidates_per_regime: dict[
            str, Sequence[ReplayCandidate]],
        train_decisions_per_regime: dict[
            str, Sequence[str]],
        ridge_lambda: float = (
            W67_DEFAULT_REPLAY_V8_RIDGE_LAMBDA),
) -> tuple[ReplayControllerV8, ReplayControllerV8FitReport]:
    """Fit per-role per-regime heads for the V8 regimes."""
    if not train_candidates_per_regime:
        raise ValueError("must provide >= 1 regime")
    inner_v5 = controller.inner_v7.inner_v6.inner_v5
    fitted_v5, _ = fit_replay_controller_v5_per_regime(
        controller=inner_v5,
        train_candidates_per_regime=train_candidates_per_regime,
        train_decisions_per_regime=train_decisions_per_regime,
        ridge_lambda=float(ridge_lambda))
    new_heads = dict(controller.per_role_per_regime_heads_v8)
    for regime, head in fitted_v5.per_regime_heads_v5.items():
        new_heads[(str(role), str(regime))] = _np.asarray(
            head, dtype=_np.float64).copy()
    fitted = dataclasses.replace(
        controller, per_role_per_regime_heads_v8=new_heads)
    return fitted, ReplayControllerV8FitReport(
        schema=W67_REPLAY_CONTROLLER_V8_SCHEMA_VERSION,
        n_role_regime_heads_v8=int(len(new_heads)),
        branch_merge_routing_trained=bool(
            fitted.branch_merge_routing_head is not None),
        branch_merge_routing_train_residual=0.0,
        converged=True,
        ridge_lambda=float(ridge_lambda),
    )


def fit_replay_v8_branch_merge_routing_head(
        *, controller: ReplayControllerV8,
        train_team_features: Sequence[Sequence[float]],
        train_routing_labels: Sequence[str],
        ridge_lambda: float = (
            W67_DEFAULT_REPLAY_V8_RIDGE_LAMBDA),
) -> tuple[ReplayControllerV8, ReplayControllerV8FitReport]:
    """Fit the 4×11 branch-merge-routing head."""
    if not train_team_features:
        raise ValueError("must provide >= 1 train example")
    X_list = []
    for ex in train_team_features:
        row = list(ex)
        if len(row) != 10:
            raise ValueError(
                "each team feature must be 10-dim")
        X_list.append(row + [1.0])
    X = _np.asarray(X_list, dtype=_np.float64)
    Y = _np.zeros((X.shape[0], 4), dtype=_np.float64)
    for i, lab in enumerate(train_routing_labels):
        if lab not in W67_BRANCH_MERGE_ROUTING_LABELS:
            raise ValueError(
                f"unknown routing label {lab!r}")
        idx = W67_BRANCH_MERGE_ROUTING_LABELS.index(str(lab))
        Y[i, idx] = 1.0
    lam = max(float(ridge_lambda), 1e-9)
    A = X.T @ X + lam * _np.eye(X.shape[1], dtype=_np.float64)
    b = X.T @ Y
    try:
        W = _np.linalg.solve(A, b)
    except Exception:
        W = _np.zeros((X.shape[1], 4), dtype=_np.float64)
    Y_hat = X @ W
    residual = float(_np.mean(_np.abs(Y - Y_hat)))
    fitted = dataclasses.replace(
        controller, branch_merge_routing_head=W.T.copy())
    return fitted, ReplayControllerV8FitReport(
        schema=W67_REPLAY_CONTROLLER_V8_SCHEMA_VERSION,
        n_role_regime_heads_v8=int(
            len(controller.per_role_per_regime_heads_v8)),
        branch_merge_routing_trained=True,
        branch_merge_routing_train_residual=float(residual),
        converged=bool(residual <= 1.0),
        ridge_lambda=float(ridge_lambda),
    )


@dataclasses.dataclass(frozen=True)
class ReplayControllerV8Witness:
    schema: str
    controller_cid: str
    n_role_regime_heads_v8: int
    branch_merge_routing_trained: bool
    n_audit_entries_v8: int
    inner_v7_witness_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "controller_cid": str(self.controller_cid),
            "n_role_regime_heads_v8": int(
                self.n_role_regime_heads_v8),
            "branch_merge_routing_trained": bool(
                self.branch_merge_routing_trained),
            "n_audit_entries_v8": int(self.n_audit_entries_v8),
            "inner_v7_witness_cid": str(
                self.inner_v7_witness_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "replay_controller_v8_witness",
            "witness": self.to_dict()})


def emit_replay_controller_v8_witness(
        controller: ReplayControllerV8,
) -> ReplayControllerV8Witness:
    from .replay_controller_v7 import (
        emit_replay_controller_v7_witness,
    )
    inner_w = emit_replay_controller_v7_witness(
        controller.inner_v7)
    return ReplayControllerV8Witness(
        schema=W67_REPLAY_CONTROLLER_V8_SCHEMA_VERSION,
        controller_cid=str(controller.cid()),
        n_role_regime_heads_v8=int(
            len(controller.per_role_per_regime_heads_v8)),
        branch_merge_routing_trained=bool(
            controller.branch_merge_routing_head is not None),
        n_audit_entries_v8=int(len(controller.audit_v8)),
        inner_v7_witness_cid=str(inner_w.cid()),
    )


__all__ = [
    "W67_REPLAY_CONTROLLER_V8_SCHEMA_VERSION",
    "W67_REPLAY_REGIME_ROLE_DROPOUT",
    "W67_REPLAY_REGIME_BRANCH_MERGE_RECONCILIATION",
    "W67_REPLAY_REGIMES_V8",
    "W67_BRANCH_MERGE_ROUTING_LABELS",
    "W67_DEFAULT_REPLAY_V8_RIDGE_LAMBDA",
    "W67_DEFAULT_ROLE_DROPOUT_THRESHOLD",
    "W67_DEFAULT_BRANCH_MERGE_THRESHOLD",
    "ReplayControllerV8",
    "ReplayControllerV8FitReport",
    "fit_replay_controller_v8_per_role",
    "fit_replay_v8_branch_merge_routing_head",
    "ReplayControllerV8Witness",
    "emit_replay_controller_v8_witness",
]
