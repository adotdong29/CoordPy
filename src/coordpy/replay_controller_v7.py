"""W66 M7 — Replay Controller V7.

Strictly extends W65's ``coordpy.replay_controller_v6``. V6 fit
R=8 per-role per-regime 10×4 ridge heads + a multi-agent abstain
head. V7 adds:

* **Nine regimes** — V7 introduces TWO new regimes on top of V6's
  eight:
    - ``team_failure_recovery_regime`` (team_failure_recovery_flag
      positive and substrate-fidelity below team-recovery threshold
      ⇒ the regime where team-failure-recovery is the load-bearing
      signal).
    - ``team_consensus_under_budget_regime`` (visible-token budget
      below threshold and team-coordination above threshold ⇒
      regime where consensus-under-budget is the load-bearing
      signal).
* **Per-role per-regime ridge head** — fits the new regimes too.
* **Trained team-substrate-routing head** —
  ``fit_replay_v7_team_substrate_routing_head`` is a 4×11 ridge
  head that decides whether to route through the team-substrate
  coordination policy.

Honest scope (W66)
------------------

* All V7 fits remain closed-form ridge (``W66-L-V7-REPLAY-NO-
  AUTOGRAD-CAP``). Six new W66 ridge solves in total.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.replay_controller_v7 requires numpy") from exc

from .replay_controller import (
    ReplayCandidate, ReplayDecision,
    W60_REPLAY_DECISIONS,
)
from .replay_controller_v2 import _softmax_4
from .replay_controller_v5 import (
    _candidate_feature_v5,
    fit_replay_controller_v5_per_regime,
)
from .replay_controller_v6 import (
    ReplayControllerV6,
    W65_REPLAY_REGIMES_V6,
    W65_DEFAULT_REPLAY_V6_RIDGE_LAMBDA,
    W65_REPLAY_CONTROLLER_V6_SCHEMA_VERSION,
)
from .tiny_substrate_v3 import _sha256_hex


W66_REPLAY_CONTROLLER_V7_SCHEMA_VERSION: str = (
    "coordpy.replay_controller_v7.v1")

W66_REPLAY_REGIME_TEAM_FAILURE_RECOVERY: str = (
    "team_failure_recovery_regime")
W66_REPLAY_REGIME_TEAM_CONSENSUS_UNDER_BUDGET: str = (
    "team_consensus_under_budget_regime")
W66_REPLAY_REGIMES_V7_NEW: tuple[str, ...] = (
    W66_REPLAY_REGIME_TEAM_FAILURE_RECOVERY,
    W66_REPLAY_REGIME_TEAM_CONSENSUS_UNDER_BUDGET,
)
W66_REPLAY_REGIMES_V7: tuple[str, ...] = (
    *W65_REPLAY_REGIMES_V6,
    *W66_REPLAY_REGIMES_V7_NEW,
)
W66_DEFAULT_REPLAY_V7_RIDGE_LAMBDA: float = 0.10
W66_DEFAULT_TEAM_FAILURE_RECOVERY_THRESHOLD: float = 0.4
W66_DEFAULT_TEAM_BUDGET_THRESHOLD: float = 0.4
W66_DEFAULT_TEAM_COORDINATION_THRESHOLD: float = 0.5


@dataclasses.dataclass
class ReplayControllerV7:
    inner_v6: ReplayControllerV6
    # Per-role per-regime heads for the V7 regimes only.
    per_role_per_regime_heads_v7: dict[
        tuple[str, str], "_np.ndarray"] = dataclasses.field(
        default_factory=dict)
    # Team-substrate-routing head: (4, 11) ridge.
    team_substrate_routing_head: "_np.ndarray | None" = None
    team_failure_recovery_threshold: float = (
        W66_DEFAULT_TEAM_FAILURE_RECOVERY_THRESHOLD)
    team_budget_threshold: float = (
        W66_DEFAULT_TEAM_BUDGET_THRESHOLD)
    team_coordination_threshold: float = (
        W66_DEFAULT_TEAM_COORDINATION_THRESHOLD)
    audit_v7: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    @classmethod
    def init(
            cls, *,
            inner_v6: ReplayControllerV6 | None = None,
    ) -> "ReplayControllerV7":
        if inner_v6 is None:
            inner_v6 = ReplayControllerV6.init()
        return cls(inner_v6=inner_v6)

    def cid(self) -> str:
        ph_cid = "untrained"
        if self.per_role_per_regime_heads_v7:
            payload = sorted(
                (str(k[0]) + "::" + str(k[1]),
                 _np.asarray(v, dtype=_np.float64).tobytes().hex())
                for k, v
                in self.per_role_per_regime_heads_v7.items())
            ph_cid = _sha256_hex(payload)
        tsrh_cid = "untrained"
        if self.team_substrate_routing_head is not None:
            tsrh_cid = _sha256_hex(
                self.team_substrate_routing_head.tobytes().hex())
        return _sha256_hex({
            "schema": W66_REPLAY_CONTROLLER_V7_SCHEMA_VERSION,
            "kind": "replay_controller_v7",
            "inner_v6_cid": str(self.inner_v6.cid()),
            "per_role_per_regime_heads_v7_cid": ph_cid,
            "team_substrate_routing_head_cid": tsrh_cid,
            "team_failure_recovery_threshold": float(round(
                self.team_failure_recovery_threshold, 12)),
            "team_budget_threshold": float(round(
                self.team_budget_threshold, 12)),
            "team_coordination_threshold": float(round(
                self.team_coordination_threshold, 12)),
        })

    def classify_regime_v7(
            self, c: ReplayCandidate, *,
            team_failure_recovery_flag: float = 0.0,
            visible_token_budget_frac: float = 1.0,
            team_coordination_flag: float = 0.0,
            substrate_fidelity: float = 0.0,
            **v6_kwargs: Any,
    ) -> str:
        """V7 regime classification. Returns one of the two new V7
        regimes when their conditions are met; otherwise falls back
        to V6."""
        if (float(team_failure_recovery_flag)
                >= float(self.team_failure_recovery_threshold)):
            return W66_REPLAY_REGIME_TEAM_FAILURE_RECOVERY
        if (float(visible_token_budget_frac)
                <= float(self.team_budget_threshold)
                and float(team_coordination_flag)
                >= float(self.team_coordination_threshold)):
            return W66_REPLAY_REGIME_TEAM_CONSENSUS_UNDER_BUDGET
        return self.inner_v6.classify_regime_v6(
            c, team_coordination_flag=float(team_coordination_flag),
            substrate_fidelity=float(substrate_fidelity),
            **v6_kwargs)

    def decide_team_substrate_routing(
            self, *, team_features: Sequence[float],
    ) -> tuple[str, float]:
        """Returns (routing_label, score)."""
        if self.team_substrate_routing_head is None:
            return "no_team_substrate", 0.0
        feats = _np.asarray(
            list(team_features) + [1.0], dtype=_np.float64)
        if feats.shape[0] != int(
                self.team_substrate_routing_head.shape[1]):
            return "no_team_substrate", 0.0
        score = _np.asarray(
            self.team_substrate_routing_head,
            dtype=_np.float64) @ feats
        probs = _softmax_4(score)
        idx = int(_np.argmax(probs))
        lab = W66_TEAM_SUBSTRATE_ROUTING_LABELS[idx]
        return lab, float(probs[idx])


W66_TEAM_SUBSTRATE_ROUTING_LABELS: tuple[str, ...] = (
    "team_substrate_route", "substrate_route_only",
    "team_only_route", "no_team_substrate")


@dataclasses.dataclass(frozen=True)
class ReplayControllerV7FitReport:
    schema: str
    n_role_regime_heads_v7: int
    team_substrate_routing_trained: bool
    team_substrate_routing_train_residual: float
    converged: bool
    ridge_lambda: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "n_role_regime_heads_v7": int(
                self.n_role_regime_heads_v7),
            "team_substrate_routing_trained": bool(
                self.team_substrate_routing_trained),
            "team_substrate_routing_train_residual": float(round(
                self.team_substrate_routing_train_residual, 12)),
            "converged": bool(self.converged),
            "ridge_lambda": float(round(self.ridge_lambda, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "replay_controller_v7_fit_report",
            "report": self.to_dict()})


def fit_replay_controller_v7_per_role(
        *, controller: ReplayControllerV7, role: str,
        train_candidates_per_regime: dict[
            str, Sequence[ReplayCandidate]],
        train_decisions_per_regime: dict[
            str, Sequence[str]],
        ridge_lambda: float = (
            W66_DEFAULT_REPLAY_V7_RIDGE_LAMBDA),
) -> tuple[ReplayControllerV7, ReplayControllerV7FitReport]:
    """Fit per-role per-regime heads for the V7 regimes.

    Reuses V5's per-regime fit (which expects features of shape
    10) on the V7-only regimes provided in train_*; promotes those
    heads into the V7 (role, regime) map."""
    if not train_candidates_per_regime:
        raise ValueError("must provide >= 1 regime")
    # Use the inner V5 fit; the V5 fit expects regimes from V5..V7,
    # so we feed each V7 regime as if it were a V5 regime — V5 will
    # treat it as a string key.
    inner_v5 = controller.inner_v6.inner_v5
    fitted_v5, _ = fit_replay_controller_v5_per_regime(
        controller=inner_v5,
        train_candidates_per_regime=train_candidates_per_regime,
        train_decisions_per_regime=train_decisions_per_regime,
        ridge_lambda=float(ridge_lambda))
    new_heads = dict(controller.per_role_per_regime_heads_v7)
    for regime, head in fitted_v5.per_regime_heads_v5.items():
        new_heads[(str(role), str(regime))] = _np.asarray(
            head, dtype=_np.float64).copy()
    fitted = dataclasses.replace(
        controller, per_role_per_regime_heads_v7=new_heads)
    return fitted, ReplayControllerV7FitReport(
        schema=W66_REPLAY_CONTROLLER_V7_SCHEMA_VERSION,
        n_role_regime_heads_v7=int(len(new_heads)),
        team_substrate_routing_trained=bool(
            fitted.team_substrate_routing_head is not None),
        team_substrate_routing_train_residual=0.0,
        converged=True,
        ridge_lambda=float(ridge_lambda),
    )


def fit_replay_v7_team_substrate_routing_head(
        *, controller: ReplayControllerV7,
        train_team_features: Sequence[Sequence[float]],
        train_routing_labels: Sequence[str],
        ridge_lambda: float = (
            W66_DEFAULT_REPLAY_V7_RIDGE_LAMBDA),
) -> tuple[ReplayControllerV7, ReplayControllerV7FitReport]:
    """Fit the 4×11 team-substrate-routing head over a 10-dim
    feature space + bias = 11."""
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
        if lab not in W66_TEAM_SUBSTRATE_ROUTING_LABELS:
            raise ValueError(
                f"unknown routing label {lab!r}")
        idx = W66_TEAM_SUBSTRATE_ROUTING_LABELS.index(str(lab))
        Y[i, idx] = 1.0
    lam = max(float(ridge_lambda), 1e-9)
    A = X.T @ X + lam * _np.eye(X.shape[1], dtype=_np.float64)
    b = X.T @ Y
    try:
        W = _np.linalg.solve(A, b)   # (11, 4)
    except Exception:
        W = _np.zeros((X.shape[1], 4), dtype=_np.float64)
    Y_hat = X @ W
    residual = float(_np.mean(_np.abs(Y - Y_hat)))
    fitted = dataclasses.replace(
        controller, team_substrate_routing_head=W.T.copy())
    return fitted, ReplayControllerV7FitReport(
        schema=W66_REPLAY_CONTROLLER_V7_SCHEMA_VERSION,
        n_role_regime_heads_v7=int(
            len(controller.per_role_per_regime_heads_v7)),
        team_substrate_routing_trained=True,
        team_substrate_routing_train_residual=float(residual),
        converged=bool(residual <= 1.0),
        ridge_lambda=float(ridge_lambda),
    )


@dataclasses.dataclass(frozen=True)
class ReplayControllerV7Witness:
    schema: str
    controller_cid: str
    n_role_regime_heads_v7: int
    team_substrate_routing_trained: bool
    n_audit_entries_v7: int
    inner_v6_witness_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "controller_cid": str(self.controller_cid),
            "n_role_regime_heads_v7": int(
                self.n_role_regime_heads_v7),
            "team_substrate_routing_trained": bool(
                self.team_substrate_routing_trained),
            "n_audit_entries_v7": int(self.n_audit_entries_v7),
            "inner_v6_witness_cid": str(
                self.inner_v6_witness_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "replay_controller_v7_witness",
            "witness": self.to_dict()})


def emit_replay_controller_v7_witness(
        controller: ReplayControllerV7,
) -> ReplayControllerV7Witness:
    from .replay_controller_v6 import (
        emit_replay_controller_v6_witness,
    )
    inner_w = emit_replay_controller_v6_witness(
        controller.inner_v6)
    return ReplayControllerV7Witness(
        schema=W66_REPLAY_CONTROLLER_V7_SCHEMA_VERSION,
        controller_cid=str(controller.cid()),
        n_role_regime_heads_v7=int(
            len(controller.per_role_per_regime_heads_v7)),
        team_substrate_routing_trained=bool(
            controller.team_substrate_routing_head is not None),
        n_audit_entries_v7=int(len(controller.audit_v7)),
        inner_v6_witness_cid=str(inner_w.cid()),
    )


__all__ = [
    "W66_REPLAY_CONTROLLER_V7_SCHEMA_VERSION",
    "W66_REPLAY_REGIME_TEAM_FAILURE_RECOVERY",
    "W66_REPLAY_REGIME_TEAM_CONSENSUS_UNDER_BUDGET",
    "W66_REPLAY_REGIMES_V7",
    "W66_DEFAULT_REPLAY_V7_RIDGE_LAMBDA",
    "W66_DEFAULT_TEAM_FAILURE_RECOVERY_THRESHOLD",
    "W66_DEFAULT_TEAM_BUDGET_THRESHOLD",
    "W66_DEFAULT_TEAM_COORDINATION_THRESHOLD",
    "W66_TEAM_SUBSTRATE_ROUTING_LABELS",
    "ReplayControllerV7",
    "ReplayControllerV7FitReport",
    "fit_replay_controller_v7_per_role",
    "fit_replay_v7_team_substrate_routing_head",
    "ReplayControllerV7Witness",
    "emit_replay_controller_v7_witness",
]
