"""W68 M7 — Replay Controller V9.

Strictly extends W67's ``coordpy.replay_controller_v8``. V8 fit 12
regimes + branch-merge-routing head. V9 adds:

* **Fourteen regimes** — V9 introduces TWO new regimes on top of
  V8's twelve:
    - ``partial_contradiction_under_delayed_reconciliation_regime``
      (multiple agents produce conflicting payloads with delayed
      arbitration ⇒ regime where partial-contradiction is the load-
      bearing signal).
    - ``agent_replacement_warm_restart_regime`` (an agent is
      replaced mid-run and warm-restarted from a substrate snapshot
      ⇒ regime where agent-replacement is the load-bearing signal).
* **Per-role per-regime ridge head** — fits the new regimes too.
* **Trained agent-replacement-routing head** —
  ``fit_replay_v9_agent_replacement_routing_head`` is a 6×11 ridge
  head over the new routing label set.

Honest scope (W68)
------------------

* All V9 fits remain closed-form ridge (``W68-L-V9-REPLAY-NO-
  AUTOGRAD-CAP``). Six new W68 ridge solves in total.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.replay_controller_v9 requires numpy") from exc

from .replay_controller import ReplayCandidate
from .replay_controller_v5 import (
    fit_replay_controller_v5_per_regime,
)
from .replay_controller_v2 import _softmax_4
from .replay_controller_v8 import (
    ReplayControllerV8,
    W67_REPLAY_REGIMES_V8,
)
from .tiny_substrate_v3 import _sha256_hex


W68_REPLAY_CONTROLLER_V9_SCHEMA_VERSION: str = (
    "coordpy.replay_controller_v9.v1")

W68_REPLAY_REGIME_PARTIAL_CONTRADICTION: str = (
    "partial_contradiction_under_delayed_reconciliation_regime")
W68_REPLAY_REGIME_AGENT_REPLACEMENT: str = (
    "agent_replacement_warm_restart_regime")
W68_REPLAY_REGIMES_V9_NEW: tuple[str, ...] = (
    W68_REPLAY_REGIME_PARTIAL_CONTRADICTION,
    W68_REPLAY_REGIME_AGENT_REPLACEMENT,
)
W68_REPLAY_REGIMES_V9: tuple[str, ...] = (
    *W67_REPLAY_REGIMES_V8,
    *W68_REPLAY_REGIMES_V9_NEW,
)
W68_AGENT_REPLACEMENT_ROUTING_LABELS: tuple[str, ...] = (
    "agent_replacement_route",
    "partial_contradiction_route",
    "branch_merge_route",
    "role_dropout_route",
    "team_substrate_route",
    "no_agent_replacement",
)
W68_DEFAULT_REPLAY_V9_RIDGE_LAMBDA: float = 0.10
W68_DEFAULT_PARTIAL_CONTRADICTION_THRESHOLD: float = 0.45
W68_DEFAULT_AGENT_REPLACEMENT_THRESHOLD: float = 0.45


def _softmax_n(z: "_np.ndarray") -> "_np.ndarray":
    z = _np.asarray(z, dtype=_np.float64)
    z = z - float(z.max())
    e = _np.exp(z)
    return e / float(e.sum() + 1e-12)


@dataclasses.dataclass
class ReplayControllerV9:
    inner_v8: ReplayControllerV8
    per_role_per_regime_heads_v9: dict[
        tuple[str, str], "_np.ndarray"] = dataclasses.field(
        default_factory=dict)
    agent_replacement_routing_head: "_np.ndarray | None" = None
    partial_contradiction_threshold: float = (
        W68_DEFAULT_PARTIAL_CONTRADICTION_THRESHOLD)
    agent_replacement_threshold: float = (
        W68_DEFAULT_AGENT_REPLACEMENT_THRESHOLD)
    audit_v9: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    @classmethod
    def init(
            cls, *, inner_v8: ReplayControllerV8 | None = None,
    ) -> "ReplayControllerV9":
        if inner_v8 is None:
            inner_v8 = ReplayControllerV8.init()
        return cls(inner_v8=inner_v8)

    def cid(self) -> str:
        ph_cid = "untrained"
        if self.per_role_per_regime_heads_v9:
            payload = sorted(
                (str(k[0]) + "::" + str(k[1]),
                 _np.asarray(v, dtype=_np.float64).tobytes().hex())
                for k, v
                in self.per_role_per_regime_heads_v9.items())
            ph_cid = _sha256_hex(payload)
        arh_cid = "untrained"
        if self.agent_replacement_routing_head is not None:
            arh_cid = _sha256_hex(
                self.agent_replacement_routing_head.tobytes().hex())
        return _sha256_hex({
            "schema": W68_REPLAY_CONTROLLER_V9_SCHEMA_VERSION,
            "kind": "replay_controller_v9",
            "inner_v8_cid": str(self.inner_v8.cid()),
            "per_role_per_regime_heads_v9_cid": ph_cid,
            "agent_replacement_routing_head_cid": arh_cid,
            "partial_contradiction_threshold": float(round(
                self.partial_contradiction_threshold, 12)),
            "agent_replacement_threshold": float(round(
                self.agent_replacement_threshold, 12)),
        })

    def classify_regime_v9(
            self, c: ReplayCandidate, *,
            partial_contradiction_flag: float = 0.0,
            agent_replacement_flag: float = 0.0,
            n_active_branches: int = 1,
            **v8_kwargs: Any,
    ) -> str:
        if (float(partial_contradiction_flag)
                >= float(self.partial_contradiction_threshold)):
            return W68_REPLAY_REGIME_PARTIAL_CONTRADICTION
        if (float(agent_replacement_flag)
                >= float(self.agent_replacement_threshold)):
            return W68_REPLAY_REGIME_AGENT_REPLACEMENT
        return self.inner_v8.classify_regime_v8(
            c, n_active_branches=int(n_active_branches),
            **v8_kwargs)

    def decide_agent_replacement_routing(
            self, *, team_features: Sequence[float],
    ) -> tuple[str, float]:
        """Returns (routing_label, score)."""
        if self.agent_replacement_routing_head is None:
            return "no_agent_replacement", 0.0
        feats = _np.asarray(
            list(team_features) + [1.0], dtype=_np.float64)
        if feats.shape[0] != int(
                self.agent_replacement_routing_head.shape[1]):
            return "no_agent_replacement", 0.0
        score = _np.asarray(
            self.agent_replacement_routing_head,
            dtype=_np.float64) @ feats
        probs = _softmax_n(score)
        idx = int(_np.argmax(probs))
        lab = W68_AGENT_REPLACEMENT_ROUTING_LABELS[idx]
        return lab, float(probs[idx])


@dataclasses.dataclass(frozen=True)
class ReplayControllerV9FitReport:
    schema: str
    n_role_regime_heads_v9: int
    agent_replacement_routing_trained: bool
    agent_replacement_routing_train_residual: float
    converged: bool
    ridge_lambda: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "n_role_regime_heads_v9": int(
                self.n_role_regime_heads_v9),
            "agent_replacement_routing_trained": bool(
                self.agent_replacement_routing_trained),
            "agent_replacement_routing_train_residual": float(round(
                self.agent_replacement_routing_train_residual, 12)),
            "converged": bool(self.converged),
            "ridge_lambda": float(round(self.ridge_lambda, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "replay_controller_v9_fit_report",
            "report": self.to_dict()})


def fit_replay_controller_v9_per_role(
        *, controller: ReplayControllerV9, role: str,
        train_candidates_per_regime: dict[
            str, Sequence[ReplayCandidate]],
        train_decisions_per_regime: dict[
            str, Sequence[str]],
        ridge_lambda: float = (
            W68_DEFAULT_REPLAY_V9_RIDGE_LAMBDA),
) -> tuple[ReplayControllerV9, ReplayControllerV9FitReport]:
    """Fit per-role per-regime heads for the V9 regimes."""
    if not train_candidates_per_regime:
        raise ValueError("must provide >= 1 regime")
    inner_v5 = controller.inner_v8.inner_v7.inner_v6.inner_v5
    fitted_v5, _ = fit_replay_controller_v5_per_regime(
        controller=inner_v5,
        train_candidates_per_regime=train_candidates_per_regime,
        train_decisions_per_regime=train_decisions_per_regime,
        ridge_lambda=float(ridge_lambda))
    new_heads = dict(controller.per_role_per_regime_heads_v9)
    for regime, head in fitted_v5.per_regime_heads_v5.items():
        new_heads[(str(role), str(regime))] = _np.asarray(
            head, dtype=_np.float64).copy()
    fitted = dataclasses.replace(
        controller, per_role_per_regime_heads_v9=new_heads)
    return fitted, ReplayControllerV9FitReport(
        schema=W68_REPLAY_CONTROLLER_V9_SCHEMA_VERSION,
        n_role_regime_heads_v9=int(len(new_heads)),
        agent_replacement_routing_trained=bool(
            fitted.agent_replacement_routing_head is not None),
        agent_replacement_routing_train_residual=0.0,
        converged=True,
        ridge_lambda=float(ridge_lambda),
    )


def fit_replay_v9_agent_replacement_routing_head(
        *, controller: ReplayControllerV9,
        train_team_features: Sequence[Sequence[float]],
        train_routing_labels: Sequence[str],
        ridge_lambda: float = (
            W68_DEFAULT_REPLAY_V9_RIDGE_LAMBDA),
) -> tuple[ReplayControllerV9, ReplayControllerV9FitReport]:
    """Fit the 6×11 agent-replacement-routing head."""
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
    Y = _np.zeros((X.shape[0], 6), dtype=_np.float64)
    for i, lab in enumerate(train_routing_labels):
        if lab not in W68_AGENT_REPLACEMENT_ROUTING_LABELS:
            raise ValueError(
                f"unknown routing label {lab!r}")
        idx = W68_AGENT_REPLACEMENT_ROUTING_LABELS.index(str(lab))
        Y[i, idx] = 1.0
    lam = max(float(ridge_lambda), 1e-9)
    A = X.T @ X + lam * _np.eye(X.shape[1], dtype=_np.float64)
    b = X.T @ Y
    try:
        W = _np.linalg.solve(A, b)
    except Exception:
        W = _np.zeros((X.shape[1], 6), dtype=_np.float64)
    Y_hat = X @ W
    residual = float(_np.mean(_np.abs(Y - Y_hat)))
    fitted = dataclasses.replace(
        controller, agent_replacement_routing_head=W.T.copy())
    return fitted, ReplayControllerV9FitReport(
        schema=W68_REPLAY_CONTROLLER_V9_SCHEMA_VERSION,
        n_role_regime_heads_v9=int(
            len(controller.per_role_per_regime_heads_v9)),
        agent_replacement_routing_trained=True,
        agent_replacement_routing_train_residual=float(residual),
        converged=bool(residual <= 1.0),
        ridge_lambda=float(ridge_lambda),
    )


@dataclasses.dataclass(frozen=True)
class ReplayControllerV9Witness:
    schema: str
    controller_cid: str
    n_role_regime_heads_v9: int
    agent_replacement_routing_trained: bool
    n_audit_entries_v9: int
    inner_v8_witness_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "controller_cid": str(self.controller_cid),
            "n_role_regime_heads_v9": int(
                self.n_role_regime_heads_v9),
            "agent_replacement_routing_trained": bool(
                self.agent_replacement_routing_trained),
            "n_audit_entries_v9": int(self.n_audit_entries_v9),
            "inner_v8_witness_cid": str(
                self.inner_v8_witness_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "replay_controller_v9_witness",
            "witness": self.to_dict()})


def emit_replay_controller_v9_witness(
        controller: ReplayControllerV9,
) -> ReplayControllerV9Witness:
    from .replay_controller_v8 import (
        emit_replay_controller_v8_witness,
    )
    inner_w = emit_replay_controller_v8_witness(
        controller.inner_v8)
    return ReplayControllerV9Witness(
        schema=W68_REPLAY_CONTROLLER_V9_SCHEMA_VERSION,
        controller_cid=str(controller.cid()),
        n_role_regime_heads_v9=int(
            len(controller.per_role_per_regime_heads_v9)),
        agent_replacement_routing_trained=bool(
            controller.agent_replacement_routing_head is not None),
        n_audit_entries_v9=int(len(controller.audit_v9)),
        inner_v8_witness_cid=str(inner_w.cid()),
    )


__all__ = [
    "W68_REPLAY_CONTROLLER_V9_SCHEMA_VERSION",
    "W68_REPLAY_REGIME_PARTIAL_CONTRADICTION",
    "W68_REPLAY_REGIME_AGENT_REPLACEMENT",
    "W68_REPLAY_REGIMES_V9_NEW",
    "W68_REPLAY_REGIMES_V9",
    "W68_AGENT_REPLACEMENT_ROUTING_LABELS",
    "W68_DEFAULT_REPLAY_V9_RIDGE_LAMBDA",
    "W68_DEFAULT_PARTIAL_CONTRADICTION_THRESHOLD",
    "W68_DEFAULT_AGENT_REPLACEMENT_THRESHOLD",
    "ReplayControllerV9",
    "ReplayControllerV9FitReport",
    "fit_replay_controller_v9_per_role",
    "fit_replay_v9_agent_replacement_routing_head",
    "ReplayControllerV9Witness",
    "emit_replay_controller_v9_witness",
]
