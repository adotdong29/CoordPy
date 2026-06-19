"""W71 M3 — Cache Controller V14.

Strictly extends W70's ``coordpy.cache_controller_v13``. V13 fit a
ten-objective stacked ridge + per-role 11-dim budget-primary
priority head. V14 adds:

* **Eleven-objective stacked ridge** — adds a *restart-dominance*
  target column on top of V13's ten.
* **Per-role 12-dim restart-priority head** — adds a twelfth
  feature (restart-pressure ratio) on top of V13's eleven.

Honest scope (W71): closed-form ridge
(``W71-L-V16-NO-AUTOGRAD-CAP``).
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.cache_controller_v14 requires numpy") from exc

from .cache_controller_v13 import (
    CacheControllerV13,
    W70_CACHE_POLICIES_V13,
    W70_CACHE_POLICY_COMPOSITE_V13,
    W70_DEFAULT_CACHE_V13_RIDGE_LAMBDA,
)
from .tiny_substrate_v3 import _ndarray_cid, _sha256_hex


W71_CACHE_CONTROLLER_V14_SCHEMA_VERSION: str = (
    "coordpy.cache_controller_v14.v1")
W71_CACHE_POLICY_ELEVEN_OBJECTIVE_V14: str = "eleven_objective_v14"
W71_CACHE_POLICY_PER_ROLE_RESTART_PRIORITY_V14: str = (
    "per_role_restart_priority_v14")
W71_CACHE_POLICY_COMPOSITE_V14: str = "composite_v14"
W71_CACHE_POLICIES_V14: tuple[str, ...] = (
    *W70_CACHE_POLICIES_V13,
    W71_CACHE_POLICY_ELEVEN_OBJECTIVE_V14,
    W71_CACHE_POLICY_PER_ROLE_RESTART_PRIORITY_V14,
    W71_CACHE_POLICY_COMPOSITE_V14,
)
W71_DEFAULT_CACHE_V14_RIDGE_LAMBDA: float = 0.10


@dataclasses.dataclass
class CacheControllerV14:
    policy: str
    inner_v13: CacheControllerV13
    eleven_objective_head: "_np.ndarray | None"
    per_role_restart_priority_heads_v14: dict[
        str, "_np.ndarray"]
    ridge_lambda: float
    fit_seed: int

    @classmethod
    def init(
            cls, *,
            policy: str = W71_CACHE_POLICY_COMPOSITE_V14,
            d_model: int = 64, d_key: int = 8,
            ridge_lambda: float = (
                W71_DEFAULT_CACHE_V14_RIDGE_LAMBDA),
            fit_seed: int = 71100,
    ) -> "CacheControllerV14":
        if policy not in W71_CACHE_POLICIES_V14:
            raise ValueError(
                f"policy must be in {W71_CACHE_POLICIES_V14}, "
                f"got {policy!r}")
        inner_v13 = CacheControllerV13.init(
            policy=W70_CACHE_POLICY_COMPOSITE_V13,
            d_model=int(d_model), d_key=int(d_key),
            ridge_lambda=float(ridge_lambda),
            fit_seed=int(fit_seed))
        return cls(
            policy=str(policy), inner_v13=inner_v13,
            eleven_objective_head=None,
            per_role_restart_priority_heads_v14={},
            ridge_lambda=float(ridge_lambda),
            fit_seed=int(fit_seed),
        )

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W71_CACHE_CONTROLLER_V14_SCHEMA_VERSION,
            "kind": "cache_controller_v14",
            "policy": str(self.policy),
            "inner_v13_cid": str(self.inner_v13.cid()),
            "eleven_objective_head_cid": (
                _ndarray_cid(self.eleven_objective_head)
                if self.eleven_objective_head is not None
                else "untrained"),
            "per_role_restart_priority_heads_v14_cids": [
                [str(k), _ndarray_cid(v)]
                for k, v in sorted(
                    self
                    .per_role_restart_priority_heads_v14
                    .items())],
            "ridge_lambda": float(round(self.ridge_lambda, 12)),
            "fit_seed": int(self.fit_seed),
        })


@dataclasses.dataclass(frozen=True)
class CacheControllerV14FitReport:
    schema: str
    fit_kind: str
    n_train: int
    n_objectives: int
    per_objective_pre_residual: tuple[float, ...]
    per_objective_post_residual: tuple[float, ...]
    converged: bool
    ridge_lambda: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "fit_kind": str(self.fit_kind),
            "n_train": int(self.n_train),
            "n_objectives": int(self.n_objectives),
            "per_objective_pre_residual": [
                float(round(float(x), 12))
                for x in self.per_objective_pre_residual],
            "per_objective_post_residual": [
                float(round(float(x), 12))
                for x in self.per_objective_post_residual],
            "converged": bool(self.converged),
            "ridge_lambda": float(round(self.ridge_lambda, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "cache_controller_v14_fit_report",
            "report": self.to_dict()})


def fit_eleven_objective_ridge_v14(
        *, controller: CacheControllerV14,
        train_features: Sequence[Sequence[float]],
        target_drop_oracle: Sequence[float],
        target_retrieval_relevance: Sequence[float],
        target_hidden_wins: Sequence[float],
        target_replay_dominance: Sequence[float],
        target_team_task_success: Sequence[float],
        target_team_failure_recovery: Sequence[float],
        target_branch_merge: Sequence[float],
        target_partial_contradiction: Sequence[float],
        target_multi_branch_rejoin: Sequence[float],
        target_budget_primary: Sequence[float],
        target_restart_dominance: Sequence[float],
        ridge_lambda: float = W71_DEFAULT_CACHE_V14_RIDGE_LAMBDA,
) -> tuple[CacheControllerV14, CacheControllerV14FitReport]:
    """Eleven-objective stacked ridge (n_features × 11)."""
    X = _np.asarray(train_features, dtype=_np.float64)
    ys = [
        _np.asarray(target_drop_oracle, dtype=_np.float64),
        _np.asarray(
            target_retrieval_relevance, dtype=_np.float64),
        _np.asarray(target_hidden_wins, dtype=_np.float64),
        _np.asarray(target_replay_dominance, dtype=_np.float64),
        _np.asarray(target_team_task_success, dtype=_np.float64),
        _np.asarray(
            target_team_failure_recovery, dtype=_np.float64),
        _np.asarray(target_branch_merge, dtype=_np.float64),
        _np.asarray(
            target_partial_contradiction, dtype=_np.float64),
        _np.asarray(
            target_multi_branch_rejoin, dtype=_np.float64),
        _np.asarray(target_budget_primary, dtype=_np.float64),
        _np.asarray(
            target_restart_dominance, dtype=_np.float64),
    ]
    n = int(X.shape[0])
    if n == 0 or any(int(y.size) != n for y in ys):
        raise ValueError(
            "fit requires positive-length matching features")
    Y = _np.stack(ys, axis=-1)
    lam = max(float(ridge_lambda), 1e-9)
    A = X.T @ X + lam * _np.eye(X.shape[1], dtype=_np.float64)
    b = X.T @ Y
    try:
        H = _np.linalg.solve(A, b)
    except Exception:
        H = _np.zeros((X.shape[1], 11), dtype=_np.float64)
    Y_hat = X @ H
    per_pre = tuple(float(_np.mean(_np.abs(y))) for y in ys)
    per_post = tuple(
        float(_np.mean(_np.abs(y - Y_hat[:, i])))
        for i, y in enumerate(ys))
    fitted = dataclasses.replace(
        controller,
        eleven_objective_head=_np.asarray(
            H, dtype=_np.float64).copy())
    report = CacheControllerV14FitReport(
        schema=W71_CACHE_CONTROLLER_V14_SCHEMA_VERSION,
        fit_kind="eleven_objective_v14",
        n_train=int(n), n_objectives=11,
        per_objective_pre_residual=per_pre,
        per_objective_post_residual=per_post,
        converged=bool(
            all(po <= pr + 1e-9
                for pr, po in zip(per_pre, per_post))),
        ridge_lambda=float(ridge_lambda),
    )
    return fitted, report


def fit_per_role_restart_priority_head_v14(
        *, controller: CacheControllerV14, role: str,
        train_features: Sequence[Sequence[float]],
        target_restart_priorities: Sequence[float],
        ridge_lambda: float = W71_DEFAULT_CACHE_V14_RIDGE_LAMBDA,
) -> tuple[CacheControllerV14, CacheControllerV14FitReport]:
    """Per-role 12-dim ridge head against restart priorities.

    Feature 12 (index 11) is the restart-pressure ratio in
    ``[0, 1]``."""
    X = _np.asarray(train_features, dtype=_np.float64)
    y = _np.asarray(
        target_restart_priorities, dtype=_np.float64)
    n = int(X.shape[0])
    if n == 0 or int(y.size) != n:
        raise ValueError(
            "fit requires positive-length matching features")
    if int(X.shape[1]) != 12:
        raise ValueError("features must be 12-dim per slot")
    lam = max(float(ridge_lambda), 1e-9)
    A = X.T @ X + lam * _np.eye(12, dtype=_np.float64)
    b = X.T @ y
    try:
        theta = _np.linalg.solve(A, b)
    except Exception:
        theta = _np.zeros((12,), dtype=_np.float64)
    y_hat = X @ theta
    pre = float(_np.mean(_np.abs(y)))
    post = float(_np.mean(_np.abs(y - y_hat)))
    new_heads = dict(
        controller.per_role_restart_priority_heads_v14)
    new_heads[str(role)] = _np.asarray(
        theta, dtype=_np.float64).copy()
    fitted = dataclasses.replace(
        controller,
        per_role_restart_priority_heads_v14=new_heads)
    report = CacheControllerV14FitReport(
        schema=W71_CACHE_CONTROLLER_V14_SCHEMA_VERSION,
        fit_kind="per_role_restart_priority_v14",
        n_train=int(n), n_objectives=1,
        per_objective_pre_residual=(pre,),
        per_objective_post_residual=(post,),
        converged=bool(post <= pre + 1e-9),
        ridge_lambda=float(ridge_lambda),
    )
    return fitted, report


@dataclasses.dataclass(frozen=True)
class CacheControllerV14Witness:
    schema: str
    controller_cid: str
    n_objectives_trained: int
    n_per_role_restart_priority_heads_trained: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "controller_cid": str(self.controller_cid),
            "n_objectives_trained": int(
                self.n_objectives_trained),
            "n_per_role_restart_priority_heads_trained": int(
                self.n_per_role_restart_priority_heads_trained),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "cache_controller_v14_witness",
            "witness": self.to_dict()})


def emit_cache_controller_v14_witness(
        controller: CacheControllerV14,
) -> CacheControllerV14Witness:
    n_obj = (
        11 if controller.eleven_objective_head is not None
        else 0)
    n_heads = int(len(
        controller.per_role_restart_priority_heads_v14))
    return CacheControllerV14Witness(
        schema=W71_CACHE_CONTROLLER_V14_SCHEMA_VERSION,
        controller_cid=str(controller.cid()),
        n_objectives_trained=int(n_obj),
        n_per_role_restart_priority_heads_trained=int(n_heads),
    )


__all__ = [
    "W71_CACHE_CONTROLLER_V14_SCHEMA_VERSION",
    "W71_CACHE_POLICY_ELEVEN_OBJECTIVE_V14",
    "W71_CACHE_POLICY_PER_ROLE_RESTART_PRIORITY_V14",
    "W71_CACHE_POLICY_COMPOSITE_V14",
    "W71_CACHE_POLICIES_V14",
    "W71_DEFAULT_CACHE_V14_RIDGE_LAMBDA",
    "CacheControllerV14",
    "CacheControllerV14FitReport",
    "fit_eleven_objective_ridge_v14",
    "fit_per_role_restart_priority_head_v14",
    "CacheControllerV14Witness",
    "emit_cache_controller_v14_witness",
]
