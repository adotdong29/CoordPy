"""W70 M3 — Cache Controller V13.

Strictly extends W69's ``coordpy.cache_controller_v12``. V12 fit a
nine-objective stacked ridge + per-role 10-dim silent-corruption
priority head. V13 adds:

* **Ten-objective stacked ridge** — adds a *budget-primary* target
  column on top of V12's nine.
* **Per-role 11-dim budget-primary priority head** — adds an
  eleventh feature (visible-token budget ratio) on top of V12's ten.

Honest scope (W70): closed-form ridge
(``W70-L-V15-NO-AUTOGRAD-CAP``).
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.cache_controller_v13 requires numpy") from exc

from .cache_controller_v12 import (
    CacheControllerV12,
    W69_CACHE_POLICIES_V12,
    W69_CACHE_POLICY_COMPOSITE_V12,
    W69_DEFAULT_CACHE_V12_RIDGE_LAMBDA,
)
from .tiny_substrate_v3 import _ndarray_cid, _sha256_hex


W70_CACHE_CONTROLLER_V13_SCHEMA_VERSION: str = (
    "coordpy.cache_controller_v13.v1")
W70_CACHE_POLICY_TEN_OBJECTIVE_V13: str = "ten_objective_v13"
W70_CACHE_POLICY_PER_ROLE_BUDGET_PRIMARY_V13: str = (
    "per_role_budget_primary_v13")
W70_CACHE_POLICY_COMPOSITE_V13: str = "composite_v13"
W70_CACHE_POLICIES_V13: tuple[str, ...] = (
    *W69_CACHE_POLICIES_V12,
    W70_CACHE_POLICY_TEN_OBJECTIVE_V13,
    W70_CACHE_POLICY_PER_ROLE_BUDGET_PRIMARY_V13,
    W70_CACHE_POLICY_COMPOSITE_V13,
)
W70_DEFAULT_CACHE_V13_RIDGE_LAMBDA: float = 0.10


@dataclasses.dataclass
class CacheControllerV13:
    policy: str
    inner_v12: CacheControllerV12
    ten_objective_head: "_np.ndarray | None"
    per_role_budget_primary_heads_v13: dict[str, "_np.ndarray"]
    ridge_lambda: float
    fit_seed: int

    @classmethod
    def init(
            cls, *,
            policy: str = W70_CACHE_POLICY_COMPOSITE_V13,
            d_model: int = 64, d_key: int = 8,
            ridge_lambda: float = (
                W70_DEFAULT_CACHE_V13_RIDGE_LAMBDA),
            fit_seed: int = 70100,
    ) -> "CacheControllerV13":
        if policy not in W70_CACHE_POLICIES_V13:
            raise ValueError(
                f"policy must be in {W70_CACHE_POLICIES_V13}, "
                f"got {policy!r}")
        inner_v12 = CacheControllerV12.init(
            policy=W69_CACHE_POLICY_COMPOSITE_V12,
            d_model=int(d_model), d_key=int(d_key),
            ridge_lambda=float(ridge_lambda),
            fit_seed=int(fit_seed))
        return cls(
            policy=str(policy), inner_v12=inner_v12,
            ten_objective_head=None,
            per_role_budget_primary_heads_v13={},
            ridge_lambda=float(ridge_lambda),
            fit_seed=int(fit_seed),
        )

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W70_CACHE_CONTROLLER_V13_SCHEMA_VERSION,
            "kind": "cache_controller_v13",
            "policy": str(self.policy),
            "inner_v12_cid": str(self.inner_v12.cid()),
            "ten_objective_head_cid": (
                _ndarray_cid(self.ten_objective_head)
                if self.ten_objective_head is not None
                else "untrained"),
            "per_role_budget_primary_heads_v13_cids": [
                [str(k), _ndarray_cid(v)]
                for k, v in sorted(
                    self
                    .per_role_budget_primary_heads_v13
                    .items())],
            "ridge_lambda": float(round(self.ridge_lambda, 12)),
            "fit_seed": int(self.fit_seed),
        })


@dataclasses.dataclass(frozen=True)
class CacheControllerV13FitReport:
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
            "kind": "cache_controller_v13_fit_report",
            "report": self.to_dict()})


def fit_ten_objective_ridge_v13(
        *, controller: CacheControllerV13,
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
        ridge_lambda: float = W70_DEFAULT_CACHE_V13_RIDGE_LAMBDA,
) -> tuple[CacheControllerV13, CacheControllerV13FitReport]:
    """Ten-objective stacked ridge (n_features × 10)."""
    X = _np.asarray(train_features, dtype=_np.float64)
    ys = [
        _np.asarray(target_drop_oracle, dtype=_np.float64),
        _np.asarray(target_retrieval_relevance, dtype=_np.float64),
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
        H = _np.zeros((X.shape[1], 10), dtype=_np.float64)
    Y_hat = X @ H
    per_pre = tuple(float(_np.mean(_np.abs(y))) for y in ys)
    per_post = tuple(
        float(_np.mean(_np.abs(y - Y_hat[:, i])))
        for i, y in enumerate(ys))
    fitted = dataclasses.replace(
        controller,
        ten_objective_head=_np.asarray(H, dtype=_np.float64).copy())
    report = CacheControllerV13FitReport(
        schema=W70_CACHE_CONTROLLER_V13_SCHEMA_VERSION,
        fit_kind="ten_objective_v13",
        n_train=int(n), n_objectives=10,
        per_objective_pre_residual=per_pre,
        per_objective_post_residual=per_post,
        converged=bool(
            all(po <= pr + 1e-9
                for pr, po in zip(per_pre, per_post))),
        ridge_lambda=float(ridge_lambda),
    )
    return fitted, report


def fit_per_role_budget_primary_head_v13(
        *, controller: CacheControllerV13, role: str,
        train_features: Sequence[Sequence[float]],
        target_budget_primary_priorities: Sequence[float],
        ridge_lambda: float = W70_DEFAULT_CACHE_V13_RIDGE_LAMBDA,
) -> tuple[CacheControllerV13, CacheControllerV13FitReport]:
    """Per-role 11-dim ridge head against budget-primary priorities.

    Feature 11 (index 10) is the visible-token budget ratio in
    ``[0, 1]``."""
    X = _np.asarray(train_features, dtype=_np.float64)
    y = _np.asarray(
        target_budget_primary_priorities, dtype=_np.float64)
    n = int(X.shape[0])
    if n == 0 or int(y.size) != n:
        raise ValueError(
            "fit requires positive-length matching features")
    if int(X.shape[1]) != 11:
        raise ValueError("features must be 11-dim per slot")
    lam = max(float(ridge_lambda), 1e-9)
    A = X.T @ X + lam * _np.eye(11, dtype=_np.float64)
    b = X.T @ y
    try:
        theta = _np.linalg.solve(A, b)
    except Exception:
        theta = _np.zeros((11,), dtype=_np.float64)
    y_hat = X @ theta
    pre = float(_np.mean(_np.abs(y)))
    post = float(_np.mean(_np.abs(y - y_hat)))
    new_heads = dict(
        controller.per_role_budget_primary_heads_v13)
    new_heads[str(role)] = _np.asarray(
        theta, dtype=_np.float64).copy()
    fitted = dataclasses.replace(
        controller,
        per_role_budget_primary_heads_v13=new_heads)
    report = CacheControllerV13FitReport(
        schema=W70_CACHE_CONTROLLER_V13_SCHEMA_VERSION,
        fit_kind="per_role_budget_primary_v13",
        n_train=int(n), n_objectives=1,
        per_objective_pre_residual=(pre,),
        per_objective_post_residual=(post,),
        converged=bool(post <= pre + 1e-9),
        ridge_lambda=float(ridge_lambda),
    )
    return fitted, report


@dataclasses.dataclass(frozen=True)
class CacheControllerV13Witness:
    schema: str
    controller_cid: str
    n_objectives_trained: int
    n_per_role_budget_primary_heads_trained: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "controller_cid": str(self.controller_cid),
            "n_objectives_trained": int(
                self.n_objectives_trained),
            "n_per_role_budget_primary_heads_trained": int(
                self.n_per_role_budget_primary_heads_trained),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "cache_controller_v13_witness",
            "witness": self.to_dict()})


def emit_cache_controller_v13_witness(
        controller: CacheControllerV13,
) -> CacheControllerV13Witness:
    n_obj = (
        10 if controller.ten_objective_head is not None else 0)
    n_heads = int(len(
        controller.per_role_budget_primary_heads_v13))
    return CacheControllerV13Witness(
        schema=W70_CACHE_CONTROLLER_V13_SCHEMA_VERSION,
        controller_cid=str(controller.cid()),
        n_objectives_trained=int(n_obj),
        n_per_role_budget_primary_heads_trained=int(n_heads),
    )


__all__ = [
    "W70_CACHE_CONTROLLER_V13_SCHEMA_VERSION",
    "W70_CACHE_POLICY_TEN_OBJECTIVE_V13",
    "W70_CACHE_POLICY_PER_ROLE_BUDGET_PRIMARY_V13",
    "W70_CACHE_POLICY_COMPOSITE_V13",
    "W70_CACHE_POLICIES_V13",
    "W70_DEFAULT_CACHE_V13_RIDGE_LAMBDA",
    "CacheControllerV13",
    "CacheControllerV13FitReport",
    "fit_ten_objective_ridge_v13",
    "fit_per_role_budget_primary_head_v13",
    "CacheControllerV13Witness",
    "emit_cache_controller_v13_witness",
]
