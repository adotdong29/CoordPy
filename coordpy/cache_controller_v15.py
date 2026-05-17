"""W72 M3 — Cache Controller V15.

Strictly extends W71's ``coordpy.cache_controller_v14``. V14 fit an
eleven-objective stacked ridge + per-role 12-dim restart-priority
head. V15 adds:

* **Twelve-objective stacked ridge** — adds a *delayed-rejoin-
  after-restart* target column on top of V14's eleven.
* **Per-role 13-dim rejoin-pressure head** — adds a thirteenth
  feature (rejoin-lag ratio) on top of V14's twelve.

Honest scope (W72): closed-form ridge
(``W72-L-V17-NO-AUTOGRAD-CAP``).
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.cache_controller_v15 requires numpy") from exc

from .cache_controller_v14 import (
    CacheControllerV14,
    W71_CACHE_POLICIES_V14,
    W71_CACHE_POLICY_COMPOSITE_V14,
    W71_DEFAULT_CACHE_V14_RIDGE_LAMBDA,
)
from .tiny_substrate_v3 import _ndarray_cid, _sha256_hex


W72_CACHE_CONTROLLER_V15_SCHEMA_VERSION: str = (
    "coordpy.cache_controller_v15.v1")
W72_CACHE_POLICY_TWELVE_OBJECTIVE_V15: str = "twelve_objective_v15"
W72_CACHE_POLICY_PER_ROLE_REJOIN_PRESSURE_V15: str = (
    "per_role_rejoin_pressure_v15")
W72_CACHE_POLICY_COMPOSITE_V15: str = "composite_v15"
W72_CACHE_POLICIES_V15: tuple[str, ...] = (
    *W71_CACHE_POLICIES_V14,
    W72_CACHE_POLICY_TWELVE_OBJECTIVE_V15,
    W72_CACHE_POLICY_PER_ROLE_REJOIN_PRESSURE_V15,
    W72_CACHE_POLICY_COMPOSITE_V15,
)
W72_DEFAULT_CACHE_V15_RIDGE_LAMBDA: float = 0.10


@dataclasses.dataclass
class CacheControllerV15:
    policy: str
    inner_v14: CacheControllerV14
    twelve_objective_head: "_np.ndarray | None"
    per_role_rejoin_pressure_heads_v15: dict[
        str, "_np.ndarray"]
    ridge_lambda: float
    fit_seed: int

    @classmethod
    def init(
            cls, *,
            policy: str = W72_CACHE_POLICY_COMPOSITE_V15,
            d_model: int = 64, d_key: int = 8,
            ridge_lambda: float = (
                W72_DEFAULT_CACHE_V15_RIDGE_LAMBDA),
            fit_seed: int = 72100,
    ) -> "CacheControllerV15":
        if policy not in W72_CACHE_POLICIES_V15:
            raise ValueError(
                f"policy must be in {W72_CACHE_POLICIES_V15}, "
                f"got {policy!r}")
        inner_v14 = CacheControllerV14.init(
            policy=W71_CACHE_POLICY_COMPOSITE_V14,
            d_model=int(d_model), d_key=int(d_key),
            ridge_lambda=float(ridge_lambda),
            fit_seed=int(fit_seed))
        return cls(
            policy=str(policy), inner_v14=inner_v14,
            twelve_objective_head=None,
            per_role_rejoin_pressure_heads_v15={},
            ridge_lambda=float(ridge_lambda),
            fit_seed=int(fit_seed),
        )

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W72_CACHE_CONTROLLER_V15_SCHEMA_VERSION,
            "kind": "cache_controller_v15",
            "policy": str(self.policy),
            "inner_v14_cid": str(self.inner_v14.cid()),
            "twelve_objective_head_cid": (
                _ndarray_cid(self.twelve_objective_head)
                if self.twelve_objective_head is not None
                else "untrained"),
            "per_role_rejoin_pressure_heads_v15_cids": [
                [str(k), _ndarray_cid(v)]
                for k, v in sorted(
                    self
                    .per_role_rejoin_pressure_heads_v15
                    .items())],
            "ridge_lambda": float(round(self.ridge_lambda, 12)),
            "fit_seed": int(self.fit_seed),
        })


@dataclasses.dataclass(frozen=True)
class CacheControllerV15FitReport:
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
            "kind": "cache_controller_v15_fit_report",
            "report": self.to_dict()})


def fit_twelve_objective_ridge_v15(
        *, controller: CacheControllerV15,
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
        target_delayed_rejoin_after_restart: Sequence[float],
        ridge_lambda: float = W72_DEFAULT_CACHE_V15_RIDGE_LAMBDA,
) -> tuple[CacheControllerV15, CacheControllerV15FitReport]:
    """Twelve-objective stacked ridge (n_features × 12)."""
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
        _np.asarray(
            target_delayed_rejoin_after_restart,
            dtype=_np.float64),
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
        H = _np.zeros((X.shape[1], 12), dtype=_np.float64)
    Y_hat = X @ H
    per_pre = tuple(float(_np.mean(_np.abs(y))) for y in ys)
    per_post = tuple(
        float(_np.mean(_np.abs(y - Y_hat[:, i])))
        for i, y in enumerate(ys))
    fitted = dataclasses.replace(
        controller,
        twelve_objective_head=_np.asarray(
            H, dtype=_np.float64).copy())
    report = CacheControllerV15FitReport(
        schema=W72_CACHE_CONTROLLER_V15_SCHEMA_VERSION,
        fit_kind="twelve_objective_v15",
        n_train=int(n), n_objectives=12,
        per_objective_pre_residual=per_pre,
        per_objective_post_residual=per_post,
        converged=bool(
            all(po <= pr + 1e-9
                for pr, po in zip(per_pre, per_post))),
        ridge_lambda=float(ridge_lambda),
    )
    return fitted, report


def fit_per_role_rejoin_pressure_head_v15(
        *, controller: CacheControllerV15, role: str,
        train_features: Sequence[Sequence[float]],
        target_rejoin_priorities: Sequence[float],
        ridge_lambda: float = W72_DEFAULT_CACHE_V15_RIDGE_LAMBDA,
) -> tuple[CacheControllerV15, CacheControllerV15FitReport]:
    """Per-role 13-dim ridge head against rejoin priorities.

    Feature 13 (index 12) is the rejoin-lag ratio in ``[0, 1]``.
    """
    X = _np.asarray(train_features, dtype=_np.float64)
    y = _np.asarray(
        target_rejoin_priorities, dtype=_np.float64)
    n = int(X.shape[0])
    if n == 0 or int(y.size) != n:
        raise ValueError(
            "fit requires positive-length matching features")
    if int(X.shape[1]) != 13:
        raise ValueError("features must be 13-dim per slot")
    lam = max(float(ridge_lambda), 1e-9)
    A = X.T @ X + lam * _np.eye(13, dtype=_np.float64)
    b = X.T @ y
    try:
        theta = _np.linalg.solve(A, b)
    except Exception:
        theta = _np.zeros((13,), dtype=_np.float64)
    y_hat = X @ theta
    pre = float(_np.mean(_np.abs(y)))
    post = float(_np.mean(_np.abs(y - y_hat)))
    new_heads = dict(
        controller.per_role_rejoin_pressure_heads_v15)
    new_heads[str(role)] = _np.asarray(
        theta, dtype=_np.float64).copy()
    fitted = dataclasses.replace(
        controller,
        per_role_rejoin_pressure_heads_v15=new_heads)
    report = CacheControllerV15FitReport(
        schema=W72_CACHE_CONTROLLER_V15_SCHEMA_VERSION,
        fit_kind="per_role_rejoin_pressure_v15",
        n_train=int(n), n_objectives=1,
        per_objective_pre_residual=(pre,),
        per_objective_post_residual=(post,),
        converged=bool(post <= pre + 1e-9),
        ridge_lambda=float(ridge_lambda),
    )
    return fitted, report


@dataclasses.dataclass(frozen=True)
class CacheControllerV15Witness:
    schema: str
    controller_cid: str
    n_objectives_trained: int
    n_per_role_rejoin_pressure_heads_trained: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "controller_cid": str(self.controller_cid),
            "n_objectives_trained": int(
                self.n_objectives_trained),
            "n_per_role_rejoin_pressure_heads_trained": int(
                self
                .n_per_role_rejoin_pressure_heads_trained),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "cache_controller_v15_witness",
            "witness": self.to_dict()})


def emit_cache_controller_v15_witness(
        controller: CacheControllerV15,
) -> CacheControllerV15Witness:
    n_obj = (
        12 if controller.twelve_objective_head is not None
        else 0)
    n_heads = int(len(
        controller.per_role_rejoin_pressure_heads_v15))
    return CacheControllerV15Witness(
        schema=W72_CACHE_CONTROLLER_V15_SCHEMA_VERSION,
        controller_cid=str(controller.cid()),
        n_objectives_trained=int(n_obj),
        n_per_role_rejoin_pressure_heads_trained=int(n_heads),
    )


__all__ = [
    "W72_CACHE_CONTROLLER_V15_SCHEMA_VERSION",
    "W72_CACHE_POLICY_TWELVE_OBJECTIVE_V15",
    "W72_CACHE_POLICY_PER_ROLE_REJOIN_PRESSURE_V15",
    "W72_CACHE_POLICY_COMPOSITE_V15",
    "W72_CACHE_POLICIES_V15",
    "W72_DEFAULT_CACHE_V15_RIDGE_LAMBDA",
    "CacheControllerV15",
    "CacheControllerV15FitReport",
    "fit_twelve_objective_ridge_v15",
    "fit_per_role_rejoin_pressure_head_v15",
    "CacheControllerV15Witness",
    "emit_cache_controller_v15_witness",
]
