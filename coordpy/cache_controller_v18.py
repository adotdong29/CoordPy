"""W75 M3 — Cache Controller V18.

Strictly extends W74's ``coordpy.cache_controller_v17``. V17 fit a
fourteen-objective stacked ridge + per-role 15-dim compound-
pressure head. V18 adds:

* **Fifteen-objective stacked ridge** — adds a *compound-chain-
  repair-after-replacement-then-rejoin* target column on top of
  V17's fourteen.
* **Per-role 16-dim compound-chain-pressure head** — adds a
  sixteenth feature (compound-chain-window ratio) on top of V17's
  fifteen.

Honest scope (W75): closed-form ridge
(``W75-L-V20-NO-AUTOGRAD-CAP``).
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.cache_controller_v18 requires numpy") from exc

from .cache_controller_v17 import (
    CacheControllerV17,
    W74_CACHE_POLICIES_V17,
    W74_CACHE_POLICY_COMPOSITE_V17,
)
from .tiny_substrate_v3 import _ndarray_cid, _sha256_hex


W75_CACHE_CONTROLLER_V18_SCHEMA_VERSION: str = (
    "coordpy.cache_controller_v18.v1")
W75_CACHE_POLICY_FIFTEEN_OBJECTIVE_V18: str = (
    "fifteen_objective_v18")
W75_CACHE_POLICY_PER_ROLE_COMPOUND_CHAIN_PRESSURE_V18: str = (
    "per_role_compound_chain_pressure_v18")
W75_CACHE_POLICY_COMPOSITE_V18: str = "composite_v18"
W75_CACHE_POLICIES_V18: tuple[str, ...] = (
    *W74_CACHE_POLICIES_V17,
    W75_CACHE_POLICY_FIFTEEN_OBJECTIVE_V18,
    W75_CACHE_POLICY_PER_ROLE_COMPOUND_CHAIN_PRESSURE_V18,
    W75_CACHE_POLICY_COMPOSITE_V18,
)
W75_DEFAULT_CACHE_V18_RIDGE_LAMBDA: float = 0.10


@dataclasses.dataclass
class CacheControllerV18:
    policy: str
    inner_v17: CacheControllerV17
    fifteen_objective_head: "_np.ndarray | None"
    per_role_compound_chain_pressure_heads_v18: dict[
        str, "_np.ndarray"]
    ridge_lambda: float
    fit_seed: int

    @classmethod
    def init(
            cls, *,
            policy: str = W75_CACHE_POLICY_COMPOSITE_V18,
            d_model: int = 64, d_key: int = 8,
            ridge_lambda: float = (
                W75_DEFAULT_CACHE_V18_RIDGE_LAMBDA),
            fit_seed: int = 75100,
    ) -> "CacheControllerV18":
        if policy not in W75_CACHE_POLICIES_V18:
            raise ValueError(
                f"policy must be in {W75_CACHE_POLICIES_V18}, "
                f"got {policy!r}")
        inner_v17 = CacheControllerV17.init(
            policy=W74_CACHE_POLICY_COMPOSITE_V17,
            d_model=int(d_model), d_key=int(d_key),
            ridge_lambda=float(ridge_lambda),
            fit_seed=int(fit_seed))
        return cls(
            policy=str(policy), inner_v17=inner_v17,
            fifteen_objective_head=None,
            per_role_compound_chain_pressure_heads_v18={},
            ridge_lambda=float(ridge_lambda),
            fit_seed=int(fit_seed),
        )

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W75_CACHE_CONTROLLER_V18_SCHEMA_VERSION,
            "kind": "cache_controller_v18",
            "policy": str(self.policy),
            "inner_v17_cid": str(self.inner_v17.cid()),
            "fifteen_objective_head_cid": (
                _ndarray_cid(self.fifteen_objective_head)
                if self.fifteen_objective_head is not None
                else "untrained"),
            "per_role_compound_chain_pressure_heads_v18_cids": [
                [str(k), _ndarray_cid(v)]
                for k, v in sorted(
                    self
                    .per_role_compound_chain_pressure_heads_v18
                    .items())],
            "ridge_lambda": float(round(self.ridge_lambda, 12)),
            "fit_seed": int(self.fit_seed),
        })


@dataclasses.dataclass(frozen=True)
class CacheControllerV18FitReport:
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
            "kind": "cache_controller_v18_fit_report",
            "report": self.to_dict()})


def fit_fifteen_objective_ridge_v18(
        *, controller: CacheControllerV18,
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
        target_replacement_after_ctr: Sequence[float],
        target_compound_repair: Sequence[float],
        target_compound_chain_repair: Sequence[float],
        ridge_lambda: float = W75_DEFAULT_CACHE_V18_RIDGE_LAMBDA,
) -> tuple[CacheControllerV18, CacheControllerV18FitReport]:
    """Fifteen-objective stacked ridge (n_features × 15)."""
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
        _np.asarray(
            target_replacement_after_ctr, dtype=_np.float64),
        _np.asarray(
            target_compound_repair, dtype=_np.float64),
        _np.asarray(
            target_compound_chain_repair, dtype=_np.float64),
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
        H = _np.zeros((X.shape[1], 15), dtype=_np.float64)
    Y_hat = X @ H
    per_pre = tuple(float(_np.mean(_np.abs(y))) for y in ys)
    per_post = tuple(
        float(_np.mean(_np.abs(y - Y_hat[:, i])))
        for i, y in enumerate(ys))
    fitted = dataclasses.replace(
        controller,
        fifteen_objective_head=_np.asarray(
            H, dtype=_np.float64).copy())
    report = CacheControllerV18FitReport(
        schema=W75_CACHE_CONTROLLER_V18_SCHEMA_VERSION,
        fit_kind="fifteen_objective_v18",
        n_train=int(n), n_objectives=15,
        per_objective_pre_residual=per_pre,
        per_objective_post_residual=per_post,
        converged=bool(
            all(po <= pr + 1e-9
                for pr, po in zip(per_pre, per_post))),
        ridge_lambda=float(ridge_lambda),
    )
    return fitted, report


def fit_per_role_compound_chain_pressure_head_v18(
        *, controller: CacheControllerV18, role: str,
        train_features: Sequence[Sequence[float]],
        target_compound_chain_priorities: Sequence[float],
        ridge_lambda: float = W75_DEFAULT_CACHE_V18_RIDGE_LAMBDA,
) -> tuple[CacheControllerV18, CacheControllerV18FitReport]:
    """Per-role 16-dim ridge head against compound-chain priorities.

    Feature 16 (index 15) is the compound-chain-window ratio in
    [0, 1].
    """
    X = _np.asarray(train_features, dtype=_np.float64)
    y = _np.asarray(
        target_compound_chain_priorities, dtype=_np.float64)
    n = int(X.shape[0])
    if n == 0 or int(y.size) != n:
        raise ValueError(
            "fit requires positive-length matching features")
    if int(X.shape[1]) != 16:
        raise ValueError("features must be 16-dim per slot")
    lam = max(float(ridge_lambda), 1e-9)
    A = X.T @ X + lam * _np.eye(16, dtype=_np.float64)
    b = X.T @ y
    try:
        theta = _np.linalg.solve(A, b)
    except Exception:
        theta = _np.zeros((16,), dtype=_np.float64)
    y_hat = X @ theta
    pre = float(_np.mean(_np.abs(y)))
    post = float(_np.mean(_np.abs(y - y_hat)))
    new_heads = dict(
        controller.per_role_compound_chain_pressure_heads_v18)
    new_heads[str(role)] = _np.asarray(
        theta, dtype=_np.float64).copy()
    fitted = dataclasses.replace(
        controller,
        per_role_compound_chain_pressure_heads_v18=new_heads)
    report = CacheControllerV18FitReport(
        schema=W75_CACHE_CONTROLLER_V18_SCHEMA_VERSION,
        fit_kind="per_role_compound_chain_pressure_v18",
        n_train=int(n), n_objectives=1,
        per_objective_pre_residual=(pre,),
        per_objective_post_residual=(post,),
        converged=bool(post <= pre + 1e-9),
        ridge_lambda=float(ridge_lambda),
    )
    return fitted, report


@dataclasses.dataclass(frozen=True)
class CacheControllerV18Witness:
    schema: str
    controller_cid: str
    n_objectives_trained: int
    n_per_role_compound_chain_pressure_heads_trained: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "controller_cid": str(self.controller_cid),
            "n_objectives_trained": int(
                self.n_objectives_trained),
            "n_per_role_compound_chain_pressure_heads_trained":
                int(
                    self
                    .n_per_role_compound_chain_pressure_heads_trained
                ),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "cache_controller_v18_witness",
            "witness": self.to_dict()})


def emit_cache_controller_v18_witness(
        controller: CacheControllerV18,
) -> CacheControllerV18Witness:
    n_obj = (
        15 if controller.fifteen_objective_head is not None
        else 0)
    n_heads = int(len(
        controller.per_role_compound_chain_pressure_heads_v18))
    return CacheControllerV18Witness(
        schema=W75_CACHE_CONTROLLER_V18_SCHEMA_VERSION,
        controller_cid=str(controller.cid()),
        n_objectives_trained=int(n_obj),
        n_per_role_compound_chain_pressure_heads_trained=int(
            n_heads),
    )


__all__ = [
    "W75_CACHE_CONTROLLER_V18_SCHEMA_VERSION",
    "W75_CACHE_POLICY_FIFTEEN_OBJECTIVE_V18",
    "W75_CACHE_POLICY_PER_ROLE_COMPOUND_CHAIN_PRESSURE_V18",
    "W75_CACHE_POLICY_COMPOSITE_V18",
    "W75_CACHE_POLICIES_V18",
    "W75_DEFAULT_CACHE_V18_RIDGE_LAMBDA",
    "CacheControllerV18",
    "CacheControllerV18FitReport",
    "fit_fifteen_objective_ridge_v18",
    "fit_per_role_compound_chain_pressure_head_v18",
    "CacheControllerV18Witness",
    "emit_cache_controller_v18_witness",
]
