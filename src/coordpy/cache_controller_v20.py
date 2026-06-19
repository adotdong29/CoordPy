"""W77 M3 — Cache Controller V20.

Strictly extends W76's ``coordpy.cache_controller_v19``. V19 fit a
sixteen-objective stacked ridge + per-role 17-dim chain-then-
restart-pressure head. V20 adds:

* **Seventeen-objective stacked ridge** — adds a *post-restart-
  replacement-repair* target column on top of V19's sixteen.
* **Per-role 18-dim post-restart-replacement-pressure head** —
  adds an eighteenth feature (post-restart-replacement-window
  ratio) on top of V19's seventeen.

Honest scope (W77): closed-form ridge
(``W77-L-V22-NO-AUTOGRAD-CAP``).
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.cache_controller_v20 requires numpy") from exc

from .cache_controller_v19 import (
    CacheControllerV19,
    W76_CACHE_POLICIES_V19,
    W76_CACHE_POLICY_COMPOSITE_V19,
)
from .tiny_substrate_v3 import _ndarray_cid, _sha256_hex


W77_CACHE_CONTROLLER_V20_SCHEMA_VERSION: str = (
    "coordpy.cache_controller_v20.v1")
W77_CACHE_POLICY_SEVENTEEN_OBJECTIVE_V20: str = (
    "seventeen_objective_v20")
W77_CACHE_POLICY_PER_ROLE_POST_RESTART_REPLACEMENT_V20: str = (
    "per_role_post_restart_replacement_pressure_v20")
W77_CACHE_POLICY_COMPOSITE_V20: str = "composite_v20"
W77_CACHE_POLICIES_V20: tuple[str, ...] = (
    *W76_CACHE_POLICIES_V19,
    W77_CACHE_POLICY_SEVENTEEN_OBJECTIVE_V20,
    W77_CACHE_POLICY_PER_ROLE_POST_RESTART_REPLACEMENT_V20,
    W77_CACHE_POLICY_COMPOSITE_V20,
)
W77_DEFAULT_CACHE_V20_RIDGE_LAMBDA: float = 0.10


@dataclasses.dataclass
class CacheControllerV20:
    policy: str
    inner_v19: CacheControllerV19
    seventeen_objective_head: "_np.ndarray | None"
    per_role_post_restart_replacement_pressure_heads_v20: dict[
        str, "_np.ndarray"]
    ridge_lambda: float
    fit_seed: int

    @classmethod
    def init(
            cls, *,
            policy: str = W77_CACHE_POLICY_COMPOSITE_V20,
            d_model: int = 64, d_key: int = 8,
            ridge_lambda: float = (
                W77_DEFAULT_CACHE_V20_RIDGE_LAMBDA),
            fit_seed: int = 77100,
    ) -> "CacheControllerV20":
        if policy not in W77_CACHE_POLICIES_V20:
            raise ValueError(
                f"policy must be in {W77_CACHE_POLICIES_V20}, "
                f"got {policy!r}")
        inner_v19 = CacheControllerV19.init(
            policy=W76_CACHE_POLICY_COMPOSITE_V19,
            d_model=int(d_model), d_key=int(d_key),
            ridge_lambda=float(ridge_lambda),
            fit_seed=int(fit_seed))
        return cls(
            policy=str(policy), inner_v19=inner_v19,
            seventeen_objective_head=None,
            per_role_post_restart_replacement_pressure_heads_v20={},
            ridge_lambda=float(ridge_lambda),
            fit_seed=int(fit_seed),
        )

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W77_CACHE_CONTROLLER_V20_SCHEMA_VERSION,
            "kind": "cache_controller_v20",
            "policy": str(self.policy),
            "inner_v19_cid": str(self.inner_v19.cid()),
            "seventeen_objective_head_cid": (
                _ndarray_cid(self.seventeen_objective_head)
                if self.seventeen_objective_head is not None
                else "untrained"),
            "per_role_post_restart_replacement_pressure_heads_v20_cids":
                [
                    [str(k), _ndarray_cid(v)]
                    for k, v in sorted(
                        self
                        .per_role_post_restart_replacement_pressure_heads_v20
                        .items())],
            "ridge_lambda": float(round(self.ridge_lambda, 12)),
            "fit_seed": int(self.fit_seed),
        })


@dataclasses.dataclass(frozen=True)
class CacheControllerV20FitReport:
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
            "kind": "cache_controller_v20_fit_report",
            "report": self.to_dict()})


def fit_seventeen_objective_ridge_v20(
        *, controller: CacheControllerV20,
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
        target_chain_then_restart_repair: Sequence[float],
        target_post_restart_replacement_repair: Sequence[float],
        ridge_lambda: float = W77_DEFAULT_CACHE_V20_RIDGE_LAMBDA,
) -> tuple[CacheControllerV20, CacheControllerV20FitReport]:
    """Seventeen-objective stacked ridge (n_features × 17)."""
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
        _np.asarray(
            target_chain_then_restart_repair, dtype=_np.float64),
        _np.asarray(
            target_post_restart_replacement_repair,
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
        H = _np.zeros((X.shape[1], 17), dtype=_np.float64)
    Y_hat = X @ H
    per_pre = tuple(float(_np.mean(_np.abs(y))) for y in ys)
    per_post = tuple(
        float(_np.mean(_np.abs(y - Y_hat[:, i])))
        for i, y in enumerate(ys))
    fitted = dataclasses.replace(
        controller,
        seventeen_objective_head=_np.asarray(
            H, dtype=_np.float64).copy())
    report = CacheControllerV20FitReport(
        schema=W77_CACHE_CONTROLLER_V20_SCHEMA_VERSION,
        fit_kind="seventeen_objective_v20",
        n_train=int(n), n_objectives=17,
        per_objective_pre_residual=per_pre,
        per_objective_post_residual=per_post,
        converged=bool(
            all(po <= pr + 1e-9
                for pr, po in zip(per_pre, per_post))),
        ridge_lambda=float(ridge_lambda),
    )
    return fitted, report


def fit_per_role_post_restart_replacement_pressure_head_v20(
        *, controller: CacheControllerV20, role: str,
        train_features: Sequence[Sequence[float]],
        target_post_restart_replacement_priorities: Sequence[
            float],
        ridge_lambda: float = W77_DEFAULT_CACHE_V20_RIDGE_LAMBDA,
) -> tuple[CacheControllerV20, CacheControllerV20FitReport]:
    """Per-role 18-dim ridge head against post-restart-replacement
    priorities.

    Feature 18 (index 17) is the post-restart-replacement-window
    ratio in [0, 1].
    """
    X = _np.asarray(train_features, dtype=_np.float64)
    y = _np.asarray(
        target_post_restart_replacement_priorities,
        dtype=_np.float64)
    n = int(X.shape[0])
    if n == 0 or int(y.size) != n:
        raise ValueError(
            "fit requires positive-length matching features")
    if int(X.shape[1]) != 18:
        raise ValueError("features must be 18-dim per slot")
    lam = max(float(ridge_lambda), 1e-9)
    A = X.T @ X + lam * _np.eye(18, dtype=_np.float64)
    b = X.T @ y
    try:
        theta = _np.linalg.solve(A, b)
    except Exception:
        theta = _np.zeros((18,), dtype=_np.float64)
    y_hat = X @ theta
    pre = float(_np.mean(_np.abs(y)))
    post = float(_np.mean(_np.abs(y - y_hat)))
    new_heads = dict(
        controller
        .per_role_post_restart_replacement_pressure_heads_v20)
    new_heads[str(role)] = _np.asarray(
        theta, dtype=_np.float64).copy()
    fitted = dataclasses.replace(
        controller,
        per_role_post_restart_replacement_pressure_heads_v20=(
            new_heads))
    report = CacheControllerV20FitReport(
        schema=W77_CACHE_CONTROLLER_V20_SCHEMA_VERSION,
        fit_kind=(
            "per_role_post_restart_replacement_pressure_v20"),
        n_train=int(n), n_objectives=1,
        per_objective_pre_residual=(pre,),
        per_objective_post_residual=(post,),
        converged=bool(post <= pre + 1e-9),
        ridge_lambda=float(ridge_lambda),
    )
    return fitted, report


@dataclasses.dataclass(frozen=True)
class CacheControllerV20Witness:
    schema: str
    controller_cid: str
    n_objectives_trained: int
    n_per_role_post_restart_replacement_pressure_heads_trained: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "controller_cid": str(self.controller_cid),
            "n_objectives_trained": int(
                self.n_objectives_trained),
            "n_per_role_post_restart_replacement_pressure_heads_trained":
                int(
                    self
                    .n_per_role_post_restart_replacement_pressure_heads_trained
                ),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "cache_controller_v20_witness",
            "witness": self.to_dict()})


def emit_cache_controller_v20_witness(
        controller: CacheControllerV20,
) -> CacheControllerV20Witness:
    n_obj = (
        17 if controller.seventeen_objective_head is not None
        else 0)
    n_heads = int(len(
        controller
        .per_role_post_restart_replacement_pressure_heads_v20))
    return CacheControllerV20Witness(
        schema=W77_CACHE_CONTROLLER_V20_SCHEMA_VERSION,
        controller_cid=str(controller.cid()),
        n_objectives_trained=int(n_obj),
        n_per_role_post_restart_replacement_pressure_heads_trained=(
            int(n_heads)),
    )


__all__ = [
    "W77_CACHE_CONTROLLER_V20_SCHEMA_VERSION",
    "W77_CACHE_POLICY_SEVENTEEN_OBJECTIVE_V20",
    "W77_CACHE_POLICY_PER_ROLE_POST_RESTART_REPLACEMENT_V20",
    "W77_CACHE_POLICY_COMPOSITE_V20",
    "W77_CACHE_POLICIES_V20",
    "W77_DEFAULT_CACHE_V20_RIDGE_LAMBDA",
    "CacheControllerV20",
    "CacheControllerV20FitReport",
    "fit_seventeen_objective_ridge_v20",
    "fit_per_role_post_restart_replacement_pressure_head_v20",
    "CacheControllerV20Witness",
    "emit_cache_controller_v20_witness",
]
