"""W67 M6 — Cache Controller V10.

Strictly extends W66's ``coordpy.cache_controller_v9``. V9 fit a
six-objective stacked ridge + per-role 7-dim eviction head. V10
adds:

* **Seven-objective stacked ridge** —
  ``fit_seven_objective_ridge_v10`` adds a *branch-merge* target
  column.
* **Per-role 8-dim eviction priority head** —
  ``fit_per_role_eviction_head_v10`` uses an extra feature
  (branch-merge flag) on top of V9's seven.

Honest scope (W67)
------------------

* ``W67-L-V10-CACHE-CONTROLLER-NO-AUTOGRAD-CAP`` documents.
* All fits are closed-form ridge solves.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.cache_controller_v10 requires numpy") from exc

from .cache_controller_v9 import (
    CacheControllerV9,
    W66_CACHE_POLICIES_V9,
    W66_CACHE_POLICY_COMPOSITE_V9,
    W66_DEFAULT_CACHE_V9_RIDGE_LAMBDA,
)
from .tiny_substrate_v3 import _ndarray_cid, _sha256_hex


W67_CACHE_CONTROLLER_V10_SCHEMA_VERSION: str = (
    "coordpy.cache_controller_v10.v1")
W67_CACHE_POLICY_SEVEN_OBJECTIVE_V10: str = "seven_objective_v10"
W67_CACHE_POLICY_PER_ROLE_EVICTION_V10: str = (
    "per_role_eviction_v10")
W67_CACHE_POLICY_COMPOSITE_V10: str = "composite_v10"
W67_CACHE_POLICIES_V10: tuple[str, ...] = (
    *W66_CACHE_POLICIES_V9,
    W67_CACHE_POLICY_SEVEN_OBJECTIVE_V10,
    W67_CACHE_POLICY_PER_ROLE_EVICTION_V10,
    W67_CACHE_POLICY_COMPOSITE_V10,
)
W67_DEFAULT_CACHE_V10_RIDGE_LAMBDA: float = 0.10


@dataclasses.dataclass
class CacheControllerV10:
    policy: str
    inner_v9: CacheControllerV9
    seven_objective_head: "_np.ndarray | None"
    per_role_eviction_heads_v10: dict[str, "_np.ndarray"]
    ridge_lambda: float
    fit_seed: int

    @classmethod
    def init(
            cls, *,
            policy: str = W67_CACHE_POLICY_COMPOSITE_V10,
            d_model: int = 64, d_key: int = 8,
            ridge_lambda: float = (
                W67_DEFAULT_CACHE_V10_RIDGE_LAMBDA),
            fit_seed: int = 67100,
    ) -> "CacheControllerV10":
        if policy not in W67_CACHE_POLICIES_V10:
            raise ValueError(
                f"policy must be in {W67_CACHE_POLICIES_V10}, "
                f"got {policy!r}")
        inner_v9 = CacheControllerV9.init(
            policy=W66_CACHE_POLICY_COMPOSITE_V9,
            d_model=int(d_model), d_key=int(d_key),
            ridge_lambda=float(ridge_lambda),
            fit_seed=int(fit_seed))
        return cls(
            policy=str(policy), inner_v9=inner_v9,
            seven_objective_head=None,
            per_role_eviction_heads_v10={},
            ridge_lambda=float(ridge_lambda),
            fit_seed=int(fit_seed),
        )

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W67_CACHE_CONTROLLER_V10_SCHEMA_VERSION,
            "kind": "cache_controller_v10",
            "policy": str(self.policy),
            "inner_v9_cid": str(self.inner_v9.cid()),
            "seven_objective_head_cid": (
                _ndarray_cid(self.seven_objective_head)
                if self.seven_objective_head is not None
                else "untrained"),
            "per_role_eviction_heads_v10_cids": [
                [str(k), _ndarray_cid(v)]
                for k, v in sorted(
                    self.per_role_eviction_heads_v10.items())],
            "ridge_lambda": float(round(self.ridge_lambda, 12)),
            "fit_seed": int(self.fit_seed),
        })


@dataclasses.dataclass(frozen=True)
class CacheControllerV10FitReport:
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
            "kind": "cache_controller_v10_fit_report",
            "report": self.to_dict()})


def fit_seven_objective_ridge_v10(
        *, controller: CacheControllerV10,
        train_features: Sequence[Sequence[float]],
        target_drop_oracle: Sequence[float],
        target_retrieval_relevance: Sequence[float],
        target_hidden_wins: Sequence[float],
        target_replay_dominance: Sequence[float],
        target_team_task_success: Sequence[float],
        target_team_failure_recovery: Sequence[float],
        target_branch_merge: Sequence[float],
        ridge_lambda: float = W67_DEFAULT_CACHE_V10_RIDGE_LAMBDA,
) -> tuple[CacheControllerV10, CacheControllerV10FitReport]:
    """Seven-objective stacked ridge (n_features × 7)."""
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
        H = _np.zeros((X.shape[1], 7), dtype=_np.float64)
    Y_hat = X @ H
    per_pre = tuple(float(_np.mean(_np.abs(y))) for y in ys)
    per_post = tuple(
        float(_np.mean(_np.abs(y - Y_hat[:, i])))
        for i, y in enumerate(ys))
    fitted = dataclasses.replace(
        controller, seven_objective_head=_np.asarray(
            H, dtype=_np.float64).copy())
    report = CacheControllerV10FitReport(
        schema=W67_CACHE_CONTROLLER_V10_SCHEMA_VERSION,
        fit_kind="seven_objective_v10",
        n_train=int(n), n_objectives=7,
        per_objective_pre_residual=per_pre,
        per_objective_post_residual=per_post,
        converged=bool(
            all(po <= pr + 1e-9
                for pr, po in zip(per_pre, per_post))),
        ridge_lambda=float(ridge_lambda),
    )
    return fitted, report


def fit_per_role_eviction_head_v10(
        *, controller: CacheControllerV10, role: str,
        train_features: Sequence[Sequence[float]],
        target_eviction_priorities: Sequence[float],
        ridge_lambda: float = W67_DEFAULT_CACHE_V10_RIDGE_LAMBDA,
) -> tuple[CacheControllerV10, CacheControllerV10FitReport]:
    """Per-role 8-dim ridge head against eviction priorities."""
    X = _np.asarray(train_features, dtype=_np.float64)
    y = _np.asarray(
        target_eviction_priorities, dtype=_np.float64)
    n = int(X.shape[0])
    if n == 0 or int(y.size) != n:
        raise ValueError(
            "fit requires positive-length matching features")
    if int(X.shape[1]) != 8:
        raise ValueError("features must be 8-dim per slot")
    lam = max(float(ridge_lambda), 1e-9)
    A = X.T @ X + lam * _np.eye(8, dtype=_np.float64)
    b = X.T @ y
    try:
        theta = _np.linalg.solve(A, b)
    except Exception:
        theta = _np.zeros((8,), dtype=_np.float64)
    y_hat = X @ theta
    pre = float(_np.mean(_np.abs(y)))
    post = float(_np.mean(_np.abs(y - y_hat)))
    new_heads = dict(controller.per_role_eviction_heads_v10)
    new_heads[str(role)] = _np.asarray(
        theta, dtype=_np.float64).copy()
    fitted = dataclasses.replace(
        controller, per_role_eviction_heads_v10=new_heads)
    report = CacheControllerV10FitReport(
        schema=W67_CACHE_CONTROLLER_V10_SCHEMA_VERSION,
        fit_kind=f"per_role_eviction_v10::{role}",
        n_train=int(n), n_objectives=1,
        per_objective_pre_residual=(pre,),
        per_objective_post_residual=(post,),
        converged=bool(post <= pre + 1e-9),
        ridge_lambda=float(ridge_lambda),
    )
    return fitted, report


def score_per_role_eviction_v10(
        controller: CacheControllerV10, *, role: str,
        flag_count: int, hidden_write: float,
        replay_age: int, attention_receive_l1: float,
        cache_key_norm: float, mean_similarity_to_others: float,
        team_failure_recovery_flag: float,
        branch_merge_flag: float,
) -> float:
    """Apply the per-role V10 eviction head. 0.0 if untrained."""
    th = controller.per_role_eviction_heads_v10.get(str(role))
    if th is None:
        return 0.0
    x = _np.array([
        float(flag_count), float(hidden_write),
        float(replay_age), float(attention_receive_l1),
        float(cache_key_norm),
        float(mean_similarity_to_others),
        float(team_failure_recovery_flag),
        float(branch_merge_flag)], dtype=_np.float64)
    return float(_np.dot(th, x))


@dataclasses.dataclass(frozen=True)
class CacheControllerV10Witness:
    schema: str
    controller_cid: str
    seven_objective_trained: bool
    n_per_role_heads_v10: int
    role_tags: tuple[str, ...]
    inner_v9_witness_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "controller_cid": str(self.controller_cid),
            "seven_objective_trained": bool(
                self.seven_objective_trained),
            "n_per_role_heads_v10": int(
                self.n_per_role_heads_v10),
            "role_tags": list(self.role_tags),
            "inner_v9_witness_cid": str(
                self.inner_v9_witness_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "cache_controller_v10_witness",
            "witness": self.to_dict()})


def emit_cache_controller_v10_witness(
        *, controller: CacheControllerV10,
) -> CacheControllerV10Witness:
    from .cache_controller_v9 import (
        emit_cache_controller_v9_witness,
    )
    inner_w = emit_cache_controller_v9_witness(
        controller=controller.inner_v9)
    return CacheControllerV10Witness(
        schema=W67_CACHE_CONTROLLER_V10_SCHEMA_VERSION,
        controller_cid=str(controller.cid()),
        seven_objective_trained=bool(
            controller.seven_objective_head is not None),
        n_per_role_heads_v10=int(
            len(controller.per_role_eviction_heads_v10)),
        role_tags=tuple(sorted(
            controller.per_role_eviction_heads_v10.keys())),
        inner_v9_witness_cid=str(inner_w.cid()),
    )


__all__ = [
    "W67_CACHE_CONTROLLER_V10_SCHEMA_VERSION",
    "W67_CACHE_POLICY_SEVEN_OBJECTIVE_V10",
    "W67_CACHE_POLICY_PER_ROLE_EVICTION_V10",
    "W67_CACHE_POLICY_COMPOSITE_V10",
    "W67_CACHE_POLICIES_V10",
    "W67_DEFAULT_CACHE_V10_RIDGE_LAMBDA",
    "CacheControllerV10",
    "CacheControllerV10FitReport",
    "fit_seven_objective_ridge_v10",
    "fit_per_role_eviction_head_v10",
    "score_per_role_eviction_v10",
    "CacheControllerV10Witness",
    "emit_cache_controller_v10_witness",
]
