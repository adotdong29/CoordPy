"""W66 M6 — Cache Controller V9.

Strictly extends W65's ``coordpy.cache_controller_v8``. V8 fit a
five-objective stacked ridge + per-role eviction head. V9 adds:

* **Six-objective stacked ridge** —
  ``fit_six_objective_ridge_v9`` adds a *team-failure-recovery*
  target column.
* **Per-role 7-dim eviction priority head** —
  ``fit_per_role_eviction_head_v9`` uses an extra feature
  (team-failure-recovery flag) on top of V8's six.

Honest scope (W66)
------------------

* ``W66-L-V9-CACHE-CONTROLLER-NO-AUTOGRAD-CAP`` documents.
* The six-objective fit picks the worst-residual column when
  applying a single head.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.cache_controller_v9 requires numpy") from exc

from .cache_controller_v8 import (
    CacheControllerV8,
    W65_CACHE_CONTROLLER_V8_SCHEMA_VERSION,
    W65_CACHE_POLICIES_V8,
    W65_CACHE_POLICY_COMPOSITE_V8,
    W65_DEFAULT_CACHE_V8_RIDGE_LAMBDA,
)
from .tiny_substrate_v3 import _ndarray_cid, _sha256_hex


W66_CACHE_CONTROLLER_V9_SCHEMA_VERSION: str = (
    "coordpy.cache_controller_v9.v1")
W66_CACHE_POLICY_SIX_OBJECTIVE_V9: str = "six_objective_v9"
W66_CACHE_POLICY_PER_ROLE_EVICTION_V9: str = (
    "per_role_eviction_v9")
W66_CACHE_POLICY_COMPOSITE_V9: str = "composite_v9"
W66_CACHE_POLICIES_V9: tuple[str, ...] = (
    *W65_CACHE_POLICIES_V8,
    W66_CACHE_POLICY_SIX_OBJECTIVE_V9,
    W66_CACHE_POLICY_PER_ROLE_EVICTION_V9,
    W66_CACHE_POLICY_COMPOSITE_V9,
)
W66_DEFAULT_CACHE_V9_RIDGE_LAMBDA: float = 0.10


@dataclasses.dataclass
class CacheControllerV9:
    policy: str
    inner_v8: CacheControllerV8
    # Six-objective head: (n_features × 6).
    six_objective_head: "_np.ndarray | None"
    # Per-role 7-dim heads.
    per_role_eviction_heads_v9: dict[str, "_np.ndarray"]
    ridge_lambda: float
    fit_seed: int

    @classmethod
    def init(
            cls, *,
            policy: str = W66_CACHE_POLICY_COMPOSITE_V9,
            d_model: int = 64, d_key: int = 8,
            ridge_lambda: float = (
                W66_DEFAULT_CACHE_V9_RIDGE_LAMBDA),
            fit_seed: int = 66090,
    ) -> "CacheControllerV9":
        if policy not in W66_CACHE_POLICIES_V9:
            raise ValueError(
                f"policy must be in {W66_CACHE_POLICIES_V9}, "
                f"got {policy!r}")
        inner_v8 = CacheControllerV8.init(
            policy=W65_CACHE_POLICY_COMPOSITE_V8,
            d_model=int(d_model), d_key=int(d_key),
            ridge_lambda=float(ridge_lambda),
            fit_seed=int(fit_seed))
        return cls(
            policy=str(policy), inner_v8=inner_v8,
            six_objective_head=None,
            per_role_eviction_heads_v9={},
            ridge_lambda=float(ridge_lambda),
            fit_seed=int(fit_seed),
        )

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W66_CACHE_CONTROLLER_V9_SCHEMA_VERSION,
            "kind": "cache_controller_v9",
            "policy": str(self.policy),
            "inner_v8_cid": str(self.inner_v8.cid()),
            "six_objective_head_cid": (
                _ndarray_cid(self.six_objective_head)
                if self.six_objective_head is not None
                else "untrained"),
            "per_role_eviction_heads_v9_cids": [
                [str(k), _ndarray_cid(v)]
                for k, v in sorted(
                    self.per_role_eviction_heads_v9.items())],
            "ridge_lambda": float(round(self.ridge_lambda, 12)),
            "fit_seed": int(self.fit_seed),
        })


@dataclasses.dataclass(frozen=True)
class CacheControllerV9FitReport:
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
            "kind": "cache_controller_v9_fit_report",
            "report": self.to_dict()})


def fit_six_objective_ridge_v9(
        *, controller: CacheControllerV9,
        train_features: Sequence[Sequence[float]],
        target_drop_oracle: Sequence[float],
        target_retrieval_relevance: Sequence[float],
        target_hidden_wins: Sequence[float],
        target_replay_dominance: Sequence[float],
        target_team_task_success: Sequence[float],
        target_team_failure_recovery: Sequence[float],
        ridge_lambda: float = W66_DEFAULT_CACHE_V9_RIDGE_LAMBDA,
) -> tuple[CacheControllerV9, CacheControllerV9FitReport]:
    """Six-objective stacked ridge. (n_features × 6)."""
    X = _np.asarray(train_features, dtype=_np.float64)
    ys = [
        _np.asarray(target_drop_oracle, dtype=_np.float64),
        _np.asarray(target_retrieval_relevance, dtype=_np.float64),
        _np.asarray(target_hidden_wins, dtype=_np.float64),
        _np.asarray(target_replay_dominance, dtype=_np.float64),
        _np.asarray(target_team_task_success, dtype=_np.float64),
        _np.asarray(
            target_team_failure_recovery, dtype=_np.float64),
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
        H = _np.zeros((X.shape[1], 6), dtype=_np.float64)
    Y_hat = X @ H
    per_pre = tuple(float(_np.mean(_np.abs(y))) for y in ys)
    per_post = tuple(
        float(_np.mean(_np.abs(y - Y_hat[:, i])))
        for i, y in enumerate(ys))
    fitted = dataclasses.replace(
        controller, six_objective_head=_np.asarray(
            H, dtype=_np.float64).copy())
    report = CacheControllerV9FitReport(
        schema=W66_CACHE_CONTROLLER_V9_SCHEMA_VERSION,
        fit_kind="six_objective_v9",
        n_train=int(n), n_objectives=6,
        per_objective_pre_residual=per_pre,
        per_objective_post_residual=per_post,
        converged=bool(
            all(po <= pr + 1e-9
                for pr, po in zip(per_pre, per_post))),
        ridge_lambda=float(ridge_lambda),
    )
    return fitted, report


def fit_per_role_eviction_head_v9(
        *, controller: CacheControllerV9, role: str,
        train_features: Sequence[Sequence[float]],
        target_eviction_priorities: Sequence[float],
        ridge_lambda: float = W66_DEFAULT_CACHE_V9_RIDGE_LAMBDA,
) -> tuple[CacheControllerV9, CacheControllerV9FitReport]:
    """Per-role 7-dim ridge head against eviction priorities."""
    X = _np.asarray(train_features, dtype=_np.float64)
    y = _np.asarray(
        target_eviction_priorities, dtype=_np.float64)
    n = int(X.shape[0])
    if n == 0 or int(y.size) != n:
        raise ValueError(
            "fit requires positive-length matching features")
    if int(X.shape[1]) != 7:
        raise ValueError("features must be 7-dim per slot")
    lam = max(float(ridge_lambda), 1e-9)
    A = X.T @ X + lam * _np.eye(7, dtype=_np.float64)
    b = X.T @ y
    try:
        theta = _np.linalg.solve(A, b)
    except Exception:
        theta = _np.zeros((7,), dtype=_np.float64)
    y_hat = X @ theta
    pre = float(_np.mean(_np.abs(y)))
    post = float(_np.mean(_np.abs(y - y_hat)))
    new_heads = dict(controller.per_role_eviction_heads_v9)
    new_heads[str(role)] = _np.asarray(
        theta, dtype=_np.float64).copy()
    fitted = dataclasses.replace(
        controller, per_role_eviction_heads_v9=new_heads)
    report = CacheControllerV9FitReport(
        schema=W66_CACHE_CONTROLLER_V9_SCHEMA_VERSION,
        fit_kind=f"per_role_eviction_v9::{role}",
        n_train=int(n), n_objectives=1,
        per_objective_pre_residual=(pre,),
        per_objective_post_residual=(post,),
        converged=bool(post <= pre + 1e-9),
        ridge_lambda=float(ridge_lambda),
    )
    return fitted, report


def score_per_role_eviction_v9(
        controller: CacheControllerV9, *, role: str,
        flag_count: int, hidden_write: float,
        replay_age: int, attention_receive_l1: float,
        cache_key_norm: float, mean_similarity_to_others: float,
        team_failure_recovery_flag: float,
) -> float:
    """Apply the per-role V9 eviction head. 0.0 if untrained."""
    th = controller.per_role_eviction_heads_v9.get(str(role))
    if th is None:
        return 0.0
    x = _np.array([
        float(flag_count), float(hidden_write),
        float(replay_age), float(attention_receive_l1),
        float(cache_key_norm),
        float(mean_similarity_to_others),
        float(team_failure_recovery_flag)], dtype=_np.float64)
    return float(_np.dot(th, x))


@dataclasses.dataclass(frozen=True)
class CacheControllerV9Witness:
    schema: str
    controller_cid: str
    six_objective_trained: bool
    n_per_role_heads_v9: int
    role_tags: tuple[str, ...]
    inner_v8_witness_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "controller_cid": str(self.controller_cid),
            "six_objective_trained": bool(
                self.six_objective_trained),
            "n_per_role_heads_v9": int(self.n_per_role_heads_v9),
            "role_tags": list(self.role_tags),
            "inner_v8_witness_cid": str(
                self.inner_v8_witness_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "cache_controller_v9_witness",
            "witness": self.to_dict()})


def emit_cache_controller_v9_witness(
        *, controller: CacheControllerV9,
) -> CacheControllerV9Witness:
    from .cache_controller_v8 import (
        emit_cache_controller_v8_witness,
    )
    inner_w = emit_cache_controller_v8_witness(
        controller=controller.inner_v8)
    return CacheControllerV9Witness(
        schema=W66_CACHE_CONTROLLER_V9_SCHEMA_VERSION,
        controller_cid=str(controller.cid()),
        six_objective_trained=bool(
            controller.six_objective_head is not None),
        n_per_role_heads_v9=int(
            len(controller.per_role_eviction_heads_v9)),
        role_tags=tuple(sorted(
            controller.per_role_eviction_heads_v9.keys())),
        inner_v8_witness_cid=str(inner_w.cid()),
    )


__all__ = [
    "W66_CACHE_CONTROLLER_V9_SCHEMA_VERSION",
    "W66_CACHE_POLICY_SIX_OBJECTIVE_V9",
    "W66_CACHE_POLICY_PER_ROLE_EVICTION_V9",
    "W66_CACHE_POLICY_COMPOSITE_V9",
    "W66_CACHE_POLICIES_V9",
    "W66_DEFAULT_CACHE_V9_RIDGE_LAMBDA",
    "CacheControllerV9",
    "CacheControllerV9FitReport",
    "fit_six_objective_ridge_v9",
    "fit_per_role_eviction_head_v9",
    "score_per_role_eviction_v9",
    "CacheControllerV9Witness",
    "emit_cache_controller_v9_witness",
]
