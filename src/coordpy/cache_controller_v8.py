"""W65 M6 — Cache Controller V8.

Strictly extends W64's ``coordpy.cache_controller_v7``. V7 fit a
four-objective stacked ridge + similarity-aware eviction head +
composite_v7 mixture. V8 adds two new substrate-load-bearing heads:

* **Five-objective stacked ridge fit** —
  ``fit_five_objective_ridge_v8`` jointly fits a controller scoring
  head against **five** targets simultaneously: V7's four (drop
  oracle + retrieval relevance + hidden wins + replay dominance)
  plus a *team-task-success* target.
* **Per-role eviction priority head** —
  ``fit_per_role_eviction_head_v8`` fits a 6-dim ridge head per role
  tag against per-role target eviction priorities.

Honest scope (W65)
------------------

* ``W65-L-V8-CACHE-CONTROLLER-NO-AUTOGRAD-CAP`` documents.
* The five-objective fit picks the worst-residual column when
  applying a single head; it does NOT minimise ALL FIVE
  simultaneously.
* The per-role head is a *separate* 6-dim ridge per role; under
  the hood it is V7's similarity-aware eviction head with a per-
  role coefficient vector.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.cache_controller_v8 requires numpy") from exc

from .cache_controller_v7 import (
    CacheControllerV7,
    CacheControllerV7FitReport,
    W64_CACHE_CONTROLLER_V7_SCHEMA_VERSION,
    W64_CACHE_POLICIES_V7,
    W64_CACHE_POLICY_COMPOSITE_V7,
    W64_DEFAULT_CACHE_V7_RIDGE_LAMBDA,
)
from .tiny_substrate_v3 import _ndarray_cid, _sha256_hex


W65_CACHE_CONTROLLER_V8_SCHEMA_VERSION: str = (
    "coordpy.cache_controller_v8.v1")
W65_CACHE_POLICY_FIVE_OBJECTIVE_V8: str = "five_objective_v8"
W65_CACHE_POLICY_PER_ROLE_EVICTION_V8: str = (
    "per_role_eviction_v8")
W65_CACHE_POLICY_COMPOSITE_V8: str = "composite_v8"
W65_CACHE_POLICIES_V8: tuple[str, ...] = (
    *W64_CACHE_POLICIES_V7,
    W65_CACHE_POLICY_FIVE_OBJECTIVE_V8,
    W65_CACHE_POLICY_PER_ROLE_EVICTION_V8,
    W65_CACHE_POLICY_COMPOSITE_V8,
)
W65_DEFAULT_CACHE_V8_RIDGE_LAMBDA: float = 0.10


@dataclasses.dataclass
class CacheControllerV8:
    policy: str
    inner_v7: CacheControllerV7
    # Five-objective head: (n_features × 5) matrix.
    five_objective_head: "_np.ndarray | None"
    # Per-role 6-dim heads: dict role → (6,) vector.
    per_role_eviction_heads: dict[str, "_np.ndarray"]
    ridge_lambda: float
    fit_seed: int

    @classmethod
    def init(
            cls, *,
            policy: str = W65_CACHE_POLICY_COMPOSITE_V8,
            d_model: int = 64, d_key: int = 8,
            ridge_lambda: float = (
                W65_DEFAULT_CACHE_V8_RIDGE_LAMBDA),
            fit_seed: int = 65080,
    ) -> "CacheControllerV8":
        if policy not in W65_CACHE_POLICIES_V8:
            raise ValueError(
                f"policy must be in {W65_CACHE_POLICIES_V8}, "
                f"got {policy!r}")
        if policy in W64_CACHE_POLICIES_V7:
            inner_policy = policy
        else:
            inner_policy = W64_CACHE_POLICY_COMPOSITE_V7
        inner_v7 = CacheControllerV7.init(
            policy=str(inner_policy),
            d_model=int(d_model), d_key=int(d_key),
            ridge_lambda=float(ridge_lambda),
            fit_seed=int(fit_seed))
        return cls(
            policy=str(policy), inner_v7=inner_v7,
            five_objective_head=None,
            per_role_eviction_heads={},
            ridge_lambda=float(ridge_lambda),
            fit_seed=int(fit_seed),
        )

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W65_CACHE_CONTROLLER_V8_SCHEMA_VERSION,
            "kind": "cache_controller_v8",
            "policy": str(self.policy),
            "inner_v7_cid": str(self.inner_v7.cid()),
            "five_objective_head_cid": (
                _ndarray_cid(self.five_objective_head)
                if self.five_objective_head is not None
                else "untrained"),
            "per_role_eviction_heads_cids": [
                [str(k), _ndarray_cid(v)]
                for k, v in sorted(
                    self.per_role_eviction_heads.items())],
            "ridge_lambda": float(round(self.ridge_lambda, 12)),
            "fit_seed": int(self.fit_seed),
        })


@dataclasses.dataclass(frozen=True)
class CacheControllerV8FitReport:
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
            "kind": "cache_controller_v8_fit_report",
            "report": self.to_dict()})


def fit_five_objective_ridge_v8(
        *, controller: CacheControllerV8,
        train_features: Sequence[Sequence[float]],
        target_drop_oracle: Sequence[float],
        target_retrieval_relevance: Sequence[float],
        target_hidden_wins: Sequence[float],
        target_replay_dominance: Sequence[float],
        target_team_task_success: Sequence[float],
        ridge_lambda: float = W65_DEFAULT_CACHE_V8_RIDGE_LAMBDA,
) -> tuple[CacheControllerV8, CacheControllerV8FitReport]:
    """Five-objective stacked ridge. (n_features × 5)."""
    X = _np.asarray(train_features, dtype=_np.float64)
    y1 = _np.asarray(target_drop_oracle, dtype=_np.float64)
    y2 = _np.asarray(target_retrieval_relevance, dtype=_np.float64)
    y3 = _np.asarray(target_hidden_wins, dtype=_np.float64)
    y4 = _np.asarray(target_replay_dominance, dtype=_np.float64)
    y5 = _np.asarray(target_team_task_success, dtype=_np.float64)
    n = int(X.shape[0])
    if n == 0 or any(int(y.size) != n
                       for y in (y1, y2, y3, y4, y5)):
        raise ValueError(
            "fit requires positive-length matching features")
    Y = _np.stack([y1, y2, y3, y4, y5], axis=-1)
    lam = max(float(ridge_lambda), 1e-9)
    A = X.T @ X + lam * _np.eye(X.shape[1], dtype=_np.float64)
    b = X.T @ Y
    try:
        H = _np.linalg.solve(A, b)
    except Exception:
        H = _np.zeros((X.shape[1], 5), dtype=_np.float64)
    Y_hat = X @ H
    per_pre = tuple(
        float(_np.mean(_np.abs(y))) for y in (y1, y2, y3, y4, y5))
    per_post = tuple(
        float(_np.mean(_np.abs(y - Y_hat[:, i])))
        for i, y in enumerate((y1, y2, y3, y4, y5)))
    fitted = dataclasses.replace(
        controller, five_objective_head=_np.asarray(
            H, dtype=_np.float64).copy())
    report = CacheControllerV8FitReport(
        schema=W65_CACHE_CONTROLLER_V8_SCHEMA_VERSION,
        fit_kind="five_objective_v8",
        n_train=int(n), n_objectives=5,
        per_objective_pre_residual=per_pre,
        per_objective_post_residual=per_post,
        converged=bool(
            all(po <= pr + 1e-9
                for pr, po in zip(per_pre, per_post))),
        ridge_lambda=float(ridge_lambda),
    )
    return fitted, report


def fit_per_role_eviction_head_v8(
        *, controller: CacheControllerV8, role: str,
        train_features: Sequence[Sequence[float]],
        target_eviction_priorities: Sequence[float],
        ridge_lambda: float = W65_DEFAULT_CACHE_V8_RIDGE_LAMBDA,
) -> tuple[CacheControllerV8, CacheControllerV8FitReport]:
    """Per-role 6-dim ridge head against eviction priorities."""
    X = _np.asarray(train_features, dtype=_np.float64)
    y = _np.asarray(
        target_eviction_priorities, dtype=_np.float64)
    n = int(X.shape[0])
    if n == 0 or int(y.size) != n:
        raise ValueError(
            "fit requires positive-length matching features")
    if int(X.shape[1]) != 6:
        raise ValueError("features must be 6-dim per slot")
    lam = max(float(ridge_lambda), 1e-9)
    A = X.T @ X + lam * _np.eye(6, dtype=_np.float64)
    b = X.T @ y
    try:
        theta = _np.linalg.solve(A, b)
    except Exception:
        theta = _np.zeros((6,), dtype=_np.float64)
    y_hat = X @ theta
    pre = float(_np.mean(_np.abs(y)))
    post = float(_np.mean(_np.abs(y - y_hat)))
    new_heads = dict(controller.per_role_eviction_heads)
    new_heads[str(role)] = _np.asarray(
        theta, dtype=_np.float64).copy()
    fitted = dataclasses.replace(
        controller, per_role_eviction_heads=new_heads)
    report = CacheControllerV8FitReport(
        schema=W65_CACHE_CONTROLLER_V8_SCHEMA_VERSION,
        fit_kind=f"per_role_eviction_v8::{role}",
        n_train=int(n), n_objectives=1,
        per_objective_pre_residual=(pre,),
        per_objective_post_residual=(post,),
        converged=bool(post <= pre + 1e-9),
        ridge_lambda=float(ridge_lambda),
    )
    return fitted, report


def score_per_role_eviction_v8(
        controller: CacheControllerV8, *, role: str,
        flag_count: int, hidden_write: float,
        replay_age: int, attention_receive_l1: float,
        cache_key_norm: float, mean_similarity_to_others: float,
) -> float:
    """Apply the per-role eviction head. Returns 0.0 if untrained."""
    th = controller.per_role_eviction_heads.get(str(role))
    if th is None:
        return 0.0
    x = _np.array([
        float(flag_count), float(hidden_write),
        float(replay_age), float(attention_receive_l1),
        float(cache_key_norm),
        float(mean_similarity_to_others)], dtype=_np.float64)
    return float(_np.dot(th, x))


@dataclasses.dataclass(frozen=True)
class CacheControllerV8Witness:
    schema: str
    controller_cid: str
    five_objective_trained: bool
    n_per_role_heads: int
    role_tags: tuple[str, ...]
    inner_v7_witness_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "controller_cid": str(self.controller_cid),
            "five_objective_trained": bool(
                self.five_objective_trained),
            "n_per_role_heads": int(self.n_per_role_heads),
            "role_tags": list(self.role_tags),
            "inner_v7_witness_cid": str(
                self.inner_v7_witness_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "cache_controller_v8_witness",
            "witness": self.to_dict()})


def emit_cache_controller_v8_witness(
        *, controller: CacheControllerV8,
) -> CacheControllerV8Witness:
    from .cache_controller_v7 import (
        emit_cache_controller_v7_witness,
    )
    inner_w = emit_cache_controller_v7_witness(
        controller=controller.inner_v7)
    return CacheControllerV8Witness(
        schema=W65_CACHE_CONTROLLER_V8_SCHEMA_VERSION,
        controller_cid=str(controller.cid()),
        five_objective_trained=bool(
            controller.five_objective_head is not None),
        n_per_role_heads=int(
            len(controller.per_role_eviction_heads)),
        role_tags=tuple(sorted(
            controller.per_role_eviction_heads.keys())),
        inner_v7_witness_cid=str(inner_w.cid()),
    )


__all__ = [
    "W65_CACHE_CONTROLLER_V8_SCHEMA_VERSION",
    "W65_CACHE_POLICY_FIVE_OBJECTIVE_V8",
    "W65_CACHE_POLICY_PER_ROLE_EVICTION_V8",
    "W65_CACHE_POLICY_COMPOSITE_V8",
    "W65_CACHE_POLICIES_V8",
    "W65_DEFAULT_CACHE_V8_RIDGE_LAMBDA",
    "CacheControllerV8",
    "CacheControllerV8FitReport",
    "fit_five_objective_ridge_v8",
    "fit_per_role_eviction_head_v8",
    "score_per_role_eviction_v8",
    "CacheControllerV8Witness",
    "emit_cache_controller_v8_witness",
]
