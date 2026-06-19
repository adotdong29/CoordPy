"""W68 M6 — Cache Controller V11.

Strictly extends W67's ``coordpy.cache_controller_v10``. V10 fit a
seven-objective stacked ridge + per-role 8-dim eviction head. V11
adds:

* **Eight-objective stacked ridge** —
  ``fit_eight_objective_ridge_v11`` adds a *partial-contradiction*
  target column on top of V10's seven.
* **Per-role 9-dim agent-replacement-priority head** —
  ``fit_per_role_agent_replacement_head_v11`` uses an extra feature
  (agent-replacement flag) on top of V10's eight.

Honest scope (W68)
------------------

* ``W68-L-V11-CACHE-CONTROLLER-NO-AUTOGRAD-CAP`` documents.
* All fits are closed-form ridge solves.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.cache_controller_v11 requires numpy") from exc

from .cache_controller_v10 import (
    CacheControllerV10,
    W67_CACHE_POLICIES_V10,
    W67_CACHE_POLICY_COMPOSITE_V10,
    W67_DEFAULT_CACHE_V10_RIDGE_LAMBDA,
)
from .tiny_substrate_v3 import _ndarray_cid, _sha256_hex


W68_CACHE_CONTROLLER_V11_SCHEMA_VERSION: str = (
    "coordpy.cache_controller_v11.v1")
W68_CACHE_POLICY_EIGHT_OBJECTIVE_V11: str = "eight_objective_v11"
W68_CACHE_POLICY_PER_ROLE_AGENT_REPLACEMENT_V11: str = (
    "per_role_agent_replacement_v11")
W68_CACHE_POLICY_COMPOSITE_V11: str = "composite_v11"
W68_CACHE_POLICIES_V11: tuple[str, ...] = (
    *W67_CACHE_POLICIES_V10,
    W68_CACHE_POLICY_EIGHT_OBJECTIVE_V11,
    W68_CACHE_POLICY_PER_ROLE_AGENT_REPLACEMENT_V11,
    W68_CACHE_POLICY_COMPOSITE_V11,
)
W68_DEFAULT_CACHE_V11_RIDGE_LAMBDA: float = 0.10


@dataclasses.dataclass
class CacheControllerV11:
    policy: str
    inner_v10: CacheControllerV10
    eight_objective_head: "_np.ndarray | None"
    per_role_agent_replacement_heads_v11: dict[str, "_np.ndarray"]
    ridge_lambda: float
    fit_seed: int

    @classmethod
    def init(
            cls, *,
            policy: str = W68_CACHE_POLICY_COMPOSITE_V11,
            d_model: int = 64, d_key: int = 8,
            ridge_lambda: float = (
                W68_DEFAULT_CACHE_V11_RIDGE_LAMBDA),
            fit_seed: int = 68100,
    ) -> "CacheControllerV11":
        if policy not in W68_CACHE_POLICIES_V11:
            raise ValueError(
                f"policy must be in {W68_CACHE_POLICIES_V11}, "
                f"got {policy!r}")
        inner_v10 = CacheControllerV10.init(
            policy=W67_CACHE_POLICY_COMPOSITE_V10,
            d_model=int(d_model), d_key=int(d_key),
            ridge_lambda=float(ridge_lambda),
            fit_seed=int(fit_seed))
        return cls(
            policy=str(policy), inner_v10=inner_v10,
            eight_objective_head=None,
            per_role_agent_replacement_heads_v11={},
            ridge_lambda=float(ridge_lambda),
            fit_seed=int(fit_seed),
        )

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W68_CACHE_CONTROLLER_V11_SCHEMA_VERSION,
            "kind": "cache_controller_v11",
            "policy": str(self.policy),
            "inner_v10_cid": str(self.inner_v10.cid()),
            "eight_objective_head_cid": (
                _ndarray_cid(self.eight_objective_head)
                if self.eight_objective_head is not None
                else "untrained"),
            "per_role_agent_replacement_heads_v11_cids": [
                [str(k), _ndarray_cid(v)]
                for k, v in sorted(
                    self
                    .per_role_agent_replacement_heads_v11
                    .items())],
            "ridge_lambda": float(round(self.ridge_lambda, 12)),
            "fit_seed": int(self.fit_seed),
        })


@dataclasses.dataclass(frozen=True)
class CacheControllerV11FitReport:
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
            "kind": "cache_controller_v11_fit_report",
            "report": self.to_dict()})


def fit_eight_objective_ridge_v11(
        *, controller: CacheControllerV11,
        train_features: Sequence[Sequence[float]],
        target_drop_oracle: Sequence[float],
        target_retrieval_relevance: Sequence[float],
        target_hidden_wins: Sequence[float],
        target_replay_dominance: Sequence[float],
        target_team_task_success: Sequence[float],
        target_team_failure_recovery: Sequence[float],
        target_branch_merge: Sequence[float],
        target_partial_contradiction: Sequence[float],
        ridge_lambda: float = W68_DEFAULT_CACHE_V11_RIDGE_LAMBDA,
) -> tuple[CacheControllerV11, CacheControllerV11FitReport]:
    """Eight-objective stacked ridge (n_features × 8)."""
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
        _np.asarray(target_partial_contradiction, dtype=_np.float64),
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
        H = _np.zeros((X.shape[1], 8), dtype=_np.float64)
    Y_hat = X @ H
    per_pre = tuple(float(_np.mean(_np.abs(y))) for y in ys)
    per_post = tuple(
        float(_np.mean(_np.abs(y - Y_hat[:, i])))
        for i, y in enumerate(ys))
    fitted = dataclasses.replace(
        controller, eight_objective_head=_np.asarray(
            H, dtype=_np.float64).copy())
    report = CacheControllerV11FitReport(
        schema=W68_CACHE_CONTROLLER_V11_SCHEMA_VERSION,
        fit_kind="eight_objective_v11",
        n_train=int(n), n_objectives=8,
        per_objective_pre_residual=per_pre,
        per_objective_post_residual=per_post,
        converged=bool(
            all(po <= pr + 1e-9
                for pr, po in zip(per_pre, per_post))),
        ridge_lambda=float(ridge_lambda),
    )
    return fitted, report


def fit_per_role_agent_replacement_head_v11(
        *, controller: CacheControllerV11, role: str,
        train_features: Sequence[Sequence[float]],
        target_replacement_priorities: Sequence[float],
        ridge_lambda: float = W68_DEFAULT_CACHE_V11_RIDGE_LAMBDA,
) -> tuple[CacheControllerV11, CacheControllerV11FitReport]:
    """Per-role 9-dim ridge head against agent-replacement
    priorities."""
    X = _np.asarray(train_features, dtype=_np.float64)
    y = _np.asarray(
        target_replacement_priorities, dtype=_np.float64)
    n = int(X.shape[0])
    if n == 0 or int(y.size) != n:
        raise ValueError(
            "fit requires positive-length matching features")
    if int(X.shape[1]) != 9:
        raise ValueError("features must be 9-dim per slot")
    lam = max(float(ridge_lambda), 1e-9)
    A = X.T @ X + lam * _np.eye(9, dtype=_np.float64)
    b = X.T @ y
    try:
        theta = _np.linalg.solve(A, b)
    except Exception:
        theta = _np.zeros((9,), dtype=_np.float64)
    y_hat = X @ theta
    pre = float(_np.mean(_np.abs(y)))
    post = float(_np.mean(_np.abs(y - y_hat)))
    new_heads = dict(
        controller.per_role_agent_replacement_heads_v11)
    new_heads[str(role)] = _np.asarray(
        theta, dtype=_np.float64).copy()
    fitted = dataclasses.replace(
        controller,
        per_role_agent_replacement_heads_v11=new_heads)
    report = CacheControllerV11FitReport(
        schema=W68_CACHE_CONTROLLER_V11_SCHEMA_VERSION,
        fit_kind=f"per_role_agent_replacement_v11::{role}",
        n_train=int(n), n_objectives=1,
        per_objective_pre_residual=(pre,),
        per_objective_post_residual=(post,),
        converged=bool(post <= pre + 1e-9),
        ridge_lambda=float(ridge_lambda),
    )
    return fitted, report


@dataclasses.dataclass(frozen=True)
class CacheControllerV11Witness:
    schema: str
    controller_cid: str
    eight_objective_trained: bool
    n_per_role_agent_replacement_heads_v11: int
    inner_v10_witness_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "controller_cid": str(self.controller_cid),
            "eight_objective_trained": bool(
                self.eight_objective_trained),
            "n_per_role_agent_replacement_heads_v11": int(
                self.n_per_role_agent_replacement_heads_v11),
            "inner_v10_witness_cid": str(
                self.inner_v10_witness_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "cache_controller_v11_witness",
            "witness": self.to_dict()})


def emit_cache_controller_v11_witness(
        *, controller: CacheControllerV11,
) -> CacheControllerV11Witness:
    from .cache_controller_v10 import (
        emit_cache_controller_v10_witness,
    )
    inner_w = emit_cache_controller_v10_witness(
        controller=controller.inner_v10)
    return CacheControllerV11Witness(
        schema=W68_CACHE_CONTROLLER_V11_SCHEMA_VERSION,
        controller_cid=str(controller.cid()),
        eight_objective_trained=bool(
            controller.eight_objective_head is not None),
        n_per_role_agent_replacement_heads_v11=int(len(
            controller.per_role_agent_replacement_heads_v11)),
        inner_v10_witness_cid=str(inner_w.cid()),
    )


__all__ = [
    "W68_CACHE_CONTROLLER_V11_SCHEMA_VERSION",
    "W68_CACHE_POLICY_EIGHT_OBJECTIVE_V11",
    "W68_CACHE_POLICY_PER_ROLE_AGENT_REPLACEMENT_V11",
    "W68_CACHE_POLICY_COMPOSITE_V11",
    "W68_CACHE_POLICIES_V11",
    "W68_DEFAULT_CACHE_V11_RIDGE_LAMBDA",
    "CacheControllerV11",
    "CacheControllerV11FitReport",
    "fit_eight_objective_ridge_v11",
    "fit_per_role_agent_replacement_head_v11",
    "CacheControllerV11Witness",
    "emit_cache_controller_v11_witness",
]
