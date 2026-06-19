"""W79 M3 — Cache Controller V22.

Strictly extends W78's ``coordpy.cache_controller_v21``. V22 adds
a **nineteen-objective stacked ridge** (V21 was eighteen) with a
per-role 20-dim replacement-then-restart-after-long-delay-pressure
head.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.cache_controller_v22 requires numpy") from exc

from .cache_controller_v21 import (
    CacheControllerV21,
    W78_CACHE_CONTROLLER_V21_SCHEMA_VERSION,
    fit_eighteen_objective_ridge_v21,
)


W79_CACHE_CONTROLLER_V22_SCHEMA_VERSION: str = (
    "coordpy.cache_controller_v22.v1")
W79_CACHE_CONTROLLER_V22_N_OBJECTIVES: int = 19
W79_CACHE_CONTROLLER_V22_PER_ROLE_HEAD_DIM: int = 20


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _ndarray_cid(arr: "_np.ndarray | None") -> str:
    if arr is None:
        return "none"
    a = _np.ascontiguousarray(_np.asarray(arr, dtype=_np.float64))
    return hashlib.sha256(a.tobytes()).hexdigest()


@dataclasses.dataclass
class CacheControllerV22:
    inner_v21: CacheControllerV21
    nineteen_objective_head: "_np.ndarray | None" = None
    per_role_replacement_then_restart_after_long_delay_pressure_head: dict[
        str, "_np.ndarray"] = dataclasses.field(
        default_factory=dict)

    @classmethod
    def init(
            cls, *, fit_seed: int = 79110,
    ) -> "CacheControllerV22":
        inner = CacheControllerV21.init(fit_seed=int(fit_seed))
        return cls(inner_v21=inner)

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W79_CACHE_CONTROLLER_V22_SCHEMA_VERSION,
            "kind": "cache_controller_v22",
            "inner_v21_cid": str(self.inner_v21.cid()),
            "nineteen_objective_head_cid": _ndarray_cid(
                self.nineteen_objective_head),
            "per_role_replacement_then_restart_after_long_delay_pressure_head_cids": {
                k: _ndarray_cid(v)
                for k, v in sorted(
                    self
                    .per_role_replacement_then_restart_after_long_delay_pressure_head
                    .items())},
        })


@dataclasses.dataclass(frozen=True)
class CacheControllerV22FitReport:
    schema: str
    n_objectives: int
    n_examples: int
    post_fit_residual: float

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w79_cache_controller_v22_fit_report",
            "schema": str(self.schema),
            "n_objectives": int(self.n_objectives),
            "n_examples": int(self.n_examples),
            "post_fit_residual": float(round(
                self.post_fit_residual, 12)),
        })


def fit_nineteen_objective_ridge_v22(
        *,
        controller: CacheControllerV22,
        train_features: Sequence[Sequence[float]],
        targets_per_objective: Sequence[Sequence[float]],
        ridge_lambda: float = 0.10,
) -> tuple[CacheControllerV22, CacheControllerV22FitReport]:
    """V22 closed-form stacked ridge across 19 objectives.

    ``targets_per_objective`` is a list of 19 target vectors,
    each shape ``(N,)`` aligned with ``train_features``.
    """
    X = _np.asarray(train_features, dtype=_np.float64)
    if X.shape[0] == 0:
        rep = CacheControllerV22FitReport(
            schema=W79_CACHE_CONTROLLER_V22_SCHEMA_VERSION,
            n_objectives=int(
                W79_CACHE_CONTROLLER_V22_N_OBJECTIVES),
            n_examples=0, post_fit_residual=0.0,
        )
        return controller, rep
    if len(list(targets_per_objective)) != int(
            W79_CACHE_CONTROLLER_V22_N_OBJECTIVES):
        raise ValueError(
            "v22 fit expects exactly 19 objectives, got "
            f"{len(list(targets_per_objective))}")
    lam = max(float(ridge_lambda), 1e-9)
    A = X.T @ X + lam * _np.eye(
        X.shape[1], dtype=_np.float64)
    Y = _np.stack(
        [_np.asarray(t, dtype=_np.float64)
         for t in targets_per_objective], axis=1)
    B = X.T @ Y
    try:
        theta = _np.linalg.solve(A, B)
    except Exception:
        theta = _np.zeros(
            (X.shape[1],
             int(W79_CACHE_CONTROLLER_V22_N_OBJECTIVES)),
            dtype=_np.float64)
    y_hat = X @ theta
    post = float(_np.mean(_np.abs(Y - y_hat)))
    fitted = dataclasses.replace(
        controller,
        nineteen_objective_head=theta.astype(_np.float64))
    rep = CacheControllerV22FitReport(
        schema=W79_CACHE_CONTROLLER_V22_SCHEMA_VERSION,
        n_objectives=int(
            W79_CACHE_CONTROLLER_V22_N_OBJECTIVES),
        n_examples=int(X.shape[0]),
        post_fit_residual=float(post))
    return fitted, rep


def fit_per_role_replacement_then_restart_after_long_delay_pressure_head_v22(
        *,
        controller: CacheControllerV22,
        role: str,
        train_features: Sequence[Sequence[float]],
        target_priorities: Sequence[float],
        ridge_lambda: float = 0.10,
) -> tuple[CacheControllerV22, dict[str, Any]]:
    X = _np.asarray(train_features, dtype=_np.float64)
    y = _np.asarray(target_priorities, dtype=_np.float64)
    if X.shape[0] == 0:
        return controller, {"n": 0}
    lam = max(float(ridge_lambda), 1e-9)
    A = X.T @ X + lam * _np.eye(
        X.shape[1], dtype=_np.float64)
    b = X.T @ y
    try:
        theta = _np.linalg.solve(A, b)
    except Exception:
        theta = _np.zeros(
            (X.shape[1],), dtype=_np.float64)
    new_heads = dict(
        controller
        .per_role_replacement_then_restart_after_long_delay_pressure_head)
    new_heads[str(role)] = theta.astype(_np.float64).copy()
    fitted = dataclasses.replace(
        controller,
        per_role_replacement_then_restart_after_long_delay_pressure_head=new_heads)
    return fitted, {"role": str(role), "n": int(X.shape[0])}


@dataclasses.dataclass(frozen=True)
class CacheControllerV22Witness:
    schema: str
    controller_cid: str
    n_objectives: int
    n_roles_with_head: int

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w79_cache_controller_v22_witness",
            "schema": str(self.schema),
            "controller_cid": str(self.controller_cid),
            "n_objectives": int(self.n_objectives),
            "n_roles_with_head": int(
                self.n_roles_with_head),
        })


def emit_cache_controller_v22_witness(
        *, controller: CacheControllerV22,
) -> CacheControllerV22Witness:
    return CacheControllerV22Witness(
        schema=W79_CACHE_CONTROLLER_V22_SCHEMA_VERSION,
        controller_cid=str(controller.cid()),
        n_objectives=int(
            W79_CACHE_CONTROLLER_V22_N_OBJECTIVES),
        n_roles_with_head=int(len(
            controller
            .per_role_replacement_then_restart_after_long_delay_pressure_head)),
    )


__all__ = [
    "W79_CACHE_CONTROLLER_V22_SCHEMA_VERSION",
    "W79_CACHE_CONTROLLER_V22_N_OBJECTIVES",
    "W79_CACHE_CONTROLLER_V22_PER_ROLE_HEAD_DIM",
    "CacheControllerV22",
    "CacheControllerV22FitReport",
    "CacheControllerV22Witness",
    "fit_nineteen_objective_ridge_v22",
    "fit_per_role_replacement_then_restart_after_long_delay_pressure_head_v22",
    "emit_cache_controller_v22_witness",
]
