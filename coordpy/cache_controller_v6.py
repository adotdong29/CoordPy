"""W63 M6 — Cache Controller V6 (three-objective + retrieval-repair
+ composite_v6).

Strictly extends W62's ``coordpy.cache_controller_v5``. V5 added a
two-objective stacked ridge, trained corruption-repair, and a
composite_v5 mixture. V6 adds three new substrate-load-bearing
mechanisms:

* **Three-objective stacked ridge fit** —
  ``fit_three_objective_ridge_v6`` jointly fits a controller's
  scoring head against **three** targets simultaneously: (a) the
  V1 leave-one-out drop oracle, (b) a retrieval-relevance target,
  and (c) a *hidden-wins* target (the per-slot
  ``hidden_vs_kv_contention`` value normalised to [0, 1]). The
  decision variable is a (n_features × 3) matrix solved by
  closed-form ridge with stacked right-hand-sides.
* **Trained retrieval-repair head** —
  ``fit_retrieval_repair_head_v6`` fits a per-slot retrieval-aware
  repair correction. Decision variable: a 5-dim ridge head over
  [flag_count, hidden_write, replay_age, attention_receive_l1,
  cache_key_norm] against a *target repair amount* (the
  retrieval-relevance margin for the corrupted slot).
* **Composite_v6** — seven-head ridge mixture (V5's six +
  retrieval-repair head). Weights fit by closed-form ridge.

Honest scope
------------

* All V6 fitted heads use closed-form linear/bilinear ridge. No
  autograd. ``W63-L-V6-CACHE-CONTROLLER-NO-AUTOGRAD-CAP``
  documents.
* The retrieval-repair head outputs an *additive* correction; it
  does not actually un-corrupt the cached state. It only changes
  the controller's eviction decision.
* The three-objective fit picks the worst-residual column when
  applying a single head; it does NOT produce a single head that
  minimises ALL THREE objectives simultaneously.
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
        "coordpy.cache_controller_v6 requires numpy") from exc

from .cache_controller_v5 import (
    CacheControllerV5,
    CacheControllerV5FitReport,
    W62_CACHE_CONTROLLER_V5_SCHEMA_VERSION,
    W62_CACHE_POLICIES_V5,
    W62_CACHE_POLICY_COMPOSITE_V5,
    W62_DEFAULT_CACHE_V5_RIDGE_LAMBDA,
)
from .tiny_substrate_v3 import (
    _ndarray_cid, _sha256_hex,
)


W63_CACHE_CONTROLLER_V6_SCHEMA_VERSION: str = (
    "coordpy.cache_controller_v6.v1")

W63_CACHE_POLICY_THREE_OBJECTIVE_V6: str = "three_objective_v6"
W63_CACHE_POLICY_RETRIEVAL_REPAIR_V6: str = "retrieval_repair_v6"
W63_CACHE_POLICY_COMPOSITE_V6: str = "composite_v6"

W63_CACHE_POLICIES_V6: tuple[str, ...] = (
    *W62_CACHE_POLICIES_V5,
    W63_CACHE_POLICY_THREE_OBJECTIVE_V6,
    W63_CACHE_POLICY_RETRIEVAL_REPAIR_V6,
    W63_CACHE_POLICY_COMPOSITE_V6,
)

W63_DEFAULT_CACHE_V6_RIDGE_LAMBDA: float = 0.10


@dataclasses.dataclass
class CacheControllerV6:
    policy: str
    inner_v5: CacheControllerV5
    # Three-objective head: (n_features × 3) matrix.
    three_objective_head: "_np.ndarray | None"
    # Retrieval-repair head: 5-dim vector.
    retrieval_repair_head_coefs: "_np.ndarray | None"
    # Composite_v6 weights (7,).
    composite_v6_weights: "_np.ndarray | None"
    ridge_lambda: float
    fit_seed: int

    @classmethod
    def init(
            cls, *, policy: str = W63_CACHE_POLICY_COMPOSITE_V6,
            d_model: int = 64, d_key: int = 8,
            ridge_lambda: float = W63_DEFAULT_CACHE_V6_RIDGE_LAMBDA,
            fit_seed: int = 63060,
    ) -> "CacheControllerV6":
        if policy not in W63_CACHE_POLICIES_V6:
            raise ValueError(
                f"policy must be in {W63_CACHE_POLICIES_V6}, "
                f"got {policy!r}")
        if policy in W62_CACHE_POLICIES_V5:
            v5_policy = policy
        else:
            v5_policy = W62_CACHE_POLICY_COMPOSITE_V5
        inner_v5 = CacheControllerV5.init(
            policy=str(v5_policy),
            d_model=int(d_model),
            d_key=int(d_key),
            ridge_lambda=float(ridge_lambda),
            fit_seed=int(fit_seed))
        return cls(
            policy=str(policy), inner_v5=inner_v5,
            three_objective_head=None,
            retrieval_repair_head_coefs=None,
            composite_v6_weights=None,
            ridge_lambda=float(ridge_lambda),
            fit_seed=int(fit_seed),
        )

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W63_CACHE_CONTROLLER_V6_SCHEMA_VERSION,
            "kind": "cache_controller_v6",
            "policy": str(self.policy),
            "inner_v5_cid": str(self.inner_v5.cid()),
            "three_objective_head_cid": (
                _ndarray_cid(self.three_objective_head)
                if self.three_objective_head is not None
                else "untrained"),
            "retrieval_repair_head_coefs_cid": (
                _ndarray_cid(self.retrieval_repair_head_coefs)
                if self.retrieval_repair_head_coefs is not None
                else "untrained"),
            "composite_v6_weights_cid": (
                _ndarray_cid(self.composite_v6_weights)
                if self.composite_v6_weights is not None
                else "untrained"),
            "ridge_lambda": float(round(self.ridge_lambda, 12)),
            "fit_seed": int(self.fit_seed),
        })

    def score_retrieval_repair_amount(
            self, *,
            flag_count: int, hidden_write: float,
            replay_age: int, attention_receive_l1: float,
            cache_key_norm: float,
    ) -> float:
        """Apply the trained retrieval-repair head."""
        if self.retrieval_repair_head_coefs is None:
            return 0.0
        feat = _np.array([
            float(flag_count), float(hidden_write),
            float(replay_age), float(attention_receive_l1),
            float(cache_key_norm)], dtype=_np.float64)
        return float(feat @ _np.asarray(
            self.retrieval_repair_head_coefs,
            dtype=_np.float64))


@dataclasses.dataclass(frozen=True)
class CacheControllerV6FitReport:
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
            "kind": "cache_controller_v6_fit_report",
            "report": self.to_dict()})


def fit_three_objective_ridge_v6(
        *, controller: CacheControllerV6,
        train_features: Sequence[Sequence[float]],
        target_drop_oracle: Sequence[float],
        target_retrieval_relevance: Sequence[float],
        target_hidden_wins: Sequence[float],
        ridge_lambda: float = W63_DEFAULT_CACHE_V6_RIDGE_LAMBDA,
) -> tuple[CacheControllerV6, CacheControllerV6FitReport]:
    """Three-objective stacked ridge fit. Decision variable:
    a (n_features × 3) matrix.
    """
    X = _np.asarray(train_features, dtype=_np.float64)
    y1 = _np.asarray(target_drop_oracle, dtype=_np.float64)
    y2 = _np.asarray(target_retrieval_relevance,
                       dtype=_np.float64)
    y3 = _np.asarray(target_hidden_wins, dtype=_np.float64)
    if X.shape[0] == 0 or X.shape[0] != y1.size or (
            X.shape[0] != y2.size) or (X.shape[0] != y3.size):
        raise ValueError(
            "fit requires positive-length matching features")
    Y = _np.stack([y1, y2, y3], axis=-1)
    lam = max(float(ridge_lambda), 1e-9)
    A = X.T @ X + lam * _np.eye(X.shape[1], dtype=_np.float64)
    b = X.T @ Y
    try:
        H = _np.linalg.solve(A, b)
    except Exception:
        H = _np.zeros((X.shape[1], 3), dtype=_np.float64)
    Y_hat = X @ H
    per_pre = (
        float(_np.mean(_np.abs(y1))),
        float(_np.mean(_np.abs(y2))),
        float(_np.mean(_np.abs(y3))))
    per_post = (
        float(_np.mean(_np.abs(y1 - Y_hat[:, 0]))),
        float(_np.mean(_np.abs(y2 - Y_hat[:, 1]))),
        float(_np.mean(_np.abs(y3 - Y_hat[:, 2]))))
    fitted = dataclasses.replace(
        controller,
        three_objective_head=_np.asarray(
            H, dtype=_np.float64).copy())
    report = CacheControllerV6FitReport(
        schema=W63_CACHE_CONTROLLER_V6_SCHEMA_VERSION,
        fit_kind="three_objective_v6",
        n_train=int(X.shape[0]),
        n_objectives=3,
        per_objective_pre_residual=per_pre,
        per_objective_post_residual=per_post,
        converged=bool(
            all(po <= pr + 1e-9
                for pr, po in zip(per_pre, per_post))),
        ridge_lambda=float(ridge_lambda),
    )
    return fitted, report


def fit_retrieval_repair_head_v6(
        *, controller: CacheControllerV6,
        train_flag_counts: Sequence[int],
        train_hidden_writes: Sequence[float],
        train_replay_ages: Sequence[int],
        train_attention_receive_l1: Sequence[float],
        train_cache_key_norms: Sequence[float],
        target_repair_amounts: Sequence[float],
        ridge_lambda: float = W63_DEFAULT_CACHE_V6_RIDGE_LAMBDA,
) -> tuple[CacheControllerV6, CacheControllerV6FitReport]:
    """Fit a 5-dim ridge head for the retrieval-aware repair
    correction."""
    n = int(len(train_flag_counts))
    if n == 0:
        raise ValueError("fit requires non-empty inputs")
    if (len(train_hidden_writes) != n
            or len(train_replay_ages) != n
            or len(train_attention_receive_l1) != n
            or len(train_cache_key_norms) != n
            or len(target_repair_amounts) != n):
        raise ValueError("all inputs must match in length")
    X = _np.stack([
        _np.asarray(train_flag_counts, dtype=_np.float64),
        _np.asarray(train_hidden_writes, dtype=_np.float64),
        _np.asarray(train_replay_ages, dtype=_np.float64),
        _np.asarray(train_attention_receive_l1,
                       dtype=_np.float64),
        _np.asarray(train_cache_key_norms, dtype=_np.float64),
    ], axis=-1)
    y = _np.asarray(target_repair_amounts, dtype=_np.float64)
    lam = max(float(ridge_lambda), 1e-9)
    A = X.T @ X + lam * _np.eye(5, dtype=_np.float64)
    b = X.T @ y
    try:
        theta = _np.linalg.solve(A, b)
    except Exception:
        theta = _np.zeros((5,), dtype=_np.float64)
    y_hat = X @ theta
    pre = float(_np.mean(_np.abs(y)))
    post = float(_np.mean(_np.abs(y - y_hat)))
    fitted = dataclasses.replace(
        controller,
        retrieval_repair_head_coefs=_np.asarray(
            theta, dtype=_np.float64).copy())
    report = CacheControllerV6FitReport(
        schema=W63_CACHE_CONTROLLER_V6_SCHEMA_VERSION,
        fit_kind="retrieval_repair_v6",
        n_train=int(n),
        n_objectives=1,
        per_objective_pre_residual=(float(pre),),
        per_objective_post_residual=(float(post),),
        converged=bool(post <= pre + 1e-9),
        ridge_lambda=float(ridge_lambda),
    )
    return fitted, report


def fit_composite_v6(
        *, controller: CacheControllerV6,
        head_scores: Sequence[Sequence[float]],
        drop_oracle: Sequence[float],
        ridge_lambda: float = W63_DEFAULT_CACHE_V6_RIDGE_LAMBDA,
) -> tuple[CacheControllerV6, CacheControllerV6FitReport]:
    """Composite_v6 ridge mixture over 7 heads against the drop
    oracle."""
    H = _np.asarray(head_scores, dtype=_np.float64)
    if H.ndim != 2 or H.shape[1] != 7:
        raise ValueError("head_scores must be (n_train, 7)")
    y = _np.asarray(drop_oracle, dtype=_np.float64)
    if H.shape[0] != y.size:
        raise ValueError(
            "head_scores rows must match drop_oracle len")
    lam = max(float(ridge_lambda), 1e-9)
    A = H.T @ H + lam * _np.eye(7, dtype=_np.float64)
    b = H.T @ y
    try:
        w = _np.linalg.solve(A, b)
    except Exception:
        w = _np.ones((7,), dtype=_np.float64) * (1.0 / 7.0)
    y_hat = H @ w
    pre = float(_np.mean(_np.abs(y - _np.mean(y))))
    post = float(_np.mean(_np.abs(y - y_hat)))
    fitted = dataclasses.replace(
        controller,
        composite_v6_weights=_np.asarray(
            w, dtype=_np.float64).copy())
    report = CacheControllerV6FitReport(
        schema=W63_CACHE_CONTROLLER_V6_SCHEMA_VERSION,
        fit_kind="composite_v6",
        n_train=int(H.shape[0]),
        n_objectives=1,
        per_objective_pre_residual=(float(pre),),
        per_objective_post_residual=(float(post),),
        converged=bool(post <= pre + 1e-9),
        ridge_lambda=float(ridge_lambda),
    )
    return fitted, report


@dataclasses.dataclass(frozen=True)
class CacheControllerV6Witness:
    schema: str
    controller_cid: str
    policy: str
    three_objective_used: bool
    retrieval_repair_used: bool
    composite_v6_used: bool
    last_retrieval_repair_sum: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "controller_cid": str(self.controller_cid),
            "policy": str(self.policy),
            "three_objective_used": bool(
                self.three_objective_used),
            "retrieval_repair_used": bool(
                self.retrieval_repair_used),
            "composite_v6_used": bool(self.composite_v6_used),
            "last_retrieval_repair_sum": float(round(
                self.last_retrieval_repair_sum, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "cache_controller_v6_witness",
            "witness": self.to_dict()})


def emit_cache_controller_v6_witness(
        *, controller: CacheControllerV6,
        last_retrieval_repair_sum: float = 0.0,
) -> CacheControllerV6Witness:
    return CacheControllerV6Witness(
        schema=W63_CACHE_CONTROLLER_V6_SCHEMA_VERSION,
        controller_cid=str(controller.cid()),
        policy=str(controller.policy),
        three_objective_used=bool(
            controller.three_objective_head is not None),
        retrieval_repair_used=bool(
            controller.retrieval_repair_head_coefs is not None),
        composite_v6_used=bool(
            controller.composite_v6_weights is not None),
        last_retrieval_repair_sum=float(
            last_retrieval_repair_sum),
    )


__all__ = [
    "W63_CACHE_CONTROLLER_V6_SCHEMA_VERSION",
    "W63_CACHE_POLICY_THREE_OBJECTIVE_V6",
    "W63_CACHE_POLICY_RETRIEVAL_REPAIR_V6",
    "W63_CACHE_POLICY_COMPOSITE_V6",
    "W63_CACHE_POLICIES_V6",
    "W63_DEFAULT_CACHE_V6_RIDGE_LAMBDA",
    "CacheControllerV6",
    "CacheControllerV6FitReport",
    "fit_three_objective_ridge_v6",
    "fit_retrieval_repair_head_v6",
    "fit_composite_v6",
    "CacheControllerV6Witness",
    "emit_cache_controller_v6_witness",
]
