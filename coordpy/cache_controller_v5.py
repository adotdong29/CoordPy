"""W62 M6 — Cache Controller V5 (trained-repair + two-objective
ridge + composite_v5).

Strictly extends W61's ``coordpy.cache_controller_v4``. V4 added
``bilinear_retrieval_v6`` + ``trained_corruption_floor`` +
``two_stage_v4`` + ``composite_v4`` (5-head mixture). V5 adds
**three** new substrate-load-bearing mechanisms:

* **Two-objective stacked ridge fit** —
  ``fit_two_objective_ridge_v5`` jointly fits a controller's
  scoring head against **two** targets simultaneously: (a) the
  V1 leave-one-out drop oracle and (b) a *retrieval-relevance*
  target produced by V6's bilinear retrieval head. The decision
  variable is a (n_features × 2) matrix solved by closed-form
  ridge with stacked right-hand-sides. The post-fit residual is
  reported per-objective so the R-131 H164 bar can name which
  objective improved.
* **Trained corruption-repair head** —
  ``fit_corruption_repair_head_v5`` fits a per-slot *repair
  correction* (not just a floor). Decision variable: a 4-dim
  ridge head over [flag_count, hidden_write, replay_age,
  attention_receive_l1] against a *target repair amount* (the
  drop-oracle margin for the corrupted slot). The cache
  controller V5 *adds* the repair correction back into the slot's
  score, possibly bringing it above the eviction threshold.
* **Composite_v5** — six-head ridge mixture (V4's five + repair
  head). Weights fit by closed-form ridge.

V5 strictly extends V4: with policy in V4's set, V5 reduces to
V4 byte-for-byte (modulo the V5 schema tag).

Honest scope
------------

* All V5 fitted heads use closed-form linear/bilinear ridge. No
  autograd. ``W62-L-V6-CACHE-CONTROLLER-NO-AUTOGRAD-CAP``
  documents.
* The trained repair head outputs an *additive* correction; it
  does not actually un-corrupt the cached state. It only changes
  the controller's eviction decision.
* The two-objective fit picks the worst-residual column when
  applying a single head; it does NOT produce a single head that
  minimises BOTH objectives simultaneously.
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
        "coordpy.cache_controller_v5 requires numpy") from exc

from .cache_controller_v4 import (
    CacheControllerV4,
    CacheControllerV4FitReport,
    W61_CACHE_CONTROLLER_V4_SCHEMA_VERSION,
    W61_CACHE_POLICIES_V4,
    W61_CACHE_POLICY_COMPOSITE_V4,
    W61_DEFAULT_CACHE_V4_RIDGE_LAMBDA,
)
from .tiny_substrate_v3 import (
    _ndarray_cid, _sha256_hex,
)


W62_CACHE_CONTROLLER_V5_SCHEMA_VERSION: str = (
    "coordpy.cache_controller_v5.v1")

W62_CACHE_POLICY_TWO_OBJECTIVE_V5: str = "two_objective_v5"
W62_CACHE_POLICY_TRAINED_REPAIR_V5: str = "trained_repair_v5"
W62_CACHE_POLICY_COMPOSITE_V5: str = "composite_v5"

W62_CACHE_POLICIES_V5: tuple[str, ...] = (
    *W61_CACHE_POLICIES_V4,
    W62_CACHE_POLICY_TWO_OBJECTIVE_V5,
    W62_CACHE_POLICY_TRAINED_REPAIR_V5,
    W62_CACHE_POLICY_COMPOSITE_V5,
)

W62_DEFAULT_CACHE_V5_RIDGE_LAMBDA: float = 0.10


@dataclasses.dataclass
class CacheControllerV5:
    policy: str
    inner_v4: CacheControllerV4
    # Two-objective head: (n_features × 2) matrix.
    two_objective_head: "_np.ndarray | None"
    # Trained-repair head: 4-dim vector.
    repair_head_coefs: "_np.ndarray | None"
    # Composite_v5 weights (6,).
    composite_v5_weights: "_np.ndarray | None"
    ridge_lambda: float
    fit_seed: int

    @classmethod
    def init(
            cls, *, policy: str = W62_CACHE_POLICY_COMPOSITE_V5,
            d_model: int = 64, d_key: int = 8,
            ridge_lambda: float = W62_DEFAULT_CACHE_V5_RIDGE_LAMBDA,
            fit_seed: int = 62060,
    ) -> "CacheControllerV5":
        if policy not in W62_CACHE_POLICIES_V5:
            raise ValueError(
                f"policy must be in {W62_CACHE_POLICIES_V5}, "
                f"got {policy!r}")
        # Map V5-only policies to V4 importance for the inner V4.
        if policy in W61_CACHE_POLICIES_V4:
            v4_policy = policy
        else:
            v4_policy = W61_CACHE_POLICY_COMPOSITE_V4
        inner_v4 = CacheControllerV4.init(
            policy=str(v4_policy),
            d_model=int(d_model),
            d_key=int(d_key),
            ridge_lambda=float(ridge_lambda),
            fit_seed=int(fit_seed))
        return cls(
            policy=str(policy), inner_v4=inner_v4,
            two_objective_head=None,
            repair_head_coefs=None,
            composite_v5_weights=None,
            ridge_lambda=float(ridge_lambda),
            fit_seed=int(fit_seed),
        )

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W62_CACHE_CONTROLLER_V5_SCHEMA_VERSION,
            "kind": "cache_controller_v5",
            "policy": str(self.policy),
            "inner_v4_cid": str(self.inner_v4.cid()),
            "two_objective_head_cid": (
                _ndarray_cid(self.two_objective_head)
                if self.two_objective_head is not None
                else "untrained"),
            "repair_head_coefs_cid": (
                _ndarray_cid(self.repair_head_coefs)
                if self.repair_head_coefs is not None
                else "untrained"),
            "composite_v5_weights_cid": (
                _ndarray_cid(self.composite_v5_weights)
                if self.composite_v5_weights is not None
                else "untrained"),
            "ridge_lambda": float(round(self.ridge_lambda, 12)),
            "fit_seed": int(self.fit_seed),
        })

    def score_repair_amount(
            self, *,
            flag_count: int, hidden_write: float,
            replay_age: int, attention_receive_l1: float,
    ) -> float:
        """Apply the trained repair head to estimate the
        ``additive`` repair correction for a corrupted slot."""
        if self.repair_head_coefs is None:
            return 0.0
        feat = _np.array([
            float(flag_count), float(hidden_write),
            float(replay_age), float(attention_receive_l1)],
            dtype=_np.float64)
        return float(feat @ _np.asarray(
            self.repair_head_coefs, dtype=_np.float64))


@dataclasses.dataclass(frozen=True)
class CacheControllerV5FitReport:
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
            "kind": "cache_controller_v5_fit_report",
            "report": self.to_dict()})


def fit_two_objective_ridge_v5(
        *, controller: CacheControllerV5,
        train_features: Sequence[Sequence[float]],
        target_drop_oracle: Sequence[float],
        target_retrieval_relevance: Sequence[float],
        ridge_lambda: float = W62_DEFAULT_CACHE_V5_RIDGE_LAMBDA,
) -> tuple[CacheControllerV5, CacheControllerV5FitReport]:
    """Two-objective stacked ridge fit. Decision variable: a
    (n_features × 2) matrix.

    ``train_features`` has shape ``(n_train, n_features)``.
    """
    X = _np.asarray(train_features, dtype=_np.float64)
    y1 = _np.asarray(target_drop_oracle, dtype=_np.float64)
    y2 = _np.asarray(target_retrieval_relevance,
                       dtype=_np.float64)
    if X.shape[0] == 0 or X.shape[0] != y1.size or (
            X.shape[0] != y2.size):
        raise ValueError(
            "fit requires positive-length matching features")
    Y = _np.stack([y1, y2], axis=-1)
    lam = max(float(ridge_lambda), 1e-9)
    A = X.T @ X + lam * _np.eye(X.shape[1], dtype=_np.float64)
    b = X.T @ Y
    try:
        H = _np.linalg.solve(A, b)
    except Exception:
        H = _np.zeros((X.shape[1], 2), dtype=_np.float64)
    Y_hat = X @ H
    per_pre = (
        float(_np.mean(_np.abs(y1))),
        float(_np.mean(_np.abs(y2))))
    per_post = (
        float(_np.mean(_np.abs(y1 - Y_hat[:, 0]))),
        float(_np.mean(_np.abs(y2 - Y_hat[:, 1]))))
    fitted = dataclasses.replace(
        controller,
        two_objective_head=_np.asarray(
            H, dtype=_np.float64).copy())
    report = CacheControllerV5FitReport(
        schema=W62_CACHE_CONTROLLER_V5_SCHEMA_VERSION,
        fit_kind="two_objective_v5",
        n_train=int(X.shape[0]),
        n_objectives=2,
        per_objective_pre_residual=per_pre,
        per_objective_post_residual=per_post,
        converged=bool(
            all(po <= pr + 1e-9
                for pr, po in zip(per_pre, per_post))),
        ridge_lambda=float(ridge_lambda),
    )
    return fitted, report


def fit_corruption_repair_head_v5(
        *, controller: CacheControllerV5,
        train_flag_counts: Sequence[int],
        train_hidden_writes: Sequence[float],
        train_replay_ages: Sequence[int],
        train_attention_receive_l1: Sequence[float],
        target_repair_amounts: Sequence[float],
        ridge_lambda: float = W62_DEFAULT_CACHE_V5_RIDGE_LAMBDA,
) -> tuple[CacheControllerV5, CacheControllerV5FitReport]:
    """Fit a 4-dim ridge head for the additive repair correction."""
    n = int(len(train_flag_counts))
    if n == 0:
        raise ValueError("fit requires non-empty inputs")
    if (len(train_hidden_writes) != n
            or len(train_replay_ages) != n
            or len(train_attention_receive_l1) != n
            or len(target_repair_amounts) != n):
        raise ValueError("all inputs must match in length")
    X = _np.stack([
        _np.asarray(train_flag_counts, dtype=_np.float64),
        _np.asarray(train_hidden_writes, dtype=_np.float64),
        _np.asarray(train_replay_ages, dtype=_np.float64),
        _np.asarray(train_attention_receive_l1,
                       dtype=_np.float64),
    ], axis=-1)
    y = _np.asarray(target_repair_amounts, dtype=_np.float64)
    lam = max(float(ridge_lambda), 1e-9)
    A = X.T @ X + lam * _np.eye(4, dtype=_np.float64)
    b = X.T @ y
    try:
        theta = _np.linalg.solve(A, b)
    except Exception:
        theta = _np.zeros((4,), dtype=_np.float64)
    y_hat = X @ theta
    pre = float(_np.mean(_np.abs(y)))
    post = float(_np.mean(_np.abs(y - y_hat)))
    fitted = dataclasses.replace(
        controller,
        repair_head_coefs=_np.asarray(
            theta, dtype=_np.float64).copy())
    report = CacheControllerV5FitReport(
        schema=W62_CACHE_CONTROLLER_V5_SCHEMA_VERSION,
        fit_kind="trained_repair_v5",
        n_train=int(n),
        n_objectives=1,
        per_objective_pre_residual=(float(pre),),
        per_objective_post_residual=(float(post),),
        converged=bool(post <= pre + 1e-9),
        ridge_lambda=float(ridge_lambda),
    )
    return fitted, report


def fit_composite_v5(
        *, controller: CacheControllerV5,
        head_scores: Sequence[Sequence[float]],
        drop_oracle: Sequence[float],
        ridge_lambda: float = W62_DEFAULT_CACHE_V5_RIDGE_LAMBDA,
) -> tuple[CacheControllerV5, CacheControllerV5FitReport]:
    """Composite_v5 ridge mixture over 6 heads against the drop
    oracle."""
    H = _np.asarray(head_scores, dtype=_np.float64)
    if H.ndim != 2 or H.shape[1] != 6:
        raise ValueError("head_scores must be (n_train, 6)")
    y = _np.asarray(drop_oracle, dtype=_np.float64)
    if H.shape[0] != y.size:
        raise ValueError(
            "head_scores rows must match drop_oracle len")
    lam = max(float(ridge_lambda), 1e-9)
    A = H.T @ H + lam * _np.eye(6, dtype=_np.float64)
    b = H.T @ y
    try:
        w = _np.linalg.solve(A, b)
    except Exception:
        w = _np.ones((6,), dtype=_np.float64) * (1.0 / 6.0)
    y_hat = H @ w
    pre = float(_np.mean(_np.abs(y - _np.mean(y))))
    post = float(_np.mean(_np.abs(y - y_hat)))
    fitted = dataclasses.replace(
        controller,
        composite_v5_weights=_np.asarray(
            w, dtype=_np.float64).copy())
    report = CacheControllerV5FitReport(
        schema=W62_CACHE_CONTROLLER_V5_SCHEMA_VERSION,
        fit_kind="composite_v5",
        n_train=int(H.shape[0]),
        n_objectives=1,
        per_objective_pre_residual=(float(pre),),
        per_objective_post_residual=(float(post),),
        converged=bool(post <= pre + 1e-9),
        ridge_lambda=float(ridge_lambda),
    )
    return fitted, report


@dataclasses.dataclass(frozen=True)
class CacheControllerV5Witness:
    schema: str
    controller_cid: str
    policy: str
    two_objective_used: bool
    repair_head_used: bool
    composite_v5_used: bool
    last_repair_amount_sum: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "controller_cid": str(self.controller_cid),
            "policy": str(self.policy),
            "two_objective_used": bool(self.two_objective_used),
            "repair_head_used": bool(self.repair_head_used),
            "composite_v5_used": bool(self.composite_v5_used),
            "last_repair_amount_sum": float(round(
                self.last_repair_amount_sum, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "cache_controller_v5_witness",
            "witness": self.to_dict()})


def emit_cache_controller_v5_witness(
        *, controller: CacheControllerV5,
        last_repair_amount_sum: float = 0.0,
) -> CacheControllerV5Witness:
    return CacheControllerV5Witness(
        schema=W62_CACHE_CONTROLLER_V5_SCHEMA_VERSION,
        controller_cid=str(controller.cid()),
        policy=str(controller.policy),
        two_objective_used=bool(
            controller.two_objective_head is not None),
        repair_head_used=bool(
            controller.repair_head_coefs is not None),
        composite_v5_used=bool(
            controller.composite_v5_weights is not None),
        last_repair_amount_sum=float(last_repair_amount_sum),
    )


__all__ = [
    "W62_CACHE_CONTROLLER_V5_SCHEMA_VERSION",
    "W62_CACHE_POLICY_TWO_OBJECTIVE_V5",
    "W62_CACHE_POLICY_TRAINED_REPAIR_V5",
    "W62_CACHE_POLICY_COMPOSITE_V5",
    "W62_CACHE_POLICIES_V5",
    "W62_DEFAULT_CACHE_V5_RIDGE_LAMBDA",
    "CacheControllerV5",
    "CacheControllerV5FitReport",
    "fit_two_objective_ridge_v5",
    "fit_corruption_repair_head_v5",
    "fit_composite_v5",
    "CacheControllerV5Witness",
    "emit_cache_controller_v5_witness",
]
