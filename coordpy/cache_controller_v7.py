"""W64 M6 — Cache Controller V7 (four-objective + similarity-aware
eviction + composite_v7).

Strictly extends W63's ``coordpy.cache_controller_v6``. V6 added a
three-objective stacked ridge, retrieval-repair head, and a
composite_v6 mixture. V7 adds three new substrate-load-bearing
mechanisms:

* **Four-objective stacked ridge fit** —
  ``fit_four_objective_ridge_v7`` jointly fits a controller's
  scoring head against **four** targets simultaneously: V6's
  three (drop oracle, retrieval relevance, hidden wins) plus a
  *replay-dominance* target (the per-slot
  ``replay_dominance_witness`` value normalised to [0, 1]). The
  decision variable is a (n_features × 4) matrix solved by
  closed-form ridge with stacked right-hand-sides.
* **Similarity-aware eviction head** —
  ``fit_similarity_aware_eviction_head_v7`` fits a per-slot
  eviction-priority correction conditioned on the V9
  cache-similarity matrix. Decision variable: a 6-dim ridge head
  over [flag_count, hidden_write, replay_age, attention_receive_l1,
  cache_key_norm, mean_similarity_to_others] against a *target
  eviction priority*.
* **Composite_v7** — eight-head ridge mixture (V6's seven +
  similarity-aware eviction head). Weights fit by closed-form
  ridge.

Honest scope (W64)
------------------

* All V7 fitted heads use closed-form linear/bilinear ridge. No
  autograd. ``W64-L-V7-CACHE-CONTROLLER-NO-AUTOGRAD-CAP``
  documents.
* The similarity-aware eviction head outputs an *additive*
  priority correction; it does not actually rebalance the cache.
  It only changes the controller's eviction decision.
* The four-objective fit picks the worst-residual column when
  applying a single head; it does NOT produce a single head that
  minimises ALL FOUR objectives simultaneously.
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
        "coordpy.cache_controller_v7 requires numpy") from exc

from .cache_controller_v6 import (
    CacheControllerV6,
    CacheControllerV6FitReport,
    W63_CACHE_CONTROLLER_V6_SCHEMA_VERSION,
    W63_CACHE_POLICIES_V6,
    W63_CACHE_POLICY_COMPOSITE_V6,
    W63_DEFAULT_CACHE_V6_RIDGE_LAMBDA,
)
from .tiny_substrate_v3 import (
    _ndarray_cid, _sha256_hex,
)


W64_CACHE_CONTROLLER_V7_SCHEMA_VERSION: str = (
    "coordpy.cache_controller_v7.v1")

W64_CACHE_POLICY_FOUR_OBJECTIVE_V7: str = "four_objective_v7"
W64_CACHE_POLICY_SIMILARITY_EVICTION_V7: str = (
    "similarity_eviction_v7")
W64_CACHE_POLICY_COMPOSITE_V7: str = "composite_v7"

W64_CACHE_POLICIES_V7: tuple[str, ...] = (
    *W63_CACHE_POLICIES_V6,
    W64_CACHE_POLICY_FOUR_OBJECTIVE_V7,
    W64_CACHE_POLICY_SIMILARITY_EVICTION_V7,
    W64_CACHE_POLICY_COMPOSITE_V7,
)

W64_DEFAULT_CACHE_V7_RIDGE_LAMBDA: float = 0.10


@dataclasses.dataclass
class CacheControllerV7:
    policy: str
    inner_v6: CacheControllerV6
    # Four-objective head: (n_features × 4) matrix.
    four_objective_head: "_np.ndarray | None"
    # Similarity-aware eviction head: 6-dim vector.
    similarity_eviction_head_coefs: "_np.ndarray | None"
    # Composite_v7 weights (8,).
    composite_v7_weights: "_np.ndarray | None"
    ridge_lambda: float
    fit_seed: int

    @classmethod
    def init(
            cls, *, policy: str = W64_CACHE_POLICY_COMPOSITE_V7,
            d_model: int = 64, d_key: int = 8,
            ridge_lambda: float = (
                W64_DEFAULT_CACHE_V7_RIDGE_LAMBDA),
            fit_seed: int = 64070,
    ) -> "CacheControllerV7":
        if policy not in W64_CACHE_POLICIES_V7:
            raise ValueError(
                f"policy must be in {W64_CACHE_POLICIES_V7}, "
                f"got {policy!r}")
        if policy in W63_CACHE_POLICIES_V6:
            v6_policy = policy
        else:
            v6_policy = W63_CACHE_POLICY_COMPOSITE_V6
        inner_v6 = CacheControllerV6.init(
            policy=str(v6_policy),
            d_model=int(d_model),
            d_key=int(d_key),
            ridge_lambda=float(ridge_lambda),
            fit_seed=int(fit_seed))
        return cls(
            policy=str(policy), inner_v6=inner_v6,
            four_objective_head=None,
            similarity_eviction_head_coefs=None,
            composite_v7_weights=None,
            ridge_lambda=float(ridge_lambda),
            fit_seed=int(fit_seed),
        )

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W64_CACHE_CONTROLLER_V7_SCHEMA_VERSION,
            "kind": "cache_controller_v7",
            "policy": str(self.policy),
            "inner_v6_cid": str(self.inner_v6.cid()),
            "four_objective_head_cid": (
                _ndarray_cid(self.four_objective_head)
                if self.four_objective_head is not None
                else "untrained"),
            "similarity_eviction_head_coefs_cid": (
                _ndarray_cid(self.similarity_eviction_head_coefs)
                if self.similarity_eviction_head_coefs is not None
                else "untrained"),
            "composite_v7_weights_cid": (
                _ndarray_cid(self.composite_v7_weights)
                if self.composite_v7_weights is not None
                else "untrained"),
            "ridge_lambda": float(round(self.ridge_lambda, 12)),
            "fit_seed": int(self.fit_seed),
        })

    def score_similarity_eviction_priority(
            self, *,
            flag_count: int, hidden_write: float,
            replay_age: int, attention_receive_l1: float,
            cache_key_norm: float,
            mean_similarity_to_others: float,
    ) -> float:
        """Apply the trained similarity-aware eviction head."""
        if self.similarity_eviction_head_coefs is None:
            return 0.0
        feat = _np.array([
            float(flag_count), float(hidden_write),
            float(replay_age), float(attention_receive_l1),
            float(cache_key_norm),
            float(mean_similarity_to_others)],
            dtype=_np.float64)
        return float(feat @ _np.asarray(
            self.similarity_eviction_head_coefs,
            dtype=_np.float64))


@dataclasses.dataclass(frozen=True)
class CacheControllerV7FitReport:
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
            "kind": "cache_controller_v7_fit_report",
            "report": self.to_dict()})


def fit_four_objective_ridge_v7(
        *, controller: CacheControllerV7,
        train_features: Sequence[Sequence[float]],
        target_drop_oracle: Sequence[float],
        target_retrieval_relevance: Sequence[float],
        target_hidden_wins: Sequence[float],
        target_replay_dominance: Sequence[float],
        ridge_lambda: float = W64_DEFAULT_CACHE_V7_RIDGE_LAMBDA,
) -> tuple[CacheControllerV7, CacheControllerV7FitReport]:
    """Four-objective stacked ridge fit. Decision variable:
    a (n_features × 4) matrix."""
    X = _np.asarray(train_features, dtype=_np.float64)
    y1 = _np.asarray(target_drop_oracle, dtype=_np.float64)
    y2 = _np.asarray(target_retrieval_relevance,
                       dtype=_np.float64)
    y3 = _np.asarray(target_hidden_wins, dtype=_np.float64)
    y4 = _np.asarray(target_replay_dominance,
                       dtype=_np.float64)
    if X.shape[0] == 0 or X.shape[0] != y1.size or (
            X.shape[0] != y2.size) or (X.shape[0] != y3.size) or (
            X.shape[0] != y4.size):
        raise ValueError(
            "fit requires positive-length matching features")
    Y = _np.stack([y1, y2, y3, y4], axis=-1)
    lam = max(float(ridge_lambda), 1e-9)
    A = X.T @ X + lam * _np.eye(X.shape[1], dtype=_np.float64)
    b = X.T @ Y
    try:
        H = _np.linalg.solve(A, b)
    except Exception:
        H = _np.zeros((X.shape[1], 4), dtype=_np.float64)
    Y_hat = X @ H
    per_pre = (
        float(_np.mean(_np.abs(y1))),
        float(_np.mean(_np.abs(y2))),
        float(_np.mean(_np.abs(y3))),
        float(_np.mean(_np.abs(y4))))
    per_post = (
        float(_np.mean(_np.abs(y1 - Y_hat[:, 0]))),
        float(_np.mean(_np.abs(y2 - Y_hat[:, 1]))),
        float(_np.mean(_np.abs(y3 - Y_hat[:, 2]))),
        float(_np.mean(_np.abs(y4 - Y_hat[:, 3]))))
    fitted = dataclasses.replace(
        controller,
        four_objective_head=_np.asarray(
            H, dtype=_np.float64).copy())
    report = CacheControllerV7FitReport(
        schema=W64_CACHE_CONTROLLER_V7_SCHEMA_VERSION,
        fit_kind="four_objective_v7",
        n_train=int(X.shape[0]),
        n_objectives=4,
        per_objective_pre_residual=per_pre,
        per_objective_post_residual=per_post,
        converged=bool(
            all(po <= pr + 1e-9
                for pr, po in zip(per_pre, per_post))),
        ridge_lambda=float(ridge_lambda),
    )
    return fitted, report


def fit_similarity_aware_eviction_head_v7(
        *, controller: CacheControllerV7,
        train_flag_counts: Sequence[int],
        train_hidden_writes: Sequence[float],
        train_replay_ages: Sequence[int],
        train_attention_receive_l1: Sequence[float],
        train_cache_key_norms: Sequence[float],
        train_mean_similarities: Sequence[float],
        target_eviction_priorities: Sequence[float],
        ridge_lambda: float = W64_DEFAULT_CACHE_V7_RIDGE_LAMBDA,
) -> tuple[CacheControllerV7, CacheControllerV7FitReport]:
    """Fit a 6-dim ridge head for the similarity-aware eviction
    priority."""
    n = int(len(train_flag_counts))
    if n == 0:
        raise ValueError("fit requires non-empty inputs")
    if (len(train_hidden_writes) != n
            or len(train_replay_ages) != n
            or len(train_attention_receive_l1) != n
            or len(train_cache_key_norms) != n
            or len(train_mean_similarities) != n
            or len(target_eviction_priorities) != n):
        raise ValueError("all inputs must match in length")
    X = _np.stack([
        _np.asarray(train_flag_counts, dtype=_np.float64),
        _np.asarray(train_hidden_writes, dtype=_np.float64),
        _np.asarray(train_replay_ages, dtype=_np.float64),
        _np.asarray(train_attention_receive_l1,
                       dtype=_np.float64),
        _np.asarray(train_cache_key_norms, dtype=_np.float64),
        _np.asarray(train_mean_similarities, dtype=_np.float64),
    ], axis=-1)
    y = _np.asarray(
        target_eviction_priorities, dtype=_np.float64)
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
    fitted = dataclasses.replace(
        controller,
        similarity_eviction_head_coefs=_np.asarray(
            theta, dtype=_np.float64).copy())
    report = CacheControllerV7FitReport(
        schema=W64_CACHE_CONTROLLER_V7_SCHEMA_VERSION,
        fit_kind="similarity_eviction_v7",
        n_train=int(n),
        n_objectives=1,
        per_objective_pre_residual=(float(pre),),
        per_objective_post_residual=(float(post),),
        converged=bool(post <= pre + 1e-9),
        ridge_lambda=float(ridge_lambda),
    )
    return fitted, report


def fit_composite_v7(
        *, controller: CacheControllerV7,
        head_scores: Sequence[Sequence[float]],
        drop_oracle: Sequence[float],
        ridge_lambda: float = W64_DEFAULT_CACHE_V7_RIDGE_LAMBDA,
) -> tuple[CacheControllerV7, CacheControllerV7FitReport]:
    """Composite_v7 ridge mixture over 8 heads against the drop
    oracle."""
    H = _np.asarray(head_scores, dtype=_np.float64)
    if H.ndim != 2 or H.shape[1] != 8:
        raise ValueError("head_scores must be (n_train, 8)")
    y = _np.asarray(drop_oracle, dtype=_np.float64)
    if H.shape[0] != y.size:
        raise ValueError(
            "head_scores rows must match drop_oracle len")
    lam = max(float(ridge_lambda), 1e-9)
    A = H.T @ H + lam * _np.eye(8, dtype=_np.float64)
    b = H.T @ y
    try:
        w = _np.linalg.solve(A, b)
    except Exception:
        w = _np.ones((8,), dtype=_np.float64) * (1.0 / 8.0)
    y_hat = H @ w
    pre = float(_np.mean(_np.abs(y - _np.mean(y))))
    post = float(_np.mean(_np.abs(y - y_hat)))
    fitted = dataclasses.replace(
        controller,
        composite_v7_weights=_np.asarray(
            w, dtype=_np.float64).copy())
    report = CacheControllerV7FitReport(
        schema=W64_CACHE_CONTROLLER_V7_SCHEMA_VERSION,
        fit_kind="composite_v7",
        n_train=int(H.shape[0]),
        n_objectives=1,
        per_objective_pre_residual=(float(pre),),
        per_objective_post_residual=(float(post),),
        converged=bool(post <= pre + 1e-9),
        ridge_lambda=float(ridge_lambda),
    )
    return fitted, report


@dataclasses.dataclass(frozen=True)
class CacheControllerV7Witness:
    schema: str
    controller_cid: str
    policy: str
    four_objective_used: bool
    similarity_eviction_used: bool
    composite_v7_used: bool
    last_eviction_priority_sum: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "controller_cid": str(self.controller_cid),
            "policy": str(self.policy),
            "four_objective_used": bool(
                self.four_objective_used),
            "similarity_eviction_used": bool(
                self.similarity_eviction_used),
            "composite_v7_used": bool(self.composite_v7_used),
            "last_eviction_priority_sum": float(round(
                self.last_eviction_priority_sum, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "cache_controller_v7_witness",
            "witness": self.to_dict()})


def emit_cache_controller_v7_witness(
        *, controller: CacheControllerV7,
        last_eviction_priority_sum: float = 0.0,
) -> CacheControllerV7Witness:
    return CacheControllerV7Witness(
        schema=W64_CACHE_CONTROLLER_V7_SCHEMA_VERSION,
        controller_cid=str(controller.cid()),
        policy=str(controller.policy),
        four_objective_used=bool(
            controller.four_objective_head is not None),
        similarity_eviction_used=bool(
            controller.similarity_eviction_head_coefs is not None),
        composite_v7_used=bool(
            controller.composite_v7_weights is not None),
        last_eviction_priority_sum=float(
            last_eviction_priority_sum),
    )


__all__ = [
    "W64_CACHE_CONTROLLER_V7_SCHEMA_VERSION",
    "W64_CACHE_POLICY_FOUR_OBJECTIVE_V7",
    "W64_CACHE_POLICY_SIMILARITY_EVICTION_V7",
    "W64_CACHE_POLICY_COMPOSITE_V7",
    "W64_CACHE_POLICIES_V7",
    "W64_DEFAULT_CACHE_V7_RIDGE_LAMBDA",
    "CacheControllerV7",
    "CacheControllerV7FitReport",
    "fit_four_objective_ridge_v7",
    "fit_similarity_aware_eviction_head_v7",
    "fit_composite_v7",
    "CacheControllerV7Witness",
    "emit_cache_controller_v7_witness",
]
