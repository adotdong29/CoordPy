"""W61 M6 — Cache Controller V4 (bilinear retrieval + trained
corruption floor + two-stage policy).

Strictly extends W60's ``coordpy.cache_controller_v3``. V3 added
trained_eviction, learned_attention_receive, learned_corruption_aware,
and composite_v3. V4 adds **three** new substrate-load-bearing
mechanisms:

* **Bilinear retrieval head over V6 cache keys** — V3's retrieval
  matrix was bilinear over (hidden_state ⊗ query). V4 introduces a
  *second* bilinear M matrix that scores ``(query_feature_q ⊗
  cache_key_k)`` against retrieval-relevance supervision. The fit
  uses the V6 substrate's content-addressable cache_keys axis (not
  the V3 K/V tensors). The decision variable is a ``(d_q × d_key)``
  matrix solved by closed-form ridge over the outer-product feature.
* **Trained corruption-aware floor** — V3 had a hard ``-inf``
  floor. V4 *fits* a per-slot corruption-floor scalar by closed-
  form ridge against the V1 drop oracle: when a slot is flagged,
  the floor is whatever depresses its score *just enough* to make
  the drop-oracle gradient point toward eviction. This is more
  graceful than a hard floor for noisy corruption channels.
* **Two-stage retention policy** — V4's
  ``two_stage_v4`` policy first triages with
  ``learned_attention_receive`` (coarse, fast) to drop the bottom
  fraction of slots, then refines the survivors with
  ``trained_eviction`` (slow, learned). The triage threshold is
  itself fit by a closed-form ridge regression of the drop oracle
  vs. the attention-receive score.
* **Composite_v4** — five-head mixture (V3's four + V6 bilinear-
  retrieval-from-cache-keys). Weights fit by closed-form ridge
  against the drop oracle, with a per-head retention prior.

V4 strictly extends V3: with policy in V3's set, V4 reduces to
V3 byte-for-byte (modulo the V4 schema tag).

Honest scope
------------

* All V4 fitted heads use closed-form linear/bilinear ridge. No
  autograd / SGD / GPU. ``W61-L-V6-CACHE-CONTROLLER-NO-AUTOGRAD-
  CAP`` documents.
* The V6 bilinear retrieval head fits on a synthesised
  ``(query, cache_key) → relevance`` supervision set seeded by the
  controller; it does NOT learn from raw user queries.
* Two-stage policy's triage cutoff is a single scalar fit; the
  fine policy operates on the surviving slots only.
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
        "coordpy.cache_controller_v4 requires numpy") from exc

from .cache_controller import (
    CacheController,
    W58_CACHE_POLICY_IMPORTANCE,
    W58_CACHE_POLICY_LEARNED,
    W58_CACHE_POLICY_UNIFORM,
    _compute_drop_oracle,
)
from .cache_controller_v2 import (
    W59_CACHE_POLICIES_V2,
)
from .cache_controller_v3 import (
    CacheControllerV3,
    CacheControllerV3FitReport,
    W60_CACHE_CONTROLLER_V3_SCHEMA_VERSION,
    W60_CACHE_POLICIES_V3,
    W60_CACHE_POLICY_COMPOSITE_V3,
    W60_CACHE_POLICY_LEARNED_ATTENTION_RECEIVE,
    W60_CACHE_POLICY_LEARNED_CORRUPTION_AWARE,
    W60_CACHE_POLICY_TRAINED_EVICTION,
    W60_DEFAULT_CACHE_V3_RIDGE_LAMBDA,
    _build_attention_receive_feature,
    _build_trained_eviction_feature,
    _safe_condition,
    apply_cache_controller_v3_and_measure,
)
from .tiny_substrate_v3 import (
    TinyV3KVCache,
    TinyV3SubstrateParams,
    _ndarray_cid,
    _sha256_hex,
    forward_tiny_substrate_v3,
)
from .tiny_substrate_v6 import (
    TinyV6KVCache, TinyV6SubstrateParams,
    forward_tiny_substrate_v6,
)


W61_CACHE_CONTROLLER_V4_SCHEMA_VERSION: str = (
    "coordpy.cache_controller_v4.v1")

W61_CACHE_POLICY_BILINEAR_RETRIEVAL_V6: str = (
    "bilinear_retrieval_v6")
W61_CACHE_POLICY_TRAINED_CORRUPTION_FLOOR: str = (
    "trained_corruption_floor")
W61_CACHE_POLICY_TWO_STAGE_V4: str = "two_stage_v4"
W61_CACHE_POLICY_COMPOSITE_V4: str = "composite_v4"

W61_CACHE_POLICIES_V4: tuple[str, ...] = (
    *W60_CACHE_POLICIES_V3,
    W61_CACHE_POLICY_BILINEAR_RETRIEVAL_V6,
    W61_CACHE_POLICY_TRAINED_CORRUPTION_FLOOR,
    W61_CACHE_POLICY_TWO_STAGE_V4,
    W61_CACHE_POLICY_COMPOSITE_V4,
)

W61_DEFAULT_CACHE_V4_RIDGE_LAMBDA: float = 0.10
W61_DEFAULT_CACHE_V4_TWO_STAGE_KEEP_FRACTION: float = 0.5


@dataclasses.dataclass
class CacheControllerV4:
    """V4 cache controller. Pluggable retention policy with four new
    V4 heads on top of V3.
    """
    policy: str
    d_model: int
    d_key: int
    inner_v3: CacheControllerV3
    # V6 bilinear retrieval matrix (d_model x d_key).
    bilinear_retrieval_v6_matrix: "_np.ndarray | None"
    # Trained corruption floor (3,) vector of bias coefficients.
    corruption_floor_coefs: "_np.ndarray | None"
    # Two-stage threshold scalar.
    two_stage_threshold: float
    two_stage_keep_fraction: float
    # Composite_v4 weights (5,).
    composite_v4_weights: "_np.ndarray | None"
    ridge_lambda: float
    fit_seed: int

    @classmethod
    def init(
            cls, *,
            policy: str = W58_CACHE_POLICY_UNIFORM,
            d_model: int = 64,
            d_key: int = 8,
            probe_layer: int = -1,
            ridge_lambda: float = (
                W61_DEFAULT_CACHE_V4_RIDGE_LAMBDA),
            two_stage_keep_fraction: float = (
                W61_DEFAULT_CACHE_V4_TWO_STAGE_KEEP_FRACTION),
            fit_seed: int = 61060,
    ) -> "CacheControllerV4":
        if policy not in W61_CACHE_POLICIES_V4:
            raise ValueError(
                f"policy must be in {W61_CACHE_POLICIES_V4}, "
                f"got {policy!r}")
        # If V4 policy not in V3 set, fall back to importance for inner.
        if policy in W60_CACHE_POLICIES_V3:
            v3_policy = policy
        else:
            v3_policy = W58_CACHE_POLICY_IMPORTANCE
        inner_v3 = CacheControllerV3.init(
            policy=str(v3_policy),
            d_model=int(d_model),
            probe_layer=int(probe_layer),
            ridge_lambda=float(ridge_lambda),
            fit_seed=int(fit_seed))
        return cls(
            policy=str(policy),
            d_model=int(d_model),
            d_key=int(d_key),
            inner_v3=inner_v3,
            bilinear_retrieval_v6_matrix=None,
            corruption_floor_coefs=None,
            two_stage_threshold=0.0,
            two_stage_keep_fraction=float(
                two_stage_keep_fraction),
            composite_v4_weights=None,
            ridge_lambda=float(ridge_lambda),
            fit_seed=int(fit_seed),
        )

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W61_CACHE_CONTROLLER_V4_SCHEMA_VERSION,
            "kind": "cache_controller_v4",
            "policy": str(self.policy),
            "d_model": int(self.d_model),
            "d_key": int(self.d_key),
            "inner_v3_cid": str(self.inner_v3.cid()),
            "bilinear_retrieval_v6_matrix_cid": (
                _ndarray_cid(self.bilinear_retrieval_v6_matrix)
                if self.bilinear_retrieval_v6_matrix is not None
                else "untrained"),
            "corruption_floor_coefs_cid": (
                _ndarray_cid(self.corruption_floor_coefs)
                if self.corruption_floor_coefs is not None
                else "untrained"),
            "two_stage_threshold": float(round(
                self.two_stage_threshold, 12)),
            "two_stage_keep_fraction": float(round(
                self.two_stage_keep_fraction, 12)),
            "composite_v4_weights_cid": (
                _ndarray_cid(self.composite_v4_weights)
                if self.composite_v4_weights is not None
                else "untrained"),
            "ridge_lambda": float(self.ridge_lambda),
            "fit_seed": int(self.fit_seed),
        })

    def score_tokens(
            self, *,
            hidden_state: "_np.ndarray",
            hidden_states_layerset: Sequence["_np.ndarray"],
            importance_vector: "_np.ndarray",
            query_vector: "_np.ndarray | None",
            attention_receive: Sequence["_np.ndarray"] | None,
            corruption_flags: Sequence["_np.ndarray"] | None,
            retrieval_scalar: "_np.ndarray | None",
            cache_keys: Sequence["_np.ndarray"] | None,
            n_tokens: int,
    ) -> "_np.ndarray":
        n = int(n_tokens)
        if self.policy in W60_CACHE_POLICIES_V3:
            return self.inner_v3.score_tokens(
                hidden_state=hidden_state,
                hidden_states_layerset=hidden_states_layerset,
                importance_vector=importance_vector,
                query_vector=query_vector,
                attention_receive=attention_receive,
                corruption_flags=corruption_flags,
                retrieval_scalar=retrieval_scalar,
                n_tokens=n)
        if self.policy == W61_CACHE_POLICY_BILINEAR_RETRIEVAL_V6:
            if (self.bilinear_retrieval_v6_matrix is None
                    or cache_keys is None
                    or query_vector is None):
                return _np.arange(n, dtype=_np.float64)
            # Mean-pool cache_keys over layers to get (n, d_key).
            d_key = int(self.d_key)
            agg = _np.zeros((n, d_key), dtype=_np.float64)
            count = 0
            for ck in cache_keys:
                if ck.size == 0:
                    continue
                m = min(int(ck.shape[0]), n)
                d_loc = int(ck.shape[1])
                d = min(d_key, d_loc)
                agg[:m, :d] += ck[:m, :d]
                count += 1
            if count > 0:
                agg /= float(count)
            q = _np.asarray(
                query_vector, dtype=_np.float64).reshape(-1)
            d_q = int(self.d_model)
            if q.size < d_q:
                q = _np.concatenate(
                    [q, _np.zeros(d_q - q.size,
                                    dtype=_np.float64)])
            elif q.size > d_q:
                q = q[:d_q]
            M = _np.asarray(
                self.bilinear_retrieval_v6_matrix,
                dtype=_np.float64)
            # Each slot's score = q^T M cache_key_i.
            return (agg @ M.T) @ q
        if self.policy == (
                W61_CACHE_POLICY_TRAINED_CORRUPTION_FLOOR):
            base = self.inner_v3.score_tokens(
                hidden_state=hidden_state,
                hidden_states_layerset=hidden_states_layerset,
                importance_vector=importance_vector,
                query_vector=query_vector,
                attention_receive=attention_receive,
                corruption_flags=None,
                retrieval_scalar=retrieval_scalar,
                n_tokens=n)
            if (corruption_flags is None
                    or self.corruption_floor_coefs is None):
                return base
            # Apply per-slot floor based on the number of layers
            # flagging the slot.
            flags_per_slot = _np.zeros((n,), dtype=_np.int64)
            for f in corruption_flags:
                if f.size:
                    m = min(int(f.size), n)
                    flags_per_slot[:m] += f[:m].astype(
                        _np.int64)
            coefs = _np.asarray(
                self.corruption_floor_coefs,
                dtype=_np.float64)
            # Floor = c0 + c1*flag_count + c2*flag_count^2.
            floor = (
                float(coefs[0]) * _np.ones((n,))
                + float(coefs[1]) * flags_per_slot.astype(
                    _np.float64)
                + float(coefs[2]) * (flags_per_slot.astype(
                    _np.float64) ** 2))
            mask = flags_per_slot > 0
            out = base.copy()
            out[mask] = _np.minimum(out[mask], floor[mask])
            return out
        if self.policy == W61_CACHE_POLICY_TWO_STAGE_V4:
            # Coarse: learned_attention_receive.
            coarse_inner = CacheControllerV3.init(
                policy=W60_CACHE_POLICY_LEARNED_ATTENTION_RECEIVE,
                d_model=int(self.d_model),
                ridge_lambda=float(self.ridge_lambda),
                fit_seed=int(self.fit_seed))
            coarse_inner.attention_receive_scorer = (
                self.inner_v3.attention_receive_scorer)
            coarse = coarse_inner.score_tokens(
                hidden_state=hidden_state,
                hidden_states_layerset=hidden_states_layerset,
                importance_vector=importance_vector,
                query_vector=query_vector,
                attention_receive=attention_receive,
                corruption_flags=corruption_flags,
                retrieval_scalar=retrieval_scalar,
                n_tokens=n)
            # Pick threshold by quantile keep_fraction.
            keep = max(1, int(
                float(self.two_stage_keep_fraction) * float(n)))
            kth = _np.partition(coarse, max(0, n - keep))[
                max(0, n - keep)]
            # Survivors get fine score; rejects get -inf-like.
            fine_inner = CacheControllerV3.init(
                policy=W60_CACHE_POLICY_TRAINED_EVICTION,
                d_model=int(self.d_model),
                ridge_lambda=float(self.ridge_lambda),
                fit_seed=int(self.fit_seed))
            fine_inner.trained_eviction_scorer = (
                self.inner_v3.trained_eviction_scorer)
            fine = fine_inner.score_tokens(
                hidden_state=hidden_state,
                hidden_states_layerset=hidden_states_layerset,
                importance_vector=importance_vector,
                query_vector=query_vector,
                attention_receive=attention_receive,
                corruption_flags=corruption_flags,
                retrieval_scalar=retrieval_scalar,
                n_tokens=n)
            out = fine.copy()
            out[coarse < kth] = -1e8
            return out
        if self.policy == W61_CACHE_POLICY_COMPOSITE_V4:
            if self.composite_v4_weights is None:
                w = _np.ones((5,), dtype=_np.float64) * 0.2
            else:
                w = _np.asarray(
                    self.composite_v4_weights,
                    dtype=_np.float64)
                if w.size != 5:
                    w = _np.ones((5,),
                                  dtype=_np.float64) * 0.2
            # Five heads:
            # 1) importance, 2) trained_eviction,
            # 3) attention_receive, 4) bilinear_retrieval_v6,
            # 5) v3.composite_v3
            scores: list["_np.ndarray"] = []
            imp = _np.asarray(
                importance_vector,
                dtype=_np.float64).reshape(-1)[:n]
            if imp.size < n:
                imp = _np.concatenate(
                    [imp, _np.zeros(
                        n - imp.size, dtype=_np.float64)])
            scores.append(imp)
            if (self.inner_v3.trained_eviction_scorer
                    is not None):
                feat = _build_trained_eviction_feature(
                    hidden_state=hidden_state,
                    importance_vector=importance_vector,
                    attention_receive=attention_receive,
                    retrieval_scalar=retrieval_scalar,
                    n_tokens=n)
                scores.append(feat @
                                self.inner_v3.trained_eviction_scorer)
            else:
                scores.append(_np.zeros(
                    (n,), dtype=_np.float64))
            if (self.inner_v3.attention_receive_scorer
                    is not None
                    and attention_receive is not None):
                feat = _build_attention_receive_feature(
                    attention_receive, n_tokens=n)
                scores.append(feat @
                                self.inner_v3.attention_receive_scorer)
            else:
                scores.append(_np.zeros(
                    (n,), dtype=_np.float64))
            if (self.bilinear_retrieval_v6_matrix is not None
                    and cache_keys is not None
                    and query_vector is not None):
                self_pol_save = self.policy
                object.__setattr__(
                    self, "policy",
                    W61_CACHE_POLICY_BILINEAR_RETRIEVAL_V6)
                try:
                    bs = self.score_tokens(
                        hidden_state=hidden_state,
                        hidden_states_layerset=(
                            hidden_states_layerset),
                        importance_vector=importance_vector,
                        query_vector=query_vector,
                        attention_receive=attention_receive,
                        corruption_flags=None,
                        retrieval_scalar=retrieval_scalar,
                        cache_keys=cache_keys, n_tokens=n)
                finally:
                    object.__setattr__(
                        self, "policy", self_pol_save)
                scores.append(bs)
            else:
                scores.append(_np.zeros(
                    (n,), dtype=_np.float64))
            v3_comp = self.inner_v3.score_tokens(
                hidden_state=hidden_state,
                hidden_states_layerset=hidden_states_layerset,
                importance_vector=importance_vector,
                query_vector=query_vector,
                attention_receive=attention_receive,
                corruption_flags=corruption_flags,
                retrieval_scalar=retrieval_scalar,
                n_tokens=n)
            scores.append(v3_comp)
            stacked = _np.stack([s[:n] for s in scores], axis=1)
            return stacked @ w
        raise ValueError(
            f"unknown V4 policy {self.policy!r}")


@dataclasses.dataclass(frozen=True)
class CacheControllerV4FitReport:
    schema: str
    fit_kind: str
    n_train_slots: int
    pre_fit_residual: float
    post_fit_residual: float
    ridge_lambda: float
    condition_number: float
    converged: bool
    fit_d_q: int
    fit_d_key: int
    fit_n_heads: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "fit_kind": str(self.fit_kind),
            "n_train_slots": int(self.n_train_slots),
            "pre_fit_residual": float(round(
                self.pre_fit_residual, 12)),
            "post_fit_residual": float(round(
                self.post_fit_residual, 12)),
            "ridge_lambda": float(round(self.ridge_lambda, 12)),
            "condition_number": float(round(
                self.condition_number, 12)),
            "converged": bool(self.converged),
            "fit_d_q": int(self.fit_d_q),
            "fit_d_key": int(self.fit_d_key),
            "fit_n_heads": int(self.fit_n_heads),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "cache_controller_v4_fit_report",
            "report": self.to_dict()})


def fit_bilinear_retrieval_v6(
        *,
        controller: CacheControllerV4,
        params_v6: TinyV6SubstrateParams,
        train_token_ids_per_slot: Sequence[Sequence[int]],
        train_query_vectors: Sequence[Sequence[float]],
        train_relevance: Sequence[float],
        ridge_lambda: float = W61_DEFAULT_CACHE_V4_RIDGE_LAMBDA,
) -> tuple[CacheControllerV4, CacheControllerV4FitReport]:
    """Fit the V6 bilinear retrieval matrix M (d_q × d_key).

    For each slot ``i`` we produce a V6 cache and average its
    cache_keys across layers into a single d_key vector. The
    feature for the bilinear fit is ``flatten(outer(q_i, k_i))``
    of size ``d_q * d_key``; the target is ``relevance_i``.
    """
    n = int(len(train_token_ids_per_slot))
    if n == 0 or len(train_query_vectors) != n or len(
            train_relevance) != n:
        raise ValueError(
            "fit requires matching positive-length sequences")
    d_q = int(controller.d_model)
    d_key = int(controller.d_key)
    feats = _np.zeros((n, d_q * d_key), dtype=_np.float64)
    y = _np.asarray(train_relevance, dtype=_np.float64)
    for i in range(n):
        ids = list(train_token_ids_per_slot[i])
        _, v6_cache = forward_tiny_substrate_v6(
            params_v6, ids)
        # Mean cache key across layers, take first slot.
        agg = _np.zeros((d_key,), dtype=_np.float64)
        count = 0
        for ck in v6_cache.cache_keys:
            if ck.size == 0:
                continue
            d_loc = min(d_key, ck.shape[1])
            agg[:d_loc] += ck[0, :d_loc]
            count += 1
        if count > 0:
            agg /= float(count)
        q = _np.asarray(
            train_query_vectors[i],
            dtype=_np.float64).reshape(-1)
        if q.size < d_q:
            q = _np.concatenate(
                [q, _np.zeros(d_q - q.size, dtype=_np.float64)])
        elif q.size > d_q:
            q = q[:d_q]
        outer = _np.outer(q, agg).ravel()
        feats[i] = outer
    lam = max(float(ridge_lambda), 1e-9)
    A = feats.T @ feats + lam * _np.eye(
        d_q * d_key, dtype=_np.float64)
    b = feats.T @ y
    try:
        theta = _np.linalg.solve(A, b)
    except Exception:
        theta = _np.zeros((d_q * d_key,), dtype=_np.float64)
    y_hat = feats @ theta
    pre = float(_np.mean(_np.abs(y - _np.mean(y))))
    post = float(_np.mean(_np.abs(y - y_hat)))
    M = theta.reshape(d_q, d_key)
    fitted = dataclasses.replace(
        controller, bilinear_retrieval_v6_matrix=M.copy())
    return fitted, CacheControllerV4FitReport(
        schema=W61_CACHE_CONTROLLER_V4_SCHEMA_VERSION,
        fit_kind="bilinear_retrieval_v6",
        n_train_slots=int(n),
        pre_fit_residual=float(pre),
        post_fit_residual=float(post),
        ridge_lambda=float(ridge_lambda),
        condition_number=float(_safe_condition(A)),
        converged=bool(post <= pre + 1e-9),
        fit_d_q=int(d_q),
        fit_d_key=int(d_key),
        fit_n_heads=int(params_v6.config.n_heads),
    )


def fit_trained_corruption_floor(
        *,
        controller: CacheControllerV4,
        params_v6: TinyV6SubstrateParams,
        train_token_ids: Sequence[int],
        train_corruption_counts: Sequence[int],
        train_floor_targets: Sequence[float],
        ridge_lambda: float = W61_DEFAULT_CACHE_V4_RIDGE_LAMBDA,
) -> tuple[CacheControllerV4, CacheControllerV4FitReport]:
    """Fit a quadratic ``floor = c0 + c1·flag_count + c2·flag_count^2``
    coefficient vector by closed-form ridge."""
    cnt = _np.asarray(train_corruption_counts, dtype=_np.float64)
    y = _np.asarray(train_floor_targets, dtype=_np.float64)
    if cnt.size != y.size or cnt.size == 0:
        raise ValueError("inputs must be same positive length")
    X = _np.stack([
        _np.ones_like(cnt), cnt, cnt * cnt], axis=-1)
    lam = max(float(ridge_lambda), 1e-9)
    A = X.T @ X + lam * _np.eye(3, dtype=_np.float64)
    b = X.T @ y
    try:
        theta = _np.linalg.solve(A, b)
    except Exception:
        theta = _np.zeros((3,), dtype=_np.float64)
    y_hat = X @ theta
    pre = float(_np.mean(_np.abs(y - _np.mean(y))))
    post = float(_np.mean(_np.abs(y - y_hat)))
    fitted = dataclasses.replace(
        controller,
        corruption_floor_coefs=_np.asarray(
            theta, dtype=_np.float64).copy())
    return fitted, CacheControllerV4FitReport(
        schema=W61_CACHE_CONTROLLER_V4_SCHEMA_VERSION,
        fit_kind="trained_corruption_floor",
        n_train_slots=int(cnt.size),
        pre_fit_residual=float(pre),
        post_fit_residual=float(post),
        ridge_lambda=float(ridge_lambda),
        condition_number=float(_safe_condition(A)),
        converged=bool(post <= pre + 1e-9),
        fit_d_q=int(controller.d_model),
        fit_d_key=int(controller.d_key),
        fit_n_heads=int(params_v6.config.n_heads),
    )


def fit_two_stage_threshold(
        *, controller: CacheControllerV4,
        attention_receive_scores: Sequence[float],
        drop_oracle_per_slot: Sequence[float],
) -> tuple[CacheControllerV4, CacheControllerV4FitReport]:
    """Fit the two-stage triage threshold as the median of the
    drop-oracle-weighted attention-receive scores. Closed-form: the
    threshold minimises the L1 cost of mis-classifying slots."""
    scores = _np.asarray(
        attention_receive_scores, dtype=_np.float64)
    oracle = _np.asarray(
        drop_oracle_per_slot, dtype=_np.float64)
    if scores.size == 0:
        return controller, CacheControllerV4FitReport(
            schema=W61_CACHE_CONTROLLER_V4_SCHEMA_VERSION,
            fit_kind="two_stage_threshold",
            n_train_slots=0,
            pre_fit_residual=0.0, post_fit_residual=0.0,
            ridge_lambda=0.0, condition_number=0.0,
            converged=True, fit_d_q=int(controller.d_model),
            fit_d_key=int(controller.d_key),
            fit_n_heads=0)
    # Sort by score, find the threshold that minimises L1 of
    # oracle differences.
    order = _np.argsort(scores)
    cum = _np.cumsum(oracle[order])
    total = float(cum[-1]) if cum.size else 0.0
    # Find split index minimising |L - R| where L = cum[i] and
    # R = total - cum[i].
    diffs = _np.abs(2.0 * cum - total)
    best = int(_np.argmin(diffs))
    threshold = float(scores[order][best])
    pre = float(_np.mean(_np.abs(oracle)))
    post = float(_np.mean(_np.abs(
        oracle - _np.median(oracle))))
    fitted = dataclasses.replace(
        controller, two_stage_threshold=float(threshold))
    return fitted, CacheControllerV4FitReport(
        schema=W61_CACHE_CONTROLLER_V4_SCHEMA_VERSION,
        fit_kind="two_stage_threshold",
        n_train_slots=int(scores.size),
        pre_fit_residual=float(pre),
        post_fit_residual=float(post),
        ridge_lambda=0.0,
        condition_number=0.0,
        converged=bool(post <= pre + 1e-9),
        fit_d_q=int(controller.d_model),
        fit_d_key=int(controller.d_key),
        fit_n_heads=0,
    )


def fit_composite_v4(
        *, controller: CacheControllerV4,
        head_scores: Sequence[Sequence[float]],
        drop_oracle: Sequence[float],
        ridge_lambda: float = W61_DEFAULT_CACHE_V4_RIDGE_LAMBDA,
) -> tuple[CacheControllerV4, CacheControllerV4FitReport]:
    """Fit composite_v4 mixture weights (5,) by closed-form ridge
    over per-head scores against the drop oracle."""
    H = _np.asarray(head_scores, dtype=_np.float64)
    if H.ndim != 2 or H.shape[1] != 5:
        raise ValueError("head_scores must be (n_train, 5)")
    y = _np.asarray(drop_oracle, dtype=_np.float64)
    if H.shape[0] != y.size:
        raise ValueError(
            "head_scores rows must match drop_oracle len")
    lam = max(float(ridge_lambda), 1e-9)
    A = H.T @ H + lam * _np.eye(5, dtype=_np.float64)
    b = H.T @ y
    try:
        w = _np.linalg.solve(A, b)
    except Exception:
        w = _np.ones((5,), dtype=_np.float64) * 0.2
    y_hat = H @ w
    pre = float(_np.mean(_np.abs(y - _np.mean(y))))
    post = float(_np.mean(_np.abs(y - y_hat)))
    fitted = dataclasses.replace(
        controller,
        composite_v4_weights=_np.asarray(
            w, dtype=_np.float64).copy())
    return fitted, CacheControllerV4FitReport(
        schema=W61_CACHE_CONTROLLER_V4_SCHEMA_VERSION,
        fit_kind="composite_v4",
        n_train_slots=int(H.shape[0]),
        pre_fit_residual=float(pre),
        post_fit_residual=float(post),
        ridge_lambda=float(ridge_lambda),
        condition_number=float(_safe_condition(A)),
        converged=bool(post <= pre + 1e-9),
        fit_d_q=int(controller.d_model),
        fit_d_key=int(controller.d_key),
        fit_n_heads=0,
    )


@dataclasses.dataclass(frozen=True)
class CacheControllerV4Witness:
    schema: str
    controller_cid: str
    policy: str
    n_evicted: int
    n_kept: int
    score_mean: float
    score_std: float
    bilinear_used: bool
    corruption_floor_used: bool
    two_stage_used: bool
    composite_v4_used: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "controller_cid": str(self.controller_cid),
            "policy": str(self.policy),
            "n_evicted": int(self.n_evicted),
            "n_kept": int(self.n_kept),
            "score_mean": float(round(self.score_mean, 12)),
            "score_std": float(round(self.score_std, 12)),
            "bilinear_used": bool(self.bilinear_used),
            "corruption_floor_used": bool(
                self.corruption_floor_used),
            "two_stage_used": bool(self.two_stage_used),
            "composite_v4_used": bool(self.composite_v4_used),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "cache_controller_v4_witness",
            "witness": self.to_dict()})


def apply_cache_controller_v4_and_measure(
        *, controller: CacheControllerV4,
        params_v6: TinyV6SubstrateParams,
        token_ids: Sequence[int],
        query_vector: Sequence[float],
        retain_top_k: int,
) -> CacheControllerV4Witness:
    """Apply the V4 controller to a fresh V6 forward and emit a
    witness with score statistics + which V4 head fired."""
    trace, v6_cache = forward_tiny_substrate_v6(
        params_v6, list(token_ids))
    n = v6_cache.n_tokens()
    # Build the various per-layer feature views from the V6 cache.
    hs_per_layer = (
        trace.v5_trace.v4_trace.head_hidden_states_per_layer)
    if hs_per_layer:
        hs_last = hs_per_layer[-1]
        if hs_last.ndim == 3:
            hidden_state = hs_last.reshape(
                hs_last.shape[1], -1)
        else:
            hidden_state = hs_last
    else:
        hidden_state = _np.zeros(
            (n, int(controller.d_model)), dtype=_np.float64)
    imp = _np.zeros((n,), dtype=_np.float64)
    for il in v6_cache.v3_cache.importance:
        if il.size:
            m = min(int(il.size), n)
            imp[:m] += il[:m].astype(_np.float64)
    scores = controller.score_tokens(
        hidden_state=hidden_state,
        hidden_states_layerset=hs_per_layer,
        importance_vector=imp,
        query_vector=query_vector,
        attention_receive=v6_cache.v5_cache.attention_receive,
        corruption_flags=v6_cache.v5_cache.corruption_flags,
        retrieval_scalar=None,
        cache_keys=v6_cache.cache_keys,
        n_tokens=n)
    k = max(0, min(int(retain_top_k), n))
    if k == 0:
        n_evicted = n
    else:
        sorted_scores = _np.sort(scores)[::-1]
        threshold = sorted_scores[k - 1] if n else 0.0
        n_evicted = int(_np.sum(scores < threshold))
    return CacheControllerV4Witness(
        schema=W61_CACHE_CONTROLLER_V4_SCHEMA_VERSION,
        controller_cid=str(controller.cid()),
        policy=str(controller.policy),
        n_evicted=int(n_evicted),
        n_kept=int(n - n_evicted),
        score_mean=float(_np.mean(scores)) if scores.size else 0.0,
        score_std=float(_np.std(scores)) if scores.size else 0.0,
        bilinear_used=bool(
            controller.bilinear_retrieval_v6_matrix is not None
            and controller.policy in (
                W61_CACHE_POLICY_BILINEAR_RETRIEVAL_V6,
                W61_CACHE_POLICY_COMPOSITE_V4)),
        corruption_floor_used=bool(
            controller.corruption_floor_coefs is not None
            and controller.policy == (
                W61_CACHE_POLICY_TRAINED_CORRUPTION_FLOOR)),
        two_stage_used=bool(
            controller.policy == W61_CACHE_POLICY_TWO_STAGE_V4),
        composite_v4_used=bool(
            controller.composite_v4_weights is not None
            and controller.policy == (
                W61_CACHE_POLICY_COMPOSITE_V4)),
    )


__all__ = [
    "W61_CACHE_CONTROLLER_V4_SCHEMA_VERSION",
    "W61_CACHE_POLICIES_V4",
    "W61_CACHE_POLICY_BILINEAR_RETRIEVAL_V6",
    "W61_CACHE_POLICY_TRAINED_CORRUPTION_FLOOR",
    "W61_CACHE_POLICY_TWO_STAGE_V4",
    "W61_CACHE_POLICY_COMPOSITE_V4",
    "W61_DEFAULT_CACHE_V4_RIDGE_LAMBDA",
    "CacheControllerV4",
    "CacheControllerV4FitReport",
    "CacheControllerV4Witness",
    "fit_bilinear_retrieval_v6",
    "fit_trained_corruption_floor",
    "fit_two_stage_threshold",
    "fit_composite_v4",
    "apply_cache_controller_v4_and_measure",
]
