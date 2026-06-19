"""W60 M6 — Cache Controller V3 (trainable retention/retrieval/eviction).

Strictly extends W59's ``coordpy.cache_controller_v2``. V2 added two
new closed-form ridge heads (``learned_hidden`` cross-layer scorer
and ``learned_retrieval`` bilinear matrix) on top of V1's three
policies. V3 adds **three** new policies and **three** new fitted
heads:

* ``learned_attention_receive`` — closed-form ridge head whose
  feature is the per-(layer, head) cumulative attention-receive
  vector that V5's ``TinyV5KVCache.attention_receive`` records.
  Decision variable: ``s = W_a · stack(receive_per_head)``. Fit
  by closed-form ridge against the V1 leave-one-out drop oracle.
* ``learned_corruption_aware`` — V2 + a hard ``corruption_mask``
  multiplier. When V5's corruption flag is set on a slot, the
  controller's score for that slot is forced to ``-inf`` —
  effectively guaranteed eviction. The fitted V2 head still
  scores the surviving slots.
* ``trained_eviction`` — V3's eviction policy is *fitted* by
  closed-form ridge against the *cumulative downstream logit
  drift* oracle: for each slot, drop it on a held-out forward
  and observe the L2 logit drift that drop induces. The trained
  scorer maps a slot's (hidden_state, importance, attention-
  receive, retrieval-score) feature to that drift estimate.

Plus:

* **Composite policy** — ``composite_v3`` mixes the four V3 heads
  (importance, learned_hidden, learned_retrieval,
  learned_attention_receive) under per-policy weights ``w_imp,
  w_h, w_r, w_a``. The mixture weights are themselves fit by
  closed-form ridge against the drop oracle.

V3 strictly extends V2: with policy ∈ V2's set, V3 reduces to V2
byte-for-byte.

Honest scope
------------

* All four V3 fitted heads use closed-form linear regression.
  ``W60-L-V5-NO-AUTOGRAD-CAP`` carries forward.
* "Trained eviction" is a *single linear ridge head*, not a deep
  network and not autograd. The W60 R-125 H-bar publishes the
  pre/post-fit residual ratio.
* Corruption-aware policy depends on an external CRC V8 detector
  to set the V5 cache's corruption flags; the controller just
  reads them.
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
        "coordpy.cache_controller_v3 requires numpy") from exc

from .cache_controller import (
    CacheController,
    W58_CACHE_POLICY_IMPORTANCE,
    W58_CACHE_POLICY_LEARNED,
    W58_CACHE_POLICY_UNIFORM,
    _compute_drop_oracle,
)
from .cache_controller_v2 import (
    CacheControllerV2,
    CacheControllerV2FitReport,
    W59_CACHE_CONTROLLER_V2_SCHEMA_VERSION,
    W59_CACHE_POLICIES_V2,
    W59_CACHE_POLICY_LEARNED_HIDDEN,
    W59_CACHE_POLICY_LEARNED_RETRIEVAL,
    W59_DEFAULT_CACHE_V2_RIDGE_LAMBDA,
    W59_DEFAULT_CACHE_V2_PROBE_LAYERS,
    _build_layerset_feature,
    apply_cache_controller_v2_and_measure,
)
from .tiny_substrate_v3 import (
    TinyV3KVCache,
    TinyV3SubstrateParams,
    _ndarray_cid,
    _sha256_hex,
    forward_tiny_substrate_v3,
)
from .tiny_substrate_v5 import (
    TinyV5KVCache,
)


W60_CACHE_CONTROLLER_V3_SCHEMA_VERSION: str = (
    "coordpy.cache_controller_v3.v1")

W60_CACHE_POLICY_LEARNED_ATTENTION_RECEIVE: str = (
    "learned_attention_receive")
W60_CACHE_POLICY_LEARNED_CORRUPTION_AWARE: str = (
    "learned_corruption_aware")
W60_CACHE_POLICY_TRAINED_EVICTION: str = (
    "trained_eviction")
W60_CACHE_POLICY_COMPOSITE_V3: str = "composite_v3"

W60_CACHE_POLICIES_V3: tuple[str, ...] = (
    *W59_CACHE_POLICIES_V2,
    W60_CACHE_POLICY_LEARNED_ATTENTION_RECEIVE,
    W60_CACHE_POLICY_LEARNED_CORRUPTION_AWARE,
    W60_CACHE_POLICY_TRAINED_EVICTION,
    W60_CACHE_POLICY_COMPOSITE_V3,
)

W60_DEFAULT_CACHE_V3_RIDGE_LAMBDA: float = 0.10
W60_DEFAULT_CACHE_V3_COMPOSITE_LAMBDA: float = 0.05


def _safe_condition(a: "_np.ndarray") -> float:
    try:
        s = _np.linalg.svd(a, compute_uv=False)
        s_max = float(_np.max(s))
        s_min = float(_np.min(s))
        if s_min < 1e-30:
            return float("inf")
        return float(s_max / s_min)
    except Exception:
        return float("nan")


def _build_attention_receive_feature(
        attention_receive: Sequence["_np.ndarray"],
        *, n_tokens: int,
) -> "_np.ndarray":
    """Stack per-(layer, head, position) attention receive into a
    (n_tokens, L*H) feature."""
    if not attention_receive:
        return _np.zeros((int(n_tokens), 1), dtype=_np.float64)
    L = len(attention_receive)
    H = max(int(a.shape[0]) for a in attention_receive)
    feat = _np.zeros((int(n_tokens), L * H), dtype=_np.float64)
    for li, a in enumerate(attention_receive):
        if a.size == 0:
            continue
        h_l, t_l = a.shape
        for hi in range(min(H, h_l)):
            for ti in range(min(int(n_tokens), t_l)):
                feat[ti, li * H + hi] = float(a[hi, ti])
    return feat


@dataclasses.dataclass
class CacheControllerV3:
    """V3 controller. Pluggable retention policy with the new
    ``learned_attention_receive``, ``learned_corruption_aware``,
    ``trained_eviction``, and ``composite_v3`` heads."""

    policy: str
    d_model: int
    inner_v2: CacheControllerV2
    attention_receive_scorer: "_np.ndarray | None"
    trained_eviction_scorer: "_np.ndarray | None"
    composite_weights: "_np.ndarray | None"   # (4,)
    corruption_neg_inf_floor: float
    ridge_lambda: float
    fit_seed: int

    @classmethod
    def init(
            cls, *,
            policy: str = W58_CACHE_POLICY_UNIFORM,
            d_model: int = 64,
            probe_layer: int = -1,
            probe_layers: Sequence[int] = (
                W59_DEFAULT_CACHE_V2_PROBE_LAYERS),
            ridge_lambda: float = (
                W60_DEFAULT_CACHE_V3_RIDGE_LAMBDA),
            corruption_neg_inf_floor: float = -1e9,
            fit_seed: int = 60060,
    ) -> "CacheControllerV3":
        if policy not in W60_CACHE_POLICIES_V3:
            raise ValueError(
                f"policy must be in {W60_CACHE_POLICIES_V3}, "
                f"got {policy!r}")
        # If V3 policy not in V2 set, choose a V2 fallback for inner.
        if policy in W59_CACHE_POLICIES_V2:
            v2_policy = policy
        else:
            v2_policy = W58_CACHE_POLICY_IMPORTANCE
        inner_v2 = CacheControllerV2.init(
            policy=str(v2_policy),
            d_model=int(d_model),
            probe_layer=int(probe_layer),
            probe_layers=tuple(int(l) for l in probe_layers),
            ridge_lambda=float(ridge_lambda),
            fit_seed=int(fit_seed))
        return cls(
            policy=str(policy),
            d_model=int(d_model),
            inner_v2=inner_v2,
            attention_receive_scorer=None,
            trained_eviction_scorer=None,
            composite_weights=None,
            corruption_neg_inf_floor=float(
                corruption_neg_inf_floor),
            ridge_lambda=float(ridge_lambda),
            fit_seed=int(fit_seed),
        )

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W60_CACHE_CONTROLLER_V3_SCHEMA_VERSION,
            "kind": "cache_controller_v3",
            "policy": str(self.policy),
            "d_model": int(self.d_model),
            "inner_v2_cid": str(self.inner_v2.cid()),
            "attention_receive_scorer_cid": (
                _ndarray_cid(self.attention_receive_scorer)
                if self.attention_receive_scorer is not None
                else "untrained"),
            "trained_eviction_scorer_cid": (
                _ndarray_cid(self.trained_eviction_scorer)
                if self.trained_eviction_scorer is not None
                else "untrained"),
            "composite_weights_cid": (
                _ndarray_cid(self.composite_weights)
                if self.composite_weights is not None
                else "untrained"),
            "corruption_neg_inf_floor": float(round(
                self.corruption_neg_inf_floor, 6)),
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
            n_tokens: int,
    ) -> "_np.ndarray":
        if self.policy in W59_CACHE_POLICIES_V2:
            return self.inner_v2.score_tokens(
                hidden_state=hidden_state,
                hidden_states_layerset=hidden_states_layerset,
                importance_vector=importance_vector,
                query_vector=query_vector,
                n_tokens=int(n_tokens))
        if self.policy == (
                W60_CACHE_POLICY_LEARNED_ATTENTION_RECEIVE):
            if (self.attention_receive_scorer is None
                    or attention_receive is None):
                return _np.arange(
                    int(n_tokens), dtype=_np.float64)
            feat = _build_attention_receive_feature(
                attention_receive, n_tokens=int(n_tokens))
            return feat @ _np.asarray(
                self.attention_receive_scorer,
                dtype=_np.float64)
        if self.policy == (
                W60_CACHE_POLICY_LEARNED_CORRUPTION_AWARE):
            base = self.inner_v2.score_tokens(
                hidden_state=hidden_state,
                hidden_states_layerset=hidden_states_layerset,
                importance_vector=importance_vector,
                query_vector=query_vector,
                n_tokens=int(n_tokens))
            if corruption_flags is not None:
                # Mask any slot flagged in any layer.
                aggregate = _np.zeros(
                    (int(n_tokens),), dtype=_np.bool_)
                for f in corruption_flags:
                    if f.size:
                        m = min(int(f.size), int(n_tokens))
                        aggregate[:m] = (
                            aggregate[:m] | f[:m].astype(_np.bool_))
                base[aggregate] = float(
                    self.corruption_neg_inf_floor)
            return base
        if self.policy == W60_CACHE_POLICY_TRAINED_EVICTION:
            if self.trained_eviction_scorer is None:
                return _np.arange(
                    int(n_tokens), dtype=_np.float64)
            feat = _build_trained_eviction_feature(
                hidden_state=hidden_state,
                importance_vector=importance_vector,
                attention_receive=attention_receive,
                retrieval_scalar=retrieval_scalar,
                n_tokens=int(n_tokens))
            return feat @ _np.asarray(
                self.trained_eviction_scorer,
                dtype=_np.float64)
        if self.policy == W60_CACHE_POLICY_COMPOSITE_V3:
            if self.composite_weights is None:
                # Default to importance weighting.
                return importance_vector[:int(n_tokens)]
            # Score under each of the four V3 heads.
            scores: list["_np.ndarray"] = []
            # Head A: importance.
            scores.append(importance_vector[:int(n_tokens)])
            # Head B: learned_hidden via inner_v2 if fitted.
            v2_h = (
                self.inner_v2.scoring_head_v2
                if self.inner_v2.scoring_head_v2 is not None
                else None)
            if v2_h is not None:
                feat = _build_layerset_feature(
                    hidden_states_layerset,
                    n_tokens=int(n_tokens))
                scores.append(feat @ v2_h)
            else:
                scores.append(_np.zeros(
                    (int(n_tokens),), dtype=_np.float64))
            # Head C: learned_retrieval.
            if (self.inner_v2.retrieval_matrix is not None
                    and query_vector is not None):
                hs = _np.asarray(
                    hidden_state, dtype=_np.float64)
                if hs.shape[0] >= n_tokens:
                    q = _np.asarray(
                        query_vector, dtype=_np.float64).reshape(-1)
                    d = int(self.d_model)
                    if q.size < d:
                        q = _np.concatenate(
                            [q, _np.zeros(
                                d - q.size,
                                dtype=_np.float64)])
                    elif q.size > d:
                        q = q[:d]
                    M = _np.asarray(
                        self.inner_v2.retrieval_matrix,
                        dtype=_np.float64)
                    scores.append(
                        hs[:int(n_tokens)] @ M.T @ q)
                else:
                    scores.append(_np.zeros(
                        (int(n_tokens),), dtype=_np.float64))
            else:
                scores.append(_np.zeros(
                    (int(n_tokens),), dtype=_np.float64))
            # Head D: learned_attention_receive.
            if (self.attention_receive_scorer is not None
                    and attention_receive is not None):
                feat = _build_attention_receive_feature(
                    attention_receive, n_tokens=int(n_tokens))
                scores.append(feat @ self.attention_receive_scorer)
            else:
                scores.append(_np.zeros(
                    (int(n_tokens),), dtype=_np.float64))
            stacked = _np.stack(
                [s[:int(n_tokens)] for s in scores], axis=1)
            w = _np.asarray(
                self.composite_weights, dtype=_np.float64)
            if w.size != 4:
                w = _np.ones((4,), dtype=_np.float64) * 0.25
            out = stacked @ w
            # Mask corruption.
            if corruption_flags is not None:
                aggregate = _np.zeros(
                    (int(n_tokens),), dtype=_np.bool_)
                for f in corruption_flags:
                    if f.size:
                        m = min(int(f.size), int(n_tokens))
                        aggregate[:m] = (
                            aggregate[:m] | f[:m].astype(_np.bool_))
                out[aggregate] = float(
                    self.corruption_neg_inf_floor)
            return out
        raise ValueError(
            f"unknown V3 policy {self.policy!r}")


def _build_trained_eviction_feature(
        *, hidden_state: "_np.ndarray",
        importance_vector: "_np.ndarray",
        attention_receive: Sequence["_np.ndarray"] | None,
        retrieval_scalar: "_np.ndarray | None",
        n_tokens: int,
) -> "_np.ndarray":
    """Build a per-token feature vector for the trained-eviction
    head: [hidden_state, importance, attention_receive_l1,
    retrieval_scalar]."""
    n = int(n_tokens)
    hs = _np.asarray(hidden_state, dtype=_np.float64)
    if hs.shape[0] < n:
        hs = _np.concatenate(
            [hs, _np.zeros((n - hs.shape[0], hs.shape[1]),
                            dtype=_np.float64)], axis=0)
    hs = hs[:n]
    imp = _np.asarray(
        importance_vector, dtype=_np.float64).reshape(-1)
    if imp.size < n:
        imp = _np.concatenate(
            [imp, _np.zeros(n - imp.size, dtype=_np.float64)])
    imp = imp[:n]
    if attention_receive is not None and attention_receive:
        ar_l1 = _np.zeros((n,), dtype=_np.float64)
        for a in attention_receive:
            if a.size:
                col = _np.sum(a, axis=0)
                m = min(int(col.size), n)
                ar_l1[:m] += col[:m]
    else:
        ar_l1 = _np.zeros((n,), dtype=_np.float64)
    if retrieval_scalar is not None:
        rs = _np.asarray(
            retrieval_scalar, dtype=_np.float64).reshape(-1)
        if rs.size < n:
            rs = _np.concatenate(
                [rs, _np.zeros(n - rs.size,
                                dtype=_np.float64)])
        rs = rs[:n]
    else:
        rs = _np.zeros((n,), dtype=_np.float64)
    return _np.concatenate(
        [hs,
         imp.reshape(-1, 1),
         ar_l1.reshape(-1, 1),
         rs.reshape(-1, 1)], axis=1)


@dataclasses.dataclass(frozen=True)
class CacheControllerV3FitReport:
    schema: str
    policy: str
    n_train_tokens: int
    pre_fit_residual: float
    post_fit_residual: float
    ridge_lambda: float
    condition_number: float
    feature_dim: int
    converged: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "policy": str(self.policy),
            "n_train_tokens": int(self.n_train_tokens),
            "pre_fit_residual": float(round(
                self.pre_fit_residual, 12)),
            "post_fit_residual": float(round(
                self.post_fit_residual, 12)),
            "ridge_lambda": float(round(self.ridge_lambda, 12)),
            "condition_number": float(round(
                self.condition_number, 12)),
            "feature_dim": int(self.feature_dim),
            "converged": bool(self.converged),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "cache_controller_v3_fit_report",
            "report": self.to_dict()})


def fit_learned_attention_receive_controller(
        params: TinyV3SubstrateParams,
        *,
        prompt_token_ids: Sequence[int],
        follow_up_token_ids: Sequence[int],
        attention_receive: Sequence["_np.ndarray"],
        ridge_lambda: float = (
            W60_DEFAULT_CACHE_V3_RIDGE_LAMBDA),
        fit_seed: int = 60061,
) -> tuple[CacheControllerV3, CacheControllerV3FitReport]:
    cfg = params.config
    prefix_trace = forward_tiny_substrate_v3(
        params, list(prompt_token_ids),
        return_attention=False)
    n = int(prefix_trace.kv_cache.n_tokens())
    feat = _build_attention_receive_feature(
        attention_receive, n_tokens=n)
    oracle = _compute_drop_oracle(
        params, list(prompt_token_ids),
        follow_up_token_ids=list(follow_up_token_ids))
    X = feat
    y = oracle[: X.shape[0]]
    d = X.shape[1]
    lam = max(float(ridge_lambda), 1e-9)
    A = X.T @ X + lam * _np.eye(d, dtype=_np.float64)
    b = X.T @ y
    w = _np.linalg.solve(A, b)
    cond = _safe_condition(A)
    pre_res = float(_np.mean(_np.abs(y)))
    post_pred = X @ w
    post_res = float(_np.mean(_np.abs(y - post_pred)))
    ctrl = CacheControllerV3.init(
        policy=W60_CACHE_POLICY_LEARNED_ATTENTION_RECEIVE,
        d_model=int(cfg.d_model),
        ridge_lambda=float(ridge_lambda),
        fit_seed=int(fit_seed))
    ctrl.attention_receive_scorer = w
    return ctrl, CacheControllerV3FitReport(
        schema=W60_CACHE_CONTROLLER_V3_SCHEMA_VERSION,
        policy=W60_CACHE_POLICY_LEARNED_ATTENTION_RECEIVE,
        n_train_tokens=int(X.shape[0]),
        pre_fit_residual=float(pre_res),
        post_fit_residual=float(post_res),
        ridge_lambda=float(ridge_lambda),
        condition_number=float(cond),
        feature_dim=int(d),
        converged=bool(post_res <= pre_res + 1e-12),
    )


def fit_trained_eviction_controller(
        params: TinyV3SubstrateParams,
        *,
        prompt_token_ids: Sequence[int],
        follow_up_token_ids: Sequence[int],
        attention_receive: Sequence["_np.ndarray"] | None = None,
        retrieval_scalar: "_np.ndarray | None" = None,
        ridge_lambda: float = (
            W60_DEFAULT_CACHE_V3_RIDGE_LAMBDA),
        fit_seed: int = 60062,
) -> tuple[CacheControllerV3, CacheControllerV3FitReport]:
    cfg = params.config
    prefix_trace = forward_tiny_substrate_v3(
        params, list(prompt_token_ids),
        return_attention=False)
    n = int(prefix_trace.kv_cache.n_tokens())
    hidden = prefix_trace.hidden_states[-1]
    if prefix_trace.kv_cache.importance and any(
            i.size for i in prefix_trace.kv_cache.importance):
        imp = _np.zeros((n,), dtype=_np.float64)
        for li in prefix_trace.kv_cache.importance:
            if li.size and int(li.shape[0]) == n:
                imp = imp + li
    else:
        imp = _np.zeros((n,), dtype=_np.float64)
    feat = _build_trained_eviction_feature(
        hidden_state=hidden, importance_vector=imp,
        attention_receive=attention_receive,
        retrieval_scalar=retrieval_scalar, n_tokens=n)
    oracle = _compute_drop_oracle(
        params, list(prompt_token_ids),
        follow_up_token_ids=list(follow_up_token_ids))
    X = feat
    y = oracle[: X.shape[0]]
    d = X.shape[1]
    lam = max(float(ridge_lambda), 1e-9)
    A = X.T @ X + lam * _np.eye(d, dtype=_np.float64)
    b = X.T @ y
    w = _np.linalg.solve(A, b)
    cond = _safe_condition(A)
    pre_res = float(_np.mean(_np.abs(y)))
    post_pred = X @ w
    post_res = float(_np.mean(_np.abs(y - post_pred)))
    ctrl = CacheControllerV3.init(
        policy=W60_CACHE_POLICY_TRAINED_EVICTION,
        d_model=int(cfg.d_model),
        ridge_lambda=float(ridge_lambda),
        fit_seed=int(fit_seed))
    ctrl.trained_eviction_scorer = w
    return ctrl, CacheControllerV3FitReport(
        schema=W60_CACHE_CONTROLLER_V3_SCHEMA_VERSION,
        policy=W60_CACHE_POLICY_TRAINED_EVICTION,
        n_train_tokens=int(X.shape[0]),
        pre_fit_residual=float(pre_res),
        post_fit_residual=float(post_res),
        ridge_lambda=float(ridge_lambda),
        condition_number=float(cond),
        feature_dim=int(d),
        converged=bool(post_res <= pre_res + 1e-12),
    )


def fit_composite_v3_controller(
        params: TinyV3SubstrateParams,
        *,
        prompt_token_ids: Sequence[int],
        follow_up_token_ids: Sequence[int],
        importance_vector: "_np.ndarray",
        learned_hidden_scores: "_np.ndarray",
        learned_retrieval_scores: "_np.ndarray",
        learned_attention_receive_scores: "_np.ndarray",
        ridge_lambda: float = (
            W60_DEFAULT_CACHE_V3_COMPOSITE_LAMBDA),
        fit_seed: int = 60063,
) -> tuple[CacheControllerV3, CacheControllerV3FitReport]:
    """Fit composite mixture weights ``w ∈ R^4`` via closed-form
    ridge against the V1 drop oracle."""
    cfg = params.config
    oracle = _compute_drop_oracle(
        params, list(prompt_token_ids),
        follow_up_token_ids=list(follow_up_token_ids))
    n = int(min(
        int(importance_vector.size),
        int(learned_hidden_scores.size),
        int(learned_retrieval_scores.size),
        int(learned_attention_receive_scores.size),
        int(oracle.size)))
    if n <= 0:
        ctrl = CacheControllerV3.init(
            policy=W60_CACHE_POLICY_COMPOSITE_V3,
            d_model=int(cfg.d_model),
            ridge_lambda=float(ridge_lambda),
            fit_seed=int(fit_seed))
        ctrl.composite_weights = _np.array(
            [0.25, 0.25, 0.25, 0.25], dtype=_np.float64)
        return ctrl, CacheControllerV3FitReport(
            schema=W60_CACHE_CONTROLLER_V3_SCHEMA_VERSION,
            policy=W60_CACHE_POLICY_COMPOSITE_V3,
            n_train_tokens=0,
            pre_fit_residual=0.0,
            post_fit_residual=0.0,
            ridge_lambda=float(ridge_lambda),
            condition_number=float("nan"),
            feature_dim=4,
            converged=True,
        )
    X = _np.stack([
        _np.asarray(importance_vector,
                      dtype=_np.float64).reshape(-1)[:n],
        _np.asarray(learned_hidden_scores,
                      dtype=_np.float64).reshape(-1)[:n],
        _np.asarray(learned_retrieval_scores,
                      dtype=_np.float64).reshape(-1)[:n],
        _np.asarray(learned_attention_receive_scores,
                      dtype=_np.float64).reshape(-1)[:n],
    ], axis=1)
    y = oracle[:n]
    d = 4
    lam = max(float(ridge_lambda), 1e-9)
    A = X.T @ X + lam * _np.eye(d, dtype=_np.float64)
    b = X.T @ y
    w = _np.linalg.solve(A, b)
    cond = _safe_condition(A)
    pre_res = float(_np.mean(_np.abs(y)))
    post_pred = X @ w
    post_res = float(_np.mean(_np.abs(y - post_pred)))
    ctrl = CacheControllerV3.init(
        policy=W60_CACHE_POLICY_COMPOSITE_V3,
        d_model=int(cfg.d_model),
        ridge_lambda=float(ridge_lambda),
        fit_seed=int(fit_seed))
    ctrl.composite_weights = w
    return ctrl, CacheControllerV3FitReport(
        schema=W60_CACHE_CONTROLLER_V3_SCHEMA_VERSION,
        policy=W60_CACHE_POLICY_COMPOSITE_V3,
        n_train_tokens=int(n),
        pre_fit_residual=float(pre_res),
        post_fit_residual=float(post_res),
        ridge_lambda=float(ridge_lambda),
        condition_number=float(cond),
        feature_dim=int(d),
        converged=bool(post_res <= pre_res + 1e-12),
    )


@dataclasses.dataclass(frozen=True)
class CacheControllerV3Witness:
    schema: str
    controller_cid: str
    policy: str
    n_prompt_tokens: int
    n_keep: int
    retention_ratio: float
    flop_full_recompute: int
    flop_with_controller: int
    flop_saved: int
    flop_savings_ratio: float
    last_position_argmax_full: int
    last_position_argmax_controlled: int
    argmax_preserved: bool
    last_logit_l1_drift: float
    last_logit_l2_drift: float
    used_attention_receive: bool
    used_corruption_flags: bool
    used_composite: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "controller_cid": str(self.controller_cid),
            "policy": str(self.policy),
            "n_prompt_tokens": int(self.n_prompt_tokens),
            "n_keep": int(self.n_keep),
            "retention_ratio": float(round(
                self.retention_ratio, 12)),
            "flop_full_recompute": int(
                self.flop_full_recompute),
            "flop_with_controller": int(
                self.flop_with_controller),
            "flop_saved": int(self.flop_saved),
            "flop_savings_ratio": float(round(
                self.flop_savings_ratio, 12)),
            "last_position_argmax_full": int(
                self.last_position_argmax_full),
            "last_position_argmax_controlled": int(
                self.last_position_argmax_controlled),
            "argmax_preserved": bool(self.argmax_preserved),
            "last_logit_l1_drift": float(round(
                self.last_logit_l1_drift, 12)),
            "last_logit_l2_drift": float(round(
                self.last_logit_l2_drift, 12)),
            "used_attention_receive": bool(
                self.used_attention_receive),
            "used_corruption_flags": bool(
                self.used_corruption_flags),
            "used_composite": bool(self.used_composite),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "cache_controller_v3_witness",
            "witness": self.to_dict()})


def apply_cache_controller_v3_and_measure(
        params: TinyV3SubstrateParams,
        controller: CacheControllerV3,
        *,
        prompt_token_ids: Sequence[int],
        follow_up_token_ids: Sequence[int],
        retention_ratio: float = 0.5,
        query_vector: Sequence[float] | None = None,
        attention_receive: Sequence["_np.ndarray"] | None = None,
        corruption_flags: Sequence["_np.ndarray"] | None = None,
        retrieval_scalar: Sequence[float] | None = None,
) -> CacheControllerV3Witness:
    cfg = params.config
    prefix_trace = forward_tiny_substrate_v3(
        params, list(prompt_token_ids),
        return_attention=False)
    full_cache = prefix_trace.kv_cache
    n_prompt = int(full_cache.n_tokens())
    keep = max(1, int(retention_ratio * n_prompt))
    full_trace = forward_tiny_substrate_v3(
        params, list(follow_up_token_ids),
        kv_cache=full_cache, return_attention=False)
    full_logits = full_trace.logits[-1]
    pl = int(controller.inner_v2.probe_layer)
    if pl < 0:
        pl = len(prefix_trace.hidden_states) + pl
    hidden = prefix_trace.hidden_states[int(pl)]
    hs_layerset: list["_np.ndarray"] = []
    for spl in controller.inner_v2.probe_layers:
        idx = int(spl)
        if idx < 0:
            idx = len(prefix_trace.hidden_states) + idx
        hs_layerset.append(prefix_trace.hidden_states[int(idx)])
    if full_cache.importance and any(
            i.size for i in full_cache.importance):
        imp = _np.zeros((n_prompt,), dtype=_np.float64)
        for li in full_cache.importance:
            if li.size and int(li.shape[0]) == n_prompt:
                imp = imp + li
    else:
        imp = _np.zeros((n_prompt,), dtype=_np.float64)
    qv = (None if query_vector is None
          else _np.asarray(query_vector, dtype=_np.float64))
    rs = (None if retrieval_scalar is None
          else _np.asarray(retrieval_scalar,
                            dtype=_np.float64))
    scores = controller.score_tokens(
        hidden_state=hidden,
        hidden_states_layerset=hs_layerset,
        importance_vector=imp,
        query_vector=qv,
        attention_receive=attention_receive,
        corruption_flags=corruption_flags,
        retrieval_scalar=rs,
        n_tokens=int(n_prompt))
    evicted_cache = full_cache.evict_weighted(
        scores.tolist(), int(keep))
    evicted_trace = forward_tiny_substrate_v3(
        params, list(follow_up_token_ids),
        kv_cache=evicted_cache, return_attention=False)
    evicted_logits = evicted_trace.logits[-1]
    flop_full = int(
        prefix_trace.flop_count + full_trace.flop_count)
    flop_ctrl = int(
        prefix_trace.flop_count + evicted_trace.flop_count)
    flop_saved = int(flop_full - flop_ctrl)
    flop_ratio = (
        float(flop_saved) / float(max(flop_full, 1)))
    last_full_argmax = int(_np.argmax(full_logits))
    last_ctrl_argmax = int(_np.argmax(evicted_logits))
    argmax_preserved = bool(
        last_full_argmax == last_ctrl_argmax)
    l1 = float(_np.sum(_np.abs(full_logits - evicted_logits)))
    l2 = float(_np.linalg.norm(full_logits - evicted_logits))
    return CacheControllerV3Witness(
        schema=W60_CACHE_CONTROLLER_V3_SCHEMA_VERSION,
        controller_cid=str(controller.cid()),
        policy=str(controller.policy),
        n_prompt_tokens=int(n_prompt),
        n_keep=int(keep),
        retention_ratio=float(keep) / float(max(n_prompt, 1)),
        flop_full_recompute=int(flop_full),
        flop_with_controller=int(flop_ctrl),
        flop_saved=int(flop_saved),
        flop_savings_ratio=float(flop_ratio),
        last_position_argmax_full=int(last_full_argmax),
        last_position_argmax_controlled=int(last_ctrl_argmax),
        argmax_preserved=bool(argmax_preserved),
        last_logit_l1_drift=float(l1),
        last_logit_l2_drift=float(l2),
        used_attention_receive=bool(
            attention_receive is not None),
        used_corruption_flags=bool(
            corruption_flags is not None),
        used_composite=bool(
            controller.policy == W60_CACHE_POLICY_COMPOSITE_V3),
    )


__all__ = [
    "W60_CACHE_CONTROLLER_V3_SCHEMA_VERSION",
    "W60_CACHE_POLICY_LEARNED_ATTENTION_RECEIVE",
    "W60_CACHE_POLICY_LEARNED_CORRUPTION_AWARE",
    "W60_CACHE_POLICY_TRAINED_EVICTION",
    "W60_CACHE_POLICY_COMPOSITE_V3",
    "W60_CACHE_POLICIES_V3",
    "W60_DEFAULT_CACHE_V3_RIDGE_LAMBDA",
    "W60_DEFAULT_CACHE_V3_COMPOSITE_LAMBDA",
    "CacheControllerV3",
    "CacheControllerV3FitReport",
    "CacheControllerV3Witness",
    "fit_learned_attention_receive_controller",
    "fit_trained_eviction_controller",
    "fit_composite_v3_controller",
    "apply_cache_controller_v3_and_measure",
]
