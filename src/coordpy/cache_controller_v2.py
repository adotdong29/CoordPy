"""W59 M6 — Cache Controller V2 (trainable retention + retrieval).

Strictly extends W58's ``coordpy.cache_controller``. V1 had three
policies (``uniform``, ``importance``, ``learned``) with the
learned head fit by single-layer closed-form ridge regression
against a drop-oracle.

V2 keeps all three V1 policies (so a V2 controller can degrade to
V1 byte-for-byte by selecting ``policy="uniform"`` etc.) and adds
**two new policies**:

* ``learned_hidden`` — closed-form ridge regression on a *cross-
  layer* hidden-state feature: instead of using a single probe
  layer, V2 concatenates hidden states from a *layer set* (default
  ``[1, 3, -1]``) into a feature vector. The fitted scorer is a
  single linear head over that concatenated feature, fit by
  closed-form ridge against the same drop-oracle V1 used.
* ``learned_retrieval`` — V2 also exposes a *retrieval* surface:
  given a **latent query vector** (e.g. the next-token decoder
  carrier or a downstream agent's query embedding) the controller
  scores each KV slot's *similarity* to that query under a fitted
  bilinear form  ``score(t) = q^T M h_t`` where ``M`` is fitted
  by closed-form ridge so that the substrate's leave-one-out drop
  oracle aligns with that score on the train set. This makes
  retrieval *query-conditional*, not just turn-static.

V2 also introduces a **trainable retrieval scorer report** with
pre/post-fit mean residuals, ridge λ, and the conditioning number
of the linear system. It is the W59 milestone's first head whose
*parameters* are non-trivially fit (the V4 KV-bridge correction
is a global α; the V2 controller's retrieval scorer is a real
``d×d`` matrix fit by closed-form ridge).

V2 strictly extends V1: ``CacheController.score_tokens`` is
preserved unchanged; the new policies are accessed through
``CacheControllerV2``.

Honest scope
------------

* Both new policies use **closed-form linear regression**. No
  SGD / autograd / GPU. ``W59-L-V4-NO-AUTOGRAD-CAP`` carries
  forward.
* "Learned" does not imply *useful*. The R-122 benchmark
  publishes the (fidelity, flops) frontier per policy; learned-
  retrieval is a real, fitted, query-conditional scorer; whether
  it beats importance on any given task is empirical.
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
        "coordpy.cache_controller_v2 requires numpy") from exc

from .cache_controller import (
    CacheController,
    CacheControllerWitness,
    W58_CACHE_CONTROLLER_SCHEMA_VERSION,
    W58_CACHE_POLICY_IMPORTANCE,
    W58_CACHE_POLICY_LEARNED,
    W58_CACHE_POLICY_UNIFORM,
    _compute_drop_oracle,
)
from .tiny_substrate_v3 import (
    TinyV3KVCache,
    TinyV3SubstrateParams,
    _ndarray_cid,
    _sha256_hex,
    forward_tiny_substrate_v3,
)


W59_CACHE_CONTROLLER_V2_SCHEMA_VERSION: str = (
    "coordpy.cache_controller_v2.v1")

W59_CACHE_POLICY_LEARNED_HIDDEN: str = "learned_hidden"
W59_CACHE_POLICY_LEARNED_RETRIEVAL: str = "learned_retrieval"

W59_CACHE_POLICIES_V2: tuple[str, ...] = (
    W58_CACHE_POLICY_UNIFORM,
    W58_CACHE_POLICY_IMPORTANCE,
    W58_CACHE_POLICY_LEARNED,
    W59_CACHE_POLICY_LEARNED_HIDDEN,
    W59_CACHE_POLICY_LEARNED_RETRIEVAL,
)

W59_DEFAULT_CACHE_V2_RIDGE_LAMBDA: float = 0.10
W59_DEFAULT_CACHE_V2_PROBE_LAYERS: tuple[int, ...] = (1, 3, -1)


@dataclasses.dataclass
class CacheControllerV2:
    """V2 controller. Pluggable retention policy with the new
    ``learned_hidden`` and ``learned_retrieval`` heads."""

    policy: str
    d_model: int
    probe_layer: int             # for V1 learned policy
    probe_layers: tuple[int, ...]  # for V2 learned_hidden
    scoring_head: "_np.ndarray | None"
    scoring_head_v2: "_np.ndarray | None"
    retrieval_matrix: "_np.ndarray | None"  # (d_model, d_model)
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
                W59_DEFAULT_CACHE_V2_RIDGE_LAMBDA),
            fit_seed: int = 59060,
    ) -> "CacheControllerV2":
        if policy not in W59_CACHE_POLICIES_V2:
            raise ValueError(
                f"policy must be in {W59_CACHE_POLICIES_V2}, "
                f"got {policy!r}")
        return cls(
            policy=str(policy),
            d_model=int(d_model),
            probe_layer=int(probe_layer),
            probe_layers=tuple(int(l) for l in probe_layers),
            scoring_head=None,
            scoring_head_v2=None,
            retrieval_matrix=None,
            ridge_lambda=float(ridge_lambda),
            fit_seed=int(fit_seed),
        )

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W59_CACHE_CONTROLLER_V2_SCHEMA_VERSION,
            "policy": str(self.policy),
            "d_model": int(self.d_model),
            "probe_layer": int(self.probe_layer),
            "probe_layers": list(self.probe_layers),
            "scoring_head_cid": (
                _ndarray_cid(self.scoring_head)
                if self.scoring_head is not None
                else "untrained"),
            "scoring_head_v2_cid": (
                _ndarray_cid(self.scoring_head_v2)
                if self.scoring_head_v2 is not None
                else "untrained"),
            "retrieval_matrix_cid": (
                _ndarray_cid(self.retrieval_matrix)
                if self.retrieval_matrix is not None
                else "untrained"),
            "ridge_lambda": float(self.ridge_lambda),
            "fit_seed": int(self.fit_seed),
        })

    def to_v1(self) -> CacheController:
        """Compatibility: produce a V1 CacheController with V1
        policy ∈ {uniform, importance, learned}. Raises if the V2
        policy has no V1 analogue."""
        if str(self.policy) in {
                W58_CACHE_POLICY_UNIFORM,
                W58_CACHE_POLICY_IMPORTANCE,
                W58_CACHE_POLICY_LEARNED,
        }:
            v1 = CacheController.init(
                policy=str(self.policy),
                d_model=int(self.d_model),
                probe_layer=int(self.probe_layer),
                ridge_lambda=float(self.ridge_lambda),
                fit_seed=int(self.fit_seed))
            if str(self.policy) == W58_CACHE_POLICY_LEARNED:
                v1.scoring_head = self.scoring_head
            return v1
        raise ValueError(
            f"V2 policy {self.policy!r} has no V1 analogue")

    def score_tokens(
            self,
            *,
            hidden_state: "_np.ndarray",
            hidden_states_layerset: Sequence["_np.ndarray"],
            importance_vector: "_np.ndarray",
            query_vector: "_np.ndarray | None",
            n_tokens: int,
    ) -> "_np.ndarray":
        if str(self.policy) in {
                W58_CACHE_POLICY_UNIFORM,
                W58_CACHE_POLICY_IMPORTANCE,
                W58_CACHE_POLICY_LEARNED,
        }:
            v1 = self.to_v1()
            return v1.score_tokens(
                hidden_state=hidden_state,
                importance_vector=importance_vector,
                n_tokens=int(n_tokens))
        if str(self.policy) == W59_CACHE_POLICY_LEARNED_HIDDEN:
            if self.scoring_head_v2 is None:
                return _np.arange(
                    int(n_tokens), dtype=_np.float64)
            feat = _build_layerset_feature(
                hidden_states_layerset, n_tokens=int(n_tokens))
            return feat @ _np.asarray(
                self.scoring_head_v2, dtype=_np.float64)
        if str(self.policy) == W59_CACHE_POLICY_LEARNED_RETRIEVAL:
            if (self.retrieval_matrix is None
                    or query_vector is None):
                return _np.arange(
                    int(n_tokens), dtype=_np.float64)
            hs = _np.asarray(hidden_state, dtype=_np.float64)
            if hs.shape[0] < n_tokens:
                return _np.arange(
                    int(n_tokens), dtype=_np.float64)
            q = _np.asarray(query_vector,
                              dtype=_np.float64).reshape(-1)
            d = int(self.d_model)
            if q.size < d:
                q = _np.concatenate(
                    [q, _np.zeros(d - q.size,
                                  dtype=_np.float64)])
            elif q.size > d:
                q = q[:d]
            m = _np.asarray(
                self.retrieval_matrix, dtype=_np.float64)
            # score(t) = q^T M h_t
            return hs[: int(n_tokens)] @ m.T @ q
        raise ValueError(
            f"unknown V2 policy {self.policy!r}")


def _build_layerset_feature(
        hidden_states_layerset: Sequence["_np.ndarray"],
        *, n_tokens: int,
) -> "_np.ndarray":
    """Concatenate hidden states from a layer set into a single
    feature vector per token."""
    parts: list["_np.ndarray"] = []
    for hs in hidden_states_layerset:
        a = _np.asarray(hs, dtype=_np.float64)
        if a.shape[0] >= n_tokens:
            parts.append(a[: int(n_tokens)])
        else:
            pad = _np.zeros(
                (int(n_tokens) - int(a.shape[0]), a.shape[1]),
                dtype=_np.float64)
            parts.append(_np.concatenate([a, pad], axis=0))
    if not parts:
        return _np.zeros((int(n_tokens), 1), dtype=_np.float64)
    return _np.concatenate(parts, axis=1)


@dataclasses.dataclass(frozen=True)
class CacheControllerV2FitReport:
    schema: str
    policy: str
    n_train_tokens: int
    pre_fit_mean_residual: float
    post_fit_mean_residual: float
    ridge_lambda: float
    condition_number: float
    feature_dim: int
    converged: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "policy": str(self.policy),
            "n_train_tokens": int(self.n_train_tokens),
            "pre_fit_mean_residual": float(round(
                self.pre_fit_mean_residual, 12)),
            "post_fit_mean_residual": float(round(
                self.post_fit_mean_residual, 12)),
            "ridge_lambda": float(round(self.ridge_lambda, 12)),
            "condition_number": float(round(
                self.condition_number, 12)),
            "feature_dim": int(self.feature_dim),
            "converged": bool(self.converged),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "cache_controller_v2_fit_report",
            "report": self.to_dict()})


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


def fit_learned_hidden_cache_controller(
        params: TinyV3SubstrateParams,
        *,
        prompt_token_ids: Sequence[int],
        follow_up_token_ids: Sequence[int],
        probe_layers: Sequence[int] = (
            W59_DEFAULT_CACHE_V2_PROBE_LAYERS),
        ridge_lambda: float = (
            W59_DEFAULT_CACHE_V2_RIDGE_LAMBDA),
        fit_seed: int = 59061,
) -> tuple[CacheControllerV2, CacheControllerV2FitReport]:
    """Closed-form ridge regression on a *cross-layer* hidden-state
    feature, targeting the V1 drop-oracle."""
    cfg = params.config
    prefix_trace = forward_tiny_substrate_v3(
        params, list(prompt_token_ids),
        return_attention=False)
    # Build the cross-layer feature.
    hs_layerset: list["_np.ndarray"] = []
    for pl in probe_layers:
        idx = int(pl)
        if idx < 0:
            idx = len(prefix_trace.hidden_states) + idx
        hs_layerset.append(prefix_trace.hidden_states[int(idx)])
    n = int(prefix_trace.kv_cache.n_tokens())
    feat = _build_layerset_feature(hs_layerset, n_tokens=n)
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
    pre_y = _np.zeros_like(y)
    pre_res = float(_np.mean(_np.abs(y - pre_y)))
    post_pred = X @ w
    post_res = float(_np.mean(_np.abs(y - post_pred)))
    ctrl = CacheControllerV2.init(
        policy=W59_CACHE_POLICY_LEARNED_HIDDEN,
        d_model=int(cfg.d_model),
        probe_layers=tuple(int(l) for l in probe_layers),
        ridge_lambda=float(ridge_lambda),
        fit_seed=int(fit_seed))
    ctrl.scoring_head_v2 = w
    return ctrl, CacheControllerV2FitReport(
        schema=W59_CACHE_CONTROLLER_V2_SCHEMA_VERSION,
        policy=W59_CACHE_POLICY_LEARNED_HIDDEN,
        n_train_tokens=int(X.shape[0]),
        pre_fit_mean_residual=float(pre_res),
        post_fit_mean_residual=float(post_res),
        ridge_lambda=float(ridge_lambda),
        condition_number=float(cond),
        feature_dim=int(d),
        converged=bool(post_res <= pre_res + 1e-12),
    )


def fit_learned_retrieval_cache_controller(
        params: TinyV3SubstrateParams,
        *,
        prompt_token_ids: Sequence[int],
        follow_up_token_ids: Sequence[int],
        query_vector: Sequence[float],
        probe_layer: int = -1,
        ridge_lambda: float = (
            W59_DEFAULT_CACHE_V2_RIDGE_LAMBDA),
        fit_seed: int = 59062,
) -> tuple[CacheControllerV2, CacheControllerV2FitReport]:
    """Closed-form ridge regression for the bilinear retrieval
    matrix ``M`` so that ``q^T M h_t`` matches the drop-oracle.

    With features ``f_t = h_t * q^T`` (outer product, but flattened
    we use the projection ``h_t (q ⊗ h_t)^T M``). To keep this a
    *linear* problem in ``M``, we use the equivalent vec form: the
    target for token t is ``y_t = oracle_t``, and the regressor is
    the outer product ``(q h_t^T).flatten()`` so the linear weights
    over the outer-product features are exactly ``vec(M)``.
    """
    cfg = params.config
    prefix_trace = forward_tiny_substrate_v3(
        params, list(prompt_token_ids),
        return_attention=False)
    pl = int(probe_layer)
    if pl < 0:
        pl = len(prefix_trace.hidden_states) + pl
    hs = prefix_trace.hidden_states[int(pl)]
    n = int(prefix_trace.kv_cache.n_tokens())
    n = min(int(n), int(hs.shape[0]))
    d = int(hs.shape[1])
    q = _np.asarray(
        query_vector, dtype=_np.float64).reshape(-1)
    if q.size < d:
        q = _np.concatenate(
            [q, _np.zeros(d - q.size, dtype=_np.float64)])
    elif q.size > d:
        q = q[:d]
    # Build per-token feature = outer(q, h_t).flatten -> dim d*d
    X = _np.zeros((int(n), d * d), dtype=_np.float64)
    for t in range(int(n)):
        outer = _np.outer(q, hs[t])
        X[t] = outer.reshape(-1)
    oracle = _compute_drop_oracle(
        params, list(prompt_token_ids),
        follow_up_token_ids=list(follow_up_token_ids))
    y = oracle[: X.shape[0]]
    lam = max(float(ridge_lambda), 1e-9)
    # Solve via X^T X + λI (size (d*d) × (d*d)).
    A = X.T @ X + lam * _np.eye(d * d, dtype=_np.float64)
    b = X.T @ y
    w = _np.linalg.solve(A, b)
    cond = _safe_condition(A)
    M = w.reshape(d, d)
    pre_res = float(_np.mean(_np.abs(y)))
    post_pred = X @ w
    post_res = float(_np.mean(_np.abs(y - post_pred)))
    ctrl = CacheControllerV2.init(
        policy=W59_CACHE_POLICY_LEARNED_RETRIEVAL,
        d_model=int(cfg.d_model),
        probe_layer=int(probe_layer),
        ridge_lambda=float(ridge_lambda),
        fit_seed=int(fit_seed))
    ctrl.retrieval_matrix = M
    return ctrl, CacheControllerV2FitReport(
        schema=W59_CACHE_CONTROLLER_V2_SCHEMA_VERSION,
        policy=W59_CACHE_POLICY_LEARNED_RETRIEVAL,
        n_train_tokens=int(X.shape[0]),
        pre_fit_mean_residual=float(pre_res),
        post_fit_mean_residual=float(post_res),
        ridge_lambda=float(ridge_lambda),
        condition_number=float(cond),
        feature_dim=int(d * d),
        converged=bool(post_res <= pre_res + 1e-12),
    )


@dataclasses.dataclass(frozen=True)
class CacheControllerV2Witness:
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
    used_retrieval_query: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "controller_cid": str(self.controller_cid),
            "policy": str(self.policy),
            "n_prompt_tokens": int(self.n_prompt_tokens),
            "n_keep": int(self.n_keep),
            "retention_ratio": float(round(
                self.retention_ratio, 12)),
            "flop_full_recompute": int(self.flop_full_recompute),
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
            "used_retrieval_query": bool(
                self.used_retrieval_query),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "cache_controller_v2_witness",
            "witness": self.to_dict()})


def apply_cache_controller_v2_and_measure(
        params: TinyV3SubstrateParams,
        controller: CacheControllerV2,
        *,
        prompt_token_ids: Sequence[int],
        follow_up_token_ids: Sequence[int],
        retention_ratio: float = 0.5,
        query_vector: Sequence[float] | None = None,
) -> CacheControllerV2Witness:
    """Run V2 controller. Same interface as V1; supports the new
    policies."""
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
    # Hidden state at probe layer.
    pl = int(controller.probe_layer)
    if pl < 0:
        pl = len(prefix_trace.hidden_states) + pl
    hidden = prefix_trace.hidden_states[int(pl)]
    hs_layerset: list["_np.ndarray"] = []
    for spl in controller.probe_layers:
        idx = int(spl)
        if idx < 0:
            idx = len(prefix_trace.hidden_states) + idx
        hs_layerset.append(prefix_trace.hidden_states[int(idx)])
    if full_cache.importance and any(
            i.size for i in full_cache.importance):
        imp = _np.zeros((n_prompt,), dtype=_np.float64)
        for layer_imp in full_cache.importance:
            if (layer_imp.size
                    and int(layer_imp.shape[0]) == n_prompt):
                imp = imp + layer_imp
    else:
        imp = _np.zeros((n_prompt,), dtype=_np.float64)
    qv = (None if query_vector is None
          else _np.asarray(query_vector, dtype=_np.float64))
    scores = controller.score_tokens(
        hidden_state=hidden,
        hidden_states_layerset=hs_layerset,
        importance_vector=imp,
        query_vector=qv,
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
    argmax_preserved = bool(last_full_argmax == last_ctrl_argmax)
    l1 = float(_np.sum(_np.abs(full_logits - evicted_logits)))
    l2 = float(_np.linalg.norm(full_logits - evicted_logits))
    return CacheControllerV2Witness(
        schema=W59_CACHE_CONTROLLER_V2_SCHEMA_VERSION,
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
        used_retrieval_query=bool(query_vector is not None),
    )


__all__ = [
    "W59_CACHE_CONTROLLER_V2_SCHEMA_VERSION",
    "W59_CACHE_POLICY_LEARNED_HIDDEN",
    "W59_CACHE_POLICY_LEARNED_RETRIEVAL",
    "W59_CACHE_POLICIES_V2",
    "W59_DEFAULT_CACHE_V2_RIDGE_LAMBDA",
    "W59_DEFAULT_CACHE_V2_PROBE_LAYERS",
    "CacheControllerV2",
    "CacheControllerV2FitReport",
    "CacheControllerV2Witness",
    "fit_learned_hidden_cache_controller",
    "fit_learned_retrieval_cache_controller",
    "apply_cache_controller_v2_and_measure",
]
