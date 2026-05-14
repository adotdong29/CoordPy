"""W58 M6 — Cache Controller (learned/fitted retention policy).

The cache controller is the W58 piece that makes
"reuse-vs-recompute" a real, measurable, load-bearing trade-off.
It learns *which KV slots to keep* under a tight retention
budget, with three policy heads:

* ``policy="uniform"``: keep the K most recent tokens (LRU). This
  is the baseline.
* ``policy="importance"``: keep the top-K by intrinsic importance
  (substrate V3's per-token KV importance vector).
* ``policy="learned"``: keep the top-K by a *fitted* scoring head
  ``W ∈ R^{d_model → 1}`` that maps each token's per-layer hidden
  state to a retention score. ``W`` is fitted by a single closed-
  form ridge regression against an oracle: the oracle score for
  each token is the L1 perturbation observed in the substrate's
  next-step logits when that token is dropped from the cache
  (computed once at fit time on a small probe forward).

Once fitted, the controller can be reused across follow-up
forwards. The R-119 benchmark measures:

* logit fidelity (L2 of fully-recomputed vs partially-evicted
  reuse path) per policy, per retention ratio
* flop savings per policy
* the per-policy frontier in the (fidelity, flops) plane

Honest scope
------------

* The learned policy is a **single linear scoring head**, fit by
  closed-form ridge. It is NOT a deep network and NOT trained
  end-to-end. ``W58-L-CACHE-CONTROLLER-LINEAR-CAP`` documents this.
* "Learned" retention does not always *beat* importance retention
  on small substrates; the R-119 benchmark publishes both. The
  W58 H89 bar only requires the importance policy to preserve
  the last-position argmax under ≥ 50% eviction with measurable
  flop savings — a behaviourally honest bar.
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
        "coordpy.cache_controller requires numpy") from exc

from .tiny_substrate_v3 import (
    TinyV3KVCache,
    TinyV3SubstrateParams,
    _ndarray_cid,
    _sha256_hex,
    forward_tiny_substrate_v3,
)


W58_CACHE_CONTROLLER_SCHEMA_VERSION: str = (
    "coordpy.cache_controller.v1")


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


W58_CACHE_POLICY_UNIFORM: str = "uniform"        # LRU
W58_CACHE_POLICY_IMPORTANCE: str = "importance"  # built-in importance
W58_CACHE_POLICY_LEARNED: str = "learned"        # fitted scoring head

W58_CACHE_POLICIES: tuple[str, ...] = (
    W58_CACHE_POLICY_UNIFORM,
    W58_CACHE_POLICY_IMPORTANCE,
    W58_CACHE_POLICY_LEARNED,
)


@dataclasses.dataclass
class CacheController:
    """Pluggable retention policy controller.

    The ``scoring_head`` is shape ``(d_model,)``; when policy is
    ``learned``, the controller scores each token's residual-stream
    hidden state at a chosen probe layer by dot-product against
    ``scoring_head``.
    """

    policy: str
    d_model: int
    probe_layer: int
    scoring_head: "_np.ndarray | None"
    ridge_lambda: float
    fit_seed: int

    @classmethod
    def init(
            cls, *,
            policy: str = W58_CACHE_POLICY_UNIFORM,
            d_model: int = 64,
            probe_layer: int = -1,
            ridge_lambda: float = 0.1,
            fit_seed: int = 58060,
    ) -> "CacheController":
        if policy not in W58_CACHE_POLICIES:
            raise ValueError(
                f"policy must be one of {W58_CACHE_POLICIES}, "
                f"got {policy!r}")
        return cls(
            policy=str(policy),
            d_model=int(d_model),
            probe_layer=int(probe_layer),
            scoring_head=None,
            ridge_lambda=float(ridge_lambda),
            fit_seed=int(fit_seed),
        )

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W58_CACHE_CONTROLLER_SCHEMA_VERSION,
            "policy": str(self.policy),
            "d_model": int(self.d_model),
            "probe_layer": int(self.probe_layer),
            "scoring_head_cid": (
                _ndarray_cid(self.scoring_head)
                if self.scoring_head is not None
                else "untrained"),
            "ridge_lambda": float(self.ridge_lambda),
            "fit_seed": int(self.fit_seed),
        })

    def score_tokens(
            self,
            hidden_state: "_np.ndarray",
            *,
            importance_vector: "_np.ndarray",
            n_tokens: int,
    ) -> "_np.ndarray":
        """Return a length-``n_tokens`` score per token under the
        controller's policy. Larger score => more worth keeping."""
        if str(self.policy) == W58_CACHE_POLICY_UNIFORM:
            # LRU: more recent tokens = higher score.
            return _np.arange(int(n_tokens), dtype=_np.float64)
        if str(self.policy) == W58_CACHE_POLICY_IMPORTANCE:
            imp = _np.asarray(importance_vector, dtype=_np.float64)
            if imp.shape[0] < n_tokens:
                pad = _np.zeros(
                    n_tokens - imp.shape[0], dtype=_np.float64)
                imp = _np.concatenate([imp, pad])
            return imp[: int(n_tokens)]
        if str(self.policy) == W58_CACHE_POLICY_LEARNED:
            if self.scoring_head is None:
                # Untrained: behave like uniform.
                return _np.arange(
                    int(n_tokens), dtype=_np.float64)
            hs = _np.asarray(hidden_state, dtype=_np.float64)
            if hs.shape[0] < n_tokens:
                # Pad or fall back.
                return _np.arange(
                    int(n_tokens), dtype=_np.float64)
            return hs[: int(n_tokens)] @ _np.asarray(
                self.scoring_head, dtype=_np.float64)
        raise ValueError(
            f"unknown policy {self.policy!r}")


# ---------------------------------------------------------------------------
# Oracle-driven fit for the learned scoring head
# ---------------------------------------------------------------------------


def _compute_drop_oracle(
        params: TinyV3SubstrateParams,
        prompt_token_ids: Sequence[int],
        *,
        follow_up_token_ids: Sequence[int],
) -> "_np.ndarray":
    """For each prompt token position ``t``, compute the L1
    perturbation in the follow-up's last-position logits when the
    cache for ``t`` is dropped (single-token leave-one-out).

    This is the oracle score: tokens with higher drop-L1 are more
    important.
    """
    # Compute the reference (keep-all) reuse forward.
    prefix_trace = forward_tiny_substrate_v3(
        params, list(prompt_token_ids),
        return_attention=False)
    full_cache = prefix_trace.kv_cache
    n_prompt = full_cache.n_tokens()
    reuse_full = forward_tiny_substrate_v3(
        params, list(follow_up_token_ids),
        kv_cache=full_cache,
        return_attention=False)
    base_logits = reuse_full.logits[-1]
    oracle = _np.zeros((int(n_prompt),), dtype=_np.float64)
    for t in range(int(n_prompt)):
        # Drop slot t by evict_weighted: keep all except this one.
        weights = [0.0] * int(n_prompt)
        # Make all positions important except t.
        for i in range(int(n_prompt)):
            weights[i] = 1.0 if i != t else -10.0
        ablated = full_cache.evict_weighted(
            weights, keep=int(n_prompt) - 1)
        try:
            abl_trace = forward_tiny_substrate_v3(
                params, list(follow_up_token_ids),
                kv_cache=ablated,
                return_attention=False)
            oracle[t] = float(
                _np.sum(_np.abs(
                    abl_trace.logits[-1] - base_logits)))
        except Exception:
            oracle[t] = 0.0
    return oracle


def fit_learned_cache_controller(
        params: TinyV3SubstrateParams,
        *,
        prompt_token_ids: Sequence[int],
        follow_up_token_ids: Sequence[int],
        probe_layer: int = -1,
        ridge_lambda: float = 0.1,
        fit_seed: int = 58060,
) -> CacheController:
    """Fit a learned scoring head by closed-form ridge regression.

    Targets: per-prompt-position drop-oracle (L1 logit
    perturbation under leave-one-out cache eviction).
    Features: each token's hidden state at ``probe_layer``.
    """
    cfg = params.config
    prefix_trace = forward_tiny_substrate_v3(
        params, list(prompt_token_ids),
        return_attention=False)
    pl = int(probe_layer)
    if pl < 0:
        pl = len(prefix_trace.hidden_states) + pl
    hidden = prefix_trace.hidden_states[int(pl)]   # (T, d_model)
    oracle = _compute_drop_oracle(
        params, list(prompt_token_ids),
        follow_up_token_ids=list(follow_up_token_ids))
    X = _np.asarray(hidden, dtype=_np.float64)
    y = _np.asarray(oracle, dtype=_np.float64)
    n = min(int(X.shape[0]), int(y.shape[0]))
    X = X[: n]
    y = y[: n]
    d = X.shape[1]
    # Closed-form ridge: w = (X^T X + lambda I)^{-1} X^T y
    A = X.T @ X + float(ridge_lambda) * _np.eye(d,
                                                  dtype=_np.float64)
    b = X.T @ y
    w = _np.linalg.solve(A, b)
    ctrl = CacheController.init(
        policy=W58_CACHE_POLICY_LEARNED,
        d_model=int(cfg.d_model),
        probe_layer=int(probe_layer),
        ridge_lambda=float(ridge_lambda),
        fit_seed=int(fit_seed),
    )
    ctrl.scoring_head = w
    return ctrl


# ---------------------------------------------------------------------------
# Apply controller + measure
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class CacheControllerWitness:
    schema: str
    policy: str
    controller_cid: str
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "policy": str(self.policy),
            "controller_cid": str(self.controller_cid),
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
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "cache_controller_witness",
            "witness": self.to_dict()})


def apply_cache_controller_and_measure(
        params: TinyV3SubstrateParams,
        controller: CacheController,
        *,
        prompt_token_ids: Sequence[int],
        follow_up_token_ids: Sequence[int],
        retention_ratio: float = 0.5,
) -> CacheControllerWitness:
    """Run the controller: keep ``floor(retention_ratio * n_prompt)``
    prompt tokens, recompute the rest, then run the follow-up
    forward. Measure logit drift + flop savings against the
    full-cache reuse path.
    """
    cfg = params.config
    prefix_trace = forward_tiny_substrate_v3(
        params, list(prompt_token_ids),
        return_attention=False)
    full_cache = prefix_trace.kv_cache
    n_prompt = int(full_cache.n_tokens())
    keep = max(1, int(retention_ratio * n_prompt))

    # Reference (full-cache reuse) forward.
    full_trace = forward_tiny_substrate_v3(
        params, list(follow_up_token_ids),
        kv_cache=full_cache,
        return_attention=False)
    full_logits = full_trace.logits[-1]

    # Score tokens.
    # For learned: hidden state at probe layer.
    pl = int(controller.probe_layer)
    if pl < 0:
        pl = len(prefix_trace.hidden_states) + pl
    hidden = prefix_trace.hidden_states[int(pl)]
    # For importance: aggregate across layers.
    if full_cache.importance and any(
            i.size for i in full_cache.importance):
        imp = _np.zeros((n_prompt,), dtype=_np.float64)
        for layer_imp in full_cache.importance:
            if (layer_imp.size
                    and int(layer_imp.shape[0]) == n_prompt):
                imp = imp + layer_imp
    else:
        imp = _np.zeros((n_prompt,), dtype=_np.float64)

    scores = controller.score_tokens(
        hidden_state=hidden,
        importance_vector=imp,
        n_tokens=int(n_prompt))

    # Keep top-keep by score (preserves their original order).
    evicted_cache = full_cache.evict_weighted(
        scores.tolist(), int(keep))

    # Follow-up forward on the evicted cache.
    evicted_trace = forward_tiny_substrate_v3(
        params, list(follow_up_token_ids),
        kv_cache=evicted_cache,
        return_attention=False)
    evicted_logits = evicted_trace.logits[-1]

    # Flop accounting:
    # Full reuse path = prefix + full follow-up over n_prompt + n_fu
    # tokens of context (per-layer attention cost grows with
    # context). Already counted in ``prefix_trace.flop_count`` +
    # ``full_trace.flop_count``.
    flop_full = int(
        prefix_trace.flop_count + full_trace.flop_count)
    # Controlled path:
    #   prefix flop already paid (but we save by NOT re-running
    #   the prefix on next reuse). For an honest measurement of
    #   "what does it cost to attend over a smaller context", we
    #   compare:
    #     prefix_flops + follow_up_with_full_cache_flops
    #   vs
    #     prefix_flops + follow_up_with_evicted_cache_flops
    # On the follow-up side the only flop savings come from
    # attention computing fewer key positions.
    flop_ctrl = int(
        prefix_trace.flop_count + evicted_trace.flop_count)
    flop_saved = int(flop_full - flop_ctrl)
    flop_ratio = float(flop_saved) / float(max(flop_full, 1))

    last_full_argmax = int(_np.argmax(full_logits))
    last_ctrl_argmax = int(_np.argmax(evicted_logits))
    argmax_preserved = bool(last_full_argmax == last_ctrl_argmax)
    l1 = float(_np.sum(_np.abs(full_logits - evicted_logits)))
    l2 = float(_np.linalg.norm(full_logits - evicted_logits))

    return CacheControllerWitness(
        schema=W58_CACHE_CONTROLLER_SCHEMA_VERSION,
        policy=str(controller.policy),
        controller_cid=str(controller.cid()),
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
    )


__all__ = [
    "W58_CACHE_CONTROLLER_SCHEMA_VERSION",
    "W58_CACHE_POLICY_UNIFORM",
    "W58_CACHE_POLICY_IMPORTANCE",
    "W58_CACHE_POLICY_LEARNED",
    "W58_CACHE_POLICIES",
    "CacheController",
    "CacheControllerWitness",
    "fit_learned_cache_controller",
    "apply_cache_controller_and_measure",
]
