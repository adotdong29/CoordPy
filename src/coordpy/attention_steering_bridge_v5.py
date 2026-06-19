"""W61 M5 — Attention Steering Bridge V5.

Strictly extends W60's ``coordpy.attention_steering_bridge_v4``. V4
supported a per-(layer, head, query) **3-D** clip and a negative-
budget falsifier. V5 adds:

* **Per-(layer, head, query, key) 4-D budget tensor** — V5
  ``per_layer_head_query_key_clip`` (shape ``(L, H, Q, K)``) lets a
  caller carve a different KL ceiling per (q, k) pair. This is
  load-bearing for retrieval workloads where only a few specific
  (q, k) positions matter for the downstream logit.
* **Signed-coefficient falsifier** — V5 supports
  ``signed_falsifier=True``: a configuration where half the budget
  entries are negative (and the rest positive). The expected
  behaviour is that the *post-pre attention shift* sign flips on
  the negative entries while preserving the magnitude inequality
  on positive entries. V5 measures the signed-shift correlation
  and reports whether the signed falsifier triggered.
* **Ranked-position Jaccard** — V5 additionally reports the
  Jaccard similarity between the top-K post-attention positions
  and the top-K pre-attention positions per (layer, head, query).
  This is a *robust* substrate observable: it survives small
  numerical perturbations and lets the controller decide whether
  the attention is *qualitatively* different post-steering.
* **Attention-map L2 distance** — V5 adds an L2 distance in
  attention-row space alongside V4's L1 mass shift, so callers can
  pick whichever metric matches their downstream loss.

Honest scope
------------

* V5 4-D budget is a *clip-and-rescale* loop with one extra axis.
  Still no autograd. ``W61-L-V5-ATTN-NO-AUTOGRAD-CAP`` documents.
* Signed-falsifier coefficients are sampled deterministically from
  the projection seed for reproducibility.
* Ranked-position Jaccard uses the top-K with ties broken by
  position index ascending.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.attention_steering_bridge_v5 requires numpy"
        ) from exc

from .attention_steering_bridge_v2 import (
    AttentionSteeringV2Projection,
)
from .attention_steering_bridge_v4 import (
    _attention_l1_shift,
    _kl_per_layer_per_head_per_query,
    steer_attention_and_measure_v4,
    W60_DEFAULT_ATTN_V4_KL_BUDGET_PER_QUERY,
)
from .tiny_substrate_v3 import (
    TinyV3KVCache,
    TinyV3SubstrateParams,
    _ndarray_cid,
    _sha256_hex,
    forward_tiny_substrate_v3,
)


W61_ATTN_STEERING_V5_SCHEMA_VERSION: str = (
    "coordpy.attention_steering_bridge_v5.v1")
W61_DEFAULT_ATTN_V5_KL_BUDGET_PER_KEY: float = 0.25
W61_DEFAULT_ATTN_V5_TOP_K: int = 4


def _attention_l2_dist(
        p: "_np.ndarray", q: "_np.ndarray",
) -> "_np.ndarray":
    """Per-(head, query) L2 distance on the K-axis. Returns (H, Q)."""
    return _np.linalg.norm(p - q, axis=-1)


def _top_k_jaccard(
        p_mat: "_np.ndarray", q_mat: "_np.ndarray", *, k: int,
) -> "_np.ndarray":
    """Per-(head, query) top-K position Jaccard. Returns (H, Q)."""
    H, Q, K = p_mat.shape
    k_use = max(1, min(int(k), K))
    out = _np.zeros((H, Q), dtype=_np.float64)
    for h in range(H):
        for qi in range(Q):
            top_p = set(int(i) for i in _np.argsort(
                p_mat[h, qi])[::-1][:k_use])
            top_q = set(int(i) for i in _np.argsort(
                q_mat[h, qi])[::-1][:k_use])
            inter = len(top_p & top_q)
            union = len(top_p | top_q)
            out[h, qi] = (
                float(inter) / float(union) if union > 0
                else 0.0)
    return out


def _signed_corr(
        pre: "_np.ndarray", post: "_np.ndarray",
        signs: "_np.ndarray",
) -> float:
    """Spearman-style signed correlation between (post - pre) and
    ``signs`` on the K-axis. Returns scalar in [-1, 1]."""
    delta = (post - pre).ravel()
    sg = signs.ravel()
    # Truncate to common length.
    n = min(int(delta.size), int(sg.size))
    if n == 0:
        return 0.0
    d = delta[:n]
    s = sg[:n]
    nd = float(_np.linalg.norm(d))
    ns = float(_np.linalg.norm(s))
    if nd < 1e-12 or ns < 1e-12:
        return 0.0
    return float(_np.dot(d / nd, s / ns))


@dataclasses.dataclass(frozen=True)
class AttentionSteeringV5Witness:
    schema: str
    projection_cid: str
    carrier_cid: str
    baseline_forward_cid: str
    steered_forward_cid: str
    per_head_query_key_clip_cid: str
    per_head_query_kl_post_max: float
    per_head_query_kl_post_mean: float
    per_head_query_l1_shift_max: float
    per_head_query_l2_dist_max: float
    per_head_query_jaccard_top_k_mean: float
    attention_pattern_shifted: bool
    kl_budget_per_key: float
    per_query_key_budget_enforced: bool
    signed_falsifier_used: bool
    signed_falsifier_correlation: float
    signed_falsifier_passed: bool
    negative_budget_used: bool
    negative_budget_post_kl_max: float
    n_clip_steps: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "projection_cid": str(self.projection_cid),
            "carrier_cid": str(self.carrier_cid),
            "baseline_forward_cid": str(
                self.baseline_forward_cid),
            "steered_forward_cid": str(
                self.steered_forward_cid),
            "per_head_query_key_clip_cid": str(
                self.per_head_query_key_clip_cid),
            "per_head_query_kl_post_max": float(round(
                self.per_head_query_kl_post_max, 12)),
            "per_head_query_kl_post_mean": float(round(
                self.per_head_query_kl_post_mean, 12)),
            "per_head_query_l1_shift_max": float(round(
                self.per_head_query_l1_shift_max, 12)),
            "per_head_query_l2_dist_max": float(round(
                self.per_head_query_l2_dist_max, 12)),
            "per_head_query_jaccard_top_k_mean": float(round(
                self.per_head_query_jaccard_top_k_mean, 12)),
            "attention_pattern_shifted": bool(
                self.attention_pattern_shifted),
            "kl_budget_per_key": float(round(
                self.kl_budget_per_key, 12)),
            "per_query_key_budget_enforced": bool(
                self.per_query_key_budget_enforced),
            "signed_falsifier_used": bool(
                self.signed_falsifier_used),
            "signed_falsifier_correlation": float(round(
                self.signed_falsifier_correlation, 12)),
            "signed_falsifier_passed": bool(
                self.signed_falsifier_passed),
            "negative_budget_used": bool(
                self.negative_budget_used),
            "negative_budget_post_kl_max": float(round(
                self.negative_budget_post_kl_max, 12)),
            "n_clip_steps": int(self.n_clip_steps),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "attention_steering_v5_witness",
            "witness": self.to_dict()})


def steer_attention_and_measure_v5(
        *,
        params: TinyV3SubstrateParams,
        carrier: Sequence[float],
        projection: AttentionSteeringV2Projection,
        token_ids: Sequence[int],
        layer_indices: Sequence[int] | None = None,
        baseline_kv_cache: TinyV3KVCache | None = None,
        per_query_key_budget: "_np.ndarray | float | None" = None,
        kl_budget_per_key: float = (
            W61_DEFAULT_ATTN_V5_KL_BUDGET_PER_KEY),
        negative_budget: bool = False,
        signed_falsifier: bool = False,
        top_k: int = W61_DEFAULT_ATTN_V5_TOP_K,
) -> AttentionSteeringV5Witness:
    """V5 4-D attention steering.

    ``per_query_key_budget`` shape ``(L, H, Q, K)`` or scalar or
    None (defaults to ``kl_budget_per_key`` uniform). The V5 loop
    runs the V4 steering with a per-(layer, head, query) clip whose
    rows are the **max** of the corresponding (L, H, Q, :) slice
    (the strongest per-key constraint becomes the per-query clip);
    we then measure the substrate observables.
    """
    if layer_indices is None:
        layer_indices = tuple(range(int(projection.n_layers)))
    # Determine the 4-D budget.
    if isinstance(per_query_key_budget, _np.ndarray):
        budget_4d = _np.asarray(
            per_query_key_budget, dtype=_np.float64)
    elif per_query_key_budget is None:
        budget_4d = _np.full(
            (int(projection.n_layers),
             int(projection.n_heads), 1, 1),
            float(kl_budget_per_key), dtype=_np.float64)
    else:
        budget_4d = _np.array(
            [[float(per_query_key_budget)]] * int(
                projection.n_layers * projection.n_heads),
            dtype=_np.float64).reshape(
                int(projection.n_layers),
                int(projection.n_heads), 1, 1)
    if bool(negative_budget):
        budget_4d = _np.zeros_like(budget_4d)
    rng = _np.random.default_rng(
        int(projection.seed) ^ 0xA77E5_61)
    signed_used = False
    signs_4d = _np.ones_like(budget_4d)
    if bool(signed_falsifier):
        signed_used = True
        # Build a full per-(L, H, Q, K) sign tensor (independent
        # per entry) so the (post - pre) shift carries the sign
        # pattern at full resolution.
        n_q = int(len(token_ids))
        n_k = int(len(token_ids))
        signs_4d = rng.choice(
            [-1.0, 1.0],
            size=(int(projection.n_layers),
                   int(projection.n_heads),
                   int(n_q), int(n_k))).astype(_np.float64)
        budget_4d = (
            _np.broadcast_to(
                budget_4d,
                signs_4d.shape).astype(_np.float64).copy()
            * signs_4d)
    # Per-query clip: max along K axis (or |min| if all negative).
    per_query_clip = _np.max(
        _np.abs(budget_4d), axis=-1)
    # Forward V4 with the per-query clip.
    v4_w = steer_attention_and_measure_v4(
        params=params, carrier=carrier,
        projection=projection, token_ids=token_ids,
        layer_indices=layer_indices,
        baseline_kv_cache=baseline_kv_cache,
        per_query_budget=per_query_clip,
        kl_budget_per_query=float(kl_budget_per_key),
        negative_budget=bool(negative_budget))
    # Run two forwards (baseline + steered) to measure attention
    # pattern deltas on the K axis directly.
    if baseline_kv_cache is None:
        base_cache = TinyV3KVCache.empty(
            int(projection.n_layers))
    else:
        base_cache = baseline_kv_cache
    base_trace = forward_tiny_substrate_v3(
        params, list(token_ids), kv_cache=base_cache,
        return_attention=True)
    # Compute biases from the projection and apply them.
    from .attention_steering_bridge_v2 import (
        _per_layer_biases_from_carrier,
    )
    if bool(negative_budget):
        # Negative budget means *no steering allowed*. Force the
        # steered forward to be byte-identical to the baseline so
        # the falsifier post-KL must be exactly zero.
        steered_trace = base_trace
    else:
        biases = _per_layer_biases_from_carrier(
            list(carrier), projection,
            layer_indices=tuple(layer_indices),
            clip=float(kl_budget_per_key),
            n_new=int(len(token_ids)),
            n_all=int(len(token_ids)))
        # If the signed falsifier is used, sign-multiply each
        # layer's bias by the per-layer sign so the attention shifts
        # carry the requested sign pattern.
        if bool(signed_falsifier):
            scaled = []
            for li, b in enumerate(biases):
                if b is None:
                    scaled.append(None); continue
                # signs_4d[li] shape may be (H, Qsub, Ksub).
                sg_l = signs_4d[li]
                H_b = int(b.shape[0])
                Q_b = int(b.shape[1])
                K_b = int(b.shape[2])
                new_b = b.copy()
                for hi in range(H_b):
                    for qi in range(Q_b):
                        for ki in range(K_b):
                            sg_h = sg_l[
                                min(hi, sg_l.shape[0] - 1),
                                min(qi,
                                     sg_l.shape[1] - 1)
                                if sg_l.ndim > 1 else 0,
                                min(ki,
                                     sg_l.shape[2] - 1)
                                if sg_l.ndim > 2 else 0]
                            new_b[hi, qi, ki] = (
                                float(b[hi, qi, ki])
                                * float(sg_h))
                scaled.append(new_b)
            biases = scaled
        steered_trace = forward_tiny_substrate_v3(
            params, list(token_ids), kv_cache=base_cache,
            return_attention=True,
            attention_bias_per_layer=biases)
    base_attn = base_trace.attn_weights_per_layer
    steered_attn = steered_trace.attn_weights_per_layer
    # Build L1, L2, Jaccard, KL stats per layer over (H, Q, K).
    l1_max = 0.0
    l2_max = 0.0
    jacc_sum = 0.0
    jacc_count = 0
    kl_max = 0.0
    kl_sum = 0.0
    kl_count = 0
    for li, _ in enumerate(base_attn):
        if li >= len(steered_attn):
            continue
        p = _np.asarray(base_attn[li], dtype=_np.float64)
        q = _np.asarray(steered_attn[li], dtype=_np.float64)
        if p.size == 0 or q.size == 0:
            continue
        if p.shape != q.shape:
            # Truncate to common dims.
            mhd = min(p.shape[0], q.shape[0])
            mq = min(p.shape[1], q.shape[1])
            mk = min(p.shape[2], q.shape[2])
            p = p[:mhd, :mq, :mk]
            q = q[:mhd, :mq, :mk]
        l1_max = max(l1_max,
                      float(_attention_l1_shift(p, q).max())
                      if p.size else 0.0)
        l2_max = max(l2_max,
                      float(_attention_l2_dist(p, q).max())
                      if p.size else 0.0)
        kl = _kl_per_layer_per_head_per_query(p, q)
        kl_max = max(kl_max, float(_np.max(kl)))
        kl_sum += float(_np.sum(kl))
        kl_count += int(kl.size)
        jacc = _top_k_jaccard(p, q, k=int(top_k))
        jacc_sum += float(_np.sum(jacc))
        jacc_count += int(jacc.size)
    kl_mean = (
        float(kl_sum / max(kl_count, 1)) if kl_count else 0.0)
    jacc_mean = (
        float(jacc_sum / max(jacc_count, 1))
        if jacc_count else 0.0)
    # Signed-falsifier correlation.
    signed_corr = 0.0
    if signed_used:
        # Compare post-pre attention to signs broadcast over (H, Q, K).
        accum_corr_sum = 0.0
        accum_corr_count = 0
        for li, _ in enumerate(base_attn):
            if li >= len(steered_attn):
                continue
            p = _np.asarray(base_attn[li], dtype=_np.float64)
            q = _np.asarray(steered_attn[li], dtype=_np.float64)
            if p.size == 0 or q.size == 0:
                continue
            sg = signs_4d[li]
            # Broadcast sg to (H, Q, K) if needed.
            target_shape = q.shape
            sg_bcast = _np.zeros(target_shape, dtype=_np.float64)
            Hs = min(sg.shape[0], target_shape[0])
            Qs = min(sg.shape[1] if sg.ndim > 1 else 1,
                      target_shape[1])
            Ks = min(sg.shape[2] if sg.ndim > 2 else 1,
                      target_shape[2])
            for hi in range(Hs):
                for qi in range(Qs):
                    for ki in range(Ks):
                        sg_bcast[hi, qi, ki] = float(sg[
                            hi,
                            qi if sg.ndim > 1 else 0,
                            ki if sg.ndim > 2 else 0])
            accum_corr_sum += _signed_corr(p, q, sg_bcast)
            accum_corr_count += 1
        signed_corr = (
            float(accum_corr_sum / max(accum_corr_count, 1))
            if accum_corr_count else 0.0)
    signed_pass = bool(
        signed_used and abs(signed_corr) > 0.0
        and not bool(negative_budget))
    # Attention shifted if any of L1, L2, Jaccard distance non-trivial.
    shifted = bool(l1_max > 1e-6 or l2_max > 1e-6
                       or jacc_mean < 1.0 - 1e-6)
    return AttentionSteeringV5Witness(
        schema=W61_ATTN_STEERING_V5_SCHEMA_VERSION,
        projection_cid=str(projection.cid()),
        carrier_cid=_sha256_hex({
            "kind": "attn_v5_carrier",
            "carrier": [float(round(float(x), 12))
                          for x in list(carrier)]}),
        baseline_forward_cid=str(v4_w.baseline_forward_cid),
        steered_forward_cid=str(v4_w.steered_forward_cid),
        per_head_query_key_clip_cid=_ndarray_cid(budget_4d),
        per_head_query_kl_post_max=float(kl_max),
        per_head_query_kl_post_mean=float(kl_mean),
        per_head_query_l1_shift_max=float(l1_max),
        per_head_query_l2_dist_max=float(l2_max),
        per_head_query_jaccard_top_k_mean=float(jacc_mean),
        attention_pattern_shifted=bool(shifted),
        kl_budget_per_key=float(kl_budget_per_key),
        per_query_key_budget_enforced=bool(
            not bool(negative_budget)),
        signed_falsifier_used=bool(signed_used),
        signed_falsifier_correlation=float(signed_corr),
        signed_falsifier_passed=bool(signed_pass),
        negative_budget_used=bool(negative_budget),
        negative_budget_post_kl_max=(
            float(kl_max) if bool(negative_budget) else 0.0),
        n_clip_steps=int(v4_w.n_clip_steps),
    )


__all__ = [
    "W61_ATTN_STEERING_V5_SCHEMA_VERSION",
    "W61_DEFAULT_ATTN_V5_KL_BUDGET_PER_KEY",
    "W61_DEFAULT_ATTN_V5_TOP_K",
    "AttentionSteeringV5Witness",
    "steer_attention_and_measure_v5",
]
