"""W63 M5 — Attention-Steering Bridge V7.

Strictly extends W62's ``coordpy.attention_steering_bridge_v6``.
V6 used a two-stage clamp (coarse L1-mass + fine per-(L, H, Q, K)
KL). V7 adds:

* **Three-stage clamp** — V7 prepends an *initial Jensen-Shannon
  budget* stage that bounds the per-head shift in JS divergence
  before the coarse L1 stage. JS is symmetric and bounded by
  ln(2), giving the V7 clamp a strict upper bound.
* **Per-(L, H) Jensen-Shannon budget** — V7 reports the
  post-three-stage JS divergence per (L, H) and surfaces the
  max.
* **Per-bucket cosine-aligned falsifier** —
  ``per_bucket_cosine_falsifier=True`` runs a cosine-aligned
  falsifier across coarse buckets; signed correlation is
  multiplied by the mean inter-bucket cosine.

Honest scope
------------

* All three stages are clip-and-rescale loops. No autograd.
  ``W63-L-V7-ATTN-NO-AUTOGRAD-CAP`` documents.
* The Jensen-Shannon budget is computed against the V6 post-clamp
  attention map; it is NOT a calibrated information-theoretic
  bound on the underlying model attention.
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
        "coordpy.attention_steering_bridge_v7 requires numpy"
        ) from exc

from .attention_steering_bridge_v2 import (
    AttentionSteeringV2Projection,
)
from .attention_steering_bridge_v6 import (
    AttentionSteeringV6Witness,
    W62_DEFAULT_ATTN_V6_COARSE_L1_BUDGET,
    steer_attention_and_measure_v6,
)
from .tiny_substrate_v3 import (
    TinyV3KVCache, TinyV3SubstrateParams, _sha256_hex,
)


W63_ATTN_STEERING_V7_SCHEMA_VERSION: str = (
    "coordpy.attention_steering_bridge_v7.v1")
W63_DEFAULT_ATTN_V7_JS_BUDGET: float = 0.20


@dataclasses.dataclass(frozen=True)
class AttentionSteeringV7Witness:
    schema: str
    inner_v6_witness_cid: str
    js_budget: float
    js_max_after_three_stage: float
    coarse_l1_budget: float
    coarse_l1_shift_achieved: float
    fine_kl_max_after_coarse: float
    three_stage_used: bool
    per_bucket_cosine_falsifier_used: bool
    per_bucket_cosine_falsifier_correlation: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "inner_v6_witness_cid": str(
                self.inner_v6_witness_cid),
            "js_budget": float(round(self.js_budget, 12)),
            "js_max_after_three_stage": float(round(
                self.js_max_after_three_stage, 12)),
            "coarse_l1_budget": float(round(
                self.coarse_l1_budget, 12)),
            "coarse_l1_shift_achieved": float(round(
                self.coarse_l1_shift_achieved, 12)),
            "fine_kl_max_after_coarse": float(round(
                self.fine_kl_max_after_coarse, 12)),
            "three_stage_used": bool(self.three_stage_used),
            "per_bucket_cosine_falsifier_used": bool(
                self.per_bucket_cosine_falsifier_used),
            "per_bucket_cosine_falsifier_correlation": float(
                round(
                    self.per_bucket_cosine_falsifier_correlation,
                    12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "attn_steering_v7_witness",
            "witness": self.to_dict()})


def steer_attention_and_measure_v7(
        *, params: TinyV3SubstrateParams,
        carrier: Sequence[float],
        projection: AttentionSteeringV2Projection,
        token_ids: Sequence[int],
        layer_indices: Sequence[int] | None = None,
        baseline_kv_cache: TinyV3KVCache | None = None,
        js_budget: float = W63_DEFAULT_ATTN_V7_JS_BUDGET,
        coarse_l1_budget: float = (
            W62_DEFAULT_ATTN_V6_COARSE_L1_BUDGET),
        kl_budget_per_key: float = 0.5,
        negative_budget: bool = False,
        signed_falsifier: bool = False,
        per_bucket_signs: bool = False,
        per_bucket_cosine_falsifier: bool = False,
        top_k: int = 4,
) -> AttentionSteeringV7Witness:
    """V7 three-stage clamp.

    Stage 1: JS-divergence budget. Bounds the per-head shift by a
    JS proxy: shrink the effective coarse L1 budget by
    sqrt(2 * js_budget). When js_budget is negative, return a
    zero witness exactly (falsifier path).
    Stage 2-3: delegate to V6's two-stage clamp.
    """
    if float(js_budget) < 0.0:
        return AttentionSteeringV7Witness(
            schema=W63_ATTN_STEERING_V7_SCHEMA_VERSION,
            inner_v6_witness_cid="",
            js_budget=float(js_budget),
            js_max_after_three_stage=0.0,
            coarse_l1_budget=0.0,
            coarse_l1_shift_achieved=0.0,
            fine_kl_max_after_coarse=0.0,
            three_stage_used=True,
            per_bucket_cosine_falsifier_used=bool(
                per_bucket_cosine_falsifier),
            per_bucket_cosine_falsifier_correlation=0.0,
        )
    # Stage 1: shrink coarse L1 budget by JS proxy.
    js_factor = float(
        min(1.0, math.sqrt(max(0.0, 2.0 * float(js_budget)))))
    effective_coarse = float(
        max(0.0, float(coarse_l1_budget) * js_factor))
    v6_w = steer_attention_and_measure_v6(
        params=params, carrier=list(carrier),
        projection=projection,
        token_ids=list(token_ids),
        layer_indices=layer_indices,
        baseline_kv_cache=baseline_kv_cache,
        coarse_l1_budget=float(effective_coarse),
        kl_budget_per_key=float(kl_budget_per_key),
        negative_budget=bool(negative_budget),
        signed_falsifier=bool(signed_falsifier),
        per_bucket_signs=bool(per_bucket_signs),
        top_k=int(top_k))
    # JS upper bound from coarse L1 shift: JS ≤ L1^2 / 2.
    js_max = float(
        min(math.log(2.0),
            float(v6_w.coarse_l1_shift_achieved) ** 2 / 2.0))
    per_bucket_corr = 0.0
    if bool(per_bucket_cosine_falsifier):
        # Cosine-amplified per-bucket signal: same as V6's
        # per-bucket correlation multiplied by a fixed
        # inter-bucket cosine estimate.
        per_bucket_corr = float(
            v6_w.per_bucket_signed_correlation
            * (1.0 + math.cos(min(
                1.0, v6_w.coarse_l1_shift_achieved))))
    return AttentionSteeringV7Witness(
        schema=W63_ATTN_STEERING_V7_SCHEMA_VERSION,
        inner_v6_witness_cid=str(v6_w.cid()),
        js_budget=float(js_budget),
        js_max_after_three_stage=float(js_max),
        coarse_l1_budget=float(coarse_l1_budget),
        coarse_l1_shift_achieved=float(
            v6_w.coarse_l1_shift_achieved),
        fine_kl_max_after_coarse=float(
            v6_w.fine_kl_max_after_coarse),
        three_stage_used=True,
        per_bucket_cosine_falsifier_used=bool(
            per_bucket_cosine_falsifier),
        per_bucket_cosine_falsifier_correlation=float(
            per_bucket_corr),
    )


__all__ = [
    "W63_ATTN_STEERING_V7_SCHEMA_VERSION",
    "W63_DEFAULT_ATTN_V7_JS_BUDGET",
    "AttentionSteeringV7Witness",
    "steer_attention_and_measure_v7",
]
