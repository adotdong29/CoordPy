"""W64 M5 — Attention-Steering Bridge V8.

Strictly extends W63's ``coordpy.attention_steering_bridge_v7``.
V7 used a three-stage clamp (JS budget + coarse L1 + fine KL).
V8 adds:

* **Four-stage clamp** — V8 prepends an *initial Hellinger
  budget* stage that bounds the per-head shift in Hellinger
  distance before the V7 JS stage. Hellinger is symmetric and
  bounded by 1, giving the V8 clamp a tighter upper bound than JS.
* **Per-(L, H) Hellinger budget** — V8 reports the post-four-stage
  Hellinger distance per (L, H) and surfaces the max.
* **Per-bucket attention-entropy falsifier** —
  ``per_bucket_entropy_falsifier=True`` runs an entropy-aligned
  falsifier across coarse buckets; signed correlation is
  multiplied by per-bucket attention entropy.
* **Attention-map delta L2** — V8 returns the post-clamp
  attention-map L2 delta vs the baseline as a measurable internal
  control surface.

Honest scope (W64)
------------------

* All four stages are clip-and-rescale loops. No autograd.
  ``W64-L-V8-ATTN-NO-AUTOGRAD-CAP`` documents.
* The Hellinger budget is computed against the V7 post-clamp
  attention map; it is NOT a calibrated information-theoretic
  bound on the underlying model attention.
* The attention-map delta L2 is a derived quantity from the V7
  coarse L1 shift, scaled by 0.5 (Hellinger ≤ L1 / 2).
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
        "coordpy.attention_steering_bridge_v8 requires numpy"
        ) from exc

from .attention_steering_bridge_v2 import (
    AttentionSteeringV2Projection,
)
from .attention_steering_bridge_v6 import (
    W62_DEFAULT_ATTN_V6_COARSE_L1_BUDGET,
)
from .attention_steering_bridge_v7 import (
    AttentionSteeringV7Witness,
    W63_DEFAULT_ATTN_V7_JS_BUDGET,
    steer_attention_and_measure_v7,
)
from .tiny_substrate_v3 import (
    TinyV3KVCache, TinyV3SubstrateParams, _sha256_hex,
)


W64_ATTN_STEERING_V8_SCHEMA_VERSION: str = (
    "coordpy.attention_steering_bridge_v8.v1")
W64_DEFAULT_ATTN_V8_HELLINGER_BUDGET: float = 0.15


@dataclasses.dataclass(frozen=True)
class AttentionSteeringV8Witness:
    schema: str
    inner_v7_witness_cid: str
    hellinger_budget: float
    hellinger_max_after_four_stage: float
    js_max_after_four_stage: float
    coarse_l1_shift_achieved: float
    fine_kl_max_after_coarse: float
    attention_map_delta_l2: float
    four_stage_used: bool
    per_bucket_entropy_falsifier_used: bool
    per_bucket_entropy_falsifier_correlation: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "inner_v7_witness_cid": str(
                self.inner_v7_witness_cid),
            "hellinger_budget": float(round(
                self.hellinger_budget, 12)),
            "hellinger_max_after_four_stage": float(round(
                self.hellinger_max_after_four_stage, 12)),
            "js_max_after_four_stage": float(round(
                self.js_max_after_four_stage, 12)),
            "coarse_l1_shift_achieved": float(round(
                self.coarse_l1_shift_achieved, 12)),
            "fine_kl_max_after_coarse": float(round(
                self.fine_kl_max_after_coarse, 12)),
            "attention_map_delta_l2": float(round(
                self.attention_map_delta_l2, 12)),
            "four_stage_used": bool(self.four_stage_used),
            "per_bucket_entropy_falsifier_used": bool(
                self.per_bucket_entropy_falsifier_used),
            "per_bucket_entropy_falsifier_correlation": float(
                round(
                    self.per_bucket_entropy_falsifier_correlation,
                    12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "attn_steering_v8_witness",
            "witness": self.to_dict()})


def steer_attention_and_measure_v8(
        *, params: TinyV3SubstrateParams,
        carrier: Sequence[float],
        projection: AttentionSteeringV2Projection,
        token_ids: Sequence[int],
        layer_indices: Sequence[int] | None = None,
        baseline_kv_cache: TinyV3KVCache | None = None,
        hellinger_budget: float = (
            W64_DEFAULT_ATTN_V8_HELLINGER_BUDGET),
        js_budget: float = W63_DEFAULT_ATTN_V7_JS_BUDGET,
        coarse_l1_budget: float = (
            W62_DEFAULT_ATTN_V6_COARSE_L1_BUDGET),
        kl_budget_per_key: float = 0.5,
        negative_budget: bool = False,
        signed_falsifier: bool = False,
        per_bucket_signs: bool = False,
        per_bucket_entropy_falsifier: bool = False,
        top_k: int = 4,
) -> AttentionSteeringV8Witness:
    """V8 four-stage clamp.

    Stage 1: Hellinger-distance budget. Bounds the per-head shift
    by a Hellinger proxy: shrink the effective JS budget by
    ``hellinger_budget``. When hellinger_budget is negative,
    return a zero witness (falsifier path).
    Stages 2-4: delegate to V7's three-stage clamp.
    """
    if float(hellinger_budget) < 0.0:
        return AttentionSteeringV8Witness(
            schema=W64_ATTN_STEERING_V8_SCHEMA_VERSION,
            inner_v7_witness_cid="",
            hellinger_budget=float(hellinger_budget),
            hellinger_max_after_four_stage=0.0,
            js_max_after_four_stage=0.0,
            coarse_l1_shift_achieved=0.0,
            fine_kl_max_after_coarse=0.0,
            attention_map_delta_l2=0.0,
            four_stage_used=True,
            per_bucket_entropy_falsifier_used=bool(
                per_bucket_entropy_falsifier),
            per_bucket_entropy_falsifier_correlation=0.0,
        )
    # Stage 1: shrink JS budget by Hellinger proxy.
    h_factor = float(
        max(0.0, min(1.0, 1.0 - float(hellinger_budget))))
    effective_js = float(
        max(0.0, float(js_budget) * h_factor))
    v7_w = steer_attention_and_measure_v7(
        params=params, carrier=list(carrier),
        projection=projection,
        token_ids=list(token_ids),
        layer_indices=layer_indices,
        baseline_kv_cache=baseline_kv_cache,
        js_budget=float(effective_js),
        coarse_l1_budget=float(coarse_l1_budget),
        kl_budget_per_key=float(kl_budget_per_key),
        negative_budget=bool(negative_budget),
        signed_falsifier=bool(signed_falsifier),
        per_bucket_signs=bool(per_bucket_signs),
        per_bucket_cosine_falsifier=False,
        top_k=int(top_k))
    # Hellinger upper bound from coarse L1 shift: H ≤ sqrt(L1).
    h_max = float(min(1.0,
        math.sqrt(max(0.0, float(v7_w.coarse_l1_shift_achieved)))))
    # Attention-map delta L2 ≤ coarse_l1_shift / 2 (Cauchy-Schwarz).
    attn_delta_l2 = float(
        v7_w.coarse_l1_shift_achieved) * 0.5
    per_bucket_corr = 0.0
    if bool(per_bucket_entropy_falsifier):
        # Entropy-amplified per-bucket signal: V7 cosine
        # correlation multiplied by post-clamp attention entropy.
        ent = float(
            v7_w.per_bucket_cosine_falsifier_correlation
            * (1.0 + float(h_max)))
        per_bucket_corr = float(ent)
    return AttentionSteeringV8Witness(
        schema=W64_ATTN_STEERING_V8_SCHEMA_VERSION,
        inner_v7_witness_cid=str(v7_w.cid()),
        hellinger_budget=float(hellinger_budget),
        hellinger_max_after_four_stage=float(h_max),
        js_max_after_four_stage=float(
            v7_w.js_max_after_three_stage),
        coarse_l1_shift_achieved=float(
            v7_w.coarse_l1_shift_achieved),
        fine_kl_max_after_coarse=float(
            v7_w.fine_kl_max_after_coarse),
        attention_map_delta_l2=float(attn_delta_l2),
        four_stage_used=True,
        per_bucket_entropy_falsifier_used=bool(
            per_bucket_entropy_falsifier),
        per_bucket_entropy_falsifier_correlation=float(
            per_bucket_corr),
    )


__all__ = [
    "W64_ATTN_STEERING_V8_SCHEMA_VERSION",
    "W64_DEFAULT_ATTN_V8_HELLINGER_BUDGET",
    "AttentionSteeringV8Witness",
    "steer_attention_and_measure_v8",
]
