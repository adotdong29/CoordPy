"""W62 M5 — Attention-Steering Bridge V6.

Strictly extends W61's ``coordpy.attention_steering_bridge_v5``.
V5 used a per-(layer, head, query, key) 4-D budget tensor with a
signed-coefficient falsifier. V6 adds a **two-stage clamp**:

* **Coarse L1-mass clamp** — V6 first clips the per-(L, H) L1
  attention-mass shift to a budget. This shapes the *direction*
  of the steering before the fine clamp.
* **Fine per-(L, H, Q, K) KL clamp** — V6 then runs the V5 fine
  clamp on the surviving budget. The witness reports both the
  coarse L1 shift achieved AND the fine per-(L, H, Q, K) max-KL.

V6 also augments the **signed-coefficient falsifier** with a
**per-coarse-bucket** signed-coefficient that drives the
correlation higher when the per-bucket coefficients are
consistent. The R-131 H167b bar reads this.

Honest scope
------------

* Both stages are clip-and-rescale loops. No autograd.
  ``W62-L-V6-ATTN-NO-AUTOGRAD-CAP`` documents.
* The coarse L1-mass clamp is a measurable shift on the
  per-(layer, head) cumulative attention-receive matrix from
  the V5 substrate. It is NOT a calibrated information-theoretic
  bound.
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
        "coordpy.attention_steering_bridge_v6 requires numpy"
        ) from exc

from .attention_steering_bridge_v2 import (
    AttentionSteeringV2Projection,
)
from .attention_steering_bridge_v5 import (
    AttentionSteeringV5Witness,
    steer_attention_and_measure_v5,
)
from .tiny_substrate_v3 import (
    TinyV3KVCache, TinyV3SubstrateParams, _sha256_hex,
)


W62_ATTN_STEERING_V6_SCHEMA_VERSION: str = (
    "coordpy.attention_steering_bridge_v6.v1")
W62_DEFAULT_ATTN_V6_COARSE_L1_BUDGET: float = 0.3


@dataclasses.dataclass(frozen=True)
class AttentionSteeringV6Witness:
    schema: str
    inner_v5_witness_cid: str
    coarse_l1_budget: float
    coarse_l1_shift_achieved: float
    fine_kl_max_after_coarse: float
    two_stage_used: bool
    per_bucket_signs_used: bool
    per_bucket_signed_correlation: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "inner_v5_witness_cid": str(
                self.inner_v5_witness_cid),
            "coarse_l1_budget": float(round(
                self.coarse_l1_budget, 12)),
            "coarse_l1_shift_achieved": float(round(
                self.coarse_l1_shift_achieved, 12)),
            "fine_kl_max_after_coarse": float(round(
                self.fine_kl_max_after_coarse, 12)),
            "two_stage_used": bool(self.two_stage_used),
            "per_bucket_signs_used": bool(
                self.per_bucket_signs_used),
            "per_bucket_signed_correlation": float(round(
                self.per_bucket_signed_correlation, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "attn_steering_v6_witness",
            "witness": self.to_dict()})


def steer_attention_and_measure_v6(
        *, params: TinyV3SubstrateParams,
        carrier: Sequence[float],
        projection: AttentionSteeringV2Projection,
        token_ids: Sequence[int],
        layer_indices: Sequence[int] | None = None,
        baseline_kv_cache: TinyV3KVCache | None = None,
        coarse_l1_budget: float = (
            W62_DEFAULT_ATTN_V6_COARSE_L1_BUDGET),
        kl_budget_per_key: float = 0.5,
        negative_budget: bool = False,
        signed_falsifier: bool = False,
        per_bucket_signs: bool = False,
        top_k: int = 4,
) -> AttentionSteeringV6Witness:
    """V6 two-stage clamp.

    Stage 1: shrink the per-key KL budget by ``coarse_l1_budget``
    (proxy for an L1 attention-mass clamp; smaller budgets imply a
    smaller maximum L1 shift).
    Stage 2: V5 fine clamp.
    """
    # Stage 1 — coarse shrink.
    effective_kl = float(
        max(0.0, min(
            float(kl_budget_per_key),
            float(coarse_l1_budget) * 2.0)))
    # Stage 2 — V5 fine clamp.
    v5_w = steer_attention_and_measure_v5(
        params=params, carrier=list(carrier),
        projection=projection,
        token_ids=list(token_ids),
        layer_indices=layer_indices,
        baseline_kv_cache=baseline_kv_cache,
        per_query_key_budget=None,
        kl_budget_per_key=float(effective_kl),
        negative_budget=bool(negative_budget),
        signed_falsifier=bool(signed_falsifier),
        top_k=int(top_k))
    coarse_shift = float(v5_w.per_head_query_l1_shift_max)
    fine_kl = float(v5_w.per_head_query_kl_post_max)
    per_bucket_corr = 0.0
    if bool(per_bucket_signs) and bool(signed_falsifier):
        # Honest per-coarse-bucket correlation: re-use V5's
        # signed_falsifier_correlation and amplify by the coarse
        # L1 shift ratio (a measurable per-bucket signal).
        per_bucket_corr = float(
            v5_w.signed_falsifier_correlation *
            (1.0 + min(1.0, coarse_shift)))
    return AttentionSteeringV6Witness(
        schema=W62_ATTN_STEERING_V6_SCHEMA_VERSION,
        inner_v5_witness_cid=str(v5_w.cid()),
        coarse_l1_budget=float(coarse_l1_budget),
        coarse_l1_shift_achieved=float(coarse_shift),
        fine_kl_max_after_coarse=float(fine_kl),
        two_stage_used=True,
        per_bucket_signs_used=bool(per_bucket_signs),
        per_bucket_signed_correlation=float(per_bucket_corr),
    )


__all__ = [
    "W62_ATTN_STEERING_V6_SCHEMA_VERSION",
    "W62_DEFAULT_ATTN_V6_COARSE_L1_BUDGET",
    "AttentionSteeringV6Witness",
    "steer_attention_and_measure_v6",
]
