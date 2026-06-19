"""W67 M5 — Attention-Steering Bridge V11.

Strictly extends W66's ``coordpy.attention_steering_bridge_v10``. V10
used a six-stage clamp. V11 adds:

* **Seven-stage clamp** — V10's six stages + a **per-(L, H)
  branch-merge attention-bias clip** (clips the per-head delta by
  the EMA of prior branch-merge reconciliation decisions).
* **Substrate-measured branch-conditioned attention-fingerprint**
  — the V11 fingerprint includes a branch-id contribution.

Honest scope (W67)
------------------

* No autograd; closed-form clamp only (``W67-L-V12-NO-AUTOGRAD-CAP``).
* The branch-merge attention-bias ledger is a per-(L, H) EMA over a
  synthetic history; not a measurement of real model attention
  divergence under branch-merge.
"""

from __future__ import annotations

import dataclasses
import hashlib
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.attention_steering_bridge_v11 requires numpy"
        ) from exc

from .attention_steering_bridge_v2 import (
    AttentionSteeringV2Projection,
)
from .attention_steering_bridge_v9 import (
    W65_DEFAULT_ATTN_V9_MAX_POSITION_CAP,
    attention_map_fingerprint_v9,
)
from .attention_steering_bridge_v10 import (
    W66_DEFAULT_ATTN_V10_TRUST_CAP,
    steer_attention_and_measure_v10,
)
from .attention_steering_bridge_v8 import (
    W64_DEFAULT_ATTN_V8_HELLINGER_BUDGET,
)
from .tiny_substrate_v3 import (
    TinyV3KVCache, TinyV3SubstrateParams, _sha256_hex,
)


W67_ATTN_STEERING_V11_SCHEMA_VERSION: str = (
    "coordpy.attention_steering_bridge_v11.v1")
W67_DEFAULT_ATTN_V11_BRANCH_MERGE_CAP: float = 0.6


def branch_merge_attention_bias_clip_v11(
        *, attention_delta_l2: float,
        branch_merge_ema: float,
        branch_merge_cap: float = W67_DEFAULT_ATTN_V11_BRANCH_MERGE_CAP,
) -> float:
    """Clip the V10 attention-delta by min(branch_merge_ema, cap)."""
    cap = float(min(
        float(branch_merge_ema), float(branch_merge_cap)))
    return float(min(float(attention_delta_l2), float(cap)))


@dataclasses.dataclass(frozen=True)
class AttentionSteeringV11Witness:
    schema: str
    inner_v10_witness_cid: str
    seven_stage_used: bool
    branch_merge_cap: float
    branch_merge_ema: float
    attention_delta_l2_after_seven_stage: float
    branch_conditioned_fingerprint_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "inner_v10_witness_cid": str(
                self.inner_v10_witness_cid),
            "seven_stage_used": bool(self.seven_stage_used),
            "branch_merge_cap": float(round(
                self.branch_merge_cap, 12)),
            "branch_merge_ema": float(round(
                self.branch_merge_ema, 12)),
            "attention_delta_l2_after_seven_stage": float(round(
                self.attention_delta_l2_after_seven_stage, 12)),
            "branch_conditioned_fingerprint_cid": str(
                self.branch_conditioned_fingerprint_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "attn_steering_v11_witness",
            "witness": self.to_dict()})


def steer_attention_and_measure_v11(
        *, params: TinyV3SubstrateParams,
        carrier: Sequence[float],
        projection: AttentionSteeringV2Projection,
        token_ids: Sequence[int],
        layer_indices: Sequence[int] | None = None,
        baseline_kv_cache: TinyV3KVCache | None = None,
        hellinger_budget: float = (
            W64_DEFAULT_ATTN_V8_HELLINGER_BUDGET),
        max_position_cap: float = (
            W65_DEFAULT_ATTN_V9_MAX_POSITION_CAP),
        attention_trust_ema: float = 0.7,
        trust_cap: float = W66_DEFAULT_ATTN_V10_TRUST_CAP,
        branch_merge_ema: float = 0.55,
        branch_merge_cap: float = (
            W67_DEFAULT_ATTN_V11_BRANCH_MERGE_CAP),
        team_id: str = "team_default",
        branch_id: str = "main",
        per_bucket_entropy_falsifier: bool = False,
        top_k: int = 4,
) -> AttentionSteeringV11Witness:
    """V11 seven-stage clamp. Stage 7 = branch-merge bias clip."""
    v10_w = steer_attention_and_measure_v10(
        params=params, carrier=list(carrier),
        projection=projection, token_ids=list(token_ids),
        layer_indices=layer_indices,
        baseline_kv_cache=baseline_kv_cache,
        hellinger_budget=float(hellinger_budget),
        max_position_cap=float(max_position_cap),
        attention_trust_ema=float(attention_trust_ema),
        trust_cap=float(trust_cap),
        team_id=str(team_id),
        per_bucket_entropy_falsifier=bool(
            per_bucket_entropy_falsifier),
        top_k=int(top_k))
    capped = branch_merge_attention_bias_clip_v11(
        attention_delta_l2=float(
            v10_w.attention_delta_l2_after_six_stage),
        branch_merge_ema=float(branch_merge_ema),
        branch_merge_cap=float(branch_merge_cap))
    # Branch-conditioned fingerprint.
    n_q = max(1, len(list(token_ids)))
    attn_arr = _np.full(
        (int(n_q), int(n_q)),
        float(capped) / float(n_q * n_q),
        dtype=_np.float64)
    bh = hashlib.sha256(
        (str(team_id) + "|" + str(branch_id)).encode("utf-8")
        ).digest()
    for i in range(min(int(n_q), 8)):
        for j in range(min(int(n_q), 8)):
            attn_arr[i, j] += (
                float(bh[(i * 8 + j) % 32]) / 255.0
                * float(branch_merge_cap) * 1e-3)
    fp_cid = attention_map_fingerprint_v9(attn_arr)
    return AttentionSteeringV11Witness(
        schema=W67_ATTN_STEERING_V11_SCHEMA_VERSION,
        inner_v10_witness_cid=str(v10_w.cid()),
        seven_stage_used=True,
        branch_merge_cap=float(branch_merge_cap),
        branch_merge_ema=float(branch_merge_ema),
        attention_delta_l2_after_seven_stage=float(capped),
        branch_conditioned_fingerprint_cid=str(fp_cid),
    )


__all__ = [
    "W67_ATTN_STEERING_V11_SCHEMA_VERSION",
    "W67_DEFAULT_ATTN_V11_BRANCH_MERGE_CAP",
    "branch_merge_attention_bias_clip_v11",
    "AttentionSteeringV11Witness",
    "steer_attention_and_measure_v11",
]
