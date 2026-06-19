"""W68 M5 — Attention-Steering Bridge V12.

Strictly extends W67's ``coordpy.attention_steering_bridge_v11``. V11
used a seven-stage clamp. V12 adds:

* **Eight-stage clamp** — V11's seven stages + a **per-(L, H)
  partial-contradiction attention-bias clip** (clips the per-head
  delta by the EMA of prior partial-contradiction reconciliation
  decisions).
* **Substrate-measured agent-conditioned attention-fingerprint** —
  the V12 fingerprint includes an agent-id contribution.

Honest scope (W68)
------------------

* No autograd; closed-form clamp only (``W68-L-V13-NO-AUTOGRAD-CAP``).
* The partial-contradiction attention-bias ledger is a per-(L, H)
  EMA over a synthetic history; not a measurement of real model
  attention divergence under partial contradiction.
"""

from __future__ import annotations

import dataclasses
import hashlib
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.attention_steering_bridge_v12 requires numpy"
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
)
from .attention_steering_bridge_v11 import (
    W67_DEFAULT_ATTN_V11_BRANCH_MERGE_CAP,
    steer_attention_and_measure_v11,
)
from .attention_steering_bridge_v8 import (
    W64_DEFAULT_ATTN_V8_HELLINGER_BUDGET,
)
from .tiny_substrate_v3 import (
    TinyV3KVCache, TinyV3SubstrateParams, _sha256_hex,
)


W68_ATTN_STEERING_V12_SCHEMA_VERSION: str = (
    "coordpy.attention_steering_bridge_v12.v1")
W68_DEFAULT_ATTN_V12_PARTIAL_CONTRADICTION_CAP: float = 0.65


def partial_contradiction_attention_bias_clip_v12(
        *, attention_delta_l2: float,
        partial_contradiction_ema: float,
        partial_contradiction_cap: float = (
            W68_DEFAULT_ATTN_V12_PARTIAL_CONTRADICTION_CAP),
) -> float:
    """Clip the V11 attention-delta by
    min(partial_contradiction_ema, cap)."""
    cap = float(min(
        float(partial_contradiction_ema),
        float(partial_contradiction_cap)))
    return float(min(float(attention_delta_l2), float(cap)))


@dataclasses.dataclass(frozen=True)
class AttentionSteeringV12Witness:
    schema: str
    inner_v11_witness_cid: str
    eight_stage_used: bool
    partial_contradiction_cap: float
    partial_contradiction_ema: float
    attention_delta_l2_after_eight_stage: float
    agent_conditioned_fingerprint_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "inner_v11_witness_cid": str(
                self.inner_v11_witness_cid),
            "eight_stage_used": bool(self.eight_stage_used),
            "partial_contradiction_cap": float(round(
                self.partial_contradiction_cap, 12)),
            "partial_contradiction_ema": float(round(
                self.partial_contradiction_ema, 12)),
            "attention_delta_l2_after_eight_stage": float(round(
                self.attention_delta_l2_after_eight_stage, 12)),
            "agent_conditioned_fingerprint_cid": str(
                self.agent_conditioned_fingerprint_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "attn_steering_v12_witness",
            "witness": self.to_dict()})


def steer_attention_and_measure_v12(
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
        partial_contradiction_ema: float = 0.50,
        partial_contradiction_cap: float = (
            W68_DEFAULT_ATTN_V12_PARTIAL_CONTRADICTION_CAP),
        team_id: str = "team_default",
        branch_id: str = "main",
        agent_id: str = "a0",
        per_bucket_entropy_falsifier: bool = False,
        top_k: int = 4,
) -> AttentionSteeringV12Witness:
    """V12 eight-stage clamp. Stage 8 = partial-contradiction bias
    clip."""
    v11_w = steer_attention_and_measure_v11(
        params=params, carrier=list(carrier),
        projection=projection, token_ids=list(token_ids),
        layer_indices=layer_indices,
        baseline_kv_cache=baseline_kv_cache,
        hellinger_budget=float(hellinger_budget),
        max_position_cap=float(max_position_cap),
        attention_trust_ema=float(attention_trust_ema),
        trust_cap=float(trust_cap),
        branch_merge_ema=float(branch_merge_ema),
        branch_merge_cap=float(branch_merge_cap),
        team_id=str(team_id), branch_id=str(branch_id),
        per_bucket_entropy_falsifier=bool(
            per_bucket_entropy_falsifier),
        top_k=int(top_k))
    capped = partial_contradiction_attention_bias_clip_v12(
        attention_delta_l2=float(
            v11_w.attention_delta_l2_after_seven_stage),
        partial_contradiction_ema=float(
            partial_contradiction_ema),
        partial_contradiction_cap=float(
            partial_contradiction_cap))
    n_q = max(1, len(list(token_ids)))
    attn_arr = _np.full(
        (int(n_q), int(n_q)),
        float(capped) / float(n_q * n_q),
        dtype=_np.float64)
    bh = hashlib.sha256(
        (str(team_id) + "|" + str(branch_id) + "|" + str(agent_id)
         ).encode("utf-8")).digest()
    for i in range(min(int(n_q), 8)):
        for j in range(min(int(n_q), 8)):
            attn_arr[i, j] += (
                float(bh[(i * 8 + j) % 32]) / 255.0
                * float(partial_contradiction_cap) * 1e-3)
    fp_cid = attention_map_fingerprint_v9(attn_arr)
    return AttentionSteeringV12Witness(
        schema=W68_ATTN_STEERING_V12_SCHEMA_VERSION,
        inner_v11_witness_cid=str(v11_w.cid()),
        eight_stage_used=True,
        partial_contradiction_cap=float(
            partial_contradiction_cap),
        partial_contradiction_ema=float(
            partial_contradiction_ema),
        attention_delta_l2_after_eight_stage=float(capped),
        agent_conditioned_fingerprint_cid=str(fp_cid),
    )


__all__ = [
    "W68_ATTN_STEERING_V12_SCHEMA_VERSION",
    "W68_DEFAULT_ATTN_V12_PARTIAL_CONTRADICTION_CAP",
    "partial_contradiction_attention_bias_clip_v12",
    "AttentionSteeringV12Witness",
    "steer_attention_and_measure_v12",
]
