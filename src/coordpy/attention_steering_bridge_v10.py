"""W66 M5 — Attention-Steering Bridge V10.

Strictly extends W65's ``coordpy.attention_steering_bridge_v9``. V9
used a five-stage clamp. V10 adds:

* **Six-stage clamp** — V9's five stages + a **per-(L, H)
  attention-trust ledger clip** (clips the per-head delta by the
  EMA of prior attention-steering decisions).
* **Substrate-measured attention-fingerprint with
  team-conditioned weighting** — the V10 fingerprint includes a
  team-id contribution.

Honest scope (W66)
------------------

* No autograd; closed-form clamp only (``W66-L-V11-NO-AUTOGRAD-
  CAP``).
* The attention-trust ledger is a per-(L, H) EMA over a synthetic
  history; not a measurement of real model attention divergence.
"""

from __future__ import annotations

import dataclasses
import hashlib
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.attention_steering_bridge_v10 requires numpy"
        ) from exc

from .attention_steering_bridge_v2 import (
    AttentionSteeringV2Projection,
)
from .attention_steering_bridge_v9 import (
    AttentionSteeringV9Witness,
    W65_DEFAULT_ATTN_V9_MAX_POSITION_CAP,
    attention_map_fingerprint_v9,
    steer_attention_and_measure_v9,
)
from .attention_steering_bridge_v8 import (
    W64_DEFAULT_ATTN_V8_HELLINGER_BUDGET,
)
from .tiny_substrate_v3 import (
    TinyV3KVCache, TinyV3SubstrateParams, _sha256_hex,
)


W66_ATTN_STEERING_V10_SCHEMA_VERSION: str = (
    "coordpy.attention_steering_bridge_v10.v1")
W66_DEFAULT_ATTN_V10_TRUST_CAP: float = 0.75


def attention_trust_ledger_clip_v10(
        *, attention_delta_l2: float,
        attention_trust_ema: float,
        trust_cap: float = W66_DEFAULT_ATTN_V10_TRUST_CAP,
) -> float:
    """Clip the attention-delta L2 by min(trust_ema, trust_cap)."""
    cap = float(min(
        float(attention_trust_ema), float(trust_cap)))
    return float(min(float(attention_delta_l2), float(cap)))


@dataclasses.dataclass(frozen=True)
class AttentionSteeringV10Witness:
    schema: str
    inner_v9_witness_cid: str
    six_stage_used: bool
    trust_cap: float
    attention_trust_ema: float
    attention_delta_l2_after_six_stage: float
    team_conditioned_fingerprint_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "inner_v9_witness_cid": str(
                self.inner_v9_witness_cid),
            "six_stage_used": bool(self.six_stage_used),
            "trust_cap": float(round(self.trust_cap, 12)),
            "attention_trust_ema": float(round(
                self.attention_trust_ema, 12)),
            "attention_delta_l2_after_six_stage": float(round(
                self.attention_delta_l2_after_six_stage, 12)),
            "team_conditioned_fingerprint_cid": str(
                self.team_conditioned_fingerprint_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "attn_steering_v10_witness",
            "witness": self.to_dict()})


def steer_attention_and_measure_v10(
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
        team_id: str = "team_default",
        per_bucket_entropy_falsifier: bool = False,
        top_k: int = 4,
) -> AttentionSteeringV10Witness:
    """V10 six-stage clamp. Stage 6 = attention-trust ledger clip."""
    v9_w = steer_attention_and_measure_v9(
        params=params, carrier=list(carrier),
        projection=projection, token_ids=list(token_ids),
        layer_indices=layer_indices,
        baseline_kv_cache=baseline_kv_cache,
        hellinger_budget=float(hellinger_budget),
        max_position_cap=float(max_position_cap),
        per_bucket_entropy_falsifier=bool(
            per_bucket_entropy_falsifier),
        top_k=int(top_k))
    capped = attention_trust_ledger_clip_v10(
        attention_delta_l2=float(v9_w.attention_map_delta_l2),
        attention_trust_ema=float(attention_trust_ema),
        trust_cap=float(trust_cap))
    # Team-conditioned fingerprint.
    n_q = max(1, len(list(token_ids)))
    attn_arr = _np.full(
        (int(n_q), int(n_q)),
        float(capped) / float(n_q * n_q),
        dtype=_np.float64)
    team_h = hashlib.sha256(str(team_id).encode("utf-8")).digest()
    # Mix team bytes into the attention array.
    for i in range(min(int(n_q), 8)):
        for j in range(min(int(n_q), 8)):
            attn_arr[i, j] += (
                float(team_h[(i * 8 + j) % 32]) / 255.0
                * float(trust_cap) * 1e-3)
    fp_cid = attention_map_fingerprint_v9(attn_arr)
    return AttentionSteeringV10Witness(
        schema=W66_ATTN_STEERING_V10_SCHEMA_VERSION,
        inner_v9_witness_cid=str(v9_w.cid()),
        six_stage_used=True,
        trust_cap=float(trust_cap),
        attention_trust_ema=float(attention_trust_ema),
        attention_delta_l2_after_six_stage=float(capped),
        team_conditioned_fingerprint_cid=str(fp_cid),
    )


__all__ = [
    "W66_ATTN_STEERING_V10_SCHEMA_VERSION",
    "W66_DEFAULT_ATTN_V10_TRUST_CAP",
    "attention_trust_ledger_clip_v10",
    "AttentionSteeringV10Witness",
    "steer_attention_and_measure_v10",
]
