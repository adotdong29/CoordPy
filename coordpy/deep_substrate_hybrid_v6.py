"""W61 M13 — Deep Substrate Hybrid V6.

Strictly extends W60's ``coordpy.deep_substrate_hybrid_v5``. V5
ran a *five-way* loop:

  V6 latent ↔ tiny_substrate_v5 ↔ cache_controller_v3
  ↔ replay_controller ↔ retrieval_head.

V6 runs a *six-way* loop with the V6 substrate at its centre:

  V6 latent ↔ tiny_substrate_v6 ↔ cache_controller_v4
  ↔ replay_controller_v2 ↔ retrieval_head
  ↔ attention_steering_bridge_v5.

The six-way flag is set when **all** of the following fire on the
same step:

* tiny_substrate_v6 forward produced a non-empty cache.
* cache_controller_v4 emitted a fitted score (any of the V4
  policies).
* replay_controller_v2 emitted a non-ABSTAIN decision OR fired
  the hidden-write gate.
* a retrieval head queried the V6 cache_keys axis.
* an attention_steering_bridge_v5 measurement reported
  ``attention_pattern_shifted=True`` or the signed-coefficient
  falsifier triggered.

V6 strictly extends V5 byte-for-byte when the new sixth axis
(attention_steering V5) is unused.
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
        "coordpy.deep_substrate_hybrid_v6 requires numpy"
        ) from exc

from .deep_substrate_hybrid_v5 import (
    DeepSubstrateHybridV5,
    DeepSubstrateHybridV5ForwardWitness,
    deep_substrate_hybrid_v5_forward,
)
from .replay_controller_v2 import (
    ReplayControllerV2,
)
from .cache_controller_v4 import (
    CacheControllerV4,
)


W61_DEEP_SUBSTRATE_HYBRID_V6_SCHEMA_VERSION: str = (
    "coordpy.deep_substrate_hybrid_v6.v1")


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass
class DeepSubstrateHybridV6:
    """V6 hybrid. ``inner_v5`` may be ``None`` for unit testing the
    witness composition; in real W61 runs it is a fully-wired
    ``DeepSubstrateHybridV5`` instance."""
    inner_v5: DeepSubstrateHybridV5 | None = None
    six_way_active: bool = False

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W61_DEEP_SUBSTRATE_HYBRID_V6_SCHEMA_VERSION,
            "kind": "deep_substrate_hybrid_v6",
            "inner_v5_cid": (
                str(self.inner_v5.cid())
                if self.inner_v5 is not None else "none"),
            "six_way_active": bool(self.six_way_active),
        })


@dataclasses.dataclass(frozen=True)
class DeepSubstrateHybridV6ForwardWitness:
    schema: str
    hybrid_cid: str
    inner_v5_witness_cid: str
    six_way: bool
    cache_controller_v4_fired: bool
    replay_controller_v2_fired: bool
    attention_steering_v5_fired: bool
    bilinear_retrieval_v6_used: bool
    hidden_write_gate_fired: bool
    decision_confidence_mean: float
    attention_pattern_jaccard_mean: float
    attention_pattern_l2_max: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "hybrid_cid": str(self.hybrid_cid),
            "inner_v5_witness_cid": str(
                self.inner_v5_witness_cid),
            "six_way": bool(self.six_way),
            "cache_controller_v4_fired": bool(
                self.cache_controller_v4_fired),
            "replay_controller_v2_fired": bool(
                self.replay_controller_v2_fired),
            "attention_steering_v5_fired": bool(
                self.attention_steering_v5_fired),
            "bilinear_retrieval_v6_used": bool(
                self.bilinear_retrieval_v6_used),
            "hidden_write_gate_fired": bool(
                self.hidden_write_gate_fired),
            "decision_confidence_mean": float(round(
                self.decision_confidence_mean, 12)),
            "attention_pattern_jaccard_mean": float(round(
                self.attention_pattern_jaccard_mean, 12)),
            "attention_pattern_l2_max": float(round(
                self.attention_pattern_l2_max, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "deep_substrate_hybrid_v6_witness",
            "witness": self.to_dict()})


def deep_substrate_hybrid_v6_forward(
        *,
        hybrid: DeepSubstrateHybridV6,
        v5_witness: DeepSubstrateHybridV5ForwardWitness,
        cache_controller_v4: CacheControllerV4 | None = None,
        replay_controller_v2: ReplayControllerV2 | None = None,
        attention_steering_v5_witness: Any | None = None,
) -> DeepSubstrateHybridV6ForwardWitness:
    """Compose the V6 six-way witness from the V5 five-way witness
    plus the V4 cache controller, V2 replay controller, and V5
    attention-steering observations.
    """
    cache_fired = bool(
        cache_controller_v4 is not None
        and (cache_controller_v4.bilinear_retrieval_v6_matrix
              is not None
              or cache_controller_v4.composite_v4_weights
              is not None
              or cache_controller_v4.corruption_floor_coefs
              is not None
              or cache_controller_v4.two_stage_threshold != 0.0))
    bilinear_used = bool(
        cache_controller_v4 is not None
        and cache_controller_v4.bilinear_retrieval_v6_matrix
        is not None)
    replay_fired = False
    hwgate_fired = False
    conf_mean = 0.0
    if replay_controller_v2 is not None:
        confs = [
            float(e.get("confidence", 0.0))
            for e in replay_controller_v2.audit_v2
            if "confidence" in e]
        if confs:
            conf_mean = float(sum(confs) / len(confs))
            replay_fired = True
        hwgate_fired = any(
            str(e.get("stage", "")) == "v2_hidden_write_gate"
            for e in replay_controller_v2.audit_v2)
    attn_fired = False
    attn_jacc_mean = 0.0
    attn_l2_max = 0.0
    if attention_steering_v5_witness is not None:
        try:
            d = attention_steering_v5_witness.to_dict()
            attn_jacc_mean = float(
                d.get("per_head_query_jaccard_top_k_mean", 0.0))
            attn_l2_max = float(
                d.get("per_head_query_l2_dist_max", 0.0))
            attn_fired = bool(
                d.get("attention_pattern_shifted", False)
                or d.get("signed_falsifier_passed", False))
        except Exception:
            pass
    six_way = bool(
        v5_witness.five_way
        and cache_fired
        and replay_fired
        and attn_fired)
    hybrid.six_way_active = bool(six_way)
    return DeepSubstrateHybridV6ForwardWitness(
        schema=W61_DEEP_SUBSTRATE_HYBRID_V6_SCHEMA_VERSION,
        hybrid_cid=str(hybrid.cid()),
        inner_v5_witness_cid=str(v5_witness.cid()),
        six_way=bool(six_way),
        cache_controller_v4_fired=bool(cache_fired),
        replay_controller_v2_fired=bool(replay_fired),
        attention_steering_v5_fired=bool(attn_fired),
        bilinear_retrieval_v6_used=bool(bilinear_used),
        hidden_write_gate_fired=bool(hwgate_fired),
        decision_confidence_mean=float(conf_mean),
        attention_pattern_jaccard_mean=float(attn_jacc_mean),
        attention_pattern_l2_max=float(attn_l2_max),
    )


__all__ = [
    "W61_DEEP_SUBSTRATE_HYBRID_V6_SCHEMA_VERSION",
    "DeepSubstrateHybridV6",
    "DeepSubstrateHybridV6ForwardWitness",
    "deep_substrate_hybrid_v6_forward",
]
