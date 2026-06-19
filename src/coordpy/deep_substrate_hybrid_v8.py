"""W63 — Deep Substrate Hybrid V8.

Strictly extends W62's ``coordpy.deep_substrate_hybrid_v7``. V7
ran a *seven-way* loop:

  V7 latent ↔ tiny_substrate_v7 ↔ cache_controller_v5
  ↔ replay_controller_v3 ↔ retrieval_head
  ↔ attention_steering_bridge_v6 ↔ hidden_vs_kv_classifier.

V8 runs an *eight-way* loop with the V8 substrate at its centre:

  V8 latent ↔ tiny_substrate_v8 ↔ cache_controller_v6
  ↔ replay_controller_v4 ↔ retrieval_head
  ↔ attention_steering_bridge_v7 ↔ three_way_bridge_classifier
  ↔ prefix_state_bridge_v7.

The eight-way flag is set when **all eight** axes fire on the same
step:

* the V7 hybrid declared seven_way=True;
* cache_controller_v6 has a fitted three_objective_head OR
  composite_v6_weights OR retrieval_repair_head_coefs;
* replay_controller_v4 emitted ≥ 1 per-regime decision with
  non-zero replay_dominance;
* three_way_bridge_classifier has been fit;
* the V8 substrate's hidden_vs_kv_contention l1 > 0;
* attention_steering_bridge_v7 reported a non-zero JS shift OR
  per-bucket cosine correlation;
* the prefix_v7 drift-curve predictor exists;
* the V8 prefix_reuse_trust ledger l1 > 0.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any

from .cache_controller_v6 import CacheControllerV6
from .deep_substrate_hybrid_v7 import (
    DeepSubstrateHybridV7, DeepSubstrateHybridV7ForwardWitness,
)
from .replay_controller_v4 import ReplayControllerV4
from .tiny_substrate_v3 import _sha256_hex


W63_DEEP_SUBSTRATE_HYBRID_V8_SCHEMA_VERSION: str = (
    "coordpy.deep_substrate_hybrid_v8.v1")


@dataclasses.dataclass
class DeepSubstrateHybridV8:
    inner_v7: DeepSubstrateHybridV7 | None = None
    eight_way_active: bool = False

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W63_DEEP_SUBSTRATE_HYBRID_V8_SCHEMA_VERSION,
            "kind": "deep_substrate_hybrid_v8",
            "inner_v7_cid": (
                str(self.inner_v7.cid())
                if self.inner_v7 is not None else "none"),
            "eight_way_active": bool(self.eight_way_active),
        })


@dataclasses.dataclass(frozen=True)
class DeepSubstrateHybridV8ForwardWitness:
    schema: str
    hybrid_cid: str
    inner_v7_witness_cid: str
    eight_way: bool
    cache_controller_v6_fired: bool
    replay_controller_v4_fired: bool
    three_way_bridge_classifier_fired: bool
    hidden_vs_kv_contention_active: bool
    attention_v7_active: bool
    prefix_v7_drift_predictor_active: bool
    prefix_reuse_trust_active: bool
    mean_replay_dominance: float
    hidden_vs_kv_contention_l1: float
    attention_v7_js_max: float
    prefix_reuse_trust_l1: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "hybrid_cid": str(self.hybrid_cid),
            "inner_v7_witness_cid": str(
                self.inner_v7_witness_cid),
            "eight_way": bool(self.eight_way),
            "cache_controller_v6_fired": bool(
                self.cache_controller_v6_fired),
            "replay_controller_v4_fired": bool(
                self.replay_controller_v4_fired),
            "three_way_bridge_classifier_fired": bool(
                self.three_way_bridge_classifier_fired),
            "hidden_vs_kv_contention_active": bool(
                self.hidden_vs_kv_contention_active),
            "attention_v7_active": bool(
                self.attention_v7_active),
            "prefix_v7_drift_predictor_active": bool(
                self.prefix_v7_drift_predictor_active),
            "prefix_reuse_trust_active": bool(
                self.prefix_reuse_trust_active),
            "mean_replay_dominance": float(round(
                self.mean_replay_dominance, 12)),
            "hidden_vs_kv_contention_l1": float(round(
                self.hidden_vs_kv_contention_l1, 12)),
            "attention_v7_js_max": float(round(
                self.attention_v7_js_max, 12)),
            "prefix_reuse_trust_l1": float(round(
                self.prefix_reuse_trust_l1, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "deep_substrate_hybrid_v8_witness",
            "witness": self.to_dict()})


def deep_substrate_hybrid_v8_forward(
        *, hybrid: DeepSubstrateHybridV8,
        v7_witness: DeepSubstrateHybridV7ForwardWitness,
        cache_controller_v6: CacheControllerV6 | None = None,
        replay_controller_v4: ReplayControllerV4 | None = None,
        hidden_vs_kv_contention_l1: float = 0.0,
        attention_v7_js_max: float = 0.0,
        prefix_v7_drift_predictor_trained: bool = False,
        prefix_reuse_trust_l1: float = 0.0,
) -> DeepSubstrateHybridV8ForwardWitness:
    cache_fired = bool(
        cache_controller_v6 is not None
        and (cache_controller_v6.three_objective_head is not None
              or cache_controller_v6.composite_v6_weights is not None
              or cache_controller_v6.retrieval_repair_head_coefs
              is not None))
    replay_fired = False
    three_way_fired = False
    dominance = 0.0
    if replay_controller_v4 is not None:
        replay_fired = bool(replay_controller_v4.audit_v4)
        three_way_fired = bool(
            replay_controller_v4.three_way_bridge_classifier
            is not None)
        doms = [
            float(e.get("replay_dominance", 0.0))
            for e in replay_controller_v4.audit_v4
            if "replay_dominance" in e]
        if doms:
            dominance = float(sum(doms) / len(doms))
    contention_active = bool(
        float(hidden_vs_kv_contention_l1) > 0.0)
    attn_v7_active = bool(float(attention_v7_js_max) > 0.0)
    prefix_active = bool(float(prefix_reuse_trust_l1) > 0.0)
    eight_way = bool(
        v7_witness.seven_way
        and cache_fired
        and replay_fired
        and three_way_fired
        and contention_active
        and attn_v7_active
        and bool(prefix_v7_drift_predictor_trained)
        and prefix_active)
    hybrid.eight_way_active = bool(eight_way)
    return DeepSubstrateHybridV8ForwardWitness(
        schema=W63_DEEP_SUBSTRATE_HYBRID_V8_SCHEMA_VERSION,
        hybrid_cid=str(hybrid.cid()),
        inner_v7_witness_cid=str(v7_witness.cid()),
        eight_way=bool(eight_way),
        cache_controller_v6_fired=bool(cache_fired),
        replay_controller_v4_fired=bool(replay_fired),
        three_way_bridge_classifier_fired=bool(three_way_fired),
        hidden_vs_kv_contention_active=bool(contention_active),
        attention_v7_active=bool(attn_v7_active),
        prefix_v7_drift_predictor_active=bool(
            prefix_v7_drift_predictor_trained),
        prefix_reuse_trust_active=bool(prefix_active),
        mean_replay_dominance=float(dominance),
        hidden_vs_kv_contention_l1=float(
            hidden_vs_kv_contention_l1),
        attention_v7_js_max=float(attention_v7_js_max),
        prefix_reuse_trust_l1=float(prefix_reuse_trust_l1),
    )


__all__ = [
    "W63_DEEP_SUBSTRATE_HYBRID_V8_SCHEMA_VERSION",
    "DeepSubstrateHybridV8",
    "DeepSubstrateHybridV8ForwardWitness",
    "deep_substrate_hybrid_v8_forward",
]
