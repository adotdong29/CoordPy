"""W64 — Deep Substrate Hybrid V9.

Strictly extends W63's ``coordpy.deep_substrate_hybrid_v8``. V8
ran an *eight-way* loop:

  V8 latent ↔ tiny_substrate_v8 ↔ cache_controller_v6
  ↔ replay_controller_v4 ↔ retrieval_head
  ↔ attention_steering_bridge_v7 ↔ three_way_bridge_classifier
  ↔ prefix_state_bridge_v7.

V9 runs a *nine-way* loop with the V9 substrate at its centre:

  V9 latent ↔ tiny_substrate_v9 ↔ cache_controller_v7
  ↔ replay_controller_v5 ↔ retrieval_head
  ↔ attention_steering_bridge_v8 ↔ four_way_bridge_classifier
  ↔ prefix_state_bridge_v8 ↔ hidden_state_bridge_v8.

The nine-way flag is set when **all nine** axes fire on the same
step:

* the V8 hybrid declared eight_way=True;
* cache_controller_v7 has a fitted four_objective_head OR
  composite_v7_weights OR similarity_eviction_head_coefs;
* replay_controller_v5 emitted ≥ 1 per-regime decision with
  non-zero replay_dominance;
* four_way_bridge_classifier has been fit;
* the V9 substrate's hidden_wins_primary l1 > 0;
* attention_steering_bridge_v8 reported a non-zero Hellinger
  shift OR per-bucket entropy correlation;
* the prefix_v8 drift-curve predictor exists;
* the V9 hidden_state_trust_ledger l1 > 0;
* the hidden_state_bridge_v8 hidden-wins-primary margin > 0.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any

from .cache_controller_v7 import CacheControllerV7
from .deep_substrate_hybrid_v8 import (
    DeepSubstrateHybridV8, DeepSubstrateHybridV8ForwardWitness,
)
from .replay_controller_v5 import ReplayControllerV5
from .tiny_substrate_v3 import _sha256_hex


W64_DEEP_SUBSTRATE_HYBRID_V9_SCHEMA_VERSION: str = (
    "coordpy.deep_substrate_hybrid_v9.v1")


@dataclasses.dataclass
class DeepSubstrateHybridV9:
    inner_v8: DeepSubstrateHybridV8 | None = None
    nine_way_active: bool = False

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W64_DEEP_SUBSTRATE_HYBRID_V9_SCHEMA_VERSION,
            "kind": "deep_substrate_hybrid_v9",
            "inner_v8_cid": (
                str(self.inner_v8.cid())
                if self.inner_v8 is not None else "none"),
            "nine_way_active": bool(self.nine_way_active),
        })


@dataclasses.dataclass(frozen=True)
class DeepSubstrateHybridV9ForwardWitness:
    schema: str
    hybrid_cid: str
    inner_v8_witness_cid: str
    nine_way: bool
    cache_controller_v7_fired: bool
    replay_controller_v5_fired: bool
    four_way_bridge_classifier_fired: bool
    hidden_wins_primary_active: bool
    attention_v8_active: bool
    prefix_v8_drift_predictor_active: bool
    hidden_state_trust_active: bool
    hsb_v8_hidden_wins_primary_active: bool
    mean_replay_dominance: float
    hidden_wins_primary_l1: float
    attention_v8_hellinger_max: float
    hidden_state_trust_ledger_l1: float
    hsb_v8_hidden_wins_primary_margin: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "hybrid_cid": str(self.hybrid_cid),
            "inner_v8_witness_cid": str(
                self.inner_v8_witness_cid),
            "nine_way": bool(self.nine_way),
            "cache_controller_v7_fired": bool(
                self.cache_controller_v7_fired),
            "replay_controller_v5_fired": bool(
                self.replay_controller_v5_fired),
            "four_way_bridge_classifier_fired": bool(
                self.four_way_bridge_classifier_fired),
            "hidden_wins_primary_active": bool(
                self.hidden_wins_primary_active),
            "attention_v8_active": bool(
                self.attention_v8_active),
            "prefix_v8_drift_predictor_active": bool(
                self.prefix_v8_drift_predictor_active),
            "hidden_state_trust_active": bool(
                self.hidden_state_trust_active),
            "hsb_v8_hidden_wins_primary_active": bool(
                self.hsb_v8_hidden_wins_primary_active),
            "mean_replay_dominance": float(round(
                self.mean_replay_dominance, 12)),
            "hidden_wins_primary_l1": float(round(
                self.hidden_wins_primary_l1, 12)),
            "attention_v8_hellinger_max": float(round(
                self.attention_v8_hellinger_max, 12)),
            "hidden_state_trust_ledger_l1": float(round(
                self.hidden_state_trust_ledger_l1, 12)),
            "hsb_v8_hidden_wins_primary_margin": float(round(
                self.hsb_v8_hidden_wins_primary_margin, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "deep_substrate_hybrid_v9_witness",
            "witness": self.to_dict()})


def deep_substrate_hybrid_v9_forward(
        *, hybrid: DeepSubstrateHybridV9,
        v8_witness: DeepSubstrateHybridV8ForwardWitness,
        cache_controller_v7: CacheControllerV7 | None = None,
        replay_controller_v5: ReplayControllerV5 | None = None,
        hidden_wins_primary_l1: float = 0.0,
        attention_v8_hellinger_max: float = 0.0,
        prefix_v8_drift_predictor_trained: bool = False,
        hidden_state_trust_ledger_l1: float = 0.0,
        hsb_v8_hidden_wins_primary_margin: float = 0.0,
) -> DeepSubstrateHybridV9ForwardWitness:
    cache_fired = bool(
        cache_controller_v7 is not None
        and (cache_controller_v7.four_objective_head is not None
              or cache_controller_v7.composite_v7_weights
                  is not None
              or cache_controller_v7.similarity_eviction_head_coefs
                  is not None))
    replay_fired = False
    four_way_fired = False
    dominance = 0.0
    if replay_controller_v5 is not None:
        replay_fired = bool(replay_controller_v5.audit_v5)
        four_way_fired = bool(
            replay_controller_v5.four_way_bridge_classifier
            is not None)
        doms = [
            float(e.get("replay_dominance", 0.0))
            for e in replay_controller_v5.audit_v5
            if "replay_dominance" in e]
        if doms:
            dominance = float(sum(doms) / len(doms))
    primary_active = bool(
        float(hidden_wins_primary_l1) > 0.0)
    attn_v8_active = bool(
        float(attention_v8_hellinger_max) > 0.0)
    hidden_state_trust_active = bool(
        float(hidden_state_trust_ledger_l1) > 0.0)
    hsb_v8_active = bool(
        float(hsb_v8_hidden_wins_primary_margin) > 0.0)
    nine_way = bool(
        v8_witness.eight_way
        and cache_fired
        and replay_fired
        and four_way_fired
        and primary_active
        and attn_v8_active
        and bool(prefix_v8_drift_predictor_trained)
        and hidden_state_trust_active
        and hsb_v8_active)
    hybrid.nine_way_active = bool(nine_way)
    return DeepSubstrateHybridV9ForwardWitness(
        schema=W64_DEEP_SUBSTRATE_HYBRID_V9_SCHEMA_VERSION,
        hybrid_cid=str(hybrid.cid()),
        inner_v8_witness_cid=str(v8_witness.cid()),
        nine_way=bool(nine_way),
        cache_controller_v7_fired=bool(cache_fired),
        replay_controller_v5_fired=bool(replay_fired),
        four_way_bridge_classifier_fired=bool(four_way_fired),
        hidden_wins_primary_active=bool(primary_active),
        attention_v8_active=bool(attn_v8_active),
        prefix_v8_drift_predictor_active=bool(
            prefix_v8_drift_predictor_trained),
        hidden_state_trust_active=bool(
            hidden_state_trust_active),
        hsb_v8_hidden_wins_primary_active=bool(hsb_v8_active),
        mean_replay_dominance=float(dominance),
        hidden_wins_primary_l1=float(hidden_wins_primary_l1),
        attention_v8_hellinger_max=float(
            attention_v8_hellinger_max),
        hidden_state_trust_ledger_l1=float(
            hidden_state_trust_ledger_l1),
        hsb_v8_hidden_wins_primary_margin=float(
            hsb_v8_hidden_wins_primary_margin),
    )


__all__ = [
    "W64_DEEP_SUBSTRATE_HYBRID_V9_SCHEMA_VERSION",
    "DeepSubstrateHybridV9",
    "DeepSubstrateHybridV9ForwardWitness",
    "deep_substrate_hybrid_v9_forward",
]
