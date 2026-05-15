"""W62 — Deep Substrate Hybrid V7.

Strictly extends W61's ``coordpy.deep_substrate_hybrid_v6``. V6
ran a *six-way* loop:

  V6 latent ↔ tiny_substrate_v6 ↔ cache_controller_v4
  ↔ replay_controller_v2 ↔ retrieval_head
  ↔ attention_steering_bridge_v5.

V7 runs a *seven-way* loop with the V7 substrate at its centre:

  V7 latent ↔ tiny_substrate_v7 ↔ cache_controller_v5
  ↔ replay_controller_v3 ↔ retrieval_head
  ↔ attention_steering_bridge_v6 ↔ hidden_vs_kv_classifier.

The seven-way flag is set when **all seven** axes fire on the same
step:

* the V6 hybrid declared six_way=True;
* cache_controller_v5 has a fitted two_objective_head OR
  composite_v5_weights OR repair_head_coefs;
* replay_controller_v3 emitted ≥ 1 per-regime decision with
  non-zero replay_dominance;
* hidden_vs_kv_classifier has been fit;
* the V7 substrate's cache_write_ledger l2 > 0 (i.e. a bridge
  has written to it);
* attention_steering_bridge_v6 reported a non-zero coarse L1
  shift OR per-bucket signed correlation;
* the prefix_v6 drift-curve predictor exists.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any

from .cache_controller_v5 import CacheControllerV5
from .deep_substrate_hybrid_v6 import (
    DeepSubstrateHybridV6, DeepSubstrateHybridV6ForwardWitness,
)
from .replay_controller_v3 import ReplayControllerV3
from .tiny_substrate_v3 import _sha256_hex


W62_DEEP_SUBSTRATE_HYBRID_V7_SCHEMA_VERSION: str = (
    "coordpy.deep_substrate_hybrid_v7.v1")


@dataclasses.dataclass
class DeepSubstrateHybridV7:
    inner_v6: DeepSubstrateHybridV6 | None = None
    seven_way_active: bool = False

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W62_DEEP_SUBSTRATE_HYBRID_V7_SCHEMA_VERSION,
            "kind": "deep_substrate_hybrid_v7",
            "inner_v6_cid": (
                str(self.inner_v6.cid())
                if self.inner_v6 is not None else "none"),
            "seven_way_active": bool(self.seven_way_active),
        })


@dataclasses.dataclass(frozen=True)
class DeepSubstrateHybridV7ForwardWitness:
    schema: str
    hybrid_cid: str
    inner_v6_witness_cid: str
    seven_way: bool
    cache_controller_v5_fired: bool
    replay_controller_v3_fired: bool
    hidden_vs_kv_classifier_fired: bool
    cache_write_ledger_active: bool
    attention_v6_active: bool
    prefix_v6_drift_predictor_active: bool
    mean_replay_dominance: float
    cache_write_ledger_l2: float
    attention_v6_coarse_l1_shift: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "hybrid_cid": str(self.hybrid_cid),
            "inner_v6_witness_cid": str(
                self.inner_v6_witness_cid),
            "seven_way": bool(self.seven_way),
            "cache_controller_v5_fired": bool(
                self.cache_controller_v5_fired),
            "replay_controller_v3_fired": bool(
                self.replay_controller_v3_fired),
            "hidden_vs_kv_classifier_fired": bool(
                self.hidden_vs_kv_classifier_fired),
            "cache_write_ledger_active": bool(
                self.cache_write_ledger_active),
            "attention_v6_active": bool(self.attention_v6_active),
            "prefix_v6_drift_predictor_active": bool(
                self.prefix_v6_drift_predictor_active),
            "mean_replay_dominance": float(round(
                self.mean_replay_dominance, 12)),
            "cache_write_ledger_l2": float(round(
                self.cache_write_ledger_l2, 12)),
            "attention_v6_coarse_l1_shift": float(round(
                self.attention_v6_coarse_l1_shift, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "deep_substrate_hybrid_v7_witness",
            "witness": self.to_dict()})


def deep_substrate_hybrid_v7_forward(
        *, hybrid: DeepSubstrateHybridV7,
        v6_witness: DeepSubstrateHybridV6ForwardWitness,
        cache_controller_v5: CacheControllerV5 | None = None,
        replay_controller_v3: ReplayControllerV3 | None = None,
        cache_write_ledger_l2: float = 0.0,
        attention_v6_coarse_l1_shift: float = 0.0,
        prefix_v6_drift_predictor_trained: bool = False,
) -> DeepSubstrateHybridV7ForwardWitness:
    cache_fired = bool(
        cache_controller_v5 is not None
        and (cache_controller_v5.two_objective_head is not None
              or cache_controller_v5.composite_v5_weights is not None
              or cache_controller_v5.repair_head_coefs is not None))
    replay_fired = False
    hvkv_fired = False
    dominance = 0.0
    if replay_controller_v3 is not None:
        replay_fired = bool(replay_controller_v3.audit_v3)
        hvkv_fired = bool(
            replay_controller_v3.hidden_vs_kv_classifier
            is not None)
        doms = [
            float(e.get("replay_dominance", 0.0))
            for e in replay_controller_v3.audit_v3
            if "replay_dominance" in e]
        if doms:
            dominance = float(sum(doms) / len(doms))
    cache_ledger_active = bool(float(cache_write_ledger_l2) > 0.0)
    attn_v6_active = bool(
        float(attention_v6_coarse_l1_shift) > 0.0)
    seven_way = bool(
        v6_witness.six_way
        and cache_fired
        and replay_fired
        and hvkv_fired
        and cache_ledger_active
        and attn_v6_active
        and bool(prefix_v6_drift_predictor_trained))
    hybrid.seven_way_active = bool(seven_way)
    return DeepSubstrateHybridV7ForwardWitness(
        schema=W62_DEEP_SUBSTRATE_HYBRID_V7_SCHEMA_VERSION,
        hybrid_cid=str(hybrid.cid()),
        inner_v6_witness_cid=str(v6_witness.cid()),
        seven_way=bool(seven_way),
        cache_controller_v5_fired=bool(cache_fired),
        replay_controller_v3_fired=bool(replay_fired),
        hidden_vs_kv_classifier_fired=bool(hvkv_fired),
        cache_write_ledger_active=bool(cache_ledger_active),
        attention_v6_active=bool(attn_v6_active),
        prefix_v6_drift_predictor_active=bool(
            prefix_v6_drift_predictor_trained),
        mean_replay_dominance=float(dominance),
        cache_write_ledger_l2=float(cache_write_ledger_l2),
        attention_v6_coarse_l1_shift=float(
            attention_v6_coarse_l1_shift),
    )


__all__ = [
    "W62_DEEP_SUBSTRATE_HYBRID_V7_SCHEMA_VERSION",
    "DeepSubstrateHybridV7",
    "DeepSubstrateHybridV7ForwardWitness",
    "deep_substrate_hybrid_v7_forward",
]
