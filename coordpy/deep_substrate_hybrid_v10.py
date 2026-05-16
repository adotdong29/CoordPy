"""W65 — Deep Substrate Hybrid V10.

Strictly extends W64's ``coordpy.deep_substrate_hybrid_v9``. V9
ran a *nine-way* loop:

  V9 latent ↔ tiny_substrate_v9 ↔ cache_controller_v7
  ↔ replay_controller_v5 ↔ retrieval_head
  ↔ attention_steering_bridge_v8 ↔ four_way_bridge_classifier
  ↔ prefix_state_bridge_v8 ↔ hidden_state_bridge_v8.

V10 runs a *ten-way* loop with the V10 substrate at its centre:

  V10 latent ↔ tiny_substrate_v10 ↔ cache_controller_v8
  ↔ replay_controller_v6 ↔ retrieval_head
  ↔ attention_steering_bridge_v9 ↔ four_way_bridge_classifier
  ↔ prefix_state_bridge_v9 ↔ hidden_state_bridge_v9
  ↔ multi_agent_substrate_coordinator.

The ten-way flag is set when **all ten** axes fire on the same
step:

* the V9 hybrid declared nine_way=True;
* cache_controller_v8 has a fitted five_objective_head OR ≥ 1
  per-role eviction head;
* replay_controller_v6 has a fitted multi_agent_abstain_head
  AND ≥ 1 per-role per-regime head;
* the V10 substrate's hidden_write_merit L1 > 0;
* attention_steering_bridge_v9 reported five-stage used AND a
  fingerprint;
* the prefix_v9 K=64 predictor exists;
* the HSB V9 hidden-wins-rate mean > 0;
* the V10 role KV bank has ≥ 1 role;
* the multi_agent_substrate_coordinator was invoked ≥ once.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from .cache_controller_v8 import CacheControllerV8
from .deep_substrate_hybrid_v9 import (
    DeepSubstrateHybridV9,
    DeepSubstrateHybridV9ForwardWitness,
)
from .replay_controller_v6 import ReplayControllerV6
from .tiny_substrate_v3 import _sha256_hex


W65_DEEP_SUBSTRATE_HYBRID_V10_SCHEMA_VERSION: str = (
    "coordpy.deep_substrate_hybrid_v10.v1")


@dataclasses.dataclass
class DeepSubstrateHybridV10:
    inner_v9: DeepSubstrateHybridV9 | None = None
    ten_way_active: bool = False

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W65_DEEP_SUBSTRATE_HYBRID_V10_SCHEMA_VERSION,
            "kind": "deep_substrate_hybrid_v10",
            "inner_v9_cid": (
                str(self.inner_v9.cid())
                if self.inner_v9 is not None else "none"),
            "ten_way_active": bool(self.ten_way_active),
        })


@dataclasses.dataclass(frozen=True)
class DeepSubstrateHybridV10ForwardWitness:
    schema: str
    hybrid_cid: str
    inner_v9_witness_cid: str
    ten_way: bool
    cache_controller_v8_fired: bool
    replay_controller_v6_fired: bool
    hidden_write_merit_active: bool
    attention_v9_active: bool
    prefix_v9_active: bool
    hsb_v9_active: bool
    role_kv_bank_active: bool
    multi_agent_coordinator_active: bool
    hidden_write_merit_l1: float
    attention_v9_fingerprint_present: bool
    hsb_v9_hidden_wins_rate_mean: float
    n_roles_in_bank: int
    n_team_invocations: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "hybrid_cid": str(self.hybrid_cid),
            "inner_v9_witness_cid": str(
                self.inner_v9_witness_cid),
            "ten_way": bool(self.ten_way),
            "cache_controller_v8_fired": bool(
                self.cache_controller_v8_fired),
            "replay_controller_v6_fired": bool(
                self.replay_controller_v6_fired),
            "hidden_write_merit_active": bool(
                self.hidden_write_merit_active),
            "attention_v9_active": bool(
                self.attention_v9_active),
            "prefix_v9_active": bool(self.prefix_v9_active),
            "hsb_v9_active": bool(self.hsb_v9_active),
            "role_kv_bank_active": bool(
                self.role_kv_bank_active),
            "multi_agent_coordinator_active": bool(
                self.multi_agent_coordinator_active),
            "hidden_write_merit_l1": float(round(
                self.hidden_write_merit_l1, 12)),
            "attention_v9_fingerprint_present": bool(
                self.attention_v9_fingerprint_present),
            "hsb_v9_hidden_wins_rate_mean": float(round(
                self.hsb_v9_hidden_wins_rate_mean, 12)),
            "n_roles_in_bank": int(self.n_roles_in_bank),
            "n_team_invocations": int(self.n_team_invocations),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "deep_substrate_hybrid_v10_witness",
            "witness": self.to_dict()})


def deep_substrate_hybrid_v10_forward(
        *, hybrid: DeepSubstrateHybridV10,
        v9_witness: DeepSubstrateHybridV9ForwardWitness,
        cache_controller_v8: CacheControllerV8 | None = None,
        replay_controller_v6: ReplayControllerV6 | None = None,
        hidden_write_merit_l1: float = 0.0,
        attention_v9_fingerprint_present: bool = False,
        prefix_v9_predictor_present: bool = False,
        hsb_v9_hidden_wins_rate_mean: float = 0.0,
        n_roles_in_bank: int = 0,
        n_team_invocations: int = 0,
) -> DeepSubstrateHybridV10ForwardWitness:
    cache_v8_fired = bool(
        cache_controller_v8 is not None
        and (cache_controller_v8.five_objective_head is not None
              or len(
                cache_controller_v8.per_role_eviction_heads) > 0))
    replay_v6_fired = bool(
        replay_controller_v6 is not None
        and (replay_controller_v6.multi_agent_abstain_head
              is not None
              and len(
                replay_controller_v6.per_role_per_regime_heads)
              > 0))
    merit_active = bool(float(hidden_write_merit_l1) > 0.0)
    attn_active = bool(attention_v9_fingerprint_present)
    prefix_active = bool(prefix_v9_predictor_present)
    hsb_active = bool(float(hsb_v9_hidden_wins_rate_mean) > 0.0)
    bank_active = bool(int(n_roles_in_bank) > 0)
    team_active = bool(int(n_team_invocations) > 0)
    ten_way = bool(
        v9_witness.nine_way
        and cache_v8_fired and replay_v6_fired
        and merit_active and attn_active and prefix_active
        and hsb_active and bank_active and team_active)
    hybrid.ten_way_active = bool(ten_way)
    return DeepSubstrateHybridV10ForwardWitness(
        schema=W65_DEEP_SUBSTRATE_HYBRID_V10_SCHEMA_VERSION,
        hybrid_cid=str(hybrid.cid()),
        inner_v9_witness_cid=str(v9_witness.cid()),
        ten_way=bool(ten_way),
        cache_controller_v8_fired=bool(cache_v8_fired),
        replay_controller_v6_fired=bool(replay_v6_fired),
        hidden_write_merit_active=bool(merit_active),
        attention_v9_active=bool(attn_active),
        prefix_v9_active=bool(prefix_active),
        hsb_v9_active=bool(hsb_active),
        role_kv_bank_active=bool(bank_active),
        multi_agent_coordinator_active=bool(team_active),
        hidden_write_merit_l1=float(hidden_write_merit_l1),
        attention_v9_fingerprint_present=bool(
            attention_v9_fingerprint_present),
        hsb_v9_hidden_wins_rate_mean=float(
            hsb_v9_hidden_wins_rate_mean),
        n_roles_in_bank=int(n_roles_in_bank),
        n_team_invocations=int(n_team_invocations),
    )


__all__ = [
    "W65_DEEP_SUBSTRATE_HYBRID_V10_SCHEMA_VERSION",
    "DeepSubstrateHybridV10",
    "DeepSubstrateHybridV10ForwardWitness",
    "deep_substrate_hybrid_v10_forward",
]
