"""W75 — Deep Substrate Hybrid V20.

Strictly extends W74's ``coordpy.deep_substrate_hybrid_v19``. V19
ran a *nineteen-way* loop. V20 runs a *twenty-way* loop with the
V20 substrate at its centre:

  V20 latent ↔ tiny_substrate_v20 ↔ cache_controller_v18
  ↔ replay_controller_v16 ↔ retrieval_head
  ↔ attention_steering_bridge_v13 ↔ five_way_bridge_classifier
  ↔ prefix_state_bridge_v13 ↔ hidden_state_bridge_v13
  ↔ multi_agent_substrate_coordinator_v11
  ↔ team_consensus_controller_v10
  ↔ multi_branch_rejoin_witness_axis
  ↔ silent_corruption_witness_axis
  ↔ substrate_self_checksum_axis
  ↔ repair_trajectory_axis
  ↔ delayed_repair_trajectory_axis
  ↔ restart_repair_trajectory_axis
  ↔ replacement_repair_trajectory_axis
  ↔ compound_repair_trajectory_axis
  ↔ compound_chain_repair_trajectory_axis.

The twenty-way flag is set when **all twenty** axes fire on the
same step.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from .cache_controller_v18 import CacheControllerV18
from .deep_substrate_hybrid_v19 import (
    DeepSubstrateHybridV19,
    DeepSubstrateHybridV19ForwardWitness,
)
from .replay_controller_v16 import ReplayControllerV16
from .tiny_substrate_v3 import _sha256_hex


W75_DEEP_SUBSTRATE_HYBRID_V20_SCHEMA_VERSION: str = (
    "coordpy.deep_substrate_hybrid_v20.v1")


@dataclasses.dataclass
class DeepSubstrateHybridV20:
    inner_v19: DeepSubstrateHybridV19 | None = None
    twenty_way_active: bool = False

    def cid(self) -> str:
        return _sha256_hex({
            "schema":
                W75_DEEP_SUBSTRATE_HYBRID_V20_SCHEMA_VERSION,
            "kind": "deep_substrate_hybrid_v20",
            "inner_v19_cid": (
                str(self.inner_v19.cid())
                if self.inner_v19 is not None else "none"),
            "twenty_way_active": bool(
                self.twenty_way_active),
        })


@dataclasses.dataclass(frozen=True)
class DeepSubstrateHybridV20ForwardWitness:
    schema: str
    hybrid_cid: str
    inner_v19_witness_cid: str
    twenty_way: bool
    cache_controller_v18_fired: bool
    replay_controller_v16_fired: bool
    compound_chain_repair_trajectory_active: bool
    compound_chain_repair_active: bool
    team_consensus_controller_v10_active: bool
    compound_chain_repair_trajectory_cid: str
    compound_chain_repair_l1: int
    compound_chain_pressure_gate_mean: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "hybrid_cid": str(self.hybrid_cid),
            "inner_v19_witness_cid": str(
                self.inner_v19_witness_cid),
            "twenty_way": bool(self.twenty_way),
            "cache_controller_v18_fired": bool(
                self.cache_controller_v18_fired),
            "replay_controller_v16_fired": bool(
                self.replay_controller_v16_fired),
            "compound_chain_repair_trajectory_active": bool(
                self.compound_chain_repair_trajectory_active),
            "compound_chain_repair_active": bool(
                self.compound_chain_repair_active),
            "team_consensus_controller_v10_active": bool(
                self.team_consensus_controller_v10_active),
            "compound_chain_repair_trajectory_cid": str(
                self.compound_chain_repair_trajectory_cid),
            "compound_chain_repair_l1": int(
                self.compound_chain_repair_l1),
            "compound_chain_pressure_gate_mean": float(round(
                self.compound_chain_pressure_gate_mean, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "deep_substrate_hybrid_v20_witness",
            "witness": self.to_dict()})


def deep_substrate_hybrid_v20_forward(
        *, hybrid: DeepSubstrateHybridV20,
        v19_witness: DeepSubstrateHybridV19ForwardWitness,
        cache_controller_v18: CacheControllerV18 | None = None,
        replay_controller_v16: ReplayControllerV16 | None = None,
        compound_chain_repair_trajectory_cid: str = "",
        compound_chain_repair_l1: int = 0,
        compound_chain_pressure_gate_mean: float = 0.0,
        n_team_consensus_v10_invocations: int = 0,
) -> DeepSubstrateHybridV20ForwardWitness:
    cache_v18_fired = bool(
        cache_controller_v18 is not None
        and (cache_controller_v18.fifteen_objective_head
             is not None
             or len(
                cache_controller_v18
                .per_role_compound_chain_pressure_heads_v18) > 0))
    replay_v16_fired = bool(
        replay_controller_v16 is not None
        and (replay_controller_v16
              .compound_chain_aware_routing_head is not None
              and len(
                replay_controller_v16
                .per_role_per_regime_heads_v16) > 0))
    ccr_active = bool(
        len(str(compound_chain_repair_trajectory_cid)) > 0)
    chain_active = bool(int(compound_chain_repair_l1) > 0)
    tcc_v10_active = bool(
        int(n_team_consensus_v10_invocations) > 0)
    twenty_way = bool(
        v19_witness.nineteen_way
        and cache_v18_fired and replay_v16_fired
        and ccr_active and chain_active and tcc_v10_active)
    hybrid.twenty_way_active = bool(twenty_way)
    return DeepSubstrateHybridV20ForwardWitness(
        schema=W75_DEEP_SUBSTRATE_HYBRID_V20_SCHEMA_VERSION,
        hybrid_cid=str(hybrid.cid()),
        inner_v19_witness_cid=str(v19_witness.cid()),
        twenty_way=bool(twenty_way),
        cache_controller_v18_fired=bool(cache_v18_fired),
        replay_controller_v16_fired=bool(replay_v16_fired),
        compound_chain_repair_trajectory_active=bool(ccr_active),
        compound_chain_repair_active=bool(chain_active),
        team_consensus_controller_v10_active=bool(tcc_v10_active),
        compound_chain_repair_trajectory_cid=str(
            compound_chain_repair_trajectory_cid),
        compound_chain_repair_l1=int(
            compound_chain_repair_l1),
        compound_chain_pressure_gate_mean=float(
            compound_chain_pressure_gate_mean),
    )


__all__ = [
    "W75_DEEP_SUBSTRATE_HYBRID_V20_SCHEMA_VERSION",
    "DeepSubstrateHybridV20",
    "DeepSubstrateHybridV20ForwardWitness",
    "deep_substrate_hybrid_v20_forward",
]
