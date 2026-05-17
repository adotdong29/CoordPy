"""W74 — Deep Substrate Hybrid V19.

Strictly extends W73's ``coordpy.deep_substrate_hybrid_v18``. V18
ran an *eighteen-way* loop. V19 runs a *nineteen-way* loop with
the V19 substrate at its centre:

  V19 latent ↔ tiny_substrate_v19 ↔ cache_controller_v17
  ↔ replay_controller_v15 ↔ retrieval_head
  ↔ attention_steering_bridge_v13 ↔ five_way_bridge_classifier
  ↔ prefix_state_bridge_v13 ↔ hidden_state_bridge_v13
  ↔ multi_agent_substrate_coordinator_v10
  ↔ team_consensus_controller_v9
  ↔ multi_branch_rejoin_witness_axis
  ↔ silent_corruption_witness_axis
  ↔ substrate_self_checksum_axis
  ↔ repair_trajectory_axis
  ↔ delayed_repair_trajectory_axis
  ↔ restart_repair_trajectory_axis
  ↔ replacement_repair_trajectory_axis
  ↔ compound_repair_trajectory_axis.

The nineteen-way flag is set when **all nineteen** axes fire on
the same step.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from .cache_controller_v17 import CacheControllerV17
from .deep_substrate_hybrid_v18 import (
    DeepSubstrateHybridV18,
    DeepSubstrateHybridV18ForwardWitness,
)
from .replay_controller_v15 import ReplayControllerV15
from .tiny_substrate_v3 import _sha256_hex


W74_DEEP_SUBSTRATE_HYBRID_V19_SCHEMA_VERSION: str = (
    "coordpy.deep_substrate_hybrid_v19.v1")


@dataclasses.dataclass
class DeepSubstrateHybridV19:
    inner_v18: DeepSubstrateHybridV18 | None = None
    nineteen_way_active: bool = False

    def cid(self) -> str:
        return _sha256_hex({
            "schema":
                W74_DEEP_SUBSTRATE_HYBRID_V19_SCHEMA_VERSION,
            "kind": "deep_substrate_hybrid_v19",
            "inner_v18_cid": (
                str(self.inner_v18.cid())
                if self.inner_v18 is not None else "none"),
            "nineteen_way_active": bool(
                self.nineteen_way_active),
        })


@dataclasses.dataclass(frozen=True)
class DeepSubstrateHybridV19ForwardWitness:
    schema: str
    hybrid_cid: str
    inner_v18_witness_cid: str
    nineteen_way: bool
    cache_controller_v17_fired: bool
    replay_controller_v15_fired: bool
    compound_repair_trajectory_active: bool
    compound_repair_active: bool
    team_consensus_controller_v9_active: bool
    compound_repair_trajectory_cid: str
    compound_repair_l1: int
    compound_pressure_gate_mean: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "hybrid_cid": str(self.hybrid_cid),
            "inner_v18_witness_cid": str(
                self.inner_v18_witness_cid),
            "nineteen_way": bool(self.nineteen_way),
            "cache_controller_v17_fired": bool(
                self.cache_controller_v17_fired),
            "replay_controller_v15_fired": bool(
                self.replay_controller_v15_fired),
            "compound_repair_trajectory_active": bool(
                self.compound_repair_trajectory_active),
            "compound_repair_active": bool(
                self.compound_repair_active),
            "team_consensus_controller_v9_active": bool(
                self.team_consensus_controller_v9_active),
            "compound_repair_trajectory_cid": str(
                self.compound_repair_trajectory_cid),
            "compound_repair_l1": int(
                self.compound_repair_l1),
            "compound_pressure_gate_mean": float(round(
                self.compound_pressure_gate_mean, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "deep_substrate_hybrid_v19_witness",
            "witness": self.to_dict()})


def deep_substrate_hybrid_v19_forward(
        *, hybrid: DeepSubstrateHybridV19,
        v18_witness: DeepSubstrateHybridV18ForwardWitness,
        cache_controller_v17: CacheControllerV17 | None = None,
        replay_controller_v15: ReplayControllerV15 | None = None,
        compound_repair_trajectory_cid: str = "",
        compound_repair_l1: int = 0,
        compound_pressure_gate_mean: float = 0.0,
        n_team_consensus_v9_invocations: int = 0,
) -> DeepSubstrateHybridV19ForwardWitness:
    cache_v17_fired = bool(
        cache_controller_v17 is not None
        and (cache_controller_v17.fourteen_objective_head
             is not None
             or len(
                cache_controller_v17
                .per_role_compound_pressure_heads_v17) > 0))
    replay_v15_fired = bool(
        replay_controller_v15 is not None
        and (replay_controller_v15
              .compound_aware_routing_head is not None
              and len(
                replay_controller_v15
                .per_role_per_regime_heads_v15) > 0))
    crt_active = bool(
        len(str(compound_repair_trajectory_cid)) > 0)
    comp_active = bool(int(compound_repair_l1) > 0)
    tcc_v9_active = bool(
        int(n_team_consensus_v9_invocations) > 0)
    nineteen_way = bool(
        v18_witness.eighteen_way
        and cache_v17_fired and replay_v15_fired
        and crt_active and comp_active and tcc_v9_active)
    hybrid.nineteen_way_active = bool(nineteen_way)
    return DeepSubstrateHybridV19ForwardWitness(
        schema=W74_DEEP_SUBSTRATE_HYBRID_V19_SCHEMA_VERSION,
        hybrid_cid=str(hybrid.cid()),
        inner_v18_witness_cid=str(v18_witness.cid()),
        nineteen_way=bool(nineteen_way),
        cache_controller_v17_fired=bool(cache_v17_fired),
        replay_controller_v15_fired=bool(replay_v15_fired),
        compound_repair_trajectory_active=bool(crt_active),
        compound_repair_active=bool(comp_active),
        team_consensus_controller_v9_active=bool(tcc_v9_active),
        compound_repair_trajectory_cid=str(
            compound_repair_trajectory_cid),
        compound_repair_l1=int(
            compound_repair_l1),
        compound_pressure_gate_mean=float(
            compound_pressure_gate_mean),
    )


__all__ = [
    "W74_DEEP_SUBSTRATE_HYBRID_V19_SCHEMA_VERSION",
    "DeepSubstrateHybridV19",
    "DeepSubstrateHybridV19ForwardWitness",
    "deep_substrate_hybrid_v19_forward",
]
