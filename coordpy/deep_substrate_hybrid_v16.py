"""W71 — Deep Substrate Hybrid V16.

Strictly extends W70's ``coordpy.deep_substrate_hybrid_v15``. V15
ran a *fifteen-way* loop. V16 runs a *sixteen-way* loop with the
V16 substrate at its centre:

  V16 latent ↔ tiny_substrate_v16 ↔ cache_controller_v14
  ↔ replay_controller_v12 ↔ retrieval_head
  ↔ attention_steering_bridge_v13 ↔ five_way_bridge_classifier
  ↔ prefix_state_bridge_v13 ↔ hidden_state_bridge_v13
  ↔ multi_agent_substrate_coordinator_v7
  ↔ team_consensus_controller_v6
  ↔ multi_branch_rejoin_witness_axis
  ↔ silent_corruption_witness_axis
  ↔ substrate_self_checksum_axis
  ↔ repair_trajectory_axis
  ↔ delayed_repair_trajectory_axis.

The sixteen-way flag is set when **all sixteen** axes fire on the
same step.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from .cache_controller_v14 import CacheControllerV14
from .deep_substrate_hybrid_v15 import (
    DeepSubstrateHybridV15,
    DeepSubstrateHybridV15ForwardWitness,
)
from .replay_controller_v12 import ReplayControllerV12
from .tiny_substrate_v3 import _sha256_hex


W71_DEEP_SUBSTRATE_HYBRID_V16_SCHEMA_VERSION: str = (
    "coordpy.deep_substrate_hybrid_v16.v1")


@dataclasses.dataclass
class DeepSubstrateHybridV16:
    inner_v15: DeepSubstrateHybridV15 | None = None
    sixteen_way_active: bool = False

    def cid(self) -> str:
        return _sha256_hex({
            "schema":
                W71_DEEP_SUBSTRATE_HYBRID_V16_SCHEMA_VERSION,
            "kind": "deep_substrate_hybrid_v16",
            "inner_v15_cid": (
                str(self.inner_v15.cid())
                if self.inner_v15 is not None else "none"),
            "sixteen_way_active": bool(self.sixteen_way_active),
        })


@dataclasses.dataclass(frozen=True)
class DeepSubstrateHybridV16ForwardWitness:
    schema: str
    hybrid_cid: str
    inner_v15_witness_cid: str
    sixteen_way: bool
    cache_controller_v14_fired: bool
    replay_controller_v12_fired: bool
    delayed_repair_trajectory_active: bool
    restart_dominance_active: bool
    team_consensus_controller_v6_active: bool
    delayed_repair_trajectory_cid: str
    restart_dominance_l1: int
    delayed_repair_gate_mean: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "hybrid_cid": str(self.hybrid_cid),
            "inner_v15_witness_cid": str(
                self.inner_v15_witness_cid),
            "sixteen_way": bool(self.sixteen_way),
            "cache_controller_v14_fired": bool(
                self.cache_controller_v14_fired),
            "replay_controller_v12_fired": bool(
                self.replay_controller_v12_fired),
            "delayed_repair_trajectory_active": bool(
                self.delayed_repair_trajectory_active),
            "restart_dominance_active": bool(
                self.restart_dominance_active),
            "team_consensus_controller_v6_active": bool(
                self.team_consensus_controller_v6_active),
            "delayed_repair_trajectory_cid": str(
                self.delayed_repair_trajectory_cid),
            "restart_dominance_l1": int(
                self.restart_dominance_l1),
            "delayed_repair_gate_mean": float(round(
                self.delayed_repair_gate_mean, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "deep_substrate_hybrid_v16_witness",
            "witness": self.to_dict()})


def deep_substrate_hybrid_v16_forward(
        *, hybrid: DeepSubstrateHybridV16,
        v15_witness: DeepSubstrateHybridV15ForwardWitness,
        cache_controller_v14: CacheControllerV14 | None = None,
        replay_controller_v12: ReplayControllerV12 | None = None,
        delayed_repair_trajectory_cid: str = "",
        restart_dominance_l1: int = 0,
        delayed_repair_gate_mean: float = 0.0,
        n_team_consensus_v6_invocations: int = 0,
) -> DeepSubstrateHybridV16ForwardWitness:
    cache_v14_fired = bool(
        cache_controller_v14 is not None
        and (cache_controller_v14.eleven_objective_head
             is not None
             or len(
                cache_controller_v14
                .per_role_restart_priority_heads_v14) > 0))
    replay_v12_fired = bool(
        replay_controller_v12 is not None
        and (replay_controller_v12
              .restart_aware_routing_head is not None
              and len(
                replay_controller_v12
                .per_role_per_regime_heads_v12) > 0))
    drt_active = bool(len(str(delayed_repair_trajectory_cid)) > 0)
    rd_active = bool(int(restart_dominance_l1) > 0)
    tcc_v6_active = bool(
        int(n_team_consensus_v6_invocations) > 0)
    sixteen_way = bool(
        v15_witness.fifteen_way
        and cache_v14_fired and replay_v12_fired
        and drt_active and rd_active and tcc_v6_active)
    hybrid.sixteen_way_active = bool(sixteen_way)
    return DeepSubstrateHybridV16ForwardWitness(
        schema=W71_DEEP_SUBSTRATE_HYBRID_V16_SCHEMA_VERSION,
        hybrid_cid=str(hybrid.cid()),
        inner_v15_witness_cid=str(v15_witness.cid()),
        sixteen_way=bool(sixteen_way),
        cache_controller_v14_fired=bool(cache_v14_fired),
        replay_controller_v12_fired=bool(replay_v12_fired),
        delayed_repair_trajectory_active=bool(drt_active),
        restart_dominance_active=bool(rd_active),
        team_consensus_controller_v6_active=bool(tcc_v6_active),
        delayed_repair_trajectory_cid=str(
            delayed_repair_trajectory_cid),
        restart_dominance_l1=int(restart_dominance_l1),
        delayed_repair_gate_mean=float(
            delayed_repair_gate_mean),
    )


__all__ = [
    "W71_DEEP_SUBSTRATE_HYBRID_V16_SCHEMA_VERSION",
    "DeepSubstrateHybridV16",
    "DeepSubstrateHybridV16ForwardWitness",
    "deep_substrate_hybrid_v16_forward",
]
