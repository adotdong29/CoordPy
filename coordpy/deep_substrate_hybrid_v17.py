"""W72 — Deep Substrate Hybrid V17.

Strictly extends W71's ``coordpy.deep_substrate_hybrid_v16``. V16
ran a *sixteen-way* loop. V17 runs a *seventeen-way* loop with the
V17 substrate at its centre:

  V17 latent ↔ tiny_substrate_v17 ↔ cache_controller_v15
  ↔ replay_controller_v13 ↔ retrieval_head
  ↔ attention_steering_bridge_v13 ↔ five_way_bridge_classifier
  ↔ prefix_state_bridge_v13 ↔ hidden_state_bridge_v13
  ↔ multi_agent_substrate_coordinator_v8
  ↔ team_consensus_controller_v7
  ↔ multi_branch_rejoin_witness_axis
  ↔ silent_corruption_witness_axis
  ↔ substrate_self_checksum_axis
  ↔ repair_trajectory_axis
  ↔ delayed_repair_trajectory_axis
  ↔ restart_repair_trajectory_axis.

The seventeen-way flag is set when **all seventeen** axes fire on
the same step.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from .cache_controller_v15 import CacheControllerV15
from .deep_substrate_hybrid_v16 import (
    DeepSubstrateHybridV16,
    DeepSubstrateHybridV16ForwardWitness,
)
from .replay_controller_v13 import ReplayControllerV13
from .tiny_substrate_v3 import _sha256_hex


W72_DEEP_SUBSTRATE_HYBRID_V17_SCHEMA_VERSION: str = (
    "coordpy.deep_substrate_hybrid_v17.v1")


@dataclasses.dataclass
class DeepSubstrateHybridV17:
    inner_v16: DeepSubstrateHybridV16 | None = None
    seventeen_way_active: bool = False

    def cid(self) -> str:
        return _sha256_hex({
            "schema":
                W72_DEEP_SUBSTRATE_HYBRID_V17_SCHEMA_VERSION,
            "kind": "deep_substrate_hybrid_v17",
            "inner_v16_cid": (
                str(self.inner_v16.cid())
                if self.inner_v16 is not None else "none"),
            "seventeen_way_active": bool(
                self.seventeen_way_active),
        })


@dataclasses.dataclass(frozen=True)
class DeepSubstrateHybridV17ForwardWitness:
    schema: str
    hybrid_cid: str
    inner_v16_witness_cid: str
    seventeen_way: bool
    cache_controller_v15_fired: bool
    replay_controller_v13_fired: bool
    restart_repair_trajectory_active: bool
    delayed_rejoin_after_restart_active: bool
    team_consensus_controller_v7_active: bool
    restart_repair_trajectory_cid: str
    delayed_rejoin_after_restart_l1: int
    rejoin_pressure_gate_mean: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "hybrid_cid": str(self.hybrid_cid),
            "inner_v16_witness_cid": str(
                self.inner_v16_witness_cid),
            "seventeen_way": bool(self.seventeen_way),
            "cache_controller_v15_fired": bool(
                self.cache_controller_v15_fired),
            "replay_controller_v13_fired": bool(
                self.replay_controller_v13_fired),
            "restart_repair_trajectory_active": bool(
                self.restart_repair_trajectory_active),
            "delayed_rejoin_after_restart_active": bool(
                self.delayed_rejoin_after_restart_active),
            "team_consensus_controller_v7_active": bool(
                self.team_consensus_controller_v7_active),
            "restart_repair_trajectory_cid": str(
                self.restart_repair_trajectory_cid),
            "delayed_rejoin_after_restart_l1": int(
                self.delayed_rejoin_after_restart_l1),
            "rejoin_pressure_gate_mean": float(round(
                self.rejoin_pressure_gate_mean, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "deep_substrate_hybrid_v17_witness",
            "witness": self.to_dict()})


def deep_substrate_hybrid_v17_forward(
        *, hybrid: DeepSubstrateHybridV17,
        v16_witness: DeepSubstrateHybridV16ForwardWitness,
        cache_controller_v15: CacheControllerV15 | None = None,
        replay_controller_v13: ReplayControllerV13 | None = None,
        restart_repair_trajectory_cid: str = "",
        delayed_rejoin_after_restart_l1: int = 0,
        rejoin_pressure_gate_mean: float = 0.0,
        n_team_consensus_v7_invocations: int = 0,
) -> DeepSubstrateHybridV17ForwardWitness:
    cache_v15_fired = bool(
        cache_controller_v15 is not None
        and (cache_controller_v15.twelve_objective_head
             is not None
             or len(
                cache_controller_v15
                .per_role_rejoin_pressure_heads_v15) > 0))
    replay_v13_fired = bool(
        replay_controller_v13 is not None
        and (replay_controller_v13
              .rejoin_aware_routing_head is not None
              and len(
                replay_controller_v13
                .per_role_per_regime_heads_v13) > 0))
    rrt_active = bool(
        len(str(restart_repair_trajectory_cid)) > 0)
    rj_active = bool(int(delayed_rejoin_after_restart_l1) > 0)
    tcc_v7_active = bool(
        int(n_team_consensus_v7_invocations) > 0)
    seventeen_way = bool(
        v16_witness.sixteen_way
        and cache_v15_fired and replay_v13_fired
        and rrt_active and rj_active and tcc_v7_active)
    hybrid.seventeen_way_active = bool(seventeen_way)
    return DeepSubstrateHybridV17ForwardWitness(
        schema=W72_DEEP_SUBSTRATE_HYBRID_V17_SCHEMA_VERSION,
        hybrid_cid=str(hybrid.cid()),
        inner_v16_witness_cid=str(v16_witness.cid()),
        seventeen_way=bool(seventeen_way),
        cache_controller_v15_fired=bool(cache_v15_fired),
        replay_controller_v13_fired=bool(replay_v13_fired),
        restart_repair_trajectory_active=bool(rrt_active),
        delayed_rejoin_after_restart_active=bool(rj_active),
        team_consensus_controller_v7_active=bool(tcc_v7_active),
        restart_repair_trajectory_cid=str(
            restart_repair_trajectory_cid),
        delayed_rejoin_after_restart_l1=int(
            delayed_rejoin_after_restart_l1),
        rejoin_pressure_gate_mean=float(
            rejoin_pressure_gate_mean),
    )


__all__ = [
    "W72_DEEP_SUBSTRATE_HYBRID_V17_SCHEMA_VERSION",
    "DeepSubstrateHybridV17",
    "DeepSubstrateHybridV17ForwardWitness",
    "deep_substrate_hybrid_v17_forward",
]
