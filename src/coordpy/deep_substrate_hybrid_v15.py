"""W70 — Deep Substrate Hybrid V15.

Strictly extends W69's ``coordpy.deep_substrate_hybrid_v14``. V14
ran a *fourteen-way* loop. V15 runs a *fifteen-way* loop with the
V15 substrate at its centre:

  V15 latent ↔ tiny_substrate_v15 ↔ cache_controller_v13
  ↔ replay_controller_v11 ↔ retrieval_head
  ↔ attention_steering_bridge_v13 ↔ five_way_bridge_classifier
  ↔ prefix_state_bridge_v13 ↔ hidden_state_bridge_v13
  ↔ multi_agent_substrate_coordinator_v6
  ↔ team_consensus_controller_v5
  ↔ multi_branch_rejoin_witness_axis
  ↔ silent_corruption_witness_axis
  ↔ substrate_self_checksum_axis
  ↔ repair_trajectory_axis.

The fifteen-way flag is set when **all fifteen** axes fire on the
same step.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from .cache_controller_v13 import CacheControllerV13
from .deep_substrate_hybrid_v14 import (
    DeepSubstrateHybridV14,
    DeepSubstrateHybridV14ForwardWitness,
)
from .replay_controller_v11 import ReplayControllerV11
from .tiny_substrate_v3 import _sha256_hex


W70_DEEP_SUBSTRATE_HYBRID_V15_SCHEMA_VERSION: str = (
    "coordpy.deep_substrate_hybrid_v15.v1")


@dataclasses.dataclass
class DeepSubstrateHybridV15:
    inner_v14: DeepSubstrateHybridV14 | None = None
    fifteen_way_active: bool = False

    def cid(self) -> str:
        return _sha256_hex({
            "schema":
                W70_DEEP_SUBSTRATE_HYBRID_V15_SCHEMA_VERSION,
            "kind": "deep_substrate_hybrid_v15",
            "inner_v14_cid": (
                str(self.inner_v14.cid())
                if self.inner_v14 is not None else "none"),
            "fifteen_way_active": bool(self.fifteen_way_active),
        })


@dataclasses.dataclass(frozen=True)
class DeepSubstrateHybridV15ForwardWitness:
    schema: str
    hybrid_cid: str
    inner_v14_witness_cid: str
    fifteen_way: bool
    cache_controller_v13_fired: bool
    replay_controller_v11_fired: bool
    repair_trajectory_active: bool
    budget_primary_active: bool
    team_consensus_controller_v5_active: bool
    repair_trajectory_cid: str
    dominant_repair_l1: int
    budget_primary_gate_mean: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "hybrid_cid": str(self.hybrid_cid),
            "inner_v14_witness_cid": str(
                self.inner_v14_witness_cid),
            "fifteen_way": bool(self.fifteen_way),
            "cache_controller_v13_fired": bool(
                self.cache_controller_v13_fired),
            "replay_controller_v11_fired": bool(
                self.replay_controller_v11_fired),
            "repair_trajectory_active": bool(
                self.repair_trajectory_active),
            "budget_primary_active": bool(
                self.budget_primary_active),
            "team_consensus_controller_v5_active": bool(
                self.team_consensus_controller_v5_active),
            "repair_trajectory_cid": str(
                self.repair_trajectory_cid),
            "dominant_repair_l1": int(self.dominant_repair_l1),
            "budget_primary_gate_mean": float(round(
                self.budget_primary_gate_mean, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "deep_substrate_hybrid_v15_witness",
            "witness": self.to_dict()})


def deep_substrate_hybrid_v15_forward(
        *, hybrid: DeepSubstrateHybridV15,
        v14_witness: DeepSubstrateHybridV14ForwardWitness,
        cache_controller_v13: CacheControllerV13 | None = None,
        replay_controller_v11: ReplayControllerV11 | None = None,
        repair_trajectory_cid: str = "",
        dominant_repair_l1: int = 0,
        budget_primary_gate_mean: float = 0.0,
        n_team_consensus_v5_invocations: int = 0,
) -> DeepSubstrateHybridV15ForwardWitness:
    cache_v13_fired = bool(
        cache_controller_v13 is not None
        and (cache_controller_v13.ten_objective_head is not None
              or len(
                cache_controller_v13
                .per_role_budget_primary_heads_v13) > 0))
    replay_v11_fired = bool(
        replay_controller_v11 is not None
        and (replay_controller_v11
              .budget_primary_routing_head is not None
              and len(
                replay_controller_v11
                .per_role_per_regime_heads_v11) > 0))
    rt_active = bool(len(str(repair_trajectory_cid)) > 0)
    bp_active = bool(float(budget_primary_gate_mean) > 0.0)
    tcc_v5_active = bool(
        int(n_team_consensus_v5_invocations) > 0)
    fifteen_way = bool(
        v14_witness.fourteen_way
        and cache_v13_fired and replay_v11_fired
        and rt_active and bp_active and tcc_v5_active)
    hybrid.fifteen_way_active = bool(fifteen_way)
    return DeepSubstrateHybridV15ForwardWitness(
        schema=W70_DEEP_SUBSTRATE_HYBRID_V15_SCHEMA_VERSION,
        hybrid_cid=str(hybrid.cid()),
        inner_v14_witness_cid=str(v14_witness.cid()),
        fifteen_way=bool(fifteen_way),
        cache_controller_v13_fired=bool(cache_v13_fired),
        replay_controller_v11_fired=bool(replay_v11_fired),
        repair_trajectory_active=bool(rt_active),
        budget_primary_active=bool(bp_active),
        team_consensus_controller_v5_active=bool(tcc_v5_active),
        repair_trajectory_cid=str(repair_trajectory_cid),
        dominant_repair_l1=int(dominant_repair_l1),
        budget_primary_gate_mean=float(budget_primary_gate_mean),
    )


__all__ = [
    "W70_DEEP_SUBSTRATE_HYBRID_V15_SCHEMA_VERSION",
    "DeepSubstrateHybridV15",
    "DeepSubstrateHybridV15ForwardWitness",
    "deep_substrate_hybrid_v15_forward",
]
