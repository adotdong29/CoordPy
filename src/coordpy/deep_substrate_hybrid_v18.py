"""W73 — Deep Substrate Hybrid V18.

Strictly extends W72's ``coordpy.deep_substrate_hybrid_v17``. V17
ran a *seventeen-way* loop. V18 runs an *eighteen-way* loop with
the V18 substrate at its centre:

  V18 latent ↔ tiny_substrate_v18 ↔ cache_controller_v16
  ↔ replay_controller_v14 ↔ retrieval_head
  ↔ attention_steering_bridge_v13 ↔ five_way_bridge_classifier
  ↔ prefix_state_bridge_v13 ↔ hidden_state_bridge_v13
  ↔ multi_agent_substrate_coordinator_v9
  ↔ team_consensus_controller_v8
  ↔ multi_branch_rejoin_witness_axis
  ↔ silent_corruption_witness_axis
  ↔ substrate_self_checksum_axis
  ↔ repair_trajectory_axis
  ↔ delayed_repair_trajectory_axis
  ↔ restart_repair_trajectory_axis
  ↔ replacement_repair_trajectory_axis.

The eighteen-way flag is set when **all eighteen** axes fire on
the same step.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from .cache_controller_v16 import CacheControllerV16
from .deep_substrate_hybrid_v17 import (
    DeepSubstrateHybridV17,
    DeepSubstrateHybridV17ForwardWitness,
)
from .replay_controller_v14 import ReplayControllerV14
from .tiny_substrate_v3 import _sha256_hex


W73_DEEP_SUBSTRATE_HYBRID_V18_SCHEMA_VERSION: str = (
    "coordpy.deep_substrate_hybrid_v18.v1")


@dataclasses.dataclass
class DeepSubstrateHybridV18:
    inner_v17: DeepSubstrateHybridV17 | None = None
    eighteen_way_active: bool = False

    def cid(self) -> str:
        return _sha256_hex({
            "schema":
                W73_DEEP_SUBSTRATE_HYBRID_V18_SCHEMA_VERSION,
            "kind": "deep_substrate_hybrid_v18",
            "inner_v17_cid": (
                str(self.inner_v17.cid())
                if self.inner_v17 is not None else "none"),
            "eighteen_way_active": bool(
                self.eighteen_way_active),
        })


@dataclasses.dataclass(frozen=True)
class DeepSubstrateHybridV18ForwardWitness:
    schema: str
    hybrid_cid: str
    inner_v17_witness_cid: str
    eighteen_way: bool
    cache_controller_v16_fired: bool
    replay_controller_v14_fired: bool
    replacement_repair_trajectory_active: bool
    replacement_after_ctr_active: bool
    team_consensus_controller_v8_active: bool
    replacement_repair_trajectory_cid: str
    replacement_after_ctr_l1: int
    replacement_pressure_gate_mean: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "hybrid_cid": str(self.hybrid_cid),
            "inner_v17_witness_cid": str(
                self.inner_v17_witness_cid),
            "eighteen_way": bool(self.eighteen_way),
            "cache_controller_v16_fired": bool(
                self.cache_controller_v16_fired),
            "replay_controller_v14_fired": bool(
                self.replay_controller_v14_fired),
            "replacement_repair_trajectory_active": bool(
                self.replacement_repair_trajectory_active),
            "replacement_after_ctr_active": bool(
                self.replacement_after_ctr_active),
            "team_consensus_controller_v8_active": bool(
                self.team_consensus_controller_v8_active),
            "replacement_repair_trajectory_cid": str(
                self.replacement_repair_trajectory_cid),
            "replacement_after_ctr_l1": int(
                self.replacement_after_ctr_l1),
            "replacement_pressure_gate_mean": float(round(
                self.replacement_pressure_gate_mean, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "deep_substrate_hybrid_v18_witness",
            "witness": self.to_dict()})


def deep_substrate_hybrid_v18_forward(
        *, hybrid: DeepSubstrateHybridV18,
        v17_witness: DeepSubstrateHybridV17ForwardWitness,
        cache_controller_v16: CacheControllerV16 | None = None,
        replay_controller_v14: ReplayControllerV14 | None = None,
        replacement_repair_trajectory_cid: str = "",
        replacement_after_ctr_l1: int = 0,
        replacement_pressure_gate_mean: float = 0.0,
        n_team_consensus_v8_invocations: int = 0,
) -> DeepSubstrateHybridV18ForwardWitness:
    cache_v16_fired = bool(
        cache_controller_v16 is not None
        and (cache_controller_v16.thirteen_objective_head
             is not None
             or len(
                cache_controller_v16
                .per_role_replacement_pressure_heads_v16) > 0))
    replay_v14_fired = bool(
        replay_controller_v14 is not None
        and (replay_controller_v14
              .replacement_aware_routing_head is not None
              and len(
                replay_controller_v14
                .per_role_per_regime_heads_v14) > 0))
    rrt_active = bool(
        len(str(replacement_repair_trajectory_cid)) > 0)
    rep_active = bool(int(replacement_after_ctr_l1) > 0)
    tcc_v8_active = bool(
        int(n_team_consensus_v8_invocations) > 0)
    eighteen_way = bool(
        v17_witness.seventeen_way
        and cache_v16_fired and replay_v14_fired
        and rrt_active and rep_active and tcc_v8_active)
    hybrid.eighteen_way_active = bool(eighteen_way)
    return DeepSubstrateHybridV18ForwardWitness(
        schema=W73_DEEP_SUBSTRATE_HYBRID_V18_SCHEMA_VERSION,
        hybrid_cid=str(hybrid.cid()),
        inner_v17_witness_cid=str(v17_witness.cid()),
        eighteen_way=bool(eighteen_way),
        cache_controller_v16_fired=bool(cache_v16_fired),
        replay_controller_v14_fired=bool(replay_v14_fired),
        replacement_repair_trajectory_active=bool(rrt_active),
        replacement_after_ctr_active=bool(rep_active),
        team_consensus_controller_v8_active=bool(tcc_v8_active),
        replacement_repair_trajectory_cid=str(
            replacement_repair_trajectory_cid),
        replacement_after_ctr_l1=int(
            replacement_after_ctr_l1),
        replacement_pressure_gate_mean=float(
            replacement_pressure_gate_mean),
    )


__all__ = [
    "W73_DEEP_SUBSTRATE_HYBRID_V18_SCHEMA_VERSION",
    "DeepSubstrateHybridV18",
    "DeepSubstrateHybridV18ForwardWitness",
    "deep_substrate_hybrid_v18_forward",
]
