"""W76 — Deep Substrate Hybrid V21.

Strictly extends W75's ``coordpy.deep_substrate_hybrid_v20``. V20
ran a *twenty-way* loop. V21 runs a *twenty-one-way* loop with the
V21 substrate at its centre:

  V21 latent ↔ tiny_substrate_v21 ↔ cache_controller_v19
  ↔ replay_controller_v17 ↔ retrieval_head
  ↔ attention_steering_bridge_v13 ↔ five_way_bridge_classifier
  ↔ prefix_state_bridge_v13 ↔ hidden_state_bridge_v13
  ↔ multi_agent_substrate_coordinator_v12
  ↔ team_consensus_controller_v11
  ↔ multi_branch_rejoin_witness_axis
  ↔ silent_corruption_witness_axis
  ↔ substrate_self_checksum_axis
  ↔ repair_trajectory_axis
  ↔ delayed_repair_trajectory_axis
  ↔ restart_repair_trajectory_axis
  ↔ replacement_repair_trajectory_axis
  ↔ compound_repair_trajectory_axis
  ↔ compound_chain_repair_trajectory_axis
  ↔ chain_then_restart_trajectory_axis.

The twenty-one-way flag is set when **all twenty-one** axes fire
on the same step.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from .cache_controller_v19 import CacheControllerV19
from .deep_substrate_hybrid_v20 import (
    DeepSubstrateHybridV20,
    DeepSubstrateHybridV20ForwardWitness,
)
from .replay_controller_v17 import ReplayControllerV17
from .tiny_substrate_v3 import _sha256_hex


W76_DEEP_SUBSTRATE_HYBRID_V21_SCHEMA_VERSION: str = (
    "coordpy.deep_substrate_hybrid_v21.v1")


@dataclasses.dataclass
class DeepSubstrateHybridV21:
    inner_v20: DeepSubstrateHybridV20 | None = None
    twenty_one_way_active: bool = False

    def cid(self) -> str:
        return _sha256_hex({
            "schema":
                W76_DEEP_SUBSTRATE_HYBRID_V21_SCHEMA_VERSION,
            "kind": "deep_substrate_hybrid_v21",
            "inner_v20_cid": (
                str(self.inner_v20.cid())
                if self.inner_v20 is not None else "none"),
            "twenty_one_way_active": bool(
                self.twenty_one_way_active),
        })


@dataclasses.dataclass(frozen=True)
class DeepSubstrateHybridV21ForwardWitness:
    schema: str
    hybrid_cid: str
    inner_v20_witness_cid: str
    twenty_one_way: bool
    cache_controller_v19_fired: bool
    replay_controller_v17_fired: bool
    chain_then_restart_trajectory_active: bool
    chain_then_restart_repair_active: bool
    team_consensus_controller_v11_active: bool
    chain_then_restart_trajectory_cid: str
    chain_then_restart_repair_l1: int
    chain_then_restart_pressure_gate_mean: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "hybrid_cid": str(self.hybrid_cid),
            "inner_v20_witness_cid": str(
                self.inner_v20_witness_cid),
            "twenty_one_way": bool(self.twenty_one_way),
            "cache_controller_v19_fired": bool(
                self.cache_controller_v19_fired),
            "replay_controller_v17_fired": bool(
                self.replay_controller_v17_fired),
            "chain_then_restart_trajectory_active": bool(
                self.chain_then_restart_trajectory_active),
            "chain_then_restart_repair_active": bool(
                self.chain_then_restart_repair_active),
            "team_consensus_controller_v11_active": bool(
                self.team_consensus_controller_v11_active),
            "chain_then_restart_trajectory_cid": str(
                self.chain_then_restart_trajectory_cid),
            "chain_then_restart_repair_l1": int(
                self.chain_then_restart_repair_l1),
            "chain_then_restart_pressure_gate_mean": float(round(
                self.chain_then_restart_pressure_gate_mean, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "deep_substrate_hybrid_v21_witness",
            "witness": self.to_dict()})


def deep_substrate_hybrid_v21_forward(
        *, hybrid: DeepSubstrateHybridV21,
        v20_witness: DeepSubstrateHybridV20ForwardWitness,
        cache_controller_v19: CacheControllerV19 | None = None,
        replay_controller_v17: ReplayControllerV17 | None = None,
        chain_then_restart_trajectory_cid: str = "",
        chain_then_restart_repair_l1: int = 0,
        chain_then_restart_pressure_gate_mean: float = 0.0,
        n_team_consensus_v11_invocations: int = 0,
) -> DeepSubstrateHybridV21ForwardWitness:
    cache_v19_fired = bool(
        cache_controller_v19 is not None
        and (cache_controller_v19.sixteen_objective_head
             is not None
             or len(
                cache_controller_v19
                .per_role_chain_then_restart_pressure_heads_v19)
             > 0))
    replay_v17_fired = bool(
        replay_controller_v17 is not None
        and (replay_controller_v17
             .chain_then_restart_aware_routing_head is not None
             and len(
                replay_controller_v17
                .per_role_per_regime_heads_v17) > 0))
    ctr_active = bool(
        len(str(chain_then_restart_trajectory_cid)) > 0)
    chain_active = bool(int(chain_then_restart_repair_l1) > 0)
    tcc_v11_active = bool(
        int(n_team_consensus_v11_invocations) > 0)
    twenty_one_way = bool(
        v20_witness.twenty_way
        and cache_v19_fired and replay_v17_fired
        and ctr_active and chain_active and tcc_v11_active)
    hybrid.twenty_one_way_active = bool(twenty_one_way)
    return DeepSubstrateHybridV21ForwardWitness(
        schema=W76_DEEP_SUBSTRATE_HYBRID_V21_SCHEMA_VERSION,
        hybrid_cid=str(hybrid.cid()),
        inner_v20_witness_cid=str(v20_witness.cid()),
        twenty_one_way=bool(twenty_one_way),
        cache_controller_v19_fired=bool(cache_v19_fired),
        replay_controller_v17_fired=bool(replay_v17_fired),
        chain_then_restart_trajectory_active=bool(ctr_active),
        chain_then_restart_repair_active=bool(chain_active),
        team_consensus_controller_v11_active=bool(tcc_v11_active),
        chain_then_restart_trajectory_cid=str(
            chain_then_restart_trajectory_cid),
        chain_then_restart_repair_l1=int(
            chain_then_restart_repair_l1),
        chain_then_restart_pressure_gate_mean=float(
            chain_then_restart_pressure_gate_mean),
    )


__all__ = [
    "W76_DEEP_SUBSTRATE_HYBRID_V21_SCHEMA_VERSION",
    "DeepSubstrateHybridV21",
    "DeepSubstrateHybridV21ForwardWitness",
    "deep_substrate_hybrid_v21_forward",
]
