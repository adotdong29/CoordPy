"""W77 — Deep Substrate Hybrid V22.

Strictly extends W76's ``coordpy.deep_substrate_hybrid_v21``. V21
ran a *twenty-one-way* loop. V22 runs a *twenty-two-way* loop with
the V22 substrate at its centre:

  V22 latent ↔ tiny_substrate_v22 ↔ cache_controller_v20
  ↔ replay_controller_v18 ↔ retrieval_head
  ↔ attention_steering_bridge_v13 ↔ five_way_bridge_classifier
  ↔ prefix_state_bridge_v13 ↔ hidden_state_bridge_v13
  ↔ multi_agent_substrate_coordinator_v13
  ↔ team_consensus_controller_v12
  ↔ multi_branch_rejoin_witness_axis
  ↔ silent_corruption_witness_axis
  ↔ substrate_self_checksum_axis
  ↔ repair_trajectory_axis
  ↔ delayed_repair_trajectory_axis
  ↔ restart_repair_trajectory_axis
  ↔ replacement_repair_trajectory_axis
  ↔ compound_repair_trajectory_axis
  ↔ compound_chain_repair_trajectory_axis
  ↔ chain_then_restart_trajectory_axis
  ↔ post_restart_replacement_trajectory_axis.

The twenty-two-way flag is set when **all twenty-two** axes fire
on the same step.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from .cache_controller_v20 import CacheControllerV20
from .deep_substrate_hybrid_v21 import (
    DeepSubstrateHybridV21,
    DeepSubstrateHybridV21ForwardWitness,
)
from .replay_controller_v18 import ReplayControllerV18
from .tiny_substrate_v3 import _sha256_hex


W77_DEEP_SUBSTRATE_HYBRID_V22_SCHEMA_VERSION: str = (
    "coordpy.deep_substrate_hybrid_v22.v1")


@dataclasses.dataclass
class DeepSubstrateHybridV22:
    inner_v21: DeepSubstrateHybridV21 | None = None
    twenty_two_way_active: bool = False

    def cid(self) -> str:
        return _sha256_hex({
            "schema":
                W77_DEEP_SUBSTRATE_HYBRID_V22_SCHEMA_VERSION,
            "kind": "deep_substrate_hybrid_v22",
            "inner_v21_cid": (
                str(self.inner_v21.cid())
                if self.inner_v21 is not None else "none"),
            "twenty_two_way_active": bool(
                self.twenty_two_way_active),
        })


@dataclasses.dataclass(frozen=True)
class DeepSubstrateHybridV22ForwardWitness:
    schema: str
    hybrid_cid: str
    inner_v21_witness_cid: str
    twenty_two_way: bool
    cache_controller_v20_fired: bool
    replay_controller_v18_fired: bool
    post_restart_replacement_trajectory_active: bool
    post_restart_replacement_repair_active: bool
    team_consensus_controller_v12_active: bool
    post_restart_replacement_trajectory_cid: str
    post_restart_replacement_repair_l1: int
    post_restart_replacement_pressure_gate_mean: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "hybrid_cid": str(self.hybrid_cid),
            "inner_v21_witness_cid": str(
                self.inner_v21_witness_cid),
            "twenty_two_way": bool(self.twenty_two_way),
            "cache_controller_v20_fired": bool(
                self.cache_controller_v20_fired),
            "replay_controller_v18_fired": bool(
                self.replay_controller_v18_fired),
            "post_restart_replacement_trajectory_active": bool(
                self.post_restart_replacement_trajectory_active),
            "post_restart_replacement_repair_active": bool(
                self.post_restart_replacement_repair_active),
            "team_consensus_controller_v12_active": bool(
                self.team_consensus_controller_v12_active),
            "post_restart_replacement_trajectory_cid": str(
                self.post_restart_replacement_trajectory_cid),
            "post_restart_replacement_repair_l1": int(
                self.post_restart_replacement_repair_l1),
            "post_restart_replacement_pressure_gate_mean": float(
                round(
                    self
                    .post_restart_replacement_pressure_gate_mean,
                    12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "deep_substrate_hybrid_v22_witness",
            "witness": self.to_dict()})


def deep_substrate_hybrid_v22_forward(
        *, hybrid: DeepSubstrateHybridV22,
        v21_witness: DeepSubstrateHybridV21ForwardWitness,
        cache_controller_v20: CacheControllerV20 | None = None,
        replay_controller_v18: ReplayControllerV18 | None = None,
        post_restart_replacement_trajectory_cid: str = "",
        post_restart_replacement_repair_l1: int = 0,
        post_restart_replacement_pressure_gate_mean: float = 0.0,
        n_team_consensus_v12_invocations: int = 0,
) -> DeepSubstrateHybridV22ForwardWitness:
    cache_v20_fired = bool(
        cache_controller_v20 is not None
        and (cache_controller_v20.seventeen_objective_head
             is not None
             or len(
                cache_controller_v20
                .per_role_post_restart_replacement_pressure_heads_v20)
             > 0))
    replay_v18_fired = bool(
        replay_controller_v18 is not None
        and (replay_controller_v18
             .post_restart_replacement_aware_routing_head
             is not None
             and len(
                replay_controller_v18
                .per_role_per_regime_heads_v18) > 0))
    pcr_active = bool(
        len(str(post_restart_replacement_trajectory_cid)) > 0)
    repair_active = bool(
        int(post_restart_replacement_repair_l1) > 0)
    tcc_v12_active = bool(
        int(n_team_consensus_v12_invocations) > 0)
    twenty_two_way = bool(
        v21_witness.twenty_one_way
        and cache_v20_fired and replay_v18_fired
        and pcr_active and repair_active and tcc_v12_active)
    hybrid.twenty_two_way_active = bool(twenty_two_way)
    return DeepSubstrateHybridV22ForwardWitness(
        schema=W77_DEEP_SUBSTRATE_HYBRID_V22_SCHEMA_VERSION,
        hybrid_cid=str(hybrid.cid()),
        inner_v21_witness_cid=str(v21_witness.cid()),
        twenty_two_way=bool(twenty_two_way),
        cache_controller_v20_fired=bool(cache_v20_fired),
        replay_controller_v18_fired=bool(replay_v18_fired),
        post_restart_replacement_trajectory_active=bool(
            pcr_active),
        post_restart_replacement_repair_active=bool(
            repair_active),
        team_consensus_controller_v12_active=bool(tcc_v12_active),
        post_restart_replacement_trajectory_cid=str(
            post_restart_replacement_trajectory_cid),
        post_restart_replacement_repair_l1=int(
            post_restart_replacement_repair_l1),
        post_restart_replacement_pressure_gate_mean=float(
            post_restart_replacement_pressure_gate_mean),
    )


__all__ = [
    "W77_DEEP_SUBSTRATE_HYBRID_V22_SCHEMA_VERSION",
    "DeepSubstrateHybridV22",
    "DeepSubstrateHybridV22ForwardWitness",
    "deep_substrate_hybrid_v22_forward",
]
