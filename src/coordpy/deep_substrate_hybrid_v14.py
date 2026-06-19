"""W69 — Deep Substrate Hybrid V14.

Strictly extends W68's ``coordpy.deep_substrate_hybrid_v13``. V13
ran a *thirteen-way* loop. V14 runs a *fourteen-way* loop with the
V14 substrate at its centre:

  V14 latent ↔ tiny_substrate_v14 ↔ cache_controller_v12
  ↔ replay_controller_v10 ↔ retrieval_head
  ↔ attention_steering_bridge_v13 ↔ five_way_bridge_classifier
  ↔ prefix_state_bridge_v13 ↔ hidden_state_bridge_v13
  ↔ multi_agent_substrate_coordinator_v5
  ↔ team_consensus_controller_v4
  ↔ multi_branch_rejoin_witness_axis
  ↔ silent_corruption_witness_axis
  ↔ substrate_self_checksum_axis.

The fourteen-way flag is set when **all fourteen** axes fire on
the same step.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from .cache_controller_v12 import CacheControllerV12
from .deep_substrate_hybrid_v13 import (
    DeepSubstrateHybridV13,
    DeepSubstrateHybridV13ForwardWitness,
)
from .replay_controller_v10 import ReplayControllerV10
from .tiny_substrate_v3 import _sha256_hex


W69_DEEP_SUBSTRATE_HYBRID_V14_SCHEMA_VERSION: str = (
    "coordpy.deep_substrate_hybrid_v14.v1")


@dataclasses.dataclass
class DeepSubstrateHybridV14:
    inner_v13: DeepSubstrateHybridV13 | None = None
    fourteen_way_active: bool = False

    def cid(self) -> str:
        return _sha256_hex({
            "schema":
                W69_DEEP_SUBSTRATE_HYBRID_V14_SCHEMA_VERSION,
            "kind": "deep_substrate_hybrid_v14",
            "inner_v13_cid": (
                str(self.inner_v13.cid())
                if self.inner_v13 is not None else "none"),
            "fourteen_way_active": bool(self.fourteen_way_active),
        })


@dataclasses.dataclass(frozen=True)
class DeepSubstrateHybridV14ForwardWitness:
    schema: str
    hybrid_cid: str
    inner_v13_witness_cid: str
    fourteen_way: bool
    cache_controller_v12_fired: bool
    replay_controller_v10_fired: bool
    multi_branch_rejoin_witness_active: bool
    silent_corruption_active: bool
    substrate_self_checksum_active: bool
    team_consensus_controller_v4_active: bool
    multi_branch_rejoin_witness_l1: float
    silent_corruption_count: int
    substrate_self_checksum_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "hybrid_cid": str(self.hybrid_cid),
            "inner_v13_witness_cid": str(
                self.inner_v13_witness_cid),
            "fourteen_way": bool(self.fourteen_way),
            "cache_controller_v12_fired": bool(
                self.cache_controller_v12_fired),
            "replay_controller_v10_fired": bool(
                self.replay_controller_v10_fired),
            "multi_branch_rejoin_witness_active": bool(
                self.multi_branch_rejoin_witness_active),
            "silent_corruption_active": bool(
                self.silent_corruption_active),
            "substrate_self_checksum_active": bool(
                self.substrate_self_checksum_active),
            "team_consensus_controller_v4_active": bool(
                self.team_consensus_controller_v4_active),
            "multi_branch_rejoin_witness_l1": float(round(
                self.multi_branch_rejoin_witness_l1, 12)),
            "silent_corruption_count": int(
                self.silent_corruption_count),
            "substrate_self_checksum_cid": str(
                self.substrate_self_checksum_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "deep_substrate_hybrid_v14_witness",
            "witness": self.to_dict()})


def deep_substrate_hybrid_v14_forward(
        *, hybrid: DeepSubstrateHybridV14,
        v13_witness: DeepSubstrateHybridV13ForwardWitness,
        cache_controller_v12: CacheControllerV12 | None = None,
        replay_controller_v10: ReplayControllerV10 | None = None,
        multi_branch_rejoin_witness_l1: float = 0.0,
        silent_corruption_count: int = 0,
        substrate_self_checksum_cid: str = "",
        n_team_consensus_v4_invocations: int = 0,
) -> DeepSubstrateHybridV14ForwardWitness:
    cache_v12_fired = bool(
        cache_controller_v12 is not None
        and (cache_controller_v12.nine_objective_head is not None
              or len(
                cache_controller_v12
                .per_role_silent_corruption_heads_v12) > 0))
    replay_v10_fired = bool(
        replay_controller_v10 is not None
        and (replay_controller_v10
              .multi_branch_rejoin_routing_head is not None
              and len(
                replay_controller_v10
                .per_role_per_regime_heads_v10) > 0))
    mbr_active = bool(
        float(multi_branch_rejoin_witness_l1) > 0.0)
    sc_active = bool(int(silent_corruption_count) > 0)
    chk_active = bool(
        len(str(substrate_self_checksum_cid)) > 0)
    tcc_v4_active = bool(
        int(n_team_consensus_v4_invocations) > 0)
    fourteen_way = bool(
        v13_witness.thirteen_way
        and cache_v12_fired and replay_v10_fired
        and mbr_active and sc_active and chk_active
        and tcc_v4_active)
    hybrid.fourteen_way_active = bool(fourteen_way)
    return DeepSubstrateHybridV14ForwardWitness(
        schema=W69_DEEP_SUBSTRATE_HYBRID_V14_SCHEMA_VERSION,
        hybrid_cid=str(hybrid.cid()),
        inner_v13_witness_cid=str(v13_witness.cid()),
        fourteen_way=bool(fourteen_way),
        cache_controller_v12_fired=bool(cache_v12_fired),
        replay_controller_v10_fired=bool(replay_v10_fired),
        multi_branch_rejoin_witness_active=bool(mbr_active),
        silent_corruption_active=bool(sc_active),
        substrate_self_checksum_active=bool(chk_active),
        team_consensus_controller_v4_active=bool(tcc_v4_active),
        multi_branch_rejoin_witness_l1=float(
            multi_branch_rejoin_witness_l1),
        silent_corruption_count=int(silent_corruption_count),
        substrate_self_checksum_cid=str(
            substrate_self_checksum_cid),
    )


__all__ = [
    "W69_DEEP_SUBSTRATE_HYBRID_V14_SCHEMA_VERSION",
    "DeepSubstrateHybridV14",
    "DeepSubstrateHybridV14ForwardWitness",
    "deep_substrate_hybrid_v14_forward",
]
