"""W68 — Deep Substrate Hybrid V13.

Strictly extends W67's ``coordpy.deep_substrate_hybrid_v12``. V12
ran a *twelve-way* loop. V13 runs a *thirteen-way* loop with the
V13 substrate at its centre:

  V13 latent ↔ tiny_substrate_v13 ↔ cache_controller_v11
  ↔ replay_controller_v9 ↔ retrieval_head
  ↔ attention_steering_bridge_v12 ↔ four_way_bridge_classifier
  ↔ prefix_state_bridge_v12 ↔ hidden_state_bridge_v12
  ↔ multi_agent_substrate_coordinator_v4
  ↔ team_consensus_controller_v3
  ↔ branch_merge_witness_axis
  ↔ partial_contradiction_witness_axis.

The thirteen-way flag is set when **all thirteen** axes fire on
the same step.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from .cache_controller_v11 import CacheControllerV11
from .deep_substrate_hybrid_v12 import (
    DeepSubstrateHybridV12,
    DeepSubstrateHybridV12ForwardWitness,
)
from .replay_controller_v9 import ReplayControllerV9
from .tiny_substrate_v3 import _sha256_hex


W68_DEEP_SUBSTRATE_HYBRID_V13_SCHEMA_VERSION: str = (
    "coordpy.deep_substrate_hybrid_v13.v1")


@dataclasses.dataclass
class DeepSubstrateHybridV13:
    inner_v12: DeepSubstrateHybridV12 | None = None
    thirteen_way_active: bool = False

    def cid(self) -> str:
        return _sha256_hex({
            "schema":
                W68_DEEP_SUBSTRATE_HYBRID_V13_SCHEMA_VERSION,
            "kind": "deep_substrate_hybrid_v13",
            "inner_v12_cid": (
                str(self.inner_v12.cid())
                if self.inner_v12 is not None else "none"),
            "thirteen_way_active": bool(self.thirteen_way_active),
        })


@dataclasses.dataclass(frozen=True)
class DeepSubstrateHybridV13ForwardWitness:
    schema: str
    hybrid_cid: str
    inner_v12_witness_cid: str
    thirteen_way: bool
    cache_controller_v11_fired: bool
    replay_controller_v9_fired: bool
    partial_contradiction_witness_active: bool
    agent_replacement_active: bool
    prefix_reuse_active: bool
    team_consensus_controller_v3_active: bool
    partial_contradiction_witness_l1: float
    agent_replacement_count: int
    prefix_reuse_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "hybrid_cid": str(self.hybrid_cid),
            "inner_v12_witness_cid": str(
                self.inner_v12_witness_cid),
            "thirteen_way": bool(self.thirteen_way),
            "cache_controller_v11_fired": bool(
                self.cache_controller_v11_fired),
            "replay_controller_v9_fired": bool(
                self.replay_controller_v9_fired),
            "partial_contradiction_witness_active": bool(
                self.partial_contradiction_witness_active),
            "agent_replacement_active": bool(
                self.agent_replacement_active),
            "prefix_reuse_active": bool(self.prefix_reuse_active),
            "team_consensus_controller_v3_active": bool(
                self.team_consensus_controller_v3_active),
            "partial_contradiction_witness_l1": float(round(
                self.partial_contradiction_witness_l1, 12)),
            "agent_replacement_count": int(
                self.agent_replacement_count),
            "prefix_reuse_count": int(self.prefix_reuse_count),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "deep_substrate_hybrid_v13_witness",
            "witness": self.to_dict()})


def deep_substrate_hybrid_v13_forward(
        *, hybrid: DeepSubstrateHybridV13,
        v12_witness: DeepSubstrateHybridV12ForwardWitness,
        cache_controller_v11: CacheControllerV11 | None = None,
        replay_controller_v9: ReplayControllerV9 | None = None,
        partial_contradiction_witness_l1: float = 0.0,
        agent_replacement_count: int = 0,
        prefix_reuse_count: int = 0,
        n_team_consensus_v3_invocations: int = 0,
) -> DeepSubstrateHybridV13ForwardWitness:
    cache_v11_fired = bool(
        cache_controller_v11 is not None
        and (cache_controller_v11.eight_objective_head is not None
              or len(
                cache_controller_v11
                .per_role_agent_replacement_heads_v11) > 0))
    replay_v9_fired = bool(
        replay_controller_v9 is not None
        and (replay_controller_v9.agent_replacement_routing_head
              is not None
              and len(
                replay_controller_v9
                .per_role_per_regime_heads_v9) > 0))
    pc_active = bool(
        float(partial_contradiction_witness_l1) > 0.0)
    ar_active = bool(int(agent_replacement_count) > 0)
    pr_active = bool(int(prefix_reuse_count) > 0)
    tcc_v3_active = bool(
        int(n_team_consensus_v3_invocations) > 0)
    thirteen_way = bool(
        v12_witness.twelve_way
        and cache_v11_fired and replay_v9_fired
        and pc_active and ar_active
        and pr_active and tcc_v3_active)
    hybrid.thirteen_way_active = bool(thirteen_way)
    return DeepSubstrateHybridV13ForwardWitness(
        schema=W68_DEEP_SUBSTRATE_HYBRID_V13_SCHEMA_VERSION,
        hybrid_cid=str(hybrid.cid()),
        inner_v12_witness_cid=str(v12_witness.cid()),
        thirteen_way=bool(thirteen_way),
        cache_controller_v11_fired=bool(cache_v11_fired),
        replay_controller_v9_fired=bool(replay_v9_fired),
        partial_contradiction_witness_active=bool(pc_active),
        agent_replacement_active=bool(ar_active),
        prefix_reuse_active=bool(pr_active),
        team_consensus_controller_v3_active=bool(tcc_v3_active),
        partial_contradiction_witness_l1=float(
            partial_contradiction_witness_l1),
        agent_replacement_count=int(
            agent_replacement_count),
        prefix_reuse_count=int(prefix_reuse_count),
    )


__all__ = [
    "W68_DEEP_SUBSTRATE_HYBRID_V13_SCHEMA_VERSION",
    "DeepSubstrateHybridV13",
    "DeepSubstrateHybridV13ForwardWitness",
    "deep_substrate_hybrid_v13_forward",
]
