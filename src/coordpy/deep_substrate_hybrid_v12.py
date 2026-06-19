"""W67 — Deep Substrate Hybrid V12.

Strictly extends W66's ``coordpy.deep_substrate_hybrid_v11``. V11
ran an *eleven-way* loop. V12 runs a *twelve-way* loop with the V12
substrate at its centre:

  V12 latent ↔ tiny_substrate_v12 ↔ cache_controller_v10
  ↔ replay_controller_v8 ↔ retrieval_head
  ↔ attention_steering_bridge_v11 ↔ four_way_bridge_classifier
  ↔ prefix_state_bridge_v11 ↔ hidden_state_bridge_v11
  ↔ multi_agent_substrate_coordinator_v3
  ↔ team_consensus_controller_v2
  ↔ branch_merge_witness_axis.

The twelve-way flag is set when **all twelve** axes fire on the
same step (each of the eleven V11 axes plus the new
branch_merge_witness axis).
"""

from __future__ import annotations

import dataclasses
from typing import Any

from .cache_controller_v10 import CacheControllerV10
from .deep_substrate_hybrid_v11 import (
    DeepSubstrateHybridV11,
    DeepSubstrateHybridV11ForwardWitness,
)
from .replay_controller_v8 import ReplayControllerV8
from .tiny_substrate_v3 import _sha256_hex


W67_DEEP_SUBSTRATE_HYBRID_V12_SCHEMA_VERSION: str = (
    "coordpy.deep_substrate_hybrid_v12.v1")


@dataclasses.dataclass
class DeepSubstrateHybridV12:
    inner_v11: DeepSubstrateHybridV11 | None = None
    twelve_way_active: bool = False

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W67_DEEP_SUBSTRATE_HYBRID_V12_SCHEMA_VERSION,
            "kind": "deep_substrate_hybrid_v12",
            "inner_v11_cid": (
                str(self.inner_v11.cid())
                if self.inner_v11 is not None else "none"),
            "twelve_way_active": bool(self.twelve_way_active),
        })


@dataclasses.dataclass(frozen=True)
class DeepSubstrateHybridV12ForwardWitness:
    schema: str
    hybrid_cid: str
    inner_v11_witness_cid: str
    twelve_way: bool
    cache_controller_v10_fired: bool
    replay_controller_v8_fired: bool
    branch_merge_witness_active: bool
    role_dropout_recovery_active: bool
    substrate_snapshot_fork_active: bool
    team_consensus_controller_v2_active: bool
    branch_merge_witness_l1: float
    role_dropout_recovery_count: int
    n_branches_active: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "hybrid_cid": str(self.hybrid_cid),
            "inner_v11_witness_cid": str(
                self.inner_v11_witness_cid),
            "twelve_way": bool(self.twelve_way),
            "cache_controller_v10_fired": bool(
                self.cache_controller_v10_fired),
            "replay_controller_v8_fired": bool(
                self.replay_controller_v8_fired),
            "branch_merge_witness_active": bool(
                self.branch_merge_witness_active),
            "role_dropout_recovery_active": bool(
                self.role_dropout_recovery_active),
            "substrate_snapshot_fork_active": bool(
                self.substrate_snapshot_fork_active),
            "team_consensus_controller_v2_active": bool(
                self.team_consensus_controller_v2_active),
            "branch_merge_witness_l1": float(round(
                self.branch_merge_witness_l1, 12)),
            "role_dropout_recovery_count": int(
                self.role_dropout_recovery_count),
            "n_branches_active": int(self.n_branches_active),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "deep_substrate_hybrid_v12_witness",
            "witness": self.to_dict()})


def deep_substrate_hybrid_v12_forward(
        *, hybrid: DeepSubstrateHybridV12,
        v11_witness: DeepSubstrateHybridV11ForwardWitness,
        cache_controller_v10: CacheControllerV10 | None = None,
        replay_controller_v8: ReplayControllerV8 | None = None,
        branch_merge_witness_l1: float = 0.0,
        role_dropout_recovery_count: int = 0,
        n_branches_active: int = 0,
        n_team_consensus_v2_invocations: int = 0,
) -> DeepSubstrateHybridV12ForwardWitness:
    cache_v10_fired = bool(
        cache_controller_v10 is not None
        and (cache_controller_v10.seven_objective_head is not None
              or len(
                cache_controller_v10
                .per_role_eviction_heads_v10) > 0))
    replay_v8_fired = bool(
        replay_controller_v8 is not None
        and (replay_controller_v8.branch_merge_routing_head
              is not None
              and len(
                replay_controller_v8
                .per_role_per_regime_heads_v8) > 0))
    bm_active = bool(float(branch_merge_witness_l1) > 0.0)
    rd_active = bool(int(role_dropout_recovery_count) > 0)
    fork_active = bool(int(n_branches_active) > 0)
    tcc_v2_active = bool(
        int(n_team_consensus_v2_invocations) > 0)
    twelve_way = bool(
        v11_witness.eleven_way
        and cache_v10_fired and replay_v8_fired
        and bm_active and rd_active
        and fork_active and tcc_v2_active)
    hybrid.twelve_way_active = bool(twelve_way)
    return DeepSubstrateHybridV12ForwardWitness(
        schema=W67_DEEP_SUBSTRATE_HYBRID_V12_SCHEMA_VERSION,
        hybrid_cid=str(hybrid.cid()),
        inner_v11_witness_cid=str(v11_witness.cid()),
        twelve_way=bool(twelve_way),
        cache_controller_v10_fired=bool(cache_v10_fired),
        replay_controller_v8_fired=bool(replay_v8_fired),
        branch_merge_witness_active=bool(bm_active),
        role_dropout_recovery_active=bool(rd_active),
        substrate_snapshot_fork_active=bool(fork_active),
        team_consensus_controller_v2_active=bool(tcc_v2_active),
        branch_merge_witness_l1=float(branch_merge_witness_l1),
        role_dropout_recovery_count=int(
            role_dropout_recovery_count),
        n_branches_active=int(n_branches_active),
    )


__all__ = [
    "W67_DEEP_SUBSTRATE_HYBRID_V12_SCHEMA_VERSION",
    "DeepSubstrateHybridV12",
    "DeepSubstrateHybridV12ForwardWitness",
    "deep_substrate_hybrid_v12_forward",
]
