"""W66 — Deep Substrate Hybrid V11.

Strictly extends W65's ``coordpy.deep_substrate_hybrid_v10``. V10
ran a *ten-way* loop. V11 runs an *eleven-way* loop with the V11
substrate at its centre:

  V11 latent ↔ tiny_substrate_v11 ↔ cache_controller_v9
  ↔ replay_controller_v7 ↔ retrieval_head
  ↔ attention_steering_bridge_v10 ↔ four_way_bridge_classifier
  ↔ prefix_state_bridge_v10 ↔ hidden_state_bridge_v10
  ↔ multi_agent_substrate_coordinator_v2
  ↔ team_consensus_controller.

The eleven-way flag is set when **all eleven** axes fire on the
same step (each of the ten V10 axes plus the new
team_consensus_controller axis).
"""

from __future__ import annotations

import dataclasses
from typing import Any

from .cache_controller_v9 import CacheControllerV9
from .deep_substrate_hybrid_v10 import (
    DeepSubstrateHybridV10,
    DeepSubstrateHybridV10ForwardWitness,
)
from .replay_controller_v7 import ReplayControllerV7
from .tiny_substrate_v3 import _sha256_hex


W66_DEEP_SUBSTRATE_HYBRID_V11_SCHEMA_VERSION: str = (
    "coordpy.deep_substrate_hybrid_v11.v1")


@dataclasses.dataclass
class DeepSubstrateHybridV11:
    inner_v10: DeepSubstrateHybridV10 | None = None
    eleven_way_active: bool = False

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W66_DEEP_SUBSTRATE_HYBRID_V11_SCHEMA_VERSION,
            "kind": "deep_substrate_hybrid_v11",
            "inner_v10_cid": (
                str(self.inner_v10.cid())
                if self.inner_v10 is not None else "none"),
            "eleven_way_active": bool(self.eleven_way_active),
        })


@dataclasses.dataclass(frozen=True)
class DeepSubstrateHybridV11ForwardWitness:
    schema: str
    hybrid_cid: str
    inner_v10_witness_cid: str
    eleven_way: bool
    cache_controller_v9_fired: bool
    replay_controller_v7_fired: bool
    replay_trust_active: bool
    team_failure_recovery_active: bool
    substrate_snapshot_diff_active: bool
    team_consensus_controller_active: bool
    replay_trust_l1: float
    team_failure_recovery_count: int
    n_team_consensus_invocations: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "hybrid_cid": str(self.hybrid_cid),
            "inner_v10_witness_cid": str(
                self.inner_v10_witness_cid),
            "eleven_way": bool(self.eleven_way),
            "cache_controller_v9_fired": bool(
                self.cache_controller_v9_fired),
            "replay_controller_v7_fired": bool(
                self.replay_controller_v7_fired),
            "replay_trust_active": bool(
                self.replay_trust_active),
            "team_failure_recovery_active": bool(
                self.team_failure_recovery_active),
            "substrate_snapshot_diff_active": bool(
                self.substrate_snapshot_diff_active),
            "team_consensus_controller_active": bool(
                self.team_consensus_controller_active),
            "replay_trust_l1": float(round(
                self.replay_trust_l1, 12)),
            "team_failure_recovery_count": int(
                self.team_failure_recovery_count),
            "n_team_consensus_invocations": int(
                self.n_team_consensus_invocations),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "deep_substrate_hybrid_v11_witness",
            "witness": self.to_dict()})


def deep_substrate_hybrid_v11_forward(
        *, hybrid: DeepSubstrateHybridV11,
        v10_witness: DeepSubstrateHybridV10ForwardWitness,
        cache_controller_v9: CacheControllerV9 | None = None,
        replay_controller_v7: ReplayControllerV7 | None = None,
        replay_trust_l1: float = 0.0,
        team_failure_recovery_count: int = 0,
        substrate_snapshot_diff_l1: float = 0.0,
        n_team_consensus_invocations: int = 0,
) -> DeepSubstrateHybridV11ForwardWitness:
    cache_v9_fired = bool(
        cache_controller_v9 is not None
        and (cache_controller_v9.six_objective_head is not None
              or len(
                cache_controller_v9
                .per_role_eviction_heads_v9) > 0))
    replay_v7_fired = bool(
        replay_controller_v7 is not None
        and (replay_controller_v7.team_substrate_routing_head
              is not None
              and len(
                replay_controller_v7
                .per_role_per_regime_heads_v7) > 0))
    replay_trust_active = bool(float(replay_trust_l1) > 0.0)
    tfr_active = bool(int(team_failure_recovery_count) > 0)
    snapshot_active = bool(float(substrate_snapshot_diff_l1) > 0.0)
    team_consensus_active = bool(
        int(n_team_consensus_invocations) > 0)
    eleven_way = bool(
        v10_witness.ten_way
        and cache_v9_fired and replay_v7_fired
        and replay_trust_active and tfr_active
        and snapshot_active and team_consensus_active)
    hybrid.eleven_way_active = bool(eleven_way)
    return DeepSubstrateHybridV11ForwardWitness(
        schema=W66_DEEP_SUBSTRATE_HYBRID_V11_SCHEMA_VERSION,
        hybrid_cid=str(hybrid.cid()),
        inner_v10_witness_cid=str(v10_witness.cid()),
        eleven_way=bool(eleven_way),
        cache_controller_v9_fired=bool(cache_v9_fired),
        replay_controller_v7_fired=bool(replay_v7_fired),
        replay_trust_active=bool(replay_trust_active),
        team_failure_recovery_active=bool(tfr_active),
        substrate_snapshot_diff_active=bool(snapshot_active),
        team_consensus_controller_active=bool(
            team_consensus_active),
        replay_trust_l1=float(replay_trust_l1),
        team_failure_recovery_count=int(
            team_failure_recovery_count),
        n_team_consensus_invocations=int(
            n_team_consensus_invocations),
    )


__all__ = [
    "W66_DEEP_SUBSTRATE_HYBRID_V11_SCHEMA_VERSION",
    "DeepSubstrateHybridV11",
    "DeepSubstrateHybridV11ForwardWitness",
    "deep_substrate_hybrid_v11_forward",
]
