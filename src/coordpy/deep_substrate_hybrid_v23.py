"""W78 — Deep Substrate Hybrid V23.

Strictly extends W77's ``coordpy.deep_substrate_hybrid_v22``. V22
ran a *twenty-two-way* loop. V23 runs a *twenty-three-way* loop
with the V23 substrate at its centre:

  V23 latent ↔ tiny_substrate_v23 ↔ cache_controller_v21
  ↔ replay_controller_v19 ↔ ... ↔ long_horizon_reconstruction_axis.

The twenty-three-way flag is set when **all twenty-three** axes
fire on the same step.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from .cache_controller_v21 import CacheControllerV21
from .deep_substrate_hybrid_v22 import (
    DeepSubstrateHybridV22,
    DeepSubstrateHybridV22ForwardWitness,
)
from .replay_controller_v19 import ReplayControllerV19
from .tiny_substrate_v3 import _sha256_hex


W78_DEEP_SUBSTRATE_HYBRID_V23_SCHEMA_VERSION: str = (
    "coordpy.deep_substrate_hybrid_v23.v1")


@dataclasses.dataclass
class DeepSubstrateHybridV23:
    inner_v22: DeepSubstrateHybridV22 | None = None
    twenty_three_way_active: bool = False

    def cid(self) -> str:
        return _sha256_hex({
            "schema":
                W78_DEEP_SUBSTRATE_HYBRID_V23_SCHEMA_VERSION,
            "kind": "deep_substrate_hybrid_v23",
            "inner_v22_cid": (
                str(self.inner_v22.cid())
                if self.inner_v22 is not None else "none"),
            "twenty_three_way_active": bool(
                self.twenty_three_way_active),
        })


@dataclasses.dataclass(frozen=True)
class DeepSubstrateHybridV23ForwardWitness:
    schema: str
    hybrid_cid: str
    inner_v22_witness_cid: str
    twenty_three_way: bool
    cache_controller_v21_fired: bool
    replay_controller_v19_fired: bool
    long_horizon_reconstruction_trajectory_active: bool
    long_horizon_reconstruction_repair_active: bool
    team_consensus_controller_v13_active: bool
    long_horizon_reconstruction_trajectory_cid: str
    long_horizon_reconstruction_repair_l1: int
    long_horizon_reconstruction_pressure_gate_mean: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "hybrid_cid": str(self.hybrid_cid),
            "inner_v22_witness_cid": str(
                self.inner_v22_witness_cid),
            "twenty_three_way": bool(self.twenty_three_way),
            "cache_controller_v21_fired": bool(
                self.cache_controller_v21_fired),
            "replay_controller_v19_fired": bool(
                self.replay_controller_v19_fired),
            "long_horizon_reconstruction_trajectory_active": bool(
                self
                .long_horizon_reconstruction_trajectory_active),
            "long_horizon_reconstruction_repair_active": bool(
                self.long_horizon_reconstruction_repair_active),
            "team_consensus_controller_v13_active": bool(
                self.team_consensus_controller_v13_active),
            "long_horizon_reconstruction_trajectory_cid": str(
                self.long_horizon_reconstruction_trajectory_cid),
            "long_horizon_reconstruction_repair_l1": int(
                self.long_horizon_reconstruction_repair_l1),
            "long_horizon_reconstruction_pressure_gate_mean":
                float(round(
                    self
                    .long_horizon_reconstruction_pressure_gate_mean,
                    12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "deep_substrate_hybrid_v23_witness",
            "witness": self.to_dict()})


def deep_substrate_hybrid_v23_forward(
        *, hybrid: DeepSubstrateHybridV23,
        v22_witness: DeepSubstrateHybridV22ForwardWitness,
        cache_controller_v21: CacheControllerV21 | None = None,
        replay_controller_v19: ReplayControllerV19 | None = None,
        long_horizon_reconstruction_trajectory_cid: str = "",
        long_horizon_reconstruction_repair_l1: int = 0,
        long_horizon_reconstruction_pressure_gate_mean: float = (
            0.0),
        n_team_consensus_v13_invocations: int = 0,
) -> DeepSubstrateHybridV23ForwardWitness:
    cache_v21_fired = bool(
        cache_controller_v21 is not None
        and (cache_controller_v21.eighteen_objective_head
             is not None
             or len(
                cache_controller_v21
                .per_role_long_horizon_reconstruction_pressure_heads_v21)
             > 0))
    replay_v19_fired = bool(
        replay_controller_v19 is not None
        and (replay_controller_v19
             .long_horizon_reconstruction_aware_routing_head
             is not None
             and len(
                replay_controller_v19
                .per_role_per_regime_heads_v19) > 0))
    lhr_active = bool(
        len(str(long_horizon_reconstruction_trajectory_cid)) > 0)
    repair_active = bool(
        int(long_horizon_reconstruction_repair_l1) > 0)
    tcc_v13_active = bool(
        int(n_team_consensus_v13_invocations) > 0)
    twenty_three_way = bool(
        v22_witness.twenty_two_way
        and cache_v21_fired and replay_v19_fired
        and lhr_active and repair_active and tcc_v13_active)
    hybrid.twenty_three_way_active = bool(twenty_three_way)
    return DeepSubstrateHybridV23ForwardWitness(
        schema=W78_DEEP_SUBSTRATE_HYBRID_V23_SCHEMA_VERSION,
        hybrid_cid=str(hybrid.cid()),
        inner_v22_witness_cid=str(v22_witness.cid()),
        twenty_three_way=bool(twenty_three_way),
        cache_controller_v21_fired=bool(cache_v21_fired),
        replay_controller_v19_fired=bool(replay_v19_fired),
        long_horizon_reconstruction_trajectory_active=bool(
            lhr_active),
        long_horizon_reconstruction_repair_active=bool(
            repair_active),
        team_consensus_controller_v13_active=bool(tcc_v13_active),
        long_horizon_reconstruction_trajectory_cid=str(
            long_horizon_reconstruction_trajectory_cid),
        long_horizon_reconstruction_repair_l1=int(
            long_horizon_reconstruction_repair_l1),
        long_horizon_reconstruction_pressure_gate_mean=float(
            long_horizon_reconstruction_pressure_gate_mean),
    )


__all__ = [
    "W78_DEEP_SUBSTRATE_HYBRID_V23_SCHEMA_VERSION",
    "DeepSubstrateHybridV23",
    "DeepSubstrateHybridV23ForwardWitness",
    "deep_substrate_hybrid_v23_forward",
]
