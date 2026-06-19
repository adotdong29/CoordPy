"""W79 — Deep Substrate Hybrid V24.

Strictly extends W78's ``coordpy.deep_substrate_hybrid_v23`` with
a **twenty-four-way bidirectional loop** that adds the
replacement-then-restart-after-long-delay channel.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any

from .deep_substrate_hybrid_v23 import (
    DeepSubstrateHybridV23,
    DeepSubstrateHybridV23ForwardWitness,
    deep_substrate_hybrid_v23_forward,
)


W79_DEEP_SUBSTRATE_HYBRID_V24_SCHEMA_VERSION: str = (
    "coordpy.deep_substrate_hybrid_v24.v1")


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass(frozen=True)
class DeepSubstrateHybridV24:
    inner_v23: DeepSubstrateHybridV23 = dataclasses.field(
        default_factory=DeepSubstrateHybridV23)

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w79_deep_substrate_hybrid_v24",
            "schema": W79_DEEP_SUBSTRATE_HYBRID_V24_SCHEMA_VERSION,
            "inner_v23_cid": str(self.inner_v23.cid()),
        })


@dataclasses.dataclass(frozen=True)
class DeepSubstrateHybridV24ForwardWitness:
    schema: str
    hybrid_cid: str
    inner_v23_witness_cid: str
    twenty_four_way: bool
    replacement_then_restart_after_long_delay_trajectory_active: bool
    replacement_then_restart_after_long_delay_repair_l1: int
    replacement_then_restart_after_long_delay_pressure_gate_mean: (
        float)
    team_consensus_controller_v14_active: bool
    replacement_then_restart_after_long_delay_trajectory_cid: str

    def cid(self) -> str:
        return _sha256_hex({
            "kind":
                "w79_deep_substrate_hybrid_v24_forward_witness",
            "schema": str(self.schema),
            "hybrid_cid": str(self.hybrid_cid),
            "inner_v23_witness_cid": str(
                self.inner_v23_witness_cid),
            "twenty_four_way": bool(self.twenty_four_way),
            "replacement_then_restart_after_long_delay_trajectory_active": bool(
                self
                .replacement_then_restart_after_long_delay_trajectory_active),
            "replacement_then_restart_after_long_delay_repair_l1": int(
                self
                .replacement_then_restart_after_long_delay_repair_l1),
            "replacement_then_restart_after_long_delay_pressure_gate_mean": float(
                round(
                    self
                    .replacement_then_restart_after_long_delay_pressure_gate_mean,
                    12)),
            "team_consensus_controller_v14_active": bool(
                self.team_consensus_controller_v14_active),
            "replacement_then_restart_after_long_delay_trajectory_cid": str(
                self
                .replacement_then_restart_after_long_delay_trajectory_cid),
        })


def deep_substrate_hybrid_v24_forward(
        *,
        hybrid: DeepSubstrateHybridV24,
        v23_witness: DeepSubstrateHybridV23ForwardWitness,
        replacement_then_restart_after_long_delay_trajectory_cid: (
            str) = "",
        replacement_then_restart_after_long_delay_repair_l1: int = 0,
        replacement_then_restart_after_long_delay_pressure_gate_mean: (
            float) = 0.0,
        n_team_consensus_v14_invocations: int = 0,
) -> DeepSubstrateHybridV24ForwardWitness:
    twenty_four_way = bool(
        v23_witness.twenty_three_way
        and (
            str(
                replacement_then_restart_after_long_delay_trajectory_cid)
            != ""
            or int(
                replacement_then_restart_after_long_delay_repair_l1)
            > 0))
    return DeepSubstrateHybridV24ForwardWitness(
        schema=W79_DEEP_SUBSTRATE_HYBRID_V24_SCHEMA_VERSION,
        hybrid_cid=str(hybrid.cid()),
        inner_v23_witness_cid=str(v23_witness.cid()),
        twenty_four_way=bool(twenty_four_way),
        replacement_then_restart_after_long_delay_trajectory_active=bool(
            str(
                replacement_then_restart_after_long_delay_trajectory_cid)
            != ""),
        replacement_then_restart_after_long_delay_repair_l1=int(
            replacement_then_restart_after_long_delay_repair_l1),
        replacement_then_restart_after_long_delay_pressure_gate_mean=float(
            replacement_then_restart_after_long_delay_pressure_gate_mean),
        team_consensus_controller_v14_active=bool(
            int(n_team_consensus_v14_invocations) > 0),
        replacement_then_restart_after_long_delay_trajectory_cid=str(
            replacement_then_restart_after_long_delay_trajectory_cid),
    )


__all__ = [
    "W79_DEEP_SUBSTRATE_HYBRID_V24_SCHEMA_VERSION",
    "DeepSubstrateHybridV24",
    "DeepSubstrateHybridV24ForwardWitness",
    "deep_substrate_hybrid_v24_forward",
]
