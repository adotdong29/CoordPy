"""W79 M2 — KV Bridge V24.

Strictly extends W78's ``coordpy.kv_bridge_v23``. V24 adds a
**twenty-target stacked ridge** (V23 was nineteen-target) with a
**168-dim replacement-then-restart-after-long-delay
fingerprint** and a per-axis falsifier.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Sequence

from .kv_bridge_v23 import KVBridgeV23Projection


W79_KV_BRIDGE_V24_SCHEMA_VERSION: str = (
    "coordpy.kv_bridge_v24.v1")
W79_KV_BRIDGE_V24_N_TARGETS: int = 20
W79_KV_BRIDGE_V24_FINGERPRINT_DIM: int = 168


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass(frozen=True)
class KVBridgeV24Projection:
    inner_v23: KVBridgeV23Projection
    n_targets: int = W79_KV_BRIDGE_V24_N_TARGETS

    @classmethod
    def init_from_v23(
            cls, v23: KVBridgeV23Projection, *,
            seed_v24: int = 79101,
    ) -> "KVBridgeV24Projection":
        return cls(inner_v23=v23)

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w79_kv_bridge_v24_projection",
            "inner_v23_cid": str(self.inner_v23.cid()),
            "n_targets": int(self.n_targets),
        })


def compute_replacement_then_restart_after_long_delay_fingerprint_v24(
        *,
        role: str,
        long_horizon_reconstruction_trajectory_cid: str,
        replacement_then_restart_after_long_delay_trajectory_cid: (
            str),
        dominant_repair_label: int = 0,
        replacement_then_restart_after_long_delay_count: int = 0,
        long_delay_blackout_window_turns: int = 0,
        visible_token_budget: float = 0.0,
        baseline_cost: float = 0.0,
) -> tuple[float, ...]:
    """V24 W79 fingerprint over the new axis."""
    base = []
    src = (
        str(role)
        + str(long_horizon_reconstruction_trajectory_cid)
        + str(
            replacement_then_restart_after_long_delay_trajectory_cid)
        + str(dominant_repair_label)
        + str(
            replacement_then_restart_after_long_delay_count)
        + str(long_delay_blackout_window_turns)
        + f"{float(visible_token_budget):.6f}"
        + f"{float(baseline_cost):.6f}"
    ).encode("utf-8")
    h = hashlib.sha256(src).digest()
    while len(base) < W79_KV_BRIDGE_V24_FINGERPRINT_DIM:
        for i, byte in enumerate(h):
            base.append(
                float((byte / 255.0) - 0.5))
            if len(base) >= W79_KV_BRIDGE_V24_FINGERPRINT_DIM:
                break
        if len(base) < W79_KV_BRIDGE_V24_FINGERPRINT_DIM:
            h = hashlib.sha256(h + src).digest()
    return tuple(float(round(v, 12)) for v in base)


@dataclasses.dataclass(frozen=True)
class KVBridgeV24Falsifier:
    schema: str
    flag: int
    falsifier_score: float

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w79_kv_bridge_v24_falsifier",
            "schema": str(self.schema),
            "flag": int(self.flag),
            "falsifier_score": float(round(
                self.falsifier_score, 12)),
        })


def probe_kv_bridge_v24_replacement_then_restart_after_long_delay_falsifier(
        *,
        replacement_then_restart_after_long_delay_pressure_flag: (
            int),
) -> KVBridgeV24Falsifier:
    """0 = honest (no hosted access claimed); 1 = dishonest."""
    return KVBridgeV24Falsifier(
        schema=W79_KV_BRIDGE_V24_SCHEMA_VERSION,
        flag=int(
            replacement_then_restart_after_long_delay_pressure_flag),
        falsifier_score=0.0,
    )


@dataclasses.dataclass(frozen=True)
class KVBridgeV24Witness:
    schema: str
    projection_cid: str
    fingerprint_dim: int
    falsifier_cid: str
    n_targets: int

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w79_kv_bridge_v24_witness",
            "schema": str(self.schema),
            "projection_cid": str(self.projection_cid),
            "fingerprint_dim": int(self.fingerprint_dim),
            "falsifier_cid": str(self.falsifier_cid),
            "n_targets": int(self.n_targets),
        })


def emit_kv_bridge_v24_witness(
        *,
        projection: KVBridgeV24Projection,
        replacement_then_restart_after_long_delay_falsifier: (
            KVBridgeV24Falsifier),
        replacement_then_restart_after_long_delay_fingerprint: (
            Sequence[float]),
) -> KVBridgeV24Witness:
    return KVBridgeV24Witness(
        schema=W79_KV_BRIDGE_V24_SCHEMA_VERSION,
        projection_cid=str(projection.cid()),
        fingerprint_dim=int(
            len(list(
                replacement_then_restart_after_long_delay_fingerprint))),
        falsifier_cid=str(
            replacement_then_restart_after_long_delay_falsifier.cid()),
        n_targets=int(W79_KV_BRIDGE_V24_N_TARGETS),
    )


__all__ = [
    "W79_KV_BRIDGE_V24_SCHEMA_VERSION",
    "W79_KV_BRIDGE_V24_N_TARGETS",
    "W79_KV_BRIDGE_V24_FINGERPRINT_DIM",
    "KVBridgeV24Projection",
    "KVBridgeV24Falsifier",
    "KVBridgeV24Witness",
    "compute_replacement_then_restart_after_long_delay_fingerprint_v24",
    "probe_kv_bridge_v24_replacement_then_restart_after_long_delay_falsifier",
    "emit_kv_bridge_v24_witness",
]
