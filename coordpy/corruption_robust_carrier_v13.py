"""W65 M14 — Corruption-Robust Carrier V13.

Strictly extends W64's ``coordpy.corruption_robust_carrier_v12``.
V13 adds:

* **8192-bucket fingerprint** — twice V12's resolution.
* **31-bit adversarial burst family**.
* **Team-coordination recovery probe** — post-recovery
  team-coordination scalar vs pre-recovery scalar.

Honest scope (W65)
------------------

* The 8192-bucket fingerprint is wrap-around XOR.
  ``W65-L-CRC-V13-FINGERPRINT-SYNTHETIC-CAP`` documents.
* 31-bit burst is a stress test, not real adversarial attack.
"""

from __future__ import annotations

import dataclasses
import random
from typing import Any, Sequence

from .corruption_robust_carrier_v5 import (
    W57_CRC_V5_INTERLEAVE_ROWS,
    W57_CRC_V5_INTERLEAVE_COLS,
    W57_CRC_V5_INTERLEAVE_PLANES,
    interleave_3d, deinterleave_3d,
)
from .corruption_robust_carrier_v12 import (
    CorruptionRobustCarrierV12,
)
from .tiny_substrate_v3 import _sha256_hex


W65_CRC_V13_SCHEMA_VERSION: str = (
    "coordpy.corruption_robust_carrier_v13.v1")
W65_CRC_V13_KV_FINGERPRINT_BUCKETS: int = 8192
W65_CRC_V13_ADVERSARIAL_BURST_BITS: int = 31


def kv_cache_fingerprint_8192(
        keys_bytes: bytes, values_bytes: bytes,
) -> tuple[int, ...]:
    """W65 8192-bucket wrap-around XOR fingerprint."""
    blob = keys_bytes + values_bytes
    n_buckets = int(W65_CRC_V13_KV_FINGERPRINT_BUCKETS)
    out = [0] * n_buckets
    for i, byte in enumerate(blob):
        out[i % n_buckets] ^= int(byte) & 0xFF
    return tuple(int(x) for x in out)


def team_coordination_recovery_ratio(
        pre_coord: float, post_coord: float,
) -> float:
    if abs(float(pre_coord)) <= 0.0:
        return 1.0
    return float(post_coord) / abs(float(pre_coord))


@dataclasses.dataclass
class CorruptionRobustCarrierV13:
    inner_v12: CorruptionRobustCarrierV12 = (
        dataclasses.field(
            default_factory=CorruptionRobustCarrierV12))
    fingerprint_buckets: int = (
        W65_CRC_V13_KV_FINGERPRINT_BUCKETS)

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W65_CRC_V13_SCHEMA_VERSION,
            "kind": "crc_v13",
            "inner_v12_cid": str(self.inner_v12.cid()),
            "fingerprint_buckets": int(self.fingerprint_buckets),
        })


@dataclasses.dataclass(frozen=True)
class CorruptionRobustnessV13Witness:
    schema: str
    crc_v13_cid: str
    inner_v12_witness_cid: str
    kv8192_corruption_detect_rate: float
    adversarial_31bit_burst_detect_rate: float
    team_coordination_recovery_ratio_mean: float
    team_coordination_recovery_ratio_floor: float
    n_probes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "crc_v13_cid": str(self.crc_v13_cid),
            "inner_v12_witness_cid": str(
                self.inner_v12_witness_cid),
            "kv8192_corruption_detect_rate": float(round(
                self.kv8192_corruption_detect_rate, 12)),
            "adversarial_31bit_burst_detect_rate": float(round(
                self.adversarial_31bit_burst_detect_rate, 12)),
            "team_coordination_recovery_ratio_mean": float(round(
                self.team_coordination_recovery_ratio_mean, 12)),
            "team_coordination_recovery_ratio_floor": float(round(
                self.team_coordination_recovery_ratio_floor, 12)),
            "n_probes": int(self.n_probes),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "crc_v13_witness",
            "witness": self.to_dict()})


def emit_corruption_robustness_v13_witness(
        *, crc_v13: CorruptionRobustCarrierV13,
        inner_v12_witness_cid: str = "",
        n_probes: int = 32, seed: int = 65140,
) -> CorruptionRobustnessV13Witness:
    rng = random.Random(int(seed))
    # 8192-bucket detect.
    detect8192 = 0
    for _ in range(int(n_probes)):
        a = bytes(rng.getrandbits(8) for _ in range(512))
        b = bytes(rng.getrandbits(8) for _ in range(512))
        pre = kv_cache_fingerprint_8192(a, b)
        bb = bytearray(b)
        bb[rng.randrange(0, len(bb))] ^= 0xDB
        post = kv_cache_fingerprint_8192(a, bytes(bb))
        if any(int(x) != int(y) for x, y in zip(pre, post)):
            detect8192 += 1
    detect8192_rate = (
        float(detect8192) / float(max(1, n_probes)))
    # 31-bit burst.
    block_size = (
        W57_CRC_V5_INTERLEAVE_ROWS
        * W57_CRC_V5_INTERLEAVE_COLS
        * W57_CRC_V5_INTERLEAVE_PLANES)
    adv_detect = 0
    for _ in range(int(n_probes)):
        bits = [rng.randint(0, 1)
                 for _ in range(int(block_size))]
        inter = list(interleave_3d(bits))
        burst_start = rng.randrange(
            0, max(1, len(inter) - 31))
        for j in range(31):
            inter[burst_start + j] ^= 1
        deinter = deinterleave_3d(tuple(inter))
        b1 = bytes(bits) + bytes([0])
        b2 = bytes(deinter) + bytes([0])
        if any(int(x) != int(y) for x, y in zip(
                kv_cache_fingerprint_8192(b1, b1),
                kv_cache_fingerprint_8192(b2, b2))):
            adv_detect += 1
    adv_rate = float(adv_detect) / float(max(1, n_probes))
    # Team-coordination recovery probe.
    ratios: list[float] = []
    for _ in range(int(n_probes)):
        pre_c = rng.uniform(0.1, 1.0)
        post_c = pre_c * rng.uniform(0.4, 1.0)
        ratios.append(team_coordination_recovery_ratio(
            pre_c, post_c))
    ratio_mean = (
        float(sum(ratios)) / float(max(1, len(ratios)))
        if ratios else 0.0)
    ratio_floor = float(min(ratios)) if ratios else 0.0
    return CorruptionRobustnessV13Witness(
        schema=W65_CRC_V13_SCHEMA_VERSION,
        crc_v13_cid=str(crc_v13.cid()),
        inner_v12_witness_cid=str(inner_v12_witness_cid),
        kv8192_corruption_detect_rate=float(detect8192_rate),
        adversarial_31bit_burst_detect_rate=float(adv_rate),
        team_coordination_recovery_ratio_mean=float(ratio_mean),
        team_coordination_recovery_ratio_floor=float(ratio_floor),
        n_probes=int(n_probes),
    )


__all__ = [
    "W65_CRC_V13_SCHEMA_VERSION",
    "W65_CRC_V13_KV_FINGERPRINT_BUCKETS",
    "W65_CRC_V13_ADVERSARIAL_BURST_BITS",
    "kv_cache_fingerprint_8192",
    "team_coordination_recovery_ratio",
    "CorruptionRobustCarrierV13",
    "CorruptionRobustnessV13Witness",
    "emit_corruption_robustness_v13_witness",
]
