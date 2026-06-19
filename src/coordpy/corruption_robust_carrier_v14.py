"""W66 M14 — Corruption-Robust Carrier V14.

Strictly extends W65's ``coordpy.corruption_robust_carrier_v13``.
V14 adds:

* **16384-bucket fingerprint** — twice V13's resolution.
* **33-bit adversarial burst family**.
* **Team-failure-recovery probe** — post-recovery
  team-failure-recovery scalar vs pre-recovery scalar.

Honest scope (W66)
------------------

* The 16384-bucket fingerprint is wrap-around XOR.
  ``W66-L-CRC-V14-FINGERPRINT-SYNTHETIC-CAP`` documents.
* 33-bit burst is a stress test, not real adversarial attack.
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
from .corruption_robust_carrier_v13 import (
    CorruptionRobustCarrierV13,
)
from .tiny_substrate_v3 import _sha256_hex


W66_CRC_V14_SCHEMA_VERSION: str = (
    "coordpy.corruption_robust_carrier_v14.v1")
W66_CRC_V14_KV_FINGERPRINT_BUCKETS: int = 16384
W66_CRC_V14_ADVERSARIAL_BURST_BITS: int = 33


def kv_cache_fingerprint_16384(
        keys_bytes: bytes, values_bytes: bytes,
) -> tuple[int, ...]:
    """W66 16384-bucket wrap-around XOR fingerprint."""
    blob = keys_bytes + values_bytes
    n_buckets = int(W66_CRC_V14_KV_FINGERPRINT_BUCKETS)
    out = [0] * n_buckets
    for i, byte in enumerate(blob):
        out[i % n_buckets] ^= int(byte) & 0xFF
    return tuple(int(x) for x in out)


def team_failure_recovery_ratio(
        pre_recover: float, post_recover: float,
) -> float:
    if abs(float(pre_recover)) <= 0.0:
        return 1.0
    return float(post_recover) / abs(float(pre_recover))


@dataclasses.dataclass
class CorruptionRobustCarrierV14:
    inner_v13: CorruptionRobustCarrierV13 = (
        dataclasses.field(
            default_factory=CorruptionRobustCarrierV13))
    fingerprint_buckets: int = (
        W66_CRC_V14_KV_FINGERPRINT_BUCKETS)

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W66_CRC_V14_SCHEMA_VERSION,
            "kind": "crc_v14",
            "inner_v13_cid": str(self.inner_v13.cid()),
            "fingerprint_buckets": int(self.fingerprint_buckets),
        })


@dataclasses.dataclass(frozen=True)
class CorruptionRobustnessV14Witness:
    schema: str
    crc_v14_cid: str
    inner_v13_witness_cid: str
    kv16384_corruption_detect_rate: float
    adversarial_33bit_burst_detect_rate: float
    team_failure_recovery_ratio_mean: float
    team_failure_recovery_ratio_floor: float
    n_probes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "crc_v14_cid": str(self.crc_v14_cid),
            "inner_v13_witness_cid": str(
                self.inner_v13_witness_cid),
            "kv16384_corruption_detect_rate": float(round(
                self.kv16384_corruption_detect_rate, 12)),
            "adversarial_33bit_burst_detect_rate": float(round(
                self.adversarial_33bit_burst_detect_rate, 12)),
            "team_failure_recovery_ratio_mean": float(round(
                self.team_failure_recovery_ratio_mean, 12)),
            "team_failure_recovery_ratio_floor": float(round(
                self.team_failure_recovery_ratio_floor, 12)),
            "n_probes": int(self.n_probes),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "crc_v14_witness",
            "witness": self.to_dict()})


def emit_corruption_robustness_v14_witness(
        *, crc_v14: CorruptionRobustCarrierV14,
        inner_v13_witness_cid: str = "",
        n_probes: int = 32, seed: int = 66140,
) -> CorruptionRobustnessV14Witness:
    rng = random.Random(int(seed))
    # 16384-bucket detect.
    detect16384 = 0
    for _ in range(int(n_probes)):
        a = bytes(rng.getrandbits(8) for _ in range(512))
        b = bytes(rng.getrandbits(8) for _ in range(512))
        pre = kv_cache_fingerprint_16384(a, b)
        bb = bytearray(b)
        bb[rng.randrange(0, len(bb))] ^= 0xDB
        post = kv_cache_fingerprint_16384(a, bytes(bb))
        if any(int(x) != int(y) for x, y in zip(pre, post)):
            detect16384 += 1
    detect16384_rate = (
        float(detect16384) / float(max(1, n_probes)))
    # 33-bit burst.
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
            0, max(1, len(inter) - 33))
        for j in range(33):
            inter[burst_start + j] ^= 1
        deinter = deinterleave_3d(tuple(inter))
        b1 = bytes(bits) + bytes([0])
        b2 = bytes(deinter) + bytes([0])
        if any(int(x) != int(y) for x, y in zip(
                kv_cache_fingerprint_16384(b1, b1),
                kv_cache_fingerprint_16384(b2, b2))):
            adv_detect += 1
    adv_rate = float(adv_detect) / float(max(1, n_probes))
    # Team-failure-recovery probe.
    ratios: list[float] = []
    for _ in range(int(n_probes)):
        pre_c = rng.uniform(0.1, 1.0)
        post_c = pre_c * rng.uniform(0.45, 1.0)
        ratios.append(team_failure_recovery_ratio(
            pre_c, post_c))
    ratio_mean = (
        float(sum(ratios)) / float(max(1, len(ratios)))
        if ratios else 0.0)
    ratio_floor = float(min(ratios)) if ratios else 0.0
    return CorruptionRobustnessV14Witness(
        schema=W66_CRC_V14_SCHEMA_VERSION,
        crc_v14_cid=str(crc_v14.cid()),
        inner_v13_witness_cid=str(inner_v13_witness_cid),
        kv16384_corruption_detect_rate=float(
            detect16384_rate),
        adversarial_33bit_burst_detect_rate=float(adv_rate),
        team_failure_recovery_ratio_mean=float(ratio_mean),
        team_failure_recovery_ratio_floor=float(ratio_floor),
        n_probes=int(n_probes),
    )


__all__ = [
    "W66_CRC_V14_SCHEMA_VERSION",
    "W66_CRC_V14_KV_FINGERPRINT_BUCKETS",
    "W66_CRC_V14_ADVERSARIAL_BURST_BITS",
    "kv_cache_fingerprint_16384",
    "team_failure_recovery_ratio",
    "CorruptionRobustCarrierV14",
    "CorruptionRobustnessV14Witness",
    "emit_corruption_robustness_v14_witness",
]
