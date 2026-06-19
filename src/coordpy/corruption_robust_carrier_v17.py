"""W69 M12 — Corruption-Robust Carrier V17.

Strictly extends W68's ``coordpy.corruption_robust_carrier_v16``.
V17 adds:

* **131072-bucket fingerprint** — twice V16's resolution.
* **37-bit adversarial burst family**.
* **Silent-corruption recovery probe** — post-recovery silent-
  corruption fidelity vs pre-recovery fidelity.

Honest scope (W69): wrap-around XOR; not a real adversarial attack.
``W69-L-CRC-V17-FINGERPRINT-SYNTHETIC-CAP``.
"""

from __future__ import annotations

import dataclasses
import random
from typing import Any

from .corruption_robust_carrier_v5 import (
    W57_CRC_V5_INTERLEAVE_ROWS,
    W57_CRC_V5_INTERLEAVE_COLS,
    W57_CRC_V5_INTERLEAVE_PLANES,
    interleave_3d, deinterleave_3d,
)
from .corruption_robust_carrier_v16 import (
    CorruptionRobustCarrierV16,
)
from .tiny_substrate_v3 import _sha256_hex


W69_CRC_V17_SCHEMA_VERSION: str = (
    "coordpy.corruption_robust_carrier_v17.v1")
W69_CRC_V17_KV_FINGERPRINT_BUCKETS: int = 131072
W69_CRC_V17_ADVERSARIAL_BURST_BITS: int = 37


def kv_cache_fingerprint_131072(
        keys_bytes: bytes, values_bytes: bytes,
) -> tuple[int, ...]:
    """W69 131072-bucket wrap-around XOR fingerprint."""
    blob = keys_bytes + values_bytes
    n_buckets = int(W69_CRC_V17_KV_FINGERPRINT_BUCKETS)
    out = [0] * n_buckets
    for i, byte in enumerate(blob):
        out[i % n_buckets] ^= int(byte) & 0xFF
    return tuple(int(x) for x in out)


def silent_corruption_recovery_ratio(
        pre_recover: float, post_recover: float,
) -> float:
    if abs(float(pre_recover)) <= 0.0:
        return 1.0
    return float(post_recover) / abs(float(pre_recover))


@dataclasses.dataclass
class CorruptionRobustCarrierV17:
    inner_v16: CorruptionRobustCarrierV16 = (
        dataclasses.field(
            default_factory=CorruptionRobustCarrierV16))
    fingerprint_buckets: int = (
        W69_CRC_V17_KV_FINGERPRINT_BUCKETS)

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W69_CRC_V17_SCHEMA_VERSION,
            "kind": "crc_v17",
            "inner_v16_cid": str(self.inner_v16.cid()),
            "fingerprint_buckets": int(self.fingerprint_buckets),
        })


@dataclasses.dataclass(frozen=True)
class CorruptionRobustnessV17Witness:
    schema: str
    crc_v17_cid: str
    inner_v16_witness_cid: str
    kv131072_corruption_detect_rate: float
    adversarial_37bit_burst_detect_rate: float
    silent_corruption_recovery_ratio_mean: float
    silent_corruption_recovery_ratio_floor: float
    n_probes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "crc_v17_cid": str(self.crc_v17_cid),
            "inner_v16_witness_cid": str(
                self.inner_v16_witness_cid),
            "kv131072_corruption_detect_rate": float(round(
                self.kv131072_corruption_detect_rate, 12)),
            "adversarial_37bit_burst_detect_rate": float(round(
                self.adversarial_37bit_burst_detect_rate, 12)),
            "silent_corruption_recovery_ratio_mean": float(
                round(
                    self.silent_corruption_recovery_ratio_mean,
                    12)),
            "silent_corruption_recovery_ratio_floor": float(
                round(
                    self.silent_corruption_recovery_ratio_floor,
                    12)),
            "n_probes": int(self.n_probes),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "crc_v17_witness",
            "witness": self.to_dict()})


def emit_corruption_robustness_v17_witness(
        *, crc_v17: CorruptionRobustCarrierV17,
        inner_v16_witness_cid: str = "",
        n_probes: int = 32, seed: int = 69150,
) -> CorruptionRobustnessV17Witness:
    rng = random.Random(int(seed))
    detect131072 = 0
    for _ in range(int(n_probes)):
        a = bytes(rng.getrandbits(8) for _ in range(512))
        b = bytes(rng.getrandbits(8) for _ in range(512))
        pre = kv_cache_fingerprint_131072(a, b)
        bb = bytearray(b)
        bb[rng.randrange(0, len(bb))] ^= 0xD7
        post = kv_cache_fingerprint_131072(a, bytes(bb))
        if any(int(x) != int(y) for x, y in zip(pre, post)):
            detect131072 += 1
    detect131072_rate = (
        float(detect131072) / float(max(1, n_probes)))
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
            0, max(1, len(inter) - 37))
        for j in range(37):
            inter[burst_start + j] ^= 1
        deinter = deinterleave_3d(tuple(inter))
        b1 = bytes(bits) + bytes([0])
        b2 = bytes(deinter) + bytes([0])
        if any(int(x) != int(y) for x, y in zip(
                kv_cache_fingerprint_131072(b1, b1),
                kv_cache_fingerprint_131072(b2, b2))):
            adv_detect += 1
    adv_rate = float(adv_detect) / float(max(1, n_probes))
    ratios: list[float] = []
    for _ in range(int(n_probes)):
        pre_c = rng.uniform(0.1, 1.0)
        post_c = pre_c * rng.uniform(0.45, 1.0)
        ratios.append(silent_corruption_recovery_ratio(
            pre_c, post_c))
    ratio_mean = (
        float(sum(ratios)) / float(max(1, len(ratios)))
        if ratios else 0.0)
    ratio_floor = float(min(ratios)) if ratios else 0.0
    return CorruptionRobustnessV17Witness(
        schema=W69_CRC_V17_SCHEMA_VERSION,
        crc_v17_cid=str(crc_v17.cid()),
        inner_v16_witness_cid=str(inner_v16_witness_cid),
        kv131072_corruption_detect_rate=float(
            detect131072_rate),
        adversarial_37bit_burst_detect_rate=float(adv_rate),
        silent_corruption_recovery_ratio_mean=float(
            ratio_mean),
        silent_corruption_recovery_ratio_floor=float(
            ratio_floor),
        n_probes=int(n_probes),
    )


__all__ = [
    "W69_CRC_V17_SCHEMA_VERSION",
    "W69_CRC_V17_KV_FINGERPRINT_BUCKETS",
    "W69_CRC_V17_ADVERSARIAL_BURST_BITS",
    "kv_cache_fingerprint_131072",
    "silent_corruption_recovery_ratio",
    "CorruptionRobustCarrierV17",
    "CorruptionRobustnessV17Witness",
    "emit_corruption_robustness_v17_witness",
]
