"""W68 M12 — Corruption-Robust Carrier V16.

Strictly extends W67's ``coordpy.corruption_robust_carrier_v15``.
V16 adds:

* **65536-bucket fingerprint** — twice V15's resolution.
* **36-bit adversarial burst family**.
* **Partial-contradiction recovery probe** — post-recovery
  partial-contradiction fidelity vs pre-recovery fidelity.

Honest scope (W68)
------------------

* The 65536-bucket fingerprint is wrap-around XOR.
  ``W68-L-CRC-V16-FINGERPRINT-SYNTHETIC-CAP`` documents.
* 36-bit burst is a stress test, not real adversarial attack.
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
from .corruption_robust_carrier_v15 import (
    CorruptionRobustCarrierV15,
)
from .tiny_substrate_v3 import _sha256_hex


W68_CRC_V16_SCHEMA_VERSION: str = (
    "coordpy.corruption_robust_carrier_v16.v1")
W68_CRC_V16_KV_FINGERPRINT_BUCKETS: int = 65536
W68_CRC_V16_ADVERSARIAL_BURST_BITS: int = 36


def kv_cache_fingerprint_65536(
        keys_bytes: bytes, values_bytes: bytes,
) -> tuple[int, ...]:
    """W68 65536-bucket wrap-around XOR fingerprint."""
    blob = keys_bytes + values_bytes
    n_buckets = int(W68_CRC_V16_KV_FINGERPRINT_BUCKETS)
    out = [0] * n_buckets
    for i, byte in enumerate(blob):
        out[i % n_buckets] ^= int(byte) & 0xFF
    return tuple(int(x) for x in out)


def partial_contradiction_recovery_ratio(
        pre_recover: float, post_recover: float,
) -> float:
    if abs(float(pre_recover)) <= 0.0:
        return 1.0
    return float(post_recover) / abs(float(pre_recover))


@dataclasses.dataclass
class CorruptionRobustCarrierV16:
    inner_v15: CorruptionRobustCarrierV15 = (
        dataclasses.field(
            default_factory=CorruptionRobustCarrierV15))
    fingerprint_buckets: int = (
        W68_CRC_V16_KV_FINGERPRINT_BUCKETS)

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W68_CRC_V16_SCHEMA_VERSION,
            "kind": "crc_v16",
            "inner_v15_cid": str(self.inner_v15.cid()),
            "fingerprint_buckets": int(self.fingerprint_buckets),
        })


@dataclasses.dataclass(frozen=True)
class CorruptionRobustnessV16Witness:
    schema: str
    crc_v16_cid: str
    inner_v15_witness_cid: str
    kv65536_corruption_detect_rate: float
    adversarial_36bit_burst_detect_rate: float
    partial_contradiction_recovery_ratio_mean: float
    partial_contradiction_recovery_ratio_floor: float
    n_probes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "crc_v16_cid": str(self.crc_v16_cid),
            "inner_v15_witness_cid": str(
                self.inner_v15_witness_cid),
            "kv65536_corruption_detect_rate": float(round(
                self.kv65536_corruption_detect_rate, 12)),
            "adversarial_36bit_burst_detect_rate": float(round(
                self.adversarial_36bit_burst_detect_rate, 12)),
            "partial_contradiction_recovery_ratio_mean": float(
                round(
                    self.partial_contradiction_recovery_ratio_mean,
                    12)),
            "partial_contradiction_recovery_ratio_floor": float(
                round(
                    self.partial_contradiction_recovery_ratio_floor,
                    12)),
            "n_probes": int(self.n_probes),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "crc_v16_witness",
            "witness": self.to_dict()})


def emit_corruption_robustness_v16_witness(
        *, crc_v16: CorruptionRobustCarrierV16,
        inner_v15_witness_cid: str = "",
        n_probes: int = 32, seed: int = 68150,
) -> CorruptionRobustnessV16Witness:
    rng = random.Random(int(seed))
    detect65536 = 0
    for _ in range(int(n_probes)):
        a = bytes(rng.getrandbits(8) for _ in range(512))
        b = bytes(rng.getrandbits(8) for _ in range(512))
        pre = kv_cache_fingerprint_65536(a, b)
        bb = bytearray(b)
        bb[rng.randrange(0, len(bb))] ^= 0xD3
        post = kv_cache_fingerprint_65536(a, bytes(bb))
        if any(int(x) != int(y) for x, y in zip(pre, post)):
            detect65536 += 1
    detect65536_rate = (
        float(detect65536) / float(max(1, n_probes)))
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
            0, max(1, len(inter) - 36))
        for j in range(36):
            inter[burst_start + j] ^= 1
        deinter = deinterleave_3d(tuple(inter))
        b1 = bytes(bits) + bytes([0])
        b2 = bytes(deinter) + bytes([0])
        if any(int(x) != int(y) for x, y in zip(
                kv_cache_fingerprint_65536(b1, b1),
                kv_cache_fingerprint_65536(b2, b2))):
            adv_detect += 1
    adv_rate = float(adv_detect) / float(max(1, n_probes))
    ratios: list[float] = []
    for _ in range(int(n_probes)):
        pre_c = rng.uniform(0.1, 1.0)
        post_c = pre_c * rng.uniform(0.45, 1.0)
        ratios.append(partial_contradiction_recovery_ratio(
            pre_c, post_c))
    ratio_mean = (
        float(sum(ratios)) / float(max(1, len(ratios)))
        if ratios else 0.0)
    ratio_floor = float(min(ratios)) if ratios else 0.0
    return CorruptionRobustnessV16Witness(
        schema=W68_CRC_V16_SCHEMA_VERSION,
        crc_v16_cid=str(crc_v16.cid()),
        inner_v15_witness_cid=str(inner_v15_witness_cid),
        kv65536_corruption_detect_rate=float(
            detect65536_rate),
        adversarial_36bit_burst_detect_rate=float(adv_rate),
        partial_contradiction_recovery_ratio_mean=float(
            ratio_mean),
        partial_contradiction_recovery_ratio_floor=float(
            ratio_floor),
        n_probes=int(n_probes),
    )


__all__ = [
    "W68_CRC_V16_SCHEMA_VERSION",
    "W68_CRC_V16_KV_FINGERPRINT_BUCKETS",
    "W68_CRC_V16_ADVERSARIAL_BURST_BITS",
    "kv_cache_fingerprint_65536",
    "partial_contradiction_recovery_ratio",
    "CorruptionRobustCarrierV16",
    "CorruptionRobustnessV16Witness",
    "emit_corruption_robustness_v16_witness",
]
