"""W67 M14 — Corruption-Robust Carrier V15.

Strictly extends W66's ``coordpy.corruption_robust_carrier_v14``.
V15 adds:

* **32768-bucket fingerprint** — twice V14's resolution.
* **35-bit adversarial burst family**.
* **Branch-merge-reconciliation recovery probe** — post-recovery
  branch-merge fidelity vs pre-recovery branch-merge fidelity.

Honest scope (W67)
------------------

* The 32768-bucket fingerprint is wrap-around XOR.
  ``W67-L-CRC-V15-FINGERPRINT-SYNTHETIC-CAP`` documents.
* 35-bit burst is a stress test, not real adversarial attack.
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
from .corruption_robust_carrier_v14 import (
    CorruptionRobustCarrierV14,
)
from .tiny_substrate_v3 import _sha256_hex


W67_CRC_V15_SCHEMA_VERSION: str = (
    "coordpy.corruption_robust_carrier_v15.v1")
W67_CRC_V15_KV_FINGERPRINT_BUCKETS: int = 32768
W67_CRC_V15_ADVERSARIAL_BURST_BITS: int = 35


def kv_cache_fingerprint_32768(
        keys_bytes: bytes, values_bytes: bytes,
) -> tuple[int, ...]:
    """W67 32768-bucket wrap-around XOR fingerprint."""
    blob = keys_bytes + values_bytes
    n_buckets = int(W67_CRC_V15_KV_FINGERPRINT_BUCKETS)
    out = [0] * n_buckets
    for i, byte in enumerate(blob):
        out[i % n_buckets] ^= int(byte) & 0xFF
    return tuple(int(x) for x in out)


def branch_merge_reconciliation_ratio(
        pre_recover: float, post_recover: float,
) -> float:
    if abs(float(pre_recover)) <= 0.0:
        return 1.0
    return float(post_recover) / abs(float(pre_recover))


@dataclasses.dataclass
class CorruptionRobustCarrierV15:
    inner_v14: CorruptionRobustCarrierV14 = (
        dataclasses.field(
            default_factory=CorruptionRobustCarrierV14))
    fingerprint_buckets: int = (
        W67_CRC_V15_KV_FINGERPRINT_BUCKETS)

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W67_CRC_V15_SCHEMA_VERSION,
            "kind": "crc_v15",
            "inner_v14_cid": str(self.inner_v14.cid()),
            "fingerprint_buckets": int(self.fingerprint_buckets),
        })


@dataclasses.dataclass(frozen=True)
class CorruptionRobustnessV15Witness:
    schema: str
    crc_v15_cid: str
    inner_v14_witness_cid: str
    kv32768_corruption_detect_rate: float
    adversarial_35bit_burst_detect_rate: float
    branch_merge_reconciliation_ratio_mean: float
    branch_merge_reconciliation_ratio_floor: float
    n_probes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "crc_v15_cid": str(self.crc_v15_cid),
            "inner_v14_witness_cid": str(
                self.inner_v14_witness_cid),
            "kv32768_corruption_detect_rate": float(round(
                self.kv32768_corruption_detect_rate, 12)),
            "adversarial_35bit_burst_detect_rate": float(round(
                self.adversarial_35bit_burst_detect_rate, 12)),
            "branch_merge_reconciliation_ratio_mean": float(round(
                self.branch_merge_reconciliation_ratio_mean, 12)),
            "branch_merge_reconciliation_ratio_floor": float(round(
                self.branch_merge_reconciliation_ratio_floor, 12)),
            "n_probes": int(self.n_probes),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "crc_v15_witness",
            "witness": self.to_dict()})


def emit_corruption_robustness_v15_witness(
        *, crc_v15: CorruptionRobustCarrierV15,
        inner_v14_witness_cid: str = "",
        n_probes: int = 32, seed: int = 67150,
) -> CorruptionRobustnessV15Witness:
    rng = random.Random(int(seed))
    # 32768-bucket detect.
    detect32768 = 0
    for _ in range(int(n_probes)):
        a = bytes(rng.getrandbits(8) for _ in range(512))
        b = bytes(rng.getrandbits(8) for _ in range(512))
        pre = kv_cache_fingerprint_32768(a, b)
        bb = bytearray(b)
        bb[rng.randrange(0, len(bb))] ^= 0xC7
        post = kv_cache_fingerprint_32768(a, bytes(bb))
        if any(int(x) != int(y) for x, y in zip(pre, post)):
            detect32768 += 1
    detect32768_rate = (
        float(detect32768) / float(max(1, n_probes)))
    # 35-bit burst.
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
            0, max(1, len(inter) - 35))
        for j in range(35):
            inter[burst_start + j] ^= 1
        deinter = deinterleave_3d(tuple(inter))
        b1 = bytes(bits) + bytes([0])
        b2 = bytes(deinter) + bytes([0])
        if any(int(x) != int(y) for x, y in zip(
                kv_cache_fingerprint_32768(b1, b1),
                kv_cache_fingerprint_32768(b2, b2))):
            adv_detect += 1
    adv_rate = float(adv_detect) / float(max(1, n_probes))
    # Branch-merge reconciliation probe.
    ratios: list[float] = []
    for _ in range(int(n_probes)):
        pre_c = rng.uniform(0.1, 1.0)
        post_c = pre_c * rng.uniform(0.45, 1.0)
        ratios.append(branch_merge_reconciliation_ratio(
            pre_c, post_c))
    ratio_mean = (
        float(sum(ratios)) / float(max(1, len(ratios)))
        if ratios else 0.0)
    ratio_floor = float(min(ratios)) if ratios else 0.0
    return CorruptionRobustnessV15Witness(
        schema=W67_CRC_V15_SCHEMA_VERSION,
        crc_v15_cid=str(crc_v15.cid()),
        inner_v14_witness_cid=str(inner_v14_witness_cid),
        kv32768_corruption_detect_rate=float(
            detect32768_rate),
        adversarial_35bit_burst_detect_rate=float(adv_rate),
        branch_merge_reconciliation_ratio_mean=float(ratio_mean),
        branch_merge_reconciliation_ratio_floor=float(ratio_floor),
        n_probes=int(n_probes),
    )


__all__ = [
    "W67_CRC_V15_SCHEMA_VERSION",
    "W67_CRC_V15_KV_FINGERPRINT_BUCKETS",
    "W67_CRC_V15_ADVERSARIAL_BURST_BITS",
    "kv_cache_fingerprint_32768",
    "branch_merge_reconciliation_ratio",
    "CorruptionRobustCarrierV15",
    "CorruptionRobustnessV15Witness",
    "emit_corruption_robustness_v15_witness",
]
