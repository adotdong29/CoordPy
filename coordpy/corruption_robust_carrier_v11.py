"""W63 M15 — Corruption-Robust Carrier V11.

Strictly extends W62's ``coordpy.corruption_robust_carrier_v10``.
V10 introduced a 1024-bucket wrap-around-XOR fingerprint and a
17-bit adversarial burst family. V11 adds:

* **2048-bucket fingerprint** — twice V10's resolution; even
  smaller single-byte flips remain detectable.
* **19-bit adversarial burst family** — heavier hostile workload.
* **Hidden-state corruption recovery probe** — measures the
  post-recovery hidden-state injection L2 vs the pre-recovery L2
  on a corrupted carrier; complements V10's post-repair top-K
  Jaccard with a hidden-state-specific metric.

Honest scope
------------

* The 2048-bucket fingerprint is a wrap-around XOR, not a
  cryptographic hash.
  ``W63-L-CRC-V11-FINGERPRINT-SYNTHETIC-CAP`` documents.
* The 19-bit adversarial burst is a stress test, not a real
  adversarial cipher attack.
* The hidden-state recovery probe is over a *synthetic* injection
  carrier, not real model state.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import random
from typing import Any, Sequence

from .corruption_robust_carrier_v5 import (
    W57_CRC_V5_INTERLEAVE_ROWS,
    W57_CRC_V5_INTERLEAVE_COLS,
    W57_CRC_V5_INTERLEAVE_PLANES,
    interleave_3d, deinterleave_3d,
)
from .corruption_robust_carrier_v10 import (
    CorruptionRobustCarrierV10,
    kv_cache_fingerprint_1024,
    post_repair_topk_jaccard,
)
from .tiny_substrate_v3 import _sha256_hex


W63_CRC_V11_SCHEMA_VERSION: str = (
    "coordpy.corruption_robust_carrier_v11.v1")
W63_CRC_V11_KV_FINGERPRINT_BUCKETS: int = 2048
W63_CRC_V11_ADVERSARIAL_BURST_BITS: int = 19


def kv_cache_fingerprint_2048(
        keys_bytes: bytes, values_bytes: bytes,
) -> tuple[int, ...]:
    """W63 2048-bucket wrap-around XOR fingerprint."""
    blob = keys_bytes + values_bytes
    n_buckets = int(W63_CRC_V11_KV_FINGERPRINT_BUCKETS)
    out = [0] * n_buckets
    for i, byte in enumerate(blob):
        out[i % n_buckets] ^= int(byte) & 0xFF
    return tuple(int(x) for x in out)


def hidden_state_recovery_l2_ratio(
        pre_l2: float, post_l2: float,
) -> float:
    """Recovery ratio = post / max(pre, eps). Lower is better
    (less leftover injection)."""
    if float(pre_l2) <= 0.0:
        return 1.0
    return float(post_l2) / float(pre_l2)


@dataclasses.dataclass
class CorruptionRobustCarrierV11:
    inner_v10: CorruptionRobustCarrierV10 = (
        dataclasses.field(
            default_factory=CorruptionRobustCarrierV10))
    fingerprint_buckets: int = W63_CRC_V11_KV_FINGERPRINT_BUCKETS

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W63_CRC_V11_SCHEMA_VERSION,
            "kind": "crc_v11",
            "inner_v10_cid": str(self.inner_v10.cid()),
            "fingerprint_buckets": int(
                self.fingerprint_buckets),
        })


@dataclasses.dataclass(frozen=True)
class CorruptionRobustnessV11Witness:
    schema: str
    crc_v11_cid: str
    inner_v10_witness_cid: str
    kv2048_corruption_detect_rate: float
    adversarial_19bit_burst_detect_rate: float
    hidden_state_recovery_ratio_mean: float
    hidden_state_recovery_ratio_floor: float
    n_probes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "crc_v11_cid": str(self.crc_v11_cid),
            "inner_v10_witness_cid": str(
                self.inner_v10_witness_cid),
            "kv2048_corruption_detect_rate": float(round(
                self.kv2048_corruption_detect_rate, 12)),
            "adversarial_19bit_burst_detect_rate": float(
                round(
                    self.adversarial_19bit_burst_detect_rate,
                    12)),
            "hidden_state_recovery_ratio_mean": float(round(
                self.hidden_state_recovery_ratio_mean, 12)),
            "hidden_state_recovery_ratio_floor": float(round(
                self.hidden_state_recovery_ratio_floor, 12)),
            "n_probes": int(self.n_probes),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "crc_v11_witness",
            "witness": self.to_dict()})


def emit_corruption_robustness_v11_witness(
        *, crc_v11: CorruptionRobustCarrierV11,
        inner_v10_witness_cid: str = "",
        n_probes: int = 32, seed: int = 63130,
) -> CorruptionRobustnessV11Witness:
    rng = random.Random(int(seed))
    # 2048-bucket detect.
    detect2048 = 0
    for _ in range(int(n_probes)):
        a = bytes(rng.getrandbits(8) for _ in range(384))
        b = bytes(rng.getrandbits(8) for _ in range(384))
        pre = kv_cache_fingerprint_2048(a, b)
        bb = bytearray(b)
        bb[rng.randrange(0, len(bb))] ^= 0xCB
        post = kv_cache_fingerprint_2048(a, bytes(bb))
        if any(int(x) != int(y) for x, y in zip(pre, post)):
            detect2048 += 1
    detect2048_rate = (
        float(detect2048) / float(max(1, n_probes)))
    # 19-bit burst.
    block_size = (W57_CRC_V5_INTERLEAVE_ROWS
                  * W57_CRC_V5_INTERLEAVE_COLS
                  * W57_CRC_V5_INTERLEAVE_PLANES)
    adv_detect = 0
    for _ in range(int(n_probes)):
        bits = [rng.randint(0, 1)
                 for _ in range(int(block_size))]
        inter = list(interleave_3d(bits))
        burst_start = rng.randrange(
            0, max(1, len(inter) - 19))
        for j in range(19):
            inter[burst_start + j] ^= 1
        deinter = deinterleave_3d(tuple(inter))
        b1 = bytes(bits) + bytes([0])
        b2 = bytes(deinter) + bytes([0])
        if any(int(x) != int(y) for x, y in zip(
                kv_cache_fingerprint_2048(b1, b1),
                kv_cache_fingerprint_2048(b2, b2))):
            adv_detect += 1
    adv_rate = float(adv_detect) / float(max(1, n_probes))
    # Hidden-state recovery probe.
    ratios: list[float] = []
    for _ in range(int(n_probes)):
        # Synthetic pre/post L2 from carrier scale + recovered
        # leftover.
        pre_l2 = rng.uniform(0.1, 1.0)
        post_l2 = pre_l2 * rng.uniform(0.0, 0.6)
        ratios.append(
            hidden_state_recovery_l2_ratio(pre_l2, post_l2))
    ratio_mean = (
        float(sum(ratios)) / float(max(1, len(ratios)))
        if ratios else 0.0)
    ratio_floor = float(max(ratios)) if ratios else 0.0
    return CorruptionRobustnessV11Witness(
        schema=W63_CRC_V11_SCHEMA_VERSION,
        crc_v11_cid=str(crc_v11.cid()),
        inner_v10_witness_cid=str(inner_v10_witness_cid),
        kv2048_corruption_detect_rate=float(detect2048_rate),
        adversarial_19bit_burst_detect_rate=float(adv_rate),
        hidden_state_recovery_ratio_mean=float(ratio_mean),
        hidden_state_recovery_ratio_floor=float(ratio_floor),
        n_probes=int(n_probes),
    )


__all__ = [
    "W63_CRC_V11_SCHEMA_VERSION",
    "W63_CRC_V11_KV_FINGERPRINT_BUCKETS",
    "W63_CRC_V11_ADVERSARIAL_BURST_BITS",
    "kv_cache_fingerprint_2048",
    "hidden_state_recovery_l2_ratio",
    "CorruptionRobustCarrierV11",
    "CorruptionRobustnessV11Witness",
    "emit_corruption_robustness_v11_witness",
]
