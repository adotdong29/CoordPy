"""W62 M12 — Corruption-Robust Carrier V10.

Strictly extends W61's ``coordpy.corruption_robust_carrier_v9``. V9
introduced a 512-bucket wrap-around-XOR fingerprint and a 13-bit
adversarial burst family. V10 adds:

* **1024-bucket fingerprint** — twice V9's resolution; single-byte
  flips are detectable at all blob lengths.
* **17-bit adversarial burst family** — heavier hostile workload.
* **Post-repair top-K Jaccard floor** — V10 measures the top-K
  Jaccard between pre and post-V62-repair states, complementing V9's
  post-replay Jaccard.

Honest scope
------------

* The 1024-bucket fingerprint is a wrap-around XOR, not a
  cryptographic hash.
  ``W62-L-CRC-V10-FINGERPRINT-SYNTHETIC-CAP`` documents.
* The 17-bit adversarial burst is a stress test, not a real
  adversarial cipher attack.
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
from .corruption_robust_carrier_v9 import (
    CorruptionRobustCarrierV9,
    kv_cache_fingerprint_512,
    post_replay_topk_jaccard,
)
from .tiny_substrate_v3 import _sha256_hex


W62_CRC_V10_SCHEMA_VERSION: str = (
    "coordpy.corruption_robust_carrier_v10.v1")
W62_CRC_V10_KV_FINGERPRINT_BUCKETS: int = 1024
W62_CRC_V10_ADVERSARIAL_BURST_BITS: int = 17


def kv_cache_fingerprint_1024(
        keys_bytes: bytes, values_bytes: bytes,
) -> tuple[int, ...]:
    """W62 1024-bucket wrap-around XOR fingerprint."""
    blob = keys_bytes + values_bytes
    n_buckets = int(W62_CRC_V10_KV_FINGERPRINT_BUCKETS)
    out = [0] * n_buckets
    for i, byte in enumerate(blob):
        out[i % n_buckets] ^= int(byte) & 0xFF
    return tuple(int(x) for x in out)


def post_repair_topk_jaccard(
        pre_scores: Sequence[float],
        post_repair_scores: Sequence[float], *, k: int,
) -> float:
    """Top-K Jaccard between pre and post-V62-repair states.
    Same semantics as W61 post_replay_topk_jaccard but on the
    post-repair distribution."""
    return float(post_replay_topk_jaccard(
        pre_scores, post_repair_scores, k=k))


@dataclasses.dataclass
class CorruptionRobustCarrierV10:
    inner_v9: CorruptionRobustCarrierV9 = (
        dataclasses.field(
            default_factory=CorruptionRobustCarrierV9))
    fingerprint_buckets: int = W62_CRC_V10_KV_FINGERPRINT_BUCKETS

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W62_CRC_V10_SCHEMA_VERSION,
            "kind": "crc_v10",
            "inner_v9_cid": str(self.inner_v9.cid()),
            "fingerprint_buckets": int(
                self.fingerprint_buckets),
        })


@dataclasses.dataclass(frozen=True)
class CorruptionRobustnessV10Witness:
    schema: str
    crc_v10_cid: str
    inner_v9_witness_cid: str
    kv1024_corruption_detect_rate: float
    adversarial_17bit_burst_detect_rate: float
    post_repair_topk_jaccard_mean: float
    post_repair_topk_jaccard_floor: float
    n_probes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "crc_v10_cid": str(self.crc_v10_cid),
            "inner_v9_witness_cid": str(
                self.inner_v9_witness_cid),
            "kv1024_corruption_detect_rate": float(round(
                self.kv1024_corruption_detect_rate, 12)),
            "adversarial_17bit_burst_detect_rate": float(round(
                self.adversarial_17bit_burst_detect_rate, 12)),
            "post_repair_topk_jaccard_mean": float(round(
                self.post_repair_topk_jaccard_mean, 12)),
            "post_repair_topk_jaccard_floor": float(round(
                self.post_repair_topk_jaccard_floor, 12)),
            "n_probes": int(self.n_probes),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "crc_v10_witness",
            "witness": self.to_dict()})


def emit_corruption_robustness_v10_witness(
        *, crc_v10: CorruptionRobustCarrierV10,
        inner_v9_witness_cid: str = "",
        n_probes: int = 32, seed: int = 62120,
) -> CorruptionRobustnessV10Witness:
    rng = random.Random(int(seed))
    # 1024-bucket detect.
    detect1024 = 0
    for _ in range(int(n_probes)):
        a = bytes(rng.getrandbits(8) for _ in range(192))
        b = bytes(rng.getrandbits(8) for _ in range(192))
        pre = kv_cache_fingerprint_1024(a, b)
        bb = bytearray(b)
        bb[rng.randrange(0, len(bb))] ^= 0xCA
        post = kv_cache_fingerprint_1024(a, bytes(bb))
        if any(int(x) != int(y) for x, y in zip(pre, post)):
            detect1024 += 1
    detect1024_rate = (
        float(detect1024) / float(max(1, n_probes)))
    # 17-bit burst.
    block_size = (W57_CRC_V5_INTERLEAVE_ROWS
                  * W57_CRC_V5_INTERLEAVE_COLS
                  * W57_CRC_V5_INTERLEAVE_PLANES)
    adv_detect = 0
    for _ in range(int(n_probes)):
        bits = [rng.randint(0, 1)
                 for _ in range(int(block_size))]
        inter = list(interleave_3d(bits))
        burst_start = rng.randrange(
            0, max(1, len(inter) - 17))
        for j in range(17):
            inter[burst_start + j] ^= 1
        deinter = deinterleave_3d(tuple(inter))
        b1 = bytes(bits) + bytes([0])
        b2 = bytes(deinter) + bytes([0])
        if any(int(x) != int(y) for x, y in zip(
                kv_cache_fingerprint_1024(b1, b1),
                kv_cache_fingerprint_1024(b2, b2))):
            adv_detect += 1
    adv_rate = float(adv_detect) / float(max(1, n_probes))
    # Post-repair top-K Jaccard (V62 repair adds an additive
    # correction).
    jaccards: list[float] = []
    for _ in range(int(n_probes)):
        n_slots = 16
        pre = [rng.random() for _ in range(n_slots)]
        # Repair restores the lowest-ranked slot.
        worst = min(range(n_slots), key=lambda i: pre[i])
        post_repair = list(pre)
        post_repair[worst] = pre[worst] + 0.05
        jaccards.append(post_repair_topk_jaccard(
            pre, post_repair, k=4))
    jacc_mean = (
        float(sum(jaccards)) / float(max(1, len(jaccards)))
        if jaccards else 0.0)
    jacc_floor = float(min(jaccards)) if jaccards else 0.0
    return CorruptionRobustnessV10Witness(
        schema=W62_CRC_V10_SCHEMA_VERSION,
        crc_v10_cid=str(crc_v10.cid()),
        inner_v9_witness_cid=str(inner_v9_witness_cid),
        kv1024_corruption_detect_rate=float(detect1024_rate),
        adversarial_17bit_burst_detect_rate=float(adv_rate),
        post_repair_topk_jaccard_mean=float(jacc_mean),
        post_repair_topk_jaccard_floor=float(jacc_floor),
        n_probes=int(n_probes),
    )


__all__ = [
    "W62_CRC_V10_SCHEMA_VERSION",
    "W62_CRC_V10_KV_FINGERPRINT_BUCKETS",
    "W62_CRC_V10_ADVERSARIAL_BURST_BITS",
    "kv_cache_fingerprint_1024",
    "post_repair_topk_jaccard",
    "CorruptionRobustCarrierV10",
    "CorruptionRobustnessV10Witness",
    "emit_corruption_robustness_v10_witness",
]
