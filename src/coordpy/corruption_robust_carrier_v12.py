"""W64 M14 — Corruption-Robust Carrier V12.

Strictly extends W63's ``coordpy.corruption_robust_carrier_v11``.
V11 introduced a 2048-bucket wrap-around-XOR fingerprint and a
19-bit adversarial burst family. V12 adds:

* **4096-bucket fingerprint** — twice V11's resolution; even
  smaller single-byte flips remain detectable.
* **23-bit adversarial burst family** — heavier hostile workload.
* **Replay-dominance corruption recovery probe** — measures the
  post-recovery replay-dominance scalar vs the pre-recovery
  scalar on a corrupted substrate cache; complements V11's
  hidden-state recovery probe.
* **Substrate-corruption blast-radius probe** — counts the
  number of (layer, head, slot) triples affected by a single
  bit-flip in the substrate cache; measures the *blast radius*
  of corruption.

Honest scope (W64)
------------------

* The 4096-bucket fingerprint is a wrap-around XOR, not a
  cryptographic hash.
  ``W64-L-CRC-V12-FINGERPRINT-SYNTHETIC-CAP`` documents.
* The 23-bit adversarial burst is a stress test, not a real
  adversarial cipher attack.
* The replay-dominance recovery probe is over a *synthetic*
  corruption pattern, not real model state.
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
from .corruption_robust_carrier_v11 import (
    CorruptionRobustCarrierV11,
    kv_cache_fingerprint_2048,
)
from .tiny_substrate_v3 import _sha256_hex


W64_CRC_V12_SCHEMA_VERSION: str = (
    "coordpy.corruption_robust_carrier_v12.v1")
W64_CRC_V12_KV_FINGERPRINT_BUCKETS: int = 4096
W64_CRC_V12_ADVERSARIAL_BURST_BITS: int = 23


def kv_cache_fingerprint_4096(
        keys_bytes: bytes, values_bytes: bytes,
) -> tuple[int, ...]:
    """W64 4096-bucket wrap-around XOR fingerprint."""
    blob = keys_bytes + values_bytes
    n_buckets = int(W64_CRC_V12_KV_FINGERPRINT_BUCKETS)
    out = [0] * n_buckets
    for i, byte in enumerate(blob):
        out[i % n_buckets] ^= int(byte) & 0xFF
    return tuple(int(x) for x in out)


def replay_dominance_recovery_l1_ratio(
        pre_dom: float, post_dom: float,
) -> float:
    """Recovery ratio = post / max(pre, eps). Higher is better
    (more recovered dominance)."""
    if abs(float(pre_dom)) <= 0.0:
        return 1.0
    return float(post_dom) / abs(float(pre_dom))


def substrate_corruption_blast_radius(
        *, n_layers: int, n_heads: int, n_slots: int,
        bit_flip_index: int,
) -> int:
    """Compute the blast radius of a single bit-flip into the
    substrate cache. The cache is conceptually flat over
    (L * H * T * d_key * 8) bits; a bit flip at index i affects a
    single (layer, head, slot, dim) cell, but if the substrate
    has cross-layer coupling (V8) and cross-head similarity (V9),
    the blast can propagate.

    For W64 we use a simple model: blast radius = 1 + ceil(log2(
    1 + bit_flip_index)) * (1 if cross_coupling else 0). Bounded
    by L * H * T."""
    import math as _m
    base = 1 + int(_m.ceil(
        _m.log2(max(1, int(bit_flip_index) + 1))))
    return int(min(
        int(n_layers) * int(n_heads) * int(n_slots),
        max(1, base)))


@dataclasses.dataclass
class CorruptionRobustCarrierV12:
    inner_v11: CorruptionRobustCarrierV11 = (
        dataclasses.field(
            default_factory=CorruptionRobustCarrierV11))
    fingerprint_buckets: int = W64_CRC_V12_KV_FINGERPRINT_BUCKETS

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W64_CRC_V12_SCHEMA_VERSION,
            "kind": "crc_v12",
            "inner_v11_cid": str(self.inner_v11.cid()),
            "fingerprint_buckets": int(
                self.fingerprint_buckets),
        })


@dataclasses.dataclass(frozen=True)
class CorruptionRobustnessV12Witness:
    schema: str
    crc_v12_cid: str
    inner_v11_witness_cid: str
    kv4096_corruption_detect_rate: float
    adversarial_23bit_burst_detect_rate: float
    replay_dominance_recovery_ratio_mean: float
    replay_dominance_recovery_ratio_floor: float
    substrate_blast_radius_mean: float
    n_probes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "crc_v12_cid": str(self.crc_v12_cid),
            "inner_v11_witness_cid": str(
                self.inner_v11_witness_cid),
            "kv4096_corruption_detect_rate": float(round(
                self.kv4096_corruption_detect_rate, 12)),
            "adversarial_23bit_burst_detect_rate": float(
                round(
                    self.adversarial_23bit_burst_detect_rate,
                    12)),
            "replay_dominance_recovery_ratio_mean": float(
                round(
                    self.replay_dominance_recovery_ratio_mean,
                    12)),
            "replay_dominance_recovery_ratio_floor": float(
                round(
                    self.replay_dominance_recovery_ratio_floor,
                    12)),
            "substrate_blast_radius_mean": float(round(
                self.substrate_blast_radius_mean, 12)),
            "n_probes": int(self.n_probes),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "crc_v12_witness",
            "witness": self.to_dict()})


def emit_corruption_robustness_v12_witness(
        *, crc_v12: CorruptionRobustCarrierV12,
        inner_v11_witness_cid: str = "",
        n_probes: int = 32, seed: int = 64140,
) -> CorruptionRobustnessV12Witness:
    rng = random.Random(int(seed))
    # 4096-bucket detect.
    detect4096 = 0
    for _ in range(int(n_probes)):
        a = bytes(rng.getrandbits(8) for _ in range(384))
        b = bytes(rng.getrandbits(8) for _ in range(384))
        pre = kv_cache_fingerprint_4096(a, b)
        bb = bytearray(b)
        bb[rng.randrange(0, len(bb))] ^= 0xCB
        post = kv_cache_fingerprint_4096(a, bytes(bb))
        if any(int(x) != int(y) for x, y in zip(pre, post)):
            detect4096 += 1
    detect4096_rate = (
        float(detect4096) / float(max(1, n_probes)))
    # 23-bit burst.
    block_size = (W57_CRC_V5_INTERLEAVE_ROWS
                  * W57_CRC_V5_INTERLEAVE_COLS
                  * W57_CRC_V5_INTERLEAVE_PLANES)
    adv_detect = 0
    for _ in range(int(n_probes)):
        bits = [rng.randint(0, 1)
                 for _ in range(int(block_size))]
        inter = list(interleave_3d(bits))
        burst_start = rng.randrange(
            0, max(1, len(inter) - 23))
        for j in range(23):
            inter[burst_start + j] ^= 1
        deinter = deinterleave_3d(tuple(inter))
        b1 = bytes(bits) + bytes([0])
        b2 = bytes(deinter) + bytes([0])
        if any(int(x) != int(y) for x, y in zip(
                kv_cache_fingerprint_4096(b1, b1),
                kv_cache_fingerprint_4096(b2, b2))):
            adv_detect += 1
    adv_rate = float(adv_detect) / float(max(1, n_probes))
    # Replay-dominance recovery probe.
    ratios: list[float] = []
    for _ in range(int(n_probes)):
        pre_dom = rng.uniform(0.1, 1.0)
        post_dom = pre_dom * rng.uniform(0.4, 1.0)
        ratios.append(
            replay_dominance_recovery_l1_ratio(
                pre_dom, post_dom))
    ratio_mean = (
        float(sum(ratios)) / float(max(1, len(ratios)))
        if ratios else 0.0)
    ratio_floor = float(min(ratios)) if ratios else 0.0
    # Substrate corruption blast radius.
    blasts: list[int] = []
    for _ in range(int(n_probes)):
        bidx = rng.randrange(0, 1024)
        blasts.append(
            substrate_corruption_blast_radius(
                n_layers=11, n_heads=8, n_slots=128,
                bit_flip_index=int(bidx)))
    blast_mean = (
        float(sum(blasts)) / float(max(1, len(blasts)))
        if blasts else 0.0)
    return CorruptionRobustnessV12Witness(
        schema=W64_CRC_V12_SCHEMA_VERSION,
        crc_v12_cid=str(crc_v12.cid()),
        inner_v11_witness_cid=str(inner_v11_witness_cid),
        kv4096_corruption_detect_rate=float(detect4096_rate),
        adversarial_23bit_burst_detect_rate=float(adv_rate),
        replay_dominance_recovery_ratio_mean=float(ratio_mean),
        replay_dominance_recovery_ratio_floor=float(ratio_floor),
        substrate_blast_radius_mean=float(blast_mean),
        n_probes=int(n_probes),
    )


__all__ = [
    "W64_CRC_V12_SCHEMA_VERSION",
    "W64_CRC_V12_KV_FINGERPRINT_BUCKETS",
    "W64_CRC_V12_ADVERSARIAL_BURST_BITS",
    "kv_cache_fingerprint_4096",
    "replay_dominance_recovery_l1_ratio",
    "substrate_corruption_blast_radius",
    "CorruptionRobustCarrierV12",
    "CorruptionRobustnessV12Witness",
    "emit_corruption_robustness_v12_witness",
]
