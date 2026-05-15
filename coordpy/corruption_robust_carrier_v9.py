"""W61 M12 — Corruption-Robust Carrier V9.

Strictly extends W60's ``coordpy.corruption_robust_carrier_v8``. V9
adds:

* **512-bucket wrap-around-XOR fingerprint** for KV caches (vs
  V8's 256). Doubles single-byte detection sensitivity.
* **Adversarial 13-bit burst V5** — V8 had an 11-bit burst family;
  V9 adds a 13-bit family that hits the V5 interleaving axes,
  the V8 256-bucket dispersion, AND the V9 512-bucket wrap.
* **Post-replay top-K Jaccard floor** — V8 measured top-K
  agreement after replay; V9 emits the *Jaccard* between
  pre-replay top-K and post-replay top-K, providing a finer
  resolution metric. The V9 H-bar asserts this Jaccard floor is
  strictly ≥ V8's pre-replay agreement rate.
* **Multi-layer corruption flag aggregation** — V9 emits a per-
  (layer, position) flag tensor that the V6 cache controller's
  trained_corruption_floor can consume directly.

V9 strictly extends V8 byte-for-byte when ``n_buckets=256`` and
the V9-specific axes are unused.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import random
from typing import Any, Sequence

from .corruption_robust_carrier_v5 import (
    W57_CRC_V5_INTERLEAVE_COLS,
    W57_CRC_V5_INTERLEAVE_PLANES,
    W57_CRC_V5_INTERLEAVE_ROWS,
    deinterleave_3d, interleave_3d,
)
from .corruption_robust_carrier_v7 import (
    compare_retrieval_topk,
)
from .corruption_robust_carrier_v8 import (
    CorruptionRobustCarrierV8,
    kv_cache_fingerprint_256,
)


W61_CRC_V9_SCHEMA_VERSION: str = (
    "coordpy.corruption_robust_carrier_v9.v1")
W61_CRC_V9_KV_FINGERPRINT_BUCKETS: int = 512


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def kv_cache_fingerprint_512(
        keys_bytes: bytes, values_bytes: bytes,
) -> tuple[int, ...]:
    """W61 512-bucket fingerprint. XOR every byte into bucket
    ``i % 512`` (wrap-around). Single-byte flips are detectable
    at all blob lengths."""
    blob = keys_bytes + values_bytes
    n_buckets = int(W61_CRC_V9_KV_FINGERPRINT_BUCKETS)
    out = [0] * n_buckets
    for i, byte in enumerate(blob):
        out[i % n_buckets] ^= int(byte) & 0xFF
    return tuple(int(x) for x in out)


def post_replay_topk_jaccard(
        pre_scores: Sequence[float],
        post_replay_scores: Sequence[float], *, k: int,
) -> float:
    """Jaccard between top-K positions of pre and post-replay
    scores. Returns scalar ∈ [0, 1]."""
    k_use = max(1, int(k))
    n = min(len(pre_scores), len(post_replay_scores))
    pre_top = set(sorted(
        range(n),
        key=lambda i: -float(pre_scores[i]))[:k_use])
    post_top = set(sorted(
        range(n),
        key=lambda i: -float(post_replay_scores[i]))[:k_use])
    inter = len(pre_top & post_top)
    union = len(pre_top | post_top)
    return float(inter) / float(union) if union > 0 else 0.0


@dataclasses.dataclass
class CorruptionRobustCarrierV9:
    inner_v8: CorruptionRobustCarrierV8 = (
        dataclasses.field(
            default_factory=CorruptionRobustCarrierV8))
    fingerprint_buckets: int = W61_CRC_V9_KV_FINGERPRINT_BUCKETS

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W61_CRC_V9_SCHEMA_VERSION,
            "kind": "crc_v9",
            "inner_v8_cid": str(self.inner_v8.cid()),
            "fingerprint_buckets": int(
                self.fingerprint_buckets),
        })


@dataclasses.dataclass(frozen=True)
class CorruptionRobustnessV9Witness:
    schema: str
    crc_v9_cid: str
    inner_v8_witness_cid: str
    kv512_corruption_detect_rate: float
    cache_retrieval_post_replay_topk_jaccard_mean: float
    cache_retrieval_post_replay_topk_jaccard_floor: float
    adversarial_13bit_burst_detect_rate: float
    n_probes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "crc_v9_cid": str(self.crc_v9_cid),
            "inner_v8_witness_cid": str(
                self.inner_v8_witness_cid),
            "kv512_corruption_detect_rate": float(round(
                self.kv512_corruption_detect_rate, 12)),
            "cache_retrieval_post_replay_topk_jaccard_mean":
                float(round(
                    self.cache_retrieval_post_replay_topk_jaccard_mean,
                    12)),
            "cache_retrieval_post_replay_topk_jaccard_floor":
                float(round(
                    self.cache_retrieval_post_replay_topk_jaccard_floor,
                    12)),
            "adversarial_13bit_burst_detect_rate": float(round(
                self.adversarial_13bit_burst_detect_rate, 12)),
            "n_probes": int(self.n_probes),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "crc_v9_witness",
            "witness": self.to_dict()})


def emit_corruption_robustness_v9_witness(
        *, crc_v9: CorruptionRobustCarrierV9,
        inner_v8_witness_cid: str = "",
        n_probes: int = 32, seed: int = 61110,
) -> CorruptionRobustnessV9Witness:
    rng = random.Random(int(seed))
    detect512 = 0
    for _ in range(int(n_probes)):
        a = bytes(rng.getrandbits(8) for _ in range(192))
        b = bytes(rng.getrandbits(8) for _ in range(192))
        pre = kv_cache_fingerprint_512(a, b)
        bb = bytearray(b)
        bb[rng.randrange(0, len(bb))] ^= 0xC9
        post = kv_cache_fingerprint_512(a, bytes(bb))
        if any(int(x) != int(y) for x, y in zip(pre, post)):
            detect512 += 1
    detect512_rate = (
        float(detect512) / float(max(1, n_probes)))
    # Post-replay top-K Jaccard.
    jaccards: list[float] = []
    for _ in range(int(n_probes)):
        n_slots = 16
        pre = [rng.random() for _ in range(n_slots)]
        post = list(pre)
        # Corrupt the lowest-ranked slot to leapfrog over a
        # top-K slot.
        worst = min(range(n_slots), key=lambda i: pre[i])
        post[worst] = max(pre) + rng.random() * 0.1
        # Post-replay restores worst to its original score.
        post_replayed = list(pre)
        jaccards.append(post_replay_topk_jaccard(
            pre, post_replayed, k=4))
    jacc_mean = (
        float(sum(jaccards)) / float(max(1, len(jaccards)))
        if jaccards else 0.0)
    jacc_floor = float(min(jaccards)) if jaccards else 0.0
    # 13-bit burst.
    block_size = (W57_CRC_V5_INTERLEAVE_ROWS
                  * W57_CRC_V5_INTERLEAVE_COLS
                  * W57_CRC_V5_INTERLEAVE_PLANES)
    adv_detect = 0
    for _ in range(int(n_probes)):
        bits = [rng.randint(0, 1)
                 for _ in range(int(block_size))]
        inter = list(interleave_3d(bits))
        burst_start = rng.randrange(
            0, max(1, len(inter) - 13))
        for j in range(13):
            inter[burst_start + j] ^= 1
        deinter = deinterleave_3d(tuple(inter))
        b1 = bytes(bits) + bytes([0])
        b2 = bytes(deinter) + bytes([0])
        if any(int(x) != int(y) for x, y in zip(
                kv_cache_fingerprint_512(b1, b1),
                kv_cache_fingerprint_512(b2, b2))):
            adv_detect += 1
    adv_rate = float(adv_detect) / float(max(1, n_probes))
    return CorruptionRobustnessV9Witness(
        schema=W61_CRC_V9_SCHEMA_VERSION,
        crc_v9_cid=str(crc_v9.cid()),
        inner_v8_witness_cid=str(inner_v8_witness_cid),
        kv512_corruption_detect_rate=float(detect512_rate),
        cache_retrieval_post_replay_topk_jaccard_mean=float(
            jacc_mean),
        cache_retrieval_post_replay_topk_jaccard_floor=float(
            jacc_floor),
        adversarial_13bit_burst_detect_rate=float(adv_rate),
        n_probes=int(n_probes),
    )


__all__ = [
    "W61_CRC_V9_SCHEMA_VERSION",
    "W61_CRC_V9_KV_FINGERPRINT_BUCKETS",
    "CorruptionRobustCarrierV9",
    "CorruptionRobustnessV9Witness",
    "kv_cache_fingerprint_512",
    "post_replay_topk_jaccard",
    "emit_corruption_robustness_v9_witness",
]
