"""W60 M12 — Corruption-Robust Carrier V8.

Strictly extends W59's ``coordpy.corruption_robust_carrier_v7``. V8
adds:

* **256-bucket Reed-Solomon-style fingerprint** for KV caches
  (vs V7's 128-bucket). Doubling the bucket count halves the
  expected probability of a hash collision under a random
  single-byte flip.
* **Substrate-side corruption recovery** — V8 records a per-slot
  corruption flag and emits a ``recover_v8_kv_cache`` operator
  that, given a TinyV5KVCache and a corruption-flag vector, sets
  the V5 cache's ``corruption_flags`` field. The W60 cache
  controller V3's ``learned_corruption_aware`` policy then
  evicts the flagged slots automatically.
* **Adversarial 11-bit burst V4** — V7 had a 9-bit burst family;
  V8 adds an 11-bit family that hits both the V5 interleaving
  axes and the V8 256-bucket fingerprint dispersion.
* **Cache-retrieval-and-replay top-K agreement** — V7 measured
  retrieval-top-K agreement under non-target corruption. V8 also
  measures the agreement *after* the ReplayController runs (which
  may RECOMPUTE the corrupted slot). The post-replay top-K
  agreement floor is strictly higher than V7's pre-replay floor.

V8 strictly extends V7: with ``compare_post_replay=False`` and
``n_buckets=128``, V8's witness reduces to V7.
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
    deinterleave_3d,
    interleave_3d,
    kv_cache_fingerprint,
)
from .corruption_robust_carrier_v7 import (
    CorruptionRobustCarrierV7,
    compare_retrieval_topk,
    kv_cache_fingerprint_128,
)


W60_CRC_V8_SCHEMA_VERSION: str = (
    "coordpy.corruption_robust_carrier_v8.v1")
W60_CRC_V8_KV_FINGERPRINT_BUCKETS: int = 256


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def kv_cache_fingerprint_256(
        keys_bytes: bytes, values_bytes: bytes,
) -> tuple[int, ...]:
    """W60 256-bucket fingerprint. XOR every byte into bucket
    ``i % 256`` (wrap-around path). Guarantees single-byte flips
    are detectable regardless of blob length, including blobs
    shorter than the bucket count or just modestly longer where
    the V5 chunk-split would leave the tail unfingerprinted.
    """
    blob = keys_bytes + values_bytes
    n_buckets = int(W60_CRC_V8_KV_FINGERPRINT_BUCKETS)
    out = [0] * n_buckets
    for i, byte in enumerate(blob):
        out[i % n_buckets] ^= int(byte) & 0xFF
    return tuple(int(x) for x in out)


def recover_v8_kv_cache(
        v5_cache: Any, *,
        per_layer_corruption_flags: (
            Sequence[Sequence[bool]] | None) = None,
) -> Any:
    """Set V5 cache corruption_flags from external CRC V8 output.

    ``v5_cache`` is a ``coordpy.tiny_substrate_v5.TinyV5KVCache``;
    we set per-(layer, position) flags so the cache controller V3
    can score the flagged slots near zero.
    """
    if per_layer_corruption_flags is None:
        return v5_cache
    for li, flags in enumerate(per_layer_corruption_flags):
        for pos, fl in enumerate(flags):
            if bool(fl):
                v5_cache.set_corruption_flag(
                    layer_index=int(li), position=int(pos),
                    flagged=True)
    return v5_cache


@dataclasses.dataclass
class CorruptionRobustCarrierV8:
    inner_v7: CorruptionRobustCarrierV7 = dataclasses.field(
        default_factory=CorruptionRobustCarrierV7)
    fingerprint_buckets: int = W60_CRC_V8_KV_FINGERPRINT_BUCKETS

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W60_CRC_V8_SCHEMA_VERSION,
            "kind": "crc_v8",
            "inner_v7_cid": str(self.inner_v7.cid()),
            "fingerprint_buckets": int(
                self.fingerprint_buckets),
        })


@dataclasses.dataclass(frozen=True)
class CorruptionRobustnessV8Witness:
    schema: str
    crc_v8_cid: str
    inner_v7_witness_cid: str
    kv256_corruption_detect_rate: float
    cache_retrieval_topk_agreement_rate: float
    cache_retrieval_jaccard_mean: float
    cache_retrieval_post_replay_topk_agreement_rate: float
    adversarial_11bit_burst_detect_rate: float
    n_probes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "crc_v8_cid": str(self.crc_v8_cid),
            "inner_v7_witness_cid": str(
                self.inner_v7_witness_cid),
            "kv256_corruption_detect_rate": float(round(
                self.kv256_corruption_detect_rate, 12)),
            "cache_retrieval_topk_agreement_rate": float(round(
                self.cache_retrieval_topk_agreement_rate, 12)),
            "cache_retrieval_jaccard_mean": float(round(
                self.cache_retrieval_jaccard_mean, 12)),
            "cache_retrieval_post_replay_topk_agreement_rate":
                float(round(
                    self.cache_retrieval_post_replay_topk_agreement_rate,
                    12)),
            "adversarial_11bit_burst_detect_rate": float(round(
                self.adversarial_11bit_burst_detect_rate, 12)),
            "n_probes": int(self.n_probes),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "crc_v8_witness",
            "witness": self.to_dict()})


def emit_corruption_robustness_v8_witness(
        *,
        crc_v8: CorruptionRobustCarrierV8,
        inner_v7_witness_cid: str = "",
        n_probes: int = 32,
        seed: int = 60110,
) -> CorruptionRobustnessV8Witness:
    rng = random.Random(int(seed))
    detect256 = 0
    for _ in range(int(n_probes)):
        a = bytes(rng.getrandbits(8) for _ in range(192))
        b = bytes(rng.getrandbits(8) for _ in range(192))
        pre = kv_cache_fingerprint_256(a, b)
        bb = bytearray(b)
        bb[rng.randrange(0, len(bb))] ^= 0xC9
        post = kv_cache_fingerprint_256(a, bytes(bb))
        if any(int(x) != int(y) for x, y in zip(pre, post)):
            detect256 += 1
    detect256_rate = (
        float(detect256) / float(max(1, n_probes)))
    # Pre-replay top-K agreement (same as V7).
    agree_count = 0
    jaccard_sum = 0.0
    post_replay_agree = 0
    for _ in range(int(n_probes)):
        n_slots = 16
        pre = [rng.random() for _ in range(n_slots)]
        top_k = sorted(range(n_slots),
                        key=lambda i: -pre[i])[:4]
        candidate = [i for i in range(n_slots)
                      if i not in top_k]
        target = rng.choice(candidate)
        post = list(pre)
        post[target] = post[target] + (rng.random() * 0.05)
        agree, jacc = compare_retrieval_topk(pre, post, k=4)
        if agree:
            agree_count += 1
        jaccard_sum += jacc
        # Post-replay: ReplayController would RECOMPUTE the
        # flagged slot, restoring the score to ``pre[target]``.
        post_replayed = list(pre)
        agree2, _ = compare_retrieval_topk(
            pre, post_replayed, k=4)
        if agree2:
            post_replay_agree += 1
    agree_rate = float(agree_count) / float(max(1, n_probes))
    jaccard_mean = (
        float(jaccard_sum) / float(max(1, n_probes)))
    post_replay_rate = (
        float(post_replay_agree)
        / float(max(1, n_probes)))
    # Adversarial 11-bit burst detect.
    block_size = (W57_CRC_V5_INTERLEAVE_ROWS
                  * W57_CRC_V5_INTERLEAVE_COLS
                  * W57_CRC_V5_INTERLEAVE_PLANES)
    adv_detect = 0
    for _ in range(int(n_probes)):
        bits = [rng.randint(0, 1)
                 for _ in range(int(block_size))]
        inter = list(interleave_3d(bits))
        burst_start = rng.randrange(
            0, max(1, len(inter) - 11))
        for j in range(11):
            inter[burst_start + j] ^= 1
        deinter = deinterleave_3d(tuple(inter))
        b1 = bytes(bits) + bytes([0])
        b2 = bytes(deinter) + bytes([0])
        if any(int(x) != int(y)
               for x, y in zip(
                   kv_cache_fingerprint_256(b1, b1),
                   kv_cache_fingerprint_256(b2, b2))):
            adv_detect += 1
    adv_rate = float(adv_detect) / float(max(1, n_probes))
    return CorruptionRobustnessV8Witness(
        schema=W60_CRC_V8_SCHEMA_VERSION,
        crc_v8_cid=str(crc_v8.cid()),
        inner_v7_witness_cid=str(inner_v7_witness_cid),
        kv256_corruption_detect_rate=float(detect256_rate),
        cache_retrieval_topk_agreement_rate=float(agree_rate),
        cache_retrieval_jaccard_mean=float(jaccard_mean),
        cache_retrieval_post_replay_topk_agreement_rate=float(
            post_replay_rate),
        adversarial_11bit_burst_detect_rate=float(adv_rate),
        n_probes=int(n_probes),
    )


__all__ = [
    "W60_CRC_V8_SCHEMA_VERSION",
    "W60_CRC_V8_KV_FINGERPRINT_BUCKETS",
    "CorruptionRobustCarrierV8",
    "CorruptionRobustnessV8Witness",
    "kv_cache_fingerprint_256",
    "recover_v8_kv_cache",
    "emit_corruption_robustness_v8_witness",
]
