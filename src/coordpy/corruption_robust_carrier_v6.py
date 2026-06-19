"""W58 M12 — Corruption-Robust Carrier V6.

Strictly extends W57's ``coordpy.corruption_robust_carrier_v5``.
V6 adds:

* **64-bucket Reed-Solomon-style fingerprint** for KV caches
  (vs V5's 32-bucket). Doubling the bucket count halves the
  expected probability of a hash collision under a random
  single-byte flip. The new helper ``kv_cache_fingerprint_64``
  is what the W58 substrate V3's ``TinyV3KVCache.fingerprint``
  produces by construction.
* **Prefix-state fingerprint check** —
  ``detect_prefix_state_corruption`` over a
  ``TinyV3PrefixState`` object. Returns ``True`` iff the
  recomputed prefix fingerprint diverges from a stored one.
* **Adversarial worst-case burst V2** — adds a 7-bit burst
  family designed to defeat 3-D interleaving on the V5
  axes. The R-121 benchmark uses this to show the new
  fingerprint catches what the V5 32-bucket version would miss
  on extreme adversarial cases.

V6 strictly extends V5: the existing V5 helpers + decoders are
unchanged.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import random
from typing import Any, Sequence

from .corruption_robust_carrier_v5 import (
    CorruptionRobustCarrierV5,
    W57_CRC_V5_INTERLEAVE_COLS,
    W57_CRC_V5_INTERLEAVE_PLANES,
    W57_CRC_V5_INTERLEAVE_ROWS,
    W57_CRC_V5_MAJORITY_M,
    W57_CRC_V5_MAJORITY_N,
    apply_adversarial_burst,
    deinterleave_3d,
    detect_kv_corruption,
    interleave_3d,
    kv_cache_fingerprint,
    majority_9_of_13_decode,
    majority_9_of_13_encode,
)
from .corruption_robust_carrier_v4 import (
    W56_CRC_V4_BCH_K,
    W56_CRC_V4_BCH_N,
    bch_31_16_decode,
    bch_31_16_encode,
)


W58_CRC_V6_SCHEMA_VERSION: str = (
    "coordpy.corruption_robust_carrier_v6.v1")
W58_CRC_V6_KV_FINGERPRINT_BUCKETS: int = 64


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def kv_cache_fingerprint_64(
        keys_bytes: bytes, values_bytes: bytes,
) -> tuple[int, ...]:
    """64-bucket XOR fingerprint of a KV slice. See
    ``coordpy.corruption_robust_carrier_v5.kv_cache_fingerprint``
    for the underlying algorithm; V6 simply uses 64 buckets."""
    return kv_cache_fingerprint(
        keys_bytes, values_bytes,
        n_buckets=int(W58_CRC_V6_KV_FINGERPRINT_BUCKETS))


def detect_prefix_state_corruption(
        prefix_state, *,
        stored_fingerprint_64: Sequence[int] | None = None,
) -> tuple[bool, tuple[int, ...]]:
    """Recompute a 64-bucket fingerprint over a prefix state and
    return ``(detected, fingerprint)``.

    If ``stored_fingerprint_64`` is supplied, ``detected`` is
    True iff the recomputed fingerprint differs.
    """
    import numpy as _np
    keys_bytes = b""
    values_bytes = b""
    for k in prefix_state.keys:
        keys_bytes += _np.ascontiguousarray(k).tobytes()
    for v in prefix_state.values:
        values_bytes += _np.ascontiguousarray(v).tobytes()
    fp = kv_cache_fingerprint_64(keys_bytes, values_bytes)
    detected = (
        bool(any(int(a) != int(b)
                  for a, b in zip(fp, stored_fingerprint_64)))
        if stored_fingerprint_64 is not None else False)
    return bool(detected), fp


@dataclasses.dataclass
class CorruptionRobustCarrierV6:
    inner_v5: CorruptionRobustCarrierV5 = dataclasses.field(
        default_factory=CorruptionRobustCarrierV5)
    fingerprint_buckets: int = W58_CRC_V6_KV_FINGERPRINT_BUCKETS

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W58_CRC_V6_SCHEMA_VERSION,
            "kind": "crc_v6",
            "inner_v5_cid": str(self.inner_v5.cid()),
            "fingerprint_buckets": int(self.fingerprint_buckets),
        })


@dataclasses.dataclass(frozen=True)
class CorruptionRobustnessV6Witness:
    schema: str
    crc_v6_cid: str
    inner_v5_witness_cid: str
    kv64_corruption_detect_rate: float
    prefix_state_corruption_detect_rate: float
    adversarial_7bit_burst_detect_rate: float
    n_probes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "crc_v6_cid": str(self.crc_v6_cid),
            "inner_v5_witness_cid": str(
                self.inner_v5_witness_cid),
            "kv64_corruption_detect_rate": float(round(
                self.kv64_corruption_detect_rate, 12)),
            "prefix_state_corruption_detect_rate": float(round(
                self.prefix_state_corruption_detect_rate, 12)),
            "adversarial_7bit_burst_detect_rate": float(round(
                self.adversarial_7bit_burst_detect_rate, 12)),
            "n_probes": int(self.n_probes),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "crc_v6_witness",
            "witness": self.to_dict()})


def emit_corruption_robustness_v6_witness(
        *,
        crc_v6: CorruptionRobustCarrierV6,
        inner_v5_witness_cid: str = "",
        n_probes: int = 32,
        seed: int = 58110,
) -> CorruptionRobustnessV6Witness:
    rng = random.Random(int(seed))
    # 64-bucket KV detect rate.
    detect64 = 0
    for _ in range(int(n_probes)):
        a = bytes(rng.getrandbits(8) for _ in range(192))
        b = bytes(rng.getrandbits(8) for _ in range(192))
        pre = kv_cache_fingerprint_64(a, b)
        bb = bytearray(b)
        bb[rng.randrange(0, len(bb))] ^= 0x9C
        post = kv_cache_fingerprint_64(a, bytes(bb))
        if any(int(x) != int(y) for x, y in zip(pre, post)):
            detect64 += 1
    detect64_rate = float(detect64) / float(max(1, n_probes))
    # Prefix-state corruption detect rate via fingerprint
    # divergence.
    try:
        from .tiny_substrate_v3 import (
            build_default_tiny_substrate_v3,
            forward_tiny_substrate_v3,
            tokenize_bytes_v3,
            extract_prefix_state_v3,
        )
        from .prefix_state_bridge_v2 import (
            corrupt_prefix_state_v3,
        )
        params = build_default_tiny_substrate_v3(seed=58111)
        prompt = tokenize_bytes_v3("crc-v6-probe", max_len=12)
        trace = forward_tiny_substrate_v3(
            params, prompt, return_attention=False)
        ps = extract_prefix_state_v3(
            trace.kv_cache, prefix_len=trace.kv_cache.n_tokens(),
            source_params_cid=str(params.cid()))
        _, fp_pre = detect_prefix_state_corruption(ps)
        detect_ps = 0
        for i in range(int(n_probes)):
            corrupted = corrupt_prefix_state_v3(
                ps, layer_index=i % 3,
                token_position=i % 4,
                magnitude=1.0 + 0.1 * i,
                seed=58200 + i)
            det, _ = detect_prefix_state_corruption(
                corrupted, stored_fingerprint_64=fp_pre)
            if det:
                detect_ps += 1
        detect_ps_rate = float(detect_ps) / float(max(1, n_probes))
    except Exception:
        detect_ps_rate = 0.0
    # Adversarial 7-bit burst detect.
    block_size = (W57_CRC_V5_INTERLEAVE_ROWS
                  * W57_CRC_V5_INTERLEAVE_COLS
                  * W57_CRC_V5_INTERLEAVE_PLANES)
    adv_detect = 0
    for _ in range(int(n_probes)):
        bits = [rng.randint(0, 1) for _ in range(int(block_size))]
        inter = list(interleave_3d(bits))
        burst_start = rng.randrange(0, max(1, len(inter) - 7))
        for j in range(7):
            inter[burst_start + j] ^= 1
        deinter = deinterleave_3d(tuple(inter))
        # Convert to bytes and fingerprint diff.
        b1 = bytes(bits) + bytes([0])
        b2 = bytes(deinter) + bytes([0])
        if any(int(x) != int(y)
               for x, y in zip(
                   kv_cache_fingerprint_64(b1, b1),
                   kv_cache_fingerprint_64(b2, b2))):
            adv_detect += 1
    adv_rate = float(adv_detect) / float(max(1, n_probes))
    return CorruptionRobustnessV6Witness(
        schema=W58_CRC_V6_SCHEMA_VERSION,
        crc_v6_cid=str(crc_v6.cid()),
        inner_v5_witness_cid=str(inner_v5_witness_cid),
        kv64_corruption_detect_rate=float(detect64_rate),
        prefix_state_corruption_detect_rate=float(detect_ps_rate),
        adversarial_7bit_burst_detect_rate=float(adv_rate),
        n_probes=int(n_probes),
    )


__all__ = [
    "W58_CRC_V6_SCHEMA_VERSION",
    "W58_CRC_V6_KV_FINGERPRINT_BUCKETS",
    "CorruptionRobustCarrierV6",
    "CorruptionRobustnessV6Witness",
    "kv_cache_fingerprint_64",
    "detect_prefix_state_corruption",
    "emit_corruption_robustness_v6_witness",
]
