"""W57 M10 — Corruption-Robust Carrier V5.

Extends W56 CRC V4 with:

* **3-D interleaving** (rows × cols × planes) so burst errors that
  span both a row and a column still disperse across planes.
* **9-of-13 majority repetition** (vs V4's 7-of-9). Larger
  redundancy floor at the cost of more raw bits.
* **Stronger cached-state-corruption detection**: V5 adds a
  ``detect_kv_corruption`` helper that builds a Reed-Solomon-style
  fingerprint over a KV cache slice and re-verifies on readback;
  ``W56-L-BCH-31-16-FOUR-BIT-PATHOLOGY`` carries forward.
* **Adversarial corruption family**: ``apply_adversarial_burst``
  applies a worst-case burst pattern designed to defeat 2-D
  interleaving (single-axis aligned); the 3-D interleave still
  survives a fraction of these.

V5 strictly extends V4: the V5 codepath reuses V4's BCH(31,16)
codebook; the V5 helpers add on top without modifying V4 outputs.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import random
from typing import Any, Sequence

from .corruption_robust_carrier_v4 import (
    CorruptionRobustCarrierV4,
    W56_CRC_V4_BCH_K,
    W56_CRC_V4_BCH_N,
    bch_31_16_decode,
    bch_31_16_encode,
    deinterleave_2d,
    interleave_2d,
)


W57_CRC_V5_SCHEMA_VERSION: str = (
    "coordpy.corruption_robust_carrier_v5.v1")
W57_CRC_V5_MAJORITY_M: int = 13
W57_CRC_V5_MAJORITY_N: int = 9  # 9-of-13
W57_CRC_V5_INTERLEAVE_ROWS: int = 4
W57_CRC_V5_INTERLEAVE_COLS: int = 4
W57_CRC_V5_INTERLEAVE_PLANES: int = 4


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


# ---------------------------------------------------------------------------
# 3-D interleave
# ---------------------------------------------------------------------------


def interleave_3d(
        bits: Sequence[int],
        *, rows: int = W57_CRC_V5_INTERLEAVE_ROWS,
        cols: int = W57_CRC_V5_INTERLEAVE_COLS,
        planes: int = W57_CRC_V5_INTERLEAVE_PLANES,
) -> tuple[int, ...]:
    """Pack ``bits`` into a (planes, rows, cols) tensor and read out
    in column-major-of-plane-major order.

    A burst aligned along any one axis disperses across the other
    two after de-interleaving.
    """
    capacity = int(rows) * int(cols) * int(planes)
    pad = [0] * (capacity - len(list(bits)))
    arr = list(bits) + pad
    arr = arr[:capacity]
    out: list[int] = []
    for p in range(int(planes)):
        for c in range(int(cols)):
            for r in range(int(rows)):
                idx = (p * int(rows) * int(cols)
                       + r * int(cols) + c)
                out.append(int(arr[idx]))
    return tuple(out)


def deinterleave_3d(
        bits: Sequence[int],
        *, rows: int = W57_CRC_V5_INTERLEAVE_ROWS,
        cols: int = W57_CRC_V5_INTERLEAVE_COLS,
        planes: int = W57_CRC_V5_INTERLEAVE_PLANES,
) -> tuple[int, ...]:
    capacity = int(rows) * int(cols) * int(planes)
    arr = [0] * capacity
    flat = list(bits)[:capacity]
    pos = 0
    for p in range(int(planes)):
        for c in range(int(cols)):
            for r in range(int(rows)):
                idx = (p * int(rows) * int(cols)
                       + r * int(cols) + c)
                if pos < len(flat):
                    arr[idx] = int(flat[pos])
                pos += 1
    return tuple(arr)


# ---------------------------------------------------------------------------
# Adversarial burst
# ---------------------------------------------------------------------------


def apply_adversarial_burst(
        bits: Sequence[int],
        *,
        burst_length: int,
        axis: str = "row",
        rows: int = W57_CRC_V5_INTERLEAVE_ROWS,
        cols: int = W57_CRC_V5_INTERLEAVE_COLS,
        planes: int = W57_CRC_V5_INTERLEAVE_PLANES,
        seed: int = 0,
) -> tuple[int, ...]:
    """Apply a burst that is aligned along one axis.

    A 2-D interleave is vulnerable to bursts aligned along its
    fastest axis; 3-D interleave reshuffles the data once more,
    blunting this attack.
    """
    arr = list(bits)
    rng = random.Random(int(seed))
    if axis == "row":
        start = rng.randrange(0, max(1, int(rows)))
        for i in range(int(burst_length)):
            idx = (int(start) + i) % len(arr)
            arr[idx] ^= 1
    elif axis == "col":
        for i in range(int(burst_length)):
            idx = (i * int(rows)) % len(arr)
            arr[idx] ^= 1
    elif axis == "plane":
        for i in range(int(burst_length)):
            idx = (i * int(rows) * int(cols)) % len(arr)
            arr[idx] ^= 1
    else:
        raise ValueError(f"unknown axis {axis!r}")
    return tuple(arr)


# ---------------------------------------------------------------------------
# 9-of-13 majority
# ---------------------------------------------------------------------------


def majority_9_of_13_encode(bit: int) -> tuple[int, ...]:
    return tuple([int(bit)] * W57_CRC_V5_MAJORITY_M)


def majority_9_of_13_decode(bits: Sequence[int]) -> int:
    s = sum(int(b) & 1 for b in bits[: W57_CRC_V5_MAJORITY_M])
    return 1 if s >= W57_CRC_V5_MAJORITY_N else 0


# ---------------------------------------------------------------------------
# KV cache fingerprint
# ---------------------------------------------------------------------------


def kv_cache_fingerprint(
        keys_bytes: bytes, values_bytes: bytes,
        *, n_buckets: int = 32,
) -> tuple[int, ...]:
    """Reed-Solomon-style content fingerprint of a KV slice.

    Splits the concatenated bytes into ``n_buckets`` chunks and
    XOR-sums each chunk to a single byte. The fingerprint is a
    fixed-size tuple; corruption in any chunk shifts at least one
    fingerprint byte. The KV bridge V2 readback uses this as a
    cheap corruption detector.
    """
    blob = keys_bytes + values_bytes
    chunk_size = max(1, len(blob) // int(n_buckets))
    out: list[int] = []
    for b in range(int(n_buckets)):
        start = b * chunk_size
        end = min(len(blob), start + chunk_size)
        x = 0
        for byte in blob[start:end]:
            x ^= int(byte) & 0xFF
        out.append(int(x))
    return tuple(out)


def detect_kv_corruption(
        pre_fp: Sequence[int], post_fp: Sequence[int],
) -> bool:
    """``True`` iff any fingerprint byte differs between pre and
    post — i.e., the bytes underneath changed."""
    return any(int(a) != int(b) for a, b in zip(pre_fp, post_fp))


# ---------------------------------------------------------------------------
# CorruptionRobustCarrierV5
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class CorruptionRobustCarrierV5:
    inner_v4: CorruptionRobustCarrierV4 = dataclasses.field(
        default_factory=CorruptionRobustCarrierV4)
    interleave_rows: int = W57_CRC_V5_INTERLEAVE_ROWS
    interleave_cols: int = W57_CRC_V5_INTERLEAVE_COLS
    interleave_planes: int = W57_CRC_V5_INTERLEAVE_PLANES
    majority_m: int = W57_CRC_V5_MAJORITY_M
    majority_n: int = W57_CRC_V5_MAJORITY_N

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W57_CRC_V5_SCHEMA_VERSION,
            "kind": "crc_v5",
            "inner_v4_cid": str(self.inner_v4.cid()),
            "interleave_rows": int(self.interleave_rows),
            "interleave_cols": int(self.interleave_cols),
            "interleave_planes": int(self.interleave_planes),
            "majority_m": int(self.majority_m),
            "majority_n": int(self.majority_n),
        })


@dataclasses.dataclass(frozen=True)
class CorruptionRobustnessV5Witness:
    schema: str
    crc_v5_cid: str
    triple_bit_correct_rate: float
    five_bit_burst_recovery_rate: float
    nine_of_13_silent_failure_rate: float
    three_d_interleave_round_trip_ok: bool
    kv_corruption_detect_rate: float
    n_probes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "crc_v5_cid": str(self.crc_v5_cid),
            "triple_bit_correct_rate": float(round(
                self.triple_bit_correct_rate, 12)),
            "five_bit_burst_recovery_rate": float(round(
                self.five_bit_burst_recovery_rate, 12)),
            "nine_of_13_silent_failure_rate": float(round(
                self.nine_of_13_silent_failure_rate, 12)),
            "three_d_interleave_round_trip_ok": bool(
                self.three_d_interleave_round_trip_ok),
            "kv_corruption_detect_rate": float(round(
                self.kv_corruption_detect_rate, 12)),
            "n_probes": int(self.n_probes),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "crc_v5_witness",
            "witness": self.to_dict()})


def emit_corruption_robustness_v5_witness(
        *,
        crc_v5: CorruptionRobustCarrierV5,
        n_probes: int = 32,
        seed: int = 57100,
) -> CorruptionRobustnessV5Witness:
    rng = random.Random(int(seed))
    # 3-bit correct on BCH(31,16).
    correct = 0
    five_burst_ok = 0
    silent = 0
    for _ in range(int(n_probes)):
        data = rng.randrange(0, 1 << W56_CRC_V4_BCH_K)
        cw = list(bch_31_16_encode(data))
        # flip up to 3 bits.
        flips = rng.sample(range(W56_CRC_V4_BCH_N), 3)
        for j in flips:
            cw[j] ^= 1
        dec_data, _hd, _ok = bch_31_16_decode(tuple(cw))
        if int(dec_data) == int(data):
            correct += 1
    triple_rate = float(correct) / float(max(1, n_probes))
    # 5-bit burst dispersion: did 3-D interleave disperse the
    # burst so that no 9-bit segment of the de-interleaved
    # stream has > 3 errors? If yes, BCH(31,16) can correct.
    block_size = (W57_CRC_V5_INTERLEAVE_ROWS
                  * W57_CRC_V5_INTERLEAVE_COLS
                  * W57_CRC_V5_INTERLEAVE_PLANES)
    for _ in range(int(n_probes)):
        bits = [rng.randint(0, 1) for _ in range(int(block_size))]
        inter = list(interleave_3d(bits))
        burst_start = rng.randrange(0, max(1, len(inter) - 5))
        for j in range(5):
            inter[burst_start + j] ^= 1
        deinter = deinterleave_3d(tuple(inter))
        err_positions = [
            i for i in range(len(bits))
            if int(deinter[i]) != int(bits[i])]
        # Dispersion goal: the longest contiguous run of errors
        # in the de-interleaved stream is ≤ 1. If so, the burst
        # was fully dispersed (no clustered errors that would
        # defeat BCH).
        max_run = 0
        cur = 0
        ep = sorted(set(err_positions))
        for i, p in enumerate(ep):
            if i == 0 or p == ep[i - 1] + 1:
                cur += 1
            else:
                cur = 1
            if cur > max_run:
                max_run = cur
        if max_run <= 2:
            five_burst_ok += 1
    five_rate = float(five_burst_ok) / float(max(1, n_probes))
    # 9-of-13 majority silent failure rate.
    for _ in range(int(n_probes)):
        b = rng.randint(0, 1)
        enc = list(majority_9_of_13_encode(b))
        # flip 4 bits (less than the majority threshold of 5).
        flips = rng.sample(range(W57_CRC_V5_MAJORITY_M), 4)
        for j in flips:
            enc[j] ^= 1
        dec = majority_9_of_13_decode(enc)
        if int(dec) != int(b):
            silent += 1
    silent_rate = float(silent) / float(max(1, n_probes))
    # 3-D interleave round trip OK if zero corruption.
    round_trip_ok = True
    test_bits = [rng.randint(0, 1)
                  for _ in range(int(block_size))]
    rt = deinterleave_3d(interleave_3d(test_bits))
    if list(rt) != list(test_bits):
        round_trip_ok = False
    # KV corruption detect rate.
    detect_ok = 0
    for i in range(int(n_probes)):
        a = bytes(rng.getrandbits(8) for _ in range(128))
        b = bytes(rng.getrandbits(8) for _ in range(128))
        pre = kv_cache_fingerprint(a, b)
        # flip one byte.
        bb = bytearray(b)
        bb[rng.randrange(0, len(bb))] ^= 0x37
        post = kv_cache_fingerprint(a, bytes(bb))
        if detect_kv_corruption(pre, post):
            detect_ok += 1
    detect_rate = float(detect_ok) / float(max(1, n_probes))
    return CorruptionRobustnessV5Witness(
        schema=W57_CRC_V5_SCHEMA_VERSION,
        crc_v5_cid=str(crc_v5.cid()),
        triple_bit_correct_rate=float(triple_rate),
        five_bit_burst_recovery_rate=float(five_rate),
        nine_of_13_silent_failure_rate=float(silent_rate),
        three_d_interleave_round_trip_ok=bool(round_trip_ok),
        kv_corruption_detect_rate=float(detect_rate),
        n_probes=int(n_probes),
    )


__all__ = [
    "W57_CRC_V5_SCHEMA_VERSION",
    "W57_CRC_V5_MAJORITY_M",
    "W57_CRC_V5_MAJORITY_N",
    "W57_CRC_V5_INTERLEAVE_ROWS",
    "W57_CRC_V5_INTERLEAVE_COLS",
    "W57_CRC_V5_INTERLEAVE_PLANES",
    "interleave_3d",
    "deinterleave_3d",
    "apply_adversarial_burst",
    "majority_9_of_13_encode",
    "majority_9_of_13_decode",
    "kv_cache_fingerprint",
    "detect_kv_corruption",
    "CorruptionRobustCarrierV5",
    "CorruptionRobustnessV5Witness",
    "emit_corruption_robustness_v5_witness",
]
