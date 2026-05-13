"""W55 M5 — Corruption-Robust Carrier V3 (BCH(15,7) double-bit
   correction + 5-of-7 repetition + bit-interleaving).

Extends W54 V2 with three layers:

* **BCH(15,7) per-segment double-bit correction** — replaces V2's
  Hamming(7,4) single-bit correction. The BCH(15,7) code has
  minimum distance 5, so it can correct up to 2 errors per
  segment and detect up to 4 errors (cap: ~5+-bit pathologies
  can mis-correct, documented as W55-L-BCH-FIVE-BIT-PATHOLOGY).
* **5-of-7 majority repetition** — replaces V2's 3-of-5. The
  per-bit error rate drops to ~``35p⁴(1-p)³`` for 4-or-more
  errors (vs V2's ``10p³(1-p)²``).
* **Bit interleaving** — after encoding all segments, bits are
  transposed so adjacent transmitted bits come from different
  segments. A burst of length L gets dispersed across segments
  so each segment sees at most ``ceil(L / n_segments)`` errors.

V3 wraps V2's ``CorruptionRobustCarrierV2`` for the bit
extraction + 5-of-7 majority repetition, and adds dedicated
BCH(15,7) encode/decode + interleave/deinterleave operations.

Honest scope: pure-Python only, capsule-layer only.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any, Sequence

from .autograd_manifold import (
    W47_DEFAULT_TRAIN_SEED,
    _DeterministicLCG,
)
from .corruption_robust_carrier import (
    HostileChannelProbeResult,
)
from .corruption_robust_carrier_v2 import (
    CorruptionRobustCarrierV2,
    W54_DEFAULT_CRC_V2_MAJORITY,
)
from .ecc_codebook_v5 import (
    ECCCodebookV5,
    ECCDecodeAttempt,
    flip_random_bit,
)
from .quantised_compression import QuantisedBudgetGate


# =============================================================================
# Schema, defaults
# =============================================================================

W55_CRC_V3_SCHEMA_VERSION: str = (
    "coordpy.corruption_robust_carrier_v3.v1")

W55_DEFAULT_CRC_V3_REPETITION: int = 7
W55_DEFAULT_CRC_V3_MAJORITY: int = 4  # 4-of-7 majority
W55_BCH_15_7_DATA_BITS: int = 7
W55_BCH_15_7_PARITY_BITS: int = 8
W55_BCH_15_7_TOTAL_BITS: int = 15
W55_BCH_15_7_MIN_DISTANCE: int = 5
W55_BCH_15_7_CORRECT_T: int = 2


# =============================================================================
# Helpers
# =============================================================================


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for i in range(n):
        ai = float(a[i])
        bi = float(b[i])
        dot += ai * bi
        na += ai * ai
        nb += bi * bi
    if na <= 1e-30 or nb <= 1e-30:
        return 0.0
    return float(dot / (math.sqrt(na) * math.sqrt(nb)))


def _hamming_distance(
        a: Sequence[int], b: Sequence[int]) -> int:
    n = min(len(a), len(b))
    return sum(
        1 for i in range(n)
        if int(a[i]) & 1 != int(b[i]) & 1)


# =============================================================================
# BCH(15,7) encode + decode
# =============================================================================


# Generator polynomial: g(x) = (x^4+x+1)(x^4+x^3+x^2+x+1)
#                            = x^8 + x^7 + x^6 + x^4 + 1
# Binary: 111010001 (coefficients of x^8 .. x^0)
_BCH_15_7_GEN_POLY: tuple[int, ...] = (
    1, 1, 1, 0, 1, 0, 0, 0, 1)


def _poly_mod(
        dividend: Sequence[int],
        divisor: Sequence[int]) -> list[int]:
    """Compute polynomial division remainder over GF(2)."""
    rem = [int(b) & 1 for b in dividend]
    d_len = len(divisor)
    for i in range(len(rem) - d_len + 1):
        if rem[i]:
            for j in range(d_len):
                rem[i + j] ^= int(divisor[j])
    return rem[-(d_len - 1):]


def bch_15_7_encode(data7: Sequence[int]) -> tuple[int, ...]:
    """Encode 7 data bits as a 15-bit BCH(15,7) systematic codeword.

    Codeword = data || parity, where parity = (data << 8) mod g(x).
    """
    d = [int(b) & 1 for b in data7]
    while len(d) < W55_BCH_15_7_DATA_BITS:
        d.append(0)
    d = d[:W55_BCH_15_7_DATA_BITS]
    shifted = list(d) + [0] * W55_BCH_15_7_PARITY_BITS
    parity = _poly_mod(shifted, _BCH_15_7_GEN_POLY)
    return tuple(d + parity)


def _bch_15_7_valid_codewords() -> list[tuple[int, ...]]:
    """Return all 128 valid BCH(15,7) codewords."""
    out: list[tuple[int, ...]] = []
    for v in range(128):
        bits = [(v >> i) & 1 for i in range(7)]
        out.append(bch_15_7_encode(bits))
    return out


_BCH_15_7_CODEBOOK: list[tuple[int, ...]] = (
    _bch_15_7_valid_codewords())


def bch_15_7_decode(
        codeword15: Sequence[int],
) -> tuple[tuple[int, int, int, int, int, int, int],
            int, int, bool]:
    """Decode a 15-bit BCH(15,7) word.

    Returns (data7, n_corrections, error_dist, double_or_more):
        * data7: best-guess 7 data bits
        * n_corrections: hamming distance to chosen codeword
            (0 = no error; 1 = 1 error corrected; 2 = 2 errors
            corrected; >2 = uncorrectable or mis-corrected)
        * error_dist: int — same as n_corrections
        * double_or_more: True iff dist >= 3 (uncorrectable)
    """
    r = [int(b) & 1 for b in codeword15]
    while len(r) < W55_BCH_15_7_TOTAL_BITS:
        r.append(0)
    r = r[:W55_BCH_15_7_TOTAL_BITS]
    # Minimum-distance decoding by table scan.
    best_dist = W55_BCH_15_7_TOTAL_BITS + 1
    best_cw: tuple[int, ...] | None = None
    for cw in _BCH_15_7_CODEBOOK:
        d = _hamming_distance(r, cw)
        if d < best_dist:
            best_dist = d
            best_cw = cw
            if d == 0:
                break
    if best_cw is None:
        # Should not happen.
        return ((0, 0, 0, 0, 0, 0, 0), 0, 0, True)
    data = tuple(int(b) for b in best_cw[
        :W55_BCH_15_7_DATA_BITS])
    return (
        (data[0], data[1], data[2], data[3],
         data[4], data[5], data[6]),
        int(best_dist), int(best_dist),
        bool(best_dist > W55_BCH_15_7_CORRECT_T))


# =============================================================================
# Bit interleaving
# =============================================================================


def interleave_bits(
        segment_bits: Sequence[Sequence[int]],
) -> list[int]:
    """Transpose segment-major to bit-major order.

    Input: list of segments, each a list of bits.
    Output: interleaved [seg0_bit0, seg1_bit0, ..., seg0_bit1, ...].
    All segments must have equal length; shorter ones get zero-padded.
    """
    if not segment_bits:
        return []
    max_len = max(len(s) for s in segment_bits)
    out: list[int] = []
    for bit_idx in range(max_len):
        for seg_idx in range(len(segment_bits)):
            seg = segment_bits[seg_idx]
            if bit_idx < len(seg):
                out.append(int(seg[bit_idx]) & 1)
            else:
                out.append(0)
    return out


def deinterleave_bits(
        flat_bits: Sequence[int],
        *,
        n_segments: int,
        bits_per_segment: int,
) -> list[list[int]]:
    """Inverse of interleave_bits."""
    out: list[list[int]] = [
        [0] * int(bits_per_segment)
        for _ in range(int(n_segments))
    ]
    idx = 0
    for bit_idx in range(int(bits_per_segment)):
        for seg_idx in range(int(n_segments)):
            if idx < len(flat_bits):
                out[seg_idx][bit_idx] = int(
                    flat_bits[idx]) & 1
                idx += 1
    return out


# =============================================================================
# 5-of-7 majority repetition
# =============================================================================


def repeat_bits_n_of_m(
        bits: Sequence[int],
        *,
        n_rep: int = W55_DEFAULT_CRC_V3_REPETITION,
) -> list[int]:
    """Send each bit ``n_rep`` times in a row."""
    out: list[int] = []
    for b in bits:
        for _ in range(int(n_rep)):
            out.append(int(b) & 1)
    return out


def majority_decode_n_of_m(
        bits: Sequence[int],
        *,
        n_rep: int = W55_DEFAULT_CRC_V3_REPETITION,
        majority: int = W55_DEFAULT_CRC_V3_MAJORITY,
) -> list[int]:
    """Decode ``n_rep`` repetition by ``majority``-of-``n_rep`` vote."""
    out: list[int] = []
    for i in range(0, len(bits), int(n_rep)):
        s = sum(
            int(bits[i + j]) & 1
            for j in range(int(n_rep))
            if i + j < len(bits))
        out.append(1 if s >= int(majority) else 0)
    return out


# =============================================================================
# Corruption-robust V3 wrapper
# =============================================================================


@dataclasses.dataclass
class CorruptionRobustCarrierV3:
    """V3 wrapper: V2 inner + BCH(15,7) + 5-of-7 repetition + interleave."""

    inner_v2: CorruptionRobustCarrierV2
    repetition: int
    majority: int

    @classmethod
    def init(
            cls, *,
            inner_v2: CorruptionRobustCarrierV2 | None = None,
            codebook: ECCCodebookV5 | None = None,
            gate: QuantisedBudgetGate | None = None,
            repetition: int = W55_DEFAULT_CRC_V3_REPETITION,
            majority: int = W55_DEFAULT_CRC_V3_MAJORITY,
    ) -> "CorruptionRobustCarrierV3":
        if inner_v2 is None:
            if codebook is None or gate is None:
                raise ValueError(
                    "CorruptionRobustCarrierV3.init: provide "
                    "inner_v2 OR (codebook, gate)")
            inner_v2 = CorruptionRobustCarrierV2.init(
                codebook=codebook, gate=gate)
        return cls(
            inner_v2=inner_v2,
            repetition=int(repetition),
            majority=int(majority),
        )

    def encode_value(
            self, carrier: Sequence[float],
    ) -> tuple[list[int], list[int]]:
        """Encode a carrier vector.

        Returns (bch_bits_per_segment, interleaved_5_of_7_payload).
        """
        cb = self.inner_v2.codebook
        coarse, fine, ultra, ultra2 = cb.encode_value(carrier)
        seg_bits = []
        for code, n_bits in [
                (coarse, cb.coarse_bits()),
                (fine, cb.fine_bits()),
                (ultra, cb.ultra_bits()),
                (ultra2, cb.ultra2_bits())]:
            bits = [(code >> i) & 1 for i in range(n_bits)]
            # Pad to 7 data bits for BCH.
            while len(bits) < W55_BCH_15_7_DATA_BITS:
                bits.append(0)
            bch_cw = bch_15_7_encode(
                bits[:W55_BCH_15_7_DATA_BITS])
            seg_bits.append(list(bch_cw))
        interleaved = interleave_bits(seg_bits)
        # Apply 5-of-7 majority repetition.
        repeated = repeat_bits_n_of_m(
            interleaved, n_rep=int(self.repetition))
        return seg_bits, repeated

    def decode_value(
            self,
            repeated_bits: Sequence[int],
            *,
            n_segments: int = 4,
    ) -> tuple[list[tuple[int, ...]], int, int]:
        """Decode repeated bits.

        Returns:
            (decoded_data_per_segment, n_segments_corrected,
             n_segments_uncorrectable)
        """
        # Step 1: majority-decode repetition.
        flat = majority_decode_n_of_m(
            repeated_bits,
            n_rep=int(self.repetition),
            majority=int(self.majority))
        # Step 2: deinterleave.
        seg_bits = deinterleave_bits(
            flat,
            n_segments=int(n_segments),
            bits_per_segment=W55_BCH_15_7_TOTAL_BITS)
        decoded: list[tuple[int, ...]] = []
        n_corrected = 0
        n_uncorrectable = 0
        for s in seg_bits:
            data, n_corr, _, double = bch_15_7_decode(s)
            decoded.append(data)
            if n_corr > 0 and n_corr <= W55_BCH_15_7_CORRECT_T:
                n_corrected += 1
            if double:
                n_uncorrectable += 1
        return decoded, n_corrected, n_uncorrectable

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w55_crc_v3",
            "inner_v2_cid": str(self.inner_v2.cid()),
            "repetition": int(self.repetition),
            "majority": int(self.majority),
        })

    @property
    def codebook(self) -> ECCCodebookV5:
        return self.inner_v2.codebook

    @property
    def gate(self) -> QuantisedBudgetGate:
        return self.inner_v2.gate


# =============================================================================
# Hostile-channel probe V3
# =============================================================================


@dataclasses.dataclass(frozen=True)
class HostileChannelProbeResultV3:
    n_carriers: int
    flip_intensity: float
    detect_rate: float
    correct_rate: float
    silent_failure_rate: float
    interleave_burst_recovery_rate: float
    bch_double_bit_correct_rate: float
    bch_three_bit_detect_rate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_carriers": int(self.n_carriers),
            "flip_intensity": float(round(
                self.flip_intensity, 12)),
            "detect_rate": float(round(self.detect_rate, 12)),
            "correct_rate": float(round(
                self.correct_rate, 12)),
            "silent_failure_rate": float(round(
                self.silent_failure_rate, 12)),
            "interleave_burst_recovery_rate": float(round(
                self.interleave_burst_recovery_rate, 12)),
            "bch_double_bit_correct_rate": float(round(
                self.bch_double_bit_correct_rate, 12)),
            "bch_three_bit_detect_rate": float(round(
                self.bch_three_bit_detect_rate, 12)),
        }


def probe_hostile_channel_v3(
        crc_v3: CorruptionRobustCarrierV3,
        *,
        carriers: Sequence[Sequence[float]],
        flip_intensity: float = 1.0,
        seed: int = W47_DEFAULT_TRAIN_SEED,
) -> HostileChannelProbeResultV3:
    """Probe BCH + interleave + repetition with bit flips.

    For each carrier:
    1. encode → repeated bits.
    2. flip ``floor(flip_intensity)`` random bits in the repeated
       stream.
    3. decode and check correctness.

    Returns rates for detect / correct / silent_failure /
    interleave_burst_recovery / bch_double_bit_correct /
    bch_three_bit_detect.
    """
    rng = _DeterministicLCG(seed=int(seed))
    n = len(carriers)
    if n == 0:
        return HostileChannelProbeResultV3(
            n_carriers=0,
            flip_intensity=float(flip_intensity),
            detect_rate=0.0, correct_rate=0.0,
            silent_failure_rate=0.0,
            interleave_burst_recovery_rate=0.0,
            bch_double_bit_correct_rate=0.0,
            bch_three_bit_detect_rate=0.0,
        )
    n_detect = 0
    n_correct = 0
    n_silent = 0
    n_burst_recover = 0
    n_double_correct = 0
    n_triple_detect = 0
    for carrier in carriers:
        # Encode and capture clean ground truth.
        seg_bits_clean, repeated_clean = (
            crc_v3.encode_value(carrier))
        decoded_clean, _, _ = crc_v3.decode_value(
            repeated_clean, n_segments=len(seg_bits_clean))
        # Apply random bit flips proportional to flip_intensity.
        repeated_corrupt = list(repeated_clean)
        n_flips = int(max(0, math.floor(
            float(flip_intensity)
            * len(repeated_corrupt) * 0.01)))
        if n_flips == 0:
            n_flips = 1
        for _ in range(n_flips):
            idx = int(rng.next_uniform()
                       * len(repeated_corrupt)) % len(
                repeated_corrupt)
            repeated_corrupt[idx] ^= 1
        decoded_corrupt, n_corr, n_unc = crc_v3.decode_value(
            repeated_corrupt,
            n_segments=len(seg_bits_clean))
        # Detection: did decoder flag any segment as
        # uncorrectable or correct?
        if n_unc > 0 or n_corr > 0:
            n_detect += 1
        # Correctness: did decoded data match clean data?
        match = (decoded_corrupt == decoded_clean)
        if match:
            n_correct += 1
        elif not match and n_unc == 0 and n_corr == 0:
            n_silent += 1
        # Burst recovery: apply a 3-bit burst and check.
        burst_corrupt = list(repeated_clean)
        burst_start = int(rng.next_uniform()
                           * (len(burst_corrupt) - 4)) % (
            max(1, len(burst_corrupt) - 4))
        for k in range(3):
            burst_corrupt[burst_start + k] ^= 1
        decoded_burst, _, _ = crc_v3.decode_value(
            burst_corrupt,
            n_segments=len(seg_bits_clean))
        if decoded_burst == decoded_clean:
            n_burst_recover += 1
        # BCH double-bit correct: flip 2 entire rep-groups in
        # segment 0 (so majority decode produces 2 bit flips
        # *inside* the deinterleaved segment-0 BCH codeword).
        bch_double = list(repeated_clean)
        n_seg = len(seg_bits_clean)
        # Bit positions for segment 0, bit-indices 0 and 1, in
        # interleaved order. Each is a rep-group of size
        # crc_v3.repetition starting at index
        # (bit_idx * n_seg + seg_idx) * rep.
        for bi in (0, 1):
            base = (bi * n_seg + 0) * crc_v3.repetition
            for r in range(crc_v3.repetition):
                if base + r < len(bch_double):
                    bch_double[base + r] ^= 1
        decoded_bch_2, n_corr_2, _ = crc_v3.decode_value(
            bch_double, n_segments=len(seg_bits_clean))
        if decoded_bch_2 == decoded_clean:
            n_double_correct += 1
        # BCH triple-bit detect: flip 3 entire rep-groups in
        # segment 0 at randomised bit-indices (honest probe —
        # some 3-bit patterns mis-correct; others are detected).
        bch_triple = list(repeated_clean)
        # Pick 3 distinct bit indices in [0, W55_BCH_15_7_TOTAL_BITS)
        # using the rng so probe is reproducible across seeds.
        candidates = list(range(
            W55_BCH_15_7_TOTAL_BITS))
        chosen = []
        for _ in range(3):
            if not candidates:
                break
            idx = int(rng.next_uniform() * len(candidates)) % (
                max(1, len(candidates)))
            chosen.append(candidates.pop(idx))
        for bi in chosen:
            base = (bi * n_seg + 0) * crc_v3.repetition
            for r in range(crc_v3.repetition):
                if base + r < len(bch_triple):
                    bch_triple[base + r] ^= 1
        _, _, n_unc_3 = crc_v3.decode_value(
            bch_triple, n_segments=len(seg_bits_clean))
        if n_unc_3 > 0:
            n_triple_detect += 1
    fn = float(max(1, n))
    return HostileChannelProbeResultV3(
        n_carriers=int(n),
        flip_intensity=float(flip_intensity),
        detect_rate=float(n_detect) / fn,
        correct_rate=float(n_correct) / fn,
        silent_failure_rate=float(n_silent) / fn,
        interleave_burst_recovery_rate=float(
            n_burst_recover) / fn,
        bch_double_bit_correct_rate=float(
            n_double_correct) / fn,
        bch_three_bit_detect_rate=float(
            n_triple_detect) / fn,
    )


# =============================================================================
# Witness
# =============================================================================


@dataclasses.dataclass(frozen=True)
class CorruptionRobustnessV3Witness:
    crc_v3_cid: str
    n_carriers: int
    flip_intensity: float
    detect_rate: float
    correct_rate: float
    silent_failure_rate: float
    interleave_burst_recovery_rate: float
    bch_double_bit_correct_rate: float
    bch_three_bit_detect_rate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "crc_v3_cid": str(self.crc_v3_cid),
            "n_carriers": int(self.n_carriers),
            "flip_intensity": float(round(
                self.flip_intensity, 12)),
            "detect_rate": float(round(
                self.detect_rate, 12)),
            "correct_rate": float(round(
                self.correct_rate, 12)),
            "silent_failure_rate": float(round(
                self.silent_failure_rate, 12)),
            "interleave_burst_recovery_rate": float(round(
                self.interleave_burst_recovery_rate, 12)),
            "bch_double_bit_correct_rate": float(round(
                self.bch_double_bit_correct_rate, 12)),
            "bch_three_bit_detect_rate": float(round(
                self.bch_three_bit_detect_rate, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w55_crc_v3_witness",
            "witness": self.to_dict()})


def emit_corruption_robustness_v3_witness(
        *,
        crc_v3: CorruptionRobustCarrierV3,
        carriers: Sequence[Sequence[float]],
        flip_intensity: float = 1.0,
        seed: int = W47_DEFAULT_TRAIN_SEED,
) -> CorruptionRobustnessV3Witness:
    result = probe_hostile_channel_v3(
        crc_v3, carriers=carriers,
        flip_intensity=float(flip_intensity),
        seed=int(seed))
    return CorruptionRobustnessV3Witness(
        crc_v3_cid=str(crc_v3.cid()),
        n_carriers=int(result.n_carriers),
        flip_intensity=float(result.flip_intensity),
        detect_rate=float(result.detect_rate),
        correct_rate=float(result.correct_rate),
        silent_failure_rate=float(result.silent_failure_rate),
        interleave_burst_recovery_rate=float(
            result.interleave_burst_recovery_rate),
        bch_double_bit_correct_rate=float(
            result.bch_double_bit_correct_rate),
        bch_three_bit_detect_rate=float(
            result.bch_three_bit_detect_rate),
    )


# =============================================================================
# Verifier
# =============================================================================

W55_CRC_V3_VERIFIER_FAILURE_MODES: tuple[str, ...] = (
    "w55_crc_v3_cid_mismatch",
    "w55_crc_v3_silent_failure_above_ceiling",
    "w55_crc_v3_bch_double_correct_below_floor",
    "w55_crc_v3_burst_recovery_below_floor",
)


def verify_corruption_robustness_v3_witness(
        witness: CorruptionRobustnessV3Witness,
        *,
        expected_crc_v3_cid: str | None = None,
        max_silent_failure: float | None = None,
        min_bch_double_correct: float | None = None,
        min_burst_recovery: float | None = None,
) -> dict[str, Any]:
    failures: list[str] = []
    if (expected_crc_v3_cid is not None
            and witness.crc_v3_cid
            != str(expected_crc_v3_cid)):
        failures.append("w55_crc_v3_cid_mismatch")
    if (max_silent_failure is not None
            and witness.silent_failure_rate
            > float(max_silent_failure)):
        failures.append(
            "w55_crc_v3_silent_failure_above_ceiling")
    if (min_bch_double_correct is not None
            and witness.bch_double_bit_correct_rate
            < float(min_bch_double_correct)):
        failures.append(
            "w55_crc_v3_bch_double_correct_below_floor")
    if (min_burst_recovery is not None
            and witness.interleave_burst_recovery_rate
            < float(min_burst_recovery)):
        failures.append(
            "w55_crc_v3_burst_recovery_below_floor")
    return {
        "ok": (len(failures) == 0),
        "failures": failures,
        "witness_cid": witness.cid(),
    }


__all__ = [
    "W55_CRC_V3_SCHEMA_VERSION",
    "W55_DEFAULT_CRC_V3_REPETITION",
    "W55_DEFAULT_CRC_V3_MAJORITY",
    "W55_BCH_15_7_DATA_BITS",
    "W55_BCH_15_7_PARITY_BITS",
    "W55_BCH_15_7_TOTAL_BITS",
    "W55_BCH_15_7_MIN_DISTANCE",
    "W55_BCH_15_7_CORRECT_T",
    "W55_CRC_V3_VERIFIER_FAILURE_MODES",
    "CorruptionRobustCarrierV3",
    "CorruptionRobustnessV3Witness",
    "HostileChannelProbeResultV3",
    "bch_15_7_encode",
    "bch_15_7_decode",
    "interleave_bits",
    "deinterleave_bits",
    "repeat_bits_n_of_m",
    "majority_decode_n_of_m",
    "probe_hostile_channel_v3",
    "emit_corruption_robustness_v3_witness",
    "verify_corruption_robustness_v3_witness",
]
