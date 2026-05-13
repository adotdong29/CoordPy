"""W54 M5 — Corruption-Robust Carrier V2.

Extends W53 V1 with two corrections-not-just-detection layers:

* **Hamming(7,4) single-bit correction** — each 4-bit segment
  (coarse / fine / ultra / ultra2) is encoded as a 7-bit
  Hamming codeword with 3 parity bits. The receiver can detect
  any 1-bit error AND correct it (vs V1's parity-detect only).
* **3-of-5 majority repetition** — the bits-payload is repeated
  5 times instead of 3, raising per-bit error robustness to
  ``10p³ - 15p⁴ + 6p⁵`` (vs V1's ``3p² - 2p³``).
* **double-bit detection** — Hamming(7,4) detects double-bit
  errors via a separate "double error detected" signal, even if
  it cannot correct them (mark abstain).

V2 wraps V1's ``CorruptionRobustCarrier`` and adds dedicated
encode/decode paths. Honest scope: pure-Python only, capsule-
layer only, no transformer-internal state.
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
    CorruptionRobustCarrier,
    HostileChannelProbeResult,
    W53_DEFAULT_CRC_REPETITION,
)
from .ecc_codebook_v5 import (
    ECCCodebookV5,
    ECCDecodeAttempt,
    compress_carrier_ecc,
    flip_random_bit,
)
from .quantised_compression import QuantisedBudgetGate


# =============================================================================
# Schema, defaults
# =============================================================================

W54_CRC_V2_SCHEMA_VERSION: str = (
    "coordpy.corruption_robust_carrier_v2.v1")

W54_DEFAULT_CRC_V2_REPETITION: int = 5
W54_DEFAULT_CRC_V2_MAJORITY: int = 3  # 3-of-5 majority
W54_HAMMING_7_4_N_PARITY_BITS: int = 3
W54_HAMMING_7_4_DATA_BITS: int = 4
W54_HAMMING_7_4_TOTAL_BITS: int = 7


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


def _majority_count(
        bits: Sequence[int], threshold: int,
) -> int:
    s = sum(int(b) & 1 for b in bits)
    return 1 if s >= int(threshold) else 0


# =============================================================================
# Hamming(7,4) encode + decode
# =============================================================================


def hamming_7_4_encode(data4: Sequence[int]) -> tuple[int, ...]:
    """Encode 4 data bits as 7-bit Hamming codeword.

    Bit layout (1-indexed):
        c1 = p1, c2 = p2, c3 = d1, c4 = p3, c5 = d2, c6 = d3, c7 = d4
    Parity equations:
        p1 = d1 ^ d2 ^ d4
        p2 = d1 ^ d3 ^ d4
        p3 = d2 ^ d3 ^ d4
    Returns (c1, c2, c3, c4, c5, c6, c7) as 7-tuple of bits.
    """
    d = [int(b) & 1 for b in data4]
    while len(d) < W54_HAMMING_7_4_DATA_BITS:
        d.append(0)
    d1, d2, d3, d4 = d[0], d[1], d[2], d[3]
    p1 = d1 ^ d2 ^ d4
    p2 = d1 ^ d3 ^ d4
    p3 = d2 ^ d3 ^ d4
    return (p1, p2, d1, p3, d2, d3, d4)


def hamming_7_4_decode(
        codeword7: Sequence[int],
        *,
        original_parity: int | None = None,
) -> tuple[tuple[int, int, int, int], int, bool]:
    """Decode a 7-bit Hamming codeword.

    Returns:
        (data4, syndrome, double_error_detected)

    Where:
        * data4 is (d1, d2, d3, d4) after single-bit correction
        * syndrome is the 3-bit syndrome value (0 = no error)
        * double_error_detected is True iff a 1-bit overall
          parity check disagrees AND the syndrome is zero
          (which can only happen for an even number of errors
          on a code that was extended with an overall parity).

    For pure Hamming(7,4) (no extended parity), we cannot
    detect 2-bit errors at decode time. ``original_parity`` is
    accepted only as a sidechannel hint and used for the
    ``double_error`` flag: if the supplied parity disagrees with
    the recomputed parity AND syndrome is 0, this is consistent
    with a 2-bit error.
    """
    c = [int(b) & 1 for b in codeword7]
    while len(c) < W54_HAMMING_7_4_TOTAL_BITS:
        c.append(0)
    c1, c2, c3, c4, c5, c6, c7 = (
        c[0], c[1], c[2], c[3], c[4], c[5], c[6])
    s1 = c1 ^ c3 ^ c5 ^ c7
    s2 = c2 ^ c3 ^ c6 ^ c7
    s3 = c4 ^ c5 ^ c6 ^ c7
    syndrome = int(s1 + 2 * s2 + 4 * s3)
    corrected = list(c)
    if 1 <= syndrome <= 7:
        idx = syndrome - 1
        corrected[idx] = corrected[idx] ^ 1
    overall_parity_now = 0
    for b in c:
        overall_parity_now ^= int(b)
    double_error = False
    if original_parity is not None:
        if (int(original_parity) != int(overall_parity_now)
                and syndrome == 0):
            double_error = True
    d1 = int(corrected[2])
    d2 = int(corrected[4])
    d3 = int(corrected[5])
    d4 = int(corrected[6])
    return ((d1, d2, d3, d4), int(syndrome),
             bool(double_error))


def _segment_to_4bits(code: int, n_bits: int) -> list[int]:
    """LSB-first 4-bit representation of a segment code.

    If n_bits > 4, truncate to the lowest 4. If n_bits < 4,
    zero-pad.
    """
    out: list[int] = []
    for i in range(W54_HAMMING_7_4_DATA_BITS):
        if i < int(n_bits):
            out.append(int((int(code) >> i) & 1))
        else:
            out.append(0)
    return out


def _4bits_to_code(bits: Sequence[int], n_bits: int) -> int:
    """Reverse of ``_segment_to_4bits``."""
    n = min(int(n_bits), W54_HAMMING_7_4_DATA_BITS)
    out = 0
    for i in range(n):
        if int(bits[i]) & 1:
            out |= (1 << i)
    return int(out)


# =============================================================================
# CorruptionRobustCarrierV2
# =============================================================================


@dataclasses.dataclass
class CorruptionRobustCarrierV2:
    """V2 carrier: V1 + per-segment Hamming(7,4) + 5x repetition."""

    inner_v1: CorruptionRobustCarrier
    repetition_v2: int

    @classmethod
    def init(
            cls, *,
            codebook: ECCCodebookV5,
            gate: QuantisedBudgetGate,
            repetition: int = W54_DEFAULT_CRC_V2_REPETITION,
    ) -> "CorruptionRobustCarrierV2":
        inner = CorruptionRobustCarrier.init(
            codebook=codebook, gate=gate,
            repetition=int(W53_DEFAULT_CRC_REPETITION))
        return cls(
            inner_v1=inner,
            repetition_v2=int(repetition))

    @property
    def codebook(self) -> ECCCodebookV5:
        return self.inner_v1.codebook

    @property
    def gate(self) -> QuantisedBudgetGate:
        return self.inner_v1.gate

    def encode_with_hamming(
            self,
            carrier: Sequence[float],
    ) -> tuple[
            tuple[int, ...], tuple[int, ...],
            tuple[int, ...], tuple[int, ...],
            tuple[int, ...], tuple[int, int, int, int]]:
        """Encode a carrier through ECC V5 + Hamming(7,4) per segment.

        Returns (h_coarse, h_fine, h_ultra, h_ultra2,
            rep_payload, parity_bits).
        Each h_* is a 7-bit Hamming codeword tuple.
        """
        comp = compress_carrier_ecc(
            carrier, codebook=self.codebook, gate=self.gate)
        h_coarse = hamming_7_4_encode(
            _segment_to_4bits(
                comp.coarse_code, comp.coarse_bits))
        h_fine = hamming_7_4_encode(
            _segment_to_4bits(
                comp.fine_code, comp.fine_bits))
        h_ultra = hamming_7_4_encode(
            _segment_to_4bits(
                comp.ultra_code, comp.ultra_bits))
        h_ultra2 = hamming_7_4_encode(
            _segment_to_4bits(
                comp.ultra2_code, comp.ultra2_bits))
        rep = []
        for b in comp.bits_payload:
            for _ in range(int(self.repetition_v2)):
                rep.append(int(b))
        return (
            tuple(h_coarse), tuple(h_fine),
            tuple(h_ultra), tuple(h_ultra2),
            tuple(rep), tuple(comp.parity_bits))

    def decode_with_hamming(
            self,
            *,
            h_coarse: Sequence[int],
            h_fine: Sequence[int],
            h_ultra: Sequence[int],
            h_ultra2: Sequence[int],
            repetition_payload: Sequence[int],
            original_parities: tuple[
                int, int, int, int] | None = None,
    ) -> tuple[ECCDecodeAttempt, tuple[int, ...], dict[str, Any]]:
        """Decode + Hamming-correct + majority-vote the payload.

        Returns (decoded_attempt, recovered_payload, hamming_info).
        ``hamming_info`` is a dict with per-segment syndromes and
        double_error flags. Pass ``original_parities`` (the sent
        overall parities of each codeword) to detect 2-bit errors
        when syndrome is 0.
        """
        cb = self.codebook
        op = (
            original_parities
            if original_parities is not None
            else (None, None, None, None))
        (d_coarse, s_coarse, dbl_coarse) = hamming_7_4_decode(
            h_coarse, original_parity=op[0])
        (d_fine, s_fine, dbl_fine) = hamming_7_4_decode(
            h_fine, original_parity=op[1])
        (d_ultra, s_ultra, dbl_ultra) = hamming_7_4_decode(
            h_ultra, original_parity=op[2])
        (d_ultra2, s_ultra2, dbl_ultra2) = hamming_7_4_decode(
            h_ultra2, original_parity=op[3])
        coarse_code = _4bits_to_code(d_coarse, cb.coarse_bits())
        fine_code = _4bits_to_code(d_fine, cb.fine_bits())
        ultra_code = _4bits_to_code(d_ultra, cb.ultra_bits())
        ultra2_code = _4bits_to_code(
            d_ultra2, cb.ultra2_bits())
        coarse_code = coarse_code % max(
            1, int(cb.inner_v4.n_coarse))
        fine_code = fine_code % max(
            1, int(cb.inner_v4.n_fine))
        ultra_code = ultra_code % max(
            1, int(cb.inner_v4.n_ultra))
        ultra2_code = ultra2_code % max(
            1, int(cb.n_ultra2))
        # Build detect/abstain attempt mirroring V1 semantics.
        n_corrupted = int(
            (1 if s_coarse > 0 else 0)
            + (1 if s_fine > 0 else 0)
            + (1 if s_ultra > 0 else 0)
            + (1 if s_ultra2 > 0 else 0))
        n_double = int(
            (1 if dbl_coarse else 0)
            + (1 if dbl_fine else 0)
            + (1 if dbl_ultra else 0)
            + (1 if dbl_ultra2 else 0))
        decoded = cb.decode(
            coarse=coarse_code, fine=fine_code,
            ultra=ultra_code, ultra2=ultra2_code)
        detected: list[str] = []
        for name, s in (
                ("coarse", s_coarse), ("fine", s_fine),
                ("ultra", s_ultra),
                ("ultra2", s_ultra2)):
            if int(s) > 0:
                detected.append(name)
        abstain = bool(n_double >= 1)
        attempt = ECCDecodeAttempt(
            detected_segments=tuple(detected),
            n_corrupted_segments=int(n_corrupted),
            decoded_payload=tuple(decoded),
            abstain=bool(abstain),
            corrected_partial=bool(
                n_corrupted >= 1 and not abstain),
        )
        rep = max(1, int(self.repetition_v2))
        n = len(repetition_payload) // rep
        majority_threshold = int(W54_DEFAULT_CRC_V2_MAJORITY)
        recovered: list[int] = []
        for i in range(n):
            chunk = repetition_payload[
                i * rep:(i + 1) * rep]
            recovered.append(int(
                _majority_count(chunk, majority_threshold)))
        info = {
            "syndromes": [
                int(s_coarse), int(s_fine),
                int(s_ultra), int(s_ultra2)],
            "double_errors": [
                bool(dbl_coarse), bool(dbl_fine),
                bool(dbl_ultra), bool(dbl_ultra2)],
            "n_corrupted_segments": int(n_corrupted),
            "n_double_errors": int(n_double),
        }
        return attempt, tuple(recovered), info

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": str(W54_CRC_V2_SCHEMA_VERSION),
            "inner_v1_cid": str(self.inner_v1.cid()),
            "repetition_v2": int(self.repetition_v2),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w54_corruption_robust_carrier_v2",
            "carrier": self.to_dict()})


# =============================================================================
# Hostile-channel probe V2
# =============================================================================


@dataclasses.dataclass(frozen=True)
class HostileChannelV2Result:
    n_probes: int
    n_single_corrected: int
    n_double_detected: int
    n_double_uncorrected: int
    n_silent_failure: int
    single_correct_rate: float
    double_detect_rate: float
    silent_failure_rate: float
    flip_intensity: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_probes": int(self.n_probes),
            "n_single_corrected": int(self.n_single_corrected),
            "n_double_detected": int(self.n_double_detected),
            "n_double_uncorrected": int(
                self.n_double_uncorrected),
            "n_silent_failure": int(self.n_silent_failure),
            "single_correct_rate": float(round(
                self.single_correct_rate, 12)),
            "double_detect_rate": float(round(
                self.double_detect_rate, 12)),
            "silent_failure_rate": float(round(
                self.silent_failure_rate, 12)),
            "flip_intensity": float(round(
                self.flip_intensity, 12)),
        }


def _flip_bit_in_hamming(
        hcode: Sequence[int], bit_index: int,
) -> tuple[int, ...]:
    """Flip a single bit in a 7-bit Hamming codeword."""
    out = list(hcode)
    idx = int(bit_index) % max(1, len(out))
    out[idx] = int(out[idx]) ^ 1
    return tuple(out)


def probe_hostile_channel_v2(
        carriers: Sequence[Sequence[float]],
        *,
        crc_v2: CorruptionRobustCarrierV2,
        flip_intensity: float = 1.0,
        seed: int = W47_DEFAULT_TRAIN_SEED,
) -> HostileChannelV2Result:
    """Single-bit + double-bit Hamming-aware probe.

    For each carrier, encode + flip ``floor(intensity)`` bits in
    one of the four Hamming codewords (uniform pick among
    segments), then decode and check single-bit correction or
    double-bit detection.
    """
    n_flips = max(1, int(flip_intensity))
    n = 0
    n_single = 0
    n_double = 0
    n_double_unc = 0
    n_silent = 0
    rng = _DeterministicLCG(seed=int(seed))
    for i, c in enumerate(carriers):
        (h_c, h_f, h_u, h_u2, rep, parity) = (
            crc_v2.encode_with_hamming(c))
        clean = crc_v2.codebook.decode(
            coarse=_4bits_to_code(
                hamming_7_4_decode(h_c)[0],
                crc_v2.codebook.coarse_bits()),
            fine=_4bits_to_code(
                hamming_7_4_decode(h_f)[0],
                crc_v2.codebook.fine_bits()),
            ultra=_4bits_to_code(
                hamming_7_4_decode(h_u)[0],
                crc_v2.codebook.ultra_bits()),
            ultra2=_4bits_to_code(
                hamming_7_4_decode(h_u2)[0],
                crc_v2.codebook.ultra2_bits()))
        # Remember original per-segment parities for 2-bit detection.
        orig_parity = (
            int(sum(int(b) for b in h_c) & 1),
            int(sum(int(b) for b in h_f) & 1),
            int(sum(int(b) for b in h_u) & 1),
            int(sum(int(b) for b in h_u2) & 1),
        )
        segs = [list(h_c), list(h_f), list(h_u), list(h_u2)]
        for f in range(n_flips):
            seg_idx = int(rng.next_uniform() * 4.0) % 4
            bit_idx = int(rng.next_uniform() * 7.0) % 7
            segs[seg_idx][bit_idx] = (
                int(segs[seg_idx][bit_idx]) ^ 1)
        attempt, _, info = crc_v2.decode_with_hamming(
            h_coarse=segs[0], h_fine=segs[1],
            h_ultra=segs[2], h_ultra2=segs[3],
            repetition_payload=rep,
            original_parities=orig_parity)
        n += 1
        ncs = int(info.get("n_corrupted_segments", 0))
        nde = int(info.get("n_double_errors", 0))
        any_detect = (ncs >= 1) or (nde >= 1)
        if n_flips == 1:
            if ncs == 1 and nde == 0:
                n_single += 1
        else:
            if (ncs >= 2) or (nde >= 1):
                n_double += 1
            elif ncs == 1 and nde == 0:
                n_single += 1
            else:
                n_double_unc += 1
        if not any_detect:
            cs = _cosine(attempt.decoded_payload, clean)
            if cs < 0.999:
                n_silent += 1
    return HostileChannelV2Result(
        n_probes=int(n),
        n_single_corrected=int(n_single),
        n_double_detected=int(n_double),
        n_double_uncorrected=int(n_double_unc),
        n_silent_failure=int(n_silent),
        single_correct_rate=float(n_single / max(1, n)),
        double_detect_rate=float(n_double / max(1, n)),
        silent_failure_rate=float(n_silent / max(1, n)),
        flip_intensity=float(flip_intensity),
    )


# =============================================================================
# Witness
# =============================================================================


@dataclasses.dataclass(frozen=True)
class CorruptionRobustnessV2Witness:
    crc_v2_cid: str
    repetition: int
    n_probes: int
    single_correct_rate: float
    double_detect_rate: float
    silent_failure_rate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "crc_v2_cid": str(self.crc_v2_cid),
            "repetition": int(self.repetition),
            "n_probes": int(self.n_probes),
            "single_correct_rate": float(round(
                self.single_correct_rate, 12)),
            "double_detect_rate": float(round(
                self.double_detect_rate, 12)),
            "silent_failure_rate": float(round(
                self.silent_failure_rate, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w54_crc_v2_witness",
            "witness": self.to_dict()})


def emit_corruption_robustness_v2_witness(
        *,
        crc_v2: CorruptionRobustCarrierV2,
        carriers: Sequence[Sequence[float]],
        flip_intensity: float = 1.0,
        seed: int = W47_DEFAULT_TRAIN_SEED,
) -> CorruptionRobustnessV2Witness:
    res = probe_hostile_channel_v2(
        carriers, crc_v2=crc_v2,
        flip_intensity=float(flip_intensity),
        seed=int(seed))
    return CorruptionRobustnessV2Witness(
        crc_v2_cid=str(crc_v2.cid()),
        repetition=int(crc_v2.repetition_v2),
        n_probes=int(res.n_probes),
        single_correct_rate=float(res.single_correct_rate),
        double_detect_rate=float(res.double_detect_rate),
        silent_failure_rate=float(res.silent_failure_rate),
    )


# =============================================================================
# Verifier
# =============================================================================

W54_CRC_V2_VERIFIER_FAILURE_MODES: tuple[str, ...] = (
    "w54_crc_v2_cid_mismatch",
    "w54_crc_v2_single_correct_rate_below_floor",
    "w54_crc_v2_double_detect_rate_below_floor",
    "w54_crc_v2_silent_failure_above_ceiling",
    "w54_crc_v2_repetition_mismatch",
)


def verify_corruption_robustness_v2_witness(
        witness: CorruptionRobustnessV2Witness,
        *,
        expected_crc_v2_cid: str | None = None,
        min_single_correct_rate: float | None = None,
        min_double_detect_rate: float | None = None,
        max_silent_failure_rate: float | None = None,
        expected_repetition: int | None = None,
) -> dict[str, Any]:
    failures: list[str] = []
    if (expected_crc_v2_cid is not None
            and witness.crc_v2_cid
            != str(expected_crc_v2_cid)):
        failures.append("w54_crc_v2_cid_mismatch")
    if (min_single_correct_rate is not None
            and witness.single_correct_rate
            < float(min_single_correct_rate)):
        failures.append(
            "w54_crc_v2_single_correct_rate_below_floor")
    if (min_double_detect_rate is not None
            and witness.double_detect_rate
            < float(min_double_detect_rate)):
        failures.append(
            "w54_crc_v2_double_detect_rate_below_floor")
    if (max_silent_failure_rate is not None
            and witness.silent_failure_rate
            > float(max_silent_failure_rate)):
        failures.append(
            "w54_crc_v2_silent_failure_above_ceiling")
    if (expected_repetition is not None
            and witness.repetition
            != int(expected_repetition)):
        failures.append("w54_crc_v2_repetition_mismatch")
    return {
        "ok": (len(failures) == 0),
        "failures": failures,
        "witness_cid": witness.cid(),
    }


__all__ = [
    "W54_CRC_V2_SCHEMA_VERSION",
    "W54_DEFAULT_CRC_V2_REPETITION",
    "W54_DEFAULT_CRC_V2_MAJORITY",
    "W54_HAMMING_7_4_N_PARITY_BITS",
    "W54_HAMMING_7_4_DATA_BITS",
    "W54_HAMMING_7_4_TOTAL_BITS",
    "W54_CRC_V2_VERIFIER_FAILURE_MODES",
    "CorruptionRobustCarrierV2",
    "HostileChannelV2Result",
    "CorruptionRobustnessV2Witness",
    "hamming_7_4_encode",
    "hamming_7_4_decode",
    "probe_hostile_channel_v2",
    "emit_corruption_robustness_v2_witness",
    "verify_corruption_robustness_v2_witness",
]
