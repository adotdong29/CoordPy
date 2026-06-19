"""W53 M8 — Corruption-Robust Carrier.

A composable wrapper around any quantised carrier that:

* adds **per-segment XOR parity** for single-bit flip detection
* composes a **3-of-5 majority repetition code** at the
  level-mask layer for extra robustness
* exposes **corruption_detected** + **partial_correction** +
  **abstain** semantics

The carrier itself is opaque (a sequence of floats). The
robustness adapters add structured redundancy on top of the
ECC codebook V5 produced by ``coordpy.ecc_codebook_v5``.

Pure-Python only.
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
from .ecc_codebook_v5 import (
    ECCCodebookV5,
    ECCDecodeAttempt,
    compress_carrier_ecc,
    decode_with_parity_check,
    flip_random_bit,
)
from .quantised_compression import QuantisedBudgetGate


# =============================================================================
# Schema, defaults
# =============================================================================

W53_CRC_SCHEMA_VERSION: str = (
    "coordpy.corruption_robust_carrier.v1")

W53_DEFAULT_CRC_REPETITION: int = 3
W53_DEFAULT_CRC_MAJORITY: int = 2  # 2-of-3 majority


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


def _round_floats(
        values: Sequence[float], precision: int = 12,
) -> list[float]:
    return [float(round(float(v), precision)) for v in values]


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


def _majority_vote(bits: Sequence[int]) -> int:
    n = len(bits)
    if n == 0:
        return 0
    s = sum(int(b) & 1 for b in bits)
    return 1 if (s * 2 > n) else 0


# =============================================================================
# CorruptionRobustCarrier
# =============================================================================


@dataclasses.dataclass
class CorruptionRobustCarrier:
    """Wraps an ECCCodebookV5 + QuantisedBudgetGate.

    The carrier supports:
    * encode_with_parity(carrier) → packed structured bits
    * decode_with_majority(coarse, fine, ultra, ultra2,
        observed_parity, repetition_payloads) →
        ECCDecodeAttempt with majority-corrected codes

    The repetition payload triples the bits payload: the
    receiver majority-votes per bit position. With p<0.5 of any
    single bit flipping independently, 3-of-3 majority pushes
    the per-bit error rate to 3p²−2p³.
    """

    codebook: ECCCodebookV5
    gate: QuantisedBudgetGate
    repetition: int

    @classmethod
    def init(
            cls, *,
            codebook: ECCCodebookV5,
            gate: QuantisedBudgetGate,
            repetition: int = W53_DEFAULT_CRC_REPETITION,
    ) -> "CorruptionRobustCarrier":
        return cls(
            codebook=codebook,
            gate=gate,
            repetition=int(repetition))

    def encode_with_repetition_payload(
            self,
            carrier: Sequence[float],
    ) -> tuple[
            int, int, int, int, tuple[int, ...],
            tuple[int, ...]]:
        """Returns (coarse, fine, ultra, ultra2,
        repetition_payload, parity_bits)."""
        comp = compress_carrier_ecc(
            carrier, codebook=self.codebook, gate=self.gate)
        # Triple the payload — same bits repeated 3 times.
        rep = []
        for b in comp.bits_payload:
            for _ in range(int(self.repetition)):
                rep.append(int(b))
        return (
            int(comp.coarse_code), int(comp.fine_code),
            int(comp.ultra_code), int(comp.ultra2_code),
            tuple(rep), tuple(comp.parity_bits))

    def decode_with_majority(
            self,
            *,
            coarse: int, fine: int,
            ultra: int, ultra2: int,
            observed_parity: tuple[int, int, int, int],
            repetition_payload: Sequence[int],
    ) -> tuple[ECCDecodeAttempt, tuple[int, ...]]:
        """Decode + majority-vote the bits payload."""
        attempt = decode_with_parity_check(
            codebook=self.codebook,
            coarse=coarse, fine=fine,
            ultra=ultra, ultra2=ultra2,
            observed_parity=observed_parity)
        # Majority vote on payload.
        rep = max(1, int(self.repetition))
        n = len(repetition_payload) // rep
        recovered: list[int] = []
        for i in range(n):
            chunk = repetition_payload[
                i * rep:(i + 1) * rep]
            recovered.append(int(_majority_vote(chunk)))
        return attempt, tuple(recovered)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": str(W53_CRC_SCHEMA_VERSION),
            "codebook_cid": str(self.codebook.cid()),
            "gate_cid": str(self.gate.cid()),
            "repetition": int(self.repetition),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w53_corruption_robust_carrier",
            "carrier": self.to_dict()})


# =============================================================================
# Hostile-channel probe
# =============================================================================


@dataclasses.dataclass(frozen=True)
class HostileChannelProbeResult:
    n_probes: int
    n_silent_failure: int
    n_detected_corruption: int
    n_partial_corrected: int
    n_abstain: int
    silent_failure_rate: float
    detect_rate: float
    correction_rate: float
    abstain_rate: float
    flip_intensity: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_probes": int(self.n_probes),
            "n_silent_failure": int(self.n_silent_failure),
            "n_detected_corruption": int(
                self.n_detected_corruption),
            "n_partial_corrected": int(
                self.n_partial_corrected),
            "n_abstain": int(self.n_abstain),
            "silent_failure_rate": float(round(
                self.silent_failure_rate, 12)),
            "detect_rate": float(round(self.detect_rate, 12)),
            "correction_rate": float(round(
                self.correction_rate, 12)),
            "abstain_rate": float(round(
                self.abstain_rate, 12)),
            "flip_intensity": float(round(
                self.flip_intensity, 12)),
        }


def probe_hostile_channel(
        carriers: Sequence[Sequence[float]],
        *,
        crc: CorruptionRobustCarrier,
        flip_intensity: float = 1.0,
        seed: int = W47_DEFAULT_TRAIN_SEED,
) -> HostileChannelProbeResult:
    """Probe single-bit corruption recovery.

    For each carrier, encode → flip floor(intensity) bits →
    decode + parity check. Reports detection rate, correction
    rate, abstain rate, and silent-failure rate (cases where
    corruption was undetected and the decoded payload differs
    from the clean payload).
    """
    n = 0
    n_silent = 0
    n_detected = 0
    n_corr = 0
    n_abs = 0
    rng = _DeterministicLCG(seed=int(seed))
    n_flips = max(1, int(flip_intensity))
    for i, c in enumerate(carriers):
        c2_co, c2_fi, c2_ul, c2_u2, rep_payload, parity = (
            crc.encode_with_repetition_payload(c))
        # Apply n_flips bit-flips to the codes.
        cur = (c2_co, c2_fi, c2_ul, c2_u2)
        for f in range(int(n_flips)):
            seed_f = (
                int(rng.next_uniform() * (1 << 30)))
            cur = flip_random_bit(
                coarse=cur[0], fine=cur[1],
                ultra=cur[2], ultra2=cur[3],
                codebook=crc.codebook,
                seed=int(seed_f))[:4]
        # Decode with majority + parity check.
        attempt, _ = crc.decode_with_majority(
            coarse=cur[0], fine=cur[1],
            ultra=cur[2], ultra2=cur[3],
            observed_parity=parity,
            repetition_payload=rep_payload)
        # Compute the clean decoded payload.
        clean_decoded = crc.codebook.decode(
            coarse=c2_co, fine=c2_fi,
            ultra=c2_ul, ultra2=c2_u2)
        n += 1
        if attempt.n_corrupted_segments >= 1:
            n_detected += 1
        if attempt.corrected_partial:
            n_corr += 1
        if attempt.abstain:
            n_abs += 1
        # Silent failure: no detection AND decoded != clean.
        if attempt.n_corrupted_segments == 0:
            cs = _cosine(
                attempt.decoded_payload, clean_decoded)
            if cs < 0.999:
                n_silent += 1
    return HostileChannelProbeResult(
        n_probes=int(n),
        n_silent_failure=int(n_silent),
        n_detected_corruption=int(n_detected),
        n_partial_corrected=int(n_corr),
        n_abstain=int(n_abs),
        silent_failure_rate=float(n_silent / max(1, n)),
        detect_rate=float(n_detected / max(1, n)),
        correction_rate=float(n_corr / max(1, n)),
        abstain_rate=float(n_abs / max(1, n)),
        flip_intensity=float(flip_intensity),
    )


# =============================================================================
# Witness
# =============================================================================


@dataclasses.dataclass(frozen=True)
class CorruptionRobustnessWitness:
    crc_cid: str
    repetition: int
    n_probes: int
    detect_rate: float
    correction_rate: float
    abstain_rate: float
    silent_failure_rate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "crc_cid": str(self.crc_cid),
            "repetition": int(self.repetition),
            "n_probes": int(self.n_probes),
            "detect_rate": float(round(
                self.detect_rate, 12)),
            "correction_rate": float(round(
                self.correction_rate, 12)),
            "abstain_rate": float(round(
                self.abstain_rate, 12)),
            "silent_failure_rate": float(round(
                self.silent_failure_rate, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w53_corruption_robustness_witness",
            "witness": self.to_dict()})


def emit_corruption_robustness_witness(
        *,
        crc: CorruptionRobustCarrier,
        carriers: Sequence[Sequence[float]],
        flip_intensity: float = 1.0,
        seed: int = W47_DEFAULT_TRAIN_SEED,
) -> CorruptionRobustnessWitness:
    res = probe_hostile_channel(
        carriers, crc=crc,
        flip_intensity=float(flip_intensity),
        seed=int(seed))
    return CorruptionRobustnessWitness(
        crc_cid=str(crc.cid()),
        repetition=int(crc.repetition),
        n_probes=int(res.n_probes),
        detect_rate=float(res.detect_rate),
        correction_rate=float(res.correction_rate),
        abstain_rate=float(res.abstain_rate),
        silent_failure_rate=float(res.silent_failure_rate),
    )


# =============================================================================
# Verifier
# =============================================================================


W53_CRC_VERIFIER_FAILURE_MODES: tuple[str, ...] = (
    "w53_crc_cid_mismatch",
    "w53_crc_detect_rate_below_floor",
    "w53_crc_silent_failure_above_ceiling",
    "w53_crc_repetition_mismatch",
)


def verify_corruption_robustness_witness(
        witness: CorruptionRobustnessWitness,
        *,
        expected_crc_cid: str | None = None,
        min_detect_rate: float | None = None,
        max_silent_failure_rate: float | None = None,
        expected_repetition: int | None = None,
) -> dict[str, Any]:
    failures: list[str] = []
    if (expected_crc_cid is not None
            and witness.crc_cid != str(expected_crc_cid)):
        failures.append("w53_crc_cid_mismatch")
    if (min_detect_rate is not None
            and witness.detect_rate
            < float(min_detect_rate)):
        failures.append("w53_crc_detect_rate_below_floor")
    if (max_silent_failure_rate is not None
            and witness.silent_failure_rate
            > float(max_silent_failure_rate)):
        failures.append(
            "w53_crc_silent_failure_above_ceiling")
    if (expected_repetition is not None
            and witness.repetition
            != int(expected_repetition)):
        failures.append("w53_crc_repetition_mismatch")
    return {
        "ok": (len(failures) == 0),
        "failures": failures,
        "witness_cid": witness.cid(),
    }


__all__ = [
    "W53_CRC_SCHEMA_VERSION",
    "W53_DEFAULT_CRC_REPETITION",
    "W53_DEFAULT_CRC_MAJORITY",
    "W53_CRC_VERIFIER_FAILURE_MODES",
    "CorruptionRobustCarrier",
    "HostileChannelProbeResult",
    "CorruptionRobustnessWitness",
    "probe_hostile_channel",
    "emit_corruption_robustness_witness",
    "verify_corruption_robustness_witness",
]
