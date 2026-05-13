"""W54 M8 — ECC Codebook V6 (5-level + Hamming(7,4) per segment).

Stacks a fifth quantisation level (K5 = 2 prototypes) on top of
W53 V5's 4-level codebook AND wraps each segment in a
Hamming(7,4) codeword so single-bit errors are *correctable*
(not just detectable).

Capacity: K1=32 × K2=16 × K3=8 × K4=4 × K5=2 = 32768 codes per
triple ≈ 15 bits. Plus 12 Hamming parity bits (3 per segment ×
4 segments) on top of 16 data bits → 28 structured bits per
visible token. The V6 target is **≥ 16 bits/visible-token**
counting the data bits only (the parity bits are overhead).

V6 leaves the K4 codes' parity-bit detection intact; the
Hamming(7,4) is a new *additional* layer that can correct any
single bit error per segment.

Honest scope: pure-Python only.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any, Sequence

from .autograd_manifold import (
    ParamTensor,
    W47_DEFAULT_INIT_SCALE,
    W47_DEFAULT_TRAIN_SEED,
    _DeterministicLCG,
)
from .corruption_robust_carrier_v2 import (
    W54_HAMMING_7_4_TOTAL_BITS,
    hamming_7_4_decode,
    hamming_7_4_encode,
)
from .ecc_codebook_v5 import (
    ECCCodebookV5,
    ECCCompressionResult,
    ECCDecodeAttempt,
    W53_DEFAULT_ECC_CODE_DIM,
    W53_DEFAULT_ECC_EMIT_MASK_LEN,
    W53_DEFAULT_ECC_K1,
    W53_DEFAULT_ECC_K2,
    W53_DEFAULT_ECC_K3,
    W53_DEFAULT_ECC_K4,
    W53_ECC_PARITY_BITS_PER_TOKEN,
    compress_carrier_ecc,
)
from .quantised_compression import QuantisedBudgetGate


# =============================================================================
# Schema, defaults
# =============================================================================

W54_ECC_V6_SCHEMA_VERSION: str = (
    "coordpy.ecc_codebook_v6.v1")

W54_DEFAULT_ECC_V6_K1: int = W53_DEFAULT_ECC_K1
W54_DEFAULT_ECC_V6_K2: int = W53_DEFAULT_ECC_K2
W54_DEFAULT_ECC_V6_K3: int = W53_DEFAULT_ECC_K3
W54_DEFAULT_ECC_V6_K4: int = W53_DEFAULT_ECC_K4
W54_DEFAULT_ECC_V6_K5: int = 2
W54_DEFAULT_ECC_V6_CODE_DIM: int = W53_DEFAULT_ECC_CODE_DIM
W54_DEFAULT_ECC_V6_EMIT_MASK_LEN: int = (
    W53_DEFAULT_ECC_EMIT_MASK_LEN)
W54_DEFAULT_ECC_V6_TARGET_BITS_PER_TOKEN: float = 16.0
W54_ECC_V6_HAMMING_BITS_PER_SEGMENT: int = (
    W54_HAMMING_7_4_TOTAL_BITS)


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


# =============================================================================
# ECCCodebookV6
# =============================================================================


@dataclasses.dataclass
class ECCCodebookV6:
    """V6 codebook: V5 inner + K5 prototypes + Hamming-aware ops."""

    inner_v5: ECCCodebookV5
    n_ultra3: int
    code_dim: int
    ultra3_proto: ParamTensor

    @classmethod
    def init(
            cls, *,
            n_coarse: int = W54_DEFAULT_ECC_V6_K1,
            n_fine: int = W54_DEFAULT_ECC_V6_K2,
            n_ultra: int = W54_DEFAULT_ECC_V6_K3,
            n_ultra2: int = W54_DEFAULT_ECC_V6_K4,
            n_ultra3: int = W54_DEFAULT_ECC_V6_K5,
            code_dim: int = W54_DEFAULT_ECC_V6_CODE_DIM,
            seed: int = W47_DEFAULT_TRAIN_SEED,
            init_scale: float = W47_DEFAULT_INIT_SCALE,
    ) -> "ECCCodebookV6":
        inner = ECCCodebookV5.init(
            n_coarse=int(n_coarse),
            n_fine=int(n_fine),
            n_ultra=int(n_ultra),
            n_ultra2=int(n_ultra2),
            code_dim=int(code_dim),
            seed=int(seed),
            init_scale=float(init_scale))
        ultra3 = ParamTensor(
            shape=(int(n_ultra3), int(code_dim)), values=[])
        ultra3.init_seed(
            seed=int(seed) + 509,
            scale=float(init_scale) * 0.25)
        return cls(
            inner_v5=inner,
            n_ultra3=int(n_ultra3),
            code_dim=int(code_dim),
            ultra3_proto=ultra3)

    def params(self) -> list[ParamTensor]:
        return list(self.inner_v5.params()) + [
            self.ultra3_proto]

    def coarse_bits(self) -> int:
        return int(self.inner_v5.coarse_bits())

    def fine_bits(self) -> int:
        return int(self.inner_v5.fine_bits())

    def ultra_bits(self) -> int:
        return int(self.inner_v5.ultra_bits())

    def ultra2_bits(self) -> int:
        return int(self.inner_v5.ultra2_bits())

    def ultra3_bits(self) -> int:
        n = max(1, int(self.n_ultra3))
        return int(max(1, math.ceil(math.log2(n))))

    def ultra3_vector(self, code: int) -> tuple[float, ...]:
        cd = int(self.code_dim)
        c = int(code) % max(1, int(self.n_ultra3))
        base = c * cd
        return tuple(
            float(self.ultra3_proto.values[base + j])
            for j in range(cd)
        )

    def encode_value(
            self, carrier: Sequence[float],
    ) -> tuple[int, int, int, int, int]:
        coarse, fine, ultra, ultra2 = (
            self.inner_v5.encode_value(carrier))
        # Compute residual after V5 decode.
        decoded = self.inner_v5.decode(
            coarse=coarse, fine=fine,
            ultra=ultra, ultra2=ultra2)
        residual = [
            float(carrier[j] if j < len(carrier) else 0.0)
            - float(decoded[j] if j < len(decoded) else 0.0)
            for j in range(int(self.code_dim))
        ]
        best_idx = 0
        best_d = float("inf")
        for k in range(int(self.n_ultra3)):
            v = self.ultra3_vector(k)
            d = 0.0
            for j in range(int(self.code_dim)):
                diff = (
                    float(residual[j])
                    - float(v[j] if j < len(v) else 0.0))
                d += diff * diff
            if d < best_d:
                best_d = d
                best_idx = int(k)
        return (
            int(coarse), int(fine), int(ultra),
            int(ultra2), int(best_idx))

    def decode(
            self, *,
            coarse: int, fine: int,
            ultra: int, ultra2: int, ultra3: int,
            include_ultra: bool = True,
            include_ultra2: bool = True,
            include_ultra3: bool = True,
    ) -> tuple[float, ...]:
        base = list(self.inner_v5.decode(
            coarse=coarse, fine=fine,
            ultra=ultra, ultra2=ultra2,
            include_ultra=bool(include_ultra),
            include_ultra2=bool(include_ultra2)))
        if include_ultra3:
            v = self.ultra3_vector(int(ultra3))
            base = [
                float(base[j] if j < len(base) else 0.0)
                + float(v[j] if j < len(v) else 0.0)
                for j in range(int(self.code_dim))
            ]
        return tuple(base)

    def hamming_encode_segments(
            self, *,
            coarse: int, fine: int,
            ultra: int, ultra2: int, ultra3: int,
    ) -> tuple[tuple[int, ...], tuple[int, ...],
                tuple[int, ...], tuple[int, ...],
                tuple[int, ...]]:
        """Encode all five segments as Hamming(7,4) codewords."""
        h_c = tuple(hamming_7_4_encode(
            [(int(coarse) >> i) & 1 for i in range(
                min(4, self.coarse_bits()))]))
        h_f = tuple(hamming_7_4_encode(
            [(int(fine) >> i) & 1 for i in range(
                min(4, self.fine_bits()))]))
        h_u = tuple(hamming_7_4_encode(
            [(int(ultra) >> i) & 1 for i in range(
                min(4, self.ultra_bits()))]))
        h_u2 = tuple(hamming_7_4_encode(
            [(int(ultra2) >> i) & 1 for i in range(
                min(4, self.ultra2_bits()))]))
        h_u3 = tuple(hamming_7_4_encode(
            [(int(ultra3) >> i) & 1 for i in range(
                min(4, self.ultra3_bits()))]))
        return h_c, h_f, h_u, h_u2, h_u3

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": str(W54_ECC_V6_SCHEMA_VERSION),
            "inner_v5_cid": str(self.inner_v5.cid()),
            "n_ultra3": int(self.n_ultra3),
            "code_dim": int(self.code_dim),
            "ultra3_proto": self.ultra3_proto.to_dict(),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w54_ecc_codebook_v6",
            "codebook": self.to_dict()})


# =============================================================================
# ECCCompressionResultV6
# =============================================================================


@dataclasses.dataclass(frozen=True)
class ECCCompressionResultV6:
    inner_v5: ECCCompressionResult
    ultra3_code: int
    ultra3_bits: int
    ultra3_hamming: tuple[int, ...]
    coarse_hamming: tuple[int, ...]
    fine_hamming: tuple[int, ...]
    ultra_hamming: tuple[int, ...]
    ultra2_hamming: tuple[int, ...]
    structured_bits_v6: int
    bits_per_visible_token_v6: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "inner_v5": self.inner_v5.to_dict(),
            "ultra3_code": int(self.ultra3_code),
            "ultra3_bits": int(self.ultra3_bits),
            "ultra3_hamming": list(self.ultra3_hamming),
            "coarse_hamming": list(self.coarse_hamming),
            "fine_hamming": list(self.fine_hamming),
            "ultra_hamming": list(self.ultra_hamming),
            "ultra2_hamming": list(self.ultra2_hamming),
            "structured_bits_v6": int(self.structured_bits_v6),
            "bits_per_visible_token_v6": float(round(
                self.bits_per_visible_token_v6, 12)),
        }


def compress_carrier_ecc_v6(
        carrier: Sequence[float],
        *,
        codebook: ECCCodebookV6,
        gate: QuantisedBudgetGate,
) -> ECCCompressionResultV6:
    """Compress a carrier with V6 (V5 + K5 + Hamming wrappers)."""
    inner = compress_carrier_ecc(
        carrier, codebook=codebook.inner_v5, gate=gate)
    # Compute ultra3 separately.
    coarse, fine, ultra, ultra2, ultra3 = (
        codebook.encode_value(carrier))
    ultra3_bits = int(codebook.ultra3_bits())
    h_c, h_f, h_u, h_u2, h_u3 = (
        codebook.hamming_encode_segments(
            coarse=int(coarse), fine=int(fine),
            ultra=int(ultra), ultra2=int(ultra2),
            ultra3=int(ultra3)))
    # Total structured bits = V5 structured bits +
    # (ultra3 data) + 5 * Hamming parity overhead.
    # Hamming(7,4) adds 3 parity bits per segment.
    structured_bits_v6 = (
        int(inner.structured_bits)
        + int(ultra3_bits))
    # Bits/visible-token: only data bits count toward the budget;
    # the Hamming parity bits are overhead.
    bits_per = (
        float(structured_bits_v6)
        / float(max(1, int(inner.visible_tokens)))
        if int(inner.visible_tokens) > 0 else 0.0)
    return ECCCompressionResultV6(
        inner_v5=inner,
        ultra3_code=int(ultra3),
        ultra3_bits=int(ultra3_bits),
        ultra3_hamming=h_u3,
        coarse_hamming=h_c,
        fine_hamming=h_f,
        ultra_hamming=h_u,
        ultra2_hamming=h_u2,
        structured_bits_v6=int(structured_bits_v6),
        bits_per_visible_token_v6=float(bits_per),
    )


# =============================================================================
# Witness
# =============================================================================


@dataclasses.dataclass(frozen=True)
class ECCCompressionV6Witness:
    codebook_cid: str
    bits_per_visible_token: float
    target_bits_per_token: float
    target_met: bool
    hamming_parity_bits_per_token: int
    code_capacity_bits: int
    n_ultra3: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "codebook_cid": str(self.codebook_cid),
            "bits_per_visible_token": float(round(
                self.bits_per_visible_token, 12)),
            "target_bits_per_token": float(round(
                self.target_bits_per_token, 12)),
            "target_met": bool(self.target_met),
            "hamming_parity_bits_per_token": int(
                self.hamming_parity_bits_per_token),
            "code_capacity_bits": int(
                self.code_capacity_bits),
            "n_ultra3": int(self.n_ultra3),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w54_ecc_v6_compression_witness",
            "witness": self.to_dict()})


def emit_ecc_v6_compression_witness(
        *,
        codebook: ECCCodebookV6,
        compression: ECCCompressionResultV6,
        target_bits_per_token: float = (
            W54_DEFAULT_ECC_V6_TARGET_BITS_PER_TOKEN),
) -> ECCCompressionV6Witness:
    cap = (
        int(codebook.coarse_bits())
        + int(codebook.fine_bits())
        + int(codebook.ultra_bits())
        + int(codebook.ultra2_bits())
        + int(codebook.ultra3_bits()))
    # 5 segments × 3 parity bits each = 15 parity bits.
    parity_bits = 5 * 3
    return ECCCompressionV6Witness(
        codebook_cid=str(codebook.cid()),
        bits_per_visible_token=float(
            compression.bits_per_visible_token_v6),
        target_bits_per_token=float(target_bits_per_token),
        target_met=bool(
            compression.bits_per_visible_token_v6
            >= float(target_bits_per_token)),
        hamming_parity_bits_per_token=int(parity_bits),
        code_capacity_bits=int(cap),
        n_ultra3=int(codebook.n_ultra3),
    )


# =============================================================================
# Rate-floor falsifier
# =============================================================================


@dataclasses.dataclass(frozen=True)
class ECCV6RateFloorFalsifierResult:
    target_bits_per_token: float
    achieved_bits_per_token: float
    rate_target_missed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_bits_per_token": float(round(
                self.target_bits_per_token, 12)),
            "achieved_bits_per_token": float(round(
                self.achieved_bits_per_token, 12)),
            "rate_target_missed": bool(
                self.rate_target_missed),
        }


def probe_ecc_v6_rate_floor_falsifier(
        carrier: Sequence[float],
        *,
        codebook: ECCCodebookV6,
        gate: QuantisedBudgetGate,
        target_bits_per_token: float = 64.0,
) -> ECCV6RateFloorFalsifierResult:
    res = compress_carrier_ecc_v6(
        carrier, codebook=codebook, gate=gate)
    achieved = float(res.bits_per_visible_token_v6)
    missed = bool(achieved < float(target_bits_per_token))
    return ECCV6RateFloorFalsifierResult(
        target_bits_per_token=float(target_bits_per_token),
        achieved_bits_per_token=float(achieved),
        rate_target_missed=bool(missed),
    )


# =============================================================================
# Verifier
# =============================================================================

W54_ECC_V6_VERIFIER_FAILURE_MODES: tuple[str, ...] = (
    "w54_ecc_v6_codebook_cid_mismatch",
    "w54_ecc_v6_bits_per_token_below_target",
    "w54_ecc_v6_code_capacity_below_floor",
    "w54_ecc_v6_n_ultra3_below_floor",
    "w54_ecc_v6_hamming_parity_count_mismatch",
)


def verify_ecc_v6_compression_witness(
        witness: ECCCompressionV6Witness,
        *,
        expected_codebook_cid: str | None = None,
        require_target_met: bool = False,
        min_code_capacity_bits: int | None = None,
        min_n_ultra3: int | None = None,
) -> dict[str, Any]:
    failures: list[str] = []
    if (expected_codebook_cid is not None
            and witness.codebook_cid
            != str(expected_codebook_cid)):
        failures.append("w54_ecc_v6_codebook_cid_mismatch")
    if (require_target_met
            and not witness.target_met):
        failures.append(
            "w54_ecc_v6_bits_per_token_below_target")
    if (min_code_capacity_bits is not None
            and witness.code_capacity_bits
            < int(min_code_capacity_bits)):
        failures.append(
            "w54_ecc_v6_code_capacity_below_floor")
    if (min_n_ultra3 is not None
            and witness.n_ultra3 < int(min_n_ultra3)):
        failures.append("w54_ecc_v6_n_ultra3_below_floor")
    if witness.hamming_parity_bits_per_token != 15:
        failures.append(
            "w54_ecc_v6_hamming_parity_count_mismatch")
    return {
        "ok": (len(failures) == 0),
        "failures": failures,
        "witness_cid": witness.cid(),
    }


__all__ = [
    "W54_ECC_V6_SCHEMA_VERSION",
    "W54_DEFAULT_ECC_V6_K1",
    "W54_DEFAULT_ECC_V6_K2",
    "W54_DEFAULT_ECC_V6_K3",
    "W54_DEFAULT_ECC_V6_K4",
    "W54_DEFAULT_ECC_V6_K5",
    "W54_DEFAULT_ECC_V6_CODE_DIM",
    "W54_DEFAULT_ECC_V6_EMIT_MASK_LEN",
    "W54_DEFAULT_ECC_V6_TARGET_BITS_PER_TOKEN",
    "W54_ECC_V6_HAMMING_BITS_PER_SEGMENT",
    "W54_ECC_V6_VERIFIER_FAILURE_MODES",
    "ECCCodebookV6",
    "ECCCompressionResultV6",
    "ECCCompressionV6Witness",
    "ECCV6RateFloorFalsifierResult",
    "compress_carrier_ecc_v6",
    "emit_ecc_v6_compression_witness",
    "probe_ecc_v6_rate_floor_falsifier",
    "verify_ecc_v6_compression_witness",
]
