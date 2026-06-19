"""W55 M8 — ECC Codebook V7 (6-level + BCH(15,7) per segment).

Stacks a sixth quantisation level (K6 = 2) on top of W54 V6's
5-level codebook AND replaces V6's Hamming(7,4) with BCH(15,7)
per segment for double-bit *correction* (not just single-bit).

Capacity: K1=32 × K2=16 × K3=8 × K4=4 × K5=2 × K6=2 = 65536
codes per triple ≈ 16 data bits. Plus 8 BCH parity bits/segment
× 5 segments (last segment K6 is a separate 1-bit overhead) =
~32-40 parity bits per triple. The V7 target is ≥ 18 bits/visible-
token counting the data bits.

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
)
from .corruption_robust_carrier_v3 import (
    W55_BCH_15_7_TOTAL_BITS,
    bch_15_7_decode,
    bch_15_7_encode,
)
from .ecc_codebook_v5 import (
    ECCCompressionResult,
    ECCDecodeAttempt,
    W53_DEFAULT_ECC_CODE_DIM,
    W53_DEFAULT_ECC_EMIT_MASK_LEN,
    W53_DEFAULT_ECC_K1,
    W53_DEFAULT_ECC_K2,
    W53_DEFAULT_ECC_K3,
    W53_DEFAULT_ECC_K4,
)
from .ecc_codebook_v6 import (
    ECCCodebookV6,
    ECCCompressionV6Witness,
    W54_DEFAULT_ECC_V6_K1,
    W54_DEFAULT_ECC_V6_K2,
    W54_DEFAULT_ECC_V6_K3,
    W54_DEFAULT_ECC_V6_K4,
    W54_DEFAULT_ECC_V6_K5,
    W54_DEFAULT_ECC_V6_TARGET_BITS_PER_TOKEN,
    compress_carrier_ecc_v6,
)
from .quantised_compression import QuantisedBudgetGate


# =============================================================================
# Schema, defaults
# =============================================================================

W55_ECC_V7_SCHEMA_VERSION: str = (
    "coordpy.ecc_codebook_v7.v1")

W55_DEFAULT_ECC_V7_K1: int = W54_DEFAULT_ECC_V6_K1
W55_DEFAULT_ECC_V7_K2: int = W54_DEFAULT_ECC_V6_K2
W55_DEFAULT_ECC_V7_K3: int = W54_DEFAULT_ECC_V6_K3
W55_DEFAULT_ECC_V7_K4: int = W54_DEFAULT_ECC_V6_K4
W55_DEFAULT_ECC_V7_K5: int = W54_DEFAULT_ECC_V6_K5
W55_DEFAULT_ECC_V7_K6: int = 2
W55_DEFAULT_ECC_V7_CODE_DIM: int = W53_DEFAULT_ECC_CODE_DIM
W55_DEFAULT_ECC_V7_EMIT_MASK_LEN: int = (
    W53_DEFAULT_ECC_EMIT_MASK_LEN)
W55_DEFAULT_ECC_V7_TARGET_BITS_PER_TOKEN: float = 18.0
W55_ECC_V7_BCH_BITS_PER_SEGMENT: int = (
    W55_BCH_15_7_TOTAL_BITS)
W55_ECC_V7_N_SEGMENTS: int = 6


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
# ECCCodebookV7
# =============================================================================


@dataclasses.dataclass
class ECCCodebookV7:
    """V7 codebook: V6 inner + K6 prototypes + BCH(15,7) ops."""

    inner_v6: ECCCodebookV6
    n_ultra4: int
    code_dim: int
    ultra4_proto: ParamTensor

    @classmethod
    def init(
            cls, *,
            n_coarse: int = W55_DEFAULT_ECC_V7_K1,
            n_fine: int = W55_DEFAULT_ECC_V7_K2,
            n_ultra: int = W55_DEFAULT_ECC_V7_K3,
            n_ultra2: int = W55_DEFAULT_ECC_V7_K4,
            n_ultra3: int = W55_DEFAULT_ECC_V7_K5,
            n_ultra4: int = W55_DEFAULT_ECC_V7_K6,
            code_dim: int = W55_DEFAULT_ECC_V7_CODE_DIM,
            seed: int = W47_DEFAULT_TRAIN_SEED,
            init_scale: float = W47_DEFAULT_INIT_SCALE,
    ) -> "ECCCodebookV7":
        inner = ECCCodebookV6.init(
            n_coarse=int(n_coarse),
            n_fine=int(n_fine),
            n_ultra=int(n_ultra),
            n_ultra2=int(n_ultra2),
            n_ultra3=int(n_ultra3),
            code_dim=int(code_dim),
            seed=int(seed),
            init_scale=float(init_scale))
        ultra4 = ParamTensor(
            shape=(int(n_ultra4), int(code_dim)), values=[])
        ultra4.init_seed(
            seed=int(seed) + 619,
            scale=float(init_scale) * 0.125)
        return cls(
            inner_v6=inner,
            n_ultra4=int(n_ultra4),
            code_dim=int(code_dim),
            ultra4_proto=ultra4)

    def params(self) -> list[ParamTensor]:
        return list(self.inner_v6.params()) + [
            self.ultra4_proto]

    def coarse_bits(self) -> int:
        return int(self.inner_v6.coarse_bits())

    def fine_bits(self) -> int:
        return int(self.inner_v6.fine_bits())

    def ultra_bits(self) -> int:
        return int(self.inner_v6.ultra_bits())

    def ultra2_bits(self) -> int:
        return int(self.inner_v6.ultra2_bits())

    def ultra3_bits(self) -> int:
        return int(self.inner_v6.ultra3_bits())

    def ultra4_bits(self) -> int:
        n = max(1, int(self.n_ultra4))
        return int(max(1, math.ceil(math.log2(n))))

    def data_bits_per_triple(self) -> int:
        """Total data bits per (coarse, fine, ultra,
        ultra2, ultra3, ultra4) triple."""
        return int(
            self.coarse_bits()
            + self.fine_bits()
            + self.ultra_bits()
            + self.ultra2_bits()
            + self.ultra3_bits()
            + self.ultra4_bits())

    def bch_parity_bits_per_triple(self) -> int:
        """Parity bits per triple: 8 per segment × 6 segments."""
        return int(8 * W55_ECC_V7_N_SEGMENTS)

    def ultra4_vector(self, code: int) -> tuple[float, ...]:
        cd = int(self.code_dim)
        c = int(code) % max(1, int(self.n_ultra4))
        base = c * cd
        return tuple(
            float(self.ultra4_proto.values[base + j])
            for j in range(cd)
        )

    def encode_value(
            self, carrier: Sequence[float],
    ) -> tuple[int, int, int, int, int, int]:
        coarse, fine, ultra, ultra2, ultra3 = (
            self.inner_v6.encode_value(carrier))
        # Residual after V6 decode.
        decoded = self.inner_v6.decode(
            coarse=coarse, fine=fine,
            ultra=ultra, ultra2=ultra2, ultra3=ultra3)
        residual = [
            float(carrier[j] if j < len(carrier) else 0.0)
            - float(decoded[j] if j < len(decoded) else 0.0)
            for j in range(int(self.code_dim))
        ]
        best_idx = 0
        best_d = float("inf")
        for k in range(int(self.n_ultra4)):
            v = self.ultra4_vector(k)
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
            int(ultra2), int(ultra3), int(best_idx))

    def decode(
            self, *,
            coarse: int, fine: int,
            ultra: int, ultra2: int,
            ultra3: int, ultra4: int,
    ) -> tuple[float, ...]:
        v6_part = self.inner_v6.decode(
            coarse=int(coarse), fine=int(fine),
            ultra=int(ultra), ultra2=int(ultra2),
            ultra3=int(ultra3))
        u4 = self.ultra4_vector(int(ultra4))
        return tuple(
            float(v6_part[j] if j < len(v6_part) else 0.0)
            + float(u4[j] if j < len(u4) else 0.0)
            for j in range(int(self.code_dim))
        )

    def encode_with_bch(
            self, carrier: Sequence[float],
    ) -> tuple[
            tuple[int, int, int, int, int, int],
            list[tuple[int, ...]]]:
        """Encode + wrap each segment with BCH(15,7)."""
        codes = self.encode_value(carrier)
        bch_per_segment: list[tuple[int, ...]] = []
        for code, n_bits in zip(
                codes,
                [self.coarse_bits(),
                 self.fine_bits(),
                 self.ultra_bits(),
                 self.ultra2_bits(),
                 self.ultra3_bits(),
                 self.ultra4_bits()]):
            bits = [(code >> i) & 1 for i in range(n_bits)]
            while len(bits) < 7:
                bits.append(0)
            cw = bch_15_7_encode(bits[:7])
            bch_per_segment.append(cw)
        return codes, bch_per_segment

    def decode_with_bch(
            self,
            bch_per_segment: Sequence[Sequence[int]],
    ) -> tuple[tuple[int, ...], int]:
        """Decode BCH codewords back to (coarse, fine, ultra, ...)."""
        decoded_codes: list[int] = []
        n_corrected = 0
        for seg, n_bits in zip(
                bch_per_segment,
                [self.coarse_bits(),
                 self.fine_bits(),
                 self.ultra_bits(),
                 self.ultra2_bits(),
                 self.ultra3_bits(),
                 self.ultra4_bits()]):
            data, dist, _, _ = bch_15_7_decode(seg)
            if dist > 0:
                n_corrected += 1
            code = 0
            for i in range(int(n_bits)):
                code |= (int(data[i]) & 1) << i
            decoded_codes.append(int(code))
        return tuple(decoded_codes), int(n_corrected)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": str(
                W55_ECC_V7_SCHEMA_VERSION),
            "inner_v6_cid": str(self.inner_v6.cid()),
            "n_ultra4": int(self.n_ultra4),
            "code_dim": int(self.code_dim),
            "ultra4_proto": self.ultra4_proto.to_dict(),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w55_ecc_v7_codebook",
            "codebook": self.to_dict()})


# =============================================================================
# Compression
# =============================================================================


@dataclasses.dataclass(frozen=True)
class ECCCompressionV7Result:
    coarse: int
    fine: int
    ultra: int
    ultra2: int
    ultra3: int
    ultra4: int
    emit_mask_bits: tuple[int, ...]
    bch_per_segment: tuple[tuple[int, ...], ...]
    structured_bits_v7: int
    visible_tokens: int
    bits_per_visible_token_v7: float
    parity_bits_per_triple: int


def compress_carrier_ecc_v7(
        carrier: Sequence[float],
        *,
        codebook: ECCCodebookV7,
        gate: QuantisedBudgetGate,
) -> ECCCompressionV7Result:
    # Reuse V6 compression (which inherits V5's visible_tokens
    # accounting) and add the ultra4 data bits on top.
    v6_res = compress_carrier_ecc_v6(
        carrier, codebook=codebook.inner_v6, gate=gate)
    codes, bch_per_seg = codebook.encode_with_bch(carrier)
    _, _, emit_mask, _ = gate.forward_value(carrier)
    ultra4_bits = int(codebook.ultra4_bits())
    structured_bits_v7 = int(
        v6_res.structured_bits_v6) + int(ultra4_bits)
    visible_tokens = int(v6_res.inner_v5.visible_tokens)
    bits_per = (
        float(structured_bits_v7)
        / float(max(1, visible_tokens))
        if visible_tokens > 0 else 0.0)
    return ECCCompressionV7Result(
        coarse=int(codes[0]),
        fine=int(codes[1]),
        ultra=int(codes[2]),
        ultra2=int(codes[3]),
        ultra3=int(codes[4]),
        ultra4=int(codes[5]),
        emit_mask_bits=tuple(int(b) & 1 for b in emit_mask),
        bch_per_segment=tuple(bch_per_seg),
        structured_bits_v7=int(structured_bits_v7),
        visible_tokens=int(visible_tokens),
        bits_per_visible_token_v7=float(bits_per),
        parity_bits_per_triple=int(
            codebook.bch_parity_bits_per_triple()),
    )


# =============================================================================
# Bits/visible-token witness
# =============================================================================


@dataclasses.dataclass(frozen=True)
class ECCCompressionV7Witness:
    codebook_cid: str
    coarse: int
    fine: int
    ultra: int
    ultra2: int
    ultra3: int
    ultra4: int
    structured_bits_v7: int
    visible_tokens: int
    parity_bits_per_triple: int
    bits_per_visible_token: float
    target_bits_per_token: float
    rate_target_met: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "codebook_cid": str(self.codebook_cid),
            "coarse": int(self.coarse),
            "fine": int(self.fine),
            "ultra": int(self.ultra),
            "ultra2": int(self.ultra2),
            "ultra3": int(self.ultra3),
            "ultra4": int(self.ultra4),
            "structured_bits_v7": int(
                self.structured_bits_v7),
            "visible_tokens": int(self.visible_tokens),
            "parity_bits_per_triple": int(
                self.parity_bits_per_triple),
            "bits_per_visible_token": float(round(
                self.bits_per_visible_token, 12)),
            "target_bits_per_token": float(round(
                self.target_bits_per_token, 12)),
            "rate_target_met": bool(self.rate_target_met),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w55_ecc_v7_witness",
            "witness": self.to_dict()})


def emit_ecc_v7_compression_witness(
        *,
        codebook: ECCCodebookV7,
        compression: ECCCompressionV7Result,
        target_bits_per_token: float = (
            W55_DEFAULT_ECC_V7_TARGET_BITS_PER_TOKEN),
) -> ECCCompressionV7Witness:
    bvt = float(compression.bits_per_visible_token_v7)
    return ECCCompressionV7Witness(
        codebook_cid=str(codebook.cid()),
        coarse=int(compression.coarse),
        fine=int(compression.fine),
        ultra=int(compression.ultra),
        ultra2=int(compression.ultra2),
        ultra3=int(compression.ultra3),
        ultra4=int(compression.ultra4),
        structured_bits_v7=int(
            compression.structured_bits_v7),
        visible_tokens=int(compression.visible_tokens),
        parity_bits_per_triple=int(
            compression.parity_bits_per_triple),
        bits_per_visible_token=float(bvt),
        target_bits_per_token=float(
            target_bits_per_token),
        rate_target_met=bool(
            float(bvt) >= float(target_bits_per_token)),
    )


def probe_ecc_v7_rate_floor_falsifier(
        *,
        codebook: ECCCodebookV7,
        gate: QuantisedBudgetGate,
        sample_carriers: Sequence[Sequence[float]],
        target_bits_per_token: float = 96.0,
) -> dict[str, Any]:
    """Probe rate floor falsifier: 96-bit target is
    structurally unachievable since K1*K2*K3*K4*K5*K6 = 65536 →
    ~16 data bits/triple cap.
    """
    if not sample_carriers:
        return {
            "n_carriers": 0,
            "rate_target_missed_count": 0,
            "rate_target_missed_rate": 1.0,
        }
    n_missed = 0
    for c in sample_carriers:
        comp = compress_carrier_ecc_v7(
            c, codebook=codebook, gate=gate)
        bvt = float(comp.bits_per_visible_token_v7)
        if bvt < float(target_bits_per_token):
            n_missed += 1
    return {
        "n_carriers": int(len(sample_carriers)),
        "rate_target_missed_count": int(n_missed),
        "rate_target_missed_rate": float(
            n_missed) / float(max(1, len(sample_carriers))),
    }


# =============================================================================
# Verifier
# =============================================================================

W55_ECC_V7_VERIFIER_FAILURE_MODES: tuple[str, ...] = (
    "w55_ecc_v7_codebook_cid_mismatch",
    "w55_ecc_v7_data_bits_below_floor",
    "w55_ecc_v7_bits_per_token_below_target",
    "w55_ecc_v7_emit_mask_sum_zero",
)


def verify_ecc_v7_witness(
        witness: ECCCompressionV7Witness,
        *,
        expected_codebook_cid: str | None = None,
        require_rate_target_met: bool = False,
) -> dict[str, Any]:
    failures: list[str] = []
    if (expected_codebook_cid is not None
            and witness.codebook_cid
            != str(expected_codebook_cid)):
        failures.append("w55_ecc_v7_codebook_cid_mismatch")
    if int(witness.data_bits_per_triple) < 1:
        failures.append("w55_ecc_v7_data_bits_below_floor")
    if (require_rate_target_met
            and not bool(witness.rate_target_met)):
        failures.append(
            "w55_ecc_v7_bits_per_token_below_target")
    if int(witness.emit_mask_sum) <= 0:
        failures.append("w55_ecc_v7_emit_mask_sum_zero")
    return {
        "ok": (len(failures) == 0),
        "failures": failures,
        "witness_cid": witness.cid(),
    }


__all__ = [
    "W55_ECC_V7_SCHEMA_VERSION",
    "W55_DEFAULT_ECC_V7_K1",
    "W55_DEFAULT_ECC_V7_K2",
    "W55_DEFAULT_ECC_V7_K3",
    "W55_DEFAULT_ECC_V7_K4",
    "W55_DEFAULT_ECC_V7_K5",
    "W55_DEFAULT_ECC_V7_K6",
    "W55_DEFAULT_ECC_V7_CODE_DIM",
    "W55_DEFAULT_ECC_V7_EMIT_MASK_LEN",
    "W55_DEFAULT_ECC_V7_TARGET_BITS_PER_TOKEN",
    "W55_ECC_V7_BCH_BITS_PER_SEGMENT",
    "W55_ECC_V7_N_SEGMENTS",
    "W55_ECC_V7_VERIFIER_FAILURE_MODES",
    "ECCCodebookV7",
    "ECCCompressionV7Result",
    "ECCCompressionV7Witness",
    "compress_carrier_ecc_v7",
    "emit_ecc_v7_compression_witness",
    "probe_ecc_v7_rate_floor_falsifier",
    "verify_ecc_v7_witness",
]
