"""W53 M5 — ECC Codebook V5 (4-level + parity bits).

Stacks a fourth quantisation level (ultra-fine-2 = K4) on top of
W52's K1×K2×K3 codebook AND adds **parity bits** to enable
single-bit corruption detection.

Capacity: K1=32 × K2=16 × K3=8 × K4=4 = 16384 codes per pair
≈ 14 bits/triple. Add 4 parity bits per (coarse, fine, ultra,
ultra2) segment → 18 bits structured per visible token.

The parity scheme is XOR-based:

    parity(coarse_bits) = xor of coarse_bits
    parity(fine_bits)   = xor of fine_bits
    parity(ultra_bits)  = xor of ultra_bits
    parity(ultra2_bits) = xor of ultra2_bits

A single-bit flip in any one segment toggles its parity bit, so
the receiver can detect *which segment* was corrupted (then
re-encode from the other surviving segments — only partial
correction).

Pure-Python only. Reuses ``QuantisedCodebookV4`` for the K1/K2/K3
prototypes and adds K4 prototypes.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from .autograd_manifold import (
    ParamTensor,
    W47_DEFAULT_INIT_SCALE,
    W47_DEFAULT_TRAIN_SEED,
    _DeterministicLCG,
)
from .quantised_compression import (
    QuantisedBudgetGate,
    QuantisedCodebookV4,
    W52_DEFAULT_QUANT_BITS_PAYLOAD_LEN,
    W52_DEFAULT_QUANT_CODE_DIM,
    W52_DEFAULT_QUANT_EMIT_MASK_LEN,
    W52_DEFAULT_QUANT_K1,
    W52_DEFAULT_QUANT_K2,
    W52_DEFAULT_QUANT_K3,
    compress_carrier_quantised,
)


# =============================================================================
# Schema, defaults
# =============================================================================

W53_ECC_SCHEMA_VERSION: str = (
    "coordpy.ecc_codebook_v5.v1")

W53_DEFAULT_ECC_K1: int = W52_DEFAULT_QUANT_K1
W53_DEFAULT_ECC_K2: int = W52_DEFAULT_QUANT_K2
W53_DEFAULT_ECC_K3: int = W52_DEFAULT_QUANT_K3
W53_DEFAULT_ECC_K4: int = 4
W53_DEFAULT_ECC_CODE_DIM: int = W52_DEFAULT_QUANT_CODE_DIM
W53_DEFAULT_ECC_EMIT_MASK_LEN: int = (
    W52_DEFAULT_QUANT_EMIT_MASK_LEN)
W53_DEFAULT_ECC_BITS_PAYLOAD_LEN: int = (
    W52_DEFAULT_QUANT_BITS_PAYLOAD_LEN)
W53_DEFAULT_ECC_TARGET_BITS_PER_TOKEN: float = 14.5
W53_ECC_PARITY_BITS_PER_TOKEN: int = 4


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


def _bits_of(code: int, n_bits: int) -> list[int]:
    """LSB-first bit decomposition."""
    return [
        int((int(code) >> i) & 1)
        for i in range(int(n_bits))
    ]


def _xor_parity(bits: Sequence[int]) -> int:
    p = 0
    for b in bits:
        p ^= int(b) & 1
    return int(p)


# =============================================================================
# ECCCodebookV5
# =============================================================================


@dataclasses.dataclass
class ECCCodebookV5:
    """Four-level codebook with XOR parity bits.

    Wraps a W52 ``QuantisedCodebookV4`` for K1/K2/K3 and adds an
    ultra2 codebook (K4 prototypes) for the residual after V4
    decoding. Each segment carries one XOR parity bit.
    """

    inner_v4: QuantisedCodebookV4
    n_ultra2: int
    code_dim: int
    ultra2_proto: ParamTensor

    @classmethod
    def init(
            cls, *,
            n_coarse: int = W53_DEFAULT_ECC_K1,
            n_fine: int = W53_DEFAULT_ECC_K2,
            n_ultra: int = W53_DEFAULT_ECC_K3,
            n_ultra2: int = W53_DEFAULT_ECC_K4,
            code_dim: int = W53_DEFAULT_ECC_CODE_DIM,
            seed: int = W47_DEFAULT_TRAIN_SEED,
            init_scale: float = W47_DEFAULT_INIT_SCALE,
    ) -> "ECCCodebookV5":
        inner = QuantisedCodebookV4.init(
            n_coarse=int(n_coarse),
            n_fine=int(n_fine),
            n_ultra=int(n_ultra),
            code_dim=int(code_dim),
            seed=int(seed),
            init_scale=float(init_scale))
        ultra2 = ParamTensor(
            shape=(int(n_ultra2), int(code_dim)), values=[])
        ultra2.init_seed(
            seed=int(seed) + 257,
            scale=float(init_scale) * 0.5)
        return cls(
            inner_v4=inner,
            n_ultra2=int(n_ultra2),
            code_dim=int(code_dim),
            ultra2_proto=ultra2)

    def params(self) -> list[ParamTensor]:
        return list(self.inner_v4.params()) + [
            self.ultra2_proto]

    def coarse_bits(self) -> int:
        return int(self.inner_v4.coarse_bits())

    def fine_bits(self) -> int:
        return int(self.inner_v4.fine_bits())

    def ultra_bits(self) -> int:
        return int(self.inner_v4.ultra_bits())

    def ultra2_bits(self) -> int:
        n = max(1, int(self.n_ultra2))
        return int(max(1, math.ceil(math.log2(n))))

    def ultra2_vector(self, code: int) -> tuple[float, ...]:
        cd = int(self.code_dim)
        c = int(code) % max(1, int(self.n_ultra2))
        base = c * cd
        return tuple(
            float(self.ultra2_proto.values[base + j])
            for j in range(cd)
        )

    def encode_value(
            self, carrier: Sequence[float],
    ) -> tuple[int, int, int, int]:
        """Greedy encode: K1 → K2 → K3 → K4, each level encodes
        the residual after decoding the previous level."""
        coarse, fine, ultra = self.inner_v4.encode_value(carrier)
        # Compute residual after V4 decode.
        decoded = self.inner_v4.decode(
            coarse=coarse, fine=fine, ultra=ultra)
        residual = [
            float(carrier[j] if j < len(carrier) else 0.0)
            - float(decoded[j] if j < len(decoded) else 0.0)
            for j in range(int(self.code_dim))
        ]
        # Pick K4 prototype that minimizes residual MSE.
        best_idx = 0
        best_d = float("inf")
        for k in range(int(self.n_ultra2)):
            v = self.ultra2_vector(k)
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
            int(best_idx))

    def decode(
            self, *,
            coarse: int, fine: int,
            ultra: int, ultra2: int,
            include_ultra: bool = True,
            include_ultra2: bool = True,
    ) -> tuple[float, ...]:
        base = list(self.inner_v4.decode(
            coarse=coarse, fine=fine, ultra=ultra,
            include_ultra=bool(include_ultra)))
        if include_ultra2:
            v = self.ultra2_vector(int(ultra2))
            base = [
                float(base[j] if j < len(base) else 0.0)
                + float(v[j] if j < len(v) else 0.0)
                for j in range(int(self.code_dim))
            ]
        return tuple(base)

    def parity_bits(
            self, *,
            coarse: int, fine: int,
            ultra: int, ultra2: int,
    ) -> tuple[int, int, int, int]:
        c_bits = _bits_of(coarse, self.coarse_bits())
        f_bits = _bits_of(fine, self.fine_bits())
        u_bits = _bits_of(ultra, self.ultra_bits())
        u2_bits = _bits_of(ultra2, self.ultra2_bits())
        return (
            int(_xor_parity(c_bits)),
            int(_xor_parity(f_bits)),
            int(_xor_parity(u_bits)),
            int(_xor_parity(u2_bits)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": str(W53_ECC_SCHEMA_VERSION),
            "inner_v4": self.inner_v4.to_dict(),
            "n_ultra2": int(self.n_ultra2),
            "code_dim": int(self.code_dim),
            "ultra2_proto": self.ultra2_proto.to_dict(),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w53_ecc_codebook_v5",
            "codebook": self.to_dict()})


# =============================================================================
# ECCCompressionResult
# =============================================================================


@dataclasses.dataclass(frozen=True)
class ECCCompressionResult:
    coarse_code: int
    fine_code: int
    ultra_code: int
    ultra2_code: int
    coarse_bits: int
    fine_bits: int
    ultra_bits: int
    ultra2_bits: int
    parity_bits: tuple[int, int, int, int]
    level_mask: tuple[int, ...]
    emit_mask: tuple[int, ...]
    bits_payload: tuple[int, ...]
    visible_tokens: int
    structured_bits: int
    bits_per_visible_token: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "coarse_code": int(self.coarse_code),
            "fine_code": int(self.fine_code),
            "ultra_code": int(self.ultra_code),
            "ultra2_code": int(self.ultra2_code),
            "coarse_bits": int(self.coarse_bits),
            "fine_bits": int(self.fine_bits),
            "ultra_bits": int(self.ultra_bits),
            "ultra2_bits": int(self.ultra2_bits),
            "parity_bits": list(self.parity_bits),
            "level_mask": list(self.level_mask),
            "emit_mask": list(self.emit_mask),
            "bits_payload": list(self.bits_payload),
            "visible_tokens": int(self.visible_tokens),
            "structured_bits": int(self.structured_bits),
            "bits_per_visible_token": float(round(
                self.bits_per_visible_token, 12)),
        }


def compress_carrier_ecc(
        carrier: Sequence[float],
        *,
        codebook: ECCCodebookV5,
        gate: QuantisedBudgetGate,
        bits_payload_len: int = (
            W53_DEFAULT_ECC_BITS_PAYLOAD_LEN),
        max_visible_tokens: int | None = None,
) -> ECCCompressionResult:
    """Compress a carrier with ECC V5.

    Token accounting for the W53 packed ``LATENT_CTRL_V5_ECC``
    block:

    * **token 0**: coarse + fine + level mask + 4 parity bits
    * **token 1**: ultra + ultra2 + emit mask
    * **token 2**: bits payload (if mask_sum > 0)

    Visible tokens cap unchanged from W52 — typically 3.
    """
    coarse, fine, ultra, ultra2 = codebook.encode_value(carrier)
    coarse_bits = codebook.coarse_bits()
    fine_bits = codebook.fine_bits()
    ultra_bits = codebook.ultra_bits()
    ultra2_bits = codebook.ultra2_bits()
    parity = codebook.parity_bits(
        coarse=coarse, fine=fine, ultra=ultra,
        ultra2=ultra2)
    decoded = codebook.decode(
        coarse=coarse, fine=fine,
        ultra=ultra, ultra2=ultra2)
    level_mask, _, emit_mask, _ = (
        gate.forward_value(carrier))
    payload: list[int] = []
    for j in range(int(bits_payload_len)):
        if j < len(carrier):
            d_j = (
                float(decoded[j])
                if j < len(decoded) else 0.0)
            resid = float(carrier[j]) - d_j
            payload.append(1 if resid >= 0.0 else 0)
        else:
            payload.append(0)
    mask_sum = sum(int(m) for m in emit_mask)
    fine_emitted = (
        int(level_mask[1]) if len(level_mask) >= 2 else 1)
    ultra_emitted = (
        int(level_mask[2]) if len(level_mask) >= 3 else 1)
    visible_tokens = (
        1 + (1 if (fine_emitted or ultra_emitted) else 0)
        + (1 if mask_sum > 0 else 0))
    if (max_visible_tokens is not None
            and visible_tokens > int(max_visible_tokens)):
        if int(max_visible_tokens) == 1:
            fine_emitted = 0
            ultra_emitted = 0
            mask_sum = 0
        elif int(max_visible_tokens) == 2:
            mask_sum = 0
        visible_tokens = int(max_visible_tokens)
    structured_bits = (
        int(coarse_bits) + 3
        + int(W53_ECC_PARITY_BITS_PER_TOKEN)
        + (
            int(fine_bits)
            + (int(ultra_bits)
               if ultra_emitted else 0)
            + (int(ultra2_bits)
               if ultra_emitted else 0)
            + int(gate.emit_mask_len)
            if (fine_emitted or ultra_emitted) else 0)
        + int(mask_sum))
    bits_per = (
        float(structured_bits) / float(max(1, visible_tokens))
        if visible_tokens > 0 else 0.0)
    return ECCCompressionResult(
        coarse_code=int(coarse),
        fine_code=int(fine),
        ultra_code=int(ultra),
        ultra2_code=int(ultra2),
        coarse_bits=int(coarse_bits),
        fine_bits=int(fine_bits),
        ultra_bits=int(ultra_bits),
        ultra2_bits=int(ultra2_bits),
        parity_bits=tuple(parity),
        level_mask=tuple(int(m) for m in level_mask),
        emit_mask=tuple(int(m) for m in emit_mask),
        bits_payload=tuple(payload),
        visible_tokens=int(visible_tokens),
        structured_bits=int(structured_bits),
        bits_per_visible_token=float(bits_per),
    )


# =============================================================================
# Corruption detection / partial correction
# =============================================================================


@dataclasses.dataclass(frozen=True)
class ECCDecodeAttempt:
    """Result of decoding a possibly-corrupted ECC compression."""

    detected_segments: tuple[str, ...]
    n_corrupted_segments: int
    decoded_payload: tuple[float, ...]
    abstain: bool
    corrected_partial: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "detected_segments": list(
                self.detected_segments),
            "n_corrupted_segments": int(
                self.n_corrupted_segments),
            "decoded_payload": list(
                _round_floats(self.decoded_payload)),
            "abstain": bool(self.abstain),
            "corrected_partial": bool(
                self.corrected_partial),
        }


def decode_with_parity_check(
        *,
        codebook: ECCCodebookV5,
        coarse: int, fine: int,
        ultra: int, ultra2: int,
        observed_parity: tuple[int, int, int, int],
) -> ECCDecodeAttempt:
    """Detect single-segment corruption via parity check.

    If exactly one parity bit disagrees, mark that segment as
    corrupted and partial-correct by zeroing its contribution
    from the decode (the remaining segments still carry signal).
    If two or more parity bits disagree, abstain.
    """
    expected_parity = codebook.parity_bits(
        coarse=coarse, fine=fine,
        ultra=ultra, ultra2=ultra2)
    detected: list[str] = []
    seg_names = ("coarse", "fine", "ultra", "ultra2")
    for i, name in enumerate(seg_names):
        if int(expected_parity[i]) != int(observed_parity[i]):
            detected.append(name)
    n_corrupted = len(detected)
    if n_corrupted == 0:
        # No corruption.
        decoded = codebook.decode(
            coarse=coarse, fine=fine,
            ultra=ultra, ultra2=ultra2)
        return ECCDecodeAttempt(
            detected_segments=(),
            n_corrupted_segments=0,
            decoded_payload=tuple(decoded),
            abstain=False,
            corrected_partial=False,
        )
    if n_corrupted == 1:
        # Partial correction: drop the corrupted segment.
        seg = detected[0]
        decoded = codebook.decode(
            coarse=coarse, fine=fine,
            ultra=ultra, ultra2=ultra2,
            include_ultra=(seg != "ultra"
                            and seg != "ultra2"),
            include_ultra2=(seg != "ultra2"
                            and seg != "ultra"))
        # If coarse or fine were corrupted, still decode but
        # downweight them by zeroing their prototype contribution
        # (modelled by re-decoding with code 0).
        if seg == "coarse":
            decoded = codebook.decode(
                coarse=0, fine=fine,
                ultra=ultra, ultra2=ultra2)
        elif seg == "fine":
            decoded = codebook.decode(
                coarse=coarse, fine=0,
                ultra=ultra, ultra2=ultra2)
        return ECCDecodeAttempt(
            detected_segments=tuple(detected),
            n_corrupted_segments=1,
            decoded_payload=tuple(decoded),
            abstain=False,
            corrected_partial=True,
        )
    # 2+ corrupted segments → abstain.
    return ECCDecodeAttempt(
        detected_segments=tuple(detected),
        n_corrupted_segments=int(n_corrupted),
        decoded_payload=tuple([0.0] * int(codebook.code_dim)),
        abstain=True,
        corrected_partial=False,
    )


def flip_random_bit(
        *,
        coarse: int, fine: int,
        ultra: int, ultra2: int,
        codebook: ECCCodebookV5,
        seed: int = 0,
) -> tuple[int, int, int, int, str]:
    """Flip a single bit somewhere in the four codes.

    Returns (coarse', fine', ultra', ultra2', segment_name).
    Deterministic given the seed.
    """
    rng = _DeterministicLCG(seed=int(seed))
    seg_n = [
        ("coarse", codebook.coarse_bits()),
        ("fine", codebook.fine_bits()),
        ("ultra", codebook.ultra_bits()),
        ("ultra2", codebook.ultra2_bits()),
    ]
    total_bits = sum(int(n) for _, n in seg_n)
    pick = int(rng.next_uniform() * float(total_bits))
    pick = max(0, min(total_bits - 1, pick))
    cur = 0
    seg_name = "coarse"
    bit_in_seg = 0
    for name, n in seg_n:
        if pick < cur + int(n):
            seg_name = str(name)
            bit_in_seg = int(pick - cur)
            break
        cur += int(n)
    if seg_name == "coarse":
        coarse = int(coarse) ^ (1 << int(bit_in_seg))
        coarse = int(coarse) % max(
            1, codebook.inner_v4.n_coarse)
    elif seg_name == "fine":
        fine = int(fine) ^ (1 << int(bit_in_seg))
        fine = int(fine) % max(
            1, codebook.inner_v4.n_fine)
    elif seg_name == "ultra":
        ultra = int(ultra) ^ (1 << int(bit_in_seg))
        ultra = int(ultra) % max(
            1, codebook.inner_v4.n_ultra)
    elif seg_name == "ultra2":
        ultra2 = int(ultra2) ^ (1 << int(bit_in_seg))
        ultra2 = int(ultra2) % max(1, codebook.n_ultra2)
    return (
        int(coarse), int(fine),
        int(ultra), int(ultra2), str(seg_name))


# =============================================================================
# Witnesses
# =============================================================================


@dataclasses.dataclass(frozen=True)
class ECCCompressionWitness:
    codebook_cid: str
    bits_per_visible_token: float
    target_bits_per_token: float
    target_met: bool
    parity_bits_count: int
    code_capacity_bits: int
    n_ultra2: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "codebook_cid": str(self.codebook_cid),
            "bits_per_visible_token": float(round(
                self.bits_per_visible_token, 12)),
            "target_bits_per_token": float(round(
                self.target_bits_per_token, 12)),
            "target_met": bool(self.target_met),
            "parity_bits_count": int(self.parity_bits_count),
            "code_capacity_bits": int(
                self.code_capacity_bits),
            "n_ultra2": int(self.n_ultra2),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w53_ecc_compression_witness",
            "witness": self.to_dict()})


def emit_ecc_compression_witness(
        *,
        codebook: ECCCodebookV5,
        compression: ECCCompressionResult,
        target_bits_per_token: float = (
            W53_DEFAULT_ECC_TARGET_BITS_PER_TOKEN),
) -> ECCCompressionWitness:
    cap = (
        int(codebook.coarse_bits())
        + int(codebook.fine_bits())
        + int(codebook.ultra_bits())
        + int(codebook.ultra2_bits()))
    return ECCCompressionWitness(
        codebook_cid=str(codebook.cid()),
        bits_per_visible_token=float(
            compression.bits_per_visible_token),
        target_bits_per_token=float(
            target_bits_per_token),
        target_met=bool(
            compression.bits_per_visible_token
            >= float(target_bits_per_token)),
        parity_bits_count=int(W53_ECC_PARITY_BITS_PER_TOKEN),
        code_capacity_bits=int(cap),
        n_ultra2=int(codebook.n_ultra2),
    )


@dataclasses.dataclass(frozen=True)
class ECCRobustnessWitness:
    n_probes: int
    n_detected: int
    n_corrected_partial: int
    n_abstain: int
    detect_rate: float
    correction_rate: float
    abstain_rate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_probes": int(self.n_probes),
            "n_detected": int(self.n_detected),
            "n_corrected_partial": int(
                self.n_corrected_partial),
            "n_abstain": int(self.n_abstain),
            "detect_rate": float(round(self.detect_rate, 12)),
            "correction_rate": float(round(
                self.correction_rate, 12)),
            "abstain_rate": float(round(
                self.abstain_rate, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w53_ecc_robustness_witness",
            "witness": self.to_dict()})


def emit_ecc_robustness_witness(
        *,
        carriers: Sequence[Sequence[float]],
        codebook: ECCCodebookV5,
        gate: QuantisedBudgetGate,
        seed: int = 0,
) -> ECCRobustnessWitness:
    """Probe single-bit corruption detect/correction across many
    carriers."""
    n = 0
    n_detected = 0
    n_corr = 0
    n_abs = 0
    for i, c in enumerate(carriers):
        comp = compress_carrier_ecc(
            c, codebook=codebook, gate=gate)
        # Flip one bit.
        c2, f2, u2, u22, seg_name = flip_random_bit(
            coarse=comp.coarse_code,
            fine=comp.fine_code,
            ultra=comp.ultra_code,
            ultra2=comp.ultra2_code,
            codebook=codebook,
            seed=int(seed) + int(i))
        # Observed parity is the parity of the original codes
        # (since the parity is sent with the original codes; it
        # was unaffected by the post-emit flip).
        observed_parity = comp.parity_bits
        attempt = decode_with_parity_check(
            codebook=codebook,
            coarse=c2, fine=f2,
            ultra=u2, ultra2=u22,
            observed_parity=observed_parity)
        n += 1
        if attempt.n_corrupted_segments >= 1:
            n_detected += 1
        if attempt.corrected_partial:
            n_corr += 1
        if attempt.abstain:
            n_abs += 1
    return ECCRobustnessWitness(
        n_probes=int(n),
        n_detected=int(n_detected),
        n_corrected_partial=int(n_corr),
        n_abstain=int(n_abs),
        detect_rate=float(n_detected / max(1, n)),
        correction_rate=float(n_corr / max(1, n)),
        abstain_rate=float(n_abs / max(1, n)),
    )


# =============================================================================
# Rate-floor falsifier
# =============================================================================


@dataclasses.dataclass(frozen=True)
class ECCRateFloorFalsifierResult:
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


def probe_ecc_rate_floor_falsifier(
        carrier: Sequence[float],
        *,
        codebook: ECCCodebookV5,
        gate: QuantisedBudgetGate,
        target_bits_per_token: float = 40.0,
) -> ECCRateFloorFalsifierResult:
    res = compress_carrier_ecc(
        carrier, codebook=codebook, gate=gate)
    return ECCRateFloorFalsifierResult(
        target_bits_per_token=float(target_bits_per_token),
        achieved_bits_per_token=float(
            res.bits_per_visible_token),
        rate_target_missed=bool(
            res.bits_per_visible_token
            < float(target_bits_per_token)),
    )


# =============================================================================
# Verifier
# =============================================================================

W53_ECC_VERIFIER_FAILURE_MODES: tuple[str, ...] = (
    "w53_ecc_codebook_cid_mismatch",
    "w53_ecc_bits_below_floor",
    "w53_ecc_parity_count_mismatch",
    "w53_ecc_n_ultra2_mismatch",
)


def verify_ecc_compression_witness(
        witness: ECCCompressionWitness,
        *,
        expected_codebook_cid: str | None = None,
        min_bits_per_token: float | None = None,
        expected_n_ultra2: int | None = None,
) -> dict[str, Any]:
    failures: list[str] = []
    if (expected_codebook_cid is not None
            and witness.codebook_cid
            != str(expected_codebook_cid)):
        failures.append("w53_ecc_codebook_cid_mismatch")
    if (min_bits_per_token is not None
            and witness.bits_per_visible_token
            < float(min_bits_per_token)):
        failures.append("w53_ecc_bits_below_floor")
    if witness.parity_bits_count != int(
            W53_ECC_PARITY_BITS_PER_TOKEN):
        failures.append("w53_ecc_parity_count_mismatch")
    if (expected_n_ultra2 is not None
            and witness.n_ultra2 != int(expected_n_ultra2)):
        failures.append("w53_ecc_n_ultra2_mismatch")
    return {
        "ok": (len(failures) == 0),
        "failures": failures,
        "witness_cid": witness.cid(),
    }


__all__ = [
    "W53_ECC_SCHEMA_VERSION",
    "W53_DEFAULT_ECC_K1",
    "W53_DEFAULT_ECC_K2",
    "W53_DEFAULT_ECC_K3",
    "W53_DEFAULT_ECC_K4",
    "W53_DEFAULT_ECC_CODE_DIM",
    "W53_DEFAULT_ECC_EMIT_MASK_LEN",
    "W53_DEFAULT_ECC_BITS_PAYLOAD_LEN",
    "W53_DEFAULT_ECC_TARGET_BITS_PER_TOKEN",
    "W53_ECC_PARITY_BITS_PER_TOKEN",
    "W53_ECC_VERIFIER_FAILURE_MODES",
    "ECCCodebookV5",
    "ECCCompressionResult",
    "ECCCompressionWitness",
    "ECCRobustnessWitness",
    "ECCDecodeAttempt",
    "ECCRateFloorFalsifierResult",
    "compress_carrier_ecc",
    "decode_with_parity_check",
    "flip_random_bit",
    "emit_ecc_compression_witness",
    "emit_ecc_robustness_witness",
    "probe_ecc_rate_floor_falsifier",
    "verify_ecc_compression_witness",
]
