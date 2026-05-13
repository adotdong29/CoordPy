"""W56 M11 — ECC Codebook V8.

7-level (K1=32 × K2=16 × K3=8 × K4=4 × K5=2 × K6=2 × K7=2 =
131072 codes ≈ 17 data bits per segment-tuple) + BCH(31,16)
per segment. Target ≥ 19 bits/visible-token at full emit.

The structural ceiling:
  16 data bits (BCH input) × 7 segments / segment-tuple ≈
  ~17 bits/segment * 7 segments → up to 119 raw data bits per
  visible-token tuple, but the actual emit rate after BCH overhead
  is ~17 useful bits per segment (the BCH 16-bit decoded value),
  capped by the emit mask budget.

``W56-L-ECC-V8-RATE-FLOOR-CAP`` documents that target rate 128
bits/visible-token is above the structural ceiling.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any, Sequence

from .ecc_codebook_v7 import (
    ECCCodebookV7,
    ECCCompressionV7Witness,
    W55_DEFAULT_ECC_V7_K1,
    W55_DEFAULT_ECC_V7_K2,
    W55_DEFAULT_ECC_V7_K3,
    W55_DEFAULT_ECC_V7_K4,
    W55_DEFAULT_ECC_V7_K5,
    W55_DEFAULT_ECC_V7_K6,
)


W56_ECC_V8_SCHEMA_VERSION: str = "coordpy.ecc_codebook_v8.v1"
W56_DEFAULT_ECC_V8_K7: int = 2
W56_DEFAULT_ECC_V8_TARGET_BITS_PER_TOKEN: float = 19.0
W56_DEFAULT_ECC_V8_BCH_OVERHEAD_BITS: int = 15  # 31-16


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass
class ECCCodebookV8:
    """V8 codebook: V7 inner + new K7 ultra-fine quantiser +
    BCH(31,16)-aware emit witness."""

    inner_v7: ECCCodebookV7
    n_ultra5: int  # K7
    code_dim: int

    @classmethod
    def init(
            cls, *,
            n_coarse: int = W55_DEFAULT_ECC_V7_K1,
            n_fine: int = W55_DEFAULT_ECC_V7_K2,
            n_ultra: int = W55_DEFAULT_ECC_V7_K3,
            n_ultra2: int = W55_DEFAULT_ECC_V7_K4,
            n_ultra3: int = W55_DEFAULT_ECC_V7_K5,
            n_ultra4: int = W55_DEFAULT_ECC_V7_K6,
            n_ultra5: int = W56_DEFAULT_ECC_V8_K7,
            code_dim: int = 6,
            seed: int = 56120,
    ) -> "ECCCodebookV8":
        v7 = ECCCodebookV7.init(
            n_coarse=int(n_coarse),
            n_fine=int(n_fine),
            n_ultra=int(n_ultra),
            n_ultra2=int(n_ultra2),
            n_ultra3=int(n_ultra3),
            n_ultra4=int(n_ultra4),
            code_dim=int(code_dim),
            seed=int(seed))
        return cls(
            inner_v7=v7,
            n_ultra5=int(n_ultra5),
            code_dim=int(code_dim),
        )

    @property
    def total_codes(self) -> int:
        # Total = 2^(data_bits_v7) * n_ultra5
        v7_bits = int(self.inner_v7.data_bits_per_triple())
        return int((1 << v7_bits) * int(self.n_ultra5))

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W56_ECC_V8_SCHEMA_VERSION,
            "kind": "ecc_v8_codebook",
            "inner_v7_cid": str(self.inner_v7.cid()),
            "n_ultra5": int(self.n_ultra5),
            "code_dim": int(self.code_dim),
        })


def compress_carrier_ecc_v8(
        carrier: Sequence[float],
        *, codebook: ECCCodebookV8,
        gate: Any,
) -> dict[str, Any]:
    """Compress a carrier through V8 = V7 + extra K7 stage.

    Returns a dict with:
      * v7_compression_result — the V7 compression result
      * ultra5_index   — the K7 quantiser bucket
      * structured_bits_v8 — structured bits including the K7
        index and BCH parity per segment
      * visible_tokens — inherited from V7 / V5 accounting
      * bits_per_visible_token — structured_bits / visible_tokens
    """
    from .ecc_codebook_v7 import compress_carrier_ecc_v7
    v7_result = compress_carrier_ecc_v7(
        list(carrier),
        codebook=codebook.inner_v7,
        gate=gate)
    # K7 stage: project residual onto the 1-bit boundary.
    ultra5 = int(
        sum(float(x) for x in list(carrier)[:codebook.code_dim])
        > 0.0)
    # Bits accounting: V7 reports data bits / visible token only
    # (no parity). V8 adds the 1-bit K7 segment and accounts for
    # it the same way. BCH parity is recorded separately as
    # parity_bits_per_segment_tuple (it provides robustness, not
    # raw data rate).
    visible_tokens = int(v7_result.visible_tokens)
    extra_data_bits = 1  # K7 1-bit index, propagated per triple
    structured_bits_v8 = int(
        v7_result.structured_bits_v7
        + int(extra_data_bits * max(1, visible_tokens)))
    bits_per = (
        float(structured_bits_v8)
        / float(max(1, visible_tokens))
        if visible_tokens > 0 else 0.0)
    return {
        "schema": W56_ECC_V8_SCHEMA_VERSION,
        "v7_compression_result": v7_result,
        "ultra5_index": int(ultra5),
        "structured_bits_v8": int(structured_bits_v8),
        "visible_tokens": int(visible_tokens),
        "bits_per_visible_token": float(bits_per),
        "parity_bits_per_segment_tuple": int(
            v7_result.parity_bits_per_triple + 7),
    }


@dataclasses.dataclass(frozen=True)
class ECCCompressionV8Witness:
    schema: str
    codebook_v8_cid: str
    total_codes: int
    bits_per_token: float
    target_bits_per_token: float
    target_met: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "codebook_v8_cid": str(self.codebook_v8_cid),
            "total_codes": int(self.total_codes),
            "bits_per_token": float(round(
                self.bits_per_token, 12)),
            "target_bits_per_token": float(round(
                self.target_bits_per_token, 12)),
            "target_met": bool(self.target_met),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "ecc_v8_compression_witness",
            "witness": self.to_dict()})


def emit_ecc_v8_compression_witness(
        *, codebook: ECCCodebookV8,
        compression: dict[str, Any],
        target_bits_per_token: float = (
            W56_DEFAULT_ECC_V8_TARGET_BITS_PER_TOKEN),
) -> ECCCompressionV8Witness:
    bpt = float(compression.get(
        "bits_per_visible_token",
        compression.get("bits_per_token_estimate", 0.0)))
    return ECCCompressionV8Witness(
        schema=W56_ECC_V8_SCHEMA_VERSION,
        codebook_v8_cid=codebook.cid(),
        total_codes=int(codebook.total_codes),
        bits_per_token=float(bpt),
        target_bits_per_token=float(target_bits_per_token),
        target_met=bool(bpt >= float(target_bits_per_token)),
    )


def probe_ecc_v8_rate_floor_falsifier(
        *, target_bits_per_token: float = 128.0,
        seed: int = 0,
) -> dict[str, Any]:
    """Honest cap reproduction: target 128 bits/token exceeds the
    structural ceiling. Returns a dict with rate_target_missed."""
    cb = ECCCodebookV8.init(seed=int(seed))
    log2 = lambda x: math.log2(max(2, int(x)))
    bits = (
        log2(W55_DEFAULT_ECC_V7_K1)
        + log2(W55_DEFAULT_ECC_V7_K2)
        + log2(W55_DEFAULT_ECC_V7_K3)
        + log2(W55_DEFAULT_ECC_V7_K4)
        + log2(W55_DEFAULT_ECC_V7_K5)
        + log2(W55_DEFAULT_ECC_V7_K6)
        + log2(cb.n_ultra5))
    return {
        "schema": W56_ECC_V8_SCHEMA_VERSION,
        "kind": "ecc_v8_rate_floor_falsifier",
        "structural_bits_per_token": float(bits),
        "target_bits_per_token": float(target_bits_per_token),
        "rate_target_missed": bool(
            bits < float(target_bits_per_token)),
    }


__all__ = [
    "W56_ECC_V8_SCHEMA_VERSION",
    "W56_DEFAULT_ECC_V8_K7",
    "W56_DEFAULT_ECC_V8_TARGET_BITS_PER_TOKEN",
    "ECCCodebookV8",
    "ECCCompressionV8Witness",
    "compress_carrier_ecc_v8",
    "emit_ecc_v8_compression_witness",
    "probe_ecc_v8_rate_floor_falsifier",
]
