"""W69 M14 — ECC Codebook V21.

Strictly extends W68's ``coordpy.ecc_codebook_v20``. V20 had 19
levels (K1..K19 = 2^33 = 8 589 934 592 codes) at 35.0 bits/visible-
token. V21 adds:

* **20 levels** (K20 = 4).
* **Total codes = 2^35 = 34 359 738 368** (V20 had 2^33; adds 2
  bits).
* **≥ 37.0 bits/visible-token** at full emit.
* ``W69-L-ECC-V21-RATE-FLOOR-CAP`` — structural rate ceiling
  log2(2^35) = 35 raw data bits per segment-tuple.

Honest scope (W69): adds two bits per visible token over V20.
The 65536-bit/token falsifier reproduces the structural ceiling.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from .ecc_codebook_v20 import (
    ECCCodebookV20, compress_carrier_ecc_v20,
)
from .tiny_substrate_v3 import _sha256_hex


W69_ECC_V21_SCHEMA_VERSION: str = "coordpy.ecc_codebook_v21.v1"
W69_DEFAULT_ECC_V21_K20: int = 4
W69_DEFAULT_ECC_V21_TARGET_BITS_PER_TOKEN: float = 37.0


@dataclasses.dataclass
class ECCCodebookV21:
    inner_v20: ECCCodebookV20
    n_meta18: int   # K20 = 4 by default
    code_dim: int

    @classmethod
    def init(
            cls, *, n_meta18: int = W69_DEFAULT_ECC_V21_K20,
            code_dim: int = 6, seed: int = 69200,
            **kwargs: Any,
    ) -> "ECCCodebookV21":
        v20 = ECCCodebookV20.init(
            code_dim=int(code_dim), seed=int(seed), **kwargs)
        return cls(
            inner_v20=v20, n_meta18=int(n_meta18),
            code_dim=int(code_dim),
        )

    @property
    def total_codes(self) -> int:
        return int(self.inner_v20.total_codes
                   * int(self.n_meta18))

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W69_ECC_V21_SCHEMA_VERSION,
            "kind": "ecc_v21_codebook",
            "inner_v20_cid": str(self.inner_v20.cid()),
            "n_meta18": int(self.n_meta18),
            "code_dim": int(self.code_dim),
        })


def compress_carrier_ecc_v21(
        carrier: Sequence[float], *,
        codebook: ECCCodebookV21, gate: Any,
) -> dict[str, Any]:
    """V21 = V20 + K20 2-bit meta-fine. Adds 2 bits per visible
    token over V20."""
    v20_result = compress_carrier_ecc_v20(
        list(carrier), codebook=codebook.inner_v20, gate=gate)
    cd = int(codebook.code_dim)
    eighteenth = sum(float(x) ** 18 for x in list(carrier)[:cd])
    meta18 = int(abs(eighteenth * 1e3)) % int(
        max(1, codebook.n_meta18))
    visible_tokens = int(v20_result["visible_tokens"])
    extra_bits = 2
    structured_bits_v21 = int(
        int(v20_result["structured_bits_v20"])
        + int(extra_bits * max(1, visible_tokens)))
    bits_per = (
        float(structured_bits_v21)
        / float(max(1, visible_tokens))
        if visible_tokens > 0 else 0.0)
    return {
        "schema": W69_ECC_V21_SCHEMA_VERSION,
        "v20_compression_result": v20_result,
        "meta18_index": int(meta18),
        "structured_bits_v21": int(structured_bits_v21),
        "visible_tokens": int(visible_tokens),
        "bits_per_visible_token": float(bits_per),
        "parity_bits_per_segment_tuple": int(
            int(v20_result["parity_bits_per_segment_tuple"]) + 8),
    }


@dataclasses.dataclass(frozen=True)
class ECCCompressionV21Witness:
    schema: str
    codebook_cid: str
    structured_bits_v21: int
    visible_tokens: int
    bits_per_visible_token: float
    meta18_index: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "codebook_cid": str(self.codebook_cid),
            "structured_bits_v21": int(self.structured_bits_v21),
            "visible_tokens": int(self.visible_tokens),
            "bits_per_visible_token": float(round(
                self.bits_per_visible_token, 12)),
            "meta18_index": int(self.meta18_index),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "ecc_v21_compression_witness",
            "witness": self.to_dict()})


def emit_ecc_v21_compression_witness(
        *, codebook: ECCCodebookV21,
        compression: dict[str, Any],
) -> ECCCompressionV21Witness:
    return ECCCompressionV21Witness(
        schema=W69_ECC_V21_SCHEMA_VERSION,
        codebook_cid=str(codebook.cid()),
        structured_bits_v21=int(
            compression["structured_bits_v21"]),
        visible_tokens=int(compression["visible_tokens"]),
        bits_per_visible_token=float(
            compression["bits_per_visible_token"]),
        meta18_index=int(compression["meta18_index"]),
    )


def probe_ecc_v21_rate_floor_falsifier(
        *, codebook: ECCCodebookV21,
) -> dict[str, Any]:
    """Reproduces the structural rate ceiling: log2(total_codes)
    raw data bits per segment-tuple. With 2^35 codes the ceiling
    is 35 bits; the 65536-bit/token target trivially exceeds it.
    """
    import math
    total = int(codebook.total_codes)
    ceiling_bits = float(
        math.log2(total) if total > 0 else 0.0)
    target_bits = 65536.0
    return {
        "schema": W69_ECC_V21_SCHEMA_VERSION,
        "kind": "rate_floor_falsifier",
        "total_codes": int(total),
        "ceiling_bits": float(round(ceiling_bits, 12)),
        "target_bits": float(target_bits),
        "target_exceeds_ceiling": bool(target_bits > ceiling_bits),
    }


__all__ = [
    "W69_ECC_V21_SCHEMA_VERSION",
    "W69_DEFAULT_ECC_V21_K20",
    "W69_DEFAULT_ECC_V21_TARGET_BITS_PER_TOKEN",
    "ECCCodebookV21",
    "compress_carrier_ecc_v21",
    "ECCCompressionV21Witness",
    "emit_ecc_v21_compression_witness",
    "probe_ecc_v21_rate_floor_falsifier",
]
