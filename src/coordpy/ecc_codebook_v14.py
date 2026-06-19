"""W62 — ECC Codebook V14.

Strictly extends W61's ``coordpy.ecc_codebook_v13``. V13 had
12 levels (K1..K12 = 2^22 = 4 194 304 codes) at 24.333
bits/visible-token. V14 adds:

* **13 levels** (K13 = 2).
* **Total codes = 2^23 = 8 388 608**.
* **25.333 bits/visible-token** at full emit (≥ 25.0 target).
* ``W62-L-ECC-V14-RATE-FLOOR-CAP`` — structural rate ceiling
  log2(2^23) = 23 raw data bits per segment-tuple.

Honest scope
------------

* Adds one bit per visible token over V13.
* The 4096-bit/token falsifier reproduces the structural ceiling.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Sequence

from .ecc_codebook_v13 import (
    ECCCodebookV13,
    compress_carrier_ecc_v13,
)
from .tiny_substrate_v3 import _sha256_hex


W62_ECC_V14_SCHEMA_VERSION: str = "coordpy.ecc_codebook_v14.v1"
W62_DEFAULT_ECC_V14_K13: int = 2
W62_DEFAULT_ECC_V14_TARGET_BITS_PER_TOKEN: float = 25.0


@dataclasses.dataclass
class ECCCodebookV14:
    inner_v13: ECCCodebookV13
    n_ultra11: int   # K13 = 2 by default
    code_dim: int

    @classmethod
    def init(
            cls, *, n_ultra11: int = W62_DEFAULT_ECC_V14_K13,
            code_dim: int = 6, seed: int = 62140, **kwargs: Any,
    ) -> "ECCCodebookV14":
        v13 = ECCCodebookV13.init(
            code_dim=int(code_dim), seed=int(seed), **kwargs)
        return cls(
            inner_v13=v13,
            n_ultra11=int(n_ultra11),
            code_dim=int(code_dim),
        )

    @property
    def total_codes(self) -> int:
        return int(self.inner_v13.total_codes
                   * int(self.n_ultra11))

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W62_ECC_V14_SCHEMA_VERSION,
            "kind": "ecc_v14_codebook",
            "inner_v13_cid": str(self.inner_v13.cid()),
            "n_ultra11": int(self.n_ultra11),
            "code_dim": int(self.code_dim),
        })


def compress_carrier_ecc_v14(
        carrier: Sequence[float], *,
        codebook: ECCCodebookV14, gate: Any,
) -> dict[str, Any]:
    """V14 = V13 + K13 1-bit hyper-fine. Adds 1 bit per visible
    token over V13."""
    v13_result = compress_carrier_ecc_v13(
        list(carrier), codebook=codebook.inner_v13, gate=gate)
    cd = int(codebook.code_dim)
    seventh = sum(float(x) ** 7 for x in list(carrier)[:cd])
    ultra11 = int(seventh > 0.0)
    visible_tokens = int(v13_result["visible_tokens"])
    extra_bits = 1
    structured_bits_v14 = int(
        int(v13_result["structured_bits_v13"])
        + int(extra_bits * max(1, visible_tokens)))
    bits_per = (
        float(structured_bits_v14)
        / float(max(1, visible_tokens))
        if visible_tokens > 0 else 0.0)
    return {
        "schema": W62_ECC_V14_SCHEMA_VERSION,
        "v13_compression_result": v13_result,
        "ultra11_index": int(ultra11),
        "structured_bits_v14": int(structured_bits_v14),
        "visible_tokens": int(visible_tokens),
        "bits_per_visible_token": float(bits_per),
        "parity_bits_per_segment_tuple": int(
            int(v13_result["parity_bits_per_segment_tuple"])
            + 8),
    }


@dataclasses.dataclass(frozen=True)
class ECCCompressionV14Witness:
    schema: str
    codebook_cid: str
    structured_bits_v14: int
    visible_tokens: int
    bits_per_visible_token: float
    ultra11_index: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "codebook_cid": str(self.codebook_cid),
            "structured_bits_v14": int(
                self.structured_bits_v14),
            "visible_tokens": int(self.visible_tokens),
            "bits_per_visible_token": float(round(
                self.bits_per_visible_token, 12)),
            "ultra11_index": int(self.ultra11_index),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "ecc_v14_compression_witness",
            "witness": self.to_dict()})


def emit_ecc_v14_compression_witness(
        *, codebook: ECCCodebookV14,
        compression: dict[str, Any],
) -> ECCCompressionV14Witness:
    return ECCCompressionV14Witness(
        schema=W62_ECC_V14_SCHEMA_VERSION,
        codebook_cid=str(codebook.cid()),
        structured_bits_v14=int(
            compression["structured_bits_v14"]),
        visible_tokens=int(compression["visible_tokens"]),
        bits_per_visible_token=float(
            compression["bits_per_visible_token"]),
        ultra11_index=int(compression["ultra11_index"]),
    )


def probe_ecc_v14_rate_floor_falsifier(
        *, codebook: ECCCodebookV14,
) -> dict[str, Any]:
    """Reproduces the structural rate ceiling: log2(total_codes)
    raw data bits per segment-tuple. With 2^23 codes the ceiling
    is 23 bits; the 4096-bit/token target trivially exceeds it,
    so the falsifier asserts the ceiling holds."""
    import math
    total = int(codebook.total_codes)
    ceiling_bits = float(
        math.log2(total) if total > 0 else 0.0)
    target_bits = 4096.0
    return {
        "schema": W62_ECC_V14_SCHEMA_VERSION,
        "kind": "rate_floor_falsifier",
        "total_codes": int(total),
        "ceiling_bits": float(round(ceiling_bits, 12)),
        "target_bits": float(target_bits),
        "target_exceeds_ceiling": bool(target_bits > ceiling_bits),
    }


__all__ = [
    "W62_ECC_V14_SCHEMA_VERSION",
    "W62_DEFAULT_ECC_V14_K13",
    "W62_DEFAULT_ECC_V14_TARGET_BITS_PER_TOKEN",
    "ECCCodebookV14",
    "compress_carrier_ecc_v14",
    "ECCCompressionV14Witness",
    "emit_ecc_v14_compression_witness",
    "probe_ecc_v14_rate_floor_falsifier",
]
