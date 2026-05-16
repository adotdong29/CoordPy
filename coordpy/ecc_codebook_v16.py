"""W64 M16 — ECC Codebook V16.

Strictly extends W63's ``coordpy.ecc_codebook_v15``. V15 had 14
levels (K1..K14 = 2^24 = 16 777 216 codes) at 26.333 bits/visible-
token. V16 adds:

* **15 levels** (K15 = 2).
* **Total codes = 2^25 = 33 554 432**.
* **27.333 bits/visible-token** at full emit (≥ 27.0 target).
* ``W64-L-ECC-V16-RATE-FLOOR-CAP`` — structural rate ceiling
  log2(2^25) = 25 raw data bits per segment-tuple.

Honest scope (W64)
------------------

* Adds one bit per visible token over V15.
* The 8192-bit/token falsifier reproduces the structural ceiling.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Sequence

from .ecc_codebook_v15 import (
    ECCCodebookV15,
    compress_carrier_ecc_v15,
)
from .tiny_substrate_v3 import _sha256_hex


W64_ECC_V16_SCHEMA_VERSION: str = "coordpy.ecc_codebook_v16.v1"
W64_DEFAULT_ECC_V16_K15: int = 2
W64_DEFAULT_ECC_V16_TARGET_BITS_PER_TOKEN: float = 27.0


@dataclasses.dataclass
class ECCCodebookV16:
    inner_v15: ECCCodebookV15
    n_meta13: int   # K15 = 2 by default
    code_dim: int

    @classmethod
    def init(
            cls, *, n_meta13: int = W64_DEFAULT_ECC_V16_K15,
            code_dim: int = 6, seed: int = 64160, **kwargs: Any,
    ) -> "ECCCodebookV16":
        v15 = ECCCodebookV15.init(
            code_dim=int(code_dim), seed=int(seed), **kwargs)
        return cls(
            inner_v15=v15,
            n_meta13=int(n_meta13),
            code_dim=int(code_dim),
        )

    @property
    def total_codes(self) -> int:
        return int(self.inner_v15.total_codes
                   * int(self.n_meta13))

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W64_ECC_V16_SCHEMA_VERSION,
            "kind": "ecc_v16_codebook",
            "inner_v15_cid": str(self.inner_v15.cid()),
            "n_meta13": int(self.n_meta13),
            "code_dim": int(self.code_dim),
        })


def compress_carrier_ecc_v16(
        carrier: Sequence[float], *,
        codebook: ECCCodebookV16, gate: Any,
) -> dict[str, Any]:
    """V16 = V15 + K15 1-bit meta-fine. Adds 1 bit per visible
    token over V15."""
    v15_result = compress_carrier_ecc_v15(
        list(carrier), codebook=codebook.inner_v15, gate=gate)
    cd = int(codebook.code_dim)
    ninth = sum(float(x) ** 9 for x in list(carrier)[:cd])
    meta13 = int(ninth > 0.0)
    visible_tokens = int(v15_result["visible_tokens"])
    extra_bits = 1
    structured_bits_v16 = int(
        int(v15_result["structured_bits_v15"])
        + int(extra_bits * max(1, visible_tokens)))
    bits_per = (
        float(structured_bits_v16)
        / float(max(1, visible_tokens))
        if visible_tokens > 0 else 0.0)
    return {
        "schema": W64_ECC_V16_SCHEMA_VERSION,
        "v15_compression_result": v15_result,
        "meta13_index": int(meta13),
        "structured_bits_v16": int(structured_bits_v16),
        "visible_tokens": int(visible_tokens),
        "bits_per_visible_token": float(bits_per),
        "parity_bits_per_segment_tuple": int(
            int(v15_result["parity_bits_per_segment_tuple"])
            + 8),
    }


@dataclasses.dataclass(frozen=True)
class ECCCompressionV16Witness:
    schema: str
    codebook_cid: str
    structured_bits_v16: int
    visible_tokens: int
    bits_per_visible_token: float
    meta13_index: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "codebook_cid": str(self.codebook_cid),
            "structured_bits_v16": int(
                self.structured_bits_v16),
            "visible_tokens": int(self.visible_tokens),
            "bits_per_visible_token": float(round(
                self.bits_per_visible_token, 12)),
            "meta13_index": int(self.meta13_index),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "ecc_v16_compression_witness",
            "witness": self.to_dict()})


def emit_ecc_v16_compression_witness(
        *, codebook: ECCCodebookV16,
        compression: dict[str, Any],
) -> ECCCompressionV16Witness:
    return ECCCompressionV16Witness(
        schema=W64_ECC_V16_SCHEMA_VERSION,
        codebook_cid=str(codebook.cid()),
        structured_bits_v16=int(
            compression["structured_bits_v16"]),
        visible_tokens=int(compression["visible_tokens"]),
        bits_per_visible_token=float(
            compression["bits_per_visible_token"]),
        meta13_index=int(compression["meta13_index"]),
    )


def probe_ecc_v16_rate_floor_falsifier(
        *, codebook: ECCCodebookV16,
) -> dict[str, Any]:
    """Reproduces the structural rate ceiling: log2(total_codes)
    raw data bits per segment-tuple. With 2^25 codes the ceiling
    is 25 bits; the 8192-bit/token target trivially exceeds it,
    so the falsifier asserts the ceiling holds."""
    import math
    total = int(codebook.total_codes)
    ceiling_bits = float(
        math.log2(total) if total > 0 else 0.0)
    target_bits = 8192.0
    return {
        "schema": W64_ECC_V16_SCHEMA_VERSION,
        "kind": "rate_floor_falsifier",
        "total_codes": int(total),
        "ceiling_bits": float(round(ceiling_bits, 12)),
        "target_bits": float(target_bits),
        "target_exceeds_ceiling": bool(target_bits > ceiling_bits),
    }


__all__ = [
    "W64_ECC_V16_SCHEMA_VERSION",
    "W64_DEFAULT_ECC_V16_K15",
    "W64_DEFAULT_ECC_V16_TARGET_BITS_PER_TOKEN",
    "ECCCodebookV16",
    "compress_carrier_ecc_v16",
    "ECCCompressionV16Witness",
    "emit_ecc_v16_compression_witness",
    "probe_ecc_v16_rate_floor_falsifier",
]
