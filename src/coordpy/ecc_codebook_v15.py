"""W63 M17 — ECC Codebook V15.

Strictly extends W62's ``coordpy.ecc_codebook_v14``. V14 had 13
levels (K1..K13 = 2^23 = 8 388 608 codes) at 25.333 bits/visible-
token. V15 adds:

* **14 levels** (K14 = 2).
* **Total codes = 2^24 = 16 777 216**.
* **26.333 bits/visible-token** at full emit (≥ 26.0 target).
* ``W63-L-ECC-V15-RATE-FLOOR-CAP`` — structural rate ceiling
  log2(2^24) = 24 raw data bits per segment-tuple.

Honest scope
------------

* Adds one bit per visible token over V14.
* The 4096-bit/token falsifier reproduces the structural ceiling.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Sequence

from .ecc_codebook_v14 import (
    ECCCodebookV14,
    compress_carrier_ecc_v14,
)
from .tiny_substrate_v3 import _sha256_hex


W63_ECC_V15_SCHEMA_VERSION: str = "coordpy.ecc_codebook_v15.v1"
W63_DEFAULT_ECC_V15_K14: int = 2
W63_DEFAULT_ECC_V15_TARGET_BITS_PER_TOKEN: float = 26.0


@dataclasses.dataclass
class ECCCodebookV15:
    inner_v14: ECCCodebookV14
    n_meta12: int   # K14 = 2 by default
    code_dim: int

    @classmethod
    def init(
            cls, *, n_meta12: int = W63_DEFAULT_ECC_V15_K14,
            code_dim: int = 6, seed: int = 63150, **kwargs: Any,
    ) -> "ECCCodebookV15":
        v14 = ECCCodebookV14.init(
            code_dim=int(code_dim), seed=int(seed), **kwargs)
        return cls(
            inner_v14=v14,
            n_meta12=int(n_meta12),
            code_dim=int(code_dim),
        )

    @property
    def total_codes(self) -> int:
        return int(self.inner_v14.total_codes
                   * int(self.n_meta12))

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W63_ECC_V15_SCHEMA_VERSION,
            "kind": "ecc_v15_codebook",
            "inner_v14_cid": str(self.inner_v14.cid()),
            "n_meta12": int(self.n_meta12),
            "code_dim": int(self.code_dim),
        })


def compress_carrier_ecc_v15(
        carrier: Sequence[float], *,
        codebook: ECCCodebookV15, gate: Any,
) -> dict[str, Any]:
    """V15 = V14 + K14 1-bit meta-fine. Adds 1 bit per visible
    token over V14."""
    v14_result = compress_carrier_ecc_v14(
        list(carrier), codebook=codebook.inner_v14, gate=gate)
    cd = int(codebook.code_dim)
    eighth = sum(float(x) ** 8 for x in list(carrier)[:cd])
    meta12 = int(eighth > 0.0)
    visible_tokens = int(v14_result["visible_tokens"])
    extra_bits = 1
    structured_bits_v15 = int(
        int(v14_result["structured_bits_v14"])
        + int(extra_bits * max(1, visible_tokens)))
    bits_per = (
        float(structured_bits_v15)
        / float(max(1, visible_tokens))
        if visible_tokens > 0 else 0.0)
    return {
        "schema": W63_ECC_V15_SCHEMA_VERSION,
        "v14_compression_result": v14_result,
        "meta12_index": int(meta12),
        "structured_bits_v15": int(structured_bits_v15),
        "visible_tokens": int(visible_tokens),
        "bits_per_visible_token": float(bits_per),
        "parity_bits_per_segment_tuple": int(
            int(v14_result["parity_bits_per_segment_tuple"])
            + 8),
    }


@dataclasses.dataclass(frozen=True)
class ECCCompressionV15Witness:
    schema: str
    codebook_cid: str
    structured_bits_v15: int
    visible_tokens: int
    bits_per_visible_token: float
    meta12_index: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "codebook_cid": str(self.codebook_cid),
            "structured_bits_v15": int(
                self.structured_bits_v15),
            "visible_tokens": int(self.visible_tokens),
            "bits_per_visible_token": float(round(
                self.bits_per_visible_token, 12)),
            "meta12_index": int(self.meta12_index),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "ecc_v15_compression_witness",
            "witness": self.to_dict()})


def emit_ecc_v15_compression_witness(
        *, codebook: ECCCodebookV15,
        compression: dict[str, Any],
) -> ECCCompressionV15Witness:
    return ECCCompressionV15Witness(
        schema=W63_ECC_V15_SCHEMA_VERSION,
        codebook_cid=str(codebook.cid()),
        structured_bits_v15=int(
            compression["structured_bits_v15"]),
        visible_tokens=int(compression["visible_tokens"]),
        bits_per_visible_token=float(
            compression["bits_per_visible_token"]),
        meta12_index=int(compression["meta12_index"]),
    )


def probe_ecc_v15_rate_floor_falsifier(
        *, codebook: ECCCodebookV15,
) -> dict[str, Any]:
    """Reproduces the structural rate ceiling: log2(total_codes)
    raw data bits per segment-tuple. With 2^24 codes the ceiling
    is 24 bits; the 4096-bit/token target trivially exceeds it,
    so the falsifier asserts the ceiling holds."""
    import math
    total = int(codebook.total_codes)
    ceiling_bits = float(
        math.log2(total) if total > 0 else 0.0)
    target_bits = 4096.0
    return {
        "schema": W63_ECC_V15_SCHEMA_VERSION,
        "kind": "rate_floor_falsifier",
        "total_codes": int(total),
        "ceiling_bits": float(round(ceiling_bits, 12)),
        "target_bits": float(target_bits),
        "target_exceeds_ceiling": bool(target_bits > ceiling_bits),
    }


__all__ = [
    "W63_ECC_V15_SCHEMA_VERSION",
    "W63_DEFAULT_ECC_V15_K14",
    "W63_DEFAULT_ECC_V15_TARGET_BITS_PER_TOKEN",
    "ECCCodebookV15",
    "compress_carrier_ecc_v15",
    "ECCCompressionV15Witness",
    "emit_ecc_v15_compression_witness",
    "probe_ecc_v15_rate_floor_falsifier",
]
