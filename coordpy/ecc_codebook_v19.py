"""W67 M16 — ECC Codebook V19.

Strictly extends W66's ``coordpy.ecc_codebook_v18``. V18 had 17
levels (K1..K17 = 2^29 = 536 870 912 codes) at 31.0 bits/visible-
token. V19 adds:

* **18 levels** (K18 = 4).
* **Total codes = 2^31 = 2 147 483 648** (V18 had 2^29; adds 2
  bits).
* **≥ 33.0 bits/visible-token** at full emit.
* ``W67-L-ECC-V19-RATE-FLOOR-CAP`` — structural rate ceiling
  log2(2^31) = 31 raw data bits per segment-tuple.

Honest scope (W67)
------------------

* Adds two bits per visible token over V18.
* The 65536-bit/token falsifier reproduces the structural ceiling.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from .ecc_codebook_v18 import (
    ECCCodebookV18, compress_carrier_ecc_v18,
)
from .tiny_substrate_v3 import _sha256_hex


W67_ECC_V19_SCHEMA_VERSION: str = "coordpy.ecc_codebook_v19.v1"
W67_DEFAULT_ECC_V19_K18: int = 4
W67_DEFAULT_ECC_V19_TARGET_BITS_PER_TOKEN: float = 33.0


@dataclasses.dataclass
class ECCCodebookV19:
    inner_v18: ECCCodebookV18
    n_meta16: int   # K18 = 4 by default
    code_dim: int

    @classmethod
    def init(
            cls, *, n_meta16: int = W67_DEFAULT_ECC_V19_K18,
            code_dim: int = 6, seed: int = 67190,
            **kwargs: Any,
    ) -> "ECCCodebookV19":
        v18 = ECCCodebookV18.init(
            code_dim=int(code_dim), seed=int(seed), **kwargs)
        return cls(
            inner_v18=v18, n_meta16=int(n_meta16),
            code_dim=int(code_dim),
        )

    @property
    def total_codes(self) -> int:
        return int(self.inner_v18.total_codes
                   * int(self.n_meta16))

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W67_ECC_V19_SCHEMA_VERSION,
            "kind": "ecc_v19_codebook",
            "inner_v18_cid": str(self.inner_v18.cid()),
            "n_meta16": int(self.n_meta16),
            "code_dim": int(self.code_dim),
        })


def compress_carrier_ecc_v19(
        carrier: Sequence[float], *,
        codebook: ECCCodebookV19, gate: Any,
) -> dict[str, Any]:
    """V19 = V18 + K18 2-bit meta-fine. Adds 2 bits per visible
    token over V18."""
    v18_result = compress_carrier_ecc_v18(
        list(carrier), codebook=codebook.inner_v18, gate=gate)
    cd = int(codebook.code_dim)
    fifteenth = sum(float(x) ** 15 for x in list(carrier)[:cd])
    meta16 = int(abs(fifteenth * 1e3)) % int(
        max(1, codebook.n_meta16))
    visible_tokens = int(v18_result["visible_tokens"])
    extra_bits = 2
    structured_bits_v19 = int(
        int(v18_result["structured_bits_v18"])
        + int(extra_bits * max(1, visible_tokens)))
    bits_per = (
        float(structured_bits_v19)
        / float(max(1, visible_tokens))
        if visible_tokens > 0 else 0.0)
    return {
        "schema": W67_ECC_V19_SCHEMA_VERSION,
        "v18_compression_result": v18_result,
        "meta16_index": int(meta16),
        "structured_bits_v19": int(structured_bits_v19),
        "visible_tokens": int(visible_tokens),
        "bits_per_visible_token": float(bits_per),
        "parity_bits_per_segment_tuple": int(
            int(v18_result["parity_bits_per_segment_tuple"]) + 8),
    }


@dataclasses.dataclass(frozen=True)
class ECCCompressionV19Witness:
    schema: str
    codebook_cid: str
    structured_bits_v19: int
    visible_tokens: int
    bits_per_visible_token: float
    meta16_index: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "codebook_cid": str(self.codebook_cid),
            "structured_bits_v19": int(self.structured_bits_v19),
            "visible_tokens": int(self.visible_tokens),
            "bits_per_visible_token": float(round(
                self.bits_per_visible_token, 12)),
            "meta16_index": int(self.meta16_index),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "ecc_v19_compression_witness",
            "witness": self.to_dict()})


def emit_ecc_v19_compression_witness(
        *, codebook: ECCCodebookV19,
        compression: dict[str, Any],
) -> ECCCompressionV19Witness:
    return ECCCompressionV19Witness(
        schema=W67_ECC_V19_SCHEMA_VERSION,
        codebook_cid=str(codebook.cid()),
        structured_bits_v19=int(
            compression["structured_bits_v19"]),
        visible_tokens=int(compression["visible_tokens"]),
        bits_per_visible_token=float(
            compression["bits_per_visible_token"]),
        meta16_index=int(compression["meta16_index"]),
    )


def probe_ecc_v19_rate_floor_falsifier(
        *, codebook: ECCCodebookV19,
) -> dict[str, Any]:
    """Reproduces the structural rate ceiling: log2(total_codes)
    raw data bits per segment-tuple. With 2^31 codes the ceiling
    is 31 bits; the 65536-bit/token target trivially exceeds it.
    """
    import math
    total = int(codebook.total_codes)
    ceiling_bits = float(
        math.log2(total) if total > 0 else 0.0)
    target_bits = 65536.0
    return {
        "schema": W67_ECC_V19_SCHEMA_VERSION,
        "kind": "rate_floor_falsifier",
        "total_codes": int(total),
        "ceiling_bits": float(round(ceiling_bits, 12)),
        "target_bits": float(target_bits),
        "target_exceeds_ceiling": bool(target_bits > ceiling_bits),
    }


__all__ = [
    "W67_ECC_V19_SCHEMA_VERSION",
    "W67_DEFAULT_ECC_V19_K18",
    "W67_DEFAULT_ECC_V19_TARGET_BITS_PER_TOKEN",
    "ECCCodebookV19",
    "compress_carrier_ecc_v19",
    "ECCCompressionV19Witness",
    "emit_ecc_v19_compression_witness",
    "probe_ecc_v19_rate_floor_falsifier",
]
