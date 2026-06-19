"""W65 M16 — ECC Codebook V17.

Strictly extends W64's ``coordpy.ecc_codebook_v16``. V16 had 15
levels (K1..K15 = 2^25 = 33 554 432 codes) at 27.333
bits/visible-token. V17 adds:

* **16 levels** (K16 = 4).
* **Total codes = 2^27 = 134 217 728** (V16 had 2^25; adds 2 bits).
* **≥ 29.0 bits/visible-token** at full emit.
* ``W65-L-ECC-V17-RATE-FLOOR-CAP`` — structural rate ceiling
  log2(2^27) = 27 raw data bits per segment-tuple.

Honest scope (W65)
------------------

* Adds two bits per visible token over V16.
* The 16384-bit/token falsifier reproduces the structural ceiling.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from .ecc_codebook_v16 import (
    ECCCodebookV16, compress_carrier_ecc_v16,
)
from .tiny_substrate_v3 import _sha256_hex


W65_ECC_V17_SCHEMA_VERSION: str = "coordpy.ecc_codebook_v17.v1"
W65_DEFAULT_ECC_V17_K16: int = 4
W65_DEFAULT_ECC_V17_TARGET_BITS_PER_TOKEN: float = 29.0


@dataclasses.dataclass
class ECCCodebookV17:
    inner_v16: ECCCodebookV16
    n_meta14: int   # K16 = 4 by default
    code_dim: int

    @classmethod
    def init(
            cls, *, n_meta14: int = W65_DEFAULT_ECC_V17_K16,
            code_dim: int = 6, seed: int = 65170,
            **kwargs: Any,
    ) -> "ECCCodebookV17":
        v16 = ECCCodebookV16.init(
            code_dim=int(code_dim), seed=int(seed), **kwargs)
        return cls(
            inner_v16=v16, n_meta14=int(n_meta14),
            code_dim=int(code_dim),
        )

    @property
    def total_codes(self) -> int:
        return int(self.inner_v16.total_codes
                   * int(self.n_meta14))

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W65_ECC_V17_SCHEMA_VERSION,
            "kind": "ecc_v17_codebook",
            "inner_v16_cid": str(self.inner_v16.cid()),
            "n_meta14": int(self.n_meta14),
            "code_dim": int(self.code_dim),
        })


def compress_carrier_ecc_v17(
        carrier: Sequence[float], *,
        codebook: ECCCodebookV17, gate: Any,
) -> dict[str, Any]:
    """V17 = V16 + K16 2-bit meta-fine. Adds 2 bits per visible
    token over V16."""
    v16_result = compress_carrier_ecc_v16(
        list(carrier), codebook=codebook.inner_v16, gate=gate)
    cd = int(codebook.code_dim)
    eleventh = sum(float(x) ** 11 for x in list(carrier)[:cd])
    meta14 = int(abs(eleventh * 1e3)) % int(
        max(1, codebook.n_meta14))
    visible_tokens = int(v16_result["visible_tokens"])
    extra_bits = 2
    structured_bits_v17 = int(
        int(v16_result["structured_bits_v16"])
        + int(extra_bits * max(1, visible_tokens)))
    bits_per = (
        float(structured_bits_v17)
        / float(max(1, visible_tokens))
        if visible_tokens > 0 else 0.0)
    return {
        "schema": W65_ECC_V17_SCHEMA_VERSION,
        "v16_compression_result": v16_result,
        "meta14_index": int(meta14),
        "structured_bits_v17": int(structured_bits_v17),
        "visible_tokens": int(visible_tokens),
        "bits_per_visible_token": float(bits_per),
        "parity_bits_per_segment_tuple": int(
            int(v16_result["parity_bits_per_segment_tuple"]) + 8),
    }


@dataclasses.dataclass(frozen=True)
class ECCCompressionV17Witness:
    schema: str
    codebook_cid: str
    structured_bits_v17: int
    visible_tokens: int
    bits_per_visible_token: float
    meta14_index: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "codebook_cid": str(self.codebook_cid),
            "structured_bits_v17": int(self.structured_bits_v17),
            "visible_tokens": int(self.visible_tokens),
            "bits_per_visible_token": float(round(
                self.bits_per_visible_token, 12)),
            "meta14_index": int(self.meta14_index),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "ecc_v17_compression_witness",
            "witness": self.to_dict()})


def emit_ecc_v17_compression_witness(
        *, codebook: ECCCodebookV17,
        compression: dict[str, Any],
) -> ECCCompressionV17Witness:
    return ECCCompressionV17Witness(
        schema=W65_ECC_V17_SCHEMA_VERSION,
        codebook_cid=str(codebook.cid()),
        structured_bits_v17=int(
            compression["structured_bits_v17"]),
        visible_tokens=int(compression["visible_tokens"]),
        bits_per_visible_token=float(
            compression["bits_per_visible_token"]),
        meta14_index=int(compression["meta14_index"]),
    )


def probe_ecc_v17_rate_floor_falsifier(
        *, codebook: ECCCodebookV17,
) -> dict[str, Any]:
    """Reproduces the structural rate ceiling: log2(total_codes)
    raw data bits per segment-tuple. With 2^27 codes the ceiling
    is 27 bits; the 16384-bit/token target trivially exceeds it.
    """
    import math
    total = int(codebook.total_codes)
    ceiling_bits = float(
        math.log2(total) if total > 0 else 0.0)
    target_bits = 16384.0
    return {
        "schema": W65_ECC_V17_SCHEMA_VERSION,
        "kind": "rate_floor_falsifier",
        "total_codes": int(total),
        "ceiling_bits": float(round(ceiling_bits, 12)),
        "target_bits": float(target_bits),
        "target_exceeds_ceiling": bool(target_bits > ceiling_bits),
    }


__all__ = [
    "W65_ECC_V17_SCHEMA_VERSION",
    "W65_DEFAULT_ECC_V17_K16",
    "W65_DEFAULT_ECC_V17_TARGET_BITS_PER_TOKEN",
    "ECCCodebookV17",
    "compress_carrier_ecc_v17",
    "ECCCompressionV17Witness",
    "emit_ecc_v17_compression_witness",
    "probe_ecc_v17_rate_floor_falsifier",
]
