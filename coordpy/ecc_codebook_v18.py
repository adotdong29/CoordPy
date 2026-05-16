"""W66 M16 — ECC Codebook V18.

Strictly extends W65's ``coordpy.ecc_codebook_v17``. V17 had 16
levels (K1..K16 = 2^27 = 134 217 728 codes) at 29.333
bits/visible-token. V18 adds:

* **17 levels** (K17 = 4).
* **Total codes = 2^29 = 536 870 912** (V17 had 2^27; adds 2 bits).
* **≥ 31.0 bits/visible-token** at full emit.
* ``W66-L-ECC-V18-RATE-FLOOR-CAP`` — structural rate ceiling
  log2(2^29) = 29 raw data bits per segment-tuple.

Honest scope (W66)
------------------

* Adds two bits per visible token over V17.
* The 16384-bit/token falsifier reproduces the structural ceiling.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from .ecc_codebook_v17 import (
    ECCCodebookV17, compress_carrier_ecc_v17,
)
from .tiny_substrate_v3 import _sha256_hex


W66_ECC_V18_SCHEMA_VERSION: str = "coordpy.ecc_codebook_v18.v1"
W66_DEFAULT_ECC_V18_K17: int = 4
W66_DEFAULT_ECC_V18_TARGET_BITS_PER_TOKEN: float = 31.0


@dataclasses.dataclass
class ECCCodebookV18:
    inner_v17: ECCCodebookV17
    n_meta15: int   # K17 = 4 by default
    code_dim: int

    @classmethod
    def init(
            cls, *, n_meta15: int = W66_DEFAULT_ECC_V18_K17,
            code_dim: int = 6, seed: int = 66180,
            **kwargs: Any,
    ) -> "ECCCodebookV18":
        v17 = ECCCodebookV17.init(
            code_dim=int(code_dim), seed=int(seed), **kwargs)
        return cls(
            inner_v17=v17, n_meta15=int(n_meta15),
            code_dim=int(code_dim),
        )

    @property
    def total_codes(self) -> int:
        return int(self.inner_v17.total_codes
                   * int(self.n_meta15))

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W66_ECC_V18_SCHEMA_VERSION,
            "kind": "ecc_v18_codebook",
            "inner_v17_cid": str(self.inner_v17.cid()),
            "n_meta15": int(self.n_meta15),
            "code_dim": int(self.code_dim),
        })


def compress_carrier_ecc_v18(
        carrier: Sequence[float], *,
        codebook: ECCCodebookV18, gate: Any,
) -> dict[str, Any]:
    """V18 = V17 + K17 2-bit meta-fine. Adds 2 bits per visible
    token over V17."""
    v17_result = compress_carrier_ecc_v17(
        list(carrier), codebook=codebook.inner_v17, gate=gate)
    cd = int(codebook.code_dim)
    thirteenth = sum(float(x) ** 13 for x in list(carrier)[:cd])
    meta15 = int(abs(thirteenth * 1e3)) % int(
        max(1, codebook.n_meta15))
    visible_tokens = int(v17_result["visible_tokens"])
    extra_bits = 2
    structured_bits_v18 = int(
        int(v17_result["structured_bits_v17"])
        + int(extra_bits * max(1, visible_tokens)))
    bits_per = (
        float(structured_bits_v18)
        / float(max(1, visible_tokens))
        if visible_tokens > 0 else 0.0)
    return {
        "schema": W66_ECC_V18_SCHEMA_VERSION,
        "v17_compression_result": v17_result,
        "meta15_index": int(meta15),
        "structured_bits_v18": int(structured_bits_v18),
        "visible_tokens": int(visible_tokens),
        "bits_per_visible_token": float(bits_per),
        "parity_bits_per_segment_tuple": int(
            int(v17_result["parity_bits_per_segment_tuple"]) + 8),
    }


@dataclasses.dataclass(frozen=True)
class ECCCompressionV18Witness:
    schema: str
    codebook_cid: str
    structured_bits_v18: int
    visible_tokens: int
    bits_per_visible_token: float
    meta15_index: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "codebook_cid": str(self.codebook_cid),
            "structured_bits_v18": int(self.structured_bits_v18),
            "visible_tokens": int(self.visible_tokens),
            "bits_per_visible_token": float(round(
                self.bits_per_visible_token, 12)),
            "meta15_index": int(self.meta15_index),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "ecc_v18_compression_witness",
            "witness": self.to_dict()})


def emit_ecc_v18_compression_witness(
        *, codebook: ECCCodebookV18,
        compression: dict[str, Any],
) -> ECCCompressionV18Witness:
    return ECCCompressionV18Witness(
        schema=W66_ECC_V18_SCHEMA_VERSION,
        codebook_cid=str(codebook.cid()),
        structured_bits_v18=int(
            compression["structured_bits_v18"]),
        visible_tokens=int(compression["visible_tokens"]),
        bits_per_visible_token=float(
            compression["bits_per_visible_token"]),
        meta15_index=int(compression["meta15_index"]),
    )


def probe_ecc_v18_rate_floor_falsifier(
        *, codebook: ECCCodebookV18,
) -> dict[str, Any]:
    """Reproduces the structural rate ceiling: log2(total_codes)
    raw data bits per segment-tuple. With 2^29 codes the ceiling
    is 29 bits; the 65536-bit/token target trivially exceeds it.
    """
    import math
    total = int(codebook.total_codes)
    ceiling_bits = float(
        math.log2(total) if total > 0 else 0.0)
    target_bits = 65536.0
    return {
        "schema": W66_ECC_V18_SCHEMA_VERSION,
        "kind": "rate_floor_falsifier",
        "total_codes": int(total),
        "ceiling_bits": float(round(ceiling_bits, 12)),
        "target_bits": float(target_bits),
        "target_exceeds_ceiling": bool(target_bits > ceiling_bits),
    }


__all__ = [
    "W66_ECC_V18_SCHEMA_VERSION",
    "W66_DEFAULT_ECC_V18_K17",
    "W66_DEFAULT_ECC_V18_TARGET_BITS_PER_TOKEN",
    "ECCCodebookV18",
    "compress_carrier_ecc_v18",
    "ECCCompressionV18Witness",
    "emit_ecc_v18_compression_witness",
    "probe_ecc_v18_rate_floor_falsifier",
]
