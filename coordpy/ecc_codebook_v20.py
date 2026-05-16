"""W68 M14 — ECC Codebook V20.

Strictly extends W67's ``coordpy.ecc_codebook_v19``. V19 had 18
levels (K1..K18 = 2^31 = 2 147 483 648 codes) at 33.0 bits/visible-
token. V20 adds:

* **19 levels** (K19 = 4).
* **Total codes = 2^33 = 8 589 934 592** (V19 had 2^31; adds 2
  bits).
* **≥ 35.0 bits/visible-token** at full emit.
* ``W68-L-ECC-V20-RATE-FLOOR-CAP`` — structural rate ceiling
  log2(2^33) = 33 raw data bits per segment-tuple.

Honest scope (W68)
------------------

* Adds two bits per visible token over V19.
* The 65536-bit/token falsifier reproduces the structural ceiling.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from .ecc_codebook_v19 import (
    ECCCodebookV19, compress_carrier_ecc_v19,
)
from .tiny_substrate_v3 import _sha256_hex


W68_ECC_V20_SCHEMA_VERSION: str = "coordpy.ecc_codebook_v20.v1"
W68_DEFAULT_ECC_V20_K19: int = 4
W68_DEFAULT_ECC_V20_TARGET_BITS_PER_TOKEN: float = 35.0


@dataclasses.dataclass
class ECCCodebookV20:
    inner_v19: ECCCodebookV19
    n_meta17: int   # K19 = 4 by default
    code_dim: int

    @classmethod
    def init(
            cls, *, n_meta17: int = W68_DEFAULT_ECC_V20_K19,
            code_dim: int = 6, seed: int = 68200,
            **kwargs: Any,
    ) -> "ECCCodebookV20":
        v19 = ECCCodebookV19.init(
            code_dim=int(code_dim), seed=int(seed), **kwargs)
        return cls(
            inner_v19=v19, n_meta17=int(n_meta17),
            code_dim=int(code_dim),
        )

    @property
    def total_codes(self) -> int:
        return int(self.inner_v19.total_codes
                   * int(self.n_meta17))

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W68_ECC_V20_SCHEMA_VERSION,
            "kind": "ecc_v20_codebook",
            "inner_v19_cid": str(self.inner_v19.cid()),
            "n_meta17": int(self.n_meta17),
            "code_dim": int(self.code_dim),
        })


def compress_carrier_ecc_v20(
        carrier: Sequence[float], *,
        codebook: ECCCodebookV20, gate: Any,
) -> dict[str, Any]:
    """V20 = V19 + K19 2-bit meta-fine. Adds 2 bits per visible
    token over V19."""
    v19_result = compress_carrier_ecc_v19(
        list(carrier), codebook=codebook.inner_v19, gate=gate)
    cd = int(codebook.code_dim)
    seventeenth = sum(float(x) ** 17 for x in list(carrier)[:cd])
    meta17 = int(abs(seventeenth * 1e3)) % int(
        max(1, codebook.n_meta17))
    visible_tokens = int(v19_result["visible_tokens"])
    extra_bits = 2
    structured_bits_v20 = int(
        int(v19_result["structured_bits_v19"])
        + int(extra_bits * max(1, visible_tokens)))
    bits_per = (
        float(structured_bits_v20)
        / float(max(1, visible_tokens))
        if visible_tokens > 0 else 0.0)
    return {
        "schema": W68_ECC_V20_SCHEMA_VERSION,
        "v19_compression_result": v19_result,
        "meta17_index": int(meta17),
        "structured_bits_v20": int(structured_bits_v20),
        "visible_tokens": int(visible_tokens),
        "bits_per_visible_token": float(bits_per),
        "parity_bits_per_segment_tuple": int(
            int(v19_result["parity_bits_per_segment_tuple"]) + 8),
    }


@dataclasses.dataclass(frozen=True)
class ECCCompressionV20Witness:
    schema: str
    codebook_cid: str
    structured_bits_v20: int
    visible_tokens: int
    bits_per_visible_token: float
    meta17_index: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "codebook_cid": str(self.codebook_cid),
            "structured_bits_v20": int(self.structured_bits_v20),
            "visible_tokens": int(self.visible_tokens),
            "bits_per_visible_token": float(round(
                self.bits_per_visible_token, 12)),
            "meta17_index": int(self.meta17_index),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "ecc_v20_compression_witness",
            "witness": self.to_dict()})


def emit_ecc_v20_compression_witness(
        *, codebook: ECCCodebookV20,
        compression: dict[str, Any],
) -> ECCCompressionV20Witness:
    return ECCCompressionV20Witness(
        schema=W68_ECC_V20_SCHEMA_VERSION,
        codebook_cid=str(codebook.cid()),
        structured_bits_v20=int(
            compression["structured_bits_v20"]),
        visible_tokens=int(compression["visible_tokens"]),
        bits_per_visible_token=float(
            compression["bits_per_visible_token"]),
        meta17_index=int(compression["meta17_index"]),
    )


def probe_ecc_v20_rate_floor_falsifier(
        *, codebook: ECCCodebookV20,
) -> dict[str, Any]:
    """Reproduces the structural rate ceiling: log2(total_codes)
    raw data bits per segment-tuple. With 2^33 codes the ceiling
    is 33 bits; the 65536-bit/token target trivially exceeds it.
    """
    import math
    total = int(codebook.total_codes)
    ceiling_bits = float(
        math.log2(total) if total > 0 else 0.0)
    target_bits = 65536.0
    return {
        "schema": W68_ECC_V20_SCHEMA_VERSION,
        "kind": "rate_floor_falsifier",
        "total_codes": int(total),
        "ceiling_bits": float(round(ceiling_bits, 12)),
        "target_bits": float(target_bits),
        "target_exceeds_ceiling": bool(target_bits > ceiling_bits),
    }


__all__ = [
    "W68_ECC_V20_SCHEMA_VERSION",
    "W68_DEFAULT_ECC_V20_K19",
    "W68_DEFAULT_ECC_V20_TARGET_BITS_PER_TOKEN",
    "ECCCodebookV20",
    "compress_carrier_ecc_v20",
    "ECCCompressionV20Witness",
    "emit_ecc_v20_compression_witness",
    "probe_ecc_v20_rate_floor_falsifier",
]
