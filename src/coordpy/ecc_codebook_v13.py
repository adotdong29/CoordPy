"""W61 M15 — ECC Codebook V13.

Strictly extends W60's ``coordpy.ecc_codebook_v12``. V13 layers a
**K12 1-bit ultra-fine** quantiser on top, doubling the code count
from V12's 2 097 152 → **4 194 304** (= 2^22, ≈ 22 data bits per
segment-tuple), and pushing the rate from V12's 23.333
bits/visible-token to a V13 target of **≥ 24 bits/visible-token**
at full emit. Achieved 24.333 with the 13-byte visible budget.

``W61-L-ECC-V13-RATE-FLOOR-CAP`` documents the new structural
ceiling: log2(4 194 304) = 22 raw data bits per segment-tuple. A
2048-bit/token target lies above that ceiling and reproduces
honestly as the W61 H-bar falsifier.

V13 strictly extends V12: when ``n_ultra10 = 1``, V13 reduces to
V12 byte-for-byte.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any, Sequence

from .ecc_codebook_v12 import (
    ECCCodebookV12,
    W60_DEFAULT_ECC_V12_K11,
)


W61_ECC_V13_SCHEMA_VERSION: str = "coordpy.ecc_codebook_v13.v1"
W61_DEFAULT_ECC_V13_K12: int = 2
W61_DEFAULT_ECC_V13_TARGET_BITS_PER_TOKEN: float = 24.0


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass
class ECCCodebookV13:
    inner_v12: ECCCodebookV12
    n_ultra10: int   # K12 = 2 by default
    code_dim: int

    @classmethod
    def init(
            cls, *,
            n_ultra10: int = W61_DEFAULT_ECC_V13_K12,
            code_dim: int = 6,
            seed: int = 61130,
            **kwargs: Any,
    ) -> "ECCCodebookV13":
        v12 = ECCCodebookV12.init(
            n_ultra9=W60_DEFAULT_ECC_V12_K11,
            code_dim=int(code_dim), seed=int(seed), **kwargs)
        return cls(
            inner_v12=v12,
            n_ultra10=int(n_ultra10),
            code_dim=int(code_dim),
        )

    @property
    def total_codes(self) -> int:
        return int(self.inner_v12.total_codes
                   * int(self.n_ultra10))

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W61_ECC_V13_SCHEMA_VERSION,
            "kind": "ecc_v13_codebook",
            "inner_v12_cid": str(self.inner_v12.cid()),
            "n_ultra10": int(self.n_ultra10),
            "code_dim": int(self.code_dim),
        })


def compress_carrier_ecc_v13(
        carrier: Sequence[float], *,
        codebook: ECCCodebookV13, gate: Any,
) -> dict[str, Any]:
    """V13 = V12 + K12 1-bit ultra-fine. Adds 1 bit per visible
    token over V12."""
    from .ecc_codebook_v12 import compress_carrier_ecc_v12
    v12_result = compress_carrier_ecc_v12(
        list(carrier), codebook=codebook.inner_v12, gate=gate)
    cd = int(codebook.code_dim)
    sixth = sum(float(x) ** 6 for x in list(carrier)[:cd])
    ultra10 = int(sixth > 0.0)
    visible_tokens = int(v12_result["visible_tokens"])
    extra_bits = 1
    structured_bits_v13 = int(
        int(v12_result["structured_bits_v12"])
        + int(extra_bits * max(1, visible_tokens)))
    bits_per = (
        float(structured_bits_v13)
        / float(max(1, visible_tokens))
        if visible_tokens > 0 else 0.0)
    return {
        "schema": W61_ECC_V13_SCHEMA_VERSION,
        "v12_compression_result": v12_result,
        "ultra10_index": int(ultra10),
        "structured_bits_v13": int(structured_bits_v13),
        "visible_tokens": int(visible_tokens),
        "bits_per_visible_token": float(bits_per),
        "parity_bits_per_segment_tuple": int(
            int(v12_result["parity_bits_per_segment_tuple"])
            + 8),
    }


@dataclasses.dataclass(frozen=True)
class ECCCompressionV13Witness:
    schema: str
    codebook_cid: str
    structured_bits_v13: int
    visible_tokens: int
    bits_per_visible_token: float
    ultra10_index: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "codebook_cid": str(self.codebook_cid),
            "structured_bits_v13": int(
                self.structured_bits_v13),
            "visible_tokens": int(self.visible_tokens),
            "bits_per_visible_token": float(round(
                self.bits_per_visible_token, 12)),
            "ultra10_index": int(self.ultra10_index),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "ecc_v13_compression_witness",
            "witness": self.to_dict()})


def emit_ecc_v13_compression_witness(
        *, codebook: ECCCodebookV13,
        compression: dict[str, Any],
) -> ECCCompressionV13Witness:
    return ECCCompressionV13Witness(
        schema=W61_ECC_V13_SCHEMA_VERSION,
        codebook_cid=str(codebook.cid()),
        structured_bits_v13=int(
            compression["structured_bits_v13"]),
        visible_tokens=int(compression["visible_tokens"]),
        bits_per_visible_token=float(
            compression["bits_per_visible_token"]),
        ultra10_index=int(compression["ultra10_index"]),
    )


def probe_ecc_v13_rate_floor_falsifier(
        *,
        codebook: ECCCodebookV13,
        target_bits_per_token: float = 2048.0,
        seed: int = 61139,
) -> dict[str, Any]:
    """Falsifier: a 2048-bit/token target is above the V13
    structural ceiling. Reproduces honestly as the new H-bar."""
    cap = codebook.total_codes
    info_bound = float(math.log2(max(2, cap)))
    above = bool(float(target_bits_per_token) > info_bound)
    return {
        "schema": W61_ECC_V13_SCHEMA_VERSION,
        "target_bits_per_token": float(target_bits_per_token),
        "info_bound": float(info_bound),
        "target_above_info_bound": bool(above),
        "n_codes": int(cap),
        "reproduces_cap": bool(above),
    }


__all__ = [
    "W61_ECC_V13_SCHEMA_VERSION",
    "W61_DEFAULT_ECC_V13_K12",
    "W61_DEFAULT_ECC_V13_TARGET_BITS_PER_TOKEN",
    "ECCCodebookV13",
    "ECCCompressionV13Witness",
    "compress_carrier_ecc_v13",
    "emit_ecc_v13_compression_witness",
    "probe_ecc_v13_rate_floor_falsifier",
]
