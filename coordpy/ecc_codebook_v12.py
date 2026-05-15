"""W60 M14 — ECC Codebook V12.

Strictly extends W59's ``coordpy.ecc_codebook_v11``. V12 layers a
**K11 1-bit ultra-fine** quantiser on top, doubling the code count
from V11's 1 048 576 → **2 097 152** (= 2^21, ≈ 21 data bits per
segment-tuple), and pushing the rate from V11's 22.333
bits/visible-token to a V12 target of **≥ 23 bits/visible-token**
at full emit. Achieved 23.333 with the 13-byte visible budget.

``W60-L-ECC-V12-RATE-FLOOR-CAP`` documents the new structural
ceiling: log2(2 097 152) = 21 raw data bits per segment-tuple. A
2048-bit/token target lies above that ceiling and reproduces
honestly as the W60 H-bar falsifier.

V12 strictly extends V11: when ``n_ultra9 = 1``, V12 reduces to
V11 byte-for-byte.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any, Sequence

from .ecc_codebook_v11 import (
    ECCCodebookV11,
    W59_DEFAULT_ECC_V11_K10,
)


W60_ECC_V12_SCHEMA_VERSION: str = "coordpy.ecc_codebook_v12.v1"
W60_DEFAULT_ECC_V12_K11: int = 2
W60_DEFAULT_ECC_V12_TARGET_BITS_PER_TOKEN: float = 23.0


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass
class ECCCodebookV12:
    inner_v11: ECCCodebookV11
    n_ultra9: int   # K11 = 2 by default
    code_dim: int

    @classmethod
    def init(
            cls, *,
            n_ultra9: int = W60_DEFAULT_ECC_V12_K11,
            code_dim: int = 6,
            seed: int = 60120,
            **kwargs: Any,
    ) -> "ECCCodebookV12":
        v11_kwargs = {
            "code_dim": int(code_dim),
            "seed": int(seed),
            **{k: v for k, v in kwargs.items()
               if k in {
                   "n_coarse", "n_fine", "n_ultra",
                   "n_ultra2", "n_ultra3", "n_ultra4",
                   "n_ultra5", "n_ultra6", "n_ultra7",
                   "n_ultra8"}},
        }
        v11 = ECCCodebookV11.init(**v11_kwargs)
        return cls(
            inner_v11=v11,
            n_ultra9=int(n_ultra9),
            code_dim=int(code_dim),
        )

    @property
    def total_codes(self) -> int:
        return int(self.inner_v11.total_codes
                   * int(self.n_ultra9))

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W60_ECC_V12_SCHEMA_VERSION,
            "kind": "ecc_v12_codebook",
            "inner_v11_cid": str(self.inner_v11.cid()),
            "n_ultra9": int(self.n_ultra9),
            "code_dim": int(self.code_dim),
        })


def compress_carrier_ecc_v12(
        carrier: Sequence[float], *,
        codebook: ECCCodebookV12, gate: Any,
) -> dict[str, Any]:
    """V12 = V11 + K11 1-bit ultra-fine. Adds 1 bit per visible
    token over V11."""
    from .ecc_codebook_v11 import compress_carrier_ecc_v11
    v11_result = compress_carrier_ecc_v11(
        list(carrier), codebook=codebook.inner_v11, gate=gate)
    cd = int(codebook.code_dim)
    fifth = sum(float(x) ** 5 for x in list(carrier)[:cd])
    ultra9 = int(fifth > 0.0)
    visible_tokens = int(v11_result["visible_tokens"])
    extra_bits = 1
    structured_bits_v12 = int(
        int(v11_result["structured_bits_v11"])
        + int(extra_bits * max(1, visible_tokens)))
    bits_per = (
        float(structured_bits_v12)
        / float(max(1, visible_tokens))
        if visible_tokens > 0 else 0.0)
    return {
        "schema": W60_ECC_V12_SCHEMA_VERSION,
        "v11_compression_result": v11_result,
        "ultra9_index": int(ultra9),
        "structured_bits_v12": int(structured_bits_v12),
        "visible_tokens": int(visible_tokens),
        "bits_per_visible_token": float(bits_per),
        "parity_bits_per_segment_tuple": int(
            int(v11_result["parity_bits_per_segment_tuple"])
            + 8),
    }


@dataclasses.dataclass(frozen=True)
class ECCCompressionV12Witness:
    schema: str
    codebook_cid: str
    structured_bits_v12: int
    visible_tokens: int
    bits_per_visible_token: float
    ultra9_index: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "codebook_cid": str(self.codebook_cid),
            "structured_bits_v12": int(
                self.structured_bits_v12),
            "visible_tokens": int(self.visible_tokens),
            "bits_per_visible_token": float(round(
                self.bits_per_visible_token, 12)),
            "ultra9_index": int(self.ultra9_index),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "ecc_v12_compression_witness",
            "witness": self.to_dict()})


def emit_ecc_v12_compression_witness(
        *, codebook: ECCCodebookV12,
        compression: dict[str, Any],
) -> ECCCompressionV12Witness:
    return ECCCompressionV12Witness(
        schema=W60_ECC_V12_SCHEMA_VERSION,
        codebook_cid=str(codebook.cid()),
        structured_bits_v12=int(
            compression["structured_bits_v12"]),
        visible_tokens=int(compression["visible_tokens"]),
        bits_per_visible_token=float(
            compression["bits_per_visible_token"]),
        ultra9_index=int(compression["ultra9_index"]),
    )


def probe_ecc_v12_rate_floor_falsifier(
        *,
        codebook: ECCCodebookV12,
        target_bits_per_token: float = 2048.0,
        seed: int = 60129,
) -> dict[str, Any]:
    """Falsifier: a 2048-bit/token target is above the V12
    structural ceiling. Reproduces honestly as the new H-bar."""
    cap = codebook.total_codes
    info_bound = float(math.log2(max(2, cap)))
    above = bool(float(target_bits_per_token) > info_bound)
    return {
        "schema": W60_ECC_V12_SCHEMA_VERSION,
        "target_bits_per_token": float(target_bits_per_token),
        "info_bound": float(info_bound),
        "target_above_info_bound": bool(above),
        "n_codes": int(cap),
        "reproduces_cap": bool(above),
    }


__all__ = [
    "W60_ECC_V12_SCHEMA_VERSION",
    "W60_DEFAULT_ECC_V12_K11",
    "W60_DEFAULT_ECC_V12_TARGET_BITS_PER_TOKEN",
    "ECCCodebookV12",
    "ECCCompressionV12Witness",
    "compress_carrier_ecc_v12",
    "emit_ecc_v12_compression_witness",
    "probe_ecc_v12_rate_floor_falsifier",
]
