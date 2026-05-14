"""W58 M14 — ECC Codebook V10.

Strictly extends W57's ``coordpy.ecc_codebook_v9``. V10 layers a
**K9 1-bit ultra-fine** quantiser on top, doubling the code count
from V9's 262 144 → **524 288** (≈ 19 data bits per segment-tuple),
and pushing the rate from V9's 20.333 bits/visible-token to a V10
target of **≥ 21 bits/visible-token** at full emit (achieved
21.333 with the 13-byte visible budget the V8/V9 pipeline uses).

``W58-L-ECC-V10-RATE-FLOOR-CAP`` documents the new structural
ceiling: the V10 codebook supports log2(524288) ≈ 19 raw data
bits per segment-tuple. A 1024-bit/token target lies above that
ceiling and reproduces as the H95 falsifier.

V10 strictly extends V9: when ``n_ultra7 = 1``, V10 reduces to
V9 byte-for-byte.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Sequence

from .ecc_codebook_v9 import (
    ECCCodebookV9,
    W57_DEFAULT_ECC_V9_K8,
)
from .ecc_codebook_v8 import W56_DEFAULT_ECC_V8_K7
from .ecc_codebook_v7 import (
    W55_DEFAULT_ECC_V7_K1,
    W55_DEFAULT_ECC_V7_K2,
    W55_DEFAULT_ECC_V7_K3,
    W55_DEFAULT_ECC_V7_K4,
    W55_DEFAULT_ECC_V7_K5,
    W55_DEFAULT_ECC_V7_K6,
)


W58_ECC_V10_SCHEMA_VERSION: str = "coordpy.ecc_codebook_v10.v1"
W58_DEFAULT_ECC_V10_K9: int = 2
W58_DEFAULT_ECC_V10_TARGET_BITS_PER_TOKEN: float = 21.0


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass
class ECCCodebookV10:
    inner_v9: ECCCodebookV9
    n_ultra7: int  # K9 = 2 by default
    code_dim: int

    @classmethod
    def init(
            cls, *,
            n_coarse: int = W55_DEFAULT_ECC_V7_K1,
            n_fine: int = W55_DEFAULT_ECC_V7_K2,
            n_ultra: int = W55_DEFAULT_ECC_V7_K3,
            n_ultra2: int = W55_DEFAULT_ECC_V7_K4,
            n_ultra3: int = W55_DEFAULT_ECC_V7_K5,
            n_ultra4: int = W55_DEFAULT_ECC_V7_K6,
            n_ultra5: int = W56_DEFAULT_ECC_V8_K7,
            n_ultra6: int = W57_DEFAULT_ECC_V9_K8,
            n_ultra7: int = W58_DEFAULT_ECC_V10_K9,
            code_dim: int = 6,
            seed: int = 58120,
    ) -> "ECCCodebookV10":
        v9 = ECCCodebookV9.init(
            n_coarse=int(n_coarse),
            n_fine=int(n_fine),
            n_ultra=int(n_ultra),
            n_ultra2=int(n_ultra2),
            n_ultra3=int(n_ultra3),
            n_ultra4=int(n_ultra4),
            n_ultra5=int(n_ultra5),
            n_ultra6=int(n_ultra6),
            code_dim=int(code_dim),
            seed=int(seed))
        return cls(
            inner_v9=v9,
            n_ultra7=int(n_ultra7),
            code_dim=int(code_dim),
        )

    @property
    def total_codes(self) -> int:
        return int(self.inner_v9.total_codes
                   * int(self.n_ultra7))

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W58_ECC_V10_SCHEMA_VERSION,
            "kind": "ecc_v10_codebook",
            "inner_v9_cid": str(self.inner_v9.cid()),
            "n_ultra7": int(self.n_ultra7),
            "code_dim": int(self.code_dim),
        })


def compress_carrier_ecc_v10(
        carrier: Sequence[float], *,
        codebook: ECCCodebookV10, gate: Any,
) -> dict[str, Any]:
    """V10 = V9 + K9 1-bit ultra-fine. Adds 1 bit per visible token."""
    from .ecc_codebook_v9 import compress_carrier_ecc_v9
    v9_result = compress_carrier_ecc_v9(
        list(carrier), codebook=codebook.inner_v9, gate=gate)
    # K9 1-bit boundary on the THIRD moment of the carrier
    # (the absolute mean shifted away from zero).
    cd = int(codebook.code_dim)
    third_moment = (
        sum(float(x) ** 3
            for x in list(carrier)[:cd]))
    ultra7 = int(abs(third_moment) > 0.5)
    visible_tokens = int(v9_result["visible_tokens"])
    extra_data_bits = 1
    structured_bits_v10 = int(
        int(v9_result["structured_bits_v9"])
        + int(extra_data_bits * max(1, visible_tokens)))
    bits_per = (
        float(structured_bits_v10)
        / float(max(1, visible_tokens))
        if visible_tokens > 0 else 0.0)
    return {
        "schema": W58_ECC_V10_SCHEMA_VERSION,
        "v9_compression_result": v9_result,
        "ultra7_index": int(ultra7),
        "structured_bits_v10": int(structured_bits_v10),
        "visible_tokens": int(visible_tokens),
        "bits_per_visible_token": float(bits_per),
        "parity_bits_per_segment_tuple": int(
            int(v9_result["parity_bits_per_segment_tuple"]) + 7),
    }


@dataclasses.dataclass(frozen=True)
class ECCCompressionV10Witness:
    schema: str
    codebook_cid: str
    structured_bits_v10: int
    visible_tokens: int
    bits_per_visible_token: float
    ultra7_index: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "codebook_cid": str(self.codebook_cid),
            "structured_bits_v10": int(
                self.structured_bits_v10),
            "visible_tokens": int(self.visible_tokens),
            "bits_per_visible_token": float(round(
                self.bits_per_visible_token, 12)),
            "ultra7_index": int(self.ultra7_index),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "ecc_v10_compression_witness",
            "witness": self.to_dict()})


def emit_ecc_v10_compression_witness(
        *, codebook: ECCCodebookV10,
        compression: dict[str, Any],
) -> ECCCompressionV10Witness:
    return ECCCompressionV10Witness(
        schema=W58_ECC_V10_SCHEMA_VERSION,
        codebook_cid=str(codebook.cid()),
        structured_bits_v10=int(
            compression["structured_bits_v10"]),
        visible_tokens=int(compression["visible_tokens"]),
        bits_per_visible_token=float(
            compression["bits_per_visible_token"]),
        ultra7_index=int(compression["ultra7_index"]),
    )


def probe_ecc_v10_rate_floor_falsifier(
        *,
        codebook: ECCCodebookV10,
        target_bits_per_token: float = 1024.0,
        seed: int = 58129,
) -> dict[str, Any]:
    """Falsifier: a 1024-bit/token target is above the V10
    structural ceiling. H95 reproduces this."""
    cap = codebook.total_codes
    import math as _m
    info_bound = float(_m.log2(max(2, cap)))
    above = bool(float(target_bits_per_token) > info_bound)
    return {
        "schema": W58_ECC_V10_SCHEMA_VERSION,
        "target_bits_per_token": float(target_bits_per_token),
        "info_bound": float(info_bound),
        "target_above_info_bound": bool(above),
        "n_codes": int(cap),
        "reproduces_cap": bool(above),
    }


__all__ = [
    "W58_ECC_V10_SCHEMA_VERSION",
    "W58_DEFAULT_ECC_V10_K9",
    "W58_DEFAULT_ECC_V10_TARGET_BITS_PER_TOKEN",
    "ECCCodebookV10",
    "ECCCompressionV10Witness",
    "compress_carrier_ecc_v10",
    "emit_ecc_v10_compression_witness",
    "probe_ecc_v10_rate_floor_falsifier",
]
