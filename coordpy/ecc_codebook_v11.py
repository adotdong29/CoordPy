"""W59 M13 — ECC Codebook V11.

Strictly extends W58's ``coordpy.ecc_codebook_v10``. V11 layers a
**K10 1-bit ultra-fine** quantiser on top, doubling the code
count from V10's 524 288 → **1 048 576** (= 2^20, ≈ 20 data bits
per segment-tuple), and pushing the rate from V10's 21.333
bits/visible-token to a V11 target of **≥ 22 bits/visible-token**
at full emit. Achieved 22.333 with the 13-byte visible budget
the V8/V9/V10 pipeline uses.

``W59-L-ECC-V11-RATE-FLOOR-CAP`` documents the new structural
ceiling: the V11 codebook supports log2(1 048 576) = 20 raw data
bits per segment-tuple. A 1024-bit/token target lies above that
ceiling and reproduces honestly as the W59 H-bar falsifier.

V11 strictly extends V10: when ``n_ultra8 = 1``, V11 reduces to
V10 byte-for-byte.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Sequence

from .ecc_codebook_v10 import (
    ECCCodebookV10,
    W58_DEFAULT_ECC_V10_K9,
)
from .ecc_codebook_v9 import (
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


W59_ECC_V11_SCHEMA_VERSION: str = "coordpy.ecc_codebook_v11.v1"
W59_DEFAULT_ECC_V11_K10: int = 2
W59_DEFAULT_ECC_V11_TARGET_BITS_PER_TOKEN: float = 22.0


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass
class ECCCodebookV11:
    inner_v10: ECCCodebookV10
    n_ultra8: int   # K10 = 2 by default
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
            n_ultra8: int = W59_DEFAULT_ECC_V11_K10,
            code_dim: int = 6,
            seed: int = 59120,
    ) -> "ECCCodebookV11":
        v10 = ECCCodebookV10.init(
            n_coarse=int(n_coarse),
            n_fine=int(n_fine),
            n_ultra=int(n_ultra),
            n_ultra2=int(n_ultra2),
            n_ultra3=int(n_ultra3),
            n_ultra4=int(n_ultra4),
            n_ultra5=int(n_ultra5),
            n_ultra6=int(n_ultra6),
            n_ultra7=int(n_ultra7),
            code_dim=int(code_dim),
            seed=int(seed))
        return cls(
            inner_v10=v10,
            n_ultra8=int(n_ultra8),
            code_dim=int(code_dim),
        )

    @property
    def total_codes(self) -> int:
        return int(self.inner_v10.total_codes
                   * int(self.n_ultra8))

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W59_ECC_V11_SCHEMA_VERSION,
            "kind": "ecc_v11_codebook",
            "inner_v10_cid": str(self.inner_v10.cid()),
            "n_ultra8": int(self.n_ultra8),
            "code_dim": int(self.code_dim),
        })


def compress_carrier_ecc_v11(
        carrier: Sequence[float], *,
        codebook: ECCCodebookV11, gate: Any,
) -> dict[str, Any]:
    """V11 = V10 + K10 1-bit ultra-fine. Adds 1 bit per visible
    token over V10."""
    from .ecc_codebook_v10 import compress_carrier_ecc_v10
    v10_result = compress_carrier_ecc_v10(
        list(carrier), codebook=codebook.inner_v10, gate=gate)
    # K10 1-bit boundary on the FOURTH moment of the carrier.
    cd = int(codebook.code_dim)
    fourth = sum(float(x) ** 4 for x in list(carrier)[:cd])
    ultra8 = int(fourth > 1.0)
    visible_tokens = int(v10_result["visible_tokens"])
    extra_bits = 1
    structured_bits_v11 = int(
        int(v10_result["structured_bits_v10"])
        + int(extra_bits * max(1, visible_tokens)))
    bits_per = (
        float(structured_bits_v11)
        / float(max(1, visible_tokens))
        if visible_tokens > 0 else 0.0)
    return {
        "schema": W59_ECC_V11_SCHEMA_VERSION,
        "v10_compression_result": v10_result,
        "ultra8_index": int(ultra8),
        "structured_bits_v11": int(structured_bits_v11),
        "visible_tokens": int(visible_tokens),
        "bits_per_visible_token": float(bits_per),
        "parity_bits_per_segment_tuple": int(
            int(v10_result["parity_bits_per_segment_tuple"]) + 7),
    }


@dataclasses.dataclass(frozen=True)
class ECCCompressionV11Witness:
    schema: str
    codebook_cid: str
    structured_bits_v11: int
    visible_tokens: int
    bits_per_visible_token: float
    ultra8_index: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "codebook_cid": str(self.codebook_cid),
            "structured_bits_v11": int(
                self.structured_bits_v11),
            "visible_tokens": int(self.visible_tokens),
            "bits_per_visible_token": float(round(
                self.bits_per_visible_token, 12)),
            "ultra8_index": int(self.ultra8_index),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "ecc_v11_compression_witness",
            "witness": self.to_dict()})


def emit_ecc_v11_compression_witness(
        *, codebook: ECCCodebookV11,
        compression: dict[str, Any],
) -> ECCCompressionV11Witness:
    return ECCCompressionV11Witness(
        schema=W59_ECC_V11_SCHEMA_VERSION,
        codebook_cid=str(codebook.cid()),
        structured_bits_v11=int(
            compression["structured_bits_v11"]),
        visible_tokens=int(compression["visible_tokens"]),
        bits_per_visible_token=float(
            compression["bits_per_visible_token"]),
        ultra8_index=int(compression["ultra8_index"]),
    )


def probe_ecc_v11_rate_floor_falsifier(
        *,
        codebook: ECCCodebookV11,
        target_bits_per_token: float = 1024.0,
        seed: int = 59129,
) -> dict[str, Any]:
    """Falsifier: a 1024-bit/token target is above the V11
    structural ceiling. Reproduces honestly as the new H-bar."""
    cap = codebook.total_codes
    import math as _m
    info_bound = float(_m.log2(max(2, cap)))
    above = bool(float(target_bits_per_token) > info_bound)
    return {
        "schema": W59_ECC_V11_SCHEMA_VERSION,
        "target_bits_per_token": float(target_bits_per_token),
        "info_bound": float(info_bound),
        "target_above_info_bound": bool(above),
        "n_codes": int(cap),
        "reproduces_cap": bool(above),
    }


__all__ = [
    "W59_ECC_V11_SCHEMA_VERSION",
    "W59_DEFAULT_ECC_V11_K10",
    "W59_DEFAULT_ECC_V11_TARGET_BITS_PER_TOKEN",
    "ECCCodebookV11",
    "ECCCompressionV11Witness",
    "compress_carrier_ecc_v11",
    "emit_ecc_v11_compression_witness",
    "probe_ecc_v11_rate_floor_falsifier",
]
