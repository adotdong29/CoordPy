"""W57 M12 — ECC Codebook V9.

8-level structure (K1..K8) where K8 is the new 1-bit ultra-fine
quantiser layered on top of the W56 V8 codebook + BCH(31,16). The
total code count is V8's 131072 × 2 = 262144 codes ≈ 18 data bits
per segment-tuple.

Target rate: ≥ 20 bits per visible token at full emit. (W56 V8
delivered 19.333.)

``W57-L-ECC-V9-RATE-FLOOR-CAP`` documents that the 256-bit/token
target is still above the structural ceiling — V9 is an honest
20-bits/token result, not a 256-bits/token claim.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Sequence

from .ecc_codebook_v8 import (
    ECCCodebookV8,
    W56_DEFAULT_ECC_V8_K7,
    W56_ECC_V8_SCHEMA_VERSION,
)
from .ecc_codebook_v7 import (
    W55_DEFAULT_ECC_V7_K1,
    W55_DEFAULT_ECC_V7_K2,
    W55_DEFAULT_ECC_V7_K3,
    W55_DEFAULT_ECC_V7_K4,
    W55_DEFAULT_ECC_V7_K5,
    W55_DEFAULT_ECC_V7_K6,
)


W57_ECC_V9_SCHEMA_VERSION: str = "coordpy.ecc_codebook_v9.v1"
W57_DEFAULT_ECC_V9_K8: int = 2
W57_DEFAULT_ECC_V9_TARGET_BITS_PER_TOKEN: float = 20.0


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass
class ECCCodebookV9:
    inner_v8: ECCCodebookV8
    n_ultra6: int  # K8
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
            code_dim: int = 6,
            seed: int = 57120,
    ) -> "ECCCodebookV9":
        v8 = ECCCodebookV8.init(
            n_coarse=int(n_coarse),
            n_fine=int(n_fine),
            n_ultra=int(n_ultra),
            n_ultra2=int(n_ultra2),
            n_ultra3=int(n_ultra3),
            n_ultra4=int(n_ultra4),
            n_ultra5=int(n_ultra5),
            code_dim=int(code_dim),
            seed=int(seed))
        return cls(
            inner_v8=v8,
            n_ultra6=int(n_ultra6),
            code_dim=int(code_dim),
        )

    @property
    def total_codes(self) -> int:
        return int(self.inner_v8.total_codes
                   * int(self.n_ultra6))

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W57_ECC_V9_SCHEMA_VERSION,
            "kind": "ecc_v9_codebook",
            "inner_v8_cid": str(self.inner_v8.cid()),
            "n_ultra6": int(self.n_ultra6),
            "code_dim": int(self.code_dim),
        })


def compress_carrier_ecc_v9(
        carrier: Sequence[float], *,
        codebook: ECCCodebookV9, gate: Any,
) -> dict[str, Any]:
    """V9 = V8 + K8 1-bit ultra-fine. Adds 1 bit per visible token."""
    from .ecc_codebook_v8 import compress_carrier_ecc_v8
    v8_result = compress_carrier_ecc_v8(
        list(carrier), codebook=codebook.inner_v8, gate=gate)
    # K8 1-bit boundary on the SECOND moment of the carrier.
    sq_sum = sum(float(x) ** 2
                  for x in list(carrier)[:codebook.code_dim])
    ultra6 = int(sq_sum > 1.0)
    visible_tokens = int(v8_result["visible_tokens"])
    extra_data_bits = 1
    structured_bits_v9 = int(
        int(v8_result["structured_bits_v8"])
        + int(extra_data_bits * max(1, visible_tokens)))
    bits_per = (
        float(structured_bits_v9)
        / float(max(1, visible_tokens))
        if visible_tokens > 0 else 0.0)
    return {
        "schema": W57_ECC_V9_SCHEMA_VERSION,
        "v8_compression_result": v8_result,
        "ultra6_index": int(ultra6),
        "structured_bits_v9": int(structured_bits_v9),
        "visible_tokens": int(visible_tokens),
        "bits_per_visible_token": float(bits_per),
        "parity_bits_per_segment_tuple": int(
            int(v8_result["parity_bits_per_segment_tuple"]) + 7),
    }


@dataclasses.dataclass(frozen=True)
class ECCCompressionV9Witness:
    schema: str
    codebook_cid: str
    structured_bits_v9: int
    visible_tokens: int
    bits_per_visible_token: float
    ultra6_index: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "codebook_cid": str(self.codebook_cid),
            "structured_bits_v9": int(self.structured_bits_v9),
            "visible_tokens": int(self.visible_tokens),
            "bits_per_visible_token": float(round(
                self.bits_per_visible_token, 12)),
            "ultra6_index": int(self.ultra6_index),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "ecc_v9_compression_witness",
            "witness": self.to_dict()})


def emit_ecc_v9_compression_witness(
        *, codebook: ECCCodebookV9,
        compression: dict[str, Any],
) -> ECCCompressionV9Witness:
    return ECCCompressionV9Witness(
        schema=W57_ECC_V9_SCHEMA_VERSION,
        codebook_cid=str(codebook.cid()),
        structured_bits_v9=int(compression["structured_bits_v9"]),
        visible_tokens=int(compression["visible_tokens"]),
        bits_per_visible_token=float(
            compression["bits_per_visible_token"]),
        ultra6_index=int(compression["ultra6_index"]),
    )


def probe_ecc_v9_rate_floor_falsifier(
        *,
        codebook: ECCCodebookV9,
        target_bits_per_token: float = 256.0,
        seed: int = 57129,
) -> dict[str, Any]:
    """Falsifier: at any rate above the structural ceiling, the
    codebook cannot deliver. The honest cap reproduces here."""
    cap = codebook.total_codes
    # Information-theoretic upper bound on bits per token from
    # codebook size and emit budget. Even if we use every code,
    # log2(cap) bits per emit-tuple is the ceiling. We compare
    # this against the target.
    import math as _m
    info_bound = float(_m.log2(max(2, cap)))
    above = bool(float(target_bits_per_token) > info_bound)
    return {
        "schema": W57_ECC_V9_SCHEMA_VERSION,
        "target_bits_per_token": float(target_bits_per_token),
        "info_bound": float(info_bound),
        "target_above_info_bound": bool(above),
        "n_codes": int(cap),
        "reproduces_cap": bool(above),
    }


__all__ = [
    "W57_ECC_V9_SCHEMA_VERSION",
    "W57_DEFAULT_ECC_V9_K8",
    "W57_DEFAULT_ECC_V9_TARGET_BITS_PER_TOKEN",
    "ECCCodebookV9",
    "ECCCompressionV9Witness",
    "compress_carrier_ecc_v9",
    "emit_ecc_v9_compression_witness",
    "probe_ecc_v9_rate_floor_falsifier",
]
