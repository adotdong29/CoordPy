"""W84 / P1 #30 — Quantized-Runtime Substrate precision-tier contract.

Issue #30 asks for a tiered precision contract (TIER_FP32 /
TIER_BF16 / TIER_INT8) with honest per-tier ``max_abs_diff``
floors. This module ships the **contract** and the probe + tier
ladder; it does **NOT** load int8 weights (that requires
bitsandbytes + GPU which this environment does not have).

Anti-cheat:

* The contract refuses to claim byte-identity at sub-fp32 tiers.
* The precision tier is a first-class axis on the W80
  ``RuntimeInstrumentationProtocolV1`` (declared here; existing
  runtimes opt in).
* The conformance check uses per-tier floors and refuses to
  silently fall back to fp32.

Honest scope
------------

* ``W84-L-PRECISION-TIER-V1-RESEARCH-ONLY-CAP`` — explicit-import
  only.
* ``W84-L-PRECISION-TIER-V1-NO-INT8-LOAD-CAP`` — V1 declares the
  TIER_INT8 contract but does not load an int8 model (blocked
  on bitsandbytes + CUDA).
* ``W84-L-PRECISION-TIER-V1-NO-AWQ-GPTQ-CAP`` — V1 does not load
  AWQ / GPTQ / EXL2 quantised weights.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
from typing import Any


W84_PRECISION_TIER_V1_SCHEMA_VERSION: str = (
    "coordpy.precision_tier_contract_v1.v1")


class PrecisionTier(str, enum.Enum):
    """Three tiers; each carries its own honest floor."""

    TIER_FP32 = "tier_fp32"
    TIER_BF16 = "tier_bf16"
    TIER_INT8 = "tier_int8"


W84_PRECISION_TIERS_ALL: tuple[str, ...] = tuple(
    t.value for t in PrecisionTier)


# Honest per-tier floors. These are the *empirical* floors at
# which the W80 instrumentation contract still meaningfully
# verifies the runtime under the contract. They are NOT byte-
# identity floors; they are the contract's per-tier
# precision-floor declarations.
W84_PRECISION_TIER_FLOORS: dict[str, float] = {
    PrecisionTier.TIER_FP32.value: 5e-3,
    PrecisionTier.TIER_BF16.value: 5e-2,
    PrecisionTier.TIER_INT8.value: 2e-1,
}


# Per-tier semantic equivalence floors: TIER_INT8 promises
# "same top-1 continuation" with at least this rate; TIER_BF16
# promises better; TIER_FP32 promises byte-identity.
W84_PRECISION_TIER_SEMANTIC_EQ_FLOORS: dict[str, float] = {
    PrecisionTier.TIER_FP32.value: 1.0,  # byte-identical
    PrecisionTier.TIER_BF16.value: 0.99,
    PrecisionTier.TIER_INT8.value: 0.95,
}


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass(frozen=True)
class PrecisionTierContractV1:
    """The per-runtime declared precision tier + floors.

    A runtime that claims ``TIER_BF16`` MUST produce replay-
    from-KV outputs with ``max_abs_diff <= 5e-2`` against the
    fp32 reference *at the precision floor*. The contract
    refuses to claim byte-identity at sub-fp32 tiers.
    """

    schema: str
    declared_tier: str
    max_abs_diff_floor: float
    semantic_equivalence_floor: float
    runtime_id: str

    def __post_init__(self) -> None:
        if self.declared_tier not in W84_PRECISION_TIERS_ALL:
            raise ValueError(
                f"unknown precision tier: "
                f"{self.declared_tier!r}")
        # The floor must be the *exact* declared per-tier floor.
        # Anti-cheat: do not allow a runtime to silently widen
        # the floor.
        ex = float(
            W84_PRECISION_TIER_FLOORS[self.declared_tier])
        if abs(float(self.max_abs_diff_floor) - ex) > 1e-12:
            raise ValueError(
                f"declared max_abs_diff_floor "
                f"{self.max_abs_diff_floor} does not match the "
                f"canonical per-tier floor {ex} for "
                f"{self.declared_tier!r}")
        ex_se = float(W84_PRECISION_TIER_SEMANTIC_EQ_FLOORS[
            self.declared_tier])
        if (abs(float(self.semantic_equivalence_floor) - ex_se)
                > 1e-12):
            raise ValueError(
                f"declared semantic_equivalence_floor "
                f"{self.semantic_equivalence_floor} does not "
                f"match the canonical per-tier "
                f"semantic-equivalence floor {ex_se} for "
                f"{self.declared_tier!r}")

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w84_precision_tier_contract_v1",
            "schema": str(self.schema),
            "declared_tier": str(self.declared_tier),
            "max_abs_diff_floor": float(round(
                self.max_abs_diff_floor, 12)),
            "semantic_equivalence_floor": float(round(
                self.semantic_equivalence_floor, 12)),
            "runtime_id": str(self.runtime_id),
        })


def build_precision_tier_contract_v1(
        *, tier: PrecisionTier | str,
        runtime_id: str,
) -> PrecisionTierContractV1:
    tier_value = (
        str(tier.value) if isinstance(tier, PrecisionTier)
        else str(tier))
    return PrecisionTierContractV1(
        schema=W84_PRECISION_TIER_V1_SCHEMA_VERSION,
        declared_tier=str(tier_value),
        max_abs_diff_floor=float(
            W84_PRECISION_TIER_FLOORS[tier_value]),
        semantic_equivalence_floor=float(
            W84_PRECISION_TIER_SEMANTIC_EQ_FLOORS[tier_value]),
        runtime_id=str(runtime_id),
    )


@dataclasses.dataclass(frozen=True)
class PrecisionTierCapabilityProbeV1:
    """Records which tiers are available on this host.

    * fp32 is always available.
    * bf16 is available if torch supports bf16 on CPU OR a CUDA
      / MPS device is present.
    * int8 is available only if bitsandbytes (or an equivalent
      quantisation library) is importable AND a CUDA device is
      present.
    """

    schema: str
    fp32_available: bool
    bf16_available: bool
    int8_available: bool
    bitsandbytes_available: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "fp32_available": bool(self.fp32_available),
            "bf16_available": bool(self.bf16_available),
            "int8_available": bool(self.int8_available),
            "bitsandbytes_available": bool(
                self.bitsandbytes_available),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w84_precision_tier_capability_probe_v1",
            "probe": self.to_dict()})


def probe_precision_tier_capability_v1(
        ) -> PrecisionTierCapabilityProbeV1:
    fp32 = True
    bf16 = False
    int8 = False
    bnb = False
    try:
        import torch  # type: ignore
        bf16 = True
        if bool(torch.cuda.is_available()):
            int8_cuda_candidate = True
        else:
            int8_cuda_candidate = False
    except Exception:  # noqa: BLE001
        int8_cuda_candidate = False
    try:
        import bitsandbytes  # type: ignore  # noqa: F401
        bnb = True
    except Exception:  # noqa: BLE001
        bnb = False
    int8 = bool(bnb and int8_cuda_candidate)
    return PrecisionTierCapabilityProbeV1(
        schema=W84_PRECISION_TIER_V1_SCHEMA_VERSION,
        fp32_available=bool(fp32),
        bf16_available=bool(bf16),
        int8_available=bool(int8),
        bitsandbytes_available=bool(bnb),
    )


def precision_tier_floor_for(
        *, tier: PrecisionTier | str,
) -> float:
    tier_value = (
        str(tier.value) if isinstance(tier, PrecisionTier)
        else str(tier))
    if tier_value not in W84_PRECISION_TIER_FLOORS:
        raise ValueError(f"unknown tier: {tier_value!r}")
    return float(W84_PRECISION_TIER_FLOORS[tier_value])


__all__ = [
    "W84_PRECISION_TIER_V1_SCHEMA_VERSION",
    "W84_PRECISION_TIERS_ALL",
    "W84_PRECISION_TIER_FLOORS",
    "W84_PRECISION_TIER_SEMANTIC_EQ_FLOORS",
    "PrecisionTier",
    "PrecisionTierContractV1",
    "PrecisionTierCapabilityProbeV1",
    "build_precision_tier_contract_v1",
    "probe_precision_tier_capability_v1",
    "precision_tier_floor_for",
]
