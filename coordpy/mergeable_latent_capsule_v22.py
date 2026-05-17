"""W74 — Mergeable Latent State Capsule V22 (MLSC V22).

Strictly extends W73's ``coordpy.mergeable_latent_capsule_v21``.
V21 added replacement-repair-trajectory and contradiction chains.
V22 adds:

* ``compound_repair_trajectory_chain`` — content-addressed witness
  chain for per-turn compound-repair-trajectory CIDs (V19 axis).
* ``delayed_repair_chain`` — content-addressed witness chain for
  per-turn delayed-repair events.
* ``algebra_signature_v22`` — adds two new V22 propagation
  signatures.

V22 merge inherits the unions of both new chains.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from .mergeable_latent_capsule_v4 import W56_MLSC_V4_ALGEBRA_MERGE
from .mergeable_latent_capsule_v21 import (
    MergeableLatentCapsuleV21, MergeOperatorV21,
)
from .tiny_substrate_v3 import _sha256_hex


W74_MLSC_V22_SCHEMA_VERSION: str = (
    "coordpy.mergeable_latent_capsule_v22.v1")
W74_MLSC_V22_ALGEBRA_COMPOUND_REPAIR_PROPAGATION: str = (
    "compound_repair_propagation_v22")
W74_MLSC_V22_ALGEBRA_DELAYED_REPAIR_PROPAGATION: str = (
    "delayed_repair_propagation_v22")
W74_MLSC_V22_KNOWN_ALGEBRA_SIGNATURES: tuple[str, ...] = (
    W56_MLSC_V4_ALGEBRA_MERGE,
    W74_MLSC_V22_ALGEBRA_COMPOUND_REPAIR_PROPAGATION,
    W74_MLSC_V22_ALGEBRA_DELAYED_REPAIR_PROPAGATION,
)


@dataclasses.dataclass(frozen=True)
class MergeableLatentCapsuleV22:
    schema: str
    inner_v21: MergeableLatentCapsuleV21
    compound_repair_trajectory_chain: tuple[str, ...]
    delayed_repair_chain: tuple[str, ...]
    algebra_signature_v22: str

    @property
    def payload(self) -> tuple[float, ...]:
        return self.inner_v21.payload

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "inner_v21_cid": str(self.inner_v21.cid()),
            "compound_repair_trajectory_chain": list(
                self.compound_repair_trajectory_chain),
            "delayed_repair_chain": list(
                self.delayed_repair_chain),
            "algebra_signature_v22": str(
                self.algebra_signature_v22),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w74_mlsc_v22",
            "capsule": self.to_dict()})


def wrap_v21_as_v22(
        v21_capsule: MergeableLatentCapsuleV21, *,
        compound_repair_trajectory_chain: Sequence[str] = (),
        delayed_repair_chain: Sequence[str] = (),
        algebra_signature_v22: str = W56_MLSC_V4_ALGEBRA_MERGE,
) -> MergeableLatentCapsuleV22:
    if (algebra_signature_v22 not in
            W74_MLSC_V22_KNOWN_ALGEBRA_SIGNATURES):
        algebra_signature_v22 = W56_MLSC_V4_ALGEBRA_MERGE
    return MergeableLatentCapsuleV22(
        schema=W74_MLSC_V22_SCHEMA_VERSION,
        inner_v21=v21_capsule,
        compound_repair_trajectory_chain=tuple(
            str(s)
            for s in compound_repair_trajectory_chain),
        delayed_repair_chain=tuple(
            str(s) for s in delayed_repair_chain),
        algebra_signature_v22=str(algebra_signature_v22),
    )


@dataclasses.dataclass
class MergeOperatorV22:
    factor_dim: int = 6

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w74_mlsc_v22_merge_operator",
            "factor_dim": int(self.factor_dim),
        })

    def merge(
            self,
            capsules: Sequence[MergeableLatentCapsuleV22],
            *,
            compound_repair_trajectory_chain: Sequence[
                str] = (),
            delayed_repair_chain: Sequence[str] = (),
            algebra_signature_v22: str = (
                W56_MLSC_V4_ALGEBRA_MERGE),
            **v21_kwargs: Any,
    ) -> MergeableLatentCapsuleV22:
        if not capsules:
            raise ValueError(
                "merge requires at least 1 capsule")
        v21_op = MergeOperatorV21(factor_dim=int(self.factor_dim))
        merged_v21 = v21_op.merge(
            [c.inner_v21 for c in capsules], **v21_kwargs)
        crc: list[str] = []
        for c in capsules:
            for a in c.compound_repair_trajectory_chain:
                if a not in crc:
                    crc.append(a)
        for a in compound_repair_trajectory_chain:
            if a not in crc:
                crc.append(a)
        drc: list[str] = []
        for c in capsules:
            for a in c.delayed_repair_chain:
                if a not in drc:
                    drc.append(a)
        for a in delayed_repair_chain:
            if a not in drc:
                drc.append(a)
        return wrap_v21_as_v22(
            merged_v21,
            compound_repair_trajectory_chain=tuple(crc),
            delayed_repair_chain=tuple(drc),
            algebra_signature_v22=str(algebra_signature_v22),
        )


@dataclasses.dataclass(frozen=True)
class MergeableLatentCapsuleV22Witness:
    schema: str
    capsule_cid: str
    inner_v21_cid: str
    compound_repair_trajectory_chain_depth: int
    delayed_repair_chain_depth: int
    algebra_signature_v22: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "capsule_cid": str(self.capsule_cid),
            "inner_v21_cid": str(self.inner_v21_cid),
            "compound_repair_trajectory_chain_depth": int(
                self.compound_repair_trajectory_chain_depth),
            "delayed_repair_chain_depth": int(
                self.delayed_repair_chain_depth),
            "algebra_signature_v22": str(
                self.algebra_signature_v22),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w74_mlsc_v22_witness",
            "witness": self.to_dict()})


def emit_mlsc_v22_witness(
        capsule: MergeableLatentCapsuleV22,
) -> MergeableLatentCapsuleV22Witness:
    return MergeableLatentCapsuleV22Witness(
        schema=W74_MLSC_V22_SCHEMA_VERSION,
        capsule_cid=str(capsule.cid()),
        inner_v21_cid=str(capsule.inner_v21.cid()),
        compound_repair_trajectory_chain_depth=int(len(
            capsule.compound_repair_trajectory_chain)),
        delayed_repair_chain_depth=int(len(
            capsule.delayed_repair_chain)),
        algebra_signature_v22=str(
            capsule.algebra_signature_v22),
    )


__all__ = [
    "W74_MLSC_V22_SCHEMA_VERSION",
    "W74_MLSC_V22_ALGEBRA_COMPOUND_REPAIR_PROPAGATION",
    "W74_MLSC_V22_ALGEBRA_DELAYED_REPAIR_PROPAGATION",
    "W74_MLSC_V22_KNOWN_ALGEBRA_SIGNATURES",
    "MergeableLatentCapsuleV22",
    "wrap_v21_as_v22",
    "MergeOperatorV22",
    "MergeableLatentCapsuleV22Witness",
    "emit_mlsc_v22_witness",
]
