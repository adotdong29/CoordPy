"""W69 M10 — Mergeable Latent State Capsule V17 (MLSC V17).

Strictly extends W68's ``coordpy.mergeable_latent_capsule_v16``.
V16 added partial-contradiction and agent-replacement chains.
V17 adds:

* ``multi_branch_rejoin_witness_chain`` — content-addressed witness
  chain for multi-branch-rejoin-after-divergent-work events.
* ``silent_corruption_witness_chain`` — content-addressed witness
  chain for silent-corruption-plus-member-replacement events.
* ``algebra_signature_v17`` — adds two new V17 propagation
  signatures.

V17 merge inherits the unions of both new chains.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from .mergeable_latent_capsule_v4 import W56_MLSC_V4_ALGEBRA_MERGE
from .mergeable_latent_capsule_v16 import (
    MergeableLatentCapsuleV16, MergeOperatorV16,
)
from .tiny_substrate_v3 import _sha256_hex


W69_MLSC_V17_SCHEMA_VERSION: str = (
    "coordpy.mergeable_latent_capsule_v17.v1")
W69_MLSC_V17_ALGEBRA_MULTI_BRANCH_REJOIN_PROPAGATION: str = (
    "multi_branch_rejoin_propagation_v17")
W69_MLSC_V17_ALGEBRA_SILENT_CORRUPTION_PROPAGATION: str = (
    "silent_corruption_propagation_v17")
W69_MLSC_V17_KNOWN_ALGEBRA_SIGNATURES: tuple[str, ...] = (
    W56_MLSC_V4_ALGEBRA_MERGE,
    W69_MLSC_V17_ALGEBRA_MULTI_BRANCH_REJOIN_PROPAGATION,
    W69_MLSC_V17_ALGEBRA_SILENT_CORRUPTION_PROPAGATION,
)


@dataclasses.dataclass(frozen=True)
class MergeableLatentCapsuleV17:
    schema: str
    inner_v16: MergeableLatentCapsuleV16
    multi_branch_rejoin_witness_chain: tuple[str, ...]
    silent_corruption_witness_chain: tuple[str, ...]
    algebra_signature_v17: str

    @property
    def payload(self) -> tuple[float, ...]:
        return self.inner_v16.payload

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "inner_v16_cid": str(self.inner_v16.cid()),
            "multi_branch_rejoin_witness_chain": list(
                self.multi_branch_rejoin_witness_chain),
            "silent_corruption_witness_chain": list(
                self.silent_corruption_witness_chain),
            "algebra_signature_v17": str(
                self.algebra_signature_v17),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w69_mlsc_v17",
            "capsule": self.to_dict()})


def wrap_v16_as_v17(
        v16_capsule: MergeableLatentCapsuleV16, *,
        multi_branch_rejoin_witness_chain: Sequence[str] = (),
        silent_corruption_witness_chain: Sequence[str] = (),
        algebra_signature_v17: str = W56_MLSC_V4_ALGEBRA_MERGE,
) -> MergeableLatentCapsuleV17:
    if (algebra_signature_v17 not in
            W69_MLSC_V17_KNOWN_ALGEBRA_SIGNATURES):
        algebra_signature_v17 = W56_MLSC_V4_ALGEBRA_MERGE
    return MergeableLatentCapsuleV17(
        schema=W69_MLSC_V17_SCHEMA_VERSION,
        inner_v16=v16_capsule,
        multi_branch_rejoin_witness_chain=tuple(
            str(s) for s in multi_branch_rejoin_witness_chain),
        silent_corruption_witness_chain=tuple(
            str(s) for s in silent_corruption_witness_chain),
        algebra_signature_v17=str(algebra_signature_v17),
    )


@dataclasses.dataclass
class MergeOperatorV17:
    factor_dim: int = 6

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w69_mlsc_v17_merge_operator",
            "factor_dim": int(self.factor_dim),
        })

    def merge(
            self,
            capsules: Sequence[MergeableLatentCapsuleV17],
            *,
            multi_branch_rejoin_witness_chain: Sequence[str] = (),
            silent_corruption_witness_chain: Sequence[str] = (),
            algebra_signature_v17: str = (
                W56_MLSC_V4_ALGEBRA_MERGE),
            **v16_kwargs: Any,
    ) -> MergeableLatentCapsuleV17:
        if not capsules:
            raise ValueError(
                "merge requires at least 1 capsule")
        v16_op = MergeOperatorV16(factor_dim=int(self.factor_dim))
        merged_v16 = v16_op.merge(
            [c.inner_v16 for c in capsules], **v16_kwargs)
        mbr: list[str] = []
        for c in capsules:
            for a in c.multi_branch_rejoin_witness_chain:
                if a not in mbr:
                    mbr.append(a)
        for a in multi_branch_rejoin_witness_chain:
            if a not in mbr:
                mbr.append(a)
        sc: list[str] = []
        for c in capsules:
            for a in c.silent_corruption_witness_chain:
                if a not in sc:
                    sc.append(a)
        for a in silent_corruption_witness_chain:
            if a not in sc:
                sc.append(a)
        return wrap_v16_as_v17(
            merged_v16,
            multi_branch_rejoin_witness_chain=tuple(mbr),
            silent_corruption_witness_chain=tuple(sc),
            algebra_signature_v17=str(algebra_signature_v17),
        )


@dataclasses.dataclass(frozen=True)
class MergeableLatentCapsuleV17Witness:
    schema: str
    capsule_cid: str
    inner_v16_cid: str
    multi_branch_rejoin_chain_depth: int
    silent_corruption_chain_depth: int
    algebra_signature_v17: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "capsule_cid": str(self.capsule_cid),
            "inner_v16_cid": str(self.inner_v16_cid),
            "multi_branch_rejoin_chain_depth": int(
                self.multi_branch_rejoin_chain_depth),
            "silent_corruption_chain_depth": int(
                self.silent_corruption_chain_depth),
            "algebra_signature_v17": str(
                self.algebra_signature_v17),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w69_mlsc_v17_witness",
            "witness": self.to_dict()})


def emit_mlsc_v17_witness(
        capsule: MergeableLatentCapsuleV17,
) -> MergeableLatentCapsuleV17Witness:
    return MergeableLatentCapsuleV17Witness(
        schema=W69_MLSC_V17_SCHEMA_VERSION,
        capsule_cid=str(capsule.cid()),
        inner_v16_cid=str(capsule.inner_v16.cid()),
        multi_branch_rejoin_chain_depth=int(len(
            capsule.multi_branch_rejoin_witness_chain)),
        silent_corruption_chain_depth=int(len(
            capsule.silent_corruption_witness_chain)),
        algebra_signature_v17=str(capsule.algebra_signature_v17),
    )


__all__ = [
    "W69_MLSC_V17_SCHEMA_VERSION",
    "W69_MLSC_V17_ALGEBRA_MULTI_BRANCH_REJOIN_PROPAGATION",
    "W69_MLSC_V17_ALGEBRA_SILENT_CORRUPTION_PROPAGATION",
    "W69_MLSC_V17_KNOWN_ALGEBRA_SIGNATURES",
    "MergeableLatentCapsuleV17",
    "wrap_v16_as_v17",
    "MergeOperatorV17",
    "MergeableLatentCapsuleV17Witness",
    "emit_mlsc_v17_witness",
]
