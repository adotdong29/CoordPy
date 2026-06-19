"""W75 — Mergeable Latent State Capsule V23 (MLSC V23).

Strictly extends W74's ``coordpy.mergeable_latent_capsule_v22``.
V22 added compound-repair-trajectory and delayed-repair chains.
V23 adds:

* ``compound_chain_repair_trajectory_chain`` — content-addressed
  witness chain for per-turn compound-chain-repair-trajectory CIDs
  (V20 axis).
* ``replacement_then_rejoin_chain`` — content-addressed witness
  chain for per-turn replacement-then-rejoin compound chains.
* ``algebra_signature_v23`` — adds two new V23 propagation
  signatures.

V23 merge inherits the unions of both new chains.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from .mergeable_latent_capsule_v4 import W56_MLSC_V4_ALGEBRA_MERGE
from .mergeable_latent_capsule_v22 import (
    MergeableLatentCapsuleV22, MergeOperatorV22,
)
from .tiny_substrate_v3 import _sha256_hex


W75_MLSC_V23_SCHEMA_VERSION: str = (
    "coordpy.mergeable_latent_capsule_v23.v1")
W75_MLSC_V23_ALGEBRA_COMPOUND_CHAIN_REPAIR_PROPAGATION: str = (
    "compound_chain_repair_propagation_v23")
W75_MLSC_V23_ALGEBRA_REPLACEMENT_THEN_REJOIN_PROPAGATION: str = (
    "replacement_then_rejoin_propagation_v23")
W75_MLSC_V23_KNOWN_ALGEBRA_SIGNATURES: tuple[str, ...] = (
    W56_MLSC_V4_ALGEBRA_MERGE,
    W75_MLSC_V23_ALGEBRA_COMPOUND_CHAIN_REPAIR_PROPAGATION,
    W75_MLSC_V23_ALGEBRA_REPLACEMENT_THEN_REJOIN_PROPAGATION,
)


@dataclasses.dataclass(frozen=True)
class MergeableLatentCapsuleV23:
    schema: str
    inner_v22: MergeableLatentCapsuleV22
    compound_chain_repair_trajectory_chain: tuple[str, ...]
    replacement_then_rejoin_chain: tuple[str, ...]
    algebra_signature_v23: str

    @property
    def payload(self) -> tuple[float, ...]:
        return self.inner_v22.payload

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "inner_v22_cid": str(self.inner_v22.cid()),
            "compound_chain_repair_trajectory_chain": list(
                self.compound_chain_repair_trajectory_chain),
            "replacement_then_rejoin_chain": list(
                self.replacement_then_rejoin_chain),
            "algebra_signature_v23": str(
                self.algebra_signature_v23),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w75_mlsc_v23",
            "capsule": self.to_dict()})


def wrap_v22_as_v23(
        v22_capsule: MergeableLatentCapsuleV22, *,
        compound_chain_repair_trajectory_chain: Sequence[str] = (),
        replacement_then_rejoin_chain: Sequence[str] = (),
        algebra_signature_v23: str = W56_MLSC_V4_ALGEBRA_MERGE,
) -> MergeableLatentCapsuleV23:
    if (algebra_signature_v23 not in
            W75_MLSC_V23_KNOWN_ALGEBRA_SIGNATURES):
        algebra_signature_v23 = W56_MLSC_V4_ALGEBRA_MERGE
    return MergeableLatentCapsuleV23(
        schema=W75_MLSC_V23_SCHEMA_VERSION,
        inner_v22=v22_capsule,
        compound_chain_repair_trajectory_chain=tuple(
            str(s)
            for s in compound_chain_repair_trajectory_chain),
        replacement_then_rejoin_chain=tuple(
            str(s) for s in replacement_then_rejoin_chain),
        algebra_signature_v23=str(algebra_signature_v23),
    )


@dataclasses.dataclass
class MergeOperatorV23:
    factor_dim: int = 6

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w75_mlsc_v23_merge_operator",
            "factor_dim": int(self.factor_dim),
        })

    def merge(
            self,
            capsules: Sequence[MergeableLatentCapsuleV23],
            *,
            compound_chain_repair_trajectory_chain: Sequence[
                str] = (),
            replacement_then_rejoin_chain: Sequence[str] = (),
            algebra_signature_v23: str = (
                W56_MLSC_V4_ALGEBRA_MERGE),
            **v22_kwargs: Any,
    ) -> MergeableLatentCapsuleV23:
        if not capsules:
            raise ValueError(
                "merge requires at least 1 capsule")
        v22_op = MergeOperatorV22(factor_dim=int(self.factor_dim))
        merged_v22 = v22_op.merge(
            [c.inner_v22 for c in capsules], **v22_kwargs)
        ccr: list[str] = []
        for c in capsules:
            for a in c.compound_chain_repair_trajectory_chain:
                if a not in ccr:
                    ccr.append(a)
        for a in compound_chain_repair_trajectory_chain:
            if a not in ccr:
                ccr.append(a)
        rtr: list[str] = []
        for c in capsules:
            for a in c.replacement_then_rejoin_chain:
                if a not in rtr:
                    rtr.append(a)
        for a in replacement_then_rejoin_chain:
            if a not in rtr:
                rtr.append(a)
        return wrap_v22_as_v23(
            merged_v22,
            compound_chain_repair_trajectory_chain=tuple(ccr),
            replacement_then_rejoin_chain=tuple(rtr),
            algebra_signature_v23=str(algebra_signature_v23),
        )


@dataclasses.dataclass(frozen=True)
class MergeableLatentCapsuleV23Witness:
    schema: str
    capsule_cid: str
    inner_v22_cid: str
    compound_chain_repair_trajectory_chain_depth: int
    replacement_then_rejoin_chain_depth: int
    algebra_signature_v23: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "capsule_cid": str(self.capsule_cid),
            "inner_v22_cid": str(self.inner_v22_cid),
            "compound_chain_repair_trajectory_chain_depth":
                int(
                    self
                    .compound_chain_repair_trajectory_chain_depth
                ),
            "replacement_then_rejoin_chain_depth": int(
                self.replacement_then_rejoin_chain_depth),
            "algebra_signature_v23": str(
                self.algebra_signature_v23),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w75_mlsc_v23_witness",
            "witness": self.to_dict()})


def emit_mlsc_v23_witness(
        capsule: MergeableLatentCapsuleV23,
) -> MergeableLatentCapsuleV23Witness:
    return MergeableLatentCapsuleV23Witness(
        schema=W75_MLSC_V23_SCHEMA_VERSION,
        capsule_cid=str(capsule.cid()),
        inner_v22_cid=str(capsule.inner_v22.cid()),
        compound_chain_repair_trajectory_chain_depth=int(len(
            capsule.compound_chain_repair_trajectory_chain)),
        replacement_then_rejoin_chain_depth=int(len(
            capsule.replacement_then_rejoin_chain)),
        algebra_signature_v23=str(
            capsule.algebra_signature_v23),
    )


__all__ = [
    "W75_MLSC_V23_SCHEMA_VERSION",
    "W75_MLSC_V23_ALGEBRA_COMPOUND_CHAIN_REPAIR_PROPAGATION",
    "W75_MLSC_V23_ALGEBRA_REPLACEMENT_THEN_REJOIN_PROPAGATION",
    "W75_MLSC_V23_KNOWN_ALGEBRA_SIGNATURES",
    "MergeableLatentCapsuleV23",
    "wrap_v22_as_v23",
    "MergeOperatorV23",
    "MergeableLatentCapsuleV23Witness",
    "emit_mlsc_v23_witness",
]
