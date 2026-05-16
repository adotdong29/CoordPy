"""W68 M10 — Mergeable Latent State Capsule V16 (MLSC V16).

Strictly extends W67's ``coordpy.mergeable_latent_capsule_v15``.
V15 added role-dropout-recovery and branch-merge-reconciliation
chains. V16 adds:

* ``partial_contradiction_witness_chain`` — content-addressed
  witness chain for partial-contradiction-under-delayed-
  reconciliation events.
* ``agent_replacement_witness_chain`` — content-addressed witness
  chain for agent-replacement-warm-restart events.
* ``algebra_signature_v16`` — adds two new V16 propagation
  signatures.

V16 merge inherits the unions of both new chains.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from .mergeable_latent_capsule_v4 import W56_MLSC_V4_ALGEBRA_MERGE
from .mergeable_latent_capsule_v15 import (
    MergeableLatentCapsuleV15, MergeOperatorV15,
)
from .tiny_substrate_v3 import _sha256_hex


W68_MLSC_V16_SCHEMA_VERSION: str = (
    "coordpy.mergeable_latent_capsule_v16.v1")
W68_MLSC_V16_ALGEBRA_PARTIAL_CONTRADICTION_PROPAGATION: str = (
    "partial_contradiction_propagation_v16")
W68_MLSC_V16_ALGEBRA_AGENT_REPLACEMENT_PROPAGATION: str = (
    "agent_replacement_propagation_v16")
W68_MLSC_V16_KNOWN_ALGEBRA_SIGNATURES: tuple[str, ...] = (
    W56_MLSC_V4_ALGEBRA_MERGE,
    W68_MLSC_V16_ALGEBRA_PARTIAL_CONTRADICTION_PROPAGATION,
    W68_MLSC_V16_ALGEBRA_AGENT_REPLACEMENT_PROPAGATION,
)


@dataclasses.dataclass(frozen=True)
class MergeableLatentCapsuleV16:
    schema: str
    inner_v15: MergeableLatentCapsuleV15
    partial_contradiction_witness_chain: tuple[str, ...]
    agent_replacement_witness_chain: tuple[str, ...]
    algebra_signature_v16: str

    @property
    def payload(self) -> tuple[float, ...]:
        return self.inner_v15.payload

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "inner_v15_cid": str(self.inner_v15.cid()),
            "partial_contradiction_witness_chain": list(
                self.partial_contradiction_witness_chain),
            "agent_replacement_witness_chain": list(
                self.agent_replacement_witness_chain),
            "algebra_signature_v16": str(
                self.algebra_signature_v16),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w68_mlsc_v16",
            "capsule": self.to_dict()})


def wrap_v15_as_v16(
        v15_capsule: MergeableLatentCapsuleV15, *,
        partial_contradiction_witness_chain: Sequence[str] = (),
        agent_replacement_witness_chain: Sequence[str] = (),
        algebra_signature_v16: str = W56_MLSC_V4_ALGEBRA_MERGE,
) -> MergeableLatentCapsuleV16:
    if (algebra_signature_v16 not in
            W68_MLSC_V16_KNOWN_ALGEBRA_SIGNATURES):
        algebra_signature_v16 = W56_MLSC_V4_ALGEBRA_MERGE
    return MergeableLatentCapsuleV16(
        schema=W68_MLSC_V16_SCHEMA_VERSION,
        inner_v15=v15_capsule,
        partial_contradiction_witness_chain=tuple(
            str(s) for s in partial_contradiction_witness_chain),
        agent_replacement_witness_chain=tuple(
            str(s) for s in agent_replacement_witness_chain),
        algebra_signature_v16=str(algebra_signature_v16),
    )


@dataclasses.dataclass
class MergeOperatorV16:
    factor_dim: int = 6

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w68_mlsc_v16_merge_operator",
            "factor_dim": int(self.factor_dim),
        })

    def merge(
            self,
            capsules: Sequence[MergeableLatentCapsuleV16],
            *,
            partial_contradiction_witness_chain: Sequence[str] = (),
            agent_replacement_witness_chain: Sequence[str] = (),
            algebra_signature_v16: str = (
                W56_MLSC_V4_ALGEBRA_MERGE),
            **v15_kwargs: Any,
    ) -> MergeableLatentCapsuleV16:
        if not capsules:
            raise ValueError(
                "merge requires at least 1 capsule")
        v15_op = MergeOperatorV15(factor_dim=int(self.factor_dim))
        merged_v15 = v15_op.merge(
            [c.inner_v15 for c in capsules], **v15_kwargs)
        pc: list[str] = []
        for c in capsules:
            for a in c.partial_contradiction_witness_chain:
                if a not in pc:
                    pc.append(a)
        for a in partial_contradiction_witness_chain:
            if a not in pc:
                pc.append(a)
        ar: list[str] = []
        for c in capsules:
            for a in c.agent_replacement_witness_chain:
                if a not in ar:
                    ar.append(a)
        for a in agent_replacement_witness_chain:
            if a not in ar:
                ar.append(a)
        return wrap_v15_as_v16(
            merged_v15,
            partial_contradiction_witness_chain=tuple(pc),
            agent_replacement_witness_chain=tuple(ar),
            algebra_signature_v16=str(algebra_signature_v16),
        )


@dataclasses.dataclass(frozen=True)
class MergeableLatentCapsuleV16Witness:
    schema: str
    capsule_cid: str
    inner_v15_cid: str
    partial_contradiction_chain_depth: int
    agent_replacement_chain_depth: int
    algebra_signature_v16: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "capsule_cid": str(self.capsule_cid),
            "inner_v15_cid": str(self.inner_v15_cid),
            "partial_contradiction_chain_depth": int(
                self.partial_contradiction_chain_depth),
            "agent_replacement_chain_depth": int(
                self.agent_replacement_chain_depth),
            "algebra_signature_v16": str(
                self.algebra_signature_v16),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w68_mlsc_v16_witness",
            "witness": self.to_dict()})


def emit_mlsc_v16_witness(
        capsule: MergeableLatentCapsuleV16,
) -> MergeableLatentCapsuleV16Witness:
    return MergeableLatentCapsuleV16Witness(
        schema=W68_MLSC_V16_SCHEMA_VERSION,
        capsule_cid=str(capsule.cid()),
        inner_v15_cid=str(capsule.inner_v15.cid()),
        partial_contradiction_chain_depth=int(len(
            capsule.partial_contradiction_witness_chain)),
        agent_replacement_chain_depth=int(len(
            capsule.agent_replacement_witness_chain)),
        algebra_signature_v16=str(capsule.algebra_signature_v16),
    )


__all__ = [
    "W68_MLSC_V16_SCHEMA_VERSION",
    "W68_MLSC_V16_ALGEBRA_PARTIAL_CONTRADICTION_PROPAGATION",
    "W68_MLSC_V16_ALGEBRA_AGENT_REPLACEMENT_PROPAGATION",
    "W68_MLSC_V16_KNOWN_ALGEBRA_SIGNATURES",
    "MergeableLatentCapsuleV16",
    "wrap_v15_as_v16",
    "MergeOperatorV16",
    "MergeableLatentCapsuleV16Witness",
    "emit_mlsc_v16_witness",
]
