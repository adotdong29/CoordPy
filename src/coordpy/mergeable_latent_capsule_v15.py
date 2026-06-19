"""W67 M12 — Mergeable Latent State Capsule V15 (MLSC V15).

Strictly extends W66's ``coordpy.mergeable_latent_capsule_v14``.
V14 added team-failure-recovery and team-consensus-under-budget
chains. V15 adds:

* ``role_dropout_recovery_witness_chain`` — content-addressed
  witness chain for role-dropout-recovery events.
* ``branch_merge_reconciliation_witness_chain`` — content-
  addressed witness chain for branch-merge reconciliation events.
* ``algebra_signature_v15`` — adds two new V15 propagation
  signatures.

V15 merge inherits the unions of both new chains.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from .mergeable_latent_capsule_v4 import W56_MLSC_V4_ALGEBRA_MERGE
from .mergeable_latent_capsule_v14 import (
    MergeableLatentCapsuleV14, MergeOperatorV14,
)
from .tiny_substrate_v3 import _sha256_hex


W67_MLSC_V15_SCHEMA_VERSION: str = (
    "coordpy.mergeable_latent_capsule_v15.v1")
W67_MLSC_V15_ALGEBRA_ROLE_DROPOUT_RECOVERY_PROPAGATION: str = (
    "role_dropout_recovery_propagation_v15")
W67_MLSC_V15_ALGEBRA_BRANCH_MERGE_RECONCILIATION_PROPAGATION: str = (
    "branch_merge_reconciliation_propagation_v15")
W67_MLSC_V15_KNOWN_ALGEBRA_SIGNATURES: tuple[str, ...] = (
    W56_MLSC_V4_ALGEBRA_MERGE,
    W67_MLSC_V15_ALGEBRA_ROLE_DROPOUT_RECOVERY_PROPAGATION,
    W67_MLSC_V15_ALGEBRA_BRANCH_MERGE_RECONCILIATION_PROPAGATION,
)


@dataclasses.dataclass(frozen=True)
class MergeableLatentCapsuleV15:
    schema: str
    inner_v14: MergeableLatentCapsuleV14
    role_dropout_recovery_witness_chain: tuple[str, ...]
    branch_merge_reconciliation_witness_chain: tuple[str, ...]
    algebra_signature_v15: str

    @property
    def payload(self) -> tuple[float, ...]:
        return self.inner_v14.payload

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "inner_v14_cid": str(self.inner_v14.cid()),
            "role_dropout_recovery_witness_chain": list(
                self.role_dropout_recovery_witness_chain),
            "branch_merge_reconciliation_witness_chain": list(
                self.branch_merge_reconciliation_witness_chain),
            "algebra_signature_v15": str(
                self.algebra_signature_v15),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w67_mlsc_v15",
            "capsule": self.to_dict()})


def wrap_v14_as_v15(
        v14_capsule: MergeableLatentCapsuleV14, *,
        role_dropout_recovery_witness_chain: Sequence[str] = (),
        branch_merge_reconciliation_witness_chain: Sequence[str] = (),
        algebra_signature_v15: str = W56_MLSC_V4_ALGEBRA_MERGE,
) -> MergeableLatentCapsuleV15:
    if (algebra_signature_v15 not in
            W67_MLSC_V15_KNOWN_ALGEBRA_SIGNATURES):
        algebra_signature_v15 = W56_MLSC_V4_ALGEBRA_MERGE
    return MergeableLatentCapsuleV15(
        schema=W67_MLSC_V15_SCHEMA_VERSION,
        inner_v14=v14_capsule,
        role_dropout_recovery_witness_chain=tuple(
            str(s) for s in role_dropout_recovery_witness_chain),
        branch_merge_reconciliation_witness_chain=tuple(
            str(s) for s
            in branch_merge_reconciliation_witness_chain),
        algebra_signature_v15=str(algebra_signature_v15),
    )


@dataclasses.dataclass
class MergeOperatorV15:
    factor_dim: int = 6

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w67_mlsc_v15_merge_operator",
            "factor_dim": int(self.factor_dim),
        })

    def merge(
            self,
            capsules: Sequence[MergeableLatentCapsuleV15],
            *,
            role_dropout_recovery_witness_chain: Sequence[str] = (),
            branch_merge_reconciliation_witness_chain: (
                Sequence[str]) = (),
            algebra_signature_v15: str = (
                W56_MLSC_V4_ALGEBRA_MERGE),
            **v14_kwargs: Any,
    ) -> MergeableLatentCapsuleV15:
        if not capsules:
            raise ValueError(
                "merge requires at least 1 capsule")
        v14_op = MergeOperatorV14(factor_dim=int(self.factor_dim))
        merged_v14 = v14_op.merge(
            [c.inner_v14 for c in capsules], **v14_kwargs)
        rd: list[str] = []
        for c in capsules:
            for a in c.role_dropout_recovery_witness_chain:
                if a not in rd:
                    rd.append(a)
        for a in role_dropout_recovery_witness_chain:
            if a not in rd:
                rd.append(a)
        bm: list[str] = []
        for c in capsules:
            for a in c.branch_merge_reconciliation_witness_chain:
                if a not in bm:
                    bm.append(a)
        for a in branch_merge_reconciliation_witness_chain:
            if a not in bm:
                bm.append(a)
        return wrap_v14_as_v15(
            merged_v14,
            role_dropout_recovery_witness_chain=tuple(rd),
            branch_merge_reconciliation_witness_chain=tuple(bm),
            algebra_signature_v15=str(algebra_signature_v15),
        )


@dataclasses.dataclass(frozen=True)
class MergeableLatentCapsuleV15Witness:
    schema: str
    capsule_cid: str
    inner_v14_cid: str
    role_dropout_recovery_chain_depth: int
    branch_merge_reconciliation_chain_depth: int
    algebra_signature_v15: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "capsule_cid": str(self.capsule_cid),
            "inner_v14_cid": str(self.inner_v14_cid),
            "role_dropout_recovery_chain_depth": int(
                self.role_dropout_recovery_chain_depth),
            "branch_merge_reconciliation_chain_depth": int(
                self.branch_merge_reconciliation_chain_depth),
            "algebra_signature_v15": str(
                self.algebra_signature_v15),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w67_mlsc_v15_witness",
            "witness": self.to_dict()})


def emit_mlsc_v15_witness(
        capsule: MergeableLatentCapsuleV15,
) -> MergeableLatentCapsuleV15Witness:
    return MergeableLatentCapsuleV15Witness(
        schema=W67_MLSC_V15_SCHEMA_VERSION,
        capsule_cid=str(capsule.cid()),
        inner_v14_cid=str(capsule.inner_v14.cid()),
        role_dropout_recovery_chain_depth=int(len(
            capsule.role_dropout_recovery_witness_chain)),
        branch_merge_reconciliation_chain_depth=int(len(
            capsule.branch_merge_reconciliation_witness_chain)),
        algebra_signature_v15=str(capsule.algebra_signature_v15),
    )


__all__ = [
    "W67_MLSC_V15_SCHEMA_VERSION",
    "W67_MLSC_V15_ALGEBRA_ROLE_DROPOUT_RECOVERY_PROPAGATION",
    "W67_MLSC_V15_ALGEBRA_BRANCH_MERGE_RECONCILIATION_PROPAGATION",
    "W67_MLSC_V15_KNOWN_ALGEBRA_SIGNATURES",
    "MergeableLatentCapsuleV15",
    "wrap_v14_as_v15",
    "MergeOperatorV15",
    "MergeableLatentCapsuleV15Witness",
    "emit_mlsc_v15_witness",
]
