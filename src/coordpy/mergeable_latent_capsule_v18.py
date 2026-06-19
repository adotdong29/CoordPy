"""W70 — Mergeable Latent State Capsule V18 (MLSC V18).

Strictly extends W69's ``coordpy.mergeable_latent_capsule_v17``.
V17 added multi-branch-rejoin and silent-corruption chains. V18
adds:

* ``repair_trajectory_chain`` — content-addressed witness chain
  for per-turn repair-trajectory CIDs.
* ``budget_primary_chain`` — content-addressed witness chain for
  budget-primary gate decisions per turn.
* ``algebra_signature_v18`` — adds two new V18 propagation
  signatures.

V18 merge inherits the unions of both new chains.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from .mergeable_latent_capsule_v4 import W56_MLSC_V4_ALGEBRA_MERGE
from .mergeable_latent_capsule_v17 import (
    MergeableLatentCapsuleV17, MergeOperatorV17,
)
from .tiny_substrate_v3 import _sha256_hex


W70_MLSC_V18_SCHEMA_VERSION: str = (
    "coordpy.mergeable_latent_capsule_v18.v1")
W70_MLSC_V18_ALGEBRA_REPAIR_TRAJECTORY_PROPAGATION: str = (
    "repair_trajectory_propagation_v18")
W70_MLSC_V18_ALGEBRA_BUDGET_PRIMARY_PROPAGATION: str = (
    "budget_primary_propagation_v18")
W70_MLSC_V18_KNOWN_ALGEBRA_SIGNATURES: tuple[str, ...] = (
    W56_MLSC_V4_ALGEBRA_MERGE,
    W70_MLSC_V18_ALGEBRA_REPAIR_TRAJECTORY_PROPAGATION,
    W70_MLSC_V18_ALGEBRA_BUDGET_PRIMARY_PROPAGATION,
)


@dataclasses.dataclass(frozen=True)
class MergeableLatentCapsuleV18:
    schema: str
    inner_v17: MergeableLatentCapsuleV17
    repair_trajectory_chain: tuple[str, ...]
    budget_primary_chain: tuple[str, ...]
    algebra_signature_v18: str

    @property
    def payload(self) -> tuple[float, ...]:
        return self.inner_v17.payload

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "inner_v17_cid": str(self.inner_v17.cid()),
            "repair_trajectory_chain": list(
                self.repair_trajectory_chain),
            "budget_primary_chain": list(
                self.budget_primary_chain),
            "algebra_signature_v18": str(
                self.algebra_signature_v18),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w70_mlsc_v18",
            "capsule": self.to_dict()})


def wrap_v17_as_v18(
        v17_capsule: MergeableLatentCapsuleV17, *,
        repair_trajectory_chain: Sequence[str] = (),
        budget_primary_chain: Sequence[str] = (),
        algebra_signature_v18: str = W56_MLSC_V4_ALGEBRA_MERGE,
) -> MergeableLatentCapsuleV18:
    if (algebra_signature_v18 not in
            W70_MLSC_V18_KNOWN_ALGEBRA_SIGNATURES):
        algebra_signature_v18 = W56_MLSC_V4_ALGEBRA_MERGE
    return MergeableLatentCapsuleV18(
        schema=W70_MLSC_V18_SCHEMA_VERSION,
        inner_v17=v17_capsule,
        repair_trajectory_chain=tuple(
            str(s) for s in repair_trajectory_chain),
        budget_primary_chain=tuple(
            str(s) for s in budget_primary_chain),
        algebra_signature_v18=str(algebra_signature_v18),
    )


@dataclasses.dataclass
class MergeOperatorV18:
    factor_dim: int = 6

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w70_mlsc_v18_merge_operator",
            "factor_dim": int(self.factor_dim),
        })

    def merge(
            self,
            capsules: Sequence[MergeableLatentCapsuleV18],
            *,
            repair_trajectory_chain: Sequence[str] = (),
            budget_primary_chain: Sequence[str] = (),
            algebra_signature_v18: str = (
                W56_MLSC_V4_ALGEBRA_MERGE),
            **v17_kwargs: Any,
    ) -> MergeableLatentCapsuleV18:
        if not capsules:
            raise ValueError(
                "merge requires at least 1 capsule")
        v17_op = MergeOperatorV17(factor_dim=int(self.factor_dim))
        merged_v17 = v17_op.merge(
            [c.inner_v17 for c in capsules], **v17_kwargs)
        rt: list[str] = []
        for c in capsules:
            for a in c.repair_trajectory_chain:
                if a not in rt:
                    rt.append(a)
        for a in repair_trajectory_chain:
            if a not in rt:
                rt.append(a)
        bp: list[str] = []
        for c in capsules:
            for a in c.budget_primary_chain:
                if a not in bp:
                    bp.append(a)
        for a in budget_primary_chain:
            if a not in bp:
                bp.append(a)
        return wrap_v17_as_v18(
            merged_v17,
            repair_trajectory_chain=tuple(rt),
            budget_primary_chain=tuple(bp),
            algebra_signature_v18=str(algebra_signature_v18),
        )


@dataclasses.dataclass(frozen=True)
class MergeableLatentCapsuleV18Witness:
    schema: str
    capsule_cid: str
    inner_v17_cid: str
    repair_trajectory_chain_depth: int
    budget_primary_chain_depth: int
    algebra_signature_v18: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "capsule_cid": str(self.capsule_cid),
            "inner_v17_cid": str(self.inner_v17_cid),
            "repair_trajectory_chain_depth": int(
                self.repair_trajectory_chain_depth),
            "budget_primary_chain_depth": int(
                self.budget_primary_chain_depth),
            "algebra_signature_v18": str(
                self.algebra_signature_v18),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w70_mlsc_v18_witness",
            "witness": self.to_dict()})


def emit_mlsc_v18_witness(
        capsule: MergeableLatentCapsuleV18,
) -> MergeableLatentCapsuleV18Witness:
    return MergeableLatentCapsuleV18Witness(
        schema=W70_MLSC_V18_SCHEMA_VERSION,
        capsule_cid=str(capsule.cid()),
        inner_v17_cid=str(capsule.inner_v17.cid()),
        repair_trajectory_chain_depth=int(len(
            capsule.repair_trajectory_chain)),
        budget_primary_chain_depth=int(len(
            capsule.budget_primary_chain)),
        algebra_signature_v18=str(capsule.algebra_signature_v18),
    )


__all__ = [
    "W70_MLSC_V18_SCHEMA_VERSION",
    "W70_MLSC_V18_ALGEBRA_REPAIR_TRAJECTORY_PROPAGATION",
    "W70_MLSC_V18_ALGEBRA_BUDGET_PRIMARY_PROPAGATION",
    "W70_MLSC_V18_KNOWN_ALGEBRA_SIGNATURES",
    "MergeableLatentCapsuleV18",
    "wrap_v17_as_v18",
    "MergeOperatorV18",
    "MergeableLatentCapsuleV18Witness",
    "emit_mlsc_v18_witness",
]
