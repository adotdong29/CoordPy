"""W71 — Mergeable Latent State Capsule V19 (MLSC V19).

Strictly extends W70's ``coordpy.mergeable_latent_capsule_v18``.
V18 added repair-trajectory and budget-primary chains. V19 adds:

* ``delayed_repair_trajectory_chain`` — content-addressed witness
  chain for per-turn delayed-repair-trajectory CIDs.
* ``restart_dominance_chain`` — content-addressed witness chain
  for per-turn restart-dominance signals.
* ``algebra_signature_v19`` — adds two new V19 propagation
  signatures.

V19 merge inherits the unions of both new chains.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from .mergeable_latent_capsule_v4 import W56_MLSC_V4_ALGEBRA_MERGE
from .mergeable_latent_capsule_v18 import (
    MergeableLatentCapsuleV18, MergeOperatorV18,
)
from .tiny_substrate_v3 import _sha256_hex


W71_MLSC_V19_SCHEMA_VERSION: str = (
    "coordpy.mergeable_latent_capsule_v19.v1")
W71_MLSC_V19_ALGEBRA_DELAYED_REPAIR_PROPAGATION: str = (
    "delayed_repair_propagation_v19")
W71_MLSC_V19_ALGEBRA_RESTART_DOMINANCE_PROPAGATION: str = (
    "restart_dominance_propagation_v19")
W71_MLSC_V19_KNOWN_ALGEBRA_SIGNATURES: tuple[str, ...] = (
    W56_MLSC_V4_ALGEBRA_MERGE,
    W71_MLSC_V19_ALGEBRA_DELAYED_REPAIR_PROPAGATION,
    W71_MLSC_V19_ALGEBRA_RESTART_DOMINANCE_PROPAGATION,
)


@dataclasses.dataclass(frozen=True)
class MergeableLatentCapsuleV19:
    schema: str
    inner_v18: MergeableLatentCapsuleV18
    delayed_repair_trajectory_chain: tuple[str, ...]
    restart_dominance_chain: tuple[str, ...]
    algebra_signature_v19: str

    @property
    def payload(self) -> tuple[float, ...]:
        return self.inner_v18.payload

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "inner_v18_cid": str(self.inner_v18.cid()),
            "delayed_repair_trajectory_chain": list(
                self.delayed_repair_trajectory_chain),
            "restart_dominance_chain": list(
                self.restart_dominance_chain),
            "algebra_signature_v19": str(
                self.algebra_signature_v19),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w71_mlsc_v19",
            "capsule": self.to_dict()})


def wrap_v18_as_v19(
        v18_capsule: MergeableLatentCapsuleV18, *,
        delayed_repair_trajectory_chain: Sequence[str] = (),
        restart_dominance_chain: Sequence[str] = (),
        algebra_signature_v19: str = W56_MLSC_V4_ALGEBRA_MERGE,
) -> MergeableLatentCapsuleV19:
    if (algebra_signature_v19 not in
            W71_MLSC_V19_KNOWN_ALGEBRA_SIGNATURES):
        algebra_signature_v19 = W56_MLSC_V4_ALGEBRA_MERGE
    return MergeableLatentCapsuleV19(
        schema=W71_MLSC_V19_SCHEMA_VERSION,
        inner_v18=v18_capsule,
        delayed_repair_trajectory_chain=tuple(
            str(s) for s in delayed_repair_trajectory_chain),
        restart_dominance_chain=tuple(
            str(s) for s in restart_dominance_chain),
        algebra_signature_v19=str(algebra_signature_v19),
    )


@dataclasses.dataclass
class MergeOperatorV19:
    factor_dim: int = 6

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w71_mlsc_v19_merge_operator",
            "factor_dim": int(self.factor_dim),
        })

    def merge(
            self,
            capsules: Sequence[MergeableLatentCapsuleV19],
            *,
            delayed_repair_trajectory_chain: Sequence[str] = (),
            restart_dominance_chain: Sequence[str] = (),
            algebra_signature_v19: str = (
                W56_MLSC_V4_ALGEBRA_MERGE),
            **v18_kwargs: Any,
    ) -> MergeableLatentCapsuleV19:
        if not capsules:
            raise ValueError(
                "merge requires at least 1 capsule")
        v18_op = MergeOperatorV18(factor_dim=int(self.factor_dim))
        merged_v18 = v18_op.merge(
            [c.inner_v18 for c in capsules], **v18_kwargs)
        drt: list[str] = []
        for c in capsules:
            for a in c.delayed_repair_trajectory_chain:
                if a not in drt:
                    drt.append(a)
        for a in delayed_repair_trajectory_chain:
            if a not in drt:
                drt.append(a)
        rd: list[str] = []
        for c in capsules:
            for a in c.restart_dominance_chain:
                if a not in rd:
                    rd.append(a)
        for a in restart_dominance_chain:
            if a not in rd:
                rd.append(a)
        return wrap_v18_as_v19(
            merged_v18,
            delayed_repair_trajectory_chain=tuple(drt),
            restart_dominance_chain=tuple(rd),
            algebra_signature_v19=str(algebra_signature_v19),
        )


@dataclasses.dataclass(frozen=True)
class MergeableLatentCapsuleV19Witness:
    schema: str
    capsule_cid: str
    inner_v18_cid: str
    delayed_repair_trajectory_chain_depth: int
    restart_dominance_chain_depth: int
    algebra_signature_v19: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "capsule_cid": str(self.capsule_cid),
            "inner_v18_cid": str(self.inner_v18_cid),
            "delayed_repair_trajectory_chain_depth": int(
                self.delayed_repair_trajectory_chain_depth),
            "restart_dominance_chain_depth": int(
                self.restart_dominance_chain_depth),
            "algebra_signature_v19": str(
                self.algebra_signature_v19),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w71_mlsc_v19_witness",
            "witness": self.to_dict()})


def emit_mlsc_v19_witness(
        capsule: MergeableLatentCapsuleV19,
) -> MergeableLatentCapsuleV19Witness:
    return MergeableLatentCapsuleV19Witness(
        schema=W71_MLSC_V19_SCHEMA_VERSION,
        capsule_cid=str(capsule.cid()),
        inner_v18_cid=str(capsule.inner_v18.cid()),
        delayed_repair_trajectory_chain_depth=int(len(
            capsule.delayed_repair_trajectory_chain)),
        restart_dominance_chain_depth=int(len(
            capsule.restart_dominance_chain)),
        algebra_signature_v19=str(
            capsule.algebra_signature_v19),
    )


__all__ = [
    "W71_MLSC_V19_SCHEMA_VERSION",
    "W71_MLSC_V19_ALGEBRA_DELAYED_REPAIR_PROPAGATION",
    "W71_MLSC_V19_ALGEBRA_RESTART_DOMINANCE_PROPAGATION",
    "W71_MLSC_V19_KNOWN_ALGEBRA_SIGNATURES",
    "MergeableLatentCapsuleV19",
    "wrap_v18_as_v19",
    "MergeOperatorV19",
    "MergeableLatentCapsuleV19Witness",
    "emit_mlsc_v19_witness",
]
