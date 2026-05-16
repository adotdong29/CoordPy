"""W65 M12 — Mergeable Latent State Capsule V13 (MLSC V13).

Strictly extends W64's ``coordpy.mergeable_latent_capsule_v12``.
V12 added replay-dominance-primary chain + hidden-state-trust
chain + total-variation distance. V13 adds:

* ``team_substrate_witness_chain`` — content-addressed witness
  chain for team-substrate-coordination events.
* ``role_conditioned_witness_chain`` — content-addressed witness
  chain for per-role coordination decisions.
* ``algebra_signature_v13`` — adds the new V13 propagation
  signature.

V13 merge inherits the unions of both new chains.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from .mergeable_latent_capsule_v4 import W56_MLSC_V4_ALGEBRA_MERGE
from .mergeable_latent_capsule_v12 import (
    MergeableLatentCapsuleV12, MergeOperatorV12,
)
from .tiny_substrate_v3 import _sha256_hex


W65_MLSC_V13_SCHEMA_VERSION: str = (
    "coordpy.mergeable_latent_capsule_v13.v1")
W65_MLSC_V13_ALGEBRA_TEAM_SUBSTRATE_PROPAGATION: str = (
    "team_substrate_propagation_v13")
W65_MLSC_V13_ALGEBRA_ROLE_CONDITIONED_PROPAGATION: str = (
    "role_conditioned_propagation_v13")
W65_MLSC_V13_KNOWN_ALGEBRA_SIGNATURES: tuple[str, ...] = (
    W56_MLSC_V4_ALGEBRA_MERGE,
    W65_MLSC_V13_ALGEBRA_TEAM_SUBSTRATE_PROPAGATION,
    W65_MLSC_V13_ALGEBRA_ROLE_CONDITIONED_PROPAGATION,
)


@dataclasses.dataclass(frozen=True)
class MergeableLatentCapsuleV13:
    schema: str
    inner_v12: MergeableLatentCapsuleV12
    team_substrate_witness_chain: tuple[str, ...]
    role_conditioned_witness_chain: tuple[str, ...]
    algebra_signature_v13: str

    @property
    def payload(self) -> tuple[float, ...]:
        return self.inner_v12.payload

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "inner_v12_cid": str(self.inner_v12.cid()),
            "team_substrate_witness_chain": list(
                self.team_substrate_witness_chain),
            "role_conditioned_witness_chain": list(
                self.role_conditioned_witness_chain),
            "algebra_signature_v13": str(
                self.algebra_signature_v13),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w65_mlsc_v13",
            "capsule": self.to_dict()})


def wrap_v12_as_v13(
        v12_capsule: MergeableLatentCapsuleV12, *,
        team_substrate_witness_chain: Sequence[str] = (),
        role_conditioned_witness_chain: Sequence[str] = (),
        algebra_signature_v13: str = W56_MLSC_V4_ALGEBRA_MERGE,
) -> MergeableLatentCapsuleV13:
    if (algebra_signature_v13 not in
            W65_MLSC_V13_KNOWN_ALGEBRA_SIGNATURES):
        algebra_signature_v13 = W56_MLSC_V4_ALGEBRA_MERGE
    return MergeableLatentCapsuleV13(
        schema=W65_MLSC_V13_SCHEMA_VERSION,
        inner_v12=v12_capsule,
        team_substrate_witness_chain=tuple(
            str(s) for s in team_substrate_witness_chain),
        role_conditioned_witness_chain=tuple(
            str(s) for s in role_conditioned_witness_chain),
        algebra_signature_v13=str(algebra_signature_v13),
    )


@dataclasses.dataclass
class MergeOperatorV13:
    factor_dim: int = 6

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w65_mlsc_v13_merge_operator",
            "factor_dim": int(self.factor_dim),
        })

    def merge(
            self,
            capsules: Sequence[MergeableLatentCapsuleV13],
            *,
            team_substrate_witness_chain: Sequence[str] = (),
            role_conditioned_witness_chain: Sequence[str] = (),
            algebra_signature_v13: str = (
                W56_MLSC_V4_ALGEBRA_MERGE),
            **v12_kwargs: Any,
    ) -> MergeableLatentCapsuleV13:
        if not capsules:
            raise ValueError("merge requires at least 1 capsule")
        v12_op = MergeOperatorV12(factor_dim=int(self.factor_dim))
        merged_v12 = v12_op.merge(
            [c.inner_v12 for c in capsules], **v12_kwargs)
        ts: list[str] = []
        for c in capsules:
            for a in c.team_substrate_witness_chain:
                if a not in ts:
                    ts.append(a)
        for a in team_substrate_witness_chain:
            if a not in ts:
                ts.append(a)
        rc: list[str] = []
        for c in capsules:
            for a in c.role_conditioned_witness_chain:
                if a not in rc:
                    rc.append(a)
        for a in role_conditioned_witness_chain:
            if a not in rc:
                rc.append(a)
        return wrap_v12_as_v13(
            merged_v12,
            team_substrate_witness_chain=tuple(ts),
            role_conditioned_witness_chain=tuple(rc),
            algebra_signature_v13=str(algebra_signature_v13),
        )


@dataclasses.dataclass(frozen=True)
class MergeableLatentCapsuleV13Witness:
    schema: str
    capsule_cid: str
    inner_v12_cid: str
    team_substrate_witness_chain_depth: int
    role_conditioned_witness_chain_depth: int
    algebra_signature_v13: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "capsule_cid": str(self.capsule_cid),
            "inner_v12_cid": str(self.inner_v12_cid),
            "team_substrate_witness_chain_depth": int(
                self.team_substrate_witness_chain_depth),
            "role_conditioned_witness_chain_depth": int(
                self.role_conditioned_witness_chain_depth),
            "algebra_signature_v13": str(
                self.algebra_signature_v13),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w65_mlsc_v13_witness",
            "witness": self.to_dict()})


def emit_mlsc_v13_witness(
        capsule: MergeableLatentCapsuleV13,
) -> MergeableLatentCapsuleV13Witness:
    return MergeableLatentCapsuleV13Witness(
        schema=W65_MLSC_V13_SCHEMA_VERSION,
        capsule_cid=str(capsule.cid()),
        inner_v12_cid=str(capsule.inner_v12.cid()),
        team_substrate_witness_chain_depth=int(len(
            capsule.team_substrate_witness_chain)),
        role_conditioned_witness_chain_depth=int(len(
            capsule.role_conditioned_witness_chain)),
        algebra_signature_v13=str(capsule.algebra_signature_v13),
    )


__all__ = [
    "W65_MLSC_V13_SCHEMA_VERSION",
    "W65_MLSC_V13_ALGEBRA_TEAM_SUBSTRATE_PROPAGATION",
    "W65_MLSC_V13_ALGEBRA_ROLE_CONDITIONED_PROPAGATION",
    "W65_MLSC_V13_KNOWN_ALGEBRA_SIGNATURES",
    "MergeableLatentCapsuleV13",
    "wrap_v12_as_v13",
    "MergeOperatorV13",
    "MergeableLatentCapsuleV13Witness",
    "emit_mlsc_v13_witness",
]
