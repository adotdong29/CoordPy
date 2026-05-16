"""W66 M12 — Mergeable Latent State Capsule V14 (MLSC V14).

Strictly extends W65's ``coordpy.mergeable_latent_capsule_v13``.
V13 added team-substrate and role-conditioned chains. V14 adds:

* ``team_failure_recovery_witness_chain`` — content-addressed
  witness chain for team-failure-recovery events.
* ``team_consensus_under_budget_witness_chain`` — content-
  addressed witness chain for team-consensus-under-budget events.
* ``algebra_signature_v14`` — adds two new V14 propagation
  signatures.

V14 merge inherits the unions of both new chains.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from .mergeable_latent_capsule_v4 import W56_MLSC_V4_ALGEBRA_MERGE
from .mergeable_latent_capsule_v13 import (
    MergeableLatentCapsuleV13, MergeOperatorV13,
)
from .tiny_substrate_v3 import _sha256_hex


W66_MLSC_V14_SCHEMA_VERSION: str = (
    "coordpy.mergeable_latent_capsule_v14.v1")
W66_MLSC_V14_ALGEBRA_TEAM_FAILURE_RECOVERY_PROPAGATION: str = (
    "team_failure_recovery_propagation_v14")
W66_MLSC_V14_ALGEBRA_TEAM_CONSENSUS_UNDER_BUDGET_PROPAGATION: str = (
    "team_consensus_under_budget_propagation_v14")
W66_MLSC_V14_KNOWN_ALGEBRA_SIGNATURES: tuple[str, ...] = (
    W56_MLSC_V4_ALGEBRA_MERGE,
    W66_MLSC_V14_ALGEBRA_TEAM_FAILURE_RECOVERY_PROPAGATION,
    W66_MLSC_V14_ALGEBRA_TEAM_CONSENSUS_UNDER_BUDGET_PROPAGATION,
)


@dataclasses.dataclass(frozen=True)
class MergeableLatentCapsuleV14:
    schema: str
    inner_v13: MergeableLatentCapsuleV13
    team_failure_recovery_witness_chain: tuple[str, ...]
    team_consensus_under_budget_witness_chain: tuple[str, ...]
    algebra_signature_v14: str

    @property
    def payload(self) -> tuple[float, ...]:
        return self.inner_v13.payload

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "inner_v13_cid": str(self.inner_v13.cid()),
            "team_failure_recovery_witness_chain": list(
                self.team_failure_recovery_witness_chain),
            "team_consensus_under_budget_witness_chain": list(
                self.team_consensus_under_budget_witness_chain),
            "algebra_signature_v14": str(
                self.algebra_signature_v14),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w66_mlsc_v14",
            "capsule": self.to_dict()})


def wrap_v13_as_v14(
        v13_capsule: MergeableLatentCapsuleV13, *,
        team_failure_recovery_witness_chain: Sequence[str] = (),
        team_consensus_under_budget_witness_chain: Sequence[str] = (),
        algebra_signature_v14: str = W56_MLSC_V4_ALGEBRA_MERGE,
) -> MergeableLatentCapsuleV14:
    if (algebra_signature_v14 not in
            W66_MLSC_V14_KNOWN_ALGEBRA_SIGNATURES):
        algebra_signature_v14 = W56_MLSC_V4_ALGEBRA_MERGE
    return MergeableLatentCapsuleV14(
        schema=W66_MLSC_V14_SCHEMA_VERSION,
        inner_v13=v13_capsule,
        team_failure_recovery_witness_chain=tuple(
            str(s) for s in team_failure_recovery_witness_chain),
        team_consensus_under_budget_witness_chain=tuple(
            str(s) for s
            in team_consensus_under_budget_witness_chain),
        algebra_signature_v14=str(algebra_signature_v14),
    )


@dataclasses.dataclass
class MergeOperatorV14:
    factor_dim: int = 6

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w66_mlsc_v14_merge_operator",
            "factor_dim": int(self.factor_dim),
        })

    def merge(
            self,
            capsules: Sequence[MergeableLatentCapsuleV14],
            *,
            team_failure_recovery_witness_chain: Sequence[str] = (),
            team_consensus_under_budget_witness_chain: (
                Sequence[str]) = (),
            algebra_signature_v14: str = (
                W56_MLSC_V4_ALGEBRA_MERGE),
            **v13_kwargs: Any,
    ) -> MergeableLatentCapsuleV14:
        if not capsules:
            raise ValueError(
                "merge requires at least 1 capsule")
        v13_op = MergeOperatorV13(factor_dim=int(self.factor_dim))
        merged_v13 = v13_op.merge(
            [c.inner_v13 for c in capsules], **v13_kwargs)
        tfr: list[str] = []
        for c in capsules:
            for a in c.team_failure_recovery_witness_chain:
                if a not in tfr:
                    tfr.append(a)
        for a in team_failure_recovery_witness_chain:
            if a not in tfr:
                tfr.append(a)
        tcb: list[str] = []
        for c in capsules:
            for a in c.team_consensus_under_budget_witness_chain:
                if a not in tcb:
                    tcb.append(a)
        for a in team_consensus_under_budget_witness_chain:
            if a not in tcb:
                tcb.append(a)
        return wrap_v13_as_v14(
            merged_v13,
            team_failure_recovery_witness_chain=tuple(tfr),
            team_consensus_under_budget_witness_chain=tuple(tcb),
            algebra_signature_v14=str(algebra_signature_v14),
        )


@dataclasses.dataclass(frozen=True)
class MergeableLatentCapsuleV14Witness:
    schema: str
    capsule_cid: str
    inner_v13_cid: str
    team_failure_recovery_chain_depth: int
    team_consensus_under_budget_chain_depth: int
    algebra_signature_v14: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "capsule_cid": str(self.capsule_cid),
            "inner_v13_cid": str(self.inner_v13_cid),
            "team_failure_recovery_chain_depth": int(
                self.team_failure_recovery_chain_depth),
            "team_consensus_under_budget_chain_depth": int(
                self.team_consensus_under_budget_chain_depth),
            "algebra_signature_v14": str(
                self.algebra_signature_v14),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w66_mlsc_v14_witness",
            "witness": self.to_dict()})


def emit_mlsc_v14_witness(
        capsule: MergeableLatentCapsuleV14,
) -> MergeableLatentCapsuleV14Witness:
    return MergeableLatentCapsuleV14Witness(
        schema=W66_MLSC_V14_SCHEMA_VERSION,
        capsule_cid=str(capsule.cid()),
        inner_v13_cid=str(capsule.inner_v13.cid()),
        team_failure_recovery_chain_depth=int(len(
            capsule.team_failure_recovery_witness_chain)),
        team_consensus_under_budget_chain_depth=int(len(
            capsule.team_consensus_under_budget_witness_chain)),
        algebra_signature_v14=str(capsule.algebra_signature_v14),
    )


__all__ = [
    "W66_MLSC_V14_SCHEMA_VERSION",
    "W66_MLSC_V14_ALGEBRA_TEAM_FAILURE_RECOVERY_PROPAGATION",
    "W66_MLSC_V14_ALGEBRA_TEAM_CONSENSUS_UNDER_BUDGET_PROPAGATION",
    "W66_MLSC_V14_KNOWN_ALGEBRA_SIGNATURES",
    "MergeableLatentCapsuleV14",
    "wrap_v13_as_v14",
    "MergeOperatorV14",
    "MergeableLatentCapsuleV14Witness",
    "emit_mlsc_v14_witness",
]
