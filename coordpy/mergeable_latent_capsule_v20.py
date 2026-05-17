"""W72 — Mergeable Latent State Capsule V20 (MLSC V20).

Strictly extends W71's ``coordpy.mergeable_latent_capsule_v19``.
V19 added delayed-repair-trajectory and restart-dominance chains.
V20 adds:

* ``restart_repair_trajectory_chain`` — content-addressed witness
  chain for per-turn restart-repair-trajectory CIDs (V17 axis).
* ``rejoin_pressure_chain`` — content-addressed witness chain for
  per-turn rejoin-pressure signals.
* ``algebra_signature_v20`` — adds two new V20 propagation
  signatures.

V20 merge inherits the unions of both new chains.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from .mergeable_latent_capsule_v4 import W56_MLSC_V4_ALGEBRA_MERGE
from .mergeable_latent_capsule_v19 import (
    MergeableLatentCapsuleV19, MergeOperatorV19,
)
from .tiny_substrate_v3 import _sha256_hex


W72_MLSC_V20_SCHEMA_VERSION: str = (
    "coordpy.mergeable_latent_capsule_v20.v1")
W72_MLSC_V20_ALGEBRA_RESTART_REPAIR_PROPAGATION: str = (
    "restart_repair_propagation_v20")
W72_MLSC_V20_ALGEBRA_REJOIN_PRESSURE_PROPAGATION: str = (
    "rejoin_pressure_propagation_v20")
W72_MLSC_V20_KNOWN_ALGEBRA_SIGNATURES: tuple[str, ...] = (
    W56_MLSC_V4_ALGEBRA_MERGE,
    W72_MLSC_V20_ALGEBRA_RESTART_REPAIR_PROPAGATION,
    W72_MLSC_V20_ALGEBRA_REJOIN_PRESSURE_PROPAGATION,
)


@dataclasses.dataclass(frozen=True)
class MergeableLatentCapsuleV20:
    schema: str
    inner_v19: MergeableLatentCapsuleV19
    restart_repair_trajectory_chain: tuple[str, ...]
    rejoin_pressure_chain: tuple[str, ...]
    algebra_signature_v20: str

    @property
    def payload(self) -> tuple[float, ...]:
        return self.inner_v19.payload

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "inner_v19_cid": str(self.inner_v19.cid()),
            "restart_repair_trajectory_chain": list(
                self.restart_repair_trajectory_chain),
            "rejoin_pressure_chain": list(
                self.rejoin_pressure_chain),
            "algebra_signature_v20": str(
                self.algebra_signature_v20),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w72_mlsc_v20",
            "capsule": self.to_dict()})


def wrap_v19_as_v20(
        v19_capsule: MergeableLatentCapsuleV19, *,
        restart_repair_trajectory_chain: Sequence[str] = (),
        rejoin_pressure_chain: Sequence[str] = (),
        algebra_signature_v20: str = W56_MLSC_V4_ALGEBRA_MERGE,
) -> MergeableLatentCapsuleV20:
    if (algebra_signature_v20 not in
            W72_MLSC_V20_KNOWN_ALGEBRA_SIGNATURES):
        algebra_signature_v20 = W56_MLSC_V4_ALGEBRA_MERGE
    return MergeableLatentCapsuleV20(
        schema=W72_MLSC_V20_SCHEMA_VERSION,
        inner_v19=v19_capsule,
        restart_repair_trajectory_chain=tuple(
            str(s) for s in restart_repair_trajectory_chain),
        rejoin_pressure_chain=tuple(
            str(s) for s in rejoin_pressure_chain),
        algebra_signature_v20=str(algebra_signature_v20),
    )


@dataclasses.dataclass
class MergeOperatorV20:
    factor_dim: int = 6

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w72_mlsc_v20_merge_operator",
            "factor_dim": int(self.factor_dim),
        })

    def merge(
            self,
            capsules: Sequence[MergeableLatentCapsuleV20],
            *,
            restart_repair_trajectory_chain: Sequence[str] = (),
            rejoin_pressure_chain: Sequence[str] = (),
            algebra_signature_v20: str = (
                W56_MLSC_V4_ALGEBRA_MERGE),
            **v19_kwargs: Any,
    ) -> MergeableLatentCapsuleV20:
        if not capsules:
            raise ValueError(
                "merge requires at least 1 capsule")
        v19_op = MergeOperatorV19(factor_dim=int(self.factor_dim))
        merged_v19 = v19_op.merge(
            [c.inner_v19 for c in capsules], **v19_kwargs)
        rrt: list[str] = []
        for c in capsules:
            for a in c.restart_repair_trajectory_chain:
                if a not in rrt:
                    rrt.append(a)
        for a in restart_repair_trajectory_chain:
            if a not in rrt:
                rrt.append(a)
        rj: list[str] = []
        for c in capsules:
            for a in c.rejoin_pressure_chain:
                if a not in rj:
                    rj.append(a)
        for a in rejoin_pressure_chain:
            if a not in rj:
                rj.append(a)
        return wrap_v19_as_v20(
            merged_v19,
            restart_repair_trajectory_chain=tuple(rrt),
            rejoin_pressure_chain=tuple(rj),
            algebra_signature_v20=str(algebra_signature_v20),
        )


@dataclasses.dataclass(frozen=True)
class MergeableLatentCapsuleV20Witness:
    schema: str
    capsule_cid: str
    inner_v19_cid: str
    restart_repair_trajectory_chain_depth: int
    rejoin_pressure_chain_depth: int
    algebra_signature_v20: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "capsule_cid": str(self.capsule_cid),
            "inner_v19_cid": str(self.inner_v19_cid),
            "restart_repair_trajectory_chain_depth": int(
                self.restart_repair_trajectory_chain_depth),
            "rejoin_pressure_chain_depth": int(
                self.rejoin_pressure_chain_depth),
            "algebra_signature_v20": str(
                self.algebra_signature_v20),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w72_mlsc_v20_witness",
            "witness": self.to_dict()})


def emit_mlsc_v20_witness(
        capsule: MergeableLatentCapsuleV20,
) -> MergeableLatentCapsuleV20Witness:
    return MergeableLatentCapsuleV20Witness(
        schema=W72_MLSC_V20_SCHEMA_VERSION,
        capsule_cid=str(capsule.cid()),
        inner_v19_cid=str(capsule.inner_v19.cid()),
        restart_repair_trajectory_chain_depth=int(len(
            capsule.restart_repair_trajectory_chain)),
        rejoin_pressure_chain_depth=int(len(
            capsule.rejoin_pressure_chain)),
        algebra_signature_v20=str(
            capsule.algebra_signature_v20),
    )


__all__ = [
    "W72_MLSC_V20_SCHEMA_VERSION",
    "W72_MLSC_V20_ALGEBRA_RESTART_REPAIR_PROPAGATION",
    "W72_MLSC_V20_ALGEBRA_REJOIN_PRESSURE_PROPAGATION",
    "W72_MLSC_V20_KNOWN_ALGEBRA_SIGNATURES",
    "MergeableLatentCapsuleV20",
    "wrap_v19_as_v20",
    "MergeOperatorV20",
    "MergeableLatentCapsuleV20Witness",
    "emit_mlsc_v20_witness",
]
