"""W73 — Mergeable Latent State Capsule V21 (MLSC V21).

Strictly extends W72's ``coordpy.mergeable_latent_capsule_v20``.
V20 added restart-repair-trajectory and rejoin-pressure chains.
V21 adds:

* ``replacement_repair_trajectory_chain`` — content-addressed
  witness chain for per-turn replacement-repair-trajectory CIDs
  (V18 axis).
* ``contradiction_chain`` — content-addressed witness chain for
  per-turn contradiction events.
* ``algebra_signature_v21`` — adds two new V21 propagation
  signatures.

V21 merge inherits the unions of both new chains.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from .mergeable_latent_capsule_v4 import W56_MLSC_V4_ALGEBRA_MERGE
from .mergeable_latent_capsule_v20 import (
    MergeableLatentCapsuleV20, MergeOperatorV20,
)
from .tiny_substrate_v3 import _sha256_hex


W73_MLSC_V21_SCHEMA_VERSION: str = (
    "coordpy.mergeable_latent_capsule_v21.v1")
W73_MLSC_V21_ALGEBRA_REPLACEMENT_REPAIR_PROPAGATION: str = (
    "replacement_repair_propagation_v21")
W73_MLSC_V21_ALGEBRA_CONTRADICTION_PROPAGATION: str = (
    "contradiction_propagation_v21")
W73_MLSC_V21_KNOWN_ALGEBRA_SIGNATURES: tuple[str, ...] = (
    W56_MLSC_V4_ALGEBRA_MERGE,
    W73_MLSC_V21_ALGEBRA_REPLACEMENT_REPAIR_PROPAGATION,
    W73_MLSC_V21_ALGEBRA_CONTRADICTION_PROPAGATION,
)


@dataclasses.dataclass(frozen=True)
class MergeableLatentCapsuleV21:
    schema: str
    inner_v20: MergeableLatentCapsuleV20
    replacement_repair_trajectory_chain: tuple[str, ...]
    contradiction_chain: tuple[str, ...]
    algebra_signature_v21: str

    @property
    def payload(self) -> tuple[float, ...]:
        return self.inner_v20.payload

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "inner_v20_cid": str(self.inner_v20.cid()),
            "replacement_repair_trajectory_chain": list(
                self.replacement_repair_trajectory_chain),
            "contradiction_chain": list(
                self.contradiction_chain),
            "algebra_signature_v21": str(
                self.algebra_signature_v21),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w73_mlsc_v21",
            "capsule": self.to_dict()})


def wrap_v20_as_v21(
        v20_capsule: MergeableLatentCapsuleV20, *,
        replacement_repair_trajectory_chain: Sequence[str] = (),
        contradiction_chain: Sequence[str] = (),
        algebra_signature_v21: str = W56_MLSC_V4_ALGEBRA_MERGE,
) -> MergeableLatentCapsuleV21:
    if (algebra_signature_v21 not in
            W73_MLSC_V21_KNOWN_ALGEBRA_SIGNATURES):
        algebra_signature_v21 = W56_MLSC_V4_ALGEBRA_MERGE
    return MergeableLatentCapsuleV21(
        schema=W73_MLSC_V21_SCHEMA_VERSION,
        inner_v20=v20_capsule,
        replacement_repair_trajectory_chain=tuple(
            str(s)
            for s in replacement_repair_trajectory_chain),
        contradiction_chain=tuple(
            str(s) for s in contradiction_chain),
        algebra_signature_v21=str(algebra_signature_v21),
    )


@dataclasses.dataclass
class MergeOperatorV21:
    factor_dim: int = 6

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w73_mlsc_v21_merge_operator",
            "factor_dim": int(self.factor_dim),
        })

    def merge(
            self,
            capsules: Sequence[MergeableLatentCapsuleV21],
            *,
            replacement_repair_trajectory_chain: Sequence[
                str] = (),
            contradiction_chain: Sequence[str] = (),
            algebra_signature_v21: str = (
                W56_MLSC_V4_ALGEBRA_MERGE),
            **v20_kwargs: Any,
    ) -> MergeableLatentCapsuleV21:
        if not capsules:
            raise ValueError(
                "merge requires at least 1 capsule")
        v20_op = MergeOperatorV20(factor_dim=int(self.factor_dim))
        merged_v20 = v20_op.merge(
            [c.inner_v20 for c in capsules], **v20_kwargs)
        rrt: list[str] = []
        for c in capsules:
            for a in c.replacement_repair_trajectory_chain:
                if a not in rrt:
                    rrt.append(a)
        for a in replacement_repair_trajectory_chain:
            if a not in rrt:
                rrt.append(a)
        ct: list[str] = []
        for c in capsules:
            for a in c.contradiction_chain:
                if a not in ct:
                    ct.append(a)
        for a in contradiction_chain:
            if a not in ct:
                ct.append(a)
        return wrap_v20_as_v21(
            merged_v20,
            replacement_repair_trajectory_chain=tuple(rrt),
            contradiction_chain=tuple(ct),
            algebra_signature_v21=str(algebra_signature_v21),
        )


@dataclasses.dataclass(frozen=True)
class MergeableLatentCapsuleV21Witness:
    schema: str
    capsule_cid: str
    inner_v20_cid: str
    replacement_repair_trajectory_chain_depth: int
    contradiction_chain_depth: int
    algebra_signature_v21: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "capsule_cid": str(self.capsule_cid),
            "inner_v20_cid": str(self.inner_v20_cid),
            "replacement_repair_trajectory_chain_depth": int(
                self.replacement_repair_trajectory_chain_depth),
            "contradiction_chain_depth": int(
                self.contradiction_chain_depth),
            "algebra_signature_v21": str(
                self.algebra_signature_v21),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w73_mlsc_v21_witness",
            "witness": self.to_dict()})


def emit_mlsc_v21_witness(
        capsule: MergeableLatentCapsuleV21,
) -> MergeableLatentCapsuleV21Witness:
    return MergeableLatentCapsuleV21Witness(
        schema=W73_MLSC_V21_SCHEMA_VERSION,
        capsule_cid=str(capsule.cid()),
        inner_v20_cid=str(capsule.inner_v20.cid()),
        replacement_repair_trajectory_chain_depth=int(len(
            capsule.replacement_repair_trajectory_chain)),
        contradiction_chain_depth=int(len(
            capsule.contradiction_chain)),
        algebra_signature_v21=str(
            capsule.algebra_signature_v21),
    )


__all__ = [
    "W73_MLSC_V21_SCHEMA_VERSION",
    "W73_MLSC_V21_ALGEBRA_REPLACEMENT_REPAIR_PROPAGATION",
    "W73_MLSC_V21_ALGEBRA_CONTRADICTION_PROPAGATION",
    "W73_MLSC_V21_KNOWN_ALGEBRA_SIGNATURES",
    "MergeableLatentCapsuleV21",
    "wrap_v20_as_v21",
    "MergeOperatorV21",
    "MergeableLatentCapsuleV21Witness",
    "emit_mlsc_v21_witness",
]
