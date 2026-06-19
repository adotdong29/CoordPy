"""W76 — Mergeable Latent State Capsule V24 (MLSC V24).

Strictly extends W75's ``coordpy.mergeable_latent_capsule_v23``.
V23 added compound-chain-repair-trajectory and replacement-then-
rejoin chains. V24 adds:

* ``chain_then_restart_trajectory_chain`` — content-addressed
  witness chain for per-turn compound-chain-then-restart-
  trajectory CIDs (V21 axis).
* ``post_compound_chain_restart_chain`` — content-addressed
  witness chain for per-turn post-compound-chain-restart windows.
* ``algebra_signature_v24`` — adds two new V24 propagation
  signatures.

V24 merge inherits the unions of both new chains.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from .mergeable_latent_capsule_v4 import W56_MLSC_V4_ALGEBRA_MERGE
from .mergeable_latent_capsule_v23 import (
    MergeableLatentCapsuleV23, MergeOperatorV23,
)
from .tiny_substrate_v3 import _sha256_hex


W76_MLSC_V24_SCHEMA_VERSION: str = (
    "coordpy.mergeable_latent_capsule_v24.v1")
W76_MLSC_V24_ALGEBRA_CHAIN_THEN_RESTART_PROPAGATION: str = (
    "chain_then_restart_propagation_v24")
W76_MLSC_V24_ALGEBRA_POST_COMPOUND_CHAIN_RESTART_PROPAGATION: str = (
    "post_compound_chain_restart_propagation_v24")
W76_MLSC_V24_KNOWN_ALGEBRA_SIGNATURES: tuple[str, ...] = (
    W56_MLSC_V4_ALGEBRA_MERGE,
    W76_MLSC_V24_ALGEBRA_CHAIN_THEN_RESTART_PROPAGATION,
    W76_MLSC_V24_ALGEBRA_POST_COMPOUND_CHAIN_RESTART_PROPAGATION,
)


@dataclasses.dataclass(frozen=True)
class MergeableLatentCapsuleV24:
    schema: str
    inner_v23: MergeableLatentCapsuleV23
    chain_then_restart_trajectory_chain: tuple[str, ...]
    post_compound_chain_restart_chain: tuple[str, ...]
    algebra_signature_v24: str

    @property
    def payload(self) -> tuple[float, ...]:
        return self.inner_v23.payload

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "inner_v23_cid": str(self.inner_v23.cid()),
            "chain_then_restart_trajectory_chain": list(
                self.chain_then_restart_trajectory_chain),
            "post_compound_chain_restart_chain": list(
                self.post_compound_chain_restart_chain),
            "algebra_signature_v24": str(
                self.algebra_signature_v24),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w76_mlsc_v24",
            "capsule": self.to_dict()})


def wrap_v23_as_v24(
        v23_capsule: MergeableLatentCapsuleV23, *,
        chain_then_restart_trajectory_chain: Sequence[str] = (),
        post_compound_chain_restart_chain: Sequence[str] = (),
        algebra_signature_v24: str = W56_MLSC_V4_ALGEBRA_MERGE,
) -> MergeableLatentCapsuleV24:
    if (algebra_signature_v24 not in
            W76_MLSC_V24_KNOWN_ALGEBRA_SIGNATURES):
        algebra_signature_v24 = W56_MLSC_V4_ALGEBRA_MERGE
    return MergeableLatentCapsuleV24(
        schema=W76_MLSC_V24_SCHEMA_VERSION,
        inner_v23=v23_capsule,
        chain_then_restart_trajectory_chain=tuple(
            str(s) for s in chain_then_restart_trajectory_chain),
        post_compound_chain_restart_chain=tuple(
            str(s) for s in post_compound_chain_restart_chain),
        algebra_signature_v24=str(algebra_signature_v24),
    )


@dataclasses.dataclass
class MergeOperatorV24:
    factor_dim: int = 6

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w76_mlsc_v24_merge_operator",
            "factor_dim": int(self.factor_dim),
        })

    def merge(
            self,
            capsules: Sequence[MergeableLatentCapsuleV24],
            *,
            chain_then_restart_trajectory_chain: Sequence[
                str] = (),
            post_compound_chain_restart_chain: Sequence[str] = (),
            algebra_signature_v24: str = (
                W56_MLSC_V4_ALGEBRA_MERGE),
            **v23_kwargs: Any,
    ) -> MergeableLatentCapsuleV24:
        if not capsules:
            raise ValueError(
                "merge requires at least 1 capsule")
        v23_op = MergeOperatorV23(factor_dim=int(self.factor_dim))
        merged_v23 = v23_op.merge(
            [c.inner_v23 for c in capsules], **v23_kwargs)
        ctr: list[str] = []
        for c in capsules:
            for a in c.chain_then_restart_trajectory_chain:
                if a not in ctr:
                    ctr.append(a)
        for a in chain_then_restart_trajectory_chain:
            if a not in ctr:
                ctr.append(a)
        pcr: list[str] = []
        for c in capsules:
            for a in c.post_compound_chain_restart_chain:
                if a not in pcr:
                    pcr.append(a)
        for a in post_compound_chain_restart_chain:
            if a not in pcr:
                pcr.append(a)
        return wrap_v23_as_v24(
            merged_v23,
            chain_then_restart_trajectory_chain=tuple(ctr),
            post_compound_chain_restart_chain=tuple(pcr),
            algebra_signature_v24=str(algebra_signature_v24),
        )


@dataclasses.dataclass(frozen=True)
class MergeableLatentCapsuleV24Witness:
    schema: str
    capsule_cid: str
    inner_v23_cid: str
    chain_then_restart_trajectory_chain_depth: int
    post_compound_chain_restart_chain_depth: int
    algebra_signature_v24: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "capsule_cid": str(self.capsule_cid),
            "inner_v23_cid": str(self.inner_v23_cid),
            "chain_then_restart_trajectory_chain_depth": int(
                self.chain_then_restart_trajectory_chain_depth),
            "post_compound_chain_restart_chain_depth": int(
                self.post_compound_chain_restart_chain_depth),
            "algebra_signature_v24": str(
                self.algebra_signature_v24),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w76_mlsc_v24_witness",
            "witness": self.to_dict()})


def emit_mlsc_v24_witness(
        capsule: MergeableLatentCapsuleV24,
) -> MergeableLatentCapsuleV24Witness:
    return MergeableLatentCapsuleV24Witness(
        schema=W76_MLSC_V24_SCHEMA_VERSION,
        capsule_cid=str(capsule.cid()),
        inner_v23_cid=str(capsule.inner_v23.cid()),
        chain_then_restart_trajectory_chain_depth=int(len(
            capsule.chain_then_restart_trajectory_chain)),
        post_compound_chain_restart_chain_depth=int(len(
            capsule.post_compound_chain_restart_chain)),
        algebra_signature_v24=str(
            capsule.algebra_signature_v24),
    )


__all__ = [
    "W76_MLSC_V24_SCHEMA_VERSION",
    "W76_MLSC_V24_ALGEBRA_CHAIN_THEN_RESTART_PROPAGATION",
    "W76_MLSC_V24_ALGEBRA_POST_COMPOUND_CHAIN_RESTART_PROPAGATION",
    "W76_MLSC_V24_KNOWN_ALGEBRA_SIGNATURES",
    "MergeableLatentCapsuleV24",
    "wrap_v23_as_v24",
    "MergeOperatorV24",
    "MergeableLatentCapsuleV24Witness",
    "emit_mlsc_v24_witness",
]
