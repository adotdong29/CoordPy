"""W77 — Mergeable Latent State Capsule V25 (MLSC V25).

Strictly extends W76's ``coordpy.mergeable_latent_capsule_v24``.
V24 added chain-then-restart-trajectory and post-compound-chain-
restart chains. V25 adds:

* ``post_restart_replacement_trajectory_chain`` — content-
  addressed witness chain for per-turn post-restart-replacement
  trajectory CIDs (V22 axis).
* ``post_restart_replacement_window_chain`` — content-addressed
  witness chain for per-turn post-restart-replacement windows.
* ``algebra_signature_v25`` — adds two new V25 propagation
  signatures.

V25 merge inherits the unions of both new chains.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from .mergeable_latent_capsule_v4 import W56_MLSC_V4_ALGEBRA_MERGE
from .mergeable_latent_capsule_v24 import (
    MergeableLatentCapsuleV24, MergeOperatorV24,
)
from .tiny_substrate_v3 import _sha256_hex


W77_MLSC_V25_SCHEMA_VERSION: str = (
    "coordpy.mergeable_latent_capsule_v25.v1")
W77_MLSC_V25_ALGEBRA_POST_RESTART_REPLACEMENT_PROPAGATION: str = (
    "post_restart_replacement_propagation_v25")
W77_MLSC_V25_ALGEBRA_POST_RESTART_REPLACEMENT_WINDOW_PROPAGATION: (
    str) = (
    "post_restart_replacement_window_propagation_v25")
W77_MLSC_V25_KNOWN_ALGEBRA_SIGNATURES: tuple[str, ...] = (
    W56_MLSC_V4_ALGEBRA_MERGE,
    W77_MLSC_V25_ALGEBRA_POST_RESTART_REPLACEMENT_PROPAGATION,
    W77_MLSC_V25_ALGEBRA_POST_RESTART_REPLACEMENT_WINDOW_PROPAGATION,
)


@dataclasses.dataclass(frozen=True)
class MergeableLatentCapsuleV25:
    schema: str
    inner_v24: MergeableLatentCapsuleV24
    post_restart_replacement_trajectory_chain: tuple[str, ...]
    post_restart_replacement_window_chain: tuple[str, ...]
    algebra_signature_v25: str

    @property
    def payload(self) -> tuple[float, ...]:
        return self.inner_v24.payload

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "inner_v24_cid": str(self.inner_v24.cid()),
            "post_restart_replacement_trajectory_chain": list(
                self.post_restart_replacement_trajectory_chain),
            "post_restart_replacement_window_chain": list(
                self.post_restart_replacement_window_chain),
            "algebra_signature_v25": str(
                self.algebra_signature_v25),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w77_mlsc_v25",
            "capsule": self.to_dict()})


def wrap_v24_as_v25(
        v24_capsule: MergeableLatentCapsuleV24, *,
        post_restart_replacement_trajectory_chain: Sequence[
            str] = (),
        post_restart_replacement_window_chain: Sequence[str] = (),
        algebra_signature_v25: str = W56_MLSC_V4_ALGEBRA_MERGE,
) -> MergeableLatentCapsuleV25:
    if (algebra_signature_v25 not in
            W77_MLSC_V25_KNOWN_ALGEBRA_SIGNATURES):
        algebra_signature_v25 = W56_MLSC_V4_ALGEBRA_MERGE
    return MergeableLatentCapsuleV25(
        schema=W77_MLSC_V25_SCHEMA_VERSION,
        inner_v24=v24_capsule,
        post_restart_replacement_trajectory_chain=tuple(
            str(s)
            for s in post_restart_replacement_trajectory_chain),
        post_restart_replacement_window_chain=tuple(
            str(s) for s
            in post_restart_replacement_window_chain),
        algebra_signature_v25=str(algebra_signature_v25),
    )


@dataclasses.dataclass
class MergeOperatorV25:
    factor_dim: int = 6

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w77_mlsc_v25_merge_operator",
            "factor_dim": int(self.factor_dim),
        })

    def merge(
            self,
            capsules: Sequence[MergeableLatentCapsuleV25],
            *,
            post_restart_replacement_trajectory_chain: Sequence[
                str] = (),
            post_restart_replacement_window_chain: Sequence[
                str] = (),
            algebra_signature_v25: str = (
                W56_MLSC_V4_ALGEBRA_MERGE),
            **v24_kwargs: Any,
    ) -> MergeableLatentCapsuleV25:
        if not capsules:
            raise ValueError(
                "merge requires at least 1 capsule")
        v24_op = MergeOperatorV24(
            factor_dim=int(self.factor_dim))
        merged_v24 = v24_op.merge(
            [c.inner_v24 for c in capsules], **v24_kwargs)
        prt: list[str] = []
        for c in capsules:
            for a in c.post_restart_replacement_trajectory_chain:
                if a not in prt:
                    prt.append(a)
        for a in post_restart_replacement_trajectory_chain:
            if a not in prt:
                prt.append(a)
        prw: list[str] = []
        for c in capsules:
            for a in c.post_restart_replacement_window_chain:
                if a not in prw:
                    prw.append(a)
        for a in post_restart_replacement_window_chain:
            if a not in prw:
                prw.append(a)
        return wrap_v24_as_v25(
            merged_v24,
            post_restart_replacement_trajectory_chain=tuple(prt),
            post_restart_replacement_window_chain=tuple(prw),
            algebra_signature_v25=str(algebra_signature_v25),
        )


@dataclasses.dataclass(frozen=True)
class MergeableLatentCapsuleV25Witness:
    schema: str
    capsule_cid: str
    inner_v24_cid: str
    post_restart_replacement_trajectory_chain_depth: int
    post_restart_replacement_window_chain_depth: int
    algebra_signature_v25: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "capsule_cid": str(self.capsule_cid),
            "inner_v24_cid": str(self.inner_v24_cid),
            "post_restart_replacement_trajectory_chain_depth": int(
                self
                .post_restart_replacement_trajectory_chain_depth),
            "post_restart_replacement_window_chain_depth": int(
                self.post_restart_replacement_window_chain_depth),
            "algebra_signature_v25": str(
                self.algebra_signature_v25),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w77_mlsc_v25_witness",
            "witness": self.to_dict()})


def emit_mlsc_v25_witness(
        capsule: MergeableLatentCapsuleV25,
) -> MergeableLatentCapsuleV25Witness:
    return MergeableLatentCapsuleV25Witness(
        schema=W77_MLSC_V25_SCHEMA_VERSION,
        capsule_cid=str(capsule.cid()),
        inner_v24_cid=str(capsule.inner_v24.cid()),
        post_restart_replacement_trajectory_chain_depth=int(len(
            capsule.post_restart_replacement_trajectory_chain)),
        post_restart_replacement_window_chain_depth=int(len(
            capsule.post_restart_replacement_window_chain)),
        algebra_signature_v25=str(
            capsule.algebra_signature_v25),
    )


__all__ = [
    "W77_MLSC_V25_SCHEMA_VERSION",
    "W77_MLSC_V25_ALGEBRA_POST_RESTART_REPLACEMENT_PROPAGATION",
    "W77_MLSC_V25_ALGEBRA_POST_RESTART_REPLACEMENT_WINDOW_PROPAGATION",
    "W77_MLSC_V25_KNOWN_ALGEBRA_SIGNATURES",
    "MergeableLatentCapsuleV25",
    "wrap_v24_as_v25",
    "MergeOperatorV25",
    "MergeableLatentCapsuleV25Witness",
    "emit_mlsc_v25_witness",
]
