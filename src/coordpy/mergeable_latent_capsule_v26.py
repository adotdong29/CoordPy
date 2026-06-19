"""W78 — Mergeable Latent State Capsule V26 (MLSC V26).

Strictly extends W77's ``coordpy.mergeable_latent_capsule_v25``.
V26 adds:

* ``long_horizon_reconstruction_trajectory_chain`` — content-
  addressed witness chain for per-turn long-horizon-reconstruction
  trajectory CIDs (V23 axis).
* ``reconstruction_request_window_chain`` — content-addressed
  witness chain for per-turn reconstruction-request windows.
* ``algebra_signature_v26`` — adds two new V26 propagation
  signatures.

V26 merge inherits the unions of both new chains.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from .mergeable_latent_capsule_v4 import W56_MLSC_V4_ALGEBRA_MERGE
from .mergeable_latent_capsule_v25 import (
    MergeableLatentCapsuleV25, MergeOperatorV25,
)
from .tiny_substrate_v3 import _sha256_hex


W78_MLSC_V26_SCHEMA_VERSION: str = (
    "coordpy.mergeable_latent_capsule_v26.v1")
W78_MLSC_V26_ALGEBRA_LONG_HORIZON_RECONSTRUCTION_PROPAGATION: str = (
    "long_horizon_reconstruction_propagation_v26")
W78_MLSC_V26_ALGEBRA_RECONSTRUCTION_REQUEST_WINDOW_PROPAGATION: (
    str) = (
    "reconstruction_request_window_propagation_v26")
W78_MLSC_V26_KNOWN_ALGEBRA_SIGNATURES: tuple[str, ...] = (
    W56_MLSC_V4_ALGEBRA_MERGE,
    W78_MLSC_V26_ALGEBRA_LONG_HORIZON_RECONSTRUCTION_PROPAGATION,
    W78_MLSC_V26_ALGEBRA_RECONSTRUCTION_REQUEST_WINDOW_PROPAGATION,
)


@dataclasses.dataclass(frozen=True)
class MergeableLatentCapsuleV26:
    schema: str
    inner_v25: MergeableLatentCapsuleV25
    long_horizon_reconstruction_trajectory_chain: tuple[str, ...]
    reconstruction_request_window_chain: tuple[str, ...]
    algebra_signature_v26: str

    @property
    def payload(self) -> tuple[float, ...]:
        return self.inner_v25.payload

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "inner_v25_cid": str(self.inner_v25.cid()),
            "long_horizon_reconstruction_trajectory_chain": list(
                self
                .long_horizon_reconstruction_trajectory_chain),
            "reconstruction_request_window_chain": list(
                self.reconstruction_request_window_chain),
            "algebra_signature_v26": str(
                self.algebra_signature_v26),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w78_mlsc_v26",
            "capsule": self.to_dict()})


def wrap_v25_as_v26(
        v25_capsule: MergeableLatentCapsuleV25, *,
        long_horizon_reconstruction_trajectory_chain: Sequence[
            str] = (),
        reconstruction_request_window_chain: Sequence[str] = (),
        algebra_signature_v26: str = W56_MLSC_V4_ALGEBRA_MERGE,
) -> MergeableLatentCapsuleV26:
    if (algebra_signature_v26 not in
            W78_MLSC_V26_KNOWN_ALGEBRA_SIGNATURES):
        algebra_signature_v26 = W56_MLSC_V4_ALGEBRA_MERGE
    return MergeableLatentCapsuleV26(
        schema=W78_MLSC_V26_SCHEMA_VERSION,
        inner_v25=v25_capsule,
        long_horizon_reconstruction_trajectory_chain=tuple(
            str(s) for s
            in long_horizon_reconstruction_trajectory_chain),
        reconstruction_request_window_chain=tuple(
            str(s) for s
            in reconstruction_request_window_chain),
        algebra_signature_v26=str(algebra_signature_v26),
    )


@dataclasses.dataclass
class MergeOperatorV26:
    factor_dim: int = 6

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w78_mlsc_v26_merge_operator",
            "factor_dim": int(self.factor_dim),
        })

    def merge(
            self,
            capsules: Sequence[MergeableLatentCapsuleV26],
            *,
            long_horizon_reconstruction_trajectory_chain: (
                Sequence[str]) = (),
            reconstruction_request_window_chain: Sequence[
                str] = (),
            algebra_signature_v26: str = (
                W56_MLSC_V4_ALGEBRA_MERGE),
            **v25_kwargs: Any,
    ) -> MergeableLatentCapsuleV26:
        if not capsules:
            raise ValueError(
                "merge requires at least 1 capsule")
        v25_op = MergeOperatorV25(
            factor_dim=int(self.factor_dim))
        merged_v25 = v25_op.merge(
            [c.inner_v25 for c in capsules], **v25_kwargs)
        lhr: list[str] = []
        for c in capsules:
            for a in (
                    c
                    .long_horizon_reconstruction_trajectory_chain):
                if a not in lhr:
                    lhr.append(a)
        for a in long_horizon_reconstruction_trajectory_chain:
            if a not in lhr:
                lhr.append(a)
        rrw: list[str] = []
        for c in capsules:
            for a in c.reconstruction_request_window_chain:
                if a not in rrw:
                    rrw.append(a)
        for a in reconstruction_request_window_chain:
            if a not in rrw:
                rrw.append(a)
        return wrap_v25_as_v26(
            merged_v25,
            long_horizon_reconstruction_trajectory_chain=tuple(
                lhr),
            reconstruction_request_window_chain=tuple(rrw),
            algebra_signature_v26=str(algebra_signature_v26),
        )


@dataclasses.dataclass(frozen=True)
class MergeableLatentCapsuleV26Witness:
    schema: str
    capsule_cid: str
    inner_v25_cid: str
    long_horizon_reconstruction_trajectory_chain_depth: int
    reconstruction_request_window_chain_depth: int
    algebra_signature_v26: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "capsule_cid": str(self.capsule_cid),
            "inner_v25_cid": str(self.inner_v25_cid),
            "long_horizon_reconstruction_trajectory_chain_depth":
                int(
                    self
                    .long_horizon_reconstruction_trajectory_chain_depth),
            "reconstruction_request_window_chain_depth": int(
                self.reconstruction_request_window_chain_depth),
            "algebra_signature_v26": str(
                self.algebra_signature_v26),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w78_mlsc_v26_witness",
            "witness": self.to_dict()})


def emit_mlsc_v26_witness(
        capsule: MergeableLatentCapsuleV26,
) -> MergeableLatentCapsuleV26Witness:
    return MergeableLatentCapsuleV26Witness(
        schema=W78_MLSC_V26_SCHEMA_VERSION,
        capsule_cid=str(capsule.cid()),
        inner_v25_cid=str(capsule.inner_v25.cid()),
        long_horizon_reconstruction_trajectory_chain_depth=int(
            len(
                capsule
                .long_horizon_reconstruction_trajectory_chain)),
        reconstruction_request_window_chain_depth=int(len(
            capsule.reconstruction_request_window_chain)),
        algebra_signature_v26=str(
            capsule.algebra_signature_v26),
    )


__all__ = [
    "W78_MLSC_V26_SCHEMA_VERSION",
    "W78_MLSC_V26_ALGEBRA_LONG_HORIZON_RECONSTRUCTION_PROPAGATION",
    "W78_MLSC_V26_ALGEBRA_RECONSTRUCTION_REQUEST_WINDOW_PROPAGATION",
    "W78_MLSC_V26_KNOWN_ALGEBRA_SIGNATURES",
    "MergeableLatentCapsuleV26",
    "wrap_v25_as_v26",
    "MergeOperatorV26",
    "MergeableLatentCapsuleV26Witness",
    "emit_mlsc_v26_witness",
]
