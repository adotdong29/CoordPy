"""W79 — Mergeable Latent State Capsule V27 (MLSC V27).

Strictly extends W78's ``coordpy.mergeable_latent_capsule_v26``.
V27 adds:

* ``replacement_then_restart_after_long_delay_trajectory_chain``
  — content-addressed witness chain for per-turn replacement-
  then-restart-after-long-delay trajectory CIDs (W79 axis).
* ``replacement_then_restart_after_long_delay_window_chain`` —
  content-addressed witness chain for per-turn replacement-then-
  restart-after-long-delay windows.
* ``algebra_signature_v27`` — adds two new V27 propagation
  signatures.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Sequence

from .mergeable_latent_capsule_v4 import W56_MLSC_V4_ALGEBRA_MERGE
from .mergeable_latent_capsule_v26 import (
    MergeOperatorV26, MergeableLatentCapsuleV26,
)


W79_MLSC_V27_SCHEMA_VERSION: str = (
    "coordpy.mergeable_latent_capsule_v27.v1")
W79_MLSC_V27_ALGEBRA_REPLACEMENT_THEN_RESTART_AFTER_LONG_DELAY_PROPAGATION: (
    str) = (
    "replacement_then_restart_after_long_delay_propagation_v27")
W79_MLSC_V27_ALGEBRA_REPLACEMENT_THEN_RESTART_AFTER_LONG_DELAY_WINDOW_PROPAGATION: (
    str) = (
    "replacement_then_restart_after_long_delay_window_propagation_v27")
W79_MLSC_V27_KNOWN_ALGEBRA_SIGNATURES: tuple[str, ...] = (
    W56_MLSC_V4_ALGEBRA_MERGE,
    W79_MLSC_V27_ALGEBRA_REPLACEMENT_THEN_RESTART_AFTER_LONG_DELAY_PROPAGATION,
    W79_MLSC_V27_ALGEBRA_REPLACEMENT_THEN_RESTART_AFTER_LONG_DELAY_WINDOW_PROPAGATION,
)


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass(frozen=True)
class MergeableLatentCapsuleV27:
    schema: str
    inner_v26: MergeableLatentCapsuleV26
    replacement_then_restart_after_long_delay_trajectory_chain: (
        tuple[str, ...])
    replacement_then_restart_after_long_delay_window_chain: tuple[
        str, ...]
    algebra_signature_v27: str

    @property
    def payload(self) -> tuple[float, ...]:
        return self.inner_v26.payload

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "inner_v26_cid": str(self.inner_v26.cid()),
            "replacement_then_restart_after_long_delay_trajectory_chain": list(
                self
                .replacement_then_restart_after_long_delay_trajectory_chain),
            "replacement_then_restart_after_long_delay_window_chain": list(
                self
                .replacement_then_restart_after_long_delay_window_chain),
            "algebra_signature_v27": str(
                self.algebra_signature_v27),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w79_mlsc_v27",
            "capsule": self.to_dict()})


def wrap_v26_as_v27(
        v26_capsule: MergeableLatentCapsuleV26, *,
        replacement_then_restart_after_long_delay_trajectory_chain: (
            Sequence[str]) = (),
        replacement_then_restart_after_long_delay_window_chain: (
            Sequence[str]) = (),
        algebra_signature_v27: str = W56_MLSC_V4_ALGEBRA_MERGE,
) -> MergeableLatentCapsuleV27:
    if (algebra_signature_v27 not in
            W79_MLSC_V27_KNOWN_ALGEBRA_SIGNATURES):
        algebra_signature_v27 = W56_MLSC_V4_ALGEBRA_MERGE
    return MergeableLatentCapsuleV27(
        schema=W79_MLSC_V27_SCHEMA_VERSION,
        inner_v26=v26_capsule,
        replacement_then_restart_after_long_delay_trajectory_chain=tuple(
            str(s) for s in
            replacement_then_restart_after_long_delay_trajectory_chain),
        replacement_then_restart_after_long_delay_window_chain=tuple(
            str(s) for s in
            replacement_then_restart_after_long_delay_window_chain),
        algebra_signature_v27=str(algebra_signature_v27),
    )


@dataclasses.dataclass
class MergeOperatorV27:
    factor_dim: int = 6

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w79_mlsc_v27_merge_operator",
            "factor_dim": int(self.factor_dim),
        })

    def merge(
            self,
            capsules: Sequence[MergeableLatentCapsuleV27],
            *,
            replacement_then_restart_after_long_delay_trajectory_chain: (
                Sequence[str]) = (),
            replacement_then_restart_after_long_delay_window_chain: (
                Sequence[str]) = (),
            algebra_signature_v27: str = (
                W56_MLSC_V4_ALGEBRA_MERGE),
            **v26_kwargs: Any,
    ) -> MergeableLatentCapsuleV27:
        if not capsules:
            raise ValueError(
                "merge requires at least 1 capsule")
        v26_op = MergeOperatorV26(
            factor_dim=int(self.factor_dim))
        merged_v26 = v26_op.merge(
            [c.inner_v26 for c in capsules], **v26_kwargs)
        rtr: list[str] = []
        for c in capsules:
            for a in (
                    c
                    .replacement_then_restart_after_long_delay_trajectory_chain):
                if a not in rtr:
                    rtr.append(a)
        for a in (
                replacement_then_restart_after_long_delay_trajectory_chain):
            if a not in rtr:
                rtr.append(a)
        rtw: list[str] = []
        for c in capsules:
            for a in (
                    c
                    .replacement_then_restart_after_long_delay_window_chain):
                if a not in rtw:
                    rtw.append(a)
        for a in (
                replacement_then_restart_after_long_delay_window_chain):
            if a not in rtw:
                rtw.append(a)
        return wrap_v26_as_v27(
            merged_v26,
            replacement_then_restart_after_long_delay_trajectory_chain=tuple(
                rtr),
            replacement_then_restart_after_long_delay_window_chain=tuple(
                rtw),
            algebra_signature_v27=str(algebra_signature_v27),
        )


@dataclasses.dataclass(frozen=True)
class MergeableLatentCapsuleV27Witness:
    schema: str
    capsule_cid: str
    inner_v26_cid: str
    replacement_then_restart_after_long_delay_trajectory_chain_depth: (
        int)
    replacement_then_restart_after_long_delay_window_chain_depth: (
        int)
    algebra_signature_v27: str

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w79_mlsc_v27_witness",
            "schema": str(self.schema),
            "capsule_cid": str(self.capsule_cid),
            "inner_v26_cid": str(self.inner_v26_cid),
            "replacement_then_restart_after_long_delay_trajectory_chain_depth": int(
                self
                .replacement_then_restart_after_long_delay_trajectory_chain_depth),
            "replacement_then_restart_after_long_delay_window_chain_depth": int(
                self
                .replacement_then_restart_after_long_delay_window_chain_depth),
            "algebra_signature_v27": str(
                self.algebra_signature_v27),
        })


def emit_mlsc_v27_witness(
        capsule: MergeableLatentCapsuleV27,
) -> MergeableLatentCapsuleV27Witness:
    return MergeableLatentCapsuleV27Witness(
        schema=W79_MLSC_V27_SCHEMA_VERSION,
        capsule_cid=str(capsule.cid()),
        inner_v26_cid=str(capsule.inner_v26.cid()),
        replacement_then_restart_after_long_delay_trajectory_chain_depth=int(
            len(
                capsule
                .replacement_then_restart_after_long_delay_trajectory_chain)),
        replacement_then_restart_after_long_delay_window_chain_depth=int(
            len(
                capsule
                .replacement_then_restart_after_long_delay_window_chain)),
        algebra_signature_v27=str(
            capsule.algebra_signature_v27),
    )


__all__ = [
    "W79_MLSC_V27_SCHEMA_VERSION",
    "W79_MLSC_V27_ALGEBRA_REPLACEMENT_THEN_RESTART_AFTER_LONG_DELAY_PROPAGATION",
    "W79_MLSC_V27_ALGEBRA_REPLACEMENT_THEN_RESTART_AFTER_LONG_DELAY_WINDOW_PROPAGATION",
    "W79_MLSC_V27_KNOWN_ALGEBRA_SIGNATURES",
    "MergeableLatentCapsuleV27",
    "wrap_v26_as_v27",
    "MergeOperatorV27",
    "MergeableLatentCapsuleV27Witness",
    "emit_mlsc_v27_witness",
]
