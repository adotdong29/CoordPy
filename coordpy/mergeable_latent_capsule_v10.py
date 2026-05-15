"""W62 M10 — Mergeable Latent State Capsule V10.

Strictly extends W61's ``coordpy.mergeable_latent_capsule_v9``.
V9 added ``attention_pattern_witness_chain``,
``cache_retrieval_witness_chain``, and a per-(layer, head) trust
matrix. V10 adds:

* ``replay_dominance_witness_chain`` — union-inherited at merge
  time. Each entry is a CID of a witness from the V62 replay
  controller V3.
* ``disagreement_wasserstein_distance`` — a Wasserstein-1
  distance scalar computed at merge time between the parents'
  payload distributions. Read by the consensus controller V8.
* Two new algebra signatures:
  ``replay_dominance_propagation`` and
  ``wasserstein_disagreement``.

Honest scope
------------

* Replay-dominance chain inherits as union; no probabilistic
  weighting.
* Wasserstein-1 distance is the L1 difference between the sorted
  empirical distributions of the parent payloads, scaled by
  1/dim. Not a calibrated Wasserstein on continuous distributions.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Mapping, Sequence

from .mergeable_latent_capsule_v4 import (
    W56_MLSC_V4_ALGEBRA_MERGE,
)
from .mergeable_latent_capsule_v9 import (
    MergeableLatentCapsuleV9,
    MergeOperatorV9,
    W61_MLSC_V9_KNOWN_ALGEBRA_SIGNATURES,
    wrap_v8_as_v9,
)
from .tiny_substrate_v3 import _sha256_hex


W62_MLSC_V10_SCHEMA_VERSION: str = (
    "coordpy.mergeable_latent_capsule_v10.v1")

W62_MLSC_V10_ALGEBRA_REPLAY_DOMINANCE_PROPAGATION: str = (
    "replay_dominance_propagation")
W62_MLSC_V10_ALGEBRA_WASSERSTEIN_DISAGREEMENT: str = (
    "wasserstein_disagreement")

W62_MLSC_V10_KNOWN_ALGEBRA_SIGNATURES: tuple[str, ...] = (
    *W61_MLSC_V9_KNOWN_ALGEBRA_SIGNATURES,
    W62_MLSC_V10_ALGEBRA_REPLAY_DOMINANCE_PROPAGATION,
    W62_MLSC_V10_ALGEBRA_WASSERSTEIN_DISAGREEMENT,
)


def _wasserstein_1_dist(
        a: Sequence[float], b: Sequence[float],
) -> float:
    """L1 distance between sorted samples (1/dim normalised)."""
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    sa = sorted(float(x) for x in a)[:n]
    sb = sorted(float(x) for x in b)[:n]
    if n == 0:
        return 0.0
    return float(sum(abs(x - y)
                     for x, y in zip(sa, sb)) / float(n))


@dataclasses.dataclass(frozen=True)
class MergeableLatentCapsuleV10:
    schema: str
    inner_v9: MergeableLatentCapsuleV9
    replay_dominance_witness_chain: tuple[str, ...]
    disagreement_wasserstein_distance: float
    algebra_signature_v10: str

    @property
    def payload(self) -> tuple[float, ...]:
        return self.inner_v9.payload

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "inner_v9_cid": str(self.inner_v9.cid()),
            "replay_dominance_witness_chain": list(
                self.replay_dominance_witness_chain),
            "disagreement_wasserstein_distance": float(round(
                self.disagreement_wasserstein_distance, 12)),
            "algebra_signature_v10": str(
                self.algebra_signature_v10),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w62_mlsc_v10",
            "capsule": self.to_dict()})


def wrap_v9_as_v10(
        v9_capsule: MergeableLatentCapsuleV9, *,
        replay_dominance_witness_chain: Sequence[str] = (),
        disagreement_wasserstein_distance: float = 0.0,
        algebra_signature_v10: str = W56_MLSC_V4_ALGEBRA_MERGE,
) -> MergeableLatentCapsuleV10:
    if (algebra_signature_v10 not in
            W62_MLSC_V10_KNOWN_ALGEBRA_SIGNATURES):
        algebra_signature_v10 = W56_MLSC_V4_ALGEBRA_MERGE
    return MergeableLatentCapsuleV10(
        schema=W62_MLSC_V10_SCHEMA_VERSION,
        inner_v9=v9_capsule,
        replay_dominance_witness_chain=tuple(
            str(s) for s in replay_dominance_witness_chain),
        disagreement_wasserstein_distance=float(
            disagreement_wasserstein_distance),
        algebra_signature_v10=str(algebra_signature_v10),
    )


@dataclasses.dataclass
class MergeOperatorV10:
    factor_dim: int = 6

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w62_mlsc_v10_merge_operator",
            "factor_dim": int(self.factor_dim),
        })

    def merge(
            self,
            capsules: Sequence[MergeableLatentCapsuleV10],
            *,
            replay_dominance_witness_chain: Sequence[str] = (),
            algebra_signature_v10: str = (
                W56_MLSC_V4_ALGEBRA_MERGE),
            **v9_kwargs: Any,
    ) -> MergeableLatentCapsuleV10:
        if not capsules:
            raise ValueError(
                "merge requires at least 1 capsule")
        v9_op = MergeOperatorV9(
            factor_dim=int(self.factor_dim))
        merged_v9 = v9_op.merge(
            [c.inner_v9 for c in capsules], **v9_kwargs)
        # Union over replay_dominance_witness_chain.
        rd_set: list[str] = []
        for c in capsules:
            for a in c.replay_dominance_witness_chain:
                if a not in rd_set:
                    rd_set.append(a)
        for a in replay_dominance_witness_chain:
            if a not in rd_set:
                rd_set.append(a)
        # Wasserstein-1 between first two parents' payloads.
        wass = 0.0
        if len(capsules) >= 2:
            wass = _wasserstein_1_dist(
                list(capsules[0].payload),
                list(capsules[1].payload))
        return wrap_v9_as_v10(
            merged_v9,
            replay_dominance_witness_chain=tuple(rd_set),
            disagreement_wasserstein_distance=float(wass),
            algebra_signature_v10=str(algebra_signature_v10),
        )


@dataclasses.dataclass(frozen=True)
class MergeableLatentCapsuleV10Witness:
    schema: str
    capsule_cid: str
    inner_v9_cid: str
    replay_dominance_witness_chain_depth: int
    disagreement_wasserstein_distance: float
    algebra_signature_v10: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "capsule_cid": str(self.capsule_cid),
            "inner_v9_cid": str(self.inner_v9_cid),
            "replay_dominance_witness_chain_depth": int(
                self.replay_dominance_witness_chain_depth),
            "disagreement_wasserstein_distance": float(round(
                self.disagreement_wasserstein_distance, 12)),
            "algebra_signature_v10": str(
                self.algebra_signature_v10),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w62_mlsc_v10_witness",
            "witness": self.to_dict()})


def emit_mlsc_v10_witness(
        capsule: MergeableLatentCapsuleV10,
) -> MergeableLatentCapsuleV10Witness:
    return MergeableLatentCapsuleV10Witness(
        schema=W62_MLSC_V10_SCHEMA_VERSION,
        capsule_cid=str(capsule.cid()),
        inner_v9_cid=str(capsule.inner_v9.cid()),
        replay_dominance_witness_chain_depth=int(len(
            capsule.replay_dominance_witness_chain)),
        disagreement_wasserstein_distance=float(
            capsule.disagreement_wasserstein_distance),
        algebra_signature_v10=str(
            capsule.algebra_signature_v10),
    )


__all__ = [
    "W62_MLSC_V10_SCHEMA_VERSION",
    "W62_MLSC_V10_ALGEBRA_REPLAY_DOMINANCE_PROPAGATION",
    "W62_MLSC_V10_ALGEBRA_WASSERSTEIN_DISAGREEMENT",
    "W62_MLSC_V10_KNOWN_ALGEBRA_SIGNATURES",
    "MergeableLatentCapsuleV10",
    "wrap_v9_as_v10",
    "MergeOperatorV10",
    "MergeableLatentCapsuleV10Witness",
    "emit_mlsc_v10_witness",
]
