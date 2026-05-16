"""W64 M12 — Mergeable Latent State Capsule V12.

Strictly extends W63's ``coordpy.mergeable_latent_capsule_v11``.
V11 added ``hidden_wins_witness_chain``, ``prefix_reuse_witness_chain``,
and ``disagreement_jensen_shannon_distance``. V12 adds:

* ``replay_dominance_primary_witness_chain`` — union-inherited at
  merge time. Each entry is a CID of a successful
  replay-dominance-primary decision from the W64 replay
  controller V5.
* ``hidden_state_trust_witness_chain`` — union-inherited at merge
  time. Each entry is a CID of a positive hidden-state-trust
  ledger update from the W64 V9 substrate.
* ``disagreement_total_variation_distance`` — total variation
  distance scalar computed at merge time between parents'
  payload distributions (complements V11's JS distance).
* Two new algebra signatures:
  ``replay_dominance_primary_propagation`` and
  ``total_variation_disagreement``.

Honest scope (W64)
------------------

* Replay-dominance-primary / hidden-state-trust chains inherit as
  union; no probabilistic weighting.
* TV distance is computed on softmax-normalised payloads; not a
  calibrated information-theoretic distance on the underlying
  probability distributions.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from .mergeable_latent_capsule_v4 import (
    W56_MLSC_V4_ALGEBRA_MERGE,
)
from .mergeable_latent_capsule_v11 import (
    MergeableLatentCapsuleV11,
    MergeOperatorV11,
    W63_MLSC_V11_KNOWN_ALGEBRA_SIGNATURES,
    wrap_v10_as_v11,
)
from .tiny_substrate_v3 import _sha256_hex


W64_MLSC_V12_SCHEMA_VERSION: str = (
    "coordpy.mergeable_latent_capsule_v12.v1")

W64_MLSC_V12_ALGEBRA_REPLAY_DOMINANCE_PRIMARY_PROPAGATION: str = (
    "replay_dominance_primary_propagation")
W64_MLSC_V12_ALGEBRA_TOTAL_VARIATION_DISAGREEMENT: str = (
    "total_variation_disagreement")

W64_MLSC_V12_KNOWN_ALGEBRA_SIGNATURES: tuple[str, ...] = (
    *W63_MLSC_V11_KNOWN_ALGEBRA_SIGNATURES,
    W64_MLSC_V12_ALGEBRA_REPLAY_DOMINANCE_PRIMARY_PROPAGATION,
    W64_MLSC_V12_ALGEBRA_TOTAL_VARIATION_DISAGREEMENT,
)


def _softmax_arr(xs: Sequence[float]) -> list[float]:
    if not xs:
        return []
    m = max(float(x) for x in xs)
    e = [math.exp(float(x) - m) for x in xs]
    s = sum(e)
    if s <= 0.0:
        return [1.0 / len(xs)] * len(xs)
    return [v / s for v in e]


def _total_variation_distance(
        a: Sequence[float], b: Sequence[float],
) -> float:
    """Total-variation distance between softmax distributions
    over a, b. Returns 0.5 * sum |pa - pb| ∈ [0, 1]."""
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    pa = _softmax_arr(list(a)[:n])
    pb = _softmax_arr(list(b)[:n])
    return float(0.5 * sum(abs(p - q) for p, q in zip(pa, pb)))


@dataclasses.dataclass(frozen=True)
class MergeableLatentCapsuleV12:
    schema: str
    inner_v11: MergeableLatentCapsuleV11
    replay_dominance_primary_witness_chain: tuple[str, ...]
    hidden_state_trust_witness_chain: tuple[str, ...]
    disagreement_total_variation_distance: float
    algebra_signature_v12: str

    @property
    def payload(self) -> tuple[float, ...]:
        return self.inner_v11.payload

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "inner_v11_cid": str(self.inner_v11.cid()),
            "replay_dominance_primary_witness_chain": list(
                self.replay_dominance_primary_witness_chain),
            "hidden_state_trust_witness_chain": list(
                self.hidden_state_trust_witness_chain),
            "disagreement_total_variation_distance": float(
                round(
                    self.disagreement_total_variation_distance,
                    12)),
            "algebra_signature_v12": str(
                self.algebra_signature_v12),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w64_mlsc_v12",
            "capsule": self.to_dict()})


def wrap_v11_as_v12(
        v11_capsule: MergeableLatentCapsuleV11, *,
        replay_dominance_primary_witness_chain: Sequence[str] = (),
        hidden_state_trust_witness_chain: Sequence[str] = (),
        disagreement_total_variation_distance: float = 0.0,
        algebra_signature_v12: str = W56_MLSC_V4_ALGEBRA_MERGE,
) -> MergeableLatentCapsuleV12:
    if (algebra_signature_v12 not in
            W64_MLSC_V12_KNOWN_ALGEBRA_SIGNATURES):
        algebra_signature_v12 = W56_MLSC_V4_ALGEBRA_MERGE
    return MergeableLatentCapsuleV12(
        schema=W64_MLSC_V12_SCHEMA_VERSION,
        inner_v11=v11_capsule,
        replay_dominance_primary_witness_chain=tuple(
            str(s) for s in
            replay_dominance_primary_witness_chain),
        hidden_state_trust_witness_chain=tuple(
            str(s) for s in
            hidden_state_trust_witness_chain),
        disagreement_total_variation_distance=float(
            disagreement_total_variation_distance),
        algebra_signature_v12=str(algebra_signature_v12),
    )


@dataclasses.dataclass
class MergeOperatorV12:
    factor_dim: int = 6

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w64_mlsc_v12_merge_operator",
            "factor_dim": int(self.factor_dim),
        })

    def merge(
            self,
            capsules: Sequence[MergeableLatentCapsuleV12],
            *,
            replay_dominance_primary_witness_chain: (
                Sequence[str]) = (),
            hidden_state_trust_witness_chain: Sequence[str] = (),
            algebra_signature_v12: str = (
                W56_MLSC_V4_ALGEBRA_MERGE),
            **v11_kwargs: Any,
    ) -> MergeableLatentCapsuleV12:
        if not capsules:
            raise ValueError(
                "merge requires at least 1 capsule")
        v11_op = MergeOperatorV11(
            factor_dim=int(self.factor_dim))
        merged_v11 = v11_op.merge(
            [c.inner_v11 for c in capsules], **v11_kwargs)
        rdp_set: list[str] = []
        for c in capsules:
            for a in c.replay_dominance_primary_witness_chain:
                if a not in rdp_set:
                    rdp_set.append(a)
        for a in replay_dominance_primary_witness_chain:
            if a not in rdp_set:
                rdp_set.append(a)
        hst_set: list[str] = []
        for c in capsules:
            for a in c.hidden_state_trust_witness_chain:
                if a not in hst_set:
                    hst_set.append(a)
        for a in hidden_state_trust_witness_chain:
            if a not in hst_set:
                hst_set.append(a)
        tv = 0.0
        if len(capsules) >= 2:
            tv = _total_variation_distance(
                list(capsules[0].payload),
                list(capsules[1].payload))
        return wrap_v11_as_v12(
            merged_v11,
            replay_dominance_primary_witness_chain=tuple(rdp_set),
            hidden_state_trust_witness_chain=tuple(hst_set),
            disagreement_total_variation_distance=float(tv),
            algebra_signature_v12=str(algebra_signature_v12),
        )


@dataclasses.dataclass(frozen=True)
class MergeableLatentCapsuleV12Witness:
    schema: str
    capsule_cid: str
    inner_v11_cid: str
    replay_dominance_primary_witness_chain_depth: int
    hidden_state_trust_witness_chain_depth: int
    disagreement_total_variation_distance: float
    algebra_signature_v12: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "capsule_cid": str(self.capsule_cid),
            "inner_v11_cid": str(self.inner_v11_cid),
            "replay_dominance_primary_witness_chain_depth": int(
                self.replay_dominance_primary_witness_chain_depth),
            "hidden_state_trust_witness_chain_depth": int(
                self.hidden_state_trust_witness_chain_depth),
            "disagreement_total_variation_distance": float(
                round(
                    self.disagreement_total_variation_distance,
                    12)),
            "algebra_signature_v12": str(
                self.algebra_signature_v12),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w64_mlsc_v12_witness",
            "witness": self.to_dict()})


def emit_mlsc_v12_witness(
        capsule: MergeableLatentCapsuleV12,
) -> MergeableLatentCapsuleV12Witness:
    return MergeableLatentCapsuleV12Witness(
        schema=W64_MLSC_V12_SCHEMA_VERSION,
        capsule_cid=str(capsule.cid()),
        inner_v11_cid=str(capsule.inner_v11.cid()),
        replay_dominance_primary_witness_chain_depth=int(len(
            capsule.replay_dominance_primary_witness_chain)),
        hidden_state_trust_witness_chain_depth=int(len(
            capsule.hidden_state_trust_witness_chain)),
        disagreement_total_variation_distance=float(
            capsule.disagreement_total_variation_distance),
        algebra_signature_v12=str(
            capsule.algebra_signature_v12),
    )


__all__ = [
    "W64_MLSC_V12_SCHEMA_VERSION",
    "W64_MLSC_V12_ALGEBRA_REPLAY_DOMINANCE_PRIMARY_PROPAGATION",
    "W64_MLSC_V12_ALGEBRA_TOTAL_VARIATION_DISAGREEMENT",
    "W64_MLSC_V12_KNOWN_ALGEBRA_SIGNATURES",
    "MergeableLatentCapsuleV12",
    "wrap_v11_as_v12",
    "MergeOperatorV12",
    "MergeableLatentCapsuleV12Witness",
    "emit_mlsc_v12_witness",
]
