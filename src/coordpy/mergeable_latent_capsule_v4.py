"""W56 M6 — Mergeable Latent State Capsule V4 (MLSC V4).

Extends W55 MLSC V3 with three additions:

(a) ``substrate_witness`` — a content-addressed witness CID of
the tiny substrate's hidden state at the merge timestamp. When
the substrate signal is unavailable, the witness is the empty
string (honest passthrough to V3 semantics).

(b) per-fact ``provenance_chain`` — walks back through the merge
ancestors to a root capsule. The chain is content-addressed.

(c) ``algebra_signature`` — which disagreement-algebra primitive
(⊕/⊖/⊗) produced the merge, recorded inline.

The V4 module re-uses every V3 invariant: content-addressed
audit, trust signature decay, K-of-N consensus quorum, abstain
semantics. ``MergeOperatorV4`` strictly extends ``MergeOperatorV3``
— V4 reduces to V3 byte-for-byte when ``substrate_witness=""``
and ``algebra_signature="merge"``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Sequence

from .mergeable_latent_capsule_v3 import (
    FactConfirmation,
    MergeAuditTrailV3,
    MergeOperatorV3,
    MergeableLatentCapsuleV3,
    W55_DEFAULT_MLSC_V3_FACTOR_DIM,
    W55_DEFAULT_MLSC_V3_TRUST_DEFAULT,
    W55_DEFAULT_MLSC_V3_TRUST_DECAY,
    W55_DEFAULT_MLSC_V3_TRUST_FLOOR,
)


W56_MLSC_V4_SCHEMA_VERSION: str = (
    "coordpy.mergeable_latent_capsule_v4.v1")
W56_MLSC_V4_ALGEBRA_MERGE: str = "merge"
W56_MLSC_V4_ALGEBRA_DIFF: str = "diff"
W56_MLSC_V4_ALGEBRA_INTERSECT: str = "intersect"


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass(frozen=True)
class MergeableLatentCapsuleV4:
    """V4 capsule = V3 capsule + substrate witness + algebra
    signature + per-fact provenance chain.
    """

    schema: str
    inner_v3: MergeableLatentCapsuleV3
    substrate_witness_cid: str
    algebra_signature: str
    per_fact_provenance: tuple[tuple[str, tuple[str, ...]], ...]

    @property
    def payload(self) -> tuple[float, ...]:
        return self.inner_v3.payload

    @property
    def trust(self) -> float:
        return float(self.inner_v3.trust)

    @property
    def confidence(self) -> float:
        return float(self.inner_v3.confidence)

    @property
    def fact_confirmations(self) -> tuple[FactConfirmation, ...]:
        return self.inner_v3.fact_confirmations

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "inner_v3_cid": str(self.inner_v3.cid()),
            "substrate_witness_cid": str(
                self.substrate_witness_cid),
            "algebra_signature": str(self.algebra_signature),
            "per_fact_provenance": [
                [str(tag), list(chain)]
                for tag, chain in self.per_fact_provenance],
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w56_mlsc_v4",
            "capsule": self.to_dict()})


def wrap_v3_as_v4(
        v3_capsule: MergeableLatentCapsuleV3, *,
        substrate_witness_cid: str = "",
        algebra_signature: str = W56_MLSC_V4_ALGEBRA_MERGE,
        per_fact_provenance: (
            Sequence[tuple[str, Sequence[str]]] | None) = None,
) -> MergeableLatentCapsuleV4:
    """Lift a V3 capsule into a V4 capsule."""
    if per_fact_provenance is None:
        # Derive provenance from V3 fact confirmations.
        per_fact_provenance = tuple(
            (fc.tag, (str(fc.tag),))
            for fc in v3_capsule.fact_confirmations)
    else:
        per_fact_provenance = tuple(
            (str(tag), tuple(str(c) for c in chain))
            for tag, chain in per_fact_provenance)
    return MergeableLatentCapsuleV4(
        schema=W56_MLSC_V4_SCHEMA_VERSION,
        inner_v3=v3_capsule,
        substrate_witness_cid=str(substrate_witness_cid),
        algebra_signature=str(algebra_signature),
        per_fact_provenance=tuple(per_fact_provenance),
    )


@dataclasses.dataclass
class MergeOperatorV4:
    """V4 merge operator: extends V3 with substrate-witness +
    algebra-signature recording.

    The V4 operator wraps a V3 operator. Merging two V4 capsules
    first merges their inner V3 capsules (preserving V3 audit),
    then records the substrate witness + algebra signature in the
    V4 envelope.
    """

    factor_dim: int = W55_DEFAULT_MLSC_V3_FACTOR_DIM
    inner_v3: MergeOperatorV3 = dataclasses.field(
        default_factory=lambda: MergeOperatorV3(
            factor_dim=W55_DEFAULT_MLSC_V3_FACTOR_DIM))

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W56_MLSC_V4_SCHEMA_VERSION,
            "kind": "w56_mlsc_v4_operator",
            "factor_dim": int(self.factor_dim),
            "inner_v3_cid": str(self.inner_v3.cid()),
        })

    def merge(
            self,
            parents: Sequence[MergeableLatentCapsuleV4],
            *,
            extra_fact_tags: Sequence[str] = (),
            turn_index: int | None = None,
            substrate_witness_cid: str = "",
            algebra_signature: str = W56_MLSC_V4_ALGEBRA_MERGE,
    ) -> MergeableLatentCapsuleV4:
        v3_parents = [p.inner_v3 for p in parents]
        merged_v3 = self.inner_v3.merge(
            v3_parents,
            extra_fact_tags=tuple(extra_fact_tags),
            turn_index=turn_index)
        # Build per-fact provenance: union of parents' provenance
        # plus the merged tag → parent_capsule_cids chain.
        prov: dict[str, list[str]] = {}
        for p in parents:
            for tag, chain in p.per_fact_provenance:
                prov.setdefault(str(tag), [])
                for c in chain:
                    if c not in prov[str(tag)]:
                        prov[str(tag)].append(str(c))
            # also add the parent CID itself as a provenance step
            for fc in p.fact_confirmations:
                prov.setdefault(str(fc.tag), [])
                if p.inner_v3.cid() not in prov[str(fc.tag)]:
                    prov[str(fc.tag)].append(p.inner_v3.cid())
        per_fact_prov = tuple(
            (tag, tuple(prov[tag]))
            for tag in sorted(prov.keys()))
        return MergeableLatentCapsuleV4(
            schema=W56_MLSC_V4_SCHEMA_VERSION,
            inner_v3=merged_v3,
            substrate_witness_cid=str(substrate_witness_cid),
            algebra_signature=str(algebra_signature),
            per_fact_provenance=per_fact_prov,
        )


@dataclasses.dataclass(frozen=True)
class MergeableLatentCapsuleV4Witness:
    schema: str
    v3_witness_cid: str
    substrate_witness_cid: str
    algebra_signature: str
    n_provenance_chains: int
    deepest_provenance_chain: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "v3_witness_cid": str(self.v3_witness_cid),
            "substrate_witness_cid": str(
                self.substrate_witness_cid),
            "algebra_signature": str(self.algebra_signature),
            "n_provenance_chains": int(self.n_provenance_chains),
            "deepest_provenance_chain": int(
                self.deepest_provenance_chain),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w56_mlsc_v4_witness",
            "witness": self.to_dict()})


def emit_mlsc_v4_witness(
        *,
        capsule: MergeableLatentCapsuleV4,
        v3_witness_cid: str,
) -> MergeableLatentCapsuleV4Witness:
    deepest = 0
    for tag, chain in capsule.per_fact_provenance:
        if len(chain) > deepest:
            deepest = int(len(chain))
    return MergeableLatentCapsuleV4Witness(
        schema=W56_MLSC_V4_SCHEMA_VERSION,
        v3_witness_cid=str(v3_witness_cid),
        substrate_witness_cid=str(
            capsule.substrate_witness_cid),
        algebra_signature=str(capsule.algebra_signature),
        n_provenance_chains=int(len(capsule.per_fact_provenance)),
        deepest_provenance_chain=int(deepest),
    )


__all__ = [
    "W56_MLSC_V4_SCHEMA_VERSION",
    "W56_MLSC_V4_ALGEBRA_MERGE",
    "W56_MLSC_V4_ALGEBRA_DIFF",
    "W56_MLSC_V4_ALGEBRA_INTERSECT",
    "MergeableLatentCapsuleV4",
    "MergeableLatentCapsuleV4Witness",
    "MergeOperatorV4",
    "wrap_v3_as_v4",
    "emit_mlsc_v4_witness",
]
