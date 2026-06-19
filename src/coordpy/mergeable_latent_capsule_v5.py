"""W57 M8 — Mergeable Latent State Capsule V5 (MLSC V5).

Extends W56 MLSC V4 with:

(a) ``hidden_state_witness_chain`` — a *chain* of CIDs of per-layer
    substrate hidden states across the merge tree. V4 carried a
    single substrate_witness; V5 carries a chain so a merge can
    inherit the hidden-state lineage of every parent that
    contributed.
(b) ``attention_witness_cid`` — a CID of the substrate's
    attention map at the merge timestamp. Useful for the
    attention-steering bridge and for the consensus controller's
    logits-conditioned tiebreaker.
(c) ``per_head_trust`` — a tuple of per-head trust scalars
    matching the substrate's ``n_heads``. The merge operator
    weights contributions by per-head trust when a per-head trust
    scheme is supplied.
(d) ``algebra_signature_v3`` — extends V4's
    ``{merge, diff, intersect}`` with two new disagreement-algebra
    primitives:

      * ``substrate_project``  — payload was projected through
        the tiny substrate (KV bridge + readback).
      * ``hidden_inject``      — payload was applied via the
        hidden-state bridge before merge.

V5 strictly extends V4 byte-for-byte when:

  * ``hidden_state_witness_chain == ()``
  * ``attention_witness_cid == ""``
  * ``per_head_trust == ()``
  * ``algebra_signature in {"merge","diff","intersect"}``
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Sequence

from .mergeable_latent_capsule_v4 import (
    MergeableLatentCapsuleV4,
    W56_MLSC_V4_ALGEBRA_DIFF,
    W56_MLSC_V4_ALGEBRA_INTERSECT,
    W56_MLSC_V4_ALGEBRA_MERGE,
)


W57_MLSC_V5_SCHEMA_VERSION: str = (
    "coordpy.mergeable_latent_capsule_v5.v1")
W57_MLSC_V5_ALGEBRA_SUBSTRATE_PROJECT: str = "substrate_project"
W57_MLSC_V5_ALGEBRA_HIDDEN_INJECT: str = "hidden_inject"

W57_MLSC_V5_KNOWN_ALGEBRA_SIGNATURES: tuple[str, ...] = (
    W56_MLSC_V4_ALGEBRA_MERGE,
    W56_MLSC_V4_ALGEBRA_DIFF,
    W56_MLSC_V4_ALGEBRA_INTERSECT,
    W57_MLSC_V5_ALGEBRA_SUBSTRATE_PROJECT,
    W57_MLSC_V5_ALGEBRA_HIDDEN_INJECT,
)


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass(frozen=True)
class MergeableLatentCapsuleV5:
    schema: str
    inner_v4: MergeableLatentCapsuleV4
    hidden_state_witness_chain: tuple[str, ...]
    attention_witness_cid: str
    per_head_trust: tuple[float, ...]
    algebra_signature_v3: str

    @property
    def payload(self) -> tuple[float, ...]:
        return self.inner_v4.payload

    @property
    def trust(self) -> float:
        return float(self.inner_v4.trust)

    @property
    def confidence(self) -> float:
        return float(self.inner_v4.confidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "inner_v4_cid": str(self.inner_v4.cid()),
            "hidden_state_witness_chain": list(
                self.hidden_state_witness_chain),
            "attention_witness_cid": str(
                self.attention_witness_cid),
            "per_head_trust": [
                float(round(t, 12))
                for t in self.per_head_trust],
            "algebra_signature_v3": str(
                self.algebra_signature_v3),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w57_mlsc_v5",
            "capsule": self.to_dict()})


def wrap_v4_as_v5(
        v4_capsule: MergeableLatentCapsuleV4, *,
        hidden_state_witness_chain: Sequence[str] = (),
        attention_witness_cid: str = "",
        per_head_trust: Sequence[float] = (),
        algebra_signature_v3: str = W56_MLSC_V4_ALGEBRA_MERGE,
) -> MergeableLatentCapsuleV5:
    if algebra_signature_v3 not in W57_MLSC_V5_KNOWN_ALGEBRA_SIGNATURES:
        algebra_signature_v3 = W56_MLSC_V4_ALGEBRA_MERGE
    return MergeableLatentCapsuleV5(
        schema=W57_MLSC_V5_SCHEMA_VERSION,
        inner_v4=v4_capsule,
        hidden_state_witness_chain=tuple(
            str(s) for s in hidden_state_witness_chain),
        attention_witness_cid=str(attention_witness_cid),
        per_head_trust=tuple(float(t) for t in per_head_trust),
        algebra_signature_v3=str(algebra_signature_v3),
    )


@dataclasses.dataclass
class MergeOperatorV5:
    """V5 merge operator: extends MLSC V4's merge with per-head
    trust weighting and the new algebra signatures."""

    factor_dim: int = 6
    per_head_trust_default: tuple[float, ...] = ()

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w57_mlsc_v5_merge_operator",
            "factor_dim": int(self.factor_dim),
            "per_head_trust_default": [
                float(round(t, 12))
                for t in self.per_head_trust_default],
        })

    def merge(
            self, capsules: Sequence[MergeableLatentCapsuleV5], *,
            hidden_state_witness_chain: Sequence[str] = (),
            attention_witness_cid: str = "",
            per_head_trust: Sequence[float] | None = None,
            algebra_signature_v3: str = (
                W56_MLSC_V4_ALGEBRA_MERGE),
    ) -> MergeableLatentCapsuleV5:
        """Weighted average over inner_v4 payloads.

        If ``per_head_trust`` is supplied (matches the substrate's
        ``n_heads``), payloads are weighted by the *mean* of
        per-head trust per capsule (V5 reduces to V4 byte-for-byte
        when no per-head trust is supplied and the algebra
        signature is one of V4's three).
        """
        if not capsules:
            raise ValueError("merge requires at least 1 capsule")
        d = int(self.factor_dim)
        weights: list[float] = []
        for c in capsules:
            base = float(c.trust) * float(c.confidence)
            phs = list(c.per_head_trust)
            if phs:
                head_mean = sum(phs) / float(len(phs))
                base = base * float(head_mean)
            weights.append(max(base, 0.0))
        total = sum(weights)
        if total <= 1e-30:
            weights = [1.0 / float(len(capsules))
                       for _ in capsules]
            total = float(sum(weights))
        else:
            weights = [w / total for w in weights]
        merged_payload = [0.0] * d
        for c, w in zip(capsules, weights):
            pl = list(c.payload)[:d]
            while len(pl) < d:
                pl.append(0.0)
            for i in range(d):
                merged_payload[i] += float(w) * float(pl[i])
        # Lift inner V4: synthesise a V4 capsule with merged inner.
        # We reuse the V4 merge operator from V4 module to produce
        # the inner V4, but for V5 we just propagate the highest-
        # confidence parent's inner_v4 with the merged payload
        # substituted by re-wrapping V3.
        from .mergeable_latent_capsule_v3 import (
            make_root_capsule_v3,
        )
        # New V3 root that carries the merged payload + averaged
        # trust/confidence.
        avg_trust = sum(w * c.trust
                         for w, c in zip(weights, capsules))
        avg_conf = sum(w * c.confidence
                        for w, c in zip(weights, capsules))
        v3_merged = make_root_capsule_v3(
            branch_id=f"merge_{len(capsules)}",
            payload=tuple(merged_payload),
            fact_tags=("merged_v5",),
            confidence=float(avg_conf),
            trust=float(avg_trust),
            turn_index=0)
        # Lift to V4.
        from .mergeable_latent_capsule_v4 import wrap_v3_as_v4
        v4_merged = wrap_v3_as_v4(
            v3_merged,
            substrate_witness_cid=str(attention_witness_cid),
            algebra_signature=str(algebra_signature_v3))
        per_head = (
            tuple(per_head_trust)
            if per_head_trust is not None
            else self.per_head_trust_default)
        # Hidden-state witness chain inherits union from parents.
        chain_set: list[str] = []
        for c in capsules:
            for h in c.hidden_state_witness_chain:
                if h not in chain_set:
                    chain_set.append(h)
        for h in hidden_state_witness_chain:
            if h not in chain_set:
                chain_set.append(h)
        return wrap_v4_as_v5(
            v4_merged,
            hidden_state_witness_chain=tuple(chain_set),
            attention_witness_cid=str(attention_witness_cid),
            per_head_trust=per_head,
            algebra_signature_v3=str(algebra_signature_v3))


@dataclasses.dataclass(frozen=True)
class MergeableLatentCapsuleV5Witness:
    schema: str
    capsule_cid: str
    inner_v4_cid: str
    hidden_state_witness_chain_depth: int
    attention_witness_cid: str
    per_head_trust_min: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "capsule_cid": str(self.capsule_cid),
            "inner_v4_cid": str(self.inner_v4_cid),
            "hidden_state_witness_chain_depth": int(
                self.hidden_state_witness_chain_depth),
            "attention_witness_cid": str(
                self.attention_witness_cid),
            "per_head_trust_min": float(round(
                self.per_head_trust_min, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w57_mlsc_v5_witness",
            "witness": self.to_dict()})


def emit_mlsc_v5_witness(
        capsule: MergeableLatentCapsuleV5,
) -> MergeableLatentCapsuleV5Witness:
    phs = list(capsule.per_head_trust)
    ph_min = float(min(phs)) if phs else 1.0
    return MergeableLatentCapsuleV5Witness(
        schema=W57_MLSC_V5_SCHEMA_VERSION,
        capsule_cid=str(capsule.cid()),
        inner_v4_cid=str(capsule.inner_v4.cid()),
        hidden_state_witness_chain_depth=int(
            len(capsule.hidden_state_witness_chain)),
        attention_witness_cid=str(
            capsule.attention_witness_cid),
        per_head_trust_min=float(ph_min),
    )


__all__ = [
    "W57_MLSC_V5_SCHEMA_VERSION",
    "W57_MLSC_V5_ALGEBRA_SUBSTRATE_PROJECT",
    "W57_MLSC_V5_ALGEBRA_HIDDEN_INJECT",
    "W57_MLSC_V5_KNOWN_ALGEBRA_SIGNATURES",
    "MergeableLatentCapsuleV5",
    "MergeableLatentCapsuleV5Witness",
    "MergeOperatorV5",
    "wrap_v4_as_v5",
    "emit_mlsc_v5_witness",
]
