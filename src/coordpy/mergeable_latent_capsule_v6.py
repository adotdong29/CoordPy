"""W58 M10 — Mergeable Latent State Capsule V6 (MLSC V6).

Strictly extends W57's ``coordpy.mergeable_latent_capsule_v5``.
V6 adds:

* **``attention_witness_chain``** — V5 carried a *single*
  ``attention_witness_cid``. V6 carries the full chain of
  attention witnesses through the merge tree (union inherited
  from parents).
* **``cache_reuse_witness_cid``** — fingerprint of the substrate
  V3 KV cache at the point this capsule was emitted. Lets the
  consensus controller V4 see whether two parents agree on the
  *cache state*, not just the payload.
* **``algebra_signature_v4``** — adds two new primitives:

    * ``cache_reuse_replay`` — payload was reproduced via prefix-
      state reuse.
    * ``attention_steer``    — payload was produced under an
      attention-steering bias.

V6 strictly extends V5 byte-for-byte when:

  * ``attention_witness_chain == ()``
  * ``cache_reuse_witness_cid == ""``
  * ``algebra_signature_v4`` ∈ V5's known signatures.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Sequence

from .mergeable_latent_capsule_v5 import (
    MergeableLatentCapsuleV5,
    W57_MLSC_V5_ALGEBRA_HIDDEN_INJECT,
    W57_MLSC_V5_ALGEBRA_SUBSTRATE_PROJECT,
    W57_MLSC_V5_KNOWN_ALGEBRA_SIGNATURES,
)
from .mergeable_latent_capsule_v4 import (
    W56_MLSC_V4_ALGEBRA_DIFF,
    W56_MLSC_V4_ALGEBRA_INTERSECT,
    W56_MLSC_V4_ALGEBRA_MERGE,
)


W58_MLSC_V6_SCHEMA_VERSION: str = (
    "coordpy.mergeable_latent_capsule_v6.v1")
W58_MLSC_V6_ALGEBRA_CACHE_REUSE_REPLAY: str = "cache_reuse_replay"
W58_MLSC_V6_ALGEBRA_ATTENTION_STEER: str = "attention_steer"

W58_MLSC_V6_KNOWN_ALGEBRA_SIGNATURES: tuple[str, ...] = (
    W56_MLSC_V4_ALGEBRA_MERGE,
    W56_MLSC_V4_ALGEBRA_DIFF,
    W56_MLSC_V4_ALGEBRA_INTERSECT,
    W57_MLSC_V5_ALGEBRA_SUBSTRATE_PROJECT,
    W57_MLSC_V5_ALGEBRA_HIDDEN_INJECT,
    W58_MLSC_V6_ALGEBRA_CACHE_REUSE_REPLAY,
    W58_MLSC_V6_ALGEBRA_ATTENTION_STEER,
)


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass(frozen=True)
class MergeableLatentCapsuleV6:
    schema: str
    inner_v5: MergeableLatentCapsuleV5
    attention_witness_chain: tuple[str, ...]
    cache_reuse_witness_cid: str
    algebra_signature_v4: str

    @property
    def payload(self) -> tuple[float, ...]:
        return self.inner_v5.payload

    @property
    def trust(self) -> float:
        return float(self.inner_v5.trust)

    @property
    def confidence(self) -> float:
        return float(self.inner_v5.confidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "inner_v5_cid": str(self.inner_v5.cid()),
            "attention_witness_chain": list(
                self.attention_witness_chain),
            "cache_reuse_witness_cid": str(
                self.cache_reuse_witness_cid),
            "algebra_signature_v4": str(
                self.algebra_signature_v4),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w58_mlsc_v6",
            "capsule": self.to_dict()})


def wrap_v5_as_v6(
        v5_capsule: MergeableLatentCapsuleV5, *,
        attention_witness_chain: Sequence[str] = (),
        cache_reuse_witness_cid: str = "",
        algebra_signature_v4: str = W56_MLSC_V4_ALGEBRA_MERGE,
) -> MergeableLatentCapsuleV6:
    if algebra_signature_v4 not in W58_MLSC_V6_KNOWN_ALGEBRA_SIGNATURES:
        algebra_signature_v4 = W56_MLSC_V4_ALGEBRA_MERGE
    return MergeableLatentCapsuleV6(
        schema=W58_MLSC_V6_SCHEMA_VERSION,
        inner_v5=v5_capsule,
        attention_witness_chain=tuple(
            str(s) for s in attention_witness_chain),
        cache_reuse_witness_cid=str(cache_reuse_witness_cid),
        algebra_signature_v4=str(algebra_signature_v4),
    )


@dataclasses.dataclass
class MergeOperatorV6:
    """V6 merge operator: extends V5 merge with attention chain
    inheritance + cache-reuse witness propagation."""

    factor_dim: int = 6

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w58_mlsc_v6_merge_operator",
            "factor_dim": int(self.factor_dim),
        })

    def merge(
            self, capsules: Sequence[MergeableLatentCapsuleV6], *,
            attention_witness_chain: Sequence[str] = (),
            cache_reuse_witness_cid: str = "",
            algebra_signature_v4: str = (
                W56_MLSC_V4_ALGEBRA_MERGE),
            per_head_trust: Sequence[float] | None = None,
    ) -> MergeableLatentCapsuleV6:
        if not capsules:
            raise ValueError("merge requires at least 1 capsule")
        # Delegate the V5 merge first.
        from .mergeable_latent_capsule_v5 import MergeOperatorV5
        v5_op = MergeOperatorV5(factor_dim=int(self.factor_dim))
        inner_v5_capsules = [c.inner_v5 for c in capsules]
        # Hidden-state-witness-chain inheritance handled by V5.
        # Attention-witness-cid we use for V5 input.
        merged_v5 = v5_op.merge(
            inner_v5_capsules,
            algebra_signature_v3=str(algebra_signature_v4)
                if algebra_signature_v4
                in W57_MLSC_V5_KNOWN_ALGEBRA_SIGNATURES
                else W56_MLSC_V4_ALGEBRA_MERGE,
            per_head_trust=per_head_trust,
        )
        # Attention witness chain inherits union from parents.
        attn_chain_set: list[str] = []
        for c in capsules:
            for a in c.attention_witness_chain:
                if a not in attn_chain_set:
                    attn_chain_set.append(a)
        for a in attention_witness_chain:
            if a not in attn_chain_set:
                attn_chain_set.append(a)
        return wrap_v5_as_v6(
            merged_v5,
            attention_witness_chain=tuple(attn_chain_set),
            cache_reuse_witness_cid=str(cache_reuse_witness_cid),
            algebra_signature_v4=str(algebra_signature_v4),
        )


@dataclasses.dataclass(frozen=True)
class MergeableLatentCapsuleV6Witness:
    schema: str
    capsule_cid: str
    inner_v5_cid: str
    attention_witness_chain_depth: int
    cache_reuse_witness_cid: str
    algebra_signature_v4: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "capsule_cid": str(self.capsule_cid),
            "inner_v5_cid": str(self.inner_v5_cid),
            "attention_witness_chain_depth": int(
                self.attention_witness_chain_depth),
            "cache_reuse_witness_cid": str(
                self.cache_reuse_witness_cid),
            "algebra_signature_v4": str(
                self.algebra_signature_v4),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w58_mlsc_v6_witness",
            "witness": self.to_dict()})


def emit_mlsc_v6_witness(
        capsule: MergeableLatentCapsuleV6,
) -> MergeableLatentCapsuleV6Witness:
    return MergeableLatentCapsuleV6Witness(
        schema=W58_MLSC_V6_SCHEMA_VERSION,
        capsule_cid=str(capsule.cid()),
        inner_v5_cid=str(capsule.inner_v5.cid()),
        attention_witness_chain_depth=int(
            len(capsule.attention_witness_chain)),
        cache_reuse_witness_cid=str(
            capsule.cache_reuse_witness_cid),
        algebra_signature_v4=str(capsule.algebra_signature_v4),
    )


__all__ = [
    "W58_MLSC_V6_SCHEMA_VERSION",
    "W58_MLSC_V6_ALGEBRA_CACHE_REUSE_REPLAY",
    "W58_MLSC_V6_ALGEBRA_ATTENTION_STEER",
    "W58_MLSC_V6_KNOWN_ALGEBRA_SIGNATURES",
    "MergeableLatentCapsuleV6",
    "MergeableLatentCapsuleV6Witness",
    "MergeOperatorV6",
    "wrap_v5_as_v6",
    "emit_mlsc_v6_witness",
]
