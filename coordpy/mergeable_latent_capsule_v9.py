"""W61 M10 — Mergeable Latent State Capsule V9 (MLSC V9).

Strictly extends W60's ``coordpy.mergeable_latent_capsule_v8``. V9
adds:

* **``attention_pattern_witness_chain``** — chain of attention-
  steering-V5 witness CIDs whose attention patterns the merge
  operator absorbed. Union-inherited from parents.
* **``cache_retrieval_witness_chain``** — chain of cache-
  controller-V4 bilinear-retrieval witness CIDs that produced the
  payload via cache-key-content addressing.
* **``per_layer_head_trust_matrix``** — per-(layer, head) trust
  scalar table alongside V8's per-backend trust table. The merge
  operator uses this for finer-grained trust-weighted aggregation.
* **``algebra_signature_v7``** — adds two new primitives:

    * ``attention_pattern_steer`` — payload produced by an
      attention-steering-V5 4-D budget tensor with sign falsifier.
    * ``cache_retrieval_query`` — payload produced by a V4
      bilinear retrieval head query against V6 cache_keys.

V9 strictly extends V8 byte-for-byte when:
  * ``attention_pattern_witness_chain == ()``
  * ``cache_retrieval_witness_chain == ()``
  * ``per_layer_head_trust_matrix == ()``
  * ``algebra_signature_v7`` ∈ V8's known signatures.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Mapping, Sequence

from .mergeable_latent_capsule_v4 import (
    W56_MLSC_V4_ALGEBRA_MERGE,
)
from .mergeable_latent_capsule_v8 import (
    MergeOperatorV8,
    MergeableLatentCapsuleV8,
    W60_MLSC_V8_KNOWN_ALGEBRA_SIGNATURES,
    wrap_v7_as_v8,
)


W61_MLSC_V9_SCHEMA_VERSION: str = (
    "coordpy.mergeable_latent_capsule_v9.v1")
W61_MLSC_V9_ALGEBRA_ATTENTION_PATTERN_STEER: str = (
    "attention_pattern_steer")
W61_MLSC_V9_ALGEBRA_CACHE_RETRIEVAL_QUERY: str = (
    "cache_retrieval_query")

W61_MLSC_V9_KNOWN_ALGEBRA_SIGNATURES: tuple[str, ...] = (
    *W60_MLSC_V8_KNOWN_ALGEBRA_SIGNATURES,
    W61_MLSC_V9_ALGEBRA_ATTENTION_PATTERN_STEER,
    W61_MLSC_V9_ALGEBRA_CACHE_RETRIEVAL_QUERY,
)


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass(frozen=True)
class MergeableLatentCapsuleV9:
    schema: str
    inner_v8: MergeableLatentCapsuleV8
    attention_pattern_witness_chain: tuple[str, ...]
    cache_retrieval_witness_chain: tuple[str, ...]
    per_layer_head_trust_matrix: tuple[
        tuple[int, int, float], ...]
    algebra_signature_v7: str

    @property
    def payload(self) -> tuple[float, ...]:
        return self.inner_v8.payload

    @property
    def trust(self) -> float:
        return float(self.inner_v8.trust)

    @property
    def confidence(self) -> float:
        return float(self.inner_v8.confidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "inner_v8_cid": str(self.inner_v8.cid()),
            "attention_pattern_witness_chain": list(
                self.attention_pattern_witness_chain),
            "cache_retrieval_witness_chain": list(
                self.cache_retrieval_witness_chain),
            "per_layer_head_trust_matrix": [
                [int(l), int(h), float(round(t, 12))]
                for l, h, t in sorted(
                    self.per_layer_head_trust_matrix)],
            "algebra_signature_v7": str(
                self.algebra_signature_v7),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w61_mlsc_v9",
            "capsule": self.to_dict()})


def wrap_v8_as_v9(
        v8_capsule: MergeableLatentCapsuleV8, *,
        attention_pattern_witness_chain: Sequence[str] = (),
        cache_retrieval_witness_chain: Sequence[str] = (),
        per_layer_head_trust_matrix: Sequence[
            tuple[int, int, float]] = (),
        algebra_signature_v7: str = W56_MLSC_V4_ALGEBRA_MERGE,
) -> MergeableLatentCapsuleV9:
    if (algebra_signature_v7 not in
            W61_MLSC_V9_KNOWN_ALGEBRA_SIGNATURES):
        algebra_signature_v7 = W56_MLSC_V4_ALGEBRA_MERGE
    return MergeableLatentCapsuleV9(
        schema=W61_MLSC_V9_SCHEMA_VERSION,
        inner_v8=v8_capsule,
        attention_pattern_witness_chain=tuple(
            str(s) for s in attention_pattern_witness_chain),
        cache_retrieval_witness_chain=tuple(
            str(s) for s in cache_retrieval_witness_chain),
        per_layer_head_trust_matrix=tuple(
            (int(l), int(h), float(t))
            for l, h, t in per_layer_head_trust_matrix),
        algebra_signature_v7=str(algebra_signature_v7),
    )


@dataclasses.dataclass
class MergeOperatorV9:
    """V9 merge operator: extends V8 merge with attention-pattern
    chain inheritance + cache-retrieval chain propagation + per-
    (layer, head) trust matrix merging."""
    factor_dim: int = 6

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w61_mlsc_v9_merge_operator",
            "factor_dim": int(self.factor_dim),
        })

    def merge(
            self,
            capsules: Sequence[MergeableLatentCapsuleV9],
            *,
            attention_pattern_witness_chain: Sequence[str] = (),
            cache_retrieval_witness_chain: Sequence[str] = (),
            per_layer_head_trust_matrix: Sequence[
                tuple[int, int, float]] = (),
            replay_witness_chain: Sequence[str] = (),
            substrate_witness_chain: Sequence[str] = (),
            provenance_trust_table: Mapping[str, float] = ({}),
            retrieval_witness_chain: Sequence[str] = (),
            controller_witness_cid: str = "",
            attention_witness_chain: Sequence[str] = (),
            cache_reuse_witness_cid: str = "",
            algebra_signature_v7: str = (
                W56_MLSC_V4_ALGEBRA_MERGE),
            per_head_trust: Sequence[float] | None = None,
    ) -> MergeableLatentCapsuleV9:
        if not capsules:
            raise ValueError(
                "merge requires at least 1 capsule")
        inner_v8_capsules = [c.inner_v8 for c in capsules]
        v8_op = MergeOperatorV8(
            factor_dim=int(self.factor_dim))
        v8_signature = (
            str(algebra_signature_v7)
            if algebra_signature_v7 in
            W60_MLSC_V8_KNOWN_ALGEBRA_SIGNATURES
            else W56_MLSC_V4_ALGEBRA_MERGE)
        merged_v8 = v8_op.merge(
            inner_v8_capsules,
            replay_witness_chain=replay_witness_chain,
            substrate_witness_chain=substrate_witness_chain,
            provenance_trust_table=provenance_trust_table,
            retrieval_witness_chain=retrieval_witness_chain,
            controller_witness_cid=controller_witness_cid,
            attention_witness_chain=attention_witness_chain,
            cache_reuse_witness_cid=cache_reuse_witness_cid,
            algebra_signature_v6=v8_signature,
            per_head_trust=per_head_trust,
        )
        # Union over attention_pattern_witness_chain.
        ap_set: list[str] = []
        for c in capsules:
            for a in c.attention_pattern_witness_chain:
                if a not in ap_set:
                    ap_set.append(a)
        for a in attention_pattern_witness_chain:
            if a not in ap_set:
                ap_set.append(a)
        # Union over cache_retrieval_witness_chain.
        cr_set: list[str] = []
        for c in capsules:
            for a in c.cache_retrieval_witness_chain:
                if a not in cr_set:
                    cr_set.append(a)
        for a in cache_retrieval_witness_chain:
            if a not in cr_set:
                cr_set.append(a)
        # Per-(L, H) trust matrix merge: max trust per cell.
        trust_mat: dict[tuple[int, int], float] = {}
        for c in capsules:
            for l, h, t in c.per_layer_head_trust_matrix:
                key = (int(l), int(h))
                if (key not in trust_mat
                        or float(t) > trust_mat[key]):
                    trust_mat[key] = float(t)
        for l, h, t in per_layer_head_trust_matrix:
            key = (int(l), int(h))
            if (key not in trust_mat
                    or float(t) > trust_mat[key]):
                trust_mat[key] = float(t)
        flat = tuple(
            (int(l), int(h), float(t))
            for (l, h), t in sorted(trust_mat.items()))
        return wrap_v8_as_v9(
            merged_v8,
            attention_pattern_witness_chain=tuple(ap_set),
            cache_retrieval_witness_chain=tuple(cr_set),
            per_layer_head_trust_matrix=flat,
            algebra_signature_v7=str(algebra_signature_v7),
        )


@dataclasses.dataclass(frozen=True)
class MergeableLatentCapsuleV9Witness:
    schema: str
    capsule_cid: str
    inner_v8_cid: str
    attention_pattern_witness_chain_depth: int
    cache_retrieval_witness_chain_depth: int
    per_layer_head_trust_matrix_size: int
    algebra_signature_v7: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "capsule_cid": str(self.capsule_cid),
            "inner_v8_cid": str(self.inner_v8_cid),
            "attention_pattern_witness_chain_depth": int(
                self.attention_pattern_witness_chain_depth),
            "cache_retrieval_witness_chain_depth": int(
                self.cache_retrieval_witness_chain_depth),
            "per_layer_head_trust_matrix_size": int(
                self.per_layer_head_trust_matrix_size),
            "algebra_signature_v7": str(
                self.algebra_signature_v7),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w61_mlsc_v9_witness",
            "witness": self.to_dict()})


def emit_mlsc_v9_witness(
        capsule: MergeableLatentCapsuleV9,
) -> MergeableLatentCapsuleV9Witness:
    return MergeableLatentCapsuleV9Witness(
        schema=W61_MLSC_V9_SCHEMA_VERSION,
        capsule_cid=str(capsule.cid()),
        inner_v8_cid=str(capsule.inner_v8.cid()),
        attention_pattern_witness_chain_depth=int(
            len(capsule.attention_pattern_witness_chain)),
        cache_retrieval_witness_chain_depth=int(
            len(capsule.cache_retrieval_witness_chain)),
        per_layer_head_trust_matrix_size=int(
            len(capsule.per_layer_head_trust_matrix)),
        algebra_signature_v7=str(
            capsule.algebra_signature_v7),
    )


__all__ = [
    "W61_MLSC_V9_SCHEMA_VERSION",
    "W61_MLSC_V9_ALGEBRA_ATTENTION_PATTERN_STEER",
    "W61_MLSC_V9_ALGEBRA_CACHE_RETRIEVAL_QUERY",
    "W61_MLSC_V9_KNOWN_ALGEBRA_SIGNATURES",
    "MergeableLatentCapsuleV9",
    "MergeableLatentCapsuleV9Witness",
    "MergeOperatorV9",
    "wrap_v8_as_v9",
    "emit_mlsc_v9_witness",
]
