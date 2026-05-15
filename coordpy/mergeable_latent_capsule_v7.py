"""W59 M9 — Mergeable Latent State Capsule V7 (MLSC V7).

Strictly extends W58's ``coordpy.mergeable_latent_capsule_v6``.
V7 adds:

* **``retrieval_witness_chain``** — V6 carried only an
  ``attention_witness_chain``. V7 carries an additional chain of
  *retrieval witnesses* — CIDs of the cache-controller-V2
  retrieval scores that contributed to the payload. Union-
  inherited from parents.
* **``controller_witness_cid``** — content-addressed identifier
  of the W59 cache controller V2 used at emit time. Lets the
  consensus controller V5 verify that two parents agree on the
  *controller state*, not just the payload.
* **``algebra_signature_v5``** — adds two new primitives:

    * ``retrieval_replay`` — payload was reproduced by replaying
      the V2 controller's retrieval score on the same cache.
    * ``partial_prefix_reuse`` — payload was produced under the
      W59 prefix-state-bridge V3 partial-reuse path.

V7 strictly extends V6 byte-for-byte when:

  * ``retrieval_witness_chain == ()``
  * ``controller_witness_cid == ""``
  * ``algebra_signature_v5`` ∈ V6's known signatures.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Sequence

from .mergeable_latent_capsule_v6 import (
    MergeOperatorV6,
    MergeableLatentCapsuleV6,
    W58_MLSC_V6_KNOWN_ALGEBRA_SIGNATURES,
    W58_MLSC_V6_ALGEBRA_ATTENTION_STEER,
    W58_MLSC_V6_ALGEBRA_CACHE_REUSE_REPLAY,
)
from .mergeable_latent_capsule_v5 import (
    MergeableLatentCapsuleV5,
    W57_MLSC_V5_ALGEBRA_HIDDEN_INJECT,
    W57_MLSC_V5_ALGEBRA_SUBSTRATE_PROJECT,
)
from .mergeable_latent_capsule_v4 import (
    W56_MLSC_V4_ALGEBRA_DIFF,
    W56_MLSC_V4_ALGEBRA_INTERSECT,
    W56_MLSC_V4_ALGEBRA_MERGE,
)


W59_MLSC_V7_SCHEMA_VERSION: str = (
    "coordpy.mergeable_latent_capsule_v7.v1")
W59_MLSC_V7_ALGEBRA_RETRIEVAL_REPLAY: str = "retrieval_replay"
W59_MLSC_V7_ALGEBRA_PARTIAL_PREFIX_REUSE: str = (
    "partial_prefix_reuse")

W59_MLSC_V7_KNOWN_ALGEBRA_SIGNATURES: tuple[str, ...] = (
    *W58_MLSC_V6_KNOWN_ALGEBRA_SIGNATURES,
    W59_MLSC_V7_ALGEBRA_RETRIEVAL_REPLAY,
    W59_MLSC_V7_ALGEBRA_PARTIAL_PREFIX_REUSE,
)


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass(frozen=True)
class MergeableLatentCapsuleV7:
    schema: str
    inner_v6: MergeableLatentCapsuleV6
    retrieval_witness_chain: tuple[str, ...]
    controller_witness_cid: str
    algebra_signature_v5: str

    @property
    def payload(self) -> tuple[float, ...]:
        return self.inner_v6.payload

    @property
    def trust(self) -> float:
        return float(self.inner_v6.trust)

    @property
    def confidence(self) -> float:
        return float(self.inner_v6.confidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "inner_v6_cid": str(self.inner_v6.cid()),
            "retrieval_witness_chain": list(
                self.retrieval_witness_chain),
            "controller_witness_cid": str(
                self.controller_witness_cid),
            "algebra_signature_v5": str(
                self.algebra_signature_v5),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w59_mlsc_v7",
            "capsule": self.to_dict()})


def wrap_v6_as_v7(
        v6_capsule: MergeableLatentCapsuleV6, *,
        retrieval_witness_chain: Sequence[str] = (),
        controller_witness_cid: str = "",
        algebra_signature_v5: str = W56_MLSC_V4_ALGEBRA_MERGE,
) -> MergeableLatentCapsuleV7:
    if (algebra_signature_v5 not in
            W59_MLSC_V7_KNOWN_ALGEBRA_SIGNATURES):
        algebra_signature_v5 = W56_MLSC_V4_ALGEBRA_MERGE
    return MergeableLatentCapsuleV7(
        schema=W59_MLSC_V7_SCHEMA_VERSION,
        inner_v6=v6_capsule,
        retrieval_witness_chain=tuple(
            str(s) for s in retrieval_witness_chain),
        controller_witness_cid=str(controller_witness_cid),
        algebra_signature_v5=str(algebra_signature_v5),
    )


@dataclasses.dataclass
class MergeOperatorV7:
    """V7 merge operator: extends V6 merge with retrieval chain
    inheritance + controller-witness propagation."""

    factor_dim: int = 6

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w59_mlsc_v7_merge_operator",
            "factor_dim": int(self.factor_dim),
        })

    def merge(
            self, capsules: Sequence[MergeableLatentCapsuleV7], *,
            retrieval_witness_chain: Sequence[str] = (),
            controller_witness_cid: str = "",
            attention_witness_chain: Sequence[str] = (),
            cache_reuse_witness_cid: str = "",
            algebra_signature_v5: str = (
                W56_MLSC_V4_ALGEBRA_MERGE),
            per_head_trust: Sequence[float] | None = None,
    ) -> MergeableLatentCapsuleV7:
        if not capsules:
            raise ValueError(
                "merge requires at least 1 capsule")
        inner_v6_capsules = [c.inner_v6 for c in capsules]
        v6_op = MergeOperatorV6(factor_dim=int(self.factor_dim))
        # V6 merge: pass algebra signature only if it is a V6-known
        # signature; otherwise default to plain merge so the v6
        # merge does not reject it.
        v6_signature = (
            str(algebra_signature_v5)
            if algebra_signature_v5
            in W58_MLSC_V6_KNOWN_ALGEBRA_SIGNATURES
            else W56_MLSC_V4_ALGEBRA_MERGE)
        merged_v6 = v6_op.merge(
            inner_v6_capsules,
            attention_witness_chain=attention_witness_chain,
            cache_reuse_witness_cid=cache_reuse_witness_cid,
            algebra_signature_v4=v6_signature,
            per_head_trust=per_head_trust,
        )
        # Retrieval witness chain inherits union from parents.
        ret_chain_set: list[str] = []
        for c in capsules:
            for a in c.retrieval_witness_chain:
                if a not in ret_chain_set:
                    ret_chain_set.append(a)
        for a in retrieval_witness_chain:
            if a not in ret_chain_set:
                ret_chain_set.append(a)
        return wrap_v6_as_v7(
            merged_v6,
            retrieval_witness_chain=tuple(ret_chain_set),
            controller_witness_cid=str(controller_witness_cid),
            algebra_signature_v5=str(algebra_signature_v5),
        )


@dataclasses.dataclass(frozen=True)
class MergeableLatentCapsuleV7Witness:
    schema: str
    capsule_cid: str
    inner_v6_cid: str
    retrieval_witness_chain_depth: int
    controller_witness_cid: str
    algebra_signature_v5: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "capsule_cid": str(self.capsule_cid),
            "inner_v6_cid": str(self.inner_v6_cid),
            "retrieval_witness_chain_depth": int(
                self.retrieval_witness_chain_depth),
            "controller_witness_cid": str(
                self.controller_witness_cid),
            "algebra_signature_v5": str(
                self.algebra_signature_v5),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w59_mlsc_v7_witness",
            "witness": self.to_dict()})


def emit_mlsc_v7_witness(
        capsule: MergeableLatentCapsuleV7,
) -> MergeableLatentCapsuleV7Witness:
    return MergeableLatentCapsuleV7Witness(
        schema=W59_MLSC_V7_SCHEMA_VERSION,
        capsule_cid=str(capsule.cid()),
        inner_v6_cid=str(capsule.inner_v6.cid()),
        retrieval_witness_chain_depth=int(
            len(capsule.retrieval_witness_chain)),
        controller_witness_cid=str(
            capsule.controller_witness_cid),
        algebra_signature_v5=str(capsule.algebra_signature_v5),
    )


__all__ = [
    "W59_MLSC_V7_SCHEMA_VERSION",
    "W59_MLSC_V7_ALGEBRA_RETRIEVAL_REPLAY",
    "W59_MLSC_V7_ALGEBRA_PARTIAL_PREFIX_REUSE",
    "W59_MLSC_V7_KNOWN_ALGEBRA_SIGNATURES",
    "MergeableLatentCapsuleV7",
    "MergeableLatentCapsuleV7Witness",
    "MergeOperatorV7",
    "wrap_v6_as_v7",
    "emit_mlsc_v7_witness",
]
