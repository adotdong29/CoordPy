"""W60 M10 — Mergeable Latent State Capsule V8 (MLSC V8).

Strictly extends W59's ``coordpy.mergeable_latent_capsule_v7``. V8
adds:

* **``replay_witness_chain``** — V7 carried an
  ``attention_witness_chain`` and a ``retrieval_witness_chain``.
  V8 carries an additional chain of *replay-controller decision
  witnesses* — CIDs of the W60 ReplayController decisions that
  produced the payload. Union-inherited from parents.
* **``substrate_witness_chain``** — V8 also exposes a chain of
  substrate-forward witness CIDs, one per backend whose actual
  substrate state was injected. Union-inherited from parents.
* **``provenance_trust_table``** — V8 carries a per-backend trust
  scalar table (``backend_id -> trust``) that the merge operator
  uses for trust-weighted aggregation alongside the V7 per-head
  trust.
* **``algebra_signature_v6``** — adds two new primitives:

    * ``replay_choice`` — payload was produced by a specific
      ReplayController decision (reuse / recompute / fallback /
      abstain).
    * ``substrate_state_inject`` — payload was produced by an
      explicit substrate-state injection on the V5 KV cache.

V8 strictly extends V7 byte-for-byte when:

  * ``replay_witness_chain == ()``
  * ``substrate_witness_chain == ()``
  * ``provenance_trust_table == {}``
  * ``algebra_signature_v6`` ∈ V7's known signatures.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Mapping, Sequence

from .mergeable_latent_capsule_v4 import (
    W56_MLSC_V4_ALGEBRA_MERGE,
)
from .mergeable_latent_capsule_v7 import (
    MergeableLatentCapsuleV7,
    MergeOperatorV7,
    W59_MLSC_V7_KNOWN_ALGEBRA_SIGNATURES,
    wrap_v6_as_v7,
)


W60_MLSC_V8_SCHEMA_VERSION: str = (
    "coordpy.mergeable_latent_capsule_v8.v1")
W60_MLSC_V8_ALGEBRA_REPLAY_CHOICE: str = "replay_choice"
W60_MLSC_V8_ALGEBRA_SUBSTRATE_STATE_INJECT: str = (
    "substrate_state_inject")

W60_MLSC_V8_KNOWN_ALGEBRA_SIGNATURES: tuple[str, ...] = (
    *W59_MLSC_V7_KNOWN_ALGEBRA_SIGNATURES,
    W60_MLSC_V8_ALGEBRA_REPLAY_CHOICE,
    W60_MLSC_V8_ALGEBRA_SUBSTRATE_STATE_INJECT,
)


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass(frozen=True)
class MergeableLatentCapsuleV8:
    schema: str
    inner_v7: MergeableLatentCapsuleV7
    replay_witness_chain: tuple[str, ...]
    substrate_witness_chain: tuple[str, ...]
    provenance_trust_table: tuple[tuple[str, float], ...]
    algebra_signature_v6: str

    @property
    def payload(self) -> tuple[float, ...]:
        return self.inner_v7.payload

    @property
    def trust(self) -> float:
        return float(self.inner_v7.trust)

    @property
    def confidence(self) -> float:
        return float(self.inner_v7.confidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "inner_v7_cid": str(self.inner_v7.cid()),
            "replay_witness_chain": list(
                self.replay_witness_chain),
            "substrate_witness_chain": list(
                self.substrate_witness_chain),
            "provenance_trust_table": [
                [str(k), float(round(v, 12))]
                for k, v in sorted(
                    self.provenance_trust_table)],
            "algebra_signature_v6": str(
                self.algebra_signature_v6),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w60_mlsc_v8",
            "capsule": self.to_dict()})


def wrap_v7_as_v8(
        v7_capsule: MergeableLatentCapsuleV7, *,
        replay_witness_chain: Sequence[str] = (),
        substrate_witness_chain: Sequence[str] = (),
        provenance_trust_table: Mapping[str, float] = (
            {}),
        algebra_signature_v6: str = (
            W56_MLSC_V4_ALGEBRA_MERGE),
) -> MergeableLatentCapsuleV8:
    if (algebra_signature_v6 not in
            W60_MLSC_V8_KNOWN_ALGEBRA_SIGNATURES):
        algebra_signature_v6 = W56_MLSC_V4_ALGEBRA_MERGE
    return MergeableLatentCapsuleV8(
        schema=W60_MLSC_V8_SCHEMA_VERSION,
        inner_v7=v7_capsule,
        replay_witness_chain=tuple(
            str(s) for s in replay_witness_chain),
        substrate_witness_chain=tuple(
            str(s) for s in substrate_witness_chain),
        provenance_trust_table=tuple(
            (str(k), float(v))
            for k, v in dict(provenance_trust_table).items()),
        algebra_signature_v6=str(algebra_signature_v6),
    )


@dataclasses.dataclass
class MergeOperatorV8:
    """V8 merge operator: extends V7 merge with replay chain
    inheritance + substrate witness propagation + provenance trust
    table merging."""

    factor_dim: int = 6

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w60_mlsc_v8_merge_operator",
            "factor_dim": int(self.factor_dim),
        })

    def merge(
            self, capsules: Sequence[MergeableLatentCapsuleV8],
            *,
            replay_witness_chain: Sequence[str] = (),
            substrate_witness_chain: Sequence[str] = (),
            provenance_trust_table: Mapping[str, float] = (
                {}),
            retrieval_witness_chain: Sequence[str] = (),
            controller_witness_cid: str = "",
            attention_witness_chain: Sequence[str] = (),
            cache_reuse_witness_cid: str = "",
            algebra_signature_v6: str = (
                W56_MLSC_V4_ALGEBRA_MERGE),
            per_head_trust: Sequence[float] | None = None,
    ) -> MergeableLatentCapsuleV8:
        if not capsules:
            raise ValueError(
                "merge requires at least 1 capsule")
        inner_v7_capsules = [c.inner_v7 for c in capsules]
        v7_op = MergeOperatorV7(factor_dim=int(self.factor_dim))
        v7_signature = (
            str(algebra_signature_v6)
            if algebra_signature_v6
            in W59_MLSC_V7_KNOWN_ALGEBRA_SIGNATURES
            else W56_MLSC_V4_ALGEBRA_MERGE)
        merged_v7 = v7_op.merge(
            inner_v7_capsules,
            retrieval_witness_chain=retrieval_witness_chain,
            controller_witness_cid=controller_witness_cid,
            attention_witness_chain=attention_witness_chain,
            cache_reuse_witness_cid=cache_reuse_witness_cid,
            algebra_signature_v5=v7_signature,
            per_head_trust=per_head_trust,
        )
        replay_set: list[str] = []
        for c in capsules:
            for a in c.replay_witness_chain:
                if a not in replay_set:
                    replay_set.append(a)
        for a in replay_witness_chain:
            if a not in replay_set:
                replay_set.append(a)
        sub_set: list[str] = []
        for c in capsules:
            for a in c.substrate_witness_chain:
                if a not in sub_set:
                    sub_set.append(a)
        for a in substrate_witness_chain:
            if a not in sub_set:
                sub_set.append(a)
        # Provenance trust merge: max trust per backend (more
        # trust = harder to forge).
        trust_table: dict[str, float] = {}
        for c in capsules:
            for k, v in c.provenance_trust_table:
                if k not in trust_table or v > trust_table[k]:
                    trust_table[k] = float(v)
        for k, v in dict(provenance_trust_table).items():
            if k not in trust_table or v > trust_table[k]:
                trust_table[k] = float(v)
        return wrap_v7_as_v8(
            merged_v7,
            replay_witness_chain=tuple(replay_set),
            substrate_witness_chain=tuple(sub_set),
            provenance_trust_table=trust_table,
            algebra_signature_v6=str(algebra_signature_v6),
        )


@dataclasses.dataclass(frozen=True)
class MergeableLatentCapsuleV8Witness:
    schema: str
    capsule_cid: str
    inner_v7_cid: str
    replay_witness_chain_depth: int
    substrate_witness_chain_depth: int
    provenance_trust_table_size: int
    algebra_signature_v6: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "capsule_cid": str(self.capsule_cid),
            "inner_v7_cid": str(self.inner_v7_cid),
            "replay_witness_chain_depth": int(
                self.replay_witness_chain_depth),
            "substrate_witness_chain_depth": int(
                self.substrate_witness_chain_depth),
            "provenance_trust_table_size": int(
                self.provenance_trust_table_size),
            "algebra_signature_v6": str(
                self.algebra_signature_v6),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w60_mlsc_v8_witness",
            "witness": self.to_dict()})


def emit_mlsc_v8_witness(
        capsule: MergeableLatentCapsuleV8,
) -> MergeableLatentCapsuleV8Witness:
    return MergeableLatentCapsuleV8Witness(
        schema=W60_MLSC_V8_SCHEMA_VERSION,
        capsule_cid=str(capsule.cid()),
        inner_v7_cid=str(capsule.inner_v7.cid()),
        replay_witness_chain_depth=int(
            len(capsule.replay_witness_chain)),
        substrate_witness_chain_depth=int(
            len(capsule.substrate_witness_chain)),
        provenance_trust_table_size=int(
            len(capsule.provenance_trust_table)),
        algebra_signature_v6=str(capsule.algebra_signature_v6),
    )


__all__ = [
    "W60_MLSC_V8_SCHEMA_VERSION",
    "W60_MLSC_V8_ALGEBRA_REPLAY_CHOICE",
    "W60_MLSC_V8_ALGEBRA_SUBSTRATE_STATE_INJECT",
    "W60_MLSC_V8_KNOWN_ALGEBRA_SIGNATURES",
    "MergeableLatentCapsuleV8",
    "MergeableLatentCapsuleV8Witness",
    "MergeOperatorV8",
    "wrap_v7_as_v8",
    "emit_mlsc_v8_witness",
]
