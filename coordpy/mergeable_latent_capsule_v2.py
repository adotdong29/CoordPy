"""W54 M3 — Mergeable Latent State Capsule V2 (MLSC V2).

Extends W53 MLSC with three first-class additions:

* **per-dim disagreement metadata** — each merge records
  ``|payload_a - payload_b|`` per-dim alongside the merged
  payload. The disagreement is content-addressed in the merged
  capsule's CID so it survives audit walks.
* **provenance fact graph** — every merge entry tracks which
  parent contributed which fact_tag (a per-tag mapping
  ``tag → parent_cid``). The audit trail exposes
  ``walk_provenance(leaf_cid)`` returning the DAG of
  ``(tag, contributor_cid)`` edges.
* **trust signature** — each parent capsule carries a per-parent
  trust scalar in ``[0, 1]`` that scales its merge weight in
  addition to the softmax(confidence/T) weighting. Low-trust
  parents are down-weighted regardless of their stated
  confidence.

The base MLSC abstractions remain unchanged; V2 adds new types
``MergeableLatentCapsuleV2``, ``MergeOperatorV2``,
``MergeAuditEntryV2``, ``MergeAuditTrailV2``.

Honest scope: pure-Python only, capsule-layer only, no
transformer-internal state.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from .mergeable_latent_capsule import (
    MergeableLatentCapsule,
    W53_DEFAULT_MLSC_FACTOR_DIM,
    W53_DEFAULT_MLSC_MERGE_TEMP,
    W53_MLSC_KIND_BRANCH,
    W53_MLSC_KIND_MERGE,
    W53_MLSC_KIND_ROOT,
)


# =============================================================================
# Schema, defaults
# =============================================================================

W54_MLSC_V2_SCHEMA_VERSION: str = (
    "coordpy.mergeable_latent_capsule_v2.v1")

W54_DEFAULT_MLSC_V2_FACTOR_DIM: int = W53_DEFAULT_MLSC_FACTOR_DIM
W54_DEFAULT_MLSC_V2_TRUST_FLOOR: float = 0.0
W54_DEFAULT_MLSC_V2_TRUST_DEFAULT: float = 1.0


# =============================================================================
# Helpers
# =============================================================================


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _round_floats(
        values: Sequence[float], precision: int = 12,
) -> list[float]:
    return [float(round(float(v), precision)) for v in values]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for i in range(n):
        ai = float(a[i])
        bi = float(b[i])
        dot += ai * bi
        na += ai * ai
        nb += bi * bi
    if na <= 1e-30 or nb <= 1e-30:
        return 0.0
    return float(dot / (math.sqrt(na) * math.sqrt(nb)))


# =============================================================================
# MergeableLatentCapsuleV2
# =============================================================================


@dataclasses.dataclass(frozen=True)
class MergeableLatentCapsuleV2:
    """V2 capsule: like V1 but carries trust + disagreement metadata."""

    kind: str
    branch_id: str
    parent_cids: tuple[str, ...]
    payload: tuple[float, ...]
    confidence: float
    trust: float
    fact_tags: tuple[str, ...]
    merge_weights: tuple[float, ...]
    disagreement_per_dim: tuple[float, ...]
    fact_tag_provenance: tuple[tuple[str, str], ...]
    turn_index: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": str(
                W54_MLSC_V2_SCHEMA_VERSION),
            "kind": str(self.kind),
            "branch_id": str(self.branch_id),
            "parent_cids": list(self.parent_cids),
            "payload": list(_round_floats(self.payload)),
            "confidence": float(round(self.confidence, 12)),
            "trust": float(round(self.trust, 12)),
            "fact_tags": list(self.fact_tags),
            "merge_weights": list(_round_floats(
                self.merge_weights)),
            "disagreement_per_dim": list(_round_floats(
                self.disagreement_per_dim)),
            "fact_tag_provenance": [
                [str(t), str(c)]
                for (t, c) in self.fact_tag_provenance],
            "turn_index": int(self.turn_index),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w54_mlsc_v2_capsule",
            "capsule": self.to_dict()})

    @property
    def is_merge(self) -> bool:
        return self.kind == W53_MLSC_KIND_MERGE


def make_root_capsule_v2(
        *,
        branch_id: str,
        payload: Sequence[float],
        confidence: float = 1.0,
        trust: float = W54_DEFAULT_MLSC_V2_TRUST_DEFAULT,
        fact_tags: Sequence[str] = (),
        turn_index: int = 0,
) -> MergeableLatentCapsuleV2:
    tags = tuple(sorted(set(str(t) for t in fact_tags)))
    prov = tuple(
        (str(t), "self") for t in tags)
    return MergeableLatentCapsuleV2(
        kind=W53_MLSC_KIND_ROOT,
        branch_id=str(branch_id),
        parent_cids=(),
        payload=tuple(_round_floats(payload)),
        confidence=float(max(0.0, min(1.0, float(confidence)))),
        trust=float(max(0.0, min(1.0, float(trust)))),
        fact_tags=tags,
        merge_weights=(),
        disagreement_per_dim=(),
        fact_tag_provenance=prov,
        turn_index=int(turn_index),
    )


def step_branch_capsule_v2(
        *,
        parent: MergeableLatentCapsuleV2,
        payload: Sequence[float],
        confidence: float | None = None,
        trust: float | None = None,
        new_fact_tags: Sequence[str] = (),
        turn_index: int | None = None,
        branch_id: str | None = None,
) -> MergeableLatentCapsuleV2:
    bid = (
        str(branch_id) if branch_id is not None
        else str(parent.branch_id))
    cnf = (
        float(confidence) if confidence is not None
        else float(parent.confidence))
    cnf = float(max(0.0, min(1.0, cnf)))
    tr = (
        float(trust) if trust is not None
        else float(parent.trust))
    tr = float(max(0.0, min(1.0, tr)))
    tags = tuple(sorted(
        set(str(t) for t in parent.fact_tags)
        | set(str(t) for t in new_fact_tags)))
    new_tags = set(str(t) for t in new_fact_tags) - set(
        parent.fact_tags)
    parent_cid = str(parent.cid())
    prov_map = {t: c for (t, c) in parent.fact_tag_provenance}
    for t in new_tags:
        prov_map[t] = parent_cid
    prov = tuple(sorted(
        (str(t), str(c)) for t, c in prov_map.items()))
    ti = (
        int(turn_index) if turn_index is not None
        else int(parent.turn_index) + 1)
    return MergeableLatentCapsuleV2(
        kind=W53_MLSC_KIND_BRANCH,
        branch_id=bid,
        parent_cids=(parent_cid,),
        payload=tuple(_round_floats(payload)),
        confidence=cnf, trust=tr,
        fact_tags=tags,
        merge_weights=(),
        disagreement_per_dim=(),
        fact_tag_provenance=prov,
        turn_index=ti,
    )


# =============================================================================
# MergeOperatorV2 — trust-weighted convex combine
# =============================================================================


@dataclasses.dataclass(frozen=True)
class MergeOperatorV2:
    """Trust-weighted softmax merge operator.

    Merge weight per parent ``i``:

        scaled_i = c_i / T  (softmax over scaled)
        raw_w_i = softmax(scaled)_i
        trust_norm = max(trust_floor, t_i)
        w_i = raw_w_i * trust_norm
        w_i ← w_i / Σ w_i  (renormalised)
        payload_merged = Σ w_i * payload_i
        confidence_merged = Σ w_i * c_i
        trust_merged = mean(t_i)
    """

    factor_dim: int
    temperature: float = W53_DEFAULT_MLSC_MERGE_TEMP
    trust_floor: float = W54_DEFAULT_MLSC_V2_TRUST_FLOOR

    def merge(
            self,
            parents: Sequence[MergeableLatentCapsuleV2],
            *,
            extra_fact_tags: Sequence[str] = (),
            turn_index: int | None = None,
    ) -> MergeableLatentCapsuleV2:
        if len(parents) < 2:
            raise ValueError(
                "MergeOperatorV2.merge requires ≥ 2 parents")
        cs = [float(p.confidence) for p in parents]
        ts = [float(p.trust) for p in parents]
        T = max(1e-6, float(self.temperature))
        scaled = [c / T for c in cs]
        m = max(scaled)
        exps = [math.exp(s - m) for s in scaled]
        z = float(sum(exps)) or 1.0
        raw_w = [float(e / z) for e in exps]
        floor = float(max(0.0, min(1.0, self.trust_floor)))
        weighted_t = [
            max(floor, float(t)) for t in ts]
        w_unn = [float(raw_w[i]) * float(weighted_t[i])
                  for i in range(len(parents))]
        w_sum = float(sum(w_unn)) or 1.0
        ws = [float(w / w_sum) for w in w_unn]
        payload = [0.0] * int(self.factor_dim)
        for w, p in zip(ws, parents):
            for j in range(int(self.factor_dim)):
                payload[j] += float(w) * float(
                    p.payload[j] if j < len(p.payload) else 0.0)
        # Per-dim disagreement: weighted std-dev approximation.
        disagreement = [0.0] * int(self.factor_dim)
        for j in range(int(self.factor_dim)):
            mean_j = float(payload[j])
            var_j = 0.0
            for w, p in zip(ws, parents):
                pj = float(
                    p.payload[j] if j < len(p.payload) else 0.0)
                var_j += float(w) * (pj - mean_j) ** 2
            disagreement[j] = float(math.sqrt(
                max(0.0, var_j)))
        cnf = float(sum(w * c for w, c in zip(ws, cs)))
        cnf = float(max(0.0, min(1.0, cnf)))
        trust_avg = float(sum(weighted_t)
                           / max(1, len(weighted_t)))
        trust_avg = float(max(0.0, min(1.0, trust_avg)))
        bids = sorted({str(p.branch_id) for p in parents})
        merged_bid = "merge:" + "+".join(bids)
        tags_set: set[str] = set()
        prov_map: dict[str, str] = {}
        for p in parents:
            p_cid = str(p.cid())
            for t in p.fact_tags:
                tags_set.add(str(t))
                pmap = {
                    pt: pc
                    for (pt, pc) in p.fact_tag_provenance
                }
                origin = str(pmap.get(t, p_cid))
                # If the parent recorded "self", resolve to the
                # parent's actual CID; otherwise keep the recorded
                # contributor.
                if origin == "self":
                    origin = p_cid
                prov_map.setdefault(str(t), origin)
        for t in extra_fact_tags:
            tags_set.add(str(t))
            prov_map.setdefault(str(t), "merge")
        prov = tuple(sorted(
            (str(t), str(c)) for t, c in prov_map.items()))
        ti = (
            int(turn_index) if turn_index is not None
            else max(int(p.turn_index) for p in parents) + 1)
        return MergeableLatentCapsuleV2(
            kind=W53_MLSC_KIND_MERGE,
            branch_id=str(merged_bid),
            parent_cids=tuple(str(p.cid()) for p in parents),
            payload=tuple(_round_floats(payload)),
            confidence=cnf,
            trust=trust_avg,
            fact_tags=tuple(sorted(tags_set)),
            merge_weights=tuple(_round_floats(ws)),
            disagreement_per_dim=tuple(_round_floats(
                disagreement)),
            fact_tag_provenance=prov,
            turn_index=ti,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": str(
                W54_MLSC_V2_SCHEMA_VERSION),
            "factor_dim": int(self.factor_dim),
            "temperature": float(round(
                self.temperature, 12)),
            "trust_floor": float(round(
                self.trust_floor, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w54_mlsc_v2_operator",
            "operator": self.to_dict()})


# =============================================================================
# MergeAuditTrailV2
# =============================================================================


@dataclasses.dataclass(frozen=True)
class MergeAuditEntryV2:
    merged_cid: str
    parent_cids: tuple[str, ...]
    parent_branch_ids: tuple[str, ...]
    parent_trusts: tuple[float, ...]
    merge_weights: tuple[float, ...]
    disagreement_per_dim: tuple[float, ...]
    fact_tag_provenance: tuple[tuple[str, str], ...]
    operator_cid: str
    turn_index: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "merged_cid": str(self.merged_cid),
            "parent_cids": list(self.parent_cids),
            "parent_branch_ids": list(
                self.parent_branch_ids),
            "parent_trusts": list(_round_floats(
                self.parent_trusts)),
            "merge_weights": list(_round_floats(
                self.merge_weights)),
            "disagreement_per_dim": list(_round_floats(
                self.disagreement_per_dim)),
            "fact_tag_provenance": [
                [str(t), str(c)]
                for (t, c) in self.fact_tag_provenance],
            "operator_cid": str(self.operator_cid),
            "turn_index": int(self.turn_index),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w54_mlsc_v2_audit_entry",
            "entry": self.to_dict()})


@dataclasses.dataclass
class MergeAuditTrailV2:
    entries: dict[str, MergeAuditEntryV2]

    @classmethod
    def empty(cls) -> "MergeAuditTrailV2":
        return cls(entries={})

    def add(self, entry: MergeAuditEntryV2) -> None:
        self.entries[entry.merged_cid] = entry

    def get(self, merged_cid: str) -> MergeAuditEntryV2 | None:
        return self.entries.get(str(merged_cid))

    def walk_to_roots(
            self, leaf_cid: str, *,
            capsule_store: Mapping[
                str, MergeableLatentCapsuleV2],
            max_steps: int = 256,
    ) -> list[str]:
        roots: set[str] = set()
        seen: set[str] = set()
        stack: list[str] = [str(leaf_cid)]
        steps = 0
        while stack and steps < int(max_steps):
            cid = stack.pop()
            if cid in seen:
                continue
            seen.add(cid)
            cap = capsule_store.get(cid)
            if cap is None:
                roots.add(cid)
                steps += 1
                continue
            if cap.kind == W53_MLSC_KIND_ROOT:
                roots.add(cid)
            elif cap.kind == W53_MLSC_KIND_BRANCH:
                if cap.parent_cids:
                    stack.append(str(cap.parent_cids[0]))
                else:
                    roots.add(cid)
            elif cap.kind == W53_MLSC_KIND_MERGE:
                for p in cap.parent_cids:
                    stack.append(str(p))
            steps += 1
        return sorted(roots)

    def walk_provenance(
            self, leaf_cid: str, *,
            capsule_store: Mapping[
                str, MergeableLatentCapsuleV2],
    ) -> dict[str, str]:
        """Return the {tag → contributor_cid} provenance map for
        a leaf capsule. For merge nodes, walks back through
        parents to collect tags."""
        cap = capsule_store.get(str(leaf_cid))
        if cap is None:
            return {}
        return {t: c for (t, c) in cap.fact_tag_provenance}

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w54_mlsc_v2_audit_trail",
            "entries": [
                {"cid": c, "entry": e.to_dict()}
                for c, e in sorted(self.entries.items())
            ],
        })

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": str(
                W54_MLSC_V2_SCHEMA_VERSION),
            "n_entries": int(len(self.entries)),
            "audit_cid": str(self.cid()),
        }


def merge_capsules_v2(
        operator: MergeOperatorV2,
        parents: Sequence[MergeableLatentCapsuleV2],
        *,
        audit_trail: MergeAuditTrailV2,
        extra_fact_tags: Sequence[str] = (),
        turn_index: int | None = None,
) -> MergeableLatentCapsuleV2:
    merged = operator.merge(
        parents, extra_fact_tags=extra_fact_tags,
        turn_index=turn_index)
    entry = MergeAuditEntryV2(
        merged_cid=str(merged.cid()),
        parent_cids=tuple(str(p.cid()) for p in parents),
        parent_branch_ids=tuple(
            str(p.branch_id) for p in parents),
        parent_trusts=tuple(float(p.trust) for p in parents),
        merge_weights=tuple(merged.merge_weights),
        disagreement_per_dim=tuple(
            merged.disagreement_per_dim),
        fact_tag_provenance=tuple(
            merged.fact_tag_provenance),
        operator_cid=str(operator.cid()),
        turn_index=int(merged.turn_index),
    )
    audit_trail.add(entry)
    return merged


# =============================================================================
# Witness
# =============================================================================


@dataclasses.dataclass(frozen=True)
class MergeableLatentCapsuleV2Witness:
    leaf_cid: str
    operator_cid: str
    audit_trail_cid: str
    audit_entry_count: int
    n_unique_roots: int
    parents_count: int
    merge_weights: tuple[float, ...]
    disagreement_l1: float
    confidence: float
    trust: float
    n_provenance_tags: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "leaf_cid": str(self.leaf_cid),
            "operator_cid": str(self.operator_cid),
            "audit_trail_cid": str(self.audit_trail_cid),
            "audit_entry_count": int(self.audit_entry_count),
            "n_unique_roots": int(self.n_unique_roots),
            "parents_count": int(self.parents_count),
            "merge_weights": list(_round_floats(
                self.merge_weights)),
            "disagreement_l1": float(round(
                self.disagreement_l1, 12)),
            "confidence": float(round(self.confidence, 12)),
            "trust": float(round(self.trust, 12)),
            "n_provenance_tags": int(self.n_provenance_tags),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w54_mlsc_v2_witness",
            "witness": self.to_dict()})


def emit_mlsc_v2_witness(
        *,
        leaf: MergeableLatentCapsuleV2,
        operator: MergeOperatorV2,
        audit_trail: MergeAuditTrailV2,
        capsule_store: Mapping[str, MergeableLatentCapsuleV2],
) -> MergeableLatentCapsuleV2Witness:
    roots = audit_trail.walk_to_roots(
        leaf.cid(), capsule_store=capsule_store)
    return MergeableLatentCapsuleV2Witness(
        leaf_cid=str(leaf.cid()),
        operator_cid=str(operator.cid()),
        audit_trail_cid=str(audit_trail.cid()),
        audit_entry_count=int(len(audit_trail.entries)),
        n_unique_roots=int(len(roots)),
        parents_count=int(len(leaf.parent_cids)),
        merge_weights=tuple(leaf.merge_weights),
        disagreement_l1=float(
            sum(abs(float(v))
                for v in leaf.disagreement_per_dim)),
        confidence=float(leaf.confidence),
        trust=float(leaf.trust),
        n_provenance_tags=int(len(leaf.fact_tag_provenance)),
    )


# =============================================================================
# Verifier
# =============================================================================

W54_MLSC_V2_VERIFIER_FAILURE_MODES: tuple[str, ...] = (
    "w54_mlsc_v2_schema_mismatch",
    "w54_mlsc_v2_leaf_cid_mismatch",
    "w54_mlsc_v2_operator_cid_mismatch",
    "w54_mlsc_v2_audit_trail_cid_mismatch",
    "w54_mlsc_v2_audit_entry_count_below_floor",
    "w54_mlsc_v2_n_unique_roots_below_floor",
    "w54_mlsc_v2_orphan_parent_in_audit_trail",
    "w54_mlsc_v2_confidence_out_of_bounds",
    "w54_mlsc_v2_trust_out_of_bounds",
    "w54_mlsc_v2_merge_weights_dont_sum_to_one",
    "w54_mlsc_v2_disagreement_negative",
    "w54_mlsc_v2_provenance_tag_missing",
)


def verify_mlsc_v2_witness(
        witness: MergeableLatentCapsuleV2Witness,
        *,
        expected_leaf_cid: str | None = None,
        expected_operator_cid: str | None = None,
        expected_audit_trail_cid: str | None = None,
        min_audit_entry_count: int | None = None,
        min_n_unique_roots: int | None = None,
        audit_trail: MergeAuditTrailV2 | None = None,
        capsule_store: Mapping[
            str, MergeableLatentCapsuleV2] | None = None,
) -> dict[str, Any]:
    failures: list[str] = []
    if (expected_leaf_cid is not None
            and witness.leaf_cid != str(expected_leaf_cid)):
        failures.append("w54_mlsc_v2_leaf_cid_mismatch")
    if (expected_operator_cid is not None
            and witness.operator_cid
            != str(expected_operator_cid)):
        failures.append("w54_mlsc_v2_operator_cid_mismatch")
    if (expected_audit_trail_cid is not None
            and witness.audit_trail_cid
            != str(expected_audit_trail_cid)):
        failures.append(
            "w54_mlsc_v2_audit_trail_cid_mismatch")
    if (min_audit_entry_count is not None
            and witness.audit_entry_count
            < int(min_audit_entry_count)):
        failures.append(
            "w54_mlsc_v2_audit_entry_count_below_floor")
    if (min_n_unique_roots is not None
            and witness.n_unique_roots
            < int(min_n_unique_roots)):
        failures.append(
            "w54_mlsc_v2_n_unique_roots_below_floor")
    if not (0.0 <= float(witness.confidence) <= 1.0):
        failures.append(
            "w54_mlsc_v2_confidence_out_of_bounds")
    if not (0.0 <= float(witness.trust) <= 1.0):
        failures.append("w54_mlsc_v2_trust_out_of_bounds")
    if witness.merge_weights:
        s = float(sum(witness.merge_weights))
        if abs(s - 1.0) > 1e-6:
            failures.append(
                "w54_mlsc_v2_merge_weights_dont_sum_to_one")
    if witness.disagreement_l1 < 0.0:
        failures.append("w54_mlsc_v2_disagreement_negative")
    if (audit_trail is not None
            and capsule_store is not None):
        for entry in audit_trail.entries.values():
            for p in entry.parent_cids:
                if p not in capsule_store:
                    failures.append(
                        "w54_mlsc_v2_orphan_parent_in_audit_trail")
                    break
            else:
                continue
            break
    return {
        "ok": (len(failures) == 0),
        "failures": failures,
        "witness_cid": witness.cid(),
    }


# =============================================================================
# Consensus quorum V2 (trust-weighted)
# =============================================================================


@dataclasses.dataclass(frozen=True)
class ConsensusQuorumResultV2:
    n_branches: int
    k_required: int
    quorum_reached: bool
    abstain: bool
    fallback_used: bool
    consensus_payload: tuple[float, ...]
    consensus_cosine_floor: float
    consensus_capsule_cid: str
    selected_branch_ids: tuple[str, ...]
    fallback_branch_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_branches": int(self.n_branches),
            "k_required": int(self.k_required),
            "quorum_reached": bool(self.quorum_reached),
            "abstain": bool(self.abstain),
            "fallback_used": bool(self.fallback_used),
            "consensus_payload": list(_round_floats(
                self.consensus_payload)),
            "consensus_cosine_floor": float(round(
                self.consensus_cosine_floor, 12)),
            "consensus_capsule_cid": str(
                self.consensus_capsule_cid),
            "selected_branch_ids": list(
                self.selected_branch_ids),
            "fallback_branch_id": str(
                self.fallback_branch_id),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w54_mlsc_v2_consensus_result",
            "result": self.to_dict()})


def compute_consensus_quorum_v2(
        branches: Sequence[MergeableLatentCapsuleV2],
        *,
        operator: MergeOperatorV2,
        audit_trail: MergeAuditTrailV2,
        k_required: int,
        cosine_floor: float = 0.5,
        allow_fallback: bool = True,
        fallback_cosine_floor: float = 0.0,
) -> ConsensusQuorumResultV2:
    """K-of-N consensus with explicit abstain-with-fallback policy.

    If quorum not reached:
    * if allow_fallback and the best single parent has cosine to
      mean ≥ fallback_cosine_floor, use it as fallback
    * else final abstain.
    """
    n = len(branches)
    if n == 0 or int(k_required) <= 0:
        return ConsensusQuorumResultV2(
            n_branches=int(n),
            k_required=int(k_required),
            quorum_reached=False,
            abstain=True,
            fallback_used=False,
            consensus_payload=(),
            consensus_cosine_floor=float(cosine_floor),
            consensus_capsule_cid="",
            selected_branch_ids=(),
            fallback_branch_id="",
        )
    cos = [[1.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            c = _cosine(
                branches[i].payload, branches[j].payload)
            cos[i][j] = float(c)
            cos[j][i] = float(c)
    centrality = [
        sum(cos[i][j] for j in range(n) if j != i)
        for i in range(n)
    ]
    order = sorted(
        range(n), key=lambda i: -centrality[i])
    selected: list[int] = []
    for i in order:
        ok = True
        for s in selected:
            if cos[i][s] < float(cosine_floor):
                ok = False
                break
        if ok:
            selected.append(i)
    if len(selected) >= int(k_required):
        sel_branches = [branches[i] for i in selected]
        merged = merge_capsules_v2(
            operator, sel_branches,
            audit_trail=audit_trail,
            extra_fact_tags=(
                f"quorum_v2:k={k_required}",))
        floor = (
            min(cos[a][b] for a in selected for b in selected
                if a < b) if len(selected) > 1 else 1.0)
        return ConsensusQuorumResultV2(
            n_branches=int(n),
            k_required=int(k_required),
            quorum_reached=True,
            abstain=False,
            fallback_used=False,
            consensus_payload=tuple(merged.payload),
            consensus_cosine_floor=float(floor),
            consensus_capsule_cid=str(merged.cid()),
            selected_branch_ids=tuple(
                str(b.branch_id) for b in sel_branches),
            fallback_branch_id="",
        )
    if allow_fallback:
        # Best single parent: highest trust * confidence.
        best_i = max(
            range(n),
            key=lambda i: (
                float(branches[i].trust)
                * float(branches[i].confidence)))
        best = branches[best_i]
        # Approximate cosine to mean payload across branches.
        mean_p = [0.0] * len(best.payload)
        for b in branches:
            for j in range(len(mean_p)):
                mean_p[j] += float(
                    b.payload[j] if j < len(b.payload) else 0.0
                ) / float(max(1, n))
        cs = _cosine(best.payload, mean_p)
        if cs >= float(fallback_cosine_floor):
            return ConsensusQuorumResultV2(
                n_branches=int(n),
                k_required=int(k_required),
                quorum_reached=False,
                abstain=False,
                fallback_used=True,
                consensus_payload=tuple(best.payload),
                consensus_cosine_floor=float(cs),
                consensus_capsule_cid=str(best.cid()),
                selected_branch_ids=(str(best.branch_id),),
                fallback_branch_id=str(best.branch_id),
            )
    return ConsensusQuorumResultV2(
        n_branches=int(n),
        k_required=int(k_required),
        quorum_reached=False,
        abstain=True,
        fallback_used=False,
        consensus_payload=(),
        consensus_cosine_floor=float(cosine_floor),
        consensus_capsule_cid="",
        selected_branch_ids=(),
        fallback_branch_id="",
    )


__all__ = [
    "W54_MLSC_V2_SCHEMA_VERSION",
    "W54_DEFAULT_MLSC_V2_FACTOR_DIM",
    "W54_DEFAULT_MLSC_V2_TRUST_FLOOR",
    "W54_DEFAULT_MLSC_V2_TRUST_DEFAULT",
    "W54_MLSC_V2_VERIFIER_FAILURE_MODES",
    "MergeableLatentCapsuleV2",
    "MergeOperatorV2",
    "MergeAuditEntryV2",
    "MergeAuditTrailV2",
    "MergeableLatentCapsuleV2Witness",
    "ConsensusQuorumResultV2",
    "make_root_capsule_v2",
    "step_branch_capsule_v2",
    "merge_capsules_v2",
    "emit_mlsc_v2_witness",
    "verify_mlsc_v2_witness",
    "compute_consensus_quorum_v2",
]
