"""W53 M3 — Mergeable Latent State Capsule (MLSC).

The load-bearing new abstraction for W53: a *content-addressed,
mergeable latent capsule* with an explicit ``merge`` operator
that produces a child capsule whose CID binds both parents' CIDs
plus the merge weights. Every merge is recorded in an audit
trail keyed by merged-CID, so a leaf capsule can be re-walked
back to its original (single-parent or root) ancestors.

This is the W53 answer to the post-W52 question
"how do we reconcile divergent shared latent states across
branches without losing auditability?"

Honest scope:
- pure-Python only, capsule-layer only
- no transformer-internal state
- merge is a deterministic, content-addressed function of
  (a, b, weights, fact_tags) — replay-deterministic
- the merge OPERATOR is a learned / fixed weighted blend; the
  capsule abstraction itself is content-addressed regardless

W53-L-MLSC-CAPSULE-IS-NOT-SUBSTRATE: this module does not
modify transformer hidden state, KV cache, or attention.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any, Mapping, Sequence


# =============================================================================
# Schema, defaults
# =============================================================================

W53_MLSC_SCHEMA_VERSION: str = (
    "coordpy.mergeable_latent_capsule.v1")

W53_MLSC_NO_PARENT: str = "no_mlsc_parent"
W53_MLSC_KIND_ROOT: str = "root"
W53_MLSC_KIND_BRANCH: str = "branch"
W53_MLSC_KIND_MERGE: str = "merge"

W53_DEFAULT_MLSC_FACTOR_DIM: int = 8
W53_DEFAULT_MLSC_MERGE_TEMP: float = 1.0


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


def _stable_sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


# =============================================================================
# MergeableLatentCapsule
# =============================================================================


@dataclasses.dataclass(frozen=True)
class MergeableLatentCapsule:
    """A content-addressed latent capsule that supports merge.

    Fields:
        kind: ``"root" | "branch" | "merge"``
        branch_id: stable string id for the branch this capsule
            belongs to (a merge node owns a synthesized branch
            id of the form ``"merge:{a_branch}+{b_branch}"``)
        parent_cids: 0 cids for root, 1 for branch, 2+ for merge
        payload: factor-dim float vector
        confidence: scalar in [0, 1]
        fact_tags: ordered tuple of tags accumulated from
            parents (deduplicated, sorted)
        merge_weights: weights used for the merge (empty unless
            kind == merge); summed to 1.0
        turn_index: integer turn index for ordering
    """

    kind: str
    branch_id: str
    parent_cids: tuple[str, ...]
    payload: tuple[float, ...]
    confidence: float
    fact_tags: tuple[str, ...]
    merge_weights: tuple[float, ...]
    turn_index: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": str(W53_MLSC_SCHEMA_VERSION),
            "kind": str(self.kind),
            "branch_id": str(self.branch_id),
            "parent_cids": list(self.parent_cids),
            "payload": list(_round_floats(self.payload)),
            "confidence": float(round(self.confidence, 12)),
            "fact_tags": list(self.fact_tags),
            "merge_weights": list(
                _round_floats(self.merge_weights)),
            "turn_index": int(self.turn_index),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w53_mlsc_capsule",
            "capsule": self.to_dict()})

    @property
    def is_merge(self) -> bool:
        return self.kind == W53_MLSC_KIND_MERGE


def make_root_capsule(
        *,
        branch_id: str,
        payload: Sequence[float],
        confidence: float = 1.0,
        fact_tags: Sequence[str] = (),
        turn_index: int = 0,
) -> MergeableLatentCapsule:
    return MergeableLatentCapsule(
        kind=W53_MLSC_KIND_ROOT,
        branch_id=str(branch_id),
        parent_cids=(),
        payload=tuple(_round_floats(payload)),
        confidence=float(max(0.0, min(1.0, float(confidence)))),
        fact_tags=tuple(sorted(set(str(t) for t in fact_tags))),
        merge_weights=(),
        turn_index=int(turn_index),
    )


def step_branch_capsule(
        *,
        parent: MergeableLatentCapsule,
        payload: Sequence[float],
        confidence: float | None = None,
        new_fact_tags: Sequence[str] = (),
        turn_index: int | None = None,
        branch_id: str | None = None,
) -> MergeableLatentCapsule:
    """Linear step from a single parent capsule.

    Inherits parent's branch_id by default; bumps turn index by 1.
    Confidence defaults to parent's confidence."""
    bid = (
        str(branch_id) if branch_id is not None
        else str(parent.branch_id))
    cnf = (
        float(confidence)
        if confidence is not None
        else float(parent.confidence))
    cnf = float(max(0.0, min(1.0, cnf)))
    tags = tuple(sorted(
        set(str(t) for t in parent.fact_tags)
        | set(str(t) for t in new_fact_tags)))
    ti = (
        int(turn_index) if turn_index is not None
        else int(parent.turn_index) + 1)
    return MergeableLatentCapsule(
        kind=W53_MLSC_KIND_BRANCH,
        branch_id=bid,
        parent_cids=(parent.cid(),),
        payload=tuple(_round_floats(payload)),
        confidence=cnf,
        fact_tags=tags,
        merge_weights=(),
        turn_index=ti,
    )


# =============================================================================
# MergeOperator — confidence-weighted convex combine
# =============================================================================


@dataclasses.dataclass(frozen=True)
class MergeOperator:
    """A confidence-weighted convex-combine merge operator.

    Given parents ``[p_1, ..., p_n]`` with confidences
    ``[c_1, ..., c_n]``:

        w_i = softmax(c_i / temperature)
        payload_merged = Σ w_i * payload_i
        confidence_merged = Σ w_i * c_i  (in [0, 1])

    The merged capsule's branch_id is a deterministic combination
    of the parents' branch_ids. Audit trail entries can be
    recovered by any consumer with the parent CIDs.
    """

    factor_dim: int
    temperature: float = W53_DEFAULT_MLSC_MERGE_TEMP

    def merge(
            self,
            parents: Sequence[MergeableLatentCapsule],
            *,
            extra_fact_tags: Sequence[str] = (),
            turn_index: int | None = None,
    ) -> MergeableLatentCapsule:
        if len(parents) < 2:
            raise ValueError(
                "MergeOperator.merge requires ≥ 2 parents")
        # Softmax weights from parent confidences.
        cs = [float(p.confidence) for p in parents]
        T = max(1e-6, float(self.temperature))
        scaled = [c / T for c in cs]
        m = max(scaled)
        exps = [math.exp(s - m) for s in scaled]
        z = float(sum(exps)) or 1.0
        ws = [float(e / z) for e in exps]
        # Convex combine the payloads.
        payload = [0.0] * int(self.factor_dim)
        for w, p in zip(ws, parents):
            for j in range(int(self.factor_dim)):
                payload[j] += float(w) * float(
                    p.payload[j] if j < len(p.payload) else 0.0)
        # Confidence: weighted mean of parent confidences.
        cnf = float(sum(w * c for w, c in zip(ws, cs)))
        cnf = float(max(0.0, min(1.0, cnf)))
        # Branch id: deterministic merge name.
        bids = sorted({str(p.branch_id) for p in parents})
        merged_bid = "merge:" + "+".join(bids)
        # Tags: union of parents' tags + extra.
        tags: set[str] = set()
        for p in parents:
            for t in p.fact_tags:
                tags.add(str(t))
        for t in extra_fact_tags:
            tags.add(str(t))
        # Turn: max of parents + 1 unless given.
        ti = (
            int(turn_index)
            if turn_index is not None
            else max(int(p.turn_index) for p in parents) + 1)
        return MergeableLatentCapsule(
            kind=W53_MLSC_KIND_MERGE,
            branch_id=str(merged_bid),
            parent_cids=tuple(str(p.cid()) for p in parents),
            payload=tuple(_round_floats(payload)),
            confidence=cnf,
            fact_tags=tuple(sorted(tags)),
            merge_weights=tuple(_round_floats(ws)),
            turn_index=ti,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": str(W53_MLSC_SCHEMA_VERSION),
            "factor_dim": int(self.factor_dim),
            "temperature": float(round(self.temperature, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w53_mlsc_operator",
            "operator": self.to_dict()})


# =============================================================================
# MergeAuditTrail — content-addressed log of merges
# =============================================================================


@dataclasses.dataclass(frozen=True)
class MergeAuditEntry:
    """One entry in the merge audit log."""

    merged_cid: str
    parent_cids: tuple[str, ...]
    parent_branch_ids: tuple[str, ...]
    merge_weights: tuple[float, ...]
    operator_cid: str
    turn_index: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "merged_cid": str(self.merged_cid),
            "parent_cids": list(self.parent_cids),
            "parent_branch_ids": list(self.parent_branch_ids),
            "merge_weights": list(
                _round_floats(self.merge_weights)),
            "operator_cid": str(self.operator_cid),
            "turn_index": int(self.turn_index),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w53_mlsc_audit_entry",
            "entry": self.to_dict()})


@dataclasses.dataclass
class MergeAuditTrail:
    """A content-addressed audit log of merge entries.

    Supports ``walk_to_roots`` from any leaf merge-capsule CID:
    returns the deduplicated set of root CIDs visible by walking
    parent_cids transitively through both the audit log and the
    capsule store.
    """

    entries: dict[str, MergeAuditEntry]

    @classmethod
    def empty(cls) -> "MergeAuditTrail":
        return cls(entries={})

    def add(self, entry: MergeAuditEntry) -> None:
        self.entries[entry.merged_cid] = entry

    def get(self, merged_cid: str) -> MergeAuditEntry | None:
        return self.entries.get(str(merged_cid))

    def walk_to_roots(
            self, leaf_cid: str,
            *,
            capsule_store: Mapping[
                str, MergeableLatentCapsule],
            max_steps: int = 256,
    ) -> list[str]:
        """Walk from a leaf back to root CIDs.

        For each branch capsule we walk its single parent_cid;
        for each merge entry we walk both parents. Returns the
        deduplicated, sorted list of root CIDs.
        """
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
                # Treat unknown CIDs as roots (orphaned but
                # explicit — important for the audit cap).
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

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w53_mlsc_audit_trail",
            "entries": [
                {"cid": c, "entry": e.to_dict()}
                for c, e in sorted(self.entries.items())
            ],
        })

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": str(W53_MLSC_SCHEMA_VERSION),
            "n_entries": int(len(self.entries)),
            "audit_cid": str(self.cid()),
        }


def merge_capsules(
        operator: MergeOperator,
        parents: Sequence[MergeableLatentCapsule],
        *,
        audit_trail: MergeAuditTrail,
        extra_fact_tags: Sequence[str] = (),
        turn_index: int | None = None,
) -> MergeableLatentCapsule:
    """Apply ``operator.merge`` and append an audit entry.

    Returns the merged capsule; mutates ``audit_trail`` in place.
    """
    merged = operator.merge(
        parents,
        extra_fact_tags=extra_fact_tags,
        turn_index=turn_index,
    )
    audit = MergeAuditEntry(
        merged_cid=str(merged.cid()),
        parent_cids=tuple(str(p.cid()) for p in parents),
        parent_branch_ids=tuple(
            str(p.branch_id) for p in parents),
        merge_weights=tuple(merged.merge_weights),
        operator_cid=str(operator.cid()),
        turn_index=int(merged.turn_index),
    )
    audit_trail.add(audit)
    return merged


# =============================================================================
# Witness
# =============================================================================


@dataclasses.dataclass(frozen=True)
class MergeableLatentCapsuleWitness:
    """Witness over a merged capsule + audit trail."""

    leaf_cid: str
    operator_cid: str
    audit_trail_cid: str
    audit_entry_count: int
    n_unique_roots: int
    parents_count: int
    merge_weights: tuple[float, ...]
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "leaf_cid": str(self.leaf_cid),
            "operator_cid": str(self.operator_cid),
            "audit_trail_cid": str(self.audit_trail_cid),
            "audit_entry_count": int(self.audit_entry_count),
            "n_unique_roots": int(self.n_unique_roots),
            "parents_count": int(self.parents_count),
            "merge_weights": list(
                _round_floats(self.merge_weights)),
            "confidence": float(round(self.confidence, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w53_mlsc_witness",
            "witness": self.to_dict()})


def emit_mlsc_witness(
        *,
        leaf: MergeableLatentCapsule,
        operator: MergeOperator,
        audit_trail: MergeAuditTrail,
        capsule_store: Mapping[str, MergeableLatentCapsule],
) -> MergeableLatentCapsuleWitness:
    roots = audit_trail.walk_to_roots(
        leaf.cid(), capsule_store=capsule_store)
    return MergeableLatentCapsuleWitness(
        leaf_cid=str(leaf.cid()),
        operator_cid=str(operator.cid()),
        audit_trail_cid=str(audit_trail.cid()),
        audit_entry_count=int(len(audit_trail.entries)),
        n_unique_roots=int(len(roots)),
        parents_count=int(len(leaf.parent_cids)),
        merge_weights=tuple(leaf.merge_weights),
        confidence=float(leaf.confidence),
    )


# =============================================================================
# Verifier
# =============================================================================


W53_MLSC_VERIFIER_FAILURE_MODES: tuple[str, ...] = (
    "w53_mlsc_schema_mismatch",
    "w53_mlsc_leaf_cid_mismatch",
    "w53_mlsc_operator_cid_mismatch",
    "w53_mlsc_audit_trail_cid_mismatch",
    "w53_mlsc_audit_entry_count_below_floor",
    "w53_mlsc_n_unique_roots_below_floor",
    "w53_mlsc_orphan_parent_in_audit_trail",
    "w53_mlsc_confidence_out_of_bounds",
    "w53_mlsc_merge_weights_dont_sum_to_one",
)


def verify_mlsc_witness(
        witness: MergeableLatentCapsuleWitness,
        *,
        expected_leaf_cid: str | None = None,
        expected_operator_cid: str | None = None,
        expected_audit_trail_cid: str | None = None,
        min_audit_entry_count: int | None = None,
        min_n_unique_roots: int | None = None,
        capsule_store: Mapping[
            str, MergeableLatentCapsule] | None = None,
        audit_trail: MergeAuditTrail | None = None,
) -> dict[str, Any]:
    failures: list[str] = []
    if (expected_leaf_cid is not None
            and witness.leaf_cid != str(expected_leaf_cid)):
        failures.append("w53_mlsc_leaf_cid_mismatch")
    if (expected_operator_cid is not None
            and witness.operator_cid
            != str(expected_operator_cid)):
        failures.append("w53_mlsc_operator_cid_mismatch")
    if (expected_audit_trail_cid is not None
            and witness.audit_trail_cid
            != str(expected_audit_trail_cid)):
        failures.append("w53_mlsc_audit_trail_cid_mismatch")
    if (min_audit_entry_count is not None
            and witness.audit_entry_count
            < int(min_audit_entry_count)):
        failures.append(
            "w53_mlsc_audit_entry_count_below_floor")
    if (min_n_unique_roots is not None
            and witness.n_unique_roots
            < int(min_n_unique_roots)):
        failures.append(
            "w53_mlsc_n_unique_roots_below_floor")
    if not (0.0 <= float(witness.confidence) <= 1.0):
        failures.append("w53_mlsc_confidence_out_of_bounds")
    # Sum-to-one check for merge weights (with tolerance).
    if witness.merge_weights:
        s = float(sum(witness.merge_weights))
        if abs(s - 1.0) > 1e-6:
            failures.append(
                "w53_mlsc_merge_weights_dont_sum_to_one")
    # Orphan check: every parent_cid in audit must be present in
    # the capsule_store.
    if (audit_trail is not None
            and capsule_store is not None):
        for entry in audit_trail.entries.values():
            for p in entry.parent_cids:
                if p not in capsule_store:
                    failures.append(
                        "w53_mlsc_orphan_parent_in_audit_trail")
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
# Branch-merge consensus quorum
# =============================================================================


@dataclasses.dataclass(frozen=True)
class ConsensusQuorumResult:
    """Result of a K-of-N consensus vote across N branch capsules.

    The vote is computed by computing pairwise cosines on
    payloads, picking the largest connected K-clique and
    emitting its merged capsule. If no K-clique exists, the
    result is ``abstain=True``.
    """

    n_branches: int
    k_required: int
    quorum_reached: bool
    abstain: bool
    consensus_payload: tuple[float, ...]
    consensus_cosine_floor: float
    consensus_capsule_cid: str
    selected_branch_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_branches": int(self.n_branches),
            "k_required": int(self.k_required),
            "quorum_reached": bool(self.quorum_reached),
            "abstain": bool(self.abstain),
            "consensus_payload": list(_round_floats(
                self.consensus_payload)),
            "consensus_cosine_floor": float(round(
                self.consensus_cosine_floor, 12)),
            "consensus_capsule_cid": str(
                self.consensus_capsule_cid),
            "selected_branch_ids": list(
                self.selected_branch_ids),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w53_mlsc_consensus_result",
            "result": self.to_dict()})


def compute_consensus_quorum(
        branches: Sequence[MergeableLatentCapsule],
        *,
        operator: MergeOperator,
        audit_trail: MergeAuditTrail,
        k_required: int,
        cosine_floor: float = 0.5,
) -> ConsensusQuorumResult:
    """Greedy consensus: pick the largest set of branches whose
    pairwise cosines are all ≥ ``cosine_floor``; require ≥ K
    members. If found, merge them via ``operator``; else
    abstain.
    """
    n = len(branches)
    if n == 0 or int(k_required) <= 0:
        return ConsensusQuorumResult(
            n_branches=int(n),
            k_required=int(k_required),
            quorum_reached=False,
            abstain=True,
            consensus_payload=(),
            consensus_cosine_floor=float(cosine_floor),
            consensus_capsule_cid="",
            selected_branch_ids=(),
        )
    # Compute pairwise cosines.
    cos = [[1.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            c = _cosine(
                branches[i].payload, branches[j].payload)
            cos[i][j] = float(c)
            cos[j][i] = float(c)
    # Greedy: find the largest set s.t. all pairwise cos ≥ floor.
    # Sort branches by sum of pairwise cosines (most central first).
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
    if len(selected) < int(k_required):
        return ConsensusQuorumResult(
            n_branches=int(n),
            k_required=int(k_required),
            quorum_reached=False,
            abstain=True,
            consensus_payload=(),
            consensus_cosine_floor=float(cosine_floor),
            consensus_capsule_cid="",
            selected_branch_ids=(),
        )
    selected_branches = [branches[i] for i in selected]
    merged = merge_capsules(
        operator, selected_branches,
        audit_trail=audit_trail,
        extra_fact_tags=(f"quorum:k={k_required}",))
    floor = min(
        cos[a][b]
        for a in selected for b in selected if a < b
    ) if len(selected) > 1 else 1.0
    return ConsensusQuorumResult(
        n_branches=int(n),
        k_required=int(k_required),
        quorum_reached=True,
        abstain=False,
        consensus_payload=tuple(merged.payload),
        consensus_cosine_floor=float(floor),
        consensus_capsule_cid=str(merged.cid()),
        selected_branch_ids=tuple(
            str(b.branch_id) for b in selected_branches),
    )


__all__ = [
    "W53_MLSC_SCHEMA_VERSION",
    "W53_MLSC_NO_PARENT",
    "W53_MLSC_KIND_ROOT",
    "W53_MLSC_KIND_BRANCH",
    "W53_MLSC_KIND_MERGE",
    "W53_DEFAULT_MLSC_FACTOR_DIM",
    "W53_DEFAULT_MLSC_MERGE_TEMP",
    "W53_MLSC_VERIFIER_FAILURE_MODES",
    "MergeableLatentCapsule",
    "MergeOperator",
    "MergeAuditEntry",
    "MergeAuditTrail",
    "MergeableLatentCapsuleWitness",
    "ConsensusQuorumResult",
    "make_root_capsule",
    "step_branch_capsule",
    "merge_capsules",
    "emit_mlsc_witness",
    "verify_mlsc_witness",
    "compute_consensus_quorum",
]
