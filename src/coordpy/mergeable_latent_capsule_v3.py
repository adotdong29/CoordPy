"""W55 M3 — Mergeable Latent State Capsule V3 (MLSC V3).

Extends W54 MLSC V2 with three first-class additions:

* **disagreement algebra primitives** ⊕ (merge), ⊖ (difference),
  ⊗ (intersection-of-agreement) operating on capsule payloads.
  Every algebra step writes a content-addressed record to the
  audit trail.
* **per-fact confirmation count** — instead of a single
  ``tag → contributor_cid`` map, MLSC V3 tracks
  ``tag → (contributor_cids: tuple, count: int)``. A fact tag
  that comes from N independent parents has count = N.
  Confirmation count rises monotonically under merges.
* **trust signature decay** — each step applies
  ``trust ← decay_factor · trust`` unless the capsule is a
  merge that reinforces the trust (trust is the convex blend
  of parent trusts, optionally capped by a reinforcement
  multiplier).

The base MLSC V2 abstractions remain unchanged; V3 wraps V2 with
``MergeableLatentCapsuleV3``, ``MergeOperatorV3``,
``MergeAuditEntryV3``, ``MergeAuditTrailV3``.

Honest scope: pure-Python only, capsule-layer only, no
transformer-internal state.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from .disagreement_algebra import (
    AlgebraTrace,
    difference_op,
    intersection_op,
    merge_op,
    W55_DEFAULT_AGREEMENT_FLOOR,
)
from .mergeable_latent_capsule import (
    W53_DEFAULT_MLSC_FACTOR_DIM,
    W53_DEFAULT_MLSC_MERGE_TEMP,
    W53_MLSC_KIND_BRANCH,
    W53_MLSC_KIND_MERGE,
    W53_MLSC_KIND_ROOT,
)
from .mergeable_latent_capsule_v2 import (
    MergeableLatentCapsuleV2,
    W54_DEFAULT_MLSC_V2_FACTOR_DIM,
    W54_DEFAULT_MLSC_V2_TRUST_DEFAULT,
    W54_DEFAULT_MLSC_V2_TRUST_FLOOR,
)


# =============================================================================
# Schema, defaults
# =============================================================================

W55_MLSC_V3_SCHEMA_VERSION: str = (
    "coordpy.mergeable_latent_capsule_v3.v1")

W55_DEFAULT_MLSC_V3_FACTOR_DIM: int = (
    W54_DEFAULT_MLSC_V2_FACTOR_DIM)
W55_DEFAULT_MLSC_V3_TRUST_DEFAULT: float = (
    W54_DEFAULT_MLSC_V2_TRUST_DEFAULT)
W55_DEFAULT_MLSC_V3_TRUST_FLOOR: float = (
    W54_DEFAULT_MLSC_V2_TRUST_FLOOR)
W55_DEFAULT_MLSC_V3_TRUST_DECAY: float = 0.95
W55_DEFAULT_MLSC_V3_TRUST_RECOVERY_FLOOR: float = 0.1


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
# MergeableLatentCapsuleV3
# =============================================================================


@dataclasses.dataclass(frozen=True)
class FactConfirmation:
    """One fact_tag's confirmation record."""

    tag: str
    contributor_cids: tuple[str, ...]
    count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "tag": str(self.tag),
            "contributor_cids": list(self.contributor_cids),
            "count": int(self.count),
        }


@dataclasses.dataclass(frozen=True)
class MergeableLatentCapsuleV3:
    """V3 capsule: V2 + per-fact confirmation + trust decay."""

    kind: str
    branch_id: str
    parent_cids: tuple[str, ...]
    payload: tuple[float, ...]
    confidence: float
    trust: float
    trust_decay: float
    fact_confirmations: tuple[FactConfirmation, ...]
    merge_weights: tuple[float, ...]
    disagreement_per_dim: tuple[float, ...]
    agreement_mask: tuple[int, ...]
    turn_index: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": str(
                W55_MLSC_V3_SCHEMA_VERSION),
            "kind": str(self.kind),
            "branch_id": str(self.branch_id),
            "parent_cids": list(self.parent_cids),
            "payload": list(_round_floats(self.payload)),
            "confidence": float(round(self.confidence, 12)),
            "trust": float(round(self.trust, 12)),
            "trust_decay": float(round(self.trust_decay, 12)),
            "fact_confirmations": [
                fc.to_dict() for fc in self.fact_confirmations],
            "merge_weights": list(_round_floats(
                self.merge_weights)),
            "disagreement_per_dim": list(_round_floats(
                self.disagreement_per_dim)),
            "agreement_mask": list(self.agreement_mask),
            "turn_index": int(self.turn_index),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w55_mlsc_v3_capsule",
            "capsule": self.to_dict()})

    @property
    def fact_tags(self) -> tuple[str, ...]:
        return tuple(fc.tag for fc in self.fact_confirmations)

    @property
    def is_merge(self) -> bool:
        return self.kind == W53_MLSC_KIND_MERGE

    def get_confirmation_count(self, tag: str) -> int:
        for fc in self.fact_confirmations:
            if fc.tag == str(tag):
                return int(fc.count)
        return 0


def make_root_capsule_v3(
        *,
        branch_id: str,
        payload: Sequence[float],
        confidence: float = 1.0,
        trust: float = W55_DEFAULT_MLSC_V3_TRUST_DEFAULT,
        trust_decay: float = W55_DEFAULT_MLSC_V3_TRUST_DECAY,
        fact_tags: Sequence[str] = (),
        turn_index: int = 0,
) -> MergeableLatentCapsuleV3:
    tags = tuple(sorted(set(str(t) for t in fact_tags)))
    fcs = tuple(
        FactConfirmation(
            tag=t, contributor_cids=("self",), count=1)
        for t in tags
    )
    return MergeableLatentCapsuleV3(
        kind=W53_MLSC_KIND_ROOT,
        branch_id=str(branch_id),
        parent_cids=(),
        payload=tuple(_round_floats(payload)),
        confidence=float(max(0.0, min(1.0, float(confidence)))),
        trust=float(max(0.0, min(1.0, float(trust)))),
        trust_decay=float(max(0.0, min(1.0, float(trust_decay)))),
        fact_confirmations=fcs,
        merge_weights=(),
        disagreement_per_dim=(),
        agreement_mask=(),
        turn_index=int(turn_index),
    )


def step_branch_capsule_v3(
        *,
        parent: MergeableLatentCapsuleV3,
        payload: Sequence[float],
        confidence: float | None = None,
        trust: float | None = None,
        new_fact_tags: Sequence[str] = (),
        turn_index: int | None = None,
        branch_id: str | None = None,
        apply_decay: bool = True,
) -> MergeableLatentCapsuleV3:
    bid = (
        str(branch_id) if branch_id is not None
        else str(parent.branch_id))
    cnf = (
        float(confidence) if confidence is not None
        else float(parent.confidence))
    cnf = float(max(0.0, min(1.0, cnf)))
    # Trust decay step.
    base_trust = (
        float(trust) if trust is not None
        else float(parent.trust))
    if apply_decay:
        tr = float(max(0.0, min(
            1.0, base_trust * float(parent.trust_decay))))
    else:
        tr = float(max(0.0, min(1.0, base_trust)))
    # Preserve trust_decay setting through the branch.
    new_tags = set(str(t) for t in new_fact_tags) - set(
        parent.fact_tags)
    parent_cid = str(parent.cid())
    # Confirmations preserved from parent; new tags get count 1
    # with parent as the contributor.
    new_confs = list(parent.fact_confirmations)
    for t in sorted(new_tags):
        new_confs.append(FactConfirmation(
            tag=t, contributor_cids=(parent_cid,), count=1))
    new_confs.sort(key=lambda fc: fc.tag)
    ti = (
        int(turn_index) if turn_index is not None
        else int(parent.turn_index) + 1)
    return MergeableLatentCapsuleV3(
        kind=W53_MLSC_KIND_BRANCH,
        branch_id=bid,
        parent_cids=(parent_cid,),
        payload=tuple(_round_floats(payload)),
        confidence=cnf, trust=tr,
        trust_decay=float(parent.trust_decay),
        fact_confirmations=tuple(new_confs),
        merge_weights=(),
        disagreement_per_dim=(),
        agreement_mask=(),
        turn_index=ti,
    )


def reinforce_capsule_trust_v3(
        capsule: MergeableLatentCapsuleV3,
        *,
        reinforcement: float = 1.0,
) -> MergeableLatentCapsuleV3:
    """Reinforce a capsule's trust (anti-decay).

    Returns a fresh capsule with same CID-shape but trust raised
    toward 1.0 by ``reinforcement`` (clamped to [0,1]).
    """
    new_trust = float(max(0.0, min(
        1.0,
        float(capsule.trust)
        + float(reinforcement)
        * (1.0 - float(capsule.trust)))))
    return dataclasses.replace(capsule, trust=new_trust)


# =============================================================================
# MergeOperatorV3 — trust-weighted convex combine with algebra
# =============================================================================


@dataclasses.dataclass(frozen=True)
class MergeOperatorV3:
    """V3 merge operator: V2 trust-softmax + algebra-traced merge.

    Merge weight per parent ``i``:
        scaled_i = c_i / T (softmax over scaled)
        raw_w_i = softmax(scaled)_i
        trust_norm = max(trust_floor, t_i)
        w_i = raw_w_i * trust_norm
        w_i ← w_i / Σ w_i
        payload_merged = Σ w_i * payload_i
        confidence_merged = Σ w_i * c_i
        trust_merged = clamp(min(weighted_trust * reinforce, 1.0))

    Plus:
        agreement_mask = 1 where all parents pairwise agree
            within agreement_floor; 0 elsewhere
        fact_confirmation_count[t] = sum of parents with t in tags
    """

    factor_dim: int
    temperature: float = W53_DEFAULT_MLSC_MERGE_TEMP
    trust_floor: float = W55_DEFAULT_MLSC_V3_TRUST_FLOOR
    agreement_floor: float = W55_DEFAULT_AGREEMENT_FLOOR
    merge_trust_reinforcement: float = 1.05

    def merge(
            self,
            parents: Sequence[MergeableLatentCapsuleV3],
            *,
            extra_fact_tags: Sequence[str] = (),
            turn_index: int | None = None,
            algebra_trace: AlgebraTrace | None = None,
    ) -> MergeableLatentCapsuleV3:
        if len(parents) < 2:
            raise ValueError(
                "MergeOperatorV3.merge requires ≥ 2 parents")
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
        # Per-dim disagreement: weighted std-dev.
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
        # Agreement mask: dim i is in agreement subspace iff
        # ALL pairwise differences are within agreement_floor.
        af = float(self.agreement_floor)
        agreement_mask = [1] * int(self.factor_dim)
        for j in range(int(self.factor_dim)):
            for a in range(len(parents)):
                for b in range(a + 1, len(parents)):
                    pa = float(
                        parents[a].payload[j]
                        if j < len(parents[a].payload)
                        else 0.0)
                    pb = float(
                        parents[b].payload[j]
                        if j < len(parents[b].payload)
                        else 0.0)
                    if abs(pa - pb) > af:
                        agreement_mask[j] = 0
                        break
                if agreement_mask[j] == 0:
                    break
        cnf = float(sum(w * c for w, c in zip(ws, cs)))
        cnf = float(max(0.0, min(1.0, cnf)))
        # Weighted trust + reinforcement.
        trust_avg = float(sum(
            w * t for w, t in zip(ws, weighted_t)))
        trust_avg = float(max(0.0, min(
            1.0, trust_avg
            * float(self.merge_trust_reinforcement))))
        # Per-fact confirmation count: sum across parents who
        # claim the tag.
        tag_to_count: dict[str, int] = {}
        tag_to_cids: dict[str, list[str]] = {}
        for p in parents:
            p_cid = str(p.cid())
            for fc in p.fact_confirmations:
                tag_to_count[fc.tag] = (
                    tag_to_count.get(fc.tag, 0)
                    + int(fc.count))
                contribs = tag_to_cids.setdefault(fc.tag, [])
                for c in fc.contributor_cids:
                    if c == "self":
                        contribs.append(p_cid)
                    elif c not in contribs:
                        contribs.append(c)
        for t in extra_fact_tags:
            tag_to_count[str(t)] = (
                tag_to_count.get(str(t), 0) + 1)
            tag_to_cids.setdefault(
                str(t), []).append("merge")
        fcs = tuple(sorted([
            FactConfirmation(
                tag=str(t),
                contributor_cids=tuple(
                    tag_to_cids.get(t, [])),
                count=int(tag_to_count[t]))
            for t in tag_to_count.keys()
        ], key=lambda fc: fc.tag))
        bids = sorted({str(p.branch_id) for p in parents})
        merged_bid = "merge:" + "+".join(bids)
        ti = (
            int(turn_index) if turn_index is not None
            else max(int(p.turn_index) for p in parents) + 1)
        # Trace into the algebra trail if provided.
        if algebra_trace is not None:
            # Walk: ⊕ over pairwise; ⊖ records disagreement.
            from .disagreement_algebra import (
                merge_op_traced as _mt,
                difference_op_traced as _dt,
                intersection_op_traced as _it,
            )
            cur = list(parents[0].payload)
            for p in parents[1:]:
                cur = _mt(cur, list(p.payload),
                           trace=algebra_trace)
            _dt(list(parents[0].payload),
                 list(parents[-1].payload),
                 trace=algebra_trace)
            _it(list(parents[0].payload),
                 list(parents[-1].payload),
                 agreement_floor=af,
                 trace=algebra_trace)
        return MergeableLatentCapsuleV3(
            kind=W53_MLSC_KIND_MERGE,
            branch_id=str(merged_bid),
            parent_cids=tuple(str(p.cid()) for p in parents),
            payload=tuple(_round_floats(payload)),
            confidence=cnf,
            trust=trust_avg,
            trust_decay=float(
                parents[0].trust_decay),
            fact_confirmations=fcs,
            merge_weights=tuple(_round_floats(ws)),
            disagreement_per_dim=tuple(_round_floats(
                disagreement)),
            agreement_mask=tuple(agreement_mask),
            turn_index=ti,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": str(
                W55_MLSC_V3_SCHEMA_VERSION),
            "factor_dim": int(self.factor_dim),
            "temperature": float(round(
                self.temperature, 12)),
            "trust_floor": float(round(
                self.trust_floor, 12)),
            "agreement_floor": float(round(
                self.agreement_floor, 12)),
            "merge_trust_reinforcement": float(round(
                self.merge_trust_reinforcement, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w55_mlsc_v3_operator",
            "operator": self.to_dict()})


# =============================================================================
# MergeAuditTrailV3
# =============================================================================


@dataclasses.dataclass(frozen=True)
class MergeAuditEntryV3:
    merged_cid: str
    parent_cids: tuple[str, ...]
    parent_branch_ids: tuple[str, ...]
    parent_trusts: tuple[float, ...]
    merge_weights: tuple[float, ...]
    disagreement_per_dim: tuple[float, ...]
    agreement_mask: tuple[int, ...]
    fact_confirmations: tuple[FactConfirmation, ...]
    operator_cid: str
    algebra_trace_cid: str
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
            "agreement_mask": list(self.agreement_mask),
            "fact_confirmations": [
                fc.to_dict() for fc in self.fact_confirmations],
            "operator_cid": str(self.operator_cid),
            "algebra_trace_cid": str(self.algebra_trace_cid),
            "turn_index": int(self.turn_index),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w55_mlsc_v3_audit_entry",
            "entry": self.to_dict()})


@dataclasses.dataclass
class MergeAuditTrailV3:
    entries: dict[str, MergeAuditEntryV3]

    @classmethod
    def empty(cls) -> "MergeAuditTrailV3":
        return cls(entries={})

    def add(self, entry: MergeAuditEntryV3) -> None:
        self.entries[entry.merged_cid] = entry

    def get(self, merged_cid: str) -> MergeAuditEntryV3 | None:
        return self.entries.get(str(merged_cid))

    def walk_to_roots(
            self, leaf_cid: str, *,
            capsule_store: Mapping[
                str, MergeableLatentCapsuleV3],
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

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w55_mlsc_v3_audit_trail",
            "entries": [
                {"cid": c, "entry": e.to_dict()}
                for c, e in sorted(self.entries.items())
            ],
        })

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": str(
                W55_MLSC_V3_SCHEMA_VERSION),
            "n_entries": int(len(self.entries)),
            "audit_cid": str(self.cid()),
        }


def merge_capsules_v3(
        operator: MergeOperatorV3,
        parents: Sequence[MergeableLatentCapsuleV3],
        *,
        audit_trail: MergeAuditTrailV3,
        algebra_trace: AlgebraTrace | None = None,
        extra_fact_tags: Sequence[str] = (),
        turn_index: int | None = None,
) -> MergeableLatentCapsuleV3:
    merged = operator.merge(
        parents,
        extra_fact_tags=extra_fact_tags,
        turn_index=turn_index,
        algebra_trace=algebra_trace)
    trace_cid = (
        algebra_trace.cid() if algebra_trace is not None
        else "")
    entry = MergeAuditEntryV3(
        merged_cid=str(merged.cid()),
        parent_cids=tuple(str(p.cid()) for p in parents),
        parent_branch_ids=tuple(
            str(p.branch_id) for p in parents),
        parent_trusts=tuple(float(p.trust) for p in parents),
        merge_weights=tuple(merged.merge_weights),
        disagreement_per_dim=tuple(
            merged.disagreement_per_dim),
        agreement_mask=tuple(merged.agreement_mask),
        fact_confirmations=tuple(merged.fact_confirmations),
        operator_cid=str(operator.cid()),
        algebra_trace_cid=str(trace_cid),
        turn_index=int(merged.turn_index),
    )
    audit_trail.add(entry)
    return merged


# =============================================================================
# Witness
# =============================================================================


@dataclasses.dataclass(frozen=True)
class MergeableLatentCapsuleV3Witness:
    leaf_cid: str
    operator_cid: str
    audit_trail_cid: str
    walk_to_roots_cids: tuple[str, ...]
    leaf_kind: str
    leaf_branch_id: str
    leaf_payload_l2: float
    leaf_confidence: float
    leaf_trust: float
    leaf_trust_decay: float
    leaf_disagreement_l2: float
    leaf_agreement_mask_sum: int
    leaf_fact_count: int
    leaf_max_confirmation_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "leaf_cid": str(self.leaf_cid),
            "operator_cid": str(self.operator_cid),
            "audit_trail_cid": str(self.audit_trail_cid),
            "walk_to_roots_cids": list(
                self.walk_to_roots_cids),
            "leaf_kind": str(self.leaf_kind),
            "leaf_branch_id": str(self.leaf_branch_id),
            "leaf_payload_l2": float(round(
                self.leaf_payload_l2, 12)),
            "leaf_confidence": float(round(
                self.leaf_confidence, 12)),
            "leaf_trust": float(round(self.leaf_trust, 12)),
            "leaf_trust_decay": float(round(
                self.leaf_trust_decay, 12)),
            "leaf_disagreement_l2": float(round(
                self.leaf_disagreement_l2, 12)),
            "leaf_agreement_mask_sum": int(
                self.leaf_agreement_mask_sum),
            "leaf_fact_count": int(self.leaf_fact_count),
            "leaf_max_confirmation_count": int(
                self.leaf_max_confirmation_count),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w55_mlsc_v3_witness",
            "witness": self.to_dict()})


def emit_mlsc_v3_witness(
        *,
        leaf: MergeableLatentCapsuleV3,
        operator: MergeOperatorV3,
        audit_trail: MergeAuditTrailV3,
        capsule_store: Mapping[
            str, MergeableLatentCapsuleV3],
) -> MergeableLatentCapsuleV3Witness:
    roots = audit_trail.walk_to_roots(
        leaf.cid(), capsule_store=capsule_store)
    l2 = math.sqrt(sum(float(v) ** 2 for v in leaf.payload))
    dis_l2 = math.sqrt(sum(
        float(v) ** 2 for v in leaf.disagreement_per_dim))
    max_count = max(
        (int(fc.count) for fc in leaf.fact_confirmations),
        default=0)
    return MergeableLatentCapsuleV3Witness(
        leaf_cid=str(leaf.cid()),
        operator_cid=str(operator.cid()),
        audit_trail_cid=str(audit_trail.cid()),
        walk_to_roots_cids=tuple(roots),
        leaf_kind=str(leaf.kind),
        leaf_branch_id=str(leaf.branch_id),
        leaf_payload_l2=float(l2),
        leaf_confidence=float(leaf.confidence),
        leaf_trust=float(leaf.trust),
        leaf_trust_decay=float(leaf.trust_decay),
        leaf_disagreement_l2=float(dis_l2),
        leaf_agreement_mask_sum=int(
            sum(leaf.agreement_mask)),
        leaf_fact_count=int(len(leaf.fact_confirmations)),
        leaf_max_confirmation_count=int(max_count),
    )


# =============================================================================
# Verifier
# =============================================================================

W55_MLSC_V3_VERIFIER_FAILURE_MODES: tuple[str, ...] = (
    "w55_mlsc_v3_leaf_cid_mismatch",
    "w55_mlsc_v3_operator_cid_mismatch",
    "w55_mlsc_v3_audit_trail_walk_orphan",
    "w55_mlsc_v3_trust_out_of_bounds",
    "w55_mlsc_v3_confidence_out_of_bounds",
    "w55_mlsc_v3_disagreement_negative",
    "w55_mlsc_v3_merge_weights_dont_sum_to_one",
    "w55_mlsc_v3_fact_count_below_floor",
)


def verify_mlsc_v3_witness(
        witness: MergeableLatentCapsuleV3Witness,
        *,
        expected_leaf_cid: str | None = None,
        expected_operator_cid: str | None = None,
        min_fact_count: int | None = None,
) -> dict[str, Any]:
    failures: list[str] = []
    if (expected_leaf_cid is not None
            and witness.leaf_cid != str(expected_leaf_cid)):
        failures.append("w55_mlsc_v3_leaf_cid_mismatch")
    if (expected_operator_cid is not None
            and witness.operator_cid
            != str(expected_operator_cid)):
        failures.append("w55_mlsc_v3_operator_cid_mismatch")
    if not (0.0 <= float(witness.leaf_trust) <= 1.0):
        failures.append("w55_mlsc_v3_trust_out_of_bounds")
    if not (0.0 <= float(witness.leaf_confidence) <= 1.0):
        failures.append(
            "w55_mlsc_v3_confidence_out_of_bounds")
    if float(witness.leaf_disagreement_l2) < 0.0:
        failures.append("w55_mlsc_v3_disagreement_negative")
    if (min_fact_count is not None
            and witness.leaf_fact_count < int(min_fact_count)):
        failures.append("w55_mlsc_v3_fact_count_below_floor")
    return {
        "ok": (len(failures) == 0),
        "failures": failures,
        "witness_cid": witness.cid(),
    }


__all__ = [
    "W55_MLSC_V3_SCHEMA_VERSION",
    "W55_DEFAULT_MLSC_V3_FACTOR_DIM",
    "W55_DEFAULT_MLSC_V3_TRUST_DEFAULT",
    "W55_DEFAULT_MLSC_V3_TRUST_FLOOR",
    "W55_DEFAULT_MLSC_V3_TRUST_DECAY",
    "W55_DEFAULT_MLSC_V3_TRUST_RECOVERY_FLOOR",
    "W55_MLSC_V3_VERIFIER_FAILURE_MODES",
    "FactConfirmation",
    "MergeableLatentCapsuleV3",
    "MergeOperatorV3",
    "MergeAuditEntryV3",
    "MergeAuditTrailV3",
    "MergeableLatentCapsuleV3Witness",
    "make_root_capsule_v3",
    "step_branch_capsule_v3",
    "reinforce_capsule_trust_v3",
    "merge_capsules_v3",
    "emit_mlsc_v3_witness",
    "verify_mlsc_v3_witness",
]
