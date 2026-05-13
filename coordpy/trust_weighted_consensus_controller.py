"""W55 M4 — Trust-Weighted Consensus Controller.

Extends the W54 K-of-N consensus controller with:

* **Continuous trust-weighted quorum** — quorum reached iff
  ``Σ_{i ∈ agreeing_subset} trust_i ≥ trust_threshold``, not just
  the K-of-N binary count. This allows a small number of highly-
  trusted parents to outweigh a large number of low-trust parents.
* **5-stage decision chain** —
    1. ``quorum_K_of_N`` — try the W54 K-of-N quorum first.
    2. ``trust_weighted_quorum`` — try the trust-weighted quorum.
    3. ``fallback_best_parent`` — pick the highest-trust
       single parent above ``fallback_cosine_floor``.
    4. ``fallback_transcript`` — emit the externally supplied
       transcript_value (passed in at decide() time).
    5. ``abstain`` — emit nothing and surface ``abstain`` as the
       final state.
* **Per-stage audit** — the audit trail records every stage
  *attempted* (not just the chosen stage) so the audit shows the
  full decision walk.

Honest scope: pure-Python only. Not a Byzantine fault-tolerant
distributed consensus protocol — only a capsule-layer agreement
mechanism. Under symmetric trust = 1.0, it reduces exactly to
W54 K-of-N (W55-L-TRUST-WEIGHTED-NOT-STRICT-DOMINANCE).
"""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
import math
from typing import Any, Mapping, Sequence

from .mergeable_latent_capsule_v3 import (
    MergeableLatentCapsuleV3,
    MergeAuditTrailV3,
    MergeOperatorV3,
    W55_DEFAULT_MLSC_V3_TRUST_FLOOR,
    merge_capsules_v3,
)


# =============================================================================
# Schema, defaults
# =============================================================================

W55_TWCC_SCHEMA_VERSION: str = (
    "coordpy.trust_weighted_consensus_controller.v1")

W55_TWCC_STAGE_K_OF_N: str = "quorum_K_of_N"
W55_TWCC_STAGE_TRUST_WEIGHTED: str = "trust_weighted_quorum"
W55_TWCC_STAGE_FALLBACK_BEST_PARENT: str = (
    "fallback_best_parent")
W55_TWCC_STAGE_FALLBACK_TRANSCRIPT: str = (
    "fallback_transcript")
W55_TWCC_STAGE_ABSTAIN: str = "abstain"

W55_TWCC_DECISION_QUORUM: str = "quorum_merged"
W55_TWCC_DECISION_TRUST_WEIGHTED: str = (
    "trust_weighted_merged")
W55_TWCC_DECISION_BEST_PARENT: str = "best_parent"
W55_TWCC_DECISION_TRANSCRIPT: str = "transcript"
W55_TWCC_DECISION_ABSTAIN: str = "abstain"

W55_DEFAULT_TWCC_K_MIN: int = 2
W55_DEFAULT_TWCC_K_MAX: int = 8
W55_DEFAULT_TWCC_COSINE_FLOOR: float = 0.5
W55_DEFAULT_TWCC_FALLBACK_COSINE_FLOOR: float = 0.3
W55_DEFAULT_TWCC_TRUST_THRESHOLD: float = 1.0


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
# Policy
# =============================================================================


@dataclasses.dataclass(frozen=True)
class TrustWeightedConsensusPolicy:
    """K-of-N + trust-weighted policy parameters."""

    k_min: int
    k_max: int
    cosine_floor: float
    fallback_cosine_floor: float
    trust_threshold: float
    allow_trust_weighted: bool
    allow_fallback_best_parent: bool
    allow_fallback_transcript: bool

    @classmethod
    def default(cls) -> "TrustWeightedConsensusPolicy":
        return cls(
            k_min=W55_DEFAULT_TWCC_K_MIN,
            k_max=W55_DEFAULT_TWCC_K_MAX,
            cosine_floor=W55_DEFAULT_TWCC_COSINE_FLOOR,
            fallback_cosine_floor=(
                W55_DEFAULT_TWCC_FALLBACK_COSINE_FLOOR),
            trust_threshold=(
                W55_DEFAULT_TWCC_TRUST_THRESHOLD),
            allow_trust_weighted=True,
            allow_fallback_best_parent=True,
            allow_fallback_transcript=True,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": str(W55_TWCC_SCHEMA_VERSION),
            "k_min": int(self.k_min),
            "k_max": int(self.k_max),
            "cosine_floor": float(round(
                self.cosine_floor, 12)),
            "fallback_cosine_floor": float(round(
                self.fallback_cosine_floor, 12)),
            "trust_threshold": float(round(
                self.trust_threshold, 12)),
            "allow_trust_weighted": bool(
                self.allow_trust_weighted),
            "allow_fallback_best_parent": bool(
                self.allow_fallback_best_parent),
            "allow_fallback_transcript": bool(
                self.allow_fallback_transcript),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w55_twcc_policy",
            "policy": self.to_dict()})


# =============================================================================
# Stage attempt + decision audit entry
# =============================================================================


@dataclasses.dataclass(frozen=True)
class StageAttempt:
    """One stage in the 5-stage decision chain."""

    stage: str
    ok: bool
    payload_l2: float
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": str(self.stage),
            "ok": bool(self.ok),
            "payload_l2": float(round(self.payload_l2, 12)),
            "detail": str(self.detail),
        }


@dataclasses.dataclass(frozen=True)
class TwccAuditEntry:
    """One recorded consensus decision (with 5-stage walk)."""

    policy_cid: str
    n_branches: int
    k_chosen: int
    decision: str
    chosen_stage: str
    quorum_capsule_cid: str
    fallback_branch_id: str
    transcript_payload_cid: str
    parent_cids: tuple[str, ...]
    selected_branch_ids: tuple[str, ...]
    stage_attempts: tuple[StageAttempt, ...]
    reason: str
    turn_index: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_cid": str(self.policy_cid),
            "n_branches": int(self.n_branches),
            "k_chosen": int(self.k_chosen),
            "decision": str(self.decision),
            "chosen_stage": str(self.chosen_stage),
            "quorum_capsule_cid": str(
                self.quorum_capsule_cid),
            "fallback_branch_id": str(
                self.fallback_branch_id),
            "transcript_payload_cid": str(
                self.transcript_payload_cid),
            "parent_cids": list(self.parent_cids),
            "selected_branch_ids": list(
                self.selected_branch_ids),
            "stage_attempts": [
                s.to_dict() for s in self.stage_attempts],
            "reason": str(self.reason),
            "turn_index": int(self.turn_index),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w55_twcc_audit_entry",
            "entry": self.to_dict()})


@dataclasses.dataclass
class TwccAuditTrail:
    entries: list[TwccAuditEntry]

    @classmethod
    def empty(cls) -> "TwccAuditTrail":
        return cls(entries=[])

    def add(self, entry: TwccAuditEntry) -> None:
        self.entries.append(entry)

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w55_twcc_audit_trail",
            "entries": [e.to_dict() for e in self.entries],
        })

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": str(W55_TWCC_SCHEMA_VERSION),
            "n_entries": int(len(self.entries)),
            "audit_cid": str(self.cid()),
        }


# =============================================================================
# DecisionResult
# =============================================================================


@dataclasses.dataclass(frozen=True)
class TwccDecisionResult:
    decision: str
    chosen_stage: str
    payload: tuple[float, ...]
    consensus_capsule: MergeableLatentCapsuleV3 | None
    fallback_branch_id: str
    confidence: float
    aggregate_trust: float
    stage_attempts: tuple[StageAttempt, ...]


# =============================================================================
# Controller
# =============================================================================


def _try_k_of_n_quorum(
        branches: Sequence[MergeableLatentCapsuleV3],
        *,
        operator: MergeOperatorV3,
        audit_trail: MergeAuditTrailV3,
        k_required: int,
        cosine_floor: float,
) -> tuple[
        StageAttempt,
        MergeableLatentCapsuleV3 | None,
        tuple[str, ...]]:
    """W54-style K-of-N quorum on MLSC V3 capsules."""
    n = len(branches)
    if n < int(k_required):
        return StageAttempt(
            stage=W55_TWCC_STAGE_K_OF_N, ok=False,
            payload_l2=0.0,
            detail=f"n_branches={n} < k_required={k_required}",
        ), None, ()
    # Find largest k-clique under cosine_floor.
    # For small N, exhaustive K-subset search is fine.
    best_subset: list[int] = []
    for size in range(int(k_required), n + 1):
        for combo in itertools.combinations(
                range(n), int(size)):
            ok = True
            for a, b in itertools.combinations(combo, 2):
                pa = branches[a].payload
                pb = branches[b].payload
                if _cosine(pa, pb) < float(cosine_floor):
                    ok = False
                    break
            if ok and len(combo) > len(best_subset):
                best_subset = list(combo)
    if len(best_subset) < int(k_required):
        return StageAttempt(
            stage=W55_TWCC_STAGE_K_OF_N, ok=False,
            payload_l2=0.0,
            detail=(
                f"no k-clique at floor={cosine_floor:.3g} "
                f"k_required={k_required}"),
        ), None, ()
    chosen = [branches[i] for i in best_subset]
    merged = merge_capsules_v3(
        operator, chosen,
        audit_trail=audit_trail,
        extra_fact_tags=("k_of_n_quorum",))
    l2 = math.sqrt(sum(
        float(v) ** 2 for v in merged.payload))
    return StageAttempt(
        stage=W55_TWCC_STAGE_K_OF_N, ok=True,
        payload_l2=float(l2),
        detail=(
            f"k_clique_size={len(best_subset)} "
            f"floor={cosine_floor:.3g}"),
    ), merged, tuple(
        str(b.branch_id) for b in chosen)


def _try_trust_weighted_quorum(
        branches: Sequence[MergeableLatentCapsuleV3],
        *,
        operator: MergeOperatorV3,
        audit_trail: MergeAuditTrailV3,
        cosine_floor: float,
        trust_threshold: float,
) -> tuple[
        StageAttempt,
        MergeableLatentCapsuleV3 | None,
        tuple[str, ...]]:
    """Continuous trust-weighted quorum."""
    n = len(branches)
    if n < 2:
        return StageAttempt(
            stage=W55_TWCC_STAGE_TRUST_WEIGHTED, ok=False,
            payload_l2=0.0,
            detail=f"n_branches={n} < 2",
        ), None, ()
    # Find the largest agreeing subset whose trust sum >=
    # trust_threshold. Greedy: sort by trust descending, then
    # incrementally add if cosine to current group mean ≥ floor.
    order = sorted(
        range(n),
        key=lambda i: -float(branches[i].trust))
    selected: list[int] = []
    payload_dim = max(
        len(b.payload) for b in branches) if branches else 0
    current_mean = [0.0] * int(payload_dim)
    current_trust = 0.0
    for idx in order:
        cand = branches[idx]
        if not selected:
            selected.append(idx)
            current_mean = [
                float(cand.payload[j]
                       if j < len(cand.payload) else 0.0)
                for j in range(payload_dim)
            ]
            current_trust = float(cand.trust)
            continue
        # Cosine to current mean
        if _cosine(cand.payload,
                    current_mean) >= float(cosine_floor):
            selected.append(idx)
            current_trust += float(cand.trust)
            for j in range(payload_dim):
                # Trust-weighted running mean update
                w = (
                    float(cand.trust)
                    / float(current_trust))
                current_mean[j] = (
                    (1.0 - w) * current_mean[j]
                    + w * float(
                        cand.payload[j]
                        if j < len(cand.payload) else 0.0))
    if current_trust < float(trust_threshold) or len(
            selected) < 2:
        return StageAttempt(
            stage=W55_TWCC_STAGE_TRUST_WEIGHTED, ok=False,
            payload_l2=0.0,
            detail=(
                f"trust_sum={current_trust:.3g} "
                f"< threshold={trust_threshold:.3g} "
                f"or n_selected={len(selected)} < 2"),
        ), None, ()
    chosen = [branches[i] for i in selected]
    merged = merge_capsules_v3(
        operator, chosen,
        audit_trail=audit_trail,
        extra_fact_tags=("trust_weighted_quorum",))
    l2 = math.sqrt(sum(
        float(v) ** 2 for v in merged.payload))
    return StageAttempt(
        stage=W55_TWCC_STAGE_TRUST_WEIGHTED, ok=True,
        payload_l2=float(l2),
        detail=(
            f"trust_sum={current_trust:.3g} "
            f"n_selected={len(selected)} "
            f"floor={cosine_floor:.3g}"),
    ), merged, tuple(
        str(b.branch_id) for b in chosen)


def _try_fallback_best_parent(
        branches: Sequence[MergeableLatentCapsuleV3],
        *,
        cosine_floor: float,
) -> tuple[
        StageAttempt,
        MergeableLatentCapsuleV3 | None,
        str]:
    """Pick the highest-(trust * confidence) parent above
    cosine_floor to the group mean."""
    if not branches:
        return StageAttempt(
            stage=W55_TWCC_STAGE_FALLBACK_BEST_PARENT,
            ok=False, payload_l2=0.0,
            detail="no_branches",
        ), None, ""
    payload_dim = max(len(b.payload) for b in branches)
    group_mean = [0.0] * payload_dim
    for b in branches:
        for j in range(payload_dim):
            group_mean[j] += float(
                b.payload[j] if j < len(b.payload) else 0.0
            ) / float(len(branches))
    score: list[tuple[float, int]] = []
    for i, b in enumerate(branches):
        cos = _cosine(b.payload, group_mean)
        if cos >= float(cosine_floor):
            sc = float(b.trust) * float(b.confidence)
            score.append((sc, i))
    if not score:
        return StageAttempt(
            stage=W55_TWCC_STAGE_FALLBACK_BEST_PARENT,
            ok=False, payload_l2=0.0,
            detail=(
                f"no parent above cosine_floor="
                f"{cosine_floor:.3g}"),
        ), None, ""
    score.sort(reverse=True)
    _, best_idx = score[0]
    best = branches[best_idx]
    l2 = math.sqrt(sum(
        float(v) ** 2 for v in best.payload))
    return StageAttempt(
        stage=W55_TWCC_STAGE_FALLBACK_BEST_PARENT,
        ok=True, payload_l2=float(l2),
        detail=(
            f"best_branch={best.branch_id} "
            f"score={score[0][0]:.3g}"),
    ), best, str(best.branch_id)


def _try_fallback_transcript(
        *,
        transcript_payload: Sequence[float] | None,
) -> tuple[StageAttempt, str]:
    """Use the externally supplied transcript payload."""
    if transcript_payload is None:
        return StageAttempt(
            stage=W55_TWCC_STAGE_FALLBACK_TRANSCRIPT,
            ok=False, payload_l2=0.0,
            detail="no transcript supplied",
        ), ""
    l2 = math.sqrt(sum(
        float(v) ** 2 for v in transcript_payload))
    cid = _sha256_hex({
        "kind": "w55_twcc_transcript_payload",
        "values": [
            float(round(float(v), 12))
            for v in transcript_payload],
    })
    return StageAttempt(
        stage=W55_TWCC_STAGE_FALLBACK_TRANSCRIPT,
        ok=True, payload_l2=float(l2),
        detail=f"transcript_cid={cid[:16]}",
    ), cid


@dataclasses.dataclass
class TrustWeightedConsensusController:
    """5-stage consensus + fallback controller."""

    policy: TrustWeightedConsensusPolicy
    operator: MergeOperatorV3
    capsule_audit: MergeAuditTrailV3
    controller_audit: TwccAuditTrail

    @classmethod
    def init(
            cls, *,
            policy: TrustWeightedConsensusPolicy | None = None,
            operator: MergeOperatorV3,
            capsule_audit: MergeAuditTrailV3 | None = None,
            controller_audit: TwccAuditTrail | None = None,
    ) -> "TrustWeightedConsensusController":
        return cls(
            policy=(
                policy if policy is not None
                else TrustWeightedConsensusPolicy.default()),
            operator=operator,
            capsule_audit=(
                capsule_audit if capsule_audit is not None
                else MergeAuditTrailV3.empty()),
            controller_audit=(
                controller_audit
                if controller_audit is not None
                else TwccAuditTrail.empty()),
        )

    def decide(
            self,
            branches: Sequence[MergeableLatentCapsuleV3],
            *,
            turn_index: int = 0,
            k_required: int | None = None,
            transcript_payload: Sequence[float] | None = None,
    ) -> tuple[TwccDecisionResult, TwccAuditEntry]:
        n = len(branches)
        k = (
            int(k_required)
            if k_required is not None
            else int(max(
                self.policy.k_min,
                min(n, self.policy.k_max))))
        attempts: list[StageAttempt] = []
        # Stage 1 — K-of-N quorum.
        s1, m_q, sel1 = _try_k_of_n_quorum(
            branches,
            operator=self.operator,
            audit_trail=self.capsule_audit,
            k_required=int(k),
            cosine_floor=float(self.policy.cosine_floor))
        attempts.append(s1)
        if s1.ok and m_q is not None:
            entry = self._record(
                turn_index=int(turn_index),
                k=int(k), n=int(n),
                branches=branches,
                decision=W55_TWCC_DECISION_QUORUM,
                stage=W55_TWCC_STAGE_K_OF_N,
                quorum_cid=str(m_q.cid()),
                fallback_branch="",
                transcript_cid="",
                selected_branch_ids=tuple(sel1),
                stage_attempts=tuple(attempts),
                reason="k_of_n_quorum_succeeded")
            return TwccDecisionResult(
                decision=W55_TWCC_DECISION_QUORUM,
                chosen_stage=W55_TWCC_STAGE_K_OF_N,
                payload=tuple(m_q.payload),
                consensus_capsule=m_q,
                fallback_branch_id="",
                confidence=float(m_q.confidence),
                aggregate_trust=float(m_q.trust),
                stage_attempts=tuple(attempts),
            ), entry
        # Stage 2 — trust-weighted quorum.
        if self.policy.allow_trust_weighted:
            s2, m_t, sel2 = _try_trust_weighted_quorum(
                branches,
                operator=self.operator,
                audit_trail=self.capsule_audit,
                cosine_floor=float(self.policy.cosine_floor),
                trust_threshold=float(
                    self.policy.trust_threshold))
            attempts.append(s2)
            if s2.ok and m_t is not None:
                entry = self._record(
                    turn_index=int(turn_index),
                    k=int(k), n=int(n),
                    branches=branches,
                    decision=W55_TWCC_DECISION_TRUST_WEIGHTED,
                    stage=W55_TWCC_STAGE_TRUST_WEIGHTED,
                    quorum_cid=str(m_t.cid()),
                    fallback_branch="",
                    transcript_cid="",
                    selected_branch_ids=tuple(sel2),
                    stage_attempts=tuple(attempts),
                    reason="trust_weighted_quorum_succeeded")
                return TwccDecisionResult(
                    decision=W55_TWCC_DECISION_TRUST_WEIGHTED,
                    chosen_stage=(
                        W55_TWCC_STAGE_TRUST_WEIGHTED),
                    payload=tuple(m_t.payload),
                    consensus_capsule=m_t,
                    fallback_branch_id="",
                    confidence=float(m_t.confidence),
                    aggregate_trust=float(m_t.trust),
                    stage_attempts=tuple(attempts),
                ), entry
        # Stage 3 — fallback best parent.
        if self.policy.allow_fallback_best_parent:
            s3, best_p, bbid = _try_fallback_best_parent(
                branches,
                cosine_floor=float(
                    self.policy.fallback_cosine_floor))
            attempts.append(s3)
            if s3.ok and best_p is not None:
                entry = self._record(
                    turn_index=int(turn_index),
                    k=int(k), n=int(n),
                    branches=branches,
                    decision=W55_TWCC_DECISION_BEST_PARENT,
                    stage=(
                        W55_TWCC_STAGE_FALLBACK_BEST_PARENT),
                    quorum_cid="",
                    fallback_branch=str(bbid),
                    transcript_cid="",
                    selected_branch_ids=(str(bbid),),
                    stage_attempts=tuple(attempts),
                    reason=(
                        f"fallback_best_parent={bbid}"))
                return TwccDecisionResult(
                    decision=W55_TWCC_DECISION_BEST_PARENT,
                    chosen_stage=(
                        W55_TWCC_STAGE_FALLBACK_BEST_PARENT),
                    payload=tuple(best_p.payload),
                    consensus_capsule=best_p,
                    fallback_branch_id=str(bbid),
                    confidence=float(best_p.confidence),
                    aggregate_trust=float(best_p.trust),
                    stage_attempts=tuple(attempts),
                ), entry
        # Stage 4 — fallback transcript.
        if self.policy.allow_fallback_transcript:
            s4, tcid = _try_fallback_transcript(
                transcript_payload=transcript_payload)
            attempts.append(s4)
            if s4.ok:
                payload = tuple(
                    float(v) for v in transcript_payload)
                entry = self._record(
                    turn_index=int(turn_index),
                    k=int(k), n=int(n),
                    branches=branches,
                    decision=W55_TWCC_DECISION_TRANSCRIPT,
                    stage=(
                        W55_TWCC_STAGE_FALLBACK_TRANSCRIPT),
                    quorum_cid="",
                    fallback_branch="",
                    transcript_cid=str(tcid),
                    selected_branch_ids=(),
                    stage_attempts=tuple(attempts),
                    reason="fallback_transcript_used")
                return TwccDecisionResult(
                    decision=W55_TWCC_DECISION_TRANSCRIPT,
                    chosen_stage=(
                        W55_TWCC_STAGE_FALLBACK_TRANSCRIPT),
                    payload=payload,
                    consensus_capsule=None,
                    fallback_branch_id="",
                    confidence=0.0,
                    aggregate_trust=0.0,
                    stage_attempts=tuple(attempts),
                ), entry
        # Stage 5 — abstain.
        attempts.append(StageAttempt(
            stage=W55_TWCC_STAGE_ABSTAIN, ok=True,
            payload_l2=0.0,
            detail="no_other_stage_succeeded",
        ))
        entry = self._record(
            turn_index=int(turn_index),
            k=int(k), n=int(n),
            branches=branches,
            decision=W55_TWCC_DECISION_ABSTAIN,
            stage=W55_TWCC_STAGE_ABSTAIN,
            quorum_cid="",
            fallback_branch="",
            transcript_cid="",
            selected_branch_ids=(),
            stage_attempts=tuple(attempts),
            reason="abstain_all_stages_exhausted")
        return TwccDecisionResult(
            decision=W55_TWCC_DECISION_ABSTAIN,
            chosen_stage=W55_TWCC_STAGE_ABSTAIN,
            payload=(),
            consensus_capsule=None,
            fallback_branch_id="",
            confidence=0.0,
            aggregate_trust=0.0,
            stage_attempts=tuple(attempts),
        ), entry

    def _record(
            self, *,
            turn_index: int,
            k: int, n: int,
            branches: Sequence[MergeableLatentCapsuleV3],
            decision: str, stage: str,
            quorum_cid: str, fallback_branch: str,
            transcript_cid: str,
            selected_branch_ids: tuple[str, ...],
            stage_attempts: tuple[StageAttempt, ...],
            reason: str,
    ) -> TwccAuditEntry:
        entry = TwccAuditEntry(
            policy_cid=str(self.policy.cid()),
            n_branches=int(n),
            k_chosen=int(k),
            decision=str(decision),
            chosen_stage=str(stage),
            quorum_capsule_cid=str(quorum_cid),
            fallback_branch_id=str(fallback_branch),
            transcript_payload_cid=str(transcript_cid),
            parent_cids=tuple(
                str(b.cid()) for b in branches),
            selected_branch_ids=tuple(selected_branch_ids),
            stage_attempts=stage_attempts,
            reason=str(reason),
            turn_index=int(turn_index),
        )
        self.controller_audit.add(entry)
        return entry

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w55_twcc_controller",
            "policy_cid": str(self.policy.cid()),
            "operator_cid": str(self.operator.cid()),
            "capsule_audit_cid": str(
                self.capsule_audit.cid()),
            "controller_audit_cid": str(
                self.controller_audit.cid()),
        })


# =============================================================================
# Witness
# =============================================================================


@dataclasses.dataclass(frozen=True)
class TwccWitness:
    controller_cid: str
    policy_cid: str
    n_decisions: int
    n_quorum: int
    n_trust_weighted: int
    n_best_parent: int
    n_transcript: int
    n_abstain: int
    quorum_rate: float
    trust_weighted_rate: float
    best_parent_rate: float
    transcript_rate: float
    abstain_rate: float
    capsule_audit_cid: str
    controller_audit_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "controller_cid": str(self.controller_cid),
            "policy_cid": str(self.policy_cid),
            "n_decisions": int(self.n_decisions),
            "n_quorum": int(self.n_quorum),
            "n_trust_weighted": int(self.n_trust_weighted),
            "n_best_parent": int(self.n_best_parent),
            "n_transcript": int(self.n_transcript),
            "n_abstain": int(self.n_abstain),
            "quorum_rate": float(round(
                self.quorum_rate, 12)),
            "trust_weighted_rate": float(round(
                self.trust_weighted_rate, 12)),
            "best_parent_rate": float(round(
                self.best_parent_rate, 12)),
            "transcript_rate": float(round(
                self.transcript_rate, 12)),
            "abstain_rate": float(round(
                self.abstain_rate, 12)),
            "capsule_audit_cid": str(
                self.capsule_audit_cid),
            "controller_audit_cid": str(
                self.controller_audit_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w55_twcc_witness",
            "witness": self.to_dict()})


def emit_twcc_witness(
        controller: TrustWeightedConsensusController,
) -> TwccWitness:
    n = int(len(controller.controller_audit.entries))
    counts = {
        W55_TWCC_DECISION_QUORUM: 0,
        W55_TWCC_DECISION_TRUST_WEIGHTED: 0,
        W55_TWCC_DECISION_BEST_PARENT: 0,
        W55_TWCC_DECISION_TRANSCRIPT: 0,
        W55_TWCC_DECISION_ABSTAIN: 0,
    }
    for e in controller.controller_audit.entries:
        counts[e.decision] = counts.get(e.decision, 0) + 1
    nf = float(max(1, n))
    return TwccWitness(
        controller_cid=str(controller.cid()),
        policy_cid=str(controller.policy.cid()),
        n_decisions=int(n),
        n_quorum=int(counts[W55_TWCC_DECISION_QUORUM]),
        n_trust_weighted=int(
            counts[W55_TWCC_DECISION_TRUST_WEIGHTED]),
        n_best_parent=int(
            counts[W55_TWCC_DECISION_BEST_PARENT]),
        n_transcript=int(counts[W55_TWCC_DECISION_TRANSCRIPT]),
        n_abstain=int(counts[W55_TWCC_DECISION_ABSTAIN]),
        quorum_rate=float(
            counts[W55_TWCC_DECISION_QUORUM]) / nf,
        trust_weighted_rate=float(
            counts[W55_TWCC_DECISION_TRUST_WEIGHTED]) / nf,
        best_parent_rate=float(
            counts[W55_TWCC_DECISION_BEST_PARENT]) / nf,
        transcript_rate=float(
            counts[W55_TWCC_DECISION_TRANSCRIPT]) / nf,
        abstain_rate=float(
            counts[W55_TWCC_DECISION_ABSTAIN]) / nf,
        capsule_audit_cid=str(
            controller.capsule_audit.cid()),
        controller_audit_cid=str(
            controller.controller_audit.cid()),
    )


# =============================================================================
# Verifier
# =============================================================================

W55_TWCC_VERIFIER_FAILURE_MODES: tuple[str, ...] = (
    "w55_twcc_controller_cid_mismatch",
    "w55_twcc_policy_cid_mismatch",
    "w55_twcc_rates_dont_sum_to_one",
    "w55_twcc_audit_walk_orphan",
    "w55_twcc_quorum_rate_below_floor",
)


def verify_twcc_witness(
        witness: TwccWitness,
        *,
        expected_controller_cid: str | None = None,
        expected_policy_cid: str | None = None,
        min_quorum_rate: float | None = None,
) -> dict[str, Any]:
    failures: list[str] = []
    if (expected_controller_cid is not None
            and witness.controller_cid
            != str(expected_controller_cid)):
        failures.append(
            "w55_twcc_controller_cid_mismatch")
    if (expected_policy_cid is not None
            and witness.policy_cid
            != str(expected_policy_cid)):
        failures.append("w55_twcc_policy_cid_mismatch")
    s = (
        float(witness.quorum_rate)
        + float(witness.trust_weighted_rate)
        + float(witness.best_parent_rate)
        + float(witness.transcript_rate)
        + float(witness.abstain_rate))
    if abs(s - 1.0) > 1e-6 and s != 0.0:
        failures.append(
            "w55_twcc_rates_dont_sum_to_one")
    if (min_quorum_rate is not None
            and witness.quorum_rate < float(min_quorum_rate)):
        failures.append(
            "w55_twcc_quorum_rate_below_floor")
    return {
        "ok": (len(failures) == 0),
        "failures": failures,
        "witness_cid": witness.cid(),
    }


__all__ = [
    "W55_TWCC_SCHEMA_VERSION",
    "W55_TWCC_STAGE_K_OF_N",
    "W55_TWCC_STAGE_TRUST_WEIGHTED",
    "W55_TWCC_STAGE_FALLBACK_BEST_PARENT",
    "W55_TWCC_STAGE_FALLBACK_TRANSCRIPT",
    "W55_TWCC_STAGE_ABSTAIN",
    "W55_TWCC_DECISION_QUORUM",
    "W55_TWCC_DECISION_TRUST_WEIGHTED",
    "W55_TWCC_DECISION_BEST_PARENT",
    "W55_TWCC_DECISION_TRANSCRIPT",
    "W55_TWCC_DECISION_ABSTAIN",
    "W55_DEFAULT_TWCC_K_MIN",
    "W55_DEFAULT_TWCC_K_MAX",
    "W55_DEFAULT_TWCC_COSINE_FLOOR",
    "W55_DEFAULT_TWCC_FALLBACK_COSINE_FLOOR",
    "W55_DEFAULT_TWCC_TRUST_THRESHOLD",
    "W55_TWCC_VERIFIER_FAILURE_MODES",
    "TrustWeightedConsensusPolicy",
    "StageAttempt",
    "TwccAuditEntry",
    "TwccAuditTrail",
    "TwccDecisionResult",
    "TrustWeightedConsensusController",
    "TwccWitness",
    "emit_twcc_witness",
    "verify_twcc_witness",
]
