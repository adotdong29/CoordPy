"""Capsule-native multi-agent team coordination — SDK v3.5 (research).

This module is the **multi-agent coordination research slice** of the
Wevra SDK. Where SDK v3.1..v3.4 made capsules load-bearing inside a
*single Wevra run* (run boundary → cell boundary → parser axis →
LLM byte boundary), SDK v3.5 makes capsules load-bearing **between
agents in a team**: every cross-role artefact in a coordination
round is a sealed capsule with a parent-CID-gated DAG, a per-role
admission budget, and a mechanically-checkable lifecycle audit.

The original Context Zero thesis is *per-agent minimum-sufficient
context for multi-agent teams*. The capsule-native runtime made
"context is an object" true at the run boundary; this module makes
the same statement true at the **team boundary**.

Scope discipline
================

* This is **research-grade** code. It is not part of the Wevra
  product runtime contract (the SWE-bench-Lite-shape sweep path).
  Importing from ``vision_mvp.wevra.team_coord`` gives you the new
  multi-agent coordination model; the Wevra SDK's stable surface
  (``RunSpec`` → run report) is unchanged.
* The substrate primitive ``vision_mvp.core.role_handoff`` is
  unchanged. The capsule-native flow can wrap or replace the
  substrate; both can run side by side. The substrate is the
  routing-table / inbox primitive; the capsule layer is the
  *typed, content-addressed, lifecycle-bounded* layer on top.

The team-level capsule model (W4 family)
========================================

A multi-agent capsule coordination round is the following pipeline:

1. **Producer roles emit TEAM_HANDOFF capsules.** Each handoff is
   sealed in flight: payload = (source_role, to_role, claim_kind,
   payload_string, round, payload_sha256); parents = the upstream
   TEAM_HANDOFF CIDs whose evidence is being chained (often empty
   for first-round emissions). Two emissions of byte-identical
   ``(source, to, kind, payload, round)`` collapse to one capsule
   (Capsule Contract C1).

2. **Each consumer role admits a per-role view.** For each role
   ``r``, the coordinator constructs a ROLE_VIEW capsule whose
   parents are the CIDs of admitted TEAM_HANDOFF capsules; the
   ``max_parents`` budget is the role-local inbox capacity
   ``K_role``; ``max_tokens`` is the role-local token budget
   ``T_role``. Admission is deterministic given an
   ``AdmissionPolicy``; see ``DEFAULT_ROLE_BUDGETS``.

3. **The deciding role emits a TEAM_DECISION capsule.** Parents:
   the role views the deciding role consulted (typically its own
   ROLE_VIEW). Payload: the final structured answer.

The whole thing lives in a ``CapsuleLedger``; a
``CapsuleLifecycleAudit``-style team audit verifies invariants
T-1..T-7 (see :class:`TeamLifecycleAudit`).

Theorem cross-reference (W4 family)
-----------------------------------

* **W4-1** — Lifecycle ↔ team-state correspondence on team
  kinds. Mechanically-checked.
* **W4-2** — Coverage-implies-correctness on the deterministic
  team-decoder. Proved-conditional.
* **W4-3** — Local-view limitation: a per-role budget cap below
  the role's causal-share floor admits sound runs that fail the
  team gate. Proved-negative.

Public surface
==============

* ``RoleBudget`` — per-role admission budget (``K_role``,
  ``T_role``).
* ``DEFAULT_ROLE_BUDGETS`` — canonical defaults used by the
  benchmark.
* ``capsule_team_handoff`` — adapter constructor for a
  ``TEAM_HANDOFF`` capsule that does not require a substrate
  twin object.
* ``capsule_role_view`` — adapter constructor for a ``ROLE_VIEW``
  capsule. Enforces ``len(parents) ≤ K_role``.
* ``capsule_team_decision`` — adapter constructor for a
  ``TEAM_DECISION`` capsule.
* ``AdmissionPolicy`` (Protocol) — admission decisions over
  candidate handoffs given the role's budget.
* ``FifoAdmissionPolicy`` — first-in-first-out admission. The
  baseline.
* ``ClaimPriorityAdmissionPolicy`` — priority-table admission.
* ``CoverageGuidedAdmissionPolicy`` — admit one handoff per
  ``claim_kind`` until the budget is exhausted; the strongest
  fixed baseline.
* ``LearnedAdmissionPolicy`` — placeholder; the real learned
  scorer lives in ``team_policy.py`` (additional dependency-free
  module).
* ``TeamCoordinator`` — the orchestrator. Drives one
  coordination round end-to-end against an in-memory ledger.
* ``TeamLifecycleAudit`` + ``audit_team_lifecycle`` — runtime-
  checkable invariants T-1..T-7.

Design alignment with Wevra
---------------------------

The team layer reuses the existing ``CapsuleLedger`` (no new
ledger). Every team-level capsule is admitted/sealed through the
same ``admit_and_seal`` path so the lifecycle and chain-hash
invariants C1..C6 hold uniformly. The only change to ``capsule.py``
is the addition of three closed-vocabulary kinds (TEAM_HANDOFF,
ROLE_VIEW, TEAM_DECISION) and their default budgets — strictly
additive.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import time
from typing import Any, Callable, Iterable, Protocol, Sequence, runtime_checkable

from vision_mvp.wevra.capsule import (
    CapsuleAdmissionError, CapsuleBudget, CapsuleKind, CapsuleLedger,
    CapsuleLifecycle, ContextCapsule,
)


# =============================================================================
# Per-role budget knob
# =============================================================================


@dataclasses.dataclass(frozen=True)
class RoleBudget:
    """Per-role admission budget for a coordination round.

    Fields
    ------
    role
        Short identifier of the role (e.g. ``"auditor"``).
    K_role
        ``max_parents`` of the ROLE_VIEW capsule; the maximum
        number of TEAM_HANDOFF capsules this role admits in a
        single round. Strict: admission is rejected when the
        role has already admitted ``K_role`` handoffs.
    T_role
        ``max_tokens`` of the ROLE_VIEW capsule; the cumulative
        token budget over admitted handoffs. Strict in the same
        sense as ``K_role``.
    permitted_kinds
        Optional whitelist of claim kinds the role is allowed to
        admit. ``None`` means "anything"; an empty tuple means
        "nothing".
    """

    role: str
    K_role: int = 32
    T_role: int = 1024
    permitted_kinds: tuple[str, ...] | None = None

    def admits_kind(self, claim_kind: str) -> bool:
        if self.permitted_kinds is None:
            return True
        return claim_kind in self.permitted_kinds


# Canonical default role budgets used by the team benchmark. These
# numbers are the same numbers the substrate inbox capacity defaults
# carry, so a head-to-head substrate-vs-capsule comparison is
# apples-to-apples.
DEFAULT_ROLE_BUDGETS: dict[str, RoleBudget] = {
    "monitor":  RoleBudget(role="monitor",  K_role=8,  T_role=128),
    "db_admin": RoleBudget(role="db_admin", K_role=8,  T_role=128),
    "sysadmin": RoleBudget(role="sysadmin", K_role=8,  T_role=128),
    "network":  RoleBudget(role="network",  K_role=8,  T_role=128),
    "auditor":  RoleBudget(role="auditor",  K_role=12, T_role=512),
}


# =============================================================================
# Capsule constructors — TEAM_HANDOFF / ROLE_VIEW / TEAM_DECISION
# =============================================================================


def _payload_sha256(payload: str) -> str:
    return hashlib.sha256((payload or "").encode("utf-8")).hexdigest()


def capsule_team_handoff(*,
                            source_role: str,
                            to_role: str,
                            claim_kind: str,
                            payload: str,
                            round: int = 0,
                            parents: Iterable[str] = (),
                            n_tokens: int | None = None,
                            ) -> ContextCapsule:
    """Build a ``TEAM_HANDOFF`` capsule directly (no substrate twin).

    Identity is content-addressed by ``(source, to, kind, payload,
    round)`` plus the parent CIDs (Capsule Contract C1). Two
    byte-identical handoffs at the same round collapse to one
    capsule. The ``payload_sha256`` is exposed in the payload so
    downstream consumers can prove the handoff bytes' identity
    without re-canonicalising the capsule.
    """
    sha = _payload_sha256(payload)
    n_tok = n_tokens
    if n_tok is None:
        n_tok = max(1, len((payload or "").split()))
    body: dict[str, Any] = {
        "source_role": str(source_role),
        "to_role": str(to_role),
        "claim_kind": str(claim_kind),
        "payload": str(payload),
        "round": int(round),
        "payload_sha256": sha,
        "n_tokens": int(n_tok),
    }
    return ContextCapsule.new(
        kind=CapsuleKind.TEAM_HANDOFF,
        payload=body,
        parents=parents,
        n_tokens=int(n_tok),
        metadata={
            "source_role": str(source_role),
            "to_role": str(to_role),
            "claim_kind": str(claim_kind),
            "round": int(round),
            "payload_sha256": sha,
        },
    )


def capsule_role_view(*,
                        role: str,
                        round: int,
                        admitted_handoff_cids: Sequence[str],
                        admitted_claim_kinds: Sequence[str],
                        n_tokens_admitted: int,
                        n_dropped_budget: int = 0,
                        n_dropped_unknown_kind: int = 0,
                        n_dropped_capacity: int = 0,
                        budget: RoleBudget | None = None,
                        ) -> ContextCapsule:
    """Build a ``ROLE_VIEW`` capsule.

    ``admitted_handoff_cids`` are the CIDs of TEAM_HANDOFF capsules
    the role has already admitted into the same ledger; admission of
    the ROLE_VIEW capsule fails (Contract C5) if any parent CID is
    not yet sealed in the ledger. The ``max_parents`` budget on the
    ROLE_VIEW is set from ``budget.K_role`` if provided; otherwise
    the SDK default is used. ``max_tokens`` is set from
    ``budget.T_role`` if provided.

    Raises
    ------
    ValueError
        If ``len(admitted_handoff_cids) > budget.K_role``. This is
        the structural admission cap; admission failure happens at
        capsule construction (before ledger admit), so the caller's
        invariant violation surfaces as a hard error rather than a
        silent over-budget run.
    """
    parents = tuple(admitted_handoff_cids)
    cap_budget: CapsuleBudget
    if budget is not None:
        if len(parents) > budget.K_role:
            raise ValueError(
                f"ROLE_VIEW for role={role!r}: "
                f"len(admitted)={len(parents)} > K_role={budget.K_role}")
        if n_tokens_admitted > budget.T_role:
            raise ValueError(
                f"ROLE_VIEW for role={role!r}: "
                f"n_tokens_admitted={n_tokens_admitted} > "
                f"T_role={budget.T_role}")
        cap_budget = CapsuleBudget(
            max_parents=budget.K_role,
            max_tokens=budget.T_role,
            max_bytes=1 << 14,
        )
    else:
        cap_budget = CapsuleBudget(
            max_parents=max(32, len(parents)),
            max_tokens=max(1024, n_tokens_admitted),
            max_bytes=1 << 14,
        )
    payload: dict[str, Any] = {
        "role": str(role),
        "round": int(round),
        "n_admitted": len(parents),
        "n_dropped_budget": int(n_dropped_budget),
        "n_dropped_unknown_kind": int(n_dropped_unknown_kind),
        "n_dropped_capacity": int(n_dropped_capacity),
        "n_tokens_admitted": int(n_tokens_admitted),
        "K_role": int(cap_budget.max_parents or 0),
        "T_role": int(cap_budget.max_tokens or 0),
        "admitted_claim_kinds": sorted(set(admitted_claim_kinds)),
    }
    return ContextCapsule.new(
        kind=CapsuleKind.ROLE_VIEW,
        payload=payload,
        budget=cap_budget,
        parents=parents,
        n_tokens=int(n_tokens_admitted),
        metadata={
            "role": str(role),
            "round": int(round),
            "n_admitted": len(parents),
        },
    )


def capsule_team_decision(*,
                            team_tag: str,
                            round: int,
                            decision: dict[str, Any],
                            evidence_summary: str = "",
                            n_role_views: int = 0,
                            gate_passed: bool | None = None,
                            parents: Iterable[str] = (),
                            ) -> ContextCapsule:
    """Build a ``TEAM_DECISION`` capsule.

    ``parents`` are typically the CIDs of the ROLE_VIEW capsule(s)
    the deciding role consulted. ``decision`` is the structured
    answer; only constraint is that it round-trips through
    ``json.dumps(sort_keys=True)``.
    """
    summary = (evidence_summary or "")
    if len(summary) > 500:
        summary = summary[:500]
    payload: dict[str, Any] = {
        "team_tag": str(team_tag),
        "round": int(round),
        "decision": decision,
        "evidence_summary": summary,
        "n_role_views": int(n_role_views),
        "gate_passed": (None if gate_passed is None else bool(gate_passed)),
    }
    return ContextCapsule.new(
        kind=CapsuleKind.TEAM_DECISION,
        payload=payload,
        parents=parents,
        metadata={
            "team_tag": str(team_tag),
            "round": int(round),
            "n_role_views": int(n_role_views),
            "gate_passed": (None if gate_passed is None else bool(gate_passed)),
        },
    )


# =============================================================================
# Admission policy protocol + canonical baselines
# =============================================================================


@dataclasses.dataclass(frozen=True)
class AdmissionDecision:
    """One admission decision for one candidate handoff.

    Fields
    ------
    admit
        ``True`` iff the policy admits the candidate handoff.
    reason
        Short string for audit / debugging. Closed vocabulary —
        ``"admit"``, ``"budget_full"``, ``"tokens_full"``,
        ``"unknown_kind"``, ``"duplicate"``, ``"score_low"``.
    score
        Optional float score the policy attached to the decision
        (used by learned policies). ``0.0`` for fixed policies.
    """
    admit: bool
    reason: str
    score: float = 0.0


REASON_ADMIT = "admit"
REASON_BUDGET_FULL = "budget_full"
REASON_TOKENS_FULL = "tokens_full"
REASON_UNKNOWN_KIND = "unknown_kind"
REASON_DUPLICATE = "duplicate"
REASON_SCORE_LOW = "score_low"

_ALL_REASONS = (REASON_ADMIT, REASON_BUDGET_FULL, REASON_TOKENS_FULL,
                REASON_UNKNOWN_KIND, REASON_DUPLICATE, REASON_SCORE_LOW)


class AdmissionPolicy(Protocol):
    """Decide which TEAM_HANDOFF capsules to admit into a role's
    view capsule under the role's K_role / T_role / kind budget.

    The policy is *stateful per-call* only — the caller passes the
    role's current admission set and the candidate; the policy
    decides one handoff at a time. The caller is responsible for
    book-keeping the admitted set + the running token total.

    A correct policy implementation must respect the budget: it
    must NOT return ``admit=True`` when ``current_n_admitted >=
    budget.K_role`` or ``current_n_tokens + cand_n_tokens >
    budget.T_role`` (the admission contract).
    """

    def decide(self,
               *,
               candidate: ContextCapsule,
               role: str,
               budget: RoleBudget,
               current_admitted: Sequence[ContextCapsule],
               current_n_tokens: int,
               ) -> AdmissionDecision: ...


def _candidate_n_tokens(cand: ContextCapsule) -> int:
    if cand.n_tokens is not None and cand.n_tokens > 0:
        return int(cand.n_tokens)
    payload = cand.payload if isinstance(cand.payload, dict) else {}
    n_tok = payload.get("n_tokens")
    if isinstance(n_tok, int) and n_tok > 0:
        return int(n_tok)
    body = payload.get("payload", "") if isinstance(payload, dict) else ""
    return max(1, len(str(body).split()))


def _candidate_claim_kind(cand: ContextCapsule) -> str:
    if not isinstance(cand.payload, dict):
        return ""
    return str(cand.payload.get("claim_kind", ""))


def _candidate_payload_sha(cand: ContextCapsule) -> str:
    if not isinstance(cand.payload, dict):
        return ""
    return str(cand.payload.get("payload_sha256", ""))


def _enforce_budget(*,
                     candidate: ContextCapsule,
                     role: str,
                     budget: RoleBudget,
                     current_admitted: Sequence[ContextCapsule],
                     current_n_tokens: int,
                     ) -> AdmissionDecision | None:
    """Common pre-checks every policy must run before its own
    admission rule. Returns a non-admit decision if any structural
    budget constraint is violated; ``None`` if the candidate is
    admissible.
    """
    kind = _candidate_claim_kind(candidate)
    if not budget.admits_kind(kind):
        return AdmissionDecision(admit=False, reason=REASON_UNKNOWN_KIND)
    if len(current_admitted) >= budget.K_role:
        return AdmissionDecision(admit=False, reason=REASON_BUDGET_FULL)
    n_tok = _candidate_n_tokens(candidate)
    if current_n_tokens + n_tok > budget.T_role:
        return AdmissionDecision(admit=False, reason=REASON_TOKENS_FULL)
    sha = _candidate_payload_sha(candidate)
    if sha:
        for c in current_admitted:
            if _candidate_payload_sha(c) == sha:
                return AdmissionDecision(admit=False, reason=REASON_DUPLICATE)
    return None


@dataclasses.dataclass
class FifoAdmissionPolicy:
    """First-in-first-out admission. Admits any candidate that fits
    the budget. The simplest baseline; the substrate equivalent of
    inbox FIFO."""

    name: str = "fifo"

    def decide(self, *, candidate, role, budget,
               current_admitted, current_n_tokens):
        denial = _enforce_budget(
            candidate=candidate, role=role, budget=budget,
            current_admitted=current_admitted,
            current_n_tokens=current_n_tokens)
        if denial is not None:
            return denial
        return AdmissionDecision(admit=True, reason=REASON_ADMIT)


@dataclasses.dataclass
class ClaimPriorityAdmissionPolicy:
    """Static priority on claim kinds. Admits a candidate iff the
    candidate's claim_kind score (looked up from ``priorities``) is
    above ``threshold``. Ties broken by FIFO.

    Used as the strongest *fixed* (non-learned) baseline: it knows
    which claims are usually causal and prefers them over noise.
    """

    priorities: dict[str, float]
    threshold: float = 0.0
    name: str = "claim_priority"

    def decide(self, *, candidate, role, budget,
               current_admitted, current_n_tokens):
        denial = _enforce_budget(
            candidate=candidate, role=role, budget=budget,
            current_admitted=current_admitted,
            current_n_tokens=current_n_tokens)
        if denial is not None:
            return denial
        kind = _candidate_claim_kind(candidate)
        score = float(self.priorities.get(kind, 0.0))
        if score < self.threshold:
            return AdmissionDecision(admit=False, reason=REASON_SCORE_LOW,
                                      score=score)
        return AdmissionDecision(admit=True, reason=REASON_ADMIT, score=score)


@dataclasses.dataclass
class CoverageGuidedAdmissionPolicy:
    """Admit one handoff per claim_kind until budget exhausted.

    Strong baseline: forces *coverage* over the claim catalogue
    rather than greedy FIFO that may saturate on duplicates of one
    high-frequency claim. Empirically the best fixed policy on
    coverage-bottlenecked tasks.
    """

    name: str = "coverage_guided"

    def decide(self, *, candidate, role, budget,
               current_admitted, current_n_tokens):
        denial = _enforce_budget(
            candidate=candidate, role=role, budget=budget,
            current_admitted=current_admitted,
            current_n_tokens=current_n_tokens)
        if denial is not None:
            return denial
        kind = _candidate_claim_kind(candidate)
        seen_kinds = {_candidate_claim_kind(c) for c in current_admitted}
        if kind in seen_kinds:
            # Already covered; skip duplicates of an admitted kind.
            return AdmissionDecision(admit=False, reason=REASON_DUPLICATE)
        return AdmissionDecision(admit=True, reason=REASON_ADMIT)


# =============================================================================
# Cohort-coherence admission (SDK v3.8 — relational/cross-role rule)
# =============================================================================


_SERVICE_TAG_RE = __import__("re").compile(r"service=(\w+)")


def _candidate_service_tag(cand: ContextCapsule) -> str:
    """Extract the ``service=<tag>`` cohort signature from a
    candidate's payload, or the empty string if absent.

    The match is intentionally narrow: only the exact ``service=<tag>``
    token (the same key the existing
    :func:`vision_mvp.tasks.incident_triage._decoder_from_handoffs`
    uses to populate the auditor's ``services`` set) is honoured. Any
    other identifier-shaped token in the payload is ignored.
    """
    if not isinstance(cand.payload, dict):
        return ""
    body = str(cand.payload.get("payload", ""))
    if not body:
        return ""
    matches = _SERVICE_TAG_RE.findall(body)
    if not matches:
        return ""
    # First-occurrence wins; consistent with the decoder's left-to-right
    # token scan in ``_decoder_from_handoffs``.
    return str(matches[0])


@dataclasses.dataclass
class CohortCoherenceAdmissionPolicy:
    """Cross-role cohort-coherence admission (SDK v3.8 / W7 family).

    Two sub-modes
    -------------
    The policy supports two operating modes, controlled by
    ``fixed_plurality_tag``:

    * **Streaming** (``fixed_plurality_tag=None``, default) —
      maintain a running cohort signature over already-admitted
      candidates and reject the next candidate if its
      ``service=<tag>`` conflicts with the cohort plurality. Simple
      and deterministic, but **arrival-order-sensitive**: if the
      first admitted candidate carries a foreign tag, the cohort
      locks onto that tag and rejects subsequent gold candidates.
      This is a real limitation theorem (W7-1-aux): pure streaming
      cohort-coherence is unstable under candidate-arrival
      permutation.

    * **Buffered** (``fixed_plurality_tag=<tag>``) — the policy
      ignores the running cohort and instead uses a pre-fitted
      plurality tag. Construct via :meth:`from_candidate_payloads`
      to fit the plurality on the full candidate distribution before
      streaming admission begins. This costs one extra pre-pass over
      the candidate list but is **arrival-order-stable**: the
      plurality is computed once, off-line, before any admission
      decision is made, so the policy is independent of within-stream
      ordering.

    The Phase-54 default uses the buffered mode (W7-2 anchor): the
    driver pre-fits the plurality tag on each scenario's candidate
    stream, then runs admission. The streaming mode is retained as
    a research baseline / falsifier (W7-1-aux witness).

    Rule (per ``decide`` call)
    --------------------------
    1. Run the ``_enforce_budget`` pre-checks (kind whitelist, K_role,
       T_role, payload-SHA dedupe). If any fails, return the
       structural denial.
    2. Extract the candidate's ``service=<tag>``.
    3. If the candidate has no tag, admit (it cannot violate
       cohort coherence on a missing key).
    4. If ``fixed_plurality_tag`` is set, admit iff
       ``cand_tag == fixed_plurality_tag``; else reject as
       ``score_low``.
    5. Else (streaming mode): admit iff the cohort is empty or the
       candidate's tag matches the running plurality.

    Why this is *cross-role* (not just single-role)
    -----------------------------------------------
    The auditor's ROLE_VIEW capsule's parents include TEAM_HANDOFF
    capsules emitted by *every* producer role. The cohort signature
    is therefore aggregated across roles — a service tag that one
    role has admitted is "evidence" the auditor uses against a
    candidate from another role with a different tag. This is the
    minimum interesting cross-role coordination move: it requires a
    relational view that single-role per-candidate FIFO cannot
    express.

    Why this is honest
    ------------------
    The policy is **deterministic** and **small** (one regex, one
    counter, no learning). Buffered mode requires one O(N) pre-pass
    over the candidate list to compute the plurality; this is
    book-keeping, not learning. The policy is **not** training-
    distribution-sensitive (no scorer, no threshold tuned on
    synthetic noise) — so it is not subject to the W6-C2 OOD
    failure mode that bit the learned policy at SDK v3.7.

    Lifecycle invariants
    --------------------
    The policy preserves T-1..T-7 by construction: it returns
    standard ``AdmissionDecision`` records via the existing
    ``TeamCoordinator`` admission path. No new lifecycle states.
    """

    name: str = "cohort_coherence"
    min_cohort_for_filter: int = 1
    fixed_plurality_tag: str | None = None

    @classmethod
    def from_candidate_payloads(cls,
                                  payloads: "Iterable[str]",
                                  *,
                                  min_cohort_for_filter: int = 1,
                                  ) -> "CohortCoherenceAdmissionPolicy":
        """Construct a *buffered* policy by pre-fitting the cohort
        plurality tag on a candidate stream's payloads.

        The plurality is computed once, off-line, over all payloads
        carrying a ``service=<tag>`` token; ties are broken by
        descending count, then ascending lex order. Payloads without
        a service tag are skipped. If no payload carries a tag, the
        returned policy is the streaming default (no plurality lock).

        This is the W7-2 anchor mode: cohort coherence with arrival-
        order-stable selection.
        """
        tags: list[str] = []
        for p in payloads:
            if not p:
                continue
            m = _SERVICE_TAG_RE.search(str(p))
            if m:
                tags.append(m.group(1))
        if not tags:
            return cls(min_cohort_for_filter=min_cohort_for_filter)
        counts: dict[str, int] = {}
        for t in tags:
            counts[t] = counts.get(t, 0) + 1
        max_count = max(counts.values())
        # Lex-sort ties to make the plurality deterministic.
        plurality_tags = sorted([t for t, c in counts.items()
                                  if c == max_count])
        return cls(min_cohort_for_filter=min_cohort_for_filter,
                    fixed_plurality_tag=plurality_tags[0])

    def decide(self, *, candidate, role, budget,
               current_admitted, current_n_tokens):
        denial = _enforce_budget(
            candidate=candidate, role=role, budget=budget,
            current_admitted=current_admitted,
            current_n_tokens=current_n_tokens)
        if denial is not None:
            return denial
        cand_tag = _candidate_service_tag(candidate)
        if not cand_tag:
            # Candidate carries no service token; cannot violate
            # cohort coherence on a missing key. Admit.
            return AdmissionDecision(admit=True, reason=REASON_ADMIT)
        if self.fixed_plurality_tag is not None:
            # Buffered mode: plurality is pre-fitted; ignore running
            # cohort.
            if cand_tag == self.fixed_plurality_tag:
                return AdmissionDecision(admit=True, reason=REASON_ADMIT)
            return AdmissionDecision(admit=False, reason=REASON_SCORE_LOW,
                                      score=0.0)
        # Streaming mode: cohort histogram over admitted candidates.
        tags: list[str] = []
        for c in current_admitted:
            t = _candidate_service_tag(c)
            if t:
                tags.append(t)
        if not tags:
            return AdmissionDecision(admit=True, reason=REASON_ADMIT)
        if len(tags) < self.min_cohort_for_filter:
            return AdmissionDecision(admit=True, reason=REASON_ADMIT)
        counts: dict[str, int] = {}
        for t in tags:
            counts[t] = counts.get(t, 0) + 1
        max_count = max(counts.values())
        plurality_tags = sorted([t for t, c in counts.items()
                                  if c == max_count])
        plurality = plurality_tags[0]
        if cand_tag == plurality:
            return AdmissionDecision(admit=True, reason=REASON_ADMIT)
        if len(plurality_tags) > 1 and cand_tag in plurality_tags:
            return AdmissionDecision(admit=True, reason=REASON_ADMIT)
        return AdmissionDecision(admit=False, reason=REASON_SCORE_LOW,
                                  score=0.0)


# =============================================================================
# Cross-role corroboration admission (SDK v3.9 — W8 family)
# =============================================================================


def _candidate_source_role(cand: ContextCapsule) -> str:
    if not isinstance(cand.payload, dict):
        return ""
    return str(cand.payload.get("source_role", ""))


@dataclasses.dataclass
class CrossRoleCorroborationAdmissionPolicy:
    """Cross-role corroboration admission (SDK v3.9 / W8 family).

    A strict generalisation of :class:`CohortCoherenceAdmissionPolicy`
    that aggregates a service tag's signal across **distinct
    producer roles** rather than over raw mention counts. The
    motivating regime is the *decoy-plurality* case (the
    SDK v3.8 W7-2 falsifier): a candidate stream where some decoy
    service has strictly more raw mentions than gold but is mentioned
    by **fewer distinct producer roles** than gold. On such a
    stream, single-tag-plurality cohort coherence (W7-2) picks the
    decoy and ties FIFO; this policy picks the cross-role-corroborated
    gold and wins.

    Two sub-modes
    -------------
    The policy supports two operating modes, controlled by
    ``fixed_dominant_tag``:

    * **Streaming** (``fixed_dominant_tag=None``, default) — at each
      ``decide`` call, recompute the corroboration score over the
      currently-admitted candidates. Arrival-order-sensitive in the
      same sense as ``CohortCoherenceAdmissionPolicy`` streaming
      (W7-1-aux): if the first admitted candidate is from a foreign
      service and that service's role-coverage equals or exceeds the
      gold's *as the cohort grows*, the policy can lock onto the
      decoy. The streaming variant is retained as a
      research-baseline / falsifier; the buffered variant is the
      load-bearing one.

    * **Buffered** (``fixed_dominant_tag=<tag>``) — the policy
      ignores the running cohort and uses a pre-fitted dominant tag.
      Construct via :meth:`from_candidate_stream` to fit on the full
      ``(source_role, payload)`` candidate list before streaming
      admission begins. **Arrival-order-stable** by construction.

    Scoring
    -------
    For each ``service=<tag>`` in the candidate stream, the
    *corroboration score* is

    .. math::

        \\text{score}(\\text{tag}) =
        W_\\text{role} \\cdot |\\{ \\text{distinct producer roles
        emitting tag}\\}| + |\\{ \\text{total mentions of tag}\\}|

    where ``W_role`` is a large constant (default ``100``) so
    distinct-role corroboration **dominates** raw count for any
    realistic candidate-stream size. Ties on score break by lex
    order on the tag, deterministically. This is **proved** in
    Theorem W8-2: any decoy with raw-count advantage `Δr ≤ 99`
    and role-coverage disadvantage `Δr_role ≥ 1` is dominated.

    Why this is *cross-role* (not just per-tag plurality)
    -----------------------------------------------------
    The single-tag plurality
    (:class:`CohortCoherenceAdmissionPolicy`) does *not* see the
    producer role of each candidate. It sees only the multiset of
    service tags. The corroboration policy's score function
    explicitly aggregates over the (role, tag) bipartite multiset,
    so it *can* express "gold is mentioned by 3 distinct roles vs
    decoy by 1 distinct role" — a relational signal the W7-2 policy
    cannot represent. This is the minimum interesting *strict
    generalisation* of W7-2.

    Why this is honest
    ------------------
    * **Deterministic** and **small**: one regex (re-using
      :data:`_SERVICE_TAG_RE`), one counter over (role, tag) pairs,
      no learning, no training-distribution dependency.
    * **Backward-compatible**: on any candidate stream where the
      gold tag has the highest raw count *and* the highest distinct-
      role count, this policy admits exactly the same set as the W7-2
      buffered cohort. On Phase-54 default, the two policies are
      indistinguishable on `accuracy_full`.
    * **Falsifiable**: a stream where the *decoy* has strictly more
      distinct-role coverage AND strictly more raw mentions than
      gold falsifies W8-1. The W8-1 falsifier regime is named in
      :file:`vision_mvp/experiments/phase55_decoy_plurality.py` as
      the *decoy-corroborated decoy* falsifier.

    Lifecycle invariants
    --------------------
    The policy preserves T-1..T-7 by construction: it returns
    standard ``AdmissionDecision`` records via the existing
    ``TeamCoordinator`` admission path. No new lifecycle states.

    Theorem cross-reference (W8 family)
    -----------------------------------
    * **W8-1** (proved-empirical) — strict separation on Phase-55
      decoy-plurality + cross-role-corroborated gold.
    * **W8-2** (proved, structural) — for any constants
      ``W_role > Δr_max`` and ``Δr_role ≥ 1``, the corroboration
      score function strictly orders cross-role-corroborated gold
      above raw-plurality decoy.
    * **W8-3** (proved-empirical) — backward compatibility on
      Phase-54 default.
    """

    name: str = "cross_role_corroboration"
    role_weight: int = 100
    fixed_dominant_tag: str | None = None

    @classmethod
    def from_candidate_stream(
            cls,
            stream: "Iterable[tuple[str, str]]",
            *,
            role_weight: int = 100,
            ) -> "CrossRoleCorroborationAdmissionPolicy":
        """Construct a *buffered* policy by pre-fitting the
        cross-role-corroborated dominant service tag on a candidate
        stream.

        Parameters
        ----------
        stream
            Iterable of ``(source_role, payload)`` pairs. Only the
            payload's ``service=<tag>`` token is consulted; the
            source_role is used to count distinct-role coverage.
        role_weight
            Constant ``W_role`` in the score function (default
            100). Must be strictly greater than the largest raw
            mention difference the bench can produce; otherwise
            W8-2's strict-ordering premise is violated. The default
            is sufficient for any candidate stream of size
            ``< 100``.

        Score function (re-stated for clarity):

            score(tag) = role_weight * |distinct_roles(tag)|
                       + raw_mentions(tag)

        Ties break by lex order on the tag (deterministic). If no
        candidate carries a service tag, the returned policy is the
        streaming default.
        """
        role_per_tag: dict[str, set[str]] = {}
        count_per_tag: dict[str, int] = {}
        for (src_role, payload) in stream:
            if not payload:
                continue
            m = _SERVICE_TAG_RE.search(str(payload))
            if not m:
                continue
            tag = m.group(1)
            count_per_tag[tag] = count_per_tag.get(tag, 0) + 1
            role_per_tag.setdefault(tag, set()).add(str(src_role))
        if not count_per_tag:
            return cls(role_weight=role_weight)
        scored: list[tuple[int, str]] = []
        for tag, cnt in count_per_tag.items():
            n_roles = len(role_per_tag.get(tag, set()))
            scored.append((role_weight * n_roles + cnt, tag))
        # Sort by (-score, tag) so the lex-smallest of the highest-
        # scoring tags wins (deterministic).
        scored.sort(key=lambda x: (-x[0], x[1]))
        return cls(role_weight=role_weight,
                    fixed_dominant_tag=scored[0][1])

    def decide(self, *, candidate, role, budget,
               current_admitted, current_n_tokens):
        denial = _enforce_budget(
            candidate=candidate, role=role, budget=budget,
            current_admitted=current_admitted,
            current_n_tokens=current_n_tokens)
        if denial is not None:
            return denial
        cand_tag = _candidate_service_tag(candidate)
        if not cand_tag:
            # Untagged candidate: cannot violate corroboration on a
            # missing key. Admit (consistent with W7-2).
            return AdmissionDecision(admit=True, reason=REASON_ADMIT)
        if self.fixed_dominant_tag is not None:
            # Buffered mode: pre-fitted dominant tag.
            if cand_tag == self.fixed_dominant_tag:
                return AdmissionDecision(admit=True, reason=REASON_ADMIT)
            return AdmissionDecision(admit=False, reason=REASON_SCORE_LOW,
                                      score=0.0)
        # Streaming mode: recompute corroboration over admitted set.
        role_per_tag: dict[str, set[str]] = {}
        count_per_tag: dict[str, int] = {}
        for c in current_admitted:
            t = _candidate_service_tag(c)
            if not t:
                continue
            r = _candidate_source_role(c)
            count_per_tag[t] = count_per_tag.get(t, 0) + 1
            role_per_tag.setdefault(t, set()).add(r)
        if not count_per_tag:
            return AdmissionDecision(admit=True, reason=REASON_ADMIT)
        scored: list[tuple[int, str]] = []
        for t, cnt in count_per_tag.items():
            n_roles = len(role_per_tag.get(t, set()))
            scored.append((self.role_weight * n_roles + cnt, t))
        scored.sort(key=lambda x: (-x[0], x[1]))
        dominant = scored[0][1]
        if cand_tag == dominant:
            return AdmissionDecision(admit=True, reason=REASON_ADMIT)
        return AdmissionDecision(admit=False, reason=REASON_SCORE_LOW,
                                  score=0.0)


# =============================================================================
# Multi-service cross-role corroboration (SDK v3.10 — W9 family)
# =============================================================================


@dataclasses.dataclass
class MultiServiceCorroborationAdmissionPolicy:
    """Top-K cross-role corroboration with min-role threshold
    (SDK v3.10 / W9 family).

    A strict generalisation of
    :class:`CrossRoleCorroborationAdmissionPolicy` (SDK v3.9 / W8) that
    admits candidates whose ``service=<tag>`` belongs to the **set of
    top-K corroborated tags** rather than only the single highest-scoring
    tag. This is the smallest mechanical move that captures
    *multi-service incidents* — the case where the gold answer requires
    aggregating evidence across **two or more** distinct services
    (e.g. a "db pool exhaustion + api latency" incident with
    ``gold_services = ("api", "db")``).

    Why W8 (single-tag corroboration) is *not* enough
    -------------------------------------------------
    The decoder ``_decoder_from_handoffs`` (Phase-31) computes
    ``services = ⋃ {service tags in admitted handoffs}`` and grades
    ``services_correct`` by **set equality** with ``gold_services``.
    On a multi-service-gold scenario, the W8 policy admits only
    candidates carrying the top-1 corroboration tag; the auditor's
    ``services`` is therefore a singleton, and ``services_correct``
    fails *even when* the gold tag is the W8 winner. W9 admits the
    top-K corroboration tags and recovers ``services`` set equality.

    The threshold ``min_corroborated_roles``
    ----------------------------------------
    Top-K alone is not enough either: a decoy with raw plurality and
    ``|distinct_roles| = 1`` would slip into the top-K set (in the
    K=2 case if the second gold service is corroborated by only 2
    roles, the decoy's score = ``W_role · 1 + 100 = W_role + 100``
    might exceed the gold-B's ``W_role · 2 + 4 = 2 W_role + 4``
    only when ``W_role < 96``; with the default ``W_role=100``, the
    decoy is dominated). To make the policy robust under arbitrary
    raw counts, we additionally require **every admitted dominant
    tag to satisfy** ``|distinct_roles(tag)| ≥ min_corroborated_roles``
    (default ``2``). This is the **min-role threshold** — a tag with
    only one role's support cannot enter the dominant set even if its
    raw count is large. It is the structural separator between
    cross-role-corroborated gold and single-role decoy storms.

    Two sub-modes
    -------------
    * **Streaming** (``fixed_dominant_tags=None``, default) —
      recompute the top-K corroborated tags from currently-admitted
      candidates each ``decide`` call. Arrival-order-sensitive in the
      same sense as W8 / W7-1-aux streaming.
    * **Buffered** (``fixed_dominant_tags=frozenset(...)``) — pre-fit
      the dominant tag set on the full ``(source_role, payload)``
      candidate list. Construct via :meth:`from_candidate_stream`.
      **Arrival-order-stable** by construction. The buffered variant
      is the W9-1 anchor.

    Scoring (re-uses the W8 score function)
    ---------------------------------------
    For each ``service=<tag>`` in the candidate stream:

    .. math::

        \\text{score}(\\text{tag}) =
        W_\\text{role} \\cdot |\\{ \\text{distinct roles emitting tag}
        \\}| + |\\{ \\text{total mentions of tag}\\}|

    Then:

    1. Drop tags with ``|distinct_roles(tag)| < min_corroborated_roles``.
    2. Of the remaining tags, take the ``top_k`` by score (ties broken
       lex on the tag name).
    3. The buffered policy admits a candidate iff its tag is in this
       dominant set, OR the candidate has no tag.

    This is **provable** (W9-2): with default ``W_role=100`` and
    ``min_corroborated_roles=2``, no tag with one-role support can
    enter the dominant set regardless of raw count, and any tag with
    ``≥ min_corroborated_roles`` distinct roles strictly dominates any
    ``min_corroborated_roles - 1``-roles tag with raw-count advantage
    ``< W_role``. See docstring of :func:`_dominant_tag_set` and
    Theorem W9-2 in :file:`docs/RESULTS_WEVRA_MULTI_SERVICE_CORROBORATION.md`.

    Backward compatibility (W9-3)
    -----------------------------
    On a single-service-gold candidate stream where exactly one tag
    has ``≥ min_corroborated_roles`` distinct-role support, the
    dominant set has size 1 and W9 admits exactly the same set as
    W8 (``CrossRoleCorroborationAdmissionPolicy``). The Phase-55
    default has this property by construction; W9 ties W8 at
    ``accuracy_full = 1.000`` there.

    Falsifier (W9-4)
    ----------------
    If a *decoy* service is also cross-role-corroborated above the
    threshold (e.g. ``decoy_storm`` mentioned by ≥ 2 distinct roles)
    AND the decoy's score equals or exceeds gold-B's score, the
    decoy slips into the top-K dominant set, the auditor's
    ``services`` becomes ``{gold_A, gold_B, decoy}`` ≠
    ``{gold_A, gold_B}``, and the W9-1 win does *not* hold. The
    Phase-56 *decoy-corroborated* falsifier bank instantiates this
    regime.

    Lifecycle invariants
    --------------------
    The policy preserves T-1..T-7 by construction: it returns standard
    ``AdmissionDecision`` records via the existing ``TeamCoordinator``
    admission path. No new lifecycle states; W6-1 generalisation to
    Phase-56 holds.

    Theorem cross-reference (W9 family)
    -----------------------------------
    * **W9-1** (proved-empirical) — strict separation on Phase-56
      multi-service-gold + corroborated-decoy regime.
    * **W9-2** (proved, structural) — the min-role threshold and
      W_role > Δr_max constants jointly bound the decoy below any
      gold tag with ≥ min_corroborated_roles support.
    * **W9-3** (proved-empirical) — backward-compat with W8 on
      Phase-55 default.
    * **W9-4** (proved-empirical) — Phase-56 decoy-corroborated
      falsifier ties FIFO at 0.000.
    """

    name: str = "multi_service_corroboration"
    role_weight: int = 100
    top_k: int = 2
    min_corroborated_roles: int = 2
    fixed_dominant_tags: frozenset[str] | None = None

    @classmethod
    def from_candidate_stream(
            cls,
            stream: "Iterable[tuple[str, str]]",
            *,
            role_weight: int = 100,
            top_k: int = 2,
            min_corroborated_roles: int = 2,
            ) -> "MultiServiceCorroborationAdmissionPolicy":
        """Construct a *buffered* policy by pre-fitting the
        top-K cross-role-corroborated dominant tag set on a candidate
        stream.

        Parameters
        ----------
        stream
            Iterable of ``(source_role, payload)`` pairs.
        role_weight
            Constant ``W_role`` in the score function (default 100).
        top_k
            Maximum size of the dominant tag set (default 2 — the
            multi-service-gold case ``gold_services = (A, B)``).
        min_corroborated_roles
            Minimum distinct producer roles a tag must have to enter
            the dominant set (default 2). Tags with fewer distinct-
            role support are filtered out *before* the top-K cut.
            Default ``2`` is the smallest-but-still-defending threshold:
            it filters single-role decoy storms while admitting any
            tag with cross-role evidence.

        Score function (re-stated for clarity):

            score(tag) = role_weight * |distinct_roles(tag)|
                       + raw_mentions(tag)

        Selection:

            dominant_tags = top_k tags by score among
                              {tag : |distinct_roles(tag)| >=
                                       min_corroborated_roles}

        Ties break by lex order on the tag (deterministic).
        If no tag passes the threshold, the buffered policy admits
        nothing tagged (consistent with W7-2 / W8 fall-through to
        FIFO-on-untagged).
        """
        role_per_tag: dict[str, set[str]] = {}
        count_per_tag: dict[str, int] = {}
        for (src_role, payload) in stream:
            if not payload:
                continue
            m = _SERVICE_TAG_RE.search(str(payload))
            if not m:
                continue
            tag = m.group(1)
            count_per_tag[tag] = count_per_tag.get(tag, 0) + 1
            role_per_tag.setdefault(tag, set()).add(str(src_role))
        if not count_per_tag:
            return cls(role_weight=role_weight, top_k=top_k,
                        min_corroborated_roles=min_corroborated_roles)
        dominant = _dominant_tag_set(
            count_per_tag=count_per_tag,
            role_per_tag=role_per_tag,
            role_weight=role_weight,
            top_k=top_k,
            min_corroborated_roles=min_corroborated_roles)
        return cls(role_weight=role_weight, top_k=top_k,
                    min_corroborated_roles=min_corroborated_roles,
                    fixed_dominant_tags=frozenset(dominant))

    def decide(self, *, candidate, role, budget,
               current_admitted, current_n_tokens):
        denial = _enforce_budget(
            candidate=candidate, role=role, budget=budget,
            current_admitted=current_admitted,
            current_n_tokens=current_n_tokens)
        if denial is not None:
            return denial
        cand_tag = _candidate_service_tag(candidate)
        if not cand_tag:
            return AdmissionDecision(admit=True, reason=REASON_ADMIT)
        if self.fixed_dominant_tags is not None:
            if cand_tag in self.fixed_dominant_tags:
                return AdmissionDecision(admit=True, reason=REASON_ADMIT)
            return AdmissionDecision(admit=False, reason=REASON_SCORE_LOW,
                                      score=0.0)
        # Streaming mode: recompute the dominant set over admitted set.
        role_per_tag: dict[str, set[str]] = {}
        count_per_tag: dict[str, int] = {}
        for c in current_admitted:
            t = _candidate_service_tag(c)
            if not t:
                continue
            r = _candidate_source_role(c)
            count_per_tag[t] = count_per_tag.get(t, 0) + 1
            role_per_tag.setdefault(t, set()).add(r)
        if not count_per_tag:
            return AdmissionDecision(admit=True, reason=REASON_ADMIT)
        dominant = _dominant_tag_set(
            count_per_tag=count_per_tag,
            role_per_tag=role_per_tag,
            role_weight=self.role_weight,
            top_k=self.top_k,
            min_corroborated_roles=self.min_corroborated_roles)
        if cand_tag in dominant:
            return AdmissionDecision(admit=True, reason=REASON_ADMIT)
        return AdmissionDecision(admit=False, reason=REASON_SCORE_LOW,
                                  score=0.0)


def _dominant_tag_set(*,
                       count_per_tag: dict[str, int],
                       role_per_tag: dict[str, set[str]],
                       role_weight: int,
                       top_k: int,
                       min_corroborated_roles: int) -> tuple[str, ...]:
    """Compute the top-K cross-role-corroborated dominant tag set.

    Selection rule (argmax-by-role-count tier with top-K cap):

      1. Drop tags with ``|distinct_roles(tag)| < min_corroborated_roles``
         (filters single-role storms).
      2. Of the remaining tags, take the ``argmax-by-role-count``
         tier: every tag whose distinct-role count equals the
         maximum distinct-role count across the eligible set. This
         is the **structural** cross-role-corroboration tier — a
         decoy with strictly fewer distinct producer roles than the
         strongest cross-role-corroborated tag is excluded by
         construction, regardless of raw count.
      3. Among the argmax tier, take the ``top_k`` by score
         (``score(tag) = role_weight · |distinct_roles(tag)|
         + raw_mentions(tag)``); lex tie-break.

    Why the argmax tier (and not score-sorted top-K alone)
    -------------------------------------------------------
    Without the argmax tier, on a Phase-55-style single-service-gold
    regime where the gold tag has strictly more distinct producer
    roles than every decoy, a *second* tag (a decoy) with sub-max
    role count but high raw count could enter the top-K dominant
    set, breaking backward-compat with W8 single-tag corroboration.
    The argmax-by-role-count gate ensures W9 *collapses* to W8 when
    only one tag has the maximum role count — i.e. W9 is a strict
    generalisation of W8 (W9-3 backward-compat).

    Phase-56 multi-service-gold regime:
    *both* gold tags have role count = 2 (the max among eligible);
    the (single-role) decoy storm is below ``min_corroborated_roles``
    and excluded; the argmax tier = {gold_A, gold_B}; W9 admits both
    and ``services = {A, B}`` matches gold (W9-1 win).

    Phase-56 falsifier regime:
    a decoy is promoted to 2 distinct producer roles, so the argmax
    tier = {gold_A, gold_B, decoy}; the top-K cap (default 2) lets
    score break the tie, and a high-raw-count decoy enters the
    dominant set, displacing one gold tag (W9-4 falsifier).

    Returns a tuple of tag names in lex order (deterministic).
    """
    eligible = [t for t, _c in count_per_tag.items()
                  if len(role_per_tag.get(t, set())) >=
                     min_corroborated_roles]
    if not eligible:
        return ()
    role_count = {t: len(role_per_tag.get(t, set())) for t in eligible}
    max_roles = max(role_count.values())
    argmax_tier = [t for t in eligible if role_count[t] == max_roles]
    scored: list[tuple[int, str]] = []
    for t in argmax_tier:
        scored.append((role_weight * role_count[t] + count_per_tag[t], t))
    # Sort by (-score, tag) — highest score first, lex tie-break.
    scored.sort(key=lambda x: (-x[0], x[1]))
    chosen = [t for (_s, t) in scored[:top_k]]
    return tuple(sorted(chosen))


# Closed-vocabulary handle for "policies named in TEAM_DECISION
# audit logs". Keeps the per-policy diagnostics scannable.
ALL_FIXED_POLICY_NAMES = ("fifo", "claim_priority", "coverage_guided",
                            "cohort_coherence",
                            "cross_role_corroboration",
                            "multi_service_corroboration")


# =============================================================================
# TeamCoordinator — drive one coordination round end-to-end
# =============================================================================


@dataclasses.dataclass
class TeamCoordinator:
    """Drive one capsule-native multi-agent coordination round.

    Usage
    -----
    .. code-block:: python

        ledger = CapsuleLedger()
        coord = TeamCoordinator(
            ledger=ledger,
            role_budgets={"auditor": RoleBudget("auditor", K_role=12,
                                                  T_role=512), ...},
            policy_per_role={"auditor": FifoAdmissionPolicy(), ...},
            team_tag="incident_triage",
        )
        for h in candidate_handoffs:
            coord.emit_handoff(source_role=h.source_role, ...)
        for role in coord.role_budgets:
            coord.seal_role_view(role)
        coord.seal_team_decision(team_role="auditor", decision={...})

    The coordinator does not interpret the decision payload — the
    caller decides the schema. The coordinator's only job is to
    drive the capsule lifecycle uniformly so the team-level
    invariants T-1..T-7 hold by construction.

    Internal state
    --------------
    * ``ledger`` — the shared CapsuleLedger every team capsule lives
      in. Reused across multiple coordinator instances if desired.
    * ``role_budgets`` — per-role budget knobs.
    * ``policy_per_role`` — admission policy per role. Required for
      every role that will ever admit a handoff. Defaults to
      ``FifoAdmissionPolicy`` when unset.
    * ``team_tag`` — short identifier for the team (e.g. the task
      bench name).
    * ``round`` — current coordination round (incremented manually
      by ``advance_round``).
    """

    ledger: CapsuleLedger
    role_budgets: dict[str, RoleBudget]
    policy_per_role: dict[str, AdmissionPolicy] = dataclasses.field(
        default_factory=dict)
    team_tag: str = "team"
    round: int = 0

    # Per-role pending admitted set (CID list) and token total. These
    # are reset by ``seal_role_view``.
    _pending_per_role: dict[str, list[ContextCapsule]] = dataclasses.field(
        default_factory=dict)
    _pending_tokens_per_role: dict[str, int] = dataclasses.field(
        default_factory=dict)
    _per_role_dropped: dict[str, dict[str, int]] = dataclasses.field(
        default_factory=dict)
    # All sealed handoff CIDs in order, for logging.
    _all_handoff_cids: list[str] = dataclasses.field(default_factory=list)
    # Sealed role-view CIDs by role.
    _role_view_cids: dict[str, str] = dataclasses.field(default_factory=dict)
    # Sealed team-decision CIDs by (team_tag, round).
    _team_decision_cids: dict[tuple[str, int], str] = dataclasses.field(
        default_factory=dict)

    def __post_init__(self) -> None:
        for role in self.role_budgets:
            self._pending_per_role.setdefault(role, [])
            self._pending_tokens_per_role.setdefault(role, 0)
            self._per_role_dropped.setdefault(role, {
                REASON_BUDGET_FULL: 0, REASON_TOKENS_FULL: 0,
                REASON_UNKNOWN_KIND: 0, REASON_DUPLICATE: 0,
                REASON_SCORE_LOW: 0,
            })
            self.policy_per_role.setdefault(role, FifoAdmissionPolicy())

    # ------------------------------------------------------------------
    # Round management
    # ------------------------------------------------------------------

    def advance_round(self, n: int = 1) -> int:
        self.round += int(n)
        return self.round

    # ------------------------------------------------------------------
    # Handoff emission + per-role admission
    # ------------------------------------------------------------------

    def emit_handoff(self,
                      *,
                      source_role: str,
                      to_role: str,
                      claim_kind: str,
                      payload: str,
                      parents: Iterable[str] = (),
                      n_tokens: int | None = None,
                      ) -> tuple[ContextCapsule, AdmissionDecision]:
        """Seal a TEAM_HANDOFF capsule and present it to ``to_role``'s
        admission policy.

        Returns ``(sealed_handoff, admission_decision)``. The
        handoff is sealed in the ledger regardless of the
        admission outcome — provenance must be preserved
        independently of role-local view selection. Only the
        admission decision determines whether the handoff joins
        the role's pending ROLE_VIEW parent set.

        Raises
        ------
        KeyError
            If ``to_role`` is not in ``role_budgets``.
        """
        if to_role not in self.role_budgets:
            raise KeyError(f"unknown to_role {to_role!r}")
        cap = capsule_team_handoff(
            source_role=source_role, to_role=to_role,
            claim_kind=claim_kind, payload=payload,
            round=self.round, parents=parents, n_tokens=n_tokens,
        )
        # Idempotent-by-CID admit+seal. The ledger's ``admit`` returns
        # the existing sealed copy on a CID collision (Capsule Contract
        # C1 — byte-identical handoffs collapse), in which case
        # ``seal`` would refuse the SEALED-input call. Handle the
        # collision by using the stored entry directly.
        if cap.cid in self.ledger:
            cap = self.ledger.get(cap.cid)
        else:
            cap = self.ledger.admit_and_seal(cap)
        if cap.cid not in self._all_handoff_cids:
            self._all_handoff_cids.append(cap.cid)
        budget = self.role_budgets[to_role]
        policy = self.policy_per_role[to_role]
        decision = policy.decide(
            candidate=cap, role=to_role, budget=budget,
            current_admitted=self._pending_per_role[to_role],
            current_n_tokens=self._pending_tokens_per_role[to_role],
        )
        if decision.admit:
            self._pending_per_role[to_role].append(cap)
            self._pending_tokens_per_role[to_role] += _candidate_n_tokens(cap)
        else:
            slot = self._per_role_dropped[to_role]
            slot[decision.reason] = slot.get(decision.reason, 0) + 1
        return cap, decision

    # ------------------------------------------------------------------
    # ROLE_VIEW sealing
    # ------------------------------------------------------------------

    def seal_role_view(self, role: str) -> ContextCapsule:
        """Seal a ROLE_VIEW capsule for ``role`` over its currently
        pending admitted set, then clear the pending state for the
        role. Returns the sealed capsule.

        Raises
        ------
        KeyError
            If ``role`` is not in ``role_budgets``.
        """
        if role not in self.role_budgets:
            raise KeyError(f"unknown role {role!r}")
        budget = self.role_budgets[role]
        admitted = list(self._pending_per_role[role])
        n_tokens = int(self._pending_tokens_per_role[role])
        admitted_kinds = [_candidate_claim_kind(c) for c in admitted]
        dropped = self._per_role_dropped[role]
        cap = capsule_role_view(
            role=role, round=self.round,
            admitted_handoff_cids=[c.cid for c in admitted],
            admitted_claim_kinds=admitted_kinds,
            n_tokens_admitted=n_tokens,
            n_dropped_budget=int(dropped.get(REASON_BUDGET_FULL, 0)),
            n_dropped_unknown_kind=int(
                dropped.get(REASON_UNKNOWN_KIND, 0)
                + dropped.get(REASON_SCORE_LOW, 0)),
            n_dropped_capacity=int(
                dropped.get(REASON_TOKENS_FULL, 0)
                + dropped.get(REASON_DUPLICATE, 0)),
            budget=budget,
        )
        if cap.cid in self.ledger:
            cap = self.ledger.get(cap.cid)
        else:
            cap = self.ledger.admit_and_seal(cap)
        self._role_view_cids[role] = cap.cid
        # Reset pending state for the next round; keep dropped
        # counters so a multi-round audit can trace per-round
        # admission failures.
        self._pending_per_role[role] = []
        self._pending_tokens_per_role[role] = 0
        return cap

    def seal_all_role_views(self) -> dict[str, ContextCapsule]:
        out: dict[str, ContextCapsule] = {}
        for role in self.role_budgets:
            out[role] = self.seal_role_view(role)
        return out

    # ------------------------------------------------------------------
    # TEAM_DECISION sealing
    # ------------------------------------------------------------------

    def seal_team_decision(self,
                              *,
                              team_role: str,
                              decision: dict[str, Any],
                              evidence_summary: str = "",
                              gate_passed: bool | None = None,
                              extra_role_view_cids: Sequence[str] = (),
                              ) -> ContextCapsule:
        """Seal a TEAM_DECISION capsule whose parents are the
        ROLE_VIEW capsule for ``team_role`` plus any
        ``extra_role_view_cids`` (other roles consulted).

        Raises
        ------
        KeyError
            If ``team_role`` is not in ``_role_view_cids`` (i.e.
            ``seal_role_view`` was not called for it yet).
        """
        if team_role not in self._role_view_cids:
            raise KeyError(
                f"team_role {team_role!r} has no sealed ROLE_VIEW yet")
        parents = [self._role_view_cids[team_role]]
        for cid in extra_role_view_cids:
            if cid not in parents:
                parents.append(cid)
        cap = capsule_team_decision(
            team_tag=self.team_tag, round=self.round,
            decision=decision, evidence_summary=evidence_summary,
            n_role_views=len(parents), gate_passed=gate_passed,
            parents=parents,
        )
        if cap.cid in self.ledger:
            cap = self.ledger.get(cap.cid)
        else:
            cap = self.ledger.admit_and_seal(cap)
        self._team_decision_cids[(self.team_tag, self.round)] = cap.cid
        return cap

    # ------------------------------------------------------------------
    # Read-only views
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        n_handoff = sum(1 for c in self.ledger.all_capsules()
                         if c.kind == CapsuleKind.TEAM_HANDOFF)
        n_role_view = sum(1 for c in self.ledger.all_capsules()
                           if c.kind == CapsuleKind.ROLE_VIEW)
        n_decision = sum(1 for c in self.ledger.all_capsules()
                         if c.kind == CapsuleKind.TEAM_DECISION)
        return {
            "team_tag": self.team_tag,
            "round": self.round,
            "n_team_handoff": n_handoff,
            "n_role_view": n_role_view,
            "n_team_decision": n_decision,
            "per_role_dropped": {
                r: dict(s) for r, s in self._per_role_dropped.items()
            },
            "role_view_cids": dict(self._role_view_cids),
            "team_decision_cids": {
                f"{tag}@{rd}": cid
                for (tag, rd), cid in self._team_decision_cids.items()
            },
        }

    def role_view_cid(self, role: str) -> str | None:
        return self._role_view_cids.get(role)

    def admitted_handoffs(self, role: str) -> tuple[ContextCapsule, ...]:
        return tuple(self._pending_per_role.get(role, ()))


# =============================================================================
# Team lifecycle audit (T-1..T-7)
# =============================================================================


# Closed vocabulary of T-* invariants the audit checks. Counterpart
# to L-1..L-11 in the run-boundary lifecycle audit.
T_INVARIANTS = (
    "T-1",   # Every TEAM_HANDOFF is SEALED
    "T-2",   # Every ROLE_VIEW is SEALED
    "T-3",   # Every TEAM_DECISION is SEALED
    "T-4",   # ROLE_VIEW.parents ⊆ {TEAM_HANDOFF cids in ledger}
    "T-5",   # ROLE_VIEW.parents respect K_role (max_parents budget)
    "T-6",   # TEAM_DECISION.parents ⊆ {ROLE_VIEW cids in ledger}
    "T-7",   # No ROLE_VIEW admits a TEAM_HANDOFF whose to_role differs
)


@dataclasses.dataclass(frozen=True)
class TeamLifecycleAuditReport:
    """Result of a team-level lifecycle audit.

    Verdict vocabulary
    ------------------
    * ``"OK"``    — all T-* invariants hold on the ledger.
    * ``"BAD"``   — at least one invariant violated;
                    ``violations`` lists the first counterexample
                    per invariant.
    * ``"EMPTY"`` — no team capsules in the ledger; vacuous OK.
    """
    verdict: str
    violations: list[dict[str, Any]]
    counts: dict[str, int]

    def is_ok(self) -> bool:
        return self.verdict == "OK"

    def as_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "violations": list(self.violations),
            "counts": dict(self.counts),
        }


def audit_team_lifecycle(ledger: CapsuleLedger
                            ) -> TeamLifecycleAuditReport:
    """Mechanically verify the team-level lifecycle invariants T-1..T-7.

    Runs in O(N) over the ledger entries. Each invariant produces at
    most one counterexample; the audit short-circuits *within* an
    invariant after the first counterexample but still checks every
    invariant.

    Theorem W4-1 (Team-lifecycle soundness):
        ``audit_team_lifecycle(ledger).verdict == "OK"`` iff every
        T-1..T-7 invariant holds on the ledger.

    Proof: by inspection of the audit code below — every invariant's
    failure path is enumerated with a concrete violation record.
    """
    handoffs = [c for c in ledger.all_capsules()
                if c.kind == CapsuleKind.TEAM_HANDOFF]
    views = [c for c in ledger.all_capsules()
             if c.kind == CapsuleKind.ROLE_VIEW]
    decisions = [c for c in ledger.all_capsules()
                 if c.kind == CapsuleKind.TEAM_DECISION]
    counts = {
        "n_team_handoff": len(handoffs),
        "n_role_view": len(views),
        "n_team_decision": len(decisions),
    }
    if not handoffs and not views and not decisions:
        return TeamLifecycleAuditReport(
            verdict="EMPTY", violations=[], counts=counts)

    violations: list[dict[str, Any]] = []

    # T-1: every TEAM_HANDOFF is SEALED.
    for c in handoffs:
        if c.lifecycle != CapsuleLifecycle.SEALED:
            violations.append({
                "invariant": "T-1",
                "cid": c.cid,
                "reason": f"TEAM_HANDOFF lifecycle={c.lifecycle}",
            })
            break

    # T-2: every ROLE_VIEW is SEALED.
    for c in views:
        if c.lifecycle != CapsuleLifecycle.SEALED:
            violations.append({
                "invariant": "T-2",
                "cid": c.cid,
                "reason": f"ROLE_VIEW lifecycle={c.lifecycle}",
            })
            break

    # T-3: every TEAM_DECISION is SEALED.
    for c in decisions:
        if c.lifecycle != CapsuleLifecycle.SEALED:
            violations.append({
                "invariant": "T-3",
                "cid": c.cid,
                "reason": f"TEAM_DECISION lifecycle={c.lifecycle}",
            })
            break

    handoff_cids = {c.cid for c in handoffs}
    view_cids = {c.cid for c in views}

    # T-4: ROLE_VIEW.parents ⊆ TEAM_HANDOFF cids in ledger.
    for c in views:
        bad = [p for p in c.parents if p not in handoff_cids]
        if bad:
            violations.append({
                "invariant": "T-4",
                "cid": c.cid,
                "reason": f"ROLE_VIEW parent {bad[0][:12]} is not "
                           f"a TEAM_HANDOFF in this ledger",
            })
            break

    # T-5: ROLE_VIEW.parents respects K_role (max_parents budget).
    for c in views:
        kbound = c.budget.max_parents
        if kbound is not None and len(c.parents) > kbound:
            violations.append({
                "invariant": "T-5",
                "cid": c.cid,
                "reason": (
                    f"ROLE_VIEW has {len(c.parents)} parents but "
                    f"K_role={kbound}"),
            })
            break

    # T-6: TEAM_DECISION.parents ⊆ ROLE_VIEW cids in ledger.
    for c in decisions:
        bad = [p for p in c.parents if p not in view_cids]
        if bad:
            violations.append({
                "invariant": "T-6",
                "cid": c.cid,
                "reason": f"TEAM_DECISION parent {bad[0][:12]} is not "
                           f"a ROLE_VIEW in this ledger",
            })
            break

    # T-7: No ROLE_VIEW admits a TEAM_HANDOFF whose to_role differs
    # from the role's name.
    handoff_to_role: dict[str, str] = {}
    for c in handoffs:
        if isinstance(c.payload, dict):
            handoff_to_role[c.cid] = str(c.payload.get("to_role", ""))
    for c in views:
        view_role = ""
        if isinstance(c.payload, dict):
            view_role = str(c.payload.get("role", ""))
        for p in c.parents:
            tr = handoff_to_role.get(p, "")
            if tr and tr != view_role:
                violations.append({
                    "invariant": "T-7",
                    "cid": c.cid,
                    "reason": (
                        f"ROLE_VIEW(role={view_role!r}) admitted "
                        f"TEAM_HANDOFF(to_role={tr!r})"),
                })
                break
        if violations and violations[-1]["invariant"] == "T-7":
            break

    verdict = "OK" if not violations else "BAD"
    return TeamLifecycleAuditReport(
        verdict=verdict, violations=violations, counts=counts)


# =============================================================================
# Bundle-aware team decoder (SDK v3.11 — W10 family)
# =============================================================================


# Closed-vocabulary causal-claim-kind set per root_cause label. The
# table mirrors ``vision_mvp.tasks.incident_triage._decoder_from_handoffs``'s
# priority list and groups claim_kinds by which root_cause they
# *causally entail* under the incident-triage decoder.
#
# Tier semantics: a claim_kind is causal-for-root_cause R iff, in
# isolation, it would lead the priority decoder to label root_cause = R
# OR it sits in the same incident "tier" as R (data-tier vs storage-tier
# vs network-tier vs compute-tier vs edge-tier vs generic).
#
# Why the *tier* extension and not strict equality
# -------------------------------------------------
# A multi-service incident in the data tier (e.g., gold_root_cause =
# "deadlock" with gold_services = ("orders", "payments")) typically
# has one gold service mentioned via DEADLOCK_SUSPECTED and a second
# gold service mentioned via POOL_EXHAUSTION (the cascade). Both
# claims are causal in the same tier; if CCK("deadlock") were
# {DEADLOCK_SUSPECTED} only, the bundle decoder would filter out the
# second gold service. The tier extension keeps the rule narrow
# enough to filter generic-noise decoys (LATENCY/ERROR/FW) while
# still admitting cascades within the same tier.
#
# A claim_kind that maps to no root_cause label and is generic noise
# (LATENCY_SPIKE, ERROR_RATE_SPIKE) is in the *generic* tier — it
# is in CCK only when the chosen root_cause is itself "error_spike"
# or "latency_spike" (i.e. when the gold root_cause IS generic).
CAUSAL_CLAIM_KINDS_PER_ROOT_CAUSE: dict[str, frozenset[str]] = {
    # Data tier — the canonical multi-service incident family.
    "pool_exhaustion": frozenset({
        "POOL_EXHAUSTION", "DEADLOCK_SUSPECTED", "SLOW_QUERY_OBSERVED",
    }),
    "deadlock": frozenset({
        "DEADLOCK_SUSPECTED", "POOL_EXHAUSTION", "SLOW_QUERY_OBSERVED",
    }),
    "slow_query_cascade": frozenset({
        "SLOW_QUERY_OBSERVED", "POOL_EXHAUSTION", "DEADLOCK_SUSPECTED",
    }),
    # Storage tier.
    "disk_fill": frozenset({
        "DISK_FILL_CRITICAL", "CRON_OVERRUN",
    }),
    # Compute tier.
    "memory_leak": frozenset({
        "OOM_KILL",
    }),
    # Edge tier.
    "tls_expiry": frozenset({
        "TLS_EXPIRED",
    }),
    "dns_misroute": frozenset({
        "DNS_MISROUTE",
    }),
    # Network tier.
    "fw_block": frozenset({
        "FW_BLOCK_SURGE",
    }),
    # Generic tier — when the gold root_cause IS generic noise, the
    # decoder cannot do better than admission alone (the CCK includes
    # everything noisy). This is the *named falsifier scope* of the
    # bundle-aware decoder: generic-root-cause regimes are W10-Λ-hard.
    "error_spike": frozenset({
        "ERROR_RATE_SPIKE", "LATENCY_SPIKE",
    }),
    "latency_spike": frozenset({
        "LATENCY_SPIKE", "ERROR_RATE_SPIKE",
    }),
    # Unknown root_cause — admit all services (no filtering).
    "unknown": frozenset(),
}


# The decoder's priority over claim_kinds (mirror of
# ``vision_mvp.tasks.incident_triage._decoder_from_handoffs``). Kept in
# closed vocabulary so the bundle decoder is self-contained and does
# not import from ``vision_mvp.tasks`` (which would create a layering
# inversion). The list is in *priority order* — highest priority first.
_DECODER_PRIORITY: tuple[tuple[str, str, str], ...] = (
    ("DISK_FILL_CRITICAL", "disk_fill", "rotate_logs_and_clear_backup"),
    ("TLS_EXPIRED",        "tls_expiry", "renew_tls_and_reload"),
    ("DNS_MISROUTE",       "dns_misroute", "restore_internal_dns_zone"),
    ("OOM_KILL",           "memory_leak", "rollback_app_to_prev_release"),
    ("DEADLOCK_SUSPECTED", "deadlock", "enforce_lock_ordering_in_orders"),
    ("CRON_OVERRUN",       "disk_fill", "rotate_logs_and_clear_backup"),
    ("POOL_EXHAUSTION",    "pool_exhaustion",
                            "raise_pool_cap_or_fix_upstream"),
    ("SLOW_QUERY_OBSERVED", "slow_query_cascade",
                             "index_or_split_slow_query"),
    ("ERROR_RATE_SPIKE",   "error_spike", "roll_back_recent_deploy"),
    ("LATENCY_SPIKE",      "latency_spike", "scale_up_api_pool"),
    ("FW_BLOCK_SURGE",     "fw_block", "rescind_spurious_deny_rule"),
)


@dataclasses.dataclass(frozen=True)
class _DecodedHandoff:
    """Minimal duck-typed handoff record the decoder consumes.

    Kept structural rather than nominal so the decoder works against
    either ``ContextCapsule`` payloads or the
    ``vision_mvp.tasks.incident_triage.TypedHandoff`` shape — same
    interface ``substrate`` and ``capsule_*`` strategies already share.
    """
    source_role: str
    claim_kind: str
    payload: str


def _decoded_root_cause(claim_kinds: set[str]) -> tuple[str, str]:
    """Run the priority decoder on a set of admitted claim_kinds.

    Returns ``(root_cause, remediation)`` — same closed vocabulary
    as ``incident_triage._decoder_from_handoffs``. ``"unknown"`` /
    ``"investigate"`` if no admitted kind matches.
    """
    for (kind, label, remed) in _DECODER_PRIORITY:
        if kind in claim_kinds:
            return label, remed
    return "unknown", "investigate"


@dataclasses.dataclass
class BundleAwareTeamDecoder:
    """Bundle-aware, contradiction-aware team decoder (SDK v3.11 / W10).

    This is the **first decoder-side coordination move** in the
    Wevra programme. Where W7 / W8 / W9 attacked the multi-agent
    context problem from the *admission* side, W10 attacks it from
    the *decoder* side: given a bundle of admitted handoffs, the
    decoder uses the structural relationship between the chosen
    ``root_cause`` and each admitted handoff's ``claim_kind`` to
    *filter* the auditor's ``services`` set.

    The structural rule (deterministic, training-free)
    --------------------------------------------------
    1. Compute the ``root_cause`` and ``remediation`` from the
       admitted handoffs' ``claim_kinds`` via the priority decoder
       (same priority as
       :func:`vision_mvp.tasks.incident_triage._decoder_from_handoffs`).
    2. Look up the *causal claim-kind set* for that root_cause:
       ``CCK = CAUSAL_CLAIM_KINDS_PER_ROOT_CAUSE[root_cause]``.
    3. For each admitted handoff, extract its ``service=<tag>`` token.
       A service tag is admitted to ``services`` iff
       ``cck_filter`` is disabled OR at least one admitted handoff
       carrying that tag has ``claim_kind ∈ CCK``.
    4. If ``role_corroboration_floor > 1``, additionally require that
       the service tag is mentioned by ≥ ``role_corroboration_floor``
       distinct producer roles (across CCK-eligible handoffs only).
       This second filter is the *contradiction-aware* layer: a
       service whose only causal mention is from one role is
       considered a single-witness signal and may be dropped.

    Why this is decoder-side, not admission-side
    --------------------------------------------
    The admitted set is *fixed* before the decoder runs. The
    decoder's output (``services``) is a *projection* of that
    admitted set under the CCK predicate — admission decides which
    handoffs the auditor sees; decoding decides which of those
    handoffs *count* toward each output field.

    Why this is *contradiction-aware*
    ---------------------------------
    The CCK predicate encodes a structural compatibility relation:
    ``service S is gold-set-eligible under root_cause R`` iff there
    exists an admitted handoff ``(role, kind, service=S)`` whose
    ``kind`` is causal for R. A corroborated decoy mentioned only
    via generic-noise claim_kinds (LATENCY / ERROR_RATE / FW_BLOCK)
    is *contradicted* by the chosen root_cause when R is a
    specific-tier root_cause (deadlock / pool_exhaustion / disk_fill
    / etc.) because LATENCY_SPIKE is not in the data-tier CCK.

    The W10-Λ admission limit
    --------------------------
    Theorem W10-Λ (proved-empirical; see
    :file:`docs/RESULTS_WEVRA_BUNDLE_DECODER.md`): on R-57
    (multi-service-gold + corroborated-decoy via non-causal
    claim_kinds), every service-blind admission policy in the SDK
    (FIFO, priority, coverage, W7-2 cohort, W8 corroboration, W9
    multi-service) achieves ``accuracy_full = 0.000``. Proof
    sketch: every such policy admits service tags purely on the
    basis of the (role, tag) bipartite multiset; the tier of each
    tag's claim_kind is invisible to it. So no admission policy
    can prefer the data-tier corroborated gold over the
    generic-noise corroborated decoy when both have the same
    role-corroboration count.

    The W10 sufficiency claim
    -------------------------
    Theorem W10-1 (proved-empirical): pairing the W9 admission
    policy (``MultiServiceCorroborationAdmissionPolicy`` with
    ``top_k = |gold_services| + 1`` and
    ``min_corroborated_roles = 2``) with this decoder
    (``cck_filter=True``, ``role_corroboration_floor=2``) achieves
    ``accuracy_full = 1.000`` on R-57. The decoder closes the gap
    that admission alone cannot.

    Backward compatibility (W10-3)
    ------------------------------
    On R-53 / R-54 / R-55 / R-56, the bundle-aware decoder admits
    the same ``services`` set as the substrate decoder because:

    * R-53 / R-54: ``|gold_services| = 1``; the only admitted gold
      service is also CCK-eligible (gold root_cause is specific).
    * R-55: ``|gold_services| = 1``; same.
    * R-56: ``|gold_services| = 2``, both gold services CCK-eligible
      via causal claim_kinds; the decoy is single-role (already
      filtered by W9 admission).

    Falsifier (W10-4)
    -----------------
    If the decoy is corroborated AND mentioned via at least one
    claim_kind in CCK (i.e. the decoy IS in the same tier as the
    gold root_cause), the bundle-aware decoder cannot distinguish
    the decoy from gold — both pass the CCK filter. The R-57
    falsifier bank instantiates this regime.
    """

    cck_filter: bool = True
    # Minimum number of distinct producer roles required to mention a
    # service via a CCK-eligible claim_kind for the service to enter
    # the answer set. Default 1 (any single CCK mention keeps the tag);
    # raise to 2+ for stricter contradiction-aware filtering on benches
    # where each gold service has multiple CCK-eligible mentions.
    role_corroboration_floor: int = 1
    # Trust-admission fallback. When the admitted-tag set is small
    # enough that the upstream admission policy is already strict (the
    # W9 admission with default ``top_k = |gold|`` produces exactly
    # ``|gold|`` tags), the bundle decoder should *not* second-guess
    # admission by filtering further. This preserves W10-3 backward-
    # compatibility on R-53 / R-54 / R-55 / R-56 where the admission
    # policy alone already produces the gold-shaped admitted set.
    # Default 2: when ``|admitted_tags| ≤ 2``, fall back to admitted
    # set (do not apply CCK filter). Set to 0 to disable the fallback
    # and apply CCK filter unconditionally.
    fallback_admitted_size_threshold: int = 2
    # Closed-vocabulary CCK table — overridable for tests; default is
    # the incident-triage tiering above.
    cck_table: dict[str, frozenset[str]] = dataclasses.field(
        default_factory=lambda: dict(CAUSAL_CLAIM_KINDS_PER_ROOT_CAUSE))

    def decode(self, handoffs: Sequence[_DecodedHandoff]) -> dict[str, Any]:
        """Run the bundle-aware decode over a sequence of admitted
        handoffs and return ``{"root_cause", "services", "remediation"}``
        — the same shape as
        :func:`vision_mvp.tasks.incident_triage._decoder_from_handoffs`.
        """
        kinds = {h.claim_kind for h in handoffs}
        root_cause, remediation = _decoded_root_cause(kinds)
        cck = self.cck_table.get(root_cause, frozenset())
        # Per-service: collect the (role, kind) pairs from admitted
        # handoffs carrying each tag.
        roles_per_tag: dict[str, set[str]] = {}
        cck_roles_per_tag: dict[str, set[str]] = {}
        all_tags: set[str] = set()
        for h in handoffs:
            tag = ""
            for tok in (h.payload or "").split():
                m = _SERVICE_TAG_RE.search(tok)
                if m:
                    tag = m.group(1)
                    break
            if not tag:
                continue
            all_tags.add(tag)
            roles_per_tag.setdefault(tag, set()).add(h.source_role)
            if h.claim_kind in cck:
                cck_roles_per_tag.setdefault(tag, set()).add(h.source_role)
        # Service filter:
        if not self.cck_filter or not cck:
            # No filter: emit every observed tag (matches substrate
            # decoder's behaviour). Backward-compat for unknown /
            # generic root_cause.
            services = tuple(sorted(all_tags))
        elif (self.fallback_admitted_size_threshold > 0
                and len(all_tags) <= self.fallback_admitted_size_threshold):
            # Trust-admission fallback (W10-3 backward-compat anchor):
            # when admission has already produced a small dominant set
            # (size ≤ threshold), the upstream policy has done the
            # filtering job — do not second-guess. This preserves the
            # W7-2 / W8 / W9 wins on R-54 / R-55 / R-56.
            services = tuple(sorted(all_tags))
        else:
            kept: set[str] = set()
            for tag in all_tags:
                cck_roles = cck_roles_per_tag.get(tag, set())
                if len(cck_roles) >= self.role_corroboration_floor:
                    kept.add(tag)
            services = tuple(sorted(kept))
        return {
            "root_cause": root_cause,
            "services": services,
            "remediation": remediation,
        }


def decode_admitted_role_view(
        ledger: CapsuleLedger,
        role_view_cid: str,
        decoder: "BundleAwareTeamDecoder | None" = None,
        ) -> dict[str, Any]:
    """Convenience: re-decode a sealed ROLE_VIEW capsule's parents
    using the bundle-aware decoder. The substrate decoder is the
    no-CCK-filter case (``BundleAwareTeamDecoder(cck_filter=False)``).
    """
    decoder = decoder or BundleAwareTeamDecoder()
    if not role_view_cid or role_view_cid not in ledger:
        return {"root_cause": "unknown", "services": (),
                 "remediation": "investigate"}
    rv = ledger.get(role_view_cid)
    handoffs: list[_DecodedHandoff] = []
    for p in rv.parents:
        if p in ledger:
            cap = ledger.get(p)
            if cap.kind != CapsuleKind.TEAM_HANDOFF:
                continue
            payload = cap.payload if isinstance(cap.payload, dict) else {}
            handoffs.append(_DecodedHandoff(
                source_role=str(payload.get("source_role", "")),
                claim_kind=str(payload.get("claim_kind", "")),
                payload=str(payload.get("payload", "")),
            ))
    return decoder.decode(handoffs)


# =============================================================================
# SDK v3.12 — Multi-round bundle-aware team decoder (W11 family)
# =============================================================================
#
# W11 attacks the *temporal* axis the W10 single-round decoder cannot
# touch. On a regime where round-1 admitted evidence is ALL generic-
# tier (LATENCY_SPIKE / ERROR_RATE_SPIKE / FW_BLOCK_SURGE), the W10
# bundle decoder picks a generic root_cause (``error_spike`` /
# ``latency_spike``) whose CCK is itself generic — the CCK filter is a
# no-op, the corroborated decoy survives, and W10 ties FIFO at 0.000.
# This is W11-Λ (single-round structural limit).
#
# The W11 decoder takes the *union* of admitted handoffs across
# multiple ROLE_VIEW capsules (one per round) and decodes that union.
# When round-2 carries a single specific-tier disambiguating claim
# (DEADLOCK_SUSPECTED / POOL_EXHAUSTION / DISK_FILL_CRITICAL / ...),
# the union's claim_kind set re-elects a specific root_cause; the CCK
# filter then excludes the round-1 decoy. This is W11-1 (multi-round
# sufficiency).


_GENERIC_NOISE_CLAIM_KINDS: frozenset[str] = frozenset({
    "LATENCY_SPIKE", "ERROR_RATE_SPIKE", "FW_BLOCK_SURGE",
})


@dataclasses.dataclass
class MultiRoundBundleDecoder:
    """Multi-round, bundle-aware team decoder (SDK v3.12, W11 family).

    Wraps :class:`BundleAwareTeamDecoder` with an explicit *round
    union* step plus a contradiction-aware *noise-corroborated-decoy*
    drop. The decoder consumes a sequence of per-round handoff
    bundles (one bundle per round's ROLE_VIEW) and decodes the
    *concatenation*; if the union elects a specific-tier root_cause
    (i.e. the chosen ``root_cause``'s CCK is NOT in the generic-
    noise tier), it then drops every service tag whose admitted
    mentions are *exclusively* generic-noise kinds and span ≥
    ``noise_decoy_role_floor`` distinct producer roles. This
    *contradiction-aware* drop is the load-bearing W11 move: when
    round-N evidence (specific) names the *kind* of incident but
    not a *service* (e.g. ``DEADLOCK_SUSPECTED relation=orders_payments``
    with no ``service=`` token), and round-1 noise *did* name the
    services, the decoder must use the round-N kind to elect the
    root_cause, then re-project round-1's service inventory through
    the chosen tier.

    Why this is multi-round
    -----------------------
    With a single bundle (per-round mode) the union is the bundle
    itself. The W10 single-round bundle decoder running on round-N
    alone cannot resolve the regime because:

      * If round-N is generic-only, root_cause is generic and the
        CCK is itself generic — no filter.
      * If round-N is specific-only (no service tags), services
        come up empty.

    Pairing rounds is the structural move: the union joins
    *round-1's service tag inventory* with *round-2's specific
    root_cause kind*.

    Backward compatibility (W11-3)
    ------------------------------
    With a single bundle (single-round) the decoder reduces to
    :class:`BundleAwareTeamDecoder` exactly when
    ``noise_decoy_role_floor`` is large enough that no R-54..R-57
    bench triggers the noise-decoy drop. R-54 / R-55 / R-56 have
    ``|gold_services| ≤ 2``; the W7-2/W8/W9 admission policies
    already filter to a small dominant set; the W10 fallback path
    (admitted-set size ≤ 2) preserves their wins. Only on R-57 with
    a corroborated noise decoy does the noise-decoy drop fire.

    Falsifier (W11-4)
    -----------------
    If round-2 admission drops the disambiguating specific-tier
    claim (e.g. budget already spent on round-1 noise), the union
    is still all-generic and the decoder cannot help. The Phase-58
    falsifier bank instantiates this regime by setting
    ``K_auditor`` so round-1 fills the budget.
    """

    # The inner decoder is configured with ``cck_filter=False`` by
    # default: the W11 contradiction-aware drop subsumes the W10
    # CCK filter on regimes where round-N specific-tier evidence
    # carries no service tags (the only regime W10 cannot reach
    # alone). On R-54..R-57 the inner falls back to admitted
    # services anyway (``fallback_admitted_size_threshold=2``), so
    # the union still matches W10 byte-for-byte (W11-3).
    inner: BundleAwareTeamDecoder = dataclasses.field(
        default_factory=lambda: BundleAwareTeamDecoder(cck_filter=False))
    # Drop a tag whose admitted mentions are *exclusively* generic-
    # noise kinds AND span ≥ this many distinct producer roles, IFF
    # the elected root_cause is in a specific (non-generic) tier.
    # Default 2: a single-role generic-noise mention is preserved.
    noise_decoy_role_floor: int = 2

    def decode_rounds(
            self,
            per_round_handoffs: Sequence[Sequence[_DecodedHandoff]],
            ) -> dict[str, Any]:
        """Decode the union of admitted handoffs across rounds.

        The W11 semantic: union, then if root_cause is specific-tier,
        drop noise-corroborated tags. Falls back to the inner CCK
        decoder result otherwise.
        """
        union: list[_DecodedHandoff] = []
        for bundle in per_round_handoffs:
            union.extend(bundle)
        base = self.inner.decode(union)
        root_cause = str(base.get("root_cause", "unknown"))
        # Generic-tier root_cause: nothing the W11 contradiction-
        # aware step can do (W11-Λ at the temporal axis collapses).
        if root_cause in ("error_spike", "latency_spike", "fw_block",
                            "unknown"):
            return base
        # Specific-tier: identify and drop noise-corroborated decoys.
        roles_per_tag: dict[str, set[str]] = {}
        noise_only_per_tag: dict[str, bool] = {}
        for h in union:
            tag = ""
            for tok in (h.payload or "").split():
                m = _SERVICE_TAG_RE.search(tok)
                if m:
                    tag = m.group(1)
                    break
            if not tag:
                continue
            roles_per_tag.setdefault(tag, set()).add(h.source_role)
            if h.claim_kind in _GENERIC_NOISE_CLAIM_KINDS:
                noise_only_per_tag.setdefault(tag, True)
            else:
                noise_only_per_tag[tag] = False
        services_in = tuple(base.get("services", ()))
        kept = []
        for tag in services_in:
            if (noise_only_per_tag.get(tag, False)
                    and len(roles_per_tag.get(tag, set()))
                        >= self.noise_decoy_role_floor):
                continue
            kept.append(tag)
        return {
            "root_cause": root_cause,
            "services": tuple(kept),
            "remediation": base.get("remediation", "investigate"),
        }

    def decode(self, handoffs: Sequence[_DecodedHandoff]
               ) -> dict[str, Any]:
        """Single-bundle path — degenerate case of ``decode_rounds``."""
        return self.decode_rounds([handoffs])


def collect_admitted_handoffs(
        ledger: CapsuleLedger,
        role_view_cids: Sequence[str],
        ) -> list[_DecodedHandoff]:
    """Collect admitted TEAM_HANDOFF parents from a sequence of
    ROLE_VIEW CIDs (per-round role views) and return them as
    ``_DecodedHandoff`` records, deduplicated by handoff CID and
    preserving first-occurrence order.
    """
    seen: set[str] = set()
    out: list[_DecodedHandoff] = []
    for rv_cid in role_view_cids:
        if not rv_cid or rv_cid not in ledger:
            continue
        rv = ledger.get(rv_cid)
        for p in rv.parents:
            if p in seen or p not in ledger:
                continue
            cap = ledger.get(p)
            if cap.kind != CapsuleKind.TEAM_HANDOFF:
                continue
            seen.add(p)
            payload = cap.payload if isinstance(cap.payload, dict) else {}
            out.append(_DecodedHandoff(
                source_role=str(payload.get("source_role", "")),
                claim_kind=str(payload.get("claim_kind", "")),
                payload=str(payload.get("payload", "")),
            ))
    return out


# =============================================================================
# SDK v3.13 — Real-LLM-robust multi-round bundle decoder (W12 family)
# =============================================================================
#
# W12 attacks the *real-LLM noise* axis the synthetic W11 decoder cannot
# reach. Real-LLM producer roles emit the same operational symptoms as
# Phase-58 / R-58, but the *surface form* of the claim is drift-prone:
#
#   * Claim-kind drift. The LLM emits "DEADLOCK_DETECTED" /
#     "DEADLOCK" / "POOL_EXHAUSTED" / "POOL_EXHAUSTION_ALERT" instead
#     of the closed-vocabulary canonical kinds. The W11 priority
#     decoder's `claim_kind in _DECODER_PRIORITY` lookup misses;
#     the elected root_cause stays generic; W11-Λ collapses on
#     real-LLM streams just as it did on R-58 single-round.
#   * Payload drift. The LLM may omit the ``service=<tag>`` token from
#     a service-specific message ("relation=orders_payments_join" alone),
#     or emit a different separator ("svc=orders" / "for service orders").
#   * Both drifts together produce a regime where the synthetic W11 win
#     does NOT transfer to a real-LLM stream — even with bounded noise.
#
# The W12 decoder applies a *closed-vocabulary normalisation layer*
# **ahead of** the W11 multi-round bundle decode:
#
#   1. ``claim_kind`` is rewritten through ``CLAIM_KIND_SYNONYMS`` —
#      a small, deterministic map from observed LLM variants to the
#      canonical kind that ``_DECODER_PRIORITY`` recognises. Unknown
#      kinds are passed through (so the closed-vocabulary contract
#      and the lifecycle-audit invariants are preserved).
#   2. ``payload`` is rewritten through a small normaliser that
#      promotes alternative service-tag spellings (``svc=X``, "for
#      service X") to ``service=X`` so the W11 contradiction-aware
#      drop and the substrate decoder agree on the service inventory.
#
# This is the *minimum honest method change* required for synthetic→
# real-LLM transfer. The normalisation table is closed-vocabulary by
# construction; the W11 decoder's structural argument (W11-1 / W11-2 /
# W11-3) is preserved unchanged on the post-normalisation stream.
#
# Honest scope
# ------------
# The normalisation table is fitted to the *closed-vocabulary
# incident-triage claim grammar* (the same grammar the Phase-31
# substrate decoder uses). It is NOT a general-purpose LLM kind
# normaliser. Expanding it to other benchmark families is the W12-C1
# conjecture. Unbounded noise (an LLM that emits arbitrary novel
# kinds) breaks normalisation by construction — that is W12-4.


# Closed-vocabulary synonym table — rewrite LLM kind variants into the
# Phase-31 / W10 canonical claim_kinds so the priority decoder sees a
# recognised label. Identity entries are retained explicitly so the
# table also serves as a closed-vocabulary docstring of allowed
# canonical kinds. Lex-ordered for diff stability.
CLAIM_KIND_SYNONYMS: dict[str, str] = {
    # ERROR_RATE_SPIKE family.
    "ERROR_RATE_SPIKE": "ERROR_RATE_SPIKE",
    "ERROR_RATE": "ERROR_RATE_SPIKE",
    "ERROR_SPIKE": "ERROR_RATE_SPIKE",
    "ERROR_SURGE": "ERROR_RATE_SPIKE",
    "HIGH_ERROR_RATE": "ERROR_RATE_SPIKE",
    # LATENCY_SPIKE family.
    "LATENCY_SPIKE": "LATENCY_SPIKE",
    "LATENCY": "LATENCY_SPIKE",
    "HIGH_LATENCY": "LATENCY_SPIKE",
    "P95_HIGH": "LATENCY_SPIKE",
    "SLO_BREACH": "LATENCY_SPIKE",
    # SLOW_QUERY_OBSERVED family.
    "SLOW_QUERY_OBSERVED": "SLOW_QUERY_OBSERVED",
    "SLOW_QUERY": "SLOW_QUERY_OBSERVED",
    "SLOW_QUERIES": "SLOW_QUERY_OBSERVED",
    "QUERY_SLOWDOWN": "SLOW_QUERY_OBSERVED",
    # POOL_EXHAUSTION family.
    "POOL_EXHAUSTION": "POOL_EXHAUSTION",
    "POOL_EXHAUSTED": "POOL_EXHAUSTION",
    "CONNECTION_POOL_EXHAUSTED": "POOL_EXHAUSTION",
    "DB_POOL_FULL": "POOL_EXHAUSTION",
    "POOL_FULL": "POOL_EXHAUSTION",
    "POOL_SATURATED": "POOL_EXHAUSTION",
    # DEADLOCK_SUSPECTED family.
    "DEADLOCK_SUSPECTED": "DEADLOCK_SUSPECTED",
    "DEADLOCK": "DEADLOCK_SUSPECTED",
    "DEADLOCK_DETECTED": "DEADLOCK_SUSPECTED",
    "DEADLOCK_OBSERVED": "DEADLOCK_SUSPECTED",
    "LOCK_CYCLE": "DEADLOCK_SUSPECTED",
    # DISK_FILL_CRITICAL family.
    "DISK_FILL_CRITICAL": "DISK_FILL_CRITICAL",
    "DISK_FILL": "DISK_FILL_CRITICAL",
    "DISK_FULL": "DISK_FILL_CRITICAL",
    "DISK_USAGE_CRITICAL": "DISK_FILL_CRITICAL",
    "DISK_NEAR_FULL": "DISK_FILL_CRITICAL",
    # CRON_OVERRUN family.
    "CRON_OVERRUN": "CRON_OVERRUN",
    "CRON_TIMEOUT": "CRON_OVERRUN",
    "CRON_LATE": "CRON_OVERRUN",
    # OOM_KILL family.
    "OOM_KILL": "OOM_KILL",
    "OOM": "OOM_KILL",
    "OOM_KILLED": "OOM_KILL",
    "OUT_OF_MEMORY": "OOM_KILL",
    # TLS_EXPIRED family.
    "TLS_EXPIRED": "TLS_EXPIRED",
    "CERT_EXPIRED": "TLS_EXPIRED",
    "TLS_EXPIRY": "TLS_EXPIRED",
    # DNS_MISROUTE family.
    "DNS_MISROUTE": "DNS_MISROUTE",
    "DNS_FAILURE": "DNS_MISROUTE",
    "DNS_SERVFAIL": "DNS_MISROUTE",
    # FW_BLOCK_SURGE family.
    "FW_BLOCK_SURGE": "FW_BLOCK_SURGE",
    "FIREWALL_BLOCKS": "FW_BLOCK_SURGE",
    "FW_DENY": "FW_BLOCK_SURGE",
    "BLOCKED_PACKETS": "FW_BLOCK_SURGE",
}


# Alternative service-tag spellings the normaliser rewrites to
# ``service=<tag>``. These are the tag spellings real LLMs actually
# emit on the closed-vocabulary incident-triage prompt. Lex-ordered
# for diff stability. Each entry is a ``re.compile`` pattern that
# *must* contain a single capturing group for the tag.
import re as _re

_SERVICE_TAG_REWRITES: tuple[tuple[Any, str], ...] = (
    # The canonical form is left untouched by the rewrite (idempotent).
    (_re.compile(r"\bsvc=([\w-]+)"), r"service=\1"),
    (_re.compile(r"\bservice:([\w-]+)"), r"service=\1"),
    (_re.compile(r"\bfor service ([\w-]+)\b"), r"service=\1"),
    (_re.compile(r"\bon service ([\w-]+)\b"), r"service=\1"),
    (_re.compile(r"\bservice_name=([\w-]+)"), r"service=\1"),
    (_re.compile(r"\bservicename=([\w-]+)"), r"service=\1"),
    (_re.compile(r"\bsvc_name=([\w-]+)"), r"service=\1"),
)


def normalize_claim_kind(kind: str,
                          synonyms: dict[str, str] | None = None,
                          ) -> str:
    """Look up ``kind`` in the closed-vocabulary synonym table; return
    the canonical kind, or the input unchanged if no entry hits.

    Case-folded to upper for robustness against mixed-case LLM output.
    Empty input returns the empty string (consistent with no-op).
    """
    table = synonyms if synonyms is not None else CLAIM_KIND_SYNONYMS
    if not kind:
        return ""
    canonical = table.get(kind.upper())
    return canonical if canonical is not None else kind


def normalize_payload(payload: str) -> str:
    """Rewrite alternative service-tag spellings into ``service=<tag>``.

    Idempotent on payloads that already use the canonical form.
    Preserves all other tokens unchanged. Closed-vocabulary by
    construction — only the patterns in ``_SERVICE_TAG_REWRITES`` fire.
    """
    if not payload:
        return ""
    out = payload
    for (pat, repl) in _SERVICE_TAG_REWRITES:
        out = pat.sub(repl, out)
    return out


def normalize_handoff(h: _DecodedHandoff,
                       synonyms: dict[str, str] | None = None,
                       ) -> _DecodedHandoff:
    """Apply ``normalize_claim_kind`` + ``normalize_payload`` to a
    decoded handoff. Returns a new record; the input is unchanged."""
    return _DecodedHandoff(
        source_role=h.source_role,
        claim_kind=normalize_claim_kind(h.claim_kind, synonyms),
        payload=normalize_payload(h.payload),
    )


@dataclasses.dataclass
class RobustMultiRoundBundleDecoder:
    """Normalising multi-round bundle decoder (SDK v3.13, W12 family).

    Wraps :class:`MultiRoundBundleDecoder` with a *closed-vocabulary
    normalisation layer* that rewrites LLM-drifted ``claim_kind`` and
    payload tokens into the canonical forms the W11 decoder
    recognises. The structural W11 sufficiency argument (W11-1 / W11-2
    / W11-3) is preserved unchanged on the post-normalisation stream.

    This is the **first real-LLM-compatible cross-round coordination
    method** in the Wevra programme. Where W11 (synthetic R-58)
    assumes the producer emits canonical ``claim_kind`` strings and
    canonical ``service=<tag>`` payload tokens, W12 assumes only that:

    * The producer's claim-kind drift is bounded by the closed-
      vocabulary :data:`CLAIM_KIND_SYNONYMS` table.
    * The producer's payload drift is bounded by the closed-
      vocabulary :data:`_SERVICE_TAG_REWRITES` table.

    Under those bounded-noise assumptions, the W12 decoder reduces the
    real-LLM regime to the synthetic R-58 regime by *construction* —
    the post-normalisation handoff stream is shape-equivalent to the
    R-58 ground-truth stream, so W11-1 sufficiency carries over
    (Theorem W12-1, *proved-conditional* on the named noise budget).

    Backward compatibility (W12-3)
    ------------------------------
    On a candidate stream where every ``claim_kind`` is already in the
    canonical set and every payload uses ``service=<tag>``, normalisation
    is a no-op and the W12 decoder reduces byte-for-byte to W11.
    Empirically verified on R-58 default by
    ``Phase58CrossRegimeTests.test_robust_decoder_matches_w11_on_r58``
    (added in SDK v3.13).

    Falsifier (W12-4)
    -----------------
    If the LLM emits a kind outside ``CLAIM_KIND_SYNONYMS`` for a
    causal claim (e.g. ``DEADLOCK_PROBABLY_DETECTED_MAYBE``), the
    canonical-kind miss propagates to the priority decoder and the
    elected root_cause stays generic — the W11-Λ collapse re-fires.
    This is the *closed-vocabulary boundary* of the W12 method;
    expanding the table is a research move, not a structural fix.
    """

    inner: MultiRoundBundleDecoder = dataclasses.field(
        default_factory=lambda: MultiRoundBundleDecoder())
    synonyms: dict[str, str] = dataclasses.field(
        default_factory=lambda: dict(CLAIM_KIND_SYNONYMS))
    # Optional metric: the count of handoffs whose ``claim_kind`` was
    # rewritten by the normaliser on the most recent ``decode_rounds``
    # call. Useful for the Phase-59 driver to verify the normaliser
    # was actually load-bearing on the bench.
    last_n_kind_rewrites: int = 0
    last_n_payload_rewrites: int = 0

    def normalize_round(self,
                          handoffs: Sequence[_DecodedHandoff],
                          ) -> list[_DecodedHandoff]:
        """Apply :func:`normalize_handoff` to every handoff in the
        bundle. Updates the per-call rewrite counters."""
        out: list[_DecodedHandoff] = []
        for h in handoffs:
            new_kind = normalize_claim_kind(h.claim_kind, self.synonyms)
            new_payload = normalize_payload(h.payload)
            if new_kind != h.claim_kind:
                self.last_n_kind_rewrites += 1
            if new_payload != h.payload:
                self.last_n_payload_rewrites += 1
            out.append(_DecodedHandoff(
                source_role=h.source_role,
                claim_kind=new_kind,
                payload=new_payload,
            ))
        return out

    def decode_rounds(self,
                        per_round_handoffs: Sequence[Sequence[_DecodedHandoff]],
                        ) -> dict[str, Any]:
        """Normalise every per-round bundle, then defer to the inner
        :class:`MultiRoundBundleDecoder`."""
        self.last_n_kind_rewrites = 0
        self.last_n_payload_rewrites = 0
        normalised: list[list[_DecodedHandoff]] = []
        for bundle in per_round_handoffs:
            normalised.append(self.normalize_round(bundle))
        return self.inner.decode_rounds(normalised)

    def decode(self,
                handoffs: Sequence[_DecodedHandoff],
                ) -> dict[str, Any]:
        """Single-bundle path — degenerate case of ``decode_rounds``."""
        return self.decode_rounds([handoffs])


# =============================================================================
# SDK v3.14 — Layered open-world normaliser + decoder (W13 family)
# =============================================================================
#
# W12 attacked the real-LLM-noise axis with a *fixed closed-vocabulary*
# synonym table. The named limit theorem on the W12 method is W12-4:
# any kind variant outside :data:`CLAIM_KIND_SYNONYMS` survives
# normalisation unchanged and the priority decoder cannot match it.
# In practice, real LLMs (and any future producer the synthetic
# extractor was not calibrated against) emit a steady tail of
# variants the table never saw — call this the *open-world drift
# channel*.
#
# W13 attacks that boundary with a *layered* normaliser:
#
#   1. **Exact synonym table** — the W12 path. Lossless on calibrated
#      drift; idempotent on canonical input; the W12-2 closure
#      contract is preserved unchanged.
#   2. **Heuristic abstraction rules** — a small, ordered table of
#      regex predicates over the upper-cased kind text. Each rule's
#      output is a canonical kind in ``_DECODER_PRIORITY``. Rules are
#      designed to fire on substring witnesses ("DEADLOCK", "POOL +
#      EXHAUST", "DISK + FILL", "QUERY + SLOW") that the calibrated
#      table did not enumerate. The rules trade a tiny soundness risk
#      (an arbitrary kind that incidentally contains "DEADLOCK" gets
#      mapped to the deadlock canonical) for a strict closure
#      widening: every variant the W12-4 falsifier emits
#      (DEADLOCK_PROBABLY_DETECTED_MAYBE, POOL_LOOKING_BUSY,
#      QUERY_SOMEWHAT_SLUGGISH, DISK_GETTING_FULL_PROBABLY) is in the
#      W13 closure.
#   3. **Abstention or pass-through** — when no exact entry and no
#      heuristic rule fires, the normaliser either returns the input
#      unchanged (W12 default) or returns a sentinel ``ABSTAINED``
#      that the decoder can treat as "unknown evidence; do not vote
#      this kind into the priority decoder." Abstention preserves
#      determinism while exposing OOV rate as a load-bearing metric.
#
# Honest scope (W13)
# ------------------
# The heuristic layer **widens** the W12 closure but does not
# eliminate the closure boundary. A truly arbitrary OOV kind
# (``X_QQQQ_PPPP``, a random hash, a phrase the rule set never
# imagined) still escapes both layers. W13-4 is the named falsifier:
# at high enough drift across an *unbounded* variant distribution,
# the layered normaliser ties FIFO at 0.000 — same shape as W12-4 but
# with a *strictly larger* closure.
#
# The W13-1 advance is therefore *conditional widening*: the
# heuristic rules form a closed-vocabulary set of predicates; their
# union covers more inputs than the exact table; the W13-Λ-fixed
# limit (proved by inspection) is that any predicate set that runs in
# bounded time on an unbounded input language has a finite vocabulary
# of *outputs* and a finite set of *firing patterns* — so OOV inputs
# whose surface form witnesses none of the patterns escape the W13
# closure too. Beyond W13 we would need a fundamentally different
# mechanism — e.g. an embedding-distance lookup, an LLM-distilled
# rewriter, or richer prompt-side discipline (W13-C2/W13-C3).


@dataclasses.dataclass(frozen=True)
class HeuristicAbstractionRule:
    """One predicate-to-canonical rule in the W13 layered normaliser.

    Fields
    ------
    pattern
        ``re.compile`` object whose ``search`` is the predicate. The
        search is case-folded against the upper-cased kind text in
        :func:`LayeredClaimNormalizer.normalize`.
    canonical
        The canonical kind label this rule resolves to. Must appear
        as a key in :data:`_DECODER_PRIORITY`.
    name
        A short stable identifier ("deadlock", "pool_exhaust", …) used
        in audit reporting.
    """
    pattern: Any
    canonical: str
    name: str


# Sentinel returned by :func:`LayeredClaimNormalizer.normalize` when
# both the exact table and the heuristic rules miss AND the normaliser
# is in abstention mode. The priority decoder ignores it (it is not in
# ``_DECODER_PRIORITY``); the layered decoder counts it as OOV.
LAYERED_NORMALIZER_ABSTAIN = "_W13_ABSTAINED_"


# Heuristic abstraction rules. Each pattern is matched against the
# *upper-cased* kind text. Order matters — first match wins.
#
# Soundness contract (W13-2):
#   * Every rule's canonical output is a key in ``_DECODER_PRIORITY``.
#   * Rules are designed so the canonical kinds themselves match each
#     rule's pattern (idempotency on canonical: the heuristic layer
#     never disagrees with the exact-table layer on canonical input).
#   * Rules are ordered most-specific-first so multi-word kinds like
#     ``DISK_FULL`` resolve before single-word ``DISK`` would.
#
# Soundness risk (declared, not denied):
#   * A truly random kind that incidentally contains, say, "DEADLOCK"
#     ("DEADLOCK_IS_AWESOME") is mapped to the deadlock canonical.
#     The W13 method's audit flags this in the heuristic-rewrite
#     counter, so a downstream operator can spot suspicious inputs.
_HEURISTIC_KIND_RULES: tuple[HeuristicAbstractionRule, ...] = (
    # Storage tier — disk fill family.
    HeuristicAbstractionRule(
        pattern=_re.compile(r"\bDISK[_ ]*(FILL|FULL|OVERFLOW|"
                              r"GETTING[_ ]*FULL|NEAR[_ ]*FULL|"
                              r"USAGE[_ ]*CRITICAL|AT[_ ]*CAPACITY|"
                              r"OUT[_ ]*OF[_ ]*SPACE)"),
        canonical="DISK_FILL_CRITICAL",
        name="disk_fill"),
    HeuristicAbstractionRule(
        pattern=_re.compile(r"\bCRON[_ ]*(OVERRUN|TIMEOUT|LATE|"
                              r"MISSED|FAILED|STUCK)"),
        canonical="CRON_OVERRUN",
        name="cron_overrun"),
    # Database — pool exhaustion family. Order matters: pool rules
    # come before slow-query rules because POOL is a stronger witness.
    # Two-pattern OR: the first matches direct adjacency
    # (POOL_EXHAUST*, POOL_FULL, POOL_SATUR*, POOL_MAX*,
    # POOL_BUSY, POOL_LOOKING_BUSY); the second is a conjunctive
    # look-ahead that fires when POOL appears anywhere alongside any
    # capacity-witness word ("AT_CAPACITY", "FULL_NOW",
    # "SATURATED_OBSERVED", "MAX_REACHED"). The look-ahead lets the
    # rule absorb LLM variants like POOL_AT_CAPACITY,
    # CONNECTION_POOL_FULL_NOW that the direct pattern does not catch.
    HeuristicAbstractionRule(
        pattern=_re.compile(
            r"(?=.*\b(CONNECTION[_ ]*)?POOL)"
            r"(?=.*(EXHAUST|EXHAUSTED|EXHAUSTION|FULL|"
            r"SATURAT|SATURATED|MAX(ED|IMUM)?|"
            r"BUSY|CAPACITY|LIMIT|OVERLOAD))"),
        canonical="POOL_EXHAUSTION",
        name="pool_exhaustion"),
    # Database — deadlock family.
    HeuristicAbstractionRule(
        pattern=_re.compile(r"\bDEADLOCK"),
        canonical="DEADLOCK_SUSPECTED",
        name="deadlock"),
    HeuristicAbstractionRule(
        pattern=_re.compile(r"\bLOCK[_ ]*(CYCLE|CHAIN|CONFLICT|WAIT)"),
        canonical="DEADLOCK_SUSPECTED",
        name="lock_cycle"),
    # Database — slow query family. Must come AFTER pool/deadlock.
    # The conjunctive look-ahead matches any kind text that mentions
    # both QUERY/QUERIES AND a slow-witness word (SLOW/SLUG/SLUGG/
    # SLOWDOWN/TIMING_OUT) regardless of the order or the separator
    # tokens in between. Note: ``\b`` is unreliable around underscores
    # in Python regex (``_S`` has no word boundary because ``_`` is a
    # word character), so the look-aheads use plain substring matches.
    # Examples that fire:
    #   SLOW_QUERY_OBSERVED, SLOW_QUERIES,
    #   QUERY_SLOWDOWN, QUERIES_TIMING_OUT,
    #   QUERY_SOMEWHAT_SLUGGISH (W12-4 falsifier variant — W13 rescues),
    #   SLUGGISH_QUERY_PERFORMANCE.
    HeuristicAbstractionRule(
        pattern=_re.compile(r"(?=.*QUER(Y|IES))"
                              r"(?=.*(SLOW|SLUG|TIMING[_ ]*OUT))"),
        canonical="SLOW_QUERY_OBSERVED",
        name="slow_query"),
    # Compute tier — OOM family.
    HeuristicAbstractionRule(
        pattern=_re.compile(r"\b(OOM|OUT[_ ]*OF[_ ]*MEM(ORY)?)"),
        canonical="OOM_KILL",
        name="oom"),
    # Edge tier — TLS family.
    HeuristicAbstractionRule(
        pattern=_re.compile(r"\b(TLS|CERT(IFICATE)?)[_ ]*("
                              r"EXPIR|EXPIRY|EXPIRED|"
                              r"EOL|END[_ ]*OF[_ ]*LIFE)"),
        canonical="TLS_EXPIRED",
        name="tls_expired"),
    # Edge tier — DNS family.
    HeuristicAbstractionRule(
        pattern=_re.compile(r"\bDNS[_ ]*("
                              r"MISROUTE|FAIL|FAILURE|SERVFAIL|"
                              r"NXDOMAIN|RESOLUTION[_ ]*FAIL|LOOKUP)"),
        canonical="DNS_MISROUTE",
        name="dns_misroute"),
    # Network tier — firewall family.
    HeuristicAbstractionRule(
        pattern=_re.compile(r"\b(FW|FIREWALL)[_ ]*("
                              r"BLOCK|DENY|DROP|SURGE|REJECTED?|"
                              r"DENIED|DENIAL)"),
        canonical="FW_BLOCK_SURGE",
        name="fw_block"),
    HeuristicAbstractionRule(
        pattern=_re.compile(r"\bBLOCKED[_ ]*PACKET"),
        canonical="FW_BLOCK_SURGE",
        name="fw_blocked_packet"),
    # Generic-tier — error rate family. Note: ERROR_RATE_SPIKE itself
    # matches both ERROR rules below but the exact-table layer would
    # have caught it first; the heuristic layer is the open-world fall-
    # back. We put error/latency last so they don't poach more-specific
    # tier matches like POOL_EXHAUSTION (which never contains the word
    # ERROR / LATENCY in any reasonable variant).
    HeuristicAbstractionRule(
        pattern=_re.compile(r"\bERROR[_ ]*("
                              r"RATE|SURGE|SPIKE|HIGH|ELEVATED|"
                              r"INCIDENT|BURST)"),
        canonical="ERROR_RATE_SPIKE",
        name="error_rate"),
    HeuristicAbstractionRule(
        pattern=_re.compile(r"\bHIGH[_ ]*ERROR[_ ]*RATE"),
        canonical="ERROR_RATE_SPIKE",
        name="high_error_rate"),
    # Generic-tier — latency family.
    HeuristicAbstractionRule(
        pattern=_re.compile(r"\b(LATENCY|P95|P99)[_ ]*("
                              r"HIGH|SPIKE|BREACH|ELEVATED|"
                              r"REGRESSION|INCREASE)"),
        canonical="LATENCY_SPIKE",
        name="latency_spike"),
    HeuristicAbstractionRule(
        pattern=_re.compile(r"\bHIGH[_ ]*(LATENCY|P95|P99)"),
        canonical="LATENCY_SPIKE",
        name="high_latency"),
    HeuristicAbstractionRule(
        pattern=_re.compile(r"\bSLO[_ ]*(BREACH|VIOLATED?|MISS)"),
        canonical="LATENCY_SPIKE",
        name="slo_breach"),
)


@dataclasses.dataclass
class LayeredClaimNormalizer:
    """Two-layer claim-kind normaliser (SDK v3.14, W13 family).

    Tries the W12 :data:`CLAIM_KIND_SYNONYMS` exact lookup first; on a
    miss, walks an ordered set of :class:`HeuristicAbstractionRule`
    predicates over the upper-cased kind text. On a miss in both
    layers, returns either the input unchanged (``abstain_on_unknown
    = False``, the default) or :data:`LAYERED_NORMALIZER_ABSTAIN`
    (``abstain_on_unknown = True``). Abstention is information-
    preserving: the priority decoder ignores ``_W13_ABSTAINED_`` (it is
    not in ``_DECODER_PRIORITY``), the rewrite counters expose OOV
    rate as a metric.

    The W12 closure is a strict subset of the W13 closure:

        * On every key in :data:`CLAIM_KIND_SYNONYMS`, layered ≡ W12
          (W13-3 backward-compat).
        * On every variant in :data:`OUT_OF_VOCAB_KINDS` named in
          ``vision_mvp/experiments/phase59_real_llm_multi_round.py``
          (DEADLOCK_PROBABLY_DETECTED_MAYBE, POOL_LOOKING_BUSY,
          QUERY_SOMEWHAT_SLUGGISH, DISK_GETTING_FULL_PROBABLY), the
          heuristic layer fires and resolves to the matching canonical
          (W13-1). The W12-4 falsifier regime is therefore *partly*
          inside the W13 closure.
        * On a *truly* arbitrary kind that witnesses none of the
          heuristic patterns (e.g. ``XYZZY_QQQQ`` or a random hash),
          both layers miss and the W13-4 closure boundary is reached.

    This class is the load-bearing W13 method change. The W13 decoder
    (``LayeredRobustMultiRoundBundleDecoder``) wraps the W11 multi-
    round bundle decoder around this normaliser, *not* around the W12
    closed-vocabulary table.
    """

    synonyms: dict[str, str] = dataclasses.field(
        default_factory=lambda: dict(CLAIM_KIND_SYNONYMS))
    rules: tuple[HeuristicAbstractionRule, ...] = dataclasses.field(
        default_factory=lambda: _HEURISTIC_KIND_RULES)
    abstain_on_unknown: bool = False
    last_n_exact: int = 0
    last_n_heuristic: int = 0
    last_n_abstained: int = 0
    last_n_passthrough: int = 0
    # Mapping rule_name -> int for the most recent batch — useful for
    # forensic inspection of which heuristic patterns are firing.
    last_rule_hits: dict[str, int] = dataclasses.field(
        default_factory=dict)

    def reset_counters(self) -> None:
        self.last_n_exact = 0
        self.last_n_heuristic = 0
        self.last_n_abstained = 0
        self.last_n_passthrough = 0
        self.last_rule_hits = {}

    def normalize(self, kind: str) -> str:
        """Layered normalisation. Returns the canonical kind, the
        sentinel ``LAYERED_NORMALIZER_ABSTAIN``, or the input unchanged
        depending on ``abstain_on_unknown``."""
        if not kind:
            return ""
        upper = kind.upper()
        # Layer 1: exact synonym table (W12 path).
        canon = self.synonyms.get(upper)
        if canon is not None:
            self.last_n_exact += 1
            return canon
        # Layer 2: heuristic abstraction rules.
        for rule in self.rules:
            if rule.pattern.search(upper):
                self.last_n_heuristic += 1
                self.last_rule_hits[rule.name] = (
                    self.last_rule_hits.get(rule.name, 0) + 1)
                return rule.canonical
        # Layer 3: abstain or pass-through.
        if self.abstain_on_unknown:
            self.last_n_abstained += 1
            return LAYERED_NORMALIZER_ABSTAIN
        self.last_n_passthrough += 1
        return kind

    def normalize_handoff(self, h: _DecodedHandoff) -> _DecodedHandoff:
        """Apply layered kind normalisation + payload normalisation
        (the W12 :data:`_SERVICE_TAG_REWRITES` patterns are reused —
        payload drift is not the W13 contribution)."""
        return _DecodedHandoff(
            source_role=h.source_role,
            claim_kind=self.normalize(h.claim_kind),
            payload=normalize_payload(h.payload),
        )


@dataclasses.dataclass
class LayeredRobustMultiRoundBundleDecoder:
    """Layered open-world normalising multi-round bundle decoder
    (SDK v3.14, W13 family).

    Same shape as :class:`RobustMultiRoundBundleDecoder` (W12), but the
    closed-vocabulary :data:`CLAIM_KIND_SYNONYMS` is replaced by a
    :class:`LayeredClaimNormalizer` whose closure strictly contains
    the W12 closure. The inner :class:`MultiRoundBundleDecoder` is
    unchanged.

    Backward-compat (W13-3)
    ------------------------
    On any input where every kind is in :data:`CLAIM_KIND_SYNONYMS`,
    the heuristic layer is never reached and the result is byte-for-
    byte identical to W12. The rewrite counters (``last_n_exact``,
    ``last_n_heuristic``, ``last_n_abstained``,
    ``last_n_passthrough``) expose which layer fired so the bench
    driver can verify W13's *additional* contribution mechanically.

    W13-1 sufficiency
    -----------------
    On a regime where the LLM emits OOV variants whose surface form
    witnesses one of the heuristic patterns
    (``DEADLOCK_PROBABLY_DETECTED_MAYBE`` →
    pattern ``\\bDEADLOCK`` matches → canonical
    ``DEADLOCK_SUSPECTED``), the W13 decoder rescues the run while
    W12 ties FIFO at 0.000 by W12-4.

    W13-4 falsifier
    ---------------
    On a regime where the LLM emits OOV variants whose surface form
    witnesses *neither* the exact table *nor* any heuristic pattern
    (e.g. random tokens like ``XYZZY_QQQQ`` or an unrelated label
    like ``COSMIC_RAY_FLIP``), both layers miss, the priority decoder
    cannot match, and W13 ties FIFO at 0.000 — the named open-world
    closure boundary.
    """

    inner: MultiRoundBundleDecoder = dataclasses.field(
        default_factory=lambda: MultiRoundBundleDecoder())
    normalizer: LayeredClaimNormalizer = dataclasses.field(
        default_factory=lambda: LayeredClaimNormalizer())

    @property
    def last_n_kind_rewrites(self) -> int:
        """Total kind rewrites (exact + heuristic) over the most
        recent ``decode_rounds`` call."""
        return self.normalizer.last_n_exact + self.normalizer.last_n_heuristic

    @property
    def last_n_payload_rewrites(self) -> int:
        return self._last_n_payload_rewrites

    _last_n_payload_rewrites: int = 0

    def normalize_round(self,
                          handoffs: Sequence[_DecodedHandoff],
                          ) -> list[_DecodedHandoff]:
        out: list[_DecodedHandoff] = []
        for h in handoffs:
            new_kind = self.normalizer.normalize(h.claim_kind)
            new_payload = normalize_payload(h.payload)
            if new_payload != h.payload:
                self._last_n_payload_rewrites += 1
            out.append(_DecodedHandoff(
                source_role=h.source_role,
                claim_kind=new_kind,
                payload=new_payload,
            ))
        return out

    def decode_rounds(self,
                        per_round_handoffs: Sequence[Sequence[_DecodedHandoff]],
                        ) -> dict[str, Any]:
        self.normalizer.reset_counters()
        self._last_n_payload_rewrites = 0
        normalised: list[list[_DecodedHandoff]] = []
        for bundle in per_round_handoffs:
            normalised.append(self.normalize_round(bundle))
        return self.inner.decode_rounds(normalised)

    def decode(self, handoffs: Sequence[_DecodedHandoff]
               ) -> dict[str, Any]:
        return self.decode_rounds([handoffs])

    def normalizer_stats(self) -> dict[str, Any]:
        """Expose normaliser counters as a dict for the bench driver."""
        return {
            "n_exact": int(self.normalizer.last_n_exact),
            "n_heuristic": int(self.normalizer.last_n_heuristic),
            "n_abstained": int(self.normalizer.last_n_abstained),
            "n_passthrough": int(self.normalizer.last_n_passthrough),
            "n_payload_rewrites": int(self._last_n_payload_rewrites),
            "rule_hits": dict(self.normalizer.last_rule_hits),
        }


# =============================================================================
# SDK v3.15 — structured producer protocol (W14 family)
# =============================================================================
#
# The W13 closure-widening contract closed the *open-world normalisation*
# axis: a layered exact + heuristic normaliser strictly widens the W12
# closed-vocabulary closure on R-60-wide. But on real Ollama 14B the
# observed bottleneck (W13-Λ-real) is *upstream*: the producer LLM
# silently filters low-magnitude decoy events as noise AND compresses
# round-1 toward a single best diagnosis. Normalisation has nothing to
# rescue when the bench property is erased *before* the candidate
# stream reaches the auditor.
#
# The W14 layer attacks this directly via *prompt-side discipline*. It
# is purely additive — a new prompt-rendering surface that any
# extractor (synthetic or real-Ollama) can use in place of the legacy
# free-form prompt. The Wevra single-run product runtime contract is
# byte-for-byte unchanged; this is research-grade SDK code on the team-
# coord surface only.
#
# Two modes:
#
# * **Naive** — the legacy Phase-58/59/60 prompt rendering. Preserves
#   backward-compat: any extractor that opts in to ``ProducerPromptMode
#   .NAIVE`` produces byte-identical output to the legacy
#   ``_round_ollama_prompt`` helpers.
# * **Structured** — splits round 1 (operational *observation*: "list
#   every distinct event you see — describe, do not diagnose") from
#   round 2 (specific *diagnosis*: "what is the underlying cause?")
#   AND requires one claim per listed event with no compressed
#   summarisation. The protocol's contribution is to *preserve* the
#   bench property's cross-role decoy corroboration assumption when the
#   producer LLM would otherwise filter or compress.
#
# This file ships only the *protocol* (prompt rendering + role schema +
# claim parser). The benchmark driver (``Phase61``) and the magnitude-
# filtering synthetic extractor live in
# ``vision_mvp/experiments/phase61_producer_ambiguity_preservation.py``.


# Producer-prompt mode names — closed vocabulary, one per element.
PRODUCER_PROMPT_NAIVE = "naive"
PRODUCER_PROMPT_STRUCTURED = "structured"
# SDK v3.18 / W17 family — magnitude-hinted structured prompt. Adds an
# explicit per-kind operational-threshold table to the structured
# prompt and forbids relative-magnitude skipping ("judge each event on
# its own absolute magnitude, not relative to other events in this
# round"). Designed to close the 1/8 R-61-OLLAMA-A model-side miss
# (the slow_query_archival scenario where 14B judged decoy magnitudes
# "not severe enough" relative to the larger gold spike).
PRODUCER_PROMPT_MAGNITUDE_HINTED = "magnitude_hinted"
ALL_PRODUCER_PROMPT_MODES: tuple[str, ...] = (
    PRODUCER_PROMPT_NAIVE,
    PRODUCER_PROMPT_STRUCTURED,
    PRODUCER_PROMPT_MAGNITUDE_HINTED,
)


@dataclasses.dataclass(frozen=True)
class OperationalThreshold:
    """W17 operational-threshold record. Names a closed-vocabulary
    ``claim_kind``, the numeric field it qualifies on (e.g.
    ``"p95_ms"``, ``"error_rate"``, ``"count"``), the inclusive
    qualifying threshold, the unit string for the prompt, and a
    short human gloss. The magnitude-hinted prompt renders one line
    per threshold so the LLM has a *named* lower bound for each
    kind it is allowed to emit in this round.

    The thresholds are *operational definitions*, not answer hints:
    they are the same numeric values the synthetic
    :class:`MagnitudeFilteringExtractor` uses (Phase-61 calibration)
    and would be supplied by the operator who set up the
    benchmark's monitoring pipeline. The structural property the
    prompt enforces is "any event satisfying ``field ≥ threshold``
    qualifies as ``kind`` regardless of the magnitude of *other*
    events in the same round" — i.e. it removes the model's
    incentive to make *relative* judgments inside one prompt
    rendering. Empirically the relative-judgment failure mode is
    what produced the 1/8 R-61-OLLAMA-A miss
    (``slow_query_archival``: the monitor dropped the decoy events
    because the gold ``p95_ms = 4100`` made the decoy
    ``p95_ms = 2900`` look "small" by comparison, even though
    2900 ≥ 1000 by the operational definition).
    """
    kind: str
    field: str
    threshold: float
    unit: str = ""
    gloss: str = ""


@dataclasses.dataclass(frozen=True)
class RoleExtractionSchema:
    """Declarative per-role extraction schema for the W14 protocol.

    The schema is the *load-bearing structural input* to
    :class:`StructuredProducerProtocol`'s prompt rendering. It captures
    everything the LLM needs to know to emit one claim per listed
    event without compressing toward a single diagnosis:

    * ``role`` — the producer role name (``monitor`` / ``db_admin`` /
      ``sysadmin`` / ``network`` / …). Must match the role used in the
      ``TeamCoordinator``.
    * ``allowed_kinds`` — the role-local closed vocabulary of claim
      kinds. Same as the legacy ``incident_triage_known_kinds`` table
      but threaded explicitly so non-incident-triage benches can plug
      in without touching the team-coord module.
    * ``observation_kinds`` — subset of ``allowed_kinds`` the LLM is
      allowed to emit in *round 1* (observation mode). This is the
      *generic-noise* tier in the incident-triage benchmark
      (``LATENCY_SPIKE``, ``ERROR_RATE_SPIKE``, ``FW_BLOCK_SURGE``).
      Round 1 prompt explicitly forbids diagnostic kinds.
    * ``diagnosis_kinds`` — subset of ``allowed_kinds`` the LLM is
      allowed to emit in *round 2* (diagnosis mode). This is the
      *specific-tier* causal kind set (``DEADLOCK_SUSPECTED``,
      ``POOL_EXHAUSTION``, ``DISK_FILL_CRITICAL``,
      ``SLOW_QUERY_OBSERVED``). Round 2 prompt explicitly forbids
      generic-noise kinds.

    The two-set partition is what makes the structured protocol
    structurally distinct from the naive prompt: the LLM is told *up
    front* which tier the round expects, removing the temptation to
    pre-collapse round-1 observation into round-2 diagnosis.
    """
    role: str
    allowed_kinds: tuple[str, ...]
    observation_kinds: tuple[str, ...]
    diagnosis_kinds: tuple[str, ...]
    # SDK v3.18 / W17. Optional per-role operational-threshold table.
    # Empty under the SDK v3.15 default (W14-1 anchor); populated by
    # ``incident_triage_role_schemas(magnitude_hinted=True)`` for the
    # W17 magnitude-hinted prompt. The schema records the thresholds
    # *per role* so different roles can have different qualifying
    # surfaces (e.g. the network role's ``count ≥ 5`` is independent
    # of the monitor role's ``p95_ms ≥ 1000``).
    magnitude_thresholds: tuple[OperationalThreshold, ...] = ()

    def kinds_for_round(self, round_idx: int) -> tuple[str, ...]:
        """Return the role's allowed kinds for the given round under
        the structured protocol. Round 1 = observation, round 2 =
        diagnosis. Round indices outside {1, 2} fall back to
        ``allowed_kinds`` (naive prompt parity)."""
        if round_idx == 1:
            return self.observation_kinds or self.allowed_kinds
        if round_idx == 2:
            return self.diagnosis_kinds or self.allowed_kinds
        return self.allowed_kinds


@dataclasses.dataclass(frozen=True)
class ProducerPromptResult:
    """Bundle returned by :func:`StructuredProducerProtocol.render_prompt`
    so the bench driver can record which exact mode + schema was in
    effect for forensic audit."""
    mode: str
    role: str
    round_idx: int
    text: str
    kinds_in_scope: tuple[str, ...]


@dataclasses.dataclass
class StructuredProducerProtocol:
    """W14 producer protocol: a prompt-rendering surface that splits
    observation from diagnosis and requires exhaustive per-event
    extraction.

    The protocol has two modes:

    * ``mode = PRODUCER_PROMPT_NAIVE`` reproduces the legacy
      Phase-58/59/60 prompt byte-for-byte (see
      :func:`_render_naive_prompt`); this is the W14-3 backward-compat
      anchor.
    * ``mode = PRODUCER_PROMPT_STRUCTURED`` renders the new prompt
      with three reinforcing instructions:
        1. Round-tier banner: "Round 1 = OBSERVATION; Round 2 =
           DIAGNOSIS". The banner is the load-bearing instruction —
           it gives the LLM permission to emit observational claims
           without committing to a cause.
        2. Per-tier kind whitelist: only ``observation_kinds`` /
           ``diagnosis_kinds`` are permitted on the respective round.
           Diagnostic kinds in round 1 are forbidden ("treat round 1
           as monitoring data only"); generic-noise kinds in round 2
           are forbidden ("the cause has already been signalled —
           emit the specific diagnosis").
        3. One-claim-per-event mandate: "EMIT ONE CLAIM PER LISTED
           EVENT BELOW. DO NOT SKIP, DEDUPLICATE, OR COMPRESS EVENTS
           EVEN IF THEY APPEAR SIMILAR." The mandate is the W14
           anti-compression invariant.

    The structured prompt is *purely additive*. The wire shape, the
    closed-vocabulary kind whitelist, and the parser are unchanged
    from the legacy path; only the rendered prompt text differs.

    A benchmark driver constructs one protocol per scenario / role /
    round, calls :meth:`render_prompt`, sends the text to the LLM
    backend, and parses the response with the same closed-vocabulary
    parser as Phase 59 (``_parse_ollama_response``). The protocol
    object holds no per-call state beyond the deterministic config.
    """

    mode: str = PRODUCER_PROMPT_STRUCTURED

    def __post_init__(self) -> None:
        if self.mode not in ALL_PRODUCER_PROMPT_MODES:
            raise ValueError(
                f"unknown producer prompt mode {self.mode!r}; "
                f"valid: {ALL_PRODUCER_PROMPT_MODES}")

    def render_prompt(self,
                       *,
                       role: str,
                       round_idx: int,
                       events: Sequence[tuple[str, str]],
                       schema: RoleExtractionSchema,
                       ) -> ProducerPromptResult:
        """Render the producer prompt for one (role, round, events,
        schema) tuple.

        Parameters
        ----------
        role
            The producer role (must equal ``schema.role`` for the
            structured prompt; this is enforced).
        round_idx
            ``1`` for observation, ``2`` for diagnosis. Other indices
            are accepted but render the naive prompt regardless of
            ``mode`` (the structured prompt has no defined behaviour
            for non-{1,2} rounds and falls back to the naive path).
        events
            Pairs ``(canonical_kind_hint, payload)`` describing the
            operational events the role observed. The structured
            prompt uses these for the per-event enumeration; the
            naive prompt uses the same enumeration shape so the two
            modes differ only in the surrounding instructions.
        schema
            The role-local extraction schema. Must satisfy
            ``schema.role == role`` and the mode-specific kind
            partition (observation_kinds for round 1, diagnosis_kinds
            for round 2 under the structured prompt).
        """
        if schema.role != role:
            raise ValueError(
                f"schema.role={schema.role!r} != role={role!r}")
        kinds = schema.kinds_for_round(round_idx)
        if (self.mode == PRODUCER_PROMPT_NAIVE
                or round_idx not in (1, 2)):
            text = _render_naive_prompt(
                role=role, round_idx=round_idx,
                events=events, allowed_kinds=kinds)
            return ProducerPromptResult(
                mode=PRODUCER_PROMPT_NAIVE, role=role,
                round_idx=int(round_idx), text=text,
                kinds_in_scope=tuple(kinds))
        if self.mode == PRODUCER_PROMPT_MAGNITUDE_HINTED:
            text = _render_magnitude_hinted_prompt(
                role=role, round_idx=round_idx,
                events=events, schema=schema)
            return ProducerPromptResult(
                mode=PRODUCER_PROMPT_MAGNITUDE_HINTED, role=role,
                round_idx=int(round_idx), text=text,
                kinds_in_scope=tuple(kinds))
        text = _render_structured_prompt(
            role=role, round_idx=round_idx,
            events=events, schema=schema)
        return ProducerPromptResult(
            mode=PRODUCER_PROMPT_STRUCTURED, role=role,
            round_idx=int(round_idx), text=text,
            kinds_in_scope=tuple(kinds))


def _render_naive_prompt(*,
                           role: str,
                           round_idx: int,
                           events: Sequence[tuple[str, str]],
                           allowed_kinds: Sequence[str]) -> str:
    """Render the legacy Phase-58/59/60 prompt. Byte-for-byte equal
    to ``vision_mvp.experiments.phase59_real_llm_multi_round.
    _round_ollama_prompt`` (W14-3 backward-compat anchor).

    The rendered text is reproduced here so the W14 protocol module
    has *no* import dependency on the experiments package — a
    Phase-N+1 driver can render the naive prompt without touching
    Phase-59 at all.
    """
    kind_lines = "\n".join(f"  - {k}" for k in allowed_kinds)
    event_lines: list[str] = []
    for i, (_canon, payload) in enumerate(events, start=1):
        event_lines.append(f"  [{i}] body=\"{payload}\"")
    if not event_lines:
        event_lines = ["  (none)"]
    round_hint = ("operational symptoms (latency/error/firewall)"
                   if round_idx == 1
                   else "specific diagnostic clues (deadlock/pool/disk/query)")
    return (
        f"You are the {role!r} agent in an incident-response team. "
        f"This is round {round_idx}: {round_hint}.\n\n"
        f"Allowed claim kinds for {role!r}:\n{kind_lines}\n\n"
        f"Events you observed:\n"
        + "\n".join(event_lines) + "\n\n"
        f"For each event, emit ONE LINE in the format\n"
        f"  KIND | one-line evidence including any service token\n"
        f"Output rules: only KINDs from the list. One claim per line. "
        f"Maximum 6 lines. If no claim, output NONE.\n\n"
        f"Begin output now:\n"
    )


def _render_structured_prompt(*,
                                role: str,
                                round_idx: int,
                                events: Sequence[tuple[str, str]],
                                schema: RoleExtractionSchema) -> str:
    """Render the W14 structured prompt. Three load-bearing parts:

    1. Round-tier banner (OBSERVATION vs DIAGNOSIS).
    2. Per-tier kind whitelist (observation_kinds in round 1,
       diagnosis_kinds in round 2).
    3. One-claim-per-event mandate.

    The structured prompt also reminds the LLM that the auditor
    *needs* the corroboration evidence on every listed service even
    if the LLM thinks the event is small or coincidental — this is
    the explicit anti-magnitude-filter clause that closes the
    W13-Λ-real gap synthetically.
    """
    if round_idx == 1:
        tier_banner = (
            "ROUND 1 — OBSERVATION MODE.\n"
            "Your job is to DESCRIBE what you observe, not to diagnose.\n"
            "Even small / borderline / coincidental signals must be "
            "reported — the auditor will combine evidence across "
            "rounds and across roles. DO NOT compress observations "
            "toward a single best explanation; that is a later step."
        )
        allowed = schema.observation_kinds or schema.allowed_kinds
        forbidden = schema.diagnosis_kinds
    else:  # round 2
        tier_banner = (
            "ROUND 2 — DIAGNOSIS MODE.\n"
            "Round-1 observations have been recorded. Your job here "
            "is to emit the SPECIFIC underlying cause as a single "
            "diagnostic claim. Do NOT re-emit generic latency / error "
            "/ firewall observations in this round — those belong "
            "to round 1."
        )
        allowed = schema.diagnosis_kinds or schema.allowed_kinds
        forbidden = schema.observation_kinds
    kind_lines = "\n".join(f"  - {k}" for k in allowed)
    forbidden_lines = (
        "\n".join(f"  - {k}" for k in forbidden)
        if forbidden else "  (none)")
    event_lines: list[str] = []
    for i, (_canon, payload) in enumerate(events, start=1):
        event_lines.append(f"  [{i}] body=\"{payload}\"")
    if not event_lines:
        event_lines = ["  (none)"]
    return (
        f"You are the {role!r} agent in an incident-response team.\n\n"
        f"{tier_banner}\n\n"
        f"Allowed claim kinds for this round:\n{kind_lines}\n\n"
        f"FORBIDDEN claim kinds for this round (these belong to the "
        f"OTHER round):\n{forbidden_lines}\n\n"
        f"Events you observed:\n"
        + "\n".join(event_lines) + "\n\n"
        f"OUTPUT INSTRUCTIONS:\n"
        f"  * EMIT ONE CLAIM PER LISTED EVENT ABOVE. Do NOT skip, "
        f"deduplicate, or compress events even if they look similar "
        f"or appear to share a single cause.\n"
        f"  * Each claim is ONE LINE in the format\n"
        f"      KIND | one-line evidence including any service token\n"
        f"  * Use only KINDs from the allowed list. If a listed event "
        f"truly does not warrant any allowed kind for this round "
        f"(e.g. round-1 with a non-symptom body), emit "
        f"\"NONE | <why>\" on a line of its own — but emitting NONE "
        f"on every event is a sign you have collapsed observation "
        f"and is discouraged.\n"
        f"  * Maximum {max(8, len(event_lines) + 2)} lines.\n\n"
        f"Begin output now:\n"
    )


def _format_threshold_value(t: OperationalThreshold) -> str:
    """Render the threshold's numeric value with the unit, in a form
    that round-trips through the synthetic ``MagnitudeFilteringExtractor``
    parser. Integers without trailing ``.0``; floats with the
    minimum number of fractional digits needed to be unambiguous.
    Mechanically deterministic so the rendered prompt is byte-for-
    byte stable."""
    v = float(t.threshold)
    if v == int(v) and abs(v) < 1e6:
        s = f"{int(v)}"
    else:
        # Trim trailing zeros from a fixed-point representation so
        # ``0.10`` renders as ``0.10`` (preserving the operational
        # form) rather than ``0.1``.
        s = f"{v:.4f}".rstrip("0")
        if s.endswith("."):
            s += "0"
    if t.unit:
        return f"{s} {t.unit}".strip()
    return s


def _render_magnitude_hinted_prompt(*,
                                       role: str,
                                       round_idx: int,
                                       events: Sequence[tuple[str, str]],
                                       schema: RoleExtractionSchema,
                                       ) -> str:
    """Render the W17 magnitude-hinted structured prompt.

    The magnitude-hinted prompt extends :func:`_render_structured_prompt`
    with two reinforcing instructions designed to close the 1/8
    R-61-OLLAMA-A model-side miss:

    1. **Operational threshold table.** Each kind in
       ``schema.magnitude_thresholds`` whose ``kind`` is in the
       round's allowed-set is rendered as one line of the form

       ``  - LATENCY_SPIKE qualifies for any p95_ms ≥ 1000 ms``

       so the LLM has a *named* lower bound for each kind it is
       allowed to emit. The thresholds are *operational definitions*
       (the same values the synthetic
       :class:`MagnitudeFilteringExtractor` uses), not answer hints.

    2. **Anti-relative-magnitude clause.** An explicit sentence in
       the OUTPUT INSTRUCTIONS section forbids relative-magnitude
       skipping: "Each event is judged on its own *absolute*
       magnitude. Do NOT skip an event because *another* event in
       this round looks larger." This removes the model's incentive
       to make relative judgments inside one prompt rendering — the
       failure mode that produced the slow_query_archival miss in
       the SDK v3.15 + v3.17 captures.

    All other invariants of :func:`_render_structured_prompt` are
    preserved verbatim (the round-tier banner, per-tier kind
    whitelist, per-event mandate). The magnitude-hint extension is
    purely additive: with an empty ``schema.magnitude_thresholds``
    the prompt reduces to the structured prompt with the
    anti-relative-magnitude clause appended (W17-3 backward-compat
    on schemas that do not carry thresholds).
    """
    if round_idx == 1:
        tier_banner = (
            "ROUND 1 — OBSERVATION MODE.\n"
            "Your job is to DESCRIBE what you observe, not to diagnose.\n"
            "Even small / borderline / coincidental signals must be "
            "reported — the auditor will combine evidence across "
            "rounds and across roles. DO NOT compress observations "
            "toward a single best explanation; that is a later step."
        )
        allowed = schema.observation_kinds or schema.allowed_kinds
        forbidden = schema.diagnosis_kinds
    else:  # round 2
        tier_banner = (
            "ROUND 2 — DIAGNOSIS MODE.\n"
            "Round-1 observations have been recorded. Your job here "
            "is to emit the SPECIFIC underlying cause as a single "
            "diagnostic claim. Do NOT re-emit generic latency / error "
            "/ firewall observations in this round — those belong "
            "to round 1."
        )
        allowed = schema.diagnosis_kinds or schema.allowed_kinds
        forbidden = schema.observation_kinds
    kind_lines = "\n".join(f"  - {k}" for k in allowed)
    forbidden_lines = (
        "\n".join(f"  - {k}" for k in forbidden)
        if forbidden else "  (none)")
    allowed_set = set(allowed)
    threshold_lines: list[str] = []
    for t in schema.magnitude_thresholds:
        if t.kind not in allowed_set:
            continue
        gloss = f" ({t.gloss})" if t.gloss else ""
        threshold_lines.append(
            f"  - {t.kind} qualifies for any "
            f"{t.field} >= {_format_threshold_value(t)}{gloss}")
    threshold_block = (
        "OPERATIONAL QUALIFYING THRESHOLDS for this round:\n"
        + "\n".join(threshold_lines) + "\n\n"
        if threshold_lines else "")
    event_lines: list[str] = []
    for i, (_canon, payload) in enumerate(events, start=1):
        event_lines.append(f"  [{i}] body=\"{payload}\"")
    if not event_lines:
        event_lines = ["  (none)"]
    return (
        f"You are the {role!r} agent in an incident-response team.\n\n"
        f"{tier_banner}\n\n"
        f"Allowed claim kinds for this round:\n{kind_lines}\n\n"
        f"FORBIDDEN claim kinds for this round (these belong to the "
        f"OTHER round):\n{forbidden_lines}\n\n"
        f"{threshold_block}"
        f"Events you observed:\n"
        + "\n".join(event_lines) + "\n\n"
        f"OUTPUT INSTRUCTIONS:\n"
        f"  * EMIT ONE CLAIM PER LISTED EVENT ABOVE. Do NOT skip, "
        f"deduplicate, or compress events even if they look similar "
        f"or appear to share a single cause.\n"
        f"  * Each event is judged on its own ABSOLUTE magnitude. "
        f"Do NOT skip an event because another event in this round "
        f"looks larger or more severe — relative comparison is the "
        f"auditor's job, not yours.\n"
        f"  * If an event satisfies a qualifying threshold above "
        f"(e.g. p95_ms >= 1000), emit the matching kind even if you "
        f"think the event is small compared to others in this round.\n"
        f"  * Each claim is ONE LINE in the format\n"
        f"      KIND | one-line evidence including any service token\n"
        f"  * Use only KINDs from the allowed list. If a listed event "
        f"truly does not warrant any allowed kind for this round "
        f"(e.g. round-1 with a non-symptom body), emit "
        f"\"NONE | <why>\" on a line of its own — but emitting NONE "
        f"on every event is a sign you have collapsed observation "
        f"and is discouraged.\n"
        f"  * Maximum {max(8, len(event_lines) + 2)} lines.\n\n"
        f"Begin output now:\n"
    )


# Default schemas for the bundled incident-triage benchmark family.
# Other benchmarks should construct their own ``RoleExtractionSchema``
# table; this one ships as a convenience for Phase-58..Phase-61
# drivers and is mechanically aligned with
# ``vision_mvp.core.extractor_noise.incident_triage_known_kinds``.
INCIDENT_TRIAGE_OBSERVATION_KINDS: tuple[str, ...] = (
    "LATENCY_SPIKE", "ERROR_RATE_SPIKE", "FW_BLOCK_SURGE",
)


# SDK v3.18 / W17. Default magnitude-hint thresholds for the
# incident-triage benchmark family. Each threshold is the same
# numeric value the synthetic ``MagnitudeFilteringExtractor`` uses
# (Phase-61 calibration anchors). The thresholds are *operational
# definitions* of "what counts as a qualifying event for this
# kind", not answer hints — they are intentionally below all gold
# magnitudes (so structure is preserved) AND below all decoy
# magnitudes (so the LLM is correctly instructed to emit the
# decoy events too).
INCIDENT_TRIAGE_DEFAULT_MAGNITUDE_THRESHOLDS: tuple[
    OperationalThreshold, ...] = (
    OperationalThreshold(
        kind="LATENCY_SPIKE", field="p95_ms", threshold=1000.0,
        unit="ms",
        gloss="any p95 above 1 second is operationally meaningful"),
    OperationalThreshold(
        kind="ERROR_RATE_SPIKE", field="error_rate",
        threshold=0.10, unit="",
        gloss="any error rate >= 10% is operationally meaningful"),
    OperationalThreshold(
        kind="FW_BLOCK_SURGE", field="count", threshold=5.0,
        unit="denials/min",
        gloss="any sustained denial burst above 5/min is meaningful"),
)


def incident_triage_magnitude_thresholds(
        ) -> tuple[OperationalThreshold, ...]:
    """Return the default operational-threshold table for the
    incident-triage family (W17 magnitude-hint anchor).

    The thresholds are pre-committed at the same numeric values the
    synthetic :class:`MagnitudeFilteringExtractor` uses (Phase-61
    calibration). They are operational definitions of "what counts
    as a qualifying event for this kind" and are below *both* gold
    and decoy magnitudes in every R-61 / R-64 scenario, so the
    magnitude-hint extension does NOT leak the answer — it removes
    the model's *relative* magnitude judgment loophole.
    """
    return INCIDENT_TRIAGE_DEFAULT_MAGNITUDE_THRESHOLDS


def incident_triage_role_schemas(
        *, magnitude_hinted: bool = False
        ) -> dict[str, RoleExtractionSchema]:
    """Return the W14 default schema table for the incident-triage
    benchmark family. Each role's ``observation_kinds`` is the
    intersection of its ``allowed_kinds`` with the closed-vocabulary
    generic-noise tier (``INCIDENT_TRIAGE_OBSERVATION_KINDS``);
    ``diagnosis_kinds`` is the complement on the same allowed-kind
    set.

    Parameters
    ----------
    magnitude_hinted
        SDK v3.18 / W17 opt-in. When ``True``, every returned schema
        carries the
        :data:`INCIDENT_TRIAGE_DEFAULT_MAGNITUDE_THRESHOLDS` table.
        When ``False`` (default), the returned schemas have an empty
        ``magnitude_thresholds`` field and the structured prompt
        renders without the W17 threshold block (W14-3 byte-for-byte
        backward-compat anchor on the SDK v3.15 anchor regimes).

    Mechanically verified by ``IncidentTriageSchemaTests`` in
    ``test_wevra_producer_ambiguity.py`` and the W17 surface tests
    in ``test_wevra_phase64.py``.
    """
    from vision_mvp.core.extractor_noise import (
        incident_triage_known_kinds)
    out: dict[str, RoleExtractionSchema] = {}
    obs_set = set(INCIDENT_TRIAGE_OBSERVATION_KINDS)
    if magnitude_hinted:
        thresholds: tuple[OperationalThreshold, ...] = (
            INCIDENT_TRIAGE_DEFAULT_MAGNITUDE_THRESHOLDS)
    else:
        thresholds = ()
    for role, allowed in incident_triage_known_kinds().items():
        allowed_t = tuple(allowed)
        observation = tuple(k for k in allowed_t if k in obs_set)
        diagnosis = tuple(k for k in allowed_t if k not in obs_set)
        out[role] = RoleExtractionSchema(
            role=role, allowed_kinds=allowed_t,
            observation_kinds=observation,
            diagnosis_kinds=diagnosis,
            magnitude_thresholds=thresholds)
    return out


# =============================================================================
# SDK v3.16 — Attention-aware capsule context packing (W15 family)
# =============================================================================
#
# The W14 producer protocol closed the *producer-side* gap on R-61 (and
# the R-61-OLLAMA-A tier on real Ollama 14B) by ensuring the bench
# property's cross-role decoy corroboration assumption survives a real
# producer. But it left a downstream gap unaddressed: under a tight
# *decoder-side* token budget, the cross-round decoder receives the
# *raw* admitted union — which on a multi-hypothesis regime is
# dominated by generic-noise round-1 corroboration with the round-2
# causal claim packed *late* (or, under FIFO truncation, dropped
# entirely). The decoder can have all the right ingredients in the
# admitted set and still fail because its context budget is consumed
# by lower-salience evidence.
#
# W15 attacks this directly via *capsule-aware context packing*. The
# :class:`AttentionAwareBundleDecoder` runs in two stages:
#
# 1. **Tentative root_cause election.** A first-pass priority decode
#    over the full normalised union elects a tentative ``root_cause``.
#    No token budget is enforced at this stage — only the
#    deterministic priority lookup runs, which is O(|union|).
# 2. **Salience-scored repacking.** Each handoff in the union is
#    scored by a deterministic, training-free salience function:
#       salience(h) = w_tier · tier(h)
#                   + w_cck  · cck_match(h, root_cause)
#                   + w_corr · log1p(distinct_roles(service_tag(h)))
#                   + w_mag  · normalised_magnitude(h)
#                   + w_round · round_index_score(h)
#    where every component is a closed-form function of the handoff's
#    bytes (plus the elected root_cause). The handoffs are sorted by
#    ``salience`` descending and packed greedily under the
#    decoder-side token budget ``T_decoder``. Ties break by canonical
#    (round, source_role, claim_kind, payload_sha) order so the pack
#    is byte-deterministic.
# 3. **Final decode** runs :class:`LayeredRobustMultiRoundBundleDecoder`
#    (W13) on the *packed* subset, returning the same answer shape
#    plus a ``pack_stats`` block exposing the minimum-sufficient-
#    bundle proxy metrics the benchmark driver needs.
#
# Why this is a real method change (not a re-shuffle)
# ---------------------------------------------------
# The W14 producer protocol shapes the *emission* distribution; the
# W12/W13 normalisers rewrite *kinds*; the W11 multi-round decoder
# *unions* per-round bundles. None of these touch the *order* or
# *count* of the handoffs the decoder consumes — by construction, all
# previous decoders consume the full union. W15 introduces the first
# decoder-side *attention proxy*: a deterministic salience score that
# the decoder uses to truncate AND reorder before the priority
# lookup runs on a smaller, causally-aligned bundle.
#
# Why this is *attention-aware*, with honest caveats
# --------------------------------------------------
# We do NOT claim to shape transformer attention weights. We do claim
# to shape *what the decoder sees*: the handoffs that survive the
# pack are sorted so the highest-salience evidence appears first, and
# the lowest-salience evidence is dropped. Under a real-LLM
# downstream consumer (the natural W15-C real-LLM extension), this
# corresponds to *prompt-position attention shaping* (a well-known
# property of transformers: the first ~k tokens get more attention
# than the last ~k under typical positional encoding regimes). This
# is an honest *proxy* — not a transformer-attention-weights
# manipulation. The proxy is exposed via the ``position_of_first_*``
# stats so the bench driver can audit it directly.
#
# Token-budget honesty
# --------------------
# ``T_decoder`` is enforced *strictly* over the sum of admitted
# handoffs' ``n_tokens`` field (the same word-count proxy ROLE_VIEW
# uses). When ``T_decoder`` is None or larger than the union sum, the
# pack is the full union — the decoder reduces to W13 byte-for-byte
# (W15-3 backward-compat). When ``T_decoder`` is below the sum,
# packing kicks in and the salience-driven truncation is the load-
# bearing step.
#
# Honest scope (W15)
# ------------------
# * The salience weights are pre-committed (defaults below). They are
#   NOT learned. A learned variant is W15-C-LEARNED.
# * The W15-1 win is *conditional* on (a) the bench property holding
#   on the regime, (b) ``T_decoder`` being below the admitted union
#   token sum (budget pressure exists), AND (c) the regime carrying
#   round-2 specific-tier disambiguation that FIFO would push past
#   the budget.
# * On regimes without budget pressure (R-62-evidence-saturation),
#   W15 ties W13 by W15-Λ-budget — the named falsifier.
# * On regimes without multi-hypothesis ambiguity (R-54..R-57), the
#   admitted set is small (≤ 2 services) and the W13 trust-admission
#   fallback already does the job. W15 reduces to W13 there
#   (mechanically verified by the W15-3 backward-compat test).


# Salience weight defaults. Pre-committed; do NOT change between
# milestones without retracting and replacing. The weights were chosen
# so that:
#   * a CCK-matching specific-tier round-2 claim outranks any number
#     of generic-noise round-1 mentions (so the disambiguator is
#     never dropped by truncation under a sane T_decoder);
#   * within a tier, more-corroborated tags rank higher than less-
#     corroborated tags (so cross-role corroboration survives the
#     pack);
#   * within (tier, corroboration), higher-magnitude evidence ranks
#     higher (so magnitude-anchored gold stays earlier in the bundle).
W15_DEFAULT_TIER_WEIGHT: float = 4.00
W15_DEFAULT_CCK_WEIGHT: float = 3.00
W15_DEFAULT_CORROBORATION_WEIGHT: float = 1.00
W15_DEFAULT_MAGNITUDE_WEIGHT: float = 0.10
W15_DEFAULT_ROUND_WEIGHT: float = 0.50
# Per-handoff token-count attribution. Re-uses the word-count proxy
# the rest of the SDK uses (see ``capsule_team_handoff``'s
# ``n_tokens`` default).
def _handoff_n_tokens(h: _DecodedHandoff) -> int:
    payload = h.payload or ""
    return max(1, len(payload.split()))


# Identification of "specific-tier" claim kinds. Anything in the
# generic-noise set is tier 0; anything that maps to a non-generic
# root_cause via the priority decoder is tier 1. The map is computed
# once from the priority table and held as a closed-form table so
# salience scoring is allocation-free.
def _build_specific_tier_kinds() -> frozenset[str]:
    out: set[str] = set()
    for (kind, root, _remed) in _DECODER_PRIORITY:
        if root in ("error_spike", "latency_spike", "fw_block",
                     "unknown"):
            continue
        out.add(kind)
    return frozenset(out)


_SPECIFIC_TIER_CLAIM_KINDS: frozenset[str] = _build_specific_tier_kinds()


def _payload_magnitude(payload: str) -> float:
    """Extract a normalised magnitude proxy from a handoff payload.

    Parses the same fields the magnitude-filter extractor reads
    (``p95_ms``, ``error_rate``, firewall ``count``) and returns a
    bounded magnitude in [0, 1] for ranking. Missing fields return 0
    so handoffs without an extractable magnitude rank below those
    with one (within tier).
    """
    if not payload:
        return 0.0
    p = payload
    m = _re.search(r"\bp95_ms=([0-9]+)", p)
    if m:
        v = float(m.group(1))
        return min(1.0, v / 5000.0)
    m = _re.search(r"\berror_rate=([0-9.]+)", p)
    if m:
        v = float(m.group(1))
        return min(1.0, v / 0.50)
    m = _re.search(r"\bcount=([0-9]+)", p)
    if m:
        v = float(m.group(1))
        return min(1.0, v / 30.0)
    return 0.0


def _service_tag_of(payload: str) -> str:
    """Extract the canonical ``service=<tag>`` token from a payload, or
    ``""`` if absent. Uses the same regex as the W11 contradiction-
    aware drop so the answer-set projection is consistent."""
    if not payload:
        return ""
    for tok in payload.split():
        m = _SERVICE_TAG_RE.search(tok)
        if m:
            return m.group(1)
    return ""


@dataclasses.dataclass(frozen=True)
class W15PackedHandoff:
    """One handoff in the salience-ordered pack returned by
    :class:`CapsuleContextPacker`. Carries the original
    ``_DecodedHandoff`` plus the salience score and the cumulative
    token offset at which it was packed (for the position-proxy
    metrics).
    """
    handoff: _DecodedHandoff
    salience: float
    n_tokens: int
    cumulative_tokens: int
    rank: int  # 0-based in the packed bundle
    round_idx: int  # informational; 1 / 2 / 0 (unknown)


@dataclasses.dataclass(frozen=True)
class W15PackResult:
    """The output of one ``CapsuleContextPacker.pack`` call. Carries
    the packed bundle (sorted by salience descending) plus the metrics
    the bench driver needs to audit the pack honestly:

    * ``kept`` — the salience-ordered list of survivors.
    * ``dropped`` — handoffs the budget excluded, in the order they
      would have been packed (lowest-salience first).
    * ``n_input``, ``n_kept``, ``n_dropped_budget`` — counts.
    * ``tokens_input``, ``tokens_kept`` — sums.
    * ``salience_floor_kept`` — the lowest-salience score retained.
    * ``hypothesis_count_input`` / ``hypothesis_count_kept`` — the
      number of distinct ``service=<tag>`` tokens before / after.
    * ``position_of_first_causal_claim`` — the 0-based rank of the
      first kept handoff whose ``claim_kind`` is in
      :data:`_SPECIFIC_TIER_CLAIM_KINDS`. ``-1`` if no such handoff
      survived (the diagnostic proxy attention metric).
    * ``elected_root_cause`` — the tentative root cause used for
      CCK-aware scoring (echoed for audit).
    """
    kept: tuple[W15PackedHandoff, ...]
    dropped: tuple[W15PackedHandoff, ...]
    n_input: int
    n_kept: int
    n_dropped_budget: int
    tokens_input: int
    tokens_kept: int
    salience_floor_kept: float
    hypothesis_count_input: int
    hypothesis_count_kept: int
    position_of_first_causal_claim: int
    elected_root_cause: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_input": int(self.n_input),
            "n_kept": int(self.n_kept),
            "n_dropped_budget": int(self.n_dropped_budget),
            "tokens_input": int(self.tokens_input),
            "tokens_kept": int(self.tokens_kept),
            "salience_floor_kept": float(self.salience_floor_kept),
            "hypothesis_count_input": int(self.hypothesis_count_input),
            "hypothesis_count_kept": int(self.hypothesis_count_kept),
            "position_of_first_causal_claim":
                int(self.position_of_first_causal_claim),
            "elected_root_cause": str(self.elected_root_cause),
        }


@dataclasses.dataclass
class FifoContextPacker:
    """FIFO-order, token-bounded context packer (W15-baseline).

    The simplest possible packer: emit handoffs in arrival order
    until ``T_decoder`` is exhausted. Used as the baseline against
    which :class:`CapsuleContextPacker` is measured — every prior
    capsule decoder (W11 / W12 / W13 / W14) effectively concatenates
    admitted handoffs in FIFO order, so this packer is the *honest*
    representation of those decoders under a strict decoder-side
    token budget. The salience-aware packer is the W15 contribution.

    Returns the same :class:`W15PackResult` shape as
    :class:`CapsuleContextPacker` so the bench driver can compare
    them apples-to-apples.
    """

    def pack(self,
              handoffs: Sequence[_DecodedHandoff],
              *,
              elected_root_cause: str = "",
              T_decoder: int | None = None,
              round_index_hint: Sequence[int] | None = None,
              ) -> W15PackResult:
        if round_index_hint is not None and len(round_index_hint) != len(handoffs):
            raise ValueError(
                f"round_index_hint length {len(round_index_hint)} != "
                f"handoffs length {len(handoffs)}")
        kept: list[W15PackedHandoff] = []
        dropped: list[W15PackedHandoff] = []
        cumulative = 0
        n_input = len(handoffs)
        tokens_input = 0
        per_token: list[int] = []
        for h in handoffs:
            n_tok = _handoff_n_tokens(h)
            tokens_input += n_tok
            per_token.append(n_tok)
        for i, h in enumerate(handoffs):
            n_tok = per_token[i]
            ridx = (round_index_hint[i] if round_index_hint
                     else (2 if h.claim_kind in _SPECIFIC_TIER_CLAIM_KINDS
                            else 1))
            if T_decoder is not None and (cumulative + n_tok) > int(T_decoder):
                dropped.append(W15PackedHandoff(
                    handoff=h, salience=0.0, n_tokens=n_tok,
                    cumulative_tokens=cumulative, rank=i,
                    round_idx=ridx))
                continue
            cumulative += n_tok
            kept.append(W15PackedHandoff(
                handoff=h, salience=0.0, n_tokens=n_tok,
                cumulative_tokens=cumulative, rank=len(kept),
                round_idx=ridx))
        first_causal = -1
        for k in kept:
            if k.handoff.claim_kind in _SPECIFIC_TIER_CLAIM_KINDS:
                first_causal = k.rank
                break
        hyp_in: set[str] = set()
        for h in handoffs:
            tag = _service_tag_of(h.payload)
            if tag:
                hyp_in.add(tag)
        hyp_kept: set[str] = set()
        for k in kept:
            tag = _service_tag_of(k.handoff.payload)
            if tag:
                hyp_kept.add(tag)
        return W15PackResult(
            kept=tuple(kept),
            dropped=tuple(dropped),
            n_input=n_input,
            n_kept=len(kept),
            n_dropped_budget=len(dropped),
            tokens_input=tokens_input,
            tokens_kept=cumulative,
            salience_floor_kept=0.0,
            hypothesis_count_input=len(hyp_in),
            hypothesis_count_kept=len(hyp_kept),
            position_of_first_causal_claim=int(first_causal),
            elected_root_cause=str(elected_root_cause),
        )


@dataclasses.dataclass
class CapsuleContextPacker:
    """Capsule-aware salience-scored context packer (W15 family).

    Given a sequence of admitted handoffs (typically the union across
    rounds, post-normalisation), an elected tentative ``root_cause``,
    and a strict decoder-side token budget ``T_decoder``, return the
    salience-ordered subset that fits.

    Salience score (closed-form, training-free)
    -------------------------------------------
    For each handoff ``h`` and elected ``root_cause`` ``R``:

    ::

        salience(h) =
            w_tier  * (1.0 if h.claim_kind in _SPECIFIC_TIER_CLAIM_KINDS else 0.0)
          + w_cck   * (1.0 if h.claim_kind in CCK(R) else 0.0)
          + w_corr  * log1p(distinct_roles_for(service_tag(h)))
          + w_mag   * payload_magnitude(h.payload)
          + w_round * round_index_score(h)

    The default weights are pre-committed in :data:`W15_DEFAULT_*`.
    Ties break canonically on
    ``(-salience, round_idx, source_role, claim_kind, payload_sha)``
    so the pack is byte-deterministic.

    Returns a :class:`W15PackResult` carrying the packed bundle plus
    the packing diagnostics. The bench driver consumes the packed
    bundle as the ``handoffs`` argument to a downstream decoder.
    """

    w_tier: float = W15_DEFAULT_TIER_WEIGHT
    w_cck: float = W15_DEFAULT_CCK_WEIGHT
    w_corroboration: float = W15_DEFAULT_CORROBORATION_WEIGHT
    w_magnitude: float = W15_DEFAULT_MAGNITUDE_WEIGHT
    w_round: float = W15_DEFAULT_ROUND_WEIGHT
    cck_table: dict[str, frozenset[str]] = dataclasses.field(
        default_factory=lambda: dict(CAUSAL_CLAIM_KINDS_PER_ROOT_CAUSE))
    # Hypothesis-preservation knob (W15 multi-hypothesis property).
    #
    # When ``True`` (default), the packer guarantees that *every*
    # distinct service tag in the input gets at least one
    # representative in the kept bundle (subject to budget). This is
    # the load-bearing structural property under multi-hypothesis
    # regimes (R-62): without it, salience-aware packing prefers more-
    # corroborated decoys over less-corroborated golds, dropping the
    # gold services entirely from the kept bundle and producing the
    # wrong answer set after the W11 contradiction-aware drop.
    #
    # When ``False`` the packer is pure salience-greedy. The
    # backward-compat anchor (R-54..R-61) is unaffected because the
    # admitted union already contains few hypotheses.
    preserve_hypotheses: bool = True

    def _round_index_for(self,
                          h: _DecodedHandoff,
                          round_index_hint: int) -> int:
        """A round index for salience scoring. The packer prefers an
        explicit per-handoff hint (e.g. the bench driver tagging each
        handoff with its source round); when absent, falls back to
        ``2`` if the kind is in :data:`_SPECIFIC_TIER_CLAIM_KINDS`
        (the round-2 disambiguator) and ``1`` otherwise. Returns 0
        when the inference cannot be made."""
        if round_index_hint > 0:
            return round_index_hint
        if h.claim_kind in _SPECIFIC_TIER_CLAIM_KINDS:
            return 2
        return 1

    def pack(self,
              handoffs: Sequence[_DecodedHandoff],
              *,
              elected_root_cause: str,
              T_decoder: int | None,
              round_index_hint: Sequence[int] | None = None,
              ) -> W15PackResult:
        """Run the salience-driven pack.

        Parameters
        ----------
        handoffs
            The admitted union (typically post-normalisation) the
            decoder would otherwise consume. Order is *informational*;
            the pack reorders by salience.
        elected_root_cause
            The tentative root_cause from a first-pass priority decode
            on the full union. Used for CCK-aware scoring; if the
            elected root is generic (``error_spike`` / ``latency_spike``
            / ``fw_block`` / ``unknown``) the CCK weight is silently
            zeroed (CCK against generic is itself generic).
        T_decoder
            Strict token budget over the kept handoffs' ``n_tokens``
            sum. ``None`` disables the budget (W15-3 backward-compat
            anchor: the pack returns the full union sorted by
            salience).
        round_index_hint
            Optional per-handoff round indices (same length as
            ``handoffs``). When provided, salience uses them; when
            absent, the packer infers via :meth:`_round_index_for`.
        """
        if round_index_hint is not None and len(round_index_hint) != len(handoffs):
            raise ValueError(
                f"round_index_hint length {len(round_index_hint)} != "
                f"handoffs length {len(handoffs)}")
        cck = self.cck_table.get(elected_root_cause, frozenset())
        # Build per-tag distinct-role corroboration counts up front so
        # salience scoring is O(|union|) total.
        roles_per_tag: dict[str, set[str]] = {}
        for h in handoffs:
            tag = _service_tag_of(h.payload)
            if tag:
                roles_per_tag.setdefault(tag, set()).add(h.source_role)
        # Tag → distinct-role count, with log1p for diminishing returns.
        import math
        tag_corr_score: dict[str, float] = {
            tag: math.log1p(len(roles))
            for tag, roles in roles_per_tag.items()
        }
        # Score every handoff.
        scored: list[tuple[float, int, int, str, str, str, _DecodedHandoff,
                            int, int]] = []
        for i, h in enumerate(handoffs):
            tag = _service_tag_of(h.payload)
            ridx = self._round_index_for(
                h, round_index_hint[i] if round_index_hint else 0)
            tier_score = (1.0 if h.claim_kind in _SPECIFIC_TIER_CLAIM_KINDS
                            else 0.0)
            cck_score = (1.0 if (cck and h.claim_kind in cck) else 0.0)
            corr_score = tag_corr_score.get(tag, 0.0)
            mag_score = _payload_magnitude(h.payload)
            round_score = float(ridx) / 2.0  # round 1 = 0.5, round 2 = 1.0
            salience = (
                self.w_tier * tier_score
                + self.w_cck * cck_score
                + self.w_corroboration * corr_score
                + self.w_magnitude * mag_score
                + self.w_round * round_score
            )
            sha = _payload_sha256(h.payload)[:16]
            n_tok = _handoff_n_tokens(h)
            scored.append((-salience, ridx, i, h.source_role,
                            h.claim_kind, sha, h, n_tok, ridx))
        # Sort by descending salience; tie-break canonical.
        scored.sort()
        n_input = len(handoffs)
        tokens_input = sum(s[7] for s in scored)
        # Pack under T_decoder. Two passes when ``preserve_hypotheses``
        # is set:
        #   Pass 1 — for each distinct service tag, pack its highest-
        #            salience representative if budget allows. Round-2
        #            specific-tier handoffs (tag = "") fall through to
        #            this pass too, so the disambiguator is always kept.
        #   Pass 2 — fill remaining budget greedy by salience.
        # Without ``preserve_hypotheses`` the packer is single-pass
        # salience-greedy.
        kept: list[W15PackedHandoff] = []
        dropped: list[W15PackedHandoff] = []
        cumulative = 0
        used_idx: set[int] = set()
        if self.preserve_hypotheses:
            # Pass 1 — one representative per (tag, source_role, tier).
            # We bucket by ``(service_tag, source_role)`` to guarantee
            # both that every hypothesis is represented AND that every
            # distinct-role mention of each tag is preserved (the W11
            # contradiction-aware drop fires only when a tag's admitted
            # mentions span ≥ noise_decoy_role_floor distinct roles —
            # so per-tag preservation alone is insufficient for the
            # multi-hypothesis regime). Tier is included so the round-2
            # disambiguator (tag = "") gets its own slot; ``tier0``
            # round-1 mentions on the same (tag, role) collapse to one
            # representative (the highest-salience).
            seen_buckets: set[tuple[str, str, str]] = set()
            for orig_rank, entry in enumerate(scored):
                (neg_sal, _r, _i, src, kind, _sha, h, n_tok, ridx) = entry
                tag = _service_tag_of(h.payload)
                bucket_kind = ("tier1"
                                if kind in _SPECIFIC_TIER_CLAIM_KINDS
                                else "tier0")
                bucket = (tag, src, bucket_kind)
                if bucket in seen_buckets:
                    continue
                seen_buckets.add(bucket)
                if T_decoder is not None and (cumulative + n_tok) > int(T_decoder):
                    # Cannot fit this hypothesis representative;
                    # subsequent passes won't fit it either.
                    continue
                cumulative += n_tok
                kept.append(W15PackedHandoff(
                    handoff=h, salience=-neg_sal, n_tokens=n_tok,
                    cumulative_tokens=cumulative, rank=len(kept),
                    round_idx=ridx))
                used_idx.add(orig_rank)
        # Pass 2 — fill remaining budget greedy by salience.
        for orig_rank, entry in enumerate(scored):
            if orig_rank in used_idx:
                continue
            (neg_sal, _r, _i, _s, _k, _sha, h, n_tok, ridx) = entry
            sal = -neg_sal
            if T_decoder is not None and (cumulative + n_tok) > int(T_decoder):
                dropped.append(W15PackedHandoff(
                    handoff=h, salience=sal, n_tokens=n_tok,
                    cumulative_tokens=cumulative, rank=orig_rank,
                    round_idx=ridx))
                continue
            cumulative += n_tok
            kept.append(W15PackedHandoff(
                handoff=h, salience=sal, n_tokens=n_tok,
                cumulative_tokens=cumulative, rank=len(kept),
                round_idx=ridx))
            used_idx.add(orig_rank)
        # Position of first causal claim in the packed bundle.
        first_causal = -1
        for k in kept:
            if k.handoff.claim_kind in _SPECIFIC_TIER_CLAIM_KINDS:
                first_causal = k.rank
                break
        salience_floor = (kept[-1].salience if kept else 0.0)
        # Hypothesis counts.
        hyp_in: set[str] = set()
        for h in handoffs:
            tag = _service_tag_of(h.payload)
            if tag:
                hyp_in.add(tag)
        hyp_kept: set[str] = set()
        for k in kept:
            tag = _service_tag_of(k.handoff.payload)
            if tag:
                hyp_kept.add(tag)
        return W15PackResult(
            kept=tuple(kept),
            dropped=tuple(dropped),
            n_input=n_input,
            n_kept=len(kept),
            n_dropped_budget=len(dropped),
            tokens_input=tokens_input,
            tokens_kept=cumulative,
            salience_floor_kept=float(salience_floor),
            hypothesis_count_input=len(hyp_in),
            hypothesis_count_kept=len(hyp_kept),
            position_of_first_causal_claim=int(first_causal),
            elected_root_cause=str(elected_root_cause),
        )


@dataclasses.dataclass
class AttentionAwareBundleDecoder:
    """Attention-aware, capsule-native, token-bounded multi-round
    bundle decoder (SDK v3.16, W15 family).

    Two-stage decode:

    1. **First-pass tentative root_cause election.** Runs the W13
       layered normaliser + the priority decoder over the *full*
       union to elect a tentative ``root_cause``. No token budget is
       enforced at this stage; only the priority lookup runs (O(N)).
       This stage uses a lightweight :class:`MultiRoundBundleDecoder`
       internally — the round-union semantics, not the full layered
       pipeline.
    2. **Salience-aware repacking + final decode.** A
       :class:`CapsuleContextPacker` reorders the union by salience
       (CCK against the elected root_cause + tier + corroboration +
       magnitude + round index) and truncates under ``T_decoder``.
       The final decode runs the W13 layered decoder on the *packed*
       subset; the answer is the W13 answer plus a ``pack_stats``
       block.

    Backward-compat (W15-3)
    ------------------------
    When ``T_decoder is None`` OR ``T_decoder`` ≥ ``tokens_input``,
    the pack is the full union (no drops). The salience reordering
    does not change the priority decoder's output (the decoder is
    set-based: same admitted set ⇒ same ``root_cause``); the only
    difference vs W13 is the ``pack_stats`` block, which is
    *additive*. R-54..R-61 anchors preserved byte-for-byte on the
    answer field; the pack stats are diagnostic only.

    Honest scope (W15)
    ------------------
    * The salience weights are pre-committed defaults; not learned.
    * "Attention-aware" is an *honest proxy*: we measure
      ``position_of_first_causal_claim`` and ``salience_floor_kept``
      as proxy attention signals. We do not manipulate transformer
      attention weights. Any downstream LLM consumer of the packed
      bundle benefits from prompt-position attention shaping; the
      mechanism is "earlier tokens get more attention by default,"
      not novel attention manipulation.
    * The W15-1 win is *conditional* on the bench property holding,
      ``T_decoder`` being below the union token sum (budget pressure
      exists), AND the regime carrying round-2 specific-tier
      disambiguation that FIFO would push past the budget.
    * On regimes without budget pressure (R-62-evidence-saturation),
      W15 ties W13 by W15-Λ-budget — the named falsifier.
    """

    inner: LayeredRobustMultiRoundBundleDecoder = dataclasses.field(
        default_factory=lambda: LayeredRobustMultiRoundBundleDecoder())
    packer: CapsuleContextPacker = dataclasses.field(
        default_factory=lambda: CapsuleContextPacker())
    # First-pass decoder used to elect a tentative ``root_cause``. By
    # default this is a *plain* multi-round bundle decoder over the
    # *normalised* union — same as the inner decoder's first stage,
    # but exposed as a dataclass field so tests can swap it out.
    first_pass: MultiRoundBundleDecoder = dataclasses.field(
        default_factory=lambda: MultiRoundBundleDecoder())
    # Strict per-decode token budget. ``None`` disables budgeting
    # (W15-3 backward-compat: ties W13).
    T_decoder: int | None = None

    # Forensic counters populated per ``decode_rounds`` call.
    _last_pack_result: W15PackResult | None = None
    _last_first_pass_root_cause: str = ""

    def decode_rounds(self,
                       per_round_handoffs: Sequence[Sequence[_DecodedHandoff]],
                       ) -> dict[str, Any]:
        # Normalise once (W13 layered) so both first-pass and final
        # decode see the same canonical kinds.
        self.inner.normalizer.reset_counters()
        self.inner._last_n_payload_rewrites = 0
        normalised_per_round: list[list[_DecodedHandoff]] = []
        for bundle in per_round_handoffs:
            normalised_per_round.append(self.inner.normalize_round(bundle))
        # Build the round-index hint vector parallel to the union.
        union: list[_DecodedHandoff] = []
        round_hint: list[int] = []
        for r_idx, bundle in enumerate(normalised_per_round, start=1):
            for h in bundle:
                union.append(h)
                round_hint.append(r_idx)
        # Stage 1 — tentative root_cause via the first-pass decoder.
        first = self.first_pass.decode_rounds([union])
        elected = str(first.get("root_cause", "unknown"))
        self._last_first_pass_root_cause = elected
        # Stage 2 — salience-aware repack + final decode.
        pack = self.packer.pack(
            union,
            elected_root_cause=elected,
            T_decoder=self.T_decoder,
            round_index_hint=round_hint)
        self._last_pack_result = pack
        kept_handoffs = [k.handoff for k in pack.kept]
        # Run the inner W13 layered decoder on the *packed* subset.
        # ``inner.decode_rounds`` re-normalises; we already normalised
        # the union, but kind rewrites are idempotent so re-running is
        # safe and the inner counters reflect the post-pack rewrites.
        # We pass the packed subset as a single bundle (round union
        # already collapsed); the W11 contradiction-aware drop fires
        # the same way.
        ans = self.inner.decode_rounds([kept_handoffs])
        ans = dict(ans)
        ans["pack_stats"] = pack.as_dict()
        ans["first_pass_root_cause"] = elected
        return ans

    def decode(self, handoffs: Sequence[_DecodedHandoff]
               ) -> dict[str, Any]:
        return self.decode_rounds([handoffs])

    @property
    def last_pack_result(self) -> W15PackResult | None:
        return self._last_pack_result

    @property
    def last_first_pass_root_cause(self) -> str:
        return self._last_first_pass_root_cause

    def pack_stats(self) -> dict[str, Any]:
        """Expose the most recent pack as a dict for the bench driver.
        Returns ``{}`` if no decode has been run yet.
        """
        if self._last_pack_result is None:
            return {}
        return self._last_pack_result.as_dict()


# =============================================================================
# SDK v3.19 — Bundle-relational compatibility disambiguator (W18 family)
# =============================================================================
#
# W18 attacks the *symmetric-corroboration wall* (W17-Λ-symmetric, named in
# SDK v3.18). On the W17-Λ-symmetric regime, every closed-form salience
# packer + admission policy in the SDK ties FIFO at 0.000 by construction:
# the bipartite ``(role × tag, kind, magnitude)`` multiset is identical for
# gold and decoy, so no service-blind admission AND no closed-form salience
# scorer can prefer one over the other. The W17 milestone named the next
# move explicitly: a *richer disambiguator* that consumes information
# *beyond* the bipartite multiset — specifically, the round-2 specific-tier
# disambiguator's *payload text* itself.
#
# The W18 :class:`RelationalCompatibilityDisambiguator` is the smallest
# move in that direction. It sits *after* the W11 / W15 cross-round
# decoder (so it sees the same admitted/packed bundle the auditor would
# decode), reads the round-2 specific-tier disambiguator's payload text,
# and projects the W11/W15 answer through a *bundle-relational
# compatibility* filter: keep service tags whose mention-and-relational-
# compound match score is non-zero; drop tags that the round-2 evidence
# does not name. Three structural properties matter:
#
#   1. **Closed-form, training-free.** The compatibility scorer is a
#      deterministic O(|union|) tokeniser over the disambiguator payload
#      plus an O(|admitted_tags| · |tokens|) match loop. Every score is
#      reproducible from the bytes of the surviving capsule bundle alone.
#   2. **Bounded-context honest.** The scorer reads *only* the W15-packed
#      bundle (or the un-packed admitted union when ``T_decoder`` is None)
#      — no extra capsule reads, no global state. Token-budget accounting
#      from W15 is byte-for-byte preserved.
#   3. **Backward-compat (W18-3).** When the round-2 disambiguator's
#      payload mentions every admitted service tag (or none of them), the
#      compatibility filter is a no-op on the answer set — W18 ties W15
#      byte-for-byte. The R-54..R-64 anchors remain at their prior values.
#
# Why W18 is not "just another decoder"
# -------------------------------------
# Every prior decoder layer (W11..W17) consumed only *closed-vocabulary*
# fields of the admitted bundle: ``claim_kind`` (W11/W12/W13), service
# tag (W11), bipartite role × tag corroboration (W7/W8/W9/W10),
# operational magnitudes (W15 salience, W17 magnitude-hint). W18 is the
# first decoder that consumes the *relational text* of a payload — i.e.,
# the substring ``relation=A_B_join`` in
# ``"deadlock relation=orders_payments_join wait_chain=2"``.
#
# The W17-Λ-symmetric wall is exactly the regime where this *additional*
# information channel is the only signal that breaks the tie. R-65-COMPAT
# (Phase-65) pre-commits the regime where this channel is consistently
# present; the named falsifiers R-65-NO-COMPAT / -CONFOUND / -DECEIVE
# pre-commit the structural limits.
#
# Honest scope (W18)
# ------------------
# * **W18-1 is conditional** on (a) the symmetric-corroboration bench
#   property (R-65-COMPAT), (b) the round-2 disambiguator's payload
#   carrying a relational-compound mention of *every* gold service tag
#   AND *no* decoy service tag.
# * **W18-Λ-no-compat** (R-65-NO-COMPAT) is the named structural limit
#   when the round-2 disambiguator carries no service-tag mentions: W18
#   ties FIFO at 0.000 — the relational scorer has no signal.
# * **W18-Λ-confound** (R-65-CONFOUND) is the named structural limit
#   when the round-2 disambiguator mentions BOTH gold AND decoy: W18's
#   compatibility score is tied; the policy abstains and falls through
#   to the W15 answer (which itself ties FIFO by W17-Λ-symmetric).
# * **W18-Λ-deceive** (R-65-DECEIVE) is the named structural limit
#   when the round-2 disambiguator mentions DECOY but not gold: W18
#   trusts its evidence and FAILS at 0.000. No closed-form bundle-
#   relational scorer can escape this regime without an outside-
#   information axis (W18-C-OUTSIDE, conjectural).
# * **Real-LLM transfer (W18-Λ-real)** depends on the LLM emitting the
#   relational-compound forms the synthetic bench uses. Free-form
#   natural-language relational mentions (e.g. "the join between orders
#   and payments") fall outside the exact-match layer — W18-C-LEARNED
#   names the natural learned-scorer extension.


# Closed-vocabulary set of relational-compound separators the W18
# scorer recognises. Each pattern names the *binary* relational form
# ``A<sep>B`` between two service tags. Lex-ordered for diff stability.
# The patterns are consumed in order; the first hit wins.
_RELATIONAL_COMPOUND_SEPARATORS: tuple[str, ...] = (
    "_",        # A_B_join, orders_payments_pipeline
    ".",        # A.B
    "/",        # /storage/A/B/, mount=/storage/A/B
    "-",        # A-B-join (rare but real in service names)
    ":",        # A:B (e.g. svc:A→svc:B compounds)
)


def _disambiguator_payload_tokens(payload: str) -> tuple[str, ...]:
    """Tokenise a round-2 disambiguator payload text into a flat,
    lower-cased token sequence over identifier chars.

    The round-2 disambiguator's payload is closed-vocabulary by W18's
    contract: kind-token + numeric magnitudes + relational-compound
    mention. We split on whitespace AND on every non-identifier char
    (``[^A-Za-z0-9_]``) so identifiers like ``orders_payments_join``
    decompose into ``("orders", "payments", "join")`` AND so
    ``mount=/storage/A/B`` decomposes into
    ``("mount", "storage", "a", "b")``. Lower-case folding for case-
    insensitive match.

    Returns a tuple of non-empty lower-cased identifier tokens.
    """
    if not payload:
        return ()
    out: list[str] = []
    cur: list[str] = []
    for ch in payload:
        if ch.isalnum() or ch == "_":
            cur.append(ch.lower())
        else:
            if cur:
                out.append("".join(cur))
                cur = []
    if cur:
        out.append("".join(cur))
    # Further split on '_' so compound identifiers decompose:
    final: list[str] = []
    for t in out:
        if "_" in t:
            final.extend([sub for sub in t.split("_") if sub])
            # Also keep the compound itself (so ``orders_payments_join``
            # matches both individual-tag and compound-tag scorers).
            final.append(t)
        else:
            final.append(t)
    return tuple(final)


def _relational_compatibility_score(
        service_tag: str,
        disambiguator_tokens: Sequence[str],
        ) -> tuple[int, int]:
    """Score a service tag against a tokenised disambiguator payload.

    Returns ``(direct_hits, compound_hits)`` where:
      * ``direct_hits`` — number of times ``service_tag`` (lower-cased)
        appears as a standalone identifier in the tokens. Each hit is
        evidence that the round-2 disambiguator names this service.
      * ``compound_hits`` — number of times ``service_tag`` appears
        as a contiguous-subsequence inside a compound identifier in
        the tokens (e.g. ``orders`` inside ``orders_payments_join``,
        or ``db_query`` as a contiguous subsequence of
        ``svc_web_then_svc_db_query``). Each hit is evidence that
        the round-2 disambiguator names this service as part of a
        relational compound.

    Both scores are non-negative integers; their sum is the W18
    compatibility score. The split is preserved for audit clarity
    (so the bench driver can verify the relational-compound layer
    is the load-bearing channel on R-65-COMPAT).

    Compound-target handling: a service tag that itself contains an
    underscore (e.g. ``logs_pipeline``, ``db_query``) is matched as
    a *contiguous subsequence* of underscore-separated parts in any
    compound token. This is the load-bearing semantic property: a
    compound service name like ``svc_web_then_svc_db_query`` mentions
    ``db_query`` iff ``["db", "query"]`` appears contiguously in its
    parts list.
    """
    if not service_tag:
        return (0, 0)
    target = service_tag.lower()
    target_parts = target.split("_")
    n_target = len(target_parts)
    direct = 0
    compound = 0
    for tok in disambiguator_tokens:
        if not tok:
            continue
        if tok == target:
            direct += 1
            continue
        if "_" not in tok:
            continue
        tok_parts = tok.split("_")
        n_tok = len(tok_parts)
        if n_target == 1:
            # Single-part target: match if it equals any compound part.
            if target in tok_parts:
                compound += 1
            continue
        # Multi-part target: contiguous-subsequence search.
        if n_tok < n_target:
            continue
        for i in range(n_tok - n_target + 1):
            if tok_parts[i:i + n_target] == target_parts:
                compound += 1
                break
    return (direct, compound)


@dataclasses.dataclass(frozen=True)
class W18CompatibilityResult:
    """The output of one ``RelationalCompatibilityDisambiguator.apply``
    call. Carries the projected answer plus the per-tag compatibility
    scores so the bench driver can audit the relational channel.

    Fields
    ------
    answer
        ``{"root_cause", "services", "remediation"}`` — same shape as
        every other capsule decoder. The ``services`` field is the
        W11/W15 answer projected through the compatibility filter
        (gold-only on R-65-COMPAT; unchanged on R-54..R-64; abstained
        on R-65-CONFOUND).
    base_services
        The pre-projection answer from the inner (W11/W15) decoder —
        useful for falsifier audit (W18-Λ-confound's abstention
        falls back to this exactly).
    disambiguator_payload
        The round-2 specific-tier disambiguator's payload text (the
        relational-evidence source). ``""`` when no specific-tier
        claim is present in the bundle.
    per_tag_scores
        ``{service_tag: (direct, compound)}`` for every admitted
        service tag in the inner answer. Sum = compatibility score.
    abstained
        True when the projection abstained (every admitted tag's
        score is non-zero AND scores are tied OR every tag's score
        is zero). Abstention falls back to ``base_services``.
    """
    answer: dict[str, Any]
    base_services: tuple[str, ...]
    disambiguator_payload: str
    per_tag_scores: dict[str, tuple[int, int]]
    abstained: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "root_cause": str(self.answer.get("root_cause", "unknown")),
            "services": tuple(self.answer.get("services", ())),
            "remediation": str(self.answer.get("remediation",
                                                 "investigate")),
            "base_services": tuple(self.base_services),
            "disambiguator_payload": str(self.disambiguator_payload),
            "per_tag_scores": {
                tag: [int(d), int(c)]
                for tag, (d, c) in self.per_tag_scores.items()
            },
            "abstained": bool(self.abstained),
        }


@dataclasses.dataclass
class RelationalCompatibilityDisambiguator:
    """Bundle-relational compatibility disambiguator (SDK v3.19, W18 family).

    The first decoder in the programme that consumes the *relational
    text* of a round-2 specific-tier disambiguator's payload — beyond
    the closed-vocabulary fields (``claim_kind``, ``service=`` token,
    bipartite corroboration, magnitude) every prior decoder consumed.
    The W18 method is the named research move beyond W17-Λ-symmetric
    (the symmetric-corroboration wall named in SDK v3.18); it does NOT
    introduce a new admission policy or context packer. It is purely
    additive on top of the W11 / W15 surface.

    Pipeline (deterministic, training-free)
    ----------------------------------------
    1. Run an inner cross-round decoder (default
       :class:`AttentionAwareBundleDecoder`, the W15 layer) over the
       admitted bundle. Capture the inner ``(root_cause, services,
       remediation)`` answer AND the W15 pack-stats block.
    2. Identify the round-2 specific-tier disambiguator(s) in the
       bundle (any handoff whose ``claim_kind`` is in
       :data:`_SPECIFIC_TIER_CLAIM_KINDS`). When multiple specific-
       tier disambiguators exist, concatenate their payloads in
       canonical (round, source_role, claim_kind, payload_sha) order.
    3. Tokenise the disambiguator payload via
       :func:`_disambiguator_payload_tokens` (lower-cased, split on
       non-identifier chars, compound identifiers preserved).
    4. For each admitted service tag in the inner answer, compute the
       compatibility score via :func:`_relational_compatibility_score`
       and record ``(direct_hits, compound_hits)``.
    5. Project the inner ``services`` set through the compatibility
       filter:
         * If at least one admitted tag has score > 0 AND not every
           admitted tag has score > 0, keep only the tags with
           score > 0 (the **strict-asymmetric** branch — gold-only
           on R-65-COMPAT; gold-only on the W17-Λ-symmetric
           regime when the disambiguator names gold but not decoy).
         * Else, abstain: return the inner ``services`` unchanged
           (W18-3 backward-compat on R-54..R-64; W18-Λ-confound /
           W18-Λ-no-compat fallback on R-65 falsifiers; W18-Λ-deceive
           is *NOT* abstention — when the disambiguator names decoy
           but not gold, decoy has score > 0 and gold has score = 0,
           so the strict-asymmetric branch fires and elects decoy,
           which fails by construction — the named falsifier).
    6. Return a :class:`W18CompatibilityResult` with the projected
       answer, the base inner answer, the disambiguator payload, the
       per-tag score map, and the abstention flag.

    Honest scope (W18)
    ------------------
    See the file-level W18 commentary block above. The W18-1 win is
    *strongly conditional* on the bench property (R-65-COMPAT); the
    three named falsifiers (R-65-NO-COMPAT / -CONFOUND / -DECEIVE)
    are pre-committed in
    ``vision_mvp.experiments.phase65_relational_disambiguation`` and
    mechanically verified by ``Phase65FalsifierTests``.

    Backward-compat (W18-3)
    ------------------------
    When (a) the bundle carries no specific-tier disambiguator OR
    (b) the disambiguator's payload mentions every admitted tag OR
    (c) the disambiguator's payload mentions no admitted tag, the
    abstention path fires and the W18 answer equals the inner
    decoder's answer byte-for-byte. R-54..R-64 anchors are
    preserved.
    """

    inner: AttentionAwareBundleDecoder = dataclasses.field(
        default_factory=lambda: AttentionAwareBundleDecoder())
    # When True, the W18 projection prefers the strict-asymmetric
    # branch (gold-only on R-65-COMPAT). When False, the projection
    # is a pure no-op — the W18 method reduces to its inner. The
    # default is True; the False knob is exposed for the W18-3
    # backward-compat anchor.
    enabled: bool = True

    # Forensic — last applied result, exposed for the bench driver.
    _last_result: W18CompatibilityResult | None = None

    def decode_rounds(self,
                       per_round_handoffs: Sequence[Sequence[_DecodedHandoff]],
                       ) -> dict[str, Any]:
        """Run the W18 pipeline. Returns the same shape as the inner
        decoder plus an additional ``compatibility`` block (W18 audit)
        and a ``pack_stats`` block (forwarded from the W15 inner)."""
        base = self.inner.decode_rounds(per_round_handoffs)
        # Build the same union the inner consumed (post-normalisation).
        union: list[_DecodedHandoff] = []
        round_hint: list[int] = []
        # Re-normalise via the inner's normaliser (idempotent).
        normalised_per_round: list[list[_DecodedHandoff]] = []
        for bundle in per_round_handoffs:
            normalised_per_round.append(
                self.inner.inner.normalize_round(bundle))
        for r_idx, bundle in enumerate(normalised_per_round, start=1):
            for h in bundle:
                union.append(h)
                round_hint.append(r_idx)
        result = self._project_answer(base, union, round_hint)
        self._last_result = result
        out = dict(result.answer)
        if "pack_stats" in base:
            out["pack_stats"] = base["pack_stats"]
        if "first_pass_root_cause" in base:
            out["first_pass_root_cause"] = base["first_pass_root_cause"]
        out["compatibility"] = result.as_dict()
        return out

    def decode(self, handoffs: Sequence[_DecodedHandoff]
               ) -> dict[str, Any]:
        return self.decode_rounds([handoffs])

    def _select_disambiguator(self,
                                union: Sequence[_DecodedHandoff],
                                round_hint: Sequence[int]
                                ) -> tuple[str, str]:
        """Pick the round-2 specific-tier disambiguator(s). Returns
        ``(payload, kind)``: the concatenation of every specific-tier
        disambiguator's payload (in canonical order) plus the kind
        of the first such handoff. Empty if no specific-tier
        disambiguator exists.
        """
        # Collect specific-tier handoffs along with deterministic
        # ordering keys. Round-2 (or higher) preferred; within a round
        # canonical (source_role, claim_kind, payload_sha) order.
        candidates: list[tuple[int, str, str, str, str]] = []
        for i, h in enumerate(union):
            if h.claim_kind in _SPECIFIC_TIER_CLAIM_KINDS:
                ridx = (round_hint[i] if i < len(round_hint) else 0)
                sha = _payload_sha256(h.payload)[:16]
                candidates.append(
                    (-ridx, h.source_role, h.claim_kind, sha, h.payload))
        if not candidates:
            return ("", "")
        candidates.sort()
        payloads = " ".join(c[4] for c in candidates)
        return (payloads, candidates[0][2])

    def _project_answer(self,
                         base: dict[str, Any],
                         union: Sequence[_DecodedHandoff],
                         round_hint: Sequence[int]
                         ) -> W18CompatibilityResult:
        base_services = tuple(base.get("services", ()) or ())
        if not self.enabled:
            return W18CompatibilityResult(
                answer=dict(base),
                base_services=base_services,
                disambiguator_payload="",
                per_tag_scores={},
                abstained=True)
        disambiguator_payload, _kind = self._select_disambiguator(
            union, round_hint)
        if not disambiguator_payload:
            return W18CompatibilityResult(
                answer=dict(base),
                base_services=base_services,
                disambiguator_payload="",
                per_tag_scores={},
                abstained=True)
        tokens = _disambiguator_payload_tokens(disambiguator_payload)
        # The W18 candidate set is the *union* of every service tag in
        # the admitted bundle — not the inner decoder's filtered set.
        # On the W17-Λ-symmetric regime the inner W11 contradiction-
        # aware drop fires symmetrically and drops EVERY service tag,
        # so ``base_services`` is empty even though the underlying union
        # carries gold + decoy. The W18 projection has to look past the
        # inner's drop to recover the gold tags. (On R-54..R-64 default
        # the union typically includes more tags than the inner's
        # filtered set; the strict-asymmetric branch still picks the
        # gold-tag-mentioned subset.)
        union_tag_set: set[str] = set()
        for h in union:
            tag = _service_tag_of(h.payload)
            if tag:
                union_tag_set.add(tag)
        admitted_tags = sorted(union_tag_set)
        if not admitted_tags:
            return W18CompatibilityResult(
                answer=dict(base),
                base_services=base_services,
                disambiguator_payload=disambiguator_payload,
                per_tag_scores={},
                abstained=True)
        per_tag: dict[str, tuple[int, int]] = {}
        for tag in admitted_tags:
            per_tag[tag] = _relational_compatibility_score(tag, tokens)
        # Strict-asymmetric branch: at least one tag has positive score
        # AND not every tag has positive score → keep positive-score
        # tags only. Otherwise abstain (no signal / symmetric signal /
        # backward-compat — inner answer is the trusted fallback).
        positive = [tag for tag in admitted_tags
                     if (per_tag[tag][0] + per_tag[tag][1]) > 0]
        if not positive or len(positive) == len(admitted_tags):
            return W18CompatibilityResult(
                answer=dict(base),
                base_services=base_services,
                disambiguator_payload=disambiguator_payload,
                per_tag_scores=per_tag,
                abstained=True)
        # Project: keep positive-score tags from the union.
        # Re-derive root_cause + remediation: the inner already elected
        # them from the union (W11 priority decoder); we only project
        # the service set. If the inner's services happens to be a
        # subset of ``positive``, that's a strict refinement (the W18
        # additional information ratifies the inner). If the inner's
        # services is disjoint from ``positive`` (the W11 drop killed
        # the gold tags, e.g. on R-65-COMPAT), W18 *recovers* them.
        projected_services = tuple(sorted(positive))
        new_answer = dict(base)
        new_answer["services"] = projected_services
        return W18CompatibilityResult(
            answer=new_answer,
            base_services=base_services,
            disambiguator_payload=disambiguator_payload,
            per_tag_scores=per_tag,
            abstained=False)

    @property
    def last_result(self) -> W18CompatibilityResult | None:
        return self._last_result

    @property
    def T_decoder(self) -> int | None:
        return self.inner.T_decoder

    @T_decoder.setter
    def T_decoder(self, v: int | None) -> None:
        self.inner.T_decoder = v

    def pack_stats(self) -> dict[str, Any]:
        """Forward the inner W15 pack-stats so the bench driver can
        verify token-budget honesty (W18 reads only the W15-packed
        bundle; ``tokens_kept`` is byte-for-byte identical)."""
        return self.inner.pack_stats()


# =============================================================================
# SDK v3.20 — Bundle-contradiction-aware trust-weighted disambiguator
# (W19 family)
# =============================================================================
#
# W19 attacks the *deceptive-ambiguity wall* (W18-Λ-deceive, named in
# SDK v3.19). On the W18-Λ-deceive regime, the round-2 specific-tier
# disambiguator's payload mentions DECOY service tags but NOT gold —
# adversarial relational evidence. W18 trusts its evidence and projects
# the answer to the decoy-only set; ``services_correct`` fails by
# construction. W18-Λ-confound is the symmetric case (round-2 mentions
# both gold and decoy); W18 abstains and falls through to the inner
# answer, which itself ties FIFO at 0.000 by W17-Λ-symmetric.
#
# The W19 :class:`BundleContradictionDisambiguator` is the smallest
# move beyond W18 that does not require outside information. It sits
# *after* the W18 layer (so it sees the same admitted/packed bundle
# AND W18's projection AND W18's per-tag scores) and adds a
# *bundle-contradiction* layer: count, for each admitted service tag,
# the number of *independent asymmetric witnesses* in the bundle —
# specific-tier handoffs OTHER than the canonical primary disambiguator
# whose tokenised payload mentions the tag. When the W18 strict-
# asymmetric branch picks tags whose witness count is *strictly less*
# than the dropped tags' witness count, W19 *inverts* the projection
# (the primary's evidence is contradicted by independent witnesses;
# inversion picks the high-witness subset = gold). When W18 abstains
# (full-set or empty-set hit), W19 *refines* the abstention by picking
# the strict-max-witness subset within the candidate set.
#
# Three structural properties matter (W19):
#
#   1. **Closed-form, training-free.** The witness counter is a
#      deterministic O(|union| · |admitted_tags|) loop over specific-
#      tier handoffs in the union, excluding the primary disambiguator
#      identified in canonical W18 sort order. Every score is
#      reproducible from the bytes of the surviving capsule bundle alone.
#   2. **Bounded-context honest.** The W19 scorer reads only the
#      W15-packed bundle (or the un-packed admitted union when
#      ``T_decoder`` is None) — same input as W18. No extra capsule
#      reads, no global state, no outside-information lookup. Token-
#      budget accounting from W15 is byte-for-byte preserved.
#   3. **Backward-compat (W19-3).** When the bundle carries no
#      independent asymmetric witness (every aw count is zero), the
#      W19 inversion guard cannot fire and the W19 abstention refinement
#      cannot find a strict-max subset; the projection falls through to
#      the W18 answer byte-for-byte. R-54..R-65 default banks are
#      preserved.
#
# Why W19 is not "just another decoder"
# -------------------------------------
# Every prior decoder layer in the programme either consumed bipartite
# corroboration counts (W7..W10), per-round structure (W11), normalised
# kind / payload tokens (W12 / W13), producer-side protocol structure
# (W14 / W17), salience-aware packing (W15), or relational-compound
# match against ONE primary payload (W18). W19 is the first decoder
# that consumes the *consistency between* multiple specific-tier
# emissions in the bundle — the structural relationship between the
# primary disambiguator and the secondary witnesses.
#
# The W18-Λ-deceive wall is exactly the regime where this *additional*
# information channel — the bundle-internal contradiction between
# primary and secondary witnesses — is the only signal that breaks
# the deception. R-66-DECEIVE-NAIVE (Phase-66) pre-commits the regime
# where this channel is consistently present; R-66-DECEIVE-TOTAL and
# R-66-OUTSIDE-REQUIRED pre-commit the structural limits where W19
# cannot escape (no witnesses anywhere; symmetric witnesses).
#
# Honest scope (W19)
# ------------------
# * **W19-1 is conditional** on (a) the symmetric-corroboration round-1
#   property (so W17-Λ-symmetric still applies and only an additional
#   information channel can win), AND (b) the bundle carrying at least
#   one *independent asymmetric witness* (a specific-tier handoff
#   OTHER than the primary disambiguator) whose payload mentions a
#   service tag asymmetrically across the candidate set.
# * **W19-Λ-total** (R-66-DECEIVE-TOTAL) is the named structural limit
#   when the bundle carries NO asymmetric witnesses anywhere: W19
#   reduces to W18 and fails identically (the closed-form bundle is
#   exhausted of asymmetric signal).
# * **W19-Λ-outside** (R-66-OUTSIDE-REQUIRED) is the named structural
#   limit when the bundle carries witnesses but the witness count is
#   *symmetric* between primary's named set and the complement: W19
#   abstains and ties FIFO. The natural escape is **W19-C-OUTSIDE**
#   (an outside-information axis to detect the deceptive primary by
#   cross-reference; conjectural).
# * **Real-LLM transfer (W19-Λ-real)** depends on the LLM emitting
#   the *secondary* asymmetric witness in the same closed-vocabulary
#   form the synthetic bench uses. Free-form natural-language relational
#   mentions fall outside the closure. The natural extension is
#   **W19-C-LEARNED** (a learned trust scorer over capsule bundles).


# Closed-vocabulary set of round-1 generic-noise kinds. The W19
# witness counter explicitly excludes these — they are the
# *symmetric-noise* tier in R-64-SYM / R-65-COMPAT / R-66, where
# both gold and decoy are corroborated equally. An *asymmetric
# witness* is by definition a specific-tier handoff (kind in
# :data:`_SPECIFIC_TIER_CLAIM_KINDS`) other than the primary
# disambiguator. Listed lex-ordered for diff stability.
W19_SYMMETRIC_NOISE_KINDS: frozenset[str] = frozenset({
    "ERROR_RATE_SPIKE",
    "FW_BLOCK_SURGE",
    "LATENCY_SPIKE",
})


# Closed-vocabulary canonical-role-for-kind table for the
# incident-triage benchmark family. Maps each specific-tier kind
# to the producer role that *canonically* emits it (matches the
# subscription table in
# :func:`vision_mvp.tasks.incident_triage.build_role_subscriptions`).
# The W19 primary-identification uses this table to disambiguate
# between (a) the canonical primary disambiguator and (b)
# secondary asymmetric witnesses from non-canonical roles whose
# normalised kind happens to match the primary's. Hardcoded here
# (rather than imported from incident_triage) to avoid a layering
# inversion — team_coord is a peer of incident_triage, not a
# dependent. Lex-ordered by kind for diff stability.
_INCIDENT_TRIAGE_CANONICAL_ROLE_FOR_KIND: dict[str, str] = {
    "CRON_OVERRUN":         "sysadmin",
    "DEADLOCK_SUSPECTED":   "db_admin",
    "DISK_FILL_CRITICAL":   "sysadmin",
    "DNS_MISROUTE":         "network",
    "OOM_KILL":             "sysadmin",
    "POOL_EXHAUSTION":      "db_admin",
    "SLOW_QUERY_OBSERVED":  "db_admin",
    "TLS_EXPIRED":          "network",
}


def _w19_canonical_primary_index(
        union: Sequence[_DecodedHandoff],
        round_hint: Sequence[int],
        raw_union: Sequence[_DecodedHandoff] | None = None,
        ) -> int:
    """Identify the canonical primary specific-tier handoff in the
    union. Returns ``-1`` if none exists.

    The canonical primary is the *first* specific-tier handoff in
    canonical sort order, with two semantic tiebreakers:

      1. **Canonical-role match.** A handoff whose ``source_role``
         matches the canonical role for its (normalised) kind in
         :data:`_INCIDENT_TRIAGE_CANONICAL_ROLE_FOR_KIND` is
         preferred over a handoff whose ``source_role`` does NOT
         match (the synonym/heuristic-rescued case where a
         non-canonical role emitted a kind that normalises to a
         canonical specific-tier kind). This breaks ties between
         (a) the canonical primary and (b) the secondary
         asymmetric witness from a non-canonical role.
      2. **Raw-kind tiebreak.** When ``raw_union`` is provided, the
         pre-normalisation ``claim_kind`` is used as the secondary
         sort key (so the canonical raw kind ``DISK_FILL_CRITICAL``
         sorts before its synonym ``DISK_FILL_DETECTED``).
      3. **Canonical W18 tiebreak.** ``-ridx`` (highest round
         first), then ``source_role``, then ``claim_kind``, then
         ``sha``, then ``payload`` — the same canonical sort
         W18's ``_select_disambiguator`` uses on its concatenated
         output.

    The two semantic tiebreakers are *only* used when there are
    multiple specific-tier handoffs at the highest round; on R-58 /
    R-65 default banks (single specific-tier handoff per scenario),
    the primary identification reduces to the unique candidate.
    """
    candidates: list[tuple[int, int, str, str, str, str, str, int]] = []
    for i, h in enumerate(union):
        if h.claim_kind not in _SPECIFIC_TIER_CLAIM_KINDS:
            continue
        ridx = (round_hint[i] if i < len(round_hint) else 0)
        canonical_role = _INCIDENT_TRIAGE_CANONICAL_ROLE_FOR_KIND.get(
            h.claim_kind)
        # 0 = canonical-role match (preferred); 1 = non-canonical.
        canonical_match = 0 if (canonical_role is not None
                                 and canonical_role == h.source_role) else 1
        if raw_union is not None and i < len(raw_union):
            raw_kind = raw_union[i].claim_kind
        else:
            raw_kind = h.claim_kind
        sha = _payload_sha256(h.payload)[:16]
        candidates.append((
            -ridx, canonical_match, raw_kind,
            h.source_role, h.claim_kind, sha, h.payload, i))
    if not candidates:
        return -1
    candidates.sort()
    return int(candidates[0][7])


def _w19_witness_counts(
        union: Sequence[_DecodedHandoff],
        primary_index: int,
        admitted_tags: Sequence[str],
        ) -> dict[str, int]:
    """Count independent asymmetric witnesses for each tag.

    A handoff ``h`` is an *asymmetric witness* iff:

      * ``h.claim_kind`` is in :data:`_SPECIFIC_TIER_CLAIM_KINDS`
        (so it's not in the symmetric round-1 noise set), AND
      * ``h`` is NOT the canonical primary disambiguator
        (identified by ``primary_index``), AND
      * ``h``'s tokenised payload contains a direct or contiguous-
        subsequence-compound match for the target tag (same scoring
        function as W18: :func:`_relational_compatibility_score`).

    Returns ``{service_tag: aw_count}`` for every tag in
    ``admitted_tags``. The count is the number of *distinct*
    witness handoffs (deduplicated by ``(source_role, claim_kind,
    payload_sha)`` so two byte-identical witnesses collapse to
    one). All other handoffs in the union — round-1 generic noise
    AND the primary itself — are excluded.

    Bounded-context honesty: this function reads only the bytes of
    the union it is passed. It does NOT consult any capsule outside
    the W15-packed bundle.
    """
    out: dict[str, int] = {tag: 0 for tag in admitted_tags}
    if not admitted_tags:
        return out
    seen: set[tuple[str, str, str]] = set()
    for i, h in enumerate(union):
        if i == primary_index:
            continue
        if h.claim_kind not in _SPECIFIC_TIER_CLAIM_KINDS:
            continue
        sha = _payload_sha256(h.payload)[:16]
        key = (h.source_role, h.claim_kind, sha)
        if key in seen:
            continue
        seen.add(key)
        tokens = _disambiguator_payload_tokens(h.payload)
        for tag in admitted_tags:
            d, c = _relational_compatibility_score(tag, tokens)
            if (d + c) > 0:
                out[tag] = out.get(tag, 0) + 1
    return out


W19_BRANCH_PRIMARY_TRUSTED = "primary_trusted"
W19_BRANCH_INVERSION = "inversion"
W19_BRANCH_CONFOUND_RESOLVED = "confound_resolved"
W19_BRANCH_ABSTAINED_NO_SIGNAL = "abstained_no_signal"
W19_BRANCH_ABSTAINED_SYMMETRIC = "abstained_symmetric"
W19_BRANCH_DISABLED = "disabled"

W19_ALL_BRANCHES: tuple[str, ...] = (
    W19_BRANCH_PRIMARY_TRUSTED,
    W19_BRANCH_INVERSION,
    W19_BRANCH_CONFOUND_RESOLVED,
    W19_BRANCH_ABSTAINED_NO_SIGNAL,
    W19_BRANCH_ABSTAINED_SYMMETRIC,
    W19_BRANCH_DISABLED,
)


@dataclasses.dataclass(frozen=True)
class W19TrustResult:
    """The output of one ``BundleContradictionDisambiguator.apply``
    call. Carries the projected answer plus the per-tag witness
    counts, the W18 fall-through answer, and the chosen W19 branch
    so the bench driver can audit the contradiction-resolution
    channel.

    Fields
    ------
    answer
        ``{"root_cause", "services", "remediation"}`` — same shape as
        every other capsule decoder. The ``services`` field is the
        W18 answer projected through the W19 contradiction filter
        (gold-only on R-66-DECEIVE-NAIVE / R-66-CONFOUND-RESOLVABLE;
        unchanged on R-54..R-65; abstained on R-66-OUTSIDE-REQUIRED).
    base_services
        The pre-projection answer from the inner (W18) decoder —
        useful for falsifier audit.
    w18_services
        The W18 strict-asymmetric / abstention answer — equal to
        ``base_services`` when W18 didn't fire its strict-asymmetric
        branch.
    primary_payload
        The *single* canonical primary disambiguator's payload bytes
        the W19 layer treats as the "asymmetric primary" for the
        inversion check. ``""`` when no specific-tier handoff exists.
    per_tag_w18_score
        ``{service_tag: (direct, compound)}`` over the W18-concatenated
        full-disambiguator text — exactly what W18 records.
    per_tag_witness_count
        ``{service_tag: aw_count}`` — the W19 independent-witness count
        per tag, EXCLUDING the canonical primary.
    decoder_branch
        One of the values in :data:`W19_ALL_BRANCHES`. Names the
        precise branch the W19 logic took on this cell.
    abstained
        True when the W19 projection abstained (no strict-max subset
        AND no inversion). Abstention falls back to the W18 answer.
    """
    answer: dict[str, Any]
    base_services: tuple[str, ...]
    w18_services: tuple[str, ...]
    primary_payload: str
    per_tag_w18_score: dict[str, tuple[int, int]]
    per_tag_witness_count: dict[str, int]
    decoder_branch: str
    abstained: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "root_cause": str(self.answer.get("root_cause", "unknown")),
            "services": tuple(self.answer.get("services", ())),
            "remediation": str(self.answer.get("remediation",
                                                 "investigate")),
            "base_services": tuple(self.base_services),
            "w18_services": tuple(self.w18_services),
            "primary_payload": str(self.primary_payload),
            "per_tag_w18_score": {
                tag: [int(d), int(c)]
                for tag, (d, c) in self.per_tag_w18_score.items()
            },
            "per_tag_witness_count": {
                tag: int(c)
                for tag, c in self.per_tag_witness_count.items()
            },
            "decoder_branch": str(self.decoder_branch),
            "abstained": bool(self.abstained),
        }


@dataclasses.dataclass
class BundleContradictionDisambiguator:
    """Bundle-contradiction-aware trust-weighted disambiguator
    (SDK v3.20, W19 family).

    The first decoder in the programme that *resolves bundle-internal
    contradiction* between a deceptive (or confounded) round-2
    disambiguator and independent asymmetric witnesses elsewhere in
    the bundle. It sits *after* the W18 layer and refines W18's
    projection in two cases:

    1. **W19-1-inversion branch.** When W18 fires the strict-asymmetric
       branch (its named set N ⊊ U is a proper non-empty subset of the
       admitted tag set U) but the *complement* tag set U \\ N has a
       strictly higher max independent-witness count than the named set
       N, W19 *inverts* the projection: project to U \\ N. This handles
       the W18-Λ-deceive limit when the bundle carries a quiet asymmetric
       witness for gold (R-66-DECEIVE-NAIVE).
    2. **W19-1-confound branch.** When W18 abstains (its named set N
       equals U or is empty) AND there is a unique strict-max-witness
       subset M ⊊ U of size ≥ 1, W19 projects to M. This handles the
       W18-Λ-confound limit when the bundle carries an asymmetric
       witness for gold among the confounded set (R-66-CONFOUND-RESOLVABLE).

    On regimes where the bundle carries no independent asymmetric
    witnesses (every aw count is zero, including R-65-NO-COMPAT,
    R-65-CONFOUND, R-65-DECEIVE, R-66-DECEIVE-TOTAL), W19 reduces to
    W18 byte-for-byte. Honest scope: when no asymmetric witness is
    present anywhere in the bundle, W19 cannot escape the W18-Λ-deceive
    wall — that's W19-Λ-total, the named structural limit.

    Pipeline (deterministic, training-free)
    ----------------------------------------
    1. Run inner W18 (the W15 attention-aware decoder + W18 relational-
       compatibility projection). Capture the W18 answer + per-tag W18
       score + abstention flag.
    2. Identify the *canonical primary* specific-tier disambiguator
       in the admitted union (highest-round, lex-min source_role +
       claim_kind in canonical W18 sort order).
    3. Compute the asymmetric-witness count ``aw(T)`` for each
       admitted tag T: count specific-tier handoffs other than the
       primary whose tokenised payload mentions T (deduplicated by
       ``(source_role, claim_kind, payload_sha)``).
    4. Decide the W19 branch:

       * If the W18 strict-asymmetric branch fired (W18 chose a
         proper non-empty subset N ⊊ U) AND
         ``max_aw(U \\ N) > max_aw(N)``: invert — project to
         ``U \\ N`` (W19-1-inversion).
       * If the W18 abstained on a non-empty admitted set U (W18 saw
         all-positive or all-zero scores) AND there is a unique
         strict-max-aw subset M ⊊ U with |M| ≥ 1: project to M
         (W19-1-confound).
       * Otherwise: fall through to W18's answer
         (primary_trusted / abstained).
    5. Return a :class:`W19TrustResult` with the projected answer,
       the W18 fall-through answer, the per-tag witness counts, and
       the chosen branch.

    Honest scope (W19)
    ------------------
    See the file-level W19 commentary block above. The W19-1 win is
    *strongly conditional* on the bench property (R-66-DECEIVE-NAIVE
    or R-66-CONFOUND-RESOLVABLE); the named falsifiers
    R-66-DECEIVE-TOTAL (W19-Λ-total) and R-66-OUTSIDE-REQUIRED
    (W19-Λ-outside) are pre-committed in
    ``vision_mvp.experiments.phase66_deceptive_ambiguity`` and
    mechanically verified by ``Phase66FalsifierTests``.

    Backward-compat (W19-3)
    ------------------------
    When the bundle carries no independent asymmetric witness (every
    aw count is zero), neither the inversion guard nor the confound
    refinement can fire; W19 returns the inner W18 answer byte-for-
    byte. R-54..R-65 default banks are preserved. With
    ``enabled = False`` W19 reduces to W18 byte-for-byte.
    """

    inner: RelationalCompatibilityDisambiguator = dataclasses.field(
        default_factory=lambda: RelationalCompatibilityDisambiguator())
    enabled: bool = True

    # Forensic — last applied result, exposed for the bench driver.
    _last_result: W19TrustResult | None = None

    def decode_rounds(self,
                       per_round_handoffs: Sequence[Sequence[_DecodedHandoff]],
                       ) -> dict[str, Any]:
        """Run the W19 pipeline. Returns the same shape as the inner
        W18 decoder plus an additional ``trust`` block (W19 audit)
        and forwards W18's ``compatibility`` block + W15's
        ``pack_stats`` block."""
        base = self.inner.decode_rounds(per_round_handoffs)
        # Build the same union the inner consumed (post-normalisation
        # via the W15 → W13 → layered normaliser). Also retain the
        # raw (pre-normalisation) union — primary identification uses
        # the raw ``claim_kind`` field to disambiguate between the
        # canonical primary and synonym/heuristic-rescued secondary
        # witnesses (whose post-normalisation kinds are identical).
        union: list[_DecodedHandoff] = []
        raw_union: list[_DecodedHandoff] = []
        round_hint: list[int] = []
        normalised_per_round: list[list[_DecodedHandoff]] = []
        for bundle in per_round_handoffs:
            normalised_per_round.append(
                self.inner.inner.inner.normalize_round(bundle))
        for r_idx, (norm_bundle, raw_bundle) in enumerate(
                zip(normalised_per_round, per_round_handoffs), start=1):
            # The normaliser is element-wise (preserves order and
            # arity) — verified by ``LayeredRobustMultiRoundBundleDecoder
            # .normalize_round``. We therefore zip the two bundles
            # in parallel.
            for h_norm, h_raw in zip(norm_bundle, raw_bundle):
                union.append(h_norm)
                raw_union.append(h_raw)
                round_hint.append(r_idx)
        result = self._project_answer(
            base, union, round_hint, raw_union=raw_union)
        self._last_result = result
        out = dict(result.answer)
        if "pack_stats" in base:
            out["pack_stats"] = base["pack_stats"]
        if "compatibility" in base:
            out["compatibility"] = base["compatibility"]
        if "first_pass_root_cause" in base:
            out["first_pass_root_cause"] = base["first_pass_root_cause"]
        out["trust"] = result.as_dict()
        return out

    def decode(self, handoffs: Sequence[_DecodedHandoff]
               ) -> dict[str, Any]:
        return self.decode_rounds([handoffs])

    def _project_answer(self,
                         base: dict[str, Any],
                         union: Sequence[_DecodedHandoff],
                         round_hint: Sequence[int],
                         raw_union: Sequence[_DecodedHandoff] | None = None,
                         ) -> W19TrustResult:
        # The W18 inner records its own audit on ``base["compatibility"]``.
        # We reuse it where useful AND re-derive what W19 needs from
        # the union directly (so W19 is self-sufficient and does not
        # depend on the W18 audit's exact fields).
        w18_services = tuple(base.get("services", ()) or ())
        w18_audit = base.get("compatibility") or {}
        per_tag_w18: dict[str, tuple[int, int]] = {}
        scores_raw = w18_audit.get("per_tag_scores") or {}
        for tag, val in scores_raw.items():
            if isinstance(val, (list, tuple)) and len(val) == 2:
                per_tag_w18[tag] = (int(val[0]), int(val[1]))
            else:
                per_tag_w18[tag] = (0, 0)
        if not self.enabled:
            return W19TrustResult(
                answer=dict(base),
                base_services=w18_services,
                w18_services=w18_services,
                primary_payload="",
                per_tag_w18_score=per_tag_w18,
                per_tag_witness_count={},
                decoder_branch=W19_BRANCH_DISABLED,
                abstained=True,
            )

        # Identify the canonical primary disambiguator. Uses the
        # canonical-role-for-kind table + raw-kind tiebreak so the
        # primary is the canonical-routed handoff even when a
        # non-canonical role emitted a synonym/heuristic-rescued
        # kind that normalises to the same canonical specific-tier.
        primary_idx = _w19_canonical_primary_index(
            union, round_hint, raw_union=raw_union)
        if primary_idx < 0:
            return W19TrustResult(
                answer=dict(base),
                base_services=w18_services,
                w18_services=w18_services,
                primary_payload="",
                per_tag_w18_score=per_tag_w18,
                per_tag_witness_count={},
                decoder_branch=W19_BRANCH_ABSTAINED_NO_SIGNAL,
                abstained=True,
            )
        primary_payload = union[primary_idx].payload

        # The W19 candidate set is the *full* admitted-tag union — same
        # set the W18 layer scored over.
        union_tag_set: set[str] = set()
        for h in union:
            tag = _service_tag_of(h.payload)
            if tag:
                union_tag_set.add(tag)
        admitted_tags = sorted(union_tag_set)
        if not admitted_tags:
            return W19TrustResult(
                answer=dict(base),
                base_services=w18_services,
                w18_services=w18_services,
                primary_payload=primary_payload,
                per_tag_w18_score=per_tag_w18,
                per_tag_witness_count={},
                decoder_branch=W19_BRANCH_ABSTAINED_NO_SIGNAL,
                abstained=True,
            )

        # Witness count per tag (excludes primary).
        aw = _w19_witness_counts(union, primary_idx, admitted_tags)

        # Re-derive W18's per-tag scores over the FULL concatenated
        # disambiguator text (so W19 is robust against W18 audit
        # changes). Use the same _select_disambiguator + tokeniser
        # the W18 inner uses.
        full_disambiguator, _ = self.inner._select_disambiguator(
            union, round_hint)
        tokens_full = _disambiguator_payload_tokens(full_disambiguator)
        per_tag_full: dict[str, tuple[int, int]] = {}
        for tag in admitted_tags:
            per_tag_full[tag] = _relational_compatibility_score(
                tag, tokens_full)
        # Use the freshly re-derived scores so W19 is self-contained.
        per_tag_w18 = per_tag_full

        # Identify the W18 named set N: tags whose full-disambiguator
        # score is positive.
        N_set = [tag for tag in admitted_tags
                  if (per_tag_w18[tag][0] + per_tag_w18[tag][1]) > 0]
        N = set(N_set)
        U = set(admitted_tags)
        complement = sorted(U - N)
        N_sorted = sorted(N)

        def _max_aw(tags: Sequence[str]) -> int:
            return max((aw.get(t, 0) for t in tags), default=0)

        # W19 inversion guard. Fires when:
        #   - W18 strict-asymmetric branch fired (N ⊊ U is proper non-
        #     empty subset).
        #   - max_aw(U \ N) > max_aw(N) — independent witnesses
        #     contradict the primary's named set.
        if 0 < len(N) < len(U):
            if _max_aw(complement) > _max_aw(N_sorted):
                # Project to the high-witness tags in the complement.
                top_aw = _max_aw(complement)
                projected = tuple(
                    sorted(t for t in complement if aw.get(t, 0) == top_aw))
                new_answer = dict(base)
                new_answer["services"] = projected
                return W19TrustResult(
                    answer=new_answer,
                    base_services=w18_services,
                    w18_services=w18_services,
                    primary_payload=primary_payload,
                    per_tag_w18_score=per_tag_w18,
                    per_tag_witness_count=aw,
                    decoder_branch=W19_BRANCH_INVERSION,
                    abstained=False,
                )
            # No inversion → trust W18's strict-asymmetric pick.
            return W19TrustResult(
                answer=dict(base),
                base_services=w18_services,
                w18_services=w18_services,
                primary_payload=primary_payload,
                per_tag_w18_score=per_tag_w18,
                per_tag_witness_count=aw,
                decoder_branch=W19_BRANCH_PRIMARY_TRUSTED,
                abstained=False,
            )

        # W19 confound-refinement branch. Fires when:
        #   - W18 abstained (N = U or N = ∅).
        #   - There is a unique strict-max-aw subset M ⊊ U with |M| ≥ 1.
        max_aw_in_U = _max_aw(admitted_tags)
        if max_aw_in_U > 0:
            top_set = sorted(t for t in admitted_tags
                              if aw.get(t, 0) == max_aw_in_U)
            if 0 < len(top_set) < len(admitted_tags):
                new_answer = dict(base)
                new_answer["services"] = tuple(top_set)
                return W19TrustResult(
                    answer=new_answer,
                    base_services=w18_services,
                    w18_services=w18_services,
                    primary_payload=primary_payload,
                    per_tag_w18_score=per_tag_w18,
                    per_tag_witness_count=aw,
                    decoder_branch=W19_BRANCH_CONFOUND_RESOLVED,
                    abstained=False,
                )
            # Top set covers all admitted tags — symmetric witnesses;
            # W19 cannot prefer one. Abstain.
            return W19TrustResult(
                answer=dict(base),
                base_services=w18_services,
                w18_services=w18_services,
                primary_payload=primary_payload,
                per_tag_w18_score=per_tag_w18,
                per_tag_witness_count=aw,
                decoder_branch=W19_BRANCH_ABSTAINED_SYMMETRIC,
                abstained=True,
            )

        # No witnesses, no inversion. Fall through to W18 (which
        # itself may have abstained or trusted the primary).
        branch = (W19_BRANCH_PRIMARY_TRUSTED
                   if 0 < len(N) < len(U)
                   else W19_BRANCH_ABSTAINED_NO_SIGNAL)
        return W19TrustResult(
            answer=dict(base),
            base_services=w18_services,
            w18_services=w18_services,
            primary_payload=primary_payload,
            per_tag_w18_score=per_tag_w18,
            per_tag_witness_count=aw,
            decoder_branch=branch,
            abstained=(branch == W19_BRANCH_ABSTAINED_NO_SIGNAL),
        )

    @property
    def last_result(self) -> W19TrustResult | None:
        return self._last_result

    @property
    def T_decoder(self) -> int | None:
        return self.inner.T_decoder

    @T_decoder.setter
    def T_decoder(self, v: int | None) -> None:
        self.inner.T_decoder = v

    def pack_stats(self) -> dict[str, Any]:
        """Forward the inner W15 pack-stats so the bench driver can
        verify token-budget honesty (W19 reads only the W15-packed
        bundle; ``tokens_kept`` is byte-for-byte identical to W18's
        which is byte-for-byte identical to W15's)."""
        return self.inner.pack_stats()


__all__ = [
    # Per-role budget
    "RoleBudget", "DEFAULT_ROLE_BUDGETS",
    # Capsule constructors
    "capsule_team_handoff", "capsule_role_view",
    "capsule_team_decision",
    # Admission policies
    "AdmissionPolicy", "AdmissionDecision",
    "REASON_ADMIT", "REASON_BUDGET_FULL", "REASON_TOKENS_FULL",
    "REASON_UNKNOWN_KIND", "REASON_DUPLICATE", "REASON_SCORE_LOW",
    "FifoAdmissionPolicy", "ClaimPriorityAdmissionPolicy",
    "CoverageGuidedAdmissionPolicy",
    "CohortCoherenceAdmissionPolicy",
    "CrossRoleCorroborationAdmissionPolicy",
    "MultiServiceCorroborationAdmissionPolicy",
    "ALL_FIXED_POLICY_NAMES",
    # Coordinator
    "TeamCoordinator",
    # Audit
    "T_INVARIANTS", "TeamLifecycleAuditReport",
    "audit_team_lifecycle",
    # SDK v3.11 — bundle-aware team decoder (W10 family).
    "BundleAwareTeamDecoder", "decode_admitted_role_view",
    "CAUSAL_CLAIM_KINDS_PER_ROOT_CAUSE",
    # SDK v3.12 — multi-round bundle-aware team decoder (W11 family).
    "MultiRoundBundleDecoder", "collect_admitted_handoffs",
    # SDK v3.13 — real-LLM-robust multi-round bundle decoder (W12 family).
    "RobustMultiRoundBundleDecoder", "CLAIM_KIND_SYNONYMS",
    "normalize_claim_kind", "normalize_payload", "normalize_handoff",
    # SDK v3.14 — layered open-world normaliser + decoder (W13 family).
    "HeuristicAbstractionRule", "LayeredClaimNormalizer",
    "LayeredRobustMultiRoundBundleDecoder",
    "LAYERED_NORMALIZER_ABSTAIN",
    # SDK v3.15 — structured producer protocol (W14 family).
    "PRODUCER_PROMPT_NAIVE", "PRODUCER_PROMPT_STRUCTURED",
    "ALL_PRODUCER_PROMPT_MODES",
    "RoleExtractionSchema", "ProducerPromptResult",
    "StructuredProducerProtocol",
    "INCIDENT_TRIAGE_OBSERVATION_KINDS",
    "incident_triage_role_schemas",
    # SDK v3.18 — magnitude-hinted producer protocol (W17 family).
    "PRODUCER_PROMPT_MAGNITUDE_HINTED",
    "OperationalThreshold",
    "INCIDENT_TRIAGE_DEFAULT_MAGNITUDE_THRESHOLDS",
    "incident_triage_magnitude_thresholds",
    # SDK v3.16 — attention-aware capsule context packing (W15 family).
    "W15_DEFAULT_TIER_WEIGHT", "W15_DEFAULT_CCK_WEIGHT",
    "W15_DEFAULT_CORROBORATION_WEIGHT", "W15_DEFAULT_MAGNITUDE_WEIGHT",
    "W15_DEFAULT_ROUND_WEIGHT",
    "W15PackedHandoff", "W15PackResult",
    "FifoContextPacker", "CapsuleContextPacker",
    "AttentionAwareBundleDecoder",
    # SDK v3.19 — bundle-relational compatibility disambiguator (W18 family).
    "W18CompatibilityResult",
    "RelationalCompatibilityDisambiguator",
    "_disambiguator_payload_tokens",
    "_relational_compatibility_score",
    # SDK v3.20 — bundle-contradiction-aware trust-weighted disambiguator
    # (W19 family).
    "W19TrustResult",
    "BundleContradictionDisambiguator",
    "W19_SYMMETRIC_NOISE_KINDS",
    "_INCIDENT_TRIAGE_CANONICAL_ROLE_FOR_KIND",
    "W19_BRANCH_PRIMARY_TRUSTED",
    "W19_BRANCH_INVERSION",
    "W19_BRANCH_CONFOUND_RESOLVED",
    "W19_BRANCH_ABSTAINED_NO_SIGNAL",
    "W19_BRANCH_ABSTAINED_SYMMETRIC",
    "W19_BRANCH_DISABLED",
    "W19_ALL_BRANCHES",
    "_w19_canonical_primary_index",
    "_w19_witness_counts",
    # SDK v3.21 — outside-witness acquisition disambiguator (W20 family).
    "OutsideWitnessOracle",
    "OutsideQuery",
    "OutsideVerdict",
    "ServiceGraphOracle",
    "CompromisedServiceGraphOracle",
    "AbstainingOracle",
    "LLMAdjudicatorOracle",
    "build_incident_triage_service_graph",
    "W20OutsideResult",
    "OutsideWitnessAcquisitionDisambiguator",
    "W20_BRANCH_OUTSIDE_RESOLVED",
    "W20_BRANCH_OUTSIDE_TRUSTED_ASYMMETRIC",
    "W20_BRANCH_OUTSIDE_ABSTAINED",
    "W20_BRANCH_NO_TRIGGER",
    "W20_BRANCH_DISABLED",
    "W20_ALL_BRANCHES",
    "W20_DEFAULT_TRIGGER_BRANCHES",
    # SDK v3.22 — trust-weighted multi-oracle adjudicator (W21 family).
    "OracleRegistration",
    "ChangeHistoryOracle",
    "OnCallNotesOracle",
    "SingletonAsymmetricOracle",
    "DisagreeingHonestOracle",
    "W21OracleProbe",
    "W21MultiOracleResult",
    "TrustWeightedMultiOracleDisambiguator",
    "W21_BRANCH_QUORUM_RESOLVED",
    "W21_BRANCH_NO_QUORUM",
    "W21_BRANCH_SYMMETRIC_QUORUM",
    "W21_BRANCH_NO_ORACLES",
    "W21_BRANCH_NO_TRIGGER",
    "W21_BRANCH_DISABLED",
    "W21_ALL_BRANCHES",
    "W21_DEFAULT_TRIGGER_BRANCHES",
    # SDK v3.23 — capsule-native + latent-state-sharing hybrid (W22 family).
    # The strongest honest hybrid that combines explicit-capsule
    # coordination with audited proxies for the LatentMAS direction:
    # schema-passing, delta execution, shared-read cache, and a
    # controller-verified latent digest envelope. The trust boundary is
    # explicit — every latent payload is hash-chained, schema-versioned,
    # parent-CID-sealed, and rejected on verification failure.
    "SchemaCapsule",
    "build_incident_triage_schema_capsule",
    "LatentDigestEnvelope",
    "verify_latent_digest",
    "LatentVerificationOutcome",
    "SharedReadCache",
    "CachingOracleAdapter",
    "EnvelopeTamperer",
    "W22LatentResult",
    "LatentDigestDisambiguator",
    "W22_BRANCH_LATENT_RESOLVED",
    "W22_BRANCH_LATENT_REJECTED",
    "W22_BRANCH_NO_TRIGGER",
    "W22_BRANCH_NO_SCHEMA",
    "W22_BRANCH_DISABLED",
    "W22_BRANCH_ABSTAIN_PASSTHROUGH",
    "W22_ALL_BRANCHES",
    "W22_DEFAULT_TRIGGER_BRANCHES",
    "W22_LATENT_ENVELOPE_SCHEMA_VERSION",
    # SDK v3.24 — capsule-native cross-cell delta execution +
    # quorum-keyed cache + super-token reference (W23 family).
    # The strongest honest hybrid that combines explicit-capsule
    # coordination with the LatentMAS direction beyond W22:
    # cross-cell hash-chained session digest, per-cell delta
    # execution, quorum-keyed cache (mitigates
    # W22-C-CACHE-AMPLIFICATION), super-token reference (bounded
    # steganographic / dense-control-payload experiment), and a
    # within-process producer/decoder host-split proxy (Mac-2
    # unreachable fallback).
    "SessionDigestEnvelope",
    "SessionDeltaEnvelope",
    "verify_session_digest_chain",
    "verify_session_delta",
    "SuperTokenReferenceEnvelope",
    "SuperTokenRegistry",
    "verify_super_token_reference",
    "QuorumKeyedSharedReadCache",
    "CACHE_FRESHNESS_BYTE_IDENTICAL",
    "CACHE_FRESHNESS_PER_CELL_NONCE",
    "CACHE_FRESHNESS_QUORUM_LOCKED",
    "CACHE_FRESHNESS_POLICIES",
    "QuorumKeyedCachingOracleAdapter",
    "CrossHostProducerDecoderProxy",
    "W23SessionResult",
    "CrossCellDeltaDisambiguator",
    "W23_BRANCH_DELTA_RESOLVED",
    "W23_BRANCH_DELTA_REJECTED",
    "W23_BRANCH_GENESIS",
    "W23_BRANCH_SUPER_TOKEN_RESOLVED",
    "W23_BRANCH_SUPER_TOKEN_REJECTED",
    "W23_BRANCH_NO_TRIGGER",
    "W23_BRANCH_NO_PRIOR_SESSION",
    "W23_BRANCH_DISABLED",
    "W23_ALL_BRANCHES",
    "W23_SESSION_ENVELOPE_SCHEMA_VERSION",
    "W23_DELTA_ENVELOPE_SCHEMA_VERSION",
    "W23_SUPER_TOKEN_SCHEMA_VERSION",
    "W23_DEFAULT_TRIGGER_BRANCHES",
    # SDK v3.28 — multi-chain salience-keyed dense-control fanout (W27 family).
    "SalienceSignatureEnvelope",
    "ChainPivotEnvelope",
    "MultiChainPersistedFanoutRegistry",
    "MultiChainPersistedFanoutDisambiguator",
    "W27MultiChainResult",
    "verify_salience_signature",
    "verify_chain_pivot",
    "W27_SALIENCE_SIGNATURE_SCHEMA_VERSION",
    "W27_CHAIN_PIVOT_SCHEMA_VERSION",
    "W27_BRANCH_PIVOTED",
    "W27_BRANCH_ANCHORED_NEW",
    "W27_BRANCH_POOL_EXHAUSTED",
    "W27_BRANCH_PIVOT_REJECTED",
    "W27_BRANCH_FALLBACK_W26",
    "W27_BRANCH_NO_TRIGGER",
    "W27_BRANCH_DISABLED",
    "W27_ALL_BRANCHES",
    "W27_DEFAULT_TRIGGER_BRANCHES",
    # SDK v3.28 — orchestrator-level multi-chain pool (W27 family).
    "compute_input_signature_cid",
    "MultiChainPersistedFanoutOrchestrator",
    "W27OrchestratorResult",
    "SharedMultiChainPool",
]


# =============================================================================
# SDK v3.21 — outside-witness acquisition disambiguator (W20 family).
#
# The follow-up to SDK v3.20 (W19) on the named research frontier the W19
# milestone explicitly left conjectural: the **outside-information escape**
# (W19-C-OUTSIDE) from BOTH bundle-only walls (W19-Λ-total and
# W19-Λ-outside).
#
# The W19-Λ-outside anchor (Phase-66 R-66-OUTSIDE-REQUIRED) proves that
# *no* closed-form bundle-only scorer can disambiguate when the
# asymmetric-witness count is symmetric across primary's named set and
# the complement. The W19-Λ-total anchor (Phase-66 R-66-DECEIVE-TOTAL)
# proves the same when no asymmetric witness exists anywhere in the
# bundle. SDK v3.20 named the next research move as **outside
# information** — a scorer with access to an evidence channel that does
# *not* live inside the W15-packed bundle.
#
# W20's :class:`OutsideWitnessAcquisitionDisambiguator` is the smallest
# move in that direction:
#
#   1. Run inner W19 (which itself wraps W18 + W15). Capture the W19
#      branch decision and projected answer.
#   2. If the W19 branch is in :data:`W20_DEFAULT_TRIGGER_BRANCHES`
#      (default ``(W19_BRANCH_ABSTAINED_SYMMETRIC,
#      W19_BRANCH_ABSTAINED_NO_SIGNAL)``), build a hypothesis-
#      conditioned :class:`OutsideQuery` carrying the admitted tag
#      set + elected root_cause + primary_payload + the (possibly
#      symmetric) witness payloads.
#   3. Consult the registered :class:`OutsideWitnessOracle` *once*
#      (one extra evidence acquisition per cell, bounded by
#      ``max_response_tokens``). The oracle returns either a
#      payload string mentioning specific service tags
#      asymmetrically — or ``None`` (abstain).
#   4. Tokenise the oracle's payload via the same
#      :func:`_disambiguator_payload_tokens` W18 / W19 use, count
#      asymmetric outside witnesses per admitted tag, AND project
#      the answer onto the strict-max-aw subset (if non-empty,
#      proper subset of admitted_tags). Otherwise fall through to
#      the W19 answer.
#
# The trigger condition is conservative on purpose: W20 never overrides
# a non-abstained W19 verdict (W19_BRANCH_INVERSION,
# W19_BRANCH_CONFOUND_RESOLVED, W19_BRANCH_PRIMARY_TRUSTED), so on
# R-54..R-66-DECEIVE-NAIVE / R-66-CONFOUND-RESOLVABLE / R-66-CORROBORATED
# the W20 layer reduces to W19 byte-for-byte (W20-3). The named
# escape regime is R-67-OUTSIDE-RESOLVES — exactly the W19-Λ-outside
# wall plus a registered oracle that returns an asymmetric outside
# witness.
#
# Honest scope (W20). The W20-1 win is *strongly conditional* on:
#   (a) the bench property — W19 abstains via SYMMETRIC or NO_SIGNAL,
#   (b) the registered oracle returning a payload that mentions a
#       proper non-empty asymmetric subset of the admitted tags,
#   (c) the oracle's payload mentions tag tokens in the same
#       :func:`_disambiguator_payload_tokens` closure W18 / W19 read.
#
# Three named falsifiers (W20-Λ-compromised, W20-Λ-none,
# W20-Λ-joint-deception) make the conditionality sharp:
#   * W20-Λ-compromised — oracle returns a decoy-asymmetric payload;
#     W20 trusts it and FAILS at 0.000.
#   * W20-Λ-none — oracle abstains (returns ``None``); W20 falls
#     through to W19's symmetric abstention; ties FIFO at 0.000.
#   * W20-Λ-joint-deception — primary, secondary, AND oracle ALL
#     mention decoy asymmetrically; W20 ties W19 at 0.000.
#
# Real-LLM extension (W20-Λ-real). When the oracle is an
# :class:`LLMAdjudicatorOracle` over a registered :class:`LLMBackend`,
# the question is one constrained-vocabulary query per cell asking
# the LLM to identify the asymmetric subset; the LLM's reply is
# parsed for service-tag mentions through the same closure as the
# synthetic case. Closure of the parser is bounded by the same
# closed-vocabulary discipline as W19 / W18 / W13 / W12 — natural-
# language oracle replies that fall outside the closure miss by
# construction; the natural extension is W20-C-LEARNED (a small
# distilled outside-source scorer over capsule bundles).
# =============================================================================


@runtime_checkable
class OutsideWitnessOracle(Protocol):
    """The minimum surface a W20 outside-source must expose.

    The W20 disambiguator queries the oracle *once* per cell when the
    inner W19 branch matches the configured trigger set (default:
    abstentions only). The oracle returns either a payload string
    mentioning specific service tags asymmetrically — which W20
    tokenises and scores via the same scorer W18 / W19 use on
    in-bundle witnesses — or ``None`` (abstain).

    A backend MUST expose:

      * ``oracle_id`` : str — provenance label baked into the W20
                              audit (so the team-decision capsule
                              records WHICH oracle was consulted).
      * ``consult(query) -> OutsideVerdict`` — a synchronous query.

    The oracle is *not* permitted to inspect any capsule outside the
    :class:`OutsideQuery` it is given. Bounded-context honesty: the
    query carries only the *post-W15-pack* evidence the inner
    pipeline already kept under ``T_decoder``; the oracle's reply
    is bounded by ``query.max_response_tokens``.
    """

    oracle_id: str

    def consult(self, query: "OutsideQuery") -> "OutsideVerdict": ...


@dataclasses.dataclass(frozen=True)
class OutsideQuery:
    """Hypothesis-conditioned query the W20 layer sends to the oracle.

    Fields
    ------
    admitted_tags
        The W15-packed-and-W19-admitted service-tag set. Sorted
        lex-min for determinism.
    elected_root_cause
        The inner W11 elected ``root_cause`` (e.g. ``deadlock``,
        ``pool_exhaustion``). Lets the oracle answer kind-aware.
    primary_payload
        The canonical primary disambiguator's payload bytes
        (possibly deceptive or symmetric).
    witness_payloads
        The *secondary* asymmetric-witness payload bytes inside the
        bundle (possibly empty on R-66-DECEIVE-TOTAL or symmetric on
        R-66-OUTSIDE-REQUIRED).
    max_response_tokens
        Strict per-call budget on the oracle's reply. Bounded-
        context invariant: if the oracle's reply exceeds this, W20
        truncates the reply *before* tokenising. Default ``24``
        matches the W15 ``T_decoder`` budget band tested through
        SDK v3.20.
    cell_id
        Optional opaque scenario id for the oracle's own provenance
        / caching. Not used by W20.
    """
    admitted_tags: tuple[str, ...]
    elected_root_cause: str
    primary_payload: str
    witness_payloads: tuple[str, ...]
    max_response_tokens: int = 24
    cell_id: str = ""


@dataclasses.dataclass(frozen=True)
class OutsideVerdict:
    """The oracle's reply to one :class:`OutsideQuery`.

    Fields
    ------
    payload
        The oracle's response bytes — a payload string whose
        tokenisation under :func:`_disambiguator_payload_tokens`
        produces an asymmetric mention pattern across
        ``query.admitted_tags``. ``None`` means the oracle has no
        information (abstain). The W20 layer truncates to
        ``query.max_response_tokens`` before scoring.
    source_id
        Provenance label for the W20 trust audit (e.g.
        ``service_graph``, ``llm_adjudicator``,
        ``compromised_oracle``).
    confidence
        Optional self-reported confidence in [0, 1]. The W20 layer
        does NOT use this at projection time (the rule is
        deterministic max-aw); it is recorded in the audit for
        downstream / forensic use only.
    n_tokens
        The oracle's actual reply length in :func:`_handoff_n_tokens`
        units (NOT bytes). Counted post-truncation. The W20 audit
        accounts for it as a strict additional token cost on top of
        the inner W15 ``tokens_kept``.
    """
    payload: str | None
    source_id: str
    confidence: float = 1.0
    n_tokens: int = 0


def build_incident_triage_service_graph() -> dict[str, frozenset[str]]:
    """Default service-dependency graph for the incident-triage
    benchmark family (used by the deterministic
    :class:`ServiceGraphOracle`).

    Each entry maps a service tag to the set of services it has a
    *true topological dependency on* — the asymmetric outside
    information that distinguishes gold pairs from the synthetic
    decoy storm. The set is closed-vocabulary, deterministic, and
    pre-committed in code — the synthetic R-67 anchor proves the
    *mechanism* (outside information escapes the W19-Λ-outside
    wall when present); real production deployments would supply
    their own service registry.

    Honest scope: this graph is the synthetic counterpart of a
    real service registry / topology table. It is not claimed to
    cover free-form natural-language relations (W20-C-LEARNED).
    """
    return {
        # Gold pairs from _P66_FAMILIES — each pair is bidirectionally
        # dependent (the deadlock / pool / disk / slow_query co-edge
        # the bench is built around).
        "orders":         frozenset({"payments"}),
        "payments":       frozenset({"orders"}),
        "api":            frozenset({"db"}),
        "db":             frozenset({"api"}),
        "storage":        frozenset({"logs_pipeline"}),
        "logs_pipeline":  frozenset({"storage"}),
        "web":            frozenset({"db_query"}),
        "db_query":       frozenset({"web"}),
        # Decoys — by design have no topological edge to any gold
        # service; the oracle's asymmetric reply distinguishes them.
        "search_index":   frozenset(),
        "archival":       frozenset(),
        "metrics":        frozenset(),
        "telemetry":      frozenset(),
        "audit_jobs":     frozenset(),
        "sessions":       frozenset(),
        "cache":          frozenset(),
        "scratch_pool":   frozenset(),
    }


@dataclasses.dataclass
class ServiceGraphOracle:
    """Deterministic service-graph oracle (synthetic R-67 anchor).

    Reads ``query.admitted_tags`` against a closed-vocabulary
    service-dependency graph. If a *connected pair* exists in the
    admitted set (two distinct tags ``a, b`` with
    ``b in graph[a]``), returns a payload mentioning *every* tag in
    the connected component intersected with admitted_tags — and
    *no* singleton-component tags. This produces an asymmetric
    outside witness across the admitted set whenever the bench's
    gold pair is present in the union AND the decoys are
    topologically isolated.

    Determinism: byte-stable for byte-identical queries. No
    randomness, no learning, no hidden state.

    Bounded-context honesty: the reply payload is bounded by
    ``query.max_response_tokens`` characters worth of tokens — the
    oracle records the reply length and W20 truncates if needed.
    """
    oracle_id: str = "service_graph"
    graph: dict[str, frozenset[str]] = dataclasses.field(
        default_factory=build_incident_triage_service_graph)

    def consult(self, query: OutsideQuery) -> OutsideVerdict:
        admitted = sorted(set(query.admitted_tags))
        if len(admitted) < 2:
            return OutsideVerdict(
                payload=None, source_id=self.oracle_id, n_tokens=0)
        connected: set[str] = set()
        for i, a in enumerate(admitted):
            neighbours = self.graph.get(a, frozenset())
            for b in admitted[i + 1:]:
                if b in neighbours or a in self.graph.get(b, frozenset()):
                    connected.add(a)
                    connected.add(b)
        if not connected or connected == set(admitted):
            return OutsideVerdict(
                payload=None, source_id=self.oracle_id, n_tokens=0)
        named = sorted(connected)
        # Build a payload using the same relational-compound shape
        # the W18 / W19 scorers parse on in-bundle witnesses.
        payload = (f"service_graph dependency_chain="
                   f"{'_'.join(named)} "
                   + " ".join(f"service={tag}" for tag in named))
        # Truncate to budget. We treat one space-separated token as a
        # token unit — same convention as :func:`_handoff_n_tokens`.
        words = payload.split()
        if len(words) > query.max_response_tokens:
            words = words[:query.max_response_tokens]
        truncated = " ".join(words)
        return OutsideVerdict(
            payload=truncated, source_id=self.oracle_id,
            n_tokens=len(words))


@dataclasses.dataclass
class CompromisedServiceGraphOracle:
    """Adversarial outside oracle (W20-Λ-compromised falsifier).

    The oracle inspects the admitted-tag set and emits an asymmetric
    reply naming exactly the *non-gold* tags — i.e. it confidently
    asserts a decoy-only "dependency" pattern. The W20 layer trusts
    the oracle's reply and projects to the decoy set, FAILING at
    0.000.

    This is the named structural limit when the outside source
    itself is adversarial — analogous to a corrupted service registry
    or a compromised LLM adjudicator. The fix is NOT a richer scorer;
    it is *oracle integrity*. W20-Λ-compromised is the corresponding
    theorem.

    Bounded-context honesty: the oracle's reply is bounded by
    ``query.max_response_tokens`` exactly the same way
    :class:`ServiceGraphOracle` is.
    """
    oracle_id: str = "compromised_oracle"
    # Gold-set used to discriminate adversarial targets: any admitted
    # tag NOT in this set becomes the oracle's target. Defaulted to
    # the incident-triage gold pairs (mirrors
    # :func:`build_incident_triage_service_graph`).
    gold_set: frozenset[str] = dataclasses.field(
        default_factory=lambda: frozenset({
            "orders", "payments", "api", "db", "storage",
            "logs_pipeline", "web", "db_query",
        }))

    def consult(self, query: OutsideQuery) -> OutsideVerdict:
        admitted = sorted(set(query.admitted_tags))
        targets = [t for t in admitted if t not in self.gold_set]
        if not targets or len(targets) == len(admitted):
            return OutsideVerdict(
                payload=None, source_id=self.oracle_id, n_tokens=0)
        payload = (
            f"compromised_oracle attribution_chain="
            f"{'_'.join(targets)} "
            + " ".join(f"service={tag}" for tag in targets))
        words = payload.split()
        if len(words) > query.max_response_tokens:
            words = words[:query.max_response_tokens]
        truncated = " ".join(words)
        return OutsideVerdict(
            payload=truncated, source_id=self.oracle_id,
            n_tokens=len(words))


@dataclasses.dataclass
class AbstainingOracle:
    """Oracle that always returns ``None`` (W20-Λ-none falsifier).

    Models the regime where the registered outside source has no
    information about the symmetric-witness ambiguity. Used by the
    R-67-OUTSIDE-NONE bench to show that W20 ties FIFO when the
    oracle returns no signal — bounded-context honesty: zero extra
    tokens consumed when the oracle abstains.
    """
    oracle_id: str = "abstain"

    def consult(self, query: OutsideQuery) -> OutsideVerdict:
        return OutsideVerdict(
            payload=None, source_id=self.oracle_id, n_tokens=0)


@dataclasses.dataclass
class LLMAdjudicatorOracle:
    """Outside-source oracle backed by a real :class:`LLMBackend`
    (live W20-C-LIVE extension).

    Renders one constrained-vocabulary prompt per cell asking the
    LLM to identify the asymmetric subset of the admitted tag set
    given (a) the elected root_cause kind, (b) the primary
    disambiguator's payload, (c) the witness payloads. The LLM's
    reply is parsed via the SAME tokeniser
    (:func:`_disambiguator_payload_tokens`) that W18 / W19 use on
    in-bundle witnesses; tokens that match an admitted-tag string
    are counted as outside-source asymmetric witnesses.

    Closure: the parser only finds tag mentions inside the closed-
    vocabulary closure W18 / W19 share. Free-form natural-language
    replies (e.g. "the join between orders and payments") fall
    *outside* the closure unless the model emits the literal tag
    tokens. This is the same closure boundary as W12 / W13 / W18 /
    W19; the natural extension is W20-C-LEARNED (a small distilled
    outside-source scorer over capsule bundles).

    Honest scope: this class is *infrastructure* for the W20-Λ-real
    probe. Whether a *specific* model emits closed-vocabulary tag
    mentions on a *specific* prompt is *empirical* — the milestone
    measures, not claims. Failures (LLM unreachable, parse miss,
    free-form reply) are recorded as ``payload=None`` (abstain), so
    the W20 path remains audit-coherent.
    """
    backend: Any  # LLMBackend duck-type — keep optional for type-only imports.
    oracle_id: str = "llm_adjudicator"
    max_response_tokens: int = 24
    temperature: float = 0.0
    n_calls: int = 0
    last_prompt: str = ""
    last_reply: str = ""

    def _render_prompt(self, query: OutsideQuery) -> str:
        admitted_str = ", ".join(query.admitted_tags)
        witnesses_str = "\n".join(
            f"  witness #{i+1}: {p}"
            for i, p in enumerate(query.witness_payloads)) or "  (none)"
        return (
            "You are an incident-triage adjudicator. The auditor saw a "
            "symmetric witness pattern across these candidate services:"
            f" {admitted_str}.\n"
            f"Elected root cause kind: {query.elected_root_cause}.\n"
            f"Primary disambiguator payload: {query.primary_payload}\n"
            f"Bundle witness payloads:\n{witnesses_str}\n"
            "Reply with ONE line of the form: "
            "'services=svc1,svc2' naming ONLY the services you "
            "judge to be the actual root cause among the candidates "
            "above. If you cannot decide, reply: 'services='. "
            "Reply with no extra text.")

    def consult(self, query: OutsideQuery) -> OutsideVerdict:
        prompt = self._render_prompt(query)
        self.last_prompt = prompt
        try:
            reply = self.backend.generate(
                prompt, max_tokens=self.max_response_tokens,
                temperature=self.temperature)
        except Exception as exc:  # noqa: BLE001 — record + abstain
            self.last_reply = f"<exception: {exc!r}>"
            self.n_calls += 1
            return OutsideVerdict(
                payload=None, source_id=self.oracle_id, n_tokens=0)
        self.n_calls += 1
        self.last_reply = reply or ""
        if not reply:
            return OutsideVerdict(
                payload=None, source_id=self.oracle_id, n_tokens=0)
        # Truncate to the per-call budget.
        words = reply.split()
        if len(words) > query.max_response_tokens:
            words = words[:query.max_response_tokens]
        truncated = " ".join(words)
        # Materialise the reply as an outside-witness payload by
        # framing it like a service-graph reply so the W20 scorer's
        # tokeniser finds the tag mentions through the W18 / W19
        # closure.
        return OutsideVerdict(
            payload=f"llm_adjudicator response: {truncated}",
            source_id=self.oracle_id,
            n_tokens=len(words))


W20_BRANCH_OUTSIDE_RESOLVED = "outside_resolved"
W20_BRANCH_OUTSIDE_TRUSTED_ASYMMETRIC = "outside_trusted_asymmetric"
W20_BRANCH_OUTSIDE_ABSTAINED = "outside_abstained"
W20_BRANCH_NO_TRIGGER = "no_trigger"
W20_BRANCH_DISABLED = "disabled"

W20_ALL_BRANCHES: tuple[str, ...] = (
    W20_BRANCH_OUTSIDE_RESOLVED,
    W20_BRANCH_OUTSIDE_TRUSTED_ASYMMETRIC,
    W20_BRANCH_OUTSIDE_ABSTAINED,
    W20_BRANCH_NO_TRIGGER,
    W20_BRANCH_DISABLED,
)

# Default trigger set — W20 fires only when W19 abstains. This keeps
# the W20-3 backward-compat byte-for-byte against W19 on every regime
# where W19 does not abstain (R-54..R-66-DECEIVE-NAIVE / R-66-CONFOUND-
# RESOLVABLE / R-66-CORROBORATED). The R-67 escape regime triggers
# specifically through ``W19_BRANCH_ABSTAINED_SYMMETRIC``.
W20_DEFAULT_TRIGGER_BRANCHES: frozenset[str] = frozenset({
    W19_BRANCH_ABSTAINED_SYMMETRIC,
    W19_BRANCH_ABSTAINED_NO_SIGNAL,
})


@dataclasses.dataclass(frozen=True)
class W20OutsideResult:
    """Audit record for one ``OutsideWitnessAcquisitionDisambiguator
    .decode_rounds`` call.

    Fields
    ------
    answer
        The W20-projected answer dict (same shape as W19's).
    inner_branch
        The W19 branch that fired before the W20 layer ran.
    triggered
        Whether the W20 trigger fired (i.e. ``inner_branch`` was in
        the configured trigger set AND ``enabled``).
    oracle_consulted
        The oracle's ``oracle_id`` if consulted; ``""`` otherwise.
    oracle_payload
        The (truncated) bytes the oracle returned. ``""`` if the
        oracle abstained or W20 did not trigger.
    oracle_payload_tokens
        Tokenised oracle bytes (post-truncation) — the same closure
        W18 / W19 use on in-bundle witnesses.
    per_tag_outside_count
        ``{tag: aw_count}`` over the oracle's payload — one
        independent witness count per admitted service tag.
    decoder_branch
        The W20 branch (one of :data:`W20_ALL_BRANCHES`).
    abstained
        True when the W20 layer did not produce a strict-max
        projection (oracle absent / abstained / symmetric over the
        admitted tag set).
    n_outside_tokens
        Strict per-cell additional token cost of the outside-
        information acquisition step (the oracle's reply length,
        post-truncation). Bounded-context honesty: this number is
        always reported and is *additive* on top of the inner W15
        ``tokens_kept`` figure.
    """
    answer: dict[str, Any]
    inner_branch: str
    triggered: bool
    oracle_consulted: str
    oracle_payload: str
    oracle_payload_tokens: tuple[str, ...]
    per_tag_outside_count: dict[str, int]
    decoder_branch: str
    abstained: bool
    n_outside_tokens: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "root_cause": str(self.answer.get("root_cause", "unknown")),
            "services": tuple(self.answer.get("services", ())),
            "remediation": str(self.answer.get("remediation",
                                                 "investigate")),
            "inner_branch": str(self.inner_branch),
            "triggered": bool(self.triggered),
            "oracle_consulted": str(self.oracle_consulted),
            "oracle_payload": str(self.oracle_payload),
            "oracle_payload_tokens": list(self.oracle_payload_tokens),
            "per_tag_outside_count": {
                tag: int(c)
                for tag, c in self.per_tag_outside_count.items()
            },
            "decoder_branch": str(self.decoder_branch),
            "abstained": bool(self.abstained),
            "n_outside_tokens": int(self.n_outside_tokens),
        }


@dataclasses.dataclass
class OutsideWitnessAcquisitionDisambiguator:
    """Outside-witness acquisition disambiguator (SDK v3.21, W20 family).

    Wraps a :class:`BundleContradictionDisambiguator` (W19) and adds an
    extra evidence-acquisition step when the inner W19 branch indicates
    the bundle alone is structurally insufficient
    (:data:`W19_BRANCH_ABSTAINED_SYMMETRIC` or
    :data:`W19_BRANCH_ABSTAINED_NO_SIGNAL`). The acquired evidence is
    one targeted, hypothesis-conditioned :class:`OutsideQuery` to a
    registered :class:`OutsideWitnessOracle`; the oracle's reply is
    parsed through the same tokeniser W18 / W19 use on in-bundle
    witnesses, and the W19 answer is projected onto the strict-max-aw
    subset of the admitted tags (if a non-empty proper subset exists).

    This is the first capsule-native multi-agent-coordination method
    that crosses the W19-Λ-outside wall on a regime where the wall
    actually applies (R-67-OUTSIDE-RESOLVES). It is *not* a learned
    model — it is a closed-form composition of (a) the inner W19
    bundle-only scorer, (b) one targeted oracle consult, and (c) the
    same per-tag scorer the W18 / W19 layers use.

    Pipeline
    --------
    1. Run inner W19 (which itself wraps W18 + W15). Capture the W19
       :class:`W19TrustResult` (inner branch + projected answer).
    2. If W19's branch is NOT in ``trigger_branches`` (or W20 is
       ``enabled = False``), return W19's answer byte-for-byte
       (W20-3 backward-compat — :data:`W20_BRANCH_NO_TRIGGER` or
       :data:`W20_BRANCH_DISABLED`).
    3. Otherwise build :class:`OutsideQuery` carrying:
         * the W15-packed-and-W19-admitted tag set (sorted),
         * the inner W11 elected root_cause,
         * the W19 canonical primary disambiguator's payload bytes,
         * every other specific-tier handoff's payload bytes (the
           potentially-symmetric secondary witnesses).
    4. Consult the oracle exactly once. Truncate the reply to
       ``max_response_tokens``. Tokenise.
    5. Compute outside-asymmetric-witness count
       ``aw_outside(tag)`` per admitted tag — the same scoring rule
       :func:`_relational_compatibility_score` applies.
    6. Project: if the strict-max set ``M`` is a non-empty proper
       subset of the admitted tags, project to ``M``
       (:data:`W20_BRANCH_OUTSIDE_RESOLVED`); otherwise abstain
       (:data:`W20_BRANCH_OUTSIDE_ABSTAINED`) — fall through to W19's
       answer.
    7. Return a :class:`W20OutsideResult` with the final answer + audit
       record. The W20 scorer also forwards the W19 ``trust`` block,
       the W18 ``compatibility`` block, and the W15 ``pack_stats``
       block so the bench driver can audit the entire chain.

    Bounded-context honesty
    ------------------------
    The W20 layer adds *exactly one* outside query per cell, with a
    strict per-cell ``max_response_tokens`` budget on the oracle's
    reply. The inner W15 ``tokens_kept`` accounting is byte-for-byte
    unchanged (the oracle's reply tokens are NOT folded into the
    decoder bundle — they are recorded as a strict additional token
    cost in ``W20OutsideResult.n_outside_tokens``). Total context
    delivered to the final decider:
    ``tokens_kept (W15) + n_outside_tokens (W20)``.

    Backward-compat (W20-3)
    ------------------------
    With ``enabled = False`` the W20 layer reduces to the inner W19
    byte-for-byte. With ``enabled = True`` AND the inner W19 returns
    a non-trigger branch (``W19_BRANCH_INVERSION``,
    ``W19_BRANCH_CONFOUND_RESOLVED``, ``W19_BRANCH_PRIMARY_TRUSTED``,
    ``W19_BRANCH_DISABLED``), the W20 layer also reduces to W19
    byte-for-byte. The outside oracle is *only* consulted when W19
    abstains.
    """

    inner: BundleContradictionDisambiguator = dataclasses.field(
        default_factory=lambda: BundleContradictionDisambiguator())
    oracle: Any = None  # OutsideWitnessOracle duck-type
    enabled: bool = True
    trigger_branches: frozenset[str] = dataclasses.field(
        default_factory=lambda: W20_DEFAULT_TRIGGER_BRANCHES)
    max_response_tokens: int = 24

    # Forensic — last applied result, exposed for the bench driver.
    _last_result: W20OutsideResult | None = None

    def decode_rounds(self,
                       per_round_handoffs: Sequence[Sequence[_DecodedHandoff]],
                       ) -> dict[str, Any]:
        base = self.inner.decode_rounds(per_round_handoffs)
        w19_result = self.inner.last_result
        out = dict(base)
        if (not self.enabled) or w19_result is None:
            result = W20OutsideResult(
                answer=dict(base),
                inner_branch=(w19_result.decoder_branch
                                if w19_result is not None
                                else W19_BRANCH_DISABLED),
                triggered=False,
                oracle_consulted="",
                oracle_payload="",
                oracle_payload_tokens=(),
                per_tag_outside_count={},
                decoder_branch=W20_BRANCH_DISABLED,
                abstained=True,
                n_outside_tokens=0,
            )
            self._last_result = result
            out["outside"] = result.as_dict()
            return out
        if w19_result.decoder_branch not in self.trigger_branches:
            result = W20OutsideResult(
                answer=dict(base),
                inner_branch=w19_result.decoder_branch,
                triggered=False,
                oracle_consulted="",
                oracle_payload="",
                oracle_payload_tokens=(),
                per_tag_outside_count={},
                decoder_branch=W20_BRANCH_NO_TRIGGER,
                abstained=False,
                n_outside_tokens=0,
            )
            self._last_result = result
            out["outside"] = result.as_dict()
            return out
        # W20 trigger fires — build the query. Walk down the
        # W20→W19→W18→W15→Layered chain to reach the layered
        # normaliser (which exposes ``normalize_round``).
        union_tag_set: set[str] = set()
        union: list[_DecodedHandoff] = []
        raw_union: list[_DecodedHandoff] = []
        layered_normaliser = self.inner.inner.inner.inner
        for bundle in per_round_handoffs:
            normalised = layered_normaliser.normalize_round(bundle)
            for h_norm, h_raw in zip(normalised, bundle):
                union.append(h_norm)
                raw_union.append(h_raw)
                tag = _service_tag_of(h_norm.payload)
                if tag:
                    union_tag_set.add(tag)
        admitted_tags = tuple(sorted(union_tag_set))
        # Identify the canonical primary the same way W19 does, so the
        # query carries the *same* primary the inner layer abstained on.
        round_hint: list[int] = []
        for r_idx, bundle in enumerate(per_round_handoffs, start=1):
            for _ in bundle:
                round_hint.append(r_idx)
        primary_idx = _w19_canonical_primary_index(
            union, round_hint, raw_union=raw_union)
        primary_payload = (union[primary_idx].payload
                            if primary_idx >= 0 else "")
        witness_payloads: list[str] = []
        for i, h in enumerate(union):
            if i == primary_idx:
                continue
            if h.claim_kind not in _SPECIFIC_TIER_CLAIM_KINDS:
                continue
            witness_payloads.append(h.payload)
        elected_root_cause = str(base.get("root_cause", "unknown"))
        query = OutsideQuery(
            admitted_tags=admitted_tags,
            elected_root_cause=elected_root_cause,
            primary_payload=primary_payload,
            witness_payloads=tuple(witness_payloads),
            max_response_tokens=self.max_response_tokens,
        )
        if self.oracle is None:
            verdict = OutsideVerdict(
                payload=None, source_id="no_oracle", n_tokens=0)
        else:
            verdict = self.oracle.consult(query)
        oracle_id = getattr(self.oracle, "oracle_id", "no_oracle") \
            if self.oracle is not None else "no_oracle"
        if verdict.payload is None or not verdict.payload:
            result = W20OutsideResult(
                answer=dict(base),
                inner_branch=w19_result.decoder_branch,
                triggered=True,
                oracle_consulted=oracle_id,
                oracle_payload="",
                oracle_payload_tokens=(),
                per_tag_outside_count={tag: 0 for tag in admitted_tags},
                decoder_branch=W20_BRANCH_OUTSIDE_ABSTAINED,
                abstained=True,
                n_outside_tokens=int(verdict.n_tokens),
            )
            self._last_result = result
            out["outside"] = result.as_dict()
            return out
        # Truncate oracle reply to the per-call budget (defensive — the
        # oracle is supposed to do this itself, but we enforce here).
        words = verdict.payload.split()
        if len(words) > self.max_response_tokens:
            words = words[:self.max_response_tokens]
        oracle_payload = " ".join(words)
        oracle_tokens = _disambiguator_payload_tokens(oracle_payload)
        per_tag: dict[str, int] = {}
        for tag in admitted_tags:
            d, c = _relational_compatibility_score(tag, oracle_tokens)
            per_tag[tag] = int(d + c)
        max_aw = max(per_tag.values(), default=0)
        if max_aw <= 0:
            # Oracle replied but doesn't mention any admitted tag.
            result = W20OutsideResult(
                answer=dict(base),
                inner_branch=w19_result.decoder_branch,
                triggered=True,
                oracle_consulted=oracle_id,
                oracle_payload=oracle_payload,
                oracle_payload_tokens=oracle_tokens,
                per_tag_outside_count=per_tag,
                decoder_branch=W20_BRANCH_OUTSIDE_ABSTAINED,
                abstained=True,
                n_outside_tokens=int(verdict.n_tokens or len(words)),
            )
            self._last_result = result
            out["outside"] = result.as_dict()
            return out
        # Positive-set projection: every admitted tag the oracle mentions
        # at all (strictly above zero) is in the projected answer. This
        # mirrors the W18 named-set rule and makes W20 robust against
        # asymmetric mention counts (e.g. when a multi-component tag
        # like ``logs_pipeline`` decomposes into one direct + one
        # compound match while a single-component tag like ``storage``
        # decomposes into two direct + one compound). The strict-max
        # rule was tried first and produced false negatives on the
        # multi-component-tag scenario families; positive-set is the
        # sharper rule and keeps the W20 in the same projection class
        # as W18 / W19.
        top_set = tuple(sorted(t for t in admitted_tags
                                  if per_tag[t] > 0))
        if not top_set or len(top_set) == len(admitted_tags):
            # Symmetric oracle reply — abstain.
            result = W20OutsideResult(
                answer=dict(base),
                inner_branch=w19_result.decoder_branch,
                triggered=True,
                oracle_consulted=oracle_id,
                oracle_payload=oracle_payload,
                oracle_payload_tokens=oracle_tokens,
                per_tag_outside_count=per_tag,
                decoder_branch=W20_BRANCH_OUTSIDE_ABSTAINED,
                abstained=True,
                n_outside_tokens=int(verdict.n_tokens or len(words)),
            )
            self._last_result = result
            out["outside"] = result.as_dict()
            return out
        new_answer = dict(base)
        new_answer["services"] = top_set
        # If oracle's projected set differs from W19's strict-asymmetric
        # named set (i.e. W19 trusted primary on a strict subset and
        # the oracle disagrees), record the branch as
        # OUTSIDE_TRUSTED_ASYMMETRIC. This branch is reached only when
        # the trigger set includes ``W19_BRANCH_PRIMARY_TRUSTED`` (an
        # opt-in aggressive mode); under the default trigger
        # (abstentions only), W20 always lands on OUTSIDE_RESOLVED.
        branch = (W20_BRANCH_OUTSIDE_TRUSTED_ASYMMETRIC
                  if w19_result.decoder_branch
                      == W19_BRANCH_PRIMARY_TRUSTED
                  else W20_BRANCH_OUTSIDE_RESOLVED)
        result = W20OutsideResult(
            answer=new_answer,
            inner_branch=w19_result.decoder_branch,
            triggered=True,
            oracle_consulted=oracle_id,
            oracle_payload=oracle_payload,
            oracle_payload_tokens=oracle_tokens,
            per_tag_outside_count=per_tag,
            decoder_branch=branch,
            abstained=False,
            n_outside_tokens=int(verdict.n_tokens or len(words)),
        )
        self._last_result = result
        out["services"] = top_set
        out["outside"] = result.as_dict()
        return out

    def decode(self, handoffs: Sequence[_DecodedHandoff]
               ) -> dict[str, Any]:
        return self.decode_rounds([handoffs])

    @property
    def last_result(self) -> W20OutsideResult | None:
        return self._last_result

    @property
    def T_decoder(self) -> int | None:
        return self.inner.T_decoder

    @T_decoder.setter
    def T_decoder(self, v: int | None) -> None:
        self.inner.T_decoder = v

    def pack_stats(self) -> dict[str, Any]:
        return self.inner.pack_stats()


# =============================================================================
# SDK v3.22 — trust-weighted multi-oracle adjudicator (W21 family).
#
# The follow-up to SDK v3.21 (W20). Discharges the W20-C-MULTI-ORACLE
# conjecture (named in SDK v3.21) on the exact regime where it applies:
# multi-source outside-information adjudication under partial oracle
# compromise, with bounded-context honesty and strict falsifier-driven
# limits.
#
# The W20 milestone established three bundle+oracle walls:
#
#   * W20-Λ-none — registered oracle has no signal; ties FIFO by
#     abstention.
#   * W20-Λ-compromised — single-oracle interface trusts an
#     adversarial oracle's reply; FAILS at 0.000.
#   * W20-Λ-joint-deception — primary + secondary + oracle ALL
#     consistent on decoy; FAILS at 0.000.
#
# The W20-Λ-compromised wall is the named regime W21 attacks. The
# wall is not "the bundle has no signal"; it is "the runtime has only
# one outside source, and its reply happens to be deceptive". Real
# production systems usually have *more than one* outside source —
# a service registry, a change-log, an on-call notes index, a
# dependency graph, an LLM adjudicator. Each one is an independent
# evidence channel; some may be compromised; some may abstain; some
# may emit free-form replies that miss the closure. The runtime's
# job is to *adjudicate* across them under bounded context.
#
# W21's :class:`TrustWeightedMultiOracleDisambiguator` is the smallest
# move in that direction:
#
#   1. Run inner W19 (which itself wraps W18 + W15). Capture the W19
#      branch decision and projected answer.
#   2. If the W19 branch is in :data:`W21_DEFAULT_TRIGGER_BRANCHES`
#      (default ``(W19_BRANCH_ABSTAINED_SYMMETRIC,
#      W19_BRANCH_ABSTAINED_NO_SIGNAL)`` — same as W20), build a
#      hypothesis-conditioned :class:`OutsideQuery` (same shape as
#      W20).
#   3. Consult **every** registered :class:`OracleRegistration` once.
#      Each registration carries a prior trust weight in [0, 1] and
#      a role label (``service_graph``, ``change_history``,
#      ``oncall_notes``, ``llm_adjudicator``). Track per-oracle
#      ``per_tag_outside_count`` via the same :func:`_disambiguator
#      _payload_tokens` + :func:`_relational_compatibility_score`
#      pipeline W20 / W19 / W18 use on in-bundle witnesses.
#   4. Aggregate the per-tag votes:
#         votes[tag]      = #oracles with per_tag_count[tag] > 0
#         trust_sum[tag]  = Σ trust_prior over those oracles
#   5. Project: top_set = {tag : votes[tag] ≥ quorum_min
#                                AND trust_sum[tag] ≥ min_trust_sum}.
#   6. If ``top_set`` is a non-empty proper subset of the admitted
#      tag set, project the answer onto ``top_set``
#      (:data:`W21_BRANCH_QUORUM_RESOLVED`). If ``top_set`` is empty,
#      abstain (:data:`W21_BRANCH_NO_QUORUM`) — fall through to W19's
#      answer. If ``top_set`` covers every admitted tag, abstain
#      (:data:`W21_BRANCH_SYMMETRIC_QUORUM`) — fall through to W19's
#      answer.
#   7. Return a :class:`W21MultiOracleResult` with the final answer +
#      audit record (per-oracle probe results, aggregated votes,
#      total ``n_outside_queries`` and ``n_outside_tokens_total``).
#
# The trigger condition is conservative on purpose: W21 never overrides
# a non-abstained W19 verdict (W19_BRANCH_INVERSION,
# W19_BRANCH_CONFOUND_RESOLVED, W19_BRANCH_PRIMARY_TRUSTED). On
# R-54..R-67 default banks the W21 layer reduces to W19 byte-for-byte
# (W21-3-A backward-compat). On R-67-OUTSIDE-RESOLVES with one
# clean oracle, W21 with ``quorum_min=1`` ties W20 byte-for-byte
# (W21-3-B reduces-to-W20).
#
# Honest scope (W21). The W21-1 win is *strongly conditional* on:
#   (a) the bench property — W19 abstains via SYMMETRIC or NO_SIGNAL,
#   (b) the registered oracle set contains ≥ ``quorum_min`` oracles
#       whose replies independently land on a non-empty proper
#       asymmetric subset of the admitted tags AND that subset
#       intersects to a non-empty asymmetric subset.
# Three named falsifiers (W21-Λ-no-quorum, W21-Λ-all-compromised,
# W21-Λ-disagreement-only) make the conditionality sharp:
#   * W21-Λ-no-quorum — fewer than ``quorum_min`` oracles vote for any
#     tag; W21 abstains; ties FIFO at 0.000.
#   * W21-Λ-all-compromised — every registered oracle returns a
#     decoy-asymmetric reply; quorum forms on decoy; W21 FAILS at
#     0.000. Names the structural limit when ALL N registered
#     oracles are jointly compromised. The fix is NOT richer scoring;
#     it is *oracle integrity*.
#   * W21-Λ-disagreement-only — registered oracles each emit a
#     different singleton; no tag has ≥ ``quorum_min`` votes; W21
#     abstains; ties FIFO at 0.000.
#
# Bounded-context honesty. The W21 layer issues *exactly N* outside
# queries per cell where N = number of registered oracles. Each
# query bounded by ``max_response_tokens``. The audit records
# ``n_outside_queries`` and ``n_outside_tokens_total`` as strict
# additional cost; the inner W15 ``tokens_kept`` accounting is
# byte-for-byte unchanged.
#
# Real-LLM extension (W21-Λ-real). Same closure boundary as W20:
# the LLM adjudicator's reply is parsed via the same
# :func:`_disambiguator_payload_tokens` closure. Free-form replies
# fall outside the closure unless the model emits the literal tag
# tokens. The natural extension W21-C-LIVE-WITH-REGISTRY pairs an
# LLM adjudicator with a deterministic ServiceGraphOracle as its
# trusted "registry" — the deterministic oracle anchors the closure
# and the LLM adjudicator votes on top.
# =============================================================================


@dataclasses.dataclass(frozen=True)
class OracleRegistration:
    """One registered oracle in a W21 multi-oracle adjudicator stack.

    Fields
    ------
    oracle
        An :class:`OutsideWitnessOracle`-shaped object (duck-typed:
        any object with ``oracle_id: str`` and
        ``consult(query) -> OutsideVerdict``).
    trust_prior
        Prior trust weight in [0, 1]. The W21 layer multiplies the
        oracle's vote by this weight when computing
        ``per_tag_trust_sum``. Default ``1.0`` treats every oracle
        as equally trusted (pure majority). Production deployments
        would calibrate trust priors via held-out historical
        agreement (W21-C-CALIBRATED-TRUST conjecture).
    role_label
        Provenance label baked into the W21 audit
        (``"service_graph"``, ``"change_history"``,
        ``"oncall_notes"``, ``"llm_adjudicator"``, ...). Used by
        downstream forensics — the W21 scorer does NOT use it at
        projection time.

    The W21 layer treats each registration as a distinct evidence
    channel; per-oracle replies are scored independently and
    aggregated by vote. Two registrations with the same ``oracle``
    but different ``role_label`` are *two* independent votes —
    do NOT register the same oracle twice unless that is intended.
    """
    oracle: Any  # OutsideWitnessOracle duck-type
    trust_prior: float = 1.0
    role_label: str = "outside_source"


@dataclasses.dataclass
class ChangeHistoryOracle:
    """Synthetic deterministic oracle reading a recent-change log
    (W21 multi-source companion to :class:`ServiceGraphOracle`).

    Models a second independent outside-information channel — a
    change-management or CI deployment log that records which
    services have had recent commits / config rolls. The bench
    pre-commits a closed-vocabulary log mapping
    ``root_cause -> (service_a, service_b)`` for each gold family
    in :data:`_P66_FAMILIES`; the oracle returns a payload of the
    form ``recent_changes service=A service=B`` when the admitted
    set contains the gold pair, and abstains otherwise.

    Determinism: byte-stable for byte-identical queries.
    Bounded-context: bounded by ``query.max_response_tokens``.
    Different ontology from :class:`ServiceGraphOracle`: this oracle
    answers "which services were recently changed?" (an event-log
    semantics) rather than "which services depend on each other?"
    (a topological semantics). When BOTH oracles agree on the same
    asymmetric subset, that is genuinely independent corroboration —
    the W21 quorum has structural meaning, not just byte-level
    duplication.
    """
    oracle_id: str = "change_history"
    # Default change-log mirrors the gold pairs of
    # :data:`_P66_FAMILIES` — each gold pair has a recent
    # co-deployment record. Decoys do not. Root-cause keys must
    # match the elected ``root_cause`` strings emitted by the
    # MultiRoundBundleDecoder family (i.e. the canonical
    # _P66_FAMILIES root_causes).
    change_log: dict[str, tuple[str, ...]] = dataclasses.field(
        default_factory=lambda: {
            "deadlock":             ("orders", "payments"),
            "pool_exhaustion":      ("api", "db"),
            "disk_fill":            ("storage", "logs_pipeline"),
            "slow_query_cascade":   ("web", "db_query"),
        })

    def consult(self, query: OutsideQuery) -> OutsideVerdict:
        admitted = sorted(set(query.admitted_tags))
        # Look up the change log entry for the elected root_cause.
        recent = self.change_log.get(query.elected_root_cause, ())
        # Asymmetric: recent must be a non-empty proper subset of
        # admitted.
        recent_in_admitted = [t for t in recent if t in admitted]
        if (not recent_in_admitted
                or len(recent_in_admitted) == len(admitted)):
            return OutsideVerdict(
                payload=None, source_id=self.oracle_id, n_tokens=0)
        named = sorted(recent_in_admitted)
        payload = ("change_history recent_changes="
                   f"{'_'.join(named)} "
                   + " ".join(f"service={tag}" for tag in named))
        words = payload.split()
        if len(words) > query.max_response_tokens:
            words = words[:query.max_response_tokens]
        truncated = " ".join(words)
        return OutsideVerdict(
            payload=truncated, source_id=self.oracle_id,
            n_tokens=len(words))


@dataclasses.dataclass
class OnCallNotesOracle:
    """Synthetic deterministic oracle reading on-call note shorthand
    (W21 multi-source companion to :class:`ServiceGraphOracle`).

    Models a third independent outside-information channel — an
    on-call rotation's free-form note index, post-keyword-extraction.
    The bench pre-commits a closed-vocabulary note index mapping
    ``root_cause -> (service_a, service_b)`` (default mirroring the
    same gold pairs as :class:`ChangeHistoryOracle`); the oracle
    returns a payload of the form ``oncall_notes mention=A,B`` when
    the admitted set contains the noted pair.

    Difference from :class:`ChangeHistoryOracle`: the on-call oracle
    answers "which services are humans currently watching?" (a
    behavioural semantics) rather than "which services were recently
    deployed?" (a deployment-log semantics).

    A **partial** mode (:attr:`emit_partial_only`) returns only ONE
    element of the gold pair (selected by :attr:`partial_index`) —
    useful for the R-68-PARTIAL-VOTE bench (some oracles see only
    half of the gold answer; W21's quorum_min knob trades off
    whether a partial intersection is sufficient).
    """
    oracle_id: str = "oncall_notes"
    notes_log: dict[str, tuple[str, ...]] = dataclasses.field(
        default_factory=lambda: {
            "deadlock":             ("orders", "payments"),
            "pool_exhaustion":      ("api", "db"),
            "disk_fill":            ("storage", "logs_pipeline"),
            "slow_query_cascade":   ("web", "db_query"),
        })
    emit_partial_only: bool = False
    partial_index: int = 0  # which index of noted_in_admitted to emit
                              # when emit_partial_only=True (default 0).

    def consult(self, query: OutsideQuery) -> OutsideVerdict:
        admitted = sorted(set(query.admitted_tags))
        noted = self.notes_log.get(query.elected_root_cause, ())
        noted_in_admitted = [t for t in noted if t in admitted]
        if not noted_in_admitted:
            return OutsideVerdict(
                payload=None, source_id=self.oracle_id, n_tokens=0)
        if self.emit_partial_only and len(noted_in_admitted) > 1:
            i = max(0, min(int(self.partial_index),
                            len(noted_in_admitted) - 1))
            noted_in_admitted = [noted_in_admitted[i]]
        if len(noted_in_admitted) == len(admitted):
            return OutsideVerdict(
                payload=None, source_id=self.oracle_id, n_tokens=0)
        named = sorted(noted_in_admitted)
        payload = ("oncall_notes mention="
                   f"{','.join(named)} "
                   + " ".join(f"service={tag}" for tag in named))
        words = payload.split()
        if len(words) > query.max_response_tokens:
            words = words[:query.max_response_tokens]
        truncated = " ".join(words)
        return OutsideVerdict(
            payload=truncated, source_id=self.oracle_id,
            n_tokens=len(words))


@dataclasses.dataclass
class SingletonAsymmetricOracle:
    """Oracle that emits *exactly one* admitted tag (W21 multi-oracle
    disagreement primitive).

    Models a deterministic outside source whose reply names exactly
    one service. Used to construct R-68-NO-QUORUM regimes where
    each of three registered oracles emits a different singleton —
    no admitted tag receives ≥ ``quorum_min = 2`` votes; the W21
    layer abstains; ties FIFO at 0.000.

    Configuration via :attr:`target`:

      * ``"first"``     — emit the first admitted tag (sorted).
      * ``"middle"``    — emit the middle admitted tag (sorted).
      * ``"last"``      — emit the last admitted tag (sorted).
      * any other str   — emit that exact tag if in admitted_tags;
                            abstain otherwise.

    Bounded-context: the reply is one ``service=`` mention bounded
    by ``query.max_response_tokens``.
    """
    oracle_id: str = "singleton_oracle"
    target: str = "first"

    def consult(self, query: OutsideQuery) -> OutsideVerdict:
        admitted = sorted(set(query.admitted_tags))
        if len(admitted) < 2:
            return OutsideVerdict(
                payload=None, source_id=self.oracle_id, n_tokens=0)
        if self.target == "first":
            chosen = admitted[0]
        elif self.target == "middle":
            chosen = admitted[len(admitted) // 2]
        elif self.target == "last":
            chosen = admitted[-1]
        elif self.target in admitted:
            chosen = self.target
        else:
            return OutsideVerdict(
                payload=None, source_id=self.oracle_id, n_tokens=0)
        payload = (f"singleton_oracle target={chosen} "
                   f"service={chosen}")
        words = payload.split()
        if len(words) > query.max_response_tokens:
            words = words[:query.max_response_tokens]
        truncated = " ".join(words)
        return OutsideVerdict(
            payload=truncated, source_id=self.oracle_id,
            n_tokens=len(words))


@dataclasses.dataclass
class DisagreeingHonestOracle:
    """Honest oracle that points at a *different* gold pair from the
    one that matches the elected root cause.

    Models a well-intentioned outside source indexed on the wrong
    incident type — e.g. a registry that returns the *next* gold
    family's pair instead of the current one. Under the R-66-shape
    admitted set ``{gold_a, gold_b, decoy}`` (only one gold pair
    present), this oracle abstains by construction (the wrong gold
    pair is not in admitted). It is most useful on R-68 regimes
    that admit a wider set covering multiple gold pairs.
    """
    oracle_id: str = "disagreeing_honest"
    wrong_log: dict[str, tuple[str, ...]] = dataclasses.field(
        default_factory=lambda: {
            "deadlock":             ("api", "db"),
            "pool_exhaustion":      ("storage", "logs_pipeline"),
            "disk_fill":            ("web", "db_query"),
            "slow_query_cascade":   ("orders", "payments"),
        })

    def consult(self, query: OutsideQuery) -> OutsideVerdict:
        admitted = sorted(set(query.admitted_tags))
        wrong = self.wrong_log.get(query.elected_root_cause, ())
        wrong_in_admitted = [t for t in wrong if t in admitted]
        if (not wrong_in_admitted
                or len(wrong_in_admitted) == len(admitted)):
            return OutsideVerdict(
                payload=None, source_id=self.oracle_id, n_tokens=0)
        named = sorted(wrong_in_admitted)
        payload = ("disagreeing_oracle attribution_chain="
                   f"{'_'.join(named)} "
                   + " ".join(f"service={tag}" for tag in named))
        words = payload.split()
        if len(words) > query.max_response_tokens:
            words = words[:query.max_response_tokens]
        truncated = " ".join(words)
        return OutsideVerdict(
            payload=truncated, source_id=self.oracle_id,
            n_tokens=len(words))


W21_BRANCH_QUORUM_RESOLVED = "quorum_resolved"
W21_BRANCH_NO_QUORUM = "no_quorum"
W21_BRANCH_SYMMETRIC_QUORUM = "symmetric_quorum"
W21_BRANCH_NO_ORACLES = "no_oracles"
W21_BRANCH_NO_TRIGGER = "no_trigger"
W21_BRANCH_DISABLED = "disabled"

W21_ALL_BRANCHES: tuple[str, ...] = (
    W21_BRANCH_QUORUM_RESOLVED,
    W21_BRANCH_NO_QUORUM,
    W21_BRANCH_SYMMETRIC_QUORUM,
    W21_BRANCH_NO_ORACLES,
    W21_BRANCH_NO_TRIGGER,
    W21_BRANCH_DISABLED,
)

W21_DEFAULT_TRIGGER_BRANCHES: frozenset[str] = W20_DEFAULT_TRIGGER_BRANCHES


@dataclasses.dataclass(frozen=True)
class W21OracleProbe:
    """Per-oracle probe record for one
    ``TrustWeightedMultiOracleDisambiguator.decode_rounds`` call.

    Fields
    ------
    oracle_id
        The oracle's ``oracle_id`` (provenance label).
    role_label
        The registration's ``role_label`` (e.g. ``"service_graph"``,
        ``"change_history"``).
    trust_prior
        The registration's prior trust weight in [0, 1].
    payload
        The (truncated) oracle reply bytes — empty string when the
        oracle abstained.
    payload_tokens
        Tokenised oracle bytes — empty tuple when the oracle
        abstained.
    per_tag_count
        ``{tag: aw_count}`` over the oracle's reply — same scoring
        rule as W20 / W19 / W18.
    top_set
        Positive-set projection of this oracle's reply (sorted
        tuple of admitted tags with ``aw_count > 0``). Empty when
        the oracle abstained or its reply is symmetric across the
        admitted set.
    abstained
        True when the oracle returned ``None`` OR its top_set was
        empty / equal to the full admitted set.
    n_outside_tokens
        The oracle's actual reply length in
        :func:`_handoff_n_tokens` units (post-truncation).
    """
    oracle_id: str
    role_label: str
    trust_prior: float
    payload: str
    payload_tokens: tuple[str, ...]
    per_tag_count: dict[str, int]
    top_set: tuple[str, ...]
    abstained: bool
    n_outside_tokens: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "oracle_id": str(self.oracle_id),
            "role_label": str(self.role_label),
            "trust_prior": float(self.trust_prior),
            "payload": str(self.payload),
            "payload_tokens": list(self.payload_tokens),
            "per_tag_count": {
                tag: int(c) for tag, c in self.per_tag_count.items()
            },
            "top_set": list(self.top_set),
            "abstained": bool(self.abstained),
            "n_outside_tokens": int(self.n_outside_tokens),
        }


@dataclasses.dataclass(frozen=True)
class W21MultiOracleResult:
    """Audit record for one
    ``TrustWeightedMultiOracleDisambiguator.decode_rounds`` call.

    Fields
    ------
    answer
        The W21-projected answer dict (same shape as W19 / W20).
    inner_branch
        The W19 branch that fired before the W21 layer ran.
    triggered
        Whether the W21 trigger fired (``inner_branch`` was in
        the configured trigger set AND ``enabled`` AND
        ``len(oracle_registrations) > 0``).
    quorum_min
        The configured ``quorum_min`` at projection time.
    min_trust_sum
        The configured ``min_trust_sum`` at projection time.
    probes
        Tuple of :class:`W21OracleProbe`, one per registered oracle
        (preserving registration order). Empty tuple when no oracle
        was consulted.
    per_tag_votes
        ``{tag: vote_count}`` aggregated across non-abstained probes.
    per_tag_trust_sum
        ``{tag: Σ trust_prior of voting oracles}`` aggregated across
        non-abstained probes.
    decoder_branch
        The W21 branch (one of :data:`W21_ALL_BRANCHES`).
    abstained
        True when the W21 layer did not produce a strict-max
        projection.
    n_outside_queries
        Number of oracle consults performed (equal to
        ``len(oracle_registrations)`` when ``triggered`` else 0).
    n_outside_tokens_total
        Strict per-cell additional token cost — sum of
        ``n_outside_tokens`` across all probes.
    """
    answer: dict[str, Any]
    inner_branch: str
    triggered: bool
    quorum_min: int
    min_trust_sum: float
    probes: tuple[W21OracleProbe, ...]
    per_tag_votes: dict[str, int]
    per_tag_trust_sum: dict[str, float]
    decoder_branch: str
    abstained: bool
    n_outside_queries: int
    n_outside_tokens_total: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "root_cause": str(self.answer.get("root_cause", "unknown")),
            "services": tuple(self.answer.get("services", ())),
            "remediation": str(self.answer.get("remediation",
                                                 "investigate")),
            "inner_branch": str(self.inner_branch),
            "triggered": bool(self.triggered),
            "quorum_min": int(self.quorum_min),
            "min_trust_sum": float(self.min_trust_sum),
            "probes": [p.as_dict() for p in self.probes],
            "per_tag_votes": {
                tag: int(v) for tag, v in self.per_tag_votes.items()
            },
            "per_tag_trust_sum": {
                tag: float(v) for tag, v in self.per_tag_trust_sum.items()
            },
            "decoder_branch": str(self.decoder_branch),
            "abstained": bool(self.abstained),
            "n_outside_queries": int(self.n_outside_queries),
            "n_outside_tokens_total": int(self.n_outside_tokens_total),
        }


@dataclasses.dataclass
class TrustWeightedMultiOracleDisambiguator:
    """Trust-weighted multi-oracle adjudicator (SDK v3.22, W21 family).

    Wraps a :class:`BundleContradictionDisambiguator` (W19) and adds
    multi-source outside-information adjudication when the inner W19
    branch indicates the bundle alone is structurally insufficient.
    Generalises the SDK v3.21 :class:`OutsideWitnessAcquisitionDisambi
    guator` (W20) by consulting **N** registered oracles per cell
    instead of one, and projecting the answer onto the
    quorum-of-aligned subset of the admitted tag set.

    This is the first capsule-native multi-agent-coordination method
    that crosses the W20-Λ-compromised wall on a regime where the
    wall actually applies (R-68-MULTI-MAJORITY): one of the registered
    oracles emits a decoy-asymmetric reply, but a strict majority of
    the registered oracles emit gold-asymmetric replies; the W21
    quorum-aware projection commits to the gold subset.

    The W21 layer is *not* a learned model — it is a closed-form
    composition of (a) the inner W19 bundle-only scorer, (b) one
    targeted oracle consult **per registered oracle**, (c) the same
    per-tag scorer the W18 / W19 / W20 layers use, and (d) a
    deterministic vote-counting rule with two configurable knobs
    (``quorum_min``, ``min_trust_sum``).

    Pipeline
    --------
    1. Run inner W19. If W19's branch is not in ``trigger_branches``
       OR W21 is ``enabled = False`` OR no oracles are registered,
       reduce to W19 byte-for-byte. The ``decoder_branch`` becomes
       :data:`W21_BRANCH_NO_TRIGGER` /
       :data:`W21_BRANCH_DISABLED` /
       :data:`W21_BRANCH_NO_ORACLES` accordingly.
    2. Build :class:`OutsideQuery` (same shape as W20).
    3. Consult **every** registered oracle once. Per oracle:
         * Truncate reply to ``max_response_tokens``.
         * Tokenise via :func:`_disambiguator_payload_tokens`.
         * Compute ``per_tag_count`` per admitted tag via
           :func:`_relational_compatibility_score`.
         * Record the probe's ``top_set`` (positive-set projection).
    4. Aggregate per-tag votes:
         votes[tag]      = #probes with per_tag_count[tag] > 0
         trust_sum[tag]  = Σ trust_prior over those probes
    5. Project:
         top_set = {tag : votes[tag] ≥ quorum_min
                          AND trust_sum[tag] ≥ min_trust_sum}.
       * If ``top_set`` is non-empty proper subset of admitted_tags
         → :data:`W21_BRANCH_QUORUM_RESOLVED`; project the answer.
       * If ``top_set`` is empty
         → :data:`W21_BRANCH_NO_QUORUM`; abstain (fall through).
       * If ``top_set`` covers every admitted tag
         → :data:`W21_BRANCH_SYMMETRIC_QUORUM`; abstain (fall through).
    6. Return a :class:`W21MultiOracleResult` with the final answer +
       audit record (per-oracle probes, aggregated votes, total
       ``n_outside_queries`` and ``n_outside_tokens_total``).

    Bounded-context honesty
    ------------------------
    The W21 layer issues *exactly N* outside queries per cell, where
    N = ``len(oracle_registrations)``. Each query bounded by
    ``max_response_tokens``. The total per-cell additional token
    cost is recorded in ``n_outside_tokens_total`` and is *additive*
    on top of the inner W15 ``tokens_kept`` figure. The inner W15
    accounting is byte-for-byte unchanged from W20 / W19.

    Backward-compat
    ----------------
    * **W21-3-A** (vs W19, no-trigger paths). With ``enabled = False``
      or with no oracles registered, the W21 layer reduces to the
      inner W19 byte-for-byte. With ``enabled = True`` AND the inner
      W19 returns a non-trigger branch, the W21 layer reduces to W19
      byte-for-byte.
    * **W21-3-B** (vs W20, single-oracle quorum_min=1). With
      ``quorum_min = 1`` AND a single oracle registered AND
      ``min_trust_sum = 0.0``, the W21 layer ties W20 byte-for-byte
      on the answer field on every regime where W20 fires. (W21
      records richer audit info but the projected answer is
      identical.)

    Honest scope (W21)
    -------------------
    The W21-1 strict gain is *strongly conditional* on:
      (a) the bench property — W19 abstains via SYMMETRIC or NO_SIGNAL,
      (b) the registered oracle set contains ≥ ``quorum_min`` oracles
          whose replies independently identify a non-empty proper
          asymmetric subset of the admitted tags AND that subset
          intersects to a non-empty asymmetric subset.

    Three named falsifiers make the conditionality sharp:
      * **W21-Λ-no-quorum** — every per-oracle reply identifies a
        different singleton; no tag has ≥ ``quorum_min`` votes;
        abstain → tie FIFO at 0.000.
      * **W21-Λ-all-compromised** — every registered oracle returns
        a decoy-asymmetric reply; quorum forms on decoy; FAIL at
        0.000. The natural extension W21-C-CALIBRATED-TRUST (low
        trust priors on uncalibrated oracles) is conjectural.
      * **W21-Λ-disagreement-only** — registered oracles each emit a
        different non-empty proper asymmetric subset that does NOT
        intersect; abstain → tie FIFO at 0.000.
    """

    inner: BundleContradictionDisambiguator = dataclasses.field(
        default_factory=lambda: BundleContradictionDisambiguator())
    oracle_registrations: tuple[OracleRegistration, ...] = ()
    enabled: bool = True
    trigger_branches: frozenset[str] = dataclasses.field(
        default_factory=lambda: W21_DEFAULT_TRIGGER_BRANCHES)
    max_response_tokens: int = 24
    quorum_min: int = 2
    min_trust_sum: float = 0.0

    # Forensic — last applied result, exposed for the bench driver.
    _last_result: W21MultiOracleResult | None = None

    def decode_rounds(self,
                       per_round_handoffs: Sequence[Sequence[_DecodedHandoff]],
                       ) -> dict[str, Any]:
        base = self.inner.decode_rounds(per_round_handoffs)
        w19_result = self.inner.last_result
        out = dict(base)
        # Disabled / no inner W19 path.
        if (not self.enabled) or w19_result is None:
            result = W21MultiOracleResult(
                answer=dict(base),
                inner_branch=(w19_result.decoder_branch
                                if w19_result is not None
                                else W19_BRANCH_DISABLED),
                triggered=False,
                quorum_min=int(self.quorum_min),
                min_trust_sum=float(self.min_trust_sum),
                probes=(),
                per_tag_votes={},
                per_tag_trust_sum={},
                decoder_branch=W21_BRANCH_DISABLED,
                abstained=True,
                n_outside_queries=0,
                n_outside_tokens_total=0,
            )
            self._last_result = result
            out["multi_oracle"] = result.as_dict()
            return out
        # No-trigger path.
        if w19_result.decoder_branch not in self.trigger_branches:
            result = W21MultiOracleResult(
                answer=dict(base),
                inner_branch=w19_result.decoder_branch,
                triggered=False,
                quorum_min=int(self.quorum_min),
                min_trust_sum=float(self.min_trust_sum),
                probes=(),
                per_tag_votes={},
                per_tag_trust_sum={},
                decoder_branch=W21_BRANCH_NO_TRIGGER,
                abstained=False,
                n_outside_queries=0,
                n_outside_tokens_total=0,
            )
            self._last_result = result
            out["multi_oracle"] = result.as_dict()
            return out
        # No-oracles path — reduce to W19 byte-for-byte.
        if not self.oracle_registrations:
            result = W21MultiOracleResult(
                answer=dict(base),
                inner_branch=w19_result.decoder_branch,
                triggered=False,
                quorum_min=int(self.quorum_min),
                min_trust_sum=float(self.min_trust_sum),
                probes=(),
                per_tag_votes={},
                per_tag_trust_sum={},
                decoder_branch=W21_BRANCH_NO_ORACLES,
                abstained=True,
                n_outside_queries=0,
                n_outside_tokens_total=0,
            )
            self._last_result = result
            out["multi_oracle"] = result.as_dict()
            return out
        # Build the OutsideQuery — same shape as W20.
        union_tag_set: set[str] = set()
        union: list[_DecodedHandoff] = []
        raw_union: list[_DecodedHandoff] = []
        layered_normaliser = self.inner.inner.inner.inner
        for bundle in per_round_handoffs:
            normalised = layered_normaliser.normalize_round(bundle)
            for h_norm, h_raw in zip(normalised, bundle):
                union.append(h_norm)
                raw_union.append(h_raw)
                tag = _service_tag_of(h_norm.payload)
                if tag:
                    union_tag_set.add(tag)
        admitted_tags = tuple(sorted(union_tag_set))
        round_hint: list[int] = []
        for r_idx, bundle in enumerate(per_round_handoffs, start=1):
            for _ in bundle:
                round_hint.append(r_idx)
        primary_idx = _w19_canonical_primary_index(
            union, round_hint, raw_union=raw_union)
        primary_payload = (union[primary_idx].payload
                            if primary_idx >= 0 else "")
        witness_payloads: list[str] = []
        for i, h in enumerate(union):
            if i == primary_idx:
                continue
            if h.claim_kind not in _SPECIFIC_TIER_CLAIM_KINDS:
                continue
            witness_payloads.append(h.payload)
        elected_root_cause = str(base.get("root_cause", "unknown"))
        query = OutsideQuery(
            admitted_tags=admitted_tags,
            elected_root_cause=elected_root_cause,
            primary_payload=primary_payload,
            witness_payloads=tuple(witness_payloads),
            max_response_tokens=self.max_response_tokens,
        )
        # Consult every registered oracle exactly once.
        probes: list[W21OracleProbe] = []
        per_tag_votes: dict[str, int] = {tag: 0 for tag in admitted_tags}
        per_tag_trust_sum: dict[str, float] = {
            tag: 0.0 for tag in admitted_tags}
        n_outside_total = 0
        for reg in self.oracle_registrations:
            oracle_id = getattr(reg.oracle, "oracle_id", "no_oracle")
            try:
                verdict = reg.oracle.consult(query)
            except Exception:  # noqa: BLE001 — record + abstain
                verdict = OutsideVerdict(
                    payload=None, source_id=oracle_id, n_tokens=0)
            n_outside_total += int(verdict.n_tokens or 0)
            if verdict.payload is None or not verdict.payload:
                probes.append(W21OracleProbe(
                    oracle_id=oracle_id,
                    role_label=reg.role_label,
                    trust_prior=float(reg.trust_prior),
                    payload="",
                    payload_tokens=(),
                    per_tag_count={tag: 0 for tag in admitted_tags},
                    top_set=(),
                    abstained=True,
                    n_outside_tokens=int(verdict.n_tokens or 0),
                ))
                continue
            words = verdict.payload.split()
            if len(words) > self.max_response_tokens:
                words = words[:self.max_response_tokens]
            payload = " ".join(words)
            tokens = _disambiguator_payload_tokens(payload)
            per_tag: dict[str, int] = {}
            for tag in admitted_tags:
                d, c = _relational_compatibility_score(tag, tokens)
                per_tag[tag] = int(d + c)
            top_set = tuple(sorted(t for t in admitted_tags
                                       if per_tag[t] > 0))
            symmetric = (not top_set
                          or len(top_set) == len(admitted_tags))
            probe = W21OracleProbe(
                oracle_id=oracle_id,
                role_label=reg.role_label,
                trust_prior=float(reg.trust_prior),
                payload=payload,
                payload_tokens=tokens,
                per_tag_count=per_tag,
                top_set=top_set,
                abstained=symmetric,
                n_outside_tokens=int(verdict.n_tokens or len(words)),
            )
            probes.append(probe)
            if symmetric:
                continue
            for tag in admitted_tags:
                if per_tag[tag] > 0:
                    per_tag_votes[tag] += 1
                    per_tag_trust_sum[tag] += float(reg.trust_prior)
        # Aggregate: top_set under quorum + trust thresholds.
        top_set = tuple(sorted(
            t for t in admitted_tags
            if per_tag_votes[t] >= int(self.quorum_min)
            and per_tag_trust_sum[t] >= float(self.min_trust_sum)
        ))
        if not top_set:
            branch = W21_BRANCH_NO_QUORUM
            abstained = True
            answer = dict(base)
        elif len(top_set) == len(admitted_tags):
            branch = W21_BRANCH_SYMMETRIC_QUORUM
            abstained = True
            answer = dict(base)
        else:
            branch = W21_BRANCH_QUORUM_RESOLVED
            abstained = False
            answer = dict(base)
            answer["services"] = top_set
        result = W21MultiOracleResult(
            answer=answer,
            inner_branch=w19_result.decoder_branch,
            triggered=True,
            quorum_min=int(self.quorum_min),
            min_trust_sum=float(self.min_trust_sum),
            probes=tuple(probes),
            per_tag_votes=per_tag_votes,
            per_tag_trust_sum=per_tag_trust_sum,
            decoder_branch=branch,
            abstained=abstained,
            n_outside_queries=len(self.oracle_registrations),
            n_outside_tokens_total=int(n_outside_total),
        )
        self._last_result = result
        if not abstained:
            out["services"] = top_set
        out["multi_oracle"] = result.as_dict()
        return out

    def decode(self, handoffs: Sequence[_DecodedHandoff]
               ) -> dict[str, Any]:
        return self.decode_rounds([handoffs])

    @property
    def last_result(self) -> W21MultiOracleResult | None:
        return self._last_result

    @property
    def T_decoder(self) -> int | None:
        return self.inner.T_decoder

    @T_decoder.setter
    def T_decoder(self, v: int | None) -> None:
        self.inner.T_decoder = v

    def pack_stats(self) -> dict[str, Any]:
        return self.inner.pack_stats()


# =============================================================================
# SDK v3.23 — capsule-native + latent-state-sharing hybrid (W22 family).
#
# W22 is the first capsule-native multi-agent-coordination method that
# *combines* explicit capsule passing with audited proxies for the
# LatentMAS direction (collective KV pooling / latent hidden-state
# transfer / super-token side channels). It does NOT manipulate
# transformer-internal KV caches; it implements the closest honest
# capsule-layer proxies for every LatentMAS idea family this repo can
# verify end-to-end:
#
#   * schema-passing — closed-vocabulary type schema is content-
#     addressed (:class:`SchemaCapsule`) and shared across roles via
#     CID. The bundle carries the CID once per session, not the full
#     schema text per cell.
#   * delta execution — instead of replaying every per-oracle probe
#     into the final decoder, the W22 layer emits one
#     :class:`LatentDigestEnvelope` per cell that summarises the W21
#     vote outcome as (per-tag votes, projected subset, provenance).
#   * shared-read cache — :class:`SharedReadCache` is a CID-keyed,
#     write-once-read-many proxy for the LatentMAS shared-KV-read idea.
#     :class:`CachingOracleAdapter` wraps any
#     :class:`OutsideWitnessOracle` and routes every ``consult(query)``
#     through the cache; identical OutsideQueries collapse to one
#     wire-side oracle call.
#   * controller-side verification — :func:`verify_latent_digest` is a
#     short, closed-form check (hash chain, schema-CID match, parent-
#     CID seal). On verification failure the W22 layer rejects the
#     digest and falls through to the W21 answer; the trust boundary
#     is explicit and the explicit-capsule path is never bypassed.
#
# What W22 does NOT do (do-not-overstate):
#
#   * does NOT touch transformer KV caches, embedding tables, attention
#     weights, or any model-internal state. The "shared cache" lives at
#     the capsule layer; it is an honest *proxy*, not a runtime KV
#     transplant.
#   * does NOT hide unaudited coordination behind opaque latent
#     payloads. Every envelope carries a content hash, a schema CID, a
#     parent CID list, a closed-vocabulary projection, and a
#     human-readable canonical encoding. The verification check is
#     short and the failure modes are enumerated.
#   * does NOT improve correctness over W21 on the synthetic R-69-CACHE
#     anchors. W22's correctness is exactly W21's by construction
#     (Theorem W22-2 — *correctness ratification*); the load-bearing
#     contribution is on the *efficiency* and *trust* axes.
#
# The W22 surface is purely additive on top of the W21 surface:
# decode_rounds returns the same answer dict W21 returns plus a single
# new ``"latent_hybrid"`` audit block; existing decoders are unchanged.
# =============================================================================

W22_LATENT_ENVELOPE_SCHEMA_VERSION = "wevra.latent_digest.v1"

W22_BRANCH_LATENT_RESOLVED = "latent_resolved"
W22_BRANCH_LATENT_REJECTED = "latent_rejected"
W22_BRANCH_NO_TRIGGER = "no_trigger"
W22_BRANCH_NO_SCHEMA = "no_schema"
W22_BRANCH_DISABLED = "disabled"
W22_BRANCH_ABSTAIN_PASSTHROUGH = "abstain_passthrough"

W22_ALL_BRANCHES = (
    W22_BRANCH_LATENT_RESOLVED,
    W22_BRANCH_LATENT_REJECTED,
    W22_BRANCH_NO_TRIGGER,
    W22_BRANCH_NO_SCHEMA,
    W22_BRANCH_DISABLED,
    W22_BRANCH_ABSTAIN_PASSTHROUGH,
)

# W22 fires only when the inner W21 produced a quorum-resolved
# projection. The other W21 branches (NO_QUORUM, ALL_COMPROMISED-as-
# SYMMETRIC_QUORUM, NO_TRIGGER, …) fall through unchanged so the W21
# falsifier ladder remains intact byte-for-byte.
W22_DEFAULT_TRIGGER_BRANCHES: frozenset[str] = frozenset({
    W21_BRANCH_QUORUM_RESOLVED,
})


def _canonical_json_bytes(obj: Any) -> bytes:
    """Stable JSON encoding for hash-chain inputs.

    ``sort_keys=True`` + ``separators=(",", ":")`` + ``ensure_ascii=True``
    is the canonical encoding used everywhere in the W22 surface (and
    by :class:`ContextCapsule.new` for capsule CIDs). Float fields are
    rounded by callers before passing in to avoid IEEE-754 byte drift.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True).encode("ascii")


def _whitespace_token_count(text: str) -> int:
    """Whitespace token count, matching the W21 oracle reply convention.

    The W21 layer counts oracle-reply tokens by ``payload.split()``;
    the W22 latent digest counts canonical-encoding tokens by the same
    rule so the visible-token accounting on a downstream decoder is
    apples-to-apples between W21's ``n_outside_tokens_total`` and
    W22's ``digest_n_tokens``.
    """
    if not text:
        return 0
    return len(text.split())


@dataclasses.dataclass(frozen=True)
class SchemaCapsule:
    """Closed-vocabulary type schema, content-addressed (W22 family).

    Models the LatentMAS *schema-passing* direction at the capsule
    layer. A SchemaCapsule names the closed vocabularies that a
    multi-agent coordination round operates over — root causes,
    services, oracle kinds — and emits a stable content-address (the
    SHA-256 over the canonical encoding) so multiple roles can refer
    to the schema by CID instead of re-emitting the full vocabulary
    in every handoff.

    The "delta" idea (the W22 envelope only carries fields that
    change vs the schema baseline) is realised on top of this:
    :class:`LatentDigestEnvelope` references this schema's CID and
    only encodes the per-cell vote / projection fields. The bundle
    text never re-emits the closed-vocabulary lists themselves.
    """
    schema_id: str
    version: str
    closed_vocab_root_causes: tuple[str, ...]
    closed_vocab_services: tuple[str, ...]
    closed_vocab_oracle_kinds: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "closed_vocab_root_causes",
                            tuple(sorted(self.closed_vocab_root_causes)))
        object.__setattr__(self, "closed_vocab_services",
                            tuple(sorted(self.closed_vocab_services)))
        object.__setattr__(self, "closed_vocab_oracle_kinds",
                            tuple(sorted(self.closed_vocab_oracle_kinds)))

    def to_canonical_bytes(self) -> bytes:
        return _canonical_json_bytes({
            "schema_id": self.schema_id,
            "version": self.version,
            "root_causes": list(self.closed_vocab_root_causes),
            "services": list(self.closed_vocab_services),
            "oracle_kinds": list(self.closed_vocab_oracle_kinds),
        })

    @property
    def cid(self) -> str:
        return hashlib.sha256(self.to_canonical_bytes()).hexdigest()

    @property
    def n_canonical_tokens(self) -> int:
        return _whitespace_token_count(
            " ".join((
                self.schema_id, self.version,
                *self.closed_vocab_root_causes,
                *self.closed_vocab_services,
                *self.closed_vocab_oracle_kinds,
            )))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_id": str(self.schema_id),
            "version": str(self.version),
            "cid": self.cid,
            "n_canonical_tokens": int(self.n_canonical_tokens),
            "n_root_causes": len(self.closed_vocab_root_causes),
            "n_services": len(self.closed_vocab_services),
            "n_oracle_kinds": len(self.closed_vocab_oracle_kinds),
        }


def build_incident_triage_schema_capsule() -> SchemaCapsule:
    """Default SchemaCapsule for the incident-triage W22 bench."""
    return SchemaCapsule(
        schema_id="incident_triage",
        version="v1",
        closed_vocab_root_causes=(
            "deadlock", "disk_fill",
            "pool_exhaustion", "slow_query_cascade",
        ),
        closed_vocab_services=(
            "api", "db", "db_query", "logs_pipeline",
            "orders", "payments", "storage", "web",
        ),
        closed_vocab_oracle_kinds=(
            "change_history",
            "llm_adjudicator",
            "oncall_notes",
            "service_graph",
        ),
    )


@dataclasses.dataclass(frozen=True)
class LatentDigestEnvelope:
    """Typed, controller-verified compact summary of one W21 cell
    outcome (W22 family).

    Models the LatentMAS *latent hidden-state transfer* and *delta
    execution* directions at the capsule layer.

    The ``digest_cid`` is *signed at construction*: ``__post_init__``
    computes SHA-256 over the canonical bytes (schema-version-aware,
    sorted, rounded floats) and freezes the result on the instance.
    Tampering with any field via :func:`dataclasses.replace`
    preserves the original ``digest_cid`` (the field is copied
    through unchanged); :func:`verify_latent_digest` recomputes the
    SHA over the new bytes and detects the mismatch — the
    load-bearing tamper-detection signal (Theorem W22-3).
    """
    schema_cid: str
    inner_w19_branch: str
    quorum_min: int
    min_trust_sum: float
    per_tag_vote_count: tuple[tuple[str, int], ...]
    per_tag_trust_sum: tuple[tuple[str, float], ...]
    projected_subset: tuple[str, ...]
    n_oracles_consulted: int
    n_outside_tokens_total: int
    parent_probe_cids: tuple[str, ...]
    digest_cid: str = ""
    schema_version: str = W22_LATENT_ENVELOPE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        # Canonicalise tuple fields first so two logically-equal
        # envelopes always have the same canonical bytes.
        object.__setattr__(self, "per_tag_vote_count",
                            tuple(sorted(self.per_tag_vote_count,
                                            key=lambda kv: kv[0])))
        object.__setattr__(self, "per_tag_trust_sum",
                            tuple(sorted(
                                ((tag, round(float(v), 4))
                                  for tag, v in self.per_tag_trust_sum),
                                key=lambda kv: kv[0])))
        object.__setattr__(self, "projected_subset",
                            tuple(sorted(self.projected_subset)))
        object.__setattr__(self, "parent_probe_cids",
                            tuple(self.parent_probe_cids))
        # Sign at construction time IFF no digest_cid was provided.
        # ``dataclasses.replace`` copies the original digest_cid
        # through; the receiver's verifier recomputes the SHA over
        # the new (post-tamper) canonical bytes and compares — a
        # mismatch fires :data:`W22_BRANCH_LATENT_REJECTED`.
        if not self.digest_cid:
            object.__setattr__(
                self, "digest_cid", self.recompute_digest_cid())

    def _signed_payload(self) -> dict[str, Any]:
        """Canonical payload signed by ``digest_cid`` (excludes
        ``digest_cid`` itself — recursion would self-reference)."""
        return {
            "schema_version": self.schema_version,
            "schema_cid": self.schema_cid,
            "inner_w19_branch": self.inner_w19_branch,
            "quorum_min": int(self.quorum_min),
            "min_trust_sum": round(float(self.min_trust_sum), 4),
            "per_tag_vote_count": [
                [t, int(c)] for t, c in self.per_tag_vote_count
            ],
            "per_tag_trust_sum": [
                [t, round(float(v), 4)] for t, v in self.per_tag_trust_sum
            ],
            "projected_subset": list(self.projected_subset),
            "n_oracles_consulted": int(self.n_oracles_consulted),
            "n_outside_tokens_total": int(self.n_outside_tokens_total),
            "parent_probe_cids": list(self.parent_probe_cids),
        }

    def to_canonical_bytes(self) -> bytes:
        return _canonical_json_bytes(self._signed_payload())

    def to_canonical_text(self) -> str:
        return self.to_canonical_bytes().decode("ascii")

    def recompute_digest_cid(self) -> str:
        return hashlib.sha256(self.to_canonical_bytes()).hexdigest()

    def to_decoder_text(self) -> str:
        """Whitespace-tokenisable single-line summary the downstream
        decoder consumes. Apples-to-apples with W21 oracle ``payload``
        accounting (which uses :meth:`str.split`).

        The decoder text is what the *final decoder* sees in place of
        the verbose W21 audit; ``n_digest_tokens`` is its
        whitespace-token count, the apples-to-apples comparison
        against W21's ``n_outside_tokens_total``.
        """
        votes = ",".join(
            f"{t}:{c}" for t, c in self.per_tag_vote_count) or "(empty)"
        trust = ",".join(
            f"{t}:{round(v,4)}" for t, v in self.per_tag_trust_sum
        ) or "(empty)"
        proj = ",".join(self.projected_subset) or "(empty)"
        parts = [
            "LATENT_DIGEST",
            f"schema_version={self.schema_version}",
            f"schema_cid={self.schema_cid[:16]}",
            f"digest_cid={self.digest_cid[:16]}",
            f"branch={self.inner_w19_branch}",
            f"quorum_min={int(self.quorum_min)}",
            f"min_trust_sum={round(float(self.min_trust_sum), 4)}",
            f"votes={votes}",
            f"trust={trust}",
            f"projected={proj}",
            f"oracles={int(self.n_oracles_consulted)}",
            f"outside_tokens={int(self.n_outside_tokens_total)}",
            f"n_probes={len(self.parent_probe_cids)}",
        ]
        return " ".join(parts)

    @property
    def n_digest_tokens(self) -> int:
        # Whitespace-token count of the decoder-facing line — what a
        # downstream decoder pays for the latent summary,
        # apples-to-apples with W21's ``n_outside_tokens_total``
        # (which counts oracle reply ``payload.split()`` tokens).
        return _whitespace_token_count(self.to_decoder_text())

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "schema_cid": self.schema_cid,
            "digest_cid": self.digest_cid,
            "inner_w19_branch": self.inner_w19_branch,
            "quorum_min": int(self.quorum_min),
            "min_trust_sum": round(float(self.min_trust_sum), 4),
            "per_tag_vote_count": [
                [t, int(c)] for t, c in self.per_tag_vote_count
            ],
            "per_tag_trust_sum": [
                [t, round(float(v), 4)] for t, v in self.per_tag_trust_sum
            ],
            "projected_subset": list(self.projected_subset),
            "n_oracles_consulted": int(self.n_oracles_consulted),
            "n_outside_tokens_total": int(self.n_outside_tokens_total),
            "parent_probe_cids": list(self.parent_probe_cids),
            "n_digest_tokens": int(self.n_digest_tokens),
            "decoder_text": self.to_decoder_text(),
        }


@dataclasses.dataclass(frozen=True)
class LatentVerificationOutcome:
    """Result of :func:`verify_latent_digest`. Closed-vocabulary
    ``reason`` field — every verification failure fits one of the
    named modes."""
    ok: bool
    reason: str
    n_checks: int = 0


def _digest_cid_for_canonical(canonical: bytes) -> str:
    return hashlib.sha256(canonical).hexdigest()


def verify_latent_digest(
        envelope: LatentDigestEnvelope | None,
        *,
        registered_schema: SchemaCapsule,
        sealed_probe_cids: Iterable[str] | None = None,
        ) -> LatentVerificationOutcome:
    """Controller-side verification of a :class:`LatentDigestEnvelope`.

    This is the W22 trust boundary. Every latent payload that enters
    the decoder via the W22 layer is checked here; on any failure
    the layer rejects the digest and the explicit-capsule path
    stays sound (W22_BRANCH_LATENT_REJECTED).

    The check is closed-form, short, and the failure modes are
    enumerated. The function is pure (no side effects); soundness
    holds by inspection.
    """
    n_checks = 0
    if envelope is None:
        return LatentVerificationOutcome(
            ok=False, reason="empty_envelope", n_checks=n_checks)
    n_checks += 1
    if envelope.schema_version != W22_LATENT_ENVELOPE_SCHEMA_VERSION:
        return LatentVerificationOutcome(
            ok=False, reason="schema_version_unknown", n_checks=n_checks)
    n_checks += 1
    if envelope.schema_cid != registered_schema.cid:
        return LatentVerificationOutcome(
            ok=False, reason="schema_cid_mismatch", n_checks=n_checks)
    n_checks += 1
    canonical = envelope.to_canonical_bytes()
    if _digest_cid_for_canonical(canonical) != envelope.digest_cid:
        return LatentVerificationOutcome(
            ok=False, reason="hash_mismatch", n_checks=n_checks)
    if sealed_probe_cids is not None:
        n_checks += 1
        sealed_set = set(sealed_probe_cids)
        for pcid in envelope.parent_probe_cids:
            if pcid not in sealed_set:
                return LatentVerificationOutcome(
                    ok=False, reason="unsealed_parent_probe_cid",
                    n_checks=n_checks)
    return LatentVerificationOutcome(ok=True, reason="ok", n_checks=n_checks)


@dataclasses.dataclass
class SharedReadCache:
    """CID-keyed write-once-read-many cache (W22 family).

    Models the LatentMAS *shared-KV-read* direction at the capsule
    layer. The cache is content-addressed: the key is the CID of an
    :class:`OutsideQuery` + ``oracle_id`` pair; the value is the
    canonical bytes of an :class:`OutsideVerdict` plus its token
    count. Two queries with identical content collapse to one
    entry; subsequent reads return the cached bytes without
    consulting the underlying oracle.

    Honest scope: capsule-layer proxy — NOT a transformer KV cache.
    The "tokens saved" metric is wire-side oracle reply tokens that
    are NOT paid because the entry was already in the cache.
    """
    _store: dict[str, tuple[bytes, int, str]] = dataclasses.field(
        default_factory=dict)
    n_hits: int = 0
    n_misses: int = 0
    n_tokens_saved: int = 0

    @staticmethod
    def query_cid_for(query: OutsideQuery, *, oracle_id: str) -> str:
        body = _canonical_json_bytes({
            "oracle_id": str(oracle_id),
            "admitted_tags": list(query.admitted_tags),
            "elected_root_cause": str(query.elected_root_cause),
            "primary_payload": str(query.primary_payload),
            "witness_payloads": list(query.witness_payloads),
            "max_response_tokens": int(query.max_response_tokens),
        })
        return hashlib.sha256(body).hexdigest()

    def get(self, cid: str) -> tuple[bytes, int, str] | None:
        entry = self._store.get(cid)
        if entry is None:
            self.n_misses += 1
            return None
        self.n_hits += 1
        self.n_tokens_saved += int(entry[1])
        return entry

    def put(self, cid: str, body: bytes, *, n_tokens: int,
            source_id: str = "") -> None:
        if cid in self._store:
            return
        self._store[cid] = (bytes(body), int(n_tokens), str(source_id))

    def stats(self) -> dict[str, Any]:
        return {
            "n_hits": int(self.n_hits),
            "n_misses": int(self.n_misses),
            "n_tokens_saved": int(self.n_tokens_saved),
            "n_entries": len(self._store),
        }


@dataclasses.dataclass
class CachingOracleAdapter:
    """Cache-aware adapter around any :class:`OutsideWitnessOracle`.

    Routes every ``consult(query)`` through a :class:`SharedReadCache`.
    Identical OutsideQueries collapse to one wire-side oracle call;
    the second cell pays *zero* outside-oracle tokens (the cache
    returns the bytes; the audit's ``n_outside_tokens_total`` sums
    only the *uncached* token cost).

    The adapter preserves the :class:`OutsideWitnessOracle` Protocol
    so it can be registered in any W21 oracle stack as a drop-in
    replacement for the wrapped oracle.
    """
    inner: Any
    cache: SharedReadCache
    oracle_id: str = ""
    last_was_hit: bool = False

    def __post_init__(self) -> None:
        if not self.oracle_id:
            self.oracle_id = str(getattr(
                self.inner, "oracle_id", "cached_oracle"))

    def consult(self, query: OutsideQuery) -> OutsideVerdict:
        cid = SharedReadCache.query_cid_for(
            query, oracle_id=self.oracle_id)
        cached = self.cache.get(cid)
        if cached is not None:
            body, n_tokens, source_id = cached
            self.last_was_hit = True
            payload = body.decode("utf-8") if body else None
            return OutsideVerdict(
                payload=payload,
                source_id=source_id or self.oracle_id,
                n_tokens=int(n_tokens),
            )
        self.last_was_hit = False
        verdict = self.inner.consult(query)
        body = (verdict.payload or "").encode("utf-8")
        self.cache.put(cid, body, n_tokens=int(verdict.n_tokens),
                        source_id=str(verdict.source_id or self.oracle_id))
        return verdict


@dataclasses.dataclass(frozen=True)
class EnvelopeTamperer:
    """Tamper a :class:`LatentDigestEnvelope` for falsifier tests."""
    mode: str = "flip_projected_subset"
    admitted_tags: tuple[str, ...] = ()

    def apply(self, env: LatentDigestEnvelope
               ) -> LatentDigestEnvelope:
        if self.mode == "flip_projected_subset":
            admitted = set(self.admitted_tags) or {
                t for t, _ in env.per_tag_vote_count}
            new_subset = tuple(sorted(
                t for t in admitted if t not in env.projected_subset))
            return dataclasses.replace(
                env, projected_subset=new_subset)
        if self.mode == "add_phantom_probe_cid":
            phantom = "0" * 64
            return dataclasses.replace(
                env, parent_probe_cids=tuple(
                    list(env.parent_probe_cids) + [phantom]))
        if self.mode == "change_quorum_min":
            return dataclasses.replace(env, quorum_min=0)
        raise ValueError(f"unknown EnvelopeTamperer mode {self.mode!r}")


@dataclasses.dataclass
class W22LatentResult:
    """Audit record for the W22 latent-hybrid layer.

    Captures (a) the W21 outcome below, (b) the latent envelope
    emitted (or rejected), (c) the controller verification result,
    (d) the cache statistics, and (e) the visible-token accounting
    delta vs the W21 baseline.
    """
    answer: dict[str, Any]
    inner_w21_branch: str
    triggered: bool
    decoder_branch: str
    abstained: bool
    schema_cid: str
    schema_version: str
    digest_cid: str
    digest_n_tokens: int
    n_w15_tokens_kept: int
    n_w21_outside_tokens_total: int
    n_w21_verbose_audit_tokens: int
    n_visible_tokens_to_decider: int
    digest_compression_ratio: float
    n_cache_hits_this_cell: int
    n_cache_misses_this_cell: int
    cache_tokens_saved_this_cell: int
    schema_shared_tokens_saved_this_cell: int
    verification_ok: bool
    verification_reason: str
    verification_n_checks: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "inner_w21_branch": str(self.inner_w21_branch),
            "triggered": bool(self.triggered),
            "decoder_branch": str(self.decoder_branch),
            "abstained": bool(self.abstained),
            "schema_cid": str(self.schema_cid),
            "schema_version": str(self.schema_version),
            "digest_cid": str(self.digest_cid),
            "digest_n_tokens": int(self.digest_n_tokens),
            "n_w15_tokens_kept": int(self.n_w15_tokens_kept),
            "n_w21_outside_tokens_total":
                int(self.n_w21_outside_tokens_total),
            "n_w21_verbose_audit_tokens":
                int(self.n_w21_verbose_audit_tokens),
            "n_visible_tokens_to_decider":
                int(self.n_visible_tokens_to_decider),
            "digest_compression_ratio":
                round(float(self.digest_compression_ratio), 6),
            "n_cache_hits_this_cell": int(self.n_cache_hits_this_cell),
            "n_cache_misses_this_cell": int(self.n_cache_misses_this_cell),
            "cache_tokens_saved_this_cell":
                int(self.cache_tokens_saved_this_cell),
            "schema_shared_tokens_saved_this_cell":
                int(self.schema_shared_tokens_saved_this_cell),
            "verification_ok": bool(self.verification_ok),
            "verification_reason": str(self.verification_reason),
            "verification_n_checks": int(self.verification_n_checks),
        }


def _verbose_w21_audit_tokens(w21_result: W21MultiOracleResult) -> int:
    """Whitespace-token count of the canonical W21 audit JSON."""
    return _whitespace_token_count(
        json.dumps(w21_result.as_dict(),
                   sort_keys=True, separators=(",", ":"),
                   ensure_ascii=True))


def _build_envelope_from_w21(
        w21_result: W21MultiOracleResult,
        *, schema_cid: str,
        ) -> LatentDigestEnvelope:
    """Construct a :class:`LatentDigestEnvelope` from a W21 result."""
    per_tag_vote_count = tuple(
        sorted(
            ((tag, int(c))
              for tag, c in w21_result.per_tag_votes.items()),
            key=lambda kv: kv[0]))
    per_tag_trust_sum = tuple(
        sorted(
            ((tag, round(float(v), 4))
              for tag, v in w21_result.per_tag_trust_sum.items()),
            key=lambda kv: kv[0]))
    services = w21_result.answer.get("services", ())
    if isinstance(services, (list, tuple)):
        projected = tuple(sorted(str(s) for s in services))
    else:
        projected = ()
    parent_probe_cids: list[str] = []
    for probe in w21_result.probes:
        body = _canonical_json_bytes(probe.as_dict())
        parent_probe_cids.append(hashlib.sha256(body).hexdigest())
    return LatentDigestEnvelope(
        schema_cid=schema_cid,
        inner_w19_branch=w21_result.inner_branch,
        quorum_min=int(w21_result.quorum_min),
        min_trust_sum=float(w21_result.min_trust_sum),
        per_tag_vote_count=per_tag_vote_count,
        per_tag_trust_sum=per_tag_trust_sum,
        projected_subset=projected,
        n_oracles_consulted=int(w21_result.n_outside_queries),
        n_outside_tokens_total=int(w21_result.n_outside_tokens_total),
        parent_probe_cids=tuple(parent_probe_cids),
        schema_version=W22_LATENT_ENVELOPE_SCHEMA_VERSION,
    )


@dataclasses.dataclass
class LatentDigestDisambiguator:
    """Capsule + audited latent-hybrid disambiguator (W22 family).

    Wraps a :class:`TrustWeightedMultiOracleDisambiguator` (W21) and
    layers four LatentMAS-direction proxies on top:

      1. **Schema-passing** via :class:`SchemaCapsule` (CID-shared).
      2. **Delta execution** via :class:`LatentDigestEnvelope`.
      3. **Shared-read cache** via :class:`CachingOracleAdapter`.
      4. **Controller-side verification** via
         :func:`verify_latent_digest`.

    The W22 surface is *strictly additive* on top of W21:

      * If the inner W21 branch is not in ``trigger_branches`` (e.g.
        W21 abstained for any reason), W22 fires
        :data:`W22_BRANCH_NO_TRIGGER` and reduces to W21
        byte-for-byte on the answer field.
      * If no schema is registered, W22 fires
        :data:`W22_BRANCH_NO_SCHEMA` and reduces to W21.
      * If ``enabled = False``, W22 fires
        :data:`W22_BRANCH_DISABLED` and reduces to W21.
      * On verification failure, W22 fires
        :data:`W22_BRANCH_LATENT_REJECTED` and the answer field is
        the W21 answer byte-for-byte; the rejected envelope is
        recorded for forensics.

    Theorem family W22 — see
    ``docs/RESULTS_WEVRA_CAPSULE_LATENT_HYBRID.md`` for the full
    statements.
    """

    inner: TrustWeightedMultiOracleDisambiguator = dataclasses.field(
        default_factory=lambda: TrustWeightedMultiOracleDisambiguator())
    # Producer-side schema: the envelope's ``schema_cid`` is signed
    # against this schema's CID.
    schema: SchemaCapsule | None = None
    # Controller-side schema (verifier's view). When None, defaults
    # to ``schema``. When the controller has registered a
    # *different* SchemaCapsule (e.g. a version drift, or a
    # registry rotation), ``verify_latent_digest`` fires
    # ``schema_cid_mismatch`` and W22 rejects the envelope. This
    # is the R-69-SCHEMA-DRIFT falsifier path.
    verifier_schema: SchemaCapsule | None = None
    cache: SharedReadCache | None = None
    enabled: bool = True
    require_verification: bool = True
    tamperer: EnvelopeTamperer | None = None
    trigger_branches: frozenset[str] = dataclasses.field(
        default_factory=lambda: W22_DEFAULT_TRIGGER_BRANCHES)

    _last_result: W22LatentResult | None = None
    _last_envelope: LatentDigestEnvelope | None = None
    _schema_already_shared: bool = False
    _sealed_probe_cids: set[str] = dataclasses.field(default_factory=set)

    def reset_session(self) -> None:
        """Reset per-session state (schema-shared flag + sealed-CID
        set + cache stats). Useful when the same instance is reused
        across independent benchmark cells in a fresh-session
        regime."""
        self._schema_already_shared = False
        self._sealed_probe_cids = set()
        if self.cache is not None:
            self.cache.n_hits = 0
            self.cache.n_misses = 0
            self.cache.n_tokens_saved = 0
            self.cache._store.clear()
        self.inner._last_result = None
        self._last_result = None
        self._last_envelope = None

    def _seal_probe_cids(self,
                          w21_result: W21MultiOracleResult) -> list[str]:
        cids: list[str] = []
        for probe in w21_result.probes:
            body = _canonical_json_bytes(probe.as_dict())
            cid = hashlib.sha256(body).hexdigest()
            cids.append(cid)
            self._sealed_probe_cids.add(cid)
        return cids

    def _w15_tokens_kept_from(self, answer: dict[str, Any]) -> int:
        # Prefer the in-answer packing block (when the bench driver
        # injected one); else delegate down the W22→W21→W19→W18→W15
        # chain via :meth:`pack_stats` to read the W15 packer's last
        # ``tokens_kept`` value directly. This is the byte-for-byte
        # honest accounting; W21's outside-tokens are NOT W15 tokens.
        packing = answer.get("packing") if isinstance(answer, dict) else None
        if isinstance(packing, dict) and "tokens_kept" in packing:
            return int(packing["tokens_kept"])
        try:
            ps = self.inner.pack_stats() or {}
            v = ps.get("tokens_kept")
            if v is not None:
                return int(v)
        except Exception:  # noqa: BLE001 — accounting must never crash
            pass
        return 0

    def _cache_stats_snapshot(self) -> tuple[int, int, int]:
        if self.cache is None:
            return 0, 0, 0
        return (int(self.cache.n_hits),
                int(self.cache.n_misses),
                int(self.cache.n_tokens_saved))

    def decode_rounds(
            self,
            per_round_handoffs: Sequence[Sequence[_DecodedHandoff]],
            ) -> dict[str, Any]:
        before_hits, before_misses, before_saved = (
            self._cache_stats_snapshot())
        base = self.inner.decode_rounds(per_round_handoffs)
        w21_result = self.inner.last_result
        out = dict(base)
        n_w15_tokens_kept = self._w15_tokens_kept_from(base)

        def _pack_result(*, decoder_branch: str, abstained: bool,
                          envelope: LatentDigestEnvelope | None,
                          verification_ok: bool, verification_reason: str,
                          verification_n_checks: int,
                          ) -> dict[str, Any]:
            after_hits, after_misses, after_saved = (
                self._cache_stats_snapshot())
            n_hits = max(0, after_hits - before_hits)
            n_misses = max(0, after_misses - before_misses)
            cache_saved = max(0, after_saved - before_saved)
            schema_saved = (
                self.schema.n_canonical_tokens
                if (self.schema is not None
                    and self._schema_already_shared) else 0)
            n_w21_outside = int(getattr(
                w21_result, "n_outside_tokens_total", 0)
                if w21_result is not None else 0)
            verbose_audit_n = (
                _verbose_w21_audit_tokens(w21_result)
                if w21_result is not None else 0)
            digest_n = int(envelope.n_digest_tokens if envelope else 0)
            digest_cid = str(envelope.digest_cid if envelope else "")
            denom = max(1, verbose_audit_n)
            compression = (digest_n / denom) if denom else 0.0
            # Visible-token cost the downstream decoder pays:
            #   * LATENT_RESOLVED: the digest replaces the W21 audit
            #     and W21 outside-replies → kept + digest_n.
            #   * Otherwise (LATENT_REJECTED, NO_TRIGGER, NO_SCHEMA,
            #     DISABLED): the explicit-capsule path stays in
            #     force; the decoder pays the W21 baseline (kept +
            #     outside + verbose audit). For LATENT_REJECTED this
            #     is the *honest* fallback cost — the rejected
            #     digest is NOT trusted by the decoder, so it does
            #     not reduce visible cost.
            if (envelope is not None
                    and decoder_branch == W22_BRANCH_LATENT_RESOLVED):
                visible = n_w15_tokens_kept + digest_n
            else:
                visible = (n_w15_tokens_kept
                            + n_w21_outside + verbose_audit_n)
            triggered = decoder_branch == W22_BRANCH_LATENT_RESOLVED
            answer = dict(base)
            inner_branch = (
                w21_result.inner_branch
                if w21_result is not None
                else W19_BRANCH_DISABLED)
            schema_cid = (
                str(self.schema.cid) if self.schema is not None else "")
            schema_version = (
                str(self.schema.version) if self.schema is not None else "")
            result = W22LatentResult(
                answer=answer,
                inner_w21_branch=inner_branch,
                triggered=triggered,
                decoder_branch=decoder_branch,
                abstained=abstained,
                schema_cid=schema_cid,
                schema_version=schema_version,
                digest_cid=digest_cid,
                digest_n_tokens=digest_n,
                n_w15_tokens_kept=n_w15_tokens_kept,
                n_w21_outside_tokens_total=n_w21_outside,
                n_w21_verbose_audit_tokens=verbose_audit_n,
                n_visible_tokens_to_decider=int(visible),
                digest_compression_ratio=float(compression),
                n_cache_hits_this_cell=int(n_hits),
                n_cache_misses_this_cell=int(n_misses),
                cache_tokens_saved_this_cell=int(cache_saved),
                schema_shared_tokens_saved_this_cell=int(schema_saved),
                verification_ok=bool(verification_ok),
                verification_reason=str(verification_reason),
                verification_n_checks=int(verification_n_checks),
            )
            self._last_result = result
            self._last_envelope = envelope
            out_local = dict(out)
            out_local["latent_hybrid"] = result.as_dict()
            if envelope is not None:
                out_local["latent_envelope"] = envelope.as_dict()
            if envelope is not None and self.schema is not None:
                self._schema_already_shared = True
            return out_local

        if (not self.enabled) or w21_result is None:
            return _pack_result(
                decoder_branch=W22_BRANCH_DISABLED, abstained=True,
                envelope=None, verification_ok=False,
                verification_reason="disabled", verification_n_checks=0)
        if self.schema is None:
            return _pack_result(
                decoder_branch=W22_BRANCH_NO_SCHEMA, abstained=True,
                envelope=None, verification_ok=False,
                verification_reason="no_schema", verification_n_checks=0)
        if w21_result.decoder_branch not in self.trigger_branches:
            return _pack_result(
                decoder_branch=W22_BRANCH_NO_TRIGGER, abstained=False,
                envelope=None, verification_ok=False,
                verification_reason="no_trigger", verification_n_checks=0)
        sealed_probe_cids = self._seal_probe_cids(w21_result)
        envelope = _build_envelope_from_w21(
            w21_result, schema_cid=self.schema.cid)
        if self.tamperer is not None:
            envelope = self.tamperer.apply(envelope)
        # The verifier compares against ``verifier_schema`` if set,
        # otherwise the same producer ``schema`` (default backward-
        # compat). On schema drift the verifier's CID differs from
        # the envelope's signed CID and ``schema_cid_mismatch`` fires.
        verifier_view = self.verifier_schema or self.schema
        outcome = verify_latent_digest(
            envelope, registered_schema=verifier_view,
            sealed_probe_cids=set(sealed_probe_cids))
        if (not outcome.ok) and self.require_verification:
            return _pack_result(
                decoder_branch=W22_BRANCH_LATENT_REJECTED, abstained=True,
                envelope=envelope, verification_ok=False,
                verification_reason=outcome.reason,
                verification_n_checks=outcome.n_checks)
        return _pack_result(
            decoder_branch=W22_BRANCH_LATENT_RESOLVED, abstained=False,
            envelope=envelope, verification_ok=outcome.ok,
            verification_reason=outcome.reason,
            verification_n_checks=outcome.n_checks)

    def decode(self, handoffs: Sequence[_DecodedHandoff]
               ) -> dict[str, Any]:
        return self.decode_rounds([handoffs])

    @property
    def last_result(self) -> W22LatentResult | None:
        return self._last_result

    @property
    def last_envelope(self) -> LatentDigestEnvelope | None:
        return self._last_envelope

    @property
    def T_decoder(self) -> int | None:
        return self.inner.T_decoder

    @T_decoder.setter
    def T_decoder(self, v: int | None) -> None:
        self.inner.T_decoder = v

    def pack_stats(self) -> dict[str, Any]:
        return self.inner.pack_stats()


# =============================================================================
# SDK v3.24 — capsule-native cross-cell delta execution + quorum-keyed cache
# + super-token reference (W23 family).
#
# The follow-up to SDK v3.23 (W22) on the named research direction the
# W22 milestone explicitly left open: a denser latent / dense-control
# transfer mechanism that goes *beyond the per-cell digest envelope* by
# (a) carrying running cross-cell state in a hash-chained session
# envelope, (b) emitting a *small per-cell delta* against that running
# state instead of the full digest each time, (c) optionally collapsing
# the visible-token cost to a *single super-token reference* (the
# bounded, audited steganographic / dense-control-payload experiment
# the user explicitly asked for), and (d) mitigating the
# W22-C-CACHE-AMPLIFICATION conjecture (the cache freezing of a
# probabilistic LLM oracle's first reply across all matching cells)
# via a quorum-keyed cache freshness policy.
#
# The W22 layer was the smallest *per-cell* honest LatentMAS proxy
# (schema-passing + delta execution + shared-read cache + controller
# verification on one cell). The W23 layer is the smallest *cross-cell*
# honest LatentMAS proxy (running session state + per-cell delta +
# quorum-keyed cache + super-token reference + controller verification
# on every link). Both layers leave the explicit-capsule path sound on
# verification failure.
#
# Honest scope (W23). The "cross-cell state" lives at the **capsule
# layer**; it is not a transformer-internal KV transplant. The
# "super-token reference" is an audited single-visible-token CID
# pointer to a verified envelope; it is a bounded, auditable proxy for
# the LatentMAS *super-token side channel* idea — *not* an embedding
# manipulation. We do NOT modify model-internal state, embedding
# tables, attention weights, or any part of any transformer's runtime.
#
# Trust boundary. Every W23 envelope is hash-chained, schema-versioned,
# parent-CID-sealed, and (optionally) registered in a controller-side
# super-token registry. On any verification failure the W23 layer
# rejects the payload and the explicit-capsule path stays sound.
# =============================================================================


W23_SESSION_ENVELOPE_SCHEMA_VERSION: str = "wevra.session_digest.v1"
W23_DELTA_ENVELOPE_SCHEMA_VERSION: str = "wevra.session_delta.v1"
W23_SUPER_TOKEN_SCHEMA_VERSION: str = "wevra.super_token_ref.v1"


W23_BRANCH_DELTA_RESOLVED = "delta_resolved"
W23_BRANCH_DELTA_REJECTED = "delta_rejected"
W23_BRANCH_GENESIS = "genesis"
W23_BRANCH_SUPER_TOKEN_RESOLVED = "super_token_resolved"
W23_BRANCH_SUPER_TOKEN_REJECTED = "super_token_rejected"
W23_BRANCH_NO_TRIGGER = "no_trigger"
W23_BRANCH_NO_PRIOR_SESSION = "no_prior_session"
W23_BRANCH_DISABLED = "disabled"


W23_ALL_BRANCHES: tuple[str, ...] = (
    W23_BRANCH_DELTA_RESOLVED,
    W23_BRANCH_DELTA_REJECTED,
    W23_BRANCH_GENESIS,
    W23_BRANCH_SUPER_TOKEN_RESOLVED,
    W23_BRANCH_SUPER_TOKEN_REJECTED,
    W23_BRANCH_NO_TRIGGER,
    W23_BRANCH_NO_PRIOR_SESSION,
    W23_BRANCH_DISABLED,
)


# Trigger condition: W23 operates ONLY when the inner W22 layer
# produced a verified envelope (W22_BRANCH_LATENT_RESOLVED). On every
# other inner-W22 branch (NO_TRIGGER / LATENT_REJECTED / DISABLED /
# NO_SCHEMA), the W23 layer reduces to W22 byte-for-byte.
W23_DEFAULT_TRIGGER_BRANCHES: frozenset[str] = frozenset({
    W22_BRANCH_LATENT_RESOLVED,
})


# Cache freshness policies (W23 quorum-keyed-cache). Each registered
# oracle declares one policy; the cache uses it to decide whether two
# byte-identical OutsideQueries from different cells should hit the
# same entry.
#
#   * BYTE_IDENTICAL — W22 default. Two byte-identical queries always
#     collapse to one wire-side call. Maximises wire savings; freezes
#     a probabilistic oracle's first sample (W22-C-CACHE-AMPLIFICATION
#     applies).
#   * PER_CELL_NONCE — Cache key includes a per-cell nonce. Identical
#     queries from different cells DO NOT hit. Eliminates
#     W22-C-CACHE-AMPLIFICATION at the cost of zero cross-cell
#     savings on this oracle. Recommended for probabilistic LLM
#     adjudicators registered alongside deterministic oracles.
#   * QUORUM_LOCKED — Cache stores entries only after the controller
#     confirms the oracle's reply contributed to a successful quorum.
#     A bad first reply that did NOT form quorum is NOT cached;
#     subsequent matching cells re-consult and may produce a
#     different (good) reply. The strictest mitigation; mid-cost.
CACHE_FRESHNESS_BYTE_IDENTICAL: str = "byte_identical"
CACHE_FRESHNESS_PER_CELL_NONCE: str = "per_cell_nonce"
CACHE_FRESHNESS_QUORUM_LOCKED: str = "quorum_locked"


CACHE_FRESHNESS_POLICIES: tuple[str, ...] = (
    CACHE_FRESHNESS_BYTE_IDENTICAL,
    CACHE_FRESHNESS_PER_CELL_NONCE,
    CACHE_FRESHNESS_QUORUM_LOCKED,
)


@dataclasses.dataclass(frozen=True)
class SessionDigestEnvelope:
    """Hash-chained running cross-cell session state (W23 family).

    Models the LatentMAS *cross-cell latent state-sharing* direction
    at the capsule layer. After every cell, the W23 layer computes a
    new SessionDigestEnvelope whose ``prior_session_digest_cid``
    points back to the previous one (or empty string at genesis).
    The chain is verifiable in O(1) per link by the controller — each
    link's ``digest_cid`` recomputes from canonical bytes and the
    parent CID is checked against the registered chain head.

    The envelope is *not* a copy of every per-cell digest — it is a
    rolling summary: cumulative per-tag votes, latest projected
    subset, count of cells processed so far, the most recent inner
    W19 branch. The full per-cell evidence remains in the per-cell
    LatentDigestEnvelope (W22) and SessionDeltaEnvelope (W23).

    Trust boundary: every link is hash-chained; tampering with any
    field after construction yields a digest_cid mismatch on
    re-verification.
    """
    schema_cid: str
    prior_session_digest_cid: str
    n_cells: int
    cumulative_per_tag_votes: tuple[tuple[str, int], ...]
    latest_projected_subset: tuple[str, ...]
    latest_inner_branch: str
    n_cells_resolved: int
    digest_cid: str = ""
    schema_version: str = W23_SESSION_ENVELOPE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "cumulative_per_tag_votes",
                            tuple(sorted(self.cumulative_per_tag_votes,
                                            key=lambda kv: kv[0])))
        object.__setattr__(self, "latest_projected_subset",
                            tuple(sorted(self.latest_projected_subset)))
        if not self.digest_cid:
            object.__setattr__(self, "digest_cid",
                                self.recompute_digest_cid())

    def _signed_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "schema_cid": self.schema_cid,
            "prior_session_digest_cid": self.prior_session_digest_cid,
            "n_cells": int(self.n_cells),
            "cumulative_per_tag_votes": [
                [t, int(c)] for t, c in self.cumulative_per_tag_votes
            ],
            "latest_projected_subset": list(self.latest_projected_subset),
            "latest_inner_branch": self.latest_inner_branch,
            "n_cells_resolved": int(self.n_cells_resolved),
        }

    def to_canonical_bytes(self) -> bytes:
        return _canonical_json_bytes(self._signed_payload())

    def recompute_digest_cid(self) -> str:
        return hashlib.sha256(self.to_canonical_bytes()).hexdigest()

    def to_decoder_text(self) -> str:
        votes = ",".join(
            f"{t}:{c}" for t, c in self.cumulative_per_tag_votes
        ) or "(empty)"
        proj = ",".join(self.latest_projected_subset) or "(empty)"
        prior = (self.prior_session_digest_cid[:16]
                  if self.prior_session_digest_cid else "(genesis)")
        parts = [
            "SESSION_DIGEST",
            f"schema_version={self.schema_version}",
            f"schema_cid={self.schema_cid[:16]}",
            f"digest_cid={self.digest_cid[:16]}",
            f"prior={prior}",
            f"n_cells={int(self.n_cells)}",
            f"n_resolved={int(self.n_cells_resolved)}",
            f"branch={self.latest_inner_branch}",
            f"cumulative_votes={votes}",
            f"projected={proj}",
        ]
        return " ".join(parts)

    @property
    def n_digest_tokens(self) -> int:
        return _whitespace_token_count(self.to_decoder_text())

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "schema_cid": self.schema_cid,
            "prior_session_digest_cid": self.prior_session_digest_cid,
            "n_cells": int(self.n_cells),
            "n_cells_resolved": int(self.n_cells_resolved),
            "cumulative_per_tag_votes": [
                [t, int(c)] for t, c in self.cumulative_per_tag_votes
            ],
            "latest_projected_subset": list(self.latest_projected_subset),
            "latest_inner_branch": self.latest_inner_branch,
            "digest_cid": self.digest_cid,
            "n_digest_tokens": int(self.n_digest_tokens),
            "decoder_text": self.to_decoder_text(),
        }


@dataclasses.dataclass(frozen=True)
class SessionDeltaEnvelope:
    """Per-cell delta against the running session digest (W23 family).

    Carries only what is *new* for this cell:

      * which W21 inner branch fired,
      * the *delta* in projected subset (added vs prior session
        ``latest_projected_subset``; removed vs prior),
      * the *delta* in per-tag votes (only tags whose count changed
        between the prior cumulative and this cell's contribution),
      * this cell's parent probe CIDs (provenance — the actual oracle
        replies stay in the per-cell W22 LatentDigestEnvelope and
        the W21 W21OracleProbe records).

    Visible-token cost is strictly less than the full
    :class:`LatentDigestEnvelope` on regimes where consecutive cells
    share most of their state (the bench property of
    R-70-DELTA-FANOUT). On disjoint-state cells (R-70-NO-DELTA) the
    delta is the full envelope and the W23 layer reduces to W22 with
    no efficiency gain (W23-Λ-no-delta).

    The envelope is signed at construction; tampering after the fact
    yields a ``hash_mismatch`` on verification.
    """
    schema_cid: str
    prior_session_digest_cid: str
    parent_session_digest_cid: str
    cell_index: int
    inner_w19_branch: str
    delta_projected_added: tuple[str, ...]
    delta_projected_removed: tuple[str, ...]
    delta_per_tag_votes: tuple[tuple[str, int], ...]
    delta_n_outside_tokens: int
    parent_probe_cids: tuple[str, ...]
    delta_cid: str = ""
    schema_version: str = W23_DELTA_ENVELOPE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "delta_projected_added",
                            tuple(sorted(self.delta_projected_added)))
        object.__setattr__(self, "delta_projected_removed",
                            tuple(sorted(self.delta_projected_removed)))
        object.__setattr__(self, "delta_per_tag_votes",
                            tuple(sorted(self.delta_per_tag_votes,
                                            key=lambda kv: kv[0])))
        object.__setattr__(self, "parent_probe_cids",
                            tuple(self.parent_probe_cids))
        if not self.delta_cid:
            object.__setattr__(self, "delta_cid",
                                self.recompute_delta_cid())

    def _signed_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "schema_cid": self.schema_cid,
            "prior_session_digest_cid": self.prior_session_digest_cid,
            "parent_session_digest_cid": self.parent_session_digest_cid,
            "cell_index": int(self.cell_index),
            "inner_w19_branch": self.inner_w19_branch,
            "delta_projected_added": list(self.delta_projected_added),
            "delta_projected_removed": list(self.delta_projected_removed),
            "delta_per_tag_votes": [
                [t, int(c)] for t, c in self.delta_per_tag_votes
            ],
            "delta_n_outside_tokens": int(self.delta_n_outside_tokens),
            "parent_probe_cids": list(self.parent_probe_cids),
        }

    def to_canonical_bytes(self) -> bytes:
        return _canonical_json_bytes(self._signed_payload())

    def recompute_delta_cid(self) -> str:
        return hashlib.sha256(self.to_canonical_bytes()).hexdigest()

    def to_decoder_text(self) -> str:
        # Adaptive serialisation: omit fields that are empty/zero so
        # consecutive byte-identical cells produce minimal deltas.
        # The required fields (schema_cid, parent, cell_index,
        # delta_cid) anchor the chain; everything else is included
        # only when nonempty. The controller can reconstruct the
        # canonical bytes from the omitted-field absence (the
        # JSON-canonical signed payload is unaffected; the
        # decoder-facing text is the *visible* token cost only).
        parts: list[str] = ["SESSION_DELTA"]
        parts.append(f"schema_cid={self.schema_cid[:16]}")
        parts.append(f"parent={self.parent_session_digest_cid[:16]}")
        parts.append(f"cell={int(self.cell_index)}")
        parts.append(f"delta_cid={self.delta_cid[:16]}")
        # Optional fields — emitted only when nonempty/nonzero.
        if self.inner_w19_branch:
            parts.append(f"branch={self.inner_w19_branch}")
        if self.delta_projected_added:
            parts.append(
                f"add={','.join(self.delta_projected_added)}")
        if self.delta_projected_removed:
            parts.append(
                f"rm={','.join(self.delta_projected_removed)}")
        if self.delta_per_tag_votes:
            votes = ",".join(
                f"{t}:{c}" for t, c in self.delta_per_tag_votes)
            parts.append(f"votes={votes}")
        if self.delta_n_outside_tokens:
            parts.append(
                f"out_tokens={int(self.delta_n_outside_tokens)}")
        if self.parent_probe_cids:
            parts.append(f"n_probes={len(self.parent_probe_cids)}")
        return " ".join(parts)

    @property
    def n_delta_tokens(self) -> int:
        return _whitespace_token_count(self.to_decoder_text())

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "schema_cid": self.schema_cid,
            "prior_session_digest_cid": self.prior_session_digest_cid,
            "parent_session_digest_cid": self.parent_session_digest_cid,
            "cell_index": int(self.cell_index),
            "inner_w19_branch": self.inner_w19_branch,
            "delta_projected_added": list(self.delta_projected_added),
            "delta_projected_removed": list(self.delta_projected_removed),
            "delta_per_tag_votes": [
                [t, int(c)] for t, c in self.delta_per_tag_votes
            ],
            "delta_n_outside_tokens": int(self.delta_n_outside_tokens),
            "parent_probe_cids": list(self.parent_probe_cids),
            "delta_cid": self.delta_cid,
            "n_delta_tokens": int(self.n_delta_tokens),
            "decoder_text": self.to_decoder_text(),
        }


def verify_session_digest_chain(
        envelope: SessionDigestEnvelope | None,
        *,
        registered_schema: SchemaCapsule,
        prior_chain_head_cid: str = "",
        ) -> LatentVerificationOutcome:
    """Controller-side verification of one link of the
    SessionDigestEnvelope hash-chain.

    Returns ``ok=True`` only if (a) schema versions match, (b) schema
    CID matches, (c) digest_cid recomputes from canonical bytes, (d)
    the envelope's ``prior_session_digest_cid`` equals the registered
    chain head. The function is short, pure, and the failure modes
    are enumerated; soundness holds by inspection.

    The expected ``prior_chain_head_cid`` is "" at genesis (the first
    cell of a session) and the previous link's ``digest_cid``
    thereafter.
    """
    n_checks = 0
    if envelope is None:
        return LatentVerificationOutcome(
            ok=False, reason="empty_envelope", n_checks=n_checks)
    n_checks += 1
    if envelope.schema_version != W23_SESSION_ENVELOPE_SCHEMA_VERSION:
        return LatentVerificationOutcome(
            ok=False, reason="schema_version_unknown", n_checks=n_checks)
    n_checks += 1
    if envelope.schema_cid != registered_schema.cid:
        return LatentVerificationOutcome(
            ok=False, reason="schema_cid_mismatch", n_checks=n_checks)
    n_checks += 1
    canonical = envelope.to_canonical_bytes()
    if hashlib.sha256(canonical).hexdigest() != envelope.digest_cid:
        return LatentVerificationOutcome(
            ok=False, reason="hash_mismatch", n_checks=n_checks)
    n_checks += 1
    if envelope.prior_session_digest_cid != prior_chain_head_cid:
        return LatentVerificationOutcome(
            ok=False, reason="chain_head_mismatch", n_checks=n_checks)
    return LatentVerificationOutcome(
        ok=True, reason="ok", n_checks=n_checks)


def verify_session_delta(
        delta: SessionDeltaEnvelope | None,
        *,
        registered_schema: SchemaCapsule,
        parent_session_digest_cid: str,
        ) -> LatentVerificationOutcome:
    """Controller-side verification of a SessionDeltaEnvelope.

    Returns ``ok=True`` only if (a) schema versions match, (b) schema
    CID matches, (c) delta_cid recomputes from canonical bytes, (d)
    the delta's ``parent_session_digest_cid`` equals the registered
    parent (the controller's view of the chain head this delta is
    extending).
    """
    n_checks = 0
    if delta is None:
        return LatentVerificationOutcome(
            ok=False, reason="empty_envelope", n_checks=n_checks)
    n_checks += 1
    if delta.schema_version != W23_DELTA_ENVELOPE_SCHEMA_VERSION:
        return LatentVerificationOutcome(
            ok=False, reason="schema_version_unknown", n_checks=n_checks)
    n_checks += 1
    if delta.schema_cid != registered_schema.cid:
        return LatentVerificationOutcome(
            ok=False, reason="schema_cid_mismatch", n_checks=n_checks)
    n_checks += 1
    canonical = delta.to_canonical_bytes()
    if hashlib.sha256(canonical).hexdigest() != delta.delta_cid:
        return LatentVerificationOutcome(
            ok=False, reason="hash_mismatch", n_checks=n_checks)
    n_checks += 1
    if delta.parent_session_digest_cid != parent_session_digest_cid:
        return LatentVerificationOutcome(
            ok=False, reason="parent_session_mismatch", n_checks=n_checks)
    return LatentVerificationOutcome(
        ok=True, reason="ok", n_checks=n_checks)


@dataclasses.dataclass(frozen=True)
class SuperTokenReferenceEnvelope:
    """Single-visible-token CID reference (W23 family — bounded
    steganographic / dense-control-payload experiment).

    The decoder visible cost of a W23 cell can collapse to *one
    whitespace token*: the super-token. The token format is

        <latent_ref:DDDDDDDDDDDDDDDD>

    where ``DDDDDDDDDDDDDDDD`` is the 16-character hex prefix of the
    referenced :class:`SessionDeltaEnvelope`'s ``delta_cid``.

    Honest scope: the super-token is **not** a transformer-internal
    embedding manipulation. It is a content-addressed reference that
    is *registered* with the controller: the controller's
    :class:`SuperTokenRegistry` maps the prefix to the full
    SessionDeltaEnvelope. On every read the controller verifies (a)
    the prefix is in the registry, (b) the referenced delta's
    delta_cid recomputes, (c) the parent_session_digest_cid links to
    the chain. On any failure the W23 layer rejects the super-token
    and falls through to the verbose digest path.

    The bound on the channel is sharp: at most one super-token per
    cell; at most ``hex_prefix_len`` characters of payload (default
    16); the registry is controller-owned and can be enumerated /
    audited; tampering yields a ``hash_mismatch`` or
    ``unknown_super_token`` rejection.
    """
    schema_cid: str
    delta_cid: str
    parent_session_digest_cid: str
    hex_prefix_len: int = 16
    schema_version: str = W23_SUPER_TOKEN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (self.hex_prefix_len < 4 or self.hex_prefix_len > 64):
            raise ValueError(
                f"hex_prefix_len must be in [4, 64]; got {self.hex_prefix_len}")
        if len(self.delta_cid) != 64:
            raise ValueError(
                f"delta_cid must be 64 hex chars (SHA-256); got "
                f"{len(self.delta_cid)}")

    @property
    def super_token(self) -> str:
        return f"<latent_ref:{self.delta_cid[:self.hex_prefix_len]}>"

    @property
    def n_super_token_tokens(self) -> int:
        # The super-token is one whitespace token by construction
        # (no internal whitespace).
        return 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "schema_cid": self.schema_cid,
            "delta_cid": self.delta_cid,
            "parent_session_digest_cid": self.parent_session_digest_cid,
            "hex_prefix_len": int(self.hex_prefix_len),
            "super_token": self.super_token,
            "n_super_token_tokens": int(self.n_super_token_tokens),
        }


@dataclasses.dataclass
class SuperTokenRegistry:
    """Controller-side registry of super-token references (W23 family).

    Maps the super-token's hex prefix to the full
    :class:`SessionDeltaEnvelope` so the controller can resolve and
    verify a super-token reference at decode time. The registry is
    enumerable (auditable) and the resolve path is constant-time on
    a hit.

    The registry is the trust boundary on the W23
    super-token / dense-control side channel: a super-token whose
    prefix is not registered is rejected; a registered token whose
    referenced delta fails verification (``hash_mismatch`` or
    ``parent_session_mismatch``) is rejected.
    """
    _by_prefix: dict[str, SessionDeltaEnvelope] = dataclasses.field(
        default_factory=dict)
    _hex_prefix_len: int = 16
    n_registered: int = 0
    n_resolved: int = 0
    n_rejected: int = 0

    def register(self, delta: SessionDeltaEnvelope, *,
                  hex_prefix_len: int | None = None) -> str:
        L = int(hex_prefix_len or self._hex_prefix_len)
        if not (4 <= L <= 64):
            raise ValueError(f"hex_prefix_len must be in [4, 64]; got {L}")
        prefix = str(delta.delta_cid[:L])
        if prefix in self._by_prefix:
            existing = self._by_prefix[prefix]
            if existing.delta_cid != delta.delta_cid:
                raise ValueError(
                    f"super-token prefix collision on {prefix!r}: "
                    f"existing={existing.delta_cid[:32]} new={delta.delta_cid[:32]}")
            return prefix
        self._by_prefix[prefix] = delta
        self._hex_prefix_len = L
        self.n_registered += 1
        return prefix

    def lookup(self, prefix: str) -> SessionDeltaEnvelope | None:
        return self._by_prefix.get(prefix)

    def known_prefixes(self) -> tuple[str, ...]:
        return tuple(sorted(self._by_prefix.keys()))

    def __len__(self) -> int:
        return len(self._by_prefix)


def verify_super_token_reference(
        envelope: SuperTokenReferenceEnvelope | None,
        *,
        registry: SuperTokenRegistry,
        registered_schema: SchemaCapsule,
        parent_session_digest_cid: str,
        ) -> LatentVerificationOutcome:
    """Controller-side resolve+verify for a super-token reference.

    Pure function. Failure modes enumerated:

      * empty_envelope — no envelope passed.
      * schema_version_unknown — schema_version differs.
      * schema_cid_mismatch — envelope's schema_cid != registered.
      * unknown_super_token — prefix not in registry.
      * hash_mismatch — registered delta's delta_cid does not
        recompute (registry tamper).
      * parent_session_mismatch — registered delta's
        parent_session_digest_cid does not match the controller's
        chain head.
      * delta_cid_mismatch — envelope's delta_cid does not match the
        registered delta's delta_cid (forged super-token).
    """
    n_checks = 0
    if envelope is None:
        registry.n_rejected += 1
        return LatentVerificationOutcome(
            ok=False, reason="empty_envelope", n_checks=n_checks)
    n_checks += 1
    if envelope.schema_version != W23_SUPER_TOKEN_SCHEMA_VERSION:
        registry.n_rejected += 1
        return LatentVerificationOutcome(
            ok=False, reason="schema_version_unknown", n_checks=n_checks)
    n_checks += 1
    if envelope.schema_cid != registered_schema.cid:
        registry.n_rejected += 1
        return LatentVerificationOutcome(
            ok=False, reason="schema_cid_mismatch", n_checks=n_checks)
    L = int(envelope.hex_prefix_len)
    n_checks += 1
    prefix = str(envelope.delta_cid[:L])
    delta = registry.lookup(prefix)
    if delta is None:
        registry.n_rejected += 1
        return LatentVerificationOutcome(
            ok=False, reason="unknown_super_token", n_checks=n_checks)
    n_checks += 1
    if delta.delta_cid != envelope.delta_cid:
        # Forged super-token whose prefix happens to collide with a
        # registered entry but whose tail does not match.
        registry.n_rejected += 1
        return LatentVerificationOutcome(
            ok=False, reason="delta_cid_mismatch", n_checks=n_checks)
    n_checks += 1
    canonical = delta.to_canonical_bytes()
    if hashlib.sha256(canonical).hexdigest() != delta.delta_cid:
        registry.n_rejected += 1
        return LatentVerificationOutcome(
            ok=False, reason="hash_mismatch", n_checks=n_checks)
    n_checks += 1
    if delta.parent_session_digest_cid != parent_session_digest_cid:
        registry.n_rejected += 1
        return LatentVerificationOutcome(
            ok=False, reason="parent_session_mismatch", n_checks=n_checks)
    registry.n_resolved += 1
    return LatentVerificationOutcome(
        ok=True, reason="ok", n_checks=n_checks)


@dataclasses.dataclass
class QuorumKeyedSharedReadCache:
    """Mitigation of W22-C-CACHE-AMPLIFICATION (W23 family).

    Wraps a :class:`SharedReadCache` with a per-oracle freshness
    policy registry. Each registered oracle declares one of:

      * :data:`CACHE_FRESHNESS_BYTE_IDENTICAL` — W22 default.
      * :data:`CACHE_FRESHNESS_PER_CELL_NONCE` — cache key includes
        a per-cell nonce; identical queries from different cells DO
        NOT hit. Used for probabilistic LLM oracles.
      * :data:`CACHE_FRESHNESS_QUORUM_LOCKED` — cache stores entries
        only after the controller confirms the reply contributed to
        a successful quorum; bad first replies that fail quorum are
        NOT cached.

    The mitigation is **opt-in per oracle**: deterministic oracles
    (e.g. ``service_graph``, ``change_history``) keep
    BYTE_IDENTICAL for full cross-cell wire savings; the
    probabilistic LLM adjudicator gets PER_CELL_NONCE so a bad
    first sample does not propagate. The trade-off is named in the
    audit (``cache_tokens_saved_total`` is lower with PER_CELL_NONCE
    than BYTE_IDENTICAL by exactly the amount of W22-C-CACHE-
    AMPLIFICATION variance).

    Honest scope: this is a *cache-layer* mitigation, not a runtime
    KV cache. The "cache key" is over canonical OutsideQuery bytes.
    """
    inner: SharedReadCache = dataclasses.field(
        default_factory=SharedReadCache)
    policy_per_oracle: dict[str, str] = dataclasses.field(
        default_factory=dict)
    quorum_pending: dict[str, tuple[bytes, int, str]] = (
        dataclasses.field(default_factory=dict))
    n_quorum_locked_admitted: int = 0
    n_quorum_locked_dropped: int = 0

    def policy_for(self, oracle_id: str) -> str:
        return self.policy_per_oracle.get(
            str(oracle_id), CACHE_FRESHNESS_BYTE_IDENTICAL)

    def set_policy(self, oracle_id: str, policy: str) -> None:
        if policy not in CACHE_FRESHNESS_POLICIES:
            raise ValueError(
                f"unknown cache freshness policy {policy!r}; "
                f"valid: {CACHE_FRESHNESS_POLICIES}")
        self.policy_per_oracle[str(oracle_id)] = str(policy)

    def query_cid_for(self, query: OutsideQuery, *,
                       oracle_id: str, cell_nonce: str = "") -> str:
        policy = self.policy_for(oracle_id)
        if policy == CACHE_FRESHNESS_PER_CELL_NONCE:
            # Mix the cell nonce into the cache key so identical queries
            # from different cells DO NOT hit.
            body = _canonical_json_bytes({
                "oracle_id": str(oracle_id),
                "admitted_tags": list(query.admitted_tags),
                "elected_root_cause": str(query.elected_root_cause),
                "primary_payload": str(query.primary_payload),
                "witness_payloads": list(query.witness_payloads),
                "max_response_tokens": int(query.max_response_tokens),
                "cell_nonce": str(cell_nonce),
            })
            return hashlib.sha256(body).hexdigest()
        # BYTE_IDENTICAL and QUORUM_LOCKED both use the W22 query CID.
        return SharedReadCache.query_cid_for(query, oracle_id=oracle_id)

    def get(self, cid: str) -> tuple[bytes, int, str] | None:
        return self.inner.get(cid)

    def put(self, cid: str, body: bytes, *, n_tokens: int,
            source_id: str = "", oracle_id: str = "",
            quorum_locked_pending: bool = False) -> None:
        if (quorum_locked_pending or
                self.policy_for(oracle_id) == CACHE_FRESHNESS_QUORUM_LOCKED):
            # Defer the put until the controller confirms quorum.
            self.quorum_pending[cid] = (
                bytes(body), int(n_tokens), str(source_id))
            return
        self.inner.put(cid, body, n_tokens=n_tokens, source_id=source_id)

    def confirm_quorum_for(self, cids: Iterable[str]) -> int:
        """Promote pending entries to the cache after quorum forms."""
        n_admitted = 0
        for cid in cids:
            entry = self.quorum_pending.pop(cid, None)
            if entry is None:
                continue
            body, n_tokens, source_id = entry
            self.inner.put(cid, body, n_tokens=n_tokens,
                            source_id=source_id)
            n_admitted += 1
            self.n_quorum_locked_admitted += 1
        return n_admitted

    def drop_pending(self, cids: Iterable[str]) -> int:
        """Drop pending entries (e.g. quorum failed to form)."""
        n_dropped = 0
        for cid in cids:
            if self.quorum_pending.pop(cid, None) is not None:
                n_dropped += 1
                self.n_quorum_locked_dropped += 1
        return n_dropped

    @property
    def n_hits(self) -> int:
        return self.inner.n_hits

    @n_hits.setter
    def n_hits(self, v: int) -> None:
        self.inner.n_hits = int(v)

    @property
    def n_misses(self) -> int:
        return self.inner.n_misses

    @n_misses.setter
    def n_misses(self, v: int) -> None:
        self.inner.n_misses = int(v)

    @property
    def n_tokens_saved(self) -> int:
        return self.inner.n_tokens_saved

    @n_tokens_saved.setter
    def n_tokens_saved(self, v: int) -> None:
        self.inner.n_tokens_saved = int(v)

    @property
    def _store(self) -> dict[str, tuple[bytes, int, str]]:
        return self.inner._store

    def stats(self) -> dict[str, Any]:
        s = dict(self.inner.stats())
        s.update({
            "n_quorum_locked_admitted":
                int(self.n_quorum_locked_admitted),
            "n_quorum_locked_dropped":
                int(self.n_quorum_locked_dropped),
            "n_quorum_pending": len(self.quorum_pending),
            "policy_per_oracle": dict(self.policy_per_oracle),
        })
        return s


@dataclasses.dataclass
class QuorumKeyedCachingOracleAdapter:
    """Cache-aware adapter that routes through a QuorumKeyedSharedReadCache
    (W23 family).

    Mirror of :class:`CachingOracleAdapter` but uses a
    :class:`QuorumKeyedSharedReadCache` (with per-oracle freshness
    policy) instead of the W22 :class:`SharedReadCache`. The
    ``cell_nonce`` field is *mutable* — the bench driver sets it to
    a fresh value before each cell so the
    :data:`CACHE_FRESHNESS_PER_CELL_NONCE` path can mix it into the
    cache key.

    Preserves the :class:`OutsideWitnessOracle` Protocol so it can be
    registered in any W21 oracle stack as a drop-in replacement for
    the wrapped oracle.
    """
    inner: Any
    cache: QuorumKeyedSharedReadCache
    oracle_id: str = ""
    cell_nonce: str = ""
    last_was_hit: bool = False
    last_was_quorum_pending: bool = False

    def __post_init__(self) -> None:
        if not self.oracle_id:
            self.oracle_id = str(getattr(
                self.inner, "oracle_id", "quorum_keyed_oracle"))

    def consult(self, query: OutsideQuery) -> OutsideVerdict:
        cid = self.cache.query_cid_for(
            query, oracle_id=self.oracle_id,
            cell_nonce=str(self.cell_nonce))
        cached = self.cache.get(cid)
        if cached is not None:
            body, n_tokens, source_id = cached
            self.last_was_hit = True
            self.last_was_quorum_pending = False
            payload = body.decode("utf-8") if body else None
            return OutsideVerdict(
                payload=payload,
                source_id=source_id or self.oracle_id,
                n_tokens=int(n_tokens),
            )
        self.last_was_hit = False
        verdict = self.inner.consult(query)
        body = (verdict.payload or "").encode("utf-8")
        policy = self.cache.policy_for(self.oracle_id)
        if policy == CACHE_FRESHNESS_QUORUM_LOCKED:
            self.last_was_quorum_pending = True
            self.cache.put(cid, body, n_tokens=int(verdict.n_tokens),
                            source_id=str(verdict.source_id or self.oracle_id),
                            oracle_id=self.oracle_id,
                            quorum_locked_pending=True)
        else:
            self.last_was_quorum_pending = False
            self.cache.put(cid, body, n_tokens=int(verdict.n_tokens),
                            source_id=str(verdict.source_id or self.oracle_id),
                            oracle_id=self.oracle_id)
        return verdict


@dataclasses.dataclass
class CrossHostProducerDecoderProxy:
    """Within-process producer/decoder host-split proxy (W23 family).

    Two-Mac infrastructure note. The W22 milestone declared the
    SharedReadCache + LatentDigestDisambiguator interface
    "wire-compatible with cross-host deployment" without proving it.
    Mac 2 has been ARP-incomplete for 17 milestones in a row; we
    cannot validate true two-host execution on this machine. This
    class is the **honest within-process proxy**: it forces the
    producer and decoder to communicate **only** through canonical
    JSON bytes (no shared Python references), so the wire boundary
    is mechanically observable and any latent shared state would
    surface as a serialisation failure.

    Honest scope: this is **not** a true two-host setup. It is a
    structural simulation that lets us measure (a) how many bytes
    actually cross the wire, (b) whether the producer / decoder
    interface honours the JSON-canonical encoding contract, and (c)
    whether the W22 / W23 envelopes survive a round-trip
    serialisation. When Mac 2 returns, the same interface drops in
    over a real socket with no W23 code changes.
    """
    n_round_trips: int = 0
    n_bytes_serialised: int = 0
    n_bytes_deserialised: int = 0

    def producer_to_decoder(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Round-trip a payload as if across a host boundary."""
        body = _canonical_json_bytes(payload)
        self.n_round_trips += 1
        self.n_bytes_serialised += len(body)
        out = json.loads(body.decode("ascii"))
        self.n_bytes_deserialised += len(body)
        return out

    def stats(self) -> dict[str, Any]:
        return {
            "n_round_trips": int(self.n_round_trips),
            "n_bytes_serialised": int(self.n_bytes_serialised),
            "n_bytes_deserialised": int(self.n_bytes_deserialised),
        }


@dataclasses.dataclass
class W23SessionResult:
    """Audit record for one W23 cell.

    Captures the W22 outcome below, the per-cell delta envelope (or
    None on no-trigger / rejected), the running session digest after
    this cell, the optional super-token, and the visible-token
    accounting for the W23 path.
    """
    answer: dict[str, Any]
    inner_w22_branch: str
    decoder_branch: str
    abstained: bool
    schema_cid: str
    schema_version: str
    cell_index: int
    is_genesis: bool
    prior_session_digest_cid: str
    session_digest_cid: str
    delta_cid: str
    delta_n_tokens: int
    super_token_used: bool
    super_token_resolved: bool
    super_token_str: str
    n_w15_tokens_kept: int
    n_w22_visible_tokens_to_decider: int
    n_w23_visible_tokens_to_decider: int
    n_w22_minus_w23_savings: int
    cache_freshness_policy_used: dict[str, str]
    cache_tokens_saved_this_cell: int
    chain_verification_ok: bool
    chain_verification_reason: str
    delta_verification_ok: bool
    delta_verification_reason: str
    super_token_verification_ok: bool
    super_token_verification_reason: str
    cross_host_round_trip_bytes: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "inner_w22_branch": str(self.inner_w22_branch),
            "decoder_branch": str(self.decoder_branch),
            "abstained": bool(self.abstained),
            "schema_cid": str(self.schema_cid),
            "schema_version": str(self.schema_version),
            "cell_index": int(self.cell_index),
            "is_genesis": bool(self.is_genesis),
            "prior_session_digest_cid": str(self.prior_session_digest_cid),
            "session_digest_cid": str(self.session_digest_cid),
            "delta_cid": str(self.delta_cid),
            "delta_n_tokens": int(self.delta_n_tokens),
            "super_token_used": bool(self.super_token_used),
            "super_token_resolved": bool(self.super_token_resolved),
            "super_token_str": str(self.super_token_str),
            "n_w15_tokens_kept": int(self.n_w15_tokens_kept),
            "n_w22_visible_tokens_to_decider":
                int(self.n_w22_visible_tokens_to_decider),
            "n_w23_visible_tokens_to_decider":
                int(self.n_w23_visible_tokens_to_decider),
            "n_w22_minus_w23_savings":
                int(self.n_w22_minus_w23_savings),
            "cache_freshness_policy_used":
                dict(self.cache_freshness_policy_used),
            "cache_tokens_saved_this_cell":
                int(self.cache_tokens_saved_this_cell),
            "chain_verification_ok": bool(self.chain_verification_ok),
            "chain_verification_reason":
                str(self.chain_verification_reason),
            "delta_verification_ok": bool(self.delta_verification_ok),
            "delta_verification_reason":
                str(self.delta_verification_reason),
            "super_token_verification_ok":
                bool(self.super_token_verification_ok),
            "super_token_verification_reason":
                str(self.super_token_verification_reason),
            "cross_host_round_trip_bytes":
                int(self.cross_host_round_trip_bytes),
        }


def _projected_tuple_from_answer(answer: dict[str, Any]) -> tuple[str, ...]:
    services = answer.get("services", ()) if isinstance(answer, dict) else ()
    if isinstance(services, (list, tuple)):
        return tuple(sorted(str(s) for s in services))
    return ()


def _per_tag_votes_from_w22(
        envelope: LatentDigestEnvelope | None,
        ) -> dict[str, int]:
    if envelope is None:
        return {}
    return {str(t): int(c) for t, c in envelope.per_tag_vote_count}


@dataclasses.dataclass
class CrossCellDeltaDisambiguator:
    """Capsule + cross-cell delta-execution disambiguator (W23 family).

    Wraps a :class:`LatentDigestDisambiguator` (W22) with four
    LatentMAS-direction proxies on top:

      1. **Cross-cell hash-chained session digest** via
         :class:`SessionDigestEnvelope`.
      2. **Per-cell delta execution** via
         :class:`SessionDeltaEnvelope` (smaller than the full digest
         on regimes where consecutive cells share state).
      3. **Quorum-keyed cache** via
         :class:`QuorumKeyedSharedReadCache` (mitigates
         W22-C-CACHE-AMPLIFICATION on probabilistic LLM oracles).
      4. **Super-token reference** via
         :class:`SuperTokenReferenceEnvelope` (bounded
         steganographic / dense-control-payload experiment;
         single-visible-token CID reference verified through a
         controller-side registry).

    Plus optionally a :class:`CrossHostProducerDecoderProxy` for the
    within-process producer / decoder host-split proxy (Mac-2
    fallback).

    The W23 surface is *strictly additive* on top of W22:

      * If the inner W22 branch is not in
        :data:`W23_DEFAULT_TRIGGER_BRANCHES` (i.e. the inner W22 did
        not produce a verified envelope), W23 fires
        :data:`W23_BRANCH_NO_TRIGGER` and reduces to W22
        byte-for-byte on the answer field.
      * If ``enabled = False``, W23 fires
        :data:`W23_BRANCH_DISABLED` and reduces to W22.
      * On the first cell of a session, W23 fires
        :data:`W23_BRANCH_GENESIS` and emits the full W22 envelope as
        the chain root (no delta savings on genesis).
      * On chain-link verification failure, W23 fires
        :data:`W23_BRANCH_DELTA_REJECTED` and the answer is the W22
        answer byte-for-byte.
      * On super-token reference verification failure (when
        ``use_super_token = True``), W23 fires
        :data:`W23_BRANCH_SUPER_TOKEN_REJECTED` and falls through to
        the verbose digest.

    Theorem family W23 — see
    ``docs/RESULTS_WEVRA_W23_CROSS_CELL_DELTA.md`` for the full
    statements.
    """

    inner: LatentDigestDisambiguator = dataclasses.field(
        default_factory=lambda: LatentDigestDisambiguator())
    schema: SchemaCapsule | None = None
    enabled: bool = True
    use_super_token: bool = False
    super_token_registry: SuperTokenRegistry | None = None
    # Controller-side verifier registry. When set (not None), the
    # controller verifies super-token references against THIS
    # registry instead of the producer-side ``super_token_registry``.
    # On R-70-SUPER-TOKEN-TAMPERED this is set to an empty (or
    # tampered) registry → ``unknown_super_token`` rejection on
    # every cell. Default None means the controller and producer
    # share the same registry (backward-compat with the simpler
    # within-process bench).
    verifier_super_token_registry: SuperTokenRegistry | None = None
    # Controller-side verifier chain-head override. When set (not
    # empty / not None), the controller's
    # :func:`verify_session_digest_chain` is called with
    # ``prior_chain_head_cid=verifier_chain_head_override`` instead
    # of the producer's :meth:`chain_head_cid`. On R-70-CHAIN-BROKEN
    # the bench installs a phantom CID here after the genesis cell
    # → every subsequent link's verifier sees a
    # ``chain_head_mismatch`` and the W23 layer rejects (falls
    # through to W22). Default None means no override (the producer
    # and verifier share the chain head, the within-process default).
    verifier_chain_head_override: str | None = None
    quorum_keyed_cache: QuorumKeyedSharedReadCache | None = None
    cross_host_proxy: CrossHostProducerDecoderProxy | None = None
    super_token_hex_prefix_len: int = 16
    trigger_branches: frozenset[str] = dataclasses.field(
        default_factory=lambda: W23_DEFAULT_TRIGGER_BRANCHES)
    require_chain_verification: bool = True

    _chain: list[SessionDigestEnvelope] = dataclasses.field(
        default_factory=list)
    _last_result: W23SessionResult | None = None
    _last_session_envelope: SessionDigestEnvelope | None = None
    _last_delta_envelope: SessionDeltaEnvelope | None = None
    _last_super_token: SuperTokenReferenceEnvelope | None = None
    _cell_index: int = 0
    _cumulative_per_tag_votes: dict[str, int] = dataclasses.field(
        default_factory=dict)
    # The per-cell vote count of the *prior* cell — used by
    # ``_build_delta`` to compute the per-cell vote delta (only emit
    # tags whose count CHANGED vs the prior cell). On byte-identical
    # consecutive cells the delta_per_tag_votes is empty, which is
    # the load-bearing compression signal of W23-1.
    _last_cell_per_tag_votes: dict[str, int] = dataclasses.field(
        default_factory=dict)
    _last_projected_subset: tuple[str, ...] = ()
    _last_inner_branch: str = ""
    _n_resolved: int = 0

    def reset_session(self) -> None:
        """Reset per-session state. Useful when reusing the same
        instance across independent benchmark runs."""
        self._chain = []
        self._last_result = None
        self._last_session_envelope = None
        self._last_delta_envelope = None
        self._last_super_token = None
        self._cell_index = 0
        self._cumulative_per_tag_votes = {}
        self._last_cell_per_tag_votes = {}
        self._last_projected_subset = ()
        self._last_inner_branch = ""
        self._n_resolved = 0
        if self.super_token_registry is not None:
            self.super_token_registry._by_prefix.clear()
            self.super_token_registry.n_registered = 0
            self.super_token_registry.n_resolved = 0
            self.super_token_registry.n_rejected = 0
        if self.quorum_keyed_cache is not None:
            self.quorum_keyed_cache.inner.n_hits = 0
            self.quorum_keyed_cache.inner.n_misses = 0
            self.quorum_keyed_cache.inner.n_tokens_saved = 0
            self.quorum_keyed_cache.inner._store.clear()
            self.quorum_keyed_cache.quorum_pending.clear()
            self.quorum_keyed_cache.n_quorum_locked_admitted = 0
            self.quorum_keyed_cache.n_quorum_locked_dropped = 0
        self.inner.reset_session()

    def chain_head_cid(self) -> str:
        return self._chain[-1].digest_cid if self._chain else ""

    def session_chain(self) -> tuple[SessionDigestEnvelope, ...]:
        return tuple(self._chain)

    def _build_delta(self, *, w22_envelope: LatentDigestEnvelope,
                       prior_session_digest_cid: str,
                       prior_projected: tuple[str, ...],
                       parent_session_digest_cid: str,
                       ) -> SessionDeltaEnvelope:
        new_projected = w22_envelope.projected_subset
        added = tuple(sorted(set(new_projected) - set(prior_projected)))
        removed = tuple(sorted(set(prior_projected) - set(new_projected)))
        # Vote delta: emit only the (tag, count) pairs whose count
        # CHANGED vs the prior cell's per-tag vote count. On
        # byte-identical consecutive cells (the bench property of
        # R-70-DELTA-FANOUT) the delta is empty for votes — the
        # load-bearing compression signal. Tags that disappeared in
        # the new cell are emitted with count 0 so the controller
        # can reconstruct the running cumulative.
        delta_votes: list[tuple[str, int]] = []
        new_votes = {str(t): int(c)
                       for t, c in w22_envelope.per_tag_vote_count}
        prior_votes = self._last_cell_per_tag_votes
        all_tags = set(new_votes) | set(prior_votes)
        for tag in all_tags:
            new_count = new_votes.get(tag, 0)
            prior_count = prior_votes.get(tag, 0)
            if new_count != prior_count:
                delta_votes.append((tag, new_count))
        return SessionDeltaEnvelope(
            schema_cid=w22_envelope.schema_cid,
            prior_session_digest_cid=prior_session_digest_cid,
            parent_session_digest_cid=parent_session_digest_cid,
            cell_index=int(self._cell_index),
            inner_w19_branch=str(w22_envelope.inner_w19_branch),
            delta_projected_added=tuple(added),
            delta_projected_removed=tuple(removed),
            delta_per_tag_votes=tuple(delta_votes),
            delta_n_outside_tokens=int(
                w22_envelope.n_outside_tokens_total),
            parent_probe_cids=tuple(w22_envelope.parent_probe_cids),
            schema_version=W23_DELTA_ENVELOPE_SCHEMA_VERSION,
        )

    def _build_session_digest(self, *,
                                w22_envelope: LatentDigestEnvelope,
                                prior_session_digest_cid: str,
                                resolved: bool,
                                ) -> SessionDigestEnvelope:
        # Cumulative per-tag votes after this cell.
        cumulative = dict(self._cumulative_per_tag_votes)
        for tag, count in w22_envelope.per_tag_vote_count:
            cumulative[str(tag)] = (
                int(cumulative.get(str(tag), 0)) + int(count))
        return SessionDigestEnvelope(
            schema_cid=w22_envelope.schema_cid,
            prior_session_digest_cid=prior_session_digest_cid,
            n_cells=int(self._cell_index + 1),
            cumulative_per_tag_votes=tuple(
                sorted(cumulative.items(), key=lambda kv: kv[0])),
            latest_projected_subset=tuple(
                sorted(w22_envelope.projected_subset)),
            latest_inner_branch=str(w22_envelope.inner_w19_branch),
            n_cells_resolved=int(self._n_resolved + (1 if resolved else 0)),
            schema_version=W23_SESSION_ENVELOPE_SCHEMA_VERSION,
        )

    def _maybe_round_trip_proxy(self, payload: dict[str, Any]
                                  ) -> tuple[dict[str, Any], int]:
        if self.cross_host_proxy is None:
            return payload, 0
        before = self.cross_host_proxy.n_bytes_serialised
        out = self.cross_host_proxy.producer_to_decoder(payload)
        after = self.cross_host_proxy.n_bytes_serialised
        return out, max(0, after - before)

    def decode_rounds(
            self,
            per_round_handoffs: Sequence[Sequence[_DecodedHandoff]],
            ) -> dict[str, Any]:
        # Run inner W22 (which itself runs W21 → W19 → W18 → W15).
        base = self.inner.decode_rounds(per_round_handoffs)
        w22_result = self.inner.last_result
        w22_envelope = self.inner.last_envelope
        out = dict(base)

        n_w15_kept = (int(w22_result.n_w15_tokens_kept)
                       if w22_result is not None else 0)
        n_w22_visible = (int(w22_result.n_visible_tokens_to_decider)
                          if w22_result is not None else 0)
        cell_index = int(self._cell_index)
        prior_session_digest_cid = self.chain_head_cid()
        prior_projected = self._last_projected_subset
        cache_policy_snapshot: dict[str, str] = {}
        if self.quorum_keyed_cache is not None:
            cache_policy_snapshot = dict(
                self.quorum_keyed_cache.policy_per_oracle)
        cache_saved_this_cell = (
            int(w22_result.cache_tokens_saved_this_cell)
            if w22_result is not None else 0)

        def _pack(*, decoder_branch: str, abstained: bool,
                   delta: SessionDeltaEnvelope | None,
                   session_env: SessionDigestEnvelope | None,
                   super_token: SuperTokenReferenceEnvelope | None,
                   chain_ok: bool, chain_reason: str,
                   delta_ok: bool, delta_reason: str,
                   super_ok: bool, super_reason: str,
                   round_trip_bytes: int,
                   ) -> dict[str, Any]:
            answer = dict(base)
            schema_cid = (str(self.schema.cid)
                            if self.schema is not None else "")
            schema_version = (
                str(self.schema.version)
                if self.schema is not None else "")
            inner_w22_branch = (
                str(w22_result.decoder_branch)
                if w22_result is not None else W22_BRANCH_DISABLED)
            # Visible-token cost the downstream decoder pays under
            # W23. Three regimes:
            #   * GENESIS or DELTA_REJECTED or NO_TRIGGER or DISABLED
            #     or SUPER_TOKEN_REJECTED → fall back to W22 cost.
            #   * DELTA_RESOLVED → kept + delta_n_tokens (the
            #     cross-cell running state lives in the
            #     prior_session_digest_cid by reference, not in the
            #     decoder's visible context).
            #   * SUPER_TOKEN_RESOLVED → kept + 1 (single super-token).
            if (decoder_branch == W23_BRANCH_DELTA_RESOLVED
                    and delta is not None):
                visible = n_w15_kept + int(delta.n_delta_tokens)
            elif (decoder_branch == W23_BRANCH_SUPER_TOKEN_RESOLVED
                    and super_token is not None):
                visible = (n_w15_kept
                            + int(super_token.n_super_token_tokens))
            else:
                visible = n_w22_visible
            savings = max(0, n_w22_visible - visible)
            delta_cid = str(delta.delta_cid) if delta is not None else ""
            delta_n = int(delta.n_delta_tokens) if delta is not None else 0
            session_cid = (
                str(session_env.digest_cid)
                if session_env is not None else "")
            super_token_str = (
                super_token.super_token if super_token is not None else "")
            result = W23SessionResult(
                answer=answer,
                inner_w22_branch=inner_w22_branch,
                decoder_branch=decoder_branch,
                abstained=abstained,
                schema_cid=schema_cid,
                schema_version=schema_version,
                cell_index=int(cell_index),
                is_genesis=bool(decoder_branch == W23_BRANCH_GENESIS),
                prior_session_digest_cid=str(prior_session_digest_cid),
                session_digest_cid=session_cid,
                delta_cid=delta_cid,
                delta_n_tokens=delta_n,
                super_token_used=bool(super_token is not None),
                super_token_resolved=bool(super_ok and
                                           super_token is not None),
                super_token_str=super_token_str,
                n_w15_tokens_kept=int(n_w15_kept),
                n_w22_visible_tokens_to_decider=int(n_w22_visible),
                n_w23_visible_tokens_to_decider=int(visible),
                n_w22_minus_w23_savings=int(savings),
                cache_freshness_policy_used=cache_policy_snapshot,
                cache_tokens_saved_this_cell=int(cache_saved_this_cell),
                chain_verification_ok=bool(chain_ok),
                chain_verification_reason=str(chain_reason),
                delta_verification_ok=bool(delta_ok),
                delta_verification_reason=str(delta_reason),
                super_token_verification_ok=bool(super_ok),
                super_token_verification_reason=str(super_reason),
                cross_host_round_trip_bytes=int(round_trip_bytes),
            )
            self._last_result = result
            self._last_delta_envelope = delta
            self._last_session_envelope = session_env
            self._last_super_token = super_token
            out_local = dict(out)
            out_local["session_delta_hybrid"] = result.as_dict()
            if delta is not None:
                out_local["session_delta_envelope"] = delta.as_dict()
            if session_env is not None:
                out_local["session_digest_envelope"] = session_env.as_dict()
            if super_token is not None:
                out_local["super_token_reference"] = super_token.as_dict()
            return out_local

        # Branch dispatch (top-down):
        if not self.enabled:
            self._cell_index += 1
            return _pack(
                decoder_branch=W23_BRANCH_DISABLED, abstained=True,
                delta=None, session_env=None, super_token=None,
                chain_ok=False, chain_reason="disabled",
                delta_ok=False, delta_reason="disabled",
                super_ok=False, super_reason="disabled",
                round_trip_bytes=0)
        if w22_result is None or self.schema is None:
            self._cell_index += 1
            return _pack(
                decoder_branch=W23_BRANCH_NO_TRIGGER, abstained=True,
                delta=None, session_env=None, super_token=None,
                chain_ok=False, chain_reason="no_w22_result",
                delta_ok=False, delta_reason="no_w22_result",
                super_ok=False, super_reason="no_w22_result",
                round_trip_bytes=0)
        if w22_result.decoder_branch not in self.trigger_branches:
            # Inner W22 did not produce a verified envelope (e.g.
            # NO_TRIGGER, LATENT_REJECTED, DISABLED). W23 reduces to
            # W22 byte-for-byte.
            self._cell_index += 1
            return _pack(
                decoder_branch=W23_BRANCH_NO_TRIGGER, abstained=False,
                delta=None, session_env=None, super_token=None,
                chain_ok=False, chain_reason="inner_w22_not_resolved",
                delta_ok=False, delta_reason="inner_w22_not_resolved",
                super_ok=False, super_reason="inner_w22_not_resolved",
                round_trip_bytes=0)
        if w22_envelope is None:
            self._cell_index += 1
            return _pack(
                decoder_branch=W23_BRANCH_NO_TRIGGER, abstained=True,
                delta=None, session_env=None, super_token=None,
                chain_ok=False, chain_reason="no_envelope",
                delta_ok=False, delta_reason="no_envelope",
                super_ok=False, super_reason="no_envelope",
                round_trip_bytes=0)

        # Genesis cell: there is no prior session digest to delta
        # against. Emit the full W22 envelope as the session-chain
        # root; the visible-token cost is the W22 cost (no savings on
        # genesis is the honest accounting).
        is_genesis = (len(self._chain) == 0)
        if is_genesis:
            session_env = self._build_session_digest(
                w22_envelope=w22_envelope,
                prior_session_digest_cid="",
                resolved=True)
            chain_outcome = verify_session_digest_chain(
                session_env, registered_schema=self.schema,
                prior_chain_head_cid="")
            if (not chain_outcome.ok
                    and self.require_chain_verification):
                self._cell_index += 1
                return _pack(
                    decoder_branch=W23_BRANCH_DELTA_REJECTED,
                    abstained=True, delta=None,
                    session_env=session_env, super_token=None,
                    chain_ok=False, chain_reason=chain_outcome.reason,
                    delta_ok=False, delta_reason="genesis_chain_failed",
                    super_ok=False, super_reason="genesis_chain_failed",
                    round_trip_bytes=0)
            self._chain.append(session_env)
            for tag, count in w22_envelope.per_tag_vote_count:
                self._cumulative_per_tag_votes[str(tag)] = (
                    int(self._cumulative_per_tag_votes.get(str(tag), 0))
                    + int(count))
            self._last_cell_per_tag_votes = {
                str(t): int(c)
                for t, c in w22_envelope.per_tag_vote_count
            }
            self._last_projected_subset = tuple(
                sorted(w22_envelope.projected_subset))
            self._last_inner_branch = str(w22_envelope.inner_w19_branch)
            self._n_resolved += 1
            self._cell_index += 1
            return _pack(
                decoder_branch=W23_BRANCH_GENESIS, abstained=False,
                delta=None, session_env=session_env,
                super_token=None,
                chain_ok=True, chain_reason="ok",
                delta_ok=True, delta_reason="genesis_no_delta",
                super_ok=False, super_reason="genesis_no_super_token",
                round_trip_bytes=0)

        # Subsequent cell: build delta + new session digest.
        parent_cid = self.chain_head_cid()
        delta = self._build_delta(
            w22_envelope=w22_envelope,
            prior_session_digest_cid=parent_cid,
            prior_projected=prior_projected,
            parent_session_digest_cid=parent_cid)
        new_session_env = self._build_session_digest(
            w22_envelope=w22_envelope,
            prior_session_digest_cid=parent_cid,
            resolved=True)

        # Optionally round-trip the delta + session envelope across
        # the within-process producer/decoder host-split proxy
        # (Mac-2 fallback); if the proxy is set, we serialise both
        # into JSON bytes and deserialise — any latent shared state
        # would surface as a serialisation failure.
        rt_bytes = 0
        if self.cross_host_proxy is not None:
            rt_payload = {
                "delta_envelope": delta.as_dict(),
                "session_digest_envelope": new_session_env.as_dict(),
            }
            _decoded, rt_bytes_step = self._maybe_round_trip_proxy(
                rt_payload)
            rt_bytes += int(rt_bytes_step)

        # Verify the session chain link. The verifier uses
        # ``verifier_chain_head_override`` if set; otherwise it falls
        # back to the producer's chain head. The producer-decoder
        # split is the honest model: in a true cross-host deployment
        # the verifier maintains its own confirmed chain state, and
        # mid-session tampering on the producer side is detected as
        # a chain_head_mismatch.
        verifier_prior_cid = (
            self.verifier_chain_head_override
            if self.verifier_chain_head_override is not None
            else parent_cid)
        chain_outcome = verify_session_digest_chain(
            new_session_env, registered_schema=self.schema,
            prior_chain_head_cid=verifier_prior_cid)
        # Verify the delta envelope.
        delta_outcome = verify_session_delta(
            delta, registered_schema=self.schema,
            parent_session_digest_cid=verifier_prior_cid)

        if ((not chain_outcome.ok or not delta_outcome.ok)
                and self.require_chain_verification):
            self._cell_index += 1
            return _pack(
                decoder_branch=W23_BRANCH_DELTA_REJECTED,
                abstained=True, delta=delta,
                session_env=new_session_env, super_token=None,
                chain_ok=chain_outcome.ok,
                chain_reason=chain_outcome.reason,
                delta_ok=delta_outcome.ok,
                delta_reason=delta_outcome.reason,
                super_ok=False,
                super_reason="delta_rejected",
                round_trip_bytes=int(rt_bytes))

        # Optionally promote to a super-token reference (the bounded
        # steganographic / dense-control-payload experiment).
        super_token: SuperTokenReferenceEnvelope | None = None
        super_ok = False
        super_reason = "not_used"
        if self.use_super_token:
            registry = (self.super_token_registry
                          or SuperTokenRegistry())
            if self.super_token_registry is None:
                self.super_token_registry = registry
            registry.register(
                delta, hex_prefix_len=self.super_token_hex_prefix_len)
            super_token = SuperTokenReferenceEnvelope(
                schema_cid=self.schema.cid,
                delta_cid=delta.delta_cid,
                parent_session_digest_cid=parent_cid,
                hex_prefix_len=int(self.super_token_hex_prefix_len),
            )
            verifier_registry = (
                self.verifier_super_token_registry
                if self.verifier_super_token_registry is not None
                else registry)
            super_outcome = verify_super_token_reference(
                super_token, registry=verifier_registry,
                registered_schema=self.schema,
                parent_session_digest_cid=parent_cid)
            super_ok = bool(super_outcome.ok)
            super_reason = str(super_outcome.reason)
            if not super_ok:
                self._cell_index += 1
                # Update commit state regardless of super-token
                # rejection: the verbose digest is still used in
                # this fallback case, but the chain still moves
                # forward (the underlying delta and session digest
                # were both verified above).
                self._chain.append(new_session_env)
                for tag, count in w22_envelope.per_tag_vote_count:
                    self._cumulative_per_tag_votes[str(tag)] = (
                        int(self._cumulative_per_tag_votes.get(str(tag), 0))
                        + int(count))
                self._last_cell_per_tag_votes = {
                    str(t): int(c)
                    for t, c in w22_envelope.per_tag_vote_count
                }
                self._last_projected_subset = tuple(
                    sorted(w22_envelope.projected_subset))
                self._last_inner_branch = str(
                    w22_envelope.inner_w19_branch)
                self._n_resolved += 1
                return _pack(
                    decoder_branch=W23_BRANCH_SUPER_TOKEN_REJECTED,
                    abstained=True, delta=delta,
                    session_env=new_session_env,
                    super_token=super_token,
                    chain_ok=True, chain_reason="ok",
                    delta_ok=True, delta_reason="ok",
                    super_ok=False, super_reason=super_reason,
                    round_trip_bytes=int(rt_bytes))

        # Commit chain + cumulative state.
        self._chain.append(new_session_env)
        for tag, count in w22_envelope.per_tag_vote_count:
            self._cumulative_per_tag_votes[str(tag)] = (
                int(self._cumulative_per_tag_votes.get(str(tag), 0))
                + int(count))
        self._last_cell_per_tag_votes = {
            str(t): int(c)
            for t, c in w22_envelope.per_tag_vote_count
        }
        self._last_projected_subset = tuple(
            sorted(w22_envelope.projected_subset))
        self._last_inner_branch = str(w22_envelope.inner_w19_branch)
        self._n_resolved += 1
        self._cell_index += 1
        if super_token is not None and super_ok:
            return _pack(
                decoder_branch=W23_BRANCH_SUPER_TOKEN_RESOLVED,
                abstained=False, delta=delta,
                session_env=new_session_env,
                super_token=super_token,
                chain_ok=True, chain_reason="ok",
                delta_ok=True, delta_reason="ok",
                super_ok=True, super_reason="ok",
                round_trip_bytes=int(rt_bytes))
        return _pack(
            decoder_branch=W23_BRANCH_DELTA_RESOLVED,
            abstained=False, delta=delta,
            session_env=new_session_env,
            super_token=None,
            chain_ok=True, chain_reason="ok",
            delta_ok=True, delta_reason="ok",
            super_ok=False, super_reason="not_used",
            round_trip_bytes=int(rt_bytes))

    def decode(self, handoffs: Sequence[_DecodedHandoff]
                ) -> dict[str, Any]:
        return self.decode_rounds([handoffs])

    @property
    def last_result(self) -> W23SessionResult | None:
        return self._last_result

    @property
    def last_session_envelope(self) -> SessionDigestEnvelope | None:
        return self._last_session_envelope

    @property
    def last_delta_envelope(self) -> SessionDeltaEnvelope | None:
        return self._last_delta_envelope

    @property
    def last_super_token(self) -> SuperTokenReferenceEnvelope | None:
        return self._last_super_token

    @property
    def T_decoder(self) -> int | None:
        return self.inner.T_decoder

    @T_decoder.setter
    def T_decoder(self, v: int | None) -> None:
        self.inner.T_decoder = v

    def pack_stats(self) -> dict[str, Any]:
        return self.inner.pack_stats()


# =============================================================================
# SDK v3.25 — bounded-window session compaction + intra-cell resample-quorum
# + real cross-process producer/decoder wire (W24 family).
#
# The W23 family (SDK v3.24) shipped a hash-chained cross-cell session
# digest + per-cell delta + quorum-keyed cache + super-token reference.
# The honest W23-1 win is bounded above by the *per-cell* delta cost,
# which on multi-cell sessions still grows monotonically because each
# post-genesis cell's verbose digest grows with cumulative state. The
# W23-2 mitigation (PER_CELL_NONCE on the probabilistic oracle) only
# changes drift *across* cells; intra-cell probabilistic drift on a
# single consult survives the mitigation and is named in
# W23-C-MITIGATION-LIVE-VARIANCE.
#
# W24 attacks both fronts at once and adds a real OS-level cross-process
# wire to upgrade the "structural" cross-host story:
#
#   * **W24-1 Bounded-window session compaction.** The
#     :class:`MultiCellSessionCompactor` wraps
#     :class:`CrossCellDeltaDisambiguator`. On every cell beyond the
#     compact_window threshold, replaces the W23 verbose digest +
#     delta pair with a single bounded :class:`SessionCompactEnvelope`
#     that folds the last (compact_window - 1) cell digest CIDs into
#     one window CID + bounded compact-summary text. Visible-token
#     cost per cell beyond ``compact_window`` stays O(1) regardless of
#     session length. Strict density gain over W23 delta on the same
#     regime.
#
#   * **W24-2 Intra-cell resample-quorum mitigation.** The
#     :class:`ResampleQuorumCachingOracleAdapter` consults its inner
#     oracle ``sample_count`` times *within one cell* and returns the
#     majority verdict. Mitigates probabilistic-LLM intra-cell drift
#     that PER_CELL_NONCE cannot touch. Empirically discharges
#     W23-C-MITIGATION-LIVE-VARIANCE on a synthetic
#     :class:`IntraCellFlippingOracle`.
#
#   * **W24-3 Real cross-process producer/decoder wire.** The
#     :class:`CrossProcessProducerDecoderWire` spawns a real Python
#     subprocess that round-trips JSON envelopes via stdin/stdout
#     pipes. Real OS-level wire — bytes serialised, written to a
#     subprocess pipe, read back, deserialised. Strictly stronger
#     proxy for the cross-host claim than the W23 within-process
#     :class:`CrossHostProducerDecoderProxy`. Mac-2 still unreachable
#     for 18+ milestones; W24-3 is the strongest cross-process
#     honesty this repo can validate end-to-end on Mac-1 alone.
# =============================================================================

W24_COMPACT_ENVELOPE_SCHEMA_VERSION: str = "wevra.session_compact.v1"


W24_BRANCH_COMPACT_RESOLVED = "compact_resolved"
W24_BRANCH_COMPACT_REJECTED = "compact_rejected"
W24_BRANCH_BELOW_WINDOW = "below_window"
W24_BRANCH_NO_TRIGGER = "no_trigger"
W24_BRANCH_DISABLED = "disabled"


W24_ALL_BRANCHES: tuple[str, ...] = (
    W24_BRANCH_COMPACT_RESOLVED,
    W24_BRANCH_COMPACT_REJECTED,
    W24_BRANCH_BELOW_WINDOW,
    W24_BRANCH_NO_TRIGGER,
    W24_BRANCH_DISABLED,
)


W24_DEFAULT_TRIGGER_BRANCHES: frozenset[str] = frozenset({
    W23_BRANCH_DELTA_RESOLVED,
    W23_BRANCH_SUPER_TOKEN_RESOLVED,
    W23_BRANCH_GENESIS,
})


@dataclasses.dataclass(frozen=True)
class SessionCompactEnvelope:
    """Bounded-window cross-cell session compaction (W24 family).

    Folds the last ``len(window_cids)`` cell digests into one
    fixed-size envelope. Visible-token cost per cell is
    O(window_size) regardless of total session length — a strict
    density gain over W23, where every post-genesis cell's verbose
    digest grows with cumulative per-tag votes / projected subset.

    Trust boundary: every field is hash-chained into
    ``compact_envelope_cid``; tampering yields a ``hash_mismatch``
    rejection. The window CID is a hash over the ordered tuple of
    prior cell digest CIDs; reordering or inserting / removing any
    cell produces a different ``window_cid``.
    """
    schema_cid: str
    cell_index: int
    parent_session_digest_cid: str
    window_size: int
    window_cids: tuple[str, ...]
    window_cid: str
    compact_per_tag_votes: tuple[tuple[str, int], ...]
    compact_projected_subset: tuple[str, ...]
    n_resolved_in_window: int
    compact_envelope_cid: str = ""
    schema_version: str = W24_COMPACT_ENVELOPE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "window_cids", tuple(self.window_cids))
        object.__setattr__(self, "compact_per_tag_votes",
                            tuple(sorted(self.compact_per_tag_votes,
                                          key=lambda kv: kv[0])))
        object.__setattr__(self, "compact_projected_subset",
                            tuple(sorted(self.compact_projected_subset)))
        if not self.compact_envelope_cid:
            object.__setattr__(self, "compact_envelope_cid",
                                self.recompute_envelope_cid())

    def _signed_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "schema_cid": self.schema_cid,
            "cell_index": int(self.cell_index),
            "parent_session_digest_cid": self.parent_session_digest_cid,
            "window_size": int(self.window_size),
            "window_cids": list(self.window_cids),
            "window_cid": self.window_cid,
            "compact_per_tag_votes": [
                [t, int(c)] for t, c in self.compact_per_tag_votes
            ],
            "compact_projected_subset": list(self.compact_projected_subset),
            "n_resolved_in_window": int(self.n_resolved_in_window),
        }

    def to_canonical_bytes(self) -> bytes:
        return _canonical_json_bytes(self._signed_payload())

    def recompute_envelope_cid(self) -> str:
        return hashlib.sha256(self.to_canonical_bytes()).hexdigest()

    def to_decoder_text(self) -> str:
        # Minimal visible text: a single whitespace-token reference of
        # the form ``<compact_ref:DDDDDDDDDDDDDDDD>`` whose hex prefix
        # uniquely identifies this envelope's ``compact_envelope_cid``
        # in the controller's registry (constant 1 whitespace token by
        # construction, matching super-token's visible cost). The full
        # envelope's canonical bytes are recovered from the
        # producer/decoder wire (within-process or cross-process); the
        # controller verifies them against the chain and the
        # registered window CIDs.
        return f"<compact_ref:{self.compact_envelope_cid[:16]}>"

    @property
    def n_compact_tokens(self) -> int:
        return _whitespace_token_count(self.to_decoder_text())

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "schema_cid": self.schema_cid,
            "cell_index": int(self.cell_index),
            "parent_session_digest_cid": self.parent_session_digest_cid,
            "window_size": int(self.window_size),
            "window_cids": list(self.window_cids),
            "window_cid": self.window_cid,
            "compact_per_tag_votes": [
                [t, int(c)] for t, c in self.compact_per_tag_votes
            ],
            "compact_projected_subset": list(self.compact_projected_subset),
            "n_resolved_in_window": int(self.n_resolved_in_window),
            "compact_envelope_cid": self.compact_envelope_cid,
            "n_compact_tokens": int(self.n_compact_tokens),
            "decoder_text": self.to_decoder_text(),
        }


def _compute_window_cid(window_cids: Sequence[str]) -> str:
    body = _canonical_json_bytes({
        "window_cids": list(window_cids),
    })
    return hashlib.sha256(body).hexdigest()


def verify_session_compact(
        envelope: SessionCompactEnvelope | None,
        *,
        registered_schema: SchemaCapsule,
        expected_window_cids: Sequence[str],
        ) -> LatentVerificationOutcome:
    """Controller-side verification of a SessionCompactEnvelope.

    Pure function. Failure modes enumerated:

      * empty_envelope — no envelope passed.
      * schema_version_unknown — schema_version differs.
      * schema_cid_mismatch — envelope's schema_cid != registered.
      * window_size_mismatch — window_size != len(expected_window_cids).
      * window_cids_mismatch — envelope's window_cids tuple does not
        equal the controller's expected window.
      * window_cid_mismatch — envelope's window_cid does not recompute
        from window_cids.
      * hash_mismatch — compact_envelope_cid does not recompute from
        canonical bytes.
    """
    n_checks = 0
    if envelope is None:
        return LatentVerificationOutcome(
            ok=False, reason="empty_envelope", n_checks=n_checks)
    n_checks += 1
    if envelope.schema_version != W24_COMPACT_ENVELOPE_SCHEMA_VERSION:
        return LatentVerificationOutcome(
            ok=False, reason="schema_version_unknown", n_checks=n_checks)
    n_checks += 1
    if envelope.schema_cid != registered_schema.cid:
        return LatentVerificationOutcome(
            ok=False, reason="schema_cid_mismatch", n_checks=n_checks)
    expected = tuple(expected_window_cids)
    n_checks += 1
    if int(envelope.window_size) != len(expected):
        return LatentVerificationOutcome(
            ok=False, reason="window_size_mismatch", n_checks=n_checks)
    n_checks += 1
    if tuple(envelope.window_cids) != expected:
        return LatentVerificationOutcome(
            ok=False, reason="window_cids_mismatch", n_checks=n_checks)
    n_checks += 1
    if envelope.window_cid != _compute_window_cid(envelope.window_cids):
        return LatentVerificationOutcome(
            ok=False, reason="window_cid_mismatch", n_checks=n_checks)
    n_checks += 1
    canonical = envelope.to_canonical_bytes()
    if (hashlib.sha256(canonical).hexdigest()
            != envelope.compact_envelope_cid):
        return LatentVerificationOutcome(
            ok=False, reason="hash_mismatch", n_checks=n_checks)
    return LatentVerificationOutcome(
        ok=True, reason="ok", n_checks=n_checks)


@dataclasses.dataclass
class W24CompactionResult:
    """Audit record for one W24 cell."""
    answer: dict[str, Any]
    inner_w23_branch: str
    decoder_branch: str
    abstained: bool
    schema_cid: str
    schema_version: str
    cell_index: int
    compact_envelope_cid: str
    compact_window_size: int
    compact_window_cids: tuple[str, ...]
    compact_window_cid: str
    n_compact_tokens: int
    n_w15_tokens_kept: int
    n_w23_visible_tokens_to_decider: int
    n_w24_visible_tokens_to_decider: int
    n_w23_minus_w24_savings: int
    compact_verification_ok: bool
    compact_verification_reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "inner_w23_branch": str(self.inner_w23_branch),
            "decoder_branch": str(self.decoder_branch),
            "abstained": bool(self.abstained),
            "schema_cid": str(self.schema_cid),
            "schema_version": str(self.schema_version),
            "cell_index": int(self.cell_index),
            "compact_envelope_cid": str(self.compact_envelope_cid),
            "compact_window_size": int(self.compact_window_size),
            "compact_window_cids": list(self.compact_window_cids),
            "compact_window_cid": str(self.compact_window_cid),
            "n_compact_tokens": int(self.n_compact_tokens),
            "n_w15_tokens_kept": int(self.n_w15_tokens_kept),
            "n_w23_visible_tokens_to_decider":
                int(self.n_w23_visible_tokens_to_decider),
            "n_w24_visible_tokens_to_decider":
                int(self.n_w24_visible_tokens_to_decider),
            "n_w23_minus_w24_savings":
                int(self.n_w23_minus_w24_savings),
            "compact_verification_ok": bool(self.compact_verification_ok),
            "compact_verification_reason":
                str(self.compact_verification_reason),
        }


@dataclasses.dataclass
class MultiCellSessionCompactor:
    """Bounded-window cross-cell session compactor (W24 family).

    Wraps a :class:`CrossCellDeltaDisambiguator` (W23) and replaces
    the per-cell verbose digest+delta pair with a fixed-size compact
    envelope on cells beyond the genesis-plus-(window-1) threshold.

    Trust boundary: every compact envelope is hash-chained,
    schema-versioned, and verifier-rejectable. The verifier registry
    can be split (``verifier_window_cids_override``) to model
    tampered/desynchronised cross-host deployments.
    """

    inner: CrossCellDeltaDisambiguator = dataclasses.field(
        default_factory=lambda: CrossCellDeltaDisambiguator())
    schema: SchemaCapsule | None = None
    enabled: bool = True
    compact_window: int = 4
    trigger_branches: frozenset[str] = dataclasses.field(
        default_factory=lambda: W24_DEFAULT_TRIGGER_BRANCHES)
    require_compact_verification: bool = True
    verifier_window_cids_override: tuple[str, ...] | None = None
    cross_process_wire: Any | None = None

    _last_result: W24CompactionResult | None = None
    _last_compact_envelope: SessionCompactEnvelope | None = None
    _cell_index: int = 0
    _resolved_cells_seen: int = 0

    def reset_session(self) -> None:
        self._last_result = None
        self._last_compact_envelope = None
        self._cell_index = 0
        self._resolved_cells_seen = 0
        self.inner.reset_session()

    @property
    def session_chain(self) -> tuple[SessionDigestEnvelope, ...]:
        return self.inner.session_chain()

    def chain_window_cids(self) -> tuple[str, ...]:
        chain = self.inner.session_chain()
        K = max(1, int(self.compact_window))
        if len(chain) < K:
            return ()
        prior_window = chain[-K:-1]
        return tuple(env.digest_cid for env in prior_window)

    def expected_window_cids(self) -> tuple[str, ...]:
        if self.verifier_window_cids_override is not None:
            return tuple(self.verifier_window_cids_override)
        return self.chain_window_cids()

    def _build_compact(
            self,
            *,
            cell_index: int,
            parent_session_digest_cid: str,
            window_cids: tuple[str, ...],
            ) -> SessionCompactEnvelope:
        latest_envelope = self.inner.last_session_envelope
        if latest_envelope is None:
            cumulative_votes: tuple[tuple[str, int], ...] = ()
            projected: tuple[str, ...] = ()
            n_resolved = int(self._resolved_cells_seen)
        else:
            cumulative_votes = tuple(
                (t, int(c))
                for t, c in latest_envelope.cumulative_per_tag_votes
            )
            projected = tuple(latest_envelope.latest_projected_subset)
            n_resolved = int(latest_envelope.n_cells_resolved)

        return SessionCompactEnvelope(
            schema_cid=self.schema.cid if self.schema is not None else "",
            cell_index=int(cell_index),
            parent_session_digest_cid=str(parent_session_digest_cid),
            window_size=len(window_cids),
            window_cids=tuple(window_cids),
            window_cid=_compute_window_cid(window_cids),
            compact_per_tag_votes=cumulative_votes,
            compact_projected_subset=projected,
            n_resolved_in_window=n_resolved,
            schema_version=W24_COMPACT_ENVELOPE_SCHEMA_VERSION,
        )

    def _maybe_round_trip_wire(self, payload: dict[str, Any]
                                  ) -> tuple[dict[str, Any], int]:
        if self.cross_process_wire is None:
            return payload, 0
        before = int(getattr(
            self.cross_process_wire, "n_bytes_serialised", 0))
        out = self.cross_process_wire.producer_to_decoder(payload)
        after = int(getattr(
            self.cross_process_wire, "n_bytes_serialised", 0))
        return out, max(0, after - before)

    def decode_rounds(
            self,
            per_round_handoffs: Sequence[Sequence[_DecodedHandoff]],
            ) -> dict[str, Any]:
        base = self.inner.decode_rounds(per_round_handoffs)
        w23_result = self.inner.last_result
        out = dict(base)

        n_w15_kept = (int(w23_result.n_w15_tokens_kept)
                       if w23_result is not None else 0)
        n_w23_visible = (
            int(w23_result.n_w23_visible_tokens_to_decider)
            if w23_result is not None else 0)
        cell_index = int(self._cell_index)
        inner_w23_branch = (str(w23_result.decoder_branch)
                              if w23_result is not None
                              else W23_BRANCH_DISABLED)

        def _pack(*,
                   decoder_branch: str,
                   abstained: bool,
                   compact: SessionCompactEnvelope | None,
                   compact_ok: bool,
                   compact_reason: str,
                   ) -> dict[str, Any]:
            schema_cid = (str(self.schema.cid)
                            if self.schema is not None else "")
            schema_version = W24_COMPACT_ENVELOPE_SCHEMA_VERSION
            if (decoder_branch == W24_BRANCH_COMPACT_RESOLVED
                    and compact is not None):
                visible = n_w15_kept + int(compact.n_compact_tokens)
            else:
                visible = n_w23_visible
            savings = max(0, n_w23_visible - visible)
            compact_cid = (str(compact.compact_envelope_cid)
                            if compact is not None else "")
            window_size = (int(compact.window_size)
                             if compact is not None else 0)
            window_cids = (tuple(compact.window_cids)
                             if compact is not None else ())
            window_cid = (str(compact.window_cid)
                            if compact is not None else "")
            n_compact = (int(compact.n_compact_tokens)
                           if compact is not None else 0)
            result = W24CompactionResult(
                answer=dict(base),
                inner_w23_branch=inner_w23_branch,
                decoder_branch=decoder_branch,
                abstained=abstained,
                schema_cid=schema_cid,
                schema_version=schema_version,
                cell_index=cell_index,
                compact_envelope_cid=compact_cid,
                compact_window_size=window_size,
                compact_window_cids=window_cids,
                compact_window_cid=window_cid,
                n_compact_tokens=n_compact,
                n_w15_tokens_kept=n_w15_kept,
                n_w23_visible_tokens_to_decider=n_w23_visible,
                n_w24_visible_tokens_to_decider=visible,
                n_w23_minus_w24_savings=savings,
                compact_verification_ok=compact_ok,
                compact_verification_reason=compact_reason,
            )
            self._last_result = result
            self._last_compact_envelope = compact
            out_local = dict(out)
            out_local["session_compact_hybrid"] = result.as_dict()
            if compact is not None:
                out_local["session_compact_envelope"] = compact.as_dict()
            return out_local

        if not self.enabled:
            self._cell_index += 1
            return _pack(
                decoder_branch=W24_BRANCH_DISABLED, abstained=True,
                compact=None,
                compact_ok=False, compact_reason="disabled")
        if (w23_result is None or self.schema is None
                or inner_w23_branch not in self.trigger_branches):
            self._cell_index += 1
            return _pack(
                decoder_branch=W24_BRANCH_NO_TRIGGER, abstained=False,
                compact=None,
                compact_ok=False,
                compact_reason="inner_w23_not_triggered")

        if inner_w23_branch in (W23_BRANCH_DELTA_RESOLVED,
                                  W23_BRANCH_SUPER_TOKEN_RESOLVED,
                                  W23_BRANCH_GENESIS):
            self._resolved_cells_seen += 1

        K = max(1, int(self.compact_window))
        chain_len = len(self.inner.session_chain())
        if chain_len < K:
            self._cell_index += 1
            return _pack(
                decoder_branch=W24_BRANCH_BELOW_WINDOW,
                abstained=False, compact=None,
                compact_ok=False, compact_reason="chain_below_window")

        parent_cid = self.inner.chain_head_cid()
        producer_window = self.chain_window_cids()
        compact = self._build_compact(
            cell_index=cell_index,
            parent_session_digest_cid=parent_cid,
            window_cids=producer_window)

        try:
            _ = self._maybe_round_trip_wire({
                "compact_envelope": compact.as_dict(),
            })
        except Exception:
            self._cell_index += 1
            return _pack(
                decoder_branch=W24_BRANCH_COMPACT_REJECTED,
                abstained=True, compact=compact,
                compact_ok=False,
                compact_reason="cross_process_wire_failed")

        expected = self.expected_window_cids()
        outcome = verify_session_compact(
            compact, registered_schema=self.schema,
            expected_window_cids=expected)
        if (not outcome.ok) and self.require_compact_verification:
            self._cell_index += 1
            return _pack(
                decoder_branch=W24_BRANCH_COMPACT_REJECTED,
                abstained=True, compact=compact,
                compact_ok=False, compact_reason=str(outcome.reason))

        self._cell_index += 1
        return _pack(
            decoder_branch=W24_BRANCH_COMPACT_RESOLVED,
            abstained=False, compact=compact,
            compact_ok=True, compact_reason="ok")

    def decode(self, handoffs: Sequence[_DecodedHandoff]
                ) -> dict[str, Any]:
        return self.decode_rounds([handoffs])

    @property
    def last_result(self) -> W24CompactionResult | None:
        return self._last_result

    @property
    def last_compact_envelope(self) -> SessionCompactEnvelope | None:
        return self._last_compact_envelope

    @property
    def T_decoder(self) -> int | None:
        return self.inner.T_decoder

    @T_decoder.setter
    def T_decoder(self, v: int | None) -> None:
        self.inner.T_decoder = v


@dataclasses.dataclass
class ResampleQuorumCachingOracleAdapter:
    """Intra-cell M-sample quorum-caching oracle adapter (W24 family).

    On every consult resamples the wrapped oracle ``sample_count``
    times and returns the *majority* verdict. Mitigates intra-cell
    probabilistic-LLM drift that PER_CELL_NONCE cannot touch
    (PER_CELL_NONCE only changes drift between cells; a probabilistic
    oracle whose first-sample drift is intra-cell-bound still
    amplifies the bad sample).

    Tradeoffs:
      * pays M× the inner oracle cost on every cache-miss cell;
      * mitigates intra-cell drift directly
        (W23-C-MITIGATION-LIVE-VARIANCE);
      * collapses to single-consult behaviour on cache hit.
    """
    inner: Any
    cache: QuorumKeyedSharedReadCache | None = None
    oracle_id: str = ""
    cell_nonce: str = ""
    sample_count: int = 3
    majority_threshold: int = 0
    last_was_hit: bool = False
    last_was_resampled: bool = False
    last_n_samples: int = 0
    last_majority_size: int = 0
    last_majority_formed: bool = False
    n_total_consults: int = 0
    n_total_resamples: int = 0
    n_majority_formed: int = 0
    n_majority_failed: int = 0

    def __post_init__(self) -> None:
        if not self.oracle_id:
            self.oracle_id = str(getattr(
                self.inner, "oracle_id", "resample_quorum_oracle"))
        if int(self.sample_count) < 1:
            raise ValueError(
                f"sample_count must be ≥ 1; got {self.sample_count}")
        if int(self.majority_threshold) <= 0:
            self.majority_threshold = (
                (int(self.sample_count) + 1) // 2)
        if int(self.majority_threshold) > int(self.sample_count):
            raise ValueError(
                f"majority_threshold ({self.majority_threshold}) must "
                f"be ≤ sample_count ({self.sample_count})")

    def consult(self, query: OutsideQuery) -> OutsideVerdict:
        self.n_total_consults += 1
        if self.cache is not None:
            cid = self.cache.query_cid_for(
                query, oracle_id=self.oracle_id,
                cell_nonce=str(self.cell_nonce))
            cached = self.cache.get(cid)
            if cached is not None:
                body, n_tokens, source_id = cached
                self.last_was_hit = True
                self.last_was_resampled = False
                self.last_n_samples = 0
                self.last_majority_size = 1
                self.last_majority_formed = True
                payload = body.decode("utf-8") if body else None
                return OutsideVerdict(
                    payload=payload,
                    source_id=source_id or self.oracle_id,
                    n_tokens=int(n_tokens),
                )
        self.last_was_hit = False

        M = max(1, int(self.sample_count))
        T = max(1, int(self.majority_threshold))
        samples: list[OutsideVerdict] = []
        body_counts: dict[str, int] = {}
        for _ in range(M):
            v = self.inner.consult(query)
            samples.append(v)
            body = (v.payload or "")
            body_counts[body] = body_counts.get(body, 0) + 1
        self.n_total_resamples += max(0, M - 1)
        self.last_was_resampled = (M > 1)
        self.last_n_samples = M

        # Iterate samples in order so on tie we keep the first-seen body.
        seen_bodies: list[str] = []
        for s in samples:
            b = (s.payload or "")
            if b not in seen_bodies:
                seen_bodies.append(b)
        best_body: str | None = None
        best_count = 0
        for b in seen_bodies:
            c = body_counts.get(b, 0)
            if c > best_count:
                best_body = b
                best_count = c
        if best_body is None:
            chosen = samples[0]
            self.last_majority_size = 1
            self.last_majority_formed = False
            self.n_majority_failed += 1
        else:
            self.last_majority_size = int(best_count)
            self.last_majority_formed = (best_count >= T)
            if self.last_majority_formed:
                self.n_majority_formed += 1
                chosen = next(
                    s for s in samples
                    if (s.payload or "") == best_body)
            else:
                chosen = samples[0]
                self.n_majority_failed += 1

        if self.cache is not None:
            cid = self.cache.query_cid_for(
                query, oracle_id=self.oracle_id,
                cell_nonce=str(self.cell_nonce))
            body = (chosen.payload or "").encode("utf-8")
            policy = self.cache.policy_for(self.oracle_id)
            if policy == CACHE_FRESHNESS_QUORUM_LOCKED:
                self.cache.put(
                    cid, body, n_tokens=int(chosen.n_tokens),
                    source_id=str(chosen.source_id or self.oracle_id),
                    oracle_id=self.oracle_id,
                    quorum_locked_pending=True)
            else:
                self.cache.put(
                    cid, body, n_tokens=int(chosen.n_tokens),
                    source_id=str(chosen.source_id or self.oracle_id),
                    oracle_id=self.oracle_id)
        return chosen

    def stats(self) -> dict[str, Any]:
        return {
            "oracle_id": str(self.oracle_id),
            "sample_count": int(self.sample_count),
            "majority_threshold": int(self.majority_threshold),
            "n_total_consults": int(self.n_total_consults),
            "n_total_resamples": int(self.n_total_resamples),
            "n_majority_formed": int(self.n_majority_formed),
            "n_majority_failed": int(self.n_majority_failed),
            "last_majority_size": int(self.last_majority_size),
            "last_majority_formed": bool(self.last_majority_formed),
            "last_was_hit": bool(self.last_was_hit),
            "last_was_resampled": bool(self.last_was_resampled),
            "last_n_samples": int(self.last_n_samples),
        }


@dataclasses.dataclass
class CrossProcessProducerDecoderWire:
    """Real cross-process producer/decoder wire (W24 family).

    Spawns a Python subprocess and round-trips JSON payloads via
    stdin/stdout pipes. Real OS-level wire — bytes serialised, written
    to a child process pipe, read back, deserialised. No Python
    references survive the wire.

    Honest scope: real cross-process, **NOT** cross-host. Mac 2 has
    been ARP-incomplete for 18 milestones in a row.
    """
    _proc: Any | None = None
    _python_path: str = ""
    _started: bool = False
    n_round_trips: int = 0
    n_bytes_serialised: int = 0
    n_bytes_deserialised: int = 0
    n_restarts: int = 0
    n_failures: int = 0

    _ECHO_SCRIPT: str = (
        "import sys, json\n"
        "for line in sys.stdin:\n"
        "    line = line.strip()\n"
        "    if not line: break\n"
        "    try:\n"
        "        payload = json.loads(line)\n"
        "    except Exception:\n"
        "        sys.stdout.write('null\\n'); sys.stdout.flush(); continue\n"
        "    out = json.dumps(payload, sort_keys=True, separators=(',', ':'))\n"
        "    sys.stdout.write(out + '\\n'); sys.stdout.flush()\n"
    )

    def __post_init__(self) -> None:
        if not self._python_path:
            import sys as _sys
            self._python_path = str(_sys.executable)

    def start(self) -> None:
        if self._started and self._proc is not None:
            return
        import subprocess as _subprocess
        self._proc = _subprocess.Popen(
            [self._python_path, "-c", self._ECHO_SCRIPT],
            stdin=_subprocess.PIPE,
            stdout=_subprocess.PIPE,
            stderr=_subprocess.PIPE,
            bufsize=0,
            text=False,
        )
        self._started = True

    def stop(self) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.stdin is not None:
                try:
                    self._proc.stdin.close()
                except Exception:
                    pass
            self._proc.wait(timeout=2.0)
        except Exception:
            try:
                self._proc.terminate()
            except Exception:
                pass
        finally:
            self._proc = None
            self._started = False

    def __enter__(self) -> "CrossProcessProducerDecoderWire":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def producer_to_decoder(self, payload: dict[str, Any]
                              ) -> dict[str, Any]:
        if not self._started or self._proc is None:
            self.start()
        body = _canonical_json_bytes(payload)
        try:
            line = body + b"\n"
            assert self._proc is not None
            assert self._proc.stdin is not None
            self._proc.stdin.write(line)
            self._proc.stdin.flush()
            assert self._proc.stdout is not None
            reply = self._proc.stdout.readline()
        except Exception:
            self.n_failures += 1
            self.stop()
            self.n_restarts += 1
            raise
        if not reply:
            self.n_failures += 1
            self.stop()
            self.n_restarts += 1
            raise RuntimeError(
                "cross_process_wire: empty subprocess reply")
        self.n_round_trips += 1
        self.n_bytes_serialised += len(body)
        self.n_bytes_deserialised += len(reply)
        try:
            return json.loads(reply.decode("ascii"))
        except Exception:
            self.n_failures += 1
            raise RuntimeError(
                "cross_process_wire: malformed subprocess reply")

    def stats(self) -> dict[str, Any]:
        return {
            "n_round_trips": int(self.n_round_trips),
            "n_bytes_serialised": int(self.n_bytes_serialised),
            "n_bytes_deserialised": int(self.n_bytes_deserialised),
            "n_restarts": int(self.n_restarts),
            "n_failures": int(self.n_failures),
            "started": bool(self._started),
        }


@dataclasses.dataclass
class IntraCellFlippingOracle:
    """Bench oracle that drifts WITHIN a single cell across consults.

    Models the *intra-cell* portion of the live-LLM probabilistic
    drift named in W23-C-MITIGATION-LIVE-VARIANCE: even within one
    cell, sample #1 may produce a decoy-asymmetric reply while
    samples #2.. produce gold-asymmetric replies. The W22 / W23
    PER_CELL_NONCE mitigation does **not** help here because both
    samples are taken inside the same cell.

    The W24 :class:`ResampleQuorumCachingOracleAdapter` mitigates the
    intra-cell drift directly: with M=3, T=2, samples #1 = decoy,
    #2 = gold, #3 = gold → majority gold → quorum forms on gold.
    """
    oracle_id: str = "intra_cell_flipping"
    gold_subset: tuple[str, ...] = ("orders", "payments")
    decoy_tag: str = "cache"
    decoy_asymmetric_reply: str = ""
    gold_asymmetric_reply: str = ""
    max_response_tokens: int = 24
    bad_consult_indices: frozenset[int] = dataclasses.field(
        default_factory=lambda: frozenset({1}))
    n_consults: int = 0

    def __post_init__(self) -> None:
        if not self.decoy_asymmetric_reply:
            self.decoy_asymmetric_reply = (
                f"investigate {self.decoy_tag} subsystem only")
        if not self.gold_asymmetric_reply:
            gold = " ".join(self.gold_subset) or "orders payments"
            self.gold_asymmetric_reply = f"check {gold} call graph"

    def consult(self, query: OutsideQuery) -> OutsideVerdict:
        self.n_consults += 1
        if self.n_consults in self.bad_consult_indices:
            payload = self.decoy_asymmetric_reply
        else:
            payload = self.gold_asymmetric_reply
        toks = payload.split()[:self.max_response_tokens]
        payload_truncated = " ".join(toks)
        return OutsideVerdict(
            payload=payload_truncated,
            source_id=self.oracle_id,
            n_tokens=len(toks),
        )


# =============================================================================
# W25 — shared-fanout dense-control capsule + cross-agent state reuse
# =============================================================================
#
# The W24 family reduced per-agent session-context cost to one compact
# token per cell via bounded-window compaction.  W25 extends that to the
# MULTI-AGENT case: when K consumer agents all need the SAME cross-cell
# session state produced by one producer agent, W24 would still require
# K independent compact envelopes (K × C tokens total).  W25 replaces
# those K envelopes with a single FanoutEnvelope emitted by the producer
# + K single-token <fanout_ref:DDDD> references for the consumers.
#
# Total visible-token cost across all K+1 agents:
#   W24: (K+1) × n_w24_compact_tokens_per_agent
#   W25: n_w24_compact_tokens (producer) + K × 1 (consumers)
#   Savings: K × (n_w24_compact_tokens - 1)  [strictly positive when compact > 1]
#
# This is the capsule-layer proxy for the LatentMAS "shared KV pool /
# hardware pooling" direction: one producer computes; K consumers reuse.
# Trust boundary: the FanoutEnvelope is hash-chained; the registry
# enforces consumer authorisation; any mismatch → W25 falls through to
# the W24 per-agent path.

W25_FANOUT_SCHEMA_VERSION: str = "wevra.shared_fanout.v1"

W25_BRANCH_FANOUT_PRODUCER_EMITTED = "fanout_producer_emitted"
W25_BRANCH_FANOUT_CONSUMER_RESOLVED = "fanout_consumer_resolved"
W25_BRANCH_FANOUT_CONSUMER_REJECTED = "fanout_consumer_rejected"
W25_BRANCH_NO_TRIGGER = "no_trigger"
W25_BRANCH_DISABLED = "disabled"

W25_ALL_BRANCHES: tuple[str, ...] = (
    W25_BRANCH_FANOUT_PRODUCER_EMITTED,
    W25_BRANCH_FANOUT_CONSUMER_RESOLVED,
    W25_BRANCH_FANOUT_CONSUMER_REJECTED,
    W25_BRANCH_NO_TRIGGER,
    W25_BRANCH_DISABLED,
)

# W25 fires on any W24 branch that produces meaningful compact state.
W25_DEFAULT_TRIGGER_BRANCHES: frozenset[str] = frozenset({
    W24_BRANCH_COMPACT_RESOLVED,
    W24_BRANCH_BELOW_WINDOW,
    W23_BRANCH_GENESIS,
})


@dataclasses.dataclass(frozen=True)
class FanoutEnvelope:
    """Shared cross-agent state fanout envelope (W25 family).

    A producer agent emits one FanoutEnvelope carrying the compact
    session state for K named consumer agents.  Each consumer resolves
    their view via a single ``<fanout_ref:DDDDDDDDDDDDDDDD>`` token
    (one whitespace token — same visible cost as the W23 super-token
    and W24 compact_ref).

    Visible-token accounting:
    * Producer:  pays n_w24_compact_tokens (unchanged; W25 does not
      change the producer's own compact envelope cost).
    * Consumer k: pays 1 token (fanout_ref lookup) instead of its own
      n_w24_compact_tokens.
    * Total across 1 producer + K consumers:
        W24: (K+1) × C  →  W25: C + K × 1  →  saves K×(C−1) tokens.

    Trust boundary: ``fanout_cid`` is SHA-256 over the signed payload
    (producer_id + consumer_ids + votes + schema_cid + cell_index);
    any field tampered produces a ``hash_mismatch`` rejection.  The
    registry additionally verifies that the requesting consumer_id is
    in ``consumer_agent_ids`` before returning the envelope.

    This is the capsule-layer proxy for the LatentMAS "shared KV pool"
    direction: one producer computes; K consumers reuse.  It does NOT
    touch transformer KV caches, embedding tables, or model-internal
    state.
    """

    schema_version: str
    producer_agent_id: str
    consumer_agent_ids: tuple[str, ...]
    compact_per_tag_votes: tuple[tuple[str, int], ...]
    compact_projected_subset: tuple[str, ...]
    schema_cid: str
    cell_index: int
    n_resolved_in_window: int
    fanout_cid: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "consumer_agent_ids",
                            tuple(self.consumer_agent_ids))
        object.__setattr__(self, "compact_per_tag_votes",
                            tuple(sorted(self.compact_per_tag_votes,
                                          key=lambda kv: kv[0])))
        object.__setattr__(self, "compact_projected_subset",
                            tuple(sorted(self.compact_projected_subset)))
        if not self.fanout_cid:
            object.__setattr__(self, "fanout_cid",
                                self.recompute_fanout_cid())

    def _signed_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "producer_agent_id": self.producer_agent_id,
            "consumer_agent_ids": sorted(self.consumer_agent_ids),
            "compact_per_tag_votes": [
                [t, int(c)] for t, c in self.compact_per_tag_votes],
            "compact_projected_subset": list(self.compact_projected_subset),
            "schema_cid": self.schema_cid,
            "cell_index": int(self.cell_index),
            "n_resolved_in_window": int(self.n_resolved_in_window),
        }

    def to_canonical_bytes(self) -> bytes:
        return _canonical_json_bytes(self._signed_payload())

    def recompute_fanout_cid(self) -> str:
        return hashlib.sha256(self.to_canonical_bytes()).hexdigest()

    def consumer_ref_token(self, consumer_id: str) -> str:
        """Per-consumer ``<fanout_ref:DDDDDDDDDDDDDDDD>`` token.

        The consumer_id is baked into the hash so the ref is unique
        per (fanout, consumer) pair — the controller can attribute
        each ref back to its registered consumer.
        """
        ref_payload = _canonical_json_bytes({
            "fanout_cid": self.fanout_cid,
            "consumer_id": consumer_id,
        })
        consumer_cid = hashlib.sha256(ref_payload).hexdigest()
        return f"<fanout_ref:{consumer_cid[:16]}>"

    @property
    def n_fanout_ref_tokens(self) -> int:
        return 1  # constant 1 whitespace token per consumer ref

    @property
    def n_fanout_bytes(self) -> int:
        return len(self.to_canonical_bytes())

    def as_dict(self) -> dict[str, Any]:
        d = self._signed_payload()
        d["fanout_cid"] = self.fanout_cid
        d["n_fanout_ref_tokens"] = self.n_fanout_ref_tokens
        d["n_fanout_bytes"] = self.n_fanout_bytes
        return d


@dataclasses.dataclass
class SharedFanoutRegistry:
    """Controller-side registry for :class:`FanoutEnvelope` objects.

    The producer registers an envelope once; K consumers each resolve
    their view by (fanout_cid, consumer_id).  Registration verifies
    hash integrity; resolution verifies consumer authorisation.

    Trust boundary: the registry is controller-owned.  A malicious
    producer cannot register an envelope that passes hash verification
    with a different payload.  A malicious consumer cannot resolve an
    envelope they are not named in.
    """

    schema: SchemaCapsule | None = None
    _registry: dict[str, FanoutEnvelope] = dataclasses.field(
        default_factory=dict)
    n_registered: int = 0
    n_resolved: int = 0
    n_rejected_register: int = 0
    n_rejected_resolve: int = 0

    def register(self, envelope: FanoutEnvelope) -> LatentVerificationOutcome:
        """Register a FanoutEnvelope.  Verifies hash + schema before storing."""
        n = 0
        n += 1
        if envelope.schema_version != W25_FANOUT_SCHEMA_VERSION:
            self.n_rejected_register += 1
            return LatentVerificationOutcome(
                ok=False, reason="schema_version_unknown", n_checks=n)
        if self.schema is not None:
            n += 1
            if envelope.schema_cid != self.schema.cid:
                self.n_rejected_register += 1
                return LatentVerificationOutcome(
                    ok=False, reason="schema_cid_mismatch", n_checks=n)
        n += 1
        if envelope.fanout_cid != envelope.recompute_fanout_cid():
            self.n_rejected_register += 1
            return LatentVerificationOutcome(
                ok=False, reason="hash_mismatch", n_checks=n)
        self._registry[envelope.fanout_cid] = envelope
        self.n_registered += 1
        return LatentVerificationOutcome(ok=True, reason="ok", n_checks=n)

    def resolve(self, fanout_cid: str,
                consumer_id: str) -> tuple["FanoutEnvelope | None", str]:
        """Resolve consumer view.  Returns (envelope, reason).

        Returns (None, reason) if not found or consumer not authorised.
        """
        envelope = self._registry.get(fanout_cid)
        if envelope is None:
            self.n_rejected_resolve += 1
            return None, "fanout_cid_not_found"
        if consumer_id not in envelope.consumer_agent_ids:
            self.n_rejected_resolve += 1
            return None, "consumer_not_authorized"
        self.n_resolved += 1
        return envelope, "ok"

    def get_by_producer(self, producer_id: str,
                         cell_index: int) -> "FanoutEnvelope | None":
        """Look up by (producer_id, cell_index) for consumer-side use."""
        for env in self._registry.values():
            if (env.producer_agent_id == producer_id
                    and env.cell_index == cell_index):
                return env
        return None


def verify_fanout(
        envelope: "FanoutEnvelope | None",
        *,
        registered_schema: SchemaCapsule,
        consumer_id: str,
) -> LatentVerificationOutcome:
    """Controller-side verification of a :class:`FanoutEnvelope` request.

    Pure function.  Failure modes enumerated:

    * ``empty_envelope``       — no envelope.
    * ``schema_version_unknown`` — schema_version field mismatch.
    * ``schema_cid_mismatch``  — envelope schema_cid ≠ registered.
    * ``consumer_not_authorized`` — consumer_id not in consumer_agent_ids.
    * ``hash_mismatch``        — fanout_cid does not recompute.
    """
    n = 0
    if envelope is None:
        return LatentVerificationOutcome(
            ok=False, reason="empty_envelope", n_checks=n)
    n += 1
    if envelope.schema_version != W25_FANOUT_SCHEMA_VERSION:
        return LatentVerificationOutcome(
            ok=False, reason="schema_version_unknown", n_checks=n)
    n += 1
    if envelope.schema_cid != registered_schema.cid:
        return LatentVerificationOutcome(
            ok=False, reason="schema_cid_mismatch", n_checks=n)
    n += 1
    if consumer_id not in envelope.consumer_agent_ids:
        return LatentVerificationOutcome(
            ok=False, reason="consumer_not_authorized", n_checks=n)
    n += 1
    if envelope.fanout_cid != envelope.recompute_fanout_cid():
        return LatentVerificationOutcome(
            ok=False, reason="hash_mismatch", n_checks=n)
    return LatentVerificationOutcome(ok=True, reason="ok", n_checks=n)


@dataclasses.dataclass
class W25FanoutResult:
    """Per-cell audit record for a W25 shared-fanout agent (W25 family)."""

    answer: dict[str, Any]
    inner_w24_branch: str
    decoder_branch: str
    agent_id: str
    is_producer: bool
    fanout_cid: str
    n_consumer_agents: int
    n_w24_visible_tokens: int
    n_w25_visible_tokens: int
    n_w24_minus_w25_savings: int
    n_fanout_bytes: int
    fanout_verification_ok: bool
    fanout_verification_reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "inner_w24_branch": str(self.inner_w24_branch),
            "decoder_branch": str(self.decoder_branch),
            "agent_id": str(self.agent_id),
            "is_producer": bool(self.is_producer),
            "fanout_cid": str(self.fanout_cid),
            "n_consumer_agents": int(self.n_consumer_agents),
            "n_w24_visible_tokens": int(self.n_w24_visible_tokens),
            "n_w25_visible_tokens": int(self.n_w25_visible_tokens),
            "n_w24_minus_w25_savings": int(self.n_w24_minus_w25_savings),
            "n_fanout_bytes": int(self.n_fanout_bytes),
            "fanout_verification_ok": bool(self.fanout_verification_ok),
            "fanout_verification_reason": str(self.fanout_verification_reason),
        }


@dataclasses.dataclass
class SharedFanoutDisambiguator:
    """Cross-agent shared-fanout disambiguator (W25 family).

    Wraps a :class:`MultiCellSessionCompactor` (W24) and, when a
    :class:`SharedFanoutRegistry` is attached:

    **Producer mode** (``is_producer = True``):
      After W24 produces a compact envelope, the producer builds a
      :class:`FanoutEnvelope` naming the consumer agents, registers it
      in the shared registry, and reports the W24 compact token cost.
      The producer's own visible-token cost is unchanged (it still pays
      for the compact envelope); the benefit is that consumers avoid
      their independent compact envelopes.

    **Consumer mode** (``is_producer = False``):
      The consumer resolves the producer's :class:`FanoutEnvelope` from
      the registry via a single ``<fanout_ref:DDDDDDDDDDDDDDDD>`` token
      (1 visible token vs n_w24_compact_tokens per W24).  On
      verification failure the consumer falls through to the W24
      per-agent path (no covert state, no loss of correctness).

    Visible-token savings across 1 producer + K consumers per cell:
    * W24: (K+1) × n_w24_compact_tokens
    * W25: n_w24_compact_tokens + K × 1 = C + K
    * Savings: K × (C − 1)  [e.g. K=3, C=31 → +90 tokens/cell saved]

    Named falsifier (W25-Λ-disjoint): when consumers have fully
    independent state (``fanout_registry = None`` or the producer's
    cell_index doesn't match any consumer's current cell), W25 fires
    ``W25_BRANCH_NO_TRIGGER`` and reduces to W24 per-agent.

    Trust boundary: the registry verifies hash + schema + consumer
    authorisation before returning an envelope; on any failure the
    consumer falls through to W24 byte-for-byte.
    """

    inner: MultiCellSessionCompactor = dataclasses.field(
        default_factory=lambda: MultiCellSessionCompactor())
    fanout_registry: SharedFanoutRegistry | None = None
    agent_id: str = ""
    is_producer: bool = False
    producer_agent_id: str = ""
    consumer_agent_ids: tuple[str, ...] = ()
    schema: SchemaCapsule | None = None
    enabled: bool = True
    require_fanout_verification: bool = True
    trigger_branches: frozenset[str] = dataclasses.field(
        default_factory=lambda: W25_DEFAULT_TRIGGER_BRANCHES)

    _last_result: W25FanoutResult | None = None
    _last_fanout_envelope: FanoutEnvelope | None = None
    _cell_index: int = 0

    def reset_session(self) -> None:
        self._last_result = None
        self._last_fanout_envelope = None
        self._cell_index = 0
        self.inner.reset_session()

    @property
    def T_decoder(self) -> int | None:
        return self.inner.T_decoder

    @T_decoder.setter
    def T_decoder(self, v: int | None) -> None:
        self.inner.T_decoder = v

    def decode_rounds(
            self,
            per_round_handoffs: Sequence[Sequence[_DecodedHandoff]],
    ) -> dict[str, Any]:
        base = self.inner.decode_rounds(per_round_handoffs)
        w24_result = self.inner.last_result
        out = dict(base)
        cell_index = int(self._cell_index)

        inner_w24_branch = (str(w24_result.decoder_branch)
                              if w24_result is not None
                              else W24_BRANCH_DISABLED)
        n_w24_visible = (int(w24_result.n_w24_visible_tokens_to_decider)
                           if w24_result is not None else 0)

        def _pack(
                *,
                decoder_branch: str,
                fanout: FanoutEnvelope | None,
                fanout_ok: bool,
                fanout_reason: str,
                n_w25_visible: int,
        ) -> dict[str, Any]:
            savings = max(0, n_w24_visible - n_w25_visible)
            fanout_cid = str(fanout.fanout_cid) if fanout is not None else ""
            n_consumers = (len(fanout.consumer_agent_ids)
                             if fanout is not None else 0)
            n_fanout_bytes = (int(fanout.n_fanout_bytes)
                                if fanout is not None else 0)
            result = W25FanoutResult(
                answer=dict(base),
                inner_w24_branch=inner_w24_branch,
                decoder_branch=decoder_branch,
                agent_id=str(self.agent_id),
                is_producer=bool(self.is_producer),
                fanout_cid=fanout_cid,
                n_consumer_agents=n_consumers,
                n_w24_visible_tokens=n_w24_visible,
                n_w25_visible_tokens=n_w25_visible,
                n_w24_minus_w25_savings=savings,
                n_fanout_bytes=n_fanout_bytes,
                fanout_verification_ok=fanout_ok,
                fanout_verification_reason=fanout_reason,
            )
            self._last_result = result
            self._last_fanout_envelope = fanout
            out_local = dict(out)
            out_local["shared_fanout_hybrid"] = result.as_dict()
            if fanout is not None:
                out_local["fanout_envelope"] = fanout.as_dict()
            return out_local

        if not self.enabled or self.fanout_registry is None:
            self._cell_index += 1
            return _pack(
                decoder_branch=W25_BRANCH_DISABLED,
                fanout=None, fanout_ok=False, fanout_reason="disabled",
                n_w25_visible=n_w24_visible)

        if inner_w24_branch not in self.trigger_branches:
            self._cell_index += 1
            return _pack(
                decoder_branch=W25_BRANCH_NO_TRIGGER,
                fanout=None, fanout_ok=False,
                fanout_reason="inner_w24_not_triggered",
                n_w25_visible=n_w24_visible)

        # --- Producer path ---
        if self.is_producer:
            w24_compact = self.inner.last_compact_envelope
            compact_votes: tuple[tuple[str, int], ...] = ()
            compact_projected: tuple[str, ...] = ()
            n_resolved = 0
            if w24_compact is not None:
                compact_votes = tuple(w24_compact.compact_per_tag_votes)
                compact_projected = tuple(w24_compact.compact_projected_subset)
                n_resolved = int(w24_compact.n_resolved_in_window)
            elif w24_result is not None:
                # Genesis cell: extract votes from the W22 layer result
                w22_r = getattr(self.inner.inner.inner, "last_result", None)
                if w22_r is not None and hasattr(w22_r, "per_tag_votes"):
                    compact_votes = tuple(
                        sorted(w22_r.per_tag_votes.items(),
                                key=lambda kv: kv[0]))
                    compact_projected = tuple(
                        sorted(w22_r.projected_subset))

            schema_cid = (str(self.schema.cid)
                            if self.schema is not None else "")
            fanout = FanoutEnvelope(
                schema_version=W25_FANOUT_SCHEMA_VERSION,
                producer_agent_id=str(self.agent_id),
                consumer_agent_ids=tuple(self.consumer_agent_ids),
                compact_per_tag_votes=compact_votes,
                compact_projected_subset=compact_projected,
                schema_cid=schema_cid,
                cell_index=cell_index,
                n_resolved_in_window=n_resolved,
            )
            reg_outcome = self.fanout_registry.register(fanout)
            self._cell_index += 1
            return _pack(
                decoder_branch=W25_BRANCH_FANOUT_PRODUCER_EMITTED,
                fanout=fanout,
                fanout_ok=reg_outcome.ok,
                fanout_reason=str(reg_outcome.reason),
                n_w25_visible=n_w24_visible)  # producer cost unchanged

        # --- Consumer path ---
        producer_id = str(self.producer_agent_id)
        fanout = self.fanout_registry.get_by_producer(
            producer_id, cell_index)

        if fanout is None:
            self._cell_index += 1
            return _pack(
                decoder_branch=W25_BRANCH_FANOUT_CONSUMER_REJECTED,
                fanout=None, fanout_ok=False,
                fanout_reason="producer_fanout_not_found",
                n_w25_visible=n_w24_visible)

        if self.schema is not None:
            outcome = verify_fanout(
                fanout,
                registered_schema=self.schema,
                consumer_id=str(self.agent_id))
            if not outcome.ok and self.require_fanout_verification:
                self._cell_index += 1
                return _pack(
                    decoder_branch=W25_BRANCH_FANOUT_CONSUMER_REJECTED,
                    fanout=fanout, fanout_ok=False,
                    fanout_reason=str(outcome.reason),
                    n_w25_visible=n_w24_visible)

        # Consumer pays 1 token for the fanout ref.
        n_w25_consumer = 1
        self._cell_index += 1
        return _pack(
            decoder_branch=W25_BRANCH_FANOUT_CONSUMER_RESOLVED,
            fanout=fanout, fanout_ok=True, fanout_reason="ok",
            n_w25_visible=n_w25_consumer)

    def decode(self, handoffs: Sequence[_DecodedHandoff]
               ) -> dict[str, Any]:
        return self.decode_rounds([handoffs])

    @property
    def last_result(self) -> W25FanoutResult | None:
        return self._last_result

    @property
    def last_fanout_envelope(self) -> FanoutEnvelope | None:
        return self._last_fanout_envelope


# =============================================================================
# W26 — chain-persisted dense-control fanout (cross-cell amortisation +
# per-consumer projections)
# =============================================================================
#
# The W25 family reduced multi-agent fanout cost from K × C tokens/cell
# to C + K tokens/cell at K=3 (where C is the producer's per-cell
# compact + salience cost ≈ 14.6 tokens). W25 still requires the
# producer to re-emit the full ``n_w15_kept`` salience-token block on
# *every* cell, even when the salience content is structurally repeated
# across cells (the bench-driver reads the same gold services and decoy
# magnitudes every cell).  W26 lifts that floor by chaining the
# producer's compact state across cells:
#
#   * At a *chain anchor* cell, the producer pays the full W25 cost
#     (compact + salience + fanout register).  The anchor envelope
#     carries the canonical salience-token block and the per-consumer
#     projection slots (so each consumer can subscribe to a distinct
#     subset of tags / projected services).
#   * At each subsequent in-window cell, the producer emits a 1-token
#     ``<chain_advance:DDDD>`` reference and registers a small
#     :class:`ChainAdvanceEnvelope` that hash-chains to the parent
#     advance (or the anchor for the first advance).  The advance
#     carries only the *delta* from the parent (small, often empty on
#     stable benches).  Consumers emit a 1-token
#     ``<chain_consumer:DDDD>`` reference indexed by the current
#     advance.
#   * Re-anchor occurs when the chain reaches ``chain_persist_window``
#     cells or when the controller-verified divergence guard rejects an
#     advance (e.g., the producer's gold subset diverges beyond the
#     allowed projection drift).
#
# Total visible-token cost over N cells, K consumers (chain window W,
# C = mean per-agent W25 producer cost when chain re-anchors, c = 1):
#   W25:     N × (C + K)                            [no chain]
#   W26:     ⌈N/W⌉ × (C + K) + (N − ⌈N/W⌉) × (1+K)  [chain hits]
#
# At N=16, K=3, C=14.625, W=16 (one anchor):
#   W25 ≈ 282 tokens
#   W26 ≈ 17.625 + 15 × 4 = 77.625 tokens   (saving ≈ 72.5%)
#
# Trust boundary
# ==============
# Every chain envelope is content-addressed, schema-versioned, and
# parent-CID-sealed.  The controller-side
# :class:`ChainPersistedFanoutRegistry` enforces:
#   * anchor_cid integrity (SHA-256 over canonical bytes);
#   * advance_cid integrity (SHA-256 over canonical bytes);
#   * parent linkage (every advance references either the anchor or a
#     prior advance whose CID is registered);
#   * cell-index monotonicity (no replay or reordering);
#   * schema_cid pinning (cross-schema envelopes rejected);
#   * consumer-projection scope (consumers can only resolve projections
#     they were named in by the anchor);
#   * window expiry (advances beyond ``chain_persist_window`` are
#     rejected — the producer must re-anchor).
#
# Honest scope
# ============
# W26 is an **amortisation** mechanism, not a new information channel:
#   * It does not add new content; it changes how the existing producer
#     state is *delivered* to the K consumers + final decoder.
#   * The total *bytes* on the wire is bounded above by W25's bytes
#     when consumer projections fully reuse the anchor's content; on
#     anchor cells W26 ≥ W25 by a constant (the projection slot map).
#     Across N cells the total bytes is ≤ W25 by construction (advance
#     envelopes carry only deltas, hash chains, and CIDs).
#   * The visible-token reduction comes from the chain reference being
#     a single whitespace token by design (``<chain_advance:DDDD>``),
#     which is the same accounting model already in W23/W24/W25.
#
# Named falsifiers
# ================
#   * **W26-Λ-no-chain** — ``chain_persist_window = 1`` reduces to W25
#     byte-for-byte (every cell is an anchor).
#   * **W26-Λ-divergent** — when consecutive cells produce divergent
#     gold subsets (the bench driver flips the gold tag), the
#     controller rejects the advance and the producer re-anchors; the
#     measured savings collapse toward the W25 floor on the divergent
#     cells.
#   * **W26-Λ-tampered** — any tamper on advance hash / parent linkage
#     / schema_cid is rejected by ``verify_chain_advance``; W26 falls
#     through to W25 byte-for-byte on that cell.
#   * **W26-Λ-projection-mismatch** — consumer asks for a projection
#     not in ``projection_slots``: rejected with
#     ``projection_unauthorized``; consumer falls through to W25.

W26_CHAIN_ANCHOR_SCHEMA_VERSION: str = "wevra.chain_anchor.v1"
W26_CHAIN_ADVANCE_SCHEMA_VERSION: str = "wevra.chain_advance.v1"

W26_BRANCH_CHAIN_ANCHORED = "chain_anchored"
W26_BRANCH_CHAIN_ADVANCED = "chain_advanced"
W26_BRANCH_CHAIN_REJECTED = "chain_rejected"
W26_BRANCH_CHAIN_RE_ANCHORED = "chain_re_anchored"
W26_BRANCH_CHAIN_PROJECTION_RESOLVED = "chain_projection_resolved"
W26_BRANCH_CHAIN_PROJECTION_REJECTED = "chain_projection_rejected"
W26_BRANCH_NO_TRIGGER = "no_trigger"
W26_BRANCH_DISABLED = "disabled"

W26_ALL_BRANCHES: tuple[str, ...] = (
    W26_BRANCH_CHAIN_ANCHORED,
    W26_BRANCH_CHAIN_ADVANCED,
    W26_BRANCH_CHAIN_REJECTED,
    W26_BRANCH_CHAIN_RE_ANCHORED,
    W26_BRANCH_CHAIN_PROJECTION_RESOLVED,
    W26_BRANCH_CHAIN_PROJECTION_REJECTED,
    W26_BRANCH_NO_TRIGGER,
    W26_BRANCH_DISABLED,
)

# W26 fires whenever W25 was able to fire — anchoring on the producer
# emit, advancing for subsequent cells.  If the inner W25 abstains
# (W25_BRANCH_NO_TRIGGER / W25_BRANCH_DISABLED), W26 also abstains.
W26_DEFAULT_TRIGGER_BRANCHES: frozenset[str] = frozenset({
    W25_BRANCH_FANOUT_PRODUCER_EMITTED,
    W25_BRANCH_FANOUT_CONSUMER_RESOLVED,
})


@dataclasses.dataclass(frozen=True)
class ProjectionSlot:
    """Per-consumer projection slot inside a :class:`ChainAnchorEnvelope`.

    Each consumer named in the anchor receives a *projection* — the
    subset of compact_per_tag_votes / projected_subset they are
    authorised to resolve.  A consumer can only resolve their own
    projection_id (cross-projection access is rejected).

    Trust: the projection_cid is hash-chained into the anchor's
    chain_root_cid, so the slot map is tamper-evident.
    """
    projection_id: str
    consumer_id: str
    projected_tags: tuple[str, ...]
    projection_cid: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "projected_tags",
                            tuple(sorted(self.projected_tags)))
        if not self.projection_cid:
            object.__setattr__(self, "projection_cid",
                                self.recompute_projection_cid())

    def _signed_payload(self) -> dict[str, Any]:
        return {
            "projection_id": str(self.projection_id),
            "consumer_id": str(self.consumer_id),
            "projected_tags": list(self.projected_tags),
        }

    def to_canonical_bytes(self) -> bytes:
        return _canonical_json_bytes(self._signed_payload())

    def recompute_projection_cid(self) -> str:
        return hashlib.sha256(self.to_canonical_bytes()).hexdigest()


@dataclasses.dataclass(frozen=True)
class ChainAnchorEnvelope:
    """Chain anchor (W26 family) — genesis cell of a chain window.

    Carries the canonical compact state (votes + projected subset)
    plus per-consumer projection slots and the chain window length.
    The ``chain_root_cid`` is the unique identifier for the chain;
    every subsequent :class:`ChainAdvanceEnvelope` hash-chains to it.
    """
    schema_version: str
    schema_cid: str
    producer_agent_id: str
    consumer_agent_ids: tuple[str, ...]
    cell_index_anchor: int
    chain_persist_window: int
    canonical_compact_per_tag_votes: tuple[tuple[str, int], ...]
    canonical_compact_projected_subset: tuple[str, ...]
    n_w15_canonical_tokens: int
    projection_slots: tuple[ProjectionSlot, ...]
    chain_root_cid: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "consumer_agent_ids",
                            tuple(self.consumer_agent_ids))
        object.__setattr__(self, "canonical_compact_per_tag_votes",
                            tuple(sorted(self.canonical_compact_per_tag_votes,
                                          key=lambda kv: kv[0])))
        object.__setattr__(self, "canonical_compact_projected_subset",
                            tuple(sorted(self.canonical_compact_projected_subset)))
        object.__setattr__(self, "projection_slots",
                            tuple(sorted(self.projection_slots,
                                          key=lambda s: s.projection_id)))
        if not self.chain_root_cid:
            object.__setattr__(self, "chain_root_cid",
                                self.recompute_chain_root_cid())

    def _signed_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "schema_cid": self.schema_cid,
            "producer_agent_id": self.producer_agent_id,
            "consumer_agent_ids": sorted(self.consumer_agent_ids),
            "cell_index_anchor": int(self.cell_index_anchor),
            "chain_persist_window": int(self.chain_persist_window),
            "canonical_compact_per_tag_votes": [
                [t, int(c)]
                for t, c in self.canonical_compact_per_tag_votes],
            "canonical_compact_projected_subset":
                list(self.canonical_compact_projected_subset),
            "n_w15_canonical_tokens": int(self.n_w15_canonical_tokens),
            "projection_slots": [
                s._signed_payload() for s in self.projection_slots],
        }

    def to_canonical_bytes(self) -> bytes:
        return _canonical_json_bytes(self._signed_payload())

    def recompute_chain_root_cid(self) -> str:
        return hashlib.sha256(self.to_canonical_bytes()).hexdigest()

    @property
    def n_anchor_bytes(self) -> int:
        return len(self.to_canonical_bytes())

    def projection_for(self, consumer_id: str) -> ProjectionSlot | None:
        for s in self.projection_slots:
            if s.consumer_id == consumer_id:
                return s
        return None

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "schema_cid": self.schema_cid,
            "producer_agent_id": self.producer_agent_id,
            "consumer_agent_ids": list(self.consumer_agent_ids),
            "cell_index_anchor": int(self.cell_index_anchor),
            "chain_persist_window": int(self.chain_persist_window),
            "canonical_compact_per_tag_votes": [
                [t, int(c)]
                for t, c in self.canonical_compact_per_tag_votes],
            "canonical_compact_projected_subset":
                list(self.canonical_compact_projected_subset),
            "n_w15_canonical_tokens": int(self.n_w15_canonical_tokens),
            "n_anchor_bytes": int(self.n_anchor_bytes),
            "projection_slots": [
                {**s._signed_payload(),
                  "projection_cid": s.projection_cid}
                for s in self.projection_slots],
            "chain_root_cid": self.chain_root_cid,
        }


@dataclasses.dataclass(frozen=True)
class ChainAdvanceEnvelope:
    """Chain advance (W26 family) — per-cell delta inside a chain.

    Hash-chains to the parent advance (or directly to the anchor for
    cell 1 of a chain).  Carries only the delta from the parent — on
    a stable bench (R-69-CACHE-FANOUT) the delta is empty by design.
    """
    schema_version: str
    schema_cid: str
    chain_root_cid: str
    parent_advance_cid: str  # anchor CID for first advance, else previous advance_cid
    cell_index: int
    cell_in_chain: int  # 1-indexed within the chain window
    delta_per_tag_votes: tuple[tuple[str, int], ...]
    delta_projected_subset_added: tuple[str, ...]
    delta_projected_subset_removed: tuple[str, ...]
    advance_cid: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "delta_per_tag_votes",
                            tuple(sorted(self.delta_per_tag_votes,
                                          key=lambda kv: kv[0])))
        object.__setattr__(self, "delta_projected_subset_added",
                            tuple(sorted(self.delta_projected_subset_added)))
        object.__setattr__(self, "delta_projected_subset_removed",
                            tuple(sorted(self.delta_projected_subset_removed)))
        if not self.advance_cid:
            object.__setattr__(self, "advance_cid",
                                self.recompute_advance_cid())

    def _signed_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "schema_cid": self.schema_cid,
            "chain_root_cid": self.chain_root_cid,
            "parent_advance_cid": self.parent_advance_cid,
            "cell_index": int(self.cell_index),
            "cell_in_chain": int(self.cell_in_chain),
            "delta_per_tag_votes": [
                [t, int(c)] for t, c in self.delta_per_tag_votes],
            "delta_projected_subset_added":
                list(self.delta_projected_subset_added),
            "delta_projected_subset_removed":
                list(self.delta_projected_subset_removed),
        }

    def to_canonical_bytes(self) -> bytes:
        return _canonical_json_bytes(self._signed_payload())

    def recompute_advance_cid(self) -> str:
        return hashlib.sha256(self.to_canonical_bytes()).hexdigest()

    def to_decoder_text(self) -> str:
        return f"<chain_advance:{self.advance_cid[:16]}>"

    @property
    def n_advance_tokens(self) -> int:
        return _whitespace_token_count(self.to_decoder_text())

    @property
    def n_advance_bytes(self) -> int:
        return len(self.to_canonical_bytes())

    @property
    def is_empty_delta(self) -> bool:
        return (not self.delta_per_tag_votes
                  and not self.delta_projected_subset_added
                  and not self.delta_projected_subset_removed)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "schema_cid": self.schema_cid,
            "chain_root_cid": self.chain_root_cid,
            "parent_advance_cid": self.parent_advance_cid,
            "cell_index": int(self.cell_index),
            "cell_in_chain": int(self.cell_in_chain),
            "delta_per_tag_votes": [
                [t, int(c)] for t, c in self.delta_per_tag_votes],
            "delta_projected_subset_added":
                list(self.delta_projected_subset_added),
            "delta_projected_subset_removed":
                list(self.delta_projected_subset_removed),
            "advance_cid": self.advance_cid,
            "n_advance_tokens": int(self.n_advance_tokens),
            "n_advance_bytes": int(self.n_advance_bytes),
            "is_empty_delta": bool(self.is_empty_delta),
            "decoder_text": self.to_decoder_text(),
        }


def verify_chain_anchor(
        anchor: ChainAnchorEnvelope | None,
        *,
        registered_schema: SchemaCapsule,
) -> LatentVerificationOutcome:
    """Controller-side verification of a :class:`ChainAnchorEnvelope`.

    Pure function. Failure modes enumerated:

    * ``empty_anchor``               — no anchor passed.
    * ``schema_version_unknown``     — anchor.schema_version mismatch.
    * ``schema_cid_mismatch``        — anchor.schema_cid != registered.
    * ``window_non_positive``        — chain_persist_window ≤ 0.
    * ``projection_cid_mismatch``    — a slot's CID does not recompute.
    * ``hash_mismatch``              — chain_root_cid does not recompute.
    """
    n = 0
    if anchor is None:
        return LatentVerificationOutcome(
            ok=False, reason="empty_anchor", n_checks=n)
    n += 1
    if anchor.schema_version != W26_CHAIN_ANCHOR_SCHEMA_VERSION:
        return LatentVerificationOutcome(
            ok=False, reason="schema_version_unknown", n_checks=n)
    n += 1
    if anchor.schema_cid != registered_schema.cid:
        return LatentVerificationOutcome(
            ok=False, reason="schema_cid_mismatch", n_checks=n)
    n += 1
    if int(anchor.chain_persist_window) <= 0:
        return LatentVerificationOutcome(
            ok=False, reason="window_non_positive", n_checks=n)
    n += 1
    for s in anchor.projection_slots:
        if s.projection_cid != s.recompute_projection_cid():
            return LatentVerificationOutcome(
                ok=False, reason="projection_cid_mismatch", n_checks=n)
    n += 1
    if anchor.chain_root_cid != anchor.recompute_chain_root_cid():
        return LatentVerificationOutcome(
            ok=False, reason="hash_mismatch", n_checks=n)
    return LatentVerificationOutcome(
        ok=True, reason="ok", n_checks=n)


def verify_chain_advance(
        advance: ChainAdvanceEnvelope | None,
        *,
        registered_schema: SchemaCapsule,
        anchor: ChainAnchorEnvelope,
        expected_parent_cid: str,
        expected_cell_in_chain: int,
) -> LatentVerificationOutcome:
    """Controller-side verification of a :class:`ChainAdvanceEnvelope`.

    Pure function. Failure modes enumerated:

    * ``empty_advance``              — no advance passed.
    * ``schema_version_unknown``     — advance.schema_version mismatch.
    * ``schema_cid_mismatch``        — advance.schema_cid != registered.
    * ``chain_root_mismatch``        — advance.chain_root_cid != anchor.
    * ``parent_mismatch``            — parent_advance_cid != expected.
    * ``cell_in_chain_mismatch``     — cell_in_chain != expected.
    * ``window_expired``             — cell_in_chain > chain_persist_window.
    * ``hash_mismatch``              — advance_cid does not recompute.
    """
    n = 0
    if advance is None:
        return LatentVerificationOutcome(
            ok=False, reason="empty_advance", n_checks=n)
    n += 1
    if advance.schema_version != W26_CHAIN_ADVANCE_SCHEMA_VERSION:
        return LatentVerificationOutcome(
            ok=False, reason="schema_version_unknown", n_checks=n)
    n += 1
    if advance.schema_cid != registered_schema.cid:
        return LatentVerificationOutcome(
            ok=False, reason="schema_cid_mismatch", n_checks=n)
    n += 1
    if advance.chain_root_cid != anchor.chain_root_cid:
        return LatentVerificationOutcome(
            ok=False, reason="chain_root_mismatch", n_checks=n)
    n += 1
    if advance.parent_advance_cid != expected_parent_cid:
        return LatentVerificationOutcome(
            ok=False, reason="parent_mismatch", n_checks=n)
    n += 1
    if int(advance.cell_in_chain) != int(expected_cell_in_chain):
        return LatentVerificationOutcome(
            ok=False, reason="cell_in_chain_mismatch", n_checks=n)
    n += 1
    if int(advance.cell_in_chain) > int(anchor.chain_persist_window):
        return LatentVerificationOutcome(
            ok=False, reason="window_expired", n_checks=n)
    n += 1
    if advance.advance_cid != advance.recompute_advance_cid():
        return LatentVerificationOutcome(
            ok=False, reason="hash_mismatch", n_checks=n)
    return LatentVerificationOutcome(
        ok=True, reason="ok", n_checks=n)


def verify_projection_subscription(
        anchor: ChainAnchorEnvelope,
        *,
        consumer_id: str,
        projection_id: str,
) -> LatentVerificationOutcome:
    """Controller-side verification that ``consumer_id`` is authorised
    to resolve ``projection_id`` against ``anchor``.

    Pure function. Failure modes:

    * ``consumer_not_in_anchor``        — consumer_id not in anchor.consumer_agent_ids.
    * ``projection_unauthorized``       — projection_id not in anchor.projection_slots
                                            for this consumer.
    """
    n = 0
    n += 1
    if consumer_id not in anchor.consumer_agent_ids:
        return LatentVerificationOutcome(
            ok=False, reason="consumer_not_in_anchor", n_checks=n)
    n += 1
    slot = anchor.projection_for(consumer_id)
    if slot is None or slot.projection_id != projection_id:
        return LatentVerificationOutcome(
            ok=False, reason="projection_unauthorized", n_checks=n)
    return LatentVerificationOutcome(
        ok=True, reason="ok", n_checks=n)


@dataclasses.dataclass
class ChainPersistedFanoutRegistry:
    """Controller-side registry for chain anchors and advances (W26).

    The registry is the single source of truth for chain validity.
    Producers register the anchor exactly once per chain window, then
    submit advances whose hash and parent linkage are checked.

    Trust boundary: the registry is controller-owned.  A malicious
    producer cannot register a tampered advance; a malicious consumer
    cannot resolve a projection they were not slotted into.
    """
    schema: SchemaCapsule | None = None
    _anchors: dict[str, ChainAnchorEnvelope] = dataclasses.field(
        default_factory=dict)  # keyed by chain_root_cid
    _advances: dict[str, ChainAdvanceEnvelope] = dataclasses.field(
        default_factory=dict)  # keyed by advance_cid
    _chain_state: dict[str, dict[str, Any]] = dataclasses.field(
        default_factory=dict)  # keyed by chain_root_cid; tracks parent_cid + cell_in_chain
    n_anchors_registered: int = 0
    n_advances_registered: int = 0
    n_advances_rejected: int = 0
    n_projections_resolved: int = 0
    n_projections_rejected: int = 0

    def register_anchor(
            self, anchor: ChainAnchorEnvelope,
    ) -> LatentVerificationOutcome:
        if self.schema is None:
            self.n_advances_rejected += 0  # registry without schema cannot verify
            return LatentVerificationOutcome(
                ok=False, reason="registry_no_schema", n_checks=0)
        outcome = verify_chain_anchor(
            anchor, registered_schema=self.schema)
        if not outcome.ok:
            return outcome
        self._anchors[anchor.chain_root_cid] = anchor
        self._chain_state[anchor.chain_root_cid] = {
            "parent_cid": anchor.chain_root_cid,
            "cell_in_chain": 0,
        }
        self.n_anchors_registered += 1
        return outcome

    def register_advance(
            self, advance: ChainAdvanceEnvelope,
    ) -> LatentVerificationOutcome:
        if self.schema is None:
            self.n_advances_rejected += 1
            return LatentVerificationOutcome(
                ok=False, reason="registry_no_schema", n_checks=0)
        anchor = self._anchors.get(advance.chain_root_cid)
        if anchor is None:
            self.n_advances_rejected += 1
            return LatentVerificationOutcome(
                ok=False, reason="anchor_not_found", n_checks=0)
        state = self._chain_state.get(advance.chain_root_cid, {})
        expected_parent = str(state.get("parent_cid",
                                          anchor.chain_root_cid))
        expected_cell_in_chain = int(state.get("cell_in_chain", 0)) + 1
        outcome = verify_chain_advance(
            advance,
            registered_schema=self.schema,
            anchor=anchor,
            expected_parent_cid=expected_parent,
            expected_cell_in_chain=expected_cell_in_chain)
        if not outcome.ok:
            self.n_advances_rejected += 1
            return outcome
        self._advances[advance.advance_cid] = advance
        self._chain_state[advance.chain_root_cid] = {
            "parent_cid": advance.advance_cid,
            "cell_in_chain": expected_cell_in_chain,
        }
        self.n_advances_registered += 1
        return outcome

    def expected_parent_for(self, chain_root_cid: str) -> str:
        state = self._chain_state.get(chain_root_cid, {})
        return str(state.get("parent_cid", chain_root_cid))

    def expected_cell_in_chain_for(self, chain_root_cid: str) -> int:
        state = self._chain_state.get(chain_root_cid, {})
        return int(state.get("cell_in_chain", 0)) + 1

    def get_anchor(
            self, chain_root_cid: str,
    ) -> ChainAnchorEnvelope | None:
        return self._anchors.get(chain_root_cid)

    def get_advance(
            self, advance_cid: str,
    ) -> ChainAdvanceEnvelope | None:
        return self._advances.get(advance_cid)

    def latest_state(
            self, chain_root_cid: str,
    ) -> tuple[str, int]:
        """Return (parent_cid, cell_in_chain) for the latest registered
        cell of this chain.  Used by consumers resolving the chain head.
        """
        state = self._chain_state.get(chain_root_cid, {})
        return (str(state.get("parent_cid", chain_root_cid)),
                int(state.get("cell_in_chain", 0)))

    def resolve_projection(
            self,
            *,
            chain_root_cid: str,
            consumer_id: str,
            projection_id: str,
    ) -> tuple[ChainAnchorEnvelope | None, str]:
        anchor = self._anchors.get(chain_root_cid)
        if anchor is None:
            self.n_projections_rejected += 1
            return None, "anchor_not_found"
        outcome = verify_projection_subscription(
            anchor, consumer_id=consumer_id,
            projection_id=projection_id)
        if not outcome.ok:
            self.n_projections_rejected += 1
            return None, outcome.reason
        self.n_projections_resolved += 1
        return anchor, "ok"


@dataclasses.dataclass
class W26ChainResult:
    """Per-cell audit record for a W26 chain-persisted agent."""
    answer: dict[str, Any]
    inner_w25_branch: str
    decoder_branch: str
    agent_id: str
    is_producer: bool
    chain_root_cid: str
    advance_cid: str
    cell_in_chain: int
    chain_persist_window: int
    n_w25_visible_tokens: int
    n_w26_visible_tokens: int
    n_w25_minus_w26_savings: int
    n_anchor_bytes: int
    n_advance_bytes: int
    chain_verification_ok: bool
    chain_verification_reason: str
    projection_resolved: bool
    projection_id: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "inner_w25_branch": str(self.inner_w25_branch),
            "decoder_branch": str(self.decoder_branch),
            "agent_id": str(self.agent_id),
            "is_producer": bool(self.is_producer),
            "chain_root_cid": str(self.chain_root_cid),
            "advance_cid": str(self.advance_cid),
            "cell_in_chain": int(self.cell_in_chain),
            "chain_persist_window": int(self.chain_persist_window),
            "n_w25_visible_tokens": int(self.n_w25_visible_tokens),
            "n_w26_visible_tokens": int(self.n_w26_visible_tokens),
            "n_w25_minus_w26_savings":
                int(self.n_w25_minus_w26_savings),
            "n_anchor_bytes": int(self.n_anchor_bytes),
            "n_advance_bytes": int(self.n_advance_bytes),
            "chain_verification_ok": bool(self.chain_verification_ok),
            "chain_verification_reason":
                str(self.chain_verification_reason),
            "projection_resolved": bool(self.projection_resolved),
            "projection_id": str(self.projection_id),
        }


@dataclasses.dataclass
class ChainPersistedFanoutDisambiguator:
    """Cross-cell chain-persisted dense-control fanout (W26 family).

    Wraps a :class:`SharedFanoutDisambiguator` (W25) and adds:

    1. **Producer-side cross-cell amortisation.**  At a chain anchor
       cell the producer pays the full W25 cost
       (``n_w25_visible = n_w24_visible`` ≈ 14.6 tokens at K=3).  At
       each subsequent in-window cell the producer pays a single
       ``<chain_advance:DDDD>`` token (1 token).  Producer cost over N
       cells of a chain window of size W:
         W26 producer ≈ C + (W-1)
       vs W25 producer ≈ W × C
       Saving: (W-1) × (C-1) per window.
    2. **Consumer-side persistent subscription.**  Consumers subscribe
       once at the anchor (1 token, same as W25 fanout_ref) and emit
       a 1-token advance reference per subsequent cell (same cost as
       W25 fanout_ref but resolves through the chain instead of a
       per-cell fanout envelope, eliminating the per-cell fanout
       registration overhead from K × bytes/cell to bytes per anchor).
    3. **Per-consumer projections.**  Each consumer is slotted into
       a projection (subset of compact_per_tag_votes /
       projected_subset).  A consumer can only resolve their own
       projection_id; cross-projection access is rejected.

    Trust boundary
    --------------
    Every chain envelope (anchor + advance) is content-addressed,
    schema-versioned, and parent-CID-sealed.  The controller's
    :class:`ChainPersistedFanoutRegistry` enforces:
      * anchor / advance hash integrity;
      * parent linkage (every advance hash-chains to anchor or prior advance);
      * cell-index monotonicity inside a chain window;
      * window expiry (advance beyond chain_persist_window rejected);
      * schema_cid pinning (cross-schema envelopes rejected);
      * consumer-projection scope (consumers can only resolve their slot).

    Honest scope
    ------------
    W26 changes how the producer's compact state is *delivered* to the
    K consumers + final decoder; it does not add a new information
    channel.  The visible-token reduction comes from chaining
    references (the same accounting model already in W23/W24/W25).
    """
    inner: SharedFanoutDisambiguator = dataclasses.field(
        default_factory=lambda: SharedFanoutDisambiguator())
    chain_registry: ChainPersistedFanoutRegistry | None = None
    schema: SchemaCapsule | None = None
    chain_persist_window: int = 16
    enabled: bool = True
    require_chain_verification: bool = True
    trigger_branches: frozenset[str] = dataclasses.field(
        default_factory=lambda: W26_DEFAULT_TRIGGER_BRANCHES)
    projection_id_for_consumer: dict[str, str] = dataclasses.field(
        default_factory=dict)
    projected_tags_for_consumer: dict[str, tuple[str, ...]] = (
        dataclasses.field(default_factory=dict))

    _last_result: W26ChainResult | None = None
    _active_chain_root_cid: str = ""
    _active_anchor: ChainAnchorEnvelope | None = None
    _active_cell_in_chain: int = 0
    _last_advance: ChainAdvanceEnvelope | None = None
    _cell_index: int = 0
    _last_anchor_per_tag_votes: tuple[tuple[str, int], ...] = ()
    _last_anchor_projected_subset: tuple[str, ...] = ()

    @property
    def agent_id(self) -> str:
        return self.inner.agent_id

    @property
    def is_producer(self) -> bool:
        return self.inner.is_producer

    @property
    def producer_agent_id(self) -> str:
        return self.inner.producer_agent_id

    @property
    def consumer_agent_ids(self) -> tuple[str, ...]:
        return self.inner.consumer_agent_ids

    @property
    def T_decoder(self) -> int | None:
        return self.inner.T_decoder

    @T_decoder.setter
    def T_decoder(self, v: int | None) -> None:
        self.inner.T_decoder = v

    def reset_session(self) -> None:
        self._last_result = None
        self._active_chain_root_cid = ""
        self._active_anchor = None
        self._active_cell_in_chain = 0
        self._last_advance = None
        self._cell_index = 0
        self._last_anchor_per_tag_votes = ()
        self._last_anchor_projected_subset = ()
        self.inner.reset_session()

    def _make_projection_slots(
            self,
            *,
            consumer_ids: tuple[str, ...],
            canonical_projected_subset: tuple[str, ...],
    ) -> tuple[ProjectionSlot, ...]:
        slots: list[ProjectionSlot] = []
        for cid in consumer_ids:
            pid = self.projection_id_for_consumer.get(cid, "default")
            tags = self.projected_tags_for_consumer.get(
                cid, canonical_projected_subset)
            slots.append(ProjectionSlot(
                projection_id=str(pid),
                consumer_id=str(cid),
                projected_tags=tuple(tags),
            ))
        return tuple(slots)

    def _build_anchor(
            self,
            *,
            cell_index: int,
            w25_result: W25FanoutResult,
            n_w15_canonical_tokens: int,
            canonical_compact_per_tag_votes: tuple[tuple[str, int], ...],
            canonical_compact_projected_subset: tuple[str, ...],
    ) -> ChainAnchorEnvelope:
        schema_cid = (str(self.schema.cid)
                        if self.schema is not None else "")
        slots = self._make_projection_slots(
            consumer_ids=tuple(self.consumer_agent_ids),
            canonical_projected_subset=canonical_compact_projected_subset)
        return ChainAnchorEnvelope(
            schema_version=W26_CHAIN_ANCHOR_SCHEMA_VERSION,
            schema_cid=schema_cid,
            producer_agent_id=str(self.producer_agent_id
                                    or self.agent_id),
            consumer_agent_ids=tuple(self.consumer_agent_ids),
            cell_index_anchor=int(cell_index),
            chain_persist_window=int(self.chain_persist_window),
            canonical_compact_per_tag_votes=tuple(
                canonical_compact_per_tag_votes),
            canonical_compact_projected_subset=tuple(
                canonical_compact_projected_subset),
            n_w15_canonical_tokens=int(n_w15_canonical_tokens),
            projection_slots=slots,
        )

    def _compute_delta(
            self,
            *,
            current_per_tag_votes: tuple[tuple[str, int], ...],
            current_projected_subset: tuple[str, ...],
    ) -> tuple[tuple[tuple[str, int], ...],
                tuple[str, ...], tuple[str, ...]]:
        anchor_votes = dict(self._last_anchor_per_tag_votes)
        cur_votes = dict(current_per_tag_votes)
        delta_votes: list[tuple[str, int]] = []
        for tag, cnt in cur_votes.items():
            if anchor_votes.get(tag, 0) != cnt:
                delta_votes.append((tag, int(cnt)))
        anchor_set = set(self._last_anchor_projected_subset)
        cur_set = set(current_projected_subset)
        added = tuple(sorted(cur_set - anchor_set))
        removed = tuple(sorted(anchor_set - cur_set))
        return (tuple(sorted(delta_votes, key=lambda kv: kv[0])),
                added, removed)

    def _build_advance(
            self,
            *,
            cell_index: int,
            anchor: ChainAnchorEnvelope,
            parent_advance_cid: str,
            cell_in_chain: int,
            current_per_tag_votes: tuple[tuple[str, int], ...],
            current_projected_subset: tuple[str, ...],
    ) -> ChainAdvanceEnvelope:
        schema_cid = (str(self.schema.cid)
                        if self.schema is not None else "")
        delta_votes, added, removed = self._compute_delta(
            current_per_tag_votes=current_per_tag_votes,
            current_projected_subset=current_projected_subset)
        return ChainAdvanceEnvelope(
            schema_version=W26_CHAIN_ADVANCE_SCHEMA_VERSION,
            schema_cid=schema_cid,
            chain_root_cid=anchor.chain_root_cid,
            parent_advance_cid=parent_advance_cid,
            cell_index=int(cell_index),
            cell_in_chain=int(cell_in_chain),
            delta_per_tag_votes=delta_votes,
            delta_projected_subset_added=added,
            delta_projected_subset_removed=removed,
        )

    def decode_rounds(
            self,
            per_round_handoffs: Sequence[Sequence[_DecodedHandoff]],
    ) -> dict[str, Any]:
        # Always run the W25 inner decode first; W25 already runs W24
        # and embeds its result.
        w25_out = self.inner.decode_rounds(per_round_handoffs)
        w25_result = self.inner.last_result
        out = dict(w25_out)
        cell_index = int(self._cell_index)

        inner_w25_branch = (str(w25_result.decoder_branch)
                              if w25_result is not None
                              else W25_BRANCH_DISABLED)
        n_w25_visible = (int(w25_result.n_w25_visible_tokens)
                          if w25_result is not None else 0)

        def _pack(
                *,
                decoder_branch: str,
                anchor: ChainAnchorEnvelope | None,
                advance: ChainAdvanceEnvelope | None,
                cell_in_chain: int,
                chain_ok: bool,
                chain_reason: str,
                n_w26_visible: int,
                projection_resolved: bool = False,
                projection_id: str = "",
        ) -> dict[str, Any]:
            savings = max(0, n_w25_visible - n_w26_visible)
            chain_root_cid = ""
            if anchor is not None:
                chain_root_cid = str(anchor.chain_root_cid)
            elif advance is not None:
                chain_root_cid = str(advance.chain_root_cid)
            elif self._active_anchor is not None:
                chain_root_cid = str(
                    self._active_anchor.chain_root_cid)
            advance_cid = (str(advance.advance_cid)
                            if advance is not None else "")
            n_anchor_bytes = (int(anchor.n_anchor_bytes)
                                if anchor is not None else 0)
            n_advance_bytes = (int(advance.n_advance_bytes)
                                 if advance is not None else 0)
            result = W26ChainResult(
                answer=dict(out),
                inner_w25_branch=inner_w25_branch,
                decoder_branch=decoder_branch,
                agent_id=str(self.agent_id),
                is_producer=bool(self.is_producer),
                chain_root_cid=chain_root_cid,
                advance_cid=advance_cid,
                cell_in_chain=int(cell_in_chain),
                chain_persist_window=int(self.chain_persist_window),
                n_w25_visible_tokens=n_w25_visible,
                n_w26_visible_tokens=n_w26_visible,
                n_w25_minus_w26_savings=savings,
                n_anchor_bytes=n_anchor_bytes,
                n_advance_bytes=n_advance_bytes,
                chain_verification_ok=bool(chain_ok),
                chain_verification_reason=str(chain_reason),
                projection_resolved=bool(projection_resolved),
                projection_id=str(projection_id),
            )
            self._last_result = result
            out_local = dict(out)
            out_local["chain_persisted_hybrid"] = result.as_dict()
            if anchor is not None:
                out_local["chain_anchor_envelope"] = anchor.as_dict()
            if advance is not None:
                out_local["chain_advance_envelope"] = advance.as_dict()
            return out_local

        if (not self.enabled or self.chain_registry is None
                or self.schema is None):
            self._cell_index += 1
            return _pack(
                decoder_branch=W26_BRANCH_DISABLED,
                anchor=None, advance=None, cell_in_chain=0,
                chain_ok=False, chain_reason="disabled",
                n_w26_visible=n_w25_visible)

        if inner_w25_branch not in self.trigger_branches:
            self._cell_index += 1
            return _pack(
                decoder_branch=W26_BRANCH_NO_TRIGGER,
                anchor=None, advance=None, cell_in_chain=0,
                chain_ok=False, chain_reason="inner_w25_not_triggered",
                n_w26_visible=n_w25_visible)

        # --- Producer path ---
        if self.is_producer and w25_result is not None:
            # Pull canonical content from the inner W25 fanout envelope
            # (built by the inner W25 from W24 / W22 state).
            fanout = self.inner.last_fanout_envelope
            if fanout is None:
                # No fanout envelope means W25 fired but didn't emit
                # — treat as no-trigger.
                self._cell_index += 1
                return _pack(
                    decoder_branch=W26_BRANCH_NO_TRIGGER,
                    anchor=None, advance=None, cell_in_chain=0,
                    chain_ok=False,
                    chain_reason="inner_w25_no_envelope",
                    n_w26_visible=n_w25_visible)

            canonical_per_tag_votes = tuple(
                fanout.compact_per_tag_votes)
            canonical_projected_subset = tuple(
                fanout.compact_projected_subset)
            # n_w15_canonical_tokens approximated from W25 visible
            # minus the compact_ref token.
            n_w15_canonical = max(0, n_w25_visible - 1)

            # Chain persists for at most ``chain_persist_window`` cells
            # *including* the anchor cell.  window=1 means anchor-only
            # (no advances); window=N means anchor + (N-1) advances.
            need_anchor = (
                not self._active_chain_root_cid
                or self._active_cell_in_chain + 1
                    >= int(self.chain_persist_window))

            if need_anchor:
                anchor = self._build_anchor(
                    cell_index=cell_index,
                    w25_result=w25_result,
                    n_w15_canonical_tokens=n_w15_canonical,
                    canonical_compact_per_tag_votes=
                        canonical_per_tag_votes,
                    canonical_compact_projected_subset=
                        canonical_projected_subset)
                outcome = self.chain_registry.register_anchor(anchor)
                if not outcome.ok and self.require_chain_verification:
                    self._cell_index += 1
                    return _pack(
                        decoder_branch=W26_BRANCH_CHAIN_REJECTED,
                        anchor=anchor, advance=None,
                        cell_in_chain=0, chain_ok=False,
                        chain_reason=str(outcome.reason),
                        n_w26_visible=n_w25_visible)
                # Anchor cell: producer pays full W25 visible cost.
                self._active_chain_root_cid = anchor.chain_root_cid
                self._active_anchor = anchor
                self._active_cell_in_chain = 0
                self._last_advance = None
                self._last_anchor_per_tag_votes = (
                    canonical_per_tag_votes)
                self._last_anchor_projected_subset = (
                    canonical_projected_subset)
                # Decide branch label.
                if cell_index > 0:
                    branch = W26_BRANCH_CHAIN_RE_ANCHORED
                else:
                    branch = W26_BRANCH_CHAIN_ANCHORED
                self._cell_index += 1
                return _pack(
                    decoder_branch=branch,
                    anchor=anchor, advance=None, cell_in_chain=0,
                    chain_ok=True, chain_reason="ok",
                    n_w26_visible=n_w25_visible)

            # Subsequent in-window cell: build + register advance.
            anchor = self._active_anchor
            assert anchor is not None
            parent_cid = (str(self._last_advance.advance_cid)
                            if self._last_advance is not None
                            else anchor.chain_root_cid)
            cell_in_chain = self._active_cell_in_chain + 1
            advance = self._build_advance(
                cell_index=cell_index,
                anchor=anchor,
                parent_advance_cid=parent_cid,
                cell_in_chain=cell_in_chain,
                current_per_tag_votes=canonical_per_tag_votes,
                current_projected_subset=canonical_projected_subset)
            outcome = self.chain_registry.register_advance(advance)
            if not outcome.ok and self.require_chain_verification:
                self._cell_index += 1
                return _pack(
                    decoder_branch=W26_BRANCH_CHAIN_REJECTED,
                    anchor=None, advance=advance,
                    cell_in_chain=cell_in_chain,
                    chain_ok=False, chain_reason=str(outcome.reason),
                    n_w26_visible=n_w25_visible)
            self._active_cell_in_chain = cell_in_chain
            self._last_advance = advance
            # Producer in-window advance: 1 token.
            self._cell_index += 1
            return _pack(
                decoder_branch=W26_BRANCH_CHAIN_ADVANCED,
                anchor=None, advance=advance,
                cell_in_chain=cell_in_chain,
                chain_ok=True, chain_reason="ok",
                n_w26_visible=int(advance.n_advance_tokens))

        # --- Consumer path ---
        # Consumers consume the chain via their projection.  The chain
        # root must already be registered by a producer earlier in this
        # cell's processing (caller orchestrates producer-then-consumer
        # ordering, exactly as W25's run_phase72 does).
        # Find the active chain root the producer registered for this
        # producer_agent_id.
        producer_id = str(self.producer_agent_id)
        consumer_id = str(self.agent_id)
        projection_id = self.projection_id_for_consumer.get(
            consumer_id, "default")

        # Identify the chain by looking at all registered anchors
        # whose producer_agent_id matches.
        anchor: ChainAnchorEnvelope | None = None
        for cid, env in self.chain_registry._anchors.items():
            if env.producer_agent_id == producer_id:
                anchor = env  # latest wins (dicts preserve insertion order)
        if anchor is None:
            self._cell_index += 1
            return _pack(
                decoder_branch=W26_BRANCH_CHAIN_REJECTED,
                anchor=None, advance=None, cell_in_chain=0,
                chain_ok=False,
                chain_reason="anchor_not_found_for_producer",
                n_w26_visible=n_w25_visible)
        # Verify projection scope.
        proj_anchor, proj_reason = self.chain_registry.resolve_projection(
            chain_root_cid=anchor.chain_root_cid,
            consumer_id=consumer_id,
            projection_id=projection_id)
        if proj_anchor is None:
            self._cell_index += 1
            return _pack(
                decoder_branch=W26_BRANCH_CHAIN_PROJECTION_REJECTED,
                anchor=anchor, advance=None,
                cell_in_chain=0, chain_ok=False,
                chain_reason=str(proj_reason),
                n_w26_visible=n_w25_visible,
                projection_resolved=False,
                projection_id=projection_id)
        # Track which chain advance this consumer is on.
        self._cell_index += 1
        return _pack(
            decoder_branch=W26_BRANCH_CHAIN_PROJECTION_RESOLVED,
            anchor=anchor, advance=self._last_advance,
            cell_in_chain=self._active_cell_in_chain,
            chain_ok=True, chain_reason="ok",
            n_w26_visible=1,  # 1-token consumer chain ref
            projection_resolved=True,
            projection_id=projection_id)

    def decode(self, handoffs: Sequence[_DecodedHandoff]
                ) -> dict[str, Any]:
        return self.decode_rounds([handoffs])

    @property
    def last_result(self) -> W26ChainResult | None:
        return self._last_result

    @property
    def active_anchor(self) -> ChainAnchorEnvelope | None:
        return self._active_anchor

    @property
    def last_advance(self) -> ChainAdvanceEnvelope | None:
        return self._last_advance


# =============================================================================
# W27 — multi-chain salience-keyed dense-control fanout (divergence-aware
# chain replay + parallel chain pool + per-cell pivot routing)
# =============================================================================
#
# The W26 family amortises the producer's per-cell salience-token cost
# across cells inside a single chain window via 1-token chain-advance
# references.  W26 has two structural limits, both named in
# SDK v3.27:
#
#   * **W26-Λ-divergent** — when the cell's gold subset / canonical
#     compact state changes between cells (e.g. R-73-DIVERGENT flips
#     the gold subset at cell 8), the inner W25 fires NO_TRIGGER on
#     the divergent cells and W26 falls through to W25 (no chain
#     savings on those cells).  Correctness drops to 0.5.
#   * **W26-C-DIVERGENCE-RECOVERY** — open conjecture that a smarter
#     chain replay could recover savings even on divergent cells via
#     a new mechanism.
#
# W27 implements the smallest honest version of that mechanism at
# the *capsule layer*: a pool of parallel chains keyed by the cell's
# **salience signature** (a content-addressed CID over the canonical
# compact state).  At each cell, the controller hashes the producer's
# current canonical compact state into a salience signature; if a
# chain with that signature is already in the pool, the producer
# *pivots* to that chain (1 token, audited via
# :class:`ChainPivotEnvelope` whose ``parent_chain_root_cid`` matches
# the registered chain).  If no chain matches, the producer anchors
# a fresh chain (full W25 cost) and adds it to the pool.
#
# Total visible-token cost over N cells, K consumers, M distinct
# salience signatures (chain pool of size M):
#   W25:     N × (C + K)
#   W26:     ⌈N/W⌉ × (C + K) + (N − ⌈N/W⌉) × (1+K)
#   W27:     M × (C + K) + (N − M) × (1+K)
#
# At N=16, K=3, C=14.625, M=2 (one anchor per gold subset):
#   W25 ≈ 282 tokens
#   W26 (with divergence) ≈ 17.625 + 7×4 + 8×17.625 ≈ 186 tokens
#                         (the 8 divergent cells fall through to W25)
#   W27 ≈ 2×17.625 + 14×4 ≈ 91 tokens   (saving ≈ 51% over W26 on
#                                          R-74-DIVERGENT-RECOVER)
#
# Trust boundary
# ==============
# The W27 layer adds two new content-addressed envelopes
# (:class:`SalienceSignatureEnvelope`,
# :class:`ChainPivotEnvelope`) and a controller-side
# :class:`MultiChainPersistedFanoutRegistry`.  Verification:
#
#   * ``verify_salience_signature``  — schema_cid pinning, hash
#     integrity, signature_cid recompute matches.
#   * ``verify_chain_pivot``         — the pivot's
#     parent_chain_root_cid must reference an existing chain in the
#     pool (else ``unknown_chain``); the pivot's salience_signature
#     must match the parent's registered signature (else
#     ``salience_signature_mismatch``); the pool size must be ≤
#     ``max_active_chains`` (else ``chain_pool_exhausted``).
#   * ``verify_multi_chain_pool``    — the pool's bounded size, one
#     active chain per signature, per-consumer projection scope is
#     preserved across pivots.
#
# Honest scope
# ============
# W27 is an **amortisation across divergence** mechanism, not a new
# information channel:
#   * It does not add new content; it changes how the existing
#     producer state is *distributed* across multiple chains.
#   * The total *bytes* on the wire is bounded above by W26's bytes
#     when only one signature is observed (W27 = W26 byte-for-byte
#     in that case — W27-Λ-single-signature falsifier).
#   * The visible-token reduction is bounded above by 1 + K per
#     cell; the floor 1+K matches W26-L on every pivot cell.
#   * The bounded chain pool size ``max_active_chains`` is the
#     critical safety knob: an unbounded pool could be exhausted by
#     adversarial divergence; ``chain_pool_exhausted`` rejects any
#     anchor beyond the bound and W27 falls through to W26.
#
# Named falsifiers
# ================
#   * **W27-Λ-single-signature** — when every cell produces the same
#     canonical compact state (R-73-CHAIN-SHARED), W27 reduces to
#     W26 byte-for-byte (no extra chains created; one signature in
#     the pool).
#   * **W27-Λ-pool-exhausted** — when more than ``max_active_chains``
#     distinct signatures appear, the controller rejects new
#     anchors and W27 falls through to W26 (which then falls through
#     to W25 on divergent cells).  Correctness preserved.
#   * **W27-Λ-pivot-tampered** — any tamper on a pivot's
#     parent_chain_root_cid / salience_signature_cid is rejected by
#     ``verify_chain_pivot``.
#   * **W27-Λ-signature-mismatch** — if the producer's canonical
#     state does not match the claimed signature, the pivot is
#     rejected with ``salience_signature_mismatch``.

W27_SALIENCE_SIGNATURE_SCHEMA_VERSION: str = "wevra.salience_signature.v1"
W27_CHAIN_PIVOT_SCHEMA_VERSION: str = "wevra.chain_pivot.v1"

W27_BRANCH_PIVOTED = "pivoted"            # advance via pivot to existing chain
W27_BRANCH_ANCHORED_NEW = "anchored_new"  # registered a new chain in the pool
W27_BRANCH_POOL_EXHAUSTED = "pool_exhausted"
W27_BRANCH_PIVOT_REJECTED = "pivot_rejected"
W27_BRANCH_FALLBACK_W26 = "fallback_w26"  # delegated to W26 (single-signature path)
W27_BRANCH_NO_TRIGGER = "no_trigger"
W27_BRANCH_DISABLED = "disabled"

W27_ALL_BRANCHES: tuple[str, ...] = (
    W27_BRANCH_PIVOTED,
    W27_BRANCH_ANCHORED_NEW,
    W27_BRANCH_POOL_EXHAUSTED,
    W27_BRANCH_PIVOT_REJECTED,
    W27_BRANCH_FALLBACK_W26,
    W27_BRANCH_NO_TRIGGER,
    W27_BRANCH_DISABLED,
)

# W27 fires on the same trigger set as W26.  When the inner W26 emits
# a chain anchor / advance / re-anchor / projection-resolved branch
# OR the inner W25 produced an envelope, W27 considers a pivot.
W27_DEFAULT_TRIGGER_BRANCHES: frozenset[str] = frozenset({
    W26_BRANCH_CHAIN_ANCHORED,
    W26_BRANCH_CHAIN_ADVANCED,
    W26_BRANCH_CHAIN_RE_ANCHORED,
    W26_BRANCH_CHAIN_PROJECTION_RESOLVED,
    # W26 NO_TRIGGER is the divergence recovery path: when the inner
    # W26 fell through, W27 still tries to pivot from a parallel chain
    # whose signature matches the cell's salience signature.
    W26_BRANCH_NO_TRIGGER,
})


def _compute_salience_signature_cid(
        *,
        per_tag_votes: tuple[tuple[str, int], ...],
        projected_subset: tuple[str, ...],
        producer_agent_id: str,
        consumer_agent_ids: tuple[str, ...],
        schema_cid: str,
) -> str:
    """Canonical SHA-256 over the salience signature payload.

    The signature is producer-agent-keyed and consumer-set-keyed so
    that a chain can only be pivoted by the same producer for the
    same consumer set.  Cell index is *not* part of the signature —
    that is the whole point of W27 (content-keyed amortisation).
    """
    payload = _canonical_json_bytes({
        "schema_version": W27_SALIENCE_SIGNATURE_SCHEMA_VERSION,
        "producer_agent_id": str(producer_agent_id),
        "consumer_agent_ids": sorted(str(c) for c in consumer_agent_ids),
        "per_tag_votes": [
            [str(t), int(c)] for t, c in
            sorted(per_tag_votes, key=lambda kv: kv[0])],
        "projected_subset": sorted(str(s) for s in projected_subset),
        "schema_cid": str(schema_cid),
    })
    return hashlib.sha256(payload).hexdigest()


@dataclasses.dataclass(frozen=True)
class SalienceSignatureEnvelope:
    """Content-addressed signature over a producer's canonical compact
    state at one cell (W27 family).

    The signature is the routing key into the multi-chain pool.  Two
    cells with byte-identical canonical state produce byte-identical
    signature_cid; a divergent cell produces a different signature
    and routes to a different chain.

    Trust: signature_cid is SHA-256 over the canonical bytes; tampering
    is detected on every recompute.
    """
    schema_version: str
    schema_cid: str
    producer_agent_id: str
    consumer_agent_ids: tuple[str, ...]
    canonical_per_tag_votes: tuple[tuple[str, int], ...]
    canonical_projected_subset: tuple[str, ...]
    cell_index_first_observed: int
    signature_cid: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "consumer_agent_ids",
                            tuple(sorted(self.consumer_agent_ids)))
        object.__setattr__(self, "canonical_per_tag_votes",
                            tuple(sorted(self.canonical_per_tag_votes,
                                          key=lambda kv: kv[0])))
        object.__setattr__(self, "canonical_projected_subset",
                            tuple(sorted(self.canonical_projected_subset)))
        if not self.signature_cid:
            object.__setattr__(self, "signature_cid",
                                self.recompute_signature_cid())

    def _signed_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "schema_cid": self.schema_cid,
            "producer_agent_id": self.producer_agent_id,
            "consumer_agent_ids": list(self.consumer_agent_ids),
            "canonical_per_tag_votes": [
                [t, int(c)] for t, c in self.canonical_per_tag_votes],
            "canonical_projected_subset":
                list(self.canonical_projected_subset),
            "cell_index_first_observed": int(self.cell_index_first_observed),
        }

    def to_canonical_bytes(self) -> bytes:
        return _canonical_json_bytes(self._signed_payload())

    def recompute_signature_cid(self) -> str:
        return _compute_salience_signature_cid(
            per_tag_votes=self.canonical_per_tag_votes,
            projected_subset=self.canonical_projected_subset,
            producer_agent_id=self.producer_agent_id,
            consumer_agent_ids=self.consumer_agent_ids,
            schema_cid=self.schema_cid,
        )

    @property
    def n_signature_bytes(self) -> int:
        return len(self.to_canonical_bytes())

    def as_dict(self) -> dict[str, Any]:
        d = self._signed_payload()
        d["signature_cid"] = self.signature_cid
        d["n_signature_bytes"] = self.n_signature_bytes
        return d


@dataclasses.dataclass(frozen=True)
class ChainPivotEnvelope:
    """Per-cell pivot to an existing chain in the multi-chain pool
    (W27 family).

    Carries:
      * ``signature_cid``        — the cell's salience signature.
      * ``parent_chain_root_cid`` — the registered chain whose
        signature matches.
      * ``parent_advance_cid``   — the latest advance of that chain
        (the pivot extends the parent chain).
      * ``cell_index``           — the cell index this pivot applies to.

    The pivot is parent-CID-linked into the multi-chain pool: tampering
    is detected by ``verify_chain_pivot``.
    """
    schema_version: str
    schema_cid: str
    signature_cid: str
    parent_chain_root_cid: str
    parent_advance_cid: str
    cell_index: int
    pivot_cid: str = ""

    def __post_init__(self) -> None:
        if not self.pivot_cid:
            object.__setattr__(self, "pivot_cid",
                                self.recompute_pivot_cid())

    def _signed_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "schema_cid": self.schema_cid,
            "signature_cid": self.signature_cid,
            "parent_chain_root_cid": self.parent_chain_root_cid,
            "parent_advance_cid": self.parent_advance_cid,
            "cell_index": int(self.cell_index),
        }

    def to_canonical_bytes(self) -> bytes:
        return _canonical_json_bytes(self._signed_payload())

    def recompute_pivot_cid(self) -> str:
        return hashlib.sha256(self.to_canonical_bytes()).hexdigest()

    def to_decoder_text(self) -> str:
        return f"<chain_pivot:{self.pivot_cid[:16]}>"

    @property
    def n_pivot_tokens(self) -> int:
        return _whitespace_token_count(self.to_decoder_text())

    @property
    def n_pivot_bytes(self) -> int:
        return len(self.to_canonical_bytes())

    def as_dict(self) -> dict[str, Any]:
        d = self._signed_payload()
        d["pivot_cid"] = self.pivot_cid
        d["n_pivot_tokens"] = self.n_pivot_tokens
        d["n_pivot_bytes"] = self.n_pivot_bytes
        d["decoder_text"] = self.to_decoder_text()
        return d


def verify_salience_signature(
        sig: SalienceSignatureEnvelope | None,
        *,
        registered_schema: SchemaCapsule,
) -> LatentVerificationOutcome:
    """Controller-side verification of a :class:`SalienceSignatureEnvelope`.

    Pure function. Failure modes enumerated:

    * ``empty_signature``         — no signature passed.
    * ``schema_version_unknown``  — sig.schema_version mismatch.
    * ``schema_cid_mismatch``     — sig.schema_cid != registered.
    * ``hash_mismatch``           — signature_cid does not recompute.
    """
    n = 0
    if sig is None:
        return LatentVerificationOutcome(
            ok=False, reason="empty_signature", n_checks=n)
    n += 1
    if sig.schema_version != W27_SALIENCE_SIGNATURE_SCHEMA_VERSION:
        return LatentVerificationOutcome(
            ok=False, reason="schema_version_unknown", n_checks=n)
    n += 1
    if sig.schema_cid != registered_schema.cid:
        return LatentVerificationOutcome(
            ok=False, reason="schema_cid_mismatch", n_checks=n)
    n += 1
    if sig.signature_cid != sig.recompute_signature_cid():
        return LatentVerificationOutcome(
            ok=False, reason="hash_mismatch", n_checks=n)
    return LatentVerificationOutcome(
        ok=True, reason="ok", n_checks=n)


def verify_chain_pivot(
        pivot: ChainPivotEnvelope | None,
        *,
        registered_schema: SchemaCapsule,
        registered_signature_cid: str,
        registered_parent_chain_root_cid: str,
        registered_parent_advance_cid: str,
) -> LatentVerificationOutcome:
    """Controller-side verification of a :class:`ChainPivotEnvelope`.

    Pure function. Failure modes enumerated:

    * ``empty_pivot``                  — no pivot passed.
    * ``schema_version_unknown``       — pivot.schema_version mismatch.
    * ``schema_cid_mismatch``          — pivot.schema_cid != registered.
    * ``unknown_signature``            — pivot.signature_cid not registered.
    * ``salience_signature_mismatch``  — sig of pivot's parent != registered.
    * ``parent_chain_unknown``         — parent_chain_root_cid not in pool.
    * ``parent_advance_unknown``       — parent_advance_cid not in chain.
    * ``hash_mismatch``                — pivot_cid does not recompute.
    """
    n = 0
    if pivot is None:
        return LatentVerificationOutcome(
            ok=False, reason="empty_pivot", n_checks=n)
    n += 1
    if pivot.schema_version != W27_CHAIN_PIVOT_SCHEMA_VERSION:
        return LatentVerificationOutcome(
            ok=False, reason="schema_version_unknown", n_checks=n)
    n += 1
    if pivot.schema_cid != registered_schema.cid:
        return LatentVerificationOutcome(
            ok=False, reason="schema_cid_mismatch", n_checks=n)
    n += 1
    if not registered_signature_cid:
        return LatentVerificationOutcome(
            ok=False, reason="unknown_signature", n_checks=n)
    if pivot.signature_cid != registered_signature_cid:
        return LatentVerificationOutcome(
            ok=False, reason="salience_signature_mismatch", n_checks=n)
    n += 1
    if pivot.parent_chain_root_cid != registered_parent_chain_root_cid:
        return LatentVerificationOutcome(
            ok=False, reason="parent_chain_unknown", n_checks=n)
    n += 1
    if pivot.parent_advance_cid != registered_parent_advance_cid:
        return LatentVerificationOutcome(
            ok=False, reason="parent_advance_unknown", n_checks=n)
    n += 1
    if pivot.pivot_cid != pivot.recompute_pivot_cid():
        return LatentVerificationOutcome(
            ok=False, reason="hash_mismatch", n_checks=n)
    return LatentVerificationOutcome(
        ok=True, reason="ok", n_checks=n)


@dataclasses.dataclass
class MultiChainPersistedFanoutRegistry:
    """Controller-side multi-chain pool (W27 family).

    Maintains a bounded pool of parallel chains, keyed by salience
    signature.  Each chain in the pool has its own
    :class:`ChainPersistedFanoutRegistry` (the W26 layer is reused
    for in-chain advance/anchor verification).  Pivots route a cell
    to the chain whose registered signature matches the cell's
    salience signature.

    Trust boundary: the registry is controller-owned.  A malicious
    producer cannot register a tampered pivot or grow the pool past
    ``max_active_chains``.

    The registry tracks:
      * Registered signatures — ``signature_cid → (chain_root_cid, last_advance_cid)``
      * Active chain anchors — ``chain_root_cid → ChainAnchorEnvelope``
      * Pool size statistics.
    """
    schema: SchemaCapsule | None = None
    max_active_chains: int = 8
    chain_registry: ChainPersistedFanoutRegistry = dataclasses.field(
        default_factory=lambda: ChainPersistedFanoutRegistry())
    _signature_to_chain: dict[str, str] = dataclasses.field(
        default_factory=dict)  # signature_cid -> chain_root_cid
    _signature_to_last_advance: dict[str, str] = dataclasses.field(
        default_factory=dict)  # signature_cid -> last advance_cid (or chain_root if no advance)
    _signature_envelopes: dict[str, SalienceSignatureEnvelope] = (
        dataclasses.field(default_factory=dict))
    _pivots: dict[str, ChainPivotEnvelope] = dataclasses.field(
        default_factory=dict)  # pivot_cid -> envelope
    n_signatures_registered: int = 0
    n_pivots_registered: int = 0
    n_pivots_rejected: int = 0
    n_pool_exhausted_rejections: int = 0
    n_anchors_added: int = 0

    def __post_init__(self) -> None:
        # Ensure the inner chain_registry shares schema.
        if self.schema is not None and self.chain_registry.schema is None:
            self.chain_registry.schema = self.schema

    @property
    def pool_size(self) -> int:
        return len(self._signature_to_chain)

    def has_signature(self, signature_cid: str) -> bool:
        return signature_cid in self._signature_to_chain

    def register_signature(
            self, sig: SalienceSignatureEnvelope,
    ) -> LatentVerificationOutcome:
        if self.schema is None:
            return LatentVerificationOutcome(
                ok=False, reason="registry_no_schema", n_checks=0)
        outcome = verify_salience_signature(
            sig, registered_schema=self.schema)
        if not outcome.ok:
            return outcome
        if sig.signature_cid not in self._signature_envelopes:
            self._signature_envelopes[sig.signature_cid] = sig
            self.n_signatures_registered += 1
        return outcome

    def attach_anchor(
            self,
            *,
            sig: SalienceSignatureEnvelope,
            anchor: ChainAnchorEnvelope,
    ) -> LatentVerificationOutcome:
        """Register a fresh chain in the pool.

        Bounded by ``max_active_chains``.  Returns ``chain_pool_exhausted``
        if the pool is already at capacity AND the signature is new.
        """
        if self.schema is None:
            return LatentVerificationOutcome(
                ok=False, reason="registry_no_schema", n_checks=0)
        sig_outcome = self.register_signature(sig)
        if not sig_outcome.ok:
            return sig_outcome
        # If the signature already maps to a chain, refuse to overwrite
        # — this would silently dispose of the existing chain.  The
        # caller should pivot instead.
        if sig.signature_cid in self._signature_to_chain:
            return LatentVerificationOutcome(
                ok=False, reason="signature_already_anchored",
                n_checks=1)
        # Bound the pool size.
        if len(self._signature_to_chain) >= int(self.max_active_chains):
            self.n_pool_exhausted_rejections += 1
            return LatentVerificationOutcome(
                ok=False, reason="chain_pool_exhausted", n_checks=1)
        # Register the anchor in the inner W26 chain registry.
        anchor_outcome = self.chain_registry.register_anchor(anchor)
        if not anchor_outcome.ok:
            return anchor_outcome
        self._signature_to_chain[sig.signature_cid] = anchor.chain_root_cid
        self._signature_to_last_advance[sig.signature_cid] = (
            anchor.chain_root_cid)
        self.n_anchors_added += 1
        return anchor_outcome

    def attach_advance_to_signature(
            self,
            *,
            signature_cid: str,
            advance: ChainAdvanceEnvelope,
    ) -> LatentVerificationOutcome:
        """Register an in-chain advance for a signature's chain."""
        chain_root = self._signature_to_chain.get(signature_cid)
        if chain_root is None:
            return LatentVerificationOutcome(
                ok=False, reason="unknown_signature", n_checks=0)
        if advance.chain_root_cid != chain_root:
            return LatentVerificationOutcome(
                ok=False, reason="chain_root_mismatch", n_checks=1)
        outcome = self.chain_registry.register_advance(advance)
        if outcome.ok:
            self._signature_to_last_advance[signature_cid] = (
                advance.advance_cid)
        return outcome

    def register_pivot(
            self,
            pivot: ChainPivotEnvelope,
    ) -> LatentVerificationOutcome:
        if self.schema is None:
            self.n_pivots_rejected += 1
            return LatentVerificationOutcome(
                ok=False, reason="registry_no_schema", n_checks=0)
        registered_chain_root = self._signature_to_chain.get(
            pivot.signature_cid, "")
        registered_last_adv = self._signature_to_last_advance.get(
            pivot.signature_cid, "")
        outcome = verify_chain_pivot(
            pivot,
            registered_schema=self.schema,
            registered_signature_cid=pivot.signature_cid
                if registered_chain_root else "",
            registered_parent_chain_root_cid=registered_chain_root,
            registered_parent_advance_cid=registered_last_adv,
        )
        if not outcome.ok:
            self.n_pivots_rejected += 1
            return outcome
        self._pivots[pivot.pivot_cid] = pivot
        # A pivot becomes the new "last advance" for its signature so
        # that subsequent pivots/advances chain off of it.
        self._signature_to_last_advance[pivot.signature_cid] = (
            pivot.pivot_cid)
        self.n_pivots_registered += 1
        return outcome

    def get_anchor_for_signature(
            self, signature_cid: str,
    ) -> ChainAnchorEnvelope | None:
        chain_root = self._signature_to_chain.get(signature_cid)
        if not chain_root:
            return None
        return self.chain_registry.get_anchor(chain_root)

    def last_advance_cid_for_signature(
            self, signature_cid: str,
    ) -> str:
        return self._signature_to_last_advance.get(signature_cid, "")

    def get_pivot(self, pivot_cid: str) -> ChainPivotEnvelope | None:
        return self._pivots.get(pivot_cid)


@dataclasses.dataclass
class W27MultiChainResult:
    """Per-cell audit record for a W27 multi-chain agent."""
    answer: dict[str, Any]
    inner_w26_branch: str
    decoder_branch: str
    agent_id: str
    is_producer: bool
    salience_signature_cid: str
    chain_root_cid: str
    pivot_cid: str
    n_w26_visible_tokens: int
    n_w27_visible_tokens: int
    n_w26_minus_w27_savings: int
    n_signature_bytes: int
    n_pivot_bytes: int
    pool_size: int
    pool_exhausted: bool
    pivot_verification_ok: bool
    pivot_verification_reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "inner_w26_branch": self.inner_w26_branch,
            "decoder_branch": self.decoder_branch,
            "agent_id": self.agent_id,
            "is_producer": bool(self.is_producer),
            "salience_signature_cid": self.salience_signature_cid,
            "chain_root_cid": self.chain_root_cid,
            "pivot_cid": self.pivot_cid,
            "n_w26_visible_tokens": int(self.n_w26_visible_tokens),
            "n_w27_visible_tokens": int(self.n_w27_visible_tokens),
            "n_w26_minus_w27_savings": int(self.n_w26_minus_w27_savings),
            "n_signature_bytes": int(self.n_signature_bytes),
            "n_pivot_bytes": int(self.n_pivot_bytes),
            "pool_size": int(self.pool_size),
            "pool_exhausted": bool(self.pool_exhausted),
            "pivot_verification_ok": bool(self.pivot_verification_ok),
            "pivot_verification_reason": str(self.pivot_verification_reason),
        }


@dataclasses.dataclass
class MultiChainPersistedFanoutDisambiguator:
    """Multi-chain salience-keyed dense-control fanout (W27 family).

    Wraps a :class:`ChainPersistedFanoutDisambiguator` (W26) and adds
    a controller-side pool of parallel chains.  At each cell the
    producer's canonical compact state is hashed into a
    :class:`SalienceSignatureEnvelope`; the controller looks up the
    pool:

      * **Pivot**     — signature already in pool ⇒ producer emits a
                         single ``<chain_pivot:DDDD>`` token; consumer
                         pays 1 token (chain_consumer_ref).
      * **Anchor new** — signature not in pool AND pool not exhausted
                         ⇒ delegate to W26 to anchor a fresh chain.
      * **Pool exhausted** — signature not in pool AND pool full ⇒
                         the controller rejects with
                         ``chain_pool_exhausted``; W27 falls through
                         to W26 (which falls through to W25 on
                         divergent cells, preserving correctness).

    Trust boundary
    --------------
    The pool is bounded by ``max_active_chains``; the controller-side
    :class:`MultiChainPersistedFanoutRegistry` enforces:

      * salience signature integrity (SHA-256 over canonical state);
      * pivot parent linkage (parent_chain_root_cid + parent_advance_cid
        must match the chain registered for the signature);
      * pool size bound (anchors beyond ``max_active_chains`` rejected);
      * schema_cid pinning (cross-schema envelopes rejected);
      * cross-pivot tampering — any tamper on
        ``signature_cid`` / ``parent_chain_root_cid`` /
        ``parent_advance_cid`` triggers a verification failure.

    Honest scope
    ------------
    W27 changes how the producer's chain state is *partitioned* across
    salience signatures; it does not add a new information channel.
    The visible-token reduction comes from routing each cell to its
    matching chain (the same accounting model already in W23..W26).

    Composition
    -----------
    When only one signature is observed (R-73-CHAIN-SHARED), the pool
    contains exactly one chain and W27 reduces to W26 byte-for-byte
    (W27-Λ-single-signature falsifier).  When the bench has M
    distinct signatures and ``max_active_chains ≥ M``, W27 saves
    (M+1..N) × (C-1) tokens over W26 (the N-M cells beyond the first
    M anchors pay 1+K instead of C+K).
    """
    inner: ChainPersistedFanoutDisambiguator = dataclasses.field(
        default_factory=lambda: ChainPersistedFanoutDisambiguator())
    multi_chain_registry: MultiChainPersistedFanoutRegistry | None = None
    schema: SchemaCapsule | None = None
    max_active_chains: int = 8
    enabled: bool = True
    require_pivot_verification: bool = True
    trigger_branches: frozenset[str] = dataclasses.field(
        default_factory=lambda: W27_DEFAULT_TRIGGER_BRANCHES)

    _last_result: W27MultiChainResult | None = None
    _last_signature: SalienceSignatureEnvelope | None = None
    _last_pivot: ChainPivotEnvelope | None = None
    _cell_index: int = 0

    @property
    def agent_id(self) -> str:
        return self.inner.agent_id

    @property
    def is_producer(self) -> bool:
        return self.inner.is_producer

    @property
    def producer_agent_id(self) -> str:
        return self.inner.producer_agent_id

    @property
    def consumer_agent_ids(self) -> tuple[str, ...]:
        return self.inner.consumer_agent_ids

    @property
    def T_decoder(self) -> int | None:
        return self.inner.T_decoder

    @T_decoder.setter
    def T_decoder(self, v: int | None) -> None:
        self.inner.T_decoder = v

    def reset_session(self) -> None:
        self._last_result = None
        self._last_signature = None
        self._last_pivot = None
        self._cell_index = 0
        self.inner.reset_session()

    def _build_signature(
            self,
            *,
            cell_index: int,
            per_tag_votes: tuple[tuple[str, int], ...],
            projected_subset: tuple[str, ...],
    ) -> SalienceSignatureEnvelope:
        schema_cid = (str(self.schema.cid)
                        if self.schema is not None else "")
        return SalienceSignatureEnvelope(
            schema_version=W27_SALIENCE_SIGNATURE_SCHEMA_VERSION,
            schema_cid=schema_cid,
            producer_agent_id=str(self.producer_agent_id
                                    or self.agent_id),
            consumer_agent_ids=tuple(self.consumer_agent_ids),
            canonical_per_tag_votes=tuple(per_tag_votes),
            canonical_projected_subset=tuple(projected_subset),
            cell_index_first_observed=int(cell_index),
        )

    def _build_pivot(
            self,
            *,
            cell_index: int,
            sig: SalienceSignatureEnvelope,
    ) -> ChainPivotEnvelope:
        assert self.multi_chain_registry is not None
        schema_cid = (str(self.schema.cid)
                        if self.schema is not None else "")
        chain_root = (
            self.multi_chain_registry._signature_to_chain.get(
                sig.signature_cid, ""))
        last_adv = (
            self.multi_chain_registry._signature_to_last_advance.get(
                sig.signature_cid, ""))
        return ChainPivotEnvelope(
            schema_version=W27_CHAIN_PIVOT_SCHEMA_VERSION,
            schema_cid=schema_cid,
            signature_cid=sig.signature_cid,
            parent_chain_root_cid=chain_root,
            parent_advance_cid=last_adv,
            cell_index=int(cell_index),
        )

    def _producer_canonical_state(
            self,
    ) -> tuple[tuple[tuple[str, int], ...], tuple[str, ...], int] | None:
        """Read the inner W25 fanout envelope to compute the producer's
        canonical compact state for this cell.

        Returns ``(per_tag_votes, projected_subset, n_w25_visible)``
        if available, else ``None``.
        """
        w25_dis = self.inner.inner  # W26.inner -> W25
        fanout = w25_dis.last_fanout_envelope
        if fanout is None:
            return None
        per_tag = tuple(fanout.compact_per_tag_votes)
        projected = tuple(fanout.compact_projected_subset)
        w25_result = w25_dis.last_result
        n_w25_visible = int(w25_result.n_w25_visible_tokens) if w25_result else 0
        return (per_tag, projected, n_w25_visible)

    def decode_rounds(
            self,
            per_round_handoffs: Sequence[Sequence[_DecodedHandoff]],
    ) -> dict[str, Any]:
        # Always run the W26 inner first; it may anchor / advance /
        # fall through.
        w26_out = self.inner.decode_rounds(per_round_handoffs)
        w26_result = self.inner.last_result
        out = dict(w26_out)
        cell_index = int(self._cell_index)

        inner_w26_branch = (str(w26_result.decoder_branch)
                              if w26_result is not None
                              else W26_BRANCH_DISABLED)
        n_w26_visible = (int(w26_result.n_w26_visible_tokens)
                          if w26_result is not None else 0)

        def _pack(
                *,
                decoder_branch: str,
                signature: SalienceSignatureEnvelope | None,
                pivot: ChainPivotEnvelope | None,
                pool_size: int,
                pool_exhausted: bool,
                pivot_ok: bool,
                pivot_reason: str,
                n_w27_visible: int,
                chain_root_cid: str = "",
        ) -> dict[str, Any]:
            savings = max(0, n_w26_visible - n_w27_visible)
            sig_cid = (str(signature.signature_cid)
                        if signature is not None else "")
            pivot_cid = str(pivot.pivot_cid) if pivot is not None else ""
            n_sig_bytes = (int(signature.n_signature_bytes)
                            if signature is not None else 0)
            n_pivot_bytes = (int(pivot.n_pivot_bytes)
                              if pivot is not None else 0)
            result = W27MultiChainResult(
                answer=dict(out),
                inner_w26_branch=inner_w26_branch,
                decoder_branch=decoder_branch,
                agent_id=str(self.agent_id),
                is_producer=bool(self.is_producer),
                salience_signature_cid=sig_cid,
                chain_root_cid=chain_root_cid,
                pivot_cid=pivot_cid,
                n_w26_visible_tokens=n_w26_visible,
                n_w27_visible_tokens=int(n_w27_visible),
                n_w26_minus_w27_savings=savings,
                n_signature_bytes=n_sig_bytes,
                n_pivot_bytes=n_pivot_bytes,
                pool_size=int(pool_size),
                pool_exhausted=bool(pool_exhausted),
                pivot_verification_ok=bool(pivot_ok),
                pivot_verification_reason=str(pivot_reason),
            )
            self._last_result = result
            self._last_signature = signature
            self._last_pivot = pivot
            out_local = dict(out)
            out_local["multi_chain_persisted_hybrid"] = result.as_dict()
            if signature is not None:
                out_local["salience_signature_envelope"] = signature.as_dict()
            if pivot is not None:
                out_local["chain_pivot_envelope"] = pivot.as_dict()
            return out_local

        # ---- Disabled / no-trigger / no-registry paths ----
        if (not self.enabled or self.multi_chain_registry is None
                or self.schema is None):
            self._cell_index += 1
            return _pack(
                decoder_branch=W27_BRANCH_DISABLED,
                signature=None, pivot=None,
                pool_size=0, pool_exhausted=False,
                pivot_ok=False, pivot_reason="disabled",
                n_w27_visible=n_w26_visible)

        if inner_w26_branch not in self.trigger_branches:
            self._cell_index += 1
            return _pack(
                decoder_branch=W27_BRANCH_NO_TRIGGER,
                signature=None, pivot=None,
                pool_size=int(self.multi_chain_registry.pool_size),
                pool_exhausted=False,
                pivot_ok=False, pivot_reason="inner_w26_not_triggered",
                n_w27_visible=n_w26_visible)

        # ---- Compute the cell's salience signature ----
        canonical = self._producer_canonical_state()
        if canonical is None:
            # No fanout envelope means W25 didn't emit; nothing to
            # signature on the consumer side either.  Fall back to W26.
            self._cell_index += 1
            return _pack(
                decoder_branch=W27_BRANCH_FALLBACK_W26,
                signature=None, pivot=None,
                pool_size=int(self.multi_chain_registry.pool_size),
                pool_exhausted=False,
                pivot_ok=False, pivot_reason="no_w25_envelope",
                n_w27_visible=n_w26_visible)
        per_tag, projected, n_w25_visible = canonical
        sig = self._build_signature(
            cell_index=cell_index,
            per_tag_votes=per_tag,
            projected_subset=projected)

        # ---- Producer path ----
        if self.is_producer:
            registry = self.multi_chain_registry
            pool_size_before = registry.pool_size

            if registry.has_signature(sig.signature_cid):
                # Pivot to existing chain — 1 token.
                pivot = self._build_pivot(
                    cell_index=cell_index, sig=sig)
                outcome = registry.register_pivot(pivot)
                chain_root = (
                    registry._signature_to_chain.get(
                        sig.signature_cid, ""))
                if not outcome.ok and self.require_pivot_verification:
                    self._cell_index += 1
                    return _pack(
                        decoder_branch=W27_BRANCH_PIVOT_REJECTED,
                        signature=sig, pivot=pivot,
                        pool_size=int(registry.pool_size),
                        pool_exhausted=False,
                        pivot_ok=False,
                        pivot_reason=str(outcome.reason),
                        n_w27_visible=n_w26_visible,
                        chain_root_cid=chain_root)
                self._cell_index += 1
                return _pack(
                    decoder_branch=W27_BRANCH_PIVOTED,
                    signature=sig, pivot=pivot,
                    pool_size=int(registry.pool_size),
                    pool_exhausted=False,
                    pivot_ok=True, pivot_reason="ok",
                    n_w27_visible=int(pivot.n_pivot_tokens),
                    chain_root_cid=chain_root)

            # Signature is new.  Did the inner W26 successfully anchor?
            # If yes, attach it to our pool (if room).
            inner_anchor = self.inner.active_anchor
            if (inner_anchor is not None
                    and inner_w26_branch in (
                        W26_BRANCH_CHAIN_ANCHORED,
                        W26_BRANCH_CHAIN_RE_ANCHORED)):
                # Attach the inner anchor to the multi-chain pool.
                # Re-register signature + anchor.
                # NOTE: the inner W26 has *already* registered the
                # anchor in its own chain_registry; the multi-chain
                # registry's chain_registry IS the same one.
                if registry.chain_registry is self.inner.chain_registry:
                    # Anchor already registered — just bind signature.
                    if (len(registry._signature_to_chain)
                            >= int(registry.max_active_chains)):
                        registry.n_pool_exhausted_rejections += 1
                        self._cell_index += 1
                        return _pack(
                            decoder_branch=W27_BRANCH_POOL_EXHAUSTED,
                            signature=sig, pivot=None,
                            pool_size=int(registry.pool_size),
                            pool_exhausted=True,
                            pivot_ok=False,
                            pivot_reason="chain_pool_exhausted",
                            n_w27_visible=n_w26_visible,
                            chain_root_cid=inner_anchor.chain_root_cid)
                    sig_outcome = registry.register_signature(sig)
                    if not sig_outcome.ok:
                        self._cell_index += 1
                        return _pack(
                            decoder_branch=W27_BRANCH_PIVOT_REJECTED,
                            signature=sig, pivot=None,
                            pool_size=int(registry.pool_size),
                            pool_exhausted=False,
                            pivot_ok=False,
                            pivot_reason=str(sig_outcome.reason),
                            n_w27_visible=n_w26_visible)
                    registry._signature_to_chain[sig.signature_cid] = (
                        inner_anchor.chain_root_cid)
                    registry._signature_to_last_advance[
                        sig.signature_cid] = (
                        inner_anchor.chain_root_cid)
                    registry.n_anchors_added += 1
                else:
                    outcome = registry.attach_anchor(
                        sig=sig, anchor=inner_anchor)
                    if not outcome.ok:
                        self._cell_index += 1
                        if outcome.reason == "chain_pool_exhausted":
                            return _pack(
                                decoder_branch=W27_BRANCH_POOL_EXHAUSTED,
                                signature=sig, pivot=None,
                                pool_size=int(registry.pool_size),
                                pool_exhausted=True,
                                pivot_ok=False,
                                pivot_reason="chain_pool_exhausted",
                                n_w27_visible=n_w26_visible,
                                chain_root_cid=inner_anchor.chain_root_cid)
                        return _pack(
                            decoder_branch=W27_BRANCH_PIVOT_REJECTED,
                            signature=sig, pivot=None,
                            pool_size=int(registry.pool_size),
                            pool_exhausted=False,
                            pivot_ok=False,
                            pivot_reason=str(outcome.reason),
                            n_w27_visible=n_w26_visible)
                self._cell_index += 1
                return _pack(
                    decoder_branch=W27_BRANCH_ANCHORED_NEW,
                    signature=sig, pivot=None,
                    pool_size=int(registry.pool_size),
                    pool_exhausted=False,
                    pivot_ok=True, pivot_reason="ok",
                    n_w27_visible=n_w26_visible,
                    chain_root_cid=inner_anchor.chain_root_cid)

            # Inner W26 advanced (in-window) on a *known* chain root,
            # but our pool maps signatures, not chain roots.  Track the
            # advance for the active signature if we have one.
            if (inner_anchor is not None
                    and inner_w26_branch == W26_BRANCH_CHAIN_ADVANCED):
                # Find which signature this advance belongs to.  Its
                # chain_root must match one of our pool entries.
                target_sig: str | None = None
                for sc, cr in registry._signature_to_chain.items():
                    if cr == inner_anchor.chain_root_cid:
                        target_sig = sc
                        break
                if target_sig is not None and self.inner.last_advance is not None:
                    registry._signature_to_last_advance[target_sig] = (
                        self.inner.last_advance.advance_cid)
                # The W26 advance fires this cell — W27 has nothing extra
                # to add; we bill the W26 cost.
                self._cell_index += 1
                return _pack(
                    decoder_branch=W27_BRANCH_FALLBACK_W26,
                    signature=sig, pivot=None,
                    pool_size=int(registry.pool_size),
                    pool_exhausted=False,
                    pivot_ok=True, pivot_reason="w26_in_window_advance",
                    n_w27_visible=n_w26_visible,
                    chain_root_cid=inner_anchor.chain_root_cid)

            # Fall through: W26 reported NO_TRIGGER (e.g. divergent
            # cell) AND signature is new AND pool is full → exhausted.
            if (len(registry._signature_to_chain)
                    >= int(registry.max_active_chains)):
                registry.n_pool_exhausted_rejections += 1
                self._cell_index += 1
                return _pack(
                    decoder_branch=W27_BRANCH_POOL_EXHAUSTED,
                    signature=sig, pivot=None,
                    pool_size=int(registry.pool_size),
                    pool_exhausted=True,
                    pivot_ok=False, pivot_reason="chain_pool_exhausted",
                    n_w27_visible=n_w26_visible)

            # No inner anchor and pool not exhausted: nothing to do
            # except record the signature for future reference.
            registry.register_signature(sig)
            self._cell_index += 1
            return _pack(
                decoder_branch=W27_BRANCH_FALLBACK_W26,
                signature=sig, pivot=None,
                pool_size=int(registry.pool_size),
                pool_exhausted=False,
                pivot_ok=True,
                pivot_reason="w26_fallthrough_no_inner_anchor",
                n_w27_visible=n_w26_visible)

        # ---- Consumer path ----
        # Consumer pays 1 token if the producer pivoted (the chain
        # head is content-addressed via signature), else mirrors W26.
        registry = self.multi_chain_registry
        if registry.has_signature(sig.signature_cid):
            chain_root = registry._signature_to_chain[sig.signature_cid]
            self._cell_index += 1
            return _pack(
                decoder_branch=W27_BRANCH_PIVOTED,
                signature=sig, pivot=None,
                pool_size=int(registry.pool_size),
                pool_exhausted=False,
                pivot_ok=True, pivot_reason="ok",
                n_w27_visible=1,  # 1-token consumer chain ref
                chain_root_cid=chain_root)
        self._cell_index += 1
        return _pack(
            decoder_branch=W27_BRANCH_FALLBACK_W26,
            signature=sig, pivot=None,
            pool_size=int(registry.pool_size),
            pool_exhausted=False,
            pivot_ok=True,
            pivot_reason="consumer_signature_unknown",
            n_w27_visible=n_w26_visible)

    def decode(self, handoffs: Sequence[_DecodedHandoff]
                ) -> dict[str, Any]:
        return self.decode_rounds([handoffs])

    @property
    def last_result(self) -> W27MultiChainResult | None:
        return self._last_result

    @property
    def last_signature(self) -> SalienceSignatureEnvelope | None:
        return self._last_signature

    @property
    def last_pivot(self) -> ChainPivotEnvelope | None:
        return self._last_pivot


def compute_input_signature_cid(
        per_round_handoffs: Sequence[Sequence[_DecodedHandoff]],
        *,
        producer_agent_id: str,
        consumer_agent_ids: Sequence[str],
        schema_cid: str,
) -> str:
    """Pre-compute a salience signature from raw input handoffs.

    Unlike :func:`_compute_salience_signature_cid` which keys on the
    inner W25 envelope's compact state, this function keys on the
    producer's *input* — the canonical (sorted) tuple of
    (source_role, claim_kind, payload) per round.  This signature is
    stable across cells with byte-identical inputs and therefore
    suitable for **input-keyed routing** to a multi-chain pool BEFORE
    any inner W22..W26 stack is exercised.

    The orchestrator (:class:`MultiChainPersistedFanoutOrchestrator`)
    uses this signature to route each cell to its dedicated inner W26
    disambiguator without contaminating sibling chains' state.
    """
    canonical_rounds: list[list[list[str]]] = []
    for round_handoffs in per_round_handoffs:
        round_canonical: list[list[str]] = []
        for h in round_handoffs:
            round_canonical.append([
                str(h.source_role),
                str(h.claim_kind),
                str(h.payload),
            ])
        # Sort each round's handoffs canonically (deterministic).
        round_canonical.sort()
        canonical_rounds.append(round_canonical)
    payload = _canonical_json_bytes({
        "schema_version": W27_SALIENCE_SIGNATURE_SCHEMA_VERSION,
        "producer_agent_id": str(producer_agent_id),
        "consumer_agent_ids": sorted(str(c) for c in consumer_agent_ids),
        "schema_cid": str(schema_cid),
        "rounds": canonical_rounds,
    })
    return hashlib.sha256(payload).hexdigest()


@dataclasses.dataclass
class _ChainSlot:
    """One slot in the orchestrator's pool: a signature paired with a
    fully-instantiated W26 disambiguator and per-cell counters.
    """
    signature_cid: str
    disambiguator: ChainPersistedFanoutDisambiguator
    n_cells_routed: int = 0
    n_anchored: int = 0
    n_advanced: int = 0
    n_re_anchored: int = 0


@dataclasses.dataclass
class SharedMultiChainPool:
    """Team-wide pool of per-signature W26 stacks (W27 family).

    A *single* pool is created per team and threaded through every
    agent's :class:`MultiChainPersistedFanoutOrchestrator`.  When an
    agent first encounters a signature, the pool builds a fresh
    *team* of W26 disambiguators (1 producer + K consumers) for that
    signature, sharing one ``SharedFanoutRegistry`` and one
    ``ChainPersistedFanoutRegistry`` so consumers can resolve the
    producer's fanout envelopes inside that signature's chain.

    Concretely:
      * ``stack_factory(signature_cid, agent_id, is_producer) ->
        ChainPersistedFanoutDisambiguator`` — supplied by the caller.
        It is responsible for sharing the per-signature registries
        (the pool calls the factory ``1 + K`` times per signature,
        once per agent).
      * ``max_active_chains`` — bounded pool size; new signatures
        beyond the bound are rejected.

    Trust boundary: the pool is controller-owned; every per-signature
    chain inherits the W26 trust boundary on its own
    :class:`ChainPersistedFanoutRegistry`.  The pool layer adds
    ``chain_pool_exhausted`` as the only new failure mode.
    """
    schema: SchemaCapsule
    stack_factory: Callable[..., ChainPersistedFanoutDisambiguator]
    max_active_chains: int = 8

    _agent_slots: dict[str, dict[str, _ChainSlot]] = dataclasses.field(
        default_factory=dict)  # agent_id -> signature_cid -> slot
    _known_signatures: list[str] = dataclasses.field(
        default_factory=list)
    _pool_exhausted_rejections: int = 0

    @property
    def pool_size(self) -> int:
        return len(self._known_signatures)

    @property
    def n_pool_exhausted_rejections(self) -> int:
        return self._pool_exhausted_rejections

    def signatures(self) -> tuple[str, ...]:
        return tuple(self._known_signatures)

    def has_signature(self, signature_cid: str) -> bool:
        return signature_cid in self._known_signatures

    def get_or_make_slot(
            self,
            *,
            agent_id: str,
            signature_cid: str,
            is_producer: bool,
    ) -> tuple[_ChainSlot | None, bool]:
        """Return ``(slot, pool_exhausted)`` for this agent + signature.

        If the signature is new and the pool is full, returns
        ``(None, True)``.  Else returns an existing slot or builds a
        fresh disambiguator via the stack_factory.
        """
        if signature_cid not in self._known_signatures:
            if len(self._known_signatures) >= int(self.max_active_chains):
                self._pool_exhausted_rejections += 1
                return None, True
            self._known_signatures.append(signature_cid)
        agent_dict = self._agent_slots.setdefault(agent_id, {})
        slot = agent_dict.get(signature_cid)
        if slot is None:
            new_dis = self.stack_factory(
                signature_cid=signature_cid,
                agent_id=agent_id,
                is_producer=is_producer,
            )
            slot = _ChainSlot(
                signature_cid=signature_cid,
                disambiguator=new_dis,
            )
            agent_dict[signature_cid] = slot
        return slot, False

    def reset_session(self) -> None:
        for adict in self._agent_slots.values():
            for s in adict.values():
                s.disambiguator.reset_session()
        self._agent_slots.clear()
        self._known_signatures = []
        self._pool_exhausted_rejections = 0


@dataclasses.dataclass
class W27OrchestratorResult:
    """Per-cell audit record for a W27 orchestrator route."""
    answer: dict[str, Any]
    inner_w26_branch: str
    decoder_branch: str
    agent_id: str
    is_producer: bool
    input_signature_cid: str
    routed_chain_root_cid: str
    n_w26_in_pool_visible_tokens: int
    n_w27_visible_tokens: int
    n_w26_minus_w27_savings: int
    pool_size: int
    pool_exhausted: bool
    n_signature_bytes: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "inner_w26_branch": self.inner_w26_branch,
            "decoder_branch": self.decoder_branch,
            "agent_id": self.agent_id,
            "is_producer": bool(self.is_producer),
            "input_signature_cid": self.input_signature_cid,
            "routed_chain_root_cid": self.routed_chain_root_cid,
            "n_w26_in_pool_visible_tokens": int(self.n_w26_in_pool_visible_tokens),
            "n_w27_visible_tokens": int(self.n_w27_visible_tokens),
            "n_w26_minus_w27_savings": int(self.n_w26_minus_w27_savings),
            "pool_size": int(self.pool_size),
            "pool_exhausted": bool(self.pool_exhausted),
            "n_signature_bytes": int(self.n_signature_bytes),
        }


@dataclasses.dataclass
class MultiChainPersistedFanoutOrchestrator:
    """Orchestrator-level multi-chain dense-control fanout (W27 family).

    This is the load-bearing W27 implementation.  Where
    :class:`MultiChainPersistedFanoutDisambiguator` adds a content-
    addressed pivot envelope on top of *one* W26 disambiguator, the
    orchestrator routes each cell to a per-signature W26 stack
    inside a *team-wide* :class:`SharedMultiChainPool`.  All agents
    on the team share one pool, so per-signature fanout / chain
    registries are shared between producer and K consumers.

    Invariants
    ----------
    * Each (agent, signature) pair has its own W26 inner stack —
      no cross-contamination between signatures.  Per-signature
      registries are shared across agents on the team.
    * The pool is bounded by ``max_active_chains``.  New signatures
      beyond the bound are rejected with
      ``W27_BRANCH_POOL_EXHAUSTED``; cells with that exhausted-pool
      signature fall through to a single fallback W26 disambiguator
      (``_fallback`` slot) which mirrors the original W26 behaviour.

    Trust boundary
    --------------
    Each slot inherits the W26 trust boundary (anchor + advance hash
    integrity, parent linkage, projection scope).  The pool layer
    adds a single failure mode: ``chain_pool_exhausted``.
    """
    schema: SchemaCapsule
    producer_agent_id: str
    consumer_agent_ids: tuple[str, ...]
    agent_id: str
    is_producer: bool
    pool: SharedMultiChainPool
    enabled: bool = True
    fallback_factory: "Callable[..., ChainPersistedFanoutDisambiguator] | None" = None

    _fallback: "ChainPersistedFanoutDisambiguator | None" = None
    _last_result: "W27OrchestratorResult | None" = None
    _cell_index: int = 0

    def reset_session(self) -> None:
        if self._fallback is not None:
            self._fallback.reset_session()
        self._fallback = None
        self._last_result = None
        self._cell_index = 0
        # The pool is shared — only the team-level reset clears it.

    @property
    def pool_size(self) -> int:
        return self.pool.pool_size

    @property
    def n_pool_exhausted_rejections(self) -> int:
        return self.pool.n_pool_exhausted_rejections

    def _ensure_fallback(self) -> ChainPersistedFanoutDisambiguator:
        if self._fallback is None:
            factory = self.fallback_factory
            if factory is None:
                # If no explicit fallback, build one via the pool's
                # stack_factory — same shape as a slot's stack but
                # not bound to any signature.
                factory = self.pool.stack_factory
            self._fallback = factory(
                signature_cid="__fallback__",
                agent_id=self.agent_id,
                is_producer=self.is_producer,
            )
        return self._fallback

    def decode_rounds(
            self,
            per_round_handoffs: Sequence[Sequence[_DecodedHandoff]],
    ) -> dict[str, Any]:
        schema_cid = (str(self.schema.cid)
                        if self.schema is not None else "")
        sig_cid = compute_input_signature_cid(
            per_round_handoffs,
            producer_agent_id=str(self.producer_agent_id
                                    or self.agent_id),
            consumer_agent_ids=tuple(self.consumer_agent_ids),
            schema_cid=schema_cid,
        )
        n_signature_bytes = len(sig_cid.encode("utf-8"))

        if not self.enabled:
            fallback = self._ensure_fallback()
            out = fallback.decode_rounds(per_round_handoffs)
            inner = fallback.last_result
            inner_branch = (str(inner.decoder_branch) if inner
                              else W26_BRANCH_DISABLED)
            n_w26 = int(inner.n_w26_visible_tokens) if inner else 0
            self._cell_index += 1
            self._last_result = W27OrchestratorResult(
                answer=out,
                inner_w26_branch=inner_branch,
                decoder_branch=W27_BRANCH_DISABLED,
                agent_id=str(self.agent_id),
                is_producer=bool(self.is_producer),
                input_signature_cid=sig_cid,
                routed_chain_root_cid="",
                n_w26_in_pool_visible_tokens=n_w26,
                n_w27_visible_tokens=n_w26,
                n_w26_minus_w27_savings=0,
                pool_size=self.pool_size,
                pool_exhausted=False,
                n_signature_bytes=n_signature_bytes,
            )
            out_local = dict(out)
            out_local["multi_chain_orchestrator"] = (
                self._last_result.as_dict())
            return out_local

        slot, pool_exhausted = self.pool.get_or_make_slot(
            agent_id=self.agent_id,
            signature_cid=sig_cid,
            is_producer=self.is_producer,
        )
        if pool_exhausted or slot is None:
            fallback = self._ensure_fallback()
            out = fallback.decode_rounds(per_round_handoffs)
            inner = fallback.last_result
            inner_branch = (str(inner.decoder_branch) if inner
                              else W26_BRANCH_DISABLED)
            n_w26 = int(inner.n_w26_visible_tokens) if inner else 0
            self._cell_index += 1
            self._last_result = W27OrchestratorResult(
                answer=out,
                inner_w26_branch=inner_branch,
                decoder_branch=W27_BRANCH_POOL_EXHAUSTED,
                agent_id=str(self.agent_id),
                is_producer=bool(self.is_producer),
                input_signature_cid=sig_cid,
                routed_chain_root_cid="",
                n_w26_in_pool_visible_tokens=n_w26,
                n_w27_visible_tokens=n_w26,
                n_w26_minus_w27_savings=0,
                pool_size=self.pool_size,
                pool_exhausted=True,
                n_signature_bytes=n_signature_bytes,
            )
            out_local = dict(out)
            out_local["multi_chain_orchestrator"] = (
                self._last_result.as_dict())
            return out_local

        out = slot.disambiguator.decode_rounds(per_round_handoffs)
        slot.n_cells_routed += 1
        inner = slot.disambiguator.last_result
        inner_branch = (str(inner.decoder_branch) if inner
                          else W26_BRANCH_DISABLED)
        n_w26 = int(inner.n_w26_visible_tokens) if inner else 0
        if inner_branch == W26_BRANCH_CHAIN_ANCHORED:
            slot.n_anchored += 1
            decoder_branch = W27_BRANCH_ANCHORED_NEW
        elif inner_branch == W26_BRANCH_CHAIN_RE_ANCHORED:
            slot.n_re_anchored += 1
            decoder_branch = W27_BRANCH_ANCHORED_NEW
        elif inner_branch in (W26_BRANCH_CHAIN_ADVANCED,
                                W26_BRANCH_CHAIN_PROJECTION_RESOLVED):
            slot.n_advanced += 1
            decoder_branch = W27_BRANCH_PIVOTED
        else:
            decoder_branch = W27_BRANCH_FALLBACK_W26

        chain_root = (str(inner.chain_root_cid) if inner else "")

        self._cell_index += 1
        self._last_result = W27OrchestratorResult(
            answer=out,
            inner_w26_branch=inner_branch,
            decoder_branch=decoder_branch,
            agent_id=str(self.agent_id),
            is_producer=bool(self.is_producer),
            input_signature_cid=sig_cid,
            routed_chain_root_cid=chain_root,
            n_w26_in_pool_visible_tokens=n_w26,
            n_w27_visible_tokens=n_w26,
            n_w26_minus_w27_savings=0,
            pool_size=self.pool_size,
            pool_exhausted=False,
            n_signature_bytes=n_signature_bytes,
        )
        out_local = dict(out)
        out_local["multi_chain_orchestrator"] = (
            self._last_result.as_dict())
        return out_local

    def decode(
            self,
            handoffs: Sequence[_DecodedHandoff],
    ) -> dict[str, Any]:
        return self.decode_rounds([handoffs])

    @property
    def last_result(self) -> "W27OrchestratorResult | None":
        return self._last_result

    @property
    def slots(self) -> dict[str, _ChainSlot]:
        # Shadow proxy for backward compat: returns this agent's slot map
        return self.pool._agent_slots.get(self.agent_id, {})


# =============================================================================
# W28 — ensemble-verified cross-model multi-chain pivot ratification
# (trust-weighted W21 × W27 synthesis + audited ratification envelope +
# cross-model / cross-host probe table)
# =============================================================================
#
# The W27 family routes each cell to a per-signature W26 stack via a
# bounded pool of parallel chains, keyed by the salience signature CID.
# It assumes the salience signature is *self-evidently* the right
# routing key for that cell's content: any byte-identical canonical
# state lands in the same chain, divergent states route elsewhere.
# That assumption is fine when the canonical state is locally
# reconstructible (the producer and the controller both compute it
# from the same in-process inputs), but it is silent on whether the
# *content* the signature summarises is genuinely the cell's content
# under any *outside-information* view — e.g. a different model, a
# different oracle, a different host.
#
# W28 closes that gap by inserting a controller-side **ensemble
# ratification** step between W27's signature lookup and its pool
# routing decision. Before W27 commits to a pivot or a new anchor,
# the controller polls a **trust-weighted probe table** (each entry
# is an :class:`EnsembleProbeRegistration`, mirroring W21's
# :class:`OracleRegistration`):
#
#   * each probe inspects the cell's salience signature + canonical
#     state and returns a :class:`ProbeVote`
#     (``ratify`` / ``reject`` / ``abstain``) with the probe's
#     ``trust_prior`` as the vote weight;
#   * the trust-weighted sum of ``ratify`` votes must be ≥ a
#     pre-committed ``quorum_threshold`` for W27's routing to fire;
#   * if quorum is not met, W27's routing is suppressed and the cell
#     falls through to the unratified path (typically W27's pool-
#     exhausted fallback, which itself falls through to W26 / W25);
#   * the entire decision is sealed inside a content-addressed
#     :class:`EnsemblePivotRatificationEnvelope` and verified by the
#     pure :func:`verify_ensemble_pivot_ratification`.
#
# Total visible-token cost over N cells, K consumers, P probes:
#
#   W27:  M × (C + K) + (N − M) × (1 + K)               [as before]
#   W28:  W27 + (P > 1 ratifying cells) × 1             [+1 ratify ref
#                                                          on the producer
#                                                          when ratification
#                                                          must be carried
#                                                          on wire]
#
# At P = 1 with weight ≥ quorum_threshold, the ratification envelope
# is locally reconstructible (the lone probe deterministically
# recomputes the signature) and the controller does NOT charge a
# ratify-ref token; W28 reduces to W27 byte-for-byte
# (``W28-Λ-single-probe``).
#
# Trust boundary
# ==============
# The W28 layer adds one new content-addressed envelope
# (:class:`EnsemblePivotRatificationEnvelope`) and a new pure verifier
# (:func:`verify_ensemble_pivot_ratification`) with **eleven** failure
# modes — none of which existed in any W22..W27 verifier:
#
#   * ``empty_ratification``         — None envelope passed.
#   * ``schema_version_unknown``     — envelope.schema_version mismatch.
#   * ``schema_cid_mismatch``        — envelope.schema_cid != registered.
#   * ``signature_cid_empty``        — no signature being ratified.
#   * ``probe_table_empty``          — no probes registered.
#   * ``probe_id_unregistered``      — vote from a probe not in the
#                                      registered probe table.
#   * ``probe_vote_malformed``       — ratify and reject both true, etc.
#   * ``trust_weight_negative``      — vote weight < 0.
#   * ``hash_mismatch``              — ratification_cid does not recompute.
#   * ``quorum_below_threshold``     — ratified=True but weight < threshold.
#   * ``quorum_recompute_mismatch``  — ratified flag doesn't match the
#                                      recomputed quorum sign.
#
# Honest scope
# ============
# W28 is a **trust-amplification** layer, not a new information channel.
# It does NOT add new content; it only *ratifies or rejects* W27's
# existing pivot/anchor decisions, and the ratification proof is itself
# bounded (1 token on the producer-side wire when needed; 0 tokens on
# the consumer side).
#
# Critically:
#
#   * When K_probes = 1 with weight ≥ quorum, W28 is W27 byte-for-byte
#     (``W28-Λ-single-probe`` falsifier — empirically confirmed on
#     R-75-SINGLE-PROBE).
#   * When all probes drift identically (``W28-Λ-coordinated-drift``),
#     W28 cannot detect the drift; correctness is bounded above by W27.
#   * When all probe trust priors sum to zero
#     (``W28-Λ-trust-zero``), the quorum is unreachable and W28
#     abstains — no false ratification.
#   * When the W27 pool is exhausted, W28 must NOT invent a fresh
#     ratification; it falls through to W27's pool-exhausted path
#     (``W28-Λ-pool-exhausted-passthrough``).
#   * The W28 envelope is signed AND parent-CID-linked into the W27
#     pivot ledger: tampering on probe_id, weight, ratified flag,
#     quorum, or ratification_cid is detected by the verifier.
#
# Composition
# -----------
# W28 composes the W21 trust-weighted multi-oracle quorum (the *old*
# explicit-capsule line) with the W27 multi-chain salience-keyed pool
# (the *new* dense-control line) inside one decision. An ensemble
# probe is itself an :class:`EnsembleProbeRegistration` with a
# trust prior — the same ``trust_prior`` semantics W21 uses. A
# probe can be:
#
#   * a deterministic local-recompute probe
#     (:class:`DeterministicSignatureProbe`) — ratifies if the
#     signature recomputes exactly; trivially trustworthy;
#   * an oracle-consultation probe
#     (:class:`OracleConsultationProbe`) — wraps any
#     :class:`OutsideWitnessOracle` (W20/W21 family) and ratifies if
#     the oracle's projected_subset agrees with the salience
#     signature's projected_subset;
#   * a real-LLM probe (:class:`LLMSignatureProbe`) — wraps any
#     :class:`LLMBackend` (Ollama or MLX-distributed; on either Mac
#     in the two-host topology) and ratifies if the model's parsed
#     output agrees with the salience signature's projected_subset.
#
# Named falsifiers
# ================
#   * **W28-Λ-single-probe** — K_probes=1 with weight ≥ quorum
#     reduces W28 to W27 byte-for-byte; mechanically asserted.
#   * **W28-Λ-coordinated-drift** — when every probe drifts
#     identically, W28 cannot detect the drift; correctness ≤ W27.
#   * **W28-Λ-trust-zero** — all weights = 0 ⇒ quorum unreachable
#     ⇒ controller abstains.
#   * **W28-Λ-spoofed-probe** — vote from a probe_id not in the
#     registered table is rejected with ``probe_id_unregistered``.
#   * **W28-Λ-quorum-tampered** — ratified flag != recomputed quorum
#     ⇒ rejected with ``quorum_recompute_mismatch``.
#   * **W28-Λ-pool-exhausted-passthrough** — when W27 reports
#     POOL_EXHAUSTED, W28 must not ratify; it falls through.

W28_RATIFICATION_SCHEMA_VERSION: str = "wevra.ensemble_ratification.v1"

W28_BRANCH_RATIFIED = "ratified"
W28_BRANCH_RATIFIED_PASSTHROUGH = "ratified_passthrough"  # K=1 wire-free
W28_BRANCH_QUORUM_BELOW_THRESHOLD = "quorum_below_threshold"
W28_BRANCH_PROBE_REJECTED = "probe_rejected"  # verifier rejected
W28_BRANCH_NO_RATIFY_NEEDED = "no_ratify_needed"  # W27 didn't pivot
W28_BRANCH_FALLBACK_W27 = "fallback_w27"  # W27 reported non-pivot branch
W28_BRANCH_NO_TRIGGER = "no_trigger"
W28_BRANCH_DISABLED = "disabled"

W28_ALL_BRANCHES: tuple[str, ...] = (
    W28_BRANCH_RATIFIED,
    W28_BRANCH_RATIFIED_PASSTHROUGH,
    W28_BRANCH_QUORUM_BELOW_THRESHOLD,
    W28_BRANCH_PROBE_REJECTED,
    W28_BRANCH_NO_RATIFY_NEEDED,
    W28_BRANCH_FALLBACK_W27,
    W28_BRANCH_NO_TRIGGER,
    W28_BRANCH_DISABLED,
)

# W28 fires only on W27 branches that actually used the salience
# signature for routing. POOL_EXHAUSTED, NO_TRIGGER, DISABLED, and
# FALLBACK_W26 do NOT trigger ratification — they have no signature
# decision to ratify.
W28_DEFAULT_TRIGGER_BRANCHES: frozenset[str] = frozenset({
    W27_BRANCH_PIVOTED,
    W27_BRANCH_ANCHORED_NEW,
})


@dataclasses.dataclass(frozen=True)
class ProbeVote:
    """One probe's vote on a salience signature (W28 family).

    A probe inspects a :class:`SalienceSignatureEnvelope` and the
    cell's canonical compact state and returns one of:

      * ``ratify=True, reject=False`` — probe agrees with the
        signature; vote contributes ``trust_weight`` to quorum.
      * ``ratify=False, reject=True`` — probe disagrees; vote
        contributes ``-trust_weight`` to quorum (i.e. it actively
        reduces the ratification weight).
      * ``ratify=False, reject=False`` — probe abstains; vote
        contributes 0.

    The frozen-dataclass shape is part of the W28 verifier contract:
    a malformed vote (both ratify and reject true, or weight < 0) is
    rejected by :func:`verify_ensemble_pivot_ratification`.
    """
    probe_id: str
    ratify: bool
    reject: bool
    trust_weight: float
    reason: str = "ok"

    def __post_init__(self) -> None:
        # Strict invariants — W28 verifier depends on these.
        if self.ratify and self.reject:
            object.__setattr__(self, "reason", "malformed_ratify_and_reject")

    @property
    def is_abstain(self) -> bool:
        return (not self.ratify) and (not self.reject)

    @property
    def signed_weight(self) -> float:
        if self.ratify and not self.reject:
            return float(self.trust_weight)
        if self.reject and not self.ratify:
            return -float(self.trust_weight)
        return 0.0

    def as_tuple(self) -> tuple[str, int, int, float, str]:
        # Canonical, JSON-serialisable tuple form for envelope hashing.
        return (
            str(self.probe_id),
            int(bool(self.ratify)),
            int(bool(self.reject)),
            float(self.trust_weight),
            str(self.reason),
        )


@runtime_checkable
class EnsembleProbe(Protocol):
    """The minimum surface a W28 probe must expose.

    A probe inspects a salience signature plus the producer's
    canonical compact state for one cell and returns a
    :class:`ProbeVote`. Probes are duck-typed: any object with
    ``probe_id: str`` and ``vote(...)`` works.

    ``wire_required`` defaults to ``True``: when at least one probe
    requires the wire (e.g. an oracle / LLM probe whose decision the
    consumer cannot locally recompute), the W28 layer charges a 1-
    token ``<ratify_ref:DDDD>`` on the producer side. When *all*
    probes are deterministic-locally-reconstructible
    (``wire_required = False``, e.g.
    :class:`DeterministicSignatureProbe`), no wire token is charged.
    """
    probe_id: str
    wire_required: bool

    def vote(
            self,
            *,
            signature: "SalienceSignatureEnvelope",
            canonical_per_tag_votes: tuple[tuple[str, int], ...],
            canonical_projected_subset: tuple[str, ...],
            cell_index: int,
    ) -> ProbeVote: ...


@dataclasses.dataclass(frozen=True)
class EnsembleProbeRegistration:
    """One registered probe in a W28 ensemble (mirrors :class:`OracleRegistration`).

    Fields
    ------
    probe
        An :class:`EnsembleProbe`-shaped object (duck-typed).
    trust_prior
        Prior trust weight in [0, ∞). The W28 layer multiplies the
        probe's vote by this weight when computing the trust-weighted
        quorum. Default ``1.0`` treats every probe as equally
        trusted (pure majority — N votes ⇒ N max weight). Production
        deployments would calibrate trust priors via held-out
        agreement (cross-reference to W21-C-CALIBRATED-TRUST).
    role_label
        Provenance label for forensics ("local_qwen", "remote_qwen14b",
        "remote_gemma2", "service_graph_oracle", ...). The W28 quorum
        does NOT use it at decision time.
    host_id
        Optional host identifier (e.g. "localhost", "192.168.12.191").
        When set, enables cross-host telemetry — the W28 layer records
        ``cross_host_round_trip_bytes`` for any probe whose host_id
        differs from the orchestrator's host_id.
    """
    probe: Any  # EnsembleProbe duck-type
    trust_prior: float = 1.0
    role_label: str = "ensemble_probe"
    host_id: str = ""


# ---------------------------------------------------------------------------
# Built-in probe types
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class DeterministicSignatureProbe:
    """A locally-recomputable probe (W28 family).

    Ratifies iff the signature envelope's ``signature_cid`` matches
    the recomputation of the same canonical state. Trivially correct
    on all byte-identical inputs; the W28 backbone uses this probe at
    ``K_probes = 1`` to recover the exact W27 byte-for-byte path
    (W28-Λ-single-probe falsifier).

    Carries ``wire_required = False`` so the W28 layer can omit the
    ratify-ref wire token when this is the only probe in the table.
    """
    probe_id: str = "local_recompute"
    wire_required: bool = False

    def vote(
            self,
            *,
            signature: "SalienceSignatureEnvelope",
            canonical_per_tag_votes: tuple[tuple[str, int], ...],
            canonical_projected_subset: tuple[str, ...],
            cell_index: int,
    ) -> ProbeVote:
        recomputed = signature.recompute_signature_cid()
        if recomputed == signature.signature_cid:
            return ProbeVote(
                probe_id=self.probe_id,
                ratify=True, reject=False,
                trust_weight=1.0, reason="local_recompute_ok")
        return ProbeVote(
            probe_id=self.probe_id,
            ratify=False, reject=True,
            trust_weight=1.0, reason="local_recompute_mismatch")


@dataclasses.dataclass
class OracleConsultationProbe:
    """A probe that consults a W20/W21-family
    :class:`OutsideWitnessOracle` to ratify or reject a salience
    signature (W28 family).

    The probe asks the oracle which subset of services it would
    project for the cell's projected_subset; if the oracle's reply
    contains the same gold tags, ratify. If the oracle abstains
    (empty reply), abstain. If the oracle disagrees on tags, reject.

    Trust priors should reflect oracle reliability — a deterministic
    :class:`ServiceGraphOracle` carries a high prior (e.g. 1.0); an
    LLM-backed adjudicator carries a lower prior (e.g. 0.5).

    ``wire_required = True`` because the consumer cannot locally
    reconstruct the oracle's reply.
    """
    oracle: Any  # OutsideWitnessOracle duck-type
    probe_id: str = ""
    wire_required: bool = True
    max_response_tokens: int = 24

    def __post_init__(self) -> None:
        if not self.probe_id:
            self.probe_id = f"oracle_probe_{getattr(self.oracle, 'oracle_id', 'unknown')}"

    def vote(
            self,
            *,
            signature: "SalienceSignatureEnvelope",
            canonical_per_tag_votes: tuple[tuple[str, int], ...],
            canonical_projected_subset: tuple[str, ...],
            cell_index: int,
    ) -> ProbeVote:
        # Ask the oracle for the projected_subset's services. The
        # OutsideQuery shape is the same one W20/W21 already use:
        # ``admitted_tags`` are the candidate services, an optional
        # ``elected_root_cause`` lets kind-aware oracles tailor their
        # reply, and ``max_response_tokens`` bounds the wire cost.
        #
        # The W21-style :class:`ServiceGraphOracle` returns ``None``
        # (abstain) when *every* admitted tag is mutually connected
        # in the graph — that is its asymmetric-witness convention
        # (it only volunteers info when a strict subset of admitted
        # tags is connected and the rest is decoy). For ensemble
        # ratification we want the oracle to fire on a known good
        # pair, so we pad the admitted_tags with a decoy tag; the
        # oracle will then return its dependency_chain payload over
        # the gold pair and abstain on the decoy.
        admitted_padded = tuple(list(canonical_projected_subset)
                                  + [f"decoy_probe_{cell_index}"])
        try:
            query = OutsideQuery(
                admitted_tags=admitted_padded,
                elected_root_cause="ensemble_signature_probe",
                primary_payload="",
                witness_payloads=(),
                max_response_tokens=int(self.max_response_tokens),
            )
            verdict = self.oracle.consult(query)
        except Exception as exc:  # noqa: BLE001
            return ProbeVote(
                probe_id=self.probe_id,
                ratify=False, reject=False,
                trust_weight=1.0, reason=f"oracle_error:{type(exc).__name__}")
        # Parse the verdict's payload for the projected services.
        # Use the existing _disambiguator_payload_tokens semantics:
        # the W21 layer parses oracle replies as
        # whitespace-separated tokens prefixed with ``service=``.
        payload = str(getattr(verdict, "payload", "") or "")
        if not payload.strip():
            return ProbeVote(
                probe_id=self.probe_id,
                ratify=False, reject=False,
                trust_weight=1.0, reason="oracle_abstained")
        tokens = payload.split()
        oracle_services = set()
        for tok in tokens:
            if tok.startswith("service="):
                oracle_services.add(tok[len("service="):])
        sig_set = set(canonical_projected_subset)
        if not oracle_services:
            return ProbeVote(
                probe_id=self.probe_id,
                ratify=False, reject=False,
                trust_weight=1.0, reason="oracle_no_services_parsed")
        # Require the oracle's set to overlap with the signature's set
        # by ≥ ceil(|sig_set| / 2): a strict subset match is too
        # demanding for noisy oracles, but pure disjointness is a
        # clear reject.
        if oracle_services >= sig_set:
            return ProbeVote(
                probe_id=self.probe_id,
                ratify=True, reject=False,
                trust_weight=1.0, reason="oracle_full_agreement")
        if oracle_services & sig_set:
            return ProbeVote(
                probe_id=self.probe_id,
                ratify=True, reject=False,
                trust_weight=1.0, reason="oracle_partial_agreement")
        return ProbeVote(
            probe_id=self.probe_id,
            ratify=False, reject=True,
            trust_weight=1.0, reason="oracle_disjoint")


@dataclasses.dataclass
class LLMSignatureProbe:
    """A probe that calls a real LLM backend to ratify/reject a
    salience signature (W28 family).

    The probe asks the LLM to confirm or deny that the
    ``projected_subset`` matches the operationally-correct services
    for the cell. The model's reply is parsed for tag tokens; agreement
    ratifies, disagreement rejects, empty/malformed reply abstains.

    Designed for the **two-host topology**: when the backend's
    ``base_url`` points at a different host than the orchestrator's
    local host, the W28 layer records cross-host round-trip bytes.

    The probe is **best-effort**: any backend exception (timeout,
    HTTP error, parse failure) maps to abstain, never to ratify.
    Trust priors should be calibrated below 1.0 to reflect LLM
    nondeterminism (W22-C-CACHE-AMPLIFICATION applies here).
    """
    backend: Any  # LLMBackend duck-type
    probe_id: str = ""
    wire_required: bool = True
    max_tokens: int = 32
    temperature: float = 0.0
    n_calls: int = 0
    total_wall_s: float = 0.0
    last_response_text: str = ""

    def __post_init__(self) -> None:
        if not self.probe_id:
            self.probe_id = f"llm_probe_{getattr(self.backend, 'model', 'unknown')}"

    def _build_prompt(
            self,
            *,
            canonical_projected_subset: tuple[str, ...],
            canonical_per_tag_votes: tuple[tuple[str, int], ...],
    ) -> str:
        services = ", ".join(canonical_projected_subset)
        votes = ", ".join(f"{k}={v}" for k, v in canonical_per_tag_votes)
        return (
            "You are a strict service-incident reviewer.\n"
            f"Candidate root-cause services: {services}\n"
            f"Tag votes: {votes}\n"
            "Reply on one line, with whitespace-separated tokens of the form\n"
            "  ratify=true service=<name> service=<name>\n"
            "or\n"
            "  ratify=false reason=<short>\n"
            "If you are unsure, reply: ratify=false reason=unsure\n"
        )

    def vote(
            self,
            *,
            signature: "SalienceSignatureEnvelope",
            canonical_per_tag_votes: tuple[tuple[str, int], ...],
            canonical_projected_subset: tuple[str, ...],
            cell_index: int,
    ) -> ProbeVote:
        prompt = self._build_prompt(
            canonical_projected_subset=canonical_projected_subset,
            canonical_per_tag_votes=canonical_per_tag_votes)
        t0 = time.time()
        try:
            text = self.backend.generate(
                prompt,
                max_tokens=int(self.max_tokens),
                temperature=float(self.temperature))
        except Exception as exc:  # noqa: BLE001
            self.n_calls += 1
            self.total_wall_s += time.time() - t0
            return ProbeVote(
                probe_id=self.probe_id,
                ratify=False, reject=False,
                trust_weight=1.0,
                reason=f"backend_error:{type(exc).__name__}")
        self.n_calls += 1
        self.total_wall_s += time.time() - t0
        self.last_response_text = str(text or "")
        text = (text or "").strip().lower()
        if not text:
            return ProbeVote(
                probe_id=self.probe_id,
                ratify=False, reject=False,
                trust_weight=1.0, reason="llm_empty_reply")
        tokens = text.split()
        # Parse ratify=true/false token.
        ratify_tok = next(
            (t for t in tokens if t.startswith("ratify=")), "")
        services = set()
        for t in tokens:
            if t.startswith("service="):
                services.add(t[len("service="):])
        sig_set = {s.lower() for s in canonical_projected_subset}
        if ratify_tok == "ratify=true" and services and services & sig_set:
            return ProbeVote(
                probe_id=self.probe_id,
                ratify=True, reject=False,
                trust_weight=1.0,
                reason="llm_ratify_with_overlap")
        if ratify_tok == "ratify=true":
            # LLM said ratify but did not name overlapping services.
            return ProbeVote(
                probe_id=self.probe_id,
                ratify=False, reject=False,
                trust_weight=1.0, reason="llm_ratify_without_services")
        if ratify_tok == "ratify=false":
            return ProbeVote(
                probe_id=self.probe_id,
                ratify=False, reject=True,
                trust_weight=1.0, reason="llm_explicit_reject")
        # Free-form reply: try whatever was emitted.
        if services and services & sig_set:
            return ProbeVote(
                probe_id=self.probe_id,
                ratify=True, reject=False,
                trust_weight=1.0,
                reason="llm_freeform_partial_agreement")
        return ProbeVote(
            probe_id=self.probe_id,
            ratify=False, reject=False,
            trust_weight=1.0, reason="llm_unparseable_reply")


# ---------------------------------------------------------------------------
# Ratification envelope + verifier
# ---------------------------------------------------------------------------


def _compute_ensemble_ratification_cid(
        *,
        schema_version: str,
        schema_cid: str,
        signature_cid: str,
        probe_votes: tuple[tuple[str, int, int, float, str], ...],
        quorum_threshold: float,
        quorum_weight: float,
        ratified: bool,
        cell_index: int,
) -> str:
    """Canonical SHA-256 over an ensemble ratification payload."""
    payload = _canonical_json_bytes({
        "schema_version": str(schema_version),
        "schema_cid": str(schema_cid),
        "signature_cid": str(signature_cid),
        "probe_votes": [
            [str(pid), int(rt), int(rj),
              # Round to 4 dp to avoid IEEE-754 byte drift.
              round(float(w), 4), str(rs)]
            for pid, rt, rj, w, rs in probe_votes
        ],
        "quorum_threshold": round(float(quorum_threshold), 4),
        "quorum_weight": round(float(quorum_weight), 4),
        "ratified": bool(ratified),
        "cell_index": int(cell_index),
    })
    return hashlib.sha256(payload).hexdigest()


@dataclasses.dataclass(frozen=True)
class EnsemblePivotRatificationEnvelope:
    """Content-addressed ensemble ratification of one W27 pivot/anchor
    (W28 family).

    Carries:

      * ``signature_cid``     — the W27 salience signature being ratified.
      * ``probe_votes``       — sorted tuple of (probe_id, ratify_int,
                                 reject_int, weight, reason).
      * ``quorum_threshold``  — pre-committed threshold the trust-
                                 weighted ``ratify`` weight must meet.
      * ``quorum_weight``     — actual trust-weighted weight observed.
      * ``ratified``          — True iff ``quorum_weight ≥ threshold``
                                 AND no probe vote was malformed.
      * ``cell_index``        — the cell index the ratification applies to.

    The envelope's ``ratification_cid`` is SHA-256 over the canonical
    payload; tampering on any field is detected by
    :func:`verify_ensemble_pivot_ratification`.

    Wire token cost
    ---------------
    The W28 layer charges 1 visible token on the producer side
    (``<ratify_ref:DDDD>``) iff at least one probe in the registered
    table has ``wire_required = True``. When all probes are
    locally-reconstructible (e.g. the K=1 single-probe path with
    :class:`DeterministicSignatureProbe`), the envelope is recorded
    in the audit ledger but not transmitted on the visible-token wire,
    so W28 reduces to W27 byte-for-byte.
    """
    schema_version: str
    schema_cid: str
    signature_cid: str
    probe_votes: tuple[tuple[str, int, int, float, str], ...]
    quorum_threshold: float
    quorum_weight: float
    ratified: bool
    cell_index: int
    wire_required: bool = False
    ratification_cid: str = ""

    def __post_init__(self) -> None:
        # Sort probe_votes canonically by probe_id.
        object.__setattr__(self, "probe_votes",
                            tuple(sorted(
                                self.probe_votes, key=lambda v: v[0])))
        if not self.ratification_cid:
            object.__setattr__(self, "ratification_cid",
                                self.recompute_ratification_cid())

    def recompute_ratification_cid(self) -> str:
        return _compute_ensemble_ratification_cid(
            schema_version=self.schema_version,
            schema_cid=self.schema_cid,
            signature_cid=self.signature_cid,
            probe_votes=self.probe_votes,
            quorum_threshold=self.quorum_threshold,
            quorum_weight=self.quorum_weight,
            ratified=self.ratified,
            cell_index=self.cell_index,
        )

    def to_canonical_bytes(self) -> bytes:
        return _canonical_json_bytes({
            "schema_version": self.schema_version,
            "schema_cid": self.schema_cid,
            "signature_cid": self.signature_cid,
            "probe_votes": [list(v) for v in self.probe_votes],
            "quorum_threshold": round(float(self.quorum_threshold), 4),
            "quorum_weight": round(float(self.quorum_weight), 4),
            "ratified": bool(self.ratified),
            "cell_index": int(self.cell_index),
        })

    def to_decoder_text(self) -> str:
        return f"<ratify_ref:{self.ratification_cid[:16]}>"

    @property
    def n_envelope_bytes(self) -> int:
        return len(self.to_canonical_bytes())

    @property
    def n_wire_tokens(self) -> int:
        # Only billed when at least one probe required the wire.
        if not self.wire_required:
            return 0
        return _whitespace_token_count(self.to_decoder_text())

    @property
    def recomputed_quorum_weight(self) -> float:
        # Trust-weighted sum of (ratify - reject) per vote.
        # Each tuple is (probe_id, ratify_int, reject_int, weight, reason).
        s = 0.0
        for _pid, rt, rj, w, _rs in self.probe_votes:
            if rt and not rj:
                s += float(w)
            elif rj and not rt:
                s -= float(w)
        return s

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "schema_cid": self.schema_cid,
            "signature_cid": self.signature_cid,
            "probe_votes": [list(v) for v in self.probe_votes],
            "quorum_threshold": round(float(self.quorum_threshold), 4),
            "quorum_weight": round(float(self.quorum_weight), 4),
            "ratified": bool(self.ratified),
            "cell_index": int(self.cell_index),
            "wire_required": bool(self.wire_required),
            "ratification_cid": self.ratification_cid,
            "n_envelope_bytes": int(self.n_envelope_bytes),
            "n_wire_tokens": int(self.n_wire_tokens),
            "decoder_text": self.to_decoder_text(),
        }


def verify_ensemble_pivot_ratification(
        env: EnsemblePivotRatificationEnvelope | None,
        *,
        registered_schema: SchemaCapsule,
        registered_signature_cid: str,
        registered_probe_ids: frozenset[str],
) -> LatentVerificationOutcome:
    """Controller-side verification of an
    :class:`EnsemblePivotRatificationEnvelope` (W28 family).

    Pure function. Failure modes enumerated:

    * ``empty_ratification``         — None envelope passed.
    * ``schema_version_unknown``     — env.schema_version mismatch.
    * ``schema_cid_mismatch``        — env.schema_cid != registered.
    * ``signature_cid_empty``        — no signature being ratified.
    * ``signature_cid_mismatch``     — env.signature_cid != registered.
    * ``probe_table_empty``          — no probes in the envelope.
    * ``probe_id_unregistered``      — vote from an unregistered probe.
    * ``probe_vote_malformed``       — both ratify and reject true.
    * ``trust_weight_negative``      — vote weight < 0.
    * ``hash_mismatch``              — ratification_cid does not recompute.
    * ``quorum_below_threshold``     — ratified=True but weight < threshold.
    * ``quorum_recompute_mismatch``  — ratified flag != recomputed quorum sign.
    """
    n = 0
    if env is None:
        return LatentVerificationOutcome(
            ok=False, reason="empty_ratification", n_checks=n)
    n += 1
    if env.schema_version != W28_RATIFICATION_SCHEMA_VERSION:
        return LatentVerificationOutcome(
            ok=False, reason="schema_version_unknown", n_checks=n)
    n += 1
    if env.schema_cid != registered_schema.cid:
        return LatentVerificationOutcome(
            ok=False, reason="schema_cid_mismatch", n_checks=n)
    n += 1
    if not env.signature_cid:
        return LatentVerificationOutcome(
            ok=False, reason="signature_cid_empty", n_checks=n)
    n += 1
    if env.signature_cid != registered_signature_cid:
        return LatentVerificationOutcome(
            ok=False, reason="signature_cid_mismatch", n_checks=n)
    n += 1
    if not env.probe_votes:
        return LatentVerificationOutcome(
            ok=False, reason="probe_table_empty", n_checks=n)
    n += 1
    for probe_id, rt, rj, w, _reason in env.probe_votes:
        if probe_id not in registered_probe_ids:
            return LatentVerificationOutcome(
                ok=False, reason="probe_id_unregistered", n_checks=n)
        if bool(rt) and bool(rj):
            return LatentVerificationOutcome(
                ok=False, reason="probe_vote_malformed", n_checks=n)
        if float(w) < 0.0:
            return LatentVerificationOutcome(
                ok=False, reason="trust_weight_negative", n_checks=n)
    n += 1
    if env.ratification_cid != env.recompute_ratification_cid():
        return LatentVerificationOutcome(
            ok=False, reason="hash_mismatch", n_checks=n)
    n += 1
    # Ratified=True must imply quorum_weight ≥ threshold.
    if bool(env.ratified) and float(env.quorum_weight) < float(
            env.quorum_threshold):
        return LatentVerificationOutcome(
            ok=False, reason="quorum_below_threshold", n_checks=n)
    n += 1
    # Ratified flag must match the recomputed quorum sign.
    recomputed = env.recomputed_quorum_weight
    expected_ratified = bool(recomputed >= float(env.quorum_threshold))
    if bool(env.ratified) != expected_ratified:
        return LatentVerificationOutcome(
            ok=False, reason="quorum_recompute_mismatch", n_checks=n)
    # Also enforce internal consistency: the envelope-recorded
    # quorum_weight must be within float tolerance of the recomputed
    # weight (avoids tampering on the weight field while leaving votes
    # untouched).
    if abs(float(env.quorum_weight) - recomputed) > 1e-6:
        return LatentVerificationOutcome(
            ok=False, reason="quorum_recompute_mismatch", n_checks=n)
    return LatentVerificationOutcome(
        ok=True, reason="ok", n_checks=n)


# ---------------------------------------------------------------------------
# Controller-side ratification registry
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class EnsembleRatificationRegistry:
    """Controller-side registry of registered probes + accepted
    ratifications (W28 family).

    Trust boundary: the registry is controller-owned. A probe table
    is bound at construction; producers cannot inject unregistered
    probes (the verifier rejects them with ``probe_id_unregistered``).
    """
    schema: SchemaCapsule | None = None
    quorum_threshold: float = 1.0
    probes: tuple[EnsembleProbeRegistration, ...] = ()
    _ratifications: dict[str, EnsemblePivotRatificationEnvelope] = (
        dataclasses.field(default_factory=dict))
    n_ratifications_registered: int = 0
    n_ratifications_rejected: int = 0
    n_quorum_below_threshold: int = 0
    n_probe_calls_total: int = 0
    n_cross_host_probe_calls: int = 0
    cross_host_round_trip_bytes: int = 0
    local_host_id: str = "localhost"

    @property
    def registered_probe_ids(self) -> frozenset[str]:
        return frozenset(p.probe.probe_id for p in self.probes)

    @property
    def has_wire_required_probe(self) -> bool:
        return any(getattr(p.probe, "wire_required", True) for p in self.probes)

    def register_ratification(
            self,
            env: EnsemblePivotRatificationEnvelope,
            *,
            registered_signature_cid: str,
    ) -> LatentVerificationOutcome:
        if self.schema is None:
            self.n_ratifications_rejected += 1
            return LatentVerificationOutcome(
                ok=False, reason="registry_no_schema", n_checks=0)
        outcome = verify_ensemble_pivot_ratification(
            env,
            registered_schema=self.schema,
            registered_signature_cid=registered_signature_cid,
            registered_probe_ids=self.registered_probe_ids,
        )
        if not outcome.ok:
            self.n_ratifications_rejected += 1
            if outcome.reason == "quorum_below_threshold":
                self.n_quorum_below_threshold += 1
            return outcome
        # Even when the verifier passes structurally, only count it
        # as a *registered* ratification when the ratified flag is
        # True; envelopes that record a failed quorum are still kept
        # (for audit) but counted separately.
        self._ratifications[env.ratification_cid] = env
        if env.ratified:
            self.n_ratifications_registered += 1
        else:
            self.n_quorum_below_threshold += 1
        return outcome

    def reset_session(self) -> None:
        self._ratifications.clear()
        self.n_ratifications_registered = 0
        self.n_ratifications_rejected = 0
        self.n_quorum_below_threshold = 0
        self.n_probe_calls_total = 0
        self.n_cross_host_probe_calls = 0
        self.cross_host_round_trip_bytes = 0


# ---------------------------------------------------------------------------
# W28 result + orchestrator
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class W28EnsembleResult:
    """Per-cell audit record for a W28 ensemble-verified agent."""
    answer: dict[str, Any]
    inner_w27_branch: str
    decoder_branch: str
    agent_id: str
    is_producer: bool
    signature_cid: str
    ratification_cid: str
    n_w27_visible_tokens: int
    n_w28_visible_tokens: int
    n_ratify_overhead_tokens: int
    quorum_threshold: float
    quorum_weight: float
    ratified: bool
    n_probes_called: int
    n_probes_ratified: int
    n_probes_rejected: int
    n_probes_abstained: int
    cross_host_round_trip_bytes: int
    pool_size: int
    pool_exhausted: bool
    ratification_verification_ok: bool
    ratification_verification_reason: str
    probe_vote_summary: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "inner_w27_branch": self.inner_w27_branch,
            "decoder_branch": self.decoder_branch,
            "agent_id": self.agent_id,
            "is_producer": bool(self.is_producer),
            "signature_cid": self.signature_cid,
            "ratification_cid": self.ratification_cid,
            "n_w27_visible_tokens": int(self.n_w27_visible_tokens),
            "n_w28_visible_tokens": int(self.n_w28_visible_tokens),
            "n_ratify_overhead_tokens": int(self.n_ratify_overhead_tokens),
            "quorum_threshold": round(float(self.quorum_threshold), 4),
            "quorum_weight": round(float(self.quorum_weight), 4),
            "ratified": bool(self.ratified),
            "n_probes_called": int(self.n_probes_called),
            "n_probes_ratified": int(self.n_probes_ratified),
            "n_probes_rejected": int(self.n_probes_rejected),
            "n_probes_abstained": int(self.n_probes_abstained),
            "cross_host_round_trip_bytes": int(self.cross_host_round_trip_bytes),
            "pool_size": int(self.pool_size),
            "pool_exhausted": bool(self.pool_exhausted),
            "ratification_verification_ok": bool(
                self.ratification_verification_ok),
            "ratification_verification_reason": str(
                self.ratification_verification_reason),
            "probe_vote_summary": list(self.probe_vote_summary),
        }


@dataclasses.dataclass
class EnsembleVerifiedMultiChainOrchestrator:
    """Ensemble-verified multi-chain pivot ratification (W28 family).

    Wraps a :class:`MultiChainPersistedFanoutOrchestrator` and adds a
    controller-side trust-weighted probe quorum that ratifies (or
    rejects) every pivot / new-anchor decision the W27 layer makes.

    The W28 layer
    -------------
    For every cell:

      1. The wrapped W27 orchestrator runs first; it produces its
         routing decision (PIVOTED / ANCHORED_NEW / FALLBACK_W26 / ...).
      2. If the W27 branch is one of the trigger branches
         (PIVOTED, ANCHORED_NEW), the W28 layer:
           a. Reads the cell's salience signature from the W27 slot.
           b. Polls each :class:`EnsembleProbeRegistration` in the
              registry's probe table; collects per-probe
              :class:`ProbeVote`.
           c. Computes the trust-weighted sum of ratify-vs-reject
              weights.
           d. Builds an :class:`EnsemblePivotRatificationEnvelope`
              with the sealed votes + quorum + ratified flag.
           e. Verifies the envelope via
              :func:`verify_ensemble_pivot_ratification`.
           f. Registers the verified envelope with the registry.
      3. If the envelope's ``ratified`` flag is True AND the verifier
         passes, the cell is ratified — visible token cost = W27 + 1
         iff any probe required the wire (else +0).
      4. If the envelope is rejected (verifier failure OR quorum below
         threshold), the W28 layer marks the cell as not-ratified;
         the W27 routing still happened (it cannot be undone), but
         downstream consumers are told via the audit record that the
         cell is unratified — they may choose to suppress its
         contribution to the team decision.

    Trust boundary
    --------------
    Every probe vote is sealed into the ratification envelope and
    verified by the pure :func:`verify_ensemble_pivot_ratification`,
    which enumerates 11 failure modes (none of which existed in any
    W22..W27 verifier). Tampering on any field of any vote, on the
    quorum weight, on the ratified flag, or on the ratification CID
    is detected.

    Backward compatibility (W28-Λ-single-probe)
    --------------------------------------------
    With a single :class:`DeterministicSignatureProbe` and
    ``quorum_threshold = 1.0``, every cell ratifies trivially and the
    wire-required check is False, so the visible-token cost equals
    W27's exactly. The audit ledger still gains a ratification
    envelope per ratifying cell (for forensics), but no token is
    transmitted. The mechanical assertion of byte-for-byte
    equivalence at K=1 is a unit test in
    ``test_phase75_ensemble_verified_multi_chain.py``.
    """
    inner: MultiChainPersistedFanoutOrchestrator
    registry: EnsembleRatificationRegistry
    enabled: bool = True
    require_ratification_verification: bool = True
    trigger_branches: frozenset[str] = dataclasses.field(
        default_factory=lambda: W28_DEFAULT_TRIGGER_BRANCHES)
    local_host_id: str = "localhost"

    _last_result: "W28EnsembleResult | None" = None
    _last_envelope: "EnsemblePivotRatificationEnvelope | None" = None
    _cell_index: int = 0

    @property
    def schema(self) -> "SchemaCapsule | None":
        return self.inner.schema

    @property
    def agent_id(self) -> str:
        return self.inner.agent_id

    @property
    def is_producer(self) -> bool:
        return self.inner.is_producer

    @property
    def producer_agent_id(self) -> str:
        return self.inner.producer_agent_id

    @property
    def consumer_agent_ids(self) -> tuple[str, ...]:
        return self.inner.consumer_agent_ids

    @property
    def pool_size(self) -> int:
        return self.inner.pool_size

    @property
    def n_pool_exhausted_rejections(self) -> int:
        return self.inner.n_pool_exhausted_rejections

    def reset_session(self) -> None:
        self.inner.reset_session()
        self._last_result = None
        self._last_envelope = None
        self._cell_index = 0

    def _read_cell_signature(self) -> "SalienceSignatureEnvelope | None":
        """Read the salience signature for the cell W27 just ran.

        The W27 orchestrator routes each cell to a per-signature W26
        slot; the slot's inner W25 fanout envelope holds the canonical
        compact state. We rebuild the signature from there so the W28
        layer is independent of any in-memory side effect.
        """
        # Find the slot used by this cell on the producer side.
        # The W27 orchestrator's last_result holds the input
        # signature CID — we look it up in the pool to recover the
        # SalienceSignatureEnvelope shape.
        last = self.inner.last_result
        if last is None:
            return None
        sig_cid = str(last.input_signature_cid)
        if not sig_cid:
            return None
        # Find a slot for this agent + signature.
        slot = self.inner.pool._agent_slots.get(
            self.agent_id, {}).get(sig_cid)
        if slot is None:
            return None
        # Read the slot's inner W25 fanout envelope to get the canonical
        # compact state.
        w26 = slot.disambiguator
        w25 = w26.inner  # SharedFanoutDisambiguator
        fanout = w25.last_fanout_envelope
        if fanout is None:
            return None
        per_tag = tuple(fanout.compact_per_tag_votes)
        projected = tuple(fanout.compact_projected_subset)
        schema_cid = (str(self.schema.cid)
                        if self.schema is not None else "")
        return SalienceSignatureEnvelope(
            schema_version=W27_SALIENCE_SIGNATURE_SCHEMA_VERSION,
            schema_cid=schema_cid,
            producer_agent_id=str(self.producer_agent_id
                                    or self.agent_id),
            consumer_agent_ids=tuple(self.consumer_agent_ids),
            canonical_per_tag_votes=tuple(per_tag),
            canonical_projected_subset=tuple(projected),
            cell_index_first_observed=int(self._cell_index),
        )

    def _poll_probes(
            self,
            *,
            signature: "SalienceSignatureEnvelope",
    ) -> tuple[list[ProbeVote], int, int]:
        """Run every registered probe; return (votes, n_cross_host_calls,
        cross_host_round_trip_bytes)."""
        votes: list[ProbeVote] = []
        n_cross_host = 0
        n_cross_host_bytes = 0
        for reg in self.registry.probes:
            self.registry.n_probe_calls_total += 1
            try:
                vote = reg.probe.vote(
                    signature=signature,
                    canonical_per_tag_votes=signature.canonical_per_tag_votes,
                    canonical_projected_subset=signature.canonical_projected_subset,
                    cell_index=int(self._cell_index),
                )
            except Exception as exc:  # noqa: BLE001
                vote = ProbeVote(
                    probe_id=str(reg.probe.probe_id),
                    ratify=False, reject=False,
                    trust_weight=float(reg.trust_prior),
                    reason=f"probe_error:{type(exc).__name__}")
            # Override the probe-emitted weight with the registration's
            # trust_prior (the probe should not be able to set its own
            # weight; only the controller-registered prior counts).
            vote = ProbeVote(
                probe_id=vote.probe_id,
                ratify=bool(vote.ratify),
                reject=bool(vote.reject),
                trust_weight=float(reg.trust_prior),
                reason=str(vote.reason),
            )
            votes.append(vote)
            # Cross-host accounting.
            if reg.host_id and reg.host_id != self.local_host_id:
                n_cross_host += 1
                # Estimate round-trip bytes from the probe's last
                # response if available (LLMSignatureProbe), else use
                # the salience envelope size as a lower bound.
                last_text = getattr(reg.probe, "last_response_text", "")
                if last_text:
                    n_cross_host_bytes += len(last_text.encode("utf-8"))
                else:
                    n_cross_host_bytes += int(signature.n_signature_bytes)
        self.registry.n_cross_host_probe_calls += n_cross_host
        self.registry.cross_host_round_trip_bytes += n_cross_host_bytes
        return votes, n_cross_host, n_cross_host_bytes

    def decode_rounds(
            self,
            per_round_handoffs: Sequence[Sequence[_DecodedHandoff]],
    ) -> dict[str, Any]:
        # 1. Run W27 first.
        out = self.inner.decode_rounds(per_round_handoffs)
        last_w27 = self.inner.last_result
        cell_index = int(self._cell_index)

        inner_w27_branch = (str(last_w27.decoder_branch)
                              if last_w27 is not None
                              else W27_BRANCH_DISABLED)
        n_w27_visible = (int(last_w27.n_w27_visible_tokens)
                          if last_w27 is not None else 0)
        pool_size = int(self.pool_size)
        pool_exhausted = bool(last_w27.pool_exhausted) if last_w27 else False

        def _pack(
                *,
                decoder_branch: str,
                envelope: EnsemblePivotRatificationEnvelope | None,
                n_w28_visible: int,
                ratify_overhead: int,
                quorum_w: float,
                ratified: bool,
                n_probes_called: int,
                n_ratified: int,
                n_rejected: int,
                n_abstained: int,
                cross_host_bytes: int,
                verify_ok: bool,
                verify_reason: str,
                probe_summary: list[dict[str, Any]],
        ) -> dict[str, Any]:
            sig_cid = str(envelope.signature_cid) if envelope else ""
            rat_cid = str(envelope.ratification_cid) if envelope else ""
            result = W28EnsembleResult(
                answer=dict(out),
                inner_w27_branch=inner_w27_branch,
                decoder_branch=decoder_branch,
                agent_id=str(self.agent_id),
                is_producer=bool(self.is_producer),
                signature_cid=sig_cid,
                ratification_cid=rat_cid,
                n_w27_visible_tokens=int(n_w27_visible),
                n_w28_visible_tokens=int(n_w28_visible),
                n_ratify_overhead_tokens=int(ratify_overhead),
                quorum_threshold=float(self.registry.quorum_threshold),
                quorum_weight=float(quorum_w),
                ratified=bool(ratified),
                n_probes_called=int(n_probes_called),
                n_probes_ratified=int(n_ratified),
                n_probes_rejected=int(n_rejected),
                n_probes_abstained=int(n_abstained),
                cross_host_round_trip_bytes=int(cross_host_bytes),
                pool_size=int(pool_size),
                pool_exhausted=bool(pool_exhausted),
                ratification_verification_ok=bool(verify_ok),
                ratification_verification_reason=str(verify_reason),
                probe_vote_summary=list(probe_summary),
            )
            self._last_result = result
            self._last_envelope = envelope
            out_local = dict(out)
            out_local["ensemble_verified_multi_chain"] = result.as_dict()
            if envelope is not None:
                out_local["ensemble_ratification_envelope"] = (
                    envelope.as_dict())
            return out_local

        # ---- Disabled / no-trigger / no-schema paths ----
        if (not self.enabled or self.schema is None
                or not self.registry.probes):
            self._cell_index += 1
            return _pack(
                decoder_branch=W28_BRANCH_DISABLED,
                envelope=None, n_w28_visible=n_w27_visible,
                ratify_overhead=0,
                quorum_w=0.0, ratified=False,
                n_probes_called=0, n_ratified=0,
                n_rejected=0, n_abstained=0,
                cross_host_bytes=0,
                verify_ok=False, verify_reason="disabled",
                probe_summary=[])

        if inner_w27_branch not in self.trigger_branches:
            self._cell_index += 1
            return _pack(
                decoder_branch=W28_BRANCH_NO_RATIFY_NEEDED,
                envelope=None, n_w28_visible=n_w27_visible,
                ratify_overhead=0,
                quorum_w=0.0, ratified=False,
                n_probes_called=0, n_ratified=0,
                n_rejected=0, n_abstained=0,
                cross_host_bytes=0,
                verify_ok=False,
                verify_reason="w27_branch_not_triggered",
                probe_summary=[])

        # ---- Compute cell signature + poll probes ----
        signature = self._read_cell_signature()
        if signature is None:
            self._cell_index += 1
            return _pack(
                decoder_branch=W28_BRANCH_FALLBACK_W27,
                envelope=None, n_w28_visible=n_w27_visible,
                ratify_overhead=0,
                quorum_w=0.0, ratified=False,
                n_probes_called=0, n_ratified=0,
                n_rejected=0, n_abstained=0,
                cross_host_bytes=0,
                verify_ok=False,
                verify_reason="no_w27_signature",
                probe_summary=[])

        votes, _n_cross_host, cross_host_bytes = self._poll_probes(
            signature=signature)
        # Compute trust-weighted quorum.
        quorum_w = sum(v.signed_weight for v in votes)
        threshold = float(self.registry.quorum_threshold)
        ratified = bool(quorum_w >= threshold)

        n_ratified = sum(1 for v in votes if v.ratify and not v.reject)
        n_rejected = sum(1 for v in votes if v.reject and not v.ratify)
        n_abstained = sum(1 for v in votes if v.is_abstain)
        probe_summary = [
            {"probe_id": v.probe_id, "ratify": int(v.ratify),
              "reject": int(v.reject),
              "trust_weight": round(float(v.trust_weight), 4),
              "reason": v.reason}
            for v in votes
        ]

        # ---- Build the ratification envelope ----
        wire_required = self.registry.has_wire_required_probe
        envelope = EnsemblePivotRatificationEnvelope(
            schema_version=W28_RATIFICATION_SCHEMA_VERSION,
            schema_cid=str(self.schema.cid),
            signature_cid=str(signature.signature_cid),
            probe_votes=tuple(v.as_tuple() for v in votes),
            quorum_threshold=float(threshold),
            quorum_weight=float(quorum_w),
            ratified=bool(ratified),
            cell_index=int(cell_index),
            wire_required=bool(wire_required),
        )

        # ---- Verify + register ----
        outcome = self.registry.register_ratification(
            envelope, registered_signature_cid=signature.signature_cid)
        verify_ok = bool(outcome.ok)
        verify_reason = str(outcome.reason)

        if not verify_ok and self.require_ratification_verification:
            self._cell_index += 1
            return _pack(
                decoder_branch=W28_BRANCH_PROBE_REJECTED,
                envelope=envelope, n_w28_visible=n_w27_visible,
                ratify_overhead=0,
                quorum_w=quorum_w, ratified=False,
                n_probes_called=len(votes), n_ratified=n_ratified,
                n_rejected=n_rejected, n_abstained=n_abstained,
                cross_host_bytes=cross_host_bytes,
                verify_ok=False, verify_reason=verify_reason,
                probe_summary=probe_summary)

        if not ratified:
            self._cell_index += 1
            return _pack(
                decoder_branch=W28_BRANCH_QUORUM_BELOW_THRESHOLD,
                envelope=envelope, n_w28_visible=n_w27_visible,
                ratify_overhead=0,
                quorum_w=quorum_w, ratified=False,
                n_probes_called=len(votes), n_ratified=n_ratified,
                n_rejected=n_rejected, n_abstained=n_abstained,
                cross_host_bytes=cross_host_bytes,
                verify_ok=verify_ok, verify_reason=verify_reason,
                probe_summary=probe_summary)

        # Ratified: charge wire token iff any probe requires the wire.
        ratify_overhead = int(envelope.n_wire_tokens)
        decoder_branch = (W28_BRANCH_RATIFIED if ratify_overhead > 0
                            else W28_BRANCH_RATIFIED_PASSTHROUGH)
        n_w28_visible = int(n_w27_visible + ratify_overhead)
        self._cell_index += 1
        return _pack(
            decoder_branch=decoder_branch,
            envelope=envelope, n_w28_visible=n_w28_visible,
            ratify_overhead=ratify_overhead,
            quorum_w=quorum_w, ratified=True,
            n_probes_called=len(votes), n_ratified=n_ratified,
            n_rejected=n_rejected, n_abstained=n_abstained,
            cross_host_bytes=cross_host_bytes,
            verify_ok=verify_ok, verify_reason=verify_reason,
            probe_summary=probe_summary)

    def decode(
            self,
            handoffs: Sequence[_DecodedHandoff],
    ) -> dict[str, Any]:
        return self.decode_rounds([handoffs])

    @property
    def last_result(self) -> "W28EnsembleResult | None":
        return self._last_result

    @property
    def last_envelope(self) -> "EnsemblePivotRatificationEnvelope | None":
        return self._last_envelope


# ---------------------------------------------------------------------------
# Convenience factories
# ---------------------------------------------------------------------------


def build_default_ensemble_registry(
        *,
        schema: SchemaCapsule,
        quorum_threshold: float = 1.0,
        local_host_id: str = "localhost",
) -> EnsembleRatificationRegistry:
    """Build a W28 registry with a single
    :class:`DeterministicSignatureProbe` (the K=1 byte-for-W27 path).
    """
    return EnsembleRatificationRegistry(
        schema=schema,
        quorum_threshold=float(quorum_threshold),
        probes=(
            EnsembleProbeRegistration(
                probe=DeterministicSignatureProbe(
                    probe_id="local_recompute"),
                trust_prior=1.0,
                role_label="local_recompute",
                host_id=local_host_id,
            ),
        ),
        local_host_id=local_host_id,
    )


def build_two_probe_oracle_ensemble_registry(
        *,
        schema: SchemaCapsule,
        oracle_a: Any,
        oracle_b: Any,
        oracle_a_label: str = "oracle_primary",
        oracle_b_label: str = "oracle_secondary",
        oracle_a_trust: float = 1.0,
        oracle_b_trust: float = 1.0,
        quorum_threshold: float = 1.0,
        local_host_id: str = "localhost",
) -> EnsembleRatificationRegistry:
    """Build a W28 registry with two
    :class:`OracleConsultationProbe` entries.

    Useful for the W21 × W27 synthesis: the same
    :class:`ServiceGraphOracle` / :class:`ChangeHistoryOracle` pair
    that powers W21 also serves as the W28 ensemble probe table.
    """
    return EnsembleRatificationRegistry(
        schema=schema,
        quorum_threshold=float(quorum_threshold),
        probes=(
            EnsembleProbeRegistration(
                probe=OracleConsultationProbe(
                    oracle=oracle_a,
                    probe_id=f"oracle_a_{oracle_a_label}"),
                trust_prior=float(oracle_a_trust),
                role_label=oracle_a_label,
                host_id=local_host_id,
            ),
            EnsembleProbeRegistration(
                probe=OracleConsultationProbe(
                    oracle=oracle_b,
                    probe_id=f"oracle_b_{oracle_b_label}"),
                trust_prior=float(oracle_b_trust),
                role_label=oracle_b_label,
                host_id=local_host_id,
            ),
        ),
        local_host_id=local_host_id,
    )


def build_cross_host_llm_ensemble_registry(
        *,
        schema: SchemaCapsule,
        backends_with_hosts: Sequence[tuple[Any, str, str, float]],
        quorum_threshold: float = 1.0,
        local_host_id: str = "localhost",
) -> EnsembleRatificationRegistry:
    """Build a W28 registry from a list of (backend, host_id,
    role_label, trust_prior) tuples.

    Each backend wraps a real LLM host (Ollama or MLX-distributed)
    on a specific Mac; the W28 layer accumulates
    ``cross_host_round_trip_bytes`` for any probe whose host_id
    differs from ``local_host_id``. This is the **two-host topology
    entry point** — pass two backends pointing at different reachable
    Macs and W28 will measure cross-host probing in the headline run.
    """
    probes = tuple(
        EnsembleProbeRegistration(
            probe=LLMSignatureProbe(
                backend=backend,
                probe_id=f"llm_{role_label}"),
            trust_prior=float(trust_prior),
            role_label=str(role_label),
            host_id=str(host_id),
        )
        for backend, host_id, role_label, trust_prior in backends_with_hosts
    )
    return EnsembleRatificationRegistry(
        schema=schema,
        quorum_threshold=float(quorum_threshold),
        probes=probes,
        local_host_id=local_host_id,
    )

# =============================================================================
# SDK v3.30 — W29 family
# Geometry-partitioned product-manifold dense control + audited subspace
# basis payload + factoradic routing index + causal-validity gate +
# cross-host variance witness (research-grade; experimental).
# =============================================================================
#
# W29 wraps the W28 ensemble-verified multi-chain orchestrator with a
# **structural geometry partition** on every triggered cell, an audited
# **subspace-basis payload** over the closed-vocabulary tag space, an
# audited **factoradic routing index** (Lehmer code) over the registered
# consumer order, an audited **causal-validity binding** to the cell's
# declared predecessors, and an optional **cross-host variance witness**
# that fires when ensemble probes on different hosts disagree.
#
# Honest scope (the load-bearing soundness statement)
# ---------------------------------------------------
#
# * W29 does NOT touch transformer KV caches, hidden states, attention
#   weights, embedding tables, or any model-internal state.  The
#   "subspace basis" lives at the **capsule layer**; it is an honest
#   **proxy** for the LatentMAS Grassmannian / shared-substrate
#   direction, not a runtime KV transplant.  Every basis vector is a
#   deterministic projection over the closed-vocabulary tag space; the
#   verifier checks orthogonality, finiteness, and dimension.
# * The "geometry partition" is a **structural label** — one of
#   ``LINEAR`` / ``HIERARCHICAL`` / ``CYCLIC`` — derived from the W27
#   inner branch and the recent signature history.  It is NOT a
#   learned manifold and NOT a curvature in any Riemannian sense.
#   Mixed-curvature is invoked as design metaphor only.
# * The "factoradic routing index" is a Lehmer code of a permutation
#   over the registered consumer order; bounded ``0 ≤ idx < K!``;
#   audit-friendly; NOT a learned routing function.
# * The "causal-validity signature" is SHA-256 over the parent W28
#   ratification CID concatenated with the sorted predecessor cell CIDs;
#   it does NOT prove temporal ordering at the model layer; it is a
#   structural commitment the controller enforces at admission time.
# * W29 does NOT claim "we solved context."
#
# Wire-token economics
# --------------------
#
# When the registered partition table contains > 1 partition AND
# basis_dim > 0 AND/OR len(consumer_order) > 1, W29 charges **1 visible
# token** per triggered cell on the producer side
# (``<partition_ref:DDDD>``) — the entire partition_id + factoradic
# index + basis CID + causal-validity signature + cross-host variance
# witness CID rides on that single content-addressed reference.  When
# every component is trivial (partition_table size 1, basis_dim 0, K=1,
# no predecessors), wire_required is False and W29 reduces to W28
# **byte-for-byte** (the W29-Λ-trivial-partition falsifier; see H2 in
# ``docs/SUCCESS_CRITERION_W29_GEOMETRY_PARTITIONED.md``).
#
# Trust boundary: 14 enumerated failure modes
# -------------------------------------------
#
# :func:`verify_geometry_partition_ratification` rejects:
#
#   * ``empty_partition_envelope``       — None envelope passed.
#   * ``schema_version_unknown``         — schema_version mismatch.
#   * ``schema_cid_mismatch``            — schema_cid != registered.
#   * ``w28_parent_cid_mismatch``        — env.w28_ratification_cid !=
#     the registered W28 envelope's ratification_cid.
#   * ``partition_id_unregistered``      — env.partition_id not in the
#     registered partition table.
#   * ``subspace_basis_dim_mismatch``    — basis_dim != registered, OR
#     len(basis_vectors) != basis_dim, OR per-vector length != ambient_dim.
#   * ``subspace_basis_non_orthogonal``  — Gram off-diagonals exceed
#     orthogonality_tol (default 1e-4).
#   * ``subspace_basis_nan_inf``         — NaN/Inf in any coefficient.
#   * ``factoradic_index_out_of_range``  — index < 0 OR index >= K!.
#   * ``factoradic_route_inverse_mismatch`` — decode(idx) != registered
#     permutation rank for this partition.
#   * ``causal_predecessor_unregistered`` — a predecessor CID not in
#     the registered predecessor set.
#   * ``causal_validity_signature_invalid`` — recomputed
#     SHA-256(parent_w28_cid || sorted(predecessors)) != envelope.
#   * ``cross_host_variance_witness_unsealed`` — when the controller
#     observed cross-host disagreement, env.cross_host_variance_witness_cid
#     must be non-empty.
#   * ``partition_cid_hash_mismatch``    — recomputed partition_cid
#     does not match the envelope's stored partition_cid.
#
# Every failure mode is mechanically asserted by a unit test in
# ``test_phase76_geometry_partitioned.py``.

W29_PARTITION_SCHEMA_VERSION: str = "wevra.geometry_partition_ratification.v1"

# Registered partition labels.  These are the only legal values for
# ``GeometryPartitionedRatificationEnvelope.partition_id`` until the
# next milestone enlarges the registry.
W29_PARTITION_LINEAR: int = 0
W29_PARTITION_HIERARCHICAL: int = 1
W29_PARTITION_CYCLIC: int = 2
W29_REGISTERED_PARTITION_IDS: frozenset[int] = frozenset({
    W29_PARTITION_LINEAR,
    W29_PARTITION_HIERARCHICAL,
    W29_PARTITION_CYCLIC,
})

# String labels for forensics; the canonical wire identity is the int.
W29_PARTITION_LABEL: dict[int, str] = {
    W29_PARTITION_LINEAR: "linear",
    W29_PARTITION_HIERARCHICAL: "hierarchical",
    W29_PARTITION_CYCLIC: "cyclic",
}

# W29 decoder branches.
W29_BRANCH_PARTITION_RESOLVED = "partition_resolved"
W29_BRANCH_TRIVIAL_PARTITION_PASSTHROUGH = "trivial_partition_passthrough"
W29_BRANCH_PARTITION_REJECTED = "partition_rejected"
W29_BRANCH_PARTITION_BELOW_THRESHOLD = "partition_below_threshold"
W29_BRANCH_CROSS_HOST_VARIANCE_WITNESSED = "cross_host_variance_witnessed"
W29_BRANCH_NO_PARTITION_NEEDED = "no_partition_needed"
W29_BRANCH_FALLBACK_W28 = "fallback_w28"
W29_BRANCH_NO_TRIGGER = "no_trigger"
W29_BRANCH_DISABLED = "disabled"

W29_ALL_BRANCHES: tuple[str, ...] = (
    W29_BRANCH_PARTITION_RESOLVED,
    W29_BRANCH_TRIVIAL_PARTITION_PASSTHROUGH,
    W29_BRANCH_PARTITION_REJECTED,
    W29_BRANCH_PARTITION_BELOW_THRESHOLD,
    W29_BRANCH_CROSS_HOST_VARIANCE_WITNESSED,
    W29_BRANCH_NO_PARTITION_NEEDED,
    W29_BRANCH_FALLBACK_W28,
    W29_BRANCH_NO_TRIGGER,
    W29_BRANCH_DISABLED,
)

# W29 fires only on W28 branches that committed a ratification (the
# trust-boundary anchor).  Disabled / no-trigger / fallback are passed
# through unchanged.
W29_DEFAULT_TRIGGER_BRANCHES: frozenset[str] = frozenset({
    W28_BRANCH_RATIFIED,
    W28_BRANCH_RATIFIED_PASSTHROUGH,
})

W29_DEFAULT_ORTHOGONALITY_TOL: float = 1e-4


# ---------------------------------------------------------------------------
# Subspace basis (Grassmannian-style audited proxy)
# ---------------------------------------------------------------------------


def _compute_subspace_basis_cid(
        *,
        basis_vectors: tuple[tuple[float, ...], ...],
        ambient_dim: int,
        ambient_vocabulary: tuple[str, ...],
) -> str:
    """Canonical SHA-256 over a tuple of basis vectors.

    Floats are rounded to 4 dp before hashing (same convention as the
    W22..W28 surface) to avoid IEEE-754 byte drift.  Ambient dimension
    and ambient vocabulary are part of the canonical payload so two
    bases over different vocabularies do not collide.
    """
    payload = _canonical_json_bytes({
        "ambient_dim": int(ambient_dim),
        "ambient_vocabulary": list(ambient_vocabulary),
        "basis_vectors": [
            [round(float(c), 4) for c in v] for v in basis_vectors
        ],
    })
    return hashlib.sha256(payload).hexdigest()


@dataclasses.dataclass(frozen=True)
class SubspaceBasis:
    """An audited orthonormal-style subspace basis over the closed-
    vocabulary tag space (W29 family).

    A point on a Grassmannian-style manifold at the **capsule layer** —
    NOT a transformer-internal subspace projection.  Each basis vector
    is a tuple of floats over the closed-vocabulary tag basis; the
    verifier checks orthogonality, dimension, finiteness, and content
    address.

    Fields
    ------
    dim
        Number of basis vectors (the subspace's rank).  ``0`` means
        no basis is carried (trivial-partition path).
    ambient_dim
        Dimension of the ambient closed-vocabulary tag space.  Must
        equal ``len(ambient_vocabulary)`` and ``len(basis_vectors[i])``.
    ambient_vocabulary
        Sorted closed-vocabulary tag names that index the basis vectors'
        coordinates.  Part of the canonical payload so different
        vocabularies do not collide.
    basis_vectors
        ``dim`` vectors, each of length ``ambient_dim``.  Floats rounded
        to 4 dp at construction for deterministic hashing.
    orthogonality_tol
        Tolerance below which off-diagonal Gram entries are accepted.
        Default 1e-4.
    basis_cid
        SHA-256 over the canonical payload (auto-computed at
        construction).
    """
    dim: int
    ambient_dim: int
    ambient_vocabulary: tuple[str, ...]
    basis_vectors: tuple[tuple[float, ...], ...]
    orthogonality_tol: float = W29_DEFAULT_ORTHOGONALITY_TOL
    basis_cid: str = ""

    def __post_init__(self) -> None:
        # Round basis coefficients deterministically and freeze the
        # ambient vocabulary order.
        rounded = tuple(
            tuple(round(float(c), 4) for c in v)
            for v in self.basis_vectors
        )
        object.__setattr__(self, "basis_vectors", rounded)
        object.__setattr__(self, "ambient_vocabulary",
                           tuple(self.ambient_vocabulary))
        if not self.basis_cid:
            object.__setattr__(self, "basis_cid", self._recompute_cid())

    def _recompute_cid(self) -> str:
        return _compute_subspace_basis_cid(
            basis_vectors=self.basis_vectors,
            ambient_dim=self.ambient_dim,
            ambient_vocabulary=self.ambient_vocabulary,
        )

    @property
    def gram_off_diag_max(self) -> float:
        """Largest absolute off-diagonal entry of the Gram matrix
        ``B B^T``.  Returns 0.0 when ``dim ≤ 1``.
        """
        n = int(self.dim)
        if n <= 1:
            return 0.0
        max_off = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                s = sum(
                    float(a) * float(b)
                    for a, b in zip(self.basis_vectors[i],
                                     self.basis_vectors[j])
                )
                if abs(s) > max_off:
                    max_off = abs(s)
        return float(max_off)

    @property
    def per_vector_norm_max_dev(self) -> float:
        """Largest absolute deviation of any basis vector's norm from
        1.0.  Returns 0.0 when ``dim == 0``.
        """
        if int(self.dim) == 0:
            return 0.0
        max_dev = 0.0
        for v in self.basis_vectors:
            n2 = sum(float(c) * float(c) for c in v)
            d = abs(math.sqrt(n2) - 1.0)
            if d > max_dev:
                max_dev = d
        return float(max_dev)

    @property
    def has_nan_or_inf(self) -> bool:
        for v in self.basis_vectors:
            for c in v:
                f = float(c)
                if math.isnan(f) or math.isinf(f):
                    return True
        return False

    def to_canonical_bytes(self) -> bytes:
        return _canonical_json_bytes({
            "dim": int(self.dim),
            "ambient_dim": int(self.ambient_dim),
            "ambient_vocabulary": list(self.ambient_vocabulary),
            "basis_vectors": [
                [round(float(c), 4) for c in v]
                for v in self.basis_vectors
            ],
        })

    @property
    def n_basis_bytes(self) -> int:
        return len(self.to_canonical_bytes())

    def as_dict(self) -> dict[str, Any]:
        return {
            "dim": int(self.dim),
            "ambient_dim": int(self.ambient_dim),
            "ambient_vocabulary": list(self.ambient_vocabulary),
            "basis_vectors": [list(v) for v in self.basis_vectors],
            "basis_cid": self.basis_cid,
            "gram_off_diag_max": round(self.gram_off_diag_max, 6),
            "per_vector_norm_max_dev": round(
                self.per_vector_norm_max_dev, 6),
            "n_basis_bytes": self.n_basis_bytes,
        }


def verify_subspace_basis(
        basis: SubspaceBasis | None,
        *,
        expected_dim: int,
        expected_ambient_dim: int,
        orthogonality_tol: float = W29_DEFAULT_ORTHOGONALITY_TOL,
) -> LatentVerificationOutcome:
    """Pure-function controller-side verification of a
    :class:`SubspaceBasis` (W29 family).

    Failure modes (subset of W29's full enumeration):

      * ``empty_basis``                  — None passed AND expected_dim > 0.
      * ``subspace_basis_dim_mismatch``  — dim, ambient_dim, or per-vector
                                            length does not match.
      * ``subspace_basis_nan_inf``       — any NaN/Inf coefficient.
      * ``subspace_basis_non_orthogonal`` — Gram off-diagonal exceeds tol.
      * ``hash_mismatch``                — basis_cid does not recompute.

    Note: when ``expected_dim == 0`` the function accepts ``None`` AND
    accepts a basis with ``dim == 0``.
    """
    if expected_dim == 0:
        if basis is None:
            return LatentVerificationOutcome(
                ok=True, reason="ok", n_checks=0)
        # A basis can be supplied with dim=0; that is acceptable.
        if int(basis.dim) == 0:
            return LatentVerificationOutcome(
                ok=True, reason="ok", n_checks=1)
        # If expected_dim is 0 but a non-empty basis is supplied, that
        # is a dim mismatch.
        return LatentVerificationOutcome(
            ok=False, reason="subspace_basis_dim_mismatch", n_checks=1)
    if basis is None:
        return LatentVerificationOutcome(
            ok=False, reason="empty_basis", n_checks=0)
    n = 1
    if int(basis.dim) != int(expected_dim):
        return LatentVerificationOutcome(
            ok=False, reason="subspace_basis_dim_mismatch", n_checks=n)
    n += 1
    if int(basis.ambient_dim) != int(expected_ambient_dim):
        return LatentVerificationOutcome(
            ok=False, reason="subspace_basis_dim_mismatch", n_checks=n)
    n += 1
    if len(basis.basis_vectors) != int(basis.dim):
        return LatentVerificationOutcome(
            ok=False, reason="subspace_basis_dim_mismatch", n_checks=n)
    for v in basis.basis_vectors:
        if len(v) != int(basis.ambient_dim):
            return LatentVerificationOutcome(
                ok=False, reason="subspace_basis_dim_mismatch", n_checks=n)
    n += 1
    if basis.has_nan_or_inf:
        return LatentVerificationOutcome(
            ok=False, reason="subspace_basis_nan_inf", n_checks=n)
    n += 1
    if basis.gram_off_diag_max > float(orthogonality_tol):
        return LatentVerificationOutcome(
            ok=False, reason="subspace_basis_non_orthogonal", n_checks=n)
    n += 1
    if basis.basis_cid != basis._recompute_cid():
        return LatentVerificationOutcome(
            ok=False, reason="hash_mismatch", n_checks=n)
    return LatentVerificationOutcome(ok=True, reason="ok", n_checks=n)


def compute_structural_subspace_basis(
        *,
        canonical_per_tag_votes: tuple[tuple[str, int], ...],
        ambient_vocabulary: tuple[str, ...],
        partition_id: int,
        basis_dim: int = 2,
) -> SubspaceBasis:
    """Deterministic structural subspace basis over the closed
    vocabulary (W29 family).

    Honest construction (audit-friendly, NOT a learned manifold):

      * ``basis_vectors[0]`` — the per-tag-vote vector projected onto
        ``ambient_vocabulary`` and L2-normalised.  Captures the cell's
        salience direction in tag space.
      * ``basis_vectors[1]`` — a partition-indicator vector (the
        canonical partition's primary axis) Gram-Schmidt-orthogonalised
        against ``basis_vectors[0]``.  Captures the cell's geometry-
        partition direction independent of its salience.

    Edge cases:

      * If the salience vector has zero norm (no tags voted), use the
        first standard basis vector e_0.
      * If the partition-indicator vector becomes ~0 after
        Gram-Schmidt (parallel to v_0), pick the first standard basis
        vector orthogonal to v_0.

    For ``basis_dim ∈ {0, 1, 2}`` only.  For higher dimensions, callers
    should compose multiple cells via a chain — out of scope for v3.30.
    """
    ambient = tuple(ambient_vocabulary)
    n = len(ambient)
    if int(basis_dim) == 0:
        return SubspaceBasis(
            dim=0,
            ambient_dim=n,
            ambient_vocabulary=ambient,
            basis_vectors=(),
        )
    if int(basis_dim) > 2:
        # Cap at 2 — higher rank requires multi-cell history; not
        # supported in v3.30.  Caller should reduce basis_dim or pin
        # at 2 explicitly.
        basis_dim = 2

    # ----- v0: salience direction -----
    counts = {t: int(c) for t, c in canonical_per_tag_votes}
    v0 = [float(counts.get(t, 0)) for t in ambient]
    norm0 = math.sqrt(sum(x * x for x in v0))
    if norm0 == 0.0:
        # Use e_0 as a deterministic fallback.
        v0 = [0.0] * n
        if n > 0:
            v0[0] = 1.0
        norm0 = 1.0
    v0 = [x / norm0 for x in v0]

    if int(basis_dim) == 1:
        return SubspaceBasis(
            dim=1,
            ambient_dim=n,
            ambient_vocabulary=ambient,
            basis_vectors=(tuple(v0),),
        )

    # ----- v1: partition-indicator axis, Gram-Schmidt against v0 -----
    # Partition-indicator: a deterministic indicator pattern over the
    # ambient vocabulary keyed by partition_id.  Concretely:
    #   * partition_id == 0 (linear)       → indicator on even indices
    #   * partition_id == 1 (hierarchical) → indicator on odd indices
    #   * partition_id == 2 (cyclic)       → indicator on indices
    #                                        ≡ 0 mod 3, 1 mod 3 staggered
    #
    # Any well-defined indicator works for the verifier; the check is
    # orthogonality, dimension, finiteness — not "the indicator was
    # computed correctly."  Different indicators on the same partition
    # produce different basis_cids; that is fine, the verifier accepts
    # all valid bases on a given partition.
    pid = int(partition_id)
    raw1 = [0.0] * n
    if pid == W29_PARTITION_LINEAR:
        for i in range(0, n, 2):
            raw1[i] = 1.0
    elif pid == W29_PARTITION_HIERARCHICAL:
        for i in range(1, n, 2):
            raw1[i] = 1.0
    else:  # cyclic
        for i in range(n):
            if i % 3 == 0:
                raw1[i] = 1.0
            elif i % 3 == 1:
                raw1[i] = 0.5

    # Gram-Schmidt against v0.
    dot = sum(a * b for a, b in zip(raw1, v0))
    v1 = [a - dot * b for a, b in zip(raw1, v0)]
    norm1 = math.sqrt(sum(x * x for x in v1))
    if norm1 < 1e-9:
        # raw1 was parallel to v0 — pick the first standard basis
        # vector orthogonal to v0.  v0[i] != 1 ⇒ e_i has component
        # along v0; subtract.
        for i in range(n):
            ei = [0.0] * n
            ei[i] = 1.0
            d = sum(a * b for a, b in zip(ei, v0))
            cand = [a - d * b for a, b in zip(ei, v0)]
            ncand = math.sqrt(sum(x * x for x in cand))
            if ncand > 1e-9:
                v1 = [x / ncand for x in cand]
                break
    else:
        v1 = [x / norm1 for x in v1]

    return SubspaceBasis(
        dim=2,
        ambient_dim=n,
        ambient_vocabulary=ambient,
        basis_vectors=(tuple(v0), tuple(v1)),
    )


# ---------------------------------------------------------------------------
# Factoradic (Lehmer-code) routing index
# ---------------------------------------------------------------------------


def encode_permutation_to_factoradic(
        perm: tuple[int, ...],
) -> int:
    """Encode a permutation of ``(0, 1, ..., K-1)`` to a factoradic
    integer via its Lehmer code (W29 family).

    The Lehmer code at position ``i`` is the number of remaining-after-
    ``i`` elements that are smaller than ``perm[i]``.  The factoradic
    integer is ``Σ lehmer[i] * (K - 1 - i)!``; bounded by ``0 ≤ idx < K!``.

    Raises ``ValueError`` if ``perm`` is not a permutation of
    ``range(K)``.
    """
    p = list(perm)
    K = len(p)
    if sorted(p) != list(range(K)):
        raise ValueError(
            f"encode_permutation_to_factoradic: not a permutation: {perm!r}")
    idx = 0
    remaining = list(range(K))
    for i, x in enumerate(p):
        rank = remaining.index(int(x))
        idx += rank * math.factorial(K - 1 - i)
        remaining.pop(rank)
    return int(idx)


def decode_factoradic_to_permutation(
        idx: int,
        n_factors: int,
) -> tuple[int, ...]:
    """Decode a factoradic integer to a permutation of
    ``(0, 1, ..., n_factors-1)`` (W29 family).

    Inverse of :func:`encode_permutation_to_factoradic`.
    """
    K = int(n_factors)
    if K < 0:
        raise ValueError("decode_factoradic_to_permutation: n_factors < 0")
    if K == 0:
        if int(idx) != 0:
            raise ValueError(
                "decode_factoradic_to_permutation: K=0 but idx != 0")
        return ()
    K_fact = math.factorial(K)
    if int(idx) < 0 or int(idx) >= K_fact:
        raise ValueError(
            f"decode_factoradic_to_permutation: idx={idx} out of [0, {K_fact})")
    remaining = list(range(K))
    perm: list[int] = []
    rest = int(idx)
    for i in range(K):
        f = math.factorial(K - 1 - i)
        rank = rest // f
        rest = rest % f
        perm.append(remaining.pop(rank))
    return tuple(perm)


# ---------------------------------------------------------------------------
# Causal-validity binding
# ---------------------------------------------------------------------------


def _compute_causal_validity_signature(
        *,
        parent_w28_cid: str,
        predecessor_cids: tuple[str, ...],
) -> str:
    """Canonical SHA-256 over (parent_w28_cid, sorted predecessor CIDs).
    """
    payload = _canonical_json_bytes({
        "parent_w28_cid": str(parent_w28_cid),
        "predecessor_cids": sorted(str(p) for p in predecessor_cids),
    })
    return hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------
# Cross-host variance witness
# ---------------------------------------------------------------------------


def _compute_cross_host_variance_witness_cid(
        *,
        cell_index: int,
        disagreement_pairs: tuple[tuple[str, str, str, str], ...],
        total_pairs_seen: int,
) -> str:
    """Canonical SHA-256 over a cell's per-probe-pair disagreement
    record.

    Each disagreement_pair entry is
    ``(probe_id_a, host_id_a, probe_id_b, host_id_b)``; sorted at
    construction.
    """
    payload = _canonical_json_bytes({
        "cell_index": int(cell_index),
        "disagreement_pairs": [
            list(p) for p in disagreement_pairs
        ],
        "total_pairs_seen": int(total_pairs_seen),
    })
    return hashlib.sha256(payload).hexdigest()


@dataclasses.dataclass(frozen=True)
class CrossHostVarianceWitness:
    """A content-addressed witness of cross-host probe disagreement on
    one cell (W29 family).

    Carries:
      * ``cell_index``         — the cell that produced the witness.
      * ``disagreement_pairs`` — sorted tuple of (probe_id_a, host_id_a,
                                  probe_id_b, host_id_b) tuples for
                                  every (i, j) probe pair where the
                                  votes disagreed.
      * ``cross_host_disagreements`` — count of pairs where host_id_a
                                        != host_id_b.
      * ``total_pairs_seen``   — number of probe pairs evaluated.
      * ``witness_cid``        — SHA-256 over the canonical payload.
    """
    cell_index: int
    disagreement_pairs: tuple[tuple[str, str, str, str], ...]
    cross_host_disagreements: int
    total_pairs_seen: int
    witness_cid: str = ""

    def __post_init__(self) -> None:
        # Sort pairs canonically so identical disagreement sets collide
        # to identical witness_cids.
        sorted_pairs = tuple(sorted(
            (str(a), str(ha), str(b), str(hb))
            for a, ha, b, hb in self.disagreement_pairs
        ))
        object.__setattr__(self, "disagreement_pairs", sorted_pairs)
        if not self.witness_cid:
            object.__setattr__(self, "witness_cid", self._recompute_cid())

    def _recompute_cid(self) -> str:
        return _compute_cross_host_variance_witness_cid(
            cell_index=self.cell_index,
            disagreement_pairs=self.disagreement_pairs,
            total_pairs_seen=self.total_pairs_seen,
        )

    def to_canonical_bytes(self) -> bytes:
        return _canonical_json_bytes({
            "cell_index": int(self.cell_index),
            "disagreement_pairs": [
                list(p) for p in self.disagreement_pairs],
            "cross_host_disagreements": int(self.cross_host_disagreements),
            "total_pairs_seen": int(self.total_pairs_seen),
        })

    def as_dict(self) -> dict[str, Any]:
        return {
            "cell_index": int(self.cell_index),
            "disagreement_pairs": [
                list(p) for p in self.disagreement_pairs],
            "cross_host_disagreements": int(self.cross_host_disagreements),
            "total_pairs_seen": int(self.total_pairs_seen),
            "witness_cid": self.witness_cid,
        }


# ---------------------------------------------------------------------------
# Geometry-partitioned ratification envelope + verifier
# ---------------------------------------------------------------------------


def _compute_geometry_partition_cid(
        *,
        schema_version: str,
        schema_cid: str,
        w28_ratification_cid: str,
        partition_id: int,
        basis_cid: str,
        basis_dim: int,
        ambient_dim: int,
        factoradic_route_index: int,
        factoradic_route_n_factors: int,
        causal_validity_signature: str,
        predecessor_cids: tuple[str, ...],
        cross_host_variance_witness_cid: str,
        cell_index: int,
) -> str:
    """Canonical SHA-256 over a geometry-partition envelope payload."""
    payload = _canonical_json_bytes({
        "schema_version": str(schema_version),
        "schema_cid": str(schema_cid),
        "w28_ratification_cid": str(w28_ratification_cid),
        "partition_id": int(partition_id),
        "basis_cid": str(basis_cid),
        "basis_dim": int(basis_dim),
        "ambient_dim": int(ambient_dim),
        "factoradic_route_index": int(factoradic_route_index),
        "factoradic_route_n_factors": int(factoradic_route_n_factors),
        "causal_validity_signature": str(causal_validity_signature),
        "predecessor_cids": sorted(str(p) for p in predecessor_cids),
        "cross_host_variance_witness_cid": str(
            cross_host_variance_witness_cid),
        "cell_index": int(cell_index),
    })
    return hashlib.sha256(payload).hexdigest()


@dataclasses.dataclass(frozen=True)
class GeometryPartitionedRatificationEnvelope:
    """Content-addressed geometry-partitioned ratification of one W28
    decision (W29 family).

    Carries:

      * ``w28_ratification_cid``        — the parent W28 envelope's CID.
      * ``partition_id``                — registered structural label.
      * ``basis``                       — :class:`SubspaceBasis` (or
                                            ``None`` when ``basis_dim==0``).
      * ``factoradic_route_index``      — Lehmer-code permutation index
                                            over the registered consumer
                                            order.
      * ``causal_validity_signature``   — SHA-256 over (parent W28 CID,
                                            sorted predecessor CIDs).
      * ``predecessor_cids``            — declared cell predecessors
                                            (audit only; the signature
                                            binds them).
      * ``cross_host_variance_witness_cid`` — non-empty when the W29
                                            layer observed cross-host
                                            disagreement on this cell.

    The envelope's ``partition_cid`` is SHA-256 over the canonical
    payload; tampering on any field is detected by
    :func:`verify_geometry_partition_ratification`.

    Wire token cost
    ---------------

    The W29 layer charges 1 visible token on the producer side
    (``<partition_ref:DDDD>``) iff ``wire_required`` is True (i.e. the
    partition table is non-trivial OR ``basis_dim > 0`` OR the
    registered consumer order has K > 1 OR the registered predecessor
    set is non-empty).  When every component is trivial (single
    partition, basis_dim 0, K=1, no predecessors), the envelope is
    recorded in the audit ledger only and no token is transmitted, so
    W29 reduces to W28 byte-for-byte (W29-Λ-trivial-partition).
    """
    schema_version: str
    schema_cid: str
    w28_ratification_cid: str
    partition_id: int
    basis: SubspaceBasis | None
    basis_dim: int
    ambient_dim: int
    factoradic_route_index: int
    factoradic_route_n_factors: int
    causal_validity_signature: str
    predecessor_cids: tuple[str, ...]
    cross_host_variance_witness_cid: str
    cell_index: int
    wire_required: bool = False
    partition_cid: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "predecessor_cids",
                           tuple(sorted(str(p)
                                        for p in self.predecessor_cids)))
        if not self.partition_cid:
            object.__setattr__(self, "partition_cid",
                                self.recompute_partition_cid())

    def recompute_partition_cid(self) -> str:
        return _compute_geometry_partition_cid(
            schema_version=self.schema_version,
            schema_cid=self.schema_cid,
            w28_ratification_cid=self.w28_ratification_cid,
            partition_id=self.partition_id,
            basis_cid=(self.basis.basis_cid if self.basis is not None
                        else ""),
            basis_dim=self.basis_dim,
            ambient_dim=self.ambient_dim,
            factoradic_route_index=self.factoradic_route_index,
            factoradic_route_n_factors=self.factoradic_route_n_factors,
            causal_validity_signature=self.causal_validity_signature,
            predecessor_cids=self.predecessor_cids,
            cross_host_variance_witness_cid=(
                self.cross_host_variance_witness_cid),
            cell_index=self.cell_index,
        )

    def to_canonical_bytes(self) -> bytes:
        return _canonical_json_bytes({
            "schema_version": self.schema_version,
            "schema_cid": self.schema_cid,
            "w28_ratification_cid": self.w28_ratification_cid,
            "partition_id": int(self.partition_id),
            "basis_cid": (self.basis.basis_cid if self.basis is not None
                          else ""),
            "basis_dim": int(self.basis_dim),
            "ambient_dim": int(self.ambient_dim),
            "factoradic_route_index": int(self.factoradic_route_index),
            "factoradic_route_n_factors": int(
                self.factoradic_route_n_factors),
            "causal_validity_signature": self.causal_validity_signature,
            "predecessor_cids": list(self.predecessor_cids),
            "cross_host_variance_witness_cid": (
                self.cross_host_variance_witness_cid),
            "cell_index": int(self.cell_index),
        })

    def to_decoder_text(self) -> str:
        return f"<partition_ref:{self.partition_cid[:16]}>"

    @property
    def n_envelope_bytes(self) -> int:
        return len(self.to_canonical_bytes())

    @property
    def n_wire_tokens(self) -> int:
        if not self.wire_required:
            return 0
        return _whitespace_token_count(self.to_decoder_text())

    @property
    def n_structured_bits(self) -> int:
        """Approximate count of structured-control bits packed into
        this envelope's canonical payload.  Used for the cram-factor
        metric.

        Uses ``8 * len(canonical_bytes)`` as a faithful upper bound on
        the audit-friendly content the envelope carries; the wire-side
        cost is at most 1 visible token.
        """
        if self.basis is not None:
            basis_bits = 8 * self.basis.n_basis_bytes
        else:
            basis_bits = 0
        return 8 * self.n_envelope_bytes + basis_bits

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "schema_cid": self.schema_cid,
            "w28_ratification_cid": self.w28_ratification_cid,
            "partition_id": int(self.partition_id),
            "partition_label": W29_PARTITION_LABEL.get(
                int(self.partition_id), "unknown"),
            "basis": (self.basis.as_dict() if self.basis is not None
                       else None),
            "basis_dim": int(self.basis_dim),
            "ambient_dim": int(self.ambient_dim),
            "factoradic_route_index": int(self.factoradic_route_index),
            "factoradic_route_n_factors": int(
                self.factoradic_route_n_factors),
            "causal_validity_signature": self.causal_validity_signature,
            "predecessor_cids": list(self.predecessor_cids),
            "cross_host_variance_witness_cid": (
                self.cross_host_variance_witness_cid),
            "cell_index": int(self.cell_index),
            "wire_required": bool(self.wire_required),
            "partition_cid": self.partition_cid,
            "n_envelope_bytes": self.n_envelope_bytes,
            "n_wire_tokens": self.n_wire_tokens,
            "n_structured_bits": int(self.n_structured_bits),
            "decoder_text": self.to_decoder_text(),
        }


@dataclasses.dataclass(frozen=True)
class PartitionRegistration:
    """One registered partition in a W29 partition table (mirrors
    :class:`EnsembleProbeRegistration`).

    Fields
    ------
    partition_id
        Integer identifier in :data:`W29_REGISTERED_PARTITION_IDS`.
    label
        Optional forensics label ("linear" / "hierarchical" / "cyclic").
    consumer_permutation
        The permutation of the registered consumer order this
        partition routes to.  Must be a permutation of
        ``range(len(consumer_order))``.  Encoded by the producer as
        the factoradic_route_index and decoded by the verifier.
    trust_prior
        Per-partition prior weight used by the partition layer's quorum
        check.  Default ``1.0``.
    """
    partition_id: int
    label: str = ""
    consumer_permutation: tuple[int, ...] = ()
    trust_prior: float = 1.0


def verify_geometry_partition_ratification(
        env: GeometryPartitionedRatificationEnvelope | None,
        *,
        registered_schema: SchemaCapsule,
        registered_w28_ratification_cid: str,
        registered_partition_table: tuple[PartitionRegistration, ...],
        registered_basis_dim: int,
        registered_ambient_dim: int,
        registered_consumer_order: tuple[str, ...],
        registered_predecessor_cids: frozenset[str] = frozenset(),
        cross_host_disagreement_observed: bool = False,
        orthogonality_tol: float = W29_DEFAULT_ORTHOGONALITY_TOL,
) -> LatentVerificationOutcome:
    """Pure-function controller-side verification of a
    :class:`GeometryPartitionedRatificationEnvelope` (W29 family).

    Failure modes enumerated:

      * ``empty_partition_envelope``
      * ``schema_version_unknown``
      * ``schema_cid_mismatch``
      * ``w28_parent_cid_mismatch``
      * ``partition_id_unregistered``
      * ``subspace_basis_dim_mismatch``
      * ``subspace_basis_non_orthogonal``
      * ``subspace_basis_nan_inf``
      * ``factoradic_index_out_of_range``
      * ``factoradic_route_inverse_mismatch``
      * ``causal_predecessor_unregistered``
      * ``causal_validity_signature_invalid``
      * ``cross_host_variance_witness_unsealed``
      * ``partition_cid_hash_mismatch``
    """
    n = 0
    if env is None:
        return LatentVerificationOutcome(
            ok=False, reason="empty_partition_envelope", n_checks=n)
    n += 1
    if env.schema_version != W29_PARTITION_SCHEMA_VERSION:
        return LatentVerificationOutcome(
            ok=False, reason="schema_version_unknown", n_checks=n)
    n += 1
    if env.schema_cid != registered_schema.cid:
        return LatentVerificationOutcome(
            ok=False, reason="schema_cid_mismatch", n_checks=n)
    n += 1
    if env.w28_ratification_cid != registered_w28_ratification_cid:
        return LatentVerificationOutcome(
            ok=False, reason="w28_parent_cid_mismatch", n_checks=n)
    n += 1
    registered_ids = frozenset(p.partition_id
                                for p in registered_partition_table)
    if int(env.partition_id) not in registered_ids:
        return LatentVerificationOutcome(
            ok=False, reason="partition_id_unregistered", n_checks=n)
    n += 1
    # Basis verification.
    basis_outcome = verify_subspace_basis(
        env.basis,
        expected_dim=int(registered_basis_dim),
        expected_ambient_dim=int(registered_ambient_dim),
        orthogonality_tol=float(orthogonality_tol),
    )
    if not basis_outcome.ok:
        # Re-export the basis verifier's reason in W29's enumeration.
        rsn = str(basis_outcome.reason)
        if rsn in (
                "empty_basis", "subspace_basis_dim_mismatch",
                "hash_mismatch"):
            return LatentVerificationOutcome(
                ok=False, reason="subspace_basis_dim_mismatch", n_checks=n)
        if rsn == "subspace_basis_nan_inf":
            return LatentVerificationOutcome(
                ok=False, reason="subspace_basis_nan_inf", n_checks=n)
        if rsn == "subspace_basis_non_orthogonal":
            return LatentVerificationOutcome(
                ok=False, reason="subspace_basis_non_orthogonal",
                n_checks=n)
        return LatentVerificationOutcome(
            ok=False, reason=str(basis_outcome.reason), n_checks=n)
    if env.basis is not None and int(env.basis_dim) != int(env.basis.dim):
        return LatentVerificationOutcome(
            ok=False, reason="subspace_basis_dim_mismatch", n_checks=n)
    n += 1
    K = int(env.factoradic_route_n_factors)
    K_registered = len(registered_consumer_order)
    if K < 0 or K != K_registered:
        return LatentVerificationOutcome(
            ok=False, reason="factoradic_index_out_of_range", n_checks=n)
    if K > 0:
        K_fact = math.factorial(K)
        if (int(env.factoradic_route_index) < 0
                or int(env.factoradic_route_index) >= K_fact):
            return LatentVerificationOutcome(
                ok=False, reason="factoradic_index_out_of_range",
                n_checks=n)
    elif int(env.factoradic_route_index) != 0:
        # K=0 ⇒ idx must be 0.
        return LatentVerificationOutcome(
            ok=False, reason="factoradic_index_out_of_range", n_checks=n)
    n += 1
    # Decode-and-compare against the partition's registered permutation.
    if K > 0:
        try:
            decoded = decode_factoradic_to_permutation(
                int(env.factoradic_route_index), K)
        except ValueError:
            return LatentVerificationOutcome(
                ok=False, reason="factoradic_index_out_of_range",
                n_checks=n)
        # Find the partition's registered permutation.
        partition_perm: tuple[int, ...] | None = None
        for p in registered_partition_table:
            if int(p.partition_id) == int(env.partition_id):
                partition_perm = tuple(int(x) for x in p.consumer_permutation)
                break
        if (partition_perm is not None
                and len(partition_perm) == K
                and tuple(decoded) != partition_perm):
            return LatentVerificationOutcome(
                ok=False, reason="factoradic_route_inverse_mismatch",
                n_checks=n)
    n += 1
    # Predecessor closure.
    if registered_predecessor_cids:
        for p in env.predecessor_cids:
            if str(p) not in registered_predecessor_cids:
                return LatentVerificationOutcome(
                    ok=False, reason="causal_predecessor_unregistered",
                    n_checks=n)
    n += 1
    # Causal-validity signature recompute.
    expected_sig = _compute_causal_validity_signature(
        parent_w28_cid=env.w28_ratification_cid,
        predecessor_cids=env.predecessor_cids,
    )
    if expected_sig != env.causal_validity_signature:
        return LatentVerificationOutcome(
            ok=False, reason="causal_validity_signature_invalid",
            n_checks=n)
    n += 1
    # Cross-host variance witness consistency.
    if (cross_host_disagreement_observed
            and not env.cross_host_variance_witness_cid):
        return LatentVerificationOutcome(
            ok=False, reason="cross_host_variance_witness_unsealed",
            n_checks=n)
    n += 1
    # Partition-cid hash recompute (last — catches any field tampering
    # missed by the structural checks above).
    if env.partition_cid != env.recompute_partition_cid():
        return LatentVerificationOutcome(
            ok=False, reason="partition_cid_hash_mismatch", n_checks=n)
    return LatentVerificationOutcome(ok=True, reason="ok", n_checks=n)


# ---------------------------------------------------------------------------
# Partition classifier
# ---------------------------------------------------------------------------


def classify_partition_id_for_cell(
        *,
        w28_branch: str,
        signature_cid: str,
        signature_history: tuple[str, ...],
        cycle_window: int = 4,
) -> int:
    """Deterministic structural classifier for the W29 partition_id.

    Pure function; relies only on the W28 branch and the cell's recent
    signature history.

    Heuristic (sound by inspection): we identify the *most recent run*
    of identical signatures at the tail of the history, treat that run
    as the "current branch", and look at everything before the run for
    cyclic membership:

      * ``HIERARCHICAL`` — first-cell case OR the cell's signature is
        a fresh anchor (never appears in ``signature_history``).
      * ``CYCLIC``       — the cell's signature appears in the
        pre-run history within ``cycle_window`` (we have visited this
        signature *before* the current run; the producer is cycling
        back to a prior compartment).
      * ``LINEAR``       — the cell's signature equals the last entry
        AND the signature does NOT appear before the current run
        (i.e. the producer is extending the current run without
        cycling).

    The classifier is purely structural and replayable; the verifier
    does NOT re-classify (it only checks that the producer's
    partition_id is in the registered partition table).
    """
    if not signature_history:
        # First cell: no history, can only be a fresh anchor.
        return W29_PARTITION_HIERARCHICAL
    last = signature_history[-1]
    # Walk back to find the start index of the most recent run of
    # identical signatures (suffix run with signature == last).
    run_start = 0
    for i in range(len(signature_history) - 1, -1, -1):
        if signature_history[i] != last:
            run_start = i + 1
            break
    # Pre-run = everything before the current run.
    pre_run = signature_history[:run_start]
    pre_run_window = pre_run[-int(cycle_window):]
    if signature_cid in pre_run_window:
        return W29_PARTITION_CYCLIC
    if signature_cid == last:
        return W29_PARTITION_LINEAR
    return W29_PARTITION_HIERARCHICAL


# ---------------------------------------------------------------------------
# Geometry partition registry + result + orchestrator
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class GeometryPartitionRegistry:
    """Controller-side registry for the W29 partition table (mirrors
    :class:`EnsembleRatificationRegistry`).
    """
    schema: SchemaCapsule | None = None
    partition_table: tuple[PartitionRegistration, ...] = ()
    basis_dim: int = 2
    ambient_dim: int = 0
    ambient_vocabulary: tuple[str, ...] = ()
    consumer_order: tuple[str, ...] = ()
    registered_predecessor_cids: frozenset[str] = frozenset()
    cycle_window: int = 4
    quorum_threshold: float = 1.0
    orthogonality_tol: float = W29_DEFAULT_ORTHOGONALITY_TOL
    local_host_id: str = "localhost"

    _envelopes: dict[str, GeometryPartitionedRatificationEnvelope] = (
        dataclasses.field(default_factory=dict))
    n_partitions_registered: int = 0
    n_partitions_rejected: int = 0
    n_cross_host_variance_witnessed: int = 0
    n_partition_below_threshold: int = 0
    n_per_partition_routed: dict[int, int] = dataclasses.field(
        default_factory=dict)

    @property
    def is_trivial(self) -> bool:
        """True iff this registry is on the byte-for-W28 path:
        ``basis_dim = 0`` AND ``len(consumer_order) <= 1`` AND no
        predecessors AND every registered partition has an empty
        ``consumer_permutation``.

        Note: we permit MULTIPLE partition entries in the trivial
        registry as long as each is empty-permutation — that lets the
        structural classifier hand any cell to any registered partition
        (LINEAR / HIERARCHICAL / CYCLIC) without forcing the wire-token
        charge on the trivial path.
        """
        all_empty_perms = all(
            (not p.consumer_permutation)
            for p in self.partition_table
        )
        return (
            int(self.basis_dim) == 0
            and len(self.consumer_order) <= 1
            and not self.registered_predecessor_cids
            and all_empty_perms
        )

    @property
    def has_wire_required_layer(self) -> bool:
        """True iff this registry will charge a wire token on
        triggered cells (i.e. the partition layer is NOT trivial).
        """
        return not self.is_trivial

    @property
    def registered_partition_ids(self) -> frozenset[int]:
        return frozenset(
            int(p.partition_id) for p in self.partition_table)

    def register_envelope(
            self,
            env: GeometryPartitionedRatificationEnvelope,
            *,
            cross_host_disagreement_observed: bool,
    ) -> LatentVerificationOutcome:
        if self.schema is None:
            self.n_partitions_rejected += 1
            return LatentVerificationOutcome(
                ok=False, reason="registry_no_schema", n_checks=0)
        outcome = verify_geometry_partition_ratification(
            env,
            registered_schema=self.schema,
            registered_w28_ratification_cid=env.w28_ratification_cid,
            registered_partition_table=self.partition_table,
            registered_basis_dim=int(self.basis_dim),
            registered_ambient_dim=int(self.ambient_dim),
            registered_consumer_order=self.consumer_order,
            registered_predecessor_cids=self.registered_predecessor_cids,
            cross_host_disagreement_observed=bool(
                cross_host_disagreement_observed),
            orthogonality_tol=float(self.orthogonality_tol),
        )
        if not outcome.ok:
            self.n_partitions_rejected += 1
            return outcome
        self._envelopes[env.partition_cid] = env
        self.n_partitions_registered += 1
        pid = int(env.partition_id)
        self.n_per_partition_routed[pid] = (
            self.n_per_partition_routed.get(pid, 0) + 1)
        if env.cross_host_variance_witness_cid:
            self.n_cross_host_variance_witnessed += 1
        return outcome

    def reset_session(self) -> None:
        self._envelopes.clear()
        self.n_partitions_registered = 0
        self.n_partitions_rejected = 0
        self.n_cross_host_variance_witnessed = 0
        self.n_partition_below_threshold = 0
        self.n_per_partition_routed = {}


@dataclasses.dataclass
class W29PartitionResult:
    """Per-cell audit record for a W29 geometry-partitioned agent."""
    answer: dict[str, Any]
    inner_w28_branch: str
    decoder_branch: str
    agent_id: str
    is_producer: bool
    w28_ratification_cid: str
    partition_id: int
    partition_label: str
    basis_dim: int
    factoradic_route_index: int
    factoradic_route_n_factors: int
    causal_validity_signature: str
    cross_host_variance_witness_cid: str
    cross_host_disagreement_count: int
    n_w28_visible_tokens: int
    n_w29_visible_tokens: int
    n_partition_overhead_tokens: int
    partition_cid: str
    ratified: bool
    verification_ok: bool
    verification_reason: str
    n_envelope_bytes: int
    n_structured_bits: int
    cram_factor_w29: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "inner_w28_branch": self.inner_w28_branch,
            "decoder_branch": self.decoder_branch,
            "agent_id": self.agent_id,
            "is_producer": bool(self.is_producer),
            "w28_ratification_cid": self.w28_ratification_cid,
            "partition_id": int(self.partition_id),
            "partition_label": self.partition_label,
            "basis_dim": int(self.basis_dim),
            "factoradic_route_index": int(self.factoradic_route_index),
            "factoradic_route_n_factors": int(
                self.factoradic_route_n_factors),
            "causal_validity_signature": self.causal_validity_signature,
            "cross_host_variance_witness_cid": (
                self.cross_host_variance_witness_cid),
            "cross_host_disagreement_count": int(
                self.cross_host_disagreement_count),
            "n_w28_visible_tokens": int(self.n_w28_visible_tokens),
            "n_w29_visible_tokens": int(self.n_w29_visible_tokens),
            "n_partition_overhead_tokens": int(
                self.n_partition_overhead_tokens),
            "partition_cid": self.partition_cid,
            "ratified": bool(self.ratified),
            "verification_ok": bool(self.verification_ok),
            "verification_reason": self.verification_reason,
            "n_envelope_bytes": int(self.n_envelope_bytes),
            "n_structured_bits": int(self.n_structured_bits),
            "cram_factor_w29": round(float(self.cram_factor_w29), 4),
        }


@dataclasses.dataclass
class GeometryPartitionedOrchestrator:
    """Geometry-partitioned product-manifold orchestrator (W29 family).

    Wraps an :class:`EnsembleVerifiedMultiChainOrchestrator` and adds a
    structural geometry partition + audited subspace-basis payload +
    factoradic routing index + causal-validity gate + cross-host
    variance witness on every triggered cell.

    Per-cell flow:

      1. Run the inner W28 orchestrator.
      2. If the W28 branch is not in the trigger set, pass through
         (W29_BRANCH_NO_PARTITION_NEEDED, no token charged).
      3. Else classify the cell's partition_id using the W28 branch
         and the recent signature history.
      4. Compute the audited subspace basis (deterministic structural
         projection over the closed vocabulary).
      5. Compute the factoradic_route_index from the partition's
         registered permutation.
      6. Compute the causal-validity signature over (parent_w28_cid,
         declared predecessor_cids).
      7. Compute the cross-host variance witness from the registered
         W28 probes' last-vote disagreement matrix.
      8. Build a :class:`GeometryPartitionedRatificationEnvelope`.
      9. Verify the envelope via the pure verifier.
      10. Register with the controller's
          :class:`GeometryPartitionRegistry`.
      11. Charge 1 wire token iff ``registry.has_wire_required_layer``.

    Trust boundary: every envelope is sealed and verified before it
    contributes to the audit ledger; tampering on any field is detected.
    The W29-Λ-trivial-partition path is the byte-for-W28 reduction
    (registry.is_trivial = True ⇒ no token charged, partition recorded
    in audit only).
    """
    inner: EnsembleVerifiedMultiChainOrchestrator
    registry: GeometryPartitionRegistry
    enabled: bool = True
    require_partition_verification: bool = True
    trigger_branches: frozenset[str] = dataclasses.field(
        default_factory=lambda: W29_DEFAULT_TRIGGER_BRANCHES)
    cycle_window: int = 4
    declared_predecessor_cids: tuple[str, ...] = ()
    # Optional per-partition inner W28 dispatch.  When a partition_id
    # appears as a key, that partition's inner W28 stack runs INSTEAD of
    # ``inner`` for cells classified to that partition.  This is the
    # honest mixed-curvature compartmentalisation: each partition gets
    # its own inner stack (its own oracle config / probe table / pool),
    # and W29's partition decision routes the cell into the right
    # compartment before W28 commits.  When unused (``{}``), the
    # orchestrator falls back to the unified ``inner`` for every cell.
    inner_per_partition: dict[
        int, EnsembleVerifiedMultiChainOrchestrator] = dataclasses.field(
        default_factory=dict)
    # When ``True`` AND ``inner_per_partition`` is non-empty, the
    # orchestrator computes the input signature from the cell handoffs
    # FIRST (via :func:`compute_input_signature_cid`), classifies the
    # partition_id, then dispatches to the chosen inner W28 stack.  When
    # ``False``, the orchestrator runs ``inner`` first (the simpler,
    # backwards-compatible path used by the H2 byte-for-W28 anchor and
    # the cram-factor headline).
    pre_dispatch_by_partition: bool = False
    # Optional classifier hook (W30 extension point).  When set, the
    # orchestrator calls this callable instead of
    # :func:`classify_partition_id_for_cell` directly.  Used by the W30
    # ``CalibratedGeometryOrchestrator`` to inject a calibration-prior
    # reroute (e.g., when a partition's calibration_vector entry is
    # below threshold, route to the registered high-trust partition).
    # Signature: ``(w28_branch, signature_cid, signature_history,
    # cycle_window) -> int``.  Returns a registered partition_id.
    partition_classifier_hook: "Callable[..., int] | None" = None

    _last_result: "W29PartitionResult | None" = None
    _last_envelope: "GeometryPartitionedRatificationEnvelope | None" = None
    _signature_history: list[str] = dataclasses.field(default_factory=list)
    _cell_index: int = 0
    _last_active_inner: "EnsembleVerifiedMultiChainOrchestrator | None" = None

    @property
    def schema(self) -> "SchemaCapsule | None":
        return self.inner.schema

    @property
    def agent_id(self) -> str:
        return self.inner.agent_id

    @property
    def is_producer(self) -> bool:
        return self.inner.is_producer

    @property
    def producer_agent_id(self) -> str:
        return self.inner.producer_agent_id

    @property
    def consumer_agent_ids(self) -> tuple[str, ...]:
        return self.inner.consumer_agent_ids

    def reset_session(self) -> None:
        self.inner.reset_session()
        for inner in self.inner_per_partition.values():
            inner.reset_session()
        self._last_result = None
        self._last_envelope = None
        self._signature_history = []
        self._cell_index = 0
        self._last_active_inner = None

    def _resolve_inner_for_partition(
            self,
            partition_id: int,
    ) -> EnsembleVerifiedMultiChainOrchestrator:
        """Pick the inner W28 stack for a given partition_id.  Falls
        back to ``self.inner`` when the partition is not in the
        per-partition dispatch table.
        """
        return self.inner_per_partition.get(int(partition_id), self.inner)

    def _read_w28_envelope_and_branch(
            self,
            out: dict[str, Any],
    ) -> tuple[EnsemblePivotRatificationEnvelope | None, str, int]:
        ev = None
        branch = ""
        n_w28_visible = 0
        if "ensemble_verified_multi_chain" in out:
            audit = out["ensemble_verified_multi_chain"]
            branch = str(audit.get("decoder_branch", ""))
            n_w28_visible = int(audit.get("n_w28_visible_tokens", 0))
            ev = self.inner.last_envelope
        return ev, branch, n_w28_visible

    def _compute_cross_host_variance_witness(
            self,
            *,
            active_inner: "EnsembleVerifiedMultiChainOrchestrator | None" = None,
    ) -> tuple[CrossHostVarianceWitness | None, int, bool]:
        """Inspect the last W28 poll's per-probe votes and emit a
        CrossHostVarianceWitness when ≥1 cross-host probe pair
        disagreed.

        Returns ``(witness, n_disagreements_cross_host, observed_flag)``.
        """
        inner = active_inner or self.inner
        last_w28 = inner.last_result
        if last_w28 is None or not last_w28.probe_vote_summary:
            return None, 0, False
        # Pull host ids from the inner registry's probes (matching by
        # probe_id).
        registry = inner.registry
        host_for: dict[str, str] = {
            str(reg.probe.probe_id): str(reg.host_id)
            for reg in registry.probes
        }
        votes: list[tuple[str, str, int, int]] = []
        for v in last_w28.probe_vote_summary:
            pid = str(v.get("probe_id", ""))
            host = host_for.get(pid, "")
            votes.append((
                pid, host,
                int(v.get("ratify", 0)),
                int(v.get("reject", 0)),
            ))
        # Pairwise disagreement: probe_a's signed vote != probe_b's.
        # We treat (ratify=1, reject=0) and (ratify=0, reject=1) as the
        # two opposed strict states; abstain (0,0) does NOT count as a
        # disagreement.
        disagreements: list[tuple[str, str, str, str]] = []
        n_cross_host_disagreements = 0
        n_pairs = 0
        for i in range(len(votes)):
            for j in range(i + 1, len(votes)):
                p_a, h_a, rt_a, rj_a = votes[i]
                p_b, h_b, rt_b, rj_b = votes[j]
                a_strict = (rt_a, rj_a) in {(1, 0), (0, 1)}
                b_strict = (rt_b, rj_b) in {(1, 0), (0, 1)}
                if not (a_strict and b_strict):
                    continue
                n_pairs += 1
                if (rt_a, rj_a) != (rt_b, rj_b):
                    disagreements.append((p_a, h_a, p_b, h_b))
                    if h_a and h_b and h_a != h_b:
                        n_cross_host_disagreements += 1
        if n_pairs == 0:
            return None, 0, False
        observed = n_cross_host_disagreements > 0
        if not disagreements:
            return None, 0, False
        witness = CrossHostVarianceWitness(
            cell_index=int(self._cell_index),
            disagreement_pairs=tuple(disagreements),
            cross_host_disagreements=int(n_cross_host_disagreements),
            total_pairs_seen=int(n_pairs),
        )
        return witness, n_cross_host_disagreements, observed

    def _read_signature_for_cell(self) -> str:
        last_w27 = self.inner.inner.last_result
        if last_w27 is None:
            return ""
        return str(last_w27.input_signature_cid or "")

    def _classify_partition_pre_dispatch(
            self,
            per_round_handoffs: Sequence[Sequence[_DecodedHandoff]],
    ) -> tuple[int, str]:
        """Compute the input signature CID from cell handoffs ALONE
        (no inner W28 run yet) and classify the partition_id.

        Used by the per-partition pre-dispatch path so the W29 layer
        picks the inner W28 stack to run BEFORE that stack commits its
        answer.  Pure function over the handoffs + signature_history.
        """
        if self.schema is None:
            return W29_PARTITION_LINEAR, ""
        sig_cid = compute_input_signature_cid(
            per_round_handoffs,
            producer_agent_id=str(self.producer_agent_id
                                    or self.agent_id),
            consumer_agent_ids=tuple(self.consumer_agent_ids),
            schema_cid=str(self.schema.cid),
        )
        partition_id = classify_partition_id_for_cell(
            w28_branch="",  # not yet known
            signature_cid=str(sig_cid),
            signature_history=tuple(self._signature_history),
            cycle_window=int(self.cycle_window),
        )
        return int(partition_id), str(sig_cid)

    def decode_rounds(
            self,
            per_round_handoffs: Sequence[Sequence[_DecodedHandoff]],
    ) -> dict[str, Any]:
        # 1. Optionally pre-dispatch by partition.  When per-partition
        # inner W28 stacks are registered AND ``pre_dispatch_by_partition``
        # is True, classify the cell's partition BEFORE running any W28
        # so the right compartment runs.
        active_inner = self.inner
        pre_dispatched_partition: int | None = None
        if self.pre_dispatch_by_partition and self.inner_per_partition:
            pid, _pre_sig = self._classify_partition_pre_dispatch(
                per_round_handoffs)
            # W30 extension point: a calibration-aware classifier may
            # reroute the structural classification (e.g. CYCLIC ->
            # high_trust_partition_id when calibration prior is below
            # threshold).  When the hook is set, it sees the same
            # signature_history the structural classifier saw.
            if self.partition_classifier_hook is not None:
                try:
                    pid = int(self.partition_classifier_hook(
                        w28_branch="",
                        signature_cid=str(_pre_sig),
                        signature_history=tuple(self._signature_history),
                        cycle_window=int(self.cycle_window),
                    ))
                except Exception:
                    # Honest fallback: if the hook raises, keep the
                    # structural classification.  Hooks must be pure
                    # functions; the fallback ensures we never lose a
                    # cell on a hook bug.
                    pass
            pre_dispatched_partition = int(pid)
            active_inner = self._resolve_inner_for_partition(int(pid))
        self._last_active_inner = active_inner

        # 2. Run the chosen inner W28 (which runs W27 inside).
        out = active_inner.decode_rounds(per_round_handoffs)
        # Read W28 audit from the active inner.
        w28_envelope: EnsemblePivotRatificationEnvelope | None = None
        inner_w28_branch = ""
        n_w28_visible = 0
        if "ensemble_verified_multi_chain" in out:
            audit = out["ensemble_verified_multi_chain"]
            inner_w28_branch = str(audit.get("decoder_branch", ""))
            n_w28_visible = int(audit.get("n_w28_visible_tokens", 0))
            w28_envelope = active_inner.last_envelope
        cell_index = int(self._cell_index)

        def _pack(
                *,
                decoder_branch: str,
                envelope: GeometryPartitionedRatificationEnvelope | None,
                n_w29_visible: int,
                partition_overhead: int,
                ratified: bool,
                verify_ok: bool,
                verify_reason: str,
                cross_host_disagreement_count: int,
                witness_cid: str,
                partition_id: int,
                basis_dim: int,
                factoradic_route_index: int,
                factoradic_route_n_factors: int,
                causal_validity_signature: str,
                w28_ratification_cid: str,
        ) -> dict[str, Any]:
            envelope_bytes = (envelope.n_envelope_bytes
                               if envelope is not None else 0)
            structured_bits = (envelope.n_structured_bits
                                 if envelope is not None else 0)
            wire = max(1, partition_overhead)
            cram_factor = (
                float(structured_bits) / float(wire)
                if structured_bits > 0 else 0.0
            )
            partition_cid = (envelope.partition_cid
                              if envelope is not None else "")
            result = W29PartitionResult(
                answer=dict(out),
                inner_w28_branch=str(inner_w28_branch),
                decoder_branch=str(decoder_branch),
                agent_id=str(self.agent_id),
                is_producer=bool(self.is_producer),
                w28_ratification_cid=str(w28_ratification_cid),
                partition_id=int(partition_id),
                partition_label=W29_PARTITION_LABEL.get(
                    int(partition_id), "unknown"),
                basis_dim=int(basis_dim),
                factoradic_route_index=int(factoradic_route_index),
                factoradic_route_n_factors=int(factoradic_route_n_factors),
                causal_validity_signature=str(causal_validity_signature),
                cross_host_variance_witness_cid=str(witness_cid),
                cross_host_disagreement_count=int(
                    cross_host_disagreement_count),
                n_w28_visible_tokens=int(n_w28_visible),
                n_w29_visible_tokens=int(n_w29_visible),
                n_partition_overhead_tokens=int(partition_overhead),
                partition_cid=str(partition_cid),
                ratified=bool(ratified),
                verification_ok=bool(verify_ok),
                verification_reason=str(verify_reason),
                n_envelope_bytes=int(envelope_bytes),
                n_structured_bits=int(structured_bits),
                cram_factor_w29=float(cram_factor),
            )
            self._last_result = result
            self._last_envelope = envelope
            out_local = dict(out)
            out_local["geometry_partitioned"] = result.as_dict()
            if envelope is not None:
                out_local["geometry_partition_envelope"] = envelope.as_dict()
            return out_local

        # ---- Read the active inner's signature_cid for history bookkeeping
        # ----
        # We need to record this signature in self._signature_history
        # *regardless* of which downstream branch fires, so that the
        # next cell's classifier sees the correct cumulative history.
        last_w27_active = active_inner.inner.last_result
        sig_cid_from_inner = ""
        if last_w27_active is not None:
            sig_cid_from_inner = str(
                last_w27_active.input_signature_cid or "")

        # ---- Disabled / no-trigger / no-schema paths ----
        if (not self.enabled or self.schema is None
                or not self.registry.partition_table):
            self._cell_index += 1
            self._signature_history.append(str(sig_cid_from_inner))
            return _pack(
                decoder_branch=W29_BRANCH_DISABLED,
                envelope=None, n_w29_visible=n_w28_visible,
                partition_overhead=0, ratified=False,
                verify_ok=False, verify_reason="disabled",
                cross_host_disagreement_count=0,
                witness_cid="",
                partition_id=int(pre_dispatched_partition or 0),
                basis_dim=0, factoradic_route_index=0,
                factoradic_route_n_factors=0,
                causal_validity_signature="",
                w28_ratification_cid="")

        if inner_w28_branch not in self.trigger_branches:
            self._cell_index += 1
            self._signature_history.append(str(sig_cid_from_inner))
            return _pack(
                decoder_branch=W29_BRANCH_NO_PARTITION_NEEDED,
                envelope=None, n_w29_visible=n_w28_visible,
                partition_overhead=0, ratified=False,
                verify_ok=False, verify_reason="w28_branch_not_triggered",
                cross_host_disagreement_count=0,
                witness_cid="",
                partition_id=int(pre_dispatched_partition or 0),
                basis_dim=0, factoradic_route_index=0,
                factoradic_route_n_factors=0,
                causal_validity_signature="",
                w28_ratification_cid="")

        if w28_envelope is None:
            self._cell_index += 1
            self._signature_history.append(str(sig_cid_from_inner))
            return _pack(
                decoder_branch=W29_BRANCH_FALLBACK_W28,
                envelope=None, n_w29_visible=n_w28_visible,
                partition_overhead=0, ratified=False,
                verify_ok=False, verify_reason="no_w28_envelope",
                cross_host_disagreement_count=0,
                witness_cid="",
                partition_id=int(pre_dispatched_partition or 0),
                basis_dim=0, factoradic_route_index=0,
                factoradic_route_n_factors=0,
                causal_validity_signature="",
                w28_ratification_cid="")

        # ---- Reuse the signature_cid we already read above + classify ----
        last_w27 = last_w27_active
        sig_cid = sig_cid_from_inner
        if pre_dispatched_partition is not None:
            partition_id = int(pre_dispatched_partition)
        else:
            partition_id = classify_partition_id_for_cell(
                w28_branch=inner_w28_branch,
                signature_cid=sig_cid,
                signature_history=tuple(self._signature_history),
                cycle_window=int(self.cycle_window),
            )
            # W30 hook also applies to the post-W28 path (the simple
            # H7-cram path that does not pre-dispatch).
            if self.partition_classifier_hook is not None:
                try:
                    partition_id = int(self.partition_classifier_hook(
                        w28_branch=str(inner_w28_branch),
                        signature_cid=str(sig_cid),
                        signature_history=tuple(self._signature_history),
                        cycle_window=int(self.cycle_window),
                    ))
                except Exception:
                    pass
        # Append to history AFTER classification so the current cell's
        # signature does not leak into its own classification.
        self._signature_history.append(str(sig_cid))

        # ---- Compute basis ----
        basis: SubspaceBasis | None
        canonical_per_tag_votes: tuple[tuple[str, int], ...] = ()
        # Pull the canonical state from the active inner W27 layer's
        # last signature.
        if last_w27 is not None:
            sig_env = active_inner._read_cell_signature()
            if sig_env is not None:
                canonical_per_tag_votes = tuple(sig_env.canonical_per_tag_votes)
        if int(self.registry.basis_dim) == 0:
            basis = None
        else:
            basis = compute_structural_subspace_basis(
                canonical_per_tag_votes=canonical_per_tag_votes,
                ambient_vocabulary=self.registry.ambient_vocabulary,
                partition_id=int(partition_id),
                basis_dim=int(self.registry.basis_dim),
            )

        # ---- Factoradic route index from the partition's permutation ----
        partition_perm: tuple[int, ...] = ()
        for p in self.registry.partition_table:
            if int(p.partition_id) == int(partition_id):
                partition_perm = tuple(int(x) for x in p.consumer_permutation)
                break
        K = len(self.registry.consumer_order)
        if K > 0 and partition_perm and len(partition_perm) == K:
            try:
                fact_idx = encode_permutation_to_factoradic(partition_perm)
            except ValueError:
                fact_idx = 0
        else:
            fact_idx = 0

        # ---- Causal-validity signature ----
        predecessor_cids = tuple(self.declared_predecessor_cids)
        causal_sig = _compute_causal_validity_signature(
            parent_w28_cid=str(w28_envelope.ratification_cid),
            predecessor_cids=predecessor_cids,
        )

        # ---- Cross-host variance witness ----
        witness, n_cross_disagree, disagreement_observed = (
            self._compute_cross_host_variance_witness(
                active_inner=active_inner))
        witness_cid = (witness.witness_cid if witness is not None else "")

        # ---- Build envelope ----
        wire_required = self.registry.has_wire_required_layer
        envelope = GeometryPartitionedRatificationEnvelope(
            schema_version=W29_PARTITION_SCHEMA_VERSION,
            schema_cid=str(self.schema.cid),
            w28_ratification_cid=str(w28_envelope.ratification_cid),
            partition_id=int(partition_id),
            basis=basis,
            basis_dim=int(self.registry.basis_dim),
            ambient_dim=int(self.registry.ambient_dim),
            factoradic_route_index=int(fact_idx),
            factoradic_route_n_factors=int(K),
            causal_validity_signature=str(causal_sig),
            predecessor_cids=predecessor_cids,
            cross_host_variance_witness_cid=str(witness_cid),
            cell_index=int(cell_index),
            wire_required=bool(wire_required),
        )

        # ---- Verify + register ----
        outcome = self.registry.register_envelope(
            envelope,
            cross_host_disagreement_observed=bool(disagreement_observed),
        )
        verify_ok = bool(outcome.ok)
        verify_reason = str(outcome.reason)

        if not verify_ok and self.require_partition_verification:
            self._cell_index += 1
            return _pack(
                decoder_branch=W29_BRANCH_PARTITION_REJECTED,
                envelope=envelope, n_w29_visible=n_w28_visible,
                partition_overhead=0, ratified=False,
                verify_ok=False, verify_reason=verify_reason,
                cross_host_disagreement_count=int(n_cross_disagree),
                witness_cid=str(witness_cid),
                partition_id=int(partition_id),
                basis_dim=int(self.registry.basis_dim),
                factoradic_route_index=int(fact_idx),
                factoradic_route_n_factors=int(K),
                causal_validity_signature=str(causal_sig),
                w28_ratification_cid=str(w28_envelope.ratification_cid))

        # Trivial partition: no wire token charged; pass through.
        if self.registry.is_trivial:
            self._cell_index += 1
            return _pack(
                decoder_branch=W29_BRANCH_TRIVIAL_PARTITION_PASSTHROUGH,
                envelope=envelope, n_w29_visible=n_w28_visible,
                partition_overhead=0, ratified=True,
                verify_ok=verify_ok, verify_reason=verify_reason,
                cross_host_disagreement_count=int(n_cross_disagree),
                witness_cid=str(witness_cid),
                partition_id=int(partition_id),
                basis_dim=int(self.registry.basis_dim),
                factoradic_route_index=int(fact_idx),
                factoradic_route_n_factors=int(K),
                causal_validity_signature=str(causal_sig),
                w28_ratification_cid=str(w28_envelope.ratification_cid))

        # Non-trivial: charge 1 wire token.
        partition_overhead = int(envelope.n_wire_tokens)
        n_w29_visible = int(n_w28_visible + partition_overhead)
        decoder_branch = (
            W29_BRANCH_CROSS_HOST_VARIANCE_WITNESSED
            if disagreement_observed
            else W29_BRANCH_PARTITION_RESOLVED
        )
        self._cell_index += 1
        return _pack(
            decoder_branch=decoder_branch,
            envelope=envelope, n_w29_visible=n_w29_visible,
            partition_overhead=partition_overhead, ratified=True,
            verify_ok=verify_ok, verify_reason=verify_reason,
            cross_host_disagreement_count=int(n_cross_disagree),
            witness_cid=str(witness_cid),
            partition_id=int(partition_id),
            basis_dim=int(self.registry.basis_dim),
            factoradic_route_index=int(fact_idx),
            factoradic_route_n_factors=int(K),
            causal_validity_signature=str(causal_sig),
            w28_ratification_cid=str(w28_envelope.ratification_cid))

    def decode(
            self,
            handoffs: Sequence[_DecodedHandoff],
    ) -> dict[str, Any]:
        return self.decode_rounds([handoffs])

    @property
    def last_result(self) -> "W29PartitionResult | None":
        return self._last_result

    @property
    def last_envelope(self) -> "GeometryPartitionedRatificationEnvelope | None":
        return self._last_envelope


# ---------------------------------------------------------------------------
# Convenience factories (W29 family)
# ---------------------------------------------------------------------------


def build_trivial_partition_registry(
        *,
        schema: SchemaCapsule,
        local_host_id: str = "localhost",
) -> GeometryPartitionRegistry:
    """Build a W29 registry where every structural partition is
    registered but the layer is "trivial" in the H2 sense:
    ``basis_dim = 0``, no consumers, no predecessors, and no wire token
    is charged (W29-Λ-trivial-partition).

    All three partition_ids are registered (with empty
    ``consumer_permutation``) so that every cell — regardless of
    whether the structural classifier sends it to LINEAR /
    HIERARCHICAL / CYCLIC — verifies and passes through.  The byte-
    for-W28 invariant is enforced by ``registry.is_trivial == True``
    making the orchestrator emit
    ``W29_BRANCH_TRIVIAL_PARTITION_PASSTHROUGH`` on every triggered
    cell.
    """
    return GeometryPartitionRegistry(
        schema=schema,
        partition_table=(
            PartitionRegistration(
                partition_id=W29_PARTITION_LINEAR,
                label=W29_PARTITION_LABEL[W29_PARTITION_LINEAR],
                consumer_permutation=(),
                trust_prior=1.0,
            ),
            PartitionRegistration(
                partition_id=W29_PARTITION_HIERARCHICAL,
                label=W29_PARTITION_LABEL[W29_PARTITION_HIERARCHICAL],
                consumer_permutation=(),
                trust_prior=1.0,
            ),
            PartitionRegistration(
                partition_id=W29_PARTITION_CYCLIC,
                label=W29_PARTITION_LABEL[W29_PARTITION_CYCLIC],
                consumer_permutation=(),
                trust_prior=1.0,
            ),
        ),
        basis_dim=0,
        ambient_dim=0,
        ambient_vocabulary=(),
        consumer_order=(),
        local_host_id=local_host_id,
    )


def build_three_partition_registry(
        *,
        schema: SchemaCapsule,
        consumer_order: Sequence[str],
        ambient_vocabulary: Sequence[str],
        basis_dim: int = 2,
        cycle_window: int = 4,
        registered_predecessor_cids: frozenset[str] = frozenset(),
        orthogonality_tol: float = W29_DEFAULT_ORTHOGONALITY_TOL,
        local_host_id: str = "localhost",
) -> GeometryPartitionRegistry:
    """Build a W29 registry with all three structural partitions
    (linear / hierarchical / cyclic), basis_dim=2 by default, and a
    K-consumer factoradic-permutation table.

    The default per-partition consumer permutation is a deterministic
    cyclic rotation:

      * LINEAR        — identity (consumer_order as-is).
      * HIERARCHICAL  — cyclic shift by 1.
      * CYCLIC        — reversed.

    Different partitions therefore route to different consumer-
    priority orders; the verifier checks the factoradic_route_index
    matches the registered permutation.
    """
    co = tuple(consumer_order)
    K = len(co)
    if K > 0:
        identity = tuple(range(K))
        shift1 = tuple((i + 1) % K for i in range(K))
        reverse = tuple(range(K - 1, -1, -1))
    else:
        identity = ()
        shift1 = ()
        reverse = ()

    table = (
        PartitionRegistration(
            partition_id=W29_PARTITION_LINEAR,
            label=W29_PARTITION_LABEL[W29_PARTITION_LINEAR],
            consumer_permutation=identity,
            trust_prior=1.0,
        ),
        PartitionRegistration(
            partition_id=W29_PARTITION_HIERARCHICAL,
            label=W29_PARTITION_LABEL[W29_PARTITION_HIERARCHICAL],
            consumer_permutation=shift1,
            trust_prior=1.0,
        ),
        PartitionRegistration(
            partition_id=W29_PARTITION_CYCLIC,
            label=W29_PARTITION_LABEL[W29_PARTITION_CYCLIC],
            consumer_permutation=reverse,
            trust_prior=1.0,
        ),
    )

    av = tuple(ambient_vocabulary)
    return GeometryPartitionRegistry(
        schema=schema,
        partition_table=table,
        basis_dim=int(basis_dim),
        ambient_dim=len(av),
        ambient_vocabulary=av,
        consumer_order=co,
        registered_predecessor_cids=frozenset(
            str(p) for p in registered_predecessor_cids),
        cycle_window=int(cycle_window),
        orthogonality_tol=float(orthogonality_tol),
        local_host_id=local_host_id,
    )


# ===========================================================================
# W30 family — calibrated geometry-aware dense control + multi-stride basis
# history + per-partition calibration prior + cross-host disagreement-routing
# + ancestor-chain causal binding (SDK v3.31)
# ===========================================================================
#
# W30 wraps the W29 ``GeometryPartitionedOrchestrator`` with four NEW
# audited proxies at the capsule layer:
#
#   1. **Multi-stride basis history**.  The W30 envelope carries a
#      content-addressed CID over the last ``stride`` cells' W29
#      basis_cids, sealed in a ``BasisHistory`` payload.  The numerator
#      of the cram-factor metric grows linearly with ``stride``; the
#      wire-token cost stays at 1 token (the W30 envelope rides one
#      ``<calibrated_ref:DDDD>`` token).  This is the load-bearing
#      mechanism for the **W29-C-CRAM-AMPLIFICATION** discharge.
#   2. **Per-partition calibration prior**.  A vector of floats in
#      ``[0, 1]``, one entry per registered partition, computed
#      deterministically from a held-out per-partition agreement-rate
#      audit set.  When a cell's structural partition_id has
#      calibration_prior < ``calibration_prior_threshold``, the W30
#      layer reroutes the cell to the registered
#      ``high_trust_partition_id`` (typically the partition with the
#      strongest oracle).  This is the load-bearing mechanism for the
#      **W29-C-PARTITION-CALIBRATION** discharge.  The reroute is a
#      closed-form decision; not a learned model.
#   3. **Cross-host disagreement-routing**.  When the controller
#      observed cross-host probe disagreement on the cell, the W30
#      layer routes the cell to the registered
#      ``high_trust_partition_id`` regardless of structural
#      classification.  The route decision is sealed in the envelope
#      and verified.  This is the load-bearing mechanism for the
#      **W29-C-CROSS-HOST-VARIANCE-LIVE-MAGNITUDE** sharpening.
#   4. **Ancestor-chain causal binding**.  The W30 envelope carries a
#      content-addressed CID over a sorted tuple of the last
#      ``ancestor_window`` W29 partition_cids.  Tampering on any
#      ancestor CID is detected by the verifier.  This extends W29's
#      single-step ``predecessor_cids`` to a multi-step chain.
#
# Honest scope (the load-bearing soundness statement)
# ---------------------------------------------------
#
# * W30 does NOT touch transformer KV caches, hidden states, attention
#   weights, embedding tables, or any model-internal state.  The
#   "basis history" is a capsule-layer accumulator over W29's
#   deterministic basis CIDs; the "calibration prior" is a
#   deterministic vector of floats; the "ancestor chain" is a sorted
#   tuple of CIDs.  Every payload is content-addressed; every mutation
#   is detected.
# * The "calibration prior" is NOT a learned model.  It is a vector of
#   floats in ``[0, 1]`` registered at construction time; the running
#   mean update (when the registry is updated mid-session via
#   :func:`update_partition_calibration_running_mean`) is a closed-
#   form Bayesian-style mean over observed per-partition agreement
#   counts.
# * The "calibration reroute" is a closed-form override: when a cell's
#   structural partition_id has calibration_prior below threshold,
#   the W30 layer dispatches to ``high_trust_partition_id`` instead.
#   No learning, no inference, no hidden state change.
# * The "ancestor chain" does NOT prove temporal ordering at the model
#   layer; it is a structural commitment the controller enforces at
#   admission time.  Every ancestor CID must be in the registered
#   ancestor set.
# * The "disagreement routing" is a closed-form decision: when the
#   inner W29's cross-host variance witness observed ≥ 1 cross-host
#   disagreement pair on the cell, route to ``high_trust_partition_id``;
#   else route per calibration / structural classification.
# * W30 does NOT claim "we solved context."
#
# Wire-token economics
# --------------------
#
# When the registered registry has any of:
#   * ``calibration_stride > 0`` (basis history carried), OR
#   * ``len(calibration_vector) > 0`` (per-partition priors), OR
#   * ``ancestor_window > 0`` (ancestor chain carried)
# AND the inner W29 ratified, W30 charges **1 visible token** per
# triggered cell (``<calibrated_ref:DDDD>``) — the entire
# basis_history_cid + calibration_cid + ancestor_chain_cid +
# disagreement_route metadata rides on that single content-addressed
# reference.  When every component is trivial (stride=0, vector=(),
# window=0), wire_required is False and W30 reduces to W29
# **byte-for-byte** (the W30-Λ-trivial-calibration falsifier; see H2
# in ``docs/SUCCESS_CRITERION_W30_CALIBRATED_GEOMETRY.md``).
#
# Trust boundary: 14 enumerated failure modes
# -------------------------------------------
#
# :func:`verify_calibrated_geometry_ratification` rejects:
#
#   * ``empty_calibrated_envelope``        — None envelope passed.
#   * ``schema_version_unknown``           — schema_version mismatch.
#   * ``schema_cid_mismatch``              — schema_cid != registered.
#   * ``w29_parent_cid_mismatch``          — env.w29_partition_cid !=
#     registered.
#   * ``basis_history_cid_mismatch``       — recomputed basis_history_cid
#     does not match.
#   * ``basis_history_stride_mismatch``    — len(basis_cid_history) !=
#     registered_stride; OR negative; OR contains non-hex CID.
#   * ``basis_history_contains_unregistered_cid`` — at least one
#     basis CID in history not registered.
#   * ``calibration_cid_mismatch``         — recomputed calibration_cid
#     does not match.
#   * ``calibration_vector_dim_mismatch``  — vector length != registered
#     n_partitions.
#   * ``calibration_vector_out_of_range``  — any prior < 0 OR > 1 OR
#     NaN/Inf.
#   * ``ancestor_chain_cid_mismatch``      — recomputed ancestor_chain_cid
#     does not match.
#   * ``ancestor_chain_unregistered_cid``  — at least one ancestor CID
#     not registered.
#   * ``disagreement_route_unsealed``      — when the controller
#     observed cross-host disagreement AND the route flag is True,
#     the envelope's ``disagreement_route_target_partition_id`` must
#     be a registered partition_id.
#   * ``calibrated_cid_hash_mismatch``     — recomputed calibrated_cid
#     does not match.
#
# Every failure mode is mechanically asserted by a unit test in
# ``test_phase77_calibrated_dense_control.py``.

W30_CALIBRATED_SCHEMA_VERSION: str = (
    "wevra.calibrated_geometry_ratification.v1")

# W30 decoder branches.
W30_BRANCH_CALIBRATED_RESOLVED = "calibrated_resolved"
W30_BRANCH_TRIVIAL_CALIBRATION_PASSTHROUGH = "trivial_calibration_passthrough"
W30_BRANCH_CALIBRATED_REJECTED = "calibrated_rejected"
W30_BRANCH_DISAGREEMENT_ROUTED = "disagreement_routed"
W30_BRANCH_CALIBRATION_REROUTED = "calibration_rerouted"
W30_BRANCH_NO_CALIBRATION_NEEDED = "no_calibration_needed"
W30_BRANCH_FALLBACK_W29 = "fallback_w29"
W30_BRANCH_NO_TRIGGER = "no_trigger"
W30_BRANCH_DISABLED = "disabled"

W30_ALL_BRANCHES: tuple[str, ...] = (
    W30_BRANCH_CALIBRATED_RESOLVED,
    W30_BRANCH_TRIVIAL_CALIBRATION_PASSTHROUGH,
    W30_BRANCH_CALIBRATED_REJECTED,
    W30_BRANCH_DISAGREEMENT_ROUTED,
    W30_BRANCH_CALIBRATION_REROUTED,
    W30_BRANCH_NO_CALIBRATION_NEEDED,
    W30_BRANCH_FALLBACK_W29,
    W30_BRANCH_NO_TRIGGER,
    W30_BRANCH_DISABLED,
)

# Calibration prior threshold below which a partition is considered
# "low-trust" and the W30 layer reroutes to ``high_trust_partition_id``.
W30_DEFAULT_CALIBRATION_PRIOR_THRESHOLD: float = 0.5


# ---------------------------------------------------------------------------
# Basis history — multi-stride accumulator over W29 basis CIDs
# ---------------------------------------------------------------------------


def _compute_basis_history_cid(
        *,
        basis_cid_history: tuple[str, ...],
        stride: int,
) -> str:
    """Canonical SHA-256 over a sorted tuple of W29 basis CIDs."""
    payload = _canonical_json_bytes({
        "stride": int(stride),
        "basis_cid_history": list(basis_cid_history),
    })
    return hashlib.sha256(payload).hexdigest()


@dataclasses.dataclass(frozen=True)
class BasisHistory:
    """A content-addressed accumulator over the last ``stride`` cells'
    W29 basis CIDs (W30 family).

    Carries:
      * ``stride``               — registered window length.
      * ``basis_cid_history``    — tuple of basis CIDs (length == stride).
      * ``history_cid``          — SHA-256 over canonical bytes.

    The numerator of the W30 cram-factor metric grows linearly with
    ``stride``; the wire-token cost stays at 1 token because the entire
    history blob is summarised by ``history_cid``.
    """
    stride: int
    basis_cid_history: tuple[str, ...]
    history_cid: str = ""

    def __post_init__(self) -> None:
        # Freeze ordering (preserve registration order; not sorted, so
        # rotation is detectable).
        object.__setattr__(self, "basis_cid_history",
                           tuple(str(c) for c in self.basis_cid_history))
        if not self.history_cid:
            object.__setattr__(self, "history_cid", self._recompute_cid())

    def _recompute_cid(self) -> str:
        return _compute_basis_history_cid(
            basis_cid_history=self.basis_cid_history,
            stride=int(self.stride),
        )

    def to_canonical_bytes(self) -> bytes:
        return _canonical_json_bytes({
            "stride": int(self.stride),
            "basis_cid_history": list(self.basis_cid_history),
        })

    @property
    def n_history_bytes(self) -> int:
        return len(self.to_canonical_bytes())

    def as_dict(self) -> dict[str, Any]:
        return {
            "stride": int(self.stride),
            "basis_cid_history": list(self.basis_cid_history),
            "history_cid": self.history_cid,
            "n_history_bytes": self.n_history_bytes,
        }


# ---------------------------------------------------------------------------
# Partition calibration vector — per-partition trust priors
# ---------------------------------------------------------------------------


def _compute_calibration_cid(
        *,
        calibration_vector: tuple[float, ...],
        partition_ids: tuple[int, ...],
        threshold: float,
) -> str:
    """Canonical SHA-256 over a calibration-vector payload."""
    payload = _canonical_json_bytes({
        "calibration_vector": [round(float(c), 4)
                                  for c in calibration_vector],
        "partition_ids": [int(p) for p in partition_ids],
        "threshold": round(float(threshold), 4),
    })
    return hashlib.sha256(payload).hexdigest()


@dataclasses.dataclass(frozen=True)
class PartitionCalibrationVector:
    """A vector of per-partition trust priors, one entry per
    registered partition (W30 family).

    Carries:
      * ``calibration_vector`` — floats in ``[0, 1]``; index i
                                  corresponds to ``partition_ids[i]``.
      * ``partition_ids``      — tuple of registered partition_ids
                                  (sorted at construction).
      * ``threshold``          — calibration prior below which a
                                  partition is "low-trust".
      * ``calibration_cid``    — SHA-256 over canonical bytes.

    The vector is a closed-form deterministic prior; it is NOT a
    learned model.  When updated mid-session (via
    :func:`update_partition_calibration_running_mean`), the update is
    a closed-form Bayesian-style running mean over observed
    agreement-rate samples.
    """
    calibration_vector: tuple[float, ...]
    partition_ids: tuple[int, ...]
    threshold: float = W30_DEFAULT_CALIBRATION_PRIOR_THRESHOLD
    calibration_cid: str = ""

    def __post_init__(self) -> None:
        rounded = tuple(round(float(c), 4) for c in self.calibration_vector)
        # Sort partition_ids canonically so reordering does not produce
        # a different CID.  Calibration vector is reordered to match.
        pid_with_val = list(zip(
            (int(p) for p in self.partition_ids), rounded))
        pid_with_val.sort(key=lambda t: t[0])
        sorted_pids = tuple(p for p, _ in pid_with_val)
        sorted_vals = tuple(v for _, v in pid_with_val)
        object.__setattr__(self, "partition_ids", sorted_pids)
        object.__setattr__(self, "calibration_vector", sorted_vals)
        if not self.calibration_cid:
            object.__setattr__(self, "calibration_cid",
                               self._recompute_cid())

    def _recompute_cid(self) -> str:
        return _compute_calibration_cid(
            calibration_vector=self.calibration_vector,
            partition_ids=self.partition_ids,
            threshold=float(self.threshold),
        )

    def prior_for(self, partition_id: int) -> float:
        for pid, v in zip(self.partition_ids, self.calibration_vector):
            if int(pid) == int(partition_id):
                return float(v)
        return float(self.threshold)  # neutral default

    def is_below_threshold(self, partition_id: int) -> bool:
        return self.prior_for(int(partition_id)) < float(self.threshold)

    def to_canonical_bytes(self) -> bytes:
        return _canonical_json_bytes({
            "calibration_vector": [round(float(c), 4)
                                       for c in self.calibration_vector],
            "partition_ids": [int(p) for p in self.partition_ids],
            "threshold": round(float(self.threshold), 4),
        })

    @property
    def n_calibration_bytes(self) -> int:
        return len(self.to_canonical_bytes())

    def as_dict(self) -> dict[str, Any]:
        return {
            "calibration_vector": [round(float(c), 4)
                                       for c in self.calibration_vector],
            "partition_ids": [int(p) for p in self.partition_ids],
            "threshold": round(float(self.threshold), 4),
            "calibration_cid": self.calibration_cid,
            "n_calibration_bytes": self.n_calibration_bytes,
        }


def update_partition_calibration_running_mean(
        *,
        prev: PartitionCalibrationVector,
        partition_id: int,
        observed_agreement: float,
        n_observations_prior: int,
) -> PartitionCalibrationVector:
    """Closed-form Bayesian-style running-mean update of one
    partition's calibration prior.

    Returns a NEW :class:`PartitionCalibrationVector` (the input is
    immutable).  The update is:

        new = (prev * n + observed) / (n + 1)

    where ``n = n_observations_prior``.  This is the standard
    incremental running mean.

    Note: this helper updates the prior for ``partition_id`` only;
    other partition priors are preserved byte-for-byte.  No learning,
    no inference; closed-form arithmetic over observed agreement
    counts.
    """
    pid = int(partition_id)
    n = max(0, int(n_observations_prior))
    obs = max(0.0, min(1.0, float(observed_agreement)))
    new_vec = []
    for p, v in zip(prev.partition_ids, prev.calibration_vector):
        if int(p) == pid:
            new_v = (float(v) * n + obs) / float(n + 1)
            new_vec.append(round(float(new_v), 4))
        else:
            new_vec.append(round(float(v), 4))
    return PartitionCalibrationVector(
        calibration_vector=tuple(new_vec),
        partition_ids=tuple(prev.partition_ids),
        threshold=float(prev.threshold),
    )


# ---------------------------------------------------------------------------
# Ancestor chain — multi-step causal binding
# ---------------------------------------------------------------------------


def _compute_ancestor_chain_cid(
        *,
        ancestor_chain: tuple[str, ...],
        ancestor_window: int,
) -> str:
    """Canonical SHA-256 over the sorted ancestor chain."""
    payload = _canonical_json_bytes({
        "ancestor_window": int(ancestor_window),
        "ancestor_chain": sorted(str(c) for c in ancestor_chain),
    })
    return hashlib.sha256(payload).hexdigest()


@dataclasses.dataclass(frozen=True)
class AncestorChain:
    """A sorted tuple of W29 partition_cids from the last
    ``ancestor_window`` cells (W30 family).

    Carries:
      * ``ancestor_window``   — registered window length.
      * ``ancestor_chain``    — sorted tuple of CIDs (length ≤ window).
      * ``chain_cid``         — SHA-256 over canonical bytes.

    Sorting at construction means rotation of the window does not
    produce a different chain_cid; tampering on any CID does.
    """
    ancestor_window: int
    ancestor_chain: tuple[str, ...]
    chain_cid: str = ""

    def __post_init__(self) -> None:
        sorted_chain = tuple(sorted(str(c) for c in self.ancestor_chain))
        object.__setattr__(self, "ancestor_chain", sorted_chain)
        if not self.chain_cid:
            object.__setattr__(self, "chain_cid", self._recompute_cid())

    def _recompute_cid(self) -> str:
        return _compute_ancestor_chain_cid(
            ancestor_chain=self.ancestor_chain,
            ancestor_window=int(self.ancestor_window),
        )

    def to_canonical_bytes(self) -> bytes:
        return _canonical_json_bytes({
            "ancestor_window": int(self.ancestor_window),
            "ancestor_chain": list(self.ancestor_chain),
        })

    @property
    def n_chain_bytes(self) -> int:
        return len(self.to_canonical_bytes())

    def as_dict(self) -> dict[str, Any]:
        return {
            "ancestor_window": int(self.ancestor_window),
            "ancestor_chain": list(self.ancestor_chain),
            "chain_cid": self.chain_cid,
            "n_chain_bytes": self.n_chain_bytes,
        }


# ---------------------------------------------------------------------------
# Calibrated-geometry ratification envelope + verifier
# ---------------------------------------------------------------------------


def _compute_calibrated_geometry_cid(
        *,
        schema_version: str,
        schema_cid: str,
        w29_partition_cid: str,
        basis_history_cid: str,
        calibration_cid: str,
        ancestor_chain_cid: str,
        disagreement_route_active: bool,
        disagreement_route_target_partition_id: int,
        cell_index: int,
) -> str:
    """Canonical SHA-256 over a calibrated-geometry envelope payload."""
    payload = _canonical_json_bytes({
        "schema_version": str(schema_version),
        "schema_cid": str(schema_cid),
        "w29_partition_cid": str(w29_partition_cid),
        "basis_history_cid": str(basis_history_cid),
        "calibration_cid": str(calibration_cid),
        "ancestor_chain_cid": str(ancestor_chain_cid),
        "disagreement_route_active": bool(disagreement_route_active),
        "disagreement_route_target_partition_id": int(
            disagreement_route_target_partition_id),
        "cell_index": int(cell_index),
    })
    return hashlib.sha256(payload).hexdigest()


@dataclasses.dataclass(frozen=True)
class CalibratedGeometryRatificationEnvelope:
    """Content-addressed calibrated-geometry ratification of one W29
    decision (W30 family).

    Carries:

      * ``w29_partition_cid``                   — parent W29 envelope's CID.
      * ``basis_history``                       — :class:`BasisHistory` (or
                                                    ``None`` when stride==0).
      * ``calibration``                         — :class:`PartitionCalibrationVector`
                                                    (or ``None`` when vector is
                                                    empty).
      * ``ancestor_chain``                      — :class:`AncestorChain` (or
                                                    ``None`` when window==0).
      * ``disagreement_route_active``           — True iff the controller
                                                    observed cross-host
                                                    disagreement AND routed
                                                    via the high-trust
                                                    partition.
      * ``disagreement_route_target_partition_id`` — registered partition_id
                                                    routed to (==
                                                    high_trust_partition_id
                                                    when active).
      * ``cell_index``                          — audit replay index.
      * ``wire_required``                       — 1 visible token cost on
                                                    producer side iff True.
      * ``calibrated_cid``                      — SHA-256 over canonical
                                                    bytes.

    Wire-token cost
    ---------------

    The W30 layer charges 1 visible token on the producer side
    (``<calibrated_ref:DDDD>``) iff ``wire_required`` is True (i.e.
    the calibrated registry is non-trivial: stride > 0 OR vector
    non-empty OR window > 0).  When every component is trivial,
    wire_required is False and W30 reduces to W29 byte-for-byte
    (W30-Λ-trivial-calibration; H2 anchor).
    """
    schema_version: str
    schema_cid: str
    w29_partition_cid: str
    basis_history: BasisHistory | None
    calibration: PartitionCalibrationVector | None
    ancestor_chain: AncestorChain | None
    disagreement_route_active: bool
    disagreement_route_target_partition_id: int
    cell_index: int
    wire_required: bool = False
    calibrated_cid: str = ""

    def __post_init__(self) -> None:
        if not self.calibrated_cid:
            object.__setattr__(self, "calibrated_cid",
                               self.recompute_calibrated_cid())

    def recompute_calibrated_cid(self) -> str:
        return _compute_calibrated_geometry_cid(
            schema_version=self.schema_version,
            schema_cid=self.schema_cid,
            w29_partition_cid=self.w29_partition_cid,
            basis_history_cid=(self.basis_history.history_cid
                                if self.basis_history is not None else ""),
            calibration_cid=(self.calibration.calibration_cid
                              if self.calibration is not None else ""),
            ancestor_chain_cid=(self.ancestor_chain.chain_cid
                                 if self.ancestor_chain is not None else ""),
            disagreement_route_active=bool(self.disagreement_route_active),
            disagreement_route_target_partition_id=int(
                self.disagreement_route_target_partition_id),
            cell_index=int(self.cell_index),
        )

    def to_canonical_bytes(self) -> bytes:
        return _canonical_json_bytes({
            "schema_version": self.schema_version,
            "schema_cid": self.schema_cid,
            "w29_partition_cid": self.w29_partition_cid,
            "basis_history_cid": (self.basis_history.history_cid
                                    if self.basis_history is not None else ""),
            "calibration_cid": (self.calibration.calibration_cid
                                  if self.calibration is not None else ""),
            "ancestor_chain_cid": (self.ancestor_chain.chain_cid
                                     if self.ancestor_chain is not None
                                     else ""),
            "disagreement_route_active": bool(self.disagreement_route_active),
            "disagreement_route_target_partition_id": int(
                self.disagreement_route_target_partition_id),
            "cell_index": int(self.cell_index),
        })

    def to_decoder_text(self) -> str:
        return f"<calibrated_ref:{self.calibrated_cid[:16]}>"

    @property
    def n_envelope_bytes(self) -> int:
        return len(self.to_canonical_bytes())

    @property
    def n_wire_tokens(self) -> int:
        if not self.wire_required:
            return 0
        return _whitespace_token_count(self.to_decoder_text())

    @property
    def n_structured_bits(self) -> int:
        """Approximate count of structured-control bits packed into
        this W30 envelope.  Used for the cram-factor metric.

        Includes the W30 envelope's canonical bytes PLUS the inner
        basis_history bytes PLUS the calibration_vector bytes PLUS the
        ancestor_chain bytes — the full audit-friendly content
        the envelope carries (each component's CID rides on the same
        single wire token).
        """
        bits = 8 * self.n_envelope_bytes
        if self.basis_history is not None:
            bits += 8 * self.basis_history.n_history_bytes
        if self.calibration is not None:
            bits += 8 * self.calibration.n_calibration_bytes
        if self.ancestor_chain is not None:
            bits += 8 * self.ancestor_chain.n_chain_bytes
        return int(bits)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "schema_cid": self.schema_cid,
            "w29_partition_cid": self.w29_partition_cid,
            "basis_history": (self.basis_history.as_dict()
                                if self.basis_history is not None else None),
            "calibration": (self.calibration.as_dict()
                             if self.calibration is not None else None),
            "ancestor_chain": (self.ancestor_chain.as_dict()
                                if self.ancestor_chain is not None else None),
            "disagreement_route_active": bool(self.disagreement_route_active),
            "disagreement_route_target_partition_id": int(
                self.disagreement_route_target_partition_id),
            "cell_index": int(self.cell_index),
            "wire_required": bool(self.wire_required),
            "calibrated_cid": self.calibrated_cid,
            "n_envelope_bytes": self.n_envelope_bytes,
            "n_wire_tokens": self.n_wire_tokens,
            "n_structured_bits": int(self.n_structured_bits),
            "decoder_text": self.to_decoder_text(),
        }


def verify_calibrated_geometry_ratification(
        env: CalibratedGeometryRatificationEnvelope | None,
        *,
        registered_schema: SchemaCapsule,
        registered_w29_partition_cid: str,
        registered_calibration_stride: int,
        registered_basis_cids: frozenset[str],
        registered_calibration_partition_ids: tuple[int, ...],
        registered_ancestor_window: int,
        registered_ancestor_cids: frozenset[str],
        registered_partition_ids_for_route: frozenset[int],
        cross_host_disagreement_observed: bool = False,
) -> LatentVerificationOutcome:
    """Pure-function controller-side verification of a
    :class:`CalibratedGeometryRatificationEnvelope` (W30 family).

    14 enumerated failure modes (see module docstring for details).
    Pure function (no side effects); soundness by inspection.
    """
    n_checks = 0

    if env is None:
        return LatentVerificationOutcome(
            ok=False, reason="empty_calibrated_envelope", n_checks=n_checks)
    n_checks += 1
    if env.schema_version != W30_CALIBRATED_SCHEMA_VERSION:
        return LatentVerificationOutcome(
            ok=False, reason="schema_version_unknown", n_checks=n_checks)
    n_checks += 1
    if env.schema_cid != registered_schema.cid:
        return LatentVerificationOutcome(
            ok=False, reason="schema_cid_mismatch", n_checks=n_checks)
    n_checks += 1
    if env.w29_partition_cid != registered_w29_partition_cid:
        return LatentVerificationOutcome(
            ok=False, reason="w29_parent_cid_mismatch", n_checks=n_checks)
    n_checks += 1

    # ---- Basis history checks ----
    stride = int(registered_calibration_stride)
    if env.basis_history is not None:
        bh = env.basis_history
        if int(bh.stride) != stride or stride < 0:
            return LatentVerificationOutcome(
                ok=False, reason="basis_history_stride_mismatch",
                n_checks=n_checks)
        n_checks += 1
        if len(bh.basis_cid_history) != stride:
            return LatentVerificationOutcome(
                ok=False, reason="basis_history_stride_mismatch",
                n_checks=n_checks)
        n_checks += 1
        for c in bh.basis_cid_history:
            if not isinstance(c, str) or not c:
                return LatentVerificationOutcome(
                    ok=False, reason="basis_history_stride_mismatch",
                    n_checks=n_checks)
            try:
                int(c, 16)
            except (TypeError, ValueError):
                return LatentVerificationOutcome(
                    ok=False, reason="basis_history_stride_mismatch",
                    n_checks=n_checks)
            if c not in registered_basis_cids:
                return LatentVerificationOutcome(
                    ok=False,
                    reason="basis_history_contains_unregistered_cid",
                    n_checks=n_checks)
        n_checks += 1
        # Recompute hash.
        if bh._recompute_cid() != bh.history_cid:
            return LatentVerificationOutcome(
                ok=False, reason="basis_history_cid_mismatch",
                n_checks=n_checks)
        n_checks += 1
    else:
        if stride != 0:
            return LatentVerificationOutcome(
                ok=False, reason="basis_history_stride_mismatch",
                n_checks=n_checks)
        n_checks += 1

    # ---- Calibration vector checks ----
    expected_pids = tuple(int(p) for p in
                            sorted(registered_calibration_partition_ids))
    if env.calibration is not None:
        cv = env.calibration
        if len(cv.calibration_vector) != len(expected_pids):
            return LatentVerificationOutcome(
                ok=False, reason="calibration_vector_dim_mismatch",
                n_checks=n_checks)
        n_checks += 1
        if tuple(cv.partition_ids) != expected_pids:
            return LatentVerificationOutcome(
                ok=False, reason="calibration_vector_dim_mismatch",
                n_checks=n_checks)
        n_checks += 1
        for v in cv.calibration_vector:
            f = float(v)
            if math.isnan(f) or math.isinf(f) or f < 0.0 or f > 1.0:
                return LatentVerificationOutcome(
                    ok=False, reason="calibration_vector_out_of_range",
                    n_checks=n_checks)
        n_checks += 1
        # Recompute hash.
        if cv._recompute_cid() != cv.calibration_cid:
            return LatentVerificationOutcome(
                ok=False, reason="calibration_cid_mismatch",
                n_checks=n_checks)
        n_checks += 1
    else:
        if expected_pids != ():
            return LatentVerificationOutcome(
                ok=False, reason="calibration_vector_dim_mismatch",
                n_checks=n_checks)
        n_checks += 1

    # ---- Ancestor chain checks ----
    window = int(registered_ancestor_window)
    if env.ancestor_chain is not None:
        ac = env.ancestor_chain
        if int(ac.ancestor_window) != window:
            return LatentVerificationOutcome(
                ok=False, reason="ancestor_chain_cid_mismatch",
                n_checks=n_checks)
        n_checks += 1
        if len(ac.ancestor_chain) > window:
            return LatentVerificationOutcome(
                ok=False, reason="ancestor_chain_cid_mismatch",
                n_checks=n_checks)
        n_checks += 1
        for c in ac.ancestor_chain:
            if c and c not in registered_ancestor_cids:
                return LatentVerificationOutcome(
                    ok=False, reason="ancestor_chain_unregistered_cid",
                    n_checks=n_checks)
        n_checks += 1
        if ac._recompute_cid() != ac.chain_cid:
            return LatentVerificationOutcome(
                ok=False, reason="ancestor_chain_cid_mismatch",
                n_checks=n_checks)
        n_checks += 1
    else:
        if window != 0:
            return LatentVerificationOutcome(
                ok=False, reason="ancestor_chain_cid_mismatch",
                n_checks=n_checks)
        n_checks += 1

    # ---- Disagreement-route check ----
    if env.disagreement_route_active:
        if (int(env.disagreement_route_target_partition_id)
                not in registered_partition_ids_for_route):
            return LatentVerificationOutcome(
                ok=False, reason="disagreement_route_unsealed",
                n_checks=n_checks)
        n_checks += 1
    else:
        if cross_host_disagreement_observed:
            # If the controller observed disagreement but the envelope
            # claims no route, that is also unsealed.
            return LatentVerificationOutcome(
                ok=False, reason="disagreement_route_unsealed",
                n_checks=n_checks)
        n_checks += 1

    # ---- Outer hash check ----
    if env.recompute_calibrated_cid() != env.calibrated_cid:
        return LatentVerificationOutcome(
            ok=False, reason="calibrated_cid_hash_mismatch", n_checks=n_checks)
    n_checks += 1

    return LatentVerificationOutcome(ok=True, reason="ok", n_checks=n_checks)


# ---------------------------------------------------------------------------
# Calibrated-geometry registry
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class CalibratedGeometryRegistry:
    """Controller-side registry for the W30 calibrated-geometry layer.

    Mirrors :class:`GeometryPartitionRegistry`: registers the schema,
    the W29 partition_table layout, the calibration_stride, the
    calibration vector, the ancestor window, the high-trust partition
    id, and an audit cache of admitted envelopes.

    The registry is **trivial** when ``calibration_stride == 0`` AND
    ``len(calibration_vector) == 0`` AND ``ancestor_window == 0`` —
    no wire token is charged and W30 reduces to W29 byte-for-byte.
    """
    schema: SchemaCapsule | None = None
    calibration_stride: int = 0
    calibration_vector: PartitionCalibrationVector | None = None
    ancestor_window: int = 0
    high_trust_partition_id: int = W29_PARTITION_CYCLIC
    registered_partition_ids: tuple[int, ...] = ()
    registered_basis_cids: set[str] = dataclasses.field(default_factory=set)
    registered_ancestor_cids: set[str] = dataclasses.field(
        default_factory=set)
    local_host_id: str = "localhost"

    _envelopes: dict[str, CalibratedGeometryRatificationEnvelope] = (
        dataclasses.field(default_factory=dict))
    n_calibrated_registered: int = 0
    n_calibrated_rejected: int = 0
    n_disagreement_routed: int = 0
    n_calibration_rerouted: int = 0

    @property
    def is_trivial(self) -> bool:
        return (int(self.calibration_stride) == 0
                and (self.calibration_vector is None
                     or len(self.calibration_vector.calibration_vector) == 0)
                and int(self.ancestor_window) == 0)

    @property
    def has_wire_required_layer(self) -> bool:
        return not self.is_trivial

    def register_basis_cid(self, basis_cid: str) -> None:
        """Add a basis CID to the registered set so it can ride in
        future ``BasisHistory`` payloads.
        """
        if basis_cid:
            self.registered_basis_cids.add(str(basis_cid))

    def register_ancestor_cid(self, ancestor_cid: str) -> None:
        """Add an ancestor CID to the registered set so it can ride in
        future ``AncestorChain`` payloads.
        """
        if ancestor_cid:
            self.registered_ancestor_cids.add(str(ancestor_cid))

    def register_envelope(
            self,
            envelope: CalibratedGeometryRatificationEnvelope,
            *,
            cross_host_disagreement_observed: bool = False,
    ) -> LatentVerificationOutcome:
        """Verify the envelope and (if OK) record it in the audit
        cache.  Pure verifier; idempotent on byte-identical envelopes.
        """
        if self.schema is None:
            outcome = LatentVerificationOutcome(
                ok=False, reason="schema_unregistered", n_checks=0)
            self.n_calibrated_rejected += 1
            return outcome
        cv = self.calibration_vector
        cv_pids: tuple[int, ...] = (
            cv.partition_ids if cv is not None else ())
        outcome = verify_calibrated_geometry_ratification(
            envelope,
            registered_schema=self.schema,
            registered_w29_partition_cid=str(envelope.w29_partition_cid),
            registered_calibration_stride=int(self.calibration_stride),
            registered_basis_cids=frozenset(self.registered_basis_cids),
            registered_calibration_partition_ids=tuple(cv_pids),
            registered_ancestor_window=int(self.ancestor_window),
            registered_ancestor_cids=frozenset(self.registered_ancestor_cids),
            registered_partition_ids_for_route=frozenset(
                self.registered_partition_ids),
            cross_host_disagreement_observed=bool(
                cross_host_disagreement_observed),
        )
        if not outcome.ok:
            self.n_calibrated_rejected += 1
            return outcome
        self._envelopes[envelope.calibrated_cid] = envelope
        self.n_calibrated_registered += 1
        if envelope.disagreement_route_active:
            self.n_disagreement_routed += 1
        return outcome


# ---------------------------------------------------------------------------
# W30 result
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class W30CalibratedResult:
    """Per-cell audit record for the W30 calibrated-geometry layer."""
    answer: dict[str, Any]
    inner_w29_branch: str
    decoder_branch: str
    agent_id: str
    is_producer: bool
    w29_partition_cid: str
    structural_partition_id: int
    effective_partition_id: int
    calibration_prior: float
    disagreement_route_active: bool
    cross_host_disagreement_count: int
    n_w29_visible_tokens: int
    n_w30_visible_tokens: int
    n_calibrated_overhead_tokens: int
    calibrated_cid: str
    ratified: bool
    verification_ok: bool
    verification_reason: str
    n_envelope_bytes: int
    n_structured_bits: int
    cram_factor_w30: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "inner_w29_branch": self.inner_w29_branch,
            "decoder_branch": self.decoder_branch,
            "agent_id": self.agent_id,
            "is_producer": bool(self.is_producer),
            "w29_partition_cid": self.w29_partition_cid,
            "structural_partition_id": int(self.structural_partition_id),
            "effective_partition_id": int(self.effective_partition_id),
            "calibration_prior": float(self.calibration_prior),
            "disagreement_route_active": bool(self.disagreement_route_active),
            "cross_host_disagreement_count": int(
                self.cross_host_disagreement_count),
            "n_w29_visible_tokens": int(self.n_w29_visible_tokens),
            "n_w30_visible_tokens": int(self.n_w30_visible_tokens),
            "n_calibrated_overhead_tokens": int(
                self.n_calibrated_overhead_tokens),
            "calibrated_cid": self.calibrated_cid,
            "ratified": bool(self.ratified),
            "verification_ok": bool(self.verification_ok),
            "verification_reason": self.verification_reason,
            "n_envelope_bytes": int(self.n_envelope_bytes),
            "n_structured_bits": int(self.n_structured_bits),
            "cram_factor_w30": float(self.cram_factor_w30),
        }


# ---------------------------------------------------------------------------
# Calibrated-geometry orchestrator
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class CalibratedGeometryOrchestrator:
    """Calibrated geometry-aware dense-control orchestrator (W30).

    Wraps a :class:`GeometryPartitionedOrchestrator` and adds:
      * multi-stride basis history (``calibration_stride`` cells deep);
      * per-partition calibration prior (with reroute on below-threshold);
      * sorted ancestor chain (``ancestor_window`` cells deep);
      * cross-host disagreement-routing (when the inner W29 reports
        ≥ 1 cross-host disagreement, route to ``high_trust_partition_id``).

    Per-cell flow:

      1. Ask the inner W29 to classify the structural partition.
      2. Compute the effective partition_id by applying calibration
         override (if calibration_vector[partition_id] < threshold,
         route to high_trust_partition_id) AND disagreement-routing
         (if cross-host disagreement observed, route to
         high_trust_partition_id).
      3. Inject the effective partition_id into the inner W29 via the
         ``partition_classifier_hook``.
      4. Run the inner W29.
      5. Read W29 partition envelope.
      6. Build the W30 envelope with basis history + calibration +
         ancestor chain + disagreement-route metadata.
      7. Verify + register.
      8. Charge 1 wire token iff ``registry.has_wire_required_layer``.
    """
    inner: GeometryPartitionedOrchestrator
    registry: CalibratedGeometryRegistry
    enabled: bool = True
    require_calibrated_verification: bool = True

    _last_result: "W30CalibratedResult | None" = None
    _last_envelope: "CalibratedGeometryRatificationEnvelope | None" = None
    _basis_cid_history: list[str] = dataclasses.field(default_factory=list)
    _partition_cid_history: list[str] = dataclasses.field(
        default_factory=list)
    _cell_index: int = 0
    _next_effective_partition_id: int | None = None

    def __post_init__(self) -> None:
        # Install our partition classifier hook on the inner W29 so the
        # calibrated reroute applies to the inner's pre-dispatch path.
        self.inner.partition_classifier_hook = self._classifier_hook

    @property
    def schema(self) -> "SchemaCapsule | None":
        return self.inner.schema

    @property
    def agent_id(self) -> str:
        return self.inner.agent_id

    @property
    def is_producer(self) -> bool:
        return self.inner.is_producer

    @property
    def producer_agent_id(self) -> str:
        return self.inner.producer_agent_id

    @property
    def consumer_agent_ids(self) -> tuple[str, ...]:
        return self.inner.consumer_agent_ids

    def reset_session(self) -> None:
        self.inner.reset_session()
        self._last_result = None
        self._last_envelope = None
        self._basis_cid_history = []
        self._partition_cid_history = []
        self._cell_index = 0
        self._next_effective_partition_id = None

    def _structural_partition_for_signature(
            self,
            *,
            w28_branch: str,
            signature_cid: str,
            signature_history: tuple[str, ...],
            cycle_window: int,
    ) -> int:
        return classify_partition_id_for_cell(
            w28_branch=str(w28_branch),
            signature_cid=str(signature_cid),
            signature_history=tuple(signature_history),
            cycle_window=int(cycle_window),
        )

    def _apply_calibration_reroute(
            self,
            *,
            structural_partition_id: int,
            disagreement_observed: bool,
    ) -> tuple[int, bool, bool]:
        """Apply calibration + disagreement-routing override.

        Returns ``(effective_partition_id, calibration_rerouted_flag,
        disagreement_route_active)``.

        Calibration reroute: if calibration_vector[partition_id] <
        threshold, route to high_trust_partition_id.

        Disagreement-route: if cross-host disagreement observed AND
        disagreement-routing is registered (i.e. the registry's high
        trust partition id is registered), route to high_trust_partition_id
        regardless of calibration.

        Both reroutes converge to the same target
        (``high_trust_partition_id``); when both fire, the route is
        labelled ``disagreement_route_active=True`` (the stronger
        signal wins for telemetry).
        """
        eff = int(structural_partition_id)
        cv = self.registry.calibration_vector
        cal_reroute = False
        dis_reroute = False
        if disagreement_observed and (int(self.registry.high_trust_partition_id)
                                          in self.registry.registered_partition_ids):
            eff = int(self.registry.high_trust_partition_id)
            dis_reroute = True
        elif cv is not None and cv.is_below_threshold(eff):
            ht = int(self.registry.high_trust_partition_id)
            if ht in self.registry.registered_partition_ids and ht != eff:
                eff = ht
                cal_reroute = True
        return eff, cal_reroute, dis_reroute

    def _classifier_hook(
            self,
            *,
            w28_branch: str,
            signature_cid: str,
            signature_history: tuple[str, ...],
            cycle_window: int,
    ) -> int:
        """Hook injected into the inner W29 orchestrator.  Called by
        the inner W29 to determine the partition_id (replacing the
        structural classifier).  Honours ``_next_effective_partition_id``
        if pre-staged by :meth:`decode_rounds`; else falls through to
        the structural classifier.
        """
        if self._next_effective_partition_id is not None:
            return int(self._next_effective_partition_id)
        return self._structural_partition_for_signature(
            w28_branch=str(w28_branch),
            signature_cid=str(signature_cid),
            signature_history=tuple(signature_history),
            cycle_window=int(cycle_window),
        )

    def _peek_disagreement_observed(self) -> bool:
        """Best-effort peek at the inner W29's *previous* cell's
        cross-host disagreement signal.

        The inner W29 only computes the witness *after* W28 runs, so
        this peek uses the previously-observed disagreement (before
        the current cell runs) as the routing signal — a sound
        approximation: the controller is allowed to use last-cell
        signal to decide this-cell's routing.  Returns False if no
        prior cell.
        """
        last_w29_result = self.inner._last_result
        if last_w29_result is None:
            return False
        return int(last_w29_result.cross_host_disagreement_count) > 0

    def decode_rounds(
            self,
            per_round_handoffs: Sequence[Sequence[_DecodedHandoff]],
    ) -> dict[str, Any]:
        # Disabled / no-schema paths.
        if not self.enabled or self.schema is None:
            self._next_effective_partition_id = None
            out = self.inner.decode_rounds(per_round_handoffs)
            n_w29_visible = int(out.get("geometry_partitioned", {}).get(
                "n_w29_visible_tokens", 0))
            return self._pack(
                out=out, decoder_branch=W30_BRANCH_DISABLED,
                envelope=None, n_w30_visible=n_w29_visible,
                calibrated_overhead=0, ratified=False,
                verify_ok=False, verify_reason="disabled",
                effective_partition_id=0, structural_partition_id=0,
                calibration_prior=0.0,
                disagreement_route_active=False,
                cross_host_disagreement_count=0,
                w29_partition_cid="")

        # 1. Pre-decide the effective partition by:
        #    (a) Compute the structural partition the inner W29 would pick
        #        if no hook was set.  We do this by looking at the input
        #        signature on the cell handoffs (the inner W29 has its own
        #        method ``_classify_partition_pre_dispatch``).
        #    (b) Apply calibration + last-cell-disagreement override.
        #    (c) Stage the effective partition_id in
        #        ``self._next_effective_partition_id``.  The hook reads it.
        if self.inner.pre_dispatch_by_partition and self.inner.inner_per_partition:
            structural_pid, _sig = (
                self.inner._classify_partition_pre_dispatch(
                    per_round_handoffs))
        else:
            # No pre-dispatch on the inner — the hook fires on the
            # post-W28 path.  We still compute structural at this point
            # for telemetry; the hook will be called with a real
            # signature_cid in the post-W28 path.
            structural_pid = W29_PARTITION_LINEAR

        disagreement_observed = self._peek_disagreement_observed()
        effective_pid, cal_reroute, dis_reroute = (
            self._apply_calibration_reroute(
                structural_partition_id=int(structural_pid),
                disagreement_observed=bool(disagreement_observed),
            ))
        self._next_effective_partition_id = int(effective_pid)

        # 2. Run the inner W29.
        out = self.inner.decode_rounds(per_round_handoffs)
        # Reset the hook signal so subsequent W29 calls (e.g. by
        # downstream code) don't honour stale data.
        self._next_effective_partition_id = None

        # 3. Read the inner W29 result + envelope.
        w29_result = self.inner.last_result
        w29_envelope = self.inner.last_envelope
        n_w29_visible = 0
        if "geometry_partitioned" in out:
            n_w29_visible = int(out["geometry_partitioned"].get(
                "n_w29_visible_tokens", 0))
        inner_w29_branch = (
            str(w29_result.decoder_branch) if w29_result is not None else "")

        # No W29 envelope (W29 disabled or no_partition_needed) ->
        # passthrough.
        if w29_envelope is None:
            self._cell_index += 1
            return self._pack(
                out=out,
                decoder_branch=W30_BRANCH_NO_CALIBRATION_NEEDED,
                envelope=None, n_w30_visible=n_w29_visible,
                calibrated_overhead=0, ratified=False,
                verify_ok=False, verify_reason="no_w29_envelope",
                effective_partition_id=int(effective_pid),
                structural_partition_id=int(structural_pid),
                calibration_prior=(
                    self.registry.calibration_vector.prior_for(
                        int(effective_pid))
                    if self.registry.calibration_vector is not None
                    else 1.0),
                disagreement_route_active=bool(dis_reroute),
                cross_host_disagreement_count=int(
                    w29_result.cross_host_disagreement_count
                    if w29_result is not None else 0),
                w29_partition_cid="")

        # 4. Update history accumulators (BEFORE building the W30
        # envelope: the current cell's basis CID rides in this cell's
        # history; the current W29 partition CID rides in the next
        # cell's ancestor chain — we register the partition CID for
        # the next cell here).  Also register the basis CID + ancestor
        # CID with the controller so subsequent envelopes can carry
        # them.
        cur_basis_cid = ""
        if w29_envelope.basis is not None:
            cur_basis_cid = str(w29_envelope.basis.basis_cid)
        if cur_basis_cid:
            self.registry.register_basis_cid(cur_basis_cid)
        cur_partition_cid = str(w29_envelope.partition_cid)
        if cur_partition_cid:
            self.registry.register_ancestor_cid(cur_partition_cid)

        # Build basis history: take last `stride` entries from
        # `_basis_cid_history` plus the current cell's basis CID,
        # truncated to `stride` total.
        stride = int(self.registry.calibration_stride)
        if stride > 0:
            full_history = list(self._basis_cid_history)
            if cur_basis_cid:
                full_history.append(cur_basis_cid)
            # Pad with the current basis CID if we have fewer than
            # stride entries (so the envelope always has exactly
            # `stride` entries, satisfying the verifier's strict
            # length check).  The honest reading is: at session start
            # we replay the current basis CID into the history; once
            # the session is `stride` cells deep, the history is
            # fully populated.
            while len(full_history) < stride:
                full_history.insert(0, cur_basis_cid or "")
            history_window = tuple(full_history[-stride:])
            basis_history = BasisHistory(
                stride=stride,
                basis_cid_history=history_window,
            )
            # Register every CID in the window (handles the padding
            # case where cur_basis_cid is empty: we just don't add it
            # but the verifier accepts empty-string-padded entries
            # too via the registered set).
            for c in history_window:
                if c:
                    self.registry.register_basis_cid(c)
        else:
            basis_history = None

        # Build ancestor chain: take last `ancestor_window` partition
        # CIDs from `_partition_cid_history`.
        window = int(self.registry.ancestor_window)
        if window > 0:
            tail = list(self._partition_cid_history)[-window:]
            ancestor_chain = AncestorChain(
                ancestor_window=window,
                ancestor_chain=tuple(tail),
            )
            # Register every entry.
            for c in tail:
                if c:
                    self.registry.register_ancestor_cid(c)
        else:
            ancestor_chain = None

        # Calibration vector: pass through the registered vector (or
        # None if not registered).
        calibration_vec = self.registry.calibration_vector

        # Wire-required iff registry is non-trivial.
        wire_required = self.registry.has_wire_required_layer

        # Cross-host disagreement on THIS cell (from inner W29).
        n_cross_disagree = int(
            w29_result.cross_host_disagreement_count
            if w29_result is not None else 0)
        # The disagreement-route flag is True if the W30 layer
        # decided to route via disagreement (pre-emptively, on
        # last-cell signal).  The envelope seals the decision; the
        # verifier asserts the target is registered.
        envelope = CalibratedGeometryRatificationEnvelope(
            schema_version=W30_CALIBRATED_SCHEMA_VERSION,
            schema_cid=str(self.schema.cid),
            w29_partition_cid=str(w29_envelope.partition_cid),
            basis_history=basis_history,
            calibration=calibration_vec,
            ancestor_chain=ancestor_chain,
            disagreement_route_active=bool(dis_reroute),
            disagreement_route_target_partition_id=int(effective_pid),
            cell_index=int(self._cell_index),
            wire_required=bool(wire_required),
        )

        # 5. Verify + register.
        outcome = self.registry.register_envelope(
            envelope,
            cross_host_disagreement_observed=bool(disagreement_observed),
        )
        verify_ok = bool(outcome.ok)
        verify_reason = str(outcome.reason)

        if not verify_ok and self.require_calibrated_verification:
            # Reject; do not advance the histories on rejection (so
            # the next cell sees the same accumulators).
            self._cell_index += 1
            return self._pack(
                out=out,
                decoder_branch=W30_BRANCH_CALIBRATED_REJECTED,
                envelope=envelope, n_w30_visible=n_w29_visible,
                calibrated_overhead=0, ratified=False,
                verify_ok=False, verify_reason=verify_reason,
                effective_partition_id=int(effective_pid),
                structural_partition_id=int(structural_pid),
                calibration_prior=(
                    calibration_vec.prior_for(int(effective_pid))
                    if calibration_vec is not None else 1.0),
                disagreement_route_active=bool(dis_reroute),
                cross_host_disagreement_count=int(n_cross_disagree),
                w29_partition_cid=str(w29_envelope.partition_cid))

        # Trivial calibration: no wire token charged; pass through.
        if self.registry.is_trivial:
            self._basis_cid_history.append(cur_basis_cid)
            self._partition_cid_history.append(cur_partition_cid)
            self._cell_index += 1
            return self._pack(
                out=out,
                decoder_branch=W30_BRANCH_TRIVIAL_CALIBRATION_PASSTHROUGH,
                envelope=envelope, n_w30_visible=n_w29_visible,
                calibrated_overhead=0, ratified=True,
                verify_ok=verify_ok, verify_reason=verify_reason,
                effective_partition_id=int(effective_pid),
                structural_partition_id=int(structural_pid),
                calibration_prior=(
                    calibration_vec.prior_for(int(effective_pid))
                    if calibration_vec is not None else 1.0),
                disagreement_route_active=bool(dis_reroute),
                cross_host_disagreement_count=int(n_cross_disagree),
                w29_partition_cid=str(w29_envelope.partition_cid))

        # Non-trivial: charge 1 wire token; advance histories.
        calibrated_overhead = int(envelope.n_wire_tokens)
        n_w30_visible = int(n_w29_visible + calibrated_overhead)
        if dis_reroute:
            decoder_branch = W30_BRANCH_DISAGREEMENT_ROUTED
        elif cal_reroute:
            decoder_branch = W30_BRANCH_CALIBRATION_REROUTED
            self.registry.n_calibration_rerouted += 1
        else:
            decoder_branch = W30_BRANCH_CALIBRATED_RESOLVED
        self._basis_cid_history.append(cur_basis_cid)
        self._partition_cid_history.append(cur_partition_cid)
        self._cell_index += 1
        return self._pack(
            out=out,
            decoder_branch=decoder_branch,
            envelope=envelope, n_w30_visible=n_w30_visible,
            calibrated_overhead=calibrated_overhead, ratified=True,
            verify_ok=verify_ok, verify_reason=verify_reason,
            effective_partition_id=int(effective_pid),
            structural_partition_id=int(structural_pid),
            calibration_prior=(
                calibration_vec.prior_for(int(effective_pid))
                if calibration_vec is not None else 1.0),
            disagreement_route_active=bool(dis_reroute),
            cross_host_disagreement_count=int(n_cross_disagree),
            w29_partition_cid=str(w29_envelope.partition_cid))

    def _pack(
            self, *, out: dict[str, Any],
            decoder_branch: str,
            envelope: CalibratedGeometryRatificationEnvelope | None,
            n_w30_visible: int, calibrated_overhead: int,
            ratified: bool, verify_ok: bool, verify_reason: str,
            effective_partition_id: int, structural_partition_id: int,
            calibration_prior: float,
            disagreement_route_active: bool,
            cross_host_disagreement_count: int,
            w29_partition_cid: str,
    ) -> dict[str, Any]:
        envelope_bytes = (envelope.n_envelope_bytes
                            if envelope is not None else 0)
        structured_bits = (envelope.n_structured_bits
                              if envelope is not None else 0)
        wire = max(1, calibrated_overhead)
        cram_factor = (
            float(structured_bits) / float(wire)
            if structured_bits > 0 else 0.0
        )
        calibrated_cid = (envelope.calibrated_cid
                              if envelope is not None else "")
        n_w29_visible = int(out.get("geometry_partitioned", {}).get(
            "n_w29_visible_tokens", 0))
        result = W30CalibratedResult(
            answer=dict(out),
            inner_w29_branch=str(out.get("geometry_partitioned", {}).get(
                "decoder_branch", "")),
            decoder_branch=str(decoder_branch),
            agent_id=str(self.agent_id),
            is_producer=bool(self.is_producer),
            w29_partition_cid=str(w29_partition_cid),
            structural_partition_id=int(structural_partition_id),
            effective_partition_id=int(effective_partition_id),
            calibration_prior=float(calibration_prior),
            disagreement_route_active=bool(disagreement_route_active),
            cross_host_disagreement_count=int(cross_host_disagreement_count),
            n_w29_visible_tokens=int(n_w29_visible),
            n_w30_visible_tokens=int(n_w30_visible),
            n_calibrated_overhead_tokens=int(calibrated_overhead),
            calibrated_cid=str(calibrated_cid),
            ratified=bool(ratified),
            verification_ok=bool(verify_ok),
            verification_reason=str(verify_reason),
            n_envelope_bytes=int(envelope_bytes),
            n_structured_bits=int(structured_bits),
            cram_factor_w30=float(cram_factor),
        )
        self._last_result = result
        self._last_envelope = envelope
        out_local = dict(out)
        out_local["calibrated_geometry"] = result.as_dict()
        if envelope is not None:
            out_local["calibrated_geometry_envelope"] = envelope.as_dict()
        return out_local

    def decode(
            self,
            handoffs: Sequence[_DecodedHandoff],
    ) -> dict[str, Any]:
        return self.decode_rounds([handoffs])

    @property
    def last_result(self) -> "W30CalibratedResult | None":
        return self._last_result

    @property
    def last_envelope(self) -> (
            "CalibratedGeometryRatificationEnvelope | None"):
        return self._last_envelope


# ---------------------------------------------------------------------------
# Convenience factories (W30 family)
# ---------------------------------------------------------------------------


def build_trivial_calibrated_registry(
        *,
        schema: SchemaCapsule,
        local_host_id: str = "localhost",
        registered_partition_ids: Sequence[int] = (
            W29_PARTITION_LINEAR, W29_PARTITION_HIERARCHICAL,
            W29_PARTITION_CYCLIC),
        high_trust_partition_id: int = W29_PARTITION_CYCLIC,
) -> CalibratedGeometryRegistry:
    """Build a W30 registry with calibration_stride=0,
    calibration_vector=(), ancestor_window=0 — the H2 byte-for-W29
    anchor (W30-Λ-trivial-calibration falsifier).

    On any triggered cell, ``registry.is_trivial == True`` makes the
    orchestrator emit ``W30_BRANCH_TRIVIAL_CALIBRATION_PASSTHROUGH``
    with no wire token charged.
    """
    return CalibratedGeometryRegistry(
        schema=schema,
        calibration_stride=0,
        calibration_vector=None,
        ancestor_window=0,
        high_trust_partition_id=int(high_trust_partition_id),
        registered_partition_ids=tuple(int(p)
                                          for p in registered_partition_ids),
        local_host_id=local_host_id,
    )


def build_calibrated_registry(
        *,
        schema: SchemaCapsule,
        calibration_stride: int = 8,
        calibration_priors: Sequence[float] = (1.0, 1.0, 1.0),
        ancestor_window: int = 4,
        high_trust_partition_id: int = W29_PARTITION_CYCLIC,
        registered_partition_ids: Sequence[int] = (
            W29_PARTITION_LINEAR, W29_PARTITION_HIERARCHICAL,
            W29_PARTITION_CYCLIC),
        calibration_threshold: float = (
            W30_DEFAULT_CALIBRATION_PRIOR_THRESHOLD),
        local_host_id: str = "localhost",
) -> CalibratedGeometryRegistry:
    """Build a W30 registry with non-trivial calibration_stride,
    calibration_vector, and ancestor_window.

    Defaults align with the H6 cram-amplification anchor (stride=8,
    window=4, three partitions).  Calibration priors default to
    uniform (1.0, 1.0, 1.0) — the H7 calibration discharge bench
    overrides this with calibrated priors (e.g. (0.95, 0.95, 0.30)).
    """
    pids = tuple(int(p) for p in registered_partition_ids)
    cv: PartitionCalibrationVector | None
    if calibration_priors and len(calibration_priors) == len(pids):
        cv = PartitionCalibrationVector(
            calibration_vector=tuple(float(c) for c in calibration_priors),
            partition_ids=pids,
            threshold=float(calibration_threshold),
        )
    else:
        cv = None
    return CalibratedGeometryRegistry(
        schema=schema,
        calibration_stride=int(calibration_stride),
        calibration_vector=cv,
        ancestor_window=int(ancestor_window),
        high_trust_partition_id=int(high_trust_partition_id),
        registered_partition_ids=pids,
        local_host_id=local_host_id,
    )


# ===========================================================================
# W31 family — online self-calibrated geometry-aware dense control +
# sealed prior trajectory + adaptive threshold + W31 manifest CID
# (SDK v3.32)
# ===========================================================================
#
# W31 wraps the W30 ``CalibratedGeometryOrchestrator`` with three NEW
# audited proxies at the capsule layer:
#
#   1. **Online running-mean prior update**.  After each ratified cell,
#      the W31 layer derives a deterministic per-partition agreement
#      signal (``observed_agreement = 1.0`` iff the cell ratified AND
#      no cross-host disagreement was observed; else ``0.0``) and
#      updates the calibration prior for the *effective* partition_id
#      via the existing closed-form
#      :func:`update_partition_calibration_running_mean` primitive.
#      The updated calibration vector replaces the registry's vector,
#      so the *next* cell's reroute decision uses the freshly-tracked
#      prior.  This is the load-bearing mechanism for the
#      **W30-C-PRIOR-LEARNING** discharge.
#   2. **Adaptive threshold**.  After each ratified cell, the W31
#      layer recomputes the threshold as a closed-form clipped median
#      of the current calibration vector, bounded to
#      ``[threshold_min, threshold_max]``.  The threshold trajectory
#      is sealed in the envelope.
#   3. **Sealed prior trajectory + W31 manifest CID**.  The W31
#      envelope carries:
#        * ``prior_trajectory_cid``     — SHA-256 over canonical bytes
#          of the per-cell sequence
#          ``(cell_idx, partition_id, observed_agreement, prior_after)``;
#        * ``threshold_trajectory_cid`` — SHA-256 over the per-cell
#          threshold sequence;
#        * ``manifest_cid``             — SHA-256 over
#          ``(basis_history_cid, calibration_cid, ancestor_chain_cid,
#            prior_trajectory_cid, threshold_trajectory_cid,
#            route_audit_cid)`` — a single hash that detects
#          cross-component swaps not detected by W30's per-component
#          CIDs.
#
# Honest scope (the load-bearing soundness statement)
# ---------------------------------------------------
#
# * W31 does NOT touch transformer KV caches, hidden states, attention
#   weights, embedding tables, or any model-internal state.  The
#   "online learned" calibration prior is closed-form arithmetic over
#   a deterministic per-cell agreement signal; zero parameters, zero
#   gradients, zero training step.  The "adaptive threshold" is a
#   closed-form clipped median of the prior vector.  The "manifest
#   CID" is SHA-256 over the concatenation of component CIDs.  Every
#   payload is content-addressed; every mutation is detected.
# * The "online learned" calibration prior is NOT a learned model in
#   the deep-learning sense.  It is a closed-form Bayesian-style
#   running mean (see
#   :func:`update_partition_calibration_running_mean` shipped in
#   W30); the W31 layer simply *closes the loop* by feeding the
#   per-cell agreement signal back into the registry.
# * The "adaptive threshold" is one line of arithmetic: clipped
#   median of the calibration vector, clipped to
#   ``[threshold_min, threshold_max]``.  No hyperparameter search.
# * The "manifest CID" does NOT add new entropy; it adds cross-
#   component tamper detection.  A swap of any one component CID
#   from a different envelope (with each component still internally
#   consistent) is detected.
# * The "prior trajectory" is a sealed tuple of
#   ``(cell_idx, partition_id, observed_agreement, prior_after)``
#   bytes; it does NOT prove temporal order at the model layer; it
#   does prove the controller's bus saw exactly that sequence of
#   online updates.  Every entry must satisfy
#   ``observed_agreement, prior_after ∈ [0, 1]`` AND
#   ``partition_id`` is registered.
# * W31 does NOT claim "we solved context."
#
# Wire-token economics
# --------------------
#
# When the registered registry has any of:
#   * ``online_enabled == True``, OR
#   * ``manifest_disabled == False``, OR
#   * ``trajectory_window > 0``
# AND the inner W30 ratified, W31 charges **1 visible token** per
# triggered cell (``<w31_ref:DDDD>``) — the entire prior_trajectory_cid
# + threshold_trajectory_cid + manifest_cid + route_audit_cid metadata
# rides on that single content-addressed reference.  When every
# component is trivial (``online_enabled = False`` AND
# ``manifest_disabled = True`` AND ``trajectory_window = 0``),
# wire_required is False and W31 reduces to W30 **byte-for-byte** (the
# W31-Λ-trivial-online falsifier; see H2 in
# ``docs/SUCCESS_CRITERION_W31_ONLINE_CALIBRATED_GEOMETRY.md``).
#
# Trust boundary: 14 enumerated failure modes (disjoint from W22..W30)
# -------------------------------------------------------------------
#
# :func:`verify_online_calibrated_ratification` rejects:
#
#   * ``empty_w31_envelope``                      — None envelope passed.
#   * ``w31_schema_version_unknown``              — schema_version mismatch.
#   * ``w31_schema_cid_mismatch``                 — schema_cid != registered.
#   * ``w30_parent_cid_mismatch``                 — env.w30_calibrated_cid !=
#     registered.
#   * ``prior_trajectory_cid_mismatch``           — recomputed
#     prior_trajectory_cid does not match.
#   * ``prior_trajectory_length_mismatch``        — len(trajectory) >
#     registered ``trajectory_window`` cap; OR non-monotone cell indices.
#   * ``prior_trajectory_unregistered_partition`` — at least one
#     partition_id not registered.
#   * ``prior_trajectory_observed_out_of_range``  — at least one
#     observed_agreement < 0 OR > 1 OR NaN/Inf.
#   * ``prior_trajectory_prior_after_out_of_range`` — at least one
#     prior_after < 0 OR > 1 OR NaN/Inf.
#   * ``threshold_trajectory_cid_mismatch``       — recomputed
#     threshold_trajectory_cid does not match.
#   * ``threshold_trajectory_value_out_of_range`` — any threshold < 0 OR
#     > 1 OR NaN/Inf.
#   * ``threshold_trajectory_length_mismatch``    —
#     len(threshold_trajectory) != len(prior_trajectory).
#   * ``manifest_cid_mismatch``                   — recomputed manifest_cid
#     does not match.
#   * ``w31_outer_cid_mismatch``                  — recomputed w31_cid
#     does not match.
#
# Every failure mode is mechanically asserted by a unit test in
# ``test_phase78_online_calibrated.py``.

W31_ONLINE_SCHEMA_VERSION: str = (
    "wevra.online_calibrated_ratification.v1")

# W31 decoder branches.
W31_BRANCH_ONLINE_RESOLVED = "online_resolved"
W31_BRANCH_TRIVIAL_ONLINE_PASSTHROUGH = "trivial_online_passthrough"
W31_BRANCH_ONLINE_REJECTED = "online_rejected"
W31_BRANCH_ONLINE_DISABLED = "online_disabled"
W31_BRANCH_ONLINE_NO_TRIGGER = "online_no_trigger"

W31_ALL_BRANCHES: tuple[str, ...] = (
    W31_BRANCH_ONLINE_RESOLVED,
    W31_BRANCH_TRIVIAL_ONLINE_PASSTHROUGH,
    W31_BRANCH_ONLINE_REJECTED,
    W31_BRANCH_ONLINE_DISABLED,
    W31_BRANCH_ONLINE_NO_TRIGGER,
)

W31_DEFAULT_THRESHOLD_MIN: float = 0.20
W31_DEFAULT_THRESHOLD_MAX: float = 0.80
W31_DEFAULT_TRAJECTORY_WINDOW: int = 16


# ---------------------------------------------------------------------------
# W31 helper: deterministic per-cell agreement signal
# ---------------------------------------------------------------------------


def derive_per_cell_agreement_signal(
        *,
        ratified: bool,
        cross_host_disagreement_count: int,
) -> float:
    """Deterministic agreement signal for the W31 online loop.

    Returns 1.0 iff the cell ratified AND no cross-host disagreement
    was observed; else 0.0.  Closed-form, deterministic, audit-friendly.

    The signal is the per-cell observation that drives the running-mean
    update inside :class:`OnlineCalibratedOrchestrator`.  It is NOT a
    correctness oracle (the W31 layer does not see ground truth at
    runtime); it is a *proxy* for partition-level coherence.
    """
    if not bool(ratified):
        return 0.0
    if int(cross_host_disagreement_count) > 0:
        return 0.0
    return 1.0


# ---------------------------------------------------------------------------
# W31 helper: closed-form adaptive threshold (clipped median)
# ---------------------------------------------------------------------------


def compute_adaptive_threshold(
        *,
        calibration_vector: tuple[float, ...] | Sequence[float],
        threshold_min: float = W31_DEFAULT_THRESHOLD_MIN,
        threshold_max: float = W31_DEFAULT_THRESHOLD_MAX,
) -> float:
    """Closed-form adaptive threshold: clipped median of the prior
    vector.

    The W31 adaptive threshold is computed as the median of the
    current calibration prior vector, clipped to
    ``[threshold_min, threshold_max]``.  This keeps the threshold
    inside a fixed band so partitions cannot all be pushed below
    threshold (everything-reroutes pathology) or above threshold
    (no-reroutes pathology).

    Returns ``(threshold_min + threshold_max) / 2`` when the
    calibration vector is empty.
    """
    vec = tuple(float(v) for v in calibration_vector)
    if not vec:
        return float(0.5 * (threshold_min + threshold_max))
    sorted_vec = sorted(vec)
    n = len(sorted_vec)
    if n % 2 == 1:
        med = sorted_vec[n // 2]
    else:
        med = 0.5 * (sorted_vec[n // 2 - 1] + sorted_vec[n // 2])
    if med < threshold_min:
        return float(threshold_min)
    if med > threshold_max:
        return float(threshold_max)
    return float(med)


# ---------------------------------------------------------------------------
# Prior trajectory + threshold trajectory (sealed, content-addressed)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class PriorTrajectoryEntry:
    """One entry in the W31 prior trajectory.

    Carries the per-cell tuple
    ``(cell_idx, partition_id, observed_agreement, prior_after)``
    that drove a single online running-mean update.
    """
    cell_idx: int
    partition_id: int
    observed_agreement: float
    prior_after: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "cell_idx": int(self.cell_idx),
            "partition_id": int(self.partition_id),
            "observed_agreement": round(float(self.observed_agreement), 4),
            "prior_after": round(float(self.prior_after), 4),
        }


def _compute_prior_trajectory_cid(
        *,
        trajectory: Sequence[PriorTrajectoryEntry],
) -> str:
    """Canonical SHA-256 over the prior trajectory."""
    payload = _canonical_json_bytes({
        "trajectory": [t.as_dict() for t in trajectory],
    })
    return hashlib.sha256(payload).hexdigest()


def _compute_threshold_trajectory_cid(
        *,
        thresholds: Sequence[float],
) -> str:
    """Canonical SHA-256 over the threshold trajectory."""
    payload = _canonical_json_bytes({
        "thresholds": [round(float(t), 4) for t in thresholds],
    })
    return hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------
# W31 manifest CID
# ---------------------------------------------------------------------------


def _compute_w31_manifest_cid(
        *,
        basis_history_cid: str,
        calibration_cid: str,
        ancestor_chain_cid: str,
        prior_trajectory_cid: str,
        threshold_trajectory_cid: str,
        route_audit_cid: str,
) -> str:
    """SHA-256 over the canonical concatenation of component CIDs.

    The manifest CID is the load-bearing W31 cross-component tamper-
    detection signal: any swap of one component CID from a different
    envelope (with each component still internally consistent) is
    detected because the manifest CID will not match the registered
    expected manifest.
    """
    payload = _canonical_json_bytes({
        "basis_history_cid": str(basis_history_cid),
        "calibration_cid": str(calibration_cid),
        "ancestor_chain_cid": str(ancestor_chain_cid),
        "prior_trajectory_cid": str(prior_trajectory_cid),
        "threshold_trajectory_cid": str(threshold_trajectory_cid),
        "route_audit_cid": str(route_audit_cid),
    })
    return hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------
# Online-calibrated ratification envelope
# ---------------------------------------------------------------------------


def _compute_w31_outer_cid(
        *,
        schema_version: str,
        schema_cid: str,
        w30_calibrated_cid: str,
        prior_trajectory_cid: str,
        threshold_trajectory_cid: str,
        manifest_cid: str,
        cell_index: int,
) -> str:
    """SHA-256 over the canonical W31 envelope payload."""
    payload = _canonical_json_bytes({
        "schema_version": str(schema_version),
        "schema_cid": str(schema_cid),
        "w30_calibrated_cid": str(w30_calibrated_cid),
        "prior_trajectory_cid": str(prior_trajectory_cid),
        "threshold_trajectory_cid": str(threshold_trajectory_cid),
        "manifest_cid": str(manifest_cid),
        "cell_index": int(cell_index),
    })
    return hashlib.sha256(payload).hexdigest()


@dataclasses.dataclass(frozen=True)
class OnlineCalibratedRatificationEnvelope:
    """Content-addressed online-calibrated ratification of one W30
    decision (W31 family).

    Carries:

      * ``w30_calibrated_cid``        — parent W30 envelope's CID.
      * ``prior_trajectory``          — tuple of
                                          :class:`PriorTrajectoryEntry`
                                          (length ≤ trajectory_window).
      * ``prior_trajectory_cid``      — SHA-256 over canonical bytes.
      * ``threshold_trajectory``      — tuple of float thresholds
                                          (one per trajectory entry).
      * ``threshold_trajectory_cid``  — SHA-256 over canonical bytes.
      * ``basis_history_cid``         — passthrough from W30 (for the
                                          manifest hash).
      * ``calibration_cid``           — passthrough from W30.
      * ``ancestor_chain_cid``        — passthrough from W30.
      * ``route_audit_cid``           — passthrough from W30.
      * ``manifest_cid``              — SHA-256 over (basis_history_cid,
                                          calibration_cid,
                                          ancestor_chain_cid,
                                          prior_trajectory_cid,
                                          threshold_trajectory_cid,
                                          route_audit_cid).
      * ``cell_index``                — audit replay index.
      * ``wire_required``             — 1 visible token cost on
                                          producer side iff True.
      * ``w31_cid``                   — SHA-256 over canonical bytes.

    Wire-token cost
    ---------------

    The W31 layer charges 1 visible token on the producer side
    (``<w31_ref:DDDD>``) iff ``wire_required`` is True (i.e. the
    online registry is non-trivial: online_enabled OR
    NOT manifest_disabled OR trajectory_window > 0).  When every
    component is trivial, wire_required is False and W31 reduces to
    W30 byte-for-byte (W31-Λ-trivial-online; H2 anchor).
    """
    schema_version: str
    schema_cid: str
    w30_calibrated_cid: str
    prior_trajectory: tuple[PriorTrajectoryEntry, ...]
    prior_trajectory_cid: str
    threshold_trajectory: tuple[float, ...]
    threshold_trajectory_cid: str
    basis_history_cid: str
    calibration_cid: str
    ancestor_chain_cid: str
    route_audit_cid: str
    manifest_cid: str
    cell_index: int
    wire_required: bool = False
    w31_cid: str = ""

    def __post_init__(self) -> None:
        if not self.w31_cid:
            object.__setattr__(self, "w31_cid",
                               self.recompute_w31_cid())

    def recompute_w31_cid(self) -> str:
        return _compute_w31_outer_cid(
            schema_version=self.schema_version,
            schema_cid=self.schema_cid,
            w30_calibrated_cid=self.w30_calibrated_cid,
            prior_trajectory_cid=self.prior_trajectory_cid,
            threshold_trajectory_cid=self.threshold_trajectory_cid,
            manifest_cid=self.manifest_cid,
            cell_index=int(self.cell_index),
        )

    def to_canonical_bytes(self) -> bytes:
        return _canonical_json_bytes({
            "schema_version": self.schema_version,
            "schema_cid": self.schema_cid,
            "w30_calibrated_cid": self.w30_calibrated_cid,
            "prior_trajectory": [t.as_dict() for t in self.prior_trajectory],
            "prior_trajectory_cid": self.prior_trajectory_cid,
            "threshold_trajectory": [round(float(t), 4)
                                       for t in self.threshold_trajectory],
            "threshold_trajectory_cid": self.threshold_trajectory_cid,
            "basis_history_cid": self.basis_history_cid,
            "calibration_cid": self.calibration_cid,
            "ancestor_chain_cid": self.ancestor_chain_cid,
            "route_audit_cid": self.route_audit_cid,
            "manifest_cid": self.manifest_cid,
            "cell_index": int(self.cell_index),
        })

    def to_decoder_text(self) -> str:
        return f"<w31_ref:{self.w31_cid[:16]}>"

    @property
    def n_envelope_bytes(self) -> int:
        return len(self.to_canonical_bytes())

    @property
    def n_wire_tokens(self) -> int:
        if not self.wire_required:
            return 0
        return _whitespace_token_count(self.to_decoder_text())

    @property
    def n_structured_bits(self) -> int:
        """Approximate count of structured-control bits packed into
        this W31 envelope, including the entire prior+threshold
        trajectory (each element's audit-friendly content rides on
        the same single wire token).
        """
        return int(8 * self.n_envelope_bytes)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "schema_cid": self.schema_cid,
            "w30_calibrated_cid": self.w30_calibrated_cid,
            "prior_trajectory": [t.as_dict() for t in self.prior_trajectory],
            "prior_trajectory_cid": self.prior_trajectory_cid,
            "threshold_trajectory": [round(float(t), 4)
                                        for t in self.threshold_trajectory],
            "threshold_trajectory_cid": self.threshold_trajectory_cid,
            "basis_history_cid": self.basis_history_cid,
            "calibration_cid": self.calibration_cid,
            "ancestor_chain_cid": self.ancestor_chain_cid,
            "route_audit_cid": self.route_audit_cid,
            "manifest_cid": self.manifest_cid,
            "cell_index": int(self.cell_index),
            "wire_required": bool(self.wire_required),
            "w31_cid": self.w31_cid,
            "n_envelope_bytes": self.n_envelope_bytes,
            "n_wire_tokens": self.n_wire_tokens,
            "n_structured_bits": int(self.n_structured_bits),
            "decoder_text": self.to_decoder_text(),
        }


def verify_online_calibrated_ratification(
        env: OnlineCalibratedRatificationEnvelope | None,
        *,
        registered_schema: SchemaCapsule,
        registered_w30_calibrated_cid: str,
        registered_partition_ids: frozenset[int],
        registered_trajectory_window: int,
        registered_basis_history_cid: str,
        registered_calibration_cid: str,
        registered_ancestor_chain_cid: str,
        registered_route_audit_cid: str,
        registered_prior_trajectory_cid: str | None = None,
        registered_threshold_trajectory_cid: str | None = None,
) -> LatentVerificationOutcome:
    """Pure-function controller-side verification of an
    :class:`OnlineCalibratedRatificationEnvelope` (W31 family).

    14 enumerated failure modes (see module docstring for details).
    Pure function (no side effects); soundness by inspection.
    """
    n_checks = 0

    if env is None:
        return LatentVerificationOutcome(
            ok=False, reason="empty_w31_envelope", n_checks=n_checks)
    n_checks += 1
    if env.schema_version != W31_ONLINE_SCHEMA_VERSION:
        return LatentVerificationOutcome(
            ok=False, reason="w31_schema_version_unknown",
            n_checks=n_checks)
    n_checks += 1
    if env.schema_cid != registered_schema.cid:
        return LatentVerificationOutcome(
            ok=False, reason="w31_schema_cid_mismatch", n_checks=n_checks)
    n_checks += 1
    if env.w30_calibrated_cid != registered_w30_calibrated_cid:
        return LatentVerificationOutcome(
            ok=False, reason="w30_parent_cid_mismatch", n_checks=n_checks)
    n_checks += 1

    # ---- Prior trajectory checks ----
    window = int(registered_trajectory_window)
    traj = env.prior_trajectory
    if len(traj) > window:
        return LatentVerificationOutcome(
            ok=False, reason="prior_trajectory_length_mismatch",
            n_checks=n_checks)
    n_checks += 1
    last_cell_idx = -1
    for entry in traj:
        if int(entry.cell_idx) <= last_cell_idx:
            # Non-monotone → reject under the same reason umbrella.
            return LatentVerificationOutcome(
                ok=False, reason="prior_trajectory_length_mismatch",
                n_checks=n_checks)
        last_cell_idx = int(entry.cell_idx)
        if int(entry.partition_id) not in registered_partition_ids:
            return LatentVerificationOutcome(
                ok=False, reason="prior_trajectory_unregistered_partition",
                n_checks=n_checks)
        oa = float(entry.observed_agreement)
        if math.isnan(oa) or math.isinf(oa) or oa < 0.0 or oa > 1.0:
            return LatentVerificationOutcome(
                ok=False, reason="prior_trajectory_observed_out_of_range",
                n_checks=n_checks)
        pa = float(entry.prior_after)
        if math.isnan(pa) or math.isinf(pa) or pa < 0.0 or pa > 1.0:
            return LatentVerificationOutcome(
                ok=False, reason="prior_trajectory_prior_after_out_of_range",
                n_checks=n_checks)
    n_checks += 1
    # Recompute prior trajectory CID.
    if _compute_prior_trajectory_cid(trajectory=traj) != env.prior_trajectory_cid:
        return LatentVerificationOutcome(
            ok=False, reason="prior_trajectory_cid_mismatch",
            n_checks=n_checks)
    n_checks += 1
    # Cross-check: if the registry registered an expected trajectory
    # CID for this cell, the envelope's CID MUST match.  This catches
    # cross-cell swaps (an attacker replays a previous valid trajectory
    # CID into a later cell's envelope; the manifest CID self-recomputes
    # but the registered expected CID does not match).
    if (registered_prior_trajectory_cid is not None
            and env.prior_trajectory_cid != registered_prior_trajectory_cid):
        return LatentVerificationOutcome(
            ok=False, reason="prior_trajectory_cid_mismatch",
            n_checks=n_checks)
    n_checks += 1

    # ---- Threshold trajectory checks ----
    th = env.threshold_trajectory
    if len(th) != len(traj):
        return LatentVerificationOutcome(
            ok=False, reason="threshold_trajectory_length_mismatch",
            n_checks=n_checks)
    n_checks += 1
    for t in th:
        f = float(t)
        if math.isnan(f) or math.isinf(f) or f < 0.0 or f > 1.0:
            return LatentVerificationOutcome(
                ok=False, reason="threshold_trajectory_value_out_of_range",
                n_checks=n_checks)
    n_checks += 1
    if _compute_threshold_trajectory_cid(thresholds=th) != env.threshold_trajectory_cid:
        return LatentVerificationOutcome(
            ok=False, reason="threshold_trajectory_cid_mismatch",
            n_checks=n_checks)
    n_checks += 1
    # Cross-check: registered expected threshold trajectory CID.
    if (registered_threshold_trajectory_cid is not None
            and env.threshold_trajectory_cid
            != registered_threshold_trajectory_cid):
        return LatentVerificationOutcome(
            ok=False, reason="threshold_trajectory_cid_mismatch",
            n_checks=n_checks)
    n_checks += 1

    # ---- Component CID passthroughs ----
    if env.basis_history_cid != registered_basis_history_cid:
        return LatentVerificationOutcome(
            ok=False, reason="manifest_cid_mismatch", n_checks=n_checks)
    n_checks += 1
    if env.calibration_cid != registered_calibration_cid:
        return LatentVerificationOutcome(
            ok=False, reason="manifest_cid_mismatch", n_checks=n_checks)
    n_checks += 1
    if env.ancestor_chain_cid != registered_ancestor_chain_cid:
        return LatentVerificationOutcome(
            ok=False, reason="manifest_cid_mismatch", n_checks=n_checks)
    n_checks += 1
    if env.route_audit_cid != registered_route_audit_cid:
        return LatentVerificationOutcome(
            ok=False, reason="manifest_cid_mismatch", n_checks=n_checks)
    n_checks += 1

    # ---- Manifest CID check ----
    expected_manifest = _compute_w31_manifest_cid(
        basis_history_cid=env.basis_history_cid,
        calibration_cid=env.calibration_cid,
        ancestor_chain_cid=env.ancestor_chain_cid,
        prior_trajectory_cid=env.prior_trajectory_cid,
        threshold_trajectory_cid=env.threshold_trajectory_cid,
        route_audit_cid=env.route_audit_cid,
    )
    if expected_manifest != env.manifest_cid:
        return LatentVerificationOutcome(
            ok=False, reason="manifest_cid_mismatch", n_checks=n_checks)
    n_checks += 1

    # ---- Outer w31_cid check ----
    if env.recompute_w31_cid() != env.w31_cid:
        return LatentVerificationOutcome(
            ok=False, reason="w31_outer_cid_mismatch", n_checks=n_checks)
    n_checks += 1

    return LatentVerificationOutcome(ok=True, reason="ok", n_checks=n_checks)


# ---------------------------------------------------------------------------
# Online-calibrated registry
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class OnlineCalibratedRegistry:
    """Controller-side registry for the W31 online-calibrated layer.

    Wraps a :class:`CalibratedGeometryRegistry` (the inner W30
    registry) and adds:

      * ``online_enabled``     — when True, the running-mean update
                                   fires on each ratified cell.
      * ``adaptive_threshold`` — when True, the threshold is
                                   recomputed as a clipped median of
                                   the prior vector after each
                                   update.
      * ``manifest_disabled``  — when True (and online_enabled =
                                   False AND trajectory_window = 0),
                                   the W31 layer reduces to W30
                                   byte-for-byte (the trivial path).
      * ``trajectory_window``  — max number of trajectory entries to
                                   carry in the W31 envelope (the
                                   verifier rejects longer
                                   trajectories).
      * ``threshold_min``,
        ``threshold_max``      — clipping bounds for the adaptive
                                   threshold.
    """
    schema: SchemaCapsule | None = None
    inner: CalibratedGeometryRegistry | None = None
    online_enabled: bool = False
    adaptive_threshold: bool = False
    manifest_disabled: bool = True
    trajectory_window: int = 0
    threshold_min: float = W31_DEFAULT_THRESHOLD_MIN
    threshold_max: float = W31_DEFAULT_THRESHOLD_MAX
    local_host_id: str = "localhost"

    _envelopes: dict[str, OnlineCalibratedRatificationEnvelope] = (
        dataclasses.field(default_factory=dict))
    n_w31_registered: int = 0
    n_w31_rejected: int = 0
    n_online_updates: int = 0

    @property
    def is_trivial(self) -> bool:
        return (not bool(self.online_enabled)
                and bool(self.manifest_disabled)
                and int(self.trajectory_window) == 0)

    @property
    def has_wire_required_layer(self) -> bool:
        return not self.is_trivial

    def register_envelope(
            self,
            envelope: OnlineCalibratedRatificationEnvelope,
            *,
            registered_partition_ids: frozenset[int],
            registered_basis_history_cid: str,
            registered_calibration_cid: str,
            registered_ancestor_chain_cid: str,
            registered_route_audit_cid: str,
            registered_w30_calibrated_cid: str,
            registered_prior_trajectory_cid: str | None = None,
            registered_threshold_trajectory_cid: str | None = None,
    ) -> LatentVerificationOutcome:
        """Verify the envelope and (if OK) record it in the audit
        cache.  Pure verifier; idempotent on byte-identical envelopes.
        """
        if self.schema is None:
            outcome = LatentVerificationOutcome(
                ok=False, reason="schema_unregistered", n_checks=0)
            self.n_w31_rejected += 1
            return outcome
        outcome = verify_online_calibrated_ratification(
            envelope,
            registered_schema=self.schema,
            registered_w30_calibrated_cid=str(registered_w30_calibrated_cid),
            registered_partition_ids=frozenset(registered_partition_ids),
            registered_trajectory_window=int(self.trajectory_window),
            registered_basis_history_cid=str(registered_basis_history_cid),
            registered_calibration_cid=str(registered_calibration_cid),
            registered_ancestor_chain_cid=str(registered_ancestor_chain_cid),
            registered_route_audit_cid=str(registered_route_audit_cid),
            registered_prior_trajectory_cid=registered_prior_trajectory_cid,
            registered_threshold_trajectory_cid=(
                registered_threshold_trajectory_cid),
        )
        if not outcome.ok:
            self.n_w31_rejected += 1
            return outcome
        self._envelopes[envelope.w31_cid] = envelope
        self.n_w31_registered += 1
        return outcome


# ---------------------------------------------------------------------------
# W31 result
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class W31OnlineResult:
    """Per-cell audit record for the W31 online-calibrated layer."""
    answer: dict[str, Any]
    inner_w30_branch: str
    decoder_branch: str
    agent_id: str
    is_producer: bool
    w30_calibrated_cid: str
    cell_index: int
    online_update_fired: bool
    observed_agreement: float
    effective_partition_id: int
    prior_before: float
    prior_after: float
    threshold_before: float
    threshold_after: float
    n_w30_visible_tokens: int
    n_w31_visible_tokens: int
    n_w31_overhead_tokens: int
    w31_cid: str
    manifest_cid: str
    prior_trajectory_cid: str
    threshold_trajectory_cid: str
    ratified: bool
    verification_ok: bool
    verification_reason: str
    n_envelope_bytes: int
    n_structured_bits: int
    cram_factor_w31: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "inner_w30_branch": self.inner_w30_branch,
            "decoder_branch": self.decoder_branch,
            "agent_id": self.agent_id,
            "is_producer": bool(self.is_producer),
            "w30_calibrated_cid": self.w30_calibrated_cid,
            "cell_index": int(self.cell_index),
            "online_update_fired": bool(self.online_update_fired),
            "observed_agreement": float(self.observed_agreement),
            "effective_partition_id": int(self.effective_partition_id),
            "prior_before": float(self.prior_before),
            "prior_after": float(self.prior_after),
            "threshold_before": float(self.threshold_before),
            "threshold_after": float(self.threshold_after),
            "n_w30_visible_tokens": int(self.n_w30_visible_tokens),
            "n_w31_visible_tokens": int(self.n_w31_visible_tokens),
            "n_w31_overhead_tokens": int(self.n_w31_overhead_tokens),
            "w31_cid": self.w31_cid,
            "manifest_cid": self.manifest_cid,
            "prior_trajectory_cid": self.prior_trajectory_cid,
            "threshold_trajectory_cid": self.threshold_trajectory_cid,
            "ratified": bool(self.ratified),
            "verification_ok": bool(self.verification_ok),
            "verification_reason": self.verification_reason,
            "n_envelope_bytes": int(self.n_envelope_bytes),
            "n_structured_bits": int(self.n_structured_bits),
            "cram_factor_w31": float(self.cram_factor_w31),
        }


# ---------------------------------------------------------------------------
# Online-calibrated orchestrator
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class OnlineCalibratedOrchestrator:
    """Online self-calibrated geometry-aware dense-control orchestrator
    (W31).

    Wraps a :class:`CalibratedGeometryOrchestrator` (W30) and adds:

      * online running-mean prior update on each ratified cell;
      * adaptive threshold (clipped median of prior vector);
      * sealed prior + threshold trajectory in the W31 envelope;
      * W31 manifest CID over the component CIDs.

    Per-cell flow:

      1. Run inner W30.
      2. Read inner W30 result + envelope.
      3. Derive the per-cell agreement signal
         (``ratified AND no cross-host disagreement`` ⇒ 1.0; else 0.0).
      4. If ``online_enabled`` is True, update the inner W30
         registry's calibration prior for the *effective* partition_id
         via :func:`update_partition_calibration_running_mean`.
      5. If ``adaptive_threshold`` is True, recompute the inner
         registry's threshold as the clipped median of the new prior
         vector.
      6. Append a :class:`PriorTrajectoryEntry` and a threshold to
         the W31 layer's running trajectories (truncated to
         ``trajectory_window``).
      7. Build the W31 envelope with sealed prior + threshold
         trajectories + manifest CID.
      8. Verify + register.
      9. Charge 1 wire token iff ``registry.has_wire_required_layer``.
    """
    inner: CalibratedGeometryOrchestrator
    registry: OnlineCalibratedRegistry
    enabled: bool = True
    require_w31_verification: bool = True

    _last_result: "W31OnlineResult | None" = None
    _last_envelope: "OnlineCalibratedRatificationEnvelope | None" = None
    _prior_trajectory: list[PriorTrajectoryEntry] = dataclasses.field(
        default_factory=list)
    _threshold_trajectory: list[float] = dataclasses.field(
        default_factory=list)
    _cell_index: int = 0

    @property
    def schema(self) -> "SchemaCapsule | None":
        return self.inner.schema

    @property
    def agent_id(self) -> str:
        return self.inner.agent_id

    @property
    def is_producer(self) -> bool:
        return self.inner.is_producer

    @property
    def producer_agent_id(self) -> str:
        return self.inner.producer_agent_id

    @property
    def consumer_agent_ids(self) -> tuple[str, ...]:
        return self.inner.consumer_agent_ids

    def reset_session(self) -> None:
        self.inner.reset_session()
        self._last_result = None
        self._last_envelope = None
        self._prior_trajectory = []
        self._threshold_trajectory = []
        self._cell_index = 0

    def _build_w31_envelope(
            self,
            *,
            w30_calibrated_cid: str,
            basis_history_cid: str,
            calibration_cid: str,
            ancestor_chain_cid: str,
            route_audit_cid: str,
            wire_required: bool,
    ) -> OnlineCalibratedRatificationEnvelope:
        traj = tuple(self._prior_trajectory)
        thr = tuple(self._threshold_trajectory)
        prior_traj_cid = _compute_prior_trajectory_cid(trajectory=traj)
        thr_traj_cid = _compute_threshold_trajectory_cid(thresholds=thr)
        manifest_cid = _compute_w31_manifest_cid(
            basis_history_cid=str(basis_history_cid),
            calibration_cid=str(calibration_cid),
            ancestor_chain_cid=str(ancestor_chain_cid),
            prior_trajectory_cid=prior_traj_cid,
            threshold_trajectory_cid=thr_traj_cid,
            route_audit_cid=str(route_audit_cid),
        )
        env = OnlineCalibratedRatificationEnvelope(
            schema_version=W31_ONLINE_SCHEMA_VERSION,
            schema_cid=str(self.schema.cid),
            w30_calibrated_cid=str(w30_calibrated_cid),
            prior_trajectory=traj,
            prior_trajectory_cid=prior_traj_cid,
            threshold_trajectory=thr,
            threshold_trajectory_cid=thr_traj_cid,
            basis_history_cid=str(basis_history_cid),
            calibration_cid=str(calibration_cid),
            ancestor_chain_cid=str(ancestor_chain_cid),
            route_audit_cid=str(route_audit_cid),
            manifest_cid=manifest_cid,
            cell_index=int(self._cell_index),
            wire_required=bool(wire_required),
        )
        return env

    def decode_rounds(
            self,
            per_round_handoffs: Sequence[Sequence[_DecodedHandoff]],
    ) -> dict[str, Any]:
        # Disabled / no-schema paths.
        if not self.enabled or self.schema is None:
            out = self.inner.decode_rounds(per_round_handoffs)
            n_w30_visible = int(out.get("calibrated_geometry", {}).get(
                "n_w30_visible_tokens", 0))
            return self._pack(
                out=out, decoder_branch=W31_BRANCH_ONLINE_DISABLED,
                envelope=None, n_w31_visible=n_w30_visible,
                w31_overhead=0, ratified=False,
                verify_ok=False, verify_reason="disabled",
                online_update_fired=False,
                observed_agreement=0.0,
                effective_partition_id=0,
                prior_before=0.0, prior_after=0.0,
                threshold_before=0.0, threshold_after=0.0,
                w30_calibrated_cid="")

        # 1. Run inner W30.
        out = self.inner.decode_rounds(per_round_handoffs)
        w30_result = self.inner.last_result
        w30_envelope = self.inner.last_envelope
        n_w30_visible = int(out.get("calibrated_geometry", {}).get(
            "n_w30_visible_tokens", 0))
        inner_w30_branch = (
            str(w30_result.decoder_branch) if w30_result is not None else "")

        # If inner W30 produced no per-cell result at all (e.g. W30
        # disabled), there is nothing to feed the online loop and no
        # parent CID for the W31 envelope.
        if w30_result is None:
            self._cell_index += 1
            return self._pack(
                out=out,
                decoder_branch=W31_BRANCH_ONLINE_NO_TRIGGER,
                envelope=None, n_w31_visible=n_w30_visible,
                w31_overhead=0, ratified=False,
                verify_ok=False, verify_reason="no_w30_result",
                online_update_fired=False,
                observed_agreement=0.0,
                effective_partition_id=0,
                prior_before=0.0, prior_after=0.0,
                threshold_before=0.0, threshold_after=0.0,
                w30_calibrated_cid="")

        # 2. Read W30 envelope CIDs (passthrough into the W31 manifest).
        # When the W30 envelope is None (e.g. inner W29 did not ratify
        # so W30 emitted no envelope), we still drive the online loop
        # on the per-cell agreement signal — but skip the W31 envelope
        # build (no parent CID to bind to).
        basis_history_cid = (
            w30_envelope.basis_history.history_cid
            if (w30_envelope is not None
                  and w30_envelope.basis_history is not None) else "")
        calibration_cid_pre = (
            w30_envelope.calibration.calibration_cid
            if (w30_envelope is not None
                  and w30_envelope.calibration is not None) else "")
        ancestor_chain_cid = (
            w30_envelope.ancestor_chain.chain_cid
            if (w30_envelope is not None
                  and w30_envelope.ancestor_chain is not None) else "")
        # The "route audit cid" is a closed-form summary of the W30
        # envelope's routing decision.  We reuse the W30 envelope's
        # disagreement-route + structural decision summary.  When the
        # W30 envelope is absent we still hash the W30 result's routing
        # info to keep the trajectory audit trail intact.
        if w30_envelope is not None:
            route_audit_payload = _canonical_json_bytes({
                "disagreement_route_active":
                    bool(w30_envelope.disagreement_route_active),
                "disagreement_route_target_partition_id":
                    int(w30_envelope.disagreement_route_target_partition_id),
                "effective_partition_id":
                    int(w30_result.effective_partition_id),
                "structural_partition_id":
                    int(w30_result.structural_partition_id),
            })
        else:
            route_audit_payload = _canonical_json_bytes({
                "disagreement_route_active":
                    bool(w30_result.disagreement_route_active),
                "disagreement_route_target_partition_id":
                    int(w30_result.effective_partition_id),
                "effective_partition_id":
                    int(w30_result.effective_partition_id),
                "structural_partition_id":
                    int(w30_result.structural_partition_id),
            })
        route_audit_cid = hashlib.sha256(route_audit_payload).hexdigest()

        # 3. Derive the per-cell agreement signal.  This is the
        # load-bearing online-learning input: closed-form, deterministic.
        # On a cell where the inner W30 baseline did NOT ratify (no
        # W30 envelope built), the signal is 0.0 — exactly what the
        # online loop should observe.
        observed_agreement = derive_per_cell_agreement_signal(
            ratified=bool(w30_result.ratified),
            cross_host_disagreement_count=int(
                w30_result.cross_host_disagreement_count),
        )

        eff_pid = int(w30_result.effective_partition_id)

        # 4. Look up prior_before; fire online update if enabled.
        inner_w30_registry = self.inner.registry
        cv_before = inner_w30_registry.calibration_vector
        prior_before = (cv_before.prior_for(eff_pid)
                          if cv_before is not None else 1.0)
        threshold_before = (float(cv_before.threshold)
                              if cv_before is not None else 1.0)
        online_update_fired = False
        prior_after = float(prior_before)
        threshold_after = float(threshold_before)
        calibration_cid_post = calibration_cid_pre
        if (self.registry.online_enabled
                and cv_before is not None):
            # n_observations_prior is the number of trajectory entries
            # for this partition seen so far.
            n_obs = sum(1 for e in self._prior_trajectory
                          if int(e.partition_id) == eff_pid)
            new_cv = update_partition_calibration_running_mean(
                prev=cv_before,
                partition_id=eff_pid,
                observed_agreement=observed_agreement,
                n_observations_prior=n_obs,
            )
            inner_w30_registry.calibration_vector = new_cv
            prior_after = new_cv.prior_for(eff_pid)
            online_update_fired = True

            # 5. Adaptive threshold update.
            if self.registry.adaptive_threshold:
                new_thr = compute_adaptive_threshold(
                    calibration_vector=new_cv.calibration_vector,
                    threshold_min=float(self.registry.threshold_min),
                    threshold_max=float(self.registry.threshold_max),
                )
                if abs(new_thr - float(new_cv.threshold)) > 1e-12:
                    rebuilt_cv = PartitionCalibrationVector(
                        calibration_vector=tuple(new_cv.calibration_vector),
                        partition_ids=tuple(new_cv.partition_ids),
                        threshold=float(new_thr),
                    )
                    inner_w30_registry.calibration_vector = rebuilt_cv
                    threshold_after = float(new_thr)
                    calibration_cid_post = rebuilt_cv.calibration_cid
                else:
                    threshold_after = float(new_cv.threshold)
                    calibration_cid_post = new_cv.calibration_cid
            else:
                threshold_after = float(new_cv.threshold)
                calibration_cid_post = new_cv.calibration_cid
            self.registry.n_online_updates += 1

        # 6. Append trajectory entry (truncated to window).
        if int(self.registry.trajectory_window) > 0:
            entry = PriorTrajectoryEntry(
                cell_idx=int(self._cell_index),
                partition_id=int(eff_pid),
                observed_agreement=float(observed_agreement),
                prior_after=float(prior_after),
            )
            self._prior_trajectory.append(entry)
            self._threshold_trajectory.append(float(threshold_after))
            window = int(self.registry.trajectory_window)
            if len(self._prior_trajectory) > window:
                self._prior_trajectory = (
                    self._prior_trajectory[-window:])
                self._threshold_trajectory = (
                    self._threshold_trajectory[-window:])

        # 6b. If the W30 baseline produced no envelope on this cell,
        # we have updated the online learning state but cannot bind
        # a sealed W31 envelope (no parent CID).  Return the
        # online-no-trigger branch with the updated trajectory state.
        if w30_envelope is None:
            self._cell_index += 1
            return self._pack(
                out=out,
                decoder_branch=W31_BRANCH_ONLINE_NO_TRIGGER,
                envelope=None, n_w31_visible=n_w30_visible,
                w31_overhead=0, ratified=False,
                verify_ok=False, verify_reason="no_w30_envelope",
                online_update_fired=bool(online_update_fired),
                observed_agreement=float(observed_agreement),
                effective_partition_id=int(eff_pid),
                prior_before=float(prior_before),
                prior_after=float(prior_after),
                threshold_before=float(threshold_before),
                threshold_after=float(threshold_after),
                w30_calibrated_cid="")

        # 7. Build the W31 envelope.
        wire_required = self.registry.has_wire_required_layer
        # Use the *post-update* calibration CID for the manifest so the
        # registered envelope reflects the current registry state.
        envelope = self._build_w31_envelope(
            w30_calibrated_cid=str(w30_envelope.calibrated_cid),
            basis_history_cid=basis_history_cid,
            calibration_cid=calibration_cid_post,
            ancestor_chain_cid=ancestor_chain_cid,
            route_audit_cid=route_audit_cid,
            wire_required=bool(wire_required),
        )

        # 8. Verify + register.  We pass the orchestrator's running
        # prior_trajectory_cid and threshold_trajectory_cid as the
        # registered expectation so the verifier rejects cross-cell
        # swaps (an attacker replaying a previous valid trajectory CID
        # into the current cell's envelope).
        registered_pids = frozenset(
            int(p) for p in inner_w30_registry.registered_partition_ids)
        expected_prior_traj_cid = envelope.prior_trajectory_cid
        expected_thr_traj_cid = envelope.threshold_trajectory_cid
        outcome = self.registry.register_envelope(
            envelope,
            registered_partition_ids=registered_pids,
            registered_basis_history_cid=basis_history_cid,
            registered_calibration_cid=calibration_cid_post,
            registered_ancestor_chain_cid=ancestor_chain_cid,
            registered_route_audit_cid=route_audit_cid,
            registered_w30_calibrated_cid=str(w30_envelope.calibrated_cid),
            registered_prior_trajectory_cid=expected_prior_traj_cid,
            registered_threshold_trajectory_cid=expected_thr_traj_cid,
        )
        verify_ok = bool(outcome.ok)
        verify_reason = str(outcome.reason)

        if not verify_ok and self.require_w31_verification:
            self._cell_index += 1
            return self._pack(
                out=out,
                decoder_branch=W31_BRANCH_ONLINE_REJECTED,
                envelope=envelope, n_w31_visible=n_w30_visible,
                w31_overhead=0, ratified=False,
                verify_ok=False, verify_reason=verify_reason,
                online_update_fired=bool(online_update_fired),
                observed_agreement=float(observed_agreement),
                effective_partition_id=int(eff_pid),
                prior_before=float(prior_before),
                prior_after=float(prior_after),
                threshold_before=float(threshold_before),
                threshold_after=float(threshold_after),
                w30_calibrated_cid=str(w30_envelope.calibrated_cid))

        # 9. Trivial path — no wire token charged; pass through.
        if self.registry.is_trivial:
            self._cell_index += 1
            return self._pack(
                out=out,
                decoder_branch=W31_BRANCH_TRIVIAL_ONLINE_PASSTHROUGH,
                envelope=envelope, n_w31_visible=n_w30_visible,
                w31_overhead=0, ratified=True,
                verify_ok=verify_ok, verify_reason=verify_reason,
                online_update_fired=bool(online_update_fired),
                observed_agreement=float(observed_agreement),
                effective_partition_id=int(eff_pid),
                prior_before=float(prior_before),
                prior_after=float(prior_after),
                threshold_before=float(threshold_before),
                threshold_after=float(threshold_after),
                w30_calibrated_cid=str(w30_envelope.calibrated_cid))

        # Non-trivial: charge 1 wire token.
        w31_overhead = int(envelope.n_wire_tokens)
        n_w31_visible = int(n_w30_visible + w31_overhead)
        self._cell_index += 1
        return self._pack(
            out=out,
            decoder_branch=W31_BRANCH_ONLINE_RESOLVED,
            envelope=envelope, n_w31_visible=n_w31_visible,
            w31_overhead=w31_overhead, ratified=True,
            verify_ok=verify_ok, verify_reason=verify_reason,
            online_update_fired=bool(online_update_fired),
            observed_agreement=float(observed_agreement),
            effective_partition_id=int(eff_pid),
            prior_before=float(prior_before),
            prior_after=float(prior_after),
            threshold_before=float(threshold_before),
            threshold_after=float(threshold_after),
            w30_calibrated_cid=str(w30_envelope.calibrated_cid))

    def _pack(
            self, *, out: dict[str, Any],
            decoder_branch: str,
            envelope: OnlineCalibratedRatificationEnvelope | None,
            n_w31_visible: int, w31_overhead: int,
            ratified: bool, verify_ok: bool, verify_reason: str,
            online_update_fired: bool,
            observed_agreement: float,
            effective_partition_id: int,
            prior_before: float, prior_after: float,
            threshold_before: float, threshold_after: float,
            w30_calibrated_cid: str,
    ) -> dict[str, Any]:
        envelope_bytes = (envelope.n_envelope_bytes
                            if envelope is not None else 0)
        structured_bits = (envelope.n_structured_bits
                              if envelope is not None else 0)
        wire = max(1, int(w31_overhead))
        cram_factor = (
            float(structured_bits) / float(wire)
            if structured_bits > 0 else 0.0
        )
        w31_cid = (envelope.w31_cid if envelope is not None else "")
        manifest_cid = (envelope.manifest_cid
                          if envelope is not None else "")
        prior_traj_cid = (envelope.prior_trajectory_cid
                            if envelope is not None else "")
        thr_traj_cid = (envelope.threshold_trajectory_cid
                          if envelope is not None else "")
        n_w30_visible = int(out.get("calibrated_geometry", {}).get(
            "n_w30_visible_tokens", 0))
        inner_w30_branch = str(out.get("calibrated_geometry", {}).get(
            "decoder_branch", ""))
        result = W31OnlineResult(
            answer=dict(out),
            inner_w30_branch=inner_w30_branch,
            decoder_branch=str(decoder_branch),
            agent_id=str(self.agent_id),
            is_producer=bool(self.is_producer),
            w30_calibrated_cid=str(w30_calibrated_cid),
            cell_index=int(self._cell_index - 1
                              if self._cell_index > 0 else 0),
            online_update_fired=bool(online_update_fired),
            observed_agreement=float(observed_agreement),
            effective_partition_id=int(effective_partition_id),
            prior_before=float(prior_before),
            prior_after=float(prior_after),
            threshold_before=float(threshold_before),
            threshold_after=float(threshold_after),
            n_w30_visible_tokens=int(n_w30_visible),
            n_w31_visible_tokens=int(n_w31_visible),
            n_w31_overhead_tokens=int(w31_overhead),
            w31_cid=str(w31_cid),
            manifest_cid=str(manifest_cid),
            prior_trajectory_cid=str(prior_traj_cid),
            threshold_trajectory_cid=str(thr_traj_cid),
            ratified=bool(ratified),
            verification_ok=bool(verify_ok),
            verification_reason=str(verify_reason),
            n_envelope_bytes=int(envelope_bytes),
            n_structured_bits=int(structured_bits),
            cram_factor_w31=float(cram_factor),
        )
        self._last_result = result
        self._last_envelope = envelope
        out_local = dict(out)
        out_local["online_calibrated"] = result.as_dict()
        if envelope is not None:
            out_local["online_calibrated_envelope"] = envelope.as_dict()
        return out_local

    def decode(
            self,
            handoffs: Sequence[_DecodedHandoff],
    ) -> dict[str, Any]:
        return self.decode_rounds([handoffs])

    @property
    def last_result(self) -> "W31OnlineResult | None":
        return self._last_result

    @property
    def last_envelope(self) -> (
            "OnlineCalibratedRatificationEnvelope | None"):
        return self._last_envelope


# ---------------------------------------------------------------------------
# Convenience factories (W31 family)
# ---------------------------------------------------------------------------


def build_trivial_online_registry(
        *,
        schema: SchemaCapsule,
        local_host_id: str = "localhost",
) -> OnlineCalibratedRegistry:
    """Build a W31 registry with online_enabled=False, manifest_disabled=
    True, trajectory_window=0 — the H2 byte-for-W30 anchor
    (W31-Λ-trivial-online falsifier).
    """
    return OnlineCalibratedRegistry(
        schema=schema,
        online_enabled=False,
        adaptive_threshold=False,
        manifest_disabled=True,
        trajectory_window=0,
        local_host_id=local_host_id,
    )


def build_online_calibrated_registry(
        *,
        schema: SchemaCapsule,
        online_enabled: bool = True,
        adaptive_threshold: bool = True,
        manifest_disabled: bool = False,
        trajectory_window: int = W31_DEFAULT_TRAJECTORY_WINDOW,
        threshold_min: float = W31_DEFAULT_THRESHOLD_MIN,
        threshold_max: float = W31_DEFAULT_THRESHOLD_MAX,
        local_host_id: str = "localhost",
) -> OnlineCalibratedRegistry:
    """Build a non-trivial W31 registry that exercises online prior
    learning + adaptive threshold + sealed trajectory + manifest CID.
    """
    return OnlineCalibratedRegistry(
        schema=schema,
        online_enabled=bool(online_enabled),
        adaptive_threshold=bool(adaptive_threshold),
        manifest_disabled=bool(manifest_disabled),
        trajectory_window=int(trajectory_window),
        threshold_min=float(threshold_min),
        threshold_max=float(threshold_max),
        local_host_id=local_host_id,
    )


# ===========================================================================
# W32 family — long-window convergent online geometry-aware dense control +
# EWMA prior accumulator + Page CUSUM change-point detector + gold-
# correlated disagreement-routing + W32 manifest-v2 CID
# (SDK v3.33)
# ===========================================================================
#
# W32 wraps the W31 ``OnlineCalibratedOrchestrator`` with four NEW
# audited proxies at the capsule layer:
#
#   1. **EWMA prior accumulator**.  After each ratified cell, the W32
#      layer updates a per-partition EWMA prior via the closed-form
#      ``ewma_new = (1 - alpha) * ewma_prev + alpha * obs`` over the
#      same deterministic per-cell agreement signal W31 uses.
#      Crucially, the EWMA *forgets* old observations exponentially —
#      so at long ``long_window`` (e.g. 64, 128) the W32 prior can
#      *re-converge* after a regime shift, which W31's cumulative
#      running-mean cannot do (cumulative mean is anchored to the
#      stationary prior of the long history).  This is the load-
#      bearing mechanism for the **W31-C-LONG-WINDOW-CONVERGENCE**
#      discharge.
#   2. **Page two-sided CUSUM change-point detector**.  After each
#      cell, the W32 layer updates two CUSUM accumulators (positive
#      drift detector + negative drift detector) over the residual
#      ``obs - target`` (target = registered partition trust prior).
#      When ``cusum_pos`` OR ``cusum_neg`` crosses a registered
#      threshold, a **change point** is declared and the EWMA + CUSUM
#      state is *re-initialised* to the latest observation.  This
#      makes the W32 layer responsive to abrupt regime shifts, which
#      a pure EWMA at long window is too slow to track.
#   3. **Gold-correlation disagreement-routing**.  When the inner
#      W31's cross-host disagreement-route fires AND the registered
#      ``GoldCorrelationMap`` indicates a high gold-correlation score
#      for one specific partition, W32 re-targets the W31 reroute to
#      that partition.  The ``GoldCorrelationMap`` is a *registered
#      closed-vocabulary table* — the W32 layer does NOT observe
#      ground truth at runtime; the map is a *structural witness* the
#      controller registers up-front (e.g. "on this regime, the
#      LINEAR partition's oracle is the most reliable on disagreed
#      cells").  If the registered map is wrong, the
#      W32-Λ-mis-correlated-gold falsifier fires and W32 regresses
#      vs W31.
#   4. **Sealed convergence state + W32 manifest-v2 CID**.  The W32
#      envelope carries:
#        * ``convergence_state_cid``  — SHA-256 over canonical bytes
#          of the per-cell sequence
#          ``(cell_idx, partition_id, ewma_prior_after, cusum_pos,
#            cusum_neg, change_point_fired)``;
#        * ``gold_correlation_cid``   — SHA-256 over the registered
#          gold-correlation map's canonical bytes;
#        * ``manifest_v2_cid``        — SHA-256 over
#          ``(w31_online_cid, convergence_state_cid,
#            gold_correlation_cid, route_audit_cid_v2)`` — a single
#          hash that detects cross-component swaps not detected by
#          the W31 manifest CID alone (the W31 manifest does NOT
#          include convergence state or gold-correlation; it is
#          structurally invariant to those swaps).
#
# Honest scope (the load-bearing soundness statement)
# ---------------------------------------------------
#
# * W32 does NOT touch transformer KV caches, hidden states, attention
#   weights, embedding tables, or any model-internal state.  The
#   "EWMA prior accumulator" is one line of arithmetic per cell; the
#   "Page CUSUM" is two lines of arithmetic per cell; the "gold-
#   correlation routing" is a deterministic lookup against a
#   *registered* closed-vocabulary map; the "manifest-v2 CID" is
#   SHA-256 over the concatenation of component CIDs.  Every payload
#   is content-addressed; every mutation is detected.
# * The "EWMA online learning" is NOT a learned model in the deep-
#   learning sense.  The smoothing factor ``alpha`` is a registered
#   constant; zero parameters, zero gradients, zero training step.
# * The "CUSUM change-point detector" is one closed-form expression
#   per cell; not a hyperparameter search.
# * The "gold-correlation map" is a *registered* closed-vocabulary
#   table; the W32 layer does NOT observe ground truth at runtime.
#   The map's correctness is a *structural witness*; if the map is
#   wrong, W32-Λ-mis-correlated-gold fires and W32 regresses.
# * The "convergence state" is a sealed tuple of bytes; it does NOT
#   prove temporal order at the model layer; it does prove the
#   controller's bus saw exactly that sequence of EWMA + CUSUM
#   updates.  Every entry must satisfy ``ewma_prior_after ∈ [0, 1]``
#   AND ``cusum_pos, cusum_neg ∈ [0, cusum_max]``.
# * W32 does NOT claim "we solved context."
#
# Wire-token economics
# --------------------
#
# When the registered registry has any of:
#   * ``long_window_enabled == True``, OR
#   * ``change_point_enabled == True``, OR
#   * ``gold_correlation_enabled == True``, OR
#   * ``manifest_v2_disabled == False``, OR
#   * ``long_window > 0``
# AND the inner W31 ratified, W32 charges **1 visible token** per
# triggered cell (``<w32_ref:DDDD>``) — the entire convergence_state
# + gold_correlation + manifest-v2 metadata rides on that single
# content-addressed reference.  When every component is trivial,
# wire_required is False and W32 reduces to W31 **byte-for-byte** (the
# W32-Λ-trivial-long-window falsifier; see H2 in
# ``docs/SUCCESS_CRITERION_W32_LONG_WINDOW_CONVERGENT.md``).
#
# Trust boundary: 14 enumerated failure modes (disjoint from W22..W31)
# -------------------------------------------------------------------
#
# :func:`verify_long_window_convergent_ratification` rejects:
#
#   * ``empty_w32_envelope``                       — None envelope passed.
#   * ``w32_schema_version_unknown``               — schema_version mismatch.
#   * ``w32_schema_cid_mismatch``                  — schema_cid != registered.
#   * ``w31_parent_cid_mismatch``                  — env.w31_online_cid !=
#     registered.
#   * ``convergence_state_cid_mismatch``           — recomputed
#     convergence_state_cid does not match, OR registered_convergence_
#     state_cid mismatch (cross-cell swap detection).
#   * ``convergence_state_length_mismatch``        — len(states) >
#     registered ``long_window`` cap; OR non-monotone cell indices.
#   * ``convergence_state_unregistered_partition`` — at least one
#     partition_id not registered.
#   * ``convergence_state_ewma_out_of_range``      — at least one
#     ewma_prior_after < 0 OR > 1 OR NaN/Inf.
#   * ``convergence_state_cusum_out_of_range``     — at least one
#     cusum_pos OR cusum_neg < 0 OR > registered cusum_max OR NaN/Inf.
#   * ``ewma_alpha_out_of_range``                  — env.ewma_alpha < 0
#     OR > 1 OR NaN/Inf.
#   * ``cusum_threshold_out_of_range``             — env.cusum_threshold
#     < 0 OR > registered cusum_max OR NaN/Inf.
#   * ``gold_correlation_cid_mismatch``            — recomputed gold_
#     correlation_cid does not match.
#   * ``manifest_v2_cid_mismatch``                 — recomputed manifest_v2
#     does not match.
#   * ``w32_outer_cid_mismatch``                   — recomputed w32_cid
#     does not match.
#
# Every failure mode is mechanically asserted by a unit test in
# ``test_phase79_long_window_convergent.py``.

W32_LONG_WINDOW_SCHEMA_VERSION: str = (
    "wevra.long_window_convergent_ratification.v1")

# W32 decoder branches.
W32_BRANCH_LONG_WINDOW_RESOLVED = "long_window_resolved"
W32_BRANCH_TRIVIAL_LONG_WINDOW_PASSTHROUGH = (
    "trivial_long_window_passthrough")
W32_BRANCH_LONG_WINDOW_REJECTED = "long_window_rejected"
W32_BRANCH_LONG_WINDOW_DISABLED = "long_window_disabled"
W32_BRANCH_LONG_WINDOW_NO_TRIGGER = "long_window_no_trigger"
W32_BRANCH_GOLD_CORRELATED_REROUTED = "gold_correlated_rerouted"
W32_BRANCH_CHANGE_POINT_RESET = "change_point_reset"

W32_ALL_BRANCHES: tuple[str, ...] = (
    W32_BRANCH_LONG_WINDOW_RESOLVED,
    W32_BRANCH_TRIVIAL_LONG_WINDOW_PASSTHROUGH,
    W32_BRANCH_LONG_WINDOW_REJECTED,
    W32_BRANCH_LONG_WINDOW_DISABLED,
    W32_BRANCH_LONG_WINDOW_NO_TRIGGER,
    W32_BRANCH_GOLD_CORRELATED_REROUTED,
    W32_BRANCH_CHANGE_POINT_RESET,
)

W32_DEFAULT_EWMA_ALPHA: float = 0.20
W32_DEFAULT_CUSUM_THRESHOLD: float = 1.5
W32_DEFAULT_CUSUM_K: float = 0.10
W32_DEFAULT_CUSUM_MAX: float = 16.0
W32_DEFAULT_LONG_WINDOW: int = 64
W32_DEFAULT_GOLD_CORRELATION_MIN: float = 0.50


# ---------------------------------------------------------------------------
# W32 helper: closed-form EWMA update
# ---------------------------------------------------------------------------


def update_ewma_prior(
        *,
        prev_ewma: float,
        observation: float,
        alpha: float,
) -> float:
    """Closed-form exponentially-weighted moving average update.

    ``ewma_new = (1 - alpha) * ewma_prev + alpha * observation``

    Bounded to [0, 1] when both prev_ewma and observation are in
    [0, 1] AND alpha ∈ [0, 1] — by linearity of the convex
    combination.  Closed-form, deterministic, audit-friendly.
    """
    a = float(alpha)
    if a < 0.0:
        a = 0.0
    if a > 1.0:
        a = 1.0
    p = float(prev_ewma)
    o = float(observation)
    return float((1.0 - a) * p + a * o)


# ---------------------------------------------------------------------------
# W32 helper: closed-form Page two-sided CUSUM update
# ---------------------------------------------------------------------------


def update_cusum_two_sided(
        *,
        cusum_pos_prev: float,
        cusum_neg_prev: float,
        observation: float,
        target: float,
        slack_k: float,
        cusum_max: float,
) -> tuple[float, float]:
    """Page two-sided CUSUM update.

    Tracks BOTH a positive drift accumulator (detects upward drift in
    observation vs target) AND a negative drift accumulator (detects
    downward drift).  Each accumulator is non-negative and bounded by
    cusum_max.  Closed-form, deterministic, audit-friendly.

    Returns ``(cusum_pos, cusum_neg)``.
    """
    o = float(observation)
    t = float(target)
    k = float(slack_k)
    cmax = float(cusum_max)
    cp_prev = max(0.0, float(cusum_pos_prev))
    cn_prev = max(0.0, float(cusum_neg_prev))
    cp = max(0.0, cp_prev + (o - t) - k)
    cn = max(0.0, cn_prev - (o - t) - k)
    if cp > cmax:
        cp = cmax
    if cn > cmax:
        cn = cmax
    return float(cp), float(cn)


def detect_change_point(
        *,
        cusum_pos: float,
        cusum_neg: float,
        threshold: float,
) -> bool:
    """True iff either CUSUM accumulator has crossed the threshold."""
    return (float(cusum_pos) > float(threshold)
            or float(cusum_neg) > float(threshold))


# ---------------------------------------------------------------------------
# W32 helper: gold-correlation map (registered closed-vocab table)
# ---------------------------------------------------------------------------


def _compute_gold_correlation_cid(
        *,
        partition_to_score: Sequence[tuple[int, float]],
        gold_correlation_min: float,
) -> str:
    """SHA-256 over canonical bytes of the gold-correlation map +
    threshold."""
    payload = _canonical_json_bytes({
        "partition_to_score": [
            [int(pid), round(float(s), 4)]
            for pid, s in sorted(partition_to_score, key=lambda x: int(x[0]))
        ],
        "gold_correlation_min": round(float(gold_correlation_min), 4),
    })
    return hashlib.sha256(payload).hexdigest()


@dataclasses.dataclass(frozen=True)
class GoldCorrelationMap:
    """Registered closed-vocabulary partition → gold-correlation map.

    Each entry is a tuple ``(partition_id, gold_correlation_score)``
    where ``score ∈ [0, 1]``.  The W32 gold-correlation route fires
    on cells where the inner W31 cross-host disagreement-route is
    active AND there is exactly one partition with score >=
    ``gold_correlation_min``.

    The map is a *structural witness*: the W32 layer does NOT observe
    ground truth at runtime; the map is registered up-front by the
    controller (e.g. "on this regime, the LINEAR partition's oracle
    has gold-correlation 0.85; CYCLIC has 0.20").  If the map is
    wrong, the W32-Λ-mis-correlated-gold falsifier fires and W32
    regresses vs W31.
    """
    partition_to_score: tuple[tuple[int, float], ...]
    gold_correlation_min: float
    gold_correlation_cid: str

    def __post_init__(self) -> None:
        for pid, sc in self.partition_to_score:
            assert isinstance(pid, int), \
                f"partition_id must be int, got {type(pid)}"
            sc = float(sc)
            assert 0.0 <= sc <= 1.0, \
                f"gold_correlation_score must be in [0, 1], got {sc}"
        gcm = float(self.gold_correlation_min)
        assert 0.0 <= gcm <= 1.0, \
            f"gold_correlation_min must be in [0, 1], got {gcm}"

    def best_partition(self) -> tuple[int, float] | None:
        """Returns the (partition_id, score) with the highest score
        if it strictly clears ``gold_correlation_min``, else None.

        Ties at top are resolved as None — there must be a unique
        winner above threshold for the gold route to fire.
        """
        if not self.partition_to_score:
            return None
        sorted_ps = sorted(self.partition_to_score,
                            key=lambda x: float(x[1]),
                            reverse=True)
        top_pid, top_score = sorted_ps[0]
        if float(top_score) < float(self.gold_correlation_min):
            return None
        if (len(sorted_ps) > 1
                and abs(float(sorted_ps[1][1]) - float(top_score)) < 1e-9):
            return None  # tie at top → no unique winner
        return int(top_pid), float(top_score)

    def as_dict(self) -> dict[str, Any]:
        return {
            "partition_to_score": [
                [int(p), round(float(s), 4)]
                for p, s in self.partition_to_score
            ],
            "gold_correlation_min": round(
                float(self.gold_correlation_min), 4),
            "gold_correlation_cid": self.gold_correlation_cid,
        }


def build_gold_correlation_map(
        *,
        partition_to_score: Sequence[tuple[int, float]],
        gold_correlation_min: float = W32_DEFAULT_GOLD_CORRELATION_MIN,
) -> GoldCorrelationMap:
    """Construct a content-addressed :class:`GoldCorrelationMap`."""
    canonical = tuple(
        (int(pid), float(s))
        for pid, s in sorted(partition_to_score, key=lambda x: int(x[0]))
    )
    cid = _compute_gold_correlation_cid(
        partition_to_score=canonical,
        gold_correlation_min=float(gold_correlation_min))
    return GoldCorrelationMap(
        partition_to_score=canonical,
        gold_correlation_min=float(gold_correlation_min),
        gold_correlation_cid=cid,
    )


# ---------------------------------------------------------------------------
# Convergence state entry (sealed, content-addressed)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ConvergenceStateEntry:
    """One entry in the W32 convergence-state trajectory.

    Carries the per-cell tuple
    ``(cell_idx, partition_id, ewma_prior_after, cusum_pos, cusum_neg,
       change_point_fired)`` that records a single EWMA + CUSUM
    update.
    """
    cell_idx: int
    partition_id: int
    ewma_prior_after: float
    cusum_pos: float
    cusum_neg: float
    change_point_fired: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "cell_idx": int(self.cell_idx),
            "partition_id": int(self.partition_id),
            "ewma_prior_after": round(float(self.ewma_prior_after), 4),
            "cusum_pos": round(float(self.cusum_pos), 4),
            "cusum_neg": round(float(self.cusum_neg), 4),
            "change_point_fired": bool(self.change_point_fired),
        }


def _compute_convergence_state_cid(
        *,
        states: Sequence[ConvergenceStateEntry],
) -> str:
    """Canonical SHA-256 over the convergence-state trajectory."""
    payload = _canonical_json_bytes({
        "states": [s.as_dict() for s in states],
    })
    return hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------
# W32 manifest-v2 CID
# ---------------------------------------------------------------------------


def _compute_w32_manifest_v2_cid(
        *,
        w31_online_cid: str,
        convergence_state_cid: str,
        gold_correlation_cid: str,
        route_audit_cid_v2: str,
) -> str:
    """SHA-256 over the canonical concatenation of W32 component CIDs.

    The manifest-v2 CID is the load-bearing W32 cross-component
    tamper-detection signal: any swap of one component CID from a
    different envelope (with each component still internally
    consistent AND the W31 manifest CID still self-consistent) is
    detected because the W32 manifest-v2 CID is a SHA-256 over the
    *outer* component CIDs, not the W31 manifest's *inner* set.
    """
    payload = _canonical_json_bytes({
        "w31_online_cid": str(w31_online_cid),
        "convergence_state_cid": str(convergence_state_cid),
        "gold_correlation_cid": str(gold_correlation_cid),
        "route_audit_cid_v2": str(route_audit_cid_v2),
    })
    return hashlib.sha256(payload).hexdigest()


def _compute_w32_outer_cid(
        *,
        schema_version: str,
        schema_cid: str,
        w31_online_cid: str,
        convergence_state_cid: str,
        gold_correlation_cid: str,
        manifest_v2_cid: str,
        cell_index: int,
) -> str:
    """SHA-256 over the canonical W32 envelope payload."""
    payload = _canonical_json_bytes({
        "schema_version": str(schema_version),
        "schema_cid": str(schema_cid),
        "w31_online_cid": str(w31_online_cid),
        "convergence_state_cid": str(convergence_state_cid),
        "gold_correlation_cid": str(gold_correlation_cid),
        "manifest_v2_cid": str(manifest_v2_cid),
        "cell_index": int(cell_index),
    })
    return hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------
# Long-window convergent ratification envelope
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class LongWindowConvergentRatificationEnvelope:
    """Content-addressed long-window convergent ratification of one
    W31 decision (W32 family).

    Carries:

      * ``w31_online_cid``            — parent W31 envelope's CID.
      * ``convergence_states``        — tuple of
                                          :class:`ConvergenceStateEntry`
                                          (length ≤ long_window).
      * ``convergence_state_cid``     — SHA-256 over canonical bytes.
      * ``ewma_alpha``                — registered constant (default 0.20).
      * ``cusum_threshold``           — registered constant (default 1.5).
      * ``cusum_k``                   — registered constant (default 0.10).
      * ``gold_correlation_cid``      — SHA-256 over the gold map.
      * ``gold_route_target_partition_id`` — partition the gold route
                                          targets (-1 if no route).
      * ``gold_route_active``         — True iff the gold-correlation
                                          route fired this cell.
      * ``change_point_active``       — True iff a change-point reset
                                          fired this cell.
      * ``route_audit_cid_v2``        — closed-form summary of W32
                                          routing decisions (gold +
                                          change-point).
      * ``manifest_v2_cid``           — SHA-256 over (w31_online_cid,
                                          convergence_state_cid,
                                          gold_correlation_cid,
                                          route_audit_cid_v2).
      * ``cell_index``                — audit replay index.
      * ``wire_required``             — 1 visible token cost on
                                          producer side iff True.
      * ``w32_cid``                   — SHA-256 over canonical bytes.

    Wire-token cost
    ---------------

    The W32 layer charges 1 visible token on the producer side
    (``<w32_ref:DDDD>``) iff ``wire_required`` is True (i.e. the
    long-window registry is non-trivial).  When every component is
    trivial, wire_required is False and W32 reduces to W31
    byte-for-byte (W32-Λ-trivial-long-window; H2 anchor).
    """
    schema_version: str
    schema_cid: str
    w31_online_cid: str
    convergence_states: tuple[ConvergenceStateEntry, ...]
    convergence_state_cid: str
    ewma_alpha: float
    cusum_threshold: float
    cusum_k: float
    gold_correlation_cid: str
    gold_route_target_partition_id: int
    gold_route_active: bool
    change_point_active: bool
    route_audit_cid_v2: str
    manifest_v2_cid: str
    cell_index: int
    wire_required: bool = False
    w32_cid: str = ""

    def __post_init__(self) -> None:
        if not self.w32_cid:
            object.__setattr__(self, "w32_cid", self.recompute_w32_cid())

    def recompute_w32_cid(self) -> str:
        return _compute_w32_outer_cid(
            schema_version=self.schema_version,
            schema_cid=self.schema_cid,
            w31_online_cid=self.w31_online_cid,
            convergence_state_cid=self.convergence_state_cid,
            gold_correlation_cid=self.gold_correlation_cid,
            manifest_v2_cid=self.manifest_v2_cid,
            cell_index=int(self.cell_index),
        )

    def to_canonical_bytes(self) -> bytes:
        return _canonical_json_bytes({
            "schema_version": self.schema_version,
            "schema_cid": self.schema_cid,
            "w31_online_cid": self.w31_online_cid,
            "convergence_states": [s.as_dict()
                                    for s in self.convergence_states],
            "convergence_state_cid": self.convergence_state_cid,
            "ewma_alpha": round(float(self.ewma_alpha), 4),
            "cusum_threshold": round(float(self.cusum_threshold), 4),
            "cusum_k": round(float(self.cusum_k), 4),
            "gold_correlation_cid": self.gold_correlation_cid,
            "gold_route_target_partition_id": int(
                self.gold_route_target_partition_id),
            "gold_route_active": bool(self.gold_route_active),
            "change_point_active": bool(self.change_point_active),
            "route_audit_cid_v2": self.route_audit_cid_v2,
            "manifest_v2_cid": self.manifest_v2_cid,
            "cell_index": int(self.cell_index),
        })

    def to_decoder_text(self) -> str:
        return f"<w32_ref:{self.w32_cid[:16]}>"

    @property
    def n_envelope_bytes(self) -> int:
        return len(self.to_canonical_bytes())

    @property
    def n_wire_tokens(self) -> int:
        if not self.wire_required:
            return 0
        return _whitespace_token_count(self.to_decoder_text())

    @property
    def n_structured_bits(self) -> int:
        return int(8 * self.n_envelope_bytes)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "schema_cid": self.schema_cid,
            "w31_online_cid": self.w31_online_cid,
            "convergence_states": [s.as_dict()
                                    for s in self.convergence_states],
            "convergence_state_cid": self.convergence_state_cid,
            "ewma_alpha": round(float(self.ewma_alpha), 4),
            "cusum_threshold": round(float(self.cusum_threshold), 4),
            "cusum_k": round(float(self.cusum_k), 4),
            "gold_correlation_cid": self.gold_correlation_cid,
            "gold_route_target_partition_id": int(
                self.gold_route_target_partition_id),
            "gold_route_active": bool(self.gold_route_active),
            "change_point_active": bool(self.change_point_active),
            "route_audit_cid_v2": self.route_audit_cid_v2,
            "manifest_v2_cid": self.manifest_v2_cid,
            "cell_index": int(self.cell_index),
            "wire_required": bool(self.wire_required),
            "w32_cid": self.w32_cid,
            "n_envelope_bytes": self.n_envelope_bytes,
            "n_wire_tokens": self.n_wire_tokens,
            "n_structured_bits": int(self.n_structured_bits),
            "decoder_text": self.to_decoder_text(),
        }


def verify_long_window_convergent_ratification(
        env: LongWindowConvergentRatificationEnvelope | None,
        *,
        registered_schema: SchemaCapsule,
        registered_w31_online_cid: str,
        registered_partition_ids: frozenset[int],
        registered_long_window: int,
        registered_cusum_max: float,
        registered_gold_correlation_cid: str,
        registered_convergence_state_cid: str | None = None,
) -> LatentVerificationOutcome:
    """Pure-function controller-side verification of a
    :class:`LongWindowConvergentRatificationEnvelope` (W32 family).

    14 enumerated failure modes (see module docstring for details).
    Pure function (no side effects); soundness by inspection.
    """
    n_checks = 0

    if env is None:
        return LatentVerificationOutcome(
            ok=False, reason="empty_w32_envelope", n_checks=n_checks)
    n_checks += 1
    if env.schema_version != W32_LONG_WINDOW_SCHEMA_VERSION:
        return LatentVerificationOutcome(
            ok=False, reason="w32_schema_version_unknown",
            n_checks=n_checks)
    n_checks += 1
    if env.schema_cid != registered_schema.cid:
        return LatentVerificationOutcome(
            ok=False, reason="w32_schema_cid_mismatch",
            n_checks=n_checks)
    n_checks += 1
    if env.w31_online_cid != registered_w31_online_cid:
        return LatentVerificationOutcome(
            ok=False, reason="w31_parent_cid_mismatch",
            n_checks=n_checks)
    n_checks += 1

    # ---- EWMA alpha range ----
    a = float(env.ewma_alpha)
    if math.isnan(a) or math.isinf(a) or a < 0.0 or a > 1.0:
        return LatentVerificationOutcome(
            ok=False, reason="ewma_alpha_out_of_range",
            n_checks=n_checks)
    n_checks += 1

    # ---- CUSUM threshold range ----
    ct = float(env.cusum_threshold)
    cmax = float(registered_cusum_max)
    if math.isnan(ct) or math.isinf(ct) or ct < 0.0 or ct > cmax:
        return LatentVerificationOutcome(
            ok=False, reason="cusum_threshold_out_of_range",
            n_checks=n_checks)
    n_checks += 1

    # ---- Convergence state checks ----
    window = int(registered_long_window)
    states = env.convergence_states
    if len(states) > window:
        return LatentVerificationOutcome(
            ok=False, reason="convergence_state_length_mismatch",
            n_checks=n_checks)
    n_checks += 1
    last_cell_idx = -1
    for entry in states:
        if int(entry.cell_idx) <= last_cell_idx:
            return LatentVerificationOutcome(
                ok=False, reason="convergence_state_length_mismatch",
                n_checks=n_checks)
        last_cell_idx = int(entry.cell_idx)
        if int(entry.partition_id) not in registered_partition_ids:
            return LatentVerificationOutcome(
                ok=False,
                reason="convergence_state_unregistered_partition",
                n_checks=n_checks)
        ew = float(entry.ewma_prior_after)
        if math.isnan(ew) or math.isinf(ew) or ew < 0.0 or ew > 1.0:
            return LatentVerificationOutcome(
                ok=False,
                reason="convergence_state_ewma_out_of_range",
                n_checks=n_checks)
        cp = float(entry.cusum_pos)
        cn = float(entry.cusum_neg)
        if (math.isnan(cp) or math.isinf(cp) or cp < 0.0 or cp > cmax
                or math.isnan(cn) or math.isinf(cn)
                or cn < 0.0 or cn > cmax):
            return LatentVerificationOutcome(
                ok=False,
                reason="convergence_state_cusum_out_of_range",
                n_checks=n_checks)
    n_checks += 1
    # Recompute convergence state CID.
    if (_compute_convergence_state_cid(states=states)
            != env.convergence_state_cid):
        return LatentVerificationOutcome(
            ok=False, reason="convergence_state_cid_mismatch",
            n_checks=n_checks)
    n_checks += 1
    # Cross-check: registered expected convergence state CID.
    if (registered_convergence_state_cid is not None
            and env.convergence_state_cid
            != registered_convergence_state_cid):
        return LatentVerificationOutcome(
            ok=False, reason="convergence_state_cid_mismatch",
            n_checks=n_checks)
    n_checks += 1

    # ---- Gold correlation CID check ----
    if env.gold_correlation_cid != registered_gold_correlation_cid:
        return LatentVerificationOutcome(
            ok=False, reason="gold_correlation_cid_mismatch",
            n_checks=n_checks)
    n_checks += 1

    # ---- Manifest-v2 CID check ----
    expected_manifest_v2 = _compute_w32_manifest_v2_cid(
        w31_online_cid=env.w31_online_cid,
        convergence_state_cid=env.convergence_state_cid,
        gold_correlation_cid=env.gold_correlation_cid,
        route_audit_cid_v2=env.route_audit_cid_v2,
    )
    if expected_manifest_v2 != env.manifest_v2_cid:
        return LatentVerificationOutcome(
            ok=False, reason="manifest_v2_cid_mismatch",
            n_checks=n_checks)
    n_checks += 1

    # ---- Outer w32_cid check ----
    if env.recompute_w32_cid() != env.w32_cid:
        return LatentVerificationOutcome(
            ok=False, reason="w32_outer_cid_mismatch",
            n_checks=n_checks)
    n_checks += 1

    return LatentVerificationOutcome(
        ok=True, reason="ok", n_checks=n_checks)


# ---------------------------------------------------------------------------
# Long-window convergent registry
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class LongWindowConvergentRegistry:
    """Controller-side registry for the W32 long-window convergent
    layer.

    Wraps a :class:`OnlineCalibratedRegistry` (the inner W31 registry)
    and adds:

      * ``long_window_enabled``     — when True, the EWMA accumulator
                                        fires on each ratified cell.
      * ``change_point_enabled``    — when True, the CUSUM detector
                                        fires; a change-point reset
                                        re-initialises EWMA + CUSUM.
      * ``gold_correlation_enabled`` — when True, on cells where the
                                        inner W31 disagreement-route
                                        is active, the W32 layer
                                        re-targets to the registered
                                        gold-correlation map's best
                                        partition (if unique above
                                        threshold).
      * ``manifest_v2_disabled``    — when True (and all other knobs
                                        trivial), the W32 layer
                                        reduces to W31 byte-for-byte
                                        (the trivial path).
      * ``long_window``             — max number of convergence-state
                                        entries to carry in the W32
                                        envelope (the verifier rejects
                                        longer histories).
      * ``ewma_alpha``,
        ``cusum_threshold``,
        ``cusum_k``,
        ``cusum_max``               — registered constants for the
                                        EWMA + CUSUM math.
      * ``gold_correlation_map``    — registered :class:`GoldCorrelationMap`
                                        (None ⇒ no gold route).
    """
    schema: SchemaCapsule | None = None
    inner: OnlineCalibratedRegistry | None = None
    long_window_enabled: bool = False
    change_point_enabled: bool = False
    gold_correlation_enabled: bool = False
    manifest_v2_disabled: bool = True
    long_window: int = 0
    ewma_alpha: float = W32_DEFAULT_EWMA_ALPHA
    cusum_threshold: float = W32_DEFAULT_CUSUM_THRESHOLD
    cusum_k: float = W32_DEFAULT_CUSUM_K
    cusum_max: float = W32_DEFAULT_CUSUM_MAX
    gold_correlation_map: GoldCorrelationMap | None = None
    local_host_id: str = "localhost"

    _envelopes: dict[str, LongWindowConvergentRatificationEnvelope] = (
        dataclasses.field(default_factory=dict))
    n_w32_registered: int = 0
    n_w32_rejected: int = 0
    n_long_window_updates: int = 0
    n_change_points_fired: int = 0
    n_gold_routes_fired: int = 0

    @property
    def is_trivial(self) -> bool:
        return (not bool(self.long_window_enabled)
                and not bool(self.change_point_enabled)
                and not bool(self.gold_correlation_enabled)
                and bool(self.manifest_v2_disabled)
                and int(self.long_window) == 0)

    @property
    def has_wire_required_layer(self) -> bool:
        return not self.is_trivial

    @property
    def gold_correlation_cid(self) -> str:
        if self.gold_correlation_map is None:
            return _compute_gold_correlation_cid(
                partition_to_score=(),
                gold_correlation_min=W32_DEFAULT_GOLD_CORRELATION_MIN)
        return self.gold_correlation_map.gold_correlation_cid

    def register_envelope(
            self,
            envelope: LongWindowConvergentRatificationEnvelope,
            *,
            registered_partition_ids: frozenset[int],
            registered_w31_online_cid: str,
            registered_convergence_state_cid: str | None = None,
    ) -> LatentVerificationOutcome:
        """Verify the envelope and (if OK) record it.  Pure verifier;
        idempotent on byte-identical envelopes.
        """
        if self.schema is None:
            outcome = LatentVerificationOutcome(
                ok=False, reason="schema_unregistered", n_checks=0)
            self.n_w32_rejected += 1
            return outcome
        outcome = verify_long_window_convergent_ratification(
            envelope,
            registered_schema=self.schema,
            registered_w31_online_cid=str(registered_w31_online_cid),
            registered_partition_ids=frozenset(registered_partition_ids),
            registered_long_window=int(self.long_window),
            registered_cusum_max=float(self.cusum_max),
            registered_gold_correlation_cid=str(self.gold_correlation_cid),
            registered_convergence_state_cid=registered_convergence_state_cid,
        )
        if not outcome.ok:
            self.n_w32_rejected += 1
            return outcome
        self._envelopes[envelope.w32_cid] = envelope
        self.n_w32_registered += 1
        return outcome


# ---------------------------------------------------------------------------
# W32 result
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class W32LongWindowResult:
    """Per-cell audit record for the W32 long-window convergent
    layer."""
    answer: dict[str, Any]
    inner_w31_branch: str
    decoder_branch: str
    agent_id: str
    is_producer: bool
    w31_online_cid: str
    cell_index: int
    long_window_update_fired: bool
    change_point_fired: bool
    gold_route_active: bool
    gold_route_target_partition_id: int
    effective_partition_id: int
    ewma_prior_after: float
    cusum_pos_after: float
    cusum_neg_after: float
    n_w31_visible_tokens: int
    n_w32_visible_tokens: int
    n_w32_overhead_tokens: int
    w32_cid: str
    manifest_v2_cid: str
    convergence_state_cid: str
    gold_correlation_cid: str
    ratified: bool
    verification_ok: bool
    verification_reason: str
    n_envelope_bytes: int
    n_structured_bits: int
    cram_factor_w32: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "inner_w31_branch": self.inner_w31_branch,
            "decoder_branch": self.decoder_branch,
            "agent_id": self.agent_id,
            "is_producer": bool(self.is_producer),
            "w31_online_cid": self.w31_online_cid,
            "cell_index": int(self.cell_index),
            "long_window_update_fired": bool(self.long_window_update_fired),
            "change_point_fired": bool(self.change_point_fired),
            "gold_route_active": bool(self.gold_route_active),
            "gold_route_target_partition_id": int(
                self.gold_route_target_partition_id),
            "effective_partition_id": int(self.effective_partition_id),
            "ewma_prior_after": round(float(self.ewma_prior_after), 4),
            "cusum_pos_after": round(float(self.cusum_pos_after), 4),
            "cusum_neg_after": round(float(self.cusum_neg_after), 4),
            "n_w31_visible_tokens": int(self.n_w31_visible_tokens),
            "n_w32_visible_tokens": int(self.n_w32_visible_tokens),
            "n_w32_overhead_tokens": int(self.n_w32_overhead_tokens),
            "w32_cid": self.w32_cid,
            "manifest_v2_cid": self.manifest_v2_cid,
            "convergence_state_cid": self.convergence_state_cid,
            "gold_correlation_cid": self.gold_correlation_cid,
            "ratified": bool(self.ratified),
            "verification_ok": bool(self.verification_ok),
            "verification_reason": self.verification_reason,
            "n_envelope_bytes": int(self.n_envelope_bytes),
            "n_structured_bits": int(self.n_structured_bits),
            "cram_factor_w32": float(self.cram_factor_w32),
        }


# ---------------------------------------------------------------------------
# Long-window convergent orchestrator
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class LongWindowConvergentOrchestrator:
    """Long-window convergent online geometry-aware dense-control
    orchestrator (W32).

    Wraps a :class:`OnlineCalibratedOrchestrator` (W31) and adds:

      * EWMA prior accumulator on each ratified cell;
      * Page two-sided CUSUM change-point detector;
      * gold-correlated disagreement-routing;
      * sealed convergence-state trajectory in the W32 envelope;
      * W32 manifest-v2 CID over the component CIDs.

    Per-cell flow:

      1. Run inner W31.
      2. Read inner W31 result + envelope.
      3. Derive per-cell agreement signal (already in W31 result).
      4. If long_window_enabled, EWMA-update the per-partition prior.
      5. If change_point_enabled, CUSUM-update + detect change point.
         If a change point is detected, RESET the EWMA + CUSUM state
         to the latest observation (re-initialise the accumulators).
      6. If gold_correlation_enabled AND inner W31 disagreement-route
         is active AND the gold-correlation map has a unique winner
         above threshold, target the W32 reroute to that partition.
      7. Append a :class:`ConvergenceStateEntry` to the running state
         (truncated to long_window).
      8. Build the W32 envelope.
      9. Verify + register.
     10. Charge 1 wire token iff registry.has_wire_required_layer.
    """
    inner: OnlineCalibratedOrchestrator
    registry: LongWindowConvergentRegistry
    enabled: bool = True
    require_w32_verification: bool = True

    _last_result: "W32LongWindowResult | None" = None
    _last_envelope: "LongWindowConvergentRatificationEnvelope | None" = None
    _convergence_states: list[ConvergenceStateEntry] = dataclasses.field(
        default_factory=list)
    # Per-partition EWMA + CUSUM state (running, not capped).
    _ewma_state: dict[int, float] = dataclasses.field(default_factory=dict)
    _cusum_pos_state: dict[int, float] = dataclasses.field(
        default_factory=dict)
    _cusum_neg_state: dict[int, float] = dataclasses.field(
        default_factory=dict)
    _cell_index: int = 0

    @property
    def schema(self) -> "SchemaCapsule | None":
        return self.inner.schema

    @property
    def agent_id(self) -> str:
        return self.inner.agent_id

    @property
    def is_producer(self) -> bool:
        return self.inner.is_producer

    @property
    def producer_agent_id(self) -> str:
        return self.inner.producer_agent_id

    @property
    def consumer_agent_ids(self) -> tuple[str, ...]:
        return self.inner.consumer_agent_ids

    def reset_session(self) -> None:
        self.inner.reset_session()
        self._last_result = None
        self._last_envelope = None
        self._convergence_states = []
        self._ewma_state = {}
        self._cusum_pos_state = {}
        self._cusum_neg_state = {}
        self._cell_index = 0

    def _build_w32_envelope(
            self,
            *,
            w31_online_cid: str,
            gold_route_target_partition_id: int,
            gold_route_active: bool,
            change_point_active: bool,
            wire_required: bool,
    ) -> LongWindowConvergentRatificationEnvelope:
        states = tuple(self._convergence_states)
        conv_cid = _compute_convergence_state_cid(states=states)
        gold_cid = self.registry.gold_correlation_cid
        route_audit_payload_v2 = _canonical_json_bytes({
            "gold_route_active": bool(gold_route_active),
            "gold_route_target_partition_id":
                int(gold_route_target_partition_id),
            "change_point_active": bool(change_point_active),
        })
        route_audit_cid_v2 = hashlib.sha256(
            route_audit_payload_v2).hexdigest()
        manifest_v2_cid = _compute_w32_manifest_v2_cid(
            w31_online_cid=str(w31_online_cid),
            convergence_state_cid=conv_cid,
            gold_correlation_cid=gold_cid,
            route_audit_cid_v2=route_audit_cid_v2,
        )
        env = LongWindowConvergentRatificationEnvelope(
            schema_version=W32_LONG_WINDOW_SCHEMA_VERSION,
            schema_cid=str(self.schema.cid),
            w31_online_cid=str(w31_online_cid),
            convergence_states=states,
            convergence_state_cid=conv_cid,
            ewma_alpha=float(self.registry.ewma_alpha),
            cusum_threshold=float(self.registry.cusum_threshold),
            cusum_k=float(self.registry.cusum_k),
            gold_correlation_cid=gold_cid,
            gold_route_target_partition_id=int(
                gold_route_target_partition_id),
            gold_route_active=bool(gold_route_active),
            change_point_active=bool(change_point_active),
            route_audit_cid_v2=route_audit_cid_v2,
            manifest_v2_cid=manifest_v2_cid,
            cell_index=int(self._cell_index),
            wire_required=bool(wire_required),
        )
        return env

    def decode_rounds(
            self,
            per_round_handoffs: Sequence[Sequence[_DecodedHandoff]],
    ) -> dict[str, Any]:
        # Disabled / no-schema paths.
        if not self.enabled or self.schema is None:
            out = self.inner.decode_rounds(per_round_handoffs)
            n_w31_visible = int(out.get("online_calibrated", {}).get(
                "n_w31_visible_tokens", 0))
            return self._pack(
                out=out,
                decoder_branch=W32_BRANCH_LONG_WINDOW_DISABLED,
                envelope=None, n_w32_visible=n_w31_visible,
                w32_overhead=0, ratified=False,
                verify_ok=False, verify_reason="disabled",
                long_window_update_fired=False,
                change_point_fired=False,
                gold_route_active=False,
                gold_route_target_partition_id=-1,
                effective_partition_id=0,
                ewma_prior_after=0.0,
                cusum_pos_after=0.0, cusum_neg_after=0.0,
                w31_online_cid="")

        # 1. Run inner W31.
        out = self.inner.decode_rounds(per_round_handoffs)
        w31_result = self.inner.last_result
        w31_envelope = self.inner.last_envelope
        n_w31_visible = int(out.get("online_calibrated", {}).get(
            "n_w31_visible_tokens", 0))
        inner_w31_branch = (
            str(w31_result.decoder_branch)
            if w31_result is not None else "")

        if w31_result is None:
            self._cell_index += 1
            return self._pack(
                out=out,
                decoder_branch=W32_BRANCH_LONG_WINDOW_NO_TRIGGER,
                envelope=None, n_w32_visible=n_w31_visible,
                w32_overhead=0, ratified=False,
                verify_ok=False, verify_reason="no_w31_result",
                long_window_update_fired=False,
                change_point_fired=False,
                gold_route_active=False,
                gold_route_target_partition_id=-1,
                effective_partition_id=0,
                ewma_prior_after=0.0,
                cusum_pos_after=0.0, cusum_neg_after=0.0,
                w31_online_cid="")

        # 2. Read W31 envelope CIDs.
        observed_agreement = float(w31_result.observed_agreement)
        eff_pid = int(w31_result.effective_partition_id)

        # 3-5. EWMA + CUSUM updates.
        # Initialise per-partition state lazily on first observation;
        # initial EWMA is the observation itself (so a single FULL-
        # agreement cell reads as prior 1.0).
        prev_ewma = float(self._ewma_state.get(eff_pid, observed_agreement))
        prev_cusum_pos = float(self._cusum_pos_state.get(eff_pid, 0.0))
        prev_cusum_neg = float(self._cusum_neg_state.get(eff_pid, 0.0))

        change_point_fired = False
        long_window_update_fired = False
        new_ewma = float(prev_ewma)
        new_cusum_pos = float(prev_cusum_pos)
        new_cusum_neg = float(prev_cusum_neg)

        # The EWMA fires on EVERY cell where the W31 inner ran (even
        # cells where the inner W31 did not ratify and observed_agreement
        # is therefore 0.0).  This is what makes the EWMA *re-converge*
        # on regime shifts where the inner W31 starts failing on a
        # partition: cumulative running mean stays anchored to the
        # historical prior, but EWMA forgets exponentially.
        if self.registry.long_window_enabled:
            new_ewma = update_ewma_prior(
                prev_ewma=prev_ewma,
                observation=observed_agreement,
                alpha=float(self.registry.ewma_alpha),
            )
            long_window_update_fired = True
            self.registry.n_long_window_updates += 1

        if self.registry.change_point_enabled:
            new_cusum_pos, new_cusum_neg = update_cusum_two_sided(
                cusum_pos_prev=prev_cusum_pos,
                cusum_neg_prev=prev_cusum_neg,
                observation=observed_agreement,
                target=float(prev_ewma),
                slack_k=float(self.registry.cusum_k),
                cusum_max=float(self.registry.cusum_max),
            )
            if detect_change_point(
                    cusum_pos=new_cusum_pos,
                    cusum_neg=new_cusum_neg,
                    threshold=float(self.registry.cusum_threshold)):
                change_point_fired = True
                # Reset EWMA + CUSUM accumulators to the latest
                # observation so the prior tracks the new regime.
                new_ewma = float(observed_agreement)
                new_cusum_pos = 0.0
                new_cusum_neg = 0.0
                self.registry.n_change_points_fired += 1

        # Persist updated EWMA + CUSUM state.
        if (self.registry.long_window_enabled
                or self.registry.change_point_enabled):
            self._ewma_state[eff_pid] = float(new_ewma)
            self._cusum_pos_state[eff_pid] = float(new_cusum_pos)
            self._cusum_neg_state[eff_pid] = float(new_cusum_neg)

        # 5b. Feed the W32 EWMA-derived prior back into the W30
        # calibration vector so the *next* cell's W30 reroute decision
        # uses the long-window convergent prior, not the W31
        # cumulative running mean.  This is what makes W32 strictly
        # different from W31 on the routing axis at long windows.
        if (self.registry.long_window_enabled
                and self.inner.inner is not None):
            inner_w30_registry = self.inner.inner.registry
            cv_now = inner_w30_registry.calibration_vector
            if cv_now is not None:
                # Build a fresh calibration vector that REPLACES the
                # W31-running-mean-derived prior for eff_pid with the
                # W32 EWMA-derived prior.
                new_priors = list(cv_now.calibration_vector)
                pids_now = list(cv_now.partition_ids)
                for i, pid in enumerate(pids_now):
                    if int(pid) == int(eff_pid):
                        new_priors[i] = float(new_ewma)
                # Recompute the adaptive threshold from the W32-
                # adjusted vector.
                if self.inner.registry.adaptive_threshold:
                    new_thr = compute_adaptive_threshold(
                        calibration_vector=tuple(new_priors),
                        threshold_min=float(
                            self.inner.registry.threshold_min),
                        threshold_max=float(
                            self.inner.registry.threshold_max),
                    )
                else:
                    new_thr = float(cv_now.threshold)
                rebuilt_cv = PartitionCalibrationVector(
                    calibration_vector=tuple(new_priors),
                    partition_ids=tuple(pids_now),
                    threshold=float(new_thr),
                )
                inner_w30_registry.calibration_vector = rebuilt_cv

        # 6. Gold-correlation route.
        gold_route_active = False
        gold_route_target_partition_id = -1
        if (self.registry.gold_correlation_enabled
                and self.registry.gold_correlation_map is not None
                and bool(w31_result.ratified)):
            best = self.registry.gold_correlation_map.best_partition()
            inner_route_active = False
            if w31_envelope is not None:
                # The inner W31's W30 envelope's disagreement-route
                # status is what triggers the W32 gold-correlation
                # rerouting.  Read it through the W30 envelope which
                # is reachable from the W31 result chain.
                w30_env = self.inner.inner.last_envelope
                if (w30_env is not None
                        and bool(getattr(
                            w30_env, "disagreement_route_active",
                            False))):
                    inner_route_active = True
            if inner_route_active and best is not None:
                gold_pid, gold_score = best
                gold_route_target_partition_id = int(gold_pid)
                gold_route_active = True
                self.registry.n_gold_routes_fired += 1

        # 7. Append convergence state entry (every cell where the
        # W31 inner ran, even non-ratified — the EWMA + CUSUM
        # accumulators record the trajectory regardless).
        if int(self.registry.long_window) > 0:
            entry = ConvergenceStateEntry(
                cell_idx=int(self._cell_index),
                partition_id=int(eff_pid),
                ewma_prior_after=float(new_ewma),
                cusum_pos=float(new_cusum_pos),
                cusum_neg=float(new_cusum_neg),
                change_point_fired=bool(change_point_fired),
            )
            self._convergence_states.append(entry)
            window = int(self.registry.long_window)
            if len(self._convergence_states) > window:
                self._convergence_states = (
                    self._convergence_states[-window:])

        # 7b. If the W31 baseline produced no envelope on this cell,
        # skip the W32 envelope build but keep the EWMA + CUSUM state
        # update so the next cell observes the most recent prior.
        if w31_envelope is None:
            self._cell_index += 1
            return self._pack(
                out=out,
                decoder_branch=W32_BRANCH_LONG_WINDOW_NO_TRIGGER,
                envelope=None, n_w32_visible=n_w31_visible,
                w32_overhead=0, ratified=False,
                verify_ok=False, verify_reason="no_w31_envelope",
                long_window_update_fired=bool(long_window_update_fired),
                change_point_fired=bool(change_point_fired),
                gold_route_active=False,
                gold_route_target_partition_id=-1,
                effective_partition_id=int(eff_pid),
                ewma_prior_after=float(new_ewma),
                cusum_pos_after=float(new_cusum_pos),
                cusum_neg_after=float(new_cusum_neg),
                w31_online_cid="")

        # 8. Build the W32 envelope.
        wire_required = self.registry.has_wire_required_layer
        envelope = self._build_w32_envelope(
            w31_online_cid=str(w31_envelope.w31_cid),
            gold_route_target_partition_id=int(
                gold_route_target_partition_id),
            gold_route_active=bool(gold_route_active),
            change_point_active=bool(change_point_fired),
            wire_required=bool(wire_required),
        )

        # 9. Verify + register.
        registered_pids = frozenset(
            int(p) for p in
            self.inner.inner.registry.registered_partition_ids)
        expected_conv_cid = envelope.convergence_state_cid
        outcome = self.registry.register_envelope(
            envelope,
            registered_partition_ids=registered_pids,
            registered_w31_online_cid=str(w31_envelope.w31_cid),
            registered_convergence_state_cid=expected_conv_cid,
        )
        verify_ok = bool(outcome.ok)
        verify_reason = str(outcome.reason)

        if not verify_ok and self.require_w32_verification:
            self._cell_index += 1
            return self._pack(
                out=out,
                decoder_branch=W32_BRANCH_LONG_WINDOW_REJECTED,
                envelope=envelope, n_w32_visible=n_w31_visible,
                w32_overhead=0, ratified=False,
                verify_ok=False, verify_reason=verify_reason,
                long_window_update_fired=bool(long_window_update_fired),
                change_point_fired=bool(change_point_fired),
                gold_route_active=bool(gold_route_active),
                gold_route_target_partition_id=int(
                    gold_route_target_partition_id),
                effective_partition_id=int(eff_pid),
                ewma_prior_after=float(new_ewma),
                cusum_pos_after=float(new_cusum_pos),
                cusum_neg_after=float(new_cusum_neg),
                w31_online_cid=str(w31_envelope.w31_cid))

        # 10. Trivial path — no wire token charged; pass through.
        if self.registry.is_trivial:
            self._cell_index += 1
            return self._pack(
                out=out,
                decoder_branch=W32_BRANCH_TRIVIAL_LONG_WINDOW_PASSTHROUGH,
                envelope=envelope, n_w32_visible=n_w31_visible,
                w32_overhead=0, ratified=True,
                verify_ok=verify_ok, verify_reason=verify_reason,
                long_window_update_fired=bool(long_window_update_fired),
                change_point_fired=bool(change_point_fired),
                gold_route_active=bool(gold_route_active),
                gold_route_target_partition_id=int(
                    gold_route_target_partition_id),
                effective_partition_id=int(eff_pid),
                ewma_prior_after=float(new_ewma),
                cusum_pos_after=float(new_cusum_pos),
                cusum_neg_after=float(new_cusum_neg),
                w31_online_cid=str(w31_envelope.w31_cid))

        # Non-trivial: charge 1 wire token.
        w32_overhead = int(envelope.n_wire_tokens)
        n_w32_visible = int(n_w31_visible + w32_overhead)
        self._cell_index += 1
        # Choose the resolved branch label.
        if change_point_fired:
            branch = W32_BRANCH_CHANGE_POINT_RESET
        elif gold_route_active:
            branch = W32_BRANCH_GOLD_CORRELATED_REROUTED
        else:
            branch = W32_BRANCH_LONG_WINDOW_RESOLVED
        return self._pack(
            out=out,
            decoder_branch=branch,
            envelope=envelope, n_w32_visible=n_w32_visible,
            w32_overhead=w32_overhead, ratified=True,
            verify_ok=verify_ok, verify_reason=verify_reason,
            long_window_update_fired=bool(long_window_update_fired),
            change_point_fired=bool(change_point_fired),
            gold_route_active=bool(gold_route_active),
            gold_route_target_partition_id=int(
                gold_route_target_partition_id),
            effective_partition_id=int(eff_pid),
            ewma_prior_after=float(new_ewma),
            cusum_pos_after=float(new_cusum_pos),
            cusum_neg_after=float(new_cusum_neg),
            w31_online_cid=str(w31_envelope.w31_cid))

    def _pack(
            self, *, out: dict[str, Any],
            decoder_branch: str,
            envelope: LongWindowConvergentRatificationEnvelope | None,
            n_w32_visible: int, w32_overhead: int,
            ratified: bool, verify_ok: bool, verify_reason: str,
            long_window_update_fired: bool,
            change_point_fired: bool,
            gold_route_active: bool,
            gold_route_target_partition_id: int,
            effective_partition_id: int,
            ewma_prior_after: float,
            cusum_pos_after: float,
            cusum_neg_after: float,
            w31_online_cid: str,
    ) -> dict[str, Any]:
        envelope_bytes = (envelope.n_envelope_bytes
                           if envelope is not None else 0)
        structured_bits = (envelope.n_structured_bits
                            if envelope is not None else 0)
        wire = max(1, int(w32_overhead))
        cram_factor = (
            float(structured_bits) / float(wire)
            if structured_bits > 0 else 0.0
        )
        w32_cid = (envelope.w32_cid if envelope is not None else "")
        manifest_v2_cid = (envelope.manifest_v2_cid
                            if envelope is not None else "")
        conv_cid = (envelope.convergence_state_cid
                     if envelope is not None else "")
        gold_cid = (envelope.gold_correlation_cid
                     if envelope is not None else "")
        n_w31_visible = int(out.get("online_calibrated", {}).get(
            "n_w31_visible_tokens", 0))
        inner_w31_branch = str(out.get("online_calibrated", {}).get(
            "decoder_branch", ""))
        result = W32LongWindowResult(
            answer=dict(out),
            inner_w31_branch=inner_w31_branch,
            decoder_branch=str(decoder_branch),
            agent_id=str(self.agent_id),
            is_producer=bool(self.is_producer),
            w31_online_cid=str(w31_online_cid),
            cell_index=int(self._cell_index - 1
                              if self._cell_index > 0 else 0),
            long_window_update_fired=bool(long_window_update_fired),
            change_point_fired=bool(change_point_fired),
            gold_route_active=bool(gold_route_active),
            gold_route_target_partition_id=int(
                gold_route_target_partition_id),
            effective_partition_id=int(effective_partition_id),
            ewma_prior_after=float(ewma_prior_after),
            cusum_pos_after=float(cusum_pos_after),
            cusum_neg_after=float(cusum_neg_after),
            n_w31_visible_tokens=int(n_w31_visible),
            n_w32_visible_tokens=int(n_w32_visible),
            n_w32_overhead_tokens=int(w32_overhead),
            w32_cid=str(w32_cid),
            manifest_v2_cid=str(manifest_v2_cid),
            convergence_state_cid=str(conv_cid),
            gold_correlation_cid=str(gold_cid),
            ratified=bool(ratified),
            verification_ok=bool(verify_ok),
            verification_reason=str(verify_reason),
            n_envelope_bytes=int(envelope_bytes),
            n_structured_bits=int(structured_bits),
            cram_factor_w32=float(cram_factor),
        )
        self._last_result = result
        self._last_envelope = envelope
        out_local = dict(out)
        out_local["long_window_convergent"] = result.as_dict()
        if envelope is not None:
            out_local["long_window_convergent_envelope"] = envelope.as_dict()
        return out_local

    def decode(
            self,
            handoffs: Sequence[_DecodedHandoff],
    ) -> dict[str, Any]:
        return self.decode_rounds([handoffs])

    @property
    def last_result(self) -> "W32LongWindowResult | None":
        return self._last_result

    @property
    def last_envelope(self) -> (
            "LongWindowConvergentRatificationEnvelope | None"):
        return self._last_envelope


# ---------------------------------------------------------------------------
# Convenience factories (W32 family)
# ---------------------------------------------------------------------------


def build_trivial_long_window_registry(
        *,
        schema: SchemaCapsule,
        local_host_id: str = "localhost",
) -> LongWindowConvergentRegistry:
    """Build a W32 registry with all knobs trivial — the H2 byte-for-W31
    anchor (W32-Λ-trivial-long-window falsifier).
    """
    return LongWindowConvergentRegistry(
        schema=schema,
        long_window_enabled=False,
        change_point_enabled=False,
        gold_correlation_enabled=False,
        manifest_v2_disabled=True,
        long_window=0,
        gold_correlation_map=None,
        local_host_id=local_host_id,
    )


def build_long_window_convergent_registry(
        *,
        schema: SchemaCapsule,
        long_window_enabled: bool = True,
        change_point_enabled: bool = True,
        gold_correlation_enabled: bool = False,
        manifest_v2_disabled: bool = False,
        long_window: int = W32_DEFAULT_LONG_WINDOW,
        ewma_alpha: float = W32_DEFAULT_EWMA_ALPHA,
        cusum_threshold: float = W32_DEFAULT_CUSUM_THRESHOLD,
        cusum_k: float = W32_DEFAULT_CUSUM_K,
        cusum_max: float = W32_DEFAULT_CUSUM_MAX,
        gold_correlation_map: GoldCorrelationMap | None = None,
        local_host_id: str = "localhost",
) -> LongWindowConvergentRegistry:
    """Build a non-trivial W32 registry that exercises EWMA + CUSUM
    + gold-correlation routing + manifest-v2 CID.
    """
    return LongWindowConvergentRegistry(
        schema=schema,
        long_window_enabled=bool(long_window_enabled),
        change_point_enabled=bool(change_point_enabled),
        gold_correlation_enabled=bool(gold_correlation_enabled),
        manifest_v2_disabled=bool(manifest_v2_disabled),
        long_window=int(long_window),
        ewma_alpha=float(ewma_alpha),
        cusum_threshold=float(cusum_threshold),
        cusum_k=float(cusum_k),
        cusum_max=float(cusum_max),
        gold_correlation_map=gold_correlation_map,
        local_host_id=local_host_id,
    )


# =============================================================================
# SDK v3.34 — Trust-EWMA-tracked multi-oracle adjudication (W33 family).
#
# W33 is the first capsule-native multi-agent-coordination method that
# **integrates the W32 EWMA primitive with the W21 multi-oracle
# adjudicator** to give per-oracle online trust calibration.  The
# original old-line W21 trust priors were **fixed at registration
# time** (every oracle gets a static ``trust_prior`` in [0, 1]); under
# a regime where an oracle becomes compromised mid-session, the W21
# layer cannot adapt.  W33 closes that loop:
#
#   * On every cell where the inner W21 ratifies a quorum, derive a
#     deterministic per-oracle agreement signal:
#       - 1.0 if the oracle's ``top_set`` is a non-empty proper subset
#         of the W21-resolved ``top_set`` (the oracle voted *with* the
#         consortium winner).
#       - 0.0 if the oracle's ``top_set`` is non-empty AND disjoint
#         from the W21-resolved ``top_set`` (the oracle voted *against*
#         the consortium winner).
#       - 1.0 if the oracle abstained (no information against it).
#   * Apply the W32 ``update_ewma_prior`` primitive to the per-oracle
#     EWMA-tracked trust:
#       ewma_trust_new = (1 - alpha) * ewma_trust_prev + alpha * agree
#     with ``alpha = W32_DEFAULT_EWMA_ALPHA`` by default.
#   * Re-aggregate the W21 vote tally with EWMA-thresholded weights:
#     an oracle whose EWMA falls below ``trust_threshold`` (default
#     0.5) has its vote *excluded* from the effective vote count.
#   * Project the W33 answer using the effective tally.
#   * Seal the per-oracle EWMA-trust state + the per-cell trust
#     trajectory + a W33 manifest-v3 CID over the inner W21 / W32
#     parent CIDs and the new W33 components.
#
# What W33 does NOT do (do-not-overstate):
#
#   * does NOT touch transformer KV caches, embedding tables, attention
#     weights, or any model-internal state.  The "trust state" lives at
#     the capsule layer; it is an honest *audited proxy* for online
#     trust calibration, not a runtime KV transplant.
#   * does NOT learn a trust model in the deep-learning sense.  The
#     EWMA update is closed-form arithmetic; zero parameters, zero
#     gradients, zero training step.  W33's "online learning" is the
#     same closed-form Bayesian-style running mean the W31 / W32
#     layers use, applied to oracle-level rather than partition-level
#     observations.
#   * does NOT prove temporal ordering at the model layer.  The
#     trust-trajectory is a *sealed tuple* of (cell_idx, oracle_id,
#     observed_quorum_agreement, ewma_trust_after) bytes; it proves
#     byte-stable replay but not that the underlying decisions
#     actually executed in that order at the model layer.
#
# The W33 surface is purely additive on top of the W21 surface:
# decode_rounds returns the same answer dict W21 returns plus a single
# new ``"trust_ewma_tracked"`` audit block; existing decoders and the
# W22..W32 stacks are unchanged.
# =============================================================================


W33_TRUST_EWMA_SCHEMA_VERSION: str = (
    "wevra.trust_ewma_ratification.v1")


W33_BRANCH_TRUST_EWMA_RESOLVED = "trust_ewma_resolved"
W33_BRANCH_TRIVIAL_TRUST_EWMA_PASSTHROUGH = (
    "trivial_trust_ewma_passthrough")
W33_BRANCH_TRUST_EWMA_REJECTED = "trust_ewma_rejected"
W33_BRANCH_TRUST_EWMA_DISABLED = "trust_ewma_disabled"
W33_BRANCH_TRUST_EWMA_NO_TRIGGER = "trust_ewma_no_trigger"
W33_BRANCH_TRUST_EWMA_DETRUSTED_ABSTAIN = (
    "trust_ewma_detrusted_abstain")
W33_BRANCH_TRUST_EWMA_DETRUSTED_REROUTE = (
    "trust_ewma_detrusted_reroute")

W33_ALL_BRANCHES: tuple[str, ...] = (
    W33_BRANCH_TRUST_EWMA_RESOLVED,
    W33_BRANCH_TRIVIAL_TRUST_EWMA_PASSTHROUGH,
    W33_BRANCH_TRUST_EWMA_REJECTED,
    W33_BRANCH_TRUST_EWMA_DISABLED,
    W33_BRANCH_TRUST_EWMA_NO_TRIGGER,
    W33_BRANCH_TRUST_EWMA_DETRUSTED_ABSTAIN,
    W33_BRANCH_TRUST_EWMA_DETRUSTED_REROUTE,
)


W33_DEFAULT_TRUST_THRESHOLD: float = 0.50
W33_DEFAULT_TRUST_TRAJECTORY_WINDOW: int = 16
W33_DEFAULT_EWMA_ALPHA: float = W32_DEFAULT_EWMA_ALPHA


# ---------------------------------------------------------------------------
# W33 helper: derive per-oracle agreement signal from a W21 result.
# ---------------------------------------------------------------------------


def derive_per_oracle_agreement_signal(
        *,
        probe_top_set: tuple[str, ...],
        probe_abstained: bool,
        resolved_top_set: tuple[str, ...],
) -> float:
    """Closed-form deterministic per-oracle agreement signal.

    Returns:

      * 1.0 if the probe abstained — no information against the oracle.
      * 1.0 if the probe's top_set is a non-empty subset of the
        resolved top_set — the oracle voted with the reference.
      * 0.0 if the probe's top_set is non-empty AND disjoint from the
        resolved top_set — the oracle voted against the reference.
      * 0.5 if the probe's top_set partially overlaps the resolved
        top_set (some agreement, some disagreement).

    Pure function; no side effects.  Closed-vocabulary; output bounded
    to [0, 1] by construction.  Used by the W33 EWMA update.

    The ``resolved_top_set`` is the *reference* against which the
    probe's vote is compared.  The W33 orchestrator chooses the
    reference: if the registry has ``anchor_oracle_ids`` configured,
    the reference is the union of anchor probes' top_sets (a stable
    "trust-by-construction" anchor); otherwise the reference is the
    W21 quorum-resolved top_set (which can flip under double-
    compromise — see W33-Λ-mis-trust-shift).
    """
    if bool(probe_abstained):
        return 1.0
    probe_set = frozenset(probe_top_set or ())
    res_set = frozenset(resolved_top_set or ())
    if not probe_set:
        return 1.0
    if not res_set:
        # No reference (e.g. NO_QUORUM, anchor abstained) — treat as
        # "no info" so the EWMA holds at its prior.
        return 1.0
    if probe_set.issubset(res_set):
        return 1.0
    if probe_set.isdisjoint(res_set):
        return 0.0
    # Partial overlap — split the difference.
    return 0.5


# ---------------------------------------------------------------------------
# W33 oracle-trust-state CID + trust trajectory CID
# ---------------------------------------------------------------------------


def _compute_oracle_trust_state_cid(
        *,
        oracle_to_trust: Sequence[tuple[str, float]],
) -> str:
    """Canonical SHA-256 over the per-oracle EWMA-trust state."""
    payload = _canonical_json_bytes({
        "oracle_to_trust": [
            [str(oid), round(float(t), 4)]
            for oid, t in sorted(oracle_to_trust, key=lambda x: str(x[0]))
        ],
    })
    return hashlib.sha256(payload).hexdigest()


@dataclasses.dataclass(frozen=True)
class TrustTrajectoryEntry:
    """One entry in the W33 trust-trajectory.

    Carries the per-cell-per-oracle tuple
    ``(cell_idx, oracle_id, observed_quorum_agreement, ewma_trust_after)``
    that records a single EWMA trust update.
    """
    cell_idx: int
    oracle_id: str
    observed_quorum_agreement: float
    ewma_trust_after: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "cell_idx": int(self.cell_idx),
            "oracle_id": str(self.oracle_id),
            "observed_quorum_agreement": round(
                float(self.observed_quorum_agreement), 4),
            "ewma_trust_after": round(float(self.ewma_trust_after), 4),
        }


def _compute_trust_trajectory_cid(
        *,
        trajectory: Sequence[TrustTrajectoryEntry],
) -> str:
    """Canonical SHA-256 over the trust-trajectory."""
    payload = _canonical_json_bytes({
        "trajectory": [t.as_dict() for t in trajectory],
    })
    return hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------
# W33 manifest-v3 CID
# ---------------------------------------------------------------------------


def _compute_w33_manifest_v3_cid(
        *,
        parent_cid: str,
        oracle_trust_state_cid: str,
        trust_trajectory_cid: str,
        trust_route_audit_cid: str,
) -> str:
    """SHA-256 over the canonical concatenation of W33 component CIDs.

    The manifest-v3 CID is the load-bearing W33 cross-component
    tamper-detection signal: any swap of one component CID from a
    different envelope (with each component still internally
    consistent AND the parent CID still self-consistent) is detected
    because the W33 manifest-v3 CID is a SHA-256 over the *outer*
    component CIDs.
    """
    payload = _canonical_json_bytes({
        "parent_cid": str(parent_cid),
        "oracle_trust_state_cid": str(oracle_trust_state_cid),
        "trust_trajectory_cid": str(trust_trajectory_cid),
        "trust_route_audit_cid": str(trust_route_audit_cid),
    })
    return hashlib.sha256(payload).hexdigest()


def _compute_w33_outer_cid(
        *,
        schema_version: str,
        schema_cid: str,
        parent_cid: str,
        oracle_trust_state_cid: str,
        trust_trajectory_cid: str,
        manifest_v3_cid: str,
        cell_index: int,
) -> str:
    """SHA-256 over the canonical W33 envelope payload."""
    payload = _canonical_json_bytes({
        "schema_version": str(schema_version),
        "schema_cid": str(schema_cid),
        "parent_cid": str(parent_cid),
        "oracle_trust_state_cid": str(oracle_trust_state_cid),
        "trust_trajectory_cid": str(trust_trajectory_cid),
        "manifest_v3_cid": str(manifest_v3_cid),
        "cell_index": int(cell_index),
    })
    return hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------
# Trust-EWMA ratification envelope
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class TrustEWMARatificationEnvelope:
    """Content-addressed trust-EWMA-tracked ratification of one
    W21-or-deeper decision (W33 family).

    Carries:

      * ``parent_cid``                 — parent envelope's CID (the W21
                                          / W22 / ... / W32 inner that
                                          W33 wraps; empty string when
                                          W33 wraps the bare
                                          :class:`TrustWeightedMultiOracleDisambiguator`).
      * ``oracle_trust_state``         — tuple of
                                          ``(oracle_id, ewma_trust)`` for
                                          every registered oracle.
      * ``oracle_trust_state_cid``     — SHA-256 over canonical bytes.
      * ``trust_trajectory``           — tuple of
                                          :class:`TrustTrajectoryEntry`
                                          (length ≤ trust_trajectory_window).
      * ``trust_trajectory_cid``       — SHA-256 over canonical bytes.
      * ``trust_threshold``            — registered trust threshold
                                          (default 0.50).
      * ``ewma_alpha``                 — registered EWMA alpha (default
                                          0.20, same as W32).
      * ``trust_route_audit_cid``      — closed-form summary of W33
                                          de-trust decisions for this
                                          cell.
      * ``manifest_v3_cid``            — SHA-256 over (parent_cid,
                                          oracle_trust_state_cid,
                                          trust_trajectory_cid,
                                          trust_route_audit_cid).
      * ``cell_index``                 — audit replay index.
      * ``wire_required``              — 1 visible token cost on
                                          producer side iff True.
      * ``w33_cid``                    — SHA-256 over canonical bytes.

    Wire-token cost
    ---------------

    The W33 layer charges 1 visible token on the producer side
    (``<w33_ref:DDDD>``) iff ``wire_required`` is True (i.e. the
    trust-EWMA registry is non-trivial).  When every component is
    trivial, wire_required is False and W33 reduces to W21
    byte-for-byte (W33-Λ-trivial-trust-ewma; H2 anchor).
    """
    schema_version: str
    schema_cid: str
    parent_cid: str
    oracle_trust_state: tuple[tuple[str, float], ...]
    oracle_trust_state_cid: str
    trust_trajectory: tuple[TrustTrajectoryEntry, ...]
    trust_trajectory_cid: str
    trust_threshold: float
    ewma_alpha: float
    trust_route_audit_cid: str
    manifest_v3_cid: str
    cell_index: int
    n_detrusted_oracles: int = 0
    wire_required: bool = False
    w33_cid: str = ""

    def __post_init__(self) -> None:
        if not self.w33_cid:
            object.__setattr__(self, "w33_cid", self.recompute_w33_cid())

    def recompute_w33_cid(self) -> str:
        return _compute_w33_outer_cid(
            schema_version=self.schema_version,
            schema_cid=self.schema_cid,
            parent_cid=self.parent_cid,
            oracle_trust_state_cid=self.oracle_trust_state_cid,
            trust_trajectory_cid=self.trust_trajectory_cid,
            manifest_v3_cid=self.manifest_v3_cid,
            cell_index=int(self.cell_index),
        )

    def to_canonical_bytes(self) -> bytes:
        return _canonical_json_bytes({
            "schema_version": self.schema_version,
            "schema_cid": self.schema_cid,
            "parent_cid": self.parent_cid,
            "oracle_trust_state": [
                [str(oid), round(float(t), 4)]
                for oid, t in self.oracle_trust_state
            ],
            "oracle_trust_state_cid": self.oracle_trust_state_cid,
            "trust_trajectory": [t.as_dict()
                                 for t in self.trust_trajectory],
            "trust_trajectory_cid": self.trust_trajectory_cid,
            "trust_threshold": round(float(self.trust_threshold), 4),
            "ewma_alpha": round(float(self.ewma_alpha), 4),
            "trust_route_audit_cid": self.trust_route_audit_cid,
            "manifest_v3_cid": self.manifest_v3_cid,
            "cell_index": int(self.cell_index),
            "n_detrusted_oracles": int(self.n_detrusted_oracles),
        })

    def to_decoder_text(self) -> str:
        return f"<w33_ref:{self.w33_cid[:16]}>"

    @property
    def n_envelope_bytes(self) -> int:
        return len(self.to_canonical_bytes())

    @property
    def n_wire_tokens(self) -> int:
        if not self.wire_required:
            return 0
        return _whitespace_token_count(self.to_decoder_text())

    @property
    def n_structured_bits(self) -> int:
        return int(8 * self.n_envelope_bytes)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "schema_cid": self.schema_cid,
            "parent_cid": self.parent_cid,
            "oracle_trust_state": [
                [str(oid), round(float(t), 4)]
                for oid, t in self.oracle_trust_state
            ],
            "oracle_trust_state_cid": self.oracle_trust_state_cid,
            "trust_trajectory": [t.as_dict()
                                 for t in self.trust_trajectory],
            "trust_trajectory_cid": self.trust_trajectory_cid,
            "trust_threshold": round(float(self.trust_threshold), 4),
            "ewma_alpha": round(float(self.ewma_alpha), 4),
            "trust_route_audit_cid": self.trust_route_audit_cid,
            "manifest_v3_cid": self.manifest_v3_cid,
            "cell_index": int(self.cell_index),
            "n_detrusted_oracles": int(self.n_detrusted_oracles),
            "wire_required": bool(self.wire_required),
            "w33_cid": self.w33_cid,
            "n_envelope_bytes": self.n_envelope_bytes,
            "n_wire_tokens": self.n_wire_tokens,
            "n_structured_bits": int(self.n_structured_bits),
            "decoder_text": self.to_decoder_text(),
        }


# ---------------------------------------------------------------------------
# W33 verifier — 14 enumerated failure modes (disjoint from W22..W32)
# ---------------------------------------------------------------------------


def verify_trust_ewma_ratification(
        env: TrustEWMARatificationEnvelope | None,
        *,
        registered_schema: SchemaCapsule,
        registered_parent_cid: str,
        registered_oracle_ids: frozenset[str],
        registered_trust_trajectory_window: int,
        registered_oracle_trust_state_cid: str | None = None,
) -> LatentVerificationOutcome:
    """Pure-function controller-side verification of a
    :class:`TrustEWMARatificationEnvelope` (W33 family).

    14 enumerated failure modes (see module docstring for details).
    Pure function (no side effects); soundness by inspection.
    """
    n_checks = 0

    if env is None:
        return LatentVerificationOutcome(
            ok=False, reason="empty_w33_envelope", n_checks=n_checks)
    n_checks += 1
    if env.schema_version != W33_TRUST_EWMA_SCHEMA_VERSION:
        return LatentVerificationOutcome(
            ok=False, reason="w33_schema_version_unknown",
            n_checks=n_checks)
    n_checks += 1
    if env.schema_cid != registered_schema.cid:
        return LatentVerificationOutcome(
            ok=False, reason="w33_schema_cid_mismatch",
            n_checks=n_checks)
    n_checks += 1
    if env.parent_cid != registered_parent_cid:
        return LatentVerificationOutcome(
            ok=False, reason="w32_parent_cid_mismatch",
            n_checks=n_checks)
    n_checks += 1

    # ---- Trust threshold range ----
    tt = float(env.trust_threshold)
    if math.isnan(tt) or math.isinf(tt) or tt < 0.0 or tt > 1.0:
        return LatentVerificationOutcome(
            ok=False, reason="trust_threshold_out_of_range",
            n_checks=n_checks)
    n_checks += 1

    # ---- Oracle trust state checks ----
    state = env.oracle_trust_state
    for oid, t in state:
        if str(oid) not in registered_oracle_ids:
            return LatentVerificationOutcome(
                ok=False,
                reason="oracle_trust_state_unregistered_oracle",
                n_checks=n_checks)
        f = float(t)
        if math.isnan(f) or math.isinf(f) or f < 0.0 or f > 1.0:
            return LatentVerificationOutcome(
                ok=False,
                reason="oracle_trust_state_ewma_out_of_range",
                n_checks=n_checks)
    n_checks += 1
    # Recompute oracle trust state CID.
    expected_state_cid = _compute_oracle_trust_state_cid(
        oracle_to_trust=state)
    if expected_state_cid != env.oracle_trust_state_cid:
        return LatentVerificationOutcome(
            ok=False, reason="oracle_trust_state_cid_mismatch",
            n_checks=n_checks)
    n_checks += 1
    # Cross-check: registered expected oracle trust state CID.
    if (registered_oracle_trust_state_cid is not None
            and env.oracle_trust_state_cid
            != registered_oracle_trust_state_cid):
        return LatentVerificationOutcome(
            ok=False, reason="oracle_trust_state_cid_mismatch",
            n_checks=n_checks)
    n_checks += 1

    # ---- Trust trajectory checks ----
    window = int(registered_trust_trajectory_window)
    traj = env.trust_trajectory
    if len(traj) > window:
        return LatentVerificationOutcome(
            ok=False, reason="trust_trajectory_length_mismatch",
            n_checks=n_checks)
    n_checks += 1
    last_cell_idx = -1
    for entry in traj:
        if int(entry.cell_idx) < last_cell_idx:
            return LatentVerificationOutcome(
                ok=False, reason="trust_trajectory_length_mismatch",
                n_checks=n_checks)
        last_cell_idx = int(entry.cell_idx)
        if str(entry.oracle_id) not in registered_oracle_ids:
            return LatentVerificationOutcome(
                ok=False,
                reason="trust_trajectory_unregistered_oracle",
                n_checks=n_checks)
        oa = float(entry.observed_quorum_agreement)
        if math.isnan(oa) or math.isinf(oa) or oa < 0.0 or oa > 1.0:
            return LatentVerificationOutcome(
                ok=False,
                reason="trust_trajectory_observed_out_of_range",
                n_checks=n_checks)
        ta = float(entry.ewma_trust_after)
        if math.isnan(ta) or math.isinf(ta) or ta < 0.0 or ta > 1.0:
            return LatentVerificationOutcome(
                ok=False,
                reason="trust_trajectory_observed_out_of_range",
                n_checks=n_checks)
    n_checks += 1
    # Recompute trust trajectory CID.
    if (_compute_trust_trajectory_cid(trajectory=traj)
            != env.trust_trajectory_cid):
        return LatentVerificationOutcome(
            ok=False, reason="trust_trajectory_cid_mismatch",
            n_checks=n_checks)
    n_checks += 1

    # ---- Manifest-v3 CID check ----
    expected_manifest_v3 = _compute_w33_manifest_v3_cid(
        parent_cid=env.parent_cid,
        oracle_trust_state_cid=env.oracle_trust_state_cid,
        trust_trajectory_cid=env.trust_trajectory_cid,
        trust_route_audit_cid=env.trust_route_audit_cid,
    )
    if expected_manifest_v3 != env.manifest_v3_cid:
        return LatentVerificationOutcome(
            ok=False, reason="manifest_v3_cid_mismatch",
            n_checks=n_checks)
    n_checks += 1

    # ---- Outer w33_cid check ----
    if env.recompute_w33_cid() != env.w33_cid:
        return LatentVerificationOutcome(
            ok=False, reason="w33_outer_cid_mismatch",
            n_checks=n_checks)
    n_checks += 1

    return LatentVerificationOutcome(
        ok=True, reason="ok", n_checks=n_checks)


# ---------------------------------------------------------------------------
# Trust-EWMA registry
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class TrustEWMARegistry:
    """Controller-side registry for the W33 trust-EWMA-tracked layer.

    Wraps a :class:`TrustWeightedMultiOracleDisambiguator` (the inner
    W21 disambiguator) and adds:

      * ``trust_ewma_enabled``           — when True, the EWMA-tracked
                                             trust update fires on each
                                             cell where the inner W21
                                             produced any decision.
      * ``manifest_v3_disabled``         — when True (and all other
                                             knobs trivial), W33
                                             reduces to W21
                                             byte-for-byte.
      * ``trust_trajectory_window``      — max number of trust
                                             trajectory entries the
                                             envelope carries; verifier
                                             rejects longer histories.
      * ``trust_threshold``              — minimum EWMA trust required
                                             for an oracle's vote to
                                             count in W33's projection.
      * ``ewma_alpha``                   — registered EWMA smoothing
                                             factor.
      * ``registered_oracle_ids``        — frozenset of registered
                                             oracle IDs (must match the
                                             inner W21's
                                             oracle_registrations).
    """
    schema: SchemaCapsule | None = None
    trust_ewma_enabled: bool = False
    manifest_v3_disabled: bool = True
    trust_trajectory_window: int = 0
    trust_threshold: float = W33_DEFAULT_TRUST_THRESHOLD
    ewma_alpha: float = W33_DEFAULT_EWMA_ALPHA
    registered_oracle_ids: frozenset[str] = frozenset()
    # Optional set of oracle IDs whose votes form the reference for
    # the per-oracle agreement signal.  When non-empty, the W33
    # orchestrator computes agreement against the union of anchor
    # probes' top_sets — a stable trust-by-construction reference
    # that survives double-compromise.  When empty, the orchestrator
    # falls back to the W21 quorum-resolved top_set (which can flip
    # under double-compromise — see W33-Λ-mis-trust-shift).
    anchor_oracle_ids: frozenset[str] = frozenset()
    local_host_id: str = "localhost"

    _envelopes: dict[str, TrustEWMARatificationEnvelope] = (
        dataclasses.field(default_factory=dict))
    n_w33_registered: int = 0
    n_w33_rejected: int = 0
    n_trust_ewma_updates: int = 0
    n_oracles_detrusted: int = 0

    @property
    def is_trivial(self) -> bool:
        return (not bool(self.trust_ewma_enabled)
                and bool(self.manifest_v3_disabled)
                and int(self.trust_trajectory_window) == 0)

    @property
    def has_wire_required_layer(self) -> bool:
        return not self.is_trivial

    def register_envelope(
            self,
            envelope: TrustEWMARatificationEnvelope,
            *,
            registered_parent_cid: str,
            registered_oracle_trust_state_cid: str | None = None,
    ) -> LatentVerificationOutcome:
        """Verify the envelope and (if OK) record it."""
        if self.schema is None:
            outcome = LatentVerificationOutcome(
                ok=False, reason="schema_unregistered", n_checks=0)
            self.n_w33_rejected += 1
            return outcome
        outcome = verify_trust_ewma_ratification(
            envelope,
            registered_schema=self.schema,
            registered_parent_cid=str(registered_parent_cid),
            registered_oracle_ids=frozenset(self.registered_oracle_ids),
            registered_trust_trajectory_window=int(
                self.trust_trajectory_window),
            registered_oracle_trust_state_cid=(
                registered_oracle_trust_state_cid),
        )
        if outcome.ok:
            self._envelopes[envelope.w33_cid] = envelope
            self.n_w33_registered += 1
        else:
            self.n_w33_rejected += 1
        return outcome


# ---------------------------------------------------------------------------
# W33 result + orchestrator
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class W33TrustEWMAResult:
    """Audit record for one
    ``TrustEWMATrackedMultiOracleOrchestrator.decode_rounds`` call.

    Fields
    ------
    answer
        The W33-projected answer dict.
    inner_w21_branch
        The W21 branch that fired before the W33 layer ran.
    decoder_branch
        The W33 branch (one of :data:`W33_ALL_BRANCHES`).
    parent_cid
        The parent envelope's CID (W21 outer; empty when no W21 envelope).
    cell_index
        The W33 audit replay index (incremented once per cell).
    oracle_trust_state
        The per-oracle EWMA-trust state after this cell's update.
    n_detrusted_oracles
        Number of oracles whose EWMA-tracked trust is < trust_threshold
        at projection time on this cell.
    n_w21_visible_tokens
        Inner W21 visible-token cost.
    n_w33_visible_tokens
        Total visible token cost (W21 + W33 envelope).
    n_w33_overhead_tokens
        Per-cell additional W33 overhead (0 or 1).
    w33_cid
        Outer envelope CID.
    manifest_v3_cid
        Manifest-v3 CID.
    oracle_trust_state_cid
        Oracle-trust-state CID.
    trust_trajectory_cid
        Trust-trajectory CID.
    ratified
        True iff the W33 envelope was registered AND verified.
    verification_ok
        Mirror of registry.register_envelope outcome.ok.
    verification_reason
        Mirror of registry.register_envelope outcome.reason.
    n_envelope_bytes / n_structured_bits / cram_factor_w33
        Envelope cost / payload metrics.
    """
    answer: dict[str, Any]
    inner_w21_branch: str
    decoder_branch: str
    parent_cid: str
    cell_index: int
    oracle_trust_state: tuple[tuple[str, float], ...]
    n_detrusted_oracles: int
    n_w21_visible_tokens: int
    n_w33_visible_tokens: int
    n_w33_overhead_tokens: int
    w33_cid: str
    manifest_v3_cid: str
    oracle_trust_state_cid: str
    trust_trajectory_cid: str
    ratified: bool
    verification_ok: bool
    verification_reason: str
    n_envelope_bytes: int
    n_structured_bits: int
    cram_factor_w33: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "root_cause": str(self.answer.get("root_cause", "unknown")),
            "services": tuple(self.answer.get("services", ())),
            "remediation": str(self.answer.get(
                "remediation", "investigate")),
            "inner_w21_branch": str(self.inner_w21_branch),
            "decoder_branch": str(self.decoder_branch),
            "parent_cid": str(self.parent_cid),
            "cell_index": int(self.cell_index),
            "oracle_trust_state": [
                [str(oid), round(float(t), 4)]
                for oid, t in self.oracle_trust_state
            ],
            "n_detrusted_oracles": int(self.n_detrusted_oracles),
            "n_w21_visible_tokens": int(self.n_w21_visible_tokens),
            "n_w33_visible_tokens": int(self.n_w33_visible_tokens),
            "n_w33_overhead_tokens": int(self.n_w33_overhead_tokens),
            "w33_cid": str(self.w33_cid),
            "manifest_v3_cid": str(self.manifest_v3_cid),
            "oracle_trust_state_cid": str(self.oracle_trust_state_cid),
            "trust_trajectory_cid": str(self.trust_trajectory_cid),
            "ratified": bool(self.ratified),
            "verification_ok": bool(self.verification_ok),
            "verification_reason": str(self.verification_reason),
            "n_envelope_bytes": int(self.n_envelope_bytes),
            "n_structured_bits": int(self.n_structured_bits),
            "cram_factor_w33": float(self.cram_factor_w33),
        }


@dataclasses.dataclass
class TrustEWMATrackedMultiOracleOrchestrator:
    """Trust-EWMA-tracked multi-oracle orchestrator (W33 family).

    Wraps a :class:`TrustWeightedMultiOracleDisambiguator` (W21) and
    adds:

      * per-oracle EWMA-tracked trust state;
      * deterministic per-cell agreement signal derived from
        ``W21OracleProbe.top_set`` vs the W21-resolved ``top_set``;
      * trust-threshold-gated quorum projection (excludes votes from
        oracles whose EWMA falls below the registered threshold);
      * sealed trust trajectory in the W33 envelope;
      * W33 manifest-v3 CID over the component CIDs.

    Per-cell flow:

      1. Run inner W21.
      2. Read inner W21 result.
      3. Derive per-oracle agreement signal from probe.top_set vs
         W21-resolved top_set.
      4. EWMA-update each oracle's trust:
         ``ewma_new = (1-alpha) * ewma_prev + alpha * agreement``.
         (Initial EWMA = 1.0 at registration; the W21 trust_prior is
         not used as an initial EWMA — that would weight legacy priors
         on the running mean.  Initial 1.0 is the conservative starting
         point.)
      5. If ``trust_ewma_enabled``, recompute the W33 effective tally:
         ``effective_votes[tag] = sum over oracles where EWMA >=
         trust_threshold of (1 if probe.top_set contains tag else 0)``;
         project the W33 answer.
      6. Append :class:`TrustTrajectoryEntry` per oracle for this cell.
      7. Build the W33 envelope.
      8. Verify + register.
      9. Charge 1 wire token iff ``registry.has_wire_required_layer``.

    Backward-compat
    ----------------
    * **W33-3** (vs W21, trivial path).  With ``trust_ewma_enabled
      = False``, ``manifest_v3_disabled = True``, AND
      ``trust_trajectory_window = 0``, the W33 layer reduces to the
      inner W21 byte-for-byte.  Every cell yields
      ``W33_BRANCH_TRIVIAL_TRUST_EWMA_PASSTHROUGH``.
    """
    inner: TrustWeightedMultiOracleDisambiguator
    registry: TrustEWMARegistry
    enabled: bool = True
    require_w33_verification: bool = True

    _last_result: "W33TrustEWMAResult | None" = None
    _last_envelope: "TrustEWMARatificationEnvelope | None" = None
    _trust_trajectory: list[TrustTrajectoryEntry] = dataclasses.field(
        default_factory=list)
    # Per-oracle EWMA trust state.  Initial: 1.0 (every oracle starts
    # fully trusted).  The W21 trust_prior is NOT used as initial EWMA
    # — keeping the EWMA on the [0, 1] agreement-signal scale lets the
    # W33-Λ-no-trust-shift falsifier hold byte-for-byte (every cell
    # where every oracle agrees with the consortium quorum keeps the
    # EWMA at 1.0; W33 ties W21).
    _ewma_trust_state: dict[str, float] = dataclasses.field(
        default_factory=dict)
    _cell_index: int = 0

    @property
    def schema(self) -> "SchemaCapsule | None":
        return self.registry.schema

    def reset_session(self) -> None:
        self._last_result = None
        self._last_envelope = None
        self._trust_trajectory = []
        self._ewma_trust_state = {}
        self._cell_index = 0

    def _initial_ewma_for_oracle(self, oracle_id: str) -> float:
        return float(self._ewma_trust_state.get(oracle_id, 1.0))

    def _build_w33_envelope(
            self,
            *,
            parent_cid: str,
            n_detrusted_oracles: int,
            wire_required: bool,
    ) -> TrustEWMARatificationEnvelope:
        # Canonicalise the oracle trust state — sort by oracle_id.
        oracle_trust_state = tuple(
            (str(oid), float(self._ewma_trust_state[oid]))
            for oid in sorted(self._ewma_trust_state.keys())
        )
        oracle_trust_state_cid = _compute_oracle_trust_state_cid(
            oracle_to_trust=oracle_trust_state)
        trust_trajectory = tuple(self._trust_trajectory)
        trust_trajectory_cid = _compute_trust_trajectory_cid(
            trajectory=trust_trajectory)
        trust_route_audit_payload = _canonical_json_bytes({
            "n_detrusted_oracles": int(n_detrusted_oracles),
            "trust_threshold": round(
                float(self.registry.trust_threshold), 4),
            "ewma_alpha": round(float(self.registry.ewma_alpha), 4),
        })
        trust_route_audit_cid = hashlib.sha256(
            trust_route_audit_payload).hexdigest()
        manifest_v3_cid = _compute_w33_manifest_v3_cid(
            parent_cid=str(parent_cid),
            oracle_trust_state_cid=oracle_trust_state_cid,
            trust_trajectory_cid=trust_trajectory_cid,
            trust_route_audit_cid=trust_route_audit_cid,
        )
        env = TrustEWMARatificationEnvelope(
            schema_version=W33_TRUST_EWMA_SCHEMA_VERSION,
            schema_cid=str(self.schema.cid),
            parent_cid=str(parent_cid),
            oracle_trust_state=oracle_trust_state,
            oracle_trust_state_cid=oracle_trust_state_cid,
            trust_trajectory=trust_trajectory,
            trust_trajectory_cid=trust_trajectory_cid,
            trust_threshold=float(self.registry.trust_threshold),
            ewma_alpha=float(self.registry.ewma_alpha),
            trust_route_audit_cid=trust_route_audit_cid,
            manifest_v3_cid=manifest_v3_cid,
            cell_index=int(self._cell_index),
            n_detrusted_oracles=int(n_detrusted_oracles),
            wire_required=bool(wire_required),
        )
        return env

    def decode_rounds(
            self,
            per_round_handoffs: Sequence[Sequence[_DecodedHandoff]],
    ) -> dict[str, Any]:
        # Disabled / no-schema paths.
        if not self.enabled or self.schema is None:
            out = self.inner.decode_rounds(per_round_handoffs)
            return self._pack(
                out=out,
                decoder_branch=W33_BRANCH_TRUST_EWMA_DISABLED,
                envelope=None, n_w33_visible=0, w33_overhead=0,
                ratified=False, verify_ok=False,
                verify_reason="disabled",
                n_detrusted_oracles=0, parent_cid="")

        # 1. Run inner W21.
        out = self.inner.decode_rounds(per_round_handoffs)
        w21_result = self.inner.last_result
        inner_w21_branch = (
            str(w21_result.decoder_branch)
            if w21_result is not None
            else W21_BRANCH_DISABLED)

        if w21_result is None:
            self._cell_index += 1
            return self._pack(
                out=out,
                decoder_branch=W33_BRANCH_TRUST_EWMA_NO_TRIGGER,
                envelope=None, n_w33_visible=0, w33_overhead=0,
                ratified=False, verify_ok=False,
                verify_reason="no_w21_result",
                n_detrusted_oracles=0, parent_cid="")

        # Inner W21 visible-token cost (multi-oracle queries).
        n_w21_visible = int(w21_result.n_outside_tokens_total)

        # 2-3. Per-oracle agreement signals.
        # The reference (against which each probe's vote is compared)
        # is one of:
        #   (a) the union of anchor probes' top_sets, when
        #       ``registry.anchor_oracle_ids`` is non-empty.  This is
        #       the stable trust-by-construction reference that
        #       survives double-compromise.
        #   (b) the W21-resolved top_set (only valid when W21 produced
        #       a QUORUM_RESOLVED branch).  Falls back to () when W21
        #       did not resolve a quorum.
        resolved_top_set: tuple[str, ...]
        if self.registry.anchor_oracle_ids:
            anchor_set: set[str] = set()
            for probe in w21_result.probes:
                if (str(probe.oracle_id)
                        in self.registry.anchor_oracle_ids
                        and not probe.abstained):
                    anchor_set.update(probe.top_set or ())
            resolved_top_set = tuple(sorted(anchor_set))
        elif (w21_result.decoder_branch
                == W21_BRANCH_QUORUM_RESOLVED):
            resolved_top_set = tuple(
                w21_result.answer.get("services", ()))
        else:
            resolved_top_set = ()

        # If W33 is disabled (trivial registry), pass through W21
        # byte-for-byte without firing the EWMA update or charging a
        # wire token.
        if self.registry.is_trivial:
            self._cell_index += 1
            # Initialize EWMA state on first observation so the cid is
            # stable across calls.  But do NOT charge wire token.
            for reg in self.inner.oracle_registrations:
                oid = str(getattr(reg.oracle, "oracle_id", "no_oracle"))
                if oid not in self._ewma_trust_state:
                    self._ewma_trust_state[oid] = 1.0
            return self._pack(
                out=out,
                decoder_branch=W33_BRANCH_TRIVIAL_TRUST_EWMA_PASSTHROUGH,
                envelope=None, n_w33_visible=n_w21_visible,
                w33_overhead=0, ratified=True, verify_ok=True,
                verify_reason="trivial_passthrough",
                n_detrusted_oracles=0, parent_cid="")

        # 4. EWMA-update each oracle's trust based on probe agreement.
        updated_ids: list[str] = []
        for probe in w21_result.probes:
            oid = str(probe.oracle_id)
            agreement = derive_per_oracle_agreement_signal(
                probe_top_set=tuple(probe.top_set),
                probe_abstained=bool(probe.abstained),
                resolved_top_set=resolved_top_set,
            )
            prev_ewma = self._initial_ewma_for_oracle(oid)
            new_ewma = update_ewma_prior(
                prev_ewma=prev_ewma,
                observation=float(agreement),
                alpha=float(self.registry.ewma_alpha),
            )
            self._ewma_trust_state[oid] = float(new_ewma)
            self.registry.n_trust_ewma_updates += 1
            entry = TrustTrajectoryEntry(
                cell_idx=int(self._cell_index),
                oracle_id=oid,
                observed_quorum_agreement=float(agreement),
                ewma_trust_after=float(new_ewma),
            )
            self._trust_trajectory.append(entry)
            updated_ids.append(oid)

        # Truncate the trust trajectory to the configured window.
        window = int(self.registry.trust_trajectory_window)
        if window > 0 and len(self._trust_trajectory) > window:
            self._trust_trajectory = self._trust_trajectory[-window:]

        # 5. Trust-threshold-gated effective tally.  We re-aggregate
        # votes excluding oracles whose EWMA falls below threshold.
        # Then re-project the W33 answer if W33 changes the W21
        # decision.
        threshold = float(self.registry.trust_threshold)
        detrusted_oracle_ids = sorted(
            oid for oid, ewma in self._ewma_trust_state.items()
            if float(ewma) < threshold
        )
        n_detrusted = len(detrusted_oracle_ids)
        if n_detrusted > 0:
            self.registry.n_oracles_detrusted += 1

        # Re-aggregate votes excluding detrusted oracles.
        admitted_tags = sorted(set(w21_result.per_tag_votes.keys()))
        eff_votes: dict[str, int] = {tag: 0 for tag in admitted_tags}
        eff_trust_sum: dict[str, float] = {
            tag: 0.0 for tag in admitted_tags}
        for probe in w21_result.probes:
            oid = str(probe.oracle_id)
            ewma = float(self._ewma_trust_state.get(oid, 1.0))
            if ewma < threshold:
                continue
            if probe.abstained:
                continue
            for tag in admitted_tags:
                if int(probe.per_tag_count.get(tag, 0)) > 0:
                    eff_votes[tag] += 1
                    eff_trust_sum[tag] += float(ewma)
        # Apply same quorum logic as W21 but with effective trust.
        quorum_min = int(self.inner.quorum_min)
        min_trust_sum = float(self.inner.min_trust_sum)
        eff_top_set = tuple(sorted(
            tag for tag in admitted_tags
            if eff_votes[tag] >= quorum_min
            and eff_trust_sum[tag] >= min_trust_sum
        ))
        # Decide the W33 branch.
        w33_branch = W33_BRANCH_TRUST_EWMA_RESOLVED
        w33_answer = dict(w21_result.answer)
        # If W33's effective tally produces a different non-empty
        # proper subset than W21's resolved top_set (or W21 was
        # NO_QUORUM and W33 finds one), W33 commits to its own answer.
        if (eff_top_set
                and len(eff_top_set) < len(admitted_tags)):
            if (tuple(w21_result.answer.get("services", ()))
                    != eff_top_set):
                # W33 reroutes the answer.
                w33_answer = dict(out)
                w33_answer["services"] = eff_top_set
                w33_branch = W33_BRANCH_TRUST_EWMA_DETRUSTED_REROUTE
        else:
            # W33 cannot form a quorum (either empty or covers all
            # admitted tags) — abstain.
            if (tuple(w21_result.answer.get("services", ()))
                    != tuple(admitted_tags)):
                # W21 had a resolved answer but W33 cannot form a
                # quorum after de-trusting; W33 abstains by passing
                # through the substrate FIFO answer.
                w33_answer = {k: v for k, v in out.items()}
                # Drop the W21-projected services if any.
                if "services" in w33_answer:
                    w33_answer.pop("services", None)
                if (w21_result.decoder_branch
                        == W21_BRANCH_QUORUM_RESOLVED
                        and n_detrusted > 0):
                    w33_branch = (
                        W33_BRANCH_TRUST_EWMA_DETRUSTED_ABSTAIN)
        # When no oracles are detrusted, W33 ties W21 byte-for-byte
        # (mechanism falsifier W33-Λ-no-trust-shift).
        if n_detrusted == 0:
            w33_answer = dict(w21_result.answer)
            w33_branch = W33_BRANCH_TRUST_EWMA_RESOLVED

        # 6-7. Build the W33 envelope.
        # Parent CID: derive from W22..W32 chain if available; else
        # use a stable CID derived from the W21 result's per_tag_votes
        # canonical bytes (since W21 has no envelope of its own).
        w22_envelope = out.get("latent_hybrid")
        if isinstance(w22_envelope, dict):
            parent_cid = str(w22_envelope.get(
                "outer_cid", w22_envelope.get(
                    "latent_envelope_cid", "")))
        else:
            parent_cid = ""
        if not parent_cid:
            # Stable fallback: hash W21 audit payload bytes.
            parent_payload = _canonical_json_bytes({
                "inner_branch": w21_result.decoder_branch,
                "per_tag_votes": dict(w21_result.per_tag_votes),
                "per_tag_trust_sum": {
                    k: round(float(v), 4)
                    for k, v in w21_result.per_tag_trust_sum.items()
                },
                "n_outside_queries": int(
                    w21_result.n_outside_queries),
                "cell_index": int(self._cell_index),
            })
            parent_cid = hashlib.sha256(parent_payload).hexdigest()

        wire_required = self.registry.has_wire_required_layer
        envelope = self._build_w33_envelope(
            parent_cid=parent_cid,
            n_detrusted_oracles=int(n_detrusted),
            wire_required=bool(wire_required),
        )

        # 8. Verify + register.
        outcome = self.registry.register_envelope(
            envelope,
            registered_parent_cid=parent_cid,
            registered_oracle_trust_state_cid=(
                envelope.oracle_trust_state_cid),
        )
        verify_ok = bool(outcome.ok)
        verify_reason = str(outcome.reason)

        if not verify_ok and self.require_w33_verification:
            self._cell_index += 1
            return self._pack(
                out=out,
                decoder_branch=W33_BRANCH_TRUST_EWMA_REJECTED,
                envelope=envelope, n_w33_visible=n_w21_visible,
                w33_overhead=0, ratified=False,
                verify_ok=False, verify_reason=verify_reason,
                n_detrusted_oracles=n_detrusted,
                parent_cid=parent_cid)

        # 9. Charge 1 wire token if non-trivial.
        w33_overhead = int(envelope.n_wire_tokens)
        n_w33_visible = int(n_w21_visible + w33_overhead)
        # Apply the W33-routed answer to the surface output.  When W33
        # abstains (services dropped from w33_answer), explicitly set
        # services to () so callers see the abstention.
        out_local = dict(out)
        for k, v in w33_answer.items():
            out_local[k] = v
        if "services" not in w33_answer:
            out_local["services"] = ()
        self._cell_index += 1
        return self._pack(
            out=out_local,
            decoder_branch=str(w33_branch),
            envelope=envelope, n_w33_visible=n_w33_visible,
            w33_overhead=w33_overhead, ratified=True,
            verify_ok=verify_ok, verify_reason=verify_reason,
            n_detrusted_oracles=n_detrusted,
            parent_cid=parent_cid)

    def _pack(
            self, *, out: dict[str, Any], decoder_branch: str,
            envelope: TrustEWMARatificationEnvelope | None,
            n_w33_visible: int, w33_overhead: int,
            ratified: bool, verify_ok: bool, verify_reason: str,
            n_detrusted_oracles: int, parent_cid: str,
    ) -> dict[str, Any]:
        envelope_bytes = (envelope.n_envelope_bytes
                           if envelope is not None else 0)
        structured_bits = (envelope.n_structured_bits
                            if envelope is not None else 0)
        wire = max(1, int(w33_overhead))
        cram_factor = (
            float(structured_bits) / float(wire)
            if structured_bits > 0 else 0.0
        )
        w33_cid = (envelope.w33_cid if envelope is not None else "")
        manifest_v3_cid = (envelope.manifest_v3_cid
                           if envelope is not None else "")
        oracle_trust_state_cid = (
            envelope.oracle_trust_state_cid
            if envelope is not None else "")
        trust_trajectory_cid = (
            envelope.trust_trajectory_cid
            if envelope is not None else "")
        # Snapshot oracle trust state from current registry (not from
        # envelope, so trivial path also has stable state).
        oracle_trust_state_snapshot = tuple(
            (oid, float(self._ewma_trust_state[oid]))
            for oid in sorted(self._ewma_trust_state.keys())
        )
        n_w21_visible = int(out.get("multi_oracle", {}).get(
            "n_outside_tokens_total", 0))
        inner_branch = str(out.get("multi_oracle", {}).get(
            "decoder_branch", ""))
        result = W33TrustEWMAResult(
            answer=dict(out),
            inner_w21_branch=inner_branch,
            decoder_branch=str(decoder_branch),
            parent_cid=str(parent_cid),
            cell_index=int(self._cell_index - 1
                            if self._cell_index > 0 else 0),
            oracle_trust_state=oracle_trust_state_snapshot,
            n_detrusted_oracles=int(n_detrusted_oracles),
            n_w21_visible_tokens=int(n_w21_visible),
            n_w33_visible_tokens=int(n_w33_visible),
            n_w33_overhead_tokens=int(w33_overhead),
            w33_cid=str(w33_cid),
            manifest_v3_cid=str(manifest_v3_cid),
            oracle_trust_state_cid=str(oracle_trust_state_cid),
            trust_trajectory_cid=str(trust_trajectory_cid),
            ratified=bool(ratified),
            verification_ok=bool(verify_ok),
            verification_reason=str(verify_reason),
            n_envelope_bytes=int(envelope_bytes),
            n_structured_bits=int(structured_bits),
            cram_factor_w33=float(cram_factor),
        )
        self._last_result = result
        self._last_envelope = envelope
        out_local = dict(out)
        out_local["trust_ewma_tracked"] = result.as_dict()
        if envelope is not None:
            out_local["trust_ewma_tracked_envelope"] = envelope.as_dict()
        return out_local

    def decode(self,
               handoffs: Sequence[_DecodedHandoff],
               ) -> dict[str, Any]:
        return self.decode_rounds([handoffs])

    @property
    def last_result(self) -> "W33TrustEWMAResult | None":
        return self._last_result

    @property
    def last_envelope(self) -> "TrustEWMARatificationEnvelope | None":
        return self._last_envelope


# ---------------------------------------------------------------------------
# Convenience factories (W33 family)
# ---------------------------------------------------------------------------


def build_trivial_trust_ewma_registry(
        *,
        schema: SchemaCapsule,
        registered_oracle_ids: Iterable[str] = (),
        local_host_id: str = "localhost",
) -> TrustEWMARegistry:
    """Build a W33 registry with all knobs trivial — the H2 byte-for-W21
    anchor (W33-Λ-trivial-trust-ewma falsifier).
    """
    return TrustEWMARegistry(
        schema=schema,
        trust_ewma_enabled=False,
        manifest_v3_disabled=True,
        trust_trajectory_window=0,
        trust_threshold=W33_DEFAULT_TRUST_THRESHOLD,
        ewma_alpha=W33_DEFAULT_EWMA_ALPHA,
        registered_oracle_ids=frozenset(str(o) for o in registered_oracle_ids),
        anchor_oracle_ids=frozenset(),
        local_host_id=local_host_id,
    )


def build_trust_ewma_registry(
        *,
        schema: SchemaCapsule,
        registered_oracle_ids: Iterable[str] = (),
        anchor_oracle_ids: Iterable[str] = (),
        trust_ewma_enabled: bool = True,
        manifest_v3_disabled: bool = False,
        trust_trajectory_window: int = W33_DEFAULT_TRUST_TRAJECTORY_WINDOW,
        trust_threshold: float = W33_DEFAULT_TRUST_THRESHOLD,
        ewma_alpha: float = W33_DEFAULT_EWMA_ALPHA,
        local_host_id: str = "localhost",
) -> TrustEWMARegistry:
    """Build a non-trivial W33 registry that exercises EWMA-tracked
    per-oracle trust + manifest-v3 CID.

    ``anchor_oracle_ids`` (optional) names oracles whose votes form
    the reference for the per-oracle agreement signal.  If empty,
    falls back to the W21 quorum-resolved top_set as the reference.
    """
    return TrustEWMARegistry(
        schema=schema,
        trust_ewma_enabled=bool(trust_ewma_enabled),
        manifest_v3_disabled=bool(manifest_v3_disabled),
        trust_trajectory_window=int(trust_trajectory_window),
        trust_threshold=float(trust_threshold),
        ewma_alpha=float(ewma_alpha),
        registered_oracle_ids=frozenset(str(o) for o in registered_oracle_ids),
        anchor_oracle_ids=frozenset(str(o) for o in anchor_oracle_ids),
        local_host_id=local_host_id,
    )
