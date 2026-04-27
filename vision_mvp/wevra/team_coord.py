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
import time
from typing import Any, Callable, Iterable, Protocol, Sequence

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
]
