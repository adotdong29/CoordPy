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
ALL_PRODUCER_PROMPT_MODES: tuple[str, ...] = (
    PRODUCER_PROMPT_NAIVE,
    PRODUCER_PROMPT_STRUCTURED,
)


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


# Default schemas for the bundled incident-triage benchmark family.
# Other benchmarks should construct their own ``RoleExtractionSchema``
# table; this one ships as a convenience for Phase-58..Phase-61
# drivers and is mechanically aligned with
# ``vision_mvp.core.extractor_noise.incident_triage_known_kinds``.
INCIDENT_TRIAGE_OBSERVATION_KINDS: tuple[str, ...] = (
    "LATENCY_SPIKE", "ERROR_RATE_SPIKE", "FW_BLOCK_SURGE",
)


def incident_triage_role_schemas() -> dict[str, RoleExtractionSchema]:
    """Return the W14 default schema table for the incident-triage
    benchmark family. Each role's ``observation_kinds`` is the
    intersection of its ``allowed_kinds`` with the closed-vocabulary
    generic-noise tier (``INCIDENT_TRIAGE_OBSERVATION_KINDS``);
    ``diagnosis_kinds`` is the complement on the same allowed-kind
    set.

    Mechanically verified by ``IncidentTriageSchemaTests`` in
    ``test_wevra_producer_ambiguity.py``.
    """
    from vision_mvp.core.extractor_noise import (
        incident_triage_known_kinds)
    out: dict[str, RoleExtractionSchema] = {}
    obs_set = set(INCIDENT_TRIAGE_OBSERVATION_KINDS)
    for role, allowed in incident_triage_known_kinds().items():
        allowed_t = tuple(allowed)
        observation = tuple(k for k in allowed_t if k in obs_set)
        diagnosis = tuple(k for k in allowed_t if k not in obs_set)
        out[role] = RoleExtractionSchema(
            role=role, allowed_kinds=allowed_t,
            observation_kinds=observation,
            diagnosis_kinds=diagnosis)
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
    # SDK v3.16 — attention-aware capsule context packing (W15 family).
    "W15_DEFAULT_TIER_WEIGHT", "W15_DEFAULT_CCK_WEIGHT",
    "W15_DEFAULT_CORROBORATION_WEIGHT", "W15_DEFAULT_MAGNITUDE_WEIGHT",
    "W15_DEFAULT_ROUND_WEIGHT",
    "W15PackedHandoff", "W15PackResult",
    "FifoContextPacker", "CapsuleContextPacker",
    "AttentionAwareBundleDecoder",
]
