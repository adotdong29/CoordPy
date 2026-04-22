"""Dynamic, bounded communication primitives for agent teams — the
Phase-35 substrate addition above ``core/role_handoff``.

This module adds *one* extra communication primitive — the
**EscalationThread** — a typed, bounded, explicitly-terminated
coordination object that sits strictly above the Phase-31 typed-
handoff layer and strictly below any unrestricted group-chat layer
(which the programme's thesis does NOT ship).

An EscalationThread is, concretely, a short-lived *temporary
coordination group* whose:

  * **membership** is declared at open time as a frozen set of
    roles — it can only shrink, never grow;
  * **issue_kind** is a short enumerated string naming the typed
    coordination question being resolved
    (e.g. ``RESOLVE_ROOT_CAUSE_CONFLICT``);
  * **candidate_claims** are an ordered tuple of (producer_role,
    claim_kind, payload) triples — the claims the thread exists
    to adjudicate;
  * **replies** are typed ``ThreadReply`` messages with a small
    enumerated ``reply_kind`` vocabulary
    (``INDEPENDENT_ROOT`` / ``DOWNSTREAM_SYMPTOM`` /
    ``UNCERTAIN`` / ``AGREE`` / ``DISAGREE`` / ``DEFER_TO``)
    and a bounded witness string;
  * **termination** is three-way: quorum-on-agree, max-round
    exhaustion, or explicit opener close;
  * **public interface** is a single ``ThreadResolution`` handoff
    emitted on close — non-member roles NEVER see thread-internal
    messages; they only ever see (if subscribed) the summary.

What this buys vs. static handoffs alone
----------------------------------------

Phase 31 proved that *static* typed handoffs + a role-subscription
table suffice whenever:

  (i) every load-bearing claim kind is producible by some role;
  (ii) the decoder can select the correct answer from the
       delivered bundle by a *fixed* priority rule.

Phase 35 identifies the smallest task family where condition (ii)
fails: **contested incidents** — scenarios where two or more
plausible root-cause claims are both in the auditor's inbox and the
correct answer depends on *producer-local* evidence that the static
priority cannot encode. Under a static decoder, one of the two
candidates is picked by author-defined priority; under an
escalation thread, the two producers are explicitly asked
"is your claim an isolated root cause or a downstream symptom?"
via typed replies; the resolution overrides priority.

Scope discipline (what this primitive is NOT)
---------------------------------------------

1. **Not a general group chat.** The thread has a fixed membership,
   a fixed max round count, a bounded witness-token budget per
   reply, and a single public summary output. Nothing leaks into
   non-member roles except the resolution.
2. **Not a consensus protocol.** Quorum here is a simple counting
   rule over typed replies, not Byzantine agreement. The hash-
   chained log is the audit trail; adversarial integrity still
   belongs to ``peer_review`` if required.
3. **Not a replacement for typed handoffs.** The thread's
   resolution is itself a typed handoff — it enters an inbox the
   same way ``DISK_FILL_CRITICAL`` does. The substrate stays
   hierarchical: threads produce handoffs, handoffs feed decoders.
4. **Not an adaptive subscription table.** A thread creates a
   *temporary* scope, not a permanent edit to the subscription
   graph. When a thread closes, the team's subscription table is
   unchanged.
5. **Not a new message bus.** Every thread event is logged in the
   existing ``HandoffLog`` with a distinct ``THREAD:*`` claim-kind
   prefix, so a single chain_hash verification covers handoffs and
   thread messages together.

Bounded-context invariant
-------------------------

Let a role ``r`` participate in ``T`` open threads per round, with
each thread carrying at most ``R_max`` replies of at most ``W``
tokens each. Then the peak active context at ``r`` per round
satisfies

    ctx(r) ≤ C_0 + R*·τ + T·R_max·W

which is still independent of the raw event stream size ``|X|``
(Theorem P35-2 in RESULTS_PHASE35.md). The Phase-31 bound
``C_0 + R*·τ`` is recovered exactly when ``T = 0``.

Files that consume this module
------------------------------

* ``vision_mvp/tasks/contested_incident.py`` — Phase-35 benchmark
  task family (contested root-cause scenarios).
* ``vision_mvp/experiments/phase35_contested_incident.py`` — driver.

Theoretical anchor: RESULTS_PHASE35.md § B.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, replace
from typing import Iterable, Sequence

from vision_mvp.core.role_handoff import (
    HandoffLog, HandoffRouter, RoleInbox, RoleSubscriptionTable,
    TypedHandoff,
)


# =============================================================================
# Typed reply and issue vocabularies
# =============================================================================


# Issue kinds — what a thread is adjudicating. Enumerated so
# membership + routing stay checkable.
THREAD_ISSUE_ROOT_CAUSE_CONFLICT = "RESOLVE_ROOT_CAUSE_CONFLICT"
THREAD_ISSUE_SEVERITY_CONFLICT = "RESOLVE_SEVERITY_CONFLICT"
THREAD_ISSUE_VERDICT_QUORUM = "RESOLVE_VERDICT_QUORUM"
THREAD_ISSUE_CONFIRM_CLAIM = "CONFIRM_CLAIM"

ALL_THREAD_ISSUES = (
    THREAD_ISSUE_ROOT_CAUSE_CONFLICT,
    THREAD_ISSUE_SEVERITY_CONFLICT,
    THREAD_ISSUE_VERDICT_QUORUM,
    THREAD_ISSUE_CONFIRM_CLAIM,
)


# Reply kinds — typed, enumerated responses. A role posting
# ``INDEPENDENT_ROOT`` is asserting "my claim is an isolated root
# cause on the witness in my payload". Every reply carries a
# bounded witness string so the resolution can cite it.
REPLY_INDEPENDENT_ROOT = "INDEPENDENT_ROOT"
REPLY_DOWNSTREAM_SYMPTOM = "DOWNSTREAM_SYMPTOM"
REPLY_UNCERTAIN = "UNCERTAIN"
REPLY_AGREE = "AGREE"
REPLY_DISAGREE = "DISAGREE"
REPLY_DEFER_TO = "DEFER_TO"

ALL_REPLY_KINDS = (
    REPLY_INDEPENDENT_ROOT, REPLY_DOWNSTREAM_SYMPTOM,
    REPLY_UNCERTAIN, REPLY_AGREE, REPLY_DISAGREE,
    REPLY_DEFER_TO,
)


# Resolution kinds — the single typed output a closed thread
# advertises. The auditor's decoder dispatches on this.
RESOLUTION_SINGLE_INDEPENDENT_ROOT = "SINGLE_INDEPENDENT_ROOT"
RESOLUTION_QUORUM_AGREE = "QUORUM_AGREE"
RESOLUTION_NO_CONSENSUS = "NO_CONSENSUS"
RESOLUTION_TIMEOUT = "TIMEOUT"
RESOLUTION_CONFLICT = "CONFLICT"

ALL_RESOLUTION_KINDS = (
    RESOLUTION_SINGLE_INDEPENDENT_ROOT, RESOLUTION_QUORUM_AGREE,
    RESOLUTION_NO_CONSENSUS, RESOLUTION_TIMEOUT,
    RESOLUTION_CONFLICT,
)


# Claim kind under which a thread's resolution enters downstream
# inboxes. The auditor's static subscription table subscribes this
# kind to the auditor role; the dynamic thread itself is invisible
# to every non-member.
CLAIM_THREAD_RESOLUTION = "THREAD_RESOLUTION"

# Internal claim kinds used for HandoffLog entries that record
# thread events. These are *logged for audit* but do NOT enter any
# static inbox — they live in the log only.
INTERNAL_CLAIM_THREAD_OPEN = "THREAD:OPEN"
INTERNAL_CLAIM_THREAD_REPLY = "THREAD:REPLY"
INTERNAL_CLAIM_THREAD_CLOSE = "THREAD:CLOSE"


# =============================================================================
# Content-address helper (mirror of role_handoff._cid)
# =============================================================================


def _cid(obj: object) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


# =============================================================================
# EscalationThread — the frozen descriptor
# =============================================================================


@dataclass(frozen=True)
class CandidateClaim:
    """One candidate claim referenced by a thread.

    Pointed at by ``ThreadReply.referenced_claim_idx``. Carries the
    producer role, claim kind, and a canonical payload; the payload
    is the exact witness the producer emitted, so a resolution can
    cite it.
    """

    producer_role: str
    claim_kind: str
    payload: str
    payload_cid: str


@dataclass(frozen=True)
class EscalationThread:
    """Frozen descriptor of an open-or-closed escalation thread.

    Fields:
      * ``thread_id``         — deterministic id (``T-`` + cid).
      * ``opener_role``       — role that opened the thread.
      * ``issue_kind``        — one of ``ALL_THREAD_ISSUES``.
      * ``members``           — frozenset of roles invited. The
        opener is always a member; non-member roles never see
        thread-internal messages.
      * ``candidate_claims``  — tuple of ``CandidateClaim``. The
        objects the thread is adjudicating.
      * ``max_rounds``        — hard round budget.
      * ``max_replies_per_member`` — per-member reply cap.
      * ``quorum``            — number of ``AGREE`` replies that
        triggers a quorum resolution.
      * ``witness_token_cap`` — max whitespace-split tokens per
        reply witness. Enforced at post-reply time.
      * ``opened_at_round``   — the substrate round at which the
        thread was opened (for sequencing).
    """

    thread_id: str
    opener_role: str
    issue_kind: str
    members: frozenset[str]
    candidate_claims: tuple[CandidateClaim, ...]
    max_rounds: int
    max_replies_per_member: int
    quorum: int
    witness_token_cap: int
    opened_at_round: int
    opened_at: float = 0.0

    def is_member(self, role: str) -> bool:
        return role in self.members

    def as_dict(self) -> dict:
        return {
            "thread_id": self.thread_id,
            "opener_role": self.opener_role,
            "issue_kind": self.issue_kind,
            "members": sorted(self.members),
            "candidate_claims": [
                {"producer_role": c.producer_role,
                 "claim_kind": c.claim_kind,
                 "payload": c.payload,
                 "payload_cid": c.payload_cid}
                for c in self.candidate_claims
            ],
            "max_rounds": self.max_rounds,
            "max_replies_per_member": self.max_replies_per_member,
            "quorum": self.quorum,
            "witness_token_cap": self.witness_token_cap,
            "opened_at_round": self.opened_at_round,
            "opened_at": round(self.opened_at, 3),
        }


@dataclass(frozen=True)
class ThreadReply:
    """A typed reply posted to a thread by a member role.

    Fields:
      * ``thread_id``            — parent thread.
      * ``replier_role``         — member role posting. Non-member
        posts are rejected (this is a specification bug, accounted
        at the router).
      * ``reply_kind``           — one of ``ALL_REPLY_KINDS``.
      * ``referenced_claim_idx`` — index into thread's
        ``candidate_claims``; the claim this reply is *about*.
      * ``witness``              — short witness string (bounded
        by ``witness_token_cap``).
      * ``round``                — emission round within the thread.
      * ``reply_cid``            — content address of the reply
        (``(thread_id, replier_role, reply_kind, claim_idx,
        witness, round)``).
    """

    thread_id: str
    replier_role: str
    reply_kind: str
    referenced_claim_idx: int
    witness: str
    round: int
    reply_cid: str

    @property
    def n_tokens(self) -> int:
        if not self.witness:
            return 0
        return max(1, len(self.witness.split()))

    def as_dict(self) -> dict:
        return {
            "thread_id": self.thread_id,
            "replier_role": self.replier_role,
            "reply_kind": self.reply_kind,
            "referenced_claim_idx": self.referenced_claim_idx,
            "witness": self.witness,
            "round": self.round,
            "reply_cid": self.reply_cid,
        }


@dataclass(frozen=True)
class ThreadResolution:
    """The single public summary emitted when a thread closes.

    Fields:
      * ``thread_id``            — parent thread.
      * ``resolution_kind``      — one of ``ALL_RESOLUTION_KINDS``.
      * ``resolved_claim_idx``   — index into ``candidate_claims``
        for the winning claim, or ``None`` if no resolution.
      * ``supporting_reply_cids`` — reply cids that witness the
        resolution (so auditor can consult the hash chain).
      * ``n_replies_total``      — total replies counted.
      * ``closed_at_round``      — round at close.
    """

    thread_id: str
    resolution_kind: str
    resolved_claim_idx: int | None
    supporting_reply_cids: tuple[str, ...]
    n_replies_total: int
    closed_at_round: int

    def as_dict(self) -> dict:
        return {
            "thread_id": self.thread_id,
            "resolution_kind": self.resolution_kind,
            "resolved_claim_idx": self.resolved_claim_idx,
            "supporting_reply_cids": list(self.supporting_reply_cids),
            "n_replies_total": self.n_replies_total,
            "closed_at_round": self.closed_at_round,
        }

    def as_payload_string(self,
                          thread: EscalationThread) -> str:
        """Canonical rendering of a resolution as a typed-handoff
        payload string. Used when the resolution is emitted into
        the auditor's inbox via the standard ``HandoffRouter``.

        Format:
          ``kind=<K> winner=<role/kind|none> losers=<r/k,r/k,...>
            thread=<tid> replies=<n>``

        ``losers`` is the comma-separated list of candidate
        (producer_role, claim_kind) tuples that the thread did NOT
        pick — so a downstream decoder can exclude their payloads
        from e.g. a services-union aggregation without needing to
        consult the thread state.
        """
        idx = self.resolved_claim_idx
        if idx is None or idx < 0 or idx >= len(thread.candidate_claims):
            winner = "none"
        else:
            cc = thread.candidate_claims[idx]
            winner = f"{cc.producer_role}/{cc.claim_kind}"
        losers: list[str] = []
        for (i, cc) in enumerate(thread.candidate_claims):
            if i == idx:
                continue
            losers.append(f"{cc.producer_role}/{cc.claim_kind}")
        losers_str = ",".join(losers) if losers else "none"
        return (f"kind={self.resolution_kind} winner={winner} "
                f"losers={losers_str} thread={self.thread_id} "
                f"replies={self.n_replies_total}")


# =============================================================================
# ThreadState — mutable state tracked in the registry
# =============================================================================


@dataclass
class ThreadState:
    """Mutable state for one open thread.

    Fields:
      * ``thread``      — the frozen descriptor.
      * ``replies``     — replies accepted so far (in arrival order).
      * ``closed``      — true once closed.
      * ``resolution``  — set iff closed; the single summary.
      * ``current_round`` — round counter used for reply ordering.
    """

    thread: EscalationThread
    replies: list[ThreadReply] = field(default_factory=list)
    closed: bool = False
    resolution: ThreadResolution | None = None
    current_round: int = 0

    def replies_by_member(self) -> dict[str, list[ThreadReply]]:
        out: dict[str, list[ThreadReply]] = {}
        for r in self.replies:
            out.setdefault(r.replier_role, []).append(r)
        return out

    def n_replies_by(self, role: str) -> int:
        return sum(1 for r in self.replies if r.replier_role == role)


# =============================================================================
# Exceptions
# =============================================================================


class DynamicCommError(Exception):
    """Raised for specification bugs in dynamic-comm usage.

    Specification bugs include: a non-member posting to a thread,
    referencing a claim index out of range, or closing a thread
    that was never opened. The router surfaces the more routine
    outcomes (reply over cap, thread already closed) through
    outcome strings rather than exceptions.
    """


# =============================================================================
# DynamicCommAccount — per-thread counters for the benchmark surface
# =============================================================================


@dataclass
class DynamicCommAccount:
    """Per-thread delivery counters.

    Each thread contributes one row; the summary is pooled over
    rows. The account is the benchmark's surface for 'how many
    thread messages did we spend, per scenario, to resolve a
    conflict?' — the Phase-35 headline.
    """

    _rows: list[dict] = field(default_factory=list)

    def record_thread(self, state: ThreadState) -> None:
        n_replies = len(state.replies)
        n_witness_tokens = sum(r.n_tokens for r in state.replies)
        res_kind = (state.resolution.resolution_kind
                    if state.resolution else "OPEN")
        resolved_idx = (state.resolution.resolved_claim_idx
                        if state.resolution else None)
        self._rows.append({
            "thread_id": state.thread.thread_id,
            "opener_role": state.thread.opener_role,
            "issue_kind": state.thread.issue_kind,
            "n_members": len(state.thread.members),
            "n_candidates": len(state.thread.candidate_claims),
            "n_replies": n_replies,
            "n_witness_tokens": n_witness_tokens,
            "resolution_kind": res_kind,
            "resolved_claim_idx": resolved_idx,
            "closed_at_round": (state.resolution.closed_at_round
                                 if state.resolution else None),
        })

    def summary(self) -> dict:
        if not self._rows:
            return {
                "n_threads": 0, "n_replies_total": 0,
                "n_witness_tokens_total": 0,
                "by_resolution_kind": {},
                "mean_members_per_thread": 0.0,
                "mean_replies_per_thread": 0.0,
            }
        by_res: dict[str, int] = {}
        n_replies = 0
        n_tokens = 0
        n_members = 0
        for r in self._rows:
            by_res[r["resolution_kind"]] = by_res.get(
                r["resolution_kind"], 0) + 1
            n_replies += r["n_replies"]
            n_tokens += r["n_witness_tokens"]
            n_members += r["n_members"]
        n = len(self._rows)
        return {
            "n_threads": n,
            "n_replies_total": n_replies,
            "n_witness_tokens_total": n_tokens,
            "by_resolution_kind": by_res,
            "mean_members_per_thread": round(n_members / n, 3),
            "mean_replies_per_thread": round(n_replies / n, 3),
        }

    def rows(self) -> list[dict]:
        return list(self._rows)


# =============================================================================
# ThreadRegistry + DynamicCommRouter — the wiring layer
# =============================================================================


@dataclass
class DynamicCommRouter:
    """Extends ``HandoffRouter`` with thread-scoped coordination.

    This is the Phase-35 orchestrator: it owns the static
    subscription table (inherited from Phase 31), an append-only
    handoff log, a registry of open threads, and a dynamic-comm
    account.

    Usage:

        base = HandoffRouter(
            subs=build_role_subscriptions(),
            log=HandoffLog(),
            inboxes={...},
        )
        router = DynamicCommRouter(base_router=base)
        # emit static handoffs exactly as before
        router.emit(...)
        # open an escalation thread
        thread = router.open_thread(
            opener_role="auditor",
            issue_kind=THREAD_ISSUE_ROOT_CAUSE_CONFLICT,
            members=frozenset({"auditor", "db_admin", "sysadmin"}),
            candidate_claims=(...,),
            max_rounds=2, quorum=1,
            max_replies_per_member=1, witness_token_cap=12,
        )
        # members post typed replies
        router.post_reply(thread.thread_id,
                          replier_role="db_admin",
                          reply_kind=REPLY_INDEPENDENT_ROOT,
                          referenced_claim_idx=1,
                          witness="deadlock relation=orders_payments")
        # close; emits ThreadResolution handoff to the auditor's
        # inbox under the usual static subscription table.
        resolution = router.close_thread(thread.thread_id)
    """

    base_router: HandoffRouter
    threads: dict[str, ThreadState] = field(default_factory=dict)
    account: DynamicCommAccount = field(default_factory=DynamicCommAccount)

    # ------- delegation (Phase 31 surface) -------

    def emit(self, source_role: str, source_agent_id: int,
             claim_kind: str, payload: str,
             source_event_ids: Sequence[int] = (),
             round: int = 0,
             ) -> tuple[TypedHandoff, dict[str, str]]:
        return self.base_router.emit(
            source_role=source_role, source_agent_id=source_agent_id,
            claim_kind=claim_kind, payload=payload,
            source_event_ids=source_event_ids, round=round,
        )

    def register_inbox(self, inbox: RoleInbox) -> None:
        self.base_router.register_inbox(inbox)

    @property
    def subs(self) -> RoleSubscriptionTable:
        return self.base_router.subs

    @property
    def log(self) -> HandoffLog:
        return self.base_router.log

    @property
    def inboxes(self) -> dict[str, RoleInbox]:
        return self.base_router.inboxes

    def log_length(self) -> int:
        return self.base_router.log_length()

    def verify(self) -> bool:
        return self.base_router.verify()

    # ------- thread ops (Phase 35 surface) -------

    def open_thread(self,
                    opener_role: str,
                    issue_kind: str,
                    members: Iterable[str],
                    candidate_claims: Sequence[tuple[str, str, str]],
                    max_rounds: int = 2,
                    max_replies_per_member: int = 1,
                    quorum: int = 1,
                    witness_token_cap: int = 16,
                    round: int = 0,
                    ) -> EscalationThread:
        """Open a new escalation thread.

        ``candidate_claims`` is a sequence of ``(producer_role,
        claim_kind, payload)`` triples. The thread is deterministic
        in (issue_kind, opener_role, sorted(members), candidate
        cids); opening an identical thread twice returns the same
        ``thread_id``.

        The opener is force-added to ``members``.

        A ``THREAD:OPEN`` internal handoff is written to the log
        for audit. No inbox receives this entry (the log is write-
        through-only; the static subscription table does not route
        ``THREAD:*`` claim kinds to any role).
        """
        if issue_kind not in ALL_THREAD_ISSUES:
            raise DynamicCommError(
                f"unknown thread issue_kind {issue_kind!r}")
        if max_rounds < 1:
            raise DynamicCommError("max_rounds must be ≥ 1")
        if max_replies_per_member < 1:
            raise DynamicCommError(
                "max_replies_per_member must be ≥ 1")
        if witness_token_cap < 1:
            raise DynamicCommError("witness_token_cap must be ≥ 1")
        mem = frozenset(members) | {opener_role}
        if quorum < 1 or quorum > len(mem):
            raise DynamicCommError(
                f"quorum {quorum} out of range for "
                f"{len(mem)} members")
        if not candidate_claims:
            raise DynamicCommError(
                "a thread needs at least one candidate claim")
        built_candidates: list[CandidateClaim] = []
        for (prod, kind, payload) in candidate_claims:
            cid = _cid({"p": str(payload), "role": prod, "k": kind})
            built_candidates.append(CandidateClaim(
                producer_role=str(prod), claim_kind=str(kind),
                payload=str(payload), payload_cid=cid))
        cand_tuple = tuple(built_candidates)
        tid_seed = {
            "issue": issue_kind, "opener": opener_role,
            "members": sorted(mem),
            "cands": [c.payload_cid for c in cand_tuple],
        }
        tid = "T-" + _cid(tid_seed)[:16]
        if tid in self.threads:
            # Idempotent re-open returns the existing descriptor;
            # state is preserved (including replies). This is what
            # lets a well-behaved decoder retry thread open without
            # duplicating state.
            return self.threads[tid].thread
        thread = EscalationThread(
            thread_id=tid, opener_role=opener_role,
            issue_kind=issue_kind, members=mem,
            candidate_claims=cand_tuple,
            max_rounds=max_rounds,
            max_replies_per_member=max_replies_per_member,
            quorum=quorum,
            witness_token_cap=witness_token_cap,
            opened_at_round=round,
            opened_at=time.time(),
        )
        state = ThreadState(thread=thread, current_round=round)
        self.threads[tid] = state
        # Log an internal audit record. This goes through the same
        # hash-chained HandoffLog so an auditor can walk the whole
        # provenance surface uniformly.
        self.base_router.log.emit(
            source_role=opener_role, source_agent_id=-1,
            to_role="__thread__",
            claim_kind=INTERNAL_CLAIM_THREAD_OPEN,
            payload=(f"open tid={tid} issue={issue_kind} "
                     f"members={','.join(sorted(mem))} "
                     f"cands={len(cand_tuple)}"),
            source_event_ids=(), round=round,
        )
        return thread

    def post_reply(self,
                   thread_id: str,
                   replier_role: str,
                   reply_kind: str,
                   referenced_claim_idx: int,
                   witness: str,
                   round: int = 0,
                   ) -> str:
        """Post a typed reply to an open thread.

        Returns one of the outcome strings:

          * ``"accepted"``      — reply added to state.
          * ``"closed"``        — thread is already closed.
          * ``"over_cap"``      — member already posted the max
            number of replies.
          * ``"rounds_exhausted"`` — thread's max_rounds budget
            hit (replies past the last round are refused).

        Raises ``DynamicCommError`` for specification bugs
        (non-member reply, bad reply_kind, bad claim idx, unknown
        thread_id).
        """
        if thread_id not in self.threads:
            raise DynamicCommError(
                f"unknown thread_id {thread_id!r}")
        state = self.threads[thread_id]
        if reply_kind not in ALL_REPLY_KINDS:
            raise DynamicCommError(
                f"unknown reply_kind {reply_kind!r}")
        if not state.thread.is_member(replier_role):
            raise DynamicCommError(
                f"role {replier_role!r} is not a member of "
                f"thread {thread_id}")
        if referenced_claim_idx < 0 or referenced_claim_idx >= len(
                state.thread.candidate_claims):
            raise DynamicCommError(
                f"claim idx {referenced_claim_idx} out of range "
                f"for thread {thread_id}")
        if state.closed:
            return "closed"
        if state.n_replies_by(replier_role) >= \
                state.thread.max_replies_per_member:
            return "over_cap"
        if round >= state.thread.opened_at_round + \
                state.thread.max_rounds:
            return "rounds_exhausted"
        # Clamp witness token count to the cap.
        toks = witness.split()
        if len(toks) > state.thread.witness_token_cap:
            witness = " ".join(toks[:state.thread.witness_token_cap])
        reply_cid = _cid({
            "t": thread_id, "r": replier_role, "k": reply_kind,
            "ci": referenced_claim_idx, "w": witness, "rnd": round,
        })
        reply = ThreadReply(
            thread_id=thread_id, replier_role=replier_role,
            reply_kind=reply_kind,
            referenced_claim_idx=referenced_claim_idx,
            witness=witness, round=round, reply_cid=reply_cid,
        )
        state.replies.append(reply)
        state.current_round = max(state.current_round, round)
        # Log an internal audit record.
        self.base_router.log.emit(
            source_role=replier_role, source_agent_id=-1,
            to_role="__thread__",
            claim_kind=INTERNAL_CLAIM_THREAD_REPLY,
            payload=(f"reply tid={thread_id} kind={reply_kind} "
                     f"ci={referenced_claim_idx} w={witness}"),
            source_event_ids=(), round=round,
        )
        return "accepted"

    def close_thread(self,
                     thread_id: str,
                     round: int | None = None,
                     ) -> ThreadResolution:
        """Close an open thread and compute its resolution.

        Resolution rules (deterministic):

          1. **SINGLE_INDEPENDENT_ROOT** — exactly one
             ``INDEPENDENT_ROOT`` reply across all members, and
             no non-``UNCERTAIN`` disagreement: the referenced
             claim wins.
          2. **QUORUM_AGREE** — ``quorum`` or more members
             returned ``AGREE`` on the same claim_idx: that claim
             wins.
          3. **CONFLICT** — at least two ``INDEPENDENT_ROOT`` on
             different claim indices, OR at least one
             ``INDEPENDENT_ROOT`` and one ``DISAGREE`` on the
             same idx. No resolution.
          4. **NO_CONSENSUS** — only ``UNCERTAIN`` /
             ``DEFER_TO`` replies. No resolution.
          5. **TIMEOUT** — closed with zero replies *and*
             max_rounds exhausted.

        Emits the ``ThreadResolution`` as a regular typed handoff
        (``claim_kind = CLAIM_THREAD_RESOLUTION``, source_role =
        opener_role) via the base router — so standard static
        subscription to ``CLAIM_THREAD_RESOLUTION`` delivers the
        resolution to the auditor's inbox.

        The resolution handoff's ``source_event_ids`` is left
        empty; provenance lives in the ``supporting_reply_cids``
        field and in the hash-chained log's ``THREAD:*`` entries.
        """
        if thread_id not in self.threads:
            raise DynamicCommError(
                f"unknown thread_id {thread_id!r}")
        state = self.threads[thread_id]
        if state.closed and state.resolution is not None:
            return state.resolution
        closed_round = round if round is not None else state.current_round
        # ---- compute resolution ----
        ir_by_idx: dict[int, list[str]] = {}
        dis_by_idx: dict[int, list[str]] = {}
        agree_by_idx: dict[int, list[str]] = {}
        support_cids_by_idx: dict[int, list[str]] = {}
        for r in state.replies:
            if r.reply_kind == REPLY_INDEPENDENT_ROOT:
                ir_by_idx.setdefault(
                    r.referenced_claim_idx, []).append(r.replier_role)
                support_cids_by_idx.setdefault(
                    r.referenced_claim_idx, []).append(r.reply_cid)
            elif r.reply_kind == REPLY_DISAGREE:
                dis_by_idx.setdefault(
                    r.referenced_claim_idx, []).append(r.replier_role)
                support_cids_by_idx.setdefault(
                    r.referenced_claim_idx, []).append(r.reply_cid)
            elif r.reply_kind == REPLY_AGREE:
                agree_by_idx.setdefault(
                    r.referenced_claim_idx, []).append(r.replier_role)
                support_cids_by_idx.setdefault(
                    r.referenced_claim_idx, []).append(r.reply_cid)
        resolved_idx: int | None = None
        res_kind: str
        support_cids: tuple[str, ...] = ()
        if not state.replies:
            # zero replies — either the thread was never used, or
            # it timed out
            if (state.current_round >= state.thread.opened_at_round
                    + state.thread.max_rounds - 1
                    or closed_round >= state.thread.opened_at_round
                    + state.thread.max_rounds - 1):
                res_kind = RESOLUTION_TIMEOUT
            else:
                res_kind = RESOLUTION_NO_CONSENSUS
        elif len(ir_by_idx) == 1:
            only_idx = next(iter(ir_by_idx.keys()))
            if dis_by_idx.get(only_idx):
                res_kind = RESOLUTION_CONFLICT
            else:
                res_kind = RESOLUTION_SINGLE_INDEPENDENT_ROOT
                resolved_idx = only_idx
                support_cids = tuple(support_cids_by_idx.get(only_idx, []))
        elif len(ir_by_idx) >= 2:
            res_kind = RESOLUTION_CONFLICT
        else:
            # 0 IR replies; check for quorum on AGREE
            best_idx: int | None = None
            best_count = 0
            for idx, roles in agree_by_idx.items():
                if len(roles) > best_count:
                    best_count = len(roles)
                    best_idx = idx
            if best_idx is not None and best_count >= state.thread.quorum:
                res_kind = RESOLUTION_QUORUM_AGREE
                resolved_idx = best_idx
                support_cids = tuple(
                    support_cids_by_idx.get(best_idx, []))
            else:
                res_kind = RESOLUTION_NO_CONSENSUS
        resolution = ThreadResolution(
            thread_id=thread_id, resolution_kind=res_kind,
            resolved_claim_idx=resolved_idx,
            supporting_reply_cids=support_cids,
            n_replies_total=len(state.replies),
            closed_at_round=closed_round,
        )
        state.closed = True
        state.resolution = resolution
        # ---- log the close ----
        self.base_router.log.emit(
            source_role=state.thread.opener_role,
            source_agent_id=-1,
            to_role="__thread__",
            claim_kind=INTERNAL_CLAIM_THREAD_CLOSE,
            payload=(f"close tid={thread_id} kind={res_kind} "
                     f"idx={resolved_idx} replies={len(state.replies)}"),
            source_event_ids=(),
            round=closed_round,
        )
        # ---- emit the single public resolution handoff ----
        # The resolution's source_role is the *opener* role; the
        # static subscription table routes CLAIM_THREAD_RESOLUTION
        # from the opener to whatever consumer(s) are declared.
        self.base_router.emit(
            source_role=state.thread.opener_role,
            source_agent_id=-1,
            claim_kind=CLAIM_THREAD_RESOLUTION,
            payload=resolution.as_payload_string(state.thread),
            source_event_ids=(), round=closed_round,
        )
        self.account.record_thread(state)
        return resolution

    # ------- introspection -------

    def get_thread(self, thread_id: str) -> EscalationThread:
        if thread_id not in self.threads:
            raise DynamicCommError(
                f"unknown thread_id {thread_id!r}")
        return self.threads[thread_id].thread

    def get_state(self, thread_id: str) -> ThreadState:
        if thread_id not in self.threads:
            raise DynamicCommError(
                f"unknown thread_id {thread_id!r}")
        return self.threads[thread_id]

    def get_resolution(self, thread_id: str
                       ) -> ThreadResolution | None:
        state = self.threads.get(thread_id)
        if state is None:
            return None
        return state.resolution

    def iter_closed(self) -> Iterable[ThreadState]:
        for s in self.threads.values():
            if s.closed:
                yield s

    def summary(self) -> dict:
        return {
            "log_length": self.log_length(),
            "chain_ok": self.verify(),
            "threads": self.account.summary(),
            "inboxes": {r: box.stats()
                        for r, box in sorted(self.inboxes.items())},
            "n_threads_open": sum(
                1 for s in self.threads.values() if not s.closed),
            "n_threads_total": len(self.threads),
        }


# =============================================================================
# Helpers exposed to callers
# =============================================================================


def build_resolution_subscriptions(
        subs: RoleSubscriptionTable,
        opener_roles: Iterable[str],
        consumer_roles: Iterable[str],
        ) -> RoleSubscriptionTable:
    """Register ``CLAIM_THREAD_RESOLUTION`` on ``subs`` for every
    ``(opener, consumer)`` pair, so thread resolutions routed
    through the base router's typed delivery end up in the
    consumer role's standard inbox.

    Returns ``subs`` for chaining.
    """
    for opener in opener_roles:
        subs.subscribe(opener, CLAIM_THREAD_RESOLUTION,
                        list(consumer_roles))
    return subs


def parse_resolution_payload(payload: str) -> dict:
    """Parse a ``ThreadResolution.as_payload_string`` back into a
    lightweight dict. Used by a decoder that receives the
    resolution handoff through the inbox and needs to dispatch on
    ``resolution_kind``.
    """
    out: dict[str, str] = {}
    for tok in payload.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k] = v
    return out


# =============================================================================
# Constants re-exported for downstream callers
# =============================================================================


__all__ = [
    "CandidateClaim", "EscalationThread", "ThreadReply",
    "ThreadResolution", "ThreadState", "DynamicCommAccount",
    "DynamicCommRouter", "DynamicCommError",
    "build_resolution_subscriptions", "parse_resolution_payload",
    # issue kinds
    "THREAD_ISSUE_ROOT_CAUSE_CONFLICT",
    "THREAD_ISSUE_SEVERITY_CONFLICT",
    "THREAD_ISSUE_VERDICT_QUORUM",
    "THREAD_ISSUE_CONFIRM_CLAIM",
    "ALL_THREAD_ISSUES",
    # reply kinds
    "REPLY_INDEPENDENT_ROOT", "REPLY_DOWNSTREAM_SYMPTOM",
    "REPLY_UNCERTAIN", "REPLY_AGREE", "REPLY_DISAGREE",
    "REPLY_DEFER_TO", "ALL_REPLY_KINDS",
    # resolution kinds
    "RESOLUTION_SINGLE_INDEPENDENT_ROOT",
    "RESOLUTION_QUORUM_AGREE", "RESOLUTION_NO_CONSENSUS",
    "RESOLUTION_TIMEOUT", "RESOLUTION_CONFLICT",
    "ALL_RESOLUTION_KINDS",
    # claim kind
    "CLAIM_THREAD_RESOLUTION",
    "INTERNAL_CLAIM_THREAD_OPEN",
    "INTERNAL_CLAIM_THREAD_REPLY",
    "INTERNAL_CLAIM_THREAD_CLOSE",
]
