"""Role-scoped typed handoffs — a content-channel substrate for agent
teams.

This module is the Phase-31 substrate improvement. It sits *between* the
routing layer (``sparse_router`` / ``agent_keys``) and the content layer
(``context_ledger``, ``task_board``): it is a small, explicit, typed
channel for one role to hand off a *named claim* to another role,
carrying the provenance (source event ids, content-address of the
payload) that would otherwise have to be recovered from the event log
or from the Merkle DAG.

The design is deliberately narrow: this is not a replacement for the
message bus, the router, or the task board. It is the primitive the
Phase-29 / Phase-30 analyses were implicitly asking for — a channel
whose *header* is semantically typed (claim kind), whose *body* is an
exact content-addressed handle (not a summary), and whose *delivery
table* is per-role-pair (not per-agent).

Why this exists (and why it is "agent-teams" rather than "graph
tool")

  Phase 29 showed: role-keyed *type-level* Bloom routing does not help
  the aggregator role — its concern is *content-level*. The natural
  next move for a general-agent-team substrate is to lift the
  load-bearing content signal from the payload into the header, so
  that downstream roles can subscribe by *claim* rather than by raw
  event type. This is how real operational teams already work: an SRE
  does not forward all of its telemetry to the auditor; it forwards a
  claim (``SLOW_QUERY_OBSERVED``) with the witnessing rows attached.

  The module lives on the *team communication* axis (§ 1.5 of the
  master plan), not on the corpus-representation axis. It does NOT
  rebuild an index of a corpus, NOT claim anything about graph
  traversal, NOT assume a fixed single-assistant reader. Its only
  claim is:

    Under a role-subscription table that covers the task's causal
    dependency graph, typed handoffs between roles preserve answer
    correctness while bounding every role's delivered-token count
    by a quantity independent of the raw event-stream size
    (Theorems P31-1, P31-3 in RESULTS_PHASE31.md).

What this module provides

  * ``TypedHandoff`` — immutable, content-addressable, provenance-
    carrying handoff message between two roles.
  * ``RoleSubscriptionTable`` — (source_role, claim_kind) → set of
    consumer roles. The programme's equivalent of "who should know
    what, when."
  * ``RoleInbox`` — per-role bounded inbox with claim-kind filtering
    and dedup-by-(source_role, claim_kind, payload_cid).
  * ``HandoffLog`` — append-only log of every handoff, hash-chained
    (so a downstream consumer can prove a handoff was or was not
    delivered by the expected source role, at the expected round).
  * ``DeliveryAccount`` — per-(source_role, to_role, claim_kind)
    counters for delivered / dropped / deduped handoffs and
    delivered-token totals.

Scope discipline

  * This is NOT a replacement for ``sparse_router``: the router still
    decides per-agent delivery inside a role; this module decides
    *cross-role* delivery by claim kind.
  * This is NOT a new message format: a ``TypedHandoff`` is designed
    to be stored as a single entry in ``MerkleDAG`` and referenced
    elsewhere by ``cid``.
  * This is NOT a security boundary: the hash chain detects accidental
    log truncation and reorder; authenticated provenance is still the
    job of ``peer_review`` (Ed25519).

Theoretical anchor: RESULTS_PHASE31.md (Theorems P31-1 .. P31-5 +
Conjectures C31-6, C31-7).
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, replace
from typing import Iterable, Sequence


# =============================================================================
# Content-address helper (duplicated from merkle_dag to avoid circular import)
# =============================================================================


def _cid(obj: object) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


# =============================================================================
# TypedHandoff
# =============================================================================


@dataclass(frozen=True)
class TypedHandoff:
    """One typed handoff from one role to one role.

    Fields:
      * ``handoff_id``          — monotone integer, assigned at emission.
      * ``source_role``         — role that produced the claim.
      * ``source_agent_id``     — agent inside the source role (for logs).
      * ``to_role``             — intended consumer role.
      * ``claim_kind``          — short enumerated string naming the
        semantic type of the claim, e.g. ``SLOW_QUERY_OBSERVED``,
        ``DISK_FILL_CRITICAL``. Subscription and routing happen at
        this granularity. Short + enumerated so the claim_kind is
        itself cheap to put in a Bloom filter.
      * ``payload``             — short canonical string carrying the
        claim's content. May be a comma-separated list of facts,
        a JSON-stringified dict, or a sentence — grader-agnostic.
      * ``source_event_ids``    — the ids of the underlying raw events
        on which the claim was produced. Not required, but carrying
        them makes the handoff verifiable.
      * ``round``               — emission round (for sequencing).
      * ``payload_cid``         — SHA-256 of the canonicalised payload.
      * ``prev_chain_hash``     — chain hash from the previous handoff
        in ``HandoffLog`` — this is the Merkle-style link that lets
        a consumer prove the handoff was not retroactively injected
        into the sequence.
      * ``chain_hash``          — SHA-256 over (prev_chain_hash +
        canonical self-fields). Always computed.

    A ``TypedHandoff`` is deliberately *hashable and frozen*: two
    identical handoffs collapse by content-address, which is what
    ``RoleInbox``'s dedup relies on.
    """

    handoff_id: int
    source_role: str
    source_agent_id: int
    to_role: str
    claim_kind: str
    payload: str
    source_event_ids: tuple[int, ...]
    round: int
    payload_cid: str
    prev_chain_hash: str
    chain_hash: str
    emitted_at: float = 0.0

    @property
    def n_tokens(self) -> int:
        """Whitespace-split token proxy of ``payload``."""
        if not self.payload:
            return 0
        return max(1, len(self.payload.split()))

    def as_dict(self) -> dict:
        return {
            "handoff_id": self.handoff_id,
            "source_role": self.source_role,
            "source_agent_id": self.source_agent_id,
            "to_role": self.to_role,
            "claim_kind": self.claim_kind,
            "payload": self.payload,
            "source_event_ids": list(self.source_event_ids),
            "round": self.round,
            "payload_cid": self.payload_cid,
            "prev_chain_hash": self.prev_chain_hash,
            "chain_hash": self.chain_hash,
            "emitted_at": round(self.emitted_at, 3),
        }


def _compute_chain_hash(prev_chain_hash: str,
                        source_role: str,
                        source_agent_id: int,
                        to_role: str,
                        claim_kind: str,
                        payload_cid: str,
                        source_event_ids: Sequence[int],
                        round: int,
                        handoff_id: int,
                        ) -> str:
    payload = {
        "prev": prev_chain_hash, "src_role": source_role,
        "src_agent": source_agent_id, "to_role": to_role,
        "kind": claim_kind, "cid": payload_cid,
        "evs": list(source_event_ids), "round": round,
        "hid": handoff_id,
    }
    return _cid(payload)


# =============================================================================
# Role subscription table
# =============================================================================


@dataclass
class RoleSubscriptionTable:
    """(source_role, claim_kind) → frozenset of consumer roles.

    The subscription table is the explicit "who should know what"
    declaration for the team. It is the analogue of ``ROLE_SUBSCRIPTIONS``
    in ``task_scale_swe`` but one layer up — it routes *claims*, not
    raw event types.

    In the Phase-31 benchmark, the table is loaded from
    ``incident_triage.build_role_subscriptions``; external callers
    construct the table explicitly from a task decomposition.

    A missing entry means "no consumers subscribe to this (source,
    claim) pair" — the router silently drops the handoff and records
    it on ``DeliveryAccount`` under ``n_dropped_no_subscriber``. This
    is the correct failure mode: an undeclared claim is a
    specification bug, not a runtime exception.
    """

    _table: dict[tuple[str, str], frozenset[str]] = field(
        default_factory=dict)

    def subscribe(self, source_role: str, claim_kind: str,
                  consumer_roles: Iterable[str]) -> None:
        key = (source_role, claim_kind)
        existing = self._table.get(key, frozenset())
        self._table[key] = existing | frozenset(consumer_roles)

    def consumers(self, source_role: str, claim_kind: str) -> frozenset[str]:
        return self._table.get((source_role, claim_kind), frozenset())

    def all_claim_kinds(self) -> frozenset[str]:
        return frozenset(k for (_, k) in self._table.keys())

    def all_pairs(self) -> list[tuple[str, str]]:
        return list(self._table.keys())

    def as_dict(self) -> dict[str, list[str]]:
        return {
            f"{s}:{k}": sorted(self._table[(s, k)])
            for (s, k) in sorted(self._table.keys())
        }


# =============================================================================
# RoleInbox
# =============================================================================


@dataclass
class RoleInbox:
    """One role's bounded, typed inbox.

    Accepts only handoffs whose ``to_role`` matches; dedups on
    ``(source_role, claim_kind, payload_cid)``; enforces a capacity
    bound. Overflow is accounted on the ``DeliveryAccount``.

    The ``RoleInbox`` is the per-role analogue of
    ``NetworkAgent._inbox`` in ``agent_network`` — the key
    differences being:

      * it is role-keyed, not agent-keyed;
      * it refuses handoffs whose ``to_role`` doesn't match (the
        type-error is caught here, not silently by a broken router);
      * it dedups by content-address, so two agents producing the
        same claim on the same evidence count once in the inbox but
        are still both in the handoff log for provenance.
    """

    role: str
    capacity: int = 64

    _items: list[TypedHandoff] = field(default_factory=list)
    _seen_cids: set[tuple[str, str, str]] = field(default_factory=set)
    _n_overflow: int = 0
    _n_dedup: int = 0
    _n_wrong_role: int = 0

    def offer(self, h: TypedHandoff) -> str:
        """Try to accept ``h``. Returns one of the string outcomes:

          * ``"accepted"``      — added to inbox.
          * ``"deduped"``       — identical (source_role, claim_kind,
                                   payload_cid) already seen.
          * ``"overflow"``      — capacity exceeded; handoff dropped.
          * ``"wrong_role"``    — ``h.to_role != self.role``.

        This is the only way handoffs enter the inbox.
        """
        if h.to_role != self.role:
            self._n_wrong_role += 1
            return "wrong_role"
        key = (h.source_role, h.claim_kind, h.payload_cid)
        if key in self._seen_cids:
            self._n_dedup += 1
            return "deduped"
        if len(self._items) >= self.capacity:
            self._n_overflow += 1
            return "overflow"
        self._items.append(h)
        self._seen_cids.add(key)
        return "accepted"

    def drain(self) -> list[TypedHandoff]:
        """Return all handoffs held and clear. Dedup memory is kept
        across drains so repeated offers stay deduped."""
        out = list(self._items)
        self._items.clear()
        return out

    def peek(self) -> tuple[TypedHandoff, ...]:
        return tuple(self._items)

    def stats(self) -> dict:
        return {
            "role": self.role,
            "capacity": self.capacity,
            "n_held": len(self._items),
            "n_overflow": self._n_overflow,
            "n_dedup": self._n_dedup,
            "n_wrong_role": self._n_wrong_role,
        }

    @property
    def n_held(self) -> int:
        return len(self._items)

    @property
    def n_overflow(self) -> int:
        return self._n_overflow

    @property
    def n_dedup(self) -> int:
        return self._n_dedup


# =============================================================================
# HandoffLog — append-only chained log
# =============================================================================


@dataclass
class HandoffLog:
    """Append-only hash-chained log of every emitted handoff.

    A consumer role can inspect this log to prove (to itself, or to
    an auditor) that a handoff came from the declared source role at
    the declared round and that no later handoff retroactively
    inserted into the past. This is the provenance surface the
    Phase-31 benchmark's failure-attribution analysis consumes
    (``missing_handoff`` vs ``llm_error`` vs ``retrieval_miss`` —
    see ``incident_triage.attribute_failure``).

    Chain semantics:
      * ``_chain_hash`` is the SHA-256 of the most recent handoff's
        chain_hash.
      * Each new handoff's ``chain_hash`` is computed over
        ``(prev_chain_hash, source_role, source_agent_id, to_role,
        claim_kind, payload_cid, source_event_ids, round, handoff_id)``
        so tampering detectably breaks the chain.
    """

    _entries: list[TypedHandoff] = field(default_factory=list)
    _next_id: int = 0
    _chain_head: str = "GENESIS"

    def emit(self, source_role: str, source_agent_id: int,
             to_role: str, claim_kind: str, payload: str,
             source_event_ids: Sequence[int],
             round: int = 0) -> TypedHandoff:
        """Build a ``TypedHandoff`` and append to the log."""
        payload_cid = _cid({"p": payload, "evs": list(source_event_ids)})
        hid = self._next_id
        chain_hash = _compute_chain_hash(
            prev_chain_hash=self._chain_head,
            source_role=source_role, source_agent_id=source_agent_id,
            to_role=to_role, claim_kind=claim_kind,
            payload_cid=payload_cid,
            source_event_ids=tuple(source_event_ids), round=round,
            handoff_id=hid,
        )
        h = TypedHandoff(
            handoff_id=hid, source_role=source_role,
            source_agent_id=source_agent_id,
            to_role=to_role, claim_kind=claim_kind,
            payload=str(payload),
            source_event_ids=tuple(source_event_ids),
            round=round, payload_cid=payload_cid,
            prev_chain_hash=self._chain_head,
            chain_hash=chain_hash,
            emitted_at=time.time(),
        )
        self._entries.append(h)
        self._chain_head = chain_hash
        self._next_id += 1
        return h

    def verify_chain(self) -> bool:
        """Re-derive every chain_hash and compare. Returns False iff
        any link is inconsistent (tamper / truncation detector)."""
        prev = "GENESIS"
        for h in self._entries:
            expected = _compute_chain_hash(
                prev_chain_hash=prev,
                source_role=h.source_role,
                source_agent_id=h.source_agent_id,
                to_role=h.to_role, claim_kind=h.claim_kind,
                payload_cid=h.payload_cid,
                source_event_ids=h.source_event_ids,
                round=h.round, handoff_id=h.handoff_id,
            )
            if expected != h.chain_hash:
                return False
            if h.prev_chain_hash != prev:
                return False
            prev = h.chain_hash
        return True

    def entries(self) -> tuple[TypedHandoff, ...]:
        return tuple(self._entries)

    def filter(self, *, source_role: str | None = None,
               to_role: str | None = None,
               claim_kind: str | None = None,
               ) -> list[TypedHandoff]:
        out: list[TypedHandoff] = []
        for h in self._entries:
            if source_role is not None and h.source_role != source_role:
                continue
            if to_role is not None and h.to_role != to_role:
                continue
            if claim_kind is not None and h.claim_kind != claim_kind:
                continue
            out.append(h)
        return out

    def __len__(self) -> int:
        return len(self._entries)


# =============================================================================
# DeliveryAccount — per-(source, to, kind) counters
# =============================================================================


@dataclass
class DeliveryAccount:
    """Per-(source_role, to_role, claim_kind) delivery counters.

    Each call to ``record`` adds one data point. ``summary()`` returns
    pooled counts so the benchmark can report "how many tokens does
    the auditor role actually consume as a function of delivery
    strategy?" — the Phase-31 headline.

    The account is *write-only during a run*; summaries are computed
    at the end and are the only aggregate the benchmark reports.
    """

    _rows: list[dict] = field(default_factory=list)

    def record(self, h: TypedHandoff, outcome: str) -> None:
        """``outcome`` ∈ {"accepted", "deduped", "overflow",
        "wrong_role", "dropped_no_subscriber"}."""
        self._rows.append({
            "source_role": h.source_role,
            "to_role": h.to_role,
            "claim_kind": h.claim_kind,
            "outcome": outcome,
            "payload_tokens": h.n_tokens,
            "round": h.round,
        })

    def record_drop(self, source_role: str, to_role: str,
                    claim_kind: str, payload_tokens: int, round: int,
                    outcome: str = "dropped_no_subscriber") -> None:
        self._rows.append({
            "source_role": source_role, "to_role": to_role,
            "claim_kind": claim_kind, "outcome": outcome,
            "payload_tokens": payload_tokens, "round": round,
        })

    def summary(self) -> dict:
        """Pooled counts. Keys:

          * ``total_handoffs``
          * ``by_outcome``          — outcome → count
          * ``by_claim_kind``       — claim_kind → count
          * ``by_to_role``          — to_role → {accepted, tokens}
          * ``tokens_delivered``    — total tokens accepted
        """
        by_outcome: dict[str, int] = {}
        by_kind: dict[str, int] = {}
        by_to_role: dict[str, dict] = {}
        total_accepted = 0
        total_tokens = 0
        for r in self._rows:
            by_outcome[r["outcome"]] = by_outcome.get(r["outcome"], 0) + 1
            by_kind[r["claim_kind"]] = by_kind.get(r["claim_kind"], 0) + 1
            tr = r["to_role"]
            slot = by_to_role.setdefault(
                tr, {"accepted": 0, "tokens": 0, "dropped": 0,
                     "deduped": 0, "overflow": 0, "wrong_role": 0})
            if r["outcome"] == "accepted":
                slot["accepted"] += 1
                slot["tokens"] += r["payload_tokens"]
                total_accepted += 1
                total_tokens += r["payload_tokens"]
            elif r["outcome"] == "deduped":
                slot["deduped"] += 1
            elif r["outcome"] == "overflow":
                slot["overflow"] += 1
            elif r["outcome"] == "wrong_role":
                slot["wrong_role"] += 1
            elif r["outcome"].startswith("dropped"):
                slot["dropped"] += 1
        return {
            "total_handoffs": len(self._rows),
            "total_accepted": total_accepted,
            "tokens_delivered": total_tokens,
            "by_outcome": by_outcome,
            "by_claim_kind": by_kind,
            "by_to_role": by_to_role,
        }


# =============================================================================
# HandoffRouter — the wiring layer
# =============================================================================


@dataclass
class HandoffRouter:
    """Ties the pieces together: emits handoffs to subscribed roles'
    inboxes, logs provenance, accounts outcomes.

    Usage (simplified):

        router = HandoffRouter(
            subs=subs, log=HandoffLog(),
            inboxes={"auditor": RoleInbox(role="auditor", capacity=16)},
            account=DeliveryAccount(),
        )
        h = router.emit(source_role="db_admin", source_agent_id=0,
                        claim_kind="SLOW_QUERY_OBSERVED",
                        payload="pg_stat_statements#12 mean_ms=4200",
                        source_event_ids=[42, 43], round=1)

    ``emit`` returns the ``TypedHandoff`` (already in the log), the
    subscribed consumer roles, and the per-role outcome map. It does
    *not* deliver to a role whose inbox is not registered — that's
    treated as a spec bug (the benchmark loops over
    ``subs.all_claim_kinds()`` to prove coverage).

    The router is deliberately stateless across roles: the only
    shared mutable state is the ``HandoffLog`` (chain head) and the
    per-role inboxes. This makes the module easy to reason about in
    tests.
    """

    subs: RoleSubscriptionTable
    log: HandoffLog = field(default_factory=HandoffLog)
    inboxes: dict[str, RoleInbox] = field(default_factory=dict)
    account: DeliveryAccount = field(default_factory=DeliveryAccount)

    def register_inbox(self, inbox: RoleInbox) -> None:
        self.inboxes[inbox.role] = inbox

    def emit(self, source_role: str, source_agent_id: int,
             claim_kind: str, payload: str,
             source_event_ids: Sequence[int] = (),
             round: int = 0,
             ) -> tuple[TypedHandoff, dict[str, str]]:
        """Write a handoff and deliver it to every subscribed role.

        Returns (handoff, outcome_by_role_dict). The outcome_by_role
        dict maps each consumer role to the ``offer`` outcome or to
        ``"dropped_no_subscriber"`` when the role is subscribed but
        its inbox is missing / ``"dropped_no_subscriber"`` at the
        subscription level when no role subscribes to this claim.
        """
        consumers = self.subs.consumers(source_role, claim_kind)
        # Emit to the log even with no subscribers — so provenance is
        # preserved for auditing and unit tests.
        h = self.log.emit(
            source_role=source_role, source_agent_id=source_agent_id,
            to_role="__multicast__" if consumers else "__orphan__",
            claim_kind=claim_kind, payload=payload,
            source_event_ids=source_event_ids, round=round,
        )
        outcomes: dict[str, str] = {}
        if not consumers:
            self.account.record_drop(
                source_role=source_role, to_role="__orphan__",
                claim_kind=claim_kind, payload_tokens=h.n_tokens,
                round=round, outcome="dropped_no_subscriber")
            return h, outcomes
        # For each consumer role, offer a variant with to_role set.
        for to_role in sorted(consumers):
            h_to = replace(h, to_role=to_role)
            inbox = self.inboxes.get(to_role)
            if inbox is None:
                self.account.record_drop(
                    source_role=source_role, to_role=to_role,
                    claim_kind=claim_kind,
                    payload_tokens=h.n_tokens, round=round,
                    outcome="dropped_no_inbox")
                outcomes[to_role] = "dropped_no_inbox"
                continue
            outcome = inbox.offer(h_to)
            self.account.record(h_to, outcome)
            outcomes[to_role] = outcome
        return h, outcomes

    # --- Read-only convenience views ---

    def log_length(self) -> int:
        return len(self.log)

    def verify(self) -> bool:
        return self.log.verify_chain()

    def summary(self) -> dict:
        return {
            "log_length": len(self.log),
            "account": self.account.summary(),
            "inboxes": {r: box.stats()
                        for r, box in sorted(self.inboxes.items())},
            "chain_ok": self.log.verify_chain(),
        }


# =============================================================================
# Module-level constants — canonical outcome strings (stable for JSON
# artifacts downstream).
# =============================================================================


OUTCOME_ACCEPTED = "accepted"
OUTCOME_DEDUPED = "deduped"
OUTCOME_OVERFLOW = "overflow"
OUTCOME_WRONG_ROLE = "wrong_role"
OUTCOME_DROPPED_NO_SUBSCRIBER = "dropped_no_subscriber"
OUTCOME_DROPPED_NO_INBOX = "dropped_no_inbox"

ALL_OUTCOMES = (OUTCOME_ACCEPTED, OUTCOME_DEDUPED, OUTCOME_OVERFLOW,
                OUTCOME_WRONG_ROLE, OUTCOME_DROPPED_NO_SUBSCRIBER,
                OUTCOME_DROPPED_NO_INBOX)
