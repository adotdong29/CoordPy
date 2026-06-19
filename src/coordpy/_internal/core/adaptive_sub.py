"""Phase 36 Part C — bounded adaptive subscriptions as a comparison primitive.

Phase 35's dynamic-coordination primitive is *one* answer to the
"static typed handoffs are not always enough" result: a typed,
bounded, explicitly-terminated escalation thread that scopes one
coordination question to a small member set. Conjecture C35-5
raised the alternative design: **bounded adaptive subscriptions** —
instead of opening a short-lived thread, the orchestrator
*temporarily edits the subscription table* to route a previously-
unrouted claim kind from a producer to a consumer for a bounded
number of rounds, then removes the edit.

Phase 36 Part C ships this alternative as a serious comparison
point, not a strawman. The module provides:

  * ``AdaptiveEdge`` — frozen record of one temporary edge
    ``(source_role, claim_kind, consumer_role, ttl_rounds)``.
  * ``AdaptiveSubscriptionTable`` — the subscription table plus a
    tick counter, a bounded registry of active edges, and a hard
    cap on the number of concurrent edges (the bounded-context
    analogue of Phase-35's ``T`` thread cap).
  * ``AdaptiveSubRouter`` — wraps ``HandoffRouter`` so that edits
    to the table are uniformly audited through a single log; also
    enforces per-round tick-down of TTLs.
  * ``AdaptiveSubAccount`` — counters for "how many edges were
    installed / how many expired / how many un-used" used by the
    benchmark.
  * ``CLAIM_CAUSALITY_HYPOTHESIS`` — the single causality-probe
    claim kind used in the Phase-36 Part C experiment: under
    adaptive subscriptions, the aggregator (auditor) asks every
    producer on a contested incident to emit one
    ``CAUSALITY_HYPOTHESIS:<reply_kind>`` handoff, then removes
    the edge when the answer is in.

What this primitive buys vs. escalation threads
-----------------------------------------------

1. **Same underlying substrate** — typed handoffs, hash-chained
   log, per-role subscription table. No new message bus.
2. **No new object type** — a thread is a *thing* you open and
   close; an adaptive edge is a *modification* of the existing
   routing table that expires. Callers who already reason about
   subscriptions do not learn new vocabulary.
3. **Weaker bounded-context guarantee** — the bounded-context
   bound (Theorem P35-2) is recovered only if the caller enforces
   the edge cap ``E_max`` and the TTL ``n_rounds`` at runtime;
   there is no type-level enforcement. An adaptive sub that
   forgets to tick TTLs grows unboundedly.
4. **No explicit member-set frozen invariant** — a subscription
   edit routes a kind from one producer to one (or more)
   consumers for a window, but the "thread" equivalent
   membership is ambiguous: is the set of roles involved the
   (producer, consumer) pair, or every consumer? The Phase-35
   frozen-member-set invariant is lost.
5. **Same expressivity on contested scenarios** — empirically
   (Phase 36 § D.3) the Phase-35 contested bank is solved by
   bounded adaptive subscriptions at equal full accuracy to
   dynamic threads, at marginally lower token cost but with
   *no typed coordination vocabulary* at the message layer.

Scope discipline (what this module does NOT do)
-----------------------------------------------

  * It does NOT modify the Phase-31 ``HandoffRouter`` or
    ``HandoffLog``. Subscription edits are recorded in a local
    edge registry and propagated into the base router's
    subscription table when installed; on TTL expiry, the
    subscription is removed from the base table.
  * It does NOT attempt to unify the adaptive-subscription and
    escalation-thread primitives. That unification question —
    "is one strictly more general than the other?" — is
    Conjecture C35-5, still open as of Phase 36.
  * It does NOT inherit any of the thread primitive's reply
    typing. A producer under an adaptive subscription emits a
    regular typed handoff; the "reply" shape is the payload
    format, not a separate vocabulary. This is how
    adaptive-sub's *lack* of a typed coordination vocabulary
    surfaces — the load-bearing bits are encoded in the payload
    by convention.
  * It does NOT compete on the peer-review / adversarial-integrity
    axis. Same audit guarantees as the base router (accidental
    tamper detection via hash chain; authenticated provenance
    still belongs to ``peer_review``).

Theoretical anchor: RESULTS_PHASE36.md § B.2 (Theorem P36-2,
Conjecture C36-5).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from coordpy._internal.core.role_handoff import (
    HandoffRouter, RoleInbox, RoleSubscriptionTable,
    TypedHandoff,
)


# =============================================================================
# The dedicated causality-hypothesis claim kind — Part C's probe
# =============================================================================


# Producer emits a payload of the shape
# "kind=<reply_kind> idx=<candidate_idx> witness=<short_witness>"
# under this claim kind when subscribed by an adaptive edge.
CLAIM_CAUSALITY_HYPOTHESIS = "CAUSALITY_HYPOTHESIS"


def format_hypothesis_payload(reply_kind: str,
                                candidate_idx: int,
                                upstream_kind: str = "",
                                witness: str = "") -> str:
    """Canonical one-line payload for a causality-hypothesis handoff.

    Mirrors the reply-encoding of ``ThreadResolution.as_payload_string``
    so a decoder can parse both formats uniformly. Carries the
    ``upstream_kind`` — the claim kind of the candidate this
    hypothesis is *about* — so a decoder does not have to recover
    it from the producer's broader claim set (the adaptive-sub
    analogue of ``thread.candidate_claims[idx]``).
    """
    return (f"kind={reply_kind} idx={candidate_idx} "
            f"upstream_kind={upstream_kind} "
            f"witness={witness.strip()}")


def parse_hypothesis_payload(payload: str) -> dict:
    """Parse a causality-hypothesis payload into a dict with keys
    ``kind``, ``idx``, ``upstream_kind``, ``witness``. Missing
    fields default to empty strings / -1.
    """
    out: dict[str, str] = {
        "kind": "", "idx": "-1",
        "upstream_kind": "", "witness": "",
    }
    # Payload shape: "kind=X idx=N upstream_kind=K witness=W words..."
    if "witness=" in payload:
        head, wit = payload.split("witness=", 1)
        out["witness"] = wit.strip()
    else:
        head = payload
    for tok in head.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            if k in ("kind", "idx", "upstream_kind"):
                out[k] = v
    return out


# =============================================================================
# AdaptiveEdge + registry
# =============================================================================


@dataclass(frozen=True)
class AdaptiveEdge:
    """One installed subscription edit.

    Fields:
      * ``source_role``    — producer role being routed.
      * ``claim_kind``     — the claim kind newly routed.
      * ``consumer_roles`` — sorted tuple of consumer roles.
      * ``installed_at``   — round at install.
      * ``ttl_rounds``     — number of additional rounds this
        edge stays active. The router's ``tick`` call decrements
        by one; at 0 the edge is removed.
      * ``edge_id``        — deterministic id of the edge.
    """

    source_role: str
    claim_kind: str
    consumer_roles: tuple[str, ...]
    installed_at: int
    ttl_rounds: int
    edge_id: str

    def as_dict(self) -> dict:
        return {
            "source_role": self.source_role,
            "claim_kind": self.claim_kind,
            "consumer_roles": list(self.consumer_roles),
            "installed_at": self.installed_at,
            "ttl_rounds": self.ttl_rounds,
            "edge_id": self.edge_id,
        }


class AdaptiveSubError(Exception):
    """Raised for specification bugs in adaptive-sub usage.

    Includes installing more edges than ``max_active_edges``,
    removing an edge that was never installed, ticking past the
    TTL on a non-existent edge.
    """


@dataclass
class AdaptiveSubAccount:
    """Per-scenario counters of edge installs, expiries, usage.

    The Phase-36 Part C benchmark surfaces four numbers per
    scenario:

      * ``n_installed``        — edges installed this scenario.
      * ``n_expired``          — edges ticked down to 0 and
        removed from the table.
      * ``n_expired_unused``   — of the above, how many saw zero
        handoffs delivered while active (over-provisioning signal).
      * ``n_handoffs_via_adaptive`` — handoffs delivered *through*
        an adaptive edge (i.e. their (source, kind) matched an
        active edge at emit time and would NOT have been routed
        otherwise).
    """

    n_installed: int = 0
    n_expired: int = 0
    n_expired_unused: int = 0
    n_handoffs_via_adaptive: int = 0
    _rows: list[dict] = field(default_factory=list)

    def record_install(self, edge: AdaptiveEdge) -> None:
        self.n_installed += 1
        self._rows.append({
            "event": "install",
            "edge": edge.as_dict(),
            "at": time.time(),
        })

    def record_expire(self, edge: AdaptiveEdge,
                       n_delivered: int) -> None:
        self.n_expired += 1
        if n_delivered == 0:
            self.n_expired_unused += 1
        self._rows.append({
            "event": "expire", "edge": edge.as_dict(),
            "n_delivered": n_delivered, "at": time.time(),
        })

    def record_delivery(self, edge: AdaptiveEdge) -> None:
        self.n_handoffs_via_adaptive += 1
        self._rows.append({
            "event": "delivery", "edge_id": edge.edge_id,
            "at": time.time(),
        })

    def summary(self) -> dict:
        return {
            "n_installed": self.n_installed,
            "n_expired": self.n_expired,
            "n_expired_unused": self.n_expired_unused,
            "n_handoffs_via_adaptive": self.n_handoffs_via_adaptive,
        }

    def rows(self) -> list[dict]:
        return list(self._rows)


# =============================================================================
# AdaptiveSubscriptionTable — wraps the subscription table with a registry
# =============================================================================


@dataclass
class AdaptiveSubscriptionTable:
    """A ``RoleSubscriptionTable`` with a bounded adaptive-edge overlay.

    The base subscription table is the Phase-31 object; adaptive
    edges are *additions* on top. ``consumers()`` returns the
    union of static consumers and active adaptive consumers for
    the given (source_role, claim_kind).

    Invariants (enforced at ``install_edge``):

      * ``max_active_edges`` cap — installing a new edge when
        the cap is hit raises ``AdaptiveSubError``.
      * ``ttl_rounds >= 1`` — a 0-TTL edge is useless.
      * edges are *deterministic in their id* — the id is a
        content address over ``(source, kind, consumers,
        installed_at)`` so an idempotent retry of the same
        install returns the same edge without duplication.
    """

    base: RoleSubscriptionTable
    max_active_edges: int = 4
    _edges: dict[str, AdaptiveEdge] = field(default_factory=dict)
    _n_delivered_per_edge: dict[str, int] = field(default_factory=dict)
    _tick: int = 0

    def _edge_id(self, source_role: str, claim_kind: str,
                  consumers: Sequence[str], installed_at: int) -> str:
        import hashlib
        blob = (f"{source_role}|{claim_kind}|"
                f"{','.join(sorted(consumers))}|{installed_at}"
                ).encode("utf-8")
        return "E-" + hashlib.sha256(blob).hexdigest()[:12]

    def install_edge(self,
                     source_role: str,
                     claim_kind: str,
                     consumer_roles: Iterable[str],
                     ttl_rounds: int = 1,
                     ) -> AdaptiveEdge:
        """Install a temporary routing edge.

        The edge stays active for ``ttl_rounds`` ticks. Each call
        to ``tick()`` decrements all edges' TTLs by one; at 0 the
        edge is removed from the active set and from the base
        subscription table (if it was added to it).

        Raises ``AdaptiveSubError`` if the per-round active-edge
        cap (``max_active_edges``) is hit.
        """
        if ttl_rounds < 1:
            raise AdaptiveSubError("ttl_rounds must be ≥ 1")
        cons_tuple = tuple(sorted(set(consumer_roles)))
        if not cons_tuple:
            raise AdaptiveSubError("at least one consumer role required")
        eid = self._edge_id(source_role, claim_kind, cons_tuple,
                              self._tick)
        if eid in self._edges:
            return self._edges[eid]
        if len(self._edges) >= self.max_active_edges:
            raise AdaptiveSubError(
                f"active-edge cap {self.max_active_edges} reached; "
                f"refuse to install {source_role}/{claim_kind}")
        edge = AdaptiveEdge(
            source_role=source_role, claim_kind=claim_kind,
            consumer_roles=cons_tuple,
            installed_at=self._tick,
            ttl_rounds=ttl_rounds,
            edge_id=eid,
        )
        self._edges[eid] = edge
        self._n_delivered_per_edge[eid] = 0
        # Add to base subscription table.
        self.base.subscribe(source_role, claim_kind, cons_tuple)
        return edge

    def remove_edge(self, edge_id: str) -> AdaptiveEdge | None:
        edge = self._edges.pop(edge_id, None)
        if edge is None:
            return None
        # Remove the subscription ONLY if no static subscription
        # declared the same (source, kind) independently. The
        # base table does not distinguish adaptive from static, so
        # we conservatively leave it in place if other consumers
        # also got subscribed via the same key. Phase-36 Part C
        # does not re-use an adaptive edge's kind with a static
        # subscription in the same experiment — the only shared
        # kind is CLAIM_CAUSALITY_HYPOTHESIS, which has no static
        # subscription by design.
        key = (edge.source_role, edge.claim_kind)
        existing = self.base._table.get(key, frozenset())
        cons = frozenset(edge.consumer_roles)
        remaining = existing - cons
        if remaining:
            self.base._table[key] = remaining
        else:
            self.base._table.pop(key, None)
        return edge

    def tick(self, n: int = 1) -> list[AdaptiveEdge]:
        """Advance the internal round counter by ``n`` and expire
        any edge whose TTL has run out. Returns the list of
        removed edges so the caller can account for them.
        """
        self._tick += n
        expired: list[AdaptiveEdge] = []
        for eid in list(self._edges.keys()):
            edge = self._edges[eid]
            age = self._tick - edge.installed_at
            if age >= edge.ttl_rounds:
                removed = self.remove_edge(eid)
                if removed is not None:
                    expired.append(removed)
        return expired

    def active_edges(self) -> tuple[AdaptiveEdge, ...]:
        return tuple(self._edges.values())

    def get_edge(self, edge_id: str) -> AdaptiveEdge | None:
        return self._edges.get(edge_id)

    def record_delivery(self, source_role: str, claim_kind: str) -> None:
        """Mark that a handoff was delivered through an active edge."""
        for eid, edge in self._edges.items():
            if (edge.source_role == source_role
                    and edge.claim_kind == claim_kind):
                self._n_delivered_per_edge[eid] = \
                    self._n_delivered_per_edge.get(eid, 0) + 1

    def n_delivered_on_edge(self, edge_id: str) -> int:
        return self._n_delivered_per_edge.get(edge_id, 0)


# =============================================================================
# AdaptiveSubRouter — HandoffRouter + the adaptive overlay
# =============================================================================


@dataclass
class AdaptiveSubRouter:
    """Extend ``HandoffRouter`` with adaptive-subscription ops.

    Usage:

        base = HandoffRouter(subs=subs)
        router = AdaptiveSubRouter(base_router=base, max_active_edges=4)
        # emit static handoffs as before
        router.emit(...)
        # install a temporary edge
        edge = router.install_edge(
            source_role="db_admin", claim_kind="CAUSALITY_HYPOTHESIS",
            consumer_roles=["auditor"], ttl_rounds=1)
        # producer emits the new claim kind; delivered only to
        # auditor while edge is active
        router.emit(source_role="db_admin", ...,
                    claim_kind="CAUSALITY_HYPOTHESIS", ...)
        # on round advance
        expired = router.tick()
        # edge automatically removed from the subscription table

    The router enforces the active-edge cap through the
    ``AdaptiveSubscriptionTable``; installing a fifth edge raises
    ``AdaptiveSubError``.
    """

    base_router: HandoffRouter
    max_active_edges: int = 4
    adaptive: AdaptiveSubscriptionTable = field(init=False)
    account: AdaptiveSubAccount = field(default_factory=AdaptiveSubAccount)

    def __post_init__(self) -> None:
        self.adaptive = AdaptiveSubscriptionTable(
            base=self.base_router.subs,
            max_active_edges=self.max_active_edges,
        )

    # ------- Phase 31 delegation -------

    def register_inbox(self, inbox: RoleInbox) -> None:
        self.base_router.register_inbox(inbox)

    @property
    def subs(self) -> RoleSubscriptionTable:
        return self.base_router.subs

    @property
    def log(self):
        return self.base_router.log

    @property
    def inboxes(self) -> dict[str, RoleInbox]:
        return self.base_router.inboxes

    def log_length(self) -> int:
        return self.base_router.log_length()

    def verify(self) -> bool:
        return self.base_router.verify()

    def emit(self, source_role: str, source_agent_id: int,
             claim_kind: str, payload: str,
             source_event_ids: Sequence[int] = (),
             round: int = 0,
             ) -> tuple[TypedHandoff, dict[str, str]]:
        h, outcomes = self.base_router.emit(
            source_role=source_role, source_agent_id=source_agent_id,
            claim_kind=claim_kind, payload=payload,
            source_event_ids=source_event_ids, round=round,
        )
        # If this handoff was routed through an active adaptive
        # edge, bump the delivery counter.
        for edge in self.adaptive.active_edges():
            if (edge.source_role == source_role
                    and edge.claim_kind == claim_kind):
                self.adaptive.record_delivery(source_role, claim_kind)
                self.account.record_delivery(edge)
                break
        return h, outcomes

    # ------- Adaptive-sub ops -------

    def install_edge(self,
                     source_role: str,
                     claim_kind: str,
                     consumer_roles: Iterable[str],
                     ttl_rounds: int = 1,
                     ) -> AdaptiveEdge:
        edge = self.adaptive.install_edge(
            source_role=source_role, claim_kind=claim_kind,
            consumer_roles=consumer_roles, ttl_rounds=ttl_rounds)
        self.account.record_install(edge)
        return edge

    def tick(self, n: int = 1) -> list[AdaptiveEdge]:
        expired = self.adaptive.tick(n)
        for edge in expired:
            n_delivered = self.adaptive.n_delivered_on_edge(edge.edge_id)
            self.account.record_expire(edge, n_delivered)
        return expired

    def summary(self) -> dict:
        return {
            "log_length": self.log_length(),
            "chain_ok": self.verify(),
            "adaptive": self.account.summary(),
            "active_edges": [e.as_dict()
                              for e in self.adaptive.active_edges()],
            "n_active_edges": len(self.adaptive.active_edges()),
            "inboxes": {r: box.stats()
                          for r, box in sorted(self.inboxes.items())},
        }


__all__ = [
    "CLAIM_CAUSALITY_HYPOTHESIS",
    "format_hypothesis_payload", "parse_hypothesis_payload",
    "AdaptiveEdge", "AdaptiveSubError", "AdaptiveSubAccount",
    "AdaptiveSubscriptionTable", "AdaptiveSubRouter",
]
