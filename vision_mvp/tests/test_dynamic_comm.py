"""Unit tests for ``vision_mvp.core.dynamic_comm`` — the Phase-35
dynamic-communication primitive layer.

Coverage:
  * EscalationThread construction (frozen, hashable, bounded).
  * ThreadRegistry idempotence: opening an identical thread twice
    returns the same thread_id and preserves state.
  * post_reply outcomes: accepted / closed / over_cap /
    rounds_exhausted + non-member DynamicCommError.
  * close_thread resolution rules: SINGLE_INDEPENDENT_ROOT,
    QUORUM_AGREE, CONFLICT, NO_CONSENSUS, TIMEOUT.
  * Thread-resolution handoff is routed through the standard
    HandoffRouter and lands in the subscribed role's inbox.
  * Hash-chain integrity is preserved across thread events
    (THREAD:OPEN / THREAD:REPLY / THREAD:CLOSE +
    CLAIM_THREAD_RESOLUTION all share the same log).
  * Witness-token cap is enforced on post_reply.
  * Non-member roles never see thread-internal messages.
"""
from __future__ import annotations

import unittest

from vision_mvp.core.dynamic_comm import (
    ALL_REPLY_KINDS, ALL_RESOLUTION_KINDS, ALL_THREAD_ISSUES,
    CLAIM_THREAD_RESOLUTION, DynamicCommError, DynamicCommRouter,
    EscalationThread, REPLY_AGREE, REPLY_DISAGREE,
    REPLY_DOWNSTREAM_SYMPTOM, REPLY_INDEPENDENT_ROOT,
    REPLY_UNCERTAIN, RESOLUTION_CONFLICT, RESOLUTION_NO_CONSENSUS,
    RESOLUTION_QUORUM_AGREE, RESOLUTION_SINGLE_INDEPENDENT_ROOT,
    THREAD_ISSUE_CONFIRM_CLAIM,
    THREAD_ISSUE_ROOT_CAUSE_CONFLICT, ThreadReply,
    build_resolution_subscriptions, parse_resolution_payload,
)
from vision_mvp.core.role_handoff import (
    HandoffRouter, RoleInbox, RoleSubscriptionTable,
)


# =============================================================================
# Fixtures
# =============================================================================


def _mk_router(auditor_consumer: bool = True) -> DynamicCommRouter:
    subs = RoleSubscriptionTable()
    if auditor_consumer:
        build_resolution_subscriptions(
            subs, opener_roles=["auditor"],
            consumer_roles=["auditor"])
    base = HandoffRouter(subs=subs)
    base.register_inbox(RoleInbox(role="auditor", capacity=16))
    base.register_inbox(RoleInbox(role="db_admin", capacity=16))
    base.register_inbox(RoleInbox(role="sysadmin", capacity=16))
    base.register_inbox(RoleInbox(role="monitor", capacity=16))
    return DynamicCommRouter(base_router=base)


_CANDS = (
    ("sysadmin", "DISK_FILL_CRITICAL", "cron backup exit=137"),
    ("db_admin", "DEADLOCK_SUSPECTED",
     "deadlock relation=orders_payments"),
)


# =============================================================================
# EscalationThread structural tests
# =============================================================================


class TestEscalationThreadBasics(unittest.TestCase):

    def test_open_thread_returns_frozen_descriptor(self):
        r = _mk_router()
        t = r.open_thread(
            opener_role="auditor",
            issue_kind=THREAD_ISSUE_ROOT_CAUSE_CONFLICT,
            members=["db_admin", "sysadmin"],
            candidate_claims=_CANDS,
        )
        self.assertIsInstance(t, EscalationThread)
        # Opener is auto-added
        self.assertIn("auditor", t.members)
        self.assertIn("db_admin", t.members)
        self.assertIn("sysadmin", t.members)
        # Frozen
        with self.assertRaises(Exception):
            t.issue_kind = "something_else"  # type: ignore[misc]

    def test_open_thread_idempotent_on_same_inputs(self):
        r = _mk_router()
        t1 = r.open_thread(
            opener_role="auditor",
            issue_kind=THREAD_ISSUE_ROOT_CAUSE_CONFLICT,
            members=["db_admin", "sysadmin"],
            candidate_claims=_CANDS,
        )
        t2 = r.open_thread(
            opener_role="auditor",
            issue_kind=THREAD_ISSUE_ROOT_CAUSE_CONFLICT,
            members=["db_admin", "sysadmin"],
            candidate_claims=_CANDS,
        )
        self.assertEqual(t1.thread_id, t2.thread_id)
        self.assertEqual(len(r.threads), 1)

    def test_unknown_issue_kind_raises(self):
        r = _mk_router()
        with self.assertRaises(DynamicCommError):
            r.open_thread(
                opener_role="auditor",
                issue_kind="NOT_A_REAL_ISSUE",
                members=["db_admin"],
                candidate_claims=_CANDS,
            )

    def test_empty_candidates_raises(self):
        r = _mk_router()
        with self.assertRaises(DynamicCommError):
            r.open_thread(
                opener_role="auditor",
                issue_kind=THREAD_ISSUE_ROOT_CAUSE_CONFLICT,
                members=["db_admin"],
                candidate_claims=(),
            )

    def test_quorum_beyond_members_raises(self):
        r = _mk_router()
        with self.assertRaises(DynamicCommError):
            r.open_thread(
                opener_role="auditor",
                issue_kind=THREAD_ISSUE_ROOT_CAUSE_CONFLICT,
                members=["db_admin"],
                candidate_claims=_CANDS,
                quorum=99,
            )


# =============================================================================
# post_reply outcomes
# =============================================================================


class TestPostReply(unittest.TestCase):

    def setUp(self):
        self.r = _mk_router()
        self.t = self.r.open_thread(
            opener_role="auditor",
            issue_kind=THREAD_ISSUE_ROOT_CAUSE_CONFLICT,
            members=["db_admin", "sysadmin"],
            candidate_claims=_CANDS,
            max_rounds=2, max_replies_per_member=1,
            quorum=1, witness_token_cap=6,
            round=0,
        )

    def test_accepted(self):
        out = self.r.post_reply(
            self.t.thread_id, "db_admin",
            REPLY_INDEPENDENT_ROOT, 1,
            "deadlock relation=orders_payments", round=1)
        self.assertEqual(out, "accepted")

    def test_non_member_raises(self):
        with self.assertRaises(DynamicCommError):
            self.r.post_reply(
                self.t.thread_id, "network",
                REPLY_INDEPENDENT_ROOT, 1, "w", round=1)

    def test_unknown_reply_kind_raises(self):
        with self.assertRaises(DynamicCommError):
            self.r.post_reply(
                self.t.thread_id, "db_admin", "FOO", 1, "w",
                round=1)

    def test_claim_idx_out_of_range_raises(self):
        with self.assertRaises(DynamicCommError):
            self.r.post_reply(
                self.t.thread_id, "db_admin",
                REPLY_INDEPENDENT_ROOT, 99, "w", round=1)

    def test_over_cap(self):
        self.r.post_reply(
            self.t.thread_id, "db_admin",
            REPLY_INDEPENDENT_ROOT, 1, "w1", round=1)
        out = self.r.post_reply(
            self.t.thread_id, "db_admin",
            REPLY_DISAGREE, 0, "w2", round=1)
        self.assertEqual(out, "over_cap")

    def test_rounds_exhausted(self):
        out = self.r.post_reply(
            self.t.thread_id, "db_admin",
            REPLY_INDEPENDENT_ROOT, 1, "w", round=5)
        self.assertEqual(out, "rounds_exhausted")

    def test_closed(self):
        self.r.post_reply(
            self.t.thread_id, "db_admin",
            REPLY_INDEPENDENT_ROOT, 1, "w", round=1)
        self.r.close_thread(self.t.thread_id, round=1)
        out = self.r.post_reply(
            self.t.thread_id, "sysadmin",
            REPLY_UNCERTAIN, 0, "w", round=1)
        self.assertEqual(out, "closed")

    def test_witness_token_cap_enforced(self):
        long = "a b c d e f g h i j k l m n o"
        self.r.post_reply(
            self.t.thread_id, "db_admin",
            REPLY_INDEPENDENT_ROOT, 1, long, round=1)
        state = self.r.get_state(self.t.thread_id)
        self.assertEqual(len(state.replies[0].witness.split()),
                         self.t.witness_token_cap)

    def test_unknown_thread_raises(self):
        with self.assertRaises(DynamicCommError):
            self.r.post_reply("T-nonexistent", "db_admin",
                              REPLY_INDEPENDENT_ROOT, 0, "w",
                              round=1)


# =============================================================================
# close_thread resolution rules
# =============================================================================


class TestCloseThreadResolution(unittest.TestCase):

    def _fresh_router(self) -> DynamicCommRouter:
        return _mk_router()

    def _open(self, r: DynamicCommRouter, **kw) -> EscalationThread:
        return r.open_thread(
            opener_role="auditor",
            issue_kind=THREAD_ISSUE_ROOT_CAUSE_CONFLICT,
            members=["db_admin", "sysadmin"],
            candidate_claims=_CANDS,
            max_rounds=2, max_replies_per_member=1, quorum=1,
            witness_token_cap=8, round=0, **kw)

    def test_single_independent_root(self):
        r = self._fresh_router()
        t = self._open(r)
        r.post_reply(t.thread_id, "db_admin",
                     REPLY_INDEPENDENT_ROOT, 1, "w", round=1)
        r.post_reply(t.thread_id, "sysadmin",
                     REPLY_UNCERTAIN, 0, "w", round=1)
        res = r.close_thread(t.thread_id, round=1)
        self.assertEqual(res.resolution_kind,
                         RESOLUTION_SINGLE_INDEPENDENT_ROOT)
        self.assertEqual(res.resolved_claim_idx, 1)

    def test_conflict_on_two_ir(self):
        r = self._fresh_router()
        t = self._open(r)
        r.post_reply(t.thread_id, "db_admin",
                     REPLY_INDEPENDENT_ROOT, 1, "w", round=1)
        r.post_reply(t.thread_id, "sysadmin",
                     REPLY_INDEPENDENT_ROOT, 0, "w", round=1)
        res = r.close_thread(t.thread_id, round=1)
        self.assertEqual(res.resolution_kind, RESOLUTION_CONFLICT)
        self.assertIsNone(res.resolved_claim_idx)

    def test_conflict_on_ir_plus_disagree(self):
        r = self._fresh_router()
        t = self._open(r)
        r.post_reply(t.thread_id, "db_admin",
                     REPLY_INDEPENDENT_ROOT, 1, "w", round=1)
        r.post_reply(t.thread_id, "sysadmin",
                     REPLY_DISAGREE, 1, "w", round=1)
        res = r.close_thread(t.thread_id, round=1)
        self.assertEqual(res.resolution_kind, RESOLUTION_CONFLICT)

    def test_quorum_agree(self):
        # Need more members so quorum = 2 is realistic.
        subs = RoleSubscriptionTable()
        build_resolution_subscriptions(
            subs, ["auditor"], ["auditor"])
        base = HandoffRouter(subs=subs)
        base.register_inbox(RoleInbox(role="auditor", capacity=8))
        r = DynamicCommRouter(base_router=base)
        t = r.open_thread(
            opener_role="auditor",
            issue_kind=THREAD_ISSUE_CONFIRM_CLAIM,
            members=["db_admin", "sysadmin", "monitor"],
            candidate_claims=(("x", "K", "p"),),
            max_rounds=2, max_replies_per_member=1, quorum=2,
            witness_token_cap=4, round=0,
        )
        r.post_reply(t.thread_id, "db_admin",
                     REPLY_AGREE, 0, "ok", round=1)
        r.post_reply(t.thread_id, "sysadmin",
                     REPLY_AGREE, 0, "ok", round=1)
        res = r.close_thread(t.thread_id, round=1)
        self.assertEqual(res.resolution_kind,
                         RESOLUTION_QUORUM_AGREE)
        self.assertEqual(res.resolved_claim_idx, 0)

    def test_no_consensus_on_only_uncertain(self):
        r = self._fresh_router()
        t = self._open(r)
        r.post_reply(t.thread_id, "db_admin",
                     REPLY_UNCERTAIN, 1, "w", round=1)
        r.post_reply(t.thread_id, "sysadmin",
                     REPLY_UNCERTAIN, 0, "w", round=1)
        res = r.close_thread(t.thread_id, round=1)
        self.assertEqual(res.resolution_kind,
                         RESOLUTION_NO_CONSENSUS)

    def test_close_is_idempotent(self):
        r = self._fresh_router()
        t = self._open(r)
        r.post_reply(t.thread_id, "db_admin",
                     REPLY_INDEPENDENT_ROOT, 1, "w", round=1)
        res1 = r.close_thread(t.thread_id, round=1)
        res2 = r.close_thread(t.thread_id, round=1)
        self.assertEqual(res1, res2)


# =============================================================================
# Public-surface integration: resolution handoff lands in inbox
# =============================================================================


class TestResolutionHandoffDelivery(unittest.TestCase):

    def test_resolution_enters_auditor_inbox(self):
        r = _mk_router(auditor_consumer=True)
        t = r.open_thread(
            opener_role="auditor",
            issue_kind=THREAD_ISSUE_ROOT_CAUSE_CONFLICT,
            members=["db_admin", "sysadmin"],
            candidate_claims=_CANDS,
        )
        r.post_reply(t.thread_id, "db_admin",
                     REPLY_INDEPENDENT_ROOT, 1, "w", round=1)
        r.close_thread(t.thread_id, round=2)
        handoffs = r.inboxes["auditor"].peek()
        kinds = {h.claim_kind for h in handoffs}
        self.assertIn(CLAIM_THREAD_RESOLUTION, kinds)

    def test_resolution_does_not_enter_non_member_inbox(self):
        # monitor is not subscribed to CLAIM_THREAD_RESOLUTION and
        # is not a thread member. Its inbox must stay empty.
        r = _mk_router(auditor_consumer=True)
        t = r.open_thread(
            opener_role="auditor",
            issue_kind=THREAD_ISSUE_ROOT_CAUSE_CONFLICT,
            members=["db_admin", "sysadmin"],
            candidate_claims=_CANDS,
        )
        r.post_reply(t.thread_id, "db_admin",
                     REPLY_INDEPENDENT_ROOT, 1, "w", round=1)
        r.close_thread(t.thread_id, round=2)
        self.assertEqual(r.inboxes["monitor"].n_held, 0)

    def test_resolution_payload_parses(self):
        r = _mk_router(auditor_consumer=True)
        t = r.open_thread(
            opener_role="auditor",
            issue_kind=THREAD_ISSUE_ROOT_CAUSE_CONFLICT,
            members=["db_admin", "sysadmin"],
            candidate_claims=_CANDS,
        )
        r.post_reply(t.thread_id, "db_admin",
                     REPLY_INDEPENDENT_ROOT, 1, "w", round=1)
        r.close_thread(t.thread_id, round=2)
        h = next(h for h in r.inboxes["auditor"].peek()
                 if h.claim_kind == CLAIM_THREAD_RESOLUTION)
        parsed = parse_resolution_payload(h.payload)
        self.assertEqual(parsed["kind"],
                         RESOLUTION_SINGLE_INDEPENDENT_ROOT)
        self.assertEqual(parsed["winner"],
                         "db_admin/DEADLOCK_SUSPECTED")
        self.assertEqual(parsed["losers"],
                         "sysadmin/DISK_FILL_CRITICAL")


# =============================================================================
# Hash-chain integrity across thread events
# =============================================================================


class TestChainIntegrity(unittest.TestCase):

    def test_chain_verifies_after_thread_lifecycle(self):
        r = _mk_router(auditor_consumer=True)
        # Emit a normal handoff first
        r.emit(source_role="db_admin", source_agent_id=0,
               claim_kind="DEADLOCK_SUSPECTED",
               payload="deadlock relation=orders_payments",
               source_event_ids=[1], round=0)
        t = r.open_thread(
            opener_role="auditor",
            issue_kind=THREAD_ISSUE_ROOT_CAUSE_CONFLICT,
            members=["db_admin", "sysadmin"],
            candidate_claims=_CANDS,
        )
        r.post_reply(t.thread_id, "db_admin",
                     REPLY_INDEPENDENT_ROOT, 1, "w", round=1)
        r.post_reply(t.thread_id, "sysadmin",
                     REPLY_UNCERTAIN, 0, "w", round=1)
        r.close_thread(t.thread_id, round=2)
        self.assertTrue(r.verify())

    def test_log_contains_thread_events(self):
        r = _mk_router(auditor_consumer=True)
        t = r.open_thread(
            opener_role="auditor",
            issue_kind=THREAD_ISSUE_ROOT_CAUSE_CONFLICT,
            members=["db_admin", "sysadmin"],
            candidate_claims=_CANDS,
        )
        r.post_reply(t.thread_id, "db_admin",
                     REPLY_INDEPENDENT_ROOT, 1, "w", round=1)
        r.close_thread(t.thread_id, round=2)
        log = r.log
        kinds = {e.claim_kind for e in log.entries()}
        self.assertIn("THREAD:OPEN", kinds)
        self.assertIn("THREAD:REPLY", kinds)
        self.assertIn("THREAD:CLOSE", kinds)
        self.assertIn(CLAIM_THREAD_RESOLUTION, kinds)


# =============================================================================
# Bounded-context accounting
# =============================================================================


class TestBoundedContext(unittest.TestCase):

    def test_mean_replies_per_thread_bound(self):
        r = _mk_router(auditor_consumer=True)
        t = r.open_thread(
            opener_role="auditor",
            issue_kind=THREAD_ISSUE_ROOT_CAUSE_CONFLICT,
            members=["db_admin", "sysadmin"],
            candidate_claims=_CANDS,
            max_rounds=2, max_replies_per_member=1,
            witness_token_cap=10, quorum=1,
        )
        r.post_reply(t.thread_id, "db_admin",
                     REPLY_INDEPENDENT_ROOT, 1,
                     "deadlock relation=orders_payments",
                     round=1)
        r.post_reply(t.thread_id, "sysadmin",
                     REPLY_UNCERTAIN, 0,
                     "backup cron archival",
                     round=1)
        r.close_thread(t.thread_id, round=2)
        summary = r.account.summary()
        # Bound: replies ≤ len(members) * max_replies_per_member
        self.assertLessEqual(summary["n_replies_total"],
                              len(t.members)
                              * t.max_replies_per_member)
        # Witness tokens ≤ replies * witness_token_cap
        self.assertLessEqual(summary["n_witness_tokens_total"],
                              summary["n_replies_total"]
                              * t.witness_token_cap)

    def test_vocabularies_declared(self):
        # Sanity on the module-level constants — any new enum member
        # must be explicitly added to the vocabulary tuples.
        self.assertIn(REPLY_INDEPENDENT_ROOT, ALL_REPLY_KINDS)
        self.assertIn(REPLY_DOWNSTREAM_SYMPTOM, ALL_REPLY_KINDS)
        self.assertIn(RESOLUTION_SINGLE_INDEPENDENT_ROOT,
                      ALL_RESOLUTION_KINDS)
        self.assertIn(THREAD_ISSUE_ROOT_CAUSE_CONFLICT,
                      ALL_THREAD_ISSUES)


# =============================================================================
# Resolution payload serialisation
# =============================================================================


class TestResolutionPayload(unittest.TestCase):

    def test_winner_losers_roundtrip(self):
        r = _mk_router(auditor_consumer=True)
        t = r.open_thread(
            opener_role="auditor",
            issue_kind=THREAD_ISSUE_ROOT_CAUSE_CONFLICT,
            members=["db_admin", "sysadmin"],
            candidate_claims=(
                ("sysadmin", "TLS_EXPIRED", "mail expired"),
                ("db_admin", "DNS_MISROUTE", "api.internal SERVFAIL"),
                ("monitor", "ERROR_RATE_SPIKE", "api"),
            ),
        )
        r.post_reply(t.thread_id, "db_admin",
                     REPLY_INDEPENDENT_ROOT, 1, "api SERVFAIL",
                     round=1)
        r.close_thread(t.thread_id, round=2)
        h = next(h for h in r.inboxes["auditor"].peek()
                 if h.claim_kind == CLAIM_THREAD_RESOLUTION)
        parsed = parse_resolution_payload(h.payload)
        self.assertEqual(parsed["winner"],
                         "db_admin/DNS_MISROUTE")
        losers = set(parsed["losers"].split(","))
        self.assertIn("sysadmin/TLS_EXPIRED", losers)
        self.assertIn("monitor/ERROR_RATE_SPIKE", losers)

    def test_losers_none_on_single_candidate(self):
        r = _mk_router(auditor_consumer=True)
        t = r.open_thread(
            opener_role="auditor",
            issue_kind=THREAD_ISSUE_CONFIRM_CLAIM,
            members=["db_admin"],
            candidate_claims=(("db_admin", "X", "p"),),
        )
        r.post_reply(t.thread_id, "db_admin",
                     REPLY_INDEPENDENT_ROOT, 0, "p", round=1)
        r.close_thread(t.thread_id, round=2)
        h = next(h for h in r.inboxes["auditor"].peek()
                 if h.claim_kind == CLAIM_THREAD_RESOLUTION)
        parsed = parse_resolution_payload(h.payload)
        self.assertEqual(parsed["losers"], "none")


if __name__ == "__main__":
    unittest.main()
