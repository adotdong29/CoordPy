"""Unit tests for ``vision_mvp.core.role_handoff``.

The role-handoff module is the Phase-31 substrate primitive: typed,
provenance-aware, role-scoped handoffs between agents in a team.
These tests cover the contract surface:

  * TypedHandoff immutability + hashability + token accounting.
  * RoleSubscriptionTable (who subscribes to what).
  * RoleInbox bounded capacity + dedup-by-(source, kind, cid) +
    wrong-role rejection.
  * HandoffLog hash-chain verification (tamper / truncation
    detector).
  * DeliveryAccount per-(source_role, to_role, claim_kind) counters.
  * HandoffRouter end-to-end delivery + dropped-no-subscriber /
    dropped-no-inbox outcomes.
"""
from __future__ import annotations

import unittest

from vision_mvp.core.role_handoff import (
    DeliveryAccount, HandoffLog, HandoffRouter, RoleInbox,
    RoleSubscriptionTable, TypedHandoff, _compute_chain_hash,
    OUTCOME_ACCEPTED, OUTCOME_DEDUPED, OUTCOME_OVERFLOW,
    OUTCOME_WRONG_ROLE, OUTCOME_DROPPED_NO_SUBSCRIBER,
    OUTCOME_DROPPED_NO_INBOX,
)


class TestRoleSubscriptionTable(unittest.TestCase):

    def test_subscribe_and_query(self):
        t = RoleSubscriptionTable()
        t.subscribe("db", "SLOW_QUERY", ["auditor"])
        self.assertEqual(t.consumers("db", "SLOW_QUERY"),
                         frozenset({"auditor"}))
        self.assertEqual(t.consumers("db", "NONEXISTENT"), frozenset())

    def test_subscribe_is_union(self):
        t = RoleSubscriptionTable()
        t.subscribe("sys", "DISK_FILL", ["auditor"])
        t.subscribe("sys", "DISK_FILL", ["db"])
        self.assertEqual(t.consumers("sys", "DISK_FILL"),
                         frozenset({"auditor", "db"}))

    def test_all_pairs_and_kinds(self):
        t = RoleSubscriptionTable()
        t.subscribe("a", "X", ["b"])
        t.subscribe("a", "Y", ["b"])
        t.subscribe("c", "X", ["d"])
        self.assertEqual(t.all_claim_kinds(), {"X", "Y"})
        self.assertEqual(sorted(t.all_pairs()),
                         [("a", "X"), ("a", "Y"), ("c", "X")])


class TestRoleInbox(unittest.TestCase):

    def _mk(self, to="auditor", src="db", kind="SLOW_QUERY",
             payload="p", evs=(1,), rnd=0, hid=0) -> TypedHandoff:
        payload_cid = f"cid-{payload}-{','.join(str(e) for e in evs)}"
        return TypedHandoff(
            handoff_id=hid, source_role=src, source_agent_id=0,
            to_role=to, claim_kind=kind, payload=payload,
            source_event_ids=tuple(evs), round=rnd,
            payload_cid=payload_cid,
            prev_chain_hash="", chain_hash="x",
        )

    def test_accept(self):
        ib = RoleInbox(role="auditor", capacity=4)
        self.assertEqual(ib.offer(self._mk()), OUTCOME_ACCEPTED)
        self.assertEqual(ib.n_held, 1)

    def test_dedup_on_same_cid(self):
        ib = RoleInbox(role="auditor", capacity=4)
        h1 = self._mk()
        h2 = self._mk()
        self.assertEqual(ib.offer(h1), OUTCOME_ACCEPTED)
        self.assertEqual(ib.offer(h2), OUTCOME_DEDUPED)
        self.assertEqual(ib.n_held, 1)
        self.assertEqual(ib.n_dedup, 1)

    def test_different_payloads_not_deduped(self):
        ib = RoleInbox(role="auditor", capacity=4)
        ib.offer(self._mk(payload="p1", hid=0))
        ib.offer(self._mk(payload="p2", hid=1))
        self.assertEqual(ib.n_held, 2)

    def test_wrong_role_rejected(self):
        ib = RoleInbox(role="auditor", capacity=4)
        self.assertEqual(ib.offer(self._mk(to="db")),
                         OUTCOME_WRONG_ROLE)
        self.assertEqual(ib.n_held, 0)

    def test_overflow(self):
        ib = RoleInbox(role="auditor", capacity=2)
        ib.offer(self._mk(hid=0, payload="a"))
        ib.offer(self._mk(hid=1, payload="b"))
        out = ib.offer(self._mk(hid=2, payload="c"))
        self.assertEqual(out, OUTCOME_OVERFLOW)
        self.assertEqual(ib.n_held, 2)
        self.assertEqual(ib.n_overflow, 1)

    def test_drain_clears_but_dedup_persists(self):
        ib = RoleInbox(role="auditor", capacity=4)
        ib.offer(self._mk(payload="p1"))
        self.assertEqual(ib.drain()[0].payload, "p1")
        self.assertEqual(ib.n_held, 0)
        # Same cid — should still be deduped after drain.
        self.assertEqual(ib.offer(self._mk(payload="p1")),
                         OUTCOME_DEDUPED)

    def test_token_accounting(self):
        h = self._mk(payload="one two three four five")
        self.assertEqual(h.n_tokens, 5)


class TestHandoffLogChain(unittest.TestCase):

    def test_empty_chain_verifies(self):
        log = HandoffLog()
        self.assertTrue(log.verify_chain())
        self.assertEqual(len(log), 0)

    def test_linear_chain_verifies(self):
        log = HandoffLog()
        for i in range(5):
            log.emit(
                source_role="db", source_agent_id=0,
                to_role="auditor", claim_kind="SLOW",
                payload=f"p{i}", source_event_ids=[i], round=0)
        self.assertEqual(len(log), 5)
        self.assertTrue(log.verify_chain())

    def test_tamper_detected(self):
        log = HandoffLog()
        log.emit(source_role="db", source_agent_id=0,
                 to_role="auditor", claim_kind="SLOW",
                 payload="p", source_event_ids=[1], round=0)
        # Rewrite one entry in place (cannot normally — simulate
        # by creating a new log with altered content).
        bad = log._entries[0]
        log._entries[0] = TypedHandoff(
            handoff_id=bad.handoff_id,
            source_role=bad.source_role,
            source_agent_id=bad.source_agent_id,
            to_role=bad.to_role,
            claim_kind="TAMPERED",           # tampered field
            payload=bad.payload,
            source_event_ids=bad.source_event_ids,
            round=bad.round,
            payload_cid=bad.payload_cid,
            prev_chain_hash=bad.prev_chain_hash,
            chain_hash=bad.chain_hash,   # old hash is now inconsistent
        )
        self.assertFalse(log.verify_chain())

    def test_filter(self):
        log = HandoffLog()
        log.emit("db", 0, "auditor", "SLOW", "p1", [1], 0)
        log.emit("sys", 1, "auditor", "DISK", "p2", [2], 0)
        log.emit("db", 0, "auditor", "POOL", "p3", [3], 0)
        self.assertEqual(len(log.filter(source_role="db")), 2)
        self.assertEqual(len(log.filter(claim_kind="SLOW")), 1)


class TestDeliveryAccount(unittest.TestCase):

    def _mk(self, **kw) -> TypedHandoff:
        defaults = dict(
            handoff_id=0, source_role="db", source_agent_id=0,
            to_role="auditor", claim_kind="SLOW", payload="p",
            source_event_ids=(1,), round=0,
            payload_cid="cid-x", prev_chain_hash="",
            chain_hash="y",
        )
        defaults.update(kw)
        return TypedHandoff(**defaults)

    def test_empty_summary(self):
        acc = DeliveryAccount()
        s = acc.summary()
        self.assertEqual(s["total_handoffs"], 0)
        self.assertEqual(s["total_accepted"], 0)

    def test_accepted_counts_tokens(self):
        acc = DeliveryAccount()
        acc.record(self._mk(payload="one two"), OUTCOME_ACCEPTED)
        acc.record(self._mk(payload="one two three"),
                    OUTCOME_ACCEPTED)
        s = acc.summary()
        self.assertEqual(s["total_accepted"], 2)
        self.assertEqual(s["tokens_delivered"], 2 + 3)

    def test_mixed_outcomes(self):
        acc = DeliveryAccount()
        acc.record(self._mk(payload="x"), OUTCOME_ACCEPTED)
        acc.record(self._mk(payload="x"), OUTCOME_DEDUPED)
        acc.record(self._mk(payload="x"), OUTCOME_OVERFLOW)
        acc.record(self._mk(payload="x"), OUTCOME_WRONG_ROLE)
        acc.record_drop("db", "auditor", "SLOW", 1, 0,
                          outcome=OUTCOME_DROPPED_NO_SUBSCRIBER)
        s = acc.summary()
        self.assertEqual(s["total_handoffs"], 5)
        by = s["by_outcome"]
        self.assertEqual(by[OUTCOME_ACCEPTED], 1)
        self.assertEqual(by[OUTCOME_DEDUPED], 1)
        self.assertEqual(by[OUTCOME_OVERFLOW], 1)
        self.assertEqual(by[OUTCOME_WRONG_ROLE], 1)
        self.assertEqual(by[OUTCOME_DROPPED_NO_SUBSCRIBER], 1)


class TestHandoffRouter(unittest.TestCase):

    def _subs(self) -> RoleSubscriptionTable:
        t = RoleSubscriptionTable()
        t.subscribe("db", "SLOW_QUERY", ["auditor"])
        t.subscribe("sys", "DISK_FILL", ["auditor", "db"])
        return t

    def test_emit_to_subscribed_role(self):
        r = HandoffRouter(subs=self._subs())
        r.register_inbox(RoleInbox(role="auditor"))
        h, out = r.emit(source_role="db", source_agent_id=0,
                         claim_kind="SLOW_QUERY", payload="q1",
                         source_event_ids=[1], round=1)
        self.assertEqual(out, {"auditor": OUTCOME_ACCEPTED})
        self.assertEqual(r.log_length(), 1)
        self.assertTrue(r.verify())
        self.assertEqual(r.inboxes["auditor"].n_held, 1)

    def test_emit_to_multiple_subscribers(self):
        r = HandoffRouter(subs=self._subs())
        r.register_inbox(RoleInbox(role="auditor"))
        r.register_inbox(RoleInbox(role="db"))
        _, out = r.emit(
            source_role="sys", source_agent_id=1,
            claim_kind="DISK_FILL", payload="d1",
            source_event_ids=[2], round=1)
        self.assertEqual(out,
                         {"auditor": OUTCOME_ACCEPTED,
                          "db": OUTCOME_ACCEPTED})

    def test_dropped_no_subscriber(self):
        r = HandoffRouter(subs=self._subs())
        r.register_inbox(RoleInbox(role="auditor"))
        _, out = r.emit(
            source_role="db", source_agent_id=0,
            claim_kind="UNKNOWN_CLAIM", payload="x",
            source_event_ids=[1], round=1)
        # No subscribers for this pair → out is empty,
        # but the log still stored the handoff for provenance.
        self.assertEqual(out, {})
        self.assertEqual(r.log_length(), 1)
        summary = r.account.summary()
        self.assertEqual(
            summary["by_outcome"].get(OUTCOME_DROPPED_NO_SUBSCRIBER, 0),
            1)

    def test_dropped_no_inbox(self):
        r = HandoffRouter(subs=self._subs())
        # Register auditor inbox but not db's — DISK_FILL targets both.
        r.register_inbox(RoleInbox(role="auditor"))
        _, out = r.emit(
            source_role="sys", source_agent_id=1,
            claim_kind="DISK_FILL", payload="d1",
            source_event_ids=[2], round=1)
        self.assertEqual(out,
                         {"auditor": OUTCOME_ACCEPTED,
                          "db": OUTCOME_DROPPED_NO_INBOX})

    def test_dedup_across_emits(self):
        r = HandoffRouter(subs=self._subs())
        r.register_inbox(RoleInbox(role="auditor"))
        r.emit("db", 0, "SLOW_QUERY", "q1", [1], 1)
        _, out = r.emit("db", 0, "SLOW_QUERY", "q1", [1], 1)
        # Same payload, same evs → same cid → deduped at inbox.
        self.assertEqual(out["auditor"], OUTCOME_DEDUPED)
        self.assertEqual(r.inboxes["auditor"].n_held, 1)
        self.assertEqual(r.log_length(), 2)   # log keeps both


class TestChainHashDeterminism(unittest.TestCase):

    def test_same_inputs_same_hash(self):
        h1 = _compute_chain_hash(
            prev_chain_hash="GENESIS", source_role="db",
            source_agent_id=0, to_role="auditor",
            claim_kind="SLOW", payload_cid="abc",
            source_event_ids=(1, 2), round=1, handoff_id=0)
        h2 = _compute_chain_hash(
            prev_chain_hash="GENESIS", source_role="db",
            source_agent_id=0, to_role="auditor",
            claim_kind="SLOW", payload_cid="abc",
            source_event_ids=(1, 2), round=1, handoff_id=0)
        self.assertEqual(h1, h2)

    def test_different_inputs_different_hash(self):
        h1 = _compute_chain_hash(
            prev_chain_hash="GENESIS", source_role="db",
            source_agent_id=0, to_role="auditor",
            claim_kind="SLOW", payload_cid="abc",
            source_event_ids=(1,), round=1, handoff_id=0)
        h2 = _compute_chain_hash(
            prev_chain_hash="GENESIS", source_role="db",
            source_agent_id=0, to_role="auditor",
            claim_kind="SLOW", payload_cid="abd",   # changed
            source_event_ids=(1,), round=1, handoff_id=0)
        self.assertNotEqual(h1, h2)


if __name__ == "__main__":
    unittest.main()
