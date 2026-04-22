"""Unit tests for ``vision_mvp.core.adaptive_sub`` — the Phase-36
Part-C adaptive-subscription primitive.

Coverage:
  * install_edge + tick expire semantics.
  * max_active_edges cap enforced (AdaptiveSubError).
  * Idempotent re-install returns same edge_id.
  * Edge delivery is routed to the consumer's inbox.
  * Expired edge is removed from the base subscription table.
  * format_hypothesis_payload / parse_hypothesis_payload
    round-trip.
  * AdaptiveSubAccount rows match install / expire / delivery.
  * Chain hash stays valid across adaptive-sub emits.
"""
from __future__ import annotations

import unittest

from vision_mvp.core.adaptive_sub import (
    AdaptiveEdge, AdaptiveSubError, AdaptiveSubRouter,
    AdaptiveSubscriptionTable, CLAIM_CAUSALITY_HYPOTHESIS,
    format_hypothesis_payload, parse_hypothesis_payload,
)
from vision_mvp.core.role_handoff import (
    HandoffRouter, RoleInbox, RoleSubscriptionTable,
)


def _mk_router(max_active_edges: int = 4) -> AdaptiveSubRouter:
    subs = RoleSubscriptionTable()
    base = HandoffRouter(subs=subs)
    base.register_inbox(RoleInbox(role="auditor", capacity=16))
    base.register_inbox(RoleInbox(role="db_admin", capacity=16))
    base.register_inbox(RoleInbox(role="network", capacity=16))
    return AdaptiveSubRouter(
        base_router=base, max_active_edges=max_active_edges)


class TestInstallTick(unittest.TestCase):
    def test_install_and_expire(self):
        r = _mk_router()
        edge = r.install_edge("db_admin",
                               CLAIM_CAUSALITY_HYPOTHESIS,
                               ["auditor"], ttl_rounds=1)
        self.assertIsInstance(edge, AdaptiveEdge)
        self.assertEqual(len(r.adaptive.active_edges()), 1)
        expired = r.tick(1)
        self.assertEqual(len(expired), 1)
        self.assertEqual(len(r.adaptive.active_edges()), 0)
        self.assertEqual(r.account.n_installed, 1)
        self.assertEqual(r.account.n_expired, 1)

    def test_idempotent_install(self):
        r = _mk_router()
        e1 = r.install_edge("db_admin",
                             CLAIM_CAUSALITY_HYPOTHESIS,
                             ["auditor"], ttl_rounds=1)
        e2 = r.install_edge("db_admin",
                             CLAIM_CAUSALITY_HYPOTHESIS,
                             ["auditor"], ttl_rounds=1)
        self.assertEqual(e1.edge_id, e2.edge_id)

    def test_max_active_edges_cap(self):
        r = _mk_router(max_active_edges=2)
        r.install_edge("db_admin", "KIND_A",
                        ["auditor"], ttl_rounds=2)
        r.install_edge("network", "KIND_B",
                        ["auditor"], ttl_rounds=2)
        with self.assertRaises(AdaptiveSubError):
            r.install_edge("network", "KIND_C",
                            ["auditor"], ttl_rounds=2)

    def test_zero_ttl_rejected(self):
        r = _mk_router()
        with self.assertRaises(AdaptiveSubError):
            r.install_edge("db_admin", "KIND_X",
                            ["auditor"], ttl_rounds=0)

    def test_empty_consumer_rejected(self):
        r = _mk_router()
        with self.assertRaises(AdaptiveSubError):
            r.install_edge("db_admin", "KIND_X",
                            [], ttl_rounds=1)


class TestDelivery(unittest.TestCase):
    def test_edge_routes_handoff_to_consumer(self):
        r = _mk_router()
        r.install_edge("db_admin", CLAIM_CAUSALITY_HYPOTHESIS,
                        ["auditor"], ttl_rounds=1)
        r.emit(source_role="db_admin", source_agent_id=1,
                claim_kind=CLAIM_CAUSALITY_HYPOTHESIS,
                payload=format_hypothesis_payload(
                    "INDEPENDENT_ROOT", 0, "DEADLOCK_SUSPECTED",
                    "deadlock orders"),
                source_event_ids=(),
                round=1)
        auditor_box = r.inboxes["auditor"]
        held = auditor_box.peek()
        self.assertEqual(len(held), 1)
        self.assertEqual(held[0].source_role, "db_admin")

    def test_expired_edge_drops_handoff(self):
        r = _mk_router()
        r.install_edge("db_admin", CLAIM_CAUSALITY_HYPOTHESIS,
                        ["auditor"], ttl_rounds=1)
        r.tick(1)  # expire
        r.emit(source_role="db_admin", source_agent_id=1,
                claim_kind=CLAIM_CAUSALITY_HYPOTHESIS,
                payload="kind=INDEPENDENT_ROOT idx=0",
                source_event_ids=(),
                round=2)
        auditor_box = r.inboxes["auditor"]
        self.assertEqual(len(auditor_box.peek()), 0)


class TestPayloadRoundtrip(unittest.TestCase):
    def test_roundtrip(self):
        payload = format_hypothesis_payload(
            "INDEPENDENT_ROOT", 0,
            upstream_kind="DEADLOCK_SUSPECTED",
            witness="deadlock on orders_payments")
        parsed = parse_hypothesis_payload(payload)
        self.assertEqual(parsed["kind"], "INDEPENDENT_ROOT")
        self.assertEqual(parsed["idx"], "0")
        self.assertEqual(parsed["upstream_kind"],
                          "DEADLOCK_SUSPECTED")
        self.assertEqual(parsed["witness"],
                          "deadlock on orders_payments")


class TestChainIntegrity(unittest.TestCase):
    def test_chain_ok_across_emits(self):
        r = _mk_router()
        r.install_edge("db_admin", CLAIM_CAUSALITY_HYPOTHESIS,
                        ["auditor"], ttl_rounds=2)
        for i in range(3):
            r.emit(source_role="db_admin", source_agent_id=0,
                    claim_kind=CLAIM_CAUSALITY_HYPOTHESIS,
                    payload=f"kind=UNCERTAIN idx={i}",
                    source_event_ids=(), round=1)
        self.assertTrue(r.verify())


if __name__ == "__main__":
    unittest.main()
