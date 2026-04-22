"""Tests for the composed routers that wire Wave-1..5 primitives."""

from __future__ import annotations

import unittest

import numpy as np

from vision_mvp.core.composed_routers import (
    AdversarialCASRRouter, DynamicCASRRouter,
)


class TestAdversarialRouter(unittest.TestCase):
    def test_committee_size(self):
        r = AdversarialCASRRouter(n_agents=64)
        # log2(64) = 6
        self.assertEqual(r.committee_size, 6)

    def test_committee_consistency_same_round(self):
        r = AdversarialCASRRouter(n_agents=20)
        a = r.elect_workspace(7)
        b = r.elect_workspace(7)
        self.assertEqual(a, b)

    def test_committee_changes_across_rounds(self):
        r = AdversarialCASRRouter(n_agents=20)
        self.assertNotEqual(r.elect_workspace(1), r.elect_workspace(2))

    def test_dp_aggregate_close_to_true_mean(self):
        # Large ε and many samples → noise small, mean close to truth
        r = AdversarialCASRRouter(n_agents=10, epsilon_dp=50.0, delta_dp=1e-3)
        values = np.ones((20, 3)) * 5.0
        out = r.aggregate_dp(values)
        np.testing.assert_allclose(out, np.array([5.0, 5.0, 5.0]), atol=2.0)

    def test_safe_control_enforces_barrier(self):
        r = AdversarialCASRRouter(n_agents=4, barrier_safety=1.0)
        state = np.array([0.99])                   # near boundary
        nominal = np.array([100.0])                # big push into unsafe
        safe = r.safe_control(state, nominal)
        self.assertLess(safe[0], 100.0)


class TestDynamicRouter(unittest.TestCase):
    def test_join_and_route(self):
        r = DynamicCASRRouter()
        r.join("a", [])
        r.join("b", ["a"])
        r.join("c", ["a", "b"])
        self.assertIn(r.route("key1"), {"a", "b", "c"})

    def test_put_get_roundtrip(self):
        r = DynamicCASRRouter()
        r.put("k", 42)
        self.assertEqual(r.get("k"), 42)

    def test_broadcast_delivers(self):
        r = DynamicCASRRouter()
        r.join("a", [])
        r.join("b", ["a"])
        r.join("c", ["a", "b"])
        r.broadcast("a", "msg")
        self.assertGreater(r.reliability("msg"), 0.5)

    def test_leave(self):
        r = DynamicCASRRouter()
        r.join("a", [])
        r.join("b", ["a"])
        r.leave("a")
        # Only b remains
        self.assertEqual(r.route("anything"), "b")

    def test_event_tick_advances_clock(self):
        r = DynamicCASRRouter()
        c0 = r._clock
        r.event_tick()
        self.assertFalse(r._clock.leq(c0) and c0.leq(r._clock))


if __name__ == "__main__":
    unittest.main()
