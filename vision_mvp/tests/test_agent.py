"""Tests for core.agent — Bayesian-updating agent state."""
from __future__ import annotations
import unittest
import numpy as np
from vision_mvp.core.agent import Agent


class TestAgent(unittest.TestCase):
    def test_initial_estimate_equals_observation(self):
        obs = np.array([1.0, 2.0, 3.0])
        a = Agent(agent_id=0, observation=obs)
        np.testing.assert_allclose(a.estimate, obs)
        self.assertEqual(a.accumulated_weight, 1.0)

    def test_bayesian_update_symmetric(self):
        # Two equal-weight observations average cleanly
        a = Agent(agent_id=0, observation=np.array([0.0]), obs_weight=1.0)
        a.bayesian_update(np.array([2.0]), other_weight=1.0)
        np.testing.assert_allclose(a.estimate, [1.0])
        self.assertEqual(a.accumulated_weight, 2.0)

    def test_bayesian_update_asymmetric(self):
        a = Agent(agent_id=0, observation=np.array([0.0]), obs_weight=1.0)
        a.bayesian_update(np.array([4.0]), other_weight=3.0)
        # Weighted mean: (1·0 + 3·4) / 4 = 3.0
        np.testing.assert_allclose(a.estimate, [3.0])
        self.assertEqual(a.accumulated_weight, 4.0)

    def test_zero_weight_noop(self):
        a = Agent(agent_id=0, observation=np.array([5.0]), obs_weight=1.0)
        a.bayesian_update(np.array([99.0]), other_weight=0.0)
        np.testing.assert_allclose(a.estimate, [5.0])

    def test_remember_and_forget(self):
        a = Agent(agent_id=0, observation=np.array([0.0]))
        a.remember("obs", 10)
        a.remember("peer", 20)
        self.assertEqual(a.current_context_tokens(), 30)
        a.forget_all()
        self.assertEqual(a.current_context_tokens(), 0)

    def test_disagreement_metric(self):
        a = Agent(agent_id=0, observation=np.array([0.0, 0.0]))
        b = Agent(agent_id=1, observation=np.array([3.0, 4.0]))
        self.assertAlmostEqual(a.disagreement(b), 5.0)


if __name__ == "__main__":
    unittest.main()
