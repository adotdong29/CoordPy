"""Tests for the public API — CASRRouter."""
from __future__ import annotations
import math
import unittest
import numpy as np
from vision_mvp import CASRRouter


class TestCASRRouterBasic(unittest.TestCase):
    def test_constructor_defaults(self):
        r = CASRRouter(n_agents=100, state_dim=32)
        # manifold dim defaults to ⌈log₂ N⌉ = 7
        self.assertEqual(r._manifold_dim, 7)

    def test_construct_rejects_tiny_n(self):
        with self.assertRaises(ValueError):
            CASRRouter(n_agents=1, state_dim=8)

    def test_step_shape(self):
        r = CASRRouter(n_agents=20, state_dim=8, task_rank=4)
        obs = np.random.default_rng(0).standard_normal((20, 8))
        est = r.step(obs)
        self.assertEqual(est.shape, (20, 8))

    def test_step_wrong_shape_raises(self):
        r = CASRRouter(n_agents=10, state_dim=4)
        with self.assertRaises(ValueError):
            r.step(np.zeros((5, 4)))   # wrong N
        with self.assertRaises(ValueError):
            r.step(np.zeros((10, 3)))  # wrong d
        with self.assertRaises(ValueError):
            r.step(np.zeros(10))       # 1D

    def test_get_estimate(self):
        r = CASRRouter(n_agents=5, state_dim=4, task_rank=2)
        obs = np.ones((5, 4))
        r.step(obs)
        e = r.get_estimate(2)
        self.assertEqual(e.shape, (4,))

    def test_get_estimate_before_step_raises(self):
        r = CASRRouter(n_agents=5, state_dim=4, task_rank=2)
        with self.assertRaises(RuntimeError):
            r.get_estimate(0)

    def test_stats_contains_expected_fields(self):
        r = CASRRouter(n_agents=10, state_dim=8, task_rank=3)
        r.step(np.ones((10, 8)))
        s = r.stats
        for k in ("peak_context_per_agent", "total_tokens",
                  "manifold_dim", "workspace_size", "rounds_executed"):
            self.assertIn(k, s)
        self.assertEqual(s["rounds_executed"], 1)

    def test_peak_context_equals_manifold_dim(self):
        """Core theoretical claim — via public API."""
        for n in (20, 100, 500):
            r = CASRRouter(n_agents=n, state_dim=32)
            obs = np.random.default_rng(0).standard_normal((n, 32))
            for _ in range(3):
                r.step(obs)
            self.assertEqual(
                r.stats["peak_context_per_agent"],
                r.stats["manifold_dim"],
                f"N={n}: peak != manifold_dim"
            )
            # manifold_dim should also equal log₂ N
            self.assertEqual(r.stats["manifold_dim"],
                             max(2, math.ceil(math.log2(n))))

    def test_reset_clears_state(self):
        r = CASRRouter(n_agents=5, state_dim=4, task_rank=2)
        r.step(np.ones((5, 4)))
        self.assertEqual(r.stats["rounds_executed"], 1)
        r.reset()
        self.assertEqual(r.stats["rounds_executed"], 0)
        self.assertIsNone(r.estimates)

    def test_convergence_on_trivial_consensus(self):
        """All agents see the same clean observation — estimates should agree
        within a round and match the observation."""
        r = CASRRouter(n_agents=10, state_dim=4, task_rank=2,
                       observation_noise=0.1, decay=0.7)
        obs = np.tile(np.array([1., 2., 3., 4.]), (10, 1))
        for _ in range(10):
            est = r.step(obs)
        # All estimates should be very close to each other
        pairwise = np.std(est, axis=0)
        self.assertLess(float(np.linalg.norm(pairwise)), 0.5)


class TestCASRRouterDeterminism(unittest.TestCase):
    def test_same_seed_same_output(self):
        obs_seq = [np.random.default_rng(i).standard_normal((10, 4)) for i in range(3)]
        r1 = CASRRouter(n_agents=10, state_dim=4, task_rank=3, seed=42)
        r2 = CASRRouter(n_agents=10, state_dim=4, task_rank=3, seed=42)
        for obs in obs_seq:
            e1 = r1.step(obs)
            e2 = r2.step(obs)
            np.testing.assert_allclose(e1, e2)


if __name__ == "__main__":
    unittest.main()
