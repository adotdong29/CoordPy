"""Tests for core.workspace — Global Workspace top-k selector."""
from __future__ import annotations
import math
import unittest
import numpy as np
from vision_mvp.core.workspace import Workspace


class TestWorkspace(unittest.TestCase):
    def test_capacity_is_log_n(self):
        for n in (2, 10, 100, 1000, 100000):
            w = Workspace(n_agents=n)
            self.assertEqual(w.capacity(), max(1, math.ceil(math.log2(n))))

    def test_select_returns_top_k(self):
        w = Workspace(n_agents=16, epsilon=0.0)  # no exploration
        # Saliences: agent 0 lowest, agent 15 highest.
        sal = np.arange(16, dtype=float)
        admitted = w.select(sal, seed=1)
        self.assertEqual(len(admitted), w.capacity())
        # Top-k must all have high scores (>= len-k threshold)
        threshold = 16 - w.capacity()
        for idx in admitted:
            self.assertGreaterEqual(idx, threshold)

    def test_select_unique_indices(self):
        w = Workspace(n_agents=50, epsilon=0.0)
        rng = np.random.default_rng(0)
        sal = rng.standard_normal(50)
        admitted = w.select(sal, seed=0)
        self.assertEqual(len(set(admitted.tolist())), len(admitted))

    def test_select_shape_validation(self):
        w = Workspace(n_agents=10)
        with self.assertRaises(ValueError):
            w.select(np.zeros((2, 3)), seed=0)  # 2D
        with self.assertRaises(ValueError):
            w.select(np.zeros(5), seed=0)        # wrong length

    def test_select_when_k_exceeds_n(self):
        # Edge case: tiny team, workspace "capacity" might exceed N
        w = Workspace(n_agents=2)
        sal = np.array([1.0, 2.0])
        admitted = w.select(sal, seed=0)
        # Should return all agents, no crash
        self.assertLessEqual(len(admitted), 2)

    def test_epsilon_exploration_probabilistic(self):
        w = Workspace(n_agents=20, epsilon=1.0)  # always explore
        sal = np.arange(20, dtype=float)
        # With eps=1, the last slot is always replaced by random agent.
        # The rest should still be top-(k-1).
        admitted = w.select(sal, seed=3)
        self.assertEqual(len(admitted), w.capacity())
        self.assertEqual(len(set(admitted.tolist())), len(admitted))


if __name__ == "__main__":
    unittest.main()
