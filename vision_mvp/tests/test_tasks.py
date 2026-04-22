"""Tests for the task generators — static and drifting consensus."""
from __future__ import annotations
import unittest
import numpy as np
from vision_mvp.tasks.consensus import ConsensusTask
from vision_mvp.tasks.drifting_consensus import DriftingConsensus


class TestConsensusTask(unittest.TestCase):
    def test_observation_shape(self):
        t = ConsensusTask(n_agents=10, dim=8, seed=1)
        t.generate()
        self.assertEqual(t.observations.shape, (10, 8))

    def test_truth_norm_nontrivial(self):
        t = ConsensusTask(n_agents=10, dim=8, seed=1)
        t.generate()
        self.assertGreater(np.linalg.norm(t.truth), 0)

    def test_evaluate_oracle_is_lower_bound(self):
        """Any random estimate performs worse than the mean oracle."""
        t = ConsensusTask(n_agents=20, dim=16, seed=2)
        t.generate()
        # Random-guess estimates
        rng = np.random.default_rng(0)
        bad = rng.standard_normal((20, 16)) * 5
        metrics = t.evaluate(bad)
        self.assertGreaterEqual(metrics["mean_accuracy_error"],
                                metrics["oracle_error"])

    def test_low_rank_has_basis(self):
        t = ConsensusTask(n_agents=50, dim=32, intrinsic_rank=8, seed=0)
        t.generate()
        self.assertIsNotNone(t.basis)
        self.assertEqual(t.basis.shape, (32, 8))


class TestDriftingConsensus(unittest.TestCase):
    def test_trajectory_shape(self):
        t = DriftingConsensus(n_agents=5, dim=6, intrinsic_rank=2,
                              n_steps=10, seed=1)
        t.generate()
        self.assertEqual(t.trajectory.shape, (10, 6))

    def test_observations_at_each_step(self):
        t = DriftingConsensus(n_agents=5, dim=6, intrinsic_rank=2,
                              n_steps=10, seed=1)
        t.generate()
        obs0 = t.observations_at(0)
        obs9 = t.observations_at(9)
        self.assertEqual(obs0.shape, (5, 6))
        self.assertEqual(obs9.shape, (5, 6))
        # Different steps should have different observations (with high prob)
        self.assertFalse(np.allclose(obs0, obs9))

    def test_shock_increases_magnitude(self):
        without_shock = DriftingConsensus(n_agents=5, dim=4, intrinsic_rank=2,
                                           n_steps=20, drift_sigma=0.01,
                                           shock_at=None, seed=3)
        without_shock.generate()
        with_shock = DriftingConsensus(n_agents=5, dim=4, intrinsic_rank=2,
                                        n_steps=20, drift_sigma=0.01,
                                        shock_at=10, shock_magnitude=5.0,
                                        seed=3)
        with_shock.generate()
        # Shock-induced trajectory should have a bigger jump at t=10
        jump_without = np.linalg.norm(
            without_shock.trajectory[10] - without_shock.trajectory[9])
        jump_with = np.linalg.norm(
            with_shock.trajectory[10] - with_shock.trajectory[9])
        self.assertGreater(jump_with, jump_without * 3)

    def test_evaluate_tracking_metrics(self):
        t = DriftingConsensus(n_agents=5, dim=6, intrinsic_rank=2,
                              n_steps=10, seed=1)
        t.generate()
        # Perfect tracking → zero error
        perfect = np.broadcast_to(t.trajectory[:, None, :], (10, 5, 6)).copy()
        metrics = t.evaluate_tracking(perfect)
        self.assertLess(metrics["mean_tracking_error"], 1e-6)


if __name__ == "__main__":
    unittest.main()
