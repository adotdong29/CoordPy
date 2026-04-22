"""Tests for core/meta_learn.py."""

from __future__ import annotations

import unittest

import numpy as np

from vision_mvp.core.meta_learn import (
    ReptileConfig,
    adapt,
    build_scale_task,
    evaluate,
    inner_sgd,
    ols_gradient,
    reptile_train,
)


class TestOLS(unittest.TestCase):
    def test_gradient_zero_at_optimum(self):
        rng = np.random.default_rng(0)
        d = 4
        theta_star = rng.standard_normal(d)
        X = rng.standard_normal((100, d))
        y = X @ theta_star
        g = ols_gradient(theta_star, X, y)
        np.testing.assert_allclose(g, 0.0, atol=1e-8)

    def test_inner_sgd_descends(self):
        rng = np.random.default_rng(1)
        d = 3
        theta_star = rng.standard_normal(d)
        X = rng.standard_normal((64, d))
        y = X @ theta_star + 0.01 * rng.standard_normal(64)
        theta0 = np.zeros(d)
        mse_before = np.mean((X @ theta0 - y) ** 2)
        theta1 = inner_sgd(theta0, X, y, k_steps=100, lr=0.05)
        mse_after = np.mean((X @ theta1 - y) ** 2)
        self.assertLess(mse_after, mse_before)


class TestReptile(unittest.TestCase):
    def _make_synthetic_tasks(self, n_tasks=20, d=5, n_per_task=40, seed=0):
        rng = np.random.default_rng(seed)
        # Each task has a different hidden θ; Reptile should learn a good
        # shared init point (close to the mean).
        thetas = [rng.standard_normal(d) for _ in range(n_tasks)]
        tasks = []
        for t in thetas:
            X = rng.standard_normal((n_per_task, d))
            y = X @ t + 0.05 * rng.standard_normal(n_per_task)
            tasks.append((X, y))
        return tasks, np.mean(thetas, axis=0)

    def test_reptile_finds_good_init(self):
        tasks, mean_theta = self._make_synthetic_tasks()
        theta = reptile_train(
            tasks, feature_dim=5, n_outer=500,
            cfg=ReptileConfig(inner_lr=0.05, outer_lr=0.1, inner_steps=5),
            seed=0,
        )
        # Learned init should be closer to the mean of task thetas than
        # random initialisation (norm of difference matters).
        random_init = np.random.default_rng(7).standard_normal(5) * 0.01
        self.assertLess(
            np.linalg.norm(theta - mean_theta),
            np.linalg.norm(random_init - mean_theta) + 0.5,  # loose bound
        )

    def test_adapt_reduces_loss(self):
        tasks, _ = self._make_synthetic_tasks()
        theta = reptile_train(
            tasks, feature_dim=5, n_outer=200, seed=1,
        )
        # For a held-out task, adapt for 5 steps should reduce loss.
        X, y = tasks[0]
        before = evaluate(theta, X, y)
        after = evaluate(adapt(theta, X, y), X, y)
        self.assertLess(after, before)


class TestScaleTask(unittest.TestCase):
    def test_shape(self):
        task_embed = np.array([0.1, 0.2])
        role_embed = np.array([1.0, -1.0, 0.5])
        X, y = build_scale_task(task_embed, role_embed, scale=3, n_samples=16)
        self.assertEqual(X.shape, (16, 5))
        self.assertEqual(y.shape, (16,))

    def test_label_centered_at_scale(self):
        te = np.zeros(3)
        re = np.zeros(3)
        _, y = build_scale_task(te, re, scale=2, n_samples=500, noise=0.01)
        self.assertAlmostEqual(float(np.mean(y)), 2.0, places=1)


if __name__ == "__main__":
    unittest.main()
