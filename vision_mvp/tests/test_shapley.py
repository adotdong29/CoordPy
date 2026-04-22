"""Tests for core/shapley.py."""

from __future__ import annotations

import unittest

import numpy as np

from vision_mvp.core.shapley import (
    efficiency_residual,
    exact_shapley,
    monte_carlo_shapley,
    samples_for_accuracy,
)


class TestExactShapley(unittest.TestCase):
    def test_unanimity_game(self):
        # Unanimity game on subset T: v(S) = 1 iff T ⊆ S, else 0.
        # Shapley: 1/|T| for members of T, 0 otherwise.
        T = frozenset({0, 1})

        def v(S):
            return 1.0 if T.issubset(S) else 0.0

        phi = exact_shapley(v, n=3)
        np.testing.assert_allclose(phi, [0.5, 0.5, 0.0], atol=1e-9)

    def test_additivity(self):
        # v(S) = |S| → every agent contributes exactly 1
        phi = exact_shapley(lambda S: float(len(S)), n=4)
        np.testing.assert_allclose(phi, np.ones(4), atol=1e-9)

    def test_efficiency(self):
        # Arbitrary bounded v — sum must equal v(N).
        def v(S):
            if not S:
                return 0.0
            return sum(S) * 1.0 + float(len(S)) ** 2

        n = 5
        phi = exact_shapley(v, n=n)
        v_full = v(frozenset(range(n)))
        self.assertAlmostEqual(phi.sum(), v_full, places=8)

    def test_symmetry(self):
        # symmetric agents get equal credit
        def v(S):
            return float(len(S)) ** 2

        phi = exact_shapley(v, n=4)
        self.assertAlmostEqual(phi.std(), 0.0, places=8)

    def test_rejects_bad_n(self):
        with self.assertRaises(ValueError):
            exact_shapley(lambda S: 0.0, n=0)


class TestMonteCarloShapley(unittest.TestCase):
    def test_matches_exact_on_small(self):
        def v(S):
            return float(len(S) * sum(S, 0)) if S else 0.0

        n = 4
        phi_exact = exact_shapley(v, n=n)
        phi_mc = monte_carlo_shapley(v, n=n, k_samples=5000, seed=0)
        np.testing.assert_allclose(phi_mc, phi_exact, atol=0.4)

    def test_efficiency_asymptotic(self):
        # Efficiency holds exactly for *every* MC sample by telescoping.
        def v(S):
            return float(len(S)) ** 1.5 if S else 0.0

        n = 5
        v_full = v(frozenset(range(n)))
        phi = monte_carlo_shapley(v, n=n, k_samples=100, seed=1)
        self.assertAlmostEqual(phi.sum(), v_full, places=8)

    def test_deterministic_with_seed(self):
        def v(S):
            return float(len(S))

        a = monte_carlo_shapley(v, n=4, k_samples=100, seed=42)
        b = monte_carlo_shapley(v, n=4, k_samples=100, seed=42)
        np.testing.assert_array_equal(a, b)

    def test_rejects_bad_k(self):
        with self.assertRaises(ValueError):
            monte_carlo_shapley(lambda S: 0.0, n=3, k_samples=0)


class TestSampleSize(unittest.TestCase):
    def test_sample_size_grows_with_precision(self):
        k1 = samples_for_accuracy(n_agents=10, v_range=1.0, eps=0.1)
        k2 = samples_for_accuracy(n_agents=10, v_range=1.0, eps=0.01)
        self.assertGreater(k2, k1)

    def test_sample_size_grows_with_n(self):
        k1 = samples_for_accuracy(n_agents=5, v_range=1.0, eps=0.05)
        k2 = samples_for_accuracy(n_agents=5000, v_range=1.0, eps=0.05)
        self.assertGreater(k2, k1)

    def test_sample_size_rejects_bad_eps(self):
        with self.assertRaises(ValueError):
            samples_for_accuracy(5, 1.0, -0.1)


class TestEfficiencyResidual(unittest.TestCase):
    def test_trivial(self):
        self.assertAlmostEqual(
            efficiency_residual(np.array([0.3, 0.7]), 1.0), 0.0,
        )


if __name__ == "__main__":
    unittest.main()
