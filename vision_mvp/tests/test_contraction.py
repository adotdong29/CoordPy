"""Tests for core/contraction.py."""

from __future__ import annotations

import math
import unittest

import numpy as np

from vision_mvp.core.contraction import (
    banach_convergence_estimate,
    central_difference_jacobian,
    contraction_report,
    is_contracting_region,
)


class TestJacobian(unittest.TestCase):
    def test_linear_map_jacobian_is_the_matrix(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        J = central_difference_jacobian(lambda x: A @ x, np.zeros(2))
        np.testing.assert_allclose(J, A, atol=1e-6)

    def test_nonlinear_map(self):
        # F(x, y) = (x², y² + x);  ∂F/∂(x,y) at (1,1) = [[2,0],[1,2]]
        F = lambda v: np.array([v[0] ** 2, v[1] ** 2 + v[0]])
        J = central_difference_jacobian(F, np.array([1.0, 1.0]))
        np.testing.assert_allclose(J, np.array([[2.0, 0.0], [1.0, 2.0]]), atol=1e-4)

    def test_rectangular_jacobian(self):
        # F: R^3 -> R^2 ;  J has shape (2, 3)
        F = lambda v: np.array([v[0] + v[1], v[1] * v[2]])
        J = central_difference_jacobian(F, np.array([1.0, 2.0, 3.0]))
        self.assertEqual(J.shape, (2, 3))


class TestContraction(unittest.TestCase):
    def test_contracting_linear_map(self):
        # Discrete iteration x -> 0.5 x is contracting with ρ = 0.5.
        # Note that "continuous contracting" is a separate condition on
        # λ_max(½(J+Jᵀ)) < 0 — it applies when F represents a flow ẋ=F(x),
        # not an iteration. For J = 0.5I the symmetric part is 0.5I > 0,
        # so the continuous flag is correctly False.
        A = 0.5 * np.eye(3)
        r = contraction_report(lambda x: A @ x, np.zeros(3))
        self.assertTrue(r.is_discrete_contracting)
        self.assertAlmostEqual(r.rate, 0.5, places=4)
        self.assertFalse(r.is_continuous_contracting)

    def test_continuous_contracting_flow(self):
        # Flow ẋ = -0.5 x: Jacobian = -0.5 I, sym part = -0.5 I, all eigs < 0.
        r = contraction_report(lambda x: -0.5 * x, np.zeros(2))
        self.assertTrue(r.is_continuous_contracting)

    def test_noncontracting_linear_map(self):
        A = 1.5 * np.eye(2)
        r = contraction_report(lambda x: A @ x, np.zeros(2))
        self.assertFalse(r.is_discrete_contracting)
        self.assertFalse(r.is_continuous_contracting)

    def test_region_check(self):
        A = 0.8 * np.eye(2)
        samples = np.random.default_rng(0).standard_normal((5, 2))
        ok, worst = is_contracting_region(lambda x: A @ x, samples)
        self.assertTrue(ok)
        self.assertAlmostEqual(worst, 0.8, places=4)

    def test_region_detects_nonuniform_contraction(self):
        # map contracts at origin but expands away
        def F(x):
            r2 = float(x @ x)
            return (0.5 + 2.0 * r2) * x

        samples = np.array([[0.0, 0.0], [5.0, 0.0]])
        ok, _ = is_contracting_region(F, samples)
        self.assertFalse(ok)


class TestBanachEstimate(unittest.TestCase):
    def test_standard_case(self):
        # rate 0.5, tol 1e-6 → need at least ceil(log(1e-6)/log(0.5)) = 20
        self.assertEqual(banach_convergence_estimate(0.5, 1e-6), 20)

    def test_rate_out_of_range(self):
        with self.assertRaises(ValueError):
            banach_convergence_estimate(1.5, 1e-3)
        with self.assertRaises(ValueError):
            banach_convergence_estimate(0.0, 1e-3)

    def test_tolerance_positive(self):
        with self.assertRaises(ValueError):
            banach_convergence_estimate(0.5, 0.0)


class TestCASRStepContraction(unittest.TestCase):
    """The real OQ1 payload: is the CASRRouter step() map a contraction?"""

    def test_consensus_iteration_is_contracting(self):
        # Model a simplified consensus step: estimates move a fraction toward
        # observations (contraction rate = 1 - forget). This mirrors the
        # `new_est = (1 - forget) * estimates + forget * obs` line in api.py.
        forget = 0.4
        N, d = 20, 4
        rng = np.random.default_rng(42)
        obs = rng.standard_normal((N, d))

        def step(flat_estimates):
            E = flat_estimates.reshape(N, d)
            return ((1 - forget) * E + forget * obs).ravel()

        r = contraction_report(step, np.zeros(N * d))
        self.assertTrue(r.is_discrete_contracting)
        # ρ should equal 1 - forget = 0.6
        self.assertAlmostEqual(r.rate, 1 - forget, places=3)


if __name__ == "__main__":
    unittest.main()
