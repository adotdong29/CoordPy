"""Tests for core/pac_bayes.py."""

from __future__ import annotations

import unittest

import numpy as np

from vision_mvp.core.pac_bayes import (
    catoni_bound,
    kl_gaussian_diag,
    kl_gaussian_full,
    mcallester_bound,
    optimal_beta_catoni,
    optimal_beta_mcallester,
    pac_bayes_report,
)


class TestKLGaussian(unittest.TestCase):
    def test_kl_identical_is_zero(self):
        mu = np.array([0.0, 1.0, -2.0])
        var = np.array([1.0, 2.0, 0.5])
        self.assertAlmostEqual(kl_gaussian_diag(mu, var, mu, var), 0.0, places=8)

    def test_kl_nonneg(self):
        rng = np.random.default_rng(0)
        for _ in range(10):
            mu_q, mu_p = rng.standard_normal(4), rng.standard_normal(4)
            var_q, var_p = rng.uniform(0.3, 3.0, 4), rng.uniform(0.3, 3.0, 4)
            self.assertGreaterEqual(
                kl_gaussian_diag(mu_q, var_q, mu_p, var_p), -1e-9,
            )

    def test_kl_full_matches_diag(self):
        mu_q = np.array([1.0, 2.0])
        mu_p = np.array([0.0, 0.0])
        var_q = np.array([0.5, 2.0])
        var_p = np.array([1.0, 1.0])
        diag_kl = kl_gaussian_diag(mu_q, var_q, mu_p, var_p)
        full_kl = kl_gaussian_full(
            mu_q, np.diag(var_q), mu_p, np.diag(var_p),
        )
        self.assertAlmostEqual(diag_kl, full_kl, places=6)

    def test_kl_diag_rejects_nonpositive_var(self):
        with self.assertRaises(ValueError):
            kl_gaussian_diag(
                np.zeros(2), np.array([1.0, -0.5]),
                np.zeros(2), np.ones(2),
            )

    def test_kl_full_rejects_singular(self):
        with self.assertRaises(ValueError):
            kl_gaussian_full(
                np.zeros(2), np.zeros((2, 2)),
                np.zeros(2), np.eye(2),
            )


class TestMcAllester(unittest.TestCase):
    def test_bound_exceeds_empirical_risk(self):
        b = mcallester_bound(emp_risk=0.1, kl=5.0, n=1000, delta=0.05)
        self.assertGreater(b, 0.1)

    def test_bound_shrinks_with_n(self):
        b1 = mcallester_bound(0.1, 10.0, 100)
        b2 = mcallester_bound(0.1, 10.0, 10000)
        self.assertGreater(b1, b2)

    def test_bound_grows_with_kl(self):
        b1 = mcallester_bound(0.1, 1.0, 1000)
        b2 = mcallester_bound(0.1, 100.0, 1000)
        self.assertGreater(b2, b1)

    def test_rejects_bad_inputs(self):
        with self.assertRaises(ValueError):
            mcallester_bound(0.1, 1.0, 1000, delta=0.0)
        with self.assertRaises(ValueError):
            mcallester_bound(0.1, 1.0, 1)
        with self.assertRaises(ValueError):
            mcallester_bound(0.1, -1.0, 1000)


class TestOptimalBetaMcAllester(unittest.TestCase):
    def test_beta_grows_with_n(self):
        # Larger n → tighter bound → higher β*
        b1 = optimal_beta_mcallester(kl=10.0, n=100)
        b2 = optimal_beta_mcallester(kl=10.0, n=10000)
        self.assertGreater(b2, b1)

    def test_beta_shrinks_with_kl(self):
        b1 = optimal_beta_mcallester(kl=1.0, n=1000)
        b2 = optimal_beta_mcallester(kl=100.0, n=1000)
        self.assertGreater(b1, b2)

    def test_beta_positive(self):
        self.assertGreater(optimal_beta_mcallester(kl=0.0, n=100), 0.0)


class TestCatoni(unittest.TestCase):
    def test_catoni_bound_monotone_decreasing_in_n(self):
        b1 = catoni_bound(0.1, 5.0, n=100, lam=1.0)
        b2 = catoni_bound(0.1, 5.0, n=10000, lam=1.0)
        self.assertGreater(b1, b2)

    def test_optimal_beta_catoni(self):
        lam_star, bound_star = optimal_beta_catoni(
            emp_risk=0.1, kl=5.0, n=1000, delta=0.05,
        )
        self.assertGreater(lam_star, 0)
        self.assertGreater(bound_star, 0.1)
        # Any other λ should give bound ≥ bound*
        for lam in [0.1, 0.5, 2.0, 10.0]:
            if abs(lam - lam_star) < 0.05:
                continue
            b = catoni_bound(0.1, 5.0, 1000, lam, 0.05)
            self.assertGreaterEqual(b, bound_star - 1e-6)


class TestReport(unittest.TestCase):
    def test_report_sane(self):
        r = pac_bayes_report(emp_risk=0.2, kl=3.0, n=1000, delta=0.05)
        self.assertGreater(r.bound, 0.2)
        self.assertGreater(r.optimal_beta, 0.0)
        self.assertIn("β*", r.summary())


if __name__ == "__main__":
    unittest.main()
