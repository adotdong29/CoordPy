"""Tests for core/conformal.py."""

from __future__ import annotations

import unittest

import numpy as np

from vision_mvp.core.conformal import (
    OnlineConformal,
    calibrate,
    empirical_coverage,
    vectorized_contains,
)


class TestCalibrate(unittest.TestCase):
    def test_basic(self):
        rng = np.random.default_rng(0)
        preds = rng.standard_normal(200)
        obs = preds + rng.standard_normal(200) * 0.5
        cal = calibrate(preds, obs, alpha=0.1)
        self.assertGreater(cal.q_hat, 0.0)

    def test_coverage_guarantee(self):
        # The marginal coverage guarantee is Pr[y ∈ set] ≥ 1 − α
        # Empirically test with independent data
        rng = np.random.default_rng(1)
        n_cal = 500
        preds_cal = rng.standard_normal(n_cal)
        obs_cal = preds_cal + rng.standard_normal(n_cal)
        cal = calibrate(preds_cal, obs_cal, alpha=0.1)

        # Independent test set from same distribution
        n_test = 10000
        preds_test = rng.standard_normal(n_test)
        obs_test = preds_test + rng.standard_normal(n_test)
        cov = empirical_coverage(cal, preds_test, obs_test)
        # Allow a small slack below 0.9 because finite-sample guarantee is
        # with high probability, not a deterministic floor.
        self.assertGreater(cov, 0.88)

    def test_higher_alpha_smaller_interval(self):
        rng = np.random.default_rng(2)
        p = rng.standard_normal(300)
        o = p + rng.standard_normal(300)
        c1 = calibrate(p, o, alpha=0.05)
        c2 = calibrate(p, o, alpha=0.5)
        self.assertGreater(c1.q_hat, c2.q_hat)

    def test_contains(self):
        cal = calibrate(np.zeros(100), np.ones(100), alpha=0.1)
        # All scores are 1.0, so q_hat = 1.0. Obs at pred ± 0.5 is inside.
        self.assertTrue(cal.contains(0.0, 0.5))
        self.assertFalse(cal.is_surprising(0.0, 0.5))

    def test_rejects_bad_alpha(self):
        with self.assertRaises(ValueError):
            calibrate(np.zeros(10), np.zeros(10), alpha=0.0)
        with self.assertRaises(ValueError):
            calibrate(np.zeros(10), np.zeros(10), alpha=1.0)

    def test_shape_mismatch(self):
        with self.assertRaises(ValueError):
            calibrate(np.zeros(5), np.zeros(6))


class TestVectorized(unittest.TestCase):
    def test_vectorized_matches_scalar(self):
        cal = calibrate(np.zeros(50), np.ones(50) * 0.5, alpha=0.1)
        preds = np.array([0.0, 1.0, 2.0])
        obs = np.array([0.2, 1.6, 1.5])
        batched = vectorized_contains(cal, preds, obs)
        scalar = np.array([cal.contains(p, o) for p, o in zip(preds, obs)])
        np.testing.assert_array_equal(batched, scalar)


class TestOnlineConformal(unittest.TestCase):
    def test_online_empty_has_infinite_interval(self):
        oc = OnlineConformal(window=100, alpha=0.1)
        self.assertEqual(oc.q_hat, float("inf"))
        self.assertFalse(oc.is_surprising(0.0, 99.0))  # infinite band — never surprising

    def test_online_coverage(self):
        rng = np.random.default_rng(7)
        oc = OnlineConformal(window=300, alpha=0.1)
        # Warm up
        for _ in range(400):
            p = rng.standard_normal()
            y = p + rng.standard_normal() * 0.7
            oc.observe(p, y)

        # Test phase: measure miscoverage rate
        n_test = 2000
        miscover = 0
        for _ in range(n_test):
            p = rng.standard_normal()
            y = p + rng.standard_normal() * 0.7
            if oc.is_surprising(p, y):
                miscover += 1
            oc.observe(p, y)
        rate = miscover / n_test
        self.assertLess(rate, 0.15)  # should be ~0.1

    def test_online_rejects_bad_window(self):
        with self.assertRaises(ValueError):
            OnlineConformal(window=0)


if __name__ == "__main__":
    unittest.main()
