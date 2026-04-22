"""Tests for core/event_triggered_control.py."""

from __future__ import annotations

import unittest

import numpy as np

from vision_mvp.core.event_triggered_control import (
    EventTrigger,
    simulate,
    solve_discrete_lyapunov,
    spectral_radius,
    synthesize_threshold,
)


class TestSpectral(unittest.TestCase):
    def test_identity_spectral_radius(self):
        self.assertAlmostEqual(spectral_radius(np.eye(3)), 1.0, places=6)

    def test_contracting_spectral_radius(self):
        A = 0.5 * np.eye(4)
        self.assertAlmostEqual(spectral_radius(A), 0.5, places=6)


class TestLyapunov(unittest.TestCase):
    def test_identity_times_half(self):
        A = 0.5 * np.eye(2)
        Q = np.eye(2)
        P = solve_discrete_lyapunov(A, Q)
        # AᵀPA − P = 0.25 P − P = −0.75 P. We want = −Q = −I.
        # So 0.75 P = I → P = (4/3) I.
        np.testing.assert_allclose(P, (4.0 / 3.0) * np.eye(2), atol=1e-6)

    def test_lyapunov_equation_satisfied(self):
        rng = np.random.default_rng(0)
        A = rng.standard_normal((3, 3))
        # Scale to Schur-stable
        rho = spectral_radius(A)
        A = A / (rho * 1.2)
        Q = np.eye(3)
        P = solve_discrete_lyapunov(A, Q)
        residual = A.T @ P @ A - P + Q
        self.assertLess(np.max(np.abs(residual)), 1e-6)

    def test_rejects_unstable_A(self):
        with self.assertRaises(ValueError):
            solve_discrete_lyapunov(2.0 * np.eye(2), np.eye(2))


class TestThresholdSynthesis(unittest.TestCase):
    def test_report_positive_sigma_star(self):
        A = 0.5 * np.eye(3)
        r = synthesize_threshold(A)
        self.assertGreater(r.sigma_star, 0.0)
        self.assertLess(r.spectral_radius, 1.0)

    def test_more_stable_gets_larger_sigma(self):
        # Faster decay → larger σ* (more broadcasts suppressible)
        r1 = synthesize_threshold(0.9 * np.eye(2))
        r2 = synthesize_threshold(0.2 * np.eye(2))
        self.assertGreater(r2.sigma_star, r1.sigma_star)

    def test_rejects_unstable(self):
        with self.assertRaises(ValueError):
            synthesize_threshold(1.5 * np.eye(2))


class TestEventTrigger(unittest.TestCase):
    def test_triggers_on_large_error(self):
        trig = EventTrigger(sigma=0.1)
        # ‖e‖²/‖x‖² = (0.5 / 1)² > 0.1? 0.25 > 0.1 → yes
        state = np.array([1.0, 0.0])
        last = np.array([0.5, 0.0])
        self.assertTrue(trig.should_broadcast(state, last))

    def test_no_trigger_on_small_error(self):
        trig = EventTrigger(sigma=0.5)
        state = np.array([1.0])
        last = np.array([0.9])
        # e²=0.01, x²=1, ratio 0.01 < 0.5 → no trigger
        self.assertFalse(trig.should_broadcast(state, last))

    def test_zero_state_zero_error(self):
        trig = EventTrigger(sigma=0.5)
        self.assertFalse(trig.should_broadcast(np.zeros(3), np.zeros(3)))

    def test_zero_state_nonzero_error_triggers(self):
        trig = EventTrigger(sigma=0.5)
        self.assertTrue(trig.should_broadcast(np.zeros(3), np.array([0.1, 0, 0])))


class TestSimulation(unittest.TestCase):
    def test_stable_system_converges(self):
        A = 0.7 * np.eye(2)
        r = synthesize_threshold(A)
        sigma = 0.5 * r.sigma_star  # strictly below σ*
        states, _, _ = simulate(A, np.array([1.0, 1.0]), sigma, n_steps=50)
        # Trajectory norm should decay to near zero.
        norms = np.linalg.norm(states, axis=1)
        self.assertLess(norms[-1], 0.05)

    def test_higher_sigma_fewer_broadcasts(self):
        A = 0.7 * np.eye(2)
        r = synthesize_threshold(A)
        _, _, n_low = simulate(A, np.array([1.0]*2), 0.1 * r.sigma_star, 50)
        _, _, n_high = simulate(A, np.array([1.0]*2), 0.8 * r.sigma_star, 50)
        self.assertGreaterEqual(n_low, n_high)


if __name__ == "__main__":
    unittest.main()
