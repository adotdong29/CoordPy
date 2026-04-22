"""Tests for predictor families — neural and vectorized bank."""
from __future__ import annotations
import unittest
import numpy as np
from vision_mvp.core.neural_predictor import NeuralPredictor
from vision_mvp.core.vectorized_predictor import PredictorBank


class TestNeuralPredictor(unittest.TestCase):
    def test_initial_prediction_is_identity(self):
        """With near-zero weights, residual ~ 0 so pred(x) ≈ x."""
        p = NeuralPredictor(dim=8, hidden=4, lr=0.01)
        x = np.ones(8)
        pred = p.predict(x)
        np.testing.assert_allclose(pred, x, atol=1e-1)

    def test_observe_returns_nonneg_surprise(self):
        p = NeuralPredictor(dim=4)
        s = p.observe(np.zeros(4), np.array([1., 0., 0., 0.]))
        self.assertGreaterEqual(s, 0.0)

    def test_observe_reduces_surprise_over_time(self):
        """On a stationary target, prediction error should decrease."""
        rng = np.random.default_rng(0)
        p = NeuralPredictor(dim=8, hidden=16, lr=0.05)
        # Learn to predict x_t+1 = x_t + delta where delta is fixed
        delta = 0.1 * rng.standard_normal(8)
        x = np.zeros(8)
        errs = []
        for _ in range(200):
            x_new = x + delta + 0.01 * rng.standard_normal(8)
            s = p.observe(x, x_new)
            errs.append(s)
            x = x_new
        early = np.mean(errs[:20])
        late = np.mean(errs[-20:])
        self.assertLess(late, early)

    def test_weights_clipped(self):
        """After many observations, weights stay bounded."""
        rng = np.random.default_rng(0)
        p = NeuralPredictor(dim=4, hidden=4, lr=1.0)  # aggressive lr
        x = rng.standard_normal(4) * 10
        for _ in range(20):
            p.observe(x, rng.standard_normal(4) * 10)
            x = rng.standard_normal(4)
        # After __post_init__ seeds weights, all in [-1, 1]
        self.assertLessEqual(np.max(np.abs(p._W1)), 1.0 + 1e-8)
        self.assertLessEqual(np.max(np.abs(p._W2)), 1.0 + 1e-8)


class TestPredictorBank(unittest.TestCase):
    def test_forward_shape(self):
        bank = PredictorBank.build(n_agents=5, dim=8, hidden=4)
        X = np.zeros((5, 8))
        Y = bank.predict(X)
        self.assertEqual(Y.shape, (5, 8))

    def test_surprise_shape(self):
        bank = PredictorBank.build(n_agents=7, dim=6, hidden=3)
        prev = np.zeros((7, 6))
        now = np.ones((7, 6))
        s = bank.observe(prev, now)
        self.assertEqual(s.shape, (7,))

    def test_agents_update_independently(self):
        """Agent 0 sees constant data, agent 1 sees drifting data;
        after many updates, agent 0's weights should be nearly unchanged
        and agent 1's should deviate."""
        bank = PredictorBank.build(n_agents=2, dim=4, hidden=4, lr=0.1)
        W1_0 = bank._W1[0].copy()
        # Agent 0: prev = now (no change needed)
        # Agent 1: constant non-zero delta
        for _ in range(50):
            prev = np.zeros((2, 4))
            now = np.zeros((2, 4))
            now[1] = np.array([1., 0., 0., 0.])
            bank.observe(prev, now)
        # Agent 0's weights should barely have moved (gradient is near zero)
        drift_0 = np.linalg.norm(bank._W1[0] - W1_0)
        # Agent 1's weights should have moved more
        drift_1 = np.linalg.norm(bank._W1[1] - W1_0)  # init was same seed
        self.assertLess(drift_0, drift_1)

    def test_match_neural_predictor_single_agent(self):
        """A 1-agent bank should produce identical predictions to a single
        NeuralPredictor with matched init — catching vectorization bugs."""
        # Seed the same, init the same way
        single = NeuralPredictor(dim=4, hidden=3, lr=0.05)
        bank = PredictorBank.build(n_agents=1, dim=4, hidden=3, lr=0.05, seed=0)
        # Copy bank's weights into single predictor, so they match exactly.
        single._init_weights(seed=0)
        # Just copy bank's weights into single:
        single._W1 = bank._W1[0].copy()
        single._b1 = bank._b1[0].copy()
        single._W2 = bank._W2[0].copy()
        single._b2 = bank._b2[0].copy()
        single._seeded = True
        x = np.array([1., 2., 3., 4.])
        pred_single = single.predict(x)
        pred_bank = bank.predict(x[None, :])[0]
        np.testing.assert_allclose(pred_single, pred_bank, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
