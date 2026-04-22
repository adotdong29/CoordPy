"""Tests for Phase-6 additions: market, shared randomness, continuous scale."""
from __future__ import annotations
import math
import unittest
import numpy as np
from vision_mvp.core.market import MarketWorkspace, salience_to_bid
from vision_mvp.core.shared_randomness import SharedRNG, DeltaChannel, naive_bits
from vision_mvp.core.continuous_scale import (
    dim_at_scale, ContinuousScaleProjector, AdaptiveScale)


class TestMarketWorkspace(unittest.TestCase):
    def test_default_capacity(self):
        mw = MarketWorkspace(n_agents=1024)
        self.assertEqual(mw.capacity(), 10)

    def test_clear_returns_top_k(self):
        mw = MarketWorkspace(n_agents=8, k=3)
        bids = np.array([1., 5., 2., 9., 4., 0., 3., 7.])
        admitted, prices = mw.clear_market(bids)
        self.assertEqual(len(admitted), 3)
        # Top-3 are agents 3 (bid 9), 7 (bid 7), 1 (bid 5)
        self.assertEqual(set(admitted.tolist()), {3, 7, 1})

    def test_vcg_price_is_uniform(self):
        mw = MarketWorkspace(n_agents=6, k=2)
        bids = np.array([10., 8., 3., 2., 1., 0.])
        admitted, prices = mw.clear_market(bids)
        # 3rd-highest bid = 3, uniform VCG price
        np.testing.assert_allclose(prices, [3., 3.])

    def test_bid_truthful_under_budget(self):
        self.assertEqual(salience_to_bid(5.0, agent_budget=10.0), 5.0)
        self.assertEqual(salience_to_bid(15.0, agent_budget=10.0), 10.0)


class TestSharedRandomness(unittest.TestCase):
    def test_encode_decode_round_trip(self):
        chan = DeltaChannel(seed=42, dim=8)
        x = np.array([1., 2., 3., 4., 5., 6., 7., 8.])
        y = chan.send(x)
        np.testing.assert_allclose(y, x, atol=1e-10)

    def test_delta_smaller_than_naive_when_value_is_predicted(self):
        # If the "value" is exactly the expected stream, delta is zero.
        rng1 = SharedRNG(seed=0, dim=100)
        rng2 = SharedRNG(seed=0, dim=100)   # separate RNG for naive-style
        expected = rng1.expected_next()
        # Sender now encodes the same value — delta should be exactly 0.
        rng_send = SharedRNG(seed=0, dim=100)
        rng_send.expected_next()            # advance to match rng1
        # Actually simpler: feed the expected back as the "value"
        chan = DeltaChannel(seed=1, dim=100, precision=1e-3)
        # Value equals the channel's own internal expected
        preview = SharedRNG(seed=1, dim=100)
        expected_val = preview.expected_next()
        bits_delta = SharedRNG.encoded_bits(chan._sender.encode(expected_val))
        bits_naive = naive_bits(expected_val)
        self.assertLessEqual(bits_delta, bits_naive)

    def test_two_rngs_same_seed_same_sequence(self):
        a = SharedRNG(seed=7, dim=5)
        b = SharedRNG(seed=7, dim=5)
        for _ in range(3):
            np.testing.assert_allclose(a.expected_next(), b.expected_next())


class TestContinuousScale(unittest.TestCase):
    def test_dim_at_scale_0_is_full(self):
        self.assertEqual(dim_at_scale(64, 0.0), 64)

    def test_dim_at_scale_doubles_halving(self):
        # Every +1 in scale halves (ceiling) the dim.
        self.assertEqual(dim_at_scale(64, 1.0), 32)
        self.assertEqual(dim_at_scale(64, 2.0), 16)
        self.assertEqual(dim_at_scale(64, 3.0), 8)
        self.assertEqual(dim_at_scale(64, 6.0), 1)

    def test_projector_round_trip_in_top_dims(self):
        d = 10
        rng = np.random.default_rng(0)
        B, _ = np.linalg.qr(rng.standard_normal((d, d)))
        proj = ContinuousScaleProjector(d_input=d, basis=B, scale=0.0)
        x = rng.standard_normal(d)
        y = proj.project(x)
        rec = proj.reconstruct(y)
        # At scale 0, projection is identity basis change: round-trip exact
        np.testing.assert_allclose(rec, x, atol=1e-10)

    def test_projector_coarser_scale_smaller_dim(self):
        d = 32
        rng = np.random.default_rng(1)
        B, _ = np.linalg.qr(rng.standard_normal((d, d)))
        proj_fine = ContinuousScaleProjector(d_input=d, basis=B, scale=0.0)
        proj_coarse = ContinuousScaleProjector(d_input=d, basis=B, scale=3.0)
        x = rng.standard_normal(d)
        self.assertLess(len(proj_coarse.project(x)), len(proj_fine.project(x)))

    def test_adaptive_scale_lowers_on_high_surprise(self):
        a = AdaptiveScale(current_scale=2.0, high_threshold=0.3,
                          low_threshold=0.05, lr=0.5)
        # Feed high surprise repeatedly — scale should drop
        start = a.current_scale
        for _ in range(20):
            a.update(5.0)
        self.assertLess(a.current_scale, start)

    def test_adaptive_scale_raises_on_low_surprise(self):
        a = AdaptiveScale(current_scale=1.0, high_threshold=0.5,
                          low_threshold=0.5, lr=0.5, max_scale=4.0)
        a._ema_surprise = 0.05   # initialize low (below low_threshold)
        start = a.current_scale
        for _ in range(10):
            a.update(0.0)
        self.assertGreater(a.current_scale, start)


if __name__ == "__main__":
    unittest.main()
