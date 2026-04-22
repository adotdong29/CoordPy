"""Tests for core/ctw_predictor.py."""

from __future__ import annotations

import math
import unittest

import numpy as np

from vision_mvp.core.ctw_predictor import CTW, CTWSurpriseDetector


class TestCTWBasics(unittest.TestCase):
    def test_initial_probability_is_half(self):
        ctw = CTW(depth=4)
        self.assertAlmostEqual(ctw.predict_prob_one(), 0.5, places=6)

    def test_symmetry_after_11(self):
        ctw = CTW(depth=3)
        ctw.observe(1)
        ctw.observe(1)
        # After observing two 1s, prediction for 1 should exceed 0.5.
        self.assertGreater(ctw.predict_prob_one(), 0.5)

    def test_symmetry_after_00(self):
        ctw = CTW(depth=3)
        ctw.observe(0)
        ctw.observe(0)
        self.assertLess(ctw.predict_prob_one(), 0.5)

    def test_surprise_bits_non_negative(self):
        ctw = CTW(depth=3)
        self.assertGreaterEqual(ctw.surprise_bits(0), 0.0)
        self.assertGreaterEqual(ctw.surprise_bits(1), 0.0)

    def test_predict_is_idempotent(self):
        # Two consecutive queries without observing should return the same value.
        ctw = CTW(depth=4)
        ctw.observe(1); ctw.observe(0); ctw.observe(1)
        p1 = ctw.predict_prob_one()
        p2 = ctw.predict_prob_one()
        self.assertAlmostEqual(p1, p2, places=9)

    def test_rejects_bad_symbol(self):
        ctw = CTW(depth=2)
        with self.assertRaises(ValueError):
            ctw.observe(2)
        with self.assertRaises(ValueError):
            ctw.surprise_bits(7)

    def test_rejects_negative_depth(self):
        with self.assertRaises(ValueError):
            CTW(depth=-1)


class TestCTWUniversality(unittest.TestCase):
    """Sanity-check: CTW approaches the entropy rate of its source."""

    def test_iid_bernoulli(self):
        # Source: Bernoulli(p=0.3). Entropy per symbol = H_2(0.3) ≈ 0.8813 bits.
        rng = np.random.default_rng(0)
        p = 0.3
        T = 3000
        stream = (rng.random(T) < p).astype(int).tolist()

        ctw = CTW(depth=4)
        bits = ctw.code_length(stream)
        avg = bits / T
        target = -p * math.log2(p) - (1 - p) * math.log2(1 - p)
        # CTW has redundancy O(log T / T) ≈ ~0.01 at T=3000; allow a margin.
        self.assertLess(avg, target + 0.1)
        # And it must be at least the entropy (up to 1-bit CTW slack at start).
        self.assertGreater(avg, target - 0.05)

    def test_periodic_source_has_low_rate(self):
        # Deterministic period-4 source: 1,1,0,0 repeated
        pattern = [1, 1, 0, 0] * 500
        ctw = CTW(depth=6)
        bits = ctw.code_length(pattern)
        avg = bits / len(pattern)
        # Entropy rate is 0. CTW should drop well below 0.2 bits/symbol.
        self.assertLess(avg, 0.2)


class TestSurpriseDetector(unittest.TestCase):
    def test_detects_sign_flip(self):
        det = CTWSurpriseDetector(depth=4, surprise_bits=1.5)
        # After a long run of positive errors, a big negative should be
        # surprising to the detector.
        for _ in range(50):
            det.update(1.0)
        # The last update here is the "first negative in a run".
        is_sup = det.update(-5.0)
        self.assertTrue(is_sup)

    def test_quiet_stream_not_surprising(self):
        det = CTWSurpriseDetector(depth=4, surprise_bits=4.0)
        # Very high threshold — nothing should trip.
        for _ in range(200):
            det.update(1.0)
        # Final update after a long run: predictor is highly confident in sign=1.
        self.assertFalse(det.update(1.0))

    def test_threshold_positive(self):
        with self.assertRaises(ValueError):
            CTWSurpriseDetector(surprise_bits=0.0)


if __name__ == "__main__":
    unittest.main()
