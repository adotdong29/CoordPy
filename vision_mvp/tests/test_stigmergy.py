"""Tests for core.stigmergy — CRDT-style shared environment."""
from __future__ import annotations
import unittest
import numpy as np
from vision_mvp.core.stigmergy import Stigmergy, StigmergyBin


class TestStigmergyBin(unittest.TestCase):
    def test_merge_commutes(self):
        a = StigmergyBin(value=np.zeros(3), weight=0.0)
        b = StigmergyBin(value=np.zeros(3), weight=0.0)
        a.merge(np.array([1., 0., 0.]), 1.0)
        a.merge(np.array([0., 2., 0.]), 2.0)
        b.merge(np.array([0., 2., 0.]), 2.0)
        b.merge(np.array([1., 0., 0.]), 1.0)
        np.testing.assert_allclose(a.mean(), b.mean())
        self.assertEqual(a.weight, b.weight)

    def test_mean_of_empty_is_zero(self):
        b = StigmergyBin(value=np.zeros(3))
        np.testing.assert_allclose(b.mean(), np.zeros(3))


class TestStigmergy(unittest.TestCase):
    def test_bin_of_nearest_anchor(self):
        s = Stigmergy.build(n_bins=4, dim=2, seed=42)
        # The agent's own anchor should be its nearest bin.
        for i in range(s.n_bins):
            anchor = s._anchors[i]
            self.assertEqual(s.bin_of(anchor), i)

    def test_write_into_nearest_bin(self):
        s = Stigmergy.build(n_bins=4, dim=2, seed=42)
        anchor0 = s._anchors[0]
        s.write(coords=anchor0, value=np.array([5., 5.]), weight=1.0)
        # bin 0 now has weight > 0
        self.assertGreater(s._bins[0].weight, 0)

    def test_read_local_returns_nearby_mean(self):
        s = Stigmergy.build(n_bins=4, dim=2, seed=42)
        anchor0 = s._anchors[0]
        s.write(coords=anchor0, value=np.array([10., 0.]), weight=1.0)
        summary = s.read_local(anchor0, k=1)   # only nearest bin
        np.testing.assert_allclose(summary, [10., 0.])

    def test_read_cost_scales_with_k(self):
        s = Stigmergy.build(n_bins=10, dim=3)
        self.assertEqual(s.read_cost(1), 3)
        self.assertEqual(s.read_cost(4), 12)

    def test_write_cost_is_dim_plus_one(self):
        s = Stigmergy.build(n_bins=4, dim=5)
        self.assertEqual(s.write_cost(), 6)


if __name__ == "__main__":
    unittest.main()
