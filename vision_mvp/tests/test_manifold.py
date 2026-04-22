"""Tests for core.manifold — Shared Latent Manifold."""
from __future__ import annotations
import math
import unittest
import numpy as np
from vision_mvp.core.manifold import Manifold


class TestManifold(unittest.TestCase):
    def test_default_dim_is_log_n(self):
        m = Manifold.build(dim_input=64, n_agents=1024, seed=0)
        # ceil(log2(1024)) = 10
        self.assertEqual(m.dim_manifold, 10)

    def test_project_shape(self):
        m = Manifold.build(dim_input=16, n_agents=16, seed=0)
        v = np.arange(16, dtype=float)
        p = m.project(v)
        self.assertEqual(p.shape, (m.dim_manifold,))

    def test_write_read_round_trip_on_single_sample(self):
        m = Manifold.build(dim_input=8, n_agents=8, seed=0)
        x = np.array([1.0, 0, 0, 0, 0, 0, 0, 0])
        proj = m.project(x)
        m.write(proj, weight=1.0)
        summary = m.read()
        np.testing.assert_allclose(summary, proj)

    def test_write_accumulates_weighted_mean(self):
        m = Manifold.build(dim_input=8, n_agents=8, seed=1)
        p1 = m.project(np.ones(8))
        p2 = m.project(np.zeros(8))
        m.write(p1, weight=2.0)
        m.write(p2, weight=2.0)
        # Mean should be p1 * 0.5 + p2 * 0.5
        np.testing.assert_allclose(m.read(), 0.5 * p1 + 0.5 * p2, rtol=1e-6)

    def test_reconstruct_matches_basis_projection(self):
        # With an orthonormal basis, project + reconstruct recovers
        # the component of x inside the basis span.
        rng = np.random.default_rng(42)
        d, m_dim = 10, 4
        A = rng.standard_normal((d, m_dim))
        Q, _ = np.linalg.qr(A)
        man = Manifold.build(dim_input=d, n_agents=2, basis=Q)
        x = Q @ np.array([1.0, 2.0, 3.0, 4.0])  # lies in span(Q)
        p = man.project(x)
        rec = man.reconstruct(p)
        np.testing.assert_allclose(rec, x, atol=1e-10)

    def test_write_cost_matches_dim(self):
        m = Manifold.build(dim_input=4, n_agents=32, seed=0)
        self.assertEqual(m.write_cost(), m.dim_manifold)
        self.assertEqual(m.read_cost(), m.dim_manifold)

    def test_basis_shape_validation(self):
        bad = np.zeros((3, 2))  # d=3 but we claim 8
        with self.assertRaises(ValueError):
            Manifold.build(dim_input=8, n_agents=10, basis=bad)

    def test_read_before_write_returns_zero(self):
        m = Manifold.build(dim_input=8, n_agents=16)
        np.testing.assert_allclose(m.read(), np.zeros(m.dim_manifold))


if __name__ == "__main__":
    unittest.main()
