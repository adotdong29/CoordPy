"""Tests for core.learned_manifold — Streaming PCA."""
from __future__ import annotations
import unittest
import numpy as np
from vision_mvp.core.learned_manifold import StreamingPCA


class TestStreamingPCA(unittest.TestCase):
    def test_initial_basis_orthonormal(self):
        pca = StreamingPCA.build(dim_input=16, dim_manifold=4, seed=0)
        B = pca.basis
        # B.T @ B should be identity
        np.testing.assert_allclose(B.T @ B, np.eye(4), atol=1e-10)

    def test_project_reconstruct_on_basis_vector(self):
        pca = StreamingPCA.build(dim_input=8, dim_manifold=3, seed=1)
        # Vector in the learned basis span: project, reconstruct, compare
        y = np.array([1.0, 2.0, 3.0])
        x = pca.basis @ y
        p = pca.project(x)
        r = pca.reconstruct(p)
        np.testing.assert_allclose(r, x, atol=1e-10)

    def test_alignment_identity(self):
        pca = StreamingPCA.build(dim_input=6, dim_manifold=3, seed=0)
        # Alignment with itself = 1
        self.assertAlmostEqual(pca.subspace_alignment(pca.basis), 1.0, places=6)

    def test_alignment_orthogonal(self):
        rng = np.random.default_rng(7)
        pca = StreamingPCA.build(dim_input=6, dim_manifold=2, seed=0)
        # Build an orthogonal complement basis
        full, _ = np.linalg.qr(rng.standard_normal((6, 6)))
        learned = pca.basis  # (6, 2)
        # Find 2 columns of `full` orthogonal to learned.basis
        proj = full - learned @ (learned.T @ full)
        q, _ = np.linalg.qr(proj)
        orth = q[:, :2]
        # Alignment between orthogonal subspaces should be near 0.
        self.assertLess(pca.subspace_alignment(orth), 0.1)

    def test_converges_to_top_eigendirection(self):
        """Feed synthetic data with a known dominant direction; PCA must find it."""
        rng = np.random.default_rng(0)
        d = 10
        # True direction (unit vector)
        true_dir = rng.standard_normal(d)
        true_dir /= np.linalg.norm(true_dir)
        pca = StreamingPCA.build(dim_input=d, dim_manifold=1, lr=0.3, seed=5)
        # Feed in 100 samples aligned with true_dir plus small noise
        for _ in range(100):
            coeff = rng.standard_normal()
            x = coeff * true_dir + 0.05 * rng.standard_normal(d)
            pca.update(x)
        # Learned basis should nearly match true direction (up to sign)
        alignment = abs(float(pca.basis[:, 0] @ true_dir))
        self.assertGreater(alignment, 0.9)


if __name__ == "__main__":
    unittest.main()
