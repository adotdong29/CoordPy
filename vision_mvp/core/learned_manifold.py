"""Learned manifold — streaming PCA via EMA of covariance.

Agents don't know the intrinsic subspace. They discover it by maintaining
an exponential moving average of the sample covariance C(t), and taking
the top-m eigenvectors as the basis.

    C(t) = (1 - α) · C(t-1) + α · (1/n) X^T X

This is more stable than Oja's rule for our setting (d small ≈ 64 so
eigendecomposition is cheap, ~30μs). The EMA tracks drift automatically
via α — small α = slow adaptation, large α = fast adaptation to new data.

The `basis` attribute gives the current top-m subspace. `subspace_alignment`
measures how well the learned subspace agrees with a known ground-truth
basis (for experiments only; real systems wouldn't have this).
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field


@dataclass
class StreamingPCA:
    dim_input: int
    dim_manifold: int
    lr: float = 0.1                  # EMA coefficient α
    _C: np.ndarray = None            # type: ignore  (d, d) covariance estimate
    _basis: np.ndarray = None        # type: ignore  (d, m)
    _n_updates: int = 0

    @classmethod
    def build(cls, dim_input: int, dim_manifold: int,
              lr: float = 0.1, seed: int = 0) -> "StreamingPCA":
        rng = np.random.default_rng(seed)
        # Initial basis: random orthonormal
        A = rng.standard_normal((dim_input, dim_manifold))
        Q, _ = np.linalg.qr(A)
        obj = cls(dim_input=dim_input, dim_manifold=dim_manifold, lr=lr,
                  _C=np.eye(dim_input) * 0.01, _basis=Q)
        return obj

    def update_batch(self, X: np.ndarray) -> None:
        """EMA update from a batch (rows of X are samples)."""
        C_sample = X.T @ X / max(len(X), 1)
        self._C = (1 - self.lr) * self._C + self.lr * C_sample
        # Refresh basis via eigendecomposition (top m eigenvectors)
        # eigh returns in ascending order — take the last m columns
        w, V = np.linalg.eigh(self._C)
        self._basis = V[:, -self.dim_manifold:].copy()
        self._n_updates += 1

    def update(self, x: np.ndarray) -> None:
        self.update_batch(x[None, :])

    def project(self, x: np.ndarray) -> np.ndarray:
        return self._basis.T @ x

    def reconstruct(self, y: np.ndarray) -> np.ndarray:
        return self._basis @ y

    def subspace_alignment(self, true_basis: np.ndarray) -> float:
        """Return alignment in [0, 1] as mean principal-angle cosine.

        1 = learned subspace matches true subspace exactly;
        0 = fully orthogonal. Mean (not min) is more informative for
        partially-learned subspaces.
        """
        U = self._basis
        V = true_basis
        # Singular values of U^T V are cosines of principal angles
        s = np.linalg.svd(U.T @ V, compute_uv=False)
        s = np.clip(s, 0.0, 1.0)
        return float(s.mean())

    @property
    def basis(self) -> np.ndarray:
        return self._basis
