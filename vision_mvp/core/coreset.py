"""Bayesian coresets via Frank-Wolfe (Campbell & Broderick 2017).

A *coreset* is a weighted subset of data whose log-likelihood function
approximates that of the full data. Frank-Wolfe finds a sparse convex
combination of data log-likelihoods that fits the full sum:

  log p_full(θ) = Σ_n ℓ_n(θ)
  target: find weights w ≥ 0 with |supp(w)| ≤ M minimising
          ‖Σ_n w_n ℓ_n − log p_full‖

where norm is taken in the Hilbert space induced by an inner product on
function space (concrete: RKHS with finite kernel).

We implement the "vectors in R^d" surrogate version: each datum is a vector
(e.g., its score vector at some test θ_t), and the coreset is built in the
Euclidean inner product. This is sufficient for the workspace-admission use
in CASR: represent each agent by a score-like vector, build a Frank-Wolfe
coreset of size M ≤ log N, and admit only the coreset agents to the workspace.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CoresetResult:
    weights: np.ndarray       # (N,) nonneg, sparse
    support: list[int]        # nonzero indices
    error: float              # residual norm

    def summary(self) -> str:
        return (
            f"coreset of size {len(self.support)}, "
            f"residual {self.error:.4f}"
        )


def frank_wolfe_coreset(
    vectors: np.ndarray,
    max_size: int = 10,
) -> CoresetResult:
    """Build a sparse nonnegative-weight coreset via Frank-Wolfe on vectors.

    Target = sum of all vectors. Each step: pick the vector most correlated
    with the residual, line-search on a scalar step, add to weights.
    """
    V = np.asarray(vectors, dtype=float)
    if V.ndim != 2:
        raise ValueError("vectors must be (N, d)")
    N, _ = V.shape
    target = V.sum(axis=0)
    w = np.zeros(N)
    current = np.zeros_like(target)

    for _ in range(max_size):
        residual = target - current
        # pick atom with largest dot-product with residual
        scores = V @ residual
        i = int(np.argmax(scores))
        # Line-search on scalar step γ: choose γ to minimize ‖current + γ v_i − target‖²
        v = V[i]
        denom = float(v @ v)
        if denom < 1e-12:
            break
        gamma = max(0.0, float((target - current) @ v) / denom)
        if gamma <= 0:
            break
        w[i] += gamma
        current = current + gamma * v
        if np.linalg.norm(residual) < 1e-8:
            break

    return CoresetResult(
        weights=w,
        support=[int(i) for i in np.where(w > 0)[0]],
        error=float(np.linalg.norm(target - current)),
    )
