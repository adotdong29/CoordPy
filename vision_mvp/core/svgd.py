"""Stein Variational Gradient Descent for distributed belief transport.

SVGD (Liu & Wang, NeurIPS 2016) is a deterministic particle update that
follows the KL-divergence gradient flow in the RKHS of an RBF kernel:

    x_i ← x_i + ε · φ(x_i)
    φ(x_i) = (1/N) Σ_j [ k(x_j, x_i) ∇_{x_j} log p(x_j)
                         + ∇_{x_j} k(x_j, x_i) ]

For a Gaussian RBF kernel k(x, y) = exp(−‖x − y‖² / (2 h²)):
    ∇_{x_j} k(x_j, x_i) = −(x_j − x_i) / h² · k(x_j, x_i)

Used here as the *implementable* OT layer in Idea 3 of `VISION_MILLIONS.md`.
Agents' belief particles get pushed along the Stein direction to match a
target posterior without needing a differentiable parameterization of the
pushforward (unlike normalizing flows).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


def rbf_kernel(X: np.ndarray, h: float) -> tuple[np.ndarray, np.ndarray]:
    """Gram matrix K and pairwise difference ∇_{x_j} k(x_j, x_i) summed over j.

    Returns (K, grad_K_sum) where
        K[i, j] = exp(−‖x_i − x_j‖² / (2 h²))
        grad_K_sum[i] = Σ_j ∇_{x_j} k(x_j, x_i)
                      = Σ_j (x_i − x_j) / h² · K[i, j]
    """
    X = np.asarray(X, dtype=float)
    # pairwise squared distances
    sq = ((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=-1)
    K = np.exp(-sq / (2 * h * h))
    # ∇_{x_j} k(x_j, x_i) = (x_i − x_j) / h² · K[i,j]
    # grad_K_sum[i] = Σ_j (x_i − x_j) K[i,j] / h²
    diff = X[:, None, :] - X[None, :, :]      # (N, N, d)
    grad = (diff * K[:, :, None]).sum(axis=1) / (h * h)
    return K, grad


def median_bandwidth(X: np.ndarray) -> float:
    """Heuristic RBF bandwidth h² = med²/log N (Liu & Wang 2016)."""
    X = np.asarray(X, dtype=float)
    N = X.shape[0]
    if N < 2:
        return 1.0
    sq = ((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=-1)
    med = float(np.sqrt(np.median(sq[np.triu_indices(N, k=1)])))
    return med / max(np.sqrt(np.log(N)), 1.0)


@dataclass
class SVGDReport:
    steps: int
    final_particles: np.ndarray
    step_norms: np.ndarray

    def summary(self) -> str:
        return (
            f"{self.steps} steps, "
            f"final ‖φ‖ = {self.step_norms[-1]:.4f}"
        )


def svgd(
    particles: np.ndarray,
    grad_log_p: Callable[[np.ndarray], np.ndarray],
    steps: int = 200,
    lr: float = 0.05,
    h: float | None = None,
) -> SVGDReport:
    """Iterate SVGD for `steps`. `grad_log_p` is vectorized:
    given (N, d) input, return (N, d) gradient of log p at each row.
    """
    X = np.asarray(particles, dtype=float).copy()
    if X.ndim != 2:
        raise ValueError("particles must be (N, d)")
    N, d = X.shape
    step_norms = np.zeros(steps)
    for t in range(steps):
        bw = h if h is not None else median_bandwidth(X)
        K, grad_K = rbf_kernel(X, bw)
        score = grad_log_p(X)       # (N, d)
        phi = (K @ score + grad_K) / N
        X = X + lr * phi
        step_norms[t] = float(np.linalg.norm(phi))
    return SVGDReport(steps=steps, final_particles=X, step_norms=step_norms)
