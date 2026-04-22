"""Wasserstein-2 distance for multi-agent opinion drift tracking.

In a multi-agent team, per-round opinions form an empirical distribution
in embedding space. The Wasserstein-2 distance between successive-round
distributions is a principled measure of "how much did the team's
opinion actually move" — sensitive to shape and spread, not just mean.

Why W2 beats centroid-drift for our diagnostic:
  - Detects POLARIZATION: if a team splits into two clusters with the
    same overall mean, centroid drift says ~0 but W2 is large.
  - Detects VARIANCE COLLAPSE: agents converging to the mean register as
    decreasing W2 even when the mean is stable.
  - For Gaussians W2 factors as Bures: ‖Δμ‖² + (σ_A − σ_B)² (1-D).

Two algorithms:
  - **Exact (Hungarian)**: for equal-mass distributions, optimal transport
    plan is a permutation. O(N³) via linear_sum_assignment.
  - **Sinkhorn (entropic regularization)**: O(N² · iters), scales to
    hundreds of agents. Used as fallback when N > 500.

Reference: Peyré–Cuturi, "Computational Optimal Transport" (2019).
"""

from __future__ import annotations
import numpy as np


def _cost_matrix(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Squared-Euclidean cost matrix C[i,j] = ‖x_i - y_j‖²."""
    # Efficient: ‖x-y‖² = ‖x‖² + ‖y‖² - 2 x·y
    xx = (X * X).sum(axis=1, keepdims=True)      # (N,1)
    yy = (Y * Y).sum(axis=1, keepdims=True)      # (M,1)
    return xx + yy.T - 2.0 * X @ Y.T             # (N,M)


def w2_exact(X: np.ndarray, Y: np.ndarray) -> float:
    """Exact Wasserstein-2 between equal-mass empirical distributions.

    Uses Hungarian assignment. O(N³).
    """
    if X.shape[0] != Y.shape[0]:
        raise ValueError("w2_exact requires equal-sized samples")
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        # Fallback: greedy assignment (not optimal but valid)
        return _greedy_w2(X, Y)
    C = _cost_matrix(X, Y)
    row_ind, col_ind = linear_sum_assignment(C)
    return float(np.sqrt(C[row_ind, col_ind].sum() / len(X)))


def _greedy_w2(X: np.ndarray, Y: np.ndarray) -> float:
    """Fallback: greedy assignment — not optimal but gives an UPPER bound on
    true W2 (valid for diagnostics, not for proofs)."""
    C = _cost_matrix(X, Y)
    N = len(X)
    used_cols = np.zeros(N, dtype=bool)
    total = 0.0
    for i in range(N):
        # Pick the smallest-cost unused column
        row = C[i].copy()
        row[used_cols] = np.inf
        j = int(np.argmin(row))
        total += row[j]
        used_cols[j] = True
    return float(np.sqrt(total / N))


def w2_sinkhorn(X: np.ndarray, Y: np.ndarray,
                epsilon: float = 0.05,
                max_iters: int = 200,
                tol: float = 1e-6) -> float:
    """Sinkhorn-regularized W2. O(N² · max_iters). Scales to larger N.

    `epsilon` controls the regularization: smaller → closer to exact W2 but
    numerically less stable. 0.05 × median(C) is a good default.
    """
    if X.shape[0] == 0 or Y.shape[0] == 0:
        return 0.0
    N, M = X.shape[0], Y.shape[0]
    C = _cost_matrix(X, Y)
    # Scale epsilon relative to cost magnitude
    if epsilon < 1.0 and np.median(C) > 0:
        epsilon_eff = epsilon * float(np.median(C))
    else:
        epsilon_eff = float(epsilon)
    K = np.exp(-C / max(epsilon_eff, 1e-8))
    a = np.ones(N) / N
    b = np.ones(M) / M
    u = np.ones(N)
    v = np.ones(M)
    for _ in range(max_iters):
        u_new = a / (K @ v + 1e-300)
        v_new = b / (K.T @ u_new + 1e-300)
        if np.abs(u_new - u).max() < tol and np.abs(v_new - v).max() < tol:
            u, v = u_new, v_new
            break
        u, v = u_new, v_new
    pi = u[:, None] * K * v[None, :]
    # Transport cost — clamp to ≥ 0 to avoid numerical negatives
    transport_cost = float((pi * C).sum())
    return float(np.sqrt(max(transport_cost, 0.0)))


def w2(X: np.ndarray, Y: np.ndarray, use_sinkhorn_above: int = 500) -> float:
    """Dispatcher: exact for small N, Sinkhorn for large N."""
    if X.shape[0] <= use_sinkhorn_above and X.shape[0] == Y.shape[0]:
        return w2_exact(X, Y)
    return w2_sinkhorn(X, Y)


def bures_decomposition(X: np.ndarray, Y: np.ndarray) -> dict:
    """Split W2² into mean-drift and spread components.

    For general distributions this is NOT exact, but for approximately
    Gaussian teams the Bures decomposition holds:
      W2² ≈ ‖μ_X - μ_Y‖² + (σ_X − σ_Y)²      (1-D)
      W2² ≈ ‖μ_X - μ_Y‖² + tr(Σ_X + Σ_Y − 2(Σ_Y^½ Σ_X Σ_Y^½)^½)  (d-D)

    We return the scalar-level mean-drift and total-spread components
    so the diagnostic can distinguish "mean moved" from "team polarized".
    """
    mu_x = X.mean(axis=0)
    mu_y = Y.mean(axis=0)
    mean_drift = float(np.linalg.norm(mu_x - mu_y))
    # Spread = sum of std deviations per dim
    sigma_x = X.std(axis=0)
    sigma_y = Y.std(axis=0)
    spread_shift = float(np.linalg.norm(sigma_x - sigma_y))
    return {
        "mean_drift": mean_drift,
        "spread_shift": spread_shift,
        "bures_lower_bound_sq": mean_drift ** 2 + spread_shift ** 2,
    }
