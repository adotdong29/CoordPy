"""Deep Equilibrium — Anderson-accelerated fixed-point solver, numpy-only.

Bai, Kolter, Koltun (NeurIPS 2019). A DEQ layer computes a fixed point z* of
a parameterised map f_θ: z* = f_θ(z*, x). Anderson acceleration (Anderson
1965) converges faster than naive Picard iteration, especially near the fixed
point: the next iterate is an affine combination of the last m iterates that
minimises the residual.

This is the *inference-time* half — training via implicit differentiation
requires autograd (tier 3 with torch). The inference-time fixed-point
solver is pure NumPy and serves as a drop-in replacement for any iterated
Banach-contraction map in the CASR stack, giving faster convergence with
contraction-rate robustness bounds.

For OQ1: if f is a contraction, Anderson preserves convergence with an
asymptotic rate improvement. If f is merely quasi-contracting, Anderson often
still converges where Picard diverges — a useful safety net.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class DEQSolveReport:
    fixed_point: np.ndarray
    residual: float
    iterations: int
    converged: bool

    def summary(self) -> str:
        return (
            f"{'✓' if self.converged else '✗'} "
            f"{self.iterations} iters, residual {self.residual:.2e}"
        )


def picard_iterate(
    f: Callable[[np.ndarray], np.ndarray],
    z0: np.ndarray,
    tol: float = 1e-6,
    max_iter: int = 200,
) -> DEQSolveReport:
    z = z0.copy()
    for i in range(max_iter):
        z_new = np.asarray(f(z), dtype=float)
        res = float(np.linalg.norm(z_new - z))
        z = z_new
        if res < tol:
            return DEQSolveReport(
                fixed_point=z, residual=res, iterations=i + 1, converged=True,
            )
    return DEQSolveReport(
        fixed_point=z, residual=res, iterations=max_iter, converged=False,
    )


def anderson_iterate(
    f: Callable[[np.ndarray], np.ndarray],
    z0: np.ndarray,
    m: int = 5,
    tol: float = 1e-6,
    max_iter: int = 200,
    lam: float = 1e-4,
) -> DEQSolveReport:
    """Anderson Acceleration with history length `m` and regularisation `lam`.

    At step k, let F_i = f(z_i) − z_i for i in the most recent m iterates.
    Solve for coefficients α minimising ‖Σ α_i F_i‖² with Σ α_i = 1, then
    set z_{k+1} = Σ α_i f(z_i).
    """
    z = z0.copy().astype(float)
    z_hist = [z]
    g_hist = [np.asarray(f(z), dtype=float)]
    for k in range(max_iter):
        # Current residual
        res = float(np.linalg.norm(g_hist[-1] - z_hist[-1]))
        if res < tol:
            return DEQSolveReport(
                fixed_point=z_hist[-1], residual=res,
                iterations=k, converged=True,
            )
        # Build residual matrix F
        F = np.stack([g - z for g, z in zip(g_hist, z_hist)], axis=0)
        nh = F.shape[0]
        if nh == 1:
            # Just fall back to Picard step
            z_next = g_hist[-1]
        else:
            # Solve min ‖Σ α_i F_i‖² s.t. Σ α_i = 1
            G = F @ F.T + lam * np.eye(nh)
            ones = np.ones(nh)
            alpha = np.linalg.solve(G, ones)
            alpha = alpha / alpha.sum()
            z_next = sum(a * g for a, g in zip(alpha, g_hist))
        g_next = np.asarray(f(z_next), dtype=float)
        z_hist.append(z_next)
        g_hist.append(g_next)
        if len(z_hist) > m:
            z_hist.pop(0)
            g_hist.pop(0)

    return DEQSolveReport(
        fixed_point=z_hist[-1],
        residual=float(np.linalg.norm(g_hist[-1] - z_hist[-1])),
        iterations=max_iter,
        converged=False,
    )
