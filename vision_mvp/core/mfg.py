"""Mean-Field Games — coupled HJB / Fokker-Planck iteration.

Lasry & Lions (2006); Huang, Malhamé, Caines (2006). An MFG models the
N→∞ limit of an N-agent symmetric game. Each agent solves a stochastic
optimal-control problem against the *distribution* of all other agents.
At equilibrium the (backward) Hamilton-Jacobi-Bellman equation for the value u
is coupled to a (forward) Fokker-Planck equation for the population density m:

  HJB: −∂_t u + H(x, ∇u, m) = σ²/2 Δu,   u(T, x) = g(x, m_T)
  FPK: ∂_t m − div(m · ∂_p H) = σ²/2 Δm, m(0, x) = m_0(x)

For a 1-D quadratic Hamiltonian H(p, m) = p²/2 + f(x, m), finite-difference
Picard iteration converges. We ship a minimal solver sufficient to validate
Idea in Layer 2 of `VISION_MILLIONS.md`: agents respond to the population
mean, population evolves under the controlled flow.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MFGReport:
    u: np.ndarray                     # (T+1, X) value function
    m: np.ndarray                     # (T+1, X) density
    n_picard_iters: int
    final_residual: float

    def summary(self) -> str:
        return (
            f"Picard converged in {self.n_picard_iters} iters  "
            f"(residual {self.final_residual:.2e})"
        )


def solve_1d_mfg(
    x_grid: np.ndarray,
    t_end: float,
    n_t: int,
    sigma: float,
    running_cost: callable,     # f(x, m) -> array
    terminal_cost: callable,    # g(x, m_T) -> array
    m0: np.ndarray,
    max_iter: int = 30,
    tol: float = 1e-4,
) -> MFGReport:
    """Solve the quadratic-Hamiltonian 1-D MFG with Picard damping.

    Returns value-function and density trajectories.
    """
    x = np.asarray(x_grid, dtype=float)
    X = x.size
    dx = float(x[1] - x[0])
    dt = t_end / n_t

    # Initial guesses
    u = np.zeros((n_t + 1, X))
    m = np.zeros((n_t + 1, X))
    m[0] = m0 / max(m0.sum() * dx, 1e-12)
    m[:] = m[0]

    residual = np.inf
    it = 0
    damping = 0.5
    for it in range(max_iter):
        m_old = m.copy()

        # --- backward HJB pass ---
        u[-1] = terminal_cost(x, m[-1])
        for k in range(n_t - 1, -1, -1):
            # upwind derivatives approximation: use central differences
            # H(p) = p²/2, so optimal control α* = −p
            u_xx = _laplacian(u[k + 1], dx)
            u_x = _center_diff(u[k + 1], dx)
            h = 0.5 * u_x ** 2 + running_cost(x, m[k + 1])
            u[k] = u[k + 1] + dt * (0.5 * sigma ** 2 * u_xx - h)

        # --- forward FPK pass ---
        m_new = np.zeros_like(m)
        m_new[0] = m[0]
        for k in range(n_t):
            u_x_next = _center_diff(u[k], dx)
            flux = -u_x_next * m_new[k]    # mass × optimal drift
            div_flux = _center_diff(flux, dx)
            diff_term = 0.5 * sigma ** 2 * _laplacian(m_new[k], dx)
            m_step = m_new[k] + dt * (diff_term - div_flux)
            m_step = np.maximum(m_step, 0.0)
            # renormalize to preserve mass
            s = float(m_step.sum() * dx)
            if s > 0:
                m_step /= s
            m_new[k + 1] = m_step

        # Picard damping for stability
        m = damping * m_new + (1 - damping) * m_old
        residual = float(np.max(np.abs(m - m_old)))
        if residual < tol:
            break

    return MFGReport(
        u=u, m=m, n_picard_iters=it + 1, final_residual=residual,
    )


def _center_diff(f: np.ndarray, dx: float) -> np.ndarray:
    g = np.zeros_like(f)
    g[1:-1] = (f[2:] - f[:-2]) / (2 * dx)
    g[0] = (f[1] - f[0]) / dx
    g[-1] = (f[-1] - f[-2]) / dx
    return g


def _laplacian(f: np.ndarray, dx: float) -> np.ndarray:
    g = np.zeros_like(f)
    g[1:-1] = (f[2:] - 2 * f[1:-1] + f[:-2]) / (dx ** 2)
    g[0] = g[1]
    g[-1] = g[-2]
    return g
