"""Kuramoto synchronization diagnostic on agent graphs.

The Kuramoto model (1975) is the canonical toy for phase-locking in coupled
oscillator systems. For N agents with intrinsic frequencies ω_i and
adjacency A:

    dθ_i/dt = ω_i + (K/N) Σ_j A_ij sin(θ_j − θ_i)

Order parameter r(t) = |mean(exp(i θ))| ∈ [0, 1]. r ≈ 0 → incoherent;
r ≈ 1 → fully synchronized. The critical coupling K_c in mean field for a
Lorentzian frequency distribution is K_c = 2/(π g(0)) where g is the density.

Used here as a dynamical-order diagnostic: project agent beliefs to a scalar
phase, integrate Kuramoto on the causal-footprint graph, and read r(t).
A team whose r climbs to ≈1 is well-coupled; a team with persistent r≈0 has
information silos.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class KuramotoReport:
    times: np.ndarray         # (T,) time points
    order_parameter: np.ndarray   # (T,) r(t)
    final_r: float
    phases: np.ndarray        # (T, N) trajectories

    def summary(self) -> str:
        return (
            f"r({self.times[-1]:.1f}) = {self.final_r:.3f}  "
            f"(min={self.order_parameter.min():.3f}, "
            f"max={self.order_parameter.max():.3f})"
        )


def order_parameter(theta: np.ndarray) -> float:
    """Kuramoto order parameter r = |mean(e^{iθ})|."""
    theta = np.asarray(theta, dtype=float).ravel()
    return float(np.abs(np.exp(1j * theta).mean()))


def simulate(
    adjacency: np.ndarray,
    omega: np.ndarray,
    coupling: float,
    theta0: np.ndarray | None = None,
    dt: float = 0.05,
    t_end: float = 20.0,
    seed: int = 0,
) -> KuramotoReport:
    """Explicit-Euler Kuramoto integrator on the given adjacency.

    `adjacency` is (N, N) nonnegative. `omega` is (N,) intrinsic frequencies.
    `coupling` is K in the Kuramoto equation.
    """
    A = np.asarray(adjacency, dtype=float)
    w = np.asarray(omega, dtype=float).ravel()
    if A.shape[0] != A.shape[1] or A.shape[0] != w.size:
        raise ValueError("adjacency and omega must share dimension N")
    N = w.size
    rng = np.random.default_rng(seed)
    theta = (
        np.asarray(theta0, dtype=float).ravel().copy()
        if theta0 is not None
        else rng.uniform(-np.pi, np.pi, size=N)
    )

    steps = int(t_end / dt)
    times = np.linspace(0.0, t_end, steps + 1)
    phases = np.zeros((steps + 1, N))
    r_hist = np.zeros(steps + 1)
    phases[0] = theta
    r_hist[0] = order_parameter(theta)

    for k in range(steps):
        diff = np.sin(theta[None, :] - theta[:, None])  # θ_j − θ_i
        # Σ_j A_ij sin(θ_j − θ_i) — note A_ij acts on diff_ij
        coupling_term = (A * diff).sum(axis=1) * (coupling / N)
        theta = theta + dt * (w + coupling_term)
        theta = (theta + np.pi) % (2 * np.pi) - np.pi
        phases[k + 1] = theta
        r_hist[k + 1] = order_parameter(theta)

    return KuramotoReport(
        times=times,
        order_parameter=r_hist,
        final_r=float(r_hist[-1]),
        phases=phases,
    )


def critical_coupling_meanfield(omega: np.ndarray, kde_bandwidth: float = 0.1) -> float:
    """Mean-field K_c = 2 / (π g(0)), where g is a Gaussian-KDE estimate of
    the frequency density at 0 (after mean-centering).
    """
    w = np.asarray(omega, dtype=float).ravel()
    w = w - w.mean()
    # Silverman-style default bandwidth if not overridden
    n = w.size
    h = kde_bandwidth if kde_bandwidth > 0 else 1.06 * w.std(ddof=1) * n ** (-1 / 5)
    # g(0) = (1 / (n h √2π)) Σ exp(−w_i² / (2 h²))
    g0 = float(
        (1 / (n * h * np.sqrt(2 * np.pi)))
        * np.exp(-(w ** 2) / (2 * h * h)).sum()
    )
    return 2.0 / (np.pi * max(g0, 1e-12))
