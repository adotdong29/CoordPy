"""Thermodynamic length — minimum dissipation bound for belief trajectories.

For a path θ(t) in a statistical manifold with Fisher information metric F(θ),
the thermodynamic length is

    ℓ = ∫₀^T √( θ̇ᵀ F(θ) θ̇ ) dt

and the Crooks–Sivak inequality bounds finite-time dissipation

    ⟨ΔS_diss⟩ ≥ ℓ² / τ

where τ is the total time. For CASR updates: if we view belief parameters θ
as the "thermodynamic" variables, the length ℓ tells us a lower bound on
information we're forced to pay for any given trajectory through belief space.
Short paths (high Fisher-weighted velocity) cost more; long careful paths cost
less.

This module computes ℓ given a discrete-time trajectory and an empirical
Fisher information (outer product of score vectors). It's a diagnostic tool
used to rank protocols by their energy-efficiency in information terms.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ThermoLengthReport:
    length: float
    dissipation_lower_bound: float    # ℓ²/τ
    tau: float
    n_steps: int

    def summary(self) -> str:
        return (
            f"ℓ={self.length:.4f}  "
            f"τ={self.tau:.2f}  "
            f"ΔS_diss ≥ {self.dissipation_lower_bound:.4f}"
        )


def empirical_fisher(scores: np.ndarray) -> np.ndarray:
    """Outer-product Fisher estimate: F ≈ (1/N) Σ s_i s_iᵀ.

    `scores` is (N, d). Returns (d, d) positive semi-definite matrix.
    """
    s = np.asarray(scores, dtype=float)
    n = s.shape[0]
    return (s.T @ s) / max(n, 1)


def thermo_length(
    trajectory: np.ndarray,
    fisher_fn,
    dt: float = 1.0,
) -> ThermoLengthReport:
    """Compute Σ √(Δθᵀ F(θ) Δθ) along a (T, d) trajectory.

    `fisher_fn` is a callable θ ↦ F(θ) returning a (d, d) PSD matrix. For a
    stationary Fisher, pass `lambda _: F_fixed`.
    """
    traj = np.asarray(trajectory, dtype=float)
    if traj.ndim != 2:
        raise ValueError("trajectory must be (T, d)")
    T, d = traj.shape
    if T < 2:
        return ThermoLengthReport(0.0, 0.0, tau=0.0, n_steps=0)

    dtheta = np.diff(traj, axis=0)
    total = 0.0
    for k in range(T - 1):
        F = np.asarray(fisher_fn(traj[k]), dtype=float)
        q = float(dtheta[k] @ F @ dtheta[k])
        total += np.sqrt(max(q, 0.0))
    tau = (T - 1) * dt
    return ThermoLengthReport(
        length=total,
        dissipation_lower_bound=total * total / max(tau, 1e-12),
        tau=tau,
        n_steps=T - 1,
    )
