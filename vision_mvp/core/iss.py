"""Input-to-State Stability (ISS) diagnostics and the small-gain theorem.

For a dynamical system x_{k+1} = F(x_k, u_k), ISS means that bounded input u
implies a bounded state x, with a contribution that decays as input decays:

    ‖x_k‖ ≤ β(‖x_0‖, k) + γ(sup_s ‖u_s‖)

Empirically we estimate the ISS gain γ as

    γ̂ = sup_{u ≠ 0} ‖x‖_steady / ‖u‖

For a feedback interconnection of two ISS systems with gains γ_1 and γ_2, the
closed loop is stable iff γ_1 · γ_2 < 1 (small-gain theorem, Zames 1966).
Composed across layers, the CASR stack is stable iff ∏ γ_i < 1.

This module gives:
  - `estimate_iss_gain(system, input_range)` — empirical γ̂.
  - `small_gain(gains)` — True iff ∏ < 1.
  - `stability_margin(gains)` — headroom below the 1 threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class ISSReport:
    gain: float
    n_probes: int
    converged: bool
    input_range: float

    def summary(self) -> str:
        return (
            f"γ={self.gain:.4f}  "
            f"({'converged' if self.converged else 'did NOT converge'}) "
            f"over {self.n_probes} probes of max input {self.input_range:.3f}"
        )


def estimate_iss_gain(
    system: Callable[[np.ndarray, np.ndarray], np.ndarray],
    state_dim: int,
    input_dim: int,
    input_range: float = 1.0,
    n_probes: int = 20,
    steps: int = 200,
    tol: float = 1e-6,
    seed: int = 0,
) -> ISSReport:
    """Empirical ISS gain — ratio of steady-state-norm to input-norm.

    `system(x, u)` returns x_{k+1}. We drive with constant u, run forward until
    convergence, and measure ‖x_∞‖ / ‖u‖. Max over several random u gives γ̂.
    """
    rng = np.random.default_rng(seed)
    worst_ratio = 0.0
    all_converged = True
    for _ in range(n_probes):
        u = rng.standard_normal(input_dim)
        u = u / max(np.linalg.norm(u), 1e-12) * input_range
        x = np.zeros(state_dim)
        converged = False
        last_norm = 0.0
        for _ in range(steps):
            x_new = np.asarray(system(x, u), dtype=float).ravel()
            if np.linalg.norm(x_new - x) < tol:
                converged = True
                x = x_new
                break
            x = x_new
            last_norm = float(np.linalg.norm(x))
        if not converged:
            all_converged = False
        u_norm = max(float(np.linalg.norm(u)), 1e-12)
        ratio = float(np.linalg.norm(x)) / u_norm
        worst_ratio = max(worst_ratio, ratio)
    return ISSReport(
        gain=worst_ratio,
        n_probes=n_probes,
        converged=all_converged,
        input_range=input_range,
    )


def small_gain(gains: list[float]) -> bool:
    """∏ γ_i < 1 ⇔ composed feedback loop is stable."""
    if not gains:
        return True
    prod = 1.0
    for g in gains:
        if g < 0:
            raise ValueError("gains must be nonnegative")
        prod *= g
    return prod < 1.0


def stability_margin(gains: list[float]) -> float:
    """1 − ∏ γ_i. Positive = stable margin; negative = potentially unstable."""
    prod = 1.0
    for g in gains:
        prod *= g
    return 1.0 - prod
