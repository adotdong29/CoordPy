"""Contraction analysis — a numerical certificate for OQ1.

The open question in `OPEN_QUESTIONS.md` asks whether the minimum-sufficient-
context iteration T* = f(T*) converges. Banach's theorem guarantees a unique
fixed point if f is a contraction; contraction analysis (Lohmiller & Slotine,
1998) gives a locally-verifiable sufficient condition for nonlinear maps.

For a discrete map x_{k+1} = F(x_k):
    ρ(x) := σ_max(∂F/∂x |_x)
F is a contraction on a region U iff sup_{x ∈ U} ρ(x) < 1. Trajectories from
any two initial conditions in U converge to the unique fixed point at rate ρ.

For a continuous flow ẋ = F(x, t):
    µ(x) := λ_max( ½(J + Jᵀ) )
F is contracting with rate λ iff µ(x) ≤ -λ < 0 uniformly on U.

Neither measure requires an analytical Jacobian — a central-difference
estimator suffices for diagnostic use at the scales this repo cares about
(state dim ≤ a few hundred).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


def central_difference_jacobian(
    F: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """Numerical Jacobian ∂F/∂x at x via central differences.

    Complexity is 2·dim(x) calls to F. Accurate to O(eps²) for smooth F.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    f0 = np.asarray(F(x)).ravel()
    m = f0.size
    J = np.empty((m, n), dtype=float)
    for i in range(n):
        e = np.zeros_like(x)
        e[i] = eps
        fp = np.asarray(F(x + e)).ravel()
        fm = np.asarray(F(x - e)).ravel()
        J[:, i] = (fp - fm) / (2.0 * eps)
    return J


@dataclass
class ContractionReport:
    rate: float               # σ_max(J) for discrete maps
    sym_eig_max: float        # λ_max(sym(J)) for continuous flows
    is_discrete_contracting: bool
    is_continuous_contracting: bool
    jacobian: np.ndarray

    def summary(self) -> str:
        return (
            f"ρ={self.rate:.4f} "
            f"(discrete{' ✓' if self.is_discrete_contracting else ' ✗'})  "
            f"µ={self.sym_eig_max:+.4f} "
            f"(cont.{' ✓' if self.is_continuous_contracting else ' ✗'})"
        )


def contraction_report(
    F: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    eps: float = 1e-5,
) -> ContractionReport:
    """Full contraction report at a single point x.

    `is_discrete_contracting` is True iff the top singular value of J is < 1.
    `is_continuous_contracting` is True iff µ(x) = λ_max(½(J+Jᵀ)) < 0.
    """
    J = central_difference_jacobian(F, x, eps=eps)
    # Discrete: spectral norm (largest singular value)
    rate = float(np.linalg.svd(J, compute_uv=False).max())
    # Continuous: largest eigenvalue of the symmetric part
    J_sym = 0.5 * (J + J.T) if J.shape[0] == J.shape[1] else None
    sym_max = float(np.linalg.eigvalsh(J_sym).max()) if J_sym is not None else float("nan")
    return ContractionReport(
        rate=rate,
        sym_eig_max=sym_max,
        is_discrete_contracting=rate < 1.0,
        is_continuous_contracting=(J_sym is not None and sym_max < 0.0),
        jacobian=J,
    )


def is_contracting_region(
    F: Callable[[np.ndarray], np.ndarray],
    samples: np.ndarray,
    eps: float = 1e-5,
) -> tuple[bool, float]:
    """Test contraction on a batch of sample points.

    Returns (all_contracting, worst_rate). `samples` has shape (k, d); the test
    is sufficient-at-samples, not a global proof — extend the guarantee to the
    surrounding region via continuity arguments (see Lohmiller-Slotine §3).
    """
    samples = np.asarray(samples, dtype=float)
    if samples.ndim == 1:
        samples = samples[None, :]
    worst = 0.0
    ok = True
    for x in samples:
        r = contraction_report(F, x, eps=eps)
        worst = max(worst, r.rate)
        if not r.is_discrete_contracting:
            ok = False
    return ok, worst


def banach_convergence_estimate(rate: float, tol: float) -> int:
    """Rounds needed to reach `tol` from a unit-radius initial error under
    Banach iteration with contraction rate `rate`.

        ‖x_k - x*‖ ≤ rate^k · ‖x_0 - x*‖
    so k ≥ log(tol) / log(rate).
    """
    if not 0 < rate < 1:
        raise ValueError(f"rate must be in (0, 1), got {rate}")
    if tol <= 0:
        raise ValueError(f"tol must be > 0, got {tol}")
    import math
    return int(math.ceil(math.log(tol) / math.log(rate)))
