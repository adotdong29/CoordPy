"""Expected Information Gain — Bayesian optimal experimental design.

For a prior p(θ), a forward model p(y | θ, d) parameterized by design d, and
an observation y, the expected information gain is

    EIG(d) = E_{p(y | d)} [ KL( p(θ | y, d) ‖ p(θ) ) ]
           = E_{p(y, θ | d)} [ log p(y | θ, d) − log p(y | d) ]

For a linear Gaussian model y = Aθ + ε, ε ~ N(0, Σ_ε), prior θ ~ N(0, Σ_θ):

    EIG(A) = ½ log det( I + Σ_ε^{-1} A Σ_θ Aᵀ )

This is the optimal design criterion for Bayesian experiments (Lindley
1956; Chaloner & Verdinelli 1995). In CASR, the design d is "which
event/agent to pull into the workspace next"; ranking by EIG is the
information-theoretic alternative to surprise-only ranking.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EIGReport:
    eig: float
    posterior_cov: np.ndarray

    def summary(self) -> str:
        return f"EIG = {self.eig:.4f} nats"


def gaussian_linear_eig(
    A: np.ndarray,
    Sigma_theta: np.ndarray,
    Sigma_eps: np.ndarray,
) -> EIGReport:
    """Closed-form EIG for y = Aθ + ε with Gaussian θ and ε.

    A:          (m, d) design
    Sigma_theta: (d, d) prior covariance
    Sigma_eps:   (m, m) noise covariance
    """
    A = np.asarray(A, dtype=float)
    Sp = np.asarray(Sigma_theta, dtype=float)
    Se = np.asarray(Sigma_eps, dtype=float)

    m, d = A.shape
    if Sp.shape != (d, d):
        raise ValueError(f"Sigma_theta must be ({d},{d})")
    if Se.shape != (m, m):
        raise ValueError(f"Sigma_eps must be ({m},{m})")

    # EIG = ½ log det( I + Σ_ε⁻¹ A Σ_θ Aᵀ )
    Se_inv = np.linalg.inv(Se)
    middle = Se_inv @ (A @ Sp @ A.T)
    eig = 0.5 * float(np.linalg.slogdet(np.eye(m) + middle)[1])

    # Posterior covariance: (Σ_θ⁻¹ + Aᵀ Σ_ε⁻¹ A)⁻¹
    Sp_inv = np.linalg.inv(Sp)
    post_cov = np.linalg.inv(Sp_inv + A.T @ Se_inv @ A)

    return EIGReport(eig=eig, posterior_cov=post_cov)


def rank_by_eig(
    designs: list[np.ndarray],
    Sigma_theta: np.ndarray,
    Sigma_eps: np.ndarray,
) -> list[tuple[int, float]]:
    """Rank candidate designs by EIG descending. Returns [(index, eig), ...]."""
    scored = []
    for i, A in enumerate(designs):
        r = gaussian_linear_eig(A, Sigma_theta, Sigma_eps)
        scored.append((i, r.eig))
    scored.sort(key=lambda t: -t[1])
    return scored


def nested_mc_eig(
    log_likelihood,
    prior_sampler,
    n_outer: int = 100,
    n_inner: int = 50,
    rng_seed: int = 0,
) -> float:
    """Nested Monte Carlo EIG for non-Gaussian models.

    log_likelihood(theta, y) -> scalar log p(y|θ).
    prior_sampler(rng) -> draws one θ.
    """
    rng = np.random.default_rng(rng_seed)
    total = 0.0
    for _ in range(n_outer):
        theta = prior_sampler(rng)
        y = theta + rng.standard_normal(*np.atleast_1d(theta).shape)
        # inner: log p(y) ≈ logsumexp over prior samples
        samples = np.array([log_likelihood(prior_sampler(rng), y) for _ in range(n_inner)])
        m = samples.max()
        log_p_y = m + np.log(np.exp(samples - m).mean())
        total += log_likelihood(theta, y) - log_p_y
    return total / n_outer
