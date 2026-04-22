"""Bayesian nonparametrics — Dirichlet-process stick-breaking + truncated VI.

A Dirichlet process DP(α, G_0) lets model capacity grow with data. The stick-
breaking construction (Sethuraman 1994) represents its samples as

    π_k = V_k · ∏_{j<k} (1 − V_j),   V_k ~ Beta(1, α),   θ_k ~ G_0

With a truncation K, we approximate the infinite sum by the first K sticks
(Blei & Jordan 2006). Mean-field variational inference gives closed-form
updates:

    q(V_k) = Beta(γ_{k,1}, γ_{k,2})
    q(θ_k) = G_0-conjugate
    q(z_n = k) = softmax( E_q[log V_k] + Σ_{j<k} E_q[log(1−V_j)] + E_q[log p(x_n|θ_k)] )

This module implements a truncated-stick-breaking Gaussian mixture with a
fixed covariance and a Gaussian prior on component means. Good enough to
grow the number of effective components with N for dynamic-membership
workspaces. Uses an in-house digamma approximation so scipy is not needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _digamma(x: float | np.ndarray) -> np.ndarray:
    """Asymptotic series for ψ(x); accurate for x > 4."""
    x = np.asarray(x, dtype=float)
    result = np.zeros_like(x)
    # Shift small x to large by recurrence ψ(x) = ψ(x+1) - 1/x
    small = x < 6
    shift = np.where(small, 6.0 - x, 0.0)
    z = x + shift
    # Asymptotic series
    y = np.log(z) - 1 / (2 * z)
    z2 = z ** -2
    y -= z2 * (1 / 12 - z2 * (1 / 120 - z2 * 1 / 252))
    # Undo the shift
    for k in range(6):
        correction = np.where((x < 6) & (x + k < 6), -1 / (x + k), 0.0)
        y = y + correction
    return y


@dataclass
class BNPReport:
    weights: np.ndarray          # (K,) posterior mean of π
    means: np.ndarray            # (K, d) component means
    assignment_probs: np.ndarray   # (N, K) responsibilities
    active_components: int       # # of components with nonnegligible weight

    def summary(self) -> str:
        return (
            f"{self.active_components} active components "
            f"(from {self.weights.size} truncation)"
        )


def truncated_dp_mixture_vi(
    X: np.ndarray,
    K: int = 10,
    alpha: float = 1.0,
    sigma: float = 1.0,
    prior_var: float = 10.0,
    n_iter: int = 50,
    seed: int = 0,
) -> BNPReport:
    """Truncated DP-Gaussian-mixture via mean-field VI.

    Fixed observation noise σ²; each component mean has prior N(0, prior_var).
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be (N, d)")
    N, d = X.shape
    rng = np.random.default_rng(seed)

    # Init
    mu = rng.standard_normal((K, d)) * np.sqrt(prior_var)
    tau = np.ones(K) / prior_var        # inverse variance of q(μ_k)
    gamma1 = np.ones(K)
    gamma2 = np.full(K, alpha)
    obs_prec = 1.0 / (sigma * sigma)

    for _ in range(n_iter):
        # E[log V_k], E[log(1-V_k)]
        e_log_v = _digamma(gamma1) - _digamma(gamma1 + gamma2)
        e_log_1v = _digamma(gamma2) - _digamma(gamma1 + gamma2)
        # E[log π_k] = E[log V_k] + Σ_{j<k} E[log(1-V_j)]
        cum_1v = np.concatenate([[0.0], np.cumsum(e_log_1v[:-1])])
        e_log_pi = e_log_v + cum_1v

        # E[log p(x_n | μ_k)] under q — Gaussian log-likelihood with μ_k expectation + variance correction
        diff = X[:, None, :] - mu[None, :, :]
        log_lik = -0.5 * obs_prec * (diff ** 2).sum(axis=-1) - 0.5 * d * np.log(2 * np.pi / obs_prec)
        # Variance correction due to uncertainty in μ: subtract 0.5 * obs_prec * d / tau_k
        log_lik = log_lik - 0.5 * obs_prec * d / tau[None, :]

        log_resp = log_lik + e_log_pi[None, :]
        # softmax row-wise
        log_resp -= log_resp.max(axis=1, keepdims=True)
        resp = np.exp(log_resp)
        resp /= resp.sum(axis=1, keepdims=True)

        # M-like step for q(μ_k), q(V_k)
        Nk = resp.sum(axis=0)
        tau_new = (1.0 / prior_var) + obs_prec * Nk
        mu_new = (obs_prec * (resp[..., None] * X[:, None, :]).sum(axis=0)) / tau_new[:, None]
        # Beta for sticks:
        gamma1 = 1.0 + Nk
        gamma2 = alpha + np.concatenate([np.cumsum(Nk[::-1])[::-1][1:], [0.0]])
        mu = mu_new
        tau = tau_new

    # Posterior mean weights
    v_mean = gamma1 / (gamma1 + gamma2)
    weights = v_mean * np.concatenate([[1.0], np.cumprod(1 - v_mean[:-1])])

    return BNPReport(
        weights=weights,
        means=mu,
        assignment_probs=resp,
        active_components=int(np.sum(weights > 0.01)),
    )
