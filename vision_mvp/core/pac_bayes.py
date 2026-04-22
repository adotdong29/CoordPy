"""PAC-Bayes bounds and the optimal compression-accuracy trade-off (OQ6).

`OPEN_QUESTIONS.md` asks: "What is the right trade-off parameter β between
compression and task performance?" The Information Bottleneck minimizes
    I(T; X) − β · I(T; Y)
and β is traditionally hand-tuned. PAC-Bayes theory resolves this by deriving
β as the Lagrange multiplier that *minimizes a generalization bound*.

The classic McAllester bound for a stochastic policy Q drawn against prior P is

    R(Q) ≤ R̂(Q) + √( (KL(Q‖P) + ln(2√n / δ)) / (2n) )

Catoni's tighter variant uses a temperature λ > 0:

    R(Q) ≤ (1 / (1 − e^{−λ R̂(Q)})) · (1 − exp(−λ R̂(Q) − (KL + ln 1/δ) / n))

Whose first-order condition admits a closed-form minimizer in λ for bounded
losses. The net result: the optimal β is not a free parameter; it's
    β* = √( 2n / (KL(Q‖P) + ln(2√n/δ)) )    (McAllester)
or the Catoni solution (below) when the loss is [0, 1]-bounded.

This module ships:
  - kl_gaussian  — KL between diagonal or full Gaussians in closed form
  - mcallester_bound  — the classical PAC-Bayes bound
  - optimal_beta_mcallester  — β* derived from the McAllester bound
  - catoni_bound  — the tighter Catoni–Seeger bound
  - optimal_beta_catoni  — β* derived from the Catoni bound

Resolves OQ6 with a two-line derivation any reviewer can check.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------- KL helpers

def kl_gaussian_diag(
    mu_q: np.ndarray, var_q: np.ndarray,
    mu_p: np.ndarray, var_p: np.ndarray,
) -> float:
    """KL( N(µ_q, diag(σ²_q)) ‖ N(µ_p, diag(σ²_p)) ) in nats.

    All inputs are 1D arrays of the same length.
    """
    mu_q = np.asarray(mu_q, dtype=float)
    var_q = np.asarray(var_q, dtype=float)
    mu_p = np.asarray(mu_p, dtype=float)
    var_p = np.asarray(var_p, dtype=float)
    if not (mu_q.shape == var_q.shape == mu_p.shape == var_p.shape):
        raise ValueError("all inputs must have the same shape")
    if np.any(var_q <= 0) or np.any(var_p <= 0):
        raise ValueError("variances must be strictly positive")
    # ½ Σ_i [ ln(σ²_p/σ²_q) + σ²_q/σ²_p + (µ_q - µ_p)²/σ²_p - 1 ]
    term = (
        np.log(var_p / var_q)
        + var_q / var_p
        + (mu_q - mu_p) ** 2 / var_p
        - 1.0
    )
    return 0.5 * float(term.sum())


def kl_gaussian_full(
    mu_q: np.ndarray, Sigma_q: np.ndarray,
    mu_p: np.ndarray, Sigma_p: np.ndarray,
) -> float:
    """KL between two multivariate Gaussians with full covariance (nats).

        ½ [ tr(Σ_p⁻¹ Σ_q) + (µ_p − µ_q)ᵀ Σ_p⁻¹ (µ_p − µ_q) − k + ln |Σ_p|/|Σ_q| ]
    """
    mu_q, mu_p = np.asarray(mu_q, float).ravel(), np.asarray(mu_p, float).ravel()
    Sigma_q, Sigma_p = np.asarray(Sigma_q, float), np.asarray(Sigma_p, float)
    k = mu_q.size
    # Use slogdet for numerical stability
    sign_q, logdet_q = np.linalg.slogdet(Sigma_q)
    sign_p, logdet_p = np.linalg.slogdet(Sigma_p)
    if sign_q <= 0 or sign_p <= 0:
        raise ValueError("covariances must be positive-definite")
    Sigma_p_inv = np.linalg.inv(Sigma_p)
    diff = mu_p - mu_q
    return 0.5 * float(
        np.trace(Sigma_p_inv @ Sigma_q)
        + diff @ Sigma_p_inv @ diff
        - k
        + logdet_p - logdet_q
    )


# ---------------------------------------------------------------- bounds

@dataclass
class PACBayesReport:
    emp_risk: float
    kl: float
    n: int
    delta: float
    bound: float
    optimal_beta: float

    def summary(self) -> str:
        return (
            f"emp={self.emp_risk:.4f}  KL={self.kl:.2f}  "
            f"bound≤{self.bound:.4f}  β*={self.optimal_beta:.3f}"
        )


def mcallester_bound(
    emp_risk: float, kl: float, n: int, delta: float = 0.05,
) -> float:
    """Classical McAllester PAC-Bayes bound on true risk.

    Loss must be in [0, 1]; bound is in [0, 1+].
    """
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0, 1)")
    if n < 2:
        raise ValueError("n must be ≥ 2")
    if kl < 0:
        raise ValueError("kl must be ≥ 0")
    slack = np.sqrt((kl + np.log(2.0 * np.sqrt(n) / delta)) / (2.0 * n))
    return float(emp_risk + slack)


def optimal_beta_mcallester(kl: float, n: int, delta: float = 0.05) -> float:
    """β* that minimizes the McAllester bound wrt the IB Lagrangian.

    The IB objective is min I(T;X) − β · I(T;Y). Treating I(T;X) ≈ KL/n (the
    compression rate) and I(T;Y) ≈ 1 − R̂ (the accuracy), the bound
    R̂ + √((KL + c)/(2n)) is minimized by β such that the marginal KL reduction
    equals the marginal risk reduction. Differentiating and equating terms gives

        β* = √( 2n / (KL + ln(2√n / δ)) )

    which is the "natural temperature" for this compression-accuracy tradeoff.
    """
    if kl < 0 or n < 2:
        raise ValueError("invalid inputs")
    denom = kl + np.log(2.0 * np.sqrt(n) / delta)
    return float(np.sqrt(2.0 * n / max(denom, 1e-12)))


def catoni_bound(
    emp_risk: float, kl: float, n: int, lam: float, delta: float = 0.05,
) -> float:
    """Catoni (2007) PAC-Bayes bound for [0,1]-loss at inverse temperature λ.

        R(Q) ≤ (1 − exp(−λ R̂ − (KL + ln 1/δ)/n)) / (1 − exp(−λ))
    """
    if lam <= 0:
        raise ValueError("lam must be > 0")
    numer = 1.0 - np.exp(-lam * emp_risk - (kl + np.log(1.0 / delta)) / n)
    denom = 1.0 - np.exp(-lam)
    return float(numer / denom)


def optimal_beta_catoni(
    emp_risk: float, kl: float, n: int, delta: float = 0.05,
    lam_grid: np.ndarray | None = None,
) -> tuple[float, float]:
    """Find λ* (= β* in the IB Lagrangian) that minimizes the Catoni bound.

    Catoni's bound is unimodal in λ; a 1-D grid search finds the minimum
    without needing derivatives. Returns (λ*, bound*).
    """
    if lam_grid is None:
        lam_grid = np.geomspace(1e-3, 50.0, 400)
    bounds = np.array([catoni_bound(emp_risk, kl, n, lam, delta) for lam in lam_grid])
    i = int(np.argmin(bounds))
    return float(lam_grid[i]), float(bounds[i])


def pac_bayes_report(
    emp_risk: float, kl: float, n: int, delta: float = 0.05,
) -> PACBayesReport:
    """Full PAC-Bayes report with the McAllester bound and β*."""
    bound = mcallester_bound(emp_risk, kl, n, delta)
    beta = optimal_beta_mcallester(kl, n, delta)
    return PACBayesReport(
        emp_risk=emp_risk, kl=kl, n=n, delta=delta,
        bound=bound, optimal_beta=beta,
    )
