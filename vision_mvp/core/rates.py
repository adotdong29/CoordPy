"""Berger-Tung inner bound for distributed Gaussian source coding.

Setup (Berger 1977; Tung 1978): two correlated Gaussian sources X_1, X_2 are
to be encoded separately and jointly decoded, with mean-squared-error
distortion constraints D_1, D_2. The achievable rate region's inner bound is

    R_1 ≥ ½ log⁺ ( Var(X_1 | U_2) / D_1 )
    R_2 ≥ ½ log⁺ ( Var(X_2 | U_1) / D_2 )
    R_1 + R_2 ≥ ½ log⁺ ( det Σ / (D_1 D_2) )

where (U_1, U_2) are auxiliary Gaussians with covariance tuned by the
decoder. For jointly Gaussian (X_1, X_2) ~ N(0, Σ) with correlation ρ, this
simplifies into closed form.

Used as a *capacity diagnostic* in CASR: given the correlation structure of
two agents' observations and the acceptable distortion, report the minimum
joint rate needed for a shared-decoder coordination protocol.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BergerTungReport:
    r1_individual: float
    r2_individual: float
    r_sum_joint: float

    def summary(self) -> str:
        return (
            f"R₁ ≥ {self.r1_individual:.3f}  "
            f"R₂ ≥ {self.r2_individual:.3f}  "
            f"R₁+R₂ ≥ {self.r_sum_joint:.3f}"
        )


def berger_tung_gaussian(
    sigma1: float, sigma2: float, rho: float,
    D1: float, D2: float,
) -> BergerTungReport:
    """Closed-form Berger-Tung inner bound for a 2-source Gaussian problem.

    sigma1, sigma2  : source standard deviations
    rho             : Pearson correlation ∈ (-1, 1)
    D1, D2          : per-source MSE distortion constraints
    """
    if not (0 < D1 <= sigma1 ** 2 and 0 < D2 <= sigma2 ** 2):
        raise ValueError("distortions must be positive and ≤ source variance")
    if not -1 < rho < 1:
        raise ValueError("rho must be in (-1, 1)")

    Sigma = np.array([
        [sigma1 ** 2, rho * sigma1 * sigma2],
        [rho * sigma1 * sigma2, sigma2 ** 2],
    ])
    det = float(np.linalg.det(Sigma))

    # Conditional variances
    var_x1_given_x2 = sigma1 ** 2 * (1 - rho ** 2)
    var_x2_given_x1 = sigma2 ** 2 * (1 - rho ** 2)

    r1 = 0.5 * max(np.log2(var_x1_given_x2 / D1), 0.0) if D1 < var_x1_given_x2 else 0.0
    r2 = 0.5 * max(np.log2(var_x2_given_x1 / D2), 0.0) if D2 < var_x2_given_x1 else 0.0
    rsum = 0.5 * max(np.log2(det / (D1 * D2)), 0.0)

    return BergerTungReport(
        r1_individual=float(r1), r2_individual=float(r2), r_sum_joint=float(rsum),
    )
