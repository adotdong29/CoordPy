"""Split-conformal prediction for distribution-free surprise calibration.

The CASR Stage-3 surprise filter drops events whose prediction error falls
below a threshold. Currently that threshold is hand-tuned. Conformal prediction
(Vovk, Gammerman, Shafer) gives a *distribution-free* coverage guarantee at any
chosen miscoverage level α ∈ (0, 1):

    Pr[ y ∉ PredSet(x) ] ≤ α     for any exchangeable (x, y) data

The split-conformal variant is trivial to implement and works for any point
predictor:

  1. Train predictor f on a proper-training set.
  2. On a held-out calibration set of size n_cal, compute nonconformity scores
     s_i = |y_i − f(x_i)|.
  3. Let q̂ = ⌈(n_cal + 1)(1 − α)⌉/n_cal-th quantile of {s_i}.
  4. PredSet(x_new) = [f(x_new) − q̂, f(x_new) + q̂].

Step 3 is *finite-sample exact* under exchangeability (Theorem 3 of Lei–Wasserman
et al., 2018). No distributional assumption required.

Used here to replace the ad-hoc `surprise_threshold` in `api.CASRRouter` with
one that has a provable miscoverage bound.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ConformalCalibration:
    q_hat: float                # half-width of the prediction interval
    n_cal: int
    alpha: float

    def contains(self, pred: float, obs: float) -> bool:
        """Does [pred − q̂, pred + q̂] contain obs?"""
        return abs(obs - pred) <= self.q_hat

    def is_surprising(self, pred: float, obs: float) -> bool:
        """Inverse of `contains` — events outside the prediction set."""
        return not self.contains(pred, obs)


def calibrate(
    predictions: np.ndarray,
    observations: np.ndarray,
    alpha: float = 0.1,
) -> ConformalCalibration:
    """Fit a symmetric-interval conformal predictor from calibration data.

    `predictions` and `observations` must be 1D arrays of equal length — the
    predictor's outputs and the realized targets on the held-out calibration
    split. `alpha` is the target miscoverage rate (e.g. 0.1 → 90% coverage).
    """
    pred = np.asarray(predictions, dtype=float).ravel()
    obs = np.asarray(observations, dtype=float).ravel()
    if pred.shape != obs.shape:
        raise ValueError("predictions and observations must have the same shape")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    n = pred.size
    if n < 1:
        raise ValueError("need at least 1 calibration point")

    scores = np.abs(pred - obs)
    # Finite-sample-exact quantile level — note the (n+1) correction.
    level = np.ceil((n + 1) * (1 - alpha)) / n
    level = min(level, 1.0)
    q_hat = float(np.quantile(scores, level, method="higher"))
    return ConformalCalibration(q_hat=q_hat, n_cal=n, alpha=alpha)


def empirical_coverage(
    cal: ConformalCalibration,
    predictions: np.ndarray,
    observations: np.ndarray,
) -> float:
    """Fraction of test points whose observation fell inside the interval."""
    pred = np.asarray(predictions, dtype=float).ravel()
    obs = np.asarray(observations, dtype=float).ravel()
    if pred.shape != obs.shape:
        raise ValueError("shape mismatch")
    return float(np.mean(np.abs(obs - pred) <= cal.q_hat))


def vectorized_contains(
    cal: ConformalCalibration,
    predictions: np.ndarray,
    observations: np.ndarray,
) -> np.ndarray:
    """Boolean array per (pred, obs) pair — batched `contains`."""
    pred = np.asarray(predictions, dtype=float)
    obs = np.asarray(observations, dtype=float)
    return np.abs(obs - pred) <= cal.q_hat


class OnlineConformal:
    """Sliding-window conformal predictor for streaming settings.

    Maintains the most-recent `window` nonconformity scores and recomputes q̂
    each time `observe()` is called. Not finite-sample-exact under
    distribution shift, but empirically strong.
    """

    def __init__(self, window: int = 500, alpha: float = 0.1):
        if window < 1:
            raise ValueError("window must be ≥ 1")
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        self.window = window
        self.alpha = alpha
        self._scores: list[float] = []

    @property
    def q_hat(self) -> float:
        if not self._scores:
            return float("inf")
        n = len(self._scores)
        level = min(np.ceil((n + 1) * (1 - self.alpha)) / n, 1.0)
        return float(np.quantile(self._scores, level, method="higher"))

    def observe(self, pred: float, obs: float) -> None:
        self._scores.append(abs(obs - pred))
        if len(self._scores) > self.window:
            self._scores = self._scores[-self.window:]

    def is_surprising(self, pred: float, obs: float) -> bool:
        return abs(obs - pred) > self.q_hat
