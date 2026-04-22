"""Bayesian persuasion — concavification of sender's value function.

Kamenica & Gentzkow (2011). A "sender" chooses a signaling scheme to shape a
"receiver's" posterior beliefs. Receiver then takes an action maximising her
own utility. Sender's problem: choose the signaling that maximises sender's
expected utility over the induced distribution of receiver posteriors.

Key result: the sender's optimal value equals the *concave hull* of her
value-as-a-function-of-receiver-posterior. Numerically we do:

  1. Evaluate sender's reduced-form value V(μ) at a grid of posteriors μ.
  2. Compute the concave hull of the (μ, V) set.
  3. The sender's value at prior μ₀ is the concave-hull value at μ₀.

For CASR, the "sender" is the orchestrator choosing what to reveal; the
"receiver" is the downstream agent. Solves for the information schema that
maximises team task performance subject to agent self-interested action.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class PersuasionReport:
    optimal_value: float        # sender's value at prior
    concave_hull: np.ndarray    # (K, 2) — ordered (μ, V) points on the hull
    prior: float                # 1-D binary prior (extend to simplex later)

    def summary(self) -> str:
        return (
            f"concave-hull value at prior {self.prior:.3f}: "
            f"{self.optimal_value:.4f}"
        )


def concave_hull_1d(mu: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Upper (concave) hull of the 2-D points (mu, v)."""
    pts = np.column_stack([np.asarray(mu, float), np.asarray(v, float)])
    order = np.argsort(pts[:, 0])
    pts = pts[order]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    hull = []
    for p in pts:
        # concave: remove points making a right turn (cross >= 0 means colinear/left)
        while len(hull) >= 2 and cross(hull[-2], hull[-1], p) >= 0:
            hull.pop()
        hull.append(p)
    return np.array(hull)


def bayesian_persuasion_value_1d(
    V: Callable[[float], float],
    prior: float,
    grid: int = 101,
) -> PersuasionReport:
    """Optimal sender value for a 2-state receiver problem with prior μ₀ ∈ [0, 1].

    V(μ) is sender's expected utility *assuming receiver best-responds to μ*.
    """
    if not 0 <= prior <= 1:
        raise ValueError("prior must be in [0, 1]")
    mu = np.linspace(0.0, 1.0, grid)
    v = np.array([V(float(m)) for m in mu])
    hull = concave_hull_1d(mu, v)

    # Interpolate the hull at the prior
    hx = hull[:, 0]
    hy = hull[:, 1]
    opt_value = float(np.interp(prior, hx, hy))
    return PersuasionReport(
        optimal_value=opt_value,
        concave_hull=hull,
        prior=prior,
    )
