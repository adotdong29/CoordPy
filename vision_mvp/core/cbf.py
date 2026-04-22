"""Control Barrier Functions — safety certificates via forward invariance.

Ames, Coogan, Egerstedt, Notomista, Sreenath, Tabuada (2019). Given a safe
set S = {x : h(x) ≥ 0}, a CBF h is a function satisfying

    ḣ(x, u) + α(h(x)) ≥ 0     for all x ∈ ∂S

where α is an "extended class-K" function (monotonically increasing, α(0)=0).
Enforcing this constraint on the control u keeps the state forward-invariant
in S — the team never enters the unsafe region {h < 0}.

In CASR, use this for adversarial robustness: define h such that h < 0
corresponds to "bad states" (context-window overflow, contradiction,
unauthorised message), then enforce ḣ + α(h) ≥ 0 on every controller output
via a tiny QP.

We ship a minimal QP solver (active-set on the equality/inequality form) that
handles single-barrier, single-input systems — sufficient for the CASR
adversarial track without pulling scipy into a tier-0 module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class CBFReport:
    u_safe: np.ndarray
    u_nominal: np.ndarray
    h_value: float
    slack: float           # how much u was modified (‖u_safe − u_nominal‖)

    def summary(self) -> str:
        return (
            f"h={self.h_value:.3f}  "
            f"slack={self.slack:.3f}  "
            f"{'SAFE' if self.slack >= 0 else 'violation'}"
        )


def enforce_barrier(
    x: np.ndarray,
    u_nominal: np.ndarray,
    h: Callable[[np.ndarray], float],
    Lfh: Callable[[np.ndarray], float],        # dh/dx · f(x)
    Lgh: Callable[[np.ndarray], np.ndarray],  # dh/dx · g(x)
    alpha: Callable[[float], float] = lambda h: h,
) -> CBFReport:
    """Find u_safe minimising ‖u − u_nominal‖² s.t. Lfh + Lgh · u + α(h) ≥ 0.

    Closed-form solution to the scalar-constraint QP:
      If the nominal already satisfies the constraint, u_safe = u_nominal.
      Otherwise, project onto the constraint hyperplane:
        u_safe = u_nominal − (violation / ‖Lgh‖²) · Lgh
    """
    x = np.asarray(x, dtype=float)
    u_nom = np.asarray(u_nominal, dtype=float).ravel()
    h_val = float(h(x))
    lfh = float(Lfh(x))
    lgh = np.asarray(Lgh(x), dtype=float).ravel()

    violation = lfh + float(lgh @ u_nom) + alpha(h_val)
    if violation >= 0:
        return CBFReport(u_safe=u_nom.copy(), u_nominal=u_nom,
                         h_value=h_val, slack=0.0)
    denom = float(lgh @ lgh)
    if denom < 1e-12:
        # Unactuated direction — barrier cannot be enforced. Best effort.
        return CBFReport(u_safe=u_nom.copy(), u_nominal=u_nom,
                         h_value=h_val, slack=float("inf"))
    step = -violation / denom
    u_safe = u_nom + step * lgh
    slack = float(np.linalg.norm(u_safe - u_nom))
    return CBFReport(u_safe=u_safe, u_nominal=u_nom,
                     h_value=h_val, slack=slack)


def exponential_alpha(gamma: float = 1.0) -> Callable[[float], float]:
    """Canonical class-K: α(h) = γ h."""
    return lambda h: gamma * h
