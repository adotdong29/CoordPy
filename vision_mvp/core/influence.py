"""Influence estimation + Friedgut junta extraction.

For a Boolean function f : {0, 1}^N → {0, 1}, the influence of variable i is

    Inf_i(f) = Pr_x[ f(x) ≠ f(x ⊕ e_i) ]

and the total influence is I(f) = Σ_i Inf_i(f). Friedgut's junta theorem
(1998) says: for every ε > 0, f is O(exp(I(f)/ε²))-close to a junta on
O(1) variables (more precisely, ε-close to depending on only
2^(O(I(f)/ε²)) coordinates).

Application to CASR workspace admission: treat the team's joint decision as
a Boolean-function-like object on per-agent indicators, estimate each agent's
influence via Monte Carlo, and admit only high-influence agents to the
workspace. Principled alternative to surprise-based admission.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class InfluenceReport:
    influences: np.ndarray       # (N,) per-variable influence
    total_influence: float       # Σ Inf_i
    junta: list[int]             # ε-junta indices
    epsilon: float

    def summary(self) -> str:
        return (
            f"I(f) = {self.total_influence:.3f}, "
            f"ε-junta of size {len(self.junta)} at ε={self.epsilon:.3f}"
        )


def monte_carlo_influence(
    f: Callable[[np.ndarray], int],
    n_vars: int,
    n_samples: int = 500,
    seed: int = 0,
) -> np.ndarray:
    """Estimate per-variable influence of Boolean-valued f via MC.

    f takes a length-`n_vars` 0/1 vector, returns 0 or 1.
    Samples random x and compares f(x) with f(x ⊕ e_i) for each i.
    """
    rng = np.random.default_rng(seed)
    inf = np.zeros(n_vars, dtype=float)
    for _ in range(n_samples):
        x = rng.integers(0, 2, size=n_vars).astype(np.int8)
        f_x = int(f(x))
        for i in range(n_vars):
            x_flip = x.copy()
            x_flip[i] = 1 - x_flip[i]
            f_flip = int(f(x_flip))
            if f_flip != f_x:
                inf[i] += 1.0
    return inf / n_samples


def extract_junta(influences: np.ndarray, epsilon: float = 0.05) -> list[int]:
    """Return indices with influence ≥ ε, sorted descending by influence."""
    order = np.argsort(-influences)
    return [int(i) for i in order if influences[int(i)] >= epsilon]


def influence_report(
    f: Callable[[np.ndarray], int],
    n_vars: int,
    n_samples: int = 500,
    epsilon: float = 0.05,
    seed: int = 0,
) -> InfluenceReport:
    inf = monte_carlo_influence(f, n_vars, n_samples, seed)
    junta = extract_junta(inf, epsilon)
    return InfluenceReport(
        influences=inf,
        total_influence=float(inf.sum()),
        junta=junta,
        epsilon=epsilon,
    )
