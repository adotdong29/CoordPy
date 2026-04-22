"""Shapley value by permutation sampling — per-agent credit assignment.

The Shapley value is the unique credit-allocation scheme satisfying efficiency
(Σφ_i = v(N)), symmetry, dummy, and additivity:

    φ_i = (1 / N!) Σ_{π ∈ permutations(N)}
            [ v( S_π(i) ∪ {i} ) − v( S_π(i) ) ]

where S_π(i) is the set of agents preceding i in permutation π.

Computing Shapley exactly costs O(2^N) evaluations of v. Castro–Gómez–Tejada
(2009) give the standard Monte Carlo estimator: sample K random permutations
and average the marginal contribution of each agent at its position in each
sample. By Hoeffding, with v bounded in [0, M],

    Pr[ |φ̂_i − φ_i| > ε ] ≤ 2 exp(−2 K ε² / M²)
    → K ≥ (M² / (2 ε²)) · ln(2 N / δ)    for a union bound over agents.

This module gives both exact (for small N) and MC estimators, plus a helper
to pick K from (ε, δ, M).

The intended use in this repo is agent-level attribution: treat v(S) as the
team's task score when only agents in S act; φ_i is how much agent i shifts the
outcome. This sharpens diagnostics over simple message-traffic counting.
"""

from __future__ import annotations

from itertools import permutations
from typing import Callable

import numpy as np


def exact_shapley(v: Callable[[frozenset], float], n: int) -> np.ndarray:
    """Exact Shapley via enumeration of N! permutations. Use only for n ≤ ~9."""
    if n < 1:
        raise ValueError("n must be ≥ 1")
    phi = np.zeros(n, dtype=float)
    fac = 1.0
    for i in range(1, n + 1):
        fac *= i
    for pi in permutations(range(n)):
        S = set()
        v_prev = v(frozenset(S))
        for agent in pi:
            S.add(agent)
            v_next = v(frozenset(S))
            phi[agent] += (v_next - v_prev) / fac
            v_prev = v_next
    return phi


def monte_carlo_shapley(
    v: Callable[[frozenset], float],
    n: int,
    k_samples: int = 1000,
    seed: int = 0,
) -> np.ndarray:
    """Monte Carlo Shapley via random permutation sampling.

    Each sample draws one permutation uniformly and adds the marginal
    contribution of every agent at its sampled position. Variance ≤ M² / k.
    """
    if n < 1:
        raise ValueError("n must be ≥ 1")
    if k_samples < 1:
        raise ValueError("k_samples must be ≥ 1")
    rng = np.random.default_rng(seed)
    phi = np.zeros(n, dtype=float)
    indices = np.arange(n)
    for _ in range(k_samples):
        rng.shuffle(indices)
        S: set[int] = set()
        v_prev = v(frozenset(S))
        for agent in indices:
            S.add(int(agent))
            v_next = v(frozenset(S))
            phi[int(agent)] += (v_next - v_prev)
            v_prev = v_next
    return phi / k_samples


def samples_for_accuracy(
    n_agents: int, v_range: float, eps: float, delta: float = 0.05,
) -> int:
    """Hoeffding-based sample-size rule: K ≥ (M²/(2ε²)) · ln(2N/δ).

    Guarantees |φ̂_i − φ_i| < ε for all i simultaneously with probability ≥ 1-δ.
    """
    if eps <= 0:
        raise ValueError("eps must be > 0")
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0, 1)")
    return int(np.ceil(v_range ** 2 / (2 * eps ** 2) * np.log(2 * n_agents / delta)))


def efficiency_residual(phi: np.ndarray, v_full: float) -> float:
    """Sum of Shapley values should equal v(N); return the residual."""
    return float(v_full - phi.sum())
