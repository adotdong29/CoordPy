"""Directed-percolation / contact process on agent interaction graphs.

Information-propagation on noisy graphs is in the directed-percolation
universality class (Janssen 1981; Grassberger 1982). Above the critical
transmission probability p_c, an initial seed survives forever with positive
probability; below p_c, it dies out. The critical exponents are universal:

  - β (order parameter): 0.2765…
  - ν_‖ (time correlation): 1.7338…
  - ν_⊥ (space): 1.0968…

This module simulates the basic contact process on a given adjacency matrix:
each "infected" agent infects each neighbor with probability p per round;
each infected agent recovers with probability 1 per round. Measures the
survival probability over time.

Use: feed an empirical footprint-graph adjacency, sweep p, and look for the
inflection where survival flips. Gives an empirical p_c for how well the team
sustains information vs loses it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PercolationReport:
    p: float
    n_trials: int
    rounds: int
    survival_probability: float
    mean_active: float
    extinct_fraction: float

    def summary(self) -> str:
        return (
            f"p={self.p:.3f}: survival={self.survival_probability:.3f}, "
            f"mean active={self.mean_active:.1f}, "
            f"extinct={self.extinct_fraction:.2f}"
        )


def contact_process(
    adjacency: np.ndarray,
    p: float,
    seed_nodes: list[int] | None = None,
    rounds: int = 50,
    seed: int = 0,
) -> np.ndarray:
    """One run of the contact process; returns (rounds+1, N) bool history.

    Rule each round:
      1. Each active node infects each neighbor with prob p.
      2. Each active node then recovers (becomes inactive).
    This is the "all-die" version of the contact process.
    """
    A = (np.asarray(adjacency) > 0).astype(np.int64)
    N = A.shape[0]
    if seed_nodes is None:
        seed_nodes = [0]
    rng = np.random.default_rng(seed)

    active = np.zeros(N, dtype=bool)
    active[seed_nodes] = True
    history = [active.copy()]

    for _ in range(rounds):
        new_active = np.zeros(N, dtype=bool)
        active_idx = np.where(active)[0]
        for i in active_idx:
            # Each neighbor j gets infected indep with probability p
            neighbors = np.where(A[i] > 0)[0]
            if neighbors.size == 0:
                continue
            mask = rng.random(neighbors.size) < p
            new_active[neighbors[mask]] = True
        active = new_active
        history.append(active.copy())
        if not active.any():
            # fill remaining rounds with zeros
            for _ in range(rounds - len(history) + 1):
                history.append(active.copy())
            break

    return np.array(history[: rounds + 1])


def percolation_sweep(
    adjacency: np.ndarray,
    p: float,
    n_trials: int = 100,
    rounds: int = 50,
    seed: int = 0,
) -> PercolationReport:
    """Run `n_trials` independent contact processes; aggregate survival."""
    rng = np.random.default_rng(seed)
    survived = 0
    total_active = 0.0
    extinct = 0
    N = adjacency.shape[0]
    for t in range(n_trials):
        seed_node = int(rng.integers(N))
        history = contact_process(
            adjacency, p, seed_nodes=[seed_node], rounds=rounds,
            seed=int(rng.integers(1 << 31)),
        )
        final_active = int(history[-1].sum())
        total_active += history.sum() / (rounds + 1)
        if final_active > 0:
            survived += 1
        if not history[-1].any():
            extinct += 1

    return PercolationReport(
        p=p,
        n_trials=n_trials,
        rounds=rounds,
        survival_probability=survived / n_trials,
        mean_active=total_active / n_trials,
        extinct_fraction=extinct / n_trials,
    )
