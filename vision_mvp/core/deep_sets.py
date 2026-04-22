"""Deep Sets — permutation-equivariant aggregation (numpy forward pass).

Zaheer et al. (NeurIPS 2017). Any permutation-invariant function of a set of
vectors {x_1, …, x_n} can be written as

    f(X) = ρ( Σ_i φ(x_i) )

for some φ and ρ. This gives a principled "synthesizer" architecture for CASR:
each agent contributes φ(state_i); the synthesizer computes the sum and
applies ρ; result is invariant to agent ordering by construction.

This module ships only the *inference-time* forward pass (numpy). Training
requires autograd (tier 3 with torch). For CASR Phase 8-9 synthesis, the
forward pass is sufficient once φ and ρ are provided (either trained
offline or hand-designed).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class DeepSetsSynth:
    """Set-to-vector synthesizer with hand-specified φ, ρ.

    phi: vector → embedding (same dim for all)
    rho: embedding → output
    reducer: how to combine embeddings (sum, mean, max). sum gives formal
    permutation-equivariance under continuous ρ (Zaheer et al. Thm 2).
    """

    phi: Callable[[np.ndarray], np.ndarray]
    rho: Callable[[np.ndarray], np.ndarray]
    reducer: str = "sum"

    def forward(self, X: np.ndarray) -> np.ndarray:
        """X is (N, d). Returns ρ(Σ_i φ(x_i))."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be (N, d)")
        embeddings = np.stack([self.phi(row) for row in X], axis=0)
        if self.reducer == "sum":
            agg = embeddings.sum(axis=0)
        elif self.reducer == "mean":
            agg = embeddings.mean(axis=0)
        elif self.reducer == "max":
            agg = embeddings.max(axis=0)
        else:
            raise ValueError(f"unknown reducer {self.reducer!r}")
        return self.rho(agg)


def identity_phi(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=float)


def identity_rho(z: np.ndarray) -> np.ndarray:
    return np.asarray(z, dtype=float)


def mean_synthesizer(dim: int) -> DeepSetsSynth:
    """The trivial permutation-invariant synth: average over embeddings."""
    return DeepSetsSynth(phi=identity_phi, rho=identity_rho, reducer="mean")
