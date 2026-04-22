"""Differential Privacy — Laplace / Gaussian mechanisms + composition.

Dwork & Roth (2014). For a query q with L₁-sensitivity Δ_1, the Laplace
mechanism adds Lap(Δ_1 / ε) noise, achieving (ε, 0)-DP. For L₂-sensitivity
Δ_2, the Gaussian mechanism with stddev σ = Δ_2 √(2 ln(1.25/δ)) / ε achieves
(ε, δ)-DP.

Composition (basic): releasing k queries at (ε, δ) each is (k ε, k δ)-DP.
Advanced composition tightens this to O(√k).

Used here to add formal privacy guarantees to the shared-manifold
aggregations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def laplace_noise(scale: float, shape, rng: np.random.Generator) -> np.ndarray:
    return rng.laplace(0.0, scale, size=shape)


def gaussian_noise(std: float, shape, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(0.0, std, size=shape)


@dataclass
class LaplaceMechanism:
    """Releases f(x) + Lap(Δ_1/ε) noise, guaranteeing (ε, 0)-DP."""
    sensitivity: float
    epsilon: float

    def __post_init__(self):
        if self.sensitivity <= 0:
            raise ValueError("sensitivity must be > 0")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be > 0")

    def release(self, value: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        value = np.asarray(value, dtype=float)
        scale = self.sensitivity / self.epsilon
        return value + laplace_noise(scale, value.shape, rng)


@dataclass
class GaussianMechanism:
    """Releases f(x) + N(0, σ² I) with σ = Δ_2 √(2 ln(1.25/δ))/ε → (ε, δ)-DP."""
    sensitivity: float
    epsilon: float
    delta: float

    def __post_init__(self):
        if not 0 < self.delta < 1:
            raise ValueError("delta must be in (0, 1)")
        if self.sensitivity <= 0 or self.epsilon <= 0:
            raise ValueError("sensitivity and epsilon must be > 0")

    @property
    def std(self) -> float:
        return self.sensitivity * np.sqrt(2.0 * np.log(1.25 / self.delta)) / self.epsilon

    def release(self, value: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        value = np.asarray(value, dtype=float)
        return value + gaussian_noise(self.std, value.shape, rng)


def basic_composition(eps: float, delta: float, k: int) -> tuple[float, float]:
    """Basic composition: k uses of (ε, δ) mechanism → (k ε, k δ)-DP."""
    if k < 1:
        raise ValueError("k must be ≥ 1")
    return k * eps, k * delta


def advanced_composition(
    eps: float, delta: float, k: int, target_delta: float = 1e-6,
) -> tuple[float, float]:
    """Dwork et al. advanced composition: tighter (ε', δ' + target_delta).

    For k adaptively-composed (ε, δ) mechanisms,
        ε' = √(2k ln(1/target_delta)) · ε + k · ε · (e^ε − 1)
        δ' = k δ + target_delta
    """
    if k < 1:
        raise ValueError("k must be ≥ 1")
    eps_prime = np.sqrt(2 * k * np.log(1.0 / target_delta)) * eps + k * eps * (np.exp(eps) - 1)
    delta_prime = k * delta + target_delta
    return float(eps_prime), float(delta_prime)


def shuffle_then_aggregate(
    local_responses: np.ndarray, rng: np.random.Generator,
) -> np.ndarray:
    """Fisher-Yates shuffle of per-agent responses before aggregation.

    Adds anonymity: the aggregator sees no mapping from response to agent.
    """
    arr = np.asarray(local_responses).copy()
    idx = np.arange(arr.shape[0])
    rng.shuffle(idx)
    return arr[idx]
