"""Eigenstate Thermalization diagnostic for multi-agent teams.

The Eigenstate Thermalization Hypothesis (Srednicki 1994; Deutsch 1991) states
that for a chaotic many-body system, the reduced density matrix of any small
subsystem looks thermal with only a few effective parameters. Translated to
multi-agent coordination: a well-mixed CASR team should make every small
sub-team's reduced-covariance spectrum close to the Gibbs spectrum of the
whole team.

This module samples subsystems and measures:

    eth_distance = ‖spec(Cov_sub) − spec(Cov_thermal)‖₁ / ‖spec(Cov_thermal)‖₁

where Cov_sub is the empirical covariance of the subsystem's states over
some window, and Cov_thermal is the truncated spectrum of the whole team's
covariance restricted to the subsystem size. Small eth_distance = team is
"ergodic"; large = subsystem carries information not shared with the bath.

Used as a debugging signal: if eth_distance grows over time, the team is
fragmenting into information silos that the global workspace is not mixing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ETHReport:
    subset_size: int
    n_samples: int
    mean_distance: float
    max_distance: float
    thermal_spec: np.ndarray

    def summary(self) -> str:
        return (
            f"subsets of size {self.subset_size} "
            f"(n={self.n_samples}): "
            f"mean={self.mean_distance:.4f}, max={self.max_distance:.4f}"
        )


def thermal_spectrum(state_trajectory: np.ndarray, k: int) -> np.ndarray:
    """Top-k eigenvalues of the full-team covariance, descending."""
    traj = np.asarray(state_trajectory, dtype=float)
    if traj.ndim != 3:
        raise ValueError("state_trajectory must have shape (T, N, d)")
    T, N, d = traj.shape
    # Flatten to (T, N*d); empirical covariance over time
    flat = traj.reshape(T, N * d)
    # Mean-centered
    flat = flat - flat.mean(axis=0, keepdims=True)
    cov = (flat.T @ flat) / max(T - 1, 1)
    spec = np.sort(np.linalg.eigvalsh(cov))[::-1]
    return spec[:k]


def subsystem_spectrum(state_trajectory: np.ndarray, agent_ids: list[int]) -> np.ndarray:
    """Eigenvalue spectrum of a subsystem's empirical covariance."""
    traj = np.asarray(state_trajectory, dtype=float)
    T, N, d = traj.shape
    sub = traj[:, agent_ids, :].reshape(T, len(agent_ids) * d)
    sub = sub - sub.mean(axis=0, keepdims=True)
    cov = (sub.T @ sub) / max(T - 1, 1)
    return np.sort(np.linalg.eigvalsh(cov))[::-1]


def eth_distance_one_subset(
    state_trajectory: np.ndarray,
    agent_ids: list[int],
) -> float:
    """L¹ distance between normalized subsystem and thermal spectra."""
    sub_spec = subsystem_spectrum(state_trajectory, agent_ids)
    k = sub_spec.size
    therm = thermal_spectrum(state_trajectory, k)
    # Normalize both by their L¹ mass so magnitudes are comparable
    s_sub = sub_spec / max(np.abs(sub_spec).sum(), 1e-12)
    s_th = therm / max(np.abs(therm).sum(), 1e-12)
    return float(np.abs(s_sub - s_th).sum())


def eth_report(
    state_trajectory: np.ndarray,
    subset_size: int = 4,
    n_samples: int = 20,
    seed: int = 0,
) -> ETHReport:
    """Sample `n_samples` random sub-teams and report the ETH-distance stats."""
    traj = np.asarray(state_trajectory, dtype=float)
    T, N, d = traj.shape
    if subset_size > N:
        raise ValueError("subset_size cannot exceed N")
    rng = np.random.default_rng(seed)
    dists = []
    for _ in range(n_samples):
        ids = rng.choice(N, size=subset_size, replace=False).tolist()
        dists.append(eth_distance_one_subset(traj, ids))
    therm = thermal_spectrum(traj, subset_size * d)
    return ETHReport(
        subset_size=subset_size,
        n_samples=n_samples,
        mean_distance=float(np.mean(dists)),
        max_distance=float(np.max(dists)),
        thermal_spec=therm,
    )
