"""Drifting consensus — θ*(t) random-walks over time.

Phase 1 task was one-shot: fix θ*, measure convergence once. Real teams
face continually shifting goals. This task generates a trajectory:

    z(0)         ~ N(0, I_r)
    z(t+1)       = z(t) + drift_sigma · ξ_t           ξ_t ~ N(0, I_r)
    θ*(t)        = U z(t)                              (low-rank lift)
    o_i(t)       = θ*(t) + noise_sigma · ε_i,t

A "shock" event optionally teleports z(t) by a large amount at a specified
time to test adaptation speed.

Success criterion is *tracking error*: mean over t of ||x_i(t) - θ*(t)||.
Agents must keep up with the drift using only their compressed protocol.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class DriftingConsensus:
    n_agents: int
    dim: int
    intrinsic_rank: int
    n_steps: int
    noise: float = 1.0
    drift_sigma: float = 0.1
    shock_at: int | None = None       # inject a big jump at this step
    shock_magnitude: float = 5.0
    seed: int = 0

    _basis: np.ndarray = None         # type: ignore   (d, r)
    _trajectory: np.ndarray = None    # type: ignore   (T, d)
    _observations: np.ndarray = None  # type: ignore   (T, N, d)

    def generate(self) -> None:
        rng = np.random.default_rng(self.seed)
        A = rng.standard_normal((self.dim, self.intrinsic_rank))
        Q, _ = np.linalg.qr(A)
        self._basis = Q

        z = rng.standard_normal(self.intrinsic_rank)
        traj = np.zeros((self.n_steps, self.dim))
        for t in range(self.n_steps):
            z = z + self.drift_sigma * rng.standard_normal(self.intrinsic_rank)
            if self.shock_at is not None and t == self.shock_at:
                z = z + self.shock_magnitude * rng.standard_normal(self.intrinsic_rank)
            traj[t] = Q @ z
        self._trajectory = traj

        noise = self.noise * rng.standard_normal(
            (self.n_steps, self.n_agents, self.dim))
        self._observations = traj[:, None, :] + noise

    @property
    def basis(self) -> np.ndarray:
        return self._basis

    @property
    def trajectory(self) -> np.ndarray:
        return self._trajectory

    def observations_at(self, t: int) -> np.ndarray:
        return self._observations[t]

    def truth_at(self, t: int) -> np.ndarray:
        return self._trajectory[t]

    def evaluate_tracking(self, estimates_over_time: np.ndarray) -> dict[str, float]:
        """estimates_over_time: (T, N, d). Returns per-time-averaged errors."""
        T = estimates_over_time.shape[0]
        truth_norms = np.linalg.norm(self._trajectory, axis=1) + 1e-8
        err = np.linalg.norm(
            estimates_over_time - self._trajectory[:, None, :], axis=2)
        mean_rel = float((err.mean(axis=1) / truth_norms).mean())
        max_rel = float((err.max(axis=1) / truth_norms).mean())
        # Oracle: full-dim sample mean each step
        obs_mean = self._observations.mean(axis=1)
        oracle_err = np.linalg.norm(
            obs_mean - self._trajectory, axis=1) / truth_norms
        return {
            "mean_tracking_error": mean_rel,
            "max_tracking_error": max_rel,
            "oracle_tracking_error": float(oracle_err.mean()),
        }
