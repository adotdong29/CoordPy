"""Distributed Vector Consensus Task.

There is a hidden truth θ* ∈ ℝ^d. Each of N agents receives a noisy
observation o_i = θ* + ε_i where ε_i ~ N(0, σ²I). Agents must cooperate
to produce estimates x_i that are jointly close to θ*.

Success criterion — a team "understands perfectly" if:
    (1)  Accuracy: mean_i ||x_i - θ*|| / ||θ*|| < ε_acc
    (2)  Agreement: max_{i,j} ||x_i - x_j|| / ||θ*|| < ε_agree

This is the million-agent equivalent of everyone-in-a-meeting converging
on the same plan that's also correct. It's non-trivial because:
  - No single agent sees θ* directly
  - The optimal estimate x* = (1/N) Σ o_i requires coordination
  - Naive broadcasting is O(N²)

The task gives us clear, quantitative metrics for "perfect understanding"
that don't require an LLM to grade anything.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class ConsensusTask:
    n_agents: int
    dim: int
    noise: float = 1.0
    seed: int = 0
    # If intrinsic_rank < dim, the truth lies in a low-dim subspace — reflecting
    # the CASR assumption that tasks have low intrinsic complexity even when
    # surface state is high-dim. If None, rank == dim (full complexity).
    intrinsic_rank: int | None = None
    _truth: np.ndarray = None              # type: ignore
    _observations: np.ndarray = None       # type: ignore
    _basis: np.ndarray | None = None       # (d, rank) orthonormal basis, if low-rank

    def generate(self) -> None:
        rng = np.random.default_rng(self.seed)
        rank = self.intrinsic_rank if self.intrinsic_rank is not None else self.dim
        if rank < self.dim:
            # Random orthonormal basis for the task-relevant subspace
            A = rng.standard_normal((self.dim, rank))
            Q, _ = np.linalg.qr(A)
            self._basis = Q  # (d, rank)
            # Hidden truth is a random direction within that subspace
            z = rng.standard_normal(rank)
            self._truth = Q @ z
        else:
            self._basis = None
            self._truth = rng.standard_normal(self.dim)
        # Per-agent noisy observations (noise in full d-dim space)
        noise = self.noise * rng.standard_normal((self.n_agents, self.dim))
        self._observations = self._truth[None, :] + noise

    @property
    def basis(self) -> np.ndarray | None:
        return self._basis

    @property
    def truth(self) -> np.ndarray:
        if self._truth is None:
            raise RuntimeError("Call generate() first")
        return self._truth

    @property
    def observations(self) -> np.ndarray:
        if self._observations is None:
            raise RuntimeError("Call generate() first")
        return self._observations

    def evaluate(self, estimates: np.ndarray) -> dict[str, float]:
        """estimates: (N, d) array of agent estimates."""
        truth_norm = float(np.linalg.norm(self._truth))
        # Accuracy: mean distance to truth
        err = np.linalg.norm(estimates - self._truth[None, :], axis=1)
        mean_acc = float(err.mean()) / truth_norm
        max_acc = float(err.max()) / truth_norm
        # Agreement: max pairwise distance (use std across agents as proxy
        # since N² pairwise would be expensive)
        std_per_dim = estimates.std(axis=0)
        agreement = float(np.linalg.norm(std_per_dim)) / truth_norm
        # Optimal possible accuracy: use (1/N) Σ o_i
        obs_mean = self._observations.mean(axis=0)
        oracle_err = float(np.linalg.norm(obs_mean - self._truth)) / truth_norm
        return {
            "mean_accuracy_error": mean_acc,    # lower is better
            "max_accuracy_error": max_acc,
            "agreement_error": agreement,        # lower is better
            "oracle_error": oracle_err,          # best achievable
        }
