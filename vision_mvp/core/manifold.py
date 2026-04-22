"""Shared Latent Manifold (SLM).

The manifold is a low-dimensional (m ≪ N) shared state that all agents can
project into and read from. Writes are projections of agent state; reads
retrieve a compressed summary of collective state.

Key idea from VISION_MILLIONS Idea 1: coordination happens through geometry
on the manifold, not direct messages. For a vector-consensus task, the
manifold tracks a running mean (the "collective opinion") and dispersion.

We implement it as a weighted Welford-style running mean that supports:
  - write(agent_id, vector, weight): project this agent's estimate
  - read(): O(m) summary of collective state
  - dimension m is chosen = ceil(log2(N)) to match the theoretical bound.

Importantly, "reading" costs exactly m tokens of context — the manifold
dimension — regardless of team size N. This is the promised O(log N)
per-agent cost.
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass


@dataclass
class Manifold:
    dim_input: int       # dimension of agent state (d)
    dim_manifold: int    # m = O(log N)
    # Random projection matrix R ∈ ℝ^{m × d}. Shared by all agents.
    # Johnson-Lindenstrauss says random projections preserve geometry
    # up to ε distortion with m = O(log N / ε²).
    _projection: np.ndarray = None  # type: ignore
    # Running statistics in projected space
    _sum_proj: np.ndarray = None    # type: ignore
    _weight: float = 0.0
    # We also keep the inverse-ish map (pseudo-inverse) so agents can
    # reconstruct a d-dim estimate from the m-dim manifold summary.
    _pinv: np.ndarray = None        # type: ignore

    @classmethod
    def build(cls, dim_input: int, n_agents: int, seed: int = 0,
              basis: np.ndarray | None = None) -> "Manifold":
        """Build manifold with projection R ∈ ℝ^{m × d}.

        If `basis` (d, m) is provided, use R = basis.T (exact task-relevant
        subspace projection — no information loss within that subspace).
        Otherwise, JL random projection.
        """
        if basis is not None:
            if basis.ndim != 2:
                raise ValueError(
                    f"basis must be 2D (d, m), got shape {basis.shape}")
            if basis.shape[0] != dim_input:
                raise ValueError(
                    f"basis first dim {basis.shape[0]} != dim_input {dim_input}")
            # Columns of basis span the task-relevant subspace; project onto it.
            R = basis.T                           # (m, d)
            dim_manifold = R.shape[0]
            # Left-inverse for orthonormal columns: pinv = basis itself
            pinv = basis                          # (d, m)
        else:
            dim_manifold = max(2, math.ceil(math.log2(max(n_agents, 2))))
            rng = np.random.default_rng(seed)
            R = rng.standard_normal((dim_manifold, dim_input)) / math.sqrt(dim_manifold)
            pinv = np.linalg.pinv(R)
        return cls(
            dim_input=dim_input,
            dim_manifold=dim_manifold,
            _projection=R,
            _sum_proj=np.zeros(dim_manifold),
            _weight=0.0,
            _pinv=pinv,
        )

    def project(self, vector: np.ndarray) -> np.ndarray:
        return self._projection @ vector

    def write(self, projected: np.ndarray, weight: float = 1.0) -> None:
        self._sum_proj = self._sum_proj + weight * projected
        self._weight += weight

    def read(self) -> np.ndarray:
        """Return the current manifold summary (projected mean)."""
        if self._weight <= 0:
            return np.zeros(self.dim_manifold)
        return self._sum_proj / self._weight

    def reconstruct(self, manifold_state: np.ndarray) -> np.ndarray:
        """Reconstruct an approximate d-dim estimate from m-dim summary.

        This is lossy — JL projection loses the components orthogonal to
        the row space of R. But the low-dim manifold summary is sufficient
        for consensus in the task-relevant subspace.
        """
        return self._pinv @ manifold_state

    def write_cost(self) -> int:
        """Tokens to write (project) an update: we send m floats."""
        return self.dim_manifold

    def read_cost(self) -> int:
        """Tokens to read the manifold summary: m floats."""
        return self.dim_manifold
