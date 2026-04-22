"""Global Workspace — VISION_MILLIONS Idea 4.

At any moment, only a small subset of agents is "conscious" — allowed to
write into the shared register. The rest read but don't write. Inspired by
Baars' Global Workspace Theory of consciousness: a few salient contents
are broadcast; most processing is unconscious.

Salience = surprise (large prediction error). Each round we pick the top-k
most-surprised agents. k = O(log N) so total write traffic is O(log N)
per round regardless of N — a sublinear scaling we couldn't achieve in
Phase 2 where every agent wrote every round.

We also occasionally pick a random agent (ε-exploration) so agents that
have been quiet for a while get a chance to surface information the rest
don't know about. This mirrors "default mode" intrusions in cognition.
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass


@dataclass
class Workspace:
    n_agents: int
    epsilon: float = 0.05     # fraction of slots for random exploration

    def capacity(self) -> int:
        """How many agents are in the workspace at a time."""
        return max(1, math.ceil(math.log2(max(self.n_agents, 2))))

    def select(self, saliences: np.ndarray, seed: int | None = None) -> np.ndarray:
        """Return indices of agents admitted to the workspace this round.

        Args:
            saliences: (n,) array of per-agent salience scores (higher = more
                likely to be admitted). `n` must equal `self.n_agents`.
            seed: optional RNG seed for the ε-exploration slot.

        Returns:
            (k,) int array of admitted agent indices, where k = capacity().
            Always returns exactly k distinct indices.
        """
        if saliences.ndim != 1:
            raise ValueError(f"saliences must be 1D, got shape {saliences.shape}")
        if len(saliences) != self.n_agents:
            raise ValueError(
                f"saliences length {len(saliences)} != n_agents {self.n_agents}")
        rng = np.random.default_rng(seed)
        k = min(self.capacity(), self.n_agents)
        # Top-k by salience (argpartition is O(N), faster than argsort O(N log N))
        if k >= self.n_agents:
            top = np.arange(self.n_agents)
        else:
            top = np.argpartition(-saliences, k - 1)[:k].copy()
        # ε-exploration: replace last slot with a uniformly random agent.
        # Only triggers with prob `epsilon` and when we have room to swap.
        if rng.random() < self.epsilon and k >= 2:
            rand_idx = int(rng.integers(self.n_agents))
            if rand_idx not in top:
                top[-1] = rand_idx
        return top
