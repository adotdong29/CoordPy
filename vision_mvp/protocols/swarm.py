"""Swarm-physics protocol — VISION_MILLIONS Idea 6.

No workspace, no orchestrator, no explicit router. Each agent follows
three local rules:
  - SEPARATION: repel from agents too similar in embedding space
    (don't duplicate what someone else is already doing).
  - ALIGNMENT: match the direction of your k nearest neighbors' estimates.
  - COHESION: move slightly toward the centroid of your k nearest neighbors.

Coordination emerges from the local potential. No agent holds context
about the whole team; everyone sees only their k nearest neighbors. This
is the Boids / Vicsek model applied to agent state space.

Per-agent peak context: O(k · d) where k is a small constant (typically 5).
Total system messages: O(N · k) — linear in N with tiny constant, no log
factor because no global reduction happens.

Expected behavior:
  - Converges slower than hierarchical (no explicit global reduction).
  - Scales trivially — same math works at N = 10^9 as at N = 10^2.
  - Fully robust to agent failure or Byzantine behavior (a lying agent
    only affects its k neighbors, not the global state).

Implementation uses a simple approximate-nearest-neighbors via small-N
pairwise distance in estimate space. For very large N, swap in an
FAISS-style index — the algorithm still works, just the neighbor lookup
gets faster.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from ..core.bus import Bus
from ..tasks.drifting_consensus import DriftingConsensus


@dataclass
class SwarmResult:
    estimates_over_time: np.ndarray
    bus_summary: dict[str, float]
    task_metrics: dict[str, float]
    mean_neighbor_agreement: list[float]


def run_swarm(task: DriftingConsensus,
              k_neighbors: int = 5,
              alignment: float = 0.25,
              cohesion: float = 0.15,
              separation: float = 0.05,
              forget: float = 0.4) -> SwarmResult:
    """Pure swarm coordination. No workspace, no manifold, no orchestrator."""
    N = task.n_agents
    d = task.dim
    T = task.n_steps

    bus = Bus()
    estimates = task.observations_at(0).copy()
    weights = np.ones(N) / (task.noise ** 2)

    out = np.zeros((T, N, d))
    neighbor_agreement: list[float] = []

    for t in range(T):
        obs = task.observations_at(t)
        # Step 1: absorb own observation with forgetting
        new_estimates = (1 - forget) * estimates + forget * obs

        # Step 2: compute pairwise distances (O(N²) — OK for demo N ≤ 10k).
        # For massive N, swap in a spatial index (kd-tree / HNSW / FAISS).
        # We use squared Euclidean here.
        sqd = np.sum(new_estimates ** 2, axis=1, keepdims=True)
        dists = sqd + sqd.T - 2 * new_estimates @ new_estimates.T
        np.fill_diagonal(dists, np.inf)

        # Step 3: for each agent, get its k nearest neighbors
        k = min(k_neighbors, N - 1)
        neigh_idx = np.argpartition(dists, k, axis=1)[:, :k]    # (N, k)
        neigh_coords = new_estimates[neigh_idx]                  # (N, k, d)

        # Bus accounting: each neighbor exchange costs d tokens
        for i in range(N):
            bus.send(i, -1, k * d, "swarm_exchange", t)
            bus.note_context(i, k * d)

        # Step 4: compute rule-based updates
        centroid = neigh_coords.mean(axis=1)                     # (N, d)
        direction = centroid - new_estimates                      # (N, d) cohesion pull

        # Alignment: move toward mean neighbor direction (velocity proxy =
        # difference from previous estimate, approx via centroid - self)
        align_dir = direction

        # Separation: repel from any neighbor that's *too* close
        too_close = dists < 0.01                                 # fixed threshold
        # Accumulate repulsion: for each neighbor within threshold, push away
        sep_vec = np.zeros_like(new_estimates)
        if np.any(too_close):
            for i in range(N):
                close_js = np.where(too_close[i])[0]
                if len(close_js) > 0:
                    diff = new_estimates[i][None, :] - new_estimates[close_js]
                    nrm = np.linalg.norm(diff, axis=1, keepdims=True) + 1e-6
                    sep_vec[i] = (diff / (nrm ** 2)).sum(axis=0)

        # Apply all three rules
        new_estimates = (new_estimates
                         + cohesion * direction
                         + alignment * align_dir
                         + separation * sep_vec)

        estimates = new_estimates
        out[t] = estimates

        # Monitor — average agreement among each agent's k neighbors
        agree = 1.0 - float(np.mean(np.linalg.norm(
            estimates[:, None, :] - neigh_coords, axis=2).mean(axis=1))
            / (np.linalg.norm(task.trajectory[t]) + 1e-8))
        neighbor_agreement.append(max(0.0, agree))

    metrics = task.evaluate_tracking(out)
    return SwarmResult(
        estimates_over_time=out,
        bus_summary=bus.summary(),
        task_metrics=metrics,
        mean_neighbor_agreement=neighbor_agreement,
    )
