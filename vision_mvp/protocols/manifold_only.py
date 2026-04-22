"""Shared Latent Manifold only (VISION_MILLIONS Idea 1).

Each round:
  1. Every agent projects its estimate to the shared manifold (m-dim).
  2. Manifold accumulates all projections (streaming weighted mean).
  3. Every agent reads the manifold summary (m floats).
  4. Agent reconstructs an estimate in full dim and Bayesian-merges.

Per-agent context cost: just m = ceil(log2 N) tokens from reading the
manifold summary — regardless of team size N.

Total tokens: N write operations (m tokens each) + N read ops (m tokens).
→ O(N * log N) total per round, vs O(N²) naive / O(N log N) gossip.

But we need only O(1) rounds for convergence in the consensus case, unlike
gossip which needs O(log N) rounds — because the manifold is a global
reduction in one shot.
"""

from __future__ import annotations
import numpy as np
from ..core.bus import Bus
from ..core.manifold import Manifold
from ..tasks.consensus import ConsensusTask
from .base import build_agents, estimates_matrix, Result


def run_manifold_only(task: ConsensusTask, rounds: int = 1) -> Result:
    agents = build_agents(task)
    bus = Bus()
    # Use the task's intrinsic basis if available (e.g., a shared pretrained
    # embedding in a real LLM deployment). Otherwise, fall back to JL.
    manifold = Manifold.build(task.dim, task.n_agents, basis=task.basis)

    for r in range(rounds):
        # Phase 1: every agent writes its projection
        for a in agents:
            proj = manifold.project(a.estimate)
            manifold.write(proj, weight=a.accumulated_weight)
            # Cost: sending m-dim projection to the manifold
            bus.send(a.agent_id, -1, manifold.write_cost(),
                     "manifold_write", r)

        # Phase 2: every agent reads the summary and updates
        summary = manifold.read()
        for a in agents:
            # Read cost: m tokens per agent
            bus.send(-1, a.agent_id, manifold.read_cost(),
                     "manifold_read", r)
            a.forget_all()
            a.remember("manifold_summary", manifold.read_cost())

            # Reconstruct a d-dim estimate and merge Bayesian-style.
            reconstructed = manifold.reconstruct(summary)
            # Treat the manifold summary as a pseudo-observation with weight
            # equal to the total mass in the manifold minus this agent's
            # own contribution (to avoid double-counting)
            effective_weight = max(manifold._weight - a.accumulated_weight, 0.0)
            a.bayesian_update(reconstructed, effective_weight)
            bus.note_context(a.agent_id, a.current_context_tokens())

    return Result(
        protocol="manifold",
        n_agents=task.n_agents,
        dim=task.dim,
        rounds=rounds,
        bus_summary=bus.summary(),
        task_metrics=task.evaluate(estimates_matrix(agents)),
    )
