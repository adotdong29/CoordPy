"""Full vision-stack: Manifold + Surprise-filtered stigmergic writes.

For the consensus task (every agent shares the same problem), the key
stigmergic benefit is NOT spatial partitioning but CRDT-style aggregation
with a surprise filter. An agent writes only when its current projection
differs meaningfully from what it last wrote (CASR Stage 3 / predictive
coding). After a few rounds, most agents have stabilized and don't write.

This combines:
  - SLM (Idea 1): shared latent manifold for global coherence
  - CRDT register (Idea 3 specialized): single environment accumulating
    precision-weighted projections
  - Surprise filter (CASR Stage 3): skip writes below threshold

Expected result vs manifold-only: similar accuracy, fewer messages (the
surprise filter kills ~half the writes after round 1).

Per-agent peak context: O(m) = O(log N) — same as manifold-only. Total
system messages: smaller than manifold-only because many writes are skipped.
"""

from __future__ import annotations
import numpy as np
from ..core.bus import Bus
from ..core.manifold import Manifold
from ..tasks.consensus import ConsensusTask
from .base import build_agents, estimates_matrix, Result


def run_full(task: ConsensusTask,
             rounds: int = 2,
             surprise_tau: float = 0.02) -> Result:
    agents = build_agents(task)
    bus = Bus()
    manifold = Manifold.build(task.dim, task.n_agents, basis=task.basis)

    # Predicted last-written projection per agent (for surprise filter).
    # Initialize to "never written" = sentinel NaN so round-0 writes go through.
    last_written = {a.agent_id: None for a in agents}

    for r in range(rounds):
        # Phase 1: surprise-filtered writes
        writes_this_round = 0
        for a in agents:
            proj = manifold.project(a.estimate)
            prev = last_written[a.agent_id]
            if prev is None:
                surprise = float("inf")
            else:
                # Surprise as relative change
                surprise = float(np.linalg.norm(proj - prev))
            if surprise > surprise_tau:
                manifold.write(proj, weight=a.accumulated_weight)
                bus.send(a.agent_id, -1, manifold.write_cost(),
                         "stigmergy_write", r)
                last_written[a.agent_id] = proj
                writes_this_round += 1

        # Phase 2: everyone reads the manifold summary and updates
        summary = manifold.read()
        for a in agents:
            bus.send(-1, a.agent_id, manifold.read_cost(),
                     "stigmergy_read", r)
            a.forget_all()
            a.remember("manifold_summary", manifold.read_cost())

            reconstructed = manifold.reconstruct(summary)
            effective_weight = max(manifold._weight - a.accumulated_weight, 0.0)
            a.bayesian_update(reconstructed, effective_weight)
            bus.note_context(a.agent_id, a.current_context_tokens())

    return Result(
        protocol="full",
        n_agents=task.n_agents,
        dim=task.dim,
        rounds=rounds,
        bus_summary=bus.summary(),
        task_metrics=task.evaluate(estimates_matrix(agents)),
    )
