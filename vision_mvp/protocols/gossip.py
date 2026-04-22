"""Gossip averaging — pairwise random exchanges.

Each round, every agent pairs with a random peer and they average their
estimates. Classical distributed computing result: converges to the true
mean in O(log N) rounds with high probability (spectral gap argument).

Complexity:
  - Messages per round: N (one per agent pair)
  - Per-agent context: O(d) — just one peer's estimate at a time
  - Total tokens: O(N * d) per round, but needs O(log N) rounds to converge
  → overall O(N * d * log N)

This is what any "modern" protocol does. Better than naive, but still bad
at scale because we need many rounds and per-message cost is full d.
"""

from __future__ import annotations
import numpy as np
from ..core.bus import Bus
from ..tasks.consensus import ConsensusTask
from .base import build_agents, estimates_matrix, Result


def run_gossip(task: ConsensusTask, rounds: int = 0, seed: int = 1) -> Result:
    if rounds == 0:
        # Auto-scale rounds: enough for convergence at spectral gap 1/log N
        import math
        rounds = max(2, 3 * math.ceil(math.log2(max(task.n_agents, 2))))

    agents = build_agents(task)
    bus = Bus()
    rng = np.random.default_rng(seed)

    for r in range(rounds):
        # Random pairing via permutation
        perm = rng.permutation(task.n_agents)
        for k in range(0, task.n_agents - 1, 2):
            i, j = int(perm[k]), int(perm[k + 1])
            a, b = agents[i], agents[j]
            # Each sends their estimate to the other (payload = d + 1 tokens)
            payload = task.dim + 1
            bus.send(i, j, payload, "gossip", r)
            bus.send(j, i, payload, "gossip", r)
            a_est, a_w = a.estimate.copy(), a.accumulated_weight
            b_est, b_w = b.estimate.copy(), b.accumulated_weight
            # Precision-weighted merge for both
            a.bayesian_update(b_est, b_w)
            b.bayesian_update(a_est, a_w)
            a.remember("gossip", payload)
            b.remember("gossip", payload)
            bus.note_context(a.agent_id, a.current_context_tokens())
            bus.note_context(b.agent_id, b.current_context_tokens())
        # Gossip protocols forget old messages — context is just this round's
        for ag in agents:
            ag.forget_all()

    # Ensure every agent has context accounted (for agents never paired)
    for ag in agents:
        bus.note_context(ag.agent_id, ag.current_context_tokens())

    return Result(
        protocol="gossip",
        n_agents=task.n_agents,
        dim=task.dim,
        rounds=rounds,
        bus_summary=bus.summary(),
        task_metrics=task.evaluate(estimates_matrix(agents)),
    )
