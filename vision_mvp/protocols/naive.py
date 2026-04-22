"""Naive all-to-all broadcast.

Every agent sends its current estimate to every other agent every round.
Each agent maintains the full list of all others' estimates in context and
averages them. This is what every existing multi-agent framework does.

Complexity:
  - Messages per round: N*(N-1) ≈ N²
  - Per-agent context: N*d tokens (everyone's estimate)
  - Total tokens: O(N² * d) per round
"""

from __future__ import annotations
import numpy as np
from ..core.agent import Agent
from ..core.bus import Bus
from ..tasks.consensus import ConsensusTask
from .base import build_agents, estimates_matrix, Result


def run_naive(task: ConsensusTask, rounds: int = 1) -> Result:
    agents = build_agents(task)
    bus = Bus()

    # Naive protocol: in one round, every agent broadcasts its observation
    # to every other agent. Each receiver averages them all in.
    for r in range(rounds):
        # Snapshot all current estimates so the round is synchronous
        snapshot = estimates_matrix(agents)
        weights = np.array([a.accumulated_weight for a in agents])

        for i, a in enumerate(agents):
            incoming_tokens = 0
            for j in range(task.n_agents):
                if i == j:
                    continue
                # Send: estimate vector + weight scalar
                payload = task.dim + 1
                bus.send(sender=j, receiver=i, payload_size=payload,
                         kind="broadcast", round_idx=r)
                incoming_tokens += payload
                # Update receiver
                a.bayesian_update(snapshot[j], float(weights[j]))
            a.remember("broadcast_round", incoming_tokens)
            bus.note_context(a.agent_id, a.current_context_tokens())

    return Result(
        protocol="naive",
        n_agents=task.n_agents,
        dim=task.dim,
        rounds=rounds,
        bus_summary=bus.summary(),
        task_metrics=task.evaluate(estimates_matrix(agents)),
    )
