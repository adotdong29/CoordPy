"""Common protocol runner scaffold."""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from ..core.agent import Agent
from ..core.bus import Bus
from ..tasks.consensus import ConsensusTask


def build_agents(task: ConsensusTask) -> list[Agent]:
    agents: list[Agent] = []
    for i in range(task.n_agents):
        a = Agent(agent_id=i, observation=task.observations[i].copy(),
                  obs_weight=1.0 / (task.noise ** 2))
        agents.append(a)
    return agents


def estimates_matrix(agents: list[Agent]) -> np.ndarray:
    return np.stack([a.estimate for a in agents])


@dataclass
class Result:
    protocol: str
    n_agents: int
    dim: int
    rounds: int
    bus_summary: dict[str, float]
    task_metrics: dict[str, float]

    def as_row(self) -> dict:
        row = {
            "protocol": self.protocol,
            "N": self.n_agents,
            "d": self.dim,
            "rounds": self.rounds,
        }
        row.update(self.bus_summary)
        row.update(self.task_metrics)
        return row
