"""Minimal agent model — no LLM, just a Bayesian estimator.

Each agent holds a noisy observation of a hidden vector θ*. Its goal is to
produce an estimate x_i that's close to θ*. Through whatever protocol is in
use, it updates x_i by incorporating information from others.

Without LLM, the agent is a simple running-average accumulator. The interest
is not in the agent's cleverness but in how protocols affect its ability to
converge to the correct answer with minimal context.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field


@dataclass
class Agent:
    agent_id: int
    observation: np.ndarray           # noisy view of θ*
    obs_weight: float = 1.0           # confidence in own observation
    estimate: np.ndarray = None       # type: ignore  # current best estimate
    accumulated_weight: float = 0.0
    # Context buffer: things this agent currently "holds" (for accounting)
    _context: list[tuple[str, int]] = field(default_factory=list)  # (kind, tokens)

    def __post_init__(self):
        if self.estimate is None:
            self.estimate = self.observation.copy()
            self.accumulated_weight = self.obs_weight

    def current_context_tokens(self) -> int:
        return sum(t for _, t in self._context)

    def remember(self, kind: str, tokens: int) -> None:
        self._context.append((kind, tokens))

    def forget_all(self) -> None:
        self._context = []

    def bayesian_update(self, other_estimate: np.ndarray, other_weight: float) -> None:
        """Incorporate an external estimate. Precision-weighted average."""
        new_weight = self.accumulated_weight + other_weight
        if new_weight <= 0:
            return
        self.estimate = (
            self.accumulated_weight * self.estimate + other_weight * other_estimate
        ) / new_weight
        self.accumulated_weight = new_weight

    def disagreement(self, other: "Agent") -> float:
        return float(np.linalg.norm(self.estimate - other.estimate))
