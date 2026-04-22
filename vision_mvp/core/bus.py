"""Communication bus — tracks every byte of context exchanged.

The bus itself does not route; each protocol decides what to send. The bus
just accounts for traffic so we can compute per-agent context size and total
token cost across protocols.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Message:
    sender: int
    receiver: int | None  # None = broadcast
    payload_size: int      # simulated "tokens" (we use float/int count)
    kind: str              # "obs", "delta", "manifold_read", etc.
    round: int


@dataclass
class Bus:
    messages: list[Message] = field(default_factory=list)
    # Per-agent context retention (tokens currently held by each agent)
    _agent_context: dict[int, int] = field(default_factory=dict)
    # Peak context ever held by each agent
    _agent_peak: dict[int, int] = field(default_factory=dict)

    def send(self, sender: int, receiver: int | None, payload_size: int,
             kind: str, round_idx: int) -> None:
        self.messages.append(
            Message(sender, receiver, payload_size, kind, round_idx)
        )

    def note_context(self, agent_id: int, tokens: int) -> None:
        """Record that this agent currently holds `tokens` in its context."""
        self._agent_context[agent_id] = tokens
        prior = self._agent_peak.get(agent_id, 0)
        if tokens > prior:
            self._agent_peak[agent_id] = tokens

    def total_tokens(self) -> int:
        return sum(m.payload_size for m in self.messages)

    def n_messages(self) -> int:
        return len(self.messages)

    def peak_per_agent_context(self) -> int:
        if not self._agent_peak:
            return 0
        return max(self._agent_peak.values())

    def mean_per_agent_context(self) -> float:
        if not self._agent_peak:
            return 0.0
        return sum(self._agent_peak.values()) / len(self._agent_peak)

    def summary(self) -> dict[str, float]:
        return {
            "total_tokens": self.total_tokens(),
            "n_messages": self.n_messages(),
            "peak_agent_context": self.peak_per_agent_context(),
            "mean_agent_context": self.mean_per_agent_context(),
        }
