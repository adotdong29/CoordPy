"""SparseRouter — top-k message routing with load balancing.

Thin wrapper around AgentKeyIndex that adds:
  - Capacity limits (no agent gets overwhelmed in a single round)
  - Global broadcast agents (BigBird-style — "coordinators" always receive)
  - Routing decision logging for observability
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field

from .agent_keys import AgentKeyIndex


@dataclass
class SparseRouter:
    keys: AgentKeyIndex
    top_k: int = 5
    capacity_per_round: int = 20        # max messages any single agent receives per round
    global_agents: list[int] = field(default_factory=list)  # always receive (like BigBird global tokens)
    _round_inbox_count: dict = field(default_factory=dict)

    def begin_round(self) -> None:
        self._round_inbox_count = {i: 0 for i in range(self.keys.n_agents)}

    def route(self, query: np.ndarray, sender_id: int | None = None,
              recipient_hints: list[int] | None = None) -> list[int]:
        """Return the list of agent ids to deliver a message to.

        - `sender_id`: excluded from delivery (no self-message).
        - `recipient_hints`: if provided, these are added to the delivery set
          (on top of whatever the router selects). Useful for "reply to X".
        """
        exclude = {sender_id} if sender_id is not None else set()
        selected = self.keys.route(query, top_k=self.top_k, exclude=exclude)
        # Always include global coordinators
        for g in self.global_agents:
            if g not in selected and g not in exclude:
                selected.append(g)
        # Honor explicit hints
        if recipient_hints:
            for h in recipient_hints:
                if h != sender_id and h not in selected:
                    selected.append(h)
        # Apply capacity limits — skip agents already full this round
        capped = []
        for a in selected:
            if self._round_inbox_count.get(a, 0) < self.capacity_per_round:
                capped.append(a)
                self._round_inbox_count[a] = self._round_inbox_count.get(a, 0) + 1
        return capped

    def stats(self) -> dict:
        loads = list(self._round_inbox_count.values())
        if not loads:
            return {"mean_load": 0, "max_load": 0}
        return {
            "mean_load": round(sum(loads) / len(loads), 2),
            "max_load": max(loads),
            "n_saturated": sum(1 for v in loads if v >= self.capacity_per_round),
        }
