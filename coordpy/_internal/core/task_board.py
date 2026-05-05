"""TaskBoard — shared DAG of subtasks with claim/complete/deps semantics.

Stigmergic coordination layer: agents coordinate via shared state rather
than direct messages. Inspired by ant-colony-optimization pheromones and
classical blackboard architectures.

Each subtask has:
  - id
  - title
  - description
  - tag_embedding (topic vector, for routing to relevant agents)
  - deps: list of subtask ids that must complete first
  - status: "pending" / "ready" / "claimed" / "done"
  - assignee: agent_id or None
  - output: text result when done

An agent can:
  - LIST ready subtasks (no deps, no assignee)
  - CLAIM a ready subtask
  - POST output for a claimed subtask (marks it done)
  - READ the output of any done subtask it depends on
"""

from __future__ import annotations
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Literal


Status = Literal["pending", "ready", "claimed", "done", "failed"]


@dataclass
class Subtask:
    id: str
    title: str
    description: str
    tag_embedding: np.ndarray | None = None
    deps: list[str] = field(default_factory=list)
    status: Status = "pending"
    assignee: int | None = None
    output: str = ""
    claimed_at: float = 0.0
    completed_at: float = 0.0

    def is_root(self) -> bool:
        return len(self.deps) == 0


@dataclass
class TaskBoard:
    subtasks: dict[str, Subtask] = field(default_factory=dict)
    _log: list[dict] = field(default_factory=list)

    def add(self, t: Subtask) -> None:
        self.subtasks[t.id] = t
        self._refresh_status()

    def _refresh_status(self) -> None:
        for t in self.subtasks.values():
            if t.status in ("done", "failed", "claimed"):
                continue
            # Is it ready (all deps done)?
            if all(self.subtasks.get(d) and self.subtasks[d].status == "done"
                   for d in t.deps):
                t.status = "ready"

    def ready_tasks(self) -> list[Subtask]:
        self._refresh_status()
        return [t for t in self.subtasks.values()
                if t.status == "ready" and t.assignee is None]

    def claim(self, subtask_id: str, agent_id: int) -> bool:
        t = self.subtasks.get(subtask_id)
        if t is None or t.status != "ready" or t.assignee is not None:
            return False
        t.assignee = agent_id
        t.status = "claimed"
        t.claimed_at = time.time()
        self._log.append({"event": "claim", "task": subtask_id, "agent": agent_id,
                          "t": t.claimed_at})
        return True

    def complete(self, subtask_id: str, agent_id: int, output: str) -> bool:
        t = self.subtasks.get(subtask_id)
        if t is None or t.assignee != agent_id or t.status != "claimed":
            return False
        t.output = output
        t.status = "done"
        t.completed_at = time.time()
        self._log.append({"event": "complete", "task": subtask_id, "agent": agent_id,
                          "t": t.completed_at})
        self._refresh_status()
        return True

    def deps_outputs(self, subtask_id: str) -> list[tuple[str, str]]:
        """Return list of (dep_id, dep_output) for a subtask's dependencies."""
        t = self.subtasks.get(subtask_id)
        if t is None:
            return []
        return [(d, self.subtasks[d].output) for d in t.deps
                if d in self.subtasks and self.subtasks[d].status == "done"]

    def done_count(self) -> int:
        return sum(1 for t in self.subtasks.values() if t.status == "done")

    def total_count(self) -> int:
        return len(self.subtasks)

    def all_done(self) -> bool:
        return all(t.status in ("done", "failed") for t in self.subtasks.values())

    def summary(self) -> dict:
        counts = {"pending": 0, "ready": 0, "claimed": 0, "done": 0, "failed": 0}
        for t in self.subtasks.values():
            counts[t.status] = counts.get(t.status, 0) + 1
        return {"counts": counts, "total": len(self.subtasks),
                "done_count": self.done_count()}
