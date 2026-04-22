"""Role hierarchy — orchestrator + workers.

Real multi-agent teams are not a flat broadcast. They have:
  - Orchestrators: operate at a coarse scale, few in number, take high-level
    decisions, read the manifold summary.
  - Workers: operate at fine scale, many in number, contribute observations
    and execute detailed actions.

The orchestrator reads the shared manifold every round and emits a "goal
vector" broadcast to workers. Workers use the goal vector to adjust their
own forgetting rate — higher when goal changes rapidly, lower when stable.

This is the minimal form of MERA-style tree hierarchy: two levels, one
contraction factor. Deeper hierarchies are a trivial extension (add another
tier of managers above orchestrators).
"""

from __future__ import annotations
import math
from dataclasses import dataclass


def split_roles(n_agents: int) -> tuple[int, int]:
    """Return (n_orchestrators, n_workers). One orchestrator per O(log N)."""
    n_orch = max(1, math.ceil(math.log2(max(n_agents, 2))))
    n_work = n_agents - n_orch
    return n_orch, n_work
