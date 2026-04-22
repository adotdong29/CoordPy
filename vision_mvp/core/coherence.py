"""MESI-style cache-coherence wrapper around a shared event store.

For a shared in-memory store accessed by many agents, the MESI (Modified,
Exclusive, Shared, Invalid) protocol ensures readers and writers never see
stale or conflicting data, at the cost of invalidation broadcasts. Classical
four-state invalidation:

  Invalid (I)    — slot has no valid data; must fetch.
  Shared (S)     — multiple agents hold a clean read-only copy.
  Exclusive (E)  — one agent holds a clean copy; others know it's exclusive.
  Modified (M)   — one agent has a dirty copy; store must be updated on eviction.

Transitions on operations:
  read when I           → fetch; if anyone had E/M, they go to S;    self → S
  read when S/E/M       → no transition
  write when I/S        → invalidate all other copies;               self → M
  write when E          → self → M (no broadcast needed)
  write when M          → no transition

Used here as the consistency substrate for CASR's stigmergic layer and for
any per-agent cache of the shared register. No claim of real concurrency —
single-process simulator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class CacheState(Enum):
    INVALID = "I"
    SHARED = "S"
    EXCLUSIVE = "E"
    MODIFIED = "M"


@dataclass
class CoherentCache:
    """Per-agent cache of a single shared key's value.

    The coordinator (`CoherenceDirectory`) tracks all caches and routes
    reads/writes.
    """
    agent_id: str
    state: CacheState = CacheState.INVALID
    value: object = None
    version: int = 0

    def is_readable(self) -> bool:
        return self.state != CacheState.INVALID


@dataclass
class CoherenceDirectory:
    """Tracks the MESI state of every cache for one shared key."""

    caches: dict[str, CoherentCache] = field(default_factory=dict)
    version: int = 0
    n_invalidations: int = 0
    n_fetches: int = 0

    def register(self, agent_id: str) -> None:
        if agent_id not in self.caches:
            self.caches[agent_id] = CoherentCache(agent_id=agent_id)

    def read(self, agent_id: str, source_value) -> object:
        """Service a read by `agent_id`. `source_value` is a callable returning
        the authoritative current value (used only on fetch misses).
        """
        self.register(agent_id)
        c = self.caches[agent_id]
        if c.is_readable():
            return c.value

        # Miss — fetch from any other cache in E or M, else from source
        for other in self.caches.values():
            if other.agent_id == agent_id:
                continue
            if other.state in (CacheState.EXCLUSIVE, CacheState.MODIFIED):
                # Downgrade them to shared
                other.state = CacheState.SHARED
                c.value = other.value
                c.state = CacheState.SHARED
                c.version = other.version
                self.n_fetches += 1
                return c.value
        # Fetch from source
        c.value = source_value() if callable(source_value) else source_value
        c.version = self.version
        # If no one else has it, we get EXCLUSIVE; else SHARED
        if all(o.state == CacheState.INVALID
               for o in self.caches.values() if o.agent_id != agent_id):
            c.state = CacheState.EXCLUSIVE
        else:
            c.state = CacheState.SHARED
        self.n_fetches += 1
        return c.value

    def write(self, agent_id: str, new_value: object) -> None:
        """Service a write by `agent_id` — invalidate others, set to M."""
        self.register(agent_id)
        c = self.caches[agent_id]
        # Invalidate other caches
        for other in self.caches.values():
            if other.agent_id == agent_id:
                continue
            if other.state != CacheState.INVALID:
                other.state = CacheState.INVALID
                other.value = None
                self.n_invalidations += 1
        c.value = new_value
        c.state = CacheState.MODIFIED
        self.version += 1
        c.version = self.version

    def summary(self) -> dict:
        return {
            "invalidations": self.n_invalidations,
            "fetches": self.n_fetches,
            "version": self.version,
            "states": {a: c.state.value for a, c in self.caches.items()},
        }
