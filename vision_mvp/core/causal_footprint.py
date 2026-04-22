"""CausalFootprint — per-agent set of agent IDs whose outputs can causally
affect this agent's action, backed by a Bloom filter.

Rationale: for N=12 agents a plain set would suffice, but the CASR framework
claims O(1) membership via a Bloom filter parameterized by (capacity, error).
We implement it that way so cost/accuracy scales with N, matching FRAMEWORK.md.

A footprint is derived from a call graph: agent A's footprint includes agent B
whenever A calls B (A needs B's signature) OR B calls A (A must anticipate how
B will invoke it). The agent itself is always in its own footprint.
"""

from __future__ import annotations

import hashlib
import math
import random
from typing import Iterable


class CausalFootprint:
    """Bloom-filter-backed approximate set over agent IDs.

    Parameters are chosen via the standard optimal formulas:
      m = ceil(-n * ln(p) / (ln 2)^2)
      k = max(1, round((m / n) * ln 2))

    where n is `capacity` (expected number of items) and p is `error_rate`.

    Membership tests are O(k). False-positive rate approaches `error_rate` as
    the filter fills to `capacity`; we stay well under that for N=12.
    """

    def __init__(self, capacity: int = 100, error_rate: float = 0.01):
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        if not (0.0 < error_rate < 1.0):
            raise ValueError("error_rate must be in (0, 1)")
        self.capacity = int(capacity)
        self.error_rate = float(error_rate)
        ln2 = math.log(2.0)
        m = int(math.ceil(-capacity * math.log(error_rate) / (ln2 * ln2)))
        k = max(1, int(round((m / capacity) * ln2)))
        self.m = m
        self.k = k
        self._bits = bytearray((m + 7) // 8)
        self._count = 0
        # Track exact membership of added items so copy() + iteration semantics
        # are reproducible. This is O(n) extra memory but does NOT affect
        # __contains__ (which still uses the Bloom filter for O(k) queries).
        self._added: set[str] = set()

    # --- internals -------------------------------------------------------
    def _hashes(self, item: str) -> Iterable[int]:
        item_b = item.encode("utf-8")
        for i in range(self.k):
            h = hashlib.md5(str(i).encode("ascii") + b":" + item_b).digest()
            # Take first 8 bytes as unsigned int
            v = int.from_bytes(h[:8], "big", signed=False)
            yield v % self.m

    def _get_bit(self, idx: int) -> bool:
        return bool(self._bits[idx >> 3] & (1 << (idx & 7)))

    def _set_bit(self, idx: int) -> None:
        self._bits[idx >> 3] |= (1 << (idx & 7))

    # --- public API ------------------------------------------------------
    def add(self, agent_id: str) -> None:
        if agent_id in self._added:
            return
        for idx in self._hashes(agent_id):
            self._set_bit(idx)
        self._added.add(agent_id)
        self._count += 1

    def __contains__(self, agent_id: str) -> bool:
        # Pure Bloom-filter membership test (O(k)).
        for idx in self._hashes(agent_id):
            if not self._get_bit(idx):
                return False
        return True

    def __len__(self) -> int:
        return self._count

    def __iter__(self):
        return iter(self._added)

    def copy(self) -> "CausalFootprint":
        new = CausalFootprint(capacity=self.capacity, error_rate=self.error_rate)
        new._bits = bytearray(self._bits)
        new._count = self._count
        new._added = set(self._added)
        return new

    def members(self) -> set[str]:
        """Return the exact set of items explicitly added (not affected by
        false positives). Useful for the ablation leg and for testing."""
        return set(self._added)


def footprint_from_call_graph(
    call_graph: dict[str, list[str]],
    agent_id: str,
    capacity: int = 100,
    error_rate: float = 0.01,
) -> CausalFootprint:
    """Build the causal footprint for `agent_id`.

    Includes:
      - `agent_id` itself (self-loop)
      - forward edges: any B such that agent_id -> B in call_graph
      - reverse edges: any X such that X -> agent_id in call_graph
    """
    fp = CausalFootprint(capacity=capacity, error_rate=error_rate)
    fp.add(agent_id)
    # Forward: agent_id -> B
    for callee in call_graph.get(agent_id, []):
        fp.add(callee)
    # Reverse: X -> agent_id
    for caller, callees in call_graph.items():
        if agent_id in callees:
            fp.add(caller)
    return fp


def random_footprint(
    all_agents: list[str],
    size: int,
    seed: int,
    self_agent: str | None = None,
    capacity: int = 100,
    error_rate: float = 0.01,
) -> CausalFootprint:
    """Build a footprint with `size` items chosen uniformly at random from
    `all_agents` (deterministic given `seed`).

    For the ablation leg. If `self_agent` is provided, it is always included
    (matching how the causal footprint always includes the agent itself); we
    draw `size - 1` additional members from the other agents.
    """
    rng = random.Random(seed)
    fp = CausalFootprint(capacity=capacity, error_rate=error_rate)
    pool = list(all_agents)
    if self_agent is not None:
        fp.add(self_agent)
        pool = [a for a in pool if a != self_agent]
        remaining = max(0, size - 1)
    else:
        remaining = size
    remaining = min(remaining, len(pool))
    # rng.sample is deterministic given seed.
    chosen = rng.sample(pool, remaining) if remaining > 0 else []
    for a in chosen:
        fp.add(a)
    return fp
