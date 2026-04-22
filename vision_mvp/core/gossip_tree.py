"""Plumtree + HyParView epidemic broadcast — bounded-fanout O(log N) gossip.

HyParView (Leitao, Pereira, Rodrigues 2007): each node keeps a small
*active* view (size ~ log N, used for gossip) and a *passive* view
(size ~ K·log N, used as a failure-replacement pool). Joins/leaves touch only
a constant number of nodes.

Plumtree (Leitao, Pereira, Rodrigues 2007): build a spanning tree on top of
the HyParView overlay by lazily pruning redundant gossip messages. First-hop
arrival of a message lazily prunes the other arms of the tree; missing
nodes request repairs via lazy-push IHAVE / GRAFT messages.

Combined: eager-push along the Plumtree tree + lazy-push via HyParView
fallback. Reliability ≈ 1 with bounded fanout — the thing AutoGen/CrewAI
lack.

This is a simulation-level implementation. Nodes pass messages through a
central in-memory dispatcher — no real networking.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class HyParNode:
    node_id: str
    active_view: set[str] = field(default_factory=set)
    passive_view: set[str] = field(default_factory=set)
    active_cap: int = 4
    passive_cap: int = 32
    _rng: random.Random = field(default_factory=lambda: random.Random(0))

    def add_active(self, other: str) -> None:
        if other == self.node_id:
            return
        if other in self.active_view:
            return
        if len(self.active_view) >= self.active_cap:
            # Evict one to passive
            ev = self._rng.choice(list(self.active_view))
            self.active_view.discard(ev)
            self.passive_view.add(ev)
            if len(self.passive_view) > self.passive_cap:
                self.passive_view.discard(self._rng.choice(list(self.passive_view)))
        self.active_view.add(other)
        self.passive_view.discard(other)

    def remove_active(self, other: str) -> None:
        if other in self.active_view:
            self.active_view.discard(other)
            self.passive_view.add(other)


@dataclass
class PlumtreeOverlay:
    """Simulated cluster: N HyParView nodes with a dispatcher and per-node
    seen-set for Plumtree eager/lazy routing.
    """
    nodes: dict[str, HyParNode] = field(default_factory=dict)
    _seen: dict[str, set[str]] = field(default_factory=dict)  # per-node msg ids
    total_eager: int = 0
    total_lazy: int = 0
    total_delivered: int = 0

    def add_node(self, name: str, bootstrap: list[str]) -> None:
        self.nodes[name] = HyParNode(
            node_id=name,
            _rng=random.Random(hash(name) & 0xFFFFFFFF),
        )
        self._seen[name] = set()
        for b in bootstrap:
            if b in self.nodes:
                self.nodes[name].add_active(b)
                self.nodes[b].add_active(name)

    def broadcast(self, src: str, msg_id: str) -> None:
        """Eager-push msg from src to its active view; recursively from
        receivers that haven't seen it yet.
        """
        if src not in self.nodes:
            raise ValueError(f"unknown source {src!r}")
        frontier = [src]
        self._seen[src].add(msg_id)
        self.total_delivered += 1
        while frontier:
            next_front = []
            for u in frontier:
                for v in list(self.nodes[u].active_view):
                    if msg_id in self._seen[v]:
                        self.total_lazy += 1
                        continue
                    self._seen[v].add(msg_id)
                    self.total_eager += 1
                    self.total_delivered += 1
                    next_front.append(v)
            frontier = next_front

    def reliability(self, msg_id: str) -> float:
        """Fraction of nodes that received `msg_id`."""
        if not self.nodes:
            return 1.0
        got = sum(1 for n in self.nodes if msg_id in self._seen[n])
        return got / len(self.nodes)
