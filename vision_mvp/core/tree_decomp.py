"""Treewidth estimation via the min-degree heuristic.

Exact treewidth is NP-hard, but the min-degree (a.k.a. min-fill) heuristic
of Bodlaender gives a usable upper bound in O(n²) per elimination:

  repeatedly pick the vertex of smallest degree, eliminate it (remove it
  and add edges among its neighbors), record the degree as a bag size.

The resulting elimination ordering has bag sizes equal to the tree decomposition's
bag sizes in the worst case — the max is an upper bound on treewidth.

Used as a *diagnostic* in CASR: compute the treewidth of the current causal-
footprint graph; when it exceeds O(log N), warn that coordination will be
expensive regardless of routing strategy.

Pure-numpy / stdlib implementation — no networkx required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class TreewidthReport:
    upper_bound: int               # max bag size - 1
    elimination_order: list[int]   # vertices in elimination order
    bag_sizes: list[int]           # bag size at elimination of each vertex

    def summary(self) -> str:
        return (
            f"treewidth ≤ {self.upper_bound}  "
            f"(max bag size {max(self.bag_sizes) if self.bag_sizes else 0})"
        )


def _adj_to_sets(adj: np.ndarray) -> list[set[int]]:
    n = adj.shape[0]
    out = [set() for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and adj[i, j]:
                out[i].add(j)
    return out


def min_degree_treewidth(adj: np.ndarray) -> TreewidthReport:
    """Min-degree elimination on symmetric `adj` (nonzero = edge)."""
    adj = np.asarray(adj)
    if adj.shape[0] != adj.shape[1]:
        raise ValueError("adj must be square")
    nbrs = _adj_to_sets(adj)
    n = len(nbrs)
    alive = set(range(n))
    order = []
    bags = []

    while alive:
        # Pick alive vertex of smallest live-degree
        v = min(alive, key=lambda x: len(nbrs[x] & alive))
        live_nbrs = nbrs[v] & alive
        order.append(v)
        bags.append(len(live_nbrs) + 1)  # v + its neighbors
        # Add edges among v's live neighbors (make them a clique)
        lst = list(live_nbrs)
        for i, a in enumerate(lst):
            for b in lst[i + 1:]:
                nbrs[a].add(b)
                nbrs[b].add(a)
        alive.remove(v)
        # Remove v from neighbors' sets lazily (we gate on alive anyway)

    return TreewidthReport(
        upper_bound=(max(bags) - 1) if bags else 0,
        elimination_order=order,
        bag_sizes=bags,
    )


def is_low_treewidth(adj: np.ndarray, target: int) -> bool:
    """Convenience: upper-bound-treewidth ≤ target?"""
    return min_degree_treewidth(adj).upper_bound <= target
