"""Index coding — selective multicast with side-information.

The index-coding problem (Birk & Kol 1998): a server has messages x_1..x_n
and N receivers, each wanting a specific x_i and already knowing a subset
of the others. What's the minimum number of broadcast transmissions to
satisfy all receivers?

A well-known upper bound is the *chromatic number* of the side-information
confusion graph: nodes are (receiver_index, desired_message_index) pairs; an
edge means two demands conflict if combined into one broadcast. A proper
coloring is a feasible transmission schedule using χ codeword groupings.

For our use: agents have partial logs and request missing entries; compute
the confusion graph, color it greedily, and batch transmissions — saving
broadcast bandwidth proportional to 1 - χ/n.

Pure-numpy, no networkx. Greedy coloring is O(N + E) but not optimal; for
N ≤ a few thousand it's fine for diagnostic use.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class IndexCodeReport:
    n_messages: int
    chromatic_bound: int         # greedy-coloring size; ≥ χ
    colors: np.ndarray            # color per demand
    savings: float                # 1 - chromatic_bound / n_messages

    def summary(self) -> str:
        return (
            f"broadcast {self.chromatic_bound}/{self.n_messages} slots  "
            f"(savings={self.savings*100:.1f}%)"
        )


def confusion_graph(
    side_info: list[set[int]],
    desires: list[int],
) -> np.ndarray:
    """Build the confusion graph.

    side_info[i]  = set of message indices receiver i already knows.
    desires[i]    = the message receiver i wants.

    Edge between receivers i and j iff combining x_desires[i] and x_desires[j]
    into one broadcast leaks information: i.e., desires[j] ∉ side_info[i] OR
    desires[i] ∉ side_info[j].
    """
    N = len(side_info)
    if len(desires) != N:
        raise ValueError("side_info and desires must have same length")
    adj = np.zeros((N, N), dtype=np.int8)
    for i in range(N):
        for j in range(i + 1, N):
            # Conflict iff neither knows the other's want
            if desires[j] not in side_info[i] or desires[i] not in side_info[j]:
                adj[i, j] = adj[j, i] = 1
    return adj


def greedy_color(adj: np.ndarray) -> np.ndarray:
    """Simple greedy vertex coloring, ordering by decreasing degree."""
    A = np.asarray(adj, dtype=np.int8)
    N = A.shape[0]
    degrees = A.sum(axis=1)
    order = np.argsort(-degrees)
    colors = -np.ones(N, dtype=int)
    for v in order:
        forbidden = set(int(colors[u]) for u in range(N) if A[v, u] and colors[u] >= 0)
        c = 0
        while c in forbidden:
            c += 1
        colors[v] = c
    return colors


def index_code_report(
    side_info: list[set[int]],
    desires: list[int],
    n_messages: int,
) -> IndexCodeReport:
    adj = confusion_graph(side_info, desires)
    colors = greedy_color(adj)
    chrom = int(colors.max() + 1) if colors.size else 0
    return IndexCodeReport(
        n_messages=n_messages,
        chromatic_bound=chrom,
        colors=colors,
        savings=1.0 - (chrom / max(n_messages, 1)),
    )
