"""Lazy-greedy submodular maximization with the (1 − 1/e) guarantee.

A set function v : 2^[N] → ℝ is submodular if for every A ⊆ B and x ∉ B,

    v(A ∪ {x}) − v(A) ≥ v(B ∪ {x}) − v(B)       (diminishing returns)

Nemhauser, Wolsey & Fisher (1978): for monotone submodular v and a cardinality
constraint |S| ≤ k, greedy selection of the highest-marginal-gain element
achieves v(S_greedy) ≥ (1 − 1/e) v(S_opt) ≈ 0.632 v(S_opt).

Minoux's lazy-greedy (1978) accelerates greedy from O(N k · T_eval) to typically
O((N + k log N) · T_eval) by maintaining a priority queue of upper bounds on
each candidate's marginal gain and reevaluating only when the top-of-queue
item's stale bound is reached.

Application in CASR: footprint selection, workspace admission, coreset
construction — anywhere we greedily pick a small subset to maximize a
monotone submodular coverage function.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class SubmodularResult:
    selected: list[int]
    value: float
    n_oracle_calls: int

    def summary(self) -> str:
        return (
            f"selected {len(self.selected)} elements, "
            f"value={self.value:.4f}, "
            f"oracle calls={self.n_oracle_calls}"
        )


def greedy(
    v: Callable[[frozenset], float],
    universe: list[int],
    k: int,
) -> SubmodularResult:
    """Naive greedy — O(N k) evaluations. Correct baseline."""
    selected: list[int] = []
    S: set[int] = set()
    current_v = v(frozenset(S))
    n_calls = 1
    for _ in range(k):
        best = None
        best_gain = -float("inf")
        for x in universe:
            if x in S:
                continue
            gain = v(frozenset(S | {x})) - current_v
            n_calls += 1
            if gain > best_gain:
                best_gain = gain
                best = x
        if best is None:
            break
        selected.append(best)
        S.add(best)
        current_v += best_gain
    return SubmodularResult(
        selected=selected, value=current_v, n_oracle_calls=n_calls,
    )


def lazy_greedy(
    v: Callable[[frozenset], float],
    universe: list[int],
    k: int,
) -> SubmodularResult:
    """Minoux lazy greedy using submodularity to skip stale candidates."""
    S: set[int] = set()
    current_v = v(frozenset(S))
    n_calls = 1

    # Initial upper bounds: marginal gain of each element alone.
    # heapq is a min-heap; store negative gains for max-heap behavior.
    heap: list[tuple[float, int, int]] = []   # (−gain, tiebreak, element)
    for i, x in enumerate(universe):
        gain = v(frozenset({x})) - current_v
        n_calls += 1
        heapq.heappush(heap, (-gain, i, x))

    selected: list[int] = []
    iteration = 0   # for tiebreaking / staleness
    while heap and len(selected) < k:
        neg_gain, _, x = heapq.heappop(heap)
        if x in S:
            continue
        # Recompute the actual marginal gain; if it's still at top of heap
        # under the new value, we commit.
        new_gain = v(frozenset(S | {x})) - current_v
        n_calls += 1
        if not heap or new_gain >= -heap[0][0]:
            selected.append(x)
            S.add(x)
            current_v += new_gain
        else:
            iteration += 1
            heapq.heappush(heap, (-new_gain, iteration + len(universe), x))

    return SubmodularResult(
        selected=selected, value=current_v, n_oracle_calls=n_calls,
    )


def approximation_ratio(greedy_value: float, optimal_value: float) -> float:
    """Empirical ratio greedy / optimal. Should be ≥ 1 − 1/e ≈ 0.632."""
    if optimal_value <= 0:
        return 1.0
    return greedy_value / optimal_value
