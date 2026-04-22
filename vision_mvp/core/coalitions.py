"""Coalition structure generation — stable team-within-team partitions.

Rahwan & Jennings (2008); Aadithya et al. (2011). Given a characteristic
function v: 2^N → ℝ, find the partition of N into disjoint coalitions that
maximises Σ_C v(C). Exact is exponential (Bell number growth).

Heuristic: *merge-and-split* local search (Apt & Witzel 2009). Start with
singletons. Repeat: pick two coalitions whose merge increases value, merge
them; else pick any coalition whose split yields value-increase, split it.
Converges to a Pareto-stable partition (no profitable merge or split).

Application: finding which sub-teams of the CASR team should form "sub-bases"
with tighter coordination, vs stay independent. Different question from general
equilibrium — asks about *structure*, not price.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Callable


@dataclass
class CoalitionReport:
    coalitions: list[frozenset[int]]
    value: float
    n_iters: int

    def summary(self) -> str:
        sizes = sorted((len(c) for c in self.coalitions), reverse=True)
        return (
            f"{len(self.coalitions)} coalitions "
            f"sizes={sizes}, total value {self.value:.3f}"
        )


def _total_value(parts: list[frozenset[int]], v: Callable[[frozenset[int]], float]) -> float:
    return sum(v(c) for c in parts)


def merge_and_split(
    agents: list[int],
    v: Callable[[frozenset[int]], float],
    max_iter: int = 100,
) -> CoalitionReport:
    """Iterative merge-split search starting from singletons."""
    parts: list[frozenset[int]] = [frozenset({a}) for a in agents]

    for it in range(max_iter):
        improved = False

        # MERGE: try each pair
        for i, j in combinations(range(len(parts)), 2):
            before = v(parts[i]) + v(parts[j])
            merged = parts[i] | parts[j]
            after = v(merged)
            if after > before + 1e-9:
                parts = (
                    [parts[k] for k in range(len(parts)) if k not in (i, j)]
                    + [merged]
                )
                improved = True
                break

        if improved:
            continue

        # SPLIT: try each partition of each coalition into two non-empty parts
        for idx, coal in enumerate(parts):
            if len(coal) < 2:
                continue
            members = sorted(coal)
            n = len(members)
            split_found = False
            # Enumerate 2^(n-1) - 1 unique bipartitions
            for mask in range(1, 1 << (n - 1)):
                left = frozenset(members[k] for k in range(n) if (mask >> k) & 1)
                right = coal - left
                if not right:
                    continue
                before = v(coal)
                after = v(left) + v(right)
                if after > before + 1e-9:
                    parts = (
                        [parts[k] for k in range(len(parts)) if k != idx]
                        + [left, right]
                    )
                    improved = True
                    split_found = True
                    break
            if split_found:
                break

        if not improved:
            return CoalitionReport(
                coalitions=list(parts),
                value=_total_value(parts, v),
                n_iters=it + 1,
            )

    return CoalitionReport(
        coalitions=list(parts),
        value=_total_value(parts, v),
        n_iters=max_iter,
    )
