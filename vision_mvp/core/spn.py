"""Sum-Product Networks — tractable exact inference for discrete mixtures.

Poon & Domingos (2011). An SPN is a DAG of sum (weighted mixture) and
product (factorised joint) nodes whose leaves are univariate distributions.
Marginalization and MPE inference are single forward passes — polynomial in
the network size — which classical Bayes-nets don't generally admit.

We ship the minimal discrete SPN sufficient for tractable joint inference over
small agent-state distributions:

  LeafCat(idx, probs)       — categorical leaf over variable `idx`
  SumNode(children, weights) — weighted mixture
  ProdNode(children)         — factorised joint (no scope overlap enforced)

Inference: `log_likelihood(evidence)` returns log P(evidence) where evidence
maps var_idx → value (missing = marginalised).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


@dataclass
class LeafCat:
    var: int
    probs: np.ndarray             # (K,) categorical distribution

    def __post_init__(self):
        self.probs = np.asarray(self.probs, dtype=float)
        if not np.isclose(self.probs.sum(), 1.0):
            raise ValueError("probs must sum to 1")

    def log_value(self, evidence: dict[int, int]) -> float:
        if self.var in evidence:
            v = evidence[self.var]
            return float(np.log(self.probs[v] + 1e-40))
        return 0.0               # marginalised: Σ prob = 1

    def scope(self) -> set[int]:
        return {self.var}


@dataclass
class SumNode:
    children: list                # list of node
    weights: np.ndarray           # (K,) mixing weights

    def __post_init__(self):
        self.weights = np.asarray(self.weights, dtype=float)
        if not np.isclose(self.weights.sum(), 1.0):
            raise ValueError("weights must sum to 1")
        if len(self.children) != self.weights.size:
            raise ValueError("child count must match weights length")

    def log_value(self, evidence: dict[int, int]) -> float:
        log_ws = np.log(self.weights + 1e-40)
        log_cs = np.array([c.log_value(evidence) for c in self.children])
        return float(_logsumexp(log_ws + log_cs))

    def scope(self) -> set[int]:
        s: set[int] = set()
        for c in self.children:
            s |= c.scope()
        return s


@dataclass
class ProdNode:
    children: list

    def log_value(self, evidence: dict[int, int]) -> float:
        return float(sum(c.log_value(evidence) for c in self.children))

    def scope(self) -> set[int]:
        s: set[int] = set()
        for c in self.children:
            s |= c.scope()
        return s


def _logsumexp(x: np.ndarray) -> float:
    m = float(x.max())
    return m + float(np.log(np.exp(x - m).sum()))


@dataclass
class SumProductNetwork:
    """SPN with a single root node."""
    root: object                  # SumNode / ProdNode / LeafCat

    def log_likelihood(self, evidence: dict[int, int]) -> float:
        return self.root.log_value(evidence)

    def likelihood(self, evidence: dict[int, int]) -> float:
        return float(np.exp(self.log_likelihood(evidence)))
