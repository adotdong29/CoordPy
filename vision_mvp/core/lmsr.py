"""Hanson's Logarithmic Market Scoring Rule for belief aggregation.

LMSR (Hanson 2003) is the canonical market maker for combinatorial prediction
markets. Given aggregate quantity q ∈ ℝ^K over K outcomes,

    C(q) = b · log Σ_k exp(q_k / b)
    p_k(q) = exp(q_k / b) / Σ_j exp(q_j / b)     (softmax)

Traders buying |Δ_k| units of outcome k pay C(q + Δ) − C(q), which is a proper
scoring rule: truthful reporting of beliefs is optimal. The liquidity parameter
b governs maximum loss: at most b · log K.

In CASR this replaces the price-setting in `core/market.py` for Idea 8
(market-cleared routing). Each agent posts its belief; LMSR aggregates to a
single price/probability vector with incentive compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LMSRReport:
    b: float
    q: np.ndarray
    prices: np.ndarray           # current probabilities
    cost_total: float            # C(q) − C(0)

    def summary(self) -> str:
        return (
            f"b={self.b:.2f}, cost={self.cost_total:.4f}, "
            f"prices={np.round(self.prices, 3).tolist()}"
        )


def cost_function(q: np.ndarray, b: float) -> float:
    """LMSR cost C(q) = b · log-sum-exp(q/b)."""
    q = np.asarray(q, dtype=float)
    if b <= 0:
        raise ValueError("b must be > 0")
    m = float(q.max() / b)       # numerical stabilisation
    return float(b * (m + np.log(np.exp(q / b - m).sum())))


def prices(q: np.ndarray, b: float) -> np.ndarray:
    """Softmax over q/b — the market's current probability vector."""
    q = np.asarray(q, dtype=float)
    if b <= 0:
        raise ValueError("b must be > 0")
    shifted = q / b - float((q / b).max())
    e = np.exp(shifted)
    return e / e.sum()


def trade_cost(q: np.ndarray, delta: np.ndarray, b: float) -> float:
    """Cost to buy `delta` units from current state `q`: C(q+Δ) − C(q)."""
    return cost_function(np.asarray(q, float) + np.asarray(delta, float), b) \
         - cost_function(q, b)


def max_loss(b: float, n_outcomes: int) -> float:
    """Worst-case market-maker loss: b · log K."""
    if b <= 0 or n_outcomes < 2:
        raise ValueError("bad inputs")
    return float(b * np.log(n_outcomes))


class LMSRMarket:
    """Stateful LMSR market maker. Accepts trades; exposes prices and cost."""

    def __init__(self, n_outcomes: int, b: float = 10.0):
        if n_outcomes < 2:
            raise ValueError("need at least 2 outcomes")
        if b <= 0:
            raise ValueError("b must be > 0")
        self.n = n_outcomes
        self.b = float(b)
        self.q = np.zeros(n_outcomes)

    def trade(self, delta: np.ndarray) -> float:
        """Apply a trade; return its net cost to the trader."""
        delta = np.asarray(delta, dtype=float).ravel()
        if delta.size != self.n:
            raise ValueError(f"delta must have length {self.n}")
        cost = trade_cost(self.q, delta, self.b)
        self.q = self.q + delta
        return cost

    def prices(self) -> np.ndarray:
        return prices(self.q, self.b)

    def report(self) -> LMSRReport:
        return LMSRReport(
            b=self.b,
            q=self.q.copy(),
            prices=self.prices(),
            cost_total=cost_function(self.q, self.b) - cost_function(np.zeros(self.n), self.b),
        )
