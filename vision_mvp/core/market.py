"""Market-cleared routing — VISION_MILLIONS Idea 8.

Each round, agents submit *bids* for admission to the shared workspace. The
top-k bidders win. Under a VCG payment rule (charge the (k+1)-th-highest
bid), truthful bidding is a dominant strategy (Theorem 8 in PROOFS.md).

A bid is the agent's expected task-performance gain from writing this round.
In practice we use the surprise score (prediction error magnitude) as a
proxy for expected gain — an agent that is very surprised has the most to
gain from broadcasting its state.

This replaces the greedy top-k workspace of Phase 3. Compared to greedy:
  - Greedy: picks agents with highest salience, no pricing.
  - Market: picks top-k *and* computes VCG prices so agents have incentive
    to report their salience truthfully rather than inflate it.

The market is O(N) per round (single sort + pricing lookup), same as
greedy — no additional overhead for the incentive-compatibility property.
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass


@dataclass
class MarketWorkspace:
    """Top-k workspace admission with VCG pricing."""
    n_agents: int
    # How many slots are "auctioned" each round
    k: int = 0

    def __post_init__(self):
        if self.k <= 0:
            self.k = max(1, math.ceil(math.log2(max(self.n_agents, 2))))

    def clear_market(self, bids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run the auction.

        Returns
        -------
        admitted: (k,) int array of winning agents.
        prices:   (k,) float array of VCG prices paid by each winner.
                  All winners pay the same price = the (k+1)-th-highest bid
                  (second-price auction clearing).
        """
        if bids.ndim != 1 or len(bids) != self.n_agents:
            raise ValueError("bids must be 1D of length n_agents")
        k = min(self.k, self.n_agents)
        if k >= self.n_agents:
            admitted = np.arange(self.n_agents)
            prices = np.zeros(self.n_agents)
            return admitted, prices

        sorted_idx = np.argsort(-bids)                 # descending
        admitted = sorted_idx[:k].copy()
        # VCG price: externality = the bid that would have won if winner
        # hadn't bid. With uniform unit-supply auction, this equals the
        # (k+1)-th-highest bid.
        clearing_price = float(bids[sorted_idx[k]])
        prices = np.full(k, clearing_price)
        return admitted, prices

    def capacity(self) -> int:
        return self.k


def salience_to_bid(salience: float, agent_budget: float = 1.0) -> float:
    """Translate an agent's salience into a budget-bounded bid.

    We linearly clip: bid = min(salience, budget). More sophisticated
    bidders could use a learned mapping, but truthful reporting under
    the VCG mechanism has bid = salience (= marginal value).
    """
    return min(float(salience), float(agent_budget))
