"""HierarchicalRouter — CASR with explicit sub-team decomposition.

Flat CASR (Phases 1-6) pools all N agents into one team. Real workflows are
nested: a 5000-agent team is best organized as 5 × 1000 worker teams plus
a small orchestrator team that synthesizes.

Usage:
    from vision_mvp import HierarchicalRouter, CASRRouter

    router = HierarchicalRouter(
        worker_teams=[CASRRouter(n_agents=200, state_dim=d, task_rank=8)
                      for _ in range(5)],
        orchestrator=CASRRouter(n_agents=10, state_dim=d, task_rank=3),
    )

    # Step with per-worker-team observations
    worker_observations = [np.random.randn(200, d) for _ in range(5)]
    orchestrator_summaries = router.step(worker_observations)

For LLM agents, see `LLMHierarchy` below which does the same thing for
natural-language tasks with decomposition + synthesis via LLM calls.

See HIERARCHICAL_DECOMPOSITION.md for the design rationale and theorems
13–15 bounding complexity.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Callable

from .bus import Bus


@dataclass
class HierarchicalRouter:
    """Two-level CASR: several worker CASRRouters + one orchestrator.

    The API is deliberately simple: `step(worker_observations)` runs one
    round on every worker team in parallel, computes a per-team summary
    (the mean of that team's estimates), forwards those summaries to the
    orchestrator as ITS observations, and runs one orchestrator round.

    Resulting per-agent context:
      - worker agent: O(log(worker_size)) tokens (as in flat CASR)
      - orchestrator agent: O(log(n_workers)) tokens
      - Total per-round cost: sum of all rounds — still sub-linear in total N.
    """
    worker_teams: list           # list of CASRRouter instances
    orchestrator: object         # CASRRouter (typically small)
    _aggregator_bus: Bus = field(default_factory=Bus)
    _round_idx: int = 0

    def __post_init__(self):
        if not self.worker_teams:
            raise ValueError("need at least one worker team")
        # All worker teams + orchestrator must agree on state_dim
        d = self.worker_teams[0].state_dim
        for t in self.worker_teams:
            if t.state_dim != d:
                raise ValueError(
                    f"worker team state_dim={t.state_dim} != {d}")
        if self.orchestrator.state_dim != d:
            raise ValueError("orchestrator state_dim mismatch")
        if self.orchestrator.n_agents != len(self.worker_teams):
            raise ValueError(
                f"orchestrator n_agents={self.orchestrator.n_agents} must "
                f"equal #worker_teams={len(self.worker_teams)}")

    # ---- Primary API ----

    def step(self, worker_observations: list) -> np.ndarray:
        """One hierarchical round.

        Args:
            worker_observations: list with one (n_i, d) array per worker team.

        Returns:
            (n_workers, d) array of orchestrator consensus estimates —
            one row per worker team, representing that team's digested view.
        """
        if len(worker_observations) != len(self.worker_teams):
            raise ValueError("worker_observations length != #worker_teams")

        team_summaries = []
        for team, obs in zip(self.worker_teams, worker_observations):
            team_est = team.step(obs)                       # (n_i, d)
            # Worker → orchestrator: one "team summary" vector (the mean
            # estimate of the team). Simulated token cost: d per team.
            summary = team_est.mean(axis=0)                  # (d,)
            self._aggregator_bus.send(-1, -1, int(team.state_dim),
                                      "team_summary", self._round_idx)
            team_summaries.append(summary)

        summaries_matrix = np.stack(team_summaries)          # (n_workers, d)
        orch_est = self.orchestrator.step(summaries_matrix)

        # Orchestrator → workers: broadcast its consensus (currently
        # advisory — workers still use their own observations next round).
        for team in self.worker_teams:
            # Each worker team "reads" the orchestrator consensus (simulated)
            self._aggregator_bus.send(-1, -1, int(team.state_dim),
                                      "orch_broadcast", self._round_idx)
        self._round_idx += 1
        return orch_est

    # ---- Stats ----

    @property
    def stats(self) -> dict:
        worker_stats = [t.stats for t in self.worker_teams]
        orch_stats = self.orchestrator.stats
        total_tokens = (sum(ws["total_tokens"] for ws in worker_stats)
                        + orch_stats["total_tokens"]
                        + self._aggregator_bus.total_tokens())
        # Peak per-agent context — max across all levels
        peak = max([ws["peak_context_per_agent"] for ws in worker_stats]
                   + [orch_stats["peak_context_per_agent"]])
        return {
            "peak_context_per_agent": int(peak),
            "total_tokens": int(total_tokens),
            "n_worker_teams": len(self.worker_teams),
            "worker_workspace": [ws["workspace_size"] for ws in worker_stats],
            "orchestrator_workspace": orch_stats["workspace_size"],
            "rounds_executed": self._round_idx,
            "inter_level_tokens": self._aggregator_bus.total_tokens(),
        }
