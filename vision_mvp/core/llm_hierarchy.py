"""LLMHierarchy — multi-level LLM-agent coordination.

Composes several `LLMTeam` sub-teams under one orchestrator LLM. Each sub-team
focuses on a specialty (e.g. security / performance / correctness) and
produces a short report; the orchestrator LLM fuses the sub-reports into a
single final answer.

This is the language-model analogue of HierarchicalRouter. For a
five-specialty review team, each with 200 reviewers, you get:
    total agents = 5 × 200 + 1 orchestrator = 1 001
    per-agent LLM calls per round = ~10 per sub-team (workspace) = ~50 total
    plus 1 orchestrator synthesis call per round

This is what lets you scale LLM review teams past the point where a single
flat team becomes slow to init / converge.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field

from .llm_client import LLMClient
from .llm_team import LLMTeam


@dataclass
class LLMHierarchy:
    """Group of specialist LLMTeams + an orchestrator LLM call that fuses
    their outputs."""
    sub_teams: list[LLMTeam]
    sub_team_names: list[str]           # one name per team, for prompt framing
    model: str = "qwen2.5-coder:7b"
    orchestrator_client: LLMClient = field(default=None)  # type: ignore
    _round_idx: int = 0
    _last_subteam_reports: list[str] = field(default_factory=list)
    _orch_total_tokens: int = 0
    _wall_llm: float = 0.0

    def __post_init__(self):
        if len(self.sub_teams) != len(self.sub_team_names):
            raise ValueError("sub_teams and sub_team_names length mismatch")
        if self.orchestrator_client is None:
            self.orchestrator_client = LLMClient(model=self.model)

    def initialize(self, progress_cb=None):
        for name, team in zip(self.sub_team_names, self.sub_teams):
            if progress_cb:
                progress_cb(f"[init] {name} sub-team (N={team.n_agents})")
            team.initialize(progress_cb=progress_cb)

    def step(self, progress_cb=None) -> dict:
        """One hierarchical round: each sub-team does one CASR round,
        its consensus is collected, orchestrator fuses."""
        self._round_idx += 1
        reports: list[str] = []
        sub_infos = []
        for name, team in zip(self.sub_team_names, self.sub_teams):
            if progress_cb:
                progress_cb(f"[round {self._round_idx}] {name} stepping")
            info = team.step(progress_cb=progress_cb)
            reports.append(info["consensus_text"])
            sub_infos.append({"name": name, **info})

        # Orchestrator: fuse the sub-team consensus reports into ONE
        # over-arching consensus. One LLM call per round per orchestrator.
        if progress_cb:
            progress_cb(f"[round {self._round_idx}] orchestrator synthesizing")
        bullet = "\n".join(f"- {n}: {r}" for n, r in zip(self.sub_team_names, reports))
        prompt = (
            "You are the lead reviewer coordinating multiple specialist teams. "
            "Each team has converged on one concern. Synthesize them into a "
            "single overarching verdict, focusing on the *most important* "
            "issue across teams.\n\n"
            "Team reports:\n"
            f"{bullet}\n\n"
            "One-sentence overall verdict:"
        )
        t0 = time.time()
        orch_answer = self.orchestrator_client.generate(
            prompt, max_tokens=120, temperature=0.2)
        self._wall_llm += time.time() - t0
        self._orch_total_tokens = self.orchestrator_client.stats.total_tokens()

        self._last_subteam_reports = reports
        return {
            "round": self._round_idx,
            "sub_team_reports": list(zip(self.sub_team_names, reports)),
            "orchestrator_verdict": orch_answer,
            "sub_infos": sub_infos,
        }

    def stats(self) -> dict:
        sub_stats = [t.stats() for t in self.sub_teams]
        sub_n = sum(t.n_agents for t in self.sub_teams)
        sub_llm = sum(s["llm_generate_calls"] for s in sub_stats)
        sub_tokens = sum(s["llm_total_tokens"] for s in sub_stats)
        return {
            "n_sub_teams": len(self.sub_teams),
            "total_agents": sub_n,
            "sub_team_llm_generate_calls": sub_llm,
            "orchestrator_llm_generate_calls":
                self.orchestrator_client.stats.n_generate_calls,
            "total_llm_generate_calls": sub_llm
                + self.orchestrator_client.stats.n_generate_calls,
            "total_llm_tokens": sub_tokens + self._orch_total_tokens,
            "rounds": self._round_idx,
            "max_workspace": max(s["workspace_size"] for s in sub_stats),
            "wall_orchestrator_s": round(self._wall_llm, 2),
        }
