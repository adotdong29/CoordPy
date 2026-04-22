"""MultiRoleTeam — agents with DIFFERENT roles doing DIFFERENT work.

This is the next step past LLMTeam. In LLMTeam, every agent is the same
kind of worker operating on similar material. A real team has a workflow:

  Researchers           produce     facts / signals from raw inputs
         │
         ▼
  Market analysts       produce     local asset views
         │
         ▼
  Strategy builders     combine     signals + views → candidate trades
         │
         ▼
  Portfolio manager     synthesize  candidate trades → final portfolio

Each role:
  - Has its own prompt template
  - Reads different inputs
  - Emits outputs in a role-specific format
  - Feeds the next role in the pipeline

This uses the same theoretical scaffolding as the rest of the library —
streaming PCA, workspace admission, per-agent predictors — but wires them
into a role-based hierarchical pipeline instead of a flat consensus team.

See `HierarchicalRouter` for the numpy-only version of the same idea.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Callable

from .llm_client import LLMClient


@dataclass
class Role:
    """One role in the pipeline.

    - `name`: short string label (research / analyst / strategy / pm).
    - `prompt_template`: fn(inputs, role_persona) -> prompt string.
    - `persona`: identity to inject into the prompt.
    - `n_agents`: how many agents at this role (1 for PM, more for workers).
    - `reads_from`: role names whose outputs this role consumes, or
      'raw' to consume task inputs directly.
    """
    name: str
    persona: str
    prompt_template: Callable[[list[str], str], str]
    n_agents: int
    reads_from: str = "raw"
    max_tokens: int = 100


@dataclass
class MultiRoleTeam:
    """Orchestrates a pipeline of role-based teams.

    Each role is a layer. Layer r is invoked after layer r-1 has produced
    its outputs. Within a layer, each agent makes one LLM call.

    Theoretical positioning:
      - Each layer is a flat team of role-identical agents.
      - Inter-layer communication = summary reports (shape-matches the
        `HierarchicalRouter` pattern).
      - Total LLM calls = sum of layer sizes. No rounds of consensus
        voting — the pipeline is a DAG, not a cycle.

    Complexity: per-agent context is O(input_size_for_role). No agent
    sees the full raw data; each sees only what its role needs.
    """
    roles: list[Role]
    raw_inputs_by_role: dict[str, list[str]]  # role_name -> list of inputs (one per agent)
    client: LLMClient = field(default=None)   # type: ignore
    model: str = "qwen2.5-coder:7b"
    _outputs_by_role: dict[str, list[str]] = field(default_factory=dict)
    _wall_by_role: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if self.client is None:
            self.client = LLMClient(model=self.model)
        # Validate role graph
        role_names = {r.name for r in self.roles}
        for r in self.roles:
            if r.reads_from != "raw" and r.reads_from not in role_names:
                raise ValueError(f"Role {r.name} reads from unknown role {r.reads_from}")
        # Validate inputs
        for r in self.roles:
            if r.reads_from == "raw" and r.name not in self.raw_inputs_by_role:
                raise ValueError(f"No raw_inputs for role {r.name}")
            # Raw-reading roles need input per agent
            if r.reads_from == "raw":
                if len(self.raw_inputs_by_role[r.name]) != r.n_agents:
                    raise ValueError(
                        f"Role {r.name}: raw_inputs length "
                        f"{len(self.raw_inputs_by_role[r.name])} != n_agents {r.n_agents}"
                    )

    def run(self, progress_cb=None) -> dict:
        """Execute the pipeline. Returns dict of role_name → list of outputs."""
        for role in self.roles:
            if progress_cb:
                progress_cb(f"\n[{role.name}] {role.n_agents} agents starting")

            if role.reads_from == "raw":
                inputs_for_agents = self.raw_inputs_by_role[role.name]
            else:
                # Downstream role reads ALL outputs of upstream role as
                # its shared context. Each agent in this role gets the
                # full set of upstream outputs.
                upstream = self._outputs_by_role[role.reads_from]
                inputs_for_agents = [
                    "\n".join(f"- {u}" for u in upstream)
                    for _ in range(role.n_agents)
                ]

            t0 = time.time()
            outputs = []
            for i, inp in enumerate(inputs_for_agents):
                if progress_cb:
                    progress_cb(f"  {role.name} agent {i+1}/{role.n_agents}")
                prompt = role.prompt_template([inp], role.persona)
                out = self.client.generate(prompt, max_tokens=role.max_tokens,
                                           temperature=0.2)
                outputs.append(out)
            wall = time.time() - t0

            self._outputs_by_role[role.name] = outputs
            self._wall_by_role[role.name] = wall
            if progress_cb:
                progress_cb(f"  [{role.name}] done in {wall:.1f}s "
                            f"({role.n_agents} calls)")

        return self._outputs_by_role

    @property
    def stats(self) -> dict:
        total_wall = sum(self._wall_by_role.values())
        return {
            "total_llm_generate_calls": self.client.stats.n_generate_calls,
            "total_llm_tokens": self.client.stats.total_tokens(),
            "total_wall_s": round(total_wall, 1),
            "per_role_wall_s": {k: round(v, 1)
                                 for k, v in self._wall_by_role.items()},
            "per_role_output_count": {k: len(v)
                                       for k, v in self._outputs_by_role.items()},
        }
