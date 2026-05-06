"""Small stable agent/team surface for CoordPy.

The released runtime is still ``RunSpec -> RunReport``, but many users
want a simpler "create a few agents and run them" entrypoint closer to
what agent-toolkit SDKs expose. This module provides that surface
without dragging callers into the research ladder.

Design constraints:

* Python-first and small enough to learn in one screen.
* Backend-neutral: works with any ``LLMBackend`` duck-type, including
  local Ollama and OpenAI-compatible providers with API keys.
* Bounded-context by default: only the latest N visible handoffs are
  threaded into the next agent's prompt, so the happy path does not
  silently devolve into token cramming.
* Capsule-native audit trail: each agent output is sealed as a
  ``TEAM_HANDOFF`` capsule so the team run can still be inspected.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from .capsule import CapsuleBudget, CapsuleLedger, render_view
from .llm_backend import LLMBackend, backend_from_config, backend_from_env
from .team_coord import capsule_team_handoff


@dataclasses.dataclass(frozen=True)
class Agent:
    """One team member in the stable lightweight agent surface."""

    name: str
    instructions: str
    role: str | None = None
    backend: Any | None = None
    temperature: float = 0.2
    max_tokens: int = 256

    @property
    def effective_role(self) -> str:
        return self.role or self.name


@dataclasses.dataclass(frozen=True)
class AgentTurn:
    """One agent invocation inside an :class:`AgentTeam` run."""

    agent_name: str
    role: str
    prompt: str
    output: str
    capsule_cid: str | None = None


@dataclasses.dataclass(frozen=True)
class TeamResult:
    """Result of a lightweight team run."""

    task: str
    final_output: str
    turns: tuple[AgentTurn, ...]
    capsule_view: dict[str, Any] | None = None
    root_cid: str | None = None


def agent(
    name: str,
    instructions: str,
    *,
    role: str | None = None,
    backend: Any | None = None,
    temperature: float = 0.2,
    max_tokens: int = 256,
) -> Agent:
    """Convenience constructor mirroring lightweight agent SDK style."""

    return Agent(
        name=name,
        instructions=instructions,
        role=role,
        backend=backend,
        temperature=temperature,
        max_tokens=max_tokens,
    )


class AgentTeam:
    """Simple sequential agent-team runner with bounded visible context.

    Example::

        team = AgentTeam.from_env(
            [
                agent("planner", "Break the task into steps."),
                agent("researcher", "Gather the facts."),
                agent("writer", "Write the final answer."),
            ],
            model="gpt-4o-mini",
            backend_name="openai",
            team_instructions="Be concise and cite the handoffs you use.",
        )
        result = team.run("Explain what CoordPy does.")
        print(result.final_output)
    """

    def __init__(
        self,
        agents: Sequence[Agent],
        *,
        backend: Any | None = None,
        team_instructions: str = "",
        max_visible_handoffs: int = 4,
        capture_capsules: bool = True,
        handoff_budget: "CapsuleBudget | None" = None,
    ) -> None:
        if not agents:
            raise ValueError("AgentTeam requires at least one agent")
        if max_visible_handoffs <= 0:
            raise ValueError("max_visible_handoffs must be > 0")
        self.agents = tuple(agents)
        self.backend = backend
        self.team_instructions = team_instructions.strip()
        self.max_visible_handoffs = int(max_visible_handoffs)
        self.capture_capsules = bool(capture_capsules)
        # Override the default per-handoff capsule budget. ``None``
        # means use ``_default_budget_for(TEAM_HANDOFF)`` (4096
        # tokens / 64 KiB), which fits typical LLM agent turns.
        # Tighten for benchmarks; widen for very long turns.
        self.handoff_budget = handoff_budget

    @classmethod
    def from_env(
        cls,
        agents: Sequence[Agent],
        *,
        model: str | None = None,
        backend_name: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        team_instructions: str = "",
        max_visible_handoffs: int = 4,
        capture_capsules: bool = True,
        handoff_budget: "CapsuleBudget | None" = None,
    ) -> "AgentTeam":
        """Construct a team from common env-driven backend settings.

        Preferred env names:
        ``COORDPY_BACKEND``, ``COORDPY_API_BASE_URL``,
        ``COORDPY_API_KEY``. Legacy ``COORDPY_LLM_*`` and OpenAI-
        compatible fallbacks ``OPENAI_BASE_URL`` / ``OPENAI_API_KEY``
        are also supported.
        """

        backend = backend_from_env(
            backend_name,
            model=model,
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
        )
        return cls(
            agents,
            backend=backend,
            team_instructions=team_instructions,
            max_visible_handoffs=max_visible_handoffs,
            capture_capsules=capture_capsules,
            handoff_budget=handoff_budget,
        )

    @classmethod
    def from_config(
        cls,
        agents: Sequence[Agent],
        *,
        config: Any,
        model: str,
        team_instructions: str = "",
        max_visible_handoffs: int = 4,
        capture_capsules: bool = True,
        handoff_budget: "CapsuleBudget | None" = None,
    ) -> "AgentTeam":
        """Construct a team from a stable ``CoordPyConfig``."""

        backend = backend_from_config(config, model=model)
        if backend is None:
            raise ValueError(
                "config did not resolve to a backend; set llm_backend / "
                "llm_base_url / ollama_url or use AgentTeam.from_env(...)")
        return cls(
            agents,
            backend=backend,
            team_instructions=team_instructions,
            max_visible_handoffs=max_visible_handoffs,
            capture_capsules=capture_capsules,
            handoff_budget=handoff_budget,
        )

    def run(self, task: str) -> TeamResult:
        """Run the team once over ``task`` and return a stable result."""

        ledger = CapsuleLedger() if self.capture_capsules else None
        turns: list[AgentTurn] = []
        recent_handoffs: list[tuple[str, str]] = []
        parent_cid: str | None = None

        for idx, member in enumerate(self.agents):
            backend = self._resolve_backend(member)
            role = member.effective_role
            prompt = self._build_prompt(
                member=member,
                task=task,
                recent_handoffs=recent_handoffs,
            )
            output = backend.generate(
                prompt,
                max_tokens=member.max_tokens,
                temperature=member.temperature,
            )

            capsule_cid: str | None = None
            if ledger is not None:
                next_role = (
                    self.agents[idx + 1].effective_role
                    if idx + 1 < len(self.agents)
                    else "team_output"
                )
                handoff = capsule_team_handoff(
                    source_role=role,
                    to_role=next_role,
                    claim_kind="agent_output",
                    payload=output,
                    round=0,
                    parents=(parent_cid,) if parent_cid else (),
                    budget=self.handoff_budget,
                )
                sealed = ledger.admit_and_seal(handoff)
                capsule_cid = sealed.cid
                parent_cid = sealed.cid

            turns.append(
                AgentTurn(
                    agent_name=member.name,
                    role=role,
                    prompt=prompt,
                    output=output,
                    capsule_cid=capsule_cid,
                )
            )
            recent_handoffs.append((role, output))
            if len(recent_handoffs) > self.max_visible_handoffs:
                recent_handoffs = recent_handoffs[-self.max_visible_handoffs :]

        view = render_view(ledger).as_dict() if ledger is not None else None
        final_output = turns[-1].output
        root_cid = (
            view.get("root_cid") if view is not None else None
        ) or parent_cid
        return TeamResult(
            task=task,
            final_output=final_output,
            turns=tuple(turns),
            capsule_view=view,
            root_cid=root_cid,
        )

    def _resolve_backend(self, member: Agent) -> LLMBackend:
        backend = member.backend or self.backend
        if backend is None:
            raise ValueError(
                "no backend configured; pass backend=... to AgentTeam or "
                "construct the team with AgentTeam.from_env(...)"
            )
        if not isinstance(backend, LLMBackend):
            raise TypeError("backend must satisfy the LLMBackend protocol")
        return backend

    def _build_prompt(
        self,
        *,
        member: Agent,
        task: str,
        recent_handoffs: Sequence[tuple[str, str]],
    ) -> str:
        parts = []
        if self.team_instructions:
            parts.append(self.team_instructions)
        parts.append(f"Agent: {member.name}")
        parts.append(f"Role: {member.effective_role}")
        parts.append(member.instructions.strip())
        parts.append(f"Task: {task.strip()}")
        if recent_handoffs:
            rendered = "\n".join(
                f"- {role}: {text}"
                for role, text in recent_handoffs[-self.max_visible_handoffs :]
            )
            parts.append(
                "Visible team handoffs (bounded to avoid token cramming):\n"
                f"{rendered}"
            )
        parts.append("Reply with your contribution for the next team member.")
        return "\n\n".join(parts)


def create_team(
    agents: Sequence[Agent],
    *,
    backend: Any | None = None,
    model: str | None = None,
    backend_name: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: float | None = None,
    team_instructions: str = "",
    max_visible_handoffs: int = 4,
    capture_capsules: bool = True,
) -> AgentTeam:
    """Convenience constructor mirroring lightweight team SDK style.

    Pass an explicit ``backend=...`` when you already have a backend
    object, or pass ``model=...`` plus ``backend_name=...`` /
    ``base_url=...`` / ``api_key=...`` to let CoordPy build a backend
    for you.
    """

    resolved_backend = backend
    if resolved_backend is None and any((
        model,
        backend_name,
        base_url,
        api_key,
    )):
        resolved_backend = backend_from_env(
            backend_name,
            model=model,
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
        )

    return AgentTeam(
        agents,
        backend=resolved_backend,
        team_instructions=team_instructions,
        max_visible_handoffs=max_visible_handoffs,
        capture_capsules=capture_capsules,
    )


__all__ = [
    "Agent",
    "AgentTurn",
    "TeamResult",
    "AgentTeam",
    "agent",
    "create_team",
]
