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
from typing import Any, Sequence, TypeVar

from .capsule import CapsuleBudget, CapsuleLedger, render_view
from .llm_backend import LLMBackend, backend_from_config, backend_from_env
from .team_coord import capsule_team_handoff


# Self-type for classmethods so subclasses are inferred as their
# own type (PEP 673 in spirit; written as a TypeVar bound for
# Python 3.10 compat). ``MyTeam.from_env(...)`` correctly
# infers as ``MyTeam`` instead of ``AgentTeam``.
_TeamT = TypeVar("_TeamT", bound="AgentTeam")


@dataclasses.dataclass(frozen=True)
class Agent:
    """One team member in the stable lightweight agent surface.

    ``instructions`` is the agent's system-prompt text. The
    ``system_prompt`` property is provided as an alias for
    callers familiar with the OpenAI / LangChain / AutoGen
    naming convention.
    """

    name: str
    instructions: str
    role: str | None = None
    backend: Any | None = None
    temperature: float = 0.2
    max_tokens: int = 256

    @property
    def system_prompt(self) -> str:
        """Alias for ``instructions``. ``system_prompt`` is the
        more common name in other agent SDKs (OpenAI Assistants,
        LangChain, AutoGen); coordpy uses ``instructions`` as
        the canonical name and exposes both for interop.
        """
        return self.instructions

    @property
    def effective_role(self) -> str:
        return self.role or self.name


@dataclasses.dataclass(frozen=True)
class AgentTurn:
    """One agent invocation inside an :class:`AgentTeam` run.

    ``output`` is the agent's reply text. The ``response``
    property is provided as an alias for callers familiar with
    OpenAI / LangChain naming.
    """

    agent_name: str
    role: str
    prompt: str
    output: str
    capsule_cid: str | None = None

    @property
    def response(self) -> str:
        """Alias for ``output`` — the agent's reply text. Other
        agent SDKs call this ``response`` / ``message`` /
        ``content``; coordpy uses ``output`` as the canonical
        name and exposes ``response`` for interop.
        """
        return self.output


@dataclasses.dataclass(frozen=True)
class TeamResult:
    """Result of a lightweight team run.

    ``final_output`` is the **last agent's reply, verbatim** — it
    is not an aggregation across all agents. ``last_output`` is
    an alias kept for clarity; new code should prefer it. To
    inspect every turn, walk ``turns``.

    ``capsule_view`` is the JSON-shape ``dict`` produced by
    ``coordpy.render_view(...).as_dict()``. It is intentionally
    a dict (not a ``CapsuleView`` dataclass) so it round-trips
    cleanly through JSON for storage / RPC / CI artefact use —
    pass it directly to ``coordpy.verify_chain_from_view_dict``
    or ``coordpy.audit_capsule_lifecycle_from_view``.

    ``root_cid`` matches ``capsule_view["root_cid"]`` by
    construction.
    """

    task: str
    final_output: str
    turns: tuple[AgentTurn, ...]
    capsule_view: dict[str, Any] | None = None
    root_cid: str | None = None

    @property
    def last_output(self) -> str:
        """Alias for ``final_output`` — explicitly the last agent's
        reply, not an aggregation.
        """
        return self.final_output


def agent(
    name: str,
    instructions: str | None = None,
    *,
    system_prompt: str | None = None,
    role: str | None = None,
    backend: Any | None = None,
    temperature: float = 0.2,
    max_tokens: int = 256,
) -> Agent:
    """Convenience constructor mirroring lightweight agent SDK style.

    The agent's system-prompt text can be passed as the second
    positional argument (``instructions``), or as the keyword
    ``system_prompt=`` for callers familiar with other agent
    SDKs. The two are aliases; passing both raises ``ValueError``.
    """

    if instructions is not None and system_prompt is not None:
        raise ValueError(
            "agent(...) accepts either ``instructions`` OR "
            "``system_prompt``, not both — they are aliases."
        )
    if instructions is None:
        instructions = system_prompt
    if instructions is None:
        raise ValueError(
            "agent(name, instructions=...) requires a non-empty "
            "string (or pass ``system_prompt=``)"
        )

    if not isinstance(name, str) or not name.strip():
        raise ValueError("agent(name=...) requires a non-empty string")
    if any(c in name for c in "\n\r\t"):
        raise ValueError(
            "agent(name=...) must not contain newlines or tabs "
            "(would mangle log lines and CLI tables)"
        )
    if len(name) > 256:
        raise ValueError(
            f"agent(name=...) is {len(name)} chars; please keep it "
            f"under 256 chars (typical agent name is one or two "
            f"words)"
        )
    if not isinstance(instructions, str) or not instructions.strip():
        raise ValueError(
            "agent(..., instructions=...) requires a non-empty string"
        )
    if not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError(
            f"agent(..., max_tokens=...) must be a positive int; "
            f"got {max_tokens!r}"
        )
    if max_tokens > 1_000_000:
        # No real LLM has a million-token output budget. Catch
        # obvious bugs (forgotten exponent, wrong unit) without
        # constraining any real use.
        raise ValueError(
            f"agent(..., max_tokens=...) is {max_tokens:,}, which "
            f"exceeds the 1,000,000-token sanity cap; this is "
            f"almost certainly a bug — pass an explicit smaller "
            f"value if you really mean it"
        )
    if not isinstance(temperature, (int, float)) or not (
        0.0 <= temperature <= 2.0
    ):
        raise ValueError(
            f"agent(..., temperature=...) must be a number in "
            f"[0.0, 2.0]; got {temperature!r}"
        )
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
        # The ledger from the most recent ``run()`` call, exposed
        # so callers can audit / retire / re-render the chain
        # without reaching into private state. ``None`` until the
        # team has been run at least once. Reset at the start of
        # every ``run()`` so successive calls don't alias.
        self.last_ledger: "CapsuleLedger | None" = None
        # Surface duplicate agent names as a warning. They are
        # valid (e.g. multiple "reviewer" agents with different
        # instructions), but accidental dupes are a footgun for
        # owner-attribution audits — turns are then
        # distinguishable only by capsule CID.
        names = [a.name for a in self.agents]
        seen, dupes = set(), []
        for n in names:
            if n in seen and n not in dupes:
                dupes.append(n)
            seen.add(n)
        if dupes:
            import warnings
            warnings.warn(
                f"AgentTeam has duplicate agent name(s): "
                f"{sorted(dupes)!r}. This is allowed but the audit "
                f"trail will need capsule CIDs to distinguish them; "
                f"prefer unique names if the trail will be reviewed.",
                stacklevel=2,
            )

    @classmethod
    def from_env(
        cls: type[_TeamT],
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
    ) -> _TeamT:
        """Construct a team from common env-driven backend settings.

        Preferred env names:
        ``COORDPY_BACKEND``, ``COORDPY_API_BASE_URL``,
        ``COORDPY_API_KEY``. Legacy ``COORDPY_LLM_*`` and OpenAI-
        compatible fallbacks ``OPENAI_BASE_URL`` / ``OPENAI_API_KEY``
        are also supported.

        Subclassing
        -----------
        ``MyTeam.from_env(...)`` returns a ``MyTeam`` instance,
        not the bare ``AgentTeam``. The return annotation is a
        ``_TeamT`` TypeVar bound to ``AgentTeam``; under PEP 563
        (``from __future__ import annotations`` in this module)
        ``inspect.signature(...).return_annotation`` reads as
        the literal string ``'_TeamT'`` — use
        ``typing.get_type_hints`` or
        ``inspect.signature(..., eval_str=True)`` (Python 3.10+)
        to resolve it to the live TypeVar object.
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
        cls: type[_TeamT],
        agents: Sequence[Agent],
        *,
        config: Any,
        model: str,
        team_instructions: str = "",
        max_visible_handoffs: int = 4,
        capture_capsules: bool = True,
        handoff_budget: "CapsuleBudget | None" = None,
    ) -> _TeamT:
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
        """Run the team once over ``task`` and return a stable result.

        Each agent's reply is sealed as a TEAM_HANDOFF capsule.
        The default per-handoff budget is 64 KiB / 4096 tokens;
        if your task or your agent replies are larger than that,
        construct the team with
        ``handoff_budget=coordpy.CapsuleBudget(max_bytes=...,
        max_tokens=...)`` to widen it.

        Raises ``ValueError`` on empty / whitespace ``task``.
        Raises ``TypeError`` if the backend's ``generate(...)``
        returns something other than ``str``.
        Raises ``CapsuleAdmissionError`` if a single handoff
        exceeds the configured budget.
        """

        if not isinstance(task, str) or not task.strip():
            raise ValueError(
                "AgentTeam.run(task=...) requires a non-empty string"
            )
        if "\x00" in task:
            raise ValueError(
                "AgentTeam.run(task=...) must not contain NUL bytes; "
                "they are valid in Python strings but a footgun for "
                "downstream JSON / log / shell consumers"
            )

        ledger = CapsuleLedger() if self.capture_capsules else None
        # Make the ledger available for post-run inspection
        # (audit / retire / re-render) without forcing the caller
        # to reach into private state.
        self.last_ledger = ledger
        turns: list[AgentTurn] = []
        recent_handoffs: list[tuple[str, str]] = []
        parent_cid: str | None = None

        # Seal the task itself as the first capsule in the chain
        # so root_cid commits to (task, agent_outputs...). Without
        # this, two distinct tasks whose final agent reply
        # happened to match would produce the same root_cid — a
        # real audit-correctness issue. Capsule kind is
        # TEAM_HANDOFF with source_role="user" / claim_kind="task".
        if ledger is not None and self.agents:
            first_role = self.agents[0].effective_role
            task_capsule = capsule_team_handoff(
                source_role="user",
                to_role=first_role,
                claim_kind="task",
                payload=task,
                round=0,
                parents=(),
                budget=self.handoff_budget,
                agent_name="user",
            )
            sealed_task = ledger.admit_and_seal(task_capsule)
            parent_cid = sealed_task.cid

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
            # Type-guard the backend's return so a misbehaving
            # backend produces a clean TypeError naming the
            # offender — not a deeper AttributeError /
            # UnicodeEncodeError from the capsule sealing path.
            if not isinstance(output, str):
                raise TypeError(
                    f"backend.generate(...) for agent "
                    f"{member.name!r} returned "
                    f"{type(output).__name__}, expected str. "
                    f"Backends must return a string; if your "
                    f"backend produces bytes, decode them first."
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
                    agent_name=member.name,
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

        # Pass the chain head as ``root_cid`` so the view and the
        # returned ``TeamResult.root_cid`` agree byte-for-byte.
        # AgentTeam runs seal a TEAM_HANDOFF chain rather than a
        # full RUN_REPORT, so without this the view has
        # ``root_cid=None`` while the TeamResult has the chain head.
        #
        # ``include_payload=True`` is on by default for AgentTeam
        # because the chain is bounded (one capsule per agent +
        # the leading task capsule), the payloads are small (the
        # task string + each agent's reply), and the result is
        # tamper-detectable end-to-end:
        # ``verify_chain_from_view_dict`` re-derives each CID from
        # the payload bytes when payloads are present.
        view = (
            render_view(
                ledger, root_cid=parent_cid, include_payload=True,
            ).as_dict()
            if ledger is not None
            else None
        )
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
        # Put the agent identity at the top of the prompt so a
        # synthetic backend's ``prompt.splitlines()[0]`` routing
        # (a common test pattern) finds it regardless of whether
        # team_instructions is set.
        parts.append(f"Agent: {member.name}")
        parts.append(f"Role: {member.effective_role}")
        if self.team_instructions:
            parts.append(self.team_instructions)
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
    handoff_budget: "CapsuleBudget | None" = None,
) -> AgentTeam:
    """Convenience constructor mirroring lightweight team SDK style.

    Pass an explicit ``backend=...`` when you already have a backend
    object, or pass ``model=...`` plus ``backend_name=...`` /
    ``base_url=...`` / ``api_key=...`` to let CoordPy build a backend
    for you.

    ``handoff_budget`` overrides the per-handoff capsule budget; see
    :class:`AgentTeam` for the default and the override semantics.
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
        handoff_budget=handoff_budget,
    )


__all__ = [
    "Agent",
    "AgentTurn",
    "TeamResult",
    "AgentTeam",
    "agent",
    "create_team",
]
