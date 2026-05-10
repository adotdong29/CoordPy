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
  The handoff carries the producing prompt's SHA-256 + byte length +
  model tag, so an external auditor can prove the call was reproducible
  without sealing a separate PROMPT capsule.
* Telemetry-first: each turn carries per-call ``prompt_tokens``,
  ``output_tokens`` and ``wall_ms`` so users can see where tokens go
  and confirm the bounded-context savings claim from real numbers.
* Replay-friendly: ``TeamResult.dump`` writes a ``team_result.json``
  bundle (schema ``coordpy.team_result.v1``) with full per-turn
  prompts and the generation params actually used. ``replay_team_result``
  re-runs the team against any backend at the original sampling
  settings and returns a fresh, sealed ``TeamResult``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Sequence

from .capsule import CapsuleBudget, CapsuleLedger, render_view
from .llm_backend import LLMBackend, backend_from_config, backend_from_env
from .team_coord import capsule_team_handoff


TEAM_RESULT_SCHEMA = "coordpy.team_result.v1"


def _sha256_str(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------
# Public dataclasses
# ----------------------------------------------------------------------


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
    """One agent invocation inside an :class:`AgentTeam` run.

    Telemetry (real tokens, from the backend):

    * ``prompt_tokens`` / ``output_tokens`` — deltas computed from
      the backend's ``usage_snapshot()``.
    * ``wall_ms`` — wall-clock cost of the ``generate`` call.

    Bounded-context evidence (word counts, computed locally — these
    are what make the bounded-context savings visible turn-by-turn):

    * ``prompt_words`` — word count of the prompt CoordPy actually
      sent (post-bounding, post-summary).
    * ``naive_prompt_words`` — word count of the prompt a naive,
      token-cramming caller would have sent: full task each turn
      plus *every* upstream handoff. The delta is the value the
      bounded-context runtime actually delivered.
    * ``visible_handoffs`` — how many upstream handoffs were
      actually threaded forward (capped by ``max_visible_handoffs``).

    Replay fidelity: ``temperature`` / ``max_tokens`` /
    ``backend_base_url`` are the generation params actually used for
    this call, persisted on ``team_result.json`` and consumed by
    :func:`replay_team_result` so a sealed manifest replays at the
    original sampling settings, not the loader's defaults. Without
    these, replay against a new backend would silently substitute
    ``temperature=0.2`` / ``max_tokens=256`` for any preset that used
    different values (e.g. ``quant_desk_team`` runs at
    ``temperature=0.0``).

    Audit: ``prompt_sha256`` is the SHA-256 of the prompt the agent
    was actually asked, and is also persisted on the
    ``TEAM_HANDOFF`` capsule so an auditor can prove "this output
    came from this prompt" without re-deriving the prompt locally.
    """

    agent_name: str
    role: str
    prompt: str
    output: str
    capsule_cid: str | None = None
    prompt_tokens: int = 0
    output_tokens: int = 0
    wall_ms: float = 0.0
    visible_handoffs: int = 0
    prompt_sha256: str = ""
    model_tag: str = ""
    prompt_words: int = 0
    naive_prompt_words: int = 0
    temperature: float = 0.2
    max_tokens: int = 256
    backend_base_url: str | None = None


@dataclasses.dataclass(frozen=True)
class ActionDecision:
    """Structured form of the synthesizer's ``ACTION:`` line.

    ``action`` is the ALL-CAPS verdict (e.g. ``"EXECUTE"``,
    ``"EXECUTE-WITH-MODS"``, ``"NO-ACTION"``). ``justification`` is
    the rest of the line, stripped. ``raw`` is the verbatim ACTION
    line for debugging. Returned by :meth:`TeamResult.parse_action`
    when the synthesizer's output contains a parseable line; ``None``
    otherwise.
    """

    action: str
    justification: str
    raw: str


@dataclasses.dataclass(frozen=True)
class TeamResult:
    """Result of a lightweight team run.

    Carries the final output, the per-turn trail, the sealed capsule
    view (``coordpy.capsule_view.v1``), the chain root, and aggregate
    telemetry: total tokens in/out, total wall time, total LLM calls.
    Use :meth:`dump` to write a re-verifiable bundle to disk.
    """

    task: str
    final_output: str
    turns: tuple[AgentTurn, ...]
    capsule_view: dict[str, Any] | None = None
    root_cid: str | None = None
    total_prompt_tokens: int = 0
    total_output_tokens: int = 0
    total_wall_ms: float = 0.0
    total_calls: int = 0
    backend_model: str = ""
    backend_base_url: str | None = None
    team_instructions: str = ""
    task_summary: str | None = None
    max_visible_handoffs: int = 0
    stopped_early: bool = False

    @property
    def total_tokens(self) -> int:
        return int(self.total_prompt_tokens + self.total_output_tokens)

    # ------------------------------------------------------------
    # Bounded-context savings vs naive cramming
    # ------------------------------------------------------------

    def cramming_estimate(self) -> dict[str, float]:
        """Estimate how much the bounded-context runtime saved.

        Compares ``naive_prompt_words`` (full task each turn + every
        upstream handoff) to ``prompt_words`` (the bounded prompt
        actually sent), then scales to real-token units using the
        ratio of the backend's reported ``prompt_tokens`` to the
        bounded word count summed across the run.

        Returns a dict with ``bounded_words``, ``naive_words``,
        ``saved_words``, ``savings_pct`` (0..100), ``words_to_tokens``
        (the empirical ratio used for scaling), and
        ``estimated_tokens_saved`` (real tokens, not words). When
        ``words_to_tokens`` cannot be estimated (no real-token
        telemetry from the backend) the token estimate falls back to
        the word saving.

        The estimate is a *lower bound*: it assumes the same model
        produces the same response under both prompt shapes, which
        is a generous assumption for naive cramming (longer prompts
        often produce longer responses too).
        """
        bounded_words = sum(t.prompt_words for t in self.turns)
        naive_words = sum(t.naive_prompt_words for t in self.turns)
        saved_words = max(0, naive_words - bounded_words)
        savings_pct = (
            100.0 * saved_words / naive_words if naive_words > 0 else 0.0
        )
        ratio = (
            self.total_prompt_tokens / bounded_words
            if bounded_words > 0 and self.total_prompt_tokens > 0
            else 0.0
        )
        if ratio > 0:
            est_tokens_saved = int(round(saved_words * ratio))
        else:
            est_tokens_saved = int(saved_words)
        return {
            "bounded_words": int(bounded_words),
            "naive_words": int(naive_words),
            "saved_words": int(saved_words),
            "savings_pct": float(savings_pct),
            "words_to_tokens": float(ratio),
            "estimated_tokens_saved": int(est_tokens_saved),
        }

    # ------------------------------------------------------------
    # Markdown rendering
    # ------------------------------------------------------------

    def render_markdown(self, *, title: str | None = None) -> str:
        """Render this run as a polished Markdown report.

        Sections: header, final output, per-turn telemetry table,
        bounded-context savings block, audit block (capsule
        chain_head + root_cid). Suitable for committing to a repo
        or emailing as a desk artifact.
        """
        return _md_render(self, title=title)

    def dump(self, out_dir: str | os.PathLike) -> dict[str, str]:
        """Write a re-verifiable bundle to ``out_dir``.

        Files written:

        * ``team_capsule_view.json`` — the sealed capsule view in
          ``coordpy.capsule_view.v1`` shape, ready for
          ``coordpy-capsule verify-view --view ...``.
        * ``final_output.txt`` — the final agent output as plain text.
        * ``team_result.json`` — schema ``coordpy.team_result.v1``,
          a structured manifest with task, totals, per-turn prompts
          and outputs, model tags, generation params, and the capsule
          root CID. Used by :func:`replay_team_result` to re-run the
          same conversation on a different backend.
        * ``team_report.md`` — polished Markdown report of the run.

        Returns the absolute paths of the written files.
        """
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        paths: dict[str, str] = {}
        note_path = out_path / "final_output.txt"
        note_path.write_text(self.final_output, encoding="utf-8")
        paths["final_output"] = str(note_path)

        if self.capsule_view is not None:
            view_path = out_path / "team_capsule_view.json"
            view_path.write_text(
                json.dumps(self.capsule_view, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            paths["capsule_view"] = str(view_path)

        manifest = {
            "schema": TEAM_RESULT_SCHEMA,
            "task": self.task,
            "task_summary": self.task_summary,
            "team_instructions": self.team_instructions,
            "max_visible_handoffs": int(self.max_visible_handoffs),
            "final_output": self.final_output,
            "root_cid": self.root_cid,
            "stopped_early": bool(self.stopped_early),
            "backend": {
                "model": self.backend_model,
                "base_url": self.backend_base_url,
            },
            "totals": {
                "prompt_tokens": int(self.total_prompt_tokens),
                "output_tokens": int(self.total_output_tokens),
                "total_tokens": int(self.total_tokens),
                "wall_ms": float(self.total_wall_ms),
                "calls": int(self.total_calls),
            },
            "turns": [
                {
                    "agent_name": t.agent_name,
                    "role": t.role,
                    "capsule_cid": t.capsule_cid,
                    "prompt": t.prompt,
                    "prompt_sha256": t.prompt_sha256,
                    "model_tag": t.model_tag,
                    "backend_base_url": t.backend_base_url,
                    "output": t.output,
                    "prompt_tokens": int(t.prompt_tokens),
                    "output_tokens": int(t.output_tokens),
                    "wall_ms": float(t.wall_ms),
                    "visible_handoffs": int(t.visible_handoffs),
                    "output_chars": len(t.output),
                    "prompt_words": int(t.prompt_words),
                    "naive_prompt_words": int(t.naive_prompt_words),
                    "temperature": float(t.temperature),
                    "max_tokens": int(t.max_tokens),
                }
                for t in self.turns
            ],
            "cramming_estimate": self.cramming_estimate(),
        }
        manifest_path = out_path / "team_result.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        paths["team_result"] = str(manifest_path)

        report_md = self.render_markdown()
        report_path = out_path / "team_report.md"
        report_path.write_text(report_md, encoding="utf-8")
        paths["report_md"] = str(report_path)
        return paths

    # ------------------------------------------------------------
    # Convenience parsers
    # ------------------------------------------------------------

    # Allow common decorations around the ACTION label, including
    # markdown bold and any combination of leading list markers:
    #
    #   ACTION: EXECUTE - ...
    #   **ACTION:** EXECUTE-WITH-MODS — ...
    #   ## ACTION
    #   - ACTION: NO-ACTION; ...
    _ACTION_LINE_RE = re.compile(
        r"^[\s\-#>*_]*ACTION[\s*:\-_]*"
        r"(?P<action>[A-Z][A-Z0-9_\-]+)"
        r"(?P<rest>.*)$",
        re.MULTILINE,
    )

    def parse_action(
        self,
        text: str | None = None,
    ) -> ActionDecision | None:
        """Extract the synthesizer's ``ACTION:`` line.

        Looks for a line shaped ``ACTION: <VERDICT> [- justification]``
        in ``text`` (default: ``self.final_output``) and returns a
        :class:`ActionDecision`. Returns ``None`` if no ACTION line
        is found.

        Common verdicts produced by the bundled quant-desk preset are
        ``EXECUTE``, ``EXECUTE-WITH-MODS``, and ``NO-ACTION``, but the
        parser is verdict-neutral.
        """
        haystack = text if text is not None else self.final_output
        match = self._ACTION_LINE_RE.search(haystack or "")
        if match is None:
            return None
        action = match.group("action").strip().upper()
        rest = (match.group("rest") or "").lstrip(" :;,-—").rstrip()
        return ActionDecision(
            action=action,
            justification=rest,
            raw=match.group(0).strip(),
        )


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

    if not isinstance(name, str) or not name.strip():
        raise ValueError("agent(name=...) requires a non-empty string")
    if not isinstance(instructions, str) or not instructions.strip():
        raise ValueError(
            "agent(..., instructions=...) requires a non-empty string"
        )
    return Agent(
        name=name,
        instructions=instructions,
        role=role,
        backend=backend,
        temperature=temperature,
        max_tokens=max_tokens,
    )


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------


def _safe_usage_snapshot(backend: Any) -> dict[str, float]:
    """Best-effort usage read. Returns zero-filled dict on backends
    that don't implement ``usage_snapshot``."""
    snap = getattr(backend, "usage_snapshot", None)
    if not callable(snap):
        return {"prompt_tokens": 0, "output_tokens": 0,
                "n_calls": 0, "wall_s": 0.0}
    try:
        out = snap()
    except Exception:
        return {"prompt_tokens": 0, "output_tokens": 0,
                "n_calls": 0, "wall_s": 0.0}
    return {
        "prompt_tokens": int(out.get("prompt_tokens", 0) or 0),
        "output_tokens": int(out.get("output_tokens", 0) or 0),
        "n_calls": int(out.get("n_calls", 0) or 0),
        "wall_s": float(out.get("wall_s", 0.0) or 0.0),
    }


# Public type aliases
ProgressCallback = Callable[[AgentTurn], None]
ShouldContinue = Callable[[AgentTurn, Sequence[AgentTurn]], bool]


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
        print(result.total_tokens, result.total_wall_ms)
    """

    def __init__(
        self,
        agents: Sequence[Agent],
        *,
        backend: Any | None = None,
        team_instructions: str = "",
        max_visible_handoffs: int = 4,
        capture_capsules: bool = True,
        task_summary: str | None = None,
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
        self.task_summary = (
            task_summary.strip() if task_summary else None)
        # Override the per-handoff capsule budget. ``None`` means
        # auto-size each handoff to ``max(member.max_tokens,
        # output_words+32, 128)`` per turn — fits typical LLM agent
        # turns out of the box. Pass an explicit
        # :class:`CapsuleBudget` to tighten for benchmarks or widen
        # for very long turns.
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
        task_summary: str | None = None,
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
            task_summary=task_summary,
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
        task_summary: str | None = None,
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
            task_summary=task_summary,
            handoff_budget=handoff_budget,
        )

    def run(
        self,
        task: str,
        *,
        progress: ProgressCallback | None = None,
        should_continue: ShouldContinue | None = None,
    ) -> TeamResult:
        """Run the team once over ``task`` and return a stable result.

        ``progress`` is called once per completed turn with the
        finalised :class:`AgentTurn`. This is the live-UX hook for
        long runs (e.g. local LLM queries that take 5..30s) — without
        it the user sits in silence until the whole team finishes.

        ``should_continue`` is called after each turn with the
        finalised turn and the running list of all turns so far.
        Return ``False`` to stop the team early; the synthesizer
        will not be called and the prior turn's output becomes
        ``final_output``. Use this for "the risk manager rejected
        all signals — abort" patterns. Default = always continue.
        """

        ledger = CapsuleLedger() if self.capture_capsules else None
        turns: list[AgentTurn] = []
        recent_handoffs: list[tuple[str, str]] = []
        all_prior_outputs: list[tuple[str, str]] = []
        parent_cid: str | None = None
        total_prompt_tokens = 0
        total_output_tokens = 0
        total_wall_ms = 0.0
        total_calls = 0
        stopped_early = False
        head_backend = self.backend
        head_model = getattr(head_backend, "model", "") or ""
        head_base = getattr(head_backend, "base_url", None)

        for idx, member in enumerate(self.agents):
            backend = self._resolve_backend(member)
            role = member.effective_role
            visible_count = min(
                len(recent_handoffs), self.max_visible_handoffs)
            prompt, naive_prompt = self._build_prompt(
                member=member,
                task=task,
                turn_index=idx,
                recent_handoffs=recent_handoffs,
                all_prior_outputs=all_prior_outputs,
            )
            prompt_sha = _sha256_str(prompt)
            prompt_bytes = len(prompt.encode("utf-8"))
            prompt_words = len(prompt.split())
            naive_prompt_words = len(naive_prompt.split())
            backend_model = getattr(backend, "model", "") or ""

            usage_before = _safe_usage_snapshot(backend)
            t0 = time.time()
            output = backend.generate(
                prompt,
                max_tokens=member.max_tokens,
                temperature=member.temperature,
            )
            wall_ms = (time.time() - t0) * 1000.0
            usage_after = _safe_usage_snapshot(backend)
            d_prompt = max(
                0,
                int(usage_after["prompt_tokens"])
                - int(usage_before["prompt_tokens"]),
            )
            d_output = max(
                0,
                int(usage_after["output_tokens"])
                - int(usage_before["output_tokens"]),
            )
            d_calls = max(
                0,
                int(usage_after["n_calls"])
                - int(usage_before["n_calls"]),
            )

            capsule_cid: str | None = None
            if ledger is not None:
                next_role = (
                    self.agents[idx + 1].effective_role
                    if idx + 1 < len(self.agents)
                    else "team_output"
                )
                # If the caller pinned an explicit handoff_budget,
                # honour it. Otherwise size per-turn to the agent's
                # own generation cap plus headroom — the
                # research-surface default of max_tokens=64 is too
                # tight for free-form LLM outputs and would otherwise
                # blow up admission on a normal-length response.
                payload_words = max(1, len((output or "").split()))
                if self.handoff_budget is not None:
                    handoff_budget = self.handoff_budget
                else:
                    handoff_max_tokens = max(
                        member.max_tokens, payload_words + 32, 128)
                    handoff_budget = CapsuleBudget(
                        max_bytes=1 << 14,
                        max_tokens=handoff_max_tokens,
                        max_parents=8,
                    )
                handoff = capsule_team_handoff(
                    source_role=role,
                    to_role=next_role,
                    claim_kind="agent_output",
                    payload=output,
                    round=0,
                    parents=(parent_cid,) if parent_cid else (),
                    n_tokens=payload_words,
                    budget=handoff_budget,
                    prompt_sha256=prompt_sha,
                    prompt_bytes=prompt_bytes,
                    model_tag=backend_model,
                )
                sealed = ledger.admit_and_seal(handoff)
                capsule_cid = sealed.cid
                parent_cid = sealed.cid

            backend_base = getattr(backend, "base_url", None)
            turn = AgentTurn(
                agent_name=member.name,
                role=role,
                prompt=prompt,
                output=output,
                capsule_cid=capsule_cid,
                prompt_tokens=d_prompt,
                output_tokens=d_output,
                wall_ms=wall_ms,
                visible_handoffs=visible_count,
                prompt_sha256=prompt_sha,
                model_tag=backend_model,
                prompt_words=prompt_words,
                naive_prompt_words=naive_prompt_words,
                temperature=float(member.temperature),
                max_tokens=int(member.max_tokens),
                backend_base_url=backend_base,
            )
            turns.append(turn)
            total_prompt_tokens += d_prompt
            total_output_tokens += d_output
            total_wall_ms += wall_ms
            total_calls += d_calls or 1
            all_prior_outputs.append((role, output))
            if progress is not None:
                try:
                    progress(turn)
                except Exception:
                    # Progress callback failures must never abort a
                    # team run; surface via stderr so users still see
                    # their callback misbehaving.
                    import traceback as _tb
                    import sys as _sys
                    print(
                        "[AgentTeam] progress callback raised; "
                        "continuing run:",
                        file=_sys.stderr,
                    )
                    _tb.print_exc()
            recent_handoffs.append((role, output))
            if len(recent_handoffs) > self.max_visible_handoffs:
                recent_handoffs = recent_handoffs[-self.max_visible_handoffs:]

            if should_continue is not None:
                try:
                    keep_going = bool(should_continue(turn, tuple(turns)))
                except Exception:
                    # Continue-callback failures should never lose
                    # the user's run. Default to "keep going".
                    import traceback as _tb
                    import sys as _sys
                    print(
                        "[AgentTeam] should_continue raised; "
                        "continuing run:",
                        file=_sys.stderr,
                    )
                    _tb.print_exc()
                    keep_going = True
                if not keep_going:
                    stopped_early = True
                    break

        # include_payload=True so the on-disk view carries the
        # actual handoff bodies. The team handoff payloads are
        # already bounded by the per-agent token budget; without the
        # payloads, the audit story degrades to a hash chain whose
        # contents are unrecoverable.
        view = (
            render_view(
                ledger, root_cid=parent_cid, include_payload=True,
            ).as_dict()
            if ledger is not None
            else None
        )
        final_output = turns[-1].output if turns else ""
        root_cid = (
            view.get("root_cid") if view is not None else None
        ) or parent_cid
        return TeamResult(
            task=task,
            final_output=final_output,
            turns=tuple(turns),
            capsule_view=view,
            root_cid=root_cid,
            total_prompt_tokens=int(total_prompt_tokens),
            total_output_tokens=int(total_output_tokens),
            total_wall_ms=float(total_wall_ms),
            total_calls=int(total_calls),
            backend_model=str(head_model),
            backend_base_url=head_base,
            team_instructions=self.team_instructions,
            task_summary=self.task_summary,
            max_visible_handoffs=int(self.max_visible_handoffs),
            stopped_early=bool(stopped_early),
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
        turn_index: int,
        recent_handoffs: Sequence[tuple[str, str]],
        all_prior_outputs: Sequence[tuple[str, str]] | None = None,
    ) -> tuple[str, str]:
        """Construct the bounded prompt CoordPy will actually send,
        plus a shadow naive-cramming prompt of the same task that a
        token-cramming caller would have built.

        The naive prompt always includes the full task (no summary)
        and ALL upstream handoffs (no bounding). It is never sent to
        the model — it is built only so the SDK can report honest
        bounded-context savings in :class:`AgentTurn`.

        Returns ``(bounded_prompt, naive_prompt)``.
        """
        all_prior = list(all_prior_outputs or recent_handoffs)

        parts: list[str] = []
        if self.team_instructions:
            parts.append(self.team_instructions)
        parts.append(f"Agent: {member.name}")
        parts.append(f"Role: {member.effective_role}")
        parts.append(member.instructions.strip())
        # Token-saver: agent 0 always sees the full task; agents
        # 1..N see ``task_summary`` (when configured) instead, since
        # the upstream handoffs already preserve the bounded context.
        # This is the bounded-context savings story made operational.
        if turn_index == 0 or self.task_summary is None:
            parts.append(f"Task: {task.strip()}")
        else:
            parts.append(f"Task summary: {self.task_summary.strip()}")
        if recent_handoffs:
            rendered = "\n".join(
                f"- {role}: {text}"
                for role, text in recent_handoffs[-self.max_visible_handoffs:]
            )
            parts.append(
                "Visible team handoffs (bounded to avoid token cramming):\n"
                f"{rendered}"
            )
        parts.append("Reply with your contribution for the next team member.")
        bounded = "\n\n".join(parts)

        # Shadow naive prompt: full task each turn, every upstream
        # handoff. The naive caller would have sent this.
        naive_parts: list[str] = []
        if self.team_instructions:
            naive_parts.append(self.team_instructions)
        naive_parts.append(f"Agent: {member.name}")
        naive_parts.append(f"Role: {member.effective_role}")
        naive_parts.append(member.instructions.strip())
        naive_parts.append(f"Task: {task.strip()}")
        if all_prior:
            naive_rendered = "\n".join(
                f"- {role}: {text}" for role, text in all_prior
            )
            naive_parts.append(
                "Full prior conversation (naive cramming "
                "counterfactual):\n" f"{naive_rendered}"
            )
        naive_parts.append(
            "Reply with your contribution for the next team member.")
        naive = "\n\n".join(naive_parts)
        return bounded, naive


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
    task_summary: str | None = None,
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
        task_summary=task_summary,
        handoff_budget=handoff_budget,
    )


# ----------------------------------------------------------------------
# Replay
# ----------------------------------------------------------------------


def replay_team_result(
    result_path: str | os.PathLike,
    *,
    backend: Any,
    capture_capsules: bool = True,
    progress: ProgressCallback | None = None,
) -> TeamResult:
    """Re-run a sealed ``team_result.json`` against a fresh backend.

    Reads the per-turn prompts (recorded by :meth:`TeamResult.dump`)
    in order and queries ``backend`` once per turn, sealing fresh
    ``TEAM_HANDOFF`` capsules. Returns a brand-new :class:`TeamResult`
    whose chain root is the trailing handoff of the replay.

    The original ``prompt_sha256`` recorded on each turn is preserved
    in the new handoff payload alongside the *new* model tag, so an
    auditor can prove "the same prompt was asked, of model M2 instead
    of M1." This is the lightweight team's reproducibility surface.

    Per-turn ``temperature`` and ``max_tokens`` are read from the
    manifest and re-applied, so a manifest sealed at temperature=0.0
    replays at temperature=0.0 even if the loader's defaults differ.
    Manifests written before the replay-fidelity fix (no
    ``temperature`` / ``max_tokens`` keys) fall back to the
    lightweight defaults.

    The replay does NOT re-derive prompts from upstream handoffs —
    it uses the prompts exactly as recorded, which is the only way
    to faithfully reproduce a temperature>0 run. If you want the
    full chain to re-derive (e.g. to test a different bounded
    context), build a fresh :class:`AgentTeam` and call ``run``.
    """
    if not isinstance(backend, LLMBackend):
        raise TypeError("backend must satisfy the LLMBackend protocol")

    with open(result_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("schema") != TEAM_RESULT_SCHEMA:
        raise ValueError(
            f"{result_path!r} is not a {TEAM_RESULT_SCHEMA} manifest "
            f"(found schema={manifest.get('schema')!r})")
    turns_in = list(manifest.get("turns") or [])
    if not turns_in:
        raise ValueError(
            f"{result_path!r} has no turns to replay")

    ledger = CapsuleLedger() if capture_capsules else None
    turns_out: list[AgentTurn] = []
    parent_cid: str | None = None
    total_prompt_tokens = 0
    total_output_tokens = 0
    total_wall_ms = 0.0
    total_calls = 0

    for idx, t in enumerate(turns_in):
        prompt = t.get("prompt") or ""
        if not prompt:
            raise ValueError(
                f"turn {idx} ({t.get('role')!r}) has no recorded "
                f"prompt; manifest is too old to replay")
        # Generation params are persisted on every turn since v0.5.20;
        # older manifests fall back to the lightweight defaults so a
        # v0.5.16..v0.5.19 ``team_result.json`` still replays.
        max_tokens = int(t.get("max_tokens") or 256)
        temperature = (
            float(t["temperature"])
            if t.get("temperature") is not None else 0.2
        )

        usage_before = _safe_usage_snapshot(backend)
        t0 = time.time()
        output = backend.generate(
            prompt, max_tokens=max_tokens, temperature=temperature)
        wall_ms = (time.time() - t0) * 1000.0
        usage_after = _safe_usage_snapshot(backend)
        d_prompt = max(
            0,
            int(usage_after["prompt_tokens"])
            - int(usage_before["prompt_tokens"]),
        )
        d_output = max(
            0,
            int(usage_after["output_tokens"])
            - int(usage_before["output_tokens"]),
        )
        d_calls = max(
            0,
            int(usage_after["n_calls"])
            - int(usage_before["n_calls"]),
        )

        prompt_sha = _sha256_str(prompt)
        prompt_bytes = len(prompt.encode("utf-8"))
        original_sha = t.get("prompt_sha256")
        if original_sha and original_sha != prompt_sha:
            # The recorded prompt's hash and the re-hashed prompt
            # don't agree -- this would only happen on a tampered
            # team_result.json. Fail loudly.
            raise ValueError(
                f"turn {idx}: prompt SHA mismatch "
                f"(manifest={original_sha[:12]}…, "
                f"recomputed={prompt_sha[:12]}…)")

        capsule_cid: str | None = None
        backend_model = getattr(backend, "model", "") or ""
        if ledger is not None:
            next_role = (
                turns_in[idx + 1].get("role", "team_output")
                if idx + 1 < len(turns_in)
                else "team_output"
            )
            payload_words = max(1, len((output or "").split()))
            handoff_max_tokens = max(max_tokens, payload_words + 32, 128)
            handoff = capsule_team_handoff(
                source_role=t.get("role") or t.get("agent_name") or "agent",
                to_role=next_role,
                claim_kind="agent_output",
                payload=output,
                round=0,
                parents=(parent_cid,) if parent_cid else (),
                n_tokens=payload_words,
                budget=CapsuleBudget(
                    max_bytes=1 << 14,
                    max_tokens=handoff_max_tokens,
                    max_parents=8,
                ),
                prompt_sha256=prompt_sha,
                prompt_bytes=prompt_bytes,
                model_tag=backend_model,
            )
            sealed = ledger.admit_and_seal(handoff)
            capsule_cid = sealed.cid
            parent_cid = sealed.cid

        turn_out = AgentTurn(
            agent_name=t.get("agent_name") or t.get("role") or "agent",
            role=t.get("role") or t.get("agent_name") or "agent",
            prompt=prompt,
            output=output,
            capsule_cid=capsule_cid,
            prompt_tokens=d_prompt,
            output_tokens=d_output,
            wall_ms=wall_ms,
            visible_handoffs=int(t.get("visible_handoffs") or 0),
            prompt_sha256=prompt_sha,
            model_tag=backend_model,
            prompt_words=int(
                t.get("prompt_words") or len(prompt.split())),
            naive_prompt_words=int(t.get("naive_prompt_words") or 0),
            temperature=float(temperature),
            max_tokens=int(max_tokens),
            backend_base_url=getattr(backend, "base_url", None),
        )
        turns_out.append(turn_out)
        total_prompt_tokens += d_prompt
        total_output_tokens += d_output
        total_wall_ms += wall_ms
        total_calls += d_calls or 1
        if progress is not None:
            try:
                progress(turn_out)
            except Exception:
                import traceback as _tb
                import sys as _sys
                print(
                    "[replay_team_result] progress callback raised; "
                    "continuing replay:",
                    file=_sys.stderr,
                )
                _tb.print_exc()

    view = (
        render_view(
            ledger, root_cid=parent_cid, include_payload=True,
        ).as_dict()
        if ledger is not None
        else None
    )
    return TeamResult(
        task=manifest.get("task") or "",
        final_output=turns_out[-1].output if turns_out else "",
        turns=tuple(turns_out),
        capsule_view=view,
        root_cid=parent_cid,
        total_prompt_tokens=int(total_prompt_tokens),
        total_output_tokens=int(total_output_tokens),
        total_wall_ms=float(total_wall_ms),
        total_calls=int(total_calls),
        backend_model=getattr(backend, "model", "") or "",
        backend_base_url=getattr(backend, "base_url", None),
        team_instructions=manifest.get("team_instructions") or "",
        task_summary=manifest.get("task_summary"),
        max_visible_handoffs=int(
            manifest.get("max_visible_handoffs") or 0),
        stopped_early=False,
    )


def _md_render(result: "TeamResult", *, title: str | None = None) -> str:
    """Polished Markdown report for a single team run.

    Includes the final output, per-turn telemetry table, the
    bounded-context savings block, and audit handles
    (chain_head / root_cid / capsule schema). Used by
    :meth:`TeamResult.render_markdown` and persisted by
    :meth:`TeamResult.dump` as ``team_report.md``.
    """
    if title is None:
        title = "CoordPy team run"
    out: list[str] = []
    out.append(f"# {title}\n")
    out.append("## Run summary\n")
    out.append("| field | value |")
    out.append("|---|---|")
    out.append(
        f"| backend | `{result.backend_model}` "
        f"@ `{result.backend_base_url or '(default)'}` |"
    )
    out.append(f"| n_turns | {len(result.turns)} |")
    out.append(
        f"| max_visible_handoffs | {result.max_visible_handoffs} |")
    out.append(
        f"| stopped_early | {'yes' if result.stopped_early else 'no'} |")
    out.append(
        f"| total_prompt_tokens | {result.total_prompt_tokens} |")
    out.append(
        f"| total_output_tokens | {result.total_output_tokens} |")
    out.append(f"| total_tokens | {result.total_tokens} |")
    out.append(
        f"| total_wall | {result.total_wall_ms / 1000.0:.2f} s |")
    out.append(f"| capsule_root | `{result.root_cid or '-'}` |")
    if result.capsule_view is not None:
        out.append(
            f"| capsule_schema | `{result.capsule_view.get('schema')}` |")
        out.append(
            f"| chain_head | "
            f"`{(result.capsule_view.get('chain_head') or '-')}` |")
        out.append(
            f"| chain_ok | "
            f"{'yes' if result.capsule_view.get('chain_ok') else 'NO'} |")
    out.append("")

    cramming = result.cramming_estimate()
    out.append("## Bounded-context savings vs naive cramming\n")
    out.append("| metric | value |")
    out.append("|---|---|")
    out.append(
        f"| bounded_prompt_words | {cramming['bounded_words']} |")
    out.append(f"| naive_prompt_words | {cramming['naive_words']} |")
    out.append(f"| saved_words | {cramming['saved_words']} |")
    out.append(f"| savings_pct | {cramming['savings_pct']:.1f}% |")
    out.append(
        f"| estimated_tokens_saved | "
        f"{cramming['estimated_tokens_saved']} |")
    out.append("")

    out.append("## Per-turn telemetry\n")
    out.append(
        "| # | role | in_tok | out_tok | wall_s | "
        "vis | bounded_w | naive_w | capsule |"
    )
    out.append("|--:|---|--:|--:|--:|--:|--:|--:|---|")
    for i, t in enumerate(result.turns):
        cid = (t.capsule_cid or "")[:12]
        out.append(
            f"| {i} | {t.role} | {t.prompt_tokens} | "
            f"{t.output_tokens} | {t.wall_ms / 1000.0:.2f} | "
            f"{t.visible_handoffs} | {t.prompt_words} | "
            f"{t.naive_prompt_words} | `{cid}` |"
        )
    out.append("")

    action = result.parse_action()
    if action is not None:
        out.append("## Parsed action\n")
        out.append(f"- **action**: `{action.action}`")
        if action.justification:
            out.append(f"- **justification**: {action.justification}")
        out.append("")

    out.append("## Final output\n")
    out.append("```")
    out.append((result.final_output or "").rstrip())
    out.append("```\n")

    out.append("## Audit\n")
    out.append(
        "Re-hash this run from disk:\n\n"
        "```\n"
        "coordpy-capsule verify-view --view team_capsule_view.json\n"
        "coordpy-capsule view        --view team_capsule_view.json --full\n"
        "```\n"
    )
    out.append(
        "Re-run the same prompts against another model:\n\n"
        "```\n"
        "coordpy-team replay --result team_result.json \\\n"
        "    --backend ollama --model gemma2:9b \\\n"
        "    --out-dir /tmp/replay\n"
        "```\n"
    )
    return "\n".join(out)


__all__ = [
    "Agent",
    "AgentTurn",
    "ActionDecision",
    "TeamResult",
    "AgentTeam",
    "ProgressCallback",
    "ShouldContinue",
    "TEAM_RESULT_SCHEMA",
    "agent",
    "create_team",
    "replay_team_result",
]
