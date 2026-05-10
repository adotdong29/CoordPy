"""Curated agent-team presets for the lightweight CoordPy surface.

These presets exist so an external builder can stand up a useful
multi-agent team in one line, without re-typing role instructions or
team-coordination prose every time. They are intentionally
opinionated: each preset has been smoke-tested against the bundled
example scenarios and the SDK v3.43 stable surface.

All presets return a configured :class:`AgentTeam`. Callers can:

* override the model / backend via env (``COORDPY_BACKEND``,
  ``COORDPY_MODEL``, ``COORDPY_OLLAMA_URL`` / ``COORDPY_API_KEY``);
* override per-agent ``temperature`` / ``max_tokens`` via the
  preset's keyword arguments;
* swap individual agents by accessing ``team.agents`` and rebuilding.

The presets module is part of the stable SDK surface and follows the
same backwards-compat policy as :mod:`coordpy.agents`.
"""

from __future__ import annotations

from typing import Sequence

from .agents import Agent, AgentTeam, agent


__all__ = [
    "quant_desk_team",
    "code_review_team",
    "research_writer_team",
    "QUANT_DESK_TASK_SUMMARY",
]


QUANT_DESK_TASK_SUMMARY = (
    "Produce a single desk note for a small US-equity quant desk. "
    "Roles: market_watcher (regime + macro + name reads), "
    "signal_researcher (<=2 candidate signals with falsifiers), "
    "risk_manager (PASS/MODIFY/REJECT under firm constraints), "
    "portfolio_synthesizer (final desk note with MARKET VIEW / "
    "SIGNAL SUMMARY / RISK VIEW / ACTION). Bounded-context: rely on "
    "visible handoffs, do not restate the scenario."
)


def quant_desk_team(
    *,
    model: str | None = None,
    backend_name: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: float | None = None,
    temperature: float = 0.0,
    max_tokens: int = 360,
    max_visible_handoffs: int = 2,
    capture_capsules: bool = True,
    extra_instructions: str = "",
) -> AgentTeam:
    """Return a four-role quant trading desk team.

    Roles, in order:

    1. ``market_watcher`` — regime + macro + name reads in <=6 bullets.
    2. ``signal_researcher`` — at most 2 candidate signals with
       falsifiers; no position sizing.
    3. ``risk_manager`` — PASS / MODIFY / REJECT each candidate
       against the firm's risk constraints.
    4. ``portfolio_synthesizer`` — final desk note with MARKET VIEW,
       SIGNAL SUMMARY, RISK VIEW, and ACTION ∈
       {EXECUTE, EXECUTE-WITH-MODS, NO-ACTION}.

    Defaults are tuned for honest desk output on local Ollama 14B
    models: ``temperature=0.0`` produces decisive ACTION verdicts;
    ``max_visible_handoffs=2`` is the bounded-context default that
    keeps the writer from re-cramming the full transcript; and
    ``task_summary`` is set so agents 2..N see a one-line context
    pointer instead of the full scenario.

    Set ``extra_instructions`` to add a project-specific note (e.g.
    "this desk runs in Asia time, prefer overnight setups") to the
    team_instructions block. Per-agent ``temperature`` / ``max_tokens``
    are uniform; if you need heterogeneous configs, build the team
    by hand using :func:`~coordpy.agent` instead.
    """
    desk: list[Agent] = [
        agent(
            "market_watcher",
            "You are a market watcher. In <=6 short bullets, summarise: "
            "(a) the regime (risk-on / mixed / risk-off), (b) the "
            "single most relevant macro signal, (c) the most "
            "important name-level read across the universe in the "
            "scenario. Be specific and quantitative. Do not "
            "recommend a trade.",
            max_tokens=max_tokens, temperature=temperature,
        ),
        agent(
            "signal_researcher",
            "You are a signal researcher. Using only the market view "
            "above, propose at most 2 candidate signals (long or "
            "short, single-name or pair). For each: name, direction, "
            "thesis in <=2 sentences, and a falsifier the risk "
            "manager can check. Do NOT size positions.",
            max_tokens=max_tokens, temperature=temperature,
        ),
        agent(
            "risk_manager",
            "You are the risk manager. Given the candidate signals, "
            "apply the firm constraints declared in the scenario "
            "(single-name caps, sector concentration caps, no new "
            "positions inside the earnings blackout, etc.). For each "
            "candidate: PASS, MODIFY, or REJECT, with one short "
            "reason. Flag any constraint you cannot verify from the "
            "visible context.",
            max_tokens=max_tokens, temperature=temperature,
        ),
        agent(
            "portfolio_synthesizer",
            "You are the portfolio synthesizer. Write a final desk "
            "note in this exact structure:\n"
            "MARKET VIEW: <=2 sentences.\n"
            "SIGNAL SUMMARY: bullet list of accepted/modified signals.\n"
            "RISK VIEW: <=2 sentences on residual risk and what "
            "would invalidate the plan.\n"
            "ACTION: ONE of {EXECUTE, EXECUTE-WITH-MODS, NO-ACTION}, "
            "followed by a one-line justification.\n"
            "Be concise, decisive, and grounded only in the visible "
            "team handoffs. Do not invent numbers.",
            max_tokens=max_tokens, temperature=temperature,
        ),
    ]
    base_team_instructions = (
        "You are members of a small quant trading desk. Work as a "
        "bounded-context team: rely on the visible TEAM_HANDOFF "
        "outputs above -- do NOT restate the full scenario. Each "
        "role produces ONE handoff for the next role; brevity is "
        "rewarded."
    )
    team_instructions = (
        f"{base_team_instructions}\n\n{extra_instructions.strip()}"
        if extra_instructions.strip()
        else base_team_instructions
    )
    return AgentTeam.from_env(
        desk,
        model=model,
        backend_name=backend_name,
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
        team_instructions=team_instructions,
        max_visible_handoffs=max_visible_handoffs,
        capture_capsules=capture_capsules,
        task_summary=QUANT_DESK_TASK_SUMMARY,
    )


def code_review_team(
    *,
    model: str | None = None,
    backend_name: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: float | None = None,
    temperature: float = 0.0,
    max_tokens: int = 480,
    max_visible_handoffs: int = 2,
    capture_capsules: bool = True,
) -> AgentTeam:
    """Return a three-role code-review team.

    Roles: ``triage`` (categorise the change), ``security_reviewer``
    (focus on injection / auth / secrets), ``synth_reviewer``
    (reconcile and produce a single summary verdict).
    """
    members = [
        agent(
            "triage",
            "You are the triage reviewer. Read the diff or code "
            "snippet, classify the change in <=4 short bullets "
            "(intent, scope, risk surface, blast radius), and "
            "highlight ONE area that warrants deeper review.",
            max_tokens=max_tokens, temperature=temperature,
        ),
        agent(
            "security_reviewer",
            "You are the security reviewer. Focus on the deeper-review "
            "area flagged above. Look for: injection (SQL, command, "
            "template), authentication / authorization gaps, secret "
            "exposure, unsafe deserialization, race conditions. List "
            "concrete findings or 'NO ISSUES FOUND'.",
            max_tokens=max_tokens, temperature=temperature,
        ),
        agent(
            "synth_reviewer",
            "You are the synthesis reviewer. Reconcile triage + "
            "security findings. Output:\n"
            "SUMMARY: 1-2 sentences.\n"
            "FINDINGS: bullet list (severity, location, fix).\n"
            "VERDICT: ONE of {APPROVE, REQUEST-CHANGES, BLOCK}, "
            "with a one-line justification.\n"
            "Be concise and specific.",
            max_tokens=max_tokens, temperature=temperature,
        ),
    ]
    return AgentTeam.from_env(
        members,
        model=model,
        backend_name=backend_name,
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
        team_instructions=(
            "You are a code-review team. Bounded context: rely on "
            "the visible handoffs; do not restate the diff."
        ),
        max_visible_handoffs=max_visible_handoffs,
        capture_capsules=capture_capsules,
    )


def research_writer_team(
    *,
    model: str | None = None,
    backend_name: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: float | None = None,
    temperature: float = 0.2,
    max_tokens: int = 480,
    max_visible_handoffs: int = 2,
    capture_capsules: bool = True,
) -> AgentTeam:
    """Return a three-role research-and-writer team.

    Roles: ``planner`` (3-step research outline), ``researcher``
    (gather facts that matter), ``writer`` (produce the final
    answer). Useful as a baseline for prompt-engineering experiments.
    """
    members = [
        agent(
            "planner",
            "You are a research planner. Break the user task into 3 "
            "concrete research steps. Be specific: each step should "
            "be falsifiable.",
            max_tokens=max_tokens, temperature=temperature,
        ),
        agent(
            "researcher",
            "You are the researcher. Address each planner step in "
            "order, citing the facts that actually answer it. If "
            "you don't know a fact, say 'unknown' rather than "
            "inventing one.",
            max_tokens=max_tokens, temperature=temperature,
        ),
        agent(
            "writer",
            "You are the writer. Produce the final answer for the "
            "user. Use the researcher's facts; do not introduce new "
            "claims. Length: <=180 words.",
            max_tokens=max_tokens, temperature=temperature,
        ),
    ]
    return AgentTeam.from_env(
        members,
        model=model,
        backend_name=backend_name,
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
        team_instructions=(
            "You are a research-writer team. Bounded context: each "
            "role produces ONE handoff for the next."
        ),
        max_visible_handoffs=max_visible_handoffs,
        capture_capsules=capture_capsules,
    )
