"""coordpy.adk — a Python-first Agent Development Kit, capsule-audited.

``coordpy.adk`` is CoordPy's library-first front door. The mental model is
the now-standard ADK one — **Agent · Tool · Runner · Session · State ·
Memory · Artifacts** — so it is familiar in one read:

    from coordpy.adk import Agent, Runner, InMemorySessionService

    def web_search(query: str) -> dict:
        '''Look up `query`.'''
        return {"status": "success", "results": [f"... {query} ..."]}

    assistant = Agent(
        name="assistant",
        model=my_backend,                 # any coordpy LLMBackend / callable
        instruction="Answer the user, using tools when helpful.",
        tools=[web_search],
    )

    runner = Runner(agent=assistant, app_name="demo",
                    session_service=InMemorySessionService())
    runner.session_service.create_session(
        app_name="demo", user_id="u", session_id="s")

    for event in runner.run(user_id="u", session_id="s",
                            new_message="What is CoordPy?"):
        if event.is_final_response():
            print(event.text)

    # …and for free, the CoordPy guarantee underneath:
    view = runner.session_capsule_view("s")   # sealed, content-addressed chain
    assert runner.verify_session("s")          # provenance re-verified from bytes

What CoordPy adds beneath the familiar surface: every model call, tool call,
and handoff seals into a typed, content-addressed, provenance-carrying
capsule (the C1..C6 Capsule Contract) that you can re-verify and replay.

This surface is **additive**. The legacy product ``coordpy.Agent`` /
``create_team`` / ``AgentTeam`` keep their signatures; the ADK ``Agent``
lives here at ``coordpy.adk.Agent`` and is a deliberately separate object.
The console scripts (``coordpy-team`` …) remain available as the secondary
CLI runtime surface.

Stability: ``coordpy.adk`` is the v1 library front door (``ADK_SURFACE_SCHEMA
== "coordpy.adk.v1"``). The reference Runner is synchronous (with an async
``run_async`` wrapper) and uses a text tool-call protocol + in-memory
services — a faithful, bounded subset, not a clone of every Google ADK
feature.
"""

from __future__ import annotations

from .agents import (
    Agent, BaseAgent, LlmAgent, LoopAgent, ParallelAgent, SequentialAgent,
)
from .artifacts import (
    Artifact, BaseArtifactService, InMemoryArtifactService,
)
from .context import (
    CallbackContext, InvocationContext, ReadonlyContext, ToolContext,
)
from .memory import (
    BaseMemoryService, InMemoryMemoryService, MemoryEntry,
    SearchMemoryResponse,
)
from .runner import ADK_RUN_REPORT_SCHEMA, InMemoryRunner, Runner
from .sessions import (
    APP_PREFIX, BaseSessionService, Event, EventActions,
    InMemorySessionService, Session, State, TEMP_PREFIX, USER_PREFIX,
)
from .tools import BaseTool, FunctionTool, to_function_tool

ADK_SURFACE_SCHEMA = "coordpy.adk.v1"

__all__ = [
    # agents + workflows
    "Agent", "LlmAgent", "BaseAgent",
    "SequentialAgent", "ParallelAgent", "LoopAgent",
    # tools
    "BaseTool", "FunctionTool", "to_function_tool",
    # runner
    "Runner", "InMemoryRunner", "ADK_RUN_REPORT_SCHEMA",
    # sessions / state / events
    "Session", "State", "Event", "EventActions",
    "BaseSessionService", "InMemorySessionService",
    "APP_PREFIX", "USER_PREFIX", "TEMP_PREFIX",
    # artifacts
    "Artifact", "BaseArtifactService", "InMemoryArtifactService",
    # memory
    "MemoryEntry", "SearchMemoryResponse",
    "BaseMemoryService", "InMemoryMemoryService",
    # context
    "ReadonlyContext", "CallbackContext", "ToolContext", "InvocationContext",
    # schema
    "ADK_SURFACE_SCHEMA",
]
