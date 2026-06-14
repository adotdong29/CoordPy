"""coordpy.adk.runner — the execution engine.

``Runner`` drives one agent over a ``SessionService`` (and optional
artifact / memory services). ``runner.run(...)`` yields a *stream* of typed
``Event``s; gate on ``event.is_final_response()`` for the user-facing answer.
Every event is persisted via ``session_service.append_event`` (applying its
``state_delta``) and sealed into a hash-chained capsule trail, so
``runner.session_capsule_view(session_id)`` returns a re-verifiable
``coordpy.capsule_view.v1`` chain.

``InMemoryRunner(agent)`` wires zero-config in-memory services — the fastest
path to a running agent. Swapping in persistent services later is a one-line
change with identical agent code (the Runner seam).
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Iterator

from coordpy.capsule import verify_chain_from_view_dict

from ._capsule_trail import CapsuleTrail
from .artifacts import InMemoryArtifactService
from .context import InvocationContext
from .memory import InMemoryMemoryService
from .sessions import (
    BaseSessionService, Event, InMemorySessionService, Session, State,
    new_invocation_id,
)

ADK_RUN_REPORT_SCHEMA = "coordpy.adk.run.v1"


class Runner:
    """Drive an agent: stream events, persist state, seal the capsule trail."""

    def __init__(self, *, agent: Any, app_name: str,
                 session_service: BaseSessionService,
                 artifact_service: Any = None,
                 memory_service: Any = None) -> None:
        self.agent = agent
        self.app_name = app_name
        self.session_service = session_service
        self.artifact_service = artifact_service
        self.memory_service = memory_service
        self._trails: dict[str, CapsuleTrail] = {}

    def _message_text(self, new_message: Any) -> str:
        if isinstance(new_message, str):
            return new_message
        return getattr(new_message, "text", None) or str(new_message)

    def run(self, *, user_id: str, session_id: str,
            new_message: Any) -> Iterator[Event]:
        text = self._message_text(new_message)
        session = self.session_service.get_session(
            app_name=self.app_name, user_id=user_id, session_id=session_id)
        if session is None:
            session = self.session_service.create_session(
                app_name=self.app_name, user_id=user_id, session_id=session_id)
        merged = self.session_service.get_merged_state(session)

        ctx = InvocationContext(
            session=session, agent=self.agent, root_agent=self.agent,
            user_content=text, invocation_id=new_invocation_id(),
            session_service=self.session_service,
            artifact_service=self.artifact_service,
            memory_service=self.memory_service, state=State(merged))

        trail = CapsuleTrail()
        self._trails[session_id] = trail

        user_ev = Event(author="user", content=text,
                        invocation_id=ctx.invocation_id)
        self.session_service.append_event(session, user_ev)
        trail.seal_event(user_ev)

        try:
            for ev in self.agent.run(ctx):
                self.session_service.append_event(session, ev)
                trail.seal_event(ev)
                yield ev
        finally:
            # The hashed summary excludes the volatile invocation_id so the
            # root_cid is deterministic for identical inputs.
            trail.seal_run_report({
                "schema": ADK_RUN_REPORT_SCHEMA,
                "app_name": self.app_name, "user_id": user_id,
                "session_id": session_id,
                "n_events": len(session.events), "n_capsules": trail.n,
            })

    async def run_async(self, *, user_id: str, session_id: str,
                        new_message: Any) -> AsyncIterator[Event]:
        for ev in self.run(user_id=user_id, session_id=session_id,
                           new_message=new_message):
            yield ev

    # -- the CoordPy guarantees, exposed -----------------------------------
    def session_capsule_view(self, session_id: str) -> dict[str, Any] | None:
        trail = self._trails.get(session_id)
        return trail.view() if trail is not None else None

    def session_root_cid(self, session_id: str) -> str | None:
        trail = self._trails.get(session_id)
        return trail.root_cid if trail is not None else None

    def verify_session(self, session_id: str) -> bool:
        view = self.session_capsule_view(session_id)
        return bool(view) and verify_chain_from_view_dict(view)


class InMemoryRunner(Runner):
    """A Runner with zero-config in-memory session / artifact / memory."""

    def __init__(self, *, agent: Any, app_name: str = "adk_app") -> None:
        super().__init__(
            agent=agent, app_name=app_name,
            session_service=InMemorySessionService(),
            artifact_service=InMemoryArtifactService(),
            memory_service=InMemoryMemoryService())
