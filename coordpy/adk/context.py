"""coordpy.adk.context — the layered context objects.

``InvocationContext`` is the runtime container for one user→response cycle:
it holds the session, the current agent, the live ``State``, and the three
services. The three public contexts are progressively-capable views over it,
mirroring Google ADK:

  ``ReadonlyContext``  ⊂  ``CallbackContext``  ⊂  ``ToolContext``

* ``ReadonlyContext`` — read-only ``state`` (for instruction providers).
* ``CallbackContext`` — mutable ``state`` + artifact save/load (callbacks).
* ``ToolContext``     — adds ``actions``, ``function_call_id``, and
  ``search_memory`` (tool execution). Declared as the trailing
  ``tool_context`` parameter of a tool function, it is auto-injected and
  never appears in the model-facing tool schema.
"""

from __future__ import annotations

import dataclasses
import types
from typing import Any

from .artifacts import Artifact
from .memory import SearchMemoryResponse
from .sessions import EventActions, Session, State


@dataclasses.dataclass
class InvocationContext:
    """Full runtime state for one invocation. Mutated as the turn runs."""

    session: Session
    agent: Any
    root_agent: Any
    user_content: str
    invocation_id: str
    session_service: Any
    artifact_service: Any = None
    memory_service: Any = None
    state: State = dataclasses.field(default_factory=State)
    _pending_artifacts: list[tuple[str, int, bytes, str]] = dataclasses.field(
        default_factory=list, repr=False)

    @property
    def app_name(self) -> str:
        return self.session.app_name

    @property
    def user_id(self) -> str:
        return self.session.user_id


class ReadonlyContext:
    """Read-only view: ``state`` is an immutable snapshot."""

    def __init__(self, invocation_context: InvocationContext,
                 *, agent_name: str | None = None) -> None:
        self._ic = invocation_context
        self._agent_name = agent_name or getattr(
            invocation_context.agent, "name", "")

    @property
    def state(self):
        return types.MappingProxyType(self._ic.state.to_dict())

    @property
    def app_name(self) -> str:
        return self._ic.app_name

    @property
    def user_id(self) -> str:
        return self._ic.user_id

    @property
    def session_id(self) -> str:
        return self._ic.session.id

    @property
    def agent_name(self) -> str:
        return self._agent_name

    @property
    def invocation_id(self) -> str:
        return self._ic.invocation_id


class CallbackContext(ReadonlyContext):
    """Adds a mutable ``state`` and artifact save/load/list."""

    @property
    def state(self) -> State:
        return self._ic.state

    def save_artifact(self, filename: str, data: bytes,
                      mime_type: str = "application/octet-stream") -> int:
        if self._ic.artifact_service is None:
            raise RuntimeError(
                "no artifact_service configured on the Runner; pass one to "
                "Runner(..., artifact_service=InMemoryArtifactService())")
        version = self._ic.artifact_service.save_artifact(
            app_name=self._ic.app_name, user_id=self._ic.user_id,
            session_id=self._ic.session.id, filename=filename,
            data=bytes(data), mime_type=mime_type)
        self._ic._pending_artifacts.append(
            (filename, version, bytes(data), mime_type))
        return version

    def load_artifact(self, filename: str,
                      version: int | None = None) -> Artifact | None:
        if self._ic.artifact_service is None:
            return None
        return self._ic.artifact_service.load_artifact(
            app_name=self._ic.app_name, user_id=self._ic.user_id,
            session_id=self._ic.session.id, filename=filename, version=version)

    def list_artifacts(self) -> list[str]:
        if self._ic.artifact_service is None:
            return []
        return self._ic.artifact_service.list_artifact_keys(
            app_name=self._ic.app_name, user_id=self._ic.user_id,
            session_id=self._ic.session.id)


class ToolContext(CallbackContext):
    """The tool-execution context: state + artifacts + actions + memory."""

    def __init__(self, invocation_context: InvocationContext,
                 *, agent_name: str | None = None,
                 function_call_id: str | None = None) -> None:
        super().__init__(invocation_context, agent_name=agent_name)
        self.function_call_id = function_call_id
        self.actions = EventActions()

    def search_memory(self, query: str) -> SearchMemoryResponse:
        if self._ic.memory_service is None:
            return SearchMemoryResponse(memories=[])
        return self._ic.memory_service.search_memory(
            app_name=self._ic.app_name, user_id=self._ic.user_id, query=query)
