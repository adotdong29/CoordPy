"""coordpy.adk.sessions — Session, State, Event, and SessionService.

This is the ADK-shaped session layer: a ``Session`` is one ongoing
conversation thread holding ``State`` (a prefix-scoped scratchpad) and a
list of ``Event`` objects (the turn-by-turn history). A ``SessionService``
owns session lifecycle; ``InMemorySessionService`` is the zero-config
default. State writes never mutate ``session.state`` directly — they flow
through ``append_event`` as ``EventActions.state_delta`` and are routed by
key prefix (``app:`` / ``user:`` / ``temp:`` / none), mirroring Google ADK.

Nothing here talks to a network or a model. The capsule/provenance layer
that seals each event lives in ``coordpy.adk.runner`` (the Runner drives
this service and a ``CapsuleTrail`` in lockstep).
"""

from __future__ import annotations

import dataclasses
import time
import uuid
from typing import Any, Iterable, Iterator, Mapping

# State key prefixes (persistence is encoded in the key, ADK-style).
APP_PREFIX = "app:"
USER_PREFIX = "user:"
TEMP_PREFIX = "temp:"


class State(Mapping):
    """A delta-tracked, prefix-aware key/value scratchpad.

    Reads see the merged live value; writes update the live value *and*
    record a pending delta. The Runner pops the delta into each emitted
    event's ``state_delta`` so the SessionService can persist it with
    prefix routing. ``temp:`` keys live only for the current invocation.
    """

    def __init__(self, initial: Mapping[str, Any] | None = None) -> None:
        self._value: dict[str, Any] = dict(initial or {})
        self._delta: dict[str, Any] = {}

    # -- read surface (Mapping) -------------------------------------------
    def __getitem__(self, key: str) -> Any:
        return self._value[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._value)

    def __len__(self) -> int:
        return len(self._value)

    def __contains__(self, key: object) -> bool:
        return key in self._value

    def get(self, key: str, default: Any = None) -> Any:
        return self._value.get(key, default)

    # -- write surface ----------------------------------------------------
    def __setitem__(self, key: str, val: Any) -> None:
        self._value[key] = val
        self._delta[key] = val

    def update(self, other: Mapping[str, Any]) -> None:
        for k, v in other.items():
            self[k] = v

    # -- delta plumbing (used by the Runner) ------------------------------
    def has_delta(self) -> bool:
        return bool(self._delta)

    def pop_delta(self) -> dict[str, Any]:
        """Return and clear the pending delta (one event's worth)."""
        d, self._delta = self._delta, {}
        return d

    def to_dict(self) -> dict[str, Any]:
        return dict(self._value)


@dataclasses.dataclass
class EventActions:
    """Side effects an event carries: state writes, artifact saves, and
    control-flow signals (delegate / early-exit)."""

    state_delta: dict[str, Any] = dataclasses.field(default_factory=dict)
    artifact_delta: dict[str, int] = dataclasses.field(default_factory=dict)
    transfer_to_agent: str | None = None
    escalate: bool = False
    skip_summarization: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "state_delta": dict(self.state_delta),
            "artifact_delta": dict(self.artifact_delta),
            "transfer_to_agent": self.transfer_to_agent,
            "escalate": self.escalate,
            "skip_summarization": self.skip_summarization,
        }


@dataclasses.dataclass
class Event:
    """The atomic unit of conversation + history.

    A turn is a *stream* of these. ``is_final_response()`` flags the one
    event a UI should display. Tool-call / tool-result / transfer events
    are intermediate (``is_final=False``).
    """

    author: str
    content: str | None = None
    actions: EventActions = dataclasses.field(default_factory=EventActions)
    invocation_id: str = ""
    id: str = dataclasses.field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: float = dataclasses.field(default_factory=time.time)
    partial: bool = False
    is_final: bool = False
    # Telemetry for tool / transfer events (not shown to the user).
    tool_call: dict[str, Any] | None = None
    tool_result: Any | None = None
    branch: str | None = None
    # Transient: raw artifact bytes for capsule sealing (not serialized).
    _artifact_blobs: list[tuple[str, int, bytes, str]] = dataclasses.field(
        default_factory=list, repr=False)

    @property
    def text(self) -> str:
        return self.content or ""

    def is_final_response(self) -> bool:
        """True iff this is the agent's user-facing final answer."""
        return self.is_final and self.author != "user" and bool(self.content) \
            and not self.partial

    def event_kind(self) -> str:
        if self.author == "user":
            return "user"
        if self.tool_call is not None:
            return "tool"
        if self.actions.transfer_to_agent is not None:
            return "transfer"
        if self.is_final:
            return "final"
        return "model"

    def as_dict(self) -> dict[str, Any]:
        return {
            "author": self.author,
            "content": self.content,
            "actions": self.actions.as_dict(),
            "invocation_id": self.invocation_id,
            "id": self.id,
            "partial": self.partial,
            "is_final": self.is_final,
            "kind": self.event_kind(),
            "tool_call": self.tool_call,
            "tool_result": self.tool_result,
        }

    def capsule_payload(self) -> dict[str, Any]:
        """Content-only view for sealing: excludes the volatile per-run
        ``invocation_id`` / event ``id`` so identical inputs yield identical
        capsule CIDs (deterministic provenance / replay)."""
        return {
            "author": self.author,
            "kind": self.event_kind(),
            "content": self.content,
            "tool_call": self.tool_call,
            "tool_result": self.tool_result,
            "actions": self.actions.as_dict(),
        }


@dataclasses.dataclass
class Session:
    """One conversation thread: identity + session-scoped state + history."""

    id: str
    app_name: str
    user_id: str
    state: dict[str, Any] = dataclasses.field(default_factory=dict)
    events: list[Event] = dataclasses.field(default_factory=list)
    last_update_time: float = 0.0


class BaseSessionService:
    """Session lifecycle interface. Subclass for a real backend."""

    def create_session(self, *, app_name: str, user_id: str,
                       session_id: str | None = None,
                       state: Mapping[str, Any] | None = None) -> Session:
        raise NotImplementedError

    def get_session(self, *, app_name: str, user_id: str,
                    session_id: str) -> Session | None:
        raise NotImplementedError

    def list_sessions(self, *, app_name: str, user_id: str) -> list[Session]:
        raise NotImplementedError

    def delete_session(self, *, app_name: str, user_id: str,
                       session_id: str) -> None:
        raise NotImplementedError

    def append_event(self, session: Session, event: Event) -> Event:
        raise NotImplementedError

    def get_merged_state(self, session: Session) -> dict[str, Any]:
        raise NotImplementedError


class InMemorySessionService(BaseSessionService):
    """Zero-config, process-local SessionService.

    Holds session-scoped state on each ``Session`` plus two cross-session
    stores keyed by prefix: ``app:`` (all users) and ``user:`` (one user).
    ``append_event`` applies ``state_delta`` with prefix routing; ``temp:``
    keys are intentionally dropped (never persisted).
    """

    def __init__(self) -> None:
        self._sessions: dict[tuple[str, str, str], Session] = {}
        self._app_state: dict[str, dict[str, Any]] = {}
        self._user_state: dict[tuple[str, str], dict[str, Any]] = {}

    def create_session(self, *, app_name: str, user_id: str,
                       session_id: str | None = None,
                       state: Mapping[str, Any] | None = None) -> Session:
        sid = session_id or uuid.uuid4().hex[:12]
        sess = Session(id=sid, app_name=app_name, user_id=user_id,
                       state=dict(state or {}), last_update_time=time.time())
        self._sessions[(app_name, user_id, sid)] = sess
        return sess

    def get_session(self, *, app_name: str, user_id: str,
                    session_id: str) -> Session | None:
        return self._sessions.get((app_name, user_id, session_id))

    def list_sessions(self, *, app_name: str, user_id: str) -> list[Session]:
        return [s for (a, u, _), s in self._sessions.items()
                if a == app_name and u == user_id]

    def delete_session(self, *, app_name: str, user_id: str,
                       session_id: str) -> None:
        self._sessions.pop((app_name, user_id, session_id), None)

    def append_event(self, session: Session, event: Event) -> Event:
        session.events.append(event)
        for key, val in (event.actions.state_delta or {}).items():
            if key.startswith(TEMP_PREFIX):
                continue  # invocation-only; never persisted
            if key.startswith(APP_PREFIX):
                self._app_state.setdefault(session.app_name, {})[key] = val
            elif key.startswith(USER_PREFIX):
                self._user_state.setdefault(
                    (session.app_name, session.user_id), {})[key] = val
            else:
                session.state[key] = val
        session.last_update_time = event.timestamp
        return event

    def get_merged_state(self, session: Session) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        merged.update(self._app_state.get(session.app_name, {}))
        merged.update(self._user_state.get(
            (session.app_name, session.user_id), {}))
        merged.update(session.state)
        return merged


def new_invocation_id() -> str:
    return "inv-" + uuid.uuid4().hex[:12]
