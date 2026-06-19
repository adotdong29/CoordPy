"""coordpy.adk.memory — long-term, cross-session recall.

Where ``Session``/``State`` is the *current* conversation, ``Memory`` is
searchable knowledge spanning *many past sessions*. ``InMemoryMemoryService``
ingests finished sessions and answers keyword queries — the zero-config
default. A real deployment swaps in a vector/RAG backend behind the same
``BaseMemoryService`` interface (the Runner seam), with no agent-code change.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid an import cycle at module load
    from .sessions import Session


@dataclasses.dataclass(frozen=True)
class MemoryEntry:
    """One recalled snippet plus where it came from."""

    text: str
    author: str
    session_id: str
    score: float = 0.0


@dataclasses.dataclass
class SearchMemoryResponse:
    memories: list[MemoryEntry] = dataclasses.field(default_factory=list)


def _tokens(text: str) -> set[str]:
    return {w for w in "".join(
        c.lower() if c.isalnum() else " " for c in text).split() if w}


class BaseMemoryService:
    def add_session_to_memory(self, session: "Session") -> None:
        raise NotImplementedError

    def search_memory(self, *, app_name: str, user_id: str,
                      query: str) -> SearchMemoryResponse:
        raise NotImplementedError


class InMemoryMemoryService(BaseMemoryService):
    """Keyword-overlap recall over ingested sessions (deterministic)."""

    def __init__(self) -> None:
        # (app_name, user_id) -> list of (text, author, session_id)
        self._entries: dict[tuple[str, str], list[tuple[str, str, str]]] = {}

    def add_session_to_memory(self, session: "Session") -> None:
        bucket = self._entries.setdefault((session.app_name, session.user_id), [])
        for ev in session.events:
            if ev.content:
                bucket.append((ev.content, ev.author, session.id))

    def search_memory(self, *, app_name: str, user_id: str,
                      query: str) -> SearchMemoryResponse:
        q = _tokens(query)
        scored: list[MemoryEntry] = []
        for text, author, sid in self._entries.get((app_name, user_id), []):
            overlap = q & _tokens(text)
            if overlap:
                scored.append(MemoryEntry(
                    text=text, author=author, session_id=sid,
                    score=len(overlap) / max(1, len(q))))
        scored.sort(key=lambda m: m.score, reverse=True)
        return SearchMemoryResponse(memories=scored)
