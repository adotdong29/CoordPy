"""coordpy.adk._capsule_trail — the audit/provenance layer under the ADK.

Every ADK ``Event`` the Runner emits is sealed into a typed,
content-addressed ``ContextCapsule`` in a hash-chained ``CapsuleLedger``.
The developer writes plain ADK code (Agent / Runner / Session); they get
CoordPy's distinctive guarantees for free:

* **content-addressed** — each capsule's CID is the SHA-256 of its
  ``(kind, payload, budget, parents)`` (deterministic; independent of
  wall-clock).
* **provenance** — declared parents form a chain; a retroactive insert
  breaks ``verify_chain_from_view_dict``.
* **replayable / auditable** — ``Runner.session_capsule_view()`` returns the
  sealed ``coordpy.capsule_view.v1`` chain you can re-verify from bytes alone.

This bridges the ADK surface (``coordpy.adk``) onto the stable capsule
primitives (``coordpy.capsule``) with no new capsule kinds.
"""

from __future__ import annotations

import json
from typing import Any

from coordpy.capsule import (
    CapsuleBudget, CapsuleKind, CapsuleLedger, ContextCapsule, render_view,
)

# Generous budget: the ADK trail is an audit log, not a compaction target.
_BUDGET = CapsuleBudget(max_tokens=4_000_000, max_bytes=16_000_000,
                        max_parents=64)

# Event-kind -> closed-vocabulary CapsuleKind.
_KIND_FOR: dict[str, str] = {
    "user": CapsuleKind.HANDOFF,
    "model": CapsuleKind.LLM_RESPONSE,
    "final": CapsuleKind.LLM_RESPONSE,
    "tool": CapsuleKind.HANDOFF,
    "transfer": CapsuleKind.TEAM_HANDOFF,
}


def _json_safe(obj: Any) -> Any:
    return json.loads(json.dumps(obj, default=str, sort_keys=True))


class CapsuleTrail:
    """Seals an ADK event stream into one hash-chained capsule ledger."""

    def __init__(self) -> None:
        self.ledger = CapsuleLedger()
        self._head: str | None = None
        self.n = 0
        self.root_cid: str | None = None

    def _seal(self, *, kind: str, payload: Any, n_tokens: int,
              metadata: dict[str, Any]) -> str:
        parents = (self._head,) if self._head else ()
        cap = ContextCapsule.new(
            kind=kind, payload=_json_safe(payload), budget=_BUDGET,
            parents=parents, n_tokens=max(1, n_tokens), metadata=metadata)
        sealed = self.ledger.admit_and_seal(cap)
        self._head = sealed.cid
        self.n += 1
        return sealed.cid

    def seal_event(self, event: Any) -> str:
        ek = event.event_kind()
        kind = _KIND_FOR.get(ek, CapsuleKind.HANDOFF)
        cid = self._seal(
            kind=kind, payload=event.capsule_payload(),
            n_tokens=len((event.text or "").split()),
            metadata={"author": event.author, "event_kind": ek})
        # Each saved artifact becomes its own content-addressed ARTIFACT capsule.
        for (fn, ver, data, mime) in getattr(event, "_artifact_blobs", []):
            self._seal(
                kind=CapsuleKind.ARTIFACT,
                payload={"name": fn, "version": ver, "n_bytes": len(data),
                         "mime_type": mime},
                n_tokens=1, metadata={"artifact": fn, "version": str(ver)})
        return cid

    def seal_run_report(self, summary: dict[str, Any]) -> str:
        cid = self._seal(kind=CapsuleKind.RUN_REPORT, payload=summary,
                         n_tokens=1, metadata={"role": "run_report"})
        self.root_cid = cid
        return cid

    def view(self) -> dict[str, Any]:
        return render_view(self.ledger, include_payload=True,
                           root_cid=self._head).as_dict()
