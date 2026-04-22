"""Trigger abstraction — coordination-surface-specific disagreement detection.

Phase 14 introduced an event-triggered control pattern (Tabuada/Heemels): only
invoke the LLM for round-2 refinement when the agent's draft genuinely
disagrees with the routed bulletin. Phase 14 wired that to a dict-key Jaccard
detector (`core/event_trigger.py`); Phase 17 added a behavior-probe variant
(`core/behavior_trigger.py`) for numerical-convention coordination.

Phase 17 made the deployment tax visible: the routing layer transferred
across tasks, but the trigger layer didn't. Each new coordination surface
required engineering its own disagreement signal. This module turns that
implicit boundary into an explicit interface so the harnesses can depend on
ONE abstraction instead of importing task-specific trigger functions.

Vocabulary:
    Trigger ............... a `should_refine(own, bulletin, threshold) -> TriggerDecision`
                            callable that quantifies disagreement on the
                            coordination surface it knows about.
    TriggerDecision ....... uniform return shape: refine (bool), score (float
                            in [0, 1]), threshold (echo), info (per-trigger
                            extras: dict-key sets, raw judge response, etc.).

The two existing trigger modules keep their free-function APIs unchanged
(`event_trigger.should_refine`, `behavior_trigger.should_refine`). Adapters
wrap them as `Trigger` objects so harnesses can swap in any implementation —
including the general LLM-judge / hybrid-heuristic trigger added in Phase 18.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable


# ---------------- Decision shape ------------------------------------------

@dataclass
class TriggerDecision:
    """Canonical trigger output. Existing per-task TriggerDecision dataclasses
    in `event_trigger.py` and `behavior_trigger.py` happen to expose the same
    `refine` / `score` / `threshold` fields, so adapters can copy through."""

    refine: bool
    score: float
    threshold: float
    info: dict[str, Any] = field(default_factory=dict)


# ---------------- Protocol -------------------------------------------------

@runtime_checkable
class Trigger(Protocol):
    """Anything callable as `should_refine(own, bulletin, threshold)`."""

    name: str

    def should_refine(
        self,
        own_draft: str,
        bulletin_drafts: list[str],
        threshold: float = 0.34,
    ) -> TriggerDecision: ...


# ---------------- Adapters for the two existing triggers ------------------

class CallableTrigger:
    """Wrap any `should_refine`-shaped free function as a `Trigger` object.

    The wrapped function may return either the canonical `TriggerDecision`
    or a per-task dataclass that exposes `.refine`, `.score`, `.threshold`
    (the existing two trigger modules both fit). Extra fields are surfaced
    under `info` for downstream observability.
    """

    def __init__(self, name: str, fn: Callable[..., Any]):
        self.name = name
        self._fn = fn

    def should_refine(
        self,
        own_draft: str,
        bulletin_drafts: list[str],
        threshold: float = 0.34,
    ) -> TriggerDecision:
        raw = self._fn(own_draft, bulletin_drafts, threshold=threshold)
        return _coerce_decision(raw, threshold)


def _coerce_decision(raw: Any, threshold: float) -> TriggerDecision:
    """Normalize per-task TriggerDecision-likes into the canonical type."""
    if isinstance(raw, TriggerDecision):
        return raw
    refine = bool(getattr(raw, "refine", False))
    score = float(getattr(raw, "score", 0.0))
    th = float(getattr(raw, "threshold", threshold))
    info: dict[str, Any] = {}
    for extra in ("own_keys", "bulletin_keys",
                  "own_functions", "bulletin_functions",
                  "raw_response", "components"):
        v = getattr(raw, extra, None)
        if v is not None:
            info[extra] = v
    return TriggerDecision(refine=refine, score=score, threshold=th, info=info)


# ---------------- Built-in factories --------------------------------------

def schema_key_trigger() -> Trigger:
    """Phase-14 trigger: dict-key Jaccard, ProtocolKit-style schema drift."""
    from .event_trigger import should_refine
    return CallableTrigger("schema-key-jaccard", should_refine)


def behavior_probe_trigger() -> Trigger:
    """Phase-17 trigger: per-pair behavior probes, NumericLedger-style
    numerical-convention drift."""
    from .behavior_trigger import should_refine
    return CallableTrigger("behavior-probe", should_refine)


# Lazy factories for the Phase-18 general triggers. Defined here so the
# registry is self-contained and test ordering doesn't matter. The actual
# classes live in general_trigger.py; the import is deferred to call time
# to avoid a circular import at module load.
def _hybrid_structural_factory() -> "Trigger":
    from .general_trigger import HybridStructuralTrigger
    return HybridStructuralTrigger()


def _general_heuristic_factory() -> "Trigger":
    from .general_trigger import GeneralTrigger
    return GeneralTrigger(client=None)


# Registry — keeps `phase18_general_trigger.py` declarative.
_REGISTRY: dict[str, Callable[[], Trigger]] = {
    "schema-key": schema_key_trigger,
    "behavior-probe": behavior_probe_trigger,
    "hybrid-structural": _hybrid_structural_factory,
    "general-heuristic": _general_heuristic_factory,
}


def register_trigger(name: str, factory: Callable[[], Trigger]) -> None:
    """Add a trigger factory to the registry. Used by `general_trigger.py`
    so phase18 can pick triggers by name without import gymnastics."""
    _REGISTRY[name] = factory


def get_trigger(name: str) -> Trigger:
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown trigger {name!r}. Registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]()


def list_triggers() -> list[str]:
    return sorted(_REGISTRY)


def get_default_trigger() -> Trigger:
    """Return the recommended default trigger for new coordination surfaces.

    Use hybrid-structural as the starting point when deploying CASR to a surface
    that does not yet have a bespoke trigger. It covers schema-naming drift
    (dict keys, string literals) and numerical-convention drift (OOM buckets,
    shared-function fuzz disagreement) with zero task-specific code.

    Fall back to per-task engineering only when:
    - fire rate is consistently 0 on your surface (no AST signal to detect), or
    - fire rate is consistently n_higher_tier (trigger can't distinguish agreement
      from disagreement — every agent always refines).

    Phase 18 benchmark: hybrid-structural matched or outperformed the per-task
    triggers on both ProtocolKit (schema drift) and NumericLedger (numeric
    conventions) under a paired A/B comparison with shared round-1 drafts.
    """
    from .general_trigger import HybridStructuralTrigger
    return HybridStructuralTrigger()
