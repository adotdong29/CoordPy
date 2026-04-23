"""Cross-domain validation adapters for the Wevra capsule system.

Each adapter maps domain event types to existing CapsuleKind values
(HANDLE, SWEEP_CELL, READINESS_CHECK, PROFILE) so that CapsuleCategory
can perform meaningful Kan extension and naturality checks.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from vision_mvp.wevra.capsule import (
    CapsuleBudget, CapsuleKind, CapsuleLedger, ContextCapsule,
)

_BUDGET = CapsuleBudget(max_bytes=4096, max_parents=8)


@dataclass
class DomainEvent:
    domain: str
    event_type: str
    data: dict


class DomainAdapter:
    DOMAIN_NAME: str = ""
    _KIND_MAP: dict[str, str] = {}
    _ROLE_SUPPORT: dict[str, list[str]] = {}

    @classmethod
    def event_types(cls) -> list[str]:
        return list(cls._KIND_MAP.keys())

    @classmethod
    def kind_for(cls, event_type: str) -> str:
        return cls._KIND_MAP[event_type]

    @classmethod
    def role_support(cls) -> dict[str, list[str]]:
        return dict(cls._ROLE_SUPPORT)

    @classmethod
    def event_type_id(cls, event_type: str) -> int:
        return cls.event_types().index(event_type)

    @classmethod
    def to_capsule(cls, event_type: str, data: dict,
                   parents: tuple[str, ...] = ()) -> ContextCapsule:
        return ContextCapsule.new(
            kind=cls.kind_for(event_type),
            payload={"domain": cls.DOMAIN_NAME, "event_type": event_type, "data": data},
            budget=_BUDGET,
            parents=parents,
        )

    @classmethod
    def generate_trace(cls, n_events: int = 100, seed: int = 0) -> list[ContextCapsule]:
        rng = random.Random(seed)
        ledger = CapsuleLedger()
        trace: list[ContextCapsule] = []
        etypes = cls.event_types()
        for i in range(n_events):
            et = rng.choice(etypes)
            pool = [c.cid for c in ledger.all_capsules()]
            k = min(rng.randint(0, 2), len(pool))
            parents = tuple(rng.sample(pool, k)) if k else ()
            cap = cls.to_capsule(et, {"step": i, "v": rng.randint(0, 999)}, parents=parents)
            trace.append(ledger.admit_and_seal(cap))
        return trace

    @classmethod
    def labels_for_role(cls, trace: list[ContextCapsule], role: str) -> list[int]:
        supported = set(cls._ROLE_SUPPORT.get(role, []))
        return [1 if c.kind in supported else 0 for c in trace]


class RoboticsDomainAdapter(DomainAdapter):
    """Sensor-fusion -> motion-planner -> executor.

    Kind mapping (reusing existing CapsuleKind vocabulary):
      SENSOR_READING    -> HANDLE
      OBSTACLE_DETECTED -> SWEEP_CELL
      WAYPOINT_REACHED  -> READINESS_CHECK
      ACTION_CMD        -> PROFILE

    Theorem KAN-1: right_kan_extension(trace, 'executor') returns exactly
    {SWEEP_CELL, READINESS_CHECK, PROFILE} -- minimal covering set.
    """
    DOMAIN_NAME = "robotics"
    _KIND_MAP: dict[str, str] = {
        "SENSOR_READING":    CapsuleKind.HANDLE,
        "OBSTACLE_DETECTED": CapsuleKind.SWEEP_CELL,
        "WAYPOINT_REACHED":  CapsuleKind.READINESS_CHECK,
        "ACTION_CMD":        CapsuleKind.PROFILE,
    }
    _ROLE_SUPPORT: dict[str, list[str]] = {
        "sensor_fusion":  [CapsuleKind.HANDLE],
        "motion_planner": [CapsuleKind.HANDLE, CapsuleKind.SWEEP_CELL, CapsuleKind.READINESS_CHECK],
        "executor":       [CapsuleKind.SWEEP_CELL, CapsuleKind.READINESS_CHECK, CapsuleKind.PROFILE],
    }


class NLPDomainAdapter(DomainAdapter):
    """Tokeniser -> encoder -> decoder.

    Kind mapping:
      RAW_TEXT   -> HANDLE
      TOKEN_IDS  -> SWEEP_CELL
      EMBEDDING  -> READINESS_CHECK
      LOGITS     -> PROFILE

    Naturality: support-restriction handoff commutes with role-morphism restriction.
    """
    DOMAIN_NAME = "nlp"
    _KIND_MAP: dict[str, str] = {
        "RAW_TEXT":  CapsuleKind.HANDLE,
        "TOKEN_IDS": CapsuleKind.SWEEP_CELL,
        "EMBEDDING": CapsuleKind.READINESS_CHECK,
        "LOGITS":    CapsuleKind.PROFILE,
    }
    _ROLE_SUPPORT: dict[str, list[str]] = {
        "tokenizer": [CapsuleKind.HANDLE],
        "encoder":   [CapsuleKind.HANDLE, CapsuleKind.SWEEP_CELL],
        "decoder":   [CapsuleKind.READINESS_CHECK, CapsuleKind.PROFILE],
    }


class PlanningDomainAdapter(DomainAdapter):
    """Goal-setter -> planner -> verifier.

    Kind mapping:
      GOAL_STATE          -> PROFILE
      PLAN_STEP           -> SWEEP_CELL
      STATE_UPDATE        -> HANDLE
      VERIFICATION_RESULT -> READINESS_CHECK

    Theorem OPERAD-1: any bracketing of (goal_setter, planner, verifier)
    produces the same root capsule CID.
    """
    DOMAIN_NAME = "planning"
    _KIND_MAP: dict[str, str] = {
        "GOAL_STATE":          CapsuleKind.PROFILE,
        "PLAN_STEP":           CapsuleKind.SWEEP_CELL,
        "STATE_UPDATE":        CapsuleKind.HANDLE,
        "VERIFICATION_RESULT": CapsuleKind.READINESS_CHECK,
    }
    _ROLE_SUPPORT: dict[str, list[str]] = {
        "goal_setter": [CapsuleKind.PROFILE],
        "planner":     [CapsuleKind.PROFILE, CapsuleKind.HANDLE, CapsuleKind.SWEEP_CELL],
        "verifier":    [CapsuleKind.SWEEP_CELL, CapsuleKind.HANDLE, CapsuleKind.READINESS_CHECK],
    }


ADAPTERS: dict[str, type[DomainAdapter]] = {
    "robotics": RoboticsDomainAdapter,
    "nlp": NLPDomainAdapter,
    "planning": PlanningDomainAdapter,
}


def get_adapter(name: str) -> type[DomainAdapter]:
    return ADAPTERS[name]


__all__ = [
    "DomainEvent", "DomainAdapter",
    "RoboticsDomainAdapter", "NLPDomainAdapter", "PlanningDomainAdapter",
    "ADAPTERS", "get_adapter",
]
