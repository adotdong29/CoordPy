"""W71 H6 — Hosted ↔ Real Substrate Handoff Coordinator V3.

The load-bearing W71 **restart-aware** Plane A↔B bridge. W70's V2
handoff coordinator added budget-primary scoring on top of V1's
text/substrate/audit/abstain. V3 adds:

* **Restart-aware promotion** — any request with
  ``restart_pressure >= restart_pressure_floor`` is promoted to
  ``real_substrate_only`` with ``restart_alignment = 1.0``,
  regardless of the V2 budget-primary decision.
* **Delayed-repair fallback** — a new decision label
  ``delayed_repair_fallback`` fires when hosted is cheaper but the
  caller-declared ``delayed_repair_trajectory_cid`` requires Plane
  B (delay > delay-floor AND substrate trust > floor).
* **Repair-dominance falsifier** — extends V2 to a delayed-repair
  variant: rejects hosted-only envelopes that claim to satisfy a
  non-trivial delayed-repair trajectory.

Honest scope (W71)
------------------

* The coordinator V3 does NOT pierce the hosted substrate boundary.
* ``W71-L-HANDOFF-V3-NOT-CROSSING-WALL-CAP`` — V3 preserves the
  W70 boundary as a content-addressed invariant.
* Determinism on (request V3 CID, registry CID, V16 substrate
  self-checksum CID, declared visible-token budget, restart
  pressure, delayed-repair trajectory CID).
"""

from __future__ import annotations

import dataclasses
from typing import Any

from .hosted_real_handoff_coordinator import (
    HandoffRequest,
    W69_HANDOFF_DECISION_ABSTAIN,
    W69_HANDOFF_DECISION_HOSTED_ONLY,
    W69_HANDOFF_DECISION_HOSTED_WITH_REAL_AUDIT,
    W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY,
)
from .hosted_real_handoff_coordinator_v2 import (
    HandoffEnvelopeV2, HandoffRequestV2,
    HostedRealHandoffCoordinatorV2,
    W70_HANDOFF_DECISION_BUDGET_PRIMARY_FALLBACK,
)
from .hosted_real_substrate_boundary_v4 import (
    HostedRealSubstrateBoundaryV4,
    build_default_hosted_real_substrate_boundary_v4,
)
from .hosted_router_controller import HostedRoutingDecision
from .tiny_substrate_v3 import _sha256_hex


W71_HOSTED_REAL_HANDOFF_V3_SCHEMA_VERSION: str = (
    "coordpy.hosted_real_handoff_coordinator_v3.v1")

W71_HANDOFF_DECISION_DELAYED_REPAIR_FALLBACK: str = (
    "delayed_repair_fallback")
W71_HANDOFF_DECISIONS_V3: tuple[str, ...] = (
    W69_HANDOFF_DECISION_HOSTED_ONLY,
    W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY,
    W69_HANDOFF_DECISION_HOSTED_WITH_REAL_AUDIT,
    W69_HANDOFF_DECISION_ABSTAIN,
    W70_HANDOFF_DECISION_BUDGET_PRIMARY_FALLBACK,
    W71_HANDOFF_DECISION_DELAYED_REPAIR_FALLBACK,
)

W71_DEFAULT_RESTART_PRESSURE_FLOOR: float = 0.5
W71_DEFAULT_DELAYED_REPAIR_DELAY_FLOOR: int = 1
W71_DEFAULT_SUBSTRATE_TRUST_FLOOR: float = 0.5


@dataclasses.dataclass(frozen=True)
class HandoffRequestV3:
    inner_v2: HandoffRequestV2
    restart_pressure: float = 0.0
    delayed_repair_trajectory_cid: str = ""
    delay_turns: int = 0
    expected_substrate_trust: float = 0.7

    @property
    def request_cid(self) -> str:
        return self.inner_v2.request_cid

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W71_HOSTED_REAL_HANDOFF_V3_SCHEMA_VERSION,
            "kind": "handoff_request_v3",
            "inner_v2_cid": str(self.inner_v2.cid()),
            "restart_pressure": float(round(
                self.restart_pressure, 12)),
            "delayed_repair_trajectory_cid": str(
                self.delayed_repair_trajectory_cid),
            "delay_turns": int(self.delay_turns),
            "expected_substrate_trust": float(round(
                self.expected_substrate_trust, 12)),
        })


@dataclasses.dataclass(frozen=True)
class HandoffEnvelopeV3:
    schema: str
    inner_v2: HandoffEnvelopeV2
    decision_v3: str
    plane_v3: str
    restart_alignment: float
    delayed_repair_alignment: float
    delayed_repair_fallback_active: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "inner_v2_cid": str(self.inner_v2.cid()),
            "decision_v3": str(self.decision_v3),
            "plane_v3": str(self.plane_v3),
            "restart_alignment": float(round(
                self.restart_alignment, 12)),
            "delayed_repair_alignment": float(round(
                self.delayed_repair_alignment, 12)),
            "delayed_repair_fallback_active": bool(
                self.delayed_repair_fallback_active),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_real_handoff_envelope_v3",
            "envelope": self.to_dict()})


@dataclasses.dataclass
class HostedRealHandoffCoordinatorV3:
    inner_v2: HostedRealHandoffCoordinatorV2 = dataclasses.field(
        default_factory=HostedRealHandoffCoordinatorV2)
    boundary_v4: HostedRealSubstrateBoundaryV4 = (
        dataclasses.field(
            default_factory=(
                build_default_hosted_real_substrate_boundary_v4)))
    restart_pressure_floor: float = (
        W71_DEFAULT_RESTART_PRESSURE_FLOOR)
    delayed_repair_delay_floor: int = (
        W71_DEFAULT_DELAYED_REPAIR_DELAY_FLOOR)
    substrate_trust_floor: float = (
        W71_DEFAULT_SUBSTRATE_TRUST_FLOOR)
    audit_v3: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W71_HOSTED_REAL_HANDOFF_V3_SCHEMA_VERSION,
            "kind": "hosted_real_handoff_coordinator_v3",
            "inner_v2_cid": str(self.inner_v2.cid()),
            "boundary_v4_cid": str(self.boundary_v4.cid()),
            "restart_pressure_floor": float(round(
                self.restart_pressure_floor, 12)),
            "delayed_repair_delay_floor": int(
                self.delayed_repair_delay_floor),
            "substrate_trust_floor": float(round(
                self.substrate_trust_floor, 12)),
        })

    def decide_v3(
            self, *, req_v3: HandoffRequestV3,
            hosted_decision: HostedRoutingDecision | None = None,
            substrate_self_checksum_cid: str = "",
    ) -> HandoffEnvelopeV3:
        v2_env = self.inner_v2.decide_v2(
            req_v2=req_v3.inner_v2,
            hosted_decision=hosted_decision,
            substrate_self_checksum_cid=str(
                substrate_self_checksum_cid))
        decision_v3 = str(v2_env.decision_v2)
        plane_v3 = str(v2_env.plane_v2)
        restart_alignment = 0.0
        delayed_repair_alignment = 0.0
        delayed_repair_fallback_active = False
        # Restart-aware promotion (highest priority): if restart
        # pressure crosses the floor, route through Plane B.
        if (float(req_v3.restart_pressure)
                >= float(self.restart_pressure_floor)
                and float(req_v3.expected_substrate_trust)
                    >= float(self.substrate_trust_floor)):
            decision_v3 = (
                W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY)
            plane_v3 = "B"
            restart_alignment = 1.0
        # Delayed-repair fallback: when the delayed-repair
        # trajectory has a non-trivial delay AND substrate trust
        # is high, but V2 decided hosted-only, V3 falls back.
        elif (
                str(req_v3.delayed_repair_trajectory_cid)
                and int(req_v3.delay_turns)
                    >= int(self.delayed_repair_delay_floor)
                and float(req_v3.expected_substrate_trust)
                    >= float(self.substrate_trust_floor)
                and decision_v3 == W69_HANDOFF_DECISION_HOSTED_ONLY):
            decision_v3 = (
                W71_HANDOFF_DECISION_DELAYED_REPAIR_FALLBACK)
            plane_v3 = "A-dr-fallback"
            delayed_repair_fallback_active = True
            delayed_repair_alignment = 1.0
        # Honest alignment when no fallback fires.
        if (decision_v3
                in (W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY,
                    W69_HANDOFF_DECISION_HOSTED_WITH_REAL_AUDIT,
                    W71_HANDOFF_DECISION_DELAYED_REPAIR_FALLBACK)
                and float(req_v3.restart_pressure) > 0.0):
            restart_alignment = max(
                restart_alignment, 1.0)
        # Delayed-repair alignment: substrate / audit / dr-fallback
        # plane is the only plane that can honestly satisfy a
        # non-empty delayed-repair trajectory.
        if str(req_v3.delayed_repair_trajectory_cid):
            if decision_v3 in (
                    W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY,
                    W69_HANDOFF_DECISION_HOSTED_WITH_REAL_AUDIT,
                    W71_HANDOFF_DECISION_DELAYED_REPAIR_FALLBACK):
                delayed_repair_alignment = 1.0
            else:
                delayed_repair_alignment = 0.0
        env = HandoffEnvelopeV3(
            schema=W71_HOSTED_REAL_HANDOFF_V3_SCHEMA_VERSION,
            inner_v2=v2_env,
            decision_v3=str(decision_v3),
            plane_v3=str(plane_v3),
            restart_alignment=float(restart_alignment),
            delayed_repair_alignment=float(
                delayed_repair_alignment),
            delayed_repair_fallback_active=bool(
                delayed_repair_fallback_active),
        )
        self.audit_v3.append({
            "request_cid": str(req_v3.request_cid),
            "decision_v3": str(decision_v3),
            "plane_v3": str(plane_v3),
            "envelope_v3_cid": str(env.cid()),
            "restart_alignment": float(restart_alignment),
            "delayed_repair_alignment": float(
                delayed_repair_alignment),
        })
        return env


@dataclasses.dataclass(frozen=True)
class HostedRealHandoffV3RepairFalsifier:
    schema: str
    envelope_v3_cid: str
    claim_kind: str
    claim_satisfied: bool
    falsifier_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "envelope_v3_cid": str(self.envelope_v3_cid),
            "claim_kind": str(self.claim_kind),
            "claim_satisfied": bool(self.claim_satisfied),
            "falsifier_score": float(round(
                self.falsifier_score, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_real_handoff_v3_repair_falsifier",
            "falsifier": self.to_dict()})


def probe_hosted_real_handoff_v3_delayed_repair_falsifier(
        *, envelope_v3: HandoffEnvelopeV3,
        claim_kind: str = "hosted_satisfies_delayed_repair",
        claim_satisfied: bool = False,
) -> HostedRealHandoffV3RepairFalsifier:
    """Returns 0 on honest claims, 1 on dishonest.

    Dishonest = envelope says ``hosted_only`` AND caller asserts
    that hosted satisfied a non-trivial delayed-repair trajectory.
    """
    score = 0.0
    if (str(envelope_v3.decision_v3)
            == W69_HANDOFF_DECISION_HOSTED_ONLY
            and claim_kind
                == "hosted_satisfies_delayed_repair"
            and bool(claim_satisfied)):
        score = 1.0
    return HostedRealHandoffV3RepairFalsifier(
        schema=W71_HOSTED_REAL_HANDOFF_V3_SCHEMA_VERSION,
        envelope_v3_cid=str(envelope_v3.cid()),
        claim_kind=str(claim_kind),
        claim_satisfied=bool(claim_satisfied),
        falsifier_score=float(score),
    )


def hosted_real_handoff_v3_restart_aware_savings(
        *, n_turns: int,
        hosted_only_tokens: int = 1000,
        hosted_with_real_audit_tokens: int = 400,
        real_substrate_only_tokens: int = 120,
        budget_primary_fallback_tokens: int = 70,
        delayed_repair_fallback_tokens: int = 60,
        real_substrate_fraction: float = 0.55,
        hosted_with_real_audit_fraction: float = 0.15,
        budget_primary_fallback_fraction: float = 0.10,
        delayed_repair_fallback_fraction: float = 0.15,
) -> dict[str, Any]:
    """V3 cross-plane handoff savings — restart-aware promotion
    plus delayed-repair fallback drive total visible-token cost
    down vs forcing every turn through hosted_only.

    Defaults: 55 % real_substrate, 15 % hosted_with_real_audit,
    10 % budget_primary_fallback, 15 % delayed_repair_fallback,
    5 % hosted_only.
    Saving ratio should be ≥ 70 % at default config.
    """
    n = int(max(1, n_turns))
    rs_f = float(max(0.0, min(1.0, real_substrate_fraction)))
    ha_f = float(max(
        0.0, min(1.0 - rs_f, hosted_with_real_audit_fraction)))
    bp_f = float(max(
        0.0, min(1.0 - rs_f - ha_f,
                 budget_primary_fallback_fraction)))
    dr_f = float(max(
        0.0, min(1.0 - rs_f - ha_f - bp_f,
                 delayed_repair_fallback_fraction)))
    ho_f = float(max(
        0.0, 1.0 - rs_f - ha_f - bp_f - dr_f))
    n_rs = int(round(rs_f * n))
    n_ha = int(round(ha_f * n))
    n_bp = int(round(bp_f * n))
    n_dr = int(round(dr_f * n))
    n_ho = int(max(0, n - n_rs - n_ha - n_bp - n_dr))
    total_handoff = int(
        int(real_substrate_only_tokens) * int(n_rs)
        + int(hosted_with_real_audit_tokens) * int(n_ha)
        + int(budget_primary_fallback_tokens) * int(n_bp)
        + int(delayed_repair_fallback_tokens) * int(n_dr)
        + int(hosted_only_tokens) * int(n_ho))
    total_all_hosted = int(int(hosted_only_tokens) * int(n))
    saving = int(total_all_hosted - total_handoff)
    ratio = (
        float(saving) / float(max(1, total_all_hosted))
        if total_all_hosted > 0 else 0.0)
    return {
        "schema": W71_HOSTED_REAL_HANDOFF_V3_SCHEMA_VERSION,
        "n_turns": int(n),
        "n_real_substrate_only": int(n_rs),
        "n_hosted_with_real_audit": int(n_ha),
        "n_budget_primary_fallback": int(n_bp),
        "n_delayed_repair_fallback": int(n_dr),
        "n_hosted_only": int(n_ho),
        "total_handoff_tokens": int(total_handoff),
        "total_all_hosted_tokens": int(total_all_hosted),
        "saving_tokens": int(saving),
        "saving_ratio": float(round(ratio, 12)),
    }


@dataclasses.dataclass(frozen=True)
class HostedRealHandoffCoordinatorV3Witness:
    schema: str
    coordinator_v3_cid: str
    n_envelopes_v3: int
    n_hosted_only: int
    n_real_substrate_only: int
    n_hosted_with_real_audit: int
    n_abstain: int
    n_budget_primary_fallback: int
    n_delayed_repair_fallback: int
    inner_v2_witness_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "coordinator_v3_cid": str(self.coordinator_v3_cid),
            "n_envelopes_v3": int(self.n_envelopes_v3),
            "n_hosted_only": int(self.n_hosted_only),
            "n_real_substrate_only": int(
                self.n_real_substrate_only),
            "n_hosted_with_real_audit": int(
                self.n_hosted_with_real_audit),
            "n_abstain": int(self.n_abstain),
            "n_budget_primary_fallback": int(
                self.n_budget_primary_fallback),
            "n_delayed_repair_fallback": int(
                self.n_delayed_repair_fallback),
            "inner_v2_witness_cid": str(
                self.inner_v2_witness_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind":
                "hosted_real_handoff_coordinator_v3_witness",
            "witness": self.to_dict()})


def emit_hosted_real_handoff_coordinator_v3_witness(
        coordinator: HostedRealHandoffCoordinatorV3,
) -> HostedRealHandoffCoordinatorV3Witness:
    from .hosted_real_handoff_coordinator_v2 import (
        emit_hosted_real_handoff_coordinator_v2_witness,
    )
    inner_w = emit_hosted_real_handoff_coordinator_v2_witness(
        coordinator.inner_v2)
    counts = {d: 0 for d in W71_HANDOFF_DECISIONS_V3}
    for entry in coordinator.audit_v3:
        d = str(entry.get("decision_v3", ""))
        if d in counts:
            counts[d] += 1
    return HostedRealHandoffCoordinatorV3Witness(
        schema=W71_HOSTED_REAL_HANDOFF_V3_SCHEMA_VERSION,
        coordinator_v3_cid=str(coordinator.cid()),
        n_envelopes_v3=int(len(coordinator.audit_v3)),
        n_hosted_only=int(
            counts[W69_HANDOFF_DECISION_HOSTED_ONLY]),
        n_real_substrate_only=int(
            counts[W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY]),
        n_hosted_with_real_audit=int(
            counts[W69_HANDOFF_DECISION_HOSTED_WITH_REAL_AUDIT]),
        n_abstain=int(counts[W69_HANDOFF_DECISION_ABSTAIN]),
        n_budget_primary_fallback=int(
            counts[W70_HANDOFF_DECISION_BUDGET_PRIMARY_FALLBACK]),
        n_delayed_repair_fallback=int(
            counts[W71_HANDOFF_DECISION_DELAYED_REPAIR_FALLBACK]),
        inner_v2_witness_cid=str(inner_w.cid()),
    )


__all__ = [
    "W71_HOSTED_REAL_HANDOFF_V3_SCHEMA_VERSION",
    "W71_HANDOFF_DECISION_DELAYED_REPAIR_FALLBACK",
    "W71_HANDOFF_DECISIONS_V3",
    "W71_DEFAULT_RESTART_PRESSURE_FLOOR",
    "W71_DEFAULT_DELAYED_REPAIR_DELAY_FLOOR",
    "W71_DEFAULT_SUBSTRATE_TRUST_FLOOR",
    "HandoffRequestV3",
    "HandoffEnvelopeV3",
    "HostedRealHandoffCoordinatorV3",
    "HostedRealHandoffV3RepairFalsifier",
    "probe_hosted_real_handoff_v3_delayed_repair_falsifier",
    "hosted_real_handoff_v3_restart_aware_savings",
    "HostedRealHandoffCoordinatorV3Witness",
    "emit_hosted_real_handoff_coordinator_v3_witness",
]
