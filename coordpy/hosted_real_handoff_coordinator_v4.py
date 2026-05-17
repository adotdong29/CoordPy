"""W72 H6 — Hosted ↔ Real Substrate Handoff Coordinator V4.

The load-bearing W72 **rejoin-aware** Plane A↔B bridge. W71's V3
handoff coordinator added restart-aware promotion + delayed-repair
fallback on top of V2's budget-primary scoring. V4 adds:

* **Rejoin-aware promotion** — any request with
  ``rejoin_pressure >= rejoin_pressure_floor`` AND substrate trust
  ≥ floor is promoted to ``real_substrate_only`` with
  ``rejoin_alignment = 1.0``, regardless of the V3 restart-aware
  decision (and only when restart pressure is not yet promoting).
* **Delayed-rejoin-after-restart fallback** — a new decision label
  ``delayed_rejoin_after_restart_fallback`` fires when hosted is
  cheaper but the caller-declared
  ``restart_repair_trajectory_cid`` requires Plane B (rejoin-lag >
  rejoin-lag-floor AND substrate trust > floor) AND the request
  did not already promote to restart-aware or delayed-repair-
  fallback paths.
* **Rejoin-pressure falsifier** — extends V3 to a delayed-rejoin
  variant: rejects hosted-only envelopes that claim to satisfy a
  non-trivial restart-repair trajectory.

Honest scope (W72)
------------------

* The coordinator V4 does NOT pierce the hosted substrate boundary.
* ``W72-L-HANDOFF-V4-NOT-CROSSING-WALL-CAP`` — V4 preserves the
  W71 boundary as a content-addressed invariant.
* Determinism on (request V4 CID, registry CID, V17 substrate
  self-checksum CID, declared visible-token budget, restart
  pressure, rejoin pressure, restart-repair trajectory CID).
"""

from __future__ import annotations

import dataclasses
from typing import Any

from .hosted_real_handoff_coordinator import (
    W69_HANDOFF_DECISION_ABSTAIN,
    W69_HANDOFF_DECISION_HOSTED_ONLY,
    W69_HANDOFF_DECISION_HOSTED_WITH_REAL_AUDIT,
    W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY,
)
from .hosted_real_handoff_coordinator_v2 import (
    W70_HANDOFF_DECISION_BUDGET_PRIMARY_FALLBACK,
)
from .hosted_real_handoff_coordinator_v3 import (
    HandoffEnvelopeV3, HandoffRequestV3,
    HostedRealHandoffCoordinatorV3,
    W71_HANDOFF_DECISION_DELAYED_REPAIR_FALLBACK,
)
from .hosted_real_substrate_boundary_v5 import (
    HostedRealSubstrateBoundaryV5,
    build_default_hosted_real_substrate_boundary_v5,
)
from .hosted_router_controller import HostedRoutingDecision
from .tiny_substrate_v3 import _sha256_hex


W72_HOSTED_REAL_HANDOFF_V4_SCHEMA_VERSION: str = (
    "coordpy.hosted_real_handoff_coordinator_v4.v1")

W72_HANDOFF_DECISION_DELAYED_REJOIN_AFTER_RESTART_FALLBACK: str = (
    "delayed_rejoin_after_restart_fallback")
W72_HANDOFF_DECISIONS_V4: tuple[str, ...] = (
    W69_HANDOFF_DECISION_HOSTED_ONLY,
    W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY,
    W69_HANDOFF_DECISION_HOSTED_WITH_REAL_AUDIT,
    W69_HANDOFF_DECISION_ABSTAIN,
    W70_HANDOFF_DECISION_BUDGET_PRIMARY_FALLBACK,
    W71_HANDOFF_DECISION_DELAYED_REPAIR_FALLBACK,
    W72_HANDOFF_DECISION_DELAYED_REJOIN_AFTER_RESTART_FALLBACK,
)

W72_DEFAULT_REJOIN_PRESSURE_FLOOR: float = 0.5
W72_DEFAULT_REJOIN_LAG_FLOOR: int = 1
W72_DEFAULT_SUBSTRATE_TRUST_FLOOR: float = 0.5


@dataclasses.dataclass(frozen=True)
class HandoffRequestV4:
    inner_v3: HandoffRequestV3
    rejoin_pressure: float = 0.0
    restart_repair_trajectory_cid: str = ""
    rejoin_lag_turns: int = 0
    expected_substrate_trust_v4: float = 0.7

    @property
    def request_cid(self) -> str:
        return self.inner_v3.request_cid

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W72_HOSTED_REAL_HANDOFF_V4_SCHEMA_VERSION,
            "kind": "handoff_request_v4",
            "inner_v3_cid": str(self.inner_v3.cid()),
            "rejoin_pressure": float(round(
                self.rejoin_pressure, 12)),
            "restart_repair_trajectory_cid": str(
                self.restart_repair_trajectory_cid),
            "rejoin_lag_turns": int(self.rejoin_lag_turns),
            "expected_substrate_trust_v4": float(round(
                self.expected_substrate_trust_v4, 12)),
        })


@dataclasses.dataclass(frozen=True)
class HandoffEnvelopeV4:
    schema: str
    inner_v3: HandoffEnvelopeV3
    decision_v4: str
    plane_v4: str
    rejoin_alignment: float
    delayed_rejoin_alignment: float
    delayed_rejoin_fallback_active: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "inner_v3_cid": str(self.inner_v3.cid()),
            "decision_v4": str(self.decision_v4),
            "plane_v4": str(self.plane_v4),
            "rejoin_alignment": float(round(
                self.rejoin_alignment, 12)),
            "delayed_rejoin_alignment": float(round(
                self.delayed_rejoin_alignment, 12)),
            "delayed_rejoin_fallback_active": bool(
                self.delayed_rejoin_fallback_active),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_real_handoff_envelope_v4",
            "envelope": self.to_dict()})


@dataclasses.dataclass
class HostedRealHandoffCoordinatorV4:
    inner_v3: HostedRealHandoffCoordinatorV3 = dataclasses.field(
        default_factory=HostedRealHandoffCoordinatorV3)
    boundary_v5: HostedRealSubstrateBoundaryV5 = (
        dataclasses.field(
            default_factory=(
                build_default_hosted_real_substrate_boundary_v5)))
    rejoin_pressure_floor: float = (
        W72_DEFAULT_REJOIN_PRESSURE_FLOOR)
    rejoin_lag_floor: int = W72_DEFAULT_REJOIN_LAG_FLOOR
    substrate_trust_floor: float = (
        W72_DEFAULT_SUBSTRATE_TRUST_FLOOR)
    audit_v4: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W72_HOSTED_REAL_HANDOFF_V4_SCHEMA_VERSION,
            "kind": "hosted_real_handoff_coordinator_v4",
            "inner_v3_cid": str(self.inner_v3.cid()),
            "boundary_v5_cid": str(self.boundary_v5.cid()),
            "rejoin_pressure_floor": float(round(
                self.rejoin_pressure_floor, 12)),
            "rejoin_lag_floor": int(self.rejoin_lag_floor),
            "substrate_trust_floor": float(round(
                self.substrate_trust_floor, 12)),
        })

    def decide_v4(
            self, *, req_v4: HandoffRequestV4,
            hosted_decision: HostedRoutingDecision | None = None,
            substrate_self_checksum_cid: str = "",
    ) -> HandoffEnvelopeV4:
        v3_env = self.inner_v3.decide_v3(
            req_v3=req_v4.inner_v3,
            hosted_decision=hosted_decision,
            substrate_self_checksum_cid=str(
                substrate_self_checksum_cid))
        decision_v4 = str(v3_env.decision_v3)
        plane_v4 = str(v3_env.plane_v3)
        rejoin_alignment = 0.0
        delayed_rejoin_alignment = 0.0
        delayed_rejoin_fallback_active = False
        # Rejoin-aware promotion — highest priority *after* V3
        # restart-aware promotion. Only fires when V3 did not
        # already route to a Plane B path.
        if (decision_v4 == W69_HANDOFF_DECISION_HOSTED_ONLY
                and float(req_v4.rejoin_pressure)
                    >= float(self.rejoin_pressure_floor)
                and float(req_v4.expected_substrate_trust_v4)
                    >= float(self.substrate_trust_floor)):
            decision_v4 = (
                W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY)
            plane_v4 = "B"
            rejoin_alignment = 1.0
        # Delayed-rejoin fallback: when the restart-repair
        # trajectory has a non-trivial rejoin-lag AND substrate
        # trust is high, but V3 decided hosted-only AND rejoin
        # promotion did not already fire, V4 falls back.
        elif (
                str(req_v4.restart_repair_trajectory_cid)
                and int(req_v4.rejoin_lag_turns)
                    >= int(self.rejoin_lag_floor)
                and float(req_v4.expected_substrate_trust_v4)
                    >= float(self.substrate_trust_floor)
                and decision_v4
                    == W69_HANDOFF_DECISION_HOSTED_ONLY):
            decision_v4 = (
                W72_HANDOFF_DECISION_DELAYED_REJOIN_AFTER_RESTART_FALLBACK)
            plane_v4 = "A-rj-fallback"
            delayed_rejoin_fallback_active = True
            delayed_rejoin_alignment = 1.0
        # Honest alignment when no fallback fires.
        if (decision_v4
                in (W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY,
                    W69_HANDOFF_DECISION_HOSTED_WITH_REAL_AUDIT,
                    W72_HANDOFF_DECISION_DELAYED_REJOIN_AFTER_RESTART_FALLBACK)
                and float(req_v4.rejoin_pressure) > 0.0):
            rejoin_alignment = max(rejoin_alignment, 1.0)
        # Delayed-rejoin alignment: substrate / audit / rj-fallback
        # plane is the only plane that can honestly satisfy a
        # non-empty restart-repair trajectory.
        if str(req_v4.restart_repair_trajectory_cid):
            if decision_v4 in (
                    W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY,
                    W69_HANDOFF_DECISION_HOSTED_WITH_REAL_AUDIT,
                    W72_HANDOFF_DECISION_DELAYED_REJOIN_AFTER_RESTART_FALLBACK):
                delayed_rejoin_alignment = 1.0
            else:
                delayed_rejoin_alignment = 0.0
        env = HandoffEnvelopeV4(
            schema=W72_HOSTED_REAL_HANDOFF_V4_SCHEMA_VERSION,
            inner_v3=v3_env,
            decision_v4=str(decision_v4),
            plane_v4=str(plane_v4),
            rejoin_alignment=float(rejoin_alignment),
            delayed_rejoin_alignment=float(
                delayed_rejoin_alignment),
            delayed_rejoin_fallback_active=bool(
                delayed_rejoin_fallback_active),
        )
        self.audit_v4.append({
            "request_cid": str(req_v4.request_cid),
            "decision_v4": str(decision_v4),
            "plane_v4": str(plane_v4),
            "envelope_v4_cid": str(env.cid()),
            "rejoin_alignment": float(rejoin_alignment),
            "delayed_rejoin_alignment": float(
                delayed_rejoin_alignment),
        })
        return env


@dataclasses.dataclass(frozen=True)
class HostedRealHandoffV4RejoinFalsifier:
    schema: str
    envelope_v4_cid: str
    claim_kind: str
    claim_satisfied: bool
    falsifier_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "envelope_v4_cid": str(self.envelope_v4_cid),
            "claim_kind": str(self.claim_kind),
            "claim_satisfied": bool(self.claim_satisfied),
            "falsifier_score": float(round(
                self.falsifier_score, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_real_handoff_v4_rejoin_falsifier",
            "falsifier": self.to_dict()})


def probe_hosted_real_handoff_v4_delayed_rejoin_falsifier(
        *, envelope_v4: HandoffEnvelopeV4,
        claim_kind: str = "hosted_satisfies_delayed_rejoin",
        claim_satisfied: bool = False,
) -> HostedRealHandoffV4RejoinFalsifier:
    """Returns 0 on honest claims, 1 on dishonest.

    Dishonest = envelope says ``hosted_only`` AND caller asserts
    that hosted satisfied a non-trivial restart-repair trajectory
    (delayed rejoin after restart).
    """
    score = 0.0
    if (str(envelope_v4.decision_v4)
            == W69_HANDOFF_DECISION_HOSTED_ONLY
            and claim_kind == "hosted_satisfies_delayed_rejoin"
            and bool(claim_satisfied)):
        score = 1.0
    return HostedRealHandoffV4RejoinFalsifier(
        schema=W72_HOSTED_REAL_HANDOFF_V4_SCHEMA_VERSION,
        envelope_v4_cid=str(envelope_v4.cid()),
        claim_kind=str(claim_kind),
        claim_satisfied=bool(claim_satisfied),
        falsifier_score=float(score),
    )


def hosted_real_handoff_v4_rejoin_aware_savings(
        *, n_turns: int,
        hosted_only_tokens: int = 1000,
        hosted_with_real_audit_tokens: int = 400,
        real_substrate_only_tokens: int = 110,
        budget_primary_fallback_tokens: int = 70,
        delayed_repair_fallback_tokens: int = 60,
        delayed_rejoin_fallback_tokens: int = 55,
        real_substrate_fraction: float = 0.50,
        hosted_with_real_audit_fraction: float = 0.12,
        budget_primary_fallback_fraction: float = 0.08,
        delayed_repair_fallback_fraction: float = 0.12,
        delayed_rejoin_fallback_fraction: float = 0.15,
) -> dict[str, Any]:
    """V4 cross-plane handoff savings — rejoin-aware promotion plus
    delayed-rejoin-after-restart fallback drive total visible-token
    cost down vs forcing every turn through hosted_only.

    Defaults: 50 % real_substrate, 12 % hosted_with_real_audit,
    8 % budget_primary_fallback, 12 % delayed_repair_fallback,
    15 % delayed_rejoin_fallback, 3 % hosted_only.
    Saving ratio should be ≥ 78 % at default config.
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
    rj_f = float(max(
        0.0, min(1.0 - rs_f - ha_f - bp_f - dr_f,
                 delayed_rejoin_fallback_fraction)))
    ho_f = float(max(
        0.0, 1.0 - rs_f - ha_f - bp_f - dr_f - rj_f))
    n_rs = int(round(rs_f * n))
    n_ha = int(round(ha_f * n))
    n_bp = int(round(bp_f * n))
    n_dr = int(round(dr_f * n))
    n_rj = int(round(rj_f * n))
    n_ho = int(max(0, n - n_rs - n_ha - n_bp - n_dr - n_rj))
    total_handoff = int(
        int(real_substrate_only_tokens) * int(n_rs)
        + int(hosted_with_real_audit_tokens) * int(n_ha)
        + int(budget_primary_fallback_tokens) * int(n_bp)
        + int(delayed_repair_fallback_tokens) * int(n_dr)
        + int(delayed_rejoin_fallback_tokens) * int(n_rj)
        + int(hosted_only_tokens) * int(n_ho))
    total_all_hosted = int(int(hosted_only_tokens) * int(n))
    saving = int(total_all_hosted - total_handoff)
    ratio = (
        float(saving) / float(max(1, total_all_hosted))
        if total_all_hosted > 0 else 0.0)
    return {
        "schema": W72_HOSTED_REAL_HANDOFF_V4_SCHEMA_VERSION,
        "n_turns": int(n),
        "n_real_substrate_only": int(n_rs),
        "n_hosted_with_real_audit": int(n_ha),
        "n_budget_primary_fallback": int(n_bp),
        "n_delayed_repair_fallback": int(n_dr),
        "n_delayed_rejoin_fallback": int(n_rj),
        "n_hosted_only": int(n_ho),
        "total_handoff_tokens": int(total_handoff),
        "total_all_hosted_tokens": int(total_all_hosted),
        "saving_tokens": int(saving),
        "saving_ratio": float(round(ratio, 12)),
    }


@dataclasses.dataclass(frozen=True)
class HostedRealHandoffCoordinatorV4Witness:
    schema: str
    coordinator_v4_cid: str
    n_envelopes_v4: int
    n_hosted_only: int
    n_real_substrate_only: int
    n_hosted_with_real_audit: int
    n_abstain: int
    n_budget_primary_fallback: int
    n_delayed_repair_fallback: int
    n_delayed_rejoin_fallback: int
    inner_v3_witness_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "coordinator_v4_cid": str(self.coordinator_v4_cid),
            "n_envelopes_v4": int(self.n_envelopes_v4),
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
            "n_delayed_rejoin_fallback": int(
                self.n_delayed_rejoin_fallback),
            "inner_v3_witness_cid": str(
                self.inner_v3_witness_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind":
                "hosted_real_handoff_coordinator_v4_witness",
            "witness": self.to_dict()})


def emit_hosted_real_handoff_coordinator_v4_witness(
        coordinator: HostedRealHandoffCoordinatorV4,
) -> HostedRealHandoffCoordinatorV4Witness:
    from .hosted_real_handoff_coordinator_v3 import (
        emit_hosted_real_handoff_coordinator_v3_witness,
    )
    inner_w = emit_hosted_real_handoff_coordinator_v3_witness(
        coordinator.inner_v3)
    counts = {d: 0 for d in W72_HANDOFF_DECISIONS_V4}
    for entry in coordinator.audit_v4:
        d = str(entry.get("decision_v4", ""))
        if d in counts:
            counts[d] += 1
    return HostedRealHandoffCoordinatorV4Witness(
        schema=W72_HOSTED_REAL_HANDOFF_V4_SCHEMA_VERSION,
        coordinator_v4_cid=str(coordinator.cid()),
        n_envelopes_v4=int(len(coordinator.audit_v4)),
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
        n_delayed_rejoin_fallback=int(
            counts[
                W72_HANDOFF_DECISION_DELAYED_REJOIN_AFTER_RESTART_FALLBACK]),
        inner_v3_witness_cid=str(inner_w.cid()),
    )


__all__ = [
    "W72_HOSTED_REAL_HANDOFF_V4_SCHEMA_VERSION",
    "W72_HANDOFF_DECISION_DELAYED_REJOIN_AFTER_RESTART_FALLBACK",
    "W72_HANDOFF_DECISIONS_V4",
    "W72_DEFAULT_REJOIN_PRESSURE_FLOOR",
    "W72_DEFAULT_REJOIN_LAG_FLOOR",
    "W72_DEFAULT_SUBSTRATE_TRUST_FLOOR",
    "HandoffRequestV4",
    "HandoffEnvelopeV4",
    "HostedRealHandoffCoordinatorV4",
    "HostedRealHandoffV4RejoinFalsifier",
    "probe_hosted_real_handoff_v4_delayed_rejoin_falsifier",
    "hosted_real_handoff_v4_rejoin_aware_savings",
    "HostedRealHandoffCoordinatorV4Witness",
    "emit_hosted_real_handoff_coordinator_v4_witness",
]
