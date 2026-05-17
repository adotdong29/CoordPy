"""W74 H6 — Hosted ↔ Real Substrate Handoff Coordinator V6.

The load-bearing W74 **compound-aware** Plane A↔B bridge. W73's V5
handoff coordinator added replacement-aware promotion + replacement-
after-CTR fallback on top of V4's rejoin-aware scoring. V6 adds:

* **Compound-aware promotion** — any request with
  ``compound_pressure >= compound_pressure_floor`` AND substrate
  trust ≥ floor is promoted to ``real_substrate_only`` with
  ``compound_alignment = 1.0``, regardless of the V5 replacement-
  aware decision (and only when other pressures are not yet
  promoting).
* **Compound-repair-after-DRTR fallback** — a new decision label
  ``compound_repair_after_delayed_repair_then_replacement_fallback``
  fires when hosted is cheaper but the caller-declared
  ``compound_repair_trajectory_cid`` requires Plane B (compound-
  window > compound-window-floor AND substrate trust > floor) AND
  the request did not already promote to restart-aware, delayed-
  repair-fallback, delayed-rejoin-fallback, or replacement-after-
  CTR-fallback paths.
* **Compound-pressure falsifier** — extends V5 to a compound-
  repair-after-DRTR variant: rejects hosted-only envelopes that
  claim to satisfy a non-trivial compound-repair trajectory.

Honest scope (W74)
------------------

* The coordinator V6 does NOT pierce the hosted substrate
  boundary.
* ``W74-L-HANDOFF-V6-NOT-CROSSING-WALL-CAP`` — V6 preserves the
  W73 boundary as a content-addressed invariant.
* Determinism on (request V6 CID, registry CID, V19 substrate
  self-checksum CID, declared visible-token budget, restart
  pressure, rejoin pressure, replacement pressure, compound
  pressure, compound-repair trajectory CID).
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
    W71_HANDOFF_DECISION_DELAYED_REPAIR_FALLBACK,
)
from .hosted_real_handoff_coordinator_v4 import (
    W72_HANDOFF_DECISION_DELAYED_REJOIN_AFTER_RESTART_FALLBACK,
)
from .hosted_real_handoff_coordinator_v5 import (
    HandoffEnvelopeV5, HandoffRequestV5,
    HostedRealHandoffCoordinatorV5,
    W73_HANDOFF_DECISION_REPLACEMENT_AFTER_CTR_FALLBACK,
)
from .hosted_real_substrate_boundary_v7 import (
    HostedRealSubstrateBoundaryV7,
    build_default_hosted_real_substrate_boundary_v7,
)
from .hosted_router_controller import HostedRoutingDecision
from .tiny_substrate_v3 import _sha256_hex


W74_HOSTED_REAL_HANDOFF_V6_SCHEMA_VERSION: str = (
    "coordpy.hosted_real_handoff_coordinator_v6.v1")

W74_HANDOFF_DECISION_COMPOUND_REPAIR_FALLBACK: str = (
    "compound_repair_after_delayed_repair_then_replacement_fallback")
W74_HANDOFF_DECISIONS_V6: tuple[str, ...] = (
    W69_HANDOFF_DECISION_HOSTED_ONLY,
    W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY,
    W69_HANDOFF_DECISION_HOSTED_WITH_REAL_AUDIT,
    W69_HANDOFF_DECISION_ABSTAIN,
    W70_HANDOFF_DECISION_BUDGET_PRIMARY_FALLBACK,
    W71_HANDOFF_DECISION_DELAYED_REPAIR_FALLBACK,
    W72_HANDOFF_DECISION_DELAYED_REJOIN_AFTER_RESTART_FALLBACK,
    W73_HANDOFF_DECISION_REPLACEMENT_AFTER_CTR_FALLBACK,
    W74_HANDOFF_DECISION_COMPOUND_REPAIR_FALLBACK,
)

W74_DEFAULT_COMPOUND_PRESSURE_FLOOR: float = 0.5
W74_DEFAULT_COMPOUND_WINDOW_FLOOR: int = 1
W74_DEFAULT_SUBSTRATE_TRUST_FLOOR: float = 0.5


@dataclasses.dataclass(frozen=True)
class HandoffRequestV6:
    inner_v5: HandoffRequestV5
    compound_pressure: float = 0.0
    compound_repair_trajectory_cid: str = ""
    compound_window_turns: int = 0
    expected_substrate_trust_v6: float = 0.7

    @property
    def request_cid(self) -> str:
        return self.inner_v5.request_cid

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W74_HOSTED_REAL_HANDOFF_V6_SCHEMA_VERSION,
            "kind": "handoff_request_v6",
            "inner_v5_cid": str(self.inner_v5.cid()),
            "compound_pressure": float(round(
                self.compound_pressure, 12)),
            "compound_repair_trajectory_cid": str(
                self.compound_repair_trajectory_cid),
            "compound_window_turns": int(
                self.compound_window_turns),
            "expected_substrate_trust_v6": float(round(
                self.expected_substrate_trust_v6, 12)),
        })


@dataclasses.dataclass(frozen=True)
class HandoffEnvelopeV6:
    schema: str
    inner_v5: HandoffEnvelopeV5
    decision_v6: str
    plane_v6: str
    compound_alignment: float
    compound_repair_drtr_alignment: float
    compound_repair_drtr_fallback_active: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "inner_v5_cid": str(self.inner_v5.cid()),
            "decision_v6": str(self.decision_v6),
            "plane_v6": str(self.plane_v6),
            "compound_alignment": float(round(
                self.compound_alignment, 12)),
            "compound_repair_drtr_alignment": float(round(
                self.compound_repair_drtr_alignment, 12)),
            "compound_repair_drtr_fallback_active": bool(
                self.compound_repair_drtr_fallback_active),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_real_handoff_envelope_v6",
            "envelope": self.to_dict()})


@dataclasses.dataclass
class HostedRealHandoffCoordinatorV6:
    inner_v5: HostedRealHandoffCoordinatorV5 = dataclasses.field(
        default_factory=HostedRealHandoffCoordinatorV5)
    boundary_v7: HostedRealSubstrateBoundaryV7 = (
        dataclasses.field(
            default_factory=(
                build_default_hosted_real_substrate_boundary_v7)))
    compound_pressure_floor: float = (
        W74_DEFAULT_COMPOUND_PRESSURE_FLOOR)
    compound_window_floor: int = W74_DEFAULT_COMPOUND_WINDOW_FLOOR
    substrate_trust_floor: float = (
        W74_DEFAULT_SUBSTRATE_TRUST_FLOOR)
    audit_v6: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W74_HOSTED_REAL_HANDOFF_V6_SCHEMA_VERSION,
            "kind": "hosted_real_handoff_coordinator_v6",
            "inner_v5_cid": str(self.inner_v5.cid()),
            "boundary_v7_cid": str(self.boundary_v7.cid()),
            "compound_pressure_floor": float(round(
                self.compound_pressure_floor, 12)),
            "compound_window_floor": int(
                self.compound_window_floor),
            "substrate_trust_floor": float(round(
                self.substrate_trust_floor, 12)),
        })

    def decide_v6(
            self, *, req_v6: HandoffRequestV6,
            hosted_decision: HostedRoutingDecision | None = None,
            substrate_self_checksum_cid: str = "",
    ) -> HandoffEnvelopeV6:
        v5_env = self.inner_v5.decide_v5(
            req_v5=req_v6.inner_v5,
            hosted_decision=hosted_decision,
            substrate_self_checksum_cid=str(
                substrate_self_checksum_cid))
        decision_v6 = str(v5_env.decision_v5)
        plane_v6 = str(v5_env.plane_v5)
        compound_alignment = 0.0
        compound_repair_drtr_alignment = 0.0
        compound_repair_drtr_fallback_active = False
        # Compound-aware promotion — fires when V5 stayed on
        # hosted_only AND compound pressure is above the floor.
        if (decision_v6 == W69_HANDOFF_DECISION_HOSTED_ONLY
                and float(req_v6.compound_pressure)
                    >= float(self.compound_pressure_floor)
                and float(req_v6.expected_substrate_trust_v6)
                    >= float(self.substrate_trust_floor)):
            decision_v6 = (
                W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY)
            plane_v6 = "B"
            compound_alignment = 1.0
        # Compound-repair-after-DRTR fallback — fires when the
        # compound-repair trajectory has a non-trivial window AND
        # substrate trust is high, AND no other V5 fallback fired.
        elif (
                str(req_v6.compound_repair_trajectory_cid)
                and int(req_v6.compound_window_turns)
                    >= int(self.compound_window_floor)
                and float(req_v6.expected_substrate_trust_v6)
                    >= float(self.substrate_trust_floor)
                and decision_v6
                    == W69_HANDOFF_DECISION_HOSTED_ONLY):
            decision_v6 = (
                W74_HANDOFF_DECISION_COMPOUND_REPAIR_FALLBACK)
            plane_v6 = "A-cmp-fallback"
            compound_repair_drtr_fallback_active = True
            compound_repair_drtr_alignment = 1.0
        # Honest alignment when no fallback fires.
        if (decision_v6
                in (W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY,
                    W69_HANDOFF_DECISION_HOSTED_WITH_REAL_AUDIT,
                    W74_HANDOFF_DECISION_COMPOUND_REPAIR_FALLBACK)
                and float(req_v6.compound_pressure) > 0.0):
            compound_alignment = max(
                compound_alignment, 1.0)
        # Compound-repair-after-DRTR alignment: substrate / audit /
        # cmp-fallback plane is the only plane that can honestly
        # satisfy a non-empty compound-repair trajectory.
        if str(req_v6.compound_repair_trajectory_cid):
            if decision_v6 in (
                    W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY,
                    W69_HANDOFF_DECISION_HOSTED_WITH_REAL_AUDIT,
                    W74_HANDOFF_DECISION_COMPOUND_REPAIR_FALLBACK):
                compound_repair_drtr_alignment = 1.0
            else:
                compound_repair_drtr_alignment = 0.0
        env = HandoffEnvelopeV6(
            schema=W74_HOSTED_REAL_HANDOFF_V6_SCHEMA_VERSION,
            inner_v5=v5_env,
            decision_v6=str(decision_v6),
            plane_v6=str(plane_v6),
            compound_alignment=float(compound_alignment),
            compound_repair_drtr_alignment=float(
                compound_repair_drtr_alignment),
            compound_repair_drtr_fallback_active=bool(
                compound_repair_drtr_fallback_active),
        )
        self.audit_v6.append({
            "request_cid": str(req_v6.request_cid),
            "decision_v6": str(decision_v6),
            "plane_v6": str(plane_v6),
            "envelope_v6_cid": str(env.cid()),
            "compound_alignment": float(
                compound_alignment),
            "compound_repair_drtr_alignment": float(
                compound_repair_drtr_alignment),
        })
        return env


@dataclasses.dataclass(frozen=True)
class HostedRealHandoffV6CompoundFalsifier:
    schema: str
    envelope_v6_cid: str
    claim_kind: str
    claim_satisfied: bool
    falsifier_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "envelope_v6_cid": str(self.envelope_v6_cid),
            "claim_kind": str(self.claim_kind),
            "claim_satisfied": bool(self.claim_satisfied),
            "falsifier_score": float(round(
                self.falsifier_score, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind":
                "hosted_real_handoff_v6_compound_falsifier",
            "falsifier": self.to_dict()})


def probe_hosted_real_handoff_v6_compound_falsifier(
        *, envelope_v6: HandoffEnvelopeV6,
        claim_kind: str = "hosted_satisfies_compound_repair_drtr",
        claim_satisfied: bool = False,
) -> HostedRealHandoffV6CompoundFalsifier:
    """Returns 0 on honest claims, 1 on dishonest.

    Dishonest = envelope says ``hosted_only`` AND caller asserts
    that hosted satisfied a non-trivial compound-repair trajectory.
    """
    score = 0.0
    if (str(envelope_v6.decision_v6)
            == W69_HANDOFF_DECISION_HOSTED_ONLY
            and claim_kind
                == "hosted_satisfies_compound_repair_drtr"
            and bool(claim_satisfied)):
        score = 1.0
    return HostedRealHandoffV6CompoundFalsifier(
        schema=W74_HOSTED_REAL_HANDOFF_V6_SCHEMA_VERSION,
        envelope_v6_cid=str(envelope_v6.cid()),
        claim_kind=str(claim_kind),
        claim_satisfied=bool(claim_satisfied),
        falsifier_score=float(score),
    )


def hosted_real_handoff_v6_compound_aware_savings(
        *, n_turns: int,
        hosted_only_tokens: int = 1000,
        hosted_with_real_audit_tokens: int = 400,
        real_substrate_only_tokens: int = 100,
        budget_primary_fallback_tokens: int = 70,
        delayed_repair_fallback_tokens: int = 60,
        delayed_rejoin_fallback_tokens: int = 55,
        replacement_after_ctr_fallback_tokens: int = 50,
        compound_repair_fallback_tokens: int = 45,
        real_substrate_fraction: float = 0.48,
        hosted_with_real_audit_fraction: float = 0.10,
        budget_primary_fallback_fraction: float = 0.06,
        delayed_repair_fallback_fraction: float = 0.08,
        delayed_rejoin_fallback_fraction: float = 0.10,
        replacement_after_ctr_fallback_fraction: float = 0.08,
        compound_repair_fallback_fraction: float = 0.08,
) -> dict[str, Any]:
    """V6 cross-plane handoff savings — compound-aware promotion
    plus compound-repair-after-DRTR fallback drive total visible-
    token cost down vs forcing every turn through hosted_only.

    Defaults: 48 % real_substrate, 10 % hosted_with_real_audit,
    6 % budget_primary_fallback, 8 % delayed_repair_fallback,
    10 % delayed_rejoin_fallback, 8 %
    replacement_after_ctr_fallback, 8 %
    compound_repair_fallback, 2 % hosted_only.
    Saving ratio should be ≥ 82 % at default config.
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
    rep_f = float(max(
        0.0, min(1.0 - rs_f - ha_f - bp_f - dr_f - rj_f,
                 replacement_after_ctr_fallback_fraction)))
    cmp_f = float(max(
        0.0, min(1.0 - rs_f - ha_f - bp_f - dr_f - rj_f - rep_f,
                 compound_repair_fallback_fraction)))
    ho_f = float(max(
        0.0,
        1.0 - rs_f - ha_f - bp_f - dr_f - rj_f - rep_f - cmp_f))
    n_rs = int(round(rs_f * n))
    n_ha = int(round(ha_f * n))
    n_bp = int(round(bp_f * n))
    n_dr = int(round(dr_f * n))
    n_rj = int(round(rj_f * n))
    n_rep = int(round(rep_f * n))
    n_cmp = int(round(cmp_f * n))
    n_ho = int(max(
        0,
        n - n_rs - n_ha - n_bp - n_dr - n_rj - n_rep - n_cmp))
    total_handoff = int(
        int(real_substrate_only_tokens) * int(n_rs)
        + int(hosted_with_real_audit_tokens) * int(n_ha)
        + int(budget_primary_fallback_tokens) * int(n_bp)
        + int(delayed_repair_fallback_tokens) * int(n_dr)
        + int(delayed_rejoin_fallback_tokens) * int(n_rj)
        + int(replacement_after_ctr_fallback_tokens) * int(n_rep)
        + int(compound_repair_fallback_tokens) * int(n_cmp)
        + int(hosted_only_tokens) * int(n_ho))
    total_all_hosted = int(int(hosted_only_tokens) * int(n))
    saving = int(total_all_hosted - total_handoff)
    ratio = (
        float(saving) / float(max(1, total_all_hosted))
        if total_all_hosted > 0 else 0.0)
    return {
        "schema": W74_HOSTED_REAL_HANDOFF_V6_SCHEMA_VERSION,
        "n_turns": int(n),
        "n_real_substrate_only": int(n_rs),
        "n_hosted_with_real_audit": int(n_ha),
        "n_budget_primary_fallback": int(n_bp),
        "n_delayed_repair_fallback": int(n_dr),
        "n_delayed_rejoin_fallback": int(n_rj),
        "n_replacement_after_ctr_fallback": int(n_rep),
        "n_compound_repair_fallback": int(n_cmp),
        "n_hosted_only": int(n_ho),
        "total_handoff_tokens": int(total_handoff),
        "total_all_hosted_tokens": int(total_all_hosted),
        "saving_tokens": int(saving),
        "saving_ratio": float(round(ratio, 12)),
    }


@dataclasses.dataclass(frozen=True)
class HostedRealHandoffCoordinatorV6Witness:
    schema: str
    coordinator_v6_cid: str
    n_envelopes_v6: int
    n_hosted_only: int
    n_real_substrate_only: int
    n_hosted_with_real_audit: int
    n_abstain: int
    n_budget_primary_fallback: int
    n_delayed_repair_fallback: int
    n_delayed_rejoin_fallback: int
    n_replacement_after_ctr_fallback: int
    n_compound_repair_fallback: int
    inner_v5_witness_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "coordinator_v6_cid": str(self.coordinator_v6_cid),
            "n_envelopes_v6": int(self.n_envelopes_v6),
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
            "n_replacement_after_ctr_fallback": int(
                self.n_replacement_after_ctr_fallback),
            "n_compound_repair_fallback": int(
                self.n_compound_repair_fallback),
            "inner_v5_witness_cid": str(
                self.inner_v5_witness_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind":
                "hosted_real_handoff_coordinator_v6_witness",
            "witness": self.to_dict()})


def emit_hosted_real_handoff_coordinator_v6_witness(
        coordinator: HostedRealHandoffCoordinatorV6,
) -> HostedRealHandoffCoordinatorV6Witness:
    from .hosted_real_handoff_coordinator_v5 import (
        emit_hosted_real_handoff_coordinator_v5_witness,
    )
    inner_w = emit_hosted_real_handoff_coordinator_v5_witness(
        coordinator.inner_v5)
    counts = {d: 0 for d in W74_HANDOFF_DECISIONS_V6}
    for entry in coordinator.audit_v6:
        d = str(entry.get("decision_v6", ""))
        if d in counts:
            counts[d] += 1
    return HostedRealHandoffCoordinatorV6Witness(
        schema=W74_HOSTED_REAL_HANDOFF_V6_SCHEMA_VERSION,
        coordinator_v6_cid=str(coordinator.cid()),
        n_envelopes_v6=int(len(coordinator.audit_v6)),
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
        n_replacement_after_ctr_fallback=int(
            counts[
                W73_HANDOFF_DECISION_REPLACEMENT_AFTER_CTR_FALLBACK]),
        n_compound_repair_fallback=int(
            counts[
                W74_HANDOFF_DECISION_COMPOUND_REPAIR_FALLBACK]),
        inner_v5_witness_cid=str(inner_w.cid()),
    )


__all__ = [
    "W74_HOSTED_REAL_HANDOFF_V6_SCHEMA_VERSION",
    "W74_HANDOFF_DECISION_COMPOUND_REPAIR_FALLBACK",
    "W74_HANDOFF_DECISIONS_V6",
    "W74_DEFAULT_COMPOUND_PRESSURE_FLOOR",
    "W74_DEFAULT_COMPOUND_WINDOW_FLOOR",
    "W74_DEFAULT_SUBSTRATE_TRUST_FLOOR",
    "HandoffRequestV6",
    "HandoffEnvelopeV6",
    "HostedRealHandoffCoordinatorV6",
    "HostedRealHandoffV6CompoundFalsifier",
    "probe_hosted_real_handoff_v6_compound_falsifier",
    "hosted_real_handoff_v6_compound_aware_savings",
    "HostedRealHandoffCoordinatorV6Witness",
    "emit_hosted_real_handoff_coordinator_v6_witness",
]
