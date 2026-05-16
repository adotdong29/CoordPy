"""W70 H6 — Hosted ↔ Real Substrate Handoff Coordinator V2.

The load-bearing W70 **budget-primary** Plane A↔B bridge. W69's V1
handoff coordinator routed each turn to ``hosted_only`` /
``real_substrate_only`` / ``hosted_with_real_substrate_audit`` /
``abstain`` based on whether the request needed text and/or
substrate state.

V2 adds **budget-primary scoring**: it picks the plane that
maximises ``team_success_per_visible_token`` rather than a binary
need flag, and adds a new ``budget_primary_fallback`` decision when
neither plane satisfies the declared visible-token budget but
hosted is at least cheaper.

V2 also exposes a **repair-dominance falsifier** that, given a
declared repair-trajectory CID, checks that the recorded envelope
plane is consistent with the V15 substrate's dominant repair label
(only Plane B can satisfy repair-dominance routing).

Honest scope (W70)
------------------

* The coordinator V2 does NOT pierce the hosted substrate boundary.
* ``W70-L-HANDOFF-V2-NOT-CROSSING-WALL-CAP`` — V2 preserves the W69
  boundary as a content-addressed invariant.
* Determinism on (request CID, registry CID, V15 substrate self-
  checksum CID, declared visible-token budget).
"""

from __future__ import annotations

import dataclasses
from typing import Any

from .hosted_real_handoff_coordinator import (
    HandoffEnvelope, HandoffRequest, HostedRealHandoffCoordinator,
    W69_HANDOFF_DECISION_ABSTAIN,
    W69_HANDOFF_DECISION_HOSTED_ONLY,
    W69_HANDOFF_DECISION_HOSTED_WITH_REAL_AUDIT,
    W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY,
)
from .hosted_real_substrate_boundary_v3 import (
    HostedRealSubstrateBoundaryV3,
    build_default_hosted_real_substrate_boundary_v3,
)
from .hosted_router_controller import HostedRoutingDecision
from .tiny_substrate_v3 import _sha256_hex


W70_HOSTED_REAL_HANDOFF_V2_SCHEMA_VERSION: str = (
    "coordpy.hosted_real_handoff_coordinator_v2.v1")

W70_HANDOFF_DECISION_BUDGET_PRIMARY_FALLBACK: str = (
    "budget_primary_fallback")
W70_HANDOFF_DECISIONS_V2: tuple[str, ...] = (
    W69_HANDOFF_DECISION_HOSTED_ONLY,
    W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY,
    W69_HANDOFF_DECISION_HOSTED_WITH_REAL_AUDIT,
    W69_HANDOFF_DECISION_ABSTAIN,
    W70_HANDOFF_DECISION_BUDGET_PRIMARY_FALLBACK,
)


@dataclasses.dataclass(frozen=True)
class HandoffRequestV2:
    inner_v1: HandoffRequest
    visible_token_budget: int = 256
    baseline_token_cost: int = 512
    expected_team_success_hosted: float = 0.65
    expected_team_success_substrate: float = 0.82
    expected_team_success_audit: float = 0.88
    repair_trajectory_cid: str = ""
    dominant_repair_label: int = 0

    @property
    def request_cid(self) -> str:
        return self.inner_v1.request_cid

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W70_HOSTED_REAL_HANDOFF_V2_SCHEMA_VERSION,
            "kind": "handoff_request_v2",
            "inner_v1_cid": str(self.inner_v1.cid()),
            "visible_token_budget": int(
                self.visible_token_budget),
            "baseline_token_cost": int(self.baseline_token_cost),
            "expected_team_success_hosted": float(round(
                self.expected_team_success_hosted, 12)),
            "expected_team_success_substrate": float(round(
                self.expected_team_success_substrate, 12)),
            "expected_team_success_audit": float(round(
                self.expected_team_success_audit, 12)),
            "repair_trajectory_cid": str(
                self.repair_trajectory_cid),
            "dominant_repair_label": int(
                self.dominant_repair_label),
        })


@dataclasses.dataclass(frozen=True)
class HandoffEnvelopeV2:
    schema: str
    inner_v1: HandoffEnvelope
    decision_v2: str
    plane_v2: str
    team_success_per_visible_token_score: float
    repair_dominance_alignment: float
    budget_primary_fallback_active: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "inner_v1_cid": str(self.inner_v1.cid()),
            "decision_v2": str(self.decision_v2),
            "plane_v2": str(self.plane_v2),
            "team_success_per_visible_token_score": float(round(
                self.team_success_per_visible_token_score, 12)),
            "repair_dominance_alignment": float(round(
                self.repair_dominance_alignment, 12)),
            "budget_primary_fallback_active": bool(
                self.budget_primary_fallback_active),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_real_handoff_envelope_v2",
            "envelope": self.to_dict()})


@dataclasses.dataclass
class HostedRealHandoffCoordinatorV2:
    inner_v1: HostedRealHandoffCoordinator = dataclasses.field(
        default_factory=HostedRealHandoffCoordinator)
    boundary_v3: HostedRealSubstrateBoundaryV3 = (
        dataclasses.field(
            default_factory=(
                build_default_hosted_real_substrate_boundary_v3)))
    audit_v2: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W70_HOSTED_REAL_HANDOFF_V2_SCHEMA_VERSION,
            "kind": "hosted_real_handoff_coordinator_v2",
            "inner_v1_cid": str(self.inner_v1.cid()),
            "boundary_v3_cid": str(self.boundary_v3.cid()),
        })

    def decide_v2(
            self, *, req_v2: HandoffRequestV2,
            hosted_decision: HostedRoutingDecision | None = None,
            substrate_self_checksum_cid: str = "",
    ) -> HandoffEnvelopeV2:
        v1_env = self.inner_v1.decide(
            req=req_v2.inner_v1,
            hosted_decision=hosted_decision,
            substrate_self_checksum_cid=str(
                substrate_self_checksum_cid))
        budget_ratio = float(
            max(0.0, min(1.0,
                         float(req_v2.visible_token_budget)
                         / float(max(1,
                                     req_v2.baseline_token_cost)))))
        # team-success-per-visible-token score per plane.
        hosted_score = (
            float(req_v2.expected_team_success_hosted)
            * float(budget_ratio))
        substrate_score = (
            float(req_v2.expected_team_success_substrate)
            * (0.4 + 0.6 * float(budget_ratio)))
        audit_score = (
            float(req_v2.expected_team_success_audit)
            * float(budget_ratio))
        # Initial decision: inherit V1 envelope plane.
        decision_v2 = str(v1_env.decision)
        plane_v2 = str(v1_env.plane)
        budget_primary_fallback_active = False
        # V2 budget-primary scoring: if budget < 0.5 of baseline AND
        # V1 chose hosted_only OR hosted_with_real_substrate_audit,
        # V2 promotes to real_substrate_only when substrate_score >
        # hosted_score.
        if (budget_ratio < 0.5
                and decision_v2
                in (W69_HANDOFF_DECISION_HOSTED_ONLY,
                    W69_HANDOFF_DECISION_HOSTED_WITH_REAL_AUDIT)):
            if (substrate_score > hosted_score
                    and substrate_score > audit_score):
                decision_v2 = (
                    W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY)
                plane_v2 = "B"
            elif (substrate_score <= hosted_score
                    and substrate_score <= audit_score):
                decision_v2 = (
                    W70_HANDOFF_DECISION_BUDGET_PRIMARY_FALLBACK)
                plane_v2 = "A-bp-fallback"
                budget_primary_fallback_active = True
        # Repair-dominance alignment: substrate plane is the only
        # plane that can satisfy nonzero dominant_repair_label.
        if (int(req_v2.dominant_repair_label) > 0
                and decision_v2 == W69_HANDOFF_DECISION_HOSTED_ONLY):
            decision_v2 = (
                W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY)
            plane_v2 = "B"
        rd_alignment = float(
            1.0 if (
                int(req_v2.dominant_repair_label) == 0
                or decision_v2
                in (W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY,
                    W69_HANDOFF_DECISION_HOSTED_WITH_REAL_AUDIT))
            else 0.0)
        chosen_score = (
            substrate_score
            if decision_v2
            == W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY
            else (audit_score
                  if decision_v2
                  == W69_HANDOFF_DECISION_HOSTED_WITH_REAL_AUDIT
                  else hosted_score))
        env = HandoffEnvelopeV2(
            schema=W70_HOSTED_REAL_HANDOFF_V2_SCHEMA_VERSION,
            inner_v1=v1_env,
            decision_v2=str(decision_v2),
            plane_v2=str(plane_v2),
            team_success_per_visible_token_score=float(
                chosen_score),
            repair_dominance_alignment=float(rd_alignment),
            budget_primary_fallback_active=bool(
                budget_primary_fallback_active),
        )
        self.audit_v2.append({
            "request_cid": str(req_v2.request_cid),
            "decision_v2": str(decision_v2),
            "plane_v2": str(plane_v2),
            "envelope_v2_cid": str(env.cid()),
            "team_success_per_visible_token_score": float(
                chosen_score),
            "repair_dominance_alignment": float(rd_alignment),
        })
        return env


@dataclasses.dataclass(frozen=True)
class HostedRealHandoffV2Falsifier:
    schema: str
    envelope_v2_cid: str
    claim_kind: str
    claim_satisfied: bool
    falsifier_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "envelope_v2_cid": str(self.envelope_v2_cid),
            "claim_kind": str(self.claim_kind),
            "claim_satisfied": bool(self.claim_satisfied),
            "falsifier_score": float(round(
                self.falsifier_score, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_real_handoff_v2_falsifier",
            "falsifier": self.to_dict()})


def probe_hosted_real_handoff_v2_repair_dominance_falsifier(
        *, envelope_v2: HandoffEnvelopeV2,
        claim_kind: str = "hosted_satisfies_repair_dominance",
        claim_satisfied: bool = False,
) -> HostedRealHandoffV2Falsifier:
    """Returns 0 on honest claims, 1 on dishonest.

    Dishonest = envelope says ``hosted_only`` AND caller asserts
    that hosted satisfied a nonzero repair-dominance label."""
    score = 0.0
    if (str(envelope_v2.decision_v2)
            == W69_HANDOFF_DECISION_HOSTED_ONLY
            and claim_kind == "hosted_satisfies_repair_dominance"
            and bool(claim_satisfied)):
        score = 1.0
    return HostedRealHandoffV2Falsifier(
        schema=W70_HOSTED_REAL_HANDOFF_V2_SCHEMA_VERSION,
        envelope_v2_cid=str(envelope_v2.cid()),
        claim_kind=str(claim_kind),
        claim_satisfied=bool(claim_satisfied),
        falsifier_score=float(score),
    )


def hosted_real_handoff_v2_budget_primary_savings(
        *, n_turns: int,
        hosted_only_tokens: int = 1000,
        hosted_with_real_audit_tokens: int = 400,
        real_substrate_only_tokens: int = 150,
        budget_primary_fallback_tokens: int = 80,
        real_substrate_fraction: float = 0.55,
        hosted_with_real_audit_fraction: float = 0.20,
        budget_primary_fallback_fraction: float = 0.15,
) -> dict[str, Any]:
    """V2 cross-plane handoff savings — budget-primary fallback
    drives total visible-token cost down vs forcing every turn
    through hosted_only.

    Defaults: 55 % real_substrate, 20 % hosted_with_real_audit,
    15 % budget_primary_fallback, 10 % hosted_only.
    Saving ratio should be ≥ 65 % at default config."""
    n = int(max(1, n_turns))
    rs_f = float(max(0.0, min(1.0, real_substrate_fraction)))
    ha_f = float(max(
        0.0, min(1.0 - rs_f, hosted_with_real_audit_fraction)))
    bp_f = float(max(
        0.0, min(1.0 - rs_f - ha_f,
                 budget_primary_fallback_fraction)))
    ho_f = float(max(0.0, 1.0 - rs_f - ha_f - bp_f))
    n_rs = int(round(rs_f * n))
    n_ha = int(round(ha_f * n))
    n_bp = int(round(bp_f * n))
    n_ho = int(max(0, n - n_rs - n_ha - n_bp))
    total_handoff = int(
        int(real_substrate_only_tokens) * int(n_rs)
        + int(hosted_with_real_audit_tokens) * int(n_ha)
        + int(budget_primary_fallback_tokens) * int(n_bp)
        + int(hosted_only_tokens) * int(n_ho))
    total_all_hosted = int(int(hosted_only_tokens) * int(n))
    saving = int(total_all_hosted - total_handoff)
    ratio = (
        float(saving) / float(max(1, total_all_hosted))
        if total_all_hosted > 0 else 0.0)
    return {
        "schema": W70_HOSTED_REAL_HANDOFF_V2_SCHEMA_VERSION,
        "n_turns": int(n),
        "n_real_substrate_only": int(n_rs),
        "n_hosted_with_real_audit": int(n_ha),
        "n_budget_primary_fallback": int(n_bp),
        "n_hosted_only": int(n_ho),
        "total_handoff_tokens": int(total_handoff),
        "total_all_hosted_tokens": int(total_all_hosted),
        "saving_tokens": int(saving),
        "saving_ratio": float(round(ratio, 12)),
    }


@dataclasses.dataclass(frozen=True)
class HostedRealHandoffCoordinatorV2Witness:
    schema: str
    coordinator_v2_cid: str
    n_envelopes_v2: int
    n_hosted_only: int
    n_real_substrate_only: int
    n_hosted_with_real_audit: int
    n_abstain: int
    n_budget_primary_fallback: int
    inner_v1_witness_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "coordinator_v2_cid": str(self.coordinator_v2_cid),
            "n_envelopes_v2": int(self.n_envelopes_v2),
            "n_hosted_only": int(self.n_hosted_only),
            "n_real_substrate_only": int(
                self.n_real_substrate_only),
            "n_hosted_with_real_audit": int(
                self.n_hosted_with_real_audit),
            "n_abstain": int(self.n_abstain),
            "n_budget_primary_fallback": int(
                self.n_budget_primary_fallback),
            "inner_v1_witness_cid": str(self.inner_v1_witness_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_real_handoff_coordinator_v2_witness",
            "witness": self.to_dict()})


def emit_hosted_real_handoff_coordinator_v2_witness(
        coordinator: HostedRealHandoffCoordinatorV2,
) -> HostedRealHandoffCoordinatorV2Witness:
    from .hosted_real_handoff_coordinator import (
        emit_hosted_real_handoff_coordinator_witness,
    )
    inner_w = emit_hosted_real_handoff_coordinator_witness(
        coordinator.inner_v1)
    counts = {d: 0 for d in W70_HANDOFF_DECISIONS_V2}
    for entry in coordinator.audit_v2:
        d = str(entry.get("decision_v2", ""))
        if d in counts:
            counts[d] += 1
    return HostedRealHandoffCoordinatorV2Witness(
        schema=W70_HOSTED_REAL_HANDOFF_V2_SCHEMA_VERSION,
        coordinator_v2_cid=str(coordinator.cid()),
        n_envelopes_v2=int(len(coordinator.audit_v2)),
        n_hosted_only=int(
            counts[W69_HANDOFF_DECISION_HOSTED_ONLY]),
        n_real_substrate_only=int(
            counts[W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY]),
        n_hosted_with_real_audit=int(
            counts[W69_HANDOFF_DECISION_HOSTED_WITH_REAL_AUDIT]),
        n_abstain=int(counts[W69_HANDOFF_DECISION_ABSTAIN]),
        n_budget_primary_fallback=int(
            counts[W70_HANDOFF_DECISION_BUDGET_PRIMARY_FALLBACK]),
        inner_v1_witness_cid=str(inner_w.cid()),
    )


__all__ = [
    "W70_HOSTED_REAL_HANDOFF_V2_SCHEMA_VERSION",
    "W70_HANDOFF_DECISION_BUDGET_PRIMARY_FALLBACK",
    "W70_HANDOFF_DECISIONS_V2",
    "HandoffRequestV2",
    "HandoffEnvelopeV2",
    "HostedRealHandoffCoordinatorV2",
    "HostedRealHandoffV2Falsifier",
    "probe_hosted_real_handoff_v2_repair_dominance_falsifier",
    "hosted_real_handoff_v2_budget_primary_savings",
    "HostedRealHandoffCoordinatorV2Witness",
    "emit_hosted_real_handoff_coordinator_v2_witness",
]
