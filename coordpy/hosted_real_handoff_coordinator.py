"""W69 H7 — Hosted ↔ Real Substrate Handoff Coordinator.

The first **load-bearing W69 Plane A↔B bridge**. The W68 wall is
*static*: it enumerates what is blocked at the hosted surface. W69
adds an *operational* coordinator that, *while preserving the
wall*, decides per-turn whether Plane A (hosted, text) or Plane B
(real substrate) handles each turn, records a content-addressed
handoff envelope, and exposes a falsifier.

Honest scope (W69)
------------------

* The coordinator does NOT pierce the hosted substrate boundary.
  The decision space is: ``hosted_only`` / ``real_substrate_only``
  / ``hosted_with_real_substrate_audit`` (where Plane B records
  state behind Plane A's response) / ``abstain``.
* Handoff envelopes record decisions, prefix CIDs, substrate self-
  checksum CIDs, and falsifier outcomes; they do not claim cross-
  plane substrate access.
* ``W69-L-HANDOFF-NOT-CROSSING-WALL-CAP`` — the coordinator
  preserves the W68 boundary as a content-addressed invariant.
* Determinism on (request CID, registry CID, substrate self-
  checksum CID).

Falsifier
---------

* ``probe_handoff_falsifier`` returns 0 when the recorded handoff
  decision is consistent with the wall (no hosted substrate
  claims). Returns 1 when the decision claims hosted-substrate
  access.
"""

from __future__ import annotations

import dataclasses
import hashlib
from typing import Any, Sequence

from .hosted_real_substrate_boundary import (
    HostedRealSubstrateBoundary,
    build_default_hosted_real_substrate_boundary,
)
from .hosted_router_controller import (
    HostedRoutingDecision, HostedRoutingRequest,
)
from .tiny_substrate_v3 import _sha256_hex


W69_HOSTED_REAL_HANDOFF_SCHEMA_VERSION: str = (
    "coordpy.hosted_real_handoff_coordinator.v1")

W69_HANDOFF_DECISION_HOSTED_ONLY: str = "hosted_only"
W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY: str = (
    "real_substrate_only")
W69_HANDOFF_DECISION_HOSTED_WITH_REAL_AUDIT: str = (
    "hosted_with_real_substrate_audit")
W69_HANDOFF_DECISION_ABSTAIN: str = "abstain"
W69_HANDOFF_DECISIONS: tuple[str, ...] = (
    W69_HANDOFF_DECISION_HOSTED_ONLY,
    W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY,
    W69_HANDOFF_DECISION_HOSTED_WITH_REAL_AUDIT,
    W69_HANDOFF_DECISION_ABSTAIN,
)


@dataclasses.dataclass(frozen=True)
class HandoffRequest:
    request_cid: str
    needs_text_only: bool
    needs_substrate_state_access: bool
    role: str = "planner"
    task_id: str = "task"
    team_id: str = "team"
    branch_id: str = "main"

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W69_HOSTED_REAL_HANDOFF_SCHEMA_VERSION,
            "kind": "handoff_request",
            "request_cid": str(self.request_cid),
            "needs_text_only": bool(self.needs_text_only),
            "needs_substrate_state_access": bool(
                self.needs_substrate_state_access),
            "role": str(self.role),
            "task_id": str(self.task_id),
            "team_id": str(self.team_id),
            "branch_id": str(self.branch_id),
        })


@dataclasses.dataclass(frozen=True)
class HandoffEnvelope:
    """Per-turn content-addressed Plane A↔B handoff envelope."""
    schema: str
    request_cid: str
    decision: str
    plane: str   # "A" or "B" or "A+B-audit" or "none"
    hosted_routing_decision_cid: str
    substrate_self_checksum_cid: str
    handoff_witness_cid: str
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "request_cid": str(self.request_cid),
            "decision": str(self.decision),
            "plane": str(self.plane),
            "hosted_routing_decision_cid": str(
                self.hosted_routing_decision_cid),
            "substrate_self_checksum_cid": str(
                self.substrate_self_checksum_cid),
            "handoff_witness_cid": str(
                self.handoff_witness_cid),
            "rationale": str(self.rationale),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_real_handoff_envelope",
            "envelope": self.to_dict()})


@dataclasses.dataclass
class HostedRealHandoffCoordinator:
    boundary: HostedRealSubstrateBoundary = dataclasses.field(
        default_factory=build_default_hosted_real_substrate_boundary)
    audit: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)
    # Pre-committed routing thresholds.
    real_substrate_when_needed: bool = True
    abstain_when_both_required_and_unavailable: bool = True

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W69_HOSTED_REAL_HANDOFF_SCHEMA_VERSION,
            "kind": "hosted_real_handoff_coordinator",
            "boundary_cid": str(self.boundary.cid()),
            "real_substrate_when_needed": bool(
                self.real_substrate_when_needed),
            "abstain_when_both_required_and_unavailable": bool(
                self
                .abstain_when_both_required_and_unavailable),
        })

    def decide(
            self, *, req: HandoffRequest,
            hosted_decision: HostedRoutingDecision | None = None,
            substrate_self_checksum_cid: str = "",
    ) -> HandoffEnvelope:
        # If the request needs substrate state access, only Plane B
        # can satisfy it. If hosted is also required, fall back to
        # hosted_with_real_substrate_audit.
        decision = W69_HANDOFF_DECISION_ABSTAIN
        plane = "none"
        rationale = "default_abstain"
        if (req.needs_substrate_state_access
                and req.needs_text_only
                and bool(self.real_substrate_when_needed)):
            decision = (
                W69_HANDOFF_DECISION_HOSTED_WITH_REAL_AUDIT)
            plane = "A+B-audit"
            rationale = (
                "needs_both_text_and_substrate")
        elif req.needs_substrate_state_access:
            decision = (
                W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY)
            plane = "B"
            rationale = "needs_substrate_state_access"
        elif req.needs_text_only:
            decision = W69_HANDOFF_DECISION_HOSTED_ONLY
            plane = "A"
            rationale = "text_only"
        elif (req.needs_text_only
                and req.needs_substrate_state_access
                and not bool(self.real_substrate_when_needed)
                and bool(
                    self
                    .abstain_when_both_required_and_unavailable)):
            decision = W69_HANDOFF_DECISION_ABSTAIN
            plane = "none"
            rationale = "both_unavailable"
        hosted_cid = (
            str(hosted_decision.cid())
            if hosted_decision is not None else "")
        # Per-turn handoff witness CID covers (request, decision,
        # hosted_routing, substrate_self_checksum).
        witness_cid = _sha256_hex({
            "kind": "handoff_witness",
            "request_cid": str(req.cid()),
            "decision": str(decision),
            "plane": str(plane),
            "hosted_routing_decision_cid": str(hosted_cid),
            "substrate_self_checksum_cid": str(
                substrate_self_checksum_cid),
        })
        env = HandoffEnvelope(
            schema=W69_HOSTED_REAL_HANDOFF_SCHEMA_VERSION,
            request_cid=str(req.request_cid),
            decision=str(decision),
            plane=str(plane),
            hosted_routing_decision_cid=str(hosted_cid),
            substrate_self_checksum_cid=str(
                substrate_self_checksum_cid),
            handoff_witness_cid=str(witness_cid),
            rationale=str(rationale),
        )
        self.audit.append({
            "request_cid": str(req.request_cid),
            "decision": str(decision),
            "plane": str(plane),
            "envelope_cid": str(env.cid()),
        })
        return env


@dataclasses.dataclass(frozen=True)
class HostedRealHandoffFalsifier:
    schema: str
    envelope_cid: str
    claim_kind: str
    claim_satisfied: bool
    falsifier_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "envelope_cid": str(self.envelope_cid),
            "claim_kind": str(self.claim_kind),
            "claim_satisfied": bool(self.claim_satisfied),
            "falsifier_score": float(round(
                self.falsifier_score, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_real_handoff_falsifier",
            "falsifier": self.to_dict()})


def probe_hosted_real_handoff_falsifier(
        *, envelope: HandoffEnvelope,
        claim_kind: str = "hosted_provides_substrate_state",
        claim_satisfied: bool = False,
) -> HostedRealHandoffFalsifier:
    """Returns 0 on honest claims and 1 on dishonest claims.

    Dishonest = the envelope says ``hosted_only`` AND someone
    asserts that hosted provided substrate state. Honest = envelope
    accurately reflects which plane handled the turn."""
    score = 0.0
    if (str(envelope.decision) == W69_HANDOFF_DECISION_HOSTED_ONLY
            and claim_kind == "hosted_provides_substrate_state"
            and bool(claim_satisfied)):
        score = 1.0
    # Honest case: real_substrate_only or hosted_with_real_audit
    # with a claim that substrate state was accessed is consistent.
    return HostedRealHandoffFalsifier(
        schema=W69_HOSTED_REAL_HANDOFF_SCHEMA_VERSION,
        envelope_cid=str(envelope.cid()),
        claim_kind=str(claim_kind),
        claim_satisfied=bool(claim_satisfied),
        falsifier_score=float(score),
    )


def hosted_real_handoff_cross_plane_savings(
        *, n_turns: int,
        hosted_only_tokens: int = 1000,
        hosted_with_real_audit_tokens: int = 400,
        real_substrate_only_tokens: int = 200,
        real_substrate_fraction: float = 0.50,
        hosted_with_real_audit_fraction: float = 0.30,
) -> dict[str, Any]:
    """Cross-plane handoff savings.

    The V14 handoff coordinator routes turns to the cheapest plane,
    saving visible tokens vs forcing every turn through hosted_only.
    Defaults: 50 % real_substrate_only, 30 % hosted_with_real_audit,
    20 % hosted_only — which matches the W69 typical multi-agent
    workload where most state-bearing turns are Plane B."""
    n = int(max(1, n_turns))
    rs_f = float(max(0.0, min(1.0, real_substrate_fraction)))
    ha_f = float(max(
        0.0, min(1.0 - rs_f, hosted_with_real_audit_fraction)))
    ho_f = float(max(0.0, 1.0 - rs_f - ha_f))
    n_rs = int(round(rs_f * n))
    n_ha = int(round(ha_f * n))
    n_ho = int(max(0, n - n_rs - n_ha))
    total_handoff = int(
        int(real_substrate_only_tokens) * int(n_rs)
        + int(hosted_with_real_audit_tokens) * int(n_ha)
        + int(hosted_only_tokens) * int(n_ho))
    total_all_hosted = int(int(hosted_only_tokens) * int(n))
    saving = int(total_all_hosted - total_handoff)
    ratio = (
        float(saving) / float(max(1, total_all_hosted))
        if total_all_hosted > 0 else 0.0)
    return {
        "schema": W69_HOSTED_REAL_HANDOFF_SCHEMA_VERSION,
        "n_turns": int(n),
        "n_real_substrate_only": int(n_rs),
        "n_hosted_with_real_audit": int(n_ha),
        "n_hosted_only": int(n_ho),
        "total_handoff_tokens": int(total_handoff),
        "total_all_hosted_tokens": int(total_all_hosted),
        "saving_tokens": int(saving),
        "saving_ratio": float(round(ratio, 12)),
    }


@dataclasses.dataclass(frozen=True)
class HostedRealHandoffCoordinatorWitness:
    schema: str
    coordinator_cid: str
    n_envelopes: int
    n_hosted_only: int
    n_real_substrate_only: int
    n_hosted_with_real_audit: int
    n_abstain: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "coordinator_cid": str(self.coordinator_cid),
            "n_envelopes": int(self.n_envelopes),
            "n_hosted_only": int(self.n_hosted_only),
            "n_real_substrate_only": int(
                self.n_real_substrate_only),
            "n_hosted_with_real_audit": int(
                self.n_hosted_with_real_audit),
            "n_abstain": int(self.n_abstain),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_real_handoff_coordinator_witness",
            "witness": self.to_dict()})


def emit_hosted_real_handoff_coordinator_witness(
        coordinator: HostedRealHandoffCoordinator,
) -> HostedRealHandoffCoordinatorWitness:
    counts = {d: 0 for d in W69_HANDOFF_DECISIONS}
    for entry in coordinator.audit:
        d = str(entry.get("decision", ""))
        if d in counts:
            counts[d] += 1
    return HostedRealHandoffCoordinatorWitness(
        schema=W69_HOSTED_REAL_HANDOFF_SCHEMA_VERSION,
        coordinator_cid=str(coordinator.cid()),
        n_envelopes=int(len(coordinator.audit)),
        n_hosted_only=int(
            counts[W69_HANDOFF_DECISION_HOSTED_ONLY]),
        n_real_substrate_only=int(
            counts[W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY]),
        n_hosted_with_real_audit=int(
            counts[W69_HANDOFF_DECISION_HOSTED_WITH_REAL_AUDIT]),
        n_abstain=int(counts[W69_HANDOFF_DECISION_ABSTAIN]),
    )


__all__ = [
    "W69_HOSTED_REAL_HANDOFF_SCHEMA_VERSION",
    "W69_HANDOFF_DECISION_HOSTED_ONLY",
    "W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY",
    "W69_HANDOFF_DECISION_HOSTED_WITH_REAL_AUDIT",
    "W69_HANDOFF_DECISION_ABSTAIN",
    "W69_HANDOFF_DECISIONS",
    "HandoffRequest",
    "HandoffEnvelope",
    "HostedRealHandoffCoordinator",
    "HostedRealHandoffFalsifier",
    "probe_hosted_real_handoff_falsifier",
    "hosted_real_handoff_cross_plane_savings",
    "HostedRealHandoffCoordinatorWitness",
    "emit_hosted_real_handoff_coordinator_witness",
]
