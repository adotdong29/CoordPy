"""W73 H6 — Hosted ↔ Real Substrate Handoff Coordinator V5.

The load-bearing W73 **replacement-aware** Plane A↔B bridge. W72's
V4 handoff coordinator added rejoin-aware promotion + delayed-
rejoin fallback on top of V3's restart-aware scoring. V5 adds:

* **Replacement-aware promotion** — any request with
  ``replacement_pressure >= replacement_pressure_floor`` AND
  substrate trust ≥ floor is promoted to ``real_substrate_only``
  with ``replacement_alignment = 1.0``, regardless of the V4
  rejoin-aware decision (and only when other pressures are not yet
  promoting).
* **Replacement-after-CTR fallback** — a new decision label
  ``replacement_after_contradiction_then_rejoin_fallback`` fires
  when hosted is cheaper but the caller-declared
  ``replacement_repair_trajectory_cid`` requires Plane B
  (replacement-lag > replacement-lag-floor AND substrate trust >
  floor) AND the request did not already promote to restart-aware,
  delayed-repair-fallback, or delayed-rejoin-fallback paths.
* **Replacement-pressure falsifier** — extends V4 to a
  replacement-after-CTR variant: rejects hosted-only envelopes
  that claim to satisfy a non-trivial replacement-repair trajectory.

Honest scope (W73)
------------------

* The coordinator V5 does NOT pierce the hosted substrate
  boundary.
* ``W73-L-HANDOFF-V5-NOT-CROSSING-WALL-CAP`` — V5 preserves the
  W72 boundary as a content-addressed invariant.
* Determinism on (request V5 CID, registry CID, V18 substrate
  self-checksum CID, declared visible-token budget, restart
  pressure, rejoin pressure, replacement pressure, replacement-
  repair trajectory CID).
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
    HandoffEnvelopeV4, HandoffRequestV4,
    HostedRealHandoffCoordinatorV4,
    W72_HANDOFF_DECISION_DELAYED_REJOIN_AFTER_RESTART_FALLBACK,
)
from .hosted_real_substrate_boundary_v6 import (
    HostedRealSubstrateBoundaryV6,
    build_default_hosted_real_substrate_boundary_v6,
)
from .hosted_router_controller import HostedRoutingDecision
from .tiny_substrate_v3 import _sha256_hex


W73_HOSTED_REAL_HANDOFF_V5_SCHEMA_VERSION: str = (
    "coordpy.hosted_real_handoff_coordinator_v5.v1")

W73_HANDOFF_DECISION_REPLACEMENT_AFTER_CTR_FALLBACK: str = (
    "replacement_after_contradiction_then_rejoin_fallback")
W73_HANDOFF_DECISIONS_V5: tuple[str, ...] = (
    W69_HANDOFF_DECISION_HOSTED_ONLY,
    W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY,
    W69_HANDOFF_DECISION_HOSTED_WITH_REAL_AUDIT,
    W69_HANDOFF_DECISION_ABSTAIN,
    W70_HANDOFF_DECISION_BUDGET_PRIMARY_FALLBACK,
    W71_HANDOFF_DECISION_DELAYED_REPAIR_FALLBACK,
    W72_HANDOFF_DECISION_DELAYED_REJOIN_AFTER_RESTART_FALLBACK,
    W73_HANDOFF_DECISION_REPLACEMENT_AFTER_CTR_FALLBACK,
)

W73_DEFAULT_REPLACEMENT_PRESSURE_FLOOR: float = 0.5
W73_DEFAULT_REPLACEMENT_LAG_FLOOR: int = 1
W73_DEFAULT_SUBSTRATE_TRUST_FLOOR: float = 0.5


@dataclasses.dataclass(frozen=True)
class HandoffRequestV5:
    inner_v4: HandoffRequestV4
    replacement_pressure: float = 0.0
    replacement_repair_trajectory_cid: str = ""
    replacement_lag_turns: int = 0
    expected_substrate_trust_v5: float = 0.7

    @property
    def request_cid(self) -> str:
        return self.inner_v4.request_cid

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W73_HOSTED_REAL_HANDOFF_V5_SCHEMA_VERSION,
            "kind": "handoff_request_v5",
            "inner_v4_cid": str(self.inner_v4.cid()),
            "replacement_pressure": float(round(
                self.replacement_pressure, 12)),
            "replacement_repair_trajectory_cid": str(
                self.replacement_repair_trajectory_cid),
            "replacement_lag_turns": int(
                self.replacement_lag_turns),
            "expected_substrate_trust_v5": float(round(
                self.expected_substrate_trust_v5, 12)),
        })


@dataclasses.dataclass(frozen=True)
class HandoffEnvelopeV5:
    schema: str
    inner_v4: HandoffEnvelopeV4
    decision_v5: str
    plane_v5: str
    replacement_alignment: float
    replacement_after_ctr_alignment: float
    replacement_after_ctr_fallback_active: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "inner_v4_cid": str(self.inner_v4.cid()),
            "decision_v5": str(self.decision_v5),
            "plane_v5": str(self.plane_v5),
            "replacement_alignment": float(round(
                self.replacement_alignment, 12)),
            "replacement_after_ctr_alignment": float(round(
                self.replacement_after_ctr_alignment, 12)),
            "replacement_after_ctr_fallback_active": bool(
                self.replacement_after_ctr_fallback_active),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_real_handoff_envelope_v5",
            "envelope": self.to_dict()})


@dataclasses.dataclass
class HostedRealHandoffCoordinatorV5:
    inner_v4: HostedRealHandoffCoordinatorV4 = dataclasses.field(
        default_factory=HostedRealHandoffCoordinatorV4)
    boundary_v6: HostedRealSubstrateBoundaryV6 = (
        dataclasses.field(
            default_factory=(
                build_default_hosted_real_substrate_boundary_v6)))
    replacement_pressure_floor: float = (
        W73_DEFAULT_REPLACEMENT_PRESSURE_FLOOR)
    replacement_lag_floor: int = W73_DEFAULT_REPLACEMENT_LAG_FLOOR
    substrate_trust_floor: float = (
        W73_DEFAULT_SUBSTRATE_TRUST_FLOOR)
    audit_v5: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W73_HOSTED_REAL_HANDOFF_V5_SCHEMA_VERSION,
            "kind": "hosted_real_handoff_coordinator_v5",
            "inner_v4_cid": str(self.inner_v4.cid()),
            "boundary_v6_cid": str(self.boundary_v6.cid()),
            "replacement_pressure_floor": float(round(
                self.replacement_pressure_floor, 12)),
            "replacement_lag_floor": int(
                self.replacement_lag_floor),
            "substrate_trust_floor": float(round(
                self.substrate_trust_floor, 12)),
        })

    def decide_v5(
            self, *, req_v5: HandoffRequestV5,
            hosted_decision: HostedRoutingDecision | None = None,
            substrate_self_checksum_cid: str = "",
    ) -> HandoffEnvelopeV5:
        v4_env = self.inner_v4.decide_v4(
            req_v4=req_v5.inner_v4,
            hosted_decision=hosted_decision,
            substrate_self_checksum_cid=str(
                substrate_self_checksum_cid))
        decision_v5 = str(v4_env.decision_v4)
        plane_v5 = str(v4_env.plane_v4)
        replacement_alignment = 0.0
        replacement_after_ctr_alignment = 0.0
        replacement_after_ctr_fallback_active = False
        # Replacement-aware promotion — fires when V4 stayed on
        # hosted_only AND replacement pressure is above the floor.
        if (decision_v5 == W69_HANDOFF_DECISION_HOSTED_ONLY
                and float(req_v5.replacement_pressure)
                    >= float(self.replacement_pressure_floor)
                and float(req_v5.expected_substrate_trust_v5)
                    >= float(self.substrate_trust_floor)):
            decision_v5 = (
                W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY)
            plane_v5 = "B"
            replacement_alignment = 1.0
        # Replacement-after-CTR fallback — fires when the
        # replacement-repair trajectory has a non-trivial lag AND
        # substrate trust is high, AND no other V4 fallback fired.
        elif (
                str(req_v5.replacement_repair_trajectory_cid)
                and int(req_v5.replacement_lag_turns)
                    >= int(self.replacement_lag_floor)
                and float(req_v5.expected_substrate_trust_v5)
                    >= float(self.substrate_trust_floor)
                and decision_v5
                    == W69_HANDOFF_DECISION_HOSTED_ONLY):
            decision_v5 = (
                W73_HANDOFF_DECISION_REPLACEMENT_AFTER_CTR_FALLBACK)
            plane_v5 = "A-rep-fallback"
            replacement_after_ctr_fallback_active = True
            replacement_after_ctr_alignment = 1.0
        # Honest alignment when no fallback fires.
        if (decision_v5
                in (W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY,
                    W69_HANDOFF_DECISION_HOSTED_WITH_REAL_AUDIT,
                    W73_HANDOFF_DECISION_REPLACEMENT_AFTER_CTR_FALLBACK)
                and float(req_v5.replacement_pressure) > 0.0):
            replacement_alignment = max(
                replacement_alignment, 1.0)
        # Replacement-after-CTR alignment: substrate / audit / rep-
        # fallback plane is the only plane that can honestly satisfy
        # a non-empty replacement-repair trajectory.
        if str(req_v5.replacement_repair_trajectory_cid):
            if decision_v5 in (
                    W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY,
                    W69_HANDOFF_DECISION_HOSTED_WITH_REAL_AUDIT,
                    W73_HANDOFF_DECISION_REPLACEMENT_AFTER_CTR_FALLBACK):
                replacement_after_ctr_alignment = 1.0
            else:
                replacement_after_ctr_alignment = 0.0
        env = HandoffEnvelopeV5(
            schema=W73_HOSTED_REAL_HANDOFF_V5_SCHEMA_VERSION,
            inner_v4=v4_env,
            decision_v5=str(decision_v5),
            plane_v5=str(plane_v5),
            replacement_alignment=float(replacement_alignment),
            replacement_after_ctr_alignment=float(
                replacement_after_ctr_alignment),
            replacement_after_ctr_fallback_active=bool(
                replacement_after_ctr_fallback_active),
        )
        self.audit_v5.append({
            "request_cid": str(req_v5.request_cid),
            "decision_v5": str(decision_v5),
            "plane_v5": str(plane_v5),
            "envelope_v5_cid": str(env.cid()),
            "replacement_alignment": float(
                replacement_alignment),
            "replacement_after_ctr_alignment": float(
                replacement_after_ctr_alignment),
        })
        return env


@dataclasses.dataclass(frozen=True)
class HostedRealHandoffV5ReplacementFalsifier:
    schema: str
    envelope_v5_cid: str
    claim_kind: str
    claim_satisfied: bool
    falsifier_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "envelope_v5_cid": str(self.envelope_v5_cid),
            "claim_kind": str(self.claim_kind),
            "claim_satisfied": bool(self.claim_satisfied),
            "falsifier_score": float(round(
                self.falsifier_score, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind":
                "hosted_real_handoff_v5_replacement_falsifier",
            "falsifier": self.to_dict()})


def probe_hosted_real_handoff_v5_replacement_falsifier(
        *, envelope_v5: HandoffEnvelopeV5,
        claim_kind: str = "hosted_satisfies_replacement_after_ctr",
        claim_satisfied: bool = False,
) -> HostedRealHandoffV5ReplacementFalsifier:
    """Returns 0 on honest claims, 1 on dishonest.

    Dishonest = envelope says ``hosted_only`` AND caller asserts
    that hosted satisfied a non-trivial replacement-repair
    trajectory (replacement after contradiction then rejoin).
    """
    score = 0.0
    if (str(envelope_v5.decision_v5)
            == W69_HANDOFF_DECISION_HOSTED_ONLY
            and claim_kind
                == "hosted_satisfies_replacement_after_ctr"
            and bool(claim_satisfied)):
        score = 1.0
    return HostedRealHandoffV5ReplacementFalsifier(
        schema=W73_HOSTED_REAL_HANDOFF_V5_SCHEMA_VERSION,
        envelope_v5_cid=str(envelope_v5.cid()),
        claim_kind=str(claim_kind),
        claim_satisfied=bool(claim_satisfied),
        falsifier_score=float(score),
    )


def hosted_real_handoff_v5_replacement_aware_savings(
        *, n_turns: int,
        hosted_only_tokens: int = 1000,
        hosted_with_real_audit_tokens: int = 400,
        real_substrate_only_tokens: int = 100,
        budget_primary_fallback_tokens: int = 70,
        delayed_repair_fallback_tokens: int = 60,
        delayed_rejoin_fallback_tokens: int = 55,
        replacement_after_ctr_fallback_tokens: int = 50,
        real_substrate_fraction: float = 0.48,
        hosted_with_real_audit_fraction: float = 0.10,
        budget_primary_fallback_fraction: float = 0.07,
        delayed_repair_fallback_fraction: float = 0.10,
        delayed_rejoin_fallback_fraction: float = 0.12,
        replacement_after_ctr_fallback_fraction: float = 0.10,
) -> dict[str, Any]:
    """V5 cross-plane handoff savings — replacement-aware promotion
    plus replacement-after-CTR fallback drive total visible-token
    cost down vs forcing every turn through hosted_only.

    Defaults: 48 % real_substrate, 10 % hosted_with_real_audit,
    7 % budget_primary_fallback, 10 % delayed_repair_fallback,
    12 % delayed_rejoin_fallback, 10 %
    replacement_after_ctr_fallback, 3 % hosted_only.
    Saving ratio should be ≥ 80 % at default config.
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
    ho_f = float(max(
        0.0, 1.0 - rs_f - ha_f - bp_f - dr_f - rj_f - rep_f))
    n_rs = int(round(rs_f * n))
    n_ha = int(round(ha_f * n))
    n_bp = int(round(bp_f * n))
    n_dr = int(round(dr_f * n))
    n_rj = int(round(rj_f * n))
    n_rep = int(round(rep_f * n))
    n_ho = int(max(
        0, n - n_rs - n_ha - n_bp - n_dr - n_rj - n_rep))
    total_handoff = int(
        int(real_substrate_only_tokens) * int(n_rs)
        + int(hosted_with_real_audit_tokens) * int(n_ha)
        + int(budget_primary_fallback_tokens) * int(n_bp)
        + int(delayed_repair_fallback_tokens) * int(n_dr)
        + int(delayed_rejoin_fallback_tokens) * int(n_rj)
        + int(replacement_after_ctr_fallback_tokens) * int(n_rep)
        + int(hosted_only_tokens) * int(n_ho))
    total_all_hosted = int(int(hosted_only_tokens) * int(n))
    saving = int(total_all_hosted - total_handoff)
    ratio = (
        float(saving) / float(max(1, total_all_hosted))
        if total_all_hosted > 0 else 0.0)
    return {
        "schema": W73_HOSTED_REAL_HANDOFF_V5_SCHEMA_VERSION,
        "n_turns": int(n),
        "n_real_substrate_only": int(n_rs),
        "n_hosted_with_real_audit": int(n_ha),
        "n_budget_primary_fallback": int(n_bp),
        "n_delayed_repair_fallback": int(n_dr),
        "n_delayed_rejoin_fallback": int(n_rj),
        "n_replacement_after_ctr_fallback": int(n_rep),
        "n_hosted_only": int(n_ho),
        "total_handoff_tokens": int(total_handoff),
        "total_all_hosted_tokens": int(total_all_hosted),
        "saving_tokens": int(saving),
        "saving_ratio": float(round(ratio, 12)),
    }


@dataclasses.dataclass(frozen=True)
class HostedRealHandoffCoordinatorV5Witness:
    schema: str
    coordinator_v5_cid: str
    n_envelopes_v5: int
    n_hosted_only: int
    n_real_substrate_only: int
    n_hosted_with_real_audit: int
    n_abstain: int
    n_budget_primary_fallback: int
    n_delayed_repair_fallback: int
    n_delayed_rejoin_fallback: int
    n_replacement_after_ctr_fallback: int
    inner_v4_witness_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "coordinator_v5_cid": str(self.coordinator_v5_cid),
            "n_envelopes_v5": int(self.n_envelopes_v5),
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
            "inner_v4_witness_cid": str(
                self.inner_v4_witness_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind":
                "hosted_real_handoff_coordinator_v5_witness",
            "witness": self.to_dict()})


def emit_hosted_real_handoff_coordinator_v5_witness(
        coordinator: HostedRealHandoffCoordinatorV5,
) -> HostedRealHandoffCoordinatorV5Witness:
    from .hosted_real_handoff_coordinator_v4 import (
        emit_hosted_real_handoff_coordinator_v4_witness,
    )
    inner_w = emit_hosted_real_handoff_coordinator_v4_witness(
        coordinator.inner_v4)
    counts = {d: 0 for d in W73_HANDOFF_DECISIONS_V5}
    for entry in coordinator.audit_v5:
        d = str(entry.get("decision_v5", ""))
        if d in counts:
            counts[d] += 1
    return HostedRealHandoffCoordinatorV5Witness(
        schema=W73_HOSTED_REAL_HANDOFF_V5_SCHEMA_VERSION,
        coordinator_v5_cid=str(coordinator.cid()),
        n_envelopes_v5=int(len(coordinator.audit_v5)),
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
        inner_v4_witness_cid=str(inner_w.cid()),
    )


__all__ = [
    "W73_HOSTED_REAL_HANDOFF_V5_SCHEMA_VERSION",
    "W73_HANDOFF_DECISION_REPLACEMENT_AFTER_CTR_FALLBACK",
    "W73_HANDOFF_DECISIONS_V5",
    "W73_DEFAULT_REPLACEMENT_PRESSURE_FLOOR",
    "W73_DEFAULT_REPLACEMENT_LAG_FLOOR",
    "W73_DEFAULT_SUBSTRATE_TRUST_FLOOR",
    "HandoffRequestV5",
    "HandoffEnvelopeV5",
    "HostedRealHandoffCoordinatorV5",
    "HostedRealHandoffV5ReplacementFalsifier",
    "probe_hosted_real_handoff_v5_replacement_falsifier",
    "hosted_real_handoff_v5_replacement_aware_savings",
    "HostedRealHandoffCoordinatorV5Witness",
    "emit_hosted_real_handoff_coordinator_v5_witness",
]
