"""W75 H6 — Hosted ↔ Real Substrate Handoff Coordinator V7.

The load-bearing W75 **compound-chain-aware** Plane A↔B bridge.
W74's V6 handoff coordinator added compound-aware promotion +
compound-repair-after-DRTR fallback on top of V5's replacement-
aware scoring. V7 adds:

* **Compound-chain-aware promotion** — any request with
  ``compound_chain_pressure >= compound_chain_pressure_floor`` AND
  substrate trust ≥ floor is promoted to ``real_substrate_only``
  with ``compound_chain_alignment = 1.0``, regardless of the V6
  compound-aware decision (and only when other pressures are not
  yet promoting).
* **Compound-repair-after-RTR fallback** — a new decision label
  ``compound_repair_after_replacement_then_rejoin_fallback`` fires
  when hosted is cheaper but the caller-declared
  ``compound_chain_repair_trajectory_cid`` requires Plane B
  (compound-chain-window > compound-chain-window-floor AND
  substrate trust > floor) AND the request did not already promote
  to restart-aware, delayed-repair-fallback, delayed-rejoin-
  fallback, replacement-after-CTR-fallback, or compound-repair-
  fallback paths.
* **Compound-chain-pressure falsifier** — extends V6 to a
  compound-repair-after-RTR variant: rejects hosted-only
  envelopes that claim to satisfy a non-trivial compound-chain-
  repair trajectory.

Honest scope (W75)
------------------

* The coordinator V7 does NOT pierce the hosted substrate
  boundary.
* ``W75-L-HANDOFF-V7-NOT-CROSSING-WALL-CAP`` — V7 preserves the
  W74 boundary as a content-addressed invariant.
* Determinism on (request V7 CID, registry CID, V20 substrate
  self-checksum CID, declared visible-token budget, restart
  pressure, rejoin pressure, replacement pressure, compound
  pressure, compound-chain pressure, compound-chain-repair
  trajectory CID).
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
    W73_HANDOFF_DECISION_REPLACEMENT_AFTER_CTR_FALLBACK,
)
from .hosted_real_handoff_coordinator_v6 import (
    HandoffEnvelopeV6, HandoffRequestV6,
    HostedRealHandoffCoordinatorV6,
    W74_HANDOFF_DECISION_COMPOUND_REPAIR_FALLBACK,
)
from .hosted_real_substrate_boundary_v8 import (
    HostedRealSubstrateBoundaryV8,
    build_default_hosted_real_substrate_boundary_v8,
)
from .hosted_router_controller import HostedRoutingDecision
from .tiny_substrate_v3 import _sha256_hex


W75_HOSTED_REAL_HANDOFF_V7_SCHEMA_VERSION: str = (
    "coordpy.hosted_real_handoff_coordinator_v7.v1")

W75_HANDOFF_DECISION_COMPOUND_CHAIN_REPAIR_FALLBACK: str = (
    "compound_repair_after_replacement_then_rejoin_fallback")
W75_HANDOFF_DECISIONS_V7: tuple[str, ...] = (
    W69_HANDOFF_DECISION_HOSTED_ONLY,
    W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY,
    W69_HANDOFF_DECISION_HOSTED_WITH_REAL_AUDIT,
    W69_HANDOFF_DECISION_ABSTAIN,
    W70_HANDOFF_DECISION_BUDGET_PRIMARY_FALLBACK,
    W71_HANDOFF_DECISION_DELAYED_REPAIR_FALLBACK,
    W72_HANDOFF_DECISION_DELAYED_REJOIN_AFTER_RESTART_FALLBACK,
    W73_HANDOFF_DECISION_REPLACEMENT_AFTER_CTR_FALLBACK,
    W74_HANDOFF_DECISION_COMPOUND_REPAIR_FALLBACK,
    W75_HANDOFF_DECISION_COMPOUND_CHAIN_REPAIR_FALLBACK,
)

W75_DEFAULT_COMPOUND_CHAIN_PRESSURE_FLOOR: float = 0.5
W75_DEFAULT_COMPOUND_CHAIN_WINDOW_FLOOR: int = 1
W75_DEFAULT_SUBSTRATE_TRUST_FLOOR: float = 0.5


@dataclasses.dataclass(frozen=True)
class HandoffRequestV7:
    inner_v6: HandoffRequestV6
    compound_chain_pressure: float = 0.0
    compound_chain_repair_trajectory_cid: str = ""
    compound_chain_window_turns: int = 0
    expected_substrate_trust_v7: float = 0.7

    @property
    def request_cid(self) -> str:
        return self.inner_v6.request_cid

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W75_HOSTED_REAL_HANDOFF_V7_SCHEMA_VERSION,
            "kind": "handoff_request_v7",
            "inner_v6_cid": str(self.inner_v6.cid()),
            "compound_chain_pressure": float(round(
                self.compound_chain_pressure, 12)),
            "compound_chain_repair_trajectory_cid": str(
                self.compound_chain_repair_trajectory_cid),
            "compound_chain_window_turns": int(
                self.compound_chain_window_turns),
            "expected_substrate_trust_v7": float(round(
                self.expected_substrate_trust_v7, 12)),
        })


@dataclasses.dataclass(frozen=True)
class HandoffEnvelopeV7:
    schema: str
    inner_v6: HandoffEnvelopeV6
    decision_v7: str
    plane_v7: str
    compound_chain_alignment: float
    compound_chain_repair_rtr_alignment: float
    compound_chain_repair_rtr_fallback_active: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "inner_v6_cid": str(self.inner_v6.cid()),
            "decision_v7": str(self.decision_v7),
            "plane_v7": str(self.plane_v7),
            "compound_chain_alignment": float(round(
                self.compound_chain_alignment, 12)),
            "compound_chain_repair_rtr_alignment": float(round(
                self.compound_chain_repair_rtr_alignment, 12)),
            "compound_chain_repair_rtr_fallback_active": bool(
                self.compound_chain_repair_rtr_fallback_active),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_real_handoff_envelope_v7",
            "envelope": self.to_dict()})


@dataclasses.dataclass
class HostedRealHandoffCoordinatorV7:
    inner_v6: HostedRealHandoffCoordinatorV6 = dataclasses.field(
        default_factory=HostedRealHandoffCoordinatorV6)
    boundary_v8: HostedRealSubstrateBoundaryV8 = (
        dataclasses.field(
            default_factory=(
                build_default_hosted_real_substrate_boundary_v8)))
    compound_chain_pressure_floor: float = (
        W75_DEFAULT_COMPOUND_CHAIN_PRESSURE_FLOOR)
    compound_chain_window_floor: int = (
        W75_DEFAULT_COMPOUND_CHAIN_WINDOW_FLOOR)
    substrate_trust_floor: float = (
        W75_DEFAULT_SUBSTRATE_TRUST_FLOOR)
    audit_v7: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W75_HOSTED_REAL_HANDOFF_V7_SCHEMA_VERSION,
            "kind": "hosted_real_handoff_coordinator_v7",
            "inner_v6_cid": str(self.inner_v6.cid()),
            "boundary_v8_cid": str(self.boundary_v8.cid()),
            "compound_chain_pressure_floor": float(round(
                self.compound_chain_pressure_floor, 12)),
            "compound_chain_window_floor": int(
                self.compound_chain_window_floor),
            "substrate_trust_floor": float(round(
                self.substrate_trust_floor, 12)),
        })

    def decide_v7(
            self, *, req_v7: HandoffRequestV7,
            hosted_decision: HostedRoutingDecision | None = None,
            substrate_self_checksum_cid: str = "",
    ) -> HandoffEnvelopeV7:
        v6_env = self.inner_v6.decide_v6(
            req_v6=req_v7.inner_v6,
            hosted_decision=hosted_decision,
            substrate_self_checksum_cid=str(
                substrate_self_checksum_cid))
        decision_v7 = str(v6_env.decision_v6)
        plane_v7 = str(v6_env.plane_v6)
        compound_chain_alignment = 0.0
        compound_chain_repair_rtr_alignment = 0.0
        compound_chain_repair_rtr_fallback_active = False
        # Compound-chain-aware promotion — fires when V6 stayed on
        # hosted_only AND compound-chain pressure is above floor.
        if (decision_v7 == W69_HANDOFF_DECISION_HOSTED_ONLY
                and float(req_v7.compound_chain_pressure)
                    >= float(self.compound_chain_pressure_floor)
                and float(req_v7.expected_substrate_trust_v7)
                    >= float(self.substrate_trust_floor)):
            decision_v7 = (
                W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY)
            plane_v7 = "B"
            compound_chain_alignment = 1.0
        # Compound-repair-after-RTR fallback — fires when the
        # compound-chain-repair trajectory has a non-trivial window
        # AND substrate trust is high, AND no other V6 fallback
        # fired.
        elif (
                str(req_v7.compound_chain_repair_trajectory_cid)
                and int(req_v7.compound_chain_window_turns)
                    >= int(self.compound_chain_window_floor)
                and float(req_v7.expected_substrate_trust_v7)
                    >= float(self.substrate_trust_floor)
                and decision_v7
                    == W69_HANDOFF_DECISION_HOSTED_ONLY):
            decision_v7 = (
                W75_HANDOFF_DECISION_COMPOUND_CHAIN_REPAIR_FALLBACK)
            plane_v7 = "A-chain-fallback"
            compound_chain_repair_rtr_fallback_active = True
            compound_chain_repair_rtr_alignment = 1.0
        # Honest alignment when no fallback fires.
        if (decision_v7
                in (W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY,
                    W69_HANDOFF_DECISION_HOSTED_WITH_REAL_AUDIT,
                    W75_HANDOFF_DECISION_COMPOUND_CHAIN_REPAIR_FALLBACK)
                and float(req_v7.compound_chain_pressure) > 0.0):
            compound_chain_alignment = max(
                compound_chain_alignment, 1.0)
        # Compound-chain-repair-after-RTR alignment: substrate /
        # audit / chain-fallback plane is the only plane that can
        # honestly satisfy a non-empty compound-chain-repair
        # trajectory.
        if str(req_v7.compound_chain_repair_trajectory_cid):
            if decision_v7 in (
                    W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY,
                    W69_HANDOFF_DECISION_HOSTED_WITH_REAL_AUDIT,
                    W75_HANDOFF_DECISION_COMPOUND_CHAIN_REPAIR_FALLBACK):
                compound_chain_repair_rtr_alignment = 1.0
            else:
                compound_chain_repair_rtr_alignment = 0.0
        env = HandoffEnvelopeV7(
            schema=W75_HOSTED_REAL_HANDOFF_V7_SCHEMA_VERSION,
            inner_v6=v6_env,
            decision_v7=str(decision_v7),
            plane_v7=str(plane_v7),
            compound_chain_alignment=float(
                compound_chain_alignment),
            compound_chain_repair_rtr_alignment=float(
                compound_chain_repair_rtr_alignment),
            compound_chain_repair_rtr_fallback_active=bool(
                compound_chain_repair_rtr_fallback_active),
        )
        self.audit_v7.append({
            "request_cid": str(req_v7.request_cid),
            "decision_v7": str(decision_v7),
            "plane_v7": str(plane_v7),
            "envelope_v7_cid": str(env.cid()),
            "compound_chain_alignment": float(
                compound_chain_alignment),
            "compound_chain_repair_rtr_alignment": float(
                compound_chain_repair_rtr_alignment),
        })
        return env


@dataclasses.dataclass(frozen=True)
class HostedRealHandoffV7CompoundChainFalsifier:
    schema: str
    envelope_v7_cid: str
    claim_kind: str
    claim_satisfied: bool
    falsifier_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "envelope_v7_cid": str(self.envelope_v7_cid),
            "claim_kind": str(self.claim_kind),
            "claim_satisfied": bool(self.claim_satisfied),
            "falsifier_score": float(round(
                self.falsifier_score, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind":
                "hosted_real_handoff_v7_compound_chain_falsifier",
            "falsifier": self.to_dict()})


def probe_hosted_real_handoff_v7_compound_chain_falsifier(
        *, envelope_v7: HandoffEnvelopeV7,
        claim_kind: str = (
            "hosted_satisfies_compound_chain_repair_rtr"),
        claim_satisfied: bool = False,
) -> HostedRealHandoffV7CompoundChainFalsifier:
    """Returns 0 on honest claims, 1 on dishonest.

    Dishonest = envelope says ``hosted_only`` AND caller asserts
    that hosted satisfied a non-trivial compound-chain-repair
    trajectory.
    """
    score = 0.0
    if (str(envelope_v7.decision_v7)
            == W69_HANDOFF_DECISION_HOSTED_ONLY
            and claim_kind
                == "hosted_satisfies_compound_chain_repair_rtr"
            and bool(claim_satisfied)):
        score = 1.0
    return HostedRealHandoffV7CompoundChainFalsifier(
        schema=W75_HOSTED_REAL_HANDOFF_V7_SCHEMA_VERSION,
        envelope_v7_cid=str(envelope_v7.cid()),
        claim_kind=str(claim_kind),
        claim_satisfied=bool(claim_satisfied),
        falsifier_score=float(score),
    )


def hosted_real_handoff_v7_compound_chain_aware_savings(
        *, n_turns: int,
        hosted_only_tokens: int = 1000,
        hosted_with_real_audit_tokens: int = 400,
        real_substrate_only_tokens: int = 100,
        budget_primary_fallback_tokens: int = 70,
        delayed_repair_fallback_tokens: int = 60,
        delayed_rejoin_fallback_tokens: int = 55,
        replacement_after_ctr_fallback_tokens: int = 50,
        compound_repair_fallback_tokens: int = 45,
        compound_chain_repair_fallback_tokens: int = 40,
        real_substrate_fraction: float = 0.46,
        hosted_with_real_audit_fraction: float = 0.10,
        budget_primary_fallback_fraction: float = 0.06,
        delayed_repair_fallback_fraction: float = 0.07,
        delayed_rejoin_fallback_fraction: float = 0.08,
        replacement_after_ctr_fallback_fraction: float = 0.07,
        compound_repair_fallback_fraction: float = 0.07,
        compound_chain_repair_fallback_fraction: float = 0.07,
) -> dict[str, Any]:
    """V7 cross-plane handoff savings — compound-chain-aware
    promotion plus compound-repair-after-RTR fallback drive total
    visible-token cost down vs forcing every turn through
    hosted_only.

    Defaults: 46 % real_substrate, 10 % hosted_with_real_audit,
    6 % budget_primary_fallback, 7 % delayed_repair_fallback,
    8 % delayed_rejoin_fallback, 7 %
    replacement_after_ctr_fallback, 7 %
    compound_repair_fallback, 7 % compound_chain_repair_fallback,
    2 % hosted_only.
    Saving ratio should be ≥ 84 % at default config.
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
    ch_f = float(max(
        0.0, min(
            1.0 - rs_f - ha_f - bp_f - dr_f - rj_f - rep_f
            - cmp_f,
            compound_chain_repair_fallback_fraction)))
    ho_f = float(max(
        0.0,
        1.0 - rs_f - ha_f - bp_f - dr_f - rj_f - rep_f
        - cmp_f - ch_f))
    n_rs = int(round(rs_f * n))
    n_ha = int(round(ha_f * n))
    n_bp = int(round(bp_f * n))
    n_dr = int(round(dr_f * n))
    n_rj = int(round(rj_f * n))
    n_rep = int(round(rep_f * n))
    n_cmp = int(round(cmp_f * n))
    n_ch = int(round(ch_f * n))
    n_ho = int(max(
        0,
        n - n_rs - n_ha - n_bp - n_dr - n_rj - n_rep - n_cmp
        - n_ch))
    total_handoff = int(
        int(real_substrate_only_tokens) * int(n_rs)
        + int(hosted_with_real_audit_tokens) * int(n_ha)
        + int(budget_primary_fallback_tokens) * int(n_bp)
        + int(delayed_repair_fallback_tokens) * int(n_dr)
        + int(delayed_rejoin_fallback_tokens) * int(n_rj)
        + int(replacement_after_ctr_fallback_tokens) * int(n_rep)
        + int(compound_repair_fallback_tokens) * int(n_cmp)
        + int(compound_chain_repair_fallback_tokens) * int(n_ch)
        + int(hosted_only_tokens) * int(n_ho))
    total_all_hosted = int(int(hosted_only_tokens) * int(n))
    saving = int(total_all_hosted - total_handoff)
    ratio = (
        float(saving) / float(max(1, total_all_hosted))
        if total_all_hosted > 0 else 0.0)
    return {
        "schema": W75_HOSTED_REAL_HANDOFF_V7_SCHEMA_VERSION,
        "n_turns": int(n),
        "n_real_substrate_only": int(n_rs),
        "n_hosted_with_real_audit": int(n_ha),
        "n_budget_primary_fallback": int(n_bp),
        "n_delayed_repair_fallback": int(n_dr),
        "n_delayed_rejoin_fallback": int(n_rj),
        "n_replacement_after_ctr_fallback": int(n_rep),
        "n_compound_repair_fallback": int(n_cmp),
        "n_compound_chain_repair_fallback": int(n_ch),
        "n_hosted_only": int(n_ho),
        "total_handoff_tokens": int(total_handoff),
        "total_all_hosted_tokens": int(total_all_hosted),
        "saving_tokens": int(saving),
        "saving_ratio": float(round(ratio, 12)),
    }


@dataclasses.dataclass(frozen=True)
class HostedRealHandoffCoordinatorV7Witness:
    schema: str
    coordinator_v7_cid: str
    n_envelopes_v7: int
    n_hosted_only: int
    n_real_substrate_only: int
    n_hosted_with_real_audit: int
    n_abstain: int
    n_budget_primary_fallback: int
    n_delayed_repair_fallback: int
    n_delayed_rejoin_fallback: int
    n_replacement_after_ctr_fallback: int
    n_compound_repair_fallback: int
    n_compound_chain_repair_fallback: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "coordinator_v7_cid": str(self.coordinator_v7_cid),
            "n_envelopes_v7": int(self.n_envelopes_v7),
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
            "n_compound_chain_repair_fallback": int(
                self.n_compound_chain_repair_fallback),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind":
                "hosted_real_handoff_coordinator_v7_witness",
            "witness": self.to_dict()})


def emit_hosted_real_handoff_coordinator_v7_witness(
        coordinator: HostedRealHandoffCoordinatorV7,
) -> HostedRealHandoffCoordinatorV7Witness:
    counts = {d: 0 for d in W75_HANDOFF_DECISIONS_V7}
    for entry in coordinator.audit_v7:
        d = str(entry.get("decision_v7", ""))
        if d in counts:
            counts[d] += 1
    return HostedRealHandoffCoordinatorV7Witness(
        schema=W75_HOSTED_REAL_HANDOFF_V7_SCHEMA_VERSION,
        coordinator_v7_cid=str(coordinator.cid()),
        n_envelopes_v7=int(len(coordinator.audit_v7)),
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
        n_compound_chain_repair_fallback=int(
            counts[
                W75_HANDOFF_DECISION_COMPOUND_CHAIN_REPAIR_FALLBACK]),
    )


__all__ = [
    "W75_HOSTED_REAL_HANDOFF_V7_SCHEMA_VERSION",
    "W75_HANDOFF_DECISION_COMPOUND_CHAIN_REPAIR_FALLBACK",
    "W75_HANDOFF_DECISIONS_V7",
    "W75_DEFAULT_COMPOUND_CHAIN_PRESSURE_FLOOR",
    "W75_DEFAULT_COMPOUND_CHAIN_WINDOW_FLOOR",
    "W75_DEFAULT_SUBSTRATE_TRUST_FLOOR",
    "HandoffRequestV7",
    "HandoffEnvelopeV7",
    "HostedRealHandoffCoordinatorV7",
    "HostedRealHandoffV7CompoundChainFalsifier",
    "probe_hosted_real_handoff_v7_compound_chain_falsifier",
    "hosted_real_handoff_v7_compound_chain_aware_savings",
    "HostedRealHandoffCoordinatorV7Witness",
    "emit_hosted_real_handoff_coordinator_v7_witness",
]
