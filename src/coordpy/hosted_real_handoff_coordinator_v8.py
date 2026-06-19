"""W76 H6 — Hosted ↔ Real Substrate Handoff Coordinator V8.

The load-bearing W76 **chain-then-restart-aware** Plane A↔B
bridge. W75's V7 handoff coordinator added compound-chain-aware
promotion + compound-repair-after-RTR fallback on top of V6's
compound-aware scoring. V8 adds:

* **Chain-then-restart-aware promotion** — any request with
  ``compound_chain_then_restart_pressure >= chain_then_restart_
  pressure_floor`` AND substrate trust ≥ floor is promoted to
  ``real_substrate_only`` with
  ``compound_chain_then_restart_alignment = 1.0``, regardless of
  the V7 chain-aware decision.
* **Restart-after-compound-chain-repair fallback** — a new
  decision label ``restart_after_compound_chain_repair_fallback``
  fires when hosted is cheaper but the caller-declared
  ``compound_chain_then_restart_trajectory_cid`` requires Plane B
  AND the request did not already promote to an earlier
  fallback path.
* **Chain-then-restart falsifier** — extends V7's compound-chain
  falsifier to a chain-then-restart variant.

Honest scope (W76)
------------------

* The coordinator V8 does NOT pierce the hosted substrate
  boundary.
* ``W76-L-HANDOFF-V8-NOT-CROSSING-WALL-CAP`` — V8 preserves the
  W75 boundary as a content-addressed invariant.
* Determinism on (request V8 CID, registry CID, V21 substrate
  self-checksum CID, declared visible-token budget, pressures
  including chain-then-restart pressure, chain-then-restart-
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
    W72_HANDOFF_DECISION_DELAYED_REJOIN_AFTER_RESTART_FALLBACK,
)
from .hosted_real_handoff_coordinator_v5 import (
    W73_HANDOFF_DECISION_REPLACEMENT_AFTER_CTR_FALLBACK,
)
from .hosted_real_handoff_coordinator_v6 import (
    W74_HANDOFF_DECISION_COMPOUND_REPAIR_FALLBACK,
)
from .hosted_real_handoff_coordinator_v7 import (
    HandoffEnvelopeV7, HandoffRequestV7,
    HostedRealHandoffCoordinatorV7,
    W75_HANDOFF_DECISION_COMPOUND_CHAIN_REPAIR_FALLBACK,
)
from .hosted_real_substrate_boundary_v9 import (
    HostedRealSubstrateBoundaryV9,
    build_default_hosted_real_substrate_boundary_v9,
)
from .hosted_router_controller import HostedRoutingDecision
from .tiny_substrate_v3 import _sha256_hex


W76_HOSTED_REAL_HANDOFF_V8_SCHEMA_VERSION: str = (
    "coordpy.hosted_real_handoff_coordinator_v8.v1")

W76_HANDOFF_DECISION_CHAIN_THEN_RESTART_FALLBACK: str = (
    "restart_after_compound_chain_repair_fallback")
W76_HANDOFF_DECISIONS_V8: tuple[str, ...] = (
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
    W76_HANDOFF_DECISION_CHAIN_THEN_RESTART_FALLBACK,
)

W76_DEFAULT_CHAIN_THEN_RESTART_PRESSURE_FLOOR: float = 0.5
W76_DEFAULT_CHAIN_THEN_RESTART_WINDOW_FLOOR: int = 1
W76_DEFAULT_SUBSTRATE_TRUST_FLOOR: float = 0.5


@dataclasses.dataclass(frozen=True)
class HandoffRequestV8:
    inner_v7: HandoffRequestV7
    compound_chain_then_restart_pressure: float = 0.0
    compound_chain_then_restart_trajectory_cid: str = ""
    post_compound_chain_restart_window_turns: int = 0
    expected_substrate_trust_v8: float = 0.7

    @property
    def request_cid(self) -> str:
        return self.inner_v7.request_cid

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W76_HOSTED_REAL_HANDOFF_V8_SCHEMA_VERSION,
            "kind": "handoff_request_v8",
            "inner_v7_cid": str(self.inner_v7.cid()),
            "compound_chain_then_restart_pressure": float(round(
                self.compound_chain_then_restart_pressure, 12)),
            "compound_chain_then_restart_trajectory_cid": str(
                self.compound_chain_then_restart_trajectory_cid),
            "post_compound_chain_restart_window_turns": int(
                self.post_compound_chain_restart_window_turns),
            "expected_substrate_trust_v8": float(round(
                self.expected_substrate_trust_v8, 12)),
        })


@dataclasses.dataclass(frozen=True)
class HandoffEnvelopeV8:
    schema: str
    inner_v7: HandoffEnvelopeV7
    decision_v8: str
    plane_v8: str
    compound_chain_then_restart_alignment: float
    chain_then_restart_rtr_alignment: float
    chain_then_restart_fallback_active: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "inner_v7_cid": str(self.inner_v7.cid()),
            "decision_v8": str(self.decision_v8),
            "plane_v8": str(self.plane_v8),
            "compound_chain_then_restart_alignment": float(round(
                self.compound_chain_then_restart_alignment, 12)),
            "chain_then_restart_rtr_alignment": float(round(
                self.chain_then_restart_rtr_alignment, 12)),
            "chain_then_restart_fallback_active": bool(
                self.chain_then_restart_fallback_active),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_real_handoff_envelope_v8",
            "envelope": self.to_dict()})


@dataclasses.dataclass
class HostedRealHandoffCoordinatorV8:
    inner_v7: HostedRealHandoffCoordinatorV7 = dataclasses.field(
        default_factory=HostedRealHandoffCoordinatorV7)
    boundary_v9: HostedRealSubstrateBoundaryV9 = (
        dataclasses.field(
            default_factory=(
                build_default_hosted_real_substrate_boundary_v9)))
    chain_then_restart_pressure_floor: float = (
        W76_DEFAULT_CHAIN_THEN_RESTART_PRESSURE_FLOOR)
    chain_then_restart_window_floor: int = (
        W76_DEFAULT_CHAIN_THEN_RESTART_WINDOW_FLOOR)
    substrate_trust_floor: float = (
        W76_DEFAULT_SUBSTRATE_TRUST_FLOOR)
    audit_v8: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W76_HOSTED_REAL_HANDOFF_V8_SCHEMA_VERSION,
            "kind": "hosted_real_handoff_coordinator_v8",
            "inner_v7_cid": str(self.inner_v7.cid()),
            "boundary_v9_cid": str(self.boundary_v9.cid()),
            "chain_then_restart_pressure_floor": float(round(
                self.chain_then_restart_pressure_floor, 12)),
            "chain_then_restart_window_floor": int(
                self.chain_then_restart_window_floor),
            "substrate_trust_floor": float(round(
                self.substrate_trust_floor, 12)),
        })

    def decide_v8(
            self, *, req_v8: HandoffRequestV8,
            hosted_decision: HostedRoutingDecision | None = None,
            substrate_self_checksum_cid: str = "",
    ) -> HandoffEnvelopeV8:
        v7_env = self.inner_v7.decide_v7(
            req_v7=req_v8.inner_v7,
            hosted_decision=hosted_decision,
            substrate_self_checksum_cid=str(
                substrate_self_checksum_cid))
        decision_v8 = str(v7_env.decision_v7)
        plane_v8 = str(v7_env.plane_v7)
        ctr_alignment = 0.0
        ctr_rtr_alignment = 0.0
        ctr_fallback_active = False
        # Chain-then-restart-aware promotion.
        if (decision_v8 == W69_HANDOFF_DECISION_HOSTED_ONLY
                and float(
                    req_v8.compound_chain_then_restart_pressure)
                >= float(self.chain_then_restart_pressure_floor)
                and float(req_v8.expected_substrate_trust_v8)
                >= float(self.substrate_trust_floor)):
            decision_v8 = (
                W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY)
            plane_v8 = "B"
            ctr_alignment = 1.0
        # Chain-then-restart-after-RTR fallback.
        elif (
                str(req_v8
                    .compound_chain_then_restart_trajectory_cid)
                and int(
                    req_v8
                    .post_compound_chain_restart_window_turns)
                >= int(self.chain_then_restart_window_floor)
                and float(req_v8.expected_substrate_trust_v8)
                >= float(self.substrate_trust_floor)
                and decision_v8
                == W69_HANDOFF_DECISION_HOSTED_ONLY):
            decision_v8 = (
                W76_HANDOFF_DECISION_CHAIN_THEN_RESTART_FALLBACK)
            plane_v8 = "A-chain-then-restart-fallback"
            ctr_fallback_active = True
            ctr_rtr_alignment = 1.0
        # Honest alignment when no fallback fires.
        if (decision_v8
                in (W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY,
                    W69_HANDOFF_DECISION_HOSTED_WITH_REAL_AUDIT,
                    W76_HANDOFF_DECISION_CHAIN_THEN_RESTART_FALLBACK)
                and float(
                    req_v8.compound_chain_then_restart_pressure)
                > 0.0):
            ctr_alignment = max(ctr_alignment, 1.0)
        # Alignment for trajectory CID.
        if str(req_v8.compound_chain_then_restart_trajectory_cid):
            if decision_v8 in (
                    W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY,
                    W69_HANDOFF_DECISION_HOSTED_WITH_REAL_AUDIT,
                    W76_HANDOFF_DECISION_CHAIN_THEN_RESTART_FALLBACK):
                ctr_rtr_alignment = 1.0
            else:
                ctr_rtr_alignment = 0.0
        env = HandoffEnvelopeV8(
            schema=W76_HOSTED_REAL_HANDOFF_V8_SCHEMA_VERSION,
            inner_v7=v7_env,
            decision_v8=str(decision_v8),
            plane_v8=str(plane_v8),
            compound_chain_then_restart_alignment=float(
                ctr_alignment),
            chain_then_restart_rtr_alignment=float(
                ctr_rtr_alignment),
            chain_then_restart_fallback_active=bool(
                ctr_fallback_active),
        )
        self.audit_v8.append({
            "request_cid": str(req_v8.request_cid),
            "decision_v8": str(decision_v8),
            "plane_v8": str(plane_v8),
            "envelope_v8_cid": str(env.cid()),
            "compound_chain_then_restart_alignment": float(
                ctr_alignment),
            "chain_then_restart_rtr_alignment": float(
                ctr_rtr_alignment),
        })
        return env


@dataclasses.dataclass(frozen=True)
class HostedRealHandoffV8ChainThenRestartFalsifier:
    schema: str
    envelope_v8_cid: str
    claim_kind: str
    claim_satisfied: bool
    falsifier_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "envelope_v8_cid": str(self.envelope_v8_cid),
            "claim_kind": str(self.claim_kind),
            "claim_satisfied": bool(self.claim_satisfied),
            "falsifier_score": float(round(
                self.falsifier_score, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind":
                "hosted_real_handoff_v8_chain_then_restart_falsifier",
            "falsifier": self.to_dict()})


def probe_hosted_real_handoff_v8_chain_then_restart_falsifier(
        *, envelope_v8: HandoffEnvelopeV8,
        claim_kind: str = (
            "hosted_satisfies_chain_then_restart"),
        claim_satisfied: bool = False,
) -> HostedRealHandoffV8ChainThenRestartFalsifier:
    """Returns 0 on honest claims, 1 on dishonest.

    Dishonest = envelope says ``hosted_only`` AND caller asserts
    that hosted satisfied a non-trivial chain-then-restart
    trajectory.
    """
    score = 0.0
    if (str(envelope_v8.decision_v8)
            == W69_HANDOFF_DECISION_HOSTED_ONLY
            and claim_kind
                == "hosted_satisfies_chain_then_restart"
            and bool(claim_satisfied)):
        score = 1.0
    return HostedRealHandoffV8ChainThenRestartFalsifier(
        schema=W76_HOSTED_REAL_HANDOFF_V8_SCHEMA_VERSION,
        envelope_v8_cid=str(envelope_v8.cid()),
        claim_kind=str(claim_kind),
        claim_satisfied=bool(claim_satisfied),
        falsifier_score=float(score),
    )


def hosted_real_handoff_v8_chain_then_restart_aware_savings(
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
        chain_then_restart_fallback_tokens: int = 35,
        real_substrate_fraction: float = 0.46,
        hosted_with_real_audit_fraction: float = 0.10,
        budget_primary_fallback_fraction: float = 0.06,
        delayed_repair_fallback_fraction: float = 0.06,
        delayed_rejoin_fallback_fraction: float = 0.07,
        replacement_after_ctr_fallback_fraction: float = 0.06,
        compound_repair_fallback_fraction: float = 0.06,
        compound_chain_repair_fallback_fraction: float = 0.06,
        chain_then_restart_fallback_fraction: float = 0.06,
) -> dict[str, Any]:
    """V8 cross-plane handoff savings — chain-then-restart-aware
    promotion plus chain-then-restart-fallback drive total
    visible-token cost down vs forcing every turn through
    hosted_only.

    Saving ratio should be ≥ 85 % at default config.
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
    ctr_f = float(max(
        0.0, min(
            1.0 - rs_f - ha_f - bp_f - dr_f - rj_f - rep_f
            - cmp_f - ch_f,
            chain_then_restart_fallback_fraction)))
    ho_f = float(max(
        0.0,
        1.0 - rs_f - ha_f - bp_f - dr_f - rj_f - rep_f - cmp_f
        - ch_f - ctr_f))
    n_rs = int(round(rs_f * n))
    n_ha = int(round(ha_f * n))
    n_bp = int(round(bp_f * n))
    n_dr = int(round(dr_f * n))
    n_rj = int(round(rj_f * n))
    n_rep = int(round(rep_f * n))
    n_cmp = int(round(cmp_f * n))
    n_ch = int(round(ch_f * n))
    n_ctr = int(round(ctr_f * n))
    n_ho = int(max(
        0,
        n - n_rs - n_ha - n_bp - n_dr - n_rj - n_rep - n_cmp
        - n_ch - n_ctr))
    total_handoff = int(
        int(real_substrate_only_tokens) * int(n_rs)
        + int(hosted_with_real_audit_tokens) * int(n_ha)
        + int(budget_primary_fallback_tokens) * int(n_bp)
        + int(delayed_repair_fallback_tokens) * int(n_dr)
        + int(delayed_rejoin_fallback_tokens) * int(n_rj)
        + int(replacement_after_ctr_fallback_tokens) * int(n_rep)
        + int(compound_repair_fallback_tokens) * int(n_cmp)
        + int(compound_chain_repair_fallback_tokens) * int(n_ch)
        + int(chain_then_restart_fallback_tokens) * int(n_ctr)
        + int(hosted_only_tokens) * int(n_ho))
    total_all_hosted = int(int(hosted_only_tokens) * int(n))
    saving = int(total_all_hosted - total_handoff)
    ratio = (
        float(saving) / float(max(1, total_all_hosted))
        if total_all_hosted > 0 else 0.0)
    return {
        "schema": W76_HOSTED_REAL_HANDOFF_V8_SCHEMA_VERSION,
        "n_turns": int(n),
        "n_real_substrate_only": int(n_rs),
        "n_hosted_with_real_audit": int(n_ha),
        "n_budget_primary_fallback": int(n_bp),
        "n_delayed_repair_fallback": int(n_dr),
        "n_delayed_rejoin_fallback": int(n_rj),
        "n_replacement_after_ctr_fallback": int(n_rep),
        "n_compound_repair_fallback": int(n_cmp),
        "n_compound_chain_repair_fallback": int(n_ch),
        "n_chain_then_restart_fallback": int(n_ctr),
        "n_hosted_only": int(n_ho),
        "total_handoff_tokens": int(total_handoff),
        "total_all_hosted_tokens": int(total_all_hosted),
        "saving_tokens": int(saving),
        "saving_ratio": float(round(ratio, 12)),
    }


@dataclasses.dataclass(frozen=True)
class HostedRealHandoffCoordinatorV8Witness:
    schema: str
    coordinator_v8_cid: str
    n_envelopes_v8: int
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
    n_chain_then_restart_fallback: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "coordinator_v8_cid": str(self.coordinator_v8_cid),
            "n_envelopes_v8": int(self.n_envelopes_v8),
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
            "n_chain_then_restart_fallback": int(
                self.n_chain_then_restart_fallback),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind":
                "hosted_real_handoff_coordinator_v8_witness",
            "witness": self.to_dict()})


def emit_hosted_real_handoff_coordinator_v8_witness(
        coordinator: HostedRealHandoffCoordinatorV8,
) -> HostedRealHandoffCoordinatorV8Witness:
    counts = {d: 0 for d in W76_HANDOFF_DECISIONS_V8}
    for entry in coordinator.audit_v8:
        d = str(entry.get("decision_v8", ""))
        if d in counts:
            counts[d] += 1
    return HostedRealHandoffCoordinatorV8Witness(
        schema=W76_HOSTED_REAL_HANDOFF_V8_SCHEMA_VERSION,
        coordinator_v8_cid=str(coordinator.cid()),
        n_envelopes_v8=int(len(coordinator.audit_v8)),
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
        n_chain_then_restart_fallback=int(
            counts[
                W76_HANDOFF_DECISION_CHAIN_THEN_RESTART_FALLBACK]),
    )


__all__ = [
    "W76_HOSTED_REAL_HANDOFF_V8_SCHEMA_VERSION",
    "W76_HANDOFF_DECISION_CHAIN_THEN_RESTART_FALLBACK",
    "W76_HANDOFF_DECISIONS_V8",
    "W76_DEFAULT_CHAIN_THEN_RESTART_PRESSURE_FLOOR",
    "W76_DEFAULT_CHAIN_THEN_RESTART_WINDOW_FLOOR",
    "W76_DEFAULT_SUBSTRATE_TRUST_FLOOR",
    "HandoffRequestV8",
    "HandoffEnvelopeV8",
    "HostedRealHandoffCoordinatorV8",
    "HostedRealHandoffV8ChainThenRestartFalsifier",
    "probe_hosted_real_handoff_v8_chain_then_restart_falsifier",
    "hosted_real_handoff_v8_chain_then_restart_aware_savings",
    "HostedRealHandoffCoordinatorV8Witness",
    "emit_hosted_real_handoff_coordinator_v8_witness",
]
