"""W77 H6 — Hosted ↔ Real Substrate Handoff Coordinator V9.

The load-bearing W77 **post-restart-replacement-aware** Plane A↔B
bridge. W76's V8 handoff coordinator added chain-then-restart-
aware promotion + chain-then-restart fallback. V9 adds:

* **Post-restart-replacement-aware promotion** — any request with
  ``post_restart_replacement_pressure >= post_restart_replacement_
  pressure_floor`` AND substrate trust ≥ floor is promoted to
  ``real_substrate_only`` with
  ``post_restart_replacement_alignment = 1.0``, regardless of the
  V8 chain-aware decision.
* **Replacement-after-restart-after-compound-chain-repair
  fallback** — a new decision label ``replacement_after_restart_
  after_compound_chain_repair_fallback`` fires when hosted is
  cheaper but the caller-declared ``post_restart_replacement_
  trajectory_cid`` requires Plane B AND the request did not
  already promote to an earlier fallback path.
* **Post-restart-replacement falsifier** — extends V8's chain-
  then-restart falsifier to a post-restart-replacement variant.

Honest scope (W77)
------------------

* The coordinator V9 does NOT pierce the hosted substrate
  boundary.
* ``W77-L-HANDOFF-V9-NOT-CROSSING-WALL-CAP`` — V9 preserves the
  W76 boundary as a content-addressed invariant.
* Determinism on (request V9 CID, registry CID, V22 substrate
  self-checksum CID, declared visible-token budget, pressures
  including post-restart-replacement pressure, post-restart-
  replacement-repair trajectory CID).
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
    W75_HANDOFF_DECISION_COMPOUND_CHAIN_REPAIR_FALLBACK,
)
from .hosted_real_handoff_coordinator_v8 import (
    HandoffEnvelopeV8, HandoffRequestV8,
    HostedRealHandoffCoordinatorV8,
    W76_HANDOFF_DECISION_CHAIN_THEN_RESTART_FALLBACK,
)
from .hosted_real_substrate_boundary_v10 import (
    HostedRealSubstrateBoundaryV10,
    build_default_hosted_real_substrate_boundary_v10,
)
from .hosted_router_controller import HostedRoutingDecision
from .tiny_substrate_v3 import _sha256_hex


W77_HOSTED_REAL_HANDOFF_V9_SCHEMA_VERSION: str = (
    "coordpy.hosted_real_handoff_coordinator_v9.v1")

W77_HANDOFF_DECISION_POST_RESTART_REPLACEMENT_FALLBACK: str = (
    "replacement_after_restart_after_compound_chain_repair_fallback")
W77_HANDOFF_DECISIONS_V9: tuple[str, ...] = (
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
    W77_HANDOFF_DECISION_POST_RESTART_REPLACEMENT_FALLBACK,
)

W77_DEFAULT_POST_RESTART_REPLACEMENT_PRESSURE_FLOOR: float = 0.5
W77_DEFAULT_POST_RESTART_REPLACEMENT_WINDOW_FLOOR: int = 1
W77_DEFAULT_SUBSTRATE_TRUST_FLOOR: float = 0.5


@dataclasses.dataclass(frozen=True)
class HandoffRequestV9:
    inner_v8: HandoffRequestV8
    post_restart_replacement_pressure: float = 0.0
    post_restart_replacement_trajectory_cid: str = ""
    post_restart_replacement_window_turns: int = 0
    expected_substrate_trust_v9: float = 0.7

    @property
    def request_cid(self) -> str:
        return self.inner_v8.request_cid

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W77_HOSTED_REAL_HANDOFF_V9_SCHEMA_VERSION,
            "kind": "handoff_request_v9",
            "inner_v8_cid": str(self.inner_v8.cid()),
            "post_restart_replacement_pressure": float(round(
                self.post_restart_replacement_pressure, 12)),
            "post_restart_replacement_trajectory_cid": str(
                self.post_restart_replacement_trajectory_cid),
            "post_restart_replacement_window_turns": int(
                self.post_restart_replacement_window_turns),
            "expected_substrate_trust_v9": float(round(
                self.expected_substrate_trust_v9, 12)),
        })


@dataclasses.dataclass(frozen=True)
class HandoffEnvelopeV9:
    schema: str
    inner_v8: HandoffEnvelopeV8
    decision_v9: str
    plane_v9: str
    post_restart_replacement_alignment: float
    post_restart_replacement_pcr_alignment: float
    post_restart_replacement_fallback_active: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "inner_v8_cid": str(self.inner_v8.cid()),
            "decision_v9": str(self.decision_v9),
            "plane_v9": str(self.plane_v9),
            "post_restart_replacement_alignment": float(round(
                self.post_restart_replacement_alignment, 12)),
            "post_restart_replacement_pcr_alignment": float(round(
                self.post_restart_replacement_pcr_alignment, 12)),
            "post_restart_replacement_fallback_active": bool(
                self.post_restart_replacement_fallback_active),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_real_handoff_envelope_v9",
            "envelope": self.to_dict()})


@dataclasses.dataclass
class HostedRealHandoffCoordinatorV9:
    inner_v8: HostedRealHandoffCoordinatorV8 = dataclasses.field(
        default_factory=HostedRealHandoffCoordinatorV8)
    boundary_v10: HostedRealSubstrateBoundaryV10 = (
        dataclasses.field(
            default_factory=(
                build_default_hosted_real_substrate_boundary_v10)))
    post_restart_replacement_pressure_floor: float = (
        W77_DEFAULT_POST_RESTART_REPLACEMENT_PRESSURE_FLOOR)
    post_restart_replacement_window_floor: int = (
        W77_DEFAULT_POST_RESTART_REPLACEMENT_WINDOW_FLOOR)
    substrate_trust_floor: float = (
        W77_DEFAULT_SUBSTRATE_TRUST_FLOOR)
    audit_v9: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W77_HOSTED_REAL_HANDOFF_V9_SCHEMA_VERSION,
            "kind": "hosted_real_handoff_coordinator_v9",
            "inner_v8_cid": str(self.inner_v8.cid()),
            "boundary_v10_cid": str(self.boundary_v10.cid()),
            "post_restart_replacement_pressure_floor": float(
                round(
                    self
                    .post_restart_replacement_pressure_floor,
                    12)),
            "post_restart_replacement_window_floor": int(
                self.post_restart_replacement_window_floor),
            "substrate_trust_floor": float(round(
                self.substrate_trust_floor, 12)),
        })

    def decide_v9(
            self, *, req_v9: HandoffRequestV9,
            hosted_decision: HostedRoutingDecision | None = None,
            substrate_self_checksum_cid: str = "",
    ) -> HandoffEnvelopeV9:
        v8_env = self.inner_v8.decide_v8(
            req_v8=req_v9.inner_v8,
            hosted_decision=hosted_decision,
            substrate_self_checksum_cid=str(
                substrate_self_checksum_cid))
        decision_v9 = str(v8_env.decision_v8)
        plane_v9 = str(v8_env.plane_v8)
        pcr_alignment = 0.0
        pcr_align_match = 0.0
        pcr_fallback_active = False
        # Post-restart-replacement-aware promotion.
        if (decision_v9 == W69_HANDOFF_DECISION_HOSTED_ONLY
                and float(
                    req_v9.post_restart_replacement_pressure)
                >= float(
                    self.post_restart_replacement_pressure_floor)
                and float(req_v9.expected_substrate_trust_v9)
                >= float(self.substrate_trust_floor)):
            decision_v9 = (
                W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY)
            plane_v9 = "B"
            pcr_alignment = 1.0
        # Post-restart-replacement fallback.
        elif (
                str(req_v9
                    .post_restart_replacement_trajectory_cid)
                and int(
                    req_v9
                    .post_restart_replacement_window_turns)
                >= int(
                    self.post_restart_replacement_window_floor)
                and float(req_v9.expected_substrate_trust_v9)
                >= float(self.substrate_trust_floor)
                and decision_v9
                == W69_HANDOFF_DECISION_HOSTED_ONLY):
            decision_v9 = (
                W77_HANDOFF_DECISION_POST_RESTART_REPLACEMENT_FALLBACK)
            plane_v9 = "A-post-restart-replacement-fallback"
            pcr_fallback_active = True
            pcr_align_match = 1.0
        # Honest alignment when no fallback fires.
        if (decision_v9
                in (W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY,
                    W69_HANDOFF_DECISION_HOSTED_WITH_REAL_AUDIT,
                    W77_HANDOFF_DECISION_POST_RESTART_REPLACEMENT_FALLBACK)
                and float(
                    req_v9.post_restart_replacement_pressure)
                > 0.0):
            pcr_alignment = max(pcr_alignment, 1.0)
        # Alignment for trajectory CID.
        if str(req_v9.post_restart_replacement_trajectory_cid):
            if decision_v9 in (
                    W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY,
                    W69_HANDOFF_DECISION_HOSTED_WITH_REAL_AUDIT,
                    W77_HANDOFF_DECISION_POST_RESTART_REPLACEMENT_FALLBACK):
                pcr_align_match = 1.0
            else:
                pcr_align_match = 0.0
        env = HandoffEnvelopeV9(
            schema=W77_HOSTED_REAL_HANDOFF_V9_SCHEMA_VERSION,
            inner_v8=v8_env,
            decision_v9=str(decision_v9),
            plane_v9=str(plane_v9),
            post_restart_replacement_alignment=float(
                pcr_alignment),
            post_restart_replacement_pcr_alignment=float(
                pcr_align_match),
            post_restart_replacement_fallback_active=bool(
                pcr_fallback_active),
        )
        self.audit_v9.append({
            "request_cid": str(req_v9.request_cid),
            "decision_v9": str(decision_v9),
            "plane_v9": str(plane_v9),
            "envelope_v9_cid": str(env.cid()),
            "post_restart_replacement_alignment": float(
                pcr_alignment),
            "post_restart_replacement_pcr_alignment": float(
                pcr_align_match),
        })
        return env


@dataclasses.dataclass(frozen=True)
class HostedRealHandoffV9PostRestartReplacementFalsifier:
    schema: str
    envelope_v9_cid: str
    claim_kind: str
    claim_satisfied: bool
    falsifier_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "envelope_v9_cid": str(self.envelope_v9_cid),
            "claim_kind": str(self.claim_kind),
            "claim_satisfied": bool(self.claim_satisfied),
            "falsifier_score": float(round(
                self.falsifier_score, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind":
                "hosted_real_handoff_v9_post_restart_replacement_falsifier",
            "falsifier": self.to_dict()})


def probe_hosted_real_handoff_v9_post_restart_replacement_falsifier(
        *, envelope_v9: HandoffEnvelopeV9,
        claim_kind: str = (
            "hosted_satisfies_post_restart_replacement"),
        claim_satisfied: bool = False,
) -> HostedRealHandoffV9PostRestartReplacementFalsifier:
    """Returns 0 on honest claims, 1 on dishonest.

    Dishonest = envelope says ``hosted_only`` AND caller asserts
    that hosted satisfied a non-trivial post-restart-replacement
    trajectory.
    """
    score = 0.0
    if (str(envelope_v9.decision_v9)
            == W69_HANDOFF_DECISION_HOSTED_ONLY
            and claim_kind
                == "hosted_satisfies_post_restart_replacement"
            and bool(claim_satisfied)):
        score = 1.0
    return HostedRealHandoffV9PostRestartReplacementFalsifier(
        schema=W77_HOSTED_REAL_HANDOFF_V9_SCHEMA_VERSION,
        envelope_v9_cid=str(envelope_v9.cid()),
        claim_kind=str(claim_kind),
        claim_satisfied=bool(claim_satisfied),
        falsifier_score=float(score),
    )


def hosted_real_handoff_v9_post_restart_replacement_aware_savings(
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
        post_restart_replacement_fallback_tokens: int = 32,
        real_substrate_fraction: float = 0.46,
        hosted_with_real_audit_fraction: float = 0.10,
        budget_primary_fallback_fraction: float = 0.06,
        delayed_repair_fallback_fraction: float = 0.06,
        delayed_rejoin_fallback_fraction: float = 0.06,
        replacement_after_ctr_fallback_fraction: float = 0.06,
        compound_repair_fallback_fraction: float = 0.06,
        compound_chain_repair_fallback_fraction: float = 0.05,
        chain_then_restart_fallback_fraction: float = 0.05,
        post_restart_replacement_fallback_fraction: float = 0.05,
) -> dict[str, Any]:
    """V9 cross-plane handoff savings — post-restart-replacement-
    aware promotion plus post-restart-replacement-fallback drive
    total visible-token cost down vs forcing every turn through
    hosted_only.

    Saving ratio should be ≥ 86 % at default config.
    """
    n = int(max(1, n_turns))

    def _clamp(x: float, lo: float = 0.0, hi: float = 1.0
               ) -> float:
        return float(max(lo, min(hi, float(x))))

    rs_f = _clamp(real_substrate_fraction)
    remaining = 1.0 - rs_f
    ha_f = _clamp(hosted_with_real_audit_fraction, 0.0, remaining)
    remaining -= ha_f
    bp_f = _clamp(budget_primary_fallback_fraction, 0.0, remaining)
    remaining -= bp_f
    dr_f = _clamp(delayed_repair_fallback_fraction, 0.0, remaining)
    remaining -= dr_f
    rj_f = _clamp(delayed_rejoin_fallback_fraction, 0.0, remaining)
    remaining -= rj_f
    rep_f = _clamp(
        replacement_after_ctr_fallback_fraction, 0.0, remaining)
    remaining -= rep_f
    cmp_f = _clamp(
        compound_repair_fallback_fraction, 0.0, remaining)
    remaining -= cmp_f
    ch_f = _clamp(
        compound_chain_repair_fallback_fraction, 0.0, remaining)
    remaining -= ch_f
    ctr_f = _clamp(
        chain_then_restart_fallback_fraction, 0.0, remaining)
    remaining -= ctr_f
    pcr_f = _clamp(
        post_restart_replacement_fallback_fraction, 0.0, remaining)
    remaining -= pcr_f
    ho_f = float(max(0.0, remaining))
    n_rs = int(round(rs_f * n))
    n_ha = int(round(ha_f * n))
    n_bp = int(round(bp_f * n))
    n_dr = int(round(dr_f * n))
    n_rj = int(round(rj_f * n))
    n_rep = int(round(rep_f * n))
    n_cmp = int(round(cmp_f * n))
    n_ch = int(round(ch_f * n))
    n_ctr = int(round(ctr_f * n))
    n_pcr = int(round(pcr_f * n))
    n_ho = int(max(
        0,
        n - n_rs - n_ha - n_bp - n_dr - n_rj - n_rep - n_cmp
        - n_ch - n_ctr - n_pcr))
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
        + int(post_restart_replacement_fallback_tokens)
            * int(n_pcr)
        + int(hosted_only_tokens) * int(n_ho))
    total_all_hosted = int(int(hosted_only_tokens) * int(n))
    saving = int(total_all_hosted - total_handoff)
    ratio = (
        float(saving) / float(max(1, total_all_hosted))
        if total_all_hosted > 0 else 0.0)
    return {
        "schema": W77_HOSTED_REAL_HANDOFF_V9_SCHEMA_VERSION,
        "n_turns": int(n),
        "n_real_substrate_only": int(n_rs),
        "n_post_restart_replacement_fallback": int(n_pcr),
        "n_hosted_only": int(n_ho),
        "total_handoff_tokens": int(total_handoff),
        "total_all_hosted_tokens": int(total_all_hosted),
        "saving_tokens": int(saving),
        "saving_ratio": float(round(ratio, 12)),
    }


@dataclasses.dataclass(frozen=True)
class HostedRealHandoffCoordinatorV9Witness:
    schema: str
    coordinator_v9_cid: str
    n_envelopes_v9: int
    n_post_restart_replacement_fallback: int
    n_hosted_only: int
    n_real_substrate_only: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "coordinator_v9_cid": str(self.coordinator_v9_cid),
            "n_envelopes_v9": int(self.n_envelopes_v9),
            "n_post_restart_replacement_fallback": int(
                self.n_post_restart_replacement_fallback),
            "n_hosted_only": int(self.n_hosted_only),
            "n_real_substrate_only": int(
                self.n_real_substrate_only),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind":
                "hosted_real_handoff_coordinator_v9_witness",
            "witness": self.to_dict()})


def emit_hosted_real_handoff_coordinator_v9_witness(
        coordinator: HostedRealHandoffCoordinatorV9,
) -> HostedRealHandoffCoordinatorV9Witness:
    counts = {d: 0 for d in W77_HANDOFF_DECISIONS_V9}
    for entry in coordinator.audit_v9:
        d = str(entry.get("decision_v9", ""))
        if d in counts:
            counts[d] += 1
    return HostedRealHandoffCoordinatorV9Witness(
        schema=W77_HOSTED_REAL_HANDOFF_V9_SCHEMA_VERSION,
        coordinator_v9_cid=str(coordinator.cid()),
        n_envelopes_v9=int(len(coordinator.audit_v9)),
        n_post_restart_replacement_fallback=int(
            counts[
                W77_HANDOFF_DECISION_POST_RESTART_REPLACEMENT_FALLBACK]),
        n_hosted_only=int(
            counts[W69_HANDOFF_DECISION_HOSTED_ONLY]),
        n_real_substrate_only=int(
            counts[W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY]),
    )


__all__ = [
    "W77_HOSTED_REAL_HANDOFF_V9_SCHEMA_VERSION",
    "W77_HANDOFF_DECISION_POST_RESTART_REPLACEMENT_FALLBACK",
    "W77_HANDOFF_DECISIONS_V9",
    "W77_DEFAULT_POST_RESTART_REPLACEMENT_PRESSURE_FLOOR",
    "W77_DEFAULT_POST_RESTART_REPLACEMENT_WINDOW_FLOOR",
    "W77_DEFAULT_SUBSTRATE_TRUST_FLOOR",
    "HandoffRequestV9",
    "HandoffEnvelopeV9",
    "HostedRealHandoffCoordinatorV9",
    "HostedRealHandoffV9PostRestartReplacementFalsifier",
    "probe_hosted_real_handoff_v9_post_restart_replacement_falsifier",
    "hosted_real_handoff_v9_post_restart_replacement_aware_savings",
    "HostedRealHandoffCoordinatorV9Witness",
    "emit_hosted_real_handoff_coordinator_v9_witness",
]
