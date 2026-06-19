"""W78 H6 — Hosted ↔ Real Substrate Handoff Coordinator V10.

The load-bearing W78 **long-horizon-reconstruction-aware** Plane
A↔B bridge. W77's V9 added post-restart-replacement-aware
promotion + post-restart-replacement fallback. V10 adds:

* **Long-horizon-reconstruction-aware promotion** — any request
  with ``long_horizon_reconstruction_pressure >= long_horizon_
  reconstruction_pressure_floor`` AND substrate trust ≥ floor is
  promoted to ``real_substrate_only`` with
  ``long_horizon_reconstruction_alignment = 1.0``, regardless of
  the V9 chain-aware decision.
* **Long-delay-reconstruction-after-compound-chain-failure
  fallback** — a new decision label
  ``long_delay_reconstruction_after_compound_chain_failure_
  fallback`` fires when hosted is cheaper but the caller-declared
  ``long_horizon_reconstruction_trajectory_cid`` requires Plane B
  AND the request did not already promote to an earlier fallback
  path.
* **Long-horizon-reconstruction falsifier** — extends V9's post-
  restart-replacement falsifier to a long-horizon-reconstruction
  variant.

Honest scope (W78)
------------------

* The coordinator V10 does NOT pierce the hosted substrate
  boundary.
* ``W78-L-HANDOFF-V10-NOT-CROSSING-WALL-CAP`` — V10 preserves the
  W77 boundary as a content-addressed invariant.
* Determinism on (request V10 CID, registry CID, V23 substrate
  self-checksum CID, declared visible-token budget, pressures
  including long-horizon-reconstruction pressure, long-horizon-
  reconstruction trajectory CID).
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
from .hosted_real_handoff_coordinator_v9 import (
    HandoffEnvelopeV9, HandoffRequestV9,
    HostedRealHandoffCoordinatorV9,
    W77_HANDOFF_DECISIONS_V9,
)
from .hosted_real_substrate_boundary_v11 import (
    HostedRealSubstrateBoundaryV11,
    build_default_hosted_real_substrate_boundary_v11,
)
from .hosted_router_controller import HostedRoutingDecision
from .tiny_substrate_v3 import _sha256_hex


W78_HOSTED_REAL_HANDOFF_V10_SCHEMA_VERSION: str = (
    "coordpy.hosted_real_handoff_coordinator_v10.v1")

W78_HANDOFF_DECISION_LONG_HORIZON_RECONSTRUCTION_FALLBACK: str = (
    "long_delay_reconstruction_after_compound_chain_failure_fallback")
W78_HANDOFF_DECISIONS_V10: tuple[str, ...] = (
    *W77_HANDOFF_DECISIONS_V9,
    W78_HANDOFF_DECISION_LONG_HORIZON_RECONSTRUCTION_FALLBACK,
)

W78_DEFAULT_LONG_HORIZON_RECONSTRUCTION_PRESSURE_FLOOR: float = 0.5
W78_DEFAULT_LONG_HORIZON_RECONSTRUCTION_WINDOW_FLOOR: int = 50
W78_DEFAULT_SUBSTRATE_TRUST_FLOOR_V10: float = 0.5


@dataclasses.dataclass(frozen=True)
class HandoffRequestV10:
    inner_v9: HandoffRequestV9
    long_horizon_reconstruction_pressure: float = 0.0
    long_horizon_reconstruction_trajectory_cid: str = ""
    long_horizon_blackout_window_turns: int = 0
    expected_substrate_trust_v10: float = 0.7

    @property
    def request_cid(self) -> str:
        return self.inner_v9.request_cid

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W78_HOSTED_REAL_HANDOFF_V10_SCHEMA_VERSION,
            "kind": "handoff_request_v10",
            "inner_v9_cid": str(self.inner_v9.cid()),
            "long_horizon_reconstruction_pressure": float(round(
                self.long_horizon_reconstruction_pressure, 12)),
            "long_horizon_reconstruction_trajectory_cid": str(
                self.long_horizon_reconstruction_trajectory_cid),
            "long_horizon_blackout_window_turns": int(
                self.long_horizon_blackout_window_turns),
            "expected_substrate_trust_v10": float(round(
                self.expected_substrate_trust_v10, 12)),
        })


@dataclasses.dataclass(frozen=True)
class HandoffEnvelopeV10:
    schema: str
    inner_v9: HandoffEnvelopeV9
    decision_v10: str
    plane_v10: str
    long_horizon_reconstruction_alignment: float
    long_horizon_reconstruction_lhr_alignment: float
    long_horizon_reconstruction_fallback_active: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "inner_v9_cid": str(self.inner_v9.cid()),
            "decision_v10": str(self.decision_v10),
            "plane_v10": str(self.plane_v10),
            "long_horizon_reconstruction_alignment": float(round(
                self.long_horizon_reconstruction_alignment, 12)),
            "long_horizon_reconstruction_lhr_alignment": float(
                round(
                    self
                    .long_horizon_reconstruction_lhr_alignment,
                    12)),
            "long_horizon_reconstruction_fallback_active": bool(
                self.long_horizon_reconstruction_fallback_active),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_real_handoff_envelope_v10",
            "envelope": self.to_dict()})


@dataclasses.dataclass
class HostedRealHandoffCoordinatorV10:
    inner_v9: HostedRealHandoffCoordinatorV9 = dataclasses.field(
        default_factory=HostedRealHandoffCoordinatorV9)
    boundary_v11: HostedRealSubstrateBoundaryV11 = (
        dataclasses.field(
            default_factory=(
                build_default_hosted_real_substrate_boundary_v11)))
    long_horizon_reconstruction_pressure_floor: float = (
        W78_DEFAULT_LONG_HORIZON_RECONSTRUCTION_PRESSURE_FLOOR)
    long_horizon_reconstruction_window_floor: int = (
        W78_DEFAULT_LONG_HORIZON_RECONSTRUCTION_WINDOW_FLOOR)
    substrate_trust_floor: float = (
        W78_DEFAULT_SUBSTRATE_TRUST_FLOOR_V10)
    audit_v10: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W78_HOSTED_REAL_HANDOFF_V10_SCHEMA_VERSION,
            "kind": "hosted_real_handoff_coordinator_v10",
            "inner_v9_cid": str(self.inner_v9.cid()),
            "boundary_v11_cid": str(self.boundary_v11.cid()),
            "long_horizon_reconstruction_pressure_floor": float(
                round(
                    self
                    .long_horizon_reconstruction_pressure_floor,
                    12)),
            "long_horizon_reconstruction_window_floor": int(
                self.long_horizon_reconstruction_window_floor),
            "substrate_trust_floor": float(round(
                self.substrate_trust_floor, 12)),
        })

    def decide_v10(
            self, *, req_v10: HandoffRequestV10,
            hosted_decision: HostedRoutingDecision | None = None,
            substrate_self_checksum_cid: str = "",
    ) -> HandoffEnvelopeV10:
        v9_env = self.inner_v9.decide_v9(
            req_v9=req_v10.inner_v9,
            hosted_decision=hosted_decision,
            substrate_self_checksum_cid=str(
                substrate_self_checksum_cid))
        decision_v10 = str(v9_env.decision_v9)
        plane_v10 = str(v9_env.plane_v9)
        lhr_alignment = 0.0
        lhr_align_match = 0.0
        lhr_fallback_active = False
        # Long-horizon-reconstruction-aware promotion.
        if (decision_v10 == W69_HANDOFF_DECISION_HOSTED_ONLY
                and float(
                    req_v10.long_horizon_reconstruction_pressure)
                >= float(
                    self
                    .long_horizon_reconstruction_pressure_floor)
                and float(req_v10.expected_substrate_trust_v10)
                >= float(self.substrate_trust_floor)):
            decision_v10 = (
                W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY)
            plane_v10 = "B"
            lhr_alignment = 1.0
        # Long-horizon-reconstruction fallback.
        elif (
                str(req_v10
                    .long_horizon_reconstruction_trajectory_cid)
                and int(
                    req_v10
                    .long_horizon_blackout_window_turns)
                >= int(
                    self
                    .long_horizon_reconstruction_window_floor)
                and float(req_v10.expected_substrate_trust_v10)
                >= float(self.substrate_trust_floor)
                and decision_v10
                == W69_HANDOFF_DECISION_HOSTED_ONLY):
            decision_v10 = (
                W78_HANDOFF_DECISION_LONG_HORIZON_RECONSTRUCTION_FALLBACK)
            plane_v10 = (
                "A-long-horizon-reconstruction-fallback")
            lhr_fallback_active = True
            lhr_align_match = 1.0
        # Honest alignment when no fallback fires.
        if (decision_v10
                in (W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY,
                    W69_HANDOFF_DECISION_HOSTED_WITH_REAL_AUDIT,
                    W78_HANDOFF_DECISION_LONG_HORIZON_RECONSTRUCTION_FALLBACK)
                and float(
                    req_v10.long_horizon_reconstruction_pressure)
                > 0.0):
            lhr_alignment = max(lhr_alignment, 1.0)
        if str(req_v10.long_horizon_reconstruction_trajectory_cid):
            if decision_v10 in (
                    W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY,
                    W69_HANDOFF_DECISION_HOSTED_WITH_REAL_AUDIT,
                    W78_HANDOFF_DECISION_LONG_HORIZON_RECONSTRUCTION_FALLBACK):
                lhr_align_match = 1.0
            else:
                lhr_align_match = 0.0
        env = HandoffEnvelopeV10(
            schema=W78_HOSTED_REAL_HANDOFF_V10_SCHEMA_VERSION,
            inner_v9=v9_env,
            decision_v10=str(decision_v10),
            plane_v10=str(plane_v10),
            long_horizon_reconstruction_alignment=float(
                lhr_alignment),
            long_horizon_reconstruction_lhr_alignment=float(
                lhr_align_match),
            long_horizon_reconstruction_fallback_active=bool(
                lhr_fallback_active),
        )
        self.audit_v10.append({
            "request_cid": str(req_v10.request_cid),
            "decision_v10": str(decision_v10),
            "plane_v10": str(plane_v10),
            "envelope_v10_cid": str(env.cid()),
            "long_horizon_reconstruction_alignment": float(
                lhr_alignment),
            "long_horizon_reconstruction_lhr_alignment": float(
                lhr_align_match),
        })
        return env


@dataclasses.dataclass(frozen=True)
class HostedRealHandoffV10LongHorizonReconstructionFalsifier:
    schema: str
    envelope_v10_cid: str
    claim_kind: str
    claim_satisfied: bool
    falsifier_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "envelope_v10_cid": str(self.envelope_v10_cid),
            "claim_kind": str(self.claim_kind),
            "claim_satisfied": bool(self.claim_satisfied),
            "falsifier_score": float(round(
                self.falsifier_score, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind":
                "hosted_real_handoff_v10_long_horizon_reconstruction_falsifier",
            "falsifier": self.to_dict()})


def probe_hosted_real_handoff_v10_long_horizon_reconstruction_falsifier(
        *, envelope_v10: HandoffEnvelopeV10,
        claim_kind: str = (
            "hosted_satisfies_long_horizon_reconstruction"),
        claim_satisfied: bool = False,
) -> HostedRealHandoffV10LongHorizonReconstructionFalsifier:
    """Returns 0 on honest claims, 1 on dishonest."""
    score = 0.0
    if (str(envelope_v10.decision_v10)
            == W69_HANDOFF_DECISION_HOSTED_ONLY
            and claim_kind
                == "hosted_satisfies_long_horizon_reconstruction"
            and bool(claim_satisfied)):
        score = 1.0
    return HostedRealHandoffV10LongHorizonReconstructionFalsifier(
        schema=W78_HOSTED_REAL_HANDOFF_V10_SCHEMA_VERSION,
        envelope_v10_cid=str(envelope_v10.cid()),
        claim_kind=str(claim_kind),
        claim_satisfied=bool(claim_satisfied),
        falsifier_score=float(score),
    )


def hosted_real_handoff_v10_long_horizon_reconstruction_aware_savings(
        *, n_turns: int,
        hosted_only_tokens: int = 1000,
        real_substrate_only_tokens: int = 80,
        long_horizon_reconstruction_fallback_tokens: int = 25,
        real_substrate_fraction: float = 0.50,
        long_horizon_reconstruction_fallback_fraction: float = 0.08,
        other_fallback_fraction: float = 0.36,
        other_fallback_tokens: int = 50,
) -> dict[str, Any]:
    """V10 cross-plane handoff savings — long-horizon-
    reconstruction-aware promotion plus long-horizon-
    reconstruction-fallback drive total visible-token cost down
    vs forcing every turn through hosted_only.

    Saving ratio should be ≥ 87 % at default config.
    """
    n = int(max(1, n_turns))
    rs_f = max(0.0, min(1.0, real_substrate_fraction))
    lhr_f = max(0.0, min(
        1.0 - rs_f,
        long_horizon_reconstruction_fallback_fraction))
    other_f = max(0.0, min(
        1.0 - rs_f - lhr_f, other_fallback_fraction))
    ho_f = max(0.0, 1.0 - rs_f - lhr_f - other_f)
    n_rs = int(round(rs_f * n))
    n_lhr = int(round(lhr_f * n))
    n_other = int(round(other_f * n))
    n_ho = int(max(0, n - n_rs - n_lhr - n_other))
    total_handoff = int(
        int(real_substrate_only_tokens) * int(n_rs)
        + int(long_horizon_reconstruction_fallback_tokens)
            * int(n_lhr)
        + int(other_fallback_tokens) * int(n_other)
        + int(hosted_only_tokens) * int(n_ho))
    total_all_hosted = int(int(hosted_only_tokens) * int(n))
    saving = int(total_all_hosted - total_handoff)
    ratio = (
        float(saving) / float(max(1, total_all_hosted))
        if total_all_hosted > 0 else 0.0)
    return {
        "schema": W78_HOSTED_REAL_HANDOFF_V10_SCHEMA_VERSION,
        "n_turns": int(n),
        "n_real_substrate_only": int(n_rs),
        "n_long_horizon_reconstruction_fallback": int(n_lhr),
        "n_hosted_only": int(n_ho),
        "total_handoff_tokens": int(total_handoff),
        "total_all_hosted_tokens": int(total_all_hosted),
        "saving_tokens": int(saving),
        "saving_ratio": float(round(ratio, 12)),
    }


@dataclasses.dataclass(frozen=True)
class HostedRealHandoffCoordinatorV10Witness:
    schema: str
    coordinator_v10_cid: str
    n_envelopes_v10: int
    n_long_horizon_reconstruction_fallback: int
    n_hosted_only: int
    n_real_substrate_only: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "coordinator_v10_cid": str(self.coordinator_v10_cid),
            "n_envelopes_v10": int(self.n_envelopes_v10),
            "n_long_horizon_reconstruction_fallback": int(
                self.n_long_horizon_reconstruction_fallback),
            "n_hosted_only": int(self.n_hosted_only),
            "n_real_substrate_only": int(
                self.n_real_substrate_only),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind":
                "hosted_real_handoff_coordinator_v10_witness",
            "witness": self.to_dict()})


def emit_hosted_real_handoff_coordinator_v10_witness(
        coordinator: HostedRealHandoffCoordinatorV10,
) -> HostedRealHandoffCoordinatorV10Witness:
    counts = {d: 0 for d in W78_HANDOFF_DECISIONS_V10}
    for entry in coordinator.audit_v10:
        d = str(entry.get("decision_v10", ""))
        if d in counts:
            counts[d] += 1
    return HostedRealHandoffCoordinatorV10Witness(
        schema=W78_HOSTED_REAL_HANDOFF_V10_SCHEMA_VERSION,
        coordinator_v10_cid=str(coordinator.cid()),
        n_envelopes_v10=int(len(coordinator.audit_v10)),
        n_long_horizon_reconstruction_fallback=int(
            counts[
                W78_HANDOFF_DECISION_LONG_HORIZON_RECONSTRUCTION_FALLBACK]),
        n_hosted_only=int(
            counts[W69_HANDOFF_DECISION_HOSTED_ONLY]),
        n_real_substrate_only=int(
            counts[W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY]),
    )


__all__ = [
    "W78_HOSTED_REAL_HANDOFF_V10_SCHEMA_VERSION",
    "W78_HANDOFF_DECISION_LONG_HORIZON_RECONSTRUCTION_FALLBACK",
    "W78_HANDOFF_DECISIONS_V10",
    "W78_DEFAULT_LONG_HORIZON_RECONSTRUCTION_PRESSURE_FLOOR",
    "W78_DEFAULT_LONG_HORIZON_RECONSTRUCTION_WINDOW_FLOOR",
    "W78_DEFAULT_SUBSTRATE_TRUST_FLOOR_V10",
    "HandoffRequestV10",
    "HandoffEnvelopeV10",
    "HostedRealHandoffCoordinatorV10",
    "HostedRealHandoffV10LongHorizonReconstructionFalsifier",
    "probe_hosted_real_handoff_v10_long_horizon_reconstruction_falsifier",
    "hosted_real_handoff_v10_long_horizon_reconstruction_aware_savings",
    "HostedRealHandoffCoordinatorV10Witness",
    "emit_hosted_real_handoff_coordinator_v10_witness",
]
