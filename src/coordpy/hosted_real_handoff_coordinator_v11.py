"""W79 H6 — Hosted ↔ Real Substrate Handoff Coordinator V11.

Strictly extends W78's V10. Adds:

* **Replacement-then-restart-after-long-delay-aware promotion**
  — any request with
  ``replacement_then_restart_after_long_delay_pressure >= floor``
  AND substrate trust ≥ floor is promoted to
  ``real_substrate_only`` with
  ``replacement_then_restart_after_long_delay_alignment = 1.0``.
* **W79 fallback decision** —
  ``replacement_then_restart_after_long_delay_fallback`` fires
  when hosted is cheaper but the caller-declared trajectory CID
  requires Plane B AND the request didn't already promote.
* **Controlled-runtime promotion** — if the request declares
  ``needs_controlled_runtime = True``, the V11 coordinator
  promotes to a new ``controlled_runtime`` plane.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any

from .hosted_real_handoff_coordinator import (
    W69_HANDOFF_DECISION_HOSTED_ONLY,
    W69_HANDOFF_DECISION_HOSTED_WITH_REAL_AUDIT,
    W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY,
)
from .hosted_real_handoff_coordinator_v10 import (
    HandoffEnvelopeV10, HandoffRequestV10,
    HostedRealHandoffCoordinatorV10,
    W78_HANDOFF_DECISIONS_V10,
)
from .hosted_real_substrate_boundary_v12 import (
    HostedRealSubstrateBoundaryV12,
    build_default_hosted_real_substrate_boundary_v12,
)


W79_HOSTED_REAL_HANDOFF_V11_SCHEMA_VERSION: str = (
    "coordpy.hosted_real_handoff_coordinator_v11.v1")

W79_HANDOFF_DECISION_REPLACEMENT_THEN_RESTART_AFTER_LONG_DELAY_FALLBACK: (
    str) = (
    "replacement_then_restart_after_long_delay_fallback")
W79_HANDOFF_DECISION_CONTROLLED_RUNTIME: str = (
    "controlled_runtime_substrate")
W79_HANDOFF_DECISIONS_V11: tuple[str, ...] = (
    *W78_HANDOFF_DECISIONS_V10,
    W79_HANDOFF_DECISION_REPLACEMENT_THEN_RESTART_AFTER_LONG_DELAY_FALLBACK,
    W79_HANDOFF_DECISION_CONTROLLED_RUNTIME,
)
W79_DEFAULT_RTRLD_PRESSURE_FLOOR: float = 0.5
W79_DEFAULT_RTRLD_WINDOW_FLOOR: int = 55
W79_DEFAULT_SUBSTRATE_TRUST_FLOOR_V11: float = 0.5


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass(frozen=True)
class HandoffRequestV11:
    inner_v10: HandoffRequestV10
    replacement_then_restart_after_long_delay_pressure: float = 0.0
    replacement_then_restart_after_long_delay_trajectory_cid: (
        str) = ""
    long_delay_blackout_window_turns: int = 0
    needs_controlled_runtime: bool = False
    expected_substrate_trust_v11: float = 0.7

    @property
    def request_cid(self) -> str:
        return self.inner_v10.request_cid

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W79_HOSTED_REAL_HANDOFF_V11_SCHEMA_VERSION,
            "kind": "handoff_request_v11",
            "inner_v10_cid": str(self.inner_v10.cid()),
            "replacement_then_restart_after_long_delay_pressure": float(
                round(
                    self
                    .replacement_then_restart_after_long_delay_pressure,
                    12)),
            "replacement_then_restart_after_long_delay_trajectory_cid": str(
                self
                .replacement_then_restart_after_long_delay_trajectory_cid),
            "long_delay_blackout_window_turns": int(
                self.long_delay_blackout_window_turns),
            "needs_controlled_runtime": bool(
                self.needs_controlled_runtime),
            "expected_substrate_trust_v11": float(round(
                self.expected_substrate_trust_v11, 12)),
        })


@dataclasses.dataclass(frozen=True)
class HandoffEnvelopeV11:
    schema: str
    inner_v10: HandoffEnvelopeV10
    decision_v11: str
    plane_v11: str
    replacement_then_restart_after_long_delay_alignment: float
    replacement_then_restart_after_long_delay_fallback_active: (
        bool)
    controlled_runtime_active: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "inner_v10_cid": str(self.inner_v10.cid()),
            "decision_v11": str(self.decision_v11),
            "plane_v11": str(self.plane_v11),
            "replacement_then_restart_after_long_delay_alignment": float(
                round(
                    self
                    .replacement_then_restart_after_long_delay_alignment,
                    12)),
            "replacement_then_restart_after_long_delay_fallback_active": bool(
                self
                .replacement_then_restart_after_long_delay_fallback_active),
            "controlled_runtime_active": bool(
                self.controlled_runtime_active),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_real_handoff_envelope_v11",
            "envelope": self.to_dict()})


@dataclasses.dataclass
class HostedRealHandoffCoordinatorV11:
    inner_v10: HostedRealHandoffCoordinatorV10 = dataclasses.field(
        default_factory=HostedRealHandoffCoordinatorV10)
    boundary_v12: HostedRealSubstrateBoundaryV12 = (
        dataclasses.field(
            default_factory=(
                build_default_hosted_real_substrate_boundary_v12)))
    replacement_then_restart_after_long_delay_pressure_floor: (
        float) = W79_DEFAULT_RTRLD_PRESSURE_FLOOR
    replacement_then_restart_after_long_delay_window_floor: int = (
        W79_DEFAULT_RTRLD_WINDOW_FLOOR)
    substrate_trust_floor: float = (
        W79_DEFAULT_SUBSTRATE_TRUST_FLOOR_V11)
    audit_v11: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    def cid(self) -> str:
        return _sha256_hex({
            "schema":
                W79_HOSTED_REAL_HANDOFF_V11_SCHEMA_VERSION,
            "kind": "hosted_real_handoff_coordinator_v11",
            "inner_v10_cid": str(self.inner_v10.cid()),
            "boundary_v12_cid": str(self.boundary_v12.cid()),
            "replacement_then_restart_after_long_delay_pressure_floor": float(
                round(
                    self
                    .replacement_then_restart_after_long_delay_pressure_floor,
                    12)),
            "replacement_then_restart_after_long_delay_window_floor": int(
                self
                .replacement_then_restart_after_long_delay_window_floor),
            "substrate_trust_floor": float(round(
                self.substrate_trust_floor, 12)),
        })

    def decide_v11(
            self, *, req_v11: HandoffRequestV11,
    ) -> HandoffEnvelopeV11:
        v10_env = self.inner_v10.decide_v10(
            req_v10=req_v11.inner_v10)
        decision_v11 = str(v10_env.decision_v10)
        plane_v11 = str(v10_env.plane_v10)
        rtrld_alignment = 0.0
        rtrld_fallback_active = False
        controlled_runtime_active = False
        # Controlled-runtime promotion has highest priority.
        if (bool(req_v11.needs_controlled_runtime)
                and float(req_v11.expected_substrate_trust_v11)
                >= float(self.substrate_trust_floor)):
            decision_v11 = W79_HANDOFF_DECISION_CONTROLLED_RUNTIME
            plane_v11 = "controlled_runtime"
            controlled_runtime_active = True
            rtrld_alignment = 1.0
        # RTRLD-aware promotion.
        elif (decision_v11 == W69_HANDOFF_DECISION_HOSTED_ONLY
                and float(
                    req_v11
                    .replacement_then_restart_after_long_delay_pressure)
                >= float(
                    self
                    .replacement_then_restart_after_long_delay_pressure_floor)
                and float(req_v11.expected_substrate_trust_v11)
                >= float(self.substrate_trust_floor)):
            decision_v11 = (
                W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY)
            plane_v11 = "B"
            rtrld_alignment = 1.0
        # RTRLD fallback.
        elif (str(req_v11
                  .replacement_then_restart_after_long_delay_trajectory_cid)
              and int(req_v11.long_delay_blackout_window_turns)
              >= int(
                  self
                  .replacement_then_restart_after_long_delay_window_floor)
              and float(req_v11.expected_substrate_trust_v11)
              >= float(self.substrate_trust_floor)
              and decision_v11
              == W69_HANDOFF_DECISION_HOSTED_ONLY):
            decision_v11 = (
                W79_HANDOFF_DECISION_REPLACEMENT_THEN_RESTART_AFTER_LONG_DELAY_FALLBACK)
            plane_v11 = "A-rtrld-fallback"
            rtrld_fallback_active = True
            rtrld_alignment = 1.0
        # Honest alignment when an inner plane already chose B
        # but the V11 RTRLD signal would still have promoted it
        # — keep the alignment at 1.0 so the V11 layer accurately
        # reports the RTRLD-aware decision.
        if (rtrld_alignment == 0.0
                and decision_v11
                == W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY
                and float(
                    req_v11
                    .replacement_then_restart_after_long_delay_pressure)
                >= float(
                    self
                    .replacement_then_restart_after_long_delay_pressure_floor)
                and float(req_v11.expected_substrate_trust_v11)
                >= float(self.substrate_trust_floor)):
            rtrld_alignment = 1.0
        env = HandoffEnvelopeV11(
            schema=W79_HOSTED_REAL_HANDOFF_V11_SCHEMA_VERSION,
            inner_v10=v10_env,
            decision_v11=str(decision_v11),
            plane_v11=str(plane_v11),
            replacement_then_restart_after_long_delay_alignment=float(
                rtrld_alignment),
            replacement_then_restart_after_long_delay_fallback_active=bool(
                rtrld_fallback_active),
            controlled_runtime_active=bool(
                controlled_runtime_active),
        )
        self.audit_v11.append({
            "request_cid": str(req_v11.request_cid),
            "decision_v11": str(decision_v11),
            "plane_v11": str(plane_v11),
            "envelope_v11_cid": str(env.cid()),
            "replacement_then_restart_after_long_delay_alignment": float(
                rtrld_alignment),
            "controlled_runtime_active": bool(
                controlled_runtime_active),
        })
        return env


@dataclasses.dataclass(frozen=True)
class HostedRealHandoffV11ReplacementThenRestartAfterLongDelayFalsifier:
    schema: str
    envelope_v11_cid: str
    claim_kind: str
    claim_satisfied: bool
    falsifier_score: float

    def cid(self) -> str:
        return _sha256_hex({
            "kind":
                "hosted_real_handoff_v11_rtrld_falsifier",
            "schema": str(self.schema),
            "envelope_v11_cid": str(self.envelope_v11_cid),
            "claim_kind": str(self.claim_kind),
            "claim_satisfied": bool(self.claim_satisfied),
            "falsifier_score": float(round(
                self.falsifier_score, 12)),
        })


def probe_hosted_real_handoff_v11_replacement_then_restart_after_long_delay_falsifier(
        *, envelope_v11: HandoffEnvelopeV11,
        claim_kind: str = (
            "hosted_satisfies_replacement_then_restart_after_long_delay"),
        claim_satisfied: bool = False,
) -> HostedRealHandoffV11ReplacementThenRestartAfterLongDelayFalsifier:
    """0 on honest claims, 1 on dishonest."""
    score = 0.0
    if (str(envelope_v11.decision_v11)
            == W69_HANDOFF_DECISION_HOSTED_ONLY
            and claim_kind
                == "hosted_satisfies_replacement_then_restart_after_long_delay"
            and bool(claim_satisfied)):
        score = 1.0
    return HostedRealHandoffV11ReplacementThenRestartAfterLongDelayFalsifier(
        schema=W79_HOSTED_REAL_HANDOFF_V11_SCHEMA_VERSION,
        envelope_v11_cid=str(envelope_v11.cid()),
        claim_kind=str(claim_kind),
        claim_satisfied=bool(claim_satisfied),
        falsifier_score=float(score),
    )


def hosted_real_handoff_v11_replacement_then_restart_after_long_delay_aware_savings(
        *, n_turns: int,
        hosted_only_tokens: int = 1000,
        real_substrate_only_tokens: int = 70,
        controlled_runtime_tokens: int = 90,
        rtrld_fallback_tokens: int = 22,
        real_substrate_fraction: float = 0.50,
        controlled_runtime_fraction: float = 0.10,
        rtrld_fallback_fraction: float = 0.08,
        other_fallback_fraction: float = 0.28,
        other_fallback_tokens: int = 45,
) -> dict[str, Any]:
    """V11 cross-plane savings. ≥ 88 % at default config."""
    n = int(max(1, n_turns))
    rs_f = max(0.0, min(1.0, real_substrate_fraction))
    cr_f = max(
        0.0,
        min(1.0 - rs_f, controlled_runtime_fraction))
    rtrld_f = max(
        0.0,
        min(1.0 - rs_f - cr_f, rtrld_fallback_fraction))
    other_f = max(
        0.0,
        min(1.0 - rs_f - cr_f - rtrld_f,
            other_fallback_fraction))
    ho_f = max(0.0, 1.0 - rs_f - cr_f - rtrld_f - other_f)
    n_rs = int(round(rs_f * n))
    n_cr = int(round(cr_f * n))
    n_rtrld = int(round(rtrld_f * n))
    n_other = int(round(other_f * n))
    n_ho = int(max(0, n - n_rs - n_cr - n_rtrld - n_other))
    total_handoff = int(
        int(real_substrate_only_tokens) * int(n_rs)
        + int(controlled_runtime_tokens) * int(n_cr)
        + int(rtrld_fallback_tokens) * int(n_rtrld)
        + int(other_fallback_tokens) * int(n_other)
        + int(hosted_only_tokens) * int(n_ho))
    total_all_hosted = int(int(hosted_only_tokens) * int(n))
    saving = int(total_all_hosted - total_handoff)
    ratio = (
        float(saving) / float(max(1, total_all_hosted))
        if total_all_hosted > 0 else 0.0)
    return {
        "schema": W79_HOSTED_REAL_HANDOFF_V11_SCHEMA_VERSION,
        "n_turns": int(n),
        "n_real_substrate_only": int(n_rs),
        "n_controlled_runtime": int(n_cr),
        "n_rtrld_fallback": int(n_rtrld),
        "n_hosted_only": int(n_ho),
        "total_handoff_tokens": int(total_handoff),
        "total_all_hosted_tokens": int(total_all_hosted),
        "saving_tokens": int(saving),
        "saving_ratio": float(round(ratio, 12)),
    }


@dataclasses.dataclass(frozen=True)
class HostedRealHandoffCoordinatorV11Witness:
    schema: str
    coordinator_v11_cid: str
    n_envelopes_v11: int
    n_rtrld_fallback: int
    n_controlled_runtime: int
    n_hosted_only: int
    n_real_substrate_only: int

    def cid(self) -> str:
        return _sha256_hex({
            "kind":
                "hosted_real_handoff_coordinator_v11_witness",
            "schema": str(self.schema),
            "coordinator_v11_cid": str(self.coordinator_v11_cid),
            "n_envelopes_v11": int(self.n_envelopes_v11),
            "n_rtrld_fallback": int(self.n_rtrld_fallback),
            "n_controlled_runtime": int(self.n_controlled_runtime),
            "n_hosted_only": int(self.n_hosted_only),
            "n_real_substrate_only": int(
                self.n_real_substrate_only),
        })


def emit_hosted_real_handoff_coordinator_v11_witness(
        coordinator: HostedRealHandoffCoordinatorV11,
) -> HostedRealHandoffCoordinatorV11Witness:
    counts = {d: 0 for d in W79_HANDOFF_DECISIONS_V11}
    for entry in coordinator.audit_v11:
        d = str(entry.get("decision_v11", ""))
        if d in counts:
            counts[d] += 1
    return HostedRealHandoffCoordinatorV11Witness(
        schema=W79_HOSTED_REAL_HANDOFF_V11_SCHEMA_VERSION,
        coordinator_v11_cid=str(coordinator.cid()),
        n_envelopes_v11=int(len(coordinator.audit_v11)),
        n_rtrld_fallback=int(
            counts[
                W79_HANDOFF_DECISION_REPLACEMENT_THEN_RESTART_AFTER_LONG_DELAY_FALLBACK]),
        n_controlled_runtime=int(
            counts[W79_HANDOFF_DECISION_CONTROLLED_RUNTIME]),
        n_hosted_only=int(
            counts[W69_HANDOFF_DECISION_HOSTED_ONLY]),
        n_real_substrate_only=int(
            counts[W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY]),
    )


__all__ = [
    "W79_HOSTED_REAL_HANDOFF_V11_SCHEMA_VERSION",
    "W79_HANDOFF_DECISION_REPLACEMENT_THEN_RESTART_AFTER_LONG_DELAY_FALLBACK",
    "W79_HANDOFF_DECISION_CONTROLLED_RUNTIME",
    "W79_HANDOFF_DECISIONS_V11",
    "W79_DEFAULT_RTRLD_PRESSURE_FLOOR",
    "W79_DEFAULT_RTRLD_WINDOW_FLOOR",
    "W79_DEFAULT_SUBSTRATE_TRUST_FLOOR_V11",
    "HandoffRequestV11",
    "HandoffEnvelopeV11",
    "HostedRealHandoffCoordinatorV11",
    "HostedRealHandoffV11ReplacementThenRestartAfterLongDelayFalsifier",
    "probe_hosted_real_handoff_v11_replacement_then_restart_after_long_delay_falsifier",
    "hosted_real_handoff_v11_replacement_then_restart_after_long_delay_aware_savings",
    "HostedRealHandoffCoordinatorV11Witness",
    "emit_hosted_real_handoff_coordinator_v11_witness",
]
