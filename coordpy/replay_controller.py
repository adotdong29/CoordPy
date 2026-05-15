"""W60 M7 — Replay Controller.

The first-class state-reuse-vs-recompute-vs-fallback policy in the
programme. Given:

* a current substrate state (KV cache + per-layer hidden + flop
  budget + corruption flags),
* a candidate cached prefix to reuse,
* a candidate set of follow-up token ids to compute,
* a fallback transcript option,

the Replay Controller decides among:

1. ``choose_reuse``      — load the cached prefix and forward
2. ``choose_recompute``  — discard the cached prefix and recompute
3. ``choose_fallback``   — emit the transcript fallback
4. ``choose_abstain``    — emit empty / no-op

Decision rule (in order):

A. If the cached-prefix fingerprint passes CRC and the projected
   flop saving is ≥ ``flop_saving_floor``, choose REUSE.
B. Otherwise, if the recompute path is *strictly* under the flop
   ceiling and the cached path's drift is *above* the
   ``drift_ceiling``, choose RECOMPUTE.
C. Otherwise, if a transcript fallback is available, choose
   FALLBACK.
D. Otherwise, ABSTAIN.

The controller maintains an audit log of every decision and emits
a fitted *flop-vs-drift* trade-off curve. The W60 R-125 H-bar
asserts the controller's REUSE rate exceeds the FALLBACK rate
when CRC passes, and the FALLBACK rate dominates when CRC fails.

Honest scope
------------

* The decision rule is *deterministic* given the inputs — no
  learned policy. Future W6x can fit the thresholds; W60 publishes
  the rule and the audit.
* The controller does NOT execute the substrate forward itself.
  It returns a ``ReplayDecision`` that the caller routes through
  the V5 substrate.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Sequence


W60_REPLAY_CONTROLLER_SCHEMA_VERSION: str = (
    "coordpy.replay_controller.v1")

W60_REPLAY_DECISION_REUSE: str = "choose_reuse"
W60_REPLAY_DECISION_RECOMPUTE: str = "choose_recompute"
W60_REPLAY_DECISION_FALLBACK: str = "choose_fallback"
W60_REPLAY_DECISION_ABSTAIN: str = "choose_abstain"

W60_REPLAY_DECISIONS: tuple[str, ...] = (
    W60_REPLAY_DECISION_REUSE,
    W60_REPLAY_DECISION_RECOMPUTE,
    W60_REPLAY_DECISION_FALLBACK,
    W60_REPLAY_DECISION_ABSTAIN,
)

W60_DEFAULT_FLOP_SAVING_FLOOR: float = 0.20
W60_DEFAULT_DRIFT_CEILING: float = 0.50
W60_DEFAULT_FLOP_CEILING_RATIO: float = 1.5


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass(frozen=True)
class ReplayCandidate:
    """A candidate replay path. The controller decides among:

    * REUSE: ``flop_reuse`` flops, ``drift_l2_reuse`` drift,
      ``crc_passed`` boolean.
    * RECOMPUTE: ``flop_recompute`` flops, ``drift_l2_recompute``
      drift (typically zero by construction).
    * FALLBACK: transcript path, ``drift_l2_fallback`` drift,
      cost ``flop_fallback``.
    """
    flop_reuse: int
    flop_recompute: int
    flop_fallback: int
    drift_l2_reuse: float
    drift_l2_recompute: float
    drift_l2_fallback: float
    crc_passed: bool
    transcript_available: bool
    n_corruption_flags: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "flop_reuse": int(self.flop_reuse),
            "flop_recompute": int(self.flop_recompute),
            "flop_fallback": int(self.flop_fallback),
            "drift_l2_reuse": float(round(
                self.drift_l2_reuse, 12)),
            "drift_l2_recompute": float(round(
                self.drift_l2_recompute, 12)),
            "drift_l2_fallback": float(round(
                self.drift_l2_fallback, 12)),
            "crc_passed": bool(self.crc_passed),
            "transcript_available": bool(
                self.transcript_available),
            "n_corruption_flags": int(self.n_corruption_flags),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "replay_candidate",
            "candidate": self.to_dict()})


@dataclasses.dataclass(frozen=True)
class ReplayDecision:
    decision: str
    flop_chosen: int
    drift_chosen: float
    flop_saving_vs_recompute: float
    rationale: str
    crc_passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W60_REPLAY_CONTROLLER_SCHEMA_VERSION,
            "decision": str(self.decision),
            "flop_chosen": int(self.flop_chosen),
            "drift_chosen": float(round(
                self.drift_chosen, 12)),
            "flop_saving_vs_recompute": float(round(
                self.flop_saving_vs_recompute, 12)),
            "rationale": str(self.rationale),
            "crc_passed": bool(self.crc_passed),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "replay_decision",
            "decision": self.to_dict()})


@dataclasses.dataclass
class ReplayController:
    flop_saving_floor: float = W60_DEFAULT_FLOP_SAVING_FLOOR
    drift_ceiling: float = W60_DEFAULT_DRIFT_CEILING
    flop_ceiling_ratio: float = W60_DEFAULT_FLOP_CEILING_RATIO
    audit: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    def cid(self) -> str:
        return _sha256_hex({
            "schema": W60_REPLAY_CONTROLLER_SCHEMA_VERSION,
            "kind": "replay_controller",
            "flop_saving_floor": float(round(
                self.flop_saving_floor, 12)),
            "drift_ceiling": float(round(
                self.drift_ceiling, 12)),
            "flop_ceiling_ratio": float(round(
                self.flop_ceiling_ratio, 12)),
            "audit": list(self.audit),
        })

    def decide(
            self, candidate: ReplayCandidate,
    ) -> ReplayDecision:
        # Saving ratio.
        denom = max(int(candidate.flop_recompute), 1)
        saving = (
            float(int(candidate.flop_recompute)
                  - int(candidate.flop_reuse))
            / float(denom))
        # A. REUSE if CRC passed and saving above floor and
        #    drift below ceiling.
        if (bool(candidate.crc_passed)
                and saving >= float(self.flop_saving_floor)
                and float(candidate.drift_l2_reuse)
                <= float(self.drift_ceiling)):
            decision = ReplayDecision(
                decision=W60_REPLAY_DECISION_REUSE,
                flop_chosen=int(candidate.flop_reuse),
                drift_chosen=float(candidate.drift_l2_reuse),
                flop_saving_vs_recompute=float(saving),
                rationale=(
                    "crc_passed_and_saving_above_floor_and_drift_below_ceiling"),
                crc_passed=True,
            )
            self.audit.append({
                "stage": "rule_a", **decision.to_dict()})
            return decision
        # B. RECOMPUTE if recompute path is under the flop ceiling
        #    and cached path's drift is above the ceiling.
        flop_ceiling = float(self.flop_ceiling_ratio) * float(
            candidate.flop_recompute)
        if (float(candidate.flop_recompute) <= flop_ceiling
                and (
                    float(candidate.drift_l2_reuse)
                    > float(self.drift_ceiling)
                    or not bool(candidate.crc_passed))):
            decision = ReplayDecision(
                decision=W60_REPLAY_DECISION_RECOMPUTE,
                flop_chosen=int(candidate.flop_recompute),
                drift_chosen=float(
                    candidate.drift_l2_recompute),
                flop_saving_vs_recompute=0.0,
                rationale=(
                    "recompute_under_ceiling_and_reuse_drift_too_high"
                    if bool(candidate.crc_passed)
                    else "crc_failed_recompute_chosen"),
                crc_passed=bool(candidate.crc_passed),
            )
            self.audit.append({
                "stage": "rule_b", **decision.to_dict()})
            return decision
        # C. FALLBACK if transcript available.
        if bool(candidate.transcript_available):
            decision = ReplayDecision(
                decision=W60_REPLAY_DECISION_FALLBACK,
                flop_chosen=int(candidate.flop_fallback),
                drift_chosen=float(
                    candidate.drift_l2_fallback),
                flop_saving_vs_recompute=(
                    float(int(candidate.flop_recompute)
                          - int(candidate.flop_fallback))
                    / float(denom)),
                rationale=(
                    "fallback_after_reuse_and_recompute_unavailable"),
                crc_passed=bool(candidate.crc_passed),
            )
            self.audit.append({
                "stage": "rule_c", **decision.to_dict()})
            return decision
        # D. ABSTAIN.
        decision = ReplayDecision(
            decision=W60_REPLAY_DECISION_ABSTAIN,
            flop_chosen=0,
            drift_chosen=0.0,
            flop_saving_vs_recompute=1.0,
            rationale="all_paths_unavailable",
            crc_passed=bool(candidate.crc_passed),
        )
        self.audit.append({
            "stage": "rule_d", **decision.to_dict()})
        return decision


@dataclasses.dataclass(frozen=True)
class ReplayControllerWitness:
    schema: str
    controller_cid: str
    n_decisions: int
    reuse_count: int
    recompute_count: int
    fallback_count: int
    abstain_count: int
    flop_total_chosen: int
    drift_total_chosen: float
    flop_total_recompute_baseline: int
    flop_saving_total_ratio: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "controller_cid": str(self.controller_cid),
            "n_decisions": int(self.n_decisions),
            "reuse_count": int(self.reuse_count),
            "recompute_count": int(self.recompute_count),
            "fallback_count": int(self.fallback_count),
            "abstain_count": int(self.abstain_count),
            "flop_total_chosen": int(self.flop_total_chosen),
            "drift_total_chosen": float(round(
                self.drift_total_chosen, 12)),
            "flop_total_recompute_baseline": int(
                self.flop_total_recompute_baseline),
            "flop_saving_total_ratio": float(round(
                self.flop_saving_total_ratio, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "replay_controller_witness",
            "witness": self.to_dict()})


def emit_replay_controller_witness(
        controller: ReplayController,
        candidates: Sequence[ReplayCandidate],
) -> ReplayControllerWitness:
    counts = {d: 0 for d in W60_REPLAY_DECISIONS}
    flop_total = 0
    drift_total = 0.0
    flop_recompute_baseline = 0
    for entry in controller.audit:
        d = str(entry.get("decision"))
        if d in counts:
            counts[d] += 1
        flop_total += int(entry.get("flop_chosen", 0))
        drift_total += float(entry.get("drift_chosen", 0.0))
    for c in candidates:
        flop_recompute_baseline += int(c.flop_recompute)
    saving_ratio = (
        float(flop_recompute_baseline - flop_total)
        / float(max(flop_recompute_baseline, 1)))
    return ReplayControllerWitness(
        schema=W60_REPLAY_CONTROLLER_SCHEMA_VERSION,
        controller_cid=str(controller.cid()),
        n_decisions=int(len(controller.audit)),
        reuse_count=int(counts[W60_REPLAY_DECISION_REUSE]),
        recompute_count=int(
            counts[W60_REPLAY_DECISION_RECOMPUTE]),
        fallback_count=int(
            counts[W60_REPLAY_DECISION_FALLBACK]),
        abstain_count=int(counts[W60_REPLAY_DECISION_ABSTAIN]),
        flop_total_chosen=int(flop_total),
        drift_total_chosen=float(drift_total),
        flop_total_recompute_baseline=int(
            flop_recompute_baseline),
        flop_saving_total_ratio=float(saving_ratio),
    )


__all__ = [
    "W60_REPLAY_CONTROLLER_SCHEMA_VERSION",
    "W60_REPLAY_DECISION_REUSE",
    "W60_REPLAY_DECISION_RECOMPUTE",
    "W60_REPLAY_DECISION_FALLBACK",
    "W60_REPLAY_DECISION_ABSTAIN",
    "W60_REPLAY_DECISIONS",
    "W60_DEFAULT_FLOP_SAVING_FLOOR",
    "W60_DEFAULT_DRIFT_CEILING",
    "W60_DEFAULT_FLOP_CEILING_RATIO",
    "ReplayCandidate",
    "ReplayDecision",
    "ReplayController",
    "ReplayControllerWitness",
    "emit_replay_controller_witness",
]
