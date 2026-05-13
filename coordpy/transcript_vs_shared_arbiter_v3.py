"""W54 M9 — Transcript-vs-Shared Arbiter V3 (4-arm: transcript /
   shared / merge_consensus / abstain-with-fallback).

Extends W53 V2's 3-arm comparator with a fourth ``merge_consensus``
arm and explicit **abstain-with-transcript-fallback** semantics.

Decision rule:
    if confidence < abstain_threshold:
        emit abstain — but if fallback_allowed, fall back to
        transcript_retention (recorded explicitly in
        `fallback_arm = transcript`).
    elif consensus_payload is available AND
            consensus_retention >= shared_retention + merge_floor:
        emit merge_consensus
    elif shared_retention - transcript_retention > prefer_shared:
        emit shared
    else:
        emit transcript

The arbiter reports a 4-arm comparison plus an explicit
``fallback_arm`` that records what would have been emitted on
abstain (always either transcript or zero).

Honest scope: pure-Python only, capsule-layer only.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any, Sequence

from .quantised_compression import (
    QuantisedBudgetGate, QuantisedCodebookV4,
)
from .transcript_vs_shared_arbiter_v2 import (
    ArbiterDecision,
    W53_TVS_ARM_ABSTAIN,
    W53_TVS_ARM_SHARED,
    W53_TVS_ARM_TRANSCRIPT,
    arbiter_decide,
    three_arm_compare,
)


# =============================================================================
# Schema, defaults
# =============================================================================

W54_TVS_ARBITER_V3_SCHEMA_VERSION: str = (
    "coordpy.transcript_vs_shared_arbiter_v3.v1")

W54_TVS_ARM_MERGE_CONSENSUS: str = "merge_consensus"
W54_TVS_ARM_ABSTAIN_FALLBACK_TRANSCRIPT: str = (
    "abstain_with_transcript_fallback")
W54_DEFAULT_TVS_V3_MERGE_FLOOR: float = 0.0
W54_DEFAULT_TVS_V3_ABSTAIN_FALLBACK: bool = True


# =============================================================================
# Helpers
# =============================================================================


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _round_floats(
        values: Sequence[float], precision: int = 12,
) -> list[float]:
    return [float(round(float(v), precision)) for v in values]


# =============================================================================
# 4-arm decision
# =============================================================================


@dataclasses.dataclass(frozen=True)
class ArbiterDecisionV3:
    """One per-turn 4-arm decision with explicit fallback record."""

    turn_index: int
    chosen_arm: str
    transcript_retention: float
    shared_retention: float
    merge_retention: float
    retention_gap: float
    bit_density_gap: float
    confidence: float
    abstain_reason: str
    fallback_arm: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_index": int(self.turn_index),
            "chosen_arm": str(self.chosen_arm),
            "transcript_retention": float(round(
                self.transcript_retention, 12)),
            "shared_retention": float(round(
                self.shared_retention, 12)),
            "merge_retention": float(round(
                self.merge_retention, 12)),
            "retention_gap": float(round(
                self.retention_gap, 12)),
            "bit_density_gap": float(round(
                self.bit_density_gap, 12)),
            "confidence": float(round(self.confidence, 12)),
            "abstain_reason": str(self.abstain_reason),
            "fallback_arm": str(self.fallback_arm),
        }


def arbiter_decide_v3(
        *,
        turn_index: int,
        carrier: Sequence[float],
        codebook: QuantisedCodebookV4,
        gate: QuantisedBudgetGate,
        budget_tokens: int,
        bits_per_natural_token: float = 6.0,
        confidence: float = 1.0,
        abstain_threshold: float = 0.15,
        prefer_shared_threshold: float = 0.0,
        merge_consensus_retention: float | None = None,
        merge_floor: float = W54_DEFAULT_TVS_V3_MERGE_FLOOR,
        abstain_fallback: bool = (
            W54_DEFAULT_TVS_V3_ABSTAIN_FALLBACK),
) -> ArbiterDecisionV3:
    """4-arm decision: transcript / shared / merge_consensus /
    abstain-with-fallback."""
    base = arbiter_decide(
        turn_index=int(turn_index),
        carrier=carrier,
        codebook=codebook, gate=gate,
        budget_tokens=int(budget_tokens),
        bits_per_natural_token=float(
            bits_per_natural_token),
        confidence=float(confidence),
        abstain_threshold=float(abstain_threshold),
        prefer_shared_threshold=float(
            prefer_shared_threshold))
    chosen = base.chosen_arm
    reason = base.abstain_reason
    fallback = ""
    mr = (
        float(merge_consensus_retention)
        if merge_consensus_retention is not None
        else 0.0)
    if chosen == W53_TVS_ARM_ABSTAIN:
        if abstain_fallback:
            chosen = (
                W54_TVS_ARM_ABSTAIN_FALLBACK_TRANSCRIPT)
            fallback = W53_TVS_ARM_TRANSCRIPT
        else:
            fallback = W53_TVS_ARM_ABSTAIN
    elif (merge_consensus_retention is not None
            and float(mr)
            >= float(base.shared_retention)
            + float(merge_floor)):
        chosen = W54_TVS_ARM_MERGE_CONSENSUS
        fallback = base.chosen_arm
    return ArbiterDecisionV3(
        turn_index=int(turn_index),
        chosen_arm=str(chosen),
        transcript_retention=float(
            base.transcript_retention),
        shared_retention=float(base.shared_retention),
        merge_retention=float(mr),
        retention_gap=float(base.retention_gap),
        bit_density_gap=float(base.bit_density_gap),
        confidence=float(confidence),
        abstain_reason=str(reason),
        fallback_arm=str(fallback),
    )


# =============================================================================
# 4-arm comparison
# =============================================================================


@dataclasses.dataclass(frozen=True)
class FourArmResult:
    """Aggregate four-arm comparison across many turns."""

    n_turns: int
    transcript_mean_retention: float
    shared_mean_retention: float
    merge_mean_retention: float
    arbiter_mean_retention: float
    pick_rate_transcript: float
    pick_rate_shared: float
    pick_rate_merge: float
    pick_rate_abstain_fallback: float
    decisions: tuple[ArbiterDecisionV3, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_turns": int(self.n_turns),
            "transcript_mean_retention": float(round(
                self.transcript_mean_retention, 12)),
            "shared_mean_retention": float(round(
                self.shared_mean_retention, 12)),
            "merge_mean_retention": float(round(
                self.merge_mean_retention, 12)),
            "arbiter_mean_retention": float(round(
                self.arbiter_mean_retention, 12)),
            "pick_rate_transcript": float(round(
                self.pick_rate_transcript, 12)),
            "pick_rate_shared": float(round(
                self.pick_rate_shared, 12)),
            "pick_rate_merge": float(round(
                self.pick_rate_merge, 12)),
            "pick_rate_abstain_fallback": float(round(
                self.pick_rate_abstain_fallback, 12)),
            "decisions": [
                d.to_dict() for d in self.decisions],
        }


def four_arm_compare(
        carriers: Sequence[Sequence[float]],
        *,
        codebook: QuantisedCodebookV4,
        gate: QuantisedBudgetGate,
        budget_tokens: int = 3,
        per_turn_confidences: Sequence[float] | None = None,
        per_turn_merge_retentions: (
            Sequence[float] | None) = None,
        abstain_threshold: float = 0.15,
        prefer_shared_threshold: float = 0.0,
        merge_floor: float = W54_DEFAULT_TVS_V3_MERGE_FLOOR,
        bits_per_natural_token: float = 6.0,
        abstain_fallback: bool = (
            W54_DEFAULT_TVS_V3_ABSTAIN_FALLBACK),
) -> FourArmResult:
    decisions: list[ArbiterDecisionV3] = []
    transcript_sum = 0.0
    shared_sum = 0.0
    merge_sum = 0.0
    arbiter_sum = 0.0
    n = 0
    n_t = 0
    n_s = 0
    n_m = 0
    n_af = 0
    confs = list(
        per_turn_confidences
        if per_turn_confidences is not None
        else [1.0] * len(carriers))
    mrs = list(
        per_turn_merge_retentions
        if per_turn_merge_retentions is not None
        else [0.0] * len(carriers))
    for i, c in enumerate(carriers):
        conf = float(confs[i] if i < len(confs) else 1.0)
        mr = float(mrs[i] if i < len(mrs) else 0.0)
        d = arbiter_decide_v3(
            turn_index=int(i),
            carrier=c,
            codebook=codebook, gate=gate,
            budget_tokens=int(budget_tokens),
            bits_per_natural_token=float(
                bits_per_natural_token),
            confidence=float(conf),
            abstain_threshold=float(abstain_threshold),
            prefer_shared_threshold=float(
                prefer_shared_threshold),
            merge_consensus_retention=float(mr),
            merge_floor=float(merge_floor),
            abstain_fallback=bool(abstain_fallback))
        decisions.append(d)
        transcript_sum += float(d.transcript_retention)
        shared_sum += float(d.shared_retention)
        merge_sum += float(d.merge_retention)
        if d.chosen_arm == W53_TVS_ARM_TRANSCRIPT:
            arbiter_sum += float(d.transcript_retention)
            n_t += 1
        elif d.chosen_arm == W53_TVS_ARM_SHARED:
            arbiter_sum += float(d.shared_retention)
            n_s += 1
        elif d.chosen_arm == W54_TVS_ARM_MERGE_CONSENSUS:
            arbiter_sum += float(d.merge_retention)
            n_m += 1
        elif d.chosen_arm == (
                W54_TVS_ARM_ABSTAIN_FALLBACK_TRANSCRIPT):
            arbiter_sum += float(d.transcript_retention)
            n_af += 1
        else:
            arbiter_sum += 0.0
        n += 1
    n_f = float(max(1, n))
    return FourArmResult(
        n_turns=int(n),
        transcript_mean_retention=float(
            transcript_sum / n_f),
        shared_mean_retention=float(shared_sum / n_f),
        merge_mean_retention=float(merge_sum / n_f),
        arbiter_mean_retention=float(arbiter_sum / n_f),
        pick_rate_transcript=float(n_t / n_f),
        pick_rate_shared=float(n_s / n_f),
        pick_rate_merge=float(n_m / n_f),
        pick_rate_abstain_fallback=float(n_af / n_f),
        decisions=tuple(decisions),
    )


# =============================================================================
# Witness
# =============================================================================


@dataclasses.dataclass(frozen=True)
class TVSArbiterV3Witness:
    n_turns: int
    transcript_mean_retention: float
    shared_mean_retention: float
    merge_mean_retention: float
    arbiter_mean_retention: float
    pick_rate_transcript: float
    pick_rate_shared: float
    pick_rate_merge: float
    pick_rate_abstain_fallback: float
    arbiter_strict_dominance_over_static: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_turns": int(self.n_turns),
            "transcript_mean_retention": float(round(
                self.transcript_mean_retention, 12)),
            "shared_mean_retention": float(round(
                self.shared_mean_retention, 12)),
            "merge_mean_retention": float(round(
                self.merge_mean_retention, 12)),
            "arbiter_mean_retention": float(round(
                self.arbiter_mean_retention, 12)),
            "pick_rate_transcript": float(round(
                self.pick_rate_transcript, 12)),
            "pick_rate_shared": float(round(
                self.pick_rate_shared, 12)),
            "pick_rate_merge": float(round(
                self.pick_rate_merge, 12)),
            "pick_rate_abstain_fallback": float(round(
                self.pick_rate_abstain_fallback, 12)),
            "arbiter_strict_dominance_over_static": bool(
                self.arbiter_strict_dominance_over_static),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w54_tvs_arbiter_v3_witness",
            "witness": self.to_dict()})


def emit_tvs_arbiter_v3_witness(
        *, result: FourArmResult,
) -> TVSArbiterV3Witness:
    best_static = max(
        float(result.transcript_mean_retention),
        float(result.shared_mean_retention),
        float(result.merge_mean_retention))
    return TVSArbiterV3Witness(
        n_turns=int(result.n_turns),
        transcript_mean_retention=float(
            result.transcript_mean_retention),
        shared_mean_retention=float(
            result.shared_mean_retention),
        merge_mean_retention=float(
            result.merge_mean_retention),
        arbiter_mean_retention=float(
            result.arbiter_mean_retention),
        pick_rate_transcript=float(
            result.pick_rate_transcript),
        pick_rate_shared=float(
            result.pick_rate_shared),
        pick_rate_merge=float(
            result.pick_rate_merge),
        pick_rate_abstain_fallback=float(
            result.pick_rate_abstain_fallback),
        arbiter_strict_dominance_over_static=bool(
            float(result.arbiter_mean_retention)
            >= float(best_static) - 1e-9),
    )


# =============================================================================
# Verifier
# =============================================================================

W54_TVS_ARBITER_V3_VERIFIER_FAILURE_MODES: tuple[str, ...] = (
    "w54_tvs_arbiter_v3_pick_rate_invalid",
    "w54_tvs_arbiter_v3_arbiter_below_static",
    "w54_tvs_arbiter_v3_n_turns_below_floor",
)


def verify_tvs_arbiter_v3_witness(
        witness: TVSArbiterV3Witness,
        *,
        require_strict_dominance: bool = False,
        min_n_turns: int | None = None,
) -> dict[str, Any]:
    failures: list[str] = []
    s = (
        float(witness.pick_rate_transcript)
        + float(witness.pick_rate_shared)
        + float(witness.pick_rate_merge)
        + float(witness.pick_rate_abstain_fallback))
    if abs(s - 1.0) > 1e-6 and s != 0.0:
        failures.append(
            "w54_tvs_arbiter_v3_pick_rate_invalid")
    if (require_strict_dominance
            and not witness
            .arbiter_strict_dominance_over_static):
        failures.append(
            "w54_tvs_arbiter_v3_arbiter_below_static")
    if (min_n_turns is not None
            and witness.n_turns < int(min_n_turns)):
        failures.append(
            "w54_tvs_arbiter_v3_n_turns_below_floor")
    return {
        "ok": (len(failures) == 0),
        "failures": failures,
        "witness_cid": witness.cid(),
    }


__all__ = [
    "W54_TVS_ARBITER_V3_SCHEMA_VERSION",
    "W54_TVS_ARM_MERGE_CONSENSUS",
    "W54_TVS_ARM_ABSTAIN_FALLBACK_TRANSCRIPT",
    "W54_DEFAULT_TVS_V3_MERGE_FLOOR",
    "W54_DEFAULT_TVS_V3_ABSTAIN_FALLBACK",
    "W54_TVS_ARBITER_V3_VERIFIER_FAILURE_MODES",
    "ArbiterDecisionV3",
    "FourArmResult",
    "TVSArbiterV3Witness",
    "arbiter_decide_v3",
    "four_arm_compare",
    "emit_tvs_arbiter_v3_witness",
    "verify_tvs_arbiter_v3_witness",
]
