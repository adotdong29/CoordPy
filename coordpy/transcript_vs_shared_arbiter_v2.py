"""W53 M9 — Transcript-vs-Shared Arbiter V2.

Adds a learned/heuristic policy on top of W52's
``transcript_vs_shared_state`` matched-budget comparator. The
arbiter picks per-turn whether to emit:

* the **transcript** arm
* the **shared-latent** arm
* a **fallback abstention** ("emit only the parent capsule CID")

based on:

* per-turn estimated information density of the carrier (L2)
* the comparator's per-turn retention gap
* the corruption-robust carrier's confidence (if available)

Reports a 3-arm comparison on each turn. The "arbiter" arm
should strictly dominate (or tie) the better of the two static
arms across a budget sweep.
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
from .transcript_vs_shared_state import (
    TranscriptVsSharedComparisonResult,
    compare_transcript_vs_shared_state,
)


# =============================================================================
# Schema, defaults
# =============================================================================

W53_TVS_ARBITER_V2_SCHEMA_VERSION: str = (
    "coordpy.transcript_vs_shared_arbiter_v2.v1")

W53_TVS_ARM_TRANSCRIPT: str = "transcript"
W53_TVS_ARM_SHARED: str = "shared"
W53_TVS_ARM_ABSTAIN: str = "abstain"


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


def _l2(values: Sequence[float]) -> float:
    s = 0.0
    for v in values:
        s += float(v) * float(v)
    return float(math.sqrt(s))


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for i in range(n):
        ai = float(a[i])
        bi = float(b[i])
        dot += ai * bi
        na += ai * ai
        nb += bi * bi
    if na <= 1e-30 or nb <= 1e-30:
        return 0.0
    return float(dot / (math.sqrt(na) * math.sqrt(nb)))


# =============================================================================
# Per-turn decision
# =============================================================================


@dataclasses.dataclass(frozen=True)
class ArbiterDecision:
    """One per-turn decision."""

    turn_index: int
    chosen_arm: str
    transcript_retention: float
    shared_retention: float
    retention_gap: float
    bit_density_gap: float
    confidence: float
    abstain_reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_index": int(self.turn_index),
            "chosen_arm": str(self.chosen_arm),
            "transcript_retention": float(round(
                self.transcript_retention, 12)),
            "shared_retention": float(round(
                self.shared_retention, 12)),
            "retention_gap": float(round(
                self.retention_gap, 12)),
            "bit_density_gap": float(round(
                self.bit_density_gap, 12)),
            "confidence": float(round(
                self.confidence, 12)),
            "abstain_reason": str(self.abstain_reason),
        }


def arbiter_decide(
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
) -> ArbiterDecision:
    """Compare arms; choose the better unless confidence is low.

    Decision rule:
    * if confidence < abstain_threshold → abstain
    * elif shared_retention - transcript_retention > prefer_shared_threshold → shared
    * else → transcript
    """
    cmp = compare_transcript_vs_shared_state(
        carrier, codebook=codebook, gate=gate,
        budget_tokens=int(budget_tokens),
        bits_per_natural_token=float(
            bits_per_natural_token))
    chosen = W53_TVS_ARM_TRANSCRIPT
    reason = ""
    if float(confidence) < float(abstain_threshold):
        chosen = W53_TVS_ARM_ABSTAIN
        reason = (
            f"confidence_below_threshold:"
            f"{float(confidence):.4g}")
    elif (cmp.shared_retention_cosine
            - cmp.transcript_retention_cosine
            > float(prefer_shared_threshold)):
        chosen = W53_TVS_ARM_SHARED
    return ArbiterDecision(
        turn_index=int(turn_index),
        chosen_arm=str(chosen),
        transcript_retention=float(
            cmp.transcript_retention_cosine),
        shared_retention=float(
            cmp.shared_retention_cosine),
        retention_gap=float(cmp.retention_gap),
        bit_density_gap=float(cmp.bit_density_gap),
        confidence=float(confidence),
        abstain_reason=str(reason),
    )


# =============================================================================
# 3-arm comparison
# =============================================================================


@dataclasses.dataclass(frozen=True)
class ThreeArmResult:
    """Aggregate three-arm comparison across many turns."""

    n_turns: int
    transcript_mean_retention: float
    shared_mean_retention: float
    arbiter_mean_retention: float
    arbiter_pick_rate_shared: float
    arbiter_pick_rate_transcript: float
    arbiter_pick_rate_abstain: float
    decisions: tuple[ArbiterDecision, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_turns": int(self.n_turns),
            "transcript_mean_retention": float(round(
                self.transcript_mean_retention, 12)),
            "shared_mean_retention": float(round(
                self.shared_mean_retention, 12)),
            "arbiter_mean_retention": float(round(
                self.arbiter_mean_retention, 12)),
            "arbiter_pick_rate_shared": float(round(
                self.arbiter_pick_rate_shared, 12)),
            "arbiter_pick_rate_transcript": float(round(
                self.arbiter_pick_rate_transcript, 12)),
            "arbiter_pick_rate_abstain": float(round(
                self.arbiter_pick_rate_abstain, 12)),
            "decisions": [
                d.to_dict() for d in self.decisions],
        }


def three_arm_compare(
        carriers: Sequence[Sequence[float]],
        *,
        codebook: QuantisedCodebookV4,
        gate: QuantisedBudgetGate,
        budget_tokens: int = 3,
        per_turn_confidences: Sequence[float] | None = None,
        abstain_threshold: float = 0.15,
        prefer_shared_threshold: float = 0.0,
        bits_per_natural_token: float = 6.0,
) -> ThreeArmResult:
    """Run the 3-arm comparison across a batch of carriers."""
    decisions: list[ArbiterDecision] = []
    transcript_sum = 0.0
    shared_sum = 0.0
    arbiter_sum = 0.0
    n = 0
    n_pick_shared = 0
    n_pick_transcript = 0
    n_pick_abstain = 0
    confs = list(
        per_turn_confidences
        if per_turn_confidences is not None
        else [1.0] * len(carriers))
    for i, c in enumerate(carriers):
        conf = float(confs[i] if i < len(confs) else 1.0)
        d = arbiter_decide(
            turn_index=int(i),
            carrier=c,
            codebook=codebook, gate=gate,
            budget_tokens=int(budget_tokens),
            bits_per_natural_token=float(
                bits_per_natural_token),
            confidence=float(conf),
            abstain_threshold=float(abstain_threshold),
            prefer_shared_threshold=float(
                prefer_shared_threshold))
        decisions.append(d)
        transcript_sum += float(d.transcript_retention)
        shared_sum += float(d.shared_retention)
        # The arbiter retention is whichever arm was chosen
        # (abstention pays 0 retention — the explicit cost).
        if d.chosen_arm == W53_TVS_ARM_TRANSCRIPT:
            arbiter_sum += float(d.transcript_retention)
            n_pick_transcript += 1
        elif d.chosen_arm == W53_TVS_ARM_SHARED:
            arbiter_sum += float(d.shared_retention)
            n_pick_shared += 1
        else:
            arbiter_sum += 0.0
            n_pick_abstain += 1
        n += 1
    n_f = float(max(1, n))
    return ThreeArmResult(
        n_turns=int(n),
        transcript_mean_retention=float(
            transcript_sum / n_f),
        shared_mean_retention=float(shared_sum / n_f),
        arbiter_mean_retention=float(arbiter_sum / n_f),
        arbiter_pick_rate_shared=float(
            n_pick_shared / n_f),
        arbiter_pick_rate_transcript=float(
            n_pick_transcript / n_f),
        arbiter_pick_rate_abstain=float(
            n_pick_abstain / n_f),
        decisions=tuple(decisions),
    )


# =============================================================================
# Witness
# =============================================================================


@dataclasses.dataclass(frozen=True)
class TVSArbiterV2Witness:
    n_turns: int
    transcript_mean_retention: float
    shared_mean_retention: float
    arbiter_mean_retention: float
    arbiter_pick_rate_shared: float
    arbiter_pick_rate_transcript: float
    arbiter_pick_rate_abstain: float
    arbiter_strict_dominance_over_static: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_turns": int(self.n_turns),
            "transcript_mean_retention": float(round(
                self.transcript_mean_retention, 12)),
            "shared_mean_retention": float(round(
                self.shared_mean_retention, 12)),
            "arbiter_mean_retention": float(round(
                self.arbiter_mean_retention, 12)),
            "arbiter_pick_rate_shared": float(round(
                self.arbiter_pick_rate_shared, 12)),
            "arbiter_pick_rate_transcript": float(round(
                self.arbiter_pick_rate_transcript, 12)),
            "arbiter_pick_rate_abstain": float(round(
                self.arbiter_pick_rate_abstain, 12)),
            "arbiter_strict_dominance_over_static": bool(
                self.arbiter_strict_dominance_over_static),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w53_tvs_arbiter_v2_witness",
            "witness": self.to_dict()})


def emit_tvs_arbiter_v2_witness(
        *,
        result: ThreeArmResult,
) -> TVSArbiterV2Witness:
    best_static = max(
        float(result.transcript_mean_retention),
        float(result.shared_mean_retention))
    return TVSArbiterV2Witness(
        n_turns=int(result.n_turns),
        transcript_mean_retention=float(
            result.transcript_mean_retention),
        shared_mean_retention=float(
            result.shared_mean_retention),
        arbiter_mean_retention=float(
            result.arbiter_mean_retention),
        arbiter_pick_rate_shared=float(
            result.arbiter_pick_rate_shared),
        arbiter_pick_rate_transcript=float(
            result.arbiter_pick_rate_transcript),
        arbiter_pick_rate_abstain=float(
            result.arbiter_pick_rate_abstain),
        arbiter_strict_dominance_over_static=bool(
            float(result.arbiter_mean_retention)
            >= float(best_static)
            - 1e-9),
    )


# =============================================================================
# Verifier
# =============================================================================

W53_TVS_ARBITER_V2_VERIFIER_FAILURE_MODES: tuple[str, ...] = (
    "w53_tvs_arbiter_v2_pick_rate_invalid",
    "w53_tvs_arbiter_v2_arbiter_below_static",
    "w53_tvs_arbiter_v2_n_turns_below_floor",
)


def verify_tvs_arbiter_v2_witness(
        witness: TVSArbiterV2Witness,
        *,
        require_strict_dominance: bool = False,
        min_n_turns: int | None = None,
) -> dict[str, Any]:
    failures: list[str] = []
    s = (
        float(witness.arbiter_pick_rate_shared)
        + float(witness.arbiter_pick_rate_transcript)
        + float(witness.arbiter_pick_rate_abstain))
    if abs(s - 1.0) > 1e-6:
        failures.append(
            "w53_tvs_arbiter_v2_pick_rate_invalid")
    if (require_strict_dominance
            and not witness
            .arbiter_strict_dominance_over_static):
        failures.append(
            "w53_tvs_arbiter_v2_arbiter_below_static")
    if (min_n_turns is not None
            and witness.n_turns < int(min_n_turns)):
        failures.append(
            "w53_tvs_arbiter_v2_n_turns_below_floor")
    return {
        "ok": (len(failures) == 0),
        "failures": failures,
        "witness_cid": witness.cid(),
    }


__all__ = [
    "W53_TVS_ARBITER_V2_SCHEMA_VERSION",
    "W53_TVS_ARM_TRANSCRIPT",
    "W53_TVS_ARM_SHARED",
    "W53_TVS_ARM_ABSTAIN",
    "W53_TVS_ARBITER_V2_VERIFIER_FAILURE_MODES",
    "ArbiterDecision",
    "ThreeArmResult",
    "TVSArbiterV2Witness",
    "arbiter_decide",
    "three_arm_compare",
    "emit_tvs_arbiter_v2_witness",
    "verify_tvs_arbiter_v2_witness",
]
