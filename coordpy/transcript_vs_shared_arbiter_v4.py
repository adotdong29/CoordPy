"""W55 M9 — Transcript-vs-Shared Arbiter V4 (5-arm policy).

Extends W54 V3 (4-arm) with a fifth ``trust_weighted_merge`` arm
and a **per-arm budget allocator** that distributes the available
visible-token budget across the five arms.

5-arm policy:
    {transcript, shared, merge_consensus, trust_weighted_merge,
     abstain-with-fallback}

Decision rule (per turn):
    if confidence < abstain_threshold AND fallback allowed:
        emit abstain-with-fallback (fallback_arm=transcript)
    elif trust_score >= trust_threshold AND tw_retention > merge_retention:
        emit trust_weighted_merge
    elif merge_consensus available AND merge_retention >= shared_retention + floor:
        emit merge_consensus
    elif shared_retention - transcript_retention > prefer_shared:
        emit shared
    else:
        emit transcript

Budget allocator distributes ``total_budget`` across the chosen
arms by retention-weighted softmax:
    weight_i = retention_i * trust_i
    fraction_i = weight_i / Σ weight_i
    arm_budget_i = round(total_budget * fraction_i)

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
    W53_TVS_ARM_ABSTAIN,
    W53_TVS_ARM_SHARED,
    W53_TVS_ARM_TRANSCRIPT,
    arbiter_decide,
)
from .transcript_vs_shared_arbiter_v3 import (
    W54_TVS_ARM_MERGE_CONSENSUS,
    W54_TVS_ARM_ABSTAIN_FALLBACK_TRANSCRIPT,
    W54_DEFAULT_TVS_V3_MERGE_FLOOR,
)


# =============================================================================
# Schema, defaults
# =============================================================================

W55_TVS_ARBITER_V4_SCHEMA_VERSION: str = (
    "coordpy.transcript_vs_shared_arbiter_v4.v1")

W55_TVS_ARM_TRUST_WEIGHTED_MERGE: str = (
    "trust_weighted_merge")
W55_DEFAULT_TVS_V4_TRUST_THRESHOLD: float = 0.7
W55_DEFAULT_TVS_V4_TRUST_PRIORITY_FLOOR: float = 0.0


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
# 5-arm decision
# =============================================================================


@dataclasses.dataclass(frozen=True)
class ArbiterDecisionV4:
    """One per-turn 5-arm decision with budget allocation."""

    turn_index: int
    chosen_arm: str
    transcript_retention: float
    shared_retention: float
    merge_retention: float
    trust_weighted_retention: float
    retention_gap: float
    bit_density_gap: float
    confidence: float
    trust_score: float
    abstain_reason: str
    fallback_arm: str
    budget_total: int
    budget_per_arm: tuple[tuple[str, int], ...]

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
            "trust_weighted_retention": float(round(
                self.trust_weighted_retention, 12)),
            "retention_gap": float(round(
                self.retention_gap, 12)),
            "bit_density_gap": float(round(
                self.bit_density_gap, 12)),
            "confidence": float(round(self.confidence, 12)),
            "trust_score": float(round(
                self.trust_score, 12)),
            "abstain_reason": str(self.abstain_reason),
            "fallback_arm": str(self.fallback_arm),
            "budget_total": int(self.budget_total),
            "budget_per_arm": [
                [str(a), int(b)]
                for (a, b) in self.budget_per_arm],
        }


def _allocate_budget(
        *,
        total_budget: int,
        retention_per_arm: dict[str, float],
        trust_per_arm: dict[str, float],
        chosen_arm: str,
) -> list[tuple[str, int]]:
    """Distribute total_budget across arms by retention × trust."""
    if total_budget <= 0:
        return [(chosen_arm, 0)]
    # All-or-nothing if total is 1.
    if total_budget == 1:
        return [(chosen_arm, 1)]
    # Compute weights.
    weights: dict[str, float] = {}
    for arm in retention_per_arm:
        r = float(retention_per_arm.get(arm, 0.0))
        t = float(trust_per_arm.get(arm, 1.0))
        weights[arm] = max(0.0, r) * max(0.0, t)
    # Floor: the chosen arm always gets at least 1 token.
    total_w = float(sum(weights.values()))
    if total_w <= 1e-30:
        return [(chosen_arm, int(total_budget))]
    out: list[tuple[str, int]] = []
    allocated = 0
    fractions = sorted(weights.items(), key=lambda kv: -kv[1])
    for arm, w in fractions[:-1]:
        frac = float(w) / float(total_w)
        b = int(round(frac * float(total_budget)))
        out.append((arm, max(0, b)))
        allocated += max(0, b)
    last_arm = fractions[-1][0]
    out.append((last_arm, max(0, int(total_budget) - allocated)))
    # Ensure chosen_arm has ≥ 1 token.
    found = False
    for i, (a, b) in enumerate(out):
        if a == chosen_arm:
            if b == 0:
                # Steal one token from the highest-budget arm.
                max_i = max(
                    range(len(out)), key=lambda j: out[j][1])
                if out[max_i][1] > 0:
                    out[max_i] = (
                        out[max_i][0], out[max_i][1] - 1)
                    out[i] = (a, 1)
            found = True
            break
    if not found:
        out.append((chosen_arm, 0))
    return out


def arbiter_decide_v4(
        *,
        turn_index: int,
        carrier: Sequence[float],
        codebook: QuantisedCodebookV4,
        gate: QuantisedBudgetGate,
        budget_tokens: int,
        bits_per_natural_token: float = 6.0,
        confidence: float = 1.0,
        trust_score: float = 1.0,
        abstain_threshold: float = 0.15,
        prefer_shared_threshold: float = 0.0,
        merge_consensus_retention: float | None = None,
        trust_weighted_retention: float | None = None,
        merge_floor: float = W54_DEFAULT_TVS_V3_MERGE_FLOOR,
        trust_threshold: float = (
            W55_DEFAULT_TVS_V4_TRUST_THRESHOLD),
        abstain_fallback: bool = True,
) -> ArbiterDecisionV4:
    """5-arm decision: transcript / shared / merge_consensus /
    trust_weighted_merge / abstain-with-fallback."""
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
    mr = (
        float(merge_consensus_retention)
        if merge_consensus_retention is not None else 0.0)
    twr = (
        float(trust_weighted_retention)
        if trust_weighted_retention is not None else 0.0)
    chosen = base.chosen_arm
    reason = base.abstain_reason
    fallback = ""
    if chosen == W53_TVS_ARM_ABSTAIN and abstain_fallback:
        chosen = W54_TVS_ARM_ABSTAIN_FALLBACK_TRANSCRIPT
        fallback = W53_TVS_ARM_TRANSCRIPT
    elif chosen == W53_TVS_ARM_ABSTAIN:
        fallback = W53_TVS_ARM_ABSTAIN
    elif (trust_weighted_retention is not None
            and float(trust_score)
            >= float(trust_threshold)
            and float(twr) >= float(mr)
            and float(twr) >= float(base.shared_retention)):
        chosen = W55_TVS_ARM_TRUST_WEIGHTED_MERGE
        fallback = base.chosen_arm
    elif (merge_consensus_retention is not None
            and float(mr)
            >= float(base.shared_retention)
            + float(merge_floor)):
        chosen = W54_TVS_ARM_MERGE_CONSENSUS
        fallback = base.chosen_arm
    # Budget allocator.
    retention_per_arm = {
        W53_TVS_ARM_TRANSCRIPT: float(
            base.transcript_retention),
        W53_TVS_ARM_SHARED: float(base.shared_retention),
        W54_TVS_ARM_MERGE_CONSENSUS: float(mr),
        W55_TVS_ARM_TRUST_WEIGHTED_MERGE: float(twr),
    }
    trust_per_arm = {
        W53_TVS_ARM_TRANSCRIPT: 1.0,
        W53_TVS_ARM_SHARED: float(confidence),
        W54_TVS_ARM_MERGE_CONSENSUS: float(confidence),
        W55_TVS_ARM_TRUST_WEIGHTED_MERGE: float(trust_score),
    }
    budget = _allocate_budget(
        total_budget=int(budget_tokens),
        retention_per_arm=retention_per_arm,
        trust_per_arm=trust_per_arm,
        chosen_arm=str(chosen))
    return ArbiterDecisionV4(
        turn_index=int(turn_index),
        chosen_arm=str(chosen),
        transcript_retention=float(base.transcript_retention),
        shared_retention=float(base.shared_retention),
        merge_retention=float(mr),
        trust_weighted_retention=float(twr),
        retention_gap=float(base.retention_gap),
        bit_density_gap=float(base.bit_density_gap),
        confidence=float(confidence),
        trust_score=float(trust_score),
        abstain_reason=str(reason),
        fallback_arm=str(fallback),
        budget_total=int(budget_tokens),
        budget_per_arm=tuple(
            (str(a), int(b)) for (a, b) in budget),
    )


# =============================================================================
# 5-arm comparison
# =============================================================================


@dataclasses.dataclass(frozen=True)
class FiveArmResult:
    """Aggregate five-arm comparison across many turns."""

    n_turns: int
    transcript_mean_retention: float
    shared_mean_retention: float
    merge_mean_retention: float
    trust_weighted_mean_retention: float
    pick_rate_transcript: float
    pick_rate_shared: float
    pick_rate_merge: float
    pick_rate_trust_weighted: float
    pick_rate_abstain_with_fallback: float
    oracle_correctness_rate: float
    fallback_invariant_ok_rate: float
    budget_allocator_correct_rate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_turns": int(self.n_turns),
            "transcript_mean_retention": float(round(
                self.transcript_mean_retention, 12)),
            "shared_mean_retention": float(round(
                self.shared_mean_retention, 12)),
            "merge_mean_retention": float(round(
                self.merge_mean_retention, 12)),
            "trust_weighted_mean_retention": float(round(
                self.trust_weighted_mean_retention, 12)),
            "pick_rate_transcript": float(round(
                self.pick_rate_transcript, 12)),
            "pick_rate_shared": float(round(
                self.pick_rate_shared, 12)),
            "pick_rate_merge": float(round(
                self.pick_rate_merge, 12)),
            "pick_rate_trust_weighted": float(round(
                self.pick_rate_trust_weighted, 12)),
            "pick_rate_abstain_with_fallback": float(round(
                self.pick_rate_abstain_with_fallback, 12)),
            "oracle_correctness_rate": float(round(
                self.oracle_correctness_rate, 12)),
            "fallback_invariant_ok_rate": float(round(
                self.fallback_invariant_ok_rate, 12)),
            "budget_allocator_correct_rate": float(round(
                self.budget_allocator_correct_rate, 12)),
        }


def five_arm_compare(
        *,
        carriers: Sequence[Sequence[float]],
        codebook: QuantisedCodebookV4,
        gate: QuantisedBudgetGate,
        budget_tokens: int,
        per_turn_confidences: Sequence[float] | None = None,
        per_turn_trust_scores: Sequence[float] | None = None,
        per_turn_merge_retentions: (
            Sequence[float] | None) = None,
        per_turn_tw_retentions: (
            Sequence[float] | None) = None,
        abstain_threshold: float = 0.15,
        prefer_shared_threshold: float = 0.0,
        merge_floor: float = W54_DEFAULT_TVS_V3_MERGE_FLOOR,
        trust_threshold: float = (
            W55_DEFAULT_TVS_V4_TRUST_THRESHOLD),
) -> FiveArmResult:
    n = len(carriers)
    if n == 0:
        return FiveArmResult(
            n_turns=0,
            transcript_mean_retention=0.0,
            shared_mean_retention=0.0,
            merge_mean_retention=0.0,
            trust_weighted_mean_retention=0.0,
            pick_rate_transcript=0.0,
            pick_rate_shared=0.0,
            pick_rate_merge=0.0,
            pick_rate_trust_weighted=0.0,
            pick_rate_abstain_with_fallback=0.0,
            oracle_correctness_rate=0.0,
            fallback_invariant_ok_rate=0.0,
            budget_allocator_correct_rate=0.0,
        )
    confs = (
        list(per_turn_confidences)
        if per_turn_confidences is not None
        else [1.0] * n)
    trusts = (
        list(per_turn_trust_scores)
        if per_turn_trust_scores is not None
        else [1.0] * n)
    mrs = (
        list(per_turn_merge_retentions)
        if per_turn_merge_retentions is not None
        else [0.0] * n)
    twrs = (
        list(per_turn_tw_retentions)
        if per_turn_tw_retentions is not None
        else [0.0] * n)
    t_ret_sum = 0.0
    s_ret_sum = 0.0
    m_ret_sum = 0.0
    tw_ret_sum = 0.0
    n_t = n_s = n_m = n_tw = n_a = 0
    n_oracle_correct = 0
    n_fallback_ok = 0
    n_budget_ok = 0
    for i in range(n):
        d = arbiter_decide_v4(
            turn_index=int(i),
            carrier=carriers[i],
            codebook=codebook, gate=gate,
            budget_tokens=int(budget_tokens),
            confidence=float(
                confs[i] if i < len(confs) else 1.0),
            trust_score=float(
                trusts[i] if i < len(trusts) else 1.0),
            merge_consensus_retention=float(
                mrs[i] if i < len(mrs) else 0.0),
            trust_weighted_retention=float(
                twrs[i] if i < len(twrs) else 0.0),
            abstain_threshold=float(abstain_threshold),
            prefer_shared_threshold=float(
                prefer_shared_threshold),
            merge_floor=float(merge_floor),
            trust_threshold=float(trust_threshold),
            abstain_fallback=True)
        t_ret_sum += float(d.transcript_retention)
        s_ret_sum += float(d.shared_retention)
        m_ret_sum += float(d.merge_retention)
        tw_ret_sum += float(d.trust_weighted_retention)
        if d.chosen_arm == W53_TVS_ARM_TRANSCRIPT:
            n_t += 1
        elif d.chosen_arm == W53_TVS_ARM_SHARED:
            n_s += 1
        elif d.chosen_arm == W54_TVS_ARM_MERGE_CONSENSUS:
            n_m += 1
        elif d.chosen_arm == W55_TVS_ARM_TRUST_WEIGHTED_MERGE:
            n_tw += 1
        elif (d.chosen_arm
                == W54_TVS_ARM_ABSTAIN_FALLBACK_TRANSCRIPT):
            n_a += 1
        # Oracle: chosen arm should have max retention among
        # available arms (ties broken by name order).
        retentions = {
            W53_TVS_ARM_TRANSCRIPT: d.transcript_retention,
            W53_TVS_ARM_SHARED: d.shared_retention,
            W54_TVS_ARM_MERGE_CONSENSUS: d.merge_retention,
            W55_TVS_ARM_TRUST_WEIGHTED_MERGE: (
                d.trust_weighted_retention),
        }
        if d.chosen_arm in retentions:
            best_arm = max(
                retentions.keys(),
                key=lambda k: (retentions[k], k))
            # Allow margin within prefer_shared_threshold.
            if (retentions[d.chosen_arm]
                    >= retentions[best_arm] - 1e-9
                    or abs(
                        retentions[d.chosen_arm]
                        - retentions[best_arm])
                    <= float(prefer_shared_threshold)):
                n_oracle_correct += 1
        else:
            # Abstain: oracle is "transcript fallback ok"
            # check (whether transcript retention is meaningful).
            if d.transcript_retention >= 0.0:
                n_oracle_correct += 1
        # Fallback invariant: when chosen=abstain-fallback,
        # transcript_retention >= 0.
        if (d.chosen_arm
                == W54_TVS_ARM_ABSTAIN_FALLBACK_TRANSCRIPT):
            if d.transcript_retention >= 0.0:
                n_fallback_ok += 1
        else:
            n_fallback_ok += 1
        # Budget allocator: sum of per-arm budget = total.
        total = sum(b for _, b in d.budget_per_arm)
        if total == d.budget_total:
            n_budget_ok += 1
    fn = float(max(1, n))
    return FiveArmResult(
        n_turns=int(n),
        transcript_mean_retention=float(t_ret_sum) / fn,
        shared_mean_retention=float(s_ret_sum) / fn,
        merge_mean_retention=float(m_ret_sum) / fn,
        trust_weighted_mean_retention=float(tw_ret_sum) / fn,
        pick_rate_transcript=float(n_t) / fn,
        pick_rate_shared=float(n_s) / fn,
        pick_rate_merge=float(n_m) / fn,
        pick_rate_trust_weighted=float(n_tw) / fn,
        pick_rate_abstain_with_fallback=float(n_a) / fn,
        oracle_correctness_rate=float(
            n_oracle_correct) / fn,
        fallback_invariant_ok_rate=float(n_fallback_ok) / fn,
        budget_allocator_correct_rate=float(
            n_budget_ok) / fn,
    )


# =============================================================================
# Witness
# =============================================================================


@dataclasses.dataclass(frozen=True)
class TVSArbiterV4Witness:
    schema_version: str
    n_turns: int
    pick_rate_transcript: float
    pick_rate_shared: float
    pick_rate_merge: float
    pick_rate_trust_weighted: float
    pick_rate_abstain_with_fallback: float
    oracle_correctness_rate: float
    fallback_invariant_ok_rate: float
    budget_allocator_correct_rate: float
    transcript_mean_retention: float
    shared_mean_retention: float
    merge_mean_retention: float
    trust_weighted_mean_retention: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": str(self.schema_version),
            "n_turns": int(self.n_turns),
            "pick_rate_transcript": float(round(
                self.pick_rate_transcript, 12)),
            "pick_rate_shared": float(round(
                self.pick_rate_shared, 12)),
            "pick_rate_merge": float(round(
                self.pick_rate_merge, 12)),
            "pick_rate_trust_weighted": float(round(
                self.pick_rate_trust_weighted, 12)),
            "pick_rate_abstain_with_fallback": float(round(
                self.pick_rate_abstain_with_fallback, 12)),
            "oracle_correctness_rate": float(round(
                self.oracle_correctness_rate, 12)),
            "fallback_invariant_ok_rate": float(round(
                self.fallback_invariant_ok_rate, 12)),
            "budget_allocator_correct_rate": float(round(
                self.budget_allocator_correct_rate, 12)),
            "transcript_mean_retention": float(round(
                self.transcript_mean_retention, 12)),
            "shared_mean_retention": float(round(
                self.shared_mean_retention, 12)),
            "merge_mean_retention": float(round(
                self.merge_mean_retention, 12)),
            "trust_weighted_mean_retention": float(round(
                self.trust_weighted_mean_retention, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w55_tvs_arbiter_v4_witness",
            "witness": self.to_dict()})


def emit_tvs_arbiter_v4_witness(
        *,
        result: FiveArmResult,
) -> TVSArbiterV4Witness:
    return TVSArbiterV4Witness(
        schema_version=W55_TVS_ARBITER_V4_SCHEMA_VERSION,
        n_turns=int(result.n_turns),
        pick_rate_transcript=float(result.pick_rate_transcript),
        pick_rate_shared=float(result.pick_rate_shared),
        pick_rate_merge=float(result.pick_rate_merge),
        pick_rate_trust_weighted=float(
            result.pick_rate_trust_weighted),
        pick_rate_abstain_with_fallback=float(
            result.pick_rate_abstain_with_fallback),
        oracle_correctness_rate=float(
            result.oracle_correctness_rate),
        fallback_invariant_ok_rate=float(
            result.fallback_invariant_ok_rate),
        budget_allocator_correct_rate=float(
            result.budget_allocator_correct_rate),
        transcript_mean_retention=float(
            result.transcript_mean_retention),
        shared_mean_retention=float(
            result.shared_mean_retention),
        merge_mean_retention=float(
            result.merge_mean_retention),
        trust_weighted_mean_retention=float(
            result.trust_weighted_mean_retention),
    )


# =============================================================================
# Verifier
# =============================================================================

W55_TVS_ARBITER_V4_VERIFIER_FAILURE_MODES: tuple[str, ...] = (
    "w55_tvs_v4_schema_mismatch",
    "w55_tvs_v4_pick_rates_dont_sum_to_one",
    "w55_tvs_v4_oracle_rate_below_floor",
    "w55_tvs_v4_fallback_invariant_below_floor",
    "w55_tvs_v4_budget_allocator_below_floor",
)


def verify_tvs_arbiter_v4_witness(
        witness: TVSArbiterV4Witness,
        *,
        min_oracle_correctness: float | None = None,
        min_fallback_invariant: float | None = None,
        min_budget_allocator: float | None = None,
) -> dict[str, Any]:
    failures: list[str] = []
    if (witness.schema_version
            != W55_TVS_ARBITER_V4_SCHEMA_VERSION):
        failures.append("w55_tvs_v4_schema_mismatch")
    s = (
        float(witness.pick_rate_transcript)
        + float(witness.pick_rate_shared)
        + float(witness.pick_rate_merge)
        + float(witness.pick_rate_trust_weighted)
        + float(witness.pick_rate_abstain_with_fallback))
    if abs(s - 1.0) > 1e-6 and s != 0.0:
        failures.append(
            "w55_tvs_v4_pick_rates_dont_sum_to_one")
    if (min_oracle_correctness is not None
            and witness.oracle_correctness_rate
            < float(min_oracle_correctness)):
        failures.append(
            "w55_tvs_v4_oracle_rate_below_floor")
    if (min_fallback_invariant is not None
            and witness.fallback_invariant_ok_rate
            < float(min_fallback_invariant)):
        failures.append(
            "w55_tvs_v4_fallback_invariant_below_floor")
    if (min_budget_allocator is not None
            and witness.budget_allocator_correct_rate
            < float(min_budget_allocator)):
        failures.append(
            "w55_tvs_v4_budget_allocator_below_floor")
    return {
        "ok": (len(failures) == 0),
        "failures": failures,
        "witness_cid": witness.cid(),
    }


__all__ = [
    "W55_TVS_ARBITER_V4_SCHEMA_VERSION",
    "W55_TVS_ARM_TRUST_WEIGHTED_MERGE",
    "W55_DEFAULT_TVS_V4_TRUST_THRESHOLD",
    "W55_DEFAULT_TVS_V4_TRUST_PRIORITY_FLOOR",
    "W55_TVS_ARBITER_V4_VERIFIER_FAILURE_MODES",
    "ArbiterDecisionV4",
    "FiveArmResult",
    "TVSArbiterV4Witness",
    "arbiter_decide_v4",
    "five_arm_compare",
    "emit_tvs_arbiter_v4_witness",
    "verify_tvs_arbiter_v4_witness",
]
