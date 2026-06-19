"""W113 / COO-9 — Lane β: tier-2 stronger-model readiness + spend rule.

W112's reachability sweep found FOUR eligible reachable stronger-than-70B
models (405B is 404x6): the SELECTED tier-1 ``meta/llama-4-maverick-17b-128e-
instruct`` plus three tier-2 models — ``qwen/qwen3-coder-480b-a35b-instruct``,
``deepseek-ai/deepseek-v4-pro``, ``mistralai/mistral-small-4-119b-2603``.

W113's main lane spends the one earned expensive run on Maverick.  This module
locks the tier-2 follow-up readiness so W114 can move immediately on the RIGHT
target once the main-lane verdict lands — WITHOUT spending any tier-2 NIM in
W113.

It encodes three LOCKED rules:

1. ``TIER2_RANKING`` — the W113 tier-2 preference order (code-specialization
   first, then scale, then cross-vendor diversity), carried from the W112 sweep
   ``rank_tier``.
2. The SAME-FILTERED-SLICE APPLICABILITY rule (delegated to
   ``livecodebench_resistant_slice_v1.slice_resistant_for_model_v1``): a tier-2
   model may only be tested on a given resistant slice if that slice is
   CERTIFIABLY resistant for it — i.e. the model's cutoff is KNOWN and the
   slice's minimum date is strictly after it.  This is the W112 model-cutoff-
   relativity lesson as a spend gate.
3. The SPEND condition: a tier-2 pilot is worth NIM iff (a) the W113 main-lane
   verdict EARNS escalation (a clean resistant reopening or a sharp need to
   localize the exposure), AND (b) a slice CERTIFIABLY resistant for that tier-2
   model exists (>= MIN_RESISTANT_SLICE problems), AND (c) the budget is the
   same cheap K=5 single-seed shape.  Absent (a)+(b), spend = 0.

The crucial W113 finding this rule surfaces: the existing test6 slice
(2025-01..04) is certifiably resistant ONLY for Maverick (Aug-2024 cutoff).  All
three tier-2 models were released in 2025-2026 with UNDOCUMENTED cutoffs that
plausibly overlap or post-date the slice, so NONE of them has a certifiable
resistant slice in the pinned corpus — they would each need a LATER post-cutoff
LiveCodeBench instrument (release_v7+).  Hence: no tier-2 spend is even
ELIGIBLE in W113 regardless of the main-lane outcome, and the readiness
deliverable is the rule + the missing-instrument finding, not a queued run.

Pure / deterministic / NIM-free.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any

from .livecodebench_resistant_slice_v1 import (
    MIN_RESISTANT_SLICE,
    cutoff_boundary_for_model_v1,
    slice_resistant_for_model_v1,
)

W113_TIER2_READINESS_V1_SCHEMA_VERSION: str = (
    "coordpy.tier2_readiness_v1.v1")


@dataclasses.dataclass(frozen=True)
class Tier2Candidate:
    model_id: str
    family: str
    rank_tier: int          # 1 = tier-1 (Maverick); 2 = tier-2 (lower preferred)
    rank_within_tier: int   # tie-break within a tier (lower preferred)
    note: str


# The W113 tier-2 ranking (carried from the W112 sweep CANDIDATES rank_tier).
# Within tier-2: code-specialized + largest scale first (Qwen3-Coder-480B),
# then the frontier general chat models by scale, then cross-vendor.
TIER2_RANKING: tuple[Tier2Candidate, ...] = (
    Tier2Candidate(
        model_id="qwen/qwen3-coder-480b-a35b-instruct", family="qwen",
        rank_tier=2, rank_within_tier=1,
        note="STRONGEST reachable CODE-specialized frontier (480B-A35B MoE); "
             "preferred tier-2 IF a certifiably-resistant slice exists"),
    Tier2Candidate(
        model_id="deepseek-ai/deepseek-v4-pro", family="deepseek",
        rank_tier=2, rank_within_tier=2,
        note="frontier chat non-reasoning; strong code; cross-vendor"),
    Tier2Candidate(
        model_id="mistralai/mistral-small-4-119b-2603", family="mistral",
        rank_tier=2, rank_within_tier=3,
        note="119B non-reasoning instruct; > 70B scale; cross-vendor"),
)


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass(frozen=True)
class Tier2ApplicabilityV1:
    model_id: str
    rank_tier: int
    rank_within_tier: int
    cutoff_boundary: str
    cutoff_confidence: str
    slice_certifiably_resistant: bool
    applicability_reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": str(self.model_id),
            "rank_tier": int(self.rank_tier),
            "rank_within_tier": int(self.rank_within_tier),
            "cutoff_boundary": str(self.cutoff_boundary),
            "cutoff_confidence": str(self.cutoff_confidence),
            "slice_certifiably_resistant": bool(self.slice_certifiably_resistant),
            "applicability_reason": str(self.applicability_reason),
        }


def assess_tier2_applicability_v1(
        *, slice_date_min: str) -> tuple[Tier2ApplicabilityV1, ...]:
    """For each tier-2 candidate, is the given resistant slice (identified by
    its min contest_date) CERTIFIABLY resistant for it?  Delegates the
    certification to ``slice_resistant_for_model_v1`` (KNOWN-cutoff-only)."""
    out: list[Tier2ApplicabilityV1] = []
    for c in TIER2_RANKING:
        cutoff = cutoff_boundary_for_model_v1(c.model_id)
        ok, reason = slice_resistant_for_model_v1(
            slice_date_min=slice_date_min, model_id=c.model_id)
        out.append(Tier2ApplicabilityV1(
            model_id=c.model_id, rank_tier=c.rank_tier,
            rank_within_tier=c.rank_within_tier,
            cutoff_boundary=cutoff.boundary_date,
            cutoff_confidence=cutoff.confidence,
            slice_certifiably_resistant=bool(ok),
            applicability_reason=str(reason)))
    return tuple(out)


@dataclasses.dataclass(frozen=True)
class Tier2SpendDecisionV1:
    schema: str
    main_lane_outcome: str
    main_lane_earns_escalation: bool
    slice_date_min: str
    applicability: tuple[Tier2ApplicabilityV1, ...]
    n_certifiable_targets: int
    top_eligible_target: str
    spend_eligible: bool
    spend_rule: str
    next_instrument_if_blocked: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "main_lane_outcome": str(self.main_lane_outcome),
            "main_lane_earns_escalation": bool(self.main_lane_earns_escalation),
            "slice_date_min": str(self.slice_date_min),
            "applicability": [a.to_dict() for a in self.applicability],
            "n_certifiable_targets": int(self.n_certifiable_targets),
            "top_eligible_target": str(self.top_eligible_target),
            "spend_eligible": bool(self.spend_eligible),
            "spend_rule": str(self.spend_rule),
            "next_instrument_if_blocked": str(self.next_instrument_if_blocked),
        }

    def cid(self) -> str:
        return _sha256_hex({"kind": "w113_tier2_spend_decision_v1",
                            "decision": self.to_dict()})


# Main-lane outcomes that EARN a tier-2 escalation (the cross-scale interp
# outcomes).  A clean reopening earns a tier-2 replication; an EXPOSURE_CONFIRMED
# earns a tier-2 escalation ONLY to localize, and only if a certifiable slice
# exists (it does not, in W113) — so in practice the gate is dominated by
# applicability.
_OUTCOMES_THAT_EARN_ESCALATION: frozenset[str] = frozenset({
    "RESISTANT_SUPERIORITY_REOPENS",   # replicate the clean win at tier-2
    "EXPOSURE_CONFIRMED",              # localize the exposure at tier-2 (if slice)
})


def decide_tier2_spend_v1(
        *, main_lane_outcome: str, slice_date_min: str) -> Tier2SpendDecisionV1:
    """LOCKED tier-2 spend rule.  Spend is eligible iff the main-lane outcome
    earns escalation AND at least one tier-2 target has a CERTIFIABLY resistant
    slice (>= MIN_RESISTANT_SLICE is checked by the caller against the actual
    resistant partition; here we gate on certifiable-resistance of the slice's
    date range).  Otherwise spend = 0 and the next instrument is named."""
    earns = str(main_lane_outcome) in _OUTCOMES_THAT_EARN_ESCALATION
    applic = assess_tier2_applicability_v1(slice_date_min=slice_date_min)
    certifiable = [a for a in applic if a.slice_certifiably_resistant]
    n_cert = len(certifiable)
    top = ""
    if certifiable:
        certifiable_sorted = sorted(
            certifiable, key=lambda a: (a.rank_tier, a.rank_within_tier))
        top = certifiable_sorted[0].model_id
    spend_eligible = bool(earns and n_cert >= 1)
    rule = (
        "Tier-2 NIM is spent iff (a) the W113 main-lane verdict earns escalation "
        f"[{main_lane_outcome} -> {earns}] AND (b) >= 1 tier-2 model has a "
        f"CERTIFIABLY resistant slice [n_certifiable={n_cert}], with the same "
        "cheap K=5 single-seed budget. Top eligible by ranking = "
        f"{top or 'NONE'}.")
    nxt = (
        "All tier-2 cutoffs are UNKNOWN/post-2025 relative to the test6 slice "
        "(2025-01..04), so NONE is certifiably resistant on the pinned corpus. "
        "The next instrument for any tier-2 follow-up is a LATER date-filtered "
        "LiveCodeBench slice (release_v7+) with problems strictly after that "
        "model's (first KNOWN) cutoff — operator-fetched + SHA-pinned. Until "
        "then, tier-2 spend is BLOCKED on the missing instrument, independent of "
        "the main-lane outcome.")
    return Tier2SpendDecisionV1(
        schema=W113_TIER2_READINESS_V1_SCHEMA_VERSION,
        main_lane_outcome=str(main_lane_outcome),
        main_lane_earns_escalation=bool(earns),
        slice_date_min=str(slice_date_min),
        applicability=applic,
        n_certifiable_targets=int(n_cert),
        top_eligible_target=str(top),
        spend_eligible=bool(spend_eligible),
        spend_rule=str(rule),
        next_instrument_if_blocked=str(nxt))


__all__ = [
    "W113_TIER2_READINESS_V1_SCHEMA_VERSION",
    "Tier2Candidate",
    "TIER2_RANKING",
    "Tier2ApplicabilityV1",
    "assess_tier2_applicability_v1",
    "Tier2SpendDecisionV1",
    "decide_tier2_spend_v1",
    "MIN_RESISTANT_SLICE",
]
