"""W114 / COO-9 — per-model post-cutoff certification + instrument-frontier gate.

W113 confirmed the W112 +10 pp was contamination EXPOSURE (Maverick FAILs on a
verifiably-resistant LiveCodeBench slice exactly as 70B did) and left ONE forward
move: build a NEW instrument certifiably contamination-resistant for a *stronger*
model than Maverick, and earn a clean shot on it.  W113's Lane β found all three
reachable tier-2 stronger models have UNKNOWN cutoffs on the pinned corpus, so it
named a ``release_v7+`` instrument as the next requirement — WITHOUT verifying
whether such a release exists or whether those models have since disclosed a
cutoff.

This module is the W114 instrument: it makes the certification decision
**mechanical and primary-source-grounded**.  It answers, for the LATEST REAL
LiveCodeBench release and the OFFICIAL model cutoffs (verified 2026-05-29 from
primary sources — see ``W114_CUTOFF_PROVENANCE``), whether ANY reachable
stronger-than-Maverick model is CERTIFIABLY contamination-resistant on the
available instrument, and if not, exactly why (the load-bearing blocker).

It does NOT duplicate the W113 logic — it IMPORTS the W113 model-cutoff registry
(``MODEL_TRAINING_CUTOFFS`` / ``cutoff_boundary_for_model_v1``), the resistance
predicate, ``MIN_RESISTANT_SLICE``, and the tier-2 ranking, and adds two things:

1. ``LatestResistantInstrumentV1`` — the verified latest resistant FUNCTIONAL
   instrument (release pin + SHA + functional date coverage + month histogram),
   with a month-granular ``n_functional_resistant_after`` so the ">= 30 problems
   strictly after a cutoff" check is computable without re-loading the corpus
   (the script re-verifies the histogram against the live corpus when present).
2. ``W114_CUTOFF_PROVENANCE`` + ``certify_model_v1`` — the C1..C4 certification
   gate per candidate, scored against the latest instrument, with a typed reason
   and a consistency guard that the W113 registry confidence matches the
   W114-verified confidence.

The KEY W114 finding this encodes: the latest resistant FUNCTIONAL instrument
(LCB ``release_v6``; functional problems 2025-01-11..2025-04-05; frontier
2025-04-05) has AGED OUT relative to the reachable frontier-model class.  A >= 30
functional resistant slice requires a KNOWN cutoff <= ~Jan-2025; the reachable
stronger-than-Maverick frontier models (Qwen3-Coder-480B 2025-07, DeepSeek-V4-pro
2025+, Mistral-Small-4 2026-03) have OFFICIALLY UNDISCLOSED cutoffs (C1 fails)
AND, where estimable, cutoffs at/after Apr-2025 (C2 fails) — the gaps COMPOUND.
Maverick is the only reachable model with a KNOWN cutoff, and it is already
SETTLED on this instrument (W113 resistant FAIL; C4 fails).  Hence NO new
stronger-model resistant pilot is certifiable on the latest real data, and the
blocker is a precise dated spend gate (``W114-L-RESISTANT-INSTRUMENT-FRONTIER-
LAGS-MODEL-FRONTIER-CAP`` + ``W114-T-STRONGER-MODEL-CUTOFFS-OFFICIALLY-
UNDISCLOSED``).

Pure / deterministic / NIM-free / read-only.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Sequence

from .livecodebench_resistant_slice_v1 import (
    CONFIDENCE_KNOWN,
    MIN_RESISTANT_SLICE,
    ModelCutoffV1,
    cutoff_boundary_for_model_v1,
)

W114_CERTIFICATION_V1_SCHEMA_VERSION: str = (
    "coordpy.stronger_model_cutoff_certification_v1.v1")

# Overall certification verdicts.
VERDICT_CERTIFIABLE: str = "CERTIFIABLE_STRONGER_MODEL"
VERDICT_NONE: str = "NO_CERTIFIABLE_STRONGER_MODEL"


@dataclasses.dataclass(frozen=True)
class CutoffProvenanceV1:
    """Primary-source provenance for one model's training cutoff (W114).

    ``verified_confidence`` MUST match the W113 ``MODEL_TRAINING_CUTOFFS``
    confidence for the same model (a consistency guard; divergence is flagged by
    ``certify_model_v1``).  ``primary_source`` is the official model card / vendor
    blog / dataset metadata the verification rests on (NOT a third-party
    aggregator or model memory).
    """

    verified_confidence: str
    primary_source: str
    verified_on: str
    note: str


# W114 official-source verification pass (2026-05-29).  PRIMARY sources only:
# official model cards + vendor blogs/docs + dataset metadata.  No guessing from
# memory; no third-party aggregators used as authority.
W114_CUTOFF_PROVENANCE: dict[str, CutoffProvenanceV1] = {
    "meta/llama-4-maverick-17b-128e-instruct": CutoffProvenanceV1(
        verified_confidence=CONFIDENCE_KNOWN,
        primary_source="Official Llama 4 model card (llama.com / Meta GitHub "
                       "llama-models MODEL_CARD.md; NVIDIA build modelcard)",
        verified_on="2026-05-29",
        note="Pretraining knowledge cutoff stated as 'August 2024' => KNOWN; "
             "boundary 2024-08-31 (W112/W113). Already SETTLED on release_v6 "
             "(W113 resistant FAIL)."),
    "qwen/qwen3-coder-480b-a35b-instruct": CutoffProvenanceV1(
        verified_confidence="UNKNOWN",
        primary_source="Official HF model card "
                       "(Qwen/Qwen3-Coder-480B-A35B-Instruct) + Qwen blog "
                       "(qwenlm.github.io/blog/qwen3-coder)",
        verified_on="2026-05-29",
        note="Both primary sources state NO knowledge/data cutoff "
             "('NO CUTOFF STATED'). Released 2025-07; an estimable cutoff would "
             "be ~2025, at/after the Apr-2025 functional frontier => UNKNOWN and "
             "almost certainly C2-exposed even if disclosed."),
    "deepseek-ai/deepseek-v4-pro": CutoffProvenanceV1(
        verified_confidence="UNKNOWN",
        primary_source="DeepSeek official sources (no V4 cutoff disclosed; V3 "
                       "system-prompt extraction ~Jul-2024 is non-official)",
        verified_on="2026-05-29",
        note="No official V4 training cutoff published. V4 post-dates V3 "
             "(2024-12), so an estimable cutoff is >= early-2025 => UNKNOWN."),
    "mistralai/mistral-small-4-119b-2603": CutoffProvenanceV1(
        verified_confidence="UNKNOWN",
        primary_source="Official Mistral docs/model card "
                       "(mistral-small-4-0-26-03) + HF "
                       "(mistralai/Mistral-Small-4-119B-2603)",
        verified_on="2026-05-29",
        note="No training cutoff stated. Released 2026-03-16, which post-dates "
             "the ENTIRE release_v6 window => exposed even if a cutoff were "
             "disclosed => UNKNOWN."),
}


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass(frozen=True)
class LatestResistantInstrumentV1:
    """The verified latest resistant FUNCTIONAL LiveCodeBench instrument.

    ``functional_month_histogram`` maps ``"YYYY-MM" -> count`` over the FUNCTIONAL
    subset (the subset the W89 mechanism can attack).  Counting is MONTH-granular
    to match the granularity at which vendors disclose cutoffs (and the W113
    "exclude the entire cutoff month" conservatism): a problem-month is resistant
    for a cutoff iff the month is strictly LATER than the cutoff's month.
    """

    release: str
    jsonl_sha256: str
    n_functional: int
    functional_date_min: str
    functional_date_max: str
    functional_month_histogram: dict[str, int]
    note: str

    def n_functional_resistant_after(self, boundary_date: str) -> int:
        """Functional problems in months strictly after the cutoff month.

        ``boundary_date`` is the registry boundary (the last in-distribution
        day, ``YYYY-MM-DD``); its month is ``boundary_date[:7]``.  A problem-month
        ``M`` is resistant iff ``M > cutoff_month`` (ISO ``YYYY-MM`` strings sort
        chronologically).  This is the conservative, disclosure-granularity test.
        """
        cutoff_month = str(boundary_date)[:7]
        return sum(
            cnt for ym, cnt in self.functional_month_histogram.items()
            if str(ym) > cutoff_month)

    def to_dict(self) -> dict[str, Any]:
        return {
            "release": str(self.release),
            "jsonl_sha256": str(self.jsonl_sha256),
            "n_functional": int(self.n_functional),
            "functional_date_min": str(self.functional_date_min),
            "functional_date_max": str(self.functional_date_max),
            "functional_month_histogram": dict(self.functional_month_histogram),
            "note": str(self.note),
        }


# The verified latest resistant FUNCTIONAL instrument (W114 § 2/§ 3).  Histogram
# computed NIM-free from the SHA-pinned local release_v6 test6.jsonl functional
# subset (63 problems); the script re-verifies it against the live corpus.
LATEST_RESISTANT_INSTRUMENT: LatestResistantInstrumentV1 = (
    LatestResistantInstrumentV1(
        release="release_v6",
        jsonl_sha256=(
            "bb4c364f71921c4495a6ad15abe1a927350b720009f4933e2e71f8af0f6fd1f5"),
        n_functional=63,
        functional_date_min="2025-01-11",
        functional_date_max="2025-04-05",
        functional_month_histogram={
            "2025-01": 14, "2025-02": 20, "2025-03": 27, "2025-04": 2},
        note="LCB code_generation_lite release_v6 (HF file tree: test6.jsonl is "
             "the highest-numbered => latest; no test7+ as of 2026-05-29). Full "
             "release May-2023..Apr-2025; the FUNCTIONAL/lite subset attacked by "
             "the W89 mechanism is 63 problems, all 2025-01..04. Instrument "
             "frontier date 2025-04-05."))


@dataclasses.dataclass(frozen=True)
class StrongerModelCandidateV1:
    """A reachable stronger-than-70B candidate for resistant certification."""

    model_id: str
    family: str
    rank_tier: int          # 1 = Maverick (tier-1); 2 = tier-2 (lower preferred)
    rank_within_tier: int
    reachable: bool         # fixed prior (W112 sweep); not re-probed in W114
    strictly_stronger_than_70b: bool
    same_budget_comparable: bool
    already_settled_on_instrument: bool  # has a recorded resistant verdict here
    settled_note: str


# Reachable stronger-than-70B candidate set (W112 sweep ranking, fixed prior).
# Maverick is tier-1 and SETTLED on release_v6 (W113); the three tier-2 are the
# genuinely-stronger-than-Maverick frontier targets.
STRONGER_MODEL_CANDIDATES: tuple[StrongerModelCandidateV1, ...] = (
    StrongerModelCandidateV1(
        model_id="meta/llama-4-maverick-17b-128e-instruct", family="llama4",
        rank_tier=1, rank_within_tier=1, reachable=True,
        strictly_stronger_than_70b=True, same_budget_comparable=True,
        already_settled_on_instrument=True,
        settled_note="W113 resistant pilot on release_v6 (CID 2afc318c): "
                     "B-A1=+0.00pp FAIL => EXPOSURE_CONFIRMED. Settled cell."),
    StrongerModelCandidateV1(
        model_id="qwen/qwen3-coder-480b-a35b-instruct", family="qwen",
        rank_tier=2, rank_within_tier=1, reachable=True,
        strictly_stronger_than_70b=True, same_budget_comparable=True,
        already_settled_on_instrument=False, settled_note=""),
    StrongerModelCandidateV1(
        model_id="deepseek-ai/deepseek-v4-pro", family="deepseek",
        rank_tier=2, rank_within_tier=2, reachable=True,
        strictly_stronger_than_70b=True, same_budget_comparable=True,
        already_settled_on_instrument=False, settled_note=""),
    StrongerModelCandidateV1(
        model_id="mistralai/mistral-small-4-119b-2603", family="mistral",
        rank_tier=2, rank_within_tier=3, reachable=True,
        strictly_stronger_than_70b=True, same_budget_comparable=True,
        already_settled_on_instrument=False, settled_note=""),
)


@dataclasses.dataclass(frozen=True)
class ModelCertificationV1:
    model_id: str
    rank_tier: int
    rank_within_tier: int
    cutoff_boundary: str
    cutoff_confidence: str
    verified_confidence: str
    confidence_consistent: bool
    primary_source: str
    n_functional_resistant: int
    # C1..C4 gates:
    c1_cutoff_known: bool
    c2_enough_resistant: bool
    c3_reachable_stronger_comparable: bool
    c4_not_already_settled: bool
    certifiable_for_new_pilot: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": str(self.model_id),
            "rank_tier": int(self.rank_tier),
            "rank_within_tier": int(self.rank_within_tier),
            "cutoff_boundary": str(self.cutoff_boundary),
            "cutoff_confidence": str(self.cutoff_confidence),
            "verified_confidence": str(self.verified_confidence),
            "confidence_consistent": bool(self.confidence_consistent),
            "primary_source": str(self.primary_source),
            "n_functional_resistant": int(self.n_functional_resistant),
            "c1_cutoff_known": bool(self.c1_cutoff_known),
            "c2_enough_resistant": bool(self.c2_enough_resistant),
            "c3_reachable_stronger_comparable": bool(
                self.c3_reachable_stronger_comparable),
            "c4_not_already_settled": bool(self.c4_not_already_settled),
            "certifiable_for_new_pilot": bool(self.certifiable_for_new_pilot),
            "reason": str(self.reason),
        }


def certify_model_v1(
        candidate: StrongerModelCandidateV1,
        *,
        instrument: LatestResistantInstrumentV1 = LATEST_RESISTANT_INSTRUMENT,
) -> ModelCertificationV1:
    """Apply the C1..C4 certification gate to one candidate on one instrument.

    ``CERTIFIABLE_for_new_pilot`` ⟺ C1 ∧ C2 ∧ C3 ∧ C4:

    * **C1** the model's cutoff is ``KNOWN`` (primary-source-stated).
    * **C2** the instrument has >= ``MIN_RESISTANT_SLICE`` functional problems in
      months strictly after the KNOWN cutoff.
    * **C3** reachable AND strictly stronger than 70B AND same-budget-comparable.
    * **C4** not already settled on this instrument (Maverick is, W113).
    """
    cutoff: ModelCutoffV1 = cutoff_boundary_for_model_v1(candidate.model_id)
    prov = W114_CUTOFF_PROVENANCE.get(candidate.model_id)
    verified_conf = prov.verified_confidence if prov else "UNKNOWN"
    primary_src = prov.primary_source if prov else "(no W114 provenance)"
    consistent = (str(cutoff.confidence) == str(verified_conf))

    n_res = instrument.n_functional_resistant_after(cutoff.boundary_date)

    c1 = cutoff.is_resistant_grade()                       # KNOWN
    c2 = n_res >= MIN_RESISTANT_SLICE
    c3 = bool(candidate.reachable
              and candidate.strictly_stronger_than_70b
              and candidate.same_budget_comparable)
    c4 = not candidate.already_settled_on_instrument
    certifiable = bool(c1 and c2 and c3 and c4)

    if certifiable:
        reason = (
            f"CERTIFIABLE: KNOWN cutoff {cutoff.boundary_date} + "
            f"{n_res} functional resistant problems (>= {MIN_RESISTANT_SLICE}) "
            f"on {instrument.release} + reachable/stronger/comparable + not "
            "already settled => earns the cheapest honest Phase-2 pilot.")
    elif not c1:
        reason = (
            f"NOT_CERTIFIABLE [C1 cutoff {cutoff.confidence}]: cannot certify "
            f"resistance against a non-KNOWN cutoff. {primary_src} => "
            f"{verified_conf}. {prov.note if prov else ''}")
    elif not c2:
        reason = (
            f"NOT_CERTIFIABLE [C2 instrument too old]: KNOWN cutoff "
            f"{cutoff.boundary_date} leaves only {n_res} functional resistant "
            f"problems on {instrument.release} (< {MIN_RESISTANT_SLICE}); the "
            f"instrument frontier {instrument.functional_date_max} does not "
            "post-date the cutoff by enough.")
    elif not c4:
        reason = (
            f"NOT_CERTIFIABLE_FOR_NEW_PILOT [C4 already settled]: "
            f"{candidate.settled_note} A second pilot on the same instrument "
            "has no verdict-changing power (redundant).")
    else:  # not c3
        reason = (
            "NOT_CERTIFIABLE [C3]: not reachable / not strictly stronger / not "
            "same-budget-comparable.")

    return ModelCertificationV1(
        model_id=candidate.model_id,
        rank_tier=candidate.rank_tier,
        rank_within_tier=candidate.rank_within_tier,
        cutoff_boundary=cutoff.boundary_date,
        cutoff_confidence=cutoff.confidence,
        verified_confidence=verified_conf,
        confidence_consistent=consistent,
        primary_source=primary_src,
        n_functional_resistant=n_res,
        c1_cutoff_known=c1, c2_enough_resistant=c2,
        c3_reachable_stronger_comparable=c3, c4_not_already_settled=c4,
        certifiable_for_new_pilot=certifiable, reason=reason)


@dataclasses.dataclass(frozen=True)
class CertificationDecisionV1:
    schema: str
    instrument: dict[str, Any]
    min_resistant_slice: int
    per_model: tuple[ModelCertificationV1, ...]
    verdict: str
    target_model: str
    maverick_certifiable_but_settled: bool
    w115_blocker: str
    next_instrument_requirement: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "instrument": self.instrument,
            "min_resistant_slice": int(self.min_resistant_slice),
            "per_model": [m.to_dict() for m in self.per_model],
            "verdict": str(self.verdict),
            "target_model": str(self.target_model),
            "maverick_certifiable_but_settled": bool(
                self.maverick_certifiable_but_settled),
            "w115_blocker": str(self.w115_blocker),
            "next_instrument_requirement": str(self.next_instrument_requirement),
        }

    def cid(self) -> str:
        return _sha256_hex({"kind": "w114_certification_decision_v1",
                            "decision": self.to_dict()})


def decide_certification_v1(
        *,
        candidates: Sequence[StrongerModelCandidateV1] = (
            STRONGER_MODEL_CANDIDATES),
        instrument: LatestResistantInstrumentV1 = LATEST_RESISTANT_INSTRUMENT,
) -> CertificationDecisionV1:
    """LOCKED W114 certification decision (RUNBOOK_W114 § 4/§ 6/§ 7).

    Verdict = ``CERTIFIABLE_STRONGER_MODEL`` (with the highest-ranked certifiable
    target) iff any candidate clears C1..C4; else ``NO_CERTIFIABLE_STRONGER_
    MODEL`` and the dated blocker is named.
    """
    per_model = tuple(
        certify_model_v1(c, instrument=instrument) for c in candidates)
    certifiable = [m for m in per_model if m.certifiable_for_new_pilot]
    certifiable_sorted = sorted(
        certifiable, key=lambda m: (m.rank_tier, m.rank_within_tier))
    target = certifiable_sorted[0].model_id if certifiable_sorted else ""
    verdict = VERDICT_CERTIFIABLE if certifiable_sorted else VERDICT_NONE

    # Maverick is C1∧C2∧C3 but C4-blocked (settled): record the distinction.
    mav = next(
        (m for m in per_model
         if m.model_id == "meta/llama-4-maverick-17b-128e-instruct"), None)
    mav_settled = bool(
        mav and mav.c1_cutoff_known and mav.c2_enough_resistant
        and mav.c3_reachable_stronger_comparable
        and not mav.c4_not_already_settled)

    blocker = (
        "NONE (a certifiable stronger model exists; run the pilot)."
        if certifiable_sorted else
        "The latest resistant FUNCTIONAL instrument "
        f"({instrument.release}; functional {instrument.functional_date_min}.."
        f"{instrument.functional_date_max}) does not post-date a single "
        "reachable stronger-than-Maverick model's VERIFIABLE cutoff. A >= "
        f"{MIN_RESISTANT_SLICE} functional resistant slice requires a KNOWN "
        "cutoff <= ~2025-01; the reachable frontier models (Qwen3-Coder-480B "
        "2025-07, DeepSeek-V4-pro 2025+, Mistral-Small-4 2026-03) have OFFICIALLY "
        "UNDISCLOSED cutoffs (C1) and, where estimable, cutoffs at/after the "
        "Apr-2025 frontier (C2) — the gaps COMPOUND. Maverick is the only KNOWN "
        "cutoff and is already SETTLED here (W113). => NO new stronger-model "
        "resistant pilot is certifiable on the latest real data.")
    nxt = (
        "A resistant FUNCTIONAL instrument with >= "
        f"{MIN_RESISTANT_SLICE} problems dated strictly after a reachable "
        "frontier model's KNOWN cutoff. Concretely: (a) a future LCB release_v7+ "
        "(or equivalent freshly-dated functional benchmark) with post-2025-04 "
        "functional problems, operator-fetched + SHA-pinned AND admitted to the "
        "loader; AND (b) a reachable stronger-than-Maverick model whose cutoff is "
        "officially DISCLOSED (KNOWN) and earlier than those problems. Neither "
        "holds as of 2026-05-29.")
    return CertificationDecisionV1(
        schema=W114_CERTIFICATION_V1_SCHEMA_VERSION,
        instrument=instrument.to_dict(),
        min_resistant_slice=MIN_RESISTANT_SLICE,
        per_model=per_model,
        verdict=verdict,
        target_model=target,
        maverick_certifiable_but_settled=mav_settled,
        w115_blocker=blocker,
        next_instrument_requirement=nxt)


__all__ = [
    "W114_CERTIFICATION_V1_SCHEMA_VERSION",
    "VERDICT_CERTIFIABLE",
    "VERDICT_NONE",
    "CutoffProvenanceV1",
    "W114_CUTOFF_PROVENANCE",
    "LatestResistantInstrumentV1",
    "LATEST_RESISTANT_INSTRUMENT",
    "StrongerModelCandidateV1",
    "STRONGER_MODEL_CANDIDATES",
    "ModelCertificationV1",
    "certify_model_v1",
    "CertificationDecisionV1",
    "decide_certification_v1",
]
