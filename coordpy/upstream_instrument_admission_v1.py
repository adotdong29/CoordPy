"""W116 / COO-9 — durable UPSTREAM instrument-ADMISSION pipeline.

W115 verified, LIVE from primary sources, that the external frontier had not moved
(LCB ``release_v6`` still latest; no reachable stronger-than-Maverick model with a
KNOWN cutoff <= ~Jan-2025) and shipped a *snapshot-checker*
(``frontier_certification_pipeline_v1``) that makes the re-check push-button.  It
STOPPED at "$0 NIM, the blocker is supply".

W116 does NOT just re-run that checker and wait.  It ATTACKS the supply side:

1. it goes ONE LEVEL UPSTREAM of the pinned ``release_v6`` and re-verifies the LCB
   ecosystem at FOUR authoritative surfaces (the lite dataset file tree, the loader
   ``ALLOWED_FILES`` version list + the ``release_latest`` resolution, the full
   ``code_generation`` dataset + its README frontier, and the GitHub repo) — not
   just the single lite file tree W115 checked; and
2. it ATTACKS the model-disclosure side from PRIMARY sources, confirming the last
   hypothesized tier-2 candidate (``mistralai/mistral-small-4-119b-2603``) is now a
   REAL model (119B MoE, released 2026-03-16) whose official card discloses NO
   training cutoff.

This module is the durable asset W116 lands so the next clean shot (W117) is
mechanical the moment the upstream world changes.  It generalises the W115
snapshot-checker into a real upstream-ADMISSION pipeline by adding, on top of the
reused W113 registry + W114 ``certify_model_v1`` gate + W115
``run_frontier_certification_v1`` pipeline (explicit-import-only, NO duplication):

* ``AdmissibilityRuleV1`` / ``assess_instrument_admissibility_v1`` — a PRE-COMMITTED
  five-criterion rule (A1 authoritative source / A2 dated problems / A3 functional-
  compatible / A4 SHA-pinnable+admittable / A5 reproducible histogram) that decides
  whether an OBSERVED upstream instrument may be admitted to the certification
  matrix.  It REFUSES an aggregator / mirror / website-intro / "planned" rumor.
* ``UpstreamSupplySnapshotV1`` — the LIVE multi-surface upstream state as DATA
  (lite-tree releases, loader version list, ``release_latest`` resolution, full-
  dataset frontier, lastModified, per-instrument observations) wrapping the W115
  ``FrontierSnapshotV1`` (the model/instrument/disclosure state).
* ``detect_upstream_change_v1`` — the upstream-change detector: compares a new
  snapshot vs an encoded baseline and flags WHAT changed across surfaces (the W117
  update signal), richer than W115's single ``newer_release_available`` boolean.
* ``build_certifiable_slice_candidate_v1`` — the certifiable-slice candidate builder
  for a ``(model, admissible instrument)`` pair (reuses ``certify_model_v1``).
* ``W116_DISCLOSURE_MATRIX`` — the four-way per-model disclosure-status matrix
  (KNOWN / ESTIMATED-but-unusable / UNKNOWN / contradictory-or-stale) from the W116
  LIVE primary-source pass.  This is documentation/audit state; it does NOT feed the
  certification CID (which re-derives byte-identically to W114/W115 = ``258b6ed7``).
* ``run_upstream_admission_v1`` / ``W117FireConditionV1`` — the push-button runner +
  the exact W117 trigger.

The push-button W117 flow: the operator updates ONE ``UpstreamSupplySnapshotV1`` (a
newer observed+admitted release, a ``release_latest`` re-point, or a newly-disclosed
primary KNOWN cutoff) and re-runs ``run_upstream_admission_v1``.  Admission stays
disciplined: a release is certifiable ONLY after it is A1..A5-admissible AND
SHA-pinned + admitted to the loader; the pipeline never fabricates a release.

Pure / deterministic / NIM-free / read-only.  Explicit-import-only.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Sequence

from .livecodebench_loader_v1 import (
    LIVECODEBENCH_HF_DATASET,
    LIVECODEBENCH_KNOWN_RELEASES,
)
from .livecodebench_resistant_slice_v1 import (
    CONFIDENCE_KNOWN,
    MIN_RESISTANT_SLICE,
)
from .stronger_model_cutoff_certification_v1 import (
    LATEST_RESISTANT_INSTRUMENT,
    STRONGER_MODEL_CANDIDATES,
    LatestResistantInstrumentV1,
    StrongerModelCandidateV1,
    certify_model_v1,
)
from .frontier_certification_pipeline_v1 import (
    VERDICT_CERTIFIABLE,
    VERDICT_NONE,
    W115_FRONTIER_SNAPSHOT,
    FrontierCertificationResultV1,
    FrontierSnapshotV1,
    frontier_date_summary_v1,
    release_version_num_v1,
    run_frontier_certification_v1,
)

W116_UPSTREAM_ADMISSION_V1_SCHEMA_VERSION: str = (
    "coordpy.upstream_instrument_admission_v1.v1")


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


# ------------------------------------------------------ admissibility rule (§3)

# Source-kind taxonomy.  Only the first two are authoritative for A1.
SOURCE_OFFICIAL_HF_DATASET: str = "official_hf_dataset"
SOURCE_OFFICIAL_LCB_GITHUB: str = "official_lcb_github"
SOURCE_AGGREGATOR: str = "aggregator"
SOURCE_WEBSITE_INTRO: str = "website_intro"
SOURCE_RUMOR: str = "rumor"
_AUTHORITATIVE_SOURCES: frozenset[str] = frozenset(
    {SOURCE_OFFICIAL_HF_DATASET, SOURCE_OFFICIAL_LCB_GITHUB})


@dataclasses.dataclass(frozen=True)
class AdmissibilityRuleV1:
    """The PRE-COMMITTED W116 upstream-admissible-instrument rule (RUNBOOK § 3).

    An upstream instrument is ADMISSIBLE iff ALL of A1..A5 hold.  A NEW admissible
    instrument additionally post-dates the currently-admitted instrument.
    """

    min_slice: int = MIN_RESISTANT_SLICE
    a1: str = ("authoritative source (official HF livecodebench/* dataset OR the "
               "official LiveCodeBench GitHub repo; NOT an aggregator / mirror / "
               "leaderboard-site intro / blog rumor / 'planned' announcement)")
    a2: str = "dated problems (each carries a contest_date time-anchor)"
    a3: str = ("functional/code-generation-compatible (has a starter_code "
               "FUNCTIONAL subset the W89 mechanism can attack)")
    a4: str = ("machine-checkable provenance (a SHA-256-pinnable JSONL artifact "
               "that can be operator-fetched + pinned + admitted to "
               "LIVECODEBENCH_KNOWN_RELEASES)")
    a5: str = ("reproducible date histogram (the functional month histogram can "
               "be re-derived from the pinned bytes)")

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_slice": int(self.min_slice),
            "A1_authoritative_source": self.a1,
            "A2_dated_problems": self.a2,
            "A3_functional_compatible": self.a3,
            "A4_sha_pinnable_admittable": self.a4,
            "A5_reproducible_histogram": self.a5,
        }


W116_ADMISSIBILITY_RULE: AdmissibilityRuleV1 = AdmissibilityRuleV1()


@dataclasses.dataclass(frozen=True)
class UpstreamInstrumentObservationV1:
    """One OBSERVED upstream instrument candidate (the external state as DATA).

    Carries the LIVE-verified facts the admissibility rule consumes.  ``release_id``
    maps to an integer version via ``release_version_num_v1`` (so a numbered release
    is comparable to the admitted set).  ``frontier_date`` is the latest functional
    ``contest_date`` (``YYYY-MM-DD``) the instrument exposes, or "" if unknown.
    """

    label: str
    source_kind: str
    source_ref: str
    release_id: str
    has_dated_problems: bool
    has_functional_subset: bool
    sha_pinnable: bool
    histogram_reproducible: bool
    admitted_to_loader: bool
    frontier_date: str
    note: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": str(self.label),
            "source_kind": str(self.source_kind),
            "source_ref": str(self.source_ref),
            "release_id": str(self.release_id),
            "has_dated_problems": bool(self.has_dated_problems),
            "has_functional_subset": bool(self.has_functional_subset),
            "sha_pinnable": bool(self.sha_pinnable),
            "histogram_reproducible": bool(self.histogram_reproducible),
            "admitted_to_loader": bool(self.admitted_to_loader),
            "frontier_date": str(self.frontier_date),
            "note": str(self.note),
        }


@dataclasses.dataclass(frozen=True)
class InstrumentAdmissibilityV1:
    label: str
    a1_authoritative: bool
    a2_dated: bool
    a3_functional: bool
    a4_sha_pinnable: bool
    a5_histogram: bool
    admissible: bool
    newer_than_admitted: bool
    admissible_new_instrument: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": str(self.label),
            "a1_authoritative": bool(self.a1_authoritative),
            "a2_dated": bool(self.a2_dated),
            "a3_functional": bool(self.a3_functional),
            "a4_sha_pinnable": bool(self.a4_sha_pinnable),
            "a5_histogram": bool(self.a5_histogram),
            "admissible": bool(self.admissible),
            "newer_than_admitted": bool(self.newer_than_admitted),
            "admissible_new_instrument": bool(self.admissible_new_instrument),
            "reason": str(self.reason),
        }


def assess_instrument_admissibility_v1(
        obs: UpstreamInstrumentObservationV1,
        *,
        rule: AdmissibilityRuleV1 = W116_ADMISSIBILITY_RULE,
        admitted_latest_num: int,
        admitted_frontier_date: str,
) -> InstrumentAdmissibilityV1:
    """Apply the A1..A5 admissibility rule (RUNBOOK § 3) to one observation.

    ``admissible`` ⟺ A1 ∧ A2 ∧ A3 ∧ A4 ∧ A5.  ``newer_than_admitted`` ⟺ the
    observation's release version exceeds the admitted latest OR its frontier date
    is strictly after the admitted instrument frontier.  ``admissible_new_instrument``
    (the thing that can earn a pilot) ⟺ admissible ∧ newer_than_admitted.
    """
    a1 = obs.source_kind in _AUTHORITATIVE_SOURCES
    a2 = bool(obs.has_dated_problems)
    a3 = bool(obs.has_functional_subset)
    a4 = bool(obs.sha_pinnable)
    a5 = bool(obs.histogram_reproducible)
    admissible = bool(a1 and a2 and a3 and a4 and a5)

    obs_num = release_version_num_v1(obs.release_id)
    newer_by_version = obs_num > int(admitted_latest_num)
    newer_by_date = bool(
        obs.frontier_date and admitted_frontier_date
        and str(obs.frontier_date) > str(admitted_frontier_date))
    newer = bool(newer_by_version or newer_by_date)
    admissible_new = bool(admissible and newer)

    if not admissible:
        failed = [
            name for name, ok in (
                ("A1_authoritative", a1), ("A2_dated", a2),
                ("A3_functional", a3), ("A4_sha_pinnable", a4),
                ("A5_histogram", a5))
            if not ok]
        reason = (
            f"NOT_ADMISSIBLE [{','.join(failed)} fail]: {obs.note}")
    elif not newer:
        reason = (
            f"ADMISSIBLE_BUT_NOT_NEWER: {obs.label} is A1..A5-admissible but does "
            f"not post-date the admitted instrument (version<= {admitted_latest_num} "
            f"and frontier<= {admitted_frontier_date}). {obs.note}")
    else:
        reason = (
            f"ADMISSIBLE_NEW_INSTRUMENT: {obs.label} is A1..A5-admissible AND newer "
            "than the admitted instrument => eligible for slice construction "
            "(RUNBOOK § 5) once operator-fetched + SHA-pinned + admitted.")

    return InstrumentAdmissibilityV1(
        label=obs.label,
        a1_authoritative=a1, a2_dated=a2, a3_functional=a3,
        a4_sha_pinnable=a4, a5_histogram=a5,
        admissible=admissible, newer_than_admitted=newer,
        admissible_new_instrument=admissible_new, reason=reason)


# ---------------------------------------------- per-model disclosure matrix (§4b)

DISCLOSURE_KNOWN: str = "KNOWN"
DISCLOSURE_ESTIMATED_UNUSABLE: str = "ESTIMATED_BUT_UNUSABLE"
DISCLOSURE_UNKNOWN: str = "UNKNOWN"
DISCLOSURE_CONTRADICTORY_STALE: str = "CONTRADICTORY_OR_STALE"


@dataclasses.dataclass(frozen=True)
class DisclosureStatusV1:
    """One model's W116 LIVE primary-source disclosure status (RUNBOOK § 4b).

    Documentation/audit state — does NOT feed the certification CID.  ``status`` is
    one of the four DISCLOSURE_* classes.  ``certifiable_blocker`` names the binding
    gate (C1/C2/C3/C4) that blocks certification on ``release_v6`` even after the
    W116 live re-check.
    """

    model_id: str
    primary_status: str
    primary_source: str
    aggregator_signal: str
    stronger_than_70b: bool
    certifiable_blocker: str
    note: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": str(self.model_id),
            "primary_status": str(self.primary_status),
            "primary_source": str(self.primary_source),
            "aggregator_signal": str(self.aggregator_signal),
            "stronger_than_70b": bool(self.stronger_than_70b),
            "certifiable_blocker": str(self.certifiable_blocker),
            "note": str(self.note),
        }


# The W116 LIVE primary-source disclosure matrix (2026-05-30).  Records the four-way
# classification per RUNBOOK § 4b.  Maverick is KNOWN-but-settled; the three
# reachable stronger-than-Maverick frontier models stay uncertifiable; Mistral
# Small 4 is now CONFIRMED REAL with a primary card carrying NO cutoff (sharper than
# W115's "hypothesized candidate").  Mistral-Small-3.2 is KNOWN-but-sub-70B (C3),
# recorded for completeness.
W116_DISCLOSURE_MATRIX: tuple[DisclosureStatusV1, ...] = (
    DisclosureStatusV1(
        model_id="meta/llama-4-maverick-17b-128e-instruct",
        primary_status=DISCLOSURE_KNOWN,
        primary_source="Official Llama 4 model card (reconfirmed 2026-05-30)",
        aggregator_signal="Aug-2024 (corroborated)",
        stronger_than_70b=True,
        certifiable_blocker="C4 (already SETTLED on release_v6; W113 FAIL)",
        note="KNOWN Aug-2024 but redundant on release_v6 => no new pilot here."),
    DisclosureStatusV1(
        model_id="qwen/qwen3-coder-480b-a35b-instruct",
        primary_status=DISCLOSURE_UNKNOWN,
        primary_source="Official HF model card (Qwen/Qwen3-Coder-480B-A35B-"
                       "Instruct) — reconfirmed 2026-05-30: NO CUTOFF STATED",
        aggregator_signal="(none usable; released 2025-07)",
        stronger_than_70b=True,
        certifiable_blocker="C1 (UNKNOWN cutoff); C2-exposed if estimated ~2025",
        note="Reconfirmed UNKNOWN from the official card."),
    DisclosureStatusV1(
        model_id="deepseek-ai/deepseek-v4-pro",
        primary_status=DISCLOSURE_UNKNOWN,
        primary_source="Official DeepSeek-V4-Pro HF card + V4 model-card PDF — "
                       "reconfirmed 2026-05-30: NO CUTOFF STATED",
        aggregator_signal="(none usable; 1.6T/49B; 2026 release)",
        stronger_than_70b=True,
        certifiable_blocker="C1 (UNKNOWN cutoff); a 2026 release => C2-exposed",
        note="Reconfirmed UNKNOWN; the V4 card exists (W115) but states no cutoff."),
    DisclosureStatusV1(
        model_id="mistralai/mistral-small-4-119b-2603",
        primary_status=DISCLOSURE_UNKNOWN,
        primary_source="Official Mistral docs model card (mistral-small-4-0-26-03) "
                       "+ official announcement (mistral.ai/news/mistral-small-4) "
                       "— 2026-05-30: NO CUTOFF STATED",
        aggregator_signal="OpenRouter '2025-06' (non-primary; also a stale "
                          "'2023-10' system-prompt carryover) => CONTRADICTORY",
        stronger_than_70b=True,
        certifiable_blocker="C1 (UNKNOWN from primary); even the 2025-06 "
                            "aggregator estimate post-dates the Apr-2025 frontier "
                            "=> C2-exposed",
        note="SHARPENED vs W115: CONFIRMED REAL (119B MoE, released 2026-03-16); "
             "primary card discloses NO cutoff; the only cutoff figure is a "
             "non-primary aggregator (2025-06) that is itself C2-exposed."),
    DisclosureStatusV1(
        model_id="mistralai/mistral-small-3.2-24b-instruct-2506",
        primary_status=DISCLOSURE_KNOWN,
        primary_source="HF discussion / aggregator (~Oct-2023)",
        aggregator_signal="2023-10",
        stronger_than_70b=False,
        certifiable_blocker="C3 (24B; NOT strictly stronger than 70B)",
        note="Context only: a KNOWN-cutoff Mistral exists but it is sub-70B (C3 "
             "fails); deprecated 2026-04-30, replaced by Small 4."),
)


def disclosure_matrix_summary_v1(
        matrix: Sequence[DisclosureStatusV1] = W116_DISCLOSURE_MATRIX,
) -> dict[str, Any]:
    """Counts per disclosure class + whether any stronger-than-70B candidate has a
    primary-KNOWN, not-settled cutoff usable for certification (the Lane β verdict)."""
    counts: dict[str, int] = {}
    for d in matrix:
        counts[d.primary_status] = counts.get(d.primary_status, 0) + 1
    # A usable new target would be: stronger-than-70B AND primary-KNOWN AND its
    # blocker is NOT C4 (settled).  Maverick is KNOWN but C4; none other is KNOWN.
    usable = [
        d for d in matrix
        if d.stronger_than_70b and d.primary_status == DISCLOSURE_KNOWN
        and "C4" not in d.certifiable_blocker]
    return {
        "counts": counts,
        "n_models": len(list(matrix)),
        "usable_new_known_cutoff_targets": [d.model_id for d in usable],
        "any_usable_new_known_cutoff_target": bool(usable),
    }


# --------------------------------------------------- certifiable-slice builder (§5)

@dataclasses.dataclass(frozen=True)
class CertifiableSliceCandidateV1:
    model_id: str
    instrument_release: str
    cutoff_boundary: str
    cutoff_confidence: str
    n_functional_resistant: int
    meets_min_slice: bool
    certifiable: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": str(self.model_id),
            "instrument_release": str(self.instrument_release),
            "cutoff_boundary": str(self.cutoff_boundary),
            "cutoff_confidence": str(self.cutoff_confidence),
            "n_functional_resistant": int(self.n_functional_resistant),
            "meets_min_slice": bool(self.meets_min_slice),
            "certifiable": bool(self.certifiable),
            "reason": str(self.reason),
        }


def build_certifiable_slice_candidate_v1(
        candidate: StrongerModelCandidateV1,
        *,
        instrument: LatestResistantInstrumentV1 = LATEST_RESISTANT_INSTRUMENT,
        min_slice: int = MIN_RESISTANT_SLICE,
) -> CertifiableSliceCandidateV1:
    """Build the certifiable resistant-slice candidate for one (model, instrument).

    Reuses the W114 ``certify_model_v1`` gate (NO re-implementation) and surfaces the
    resistant-slice size + whether it clears ``min_slice``.  ``certifiable`` mirrors
    the gate's C1∧C2∧C3∧C4 verdict.
    """
    cert = certify_model_v1(candidate, instrument=instrument)
    return CertifiableSliceCandidateV1(
        model_id=candidate.model_id,
        instrument_release=instrument.release,
        cutoff_boundary=cert.cutoff_boundary,
        cutoff_confidence=cert.cutoff_confidence,
        n_functional_resistant=cert.n_functional_resistant,
        meets_min_slice=bool(cert.n_functional_resistant >= int(min_slice)),
        certifiable=bool(cert.certifiable_for_new_pilot),
        reason=cert.reason)


# ------------------------------------------------------- upstream supply snapshot

@dataclasses.dataclass(frozen=True)
class UpstreamSupplySnapshotV1:
    """The LIVE multi-surface upstream supply state as DATA (the W117 re-run input).

    Generalises the W115 ``FrontierSnapshotV1`` (a single observed-releases list)
    into the four-surface upstream view W116 verified.  ``frontier_snapshot`` carries
    the model/instrument/disclosure state for the reused W115 certification matrix.
    """

    verified_on: str
    source_note: str
    hf_dataset: str
    lite_tree_releases: tuple[str, ...]
    loader_allowed_versions: tuple[str, ...]
    release_latest_resolves_to: str
    full_dataset_frontier_date: str
    lite_last_modified: str
    observations: tuple[UpstreamInstrumentObservationV1, ...]
    frontier_snapshot: FrontierSnapshotV1

    def to_dict(self) -> dict[str, Any]:
        return {
            "verified_on": str(self.verified_on),
            "source_note": str(self.source_note),
            "hf_dataset": str(self.hf_dataset),
            "lite_tree_releases": list(self.lite_tree_releases),
            "loader_allowed_versions": list(self.loader_allowed_versions),
            "release_latest_resolves_to": str(self.release_latest_resolves_to),
            "full_dataset_frontier_date": str(self.full_dataset_frontier_date),
            "lite_last_modified": str(self.lite_last_modified),
            "observations": [o.to_dict() for o in self.observations],
            "frontier_snapshot": self.frontier_snapshot.to_dict(),
        }


# The LOCKED W116 upstream snapshot — the LIVE primary-source re-verification of
# 2026-05-30 (RUNBOOK_W116 § 2).  Four upstream surfaces, all pointing at release_v6;
# the model/instrument/disclosure state reuses the W115 snapshot (unchanged => the
# certification decision re-derives byte-identically, CID 258b6ed7).
W116_UPSTREAM_SNAPSHOT: UpstreamSupplySnapshotV1 = UpstreamSupplySnapshotV1(
    verified_on="2026-05-30",
    source_note=(
        "W116 live primary-source upstream-supply attack (RUNBOOK_W116 § 2). FOUR "
        "authoritative surfaces: (1) HF code_generation_lite file tree highest = "
        "test6.jsonl, no test7+, lastModified 2025-06-05; (2) loader "
        "code_generation_lite.py ALLOWED_FILES v_list=[v1..v6], release_latest -> "
        "release_v6 files; (3) full code_generation dataset README release_v6 = May "
        "2023..Apr 2025 (1055 problems, frontier 2025-04-05); (4) GitHub repo README "
        "tops out at release_v6, no v7 tag. The 'planned v7' search hint is "
        "non-primary and INADMISSIBLE. Model side: Mistral-Small-4-119B-2603 "
        "CONFIRMED REAL (official card NO cutoff)."),
    hf_dataset=LIVECODEBENCH_HF_DATASET,
    lite_tree_releases=(
        "release_v1", "release_v2", "release_v3",
        "release_v4", "release_v5", "release_v6"),
    loader_allowed_versions=(
        "release_v1", "release_v2", "release_v3", "release_v4",
        "release_v5", "release_v6", "release_latest"),
    release_latest_resolves_to="release_v6",
    full_dataset_frontier_date="2025-04-05",
    lite_last_modified="2025-06-05",
    observations=(
        UpstreamInstrumentObservationV1(
            label="release_v6 (admitted instrument)",
            source_kind=SOURCE_OFFICIAL_HF_DATASET,
            source_ref="hf://livecodebench/code_generation_lite test6.jsonl "
                       "(SHA bb4c364f...)",
            release_id="release_v6",
            has_dated_problems=True, has_functional_subset=True,
            sha_pinnable=True, histogram_reproducible=True,
            admitted_to_loader=True,
            frontier_date="2025-04-05",
            note="The current admitted instrument: admissible but NOT new "
                 "(functional frontier 2025-04-05, already the certification base)."),
        UpstreamInstrumentObservationV1(
            label="full code_generation dataset",
            source_kind=SOURCE_OFFICIAL_HF_DATASET,
            source_ref="hf://livecodebench/code_generation test.jsonl (9.4GB)",
            release_id="release_v6",
            has_dated_problems=True, has_functional_subset=True,
            sha_pinnable=True, histogram_reproducible=True,
            admitted_to_loader=False,
            frontier_date="2025-04-05",
            note="The full (non-lite) dataset: authoritative + dated + functional, "
                 "but its README frontier is the SAME Apr-2025 => admissible-as-"
                 "source but NOT newer than the admitted lite v6."),
        UpstreamInstrumentObservationV1(
            label="release_latest alias",
            source_kind=SOURCE_OFFICIAL_HF_DATASET,
            source_ref="code_generation_lite.py ALLOWED_FILES['release_latest']",
            release_id="release_v6",
            has_dated_problems=True, has_functional_subset=True,
            sha_pinnable=True, histogram_reproducible=True,
            admitted_to_loader=True,
            frontier_date="2025-04-05",
            note="The upstream loader's 'latest' alias resolves to the SAME six "
                 "files as release_v6 => no hidden newer-than-v6 supply."),
        UpstreamInstrumentObservationV1(
            label="planned release_v7 (search rumor)",
            source_kind=SOURCE_RUMOR,
            source_ref="non-primary WebSearch summary ('planned v7, late-2025/"
                       "early-2026')",
            release_id="release_v7",
            has_dated_problems=False, has_functional_subset=False,
            sha_pinnable=False, histogram_reproducible=False,
            admitted_to_loader=False,
            frontier_date="",
            note="A 'planned v7' mentioned only by a non-primary summary; "
                 "contradicted by all four authoritative surfaces; no artifact, no "
                 "SHA => REFUSED by A1 + A4. Recorded as the W117 watch signal."),
    ),
    frontier_snapshot=W115_FRONTIER_SNAPSHOT)


# --------------------------------------------------------- upstream-change detector

@dataclasses.dataclass(frozen=True)
class UpstreamChangeV1:
    """What changed across upstream surfaces vs the encoded baseline (W117 signal)."""

    new_numbered_release: bool
    release_latest_repointed: bool
    full_dataset_frontier_advanced: bool
    lite_last_modified_changed: bool
    loader_versions_extended: bool
    new_admissible_instrument: bool
    any_change: bool
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "new_numbered_release": bool(self.new_numbered_release),
            "release_latest_repointed": bool(self.release_latest_repointed),
            "full_dataset_frontier_advanced": bool(
                self.full_dataset_frontier_advanced),
            "lite_last_modified_changed": bool(self.lite_last_modified_changed),
            "loader_versions_extended": bool(self.loader_versions_extended),
            "new_admissible_instrument": bool(self.new_admissible_instrument),
            "any_change": bool(self.any_change),
            "notes": list(self.notes),
        }


def _max_release_num(releases: Sequence[str]) -> int:
    nums = [release_version_num_v1(r) for r in releases]
    return max(nums) if nums else 0


def detect_upstream_change_v1(
        new_snapshot: UpstreamSupplySnapshotV1,
        *,
        baseline: UpstreamSupplySnapshotV1 = W116_UPSTREAM_SNAPSHOT,
) -> UpstreamChangeV1:
    """Compare a new upstream snapshot vs the encoded baseline; flag WHAT changed.

    This is the W117 update signal — richer than W115's single boolean.  A change in
    ANY surface (a newer numbered release on the lite tree, ``release_latest`` now
    resolving past the baseline, the full-dataset frontier advancing, the lite
    lastModified bumping, the loader version list extending, or a new
    admissible-and-newer observation) flips ``any_change`` and the relevant field.
    """
    notes: list[str] = []

    new_num = _max_release_num(new_snapshot.lite_tree_releases)
    base_num = _max_release_num(baseline.lite_tree_releases)
    new_numbered = new_num > base_num
    if new_numbered:
        notes.append(
            f"lite tree max release v{base_num} -> v{new_num}")

    repointed = bool(
        release_version_num_v1(new_snapshot.release_latest_resolves_to)
        > release_version_num_v1(baseline.release_latest_resolves_to))
    if repointed:
        notes.append(
            f"release_latest re-pointed {baseline.release_latest_resolves_to} -> "
            f"{new_snapshot.release_latest_resolves_to}")

    frontier_adv = bool(
        new_snapshot.full_dataset_frontier_date
        and baseline.full_dataset_frontier_date
        and str(new_snapshot.full_dataset_frontier_date)
        > str(baseline.full_dataset_frontier_date))
    if frontier_adv:
        notes.append(
            f"full-dataset frontier {baseline.full_dataset_frontier_date} -> "
            f"{new_snapshot.full_dataset_frontier_date}")

    lm_changed = bool(
        str(new_snapshot.lite_last_modified)
        != str(baseline.lite_last_modified))
    if lm_changed:
        notes.append(
            f"lite lastModified {baseline.lite_last_modified} -> "
            f"{new_snapshot.lite_last_modified}")

    base_versions = set(baseline.loader_allowed_versions)
    extended = bool(set(new_snapshot.loader_allowed_versions) - base_versions)
    if extended:
        notes.append(
            "loader ALLOWED_FILES extended: "
            f"{sorted(set(new_snapshot.loader_allowed_versions) - base_versions)}")

    # A new admissible-and-newer observation vs the admitted base.
    adm_latest_num = base_num
    adm_frontier = baseline.full_dataset_frontier_date
    new_admissible = any(
        assess_instrument_admissibility_v1(
            o, admitted_latest_num=adm_latest_num,
            admitted_frontier_date=adm_frontier).admissible_new_instrument
        for o in new_snapshot.observations)
    if new_admissible:
        notes.append("a new admissible-and-newer instrument observation appeared")

    any_change = bool(
        new_numbered or repointed or frontier_adv or lm_changed
        or extended or new_admissible)
    if not any_change:
        notes.append("no upstream change vs baseline (frontier unchanged)")

    return UpstreamChangeV1(
        new_numbered_release=new_numbered,
        release_latest_repointed=repointed,
        full_dataset_frontier_advanced=frontier_adv,
        lite_last_modified_changed=lm_changed,
        loader_versions_extended=extended,
        new_admissible_instrument=new_admissible,
        any_change=any_change,
        notes=tuple(notes))


# ------------------------------------------------------------ W117 fire condition

@dataclasses.dataclass(frozen=True)
class W117FireConditionV1:
    """The pre-committed triggers that flip the no-go verdict (RUNBOOK § 9)."""

    instrument_trigger: str
    cutoff_trigger: str
    instrument_trigger_met: bool
    cutoff_trigger_met: bool
    fires_now: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "instrument_trigger": str(self.instrument_trigger),
            "cutoff_trigger": str(self.cutoff_trigger),
            "instrument_trigger_met": bool(self.instrument_trigger_met),
            "cutoff_trigger_met": bool(self.cutoff_trigger_met),
            "fires_now": bool(self.fires_now),
        }


# ------------------------------------------------------------- the push-button API

@dataclasses.dataclass(frozen=True)
class UpstreamAdmissionResultV1:
    schema: str
    verified_on: str
    admissibility_rule: dict[str, Any]
    instrument_admissibility: tuple[InstrumentAdmissibilityV1, ...]
    n_admissible_new_instruments: int
    upstream_change: UpstreamChangeV1
    disclosure_matrix: tuple[DisclosureStatusV1, ...]
    disclosure_summary: dict[str, Any]
    frontier_certification: FrontierCertificationResultV1
    verdict: str
    target_model: str
    w117_fire_condition: W117FireConditionV1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "verified_on": str(self.verified_on),
            "admissibility_rule": self.admissibility_rule,
            "instrument_admissibility": [
                a.to_dict() for a in self.instrument_admissibility],
            "n_admissible_new_instruments": int(
                self.n_admissible_new_instruments),
            "upstream_change": self.upstream_change.to_dict(),
            "disclosure_matrix": [d.to_dict() for d in self.disclosure_matrix],
            "disclosure_summary": self.disclosure_summary,
            "frontier_certification": self.frontier_certification.to_dict(),
            "verdict": str(self.verdict),
            "target_model": str(self.target_model),
            "w117_fire_condition": self.w117_fire_condition.to_dict(),
        }

    def cid(self) -> str:
        return _sha256_hex({"kind": "w116_upstream_admission_result_v1",
                            "result": self.to_dict()})


def run_upstream_admission_v1(
        snapshot: UpstreamSupplySnapshotV1 = W116_UPSTREAM_SNAPSHOT,
        *,
        rule: AdmissibilityRuleV1 = W116_ADMISSIBILITY_RULE,
        min_slice: int = MIN_RESISTANT_SLICE,
) -> UpstreamAdmissionResultV1:
    """The push-button W116 upstream-admission + certification matrix (RUNBOOK § 4).

    Reuses the W115 ``run_frontier_certification_v1`` for the per-model gate + verdict
    (which reuses the W114 ``decide_certification_v1`` — no re-implementation; the
    decision CID re-derives byte-identically) and wraps it with the admissibility
    rule, the multi-surface upstream-change detector, the four-way disclosure matrix,
    and the structured W117 fire condition.  Re-running this on an updated snapshot is
    the W117 operation.
    """
    frontier = run_frontier_certification_v1(
        snapshot.frontier_snapshot, min_slice=min_slice)

    # The admitted BASELINE is the loader's admitted release set + the admitted
    # instrument's frontier (NOT the observed lite tree, which may carry a
    # not-yet-admitted release): a NEW admissible instrument is one newer than what
    # the loader currently admits.
    adm_latest_num = _max_release_num(LIVECODEBENCH_KNOWN_RELEASES)
    adm_frontier = snapshot.frontier_snapshot.instrument.functional_date_max
    admissibility = tuple(
        assess_instrument_admissibility_v1(
            o, rule=rule, admitted_latest_num=adm_latest_num,
            admitted_frontier_date=adm_frontier)
        for o in snapshot.observations)
    n_admissible_new = sum(
        1 for a in admissibility if a.admissible_new_instrument)

    change = detect_upstream_change_v1(snapshot, baseline=snapshot)
    disclosure_summary = disclosure_matrix_summary_v1(W116_DISCLOSURE_MATRIX)

    certifiable = (frontier.verdict == VERDICT_CERTIFIABLE)
    # Instrument trigger: a newer ADMISSIBLE instrument is available to admit
    # (either flagged by the supply snapshot OR by the reused release detector).
    instrument_met = bool(
        n_admissible_new >= 1
        or frontier.release_detection.newer_release_available)
    # Cutoff trigger: a reachable stronger-than-Maverick model now certifies (some
    # non-Maverick candidate clears C1∧C2∧C3∧C4 — exactly the certifiable case).
    cutoff_met = bool(
        certifiable and frontier.target_model
        and "maverick" not in frontier.target_model.lower())
    fires_now = bool(certifiable)

    max_month = frontier.frontier_summary.max_cutoff_month_for_min_slice or "n/a"
    fire = W117FireConditionV1(
        instrument_trigger=(
            "An admissible NEW upstream instrument (a release_v7+ on the lite tree, "
            "OR release_latest re-pointing past release_v6, OR a distinct upstream "
            "functional dataset with post-2025-04 dated problems) is observed, "
            "operator-fetched + SHA-pinned + admitted to LIVECODEBENCH_KNOWN_"
            f"RELEASES, with >= {min_slice} functional problems dated strictly after "
            "a reachable stronger-than-Maverick model's primary-KNOWN cutoff "
            "(RUNBOOK § 3 + § 5)."),
        cutoff_trigger=(
            "A reachable stronger-than-Maverick model discloses, from a PRIMARY "
            f"source, a KNOWN cutoff month <= {max_month} (so the current "
            f"{frontier.frontier_summary.release} admits a >= {min_slice} resistant "
            "slice for it); update MODEL_TRAINING_CUTOFFS + W114_CUTOFF_PROVENANCE, "
            "then re-run."),
        instrument_trigger_met=instrument_met,
        cutoff_trigger_met=cutoff_met,
        fires_now=fires_now)

    return UpstreamAdmissionResultV1(
        schema=W116_UPSTREAM_ADMISSION_V1_SCHEMA_VERSION,
        verified_on=snapshot.verified_on,
        admissibility_rule=rule.to_dict(),
        instrument_admissibility=admissibility,
        n_admissible_new_instruments=int(n_admissible_new),
        upstream_change=change,
        disclosure_matrix=W116_DISCLOSURE_MATRIX,
        disclosure_summary=disclosure_summary,
        frontier_certification=frontier,
        verdict=frontier.verdict,
        target_model=frontier.target_model,
        w117_fire_condition=fire)


__all__ = [
    "W116_UPSTREAM_ADMISSION_V1_SCHEMA_VERSION",
    "SOURCE_OFFICIAL_HF_DATASET",
    "SOURCE_OFFICIAL_LCB_GITHUB",
    "SOURCE_AGGREGATOR",
    "SOURCE_WEBSITE_INTRO",
    "SOURCE_RUMOR",
    "AdmissibilityRuleV1",
    "W116_ADMISSIBILITY_RULE",
    "UpstreamInstrumentObservationV1",
    "InstrumentAdmissibilityV1",
    "assess_instrument_admissibility_v1",
    "DISCLOSURE_KNOWN",
    "DISCLOSURE_ESTIMATED_UNUSABLE",
    "DISCLOSURE_UNKNOWN",
    "DISCLOSURE_CONTRADICTORY_STALE",
    "DisclosureStatusV1",
    "W116_DISCLOSURE_MATRIX",
    "disclosure_matrix_summary_v1",
    "CertifiableSliceCandidateV1",
    "build_certifiable_slice_candidate_v1",
    "UpstreamSupplySnapshotV1",
    "W116_UPSTREAM_SNAPSHOT",
    "UpstreamChangeV1",
    "detect_upstream_change_v1",
    "W117FireConditionV1",
    "UpstreamAdmissionResultV1",
    "run_upstream_admission_v1",
    "VERDICT_CERTIFIABLE",
    "VERDICT_NONE",
]
