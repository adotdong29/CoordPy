"""W115 / COO-9 — durable future-fire certification / instrument-supply pipeline.

W114 verified, from primary sources (2026-05-29), that NO reachable model stronger
than Llama-4-Maverick is certifiably contamination-resistant on the latest real
instrument (LCB ``release_v6``; functional frontier 2025-04-05): the
resistant-instrument frontier has AGED OUT relative to the reachable model
frontier, whose cutoffs are officially undisclosed.  W114's
``stronger_model_cutoff_certification_v1`` answered that question for ONE hard-coded
instrument + ONE verification pass.

This module GENERALISES that one-shot certification into a durable, push-button
**supply-chain pipeline** so the next clean shot (W116) is mechanical the moment
the external world changes.  The blocker is supply (instrument + cutoff
disclosure), so the pipeline attacks the supply side directly:

1. ``detect_latest_release_v1`` — the **latest-official-release detector**: compares
   the releases OBSERVED on the live source (recorded in the snapshot, from the HF
   file tree) against the releases the loader is ADMITTED to operate on
   (``LIVECODEBENCH_KNOWN_RELEASES``), and flags ``newer_release_available`` (an
   observed release the loader has not yet SHA-pinned/admitted).  This turns "is
   there a release_v7+?" from a narrative re-check into an operational boolean.
2. ``frontier_date_summary_v1`` — the **frontier-date histogram / summary
   artifact**, generalised over ANY instrument: the month histogram + frontier date
   + the threshold table (``max KNOWN cutoff month that still admits >= N resistant
   problems``).  This makes "which cutoff months admit a >= 30 slice" explicit and
   reusable, not buried in one constant.
3. ``certifiable_slice_candidates_v1`` — the **certifiable-slice candidate builder**:
   per-(model, instrument) certifiable slice size + the binding blocker, reusing
   the W114 ``certify_model_v1`` gate (no re-implementation).
4. ``run_frontier_certification_v1`` — the **per-model go/no-go matrix with exact
   blocker reasons**, driven by a ``FrontierSnapshotV1`` (the external state as
   DATA: observed releases + the latest admitted instrument + the live per-model
   cutoff disclosures).  Reuses the W114 ``decide_certification_v1`` for the verdict
   and adds the release detector, the frontier summary, a disclosure-consistency
   guard (the snapshot's live disclosures vs the encoded W113/W114 registry — a
   divergence is the W116 update signal), and the structured W116 fire condition.

The push-button W116 flow: the operator updates ONE ``FrontierSnapshotV1`` (a newer
observed release, or a newly-disclosed KNOWN cutoff) and re-runs
``run_frontier_certification_v1``.  If a newer release is observed, the pipeline
flags it for operator-fetch + loader admission (it never fabricates a release); if
a stronger model now has a KNOWN cutoff <= the instrument frontier, the matrix
certifies it and names the pilot target.  Admission stays disciplined: a release
is only certifiable after it is SHA-pinned + admitted to the loader (the snapshot
detects the change; the operator admits it).

Pure / deterministic / NIM-free / read-only.  Explicit-import-only; imports the
W113 registry + the W114 gate/instrument + the loader's release set, with NO
duplication of the certification logic.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from typing import Any, Sequence

from .livecodebench_loader_v1 import LIVECODEBENCH_KNOWN_RELEASES
from .livecodebench_resistant_slice_v1 import (
    CONFIDENCE_KNOWN,
    MIN_RESISTANT_SLICE,
    MODEL_TRAINING_CUTOFFS,
)
from .stronger_model_cutoff_certification_v1 import (
    LATEST_RESISTANT_INSTRUMENT,
    STRONGER_MODEL_CANDIDATES,
    VERDICT_CERTIFIABLE,
    VERDICT_NONE,
    W114_CUTOFF_PROVENANCE,
    CertificationDecisionV1,
    LatestResistantInstrumentV1,
    StrongerModelCandidateV1,
    decide_certification_v1,
)

W115_FRONTIER_PIPELINE_V1_SCHEMA_VERSION: str = (
    "coordpy.frontier_certification_pipeline_v1.v1")


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


# A LiveCodeBench release version is identified by a trailing integer.  The HF
# file tree names the FIRST release ``test.jsonl`` (v1) and subsequent ones
# ``testN.jsonl`` (vN); the loader names them ``release_vN``.  This parser accepts
# every form so an observed file-tree listing and the admitted release set are
# comparable on the same integer axis.
_RELEASE_NUM_RE = re.compile(r"(?:release_v|test)(\d+)")


def release_version_num_v1(release_or_file: str) -> int:
    """Map a release id / file name to its integer version (>= 1).

    ``release_v6`` -> 6; ``test6.jsonl`` -> 6; ``test.jsonl`` -> 1 (the first
    release carries no number); a bare ``release_v1`` -> 1.  Returns 0 for an
    unrecognised string (so it never compares as "newer").
    """
    s = str(release_or_file or "").strip().lower()
    if not s:
        return 0
    m = _RELEASE_NUM_RE.search(s)
    if m:
        return int(m.group(1))
    # ``test.jsonl`` (no digit) is the v1 artifact.
    if s in ("test.jsonl", "release_v", "release", "test"):
        return 1
    return 0


# ----------------------------------------------------------- release detection

@dataclasses.dataclass(frozen=True)
class ReleaseDetectionV1:
    """Whether the live source shows a release the loader has not yet admitted."""

    admitted_releases: tuple[str, ...]
    observed_releases: tuple[str, ...]
    latest_admitted: str
    latest_admitted_num: int
    latest_observed: str
    latest_observed_num: int
    newer_release_available: bool
    observed_not_admitted: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "admitted_releases": list(self.admitted_releases),
            "observed_releases": list(self.observed_releases),
            "latest_admitted": str(self.latest_admitted),
            "latest_admitted_num": int(self.latest_admitted_num),
            "latest_observed": str(self.latest_observed),
            "latest_observed_num": int(self.latest_observed_num),
            "newer_release_available": bool(self.newer_release_available),
            "observed_not_admitted": list(self.observed_not_admitted),
        }


def detect_latest_release_v1(
        observed_releases: Sequence[str],
        *,
        admitted_releases: Sequence[str] = LIVECODEBENCH_KNOWN_RELEASES,
) -> ReleaseDetectionV1:
    """Compare observed (live file-tree) releases vs the loader-admitted set.

    ``newer_release_available`` ⟺ the highest OBSERVED version exceeds the highest
    ADMITTED version (i.e. an operator-fetch + loader-admission + SHA-pin is
    required before the new release can be certified).  ``observed_not_admitted``
    lists every observed release whose integer version is not in the admitted set.
    """
    adm = tuple(str(r) for r in admitted_releases)
    obs = tuple(str(r) for r in observed_releases)
    adm_nums = {release_version_num_v1(r) for r in adm}
    latest_adm_num = max(adm_nums) if adm_nums else 0
    latest_adm = next(
        (r for r in adm if release_version_num_v1(r) == latest_adm_num), "")
    obs_nums = {release_version_num_v1(r) for r in obs}
    latest_obs_num = max(obs_nums) if obs_nums else 0
    latest_obs = next(
        (r for r in obs if release_version_num_v1(r) == latest_obs_num), "")
    not_admitted = tuple(
        r for r in obs if release_version_num_v1(r) not in adm_nums)
    return ReleaseDetectionV1(
        admitted_releases=adm,
        observed_releases=obs,
        latest_admitted=latest_adm,
        latest_admitted_num=latest_adm_num,
        latest_observed=latest_obs,
        latest_observed_num=latest_obs_num,
        newer_release_available=bool(latest_obs_num > latest_adm_num),
        observed_not_admitted=not_admitted)


# -------------------------------------------------------- frontier-date summary

@dataclasses.dataclass(frozen=True)
class FrontierDateSummaryV1:
    """Generalised instrument-frontier summary for ANY instrument.

    ``threshold_table`` maps a candidate cutoff month ``"YYYY-MM"`` to the count of
    functional problems strictly after it (i.e. resistant for a cutoff in that
    month).  ``max_cutoff_month_for_min_slice`` is the LATEST cutoff month that
    still admits ``>= min_slice`` resistant problems (a later cutoff ⇒ fewer
    resistant) — the binding "the model's KNOWN cutoff must be <= this month"
    threshold.  Empty string if no cutoff month admits a >= min_slice slice.
    """

    release: str
    n_functional: int
    functional_date_min: str
    functional_date_max: str
    month_histogram: dict[str, int]
    min_slice: int
    threshold_table: dict[str, int]
    max_cutoff_month_for_min_slice: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "release": str(self.release),
            "n_functional": int(self.n_functional),
            "functional_date_min": str(self.functional_date_min),
            "functional_date_max": str(self.functional_date_max),
            "month_histogram": dict(self.month_histogram),
            "min_slice": int(self.min_slice),
            "threshold_table": dict(self.threshold_table),
            "max_cutoff_month_for_min_slice": str(
                self.max_cutoff_month_for_min_slice),
        }


def frontier_date_summary_v1(
        instrument: LatestResistantInstrumentV1 = LATEST_RESISTANT_INSTRUMENT,
        *,
        min_slice: int = MIN_RESISTANT_SLICE,
) -> FrontierDateSummaryV1:
    """Build the frontier-date summary + threshold table for an instrument.

    The threshold table evaluates, for each distinct month in the histogram AND
    the month immediately before the earliest (so the "all problems resistant"
    case is captured), how many functional problems are strictly after a cutoff in
    that month.  ``max_cutoff_month_for_min_slice`` = the latest such month with
    ``>= min_slice`` resistant problems.
    """
    hist = dict(instrument.functional_month_histogram)
    months = sorted(hist.keys())
    # Candidate cutoff months: each present month, plus "the month before the
    # first" (encoded as the first month's year-1/12 is overkill; instead we add
    # a synthetic "earlier than all" sentinel by using the boundary count).
    table: dict[str, int] = {}
    for ym in months:
        # a cutoff IN month ``ym`` keeps months strictly after ``ym``.
        boundary = f"{ym}-28"  # any day in the month; comparison is month-granular
        table[ym] = instrument.n_functional_resistant_after(boundary)
    # The "cutoff before all problems" case (everything resistant):
    if months:
        table["<before-all>"] = instrument.n_functional
    # Latest cutoff month admitting >= min_slice (scan present months descending).
    max_month = ""
    for ym in sorted(months):
        if instrument.n_functional_resistant_after(f"{ym}-28") >= min_slice:
            max_month = ym  # keep the latest qualifying month
    return FrontierDateSummaryV1(
        release=instrument.release,
        n_functional=instrument.n_functional,
        functional_date_min=instrument.functional_date_min,
        functional_date_max=instrument.functional_date_max,
        month_histogram=hist,
        min_slice=int(min_slice),
        threshold_table=table,
        max_cutoff_month_for_min_slice=max_month)


# -------------------------------------------------- live per-model disclosures

@dataclasses.dataclass(frozen=True)
class ModelDisclosureV1:
    """One model's LIVE-verified cutoff disclosure (the external state as data).

    This is what the operator re-verifies each milestone.  ``confidence`` /
    ``boundary_date`` are checked against the encoded W113 registry + W114
    provenance; a divergence is the W116 update signal (the registry must be
    updated before a newly-disclosed cutoff can certify).
    """

    model_id: str
    confidence: str
    boundary_date: str
    primary_source: str
    verified_on: str
    note: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": str(self.model_id),
            "confidence": str(self.confidence),
            "boundary_date": str(self.boundary_date),
            "primary_source": str(self.primary_source),
            "verified_on": str(self.verified_on),
            "note": str(self.note),
        }


@dataclasses.dataclass(frozen=True)
class DisclosureConsistencyV1:
    model_id: str
    snapshot_confidence: str
    registry_confidence: str
    provenance_confidence: str
    consistent: bool
    note: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": str(self.model_id),
            "snapshot_confidence": str(self.snapshot_confidence),
            "registry_confidence": str(self.registry_confidence),
            "provenance_confidence": str(self.provenance_confidence),
            "consistent": bool(self.consistent),
            "note": str(self.note),
        }


# ------------------------------------------------------------ frontier snapshot

@dataclasses.dataclass(frozen=True)
class FrontierSnapshotV1:
    """The LIVE external state as data — the single thing W116 updates + re-runs.

    Carries: the releases observed on the official source (HF file tree); the
    latest ADMITTED resistant FUNCTIONAL instrument (the loader-pinned one the
    matrix certifies against); the reachable stronger-than-70B candidate set; and
    the per-model live cutoff disclosures.  Re-running
    ``run_frontier_certification_v1`` on an updated snapshot is the push-button
    W116 operation.
    """

    verified_on: str
    source_note: str
    observed_releases: tuple[str, ...]
    instrument: LatestResistantInstrumentV1
    candidates: tuple[StrongerModelCandidateV1, ...]
    model_disclosures: tuple[ModelDisclosureV1, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "verified_on": str(self.verified_on),
            "source_note": str(self.source_note),
            "observed_releases": list(self.observed_releases),
            "instrument": self.instrument.to_dict(),
            "candidates": [c.model_id for c in self.candidates],
            "model_disclosures": [d.to_dict() for d in self.model_disclosures],
        }


# The LOCKED W115 frontier snapshot — the LIVE primary-source re-verification of
# 2026-05-29 (RUNBOOK_W115 § 2).  Observed releases mirror the live HF file tree
# (test.jsonl..test6.jsonl => release_v1..v6; no test7+).  Disclosures mirror the
# encoded W113/W114 registry (all consistent — the external frontier did not move).
W115_FRONTIER_SNAPSHOT: FrontierSnapshotV1 = FrontierSnapshotV1(
    verified_on="2026-05-29",
    source_note=(
        "W115 live primary-source re-verification (RUNBOOK_W115 § 2). LCB HF file "
        "tree: highest test file = test6.jsonl (134 MB; 'add v6' ~1yr; no test7+) "
        "=> release_v6 still latest, frontier 2025-04-05 UNCHANGED. Cutoffs re-"
        "checked against official cards/PDFs; the one external change since W114 "
        "(DeepSeek V4 card published 2026-04-27) discloses NO cutoff => verdict "
        "unchanged."),
    observed_releases=(
        "release_v1", "release_v2", "release_v3",
        "release_v4", "release_v5", "release_v6"),
    instrument=LATEST_RESISTANT_INSTRUMENT,
    candidates=STRONGER_MODEL_CANDIDATES,
    model_disclosures=(
        ModelDisclosureV1(
            model_id="meta/llama-4-maverick-17b-128e-instruct",
            confidence=CONFIDENCE_KNOWN, boundary_date="2024-08-31",
            primary_source="Official Llama 4 model card (model-cards docs; "
                           "corroborated multi-source)",
            verified_on="2026-05-29",
            note="Pretraining cutoff 'August 2024' KNOWN (reconfirmed). Settled "
                 "on release_v6 (W113 resistant FAIL) => C4."),
        ModelDisclosureV1(
            model_id="qwen/qwen3-coder-480b-a35b-instruct",
            confidence="UNKNOWN", boundary_date="2025-07-01",
            primary_source="Official HF model card "
                           "(Qwen/Qwen3-Coder-480B-A35B-Instruct) — fetched live",
            verified_on="2026-05-29",
            note="Card states NO knowledge/training cutoff (reconfirmed live). "
                 "Released 2025-07-22; estimable cutoff ~2025 >= Apr-2025 frontier "
                 "=> C2-exposed even if disclosed."),
        ModelDisclosureV1(
            model_id="deepseek-ai/deepseek-v4-pro",
            confidence="UNKNOWN", boundary_date="2025-01-01",
            primary_source="Official DeepSeek V4 model card PDF "
                           "(fe-static.deepseek.com; published 2026-04-27; "
                           "Pro=1.6T/49B) — fetched + text-extracted live",
            verified_on="2026-05-29",
            note="SHARPENED vs W114: the official V4 card now EXISTS (2026-04-27) "
                 "but contains NO 'cutoff' string and no training-data date => "
                 "UNKNOWN. A 2026-04 release => real cutoff >= 2025 => C2-exposed."),
        ModelDisclosureV1(
            model_id="mistralai/mistral-small-4-119b-2603",
            confidence="UNKNOWN", boundary_date="2026-03-01",
            primary_source="Official Mistral docs/HF (real line = "
                           "Mistral-Small-3.2-2506) — searched live",
            verified_on="2026-05-29",
            note="No cutoff stated for the candidate; the real reachable Mistral "
                 "line (Small 3.2, 2025-06) is weaker than Maverick. 2026-03 tag "
                 "post-dates the whole release_v6 window => C2-exposed regardless."),
    ))


def check_disclosure_consistency_v1(
        snapshot: FrontierSnapshotV1,
) -> tuple[DisclosureConsistencyV1, ...]:
    """Each live disclosure's confidence vs the encoded W113 registry + W114
    provenance.  A divergence means the live world moved relative to the encoded
    state — the operator must update the registry before the matrix can act on it
    (the W116 update signal)."""
    out: list[DisclosureConsistencyV1] = []
    for d in snapshot.model_disclosures:
        reg = MODEL_TRAINING_CUTOFFS.get(d.model_id)
        prov = W114_CUTOFF_PROVENANCE.get(d.model_id)
        reg_conf = reg.confidence if reg else "(unregistered)"
        prov_conf = prov.verified_confidence if prov else "(no provenance)"
        consistent = (
            reg is not None
            and str(d.confidence) == str(reg_conf)
            and str(d.confidence) == str(prov_conf))
        note = (
            "live disclosure matches the encoded W113 registry + W114 provenance"
            if consistent else
            "DIVERGENCE: live disclosure != encoded registry/provenance => update "
            "MODEL_TRAINING_CUTOFFS + W114_CUTOFF_PROVENANCE before certifying")
        out.append(DisclosureConsistencyV1(
            model_id=d.model_id, snapshot_confidence=d.confidence,
            registry_confidence=reg_conf, provenance_confidence=prov_conf,
            consistent=consistent, note=note))
    return tuple(out)


# ------------------------------------------------- the push-button go/no-go API

@dataclasses.dataclass(frozen=True)
class W116FireConditionV1:
    """The two pre-committed triggers that flip the no-go verdict (RUNBOOK § 9)."""

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


@dataclasses.dataclass(frozen=True)
class FrontierCertificationResultV1:
    schema: str
    verified_on: str
    release_detection: ReleaseDetectionV1
    frontier_summary: FrontierDateSummaryV1
    decision: CertificationDecisionV1
    disclosure_consistency: tuple[DisclosureConsistencyV1, ...]
    disclosure_consistency_ok: bool
    verdict: str
    target_model: str
    w116_fire_condition: W116FireConditionV1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "verified_on": str(self.verified_on),
            "release_detection": self.release_detection.to_dict(),
            "frontier_summary": self.frontier_summary.to_dict(),
            "decision": self.decision.to_dict(),
            "disclosure_consistency": [
                d.to_dict() for d in self.disclosure_consistency],
            "disclosure_consistency_ok": bool(self.disclosure_consistency_ok),
            "verdict": str(self.verdict),
            "target_model": str(self.target_model),
            "w116_fire_condition": self.w116_fire_condition.to_dict(),
        }

    def cid(self) -> str:
        return _sha256_hex({"kind": "w115_frontier_certification_result_v1",
                            "result": self.to_dict()})


def run_frontier_certification_v1(
        snapshot: FrontierSnapshotV1 = W115_FRONTIER_SNAPSHOT,
        *,
        min_slice: int = MIN_RESISTANT_SLICE,
) -> FrontierCertificationResultV1:
    """The push-button W115 go/no-go matrix (RUNBOOK_W115 § 4).

    Reuses the W114 ``decide_certification_v1`` for the per-model gate + verdict
    (no re-implementation), and wraps it with the release detector, the
    frontier-date summary, the disclosure-consistency guard, and the structured
    W116 fire condition — all driven by the ``FrontierSnapshotV1``.  Re-running
    this on an updated snapshot is the W116 operation.
    """
    release_detection = detect_latest_release_v1(snapshot.observed_releases)
    frontier = frontier_date_summary_v1(snapshot.instrument, min_slice=min_slice)
    decision = decide_certification_v1(
        candidates=snapshot.candidates, instrument=snapshot.instrument)
    consistency = check_disclosure_consistency_v1(snapshot)
    consistency_ok = all(c.consistent for c in consistency)

    certifiable = (decision.verdict == VERDICT_CERTIFIABLE)
    # The instrument trigger fires when a NEWER release is observed AND it would
    # admit a >= min_slice resistant slice for a reachable KNOWN-cutoff stronger
    # model — operationally surfaced as "a newer release is available to admit".
    instrument_met = bool(release_detection.newer_release_available)
    # The cutoff trigger fires when a reachable stronger model now has a KNOWN
    # cutoff that admits a >= min_slice slice on the current instrument (i.e. some
    # non-Maverick candidate clears C1∧C2∧C3∧C4 — exactly the certifiable case).
    cutoff_met = bool(certifiable and decision.target_model
                      and "maverick" not in decision.target_model.lower())
    fires_now = bool(certifiable)
    fire = W116FireConditionV1(
        instrument_trigger=(
            "A newer official LCB release (release_v7+) is observed on the source, "
            "operator-fetched + SHA-pinned + admitted to LIVECODEBENCH_KNOWN_"
            "RELEASES, with >= "
            f"{min_slice} functional problems dated strictly after a reachable "
            "stronger-than-Maverick model's KNOWN cutoff."),
        cutoff_trigger=(
            "A reachable stronger-than-Maverick model discloses a KNOWN cutoff "
            f"month <= {frontier.max_cutoff_month_for_min_slice or 'n/a'} (so the "
            f"current {frontier.release} admits a >= {min_slice} resistant slice "
            "for it); update MODEL_TRAINING_CUTOFFS + W114_CUTOFF_PROVENANCE, then "
            "re-run."),
        instrument_trigger_met=instrument_met,
        cutoff_trigger_met=cutoff_met,
        fires_now=fires_now)

    return FrontierCertificationResultV1(
        schema=W115_FRONTIER_PIPELINE_V1_SCHEMA_VERSION,
        verified_on=snapshot.verified_on,
        release_detection=release_detection,
        frontier_summary=frontier,
        decision=decision,
        disclosure_consistency=consistency,
        disclosure_consistency_ok=consistency_ok,
        verdict=decision.verdict,
        target_model=decision.target_model,
        w116_fire_condition=fire)


__all__ = [
    "W115_FRONTIER_PIPELINE_V1_SCHEMA_VERSION",
    "release_version_num_v1",
    "ReleaseDetectionV1",
    "detect_latest_release_v1",
    "FrontierDateSummaryV1",
    "frontier_date_summary_v1",
    "ModelDisclosureV1",
    "DisclosureConsistencyV1",
    "FrontierSnapshotV1",
    "W115_FRONTIER_SNAPSHOT",
    "check_disclosure_consistency_v1",
    "W116FireConditionV1",
    "FrontierCertificationResultV1",
    "run_frontier_certification_v1",
    "VERDICT_CERTIFIABLE",
    "VERDICT_NONE",
]
