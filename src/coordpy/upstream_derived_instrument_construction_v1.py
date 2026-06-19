"""W117 / COO-9 — durable upstream-DERIVED instrument-CONSTRUCTION pipeline.

W116 attacked the upstream instrument SUPPLY at four authoritative surfaces and
shipped ``upstream_instrument_admission_v1``: an A1..A5 rule that decides whether an
OBSERVED *packaged* release may be ADMITTED, a multi-surface change detector, a
certifiable-slice builder, a disclosure matrix, and the W117 fire condition.  W116
answered "is there an admissible *packaged* ``release_v7``?" — No.

W117 asks the harder CONSTRUCTION question and does NOT just re-run the W116 checker
and wait:

> Even before a numbered ``release_v7`` is *packaged*, can we CONSTRUCT an
> upstream-authoritative, machine-checkable, certifiable post-cutoff functional
> instrument from official upstream PROVENANCE — the HF dataset revision/commit/
> discussion history, the official LCB GitHub data-generation pipeline, and the
> actual upstream contest provenance LCB draws from?

This module is the durable asset W117 lands so the next clean shot (W118) is
mechanical the moment the upstream *provenance* (not just the release label) changes.
It generalises the W116 packaged-admission pipeline into a real upstream-DERIVED
CONSTRUCTION pipeline, adding — on top of the reused W113 registry + W114
``certify_model_v1`` + W115 ``run_frontier_certification_v1`` + W116
``run_upstream_admission_v1`` / ``assess_instrument_admissibility_v1``
(explicit-import-only, NO duplication):

* ``ConstructionRuleV1`` / ``assess_construction_admissibility_v1`` — the
  PRE-COMMITTED construction rule: A1..A5 (reused) PLUS **B1 authoritative
  construction provenance** (the problem set is fully defined by an LCB-PUBLISHED
  artifact — a dataset revision/commit/PR OR a published collection pipeline +
  problem-id manifest — NOT a raw-contest hand-assembly) AND **B2 no operator
  curation** (the selection + ordering are reproducible from the published provenance;
  no operator discretion).  It REFUSES the raw-contest / aggregator / hand-curated
  path the W117 directive (and the anti-cherry-pick discipline) warns against.
* ``ProvenanceSurfaceObservationV1`` / ``UpstreamProvenanceSnapshotV1`` — the LIVE
  EIGHT-surface construction-provenance state as DATA (HF commit log / refs /
  discussions, GitHub commits / tags / repo pipeline structure, the dataset README
  provenance, the runner loader path), wrapping the W116 ``UpstreamSupplySnapshotV1``
  (the packaged-admission + model/instrument/disclosure state).
* ``construct_upstream_derived_candidate_v1`` — the candidate-instrument constructor:
  scans the provenance surfaces + the construction-candidate proposals; if any
  proposal is construction-admissible AND its LCB-published artifact actually exists,
  it DERIVES the post-v6 functional observation (handed to the W116 admission layer +
  the W114 slice builder); otherwise it returns ``constructed=False`` and the EXACT
  missing upstream provenance artifact (load-bearing).
* ``W117_DISCLOSURE_MATRIX`` — the sharpened per-model disclosure matrix (DeepSeek V4
  primary-PDF re-confirmed no-cutoff; Maverick "August 2024" verbatim; nothing
  newly-disclosed since W116).  Documentation/audit state — does NOT feed the
  certification CID (which re-derives byte-identically to W114/W115/W116 = ``258b6ed7``).
* ``run_upstream_construction_v1`` / ``W118FireConditionV1`` — the push-button runner
  + the exact W118 trigger (a packaged ``release_v7``+, OR an LCB-published post-v6
  CONSTRUCTION provenance, OR a reachable stronger model's primary-KNOWN cutoff <=
  2025-01).

The push-button W118 flow: the operator updates ONE ``UpstreamProvenanceSnapshotV1``
(a new LCB-published provenance artifact / a packaged release / a newly-disclosed
primary cutoff) and re-runs ``run_upstream_construction_v1``.  Construction stays
disciplined: a candidate is realizable ONLY after its problems are defined by an
LCB-published artifact (B1) reproducibly (B2) AND that artifact actually exists; the
pipeline never hand-assembles a release from raw contest sites.

Pure / deterministic / NIM-free / read-only.  Explicit-import-only.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Optional, Sequence

from .livecodebench_loader_v1 import LIVECODEBENCH_KNOWN_RELEASES
from .livecodebench_resistant_slice_v1 import MIN_RESISTANT_SLICE
from .frontier_certification_pipeline_v1 import release_version_num_v1
from .upstream_instrument_admission_v1 import (
    DISCLOSURE_KNOWN,
    DISCLOSURE_UNKNOWN,
    SOURCE_AGGREGATOR,
    SOURCE_OFFICIAL_HF_DATASET,
    SOURCE_OFFICIAL_LCB_GITHUB,
    VERDICT_CERTIFIABLE,
    VERDICT_NONE,
    AdmissibilityRuleV1,
    DisclosureStatusV1,
    UpstreamAdmissionResultV1,
    UpstreamInstrumentObservationV1,
    UpstreamSupplySnapshotV1,
    W116_ADMISSIBILITY_RULE,
    W116_UPSTREAM_SNAPSHOT,
    assess_instrument_admissibility_v1,
    disclosure_matrix_summary_v1,
    run_upstream_admission_v1,
)

W117_CONSTRUCTION_V1_SCHEMA_VERSION: str = (
    "coordpy.upstream_derived_instrument_construction_v1.v1")


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _max_release_num(releases: Sequence[str]) -> int:
    nums = [release_version_num_v1(r) for r in releases]
    return max(nums) if nums else 0


# ----------------------------------------------------- construction rule (§3 B1/B2)

# Derivation-kind taxonomy for a CONSTRUCTED candidate (how its problems would be
# obtained).  Only the first two are LCB-published provenance (B1-authoritative).
DERIVATION_LCB_DATASET_REVISION: str = "lcb_dataset_revision"     # an LCB HF commit/PR
DERIVATION_LCB_PUBLISHED_PIPELINE: str = "lcb_published_pipeline"  # LCB repo pipeline + manifest
DERIVATION_RAW_CONTEST_ASSEMBLY: str = "raw_contest_assembly"     # operator scrapes contests
DERIVATION_AGGREGATOR_MIRROR: str = "aggregator_mirror"          # a third-party mirror
_LCB_PUBLISHED_DERIVATIONS: frozenset[str] = frozenset(
    {DERIVATION_LCB_DATASET_REVISION, DERIVATION_LCB_PUBLISHED_PIPELINE})

# A raw-contest source is NOT one of the W116 _AUTHORITATIVE_SOURCES (only the
# official LCB HF dataset / GitHub repo are).  Recorded distinctly so the audit shows
# WHY the hand-assembly path is refused (it is the actual upstream problem origin, but
# not the LCB-curated artifact).
SOURCE_RAW_CONTEST: str = "raw_contest_site"


@dataclasses.dataclass(frozen=True)
class ConstructionRuleV1:
    """The PRE-COMMITTED W117 upstream-derived construction rule (RUNBOOK § 3).

    A CONSTRUCTED candidate is CONSTRUCTION-ADMISSIBLE iff ALL of the reused W116
    A1..A5 hold AND BOTH B1, B2 hold.  A NEW one additionally post-dates the admitted
    instrument AND its LCB-published provenance artifact actually exists.
    """

    min_slice: int = MIN_RESISTANT_SLICE
    a_rule: AdmissibilityRuleV1 = W116_ADMISSIBILITY_RULE
    b1: str = ("authoritative construction provenance: the candidate's problem set is "
               "fully defined by an LCB-PUBLISHED artifact (a livecodebench/* HF "
               "dataset revision/commit/PR OR the official LCB GitHub data-generation "
               "pipeline + a published problem-id/URL manifest); a raw-contest-site "
               "scrape / operator hand-selection / non-LCB-published assembly is NOT")
    b2: str = ("no operator curation (reproducibility): the selection AND ordering of "
               "problems is fully determined by the published provenance, so anyone "
               "re-running it obtains the byte-identical set; the operator contributes "
               "NO discretionary choice (the anti-cherry-pick / no-vibes criterion)")

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_slice": int(self.min_slice),
            "A1..A5": self.a_rule.to_dict(),
            "B1_authoritative_construction_provenance": self.b1,
            "B2_no_operator_curation": self.b2,
        }


W117_CONSTRUCTION_RULE: ConstructionRuleV1 = ConstructionRuleV1()


# -------------------------------------------------- construction-candidate proposal

@dataclasses.dataclass(frozen=True)
class ConstructionCandidateProposalV1:
    """A PROPOSED upstream-derived post-v6 candidate (the external state as DATA).

    Carries the facts the construction rule consumes.  ``provenance_artifact_exists``
    is the binding realizability bit: a proposal can be construction-admissible
    *in principle* (a published pipeline WOULD be admissible) yet not buildable now
    because the artifact does not exist on the live surface.
    """

    label: str
    derivation_kind: str
    provenance_artifact_ref: str
    provenance_artifact_exists: bool
    is_lcb_published_provenance: bool   # B1 input
    is_operator_curation: bool          # B2 input (True => fails B2)
    has_dated_problems: bool            # A2
    has_functional_subset: bool         # A3
    sha_pinnable: bool                  # A4
    histogram_reproducible: bool        # A5
    claimed_frontier_date: str
    note: str

    def _source_kind(self) -> str:
        """Map the derivation kind to a W116 source-kind for the reused A1 test.

        An LCB dataset revision is the official HF dataset (A1-authoritative); an LCB
        published pipeline is the official LCB GitHub repo (A1-authoritative); a
        raw-contest assembly is NOT authoritative; an aggregator mirror is NOT.
        """
        if self.derivation_kind == DERIVATION_LCB_DATASET_REVISION:
            return SOURCE_OFFICIAL_HF_DATASET
        if self.derivation_kind == DERIVATION_LCB_PUBLISHED_PIPELINE:
            return SOURCE_OFFICIAL_LCB_GITHUB
        if self.derivation_kind == DERIVATION_RAW_CONTEST_ASSEMBLY:
            return SOURCE_RAW_CONTEST
        return SOURCE_AGGREGATOR

    def to_observation(self) -> UpstreamInstrumentObservationV1:
        """Project to the W116 observation shape so A1..A5 reuse the W116 assessor."""
        return UpstreamInstrumentObservationV1(
            label=self.label,
            source_kind=self._source_kind(),
            source_ref=self.provenance_artifact_ref,
            release_id="release_v7_constructed",  # a post-v6 id (newer-by-version)
            has_dated_problems=self.has_dated_problems,
            has_functional_subset=self.has_functional_subset,
            sha_pinnable=self.sha_pinnable,
            histogram_reproducible=self.histogram_reproducible,
            admitted_to_loader=False,
            frontier_date=self.claimed_frontier_date,
            note=self.note)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": str(self.label),
            "derivation_kind": str(self.derivation_kind),
            "provenance_artifact_ref": str(self.provenance_artifact_ref),
            "provenance_artifact_exists": bool(self.provenance_artifact_exists),
            "is_lcb_published_provenance": bool(self.is_lcb_published_provenance),
            "is_operator_curation": bool(self.is_operator_curation),
            "has_dated_problems": bool(self.has_dated_problems),
            "has_functional_subset": bool(self.has_functional_subset),
            "sha_pinnable": bool(self.sha_pinnable),
            "histogram_reproducible": bool(self.histogram_reproducible),
            "claimed_frontier_date": str(self.claimed_frontier_date),
            "note": str(self.note),
        }


@dataclasses.dataclass(frozen=True)
class ConstructionAdmissibilityV1:
    label: str
    a1_authoritative: bool
    a2_dated: bool
    a3_functional: bool
    a4_sha_pinnable: bool
    a5_histogram: bool
    a_admissible: bool                 # A1..A5
    b1_authoritative_provenance: bool
    b2_no_operator_curation: bool
    construction_admissible: bool      # A1..A5 ∧ B1 ∧ B2 (rule-level)
    newer_than_admitted: bool
    artifact_exists: bool
    realizable: bool                   # construction_admissible ∧ newer ∧ artifact_exists
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": str(self.label),
            "a1_authoritative": bool(self.a1_authoritative),
            "a2_dated": bool(self.a2_dated),
            "a3_functional": bool(self.a3_functional),
            "a4_sha_pinnable": bool(self.a4_sha_pinnable),
            "a5_histogram": bool(self.a5_histogram),
            "a_admissible": bool(self.a_admissible),
            "b1_authoritative_provenance": bool(self.b1_authoritative_provenance),
            "b2_no_operator_curation": bool(self.b2_no_operator_curation),
            "construction_admissible": bool(self.construction_admissible),
            "newer_than_admitted": bool(self.newer_than_admitted),
            "artifact_exists": bool(self.artifact_exists),
            "realizable": bool(self.realizable),
            "reason": str(self.reason),
        }


def assess_construction_admissibility_v1(
        proposal: ConstructionCandidateProposalV1,
        *,
        rule: ConstructionRuleV1 = W117_CONSTRUCTION_RULE,
        admitted_latest_num: int,
        admitted_frontier_date: str,
) -> ConstructionAdmissibilityV1:
    """Apply A1..A5 (reused W116 assessor) ∧ B1 ∧ B2 to one construction proposal.

    ``construction_admissible`` ⟺ A1∧A2∧A3∧A4∧A5∧B1∧B2 (the rule would admit a
    candidate of this shape).  ``realizable`` (the thing that can earn a pilot) ⟺
    ``construction_admissible`` ∧ newer-than-admitted ∧ the LCB-published artifact
    actually exists — a published pipeline WOULD be admissible, but cannot be built
    until its artifact is real.
    """
    a = assess_instrument_admissibility_v1(
        proposal.to_observation(),
        rule=rule.a_rule,
        admitted_latest_num=admitted_latest_num,
        admitted_frontier_date=admitted_frontier_date)
    b1 = bool(proposal.is_lcb_published_provenance
              and proposal.derivation_kind in _LCB_PUBLISHED_DERIVATIONS)
    b2 = bool(not proposal.is_operator_curation)
    construction_admissible = bool(a.admissible and b1 and b2)
    artifact_exists = bool(proposal.provenance_artifact_exists)
    realizable = bool(
        construction_admissible and a.newer_than_admitted and artifact_exists)

    if not a.admissible:
        failed = [
            name for name, ok in (
                ("A1_authoritative", a.a1_authoritative),
                ("A2_dated", a.a2_dated), ("A3_functional", a.a3_functional),
                ("A4_sha_pinnable", a.a4_sha_pinnable),
                ("A5_histogram", a.a5_histogram))
            if not ok]
        reason = (f"CONSTRUCTION_INADMISSIBLE [{','.join(failed)} fail]: {proposal.note}")
    elif not b1:
        reason = (
            "CONSTRUCTION_INADMISSIBLE [B1 fail — not LCB-published provenance]: "
            f"derivation '{proposal.derivation_kind}' is not a livecodebench/* "
            "dataset revision/PR or the official LCB GitHub pipeline+manifest "
            "(raw-contest scrape / hand-assembly / mirror). A SHA-pinnable dated "
            "JSONL is NOT enough — the problem SET must be defined by an "
            f"LCB-published artifact. {proposal.note}")
    elif not b2:
        reason = (
            "CONSTRUCTION_INADMISSIBLE [B2 fail — operator curation]: the problem "
            "selection involves operator discretion (not reproducible from a "
            f"published manifest) => vibes-based cherry-picking refused. {proposal.note}")
    elif not a.newer_than_admitted:
        reason = (
            f"CONSTRUCTION_ADMISSIBLE_BUT_NOT_NEWER: {proposal.label} satisfies "
            "A1..A5 ∧ B1 ∧ B2 but does not post-date the admitted instrument.")
    elif not artifact_exists:
        reason = (
            f"CONSTRUCTION_ADMISSIBLE_BUT_ARTIFACT_ABSENT: {proposal.label} WOULD be "
            "construction-admissible (A1..A5 ∧ B1 ∧ B2, post-frontier) — but its "
            "LCB-published provenance artifact does not exist on the live surface "
            f"({proposal.provenance_artifact_ref}). This names the exact missing "
            "upstream artifact (RUNBOOK § 3 + § 7); not buildable now.")
    else:
        reason = (
            f"REALIZABLE: {proposal.label} is construction-admissible AND newer AND "
            "its LCB-published artifact exists => construct + SHA-pin + admit + slice "
            "(RUNBOOK § 5).")

    return ConstructionAdmissibilityV1(
        label=proposal.label,
        a1_authoritative=a.a1_authoritative, a2_dated=a.a2_dated,
        a3_functional=a.a3_functional, a4_sha_pinnable=a.a4_sha_pinnable,
        a5_histogram=a.a5_histogram, a_admissible=a.admissible,
        b1_authoritative_provenance=b1, b2_no_operator_curation=b2,
        construction_admissible=construction_admissible,
        newer_than_admitted=a.newer_than_admitted,
        artifact_exists=artifact_exists, realizable=realizable, reason=reason)


# ------------------------------------------------ construction-provenance surfaces

SURFACE_HF_COMMIT_LOG: str = "hf_dataset_commit_log"
SURFACE_HF_REFS: str = "hf_dataset_refs"
SURFACE_HF_DISCUSSIONS: str = "hf_dataset_discussions"
SURFACE_GITHUB_COMMITS: str = "lcb_github_commits"
SURFACE_GITHUB_TAGS: str = "lcb_github_tags"
SURFACE_GITHUB_PIPELINE: str = "lcb_github_pipeline_structure"
SURFACE_DATASET_README: str = "dataset_readme_provenance"
SURFACE_RUNNER_LOADER: str = "runner_loader_path"


@dataclasses.dataclass(frozen=True)
class ProvenanceSurfaceObservationV1:
    """One OBSERVED construction-provenance surface (the upstream state as DATA).

    ``has_post_frontier_lcb_artifact`` is the binding bit: does this surface expose an
    LCB-PUBLISHED artifact (a dataset revision/commit/PR / a published pipeline+
    manifest) that adds functional problems dated strictly after the admitted
    instrument frontier (2025-04-05)?  On the W117 live snapshot every surface is
    False — the construction supply is absent at the provenance level, not just the
    release-label level.
    """

    surface_kind: str
    source_ref: str
    finding: str
    has_post_frontier_lcb_artifact: bool
    note: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "surface_kind": str(self.surface_kind),
            "source_ref": str(self.source_ref),
            "finding": str(self.finding),
            "has_post_frontier_lcb_artifact": bool(
                self.has_post_frontier_lcb_artifact),
            "note": str(self.note),
        }


# --------------------------------------------------- upstream provenance snapshot

@dataclasses.dataclass(frozen=True)
class UpstreamProvenanceSnapshotV1:
    """The LIVE EIGHT-surface construction-provenance state as DATA (W118 re-input).

    Wraps the W116 ``UpstreamSupplySnapshotV1`` (the packaged-admission +
    model/instrument/disclosure state, reused unchanged so the certification decision
    re-derives byte-identically) and adds the construction-provenance surfaces + the
    construction-candidate proposals.
    """

    verified_on: str
    source_note: str
    provenance_surfaces: tuple[ProvenanceSurfaceObservationV1, ...]
    proposals: tuple[ConstructionCandidateProposalV1, ...]
    supply_snapshot: UpstreamSupplySnapshotV1

    def to_dict(self) -> dict[str, Any]:
        return {
            "verified_on": str(self.verified_on),
            "source_note": str(self.source_note),
            "provenance_surfaces": [
                s.to_dict() for s in self.provenance_surfaces],
            "proposals": [p.to_dict() for p in self.proposals],
            "supply_snapshot": self.supply_snapshot.to_dict(),
        }


# The LOCKED W117 provenance snapshot — the LIVE primary-source upstream-provenance
# attack of 2026-05-30 (RUNBOOK_W117 § 2).  EIGHT surfaces, none exposing a post-v6
# LCB-published artifact; two proposals (the only post-v6 paths): a raw-contest
# hand-assembly (REFUSED by A1 ∧ B1 ∧ B2) and an LCB-published-pipeline template
# (construction-admissible in principle, but its artifact does not exist).  The
# packaged-admission/model state reuses the W116 snapshot (=> CID 258b6ed7).
W117_PROVENANCE_SNAPSHOT: UpstreamProvenanceSnapshotV1 = UpstreamProvenanceSnapshotV1(
    verified_on="2026-05-30",
    source_note=(
        "W117 live primary-source upstream-PROVENANCE / construction attack "
        "(RUNBOOK_W117 § 2). EIGHT surfaces verify the CONSTRUCTION supply (revision "
        "history + collection mechanism), not just the release label: no post-v6 "
        "LCB-published artifact exists at any surface; the only post-v6 path is a "
        "raw-contest hand-assembly, which is CONSTRUCTION-INADMISSIBLE (B1 + B2). "
        "Model side re-confirmed deeper from primary sources: DeepSeek V4 official "
        "model-card PDF states NO cutoff; Maverick 'August 2024' verbatim."),
    provenance_surfaces=(
        ProvenanceSurfaceObservationV1(
            surface_kind=SURFACE_HF_COMMIT_LOG,
            source_ref="hf api datasets/livecodebench/code_generation_lite/"
                       "commits/main",
            finding="20 commits; latest data-bearing = 'add v6' 2025-04-21; HEAD = "
                    "'fix typos (#4)' 2025-06-05.",
            has_post_frontier_lcb_artifact=False,
            note="No post-v6 data commit/revision; no test7. The revision history "
                 "itself (not just the file tree) confirms v6 is the newest data."),
        ProvenanceSurfaceObservationV1(
            surface_kind=SURFACE_HF_REFS,
            source_ref="hf api datasets/livecodebench/code_generation_lite/refs",
            finding="1 branch (main), 0 tags.",
            has_post_frontier_lcb_artifact=False,
            note="No staging / preview / v7 branch or tag carrying newer data."),
        ProvenanceSurfaceObservationV1(
            surface_kind=SURFACE_HF_DISCUSSIONS,
            source_ref="hf datasets/livecodebench/code_generation_lite/discussions",
            finding="Newest threads = 'LCB pull request' #14 + 'Clarification on v6 "
                    "size (454 vs 175)' #13; older = parquet/loader-script issues.",
            has_post_frontier_lcb_artifact=False,
            note="No v7 / newer-data / post-Apr-2025 discussion or PR proposing new "
                 "problems."),
        ProvenanceSurfaceObservationV1(
            surface_kind=SURFACE_GITHUB_COMMITS,
            source_ref="github api repos/LiveCodeBench/LiveCodeBench/commits",
            finding="Newest 2025-07-16 ('fix: typo in Explorer URL'); recent activity "
                    "= runner maintenance (model infos, gemini, ERRATA, util fixes).",
            has_post_frontier_lcb_artifact=False,
            note="No data-collection commit; no v7. The repo is the runner/eval "
                 "harness, not a data-collection pipeline."),
        ProvenanceSurfaceObservationV1(
            surface_kind=SURFACE_GITHUB_TAGS,
            source_ref="github api repos/LiveCodeBench/LiveCodeBench/tags",
            finding="Empty (0 tags).",
            has_post_frontier_lcb_artifact=False,
            note="No release_v7 / dated release tag."),
        ProvenanceSurfaceObservationV1(
            surface_kind=SURFACE_GITHUB_PIPELINE,
            source_ref="github repo root contents (LiveCodeBench/LiveCodeBench)",
            finding="Top-level = lcb_runner/ + assets/ + configs (pyproject, "
                    "uv.lock, ...); NO scraper / data / collect / contest directory.",
            has_post_frontier_lcb_artifact=False,
            note="The public repo publishes NO data-generation/collection pipeline => "
                 "no published mechanism from which a post-v6 slice could be "
                 "reproducibly CONSTRUCTED."),
        ProvenanceSurfaceObservationV1(
            surface_kind=SURFACE_DATASET_README,
            source_ref="hf datasets/livecodebench/code_generation_lite/raw/main/"
                       "README.md",
            finding="Provenance = LeetCode / AtCoder / Codeforces contests; each "
                    "release_vN a temporal snapshot (README prose lags at v5). "
                    "Documents ONLY loading published releases.",
            has_post_frontier_lcb_artifact=False,
            note="No generation tool / script / problem-id manifest documented for "
                 "collecting NEW problems => the published releases ARE the provenance."),
        ProvenanceSurfaceObservationV1(
            surface_kind=SURFACE_RUNNER_LOADER,
            source_ref="github raw lcb_runner/benchmarks/code_generation.py",
            finding="Loads EXCLUSIVELY via load_dataset('livecodebench/"
                    "code_generation_lite', version_tag=...); no local scraping.",
            has_post_frontier_lcb_artifact=False,
            note="The provenance flows contest-sites -> LCB's (private, unpublished) "
                 "collection -> HF release -> runner. Construction can only enter at "
                 "the HF-release level, which requires LCB to publish it."),
    ),
    proposals=(
        ConstructionCandidateProposalV1(
            label="raw-contest hand-assembly (post-2025-04)",
            derivation_kind=DERIVATION_RAW_CONTEST_ASSEMBLY,
            provenance_artifact_ref="operator-scraped LeetCode/AtCoder/Codeforces "
                                    "problems dated after 2025-04-05 (hand-selected)",
            provenance_artifact_exists=True,   # the contests are real + dated...
            is_lcb_published_provenance=False,  # ...but NOT an LCB-published artifact
            is_operator_curation=True,          # operator chooses which problems
            has_dated_problems=True, has_functional_subset=True,
            sha_pinnable=True, histogram_reproducible=True,
            claimed_frontier_date="2025-12-31",
            note="The ONLY post-v6 path that exists today. A SHA-pinnable, dated, "
                 "functional JSONL CAN be assembled from raw contests — but it is "
                 "REFUSED: not an LCB-published provenance artifact (B1) and "
                 "operator-curated / not reproducible from a published manifest (B2); "
                 "also not the official livecodebench/* source (A1). Selling it as a "
                 "LiveCodeBench-grade resistant instrument would be exactly the "
                 "vibes-based cherry-picking the discipline forbids."),
        ConstructionCandidateProposalV1(
            label="LCB-published collection pipeline + manifest (template)",
            derivation_kind=DERIVATION_LCB_PUBLISHED_PIPELINE,
            provenance_artifact_ref="(hypothetical) official LCB GitHub data-"
                                    "generation pipeline + a published post-2025-04 "
                                    "problem-id/URL manifest",
            provenance_artifact_exists=False,   # NOT published on the live surface
            is_lcb_published_provenance=True,   # B1 would pass
            is_operator_curation=False,         # B2 would pass (reproducible)
            has_dated_problems=True, has_functional_subset=True,
            sha_pinnable=True, histogram_reproducible=True,
            claimed_frontier_date="2025-12-31",
            note="The construction path that WOULD be admissible: if LCB published its "
                 "collection pipeline + a problem-id manifest for post-2025-04 "
                 "problems, anyone could reproducibly construct the slice. It "
                 "satisfies A1..A5 ∧ B1 ∧ B2 — but the artifact does NOT exist on the "
                 "live surface (surface 6 + 7). This is the EXACT missing upstream "
                 "provenance artifact (the load-bearing W118 trigger)."),
    ),
    supply_snapshot=W116_UPSTREAM_SNAPSHOT)


# ------------------------------------------------------- candidate-instrument constructor

@dataclasses.dataclass(frozen=True)
class ConstructionAttemptV1:
    """The result of attempting to CONSTRUCT a post-v6 candidate from provenance."""

    constructed: bool
    n_construction_admissible_new: int       # realizable proposals
    n_construction_admissible_in_principle: int
    proposal_admissibility: tuple[ConstructionAdmissibilityV1, ...]
    surfaces_with_post_frontier_artifact: tuple[str, ...]
    derived_observation: Optional[UpstreamInstrumentObservationV1]
    missing_artifact: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "constructed": bool(self.constructed),
            "n_construction_admissible_new": int(
                self.n_construction_admissible_new),
            "n_construction_admissible_in_principle": int(
                self.n_construction_admissible_in_principle),
            "proposal_admissibility": [
                p.to_dict() for p in self.proposal_admissibility],
            "surfaces_with_post_frontier_artifact": list(
                self.surfaces_with_post_frontier_artifact),
            "derived_observation": (
                self.derived_observation.to_dict()
                if self.derived_observation is not None else None),
            "missing_artifact": str(self.missing_artifact),
        }


_MISSING_ARTIFACT_W117: str = (
    "An LCB-published post-v6 provenance artifact: a livecodebench/code_generation"
    "(_lite) dataset revision/commit/PR adding functional problems dated strictly "
    "after 2025-04-05, OR the official LCB GitHub data-generation pipeline + a "
    "published problem-id/URL manifest enabling reproducible (B1 ∧ B2) construction. "
    "Verified ABSENT at all eight provenance surfaces (commit log / refs / "
    "discussions / GitHub commits / GitHub tags / repo pipeline structure / README / "
    "runner loader). The only post-v6 path (raw-contest hand-assembly) is refused by "
    "A1 ∧ B1 ∧ B2.")


def construct_upstream_derived_candidate_v1(
        snapshot: UpstreamProvenanceSnapshotV1 = W117_PROVENANCE_SNAPSHOT,
        *,
        rule: ConstructionRuleV1 = W117_CONSTRUCTION_RULE,
) -> ConstructionAttemptV1:
    """Attempt to CONSTRUCT a post-v6 functional candidate from upstream provenance.

    Assesses each construction-candidate proposal (A1..A5 ∧ B1 ∧ B2 + realizability)
    against the loader-admitted baseline.  ``constructed`` ⟺ at least one proposal is
    REALIZABLE (construction-admissible ∧ newer ∧ its LCB-published artifact exists);
    in that case the derived observation is handed to the W116 admission layer + the
    W114 slice builder.  Otherwise it returns the EXACT missing provenance artifact.
    """
    adm_latest_num = _max_release_num(LIVECODEBENCH_KNOWN_RELEASES)
    adm_frontier = snapshot.supply_snapshot.frontier_snapshot.instrument.\
        functional_date_max
    assessed = tuple(
        assess_construction_admissibility_v1(
            p, rule=rule, admitted_latest_num=adm_latest_num,
            admitted_frontier_date=adm_frontier)
        for p in snapshot.proposals)
    realizable = [a for a in assessed if a.realizable]
    in_principle = [a for a in assessed if a.construction_admissible]
    surfaces_hot = tuple(
        s.surface_kind for s in snapshot.provenance_surfaces
        if s.has_post_frontier_lcb_artifact)

    constructed = bool(realizable)
    derived_obs: Optional[UpstreamInstrumentObservationV1] = None
    if constructed:
        first = realizable[0]
        prop = next(p for p in snapshot.proposals if p.label == first.label)
        derived_obs = dataclasses.replace(
            prop.to_observation(), admitted_to_loader=True,
            note=f"CONSTRUCTED from {prop.provenance_artifact_ref}")
    missing = (
        "NONE — a construction-admissible instrument was built." if constructed
        else _MISSING_ARTIFACT_W117)

    return ConstructionAttemptV1(
        constructed=constructed,
        n_construction_admissible_new=len(realizable),
        n_construction_admissible_in_principle=len(in_principle),
        proposal_admissibility=assessed,
        surfaces_with_post_frontier_artifact=surfaces_hot,
        derived_observation=derived_obs,
        missing_artifact=missing)


# ------------------------------------------------ sharpened disclosure matrix (§4b)

DISCLOSURE_NEWLY_DISCLOSED: str = "NEWLY_DISCLOSED_SINCE_W116"

# The W117 LIVE primary-source disclosure matrix (2026-05-30), sharper than W116:
# DeepSeek V4 re-checked at its official model-card PDF (still NO cutoff; the only
# figure is a non-primary aggregator that is itself C2-exposed — now mirroring
# Mistral); Maverick "August 2024" re-confirmed VERBATIM from the Meta MODEL_CARD.md.
# Nothing is newly-disclosed since W116.  Documentation/audit state; does NOT feed the
# certification CID.
W117_DISCLOSURE_MATRIX: tuple[DisclosureStatusV1, ...] = (
    DisclosureStatusV1(
        model_id="meta/llama-4-maverick-17b-128e-instruct",
        primary_status=DISCLOSURE_KNOWN,
        primary_source="Official Meta Llama 4 MODEL_CARD.md (meta-llama/llama-models) "
                       "— 2026-05-30: 'Knowledge cutoff' = 'August 2024' VERBATIM",
        aggregator_signal="Aug-2024 (corroborated)",
        stronger_than_70b=True,
        certifiable_blocker="C4 (already SETTLED on release_v6; W113 FAIL)",
        note="Re-confirmed verbatim from the primary model card. KNOWN Aug-2024 but "
             "redundant on release_v6 => no new pilot here."),
    DisclosureStatusV1(
        model_id="qwen/qwen3-coder-480b-a35b-instruct",
        primary_status=DISCLOSURE_UNKNOWN,
        primary_source="Official HF model card raw README "
                       "(Qwen/Qwen3-Coder-480B-A35B-Instruct) — 2026-05-30: NO "
                       "CUTOFF STATED",
        aggregator_signal="(none usable; tech report arXiv:2505.09388, 2025)",
        stronger_than_70b=True,
        certifiable_blocker="C1 (UNKNOWN cutoff); C2-exposed if estimated ~2025",
        note="Reconfirmed UNKNOWN from the official card raw markdown."),
    DisclosureStatusV1(
        model_id="deepseek-ai/deepseek-v4-pro",
        primary_status=DISCLOSURE_UNKNOWN,
        primary_source="Official DeepSeek V4 model-card PDF "
                       "(fe-static.deepseek.com/.../deepseek-V4-model-card-EN.pdf) — "
                       "2026-05-30: NO CUTOFF STATED",
        aggregator_signal="non-primary blogs say 'Apr 2026' (C2-exposed; a year "
                          "past the Apr-2025 frontier) => CONTRADICTORY-with-primary",
        stronger_than_70b=True,
        certifiable_blocker="C1 (UNKNOWN from primary); the aggregator 'Apr 2026' "
                            "figure post-dates the frontier => C2-exposed",
        note="SHARPENED vs W116: the official V4 PDF re-checked at primary still "
             "states no cutoff; the only figure is a non-primary aggregator (Apr-"
             "2026) that is itself C2-exposed => now the SAME pattern as Mistral "
             "Small 4 (UNKNOWN-primary + C2-exposed-aggregator)."),
    DisclosureStatusV1(
        model_id="mistralai/mistral-small-4-119b-2603",
        primary_status=DISCLOSURE_UNKNOWN,
        primary_source="Official Mistral docs models overview "
                       "(docs.mistral.ai/.../models_overview) — 2026-05-30: NO "
                       "CUTOFF STATED",
        aggregator_signal="OpenRouter '2025-06' (non-primary; post-dates frontier "
                          "=> C2-exposed)",
        stronger_than_70b=True,
        certifiable_blocker="C1 (UNKNOWN from primary); the 2025-06 aggregator "
                            "estimate post-dates the Apr-2025 frontier => C2-exposed",
        note="CONFIRMED REAL (119B MoE, 2026-03-16); primary docs disclose NO "
             "cutoff. Same UNKNOWN-primary + C2-exposed-aggregator pattern as "
             "DeepSeek V4."),
    DisclosureStatusV1(
        model_id="mistralai/mistral-small-3.2-24b-instruct-2506",
        primary_status=DISCLOSURE_KNOWN,
        primary_source="HF discussion / aggregator (~Oct-2023)",
        aggregator_signal="2023-10",
        stronger_than_70b=False,
        certifiable_blocker="C3 (24B; NOT strictly stronger than 70B)",
        note="Context only: a KNOWN-cutoff Mistral exists but is sub-70B (C3); "
             "deprecated 2026-04-30, replaced by Small 4."),
)


def disclosure_delta_since_w116_v1(
        matrix: Sequence[DisclosureStatusV1] = W117_DISCLOSURE_MATRIX,
) -> dict[str, Any]:
    """The Lane β summary: counts + whether anything was NEWLY DISCLOSED since W116
    (a stronger-than-70B model gaining a primary-KNOWN, not-settled cutoff)."""
    base = disclosure_matrix_summary_v1(matrix)
    newly = [
        d.model_id for d in matrix
        if d.primary_status == DISCLOSURE_NEWLY_DISCLOSED]
    base["newly_disclosed_since_w116"] = newly
    base["any_newly_disclosed_since_w116"] = bool(newly)
    return base


# ------------------------------------------------------------ W118 fire condition

@dataclasses.dataclass(frozen=True)
class W118FireConditionV1:
    """The pre-committed triggers that flip the no-go verdict (RUNBOOK § 9)."""

    packaged_release_trigger: str
    construction_provenance_trigger: str
    cutoff_trigger: str
    packaged_release_trigger_met: bool
    construction_provenance_trigger_met: bool
    cutoff_trigger_met: bool
    fires_now: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "packaged_release_trigger": str(self.packaged_release_trigger),
            "construction_provenance_trigger": str(
                self.construction_provenance_trigger),
            "cutoff_trigger": str(self.cutoff_trigger),
            "packaged_release_trigger_met": bool(
                self.packaged_release_trigger_met),
            "construction_provenance_trigger_met": bool(
                self.construction_provenance_trigger_met),
            "cutoff_trigger_met": bool(self.cutoff_trigger_met),
            "fires_now": bool(self.fires_now),
        }


# ------------------------------------------------------------- the push-button API

@dataclasses.dataclass(frozen=True)
class UpstreamConstructionResultV1:
    schema: str
    verified_on: str
    construction_rule: dict[str, Any]
    provenance_surfaces: tuple[ProvenanceSurfaceObservationV1, ...]
    n_surfaces_with_post_frontier_artifact: int
    construction_attempt: ConstructionAttemptV1
    upstream_admission: UpstreamAdmissionResultV1
    disclosure_matrix: tuple[DisclosureStatusV1, ...]
    disclosure_summary: dict[str, Any]
    verdict: str
    target_model: str
    w118_fire_condition: W118FireConditionV1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "verified_on": str(self.verified_on),
            "construction_rule": self.construction_rule,
            "provenance_surfaces": [
                s.to_dict() for s in self.provenance_surfaces],
            "n_surfaces_with_post_frontier_artifact": int(
                self.n_surfaces_with_post_frontier_artifact),
            "construction_attempt": self.construction_attempt.to_dict(),
            "upstream_admission": self.upstream_admission.to_dict(),
            "disclosure_matrix": [d.to_dict() for d in self.disclosure_matrix],
            "disclosure_summary": self.disclosure_summary,
            "verdict": str(self.verdict),
            "target_model": str(self.target_model),
            "w118_fire_condition": self.w118_fire_condition.to_dict(),
        }

    def cid(self) -> str:
        return _sha256_hex({"kind": "w117_upstream_construction_result_v1",
                            "result": self.to_dict()})


def run_upstream_construction_v1(
        snapshot: UpstreamProvenanceSnapshotV1 = W117_PROVENANCE_SNAPSHOT,
        *,
        rule: ConstructionRuleV1 = W117_CONSTRUCTION_RULE,
        min_slice: int = MIN_RESISTANT_SLICE,
) -> UpstreamConstructionResultV1:
    """The push-button W117 construction + admission + certification (RUNBOOK § 4).

    Reuses the W116 ``run_upstream_admission_v1`` for the packaged-admission layer +
    the per-model gate + verdict (which reuses the W115 → W114 chain; the decision CID
    re-derives byte-identically, ``258b6ed7``) and wraps it with the construction rule,
    the EIGHT-surface provenance attempt, the sharpened disclosure matrix, and the
    structured W118 fire condition.  Re-running this on an updated snapshot is the W118
    operation.
    """
    admission = run_upstream_admission_v1(
        snapshot.supply_snapshot, min_slice=min_slice)
    attempt = construct_upstream_derived_candidate_v1(snapshot, rule=rule)
    disclosure_summary = disclosure_delta_since_w116_v1(W117_DISCLOSURE_MATRIX)

    certifiable = (admission.verdict == VERDICT_CERTIFIABLE)
    n_hot = sum(
        1 for s in snapshot.provenance_surfaces
        if s.has_post_frontier_lcb_artifact)

    # Packaged trigger: a newer packaged release is admissible (the W116 instrument
    # trigger).  Construction trigger: a post-v6 candidate was actually CONSTRUCTED
    # from LCB-published provenance.  Cutoff trigger: a non-Maverick model certifies.
    packaged_met = bool(
        admission.w117_fire_condition.instrument_trigger_met)
    construction_met = bool(attempt.constructed)
    cutoff_met = bool(admission.w117_fire_condition.cutoff_trigger_met)
    # The binding "run a pilot now" gate stays certification (a constructed instrument
    # is only useful if it certifies a stronger model) — identical to W116/W115.
    fires_now = bool(certifiable)

    max_month = (admission.frontier_certification.frontier_summary
                 .max_cutoff_month_for_min_slice or "n/a")
    release = admission.frontier_certification.frontier_summary.release
    fire = W118FireConditionV1(
        packaged_release_trigger=(
            "A newer PACKAGED LCB release (release_v7+ on the lite tree, OR "
            "release_latest re-pointing past release_v6) is observed, operator-"
            "fetched + SHA-pinned + admitted to LIVECODEBENCH_KNOWN_RELEASES, with "
            f">= {min_slice} functional problems dated strictly after a reachable "
            "stronger-than-Maverick model's primary-KNOWN cutoff (RUNBOOK § 3 + § 5)."),
        construction_provenance_trigger=(
            "An LCB-PUBLISHED post-v6 CONSTRUCTION provenance appears — a "
            "livecodebench/* dataset revision/commit/PR adding post-2025-04 "
            "functional problems, OR the official LCB GitHub collection pipeline + a "
            "published problem-id/URL manifest — enabling a B1 ∧ B2 reproducible "
            f">= {min_slice} functional post-cutoff slice (RUNBOOK § 3 B1/B2). "
            "Missing artifact: " + attempt.missing_artifact),
        cutoff_trigger=(
            "A reachable stronger-than-Maverick model discloses, from a PRIMARY "
            f"source, a KNOWN cutoff month <= {max_month} (so the current {release} "
            f"admits a >= {min_slice} resistant slice for it); update "
            "MODEL_TRAINING_CUTOFFS + W114_CUTOFF_PROVENANCE, then re-run."),
        packaged_release_trigger_met=packaged_met,
        construction_provenance_trigger_met=construction_met,
        cutoff_trigger_met=cutoff_met,
        fires_now=fires_now)

    return UpstreamConstructionResultV1(
        schema=W117_CONSTRUCTION_V1_SCHEMA_VERSION,
        verified_on=snapshot.verified_on,
        construction_rule=rule.to_dict(),
        provenance_surfaces=snapshot.provenance_surfaces,
        n_surfaces_with_post_frontier_artifact=int(n_hot),
        construction_attempt=attempt,
        upstream_admission=admission,
        disclosure_matrix=W117_DISCLOSURE_MATRIX,
        disclosure_summary=disclosure_summary,
        verdict=admission.verdict,
        target_model=admission.target_model,
        w118_fire_condition=fire)


__all__ = [
    "W117_CONSTRUCTION_V1_SCHEMA_VERSION",
    "DERIVATION_LCB_DATASET_REVISION",
    "DERIVATION_LCB_PUBLISHED_PIPELINE",
    "DERIVATION_RAW_CONTEST_ASSEMBLY",
    "DERIVATION_AGGREGATOR_MIRROR",
    "SOURCE_RAW_CONTEST",
    "ConstructionRuleV1",
    "W117_CONSTRUCTION_RULE",
    "ConstructionCandidateProposalV1",
    "ConstructionAdmissibilityV1",
    "assess_construction_admissibility_v1",
    "SURFACE_HF_COMMIT_LOG",
    "SURFACE_HF_REFS",
    "SURFACE_HF_DISCUSSIONS",
    "SURFACE_GITHUB_COMMITS",
    "SURFACE_GITHUB_TAGS",
    "SURFACE_GITHUB_PIPELINE",
    "SURFACE_DATASET_README",
    "SURFACE_RUNNER_LOADER",
    "ProvenanceSurfaceObservationV1",
    "UpstreamProvenanceSnapshotV1",
    "W117_PROVENANCE_SNAPSHOT",
    "ConstructionAttemptV1",
    "construct_upstream_derived_candidate_v1",
    "DISCLOSURE_NEWLY_DISCLOSED",
    "W117_DISCLOSURE_MATRIX",
    "disclosure_delta_since_w116_v1",
    "W118FireConditionV1",
    "UpstreamConstructionResultV1",
    "run_upstream_construction_v1",
    "VERDICT_CERTIFIABLE",
    "VERDICT_NONE",
]
