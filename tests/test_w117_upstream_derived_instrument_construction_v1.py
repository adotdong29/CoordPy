"""W117 / COO-9 — tests for the durable upstream-DERIVED construction pipeline.

Covers ``coordpy.upstream_derived_instrument_construction_v1``:

* the construction rule A1..A5 (reused) ∧ B1 (authoritative LCB-published provenance)
  ∧ B2 (no operator curation): the raw-contest hand-assembly is REFUSED (A1 ∧ B1 ∧ B2
  all fail); the LCB-published-pipeline template is construction-admissible IN
  PRINCIPLE but NOT realizable (artifact absent); B1 + B2 are each independently
  load-bearing (isolation tests);
* the EIGHT construction-provenance surfaces (none expose a post-frontier LCB
  artifact);
* the construction attempt: constructed == False; 0 realizable; 1 admissible-in-
  principle; the exact missing upstream artifact is named;
* the sharpened disclosure matrix (DeepSeek V4 UNKNOWN-from-primary with an
  aggregator figure that is C2-exposed; Maverick 'August 2024' verbatim; nothing
  newly-disclosed since W116);
* the real-snapshot verdict == NO_CERTIFIABLE_STRONGER_MODEL; W118 fires_now == False;
* the REUSE INVARIANT: the certification decision re-derives byte-identically to the
  W114/W115/W116 decision (prefix 258b6ed7) — W117 wraps, never forks, the gate;
* a FALSIFIABILITY test: a synthetic snapshot with a REALIZABLE construction-admissible
  instrument + a certifying supply snapshot DOES construct, certify, name the target,
  and fire the W118 construction + cutoff triggers.
"""
from __future__ import annotations

import dataclasses

from coordpy.upstream_derived_instrument_construction_v1 import (
    DERIVATION_LCB_PUBLISHED_PIPELINE,
    DERIVATION_RAW_CONTEST_ASSEMBLY,
    DISCLOSURE_KNOWN,
    DISCLOSURE_UNKNOWN,
    VERDICT_CERTIFIABLE,
    VERDICT_NONE,
    ConstructionCandidateProposalV1,
    UpstreamProvenanceSnapshotV1,
    W117_DISCLOSURE_MATRIX,
    W117_PROVENANCE_SNAPSHOT,
    assess_construction_admissibility_v1,
    construct_upstream_derived_candidate_v1,
    disclosure_delta_since_w116_v1,
    run_upstream_construction_v1,
)
from coordpy.upstream_instrument_admission_v1 import (
    UpstreamInstrumentObservationV1,
    UpstreamSupplySnapshotV1,
    W116_UPSTREAM_SNAPSHOT,
    SOURCE_OFFICIAL_HF_DATASET,
)
from coordpy.frontier_certification_pipeline_v1 import FrontierSnapshotV1
from coordpy.stronger_model_cutoff_certification_v1 import (
    STRONGER_MODEL_CANDIDATES,
    LatestResistantInstrumentV1,
    decide_certification_v1,
)
from coordpy.livecodebench_loader_v1 import LIVECODEBENCH_KNOWN_RELEASES


# --------------------------------------------------------- construction rule (§3)

def _assess(proposal, *, latest_num=6, frontier="2025-04-05"):
    return assess_construction_admissibility_v1(
        proposal, admitted_latest_num=latest_num, admitted_frontier_date=frontier)


def test_raw_contest_assembly_is_refused_by_A1_B1_B2():
    by_label = {p.label: p for p in W117_PROVENANCE_SNAPSHOT.proposals}
    raw = by_label["raw-contest hand-assembly (post-2025-04)"]
    a = _assess(raw)
    assert a.a1_authoritative is False        # not the official livecodebench/* source
    assert a.b1_authoritative_provenance is False  # not LCB-published provenance
    assert a.b2_no_operator_curation is False      # operator-curated
    assert a.construction_admissible is False
    assert a.realizable is False
    assert "CONSTRUCTION_INADMISSIBLE" in a.reason


def test_lcb_pipeline_template_admissible_in_principle_but_artifact_absent():
    by_label = {p.label: p for p in W117_PROVENANCE_SNAPSHOT.proposals}
    pipe = by_label["LCB-published collection pipeline + manifest (template)"]
    a = _assess(pipe)
    assert a.a_admissible is True             # A1..A5 all pass
    assert a.b1_authoritative_provenance is True
    assert a.b2_no_operator_curation is True
    assert a.construction_admissible is True  # rule WOULD admit it
    assert a.artifact_exists is False         # ...but the artifact does not exist
    assert a.realizable is False
    assert "ARTIFACT_ABSENT" in a.reason


def test_B2_is_independently_load_bearing():
    # An LCB-published pipeline (A1 + B1 pass) that the operator hand-curates fails
    # ONLY on B2 — isolating the anti-cherry-pick criterion.
    curated = ConstructionCandidateProposalV1(
        label="LCB pipeline but operator-curated",
        derivation_kind=DERIVATION_LCB_PUBLISHED_PIPELINE,
        provenance_artifact_ref="official pipeline, but operator picks the contests",
        provenance_artifact_exists=True,
        is_lcb_published_provenance=True,   # B1 passes
        is_operator_curation=True,          # B2 FAILS
        has_dated_problems=True, has_functional_subset=True,
        sha_pinnable=True, histogram_reproducible=True,
        claimed_frontier_date="2025-12-31", note="isolate B2")
    a = _assess(curated)
    assert a.a_admissible is True
    assert a.b1_authoritative_provenance is True
    assert a.b2_no_operator_curation is False
    assert a.construction_admissible is False
    assert "B2 fail" in a.reason


def test_realizable_construction_admissible_proposal():
    # A published, reproducible, EXISTING post-frontier artifact is realizable.
    real = ConstructionCandidateProposalV1(
        label="real LCB post-v6 revision",
        derivation_kind="lcb_dataset_revision",
        provenance_artifact_ref="livecodebench/code_generation_lite test7.jsonl "
                                "(real revision)",
        provenance_artifact_exists=True,
        is_lcb_published_provenance=True,
        is_operator_curation=False,
        has_dated_problems=True, has_functional_subset=True,
        sha_pinnable=True, histogram_reproducible=True,
        claimed_frontier_date="2026-03-31", note="synthetic real revision")
    a = _assess(real)
    assert a.construction_admissible is True
    assert a.newer_than_admitted is True
    assert a.artifact_exists is True
    assert a.realizable is True
    assert "REALIZABLE" in a.reason


# ----------------------------------------------------- construction-provenance surfaces

def test_eight_provenance_surfaces_none_hot():
    surfaces = W117_PROVENANCE_SNAPSHOT.provenance_surfaces
    assert len(surfaces) == 8
    assert all(s.has_post_frontier_lcb_artifact is False for s in surfaces)


def test_construct_real_snapshot_returns_no_candidate_with_missing_artifact():
    att = construct_upstream_derived_candidate_v1()
    assert att.constructed is False
    assert att.n_construction_admissible_new == 0
    # the pipeline template is admissible-in-principle (1); the raw assembly is not.
    assert att.n_construction_admissible_in_principle == 1
    assert att.derived_observation is None
    assert att.surfaces_with_post_frontier_artifact == ()
    assert "LCB-published" in att.missing_artifact
    assert "raw-contest hand-assembly" in att.missing_artifact


# ------------------------------------------------- sharpened disclosure matrix (§4b)

def test_disclosure_matrix_deepseek_sharpened_unknown_from_primary():
    by_id = {d.model_id: d for d in W117_DISCLOSURE_MATRIX}
    ds = by_id["deepseek-ai/deepseek-v4-pro"]
    assert ds.primary_status == DISCLOSURE_UNKNOWN
    assert "PDF" in ds.primary_source          # re-checked at the primary PDF
    assert "C2-exposed" in ds.certifiable_blocker


def test_disclosure_matrix_maverick_verbatim_august_2024():
    by_id = {d.model_id: d for d in W117_DISCLOSURE_MATRIX}
    mav = by_id["meta/llama-4-maverick-17b-128e-instruct"]
    assert mav.primary_status == DISCLOSURE_KNOWN
    assert "August 2024" in mav.primary_source  # verbatim from MODEL_CARD.md
    assert "C4" in mav.certifiable_blocker


def test_disclosure_delta_nothing_newly_disclosed_since_w116():
    s = disclosure_delta_since_w116_v1()
    assert s["any_newly_disclosed_since_w116"] is False
    assert s["newly_disclosed_since_w116"] == []
    assert s["any_usable_new_known_cutoff_target"] is False
    assert s["counts"].get(DISCLOSURE_UNKNOWN, 0) == 3


# ------------------------------------------------- real-snapshot verdict + invariant

def test_real_snapshot_is_no_certifiable_stronger_model():
    r = run_upstream_construction_v1()
    assert r.verdict == VERDICT_NONE
    assert r.target_model == ""
    assert r.n_surfaces_with_post_frontier_artifact == 0
    assert r.construction_attempt.constructed is False
    assert r.w118_fire_condition.fires_now is False
    assert r.w118_fire_condition.packaged_release_trigger_met is False
    assert r.w118_fire_condition.construction_provenance_trigger_met is False
    assert r.w118_fire_condition.cutoff_trigger_met is False
    assert r.cid()


def test_decision_cid_is_byte_identical_to_w114_w115_w116():
    # The reuse invariant: the wrapped certification decision must equal the W114
    # decide_certification_v1 decision (prefix 258b6ed7) — W117 wraps, never forks.
    r = run_upstream_construction_v1()
    d = decide_certification_v1()
    assert r.upstream_admission.frontier_certification.decision.cid() == d.cid()
    assert d.cid().startswith("258b6ed7")


def test_real_snapshot_supply_is_the_w116_snapshot():
    # W117 reuses the W116 packaged-admission/model/instrument/disclosure state
    # unchanged (so the decision re-derives byte-identically); the NEW W117 work is
    # the construction-provenance layer.
    assert W117_PROVENANCE_SNAPSHOT.supply_snapshot is W116_UPSTREAM_SNAPSHOT


# ------------------------------------------------------------ falsifiability

def test_realizable_construction_with_known_cutoff_model_certifies_and_fires():
    """The pipeline is not hard-wired to no-go: a synthetic snapshot with a REALIZABLE
    construction-admissible instrument (its LCB-published artifact exists) + a
    certifying supply snapshot (KNOWN-cutoff, not-settled stronger candidate on a
    post-2025 instrument) constructs the candidate, certifies the target, fires the
    W118 construction + packaged triggers, and is fires_now == True.

    The certifying candidate here is Maverick (the only KNOWN-cutoff registry entry),
    so this is the "Maverick certifiable on a GENUINELY NEW instrument" case: the
    packaged + construction triggers fire, but the cutoff trigger (specifically for a
    NON-Maverick stronger model gaining a primary-KNOWN cutoff) correctly stays False."""
    future_inst = LatestResistantInstrumentV1(
        release="release_v7_synthetic",
        jsonl_sha256="0" * 64,
        n_functional=60,
        functional_date_min="2026-01-01",
        functional_date_max="2026-06-30",
        functional_month_histogram={
            "2026-01": 15, "2026-02": 15, "2026-03": 15, "2026-06": 15},
        note="synthetic post-2025 instrument for falsifiability")
    cand = dataclasses.replace(
        STRONGER_MODEL_CANDIDATES[0], already_settled_on_instrument=False)
    future_frontier = FrontierSnapshotV1(
        verified_on="2026-07-01",
        source_note="synthetic future frontier snapshot",
        observed_releases=tuple(LIVECODEBENCH_KNOWN_RELEASES) + ("test7.jsonl",),
        instrument=future_inst,
        candidates=(cand,),
        model_disclosures=())  # empty => consistency vacuously ok
    new_obs = UpstreamInstrumentObservationV1(
        label="release_v7 (synthetic official)",
        source_kind=SOURCE_OFFICIAL_HF_DATASET,
        source_ref="hf://livecodebench/code_generation_lite test7.jsonl",
        release_id="release_v7",
        has_dated_problems=True, has_functional_subset=True,
        sha_pinnable=True, histogram_reproducible=True,
        admitted_to_loader=True, frontier_date="2026-06-30",
        note="synthetic admissible NEW instrument")
    supply = UpstreamSupplySnapshotV1(
        verified_on="2026-07-01",
        source_note="synthetic future upstream snapshot",
        hf_dataset="livecodebench/code_generation_lite",
        lite_tree_releases=tuple(LIVECODEBENCH_KNOWN_RELEASES) + ("release_v7",),
        loader_allowed_versions=tuple(LIVECODEBENCH_KNOWN_RELEASES) + (
            "release_v7", "release_latest"),
        release_latest_resolves_to="release_v7",
        full_dataset_frontier_date="2026-06-30",
        lite_last_modified="2026-07-01",
        observations=(new_obs,),
        frontier_snapshot=future_frontier)
    realizable_proposal = ConstructionCandidateProposalV1(
        label="real LCB post-v6 revision (exists)",
        derivation_kind="lcb_dataset_revision",
        provenance_artifact_ref="livecodebench/code_generation_lite test7.jsonl",
        provenance_artifact_exists=True,
        is_lcb_published_provenance=True,
        is_operator_curation=False,
        has_dated_problems=True, has_functional_subset=True,
        sha_pinnable=True, histogram_reproducible=True,
        claimed_frontier_date="2026-06-30", note="synthetic realizable")
    surf = dataclasses.replace(
        W117_PROVENANCE_SNAPSHOT.provenance_surfaces[0],
        has_post_frontier_lcb_artifact=True)
    snap = UpstreamProvenanceSnapshotV1(
        verified_on="2026-07-01",
        source_note="synthetic future provenance snapshot",
        provenance_surfaces=(surf,),
        proposals=(realizable_proposal,),
        supply_snapshot=supply)
    r = run_upstream_construction_v1(snapshot=snap)
    assert r.construction_attempt.constructed is True
    assert r.construction_attempt.n_construction_admissible_new == 1
    assert r.construction_attempt.derived_observation is not None
    assert r.n_surfaces_with_post_frontier_artifact == 1
    assert r.verdict == VERDICT_CERTIFIABLE
    assert r.target_model == cand.model_id
    assert r.w118_fire_condition.fires_now is True
    assert r.w118_fire_condition.construction_provenance_trigger_met is True
    assert r.w118_fire_condition.packaged_release_trigger_met is True
    # Maverick certifies => cutoff trigger (non-Maverick new-cutoff) stays False.
    assert r.w118_fire_condition.cutoff_trigger_met is False
