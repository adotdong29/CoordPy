"""W116 / COO-9 — tests for the durable upstream instrument-admission pipeline.

Covers ``coordpy.upstream_instrument_admission_v1``:

* the A1..A5 admissibility rule (release_v6 admissible-but-not-new; the full dataset
  + release_latest alias admissible-but-not-new; the 'planned v7' rumor REFUSED by
  A1 + A4; a synthetic post-2025 release_v7 IS an admissible NEW instrument);
* the four-way per-model disclosure-status matrix (Mistral-Small-4 CONFIRMED REAL +
  UNKNOWN-from-primary; Mistral-Small-3.2 KNOWN-but-sub-70B; no usable NEW
  KNOWN-cutoff target);
* the certifiable-slice candidate builder (reuses the W114 gate: Maverick C4-settled
  on release_v6; Qwen C1-blocked);
* the upstream-change detector (no change vs the encoded baseline; a synthetic newer
  snapshot flags every changed surface);
* the real-snapshot verdict == NO_CERTIFIABLE_STRONGER_MODEL with 0 admissible NEW
  instruments and W117 fires_now == False;
* the REUSE INVARIANT: the certification decision re-derives byte-identically to the
  W114/W115 decision (decision CID == decide_certification_v1().cid(), prefix
  258b6ed7) — the pipeline wraps, never forks, the gate;
* a FALSIFIABILITY test: a synthetic snapshot with an admissible NEW release_v7
  instrument (post-Apr-2025 functional problems) + a KNOWN-cutoff not-settled
  stronger candidate DOES certify, names the target, and fires W117.
"""
from __future__ import annotations

import dataclasses

from coordpy.upstream_instrument_admission_v1 import (
    DISCLOSURE_KNOWN,
    DISCLOSURE_UNKNOWN,
    SOURCE_OFFICIAL_HF_DATASET,
    SOURCE_RUMOR,
    VERDICT_CERTIFIABLE,
    VERDICT_NONE,
    W116_ADMISSIBILITY_RULE,
    W116_DISCLOSURE_MATRIX,
    W116_UPSTREAM_SNAPSHOT,
    UpstreamInstrumentObservationV1,
    UpstreamSupplySnapshotV1,
    assess_instrument_admissibility_v1,
    build_certifiable_slice_candidate_v1,
    detect_upstream_change_v1,
    disclosure_matrix_summary_v1,
    run_upstream_admission_v1,
)
from coordpy.frontier_certification_pipeline_v1 import (
    FrontierSnapshotV1,
    W115_FRONTIER_SNAPSHOT,
)
from coordpy.stronger_model_cutoff_certification_v1 import (
    STRONGER_MODEL_CANDIDATES,
    LatestResistantInstrumentV1,
    decide_certification_v1,
)
from coordpy.livecodebench_loader_v1 import LIVECODEBENCH_KNOWN_RELEASES


# --------------------------------------------------------- admissibility rule (§3)

def _admit(obs, *, latest_num=6, frontier="2025-04-05"):
    return assess_instrument_admissibility_v1(
        obs, admitted_latest_num=latest_num, admitted_frontier_date=frontier)


def test_release_v6_observation_is_admissible_but_not_new():
    by_label = {o.label: o for o in W116_UPSTREAM_SNAPSHOT.observations}
    v6 = by_label["release_v6 (admitted instrument)"]
    a = _admit(v6)
    assert a.admissible is True
    assert a.newer_than_admitted is False
    assert a.admissible_new_instrument is False
    assert "ADMISSIBLE_BUT_NOT_NEWER" in a.reason


def test_release_latest_alias_resolves_to_v6_not_new():
    by_label = {o.label: o for o in W116_UPSTREAM_SNAPSHOT.observations}
    alias = by_label["release_latest alias"]
    a = _admit(alias)
    assert a.admissible is True
    assert a.admissible_new_instrument is False


def test_full_dataset_admissible_but_not_newer():
    by_label = {o.label: o for o in W116_UPSTREAM_SNAPSHOT.observations}
    full = by_label["full code_generation dataset"]
    a = _admit(full)
    assert a.admissible is True
    assert a.admissible_new_instrument is False


def test_planned_v7_rumor_is_refused_by_A1_and_A4():
    by_label = {o.label: o for o in W116_UPSTREAM_SNAPSHOT.observations}
    rumor = by_label["planned release_v7 (search rumor)"]
    a = _admit(rumor)
    assert a.a1_authoritative is False   # non-primary rumor source
    assert a.a4_sha_pinnable is False    # no artifact / no SHA
    assert a.admissible is False
    assert a.admissible_new_instrument is False
    assert "NOT_ADMISSIBLE" in a.reason


def test_synthetic_official_release_v7_is_admissible_new_instrument():
    obs = UpstreamInstrumentObservationV1(
        label="release_v7 (synthetic official)",
        source_kind=SOURCE_OFFICIAL_HF_DATASET,
        source_ref="hf://livecodebench/code_generation_lite test7.jsonl",
        release_id="release_v7",
        has_dated_problems=True, has_functional_subset=True,
        sha_pinnable=True, histogram_reproducible=True,
        admitted_to_loader=False, frontier_date="2026-03-31",
        note="synthetic post-Apr-2025 release for the admissibility test")
    a = _admit(obs)
    assert a.admissible is True
    assert a.newer_than_admitted is True
    assert a.admissible_new_instrument is True
    assert "ADMISSIBLE_NEW_INSTRUMENT" in a.reason


# ---------------------------------------------- per-model disclosure matrix (§4b)

def test_disclosure_matrix_mistral_small_4_unknown_from_primary():
    by_id = {d.model_id: d for d in W116_DISCLOSURE_MATRIX}
    m4 = by_id["mistralai/mistral-small-4-119b-2603"]
    assert m4.primary_status == DISCLOSURE_UNKNOWN
    assert m4.stronger_than_70b is True
    assert "CONFIRMED REAL" in m4.note
    # the only cutoff figure is a non-primary aggregator that is itself C2-exposed
    assert "2025-06" in m4.aggregator_signal


def test_disclosure_matrix_mistral_3_2_known_but_sub_70b():
    by_id = {d.model_id: d for d in W116_DISCLOSURE_MATRIX}
    m32 = by_id["mistralai/mistral-small-3.2-24b-instruct-2506"]
    assert m32.primary_status == DISCLOSURE_KNOWN
    assert m32.stronger_than_70b is False
    assert "C3" in m32.certifiable_blocker


def test_disclosure_summary_no_usable_new_known_cutoff_target():
    s = disclosure_matrix_summary_v1()
    # Maverick is KNOWN but C4-settled => not a usable NEW target; all reachable
    # stronger-than-Maverick frontier models are UNKNOWN from primary.
    assert s["any_usable_new_known_cutoff_target"] is False
    assert s["usable_new_known_cutoff_targets"] == []
    assert s["counts"].get(DISCLOSURE_UNKNOWN, 0) == 3


# --------------------------------------------------- certifiable-slice builder (§5)

def test_slice_builder_maverick_c4_settled_on_release_v6():
    mav = next(c for c in STRONGER_MODEL_CANDIDATES
               if "maverick" in c.model_id)
    cand = build_certifiable_slice_candidate_v1(mav)
    assert cand.n_functional_resistant == 63   # all 63 functional > Aug-2024
    assert cand.meets_min_slice is True
    assert cand.certifiable is False           # C4 settled (W113)
    assert "C4" in cand.reason


def test_slice_builder_qwen_c1_blocked():
    qwen = next(c for c in STRONGER_MODEL_CANDIDATES
                if "qwen" in c.model_id)
    cand = build_certifiable_slice_candidate_v1(qwen)
    assert cand.cutoff_confidence == "UNKNOWN"
    assert cand.certifiable is False
    assert "C1" in cand.reason


# --------------------------------------------------------- upstream-change detector

def test_no_upstream_change_vs_self_baseline():
    ch = detect_upstream_change_v1(
        W116_UPSTREAM_SNAPSHOT, baseline=W116_UPSTREAM_SNAPSHOT)
    assert ch.any_change is False
    assert ch.new_numbered_release is False
    assert ch.release_latest_repointed is False
    assert ch.full_dataset_frontier_advanced is False
    assert ch.new_admissible_instrument is False


def test_synthetic_newer_snapshot_flags_every_surface():
    new_obs = UpstreamInstrumentObservationV1(
        label="release_v7 (synthetic official)",
        source_kind=SOURCE_OFFICIAL_HF_DATASET,
        source_ref="hf://livecodebench/code_generation_lite test7.jsonl",
        release_id="release_v7",
        has_dated_problems=True, has_functional_subset=True,
        sha_pinnable=True, histogram_reproducible=True,
        admitted_to_loader=False, frontier_date="2026-03-31",
        note="synthetic")
    newer = dataclasses.replace(
        W116_UPSTREAM_SNAPSHOT,
        verified_on="2026-09-01",
        lite_tree_releases=tuple(LIVECODEBENCH_KNOWN_RELEASES) + ("release_v7",),
        loader_allowed_versions=tuple(
            W116_UPSTREAM_SNAPSHOT.loader_allowed_versions) + ("release_v7",),
        release_latest_resolves_to="release_v7",
        full_dataset_frontier_date="2026-03-31",
        lite_last_modified="2026-08-15",
        observations=W116_UPSTREAM_SNAPSHOT.observations + (new_obs,))
    ch = detect_upstream_change_v1(newer, baseline=W116_UPSTREAM_SNAPSHOT)
    assert ch.any_change is True
    assert ch.new_numbered_release is True
    assert ch.release_latest_repointed is True
    assert ch.full_dataset_frontier_advanced is True
    assert ch.lite_last_modified_changed is True
    assert ch.loader_versions_extended is True
    assert ch.new_admissible_instrument is True


# ------------------------------------------------- real-snapshot verdict + invariant

def test_real_snapshot_is_no_certifiable_stronger_model():
    r = run_upstream_admission_v1()
    assert r.verdict == VERDICT_NONE
    assert r.target_model == ""
    assert r.n_admissible_new_instruments == 0
    assert r.upstream_change.any_change is False
    assert r.w117_fire_condition.fires_now is False
    assert r.w117_fire_condition.instrument_trigger_met is False
    assert r.w117_fire_condition.cutoff_trigger_met is False
    assert r.frontier_certification.disclosure_consistency_ok is True
    assert r.cid()  # stable content id present


def test_decision_cid_is_byte_identical_to_w114_w115():
    # The reuse invariant: the wrapped certification decision must equal the W114
    # decide_certification_v1 decision (prefix 258b6ed7) on the default snapshot.
    r = run_upstream_admission_v1()
    d = decide_certification_v1()
    assert r.frontier_certification.decision.cid() == d.cid()
    assert d.cid().startswith("258b6ed7")


def test_real_snapshot_frontier_snapshot_is_the_w115_snapshot():
    # W116 reuses the W115 model/instrument/disclosure state unchanged (so the
    # decision re-derives byte-identically); the NEW W116 work is the upstream layer.
    assert W116_UPSTREAM_SNAPSHOT.frontier_snapshot is W115_FRONTIER_SNAPSHOT


# ------------------------------------------------------------ falsifiability

def test_admissible_new_instrument_with_known_cutoff_model_certifies_and_fires():
    """The pipeline is not hard-wired to no-go: a synthetic snapshot with an
    admissible NEW release_v7 (all problems post-Apr-2025) + a KNOWN-cutoff,
    reachable, not-settled stronger candidate certifies, names the target, fires the
    W117 instrument trigger, and the upstream-change detector flags the new supply."""
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
    snap = UpstreamSupplySnapshotV1(
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
    r = run_upstream_admission_v1(snapshot=snap)
    assert r.verdict == VERDICT_CERTIFIABLE
    assert r.target_model == cand.model_id
    assert r.n_admissible_new_instruments == 1
    assert r.w117_fire_condition.fires_now is True
    assert r.w117_fire_condition.instrument_trigger_met is True
