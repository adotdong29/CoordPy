"""W115 / COO-9 — tests for the durable future-fire certification pipeline.

Covers ``coordpy.frontier_certification_pipeline_v1``:

* the release-version parser + latest-official-release detector (no newer release
  on the real W115 snapshot; a synthetic release_v7 IS flagged);
* the generalised frontier-date summary + threshold table (max KNOWN cutoff month
  admitting a >=30 resistant slice == 2025-01 on release_v6);
* the per-model go/no-go matrix on the LIVE W115 snapshot ==
  ``NO_CERTIFIABLE_STRONGER_MODEL`` (the external frontier did not move since W114);
* the disclosure-consistency guard (live disclosures match the encoded registry;
  a divergent disclosure is flagged);
* a FALSIFIABILITY test: a synthetic snapshot with a newer admitted instrument
  (release_v7, post-Apr-2025 functional problems) + a KNOWN-cutoff not-settled
  stronger candidate DOES certify, names the target, and fires W116 (the pipeline
  is not hard-wired to no-go).
"""
from __future__ import annotations

import dataclasses

from coordpy.frontier_certification_pipeline_v1 import (
    FrontierSnapshotV1,
    ModelDisclosureV1,
    VERDICT_CERTIFIABLE,
    VERDICT_NONE,
    W115_FRONTIER_SNAPSHOT,
    check_disclosure_consistency_v1,
    detect_latest_release_v1,
    frontier_date_summary_v1,
    release_version_num_v1,
    run_frontier_certification_v1,
)
from coordpy.stronger_model_cutoff_certification_v1 import (
    LATEST_RESISTANT_INSTRUMENT,
    STRONGER_MODEL_CANDIDATES,
    LatestResistantInstrumentV1,
)
from coordpy.livecodebench_loader_v1 import LIVECODEBENCH_KNOWN_RELEASES
from coordpy.livecodebench_resistant_slice_v1 import MIN_RESISTANT_SLICE


# ------------------------------------------------------ release version parser

def test_release_version_num_parses_all_forms():
    assert release_version_num_v1("release_v6") == 6
    assert release_version_num_v1("test6.jsonl") == 6
    assert release_version_num_v1("release_v1") == 1
    assert release_version_num_v1("test.jsonl") == 1  # first release, no digit
    assert release_version_num_v1("test10.jsonl") == 10
    assert release_version_num_v1("") == 0
    assert release_version_num_v1("garbage") == 0


# --------------------------------------------------- latest-release detector

def test_detector_no_newer_release_on_real_snapshot():
    # The live HF file tree (recorded in the snapshot) tops out at release_v6,
    # exactly the loader's admitted latest => no newer release to admit.
    rd = detect_latest_release_v1(W115_FRONTIER_SNAPSHOT.observed_releases)
    assert rd.latest_admitted_num == 6
    assert rd.latest_observed_num == 6
    assert rd.newer_release_available is False
    assert rd.observed_not_admitted == ()
    assert rd.latest_admitted == "release_v6"


def test_detector_flags_synthetic_release_v7():
    observed = tuple(LIVECODEBENCH_KNOWN_RELEASES) + ("test7.jsonl",)
    rd = detect_latest_release_v1(observed)
    assert rd.latest_observed_num == 7
    assert rd.latest_admitted_num == 6
    assert rd.newer_release_available is True
    assert "test7.jsonl" in rd.observed_not_admitted


# ------------------------------------------------------- frontier-date summary

def test_frontier_summary_threshold_table_and_max_cutoff_month():
    fs = frontier_date_summary_v1(LATEST_RESISTANT_INSTRUMENT)
    assert fs.release == "release_v6"
    assert fs.n_functional == 63
    assert fs.functional_date_max == "2025-04-05"
    # threshold table: a cutoff in 2025-01 keeps Feb+Mar+Apr = 49; 2025-02 -> 29.
    assert fs.threshold_table["2025-01"] == 49
    assert fs.threshold_table["2025-02"] == 29
    assert fs.threshold_table["2025-04"] == 0
    assert fs.threshold_table["<before-all>"] == 63
    # the binding fact: max KNOWN cutoff month admitting >= 30 resistant == 2025-01.
    assert fs.max_cutoff_month_for_min_slice == "2025-01"
    assert fs.min_slice == MIN_RESISTANT_SLICE


# ----------------------------------------------- go/no-go matrix (real snapshot)

def test_real_snapshot_is_no_certifiable_stronger_model():
    r = run_frontier_certification_v1()
    assert r.verdict == VERDICT_NONE
    assert r.target_model == ""
    assert r.disclosure_consistency_ok is True
    assert r.w116_fire_condition.fires_now is False
    assert r.w116_fire_condition.instrument_trigger_met is False
    assert r.w116_fire_condition.cutoff_trigger_met is False
    assert r.cid()  # stable content id present
    # Maverick certifiable-but-settled is carried from the W114 decision.
    assert r.decision.maverick_certifiable_but_settled is True


def test_real_snapshot_per_model_blockers():
    r = run_frontier_certification_v1()
    by_id = {m.model_id: m for m in r.decision.per_model}
    # Maverick: C1∧C2∧C3 true, C4 false (settled).
    mav = by_id["meta/llama-4-maverick-17b-128e-instruct"]
    assert mav.c1_cutoff_known and mav.c2_enough_resistant
    assert mav.c4_not_already_settled is False
    # Qwen3-Coder + DeepSeek + Mistral: C1 false (UNKNOWN cutoff).
    assert by_id["qwen/qwen3-coder-480b-a35b-instruct"].c1_cutoff_known is False
    assert by_id["deepseek-ai/deepseek-v4-pro"].c1_cutoff_known is False
    assert by_id["mistralai/mistral-small-4-119b-2603"].c1_cutoff_known is False


# --------------------------------------------------- disclosure-consistency guard

def test_disclosure_consistency_all_consistent_on_real_snapshot():
    cons = check_disclosure_consistency_v1(W115_FRONTIER_SNAPSHOT)
    assert len(cons) == 4
    for c in cons:
        assert c.consistent is True, c.model_id


def test_disclosure_divergence_is_flagged():
    # Simulate a future world where DeepSeek-V4 disclosed a KNOWN cutoff while the
    # encoded registry still says UNKNOWN -> the guard must flag the divergence
    # (the W116 signal: update the registry before certifying).
    diverged = []
    for d in W115_FRONTIER_SNAPSHOT.model_disclosures:
        if d.model_id == "deepseek-ai/deepseek-v4-pro":
            diverged.append(dataclasses.replace(d, confidence="KNOWN"))
        else:
            diverged.append(d)
    snap = dataclasses.replace(
        W115_FRONTIER_SNAPSHOT, model_disclosures=tuple(diverged))
    cons = {c.model_id: c for c in check_disclosure_consistency_v1(snap)}
    assert cons["deepseek-ai/deepseek-v4-pro"].consistent is False
    assert "DIVERGENCE" in cons["deepseek-ai/deepseek-v4-pro"].note


# ------------------------------------------------------------ falsifiability

def test_newer_release_with_known_cutoff_model_DOES_certify_and_fires():
    """The pipeline is not hard-wired to no-go: a synthetic snapshot with a newer
    admitted instrument (release_v7, all problems post-Apr-2025) + a KNOWN-cutoff,
    reachable, not-settled stronger candidate certifies, names the target, and
    fires the W116 instrument trigger."""
    future_inst = LatestResistantInstrumentV1(
        release="release_v7_synthetic",
        jsonl_sha256="0" * 64,
        n_functional=60,
        functional_date_min="2026-01-01",
        functional_date_max="2026-06-30",
        functional_month_histogram={
            "2026-01": 15, "2026-02": 15, "2026-03": 15, "2026-06": 15},
        note="synthetic post-2025 instrument for falsifiability")
    # Reuse Maverick's KNOWN cutoff (2024-08-31) but mark NOT settled (a NEW
    # instrument it never ran) so C4 passes — exactly the W114 falsifiability shape.
    cand = dataclasses.replace(
        STRONGER_MODEL_CANDIDATES[0], already_settled_on_instrument=False)
    snap = FrontierSnapshotV1(
        verified_on="2026-07-01",
        source_note="synthetic future snapshot",
        observed_releases=tuple(LIVECODEBENCH_KNOWN_RELEASES) + ("test7.jsonl",),
        instrument=future_inst,
        candidates=(cand,),
        model_disclosures=())  # empty => consistency vacuously ok
    r = run_frontier_certification_v1(snapshot=snap)
    assert r.verdict == VERDICT_CERTIFIABLE
    assert r.target_model == cand.model_id
    assert r.w116_fire_condition.fires_now is True
    # release_v7 observed + not admitted => the instrument trigger is met.
    assert r.release_detection.newer_release_available is True
    assert r.w116_fire_condition.instrument_trigger_met is True
    # frontier summary on the synthetic instrument: all 60 are post-Aug-2024.
    assert r.frontier_summary.n_functional == 60


def test_pipeline_reuses_w114_decision_no_duplication():
    # The pipeline's verdict must equal the W114 decide_certification_v1 verdict on
    # the same (default) instrument + candidates — proving it wraps, not forks.
    from coordpy.stronger_model_cutoff_certification_v1 import (
        decide_certification_v1,
    )
    r = run_frontier_certification_v1()
    d = decide_certification_v1()
    assert r.verdict == d.verdict
    assert r.target_model == d.target_model
    assert r.decision.cid() == d.cid()
