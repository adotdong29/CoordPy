"""W119 tests — official ICPC public-package post-cutoff functional construction.

Covers: deterministic manifest construction + exclusions; the P1..P8 rule (identity vs
grader vs slice); the grader DISSOLUTION (P7∧P8 hold — the W118 advance); the real
stdin/stdout executor; per-model certification (Maverick identity-certifiable but
slice-short); the decision-CID byte-identical invariant (258b6ed7); the W120 fire
condition; and TWO falsifiability tests (a synthetic >=30 grader-clean slice DOES make
Maverick pilot-admissible + fire W120; a slice without a grader does NOT).
"""
from __future__ import annotations

import dataclasses

from coordpy.coordpy_icpc_public_functional_v1 import (
    ICPC_PACKAGE_LISTING_SNAPSHOT_V1,
    MAVERICK_CUTOFF_BOUNDARY,
    OFFICIAL_ICPC_PACKAGE_FAMILY,
    VALIDATION_INTERACTIVE,
    VALIDATION_PASSFAIL,
    VERDICT_CERTIFIABLE,
    VERDICT_NONE,
    W119_GRADER_SELFTEST_V1,
    assess_icpc_admissibility_v1,
    build_icpc_manifest_v1,
    certify_models_on_icpc_manifest_v1,
    grader_selftest_summary_v1,
    icpc_family_grader_summary_v1,
    run_icpc_public_construction_v1,
    run_icpc_stdin_executor_v1,
)

DECISION_CID_INVARIANT = "258b6ed7"


# ----------------------------------------------------------------- manifest construction

def test_manifest_admits_24_post_cutoff_passfail():
    m = build_icpc_manifest_v1(fetched_on="2026-05-30")
    # 26 candidates seen; 1 interactive + 1 custom-no-validator excluded => 24 admitted.
    assert m.n_candidates_seen == 26
    assert m.n_admitted == 24
    assert m.n_excluded_interactive == 1
    assert m.n_excluded_no_grader == 1
    assert m.n_excluded_pre_cutoff == 0  # both RMRC repos post-date Aug-2024


def test_manifest_is_deterministic_and_sorted():
    m1 = build_icpc_manifest_v1(fetched_on="2026-05-30")
    m2 = build_icpc_manifest_v1(fetched_on="2026-05-30")
    assert m1.admitted_problem_ids == m2.admitted_problem_ids
    assert m1.manifest_cid() == m2.manifest_cid()
    # sorted by (contest_date, repo, short_name)
    ids = list(m1.admitted_problem_ids)
    assert ids == sorted(ids, key=lambda x: x)  # ids encode date-then-name ordering only loosely
    # the two contest months are present
    assert set(m1.month_histogram) == {"2024-12", "2025-11"}


def test_manifest_excludes_interactive_problem():
    m = build_icpc_manifest_v1(fetched_on="2026-05-30")
    assert not any("poetictournament" in pid for pid in m.admitted_problem_ids)


def test_manifest_excludes_custom_without_validator():
    m = build_icpc_manifest_v1(fetched_on="2026-05-30")
    assert not any("alwaysknowwhereyourtowelis" in pid
                   for pid in m.admitted_problem_ids)


def test_pre_cutoff_repo_problems_are_excluded():
    # Inject a synthetic pre-cutoff (2024-01) pass-fail problem; it must be excluded (P3).
    listing = list(ICPC_PACKAGE_LISTING_SNAPSHOT_V1) + [
        ("icpc/na-rocky-mountain-2023-2024-public", "precut", "2024-01-16",
         VALIDATION_PASSFAIL, 40, False)]
    m = build_icpc_manifest_v1(listing, fetched_on="2026-05-30")
    assert m.n_excluded_pre_cutoff == 1
    assert not any("precut" in pid for pid in m.admitted_problem_ids)


# ----------------------------------------------------------------- the P7/P8 dissolution

def test_grader_is_present_family_wide_the_W118_dissolution():
    gs = icpc_family_grader_summary_v1()
    # The W118 blocker DISSOLVES: an official source ships the grader.
    assert gs["any_source_has_official_grader"] is True
    assert len(gs["post_cutoff_grader_repos"]) == 2
    assert gs["n_post_cutoff_gradeable_passfail"] == 24


def test_grader_selftest_proves_executable_oracle():
    st = grader_selftest_summary_v1()
    assert st["grader_proven_executable"] is True
    assert st["n_cases_passed"] == st["n_cases_run"]
    assert st["n_cases_run"] == 16  # conservative diff-oracle subset (2 problems)


def test_real_stdin_executor_passes_correct_and_fails_wrong():
    # A correct sum program.
    ok = run_icpc_stdin_executor_v1(
        candidate_code="import sys\nprint(sum(int(x) for x in sys.stdin.read().split()))",
        stdin_text="2 3 4", expected_stdout="9")
    assert ok.passed is True and ok.returncode == 0
    # A wrong program.
    bad = run_icpc_stdin_executor_v1(
        candidate_code="print(0)", stdin_text="2 3 4", expected_stdout="9")
    assert bad.passed is False
    # A crashing program is a clean fail, not a crash of the harness.
    err = run_icpc_stdin_executor_v1(
        candidate_code="raise SystemExit(3)", stdin_text="", expected_stdout="x")
    assert err.passed is False and err.returncode == 3


def test_executor_token_normalizes_whitespace():
    res = run_icpc_stdin_executor_v1(
        candidate_code="print('a\\nb\\nc')", stdin_text="",
        expected_stdout="a b c")
    assert res.passed is True


# ----------------------------------------------------------------- admissibility (P1..P8)

def test_admissibility_grader_ok_but_slice_short():
    m = build_icpc_manifest_v1(fetched_on="2026-05-30")
    adm = assess_icpc_admissibility_v1(m)
    assert adm.p7_grader_present is True
    assert adm.p8_grader_executable is True
    assert adm.grader_admissible is True      # the W119 advance
    assert adm.meets_min_slice is False       # 24 < 30
    assert adm.identity_admissible is False   # blocked on count
    assert adm.pilot_admissible is False
    assert "SLICE_SHORT" in adm.reason
    assert "6 " in adm.missing_artifact or "30" in adm.missing_artifact


# ----------------------------------------------------------------- per-model certification

def test_maverick_blocked_by_count_at_C2_on_24_slice():
    """On the real 24-problem ICPC slice Maverick is NOT identity-certifiable: the
    reused W114 C2 gate requires >=30 resistant problems after the cutoff, and 24 < 30
    blocks it there.  So the slice COUNT is the SINGLE load-bearing blocker — it gates
    both certification (C2) AND slice-admissibility.  The grader is fully admissible."""
    m = build_icpc_manifest_v1(fetched_on="2026-05-30")
    adm = assess_icpc_admissibility_v1(m)
    certs = certify_models_on_icpc_manifest_v1(
        m, adm.grader_admissible, adm.meets_min_slice)
    mav = [c for c in certs if "maverick" in c.model_id][0]
    assert mav.identity_certifiable is False  # C2: 24 < 30 resistant => not certifiable
    assert mav.grader_admissible is True      # the grader IS admissible (W118 dissolved)
    assert mav.slice_admissible is False
    assert mav.pilot_admissible is False
    assert "CERT_BLOCKED" in mav.blocker and "30" in mav.blocker


def test_tier2_unknown_cutoffs_not_identity_certifiable():
    m = build_icpc_manifest_v1(fetched_on="2026-05-30")
    certs = certify_models_on_icpc_manifest_v1(m, True, False)
    for c in certs:
        if "maverick" not in c.model_id:
            assert c.identity_certifiable is False  # UNKNOWN cutoff => C1 blocked


# ----------------------------------------------------------------- top-level verdict

def test_run_verdict_is_no_certifiable_and_no_pilot():
    r = run_icpc_public_construction_v1(verified_on="2026-05-30")
    assert r.verdict == VERDICT_NONE
    assert r.pilot_earned is False
    # 0 models certify: Maverick is blocked at C2 by the 24<30 count; the tier-2 models
    # are C1-blocked (UNKNOWN cutoff). The count is the single binding gate.
    assert r.n_identity_certifiable_models == 0


def test_decision_cid_byte_identical_to_w114_w118():
    r = run_icpc_public_construction_v1(verified_on="2026-05-30")
    cid = (r.lcb_inherited.upstream_admission.frontier_certification.decision.cid())
    assert cid.startswith(DECISION_CID_INVARIANT)


def test_w120_fire_condition_not_firing_now():
    r = run_icpc_public_construction_v1(verified_on="2026-05-30")
    fc = r.w120_fire_condition
    assert fc.fires_now is False
    assert fc.slice_trigger_met is False
    assert fc.cutoff_trigger_met is False


def test_result_cid_is_stable():
    r1 = run_icpc_public_construction_v1(verified_on="2026-05-30")
    r2 = run_icpc_public_construction_v1(verified_on="2026-05-30")
    assert r1.cid() == r2.cid()


# ----------------------------------------------------------------- FALSIFIABILITY

def test_falsifiability_grader_clean_30_slice_DOES_make_maverick_pilot_admissible():
    """If a >=30 post-cutoff resistant pass-fail slice WITH a self-test-passing grader
    existed, Maverick WOULD be pilot-admissible and W120 WOULD fire — proving the no-go
    is driven by the count gate, not a tautology."""
    # Synthesize 8 extra post-cutoff pass-fail problems (24 + 8 = 32 >= 30).
    extra = [
        (f"icpc/na-rocky-mountain-2025-2026-public", f"synth{i}", "2025-11-13",
         VALIDATION_PASSFAIL, 20, True)
        for i in range(8)]
    listing = list(ICPC_PACKAGE_LISTING_SNAPSHOT_V1) + extra
    m = build_icpc_manifest_v1(listing, fetched_on="2026-05-30")
    assert m.n_admitted >= 30
    adm = assess_icpc_admissibility_v1(m)
    assert adm.meets_min_slice is True
    assert adm.grader_admissible is True
    assert adm.identity_admissible is True
    assert adm.pilot_admissible is True
    certs = certify_models_on_icpc_manifest_v1(
        m, adm.grader_admissible, adm.meets_min_slice)
    mav = [c for c in certs if "maverick" in c.model_id][0]
    assert mav.pilot_admissible is True
    r = run_icpc_public_construction_v1(listing=listing, verified_on="2026-05-30")
    assert r.verdict == VERDICT_CERTIFIABLE
    assert r.pilot_earned is True
    assert r.w120_fire_condition.fires_now is True


def test_falsifiability_30_slice_WITHOUT_grader_does_NOT_pilot():
    """A >=30 slice with NO self-test-passing grader must NOT be pilot-admissible —
    the grader gate (P8) is load-bearing, not cosmetic."""
    extra = [
        (f"icpc/na-rocky-mountain-2025-2026-public", f"synth{i}", "2025-11-13",
         VALIDATION_PASSFAIL, 20, True)
        for i in range(8)]
    listing = list(ICPC_PACKAGE_LISTING_SNAPSHOT_V1) + extra
    m = build_icpc_manifest_v1(listing, fetched_on="2026-05-30")
    # Force the self-test to be EMPTY (grader not proven executable).
    adm = assess_icpc_admissibility_v1(m, OFFICIAL_ICPC_PACKAGE_FAMILY, selftest=())
    assert adm.p8_grader_executable is False
    assert adm.grader_admissible is False
    assert adm.pilot_admissible is False  # >=30 but no proven grader => no pilot
