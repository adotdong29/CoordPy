"""W118 tests — CoordPy-OWNED post-v6 functional-instrument construction.

Deterministic + offline: a small SYNTHETIC Codeforces payload (no network) exercises
the pure constructor, the O1..O7 admissibility split (identity tier vs grader tier),
the reused C1..C4 certification gate on the manifest, the family-wide grader blocker,
the LCB-inherited decision-CID byte-identical invariant (258b6ed7), and a
FALSIFIABILITY test (a synthetic source WITH an official grader + a KNOWN-cutoff model
DOES become pilot-admissible and fires W119).
"""
from __future__ import annotations

import calendar
import dataclasses

import pytest

from coordpy.coordpy_frontier_functional_v1 import (
    COORDPY_FRONTIER_FUNCTIONAL_V1,
    FRONTIER_DATE,
    OFFICIAL_SOURCE_FAMILY,
    VERDICT_CERTIFIABLE,
    VERDICT_NONE,
    OfficialSourceV1,
    SOURCE_CODEFORCES_API,
    assess_frontier_functional_admissibility_v1,
    build_frontier_manifest_from_codeforces_v1,
    certify_models_on_manifest_v1,
    disclosure_delta_since_w117_v1,
    run_frontier_functional_construction_v1,
    source_family_grader_summary_v1,
    _epoch_to_utc_day,
)


def _epoch(y: int, mo: int, d: int, h: int = 12) -> int:
    """Deterministic UTC epoch seconds (calendar.timegm — no wall clock)."""
    return calendar.timegm((y, mo, d, h, 0, 0, 0, 0, 0))


def _synthetic_payloads(*, n_admit_a: int = 20, n_admit_b: int = 18):
    """A synthetic Codeforces payload exercising every inclusion/exclusion branch.

    * contest 100 (2025-03-01, FINISHED): 2 PROGRAMMING => pre-frontier (excluded)
    * contest 200 (2025-04-05, FINISHED): 2 PROGRAMMING => ON the frontier day
      (excluded by the STRICT day boundary)
    * contest 300 (2025-05-10, FINISHED): ``n_admit_a`` PROGRAMMING + 1 QUESTION =>
      admitted (+ the QUESTION excluded not_programming)
    * contest 400 (2025-06-15, FINISHED): ``n_admit_b`` PROGRAMMING => admitted
    * contest 500 (2026-01-01, BEFORE): 3 PROGRAMMING => not finished (excluded)
    """
    contest_list = {"status": "OK", "result": [
        {"id": 100, "phase": "FINISHED", "startTimeSeconds": _epoch(2025, 3, 1)},
        {"id": 200, "phase": "FINISHED", "startTimeSeconds": _epoch(2025, 4, 5)},
        {"id": 300, "phase": "FINISHED", "startTimeSeconds": _epoch(2025, 5, 10)},
        {"id": 400, "phase": "FINISHED", "startTimeSeconds": _epoch(2025, 6, 15)},
        {"id": 500, "phase": "BEFORE", "startTimeSeconds": _epoch(2026, 1, 1)},
    ]}
    probs = []
    for i in range(2):
        probs.append({"contestId": 100, "index": chr(65 + i),
                      "name": f"pre{i}", "type": "PROGRAMMING"})
    for i in range(2):
        probs.append({"contestId": 200, "index": chr(65 + i),
                      "name": f"onfrontier{i}", "type": "PROGRAMMING"})
    for i in range(n_admit_a):
        probs.append({"contestId": 300, "index": chr(65 + i),
                      "name": f"a{i}", "type": "PROGRAMMING"})
    probs.append({"contestId": 300, "index": "Q",
                  "name": "question", "type": "QUESTION"})
    for i in range(n_admit_b):
        probs.append({"contestId": 400, "index": chr(65 + i),
                      "name": f"b{i}", "type": "PROGRAMMING"})
    for i in range(3):
        probs.append({"contestId": 500, "index": chr(65 + i),
                      "name": f"future{i}", "type": "PROGRAMMING"})
    problemset = {"status": "OK", "result": {"problems": probs}}
    return contest_list, problemset


# --------------------------------------------------------------- epoch -> day

def test_epoch_to_utc_day_known_points():
    assert _epoch_to_utc_day(0) == "1970-01-01"
    assert _epoch_to_utc_day(_epoch(2025, 4, 5, 0)) == "2025-04-05"
    assert _epoch_to_utc_day(_epoch(2024, 2, 29, 23)) == "2024-02-29"   # leap day
    assert _epoch_to_utc_day(_epoch(2026, 5, 30, 6)) == "2026-05-30"


# --------------------------------------------------------------- manifest

def test_manifest_admits_only_post_frontier_finished_programming():
    cl, ps = _synthetic_payloads(n_admit_a=20, n_admit_b=18)
    m = build_frontier_manifest_from_codeforces_v1(
        cl, ps, fetched_on="2026-05-30")
    assert m.instrument_id == COORDPY_FRONTIER_FUNCTIONAL_V1
    assert m.n_candidates_seen == 2 + 2 + 21 + 18 + 3   # all problems seen
    assert m.n_admitted == 38                            # 20 + 18
    assert m.n_excluded_not_programming == 1             # the QUESTION
    assert m.n_excluded_not_finished == 3                # contest 500 BEFORE
    assert m.n_excluded_not_after_frontier == 2 + 2      # pre + on-frontier-day
    assert m.date_min == "2025-05-10"
    assert m.date_max == "2025-06-15"
    assert m.n_contests == 2
    assert m.month_histogram == {"2025-05": 20, "2025-06": 18}


def test_strict_frontier_day_boundary_excludes_frontier_day():
    """A problem dated exactly on the frontier 2025-04-05 is EXCLUDED (strict >)."""
    cl, ps = _synthetic_payloads()
    m = build_frontier_manifest_from_codeforces_v1(
        cl, ps, fetched_on="2026-05-30")
    assert all(pid not in m.admitted_problem_ids
               for pid in ("cf_200_A", "cf_200_B"))
    assert m.date_min > FRONTIER_DATE


def test_manifest_is_deterministic_same_bytes_same_cid():
    cl, ps = _synthetic_payloads()
    m1 = build_frontier_manifest_from_codeforces_v1(cl, ps, fetched_on="x")
    m2 = build_frontier_manifest_from_codeforces_v1(cl, ps, fetched_on="y")
    # fetched_on differs but the ADMITTED SET (what manifest_cid addresses) is identical
    assert m1.manifest_cid() == m2.manifest_cid()
    assert m1.admitted_problem_ids == m2.admitted_problem_ids


def test_manifest_admitted_ids_sorted_deterministically():
    cl, ps = _synthetic_payloads(n_admit_a=20, n_admit_b=18)
    m = build_frontier_manifest_from_codeforces_v1(cl, ps, fetched_on="x")
    ids = list(m.admitted_problem_ids)
    # Sorted by (contest_date, contest_id, index): contest 300 (May) precedes 400 (June).
    assert ids[:20] == [f"cf_300_{chr(65 + i)}" for i in range(20)]
    assert ids[20:] == [f"cf_400_{chr(65 + i)}" for i in range(18)]


# --------------------------------------------------------------- O1..O7 admissibility

def test_identity_admissible_but_grader_absent_on_real_family():
    cl, ps = _synthetic_payloads()
    m = build_frontier_manifest_from_codeforces_v1(cl, ps, fetched_on="2026-05-30")
    cf = OFFICIAL_SOURCE_FAMILY[0]
    assert cf.source_kind == SOURCE_CODEFORCES_API
    adm = assess_frontier_functional_admissibility_v1(m, cf)
    assert adm.o1_official_source and adm.o2_dated and adm.o3_post_v6
    assert adm.o4_functional and adm.o5_deterministic_no_curation
    assert adm.o6_machine_manifest
    assert adm.o7_official_grader is False
    assert adm.identity_admissible is True
    assert adm.grader_admissible is False
    assert adm.pilot_admissible is False
    assert "EXECUTABLE TEST SUITE" in adm.missing_artifact


def test_below_min_slice_is_not_identity_admissible():
    cl, ps = _synthetic_payloads(n_admit_a=5, n_admit_b=4)   # 9 < 30
    m = build_frontier_manifest_from_codeforces_v1(cl, ps, fetched_on="x")
    assert m.n_admitted == 9
    adm = assess_frontier_functional_admissibility_v1(m, OFFICIAL_SOURCE_FAMILY[0])
    assert adm.meets_min_slice is False
    assert adm.identity_admissible is False
    assert adm.pilot_admissible is False


def test_source_family_grader_summary_no_official_grader():
    s = source_family_grader_summary_v1()
    assert s["any_source_has_official_grader"] is False
    assert s["any_source_has_clean_identity_api"] is True
    assert s["n_sources"] == 3


# --------------------------------------------------- per-model certification on manifest

def test_maverick_identity_certifiable_but_grader_blocked():
    cl, ps = _synthetic_payloads()
    m = build_frontier_manifest_from_codeforces_v1(cl, ps, fetched_on="x")
    # grader absent (the live family truth)
    per = certify_models_on_manifest_v1(m, grader_admissible=False)
    mav = next(x for x in per
               if x.model_id == "meta/llama-4-maverick-17b-128e-instruct")
    assert mav.identity_certifiable is True        # KNOWN cutoff + 38 resistant + new
    assert mav.pilot_admissible is False           # grader blocks it
    assert "GRADER_BLOCKED" in mav.blocker


def test_tier2_unknown_cutoffs_not_identity_certifiable():
    cl, ps = _synthetic_payloads()
    m = build_frontier_manifest_from_codeforces_v1(cl, ps, fetched_on="x")
    per = certify_models_on_manifest_v1(m, grader_admissible=False)
    for x in per:
        if x.model_id != "meta/llama-4-maverick-17b-128e-instruct":
            assert x.identity_certifiable is False      # C1 UNKNOWN
            assert x.pilot_admissible is False


# --------------------------------------------------- full pipeline + CID invariant

def test_full_pipeline_no_pilot_and_decision_cid_invariant():
    cl, ps = _synthetic_payloads()
    res = run_frontier_functional_construction_v1(
        cl, ps, verified_on="2026-05-30")
    assert res.verdict == VERDICT_NONE
    assert res.pilot_earned is False
    assert res.n_identity_certifiable_models == 1       # only Maverick
    assert res.admissibility.identity_admissible is True
    assert res.admissibility.pilot_admissible is False
    # LCB-inherited decision CID byte-identical to W114/W115/W116/W117.
    decision_cid = (
        res.lcb_inherited.upstream_admission.frontier_certification.decision.cid())
    assert decision_cid.startswith("258b6ed7")
    assert res.w119_fire_condition.fires_now is False


def test_disclosure_matrix_glm5_noted_but_nothing_newly_disclosed():
    s = disclosure_delta_since_w117_v1()
    assert s["any_newly_disclosed_since_w117"] is False
    assert "zai-org/glm-5" in s["newly_noted_uncertifiable"]
    assert s["counts"].get("KNOWN") == 1                # only Maverick KNOWN


def test_result_cid_changes_with_admitted_set():
    cl1, ps1 = _synthetic_payloads(n_admit_a=20, n_admit_b=18)
    cl2, ps2 = _synthetic_payloads(n_admit_a=21, n_admit_b=18)
    r1 = run_frontier_functional_construction_v1(cl1, ps1, verified_on="x")
    r2 = run_frontier_functional_construction_v1(cl2, ps2, verified_on="x")
    assert r1.manifest.manifest_cid() != r2.manifest.manifest_cid()
    assert r1.cid() != r2.cid()


# --------------------------------------------------- FALSIFIABILITY

def test_falsifiability_official_grader_unlocks_maverick_pilot_and_fires():
    """If a synthetic source HAS an official executable grader (O7) AND >= 30 admitted
    post-v6 problems exist, Maverick (KNOWN cutoff) becomes PILOT-ADMISSIBLE, the
    verdict flips to CERTIFIABLE, and W119 fires — proving the no-go is the grader's
    absence, not a tautology."""
    cl, ps = _synthetic_payloads(n_admit_a=20, n_admit_b=18)   # 38 >= 30
    grader_source = OfficialSourceV1(
        source_kind=SOURCE_CODEFORCES_API,
        official_surface="(synthetic: official source WITH an executable grader)",
        has_problem_metadata_api=True, has_official_dates=True,
        has_statements=True, has_sample_tests=True,
        has_official_executable_test_suite=True,        # O7 satisfied
        test_artifact_status="CLEAN_OFFICIAL_API",
        note="synthetic falsifier")
    res = run_frontier_functional_construction_v1(
        cl, ps, verified_on="x",
        source=grader_source, family=(grader_source,))
    assert res.admissibility.o7_official_grader is True
    assert res.admissibility.pilot_admissible is True
    mav = next(x for x in res.per_model
               if x.model_id == "meta/llama-4-maverick-17b-128e-instruct")
    assert mav.identity_certifiable is True
    assert mav.pilot_admissible is True                  # grader now present
    assert res.pilot_earned is True
    assert res.verdict == VERDICT_CERTIFIABLE
    assert res.w119_fire_condition.fires_now is True


def test_falsifiability_grader_alone_without_min_slice_does_not_fire():
    """Grader present but < 30 admitted => still not pilot-admissible (both gates needed)."""
    cl, ps = _synthetic_payloads(n_admit_a=5, n_admit_b=4)     # 9 < 30
    grader_source = dataclasses.replace(
        OFFICIAL_SOURCE_FAMILY[0], has_official_executable_test_suite=True,
        test_artifact_status="CLEAN_OFFICIAL_API")
    res = run_frontier_functional_construction_v1(
        cl, ps, verified_on="x",
        source=grader_source, family=(grader_source,))
    assert res.admissibility.o7_official_grader is True
    assert res.admissibility.identity_admissible is False      # < min_slice
    assert res.admissibility.pilot_admissible is False
    assert res.pilot_earned is False
    assert res.verdict == VERDICT_NONE


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
