"""W121 tests — matched EXPOSED official-ICPC control + dual-field contrast.

Pure / NIM-free.  Covers: the pinned listing SHA, the FLIPPED (exposed) date rule, the
>=30 tier-1 count, manifest CID determinism, the typed exclusion audit, Maverick EXPOSED
certification (+ tier-2 UNKNOWN non-certification), the exposed-pilot earn gate, the
matched-family comparison vs the W120 resistant battlefield, the LCB-inherited decision-
CID invariant, the per-surface grader self-test, the deterministic stratified slice, and
the THREE-branch exposed-vs-resistant interpreter (the pre-committed truth-surface logic)
with FALSIFIABILITY tests on each branch.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import hashlib  # noqa: E402
import json  # noqa: E402

from coordpy.coordpy_icpc_exposed_control_v1 import (  # noqa: E402
    AMBIGUITY_BAND_PP,
    EXPOSED_MARGIN_PASS_PP,
    ICPC_EXPOSED_LISTING_SNAPSHOT_V1,
    OUTCOME_AMBIGUOUS,
    OUTCOME_CONFOUND_WEAKENS,
    OUTCOME_LOOPHOLE_CLOSED,
    TIER_CORE,
    W121_EXPOSED_RAW_CLASSIFICATION_SHA256,
    assess_exposed_admissibility_v1,
    build_exposed_manifest_v1,
    build_matched_family_comparison_v1,
    certify_models_on_exposed_v1,
    classify_exposed_listing_v1,
    exposed_grader_selftest_summary_v1,
    interpret_exposed_vs_resistant_v1,
    run_exposed_control_construction_v1,
    select_exposed_core_slice_v1,
)
from coordpy.coordpy_icpc_battlefield_v1 import core_slice_cid_v1  # noqa: E402
from coordpy.coordpy_icpc_public_functional_v1 import MAVERICK_CUTOFF_BOUNDARY  # noqa: E402

MAVERICK = "meta/llama-4-maverick-17b-128e-instruct"
_VON = "2026-05-31"


def test_listing_sha_pinned():
    canon = json.dumps([list(r) for r in ICPC_EXPOSED_LISTING_SNAPSHOT_V1],
                       sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(canon).hexdigest() == W121_EXPOSED_RAW_CLASSIFICATION_SHA256


def test_all_admitted_are_exposed_and_core_ge_30():
    probs = classify_exposed_listing_v1()
    adm = [p for p in probs if p.admitted]
    # every admitted problem is dated AT OR BEFORE the Maverick cutoff (EXPOSED)
    assert all(p.contest_date <= MAVERICK_CUTOFF_BOUNDARY for p in adm)
    core = [p for p in adm if p.tier == TIER_CORE]
    assert len(core) == 42
    assert len(core) >= 30


def test_manifest_counts_dates_and_cid_deterministic():
    m1 = build_exposed_manifest_v1(fetched_on=_VON)
    m2 = build_exposed_manifest_v1(fetched_on="different-day")
    assert m1.n_core_passfail == 42
    assert m1.date_min == "2022-03-14" and m1.date_max == "2023-11-11"
    assert m1.date_max <= MAVERICK_CUTOFF_BOUNDARY            # exposed
    assert set(m1.surfaces) == {"ECNA", "RMRC"}
    # CID is over the admitted-id set, independent of fetched_on
    assert m1.manifest_cid() == m2.manifest_cid()


def test_exposed_date_rule_is_flipped_falsifiability():
    """A post-cutoff row must be EXCLUDED by the exposed rule (the complement of W120)."""
    listing = list(ICPC_EXPOSED_LISTING_SNAPSHOT_V1)
    # flip the first row's date to AFTER the cutoff
    repo, short, _d, kind, ns, na = listing[0]
    listing[0] = (repo, short, "2025-01-01", kind, ns, na)
    probs = classify_exposed_listing_v1(tuple(listing))
    flipped = [p for p in probs if p.short_name == short]
    assert flipped and not flipped[0].admitted
    assert flipped[0].exclusion_reason == "post_cutoff_or_undated"


def test_exclusion_audit_typed():
    probs = classify_exposed_listing_v1()
    from coordpy.coordpy_icpc_battlefield_v1 import exclusion_audit_v1
    audit = exclusion_audit_v1(probs)
    # the two custom-no-validator problems (teamchange, colortubes) are the exclusions
    assert audit.by_exclusion_reason.get("excluded_kind:custom_no_validator") == 2


def test_maverick_exposed_certifiable_tier2_not():
    m = build_exposed_manifest_v1(fetched_on=_VON)
    certs = certify_models_on_exposed_v1(m, grader_admissible=True, slice_admissible=True)
    by = {c.model_id: c for c in certs}
    mav = by[MAVERICK]
    assert mav.c1_cutoff_known and mav.c2e_enough_exposed and mav.c3_reachable_stronger_comparable
    assert mav.exposed_certifiable and mav.pilot_admissible
    assert mav.n_exposed_before >= 30
    # tier-2 models have UNKNOWN cutoffs => NOT exposed-certifiable (C1 fails), same as W120
    for mid, c in by.items():
        if mid != MAVERICK:
            assert not c.exposed_certifiable and not c.pilot_admissible


def test_exposed_pilot_earned_and_admissible():
    res = run_exposed_control_construction_v1(verified_on=_VON)
    assert res.admissibility.pilot_admissible
    assert res.exposed_pilot_earned
    assert res.n_exposed_certifiable_models == 1


def test_c2e_falsifiability_sub30_blocks_maverick():
    """A truncated <30 exposed listing must make Maverick NOT exposed-certifiable (C2e)."""
    short_listing = tuple(ICPC_EXPOSED_LISTING_SNAPSHOT_V1[:10])
    m = build_exposed_manifest_v1(short_listing, fetched_on=_VON)
    assert m.n_functional_exposed_before(MAVERICK_CUTOFF_BOUNDARY) < 30
    certs = certify_models_on_exposed_v1(m, grader_admissible=True, slice_admissible=False)
    mav = next(c for c in certs if c.model_id == MAVERICK)
    assert not mav.c2e_enough_exposed
    assert not mav.exposed_certifiable
    assert not mav.pilot_admissible


def test_grader_selftest_each_surface():
    st = exposed_grader_selftest_summary_v1()
    assert st["grader_proven_executable_each_surface"] is True
    assert st["n_surfaces"] == 4
    assert st["n_problems_self_tested"] == 30
    # every exposed surface has >= 1 all-pass problem
    for sk, sv in st["per_surface"].items():
        assert sv["all_pass"] and sv["n_problems"] >= 1


def test_matched_family_same_family_differs_only_in_cutoff_side():
    m = build_exposed_manifest_v1(fetched_on=_VON)
    probs = classify_exposed_listing_v1()
    sl = select_exposed_core_slice_v1(probs, n_problems=30)
    mf = build_matched_family_comparison_v1(m, core_slice_cid_v1(sl), verified_on=_VON)
    assert mf.differs_only_in_cutoff_side
    assert set(mf.shared_surface_families) == {"ECNA", "RMRC"}
    assert mf.same_org and mf.same_package_format_family and mf.same_grader_and_oracle
    assert mf.same_tier_discipline and mf.same_difficulty_class
    assert mf.same_model_and_evaluator_line
    # exposed dates strictly BEFORE resistant dates (opposite cutoff sides)
    assert mf.exposed_date_max < mf.resistant_date_min
    assert mf.exposed_n_core == 42 and mf.resistant_n_core == 45


def test_lcb_inherited_decision_cid_invariant():
    res = run_exposed_control_construction_v1(verified_on=_VON)
    cid = res.to_dict()["lcb_inherited_decision_cid"]
    assert cid.startswith("258b6ed7")


def test_slice_selector_deterministic_and_balanced():
    probs = classify_exposed_listing_v1()
    s1 = select_exposed_core_slice_v1(probs, n_problems=30)
    s2 = select_exposed_core_slice_v1(probs, n_problems=30)
    assert [p.problem_id for p in s1] == [p.problem_id for p in s2]
    assert len(s1) == 30
    # spans BOTH surface families and multiple contest years (not all one drop)
    assert len({p.surface for p in s1}) == 2
    assert len({p.contest_date for p in s1}) >= 3


# --------------------------------------- the THREE-branch interpreter (falsifiability)

def test_interpret_exposed_margin_closes_loophole():
    """Exposed margin >= +5 while resistant ~0 => difficulty loophole CLOSED."""
    o = interpret_exposed_vs_resistant_v1(
        exposed_b_minus_a1=6.67, resistant_b_minus_a1=0.0)
    assert o.outcome == OUTCOME_LOOPHOLE_CLOSED
    assert o.exposed_shows_margin and not o.paired_seed_earned


def test_interpret_exposed_null_weakens_confound():
    """Exposed null too (within band) => contamination-confound WEAKENS, ceiling hardens."""
    o = interpret_exposed_vs_resistant_v1(
        exposed_b_minus_a1=0.0, resistant_b_minus_a1=0.0)
    assert o.outcome == OUTCOME_CONFOUND_WEAKENS
    assert o.exposed_is_null_too and not o.paired_seed_earned


def test_interpret_ambiguous_earns_paired_seed():
    """Between the margin bar and the null band => AMBIGUOUS, earn ONE paired seed."""
    mid = (EXPOSED_MARGIN_PASS_PP + AMBIGUITY_BAND_PP) / 2.0  # ~4.17 pp
    o = interpret_exposed_vs_resistant_v1(
        exposed_b_minus_a1=mid, resistant_b_minus_a1=0.0)
    assert o.outcome == OUTCOME_AMBIGUOUS
    assert o.is_ambiguous and o.paired_seed_earned


def test_interpret_band_boundaries_are_pre_committed():
    assert EXPOSED_MARGIN_PASS_PP == 5.0
    assert abs(AMBIGUITY_BAND_PP - 3.34) < 1e-9


def test_interpret_actual_w121_pilot_result():
    """Pin the ACTUAL W121 pilot outcome: exposed +3.33 pp (one net rescue at n=30) vs
    resistant +0.00 pp lands within the pre-committed ±3.34 null band ⇒ CONFOUND_WEAKENS,
    NO paired seed.  (Guards the razor-edge 3.33 <= 3.34 boundary against drift.)"""
    o = interpret_exposed_vs_resistant_v1(
        exposed_b_minus_a1=3.33, resistant_b_minus_a1=0.0)
    assert o.outcome == OUTCOME_CONFOUND_WEAKENS
    assert o.exposed_is_null_too and not o.exposed_shows_margin
    assert not o.paired_seed_earned
