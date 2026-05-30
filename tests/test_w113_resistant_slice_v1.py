"""W113 / COO-9 — tests for the resistant-for-Llama-4 slice machinery.

Covers the three new explicit-import-only modules:

* ``livecodebench_resistant_slice_v1`` — date normalization, the strictly-after
  boundary rule, resistant partitioning + typed exclusions, the model-cutoff
  registry, and the KNOWN-cutoff-only certification gate.
* ``cross_scale_resistant_interpretation_v1`` — the falsifiable W113 verdict
  branches (clean reopening vs ambiguous margin vs exposure confirmed).
* ``tier2_readiness_v1`` — tier-2 applicability + the locked spend rule.

The real-corpus equivalence test (resistant 30-slice == the W108 slice CID)
runs only when the SHA-pinned cache is present, and otherwise skips.
"""
from __future__ import annotations

import dataclasses
import os

import pytest

from coordpy.livecodebench_resistant_slice_v1 import (
    CONFIDENCE_KNOWN,
    CONFIDENCE_UNKNOWN,
    MIN_RESISTANT_SLICE,
    MODEL_TRAINING_CUTOFFS,
    cutoff_boundary_for_model_v1,
    is_resistant_for_boundary_v1,
    normalize_contest_date_v1,
    partition_resistant_v1,
    resistant_partition_for_model_v1,
    slice_resistant_for_model_v1,
)
from coordpy.cross_scale_resistant_interpretation_v1 import (
    W108_RESISTANT_LCB_SLICE_CID,
    interpret_cross_scale_resistant_result_v1,
)
from coordpy.tier2_readiness_v1 import (
    TIER2_RANKING,
    assess_tier2_applicability_v1,
    decide_tier2_spend_v1,
)

MAVERICK = "meta/llama-4-maverick-17b-128e-instruct"
LLAMA33 = "meta/llama-3.3-70b-instruct"


# ---- A tiny fake problem so the partition tests need no 134 MB corpus -------

@dataclasses.dataclass(frozen=True)
class _FakeProblem:
    question_id: str
    contest_date: str
    difficulty: str = "medium"


# ---------------------------------------------------------------------------
# date normalization + boundary rule
# ---------------------------------------------------------------------------

def test_normalize_contest_date_parses_iso_datetime():
    assert normalize_contest_date_v1("2025-01-11T18:30:00") == "2025-01-11"
    assert normalize_contest_date_v1("2024-09-01") == "2024-09-01"


@pytest.mark.parametrize("bad", ["", "   ", None, "not-a-date", "2025/01/11",
                                 "2025-13-40", "2025-00-10", "2025-05-00"])
def test_normalize_contest_date_excludes_unparseable(bad):
    assert normalize_contest_date_v1(bad) is None


def test_strictly_after_boundary():
    # boundary 2024-08-31 -> resistant iff day >= 2024-09-01 (strictly after Aug)
    assert is_resistant_for_boundary_v1("2024-09-01", "2024-08-31") is True
    assert is_resistant_for_boundary_v1("2025-01-11", "2024-08-31") is True
    # the entire August window is EXCLUDED (cannot certify strictly-after)
    assert is_resistant_for_boundary_v1("2024-08-31", "2024-08-31") is False
    assert is_resistant_for_boundary_v1("2024-08-01", "2024-08-31") is False
    assert is_resistant_for_boundary_v1("2024-01-15", "2024-08-31") is False
    # None (missing/unparseable) is never resistant
    assert is_resistant_for_boundary_v1(None, "2024-08-31") is False


# ---------------------------------------------------------------------------
# partition with typed exclusions
# ---------------------------------------------------------------------------

def test_partition_resistant_typed_exclusions():
    subset = [
        _FakeProblem("R1", "2025-01-11T00:00:00"),   # resistant
        _FakeProblem("R2", "2024-09-02"),            # resistant (just after Aug)
        _FakeProblem("E1", "2024-08-15"),            # in-August -> not after
        _FakeProblem("E2", "2024-01-01"),            # pre-cutoff -> not after
        _FakeProblem("M1", ""),                      # missing date
        _FakeProblem("U1", "garbage"),               # unparseable date
    ]
    part = partition_resistant_v1(subset, boundary_date="2024-08-31")
    assert part.n_total == 6
    assert part.n_resistant == 2
    assert set(part.resistant_question_ids) == {"R1", "R2"}
    assert set(part.excluded_not_after_cutoff) == {"E1", "E2"}
    assert set(part.excluded_missing_date) == {"M1"}
    assert set(part.excluded_unparseable_date) == {"U1"}
    assert part.resistant_date_min == "2024-09-02"
    assert part.resistant_date_max == "2025-01-11"
    # cid is stable + order-insensitive of construction (ids preserve input order)
    assert len(part.partition_cid()) == 64


def test_partition_preserves_input_order_for_resistant_ids():
    subset = [
        _FakeProblem("B", "2025-03-01"),
        _FakeProblem("A", "2025-01-01"),
    ]
    part = partition_resistant_v1(subset, boundary_date="2024-08-31")
    # order follows input (loader order), NOT date sort
    assert part.resistant_question_ids == ("B", "A")


# ---------------------------------------------------------------------------
# model-cutoff registry + KNOWN-only certification
# ---------------------------------------------------------------------------

def test_registry_maverick_known_aug_2024():
    c = cutoff_boundary_for_model_v1(MAVERICK)
    assert c.boundary_date == "2024-08-31"
    assert c.confidence == CONFIDENCE_KNOWN
    assert c.is_resistant_grade() is True


def test_registry_llama33_known():
    c = cutoff_boundary_for_model_v1(LLAMA33)
    assert c.confidence == CONFIDENCE_KNOWN
    # matches the W108 post-2024-01-01 convention
    assert c.boundary_date == "2023-12-31"


def test_registry_tier2_all_unknown():
    for mid in ("qwen/qwen3-coder-480b-a35b-instruct",
                "deepseek-ai/deepseek-v4-pro",
                "mistralai/mistral-small-4-119b-2603"):
        assert MODEL_TRAINING_CUTOFFS[mid].confidence == CONFIDENCE_UNKNOWN
        assert MODEL_TRAINING_CUTOFFS[mid].is_resistant_grade() is False


def test_registry_unknown_model_raises():
    with pytest.raises(KeyError):
        cutoff_boundary_for_model_v1("acme/not-a-real-model")


def test_resistant_partition_for_model_maverick():
    subset = [
        _FakeProblem("R1", "2025-01-11"),
        _FakeProblem("E1", "2024-06-15"),  # exposed for Maverick
    ]
    part = resistant_partition_for_model_v1(subset, model_id=MAVERICK)
    assert part.n_resistant == 1
    assert part.resistant_question_ids == ("R1",)


# ---------------------------------------------------------------------------
# slice applicability certification (the spend gate primitive)
# ---------------------------------------------------------------------------

def test_slice_resistant_for_maverick_2025_slice():
    ok, reason = slice_resistant_for_model_v1(
        slice_date_min="2025-01-11", model_id=MAVERICK)
    assert ok is True
    assert "RESISTANT" in reason


def test_slice_exposed_for_maverick_if_min_in_august():
    ok, reason = slice_resistant_for_model_v1(
        slice_date_min="2024-08-15", model_id=MAVERICK)
    assert ok is False
    assert "EXPOSED" in reason


def test_slice_refused_for_unknown_cutoff_tier2():
    # Even though the 2025 slice POST-dates a guessed boundary, an UNKNOWN
    # cutoff cannot CERTIFY resistance -> refused.
    ok, reason = slice_resistant_for_model_v1(
        slice_date_min="2025-01-11",
        model_id="deepseek-ai/deepseek-v4-pro")
    assert ok is False
    assert "UNKNOWN" in reason


def test_slice_refused_for_unregistered_model():
    ok, reason = slice_resistant_for_model_v1(
        slice_date_min="2025-01-11", model_id="acme/nope")
    assert ok is False
    assert "UNREGISTERED_MODEL" in reason


# ---------------------------------------------------------------------------
# cross-scale interpretation branches (falsifiable)
# ---------------------------------------------------------------------------

def _interp(label):
    return interpret_cross_scale_resistant_result_v1(
        model_id=MAVERICK,
        resistant_benchmark="LiveCodeBench release_v6",
        verdict_label=label, b_minus_a1_pp=0.0, mlb2_rescue_rate=0.0)


def test_interp_pass_mechanism_driven_reopens_clean():
    r = _interp("PASS_MECHANISM_DRIVEN")
    assert r.outcome == "RESISTANT_SUPERIORITY_REOPENS"
    assert r.clean_resistant_reopening is True
    assert r.entitled_stronger_superiority_claim is True
    assert "W114" in r.w114_branch


def test_interp_pass_non_mechanism_driven_is_ambiguous_not_clean():
    r = _interp("PASS_NON_MECHANISM_DRIVEN")
    assert r.outcome == "RESISTANT_MARGIN_NON_MECHANISM"
    assert r.clean_resistant_reopening is False
    assert r.entitled_stronger_superiority_claim is False
    assert r.confound_direction == "UNCHANGED"


def test_interp_fail_confirms_exposure():
    r = _interp("FAIL")
    assert r.outcome == "EXPOSURE_CONFIRMED"
    assert r.clean_resistant_reopening is False
    assert r.entitled_stronger_superiority_claim is False
    assert r.confound_direction == "STRENGTHENS"


def test_interp_cid_stable_and_unique_per_branch():
    cids = {_interp(l).cid() for l in ("PASS_MECHANISM_DRIVEN",
                                       "PASS_NON_MECHANISM_DRIVEN", "FAIL")}
    assert len(cids) == 3


# ---------------------------------------------------------------------------
# tier-2 readiness + spend rule
# ---------------------------------------------------------------------------

def test_tier2_ranking_order_code_first():
    assert TIER2_RANKING[0].model_id == "qwen/qwen3-coder-480b-a35b-instruct"
    assert [c.rank_within_tier for c in TIER2_RANKING] == [1, 2, 3]


def test_tier2_none_certifiable_on_2025_slice():
    applic = assess_tier2_applicability_v1(slice_date_min="2025-01-11")
    assert len(applic) == 3
    assert all(not a.slice_certifiably_resistant for a in applic)


def test_tier2_spend_blocked_even_if_main_lane_reopens():
    # the missing-instrument block dominates: no certifiable tier-2 slice exists
    d = decide_tier2_spend_v1(
        main_lane_outcome="RESISTANT_SUPERIORITY_REOPENS",
        slice_date_min="2025-01-11")
    assert d.main_lane_earns_escalation is True
    assert d.n_certifiable_targets == 0
    assert d.spend_eligible is False
    assert "release_v7" in d.next_instrument_if_blocked


def test_tier2_spend_blocked_on_exposure_too():
    d = decide_tier2_spend_v1(
        main_lane_outcome="EXPOSURE_CONFIRMED", slice_date_min="2025-01-11")
    assert d.spend_eligible is False


def test_tier2_spend_not_earned_on_ambiguous_outcome():
    d = decide_tier2_spend_v1(
        main_lane_outcome="RESISTANT_MARGIN_NON_MECHANISM",
        slice_date_min="2025-01-11")
    assert d.main_lane_earns_escalation is False
    assert d.spend_eligible is False


# ---------------------------------------------------------------------------
# real-corpus equivalence: resistant 30-slice == W108 slice (clean cross-scale)
# ---------------------------------------------------------------------------

_CACHE = os.path.expanduser("~/.cache/coordpy/livecodebench-test6.jsonl")
_SHA = "bb4c364f71921c4495a6ad15abe1a927350b720009f4933e2e71f8af0f6fd1f5"


@pytest.mark.skipif(not os.path.exists(_CACHE),
                    reason="SHA-pinned LiveCodeBench cache not present")
def test_real_corpus_resistant_slice_equals_w108_slice():
    import hashlib
    import json as _json
    from coordpy.livecodebench_loader_v1 import (
        load_livecodebench_functional_v1)
    from coordpy.livecodebench_reflexion_bench_v1 import (
        select_livecodebench_functional_slice_v1)
    subset = load_livecodebench_functional_v1(
        release="release_v6", cache_path=_CACHE, expected_sha256=_SHA)
    part = resistant_partition_for_model_v1(subset, model_id=MAVERICK)
    # the entire release_v6 functional increment is 2025 => all resistant
    assert part.n_resistant == len(subset) >= MIN_RESISTANT_SLICE
    assert part.n_resistant == 63
    rset = set(part.resistant_question_ids)
    resistant_subset = tuple(p for p in subset if p.question_id in rset)
    sl = select_livecodebench_functional_slice_v1(
        resistant_subset, n_problems=30)
    qids = [p.question_id for p in sl]
    cid = hashlib.sha256(_json.dumps(
        {"kind": "w108_lcb_pilot_slice_v1", "question_ids": qids},
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert cid == W108_RESISTANT_LCB_SLICE_CID
