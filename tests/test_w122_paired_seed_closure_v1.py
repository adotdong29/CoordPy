"""W122 tests — matched-family paired-seed closure rule + ICPC M3 signal audit.

Falsifiability-first: every closure branch (B1..B4) has a test that FLIPS only the input
that should change the branch, and the Lane-beta audit has a test proving the verdict
flips KILL->BUILD exactly when the grading regime reveals the hidden expected on >= the
floor of HIDDEN_ONLY turns.  Pure / deterministic / NIM-free.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coordpy.coordpy_icpc_paired_seed_closure_v1 import (  # noqa: E402
    B1_CAVEAT_CLOSED, B2_EXPOSED_RESTRENGTHENS, B3_RESISTANT_CANDIDATE, B4_AMBIGUOUS,
    MARGIN_PASS_PP, NULL_BAND_PP, M3_SIGNAL_FLOOR,
    VERDICT_BUILD_M3, VERDICT_KILL_M3,
    TURN_HIDDEN_ONLY, TURN_PUBLIC_SAMPLE_WRONG, TURN_RUNTIME_TRACEBACK,
    TURN_TIMEOUT, TURN_NO_SIGNAL,
    FieldSeedResultV1, aggregate_field_seeds_v1,
    audit_icpc_mechanism_signal_v1, classify_reflexion_turn_v1,
    interpret_paired_closure_v1, w123_fire_condition_v1,
)


# --------------------------------------------------------------- helpers

def _seed(field, seed, margin, clean=False, a0=10.0, mlb2=0.10):
    return FieldSeedResultV1(
        field=field, seed=seed, b_minus_a1_pp=float(margin),
        verdict_label=("PASS_MECHANISM_DRIVEN" if clean else "FAIL"),
        a0_pass_at_1_pct=float(a0), mlb2_rescue_rate=float(mlb2))


def _agg(field, margins, clean=False):
    return aggregate_field_seeds_v1(
        [_seed(field, 120_001 + i, m, clean=clean) for i, m in enumerate(margins)])


# --------------------------------------------------------------- aggregate

def test_aggregate_mean_and_order_independence():
    a = aggregate_field_seeds_v1([
        _seed("resistant", 120_002, 3.33), _seed("resistant", 120_001, 0.0)])
    assert a.n_seeds == 2
    assert a.seeds == (120_001, 120_002)          # sorted
    assert abs(a.mean_b_minus_a1_pp - 1.665) < 1e-6
    assert a.all_seeds_clean_pass is False


def test_aggregate_rejects_mixed_fields():
    with pytest.raises(ValueError):
        aggregate_field_seeds_v1(
            [_seed("resistant", 1, 0.0), _seed("exposed", 2, 0.0)])


# --------------------------------------------------------------- B1 (caveat closed)

def test_b1_both_null_closes_caveat():
    # the actual W122 expected case: seed1 (+0.00 / +3.33), a benign seed2 keeps both in band
    res = interpret_paired_closure_v1(
        resistant=_agg("resistant", [0.0, 0.0]),
        exposed=_agg("exposed", [3.33, 0.0]))
    assert res.branch == B1_CAVEAT_CLOSED
    assert res.caveat_closed is True
    assert res.third_seed_earned is False
    assert res.resistant_in_null_band and res.exposed_in_null_band


# --------------------------------------------------------------- B2 (exposed restrengthens)

def test_b2_exposed_margin_clean_while_resistant_null():
    res = interpret_paired_closure_v1(
        resistant=_agg("resistant", [0.0, 0.0]),
        exposed=_agg("exposed", [6.67, 6.67], clean=True))   # mean +6.67, clean per seed
    assert res.branch == B2_EXPOSED_RESTRENGTHENS
    assert res.exposed_shows_margin is True
    assert res.caveat_closed is True


def test_b2_requires_clean_per_seed_gates():
    # same +6.67 exposed mean but NOT clean per seed => not B2; resistant null => B4
    res = interpret_paired_closure_v1(
        resistant=_agg("resistant", [0.0, 0.0]),
        exposed=_agg("exposed", [6.67, 6.67], clean=False))
    assert res.branch == B4_AMBIGUOUS
    assert res.exposed_shows_margin is False       # margin not "clean"


# --------------------------------------------------------------- B3 (resistant candidate)

def test_b3_resistant_margin_is_candidate_third_retirement():
    res = interpret_paired_closure_v1(
        resistant=_agg("resistant", [5.0, 6.67], clean=True),
        exposed=_agg("exposed", [3.33, 0.0]))
    assert res.branch == B3_RESISTANT_CANDIDATE
    assert res.resistant_shows_margin is True
    assert res.caveat_closed is False              # a candidate, not a closure


def test_b3_takes_precedence_over_b2():
    # both fields show a clean margin => B3 wins (resistant is the headline)
    res = interpret_paired_closure_v1(
        resistant=_agg("resistant", [6.67, 6.67], clean=True),
        exposed=_agg("exposed", [6.67, 6.67], clean=True))
    assert res.branch == B3_RESISTANT_CANDIDATE


# --------------------------------------------------------------- B4 (ambiguous => 3rd seed)

def test_b4_mean_in_the_gap_earns_third_seed():
    # exposed mean lands in the (3.34, 5.00) gap => ambiguous
    res = interpret_paired_closure_v1(
        resistant=_agg("resistant", [0.0, 0.0]),
        exposed=_agg("exposed", [3.34001, 5.0 - 1e-6]))      # mean ~4.17
    assert res.branch == B4_AMBIGUOUS
    assert res.third_seed_earned is True


def test_b4_direction_disagreement_earns_third_seed():
    # one field strongly negative beyond band, other null => not jointly decisive
    res = interpret_paired_closure_v1(
        resistant=_agg("resistant", [-6.67, -6.67]),         # mean -6.67, beyond band
        exposed=_agg("exposed", [0.0, 0.0]))
    assert res.branch == B4_AMBIGUOUS


# --------------------------------------------------------------- Lane beta audit

def _icpc_like_turns():
    # 60 turns ~ the real W120+W121 distribution: mostly public-sample-wrong, ~0 hidden-only
    return ([TURN_PUBLIC_SAMPLE_WRONG] * 38 + [TURN_RUNTIME_TRACEBACK] * 8
            + [TURN_NO_SIGNAL] * 11 + [TURN_TIMEOUT] * 2 + [TURN_HIDDEN_ONLY] * 1)


def test_audit_kills_m3_on_icpc_secret_token_diff_regime():
    a = audit_icpc_mechanism_signal_v1(
        turn_classes=_icpc_like_turns(),
        grader_reveals_hidden_expected=False)        # ICPC secret token-diff
    assert a.verdict == VERDICT_KILL_M3
    assert a.m3_exclusive_signal_fraction == 0.0     # hidden expected is secret


def test_audit_falsifiability_build_when_regime_reveals_and_enough_hidden():
    # synthetic regime that DOES reveal hidden expected, with >= floor hidden-only turns
    turns = [TURN_HIDDEN_ONLY] * 40 + [TURN_PUBLIC_SAMPLE_WRONG] * 60
    a = audit_icpc_mechanism_signal_v1(
        turn_classes=turns, grader_reveals_hidden_expected=True)
    assert a.hidden_only_fraction >= M3_SIGNAL_FLOOR
    assert a.m3_exclusive_signal_fraction == a.hidden_only_fraction
    assert a.verdict == VERDICT_BUILD_M3


def test_audit_kills_even_revealing_regime_if_below_floor():
    # revealing regime but only 10% hidden-only => still below the 33% floor => KILL
    turns = [TURN_HIDDEN_ONLY] * 10 + [TURN_PUBLIC_SAMPLE_WRONG] * 90
    a = audit_icpc_mechanism_signal_v1(
        turn_classes=turns, grader_reveals_hidden_expected=True)
    assert a.verdict == VERDICT_KILL_M3


# --------------------------------------------------------------- turn classifier

def test_classifier_skips_initial_prompt():
    assert classify_reflexion_turn_v1("You are an expert competitive programmer ...") is None


def test_classifier_public_sample_wrong():
    p = ("reflective debugging loop\n--- Attempt 1 (REJECTED by the judge) ---\n"
         "```python\nx\n```\nPublic sample results:\n  sample 1: WRONG (expected `5`)")
    assert classify_reflexion_turn_v1(p) == TURN_PUBLIC_SAMPLE_WRONG


def test_classifier_hidden_only():
    p = ("reflective debugging loop\n--- Attempt 1 (REJECTED by the judge) ---\n"
         "```python\nx\n```\nPublic sample results:\n  sample 1: PASS\n  sample 2: PASS")
    assert classify_reflexion_turn_v1(p) == TURN_HIDDEN_ONLY


def test_classifier_runtime_traceback():
    p = ("reflective debugging loop\n--- Attempt 1 (REJECTED by the judge) ---\n"
         "```python\nx\n```\nExecutor stderr (tail):\nTraceback ... ValueError: boom")
    assert classify_reflexion_turn_v1(p) == TURN_RUNTIME_TRACEBACK


# --------------------------------------------------------------- W123 fire

def test_w123_fire_condition_per_branch():
    for br in (B1_CAVEAT_CLOSED, B2_EXPOSED_RESTRENGTHENS,
               B3_RESISTANT_CANDIDATE, B4_AMBIGUOUS):
        fc = w123_fire_condition_v1(br)
        assert fc.closure_branch == br
        assert len(fc.fires_on) > 0
