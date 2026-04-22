"""Phase 37 Part C — nested-contest task family tests."""

from __future__ import annotations

import pytest

from vision_mvp.tasks.nested_contested_incident import (
    CLAIM_COORDINATION_BRIEFING, NESTED_ALL_STRATEGIES,
    STRATEGY_NESTED_ADAPTIVE_1R, STRATEGY_NESTED_ADAPTIVE_2R,
    STRATEGY_NESTED_DYNAMIC, STRATEGY_NESTED_STATIC,
    build_nested_bank, grade_nested, nested_round_oracle,
    run_nested_bank, run_nested_one_round_adaptive_sub,
    run_nested_two_round_adaptive_sub, run_nested_two_round_thread,
    _format_briefing, _parse_briefing,
)


def test_build_nested_bank_has_three_scenarios():
    bank = build_nested_bank(seed=37, distractors_per_role=4)
    assert len(bank) == 3
    ids = {s.scenario_id for s in bank}
    assert "nested_tls_requires_sysadmin_witness" in ids


def test_round_oracle_reports_round_dependent_class():
    bank = build_nested_bank(seed=37, distractors_per_role=4)
    tls = bank[0]
    r1 = nested_round_oracle(tls, 1, "network", "TLS_EXPIRED",
                               "tls")
    r2 = nested_round_oracle(tls, 2, "network", "TLS_EXPIRED",
                               "tls")
    assert r1 == "UNCERTAIN"
    assert r2 == "INDEPENDENT_ROOT"


def test_dynamic_nested_2r_resolves_tls_correctly():
    bank = build_nested_bank(seed=37, distractors_per_role=4)
    tls = bank[0]
    _, handoffs, debug = run_nested_two_round_thread(
        tls, witness_token_cap=12, max_events_per_role=200)
    assert debug.thread_id is not None
    assert len(debug.round1_replies) >= 2
    assert len(debug.round2_replies) >= 1
    grading = grade_nested(tls, handoffs)
    assert grading["full_correct"] is True
    assert grading["selected_claim_kind"] == "TLS_EXPIRED"


def test_adaptive_sub_1r_fails_on_nested_tls():
    bank = build_nested_bank(seed=37, distractors_per_role=4)
    tls = bank[0]
    _, handoffs, debug = run_nested_one_round_adaptive_sub(
        tls, witness_token_cap=12, max_events_per_role=200)
    grading = grade_nested(tls, handoffs)
    assert grading["full_correct"] is False
    # Single-round returns NO_CONSENSUS (no IR hypotheses).
    assert debug.resolution_kind in ("NO_CONSENSUS",
                                       "CONFLICT",
                                       "SINGLE_INDEPENDENT_ROOT")


def test_adaptive_sub_2r_recovers_on_nested_tls():
    bank = build_nested_bank(seed=37, distractors_per_role=4)
    tls = bank[0]
    _, handoffs, debug = run_nested_two_round_adaptive_sub(
        tls, witness_token_cap=12, max_events_per_role=200)
    assert debug.n_briefings_installed >= 1
    assert len(debug.round2_replies) >= 1
    grading = grade_nested(tls, handoffs)
    assert grading["full_correct"] is True


def test_briefing_roundtrip():
    round1 = [("network", "INDEPENDENT_ROOT", 0),
               ("sysadmin", "UNCERTAIN", 1)]
    payload = _format_briefing(round1)
    parsed = _parse_briefing(payload)
    assert parsed == round1


def test_claim_coordination_briefing_kind_name():
    assert CLAIM_COORDINATION_BRIEFING == "COORDINATION_BRIEFING"


def test_run_nested_bank_pooled_accuracy():
    bank = build_nested_bank(seed=37, distractors_per_role=4)
    ms = run_nested_bank(bank,
                         strategies=(STRATEGY_NESTED_STATIC,
                                      STRATEGY_NESTED_ADAPTIVE_1R,
                                      STRATEGY_NESTED_ADAPTIVE_2R,
                                      STRATEGY_NESTED_DYNAMIC))
    by_strat: dict[str, list[bool]] = {}
    for m in ms:
        by_strat.setdefault(m.strategy, []).append(
            m.grading["full_correct"])
    # Static and adaptive_sub_1r should fail on nested bank.
    assert all(v is False for v in by_strat[STRATEGY_NESTED_STATIC])
    assert all(v is False for v in by_strat[
        STRATEGY_NESTED_ADAPTIVE_1R])
    # Both 2-round strategies should reach 100 % on the 3
    # nested scenarios.
    assert all(v is True for v in by_strat[
        STRATEGY_NESTED_ADAPTIVE_2R])
    assert all(v is True for v in by_strat[
        STRATEGY_NESTED_DYNAMIC])


def test_dynamic_vs_adaptive_2r_accuracy_equivalent_on_nested():
    bank = build_nested_bank(seed=37, distractors_per_role=6)
    ms = run_nested_bank(bank,
                         strategies=(STRATEGY_NESTED_ADAPTIVE_2R,
                                      STRATEGY_NESTED_DYNAMIC))
    dyn = [m.grading["full_correct"] for m in ms
             if m.strategy == STRATEGY_NESTED_DYNAMIC]
    adp = [m.grading["full_correct"] for m in ms
             if m.strategy == STRATEGY_NESTED_ADAPTIVE_2R]
    assert sum(dyn) == sum(adp)
    assert sum(dyn) == len(dyn)


def test_dynamic_nested_2r_uses_zero_briefings():
    """Structural claim: the dynamic thread needs NO inter-round
    briefing edges — it reads the thread ``replies`` list natively.
    """
    bank = build_nested_bank(seed=37, distractors_per_role=4)
    ms = run_nested_bank(bank,
                         strategies=(STRATEGY_NESTED_DYNAMIC,))
    for m in ms:
        assert m.debug.get("n_briefings_installed", 0) == 0


def test_adaptive_sub_2r_uses_positive_briefings():
    """Structural claim: the adaptive-sub 2-round analogue uses
    at least one inter-round briefing edge per contested scenario.
    """
    bank = build_nested_bank(seed=37, distractors_per_role=4)
    ms = run_nested_bank(bank,
                         strategies=(STRATEGY_NESTED_ADAPTIVE_2R,))
    # Every scenario in the bank is nested-contested, so each
    # triggers at least one briefing.
    for m in ms:
        assert m.debug.get("n_briefings_installed", 0) >= 1
