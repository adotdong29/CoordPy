"""Phase 38 Part B — minimum dynamic primitive ablation tests."""

from __future__ import annotations

import pytest

from vision_mvp.core.primitive_ablation import (
    AblatedFeatures, FEATURES, full_features, no_features,
    only_missing, run_ablated_thread_contested,
    run_ablated_thread_nested,
)
from vision_mvp.tasks.contested_incident import (
    build_contested_bank, decoder_from_handoffs_phase35,
)
from vision_mvp.tasks.nested_contested_incident import (
    build_nested_bank, grade_nested,
)


def test_features_ordering():
    assert FEATURES == (
        "typed_vocab", "bounded_witness",
        "terminating_resolution", "round_aware_state",
        "frozen_membership")


def test_full_features_default_on():
    f = full_features()
    assert f.typed_vocab is True
    assert f.bounded_witness is True
    assert f.terminating_resolution is True
    assert f.round_aware_state is True
    assert f.frozen_membership is True


def test_no_features_all_off():
    f = no_features()
    for k in FEATURES:
        assert getattr(f, k) is False


def test_only_missing_typed_vocab():
    f = only_missing("typed_vocab")
    assert f.typed_vocab is False
    assert f.bounded_witness is True
    assert f.terminating_resolution is True
    assert f.round_aware_state is True


def test_only_missing_unknown_raises():
    with pytest.raises(ValueError):
        only_missing("no_such_feature")


def test_ablated_thread_contested_full_recovers():
    bank = build_contested_bank(seed=35, distractors_per_role=4)
    n_correct = 0
    for scen in bank:
        router, handoffs, dbg = run_ablated_thread_contested(
            scen, full_features())
        cue = decoder_from_handoffs_phase35(handoffs)
        ok = (cue["root_cause"] == scen.gold_root_cause
              and tuple(sorted(cue["services"]))
                  == tuple(sorted(scen.gold_services))
              and cue["remediation"] == scen.gold_remediation)
        if ok:
            n_correct += 1
    # Full-featured thread should solve every contested scenario.
    assert n_correct == len(bank)


def test_ablated_thread_no_terminating_resolution_falls_back():
    bank = build_contested_bank(seed=35, distractors_per_role=4)
    n_correct = 0
    for scen in bank:
        router, handoffs, dbg = run_ablated_thread_contested(
            scen, only_missing("terminating_resolution"))
        cue = decoder_from_handoffs_phase35(handoffs)
        ok = (cue["root_cause"] == scen.gold_root_cause)
        if ok:
            n_correct += 1
    # At most the concordant scenarios should survive (static
    # priority can match only the concordant + a couple of
    # contested scenarios whose static pick coincides with gold).
    assert n_correct < len(bank)


def test_ablated_thread_nested_full_recovers():
    bank = build_nested_bank(seed=37, distractors_per_role=4)
    n_correct = 0
    for scen in bank:
        router, handoffs, dbg = run_ablated_thread_nested(
            scen, full_features())
        grading = grade_nested(scen, handoffs)
        if grading["full_correct"]:
            n_correct += 1
    assert n_correct == len(bank)


def test_ablated_thread_nested_no_round_aware_state_collapses():
    bank = build_nested_bank(seed=37, distractors_per_role=4)
    n_correct = 0
    for scen in bank:
        router, handoffs, dbg = run_ablated_thread_nested(
            scen, only_missing("round_aware_state"))
        grading = grade_nested(scen, handoffs)
        if grading["full_correct"]:
            n_correct += 1
    # On the 3-scenario nested bank, dropping round_aware_state
    # collapses all three.
    assert n_correct == 0


def test_features_label_roundtrip():
    f = only_missing("typed_vocab")
    label = f.label()
    assert "-typed_vocab" in label
    assert "+bounded_witness" in label


def test_ablated_frozen_membership_is_null_control():
    bank = build_contested_bank(seed=35, distractors_per_role=4)
    n_correct = 0
    for scen in bank:
        router, handoffs, dbg = run_ablated_thread_contested(
            scen, only_missing("frozen_membership"))
        cue = decoder_from_handoffs_phase35(handoffs)
        ok = (cue["root_cause"] == scen.gold_root_cause
              and tuple(sorted(cue["services"]))
                  == tuple(sorted(scen.gold_services))
              and cue["remediation"] == scen.gold_remediation)
        if ok:
            n_correct += 1
    # Null control: no accuracy change from full.
    assert n_correct == len(bank)
