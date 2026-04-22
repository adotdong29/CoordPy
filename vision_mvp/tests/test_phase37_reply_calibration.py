"""Phase 37 Part A — reply-calibration wrapper unit tests."""

from __future__ import annotations

import pytest

from vision_mvp.core.dynamic_comm import (
    REPLY_DOWNSTREAM_SYMPTOM, REPLY_INDEPENDENT_ROOT,
    REPLY_UNCERTAIN,
)
from vision_mvp.core.llm_thread_replier import (
    DeterministicMockReplier, LLMReplyConfig, LLMThreadReplier,
)
from vision_mvp.core.reply_calibration import (
    CAL_CORRECT, CAL_MALFORMED, CAL_SEM_ROOT_AS_SYMPTOM,
    CAL_SEM_ROOT_AS_UNCERTAIN, CAL_SEM_SYMPTOM_AS_ROOT,
    CalibratingReplier, ReplyCalibrationReport,
    causality_extractor_from_calibrating_replier,
)


class _FakeScenario:
    scenario_id = "fake_s"


def _oracle_for(mapping):
    def _oracle(scenario, role, kind, payload):
        return mapping.get((role, kind), "UNCERTAIN")
    return _oracle


def test_calibration_correct_bucket():
    # Oracle = IR, replier = IR → CAL_CORRECT.
    inner = LLMThreadReplier(
        llm_call=DeterministicMockReplier(
            kind_replies={("db_admin", "DEADLOCK_SUSPECTED"):
                               REPLY_INDEPENDENT_ROOT}),
        config=LLMReplyConfig(witness_token_cap=12),
    )
    oracle = _oracle_for({
        ("db_admin", "DEADLOCK_SUSPECTED"): "INDEPENDENT_ROOT"})
    wrapper = CalibratingReplier(inner=inner, oracle=oracle)
    scenario = _FakeScenario()
    rk, wit, wf = wrapper(scenario, "db_admin",
                           "DEADLOCK_SUSPECTED",
                           "deadlock orders_payments")
    assert wf
    assert rk == REPLY_INDEPENDENT_ROOT
    assert wrapper.report.buckets[CAL_CORRECT] == 1
    assert wrapper.report.rates()["correct_rate"] == 1.0
    assert wrapper.report.rates()["semantic_wrong_rate"] == 0.0


def test_calibration_sem_root_as_symptom_bucket():
    # Oracle = IR, replier = DOWNSTREAM → root_as_symptom.
    inner = LLMThreadReplier(
        llm_call=DeterministicMockReplier(
            kind_replies={("network", "TLS_EXPIRED"):
                               REPLY_DOWNSTREAM_SYMPTOM}),
        config=LLMReplyConfig(witness_token_cap=12),
    )
    oracle = _oracle_for({
        ("network", "TLS_EXPIRED"): "INDEPENDENT_ROOT"})
    wrapper = CalibratingReplier(inner=inner, oracle=oracle)
    scenario = _FakeScenario()
    wrapper(scenario, "network", "TLS_EXPIRED",
            "tls service=api reason=expired")
    assert wrapper.report.buckets[CAL_SEM_ROOT_AS_SYMPTOM] == 1
    rates = wrapper.report.rates()
    assert rates["semantic_wrong_rate"] == 1.0
    assert rates["effective_mislabel_rate"] == 1.0


def test_calibration_sem_root_as_uncertain_bucket():
    # Oracle = IR, replier = UNCERTAIN → root_as_uncertain.
    inner = LLMThreadReplier(
        llm_call=DeterministicMockReplier(
            kind_replies={("sysadmin", "OOM_KILL"):
                               REPLY_UNCERTAIN}),
        config=LLMReplyConfig(witness_token_cap=12),
    )
    oracle = _oracle_for({
        ("sysadmin", "OOM_KILL"): "INDEPENDENT_ROOT"})
    wrapper = CalibratingReplier(inner=inner, oracle=oracle)
    scenario = _FakeScenario()
    wrapper(scenario, "sysadmin", "OOM_KILL",
            "oom_kill pid=42 service=api")
    assert wrapper.report.buckets[CAL_SEM_ROOT_AS_UNCERTAIN] == 1
    assert (wrapper.report.rates()
            ["effective_drop_prob_conditional_on_ir"]) == 1.0


def test_calibration_malformed_bucket():
    # Replier emits prose (malformed_prob=1.0) — always malformed.
    inner = LLMThreadReplier(
        llm_call=DeterministicMockReplier(
            kind_replies={("db_admin", "DEADLOCK_SUSPECTED"):
                               REPLY_INDEPENDENT_ROOT},
            malformed_prob=1.0),
        config=LLMReplyConfig(witness_token_cap=12),
    )
    oracle = _oracle_for({
        ("db_admin", "DEADLOCK_SUSPECTED"): "INDEPENDENT_ROOT"})
    wrapper = CalibratingReplier(inner=inner, oracle=oracle)
    scenario = _FakeScenario()
    _, _, wf = wrapper(scenario, "db_admin",
                        "DEADLOCK_SUSPECTED",
                        "deadlock orders_payments")
    assert wf is False
    assert wrapper.report.buckets[CAL_MALFORMED] == 1


def test_calibration_normalisation_treats_agree_as_uncertain():
    # Replier emits AGREE — normalised to UNCERTAIN by the
    # bucket classifier.
    inner = LLMThreadReplier(
        llm_call=DeterministicMockReplier(
            kind_replies={("db_admin", "DEADLOCK_SUSPECTED"):
                               "AGREE"}),
        config=LLMReplyConfig(
            witness_token_cap=12,
            allowed_reply_kinds=(REPLY_INDEPENDENT_ROOT,
                                  REPLY_DOWNSTREAM_SYMPTOM,
                                  REPLY_UNCERTAIN, "AGREE"),
        ),
    )
    oracle = _oracle_for({
        ("db_admin", "DEADLOCK_SUSPECTED"): "INDEPENDENT_ROOT"})
    wrapper = CalibratingReplier(inner=inner, oracle=oracle)
    scenario = _FakeScenario()
    wrapper(scenario, "db_admin", "DEADLOCK_SUSPECTED",
            "deadlock orders_payments")
    # AGREE is not in the trio; classifier rounds to UNCERTAIN.
    # Oracle = IR → bucket = sem_root_as_uncertain.
    assert wrapper.report.buckets[CAL_SEM_ROOT_AS_UNCERTAIN] == 1


def test_causality_extractor_from_calibrating_replier_shape():
    inner = LLMThreadReplier(
        llm_call=DeterministicMockReplier(
            kind_replies={("db_admin", "DEADLOCK_SUSPECTED"):
                               REPLY_INDEPENDENT_ROOT}),
        config=LLMReplyConfig(witness_token_cap=12),
    )
    oracle = _oracle_for({
        ("db_admin", "DEADLOCK_SUSPECTED"): "INDEPENDENT_ROOT"})
    wrapper = CalibratingReplier(inner=inner, oracle=oracle)
    ext = causality_extractor_from_calibrating_replier(wrapper)
    scenario = _FakeScenario()
    out = ext(scenario, "db_admin", "DEADLOCK_SUSPECTED",
              "deadlock orders_payments")
    assert out == "INDEPENDENT_ROOT"
    # The extractor call should have populated the report.
    assert wrapper.report.n_calls >= 1


def test_rates_dict_shape():
    rep = ReplyCalibrationReport()
    rates = rep.rates()
    assert "n_calls" in rates
    assert "correct_rate" in rates
    assert "semantic_wrong_rate" in rates
    assert "effective_drop_prob_conditional_on_ir" in rates
    assert "effective_mislabel_rate" in rates
