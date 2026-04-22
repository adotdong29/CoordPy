"""Phase 38 Part C — prompt-variant calibration tests."""

from __future__ import annotations

import pytest

from vision_mvp.core.llm_thread_replier import (
    LLMReplyConfig, LLMThreadReplier, parse_llm_reply_json,
)
from vision_mvp.core.prompt_variants import (
    ALL_PROMPT_VARIANTS, PROMPT_VARIANT_CONTRASTIVE,
    PROMPT_VARIANT_DEFAULT, PROMPT_VARIANT_FEW_SHOT,
    PROMPT_VARIANT_FORCED_ORDER, PROMPT_VARIANT_RUBRIC,
    VARIANT_BUILDERS, build_thread_reply_prompt_variant,
)


def test_all_variants_registered():
    assert set(VARIANT_BUILDERS.keys()) == set(
        ALL_PROMPT_VARIANTS)


def test_default_variant_builds():
    p = build_thread_reply_prompt_variant(
        variant=PROMPT_VARIANT_DEFAULT, role="network",
        candidate_role="network",
        candidate_kind="TLS_EXPIRED",
        candidate_payload="tls service=api reason=expired")
    assert "YOUR CLAIM" in p
    assert "REPLY FORMAT" in p


def test_contrastive_variant_has_decision_contrast():
    p = build_thread_reply_prompt_variant(
        variant=PROMPT_VARIANT_CONTRASTIVE, role="network",
        candidate_role="network",
        candidate_kind="TLS_EXPIRED",
        candidate_payload="tls reason=expired")
    assert "DECISION CONTRAST" in p


def test_few_shot_variant_has_examples():
    p = build_thread_reply_prompt_variant(
        variant=PROMPT_VARIANT_FEW_SHOT, role="network",
        candidate_role="network",
        candidate_kind="TLS_EXPIRED",
        candidate_payload="tls reason=expired")
    assert "EXAMPLES" in p
    assert "Example 1" in p
    assert "Example 2" in p


def test_rubric_variant_has_three_steps():
    p = build_thread_reply_prompt_variant(
        variant=PROMPT_VARIANT_RUBRIC, role="network",
        candidate_role="network",
        candidate_kind="TLS_EXPIRED",
        candidate_payload="tls reason=expired")
    assert "DECISION RUBRIC" in p
    assert "Step 1" in p
    assert "Step 2" in p
    assert "Step 3" in p


def test_forced_order_has_output_order():
    p = build_thread_reply_prompt_variant(
        variant=PROMPT_VARIANT_FORCED_ORDER, role="network",
        candidate_role="network",
        candidate_kind="TLS_EXPIRED",
        candidate_payload="tls reason=expired")
    assert "OUTPUT ORDER" in p


def test_variant_unknown_raises():
    with pytest.raises(ValueError):
        build_thread_reply_prompt_variant(
            variant="no_such_variant", role="r",
            candidate_role="r", candidate_kind="k",
            candidate_payload="p")


def test_all_variants_preserve_allowed_kinds():
    for v in ALL_PROMPT_VARIANTS:
        p = build_thread_reply_prompt_variant(
            variant=v, role="network",
            candidate_role="network",
            candidate_kind="TLS_EXPIRED",
            candidate_payload="tls reason=expired")
        assert "INDEPENDENT_ROOT" in p
        assert "DOWNSTREAM_SYMPTOM" in p
        assert "UNCERTAIN" in p


def test_all_variants_preserve_witness_cap():
    cfg = LLMReplyConfig(witness_token_cap=7)
    for v in ALL_PROMPT_VARIANTS:
        p = build_thread_reply_prompt_variant(
            variant=v, role="network",
            candidate_role="network",
            candidate_kind="TLS_EXPIRED",
            candidate_payload="tls reason=expired", cfg=cfg)
        # Cap is mentioned in the contract line.
        assert "7 whitespace tokens" in p


def test_variant_replier_integrates_with_parse():
    from vision_mvp.experiments.phase38_prompt_calibration import (
        BiasShiftMockReplier, VariantLLMThreadReplier,
    )
    stub = BiasShiftMockReplier(variant="rubric")
    inner = LLMThreadReplier(
        llm_call=stub,
        config=LLMReplyConfig(witness_token_cap=12),
        cache={})
    rep = VariantLLMThreadReplier(inner=inner, variant="rubric")

    class _FakeScen:
        scenario_id = "fake"

    out = rep(_FakeScen(), "network", "TLS_EXPIRED",
               "tls service=api reason=expired")
    assert out[0] in ("INDEPENDENT_ROOT",
                       "DOWNSTREAM_SYMPTOM", "UNCERTAIN")
    assert isinstance(out[2], bool)


def test_biasshift_mock_variant_changes_ir_rate():
    """Validate that the mock bias table is variant-sensitive.

    For a payload whose true class is IR, the ``rubric`` variant
    has p_ir_on_ir = 0.80 while ``default`` has p_ir_on_ir = 0.10.
    Over many calls the empirical rates should differ materially.
    """
    from vision_mvp.experiments.phase38_prompt_calibration import (
        BiasShiftMockReplier,
    )
    # Build synthetic prompts with a TLS_EXPIRED (IR) claim.
    prompt_template = (
        "YOUR CLAIM: [network/TLS_EXPIRED] tls "
        "service=api reason=expired cert\n")
    # Default variant.
    stub_d = BiasShiftMockReplier(variant="default")
    stub_r = BiasShiftMockReplier(variant="rubric")
    n_ir_d = 0
    n_ir_r = 0
    N = 200
    from vision_mvp.core.llm_thread_replier import (
        LLMReplyConfig, parse_llm_reply_json,
    )
    cfg = LLMReplyConfig(witness_token_cap=12)
    for i in range(N):
        p = prompt_template + f"call={i}\n"
        r_d = stub_d(p)
        r_r = stub_r(p)
        k_d, _, wf_d = parse_llm_reply_json(r_d, cfg)
        k_r, _, wf_r = parse_llm_reply_json(r_r, cfg)
        if wf_d and k_d == "INDEPENDENT_ROOT":
            n_ir_d += 1
        if wf_r and k_r == "INDEPENDENT_ROOT":
            n_ir_r += 1
    # Rubric should emit IR much more often than default.
    assert n_ir_r > n_ir_d + N * 0.3
