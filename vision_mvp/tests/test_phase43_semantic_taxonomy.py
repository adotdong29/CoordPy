"""Phase 43 tests — semantic failure taxonomy + frontier-headroom driver.

The Phase-43 taxonomy module is structural: every (buggy_source,
gold_patch, proposed_patch, error_kind, test_passed) tuple yields
exactly one label from `ALL_SEMANTIC_LABELS`. These tests fix the
label assignment for each category and verify the classifier's
priority order is stable across edge cases.

The Phase-43 analysis driver is exercised on a synthetic artifact
and the canonical Phase-42 artifact shape.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

from vision_mvp.tasks.swe_semantic_taxonomy import (
    SEM_INCOMPLETE_MULTI_HUNK, SEM_NO_MATCH_RESIDUAL, SEM_OK,
    SEM_PARSE_FAIL, SEM_RIGHT_SITE_WRONG_LOGIC,
    SEM_STRUCTURAL_SEMANTIC_INERT, SEM_SYNTAX_INVALID,
    SEM_TEST_OVERFIT, SEM_WRONG_EDIT_SITE, ALL_SEMANTIC_LABELS,
    SemanticCounter, classify_semantic_outcome,
)


# ---------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------


_SRC = (
    "def factorial(n):\n"
    "    result = 0\n"  # bug
    "    for i in range(1, n + 1):\n"
    "        result *= i\n"
    "    return result\n"
)


_GOLD_SINGLE = (("    result = 0\n", "    result = 1\n"),)


_GOLD_MULTI = (
    ("    result = 0\n", "    result = 1\n"),
    ("    return result\n", "    return int(result)\n"),
)


# ---------------------------------------------------------------
# Pass classification
# ---------------------------------------------------------------


def test_sem_ok_when_test_passed():
    label = classify_semantic_outcome(
        buggy_source=_SRC, gold_patch=_GOLD_SINGLE,
        proposed_patch=_GOLD_SINGLE,
        error_kind="", test_passed=True,
    )
    assert label == SEM_OK


def test_sem_ok_dominates_other_signals():
    # Even if error_kind were non-empty, test_passed=True wins.
    label = classify_semantic_outcome(
        buggy_source=_SRC, gold_patch=_GOLD_SINGLE,
        proposed_patch=_GOLD_SINGLE,
        error_kind="test_assert", test_passed=True,
    )
    assert label == SEM_OK


# ---------------------------------------------------------------
# Parse failure
# ---------------------------------------------------------------


def test_sem_parse_fail_when_no_substitutions():
    label = classify_semantic_outcome(
        buggy_source=_SRC, gold_patch=_GOLD_SINGLE,
        proposed_patch=(),
        error_kind="patch_no_match", test_passed=False,
    )
    assert label == SEM_PARSE_FAIL


# ---------------------------------------------------------------
# No-match residual
# ---------------------------------------------------------------


def test_sem_no_match_residual_on_patch_no_match():
    # Proposed is non-empty but apply_patch rejected it.
    label = classify_semantic_outcome(
        buggy_source=_SRC, gold_patch=_GOLD_SINGLE,
        proposed_patch=(("some unmatched old", "replacement"),),
        error_kind="patch_no_match", test_passed=False,
    )
    assert label == SEM_NO_MATCH_RESIDUAL


# ---------------------------------------------------------------
# Syntax-invalid
# ---------------------------------------------------------------


def test_sem_syntax_invalid_on_syntax_error():
    label = classify_semantic_outcome(
        buggy_source=_SRC, gold_patch=_GOLD_SINGLE,
        proposed_patch=_GOLD_SINGLE,
        error_kind="syntax", test_passed=False,
    )
    assert label == SEM_SYNTAX_INVALID


# ---------------------------------------------------------------
# Incomplete multi-hunk
# ---------------------------------------------------------------


def test_sem_incomplete_multi_hunk_when_gold_has_more():
    # Gold has 2 hunks, proposed has only 1.
    proposed = (("    result = 0\n", "    result = 1\n"),)
    label = classify_semantic_outcome(
        buggy_source=_SRC, gold_patch=_GOLD_MULTI,
        proposed_patch=proposed,
        error_kind="test_assert", test_passed=False,
    )
    assert label == SEM_INCOMPLETE_MULTI_HUNK


# ---------------------------------------------------------------
# Wrong edit site
# ---------------------------------------------------------------


def test_sem_wrong_edit_site_when_old_anchors_disagree():
    # Proposed OLD shares no normalised lines with any gold OLD.
    proposed = (("    for i in range(1, n + 1):\n"
                 "        result *= i\n",
                 "    for i in range(n):\n"
                 "        result *= i\n"),)
    label = classify_semantic_outcome(
        buggy_source=_SRC, gold_patch=_GOLD_SINGLE,
        proposed_patch=proposed,
        error_kind="test_assert", test_passed=False,
    )
    # 1-line OLD sharing with gold is below threshold; classifier
    # returns SEM_WRONG_EDIT_SITE.
    assert label == SEM_WRONG_EDIT_SITE


# ---------------------------------------------------------------
# Right site, wrong logic
# ---------------------------------------------------------------


def test_sem_right_site_wrong_logic_under_assert():
    # Proposed matches gold OLD (same lines) but NEW differs.
    proposed = (
        ("    result = 0\n"
         "    for i in range(1, n + 1):\n",
         "    result = -1\n"  # wrong fix — negative seed
         "    for i in range(1, n + 1):\n"),
    )
    # Shares ≥ 2 normalised lines with gold OLD hunk window (including
    # context). Gold's OLD is just a single line; pad it.
    gold = (("    result = 0\n"
              "    for i in range(1, n + 1):\n",
              "    result = 1\n"
              "    for i in range(1, n + 1):\n"),)
    label = classify_semantic_outcome(
        buggy_source=_SRC, gold_patch=gold,
        proposed_patch=proposed,
        error_kind="test_assert", test_passed=False,
        error_detail="",  # no overfit trigger — classifier returns wrong_logic
    )
    # With len(proposed) == len(gold) and NEW differs AND error_detail is
    # non-empty via the "assert" trigger path, classifier routes to
    # SEM_TEST_OVERFIT. Here we pass empty error_detail so we should get
    # SEM_RIGHT_SITE_WRONG_LOGIC — but the classifier's _overfit_proxy
    # heuristic checks error_detail truthiness; empty → fallback to
    # SEM_RIGHT_SITE_WRONG_LOGIC.
    assert label == SEM_RIGHT_SITE_WRONG_LOGIC


# ---------------------------------------------------------------
# Test overfit
# ---------------------------------------------------------------


def test_sem_test_overfit_on_assert_with_detail():
    proposed = (
        ("    result = 0\n"
         "    for i in range(1, n + 1):\n",
         "    result = 1 if n > 0 else 0\n"  # overfits n>0
         "    for i in range(1, n + 1):\n"),
    )
    gold = (("    result = 0\n"
              "    for i in range(1, n + 1):\n",
              "    result = 1\n"
              "    for i in range(1, n + 1):\n"),)
    label = classify_semantic_outcome(
        buggy_source=_SRC, gold_patch=gold,
        proposed_patch=proposed,
        error_kind="test_assert", test_passed=False,
        error_detail="assert module.factorial(0) == 1",
    )
    assert label == SEM_TEST_OVERFIT


# ---------------------------------------------------------------
# Structural semantic inert
# ---------------------------------------------------------------


def test_sem_structural_semantic_inert_on_test_exception():
    # Applies cleanly on the right site but throws at runtime.
    gold = (("    result = 0\n"
              "    for i in range(1, n + 1):\n",
              "    result = 1\n"
              "    for i in range(1, n + 1):\n"),)
    proposed = (
        ("    result = 0\n"
         "    for i in range(1, n + 1):\n",
         "    result = 'one'\n"  # TypeError on *= i
         "    for i in range(1, n + 1):\n"),
    )
    label = classify_semantic_outcome(
        buggy_source=_SRC, gold_patch=gold,
        proposed_patch=proposed,
        error_kind="test_exception", test_passed=False,
    )
    assert label == SEM_STRUCTURAL_SEMANTIC_INERT


# ---------------------------------------------------------------
# Taxonomy is exhaustive
# ---------------------------------------------------------------


def test_all_labels_exercised_in_fixtures():
    # Every label except SEM_TEST_OVERFIT and SEM_STRUCTURAL_SEMANTIC_INERT
    # is exercised above; we cover those two in the dedicated tests.
    exercised = {
        SEM_OK, SEM_PARSE_FAIL, SEM_NO_MATCH_RESIDUAL,
        SEM_SYNTAX_INVALID, SEM_INCOMPLETE_MULTI_HUNK,
        SEM_WRONG_EDIT_SITE, SEM_RIGHT_SITE_WRONG_LOGIC,
        SEM_TEST_OVERFIT, SEM_STRUCTURAL_SEMANTIC_INERT,
    }
    assert exercised == set(ALL_SEMANTIC_LABELS)


# ---------------------------------------------------------------
# SemanticCounter aggregation
# ---------------------------------------------------------------


def test_semantic_counter_aggregates_per_strategy():
    ctr = SemanticCounter()
    ctr.record(SEM_OK, strategy="substrate")
    ctr.record(SEM_OK, strategy="substrate")
    ctr.record(SEM_WRONG_EDIT_SITE, strategy="substrate")
    ctr.record(SEM_OK, strategy="naive")
    ctr.record(SEM_NO_MATCH_RESIDUAL, strategy="naive")

    d = ctr.as_dict()
    assert d["n_records"] == 5
    assert d["by_strategy"]["substrate"][SEM_OK] == 2
    assert d["by_strategy"]["substrate"][SEM_WRONG_EDIT_SITE] == 1
    assert d["by_strategy"]["naive"][SEM_OK] == 1
    assert ctr.pass_rate(strategy="substrate") == 2 / 3
    assert ctr.pass_rate(strategy="naive") == 1 / 2
    assert ctr.pass_rate() == 3 / 5


def test_semantic_counter_failure_mix_excludes_ok():
    ctr = SemanticCounter()
    ctr.record(SEM_OK, strategy="s")
    ctr.record(SEM_OK, strategy="s")
    ctr.record(SEM_WRONG_EDIT_SITE, strategy="s")
    ctr.record(SEM_NO_MATCH_RESIDUAL, strategy="s")
    mix = ctr.failure_mix(strategy="s")
    assert SEM_OK not in mix
    # Two failures, each 50 %.
    assert mix[SEM_WRONG_EDIT_SITE] == 0.5
    assert mix[SEM_NO_MATCH_RESIDUAL] == 0.5


# ---------------------------------------------------------------
# Public-style loader self-test (uses the real 57-instance bank)
# ---------------------------------------------------------------


def test_phase43_public_style_loader_selftest_runs():
    from vision_mvp.experiments.phase43_frontier_headroom import (
        verify_public_style_loader,
    )
    # The bundled 57-instance bank is SWE-bench-Lite-shape; pointing
    # the Phase-43 loader self-test at it must succeed on every
    # instance in the limit range under strict matcher.
    bank_path = os.path.join(
        os.path.dirname(__file__), "..", "tasks", "data",
        "swe_lite_style_bank.jsonl")
    out = verify_public_style_loader(bank_path, limit=10)
    assert out["ok"] is True
    assert out["n"] == 10
    assert out["n_parsed"] == 10
    assert out["n_oracle_pass"] == 10


def test_phase43_public_style_loader_full_bank():
    from vision_mvp.experiments.phase43_frontier_headroom import (
        verify_public_style_loader,
    )
    bank_path = os.path.join(
        os.path.dirname(__file__), "..", "tasks", "data",
        "swe_lite_style_bank.jsonl")
    out = verify_public_style_loader(bank_path, limit=1000)
    # 57 instances in the bundled bank; every one oracle-saturates.
    assert out["n"] == 57
    assert out["n_oracle_pass"] == 57
    assert out["ok"] is True


# ---------------------------------------------------------------
# LLMClient think kwarg — Phase 43 extension
# ---------------------------------------------------------------


def test_phase43_partial_delim_close_at_eos_recovers_without_garbage():
    """Phase 43 § D.4 regression — qwen3.5:35b closes the NEW block
    with ``<<`` (two angle brackets) instead of the canonical
    ``<<<``. Before the Phase-43 fix, the loose-delimiter recovery
    kept ``<<`` in the NEW payload, producing a syntactically-broken
    patched file. The extended ``_strip_trailing_prose`` pattern
    ``\\n\\s*<{2,4}\\s*\\Z`` removes a trailing partial or full
    delimiter so the recovered NEW payload is clean code."""
    from vision_mvp.tasks.swe_patch_parser import (
        PARSER_ROBUST, parse_patch_block, RECOVERY_CLOSED_AT_EOS,
    )
    # Exact 35B failure shape observed on ext-calc-001.
    text_35b_shape = (
        "OLD>>>\n"
        "def factorial(n):\n"
        "    result = 0\n"
        "    for i in range(1, n+1):\n"
        "        result *= i\n"
        "    return result\n"
        "<<<NEW>>>\n"
        "def factorial(n):\n"
        "    result = 1\n"
        "    for i in range(1, n+1):\n"
        "        result *= i\n"
        "    return result\n"
        "<<"
    )
    outcome = parse_patch_block(text_35b_shape, mode=PARSER_ROBUST)
    assert outcome.ok is True
    assert outcome.recovery == RECOVERY_CLOSED_AT_EOS
    assert len(outcome.substitutions) == 1
    old, new = outcome.substitutions[0]
    # The NEW payload must NOT contain the trailing ``<<`` which
    # would produce a syntax error when applied.
    assert "<<" not in new, f"trailing partial delimiter leaked into NEW: {new!r}"
    assert new.strip().endswith("return result")


def test_phase43_partial_delim_variants_all_stripped():
    """Verify the ``<{2,4}`` pattern handles 2/3/4 trailing ``<``
    characters — models have been observed emitting all three
    variants when truncated."""
    from vision_mvp.tasks.swe_patch_parser import (
        PARSER_ROBUST, parse_patch_block, RECOVERY_CLOSED_AT_EOS,
    )
    for tail in ("<<", "<<<", "<<<<"):
        text = (
            "OLD>>>\n"
            "x = 0\n"
            "<<<NEW>>>\n"
            "x = 1\n"
            f"{tail}"
        )
        outcome = parse_patch_block(text, mode=PARSER_ROBUST)
        assert outcome.ok is True, f"tail={tail!r} failed to recover"
        new = outcome.substitutions[0][1]
        assert "<" not in new, f"tail={tail!r} leaked {new!r}"


def test_llm_client_think_field_passthrough():
    """When ``think`` is None (default), the payload has no ``think``
    key; when ``think`` is False/True, the payload includes it. This
    preserves Phase 42 byte-for-byte semantics on the default path
    and enables the Phase 43 thinking-model opt-out."""
    import json as _json
    from vision_mvp.core import llm_client as _lc

    captured: dict = {}

    def _fake_post(path, payload, timeout, base_url=None):
        captured.clear()
        captured.update(payload)
        return {"response": "hi",
                 "prompt_eval_count": 1, "eval_count": 1}

    orig = _lc._post
    _lc._post = _fake_post
    try:
        c = _lc.LLMClient(model="x")
        c.generate("hello", max_tokens=5)
        assert "think" not in captured

        c2 = _lc.LLMClient(model="x", think=False)
        c2.generate("hello", max_tokens=5)
        assert captured.get("think") is False

        c3 = _lc.LLMClient(model="x", think=True)
        c3.generate("hello", max_tokens=5)
        assert captured.get("think") is True
    finally:
        _lc._post = orig
