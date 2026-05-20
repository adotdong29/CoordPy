"""W86 — long-context hidden-state intercept bench surface tests.

The end-to-end frontier run is in the Vertex AI execution and is
recorded in
``results/w86/.../27_long_context_intercept.json``. This file
covers the structural surface: prompt-corpus determinism, the
honest lean-env return shape, the schema CID.
"""
from __future__ import annotations


def test_w86_prompt_builder_deterministic():
    from coordpy.long_context_intercept_bench_v1 import (
        build_long_haystack_token_prompt_v1,
    )
    p1 = build_long_haystack_token_prompt_v1(
        n_tokens=1000, needle_position=500,
        needle_value=42, seed=1)
    p2 = build_long_haystack_token_prompt_v1(
        n_tokens=1000, needle_position=500,
        needle_value=42, seed=1)
    assert p1 == p2


def test_w86_prompt_builder_changes_with_seed():
    from coordpy.long_context_intercept_bench_v1 import (
        build_long_haystack_token_prompt_v1,
    )
    p1 = build_long_haystack_token_prompt_v1(
        n_tokens=1000, needle_position=500,
        needle_value=42, seed=1)
    p2 = build_long_haystack_token_prompt_v1(
        n_tokens=1000, needle_position=500,
        needle_value=42, seed=2)
    assert p1 != p2


def test_w86_prompt_contains_needle_marker():
    from coordpy.long_context_intercept_bench_v1 import (
        build_long_haystack_token_prompt_v1,
    )
    prompt = build_long_haystack_token_prompt_v1(
        n_tokens=200, needle_position=100,
        needle_value=999, seed=1)
    assert "NEEDLE_VALUE=999" in prompt


def test_w86_prompt_no_short_snippet_repetition():
    """Anti-cheat: every haystack identifier must be unique."""
    from coordpy.long_context_intercept_bench_v1 import (
        build_long_haystack_token_prompt_v1,
    )
    prompt = build_long_haystack_token_prompt_v1(
        n_tokens=400, needle_position=200,
        needle_value=42, seed=1)
    items = [
        tok for tok in prompt.split()
        if tok.startswith("item-")]
    assert len(items) > 0
    assert len(set(items)) == len(items)


def test_w86_bench_returns_honest_unavailable_in_lean_env():
    """Without torch/transformers the bench must NOT fake a
    positive intercept-moves-CID result."""
    from coordpy.long_context_intercept_bench_v1 import (
        run_long_context_intercept_bench_v1,
    )
    rep = run_long_context_intercept_bench_v1(
        model_name="fake/model",
        horizons=(32_768,),
    )
    assert rep.transformers_available is False
    assert rep.intercept_moves_cid_at_min_32k is False
    assert rep.n_horizons_pass == 0
    # Must still emit a content-addressed CID for the report.
    assert isinstance(rep.cid(), str)
    assert len(rep.cid()) == 64


def test_w86_bench_report_round_trips_to_dict():
    from coordpy.long_context_intercept_bench_v1 import (
        run_long_context_intercept_bench_v1,
    )
    rep = run_long_context_intercept_bench_v1(
        model_name="fake/model",
    )
    d = rep.to_dict()
    assert d["schema"] == (
        "coordpy.long_context_intercept_bench_v1.v1")
    assert d["transformers_available"] is False
    assert d["intercept_moves_cid_at_min_32k"] is False
    assert d["n_horizons_pass"] == 0


def test_w86_bench_schema_version_present():
    from coordpy.long_context_intercept_bench_v1 import (
        W86_LONG_CONTEXT_INTERCEPT_V1_SCHEMA_VERSION,
    )
    assert W86_LONG_CONTEXT_INTERCEPT_V1_SCHEMA_VERSION == (
        "coordpy.long_context_intercept_bench_v1.v1")
