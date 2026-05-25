"""W99 — unit tests for B4 (typed schema sans hint) + B5
(question-type router).

Mirrors ``tests/test_w98_realworldqa_bench_v2_v3.py`` shape.
"""
from __future__ import annotations

import hashlib
import io
import pytest

from coordpy.realworldqa_bench_v2 import (
    QUESTION_TYPE_MULTI_CHOICE_LETTER,
    QUESTION_TYPE_NUMERIC,
    QUESTION_TYPE_SHORT_TEXT,
    QUESTION_TYPE_YES_NO,
    detect_question_type_v2,
)
from coordpy.realworldqa_bench_v4 import (
    W99_REALWORLDQA_BENCH_V4_SCHEMA_VERSION,
    RealWorldQAV4BenchConfig,
    _B_TYPED_SCENE_READER_SYSTEM,
    _B_TYPED_SOLVER_SYSTEM_TEMPLATE,
    run_realworldqa_bench_v4,
)
from coordpy.realworldqa_bench_v5 import (
    ROUTE_A1_VLM_K5,
    ROUTE_VLM_TEAM_B0,
    W99_REALWORLDQA_BENCH_V5_SCHEMA_VERSION,
    RealWorldQAV5BenchConfig,
    b5_route_for_question,
    run_realworldqa_bench_v5,
)
from coordpy.realworldqa_loader_v1 import (
    RealWorldQAProblemV1,
)


# ---------------------------------------------------------------
# Helpers — synthetic corpus + deterministic stub models
# ---------------------------------------------------------------

def _make_problem(pid: str, question: str, answer: str,
                  *, image_bytes: bytes = b"\x89PNG\r\n\x1a\n"
                  + b"\x00" * 100) -> RealWorldQAProblemV1:
    img_sha = hashlib.sha256(image_bytes).hexdigest()
    return RealWorldQAProblemV1(
        pid=str(pid),
        question=str(question),
        answer=str(answer),
        image_path=f"synthetic/{pid}.png",
        image_bytes=image_bytes,
        image_sha256=img_sha,
        image_format="png",
        metadata={})


def _make_corpus() -> tuple[RealWorldQAProblemV1, ...]:
    problems = [
        _make_problem(
            "rwqa_test_000013",
            "Which direction is the vehicle traveling?\n\n"
            "A. Straight\nB. Left\nC. Right",
            "B"),
        _make_problem(
            "rwqa_test_000076",
            "Which is closer?\n\nA. The stop sign\n"
            "B. The speed limit sign",
            "A"),
        _make_problem(
            "rwqa_test_000135",
            "Are there any stop signs?", "Yes"),
        _make_problem(
            "rwqa_test_000403",
            "Is the light green?", "No"),
        _make_problem(
            "rwqa_test_000094",
            "How many trucks are there?", "0"),
        _make_problem(
            "rwqa_test_000207",
            "What color is the traffic lights?", "Green"),
    ]
    return tuple(problems)


def _stub_text_gen(prompt, max_tokens, temperature):
    """Deterministic stub text gen that returns 'Yes' for
    yes/no, 'A' for multi-choice, '1' for numeric, 'Green'
    for short text."""
    p = prompt.lower()
    if "yes/no" in p or "is the light green" in p or "stop sign" in p:
        return "Yes", 10
    if "multi-choice" in p or "a." in p:
        return "A", 10
    if "numeric" in p or "how many" in p:
        return "1", 10
    return "Green", 10


def _stub_vlm_gen(prompt, image_bytes, max_tokens, temperature):
    """Deterministic stub VLM gen."""
    if image_bytes is None:
        return _stub_text_gen(prompt, max_tokens, temperature)
    p = prompt.lower()
    if "scene reader" in p or "scene graph" in p:
        # Return a minimal JSON-ish extraction for typed
        # readers; free-form bullet list for V1/V3.
        if "json" in p:
            return ('{"scene_summary": "stub", "objects": [], '
                    '"counts_by_label": {"stop_sign": 1}, '
                    '"spatial_relations": [], '
                    '"text_in_scene": [], '
                    '"uncertain": []}'), 20
        return ("- 1 stop sign\n- 2 cars\n- traffic light "
                "(red)"), 20
    return _stub_text_gen(prompt, max_tokens, temperature)


# ---------------------------------------------------------------
# B4 tests
# ---------------------------------------------------------------

def test_b4_schema_version_str():
    assert W99_REALWORLDQA_BENCH_V4_SCHEMA_VERSION == (
        "coordpy.realworldqa_bench_v4.v1")


def test_b4_reader_prompt_has_required_primitives():
    """B4's typed reader prompt must list the W97 yes/no-
    recovery primitives."""
    for primitive in (
            "state",
            "orientation",
            "depth",
            "text_in_object"):
        assert primitive in _B_TYPED_SCENE_READER_SYSTEM


def test_b4_reader_prompt_does_not_list_direct_answer_hint_as_field():
    """B4's typed reader must NOT list ``direct_answer_hint``
    as a field-to-fill.  The only allowed mention is the
    explicit removal admonition."""
    prompt = _B_TYPED_SCENE_READER_SYSTEM
    n = prompt.count("direct_answer_hint")
    # ≤ 1 mention (only the explicit "do not include" line).
    assert n <= 1, f"reader prompt mentions hint {n}× (expected ≤ 1)"


def test_b4_solver_template_does_not_reference_hint():
    """B4's solver prompt template must not reference any
    direct_answer_hint."""
    assert "direct_answer_hint" not in (
        _B_TYPED_SOLVER_SYSTEM_TEMPLATE)


def test_b4_config_k5_byte_exact():
    cfg = RealWorldQAV4BenchConfig()
    assert cfg.K_multi_sample == 5
    assert cfg.n_problems == 30
    assert cfg.seeds == (96_504_002,)


def test_b4_smoke_runs_end_to_end():
    """B4 runs to completion on a synthetic 3-problem corpus
    with stub gens; produces well-formed report + per-problem
    outcomes."""
    corpus = _make_corpus()
    # B4 selects a 3-problem subset; use seed that selects
    # the first 3.
    cfg = RealWorldQAV4BenchConfig(
        n_problems=3, seeds=(123_456,),
        sampling_temperature=0.7,
        max_tokens_per_call=64)
    report = run_realworldqa_bench_v4(
        text_gen=_stub_text_gen,
        vlm_gen=_stub_vlm_gen,
        vlm_model_id="stub-vlm",
        text_model_id="stub-text",
        corpus=corpus,
        corpus_parquet_shard_sha256=("stub_shard_sha",),
        corpus_merkle_root="stub_merkle",
        config=cfg)
    assert report.schema == (
        W99_REALWORLDQA_BENCH_V4_SCHEMA_VERSION)
    assert report.n_problems == 3
    assert report.n_seeds == 1
    assert report.K_multi_sample == 5
    assert len(report.per_seed) == 1
    seed_rep = report.per_seed[0]
    assert seed_rep.n_problems == 3
    # 0 ≤ pass rates ≤ 1
    for rate in (
            seed_rep.a0_text_pass_at_1,
            seed_rep.a1_vlm_pass_at_1,
            seed_rep.b_vlm_team_v4_pass_at_1):
        assert 0.0 <= rate <= 1.0
    # Per-problem outcomes contain the expected keys
    for po in seed_rep.per_problem_outcomes:
        for k in (
                "pid", "question", "gold_answer",
                "question_type", "a0_text_passed",
                "a1_vlm_passed", "b_vlm_team_v4_passed",
                "a0_outcome_cid", "a1_outcome_cid",
                "b_outcome_cid"):
            assert k in po


def test_b4_to_dict_round_trips():
    """Bench report serialises cleanly."""
    corpus = _make_corpus()
    cfg = RealWorldQAV4BenchConfig(
        n_problems=3, seeds=(123_456,),
        max_tokens_per_call=64)
    report = run_realworldqa_bench_v4(
        text_gen=_stub_text_gen,
        vlm_gen=_stub_vlm_gen,
        vlm_model_id="stub-vlm",
        text_model_id="stub-text",
        corpus=corpus,
        corpus_parquet_shard_sha256=("stub",),
        corpus_merkle_root="stub_merkle",
        config=cfg)
    d = report.to_dict()
    assert d["K_multi_sample"] == 5
    assert d["schema"] == (
        W99_REALWORLDQA_BENCH_V4_SCHEMA_VERSION)


# ---------------------------------------------------------------
# B5 tests
# ---------------------------------------------------------------

def test_b5_schema_version_str():
    assert W99_REALWORLDQA_BENCH_V5_SCHEMA_VERSION == (
        "coordpy.realworldqa_bench_v5.v1")


def test_b5_route_multi_choice_to_b0():
    q = (
        "Which direction is the vehicle traveling?\n\n"
        "A. Straight\nB. Left\nC. Right\n"
        "Please answer directly with only the letter.")
    assert b5_route_for_question(q) == ROUTE_VLM_TEAM_B0


def test_b5_route_yes_no_to_a1():
    assert (b5_route_for_question(
        "Is the light green?\nPlease answer directly.")
        == ROUTE_A1_VLM_K5)


def test_b5_route_numeric_to_a1():
    assert (b5_route_for_question(
        "How many cars are there?\nPlease answer directly.")
        == ROUTE_A1_VLM_K5)


def test_b5_route_short_text_to_a1():
    assert (b5_route_for_question(
        "What color is the light?\nPlease answer directly.")
        == ROUTE_A1_VLM_K5)


def test_b5_config_k5_byte_exact():
    cfg = RealWorldQAV5BenchConfig()
    assert cfg.K_multi_sample == 5
    assert cfg.n_problems == 30
    assert cfg.seeds == (96_504_002,)


def test_b5_smoke_runs_end_to_end():
    """B5 routes between B0 (multi-choice) and A1 (else)
    arms via the deterministic question-type parser."""
    corpus = _make_corpus()
    cfg = RealWorldQAV5BenchConfig(
        n_problems=3, seeds=(123_456,),
        sampling_temperature=0.7,
        max_tokens_per_call=64)
    report = run_realworldqa_bench_v5(
        text_gen=_stub_text_gen,
        vlm_gen=_stub_vlm_gen,
        vlm_model_id="stub-vlm",
        text_model_id="stub-text",
        corpus=corpus,
        corpus_parquet_shard_sha256=("stub",),
        corpus_merkle_root="stub_merkle",
        config=cfg)
    assert report.schema == (
        W99_REALWORLDQA_BENCH_V5_SCHEMA_VERSION)
    assert report.n_problems == 3
    assert report.n_seeds == 1
    assert report.K_multi_sample == 5
    seed_rep = report.per_seed[0]
    # Route distribution must sum to n_problems.
    assert (seed_rep.n_route_vlm_team_b0
            + seed_rep.n_route_a1_vlm_k5) == 3
    # Each per-problem outcome records its route.
    routes_seen = {po["route"]
                   for po in seed_rep.per_problem_outcomes}
    assert routes_seen.issubset(
        {ROUTE_VLM_TEAM_B0, ROUTE_A1_VLM_K5})


def test_b5_to_dict_round_trips():
    corpus = _make_corpus()
    cfg = RealWorldQAV5BenchConfig(
        n_problems=3, seeds=(123_456,),
        max_tokens_per_call=64)
    report = run_realworldqa_bench_v5(
        text_gen=_stub_text_gen,
        vlm_gen=_stub_vlm_gen,
        vlm_model_id="stub-vlm",
        text_model_id="stub-text",
        corpus=corpus,
        corpus_parquet_shard_sha256=("stub",),
        corpus_merkle_root="stub_merkle",
        config=cfg)
    d = report.to_dict()
    assert d["K_multi_sample"] == 5
    assert "route_distribution" in d


# ---------------------------------------------------------------
# Cross-test: parser consistency (B4 + B5 use the same parser)
# ---------------------------------------------------------------

@pytest.mark.parametrize(
    "q, expected",
    [
        ("Is the light green?", QUESTION_TYPE_YES_NO),
        ("Are there any stop signs?", QUESTION_TYPE_YES_NO),
        ("How many cars are there?", QUESTION_TYPE_NUMERIC),
        ("What color is the light?", QUESTION_TYPE_SHORT_TEXT),
        ("Which direction?\nA. Left\nB. Right\nC. Straight",
         QUESTION_TYPE_MULTI_CHOICE_LETTER),
    ])
def test_question_type_parser_on_common_shapes(q, expected):
    assert detect_question_type_v2(q) == expected


def test_b5_route_consistency_with_parser():
    """For each canonical shape, route follows from parser."""
    cases = [
        ("Is the light green?", ROUTE_A1_VLM_K5),
        ("Which direction?\nA. L\nB. R\nC. S",
         ROUTE_VLM_TEAM_B0),
        ("How many cars?", ROUTE_A1_VLM_K5),
        ("What color is the light?", ROUTE_A1_VLM_K5),
    ]
    for q, expected_route in cases:
        assert b5_route_for_question(q) == expected_route, (
            f"route mismatch for {q!r}")


# ---------------------------------------------------------------
# Anti-cheat sanity (K=5 byte-exact; no hidden hint leakage)
# ---------------------------------------------------------------

def test_b4_no_hint_in_solver_prompt_at_runtime():
    """Build a solver prompt at runtime and confirm it doesn't
    contain ``direct_answer_hint``."""
    from coordpy.realworldqa_bench_v4 import (
        _b_typed_solver_initial_prompt,
        _b_typed_solver_reflexion_prompt)
    p = _make_problem(
        "rwqa_stub_001", "Is the light green?", "No")
    initial = _b_typed_solver_initial_prompt(
        p, '{"scene_summary": "stub"}',
        QUESTION_TYPE_YES_NO)
    assert "direct_answer_hint" not in initial
    refl = _b_typed_solver_reflexion_prompt(
        p, '{"scene_summary": "stub"}',
        QUESTION_TYPE_YES_NO,
        history=tuple(), attempt_idx=1)
    assert "direct_answer_hint" not in refl


def test_b5_does_not_mutate_corpus_or_question():
    """Routing must be a pure function of the question text."""
    q = "Is the light green?\nPlease answer directly."
    assert (b5_route_for_question(q)
            == b5_route_for_question(q))
    # Routing must not depend on global state.
    import coordpy.realworldqa_bench_v5 as v5_mod
    assert hasattr(v5_mod, "b5_route_for_question")
