"""W95 — MathVista loader / executor / preflight CI tests.

These tests use synthetic corpora and do NOT hit the network or
NIM.  The live parquet fetch + 4-probe run is covered by
`scripts/run_w95_mathvista_preflight.py` and is exercised in
`docs/RESULTS_W95_MATHVISTA_PREFLIGHT_V1.md`."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from coordpy.mathvista_executor_v1 import (
    W95_MATHVISTA_EXECUTOR_V1_SCHEMA_VERSION,
    evaluate_answer_v1,
    executor_self_test_on_gold_v1,
)
from coordpy.mathvista_loader_v1 import (
    MATHVISTA_TESTMINI_EXPECTED_N_PROBLEMS,
    MATHVISTA_TESTMINI_EXPECTED_PARQUET_BYTES_LOWER,
    MATHVISTA_TESTMINI_EXPECTED_PARQUET_BYTES_UPPER,
    MATHVISTA_TESTMINI_PARQUET_URL,
    MathVistaCorpusManifestV1,
    MathVistaProblemV1,
    W95_MATHVISTA_LOADER_V1_SCHEMA_VERSION,
    compute_corpus_merkle_root_v1,
    manifest_for_corpus_v1,
    select_mathvista_subset_v1,
)
from coordpy.mathvista_preflight_v1 import (
    MATHVISTA_PUBLISHED_SOTA_SINGLE_SHOT_BY_MODEL,
    W95_MATHVISTA_PREFLIGHT_V1_SCHEMA_VERSION,
    estimate_a1_k5_pass_rate_v1,
    probe_a1_failure_residual_v1,
    probe_corpus_integrity_v1,
    probe_decomposition_argument_v1,
    probe_executor_self_test_v1,
    run_mathvista_preflight_v1,
)


_FAKE_PNG = b"\x89PNG\r\n\x1a\n" + b"fake-png-payload-zzz"
_FAKE_PNG_SHA = hashlib.sha256(_FAKE_PNG).hexdigest()


def _make_mc(pid: str, gold: str, choices: tuple[str, ...],
             *, task: str = "figure_qa") -> MathVistaProblemV1:
    return MathVistaProblemV1(
        pid=pid, question="Q?", query="Q?", choices=choices,
        answer=gold, answer_type="text",
        question_type="multi_choice", unit="", precision=0.0,
        image_bytes=_FAKE_PNG, image_sha256=_FAKE_PNG_SHA,
        image_format="png",
        metadata={"task": task, "category": "visual",
                  "skills": [task]})


def _make_num(pid: str, gold: str, *, precision: float = 0.0,
              answer_type: str = "integer",
              task: str = "geometry_problem_solving",
              ) -> MathVistaProblemV1:
    return MathVistaProblemV1(
        pid=pid, question="Q?", query="Q?", choices=(),
        answer=gold, answer_type=answer_type,
        question_type="free_form", unit="",
        precision=float(precision),
        image_bytes=_FAKE_PNG, image_sha256=_FAKE_PNG_SHA,
        image_format="png",
        metadata={"task": task, "category": "math",
                  "skills": [task]})


def _make_text(pid: str, gold: str,
               *, task: str = "math_word_problem",
               ) -> MathVistaProblemV1:
    return MathVistaProblemV1(
        pid=pid, question="Q?", query="Q?", choices=(),
        answer=gold, answer_type="text",
        question_type="free_form", unit="", precision=0.0,
        image_bytes=_FAKE_PNG, image_sha256=_FAKE_PNG_SHA,
        image_format="png",
        metadata={"task": task, "category": "text",
                  "skills": [task]})


# ---------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------

def test_w95_loader_schema_version():
    assert (
        W95_MATHVISTA_LOADER_V1_SCHEMA_VERSION
        == "coordpy.mathvista_loader_v1.v1")


def test_w95_executor_schema_version():
    assert (
        W95_MATHVISTA_EXECUTOR_V1_SCHEMA_VERSION
        == "coordpy.mathvista_executor_v1.v1")


def test_w95_preflight_schema_version():
    assert (
        W95_MATHVISTA_PREFLIGHT_V1_SCHEMA_VERSION
        == "coordpy.mathvista_preflight_v1.v1")


def test_w95_testmini_expected_constants():
    assert MATHVISTA_TESTMINI_EXPECTED_N_PROBLEMS == 1000
    assert (MATHVISTA_TESTMINI_EXPECTED_PARQUET_BYTES_LOWER
            < MATHVISTA_TESTMINI_EXPECTED_PARQUET_BYTES_UPPER)


def test_w95_loader_url_points_to_huggingface_resolve():
    assert MATHVISTA_TESTMINI_PARQUET_URL.startswith(
        "https://huggingface.co/datasets/AI4Math/MathVista/")
    assert "resolve/main/data/testmini" in (
        MATHVISTA_TESTMINI_PARQUET_URL)


# ---------------------------------------------------------------
# Executor — multi-choice
# ---------------------------------------------------------------

def test_w95_executor_multi_choice_letter():
    p = _make_mc("p_mc1", "dog", ("cat", "dog"))
    v = evaluate_answer_v1(prediction="B", problem=p)
    assert v.passed
    assert v.matched_rule == "multi_choice_letter"


def test_w95_executor_multi_choice_text_exact_correct():
    p = _make_mc("p_mc2", "dog", ("cat", "dog"))
    v = evaluate_answer_v1(prediction="dog", problem=p)
    assert v.passed
    assert v.matched_rule == "multi_choice_text"


def test_w95_executor_multi_choice_text_exact_wrong():
    p = _make_mc("p_mc3", "dog", ("cat", "dog"))
    v = evaluate_answer_v1(prediction="cat", problem=p)
    assert not v.passed
    assert v.matched_rule == "multi_choice_text_wrong"


def test_w95_executor_multi_choice_letter_wrong_letter():
    p = _make_mc("p_mc4", "dog", ("cat", "dog"))
    v = evaluate_answer_v1(prediction="A", problem=p)
    assert not v.passed


def test_w95_executor_multi_choice_unique_contained():
    p = _make_mc("p_mc5", "dog", ("cat", "dog"))
    v = evaluate_answer_v1(
        prediction="The answer must be dog because of fur.",
        problem=p)
    assert v.passed
    assert v.matched_rule == "multi_choice_unique_contained"


# ---------------------------------------------------------------
# Executor — numeric
# ---------------------------------------------------------------

def test_w95_executor_integer_correct():
    p = _make_num("p_int1", "42", precision=0.0,
                  answer_type="integer")
    v = evaluate_answer_v1(prediction="42", problem=p)
    assert v.passed


def test_w95_executor_integer_with_prose_correct():
    p = _make_num("p_int2", "42", precision=0.0,
                  answer_type="integer")
    v = evaluate_answer_v1(
        prediction="The answer is 42.", problem=p)
    assert v.passed


def test_w95_executor_integer_wrong():
    p = _make_num("p_int3", "42", precision=0.0,
                  answer_type="integer")
    v = evaluate_answer_v1(prediction="41", problem=p)
    assert not v.passed


def test_w95_executor_float_precision2_correct():
    p = _make_num("p_flt1", "3.14", precision=2.0,
                  answer_type="float")
    v = evaluate_answer_v1(prediction="3.14", problem=p)
    assert v.passed


def test_w95_executor_float_precision2_rounded_correct():
    p = _make_num("p_flt2", "3.14", precision=2.0,
                  answer_type="float")
    # 3.144 → rounded to 2 decimals → 3.14 — passes.
    v = evaluate_answer_v1(prediction="3.144", problem=p)
    assert v.passed


def test_w95_executor_float_precision2_off_by_one_wrong():
    p = _make_num("p_flt3", "3.14", precision=2.0,
                  answer_type="float")
    v = evaluate_answer_v1(prediction="3.13", problem=p)
    assert not v.passed


def test_w95_executor_percent_pct_to_decimal_correct():
    p = _make_num("p_pct1", "0.25", precision=2.0,
                  answer_type="float")
    v = evaluate_answer_v1(prediction="25%", problem=p)
    assert v.passed


def test_w95_executor_percent_wrong():
    p = _make_num("p_pct2", "0.25", precision=2.0,
                  answer_type="float")
    v = evaluate_answer_v1(prediction="50%", problem=p)
    assert not v.passed


def test_w95_executor_fraction_correct():
    p = _make_num("p_frac1", "0.5", precision=2.0,
                  answer_type="float")
    v = evaluate_answer_v1(prediction="1/2", problem=p)
    assert v.passed


def test_w95_executor_comma_separated_thousands_correct():
    p = _make_num("p_thou1", "1234", precision=0.0,
                  answer_type="integer")
    v = evaluate_answer_v1(prediction="1,234", problem=p)
    assert v.passed


def test_w95_executor_unparseable_numeric():
    p = _make_num("p_un", "42", precision=0.0,
                  answer_type="integer")
    v = evaluate_answer_v1(prediction="lots", problem=p)
    assert not v.passed
    assert v.matched_rule == "numeric_unparseable"


# ---------------------------------------------------------------
# Executor — text
# ---------------------------------------------------------------

def test_w95_executor_text_exact_correct():
    p = _make_text("p_txt1", "Paris")
    v = evaluate_answer_v1(prediction="paris", problem=p)
    assert v.passed
    assert v.matched_rule == "text_exact"


def test_w95_executor_text_wrong():
    p = _make_text("p_txt2", "Paris")
    v = evaluate_answer_v1(prediction="London", problem=p)
    assert not v.passed


def test_w95_executor_self_test_on_gold_round_trip():
    corpus = (
        _make_mc("p1", "dog", ("cat", "dog")),
        _make_num("p2", "42", precision=0.0,
                  answer_type="integer"),
        _make_num("p3", "3.14", precision=2.0,
                  answer_type="float"),
        _make_text("p4", "Paris"),
    )
    res = executor_self_test_on_gold_v1(corpus)
    assert res["n_pass"] == 4
    assert res["n_problems"] == 4
    assert res["pass_rate"] == 1.0


# ---------------------------------------------------------------
# Loader — schema, merkle, slice selection
# ---------------------------------------------------------------

def test_w95_problem_to_dict_no_image_omits_bytes():
    p = _make_num("p1", "42")
    d = p.to_dict_no_image()
    assert "image_sha256" in d
    assert "image_bytes" not in d
    assert d["image_sha256"] == _FAKE_PNG_SHA


def test_w95_corpus_merkle_root_invariant_to_row_order():
    a = _make_mc("a1", "dog", ("cat", "dog"))
    b = _make_num("b1", "42")
    c = _make_text("c1", "Paris")
    root_abc = compute_corpus_merkle_root_v1((a, b, c))
    root_cba = compute_corpus_merkle_root_v1((c, b, a))
    assert root_abc == root_cba
    assert len(root_abc) == 64


def test_w95_select_subset_deterministic():
    corpus = tuple(_make_num(f"p{i:03d}", str(i)) for i in range(20))
    s1 = select_mathvista_subset_v1(
        seed=12345, n_problems=5, corpus=corpus)
    s2 = select_mathvista_subset_v1(
        seed=12345, n_problems=5, corpus=corpus)
    assert tuple(p.pid for p in s1) == tuple(p.pid for p in s2)
    assert len(s1) == 5


def test_w95_select_subset_different_seeds_disagree():
    corpus = tuple(_make_num(f"p{i:03d}", str(i)) for i in range(40))
    s1 = select_mathvista_subset_v1(
        seed=11111, n_problems=10, corpus=corpus)
    s2 = select_mathvista_subset_v1(
        seed=22222, n_problems=10, corpus=corpus)
    assert {p.pid for p in s1} != {p.pid for p in s2}


def test_w95_manifest_construction_records_sha_and_count():
    corpus = (_make_num("p1", "42"),)
    m = manifest_for_corpus_v1(
        parquet_path=Path("/dev/null"),
        problems=corpus,
        parquet_sha256="deadbeef" * 8,
        parquet_bytes=140_000_000)
    assert isinstance(m, MathVistaCorpusManifestV1)
    assert m.parquet_sha256 == "deadbeef" * 8
    assert m.n_problems == 1
    assert len(m.corpus_merkle_root) == 64


# ---------------------------------------------------------------
# Preflight — P1 corpus integrity
# ---------------------------------------------------------------

def _full_synthetic_corpus_1000():
    corpus = []
    for i in range(1000):
        if i % 3 == 0:
            corpus.append(_make_mc(
                f"pid_{i:04d}", "dog", ("cat", "dog")))
        elif i % 3 == 1:
            corpus.append(_make_num(
                f"pid_{i:04d}", str(i), precision=0.0,
                answer_type="integer"))
        else:
            corpus.append(_make_text(f"pid_{i:04d}", "Paris"))
    return tuple(corpus)


def test_w95_probe_p1_passes_on_well_formed_corpus():
    corpus = _full_synthetic_corpus_1000()
    manifest = manifest_for_corpus_v1(
        parquet_path=Path("/dev/null"),
        problems=corpus,
        parquet_sha256="aa" * 32,
        parquet_bytes=150_000_000)
    p = probe_corpus_integrity_v1(
        manifest=manifest, problems=corpus)
    assert p.passed
    assert p.probe_id == "P1_corpus_integrity"


def test_w95_probe_p1_fails_on_wrong_count():
    corpus = _full_synthetic_corpus_1000()[:999]
    manifest = manifest_for_corpus_v1(
        parquet_path=Path("/dev/null"),
        problems=corpus,
        parquet_sha256="aa" * 32,
        parquet_bytes=150_000_000)
    p = probe_corpus_integrity_v1(
        manifest=manifest, problems=corpus)
    assert not p.passed


def test_w95_probe_p1_fails_on_bytes_out_of_range():
    corpus = _full_synthetic_corpus_1000()
    manifest = manifest_for_corpus_v1(
        parquet_path=Path("/dev/null"),
        problems=corpus,
        parquet_sha256="aa" * 32,
        parquet_bytes=10)  # too small
    p = probe_corpus_integrity_v1(
        manifest=manifest, problems=corpus)
    assert not p.passed


# ---------------------------------------------------------------
# Preflight — P2 executor self-test
# ---------------------------------------------------------------

def test_w95_probe_p2_passes_on_well_formed_corpus():
    corpus = _full_synthetic_corpus_1000()
    p = probe_executor_self_test_v1(
        problems=corpus, min_pass_rate=0.98)
    assert p.passed


def test_w95_probe_p2_fails_when_gold_unparseable():
    # Inject a numeric problem whose gold isn't parseable as a
    # number → executor cannot match it.
    bad = MathVistaProblemV1(
        pid="bad", question="Q", query="Q", choices=(),
        answer="lots of dogs",  # no number
        answer_type="integer", question_type="free_form",
        unit="", precision=0.0,
        image_bytes=_FAKE_PNG, image_sha256=_FAKE_PNG_SHA,
        image_format="png", metadata={"task": "x"})
    corpus = (bad,) + _full_synthetic_corpus_1000()[:9]
    # 10 problems, 1 expected fail under self-test → 90% pass
    p = probe_executor_self_test_v1(
        problems=corpus, min_pass_rate=0.95)
    assert not p.passed


# ---------------------------------------------------------------
# Preflight — P3 A1 failure-residual estimate
# ---------------------------------------------------------------

def test_w95_estimator_known_anchor_values():
    # Llama-3.2-11B-Vision-Instruct published single-shot 33%
    # → ~60% A1@K=5 with correlation=0.5
    e11 = estimate_a1_k5_pass_rate_v1(33.0)
    assert 55.0 <= e11 <= 65.0
    e90 = estimate_a1_k5_pass_rate_v1(49.0)
    assert 68.0 <= e90 <= 78.0


def test_w95_probe_p3_passes_for_llama_11b_vision():
    p = probe_a1_failure_residual_v1(
        candidate_model="meta/llama-3.2-11b-vision-instruct",
        max_acceptable_a1_k5_pass_rate=80.0)
    assert p.passed


def test_w95_probe_p3_fails_for_unknown_model():
    p = probe_a1_failure_residual_v1(
        candidate_model="some-unlisted-model-xyz",
        max_acceptable_a1_k5_pass_rate=80.0)
    assert not p.passed


def test_w95_probe_p3_fails_when_estimated_a1_saturates():
    p = probe_a1_failure_residual_v1(
        candidate_model="meta/llama-3.2-90b-vision-instruct",
        max_acceptable_a1_k5_pass_rate=60.0)  # tighter ceiling
    assert not p.passed


# ---------------------------------------------------------------
# Preflight — P4 decomposition argument
# ---------------------------------------------------------------

def test_w95_probe_p4_passes_with_long_argument_and_geo_corpus():
    corpus = _full_synthetic_corpus_1000()
    # 1/3 are figure_qa, 1/3 geometry_problem_solving → 2/3 geo.
    p = probe_decomposition_argument_v1(
        problems=corpus,
        decomposition_argument=("x" * 250))
    assert p.passed


def test_w95_probe_p4_fails_on_short_argument():
    corpus = _full_synthetic_corpus_1000()
    p = probe_decomposition_argument_v1(
        problems=corpus,
        decomposition_argument="short")
    assert not p.passed


def test_w95_probe_p4_fails_on_corpus_with_low_geo_share():
    # No geometry/chart tags → P4's structural-fit check fails
    # regardless of argument length.
    corpus = tuple(
        _make_text(f"p{i:03d}", "Paris", task="trivia")
        for i in range(500))
    p = probe_decomposition_argument_v1(
        problems=corpus,
        decomposition_argument="y" * 300)
    assert not p.passed


# ---------------------------------------------------------------
# Composite preflight
# ---------------------------------------------------------------

def test_w95_run_preflight_full_pass_on_synthetic_well_formed():
    corpus = _full_synthetic_corpus_1000()
    manifest = manifest_for_corpus_v1(
        parquet_path=Path("/dev/null"),
        problems=corpus,
        parquet_sha256="bb" * 32,
        parquet_bytes=150_000_000)
    v = run_mathvista_preflight_v1(
        manifest=manifest,
        problems=corpus,
        candidate_model="meta/llama-3.2-11b-vision-instruct",
        decomposition_argument="z" * 300)
    assert v.overall_passes
    assert len(v.probes) == 4
    assert v.schema == W95_MATHVISTA_PREFLIGHT_V1_SCHEMA_VERSION
    assert len(v.verdict_cid) == 64


def test_w95_run_preflight_fails_when_any_probe_fails():
    corpus = _full_synthetic_corpus_1000()
    manifest = manifest_for_corpus_v1(
        parquet_path=Path("/dev/null"),
        problems=corpus,
        parquet_sha256="cc" * 32,
        parquet_bytes=150_000_000)
    # Unknown model → P3 fails.
    v = run_mathvista_preflight_v1(
        manifest=manifest,
        problems=corpus,
        candidate_model="unknown-model",
        decomposition_argument="z" * 300)
    assert not v.overall_passes


def test_w95_published_sota_anchors_include_llama_vision():
    keys = set(
        MATHVISTA_PUBLISHED_SOTA_SINGLE_SHOT_BY_MODEL.keys())
    assert "llama-3.2-11b-vision-instruct" in keys
    assert "llama-3.2-90b-vision-instruct" in keys
