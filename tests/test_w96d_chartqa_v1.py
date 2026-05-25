"""W96-D — ChartQA loader / executor / preflight CI tests.

These tests use synthetic corpora and do NOT hit the network or
NIM.  The live parquet fetch + 4-probe run is covered by
`scripts/run_w96d_chartqa_preflight.py` and is exercised in
`docs/RESULTS_W96D_CHARTQA_PREFLIGHT_V1.md`."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from coordpy.chartqa_executor_v1 import (
    CHARTQA_RELAXED_ABSOLUTE_FLOOR,
    CHARTQA_RELAXED_RELATIVE_TOLERANCE,
    W96_CHARTQA_EXECUTOR_V1_SCHEMA_VERSION,
    evaluate_chartqa_answer_v1,
    executor_self_test_on_gold_v1,
)
from coordpy.chartqa_loader_v1 import (
    CHARTQA_TEST_EXPECTED_N_PROBLEMS_LOWER,
    CHARTQA_TEST_EXPECTED_N_PROBLEMS_UPPER,
    CHARTQA_TEST_EXPECTED_PARQUET_BYTES_LOWER,
    CHARTQA_TEST_EXPECTED_PARQUET_BYTES_UPPER,
    CHARTQA_TEST_EXPECTED_PARQUET_SHA256,
    CHARTQA_TEST_PARQUET_URL,
    ChartQACorpusManifestV1,
    ChartQAProblemV1,
    W96_CHARTQA_LOADER_V1_SCHEMA_VERSION,
    compute_corpus_merkle_root_v1,
    manifest_for_corpus_v1,
    select_chartqa_subset_v1,
)
from coordpy.chartqa_preflight_v1 import (
    CHARTQA_PUBLISHED_SOTA_SINGLE_SHOT_BY_MODEL,
    W96_CHARTQA_PREFLIGHT_V1_SCHEMA_VERSION,
    estimate_a1_k5_pass_rate_v1,
    probe_a1_failure_residual_v1,
    probe_corpus_integrity_v1,
    probe_decomposition_argument_v1,
    probe_executor_self_test_v1,
    run_chartqa_preflight_v1,
)


_FAKE_PNG = b"\x89PNG\r\n\x1a\n" + b"fake-png-payload-zzz"
_FAKE_PNG_SHA = hashlib.sha256(_FAKE_PNG).hexdigest()


def _make_chartqa(
        pid: str, query: str, labels: tuple[str, ...],
        *, human: bool = True) -> ChartQAProblemV1:
    try:
        row_idx = int(pid.split("_")[-1])
    except ValueError:
        row_idx = 0
    return ChartQAProblemV1(
        pid=pid, query=query, labels=labels,
        human_or_machine=("human" if human else "machine"),
        image_bytes=_FAKE_PNG, image_sha256=_FAKE_PNG_SHA,
        image_format="png",
        metadata={"row_index": row_idx,
                  "type": ("human_test" if human
                           else "augmented_test")})


# ---------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------

def test_w96d_chartqa_loader_schema_version():
    assert (
        W96_CHARTQA_LOADER_V1_SCHEMA_VERSION
        == "coordpy.chartqa_loader_v1.v1")


def test_w96d_chartqa_executor_schema_version():
    assert (
        W96_CHARTQA_EXECUTOR_V1_SCHEMA_VERSION
        == "coordpy.chartqa_executor_v1.v1")


def test_w96d_chartqa_preflight_schema_version():
    assert (
        W96_CHARTQA_PREFLIGHT_V1_SCHEMA_VERSION
        == "coordpy.chartqa_preflight_v1.v1")


def test_w96d_chartqa_test_bounds_constants():
    assert (
        CHARTQA_TEST_EXPECTED_N_PROBLEMS_LOWER
        < CHARTQA_TEST_EXPECTED_N_PROBLEMS_UPPER)
    assert (
        CHARTQA_TEST_EXPECTED_PARQUET_BYTES_LOWER
        < CHARTQA_TEST_EXPECTED_PARQUET_BYTES_UPPER)


def test_w96d_chartqa_loader_url_points_to_huggingface():
    assert CHARTQA_TEST_PARQUET_URL.startswith(
        "https://huggingface.co/datasets/")
    assert CHARTQA_TEST_PARQUET_URL.endswith(".parquet")
    assert "lmms-lab/ChartQA" in CHARTQA_TEST_PARQUET_URL


def test_w96d_chartqa_expected_sha_is_64_hex_chars():
    assert (
        len(CHARTQA_TEST_EXPECTED_PARQUET_SHA256) == 64
        and all(
            c in "0123456789abcdef"
            for c in CHARTQA_TEST_EXPECTED_PARQUET_SHA256))


def test_w96d_chartqa_executor_tolerances_match_paper():
    # ChartQA relaxed-accuracy (Masry et al., 2022) is 5 % relative.
    assert CHARTQA_RELAXED_RELATIVE_TOLERANCE == 0.05
    assert CHARTQA_RELAXED_ABSOLUTE_FLOOR == 0.05


def test_w96d_chartqa_published_sota_table_has_llama_keys():
    assert (
        "llama-3.2-11b-vision-instruct"
        in CHARTQA_PUBLISHED_SOTA_SINGLE_SHOT_BY_MODEL)
    assert (
        "llama-3.2-90b-vision-instruct"
        in CHARTQA_PUBLISHED_SOTA_SINGLE_SHOT_BY_MODEL)
    # Recorded SOTAs sit in the published ChartQA test band 70-90 %.
    for k, v in (
            CHARTQA_PUBLISHED_SOTA_SINGLE_SHOT_BY_MODEL.items()):
        assert 60.0 <= v <= 95.0, (k, v)


# ---------------------------------------------------------------
# Executor — relaxed numeric + text rules
# ---------------------------------------------------------------

def test_w96d_chartqa_executor_numeric_exact_passes():
    p = _make_chartqa("p1", "v?", ("42",))
    r = evaluate_chartqa_answer_v1(prediction="42", problem=p)
    assert r.passed
    assert r.matched_rule == "numeric_relaxed"


def test_w96d_chartqa_executor_numeric_relaxed_within_5pct_passes():
    p = _make_chartqa("p1", "v?", ("100",))
    # 105 vs 100 = 5% — at boundary, passes (≤).
    r = evaluate_chartqa_answer_v1(prediction="105", problem=p)
    assert r.passed


def test_w96d_chartqa_executor_numeric_outside_5pct_fails():
    p = _make_chartqa("p1", "v?", ("100",))
    r = evaluate_chartqa_answer_v1(prediction="106", problem=p)
    assert not r.passed


def test_w96d_chartqa_executor_text_exact_passes():
    p = _make_chartqa("p2", "color?", ("blue",))
    r = evaluate_chartqa_answer_v1(
        prediction="blue", problem=p)
    assert r.passed
    assert r.matched_rule == "text_exact"


def test_w96d_chartqa_executor_text_case_insensitive():
    p = _make_chartqa("p2", "color?", ("blue",))
    r = evaluate_chartqa_answer_v1(
        prediction="BLUE", problem=p)
    assert r.passed


def test_w96d_chartqa_executor_numeric_near_zero_abs_floor():
    p = _make_chartqa("p3", "v?", ("0",))
    r = evaluate_chartqa_answer_v1(
        prediction="0.04", problem=p)
    assert r.passed
    r2 = evaluate_chartqa_answer_v1(
        prediction="0.10", problem=p)
    assert not r2.passed


def test_w96d_chartqa_executor_multiple_labels_any_match_passes():
    p = _make_chartqa("p4", "?", ("blue", "navy", "dark blue"))
    r = evaluate_chartqa_answer_v1(
        prediction="navy", problem=p)
    assert r.passed
    assert r.matched_label_idx == 1


def test_w96d_chartqa_executor_no_match_fails():
    p = _make_chartqa("p5", "?", ("blue",))
    r = evaluate_chartqa_answer_v1(prediction="red", problem=p)
    assert not r.passed
    assert r.matched_rule == "no_match"


def test_w96d_chartqa_executor_self_test_on_synthetic_corpus():
    corpus = (
        _make_chartqa("p_001", "v?", ("42",)),
        _make_chartqa("p_002", "color?", ("blue",)),
        _make_chartqa("p_003", "?", ("3.14",)),
    )
    result = executor_self_test_on_gold_v1(corpus)
    assert result["n_pass"] == 3
    assert result["pass_rate"] == 1.0


# ---------------------------------------------------------------
# Loader — manifest + deterministic subset
# ---------------------------------------------------------------

def test_w96d_chartqa_manifest_carries_url_and_sha():
    corpus = tuple(
        _make_chartqa(f"p_{i:03d}", "?", ("ans",))
        for i in range(5))
    m = manifest_for_corpus_v1(
        parquet_path=Path("/tmp/fake.parquet"),
        problems=corpus,
        parquet_sha256="deadbeef" * 8,
        parquet_bytes=12345)
    assert m.parquet_sha256 == "deadbeef" * 8
    assert m.parquet_bytes == 12345
    assert m.n_problems == 5
    assert m.parquet_url == CHARTQA_TEST_PARQUET_URL
    assert m.corpus_merkle_root  # non-empty
    assert m.schema == W96_CHARTQA_LOADER_V1_SCHEMA_VERSION


def test_w96d_chartqa_subset_is_deterministic_by_seed():
    corpus = tuple(
        _make_chartqa(f"p_{i:03d}", "?", ("ans",))
        for i in range(20))
    a = select_chartqa_subset_v1(
        seed=96_504_001, n_problems=5, corpus=corpus)
    b = select_chartqa_subset_v1(
        seed=96_504_001, n_problems=5, corpus=corpus)
    c = select_chartqa_subset_v1(
        seed=96_504_002, n_problems=5, corpus=corpus)
    assert len(a) == 5
    assert tuple(p.pid for p in a) == tuple(p.pid for p in b)
    # Different seeds give different slices.
    assert tuple(p.pid for p in a) != tuple(p.pid for p in c)


def test_w96d_chartqa_merkle_root_invariant_to_input_order():
    a = (
        _make_chartqa("p_001", "?", ("1",)),
        _make_chartqa("p_002", "?", ("2",)),
        _make_chartqa("p_003", "?", ("3",)),
    )
    b = (a[2], a[0], a[1])
    assert (
        compute_corpus_merkle_root_v1(a)
        == compute_corpus_merkle_root_v1(b))


# ---------------------------------------------------------------
# Preflight — P3 saturation logic
# ---------------------------------------------------------------

def test_w96d_chartqa_a1_k5_estimator_matches_w95_shape():
    # Identical correlation=0.5 / K=5 estimator as W95.
    est = estimate_a1_k5_pass_rate_v1(50.0, k=5)
    # At p=0.5: iid_upper = 1 - 0.5^5 = 0.96875; blended = 0.5*0.96875 + 0.5*0.5 = 0.734375
    assert abs(est - 73.4375) < 0.001


def test_w96d_chartqa_p3_fails_for_saturated_chartqa_models():
    # Recorded Llama-3.2-11B-Vision-Instruct ChartQA = 83.4 % → A1@K=5 ≈ 91.7 %
    r = probe_a1_failure_residual_v1(
        candidate_model="meta/llama-3.2-11b-vision-instruct")
    assert not r.passed
    assert r.evidence["single_shot_pct"] == 83.4
    # Recorded Llama-3.2-90B-Vision-Instruct ChartQA = 85.5 % → A1@K=5 ≈ 92.75 %
    r90 = probe_a1_failure_residual_v1(
        candidate_model="meta/llama-3.2-90b-vision-instruct")
    assert not r90.passed
    assert r90.evidence["single_shot_pct"] == 85.5


def test_w96d_chartqa_p3_passes_for_hypothetical_unsaturated_model():
    # Hypothetical 60 % single-shot → A1@K=5 ≈ 79.5 % → PASS at 80 % ceiling.
    r = probe_a1_failure_residual_v1(
        candidate_model="hypothetical-weaker-vlm",
        published_sota_table={"hypothetical-weaker-vlm": 60.0})
    assert r.passed
    assert r.evidence["estimated_a1_k5_pct"] < 80.0


def test_w96d_chartqa_p3_fails_for_unknown_model():
    r = probe_a1_failure_residual_v1(
        candidate_model="unknown/unknown-vision")
    assert not r.passed
    assert "no published single-shot" in r.summary.lower()


# ---------------------------------------------------------------
# Composite verdict — synthetic corpus full P1..P4 run
# ---------------------------------------------------------------

def test_w96d_chartqa_composite_fails_when_p3_saturated():
    corpus = tuple(
        _make_chartqa(
            f"p_{i:06d}", "What is X?", ("42",), human=(i < 50))
        for i in range(100))
    manifest = manifest_for_corpus_v1(
        parquet_path=Path("/tmp/fake.parquet"),
        problems=corpus,
        parquet_sha256="d" * 64,
        parquet_bytes=int(
            (CHARTQA_TEST_EXPECTED_PARQUET_BYTES_LOWER
             + CHARTQA_TEST_EXPECTED_PARQUET_BYTES_UPPER) // 2))
    # n=100 will fail P1 (out of [2000,3000]) → composite FAIL.
    verdict = run_chartqa_preflight_v1(
        manifest=manifest,
        problems=corpus,
        candidate_model=(
            "meta/llama-3.2-11b-vision-instruct"),
        decomposition_argument=(
            "x" * 250))
    assert not verdict.overall_passes
    # P1 fails because n=100 is below the lower bound.
    p1 = next(
        p for p in verdict.probes
        if p.probe_id == "P1_corpus_integrity")
    assert not p1.passed
    # P3 still fails on saturated Llama-3.2-11B.
    p3 = next(
        p for p in verdict.probes
        if p.probe_id == "P3_a1_failure_residual")
    assert not p3.passed
