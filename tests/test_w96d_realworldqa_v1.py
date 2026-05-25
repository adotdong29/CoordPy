"""W96-D — RealWorldQA loader / executor / preflight CI tests.

These tests use synthetic corpora and do NOT hit the network or
NIM.  The live parquet fetch + 4-probe run is covered by
`scripts/run_w96d_realworldqa_preflight.py` and is exercised in
`docs/RESULTS_W96D_REALWORLDQA_PREFLIGHT_V1.md`."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from coordpy.realworldqa_executor_v1 import (
    REALWORLDQA_RELAXED_ABSOLUTE_FLOOR,
    REALWORLDQA_RELAXED_RELATIVE_TOLERANCE,
    W96_REALWORLDQA_EXECUTOR_V1_SCHEMA_VERSION,
    evaluate_realworldqa_answer_v1,
    executor_self_test_on_gold_v1,
)
from coordpy.realworldqa_loader_v1 import (
    REALWORLDQA_TEST_EXPECTED_N_PROBLEMS_LOWER,
    REALWORLDQA_TEST_EXPECTED_N_PROBLEMS_UPPER,
    REALWORLDQA_TEST_EXPECTED_PARQUET_BYTES_LOWER,
    REALWORLDQA_TEST_EXPECTED_PARQUET_BYTES_UPPER,
    REALWORLDQA_TEST_EXPECTED_PARQUET_SHA256,
    REALWORLDQA_TEST_PARQUET_URLS,
    RealWorldQACorpusManifestV1,
    RealWorldQAProblemV1,
    W96_REALWORLDQA_LOADER_V1_SCHEMA_VERSION,
    compute_corpus_merkle_root_v1,
    manifest_for_corpus_v1,
    select_realworldqa_subset_v1,
)
from coordpy.realworldqa_preflight_v1 import (
    REALWORLDQA_PUBLISHED_SOTA_SINGLE_SHOT_BY_MODEL,
    W96_REALWORLDQA_PREFLIGHT_V1_SCHEMA_VERSION,
    estimate_a1_k5_pass_rate_v1,
    probe_a1_failure_residual_v1,
    probe_corpus_integrity_v1,
    probe_decomposition_argument_v1,
    probe_executor_self_test_v1,
    run_realworldqa_preflight_v1,
)


_FAKE_PNG = b"\x89PNG\r\n\x1a\n" + b"fake-png-payload-zzz"
_FAKE_PNG_SHA = hashlib.sha256(_FAKE_PNG).hexdigest()


def _make_rwqa(
        pid: str, question: str, answer: str,
        *, image_path: str = "") -> RealWorldQAProblemV1:
    try:
        row_idx = int(pid.split("_")[-1])
    except ValueError:
        row_idx = 0
    return RealWorldQAProblemV1(
        pid=pid, question=question, answer=answer,
        image_path=image_path,
        image_bytes=_FAKE_PNG, image_sha256=_FAKE_PNG_SHA,
        image_format="png",
        metadata={
            "shard_idx": 0,
            "row_index_in_shard": row_idx,
        })


# ---------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------

def test_w96d_realworldqa_loader_schema_version():
    assert (
        W96_REALWORLDQA_LOADER_V1_SCHEMA_VERSION
        == "coordpy.realworldqa_loader_v1.v1")


def test_w96d_realworldqa_executor_schema_version():
    assert (
        W96_REALWORLDQA_EXECUTOR_V1_SCHEMA_VERSION
        == "coordpy.realworldqa_executor_v1.v1")


def test_w96d_realworldqa_preflight_schema_version():
    assert (
        W96_REALWORLDQA_PREFLIGHT_V1_SCHEMA_VERSION
        == "coordpy.realworldqa_preflight_v1.v1")


def test_w96d_realworldqa_test_bounds_constants():
    assert (
        REALWORLDQA_TEST_EXPECTED_N_PROBLEMS_LOWER
        < REALWORLDQA_TEST_EXPECTED_N_PROBLEMS_UPPER)
    assert (
        REALWORLDQA_TEST_EXPECTED_PARQUET_BYTES_LOWER
        < REALWORLDQA_TEST_EXPECTED_PARQUET_BYTES_UPPER)


def test_w96d_realworldqa_loader_urls_point_to_huggingface():
    assert len(REALWORLDQA_TEST_PARQUET_URLS) == 2
    for url in REALWORLDQA_TEST_PARQUET_URLS:
        assert url.startswith(
            "https://huggingface.co/datasets/")
        assert url.endswith(".parquet")
        assert "lmms-lab/RealWorldQA" in url


def test_w96d_realworldqa_shard_shas_are_recorded():
    assert len(REALWORLDQA_TEST_EXPECTED_PARQUET_SHA256) == 2
    for s in REALWORLDQA_TEST_EXPECTED_PARQUET_SHA256:
        assert s is not None
        assert len(s) == 64
        assert all(c in "0123456789abcdef" for c in s)


def test_w96d_realworldqa_executor_tolerances():
    assert REALWORLDQA_RELAXED_RELATIVE_TOLERANCE == 0.05
    assert REALWORLDQA_RELAXED_ABSOLUTE_FLOOR == 0.05


def test_w96d_realworldqa_published_sota_table_has_llama_keys():
    assert (
        "llama-3.2-11b-vision-instruct"
        in REALWORLDQA_PUBLISHED_SOTA_SINGLE_SHOT_BY_MODEL)
    assert (
        "llama-3.2-90b-vision-instruct"
        in REALWORLDQA_PUBLISHED_SOTA_SINGLE_SHOT_BY_MODEL)


# ---------------------------------------------------------------
# Executor — three rule classes
# ---------------------------------------------------------------

def test_w96d_realworldqa_executor_multi_choice_letter_passes():
    p = _make_rwqa("p1", "Choose:", "B")
    r = evaluate_realworldqa_answer_v1(
        prediction="B", problem=p)
    assert r.passed
    assert r.matched_rule == "multi_choice_letter"


def test_w96d_realworldqa_executor_multi_choice_in_prose():
    p = _make_rwqa("p1", "Choose:", "B")
    r = evaluate_realworldqa_answer_v1(
        prediction="The answer is B.", problem=p)
    assert r.passed
    assert r.matched_rule == "multi_choice_letter"


def test_w96d_realworldqa_executor_multi_choice_wrong_fails():
    p = _make_rwqa("p1", "Choose:", "B")
    r = evaluate_realworldqa_answer_v1(
        prediction="C", problem=p)
    assert not r.passed


def test_w96d_realworldqa_executor_numeric_exact_passes():
    p = _make_rwqa("p2", "How many?", "3")
    r = evaluate_realworldqa_answer_v1(
        prediction="3", problem=p)
    assert r.passed
    assert r.matched_rule == "numeric_relaxed"


def test_w96d_realworldqa_executor_numeric_relaxed_5pct_passes():
    p = _make_rwqa("p2", "How many?", "100")
    r = evaluate_realworldqa_answer_v1(
        prediction="104", problem=p)
    assert r.passed


def test_w96d_realworldqa_executor_text_exact_passes():
    p = _make_rwqa("p3", "What color?", "red")
    r = evaluate_realworldqa_answer_v1(
        prediction="red", problem=p)
    assert r.passed
    assert r.matched_rule == "text_exact"


def test_w96d_realworldqa_executor_self_test_on_synthetic_corpus():
    corpus = (
        _make_rwqa("p_001", "Choose:", "B"),
        _make_rwqa("p_002", "How many?", "3"),
        _make_rwqa("p_003", "Color?", "red"),
    )
    result = executor_self_test_on_gold_v1(corpus)
    assert result["n_pass"] == 3
    assert result["pass_rate"] == 1.0


# ---------------------------------------------------------------
# Loader — manifest + deterministic subset
# ---------------------------------------------------------------

def test_w96d_realworldqa_manifest_carries_shards_and_total():
    corpus = tuple(
        _make_rwqa(f"p_{i:03d}", "?", "ans")
        for i in range(5))
    m = manifest_for_corpus_v1(
        parquet_paths=(Path("/tmp/a"), Path("/tmp/b")),
        problems=corpus,
        parquet_shard_sha256=("aa" * 32, "bb" * 32),
        parquet_total_bytes=12345)
    assert m.parquet_total_bytes == 12345
    assert m.n_problems == 5
    assert m.parquet_shard_sha256 == ("aa" * 32, "bb" * 32)
    assert m.parquet_urls == REALWORLDQA_TEST_PARQUET_URLS
    assert m.corpus_merkle_root
    assert m.schema == W96_REALWORLDQA_LOADER_V1_SCHEMA_VERSION


def test_w96d_realworldqa_subset_is_deterministic_by_seed():
    corpus = tuple(
        _make_rwqa(f"p_{i:03d}", "?", "ans")
        for i in range(20))
    a = select_realworldqa_subset_v1(
        seed=96_504_002, n_problems=5, corpus=corpus)
    b = select_realworldqa_subset_v1(
        seed=96_504_002, n_problems=5, corpus=corpus)
    c = select_realworldqa_subset_v1(
        seed=96_504_001, n_problems=5, corpus=corpus)
    assert len(a) == 5
    assert tuple(p.pid for p in a) == tuple(p.pid for p in b)
    assert tuple(p.pid for p in a) != tuple(p.pid for p in c)


def test_w96d_realworldqa_merkle_root_order_invariant():
    a = (
        _make_rwqa("p_001", "?", "1"),
        _make_rwqa("p_002", "?", "2"),
        _make_rwqa("p_003", "?", "3"),
    )
    b = (a[2], a[0], a[1])
    assert (
        compute_corpus_merkle_root_v1(a)
        == compute_corpus_merkle_root_v1(b))


# ---------------------------------------------------------------
# Preflight — P3 saturation logic
# ---------------------------------------------------------------

def test_w96d_realworldqa_a1_k5_estimator_matches_shape():
    est = estimate_a1_k5_pass_rate_v1(50.0, k=5)
    assert abs(est - 73.4375) < 0.001


def test_w96d_realworldqa_p3_passes_for_llama_11b():
    # Recorded llama-3.2-11b-vision = 50% → A1@K=5 ≈ 73.4% → PASS at 80%.
    r = probe_a1_failure_residual_v1(
        candidate_model="meta/llama-3.2-11b-vision-instruct")
    assert r.passed
    assert r.evidence["single_shot_pct"] == 50.0
    assert r.evidence["estimated_a1_k5_pct"] < 80.0


def test_w96d_realworldqa_p3_passes_narrowly_for_llama_90b():
    # Recorded llama-3.2-90b-vision = 60% → A1@K=5 ≈ 79.5% → PASS at 80%.
    r = probe_a1_failure_residual_v1(
        candidate_model="meta/llama-3.2-90b-vision-instruct")
    assert r.passed
    assert r.evidence["single_shot_pct"] == 60.0
    assert r.evidence["estimated_a1_k5_pct"] < 80.0
    # The 90B margin is narrow (≤ 1 pp above floor).
    assert (
        r.evidence["max_acceptable_a1_k5_pct"]
        - r.evidence["estimated_a1_k5_pct"]
        < 5.0)


def test_w96d_realworldqa_p3_fails_for_saturated_model():
    # GPT-4o is reported around 75% single-shot → A1@K=5 ≈ 87.6%.
    r = probe_a1_failure_residual_v1(
        candidate_model="gpt-4o")
    assert not r.passed


def test_w96d_realworldqa_p3_fails_for_unknown_model():
    r = probe_a1_failure_residual_v1(
        candidate_model="unknown/unknown-vision")
    assert not r.passed


# ---------------------------------------------------------------
# Composite verdict — synthetic corpus full P1..P4 run
# ---------------------------------------------------------------

def test_w96d_realworldqa_composite_synthetic_undersized_corpus():
    corpus = tuple(
        _make_rwqa(f"p_{i:06d}", "Q?", "A")
        for i in range(10))
    manifest = manifest_for_corpus_v1(
        parquet_paths=(Path("/tmp/a"), Path("/tmp/b")),
        problems=corpus,
        parquet_shard_sha256=("aa" * 32, "bb" * 32),
        parquet_total_bytes=int(
            (REALWORLDQA_TEST_EXPECTED_PARQUET_BYTES_LOWER
             + REALWORLDQA_TEST_EXPECTED_PARQUET_BYTES_UPPER)
            // 2))
    verdict = run_realworldqa_preflight_v1(
        manifest=manifest,
        problems=corpus,
        candidate_model=(
            "meta/llama-3.2-11b-vision-instruct"),
        decomposition_argument="x" * 250)
    # n=10 fails P1 bounds [700, 800].
    assert not verdict.overall_passes
    p1 = next(
        p for p in verdict.probes
        if p.probe_id == "P1_corpus_integrity")
    assert not p1.passed
    p3 = next(
        p for p in verdict.probes
        if p.probe_id == "P3_a1_failure_residual")
    # P3 PASSes for 11B (well below 80%).
    assert p3.passed


def test_w96d_realworldqa_composite_synthetic_p3_pass_p1_fail_match():
    """Composite FAILs even when P3 PASSes if P1 fails — proves
    the AND-aggregation."""
    corpus = (
        _make_rwqa("p_001", "Q?", "A"),
        _make_rwqa("p_002", "Q?", "A"),
    )
    manifest = manifest_for_corpus_v1(
        parquet_paths=(Path("/tmp/a"),),
        problems=corpus,
        parquet_shard_sha256=("dead" * 16,),
        parquet_total_bytes=1)  # below bytes range too
    verdict = run_realworldqa_preflight_v1(
        manifest=manifest,
        problems=corpus,
        candidate_model=(
            "meta/llama-3.2-11b-vision-instruct"),
        decomposition_argument="x" * 250)
    assert not verdict.overall_passes
