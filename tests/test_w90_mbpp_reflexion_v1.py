"""W90 — MBPP sequential-reflexion bench V1 CI tests."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from coordpy.mbpp_reflexion_bench_v1 import (
    W90_MBPP_REFLEXION_BENCH_V1_SCHEMA_VERSION,
    MBPPBenchConfigV1,
    MBPPProblemV1,
    _extract_entry_point_from_test,
    run_mbpp_reflexion_bench_v1,
)


def test_w90_schema_version():
    assert (W90_MBPP_REFLEXION_BENCH_V1_SCHEMA_VERSION
            == "coordpy.mbpp_reflexion_bench_v1.v1")


def test_w90_entry_point_extraction():
    """Extracts the function name from various assertion forms."""
    cases = [
        ("assert foo(1, 2) == 3", "foo"),
        ("assert set(my_func((1,2,3))) == set([1,2])", "my_func"),
        ("assert sorted(g([3,1,2])) == [1,2,3]", "g"),
        ("assert abs(h(0.1)-0.2) < 1e-6", "h"),
    ]
    for s, expected in cases:
        assert _extract_entry_point_from_test(s) == expected


def _toy_problem() -> MBPPProblemV1:
    return MBPPProblemV1(
        task_id=99999,
        text="Return x + 1.",
        code="def f(x): return x + 1",
        test_list=(
            "assert f(0) == 1",
            "assert f(1) == 2",
            "assert f(-1) == 0",
        ),
        test_imports=(),
        entry_point="f")


def _make_passing_gen():
    def gen(prompt, max_tokens, temperature):
        return (
            "```python\ndef f(x):\n    return x + 1\n```", 50)
    return gen


def _make_failing_gen():
    def gen(prompt, max_tokens, temperature):
        return (
            "```python\ndef f(x):\n    return x\n```", 50)
    return gen


def _make_reflexion_gen():
    def gen(prompt, max_tokens, temperature):
        if "Executor stderr" in prompt:
            return (
                "```python\ndef f(x):\n    return x + 1\n```", 50)
        return (
            "```python\ndef f(x):\n    return x\n```", 50)
    return gen


def _run_bench(gen, n_problems=1, n_seeds=1):
    corpus = (_toy_problem(), _toy_problem(), _toy_problem())
    cfg = MBPPBenchConfigV1(
        n_problems=n_problems, K_multi_sample=5,
        seeds=tuple(range(90_001, 90_001 + n_seeds)),
        sampling_temperature=0.7,
        max_tokens_per_call=64)
    return run_mbpp_reflexion_bench_v1(
        gen=gen, model_id="synth", corpus=corpus, config=cfg)


def test_w90_bench_all_pass():
    report = _run_bench(_make_passing_gen(),
                        n_problems=2, n_seeds=2)
    assert report.a0_mean_pass_at_1 == 1.0
    assert report.a1_mean_pass_at_1 == 1.0
    assert report.b_mean_pass_at_1 == 1.0


def test_w90_bench_all_fail():
    report = _run_bench(_make_failing_gen(),
                        n_problems=2, n_seeds=2)
    assert report.a0_mean_pass_at_1 == 0.0
    assert report.a1_mean_pass_at_1 == 0.0
    assert report.b_mean_pass_at_1 == 0.0
    assert not report.b_mean_strictly_beats_a0_mean
    assert not report.b_mean_strictly_beats_a1_mean


def test_w90_bench_reflexion_gen_b_beats_a0_a1():
    report = _run_bench(_make_reflexion_gen(),
                        n_problems=2, n_seeds=1)
    assert report.a0_mean_pass_at_1 == 0.0
    assert report.a1_mean_pass_at_1 == 0.0
    assert report.b_mean_pass_at_1 == 1.0
    assert report.b_mean_strictly_beats_a0_mean
    assert report.b_mean_strictly_beats_a1_mean
    assert report.b_mean_minus_a1_mean_pp == 100.0


def test_w90_audit_chain_re_derives():
    report = _run_bench(_make_reflexion_gen(),
                        n_problems=2, n_seeds=2)
    all_cids = []
    for s in report.per_seed:
        all_cids.extend(s.outcome_cids)
        derived = hashlib.sha256(
            json.dumps({
                "kind": "w90_mbpp_seed_merkle_root",
                "seed": int(s.seed),
                "outcome_cids": list(s.outcome_cids),
            }, sort_keys=True, separators=(",", ":"),
                default=str).encode("utf-8")).hexdigest()
        assert derived == s.seed_merkle_root
    bench_derived = hashlib.sha256(
        json.dumps({
            "kind": "w90_mbpp_bench_merkle_root",
            "model_id": report.model_id,
            "outcome_cids": list(all_cids),
            "seeds": [int(s.seed) for s in report.per_seed],
        }, sort_keys=True, separators=(",", ":"),
            default=str).encode("utf-8")).hexdigest()
    assert bench_derived == report.bench_merkle_root


def test_w90_module_surface_explicit_import():
    import importlib
    mod = importlib.import_module(
        "coordpy.mbpp_reflexion_bench_v1")
    assert hasattr(
        mod, "W90_MBPP_REFLEXION_BENCH_V1_SCHEMA_VERSION")
    import coordpy
    assert not hasattr(coordpy, "run_mbpp_reflexion_bench_v1")
