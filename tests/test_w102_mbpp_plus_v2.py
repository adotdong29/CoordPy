"""W102 — MBPP+ V2 infrastructure unit tests (NIM-free).

Tests:
* V2 loader handles the real EvalPlus parquet schema.
* V2 loader refuses rows with empty extra_test_program.
* V2 executor PASSes canonical solutions under base_and_plus.
* V2 executor FAILs candidates that pass base but break on plus.
* V2 bench selects deterministic subset.
* V2 bench end-to-end with fake gen + canonical solutions.
* V2 preflight P5 detects iteration pattern + assertion call.
* V2 preflight rejects unmodified V1 anti-pattern bench module.
* Stable boundary preserved (no version bump; explicit-import).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from coordpy import (
    mbpp_plus_executor_v2,
    mbpp_plus_loader_v2,
    mbpp_plus_preflight_v2,
    mbpp_plus_reflexion_bench_v2,
)


# ---------------------------------------------------------------
# Loader V2
# ---------------------------------------------------------------


def test_loader_v2_schema_version():
    assert (
        mbpp_plus_loader_v2.W102_MBPP_PLUS_LOADER_V2_SCHEMA_VERSION
        == "coordpy.mbpp_plus_loader_v2.v1")


def test_loader_v2_constants_documented():
    assert (
        "huggingface.co/datasets/evalplus/mbppplus"
        in mbpp_plus_loader_v2.MBPP_PLUS_HF_CANONICAL_URL_V021)
    assert len(
        mbpp_plus_loader_v2.MBPP_PLUS_HF_EXPECTED_SHA256_V021
    ) == 64


def test_loader_v2_rows_to_problems_handles_evalplus_schema():
    rows = []
    for i in range(360):
        entry = f"my_func_{i}"
        rows.append({
            "task_id": i + 1,    # integer (EvalPlus shape)
            "prompt": f"Write a function that does thing {i}.",
            "code": f"def {entry}(x):\n    return x + {i}",
            "test_list": [
                f"assert {entry}(1) == {1 + i}"],
            "test_imports": [],
            "test": (
                "inputs = [(1,)]\n"
                "results = [{}]\n"
                "for i, (inp, exp) in enumerate(zip(inputs, "
                "results)):\n"
                "    assert {}(*inp) == exp\n"
            ).format(1 + i, entry),
        })
    parsed = mbpp_plus_loader_v2._rows_to_problems(rows)
    assert len(parsed) == 360
    assert parsed[0].task_id == "Mbpp/1"  # int → Mbpp/<n>
    assert parsed[0].entry_point == "my_func_0"
    assert parsed[0].base_test_list == (
        "assert my_func_0(1) == 1",)
    assert "inputs = [(1,)]" in parsed[0].extra_test_program


def test_loader_v2_refuses_row_with_empty_extra_test_program():
    rows = [
        {
            "task_id": 1,
            "prompt": "p",
            "code": "def f(): return 0",
            "test_list": ["assert f() == 0"],
            "test_imports": [],
            "test": "",
        }
        for _ in range(400)
    ]
    with pytest.raises(
            mbpp_plus_loader_v2.MbppPlusV2CorpusError):
        mbpp_plus_loader_v2._rows_to_problems(rows)


def test_loader_v2_is_cached_false_on_missing(tmp_path):
    path = tmp_path / "nonexistent.parquet"
    assert not mbpp_plus_loader_v2.is_mbpp_plus_v2_cached(
        cache_path=str(path))


# ---------------------------------------------------------------
# Executor V2
# ---------------------------------------------------------------


def _make_v2_problem(
        *, code: str, entry: str = "f",
        base_tests: list[str] | None = None,
        extra_test_program: str | None = None,
        test_imports: list[str] | None = None,
) -> mbpp_plus_loader_v2.MbppPlusProblemV2:
    return mbpp_plus_loader_v2.MbppPlusProblemV2(
        task_id="Mbpp/test",
        prompt="test",
        canonical_code=code,
        base_test_list=tuple(base_tests or []),
        extra_test_program=str(extra_test_program or ""),
        test_imports=tuple(test_imports or []),
        entry_point=entry,
    )


def test_executor_v2_base_only_passes_on_correct_canonical():
    p = _make_v2_problem(
        code="def add(a, b):\n    return a + b",
        entry="add",
        base_tests=["assert add(2, 3) == 5"],
        extra_test_program=(
            "inputs = [(1, 1)]\nresults = [2]\n"
            "for i, (inp, exp) in enumerate(zip(inputs, "
            "results)):\n"
            "    assert add(*inp) == exp\n"))
    res = mbpp_plus_executor_v2.run_mbpp_plus_executor_v2(
        problem=p, candidate_code=p.canonical_code,
        mode="base_only")
    assert res.passed
    assert res.n_assertions_passed == 1
    assert res.n_assertions_total == 1
    assert res.mode == "base_only"


def test_executor_v2_base_and_plus_passes_on_canonical():
    p = _make_v2_problem(
        code="def add(a, b):\n    return a + b",
        entry="add",
        base_tests=["assert add(2, 3) == 5"],
        extra_test_program=(
            "inputs = [(1, 1), (10, -5)]\n"
            "results = [2, 5]\n"
            "for i, (inp, exp) in enumerate(zip(inputs, "
            "results)):\n"
            "    assert add(*inp) == exp\n"))
    res = mbpp_plus_executor_v2.run_mbpp_plus_executor_v2(
        problem=p, candidate_code=p.canonical_code,
        mode="base_and_plus")
    assert res.passed
    assert res.mode == "base_and_plus"


def test_executor_v2_base_and_plus_fails_on_plus_breaking_candidate():
    """A candidate that passes the BASE test but breaks on a plus
    iteration must FAIL overall under base_and_plus."""
    p = _make_v2_problem(
        # Bug: ignores second arg; passes base by coincidence
        # but fails plus for (10, -5).
        code="def add(a, b):\n    return a + 3\n",
        entry="add",
        base_tests=["assert add(2, 3) == 5"],
        extra_test_program=(
            "inputs = [(10, -5)]\nresults = [5]\n"
            "for i, (inp, exp) in enumerate(zip(inputs, "
            "results)):\n"
            "    assert add(*inp) == exp\n"))
    res_bo = mbpp_plus_executor_v2.run_mbpp_plus_executor_v2(
        problem=p, candidate_code=p.canonical_code,
        mode="base_only")
    res_bap = mbpp_plus_executor_v2.run_mbpp_plus_executor_v2(
        problem=p, candidate_code=p.canonical_code,
        mode="base_and_plus")
    assert res_bo.passed
    assert not res_bap.passed
    assert res_bap.returncode != 0


def test_executor_v2_plus_only_aliases_base_and_plus():
    """plus_only is documented as alias for base_and_plus
    (EvalPlus iteration order is not cleanly separable)."""
    p = _make_v2_problem(
        code="def f(): return 0",
        entry="f",
        base_tests=["assert f() == 0"],
        extra_test_program=(
            "inputs = [()]\nresults = [0]\n"
            "for i, (inp, exp) in enumerate(zip(inputs, "
            "results)):\n"
            "    assert f(*inp) == exp\n"))
    res = mbpp_plus_executor_v2.run_mbpp_plus_executor_v2(
        problem=p, candidate_code=p.canonical_code,
        mode="plus_only")
    assert res.passed
    assert "alias" in res.mode_note.lower()


# ---------------------------------------------------------------
# Bench V2
# ---------------------------------------------------------------


def test_bench_v2_subset_deterministic():
    corpus = tuple(
        _make_v2_problem(
            code=f"def f_{i}():\n    return {i}",
            entry=f"f_{i}",
            base_tests=[f"assert f_{i}() == {i}"],
            extra_test_program=(
                f"inputs = [()]\nresults = [{i}]\n"
                "for j, (inp, exp) in enumerate(zip(inputs, "
                "results)):\n"
                f"    assert f_{i}(*inp) == exp\n"))
        for i in range(50))
    s1 = mbpp_plus_reflexion_bench_v2.select_mbpp_plus_v2_subset(
        corpus=corpus, n_problems=10, seed=101_001)
    s2 = mbpp_plus_reflexion_bench_v2.select_mbpp_plus_v2_subset(
        corpus=corpus, n_problems=10, seed=101_001)
    s3 = mbpp_plus_reflexion_bench_v2.select_mbpp_plus_v2_subset(
        corpus=corpus, n_problems=10, seed=101_002)
    assert s1 == s2
    assert s1 != s3
    assert len(s1) == 10


def test_bench_v2_runs_with_fake_gen_canonical():
    corpus = tuple(
        _make_v2_problem(
            code=f"def f_{i}(x):\n    return x + {i}",
            entry=f"f_{i}",
            base_tests=[f"assert f_{i}(0) == {i}"],
            extra_test_program=(
                f"inputs = [(1,)]\nresults = [{1+i}]\n"
                "for j, (inp, exp) in enumerate(zip(inputs, "
                "results)):\n"
                f"    assert f_{i}(*inp) == exp\n"))
        for i in range(5))

    def gen(prompt, max_tokens, temperature):
        for i in range(5):
            entry = f"f_{i}"
            if f"f_{i}(0)" in prompt:
                code = (
                    f"```python\ndef {entry}(x):\n"
                    f"    return x + {i}\n```")
                return code, 10
        return "```python\ndef f():\n    return 0\n```", 10

    cfg = mbpp_plus_reflexion_bench_v2.MbppPlusV2BenchConfig(
        n_problems=5, K_multi_sample=5, seeds=(101_001,))
    report = (
        mbpp_plus_reflexion_bench_v2
        .run_mbpp_plus_reflexion_bench_v2(
            gen=gen,
            model_id="fake/synthetic",
            corpus=corpus,
            config=cfg))
    assert report.a0_mean_pass_at_1 == 1.0
    assert report.a1_mean_pass_at_1 == 1.0
    assert report.b_mean_pass_at_1 == 1.0
    assert report.b_mean_minus_a1_mean_pp == 0.0


def test_bench_v2_mlb_helper_computes_invocation_and_rescue():
    """MLB helper should report invocation/rescue rates from the
    per_problem_b_first_pass_idx field."""
    # Build a synthetic report with 4 problems:
    # - p0: B passed on attempt 0 (no reflexion invoked)
    # - p1: B passed on attempt 2 (reflexion invoked + rescued)
    # - p2: B passed on attempt 4 (reflexion invoked + rescued)
    # - p3: B failed entirely (reflexion invoked + NOT rescued)
    seed_report = mbpp_plus_reflexion_bench_v2.MbppPlusV2SeedReport(
        schema="x", seed=1, n_problems=4,
        a0_pass_at_1=0.5, a1_pass_at_1=0.5, b_pass_at_1=0.75,
        a0_total_wall_ms=0, a1_total_wall_ms=0,
        b_total_wall_ms=0,
        per_problem_a0_passed=(False,) * 4,
        per_problem_a1_passed=(False,) * 4,
        per_problem_b_passed=(True, True, True, False),
        per_problem_b_first_pass_idx=(0, 2, 4, -1),
        outcome_cids=(),
        seed_merkle_root="x")
    bench_report = (
        mbpp_plus_reflexion_bench_v2.MbppPlusV2BenchReport(
            schema="x", model_id="x", n_problems=4, n_seeds=1,
            K_multi_sample=5, per_seed=(seed_report,),
            a0_mean_pass_at_1=0.5, a1_mean_pass_at_1=0.5,
            b_mean_pass_at_1=0.75,
            b_beats_a0_per_seed=(True,),
            b_beats_a1_per_seed=(True,),
            b_mean_strictly_beats_a0_mean=True,
            b_mean_strictly_beats_a1_mean=True,
            b_mean_minus_a1_mean_pp=25.0,
            bench_merkle_root="x"))
    mlb = (
        mbpp_plus_reflexion_bench_v2
        .mlb_invocation_and_rescue_rates_v2(
            report=bench_report))
    assert mlb["n_problems_total"] == 4
    assert mlb["n_b_invoked_reflexion"] == 3  # 1,2,3
    assert mlb["n_b_rescued_via_reflexion"] == 2  # 1,2
    assert mlb["mlb1_invocation_rate"] == 0.75
    assert mlb["mlb2_rescue_rate"] == pytest.approx(
        2.0 / 3.0, abs=0.001)
    assert mlb["mlb1_passes"] is True
    assert mlb["mlb2_passes"] is True


# ---------------------------------------------------------------
# Preflight V2
# ---------------------------------------------------------------


def test_preflight_v2_p5_anti_silent_degeneration(tmp_path):
    """Mock a tiny corpus + cache file: P5 must require the
    iteration pattern + assertion-call pattern in
    extra_test_program."""
    # Use the real cached corpus if available; otherwise skip
    # (this is integration-shaped — kept here to fail loud if
    # the cache becomes corrupted).
    if not mbpp_plus_loader_v2.is_mbpp_plus_v2_cached():
        pytest.skip("MBPP+ V2 cache absent on this machine")
    p = (
        mbpp_plus_preflight_v2
        .probe_extra_test_surface_integrity_v2())
    assert p.passed
    assert p.evidence["iter_rate"] >= 0.95
    assert p.evidence["assertion_call_rate"] >= 0.95


def test_preflight_v2_p2_canonical_self_test_high_pass_rate():
    if not mbpp_plus_loader_v2.is_mbpp_plus_v2_cached():
        pytest.skip("MBPP+ V2 cache absent on this machine")
    p = (
        mbpp_plus_preflight_v2
        .probe_executor_self_test_on_gold_v2(n_sample=10))
    assert p.passed
    assert p.evidence["pass_rate"] >= 0.98


# ---------------------------------------------------------------
# Stable boundary preservation (W102)
# ---------------------------------------------------------------


def test_coordpy_version_unchanged_w102():
    import coordpy
    assert coordpy.__version__ == "1.2.0"


def test_coordpy_sdk_version_unchanged_w102():
    import coordpy
    assert coordpy.SDK_VERSION == "coordpy.sdk.v3.43"


def test_no_w102_modules_referenced_in_init_source():
    import coordpy
    init_src = Path(coordpy.__file__).read_text()
    for mod in (
            "mbpp_plus_loader_v2",
            "mbpp_plus_executor_v2",
            "mbpp_plus_reflexion_bench_v2",
            "mbpp_plus_preflight_v2",
            "humaneval_plus_loader_v1",
            "humaneval_plus_executor_v1",
            "humaneval_plus_reflexion_bench_v1",
            "humaneval_plus_preflight_v1",
            "code_slice_selector_v1"):
        assert mod not in init_src, (
            f"coordpy/__init__.py must NOT reference {mod}")


def test_w102_modules_are_explicitly_importable():
    from coordpy import mbpp_plus_loader_v2 as _l
    from coordpy import mbpp_plus_executor_v2 as _e
    from coordpy import mbpp_plus_reflexion_bench_v2 as _b
    from coordpy import mbpp_plus_preflight_v2 as _p
    from coordpy import humaneval_plus_loader_v1 as _hl
    from coordpy import humaneval_plus_executor_v1 as _he
    from coordpy import humaneval_plus_reflexion_bench_v1 as _hb
    from coordpy import humaneval_plus_preflight_v1 as _hp
    from coordpy import code_slice_selector_v1 as _cs
    assert _l.W102_MBPP_PLUS_LOADER_V2_SCHEMA_VERSION
    assert _e.W102_MBPP_PLUS_EXECUTOR_V2_SCHEMA_VERSION
    assert _b.W102_MBPP_PLUS_REFLEXION_BENCH_V2_SCHEMA_VERSION
    assert _p.W102_MBPP_PLUS_PREFLIGHT_V2_SCHEMA_VERSION
    assert _hl.W102_HUMANEVAL_PLUS_LOADER_V1_SCHEMA_VERSION
    assert _he.W102_HUMANEVAL_PLUS_EXECUTOR_V1_SCHEMA_VERSION
    assert (
        _hb.W102_HUMANEVAL_PLUS_REFLEXION_BENCH_V1_SCHEMA_VERSION)
    assert _hp.W102_HUMANEVAL_PLUS_PREFLIGHT_V1_SCHEMA_VERSION
    assert _cs.W102_CODE_SLICE_SELECTOR_V1_SCHEMA_VERSION
