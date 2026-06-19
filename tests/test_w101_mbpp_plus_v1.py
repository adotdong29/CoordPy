"""W101 — MBPP+ infrastructure unit tests (NIM-free)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from coordpy import mbpp_plus_executor_v1
from coordpy import mbpp_plus_loader_v1
from coordpy import mbpp_plus_preflight_v1
from coordpy import mbpp_plus_reflexion_bench_v1


# ---------------------------------------------------------------
# Loader
# ---------------------------------------------------------------


def test_loader_schema_version():
    assert mbpp_plus_loader_v1.W101_MBPP_PLUS_LOADER_V1_SCHEMA_VERSION == (
        "coordpy.mbpp_plus_loader_v1.v1")


def test_loader_constants_documented():
    assert (
        "github.com/evalplus/evalplus"
        in mbpp_plus_loader_v1.MBPP_PLUS_CANONICAL_URL_V020)
    assert (
        len(
            mbpp_plus_loader_v1.MBPP_PLUS_EXPECTED_SHA256_V020)
        == 64)
    assert (
        mbpp_plus_loader_v1.MBPP_PLUS_EXPECTED_PROBLEM_COUNT_MIN
        < mbpp_plus_loader_v1.MBPP_PLUS_EXPECTED_PROBLEM_COUNT_MAX)


def test_loader_refuses_unpinned_sha_by_default():
    """Loader must refuse to operate against the all-zeros
    placeholder pin unless allow_unpinned=True."""
    with pytest.raises(
            mbpp_plus_loader_v1.MbppPlusCorpusError) as ei:
        mbpp_plus_loader_v1.load_mbpp_plus_corpus_v1(
            cache_path="/nonexistent/path",
            url="https://example.invalid/never-fetched",
            timeout=0.1,
            allow_unpinned=False)
    assert "SHA-256 pin" in str(ei.value)


def test_loader_parse_jsonl_synthetic():
    """Construct a synthetic MBPP+-shaped JSONL payload and
    verify the parser handles it (no network; no cache; no
    SHA pin)."""
    rows = []
    for i in range(360):
        entry = f"my_func_{i}"
        rows.append({
            "task_id": f"Mbpp/{i+1}",
            "prompt": f"Write a function that does thing {i}.",
            "code": f"def {entry}(x):\n    return x + {i}",
            "assertion": f"assert {entry}(1) == {1 + i}",
            "test_list": [f"assert {entry}(1) == {1 + i}"],
            "plus_input": ["(1,)", "(2,)", "(3,)"],
            "plus_output": [
                f"{1+i}", f"{2+i}", f"{3+i}"],
        })
    payload = "\n".join(
        json.dumps(r) for r in rows).encode("utf-8")
    parsed = mbpp_plus_loader_v1.parse_mbpp_plus_jsonl_gz(
        payload)
    assert len(parsed) == 360
    assert parsed[0].task_id == "Mbpp/1"
    assert parsed[0].entry_point == "my_func_0"
    assert parsed[0].plus_input == ("(1,)", "(2,)", "(3,)")
    assert parsed[0].plus_output == ("1", "2", "3")
    cid = parsed[0].problem_cid()
    assert isinstance(cid, str) and len(cid) == 64


def test_loader_parse_jsonl_rejects_too_few_problems():
    """A payload with only 5 problems should raise (below the
    min threshold)."""
    payload_rows = [
        json.dumps({
            "task_id": f"Mbpp/{i+1}",
            "prompt": "p",
            "code": "def f():\n    return 0",
            "assertion": "assert f() == 0",
            "test_list": ["assert f() == 0"],
            "plus_input": ["()"],
            "plus_output": ["0"],
        })
        for i in range(5)
    ]
    payload = "\n".join(payload_rows).encode("utf-8")
    with pytest.raises(
            mbpp_plus_loader_v1.MbppPlusCorpusError):
        mbpp_plus_loader_v1.parse_mbpp_plus_jsonl_gz(payload)


def test_loader_is_mbpp_plus_cached_false_on_missing(tmp_path):
    path = tmp_path / "nonexistent.jsonl.gz"
    assert not mbpp_plus_loader_v1.is_mbpp_plus_cached(
        cache_path=str(path))


# ---------------------------------------------------------------
# Executor
# ---------------------------------------------------------------


def _make_problem(
        *, code: str, entry: str = "f",
        base_tests: list[str] | None = None,
        plus_input: list[str] | None = None,
        plus_output: list[str] | None = None,
) -> mbpp_plus_loader_v1.MbppPlusProblemV1:
    return mbpp_plus_loader_v1.MbppPlusProblemV1(
        task_id="Mbpp/test",
        prompt="test",
        code=code,
        base_test_list=tuple(base_tests or []),
        plus_input=tuple(plus_input or []),
        plus_output=tuple(plus_output or []),
        assertion=(
            base_tests[0] if base_tests
            else f"assert {entry}() == 0"),
        entry_point=entry,
    )


def test_executor_passes_on_correct_canonical_solution():
    p = _make_problem(
        code="def add(a, b):\n    return a + b",
        entry="add",
        base_tests=["assert add(2, 3) == 5"],
        plus_input=["(1, 1)", "(10, -5)"],
        plus_output=["2", "5"],
    )
    res = (
        mbpp_plus_executor_v1.run_mbpp_plus_executor_v1(
            problem=p, candidate_code=p.code))
    assert res.passed
    assert res.n_base_passed == 1
    assert res.n_base_total == 1
    assert res.n_plus_passed == 2
    assert res.n_plus_total == 2
    assert res.mode == "base_and_plus"


def test_executor_fails_on_wrong_plus_test():
    """A candidate that passes the BASE test but fails an
    EvalPlus extra test must FAIL overall."""
    p = _make_problem(
        # Bug: ignores second arg; passes the base test by
        # coincidence but fails the plus test for (10, -5).
        code=(
            "def add(a, b):\n"
            "    return a + 3\n"),
        entry="add",
        base_tests=["assert add(2, 3) == 5"],
        plus_input=["(10, -5)"],
        plus_output=["5"],
    )
    res = (
        mbpp_plus_executor_v1.run_mbpp_plus_executor_v1(
            problem=p, candidate_code=p.code))
    assert not res.passed
    # base passes (add(2,3)==5 coincidentally), plus fails
    assert res.n_base_passed == 1
    assert res.n_plus_passed == 0


def test_executor_mode_base_only():
    """In base_only mode, the executor ignores the plus tests."""
    p = _make_problem(
        code="def f():\n    return 0",
        entry="f",
        base_tests=["assert f() == 0"],
        plus_input=["()"],
        plus_output=["999"],  # would fail in plus mode
    )
    res = (
        mbpp_plus_executor_v1.run_mbpp_plus_executor_v1(
            problem=p, candidate_code=p.code,
            mode="base_only"))
    assert res.passed


def test_build_plus_assertions_handles_asymmetric():
    """Asymmetric plus_input vs plus_output → no assertions
    emitted (refuses to silently mis-pair)."""
    p = _make_problem(
        code="def f():\n    return 0",
        entry="f",
        plus_input=["(1,)", "(2,)"],
        plus_output=["0"],  # length mismatch
    )
    asserts = (
        mbpp_plus_executor_v1.build_plus_assertions(p))
    assert asserts == []


# ---------------------------------------------------------------
# Bench
# ---------------------------------------------------------------


def test_bench_select_subset_deterministic():
    """Same seed → same subset across calls."""
    corpus = tuple(
        _make_problem(
            code=f"def f_{i}():\n    return {i}",
            entry=f"f_{i}",
            base_tests=[f"assert f_{i}() == {i}"])
        for i in range(50))
    s1 = (
        mbpp_plus_reflexion_bench_v1
        .select_mbpp_plus_subset_v1(
            corpus=corpus, n_problems=10, seed=101_001))
    s2 = (
        mbpp_plus_reflexion_bench_v1
        .select_mbpp_plus_subset_v1(
            corpus=corpus, n_problems=10, seed=101_001))
    s3 = (
        mbpp_plus_reflexion_bench_v1
        .select_mbpp_plus_subset_v1(
            corpus=corpus, n_problems=10, seed=101_002))
    assert s1 == s2
    assert s1 != s3
    assert len(s1) == 10


def test_bench_runs_with_fake_gen():
    """End-to-end bench smoke test using a fake gen that emits
    the canonical solution.  Validates A0 + A1 + B all pass on
    a tiny synthetic corpus."""
    corpus = tuple(
        _make_problem(
            code=(
                f"def f_{i}(x):\n    return x + {i}"),
            entry=f"f_{i}",
            base_tests=[f"assert f_{i}(0) == {i}"],
            plus_input=["(1,)"],
            plus_output=[f"{1 + i}"])
        for i in range(5))

    def gen(prompt, max_tokens, temperature):
        # Inspect the prompt to figure out which problem we are
        # answering and emit the canonical solution.
        for i in range(5):
            entry = f"f_{i}"
            if f"f_{i}(0)" in prompt:
                code = (
                    f"```python\ndef {entry}(x):\n"
                    f"    return x + {i}\n```")
                return code, 10
        return "```python\ndef f():\n    return 0\n```", 10

    cfg = (
        mbpp_plus_reflexion_bench_v1.MbppPlusBenchConfigV1(
            n_problems=5,
            K_multi_sample=5,
            seeds=(101_001,)))
    report = (
        mbpp_plus_reflexion_bench_v1
        .run_mbpp_plus_reflexion_bench_v1(
            gen=gen,
            model_id="fake/synthetic",
            corpus=corpus,
            config=cfg))
    assert report.a0_mean_pass_at_1 == 1.0
    assert report.a1_mean_pass_at_1 == 1.0
    assert report.b_mean_pass_at_1 == 1.0
    assert report.b_mean_minus_a1_mean_pp == 0.0
    assert len(report.per_seed) == 1
    assert report.per_seed[0].n_problems == 5


# ---------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------


def test_preflight_published_drop_constants_sane():
    """Published EvalPlus drops should be in a plausible range
    and the min should be ≤ the mean ≤ the max."""
    drops = list(
        mbpp_plus_preflight_v1
        .MBPP_PLUS_PUBLISHED_BASE_TO_PLUS_DROP_PP
        .values())
    assert all(5.0 <= d <= 30.0 for d in drops)
    assert (
        min(drops)
        <= sum(drops) / len(drops)
        <= max(drops))


def test_preflight_p3_predicts_unsaturated_mbpp_plus_a1():
    """The published-drop extrapolation must predict an A1 well
    below the saturation threshold."""
    p = (
        mbpp_plus_preflight_v1
        .probe_a1_failure_residual_v1())
    assert p.passed
    assert (
        p.evidence["a1_predicted_mbpp_plus_pp"]
        <= mbpp_plus_preflight_v1.A1_SATURATION_THRESHOLD_PP)
    # Must leave at least the +5 pp Phase 2 margin floor.
    assert (
        p.evidence["a1_predicted_mbpp_plus_pp"]
        + mbpp_plus_preflight_v1.W101_PHASE2_MARGIN_FLOOR_PP
        <= 100.0)


def test_preflight_p4_argument_long_enough():
    p = (
        mbpp_plus_preflight_v1
        .probe_decomposition_argument_v1())
    assert p.passed
    assert p.evidence["argument_length_chars"] >= 800


def test_preflight_anti_pattern_guard_on_clean_module(
        tmp_path):
    """A module file with no anti-pattern tokens should PASS the
    guard."""
    clean = tmp_path / "clean_bench.py"
    clean.write_text(
        '"""W101-shape bench module.\n"""\n'
        "from coordpy.mbpp_plus_executor_v1 import "
        "run_mbpp_plus_executor_v1\n"
        "def run_bench():\n"
        "    return 'ok'\n")
    p = (
        mbpp_plus_preflight_v1
        .probe_addr_no_anti_pattern_v1(
            bench_module_path=clean))
    assert p.passed
    assert p.evidence["hits"] == []


def test_preflight_anti_pattern_guard_on_dirty_module(
        tmp_path):
    """A module file that mentions a forbidden token should FAIL
    the guard."""
    dirty = tmp_path / "dirty_bench.py"
    dirty.write_text(
        "from coordpy.bounded_window_baseline_v1 import "
        "BoundedWindowBaselineV1\n"
        "def run_bench():\n"
        "    return 'ok'\n")
    p = (
        mbpp_plus_preflight_v1
        .probe_addr_no_anti_pattern_v1(
            bench_module_path=dirty))
    assert not p.passed
    assert "bounded_window" in p.evidence["hits"]


def test_preflight_corpus_integrity_deferred_when_cache_absent(
        tmp_path):
    """If the MBPP+ cache is absent, P1 must DEFER cleanly
    (passed=False) with an explicit operator-facing summary."""
    p = (
        mbpp_plus_preflight_v1
        .probe_corpus_integrity_v1(
            cache_path=str(tmp_path / "absent.jsonl.gz")))
    assert not p.passed
    assert "operator" in p.summary.lower()


def test_preflight_addr_mechanism_load_bearing_reads_mining(
        tmp_path):
    """Provide a synthetic mining report with a high rescue
    fraction on one bench; AddrW101-P1 should PASS."""
    mining = {
        "schema": "coordpy.w101_arsenal_mining_v1",
        "humaneval": {
            "bench_kind": "humaneval",
            "mechanism_load_bearing_estimate": {
                "fraction_b_wins_from_reflexion_rescue": (
                    0.12),
                "n_b_wins_total": 100,
                "n_b_only_rescues": 12,
            },
        },
        "mbpp": {
            "bench_kind": "mbpp",
            "mechanism_load_bearing_estimate": {
                "fraction_b_wins_from_reflexion_rescue": (
                    0.03),
                "n_b_wins_total": 100,
                "n_b_only_rescues": 3,
            },
        },
    }
    p_path = tmp_path / "mining.json"
    p_path.write_text(json.dumps(mining))
    res = (
        mbpp_plus_preflight_v1
        .probe_addr_mechanism_load_bearing_v1(
            arsenal_mining_report_path=p_path))
    assert res.passed
    fr = {
        f["bench"]: f["rescue_fraction"]
        for f in res.evidence["findings"]}
    assert fr["humaneval"] >= 0.10
    assert fr["mbpp"] < 0.05


def test_preflight_addr_cluster_structure_reads_mining(
        tmp_path):
    mining = {
        "humaneval": {
            "aggregate": {
                "n_a1_only_wins": 3,
                "n_b_only_wins": 8,
                "n_shared_wins": 74,
                "n_shared_fails": 5,
            },
        },
        "mbpp": {
            "aggregate": {
                "n_a1_only_wins": 3,
                "n_b_only_wins": 5,
                "n_shared_wins": 121,
                "n_shared_fails": 21,
            },
        },
    }
    p_path = tmp_path / "mining.json"
    p_path.write_text(json.dumps(mining))
    res = (
        mbpp_plus_preflight_v1
        .probe_addr_cluster_structure_v1(
            arsenal_mining_report_path=p_path))
    assert res.passed
    findings = {
        f["bench"]: f for f in res.evidence["findings"]}
    assert findings["humaneval"]["total"] == 90
    assert findings["mbpp"]["total"] == 150


def test_preflight_run_end_to_end_deferred_when_cache_absent(
        tmp_path):
    """End-to-end preflight run when MBPP+ cache absent + a
    valid mining report exists: P1+P2 DEFERRED, substantive
    probes PASS, overall verdict overall_passes=False with
    n_required preserving the discipline that the cheap pilot
    is conditional."""
    mining = {
        "humaneval": {
            "aggregate": {
                "n_a1_only_wins": 3,
                "n_b_only_wins": 8,
                "n_shared_wins": 74,
                "n_shared_fails": 5,
            },
            "mechanism_load_bearing_estimate": {
                "fraction_b_wins_from_reflexion_rescue": (
                    0.0976),
                "n_b_wins_total": 82,
                "n_b_only_rescues": 8,
            },
        },
        "mbpp": {
            "aggregate": {
                "n_a1_only_wins": 3,
                "n_b_only_wins": 5,
                "n_shared_wins": 121,
                "n_shared_fails": 21,
            },
            "mechanism_load_bearing_estimate": {
                "fraction_b_wins_from_reflexion_rescue": (
                    0.0397),
                "n_b_wins_total": 126,
                "n_b_only_rescues": 5,
            },
            "per_seed": {
                "90001": {"a1_pass_rate": 0.90},
                "90002": {"a1_pass_rate": 0.70},
                "90003": {"a1_pass_rate": 0.833},
                "90004": {"a1_pass_rate": 0.90},
                "90005": {"a1_pass_rate": 0.80},
            },
        },
    }
    mining_path = tmp_path / "mining.json"
    mining_path.write_text(json.dumps(mining))
    bench_module = (
        Path(mbpp_plus_reflexion_bench_v1.__file__))
    verdict = (
        mbpp_plus_preflight_v1
        .run_mbpp_plus_preflight_v1(
            cache_path=str(tmp_path / "absent.jsonl.gz"),
            arsenal_mining_report_path=mining_path,
            bench_module_path=bench_module,
            run_executor_self_test=False))
    # 8 probes total; 2 deferred (P1 + P2); 6 substantive PASS.
    assert len(verdict.probes) == 8
    n_pass = sum(1 for p in verdict.probes if p.passed)
    assert n_pass == 6
    # n_required allows ONE deferred probe; with TWO deferred,
    # overall must FAIL until corpus is fetched.
    assert not verdict.overall_passes


# ---------------------------------------------------------------
# Stable boundary preservation
# ---------------------------------------------------------------


def test_coordpy_version_unchanged():
    import coordpy
    assert coordpy.__version__ == "1.2.1"


def test_coordpy_sdk_version_unchanged():
    import coordpy
    assert coordpy.SDK_VERSION == "coordpy.sdk.v3.43"


def test_no_w101_modules_referenced_in_init_source():
    """The W101 modules must be explicit-import only — they
    must NOT be referenced in coordpy/__init__.py's source.
    (Note: `hasattr` would be a false positive because Python
    auto-attaches an imported submodule to the parent package;
    the source-text check is the authoritative one.)"""
    import coordpy
    init_src = Path(coordpy.__file__).read_text()
    for mod in (
            "mbpp_plus_loader_v1",
            "mbpp_plus_executor_v1",
            "mbpp_plus_reflexion_bench_v1",
            "mbpp_plus_preflight_v1"):
        assert mod not in init_src, (
            f"coordpy/__init__.py must NOT reference {mod}")


def test_w101_modules_are_explicitly_importable():
    """Confirm explicit-import path works even though not in
    __init__."""
    from coordpy import mbpp_plus_loader_v1 as _l
    from coordpy import mbpp_plus_executor_v1 as _e
    from coordpy import mbpp_plus_reflexion_bench_v1 as _b
    from coordpy import mbpp_plus_preflight_v1 as _p
    assert _l.W101_MBPP_PLUS_LOADER_V1_SCHEMA_VERSION
    assert _e.W101_MBPP_PLUS_EXECUTOR_V1_SCHEMA_VERSION
    assert _b.W101_MBPP_PLUS_REFLEXION_BENCH_V1_SCHEMA_VERSION
    assert _p.W101_MBPP_PLUS_PREFLIGHT_V1_SCHEMA_VERSION
