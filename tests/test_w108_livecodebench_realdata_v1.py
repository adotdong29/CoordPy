"""W108 / COO-9 — LiveCodeBench real-data bug-fix regression tests.

Locks the W108 gold-path failure that the partial-scaffold offline smoke
surfaced (``n_problems: 1; A0: 0.0; A1: 0.0; B: 0.0`` even on a gold zigzag
solution).  Root cause: the real ``code_generation_lite`` ``release_v6``
corpus stores ``metadata`` as a JSON *string* (e.g.
``'{"func_name": "zigzagTraversal"}'``), but the W107 loader read
``func_name`` only when ``metadata`` was already a ``dict``; on real data
``func_name`` was silently ``""`` → the executor's entry resolver returned
``None`` → ``ENTRY_NOT_FOUND`` (rc=3) → every arm FAILed.

These tests reproduce the failure mode at unit granularity (no 134 MB
corpus needed) and prove the fix; the real-corpus tests run when the SHA-
pinned cache is present and otherwise skip.
"""
from __future__ import annotations

import hashlib
import json
import os

import pytest

from coordpy.livecodebench_loader_v1 import (
    LiveCodeBenchFunctionalTestV1,
    LiveCodeBenchProblemV1,
    _resolve_func_name,
    parse_functional_subset,
)
from coordpy.livecodebench_executor_v2 import (
    run_livecodebench_executor_v2,
)
from coordpy.livecodebench_reflexion_bench_v1 import (
    LCBBenchConfigV1,
    extract_candidate_code_v1,
    run_livecodebench_reflexion_bench_v1,
    select_livecodebench_functional_slice_v1,
)

# ---- shared fixtures -------------------------------------------------------

GOLD_ZIGZAG = (
    "class Solution:\n"
    "    def zigzagTraversal(self, grid):\n"
    "        m = len(grid); n = len(grid[0]); res = []; idx = 0\n"
    "        for i in range(m):\n"
    "            cols = range(n) if i % 2 == 0 else range(n - 1, -1, -1)\n"
    "            for j in cols:\n"
    "                if idx % 2 == 0:\n"
    "                    res.append(grid[i][j])\n"
    "                idx += 1\n"
    "        return res\n")

ZIGZAG_TESTS = [
    {"input": "[[1, 2], [3, 4]]", "output": "[1, 4]"},
    {"input": "[[2, 1], [2, 1], [2, 1]]", "output": "[2, 1, 2]"},
]

REAL_CORPUS_PATH = os.path.expanduser(
    "~/.cache/coordpy/livecodebench-test6.jsonl")
REAL_CORPUS_SHA256 = (
    "bb4c364f71921c4495a6ad15abe1a927350b720009f4933e2e71f8af0f6fd1f5")
_has_real_corpus = (
    os.path.exists(REAL_CORPUS_PATH)
    and os.path.getsize(REAL_CORPUS_PATH) > 0)


def _functional_row(*, metadata, starter_code=None, tests=None):
    """Build one schema-valid functional row (test-cases + metadata as
    the JSON strings the real corpus uses)."""
    sc = (
        starter_code
        if starter_code is not None
        else ("class Solution:\n    def zigzagTraversal"
              "(self, grid: List[List[int]]) -> List[int]:"))
    tc = tests if tests is not None else ZIGZAG_TESTS
    pub = [{"input": t["input"], "output": t["output"],
            "testtype": "functional"} for t in tc]
    return {
        "question_id": "TEST-1",
        "question_content": "stub problem statement",
        "starter_code": sc,
        "metadata": metadata,
        "public_test_cases": json.dumps(pub),
        "platform": "leetcode",
        "difficulty": "easy",
        "contest_date": "2025-01-11T00:00:00",
    }


# ---- 1-4: the exact loader bug (metadata encoding) -------------------------

def test_loader_metadata_json_string_resolves_func_name():
    """THE W108 bug: metadata as a JSON *string* must still yield
    func_name (W107 read it only when metadata was a dict → "")."""
    row = _functional_row(
        metadata=json.dumps({"func_name": "zigzagTraversal"}))
    assert _resolve_func_name(row) == "zigzagTraversal"
    (p,) = parse_functional_subset(
        (json.dumps(row) + "\n").encode("utf-8"))
    assert p.func_name == "zigzagTraversal"
    assert len(p.tests) == 2


def test_loader_metadata_dict_back_compat():
    """A native-dict metadata (older/aliased corpus) still works."""
    row = _functional_row(metadata={"func_name": "zigzagTraversal"})
    assert _resolve_func_name(row) == "zigzagTraversal"


def test_loader_starter_code_fallback():
    """No func_name in metadata ⇒ recover the entry from starter_code."""
    row = _functional_row(metadata=json.dumps({}))
    assert _resolve_func_name(row) == "zigzagTraversal"
    row2 = _functional_row(metadata="not-json-at-all")
    assert _resolve_func_name(row2) == "zigzagTraversal"


def test_loader_func_name_empty_only_when_no_signal():
    """No metadata + no def in starter_code ⇒ honestly empty (not a crash)."""
    row = {"starter_code": "class Solution:\n    pass"}
    assert _resolve_func_name(row) == ""
    row2 = _functional_row(
        metadata=json.dumps({}),
        starter_code="class Solution:\n    def __init__(self): ...")
    # dunder-only ⇒ falls back to the last def (here __init__)
    assert _resolve_func_name(row2) == "__init__"


# ---- 5-9: executor_v2 machinery on the confirmed encoding ------------------

def test_executor_v2_passes_gold_solution_method():
    """Gold Solution method + correct func_name + newline-decoded args ⇒
    exit 0 (proves the V2 machinery is correct on the real encoding)."""
    r = run_livecodebench_executor_v2(
        question_id="TEST-1", func_name="zigzagTraversal",
        tests=ZIGZAG_TESTS, candidate_code=GOLD_ZIGZAG)
    assert r.passed is True
    assert r.returncode == 0


def test_executor_v2_empty_func_name_is_entry_not_found():
    """REGRESSION GUARD for the bug's failure mode: an empty func_name
    must surface ENTRY_NOT_FOUND (rc=3), never a silent pass."""
    r = run_livecodebench_executor_v2(
        question_id="TEST-1", func_name="",
        tests=ZIGZAG_TESTS, candidate_code=GOLD_ZIGZAG)
    assert r.passed is False
    assert r.returncode == 3
    assert "ENTRY_NOT_FOUND" in r.stderr_tail


def test_executor_v2_multi_arg_newline_decode():
    """Multi-line input ⇒ one JSON value per positional arg (the confirmed
    real LiveCodeBench functional encoding)."""
    code = ("class Solution:\n"
            "    def addThree(self, a, b, c):\n"
            "        return a + b + c\n")
    tests = [{"input": "1\n2\n3", "output": "6"},
             {"input": "10\n-4\n[]", "output": "6"}]  # 2nd is a fail probe
    r_ok = run_livecodebench_executor_v2(
        question_id="T", func_name="addThree",
        tests=[tests[0]], candidate_code=code)
    assert r_ok.passed is True
    # list arg + int ⇒ TypeError inside ⇒ CASE_EXC ⇒ fail (no false pass)
    r_bad = run_livecodebench_executor_v2(
        question_id="T", func_name="addThree",
        tests=[tests[1]], candidate_code=code)
    assert r_bad.passed is False


def test_executor_v2_wrong_solution_fails():
    """A wrong solution must FAIL (no false-pass; exact-match semantics)."""
    wrong = ("class Solution:\n"
             "    def zigzagTraversal(self, grid):\n"
             "        return []\n")
    r = run_livecodebench_executor_v2(
        question_id="TEST-1", func_name="zigzagTraversal",
        tests=ZIGZAG_TESTS, candidate_code=wrong)
    assert r.passed is False
    assert r.returncode == 1
    assert "CASE_FAIL" in r.stderr_tail


def test_executor_v2_no_llm_judge_only_subprocess():
    """Anti-cheat: the executor never imports/calls any model — PASS is a
    pure subprocess exit code."""
    import coordpy.livecodebench_executor_v2 as ex
    src = ex._HARNESS_TEMPLATE
    for forbidden in ("openai", "nvidia", "requests", "urllib",
                      "anthropic", "judge"):
        assert forbidden not in src.lower()


# ---- 10: the full bench gold-path (the smoke that was failing) -------------

def _stub_gen_returning(code):
    """A deterministic gen that always returns ``code`` in a fence."""
    def _gen(prompt, max_tokens, temperature):
        return f"Here is the solution:\n```python\n{code}\n```\n", 1
    return _gen


def test_bench_gold_path_end_to_end_all_arms_pass():
    """THE smoke, now fixed: a problem carrying the correct func_name run
    through A0+A1+B with a stub gen returning the gold solution must give
    A0=A1=B=1.0 (the partial scaffold gave 0.0/0.0/0.0)."""
    problem = LiveCodeBenchProblemV1(
        question_id="3708",
        question_content="zigzag grid traversal with skip",
        starter_code=("class Solution:\n    def zigzagTraversal"
                      "(self, grid: List[List[int]]) -> List[int]:"),
        func_name="zigzagTraversal",
        platform="leetcode", difficulty="easy",
        contest_date="2025-01-11T00:00:00",
        tests=tuple(
            LiveCodeBenchFunctionalTestV1(
                input_repr=t["input"], output_repr=t["output"])
            for t in ZIGZAG_TESTS))
    report = run_livecodebench_reflexion_bench_v1(
        gen=_stub_gen_returning(GOLD_ZIGZAG),
        model_id="stub/gold",
        subset=[problem],
        config=LCBBenchConfigV1(K_multi_sample=5, seeds=(108_001,)))
    assert report.n_problems == 1
    assert report.a0_mean_pass_at_1 == 1.0
    assert report.a1_mean_pass_at_1 == 1.0
    assert report.b_mean_pass_at_1 == 1.0


def test_bench_gold_path_with_empty_func_name_fails_all_arms():
    """Same problem but func_name="" reproduces the partial-scaffold smoke
    (0/0/0) — proving the loader-supplied func_name is the load-bearing fix,
    not the bench."""
    problem = LiveCodeBenchProblemV1(
        question_id="3708", question_content="zigzag", starter_code="x",
        func_name="", platform="leetcode", difficulty="easy",
        contest_date="2025-01-11",
        tests=tuple(
            LiveCodeBenchFunctionalTestV1(
                input_repr=t["input"], output_repr=t["output"])
            for t in ZIGZAG_TESTS))
    report = run_livecodebench_reflexion_bench_v1(
        gen=_stub_gen_returning(GOLD_ZIGZAG), model_id="stub/gold",
        subset=[problem],
        config=LCBBenchConfigV1(K_multi_sample=5, seeds=(108_001,)))
    assert report.a0_mean_pass_at_1 == 0.0
    assert report.a1_mean_pass_at_1 == 0.0
    assert report.b_mean_pass_at_1 == 0.0


def test_extract_candidate_prefers_last_fence():
    """The model may restate the prompt fence then answer; take the last."""
    text = ("```python\n# restated signature\nclass Solution: ...\n```\n"
            "thinking...\n```python\nFINAL = 1\n```")
    assert extract_candidate_code_v1(
        response_text=text, starter_code="x").strip() == "FINAL = 1"


# ---- real-corpus binding (skips when the SHA-pinned cache is absent) -------

# ---- slice selector (the cheap-pilot earning surface) ----------------------

def _mk_problems(spec):
    """spec: list of (qid, difficulty, contest_date)."""
    return [
        LiveCodeBenchProblemV1(
            question_id=q, question_content="x",
            starter_code="class Solution:\n    def f(self): ...",
            func_name="f", platform="leetcode", difficulty=d,
            contest_date=dt,
            tests=(LiveCodeBenchFunctionalTestV1(
                input_repr="", output_repr="null"),))
        for q, d, dt in spec]


def _synthetic_63():
    spec = []
    for i in range(17):
        spec.append((f"e{i:02d}", "easy", f"2025-01-{(i % 28) + 1:02d}"))
    for i in range(26):
        spec.append((f"m{i:02d}", "medium", f"2025-02-{(i % 28) + 1:02d}"))
    for i in range(20):
        spec.append((f"h{i:02d}", "hard", f"2025-03-{(i % 28) + 1:02d}"))
    return _mk_problems(spec)


def test_slice_selector_deterministic():
    subset = _synthetic_63()
    a = select_livecodebench_functional_slice_v1(subset, n_problems=30)
    b = select_livecodebench_functional_slice_v1(
        list(reversed(subset)), n_problems=30)
    assert [p.question_id for p in a] == [p.question_id for p in b]


def test_slice_selector_difficulty_stratified():
    """30 of {17 easy, 26 medium, 20 hard} ⇒ largest-remainder 8/12/10 —
    guarantees medium+hard (no all-easy A1-saturation slice)."""
    sl = select_livecodebench_functional_slice_v1(
        _synthetic_63(), n_problems=30)
    from collections import Counter
    mix = Counter(p.difficulty for p in sl)
    assert dict(mix) == {"easy": 8, "medium": 12, "hard": 10}
    assert len(sl) == 30


def test_slice_selector_sorted_by_date_then_id():
    sl = select_livecodebench_functional_slice_v1(
        _synthetic_63(), n_problems=30)
    keys = [(p.contest_date, p.question_id) for p in sl]
    assert keys == sorted(keys)


def test_slice_selector_caps_at_subset_size():
    small = _mk_problems([("a", "easy", "2025-01-01"),
                          ("b", "hard", "2025-01-02")])
    sl = select_livecodebench_functional_slice_v1(small, n_problems=30)
    assert len(sl) == 2


@pytest.mark.skipif(not _has_real_corpus, reason="real corpus not cached")
def test_real_corpus_sha_pin_matches():
    h = hashlib.sha256()
    with open(REAL_CORPUS_PATH, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    assert h.hexdigest() == REAL_CORPUS_SHA256


@pytest.mark.skipif(not _has_real_corpus, reason="real corpus not cached")
def test_real_corpus_functional_subset_all_have_func_name():
    with open(REAL_CORPUS_PATH, "rb") as f:
        raw = f.read()
    subset = parse_functional_subset(raw)
    assert len(subset) >= 30
    # The whole point of the fix: every functional problem resolves an entry.
    assert all(p.func_name for p in subset)


@pytest.mark.skipif(not _has_real_corpus, reason="real corpus not cached")
def test_real_corpus_gold_zigzag_passes_end_to_end():
    with open(REAL_CORPUS_PATH, "rb") as f:
        raw = f.read()
    subset = parse_functional_subset(raw)
    zig = [p for p in subset if p.func_name == "zigzagTraversal"]
    assert zig, "zigzag problem expected in release_v6 functional subset"
    p = zig[0]
    r = run_livecodebench_executor_v2(
        question_id=p.question_id, func_name=p.func_name,
        tests=[{"input": t.input_repr, "output": t.output_repr}
               for t in p.tests],
        candidate_code=GOLD_ZIGZAG)
    assert r.passed is True
