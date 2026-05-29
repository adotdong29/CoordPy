"""W111 — executor-grounded structured-failure patcher (M3) tests (NIM-free).

A stub ``gen`` stands in for the model so the bench runs offline; the synthetic
problem uses only the standard library, so the default ``sys.executable``
executor works (no venv needed). Mirrors
``tests/test_w110_bigcodebench_reflexion_bench_v1.py``.
"""
from __future__ import annotations

from coordpy.bigcodebench_loader_v1 import BigCodeBenchProblemV1
from coordpy.executor_grounded_patcher_v1 import (
    PatcherBenchConfigV1,
    parse_failure_digest_v1,
    run_executor_grounded_patcher_bench_v1,
)

_TEST_SRC = (
    "import unittest\n"
    "class TestCases(unittest.TestCase):\n"
    "    def test_a(self):\n"
    "        self.assertEqual(task_func(2, 3), 5)\n"
    "    def test_b(self):\n"
    "        self.assertEqual(task_func(10, -4), 6)\n")
_GOLD = "```python\ndef task_func(a, b):\n    return a + b\n```"
_WRONG = "```python\ndef task_func(a, b):\n    return a - b\n```"

_PATCH_MARKER = "EXECUTOR-GROUNDED PATCHING"


def _problem(task_id="BigCodeBench/0", n_libs=2):
    libs = tuple(f"lib{i}" for i in range(n_libs))
    return BigCodeBenchProblemV1(
        task_id=task_id,
        complete_prompt="def task_func(a, b):\n    \"\"\"add\"\"\"\n",
        code_prompt="def task_func(a, b):\n",
        canonical_solution="    return a + b\n",
        test=_TEST_SRC, entry_point="task_func", libs=libs)


def _gold_gen(prompt, max_tokens, temperature):
    return _GOLD, 1


def _rescue_gen(prompt, max_tokens, temperature):
    """Initial attempts fail; the structured-patch turn fixes it. This is the
    M3 mechanism under test: rescue a failing initial via executor grounding."""
    if _PATCH_MARKER in prompt:
        return _GOLD, 1
    return _WRONG, 1


def _always_wrong_gen(prompt, max_tokens, temperature):
    return _WRONG, 1


# --------------------------- digest parser --------------------------------

def test_parse_failure_digest_extracts_fields():
    tail = (
        "FAIL: test_case_3 (_bcb_solution_mod.TestCases.test_case_3)\n"
        "----------------------------------------------------------------------\n"
        "Traceback (most recent call last):\n"
        '  File "<test>", line 31, in test_case_3\n'
        "AssertionError: None != '/mock_dir/access.log.123'\n")
    d = parse_failure_digest_v1(stderr_tail=tail, timed_out=False)
    assert "test_case_3" in d.failing_tests
    assert d.exception_type == "AssertionError"
    assert d.actual_repr == "None"
    assert "/mock_dir/access.log.123" in d.expected_repr


def test_parse_failure_digest_timeout():
    d = parse_failure_digest_v1(stderr_tail="", timed_out=True)
    assert d.exception_type == "Timeout"
    assert d.failing_tests == ()


def test_parse_failure_digest_non_assertion_exception():
    tail = ("ERROR: test_x (M.T.test_x)\n"
            "ValueError: bad shape\n")
    d = parse_failure_digest_v1(stderr_tail=tail, timed_out=False)
    assert d.exception_type == "ValueError"
    assert "test_x" in d.failing_tests


# --------------------------- bench arms -----------------------------------

def test_gold_path_all_arms_pass():
    cfg = PatcherBenchConfigV1(K_multi_sample=5, seeds=(111_001,))
    rep = run_executor_grounded_patcher_bench_v1(
        gen=_gold_gen, model_id="stub", subset=[_problem()], config=cfg)
    assert rep.a0_mean_pass_at_1 == 1.0
    assert rep.a1_mean_pass_at_1 == 1.0
    assert rep.m3_mean_pass_at_1 == 1.0
    assert rep.m3_mean_minus_a1_mean_pp == 0.0


def test_m3_rescues_where_a1_fails():
    """The core mechanism: initial sample fails, the structured-patch turn
    rescues. A1 (all initial samples) fails; M3 passes via the patch loop."""
    cfg = PatcherBenchConfigV1(K_multi_sample=5, seeds=(111_001,))
    rep = run_executor_grounded_patcher_bench_v1(
        gen=_rescue_gen, model_id="stub", subset=[_problem()], config=cfg)
    s = rep.per_seed[0]
    assert s.a1_pass_at_1 == 0.0           # every initial sample is wrong
    assert s.m3_pass_at_1 == 1.0           # the patch turn rescued it
    assert s.per_problem_m3_first_pass_idx[0] == 1   # rescued at first patch
    assert rep.m3_mean_minus_a1_mean_pp == 100.0


def test_m3_never_passes_when_unfixable():
    cfg = PatcherBenchConfigV1(K_multi_sample=5, seeds=(111_001,))
    rep = run_executor_grounded_patcher_bench_v1(
        gen=_always_wrong_gen, model_id="stub", subset=[_problem()], config=cfg)
    s = rep.per_seed[0]
    assert s.m3_pass_at_1 == 0.0
    assert s.per_problem_m3_first_pass_idx[0] == -1


def test_budget_byte_exact_a1_equals_m3():
    """A1 and M3 must spend the SAME K model calls (the same-budget contract)."""
    captured = []

    def _counting_gen(prompt, max_tokens, temperature):
        captured.append(prompt)
        return _WRONG, 1

    cfg = PatcherBenchConfigV1(K_multi_sample=5, seeds=(111_001,))
    run_executor_grounded_patcher_bench_v1(
        gen=_counting_gen, model_id="stub", subset=[_problem()], config=cfg)
    # A0 (1) + A1 (5) + M3 (5) = 11 calls for one problem.
    assert len(captured) == 11


def test_m3_patch_prompt_never_leaks_test_source():
    """FAIRNESS GUARD: M3 must see ONLY the executor stderr, never the hidden
    ``test`` source (that would be oracle leakage). The patch prompts must not
    contain the test body."""
    captured = []

    def _capturing_gen(prompt, max_tokens, temperature):
        captured.append(prompt)
        return _WRONG, 1

    cfg = PatcherBenchConfigV1(K_multi_sample=5, seeds=(111_001,))
    run_executor_grounded_patcher_bench_v1(
        gen=_capturing_gen, model_id="stub", subset=[_problem()], config=cfg)
    patch_prompts = [p for p in captured if _PATCH_MARKER in p]
    assert patch_prompts, "expected at least one structured-patch prompt"
    for p in patch_prompts:
        # the literal test source must never appear in an M3 prompt
        assert "class TestCases" not in p
        assert "def test_a" not in p
        assert "self.assertEqual(task_func(2, 3), 5)" not in p


def test_outcomes_deterministic_and_cid_rehash_stable():
    """Outcome arrays are a deterministic function of the (gen, executor) — the
    bench Merkle itself embeds real wall_ms (a per-run audit root, exactly like
    the W110 bench), so it is NOT a cross-run constant; the audit property is
    that a report's CID re-derives from its recorded ``to_dict`` (offline
    re-verifiability)."""
    cfg = PatcherBenchConfigV1(K_multi_sample=5, seeds=(111_001,))
    r1 = run_executor_grounded_patcher_bench_v1(
        gen=_gold_gen, model_id="stub", subset=[_problem()], config=cfg)
    r2 = run_executor_grounded_patcher_bench_v1(
        gen=_gold_gen, model_id="stub", subset=[_problem()], config=cfg)
    # deterministic outcomes (gold gen + deterministic a+b executor)
    assert (r1.per_seed[0].per_problem_m3_passed
            == r2.per_seed[0].per_problem_m3_passed == (True,))
    assert r1.a1_mean_pass_at_1 == r2.a1_mean_pass_at_1 == 1.0
    # CID re-derives from the recorded dict (offline re-verifiability, gate G8)
    from coordpy.executor_grounded_patcher_v1 import _sha256_hex
    assert r1.cid() == _sha256_hex(
        {"kind": "w111_patcher_bench_report_v1", "report": r1.to_dict()})
