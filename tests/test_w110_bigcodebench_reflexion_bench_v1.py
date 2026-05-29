"""W110 — BigCodeBench reflexion bench + executor tests (NIM-free, stdlib).

A stub ``gen`` stands in for the model so the bench runs offline; the synthetic
problems use only the standard library, so the default ``sys.executable``
executor works (no venv needed)."""
from __future__ import annotations

from coordpy.bigcodebench_loader_v1 import BigCodeBenchProblemV1
from coordpy.bigcodebench_executor_v1 import run_bigcodebench_executor_v1
from coordpy.bigcodebench_reflexion_bench_v1 import (
    BigCodeBenchBenchConfigV1,
    run_bigcodebench_reflexion_bench_v1,
    select_bigcodebench_slice_v1,
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


def _problem(task_id="BigCodeBench/0", n_libs=2):
    libs = tuple(f"lib{i}" for i in range(n_libs))
    return BigCodeBenchProblemV1(
        task_id=task_id,
        complete_prompt="def task_func(a, b):\n    \"\"\"add\"\"\"\n",
        code_prompt="def task_func(a, b):\n",
        canonical_solution="    return a + b\n",
        test=_TEST_SRC, entry_point="task_func", libs=libs)


def test_executor_self_test():
    g = run_bigcodebench_executor_v1(
        problem_id="g", test_source=_TEST_SRC, entry_point="task_func",
        candidate_code="def task_func(a, b):\n    return a + b\n")
    assert g.passed
    w = run_bigcodebench_executor_v1(
        problem_id="w", test_source=_TEST_SRC, entry_point="task_func",
        candidate_code="def task_func(a, b):\n    return a - b\n")
    assert not w.passed
    lo = run_bigcodebench_executor_v1(
        problem_id="loop", test_source=_TEST_SRC, entry_point="task_func",
        candidate_code="def task_func(a, b):\n    while True:\n        pass\n",
        timeout_s=2.0, kill_after_s=3.0)
    assert lo.timed_out and not lo.passed
    # missing entry point -> ENTRY_NOT_FOUND (rc 3)
    ne = run_bigcodebench_executor_v1(
        problem_id="ne", test_source=_TEST_SRC, entry_point="task_func",
        candidate_code="def other(a, b):\n    return a + b\n")
    assert not ne.passed and ne.returncode == 3


def test_full_bench_gold_path():
    """A0=A1=B=1.0 when the stub gen always returns the gold."""
    probs = [_problem(f"BigCodeBench/{i}") for i in range(3)]

    def gen(prompt, max_tokens, temperature):
        return _GOLD, 1
    rep = run_bigcodebench_reflexion_bench_v1(
        gen=gen, model_id="stub", subset=probs,
        config=BigCodeBenchBenchConfigV1(K_multi_sample=5, seeds=(110001,)))
    assert rep.a0_mean_pass_at_1 == 1.0
    assert rep.a1_mean_pass_at_1 == 1.0
    assert rep.b_mean_pass_at_1 == 1.0
    assert rep.b_mean_minus_a1_mean_pp == 0.0
    assert rep.bench_merkle_root  # non-empty


def test_reflexion_rescue_mlb():
    """B rescues where A1 fails: wrong on the initial prompt, gold once the
    reflexion prompt appears -> B passes, A1 fails (MLB invoked + rescued)."""
    probs = [_problem("BigCodeBench/7")]

    def gen(prompt, max_tokens, temperature):
        # the reflexion prompt contains 'reflective debugging loop'
        if "reflective debugging loop" in prompt:
            return _GOLD, 1
        return _WRONG, 1
    rep = run_bigcodebench_reflexion_bench_v1(
        gen=gen, model_id="stub", subset=probs,
        config=BigCodeBenchBenchConfigV1(K_multi_sample=5, seeds=(110001,)))
    s = rep.per_seed[0]
    assert s.a1_pass_at_1 == 0.0          # A1 never sees a reflexion prompt
    assert s.b_pass_at_1 == 1.0           # B rescued via reflexion
    assert s.per_problem_b_first_pass_idx[0] >= 1   # rescued after attempt 0


def test_slice_determinism_and_stratification():
    # 24 libs2 + 36 libs3plus problems
    pool = ([_problem(f"BigCodeBench/{i}", n_libs=2) for i in range(24)]
            + [_problem(f"BigCodeBench/{100+i}", n_libs=4) for i in range(36)])
    s1 = select_bigcodebench_slice_v1(pool, n_problems=30)
    s2 = select_bigcodebench_slice_v1(pool, n_problems=30)
    assert [p.task_id for p in s1] == [p.task_id for p in s2]  # deterministic
    assert len(s1) == 30
    # stratified: both buckets represented (not all from one)
    n2 = sum(1 for p in s1 if p.n_libs() == 2)
    n3 = sum(1 for p in s1 if p.n_libs() >= 3)
    assert n2 > 0 and n3 > 0
