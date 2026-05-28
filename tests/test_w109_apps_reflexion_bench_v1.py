"""W109 / COO-9 — APPS contamination-control reflexion-bench tests.

Locks the W109 APPS sequential-reflexion bench (the contamination-EXPOSED
control counterpart to the W108 LiveCodeBench bench).  The mechanism is
byte-identical in shape to W88/W89/W102/W108; these tests prove:

* fence extraction;
* the deterministic OUTCOME-BLIND difficulty-stratified slice + stable CID;
* the full-bench gold-path smoke (A0 = A1 = B = 1.0 with a gold stub gen);
* reflexion is load-bearing when the initial attempt fails (B rescues past A1
  → MLB-2 invoked + rescued);
* the ``max_tests_per_problem`` cap applies identically to all arms;
* (guarded) the REAL pinned corpus binds to the locked 30-slice CID.
"""
from __future__ import annotations

import hashlib
import json
import os

import pytest

from coordpy.apps_loader_v1 import (
    AppsFunctionalProblemV1,
    AppsFunctionalTestV1,
)
from coordpy.apps_reflexion_bench_v1 import (
    AppsBenchConfigV1,
    extract_candidate_code_v1,
    run_apps_reflexion_bench_v1,
    select_apps_functional_slice_v1,
)

_GOLD = (
    "class Solution:\n    def add(self, a, b):\n        return a + b\n")
_WRONG = (
    "class Solution:\n    def add(self, a, b):\n        return a - b\n")


def _fence(code: str) -> str:
    return f"Here is my answer:\n```python\n{code}```\n"


def _problem(pid: str, difficulty: str = "interview") -> AppsFunctionalProblemV1:
    return AppsFunctionalProblemV1(
        problem_id=pid, question="implement add(a, b)", starter_code="",
        fn_name="add", difficulty=difficulty, url="http://x",
        tests=(
            AppsFunctionalTestV1(args_repr=json.dumps([2, 3]),
                                 output_repr=json.dumps(5)),
            AppsFunctionalTestV1(args_repr=json.dumps([10, 20]),
                                 output_repr=json.dumps([30])),  # wrapper
        ))


def test_extract_candidate_prefers_last_fence():
    text = "```python\nWRONG\n```\nactually:\n```python\nRIGHT\n```"
    assert extract_candidate_code_v1(
        response_text=text, starter_code="").strip() == "RIGHT"


def test_slice_is_deterministic_outcome_blind_and_stratified():
    subset = [_problem(str(i), "interview") for i in range(20)] + \
             [_problem(str(100 + i), "introductory") for i in range(10)]
    s1 = select_apps_functional_slice_v1(subset, n_problems=12)
    s2 = select_apps_functional_slice_v1(subset, n_problems=12)
    ids1 = [p.problem_id for p in s1]
    assert ids1 == [p.problem_id for p in s2]  # deterministic
    # difficulty-stratified (largest-remainder proportional to 20:10 = 2:1)
    diffs = [p.difficulty for p in s1]
    assert diffs.count("interview") == 8 and diffs.count("introductory") == 4
    # emitted in problem_id order (stable CID)
    assert ids1 == sorted(ids1, key=lambda x: int(x))


def test_full_bench_gold_path_a0_a1_b_all_pass():
    subset = [_problem("1"), _problem("2"), _problem("3")]
    gen = lambda prompt, mt, temp: (_fence(_GOLD), 1)  # noqa: E731
    cfg = AppsBenchConfigV1(K_multi_sample=5, seeds=(109_001,))
    rep = run_apps_reflexion_bench_v1(
        gen=gen, model_id="stub", subset=subset, config=cfg)
    assert rep.a0_mean_pass_at_1 == 1.0
    assert rep.a1_mean_pass_at_1 == 1.0
    assert rep.b_mean_pass_at_1 == 1.0
    assert rep.b_mean_minus_a1_mean_pp == 0.0


def test_reflexion_is_load_bearing_when_initial_fails():
    """Initial prompt → WRONG; reflexion prompt → GOLD. So A0/A1 fail but B
    rescues (first_pass_idx>0): the MLB-2 mechanism is exercised."""
    subset = [_problem("1")]

    def gen(prompt, mt, temp):
        if "reflective debugging loop" in prompt:
            return _fence(_GOLD), 1
        return _fence(_WRONG), 1

    cfg = AppsBenchConfigV1(K_multi_sample=5, seeds=(109_001,))
    rep = run_apps_reflexion_bench_v1(
        gen=gen, model_id="stub", subset=subset, config=cfg)
    s = rep.per_seed[0]
    assert s.per_problem_a0_passed == (False,)
    assert s.per_problem_a1_passed == (False,)   # all 5 samples are WRONG
    assert s.per_problem_b_passed == (True,)     # reflexion rescued
    assert s.per_problem_b_first_pass_idx[0] == 1  # rescued at attempt 2


def test_max_tests_cap_truncates_uniformly():
    p = AppsFunctionalProblemV1(
        problem_id="1", question="q", starter_code="", fn_name="add",
        difficulty="interview", url="",
        tests=tuple(
            AppsFunctionalTestV1(args_repr=json.dumps([i, 0]),
                                 output_repr=json.dumps(i))
            for i in range(100)))
    gen = lambda prompt, mt, temp: (_fence(_GOLD), 1)  # noqa: E731
    cfg = AppsBenchConfigV1(K_multi_sample=2, seeds=(1,),
                            max_tests_per_problem=5)
    rep = run_apps_reflexion_bench_v1(
        gen=gen, model_id="stub", subset=[p], config=cfg)
    # gold passes the first 5 (and all) → still passes under the cap
    assert rep.a0_mean_pass_at_1 == 1.0


_PIN = "f6c44d76be0eea7669f0ccbd90b6b45fb03a4327d06682073b5cd8f905310918"
_CACHE = os.path.expanduser("~/.cache/coordpy/apps-test.jsonl")
_LOCKED_SLICE_CID = (
    "783687d6109d2e452aba8a32952b5569ed7c03d8aa1d040f1a22ef18688c6dcc")


@pytest.mark.skipif(
    not (os.path.exists(_CACHE) and os.path.getsize(_CACHE) > 0),
    reason="real APPS corpus not fetched (operator step)")
def test_real_corpus_binds_to_locked_slice_cid():
    from coordpy.apps_loader_v1 import load_apps_call_based_v1
    subset = load_apps_call_based_v1(expected_sha256=_PIN)
    assert len(subset) == 38
    assert all(p.fn_name for p in subset)
    sl = select_apps_functional_slice_v1(subset, n_problems=30)
    ids = [str(p.problem_id) for p in sl]
    cid = hashlib.sha256(json.dumps(
        {"kind": "w109_apps_pilot_slice_v1", "problem_ids": ids},
        sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    assert cid == _LOCKED_SLICE_CID
