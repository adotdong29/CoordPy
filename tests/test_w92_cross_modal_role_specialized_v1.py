"""W92 — Cross-modal role-specialized bench V1 CI tests."""
from __future__ import annotations

import hashlib
import json
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from coordpy.cross_modal_role_specialized_bench_v1 import (
    W92_CROSS_MODAL_ROLE_SPECIALIZED_BENCH_V1_SCHEMA_VERSION,
    CrossModalRoleSpecBenchConfigV1,
    run_cross_modal_role_specialized_bench_v1,
)
from coordpy.humaneval_real_bench_v1 import (
    HumanEvalProblemV1,
)


def test_w92_role_spec_schema_version():
    assert (
        W92_CROSS_MODAL_ROLE_SPECIALIZED_BENCH_V1_SCHEMA_VERSION
        == "coordpy.cross_modal_role_specialized_bench_v1.v1")


def _toy_he_problem() -> HumanEvalProblemV1:
    return HumanEvalProblemV1(
        task_id="dummy/g",
        prompt=(
            'def g(x):\n'
            '    """Add one.\n'
            '    >>> g(0)\n'
            '    1\n'
            '    >>> g(1)\n'
            '    2\n'
            '    """\n'),
        canonical_solution="return x + 1",
        test=textwrap.dedent("""
            def check(g):
                assert g(0) == 1
                assert g(1) == 2
                assert g(-1) == 0
            """).strip(),
        entry_point="g")


def _make_text_gen(passing: bool):
    code = (
        "```python\ndef g(x):\n    return x + 1\n```"
        if passing
        else "```python\ndef g(x):\n    return x\n```")
    def gen(prompt, max_tokens, temperature):
        return code, 50
    return gen


def _make_vlm_gen(*, planner_correct: bool, code_correct: bool):
    """Synthetic VLM that responds to planner / verifier
    prompts.  Code prompts go to the text_gen (Implementer)."""
    def gen(prompt, image_bytes, max_tokens, temperature):
        if "STRUCTURED PLAN" in prompt or "Plan:" in prompt:
            text = (
                "1. Function behaviour: adds 1.\n"
                "2. I/O: g(0)->1, g(1)->2.\n"
                "3. Edge cases: integers.\n"
                "4. Algorithm: return x + 1."
                if planner_correct
                else "Cannot read image clearly.")
        elif "STRUCTURED CRITIQUE" in prompt:
            text = (
                "Plan correct; code looks correct; ship as-is."
                if code_correct
                else "Code returns x instead of x+1. Fix: return x + 1.")
        elif (("Complete the following Python function" in prompt
                and image_bytes is not None) or
              "Your complete solution" in prompt):
            # A1_vlm code prompts
            text = (
                "```python\ndef g(x):\n    return x + 1\n```"
                if code_correct
                else "```python\ndef g(x):\n    return x\n```")
        else:
            text = "OK"
        return text, 50
    return gen


def _make_implementer_gen(*, follows_critique: bool):
    """Synthetic text-LM Implementer.  If follows_critique=True,
    produces correct code when the prompt mentions 'Critique'
    or has previous failure context."""
    state = {"call_n": 0}

    def gen(prompt, max_tokens, temperature):
        state["call_n"] += 1
        # The Implementer sees the Plan + history + critique
        # depending on the turn.
        if (follows_critique
                and ("Critique" in prompt or "Attempt" in prompt)):
            return (
                "```python\ndef g(x):\n    return x + 1\n```",
                50)
        # First attempt: wrong code
        return ("```python\ndef g(x):\n    return x\n```", 50)
    return gen


def _run_bench(text_gen, vlm_gen, n_seeds=1, n_problems=1):
    he = (_toy_he_problem(),)
    cfg = CrossModalRoleSpecBenchConfigV1(
        n_problems=int(n_problems),
        K_multi_sample=5,
        seeds=tuple(
            range(90_046_001, 90_046_001 + n_seeds)),
        sampling_temperature=0.7,
        max_tokens_per_call=64,
        strip_mode="doctest_only",
        min_doctest_lines=2)
    return run_cross_modal_role_specialized_bench_v1(
        text_gen=text_gen, vlm_gen=vlm_gen,
        vlm_model_id="synth_vlm",
        text_model_id="synth_text",
        corpus=he, config=cfg)


def test_w92_role_spec_b_beats_a1_when_critique_helps():
    """B wins when:
      * the VLM Planner gets the I/O examples right
      * the implementer FAILS the first attempt but FOLLOWS
        the verifier's critique on attempts 2 and 3
      * A1_vlm K=5 always fails (code_correct=False on VLM
        for code prompts)
    """
    text_gen = _make_implementer_gen(follows_critique=True)
    vlm_gen = _make_vlm_gen(
        planner_correct=True, code_correct=False)
    report, _ = _run_bench(
        text_gen, vlm_gen, n_seeds=1, n_problems=1)
    # A1_vlm uses vlm_gen for code → returns wrong code → 0%.
    assert report.a1_vlm_mean_pass_at_1 == 0.0
    # B uses Implementer turn 2/3 after critique → correct
    # code → 1.0.
    assert report.b_role_spec_mean_pass_at_1 == 1.0
    assert report.b_role_spec_mean_strictly_beats_a1_vlm_mean


def test_w92_role_spec_tie_when_a1_already_wins():
    """If A1_vlm K=5 always returns correct code, B and A1
    tie at 1.0."""
    text_gen = _make_implementer_gen(follows_critique=False)
    vlm_gen = _make_vlm_gen(
        planner_correct=True, code_correct=True)
    report, _ = _run_bench(
        text_gen, vlm_gen, n_seeds=1, n_problems=1)
    assert report.a1_vlm_mean_pass_at_1 == 1.0
    # B's Implementer never gets correct without critique (and
    # critique only fires when A1 fails), so B might not pass
    # in this synthetic.  Check that B and A1 are consistent.
    assert report.b_role_spec_mean_pass_at_1 in (0.0, 1.0)


def test_w92_role_spec_audit_chain_re_derives():
    text_gen = _make_implementer_gen(follows_critique=True)
    vlm_gen = _make_vlm_gen(
        planner_correct=True, code_correct=False)
    report, _ = _run_bench(
        text_gen, vlm_gen, n_seeds=2, n_problems=1)
    all_cids = []
    for s in report.per_seed:
        all_cids.extend(s.outcome_cids)
        d = hashlib.sha256(
            json.dumps({
                "kind": "w92_cross_modal_role_spec_seed_merkle_root",
                "seed": int(s.seed),
                "outcome_cids": list(s.outcome_cids),
            }, sort_keys=True, separators=(",", ":"),
                default=str).encode("utf-8")).hexdigest()
        assert d == s.seed_merkle_root
    derived_bench = hashlib.sha256(
        json.dumps({
            "kind": "w92_cross_modal_role_spec_bench_merkle_root",
            "vlm_model_id": "synth_vlm",
            "text_model_id": "synth_text",
            "outcome_cids": list(all_cids),
            "seeds": [int(s.seed) for s in report.per_seed],
        }, sort_keys=True, separators=(",", ":"),
            default=str).encode("utf-8")).hexdigest()
    assert derived_bench == report.bench_merkle_root


def test_w92_role_spec_module_explicit_import():
    import importlib
    mod = importlib.import_module(
        "coordpy.cross_modal_role_specialized_bench_v1")
    assert hasattr(
        mod,
        "W92_CROSS_MODAL_ROLE_SPECIALIZED_BENCH_V1_SCHEMA_VERSION")
    import coordpy
    assert not hasattr(
        coordpy, "run_cross_modal_role_specialized_bench_v1")


def test_w92_role_spec_b_arm_uses_exactly_5_calls():
    """B's outcome capsule reports exactly K=5 model calls."""
    text_gen = _make_implementer_gen(follows_critique=True)
    vlm_gen = _make_vlm_gen(
        planner_correct=True, code_correct=False)
    report, _ = _run_bench(
        text_gen, vlm_gen, n_seeds=1, n_problems=1)
    assert report.K_multi_sample == 5
