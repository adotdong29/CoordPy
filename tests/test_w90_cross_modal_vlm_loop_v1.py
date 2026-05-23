"""W90 — Cross-modal VLM-in-loop bench V1 CI tests."""
from __future__ import annotations

import hashlib
import json
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from coordpy.cross_modal_vlm_loop_bench_v1 import (
    W90_CROSS_MODAL_VLM_LOOP_BENCH_V1_SCHEMA_VERSION,
    CrossModalVlmLoopBenchConfigV1,
    run_cross_modal_vlm_loop_bench_v1,
)
from coordpy.humaneval_real_bench_v1 import (
    HumanEvalProblemV1,
)


def test_w90_xm_loop_schema_version():
    assert (
        W90_CROSS_MODAL_VLM_LOOP_BENCH_V1_SCHEMA_VERSION
        == "coordpy.cross_modal_vlm_loop_bench_v1.v1")


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


def _make_vlm_gen(*, code_correct: bool,
                  reflexion_correct: bool = False):
    """Synthetic VLM that responds to code-generation prompts.
    If reflexion_correct=True, returns correct code ONLY when
    prior-attempt stderr is in the prompt; otherwise wrong.
    """
    def gen(prompt, image_bytes, max_tokens, temperature):
        if reflexion_correct and "Executor stderr" in prompt:
            return (
                "```python\ndef g(x):\n    return x + 1\n```",
                50)
        if code_correct:
            return (
                "```python\ndef g(x):\n    return x + 1\n```",
                50)
        return (
            "```python\ndef g(x):\n    return x\n```", 50)
    return gen


def test_w90_xm_loop_b_beats_a1_when_only_reflexion_works():
    """B (VLM-in-loop) wins when only stderr-aware turns
    produce correct code; A1 (K=5 independent) all fail."""
    he = (_toy_he_problem(),)
    cfg = CrossModalVlmLoopBenchConfigV1(
        n_problems=1, K_multi_sample=5,
        seeds=(90_046_001,),
        sampling_temperature=0.7, max_tokens_per_call=64,
        strip_mode="doctest_only", min_doctest_lines=2)
    text_gen = _make_text_gen(passing=False)  # A0_text fails
    vlm_gen = _make_vlm_gen(
        code_correct=False, reflexion_correct=True)
    report, _ = run_cross_modal_vlm_loop_bench_v1(
        text_gen=text_gen, vlm_gen=vlm_gen,
        vlm_model_id="synth_vlm",
        text_model_id="synth_text",
        corpus=he, config=cfg)
    assert report.a0_text_mean_pass_at_1 == 0.0
    assert report.a1_vlm_mean_pass_at_1 == 0.0
    assert report.b_vlm_loop_mean_pass_at_1 == 1.0
    assert report.b_vlm_loop_mean_strictly_beats_a0_text_mean
    assert report.b_vlm_loop_mean_strictly_beats_a1_vlm_mean


def test_w90_xm_loop_tie_when_a1_also_wins():
    """If A1 K=5 already finds correct code, B and A1 tie."""
    he = (_toy_he_problem(),)
    cfg = CrossModalVlmLoopBenchConfigV1(
        n_problems=1, K_multi_sample=5,
        seeds=(90_046_001,),
        sampling_temperature=0.7, max_tokens_per_call=64,
        strip_mode="doctest_only", min_doctest_lines=2)
    text_gen = _make_text_gen(passing=False)
    vlm_gen = _make_vlm_gen(code_correct=True)
    report, _ = run_cross_modal_vlm_loop_bench_v1(
        text_gen=text_gen, vlm_gen=vlm_gen,
        vlm_model_id="synth_vlm",
        text_model_id="synth_text",
        corpus=he, config=cfg)
    assert report.a1_vlm_mean_pass_at_1 == 1.0
    assert report.b_vlm_loop_mean_pass_at_1 == 1.0
    # Tie: strict-improvement is False.
    assert not report.b_vlm_loop_mean_strictly_beats_a1_vlm_mean


def test_w90_xm_loop_audit_chain_re_derives():
    he = (_toy_he_problem(),)
    cfg = CrossModalVlmLoopBenchConfigV1(
        n_problems=1, K_multi_sample=5,
        seeds=(90_046_001, 90_046_002),
        sampling_temperature=0.7, max_tokens_per_call=64,
        strip_mode="doctest_only", min_doctest_lines=2)
    text_gen = _make_text_gen(passing=False)
    vlm_gen = _make_vlm_gen(
        code_correct=False, reflexion_correct=True)
    report, _ = run_cross_modal_vlm_loop_bench_v1(
        text_gen=text_gen, vlm_gen=vlm_gen,
        vlm_model_id="synth_vlm",
        text_model_id="synth_text",
        corpus=he, config=cfg)
    all_cids = []
    for s in report.per_seed:
        all_cids.extend(s.outcome_cids)
        d = hashlib.sha256(
            json.dumps({
                "kind": (
                    "w90_cross_modal_vlm_loop_seed_merkle_root"),
                "seed": int(s.seed),
                "outcome_cids": list(s.outcome_cids),
            }, sort_keys=True, separators=(",", ":"),
                default=str).encode("utf-8")).hexdigest()
        assert d == s.seed_merkle_root
    db = hashlib.sha256(
        json.dumps({
            "kind": (
                "w90_cross_modal_vlm_loop_bench_merkle_root"),
            "vlm_model_id": "synth_vlm",
            "text_model_id": "synth_text",
            "outcome_cids": list(all_cids),
            "seeds": [int(s.seed) for s in report.per_seed],
        }, sort_keys=True, separators=(",", ":"),
            default=str).encode("utf-8")).hexdigest()
    assert db == report.bench_merkle_root


def test_w90_xm_loop_module_explicit_import():
    import importlib
    mod = importlib.import_module(
        "coordpy.cross_modal_vlm_loop_bench_v1")
    assert hasattr(
        mod, "W90_CROSS_MODAL_VLM_LOOP_BENCH_V1_SCHEMA_VERSION")
    import coordpy
    assert not hasattr(
        coordpy, "run_cross_modal_vlm_loop_bench_v1")
