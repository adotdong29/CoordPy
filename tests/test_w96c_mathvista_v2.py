"""W96-C — MathVista bench V2 (C1 VLM-Verifier-Final-Turn) CI tests.

NIM-free tests with mock gen functions.  Mirrors the
``tests/test_w95_mathvista_v1.py`` patterns; V2 reuses V1's
capsule + executor + loader unchanged, so we focus the V2 tests
on the new B arm's wiring + the verifier-rescue accounting.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from coordpy.mathvista_loader_v1 import MathVistaProblemV1
from coordpy.mathvista_bench_v2 import (
    MathVistaBenchConfigV2,
    MathVistaBenchReportV2,
    W96_MATHVISTA_BENCH_V2_SCHEMA_VERSION,
    _b_vlm_verifier_final_prompt_v2,
    _run_b_vlm_team_v2,
    run_mathvista_bench_v2,
)
from coordpy.mathvista_executor_v1 import evaluate_answer_v1


_FAKE_PNG = b"\x89PNG\r\n\x1a\n" + b"fake-png-payload-w96c"
_FAKE_PNG_SHA = hashlib.sha256(_FAKE_PNG).hexdigest()


def _make_num(pid: str, gold: str,
              *, precision: float = 0.0,
              answer_type: str = "integer",
              task: str = "geometry_problem_solving"
              ) -> MathVistaProblemV1:
    return MathVistaProblemV1(
        pid=str(pid),
        question="What is the value?",
        query="What is the value?",
        choices=(),
        answer=str(gold),
        answer_type=str(answer_type),
        question_type="free_form",
        unit="", precision=float(precision),
        image_bytes=_FAKE_PNG, image_sha256=_FAKE_PNG_SHA,
        image_format="png", metadata={"task": task})


def _make_mc(pid: str, gold: str,
             choices: tuple[str, ...],
             *, task: str = "figure_qa") -> MathVistaProblemV1:
    return MathVistaProblemV1(
        pid=str(pid),
        question="Pick one.",
        query="Pick one.",
        choices=tuple(choices),
        answer=str(gold),
        answer_type="text",
        question_type="multi_choice",
        unit="", precision=0.0,
        image_bytes=_FAKE_PNG, image_sha256=_FAKE_PNG_SHA,
        image_format="png", metadata={"task": task})


def test_w96c_bench_v2_schema_version():
    assert (
        W96_MATHVISTA_BENCH_V2_SCHEMA_VERSION
        == "coordpy.mathvista_bench_v2.v2")


def test_w96c_b_v2_requires_K_eq_5():
    p = _make_num("p1", "42")
    bad_K = 4
    with pytest.raises(ValueError):
        _run_b_vlm_team_v2(
            seed=1, p=p, K=bad_K, temperature=0.7,
            vlm_gen=lambda *a, **k: ("42", 1),
            text_gen=lambda *a, **k: ("42", 1),
            max_tokens=32)


def test_w96c_b_v2_byte_exact_K5_budget_when_text_only_passes():
    """When text-only solver passes immediately, we STILL run
    the verifier (budget accounting is byte-exact at K=5 on
    every problem).  This matches V1's padding discipline."""
    p = _make_num("p1", "42")

    n_text_calls = 0
    n_vlm_calls = 0

    def text_gen(prompt, max_tokens, temperature):
        nonlocal n_text_calls
        n_text_calls += 1
        return "42", 1

    def vlm_gen(prompt, image_bytes, max_tokens, temperature):
        nonlocal n_vlm_calls
        n_vlm_calls += 1
        return "42", 1

    out, exes = _run_b_vlm_team_v2(
        seed=1, p=p, K=5, temperature=0.7,
        vlm_gen=vlm_gen, text_gen=text_gen, max_tokens=32)
    # Budget: 1 vlm (reader) + 3 text (solver/reflexion×2) + 1
    # vlm (verifier) = 5 model calls byte-exact.
    assert out.n_model_calls == 5
    assert n_text_calls == 3
    assert n_vlm_calls == 2
    assert len(exes) == 4  # 3 solver + 1 verifier
    assert out.final_passed


def test_w96c_b_v2_verifier_rescues_when_text_only_all_fail():
    """When all text-only solvers fail and the verifier passes,
    the verifier's answer is shipped and verifier-rescue counts
    increment."""
    p = _make_num("p1", "42")

    def wrong_text(prompt, max_tokens, temperature):
        return "999", 1  # always wrong

    def perfect_vlm(prompt, image_bytes, max_tokens, temperature):
        return "42", 1  # always right

    out, exes = _run_b_vlm_team_v2(
        seed=1, p=p, K=5, temperature=0.7,
        vlm_gen=perfect_vlm, text_gen=wrong_text, max_tokens=32)
    assert out.final_passed
    # verifier was the load-bearing call
    assert exes[3].passed  # verifier passes
    assert not exes[0].passed  # solver_v1 fails
    assert not exes[1].passed  # solver_v2 fails
    assert not exes[2].passed  # solver_v3 fails


def test_w96c_b_v2_text_only_pass_short_circuits_verifier_selection():
    """When solver_v1 passes, ship that even if verifier later
    proposes a different answer.  This preserves W95-B0 wins."""
    p = _make_mc("p1", "dog", ("cat", "dog"))

    def text_gen(prompt, max_tokens, temperature):
        # solver_v1 returns the right answer
        return "dog", 1

    def vlm_gen(prompt, image_bytes, max_tokens, temperature):
        # verifier returns a WRONG answer (the verifier might
        # disagree, but text-only already passed → we ship
        # text-only)
        return "cat", 1

    out, exes = _run_b_vlm_team_v2(
        seed=1, p=p, K=5, temperature=0.7,
        vlm_gen=vlm_gen, text_gen=text_gen, max_tokens=32)
    assert out.final_passed
    assert exes[0].passed  # solver_v1 passed
    assert not exes[3].passed  # verifier wrong (but ignored)


def test_w96c_b_v2_ships_verifier_when_all_fail_and_verifier_also_fails():
    """When text-only all fail AND verifier fails, we still ship
    the verifier's answer (image-grounded last guess).  V2's
    fallback differs from V1 (V1 ships the last text-only)."""
    p = _make_num("p1", "42")

    def wrong_text(prompt, max_tokens, temperature):
        return "100", 1  # wrong

    def wrong_vlm(prompt, image_bytes, max_tokens, temperature):
        return "200", 1  # also wrong, but DIFFERENT

    out, exes = _run_b_vlm_team_v2(
        seed=1, p=p, K=5, temperature=0.7,
        vlm_gen=wrong_vlm, text_gen=wrong_text, max_tokens=32)
    assert not out.final_passed
    assert out.n_model_calls == 5


def test_w96c_run_bench_v2_e2e_with_uniform_mocks():
    """End-to-end smoke: A0 / A1 / B all get the same gold-answer
    mock → all arms pass 100 %."""
    uniform = (
        _make_num("u_p1", "42"),
        _make_num("u_p2", "42"),
    )

    def gen_text(prompt, max_tokens, temperature):
        return "42", 1

    def gen_vlm(prompt, image_bytes, max_tokens, temperature):
        return "42", 1

    cfg = MathVistaBenchConfigV2(
        n_problems=2, K_multi_sample=5,
        seeds=(95_005_001,),
        sampling_temperature=0.7,
        max_tokens_per_call=32)
    report = run_mathvista_bench_v2(
        text_gen=gen_text, vlm_gen=gen_vlm,
        vlm_model_id="mock-vlm",
        text_model_id="mock-text",
        corpus=uniform,
        corpus_parquet_sha256="ff" * 32,
        corpus_merkle_root="aa" * 32,
        config=cfg)
    assert isinstance(report, MathVistaBenchReportV2)
    assert report.a0_text_mean_pass_at_1 == 1.0
    assert report.a1_vlm_mean_pass_at_1 == 1.0
    assert report.b_vlm_team_v2_mean_pass_at_1 == 1.0
    assert report.K_multi_sample == 5
    assert len(report.per_seed) == 1
    ps = report.per_seed[0]
    assert ps.n_problems == 2
    assert len(ps.outcome_cids) == 3 * 2  # 3 arms × 2 problems
    assert len(ps.seed_merkle_root) == 64
    assert len(report.bench_merkle_root) == 64
    # All text-only solvers pass → no verifier rescues.
    assert report.n_verifier_rescues_per_seed == (0,)
    assert report.n_text_only_passes_per_seed == (2,)


def test_w96c_run_bench_v2_records_verifier_rescues():
    """Mock setup where text-only ALWAYS fails but the VLM
    (acting as both A1 sampler and B's verifier) ALWAYS gets
    the right answer → the verifier rescues every B problem
    while A0 fails every problem.  Problems use a UNIFORM gold
    answer so the mock VLM doesn't need pid-routing (the per-arm
    prompts in mathvista_bench_v1 don't embed the pid)."""
    uniform = (
        _make_num("vr_p1", "42"),
        _make_num("vr_p2", "42"),  # shared gold; mock-friendly
    )

    def wrong_text(prompt, max_tokens, temperature):
        return "wrong", 1

    def gold_vlm(prompt, image_bytes, max_tokens, temperature):
        return "42", 1

    cfg = MathVistaBenchConfigV2(
        n_problems=2, K_multi_sample=5,
        seeds=(95_005_001,),
        sampling_temperature=0.7,
        max_tokens_per_call=32)
    report = run_mathvista_bench_v2(
        text_gen=wrong_text, vlm_gen=gold_vlm,
        vlm_model_id="mock-vlm",
        text_model_id="mock-text",
        corpus=uniform,
        corpus_parquet_sha256="ff" * 32,
        corpus_merkle_root="aa" * 32,
        config=cfg)
    # A0 (text-only) fails everywhere.
    assert report.a0_text_mean_pass_at_1 == 0.0
    # A1 (VLM K=5) passes everywhere.
    assert report.a1_vlm_mean_pass_at_1 == 1.0
    # B v2 passes everywhere VIA the verifier (text-only solver
    # gets wrong_text; verifier is VLM gold_vlm = "42").
    assert report.b_vlm_team_v2_mean_pass_at_1 == 1.0
    # Every B problem was a verifier-rescue.
    assert report.n_verifier_rescues_per_seed == (2,)
    assert report.n_text_only_passes_per_seed == (0,)


def test_w96c_run_bench_v2_no_regression_when_text_only_wins():
    """Setup where the text-only solver always gets it right;
    the verifier should never be the load-bearing call."""
    uniform = (_make_num("tw_p1", "42"),)

    def gold_text(prompt, max_tokens, temperature):
        return "42", 1

    def wrong_vlm(prompt, image_bytes, max_tokens, temperature):
        # VLM doesn't help — but text-only PASS short-circuits,
        # so this never matters for B's final_passed.
        return "999", 1

    cfg = MathVistaBenchConfigV2(
        n_problems=1, K_multi_sample=5,
        seeds=(95_005_001,),
        sampling_temperature=0.7,
        max_tokens_per_call=32)
    report = run_mathvista_bench_v2(
        text_gen=gold_text, vlm_gen=wrong_vlm,
        vlm_model_id="mock-vlm",
        text_model_id="mock-text",
        corpus=uniform,
        corpus_parquet_sha256="ff" * 32,
        corpus_merkle_root="aa" * 32,
        config=cfg)
    assert report.a0_text_mean_pass_at_1 == 1.0
    assert report.a1_vlm_mean_pass_at_1 == 0.0  # VLM always wrong
    assert report.b_vlm_team_v2_mean_pass_at_1 == 1.0  # text-only saved B
    assert report.n_verifier_rescues_per_seed == (0,)
    assert report.n_text_only_passes_per_seed == (1,)


def test_w96c_verifier_prompt_contains_image_emphasis_and_history():
    """The verifier prompt must (a) emphasize that the verifier
    sees the image while the text-only solver did not, and (b)
    include each prior candidate + executor verdict."""
    p = _make_num("p1", "42")
    extraction = "Bullet 1: the chart shows 42 mph"
    candidates = ["100", "55", "42"]
    exes = [
        evaluate_answer_v1(prediction=c, problem=p)
        for c in candidates]
    prompt = _b_vlm_verifier_final_prompt_v2(
        p, extraction, candidates, exes)
    # Image emphasis.
    assert "image" in prompt.lower()
    assert "could not" in prompt.lower()  # "could not see image"
    # Extraction is included.
    assert extraction in prompt
    # Every prior candidate is included.
    for c in candidates:
        assert c in prompt
    # Final-answer cue is present (matches extract_candidate_answer_v1 last-line heuristic).
    assert "Final answer:" in prompt


def test_w96c_v2_per_problem_outcome_records_v2_fields():
    """Per-problem outcome rows must include the v2-specific
    fields: b_vlm_team_v2_passed, b_text_only_passed,
    b_verifier_rescued."""
    uniform = (_make_num("v2_p1", "42"),)

    def wrong_text(prompt, max_tokens, temperature):
        return "0", 1

    def gold_vlm(prompt, image_bytes, max_tokens, temperature):
        return "42", 1

    cfg = MathVistaBenchConfigV2(
        n_problems=1, K_multi_sample=5,
        seeds=(95_005_001,),
        sampling_temperature=0.7,
        max_tokens_per_call=32)
    report = run_mathvista_bench_v2(
        text_gen=wrong_text, vlm_gen=gold_vlm,
        vlm_model_id="mock-vlm",
        text_model_id="mock-text",
        corpus=uniform,
        corpus_parquet_sha256="ff" * 32,
        corpus_merkle_root="aa" * 32,
        config=cfg)
    po = report.per_seed[0].per_problem_outcomes[0]
    assert po["pid"] == "v2_p1"
    # v2-specific fields
    assert "b_vlm_team_v2_passed" in po
    assert "b_text_only_passed" in po
    assert "b_verifier_rescued" in po
    assert po["b_vlm_team_v2_passed"] is True
    assert po["b_text_only_passed"] is False
    assert po["b_verifier_rescued"] is True
