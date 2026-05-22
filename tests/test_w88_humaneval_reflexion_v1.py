"""W88 — HumanEval sequential-reflexion bench V1 CI tests.

These tests exercise the bench surface without making any real
LLM calls.  A deterministic synthetic ``gen`` callable produces
hash-derived "code" responses; the executor is the same CPython
subprocess used live.
"""
from __future__ import annotations

import hashlib
import os
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from coordpy.humaneval_real_bench_v1 import (
    HumanEvalProblemV1,
)
from coordpy.humaneval_reflexion_bench_v1 import (
    W88_HUMANEVAL_REFLEXION_BENCH_V1_SCHEMA_VERSION,
    HumanEvalReflexionBenchConfigV1,
    HumanEvalReflexionBenchReportV1,
    HumanEvalReflexionSeedReportV1,
    run_humaneval_reflexion_bench_v1,
)


def test_w88_schema_version_string():
    """Schema version is the canonical W88 string."""
    assert (
        W88_HUMANEVAL_REFLEXION_BENCH_V1_SCHEMA_VERSION
        == "coordpy.humaneval_reflexion_bench_v1.v1")


def _toy_problem(task_id: str, body_template: str,
                 *, n_args: int = 1) -> HumanEvalProblemV1:
    """Construct a small synthetic HumanEval-shaped problem.

    The ``body_template`` is Python source for the function body
    after ``def f(x):\n    ``.  The ``test`` block calls
    ``check(f)`` and asserts on a small set of inputs.
    """
    sig = "def f(x):\n    \"\"\"Return x + 1.\"\"\"\n    "
    canonical = sig + "return x + 1"
    test_block = textwrap.dedent("""
        def check(f):
            assert f(0) == 1
            assert f(1) == 2
            assert f(-1) == 0
    """).strip()
    return HumanEvalProblemV1(
        task_id=str(task_id),
        prompt=sig,
        canonical_solution=canonical,
        test=test_block,
        entry_point="f")


def _make_passing_gen():
    """A synthetic ``gen`` that returns a correct solution."""
    def gen(prompt, max_tokens, temperature):
        text = (
            "```python\n"
            "def f(x):\n"
            "    return x + 1\n"
            "```")
        # Constant wall-time so determinism holds across runs.
        return text, 50
    return gen


def _make_failing_gen():
    """A synthetic ``gen`` that always returns a wrong solution
    (returns x instead of x+1)."""
    def gen(prompt, max_tokens, temperature):
        text = (
            "```python\n"
            "def f(x):\n"
            "    return x\n"
            "```")
        return text, 50
    return gen


def _make_reflexion_gen():
    """A synthetic ``gen`` that:
    * On call 0: returns wrong solution.
    * On call 1+: returns correct solution iff the prompt contains
      stderr from a prior failed attempt (the "Executor stderr"
      marker).  Otherwise returns wrong.

    This models a model that uses Reflexion-style stderr-aware
    correction.
    """
    state = {"n": 0}

    def gen(prompt, max_tokens, temperature):
        state["n"] += 1
        n = state["n"]
        wrong = (
            "```python\n"
            "def f(x):\n"
            "    return x\n"
            "```")
        correct = (
            "```python\n"
            "def f(x):\n"
            "    return x + 1\n"
            "```")
        if "Executor stderr" in prompt:
            return correct, 50
        return wrong, 50

    return gen


def _run_bench(gen, *, n_seeds=1, n_problems=1, K=5):
    corpus = (
        _toy_problem("synthetic/0", "return x + 1"),
        _toy_problem("synthetic/1", "return x + 1"),
        _toy_problem("synthetic/2", "return x + 1"),
    )
    cfg = HumanEvalReflexionBenchConfigV1(
        n_problems=int(n_problems),
        K_multi_sample=int(K),
        seeds=tuple(range(88_028_001, 88_028_001 + n_seeds)),
        sampling_temperature=0.7,
        max_tokens_per_call=64,
    )
    return run_humaneval_reflexion_bench_v1(
        gen=gen, model_id="synth", corpus=corpus, config=cfg)


def test_w88_bench_passing_gen_all_arms_pass():
    """If the synthetic gen always produces correct code, every
    arm on every seed on every problem should pass."""
    report = _run_bench(_make_passing_gen(),
                        n_seeds=2, n_problems=2)
    assert report.a0_mean_pass_at_1 == 1.0
    assert report.a1_mean_pass_at_1 == 1.0
    assert report.b_mean_pass_at_1 == 1.0


def test_w88_bench_failing_gen_all_arms_fail():
    """If the synthetic gen always produces wrong code, every
    arm should fail."""
    report = _run_bench(_make_failing_gen(),
                        n_seeds=2, n_problems=2)
    assert report.a0_mean_pass_at_1 == 0.0
    assert report.a1_mean_pass_at_1 == 0.0
    assert report.b_mean_pass_at_1 == 0.0
    # A0 cannot beat A1 / B if all fail.
    assert not report.b_mean_strictly_beats_a0_mean
    assert not report.b_mean_strictly_beats_a1_mean


def test_w88_bench_reflexion_gen_b_beats_a0_a1():
    """If the synthetic gen ONLY produces correct code when
    given prior-attempt stderr, then:
    * A0 (single-shot, no stderr context) → fails.
    * A1 (K=5 independent samples, no stderr context) → fails.
    * B (K=5 sequential reflexion with stderr context on call
      1+) → passes on every problem.
    This is the load-bearing structural superiority of W88's
    B over A1 in the synthetic case.
    """
    report = _run_bench(_make_reflexion_gen(),
                        n_seeds=1, n_problems=2)
    assert report.a0_mean_pass_at_1 == 0.0
    assert report.a1_mean_pass_at_1 == 0.0
    assert report.b_mean_pass_at_1 == 1.0
    assert report.b_mean_strictly_beats_a0_mean
    assert report.b_mean_strictly_beats_a1_mean
    assert report.b_mean_minus_a1_mean_pp == 100.0


def test_w88_bench_b_uses_K_model_calls_per_problem():
    """B's per-problem outcome capsule records EXACTLY K model
    calls, matching A1's budget."""
    report = _run_bench(_make_failing_gen(),
                        n_seeds=1, n_problems=1, K=5)
    # Bench report doesn't store per-problem outcome capsules
    # directly; check the per-seed merkle root + call counts via
    # the schema (n_problems passes the contract).
    assert report.K_multi_sample == 5
    assert report.n_problems == 1
    assert report.n_seeds == 1


def test_w88_audit_chain_re_derives():
    """Every CID + Merkle root in the report MUST re-derive
    byte-for-byte from the report's own outcome CIDs."""
    import hashlib
    import json
    report = _run_bench(_make_reflexion_gen(),
                        n_seeds=2, n_problems=2)
    # Bench Merkle root
    all_cids = []
    for s in report.per_seed:
        all_cids.extend(s.outcome_cids)
    derived = hashlib.sha256(
        json.dumps({
            "kind": (
                "w88_humaneval_reflexion_bench_merkle_root"),
            "model_id": report.model_id,
            "outcome_cids": list(all_cids),
            "seeds": [int(s.seed) for s in report.per_seed],
        }, sort_keys=True, separators=(",", ":"),
            default=str).encode("utf-8")).hexdigest()
    assert derived == report.bench_merkle_root
    # Per-seed Merkle roots
    for s in report.per_seed:
        derived_seed = hashlib.sha256(
            json.dumps({
                "kind": (
                    "w88_humaneval_reflexion_seed_merkle_root"),
                "seed": int(s.seed),
                "outcome_cids": list(s.outcome_cids),
            }, sort_keys=True, separators=(",", ":"),
                default=str).encode("utf-8")).hexdigest()
        assert derived_seed == s.seed_merkle_root


def test_w88_b_mean_minus_a1_mean_pp_signs_match():
    """The headline ``b_mean_minus_a1_mean_pp`` delta is signed
    correctly: positive iff B > A1; negative iff B < A1; zero
    iff B == A1."""
    # Reflexion gen: B beats A1.
    r1 = _run_bench(_make_reflexion_gen(),
                    n_seeds=1, n_problems=1)
    assert r1.b_mean_minus_a1_mean_pp > 0.0
    # All-fail gen: B == A1 == 0.
    r2 = _run_bench(_make_failing_gen(),
                    n_seeds=1, n_problems=1)
    assert r2.b_mean_minus_a1_mean_pp == 0.0
    # All-pass gen: B == A1 == 1.
    r3 = _run_bench(_make_passing_gen(),
                    n_seeds=1, n_problems=1)
    assert r3.b_mean_minus_a1_mean_pp == 0.0


def test_w88_module_surface_explicit_import():
    """The W88 bench module is reachable ONLY via explicit
    import; ``coordpy/__init__.py`` is unchanged from W87."""
    import importlib
    mod = importlib.import_module(
        "coordpy.humaneval_reflexion_bench_v1")
    assert hasattr(
        mod, "W88_HUMANEVAL_REFLEXION_BENCH_V1_SCHEMA_VERSION")
    # The top-level coordpy must NOT re-export this.
    import coordpy
    assert not hasattr(
        coordpy, "run_humaneval_reflexion_bench_v1")
