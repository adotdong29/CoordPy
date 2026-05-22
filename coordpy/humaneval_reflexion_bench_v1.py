"""W88 / post-W87 — HumanEval sequential-reflexion bench V1.

Targets the canonical W86 carry-forward
``W86-L-HUMANEVAL-V1-A1-SAME-BUDGET-NOT-BEATEN`` head-on.  The W86
B-pipeline (solver_1, solver_2, critic, reviser, judge) lost the
same-budget head-to-head against A1 (first-pass-among-K=5) by
8.9 pp on the mean across 3 seeds.  The post-mortem in
``docs/RESULTS_W86_HUMANEVAL_HEAD_TO_HEAD.md`` identified the
likely cause: only 3 of W86's 5 calls were code-producing
(solver_1, solver_2, reviser); the critic and judge added
overhead but not new candidates.

W88's B-pipeline reallocates the same K=5 budget differently:

  * Every one of the 5 model calls IS a code-producing call.
  * Each call is conditioned on the cumulative history of
    every prior candidate AND the executor's stderr on each.
  * Reading the actual subprocess traceback (last 500 chars) is
    the load-bearing differentiator vs A1's independent samples.
  * Ship the first PASS; if no PASS, ship the candidate with the
    lowest content-addressed CID lexicographically (a
    deterministic, content-addressed tie-breaker).

This is structurally a "Reflexion-K=5" / "Self-Debug-K=5" shape
where each turn is a reflective code-producing turn rather than a
pure-text reflection.  The literature reports same-budget wins
for this shape over independent sampling on HumanEval at multiple
model scales, but it had not previously been verified at scale on
Llama-3.1-8B-Instruct with the W86 / CoordPy audit-chain
discipline.

Three arms (mirrors the W86 V1 anti-cheat surface):

* ``A0`` — stock single-shot at T=0.0.  Same as W86.
* ``A1`` — first-pass-among-K=5 self-consistency at T=0.7.  Same
  as W86.  This is the empirical baseline that W86's B failed
  to beat; W88's B is built to beat it.
* ``B`` — ``executor_guided_sequential_reflexion_v1``.  5
  sequential model calls at T=0.7, each conditioned on the
  cumulative (candidate_k, executor_stderr_k) history.

Anti-cheat (mirrors the W86 surface verbatim where applicable):

* Same model on every arm.
* Same task subset per seed across arms (the W86
  ``select_humaneval_subset_v1`` discipline is preserved).
* Same prompt budget (A1 = K=5; B = K=5).
* Same retry policy on transient provider errors.
* No selective retries; each (seed, problem, arm) triple is one
  set of calls.
* Executor truth = full ``problem.test`` block — same oracle
  for every arm.
* Audit chain re-derives offline from the persisted JSONL
  sidecar.

Honest scope (W88)
------------------

* ``W88-L-HUMANEVAL-REFLEXION-V1-NIM-DEPENDENT-CAP`` — V1 drives
  the bench through any ``LLMBackend``-shaped client; provider
  determinism is not assumed beyond temperature=0.
* ``W88-L-HUMANEVAL-REFLEXION-V1-SUBPROCESS-PYTHON-EXECUTOR-CAP``
  — inherited from W86; identical executor.
* ``W88-L-HUMANEVAL-REFLEXION-V1-NETWORK-FETCH-CAP`` — inherited
  from W86.
* ``W88-L-HUMANEVAL-REFLEXION-V1-PROMPT-LEN-CAP`` — V1's later
  reflexion prompts include the cumulative history of prior
  attempts; long traces are truncated by the executor's
  500-char stderr tail policy, but the prompt can still grow
  to 3-4 KB on the K=5 attempt.  Within Llama-3.1-8B's 128k
  context this is comfortable.
* The W86 ``W86-L-HUMANEVAL-V1-A1-SAME-BUDGET-NOT-BEATEN``
  carry-forward is the EXPLICIT TARGET of this milestone.  If
  the W88 bench produces ``b_mean_strictly_beats_a1_mean = True``
  at the strong-success bar pre-committed in
  ``docs/RUNBOOK_W88.md``, the carry-forward retires; otherwise
  it stays.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import random
from typing import Any, Callable, Sequence

from .humaneval_real_bench_v1 import (
    W86_HUMANEVAL_EXECUTOR_KILL_AFTER_S,
    W86_HUMANEVAL_EXECUTOR_TIMEOUT_S,
    HumanEvalArmCallCapsuleV1,
    HumanEvalArmOutcomeCapsuleV1,
    HumanEvalExecutorResultV1,
    HumanEvalProblemV1,
    extract_candidate_code_v1,
    run_humaneval_executor_v1,
    select_humaneval_subset_v1,
)


W88_HUMANEVAL_REFLEXION_BENCH_V1_SCHEMA_VERSION: str = (
    "coordpy.humaneval_reflexion_bench_v1.v1")


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


# ---------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------

_SYSTEM_REFLEXION = (
    "You are an expert Python programmer.  When given a function "
    "signature and docstring, output ONLY the complete function "
    "(inside a ```python ... ``` code fence).  Do not include any "
    "prose before or after the code fence.")


def _initial_prompt(problem: HumanEvalProblemV1) -> str:
    return (
        f"{_SYSTEM_REFLEXION}\n\n"
        "Complete the following Python function.  Provide the "
        "full function including the signature.\n\n"
        f"```python\n{problem.prompt}```\n\n"
        "Your complete solution:")


def _reflexion_prompt(
        problem: HumanEvalProblemV1,
        history: Sequence[tuple[str, HumanEvalExecutorResultV1]],
        attempt_idx: int,
) -> str:
    """Reflexion turn k: cumulative history of all prior
    (candidate, executor_result) pairs is embedded.  The model
    is asked to identify the bug class in each failed attempt
    and produce a corrected solution.
    """
    chunks: list[str] = []
    for i, (cand, exe) in enumerate(history):
        # Truncate long candidates to keep prompt size manageable.
        cand_trimmed = cand
        if len(cand_trimmed) > 1500:
            cand_trimmed = (
                cand_trimmed[:1500]
                + "\n# ... (truncated)\n")
        if exe.passed:
            verdict = "PASSED visible tests"
            stderr_excerpt = ""
        else:
            verdict = (
                f"FAILED (returncode={exe.returncode}"
                + (", TIMED OUT" if exe.timed_out else "")
                + ")")
            stderr_text = exe.stderr_tail.strip()
            if stderr_text:
                stderr_excerpt = (
                    f"\nExecutor stderr (tail):\n{stderr_text}")
            else:
                stderr_excerpt = ""
        chunks.append(
            f"--- Attempt {i+1} ({verdict}) ---\n"
            f"```python\n{cand_trimmed}\n```{stderr_excerpt}")
    return (
        f"{_SYSTEM_REFLEXION}\n\n"
        "[Role: reflective code generator]\n"
        "You are on attempt "
        f"{attempt_idx + 1} out of 5.  The following are your "
        "previous attempts and the Python executor's verdict + "
        "stderr tail for each.  Diagnose the bug class in each "
        "failing attempt and produce a NEW corrected complete "
        "Python function.  Do not repeat a previous attempt "
        "verbatim.\n\n"
        f"Target function:\n```python\n{problem.prompt}```\n\n"
        f"{chr(10).join(chunks)}\n\n"
        "Provide ONLY the corrected complete Python function in "
        "a ```python ... ``` fence:")


# ---------------------------------------------------------------
# Per-arm runners
# ---------------------------------------------------------------

_GenerateFn = Callable[[str, int, float], tuple[str, int]]


def _run_a0_single_shot(
        *, seed: int, problem: HumanEvalProblemV1,
        gen: _GenerateFn, max_tokens: int,
        executor_kwargs: dict[str, Any],
) -> tuple[HumanEvalArmOutcomeCapsuleV1, HumanEvalExecutorResultV1]:
    prompt = _initial_prompt(problem)
    text, wall = gen(prompt, max_tokens, 0.0)
    code = extract_candidate_code_v1(
        response_text=text, prompt=problem.prompt,
        entry_point=problem.entry_point)
    exe = run_humaneval_executor_v1(
        problem=problem, candidate_code=code,
        **executor_kwargs)
    call = HumanEvalArmCallCapsuleV1(
        schema=W88_HUMANEVAL_REFLEXION_BENCH_V1_SCHEMA_VERSION,
        seed=int(seed),
        task_id=str(problem.task_id),
        arm_id="A0",
        role="solver",
        call_idx=0,
        temperature=0.0,
        prompt_cid=hashlib.sha256(
            prompt.encode("utf-8")).hexdigest(),
        response_cid=hashlib.sha256(
            text.encode("utf-8")).hexdigest(),
        wall_ms=int(wall))
    out = HumanEvalArmOutcomeCapsuleV1(
        schema=W88_HUMANEVAL_REFLEXION_BENCH_V1_SCHEMA_VERSION,
        seed=int(seed),
        task_id=str(problem.task_id),
        arm_id="A0",
        final_passed=bool(exe.passed),
        final_candidate_code_cid=str(exe.candidate_code_cid),
        n_model_calls=1,
        n_executor_calls=1,
        total_wall_ms=int(wall + exe.wall_ms),
        call_capsule_cids=(call.cid(),),
        executor_result_cids=(exe.cid(),))
    return out, exe


def _run_a1_first_pass_among_K(
        *, seed: int, problem: HumanEvalProblemV1,
        K: int, temperature: float,
        gen: _GenerateFn, max_tokens: int,
        executor_kwargs: dict[str, Any],
) -> tuple[HumanEvalArmOutcomeCapsuleV1,
           list[HumanEvalExecutorResultV1]]:
    """A1 baseline (verbatim same shape as W86's A1)."""
    prompt = _initial_prompt(problem)
    calls: list[HumanEvalArmCallCapsuleV1] = []
    exes: list[HumanEvalExecutorResultV1] = []
    total = 0
    chosen_passed = False
    chosen_code_cid = ""
    for k in range(int(K)):
        text, wall = gen(prompt, max_tokens, float(temperature))
        code = extract_candidate_code_v1(
            response_text=text, prompt=problem.prompt,
            entry_point=problem.entry_point)
        exe = run_humaneval_executor_v1(
            problem=problem, candidate_code=code,
            **executor_kwargs)
        calls.append(HumanEvalArmCallCapsuleV1(
            schema=(
                W88_HUMANEVAL_REFLEXION_BENCH_V1_SCHEMA_VERSION),
            seed=int(seed),
            task_id=str(problem.task_id),
            arm_id="A1",
            role="sample",
            call_idx=int(k),
            temperature=float(temperature),
            prompt_cid=hashlib.sha256(
                prompt.encode("utf-8")).hexdigest(),
            response_cid=hashlib.sha256(
                text.encode("utf-8")).hexdigest(),
            wall_ms=int(wall)))
        exes.append(exe)
        total += int(wall) + int(exe.wall_ms)
        if exe.passed and not chosen_passed:
            chosen_passed = True
            chosen_code_cid = str(exe.candidate_code_cid)
    if not chosen_passed:
        chosen_code_cid = str(exes[0].candidate_code_cid)
    out = HumanEvalArmOutcomeCapsuleV1(
        schema=W88_HUMANEVAL_REFLEXION_BENCH_V1_SCHEMA_VERSION,
        seed=int(seed),
        task_id=str(problem.task_id),
        arm_id="A1",
        final_passed=bool(chosen_passed),
        final_candidate_code_cid=str(chosen_code_cid),
        n_model_calls=int(K),
        n_executor_calls=int(K),
        total_wall_ms=int(total),
        call_capsule_cids=tuple(c.cid() for c in calls),
        executor_result_cids=tuple(e.cid() for e in exes))
    return out, exes


def _run_b_sequential_reflexion(
        *, seed: int, problem: HumanEvalProblemV1,
        K: int, temperature: float,
        gen: _GenerateFn, max_tokens: int,
        executor_kwargs: dict[str, Any],
) -> tuple[HumanEvalArmOutcomeCapsuleV1,
           list[HumanEvalExecutorResultV1]]:
    """B: K sequential model calls, each conditioned on the
    cumulative history of prior (candidate, stderr) pairs.

    Same total model-call budget as A1 (K=5).  No early-stop on
    PASS — every call runs to completion, so the budget is
    EXACTLY K per problem (matching A1 spend exactly).  Ship the
    first PASS; if no PASS, ship the candidate with the
    lexicographically smallest CID (deterministic content-
    addressed fallback).
    """
    history: list[tuple[str, HumanEvalExecutorResultV1]] = []
    calls: list[HumanEvalArmCallCapsuleV1] = []
    exes: list[HumanEvalExecutorResultV1] = []
    candidates_code: list[str] = []
    total = 0
    for k in range(int(K)):
        if k == 0:
            prompt = _initial_prompt(problem)
        else:
            prompt = _reflexion_prompt(
                problem, tuple(history), attempt_idx=int(k))
        text, wall = gen(prompt, max_tokens, float(temperature))
        code = extract_candidate_code_v1(
            response_text=text, prompt=problem.prompt,
            entry_point=problem.entry_point)
        exe = run_humaneval_executor_v1(
            problem=problem, candidate_code=code,
            **executor_kwargs)
        calls.append(HumanEvalArmCallCapsuleV1(
            schema=(
                W88_HUMANEVAL_REFLEXION_BENCH_V1_SCHEMA_VERSION),
            seed=int(seed),
            task_id=str(problem.task_id),
            arm_id="B",
            role="reflexion" if k > 0 else "initial",
            call_idx=int(k),
            temperature=float(temperature),
            prompt_cid=hashlib.sha256(
                prompt.encode("utf-8")).hexdigest(),
            response_cid=hashlib.sha256(
                text.encode("utf-8")).hexdigest(),
            wall_ms=int(wall)))
        exes.append(exe)
        candidates_code.append(code)
        history.append((code, exe))
        total += int(wall) + int(exe.wall_ms)
    # Selection: first PASS by attempt index; else
    # lexicographically smallest CID among attempts (deterministic
    # content-addressed fallback so the outcome is not
    # provider-dependent).
    final_passed = False
    final_code_cid = ""
    for e in exes:
        if e.passed:
            final_passed = True
            final_code_cid = str(e.candidate_code_cid)
            break
    if not final_passed:
        cids = sorted(str(e.candidate_code_cid) for e in exes)
        final_code_cid = cids[0]
    out = HumanEvalArmOutcomeCapsuleV1(
        schema=W88_HUMANEVAL_REFLEXION_BENCH_V1_SCHEMA_VERSION,
        seed=int(seed),
        task_id=str(problem.task_id),
        arm_id="B",
        final_passed=bool(final_passed),
        final_candidate_code_cid=str(final_code_cid),
        n_model_calls=int(K),
        n_executor_calls=int(K),
        total_wall_ms=int(total),
        call_capsule_cids=tuple(c.cid() for c in calls),
        executor_result_cids=tuple(e.cid() for e in exes))
    return out, exes


# ---------------------------------------------------------------
# Report
# ---------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class HumanEvalReflexionSeedReportV1:
    schema: str
    seed: int
    n_problems: int
    a0_pass_at_1: float
    a1_pass_at_1: float
    b_pass_at_1: float
    a0_total_wall_ms: int
    a1_total_wall_ms: int
    b_total_wall_ms: int
    outcome_cids: tuple[str, ...]
    seed_merkle_root: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "seed": int(self.seed),
            "n_problems": int(self.n_problems),
            "a0_pass_at_1": float(round(
                self.a0_pass_at_1, 6)),
            "a1_pass_at_1": float(round(
                self.a1_pass_at_1, 6)),
            "b_pass_at_1": float(round(
                self.b_pass_at_1, 6)),
            "a0_total_wall_ms": int(self.a0_total_wall_ms),
            "a1_total_wall_ms": int(self.a1_total_wall_ms),
            "b_total_wall_ms": int(self.b_total_wall_ms),
            "outcome_cids": list(self.outcome_cids),
            "seed_merkle_root": str(self.seed_merkle_root),
        }


@dataclasses.dataclass(frozen=True)
class HumanEvalReflexionBenchReportV1:
    schema: str
    model_id: str
    n_problems: int
    n_seeds: int
    K_multi_sample: int
    per_seed: tuple[HumanEvalReflexionSeedReportV1, ...]
    a0_mean_pass_at_1: float
    a1_mean_pass_at_1: float
    b_mean_pass_at_1: float
    b_beats_a0_per_seed: tuple[bool, ...]
    b_beats_a1_per_seed: tuple[bool, ...]
    b_strictly_beats_a0_on_all_seeds: bool
    b_strictly_beats_a1_on_all_seeds: bool
    b_mean_strictly_beats_a0_mean: bool
    b_mean_strictly_beats_a1_mean: bool
    b_mean_minus_a1_mean_pp: float
    bench_merkle_root: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "model_id": str(self.model_id),
            "n_problems": int(self.n_problems),
            "n_seeds": int(self.n_seeds),
            "K_multi_sample": int(self.K_multi_sample),
            "per_seed": [s.to_dict() for s in self.per_seed],
            "a0_mean_pass_at_1": float(round(
                self.a0_mean_pass_at_1, 6)),
            "a1_mean_pass_at_1": float(round(
                self.a1_mean_pass_at_1, 6)),
            "b_mean_pass_at_1": float(round(
                self.b_mean_pass_at_1, 6)),
            "b_beats_a0_per_seed": list(
                self.b_beats_a0_per_seed),
            "b_beats_a1_per_seed": list(
                self.b_beats_a1_per_seed),
            "b_strictly_beats_a0_on_all_seeds": bool(
                self.b_strictly_beats_a0_on_all_seeds),
            "b_strictly_beats_a1_on_all_seeds": bool(
                self.b_strictly_beats_a1_on_all_seeds),
            "b_mean_strictly_beats_a0_mean": bool(
                self.b_mean_strictly_beats_a0_mean),
            "b_mean_strictly_beats_a1_mean": bool(
                self.b_mean_strictly_beats_a1_mean),
            "b_mean_minus_a1_mean_pp": float(round(
                self.b_mean_minus_a1_mean_pp, 4)),
            "bench_merkle_root": str(self.bench_merkle_root),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w88_humaneval_reflexion_bench_report_v1",
            "report": self.to_dict(),
        })


# ---------------------------------------------------------------
# Config + driver
# ---------------------------------------------------------------

@dataclasses.dataclass
class HumanEvalReflexionBenchConfigV1:
    schema: str = W88_HUMANEVAL_REFLEXION_BENCH_V1_SCHEMA_VERSION
    n_problems: int = 30
    K_multi_sample: int = 5
    seeds: tuple[int, ...] = (88_028_001, 88_028_002, 88_028_003)
    sampling_temperature: float = 0.7
    max_tokens_per_call: int = 768
    executor_timeout_s: float = (
        W86_HUMANEVAL_EXECUTOR_TIMEOUT_S)
    executor_kill_after_s: float = (
        W86_HUMANEVAL_EXECUTOR_KILL_AFTER_S)


def run_humaneval_reflexion_bench_v1(
        *,
        gen: _GenerateFn,
        model_id: str,
        corpus: Sequence[HumanEvalProblemV1],
        config: HumanEvalReflexionBenchConfigV1 | None = None,
        on_problem_start: (
            Callable[[int, int, str], None] | None) = None,
) -> HumanEvalReflexionBenchReportV1:
    """Run A0 + A1 + B (sequential reflexion) on a deterministic
    subset for each seed.  Returns a content-addressed bench
    report.  Mirrors ``run_humaneval_real_bench_v1`` from W86
    structurally; the only differences are:

    * the B arm calls ``_run_b_sequential_reflexion`` instead of
      W86's ``_run_b_executor_critic``;
    * the schema strings are W88 so the audit chains do not
      collide;
    * the report carries the explicit ``b_mean_minus_a1_mean_pp``
      delta for the headline retirement-bar comparison.
    """
    cfg = config or HumanEvalReflexionBenchConfigV1()
    executor_kwargs = {
        "timeout_s": float(cfg.executor_timeout_s),
        "kill_after_s": float(cfg.executor_kill_after_s),
    }
    per_seed: list[HumanEvalReflexionSeedReportV1] = []
    all_outcome_cids: list[str] = []
    for seed in cfg.seeds:
        subset = select_humaneval_subset_v1(
            corpus=corpus, n_problems=int(cfg.n_problems),
            seed=int(seed))
        a0_outs: list[HumanEvalArmOutcomeCapsuleV1] = []
        a1_outs: list[HumanEvalArmOutcomeCapsuleV1] = []
        b_outs: list[HumanEvalArmOutcomeCapsuleV1] = []
        for p_idx, problem in enumerate(subset):
            if on_problem_start is not None:
                on_problem_start(
                    int(seed), int(p_idx), str(problem.task_id))
            a0_out, _ = _run_a0_single_shot(
                seed=int(seed), problem=problem,
                gen=gen,
                max_tokens=int(cfg.max_tokens_per_call),
                executor_kwargs=executor_kwargs)
            a0_outs.append(a0_out)
            a1_out, _ = _run_a1_first_pass_among_K(
                seed=int(seed), problem=problem,
                K=int(cfg.K_multi_sample),
                temperature=float(cfg.sampling_temperature),
                gen=gen,
                max_tokens=int(cfg.max_tokens_per_call),
                executor_kwargs=executor_kwargs)
            a1_outs.append(a1_out)
            b_out, _ = _run_b_sequential_reflexion(
                seed=int(seed), problem=problem,
                K=int(cfg.K_multi_sample),
                temperature=float(cfg.sampling_temperature),
                gen=gen,
                max_tokens=int(cfg.max_tokens_per_call),
                executor_kwargs=executor_kwargs)
            b_outs.append(b_out)
        n = float(len(a0_outs))
        a0_acc = sum(
            1 for o in a0_outs if o.final_passed) / n
        a1_acc = sum(
            1 for o in a1_outs if o.final_passed) / n
        b_acc = sum(
            1 for o in b_outs if o.final_passed) / n
        outcome_cids = tuple(
            [o.cid() for o in a0_outs]
            + [o.cid() for o in a1_outs]
            + [o.cid() for o in b_outs])
        seed_merkle = _sha256_hex({
            "kind": "w88_humaneval_reflexion_seed_merkle_root",
            "seed": int(seed),
            "outcome_cids": list(outcome_cids),
        })
        per_seed.append(HumanEvalReflexionSeedReportV1(
            schema=(
                W88_HUMANEVAL_REFLEXION_BENCH_V1_SCHEMA_VERSION),
            seed=int(seed),
            n_problems=int(len(a0_outs)),
            a0_pass_at_1=float(a0_acc),
            a1_pass_at_1=float(a1_acc),
            b_pass_at_1=float(b_acc),
            a0_total_wall_ms=sum(
                int(o.total_wall_ms) for o in a0_outs),
            a1_total_wall_ms=sum(
                int(o.total_wall_ms) for o in a1_outs),
            b_total_wall_ms=sum(
                int(o.total_wall_ms) for o in b_outs),
            outcome_cids=outcome_cids,
            seed_merkle_root=str(seed_merkle)))
        all_outcome_cids.extend(outcome_cids)
    nseeds = float(len(per_seed))
    a0_mean = sum(s.a0_pass_at_1 for s in per_seed) / nseeds
    a1_mean = sum(s.a1_pass_at_1 for s in per_seed) / nseeds
    b_mean = sum(s.b_pass_at_1 for s in per_seed) / nseeds
    b_beats_a0 = tuple(
        s.b_pass_at_1 > s.a0_pass_at_1 for s in per_seed)
    b_beats_a1 = tuple(
        s.b_pass_at_1 > s.a1_pass_at_1 for s in per_seed)
    bench_merkle = _sha256_hex({
        "kind": "w88_humaneval_reflexion_bench_merkle_root",
        "model_id": str(model_id),
        "outcome_cids": list(all_outcome_cids),
        "seeds": list(cfg.seeds),
    })
    return HumanEvalReflexionBenchReportV1(
        schema=W88_HUMANEVAL_REFLEXION_BENCH_V1_SCHEMA_VERSION,
        model_id=str(model_id),
        n_problems=int(cfg.n_problems),
        n_seeds=int(len(cfg.seeds)),
        K_multi_sample=int(cfg.K_multi_sample),
        per_seed=tuple(per_seed),
        a0_mean_pass_at_1=float(a0_mean),
        a1_mean_pass_at_1=float(a1_mean),
        b_mean_pass_at_1=float(b_mean),
        b_beats_a0_per_seed=b_beats_a0,
        b_beats_a1_per_seed=b_beats_a1,
        b_strictly_beats_a0_on_all_seeds=bool(all(b_beats_a0)),
        b_strictly_beats_a1_on_all_seeds=bool(all(b_beats_a1)),
        b_mean_strictly_beats_a0_mean=bool(b_mean > a0_mean),
        b_mean_strictly_beats_a1_mean=bool(b_mean > a1_mean),
        b_mean_minus_a1_mean_pp=float(
            (b_mean - a1_mean) * 100.0),
        bench_merkle_root=str(bench_merkle))


__all__ = [
    "W88_HUMANEVAL_REFLEXION_BENCH_V1_SCHEMA_VERSION",
    "HumanEvalReflexionSeedReportV1",
    "HumanEvalReflexionBenchReportV1",
    "HumanEvalReflexionBenchConfigV1",
    "run_humaneval_reflexion_bench_v1",
]
