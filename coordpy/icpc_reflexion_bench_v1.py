"""W120 / COO-9 — official-ICPC stdin/stdout sequential-reflexion bench.

The same W88/W89 three-arm same-budget mechanism that retired the HumanEval-70B cap
(A0 single-shot / A1 first-pass-among-K / B sequential-reflexion-K), ported from the
LeetCode-style function-call corpus (``livecodebench_reflexion_bench_v1``) to the
official-ICPC **stdin/stdout** corpus.  The ONLY differences from the LCB bench are:

* the corpus = official ICPC problem packages (statement + samples + secret cases);
* the I/O model = a complete Python program reading stdin / writing stdout (NOT a
  ``class Solution``);
* the grader = the official secret cases via ``grade_icpc_candidate_case_v1`` (tier-1
  token-diff or tier-2 deterministic float oracle) — exit-0-iff-EVERY-secret-case-passes,
  NO LLM-as-judge.

Anti-cheat (carried forward verbatim) + the W120 reflexion-feedback discipline:

* Same model on every arm; same K budget on A1 and B (byte-exact, no early stop); no
  selective retries.
* Executor truth = the OFFICIAL secret test suite (the grader W119 proved present +
  self-test-passing).  ``final_passed`` ⟺ all secret cases pass.
* **Reflexion feedback NEVER leaks secret test data.**  Between B attempts the model
  sees only PUBLIC information — the judge verdict bit (accepted / rejected, exactly
  what a real online judge returns), the Python executor's stderr tail (a traceback,
  no inputs), and pass/fail + (input, expected, got) diffs on the PUBLIC SAMPLE cases
  shipped in the statement.  The secret ``.in``/``.ans`` are used ONLY to score, never
  shown.  This mirrors how a real contestant debugs and is strictly anti-cheat.

The report shape is byte-compatible with ``_mlb_rates`` + ``_evaluate_phase2_gates``
(imported verbatim from the W108 driver by the pilot script), so W120's MLB-1 / MLB-2
sub-gates + the 9 Phase-2 gates are scored by the SAME code as W103/W105/W108/W113.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from typing import Any, Callable, Optional, Sequence

from .coordpy_icpc_battlefield_v1 import (
    KIND_PASSFAIL,
    grade_icpc_candidate_case_v1,
)

W120_ICPC_REFLEXION_BENCH_V1_SCHEMA_VERSION: str = (
    "coordpy.icpc_reflexion_bench_v1.v1")


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def extract_candidate_code_v1(*, response_text: str) -> str:
    """Pull the candidate Python program from a model response (LAST fence preferred —
    verbatim shape from the LCB bench)."""
    text = str(response_text or "")
    fences = _FENCE_RE.findall(text)
    return (fences[-1].strip() if fences else text.strip())


# ============================================================ the pilot problem record

@dataclasses.dataclass(frozen=True)
class IcpcPilotProblemV1:
    """One admitted official-ICPC problem prepared for the pilot.

    ``secret_cases`` is the GRADER (official hidden tests); ``samples`` is the PUBLIC
    feedback surface for reflexion.  ``kind`` selects the oracle (tier-1 diff / tier-2
    float); ``float_tol`` is the package's validator tolerance.
    """

    problem_id: str
    short_name: str
    source_repo: str
    contest_date: str
    statement: str
    kind: str
    float_tol: float
    samples: tuple[tuple[str, str], ...]
    secret_cases: tuple[tuple[str, str], ...]

    @property
    def question_id(self) -> str:    # alias so progress callbacks match the LCB driver
        return self.problem_id


# ============================================================ prompts (stdin/stdout)

def _samples_block(problem: IcpcPilotProblemV1, *, max_n: int = 3) -> str:
    chunks = []
    for i, (inp, out) in enumerate(problem.samples[:max_n]):
        chunks.append(f"Sample Input {i+1}:\n{inp.rstrip()}\n\n"
                      f"Sample Output {i+1}:\n{out.rstrip()}")
    return "\n\n".join(chunks) if chunks else "(no sample cases shipped)"


def _initial_prompt(problem: IcpcPilotProblemV1) -> str:
    return (
        "You are an expert competitive programmer at the ICPC. Solve the problem "
        "below by writing a COMPLETE Python 3 program that reads ALL input from "
        "standard input and writes the answer to standard output.\n\n"
        "Output ONLY the complete program inside a single ```python ... ``` code "
        "fence. No explanation, no tests. The program must run as a standalone "
        "script (read stdin, print to stdout).\n\n"
        f"Problem:\n{problem.statement}\n\n"
        f"{_samples_block(problem)}\n\n"
        "Your complete Python 3 program:")


def _reflexion_prompt(problem: IcpcPilotProblemV1,
                      history: Sequence[tuple[str, bool, str, str]],
                      attempt_idx: int) -> str:
    chunks: list[str] = []
    for i, (cand, passed, stderr_tail, sample_fb) in enumerate(history):
        cand_trim = cand if len(cand) <= 1500 else (cand[:1500] + "\n# ...(truncated)\n")
        verdict = ("ACCEPTED by the judge (all hidden tests passed)" if passed
                   else "REJECTED by the judge (failed at least one hidden test)")
        se = f"\nExecutor stderr (tail):\n{stderr_tail.strip()}" if stderr_tail.strip() else ""
        sf = f"\nPublic sample results:\n{sample_fb}" if sample_fb.strip() else ""
        chunks.append(f"--- Attempt {i+1} ({verdict}) ---\n"
                      f"```python\n{cand_trim}\n```{se}{sf}")
    return (
        "You are an expert ICPC competitor on a reflective debugging loop. You are on "
        f"attempt {attempt_idx + 1} out of 5. Below are your previous attempts with the "
        "judge verdict, the Python executor's stderr tail, and the PUBLIC sample-case "
        "results (the hidden tests are NOT shown). Diagnose the bug in each failing "
        "attempt and produce a NEW corrected COMPLETE Python 3 stdin/stdout program. Do "
        "not repeat a previous attempt verbatim.\n\n"
        f"Problem:\n{problem.statement}\n\n"
        f"{_samples_block(problem)}\n\n"
        f"{chr(10).join(chunks)}\n\n"
        "Provide ONLY the corrected complete Python 3 program in a "
        "```python ... ``` fence:")


_GenerateFn = Callable[[str, int, float], tuple[str, int]]


# ============================================================ grading helpers

def grade_on_secret_v1(problem: IcpcPilotProblemV1, code: str,
                       *, timeout_s: float) -> tuple[bool, str, int]:
    """PASS iff EVERY official secret case passes (short-circuit on first failure).
    Returns (passed, stderr_tail_of_first_failure, n_cases_checked)."""
    n = 0
    for inp, exp in problem.secret_cases:
        n += 1
        r = grade_icpc_candidate_case_v1(
            candidate_code=code, stdin_text=inp, expected_stdout=exp,
            kind=problem.kind, float_tol=problem.float_tol, timeout_s=timeout_s)
        if not r.passed:
            tail = r.stderr_tail if r.stderr_tail else (
                "TIMEOUT" if r.timed_out else "wrong answer on a hidden case")
            return False, tail, n
    return True, "", n


def sample_feedback_v1(problem: IcpcPilotProblemV1, code: str,
                       *, timeout_s: float, max_n: int = 3) -> str:
    """PUBLIC reflexion feedback: run candidate on the SAMPLE cases only."""
    if not problem.samples:
        return "(no public samples to check)"
    lines = []
    for i, (inp, exp) in enumerate(problem.samples[:max_n]):
        r = grade_icpc_candidate_case_v1(
            candidate_code=code, stdin_text=inp, expected_stdout=exp,
            kind=problem.kind, float_tol=problem.float_tol, timeout_s=timeout_s)
        if r.passed:
            lines.append(f"  sample {i+1}: PASS")
        elif r.timed_out:
            lines.append(f"  sample {i+1}: TIMEOUT")
        elif r.returncode != 0:
            lines.append(f"  sample {i+1}: runtime error (rc={r.returncode})")
        else:
            lines.append(f"  sample {i+1}: WRONG (expected `{exp.strip()[:60]}`)")
    return "\n".join(lines)


# ============================================================ arm outcomes / reports
# (report shape mirrors LCBArmOutcomeV1 / LCBSeedReportV1 / LCBBenchReportV1 so the
#  W108 _mlb_rates + _evaluate_phase2_gates score W120 byte-identically.)

@dataclasses.dataclass(frozen=True)
class IcpcArmOutcomeV1:
    schema: str
    seed: int
    question_id: str
    arm_id: str
    final_passed: bool
    n_model_calls: int
    per_call_passed: tuple[bool, ...]
    first_pass_attempt_idx: int

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema, "seed": int(self.seed),
                "question_id": self.question_id, "arm_id": self.arm_id,
                "final_passed": bool(self.final_passed),
                "n_model_calls": int(self.n_model_calls),
                "per_call_passed": list(self.per_call_passed),
                "first_pass_attempt_idx": int(self.first_pass_attempt_idx)}

    def cid(self) -> str:
        return _sha256_hex({"kind": "w120_icpc_arm_outcome_v1", "o": self.to_dict()})


def _run_a0(*, seed, p, gen, max_tokens, timeout_s) -> IcpcArmOutcomeV1:
    text, _ = gen(_initial_prompt(p), max_tokens, 0.0)
    code = extract_candidate_code_v1(response_text=text)
    passed, _, _ = grade_on_secret_v1(p, code, timeout_s=timeout_s)
    return IcpcArmOutcomeV1(
        schema=W120_ICPC_REFLEXION_BENCH_V1_SCHEMA_VERSION, seed=int(seed),
        question_id=p.question_id, arm_id="A0", final_passed=bool(passed),
        n_model_calls=1, per_call_passed=(bool(passed),),
        first_pass_attempt_idx=(0 if passed else -1))


def _run_a1(*, seed, p, K, temperature, gen, max_tokens, timeout_s) -> IcpcArmOutcomeV1:
    prompt = _initial_prompt(p)
    per_call: list[bool] = []
    chosen_idx = -1
    for k in range(int(K)):
        text, _ = gen(prompt, max_tokens, float(temperature))
        code = extract_candidate_code_v1(response_text=text)
        passed, _, _ = grade_on_secret_v1(p, code, timeout_s=timeout_s)
        per_call.append(bool(passed))
        if passed and chosen_idx == -1:
            chosen_idx = int(k)
    return IcpcArmOutcomeV1(
        schema=W120_ICPC_REFLEXION_BENCH_V1_SCHEMA_VERSION, seed=int(seed),
        question_id=p.question_id, arm_id="A1",
        final_passed=bool(chosen_idx >= 0), n_model_calls=int(K),
        per_call_passed=tuple(per_call), first_pass_attempt_idx=int(chosen_idx))


def _run_b(*, seed, p, K, temperature, gen, max_tokens, timeout_s) -> IcpcArmOutcomeV1:
    history: list[tuple[str, bool, str, str]] = []
    per_call: list[bool] = []
    first_pass_idx = -1
    for k in range(int(K)):
        prompt = (_initial_prompt(p) if k == 0
                  else _reflexion_prompt(p, tuple(history), attempt_idx=int(k)))
        text, _ = gen(prompt, max_tokens, float(temperature))
        code = extract_candidate_code_v1(response_text=text)
        passed, stderr_tail, _ = grade_on_secret_v1(p, code, timeout_s=timeout_s)
        per_call.append(bool(passed))
        sfb = sample_feedback_v1(p, code, timeout_s=timeout_s)
        history.append((code, bool(passed), stderr_tail, sfb))
        if passed and first_pass_idx == -1:
            first_pass_idx = int(k)
    return IcpcArmOutcomeV1(
        schema=W120_ICPC_REFLEXION_BENCH_V1_SCHEMA_VERSION, seed=int(seed),
        question_id=p.question_id, arm_id="B",
        final_passed=bool(first_pass_idx >= 0), n_model_calls=int(K),
        per_call_passed=tuple(per_call), first_pass_attempt_idx=int(first_pass_idx))


@dataclasses.dataclass(frozen=True)
class IcpcSeedReportV1:
    schema: str
    seed: int
    n_problems: int
    a0_pass_at_1: float
    a1_pass_at_1: float
    b_pass_at_1: float
    per_problem_a0_passed: tuple[bool, ...]
    per_problem_a1_passed: tuple[bool, ...]
    per_problem_b_passed: tuple[bool, ...]
    per_problem_b_first_pass_idx: tuple[int, ...]
    question_ids: tuple[str, ...]
    seed_merkle_root: str

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema, "seed": int(self.seed),
                "n_problems": int(self.n_problems),
                "a0_pass_at_1": round(self.a0_pass_at_1, 6),
                "a1_pass_at_1": round(self.a1_pass_at_1, 6),
                "b_pass_at_1": round(self.b_pass_at_1, 6),
                "per_problem_a0_passed": list(self.per_problem_a0_passed),
                "per_problem_a1_passed": list(self.per_problem_a1_passed),
                "per_problem_b_passed": list(self.per_problem_b_passed),
                "per_problem_b_first_pass_idx": list(self.per_problem_b_first_pass_idx),
                "question_ids": list(self.question_ids),
                "seed_merkle_root": self.seed_merkle_root}


@dataclasses.dataclass(frozen=True)
class IcpcBenchReportV1:
    schema: str
    model_id: str
    n_problems: int
    n_seeds: int
    K_multi_sample: int
    per_seed: tuple[IcpcSeedReportV1, ...]
    a0_mean_pass_at_1: float
    a1_mean_pass_at_1: float
    b_mean_pass_at_1: float
    b_mean_minus_a1_mean_pp: float
    bench_merkle_root: str

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema, "model_id": self.model_id,
                "n_problems": int(self.n_problems), "n_seeds": int(self.n_seeds),
                "K_multi_sample": int(self.K_multi_sample),
                "per_seed": [s.to_dict() for s in self.per_seed],
                "a0_mean_pass_at_1": round(self.a0_mean_pass_at_1, 6),
                "a1_mean_pass_at_1": round(self.a1_mean_pass_at_1, 6),
                "b_mean_pass_at_1": round(self.b_mean_pass_at_1, 6),
                "b_mean_minus_a1_mean_pp": round(self.b_mean_minus_a1_mean_pp, 4),
                "bench_merkle_root": self.bench_merkle_root}

    def cid(self) -> str:
        return _sha256_hex({"kind": "w120_icpc_bench_report_v1", "r": self.to_dict()})


@dataclasses.dataclass
class IcpcBenchConfigV1:
    schema: str = W120_ICPC_REFLEXION_BENCH_V1_SCHEMA_VERSION
    K_multi_sample: int = 5
    seeds: tuple[int, ...] = (120_001,)
    sampling_temperature: float = 0.7
    max_tokens_per_call: int = 1536
    executor_timeout_s: float = 15.0


def run_icpc_reflexion_bench_v1(
        *, gen: _GenerateFn, model_id: str,
        subset: Sequence[IcpcPilotProblemV1],
        config: Optional[IcpcBenchConfigV1] = None,
        on_problem_start: Optional[Callable[[int, int, str], None]] = None,
) -> IcpcBenchReportV1:
    """Run A0 + A1 + B (sequential reflexion) on a pre-selected ordered subset per seed.
    The subset (the deterministic core-tier pilot slice) is consumed verbatim."""
    cfg = config or IcpcBenchConfigV1()
    per_seed: list[IcpcSeedReportV1] = []
    all_cids: list[str] = []
    for seed in cfg.seeds:
        a0o: list[IcpcArmOutcomeV1] = []
        a1o: list[IcpcArmOutcomeV1] = []
        bo: list[IcpcArmOutcomeV1] = []
        for p_idx, p in enumerate(subset):
            if on_problem_start is not None:
                on_problem_start(int(seed), int(p_idx), str(p.question_id))
            a0o.append(_run_a0(seed=seed, p=p, gen=gen,
                               max_tokens=cfg.max_tokens_per_call,
                               timeout_s=cfg.executor_timeout_s))
            a1o.append(_run_a1(seed=seed, p=p, K=cfg.K_multi_sample,
                               temperature=cfg.sampling_temperature, gen=gen,
                               max_tokens=cfg.max_tokens_per_call,
                               timeout_s=cfg.executor_timeout_s))
            bo.append(_run_b(seed=seed, p=p, K=cfg.K_multi_sample,
                             temperature=cfg.sampling_temperature, gen=gen,
                             max_tokens=cfg.max_tokens_per_call,
                             timeout_s=cfg.executor_timeout_s))
        n = float(len(a0o)) or 1.0
        a0_acc = sum(1 for o in a0o if o.final_passed) / n
        a1_acc = sum(1 for o in a1o if o.final_passed) / n
        b_acc = sum(1 for o in bo if o.final_passed) / n
        cids = tuple([o.cid() for o in a0o] + [o.cid() for o in a1o]
                     + [o.cid() for o in bo])
        seed_merkle = _sha256_hex({"kind": "w120_icpc_seed_merkle_v1",
                                   "seed": int(seed), "cids": list(cids)})
        per_seed.append(IcpcSeedReportV1(
            schema=W120_ICPC_REFLEXION_BENCH_V1_SCHEMA_VERSION, seed=int(seed),
            n_problems=int(len(a0o)), a0_pass_at_1=float(a0_acc),
            a1_pass_at_1=float(a1_acc), b_pass_at_1=float(b_acc),
            per_problem_a0_passed=tuple(bool(o.final_passed) for o in a0o),
            per_problem_a1_passed=tuple(bool(o.final_passed) for o in a1o),
            per_problem_b_passed=tuple(bool(o.final_passed) for o in bo),
            per_problem_b_first_pass_idx=tuple(int(o.first_pass_attempt_idx) for o in bo),
            question_ids=tuple(str(o.question_id) for o in a0o),
            seed_merkle_root=str(seed_merkle)))
        all_cids.extend(cids)
    ns = float(len(per_seed)) or 1.0
    a0m = sum(s.a0_pass_at_1 for s in per_seed) / ns
    a1m = sum(s.a1_pass_at_1 for s in per_seed) / ns
    bm = sum(s.b_pass_at_1 for s in per_seed) / ns
    bench_merkle = _sha256_hex({"kind": "w120_icpc_bench_merkle_v1",
                                "model_id": str(model_id), "cids": all_cids,
                                "seeds": list(cfg.seeds)})
    return IcpcBenchReportV1(
        schema=W120_ICPC_REFLEXION_BENCH_V1_SCHEMA_VERSION, model_id=str(model_id),
        n_problems=int(len(subset)), n_seeds=int(len(per_seed)),
        K_multi_sample=int(cfg.K_multi_sample), per_seed=tuple(per_seed),
        a0_mean_pass_at_1=float(a0m), a1_mean_pass_at_1=float(a1m),
        b_mean_pass_at_1=float(bm),
        b_mean_minus_a1_mean_pp=float((bm - a1m) * 100.0),
        bench_merkle_root=str(bench_merkle))


__all__ = [
    "W120_ICPC_REFLEXION_BENCH_V1_SCHEMA_VERSION",
    "extract_candidate_code_v1", "IcpcPilotProblemV1",
    "grade_on_secret_v1", "sample_feedback_v1",
    "IcpcArmOutcomeV1", "IcpcSeedReportV1", "IcpcBenchReportV1",
    "IcpcBenchConfigV1", "run_icpc_reflexion_bench_v1",
]
