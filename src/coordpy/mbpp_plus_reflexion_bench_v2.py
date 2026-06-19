"""W102 / COO-9 — MBPP+ V2 sequential-reflexion bench.

Wires the W102 MBPP+ V2 loader + executor with the W88 / W89
sequential-reflexion B-pipeline that retired the same-budget
HumanEval-70B cap.  The mechanism is **byte-identical** to the
W88 / W90 / W101-V1 reflexion shape (K=5 same-budget; first-pass-
among-K for A1; sequential reflexion with cumulative stderr
history for B); the only changes from V1 are:

* The corpus is loaded by `coordpy.mbpp_plus_loader_v2` (actual
  EvalPlus HF parquet schema).
* The executor is `coordpy.mbpp_plus_executor_v2` (runs the real
  `test` program; PASSes iff subprocess returncode==0).
* The schema strings are W102 so the audit chains do not collide
  with the W101-V1 (broken) runs.

Three arms (mirrors W88 / W90 verbatim where applicable):

* ``A0`` — stock single-shot at T=0.0.
* ``A1`` — first-pass-among-K=5 self-consistency at T=0.7.
* ``B`` — sequential-reflexion-K=5 at T=0.7, each turn conditioned
  on the cumulative (candidate, executor_stderr) history.

Anti-cheat (carried forward from W88 / W90 / W101 verbatim):

* Same model on every arm.
* Same task subset per seed across arms.
* Same K=5 budget on A1 and B (byte-exact; no early-stop).
* No selective retries.
* Executor truth = run the EvalPlus `test` program and exit 0.
* Per-call SHA-256 + per-seed Merkle + bench Merkle re-verifiable
  offline.

Honest scope (W102)
-------------------

* ``W102-L-MBPP-PLUS-REFLEXION-V2-NIM-DEPENDENT-CAP`` — V2 drives
  the bench through any ``LLMBackend``-shaped client; provider
  determinism is not assumed beyond temperature=0.
* ``W102-L-MBPP-PLUS-REFLEXION-V2-EXECUTOR-TEST-PROGRAM-CAP`` —
  the executor runs the canonical EvalPlus `test` program in a
  fresh CPython subprocess with the `-I` flag (no current dir,
  no PYTHON* env vars; but site packages ENABLED so the
  EvalPlus tests can import numpy).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import random
from typing import Any, Callable, Sequence

from .mbpp_plus_executor_v2 import (
    W102_MBPP_PLUS_EXECUTOR_V2_KILL_AFTER_S,
    W102_MBPP_PLUS_EXECUTOR_V2_TIMEOUT_S,
    MbppPlusV2ExecutorResult,
    run_mbpp_plus_executor_v2,
)
from .mbpp_plus_loader_v2 import MbppPlusProblemV2
from .mbpp_reflexion_bench_v1 import (
    extract_candidate_code_v1,
)


W102_MBPP_PLUS_REFLEXION_BENCH_V2_SCHEMA_VERSION: str = (
    "coordpy.mbpp_plus_reflexion_bench_v2.v1")


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


# ---------------------------------------------------------------
# Prompts (verbatim port from V1 / MBPP base reflexion bench)
# ---------------------------------------------------------------

_SYSTEM = (
    "You are an expert Python programmer.  When given a function "
    "description and a sample assertion, output ONLY the complete "
    "Python function (inside a ```python ... ``` code fence).  Do "
    "not include any prose before or after the code fence.")


def _initial_prompt(p: MbppPlusProblemV2) -> str:
    sample_assert = (
        p.base_test_list[0] if p.base_test_list else "")
    return (
        f"{_SYSTEM}\n\n"
        "Write a Python function that satisfies the description "
        "below.  Your function must pass the sample assertion "
        "AND any hidden assertions.\n\n"
        f"Description: {p.prompt}\n\n"
        f"Sample assertion (the function must pass this and "
        f"similar):\n```python\n{sample_assert}\n```\n\n"
        "Your complete Python function:")


def _reflexion_prompt(
        p: MbppPlusProblemV2,
        history: Sequence[tuple[str, MbppPlusV2ExecutorResult]],
        attempt_idx: int,
) -> str:
    chunks: list[str] = []
    for i, (cand, exe) in enumerate(history):
        cand_trim = cand
        if len(cand_trim) > 1500:
            cand_trim = cand_trim[:1500] + "\n# ... (truncated)\n"
        if exe.passed:
            verdict = (
                "PASSED all base + EvalPlus extra assertions")
            stderr_excerpt = ""
        else:
            verdict = (
                f"FAILED (returncode={exe.returncode}"
                + (", TIMED OUT" if exe.timed_out else "")
                + ")")
            stderr_text = exe.stderr_tail.strip()
            stderr_excerpt = (
                f"\nExecutor stderr (tail):\n{stderr_text}"
                if stderr_text else "")
        chunks.append(
            f"--- Attempt {i+1} ({verdict}) ---\n"
            f"```python\n{cand_trim}\n```{stderr_excerpt}")
    sample_assert = (
        p.base_test_list[0] if p.base_test_list else "")
    return (
        f"{_SYSTEM}\n\n"
        "[Role: reflective code generator]\n"
        f"You are on attempt {attempt_idx + 1} out of 5.\n\n"
        f"Description: {p.prompt}\n\n"
        f"Sample assertion:\n```python\n{sample_assert}\n```\n\n"
        f"{chr(10).join(chunks)}\n\n"
        "Diagnose the bug class in the failing attempt(s) and "
        "produce a NEW corrected Python function.  Do not repeat "
        "a previous attempt verbatim.  Provide ONLY the corrected "
        "function in a ```python ... ``` fence:")


# ---------------------------------------------------------------
# Capsules + per-arm runners
# ---------------------------------------------------------------

_GenerateFn = Callable[[str, int, float], tuple[str, int]]


@dataclasses.dataclass(frozen=True)
class MbppPlusV2ArmCallCapsule:
    schema: str
    seed: int
    task_id: str
    arm_id: str
    role: str
    call_idx: int
    temperature: float
    prompt_cid: str
    response_cid: str
    wall_ms: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "seed": int(self.seed),
            "task_id": str(self.task_id),
            "arm_id": str(self.arm_id),
            "role": str(self.role),
            "call_idx": int(self.call_idx),
            "temperature": float(round(
                self.temperature, 6)),
            "prompt_cid": str(self.prompt_cid),
            "response_cid": str(self.response_cid),
            "wall_ms": int(self.wall_ms),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w102_mbpp_plus_v2_arm_call_capsule",
            "capsule": self.to_dict(),
        })


@dataclasses.dataclass(frozen=True)
class MbppPlusV2ArmOutcomeCapsule:
    schema: str
    seed: int
    task_id: str
    arm_id: str
    final_passed: bool
    final_candidate_code_cid: str
    n_model_calls: int
    n_executor_calls: int
    total_wall_ms: int
    call_capsule_cids: tuple[str, ...]
    executor_result_cids: tuple[str, ...]
    per_call_passed: tuple[bool, ...]
    first_pass_attempt_idx: int  # -1 if no PASS; MLB diagnostics

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "seed": int(self.seed),
            "task_id": str(self.task_id),
            "arm_id": str(self.arm_id),
            "final_passed": bool(self.final_passed),
            "final_candidate_code_cid": str(
                self.final_candidate_code_cid),
            "n_model_calls": int(self.n_model_calls),
            "n_executor_calls": int(self.n_executor_calls),
            "total_wall_ms": int(self.total_wall_ms),
            "call_capsule_cids": list(self.call_capsule_cids),
            "executor_result_cids": list(
                self.executor_result_cids),
            "per_call_passed": list(self.per_call_passed),
            "first_pass_attempt_idx": int(
                self.first_pass_attempt_idx),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w102_mbpp_plus_v2_arm_outcome_capsule",
            "capsule": self.to_dict(),
        })


def _run_a0_single_shot(
        *, seed: int, p: MbppPlusProblemV2,
        gen: _GenerateFn, max_tokens: int,
        executor_kwargs: dict[str, Any],
) -> MbppPlusV2ArmOutcomeCapsule:
    prompt = _initial_prompt(p)
    text, wall = gen(prompt, max_tokens, 0.0)
    code = extract_candidate_code_v1(
        response_text=text, entry_point=p.entry_point)
    exe = run_mbpp_plus_executor_v2(
        problem=p, candidate_code=code, **executor_kwargs)
    call = MbppPlusV2ArmCallCapsule(
        schema=W102_MBPP_PLUS_REFLEXION_BENCH_V2_SCHEMA_VERSION,
        seed=int(seed), task_id=str(p.task_id),
        arm_id="A0", role="solver", call_idx=0,
        temperature=0.0,
        prompt_cid=hashlib.sha256(
            prompt.encode("utf-8")).hexdigest(),
        response_cid=hashlib.sha256(
            text.encode("utf-8")).hexdigest(),
        wall_ms=int(wall))
    return MbppPlusV2ArmOutcomeCapsule(
        schema=W102_MBPP_PLUS_REFLEXION_BENCH_V2_SCHEMA_VERSION,
        seed=int(seed), task_id=str(p.task_id),
        arm_id="A0",
        final_passed=bool(exe.passed),
        final_candidate_code_cid=str(exe.candidate_code_cid),
        n_model_calls=1,
        n_executor_calls=1,
        total_wall_ms=int(wall + exe.wall_ms),
        call_capsule_cids=(call.cid(),),
        executor_result_cids=(exe.cid(),),
        per_call_passed=(bool(exe.passed),),
        first_pass_attempt_idx=(0 if exe.passed else -1))


def _run_a1_first_pass_among_K(
        *, seed: int, p: MbppPlusProblemV2, K: int,
        temperature: float, gen: _GenerateFn, max_tokens: int,
        executor_kwargs: dict[str, Any],
) -> MbppPlusV2ArmOutcomeCapsule:
    prompt = _initial_prompt(p)
    calls: list[MbppPlusV2ArmCallCapsule] = []
    exes: list[MbppPlusV2ExecutorResult] = []
    total = 0
    chosen_passed = False
    chosen_code_cid = ""
    chosen_idx = -1
    per_call_passed: list[bool] = []
    for k in range(int(K)):
        text, wall = gen(prompt, max_tokens, float(temperature))
        code = extract_candidate_code_v1(
            response_text=text, entry_point=p.entry_point)
        exe = run_mbpp_plus_executor_v2(
            problem=p, candidate_code=code,
            **executor_kwargs)
        calls.append(MbppPlusV2ArmCallCapsule(
            schema=(
                W102_MBPP_PLUS_REFLEXION_BENCH_V2_SCHEMA_VERSION),
            seed=int(seed), task_id=str(p.task_id),
            arm_id="A1", role="sample", call_idx=int(k),
            temperature=float(temperature),
            prompt_cid=hashlib.sha256(
                prompt.encode("utf-8")).hexdigest(),
            response_cid=hashlib.sha256(
                text.encode("utf-8")).hexdigest(),
            wall_ms=int(wall)))
        exes.append(exe)
        per_call_passed.append(bool(exe.passed))
        total += int(wall) + int(exe.wall_ms)
        if exe.passed and not chosen_passed:
            chosen_passed = True
            chosen_code_cid = str(exe.candidate_code_cid)
            chosen_idx = int(k)
    if not chosen_passed:
        chosen_code_cid = str(exes[0].candidate_code_cid)
    return MbppPlusV2ArmOutcomeCapsule(
        schema=W102_MBPP_PLUS_REFLEXION_BENCH_V2_SCHEMA_VERSION,
        seed=int(seed), task_id=str(p.task_id),
        arm_id="A1",
        final_passed=bool(chosen_passed),
        final_candidate_code_cid=str(chosen_code_cid),
        n_model_calls=int(K),
        n_executor_calls=int(K),
        total_wall_ms=int(total),
        call_capsule_cids=tuple(c.cid() for c in calls),
        executor_result_cids=tuple(e.cid() for e in exes),
        per_call_passed=tuple(per_call_passed),
        first_pass_attempt_idx=int(chosen_idx))


def _run_b_sequential_reflexion(
        *, seed: int, p: MbppPlusProblemV2, K: int,
        temperature: float, gen: _GenerateFn, max_tokens: int,
        executor_kwargs: dict[str, Any],
) -> MbppPlusV2ArmOutcomeCapsule:
    history: list[tuple[str, MbppPlusV2ExecutorResult]] = []
    calls: list[MbppPlusV2ArmCallCapsule] = []
    exes: list[MbppPlusV2ExecutorResult] = []
    total = 0
    per_call_passed: list[bool] = []
    first_pass_idx = -1
    for k in range(int(K)):
        if k == 0:
            prompt = _initial_prompt(p)
        else:
            prompt = _reflexion_prompt(
                p, tuple(history), attempt_idx=int(k))
        text, wall = gen(prompt, max_tokens, float(temperature))
        code = extract_candidate_code_v1(
            response_text=text, entry_point=p.entry_point)
        exe = run_mbpp_plus_executor_v2(
            problem=p, candidate_code=code,
            **executor_kwargs)
        calls.append(MbppPlusV2ArmCallCapsule(
            schema=(
                W102_MBPP_PLUS_REFLEXION_BENCH_V2_SCHEMA_VERSION),
            seed=int(seed), task_id=str(p.task_id),
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
        per_call_passed.append(bool(exe.passed))
        history.append((code, exe))
        total += int(wall) + int(exe.wall_ms)
        if exe.passed and first_pass_idx == -1:
            first_pass_idx = int(k)
    final_passed = (first_pass_idx >= 0)
    if final_passed:
        final_code_cid = str(
            exes[first_pass_idx].candidate_code_cid)
    else:
        cids = sorted(
            str(e.candidate_code_cid) for e in exes)
        final_code_cid = cids[0]
    return MbppPlusV2ArmOutcomeCapsule(
        schema=W102_MBPP_PLUS_REFLEXION_BENCH_V2_SCHEMA_VERSION,
        seed=int(seed), task_id=str(p.task_id),
        arm_id="B",
        final_passed=bool(final_passed),
        final_candidate_code_cid=str(final_code_cid),
        n_model_calls=int(K),
        n_executor_calls=int(K),
        total_wall_ms=int(total),
        call_capsule_cids=tuple(c.cid() for c in calls),
        executor_result_cids=tuple(e.cid() for e in exes),
        per_call_passed=tuple(per_call_passed),
        first_pass_attempt_idx=int(first_pass_idx))


# ---------------------------------------------------------------
# Reports
# ---------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class MbppPlusV2SeedReport:
    schema: str
    seed: int
    n_problems: int
    a0_pass_at_1: float
    a1_pass_at_1: float
    b_pass_at_1: float
    a0_total_wall_ms: int
    a1_total_wall_ms: int
    b_total_wall_ms: int
    per_problem_a0_passed: tuple[bool, ...]
    per_problem_a1_passed: tuple[bool, ...]
    per_problem_b_passed: tuple[bool, ...]
    per_problem_b_first_pass_idx: tuple[int, ...]
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
            "per_problem_a0_passed": list(
                self.per_problem_a0_passed),
            "per_problem_a1_passed": list(
                self.per_problem_a1_passed),
            "per_problem_b_passed": list(
                self.per_problem_b_passed),
            "per_problem_b_first_pass_idx": list(
                self.per_problem_b_first_pass_idx),
            "outcome_cids": list(self.outcome_cids),
            "seed_merkle_root": str(self.seed_merkle_root),
        }


@dataclasses.dataclass(frozen=True)
class MbppPlusV2BenchReport:
    schema: str
    model_id: str
    n_problems: int
    n_seeds: int
    K_multi_sample: int
    per_seed: tuple[MbppPlusV2SeedReport, ...]
    a0_mean_pass_at_1: float
    a1_mean_pass_at_1: float
    b_mean_pass_at_1: float
    b_beats_a0_per_seed: tuple[bool, ...]
    b_beats_a1_per_seed: tuple[bool, ...]
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
            "kind": "w102_mbpp_plus_v2_bench_report",
            "report": self.to_dict(),
        })


# ---------------------------------------------------------------
# Subset + driver
# ---------------------------------------------------------------

def select_mbpp_plus_v2_subset(
        *, corpus: Sequence[MbppPlusProblemV2],
        n_problems: int, seed: int,
) -> tuple[MbppPlusProblemV2, ...]:
    """Deterministically select ``n_problems`` from the V2 corpus.

    Uses the same seed namespace as V1 (101_001..) so the cheap-
    pilot slice is comparable across V1 → V2 (V1 was a no-op
    bench because of the schema bug; V2 actually exercises the
    EvalPlus extra tests)."""
    rng = random.Random(int(seed))
    idxs = list(range(len(corpus)))
    rng.shuffle(idxs)
    chosen = idxs[: int(n_problems)]
    return tuple(corpus[i] for i in chosen)


@dataclasses.dataclass
class MbppPlusV2BenchConfig:
    schema: str = (
        W102_MBPP_PLUS_REFLEXION_BENCH_V2_SCHEMA_VERSION)
    n_problems: int = 30
    K_multi_sample: int = 5
    seeds: tuple[int, ...] = (101_001,)  # 1-seed cheap pilot
    sampling_temperature: float = 0.7
    max_tokens_per_call: int = 768
    executor_timeout_s: float = (
        W102_MBPP_PLUS_EXECUTOR_V2_TIMEOUT_S)
    executor_kill_after_s: float = (
        W102_MBPP_PLUS_EXECUTOR_V2_KILL_AFTER_S)
    executor_mode: str = "base_and_plus"


def run_mbpp_plus_reflexion_bench_v2(
        *,
        gen: _GenerateFn,
        model_id: str,
        corpus: Sequence[MbppPlusProblemV2],
        config: MbppPlusV2BenchConfig | None = None,
        on_problem_start: (
            Callable[[int, int, str], None] | None) = None,
) -> MbppPlusV2BenchReport:
    """Run A0 + A1 + B (sequential reflexion) on a deterministic
    subset for each seed.  Returns a content-addressed bench
    report."""
    cfg = config or MbppPlusV2BenchConfig()
    executor_kwargs = {
        "timeout_s": float(cfg.executor_timeout_s),
        "kill_after_s": float(cfg.executor_kill_after_s),
        "mode": str(cfg.executor_mode),
    }
    per_seed: list[MbppPlusV2SeedReport] = []
    all_outcome_cids: list[str] = []
    for seed in cfg.seeds:
        subset = select_mbpp_plus_v2_subset(
            corpus=corpus, n_problems=int(cfg.n_problems),
            seed=int(seed))
        a0_outs: list[MbppPlusV2ArmOutcomeCapsule] = []
        a1_outs: list[MbppPlusV2ArmOutcomeCapsule] = []
        b_outs: list[MbppPlusV2ArmOutcomeCapsule] = []
        for p_idx, problem in enumerate(subset):
            if on_problem_start is not None:
                on_problem_start(
                    int(seed), int(p_idx),
                    str(problem.task_id))
            a0_outs.append(_run_a0_single_shot(
                seed=int(seed), p=problem, gen=gen,
                max_tokens=int(cfg.max_tokens_per_call),
                executor_kwargs=executor_kwargs))
            a1_outs.append(_run_a1_first_pass_among_K(
                seed=int(seed), p=problem,
                K=int(cfg.K_multi_sample),
                temperature=float(cfg.sampling_temperature),
                gen=gen,
                max_tokens=int(cfg.max_tokens_per_call),
                executor_kwargs=executor_kwargs))
            b_outs.append(_run_b_sequential_reflexion(
                seed=int(seed), p=problem,
                K=int(cfg.K_multi_sample),
                temperature=float(cfg.sampling_temperature),
                gen=gen,
                max_tokens=int(cfg.max_tokens_per_call),
                executor_kwargs=executor_kwargs))
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
            "kind": "w102_mbpp_plus_v2_seed_merkle_root",
            "seed": int(seed),
            "outcome_cids": list(outcome_cids),
        })
        per_seed.append(MbppPlusV2SeedReport(
            schema=(
                W102_MBPP_PLUS_REFLEXION_BENCH_V2_SCHEMA_VERSION),
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
            per_problem_a0_passed=tuple(
                bool(o.final_passed) for o in a0_outs),
            per_problem_a1_passed=tuple(
                bool(o.final_passed) for o in a1_outs),
            per_problem_b_passed=tuple(
                bool(o.final_passed) for o in b_outs),
            per_problem_b_first_pass_idx=tuple(
                int(o.first_pass_attempt_idx) for o in b_outs),
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
        "kind": "w102_mbpp_plus_v2_bench_merkle_root",
        "model_id": str(model_id),
        "outcome_cids": all_outcome_cids,
        "seeds": list(cfg.seeds),
    })
    return MbppPlusV2BenchReport(
        schema=W102_MBPP_PLUS_REFLEXION_BENCH_V2_SCHEMA_VERSION,
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
        b_mean_strictly_beats_a0_mean=bool(b_mean > a0_mean),
        b_mean_strictly_beats_a1_mean=bool(b_mean > a1_mean),
        b_mean_minus_a1_mean_pp=float(
            (b_mean - a1_mean) * 100.0),
        bench_merkle_root=str(bench_merkle))


# ---------------------------------------------------------------
# MLB sub-gate helpers (post-pilot evaluation)
# ---------------------------------------------------------------

def mlb_invocation_and_rescue_rates_v2(
        *, report: MbppPlusV2BenchReport,
) -> dict[str, float]:
    """Compute the MLB-1 (reflexion-cycle invocation rate) +
    MLB-2 (reflexion rescue rate) sub-gates from a V2 bench
    report, mirroring the W100 RealWorldQA MLB pattern.

    MLB-1 = (# problems where B attempt 0 FAILed) / n_problems.
    MLB-2 = (# problems where B attempt 0 FAILed AND B PASSed
            on a later attempt) / (# MLB-1 numerator).
    """
    n = 0
    invoked = 0
    rescued = 0
    for s in report.per_seed:
        for i in range(len(s.per_problem_b_passed)):
            n += 1
            first_pass_idx = int(
                s.per_problem_b_first_pass_idx[i])
            b_passed = bool(s.per_problem_b_passed[i])
            attempt_0_failed = (first_pass_idx != 0)
            if attempt_0_failed:
                invoked += 1
                if b_passed:
                    rescued += 1
    inv_rate = (
        float(invoked / n) if n > 0 else 0.0)
    rescue_rate = (
        float(rescued / invoked) if invoked > 0 else 0.0)
    return {
        "n_problems_total": int(n),
        "n_b_invoked_reflexion": int(invoked),
        "n_b_rescued_via_reflexion": int(rescued),
        "mlb1_invocation_rate": float(round(inv_rate, 4)),
        "mlb2_rescue_rate": float(round(rescue_rate, 4)),
        "mlb1_floor": 0.33,
        "mlb2_floor": 0.33,
        "mlb1_passes": bool(inv_rate >= 0.33),
        "mlb2_passes": bool(rescue_rate >= 0.33),
    }


__all__ = [
    "W102_MBPP_PLUS_REFLEXION_BENCH_V2_SCHEMA_VERSION",
    "MbppPlusV2ArmCallCapsule",
    "MbppPlusV2ArmOutcomeCapsule",
    "MbppPlusV2SeedReport",
    "MbppPlusV2BenchReport",
    "MbppPlusV2BenchConfig",
    "select_mbpp_plus_v2_subset",
    "run_mbpp_plus_reflexion_bench_v2",
    "mlb_invocation_and_rescue_rates_v2",
]
