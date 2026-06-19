"""W101 / COO-9 — MBPP+ sequential-reflexion bench V1.

Wires the W101 MBPP+ loader + executor with the W88 / W89
sequential-reflexion B-pipeline that retired the same-budget
HumanEval-70B cap.  The mechanism is **byte-identical** to the
W88 / W90 reflexion shape (K=5 same-budget; first-pass-among-K
for A1; sequential reflexion with cumulative stderr history for
B); the only differences are:

* The corpus is EvalPlus's MBPP+ (378 problems; ~35× more hidden
  tests per problem than base MBPP).
* The executor runs the candidate against BOTH the base MBPP
  assertions AND the EvalPlus extra tests.
* The schema strings are W101 so the audit chains do not
  collide with the W90 / W91 base MBPP runs.

Three arms (mirrors W88 / W90 verbatim where applicable):

* ``A0`` — stock single-shot at T=0.0.
* ``A1`` — first-pass-among-K=5 self-consistency at T=0.7.
* ``B`` — sequential-reflexion-K=5 at T=0.7, each turn conditioned
  on the cumulative (candidate, executor_stderr) history.

Anti-cheat (carried forward from W88 / W90 verbatim):

* Same model on every arm.
* Same task subset per seed across arms.
* Same K=5 budget on A1 and B (byte-exact; no early-stop).
* No selective retries.
* Executor truth = pass on every base assertion AND every
  EvalPlus extra-test assertion.
* Per-call SHA-256 + per-seed Merkle + bench Merkle re-verifiable
  offline.

Honest scope (W101)
-------------------

* ``W101-L-MBPP-PLUS-REFLEXION-V1-NIM-DEPENDENT-CAP`` — V1 drives
  the bench through any ``LLMBackend``-shaped client; provider
  determinism is not assumed beyond temperature=0.
* ``W101-L-MBPP-PLUS-REFLEXION-V1-EXECUTOR-EXTRA-TESTS-CAP`` — the
  executor compares against the EvalPlus `plus_input` /
  `plus_output` parallel arrays as it received them in the
  corpus; encoding edge-cases (`None`, custom comparators) are
  handled by the loader but may degrade if EvalPlus re-releases
  the artifact with a new encoding shape.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import random
from typing import Any, Callable, Sequence

from .mbpp_plus_executor_v1 import (
    W101_MBPP_PLUS_EXECUTOR_KILL_AFTER_S,
    W101_MBPP_PLUS_EXECUTOR_TIMEOUT_S,
    MbppPlusExecutorResultV1,
    run_mbpp_plus_executor_v1,
)
from .mbpp_plus_loader_v1 import MbppPlusProblemV1
from .mbpp_reflexion_bench_v1 import (
    extract_candidate_code_v1,
)


W101_MBPP_PLUS_REFLEXION_BENCH_V1_SCHEMA_VERSION: str = (
    "coordpy.mbpp_plus_reflexion_bench_v1.v1")


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


# ---------------------------------------------------------------
# Prompts (verbatim port from MBPP reflexion bench V1)
# ---------------------------------------------------------------

_SYSTEM = (
    "You are an expert Python programmer.  When given a function "
    "description and a sample assertion, output ONLY the complete "
    "Python function (inside a ```python ... ``` code fence).  Do "
    "not include any prose before or after the code fence.")


def _initial_prompt(p: MbppPlusProblemV1) -> str:
    sample_assert = (
        p.assertion
        or (p.base_test_list[0] if p.base_test_list else ""))
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
        p: MbppPlusProblemV1,
        history: Sequence[tuple[str, MbppPlusExecutorResultV1]],
        attempt_idx: int,
) -> str:
    chunks: list[str] = []
    for i, (cand, exe) in enumerate(history):
        cand_trim = cand
        if len(cand_trim) > 1500:
            cand_trim = cand_trim[:1500] + "\n# ... (truncated)\n"
        if exe.passed:
            verdict = (
                f"PASSED all base ({exe.n_base_total}) "
                f"+ plus ({exe.n_plus_total}) assertions")
            stderr_excerpt = ""
        else:
            verdict = (
                f"FAILED (base {exe.n_base_passed}/"
                f"{exe.n_base_total}; plus "
                f"{exe.n_plus_passed}/{exe.n_plus_total}; "
                f"returncode={exe.returncode}"
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
        p.assertion
        or (p.base_test_list[0] if p.base_test_list else ""))
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
# Per-arm runners
# ---------------------------------------------------------------

_GenerateFn = Callable[[str, int, float], tuple[str, int]]


@dataclasses.dataclass(frozen=True)
class MbppPlusArmCallCapsuleV1:
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
            "kind": "w101_mbpp_plus_arm_call_capsule_v1",
            "capsule": self.to_dict(),
        })


@dataclasses.dataclass(frozen=True)
class MbppPlusArmOutcomeCapsuleV1:
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
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w101_mbpp_plus_arm_outcome_capsule_v1",
            "capsule": self.to_dict(),
        })


def _run_a0_single_shot(
        *, seed: int, p: MbppPlusProblemV1,
        gen: _GenerateFn, max_tokens: int,
        executor_kwargs: dict[str, Any],
) -> MbppPlusArmOutcomeCapsuleV1:
    prompt = _initial_prompt(p)
    text, wall = gen(prompt, max_tokens, 0.0)
    code = extract_candidate_code_v1(
        response_text=text, entry_point=p.entry_point)
    exe = run_mbpp_plus_executor_v1(
        problem=p, candidate_code=code, **executor_kwargs)
    call = MbppPlusArmCallCapsuleV1(
        schema=W101_MBPP_PLUS_REFLEXION_BENCH_V1_SCHEMA_VERSION,
        seed=int(seed), task_id=str(p.task_id),
        arm_id="A0", role="solver", call_idx=0,
        temperature=0.0,
        prompt_cid=hashlib.sha256(
            prompt.encode("utf-8")).hexdigest(),
        response_cid=hashlib.sha256(
            text.encode("utf-8")).hexdigest(),
        wall_ms=int(wall))
    return MbppPlusArmOutcomeCapsuleV1(
        schema=W101_MBPP_PLUS_REFLEXION_BENCH_V1_SCHEMA_VERSION,
        seed=int(seed), task_id=str(p.task_id),
        arm_id="A0",
        final_passed=bool(exe.passed),
        final_candidate_code_cid=str(exe.candidate_code_cid),
        n_model_calls=1,
        n_executor_calls=1,
        total_wall_ms=int(wall + exe.wall_ms),
        call_capsule_cids=(call.cid(),),
        executor_result_cids=(exe.cid(),),
        per_call_passed=(bool(exe.passed),))


def _run_a1_first_pass_among_K(
        *, seed: int, p: MbppPlusProblemV1, K: int,
        temperature: float, gen: _GenerateFn, max_tokens: int,
        executor_kwargs: dict[str, Any],
) -> MbppPlusArmOutcomeCapsuleV1:
    prompt = _initial_prompt(p)
    calls: list[MbppPlusArmCallCapsuleV1] = []
    exes: list[MbppPlusExecutorResultV1] = []
    total = 0
    chosen_passed = False
    chosen_code_cid = ""
    per_call_passed: list[bool] = []
    for k in range(int(K)):
        text, wall = gen(prompt, max_tokens, float(temperature))
        code = extract_candidate_code_v1(
            response_text=text, entry_point=p.entry_point)
        exe = run_mbpp_plus_executor_v1(
            problem=p, candidate_code=code,
            **executor_kwargs)
        calls.append(MbppPlusArmCallCapsuleV1(
            schema=(
                W101_MBPP_PLUS_REFLEXION_BENCH_V1_SCHEMA_VERSION),
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
    if not chosen_passed:
        chosen_code_cid = str(exes[0].candidate_code_cid)
    return MbppPlusArmOutcomeCapsuleV1(
        schema=W101_MBPP_PLUS_REFLEXION_BENCH_V1_SCHEMA_VERSION,
        seed=int(seed), task_id=str(p.task_id),
        arm_id="A1",
        final_passed=bool(chosen_passed),
        final_candidate_code_cid=str(chosen_code_cid),
        n_model_calls=int(K),
        n_executor_calls=int(K),
        total_wall_ms=int(total),
        call_capsule_cids=tuple(c.cid() for c in calls),
        executor_result_cids=tuple(e.cid() for e in exes),
        per_call_passed=tuple(per_call_passed))


def _run_b_sequential_reflexion(
        *, seed: int, p: MbppPlusProblemV1, K: int,
        temperature: float, gen: _GenerateFn, max_tokens: int,
        executor_kwargs: dict[str, Any],
) -> MbppPlusArmOutcomeCapsuleV1:
    history: list[tuple[str, MbppPlusExecutorResultV1]] = []
    calls: list[MbppPlusArmCallCapsuleV1] = []
    exes: list[MbppPlusExecutorResultV1] = []
    total = 0
    per_call_passed: list[bool] = []
    for k in range(int(K)):
        if k == 0:
            prompt = _initial_prompt(p)
        else:
            prompt = _reflexion_prompt(
                p, tuple(history), attempt_idx=int(k))
        text, wall = gen(prompt, max_tokens, float(temperature))
        code = extract_candidate_code_v1(
            response_text=text, entry_point=p.entry_point)
        exe = run_mbpp_plus_executor_v1(
            problem=p, candidate_code=code,
            **executor_kwargs)
        calls.append(MbppPlusArmCallCapsuleV1(
            schema=(
                W101_MBPP_PLUS_REFLEXION_BENCH_V1_SCHEMA_VERSION),
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
    return MbppPlusArmOutcomeCapsuleV1(
        schema=W101_MBPP_PLUS_REFLEXION_BENCH_V1_SCHEMA_VERSION,
        seed=int(seed), task_id=str(p.task_id),
        arm_id="B",
        final_passed=bool(final_passed),
        final_candidate_code_cid=str(final_code_cid),
        n_model_calls=int(K),
        n_executor_calls=int(K),
        total_wall_ms=int(total),
        call_capsule_cids=tuple(c.cid() for c in calls),
        executor_result_cids=tuple(e.cid() for e in exes),
        per_call_passed=tuple(per_call_passed))


# ---------------------------------------------------------------
# Report
# ---------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class MbppPlusSeedReportV1:
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
            "outcome_cids": list(self.outcome_cids),
            "seed_merkle_root": str(self.seed_merkle_root),
        }


@dataclasses.dataclass(frozen=True)
class MbppPlusBenchReportV1:
    schema: str
    model_id: str
    n_problems: int
    n_seeds: int
    K_multi_sample: int
    per_seed: tuple[MbppPlusSeedReportV1, ...]
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
            "kind": "w101_mbpp_plus_bench_report_v1",
            "report": self.to_dict(),
        })


# ---------------------------------------------------------------
# Subset + driver
# ---------------------------------------------------------------

def select_mbpp_plus_subset_v1(
        *, corpus: Sequence[MbppPlusProblemV1],
        n_problems: int, seed: int,
) -> tuple[MbppPlusProblemV1, ...]:
    """Deterministically select ``n_problems`` from the corpus.

    Note: the seed namespace deliberately diverges from W90's
    `select_mbpp_subset_v1` (90_001..90_005) because the EvalPlus
    corpus shape differs from MBPP-sanitized (378 vs 427
    problems with a different upstream ordering); the W101 cheap
    pilot uses dedicated seeds 101_001..101_005 to preserve W90 /
    W91 audit isolation."""
    rng = random.Random(int(seed))
    idxs = list(range(len(corpus)))
    rng.shuffle(idxs)
    chosen = idxs[: int(n_problems)]
    return tuple(corpus[i] for i in chosen)


@dataclasses.dataclass
class MbppPlusBenchConfigV1:
    schema: str = (
        W101_MBPP_PLUS_REFLEXION_BENCH_V1_SCHEMA_VERSION)
    n_problems: int = 30
    K_multi_sample: int = 5
    seeds: tuple[int, ...] = (101_001,)  # 1-seed cheap pilot
    sampling_temperature: float = 0.7
    max_tokens_per_call: int = 768
    executor_timeout_s: float = (
        W101_MBPP_PLUS_EXECUTOR_TIMEOUT_S)
    executor_kill_after_s: float = (
        W101_MBPP_PLUS_EXECUTOR_KILL_AFTER_S)
    executor_mode: str = "base_and_plus"


def run_mbpp_plus_reflexion_bench_v1(
        *,
        gen: _GenerateFn,
        model_id: str,
        corpus: Sequence[MbppPlusProblemV1],
        config: MbppPlusBenchConfigV1 | None = None,
        on_problem_start: (
            Callable[[int, int, str], None] | None) = None,
) -> MbppPlusBenchReportV1:
    """Run A0 + A1 + B (sequential reflexion) on a deterministic
    subset for each seed.  Returns a content-addressed bench
    report."""
    cfg = config or MbppPlusBenchConfigV1()
    executor_kwargs = {
        "timeout_s": float(cfg.executor_timeout_s),
        "kill_after_s": float(cfg.executor_kill_after_s),
        "mode": str(cfg.executor_mode),
    }
    per_seed: list[MbppPlusSeedReportV1] = []
    all_outcome_cids: list[str] = []
    for seed in cfg.seeds:
        subset = select_mbpp_plus_subset_v1(
            corpus=corpus, n_problems=int(cfg.n_problems),
            seed=int(seed))
        a0_outs: list[MbppPlusArmOutcomeCapsuleV1] = []
        a1_outs: list[MbppPlusArmOutcomeCapsuleV1] = []
        b_outs: list[MbppPlusArmOutcomeCapsuleV1] = []
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
            "kind": "w101_mbpp_plus_seed_merkle_root",
            "seed": int(seed),
            "outcome_cids": list(outcome_cids),
        })
        per_seed.append(MbppPlusSeedReportV1(
            schema=(
                W101_MBPP_PLUS_REFLEXION_BENCH_V1_SCHEMA_VERSION),
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
        "kind": "w101_mbpp_plus_bench_merkle_root",
        "model_id": str(model_id),
        "outcome_cids": all_outcome_cids,
        "seeds": list(cfg.seeds),
    })
    return MbppPlusBenchReportV1(
        schema=W101_MBPP_PLUS_REFLEXION_BENCH_V1_SCHEMA_VERSION,
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


__all__ = [
    "W101_MBPP_PLUS_REFLEXION_BENCH_V1_SCHEMA_VERSION",
    "MbppPlusArmCallCapsuleV1",
    "MbppPlusArmOutcomeCapsuleV1",
    "MbppPlusSeedReportV1",
    "MbppPlusBenchReportV1",
    "MbppPlusBenchConfigV1",
    "select_mbpp_plus_subset_v1",
    "run_mbpp_plus_reflexion_bench_v1",
]
