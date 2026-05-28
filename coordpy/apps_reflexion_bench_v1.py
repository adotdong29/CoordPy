"""W109 / COO-9 — APPS (call-based / functional) sequential-reflexion bench.

The **contamination-control counterpart** to the W108 LiveCodeBench bench.
APPS (Hendrycks et al., 2021) is contamination-EXPOSED (2021 vintage, almost
certainly inside the Llama-3.x training corpus → C7 = C), whereas
LiveCodeBench 2025 is contamination-RESISTANT (C7 = A).  W108's first
contamination-resistant test of the W89 mechanism FAILed (B − A1 = −3.33 pp).
W109 asks the direct control question: does the SAME mechanism, under the SAME
same-budget K=5 contract, RECOVER on a contamination-EXPOSED 2021 benchmark?

The mechanism is **byte-identical in shape** to the W88/W89 HumanEval bench,
the W102 HumanEval+ bench, and the W108 LiveCodeBench bench
(``livecodebench_reflexion_bench_v1``).  Only three things differ from W108:
the corpus (``apps_loader_v1.AppsFunctionalProblemV1``), the executor
(``apps_executor_v1`` — JSON-arg-list decode + 1-element-output-wrapper
tolerance, NOT the LiveCodeBench newline decoder), and the prompt (the model
must implement the call-based ``fn_name`` entry).  Everything load-bearing —
A0/A1/B definitions, the K=5 byte-exact budget, first-pass-among-K for A1,
sequential reflexion with cumulative (candidate, executor-stderr) history for
B, per-call CIDs + per-seed/bench Merkle — is the same.

Three arms (mirrors W88/W89/W102/W108 verbatim):

* ``A0`` — stock single-shot at T=0.0.
* ``A1`` — first-pass-among-K=5 self-consistency at T=0.7.
* ``B``  — sequential-reflexion-K=5 at T=0.7, each turn conditioned on the
  cumulative (candidate, executor_stderr) history.

Anti-cheat (carried forward verbatim):

* Same model on every arm.
* Same K=5 budget on A1 and B (byte-exact; no early-stop).
* No selective retries.
* Executor truth = the call-based functional tests; exit 0 iff every case
  matches.  NO LLM-as-judge.
* Per-call CIDs + per-seed Merkle + bench Merkle re-verifiable offline.
* Slice is deterministic + OUTCOME-BLIND (APPS has no in-repo sidecar; cannot
  cherry-pick rescues we have not yet measured).

Honest scope (W109):

* ``W109-L-APPS-REFLEXION-BENCH-V1-PUBLIC-TESTS-CAP`` — the bench scores
  against the first ``max_tests_per_problem`` call-based tests the loader
  pins (default 25), in upstream order, to bound the embedded-harness program
  size.  The cap applies IDENTICALLY to A0/A1/B, so the B − A1 contrast (the
  W109 control quantity) is fair; only the ABSOLUTE pass rates are upper
  bounds.  Confirm the actual per-problem test counts at fetch time.
* APPS results are contamination-EXPOSED (2021 vintage) and are
  CONTROL/BACKUP evidence only — never the publication-grade time-anchored
  claim surface (``W108-L-APPS-CONTAMINATION-EXPOSED-2021-VINTAGE-CAP``).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from typing import Any, Callable, Sequence

from .apps_executor_v1 import (
    W108_APPS_EXECUTOR_V1_KILL_AFTER_S,
    W108_APPS_EXECUTOR_V1_TIMEOUT_S,
    AppsExecutorResultV1,
    run_apps_executor_v1,
)
from .apps_loader_v1 import AppsFunctionalProblemV1

W109_APPS_REFLEXION_BENCH_V1_SCHEMA_VERSION: str = (
    "coordpy.apps_reflexion_bench_v1.v1")

W109_APPS_DEFAULT_MAX_TESTS_PER_PROBLEM: int = 25


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def extract_candidate_code_v1(*, response_text: str,
                              starter_code: str) -> str:
    """Pull the candidate Python from a model response (verbatim from the
    LiveCodeBench bench): the LAST ```python ...``` fence, else any fence,
    else the raw text."""
    text = str(response_text or "")
    fences = _FENCE_RE.findall(text)
    if fences:
        code = fences[-1].strip()
    else:
        code = text.strip()
    return code


def _signature_hint(problem: AppsFunctionalProblemV1) -> str:
    starter = (problem.starter_code or "").strip()
    if starter:
        return f"Complete this starter code:\n```python\n{starter}\n```"
    return (
        f"Define the entry function exactly as `def {problem.fn_name}(...)` "
        "(or a `class Solution` with a method named "
        f"`{problem.fn_name}`) taking the positional arguments described "
        "above.")


def _initial_prompt(problem: AppsFunctionalProblemV1) -> str:
    return (
        "You are an expert competitive programmer. Solve the following "
        "problem by implementing the required Python function.\n\n"
        "Output ONLY the complete solution inside a single "
        "```python ... ``` code fence. Do not include explanations, tests, "
        "or input parsing — the function is called directly with decoded "
        f"positional arguments and its RETURN VALUE is checked. The entry "
        f"function is named `{problem.fn_name}`.\n\n"
        f"Problem:\n{problem.question}\n\n"
        f"{_signature_hint(problem)}\n\n"
        "Your complete solution:")


def _reflexion_prompt(
        problem: AppsFunctionalProblemV1,
        history: Sequence[tuple[str, AppsExecutorResultV1]],
        attempt_idx: int,
) -> str:
    chunks: list[str] = []
    for i, (cand, exe) in enumerate(history):
        cand_trim = cand if len(cand) <= 1500 else (
            cand[:1500] + "\n# ... (truncated)\n")
        if exe.passed:
            verdict = "PASSED the functional tests"
            stderr_excerpt = ""
        else:
            verdict = (
                f"FAILED (returncode={exe.returncode}"
                + (", TIMED OUT" if exe.timed_out else "") + ")")
            stderr_text = exe.stderr_tail.strip()
            stderr_excerpt = (
                f"\nExecutor stderr (tail):\n{stderr_text}"
                if stderr_text else "")
        chunks.append(
            f"--- Attempt {i+1} ({verdict}) ---\n"
            f"```python\n{cand_trim}\n```{stderr_excerpt}")
    return (
        "You are an expert competitive programmer on a reflective debugging "
        f"loop. You are on attempt {attempt_idx + 1} out of 5. Below are your "
        "previous attempts and the Python executor's verdict + stderr tail "
        "for each. Diagnose the bug class in each failing attempt and produce "
        f"a NEW corrected complete solution (entry `{problem.fn_name}`). Do "
        "not repeat a previous attempt verbatim.\n\n"
        f"Problem:\n{problem.question}\n\n"
        f"{_signature_hint(problem)}\n\n"
        f"{chr(10).join(chunks)}\n\n"
        "Provide ONLY the corrected complete solution in a "
        "```python ... ``` fence:")


_GenerateFn = Callable[[str, int, float], tuple[str, int]]


@dataclasses.dataclass(frozen=True)
class AppsArmOutcomeV1:
    schema: str
    seed: int
    problem_id: str
    arm_id: str
    final_passed: bool
    final_candidate_code_cid: str
    n_model_calls: int
    n_executor_calls: int
    total_wall_ms: int
    executor_result_cids: tuple[str, ...]
    per_call_passed: tuple[bool, ...]
    first_pass_attempt_idx: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema), "seed": int(self.seed),
            "problem_id": str(self.problem_id), "arm_id": str(self.arm_id),
            "final_passed": bool(self.final_passed),
            "final_candidate_code_cid": str(self.final_candidate_code_cid),
            "n_model_calls": int(self.n_model_calls),
            "n_executor_calls": int(self.n_executor_calls),
            "total_wall_ms": int(self.total_wall_ms),
            "executor_result_cids": list(self.executor_result_cids),
            "per_call_passed": list(self.per_call_passed),
            "first_pass_attempt_idx": int(self.first_pass_attempt_idx),
        }

    def cid(self) -> str:
        return _sha256_hex({"kind": "w109_apps_arm_outcome_v1",
                            "capsule": self.to_dict()})


def _tests_for(problem: AppsFunctionalProblemV1,
               *, max_tests: int) -> list[dict]:
    """The call-based tests in the executor's ``{"args","output"}`` shape,
    capped at ``max_tests`` (upstream order) to bound the embedded harness."""
    out = [{"args": t.args_repr, "output": t.output_repr}
           for t in problem.tests]
    if max_tests is not None and max_tests > 0:
        out = out[:int(max_tests)]
    return out


def _problem_sort_key(p: AppsFunctionalProblemV1) -> tuple[int, str]:
    """Stable order: numeric problem_id first (APPS ids are numeric strings),
    string fallback for safety."""
    pid = str(p.problem_id)
    try:
        return (int(pid), pid)
    except (TypeError, ValueError):
        return (1 << 62, pid)


def _largest_remainder(counts: dict[str, int], total: int) -> dict[str, int]:
    """Hamilton / largest-remainder apportionment; deterministic tie-break by
    key (verbatim from the LiveCodeBench slice selector)."""
    s = sum(counts.values()) or 1
    raw = {k: (total * v / s) for k, v in counts.items()}
    base = {k: int(x) for k, x in raw.items()}
    rem = int(total) - sum(base.values())
    order = sorted(counts, key=lambda k: (-(raw[k] - base[k]), k))
    for k in order[:max(0, rem)]:
        base[k] += 1
    return base


def select_apps_functional_slice_v1(
        subset: Sequence[AppsFunctionalProblemV1],
        *, n_problems: int = 30,
        difficulty_order: tuple[str, ...] = (
            "introductory", "interview", "competition"),
) -> tuple[AppsFunctionalProblemV1, ...]:
    """Deterministic, pre-committed, OUTCOME-BLIND cheap-pilot slice — the
    direct analogue of ``select_livecodebench_functional_slice_v1``.

    APPS has NO in-repo sidecar (just like LiveCodeBench), so the slice MUST
    be a purely structural, outcome-blind, reproducible selection:

    * difficulty-stratified, with per-difficulty targets proportional to the
      full call-based mix (largest-remainder) — guarantees the slice carries
      interview/competition problems where reflexion is load-bearing and is
      not an all-introductory slice that would saturate A1 (the G2 concern);
    * within each difficulty, take in ``problem_id`` order;
    * spill short buckets to the remaining problems in global ``problem_id``
      order;
    * emit the final slice ordered by ``problem_id`` so the slice CID is
      stable and independent of the difficulty pass.

    No outcome data is consulted (there is none) — anti-cheat by
    construction."""
    by_diff: dict[str, list[AppsFunctionalProblemV1]] = {}
    for p in subset:
        by_diff.setdefault(str(p.difficulty), []).append(p)
    for k in by_diff:
        by_diff[k].sort(key=_problem_sort_key)
    counts = {k: len(v) for k, v in by_diff.items()}
    n = min(int(n_problems), len(subset))
    targets = _largest_remainder(counts, n)
    chosen: list[AppsFunctionalProblemV1] = []
    diff_pass = list(difficulty_order) + sorted(
        k for k in by_diff if k not in difficulty_order)
    for d in diff_pass:
        chosen.extend(by_diff.get(d, [])[:targets.get(d, 0)])
    if len(chosen) < n:
        chosen_ids = {p.problem_id for p in chosen}
        rest = sorted(
            (p for p in subset if p.problem_id not in chosen_ids),
            key=_problem_sort_key)
        chosen.extend(rest[:n - len(chosen)])
    chosen = chosen[:n]
    chosen.sort(key=_problem_sort_key)
    return tuple(chosen)


def _run_a0(*, seed, p, gen, max_tokens, ek, max_tests) -> AppsArmOutcomeV1:
    text, wall = gen(_initial_prompt(p), max_tokens, 0.0)
    code = extract_candidate_code_v1(
        response_text=text, starter_code=p.starter_code)
    exe = run_apps_executor_v1(
        problem_id=p.problem_id, func_name=p.fn_name,
        tests=_tests_for(p, max_tests=max_tests), candidate_code=code, **ek)
    return AppsArmOutcomeV1(
        schema=W109_APPS_REFLEXION_BENCH_V1_SCHEMA_VERSION,
        seed=int(seed), problem_id=str(p.problem_id), arm_id="A0",
        final_passed=bool(exe.passed),
        final_candidate_code_cid=str(exe.candidate_code_cid),
        n_model_calls=1, n_executor_calls=1,
        total_wall_ms=int(wall + exe.wall_ms),
        executor_result_cids=(exe.cid(),),
        per_call_passed=(bool(exe.passed),),
        first_pass_attempt_idx=(0 if exe.passed else -1))


def _run_a1(*, seed, p, K, temperature, gen, max_tokens, ek,
            max_tests) -> AppsArmOutcomeV1:
    prompt = _initial_prompt(p)
    exes: list[AppsExecutorResultV1] = []
    total = 0
    chosen_passed = False
    chosen_cid = ""
    chosen_idx = -1
    per_call_passed: list[bool] = []
    for k in range(int(K)):
        text, wall = gen(prompt, max_tokens, float(temperature))
        code = extract_candidate_code_v1(
            response_text=text, starter_code=p.starter_code)
        exe = run_apps_executor_v1(
            problem_id=p.problem_id, func_name=p.fn_name,
            tests=_tests_for(p, max_tests=max_tests), candidate_code=code,
            **ek)
        exes.append(exe)
        per_call_passed.append(bool(exe.passed))
        total += int(wall) + int(exe.wall_ms)
        if exe.passed and not chosen_passed:
            chosen_passed = True
            chosen_cid = str(exe.candidate_code_cid)
            chosen_idx = int(k)
    if not chosen_passed:
        chosen_cid = str(exes[0].candidate_code_cid)
    return AppsArmOutcomeV1(
        schema=W109_APPS_REFLEXION_BENCH_V1_SCHEMA_VERSION,
        seed=int(seed), problem_id=str(p.problem_id), arm_id="A1",
        final_passed=bool(chosen_passed),
        final_candidate_code_cid=str(chosen_cid),
        n_model_calls=int(K), n_executor_calls=int(K),
        total_wall_ms=int(total),
        executor_result_cids=tuple(e.cid() for e in exes),
        per_call_passed=tuple(per_call_passed),
        first_pass_attempt_idx=int(chosen_idx))


def _run_b(*, seed, p, K, temperature, gen, max_tokens, ek,
           max_tests) -> AppsArmOutcomeV1:
    history: list[tuple[str, AppsExecutorResultV1]] = []
    exes: list[AppsExecutorResultV1] = []
    total = 0
    per_call_passed: list[bool] = []
    first_pass_idx = -1
    for k in range(int(K)):
        prompt = (_initial_prompt(p) if k == 0
                  else _reflexion_prompt(p, tuple(history), attempt_idx=int(k)))
        text, wall = gen(prompt, max_tokens, float(temperature))
        code = extract_candidate_code_v1(
            response_text=text, starter_code=p.starter_code)
        exe = run_apps_executor_v1(
            problem_id=p.problem_id, func_name=p.fn_name,
            tests=_tests_for(p, max_tests=max_tests), candidate_code=code,
            **ek)
        exes.append(exe)
        per_call_passed.append(bool(exe.passed))
        history.append((code, exe))
        total += int(wall) + int(exe.wall_ms)
        if exe.passed and first_pass_idx == -1:
            first_pass_idx = int(k)
    final_passed = (first_pass_idx >= 0)
    if final_passed:
        final_cid = str(exes[first_pass_idx].candidate_code_cid)
    else:
        final_cid = sorted(str(e.candidate_code_cid) for e in exes)[0]
    return AppsArmOutcomeV1(
        schema=W109_APPS_REFLEXION_BENCH_V1_SCHEMA_VERSION,
        seed=int(seed), problem_id=str(p.problem_id), arm_id="B",
        final_passed=bool(final_passed),
        final_candidate_code_cid=str(final_cid),
        n_model_calls=int(K), n_executor_calls=int(K),
        total_wall_ms=int(total),
        executor_result_cids=tuple(e.cid() for e in exes),
        per_call_passed=tuple(per_call_passed),
        first_pass_attempt_idx=int(first_pass_idx))


@dataclasses.dataclass(frozen=True)
class AppsSeedReportV1:
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
    problem_ids: tuple[str, ...]
    seed_merkle_root: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema), "seed": int(self.seed),
            "n_problems": int(self.n_problems),
            "a0_pass_at_1": float(round(self.a0_pass_at_1, 6)),
            "a1_pass_at_1": float(round(self.a1_pass_at_1, 6)),
            "b_pass_at_1": float(round(self.b_pass_at_1, 6)),
            "per_problem_a0_passed": list(self.per_problem_a0_passed),
            "per_problem_a1_passed": list(self.per_problem_a1_passed),
            "per_problem_b_passed": list(self.per_problem_b_passed),
            "per_problem_b_first_pass_idx": list(
                self.per_problem_b_first_pass_idx),
            "problem_ids": list(self.problem_ids),
            "seed_merkle_root": str(self.seed_merkle_root),
        }


@dataclasses.dataclass(frozen=True)
class AppsBenchReportV1:
    schema: str
    model_id: str
    n_problems: int
    n_seeds: int
    K_multi_sample: int
    per_seed: tuple[AppsSeedReportV1, ...]
    a0_mean_pass_at_1: float
    a1_mean_pass_at_1: float
    b_mean_pass_at_1: float
    b_mean_minus_a1_mean_pp: float
    bench_merkle_root: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema), "model_id": str(self.model_id),
            "n_problems": int(self.n_problems), "n_seeds": int(self.n_seeds),
            "K_multi_sample": int(self.K_multi_sample),
            "per_seed": [s.to_dict() for s in self.per_seed],
            "a0_mean_pass_at_1": float(round(self.a0_mean_pass_at_1, 6)),
            "a1_mean_pass_at_1": float(round(self.a1_mean_pass_at_1, 6)),
            "b_mean_pass_at_1": float(round(self.b_mean_pass_at_1, 6)),
            "b_mean_minus_a1_mean_pp": float(round(
                self.b_mean_minus_a1_mean_pp, 4)),
            "bench_merkle_root": str(self.bench_merkle_root),
        }

    def cid(self) -> str:
        return _sha256_hex({"kind": "w109_apps_bench_report_v1",
                            "report": self.to_dict()})


@dataclasses.dataclass
class AppsBenchConfigV1:
    schema: str = W109_APPS_REFLEXION_BENCH_V1_SCHEMA_VERSION
    K_multi_sample: int = 5
    seeds: tuple[int, ...] = (109_001,)
    sampling_temperature: float = 0.7
    max_tokens_per_call: int = 1024
    max_tests_per_problem: int = W109_APPS_DEFAULT_MAX_TESTS_PER_PROBLEM
    executor_timeout_s: float = W108_APPS_EXECUTOR_V1_TIMEOUT_S
    executor_kill_after_s: float = W108_APPS_EXECUTOR_V1_KILL_AFTER_S


def run_apps_reflexion_bench_v1(
        *,
        gen: _GenerateFn,
        model_id: str,
        subset: Sequence[AppsFunctionalProblemV1],
        config: AppsBenchConfigV1 | None = None,
        on_problem_start: Callable[[int, int, str], None] | None = None,
) -> AppsBenchReportV1:
    """Run A0 + A1 + B (sequential reflexion) on a PRE-SELECTED, ordered APPS
    call-based subset for each seed.  The subset (the deterministic
    difficulty-stratified slice) is consumed verbatim — no internal
    reshuffle, exactly like the LiveCodeBench bench."""
    cfg = config or AppsBenchConfigV1()
    ek = {"timeout_s": float(cfg.executor_timeout_s),
          "kill_after_s": float(cfg.executor_kill_after_s)}
    mt = int(cfg.max_tests_per_problem)
    per_seed: list[AppsSeedReportV1] = []
    all_cids: list[str] = []
    for seed in cfg.seeds:
        a0o: list[AppsArmOutcomeV1] = []
        a1o: list[AppsArmOutcomeV1] = []
        bo: list[AppsArmOutcomeV1] = []
        for p_idx, p in enumerate(subset):
            if on_problem_start is not None:
                on_problem_start(int(seed), int(p_idx), str(p.problem_id))
            a0o.append(_run_a0(seed=seed, p=p, gen=gen,
                               max_tokens=cfg.max_tokens_per_call, ek=ek,
                               max_tests=mt))
            a1o.append(_run_a1(seed=seed, p=p, K=cfg.K_multi_sample,
                               temperature=cfg.sampling_temperature, gen=gen,
                               max_tokens=cfg.max_tokens_per_call, ek=ek,
                               max_tests=mt))
            bo.append(_run_b(seed=seed, p=p, K=cfg.K_multi_sample,
                             temperature=cfg.sampling_temperature, gen=gen,
                             max_tokens=cfg.max_tokens_per_call, ek=ek,
                             max_tests=mt))
        n = float(len(a0o)) or 1.0
        a0_acc = sum(1 for o in a0o if o.final_passed) / n
        a1_acc = sum(1 for o in a1o if o.final_passed) / n
        b_acc = sum(1 for o in bo if o.final_passed) / n
        cids = tuple([o.cid() for o in a0o] + [o.cid() for o in a1o]
                     + [o.cid() for o in bo])
        seed_merkle = _sha256_hex({"kind": "w109_apps_seed_merkle_v1",
                                   "seed": int(seed), "cids": list(cids)})
        per_seed.append(AppsSeedReportV1(
            schema=W109_APPS_REFLEXION_BENCH_V1_SCHEMA_VERSION,
            seed=int(seed), n_problems=int(len(a0o)),
            a0_pass_at_1=float(a0_acc), a1_pass_at_1=float(a1_acc),
            b_pass_at_1=float(b_acc),
            per_problem_a0_passed=tuple(bool(o.final_passed) for o in a0o),
            per_problem_a1_passed=tuple(bool(o.final_passed) for o in a1o),
            per_problem_b_passed=tuple(bool(o.final_passed) for o in bo),
            per_problem_b_first_pass_idx=tuple(
                int(o.first_pass_attempt_idx) for o in bo),
            problem_ids=tuple(str(o.problem_id) for o in a0o),
            seed_merkle_root=str(seed_merkle)))
        all_cids.extend(cids)
    ns = float(len(per_seed)) or 1.0
    a0m = sum(s.a0_pass_at_1 for s in per_seed) / ns
    a1m = sum(s.a1_pass_at_1 for s in per_seed) / ns
    bm = sum(s.b_pass_at_1 for s in per_seed) / ns
    bench_merkle = _sha256_hex({"kind": "w109_apps_bench_merkle_v1",
                                "model_id": str(model_id),
                                "cids": all_cids, "seeds": list(cfg.seeds)})
    return AppsBenchReportV1(
        schema=W109_APPS_REFLEXION_BENCH_V1_SCHEMA_VERSION,
        model_id=str(model_id), n_problems=int(len(subset)),
        n_seeds=int(len(cfg.seeds)), K_multi_sample=int(cfg.K_multi_sample),
        per_seed=tuple(per_seed),
        a0_mean_pass_at_1=float(a0m), a1_mean_pass_at_1=float(a1m),
        b_mean_pass_at_1=float(bm),
        b_mean_minus_a1_mean_pp=float((bm - a1m) * 100.0),
        bench_merkle_root=str(bench_merkle))


__all__ = [
    "W109_APPS_REFLEXION_BENCH_V1_SCHEMA_VERSION",
    "W109_APPS_DEFAULT_MAX_TESTS_PER_PROBLEM",
    "extract_candidate_code_v1",
    "select_apps_functional_slice_v1",
    "AppsArmOutcomeV1",
    "AppsSeedReportV1",
    "AppsBenchReportV1",
    "AppsBenchConfigV1",
    "run_apps_reflexion_bench_v1",
]
