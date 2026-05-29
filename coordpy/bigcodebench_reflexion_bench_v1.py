"""W110 / COO-9 — BigCodeBench sequential-reflexion bench (SECOND resistant).

The SECOND contamination-RESISTANT counterpart to the W108 LiveCodeBench bench.
BigCodeBench (BigCode, 2024-06) is contamination-RESISTANT (released after the
≈2024-01 Llama-3.x cutoff; novel compositional tasks + post-cutoff hidden
``unittest`` oracles — C7 = A-grade by release-date anchoring, the honest
caveat in ``docs/RUNBOOK_W110.md`` § 2.3). W108's first contamination-resistant
test FAILed (B − A1 = −3.33 pp); W110 asks the verdict-changing question: is
that FAIL LiveCodeBench-SPECIFIC, or GENERAL to contamination-resistant code?

The mechanism is **byte-identical in shape** to the W88/W89 HumanEval bench,
the W102 HumanEval+ bench, the W108 LiveCodeBench bench, and the W109 APPS
bench (``apps_reflexion_bench_v1``). Only three things differ from W109: the
corpus (``bigcodebench_loader_v1.BigCodeBenchProblemV1``), the executor
(``bigcodebench_executor_v1`` — runs the row's ``unittest`` oracle, NOT a
call-based loop), and the prompt (the model implements the ``entry_point``
function from the ``complete_prompt`` spec). Everything load-bearing — A0/A1/B
definitions, the K=5 byte-exact budget, first-pass-among-K for A1, sequential
reflexion with cumulative (candidate, executor-stderr) history for B, per-call
CIDs + per-seed/bench Merkle — is the same.

Three arms (mirrors W88/W89/W102/W108/W109 verbatim):

* ``A0`` — stock single-shot at T=0.0.
* ``A1`` — first-pass-among-K=5 self-consistency at T=0.7.
* ``B``  — sequential-reflexion-K=5 at T=0.7, each turn conditioned on the
  cumulative (candidate, executor_stderr) history.

Anti-cheat (carried forward verbatim): same model on every arm; same K=5
budget on A1 and B (byte-exact; no early-stop); no selective retries; executor
truth = the ``unittest`` oracle, exit 0 iff the suite is successful, NO
LLM-as-judge; per-call CIDs + per-seed Merkle + bench Merkle re-verifiable
offline; slice is deterministic + OUTCOME-BLIND (selected from the gold-green
pool — an executor-environment property, never a model outcome — so it cannot
cherry-pick model rescues).

Honest scope (W110):

* ``W110-L-BIGCODEBENCH-GOLD-GREEN-SUBSET-ONLY-CAP`` — the bench is run on the
  PRE-FILTERED gold-green subset (the loader's golds that pass in the run
  environment); absolute pass rates are conditioned on that subset, but the
  B − A1 contrast (the W110 quantity) is fair because A0/A1/B see the identical
  subset + oracle.
* ``W110-L-BIGCODEBENCH-RELEASE-DATE-RESISTANCE-NOT-CONTEST-DATE-CAP`` — the
  resistance is release-date + novel-composition, not strict contest-date
  anchoring (see the loader).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from typing import Any, Callable, Sequence

from .bigcodebench_executor_v1 import (
    W110_BIGCODEBENCH_EXECUTOR_V1_KILL_AFTER_S,
    W110_BIGCODEBENCH_EXECUTOR_V1_TIMEOUT_S,
    BigCodeBenchExecutorResultV1,
    run_bigcodebench_executor_v1,
)
from .bigcodebench_loader_v1 import BigCodeBenchProblemV1

W110_BIGCODEBENCH_REFLEXION_BENCH_V1_SCHEMA_VERSION: str = (
    "coordpy.bigcodebench_reflexion_bench_v1.v1")


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def extract_candidate_code_v1(*, response_text: str) -> str:
    """Pull the candidate Python from a model response (verbatim from the
    LiveCodeBench/APPS benches): the LAST ```python ...``` fence, else any
    fence, else the raw text."""
    text = str(response_text or "")
    fences = _FENCE_RE.findall(text)
    if fences:
        return fences[-1].strip()
    return text.strip()


def _initial_prompt(problem: BigCodeBenchProblemV1) -> str:
    return (
        "You are an expert Python programmer. Implement the function "
        "described by the following signature and docstring.\n\n"
        "Output ONLY the COMPLETE solution — all required imports plus the "
        f"full `def {problem.entry_point}(...)` implementation — inside a "
        "single ```python ... ``` code fence. Do NOT include explanations, "
        "example usage, or tests. A hidden unittest suite calls "
        f"`{problem.entry_point}` directly.\n\n"
        f"```python\n{problem.complete_prompt}\n```\n\n"
        "Your complete solution:")


def _reflexion_prompt(
        problem: BigCodeBenchProblemV1,
        history: Sequence[tuple[str, BigCodeBenchExecutorResultV1]],
        attempt_idx: int,
) -> str:
    chunks: list[str] = []
    for i, (cand, exe) in enumerate(history):
        cand_trim = cand if len(cand) <= 1500 else (
            cand[:1500] + "\n# ... (truncated)\n")
        if exe.passed:
            verdict = "PASSED the unittest suite"
            stderr_excerpt = ""
        else:
            verdict = (
                f"FAILED (returncode={exe.returncode}"
                + (", TIMED OUT" if exe.timed_out else "") + ")")
            stderr_text = exe.stderr_tail.strip()
            stderr_excerpt = (
                f"\nExecutor stderr (unittest tail):\n{stderr_text}"
                if stderr_text else "")
        chunks.append(
            f"--- Attempt {i+1} ({verdict}) ---\n"
            f"```python\n{cand_trim}\n```{stderr_excerpt}")
    return (
        "You are an expert Python programmer on a reflective debugging loop. "
        f"You are on attempt {attempt_idx + 1} out of 5. Below are your "
        "previous attempts and the Python executor's verdict + unittest stderr "
        "tail for each. Diagnose the bug class in each failing attempt and "
        f"produce a NEW corrected COMPLETE solution (entry "
        f"`{problem.entry_point}`). Do not repeat a previous attempt "
        "verbatim.\n\n"
        f"```python\n{problem.complete_prompt}\n```\n\n"
        f"{chr(10).join(chunks)}\n\n"
        "Provide ONLY the corrected complete solution in a "
        "```python ... ``` fence:")


_GenerateFn = Callable[[str, int, float], tuple[str, int]]


@dataclasses.dataclass(frozen=True)
class BigCodeBenchArmOutcomeV1:
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
        return _sha256_hex({"kind": "w110_bigcodebench_arm_outcome_v1",
                            "capsule": self.to_dict()})


def _problem_sort_key(p: BigCodeBenchProblemV1) -> tuple[int, str]:
    """Stable order by the trailing integer of the task_id
    (``BigCodeBench/<n>``), string fallback for safety."""
    pid = str(p.task_id)
    tail = pid.rsplit("/", 1)[-1]
    try:
        return (int(tail), pid)
    except (TypeError, ValueError):
        return (1 << 62, pid)


def _libs_bucket(p: BigCodeBenchProblemV1) -> str:
    n = p.n_libs()
    if n <= 1:
        return "libs0_1"
    if n == 2:
        return "libs2"
    return "libs3plus"


def _largest_remainder(counts: dict[str, int], total: int) -> dict[str, int]:
    """Hamilton / largest-remainder apportionment; deterministic tie-break by
    key (verbatim from the APPS/LiveCodeBench slice selectors)."""
    s = sum(counts.values()) or 1
    raw = {k: (total * v / s) for k, v in counts.items()}
    base = {k: int(x) for k, x in raw.items()}
    rem = int(total) - sum(base.values())
    order = sorted(counts, key=lambda k: (-(raw[k] - base[k]), k))
    for k in order[:max(0, rem)]:
        base[k] += 1
    return base


def select_bigcodebench_slice_v1(
        subset: Sequence[BigCodeBenchProblemV1],
        *, n_problems: int = 30,
        bucket_order: tuple[str, ...] = ("libs0_1", "libs2", "libs3plus"),
) -> tuple[BigCodeBenchProblemV1, ...]:
    """Deterministic, pre-committed, OUTCOME-BLIND cheap-pilot slice — the
    direct analogue of ``select_apps_functional_slice_v1``.

    BigCodeBench has no ``difficulty`` field; the slice is stratified by an
    ``n_libs`` bucket (0-1 / 2 / 3+) so it carries both simple and complex
    library-composition problems (a slice of only single-lib tasks could
    saturate A1 — the G2 concern). Within each bucket, take in ``task_id``
    order; spill short buckets to the remaining problems in global ``task_id``
    order; emit the final slice ordered by ``task_id`` so the slice CID is
    stable and independent of the bucket pass. No outcome data is consulted —
    anti-cheat by construction (the caller passes the gold-green pool, an
    executor-environment property, not a model outcome)."""
    by_bucket: dict[str, list[BigCodeBenchProblemV1]] = {}
    for p in subset:
        by_bucket.setdefault(_libs_bucket(p), []).append(p)
    for k in by_bucket:
        by_bucket[k].sort(key=_problem_sort_key)
    counts = {k: len(v) for k, v in by_bucket.items()}
    n = min(int(n_problems), len(subset))
    targets = _largest_remainder(counts, n)
    chosen: list[BigCodeBenchProblemV1] = []
    pass_order = list(bucket_order) + sorted(
        k for k in by_bucket if k not in bucket_order)
    for b in pass_order:
        chosen.extend(by_bucket.get(b, [])[:targets.get(b, 0)])
    if len(chosen) < n:
        chosen_ids = {p.task_id for p in chosen}
        rest = sorted(
            (p for p in subset if p.task_id not in chosen_ids),
            key=_problem_sort_key)
        chosen.extend(rest[:n - len(chosen)])
    chosen = chosen[:n]
    chosen.sort(key=_problem_sort_key)
    return tuple(chosen)


def _run_a0(*, seed, p, gen, max_tokens, ek) -> BigCodeBenchArmOutcomeV1:
    text, wall = gen(_initial_prompt(p), max_tokens, 0.0)
    code = extract_candidate_code_v1(response_text=text)
    exe = run_bigcodebench_executor_v1(
        problem_id=p.task_id, test_source=p.test, entry_point=p.entry_point,
        candidate_code=code, **ek)
    return BigCodeBenchArmOutcomeV1(
        schema=W110_BIGCODEBENCH_REFLEXION_BENCH_V1_SCHEMA_VERSION,
        seed=int(seed), problem_id=str(p.task_id), arm_id="A0",
        final_passed=bool(exe.passed),
        final_candidate_code_cid=str(exe.candidate_code_cid),
        n_model_calls=1, n_executor_calls=1,
        total_wall_ms=int(wall + exe.wall_ms),
        executor_result_cids=(exe.cid(),),
        per_call_passed=(bool(exe.passed),),
        first_pass_attempt_idx=(0 if exe.passed else -1))


def _run_a1(*, seed, p, K, temperature, gen, max_tokens,
            ek) -> BigCodeBenchArmOutcomeV1:
    prompt = _initial_prompt(p)
    exes: list[BigCodeBenchExecutorResultV1] = []
    total = 0
    chosen_passed = False
    chosen_cid = ""
    chosen_idx = -1
    per_call_passed: list[bool] = []
    for k in range(int(K)):
        text, wall = gen(prompt, max_tokens, float(temperature))
        code = extract_candidate_code_v1(response_text=text)
        exe = run_bigcodebench_executor_v1(
            problem_id=p.task_id, test_source=p.test,
            entry_point=p.entry_point, candidate_code=code, **ek)
        exes.append(exe)
        per_call_passed.append(bool(exe.passed))
        total += int(wall) + int(exe.wall_ms)
        if exe.passed and not chosen_passed:
            chosen_passed = True
            chosen_cid = str(exe.candidate_code_cid)
            chosen_idx = int(k)
    if not chosen_passed:
        chosen_cid = str(exes[0].candidate_code_cid)
    return BigCodeBenchArmOutcomeV1(
        schema=W110_BIGCODEBENCH_REFLEXION_BENCH_V1_SCHEMA_VERSION,
        seed=int(seed), problem_id=str(p.task_id), arm_id="A1",
        final_passed=bool(chosen_passed),
        final_candidate_code_cid=str(chosen_cid),
        n_model_calls=int(K), n_executor_calls=int(K),
        total_wall_ms=int(total),
        executor_result_cids=tuple(e.cid() for e in exes),
        per_call_passed=tuple(per_call_passed),
        first_pass_attempt_idx=int(chosen_idx))


def _run_b(*, seed, p, K, temperature, gen, max_tokens,
           ek) -> BigCodeBenchArmOutcomeV1:
    history: list[tuple[str, BigCodeBenchExecutorResultV1]] = []
    exes: list[BigCodeBenchExecutorResultV1] = []
    total = 0
    first_pass_idx = -1
    per_call_passed: list[bool] = []
    for k in range(int(K)):
        prompt = (_initial_prompt(p) if k == 0
                  else _reflexion_prompt(p, tuple(history), attempt_idx=int(k)))
        text, wall = gen(prompt, max_tokens, float(temperature))
        code = extract_candidate_code_v1(response_text=text)
        exe = run_bigcodebench_executor_v1(
            problem_id=p.task_id, test_source=p.test,
            entry_point=p.entry_point, candidate_code=code, **ek)
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
    return BigCodeBenchArmOutcomeV1(
        schema=W110_BIGCODEBENCH_REFLEXION_BENCH_V1_SCHEMA_VERSION,
        seed=int(seed), problem_id=str(p.task_id), arm_id="B",
        final_passed=bool(final_passed),
        final_candidate_code_cid=str(final_cid),
        n_model_calls=int(K), n_executor_calls=int(K),
        total_wall_ms=int(total),
        executor_result_cids=tuple(e.cid() for e in exes),
        per_call_passed=tuple(per_call_passed),
        first_pass_attempt_idx=int(first_pass_idx))


@dataclasses.dataclass(frozen=True)
class BigCodeBenchSeedReportV1:
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
class BigCodeBenchBenchReportV1:
    schema: str
    model_id: str
    n_problems: int
    n_seeds: int
    K_multi_sample: int
    per_seed: tuple[BigCodeBenchSeedReportV1, ...]
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
        return _sha256_hex({"kind": "w110_bigcodebench_bench_report_v1",
                            "report": self.to_dict()})


@dataclasses.dataclass
class BigCodeBenchBenchConfigV1:
    schema: str = W110_BIGCODEBENCH_REFLEXION_BENCH_V1_SCHEMA_VERSION
    K_multi_sample: int = 5
    seeds: tuple[int, ...] = (110_001,)
    sampling_temperature: float = 0.7
    max_tokens_per_call: int = 1024
    executor_timeout_s: float = W110_BIGCODEBENCH_EXECUTOR_V1_TIMEOUT_S
    executor_kill_after_s: float = W110_BIGCODEBENCH_EXECUTOR_V1_KILL_AFTER_S
    # The CPython used by the executor subprocess. None -> sys.executable.
    # W110 points this at a --system-site-packages venv carrying the
    # BigCodeBench library stack, so the -I subprocess (which drops only the
    # USER site, not the venv/system site) imports the tasks' deps.
    executor_python_exe: str | None = None


def run_bigcodebench_reflexion_bench_v1(
        *,
        gen: _GenerateFn,
        model_id: str,
        subset: Sequence[BigCodeBenchProblemV1],
        config: BigCodeBenchBenchConfigV1 | None = None,
        on_problem_start: Callable[[int, int, str], None] | None = None,
) -> BigCodeBenchBenchReportV1:
    """Run A0 + A1 + B (sequential reflexion) on a PRE-SELECTED, ordered
    gold-green BigCodeBench subset for each seed. The subset (the deterministic
    n_libs-stratified slice) is consumed verbatim — no internal reshuffle,
    exactly like the APPS/LiveCodeBench benches."""
    cfg = config or BigCodeBenchBenchConfigV1()
    ek = {"timeout_s": float(cfg.executor_timeout_s),
          "kill_after_s": float(cfg.executor_kill_after_s),
          "python_exe": cfg.executor_python_exe}
    per_seed: list[BigCodeBenchSeedReportV1] = []
    all_cids: list[str] = []
    for seed in cfg.seeds:
        a0o: list[BigCodeBenchArmOutcomeV1] = []
        a1o: list[BigCodeBenchArmOutcomeV1] = []
        bo: list[BigCodeBenchArmOutcomeV1] = []
        for p_idx, p in enumerate(subset):
            if on_problem_start is not None:
                on_problem_start(int(seed), int(p_idx), str(p.task_id))
            a0o.append(_run_a0(seed=seed, p=p, gen=gen,
                               max_tokens=cfg.max_tokens_per_call, ek=ek))
            a1o.append(_run_a1(seed=seed, p=p, K=cfg.K_multi_sample,
                               temperature=cfg.sampling_temperature, gen=gen,
                               max_tokens=cfg.max_tokens_per_call, ek=ek))
            bo.append(_run_b(seed=seed, p=p, K=cfg.K_multi_sample,
                             temperature=cfg.sampling_temperature, gen=gen,
                             max_tokens=cfg.max_tokens_per_call, ek=ek))
        n = float(len(a0o)) or 1.0
        a0_acc = sum(1 for o in a0o if o.final_passed) / n
        a1_acc = sum(1 for o in a1o if o.final_passed) / n
        b_acc = sum(1 for o in bo if o.final_passed) / n
        cids = tuple([o.cid() for o in a0o] + [o.cid() for o in a1o]
                     + [o.cid() for o in bo])
        seed_merkle = _sha256_hex({"kind": "w110_bigcodebench_seed_merkle_v1",
                                   "seed": int(seed), "cids": list(cids)})
        per_seed.append(BigCodeBenchSeedReportV1(
            schema=W110_BIGCODEBENCH_REFLEXION_BENCH_V1_SCHEMA_VERSION,
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
    bench_merkle = _sha256_hex({"kind": "w110_bigcodebench_bench_merkle_v1",
                                "model_id": str(model_id),
                                "cids": all_cids, "seeds": list(cfg.seeds)})
    return BigCodeBenchBenchReportV1(
        schema=W110_BIGCODEBENCH_REFLEXION_BENCH_V1_SCHEMA_VERSION,
        model_id=str(model_id), n_problems=int(len(subset)),
        n_seeds=int(len(cfg.seeds)), K_multi_sample=int(cfg.K_multi_sample),
        per_seed=tuple(per_seed),
        a0_mean_pass_at_1=float(a0m), a1_mean_pass_at_1=float(a1m),
        b_mean_pass_at_1=float(bm),
        b_mean_minus_a1_mean_pp=float((bm - a1m) * 100.0),
        bench_merkle_root=str(bench_merkle))


__all__ = [
    "W110_BIGCODEBENCH_REFLEXION_BENCH_V1_SCHEMA_VERSION",
    "extract_candidate_code_v1",
    "select_bigcodebench_slice_v1",
    "BigCodeBenchArmOutcomeV1",
    "BigCodeBenchSeedReportV1",
    "BigCodeBenchBenchReportV1",
    "BigCodeBenchBenchConfigV1",
    "run_bigcodebench_reflexion_bench_v1",
]
