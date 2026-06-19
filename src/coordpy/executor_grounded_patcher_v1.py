"""W111 / COO-9 — Executor-grounded structured-failure patcher (M3).

A genuinely DIFFERENT same-budget mechanism from the W89/W110 sequential
*reflexion* bench (``bigcodebench_reflexion_bench_v1``), built to test the W111
Lane α question: **does a non-reflexion mechanism beat same-budget
self-consistency (A1) on contamination-resistant code at 70B?**

The W111 NIM-free mechanism-mining pass
(``scripts/mine_w111_resistant_failure_modes_v1.py`` →
``results/w111/mechanism_mining/w110_bcb_failure_census.json``) showed the
contamination-resistant BigCodeBench failure distribution is **81.6 %
SEMANTIC_LOGIC** (the model writes correctly-importing, plausible code that
yields the wrong output) and only **1.8 % API_GROUNDING**. So the failure to
attack is *failure-feedback actionability on semantic errors*, not API
hallucination (which kills the M2 introspection candidate) or library
composition (which, with no executor grounding, kills the M1 planner candidate).

**M3 is materially different from reflexion (B), not a prompt variant** — three
load-bearing axes (``docs/RUNBOOK_W111.md`` § 2.3):

1. **Typed failure digest, not prose history.** ``parse_failure_digest_v1``
   parses the executor ``stderr_tail`` into ``{failing_tests, exception_type,
   expected_repr, actual_repr, assertion_lines}`` and presents it as an explicit
   contract — vs reflexion's free-text "diagnose the bug class".
2. **Explicit target-value contract.** The patch prompt states "the suite
   required EXPECTED=… ; your code produced ACTUAL=… ; make that case equal
   EXPECTED" — reflexion never states an explicit expected/actual contract.
3. **Minimal targeted patch, not full rewrite.** M3 conditions only on the
   *latest* candidate + its typed digest (not the cumulative prose history) and
   asks for the smallest change that makes the failing assertion pass — aimed at
   the regression rate that cancelled reflexion's rescues (W110 net 0).

Three arms (A0 / A1 byte-identical to the W110 BigCodeBench bench; M3 replaces
B at the SAME K=5 byte-exact budget and the SAME information regime — docstring
+ executor pass/fail + 800-char ``stderr_tail``, **never the hidden test
source**, which would be oracle leakage):

* ``A0`` — stock single-shot at T=0.0.
* ``A1`` — first-pass-among-K=5 self-consistency at T=0.7.
* ``M3`` — 1 initial sample (T=0.7) + 4 executor-grounded structured-patch
  turns, each conditioned on the latest candidate + its typed failure digest.

Anti-cheat (carried forward verbatim): same model on every arm; same K=5 budget
on A1 and M3 (byte-exact; no early-stop); no selective retries; executor truth =
the BigCodeBench ``unittest`` oracle (exit 0 iff successful), NO LLM-as-judge;
per-call CIDs + per-seed Merkle + bench Merkle re-verifiable offline; the slice
is deterministic + OUTCOME-BLIND. **M3 never sees the ``test`` source** — only
the executor verdict + ``stderr_tail`` (identical to what reflexion B saw), so
any M3 win is mechanism, not oracle access.
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
from .bigcodebench_reflexion_bench_v1 import extract_candidate_code_v1

W111_EXECUTOR_GROUNDED_PATCHER_V1_SCHEMA_VERSION: str = (
    "coordpy.executor_grounded_patcher_v1.v1")


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


# ---------------------------------------------------------------------------
# Typed failure digest (the M3 differentiator vs reflexion's prose stderr tail)
# ---------------------------------------------------------------------------

_FAIL_HDR_RE = re.compile(r"^(FAIL|ERROR): (\S+)", re.M)
_EXC_RE = re.compile(
    r"^([A-Za-z_][A-Za-z0-9_\.]*(?:Error|Exception)):(.*)$", re.M)
# unittest assertEqual prints "<a> != <b>"; capture both sides from the
# AssertionError message (best-effort; falls back to the raw line).
_NEQ_RE = re.compile(r"^AssertionError:\s*(.*?)\s*!=\s*(.*)$", re.M | re.S)


@dataclasses.dataclass(frozen=True)
class FailureDigestV1:
    """Typed parse of an executor ``stderr_tail`` (NO test source consulted)."""
    schema: str
    failing_tests: tuple[str, ...]
    exception_type: str
    assertion_line: str
    expected_repr: str
    actual_repr: str
    raw_tail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "failing_tests": list(self.failing_tests),
            "exception_type": str(self.exception_type),
            "assertion_line": str(self.assertion_line),
            "expected_repr": str(self.expected_repr),
            "actual_repr": str(self.actual_repr),
            "raw_tail": str(self.raw_tail),
        }


def parse_failure_digest_v1(*, stderr_tail: str,
                            timed_out: bool) -> FailureDigestV1:
    """Parse the executor's 800-char ``stderr_tail`` into a typed record.

    Pure, deterministic, NO LLM. Uses ONLY the executor stderr (the same signal
    reflexion B received) — never the hidden ``test`` source."""
    tail = str(stderr_tail or "")
    if timed_out:
        return FailureDigestV1(
            schema=W111_EXECUTOR_GROUNDED_PATCHER_V1_SCHEMA_VERSION,
            failing_tests=(), exception_type="Timeout",
            assertion_line="execution exceeded the wall-clock limit",
            expected_repr="", actual_repr="", raw_tail=tail)
    fails = tuple(name for _kind, name in _FAIL_HDR_RE.findall(tail))
    exc_matches = _EXC_RE.findall(tail)
    exc_type = exc_matches[-1][0].split(".")[-1] if exc_matches else ""
    assertion_line = ""
    expected_repr = ""
    actual_repr = ""
    neq = _NEQ_RE.search(tail)
    if neq:
        # unittest assertEqual(actual, expected) -> "<actual> != <expected>".
        actual_repr = neq.group(1).strip()[:300]
        expected_repr = neq.group(2).strip().splitlines()[0][:300]
        assertion_line = ("AssertionError: "
                          + (actual_repr + " != " + expected_repr))[:400]
    elif exc_matches:
        assertion_line = (exc_matches[-1][0] + ":" + exc_matches[-1][1]).strip()[:400]
    return FailureDigestV1(
        schema=W111_EXECUTOR_GROUNDED_PATCHER_V1_SCHEMA_VERSION,
        failing_tests=fails, exception_type=str(exc_type),
        assertion_line=str(assertion_line),
        expected_repr=str(expected_repr), actual_repr=str(actual_repr),
        raw_tail=tail)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

def _initial_prompt(problem: BigCodeBenchProblemV1) -> str:
    """Byte-identical to the W110 BigCodeBench bench initial prompt (so A0/A1
    are directly comparable to W110)."""
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


def _patch_prompt(problem: BigCodeBenchProblemV1, latest_code: str,
                  digest: FailureDigestV1, attempt_idx: int) -> str:
    """The M3 differentiator: a TYPED failure contract + minimal-patch framing,
    conditioned on the LATEST candidate only (not prose history)."""
    code_trim = latest_code if len(latest_code) <= 1800 else (
        latest_code[:1800] + "\n# ... (truncated)\n")
    tests = ", ".join(digest.failing_tests[:4]) or "(unnamed)"
    contract_lines: list[str] = [
        f"- FAILING TEST(S): {tests}",
        f"- ERROR TYPE: {digest.exception_type or 'test failure'}",
    ]
    if digest.expected_repr or digest.actual_repr:
        contract_lines.append(
            f"- YOUR CODE PRODUCED (actual): {digest.actual_repr or '(see below)'}")
        contract_lines.append(
            f"- THE SUITE REQUIRED (expected): {digest.expected_repr or '(see below)'}")
    elif digest.assertion_line:
        contract_lines.append(f"- FAILURE: {digest.assertion_line}")
    contract = "\n".join(contract_lines)
    raw = digest.raw_tail.strip()
    raw_block = (f"\nFull executor stderr tail:\n{raw}" if raw else "")
    return (
        "You are an expert Python programmer doing EXECUTOR-GROUNDED PATCHING. "
        f"You are on patch attempt {attempt_idx + 1} of 4. Your latest "
        "solution FAILED the hidden unittest suite. Below is a STRUCTURED "
        "failure digest extracted from the executor (you do NOT get the test "
        "source — only this verdict).\n\n"
        "TASK SPEC:\n"
        f"```python\n{problem.complete_prompt}\n```\n\n"
        "YOUR LATEST SOLUTION:\n"
        f"```python\n{code_trim}\n```\n\n"
        "STRUCTURED FAILURE DIGEST:\n"
        f"{contract}{raw_block}\n\n"
        "Make the SMALLEST change to your latest solution that makes the "
        "failing test(s) pass WITHOUT breaking the parts that already work. If "
        "an expected value is given, ensure your function returns EXACTLY that "
        "for the failing case. Output ONLY the corrected COMPLETE solution "
        f"(entry `{problem.entry_point}`, all imports) in a single "
        "```python ... ``` fence:")


_GenerateFn = Callable[[str, int, float], tuple[str, int]]


# ---------------------------------------------------------------------------
# Arm outcome (mirrors BigCodeBenchArmOutcomeV1 for audit-chain parity)
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class PatcherArmOutcomeV1:
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
        return _sha256_hex({"kind": "w111_patcher_arm_outcome_v1",
                            "capsule": self.to_dict()})


def _exec(p: BigCodeBenchProblemV1, code: str, ek) -> BigCodeBenchExecutorResultV1:
    return run_bigcodebench_executor_v1(
        problem_id=p.task_id, test_source=p.test, entry_point=p.entry_point,
        candidate_code=code, **ek)


def _run_a0(*, seed, p, gen, max_tokens, ek) -> PatcherArmOutcomeV1:
    text, wall = gen(_initial_prompt(p), max_tokens, 0.0)
    code = extract_candidate_code_v1(response_text=text)
    exe = _exec(p, code, ek)
    return PatcherArmOutcomeV1(
        schema=W111_EXECUTOR_GROUNDED_PATCHER_V1_SCHEMA_VERSION,
        seed=int(seed), problem_id=str(p.task_id), arm_id="A0",
        final_passed=bool(exe.passed),
        final_candidate_code_cid=str(exe.candidate_code_cid),
        n_model_calls=1, n_executor_calls=1,
        total_wall_ms=int(wall + exe.wall_ms),
        executor_result_cids=(exe.cid(),),
        per_call_passed=(bool(exe.passed),),
        first_pass_attempt_idx=(0 if exe.passed else -1))


def _run_a1(*, seed, p, K, temperature, gen, max_tokens, ek) -> PatcherArmOutcomeV1:
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
        exe = _exec(p, code, ek)
        exes.append(exe)
        per_call_passed.append(bool(exe.passed))
        total += int(wall) + int(exe.wall_ms)
        if exe.passed and not chosen_passed:
            chosen_passed = True
            chosen_cid = str(exe.candidate_code_cid)
            chosen_idx = int(k)
    if not chosen_passed:
        chosen_cid = str(exes[0].candidate_code_cid)
    return PatcherArmOutcomeV1(
        schema=W111_EXECUTOR_GROUNDED_PATCHER_V1_SCHEMA_VERSION,
        seed=int(seed), problem_id=str(p.task_id), arm_id="A1",
        final_passed=bool(chosen_passed),
        final_candidate_code_cid=str(chosen_cid),
        n_model_calls=int(K), n_executor_calls=int(K),
        total_wall_ms=int(total),
        executor_result_cids=tuple(e.cid() for e in exes),
        per_call_passed=tuple(per_call_passed),
        first_pass_attempt_idx=int(chosen_idx))


def _run_m3(*, seed, p, K, temperature, gen, max_tokens, ek) -> PatcherArmOutcomeV1:
    """1 initial sample + (K-1) executor-grounded structured-patch turns. Each
    patch conditions on the LATEST candidate + its TYPED failure digest (NOT the
    cumulative prose history). Same K-call byte-exact budget as A1/B."""
    exes: list[BigCodeBenchExecutorResultV1] = []
    total = 0
    first_pass_idx = -1
    per_call_passed: list[bool] = []
    latest_code = ""
    for k in range(int(K)):
        if k == 0:
            prompt = _initial_prompt(p)
        else:
            digest = parse_failure_digest_v1(
                stderr_tail=exes[-1].stderr_tail, timed_out=exes[-1].timed_out)
            prompt = _patch_prompt(p, latest_code, digest, attempt_idx=int(k))
        text, wall = gen(prompt, max_tokens, float(temperature))
        code = extract_candidate_code_v1(response_text=text)
        latest_code = code
        exe = _exec(p, code, ek)
        exes.append(exe)
        per_call_passed.append(bool(exe.passed))
        total += int(wall) + int(exe.wall_ms)
        if exe.passed and first_pass_idx == -1:
            first_pass_idx = int(k)
    final_passed = (first_pass_idx >= 0)
    if final_passed:
        final_cid = str(exes[first_pass_idx].candidate_code_cid)
    else:
        final_cid = sorted(str(e.candidate_code_cid) for e in exes)[0]
    return PatcherArmOutcomeV1(
        schema=W111_EXECUTOR_GROUNDED_PATCHER_V1_SCHEMA_VERSION,
        seed=int(seed), problem_id=str(p.task_id), arm_id="M3",
        final_passed=bool(final_passed),
        final_candidate_code_cid=str(final_cid),
        n_model_calls=int(K), n_executor_calls=int(K),
        total_wall_ms=int(total),
        executor_result_cids=tuple(e.cid() for e in exes),
        per_call_passed=tuple(per_call_passed),
        first_pass_attempt_idx=int(first_pass_idx))


# ---------------------------------------------------------------------------
# Reports (mirror the BigCodeBench bench so the 9-gate evaluator consumes them)
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class PatcherSeedReportV1:
    schema: str
    seed: int
    n_problems: int
    a0_pass_at_1: float
    a1_pass_at_1: float
    m3_pass_at_1: float
    per_problem_a0_passed: tuple[bool, ...]
    per_problem_a1_passed: tuple[bool, ...]
    per_problem_m3_passed: tuple[bool, ...]
    per_problem_m3_first_pass_idx: tuple[int, ...]
    problem_ids: tuple[str, ...]
    seed_merkle_root: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema), "seed": int(self.seed),
            "n_problems": int(self.n_problems),
            "a0_pass_at_1": float(round(self.a0_pass_at_1, 6)),
            "a1_pass_at_1": float(round(self.a1_pass_at_1, 6)),
            "m3_pass_at_1": float(round(self.m3_pass_at_1, 6)),
            "per_problem_a0_passed": list(self.per_problem_a0_passed),
            "per_problem_a1_passed": list(self.per_problem_a1_passed),
            "per_problem_m3_passed": list(self.per_problem_m3_passed),
            "per_problem_m3_first_pass_idx": list(
                self.per_problem_m3_first_pass_idx),
            "problem_ids": list(self.problem_ids),
            "seed_merkle_root": str(self.seed_merkle_root),
        }


@dataclasses.dataclass(frozen=True)
class PatcherBenchReportV1:
    schema: str
    model_id: str
    n_problems: int
    n_seeds: int
    K_multi_sample: int
    per_seed: tuple[PatcherSeedReportV1, ...]
    a0_mean_pass_at_1: float
    a1_mean_pass_at_1: float
    m3_mean_pass_at_1: float
    m3_mean_minus_a1_mean_pp: float
    bench_merkle_root: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema), "model_id": str(self.model_id),
            "n_problems": int(self.n_problems), "n_seeds": int(self.n_seeds),
            "K_multi_sample": int(self.K_multi_sample),
            "per_seed": [s.to_dict() for s in self.per_seed],
            "a0_mean_pass_at_1": float(round(self.a0_mean_pass_at_1, 6)),
            "a1_mean_pass_at_1": float(round(self.a1_mean_pass_at_1, 6)),
            "m3_mean_pass_at_1": float(round(self.m3_mean_pass_at_1, 6)),
            "m3_mean_minus_a1_mean_pp": float(round(
                self.m3_mean_minus_a1_mean_pp, 4)),
            "bench_merkle_root": str(self.bench_merkle_root),
        }

    def cid(self) -> str:
        return _sha256_hex({"kind": "w111_patcher_bench_report_v1",
                            "report": self.to_dict()})


@dataclasses.dataclass
class PatcherBenchConfigV1:
    schema: str = W111_EXECUTOR_GROUNDED_PATCHER_V1_SCHEMA_VERSION
    K_multi_sample: int = 5
    seeds: tuple[int, ...] = (111_001,)
    sampling_temperature: float = 0.7
    max_tokens_per_call: int = 1024
    executor_timeout_s: float = W110_BIGCODEBENCH_EXECUTOR_V1_TIMEOUT_S
    executor_kill_after_s: float = W110_BIGCODEBENCH_EXECUTOR_V1_KILL_AFTER_S
    executor_python_exe: str | None = None


def run_executor_grounded_patcher_bench_v1(
        *,
        gen: _GenerateFn,
        model_id: str,
        subset: Sequence[BigCodeBenchProblemV1],
        config: PatcherBenchConfigV1 | None = None,
        on_problem_start: Callable[[int, int, str], None] | None = None,
) -> PatcherBenchReportV1:
    """Run A0 + A1 + M3 on a PRE-SELECTED, ordered gold-green BigCodeBench
    subset for each seed. Subset consumed verbatim (no reshuffle), exactly like
    the W110 BigCodeBench bench."""
    cfg = config or PatcherBenchConfigV1()
    ek = {"timeout_s": float(cfg.executor_timeout_s),
          "kill_after_s": float(cfg.executor_kill_after_s),
          "python_exe": cfg.executor_python_exe}
    per_seed: list[PatcherSeedReportV1] = []
    all_cids: list[str] = []
    for seed in cfg.seeds:
        a0o: list[PatcherArmOutcomeV1] = []
        a1o: list[PatcherArmOutcomeV1] = []
        m3o: list[PatcherArmOutcomeV1] = []
        for p_idx, p in enumerate(subset):
            if on_problem_start is not None:
                on_problem_start(int(seed), int(p_idx), str(p.task_id))
            a0o.append(_run_a0(seed=seed, p=p, gen=gen,
                               max_tokens=cfg.max_tokens_per_call, ek=ek))
            a1o.append(_run_a1(seed=seed, p=p, K=cfg.K_multi_sample,
                               temperature=cfg.sampling_temperature, gen=gen,
                               max_tokens=cfg.max_tokens_per_call, ek=ek))
            m3o.append(_run_m3(seed=seed, p=p, K=cfg.K_multi_sample,
                               temperature=cfg.sampling_temperature, gen=gen,
                               max_tokens=cfg.max_tokens_per_call, ek=ek))
        n = float(len(a0o)) or 1.0
        a0_acc = sum(1 for o in a0o if o.final_passed) / n
        a1_acc = sum(1 for o in a1o if o.final_passed) / n
        m3_acc = sum(1 for o in m3o if o.final_passed) / n
        cids = tuple([o.cid() for o in a0o] + [o.cid() for o in a1o]
                     + [o.cid() for o in m3o])
        seed_merkle = _sha256_hex({"kind": "w111_patcher_seed_merkle_v1",
                                   "seed": int(seed), "cids": list(cids)})
        per_seed.append(PatcherSeedReportV1(
            schema=W111_EXECUTOR_GROUNDED_PATCHER_V1_SCHEMA_VERSION,
            seed=int(seed), n_problems=int(len(a0o)),
            a0_pass_at_1=float(a0_acc), a1_pass_at_1=float(a1_acc),
            m3_pass_at_1=float(m3_acc),
            per_problem_a0_passed=tuple(bool(o.final_passed) for o in a0o),
            per_problem_a1_passed=tuple(bool(o.final_passed) for o in a1o),
            per_problem_m3_passed=tuple(bool(o.final_passed) for o in m3o),
            per_problem_m3_first_pass_idx=tuple(
                int(o.first_pass_attempt_idx) for o in m3o),
            problem_ids=tuple(str(o.problem_id) for o in a0o),
            seed_merkle_root=str(seed_merkle)))
        all_cids.extend(cids)
    ns = float(len(per_seed)) or 1.0
    a0m = sum(s.a0_pass_at_1 for s in per_seed) / ns
    a1m = sum(s.a1_pass_at_1 for s in per_seed) / ns
    m3m = sum(s.m3_pass_at_1 for s in per_seed) / ns
    bench_merkle = _sha256_hex({"kind": "w111_patcher_bench_merkle_v1",
                                "model_id": str(model_id),
                                "cids": all_cids, "seeds": list(cfg.seeds)})
    return PatcherBenchReportV1(
        schema=W111_EXECUTOR_GROUNDED_PATCHER_V1_SCHEMA_VERSION,
        model_id=str(model_id), n_problems=int(len(subset)),
        n_seeds=int(len(cfg.seeds)), K_multi_sample=int(cfg.K_multi_sample),
        per_seed=tuple(per_seed),
        a0_mean_pass_at_1=float(a0m), a1_mean_pass_at_1=float(a1m),
        m3_mean_pass_at_1=float(m3m),
        m3_mean_minus_a1_mean_pp=float((m3m - a1m) * 100.0),
        bench_merkle_root=str(bench_merkle))


__all__ = [
    "W111_EXECUTOR_GROUNDED_PATCHER_V1_SCHEMA_VERSION",
    "FailureDigestV1",
    "parse_failure_digest_v1",
    "PatcherArmOutcomeV1",
    "PatcherSeedReportV1",
    "PatcherBenchReportV1",
    "PatcherBenchConfigV1",
    "run_executor_grounded_patcher_bench_v1",
]
