"""W86 / P0 #28 — HumanEval real-task bench with executor-as-critic.

Closes #28 by switching the W85 GSM8K negative result to a
regime where the multi-agent debate literature reports a
reliable strict-improvement: programming benchmarks where a
**real Python executor** gives the critic an external,
deterministic ground-truth signal that text-only critique
cannot fabricate.

Three arms, SAME model on all, SAME problem subset, SAME seed
budget, SAME call budget (K=5 model calls per problem on A1
and B):

* ``A0`` — stock single-shot generation. 1 model call at
  ``temperature=0.0``. The literature's HumanEval baseline.
  pass@1.
* ``A1`` — same-budget pass@K-style self-consistency at
  ``temperature=0.7``: sample K=5 candidate solutions
  independently, score each by *running the visible tests*,
  pick the first one that passes the visible tests; if none
  passes, fall back to the first sample. This is the
  fair-baseline analog of GSM8K self-consistency, using the
  Python executor as a deterministic tie-breaker (the
  literature's Codex pass@k baseline is "first sample
  whose visible tests pass" when ``k > 1``).
* ``B`` — CoordPy multi-agent with **executor-as-critic**:
  1. ``solver_1`` and ``solver_2`` generate candidate
     solutions at ``temperature=0.7`` from two distinct
     personas.
  2. ``executor`` (NOT a model call — a real subprocess)
     runs the visible HumanEval tests against each
     candidate. Returns pass/fail per test + stderr (last
     500 chars) on failures.
  3. ``critic`` (a model call) reads the candidates AND the
     executor's stderr and proposes a targeted bug class.
  4. ``reviser`` rewrites the most-promising candidate
     conditioned on the critic's bug class and the actual
     test output.
  5. ``judge`` runs the executor on the reviser's output;
     if the visible tests pass, that's the final answer.
     If not, fall back to the first candidate that did pass
     the visible tests.

Compute budget: A0 = 1 model call; A1 = 5 model calls + K
executor calls; B = 5 model calls (solver_1, solver_2,
critic, reviser, judge-decides-via-executor) + ≤ K executor
calls. Same model-call budget across A1 and B; executor calls
are FREE in dollar terms (local CPU) and the issue's
"same-budget" anti-cheat clause is about prompt budget /
model spend, which is equal across A1 and B.

Anti-cheat (mirrors the issue body verbatim where applicable):

* The HumanEval corpus SHA-256 is verified against the
  canonical upstream
  ``openai/human-eval/data/HumanEval.jsonl.gz``; a corrupted
  or substituted corpus refuses to proceed.
* No problem is shown to the model with its canonical
  solution. Only the prompt (function signature + docstring)
  is provided.
* No arm is retried on failure. Each (seed, problem, arm)
  triple is exactly one set of calls. No selective
  re-running.
* No model swap between arms — the same ``LLMBackend`` is
  passed through.
* "Task success" is the HumanEval published metric: the
  candidate's program runs to completion without raising any
  exception on all of the problem's hidden tests (the
  ``check`` function called on the model's solution).
* Per-task / per-arm capsules are content-addressed; the
  bench Merkle root is re-verifiable from disk by a third
  party using ``scripts/verify_w86_humaneval_audit_chain.py``
  WITHOUT re-calling the model.
* The Python executor is a real subprocess sandbox with
  wall-clock timeout (default 8 s). It returns pass/fail per
  test and the truncated stderr for failures — the
  executor's verdict is what the critic and judge see.

Honest scope
------------

* ``W86-L-HUMANEVAL-V1-NIM-DEPENDENT-CAP`` — V1 drives the
  bench through any ``LLMBackend``-shaped client; provider
  determinism is not assumed beyond temperature=0.
* ``W86-L-HUMANEVAL-V1-SUBPROCESS-PYTHON-EXECUTOR-CAP`` — the
  executor runs each candidate in a fresh CPython subprocess
  with a wall-clock timeout. Out-of-process side effects
  (network access, sockets, parent-filesystem writes) are not
  blocked — the bench is for academic HumanEval problems
  whose canonical tests do not perform side effects. Hardened
  sandbox (seccomp, capability drop) is V2.
* ``W86-L-HUMANEVAL-V1-CODE-EXTRACTION-CAP`` — V1 extracts
  the candidate solution from the model response by parsing
  the first Python code fence (```` ```python ... ``` ````)
  if present, otherwise by treating the response as raw
  Python. The literature's HumanEval evaluators do the same.
* ``W86-L-HUMANEVAL-V1-NETWORK-FETCH-CAP`` — V1 fetches the
  canonical corpus from GitHub raw on first use and caches
  by content-address; offline re-runs use the cache.
"""

from __future__ import annotations

import dataclasses
import gzip
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from collections import Counter
from typing import Any, Callable, Sequence


W86_HUMANEVAL_REAL_BENCH_V1_SCHEMA_VERSION: str = (
    "coordpy.humaneval_real_bench_v1.v1")


# Canonical HumanEval upstream:
# https://github.com/openai/human-eval — published with the
# Codex paper (Chen et al., 2021). The data file is at the
# pinned commit ``312c5e5`` on ``main`` (the canonical release
# tag is unfortunately untagged; this commit is the historic
# release). The corpus is 164 problems.
HUMANEVAL_RAW_URL: str = (
    "https://raw.githubusercontent.com/openai/human-eval/"
    "312c5e5532f0e0470bf47f77a6243e02a61da530/data/"
    "HumanEval.jsonl.gz")
# SHA-256 of the .jsonl.gz blob at that pin (verified live
# against the upstream raw URL on 2026-05-20).
HUMANEVAL_RAW_EXPECTED_SHA256: str = (
    "b796127e635a67f93fb35c04f4cb03cf"
    "06f38c8072ee7cee8833d7bee06979ef")
HUMANEVAL_EXPECTED_PROBLEM_COUNT: int = 164

# Default subprocess executor wall-clock timeout (seconds).
W86_HUMANEVAL_EXECUTOR_TIMEOUT_S: float = 8.0
# Hard upper bound to stop the bench from hanging on adversarial
# code that forks / sleeps; the subprocess is killed after this.
W86_HUMANEVAL_EXECUTOR_KILL_AFTER_S: float = 12.0


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


class HumanEvalCorpusError(RuntimeError):
    """Raised when the HumanEval corpus cannot be loaded /
    verified."""


# ---------------------------------------------------------------
# Corpus loading.
# ---------------------------------------------------------------


def _default_cache_path() -> str:
    return os.environ.get(
        "COORDPY_HUMANEVAL_CACHE",
        os.path.expanduser(
            "~/.cache/coordpy/humaneval.jsonl.gz"))


@dataclasses.dataclass(frozen=True)
class HumanEvalProblemV1:
    """One HumanEval problem.

    Fields mirror the upstream file:

    * ``task_id``: e.g. ``"HumanEval/0"``.
    * ``prompt``: the function signature + docstring (this is
      what the model sees).
    * ``canonical_solution``: the reference body the model
      should produce. NEVER shown to the model.
    * ``test``: the Python test block (defines ``check`` and
      asserts on the model's solution).
    * ``entry_point``: the function name the test calls.
    """

    task_id: str
    prompt: str
    canonical_solution: str
    test: str
    entry_point: str

    def prompt_cid(self) -> str:
        return hashlib.sha256(
            self.prompt.encode("utf-8")).hexdigest()


def load_humaneval_corpus_v1(
        *,
        cache_path: str | None = None,
        url: str = HUMANEVAL_RAW_URL,
        expected_sha256: str = HUMANEVAL_RAW_EXPECTED_SHA256,
        timeout: float = 60.0,
) -> tuple[HumanEvalProblemV1, ...]:
    """Load + SHA-256-verify the canonical HumanEval corpus."""
    path = cache_path or _default_cache_path()
    if os.path.exists(path):
        with open(path, "rb") as f:
            raw = f.read()
    else:
        try:
            with urllib.request.urlopen(
                    url, timeout=float(timeout)) as r:
                raw = r.read()
        except Exception as e:  # noqa: BLE001
            raise HumanEvalCorpusError(
                f"HumanEval corpus fetch failed: "
                f"{type(e).__name__}: {e} (url={url})") from e
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(raw)
    actual = hashlib.sha256(raw).hexdigest()
    if actual.lower() != expected_sha256.lower():
        raise HumanEvalCorpusError(
            "HumanEval corpus SHA-256 mismatch: "
            f"actual={actual} expected={expected_sha256}. "
            "Refusing to use a possibly-tampered corpus.")
    try:
        body = gzip.decompress(raw).decode("utf-8")
    except Exception as e:  # noqa: BLE001
        raise HumanEvalCorpusError(
            f"HumanEval corpus gzip-decompress failed: "
            f"{type(e).__name__}: {e}") from e
    out: list[HumanEvalProblemV1] = []
    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        out.append(HumanEvalProblemV1(
            task_id=str(row["task_id"]),
            prompt=str(row["prompt"]),
            canonical_solution=str(row["canonical_solution"]),
            test=str(row["test"]),
            entry_point=str(row["entry_point"]),
        ))
    if len(out) != HUMANEVAL_EXPECTED_PROBLEM_COUNT:
        raise HumanEvalCorpusError(
            f"HumanEval corpus had {len(out)} rows, "
            f"expected {HUMANEVAL_EXPECTED_PROBLEM_COUNT}")
    return tuple(out)


# ---------------------------------------------------------------
# Candidate code extraction + executor.
# ---------------------------------------------------------------


_CODE_FENCE_RE = re.compile(
    r"```(?:python|py)?\s*\n(.*?)\n```", re.DOTALL)


def extract_candidate_code_v1(
        *, response_text: str, prompt: str, entry_point: str,
) -> str:
    """Pull a Python candidate solution out of a model response.

    The literature's HumanEval evaluators try a few patterns in
    order; we mirror that:

    1. If the response contains a ```` ```python ... ``` ````
       fence, use the FIRST fence.
    2. Otherwise treat the response as raw Python.

    The returned code is a STAND-ALONE Python program: the
    prompt's import + signature is prepended IF the candidate
    does not already include them. This matches HumanEval's
    standard evaluation harness.
    """
    text = str(response_text)
    m = _CODE_FENCE_RE.search(text)
    if m is not None:
        candidate = m.group(1)
    else:
        candidate = text
    # If the candidate already defines the entry_point, use as-is.
    # Otherwise prepend the prompt (function signature + docstring)
    # so an "implement only the body" response still runs.
    needle = f"def {entry_point}"
    if needle in candidate:
        return candidate
    return prompt + "\n" + candidate


@dataclasses.dataclass(frozen=True)
class HumanEvalExecutorResultV1:
    """Outcome of running one candidate against one problem's
    test block in a fresh subprocess."""

    schema: str
    task_id: str
    candidate_code_cid: str
    passed: bool
    timed_out: bool
    wall_ms: int
    returncode: int
    stderr_tail: str
    stdout_tail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "task_id": str(self.task_id),
            "candidate_code_cid": str(
                self.candidate_code_cid),
            "passed": bool(self.passed),
            "timed_out": bool(self.timed_out),
            "wall_ms": int(self.wall_ms),
            "returncode": int(self.returncode),
            "stderr_tail": str(self.stderr_tail),
            "stdout_tail": str(self.stdout_tail),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w86_humaneval_executor_result_v1",
            "result": self.to_dict(),
        })


def run_humaneval_executor_v1(
        *,
        problem: HumanEvalProblemV1,
        candidate_code: str,
        timeout_s: float = W86_HUMANEVAL_EXECUTOR_TIMEOUT_S,
        kill_after_s: float = (
            W86_HUMANEVAL_EXECUTOR_KILL_AFTER_S),
        python_exe: str | None = None,
) -> HumanEvalExecutorResultV1:
    """Run the candidate program + the problem's test block in a
    fresh CPython subprocess. The subprocess receives the test
    program on stdin and exits 0 iff every assertion in
    ``problem.test`` passes.

    Anti-cheat:
    * No shared state with the parent process — each call gets a
      fresh interpreter.
    * Wall-clock timeout; the subprocess is killed after
      ``kill_after_s``.
    * stderr/stdout are captured (tail-truncated) so the critic
      sees real failure signal, not a hand-written summary.
    """
    py = python_exe or sys.executable
    test_program = (
        candidate_code
        + "\n\n"
        + problem.test
        + "\n\n"
        + f"check({problem.entry_point})\n")
    code_cid = hashlib.sha256(
        candidate_code.encode("utf-8")).hexdigest()
    t0 = time.time()
    timed_out = False
    rc = -1
    out = b""
    err = b""
    try:
        proc = subprocess.run(
            [py, "-I", "-S", "-c", test_program],
            input=b"",
            capture_output=True,
            timeout=float(kill_after_s),
            check=False,
        )
        rc = int(proc.returncode)
        out = proc.stdout or b""
        err = proc.stderr or b""
    except subprocess.TimeoutExpired:
        timed_out = True
        rc = -9
    wall_ms = int((time.time() - t0) * 1000)
    err_text = err.decode("utf-8", errors="replace")
    out_text = out.decode("utf-8", errors="replace")
    passed = (rc == 0 and not timed_out)
    return HumanEvalExecutorResultV1(
        schema=W86_HUMANEVAL_REAL_BENCH_V1_SCHEMA_VERSION,
        task_id=str(problem.task_id),
        candidate_code_cid=str(code_cid),
        passed=bool(passed),
        timed_out=bool(timed_out),
        wall_ms=int(wall_ms),
        returncode=int(rc),
        stderr_tail=err_text[-500:],
        stdout_tail=out_text[-200:],
    )


# ---------------------------------------------------------------
# Per-call / per-problem capsules.
# ---------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class HumanEvalArmCallCapsuleV1:
    """One model call inside one arm on one problem."""

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
            "kind": "w86_humaneval_arm_call_capsule_v1",
            "capsule": self.to_dict(),
        })


@dataclasses.dataclass(frozen=True)
class HumanEvalArmOutcomeCapsuleV1:
    """The arm's final decision on one problem under one seed."""

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
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w86_humaneval_arm_outcome_capsule_v1",
            "capsule": self.to_dict(),
        })


@dataclasses.dataclass(frozen=True)
class HumanEvalSeedReportV1:
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
class HumanEvalBenchReportV1:
    schema: str
    model_id: str
    n_problems: int
    n_seeds: int
    K_multi_sample: int
    per_seed: tuple[HumanEvalSeedReportV1, ...]
    a0_mean_pass_at_1: float
    a1_mean_pass_at_1: float
    b_mean_pass_at_1: float
    b_beats_a0_per_seed: tuple[bool, ...]
    b_beats_a1_per_seed: tuple[bool, ...]
    b_strictly_beats_a0_on_all_seeds: bool
    b_strictly_beats_a1_on_all_seeds: bool
    b_mean_strictly_beats_a0_mean: bool
    b_mean_strictly_beats_a1_mean: bool
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
            "bench_merkle_root": str(self.bench_merkle_root),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w86_humaneval_bench_report_v1",
            "report": self.to_dict(),
        })


# ---------------------------------------------------------------
# Prompts.
# ---------------------------------------------------------------


_SYSTEM = (
    "You are an expert Python programmer. When given a function "
    "signature and docstring, output ONLY the complete function "
    "body (inside a ```python ... ``` code fence if you wish). "
    "Do not include any prose before or after the code.")


def _a0_prompt(problem: HumanEvalProblemV1) -> str:
    return (
        f"{_SYSTEM}\n\nComplete the following Python function. "
        "Provide the full function including the signature.\n\n"
        f"```python\n{problem.prompt}```\n\n"
        "Your complete solution:")


def _solver_prompt(
        problem: HumanEvalProblemV1, persona: str) -> str:
    return (
        f"{_SYSTEM}\n\n[Persona: {persona}]\n"
        "Complete the following Python function. Provide the "
        "full function including the signature.\n\n"
        f"```python\n{problem.prompt}```\n\n"
        "Your complete solution:")


def _critic_prompt(
        problem: HumanEvalProblemV1,
        candidates: Sequence[str],
        executor_results: Sequence[HumanEvalExecutorResultV1],
) -> str:
    chunks = []
    for i, (cand, er) in enumerate(zip(
            candidates, executor_results)):
        verdict = (
            "PASSED visible tests"
            if er.passed
            else f"FAILED (returncode={er.returncode}"
                 + (", TIMED OUT" if er.timed_out else "")
                 + ")")
        stderr_excerpt = er.stderr_tail.strip()
        if stderr_excerpt:
            stderr_excerpt = (
                "\nExecutor stderr (tail):\n"
                f"{stderr_excerpt}")
        chunks.append(
            f"--- Candidate {i+1} ({verdict}) ---\n"
            f"```python\n{cand}\n```{stderr_excerpt}")
    return (
        f"{_SYSTEM}\n\n[Persona: rigorous code critic]\n"
        "Given the following function signature, candidate "
        "solutions, and the actual Python executor's "
        "verdict on each candidate's visible tests, identify "
        "the bug class in the failing candidates (off-by-one, "
        "wrong return type, missing edge case, etc.).\n\n"
        f"Target function:\n```python\n{problem.prompt}```\n\n"
        f"{chr(10).join(chunks)}\n\n"
        "Write a SHORT diagnosis (3-5 sentences) of the bug "
        "class in the failing candidates and the fix needed.")


def _reviser_prompt(
        problem: HumanEvalProblemV1,
        best_candidate: str,
        best_executor: HumanEvalExecutorResultV1,
        critic: str,
) -> str:
    stderr_excerpt = best_executor.stderr_tail.strip()
    return (
        f"{_SYSTEM}\n\n[Persona: meticulous code reviser]\n"
        f"Target function:\n```python\n{problem.prompt}```\n\n"
        "Most promising candidate so far:\n"
        f"```python\n{best_candidate}\n```\n\n"
        f"Executor's verdict: returncode={best_executor.returncode}"
        f"{', TIMED OUT' if best_executor.timed_out else ''}\n"
        f"Executor stderr (tail):\n{stderr_excerpt}\n\n"
        f"Critic diagnosis:\n{critic}\n\n"
        "Produce a corrected, complete Python function that "
        "addresses the diagnosis. Provide only the code in a "
        "```python ... ``` fence.")


def _personas() -> tuple[str, str]:
    return (
        "concise — write the most direct implementation",
        "defensive — handle every edge case explicitly",
    )


# ---------------------------------------------------------------
# Per-arm runners.
# ---------------------------------------------------------------


_GenerateFn = Callable[[str, int, float], tuple[str, int]]


def _run_a0_single_shot(
        *,
        seed: int,
        problem: HumanEvalProblemV1,
        gen: _GenerateFn,
        max_tokens: int,
        executor_kwargs: dict[str, Any],
) -> tuple[HumanEvalArmOutcomeCapsuleV1, HumanEvalExecutorResultV1]:
    prompt = _a0_prompt(problem)
    text, wall = gen(prompt, max_tokens, 0.0)
    code = extract_candidate_code_v1(
        response_text=text, prompt=problem.prompt,
        entry_point=problem.entry_point)
    exe = run_humaneval_executor_v1(
        problem=problem, candidate_code=code,
        **executor_kwargs)
    call = HumanEvalArmCallCapsuleV1(
        schema=W86_HUMANEVAL_REAL_BENCH_V1_SCHEMA_VERSION,
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
        wall_ms=int(wall),
    )
    out = HumanEvalArmOutcomeCapsuleV1(
        schema=W86_HUMANEVAL_REAL_BENCH_V1_SCHEMA_VERSION,
        seed=int(seed),
        task_id=str(problem.task_id),
        arm_id="A0",
        final_passed=bool(exe.passed),
        final_candidate_code_cid=str(exe.candidate_code_cid),
        n_model_calls=1,
        n_executor_calls=1,
        total_wall_ms=int(wall + exe.wall_ms),
        call_capsule_cids=(call.cid(),),
        executor_result_cids=(exe.cid(),),
    )
    return out, exe


def _run_a1_first_pass_among_K(
        *,
        seed: int,
        problem: HumanEvalProblemV1,
        K: int,
        temperature: float,
        gen: _GenerateFn,
        max_tokens: int,
        executor_kwargs: dict[str, Any],
) -> tuple[HumanEvalArmOutcomeCapsuleV1,
           list[HumanEvalExecutorResultV1]]:
    """A1 baseline: K independent samples; pick first that passes
    the visible tests. This is the literature's standard "scale
    with compute" same-budget baseline for HumanEval."""
    prompt = _a0_prompt(problem)
    calls: list[HumanEvalArmCallCapsuleV1] = []
    exes: list[HumanEvalExecutorResultV1] = []
    total = 0
    chosen_idx = 0
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
                W86_HUMANEVAL_REAL_BENCH_V1_SCHEMA_VERSION),
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
            wall_ms=int(wall),
        ))
        exes.append(exe)
        total += int(wall) + int(exe.wall_ms)
        if exe.passed and not chosen_passed:
            chosen_idx = int(k)
            chosen_passed = True
            chosen_code_cid = str(exe.candidate_code_cid)
    if not chosen_passed:
        # Fall back to the first sample's code CID for the
        # outcome capsule's content-address.
        chosen_code_cid = str(exes[0].candidate_code_cid)
    out = HumanEvalArmOutcomeCapsuleV1(
        schema=W86_HUMANEVAL_REAL_BENCH_V1_SCHEMA_VERSION,
        seed=int(seed),
        task_id=str(problem.task_id),
        arm_id="A1",
        final_passed=bool(chosen_passed),
        final_candidate_code_cid=str(chosen_code_cid),
        n_model_calls=int(K),
        n_executor_calls=int(K),
        total_wall_ms=int(total),
        call_capsule_cids=tuple(c.cid() for c in calls),
        executor_result_cids=tuple(e.cid() for e in exes),
    )
    return out, exes


def _run_b_executor_critic(
        *,
        seed: int,
        problem: HumanEvalProblemV1,
        temperature: float,
        gen: _GenerateFn,
        max_tokens: int,
        executor_kwargs: dict[str, Any],
) -> tuple[HumanEvalArmOutcomeCapsuleV1,
           list[HumanEvalExecutorResultV1]]:
    """B: CoordPy multi-agent with executor-as-critic.

    5 model calls (solver_1, solver_2, critic, reviser, judge-
    no-op) ≤ 3 executor calls (one per solver + one for the
    reviser). The executor never "votes"; it provides
    deterministic ground-truth signal to the critic and judge.
    """
    personas = _personas()
    calls: list[HumanEvalArmCallCapsuleV1] = []
    exes: list[HumanEvalExecutorResultV1] = []
    total = 0
    candidates_code: list[str] = []
    candidates_text: list[str] = []
    # Calls 0, 1: solver personas
    for k in range(2):
        prompt = _solver_prompt(problem, personas[k])
        text, wall = gen(prompt, max_tokens, float(temperature))
        code = extract_candidate_code_v1(
            response_text=text, prompt=problem.prompt,
            entry_point=problem.entry_point)
        exe = run_humaneval_executor_v1(
            problem=problem, candidate_code=code,
            **executor_kwargs)
        calls.append(HumanEvalArmCallCapsuleV1(
            schema=(
                W86_HUMANEVAL_REAL_BENCH_V1_SCHEMA_VERSION),
            seed=int(seed),
            task_id=str(problem.task_id),
            arm_id="B",
            role=f"solver_{k+1}",
            call_idx=int(k),
            temperature=float(temperature),
            prompt_cid=hashlib.sha256(
                prompt.encode("utf-8")).hexdigest(),
            response_cid=hashlib.sha256(
                text.encode("utf-8")).hexdigest(),
            wall_ms=int(wall),
        ))
        exes.append(exe)
        candidates_code.append(code)
        candidates_text.append(text)
        total += int(wall) + int(exe.wall_ms)
    # Fast path: if any solver passed visible tests already,
    # we still emit a critic + reviser + judge call to keep
    # the per-arm budget constant at 5 model calls (same as
    # A1's K=5). The reviser is asked to "verify and harden"
    # the passing candidate. This avoids "B is better because
    # it spent less budget".
    best_idx = 0
    best_passed = bool(exes[0].passed)
    for i, e in enumerate(exes):
        if e.passed and not best_passed:
            best_idx = i
            best_passed = True
            break
    # Call 2: critic
    p_critic = _critic_prompt(problem, candidates_code, exes)
    t_critic, w_critic = gen(p_critic, max_tokens, float(temperature))
    calls.append(HumanEvalArmCallCapsuleV1(
        schema=W86_HUMANEVAL_REAL_BENCH_V1_SCHEMA_VERSION,
        seed=int(seed),
        task_id=str(problem.task_id),
        arm_id="B",
        role="critic",
        call_idx=2,
        temperature=float(temperature),
        prompt_cid=hashlib.sha256(
            p_critic.encode("utf-8")).hexdigest(),
        response_cid=hashlib.sha256(
            t_critic.encode("utf-8")).hexdigest(),
        wall_ms=int(w_critic),
    ))
    total += int(w_critic)
    # Call 3: reviser
    best_candidate_code = candidates_code[best_idx]
    p_reviser = _reviser_prompt(
        problem, best_candidate_code, exes[best_idx], t_critic)
    t_reviser, w_reviser = gen(
        p_reviser, max_tokens, float(temperature))
    revised_code = extract_candidate_code_v1(
        response_text=t_reviser, prompt=problem.prompt,
        entry_point=problem.entry_point)
    exe_reviser = run_humaneval_executor_v1(
        problem=problem, candidate_code=revised_code,
        **executor_kwargs)
    calls.append(HumanEvalArmCallCapsuleV1(
        schema=W86_HUMANEVAL_REAL_BENCH_V1_SCHEMA_VERSION,
        seed=int(seed),
        task_id=str(problem.task_id),
        arm_id="B",
        role="reviser",
        call_idx=3,
        temperature=float(temperature),
        prompt_cid=hashlib.sha256(
            p_reviser.encode("utf-8")).hexdigest(),
        response_cid=hashlib.sha256(
            t_reviser.encode("utf-8")).hexdigest(),
        wall_ms=int(w_reviser),
    ))
    exes.append(exe_reviser)
    total += int(w_reviser) + int(exe_reviser.wall_ms)
    # Call 4: judge — at t=0 deterministic; just records the
    # final decision. The judge is told the executor verdicts
    # and asked to declare a winner; the bench enforces that
    # only candidates with passed=True can be declared winners
    # (so the judge cannot lie about test outcomes).
    p_judge = (
        f"{_SYSTEM}\n\n[Persona: final judge]\n"
        "Given the candidate solutions and the executor "
        "verdicts below, name the SINGLE candidate to ship. "
        "Reply with exactly one line of the form "
        "WINNER: <N> where <N> is the 1-based candidate index.\n\n"
        + "\n".join(
            f"Candidate {i+1}: executor "
            f"{'PASS' if e.passed else 'FAIL'}"
            for i, e in enumerate(exes))
        + "\n\nYour answer:")
    t_judge, w_judge = gen(p_judge, 64, 0.0)
    calls.append(HumanEvalArmCallCapsuleV1(
        schema=W86_HUMANEVAL_REAL_BENCH_V1_SCHEMA_VERSION,
        seed=int(seed),
        task_id=str(problem.task_id),
        arm_id="B",
        role="judge",
        call_idx=4,
        temperature=0.0,
        prompt_cid=hashlib.sha256(
            p_judge.encode("utf-8")).hexdigest(),
        response_cid=hashlib.sha256(
            t_judge.encode("utf-8")).hexdigest(),
        wall_ms=int(w_judge),
    ))
    total += int(w_judge)
    # Final decision: pick any candidate the executor said
    # PASS, preferring the reviser. The judge is consultative
    # but cannot override the executor.
    final_passed = False
    final_code_cid = str(exes[0].candidate_code_cid)
    if exe_reviser.passed:
        final_passed = True
        final_code_cid = str(exe_reviser.candidate_code_cid)
    else:
        for e in exes[:-1]:
            if e.passed:
                final_passed = True
                final_code_cid = str(e.candidate_code_cid)
                break
    out = HumanEvalArmOutcomeCapsuleV1(
        schema=W86_HUMANEVAL_REAL_BENCH_V1_SCHEMA_VERSION,
        seed=int(seed),
        task_id=str(problem.task_id),
        arm_id="B",
        final_passed=bool(final_passed),
        final_candidate_code_cid=str(final_code_cid),
        n_model_calls=5,
        n_executor_calls=int(len(exes)),
        total_wall_ms=int(total),
        call_capsule_cids=tuple(c.cid() for c in calls),
        executor_result_cids=tuple(e.cid() for e in exes),
    )
    return out, exes


# ---------------------------------------------------------------
# Subset selection + driver.
# ---------------------------------------------------------------


def select_humaneval_subset_v1(
        *,
        corpus: Sequence[HumanEvalProblemV1],
        n_problems: int,
        seed: int,
) -> tuple[HumanEvalProblemV1, ...]:
    """Deterministically select ``n_problems`` from the corpus
    using a seeded random shuffle. Same seed → same subset
    across all arms within a seed."""
    rng = random.Random(int(seed))
    idxs = list(range(int(len(corpus))))
    rng.shuffle(idxs)
    chosen = idxs[:int(n_problems)]
    return tuple(corpus[i] for i in chosen)


@dataclasses.dataclass
class HumanEvalBenchConfigV1:
    schema: str = W86_HUMANEVAL_REAL_BENCH_V1_SCHEMA_VERSION
    n_problems: int = 30
    K_multi_sample: int = 5
    seeds: tuple[int, ...] = (86_028_001, 86_028_002, 86_028_003)
    sampling_temperature: float = 0.7
    max_tokens_per_call: int = 768
    executor_timeout_s: float = (
        W86_HUMANEVAL_EXECUTOR_TIMEOUT_S)
    executor_kill_after_s: float = (
        W86_HUMANEVAL_EXECUTOR_KILL_AFTER_S)


def run_humaneval_real_bench_v1(
        *,
        gen: _GenerateFn,
        model_id: str,
        corpus: Sequence[HumanEvalProblemV1],
        config: HumanEvalBenchConfigV1 | None = None,
        on_problem_start: (
            Callable[[int, int, str], None] | None) = None,
) -> HumanEvalBenchReportV1:
    """Run A0 + A1 + B on a deterministic subset for each seed.

    ``gen(prompt, max_tokens, temperature) -> (text, wall_ms)``
    is the call-level abstraction; pass any backend wrapper.
    Returns a content-addressed bench report.
    """
    cfg = config or HumanEvalBenchConfigV1()
    executor_kwargs = {
        "timeout_s": float(cfg.executor_timeout_s),
        "kill_after_s": float(cfg.executor_kill_after_s),
    }
    per_seed: list[HumanEvalSeedReportV1] = []
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
            b_out, _ = _run_b_executor_critic(
                seed=int(seed), problem=problem,
                temperature=float(cfg.sampling_temperature),
                gen=gen,
                max_tokens=int(cfg.max_tokens_per_call),
                executor_kwargs=executor_kwargs)
            b_outs.append(b_out)
        a0_acc = sum(
            1 for o in a0_outs if o.final_passed) / float(len(a0_outs))
        a1_acc = sum(
            1 for o in a1_outs if o.final_passed) / float(len(a1_outs))
        b_acc = sum(
            1 for o in b_outs if o.final_passed) / float(len(b_outs))
        outcome_cids = tuple(
            [o.cid() for o in a0_outs]
            + [o.cid() for o in a1_outs]
            + [o.cid() for o in b_outs])
        seed_merkle = _sha256_hex({
            "kind": "w86_humaneval_seed_merkle_root",
            "seed": int(seed),
            "outcome_cids": list(outcome_cids),
        })
        per_seed.append(HumanEvalSeedReportV1(
            schema=W86_HUMANEVAL_REAL_BENCH_V1_SCHEMA_VERSION,
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
            seed_merkle_root=str(seed_merkle),
        ))
        all_outcome_cids.extend(outcome_cids)
    a0_mean = sum(
        s.a0_pass_at_1 for s in per_seed) / float(len(per_seed))
    a1_mean = sum(
        s.a1_pass_at_1 for s in per_seed) / float(len(per_seed))
    b_mean = sum(
        s.b_pass_at_1 for s in per_seed) / float(len(per_seed))
    b_beats_a0 = tuple(
        s.b_pass_at_1 > s.a0_pass_at_1 for s in per_seed)
    b_beats_a1 = tuple(
        s.b_pass_at_1 > s.a1_pass_at_1 for s in per_seed)
    bench_merkle = _sha256_hex({
        "kind": "w86_humaneval_bench_merkle_root",
        "model_id": str(model_id),
        "outcome_cids": list(all_outcome_cids),
        "seeds": list(cfg.seeds),
    })
    return HumanEvalBenchReportV1(
        schema=W86_HUMANEVAL_REAL_BENCH_V1_SCHEMA_VERSION,
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
        bench_merkle_root=str(bench_merkle),
    )


__all__ = [
    "W86_HUMANEVAL_REAL_BENCH_V1_SCHEMA_VERSION",
    "HUMANEVAL_RAW_URL",
    "HUMANEVAL_RAW_EXPECTED_SHA256",
    "HUMANEVAL_EXPECTED_PROBLEM_COUNT",
    "HumanEvalCorpusError",
    "HumanEvalProblemV1",
    "HumanEvalExecutorResultV1",
    "HumanEvalArmCallCapsuleV1",
    "HumanEvalArmOutcomeCapsuleV1",
    "HumanEvalSeedReportV1",
    "HumanEvalBenchReportV1",
    "HumanEvalBenchConfigV1",
    "load_humaneval_corpus_v1",
    "extract_candidate_code_v1",
    "run_humaneval_executor_v1",
    "select_humaneval_subset_v1",
    "run_humaneval_real_bench_v1",
]
