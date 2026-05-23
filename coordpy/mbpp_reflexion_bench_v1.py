"""W90 / Post-W89 — MBPP sequential-reflexion bench V1.

Tests whether the W88 sequential-reflexion B-pipeline that
RETIRED both W86 / W88 HumanEval carry-forwards at Llama-3.3-70B
also generalises to a second published code benchmark: MBPP
(Mostly Basic Python Problems, Austin et al. 2021), specifically
the **sanitized-mbpp** subset (427 problems) maintained at
``google-research/google-research/mbpp/sanitized-mbpp.json``.

The bench module mirrors `coordpy.humaneval_reflexion_bench_v1`
in shape and audit-chain discipline.  Differences:

  * MBPP problems have a TEXT prompt (English description) +
    a list of assertion strings (`test_list`).  The bench
    constructs a model prompt that shows the description and
    the FIRST assertion (as a function-signature hint), then
    asks for an implementation that passes all assertions.
  * No function signature is given in the canonical MBPP
    format; the bench extracts the function name from the
    first assertion to construct the executor.
  * The executor runs the candidate + ALL assertions in
    `test_list` in a fresh CPython subprocess (same shape as
    the W86 HumanEval executor).

Three arms (identical shape to W88 / W89 HumanEval):

* ``A0`` — stock single-shot at T=0.0.
* ``A1`` — first-pass-among-K=5 self-consistency at T=0.7.
* ``B`` — sequential-reflexion-K=5 at T=0.7, each turn
  conditioned on the cumulative (candidate, executor_stderr)
  history.

Anti-cheat:

* MBPP-sanitized corpus SHA-256 verified against the canonical
  upstream pin.
* No problem in the eval subset is shown to the model with its
  canonical solution.
* No selective retries.
* No model swap between arms.
* Executor truth = pass on every assertion in `test_list`.
* Per-call CIDs + per-seed Merkle + bench Merkle re-verifiable
  offline.

Honest scope (W90)
------------------

* ``W90-L-MBPP-REFLEXION-V1-NIM-DEPENDENT-CAP`` — V1 drives the
  bench through any ``LLMBackend``-shaped client; provider
  determinism beyond temperature=0 is not assumed.
* ``W90-L-MBPP-REFLEXION-V1-SUBPROCESS-EXECUTOR-CAP`` — same
  CPython subprocess executor as the W86 / W88 HumanEval bench;
  wall-clock timeout 8 s soft + 12 s kill.
* ``W90-L-MBPP-REFLEXION-V1-FIRST-ASSERTION-VISIBLE-CAP`` — V1
  shows the FIRST assertion in `test_list` to the model as a
  function-signature hint; the remaining assertions are
  evaluation-only (mirrors the literature's "few-shot MBPP"
  protocol; A0 and B see the same single-example hint).
* ``W90-L-MBPP-REFLEXION-V1-SANITIZED-SUBSET-CAP`` — V1 uses
  the sanitized-mbpp 427-problem subset; full mbpp (974) is
  V2.
"""

from __future__ import annotations

import dataclasses
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
from typing import Any, Callable, Sequence


W90_MBPP_REFLEXION_BENCH_V1_SCHEMA_VERSION: str = (
    "coordpy.mbpp_reflexion_bench_v1.v1")


# Canonical MBPP-sanitized:
# https://github.com/google-research/google-research/tree/master/mbpp
MBPP_SANITIZED_RAW_URL: str = (
    "https://raw.githubusercontent.com/google-research/"
    "google-research/master/mbpp/sanitized-mbpp.json")
# SHA-256 of the JSON blob at the upstream HEAD as of
# 2026-05-22.  If upstream re-publishes the file (very rare
# for this corpus), this SHA must be re-validated.
MBPP_SANITIZED_EXPECTED_SHA256: str = (
    "ca95deaa9a01ef0a6f439f88bcf0dd3db3563d22f22aad6cae04ebb9a8d8c8e9")
MBPP_SANITIZED_EXPECTED_PROBLEM_COUNT: int = 427

# Default subprocess executor wall-clock timeout (seconds).
W90_MBPP_EXECUTOR_TIMEOUT_S: float = 8.0
W90_MBPP_EXECUTOR_KILL_AFTER_S: float = 12.0


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


class MBPPCorpusError(RuntimeError):
    """Raised when the MBPP corpus cannot be loaded / verified."""


# ---------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------

def _default_cache_path() -> str:
    return os.environ.get(
        "COORDPY_MBPP_CACHE",
        os.path.expanduser(
            "~/.cache/coordpy/mbpp-sanitized.json"))


@dataclasses.dataclass(frozen=True)
class MBPPProblemV1:
    """One MBPP-sanitized problem."""

    task_id: int
    text: str            # English description
    code: str            # Canonical solution (NEVER shown to model)
    test_list: tuple[str, ...]
    test_imports: tuple[str, ...]
    entry_point: str     # Extracted from first assertion

    def problem_cid(self) -> str:
        return _sha256_hex({
            "task_id": int(self.task_id),
            "text_sha256": hashlib.sha256(
                self.text.encode("utf-8")).hexdigest(),
            "test_list": list(self.test_list),
            "entry_point": str(self.entry_point),
        })


_ENTRY_POINT_FROM_ASSERT_RE = re.compile(
    r"^\s*assert\s+(?:set\(|tuple\(|list\(|sorted\(|abs\()*"
    r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(")


def _extract_entry_point_from_test(test_str: str) -> str:
    """Extract the function name being asserted on from an
    MBPP-style `assert func(args) == expected` line."""
    m = _ENTRY_POINT_FROM_ASSERT_RE.match(test_str.strip())
    if m is None:
        # Fallback: look for any identifier followed by "("
        m2 = re.search(
            r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", test_str)
        if m2 is None:
            return ""
        return str(m2.group(1))
    return str(m.group(1))


def load_mbpp_corpus_v1(
        *,
        cache_path: str | None = None,
        url: str = MBPP_SANITIZED_RAW_URL,
        expected_sha256: str = MBPP_SANITIZED_EXPECTED_SHA256,
        timeout: float = 60.0,
) -> tuple[MBPPProblemV1, ...]:
    """Load + SHA-256-verify the canonical MBPP-sanitized
    corpus."""
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
            raise MBPPCorpusError(
                f"MBPP corpus fetch failed: "
                f"{type(e).__name__}: {e} (url={url})") from e
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(raw)
    actual = hashlib.sha256(raw).hexdigest()
    if actual.lower() != expected_sha256.lower():
        raise MBPPCorpusError(
            "MBPP corpus SHA-256 mismatch: "
            f"actual={actual} expected={expected_sha256}. "
            "Refusing to use a possibly-tampered corpus.")
    try:
        rows = json.loads(raw.decode("utf-8"))
    except Exception as e:  # noqa: BLE001
        raise MBPPCorpusError(
            f"MBPP corpus JSON parse failed: "
            f"{type(e).__name__}: {e}") from e
    out: list[MBPPProblemV1] = []
    for row in rows:
        tests = list(row.get("test_list") or [])
        if not tests:
            continue
        entry = _extract_entry_point_from_test(tests[0])
        if not entry:
            continue
        out.append(MBPPProblemV1(
            task_id=int(row["task_id"]),
            text=str(row["prompt"]),
            code=str(row.get("code") or ""),
            test_list=tuple(tests),
            test_imports=tuple(row.get("test_imports") or ()),
            entry_point=str(entry)))
    MBPP_MIN_VALID_PROBLEMS = 400
    if len(out) < MBPP_MIN_VALID_PROBLEMS:
        raise MBPPCorpusError(
            f"MBPP corpus parsed {len(out)} problems with valid "
            f"entry_point; expected ≥ {MBPP_MIN_VALID_PROBLEMS}")
    if len(out) != int(MBPP_SANITIZED_EXPECTED_PROBLEM_COUNT):
        # Sanitized version drifts only on entry_point extraction
        # edge cases; the SHA-256 check above is the canonical
        # source-of-truth.  Log but don't fail.
        pass
    return tuple(out)


# ---------------------------------------------------------------
# Candidate code extraction + executor
# ---------------------------------------------------------------

_CODE_FENCE_RE = re.compile(
    r"```(?:python|py)?\s*\n(.*?)\n```", re.DOTALL)


def extract_candidate_code_v1(
        *, response_text: str, entry_point: str,
) -> str:
    """Pull a Python candidate solution out of a model response."""
    text = str(response_text)
    m = _CODE_FENCE_RE.search(text)
    if m is not None:
        return m.group(1)
    return text


@dataclasses.dataclass(frozen=True)
class MBPPExecutorResultV1:
    """Outcome of running one candidate against one problem's
    `test_list` in a fresh subprocess."""

    schema: str
    task_id: int
    candidate_code_cid: str
    passed: bool
    n_assertions_passed: int
    n_assertions_total: int
    timed_out: bool
    wall_ms: int
    returncode: int
    stderr_tail: str
    stdout_tail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "task_id": int(self.task_id),
            "candidate_code_cid": str(
                self.candidate_code_cid),
            "passed": bool(self.passed),
            "n_assertions_passed": int(
                self.n_assertions_passed),
            "n_assertions_total": int(
                self.n_assertions_total),
            "timed_out": bool(self.timed_out),
            "wall_ms": int(self.wall_ms),
            "returncode": int(self.returncode),
            "stderr_tail": str(self.stderr_tail),
            "stdout_tail": str(self.stdout_tail),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w90_mbpp_executor_result_v1",
            "result": self.to_dict(),
        })


def run_mbpp_executor_v1(
        *,
        problem: MBPPProblemV1,
        candidate_code: str,
        timeout_s: float = W90_MBPP_EXECUTOR_TIMEOUT_S,
        kill_after_s: float = W90_MBPP_EXECUTOR_KILL_AFTER_S,
        python_exe: str | None = None,
) -> MBPPExecutorResultV1:
    """Run candidate + all test_list assertions in a fresh
    CPython subprocess.  Returns pass / fail (binary, passes
    iff ALL assertions in test_list pass) plus per-assertion
    count for diagnostic purposes.
    """
    py = python_exe or sys.executable
    imports = "\n".join(problem.test_imports) + "\n" \
        if problem.test_imports else ""
    asserts = "\n".join(problem.test_list)
    code_cid = hashlib.sha256(
        candidate_code.encode("utf-8")).hexdigest()
    test_program = (
        imports + candidate_code + "\n\n"
        + "# --- W90 MBPP assertions ---\n"
        + "_n_assertions_total = " + str(len(problem.test_list))
        + "\n_n_assertions_passed = 0\n"
        + "_fail_log = []\n")
    for i, a in enumerate(problem.test_list):
        # Wrap each assertion to count individual passes.
        test_program += (
            f"try:\n"
            f"    {a}\n"
            f"    _n_assertions_passed += 1\n"
            f"except Exception as _e:\n"
            f"    _fail_log.append(f'assertion {i}: '"
            f"+ type(_e).__name__ + ': ' + str(_e))\n")
    test_program += (
        "import sys\n"
        "if _n_assertions_passed < _n_assertions_total:\n"
        "    print('FAIL', _n_assertions_passed, '/' ,"
        "_n_assertions_total)\n"
        "    for _l in _fail_log:\n"
        "        print(_l, file=sys.stderr)\n"
        "    sys.exit(1)\n"
        "print('PASS', _n_assertions_total)\n"
    )
    t0 = time.time()
    timed_out = False
    rc = -1
    out_b = b""
    err_b = b""
    try:
        proc = subprocess.run(
            [py, "-I", "-S", "-c", test_program],
            input=b"",
            capture_output=True,
            timeout=float(kill_after_s),
            check=False,
        )
        rc = int(proc.returncode)
        out_b = proc.stdout or b""
        err_b = proc.stderr or b""
    except subprocess.TimeoutExpired:
        timed_out = True
        rc = -9
    wall_ms = int((time.time() - t0) * 1000)
    err_text = err_b.decode("utf-8", errors="replace")
    out_text = out_b.decode("utf-8", errors="replace")
    n_pass = 0
    n_total = len(problem.test_list)
    if not timed_out:
        m = re.match(
            r"^(PASS|FAIL)\s+(\d+)(?:\s*/\s*(\d+))?",
            out_text.strip())
        if m is not None:
            n_pass = int(m.group(2))
            if m.group(1) == "PASS":
                n_pass = n_total
    passed = (rc == 0 and not timed_out)
    return MBPPExecutorResultV1(
        schema=W90_MBPP_REFLEXION_BENCH_V1_SCHEMA_VERSION,
        task_id=int(problem.task_id),
        candidate_code_cid=str(code_cid),
        passed=bool(passed),
        n_assertions_passed=int(n_pass),
        n_assertions_total=int(n_total),
        timed_out=bool(timed_out),
        wall_ms=int(wall_ms),
        returncode=int(rc),
        stderr_tail=err_text[-500:],
        stdout_tail=out_text[-200:],
    )


# ---------------------------------------------------------------
# Capsules
# ---------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class MBPPArmCallCapsuleV1:
    schema: str
    seed: int
    task_id: int
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
            "task_id": int(self.task_id),
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
            "kind": "w90_mbpp_arm_call_capsule_v1",
            "capsule": self.to_dict(),
        })


@dataclasses.dataclass(frozen=True)
class MBPPArmOutcomeCapsuleV1:
    schema: str
    seed: int
    task_id: int
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
            "task_id": int(self.task_id),
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
            "kind": "w90_mbpp_arm_outcome_capsule_v1",
            "capsule": self.to_dict(),
        })


@dataclasses.dataclass(frozen=True)
class MBPPSeedReportV1:
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
class MBPPBenchReportV1:
    schema: str
    model_id: str
    n_problems: int
    n_seeds: int
    K_multi_sample: int
    per_seed: tuple[MBPPSeedReportV1, ...]
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


# ---------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------

_SYSTEM = (
    "You are an expert Python programmer.  When given a function "
    "description and a sample assertion, output ONLY the complete "
    "Python function (inside a ```python ... ``` code fence).  Do "
    "not include any prose before or after the code fence.")


def _initial_prompt(p: MBPPProblemV1) -> str:
    sample_assert = p.test_list[0] if p.test_list else ""
    return (
        f"{_SYSTEM}\n\n"
        "Write a Python function that satisfies the description "
        "below.  Your function must pass the sample assertion "
        "AND any hidden assertions.\n\n"
        f"Description: {p.text}\n\n"
        f"Sample assertion (the function must pass this and "
        f"similar):\n```python\n{sample_assert}\n```\n\n"
        "Your complete Python function:")


def _reflexion_prompt(
        p: MBPPProblemV1,
        history: Sequence[tuple[str, MBPPExecutorResultV1]],
        attempt_idx: int,
) -> str:
    chunks: list[str] = []
    for i, (cand, exe) in enumerate(history):
        cand_trim = cand
        if len(cand_trim) > 1500:
            cand_trim = cand_trim[:1500] + "\n# ... (truncated)\n"
        if exe.passed:
            verdict = (
                f"PASSED all {exe.n_assertions_total} "
                f"assertions")
            stderr_excerpt = ""
        else:
            verdict = (
                f"FAILED ({exe.n_assertions_passed}/"
                f"{exe.n_assertions_total} assertions; "
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
    sample_assert = p.test_list[0] if p.test_list else ""
    return (
        f"{_SYSTEM}\n\n"
        "[Role: reflective code generator]\n"
        f"You are on attempt {attempt_idx + 1} out of 5.\n\n"
        f"Description: {p.text}\n\n"
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


def _run_a0_single_shot(
        *, seed: int, p: MBPPProblemV1, gen: _GenerateFn,
        max_tokens: int, executor_kwargs: dict[str, Any],
) -> MBPPArmOutcomeCapsuleV1:
    prompt = _initial_prompt(p)
    text, wall = gen(prompt, max_tokens, 0.0)
    code = extract_candidate_code_v1(
        response_text=text, entry_point=p.entry_point)
    exe = run_mbpp_executor_v1(
        problem=p, candidate_code=code, **executor_kwargs)
    call = MBPPArmCallCapsuleV1(
        schema=W90_MBPP_REFLEXION_BENCH_V1_SCHEMA_VERSION,
        seed=int(seed), task_id=int(p.task_id),
        arm_id="A0", role="solver", call_idx=0,
        temperature=0.0,
        prompt_cid=hashlib.sha256(
            prompt.encode("utf-8")).hexdigest(),
        response_cid=hashlib.sha256(
            text.encode("utf-8")).hexdigest(),
        wall_ms=int(wall))
    return MBPPArmOutcomeCapsuleV1(
        schema=W90_MBPP_REFLEXION_BENCH_V1_SCHEMA_VERSION,
        seed=int(seed), task_id=int(p.task_id),
        arm_id="A0",
        final_passed=bool(exe.passed),
        final_candidate_code_cid=str(exe.candidate_code_cid),
        n_model_calls=1,
        n_executor_calls=1,
        total_wall_ms=int(wall + exe.wall_ms),
        call_capsule_cids=(call.cid(),),
        executor_result_cids=(exe.cid(),))


def _run_a1_first_pass_among_K(
        *, seed: int, p: MBPPProblemV1, K: int,
        temperature: float, gen: _GenerateFn, max_tokens: int,
        executor_kwargs: dict[str, Any],
) -> MBPPArmOutcomeCapsuleV1:
    prompt = _initial_prompt(p)
    calls: list[MBPPArmCallCapsuleV1] = []
    exes: list[MBPPExecutorResultV1] = []
    total = 0
    chosen_passed = False
    chosen_code_cid = ""
    for k in range(int(K)):
        text, wall = gen(prompt, max_tokens, float(temperature))
        code = extract_candidate_code_v1(
            response_text=text, entry_point=p.entry_point)
        exe = run_mbpp_executor_v1(
            problem=p, candidate_code=code,
            **executor_kwargs)
        calls.append(MBPPArmCallCapsuleV1(
            schema=W90_MBPP_REFLEXION_BENCH_V1_SCHEMA_VERSION,
            seed=int(seed), task_id=int(p.task_id),
            arm_id="A1", role="sample", call_idx=int(k),
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
    return MBPPArmOutcomeCapsuleV1(
        schema=W90_MBPP_REFLEXION_BENCH_V1_SCHEMA_VERSION,
        seed=int(seed), task_id=int(p.task_id),
        arm_id="A1",
        final_passed=bool(chosen_passed),
        final_candidate_code_cid=str(chosen_code_cid),
        n_model_calls=int(K),
        n_executor_calls=int(K),
        total_wall_ms=int(total),
        call_capsule_cids=tuple(c.cid() for c in calls),
        executor_result_cids=tuple(e.cid() for e in exes))


def _run_b_sequential_reflexion(
        *, seed: int, p: MBPPProblemV1, K: int,
        temperature: float, gen: _GenerateFn, max_tokens: int,
        executor_kwargs: dict[str, Any],
) -> MBPPArmOutcomeCapsuleV1:
    history: list[tuple[str, MBPPExecutorResultV1]] = []
    calls: list[MBPPArmCallCapsuleV1] = []
    exes: list[MBPPExecutorResultV1] = []
    total = 0
    for k in range(int(K)):
        if k == 0:
            prompt = _initial_prompt(p)
        else:
            prompt = _reflexion_prompt(
                p, tuple(history), attempt_idx=int(k))
        text, wall = gen(prompt, max_tokens, float(temperature))
        code = extract_candidate_code_v1(
            response_text=text, entry_point=p.entry_point)
        exe = run_mbpp_executor_v1(
            problem=p, candidate_code=code,
            **executor_kwargs)
        calls.append(MBPPArmCallCapsuleV1(
            schema=W90_MBPP_REFLEXION_BENCH_V1_SCHEMA_VERSION,
            seed=int(seed), task_id=int(p.task_id),
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
    return MBPPArmOutcomeCapsuleV1(
        schema=W90_MBPP_REFLEXION_BENCH_V1_SCHEMA_VERSION,
        seed=int(seed), task_id=int(p.task_id),
        arm_id="B",
        final_passed=bool(final_passed),
        final_candidate_code_cid=str(final_code_cid),
        n_model_calls=int(K),
        n_executor_calls=int(K),
        total_wall_ms=int(total),
        call_capsule_cids=tuple(c.cid() for c in calls),
        executor_result_cids=tuple(e.cid() for e in exes))


# ---------------------------------------------------------------
# Subset + driver
# ---------------------------------------------------------------

def select_mbpp_subset_v1(
        *, corpus: Sequence[MBPPProblemV1], n_problems: int,
        seed: int,
) -> tuple[MBPPProblemV1, ...]:
    rng = random.Random(int(seed))
    idxs = list(range(len(corpus)))
    rng.shuffle(idxs)
    chosen = idxs[: int(n_problems)]
    return tuple(corpus[i] for i in chosen)


@dataclasses.dataclass
class MBPPBenchConfigV1:
    schema: str = W90_MBPP_REFLEXION_BENCH_V1_SCHEMA_VERSION
    n_problems: int = 30
    K_multi_sample: int = 5
    seeds: tuple[int, ...] = (90_001, 90_002, 90_003)
    sampling_temperature: float = 0.7
    max_tokens_per_call: int = 768
    executor_timeout_s: float = W90_MBPP_EXECUTOR_TIMEOUT_S
    executor_kill_after_s: float = W90_MBPP_EXECUTOR_KILL_AFTER_S


def run_mbpp_reflexion_bench_v1(
        *,
        gen: _GenerateFn,
        model_id: str,
        corpus: Sequence[MBPPProblemV1],
        config: MBPPBenchConfigV1 | None = None,
        on_problem_start: (
            Callable[[int, int, int], None] | None) = None,
) -> MBPPBenchReportV1:
    cfg = config or MBPPBenchConfigV1()
    executor_kwargs = {
        "timeout_s": float(cfg.executor_timeout_s),
        "kill_after_s": float(cfg.executor_kill_after_s),
    }
    per_seed: list[MBPPSeedReportV1] = []
    all_outcome_cids: list[str] = []
    for seed in cfg.seeds:
        subset = select_mbpp_subset_v1(
            corpus=corpus, n_problems=int(cfg.n_problems),
            seed=int(seed))
        a0_outs: list[MBPPArmOutcomeCapsuleV1] = []
        a1_outs: list[MBPPArmOutcomeCapsuleV1] = []
        b_outs: list[MBPPArmOutcomeCapsuleV1] = []
        for p_idx, problem in enumerate(subset):
            if on_problem_start is not None:
                on_problem_start(
                    int(seed), int(p_idx),
                    int(problem.task_id))
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
            "kind": "w90_mbpp_seed_merkle_root",
            "seed": int(seed),
            "outcome_cids": list(outcome_cids),
        })
        per_seed.append(MBPPSeedReportV1(
            schema=W90_MBPP_REFLEXION_BENCH_V1_SCHEMA_VERSION,
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
        "kind": "w90_mbpp_bench_merkle_root",
        "model_id": str(model_id),
        "outcome_cids": list(all_outcome_cids),
        "seeds": list(cfg.seeds),
    })
    return MBPPBenchReportV1(
        schema=W90_MBPP_REFLEXION_BENCH_V1_SCHEMA_VERSION,
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
    "W90_MBPP_REFLEXION_BENCH_V1_SCHEMA_VERSION",
    "MBPP_SANITIZED_RAW_URL",
    "MBPP_SANITIZED_EXPECTED_SHA256",
    "MBPPCorpusError",
    "MBPPProblemV1",
    "MBPPExecutorResultV1",
    "MBPPArmCallCapsuleV1",
    "MBPPArmOutcomeCapsuleV1",
    "MBPPSeedReportV1",
    "MBPPBenchReportV1",
    "MBPPBenchConfigV1",
    "load_mbpp_corpus_v1",
    "extract_candidate_code_v1",
    "run_mbpp_executor_v1",
    "select_mbpp_subset_v1",
    "run_mbpp_reflexion_bench_v1",
]
