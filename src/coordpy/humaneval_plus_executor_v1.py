"""W102 / COO-9 — HumanEval+ V1 executor.

Runs a candidate Python program against the EvalPlus HumanEval+
hidden tests in a fresh CPython subprocess.

The HumanEval+ row schema carries a ``test`` field that defines a
``check(candidate)`` function which executes the EvalPlus hidden
tests against the candidate's entry-point function.  This executor
constructs a subprocess program that:

  1. Concatenates the candidate code,
  2. Inlines the row's `test` block,
  3. Calls `check(<entry_point>)`.

A candidate PASSes iff the subprocess exits 0.

Mirrors the W86 base-HumanEval executor discipline:

* Fresh CPython subprocess; no shared state with parent.
* Wall-clock timeout (15 s soft + 20 s kill; bigger than base
  HumanEval because the EvalPlus extra-test surface iterates
  over ~1500-2500 hidden inputs).
* stderr/stdout tail-truncated; returned to caller so the
  reflexion mechanism can read real failure signal.
* Anti-cheat: subprocess uses `-I` (isolated; no current dir, no
  PYTHON* env vars).  Unlike W86 base-HumanEval (which uses `-I
  -S`), V1 HumanEval+ keeps the system `site` machinery so the
  test programs can `import numpy as np`.

Honest scope (W102)
-------------------

* ``W102-L-HUMANEVAL-PLUS-EXECUTOR-V1-NUMPY-DEPENDENCY-CAP`` —
  see loader module docstring.
* ``W102-L-HUMANEVAL-PLUS-EXECUTOR-V1-LONGER-TIMEOUT-CAP`` —
  default wall timeout is 15 s soft / 20 s kill (vs W86's
  8 s / 12 s) because the EvalPlus extra-test surface has
  ~80× more iterations.  Candidate programs with O(n^2)
  algorithms on small inputs still finish well within budget;
  exponential candidates time out cleanly.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
import sys
import time
from typing import Any

from .humaneval_plus_loader_v1 import HumanEvalPlusProblemV1


W102_HUMANEVAL_PLUS_EXECUTOR_V1_SCHEMA_VERSION: str = (
    "coordpy.humaneval_plus_executor_v1.v1")


W102_HUMANEVAL_PLUS_EXECUTOR_V1_TIMEOUT_S: float = 15.0
W102_HUMANEVAL_PLUS_EXECUTOR_V1_KILL_AFTER_S: float = 20.0


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass(frozen=True)
class HumanEvalPlusExecutorResultV1:
    """Outcome of running one candidate against one HumanEval+
    problem's `check()` block in a fresh subprocess."""

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
            "kind": "w102_humaneval_plus_executor_result_v1",
            "result": self.to_dict(),
        })


def run_humaneval_plus_executor_v1(
        *,
        problem: HumanEvalPlusProblemV1,
        candidate_code: str,
        timeout_s: float = (
            W102_HUMANEVAL_PLUS_EXECUTOR_V1_TIMEOUT_S),
        kill_after_s: float = (
            W102_HUMANEVAL_PLUS_EXECUTOR_V1_KILL_AFTER_S),
        python_exe: str | None = None,
) -> HumanEvalPlusExecutorResultV1:
    """Run candidate + the row's `test` block in a fresh CPython
    subprocess.  The subprocess defines `check(candidate)` and
    invokes `check(<entry_point>)`; exits 0 iff every assertion
    in `check` passes.
    """
    py = python_exe or sys.executable
    test_program = (
        candidate_code + "\n\n"
        + problem.test + "\n\n"
        + f"check({problem.entry_point})\n")
    code_cid = hashlib.sha256(
        candidate_code.encode("utf-8")).hexdigest()
    t0 = time.time()
    timed_out = False
    rc = -1
    out_b = b""
    err_b = b""
    try:
        proc = subprocess.run(
            [py, "-I", "-c", test_program],
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
    passed = (rc == 0 and not timed_out)
    return HumanEvalPlusExecutorResultV1(
        schema=(
            W102_HUMANEVAL_PLUS_EXECUTOR_V1_SCHEMA_VERSION),
        task_id=str(problem.task_id),
        candidate_code_cid=str(code_cid),
        passed=bool(passed),
        timed_out=bool(timed_out),
        wall_ms=int(wall_ms),
        returncode=int(rc),
        stderr_tail=err_text[-500:],
        stdout_tail=out_text[-200:],
    )


__all__ = [
    "W102_HUMANEVAL_PLUS_EXECUTOR_V1_SCHEMA_VERSION",
    "W102_HUMANEVAL_PLUS_EXECUTOR_V1_TIMEOUT_S",
    "W102_HUMANEVAL_PLUS_EXECUTOR_V1_KILL_AFTER_S",
    "HumanEvalPlusExecutorResultV1",
    "run_humaneval_plus_executor_v1",
]
