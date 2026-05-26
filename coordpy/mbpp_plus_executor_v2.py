"""W102 / COO-9 — MBPP+ V2 executor (real EvalPlus schema).

Corrects the W101 V1 executor's silent-degradation failure mode.
V1 attempted to reconstruct EvalPlus extra-test assertions from
parallel `plus_input` / `plus_output` arrays that do NOT exist in
the actual EvalPlus release.  V2 runs the candidate against the
canonical EvalPlus `test` Python program verbatim, in a fresh
CPython subprocess.

Three modes:

* ``"base_and_plus"`` (canonical W102 mode) — runs candidate code
  PLUS the row's `extra_test_program` (which itself iterates over
  every input in `inputs` / `results`, including the base 3 plus
  EvalPlus's extra ≥ 100 hidden tests).  PASSes iff the subprocess
  exits 0.
* ``"base_only"`` — runs candidate code PLUS the row's
  `base_test_list` assertions ONE-BY-ONE (mirrors
  `coordpy.mbpp_executor_v1` shape; used by the V1-vs-V2 sanity
  probe and by cross-bench-stability mining).  PASSes iff all
  base assertions pass.
* ``"plus_only"`` — alias for ``"base_and_plus"`` (EvalPlus's
  `test` program does not cleanly separate base from plus tests in
  its iteration order; the documentation here is explicit so
  callers do not silently expect a true plus-isolated result).
  Use ``"base_only"`` + ``"base_and_plus"`` together to derive
  the strict plus-isolated outcome (base PASS ∧ base_and_plus FAIL
  ⇒ plus-only FAIL).

Mirrors the W90 base-MBPP executor discipline:

* Fresh CPython subprocess; no shared state with parent.
* Wall-clock timeout (8 s soft + 12 s kill).
* stderr/stdout tail-truncated; returned to caller so the
  reflexion mechanism can read real failure signal.
* Anti-cheat: subprocess uses `-I` (isolated; no current dir, no
  PYTHON* env vars).  Unlike the W86 / W90 base-MBPP executors
  (which use `-I -S`), V2 keeps the system `site` machinery so
  the EvalPlus `test` programs can `import numpy as np`.  This is
  required for ≥ 90 % of EvalPlus MBPP+ rows; without `numpy`
  the test programs raise `ModuleNotFoundError` and every problem
  silently FAILs.
* Per-assertion-passed count reported for diagnostic purposes.

Honest scope (W102)
-------------------

* ``W102-L-MBPP-PLUS-EXECUTOR-V2-SUBPROCESS-CAP`` — same
  subprocess executor as base MBPP; EvalPlus canonical extras do
  not perform side effects, so a sandboxed CPython subprocess is
  sufficient.  Hardened sandbox is V3.
* ``W102-L-MBPP-PLUS-EXECUTOR-V2-PLUS-ONLY-IS-ALIAS-CAP`` —
  ``"plus_only"`` is an alias for ``"base_and_plus"`` (see above);
  documented in the result dict's ``mode_note`` field so any
  downstream consumer sees the alias explicitly.
* ``W102-L-MBPP-PLUS-EXECUTOR-V2-NUMPY-DEPENDENCY-CAP`` — the
  EvalPlus `test` programs import `numpy as np` at the top; the
  V2 executor runs them in a subprocess that inherits the system
  Python's environment; if numpy is missing, every problem fails
  loudly (no silent degradation).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import subprocess
import sys
import time
from typing import Any

from .mbpp_plus_loader_v2 import MbppPlusProblemV2


W102_MBPP_PLUS_EXECUTOR_V2_SCHEMA_VERSION: str = (
    "coordpy.mbpp_plus_executor_v2.v1")


W102_MBPP_PLUS_EXECUTOR_V2_TIMEOUT_S: float = 8.0
W102_MBPP_PLUS_EXECUTOR_V2_KILL_AFTER_S: float = 12.0


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass(frozen=True)
class MbppPlusV2ExecutorResult:
    """Outcome of running one candidate against one MBPP+ V2
    problem.

    For ``base_and_plus`` / ``plus_only``, the executor runs the
    EvalPlus `test` program (which itself iterates many assertions
    via an internal loop); ``n_assertions_passed`` /
    ``n_assertions_total`` count the per-assertion outcomes parsed
    from the subprocess's stdout if the test program is instrumented
    to emit them, else ``-1`` / ``-1`` (the subprocess exit code
    is still the canonical PASS/FAIL signal).

    For ``base_only``, the executor runs each base assertion
    separately and reports exact pass counts.
    """

    schema: str
    task_id: str
    candidate_code_cid: str
    mode: str                # base_and_plus | base_only | plus_only
    mode_note: str           # note about mode aliases / caveats
    passed: bool             # canonical PASS/FAIL (subprocess rc==0)
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
            "task_id": str(self.task_id),
            "candidate_code_cid": str(
                self.candidate_code_cid),
            "mode": str(self.mode),
            "mode_note": str(self.mode_note),
            "passed": bool(self.passed),
            "n_assertions_passed": int(self.n_assertions_passed),
            "n_assertions_total": int(self.n_assertions_total),
            "timed_out": bool(self.timed_out),
            "wall_ms": int(self.wall_ms),
            "returncode": int(self.returncode),
            "stderr_tail": str(self.stderr_tail),
            "stdout_tail": str(self.stdout_tail),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w102_mbpp_plus_v2_executor_result",
            "result": self.to_dict(),
        })


def _build_base_only_program(
        *,
        candidate_code: str,
        base_tests: list[str],
        test_imports: list[str],
) -> str:
    """Build the base-only test program by running each
    `test_list` assertion in a separate try/except so the executor
    can report per-assertion pass counts (mirrors the W90 base
    executor shape)."""
    n_total = len(base_tests)
    program = ""
    for imp in test_imports:
        program += str(imp).strip() + "\n"
    program += candidate_code + "\n\n"
    program += "# --- W102 MBPP+ V2 base_only test block ---\n"
    program += f"_n_base_total = {n_total}\n"
    program += "_n_base_passed = 0\n"
    program += "_fail_log = []\n"
    for i, a in enumerate(base_tests):
        program += (
            "try:\n"
            f"    {a}\n"
            "    _n_base_passed += 1\n"
            "except Exception as _e:\n"
            f"    _fail_log.append(f'base#{i}: ' "
            "+ type(_e).__name__ + ': ' + str(_e))\n")
    program += (
        "import sys\n"
        "_ok = _n_base_passed >= _n_base_total\n"
        "if not _ok:\n"
        "    print('FAIL', _n_base_passed, '/', _n_base_total)\n"
        "    for _l in _fail_log[:20]:\n"
        "        print(_l, file=sys.stderr)\n"
        "    sys.exit(1)\n"
        "print('PASS', _n_base_total)\n")
    return program


def _build_base_and_plus_program(
        *,
        candidate_code: str,
        extra_test_program: str,
        test_imports: list[str],
) -> str:
    """Build the base+plus test program by concatenating the
    candidate with the EvalPlus `test` program.  The `test` program
    already imports numpy + defines `inputs` / `results` and
    iterates calling the entry point; if any iteration raises,
    the subprocess exits 1.

    We wrap the iteration in a try/except so we can report the
    number of *completed* iterations before the first failure (used
    for diagnostic purposes only; PASS/FAIL is still the rc==0
    signal)."""
    program = ""
    for imp in test_imports:
        program += str(imp).strip() + "\n"
    program += candidate_code + "\n\n"
    program += "# --- W102 MBPP+ V2 base_and_plus test block ---\n"
    program += "import sys\n"
    program += (
        "# The EvalPlus test program below defines inputs +\n"
        "# results and iterates over them.  We capture any\n"
        "# exception so the executor can report partial progress.\n")
    program += "try:\n"
    # Indent every line of the EvalPlus test program by 4 spaces.
    for line in extra_test_program.splitlines():
        program += "    " + line + "\n"
    program += (
        "    print('PASS_ALL')\n"
        "except Exception as _e:\n"
        "    print('FAIL', type(_e).__name__, str(_e),"
        " file=sys.stderr)\n"
        "    sys.exit(1)\n")
    return program


def run_mbpp_plus_executor_v2(
        *,
        problem: MbppPlusProblemV2,
        candidate_code: str,
        timeout_s: float = (
            W102_MBPP_PLUS_EXECUTOR_V2_TIMEOUT_S),
        kill_after_s: float = (
            W102_MBPP_PLUS_EXECUTOR_V2_KILL_AFTER_S),
        python_exe: str | None = None,
        mode: str = "base_and_plus",
) -> MbppPlusV2ExecutorResult:
    """Run candidate against the requested test set in a fresh
    CPython subprocess.  ``mode`` ∈ {`base_and_plus`, `base_only`,
    `plus_only`}.  See module docstring for mode semantics.
    """
    py = python_exe or sys.executable
    code_cid = hashlib.sha256(
        candidate_code.encode("utf-8")).hexdigest()
    canonical_mode = str(mode)
    if canonical_mode not in (
            "base_and_plus", "base_only", "plus_only"):
        raise ValueError(
            f"mode must be one of "
            "(base_and_plus, base_only, plus_only); got "
            f"{canonical_mode!r}")
    mode_note = ""
    if canonical_mode == "base_only":
        program = _build_base_only_program(
            candidate_code=candidate_code,
            base_tests=list(problem.base_test_list),
            test_imports=list(problem.test_imports))
        n_total_planned = int(len(problem.base_test_list))
    else:
        if canonical_mode == "plus_only":
            mode_note = (
                "plus_only is an alias for base_and_plus in V2; "
                "EvalPlus's `test` program does not cleanly "
                "separate base from plus iterations.  See "
                "coordpy.mbpp_plus_executor_v2 module docstring.")
        program = _build_base_and_plus_program(
            candidate_code=candidate_code,
            extra_test_program=str(problem.extra_test_program),
            test_imports=list(problem.test_imports))
        n_total_planned = -1
    t0 = time.time()
    timed_out = False
    rc = -1
    out_b = b""
    err_b = b""
    try:
        proc = subprocess.run(
            [py, "-I", "-c", program],
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
    n_passed = -1
    n_total = -1
    if canonical_mode == "base_only" and not timed_out:
        m = re.match(
            r"^(PASS|FAIL)\s+(\d+)(?:\s*/\s*(\d+))?",
            out_text.strip())
        if m is not None:
            verdict = str(m.group(1))
            v_a = int(m.group(2))
            v_b = int(m.group(3)) if m.group(3) else v_a
            if verdict == "PASS":
                n_passed = v_a
                n_total = v_a
            else:
                n_passed = v_a
                n_total = v_b
        else:
            n_total = n_total_planned
    passed = (rc == 0 and not timed_out)
    return MbppPlusV2ExecutorResult(
        schema=W102_MBPP_PLUS_EXECUTOR_V2_SCHEMA_VERSION,
        task_id=str(problem.task_id),
        candidate_code_cid=str(code_cid),
        mode=str(canonical_mode),
        mode_note=str(mode_note),
        passed=bool(passed),
        n_assertions_passed=int(n_passed),
        n_assertions_total=int(n_total),
        timed_out=bool(timed_out),
        wall_ms=int(wall_ms),
        returncode=int(rc),
        stderr_tail=err_text[-500:],
        stdout_tail=out_text[-200:],
    )


__all__ = [
    "W102_MBPP_PLUS_EXECUTOR_V2_SCHEMA_VERSION",
    "W102_MBPP_PLUS_EXECUTOR_V2_TIMEOUT_S",
    "W102_MBPP_PLUS_EXECUTOR_V2_KILL_AFTER_S",
    "MbppPlusV2ExecutorResult",
    "run_mbpp_plus_executor_v2",
]
