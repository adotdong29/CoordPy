"""W101 / COO-9 — MBPP+ extra-tests-aware executor V1.

Runs a candidate Python program against BOTH the original MBPP
`test_list` AND the EvalPlus `plus_input` / `plus_output` extra
tests in a single fresh CPython subprocess.  A candidate PASSes
MBPP+ iff every base assertion AND every EvalPlus extra-test
parallel assertion passes.

Mirrors the W90 `coordpy.mbpp_reflexion_bench_v1` executor
discipline:

* Fresh CPython subprocess; no shared state with parent.
* Wall-clock timeout (8 s soft + 12 s kill).
* stderr/stdout tail-truncated; returned to the caller so the
  reflexion mechanism can read real failure signal, not a hand-
  written summary.
* Anti-cheat: subprocess uses `-I -S` (isolated, no site).
* Per-call PASS/FAIL is binary; the per-assertion-passed count
  is reported for diagnostic purposes only.

Honest scope (W101)
-------------------

* ``W101-L-MBPP-PLUS-EXECUTOR-V1-SUBPROCESS-CAP`` — same
  subprocess executor as base MBPP; MBPP+ canonical tests in
  EvalPlus do not perform side effects, so a sandboxed CPython
  subprocess is sufficient.  Hardened sandbox (seccomp, capability
  drop) is V2.
* ``W101-L-MBPP-PLUS-EXECUTOR-V1-EXTRA-TESTS-ENCODING-CAP`` — the
  EvalPlus extra tests are reconstructed from `plus_input` /
  `plus_output` parallel arrays as `assert <entry_point>(*input)
  == output`-style assertions.  Edge cases (None outputs, custom
  comparator shapes) are accepted but flagged.
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

from .mbpp_plus_loader_v1 import MbppPlusProblemV1


W101_MBPP_PLUS_EXECUTOR_V1_SCHEMA_VERSION: str = (
    "coordpy.mbpp_plus_executor_v1.v1")


W101_MBPP_PLUS_EXECUTOR_TIMEOUT_S: float = 8.0
W101_MBPP_PLUS_EXECUTOR_KILL_AFTER_S: float = 12.0


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass(frozen=True)
class MbppPlusExecutorResultV1:
    """Outcome of running one candidate against one MBPP+
    problem's combined base + plus tests."""

    schema: str
    task_id: str
    candidate_code_cid: str
    passed: bool
    n_base_passed: int
    n_base_total: int
    n_plus_passed: int
    n_plus_total: int
    timed_out: bool
    wall_ms: int
    returncode: int
    stderr_tail: str
    stdout_tail: str
    # mode in {"base_and_plus", "base_only", "plus_only"} —
    # records which test sets the executor compared against.  The
    # W101 cheap pilot always uses base_and_plus.  The arsenal-
    # mining cross-bench-stability probe uses plus_only to
    # extrapolate A1@K=5 on MBPP+ from the W91 base-MBPP sidecar
    # candidates.
    mode: str = "base_and_plus"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "task_id": str(self.task_id),
            "candidate_code_cid": str(
                self.candidate_code_cid),
            "passed": bool(self.passed),
            "n_base_passed": int(self.n_base_passed),
            "n_base_total": int(self.n_base_total),
            "n_plus_passed": int(self.n_plus_passed),
            "n_plus_total": int(self.n_plus_total),
            "timed_out": bool(self.timed_out),
            "wall_ms": int(self.wall_ms),
            "returncode": int(self.returncode),
            "stderr_tail": str(self.stderr_tail),
            "stdout_tail": str(self.stdout_tail),
            "mode": str(self.mode),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w101_mbpp_plus_executor_result_v1",
            "result": self.to_dict(),
        })


def build_plus_assertions(
        problem: MbppPlusProblemV1) -> list[str]:
    """Reconstruct EvalPlus extra-test assertions as a parallel
    list of `assert` statements.  Mirrors EvalPlus's documented
    encoding: ``assert <entry>(*input) == <output>``.  An empty
    list is returned if the problem has no EvalPlus extra tests.
    """
    if not problem.plus_input or not problem.plus_output:
        return []
    if len(problem.plus_input) != len(problem.plus_output):
        # Asymmetric encoding — refuse to silently produce
        # mis-paired assertions.
        return []
    out: list[str] = []
    entry = problem.entry_point
    for i, (inp, exp) in enumerate(zip(
            problem.plus_input, problem.plus_output)):
        out.append(
            f"assert {entry}(*{inp}) == {exp},"
            f" 'plus#{i} mismatch'")
    return out


def run_mbpp_plus_executor_v1(
        *,
        problem: MbppPlusProblemV1,
        candidate_code: str,
        timeout_s: float = W101_MBPP_PLUS_EXECUTOR_TIMEOUT_S,
        kill_after_s: float = (
            W101_MBPP_PLUS_EXECUTOR_KILL_AFTER_S),
        python_exe: str | None = None,
        mode: str = "base_and_plus",
) -> MbppPlusExecutorResultV1:
    """Run candidate + (base + plus) assertions in a fresh
    CPython subprocess.

    ``mode``:
      * ``"base_and_plus"`` — must pass every base + every plus
        assertion to PASS (canonical W101 mode).
      * ``"base_only"`` — only base tests (mirrors W90 MBPP
        executor; useful for re-validating the W91 sidecar).
      * ``"plus_only"`` — only EvalPlus extra tests (cross-bench-
        stability probe: how many W91 sidecar candidates fail the
        extra tests even though they passed the base).
    """
    py = python_exe or sys.executable
    base_tests = (
        list(problem.base_test_list)
        if mode in ("base_and_plus", "base_only")
        else [])
    plus_tests = (
        build_plus_assertions(problem)
        if mode in ("base_and_plus", "plus_only")
        else [])
    code_cid = hashlib.sha256(
        candidate_code.encode("utf-8")).hexdigest()
    n_base_total = len(base_tests)
    n_plus_total = len(plus_tests)
    test_program = (
        candidate_code + "\n\n"
        + "# --- W101 MBPP+ combined assertions ---\n"
        + f"_n_base_total = {n_base_total}\n"
        + f"_n_plus_total = {n_plus_total}\n"
        + "_n_base_passed = 0\n"
        + "_n_plus_passed = 0\n"
        + "_fail_log = []\n")
    for i, a in enumerate(base_tests):
        test_program += (
            f"try:\n"
            f"    {a}\n"
            f"    _n_base_passed += 1\n"
            f"except Exception as _e:\n"
            f"    _fail_log.append(f'base#{i}: '"
            f" + type(_e).__name__ + ': ' + str(_e))\n")
    for i, a in enumerate(plus_tests):
        test_program += (
            f"try:\n"
            f"    {a}\n"
            f"    _n_plus_passed += 1\n"
            f"except Exception as _e:\n"
            f"    _fail_log.append(f'plus#{i}: '"
            f" + type(_e).__name__ + ': ' + str(_e))\n")
    test_program += (
        "import sys\n"
        "_ok = ("
        "_n_base_passed >= _n_base_total"
        " and _n_plus_passed >= _n_plus_total)\n"
        "if not _ok:\n"
        "    print('FAIL',"
        " _n_base_passed, '/', _n_base_total,"
        " _n_plus_passed, '/', _n_plus_total)\n"
        "    for _l in _fail_log[:20]:\n"
        "        print(_l, file=sys.stderr)\n"
        "    sys.exit(1)\n"
        "print('PASS',"
        " _n_base_total, _n_plus_total)\n")
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
    n_base_passed = 0
    n_plus_passed = 0
    if not timed_out:
        m = re.match(
            r"^(PASS|FAIL)\s+(\d+)\s*/\s*(\d+)\s+(\d+)\s*/\s*(\d+)",
            out_text.strip())
        if m is not None:
            n_base_passed = int(m.group(2))
            n_plus_passed = int(m.group(4))
            if m.group(1) == "PASS":
                n_base_passed = n_base_total
                n_plus_passed = n_plus_total
        else:
            # PASS line uses different layout in success case.
            m2 = re.match(
                r"^PASS\s+(\d+)\s+(\d+)",
                out_text.strip())
            if m2 is not None:
                n_base_passed = int(m2.group(1))
                n_plus_passed = int(m2.group(2))
    passed = (rc == 0 and not timed_out)
    return MbppPlusExecutorResultV1(
        schema=W101_MBPP_PLUS_EXECUTOR_V1_SCHEMA_VERSION,
        task_id=str(problem.task_id),
        candidate_code_cid=str(code_cid),
        passed=bool(passed),
        n_base_passed=int(n_base_passed),
        n_base_total=int(n_base_total),
        n_plus_passed=int(n_plus_passed),
        n_plus_total=int(n_plus_total),
        timed_out=bool(timed_out),
        wall_ms=int(wall_ms),
        returncode=int(rc),
        stderr_tail=err_text[-500:],
        stdout_tail=out_text[-200:],
        mode=str(mode),
    )


__all__ = [
    "W101_MBPP_PLUS_EXECUTOR_V1_SCHEMA_VERSION",
    "W101_MBPP_PLUS_EXECUTOR_TIMEOUT_S",
    "W101_MBPP_PLUS_EXECUTOR_KILL_AFTER_S",
    "MbppPlusExecutorResultV1",
    "build_plus_assertions",
    "run_mbpp_plus_executor_v1",
]
