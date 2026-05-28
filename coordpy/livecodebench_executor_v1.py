"""W107 / COO-9 — LiveCodeBench FUNCTIONAL-form V1 executor.

Runs a candidate Python program against a LiveCodeBench functional
problem's test cases in a fresh CPython subprocess, returning a binary
PASS/FAIL.  This mirrors the ``humaneval_plus_executor_v1`` cleanness
discipline exactly — the property that earned executor-cleanness gate
G9 across W86 → W105:

* Fresh CPython subprocess (``-I`` isolated; no current dir, no
  ``PYTHON*`` env vars); no shared state with the parent.
* Wall-clock timeout (soft + hard kill).
* stderr/stdout tail-truncated and returned to the caller so the W89
  reflexion mechanism reads a real failure signal.
* A candidate PASSes iff the subprocess exits 0 (all test cases match).
* NO LLM-as-judge anywhere in the chain (the W85–W106 anti-cheat
  surface).

Functional contract.  A LiveCodeBench functional problem names an
entry callable ``func_name`` that is EITHER a top-level function OR a
method on a ``Solution`` class (the LeetCode-style starter_code form).
The executor harness:

  1. Inlines the candidate code,
  2. Resolves the entry callable (top-level ``func_name`` first, then
     ``Solution().<func_name>``),
  3. For each test case, decodes the input arguments, calls the entry
     callable, and compares the normalized result to the expected
     output,
  4. Exits 0 iff every test matches.

Input/output decoding.  The DEFAULT decoding treats each test's
``input`` as a JSON array of positional arguments and ``output`` as a
JSON value (the result).  This is clean and self-testable.  The EXACT
upstream encoding for a given release_vN MUST be confirmed at
operator-fetch time (the W102 silent-degeneration lesson) — see the
cap below; this executor's offline self-test proves the MACHINERY is
clean, not the live-corpus binding.

Honest scope (W107)
-------------------

* ``W107-L-LIVECODEBENCH-EXECUTOR-V1-INPUT-ENCODING-CONFIRM-AT-FETCH-CAP``
  — the default JSON-positional-args decoding must be confirmed
  against the live release_vN test-case encoding before any pilot;
  if the upstream encoding differs, the decoder is swapped at
  operator-fetch time, NOT silently mis-parsed.
* ``W107-L-LIVECODEBENCH-EXECUTOR-V1-FUNCTIONAL-ONLY-CAP`` — only the
  functional form is executed; stdin/stdout problems are out of scope
  (filtered by the loader).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
import sys
import time
from typing import Any

W107_LIVECODEBENCH_EXECUTOR_V1_SCHEMA_VERSION: str = (
    "coordpy.livecodebench_executor_v1.v1")

W107_LIVECODEBENCH_EXECUTOR_V1_TIMEOUT_S: float = 15.0
W107_LIVECODEBENCH_EXECUTOR_V1_KILL_AFTER_S: float = 20.0


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


# The in-subprocess harness.  Kept as a module-level template so the
# executor body stays auditable.  It is intentionally strict: any
# mismatch, exception, or unresolved entry callable exits non-zero.
_HARNESS_TEMPLATE = r'''
import json, sys

def _normalize(v):
    # order-insensitive only at the top level is NOT assumed; exact
    # structural equality after JSON round-trip.
    return json.loads(json.dumps(v, sort_keys=True, default=str))

def _resolve_entry(g, func_name):
    if func_name and func_name in g and callable(g[func_name]):
        return g[func_name]
    sol = g.get("Solution")
    if sol is not None:
        try:
            inst = sol()
        except Exception:
            inst = None
        if inst is not None and func_name and hasattr(inst, func_name):
            m = getattr(inst, func_name)
            if callable(m):
                return m
    return None

_FUNC_NAME = __FUNC_NAME__
_TESTS = __TESTS_JSON__

entry = _resolve_entry(globals(), _FUNC_NAME)
if entry is None:
    sys.stderr.write("ENTRY_NOT_FOUND:" + repr(_FUNC_NAME))
    sys.exit(3)

_failures = 0
for _i, _tc in enumerate(_TESTS):
    try:
        _args = json.loads(_tc["input"]) if _tc["input"] != "" else []
        if not isinstance(_args, list):
            _args = [_args]
        _expected = (
            json.loads(_tc["output"]) if _tc["output"] != "" else None)
        _got = entry(*_args)
        if _normalize(_got) != _normalize(_expected):
            _failures += 1
            sys.stderr.write(
                "CASE_FAIL idx=%d got=%r expected=%r\n"
                % (_i, _got, _expected))
            break
    except Exception as _e:
        _failures += 1
        sys.stderr.write("CASE_EXC idx=%d %s: %s\n"
                         % (_i, type(_e).__name__, _e))
        break

sys.exit(0 if _failures == 0 else 1)
'''


@dataclasses.dataclass(frozen=True)
class LiveCodeBenchExecutorResultV1:
    """Outcome of running one candidate against one LiveCodeBench
    functional problem in a fresh subprocess."""

    schema: str
    question_id: str
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
            "question_id": str(self.question_id),
            "candidate_code_cid": str(self.candidate_code_cid),
            "passed": bool(self.passed),
            "timed_out": bool(self.timed_out),
            "wall_ms": int(self.wall_ms),
            "returncode": int(self.returncode),
            "stderr_tail": str(self.stderr_tail),
            "stdout_tail": str(self.stdout_tail),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w107_livecodebench_executor_result_v1",
            "result": self.to_dict(),
        })


def _build_test_program(
        *, candidate_code: str, func_name: str,
        tests: list[dict]) -> str:
    harness = (
        _HARNESS_TEMPLATE
        .replace("__FUNC_NAME__", json.dumps(str(func_name)))
        .replace("__TESTS_JSON__", json.dumps(tests)))
    return candidate_code + "\n\n" + harness + "\n"


def run_livecodebench_executor_v1(
        *,
        question_id: str,
        func_name: str,
        tests: list[dict],
        candidate_code: str,
        timeout_s: float = (
            W107_LIVECODEBENCH_EXECUTOR_V1_TIMEOUT_S),
        kill_after_s: float = (
            W107_LIVECODEBENCH_EXECUTOR_V1_KILL_AFTER_S),
        python_exe: str | None = None,
) -> LiveCodeBenchExecutorResultV1:
    """Run candidate + the functional test harness in a fresh CPython
    subprocess.  Exits 0 iff every test case matches.

    ``tests`` is a list of ``{"input": <json-str>, "output":
    <json-str>}`` dicts (the loader's ``LiveCodeBenchFunctionalTestV1``
    fields).
    """
    py = python_exe or sys.executable
    test_program = _build_test_program(
        candidate_code=candidate_code, func_name=func_name,
        tests=list(tests))
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
    return LiveCodeBenchExecutorResultV1(
        schema=W107_LIVECODEBENCH_EXECUTOR_V1_SCHEMA_VERSION,
        question_id=str(question_id),
        candidate_code_cid=str(code_cid),
        passed=bool(passed),
        timed_out=bool(timed_out),
        wall_ms=int(wall_ms),
        returncode=int(rc),
        stderr_tail=err_text[-500:],
        stdout_tail=out_text[-200:],
    )


__all__ = [
    "W107_LIVECODEBENCH_EXECUTOR_V1_SCHEMA_VERSION",
    "W107_LIVECODEBENCH_EXECUTOR_V1_TIMEOUT_S",
    "W107_LIVECODEBENCH_EXECUTOR_V1_KILL_AFTER_S",
    "LiveCodeBenchExecutorResultV1",
    "run_livecodebench_executor_v1",
]
