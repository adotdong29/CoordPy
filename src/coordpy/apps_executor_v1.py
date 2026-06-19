"""W108 / COO-9 — APPS call-based (functional) executor V1 (BACKUP lane).

Runs a candidate Python program against an APPS call-based problem's tests in
a fresh CPython subprocess, returning a binary PASS/FAIL.  Mirrors the
``livecodebench_executor_v2`` / ``humaneval_plus_executor_v1`` cleanness
discipline exactly (the property that earned executor-cleanness gate G9 across
W86 → W107):

* Fresh CPython subprocess (``-I`` isolated; no current dir, no ``PYTHON*``);
  no shared state with the parent.
* Soft + hard-kill wall timeout.
* stderr/stdout tail-truncated + returned for the W89 reflexion signal.
* PASS iff the subprocess exits 0 (all test cases match).
* NO LLM-as-judge anywhere.

APPS-specific decoding (differs from LiveCodeBench):

* ``inputs[i]`` is ALREADY a JSON list of positional arguments (NOT the
  LiveCodeBench newline-per-arg string).  ``args_repr`` is therefore decoded
  with a single ``json.loads`` and splatted.
* APPS frequently wraps a single expected return as a 1-element list
  (``outputs[i] == [expected]``).  The harness accepts a match against the
  bare expected OR its single-element unwrap — documented as
  ``W108-L-APPS-EXECUTOR-V1-OUTPUT-WRAPPER-TOLERANCE-CAP`` (confirm the exact
  wrapper convention against the live corpus before any APPS pilot).

Honest scope: ``W108-L-APPS-EXECUTOR-V1-PLAIN-JSON-ARG-CAP`` (no
ListNode/TreeNode reconstruction) + ``...-EXACT-MATCH-CAP`` (structural
equality after JSON round-trip), identical in spirit to the LiveCodeBench
executor caps.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
import sys
import time
from typing import Any

W108_APPS_EXECUTOR_V1_SCHEMA_VERSION: str = "coordpy.apps_executor_v1.v1"

W108_APPS_EXECUTOR_V1_TIMEOUT_S: float = 15.0
W108_APPS_EXECUTOR_V1_KILL_AFTER_S: float = 20.0


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


_HARNESS_TEMPLATE = r'''
import json, sys
from typing import *  # APPS starter code uses List/Optional/etc unqualified
import math, collections, heapq, bisect, itertools, functools, re, string
from collections import deque, defaultdict, Counter, OrderedDict
try:
    from math import inf
except Exception:
    inf = float("inf")

def _normalize(v):
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

def _matches(got, expected):
    ng = _normalize(got)
    if ng == _normalize(expected):
        return True
    # APPS 1-element-list output wrapper tolerance.
    if isinstance(expected, list) and len(expected) == 1:
        if ng == _normalize(expected[0]):
            return True
    return False

_FUNC_NAME = __FUNC_NAME__
_TESTS = __TESTS_JSON__

entry = _resolve_entry(globals(), _FUNC_NAME)
if entry is None:
    sys.stderr.write("ENTRY_NOT_FOUND:" + repr(_FUNC_NAME))
    sys.exit(3)

_failures = 0
for _i, _tc in enumerate(_TESTS):
    try:
        _args = json.loads(_tc["args"]) if _tc["args"] != "" else []
        if not isinstance(_args, list):
            _args = [_args]
        _expected = (
            json.loads(_tc["output"]) if _tc["output"] != "" else None)
        _got = entry(*_args)
        if not _matches(_got, _expected):
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
class AppsExecutorResultV1:
    schema: str
    problem_id: str
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
            "problem_id": str(self.problem_id),
            "candidate_code_cid": str(self.candidate_code_cid),
            "passed": bool(self.passed),
            "timed_out": bool(self.timed_out),
            "wall_ms": int(self.wall_ms),
            "returncode": int(self.returncode),
            "stderr_tail": str(self.stderr_tail),
            "stdout_tail": str(self.stdout_tail),
        }

    def cid(self) -> str:
        return _sha256_hex({"kind": "w108_apps_executor_result_v1",
                           "result": self.to_dict()})


def _build_test_program(*, candidate_code: str, func_name: str,
                        tests: list[dict]) -> str:
    harness = (
        _HARNESS_TEMPLATE
        .replace("__FUNC_NAME__", json.dumps(str(func_name)))
        .replace("__TESTS_JSON__", json.dumps(tests)))
    return candidate_code + "\n\n" + harness + "\n"


def run_apps_executor_v1(
        *, problem_id: str, func_name: str, tests: list[dict],
        candidate_code: str,
        timeout_s: float = W108_APPS_EXECUTOR_V1_TIMEOUT_S,
        kill_after_s: float = W108_APPS_EXECUTOR_V1_KILL_AFTER_S,
        python_exe: str | None = None) -> AppsExecutorResultV1:
    """Run candidate + the call-based test harness in a fresh CPython
    subprocess.  Exits 0 iff every test case matches.

    ``tests`` is a list of ``{"args": <json-list-str>, "output": <json-str>}``
    dicts (the loader's ``AppsFunctionalTestV1`` fields)."""
    py = python_exe or sys.executable
    test_program = _build_test_program(
        candidate_code=candidate_code, func_name=func_name,
        tests=list(tests))
    code_cid = hashlib.sha256(candidate_code.encode("utf-8")).hexdigest()
    t0 = time.time()
    timed_out = False
    rc = -1
    out_b = b""
    err_b = b""
    try:
        proc = subprocess.run(
            [py, "-I", "-c", test_program], input=b"",
            capture_output=True, timeout=float(kill_after_s), check=False)
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
    return AppsExecutorResultV1(
        schema=W108_APPS_EXECUTOR_V1_SCHEMA_VERSION,
        problem_id=str(problem_id), candidate_code_cid=str(code_cid),
        passed=bool(passed), timed_out=bool(timed_out),
        wall_ms=int(wall_ms), returncode=int(rc),
        stderr_tail=err_text[-500:], stdout_tail=out_text[-200:])


__all__ = [
    "W108_APPS_EXECUTOR_V1_SCHEMA_VERSION",
    "W108_APPS_EXECUTOR_V1_TIMEOUT_S",
    "W108_APPS_EXECUTOR_V1_KILL_AFTER_S",
    "AppsExecutorResultV1",
    "run_apps_executor_v1",
]
