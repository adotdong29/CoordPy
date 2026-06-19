"""W108 / COO-9 — LiveCodeBench FUNCTIONAL-form V2 executor (real-data
encoding fix).

V1 (``coordpy.livecodebench_executor_v1``) decoded a functional test's
``input`` as a SINGLE ``json.loads(input)`` blob (a JSON array of positional
args, else a single arg).  W108 fetched the real
``livecodebench/code_generation_lite`` ``release_v6`` corpus (file
``test6.jsonl``; dataset commit ``0fe84c39…``; SHA-256
``bb4c364f71921c4495a6ad15abe1a927350b720009f4933e2e71f8af0f6fd1f5``) and
CONFIRMED the real functional encoding is different:

    input  = one JSON value per NEWLINE-separated line, one line per
             positional argument (signature order, after ``self``).
    output = a single JSON value (the expected return).

Examples confirmed from the live corpus:

    zigzagTraversal(grid)                 input='[[1, 2], [3, 4]]'        (1 line  -> 1 arg)
    minMaxWeight(n, edges, threshold)     input='5\\n[[1,0,1],…]\\n2'      (3 lines -> 3 args)
    countNonDecreasingSubarrays(nums, k)  input='[6, 3, 1, 2, 4, 4]\\n7'   (2 lines -> 2 args)

V1's ``json.loads(whole_input)`` decoder is therefore WRONG on real data:
it raises a JSONDecodeError on multi-line (multi-arg) inputs and mis-splits a
single-list argument (``json.loads('[[1,2],[3,4]]')`` -> ``[[1,2],[3,4]]`` ->
used as two args).  This is exactly the silent-degeneration failure mode the
W102 lesson + the W107
``W107-L-LIVECODEBENCH-EXECUTOR-V1-INPUT-ENCODING-CONFIRM-AT-FETCH-CAP``
warned about.  V2 fixes the decoder on real data; V1 is retained as a
historical artifact + anti-pattern (the wrong-decoder example), exactly as the
W101 MBPP+ V1 loader was retained after the W102 V2 fix.

This mirrors the ``humaneval_plus_executor_v1`` cleanness discipline:

* Fresh CPython subprocess (``-I`` isolated; no current dir, no ``PYTHON*``);
  no shared state with the parent.
* Soft + hard-kill wall timeout.
* stderr/stdout tail-truncated + returned for the W89 reflexion signal.
* PASS iff the subprocess exits 0 (all test cases match).
* NO LLM-as-judge anywhere (the W85–W107 anti-cheat surface).

Honest scope (W108)
-------------------

* ``W108-L-LIVECODEBENCH-EXECUTOR-V2-PLAIN-JSON-ARG-CAP`` — V2 decodes plain
  JSON arguments (ints / floats / strings / bools / lists / dicts).  Problems
  whose arguments are LeetCode datastructures requiring deserialisation
  (``ListNode`` / ``TreeNode``) are NOT reconstructed; they are scored as the
  candidate produced (typically a clean FAIL across ALL arms equally — fair but
  lower-signal).  The real-data preflight reports how many of the pinned
  functional subset are plain-arg.
* ``W108-L-LIVECODEBENCH-EXECUTOR-V2-EXACT-MATCH-CAP`` — outputs are compared by
  exact structural equality after a JSON round-trip (the
  ``code_generation_lite`` default).  Problems with multiple valid answers /
  float tolerance are out of scope (counted, not silently passed).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
import sys
import time
from typing import Any

W108_LIVECODEBENCH_EXECUTOR_V2_SCHEMA_VERSION: str = (
    "coordpy.livecodebench_executor_v2.v1")

W108_LIVECODEBENCH_EXECUTOR_V2_TIMEOUT_S: float = 15.0
W108_LIVECODEBENCH_EXECUTOR_V2_KILL_AFTER_S: float = 20.0


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


# The in-subprocess harness.  Kept as a module-level template so the executor
# body stays auditable.  Strict: any mismatch / exception / unresolved entry
# exits non-zero.  The decoder splits ``input`` on newlines (one JSON value per
# positional argument) — the CONFIRMED real LiveCodeBench functional encoding.
_HARNESS_TEMPLATE = r'''
import json, sys
from typing import *  # LeetCode starter code uses List/Optional/etc unqualified
import math, collections, heapq, bisect, itertools, functools, re, string
from collections import deque, defaultdict, Counter, OrderedDict
try:
    from math import inf
except Exception:
    inf = float("inf")

def _normalize(v):
    return json.loads(json.dumps(v, sort_keys=True, default=str))

def _decode_args(raw_input):
    # CONFIRMED LiveCodeBench functional encoding: one JSON value per
    # newline-separated line == one positional argument (signature order).
    if raw_input == "":
        return []
    lines = raw_input.split("\n")
    args = []
    for ln in lines:
        s = ln.strip()
        if s == "":
            continue
        args.append(json.loads(s))
    return args

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
        _args = _decode_args(_tc["input"])
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
class LiveCodeBenchExecutorV2ResultV1:
    """Outcome of running one candidate against one LiveCodeBench functional
    problem in a fresh subprocess (V2 decoder)."""

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
            "kind": "w108_livecodebench_executor_v2_result_v1",
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


def run_livecodebench_executor_v2(
        *,
        question_id: str,
        func_name: str,
        tests: list[dict],
        candidate_code: str,
        timeout_s: float = (
            W108_LIVECODEBENCH_EXECUTOR_V2_TIMEOUT_S),
        kill_after_s: float = (
            W108_LIVECODEBENCH_EXECUTOR_V2_KILL_AFTER_S),
        python_exe: str | None = None,
) -> LiveCodeBenchExecutorV2ResultV1:
    """Run candidate + the functional test harness in a fresh CPython
    subprocess (V2 newline-per-argument decoder).  Exits 0 iff every test
    case matches.

    ``tests`` is a list of ``{"input": <str>, "output": <str>}`` dicts (the
    loader's ``LiveCodeBenchFunctionalTestV1`` fields).
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
    return LiveCodeBenchExecutorV2ResultV1(
        schema=W108_LIVECODEBENCH_EXECUTOR_V2_SCHEMA_VERSION,
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
    "W108_LIVECODEBENCH_EXECUTOR_V2_SCHEMA_VERSION",
    "W108_LIVECODEBENCH_EXECUTOR_V2_TIMEOUT_S",
    "W108_LIVECODEBENCH_EXECUTOR_V2_KILL_AFTER_S",
    "LiveCodeBenchExecutorV2ResultV1",
    "run_livecodebench_executor_v2",
]
