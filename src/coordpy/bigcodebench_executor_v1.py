"""W110 / COO-9 — BigCodeBench unittest executor V1 (SECOND resistant lane).

Runs a candidate Python solution against a BigCodeBench problem's
``unittest.TestCase`` oracle in a fresh CPython subprocess, returning a binary
PASS/FAIL. Mirrors the ``apps_executor_v1`` / ``livecodebench_executor_v2``
cleanness discipline exactly (the property that earned executor-cleanness gate
G9 across W86 → W109):

* Fresh CPython subprocess (``-I`` isolated; no user-site, no ``PYTHON*`` env);
  installed third-party libs the BigCodeBench tasks need (numpy/pandas/...) are
  still importable (``-I`` removes only the *user* site dir, not the main
  site-packages).
* Soft + hard-kill wall timeout (BigCodeBench tests pay heavier import cost —
  pandas/sklearn/matplotlib — so the default is wider than APPS).
* stderr tail (the unittest failure/traceback summary) returned for the W89
  reflexion signal.
* PASS iff the subprocess exits 0 (the suite ``wasSuccessful``).
* NO LLM-as-judge anywhere.

BigCodeBench-specific harness (differs from the call-based APPS executor):

* The candidate and the ``test`` source are exec'd into ONE controlled
  namespace whose ``__name__`` is NOT ``"__main__"`` — so any stray
  ``unittest.main()`` at the bottom of a test module is a no-op (and a
  ``SystemExit`` from one is swallowed). The harness then discovers EVERY
  ``unittest.TestCase`` subclass defined in that namespace and runs them all.
* Distinct non-zero exit codes make the failure mode legible for the reflexion
  signal: 2 = candidate import error, 3 = entry point missing, 4 = test import
  error, 5 = no TestCase found, 1 = test failures.

Honest scope (W110):

* ``W110-L-BIGCODEBENCH-EXECUTOR-V1-SUBPROCESS-SITE-CAP`` — isolation is by
  process boundary + ``-I`` (no user-site), NOT by capability; BigCodeBench
  tasks legitimately touch the filesystem (tempfiles) and use ``unittest.mock``
  — that is the benchmark's design and is run faithfully, not sandbox-escaped.
  The ``-I`` flag drops only the USER site, so deps must live in the venv/main
  site (W110 uses a ``--system-site-packages`` venv; see RUNBOOK_W110 § 3).
* ``W110-L-BIGCODEBENCH-EXECUTOR-V1-EXEC-NAMESPACE-NOT-FILE-MODULE-CAP`` — the
  candidate + ``test`` are exec'd into ONE in-memory namespace, not written as
  an importable file module. The handful of BigCodeBench tasks whose tests
  re-import the solution by module name (or spawn a subprocess that does) fail
  here and are DROPPED by the gold-green filter — they never false-PASS. (At
  the W110 preflight: ~4 of 1140.)
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
import sys
import time
from typing import Any

W110_BIGCODEBENCH_EXECUTOR_V1_SCHEMA_VERSION: str = (
    "coordpy.bigcodebench_executor_v1.v1")

# Wider than APPS (15/20): pandas/sklearn/matplotlib import + render is slow.
W110_BIGCODEBENCH_EXECUTOR_V1_TIMEOUT_S: float = 30.0
W110_BIGCODEBENCH_EXECUTOR_V1_KILL_AFTER_S: float = 45.0


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


# The harness exec's the candidate then the test into a single non-__main__
# namespace, discovers all TestCase subclasses, runs them, and exits 0 iff the
# combined suite is successful. Source strings are embedded as JSON literals
# (valid Python string literals; non-ASCII becomes \uXXXX).
_HARNESS_TEMPLATE = r'''
import os as _os
# Headless: never pop a GUI chart window, and never let plt.show() block (which
# would hit the wall timeout and FALSELY fail a correct chart solution). Set
# BEFORE the candidate imports matplotlib/seaborn so they pick up the Agg backend.
_os.environ["MPLBACKEND"] = "Agg"
_os.environ.pop("DISPLAY", None)
try:
    import matplotlib as _mpl
    _mpl.use("Agg", force=True)
except Exception:
    pass

import io, sys, unittest, traceback

_ns = {"__name__": "_bcb_solution_mod"}
_SOLUTION_SRC = __SOLUTION_SRC__
_TEST_SRC = __TEST_SRC__
_ENTRY = __ENTRY__

try:
    exec(compile(_SOLUTION_SRC, "<solution>", "exec"), _ns)
except SystemExit:
    pass
except BaseException as _e:
    sys.stderr.write("SOLUTION_IMPORT_ERROR %s: %s\n"
                     % (type(_e).__name__, _e))
    traceback.print_exc()
    sys.exit(2)

if _ENTRY and (_ENTRY not in _ns or not callable(_ns.get(_ENTRY))):
    sys.stderr.write("ENTRY_NOT_FOUND: %r\n" % _ENTRY)
    sys.exit(3)

try:
    exec(compile(_TEST_SRC, "<test>", "exec"), _ns)
except SystemExit:
    pass
except BaseException as _e:
    sys.stderr.write("TEST_IMPORT_ERROR %s: %s\n" % (type(_e).__name__, _e))
    traceback.print_exc()
    sys.exit(4)

_cases = [v for v in list(_ns.values())
          if isinstance(v, type) and issubclass(v, unittest.TestCase)
          and v is not unittest.TestCase]
if not _cases:
    sys.stderr.write("NO_TESTCASE_FOUND\n")
    sys.exit(5)

_loader = unittest.TestLoader()
_suite = unittest.TestSuite()
for _c in _cases:
    _suite.addTests(_loader.loadTestsFromTestCase(_c))

_buf = io.StringIO()
_res = unittest.TextTestRunner(stream=_buf, verbosity=1).run(_suite)
_out = _buf.getvalue()
sys.stderr.write(_out[-3000:])
sys.exit(0 if _res.wasSuccessful() else 1)
'''


@dataclasses.dataclass(frozen=True)
class BigCodeBenchExecutorResultV1:
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
        return _sha256_hex({"kind": "w110_bigcodebench_executor_result_v1",
                           "result": self.to_dict()})


def _build_test_program(*, candidate_code: str, test_source: str,
                        entry_point: str) -> str:
    return (
        _HARNESS_TEMPLATE
        .replace("__SOLUTION_SRC__", json.dumps(str(candidate_code)))
        .replace("__TEST_SRC__", json.dumps(str(test_source)))
        .replace("__ENTRY__", json.dumps(str(entry_point))))


def run_bigcodebench_executor_v1(
        *, problem_id: str, test_source: str, entry_point: str,
        candidate_code: str,
        timeout_s: float = W110_BIGCODEBENCH_EXECUTOR_V1_TIMEOUT_S,
        kill_after_s: float = W110_BIGCODEBENCH_EXECUTOR_V1_KILL_AFTER_S,
        python_exe: str | None = None) -> BigCodeBenchExecutorResultV1:
    """Run candidate + the row's ``unittest`` oracle in a fresh CPython
    subprocess. Exits 0 iff the combined suite is successful."""
    py = python_exe or sys.executable
    program = _build_test_program(
        candidate_code=candidate_code, test_source=test_source,
        entry_point=entry_point)
    code_cid = hashlib.sha256(candidate_code.encode("utf-8")).hexdigest()
    t0 = time.time()
    timed_out = False
    rc = -1
    out_b = b""
    err_b = b""
    import os as _os
    child_env = dict(_os.environ)
    child_env["MPLBACKEND"] = "Agg"   # headless matplotlib (no GUI windows)
    child_env.pop("DISPLAY", None)
    try:
        proc = subprocess.run(
            [py, "-I", "-c", program], input=b"",
            capture_output=True, timeout=float(kill_after_s), check=False,
            env=child_env)
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
    return BigCodeBenchExecutorResultV1(
        schema=W110_BIGCODEBENCH_EXECUTOR_V1_SCHEMA_VERSION,
        problem_id=str(problem_id), candidate_code_cid=str(code_cid),
        passed=bool(passed), timed_out=bool(timed_out),
        wall_ms=int(wall_ms), returncode=int(rc),
        stderr_tail=err_text[-800:], stdout_tail=out_text[-200:])


__all__ = [
    "W110_BIGCODEBENCH_EXECUTOR_V1_SCHEMA_VERSION",
    "W110_BIGCODEBENCH_EXECUTOR_V1_TIMEOUT_S",
    "W110_BIGCODEBENCH_EXECUTOR_V1_KILL_AFTER_S",
    "BigCodeBenchExecutorResultV1",
    "run_bigcodebench_executor_v1",
]
