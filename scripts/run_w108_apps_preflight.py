#!/usr/bin/env python3
"""W108-γ — APPS BACKUP-lane NIM-free preflight + pivot conditions.

Proves the APPS call-based (functional) machinery is clean OFFLINE and records
the exact conditions under which APPS becomes the active battlefield IN the
W108 milestone (RUNBOOK_W108 § 6).  ZERO NIM calls.  This is the BACKUP lane:
LiveCodeBench PASSED its real-data structural-soundness test (W108-β), so no
pivot is triggered — but APPS is built to *real* so a future structural
failure pivots without a paperwork milestone.

Probes:

* APPS executor self-test (gold top-level + gold Solution method PASS, wrong
  FAIL, infinite-loop TIMEOUT, APPS 1-element-output-wrapper tolerance).
* APPS loader schema self-test (valid call-based row accepted; stdin row
  accepted-then-filtered; missing-field row REFUSED — the W102 guard).
* Selection verdict: APPS = BACKUP; contamination cap (C7 = C, 2021 vintage);
  the exact LiveCodeBench-failure pivot conditions.

Usage::

    python scripts/run_w108_apps_preflight.py
    python scripts/run_w108_apps_preflight.py --print-fetch-playbook
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import sys  # noqa: E402
sys.path.insert(0, str(ROOT))

from coordpy.apps_executor_v1 import run_apps_executor_v1  # noqa: E402
from coordpy.apps_loader_v1 import (  # noqa: E402
    is_call_based_row,
    validate_row_schema,
)

OUT_ROOT = ROOT / "results" / "w108" / "apps_preflight"

# LiveCodeBench-failure pivot conditions (RUNBOOK_W108 § 6).  APPS becomes the
# active battlefield IN-milestone iff ANY fire on the LiveCodeBench lane.
PIVOT_CONDITIONS = [
    "LCB P1 corpus integrity FAIL (SHA / schema mismatch loader cannot bind)",
    "LCB P2 executor-V2 self-test FAIL on real data (gold-path fails OR a "
    "wrong solution false-passes)",
    "LCB P3 functional-subset size < 30 after the real fetch",
    "LCB gold-path correctness bar (RUNBOOK_W108 § 4) cannot be made green",
]
PIVOT_TRIGGERED_W108 = False  # LiveCodeBench passed real-data soundness.

_GOLD_TOPLEVEL = "def add(a, b):\n    return a + b\n"
_GOLD_SOLUTION = (
    "class Solution:\n    def add(self, a, b):\n        return a + b\n")
_WRONG = "def add(a, b):\n    return a - b\n"
_LOOP = "def add(a, b):\n    while True:\n        pass\n"
# APPS encoding: args is a JSON list; output sometimes a 1-element wrapper.
_SYNTH_TESTS = [
    {"args": json.dumps([2, 3]), "output": json.dumps(5)},
    {"args": json.dumps([-1, 1]), "output": json.dumps([0])},  # wrapper
    {"args": json.dumps([10, 20]), "output": json.dumps(30)},
]


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _cid(payload: object) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _run_executor_self_test() -> dict:
    gold_top = run_apps_executor_v1(
        problem_id="synth/gold_toplevel", func_name="add",
        tests=_SYNTH_TESTS, candidate_code=_GOLD_TOPLEVEL)
    gold_sol = run_apps_executor_v1(
        problem_id="synth/gold_solution", func_name="add",
        tests=_SYNTH_TESTS, candidate_code=_GOLD_SOLUTION)
    wrong = run_apps_executor_v1(
        problem_id="synth/wrong", func_name="add",
        tests=_SYNTH_TESTS, candidate_code=_WRONG)
    loop = run_apps_executor_v1(
        problem_id="synth/loop", func_name="add",
        tests=_SYNTH_TESTS, candidate_code=_LOOP,
        timeout_s=2.0, kill_after_s=3.0)
    checks = {
        "gold_toplevel_passes": bool(gold_top.passed),
        "gold_solution_method_passes": bool(gold_sol.passed),
        "output_wrapper_tolerance_ok": bool(gold_top.passed),  # mid test wrapped
        "wrong_fails": bool(not wrong.passed),
        "infinite_loop_times_out": bool(loop.timed_out),
    }
    return {"checks": checks, "all_pass": all(checks.values()),
            "evidence": {"gold_toplevel_rc": gold_top.returncode,
                        "gold_solution_rc": gold_sol.returncode,
                        "wrong_rc": wrong.returncode,
                        "wrong_stderr_tail": wrong.stderr_tail,
                        "loop_timed_out": loop.timed_out}}


def _run_loader_schema_self_test() -> dict:
    good = {
        "problem_id": "apps/1", "question": "implement add",
        "starter_code": "",
        "input_output": json.dumps({
            "fn_name": "add", "inputs": [[1, 2]], "outputs": [3]}),
        "difficulty": "introductory",
    }
    stdin_row = {
        "problem_id": "apps/2", "question": "read two ints",
        "input_output": json.dumps({
            "inputs": ["1 2\n"], "outputs": ["3\n"]}),  # no fn_name
        "difficulty": "interview",
    }
    missing = {"problem_id": "apps/3"}  # missing question + input_output
    ok_good, _ = validate_row_schema(good)
    ok_stdin, _ = validate_row_schema(stdin_row)
    ok_missing, missing_reason = validate_row_schema(missing)
    checks = {
        "valid_call_based_row_accepted": bool(ok_good),
        "call_based_row_is_call_based": bool(is_call_based_row(good)),
        "stdin_row_accepted_then_filtered": bool(
            ok_stdin and not is_call_based_row(stdin_row)),
        "missing_required_field_refused": bool(not ok_missing),
    }
    return {"checks": checks, "all_pass": all(checks.values()),
            "missing_field_reason": missing_reason}


FETCH_PLAYBOOK = r"""
# W108-γ APPS operator corpus-fetch playbook (ONLY needed if a pivot fires)
# ----------------------------------------------------------------------
# APPS is BACKUP. Fetch only if a LiveCodeBench pivot condition fires
# (RUNBOOK_W108 § 6). Then:
#   1) Fetch codeparrot/apps test split (the call-based subset is small):
#        mkdir -p ~/.cache/coordpy
#        # (datasets lib or the resolve/ URL for the pinned split)
#        sha256sum ~/.cache/coordpy/apps-test.jsonl
#        export APPS_TRUSTED_SHA256_OVERRIDE=<sha_from_above>
#   2) Confirm the input_output encoding (fn_name; inputs = arg-lists;
#      whether outputs[i] is bare or 1-element-wrapped). Adjust the executor
#      output-wrapper tolerance at THIS step if it differs.
#   3) Re-run a real-data APPS preflight (mirror run_w108_livecodebench_preflight).
#   4) REMEMBER: any APPS result is contamination-exposed (2021 vintage) and is
#      BACKUP evidence only — never the publication-grade time-anchored claim.
""".strip()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="W108-γ APPS BACKUP NIM-free preflight")
    ap.add_argument("--print-fetch-playbook", action="store_true")
    ap.add_argument("--out-root", default=str(OUT_ROOT))
    args = ap.parse_args()
    if args.print_fetch_playbook:
        print(FETCH_PLAYBOOK)
        return 0

    exec_self = _run_executor_self_test()
    loader_self = _run_loader_schema_self_test()
    machinery_ok = exec_self["all_pass"] and loader_self["all_pass"]

    verdict = {
        "schema": "coordpy.w108_apps_preflight.v1",
        "milestone": "W108-gamma",
        "lane": "APPS backup (structural-pivot readiness)",
        "executor_self_test": exec_self,
        "loader_schema_self_test": loader_self,
        "machinery_clean_offline": bool(machinery_ok),
        "contamination_cap": (
            "C7 = C: APPS is 2021 vintage, almost certainly inside the "
            "Llama-3.x training corpus; any APPS result is contamination-"
            "exposed and is BACKUP evidence only "
            "(W108-L-APPS-CONTAMINATION-EXPOSED-2021-VINTAGE-CAP)"),
        "pivot_conditions": PIVOT_CONDITIONS,
        "pivot_triggered_w108": PIVOT_TRIGGERED_W108,
        "pivot_status": (
            "NOT TRIGGERED — LiveCodeBench PASSED its real-data structural-"
            "soundness test (W108-β); APPS stays backup-ready"),
        "operator_deferred_probes": [
            "real-corpus SHA pin + functional-subset size — needs fetched APPS",
            "input_output encoding + output-wrapper convention confirm",
            "live A1@K=5 residual — needs fetched corpus + NIM (only on pivot)",
        ],
        "ts_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    verdict["verdict_cid"] = _cid(
        {k: v for k, v in verdict.items() if k != "ts_utc"})

    out_dir = Path(args.out_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "preflight_verdict.json").write_text(
        json.dumps(verdict, indent=2, default=str))

    print("=== W108-γ APPS BACKUP NIM-free preflight ===")
    print(f"  executor self-test : "
          f"{'PASS' if exec_self['all_pass'] else 'FAIL'} {exec_self['checks']}")
    print(f"  loader schema test : "
          f"{'PASS' if loader_self['all_pass'] else 'FAIL'} "
          f"{loader_self['checks']}")
    print(f"  machinery clean    : {machinery_ok}")
    print(f"  pivot triggered    : {PIVOT_TRIGGERED_W108} "
          "(LiveCodeBench passed real-data soundness)")
    print(f"  contamination cap  : C7=C (2021 vintage; backup evidence only)")
    print(f"  verdict_cid        : {verdict['verdict_cid']}")
    print(f"  written            : {out_dir / 'preflight_verdict.json'}")
    return 0 if machinery_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
