#!/usr/bin/env python3
"""W111 / COO-9 — NIM-free mechanism-mining pass over the resistant pilots.

Re-executes the EXISTING W110 BigCodeBench (and optionally W108 LiveCodeBench)
pilot transcripts through the real deterministic executor and classifies every
FAILING candidate's executor signal into a failure taxonomy, then maps each
failure class onto the weakness that each W111 candidate mechanism claims to
attack. This is a $0-NIM cheap probe: it spends NO model calls — it only
re-runs the local executor over candidate code the W110 pilot already produced.

The point: a "different mechanism" is only worth a NIM pilot if the weakness it
attacks actually accounts for a material fraction of the resistant failures.
This script measures that fraction directly from real data.

Failure taxonomy (executor stderr last-exception):
  API_GROUNDING  : ImportError/ModuleNotFoundError/AttributeError/NameError +
                   TypeError that is a call-signature error  -> attacked by
                   M1 (library plan) and M2 (local symbol/doc introspection)
  SEMANTIC_LOGIC : AssertionError / ValueError / KeyError / IndexError /
                   wrong-output  -> the model ran the libraries fine but the
                   output is wrong (hidden-test-coupling / spec under-
                   specification). Only M3 (executor-grounded structured
                   patcher) plausibly attacks this, and only when the digest
                   carries information the raw reflexion tail did not.
  ENV_HARNESS    : FileNotFoundError / harness/mock artifacts not in the model's
                   control.
  TIMEOUT        : non-termination.
  SYNTAX         : SyntaxError/IndentationError.
  OTHER/UNKNOWN  : anything else.

Usage:
  python scripts/mine_w111_resistant_failure_modes_v1.py \
      --run results/w110/bigcodebench_pilot/<run> \
      --corpus ~/.cache/coordpy/bigcodebench-v0_1_4.jsonl \
      --bench bigcodebench \
      --out results/w111/mechanism_mining/w110_bcb_failure_census.json \
      [--timeout-s 12 --kill-after-s 15]
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

# Repo root on sys.path (mirrors the W110 pilot script) so `coordpy` imports
# regardless of CWD.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Force headless matplotlib for the BigCodeBench plotting tasks (W110 lesson).
os.environ.setdefault("MPLBACKEND", "Agg")


_EXC_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_\.]*(?:Error|Exception)):", re.M)
_API_EXC = {"ImportError", "ModuleNotFoundError", "AttributeError", "NameError"}
_SEMANTIC_EXC = {"AssertionError", "ValueError", "KeyError", "IndexError",
                 "ArithmeticError", "ZeroDivisionError", "StopIteration"}
_ENV_EXC = {"FileNotFoundError", "PermissionError", "OSError",
            "NotADirectoryError", "IsADirectoryError"}
_SYNTAX_EXC = {"SyntaxError", "IndentationError", "TabError"}


def classify_failure(*, stderr: str, timed_out: bool) -> tuple[str, str]:
    """Return (taxonomy_class, last_exception_name)."""
    if timed_out or "TIMED OUT" in stderr:
        return "TIMEOUT", "Timeout"
    excs = _EXC_RE.findall(stderr)
    last = excs[-1].split(".")[-1] if excs else ""
    if last in _SYNTAX_EXC:
        return "SYNTAX", last
    if last in _API_EXC:
        return "API_GROUNDING", last
    if last == "TypeError":
        # call-signature TypeErrors are API-grounding; otherwise logic.
        sig = ("positional argument" in stderr or "unexpected keyword" in stderr
               or "required argument" in stderr or "takes" in stderr)
        return ("API_GROUNDING" if sig else "SEMANTIC_LOGIC"), last
    if last in _ENV_EXC:
        return "ENV_HARNESS", last
    if last in _SEMANTIC_EXC:
        return "SEMANTIC_LOGIC", last
    # unittest assertion failures without a top-level exception line
    if "AssertionError" in stderr or "FAIL:" in stderr:
        return "SEMANTIC_LOGIC", "AssertionError"
    if last:
        return "OTHER", last
    return "UNKNOWN", ""


# Which taxonomy classes each candidate's attacked-weakness can cover.
CANDIDATE_COVERAGE = {
    "M1_spec_grounded_planner": {"SEMANTIC_LOGIC", "API_GROUNDING"},
    "M2_tool_symbol_introspection": {"API_GROUNDING"},
    "M3_structured_failure_patcher": {"SEMANTIC_LOGIC"},
}


def _load_calls(run: Path) -> list[dict[str, Any]]:
    for name in ("bigcodebench_reflexion_calls.jsonl",
                 "livecodebench_reflexion_calls.jsonl",
                 "apps_reflexion_calls.jsonl"):
        p = run / name
        if p.exists():
            return [json.loads(l) for l in p.open()]
    raise SystemExit(f"no *_reflexion_calls.jsonl in {run}")


def _load_report(run: Path) -> dict[str, Any]:
    for name in ("bigcodebench_reflexion_bench_report.json",
                 "livecodebench_reflexion_bench_report.json",
                 "apps_reflexion_bench_report.json"):
        p = run / name
        if p.exists():
            return json.load(p.open())
    raise SystemExit(f"no *_bench_report.json in {run}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--bench", default="bigcodebench",
                    choices=["bigcodebench", "livecodebench", "apps"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout-s", type=float, default=12.0)
    ap.add_argument("--kill-after-s", type=float, default=15.0)
    ap.add_argument("--python-exe",
                    default=str(Path.home() / ".cache/coordpy/bcb_venv/bin/python"))
    args = ap.parse_args()

    run = Path(args.run).expanduser()
    calls = _load_calls(run)
    report = _load_report(run)
    seed = report["per_seed"][0]
    pids = seed["problem_ids"]
    n = len(pids)
    assert len(calls) == n * 11, f"expected {n*11} calls, got {len(calls)}"

    # corpus
    corpus: dict[str, dict[str, Any]] = {}
    for line in Path(args.corpus).expanduser().open():
        r = json.loads(line)
        corpus[r["task_id"]] = r

    if args.bench == "bigcodebench":
        from coordpy.bigcodebench_executor_v1 import run_bigcodebench_executor_v1
        from coordpy.bigcodebench_reflexion_bench_v1 import extract_candidate_code_v1

        def run_exec(pid: str, code: str):
            p = corpus[pid]
            r = run_bigcodebench_executor_v1(
                problem_id=pid, test_source=p["test"],
                entry_point=p["entry_point"], candidate_code=code,
                timeout_s=args.timeout_s, kill_after_s=args.kill_after_s,
                python_exe=args.python_exe)
            return r.passed, r.stderr_tail, r.timed_out
    else:
        raise SystemExit("only bigcodebench wired in this pass")

    # census over A1 (calls 1..5) and B (calls 6..10) attempts per problem.
    census = collections.Counter()
    per_problem: dict[str, dict[str, Any]] = {}
    exception_hist = collections.Counter()
    n_exec = 0
    for blk in range(n):
        pid = pids[blk]
        base = blk * 11
        arms = {"A1": calls[base + 1:base + 6], "B": calls[base + 6:base + 11]}
        prob_classes = collections.Counter()
        for arm, arm_calls in arms.items():
            for c in arm_calls:
                code = extract_candidate_code_v1(response_text=c["response_text"])
                passed, stderr, timed = run_exec(pid, code)
                n_exec += 1
                if passed:
                    continue
                klass, exc = classify_failure(stderr=stderr, timed_out=timed)
                census[klass] += 1
                exception_hist[exc] += 1
                prob_classes[klass] += 1
        per_problem[pid] = dict(prob_classes)
        print(f"  [{blk+1}/{n}] {pid}: {dict(prob_classes)}", flush=True)

    total_fail = sum(census.values())
    coverage = {}
    for cand, classes in CANDIDATE_COVERAGE.items():
        covered = sum(census[k] for k in classes)
        coverage[cand] = {
            "attacked_classes": sorted(classes),
            "failures_covered": covered,
            "coverage_fraction": round(covered / total_fail, 4) if total_fail else 0.0,
        }

    out = {
        "schema": "coordpy.w111_resistant_failure_census.v1",
        "bench": args.bench,
        "run": str(run),
        "n_problems": n,
        "n_candidate_executions": n_exec,
        "n_failing_executions": total_fail,
        "failure_taxonomy": dict(census.most_common()),
        "exception_histogram": dict(exception_hist.most_common()),
        "candidate_weakness_coverage": coverage,
        "per_problem_classes": per_problem,
    }
    outp = Path(args.out).expanduser()
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2, sort_keys=True))
    print("\n=== FAILURE TAXONOMY (re-executed census) ===")
    for k, v in census.most_common():
        frac = v / total_fail if total_fail else 0
        print(f"  {k:16s} {v:4d}  ({frac:.1%})")
    print("\n=== CANDIDATE WEAKNESS-COVERAGE (of resistant failures) ===")
    for cand, info in coverage.items():
        print(f"  {cand:32s} {info['coverage_fraction']:.1%}  "
              f"attacks {info['attacked_classes']}")
    print(f"\nWrote {outp}")


if __name__ == "__main__":
    main()
