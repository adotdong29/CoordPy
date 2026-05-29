#!/usr/bin/env python3
"""W110-α — BigCodeBench real-data preflight (NIM-free).

Earns (or refuses) the W110 cheap pilot per ``docs/RUNBOOK_W110.md`` § 4.
Runs entirely offline (NO model calls): the only "execution" is the
deterministic ``unittest`` oracle over gold / synthetic candidates.

Gates (ALL must pass to earn the pilot):

* **P1 corpus integrity** — the SHA-pinned loader reads the expected number of
  problems; schema matches.
* **P2 executor self-test** — synthetic gold PASS, wrong FAIL, infinite-loop
  TIMEOUT, AND a REAL ``canonical_solution`` PASSes / a corrupted gold FAILs on
  the live corpus (no false-pass on real data).
* **P3 loader real-data + gold-green** — scan every problem's
  ``canonical_solution`` through the real executor in the pinned venv; keep the
  ``gold_green`` subset (gold passes in this environment); require
  ``gold_green`` >= n_slice. Records the dropped problems (missing-dep vs
  non-dep) — no silent truncation. Records the contamination framing
  (C7 = A-grade release-date resistance; 2024-06; the SECOND resistant bench).
* **P4 deterministic slice** — the n_libs-stratified outcome-blind slice from
  ``select_bigcodebench_slice_v1`` over ``gold_green`` reproduces a stable
  slice CID across two calls.

Writes ``results/w110/bigcodebench_preflight/preflight_verdict.json`` (incl. the
gold_green + slice task-id lists the pilot consumes verbatim) and prints the
pins for ``docs/RUNBOOK_W110.md`` § 3 / § 5.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from coordpy.bigcodebench_loader_v1 import (  # noqa: E402
    BIGCODEBENCH_DEFAULT_SPLIT,
    canonical_program_v1,
    load_bigcodebench_v1,
)
from coordpy.bigcodebench_executor_v1 import run_bigcodebench_executor_v1
from coordpy.bigcodebench_reflexion_bench_v1 import select_bigcodebench_slice_v1

DEFAULT_VENV = os.path.expanduser("~/.cache/coordpy/bcb_venv/bin/python")
OUT_DIR = "results/w110/bigcodebench_preflight"
MISSING_RE = re.compile(r"No module named '([^']+)'")


def _canon(payload) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      default=str).encode("utf-8")


def _sha(payload) -> str:
    return hashlib.sha256(_canon(payload)).hexdigest()


def _slice_cid(slice_probs) -> str:
    return _sha({"kind": "w110_bigcodebench_slice_v1",
                 "task_ids": [p.task_id for p in slice_probs],
                 "problem_cids": [p.problem_cid() for p in slice_probs]})


# --- P2 synthetic executor self-test fixtures ---------------------------------
_SYN_TEST = (
    "import unittest\n"
    "class TestCases(unittest.TestCase):\n"
    "    def test_a(self):\n"
    "        self.assertEqual(task_func(2, 3), 5)\n"
    "    def test_b(self):\n"
    "        self.assertEqual(task_func(-1, 1), 0)\n")
_SYN_GOLD = "def task_func(a, b):\n    return a + b\n"
_SYN_WRONG = "def task_func(a, b):\n    return a - b\n"
_SYN_LOOP = "def task_func(a, b):\n    while True:\n        pass\n"


def _p2_executor_self_test(py: str, probs) -> dict:
    g = run_bigcodebench_executor_v1(
        problem_id="syn-gold", test_source=_SYN_TEST, entry_point="task_func",
        candidate_code=_SYN_GOLD, python_exe=py)
    w = run_bigcodebench_executor_v1(
        problem_id="syn-wrong", test_source=_SYN_TEST, entry_point="task_func",
        candidate_code=_SYN_WRONG, python_exe=py)
    lo = run_bigcodebench_executor_v1(
        problem_id="syn-loop", test_source=_SYN_TEST, entry_point="task_func",
        candidate_code=_SYN_LOOP, python_exe=py, timeout_s=3.0,
        kill_after_s=5.0)
    # real gold passes; a corrupted gold (body deleted) fails — pick a problem
    # whose gold is green so the "real gold PASSes" leg is meaningful.
    real_pass = real_fail = None
    for p in probs[:60]:
        gr = run_bigcodebench_executor_v1(
            problem_id=p.task_id, test_source=p.test,
            entry_point=p.entry_point,
            candidate_code=canonical_program_v1(p), python_exe=py)
        if gr.passed:
            corrupt = p.complete_prompt + "\n    raise NotImplementedError\n"
            cr = run_bigcodebench_executor_v1(
                problem_id=p.task_id, test_source=p.test,
                entry_point=p.entry_point, candidate_code=corrupt,
                python_exe=py)
            real_pass, real_fail = bool(gr.passed), bool(not cr.passed)
            break
    checks = {
        "synthetic_gold_pass": bool(g.passed),
        "synthetic_wrong_fail": bool(not w.passed),
        "infinite_loop_timeout": bool(lo.timed_out),
        "real_gold_pass": bool(real_pass),
        "corrupted_gold_fail": bool(real_fail),
    }
    return {"pass": all(checks.values()), "checks": checks}


def _gold_green_scan(py: str, probs, workers: int) -> dict:
    results: dict[str, dict] = {}

    def one(p):
        r = run_bigcodebench_executor_v1(
            problem_id=p.task_id, test_source=p.test,
            entry_point=p.entry_point,
            candidate_code=canonical_program_v1(p), python_exe=py)
        mm = MISSING_RE.search(r.stderr_tail)
        return p.task_id, {
            "passed": bool(r.passed), "rc": int(r.returncode),
            "timed_out": bool(r.timed_out),
            "missing_module": (mm.group(1).split(".")[0] if mm else "")}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for tid, rec in ex.map(one, probs):
            results[tid] = rec
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description="W110 BigCodeBench preflight")
    ap.add_argument("--venv-python", default=DEFAULT_VENV)
    ap.add_argument("--sha", default=os.environ.get(
        "BIGCODEBENCH_TRUSTED_SHA256_OVERRIDE"))
    ap.add_argument("--n-slice", type=int, default=30)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(OUT_DIR,
                                                  "preflight_verdict.json"))
    args = ap.parse_args()
    if not args.sha:
        raise SystemExit("set --sha or BIGCODEBENCH_TRUSTED_SHA256_OVERRIDE")
    py = args.venv_python
    if not os.path.exists(py):
        raise SystemExit(f"venv python missing: {py} (run the venv setup)")

    t0 = time.time()
    # P1
    probs = load_bigcodebench_v1(expected_sha256=args.sha)
    p1 = {"pass": len(probs) >= 1000, "n_problems": len(probs),
          "split": BIGCODEBENCH_DEFAULT_SPLIT, "corpus_sha256": args.sha}
    print(f"P1 corpus: {len(probs)} problems (SHA {args.sha[:12]}…)")

    # P2
    p2 = _p2_executor_self_test(py, probs)
    print(f"P2 executor self-test: {p2['pass']}  {p2['checks']}")

    # P3 gold-green scan (threaded)
    print(f"P3 gold-green scan over {len(probs)} problems "
          f"(workers={args.workers})…")
    scan = _gold_green_scan(py, probs, args.workers)
    gold_green = [p for p in probs if scan[p.task_id]["passed"]]
    missing_hist: dict[str, int] = {}
    nondep_fail = 0
    for tid, rec in scan.items():
        if rec["passed"]:
            continue
        if rec["missing_module"]:
            missing_hist[rec["missing_module"]] = missing_hist.get(
                rec["missing_module"], 0) + 1
        else:
            nondep_fail += 1
    p3 = {
        "pass": len(gold_green) >= args.n_slice,
        "n_gold_green": len(gold_green),
        "n_dropped_missing_dep": sum(missing_hist.values()),
        "n_dropped_nondep": nondep_fail,
        "missing_module_histogram": dict(sorted(
            missing_hist.items(), key=lambda kv: -kv[1])),
        "contamination": {
            "c7_grade": "A (release-date resistance)",
            "release": "2024-06 (post Llama-3.x ~2024-01 cutoff)",
            "role": "SECOND contamination-resistant benchmark",
            "caps": ["W110-L-BIGCODEBENCH-RELEASE-DATE-RESISTANCE-NOT-CONTEST-DATE-CAP",
                     "W110-L-BIGCODEBENCH-GOLD-GREEN-SUBSET-ONLY-CAP"]},
    }
    print(f"P3 gold-green: {len(gold_green)}/{len(probs)} "
          f"(dropped dep={p3['n_dropped_missing_dep']} "
          f"nondep={nondep_fail}); top-missing={list(p3['missing_module_histogram'].items())[:6]}")

    # P4 deterministic slice over gold_green
    sl1 = select_bigcodebench_slice_v1(gold_green, n_problems=args.n_slice)
    sl2 = select_bigcodebench_slice_v1(gold_green, n_problems=args.n_slice)
    cid1, cid2 = _slice_cid(sl1), _slice_cid(sl2)
    from collections import Counter
    sl_buckets = Counter()
    for p in sl1:
        n = p.n_libs()
        sl_buckets["libs2" if n == 2 else ("libs3plus" if n >= 3 else "libs0_1")] += 1
    p4 = {"pass": (cid1 == cid2 and len(sl1) == args.n_slice),
          "slice_cid": cid1, "n_slice": len(sl1),
          "slice_task_ids": [p.task_id for p in sl1],
          "slice_n_libs_buckets": dict(sl_buckets)}
    print(f"P4 slice: {len(sl1)} problems, CID {cid1[:12]}… "
          f"(deterministic={cid1==cid2}); buckets={dict(sl_buckets)}")

    overall = bool(p1["pass"] and p2["pass"] and p3["pass"] and p4["pass"])
    verdict = {
        "schema": "coordpy.w110_bigcodebench_preflight.v1",
        "overall_pass": overall,
        "venv_python": py,
        "corpus_sha256": args.sha,
        "wall_s": round(time.time() - t0, 1),
        "P1_corpus_integrity": p1,
        "P2_executor_self_test": p2,
        "P3_gold_green": p3,
        "P4_deterministic_slice": p4,
        "gold_green_task_ids": [p.task_id for p in gold_green],
    }
    verdict["verdict_cid"] = _sha({k: v for k, v in verdict.items()
                                   if k != "gold_green_task_ids"})
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(verdict, f, indent=2, sort_keys=True)

    print("\n=== W110 BigCodeBench preflight ===")
    print(f"  OVERALL: {'PASS — pilot EARNED' if overall else 'FAIL'}")
    print(f"  verdict : {args.out}")
    print(f"  verdict CID: {verdict['verdict_cid']}")
    print(f"  slice CID  : {p4['slice_cid']}")
    print(f"  wall    : {verdict['wall_s']}s")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
