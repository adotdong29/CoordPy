#!/usr/bin/env python3
"""W109-α — APPS REAL-DATA contamination-control preflight (NIM-free).

The contamination-control counterpart to the W108 LiveCodeBench preflight.
W108's first contamination-RESISTANT test of the W89 mechanism (LiveCodeBench
2025) FAILed (B − A1 = −3.33 pp; MLB-2 = 25 %).  W109 asks the direct control
question: does the SAME mechanism, SAME K=5 same-budget contract, RECOVER on a
contamination-EXPOSED 2021 benchmark (APPS)?  A PASS here while LiveCodeBench
FAILed would be evidence CONSISTENT with a contamination-confound; a FAIL would
materially WEAKEN that hypothesis and tighten the mechanism's boundary.

This preflight EARNS (or denies) the W109 APPS cheap pilot.  It runs ALL probes
against the REAL pinned corpus (``~/.cache/coordpy/apps-test.jsonl``, built by
``scripts/fetch_w109_apps_corpus.py`` from ``codeparrot/apps`` @ refs/convert/
parquet), with ZERO NIM calls:

* P1 — corpus integrity: load via the SHA-pinned ``apps_loader_v1`` (refuses
  on missing cache / SHA mismatch / schema mismatch — the W102 guard).
* P2 — executor self-test: synthetic gold (top-level + Solution method) PASS,
  wrong FAIL, infinite-loop TIMEOUT, AND a REAL gold ``reverseWords`` solution
  PASSes / a deliberately-wrong one FAILs on the live corpus (no false-pass on
  real data; confirms the heterogeneous output-wrapper handling).
* P3 — loader real-data self-test: EVERY call-based problem resolves a
  non-empty ``fn_name``, functional-subset size >= 30, difficulty mix, AND the
  contamination framing (C7 = C: 2021 vintage, EXPOSED — the CONTROL property).
* P4 — deterministic, pre-committed, OUTCOME-BLIND cheap-pilot slice
  (difficulty-stratified) + its pinned slice CID + A0/A1/B spec + the SAME 9
  Phase-2 gates + MLB-1/MLB-2 sub-gates the pilot will evaluate (byte-identical
  to W108).

``overall_pass`` is True iff P1∧P2∧P3∧P4 hold — and ONLY THEN is the W109 APPS
cheap pilot earned (RUNBOOK_W109 § 4).  A PASS does NOT make APPS
publication-grade — APPS stays contamination-EXPOSED CONTROL evidence only.

Usage::

    python scripts/run_w109_apps_preflight.py
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import sys  # noqa: E402
sys.path.insert(0, str(ROOT))

from coordpy.apps_loader_v1 import (  # noqa: E402
    AppsCorpusError,
    load_apps_call_based_v1,
)
from coordpy.apps_executor_v1 import run_apps_executor_v1  # noqa: E402
from coordpy.apps_reflexion_bench_v1 import (  # noqa: E402
    select_apps_functional_slice_v1,
)

OUT_ROOT = ROOT / "results" / "w109" / "apps_preflight"

# The W109 operator-built + verified pin (fetch_w109_apps_corpus.py output).
W109_APPS_CACHE_PATH = os.path.expanduser("~/.cache/coordpy/apps-test.jsonl")
W109_APPS_JSONL_SHA256 = (
    "f6c44d76be0eea7669f0ccbd90b6b45fb03a4327d06682073b5cd8f905310918")
W109_APPS_CONVERT_COMMIT = "0f10e424e13e1c2a69f851e153097b71b6734a1f"

# Llama-3.x training-data cutoff boundary used for the contamination framing.
LLAMA_3X_CUTOFF_DATE = "2024-01-01"

# Byte-identical mechanism contract to W89/W103/W105/W108 (only the corpus +
# executor differ).
A0A1B_SPEC = {
    "A0": "stock single-shot at T=0.0 (1 model call / problem)",
    "A1": "first-pass-among-K=5 self-consistency at T=0.7 (5 calls)",
    "B": ("sequential-reflexion-K=5 at T=0.7, each turn conditioned on the "
          "cumulative (candidate, executor stderr) history (5 calls)"),
    "budget": "A1 and B are byte-exact K=5; no early-stop; same model on all arms",
    "executor_truth": ("call-based functional tests; subprocess exit 0 iff "
                       "every case matches; NO LLM-as-judge"),
}

PHASE2_GATES = {
    "G1_slice_pre_committed": "slice CID pinned before the run",
    "G2_a1_lt_90pct": "A1@K=5 < 90% (non-saturated; real headroom)",
    "G3_b_gt_a1": "B > A1 (strict)",
    "G4_margin_geq_5pp": "(B - A1) >= +5.0 pp",
    "G5_b_gt_a0_by_geq_5pp": "(B - A0) >= +5.0 pp",
    "G6_per_problem_majority": "B did not regress vs A1 on >= 16 of 30",
    "G7_budget_exact": "A1/B budget byte-exact",
    "G8_audit_chain_re_derives": "per-call CIDs + per-seed/bench Merkle re-derive",
    "G9_executor_clean": "no-LLM-judge subprocess executor",
    "MLB1_invocation_rate_geq_33pct": "reflexion invoked on >= 33% of problems",
    "MLB2_rescue_rate_geq_33pct": "of invocations, >= 33% rescued by reflexion",
}


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _cid(payload: object) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


# ---- P2 executor self-test (synthetic + REAL gold) -------------------------
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

# REAL gold on the live corpus: APPS pid 2303 (interview) fn_name reverseWords.
# Verified to PASS all 5 of its real call-based tests (W109 validation).
_REAL_GOLD_FN = "reverseWords"
_REAL_GOLD_SOLUTION = (
    "class Solution:\n"
    "    def reverseWords(self, s):\n"
    "        return ' '.join(s.split()[::-1])\n")
_REAL_WRONG_SOLUTION = (
    "class Solution:\n"
    "    def reverseWords(self, s):\n        return s\n")


def _tests_for(problem, *, max_tests: int = 25) -> list[dict]:
    out = [{"args": t.args_repr, "output": t.output_repr}
           for t in problem.tests]
    return out[:int(max_tests)] if max_tests else out


def _run_executor_self_test(subset) -> dict:
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
    real_checks: dict[str, object] = {}
    real = [p for p in subset if p.fn_name == _REAL_GOLD_FN]
    if real:
        p = real[0]
        tests = _tests_for(p)
        rg = run_apps_executor_v1(
            problem_id=p.problem_id, func_name=p.fn_name,
            tests=tests, candidate_code=_REAL_GOLD_SOLUTION)
        rw = run_apps_executor_v1(
            problem_id=p.problem_id, func_name=p.fn_name,
            tests=tests, candidate_code=_REAL_WRONG_SOLUTION)
        real_checks = {
            "real_gold_passes": bool(rg.passed),
            "real_wrong_fails": bool(not rw.passed),
            "real_gold_rc": rg.returncode,
            "real_wrong_rc": rw.returncode,
            "real_problem_id": str(p.problem_id),
            "real_fn_name": p.fn_name,
            "real_n_tests": len(tests),
        }
    checks = {
        "gold_toplevel_passes": bool(gold_top.passed),
        "gold_solution_method_passes": bool(gold_sol.passed),
        "output_wrapper_tolerance_ok": bool(gold_top.passed),  # mid test wrapped
        "wrong_fails": bool(not wrong.passed),
        "infinite_loop_times_out": bool(loop.timed_out),
        "real_gold_passes": bool(real_checks.get("real_gold_passes", False)),
        "real_wrong_fails": bool(real_checks.get("real_wrong_fails", False)),
    }
    return {
        "checks": checks,
        "all_pass": all(checks.values()),
        "evidence": {
            "gold_toplevel_rc": gold_top.returncode,
            "gold_solution_rc": gold_sol.returncode,
            "wrong_rc": wrong.returncode,
            "wrong_stderr_tail": wrong.stderr_tail,
            "loop_timed_out": loop.timed_out,
            "loop_wall_ms": loop.wall_ms,
            **real_checks,
        },
    }


def _run_loader_realdata_self_test(subset) -> dict:
    import re as _re
    n = len(subset)
    n_func_named = sum(1 for p in subset if p.fn_name)
    ds_re = _re.compile(r"ListNode|TreeNode|Optional\[")
    n_plain = sum(
        1 for p in subset if not ds_re.search(p.starter_code or ""))
    diffs = Counter(str(p.difficulty) for p in subset)
    tests_counts = sorted(len(p.tests) for p in subset)
    checks = {
        "functional_subset_size_geq_30": bool(n >= 30),
        "all_fn_names_resolved": bool(n_func_named == n and n > 0),
        "every_problem_has_tests": bool(all(len(p.tests) > 0 for p in subset)),
    }
    return {
        "checks": checks,
        "all_pass": all(checks.values()),
        "n_functional": int(n),
        "n_fn_name_resolved": int(n_func_named),
        "n_plain_arg": int(n_plain),
        "difficulty_mix": dict(diffs),
        "tests_per_problem_min": tests_counts[0] if tests_counts else 0,
        "tests_per_problem_max": tests_counts[-1] if tests_counts else 0,
        "contamination_framing": (
            "C7 = C (CONTROL): APPS is 2021 vintage, PRE the Llama-3.x "
            f"{LLAMA_3X_CUTOFF_DATE} cutoff — contamination-EXPOSED. This is "
            "the POINT of the W109 control: does the mechanism recover on "
            "exposed data after failing on contamination-resistant "
            "LiveCodeBench 2025? A PASS is NOT publication-grade."),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="W109-α APPS REAL-DATA contamination-control preflight")
    ap.add_argument("--cache-path", default=W109_APPS_CACHE_PATH)
    ap.add_argument("--expected-sha256", default=W109_APPS_JSONL_SHA256)
    ap.add_argument("--n-problems", type=int, default=30)
    ap.add_argument("--out-root", default=str(OUT_ROOT))
    args = ap.parse_args()

    # P1 — corpus integrity via the SHA-pinned loader (refuses on mismatch).
    try:
        subset = load_apps_call_based_v1(
            cache_path=str(args.cache_path),
            expected_sha256=str(args.expected_sha256))
        p1 = {"loaded": True, "n_functional": len(subset),
              "sha256_pin": str(args.expected_sha256),
              "convert_commit": W109_APPS_CONVERT_COMMIT,
              "all_pass": bool(len(subset) > 0)}
    except AppsCorpusError as e:
        p1 = {"loaded": False, "error": str(e), "all_pass": False}
        verdict = {
            "schema": "coordpy.w109_apps_preflight.v1",
            "milestone": "W109-alpha", "P1_corpus_integrity": p1,
            "overall_pass": False,
            "ts_utc": _dt.datetime.now(_dt.timezone.utc).isoformat()}
        out_dir = Path(args.out_root); out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "preflight_verdict.json").write_text(
            json.dumps(verdict, indent=2, default=str))
        print("=== W109-α APPS REAL-DATA preflight ===")
        print(f"  P1 corpus integrity: FAIL ({e})")
        return 1

    p2 = _run_executor_self_test(subset)
    p3 = _run_loader_realdata_self_test(subset)

    # P4 — deterministic, pre-committed, outcome-blind slice.
    pilot_slice = select_apps_functional_slice_v1(
        subset, n_problems=int(args.n_problems))
    slice_ids = [str(p.problem_id) for p in pilot_slice]
    slice_cid = _cid({"kind": "w109_apps_pilot_slice_v1",
                      "problem_ids": slice_ids})
    slice_diffs = Counter(str(p.difficulty) for p in pilot_slice)
    p4 = {
        "n_problems": len(pilot_slice),
        "slice_cid": slice_cid,
        "problem_ids": slice_ids,
        "difficulty_mix": dict(slice_diffs),
        "selection_rule": (
            "difficulty-stratified (largest-remainder proportional to the "
            "full call-based mix), problem_id-ordered, OUTCOME-BLIND (no "
            "sidecar exists for APPS)"),
        "a0_a1_b_spec": A0A1B_SPEC,
        "phase2_gates": PHASE2_GATES,
        "all_pass": bool(len(pilot_slice) == int(args.n_problems)),
    }

    overall = bool(
        p1["all_pass"] and p2["all_pass"] and p3["all_pass"]
        and p4["all_pass"])

    selection = {
        "S1_executor_clean_subset_exists": bool(p2["all_pass"]),
        "S2_residual_measured_by_pilot": True,
        "S2_grade": (
            "the LIVE A1@K=5 residual is measured by the W109 cheap pilot "
            "itself (G2); no in-repo APPS sidecar exists."),
        "S3_w89_decomposition_fits": True,
        "C7_contamination_resistance": (
            "C = CONTROL (contamination-EXPOSED 2021 vintage); the OPPOSITE "
            "of LiveCodeBench's C7 = A. APPS evidence is control/backup only, "
            "NEVER publication-grade time-anchored superiority."),
        "role": "contamination-EXPOSED CONTROL vs the W108 contamination-"
                "RESISTANT LiveCodeBench FAIL",
        "apps_structurally_sound_on_real_data": overall,
        "pilot_earned": overall,
    }

    verdict = {
        "schema": "coordpy.w109_apps_preflight.v1",
        "milestone": "W109-alpha",
        "lane": "APPS contaminated-control main lane",
        "P1_corpus_integrity": p1,
        "P2_executor_self_test": p2,
        "P3_loader_realdata_self_test": p3,
        "P4_pilot_slice": p4,
        "selection_verdict": selection,
        "overall_pass": overall,
        "pilot_earned": overall,
        "ts_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    verdict["verdict_cid"] = _cid(
        {k: v for k, v in verdict.items() if k != "ts_utc"})

    out_dir = Path(args.out_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "preflight_verdict.json").write_text(
        json.dumps(verdict, indent=2, default=str))

    print("=== W109-α APPS REAL-DATA contamination-control preflight ===")
    print(f"  P1 corpus integrity  : "
          f"{'PASS' if p1['all_pass'] else 'FAIL'} "
          f"({p1.get('n_functional')} call-based; SHA pinned)")
    print(f"  P2 executor test     : "
          f"{'PASS' if p2['all_pass'] else 'FAIL'} {p2['checks']}")
    print(f"  P3 loader real-data  : "
          f"{'PASS' if p3['all_pass'] else 'FAIL'} "
          f"(call_based={p3['n_functional']}; fn_name_resolved="
          f"{p3['n_fn_name_resolved']}; diff={p3['difficulty_mix']}; "
          f"tests/problem {p3['tests_per_problem_min']}..{p3['tests_per_problem_max']})")
    print(f"  P4 pilot slice       : "
          f"{'PASS' if p4['all_pass'] else 'FAIL'} "
          f"(n={p4['n_problems']}; diff={p4['difficulty_mix']})")
    print(f"     slice_cid         : {slice_cid}")
    print(f"  C7 contamination     : {selection['C7_contamination_resistance']}")
    print(f"  OVERALL              : "
          f"{'PASS — pilot EARNED (control)' if overall else 'FAIL — pilot NOT earned'}")
    print(f"  verdict_cid          : {verdict['verdict_cid']}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
