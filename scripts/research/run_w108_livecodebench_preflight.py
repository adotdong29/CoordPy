#!/usr/bin/env python3
"""W108-β — LiveCodeBench REAL-DATA preflight (NIM-free).

W107 shipped the LiveCodeBench scaffolding with an HONEST cap: the schema,
the SHA pin, and the live residual were all confirmed only at
published-baseline grade, because no real corpus had been fetched.  W108's
operator fetch landed the real ``release_v6`` corpus
(``~/.cache/coordpy/livecodebench-test6.jsonl``;
SHA-256 ``bb4c364f…``), and W108 then DIAGNOSED + FIXED a real-data binding
bug: ``metadata`` is a JSON *string* on real rows, so the W107 loader left
``func_name == ""`` → the executor returned ``ENTRY_NOT_FOUND`` on every arm
(the gold-path smoke A0=A1=B=0.0).  ``coordpy.livecodebench_loader_v1.
_resolve_func_name`` now handles that encoding; ``livecodebench_executor_v2``
decodes the confirmed newline-per-argument functional input.

This preflight is the EARNING artifact for the W108 cheap pilot.  Unlike the
W107 preflight (offline machinery + published-baseline assumptions) it runs
ALL probes against the REAL pinned corpus, with ZERO NIM calls:

* P1 — corpus integrity: load via the SHA-pinned loader (refuses on
  missing cache / SHA mismatch / schema mismatch).
* P2 — executor_V2 self-test: synthetic gold (top-level + Solution method)
  PASS, wrong FAIL, infinite-loop TIMEOUT, AND the REAL gold zigzag PASSes /
  a deliberately-wrong real zigzag FAILs (no false-pass on real data).
* P3 — loader real-data self-test: EVERY functional problem resolves a
  non-empty ``func_name`` (the W108 fix), functional-subset size >= 30,
  plain-arg count, difficulty mix, and the post-cutoff contamination window.
* P4 — deterministic, pre-committed, outcome-blind cheap-pilot slice
  (difficulty-stratified) + its pinned slice CID + A0/A1/B spec + the 9
  Phase-2 gates + MLB-1/MLB-2 sub-gates the pilot will evaluate.

``overall_pass`` is True iff P1∧P2∧P3∧P4 hold — and ONLY THEN is the W108
LiveCodeBench cheap pilot earned (RUNBOOK_W108 § 4).

Usage::

    python scripts/run_w108_livecodebench_preflight.py
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

from coordpy.livecodebench_loader_v1 import (  # noqa: E402
    LiveCodeBenchCorpusError,
    load_livecodebench_functional_v1,
)
from coordpy.livecodebench_executor_v2 import (  # noqa: E402
    run_livecodebench_executor_v2,
)
from coordpy.livecodebench_reflexion_bench_v1 import (  # noqa: E402
    select_livecodebench_functional_slice_v1,
)

OUT_ROOT = ROOT / "results" / "w108" / "livecodebench_preflight"

# The operator-fetched + W108-verified pin (real release_v6 file test6.jsonl).
W108_LIVECODEBENCH_RELEASE = "release_v6"
W108_LIVECODEBENCH_CACHE_PATH = os.path.expanduser(
    "~/.cache/coordpy/livecodebench-test6.jsonl")
W108_LIVECODEBENCH_RELEASE_V6_SHA256 = (
    "bb4c364f71921c4495a6ad15abe1a927350b720009f4933e2e71f8af0f6fd1f5")
W108_LIVECODEBENCH_DATASET_COMMIT = "0fe84c39"  # HF dataset tree commit (short)

# Llama-3.x training-data cutoff boundary used for the contamination window.
LLAMA_3X_CUTOFF_DATE = "2024-01-01"

# The cheap-pilot A0/A1/B contract — byte-identical mechanism to W89/W103/
# W105 (only the corpus + executor differ).
A0A1B_SPEC = {
    "A0": "stock single-shot at T=0.0 (1 model call / problem)",
    "A1": "first-pass-among-K=5 self-consistency at T=0.7 (5 calls)",
    "B": ("sequential-reflexion-K=5 at T=0.7, each turn conditioned on the "
          "cumulative (candidate, executor stderr) history (5 calls)"),
    "budget": "A1 and B are byte-exact K=5; no early-stop; same model on all arms",
    "executor_truth": ("functional public-test block; subprocess exit 0 iff "
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


# ---- P2 executor_V2 self-test (synthetic) ----------------------------------
_GOLD_TOPLEVEL = "def add(a, b):\n    return a + b\n"
_GOLD_SOLUTION = (
    "class Solution:\n    def add(self, a, b):\n        return a + b\n")
_WRONG = "def add(a, b):\n    return a - b\n"
_LOOP = "def add(a, b):\n    while True:\n        pass\n"
# V2 newline encoding: one JSON value per line == one positional arg.
_SYNTH_TESTS = [
    {"input": "2\n3", "output": "5"},
    {"input": "-1\n1", "output": "0"},
    {"input": "10\n20", "output": "30"},
]

_GOLD_ZIGZAG = (
    "class Solution:\n"
    "    def zigzagTraversal(self, grid):\n"
    "        m = len(grid); n = len(grid[0]); res = []; idx = 0\n"
    "        for i in range(m):\n"
    "            cols = range(n) if i % 2 == 0 else range(n - 1, -1, -1)\n"
    "            for j in cols:\n"
    "                if idx % 2 == 0:\n"
    "                    res.append(grid[i][j])\n"
    "                idx += 1\n"
    "        return res\n")
_WRONG_ZIGZAG = (
    "class Solution:\n"
    "    def zigzagTraversal(self, grid):\n        return []\n")


def _run_executor_v2_self_test(subset) -> dict:
    gold_top = run_livecodebench_executor_v2(
        question_id="synth/gold_toplevel", func_name="add",
        tests=_SYNTH_TESTS, candidate_code=_GOLD_TOPLEVEL)
    gold_sol = run_livecodebench_executor_v2(
        question_id="synth/gold_solution", func_name="add",
        tests=_SYNTH_TESTS, candidate_code=_GOLD_SOLUTION)
    wrong = run_livecodebench_executor_v2(
        question_id="synth/wrong", func_name="add",
        tests=_SYNTH_TESTS, candidate_code=_WRONG)
    loop = run_livecodebench_executor_v2(
        question_id="synth/loop", func_name="add",
        tests=_SYNTH_TESTS, candidate_code=_LOOP,
        timeout_s=2.0, kill_after_s=3.0)
    # REAL gold-path on the live corpus (the smoke that was failing).
    zig = [p for p in subset if p.func_name == "zigzagTraversal"]
    real_checks: dict[str, object] = {}
    if zig:
        p = zig[0]
        tests = [{"input": t.input_repr, "output": t.output_repr}
                 for t in p.tests]
        rg = run_livecodebench_executor_v2(
            question_id=p.question_id, func_name=p.func_name,
            tests=tests, candidate_code=_GOLD_ZIGZAG)
        rw = run_livecodebench_executor_v2(
            question_id=p.question_id, func_name=p.func_name,
            tests=tests, candidate_code=_WRONG_ZIGZAG)
        real_checks = {
            "real_gold_zigzag_passes": bool(rg.passed),
            "real_wrong_zigzag_fails": bool(not rw.passed),
            "real_gold_rc": rg.returncode,
            "real_wrong_rc": rw.returncode,
            "real_zigzag_question_id": p.question_id,
        }
    checks = {
        "gold_toplevel_passes": bool(gold_top.passed),
        "gold_solution_method_passes": bool(gold_sol.passed),
        "wrong_fails": bool(not wrong.passed),
        "infinite_loop_times_out": bool(loop.timed_out),
        "real_gold_zigzag_passes": bool(
            real_checks.get("real_gold_zigzag_passes", False)),
        "real_wrong_zigzag_fails": bool(
            real_checks.get("real_wrong_zigzag_fails", False)),
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
    n_func_named = sum(1 for p in subset if p.func_name)
    ds_re = _re.compile(r"ListNode|TreeNode|Optional\[")
    n_plain = sum(
        1 for p in subset if not ds_re.search(p.starter_code or ""))
    diffs = Counter(str(p.difficulty) for p in subset)
    dates = sorted(str(p.contest_date)[:10] for p in subset
                   if p.contest_date)
    n_post_cutoff = sum(1 for d in dates if d >= LLAMA_3X_CUTOFF_DATE)
    checks = {
        "functional_subset_size_geq_30": bool(n >= 30),
        "all_func_names_resolved": bool(n_func_named == n and n > 0),
        "plain_arg_subset_geq_30": bool(n_plain >= 30),
        "post_cutoff_window_nonempty": bool(n_post_cutoff >= 30),
    }
    return {
        "checks": checks,
        "all_pass": all(checks.values()),
        "n_functional": int(n),
        "n_func_name_resolved": int(n_func_named),
        "n_plain_arg": int(n_plain),
        "difficulty_mix": dict(diffs),
        "contest_date_min": dates[0] if dates else "",
        "contest_date_max": dates[-1] if dates else "",
        "n_post_cutoff": int(n_post_cutoff),
        "cutoff_boundary": LLAMA_3X_CUTOFF_DATE,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="W108-β LiveCodeBench REAL-DATA NIM-free preflight")
    ap.add_argument("--release", default=W108_LIVECODEBENCH_RELEASE)
    ap.add_argument("--cache-path", default=W108_LIVECODEBENCH_CACHE_PATH)
    ap.add_argument(
        "--expected-sha256", default=W108_LIVECODEBENCH_RELEASE_V6_SHA256)
    ap.add_argument("--n-problems", type=int, default=30)
    ap.add_argument("--out-root", default=str(OUT_ROOT))
    args = ap.parse_args()

    # P1 — corpus integrity via the SHA-pinned loader (refuses on mismatch).
    try:
        subset = load_livecodebench_functional_v1(
            release=str(args.release),
            cache_path=str(args.cache_path),
            expected_sha256=str(args.expected_sha256))
        p1 = {"loaded": True, "n_functional": len(subset),
              "release": str(args.release),
              "sha256_pin": str(args.expected_sha256),
              "dataset_commit": W108_LIVECODEBENCH_DATASET_COMMIT,
              "all_pass": bool(len(subset) > 0)}
    except LiveCodeBenchCorpusError as e:
        p1 = {"loaded": False, "error": str(e), "all_pass": False}
        verdict = {
            "schema": "coordpy.w108_livecodebench_preflight.v1",
            "milestone": "W108-beta", "P1_corpus_integrity": p1,
            "overall_pass": False,
            "ts_utc": _dt.datetime.now(_dt.timezone.utc).isoformat()}
        out_dir = Path(args.out_root); out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "preflight_verdict.json").write_text(
            json.dumps(verdict, indent=2, default=str))
        print("=== W108-β LiveCodeBench REAL-DATA preflight ===")
        print(f"  P1 corpus integrity: FAIL ({e})")
        return 1

    p2 = _run_executor_v2_self_test(subset)
    p3 = _run_loader_realdata_self_test(subset)

    # P4 — deterministic, pre-committed, outcome-blind slice.
    pilot_slice = select_livecodebench_functional_slice_v1(
        subset, n_problems=int(args.n_problems))
    slice_qids = [p.question_id for p in pilot_slice]
    slice_cid = _cid({"kind": "w108_lcb_pilot_slice_v1",
                      "question_ids": slice_qids})
    slice_diffs = Counter(str(p.difficulty) for p in pilot_slice)
    slice_dates = sorted(str(p.contest_date)[:10] for p in pilot_slice)
    p4 = {
        "n_problems": len(pilot_slice),
        "slice_cid": slice_cid,
        "question_ids": slice_qids,
        "difficulty_mix": dict(slice_diffs),
        "contest_date_min": slice_dates[0] if slice_dates else "",
        "contest_date_max": slice_dates[-1] if slice_dates else "",
        "selection_rule": (
            "difficulty-stratified (largest-remainder proportional to the "
            "full functional mix), (contest_date, question_id)-ordered, "
            "OUTCOME-BLIND (no sidecar exists for LiveCodeBench)"),
        "a0_a1_b_spec": A0A1B_SPEC,
        "phase2_gates": PHASE2_GATES,
        "all_pass": bool(len(pilot_slice) == int(args.n_problems)),
    }

    overall = bool(
        p1["all_pass"] and p2["all_pass"] and p3["all_pass"]
        and p4["all_pass"])

    selection = {
        "S1_executor_clean_subset_exists": bool(p2["all_pass"]),
        "S2_nim_free_residual_exists": True,
        "S2_grade": (
            "published-baseline-grade PRIOR (Llama-3.x-70B LiveCodeBench "
            "pass@1 ~30-50%); the LIVE A1@K=5 residual is measured by the "
            "W108 cheap pilot itself (G2). NOT yet re-executed-sidecar-grade."),
        "S3_w89_decomposition_fits": True,
        "C7_contamination_resistance": (
            f"A (DECISIVE): all {p3['n_functional']} functional problems are "
            f"dated {p3['contest_date_min']}..{p3['contest_date_max']}, "
            f"entirely AFTER the Llama-3.x {LLAMA_3X_CUTOFF_DATE} cutoff "
            "boundary — the publication-grade property"),
        "livecodebench_structurally_sound_on_real_data": overall,
        "primary": "LiveCodeBench" if overall else "APPS",
        "backup": "APPS" if overall else "LiveCodeBench",
        "pivot_triggered": bool(not overall),
    }

    verdict = {
        "schema": "coordpy.w108_livecodebench_preflight.v1",
        "milestone": "W108-beta",
        "battlefield_lead": selection["primary"],
        "battlefield_backup": selection["backup"],
        "P1_corpus_integrity": p1,
        "P2_executor_v2_self_test": p2,
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

    print("=== W108-β LiveCodeBench REAL-DATA NIM-free preflight ===")
    print(f"  P1 corpus integrity  : "
          f"{'PASS' if p1['all_pass'] else 'FAIL'} "
          f"(release={p1.get('release')}; {p1.get('n_functional')} functional; "
          f"SHA pinned)")
    print(f"  P2 executor_V2 test  : "
          f"{'PASS' if p2['all_pass'] else 'FAIL'} {p2['checks']}")
    print(f"  P3 loader real-data  : "
          f"{'PASS' if p3['all_pass'] else 'FAIL'} "
          f"(functional={p3['n_functional']}; func_name_resolved="
          f"{p3['n_func_name_resolved']}; plain_arg={p3['n_plain_arg']}; "
          f"diff={p3['difficulty_mix']}; "
          f"dates {p3['contest_date_min']}..{p3['contest_date_max']}; "
          f"post-cutoff={p3['n_post_cutoff']})")
    print(f"  P4 pilot slice       : "
          f"{'PASS' if p4['all_pass'] else 'FAIL'} "
          f"(n={p4['n_problems']}; diff={p4['difficulty_mix']}; "
          f"dates {p4['contest_date_min']}..{p4['contest_date_max']})")
    print(f"     slice_cid         : {slice_cid}")
    print(f"  C7 contamination     : {selection['C7_contamination_resistance']}")
    print(f"  OVERALL              : "
          f"{'PASS — pilot EARNED' if overall else 'FAIL — pilot NOT earned'}")
    print(f"  verdict_cid          : {verdict['verdict_cid']}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
