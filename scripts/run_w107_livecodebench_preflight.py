#!/usr/bin/env python3
"""W107-β — LiveCodeBench next-code-battlefield NIM-free preflight.

Executes the W107 RUNBOOK § 4 selection path + cheap integrity probes
for the post-EvalPlus code battlefield, with ZERO NIM calls.  It:

* runs a REAL offline executor self-test (P2) — gold PASSes, wrong
  FAILs, infinite-loop times out — proving the functional executor
  MACHINERY is clean (gate G9 in miniature);
* runs the loader schema-shape self-checks (refuse-on-mismatch; the
  W102 silent-degeneration guard);
* records the published-baseline-grade A1@K=5 failure-residual estimate
  (P3) with operator-verification flagged;
* records the decomposition argument (P4 / S3) and the executor
  cleanness verdict (S1) and the residual verdict (S2);
* emits the LiveCodeBench-primary / APPS-backup structural-soundness
  decision (pivot to APPS iff S1∧S2∧S3 fail for LiveCodeBench);
* prints the operator corpus-fetch playbook (the real-data probes P1
  + the live A1@K=5 measurement are W108/operator work).

Usage::

    python scripts/run_w107_livecodebench_preflight.py
    python scripts/run_w107_livecodebench_preflight.py --print-fetch-playbook
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

from coordpy.livecodebench_executor_v1 import (  # noqa: E402
    run_livecodebench_executor_v1)
from coordpy.livecodebench_loader_v1 import (  # noqa: E402
    LIVECODEBENCH_KNOWN_RELEASES,
    is_functional_row,
    validate_row_schema,
)

OUT_ROOT = ROOT / "results" / "w107" / "livecodebench_preflight"

# ---- P3 published-baseline-grade residual estimate --------------------
# LiveCodeBench code_generation pass@1 for the Llama-3.x-70B class on
# recent (post-cutoff) windows sits well below the 90% saturation
# ceiling — the property that makes the +5pp Phase-2 margin bar
# structurally reachable.  These are PUBLISHED-leaderboard-grade
# figures and MUST be re-confirmed by the operator against the exact
# pinned release_vN + window before any pilot (W102 lesson: published
# baseline != re-executed local sidecar).
P3_PUBLISHED_A1_ESTIMATE = {
    "model_class": "meta/llama-3.x-70b-instruct",
    "estimated_pass_at_1_range_pct": [30.0, 50.0],
    "saturation_ceiling_pct": 90.0,
    "estimated_residual_headroom_pp_min": 40.0,
    "grade": "published-baseline-grade (NOT re-executed local sidecar)",
    "operator_must_verify": True,
    "verify_note": (
        "Re-confirm pass@1 for the EXACT pinned release_vN + "
        "post-cutoff window from the LiveCodeBench leaderboard, OR "
        "by re-executing a held A1 sidecar against the functional "
        "subset, before launching any cheap pilot."),
}


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _cid(payload: object) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


# A synthetic functional problem exercising the clean executor path:
# entry is a top-level function AND (separately) a Solution method.
_GOLD_TOPLEVEL = "def add(a, b):\n    return a + b\n"
_GOLD_SOLUTION = (
    "class Solution:\n"
    "    def add(self, a, b):\n"
    "        return a + b\n")
_WRONG = "def add(a, b):\n    return a - b\n"
_LOOP = "def add(a, b):\n    while True:\n        pass\n"
_SYNTH_TESTS = [
    {"input": json.dumps([2, 3]), "output": json.dumps(5)},
    {"input": json.dumps([-1, 1]), "output": json.dumps(0)},
    {"input": json.dumps([10, 20]), "output": json.dumps(30)},
]


def _run_executor_self_test() -> dict:
    """P2 — real offline executor self-test (no NIM)."""
    gold_top = run_livecodebench_executor_v1(
        question_id="synth/gold_toplevel", func_name="add",
        tests=_SYNTH_TESTS, candidate_code=_GOLD_TOPLEVEL)
    gold_sol = run_livecodebench_executor_v1(
        question_id="synth/gold_solution", func_name="add",
        tests=_SYNTH_TESTS, candidate_code=_GOLD_SOLUTION)
    wrong = run_livecodebench_executor_v1(
        question_id="synth/wrong", func_name="add",
        tests=_SYNTH_TESTS, candidate_code=_WRONG)
    loop = run_livecodebench_executor_v1(
        question_id="synth/loop", func_name="add",
        tests=_SYNTH_TESTS, candidate_code=_LOOP,
        timeout_s=2.0, kill_after_s=3.0)
    checks = {
        "gold_toplevel_passes": bool(gold_top.passed),
        "gold_solution_method_passes": bool(gold_sol.passed),
        "wrong_fails": bool(not wrong.passed),
        "infinite_loop_times_out": bool(loop.timed_out),
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
        },
    }


def _run_loader_schema_self_test() -> dict:
    """Offline schema-shape self-checks for the loader's refuse-on-
    mismatch guard (W102 P5)."""
    good_func_row = {
        "question_id": "lc/1",
        "question_content": "implement add",
        "starter_code": "class Solution:\n    def add(self, a, b):",
        "public_test_cases": json.dumps([
            {"input": "[1,2]", "output": "3",
             "testtype": "functional"}]),
        "metadata": {"func_name": "add"},
    }
    stdin_row = dict(good_func_row)
    stdin_row["starter_code"] = ""
    missing_field_row = {
        "question_id": "lc/2", "starter_code": "def f():",
        "public_test_cases": "[]",
    }  # missing question_content
    ok_good, _ = validate_row_schema(good_func_row)
    ok_stdin, _ = validate_row_schema(stdin_row)
    ok_missing, missing_reason = validate_row_schema(missing_field_row)
    checks = {
        "valid_functional_row_accepted": bool(ok_good),
        "valid_stdin_row_accepted_then_filtered": bool(
            ok_stdin and not is_functional_row(stdin_row)),
        "functional_row_is_functional": bool(
            is_functional_row(good_func_row)),
        "missing_required_field_refused": bool(not ok_missing),
    }
    return {
        "checks": checks,
        "all_pass": all(checks.values()),
        "missing_field_reason": missing_reason,
    }


def _selection_verdict(exec_ok: bool, loader_ok: bool) -> dict:
    """The W107 RUNBOOK § 4 C1–C8 + S1–S3 selection verdict."""
    # S1 executor cleanness reachable: proven by the offline self-test.
    s1 = bool(exec_ok and loader_ok)
    # S2 NIM-free residual exists: published-baseline-grade headroom
    # >= +10pp documented (operator must verify exact number).
    s2 = bool(
        P3_PUBLISHED_A1_ESTIMATE["estimated_residual_headroom_pp_min"]
        >= 10.0)
    # S3 W89 decomposition fit: functional form == "produce a complete
    # function", the exact W89 read->solve->execute->reflect->repair
    # shape.
    s3 = True
    lcb_sound = bool(s1 and s2 and s3)
    primary = "LiveCodeBench" if lcb_sound else "APPS"
    return {
        "c1_ceiling_pressure": "A (published residual >= +40pp; non-saturated)",
        "c2_executor_cleanness": (
            "B->A: functional subset has a clean deterministic "
            "subprocess executor (proven offline)"),
        "c3_decomposition_fit": "A: functional form == W89 complete-function shape",
        "c7_contamination_resistance": (
            "A: time-anchored release_vN — the decisive publication-"
            "grade property; the reason LiveCodeBench leads APPS"),
        "S1_executor_clean_subset_exists": s1,
        "S2_nim_free_residual_exists": s2,
        "S2_grade": P3_PUBLISHED_A1_ESTIMATE["grade"],
        "S3_w89_decomposition_fits": s3,
        "livecodebench_structurally_sound": lcb_sound,
        "primary": primary,
        "backup": "APPS" if primary == "LiveCodeBench" else "LiveCodeBench",
        "pivot_triggered": bool(not lcb_sound),
        "rationale": (
            "LiveCodeBench REMAINS primary: S1∧S2∧S3 all hold for "
            "the functional (starter_code) subset, and its "
            "time-anchored contamination resistance (C7) is the "
            "decisive property for a publication-grade multi-agent-"
            "superiority claim. APPS is held as the structural-pivot "
            "backup (cleaner stdin/stdout executor fit, but 2021 "
            "vintage => contamination-exposed, which would weaken any "
            "claim built on it)."
            if lcb_sound else
            "LiveCodeBench failed a structural-soundness gate; "
            "pivoting to APPS in the same milestone per RUNBOOK § 4.2."),
    }


FETCH_PLAYBOOK = r"""
# W107-β operator corpus-fetch playbook (run before any W108 pilot)
# ----------------------------------------------------------------------
# 1) Pick + pin ONE release that is AFTER the model's training cutoff
#    (the contamination-resistance window). Example: release_v5.
#
# 2) Fetch the release JSONL from Hugging Face and record its SHA-256:
#      mkdir -p ~/.cache/coordpy
#      # (use the datasets lib or the resolve/ URL for the pinned file)
#      sha256sum ~/.cache/coordpy/livecodebench-release_v5.jsonl
#      export LIVECODEBENCH_TRUSTED_SHA256_OVERRIDE=<sha_from_above>
#
# 3) CONFIRM THE SCHEMA against the live rows (W102 lesson): verify the
#    field names this loader depends on (question_content, starter_code,
#    public_test_cases, metadata.func_name) and the test-case JSON
#    encoding. If they differ, adjust the loader aliases / executor
#    decoder at THIS step — do NOT let a wrong assumption silently
#    degrade a pilot.
#
# 4) Re-run this preflight with the corpus present to compute the REAL
#    P1 corpus-integrity + functional-subset-size + live A1@K=5
#    residual probes. Only then is a W108 cheap pilot earnable.
""".strip()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="W107-β LiveCodeBench NIM-free preflight")
    ap.add_argument(
        "--print-fetch-playbook", action="store_true",
        help="Print the operator corpus-fetch playbook and exit")
    ap.add_argument("--out-root", default=str(OUT_ROOT))
    args = ap.parse_args()
    if args.print_fetch_playbook:
        print(FETCH_PLAYBOOK)
        return 0

    exec_self = _run_executor_self_test()
    loader_self = _run_loader_schema_self_test()
    selection = _selection_verdict(
        exec_self["all_pass"], loader_self["all_pass"])

    verdict = {
        "schema": "coordpy.w107_livecodebench_preflight.v1",
        "milestone": "W107-beta",
        "battlefield_lead": selection["primary"],
        "battlefield_backup": selection["backup"],
        "known_releases": list(LIVECODEBENCH_KNOWN_RELEASES),
        "P2_executor_self_test": exec_self,
        "loader_schema_self_test": loader_self,
        "P3_published_baseline_residual": P3_PUBLISHED_A1_ESTIMATE,
        "selection_verdict": selection,
        "offline_probes_pass": bool(
            exec_self["all_pass"] and loader_self["all_pass"]),
        "operator_deferred_probes": [
            "P1 corpus integrity (SHA pin) — needs fetched release_vN",
            "functional-subset size >= 30 — needs fetched corpus",
            "live A1@K=5 residual measurement — needs fetched corpus",
        ],
        "ts_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    verdict["verdict_cid"] = _cid(
        {k: v for k, v in verdict.items() if k != "ts_utc"})

    out_dir = Path(args.out_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "preflight_verdict.json"
    out_path.write_text(json.dumps(verdict, indent=2, default=str))

    print("=== W107-β LiveCodeBench NIM-free preflight ===")
    print(f"  executor self-test  : "
          f"{'PASS' if exec_self['all_pass'] else 'FAIL'} "
          f"{exec_self['checks']}")
    print(f"  loader schema test  : "
          f"{'PASS' if loader_self['all_pass'] else 'FAIL'} "
          f"{loader_self['checks']}")
    print(f"  S1 executor clean   : "
          f"{selection['S1_executor_clean_subset_exists']}")
    print(f"  S2 NIM-free residual: "
          f"{selection['S2_nim_free_residual_exists']} "
          f"({selection['S2_grade']})")
    print(f"  S3 W89 decomp fit   : "
          f"{selection['S3_w89_decomposition_fits']}")
    print(f"  LEAD battlefield    : {selection['primary']} "
          f"(backup {selection['backup']}; "
          f"pivot={selection['pivot_triggered']})")
    print(f"  verdict_cid         : {verdict['verdict_cid']}")
    print(f"  written             : {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
