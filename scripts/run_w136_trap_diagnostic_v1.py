#!/usr/bin/env python3
"""W136 — TRAP ROOT-CAUSE DIAGNOSTIC (capture the model's ACTUAL code + the execution diff vs the oracle).

The enhanced T1 (full 2-D DP table + recurrence scaffold) cracked 0/3 traps.  "Capability-bound" is a
symptom, not a reason.  This diagnostic finds the TRUE reason: it runs enhanced-T1 on each trap with FULL
response capture, then for each attempt
  (a) extracts the model's code, grades it on the FULL secret bank,
  (b) for the first failing hidden case shows (input, expected, the model's output),
  (c) runs the MODEL'S OWN code on the trap's small sub-instances and diffs it against the oracle (the
      execution diff — WHERE the model's own computation departs from correct), and
  (d) heuristically classifies the bug (unbounded-vs-0/1 knapsack forward-loop; TLE/timeout; crash;
      greedy-not-DP; off-by-one; correct-but-fails).

    python scripts/run_w136_trap_diagnostic_v1.py            # ~15 NIM (3 traps x K=5)
"""
from __future__ import annotations

import datetime as _dt
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from coordpy.noncomplexity_structure_corpus_v1 import (  # noqa: E402
    MINTED_DATE as W135_MINTED_DATE, load_corpus_v1, select_dev_bench_slice_v1,
)
from coordpy.algorithm_state_trace_v1 import (  # noqa: E402
    ARM_T1_TRACE_REWRITE, build_algorithm_state_trace_v1,
)
from coordpy.exact_oracle_witness_v1 import build_witness_probe_set_v1  # noqa: E402
from coordpy.icpc_reflexion_bench_v1 import (  # noqa: E402
    _initial_prompt, extract_candidate_code_v1, grade_on_secret_v1,
)
from coordpy.algorithm_state_trace_v1 import _trace_reflexion_prompt  # noqa: E402
from coordpy.resistant_by_construction_battlefield_v1 import _exec_capture_v1, _tok_count  # noqa: E402
from coordpy.coordpy_icpc_battlefield_v1 import judge_icpc_output_v1  # noqa: E402
from scripts.run_w108_livecodebench_pilot import _build_nim_gen  # noqa: E402

MODEL = "meta/llama-3.3-70b-instruct"
TRAPS = ["rbc_se_lattice_paths_blocked", "rbc_wa_knapsack_01", "rbc_wa_weighted_interval_scheduling"]
OUT = ROOT / "results" / "w136" / "diagnostic"


def _classify_bug(code: str, p, first_fail, sub_diff):
    code_l = code.lower()
    tags = []
    if "for c in range(w" in code_l and "wt-1" not in code_l and "-1)" not in code_l.replace(" ", ""):
        tags.append("MAYBE_UNBOUNDED_FORWARD_LOOP")
    if re.search(r"for\s+\w+\s+in\s+range\([^)]*\):\s*\n\s*for\s+\w+\s+in\s+range\(", code):
        tags.append("HAS_NESTED_LOOP_DP")
    if "sort" in code_l and ("/" in code or "ratio" in code_l or "key=" in code_l):
        tags.append("MAYBE_GREEDY")
    if "lru_cache" in code_l or "@cache" in code_l or "def solve(" in code_l and "return" in code_l:
        tags.append("MAYBE_RECURSION")
    if first_fail and first_fail.get("observed_kind") == "TIMEOUT":
        tags.append("TLE_ON_HIDDEN")
    if first_fail and first_fail.get("observed_kind") == "RUNTIME_ERROR":
        tags.append("CRASH_ON_HIDDEN")
    if sub_diff and sub_diff.get("first_wrong_sub") is not None:
        tags.append("WRONG_ON_SMALL_SUBINSTANCE")
    elif sub_diff and sub_diff.get("n_sub") and sub_diff.get("first_wrong_sub") is None:
        tags.append("CORRECT_ON_SMALL_SUB_FAILS_HIDDEN")  # right on small, wrong on large => scale/edge
    return tags


def _exec_diff_on_subinstances(code, template, x, secret_inputs):
    """Run the MODEL'S code on the trap's small sub-instances and diff vs the oracle (ref)."""
    from coordpy.algorithm_state_trace_v1 import _typed_subinstances_v1
    out = {"n_sub": 0, "first_wrong_sub": None, "rows": []}
    for s in _typed_subinstances_v1(x):
        if s in secret_inputs:
            continue
        ro = _exec_capture_v1(template.ref_source, s, timeout_s=4.0)
        if ro.timed_out or ro.returncode != 0:
            continue
        opt = ro.stdout.strip()
        rc = _exec_capture_v1(code, s, timeout_s=4.0)
        got = rc.stdout.strip() if (not rc.timed_out and rc.returncode == 0) else (
            "TIMEOUT" if rc.timed_out else "CRASH")
        ok = (got == opt)
        out["n_sub"] += 1
        out["rows"].append({"sub_tokens": _tok_count(s), "optimal": opt[:40], "model": got[:40], "ok": ok})
        if not ok and out["first_wrong_sub"] is None:
            out["first_wrong_sub"] = {"sub": s[:80], "optimal": opt[:40], "model": got[:40]}
    return out


def _first_failing_secret(pilot, code):
    for inp, exp in pilot.secret_cases:
        r = _exec_capture_v1(code, inp, timeout_s=8.0)
        if r.timed_out:
            return {"input": inp[:80], "expected": exp[:40], "observed": "", "observed_kind": "TIMEOUT"}
        if r.returncode != 0:
            return {"input": inp[:80], "expected": exp[:40], "observed": "", "observed_kind": "RUNTIME_ERROR"}
        if not judge_icpc_output_v1(got_stdout=r.stdout, expected=exp, kind=pilot.kind, float_tol=pilot.float_tol):
            return {"input": inp[:80], "expected": exp[:40], "observed": r.stdout.strip()[:40],
                    "observed_kind": "WRONG_ANSWER"}
    return None


def main() -> int:
    w135 = load_corpus_v1(ROOT / "results" / "w135" / "corpus" / "corpus_cache.pkl")
    tmpl = w135.template_by_problem_id()
    by_base = {p.problem_id.split("__")[0]: p for p in select_dev_bench_slice_v1(w135.dev, per_family=1)}
    OUT.mkdir(parents=True, exist_ok=True)
    run_id = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = OUT / f"diag_{run_id}"; out_dir.mkdir(parents=True, exist_ok=True)
    base_gen = _build_nim_gen(model=MODEL, sidecar_writer=lambda r: None)
    t0 = time.time()
    report = {}
    for base in TRAPS:
        p = by_base[base]
        t = tmpl[p.problem_id]
        pilot = p.to_pilot_problem(minted_date=W135_MINTED_DATE)
        probe = build_witness_probe_set_v1(t, p, witness_seed=999_136, timeout_s=2.0)
        secret_inputs = {inp for inp, _ in p.secret_cases}
        history = []
        attempts = []
        for k in range(5):
            if k == 0:
                prompt = _initial_prompt(pilot)
            else:
                last_code = history[-1][0]
                trace = build_algorithm_state_trace_v1(last_code, p, probe, t, timeout_s=2.0, oracle_timeout_s=4.0)
                block = trace.to_capsule_block(ARM_T1_TRACE_REWRITE)
                prompt = _trace_reflexion_prompt(pilot, tuple(history), block, attempt_idx=k)
            text, _ = base_gen(prompt, 1536, 0.7)
            code = extract_candidate_code_v1(response_text=text)
            passed, stderr_tail, _ = grade_on_secret_v1(pilot, code, timeout_s=8.0)
            history.append((code, bool(passed), stderr_tail, ""))
            first_fail = None if passed else _first_failing_secret(pilot, code)
            sub_diff = _exec_diff_on_subinstances(code, t, probe.small[0][0] if probe.small else "", secret_inputs)
            attempts.append({"k": k, "passed": bool(passed), "code_len": len(code),
                             "code_head": code[:600], "first_failing_secret": first_fail,
                             "exec_diff_vs_oracle": sub_diff,
                             "bug_tags": _classify_bug(code, p, first_fail, sub_diff)})
            print(f"  [{base}] attempt {k}: passed={passed} tags={attempts[-1]['bug_tags']}", flush=True)
        report[base] = {"mode": p.mode, "any_passed": any(a["passed"] for a in attempts), "attempts": attempts}
    payload = {"schema": "coordpy.w136_trap_diagnostic_v1", "model_id": MODEL,
               "wall_s": round(time.time() - t0, 2), "report": report}
    (out_dir / "trap_diagnostic_report.json").write_text(json.dumps(payload, indent=2, default=str))
    (OUT / "latest.txt").write_text(out_dir.name + "\n")
    print(f"\n  wrote {out_dir / 'trap_diagnostic_report.json'} (wall {time.time()-t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
