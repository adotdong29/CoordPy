#!/usr/bin/env python3
"""W136 — EXECUTION-GROUNDED I/O-REPAIR probe (the root-cause-driven WIN attempt).

The diagnostic proved the 3 capability-bound traps fail for an I/O-FORMAT reason, not an algorithm
reason: the W132 battlefield emits all tokens WHITESPACE-FLATTENED (pairs/triples/grid-rows on one line)
while the model assumes one-per-line, so it crashes/misparses even the PUBLIC samples — and the algorithm
hints (counterexample / structure / 2-D state table) never helped because they addressed the wrong bug.
Confirmed at $0: the SAME correct 0/1-knapsack DP passes 7/7 with `sys.stdin.read().split()` parsing and
fails 7/7 with `input()`-per-line parsing.

This arm (T-IO) is execution-grounded: between attempts it RUNS the model's own code on the public
samples; if the code crashes / prints nothing / prints 'invalid' on a VALID public input, it prepends an
explicit, generic (no-leakage) whitespace-parsing directive to the algorithm-state trace.  Same-budget
K=5.  Tested on the 3 traps + 2 controls; if it cracks >=2 traps spanning >=2 modes the §7a gate clears.

    python scripts/run_w136_io_grounded_repair_probe_v1.py        # ~25 NIM
"""
from __future__ import annotations

import datetime as _dt
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from coordpy.noncomplexity_structure_corpus_v1 import (  # noqa: E402
    MINTED_DATE as W135_MINTED_DATE, load_corpus_v1, select_dev_bench_slice_v1,
)
from coordpy.algorithm_state_trace_v1 import (  # noqa: E402
    ARM_T1_TRACE_REWRITE, build_algorithm_state_trace_v1, _trace_reflexion_prompt,
)
from coordpy.exact_oracle_witness_v1 import build_witness_probe_set_v1  # noqa: E402
from coordpy.icpc_reflexion_bench_v1 import (  # noqa: E402
    IcpcArmOutcomeV1, W120_ICPC_REFLEXION_BENCH_V1_SCHEMA_VERSION,
    _initial_prompt, extract_candidate_code_v1, grade_on_secret_v1, sample_feedback_v1,
)
from coordpy.resistant_by_construction_battlefield_v1 import _exec_capture_v1  # noqa: E402
from scripts.run_w108_livecodebench_pilot import _build_nim_gen  # noqa: E402

MODEL = "meta/llama-3.3-70b-instruct"
TRAPS = ["rbc_se_lattice_paths_blocked", "rbc_wa_knapsack_01", "rbc_wa_weighted_interval_scheduling"]
CONTROLS = ["rbc_se_count_bsts_catalan", "rbc_wa_max_nonadjacent_sum"]
OUT = ROOT / "results" / "w136" / "io_repair"

IO_DIRECTIVE = (
    "CRITICAL — FIX YOUR INPUT READING FIRST. Your program crashed, printed nothing, or printed an "
    "error on a VALID sample input, which means it is reading the input in the wrong SHAPE. In THIS "
    "problem every token is WHITESPACE-separated: numbers, grid rows, pairs and triples may be separated "
    "by spaces and/or newlines interchangeably — do NOT assume one row/pair/triple per line. Read the "
    "whole input at once and consume tokens in order:\n"
    "    import sys\n    data = sys.stdin.buffer.read().split()\n"
    "    # then take data[0], data[1], ... (grid rows are the next R whitespace tokens)\n"
    "After fixing the input reading, apply your algorithm. Exact diagnostic of the failing attempt:\n")


def _code_fails_public(code, pilot) -> bool:
    """Execution-grounded I/O check: does the model's own code crash / print nothing / print 'invalid'
    on a VALID public sample? (the trap's public samples are the same whitespace-flattened shape)."""
    for inp, _exp in pilot.samples[:3]:
        r = _exec_capture_v1(code, inp, timeout_s=4.0)
        out = (r.stdout or "").strip().lower()
        if r.timed_out or r.returncode != 0 or out == "" or "invalid" in out or "error" in out:
            return True
    return False


def _run_io_arm(problem, template, probe, gen, K=5):
    pilot = problem.to_pilot_problem(minted_date=W135_MINTED_DATE)
    history = []
    per_call = []
    first_pass = -1
    io_fired = []
    for k in range(K):
        if k == 0:
            prompt = _initial_prompt(pilot)
        else:
            last_code = history[-1][0]
            trace = build_algorithm_state_trace_v1(last_code, problem, probe, template,
                                                   timeout_s=2.0, oracle_timeout_s=4.0)
            io_bad = _code_fails_public(last_code, pilot)
            io_fired.append(bool(io_bad))
            block = (IO_DIRECTIVE + trace.to_capsule_block(ARM_T1_TRACE_REWRITE)) if io_bad \
                else trace.to_capsule_block(ARM_T1_TRACE_REWRITE)
            prompt = _trace_reflexion_prompt(pilot, tuple(history), block, attempt_idx=k)
        text, _ = gen(prompt, 1536, 0.7)
        code = extract_candidate_code_v1(response_text=text)
        passed, stderr_tail, _ = grade_on_secret_v1(pilot, code, timeout_s=8.0)
        per_call.append(bool(passed))
        history.append((code, bool(passed), stderr_tail, sample_feedback_v1(pilot, code, timeout_s=8.0)))
        if passed and first_pass == -1:
            first_pass = k
    return IcpcArmOutcomeV1(
        schema=W120_ICPC_REFLEXION_BENCH_V1_SCHEMA_VERSION, seed=136_201, question_id=problem.problem_id,
        arm_id="T_IO", final_passed=bool(first_pass >= 0), n_model_calls=K,
        per_call_passed=tuple(per_call), first_pass_attempt_idx=int(first_pass)), io_fired


def main() -> int:
    w135 = load_corpus_v1(ROOT / "results" / "w135" / "corpus" / "corpus_cache.pkl")
    tmpl = w135.template_by_problem_id()
    by = {p.problem_id.split("__")[0]: p for p in select_dev_bench_slice_v1(w135.dev, per_family=1)}
    probs = [(b, by[b]) for b in (TRAPS + CONTROLS) if b in by]
    OUT.mkdir(parents=True, exist_ok=True)
    run_id = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = OUT / f"io_{run_id}"; out_dir.mkdir(parents=True, exist_ok=True)
    sc = open(out_dir / "calls.jsonl", "w")
    gen = _build_nim_gen(model=MODEL, sidecar_writer=lambda r: (sc.write(json.dumps(r) + "\n"), sc.flush()))
    print(f"  T-IO (execution-grounded I/O repair) probe: {len(probs)} problems, model={MODEL}")
    t0 = time.time()
    results = {}
    for base, p in probs:
        probe = build_witness_probe_set_v1(tmpl[p.problem_id], p, witness_seed=999_136, timeout_s=2.0)
        o, io_fired = _run_io_arm(p, tmpl[p.problem_id], probe, gen)
        results[base] = {"mode": p.mode, "passed": bool(o.final_passed),
                         "per_call": list(o.per_call_passed), "io_directive_fired": io_fired,
                         "is_trap": base in TRAPS}
        print(f"    [{'TRAP' if base in TRAPS else 'ctrl'}] {base} ({p.mode}) passed={o.final_passed} "
              f"io_fired={io_fired}", flush=True)
    sc.close()
    cracked = [b for b in TRAPS if results.get(b, {}).get("passed")]
    modes = sorted({results[b]["mode"] for b in cracked})
    controls_ok = all(results.get(b, {}).get("passed") for b in CONTROLS if b in results)
    t1_pct = round((13 + len(cracked)) / 16 * 100.0, 2)
    dev_gate = bool(len(cracked) >= 2 and len(modes) >= 2 and controls_ok)
    payload = {"schema": "coordpy.w136_io_grounded_repair_probe_v1", "model_id": MODEL,
               "wall_s": round(time.time() - t0, 2), "results": results, "traps_cracked": cracked,
               "n_traps_cracked": len(cracked), "modes_cracked": modes, "controls_ok": controls_ok,
               "implied_T_IO_pct": t1_pct, "implied_T_IO_minus_B0_pp": round(len(cracked) / 16 * 100, 2),
               "dev_gate_pass": dev_gate,
               "note": ("execution-grounded I/O-format repair; the algorithm hints (T1) addressed the "
                        "wrong bug — the true discriminator is the whitespace-flattened input format "
                        "(confirmed $0: same DP passes 7/7 robust / fails 7/7 per-line).")}
    (out_dir / "io_repair_report.json").write_text(json.dumps(payload, indent=2, default=str))
    (OUT / "latest.txt").write_text(out_dir.name + "\n")
    print(f"\n  traps cracked by T-IO: {cracked} (modes {modes}); controls_ok={controls_ok}")
    print(f"  implied T-IO={t1_pct:.2f}% (vs B0/S4/T1=81.25%)  DEV_GATE={'PASS' if dev_gate else 'FAIL'}")
    print(f"  out: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
