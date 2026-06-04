#!/usr/bin/env python3
"""W136-beta (targeted) — the decisive ENHANCED-T1 trap probe.

The W135 field has only 3 capability-bound traps (`se_lattice_paths_blocked` / `wa_knapsack_01` /
`wa_weighted_interval_scheduling`); the other 13 problems are solved by EVERY arm (A0=A1=B0=C1=S4=81.25%)
and by T1 too (verified: the 1D-T1 run passed every easy problem it reached).  So the §7a dev gate is
determined ENTIRELY by how many of the 3 traps the enhanced T1 cracks (and whether they span >=2 modes).

This probe runs the ENHANCED T1 (full 2-D subproblem-state grid for knapsack [items x capacity] + 1-D
dual-trajectory for lattice/weighted-interval + the recurrence-derivation scaffold) on the 3 traps PLUS
2 easy controls (1 SE, 1 WA) to confirm T1 still solves the easy field.  Same-budget K=5; reuses the
W135 dev problems.  Cheap (~5 problems x K=5 = ~25 NIM) and decisive — escalate to the full bench + eval
ONLY if it cracks >=2 traps spanning >=2 modes.

    python scripts/run_w136_trap_probe_v1.py            # ~25 NIM
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
from coordpy.algorithm_state_trace_v1 import ARM_T1_TRACE_REWRITE, run_trace_arm_v1  # noqa: E402
from coordpy.exact_oracle_witness_v1 import build_witness_probe_set_v1  # noqa: E402
from scripts.run_w108_livecodebench_pilot import _build_nim_gen  # noqa: E402

MODEL = "meta/llama-3.3-70b-instruct"
T_WSEED = 999_136
SEED = 136_201
TRAPS = ["rbc_se_lattice_paths_blocked", "rbc_wa_knapsack_01", "rbc_wa_weighted_interval_scheduling"]
CONTROLS = ["rbc_se_count_bsts_catalan", "rbc_wa_max_nonadjacent_sum"]   # 1 SE-easy + 1 WA-easy
OUT = ROOT / "results" / "w136" / "trap_probe"


def main() -> int:
    w135 = load_corpus_v1(ROOT / "results" / "w135" / "corpus" / "corpus_cache.pkl")
    if w135 is None:
        raise SystemExit("W135 corpus cache missing.")
    tmpl = w135.template_by_problem_id()
    by_base = {p.problem_id.split("__")[0]: p for p in select_dev_bench_slice_v1(w135.dev, per_family=1)}
    want = TRAPS + CONTROLS
    probs = [by_base[b] for b in want if b in by_base]
    print(f"  enhanced-T1 trap probe: {len(probs)} problems (3 traps + 2 controls), model={MODEL}")

    run_id = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    OUT.mkdir(parents=True, exist_ok=True)
    out_dir = OUT / f"probe_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    sidecar = open(out_dir / "calls.jsonl", "w")
    gen = _build_nim_gen(model=MODEL, sidecar_writer=lambda r: (sidecar.write(json.dumps(r) + "\n"), sidecar.flush()))

    t0 = time.time()
    results = {}
    for p in probs:
        probe = build_witness_probe_set_v1(tmpl[p.problem_id], p, witness_seed=T_WSEED, timeout_s=2.0)
        o, tr = run_trace_arm_v1(seed=SEED, template=tmpl[p.problem_id], problem=p, probe=probe, gen=gen,
                                 K=5, temperature=0.7, max_tokens=1536, timeout_s=8.0,
                                 arm=ARM_T1_TRACE_REWRITE, minted_date=W135_MINTED_DATE, witness_timeout_s=2.0)
        base = p.problem_id.split("__")[0]
        results[base] = {"mode": p.mode, "passed": bool(o.final_passed),
                         "per_call": list(o.per_call_passed), "trace": tr.to_dict(),
                         "is_trap": base in TRAPS}
        tag = "TRAP" if base in TRAPS else "ctrl"
        print(f"    [{tag}] {base} ({p.mode}) passed={o.final_passed} "
              f"genuinely_new={tr.any_genuinely_new}", flush=True)
    sidecar.close()

    cracked = [b for b in TRAPS if results.get(b, {}).get("passed")]
    modes_cracked = sorted({results[b]["mode"] for b in cracked})
    controls_ok = all(results.get(b, {}).get("passed") for b in CONTROLS if b in results)
    # §7a-equivalent (dev) on the trap-determined field: T1 = 13 easy + cracks; needs >=2 traps spanning
    # >=2 modes for both (T1-B0)>=+3.33 ∧ (T1-S4)>=+3.33 ∧ span (B0=S4=81.25%, the 3 traps unsolved).
    t1_score = 13 + len(cracked)
    t1_minus_b0 = round((t1_score - 13) / 16 * 100.0, 2)
    span_ok = len(modes_cracked) >= 2
    dev_gate = bool(len(cracked) >= 2 and span_ok and controls_ok)
    payload = {"schema": "coordpy.w136_trap_probe_v1", "model_id": MODEL, "seed": SEED,
               "wall_s": round(time.time() - t0, 2), "results": results,
               "traps_cracked": cracked, "n_traps_cracked": len(cracked),
               "modes_cracked": modes_cracked, "controls_ok": controls_ok,
               "implied_T1_pct": round(t1_score / 16 * 100.0, 2),
               "implied_T1_minus_B0_pp": t1_minus_b0, "implied_T1_minus_S4_pp": t1_minus_b0,
               "dev_gate_pass": dev_gate,
               "note": ("B0=S4=C1=81.25% (13/16); the 3 traps are the only headroom; cracks determine "
                        "the gate. >=2 cracks spanning >=2 modes => §7a clears => escalate to full bench "
                        "+ eval; else register the deeper machine-state cap.")}
    (out_dir / "trap_probe_report.json").write_text(json.dumps(payload, indent=2, default=str))
    (OUT / "latest.txt").write_text(out_dir.name + "\n")
    print(f"\n  traps cracked: {cracked} (modes {modes_cracked}); controls_ok={controls_ok}")
    print(f"  implied T1={payload['implied_T1_pct']:.2f}% (T1-B0=T1-S4={t1_minus_b0:+.2f}pp)  "
          f"DEV_GATE={'PASS' if dev_gate else 'FAIL'}")
    print(f"  out: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
