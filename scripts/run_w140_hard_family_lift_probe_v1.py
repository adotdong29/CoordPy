#!/usr/bin/env python3
"""W140 iter-2 — hard-family LIFT probe (falsifier-first): does the tutor BEAT resampling where A1≈0?

The W140 dev showed the tutor TIES A1 on families the 8B already partly knows (A1 high) — at K=5,
self-consistency captures the same headroom. Failure analysis (the 8B sidecar) showed the tutor GENUINELY
shifts the 8B's distribution toward the technique (two-deque: 0/60 self-generated → 24/24 with the holed
skeleton on public). So the LIFT must appear where the model's plain distribution has ~ZERO mass on the
technique: a knob LARGE enough that the correct brute TLEs (so A1≈0) OR a technique the 8B does not know
(binary-search-on-answer, prefix-min/suffix-max, monotonic-stack, two-deque).

FALSIFIER-FIRST (pre-committed): run the baseline A0/A1/B0 on each cell; a cell only QUALIFIES for the
lift claim if its A1 ≤ A1_CAP (the substitute regime is excluded by construction). Then run the tutor
arms (T2 = skeleton, T4 = controller) ONLY on qualifying cells. Same-budget K=5; graded on secret.

Run:  .venv/bin/python scripts/run_w140_hard_family_lift_probe_v1.py
        --cells subarrays_sum_and_range@30000,sum_nearest_smaller_left@50000,max_j_minus_i_le@50000,kth_smallest_pair_distance@20000
        --tiers small,strong --n 4 [--a1-cap 0.34] [--dry-run]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from coordpy.headroom_band_slate_v3 import CX_FACTORIES, EXTRA_CX_FACTORIES, FUNC_FACTORIES
from coordpy.parser_neutral_io_v1 import parser_neutrality_gate_v1
from coordpy.resistant_by_construction_battlefield_v1 import mint_problem_v1
from coordpy.headroom_band_corpus_v3 import DEFAULT_MINTED_DATE
from coordpy.icpc_reflexion_bench_v1 import IcpcBenchConfigV1, run_icpc_reflexion_bench_v1
from coordpy.per_tier_band_calibration_v1 import LADDER_V2
from coordpy import family_tutor_compiler_v1 as T
from coordpy import cross_tier_tutor_bench_v1 as B
from coordpy.family_tutor_compiler_v1 import _sha256_hex
from scripts.run_w108_livecodebench_pilot import _build_nim_gen

EXEC_TIMEOUT_S = 8.0
PROBE_SEED_BASE = 140_950_000
A1_CAP_DEFAULT = 0.34  # a cell qualifies for the lift claim only if A1 <= this (substitute regime out)


def _factory_for(fam):
    return CX_FACTORIES.get(fam) or FUNC_FACTORIES.get(fam) or EXTRA_CX_FACTORIES.get(fam)


def _seed_for(fam, knob):
    return PROBE_SEED_BASE + int(_sha256_hex({"hf": fam, "k": knob})[:6], 16) % 100000


def _acc(a):
    return sum(1 for x in a if x) / len(a) if a else 0.0


def _mint_cell(fam, knob, n):
    fac = _factory_for(fam)
    if fac is None:
        return None, []
    tmpl = fac(int(knob))
    base = _seed_for(fam, knob)
    out = []
    for r in range(int(n)):
        p = mint_problem_v1(tmpl.minted, global_seed=base + r, timeout_s=1.0)
        hc1 = parser_neutrality_gate_v1([i for i, _ in p.secret_cases], tmpl.io_shape)
        if p.gates.admitted and hc1.is_parser_neutral:
            out.append(p)
    return tmpl, out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=("subarrays_sum_and_range@30000,sum_nearest_smaller_left@50000,"
                                        "max_j_minus_i_le@50000,kth_smallest_pair_distance@20000"))
    ap.add_argument("--tiers", default="small,strong")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--a1-cap", type=float, default=A1_CAP_DEFAULT)
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--timeout", type=float, default=EXEC_TIMEOUT_S,
                    help="grading executor timeout (W134 verified timeout-invariant; 4.0 ~2x faster "
                         "than 8.0 with identical naive-TLE/technique-pass discrimination)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    exec_timeout = float(args.timeout)

    cells = []
    for c in args.cells.split(","):
        c = c.strip()
        if not c:
            continue
        fam, knob = c.rsplit("@", 1)
        cells.append((fam, int(knob)))
    want = [t.strip() for t in args.tiers.split(",") if t.strip()]
    ladder = [m for m in LADDER_V2 if m.tier in want]

    print(f"cells: {cells}")
    print(f"tiers: {[m.tier for m in ladder]}  n={args.n}  A1_CAP={args.a1_cap}")
    # $0 pre-flight: confirm every cell mints + admits + the tutor is leak-clean
    for fam, knob in cells:
        tmpl, probs = _mint_cell(fam, knob, args.n)
        if not probs:
            print(f"  [PREFLIGHT] {fam}@{knob}: NO admitted instances — SKIP")
            continue
        tut = B.compile_family_tutors_v1(tmpl)[T.TC2_REWRITE]
        lr = T.tutor_leak_gate_v1(tut, tmpl, probs[0])
        comp = T.skeleton_is_completable_v1(tmpl, probs[0])
        print(f"  [PREFLIGHT] {fam}@{knob}: admitted={len(probs)}/{args.n} algo_sig={tmpl.minted.algo_sig} "
              f"tutor_leak={lr.leaked} completable={comp.get('completable')} obs={B.tutor_observed_kind_for_template(tmpl)}")
    if args.dry_run:
        print("--dry-run: stopping before NIM.")
        return 0

    run_id = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = os.path.join(ROOT, "results", "w140", "hard_lift", f"w140_hardlift_{run_id}")
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    results = {}
    for m in ladder:
        sc = open(os.path.join(out_dir, f"calls_{m.tier}.jsonl"), "w")
        gen = _build_nim_gen(model=m.model_id,
                             sidecar_writer=lambda r: (sc.write(json.dumps(r) + "\n"), sc.flush()))
        results[m.tier] = {}
        for fam, knob in cells:
            tmpl, probs = _mint_cell(fam, knob, args.n)
            if not probs:
                continue
            pilots = [p.to_pilot_problem(minted_date=DEFAULT_MINTED_DATE) for p in probs]
            cfg = IcpcBenchConfigV1(K_multi_sample=5, seeds=(_seed_for(fam, knob),),
                                    sampling_temperature=0.7, max_tokens_per_call=int(args.max_tokens),
                                    executor_timeout_s=exec_timeout)
            print(f"  [{m.tier}] {fam}@{knob}: baseline A0/A1/B0 on {len(probs)} ({m.model_id})...", flush=True)
            base = run_icpc_reflexion_bench_v1(gen=gen, model_id=m.model_id, subset=pilots, config=cfg)
            a1 = base.a1_mean_pass_at_1
            b0 = base.b_mean_pass_at_1
            rec = {"family": fam, "knob": knob, "n": len(probs), "a0": base.a0_mean_pass_at_1,
                   "a1": a1, "b0": b0, "qualifies": bool(a1 <= args.a1_cap)}
            print(f"      A0={base.a0_mean_pass_at_1*100:.0f} A1={a1*100:.0f} B0={b0*100:.0f} "
                  f"qualifies(A1<= {args.a1_cap})={rec['qualifies']}", flush=True)
            if rec["qualifies"]:
                tutors = B.compile_family_tutors_v1(tmpl)
                for arm, tc in [("T2", T.TC2_REWRITE), ("T4", T.TC2_REWRITE)]:
                    passed = []
                    for p in probs:
                        if arm == "T4":
                            o, _ = B.run_tutor_controller_arm_v1(seed=_seed_for(fam, knob), template=tmpl,
                                    problem=p, tutor=tutors[tc], gen=gen, K=5, temperature=0.7,
                                    max_tokens=int(args.max_tokens), timeout_s=exec_timeout,
                                    minted_date=DEFAULT_MINTED_DATE, tutor_eligible=True)
                        else:
                            o, _ = B.run_tutor_arm_v1(seed=_seed_for(fam, knob), template=tmpl, problem=p,
                                    tutor=tutors[tc], gen=gen, K=5, temperature=0.7,
                                    max_tokens=int(args.max_tokens), timeout_s=exec_timeout,
                                    minted_date=DEFAULT_MINTED_DATE, arm_id=arm)
                        passed.append(bool(o.final_passed))
                    rec[arm] = {"acc": _acc(passed), "minus_a1_pp": round((_acc(passed) - a1) * 100, 2),
                                "minus_b0_pp": round((_acc(passed) - b0) * 100, 2), "passed": passed}
                    print(f"      {arm}: acc={_acc(passed)*100:.0f} -A1={rec[arm]['minus_a1_pp']:+.1f} "
                          f"-B0={rec[arm]['minus_b0_pp']:+.1f}", flush=True)
            results[m.tier][f"{fam}@{knob}"] = rec
        sc.close()

    # lift verdict: per tier, families where A1<=cap AND T2(or T4)-A1 >= +5pp
    def lift_families(tier, arm):
        r = results.get(tier, {})
        return [c for c, rec in r.items()
                if rec.get("qualifies") and rec.get(arm, {}).get("minus_a1_pp", -99) >= 5.0]
    verdict = {}
    for m in ladder:
        verdict[m.tier] = {"T2_lift_families": lift_families(m.tier, "T2"),
                           "T4_lift_families": lift_families(m.tier, "T4")}
    wall = time.time() - t0
    report = {"schema": "w140_hard_family_lift_probe_v1", "cells": [f"{f}@{k}" for f, k in cells],
              "a1_cap": args.a1_cap, "n": args.n, "results": results, "verdict": verdict,
              "wall_s": round(wall, 1)}
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)

    print("\n=== W140 HARD-FAMILY LIFT PROBE ===")
    for m in ladder:
        print(f"  [{m.tier}]")
        for c, rec in results.get(m.tier, {}).items():
            line = f"    {c:34s} A1={rec['a1']*100:.0f} B0={rec['b0']*100:.0f} qual={rec['qualifies']}"
            if "T2" in rec:
                line += f" | T2={rec['T2']['acc']*100:.0f}({rec['T2']['minus_a1_pp']:+.0f}) T4={rec['T4']['acc']*100:.0f}({rec['T4']['minus_a1_pp']:+.0f})"
            print(line)
        print(f"    -> T2 lift families: {verdict[m.tier]['T2_lift_families']}")
        print(f"    -> T4 lift families: {verdict[m.tier]['T4_lift_families']}")
    print(f"wall {wall:.0f}s -> {out_dir}/report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
