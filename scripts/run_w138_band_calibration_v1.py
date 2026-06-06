#!/usr/bin/env python3
"""W138 Lane α — model-ladder band calibration (A1-as-RATE + HB3/HB4) — RUNBOOK_W138 §6.

Sweeps the (family, knob) candidate grid, measures A0 across the ladder + A1 as a POPULATION RATE over
n_a1>=8 instances at the strong anchor, and admits the headroom band (HC3 ∧ HB3 ∧ HB4).  Concurrency:
cells run in a thread pool (the NIM gen is stateless/thread-safe; the sidecar write is locked).  Writes
``results/w138/w138_calibration_v1.json`` with per-cell verdicts + best-knob-per-family + calibration_cid
(LOCKED before any corpus/Lane-β spend).

Run:  .venv/bin/python scripts/run_w138_band_calibration_v1.py [--n-a0 5] [--n-a1 8] [--k-a1 5]
        [--workers 6] [--cx-knobs 20000,50000] [--func-knobs 4000,30000] [--limit N] [--smoke]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from coordpy.headroom_band_slate_v3 import build_band_candidates_v3, band_slate_fingerprint_cid_v1  # noqa: E402
from coordpy.headroom_band_calibration_v2 import (  # noqa: E402
    calibrate_band_cell_v1, build_band_calibration_report_v1)
from coordpy.model_ladder_calibration_v1 import LADDER_V1  # noqa: E402
from scripts.run_w108_livecodebench_pilot import _build_nim_gen  # noqa: E402

OUT = "results/w138/w138_calibration_v1.json"
CALLS = "results/w138/w138_calibration_calls.jsonl"


def _knobs(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in s.split(",") if x.strip())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-a0", type=int, default=5)
    ap.add_argument("--n-a1", type=int, default=8)
    ap.add_argument("--k-a1", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--timeout", type=float, default=8.0, help="LOCKED grading budget")
    ap.add_argument("--mint-timeout", type=float, default=1.0, help="bounds the $0 build (naive TLE)")
    ap.add_argument("--cx-knobs", type=_knobs, default=(20_000, 50_000))
    ap.add_argument("--func-knobs", type=_knobs, default=(4_000, 30_000))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--include-extra", action="store_true",
                    help="add the harder hedge complexity families (changes the slate CID)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N cells (probe)")
    ap.add_argument("--smoke", action="store_true", help="2 cells, n_a0=2 n_a1=2 — gauge endpoint speed")
    args = ap.parse_args()

    os.makedirs("results/w138", exist_ok=True)
    slate_cid = band_slate_fingerprint_cid_v1(cx_knobs=args.cx_knobs, func_knobs=args.func_knobs,
                                              include_extra=args.include_extra)
    cands = build_band_candidates_v3(cx_knobs=args.cx_knobs, func_knobs=args.func_knobs,
                                     include_extra=args.include_extra)
    if args.smoke:
        cands = cands[:2]
        args.n_a0, args.n_a1, args.k_a1 = 2, 2, 5
    elif args.limit > 0:
        cands = cands[:args.limit]

    calls_fp = open(CALLS, "a")
    lock = threading.Lock()

    def sidecar(rec: dict) -> None:
        with lock:
            calls_fp.write(json.dumps(rec) + "\n")
            calls_fp.flush()

    _gen_cache: dict = {}

    def gen_for_model(model_id: str):
        with lock:
            if model_id not in _gen_cache:
                _gen_cache[model_id] = _build_nim_gen(model=model_id, sidecar_writer=sidecar)
            return _gen_cache[model_id]

    n_calls = {"n": 0}

    def on_call(tpl: str, model: str, idx: int) -> None:
        with lock:
            n_calls["n"] += 1

    print(f"slate_fingerprint_cid: {slate_cid[:16]}")
    print(f"calibrating {len(cands)} cells x (n_a0={args.n_a0} both tiers + n_a1={args.n_a1} K={args.k_a1} "
          f"strong) on ladder {[m.model_id for m in LADDER_V1]}; workers={args.workers}")
    t0 = time.time()

    def run_cell(c):
        t = time.time()
        v = calibrate_band_cell_v1(
            c, gen_for_model=gen_for_model, ladder=LADDER_V1, n_a0=args.n_a0, n_a1=args.n_a1,
            K_a1=args.k_a1, max_tokens=args.max_tokens, timeout_s=args.timeout,
            mint_timeout_s=args.mint_timeout, on_call=on_call)
        print(f"  [{time.time()-t:5.0f}s] {c.cell_id:34s} a0={v.strong_a0_rate:.2f} "
              f"a1={v.strong_a1_rate:.2f} sm_a0={v.small_a0_rate:.2f} "
              f"W[{v.wilson_lo:.2f},{v.wilson_hi:.2f}] {'ADMIT' if v.admitted else 'cull':>5s} {v.reason}",
              flush=True)
        return v

    verdicts = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_cell, c): c for c in cands}
        for f in as_completed(futs):
            verdicts.append(f.result())
    verdicts.sort(key=lambda v: v.cell_id)
    calls_fp.close()
    wall = time.time() - t0

    rep = build_band_calibration_report_v1(
        verdicts, ladder=LADDER_V1, n_a0=args.n_a0, n_a1=args.n_a1, K_a1=args.k_a1)
    out = rep.to_dict()
    out["slate_fingerprint_cid"] = slate_cid
    out["wall_s"] = round(wall, 1)
    out["n_nim_calls"] = n_calls["n"]
    out["cx_knobs"] = list(args.cx_knobs)
    out["func_knobs"] = list(args.func_knobs)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)

    bk = rep.best_knob_per_family()
    print("\n=== W138 BAND CALIBRATION ===")
    print(f"admitted cells: {len(rep.admitted_cells())}/{len(verdicts)}")
    print(f"surviving FAMILIES ({len(rep.surviving_families())}): {list(rep.surviving_families())}")
    print(f"surviving MODES ({len(rep.surviving_modes())}): {list(rep.surviving_modes())}")
    print("best knob per family (a1_rate closest to 0.5):")
    for fam, c in sorted(bk.items(), key=lambda kv: kv[1].rank_key):
        print(f"  {fam:30s} {c.cell_id:30s} a1={c.strong_a1_rate:.2f} mode={c.mode}")
    print(f"calibration_cid: {rep.calibration_cid()}")
    print(f"wall {wall:.0f}s  NIM calls {n_calls['n']} -> {OUT}")
    # band-search go/no-go signal for the operator
    enough = (len(rep.surviving_families()) >= 3) or (len(rep.surviving_modes()) >= 2)
    print(f"\nBAND GO/NO-GO: surviving_families>=3 OR modes>=2 ? {enough} "
          f"(families={len(rep.surviving_families())}, modes={len(rep.surviving_modes())})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
