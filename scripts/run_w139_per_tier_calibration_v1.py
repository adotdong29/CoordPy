#!/usr/bin/env python3
"""W139 Lane α — per-MODEL (per-tier) band calibration + witness-usability capability prior (RUNBOOK §6).

For the SAME shared family set, sweeps the continuous-N knob grid and measures A1-AS-A-RATE at EVERY
ladder tier (8b / 3.1-70b / 3.3-70b anchor), then admits a SEPARATE per-tier band (each tier at its own
p≈0.5).  After the bands are found it measures, per tier, the witness-usability rate (the capability
prior the W139 capability-matched controller routes on) on that tier's representative in-band cell.

Writes ``results/w139/w139_per_tier_calibration_v1.json`` with per-cell per-tier verdicts +
per-tier band knobs + witness-usability + per_tier_calibration_cid (LOCKED before any Lane-β spend).

Run:  .venv/bin/python scripts/run_w139_per_tier_calibration_v1.py [--n-cal 4] [--k 5] [--workers 6]
        [--cx-knobs 6000,20000,50000] [--func-knobs 1500,4000,30000] [--n-wu 6] [--smoke]
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

from coordpy.headroom_band_slate_v3 import (  # noqa: E402
    build_band_candidates_v3, band_slate_fingerprint_cid_v1)
from coordpy.per_tier_band_calibration_v1 import (  # noqa: E402
    CX_KNOB_GRID_V139, FUNC_KNOB_GRID_V139, LADDER_V2, W139_FAMILIES,
    build_per_tier_band_report_v1, calibrate_cell_per_tier_v1)
from coordpy.capability_matched_witness_compiler_v1 import measure_witness_usability_v1  # noqa: E402
from scripts.run_w108_livecodebench_pilot import _build_nim_gen  # noqa: E402

OUT = "results/w139/w139_per_tier_calibration_v1.json"
CALLS = "results/w139/w139_per_tier_calibration_calls.jsonl"
WITNESS_SEED = 999_139


def _knobs(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in s.split(",") if x.strip())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cal", type=int, default=4, help="calibration instances per cell (A0 + A1)")
    ap.add_argument("--k", type=int, default=5, help="K for A1 (any-of-K)")
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--timeout", type=float, default=8.0)
    ap.add_argument("--mint-timeout", type=float, default=1.0)
    ap.add_argument("--cx-knobs", type=_knobs, default=(6_000, 20_000, 50_000))
    ap.add_argument("--func-knobs", type=_knobs, default=(1_500, 4_000, 30_000))
    ap.add_argument("--families", default=",".join(W139_FAMILIES))
    ap.add_argument("--n-wu", type=int, default=6, help="witness-usability probes per tier")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="2 cells, n_cal=2 — gauge endpoint speed")
    args = ap.parse_args()

    os.makedirs("results/w139", exist_ok=True)
    fams = set(x for x in args.families.split(",") if x.strip())
    slate_cid = band_slate_fingerprint_cid_v1(cx_knobs=tuple(args.cx_knobs),
                                              func_knobs=tuple(args.func_knobs))
    cands = [c for c in build_band_candidates_v3(cx_knobs=tuple(args.cx_knobs),
                                                 func_knobs=tuple(args.func_knobs))
             if c.family in fams]
    if args.smoke:
        cands = cands[:2]
        args.n_cal, args.n_wu = 2, 2
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

    def on_call(cell: str, model: str, idx: int) -> None:
        with lock:
            n_calls["n"] += 1

    print(f"slate_fingerprint_cid: {slate_cid[:16]}")
    print(f"per-tier calibrating {len(cands)} cells x 3 tiers "
          f"(n_cal={args.n_cal} A0 + n_cal x K={args.k} A1 each) on {[m.model_id for m in LADDER_V2]}; "
          f"workers={args.workers}")
    t0 = time.time()

    def run_cell(c):
        t = time.time()
        v = calibrate_cell_per_tier_v1(
            c, gen_for_model=gen_for_model, ladder=LADDER_V2, n_cal=args.n_cal, K=args.k,
            max_tokens=args.max_tokens, timeout_s=args.timeout, mint_timeout_s=args.mint_timeout,
            on_call=on_call)
        bits = "  ".join(f"{t.tier}:a0={t.a0_rate:.2f},a1={t.a1_rate:.2f}{'*' if t.in_band() else ''}"
                         for t in v.per_tier)
        print(f"  [{time.time()-t:5.0f}s] {c.cell_id:30s} {bits}  inband={list(v.in_band_tiers())}",
              flush=True)
        return v

    verdicts = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_cell, c): c for c in cands}
        for f in as_completed(futs):
            verdicts.append(f.result())
    verdicts.sort(key=lambda v: v.cell_id)

    rep = build_per_tier_band_report_v1(verdicts, ladder=LADDER_V2, n_cal=args.n_cal, K=args.k)
    band_for_tier = {t: rep.band_for_tier(t) for t in rep.all_tier_names()}

    # witness-usability: per tier, on that tier's representative in-band cell (prefer a COMPLEXITY cell,
    # which is the witness's home mode; fall back to any in-band cell).
    cand_by_cell = {c.cell_id: c for c in cands}
    usability = {}
    for cm in LADDER_V2:
        cells = band_for_tier.get(cm.tier, {})
        pick = None
        for fam, cell in sorted(cells.items()):
            cnd = cand_by_cell.get(cell.cell_id)
            if cnd is not None and cnd.mode == "COMPLEXITY_BLIND":
                pick = cnd
                break
        if pick is None and cells:
            any_cell = sorted(cells.values(), key=lambda c: c.cell_id)[0]
            pick = cand_by_cell.get(any_cell.cell_id)
        if pick is None:
            usability[cm.tier] = {"model_id": cm.model_id, "tier": cm.tier, "n_eligible": 0,
                                  "witness_usability_rate": 0.0, "witness_eligible": False,
                                  "note": "no in-band cell for this tier"}
            continue
        wu = measure_witness_usability_v1(
            pick.template, gen=gen_for_model(cm.model_id), model_id=cm.model_id, tier=cm.tier,
            n_wu=args.n_wu, witness_seed=WITNESS_SEED, max_tokens=args.max_tokens,
            timeout_s=args.timeout, mint_timeout_s=args.mint_timeout)
        d = wu.to_dict()
        d["probe_cell"] = pick.cell_id
        usability[cm.tier] = d
        print(f"  witness-usability {cm.tier:6s} ({pick.cell_id}): rate={wu.rate:.2f} "
              f"eligible={wu.witness_eligible}", flush=True)

    calls_fp.close()
    wall = time.time() - t0

    out = rep.to_dict()
    out["slate_fingerprint_cid"] = slate_cid
    out["cx_knobs"] = list(args.cx_knobs)
    out["func_knobs"] = list(args.func_knobs)
    out["witness_usability"] = usability
    out["witness_eligible_tiers"] = [t for t, d in usability.items() if d.get("witness_eligible")]
    out["wall_s"] = round(wall, 1)
    out["n_nim_calls"] = n_calls["n"]
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)

    print("\n=== W139 PER-TIER BAND CALIBRATION ===")
    for t in rep.all_tier_names():
        fams_t = band_for_tier.get(t, {})
        print(f"  {t:6s} band ({len(fams_t)} fam): "
              + ", ".join(f"{f}@{c.knob_value}(a1={c.tier(t).a1_rate:.2f})"
                          for f, c in sorted(fams_t.items())))
    print(f"shared families (anchor & >=1 other tier): {list(rep.shared_families())}")
    print(f"witness-eligible tiers: {out['witness_eligible_tiers']}")
    print(f"per_tier_calibration_cid: {rep.per_tier_calibration_cid()}")
    print(f"wall {wall:.0f}s  NIM calls {n_calls['n']} -> {OUT}")
    enough = len(rep.shared_families()) >= 1
    print(f"\nLANE-α GO/NO-GO: shared family on anchor & >=1 other tier ? {enough} "
          f"(shared={list(rep.shared_families())})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
