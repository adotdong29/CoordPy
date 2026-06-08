#!/usr/bin/env python3
"""W140 Lane α/β — per-tier TUTOR-usability calibration (RUNBOOK_W140 §4/§6).

The ONLY NEW W140 calibration NIM.  Per tier × per shared family, measures how often ONE
tutor-reflexion attempt flips a secret-FAILING plain candidate to secret-PASSING — the capability
prior ``T4`` routes on (the tutor analogue of W139's witness-usability, which was 0.00 on the 8B).
THE W140 THESIS: does compiling the witness into a family-level tutor lift the weak tier above tau?

Reuses the W139 per-tier band ($0) for the per-tier knobs + the LADDER_V2 model ladder.  Bounded:
<= 2 * n_tu * (tier,family) calls.

Run:  .venv/bin/python scripts/run_w140_tutor_usability_calibration_v1.py
        [--tc TC2_WITNESS_TO_REWRITE] [--n-tu 6] [--tiers small,mid,strong] [--dry-run]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from coordpy.headroom_band_slate_v3 import (  # noqa: E402
    CX_FACTORIES, EXTRA_CX_FACTORIES, FUNC_FACTORIES)
from coordpy.per_tier_band_calibration_v1 import LADDER_V2  # noqa: E402
from coordpy import family_tutor_compiler_v1 as T  # noqa: E402
from coordpy import cross_tier_tutor_bench_v1 as B  # noqa: E402
from coordpy.family_tutor_compiler_v1 import _sha256_hex  # noqa: E402
from scripts.run_w108_livecodebench_pilot import _build_nim_gen  # noqa: E402

DEFAULT_TC = T.TC2_REWRITE
EXEC_TIMEOUT_S = 8.0


def _factory_for(fam):
    return CX_FACTORIES.get(fam) or FUNC_FACTORIES.get(fam) or EXTRA_CX_FACTORIES.get(fam)


def _shared_families(cal, anchor_tier, weak_tier):
    bft = cal.get("band_for_tier", {})
    a, w = bft.get(anchor_tier, {}), bft.get(weak_tier, {})
    return sorted(set(a) & set(w))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration", default="results/w139/w139_per_tier_calibration_v1.json")
    ap.add_argument("--tc", default=DEFAULT_TC)
    ap.add_argument("--n-tu", type=int, default=6)
    ap.add_argument("--tiers", default="small,mid,strong")
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--out", default="results/w140/w140_tutor_usability_v1.json")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    with open(os.path.join(ROOT, args.calibration)) as f:
        cal = json.load(f)
    anchor_tier = cal.get("anchor_tier", "strong")
    weak_tier = "small"
    band_for_tier = cal.get("band_for_tier", {})
    shared = _shared_families(cal, anchor_tier, weak_tier)
    want = [t.strip() for t in args.tiers.split(",") if t.strip()]
    ladder = [m for m in LADDER_V2 if m.tier in want]

    print(f"reused per_tier_calibration_cid: {cal.get('per_tier_calibration_cid')}")
    print(f"shared families: {shared}  tc={args.tc}  n_tu={args.n_tu}")
    # plan: each tier benches the shared families it has a band cell for
    plan = []
    for m in ladder:
        cells = band_for_tier.get(m.tier, {})
        for fam in shared:
            if fam in cells:
                plan.append((m, fam, int(cells[fam]["knob_value"])))
    for m, fam, knob in plan:
        print(f"  {m.tier:6s} {m.model_id:32s} {fam}@{knob}")
    print(f"upper-bound NIM: {2 * args.n_tu * len(plan)} calls")
    if args.dry_run:
        print("--dry-run: stopping before NIM.")
        return 0

    run_id = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = os.path.join(ROOT, "results", "w140", "calibration", f"w140_tu_{run_id}")
    os.makedirs(out_dir, exist_ok=True)
    sc = open(os.path.join(out_dir, "calls.jsonl"), "w")

    results = {}
    eligible = {}  # (tier, family) -> bool ; a tier is tutor-eligible on a family
    for m, fam, knob in plan:
        tmpl = _factory_for(fam)(knob)
        tutor = T.COMPILERS_BY_KIND[args.tc](tmpl)
        gen = _build_nim_gen(model=m.model_id,
                             sidecar_writer=lambda r: (sc.write(json.dumps(r) + "\n"), sc.flush()))
        print(f"  [{m.tier}] tutor-usability on {fam}@{knob} ({m.model_id})...", flush=True)
        tu = B.measure_tutor_usability_v1(tmpl, gen=gen, model_id=m.model_id, tier=m.tier,
                                          tutor=tutor, n_tu=args.n_tu, max_tokens=args.max_tokens,
                                          timeout_s=EXEC_TIMEOUT_S)
        results.setdefault(m.tier, {})[fam] = tu.to_dict()
        eligible[f"{m.tier}|{fam}"] = tu.tutor_eligible
        print(f"     -> rate={tu.rate:.3f} eligible={tu.tutor_eligible} "
              f"(flipped {tu.n_flipped}/{tu.n_eligible})", flush=True)
    sc.close()

    tutor_usability_cid = _sha256_hex({"k": "w140_tutor_usability_v1", "tc": args.tc,
                                       "results": results})
    report = {"schema": "w140_tutor_usability_v1", "tc_kind": args.tc, "n_tu": args.n_tu,
              "reused_per_tier_calibration_cid": cal.get("per_tier_calibration_cid"),
              "anchor_tier": anchor_tier, "weak_tier": weak_tier, "shared_families": shared,
              "by_tier": results, "tutor_eligible": eligible,
              "tutor_usability_cid": tutor_usability_cid}
    with open(os.path.join(ROOT, args.out), "w") as f:
        json.dump(report, f, indent=2, default=str)
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)

    print("\n=== W140 TUTOR-USABILITY ===")
    for tier in results:
        for fam, d in results[tier].items():
            print(f"  {tier:6s} {fam:28s} rate={d['tutor_usability_rate']:.3f} "
                  f"eligible={d['tutor_eligible']}")
    print(f"tutor_usability_cid: {tutor_usability_cid}")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
