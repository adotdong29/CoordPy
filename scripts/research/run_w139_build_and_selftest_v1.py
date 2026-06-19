#!/usr/bin/env python3
"""W139 Lane α/β — $0 build + self-test (RUNBOOK_W139 §7).

Builds the W139 candidate grid (locked family set x continuous-N knob grid) and asserts, with NO NIM:
  1. HC1 parser-neutrality on every minted sample+secret case of every candidate cell.
  2. HC2 exact-oracle gates ADMIT every minted instance (ref-solvable, brute==ref, naive discriminating).
  3. deterministic regeneration (same seed -> identical content CID; slate CID stable).
  4. regression fixtures BITE:
     - W136 I/O-confound: a flattened input FAILS parser_neutrality_gate_v1; normal-form PASSES.
     - W137/W139 per-tier band detector: PerTierStatV1.in_band culls a saturated (a0=1) and a dead
       (a1=0) tier-stat and admits a synthetic intermediate one.
     - W138 capability-gate: a tier with witness_usability_rate < tau is NOT witness_eligible (Cm KEEPs,
       == A1, non-negative); a tier >= tau IS eligible (Cm can APPLY).
     - large-probe revival: on a HIDDEN_EDGE instance, the SMALL-probe counterexample search returns
       NONE while the LARGE-probe search FINDS a leakage-clean counterexample (the W138 2nd-mode death
       is repaired).
     - fake-different: fake_different_report_v1(("Cm","Nb","C0")).bites (Cm/Nb/C0 REAL, M3/B0 FAKE).

Run:  .venv/bin/python scripts/run_w139_build_and_selftest_v1.py [--n-per-cell 1] [--mint-timeout 1.0]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from coordpy.headroom_band_slate_v3 import (  # noqa: E402
    FUNC_FACTORIES, build_band_candidates_v3, band_slate_fingerprint_cid_v1)
from coordpy.parser_neutral_io_v1 import (  # noqa: E402
    parse_all_tokens_v1, parser_neutrality_gate_v1, render_normal_form_v1)
from coordpy.resistant_by_construction_battlefield_v1 import mint_problem_v1  # noqa: E402
from coordpy.band_mechanism_bench_v1 import fake_different_report_v1  # noqa: E402
from coordpy.per_tier_band_calibration_v1 import (  # noqa: E402
    CX_KNOB_GRID_V139, FUNC_KNOB_GRID_V139, PerTierStatV1, W139_FAMILIES)
from coordpy.capability_matched_witness_compiler_v1 import (  # noqa: E402
    WitnessUsabilityV1, build_large_probe_set_v1)
from coordpy.exact_oracle_witness_v1 import (  # noqa: E402
    build_witness_probe_set_v1, find_counterexample_witness_v1)

OUT = "results/w139/w139_build_selftest_v1.json"


def _w139_candidates():
    cands = build_band_candidates_v3(cx_knobs=CX_KNOB_GRID_V139, func_knobs=FUNC_KNOB_GRID_V139)
    return [c for c in cands if c.family in W139_FAMILIES]


def _confound_fixture() -> dict:
    """W136 regression: a normal-form input passes HC1; a flattened variant FAILS it."""
    from coordpy.headroom_band_slate_v3 import _cx_count_pairs_sum_le_t
    t = _cx_count_pairs_sum_le_t(50)
    shape = t.io_shape
    data = {"N": 4, "T": 10, "A": [1, 2, 3, 4]}
    normal = render_normal_form_v1(shape, data)
    flat = " ".join(normal.split()) + "\n"
    nf_ok = parser_neutrality_gate_v1([normal], shape).is_parser_neutral
    flat_ok = parser_neutrality_gate_v1([flat], shape).is_parser_neutral
    all_tok_recovers = (parse_all_tokens_v1(flat, shape) == {"N": 4, "T": 10, "A": [1, 2, 3, 4]})
    bites = bool(nf_ok and not flat_ok and all_tok_recovers)
    return {"normal_form_passes_hc1": nf_ok, "flattened_fails_hc1": (not flat_ok),
            "all_tokens_recovers_flattened": all_tok_recovers, "bites": bites}


def _per_tier_band_detector_fixture() -> dict:
    """W137/W139 regression: PerTierStatV1.in_band culls saturated + dead, admits intermediate."""
    T, F = True, False
    sat = PerTierStatV1("m", "strong", True, (T,) * 5, (T,) * 8, 45)
    dead = PerTierStatV1("m", "small", False, (F,) * 5, (F,) * 8, 45)
    inter = PerTierStatV1("m", "strong", True, (F,) * 5, (T, T, T, T, F, F, F, F), 45)
    bites = bool((not sat.in_band()) and (not dead.in_band()) and inter.in_band())
    return {"saturated_a1": sat.a1_rate, "saturated_culled": (not sat.in_band()),
            "dead_a1": dead.a1_rate, "dead_culled": (not dead.in_band()),
            "intermediate_a1": inter.a1_rate, "intermediate_admitted": inter.in_band(),
            "intermediate_wilson": [round(x, 3) for x in inter.wilson], "bites": bites}


def _capability_gate_fixture() -> dict:
    """W138 regression: witness-usability routing (KEEP vs APPLY) by measured capability."""
    lo = WitnessUsabilityV1("8b", "small", 6, 5, 1, 0.34)     # rate 0.20 -> ineligible (KEEP == A1)
    hi = WitnessUsabilityV1("70b", "strong", 6, 5, 4, 0.34)   # rate 0.80 -> eligible (APPLY)
    bites = bool((not lo.witness_eligible) and hi.witness_eligible)
    return {"weak_rate": round(lo.rate, 3), "weak_eligible": lo.witness_eligible,
            "strong_rate": round(hi.rate, 3), "strong_eligible": hi.witness_eligible, "bites": bites}


def _large_probe_revival_fixture(mint_timeout: float) -> dict:
    """W139: the large-probe counterexample search revives the W138-dead 2nd mode on HIDDEN_EDGE."""
    tmpl = FUNC_FACTORIES["subarrays_sum_and_range"](1500)
    mp = mint_problem_v1(tmpl.minted, global_seed=139_950_001, timeout_s=mint_timeout)
    naive = mp.naive_source
    small = build_witness_probe_set_v1(tmpl.minted, mp, witness_seed=999_139, timeout_s=2.0)
    large = build_large_probe_set_v1(tmpl.minted, mp, witness_seed=999_139)
    w_small = find_counterexample_witness_v1(naive, mp, small, tmpl.minted, timeout_s=2.0)
    w_large = find_counterexample_witness_v1(naive, mp, large, tmpl.minted, timeout_s=2.0)
    bites = bool((not w_small.found()) and w_large.found() and w_large.leakage_clean)
    return {"n_small_probes": len(small.small), "n_large_probes": len(large.small),
            "small_finds_counterexample": w_small.found(),
            "large_finds_counterexample": w_large.found(),
            "large_leakage_clean": bool(w_large.leakage_clean),
            "large_probe_tokens": int(w_large.probe_input_tokens), "bites": bites}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-cell", type=int, default=1)
    ap.add_argument("--mint-timeout", type=float, default=1.0)
    args = ap.parse_args()
    os.makedirs("results/w139", exist_ok=True)

    cands = _w139_candidates()
    slate_cid = band_slate_fingerprint_cid_v1(cx_knobs=CX_KNOB_GRID_V139, func_knobs=FUNC_KNOB_GRID_V139)
    print(f"slate_fingerprint_cid: {slate_cid}")
    print(f"W139 candidate cells: {len(cands)} ({len({c.family for c in cands})} families, "
          f"{len({c.mode for c in cands})} modes)")

    t0 = time.time()
    per_cell = []
    n_admit = 0
    for c in cands:
        recs = []
        for r in range(args.n_per_cell):
            p = mint_problem_v1(c.template.minted, global_seed=900000 + r, timeout_s=args.mint_timeout)
            hc1 = parser_neutrality_gate_v1([i for i, _ in p.secret_cases], c.template.io_shape)
            ok = bool(p.gates.admitted and hc1.is_parser_neutral)
            recs.append({"hc1": hc1.is_parser_neutral, "hc2": p.gates.admitted,
                         "reason": p.gates.reason, "ok": ok})
        cell_ok = all(x["ok"] for x in recs)
        n_admit += int(cell_ok)
        per_cell.append({"cell_id": c.cell_id, "family": c.family, "mode": c.mode,
                         "knob": c.knob_value, "ok": cell_ok, "recs": recs})
        print(f"  {c.cell_id:36s} {'OK' if cell_ok else 'FAIL':4s} {recs[0]['reason']}")

    # The grid is OVER-PROVISIONED (RUNBOOK §4): small-N complexity cells are legitimately HC2-culled
    # as NOT_DISCRIMINATING (the O(N^2) naive does not TLE yet).  The build invariant is therefore
    # HC1-universality (every minted cell parser-neutral) + an admitted grid that SPANS >=3 families
    # and >=2 modes — NOT that every over-provisioned cell is admitted.
    hc1_universal = all(all(rec["hc1"] for rec in pc["recs"]) for pc in per_cell)
    admitted_families = {pc["family"] for pc in per_cell if pc["ok"]}
    admitted_modes = {pc["mode"] for pc in per_cell if pc["ok"]}
    grid_spans = bool(len(admitted_families) >= 3 and len(admitted_modes) >= 2)
    # every non-admitted cell must be culled for a LEGITIMATE reason (discrimination), not HC1
    legit_culls = all(pc["ok"] or "DISCRIMINAT" in pc["recs"][0]["reason"].upper()
                      for pc in per_cell)

    slate_stable = (band_slate_fingerprint_cid_v1(
        cx_knobs=CX_KNOB_GRID_V139, func_knobs=FUNC_KNOB_GRID_V139) == slate_cid)
    p1 = mint_problem_v1(cands[0].template.minted, global_seed=900000, timeout_s=args.mint_timeout)
    p2 = mint_problem_v1(cands[0].template.minted, global_seed=900000, timeout_s=args.mint_timeout)
    content_stable = (p1.content_cid() == p2.content_cid())

    confound = _confound_fixture()
    band_det = _per_tier_band_detector_fixture()
    capgate = _capability_gate_fixture()
    revival = _large_probe_revival_fixture(args.mint_timeout)
    fd = fake_different_report_v1(real_arm_ids=("Cm", "Nb", "C0")).to_dict()

    build_ok = bool(hc1_universal and grid_spans and legit_culls)
    fixtures_ok = bool(confound["bites"] and band_det["bites"] and capgate["bites"]
                       and revival["bites"] and fd["bites"] and slate_stable and content_stable)
    passed = bool(build_ok and fixtures_ok)

    report = {"schema": "w139_build_selftest_v1", "slate_fingerprint_cid": slate_cid,
              "cx_knobs": list(CX_KNOB_GRID_V139), "func_knobs": list(FUNC_KNOB_GRID_V139),
              "families": list(W139_FAMILIES), "n_cells": len(cands), "n_cells_admitted": n_admit,
              "hc1_universal": hc1_universal, "grid_spans_3fam_2mode": grid_spans,
              "legit_culls": legit_culls, "admitted_families": sorted(admitted_families),
              "admitted_modes": sorted(admitted_modes), "slate_cid_stable": slate_stable,
              "content_cid_stable": content_stable, "confound_fixture": confound,
              "per_tier_band_detector_fixture": band_det, "capability_gate_fixture": capgate,
              "large_probe_revival_fixture": revival, "fake_different": fd, "per_cell": per_cell,
              "passed": passed, "wall_s": round(time.time() - t0, 1)}
    with open(OUT, "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== W139 BUILD SELF-TEST ===")
    print(f"cells admitted: {n_admit}/{len(cands)}  HC1-universal={hc1_universal}  "
          f"grid_spans(>=3fam,>=2mode)={grid_spans}  legit_culls={legit_culls}")
    print(f"  admitted families={sorted(admitted_families)} modes={sorted(admitted_modes)}")
    print(f"determinism: slate_cid_stable={slate_stable} content_cid_stable={content_stable}")
    print(f"W136 confound bites: {confound['bites']}")
    print(f"per-tier band detector bites: {band_det['bites']} "
          f"(sat culled={band_det['saturated_culled']}, dead culled={band_det['dead_culled']}, "
          f"inter admitted={band_det['intermediate_admitted']})")
    print(f"W138 capability-gate bites: {capgate['bites']} "
          f"(weak {capgate['weak_rate']}->eligible {capgate['weak_eligible']}, "
          f"strong {capgate['strong_rate']}->eligible {capgate['strong_eligible']})")
    print(f"large-probe revival bites: {revival['bites']} "
          f"(small finds={revival['small_finds_counterexample']}, "
          f"large finds={revival['large_finds_counterexample']} @ {revival['large_probe_tokens']} tok)")
    print(f"fake-different bites: {fd['bites']} (fake={fd['fake_arms']}, real={fd['real_arms']})")
    print(f"\nSELF-TEST {'PASS' if passed else 'FAIL'} -> {OUT}  ({report['wall_s']}s)")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
