#!/usr/bin/env python3
"""W138 Lane α — $0 build + self-test (RUNBOOK_W138 §7).

Builds the headroom-band candidate grid and asserts, with NO NIM:
  1. HC1 parser-neutrality on every minted sample+secret case of every candidate cell.
  2. HC2 exact-oracle gates ADMIT every minted instance (ref-solvable, brute==ref, naive discriminating
     with the declared kind, split integrity).
  3. deterministic regeneration (same seed -> identical content CID; slate CID stable).
  4. regression fixtures BITE:
     - W136 I/O-confound: a flattened input FAILS parser_neutrality_gate_v1; normal-form PASSES.
     - W137 bimodality detector: band_verdict_v1 culls a saturated (a0=1) and a dead (a1=0) cell and
       admits a synthetic intermediate cell.
     - W133/W134 fake-different: fake_different_report_v1().bites (M3/B0 -> FAKE_DIFFERENT, witness -> REAL).

Run:  .venv/bin/python scripts/run_w138_build_and_selftest_v1.py [--n-per-cell 1] [--mint-timeout 1.0]
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
    build_band_candidates_v3, band_slate_fingerprint_cid_v1)
from coordpy.parser_neutral_io_v1 import (  # noqa: E402
    parse_all_tokens_v1, parser_neutrality_gate_v1, render_normal_form_v1)
from coordpy.resistant_by_construction_battlefield_v1 import mint_problem_v1  # noqa: E402
from coordpy.headroom_band_calibration_v2 import band_verdict_v1  # noqa: E402
from coordpy.model_ladder_calibration_v1 import (  # noqa: E402
    ModelTemplateStatV1, TemplateCalibrationV1)
from coordpy.band_mechanism_bench_v1 import fake_different_report_v1  # noqa: E402

OUT = "results/w138/w138_build_selftest_v1.json"


def _confound_fixture() -> dict:
    """W136 regression: a normal-form input passes HC1; a flattened variant FAILS it."""
    from coordpy.headroom_band_slate_v3 import _cx_count_pairs_sum_le_t
    t = _cx_count_pairs_sum_le_t(50)
    shape = t.io_shape
    data = {"N": 4, "T": 10, "A": [1, 2, 3, 4]}
    normal = render_normal_form_v1(shape, data)              # "4 10\n1 2 3 4\n"
    flat = " ".join(normal.split()) + "\n"                   # "4 10 1 2 3 4\n"  (W136 confound)
    nf_ok = parser_neutrality_gate_v1([normal], shape).is_parser_neutral
    flat_ok = parser_neutrality_gate_v1([flat], shape).is_parser_neutral
    # the all-tokens reader still parses the flattened body (that is exactly why it is a confound)
    all_tok_recovers = (parse_all_tokens_v1(flat, shape) == {"N": 4, "T": 10, "A": [1, 2, 3, 4]})
    bites = bool(nf_ok and not flat_ok and all_tok_recovers)
    return {"normal_form_passes_hc1": nf_ok, "flattened_fails_hc1": (not flat_ok),
            "all_tokens_recovers_flattened": all_tok_recovers, "bites": bites}


def _bimodality_detector_fixture() -> dict:
    """W137 regression: band_verdict_v1 culls saturated + dead, admits a synthetic intermediate."""
    def cal(name, s_a0, s_a1, sm_a0):
        strong = ModelTemplateStatV1(model_id="meta/llama-3.3-70b-instruct", tier="strong",
                                     a0_passed=tuple(s_a0), a1_passed=tuple(s_a1),
                                     n_calls=len(s_a0) + len(s_a1) * 5)
        small = ModelTemplateStatV1(model_id="meta/llama-3.1-8b-instruct", tier="small",
                                    a0_passed=tuple(sm_a0), a1_passed=(), n_calls=len(sm_a0))
        return TemplateCalibrationV1(template_name=name, family=name, mode="COMPLEXITY_BLIND",
                                     per_model=(strong, small), hc3_has_headroom=False,
                                     hc4_not_dead=False, discriminates=False, admitted=False,
                                     reason="")
    T, F = True, False
    sat = band_verdict_v1(cal("sat", [T] * 5, [T] * 8, [T] * 5), cell_id="sat@1", knob_value=1)
    dead = band_verdict_v1(cal("dead", [F] * 5, [F] * 8, [F] * 5), cell_id="dead@1", knob_value=1)
    inter = band_verdict_v1(cal("inter", [F] * 5, [T, T, T, T, F, F, F, F], [F] * 5),
                            cell_id="inter@1", knob_value=1)
    bites = bool((not sat.admitted) and sat.reason.startswith("HC3")
                 and (not dead.admitted) and "DEAD" in dead.reason
                 and inter.admitted)
    return {"saturated_culled": (not sat.admitted), "saturated_reason": sat.reason,
            "dead_culled": (not dead.admitted), "dead_reason": dead.reason,
            "intermediate_admitted": inter.admitted, "intermediate_a1_rate": inter.strong_a1_rate,
            "intermediate_wilson": [inter.wilson_lo, inter.wilson_hi], "bites": bites}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-cell", type=int, default=1, help="instances minted per cell for HC2 check")
    ap.add_argument("--mint-timeout", type=float, default=1.0, help="bounds the $0 build (naive TLE)")
    args = ap.parse_args()
    os.makedirs("results/w138", exist_ok=True)

    cands = build_band_candidates_v3()
    slate_cid = band_slate_fingerprint_cid_v1()
    print(f"slate_fingerprint_cid: {slate_cid}")
    print(f"candidate cells: {len(cands)} ({len({c.family for c in cands})} families, "
          f"{len({c.mode for c in cands})} modes)")

    # 1+2: HC1 + HC2 on every cell (mint n_per_cell instances)
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

    # 3: determinism (slate CID stable; same-seed content CID stable)
    slate_stable = (band_slate_fingerprint_cid_v1() == slate_cid)
    p1 = mint_problem_v1(cands[0].template.minted, global_seed=900000, timeout_s=args.mint_timeout)
    p2 = mint_problem_v1(cands[0].template.minted, global_seed=900000, timeout_s=args.mint_timeout)
    content_stable = (p1.content_cid() == p2.content_cid())

    # 4: regression fixtures
    confound = _confound_fixture()
    bimodal = _bimodality_detector_fixture()
    fd = fake_different_report_v1().to_dict()

    all_cells_ok = (n_admit == len(cands))
    fixtures_ok = bool(confound["bites"] and bimodal["bites"] and fd["bites"]
                       and slate_stable and content_stable)
    passed = bool(all_cells_ok and fixtures_ok)

    report = {"schema": "w138_build_selftest_v1", "slate_fingerprint_cid": slate_cid,
              "n_cells": len(cands), "n_cells_ok": n_admit, "all_cells_ok": all_cells_ok,
              "slate_cid_stable": slate_stable, "content_cid_stable": content_stable,
              "confound_fixture": confound, "bimodality_detector_fixture": bimodal,
              "fake_different": fd, "per_cell": per_cell, "passed": passed,
              "wall_s": round(time.time() - t0, 1)}
    with open(OUT, "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== W138 BUILD SELF-TEST ===")
    print(f"cells OK: {n_admit}/{len(cands)}")
    print(f"determinism: slate_cid_stable={slate_stable} content_cid_stable={content_stable}")
    print(f"W136 confound fixture bites: {confound['bites']}  "
          f"(normal passes={confound['normal_form_passes_hc1']}, flat fails={confound['flattened_fails_hc1']})")
    print(f"W137 bimodality detector bites: {bimodal['bites']}  "
          f"(sat culled={bimodal['saturated_culled']}, dead culled={bimodal['dead_culled']}, "
          f"inter admitted={bimodal['intermediate_admitted']})")
    print(f"fake-different bites: {fd['bites']}  (fake={fd['fake_arms']}, real={fd['real_arms']})")
    print(f"\nSELF-TEST {'PASS' if passed else 'FAIL'} -> {OUT}  ({report['wall_s']}s)")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
