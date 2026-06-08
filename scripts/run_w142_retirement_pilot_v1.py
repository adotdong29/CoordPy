"""W142 Lane β — high-K discover reliability + family-level retirement attempt.

Runs ONLY on the LOCKED admitted moderate-`p` family set from Lane α (passed via --families).  For each
family: DISCOVER a self-derived scaffold (raised discover-K), then AMORTIZE across M held-out members at
the STANDARD amortize-K, and compare arms at the pre-registered budget:

  A1  fair neutral self-consistency pool@K_a       (floor bar)
  B0  no-oracle verified-selection@K_a per member  (the STRONG bar — must re-discover per problem)
  ST4 self-tutoring, W141 shape (discover-K = K_a)
  STd self-tutoring, RAISED discover-K, amortize K_a held   (the W142 discover-reliability arm)
  NEG fake-scaffold control: a WRONG-family scaffold applied to this family must NOT lift (else the lift
      is scaffold-shape leakage, not technique transfer)

The retirement earn (RUNBOOK §7): ST beats B0 by a retirement-grade margin aggregated over the LOCKED
family set, spanning ≥3 families OR ≥2 modes, NEG fails, no-oracle audit holds, at the pre-registered
equal-total-family budget (``discover_amortize_accounting_v1``).  One/two families is NOT a retirement.

Usage:
  python scripts/run_w142_retirement_pilot_v1.py \
      --families count_pairs_sum_le_t,count_pairs_product_le_t,subarrays_sum_and_range \
      --K-amortize 4 --K-discover 12 --members 4
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from coordpy.moderate_p_family_slate_v1 import build_screen_slate_v1  # noqa: E402
from coordpy.resistant_by_construction_battlefield_v1 import mint_problem_v1  # noqa: E402
from coordpy.self_tutoring_controller_v1 import (  # noqa: E402
    discover_self_scaffold_v1, run_member_arms_v1, amortization_verdict_v1)
from coordpy.discover_amortize_accounting_v1 import (  # noqa: E402
    amortized_budget_parity_v1, per_member_superiority_pp_v1)

MODEL = "meta/llama-3.3-70b-instruct"
MINTED_DATE = "2026-06-08"
RETIRE_MARGIN_PP = 5.0          # ST - B0 aggregated retirement-grade margin


def _cand(fam: str):
    for c in build_screen_slate_v1():
        if c.family == fam:
            return c
    raise SystemExit(f"unknown family {fam}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", required=True, help="comma LOCKED admitted family ids")
    ap.add_argument("--K-amortize", type=int, default=4, help="standard per-member amortize budget")
    ap.add_argument("--K-discover", type=int, default=12, help="raised discover budget (one-time/family)")
    ap.add_argument("--members", type=int, default=4)
    ap.add_argument("--teacher-seed", type=int, default=1)
    ap.add_argument("--member-seed-base", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--timeout", type=float, default=4.0)
    ap.add_argument("--out", default="")
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()

    fams = [f.strip() for f in args.families.split(",") if f.strip()]
    out_dir = os.path.dirname(args.out) if args.out else os.path.join(ROOT, "results", "w142", "pilot")
    os.makedirs(out_dir, exist_ok=True)
    sidecar_path = (args.out.replace(".json", "") if args.out else os.path.join(out_dir, "pilot")) + "_calls.jsonl"
    sidecar = open(sidecar_path, "w")  # noqa: SIM115

    def _writer(rec: dict) -> None:
        sidecar.write(json.dumps(rec) + "\n"); sidecar.flush()

    if args.mock:
        gen = None
    else:
        from scripts.run_w108_livecodebench_pilot import _build_nim_gen  # noqa: E402
        gen = _build_nim_gen(model=MODEL, sidecar_writer=_writer, inter_call_sleep_s=0.0)

    K_a, K_d, M = args.K_amortize, args.K_discover, args.members
    cands = {f: _cand(f) for f in fams}

    # ---- DISCOVER one scaffold per family at the raised discover-K (STd); ST4 reuses K_a-discover -----
    scaffolds = {}
    discover_reports = {}
    for fam, c in cands.items():
        template = c.factory(c.knob)
        teacher = mint_problem_v1(c.factory(c.knob).minted, global_seed=args.teacher_seed)
        if gen is None:
            from scripts.run_w141_self_tutoring_dev_probe_v1 import _mock_gen
            g = _mock_gen(template)
        else:
            g = gen
        disc = discover_self_scaffold_v1(template, teacher, gen=g, K=K_d, temperature=args.temperature,
                                         max_tokens=args.max_tokens, timeout_s=args.timeout,
                                         minted_date=MINTED_DATE)
        scaffolds[fam] = disc.scaffold
        discover_reports[fam] = disc.to_dict()
        print(f"[discover] {fam}: discovered={disc.discovered} reason={disc.selection.reason} "
              f"winner_passes_secret={disc.winner_passes_secret}", flush=True)

    # a WRONG-family scaffold for the NEG control (the next family's scaffold, cyclically)
    fam_list = list(cands)
    wrong_scaffold = {fam_list[i]: scaffolds.get(fam_list[(i + 1) % len(fam_list)])
                      for i in range(len(fam_list))} if len(fam_list) > 1 else {}

    # ---- AMORTIZE over M members; arms B0/STd at K_a; NEG = wrong-family scaffold -----
    results = []
    for fam, c in cands.items():
        template = c.factory(c.knob)
        if gen is None:
            from scripts.run_w141_self_tutoring_dev_probe_v1 import _mock_gen
            g = _mock_gen(template)
        else:
            g = gen
        std_members, neg_members = [], []
        for s in range(M):
            m = mint_problem_v1(c.factory(c.knob).minted, global_seed=args.member_seed_base + s)
            std_members.append(run_member_arms_v1(template, m, scaffolds[fam], gen=g, K=K_a, K_re=K_a,
                                                  temperature=args.temperature, max_tokens=args.max_tokens,
                                                  timeout_s=args.timeout, minted_date=MINTED_DATE,
                                                  member_key=f"m{s}"))
            if wrong_scaffold.get(fam) is not None:
                m2 = mint_problem_v1(c.factory(c.knob).minted, global_seed=args.member_seed_base + s)
                neg_members.append(run_member_arms_v1(template, m2, wrong_scaffold[fam], gen=g, K=K_a,
                                                     K_re=K_a, temperature=args.temperature,
                                                     max_tokens=args.max_tokens, timeout_s=args.timeout,
                                                     minted_date=MINTED_DATE, member_key=f"neg{s}"))
        v = amortization_verdict_v1(fam, scaffolds[fam] is not None, std_members)
        neg_v = (amortization_verdict_v1(fam + "_NEG", wrong_scaffold.get(fam) is not None, neg_members)
                 if neg_members else None)
        budget = amortized_budget_parity_v1(M=M, K_a=K_a, K_d=K_d)
        theory_pp = per_member_superiority_pp_v1(v.raw_rate_p, K_a)
        print(f"[amortize] {fam}: A1={v.a1_solved} B0={v.b_solved} ST={v.st_solved}/{v.n_members} "
              f"ST-B0={v.st_minus_b:+d} p={v.raw_rate_p:.2f} q={v.scaffold_rate_q:.2f} "
              f"(theory (1-p)^K_a={theory_pp:.1f}pp) NEG_lift={(neg_v.st_minus_b if neg_v else 'NA')}",
              flush=True)
        results.append({"family": fam, "mode": c.mode, "vein": c.vein,
                        "discover": discover_reports[fam], "verdict": v.to_dict(),
                        "neg_verdict": (neg_v.to_dict() if neg_v else None),
                        "budget": budget.to_dict(), "theory_superiority_pp": theory_pp,
                        "members": [m.to_dict() for m in std_members]})

    # ---- earn rule -----
    n_fam = len(results)
    st_total = sum(r["verdict"]["st_solved"] for r in results)
    b0_total = sum(r["verdict"]["b_solved"] for r in results)
    n_members_total = sum(r["verdict"]["n_members"] for r in results)
    st_minus_b0_pp = (100.0 * (st_total - b0_total) / n_members_total) if n_members_total else 0.0
    fams_st_gt_b0 = [r["family"] for r in results if r["verdict"]["st_solved"] > r["verdict"]["b_solved"]]
    modes_won = sorted({r["mode"] for r in results if r["family"] in fams_st_gt_b0})
    neg_lifts = [r["neg_verdict"]["st_minus_b"] for r in results if r["neg_verdict"]]
    neg_fails = all(x <= 0 for x in neg_lifts) if neg_lifts else None
    span_ok = (len(fams_st_gt_b0) >= 3) or (len(modes_won) >= 2)
    earned = bool(st_minus_b0_pp >= RETIRE_MARGIN_PP and span_ok and (neg_fails in (True, None)))

    summary = {"schema": "coordpy.w142_retirement_pilot.v1", "model": MODEL, "families": fams,
               "K_amortize": K_a, "K_discover": K_d, "members": M,
               "st_total": st_total, "b0_total": b0_total, "n_members_total": n_members_total,
               "st_minus_b0_pp_aggregate": round(st_minus_b0_pp, 2),
               "families_st_gt_b0": fams_st_gt_b0, "modes_won": modes_won,
               "neg_fails_no_lift": neg_fails, "span_ok_>=3fam_OR_>=2mode": span_ok,
               "retirement_earned": earned, "results": results}
    sidecar.close()
    out_path = args.out or os.path.join(out_dir, "w142_pilot.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n=== W142 Lane β: ST-B0={st_minus_b0_pp:+.2f}pp over {n_fam} families; "
          f"families ST>B0={fams_st_gt_b0}; modes={modes_won}; NEG_no_lift={neg_fails}; "
          f"span_ok={span_ok}; RETIREMENT_EARNED={earned} ===", flush=True)
    print(f"-> {out_path}\n-> {sidecar_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
