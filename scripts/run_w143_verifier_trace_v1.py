"""W143 $0 verifier trace — WHY does select_winner_v2 abstain when GOLD (correct+efficient) candidates
exist?  Loads one (seed,arm) block from the probe sidecar, grades each candidate on SECRET (gold flag),
runs select_winner_v2, and dumps the per-candidate verdict (passes_public / agrees_with_brute / efficient
/ is_winner) so we can see exactly which gate rejects the gold:
  - gold with agrees_with_brute=False  => CLUSTERING failure (gold disagrees with the correct brute on
    the constraint-covering bank => clusters separately => not in the brute-anchored ref cluster)
  - gold with efficient=False          => EFFICIENCY false-negative (the efficient form flagged slow)
  - gold with is_winner=False but both True => a ranking/ref-cluster bug

Usage: python scripts/run_w143_verifier_trace_v1.py --family subarrays_sum_and_range \
           --sidecar results/w143/probe_subarrays_calls.jsonl --seed 1 --block-index 0 \
           --K-discover 10 --K-brutes 5
(block-index: 0=seed1-ST, 1=seed1-MA, 2=seed2-ST, ... in the ST,MA arm order.)
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
from coordpy.icpc_reflexion_bench_v1 import extract_candidate_code_v1, grade_on_secret_v1  # noqa: E402
from coordpy.no_oracle_verifier_v2 import select_winner_v2, constraint_covering_bank_v1  # noqa: E402

MINTED_DATE = "2026-06-08"


def _cand(fam):
    for c in build_screen_slate_v1():
        if c.family == fam:
            return c
    raise SystemExit(f"unknown family {fam}")


def _passes_secret(minted, code, timeout_s):
    if not code or not code.strip():
        return False
    p = minted.to_pilot_problem(minted_date=MINTED_DATE)
    ok, _, _ = grade_on_secret_v1(p, code, timeout_s=timeout_s)
    return bool(ok)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--sidecar", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--block-index", type=int, required=True)
    ap.add_argument("--K-discover", type=int, default=10)
    ap.add_argument("--K-brutes", type=int, default=5)
    ap.add_argument("--timeout", type=float, default=4.0)
    args = ap.parse_args()

    c = _cand(args.family)
    template = c.factory(c.knob)
    m = mint_problem_v1(template.minted, global_seed=args.seed)
    mp = m.to_pilot_problem(minted_date=MINTED_DATE)
    rows = [json.loads(ln) for ln in open(args.sidecar) if ln.strip()]
    block = args.K_brutes + args.K_discover
    blk = rows[args.block_index * block:(args.block_index + 1) * block]
    brutes = [extract_candidate_code_v1(response_text=r.get("response_text", "")) for r in blk[:args.K_brutes]]
    cands = [extract_candidate_code_v1(response_text=r.get("response_text", "")) for r in blk[args.K_brutes:]]
    cands = [x for x in cands if x.strip()]  # drop the MA ANALYZE (no code)

    gold_idx = {i for i, code in enumerate(cands) if _passes_secret(m, code, args.timeout)}
    print(f"family={args.family} seed={args.seed} block={args.block_index} knob={c.knob}", flush=True)
    print(f"  candidates_with_code={len(cands)}  GOLD idx={sorted(gold_idx)}", flush=True)

    bank = constraint_covering_bank_v1(template.io_shape, list(m.samples))
    print(f"  constraint_covering_bank size={len(bank)}", flush=True)

    sel = select_winner_v2(cands, statement=mp.statement, samples=list(m.samples), small_inputs=[],
                           brute_codes=brutes, io_shape=template.io_shape, timeout_s=args.timeout)
    print(f"  select_winner_v2: abstained={sel.abstained} reason={sel.reason} winner_idx={sel.winner_idx}", flush=True)
    print("  per-candidate verdicts (G=gold):", flush=True)
    for v in sel.verdicts:
        g = "G" if v.idx in gold_idx else " "
        print(f"    [{g}] idx={v.idx} pub={int(v.passes_public)} agrees_brute={int(v.agrees_with_brute)} "
              f"n_brute_cases={v.n_brute_cases} eff={int(v.efficient)} witness={v.witness_kind} "
              f"winner={int(v.is_winner)} sig={str(v.output_sig)[:24]}", flush=True)

    # the decisive lines: the GOLD candidates' gates
    print("\n  === GOLD GATE ANALYSIS ===", flush=True)
    vmap = {v.idx: v for v in sel.verdicts}
    for gi in sorted(gold_idx):
        v = vmap.get(gi)
        if v is None:
            print(f"    gold idx={gi}: NOT in verdicts (filtered before clustering — e.g. failed public?)", flush=True)
            continue
        diag = []
        if not v.passes_public:
            diag.append("FAILS_PUBLIC(gold fails its own public samples?!)")
        if not v.agrees_with_brute:
            diag.append("DISAGREES_WITH_BRUTE(clusters separately from the correct brute)")
        if not v.efficient:
            diag.append("FLAGGED_INEFFICIENT(efficiency false-negative)")
        if v.passes_public and v.agrees_with_brute and v.efficient and not v.is_winner:
            diag.append("ALL_GATES_PASS_BUT_NOT_WINNER(ref-cluster/ranking bug)")
        print(f"    gold idx={gi}: {', '.join(diag) or 'is_winner='+str(v.is_winner)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
