"""W143 $0 gold-position analysis — the GENERATION-side load-bearing test across budgets, deterministic
and load-free (grade once, then pure combinatorics).

For each (seed,arm) block of the probe sidecar, grade every candidate on SECRET (gold = correct+efficient)
and record its POSITION in generation order.  Then, at each candidate budget K, compute the
generation-CEILING disc-rate = P(a gold appears in the first K candidates), with MA paying the analyze
tax (MA uses its first K-1 IMPLEMENT candidates; ST uses its first K i.i.d. candidates).  If role-diverse
MA gets gold into a SMALL budget more reliably than i.i.d. ST, the team lifts the generation ceiling at
the fragile budget (the W128 effect).  If ST >= MA at every K, the team adds no generation value here.

Usage: python scripts/run_w143_gold_position_analysis_v1.py --family subarrays_sum_and_range \
           --sidecar results/w143/probe_subarrays_calls.jsonl --K-discover 10 --K-brutes 5 \
           --discover-seeds 3 --teacher-seed 1 --arms ST,MA_FULL
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
    ap.add_argument("--K-discover", type=int, default=10)
    ap.add_argument("--K-brutes", type=int, default=5)
    ap.add_argument("--discover-seeds", type=int, default=3)
    ap.add_argument("--teacher-seed", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=4.0)
    ap.add_argument("--arms", default="ST,MA_FULL")
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    c = _cand(args.family)
    template = c.factory(c.knob)
    rows = [json.loads(ln) for ln in open(args.sidecar) if ln.strip()]
    block = args.K_brutes + args.K_discover
    idx = 0
    # first_gold_pos[arm] = list over seeds of the first gold position (0-indexed in code-candidate order),
    # or None if no gold in the block.
    first_gold = {a: [] for a in arms}
    for seed in range(args.teacher_seed, args.teacher_seed + args.discover_seeds):
        m = mint_problem_v1(template.minted, global_seed=seed)
        for arm in arms:
            blk = rows[idx:idx + block]; idx += block
            if len(blk) < block:
                continue
            cand_rows = blk[args.K_brutes:]
            codes = [extract_candidate_code_v1(response_text=r.get("response_text", "")) for r in cand_rows]
            codes = [x for x in codes if x.strip()]  # drop the MA ANALYZE (no code)
            fg = None
            for pos, code in enumerate(codes):
                if _passes_secret(m, code, args.timeout):
                    fg = pos
                    break
            first_gold[arm].append(fg)
            print(f"  [seed {seed} {arm}] n_code={len(codes)} first_gold_pos={fg}", flush=True)

    print("\n=== GENERATION-CEILING disc-rate at budget K (gold in first-K pool; MA pays analyze tax) ===", flush=True)
    for K in (4, 6, 8, 10):
        line = {}
        for a in arms:
            # MA's candidate budget at K_d=K is K-1 implements (1 call spent on ANALYZE); ST gets K i.i.d.
            eff = (K - 1) if a != "ST" else K
            hits = sum(1 for fg in first_gold[a] if fg is not None and fg < eff)
            n = len(first_gold[a])
            line[a] = f"{hits}/{n}" + (f"(<{eff})" if n else "")
        print(f"  K={K:2d}: " + "  ".join(f"{a}={line[a]}" for a in arms), flush=True)
    print(f"\n  first_gold_pos per arm: " + "  ".join(f"{a}={first_gold[a]}" for a in arms), flush=True)
    print("  INTERPRETATION: if ST hits >= MA hits at every K, role-diversity adds NO generation-ceiling "
          "value for this amortizable family (the efficient form is i.i.d.-reachable; the W128 ceiling-lift "
          "does not transfer to a p>=0.17 counting family).", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
