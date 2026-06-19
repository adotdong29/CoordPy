"""W143 $0 clean re-selection — re-run the no-oracle SELECT + EXTRACT on the ALREADY-PAID probe
generations, UNLOADED and one block at a time, to get a clean disc-rate free of the machine-load
artifact that corrupted the concurrent probe (the efficiency gate `_fast_efficient_v1` is wall-clock
timeout based and false-TLEs O(N) candidates under load).  No new NIM.

Reports per (seed,arm) block: committed?, winner_idx, extracted-scaffold? (= discovered), and the
gold pool (correct+efficient by the secret grader).  Isolates: does the single-controller verifier
reliably SELECT+EXTRACT the gold that exists, when run unloaded?

Usage: python scripts/run_w143_clean_reselect_v1.py --family subarrays_sum_and_range \
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
from coordpy.no_oracle_verifier_v2 import select_winner_v2  # noqa: E402
from coordpy.self_tutoring_technique_extractor_v1 import compile_tutor_from_winner_v1  # noqa: E402

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


def _reselect_and_extract(cands, brutes, mp, minted, template, timeout_s):
    """Mirror team_discover_v1's post-generation SELECT + multi-winner EXTRACT (no retry)."""
    sel = select_winner_v2(cands, statement=mp.statement, samples=list(minted.samples), small_inputs=[],
                           brute_codes=brutes, io_shape=template.io_shape, timeout_s=timeout_s)
    if sel.abstained:
        return False, sel.reason, None
    winners = [cands[v.idx] for v in sel.verdicts if v.is_winner and 0 <= v.idx < len(cands)]
    if sel.winner_code:
        winners = [sel.winner_code] + [w for w in winners if w != sel.winner_code]
    for w in winners:
        scf, _cr = compile_tutor_from_winner_v1(w, template, minted, timeout_s=timeout_s)
        if scf is not None:
            return True, sel.reason, sel.winner_idx
    return False, sel.reason + "+no_extract", sel.winner_idx


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
    disc = {a: [] for a in arms}
    for seed in range(args.teacher_seed, args.teacher_seed + args.discover_seeds):
        m = mint_problem_v1(template.minted, global_seed=seed)
        mp = m.to_pilot_problem(minted_date=MINTED_DATE)
        for arm in arms:
            blk = rows[idx:idx + block]; idx += block
            if len(blk) < block:
                continue
            brutes = [extract_candidate_code_v1(response_text=r.get("response_text", "")) for r in blk[:args.K_brutes]]
            cands = [extract_candidate_code_v1(response_text=r.get("response_text", "")) for r in blk[args.K_brutes:]]
            cands = [x for x in cands if x.strip()]
            gold = sum(1 for code in cands if _passes_secret(m, code, args.timeout))
            discovered, reason, widx = _reselect_and_extract(cands, brutes, mp, m, template, args.timeout)
            disc[arm].append(discovered)
            print(f"  [seed {seed} {arm}] gold={gold} discovered={int(discovered)} winner_idx={widx} reason={reason}", flush=True)

    print("\n=== CLEAN (unloaded, $0) disc-rate ===", flush=True)
    for a in arms:
        n = len(disc[a]); k = sum(1 for d in disc[a] if d)
        print(f"  {a}: {k}/{n} = {round(k / n, 3) if n else None}", flush=True)
    st = disc.get("ST", []); ma = disc.get("MA_FULL", [])
    if st and ma:
        st_r = sum(1 for d in st if d) / len(st); ma_r = sum(1 for d in ma if d) / len(ma)
        print(f"\n  DPI-band(ST<1)={st_r < 1.0}  team_lifts(MA>ST)={ma_r > st_r}  "
              f"(ST={st_r:.2f} MA={ma_r:.2f})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
