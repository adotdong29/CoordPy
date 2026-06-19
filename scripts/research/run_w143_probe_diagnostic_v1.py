"""W143 $0 probe diagnostic — grade the already-paid probe generations (oracle, SCORING-ONLY) to
separate a GENERATION cap from a SELECTION cap (the W128 question at the discovery step).

For each (seed, arm) block of the probe sidecar, grade every candidate on PUBLIC (looks-right) and on
SECRET (correct AND efficient — the large secret input TLEs a slow solution). Report:
  - pool_gold       = #candidates passing SECRET (correct+efficient) — the GENERATION CEILING
  - pool_looksright = #candidates passing PUBLIC but FAILING SECRET (correct-but-slow / W125)
  - n_correct_brute = #brutes passing PUBLIC (the quorum anchor supply)
If pool_gold > 0 while the probe abstained => SELECTION cap (verifier missed the gold).
If pool_gold == 0 => GENERATION cap (no efficient form generated; the team cannot select what is absent).

Usage: python scripts/run_w143_probe_diagnostic_v1.py --family subarrays_sum_and_range \
           --sidecar results/w143/probe_subarrays_calls.jsonl --K-discover 10 --K-brutes 5 \
           --discover-seeds 3 --teacher-seed 1
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
from coordpy.no_oracle_verifier_v1 import _run_sig  # noqa: E402

MINTED_DATE = "2026-06-08"


def _cand(fam):
    for c in build_screen_slate_v1():
        if c.family == fam:
            return c
    raise SystemExit(f"unknown family {fam}")


def _passes_public(pilot, code, timeout_s):
    if not code or not code.strip():
        return False
    for inp, exp in pilot.samples:
        try:
            if _run_sig(code, inp, timeout_s=timeout_s) != exp.strip():
                return False
        except Exception:  # noqa: BLE001
            return False
    return True


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
    print(f"sidecar: {len(rows)} calls; family={args.family} knob={c.knob} arms={arms}", flush=True)

    K_b, K_d = args.K_brutes, args.K_discover
    block = K_b + K_d  # calls per (seed,arm) block
    idx = 0
    summary = []
    for seed in range(args.teacher_seed, args.teacher_seed + args.discover_seeds):
        m = mint_problem_v1(template.minted, global_seed=seed)
        mp = m.to_pilot_problem(minted_date=MINTED_DATE)
        for arm in arms:
            blk = rows[idx:idx + block]
            idx += block
            if len(blk) < block:
                print(f"  [seed {seed} {arm}] INCOMPLETE block ({len(blk)}/{block}), skipping", flush=True)
                continue
            brute_rows = blk[:K_b]
            cand_rows = blk[K_b:]
            n_correct_brute = 0
            for r in brute_rows:
                code = extract_candidate_code_v1(response_text=r.get("response_text", ""))
                if _passes_public(mp, code, args.timeout):
                    n_correct_brute += 1
            gold = looksright = wrong = 0
            for r in cand_rows:
                code = extract_candidate_code_v1(response_text=r.get("response_text", ""))
                if not code.strip():
                    continue  # e.g. the MA ANALYZE call (sketches, not code)
                pub = _passes_public(mp, code, args.timeout)
                sec = _passes_secret(m, code, args.timeout) if pub else False
                if sec:
                    gold += 1
                elif pub:
                    looksright += 1
                else:
                    wrong += 1
            rec = {"seed": seed, "arm": arm, "n_correct_brute": n_correct_brute,
                   "pool_gold": gold, "pool_looksright": looksright, "pool_wrong": wrong,
                   "n_candidates_with_code": gold + looksright + wrong}
            summary.append(rec)
            print(f"  [seed {seed} {arm}] correct_brute={n_correct_brute}/{K_b}  "
                  f"GOLD(correct+efficient)={gold}  looksright_fails_hidden={looksright}  "
                  f"wrong={wrong}", flush=True)

    tot_gold = sum(r["pool_gold"] for r in summary)
    tot_lr = sum(r["pool_looksright"] for r in summary)
    tot_brute = sum(r["n_correct_brute"] for r in summary)
    print("\n=== DIAGNOSIS ===", flush=True)
    print(f"total GOLD (correct+efficient candidates across all blocks) = {tot_gold}", flush=True)
    print(f"total looks-right-fails-hidden (correct-but-slow / W125)     = {tot_lr}", flush=True)
    print(f"total correct brutes (quorum anchors)                        = {tot_brute}", flush=True)
    if tot_gold == 0:
        print("VERDICT: GENERATION CAP — no efficient form generated at this budget; the team cannot "
              "select what is absent. (Diversity lifts the ceiling only if the efficient form is "
              "REACHABLE; here the 70B does not write the efficient subarrays solution at temp=0.7.)", flush=True)
    else:
        print(f"VERDICT: gold EXISTS ({tot_gold}) but the probe abstained => SELECTION/CLUSTERING cap "
              "(the verifier-quorum failed to anchor/select the gold). A stronger quorum could recover it.", flush=True)
    print(json.dumps({"family": args.family, "summary": summary,
                      "total_gold": tot_gold, "total_looksright": tot_lr, "total_correct_brute": tot_brute}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
