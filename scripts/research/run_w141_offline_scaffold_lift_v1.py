"""W141 offline scaffold-lift analysis ($0) — measure the SELF-DERIVED scaffold's generation lift.

The integer member-solve (pool@K) is coarse; this grades EVERY already-paid generation in a dev-probe
sidecar to recover the per-sample rates the solve hides: the raw efficient-pass rate p (plain
candidates) vs the scaffolded efficient-pass rate q (scaffolded candidates), on the held-out members.
q >> p is the direct, regime-independent evidence that the self-derived scaffold lifts re-generation
(the amortization mechanism's core claim).  No NIM; grades against the hidden bank for SCORING only.

Usage: python scripts/run_w141_offline_scaffold_lift_v1.py --sidecar results/w141/dev/<name>_calls.jsonl \
           --family sum_nearest_smaller_left --knob 20000 --member-seeds 100,101,102
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from coordpy.headroom_band_slate_v3 import CX_FACTORIES, FUNC_FACTORIES  # noqa: E402
from coordpy.resistant_by_construction_battlefield_v1 import mint_problem_v1  # noqa: E402
from coordpy.icpc_reflexion_bench_v1 import extract_candidate_code_v1, grade_on_secret_v1  # noqa: E402

MINTED_DATE = "2026-06-08"


def _classify(prompt: str) -> str:
    p = prompt.lower()
    if "obviously-correct" in p:
        return "brute"
    if "prints several small" in p:
        return "adversarial"
    if "technique tutor" in prompt or "fill in every blank" in p or "fill in the blanks" in p:
        return "scaffold"
    if "efficient python" in p:
        return "plain"
    return "other"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sidecar", required=True)
    ap.add_argument("--family", default="sum_nearest_smaller_left")
    ap.add_argument("--knob", type=int, default=20000)
    ap.add_argument("--member-seeds", default="100,101,102")
    ap.add_argument("--timeout", type=float, default=4.0)
    args = ap.parse_args()

    fac = CX_FACTORIES.get(args.family) or FUNC_FACTORIES.get(args.family)
    rows = [json.loads(l) for l in open(args.sidecar)]
    seeds = [int(s) for s in args.member_seeds.split(",")]

    tot_p_pass = tot_p_n = tot_q_pass = tot_q_n = 0
    per_member = []
    for s in seeds:
        prob = mint_problem_v1(fac(args.knob).minted, global_seed=s)
        pilot = prob.to_pilot_problem(minted_date=MINTED_DATE)
        stmt = pilot.statement
        pp = pn = qp = qn = 0
        for r in rows:
            pr = r.get("prompt", "")
            if not pr.startswith(stmt):       # match this member's problem exactly
                continue
            kind = _classify(pr)
            if kind not in ("plain", "scaffold"):
                continue
            code = extract_candidate_code_v1(response_text=r.get("response_text", ""))
            passed, _stderr, _n = grade_on_secret_v1(pilot, code, timeout_s=args.timeout)
            if kind == "plain":
                pn += 1
                pp += int(bool(passed))
            else:
                qn += 1
                qp += int(bool(passed))
        per_member.append({"seed": s, "plain": f"{pp}/{pn}", "scaffold": f"{qp}/{qn}"})
        tot_p_pass += pp
        tot_p_n += pn
        tot_q_pass += qp
        tot_q_n += qn
        print(f"  member seed {s}: plain (raw) {pp}/{pn}   scaffolded {qp}/{qn}", flush=True)

    p = tot_p_pass / tot_p_n if tot_p_n else 0.0
    q = tot_q_pass / tot_q_n if tot_q_n else 0.0
    out = {
        "schema": "coordpy.w141_offline_scaffold_lift.v1", "family": args.family, "knob": args.knob,
        "raw_rate_p": round(p, 4), "raw_pass": tot_p_pass, "raw_n": tot_p_n,
        "scaffold_rate_q": round(q, 4), "scaffold_pass": tot_q_pass, "scaffold_n": tot_q_n,
        "scaffold_lift_pp": round(100 * (q - p), 2), "per_member": per_member,
    }
    out_path = args.sidecar.replace("_calls.jsonl", "_scaffold_lift.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n=== SELF-DERIVED SCAFFOLD LIFT: raw p={p:.2f} ({tot_p_pass}/{tot_p_n}) -> "
          f"scaffolded q={q:.2f} ({tot_q_pass}/{tot_q_n}) = {100*(q-p):+.1f}pp ===")
    print(f"-> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
