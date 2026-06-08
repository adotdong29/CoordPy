"""W142 Lane β — SINGLE-FAMILY discover→amortize pilot (parallelizable).

Runs the full Lane β arm slate for ONE admitted moderate-`p` family so the pilot can be launched in
PARALLEL across the locked admitted set (each family is independent). Per family:

  DISCOVER a self-derived scaffold at the RAISED discover-K (STd); also build a NEG alien scaffold $0
  from a DIFFERENT-vein family's reference (a valid, leak-clean, WRONG-technique scaffold —
  ``compile_tutor_from_winner_v1`` is deterministic, no NIM). Then AMORTIZE over M held-out members:

    A1   self-consistency pool@K_a        B0  no-oracle verified-selection@K_a (the STRONG bar)
    ST   self-tutoring (real scaffold)     NEG self-tutoring with the ALIEN scaffold (must NOT lift)

The retirement earn is aggregated ACROSS families by ``run_w142_retirement_aggregate_v1`` (ST−B0 ≥ +5pp,
span ≥3 families OR ≥2 modes, NEG no-lift, equal-total-family budget via ``discover_amortize_accounting_v1``).
``grade_on_secret_v1`` is SCORING-ONLY. Explicit-import only; ``coordpy/__init__.py`` untouched.

Usage (one per family, in parallel):
  python scripts/run_w142_pilot_family_v1.py --family count_pairs_sum_le_t \
      --K-amortize 4 --K-discover 12 --members 4 --out results/w142/pilot/pf_count_pairs_sum_le_t.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from coordpy.moderate_p_family_slate_v1 import build_screen_slate_v1, VEIN_SORT_TWO_POINTER  # noqa: E402
from coordpy.resistant_by_construction_battlefield_v1 import mint_problem_v1  # noqa: E402
from coordpy.self_tutoring_controller_v1 import (  # noqa: E402
    discover_self_scaffold_v1, run_member_arms_v1, amortization_verdict_v1)
from coordpy.self_tutoring_technique_extractor_v1 import compile_tutor_from_winner_v1  # noqa: E402
from coordpy.discover_amortize_accounting_v1 import (  # noqa: E402
    amortized_budget_parity_v1, per_member_superiority_pp_v1)

MODEL = "meta/llama-3.3-70b-instruct"
MINTED_DATE = "2026-06-08"


def _cand(fam: str):
    for c in build_screen_slate_v1():
        if c.family == fam:
            return c
    raise SystemExit(f"unknown family {fam}")


def _alien_scaffold(cand):
    """A valid, leak-clean, WRONG-technique scaffold built $0 from a DIFFERENT-vein family's reference
    (the fake-scaffold NEG control; ``compile_tutor_from_winner_v1`` is deterministic — no NIM)."""
    # pick an alien of a DIFFERENT vein; monotonic-stack vs sort+two-pointer are maximally distinct
    alien_fam = ("sum_nearest_smaller_left" if cand.vein == VEIN_SORT_TWO_POINTER
                 else "count_pairs_sum_le_t")
    ac = _cand(alien_fam)
    at = ac.factory(ac.knob)
    ap = mint_problem_v1(at.minted, global_seed=1)
    tutor, _rep = compile_tutor_from_winner_v1(at.minted.ref_source, at, ap, timeout_s=6.0)
    return tutor, alien_fam


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--K-amortize", type=int, default=4)
    ap.add_argument("--K-discover", type=int, default=12)
    ap.add_argument("--members", type=int, default=4)
    ap.add_argument("--teacher-seed", type=int, default=1)
    ap.add_argument("--member-seed-base", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--timeout", type=float, default=4.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    c = _cand(args.family)
    template = c.factory(c.knob)
    out = args.out or os.path.join(ROOT, "results", "w142", "pilot", f"pf_{args.family}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    sidecar = open(out.replace(".json", "") + "_calls.jsonl", "w")  # noqa: SIM115

    def _w(rec):
        sidecar.write(json.dumps(rec) + "\n"); sidecar.flush()
    from scripts.run_w108_livecodebench_pilot import _build_nim_gen  # noqa: E402
    gen = _build_nim_gen(model=MODEL, sidecar_writer=_w, inter_call_sleep_s=0.0)

    K_a, K_d, M = args.K_amortize, args.K_discover, args.members
    teacher = mint_problem_v1(template.minted, global_seed=args.teacher_seed)
    disc = discover_self_scaffold_v1(template, teacher, gen=gen, K=K_d, temperature=args.temperature,
                                     max_tokens=args.max_tokens, timeout_s=args.timeout,
                                     minted_date=MINTED_DATE)
    alien, alien_fam = _alien_scaffold(c)
    print(f"[{args.family}] discovered={disc.discovered} reason={disc.selection.reason} "
          f"winner_passes_secret={disc.winner_passes_secret} alien={alien_fam}", flush=True)

    std, neg = [], []
    for s in range(M):
        m = mint_problem_v1(template.minted, global_seed=args.member_seed_base + s)
        std.append(run_member_arms_v1(template, m, disc.scaffold, gen=gen, K=K_a, K_re=K_a,
                                      temperature=args.temperature, max_tokens=args.max_tokens,
                                      timeout_s=args.timeout, minted_date=MINTED_DATE, member_key=f"m{s}"))
        m2 = mint_problem_v1(template.minted, global_seed=args.member_seed_base + s)
        neg.append(run_member_arms_v1(template, m2, alien, gen=gen, K=K_a, K_re=K_a,
                                      temperature=args.temperature, max_tokens=args.max_tokens,
                                      timeout_s=args.timeout, minted_date=MINTED_DATE, member_key=f"neg{s}"))
    v = amortization_verdict_v1(args.family, disc.discovered, std)
    nv = amortization_verdict_v1(args.family + "_NEG", alien is not None, neg)
    budget = amortized_budget_parity_v1(M=M, K_a=K_a, K_d=K_d)
    out_obj = {"schema": "coordpy.w142_pilot_family.v1", "model": MODEL, "family": args.family,
               "vein": c.vein, "mode": c.mode, "K_amortize": K_a, "K_discover": K_d, "members": M,
               "alien_family": alien_fam, "discover": disc.to_dict(), "verdict": v.to_dict(),
               "neg_verdict": nv.to_dict(), "budget": budget.to_dict(),
               "theory_superiority_pp": per_member_superiority_pp_v1(v.raw_rate_p, K_a),
               "members": [m.to_dict() for m in std], "neg_members": [m.to_dict() for m in neg]}
    sidecar.close()
    with open(out, "w") as f:
        json.dump(out_obj, f, indent=2, default=str)
    print(f"[{args.family}] A1={v.a1_solved} B0={v.b_solved} ST={v.st_solved}/{v.n_members} "
          f"ST-B0={v.st_minus_b:+d} p={v.raw_rate_p:.2f} q={v.scaffold_rate_q:.2f} "
          f"NEG(ST-B0)={nv.st_minus_b:+d} -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
