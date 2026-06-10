"""W142b strict-NEG investigation: is the subarrays HIDDEN_EDGE lift technique-real or generic-structural?

The committed NEG arm uses count_pairs as the subarrays alien — but BOTH are counting-by-accept-predicate
families, so the count_pairs scaffold's STRUCTURE can transfer (seed2 NEG=8). This re-runs the subarrays
NEG arm with a STRUCTURALLY-DISTANT alien (sum_nearest_smaller_left — monotonic-stack, assigns a value,
NO count accumulator). If this strict alien does NOT lift (NEG_strict ≈ B0), the count_pairs→subarrays
transfer was counting-class-specific and the subarrays win over a true alien is technique-real.
"""
from __future__ import annotations
import argparse, json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from coordpy.moderate_p_family_slate_v1 import build_screen_slate_v1
from coordpy.resistant_by_construction_battlefield_v1 import mint_problem_v1
from coordpy.icpc_reflexion_bench_v1 import extract_candidate_code_v1, grade_on_secret_v1
from coordpy.self_tutoring_controller_v1 import _scaffold_prompt, _gen_text
from coordpy.self_tutoring_technique_extractor_v1 import compile_tutor_from_winner_v1
MODEL = "meta/llama-3.3-70b-instruct"; MINTED = "2026-06-08"


def _cand(fam):
    return next(c for c in build_screen_slate_v1() if c.family == fam)


def _secret(prob, code, to):
    if not code.strip():
        return False
    ok, _, _ = grade_on_secret_v1(prob.to_pilot_problem(minted_date=MINTED), code, timeout_s=to)
    return bool(ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K-amortize", type=int, default=4)
    ap.add_argument("--members", type=int, default=10)
    ap.add_argument("--member-seed-base", type=int, default=200)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    to = 4.0
    from scripts.run_w108_livecodebench_pilot import _build_nim_gen
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    sc = open(a.out.replace(".json", "") + "_calls.jsonl", "w")  # noqa: SIM115
    gen = _build_nim_gen(model=MODEL, sidecar_writer=lambda r: (sc.write(json.dumps(r) + "\n"), sc.flush()),
                         inter_call_sleep_s=0.0)
    sub = _cand("subarrays_sum_and_range"); subt = sub.factory(sub.knob)
    # STRICT alien = sum_nearest (monotonic-stack, non-counting, structurally distant)
    alien_c = _cand("sum_nearest_smaller_left"); at = alien_c.factory(alien_c.knob)
    ap_prob = mint_problem_v1(at.minted, global_seed=1)
    strict_alien, rep = compile_tutor_from_winner_v1(at.minted.ref_source, at, ap_prob, timeout_s=6.0)
    print(f"strict alien (sum_nearest) compiled={rep.compiled}", flush=True)
    neg_strict = 0
    for s in range(a.members):
        m = mint_problem_v1(subt.minted, global_seed=a.member_seed_base + s)
        mp = m.to_pilot_problem(minted_date=MINTED)
        solved = any(_secret(m, extract_candidate_code_v1(response_text=_gen_text(gen, _scaffold_prompt(mp, strict_alien), 1536, 0.7)), to)
                     for _ in range(a.K_amortize))
        neg_strict += int(solved)
        print(f"  m{s}: NEG_strict={int(solved)}", flush=True)
    sc.close()
    json.dump({"schema": "w142b_strict_neg", "family": "subarrays_sum_and_range",
               "strict_alien": "sum_nearest_smaller_left", "members": a.members,
               "member_seed_base": a.member_seed_base, "NEG_strict": neg_strict},
              open(a.out, "w"), indent=2)
    print(f"DONE: NEG_strict={neg_strict}/{a.members} (compare ST=9, B0=4, NEG_countpairs=8) -> {a.out}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
