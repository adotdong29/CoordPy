"""W142b Lane β — SINGLE-FAMILY discover→amortize pilot with the ROBUST v2 verifier (parallelizable).

Re-runs the W142 pilot using `no_oracle_verifier_v2.select_winner_v2` (cluster-with-a-brute on a
constraint-covering bank) + K_b independent self-brutes from a convention-explicit prompt — the fix for
the W142 FN (count_pairs: a buggy single brute vetoed the correct candidates) and FP (subarrays: the
sum-only naive was committed). Discover, B0 (no-oracle verified-selection), and ST all select via v2.
NEG = an alien-vein scaffold built $0. Still strictly no-oracle; non-negative (ABSTAIN ⇒ KEEP).

Usage (one per family, parallel):
  python scripts/run_w142b_pilot_family_v2.py --family count_pairs_sum_le_t \
      --K-amortize 4 --K-discover 24 --K-brutes 5 --members 4 --out results/w142b/pf_count_pairs_sum_le_t.json
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
from coordpy.icpc_reflexion_bench_v1 import extract_candidate_code_v1, grade_on_secret_v1  # noqa: E402
from coordpy.self_tutoring_controller_v1 import (  # noqa: E402
    _efficient_prompt, _scaffold_prompt, _build_small_bank_v1, _gen_text)
from coordpy.no_oracle_verifier_v2 import select_winner_v2  # noqa: E402
from coordpy.self_tutoring_technique_extractor_v1 import compile_tutor_from_winner_v1  # noqa: E402
from coordpy.discover_amortize_accounting_v1 import (  # noqa: E402
    amortized_budget_parity_v1, per_member_superiority_pp_v1)

MODEL = "meta/llama-3.3-70b-instruct"
MINTED_DATE = "2026-06-08"


def _brute_prompt_v2(pilot) -> str:
    return (f"{pilot.statement}\n\n"
            "Write a SIMPLE, OBVIOUSLY-CORRECT brute-force Python 3 solution. Prioritize CORRECTNESS "
            "over speed (use the most direct method, even O(N^2) or O(N^3)). Implement the EXACT "
            "definition in the statement: pay careful attention to WHICH items are counted (e.g. count "
            "each unordered pair {i,j} with i<j exactly once, not ordered pairs) and to EVERY stated "
            "condition (if several conditions must ALL hold for an item to count, check them all). "
            "Make sure your output matches the public example(s) EXACTLY. Read all input from stdin, "
            "write only the answer to stdout. Return ONLY one ```python code block.")


def _cand(fam):
    for c in build_screen_slate_v1():
        if c.family == fam:
            return c
    raise SystemExit(f"unknown family {fam}")


def _passes_secret(problem, code, timeout_s):
    if not code.strip():
        return False
    p = problem.to_pilot_problem(minted_date=MINTED_DATE)
    ok, _, _ = grade_on_secret_v1(p, code, timeout_s=timeout_s)
    return bool(ok)


def _alien_scaffold(cand):
    alien_fam = ("sum_nearest_smaller_left" if cand.vein == VEIN_SORT_TWO_POINTER
                 else "count_pairs_sum_le_t")
    ac = _cand(alien_fam); at = ac.factory(ac.knob)
    ap = mint_problem_v1(at.minted, global_seed=1)
    tutor, _ = compile_tutor_from_winner_v1(at.minted.ref_source, at, ap, timeout_s=6.0)
    return tutor, alien_fam


def _gen_brutes(gen, pilot, K_b, max_tokens, temperature, timeout_s):
    return [extract_candidate_code_v1(response_text=_gen_text(gen, _brute_prompt_v2(pilot), max_tokens, temperature))
            for _ in range(K_b)]


def _select(cands, pilot, problem, template, brutes, small, timeout_s):
    return select_winner_v2(cands, statement=pilot.statement, samples=list(problem.samples),
                            small_inputs=small, brute_codes=brutes, io_shape=template.io_shape,
                            timeout_s=timeout_s)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--K-amortize", type=int, default=4)
    ap.add_argument("--K-discover", type=int, default=24)
    ap.add_argument("--K-brutes", type=int, default=5)
    ap.add_argument("--max-disc-tries", type=int, default=4)
    ap.add_argument("--members", type=int, default=4)
    ap.add_argument("--teacher-seed", type=int, default=1)
    ap.add_argument("--member-seed-base", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--brute-temp", type=float, default=0.4)
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--timeout", type=float, default=4.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    c = _cand(args.family); template = c.factory(c.knob)
    out = args.out or os.path.join(ROOT, "results", "w142b", f"pf_{args.family}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    sc = open(out.replace(".json", "") + "_calls.jsonl", "w")  # noqa: SIM115

    def _w(rec):
        sc.write(json.dumps(rec) + "\n"); sc.flush()
    from scripts.run_w108_livecodebench_pilot import _build_nim_gen  # noqa: E402
    gen = _build_nim_gen(model=MODEL, sidecar_writer=_w, inter_call_sleep_s=0.0)
    K_a, K_d, K_b, M, to = args.K_amortize, args.K_discover, args.K_brutes, args.members, args.timeout

    # ---- DISCOVER (v2) ----
    teacher = mint_problem_v1(template.minted, global_seed=args.teacher_seed)
    tp = teacher.to_pilot_problem(minted_date=MINTED_DATE)
    brutes_t = _gen_brutes(gen, tp, K_b, args.max_tokens, args.brute_temp, to)
    cands_t = [extract_candidate_code_v1(response_text=_gen_text(gen, _efficient_prompt(tp), args.max_tokens, args.temperature))
               for _ in range(K_d)]
    small_t = []  # v2 clusters on public+aug only (ignores small_inputs) — skip the NIM bank-gen calls
    # W142b-3: DISCOVERY-RETRY loop — discovery is the one-time amortized cost, so make it DETERMINISTIC
    # not lucky: keep generating fresh brute+candidate batches until select_winner_v2 commits a winner
    # that EXTRACTS a clean scaffold (multi-winner: try EVERY verified-correct winner per batch). This
    # removes the per-seed discovery stochasticity (the s2f disc=False failure). All generation is the
    # teacher-side amortized cost; correctness is still no-oracle (brute-anchored cluster).
    scaffold = None; comp = None; win_secret = None; sel = None; n_disc_tries = 0
    while scaffold is None and n_disc_tries < args.max_disc_tries:
        n_disc_tries += 1
        if n_disc_tries > 1:
            brutes_t = _gen_brutes(gen, tp, K_b, args.max_tokens, args.brute_temp, to)
            cands_t = cands_t + [extract_candidate_code_v1(response_text=_gen_text(gen, _efficient_prompt(tp), args.max_tokens, args.temperature))
                                 for _ in range(K_d)]
        sel = _select(cands_t, tp, teacher, template, brutes_t, small_t, to)
        if sel.abstained:
            continue
        winners = [cands_t[v.idx] for v in sel.verdicts if v.is_winner]
        if sel.winner_code:
            winners = [sel.winner_code] + [w for w in winners if w != sel.winner_code]
        for w in winners:
            ws = _passes_secret(teacher, w, to)
            scf, cr = compile_tutor_from_winner_v1(w, template, teacher, timeout_s=to)
            if win_secret is None:
                win_secret, comp = ws, cr
            if scf is not None:
                scaffold, comp, win_secret = scf, cr, ws
                break
    n_correct_brute = sum(1 for b in brutes_t if _passes_secret_brute(b, teacher, to))
    print(f"[{args.family}] discover_tries={n_disc_tries} scaffold={'YES' if scaffold else 'NO'}", flush=True)
    print(f"[{args.family}] discover: reason={sel.reason} winner_idx={sel.winner_idx} "
          f"winner_secret={win_secret} compiled={(comp.compiled if comp else None)} "
          f"K_b={K_b} correct_brutes={n_correct_brute} alien=...", flush=True)
    alien, alien_fam = _alien_scaffold(c)

    # ---- AMORTIZE (v2) ----
    members = []
    for s in range(M):
        m = mint_problem_v1(template.minted, global_seed=args.member_seed_base + s)
        mp = m.to_pilot_problem(minted_date=MINTED_DATE)
        small = []  # v2 clusters on public+aug only — skip the NIM bank-gen calls
        brutes = _gen_brutes(gen, mp, K_b, args.max_tokens, args.brute_temp, to)
        plain = [extract_candidate_code_v1(response_text=_gen_text(gen, _efficient_prompt(mp), args.max_tokens, args.temperature))
                 for _ in range(K_a)]
        a1 = any(_passes_secret(m, p, to) for p in plain)
        selB = _select(plain, mp, m, template, brutes, small, to)
        b = bool((not selB.abstained) and selB.winner_code and _passes_secret(m, selB.winner_code, to))
        def st_arm(scaf):
            if scaf is None:
                return b  # KEEP == B0
            stc = [extract_candidate_code_v1(response_text=_gen_text(gen, _scaffold_prompt(mp, scaf), args.max_tokens, args.temperature))
                   for _ in range(K_a)]
            selS = _select(stc, mp, m, template, brutes, small, to)
            return bool((not selS.abstained) and selS.winner_code and _passes_secret(m, selS.winner_code, to))
        st = st_arm(scaffold)
        neg = st_arm(alien)
        members.append({"m": s, "a1": a1, "b0": b, "st": st, "neg": neg})
        print(f"  [{args.family}] m{s}: A1={int(a1)} B0={int(b)} ST={int(st)} NEG={int(neg)}", flush=True)

    A1 = sum(x["a1"] for x in members); B0 = sum(x["b0"] for x in members)
    ST = sum(x["st"] for x in members); NEG = sum(x["neg"] for x in members)
    budget = amortized_budget_parity_v1(M=M, K_a=K_a, K_d=K_d)
    obj = {"schema": "coordpy.w142b_pilot_family_v2.v1", "model": MODEL, "family": args.family,
           "vein": c.vein, "mode": c.mode, "K_amortize": K_a, "K_discover": K_d, "K_brutes": K_b,
           "members": M, "discovered": scaffold is not None, "winner_passes_secret": win_secret,
           "correct_brutes_at_discover": n_correct_brute, "discover_reason": sel.reason,
           "A1": A1, "B0": B0, "ST": ST, "NEG": NEG, "st_minus_b0": ST - B0, "neg_minus_b0": NEG - B0,
           "alien_family": alien_fam, "budget": budget.to_dict(),
           "per_member": members}
    sc.close()
    with open(out, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    print(f"[{args.family}] DONE: A1={A1} B0={B0} ST={ST} NEG={NEG} /{M}  ST-B0={ST-B0:+d}  "
          f"NEG-B0={NEG-B0:+d} discovered={scaffold is not None} -> {out}", flush=True)
    return 0


def _passes_secret_brute(code, problem, timeout_s):
    # scoring-only brute-correctness probe (a brute may TLE on secret-large => use a small grade)
    if not code.strip():
        return False
    p = problem.to_pilot_problem(minted_date=MINTED_DATE)
    # grade on the SMALL public samples only (brute is slow); correctness proxy
    for inp, exp in p.samples:
        from coordpy.no_oracle_verifier_v1 import _run_sig
        if _run_sig(code, inp, timeout_s=timeout_s) != exp.strip():
            return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())
