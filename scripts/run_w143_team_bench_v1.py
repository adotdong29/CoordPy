"""W143 — multi-agent DISCOVER-THEN-AMORTIZE team bench (Lane β).

Two staged modes (expensive-run discipline: cheap ``probe`` first, full ``bench`` only if the probe
shows the §14 DPI-band signal):

  --mode probe : the DISCOVERY load-bearing probe.  Runs each arm's DISCOVERY across ``--discover-seeds``
                 teacher seeds at the FRAGILE budget (max_disc_tries=1) and reports the per-arm disc-rate.
                 The §14 pre-condition: ST disc-rate < 1 (single-context best-of-K demonstrably fails to
                 discover) AND MA_FULL disc-rate > ST disc-rate (the team discovers where i.i.d. fails).
                 No amortize, so cheap.

  --mode bench : the full team bench.  Per arm: DISCOVER once (single fragile shot) + AMORTIZE M members.
                 Reports A0/A1/B0/ST/MA_*/NEG per member + the strict earn verdict.

Same-budget (RUNBOOK §7): every arm spends one-time discovery G_d = K_d + K_b + amortize M*K_a
(``team_budget_parity_v1``).  No-oracle: brutes + sketches are model-generated; the grader is SCORING-ONLY.
Non-negative: a failed discover (no clean extractable verified winner) => KEEP (== B0).

Usage:
  python scripts/run_w143_team_bench_v1.py --family subarrays_sum_and_range --mode probe \
      --K-discover 10 --K-brutes 5 --discover-seeds 4 --arms ST,MA_FULL
  python scripts/run_w143_team_bench_v1.py --family subarrays_sum_and_range --mode bench \
      --K-amortize 4 --K-discover 10 --K-brutes 5 --members 6 --teacher-seed 1 \
      --arms ST,MA_FULL,MA_RD,MA_Q,MA_SS,NEG_RAT,NEG_TECH
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
from coordpy.self_tutoring_controller_v1 import _efficient_prompt, _gen_text  # noqa: E402
from coordpy.no_oracle_verifier_v2 import select_winner_v2  # noqa: E402
from coordpy.self_tutoring_technique_extractor_v1 import compile_tutor_from_winner_v1  # noqa: E402
from coordpy import multi_agent_discover_amortize_v1 as MA  # noqa: E402

MODEL = "meta/llama-3.3-70b-instruct"
MINTED_DATE = "2026-06-08"
ALIEN_STRICT_FAMILY = "sum_nearest_smaller_left"   # structurally-distant alien (monotonic-stack, non-counting)


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


def _alien_strict_scaffold(c, timeout_s):
    """The structurally-distant alien (sum_nearest_smaller_left) scaffold for the NEG_TECH control —
    tests technique-CLASS specificity (a non-counting monotonic-stack scaffold must NOT lift the
    counting families)."""
    alien_fam = ALIEN_STRICT_FAMILY if c.family != ALIEN_STRICT_FAMILY else "count_pairs_sum_le_t"
    ac = _cand(alien_fam)
    at = ac.factory(ac.knob)
    ap = mint_problem_v1(at.minted, global_seed=1)
    tutor, _ = compile_tutor_from_winner_v1(at.minted.ref_source, at, ap, timeout_s=timeout_s)
    return tutor, alien_fam


def _gen_brutes(gen, mp, *, K_b, brute_diverse, max_tokens, brute_temp):
    return MA._gen_brutes(gen, mp, K_b=K_b, brute_diverse=brute_diverse,
                          max_tokens=max_tokens, temperature=brute_temp)


def _discover_arm(gen, arm, c, template, *, K_d, K_b, teacher_seed, max_tokens, temperature,
                  brute_temp, timeout_s, max_disc_tries):
    """Discover for one arm at the (fragile) budget.  NEG_TECH is handled specially (alien scaffold,
    no team discovery); all other arms route through team_discover_v1."""
    teacher = mint_problem_v1(template.minted, global_seed=teacher_seed)
    if arm == "NEG_TECH":
        alien, alien_fam = _alien_strict_scaffold(c, timeout_s)
        return MA.TeamDiscoverResultV1(
            arm="NEG_TECH", discovered=alien is not None, scaffold=alien, winner_code=None,
            diversity_classify="NA", diversity=None, n_analyze=0, n_candidates=0, n_brutes=0,
            n_model_calls=0, select_reason=f"alien={alien_fam}", winner_passes_secret=None,
            n_correct_brutes=None, n_disc_tries=0), teacher
    cfg = MA.arm_config(arm)
    rationale = None
    if cfg.rationale_alien:
        # alien rationale = STRATEGIST sketches from a DIFFERENT-family problem (the Q1 noise control)
        alt_fam = "count_pairs_sum_le_t" if c.family != "count_pairs_sum_le_t" else "subarrays_sum_and_range"
        ac = _cand(alt_fam)
        rationale = mint_problem_v1(ac.factory(ac.knob).minted, global_seed=teacher_seed + 7)
    res = MA.team_discover_v1(gen, teacher, template, config=cfg, K_d=K_d, K_b=K_b,
                              max_tokens=max_tokens, temperature=temperature, brute_temp=brute_temp,
                              timeout_s=timeout_s, minted_date=MINTED_DATE, max_disc_tries=max_disc_tries,
                              passes_secret_fn=_passes_secret, rationale_minted=rationale)
    return res, teacher


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--mode", choices=["probe", "bench"], default="bench")
    ap.add_argument("--arms", default="ST,MA_FULL,MA_RD,MA_Q,MA_SS,NEG_RAT,NEG_TECH")
    ap.add_argument("--K-amortize", type=int, default=4)
    ap.add_argument("--K-discover", type=int, default=10)   # fragile candidate-side budget
    ap.add_argument("--K-brutes", type=int, default=5)      # fragile brute-side budget (W142b fragile point)
    ap.add_argument("--max-disc-tries", type=int, default=1)  # 1 = single fragile shot (load-bearing test)
    ap.add_argument("--members", type=int, default=6)
    ap.add_argument("--discover-seeds", type=int, default=4)  # probe mode: # teacher seeds
    ap.add_argument("--bench-seeds", type=int, default=1)     # bench mode: # teacher seeds (each = 1 discovery trial + M members)
    ap.add_argument("--teacher-seed", type=int, default=1)
    ap.add_argument("--member-seed-base", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--brute-temp", type=float, default=0.4)
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--timeout", type=float, default=4.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    c = _cand(args.family)
    template = c.factory(c.knob)
    out = args.out or os.path.join(ROOT, "results", "w143", f"{args.mode}_{args.family}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    sc = open(out.replace(".json", "") + "_calls.jsonl", "a")  # append: survive interruption/resume (noqa: SIM115)

    def _w(rec):
        sc.write(json.dumps(rec) + "\n"); sc.flush()
    from scripts.run_w108_livecodebench_pilot import _build_nim_gen  # noqa: E402
    gen = _build_nim_gen(model=MODEL, sidecar_writer=_w, inter_call_sleep_s=0.0)
    K_a, K_d, K_b = args.K_amortize, args.K_discover, args.K_brutes
    to, mt = args.timeout, args.max_tokens

    # ============================== PROBE MODE (discovery load-bearing) =========================
    if args.mode == "probe":
        # RESUMABLE: each (seed,arm) row is appended to a progress jsonl immediately; on restart we
        # skip already-done pairs (NIM endpoint is historically unstable — don't lose paid discoveries).
        prog_path = out.replace(".json", "") + ".progress.jsonl"
        rows = []
        done = set()
        if os.path.exists(prog_path):
            with open(prog_path) as pf:
                for ln in pf:
                    try:
                        r = json.loads(ln)
                        rows.append(r); done.add((r["seed"], r["arm"]))
                    except Exception:  # noqa: BLE001
                        pass
            if done:
                print(f"[resume] loaded {len(done)} done (seed,arm) pairs from {prog_path}", flush=True)
        pf = open(prog_path, "a")  # noqa: SIM115
        for seed in range(args.teacher_seed, args.teacher_seed + args.discover_seeds):
            for arm in arms:
                if (seed, arm) in done:
                    print(f"  probe seed={seed} arm={arm} [resumed, skip]", flush=True)
                    continue
                res, _teacher = _discover_arm(gen, arm, c, template, K_d=K_d, K_b=K_b, teacher_seed=seed,
                                              max_tokens=mt, temperature=args.temperature,
                                              brute_temp=args.brute_temp, timeout_s=to,
                                              max_disc_tries=args.max_disc_tries)
                row = {"seed": seed, "arm": arm, "discovered": res.discovered,
                       "diversity": res.diversity_classify, "winner_secret": res.winner_passes_secret,
                       "reason": res.select_reason, "n_model_calls": res.n_model_calls}
                rows.append(row)
                pf.write(json.dumps(row) + "\n"); pf.flush()
                print(f"  probe seed={seed} arm={arm} disc={int(res.discovered)} "
                      f"div={res.diversity_classify} winner_secret={res.winner_passes_secret} "
                      f"reason={res.select_reason}", flush=True)
        pf.close()
        disc_rate = {}
        for arm in arms:
            ds = [r["discovered"] for r in rows if r["arm"] == arm]
            disc_rate[arm] = round(sum(1 for d in ds if d) / len(ds), 4) if ds else None
        st_rate = disc_rate.get("ST")
        ma_rate = disc_rate.get("MA_FULL")
        dpi_band_ok = (st_rate is not None and st_rate < 1.0)
        team_lifts = (st_rate is not None and ma_rate is not None and ma_rate > st_rate)
        obj = {"schema": "coordpy.w143_team_bench_v1.probe", "model": MODEL, "family": args.family,
               "vein": c.vein, "mode_label": c.mode, "K_discover": K_d, "K_brutes": K_b,
               "fragile_budget": {"K_d": K_d, "K_b": K_b, "max_disc_tries": args.max_disc_tries},
               "discover_seeds": args.discover_seeds, "disc_rate": disc_rate,
               "dpi_band_ok_st_disc_rate_lt_1": dpi_band_ok, "team_lifts_disc_rate": team_lifts,
               "rows": rows}
        sc.close()
        with open(out, "w") as f:
            json.dump(obj, f, indent=2, default=str)
        print(f"\n[{args.family}] PROBE disc_rate={disc_rate}  DPI-band(ST<1)={dpi_band_ok}  "
              f"team_lifts(MA>ST)={team_lifts} -> {out}", flush=True)
        return 0

    # ============================== BENCH MODE (discover + amortize, multi-seed) =================
    # Each bench-seed = ONE discovery trial (per arm) + M members.  Discovery is the per-seed variance,
    # so we aggregate over bench-seeds x members.  RESUMABLE: per-(bench_seed,member) rows appended.
    empty_tutor = MA.make_negative_control_tutor_v1(template)
    prog_path = out.replace(".json", "") + ".progress.jsonl"
    rows = []
    done = set()
    if os.path.exists(prog_path):
        for ln in open(prog_path):
            try:
                r = json.loads(ln); rows.append(r); done.add((r["bench_seed"], r["m"]))
            except Exception:  # noqa: BLE001
                pass
        if done:
            print(f"[resume] loaded {len(done)} done (bench_seed,member) rows", flush=True)
    pf = open(prog_path, "a")  # noqa: SIM115
    disc_log = {}
    from dataclasses import replace as _replace
    for bs in range(args.bench_seeds):
        tseed = args.teacher_seed + bs
        discovers = {}
        for arm in arms:
            res, _teacher = _discover_arm(gen, arm, c, template, K_d=K_d, K_b=K_b, teacher_seed=tseed,
                                          max_tokens=mt, temperature=args.temperature, brute_temp=args.brute_temp,
                                          timeout_s=to, max_disc_tries=args.max_disc_tries)
            discovers[arm] = res
            print(f"[{args.family}] seed={tseed} DISCOVER arm={arm} disc={int(res.discovered)} "
                  f"div={res.diversity_classify} winner_secret={res.winner_passes_secret} reason={res.select_reason}", flush=True)
        disc_log[tseed] = {a: discovers[a].discovered for a in arms}
        for s in range(args.members):
            if (tseed, s) in done:
                continue
            m = mint_problem_v1(template.minted, global_seed=args.member_seed_base + tseed * 1000 + s)
            mp = m.to_pilot_problem(minted_date=MINTED_DATE)
            brutes = _gen_brutes(gen, mp, K_b=K_b, brute_diverse=True, max_tokens=mt, brute_temp=args.brute_temp)
            plain = [extract_candidate_code_v1(response_text=_gen_text(gen, _efficient_prompt(mp), mt, args.temperature))
                     for _ in range(K_a)]
            a0 = _passes_secret(m, plain[0], to) if plain else False
            a1 = any(_passes_secret(m, p, to) for p in plain)
            selB = select_winner_v2(plain, statement=mp.statement, samples=list(m.samples), small_inputs=[],
                                    brute_codes=brutes, io_shape=template.io_shape, timeout_s=to)
            b0 = bool((not selB.abstained) and selB.winner_code and _passes_secret(m, selB.winner_code, to))
            row = {"bench_seed": tseed, "m": s, "a0": a0, "a1": a1, "b0": b0}
            for arm in arms:
                cfg = MA.arm_config(arm) if arm != "NEG_TECH" else MA.arm_config("MA_FULL")
                transfer = MA.TRANSFER_SHARED_STATE if arm == "NEG_TECH" else cfg.transfer
                cfg_eff = _replace(MA.arm_config("ST"), arm=arm, transfer=transfer) if arm == "NEG_TECH" else cfg
                row[arm] = MA.amortize_member_v1(gen, m, template, config=cfg_eff, discover=discovers[arm],
                                                 brutes=brutes, K_a=K_a, max_tokens=mt, temperature=args.temperature,
                                                 timeout_s=to, minted_date=MINTED_DATE, b0_pass=b0,
                                                 empty_tutor=empty_tutor, passes_secret_fn=_passes_secret)
            rows.append(row)
            pf.write(json.dumps(row) + "\n"); pf.flush()
            print(f"  [{args.family}] s{tseed}.m{s}: A0={int(a0)} A1={int(a1)} B0={int(b0)} "
                  + " ".join(f"{a}={int(row[a])}" for a in arms), flush=True)
    pf.close()

    N = len(rows)
    agg = {k: sum(int(x.get(k, 0)) for x in rows) for k in (["a0", "a1", "b0"] + arms)}
    pp = lambda k: 100.0 * agg[k] / N if N else 0.0  # noqa: E731
    budget = MA.team_budget_parity_v1(M=args.members, K_a=K_a, K_d=K_d, K_b=K_b)
    disc_rate = {a: round(sum(1 for ts in disc_log if disc_log[ts].get(a)) / max(1, len(disc_log)), 3) for a in arms}
    verdict = None
    if "MA_FULL" in arms and "ST" in arms:
        verdict = MA.apply_team_earn_gate_v1(
            ma_full_pp_over_a1=pp("MA_FULL") - pp("a1"), ma_full_pp_over_b0=pp("MA_FULL") - pp("b0"),
            ma_minus_st_pp=pp("MA_FULL") - pp("ST"), n_modes=1, neg_le_b0=(agg.get("NEG_RAT", 0) <= agg["b0"]),
            ma_gt_neg=(agg["MA_FULL"] > agg.get("NEG_RAT", 0)), diversity_real=True,
            st_disc_rate=disc_rate.get("ST", 0.0)).to_dict()
    obj = {"schema": "coordpy.w143_team_bench_v1.bench", "model": MODEL, "family": args.family,
           "vein": c.vein, "mode_label": c.mode, "K_amortize": K_a, "K_discover": K_d, "K_brutes": K_b,
           "members": args.members, "bench_seeds": args.bench_seeds, "n_cells": N, "arms": arms, "agg": agg,
           "disc_rate_over_seeds": disc_rate, "disc_log": disc_log,
           "ma_full_minus_a1_pp": round(pp("MA_FULL") - pp("a1"), 2) if "MA_FULL" in arms else None,
           "ma_full_minus_b0_pp": round(pp("MA_FULL") - pp("b0"), 2) if "MA_FULL" in arms else None,
           "ma_full_minus_st_pp": round(pp("MA_FULL") - pp("ST"), 2) if ("MA_FULL" in arms and "ST" in arms) else None,
           "earn_verdict_single_mode": verdict, "budget": budget.to_dict(), "per_cell": rows}
    sc.close()
    with open(out, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    summ = " ".join(f"{a}={agg[a]}/{N}" for a in (["a0", "a1", "b0"] + arms))
    print(f"\n[{args.family}] BENCH ({args.bench_seeds} seeds x {args.members} members = {N} cells): {summ}", flush=True)
    print(f"  disc_rate_over_seeds={disc_rate}", flush=True)
    if "MA_FULL" in arms:
        print(f"  MA_FULL-A1={obj['ma_full_minus_a1_pp']:+}pp  MA_FULL-B0={obj['ma_full_minus_b0_pp']:+}pp  "
              f"MA_FULL-ST={obj['ma_full_minus_st_pp']}pp -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
