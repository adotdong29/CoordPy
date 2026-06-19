#!/usr/bin/env python3
"""W140 Lane β — cross-tier TUTOR mechanism bench (RUNBOOK_W140 §7a/§7b/§8).

For each ladder tier, mints THAT TIER'S OWN per-tier band slice over the SHARED families (reused from
the W139 per-tier calibration) and runs the same-budget arms: baseline A0/A1/B0, C0 (the W139 raw
witness the 8B could not act on), and the tutor arms T1/T2/T3 + the LEAD T4 (capability-matched tutor
controller, routed by the W140 tutor-usability prior).  Computes the W140 weak-tier-lift earn rule.

Same-budget: every arm K=5, attempt-0 standard, one model call/attempt, no early stop, graded on
secret; the tutor/witness payload is $0 (compiled outside K).

Run:  .venv/bin/python scripts/run_w140_cross_tier_tutor_bench_v1.py --mode dev
        [--tutor-usability results/w140/w140_tutor_usability_v1.json] [--n-per-cell 3]
        [--arms A0,A1,B0,C0,T1,T2,T3,T4] [--tiers small,mid,strong] [--dry-run]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from coordpy.headroom_band_slate_v3 import (  # noqa: E402
    CX_FACTORIES, EXTRA_CX_FACTORIES, FUNC_FACTORIES, band_slate_fingerprint_cid_v1)
from coordpy.parser_neutral_io_v1 import parser_neutrality_gate_v1  # noqa: E402
from coordpy.resistant_by_construction_battlefield_v1 import mint_problem_v1  # noqa: E402
from coordpy.headroom_band_corpus_v3 import CorpusProblemV3, DEFAULT_MINTED_DATE  # noqa: E402
from coordpy.icpc_reflexion_bench_v1 import (  # noqa: E402
    IcpcBenchConfigV1, run_icpc_reflexion_bench_v1)
from coordpy.exact_oracle_witness_v1 import (  # noqa: E402
    ARM_C3_CONTROLLER, build_witness_probe_set_v1, run_witness_arm_v1)
from coordpy.per_tier_band_calibration_v1 import LADDER_V2  # noqa: E402
from coordpy import family_tutor_compiler_v1 as T  # noqa: E402
from coordpy import cross_tier_tutor_bench_v1 as B  # noqa: E402
from coordpy.family_tutor_compiler_v1 import _sha256_hex  # noqa: E402
from scripts.run_w108_livecodebench_pilot import _build_nim_gen  # noqa: E402

EXEC_TIMEOUT_S = 8.0
WITNESS_TIMEOUT_S = 2.0
WITNESS_SEED = 999_140
NONNEG_TOL_PP = 1.0
W140_SPLIT_SEED_BASE = {"dev": 140_200_000, "eval": 140_300_000, "frontier": 140_400_000}
TC_FOR_ARM = {"T1": T.TC1_CARD, "T2": T.TC2_REWRITE, "T3": T.TC3_COMPRESSED, "T4": T.TC2_REWRITE,
              "T5": T.TC5_STAGED}


def _factory_for(fam):
    return CX_FACTORIES.get(fam) or FUNC_FACTORIES.get(fam) or EXTRA_CX_FACTORIES.get(fam)


def _shared_families(cal, anchor_tier, weak_tier):
    bft = cal.get("band_for_tier", {})
    a, w = bft.get(anchor_tier, {}), bft.get(weak_tier, {})
    return sorted(set(a) & set(w))


def _family_seed_offset(fam):
    """Deterministic per-family seed offset (NOT Python's randomized hash) so eval/frontier slices
    are reproducible + CID-lockable."""
    return int(_sha256_hex({"w140_fam_seed": fam})[:8], 16) % 1000


def _mint_family_slice(mode, fam, knob, n_replicas, mint_timeout):
    base = W140_SPLIT_SEED_BASE[mode]
    tmpl = _factory_for(fam)(int(knob))
    out = []
    for r in range(int(n_replicas)):
        s = base + _family_seed_offset(fam) * 1000 + r
        p = mint_problem_v1(tmpl.minted, global_seed=s, timeout_s=float(mint_timeout))
        hc1 = parser_neutrality_gate_v1([i for i, _ in p.secret_cases], tmpl.io_shape)
        out.append((tmpl, CorpusProblemV3(split=mode, seed=s, cell_id=tmpl.minted.name,
                    family=tmpl.minted.family, mode=tmpl.minted.mode, minted=p,
                    hc1_parser_neutral=bool(hc1.is_parser_neutral),
                    gate_admitted=bool(p.gates.admitted))))
    return [(t, c) for (t, c) in out if c.admitted]


def _acc(arr):
    return sum(1 for x in arr if x) / len(arr) if arr else 0.0


def run_tier(*, tier, model_id, families_knobs, eligible_by_fam, mode, n_per_cell, arms,
             max_tokens, mint_timeout, out_dir):
    """Run baseline + the configured arms on this tier's per-family band slices."""
    sc = open(os.path.join(out_dir, f"calls_{tier}.jsonl"), "w")
    gen = _build_nim_gen(model=model_id,
                         sidecar_writer=lambda r: (sc.write(json.dumps(r) + "\n"), sc.flush()))
    per_family = {}
    pooled = {a: [] for a in ["A1", "B0"] + [x for x in arms if x not in ("A0", "A1", "B0")]}
    pooled_a1, pooled_modes, pooled_fams = [], [], []
    pooled_lead, pooled_b0, pooled_struct = [], [], []
    for fam, knob in sorted(families_knobs.items()):
        slice_ = _mint_family_slice(mode, fam, knob, n_per_cell, mint_timeout)
        if not slice_:
            continue
        tmpl0 = slice_[0][0]
        pilots = [c.to_pilot(minted_date=DEFAULT_MINTED_DATE) for _, c in slice_]
        cfg = IcpcBenchConfigV1(K_multi_sample=5, seeds=(W140_SPLIT_SEED_BASE[mode],),
                                sampling_temperature=0.7, max_tokens_per_call=int(max_tokens),
                                executor_timeout_s=EXEC_TIMEOUT_S)
        print(f"  [{tier}] {fam}@{knob}: baseline A0/A1/B0 on {len(pilots)} ({model_id})...", flush=True)
        base = run_icpc_reflexion_bench_v1(gen=gen, model_id=model_id, subset=pilots, config=cfg)
        bsr = base.per_seed[0]
        per_a1 = [bool(x) for x in bsr.per_problem_a1_passed]
        per_b0 = [bool(x) for x in bsr.per_problem_b_passed]
        fam_rec = {"family": fam, "knob": int(knob), "mode": tmpl0.minted.mode,
                   "n": len(pilots), "a0": base.a0_mean_pass_at_1, "a1": base.a1_mean_pass_at_1,
                   "b0": base.b_mean_pass_at_1, "per_a1": per_a1, "per_b0": per_b0, "arms": {}}
        tutor_elig = bool(eligible_by_fam.get(fam, False))
        tutors = B.compile_family_tutors_v1(tmpl0)
        for arm in arms:
            if arm in ("A0", "A1", "B0"):
                continue
            passed = []
            for (tmpl, c) in slice_:
                if arm == "C0":  # the W139 raw witness reflexion (always applies the witness)
                    probe = build_witness_probe_set_v1(tmpl.minted, c.minted,
                                                       witness_seed=WITNESS_SEED, timeout_s=EXEC_TIMEOUT_S)
                    o, _ = run_witness_arm_v1(seed=W140_SPLIT_SEED_BASE[mode], template=tmpl.minted,
                            problem=c.minted, probe=probe, gen=gen, K=5, temperature=0.7,
                            max_tokens=int(max_tokens), timeout_s=EXEC_TIMEOUT_S, arm=ARM_C3_CONTROLLER,
                            minted_date=DEFAULT_MINTED_DATE, witness_timeout_s=WITNESS_TIMEOUT_S)
                elif arm == "T4":
                    o, _ = B.run_tutor_controller_arm_v1(seed=W140_SPLIT_SEED_BASE[mode],
                            template=tmpl, problem=c.minted, tutor=tutors[T.TC2_REWRITE], gen=gen, K=5,
                            temperature=0.7, max_tokens=int(max_tokens), timeout_s=EXEC_TIMEOUT_S,
                            minted_date=DEFAULT_MINTED_DATE, tutor_eligible=tutor_elig)
                else:  # T1/T2/T3/T5 tutor reflexion arms
                    o, _ = B.run_tutor_arm_v1(seed=W140_SPLIT_SEED_BASE[mode], template=tmpl,
                            problem=c.minted, tutor=tutors[TC_FOR_ARM[arm]], gen=gen, K=5,
                            temperature=0.7, max_tokens=int(max_tokens), timeout_s=EXEC_TIMEOUT_S,
                            minted_date=DEFAULT_MINTED_DATE, arm_id=arm)
                passed.append(bool(o.final_passed))
            fam_rec["arms"][arm] = {"passed": passed, "acc": _acc(passed),
                                    "minus_a1_pp": round((_acc(passed) - _acc(per_a1)) * 100, 2),
                                    "minus_b0_pp": round((_acc(passed) - _acc(per_b0)) * 100, 2),
                                    "tutor_eligible": tutor_elig if arm in ("T4",) else None}
            print(f"      {arm}: acc={_acc(passed)*100:.1f} -A1={fam_rec['arms'][arm]['minus_a1_pp']:+.1f} "
                  f"-B0={fam_rec['arms'][arm]['minus_b0_pp']:+.1f}", flush=True)
            pooled[arm].extend(passed)
        per_family[fam] = fam_rec
        # pool for the anchor gate (lead = T4)
        if "T4" in fam_rec["arms"]:
            pooled_lead.extend(fam_rec["arms"]["T4"]["passed"])
            pooled_a1.extend(per_a1)
            pooled_b0.extend(per_b0)
            pooled_modes.extend([tmpl0.minted.mode] * len(per_a1))
            pooled_fams.extend([fam] * len(per_a1))
            pooled_struct.extend([True] * len(per_a1))
    sc.close()
    return {"tier": tier, "model": model_id, "per_family": per_family,
            "pooled": {"lead": pooled_lead, "a1": pooled_a1, "b0": pooled_b0,
                       "modes": pooled_modes, "families": pooled_fams, "structural": pooled_struct}}


def w140_verdict(tier_results, anchor_tier, weak_tier, margin_pp):
    """The W140 weak-tier-lift earn rule (RUNBOOK_W140 §7a/§7b)."""
    def t4_minus(tier, which):
        tr = tier_results.get(tier, {})
        pf = tr.get("per_family", {})
        gains = {f: pf[f]["arms"].get("T4", {}).get(f"minus_{which}_pp", 0.0) for f in pf
                 if "T4" in pf[f]["arms"]}
        return gains
    anchor_a1 = t4_minus(anchor_tier, "a1")
    anchor_b0 = t4_minus(anchor_tier, "b0")
    weak_a1 = t4_minus(weak_tier, "a1")
    # pooled per-tier T4-A1
    def pooled_delta(tier):
        tr = tier_results.get(tier, {})
        p = tr.get("pooled", {})
        if not p.get("a1"):
            return None, None
        la = (_acc(p["lead"]) - _acc(p["a1"])) * 100
        lb = (_acc(p["lead"]) - _acc(p["b0"])) * 100
        return round(la, 2), round(lb, 2)
    per_tier_delta = {t: pooled_delta(t) for t in tier_results}
    anchor_da1, anchor_db0 = per_tier_delta.get(anchor_tier, (None, None))
    weak_da1, weak_db0 = per_tier_delta.get(weak_tier, (None, None))
    # weak-tier span: families where weak T4-A1 >= margin (the gain is realized, not concentrated)
    weak_lift_families = [f for f, g in weak_a1.items() if g >= margin_pp]
    # non-negativity vs A1 AND vs B0 on every tested tier
    all_nonneg = True
    for t, (da1, db0) in per_tier_delta.items():
        if da1 is None:
            continue
        if da1 < -NONNEG_TOL_PP or db0 < -NONNEG_TOL_PP:
            all_nonneg = False
    anchor_pass = (anchor_da1 is not None and anchor_da1 >= margin_pp and anchor_db0 >= -NONNEG_TOL_PP)
    weak_pass = (weak_da1 is not None and weak_da1 >= margin_pp)
    span_ok = len(weak_lift_families) >= 2
    earned = bool(anchor_pass and weak_pass and all_nonneg and span_ok)
    return {"anchor_tier": anchor_tier, "weak_tier": weak_tier, "margin_pp": margin_pp,
            "per_tier_T4_minus_A1_B0_pp": per_tier_delta,
            "anchor_T4_minus_A1_pp": anchor_da1, "anchor_T4_minus_B0_pp": anchor_db0,
            "weak_T4_minus_A1_pp": weak_da1, "weak_T4_minus_B0_pp": weak_db0,
            "weak_per_family_T4_minus_A1_pp": weak_a1, "weak_lift_families": weak_lift_families,
            "anchor_pass": anchor_pass, "weak_pass": weak_pass, "all_tiers_nonneg": all_nonneg,
            "weak_span_ok_ge2_families": span_ok, "earned": earned}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dev", "eval", "frontier"], default="dev")
    ap.add_argument("--calibration", default="results/w139/w139_per_tier_calibration_v1.json")
    ap.add_argument("--tutor-usability", default="results/w140/w140_tutor_usability_v1.json")
    ap.add_argument("--n-per-cell", type=int, default=3)
    ap.add_argument("--arms", default="A0,A1,B0,C0,T1,T2,T3,T4")
    ap.add_argument("--tiers", default="small,mid,strong")
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--mint-timeout", type=float, default=1.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    with open(os.path.join(ROOT, args.calibration)) as f:
        cal = json.load(f)
    assert cal.get("slate_fingerprint_cid") == band_slate_fingerprint_cid_v1(
        cx_knobs=tuple(cal.get("cx_knobs", (20000, 50000))),
        func_knobs=tuple(cal.get("func_knobs", (1500, 4000)))), "SLATE DRIFT — refusing NIM"
    anchor_tier = cal.get("anchor_tier", "strong")
    weak_tier = "small"
    band_for_tier = cal.get("band_for_tier", {})
    shared = _shared_families(cal, anchor_tier, weak_tier)
    tu = {}
    tu_path = os.path.join(ROOT, args.tutor_usability)
    if os.path.exists(tu_path):
        with open(tu_path) as f:
            tu = json.load(f)
    elig = tu.get("tutor_eligible", {})  # "tier|family" -> bool
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    want = [t.strip() for t in args.tiers.split(",") if t.strip()]
    ladder = [m for m in LADDER_V2 if m.tier in want]

    print(f"mode={args.mode} reused_calibration_cid={cal.get('per_tier_calibration_cid')}")
    print(f"shared families: {shared}  arms: {arms}")
    print(f"tutor_usability_cid: {tu.get('tutor_usability_cid', '(none yet)')}")
    plan = []
    for m in ladder:
        cells = band_for_tier.get(m.tier, {})
        fk = {f: int(cells[f]["knob_value"]) for f in shared if f in cells}
        eb = {f: bool(elig.get(f"{m.tier}|{f}", False)) for f in fk}
        plan.append((m, fk, eb))
        print(f"  {m.tier:6s}: " + ", ".join(f"{f}@{k}(elig={eb[f]})" for f, k in fk.items()))
    fd = B.fake_different_report_v1(real_arm_ids=("T1", "T2", "T3", "T4")).to_dict()
    print(f"fake-different (T1-T4 REAL, T6/B0 FAKE): bites={fd['bites']}")
    if args.dry_run:
        print("--dry-run: stopping before NIM.")
        return 0

    run_id = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = os.path.join(ROOT, "results", "w140", args.mode, f"w140_{args.mode}_{run_id}")
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    tier_results = {}
    for m, fk, eb in plan:
        if not fk:
            continue
        tier_results[m.tier] = run_tier(tier=m.tier, model_id=m.model_id, families_knobs=fk,
                                         eligible_by_fam=eb, mode=args.mode, n_per_cell=args.n_per_cell,
                                         arms=arms, max_tokens=args.max_tokens,
                                         mint_timeout=args.mint_timeout, out_dir=out_dir)
    margin = 3.33 if args.mode == "dev" else 5.00
    verdict = w140_verdict(tier_results, anchor_tier, weak_tier, margin)
    # the standard anchor gate object (reuse the validated machinery, lead = T4)
    gate = None
    anc = tier_results.get(anchor_tier, {}).get("pooled", {})
    if anc.get("a1"):
        gate = B.evaluate_gate_v1(name=("dev_gate" if args.mode == "dev" else "eval_earn"),
                per_lead=anc["lead"], per_a1=anc["a1"], per_b0=anc["b0"], modes=anc["modes"],
                families=anc["families"], rescue_is_structural=anc["structural"], margin_pp=margin,
                two_tier_same_sign=verdict["earned"]).to_dict()
    wall = time.time() - t0
    report = {"schema": "w140_cross_tier_tutor_bench_v1", "mode": args.mode,
              "reused_per_tier_calibration_cid": cal.get("per_tier_calibration_cid"),
              "tutor_usability_cid": tu.get("tutor_usability_cid"), "arms": arms,
              "shared_families": shared, "anchor_tier": anchor_tier, "weak_tier": weak_tier,
              "tier_results": tier_results, "w140_verdict": verdict, "anchor_gate": gate,
              "fake_different": fd, "wall_s": round(wall, 1),
              "report_cid": _sha256_hex({"mode": args.mode, "tiers": sorted(tier_results)})}
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)

    print("\n=== W140 CROSS-TIER TUTOR BENCH ===")
    for t, r in tier_results.items():
        da1, db0 = verdict["per_tier_T4_minus_A1_B0_pp"].get(t, (None, None))
        print(f"  {t:6s}: T4-A1={da1} T4-B0={db0}")
        for fam, fr in r["per_family"].items():
            arms_s = " ".join(f"{a}={fr['arms'][a]['acc']*100:.0f}({fr['arms'][a]['minus_a1_pp']:+.0f})"
                              for a in arms if a in fr["arms"])
            print(f"     {fam:26s} A0={fr['a0']*100:.0f} A1={fr['a1']*100:.0f} B0={fr['b0']*100:.0f} | {arms_s}")
    print(f"\nW140 verdict: earned={verdict['earned']} anchor_pass={verdict['anchor_pass']} "
          f"weak_pass={verdict['weak_pass']} nonneg={verdict['all_tiers_nonneg']} "
          f"weak_span2={verdict['weak_span_ok_ge2_families']}")
    print(f"  weak per-family T4-A1: {verdict['weak_per_family_T4_minus_A1_pp']}")
    if gate:
        print(f"  anchor gate (lead=T4): passed={gate['passed']} reason={gate['reason']}")
    print(f"wall {wall:.0f}s -> {out_dir}/report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
