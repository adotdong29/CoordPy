#!/usr/bin/env python3
"""W138 Lane β — band mechanism bench (A0/A1/B0 + C0/N0/X1 + X2 $0 control) — RUNBOOK_W138 §7a/§7b/§8.

Mints the held-out band corpus for the CALIBRATION-SURVIVING (family, knob) cells, runs the validated
same-budget arms VERBATIM, and applies the §7a dev gate / §7b eval-earn rule.  X1 (the family-routed
counterexample-else-complexity controller, exact-oracle arm C3 — the M1 controller that scored +33pp in
W137) is the LEAD.  C0 (complexity witness) and N0 (counterexample witness) are diagnostic arms so the
report shows WHICH mode each rescue came from.

Same-budget: every arm K=5, attempt-0 = standard prompt, one model call/attempt, no early stop, graded
on secret; witness generation is $0 (owned-oracle on FRESH probes), outside K.  ``--tier`` runs the
small ladder model for the §7b two-tier same-sign check.

Run:  .venv/bin/python scripts/run_w138_mechanism_bench_v1.py --mode dev
        [--calibration results/w138/w138_calibration_v1.json] [--model meta/llama-3.3-70b-instruct]
        [--arms full|lead] [--n-per-cell 3] [--n-slice 0] [--dry-run]
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
from coordpy.headroom_band_corpus_v3 import (  # noqa: E402
    DEFAULT_MINTED_DATE, corpus_cid_v3, mint_split_v3, per_instance_novelty_v3, summarize_split_v3)
from coordpy.exact_oracle_witness_v1 import build_witness_probe_set_v1, run_witness_arm_v1  # noqa: E402
from coordpy.icpc_reflexion_bench_v1 import (  # noqa: E402
    IcpcBenchConfigV1, IcpcBenchReportV1, IcpcSeedReportV1,
    W120_ICPC_REFLEXION_BENCH_V1_SCHEMA_VERSION, run_icpc_reflexion_bench_v1)
from coordpy.band_mechanism_bench_v1 import (  # noqa: E402
    BAND_ARM_DISPATCH_V1, BAND_ARM_SLATE_V1, arm_scored_on_problem_v1,
    evaluate_gate_v1, fake_different_report_v1)
from scripts.run_w108_livecodebench_pilot import (  # noqa: E402
    _build_nim_gen, _evaluate_phase2_gates, _mlb_rates)

DEFAULT_MODEL = "meta/llama-3.3-70b-instruct"
SMALL_MODEL = "meta/llama-3.1-8b-instruct"
EXEC_TIMEOUT_S = 8.0
WITNESS_TIMEOUT_S = 2.0
WITNESS_SEED = 999_138
SPLIT_SEEDS = {"dev": 138_701, "eval": 138_702, "frontier": 138_703}


def _factory_for(fam: str):
    return CX_FACTORIES.get(fam) or FUNC_FACTORIES.get(fam) or EXTRA_CX_FACTORIES.get(fam)


def _select_family_knobs(cal: dict, mode: str) -> list[tuple[str, int]]:
    """Select (family, knob) for the bench.
    * ``admitted`` — the HB3∧HB4 band survivors (best knob per family); the disciplined locked set.
    * ``non_saturated`` — ALSO include every NON-saturated complexity family (a cell with
      strong_a0<0.80 AND strong_a1<0.95, i.e. room for the witness to rescue) even if HB3 culled it
      as baseline-dead (a1==0).  W137 proved the complexity witness rescues a1==0 cells (+33pp on
      absdiff); the §7a/§7b gate honestly counts only ACTUAL rescues, so this widens the rescue
      surface without weakening the earn thresholds.  Per family the cell with a1 closest to 0.5 is
      chosen (ties -> lower knob)."""
    bk = cal.get("best_knob_per_family", {})
    chosen: dict[str, int] = {fam: int(c["knob_value"]) for fam, c in bk.items()}
    if mode == "non_saturated":
        by_fam: dict[str, list[dict]] = {}
        for c in cal.get("per_cell", []):
            if c.get("strong_a0_rate", 1.0) < 0.80 and c.get("strong_a1_rate", 1.0) < 0.95:
                by_fam.setdefault(c["family"], []).append(c)
        for fam, cells in by_fam.items():
            if fam in chosen:
                continue
            best = min(cells, key=lambda c: (abs(c.get("strong_a1_rate", 0.0) - 0.5),
                                             int(c.get("knob_value", 0))))
            chosen[fam] = int(best["knob_value"])
    return sorted(chosen.items())


def _templates_from_calibration(cal: dict, *, family_mode: str = "admitted"):
    """Reconstruct the bench templates from the calibration JSON per the selection mode."""
    tpls = []
    for fam, knob in _select_family_knobs(cal, family_mode):
        fac = _factory_for(fam)
        if fac is None:
            print(f"  WARN: no factory for family {fam}; skipping")
            continue
        tpls.append(fac(knob))
    return tpls


def _arm_report(model_id, base_seed_rep, arm_outcomes, arm_id, K):
    qids = base_seed_rep.question_ids
    b_passed = tuple(bool(o.final_passed) for o in arm_outcomes)
    b_idx = tuple(int(o.first_pass_attempt_idx) for o in arm_outcomes)
    n = float(len(qids)) or 1.0
    b_acc = sum(1 for x in b_passed if x) / n
    seed_rep = IcpcSeedReportV1(
        schema=W120_ICPC_REFLEXION_BENCH_V1_SCHEMA_VERSION, seed=base_seed_rep.seed,
        n_problems=base_seed_rep.n_problems, a0_pass_at_1=base_seed_rep.a0_pass_at_1,
        a1_pass_at_1=base_seed_rep.a1_pass_at_1, b_pass_at_1=float(b_acc),
        per_problem_a0_passed=base_seed_rep.per_problem_a0_passed,
        per_problem_a1_passed=base_seed_rep.per_problem_a1_passed,
        per_problem_b_passed=b_passed, per_problem_b_first_pass_idx=b_idx,
        question_ids=qids, seed_merkle_root=f"w138_arm_{arm_id}_{base_seed_rep.seed}")
    return IcpcBenchReportV1(
        schema=W120_ICPC_REFLEXION_BENCH_V1_SCHEMA_VERSION, model_id=str(model_id),
        n_problems=base_seed_rep.n_problems, n_seeds=1, K_multi_sample=int(K),
        per_seed=(seed_rep,), a0_mean_pass_at_1=base_seed_rep.a0_pass_at_1,
        a1_mean_pass_at_1=base_seed_rep.a1_pass_at_1, b_mean_pass_at_1=float(b_acc),
        b_mean_minus_a1_mean_pp=float((b_acc - base_seed_rep.a1_pass_at_1) * 100.0),
        bench_merkle_root=f"w138_arm_{arm_id}")


def _run_witness_arms(arms, run_problems, tmpl_by_cell, probes, gen, model, seed, base_seed_rep,
                      max_tokens):
    """Run each witness arm over the problems it is scored on (C0=complexity, N0=noncomplexity, X1=all);
    arms not scored on a problem inherit the A1 outcome there (no-op, same budget)."""
    from coordpy.icpc_reflexion_bench_v1 import IcpcArmOutcomeV1  # local import
    a1_passed = list(base_seed_rep.per_problem_a1_passed)
    out = {}
    for arm in arms:
        kind, oracle_arm = BAND_ARM_DISPATCH_V1[arm]
        outcomes, traces = [], []
        for i, p in enumerate(run_problems):
            if not arm_scored_on_problem_v1(arm, p.mode):
                # not scored here (C0=complexity-only, N0=noncomplexity-only): inherit A1 (same-budget
                # no-op) so the diagnostic arm's accuracy stays aligned with the full problem set
                outcomes.append(IcpcArmOutcomeV1(
                    schema=W120_ICPC_REFLEXION_BENCH_V1_SCHEMA_VERSION, seed=seed,
                    question_id=p.minted.problem_id, arm_id=arm,
                    final_passed=bool(a1_passed[i]), n_model_calls=5,
                    per_call_passed=(bool(a1_passed[i]),), first_pass_attempt_idx=0))
                traces.append({"inherited_a1": True, "rescue_is_algorithmic": True})
                continue
            o, tr = run_witness_arm_v1(
                seed=seed, template=tmpl_by_cell[p.cell_id].minted, problem=p.minted,
                probe=probes[p.minted.problem_id], gen=gen, K=5, temperature=0.7,
                max_tokens=int(max_tokens), timeout_s=EXEC_TIMEOUT_S, arm=oracle_arm,
                minted_date=DEFAULT_MINTED_DATE, witness_timeout_s=WITNESS_TIMEOUT_S)
            outcomes.append(o)
            traces.append(tr.to_dict() if hasattr(tr, "to_dict") else {})
            print(f"    {arm} {i+1}/{len(run_problems)} {p.cell_id} passed={o.final_passed}", flush=True)
        rep = _arm_report(model, base_seed_rep, outcomes, arm, 5)
        mlb = _mlb_rates(rep)
        out[arm] = {
            "phase2": _evaluate_phase2_gates(report=rep, mlb=mlb), "mlb": mlb,
            "per_problem_passed": [bool(o.final_passed) for o in outcomes],
            "rescue_is_algorithmic": [bool(t.get("rescue_is_algorithmic", True)) if isinstance(t, dict)
                                      else True for t in traces]}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dev", "eval", "frontier"], default="dev")
    ap.add_argument("--calibration", default="results/w138/w138_calibration_v1.json")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--arms", choices=["full", "lead"], default="full")
    ap.add_argument("--dev-families", choices=["admitted", "non_saturated"], default="admitted",
                    help="admitted=HB3 band survivors; non_saturated=also include witness-rescuable a1==0 complexity families")
    ap.add_argument("--lead-arm", choices=["X1", "C0", "N0"], default="X1",
                    help="the arm the §7a/§7b gate evaluates (X1=controller; C0=complexity witness, the "
                         "operative lead once the counterexample 2nd-mode is shown dead)")
    ap.add_argument("--complexity-only", action="store_true",
                    help="restrict to COMPLEXITY_BLIND families (the live witness mode)")
    ap.add_argument("--n-per-cell", type=int, default=3, help="replicas per surviving cell for this split")
    ap.add_argument("--n-slice", type=int, default=0, help="bench slice size (0=all admitted)")
    ap.add_argument("--cross-tier-model", default="",
                    help="§7b two-tier check: also run A1+X1 at this model on the SAME slice and "
                         "require same-sign X1-A1 gain (e.g. meta/llama-3.1-8b-instruct)")
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--mint-timeout", type=float, default=1.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    with open(args.calibration) as f:
        cal = json.load(f)
    assert cal.get("slate_fingerprint_cid") == band_slate_fingerprint_cid_v1(
        cx_knobs=tuple(cal.get("cx_knobs", (20000, 50000))),
        func_knobs=tuple(cal.get("func_knobs", (4000, 30000)))), "SLATE DRIFT — refusing NIM"
    surv_fams = list(cal.get("surviving_families", []))
    surv_modes = list(cal.get("surviving_modes", []))
    print(f"surviving families ({len(surv_fams)}): {surv_fams}")
    print(f"surviving modes ({len(surv_modes)}): {surv_modes}")
    tpls = _templates_from_calibration(cal, family_mode=args.dev_families)
    if args.complexity_only:
        tpls = [t for t in tpls if t.minted.mode == "COMPLEXITY_BLIND"]
    print(f"dev-families mode={args.dev_families} complexity_only={args.complexity_only} -> "
          f"{len(tpls)} families: {[t.minted.family for t in tpls]}")
    if not tpls:
        print("NO surviving band cells -> register blocker (no bench).")
        return 0
    tmpl_by_cell = {t.minted.name: t for t in tpls}

    t_mint = time.time()
    minted = mint_split_v3(args.mode, templates=tpls, n_replicas=args.n_per_cell,
                           timeout_s=EXEC_TIMEOUT_S, mint_timeout_s=args.mint_timeout)
    admitted = [p for p in minted if p.admitted]
    summ = summarize_split_v3(minted)
    print(f"minted {len(minted)} -> admitted {len(admitted)} ({time.time()-t_mint:.0f}s); "
          f"modes={summ.mode_histogram} families={summ.family_histogram}")
    run_problems = admitted if args.n_slice <= 0 else admitted[:args.n_slice]
    by_mode = {}
    for p in run_problems:
        by_mode[p.mode] = by_mode.get(p.mode, 0) + 1

    fd = fake_different_report_v1().to_dict()
    print(f"fake-different (X2 control): {fd}")
    novelty = per_instance_novelty_v3(minted)
    print(f"novelty: {novelty}")
    if args.dry_run:
        print("--dry-run: stopping before NIM.")
        return 0

    arms = ["C0", "N0", "X1"] if args.arms == "full" else [args.lead_arm]
    if args.lead_arm not in arms:
        arms.append(args.lead_arm)
    seed = SPLIT_SEEDS[args.mode]
    run_id = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe = str(args.model).replace("/", "_")
    out_dir = os.path.join(ROOT, "results", "w138", args.mode, f"w138_{args.mode}_{safe}_{run_id}")
    os.makedirs(out_dir, exist_ok=True)
    sc = open(os.path.join(out_dir, "calls.jsonl"), "w")
    gen = _build_nim_gen(model=str(args.model),
                         sidecar_writer=lambda r: (sc.write(json.dumps(r) + "\n"), sc.flush()))

    pilots = [p.to_pilot(minted_date=DEFAULT_MINTED_DATE) for p in run_problems]
    cfg = IcpcBenchConfigV1(K_multi_sample=5, seeds=(seed,), sampling_temperature=0.7,
                            max_tokens_per_call=int(args.max_tokens), executor_timeout_s=EXEC_TIMEOUT_S)
    n_done = {"n": 0}

    def on_start(s, idx, qid):
        n_done["n"] += 1
        print(f"  base {n_done['n']}/{len(pilots)} {qid}", flush=True)

    t0 = time.time()
    print(f"baseline A0/A1/B0 on {len(pilots)} problems (model {args.model})...")
    base = run_icpc_reflexion_bench_v1(gen=gen, model_id=str(args.model), subset=pilots,
                                       config=cfg, on_problem_start=on_start)
    bsr = base.per_seed[0]
    base_gates = _evaluate_phase2_gates(report=base, mlb=_mlb_rates(base))

    probes = {p.minted.problem_id: build_witness_probe_set_v1(
        tmpl_by_cell[p.cell_id].minted, p.minted, witness_seed=WITNESS_SEED,
        timeout_s=EXEC_TIMEOUT_S) for p in run_problems}
    arm_results = _run_witness_arms(arms, run_problems, tmpl_by_cell, probes, gen, args.model, seed,
                                    bsr, args.max_tokens)

    modes = [p.mode for p in run_problems]
    fams = [p.family for p in run_problems]
    per_a1 = list(bsr.per_problem_a1_passed)
    per_b0 = list(bsr.per_problem_b_passed)
    lead = arm_results.get(args.lead_arm)

    # §7b two-tier check: run A1 + the LEAD arm at the cross-tier model on the SAME slice; same-sign gain
    cross_tier = None
    two_tier_same_sign = None
    if args.cross_tier_model and lead is not None:
        from coordpy.band_mechanism_bench_v1 import BAND_ARM_DISPATCH_V1  # local
        _, x1_oracle_arm = BAND_ARM_DISPATCH_V1[args.lead_arm]
        print(f"\ncross-tier ({args.cross_tier_model}) A1 + {args.lead_arm} on {len(pilots)} problems...")
        sc_ct = open(os.path.join(out_dir, "calls_cross_tier.jsonl"), "w")
        gen_ct = _build_nim_gen(model=str(args.cross_tier_model),
                                sidecar_writer=lambda r: (sc_ct.write(json.dumps(r) + "\n"), sc_ct.flush()))
        base_ct = run_icpc_reflexion_bench_v1(gen=gen_ct, model_id=str(args.cross_tier_model),
                                              subset=pilots, config=cfg)
        a1_ct = float(base_ct.per_seed[0].a1_pass_at_1)
        x1_ct_pass = []
        for i, p in enumerate(run_problems):
            o, _ = run_witness_arm_v1(
                seed=seed, template=tmpl_by_cell[p.cell_id].minted, problem=p.minted,
                probe=probes[p.minted.problem_id], gen=gen_ct, K=5, temperature=0.7,
                max_tokens=int(args.max_tokens), timeout_s=EXEC_TIMEOUT_S, arm=x1_oracle_arm,
                minted_date=DEFAULT_MINTED_DATE, witness_timeout_s=WITNESS_TIMEOUT_S)
            x1_ct_pass.append(bool(o.final_passed))
            print(f"    X1@ct {i+1}/{len(run_problems)} {p.cell_id} passed={o.final_passed}", flush=True)
        sc_ct.close()
        x1_ct_acc = sum(x1_ct_pass) / (len(x1_ct_pass) or 1)
        x1_minus_a1_primary = (sum(lead["per_problem_passed"]) - sum(per_a1)) / (len(per_a1) or 1)
        x1_minus_a1_ct = x1_ct_acc - a1_ct
        two_tier_same_sign = bool(x1_minus_a1_primary > 0 and x1_minus_a1_ct > 0)
        cross_tier = {"model": args.cross_tier_model, "a1": round(a1_ct, 4),
                      "x1": round(x1_ct_acc, 4), "x1_minus_a1_pp": round(x1_minus_a1_ct * 100, 2),
                      "primary_x1_minus_a1_pp": round(x1_minus_a1_primary * 100, 2),
                      "same_sign": two_tier_same_sign}
        print(f"cross-tier: A1={a1_ct*100:.2f} X1={x1_ct_acc*100:.2f} "
              f"X1-A1={x1_minus_a1_ct*100:+.2f}  same_sign={two_tier_same_sign}")

    gate = None
    if lead is not None:
        gate = evaluate_gate_v1(
            name=("dev_gate" if args.mode == "dev" else "eval_earn"),
            per_lead=lead["per_problem_passed"], per_a1=per_a1, per_b0=per_b0,
            modes=modes, families=fams, rescue_is_structural=lead["rescue_is_algorithmic"],
            margin_pp=(3.33 if args.mode == "dev" else 5.00),
            two_tier_same_sign=two_tier_same_sign).to_dict()

    wall = time.time() - t0
    report = {
        "schema": "w138_mechanism_bench_v1", "mode": args.mode, "model": args.model,
        "n_problems": len(run_problems), "modes": by_mode,
        "family_histogram": dict(summ.family_histogram), "arms": arms,
        "surviving_families": surv_fams, "surviving_modes": surv_modes,
        "n_per_cell": args.n_per_cell, "slate_fingerprint_cid": band_slate_fingerprint_cid_v1(
            cx_knobs=tuple(cal.get("cx_knobs", (20000, 50000))),
            func_knobs=tuple(cal.get("func_knobs", (4000, 30000)))),
        "calibration_cid": cal.get("calibration_cid"),
        "split_cid": summ.split_cid(),
        "baseline": {"a0": base.a0_mean_pass_at_1, "a1": base.a1_mean_pass_at_1,
                     "b0": base.b_mean_pass_at_1, "b0_gates": base_gates},
        "per_problem": {"cell": [p.cell_id for p in run_problems], "mode": modes, "family": fams,
                        "a0": [bool(x) for x in bsr.per_problem_a0_passed],
                        "a1": [bool(x) for x in per_a1], "b0": [bool(x) for x in per_b0],
                        **{a: r["per_problem_passed"] for a, r in arm_results.items()}},
        "arms_result": {a: {"phase2": r["phase2"], "mlb": r["mlb"],
                            "per_problem_passed": r["per_problem_passed"]}
                        for a, r in arm_results.items()},
        "fake_different": fd, "novelty": novelty, "gate": gate, "cross_tier": cross_tier,
        "arm_slate": [a.to_dict() for a in BAND_ARM_SLATE_V1], "wall_s": round(wall, 1)}
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    sc.close()

    print("\n=== W138 mechanism bench ===")
    print(f"A0={base.a0_mean_pass_at_1*100:.2f}  A1={base.a1_mean_pass_at_1*100:.2f}  "
          f"B0={base.b_mean_pass_at_1*100:.2f}")
    for a, r in arm_results.items():
        g = r["phase2"]
        print(f"{a}: acc={g['b_pct']:.2f}  {a}-A1={g['b_minus_a1_pp']:+.2f}  "
              f"MLB1={r['mlb']['mlb1_invocation_rate']:.2f} MLB2={r['mlb']['mlb2_rescue_rate']:.2f}")
    if gate:
        print(f"\n§7a/§7b lead({args.lead_arm}) gate: {gate['reason']}  passed={gate['passed']}")
        print(f"  lead-A1={gate['lead_minus_a1_pp']:+.2f}  lead-B0={gate['lead_minus_b0_pp']:+.2f}  "
              f"rescues={gate['n_rescues_vs_b0']} modes={gate['rescue_modes']} fams={gate['rescue_families']}")
    print(f"wall {wall:.0f}s -> {out_dir}/report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
