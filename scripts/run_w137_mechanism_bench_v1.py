#!/usr/bin/env python3
"""W137 Lane β — repaired-field model-ladder mechanism bench (A0/A1/B0/C0/M1/M2 + M3 $0 control).

Mints the repaired-field split for the CALIBRATION-SURVIVING templates, runs the validated
same-budget arms verbatim, and applies the RUNBOOK_W137 §7a dev gate / §7b eval-earn rule.  M1 (the
auto-routing counterexample-else-complexity controller, exact-oracle arm C3) is the LEAD.

Same-budget: every arm K=5, attempt-0 = standard prompt, one model call/attempt, no early stop,
graded on secret; witness generation is $0 (owned-oracle on FRESH probes), outside K.

Run:  .venv/bin/python scripts/run_w137_mechanism_bench_v1.py --mode dev --survivors results/w137/w137_calibration_v1.json [--arms full|lead] [--n-per-template 2]
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

from coordpy.hard_battlefield_slate_v2 import build_hard_slate_v2, slate_fingerprint_cid_v1  # noqa: E402
from coordpy.hard_battlefield_corpus_v2 import (  # noqa: E402
    DEFAULT_MINTED_DATE, admit_by_template, mint_split_v2, summarize_split_v2)
from coordpy.exact_oracle_witness_v1 import (  # noqa: E402
    ARM_C2_COMPLEXITY, ARM_C3_CONTROLLER, build_witness_probe_set_v1, run_witness_arm_v1)
from coordpy.deployable_complexity_witness_v1 import (  # noqa: E402
    ARM_D3_CONTROLLER, LADDER_VALUE_SEED, run_deployable_witness_arm_v1)
from coordpy.icpc_reflexion_bench_v1 import (  # noqa: E402
    IcpcBenchConfigV1, IcpcBenchReportV1, IcpcSeedReportV1,
    W120_ICPC_REFLEXION_BENCH_V1_SCHEMA_VERSION, run_icpc_reflexion_bench_v1)
from coordpy.repaired_field_mechanism_bench_v1 import (  # noqa: E402
    ARM_SLATE_V1, evaluate_gate_v1, fake_different_report_v1)
from scripts.run_w108_livecodebench_pilot import (  # noqa: E402
    _build_nim_gen, _evaluate_phase2_gates, _mlb_rates)

LOCKED_SLATE_CID = "2ce207c567324e4322f308e58a1fc2c88a8d4bdd0e340d2ec8a1b867d82b3f70"
DEFAULT_MODEL = "meta/llama-3.3-70b-instruct"
EXEC_TIMEOUT_S = 8.0
WITNESS_TIMEOUT_S = 2.0
WITNESS_SEED = 999_137
SPLIT_SEEDS = {"dev": 137_701, "eval": 137_702, "frontier": 137_703}
# arm id -> (kind, exact-oracle arm constant)
ARM_DISPATCH = {"C0": ("witness", ARM_C2_COMPLEXITY), "M1": ("witness", ARM_C3_CONTROLLER),
                "M2": ("deployable", ARM_D3_CONTROLLER)}


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
        question_ids=qids, seed_merkle_root=f"w137_arm_{arm_id}_{base_seed_rep.seed}")
    return IcpcBenchReportV1(
        schema=W120_ICPC_REFLEXION_BENCH_V1_SCHEMA_VERSION, model_id=str(model_id),
        n_problems=base_seed_rep.n_problems, n_seeds=1, K_multi_sample=int(K),
        per_seed=(seed_rep,), a0_mean_pass_at_1=base_seed_rep.a0_pass_at_1,
        a1_mean_pass_at_1=base_seed_rep.a1_pass_at_1, b_mean_pass_at_1=float(b_acc),
        b_mean_minus_a1_mean_pp=float((b_acc - base_seed_rep.a1_pass_at_1) * 100.0),
        bench_merkle_root=f"w137_arm_{arm_id}")


def _stratified_slice(problems, n):
    """Mode-stratified, deterministic slice of size n."""
    by_mode = {}
    for p in sorted(problems, key=lambda p: (p.mode, p.template_name, p.seed)):
        by_mode.setdefault(p.mode, []).append(p)
    out, modes = [], sorted(by_mode)
    i = 0
    while len(out) < n and any(by_mode.values()):
        m = modes[i % len(modes)]
        if by_mode[m]:
            out.append(by_mode[m].pop(0))
        i += 1
        if i > 100000:
            break
    return out[:n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dev", "eval", "frontier"], default="dev")
    ap.add_argument("--survivors", default="results/w137/w137_calibration_v1.json")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--arms", choices=["full", "lead"], default="full")
    ap.add_argument("--n-per-template", type=int, default=2, help="replicas/template for this split")
    ap.add_argument("--n-slice", type=int, default=0, help="bench slice size (0=all admitted)")
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--mint-timeout", type=float, default=3.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    assert slate_fingerprint_cid_v1() == LOCKED_SLATE_CID, "SLATE DRIFT — refusing NIM"
    with open(args.survivors) as f:
        cal = json.load(f)
    survivors = list(cal.get("surviving", []))
    assert cal.get("slate_fingerprint_cid") == LOCKED_SLATE_CID, "calibration slate drift"
    print(f"survivors ({len(survivors)}): {survivors}")
    if not survivors:
        print("NO survivors -> repaired field has no headroom band; register blocker (no bench).")
        return 0

    tmpl_by_name = {t.minted.name: t for t in build_hard_slate_v2()}
    surv_tpls = [tmpl_by_name[n] for n in survivors if n in tmpl_by_name]

    t_mint = time.time()
    minted = mint_split_v2(args.mode, n_replicas=args.n_per_template,
                           timeout_s=args.mint_timeout, templates=surv_tpls)
    admitted = admit_by_template(minted, survivors)
    summ = summarize_split_v2(minted)
    print(f"minted {len(minted)} -> admitted {len(admitted)} ({time.time()-t_mint:.0f}s mint); "
          f"modes={summ.mode_histogram}")
    run_problems = _stratified_slice(admitted, args.n_slice) if args.n_slice > 0 else admitted
    by_mode = {}
    for p in run_problems:
        by_mode[p.mode] = by_mode.get(p.mode, 0) + 1
    print(f"bench slice: {len(run_problems)} problems; modes={by_mode}")

    # fake-different discipline (M3 $0)
    fd = fake_different_report_v1()
    print(f"fake-different (M3 control): {fd.to_dict()}")

    if args.dry_run:
        print("--dry-run: stopping before NIM.")
        return 0

    arms = ["C0", "M1", "M2"] if args.arms == "full" else ["M1"]
    seed = SPLIT_SEEDS[args.mode]
    run_id = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe = str(args.model).replace("/", "_")
    out_dir = os.path.join(ROOT, "results", "w137", args.mode, f"w137_{args.mode}_{safe}_{run_id}")
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
    print(f"baseline A0/A1/B0 on {len(pilots)} problems...")
    base = run_icpc_reflexion_bench_v1(gen=gen, model_id=str(args.model), subset=pilots,
                                       config=cfg, on_problem_start=on_start)
    bsr = base.per_seed[0]
    base_gates = _evaluate_phase2_gates(report=base, mlb=_mlb_rates(base))

    # probe sets for the witness arms (built once; $0 owned-oracle on FRESH inputs)
    probes = {p.minted.problem_id: build_witness_probe_set_v1(
        tmpl_by_name[p.template_name].minted, p.minted, witness_seed=WITNESS_SEED,
        timeout_s=EXEC_TIMEOUT_S) for p in run_problems}

    arm_results = {}
    for arm in arms:
        kind, oracle_arm = ARM_DISPATCH[arm]
        outcomes, traces = [], []
        for i, p in enumerate(run_problems):
            if kind == "witness":
                o, tr = run_witness_arm_v1(
                    seed=seed, template=tmpl_by_name[p.template_name].minted, problem=p.minted,
                    probe=probes[p.minted.problem_id], gen=gen, K=5, temperature=0.7,
                    max_tokens=int(args.max_tokens), timeout_s=EXEC_TIMEOUT_S, arm=oracle_arm,
                    minted_date=DEFAULT_MINTED_DATE, witness_timeout_s=WITNESS_TIMEOUT_S)
            else:
                o, tr = run_deployable_witness_arm_v1(
                    seed=seed, pilot=p.to_pilot(minted_date=DEFAULT_MINTED_DATE), gen=gen, K=5,
                    temperature=0.7, max_tokens=int(args.max_tokens), timeout_s=EXEC_TIMEOUT_S,
                    arm=oracle_arm, ladder_seed=LADDER_VALUE_SEED, witness_timeout_s=WITNESS_TIMEOUT_S)
            outcomes.append(o)
            traces.append(tr.to_dict())
            print(f"    {arm} {i+1}/{len(run_problems)} {p.minted.problem_id} passed={o.final_passed}", flush=True)
        rep = _arm_report(args.model, bsr, outcomes, arm, 5)
        mlb = _mlb_rates(rep)
        arm_results[arm] = {
            "phase2": _evaluate_phase2_gates(report=rep, mlb=mlb), "mlb": mlb,
            "per_problem_passed": [bool(o.final_passed) for o in outcomes],
            "traces": traces,
            "rescue_is_algorithmic": [bool(t.get("rescue_is_algorithmic", True)) if isinstance(t, dict) else True
                                      for t in traces]}

    # §7a dev gate on the lead (M1) — structural rescue audit from the arm trace
    modes = [p.mode for p in run_problems]
    fams = [p.family for p in run_problems]
    per_a1 = list(bsr.per_problem_a1_passed)
    per_b0 = list(bsr.per_problem_b_passed)
    lead = arm_results.get("M1")
    dev_gate = None
    if lead is not None:
        # a rescue counts as structural iff the arm's witness was algorithmic AND the problem is not
        # a parsing artifact (HC1-guaranteed) — on the parser-neutral field there are no parsing rescues
        struct = lead["rescue_is_algorithmic"]
        dev_gate = evaluate_gate_v1(
            name=("dev_gate" if args.mode == "dev" else "eval_earn"),
            per_lead=lead["per_problem_passed"], per_a1=per_a1, per_b0=per_b0,
            modes=modes, families=fams, rescue_is_structural=struct,
            margin_pp=(3.33 if args.mode == "dev" else 5.00)).to_dict()

    wall = time.time() - t0
    report = {
        "schema": "w137_mechanism_bench_v1", "mode": args.mode, "model": args.model,
        "n_problems": len(run_problems), "modes": by_mode, "arms": arms,
        "survivors": survivors, "n_per_template": args.n_per_template,
        "slate_fingerprint_cid": slate_fingerprint_cid_v1(),
        "baseline": {"a0": base.a0_mean_pass_at_1, "a1": base.a1_mean_pass_at_1,
                     "b0": base.b_mean_pass_at_1, "b0_gates": base_gates},
        "arms_result": {a: {"phase2": r["phase2"], "mlb": r["mlb"],
                            "per_problem_passed": r["per_problem_passed"]}
                        for a, r in arm_results.items()},
        "fake_different": fd.to_dict(), "dev_gate": dev_gate,
        "arm_slate": [a.__dict__ for a in ARM_SLATE_V1], "wall_s": round(wall, 1)}
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    sc.close()

    print("\n=== W137 mechanism bench ===")
    print(f"A0={base.a0_mean_pass_at_1*100:.2f}  A1={base.a1_mean_pass_at_1*100:.2f}  "
          f"B0={base.b_mean_pass_at_1*100:.2f}")
    for a, r in arm_results.items():
        g = r["phase2"]
        print(f"{a}: acc={g['b_pct']:.2f}  {a}-A1={g['b_minus_a1_pp']:+.2f}  "
              f"MLB1={r['mlb']['mlb1_invocation_rate']:.2f} MLB2={r['mlb']['mlb2_rescue_rate']:.2f}")
    if dev_gate:
        print(f"\n§7a/§7b lead(M1) gate: {dev_gate['reason']}  passed={dev_gate['passed']}")
        print(f"  lead-A1={dev_gate['lead_minus_a1_pp']:+.2f}  lead-B0={dev_gate['lead_minus_b0_pp']:+.2f}  "
              f"rescues={dev_gate['n_rescues_vs_b0']} modes={dev_gate['rescue_modes']}")
    print(f"wall {wall:.0f}s -> {out_dir}/report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
