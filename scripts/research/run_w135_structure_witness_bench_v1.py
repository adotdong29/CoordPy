#!/usr/bin/env python3
"""W135-beta — held-out NON-COMPLEXITY structure-witness mechanism bench (dev / eval / frontier).

Arms (RUNBOOK_W135 §6), all same-budget K=5, each scored in the "B" slot so "arm - A1" is
byte-identical to "B - A1" (the verbatim W108 `_mlb_rates` / `_evaluate_phase2_gates`):

  A0/A1/B0   — the validated blind W120/W132 stack (`run_icpc_reflexion_bench_v1`)
  C1         — the exact-oracle EW1 COUNTEREXAMPLE witness (W133 `ARM_C1_COUNTEREXAMPLE`); the FLAT
               baseline the structure arms must beat by >= +5pp (the W135 thesis)
  D0         — the W134 deployable COMPLEXITY witness controller; NEGATIVE control (≈B0 on non-cx)
  S1/S2/S3/S4 — the solution-STRUCTURE witness arms (`run_structure_witness_arm_v1`); S4 = lead

Modes::
    python scripts/run_w135_structure_witness_bench_v1.py --mode dev --dry-run          # 0 NIM
    python scripts/run_w135_structure_witness_bench_v1.py --mode dev                     # full slate DEV
    python scripts/run_w135_structure_witness_bench_v1.py --mode eval --lead S4          # lead+C1+baselines
    python scripts/run_w135_structure_witness_bench_v1.py --mode frontier --lead S4

Spend discipline (RUNBOOK §7/§9): dev is the go/no-go; eval runs only if the dev gate clears;
frontier runs only if the eval earn rule passes.  The eval/frontier slice CIDs are asserted before
spend.  Requires `NVIDIA_API_KEY`.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import sys  # noqa: E402
sys.path.insert(0, str(ROOT))

from coordpy.noncomplexity_structure_corpus_v1 import (  # noqa: E402
    MIN_FRONTIER, MINTED_DATE, load_or_build_corpus_v1, select_dev_bench_slice_v1,
    select_frontier_slice_v1,
)
from coordpy.solution_structure_witness_v1 import (  # noqa: E402
    ARM_S1_GREEDY, ARM_S2_SUBSTRUCTURE, ARM_S3_SEARCH, ARM_S4_CONTROLLER,
    run_structure_witness_arm_v1,
)
from coordpy.exact_oracle_witness_v1 import (  # noqa: E402
    ARM_C1_COUNTEREXAMPLE, build_witness_probe_set_v1, run_witness_arm_v1,
)
from coordpy.deployable_complexity_witness_v1 import (  # noqa: E402
    ARM_D3_CONTROLLER, LADDER_VALUE_SEED, run_deployable_witness_arm_v1,
)
from coordpy.icpc_reflexion_bench_v1 import (  # noqa: E402
    IcpcBenchConfigV1, IcpcBenchReportV1, IcpcSeedReportV1,
    W120_ICPC_REFLEXION_BENCH_V1_SCHEMA_VERSION, run_icpc_reflexion_bench_v1,
)
from scripts.run_w108_livecodebench_pilot import (  # noqa: E402
    _build_nim_gen, _evaluate_phase2_gates, _mlb_rates,
)

DEFAULT_MODEL = "meta/llama-3.3-70b-instruct"
EXEC_TIMEOUT_S = 8.0
WITNESS_TIMEOUT_S = 2.0
C1_WITNESS_SEED = 999_135
STRUCT_ARMS = (ARM_S1_GREEDY, ARM_S2_SUBSTRUCTURE, ARM_S3_SEARCH, ARM_S4_CONTROLLER)
CORE_STRUCT_ARMS = (ARM_S2_SUBSTRUCTURE, ARM_S4_CONTROLLER)   # latency-throttle subset
C1_ARM, D0_ARM = "C1", "D0"
LATENCY_THROTTLE_S = 12.0
# LOCKED before any eval/frontier spend (filled from results/w135/corpus/corpus_build_v1.json,
# the $0 Lane-α build at timeout_s=8.0 predating all β NIM). Asserted on eval/frontier.
LOCKED_CORPUS_CID = "306610aee0819ac9e40244e2de09538e85ae71ceba2b87909b4c81bdc567ca18"
LOCKED_EVAL_SPLIT_CID_PREFIX = "3f6e3e599dc0abf3"
LOCKED_FRONTIER_SLICE_CID_PREFIX = "8aa535644d934f3c"
CORPUS_CACHE = ROOT / "results" / "w135" / "corpus" / "corpus_cache.pkl"
SPLIT_SEEDS = {"dev": 135_201, "eval": 135_202, "frontier": 135_203}


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
        question_ids=qids, seed_merkle_root=f"w135_arm_{arm_id}_{base_seed_rep.seed}")
    return IcpcBenchReportV1(
        schema=W120_ICPC_REFLEXION_BENCH_V1_SCHEMA_VERSION, model_id=str(model_id),
        n_problems=base_seed_rep.n_problems, n_seeds=1, K_multi_sample=int(K),
        per_seed=(seed_rep,), a0_mean_pass_at_1=base_seed_rep.a0_pass_at_1,
        a1_mean_pass_at_1=base_seed_rep.a1_pass_at_1, b_mean_pass_at_1=float(b_acc),
        b_mean_minus_a1_mean_pp=float((b_acc - base_seed_rep.a1_pass_at_1) * 100.0),
        bench_merkle_root=f"w135_arm_{arm_id}")


def _rescues_vs(per_b, per_ref, qids):
    return [q for i, q in enumerate(qids) if per_b[i] and not per_ref[i]]


def _select(mode, per_family):
    corpus = load_or_build_corpus_v1(CORPUS_CACHE, expect_corpus_cid=None)
    if LOCKED_CORPUS_CID != "__FILL_FROM_ALPHA_BUILD__" and corpus.corpus_cid() != LOCKED_CORPUS_CID:
        raise SystemExit(f"corpus CID {corpus.corpus_cid()[:16]} != locked {LOCKED_CORPUS_CID[:16]}; refusing NIM.")
    if not corpus.meets_floors:
        raise SystemExit("corpus does not meet the 36/36/36/30 floors; refusing NIM.")
    mode_by_id = corpus.mode_by_problem_id()
    if mode == "dev":
        probs = list(select_dev_bench_slice_v1(corpus.dev, per_family=per_family))
        return corpus, probs, "dev_bench", {p.problem_id: p.family for p in probs}, mode_by_id
    if mode == "eval":
        if LOCKED_EVAL_SPLIT_CID_PREFIX != "__FILL__" and not corpus.eval.split_cid.startswith(LOCKED_EVAL_SPLIT_CID_PREFIX):
            raise SystemExit(f"eval split CID {corpus.eval.split_cid[:16]} != locked; refusing eval NIM.")
        probs = list(select_frontier_slice_v1(corpus.eval, n=MIN_FRONTIER))
        return corpus, probs, corpus.eval.split_cid, {p.problem_id: p.family for p in probs}, mode_by_id
    fslice = select_frontier_slice_v1(corpus.frontier, n=MIN_FRONTIER)
    if LOCKED_FRONTIER_SLICE_CID_PREFIX != "__FILL__" and not corpus.frontier_slice_cid.startswith(LOCKED_FRONTIER_SLICE_CID_PREFIX):
        raise SystemExit(f"frontier slice CID {corpus.frontier_slice_cid[:16]} != locked; refusing frontier NIM.")
    probs = list(fslice)
    return corpus, probs, corpus.frontier_slice_cid, {p.problem_id: p.family for p in probs}, mode_by_id


def _run_one_arm(arm, run_problems, probes, tmpl_by_id, gen, seed, max_tokens):
    outcomes, traces = [], []
    for i, p in enumerate(run_problems):
        if arm == C1_ARM:
            o, tr = run_witness_arm_v1(
                seed=int(seed), template=tmpl_by_id[p.problem_id], problem=p,
                probe=probes[p.problem_id], gen=gen, K=5, temperature=0.7,
                max_tokens=int(max_tokens), timeout_s=EXEC_TIMEOUT_S,
                arm=ARM_C1_COUNTEREXAMPLE, minted_date=MINTED_DATE, witness_timeout_s=WITNESS_TIMEOUT_S)
            traces.append(tr.to_dict())
        elif arm == D0_ARM:
            o, tr = run_deployable_witness_arm_v1(
                seed=int(seed), pilot=p.to_pilot_problem(minted_date=MINTED_DATE), gen=gen, K=5,
                temperature=0.7, max_tokens=int(max_tokens), timeout_s=EXEC_TIMEOUT_S,
                arm=ARM_D3_CONTROLLER, ladder_seed=LADDER_VALUE_SEED, witness_timeout_s=WITNESS_TIMEOUT_S)
            traces.append(tr.to_dict())
        else:  # structure arm
            o, tr = run_structure_witness_arm_v1(
                seed=int(seed), template=tmpl_by_id[p.problem_id], problem=p,
                probe=probes[p.problem_id], gen=gen, K=5, temperature=0.7,
                max_tokens=int(max_tokens), timeout_s=EXEC_TIMEOUT_S, arm=arm,
                minted_date=MINTED_DATE, witness_timeout_s=WITNESS_TIMEOUT_S)
            traces.append(tr.to_dict())
        outcomes.append(o)
        print(f"    {arm} {i+1}/{len(run_problems)} {p.problem_id} passed={o.final_passed}", flush=True)
    return outcomes, traces


def main() -> int:
    ap = argparse.ArgumentParser(description="W135 structure-witness non-complexity bench")
    ap.add_argument("--mode", choices=["dev", "eval", "frontier"], required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--lead", choices=list(STRUCT_ARMS), default=ARM_S4_CONTROLLER)
    ap.add_argument("--per-family", type=int, default=1, help="dev bench instances/family (§9)")
    ap.add_argument("--arms", choices=["full", "core", "lead"], default="full")
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "w135"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    corpus, run_problems, slice_cid, fams, mode_by_id = _select(args.mode, args.per_family)
    tmpl_by_id = corpus.template_by_problem_id()
    by_fam, by_mode = {}, {}
    for p in run_problems:
        by_fam[p.family] = by_fam.get(p.family, 0) + 1
        by_mode[p.mode] = by_mode.get(p.mode, 0) + 1
    print(f"  mode={args.mode} n={len(run_problems)} families={len(by_fam)} modes={by_mode} "
          f"slice_cid={str(slice_cid)[:16]} model={args.model}")
    if args.dry_run:
        print(f"  --dry-run: validated {len(run_problems)} problems "
              f"(corpus_cid={corpus.corpus_cid()[:16]}); stopping before NIM.")
        return 0

    run_id = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_model = str(args.model).replace("/", "_")
    out_dir = Path(args.out_dir) / args.mode / f"w135_{args.mode}_{safe_model}_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    sidecar_f = open(out_dir / "calls.jsonl", "w")

    def sidecar_writer(rec):
        sidecar_f.write(json.dumps(rec, separators=(",", ":")) + "\n")
        sidecar_f.flush()

    gen = _build_nim_gen(model=str(args.model), sidecar_writer=sidecar_writer)

    # latency probe (1 tiny call): throttle the dev arm set if the endpoint is slow (§9)
    arm_set = args.arms
    if args.mode == "dev" and args.arms == "full":
        t_probe = time.time()
        try:
            gen("Reply with the single token OK.", 8, 0.0)
        except Exception as e:  # noqa: BLE001
            print(f"  latency probe error: {e}")
        dt_probe = time.time() - t_probe
        if dt_probe > LATENCY_THROTTLE_S:
            arm_set = "core"
            print(f"  latency probe {dt_probe:.1f}s > {LATENCY_THROTTLE_S}s ⇒ THROTTLE to core arms")
        else:
            print(f"  latency probe {dt_probe:.1f}s ⇒ full arm set")

    if args.mode == "dev":
        if arm_set == "full":
            arms = [C1_ARM, D0_ARM] + list(STRUCT_ARMS)
        elif arm_set == "core":
            arms = [C1_ARM] + list(CORE_STRUCT_ARMS)
        else:  # lead: the minimal DECISIVE set — base + C1 (flat baseline) + S4 (pre-committed lead,
               # which renders the union of SW1/SW2/SW3 structure). Ablations (S1/S2/S3) + D0 are a
               # $0-staged follow-up run ONLY if S4 clears the dev gate vs C1.
            arms = [C1_ARM, ARM_S4_CONTROLLER]
    else:
        arms = [C1_ARM, args.lead]
    seed = SPLIT_SEEDS[args.mode]
    cfg = IcpcBenchConfigV1(K_multi_sample=5, seeds=(int(seed),), sampling_temperature=0.7,
                            max_tokens_per_call=int(args.max_tokens), executor_timeout_s=EXEC_TIMEOUT_S)
    pilot_subset = [p.to_pilot_problem(minted_date=MINTED_DATE) for p in run_problems]

    t0 = time.time()
    print("  [1] A0/A1/B0 (validated blind stack) ...")
    base = run_icpc_reflexion_bench_v1(
        gen=gen, model_id=str(args.model), subset=pilot_subset, config=cfg,
        on_problem_start=lambda s, i, q: print(f"    base {i+1}/{len(pilot_subset)} {q}", flush=True))
    base_seed_rep = base.per_seed[0]

    print("  [2] witness probe sets ($0 NIM) ...")
    probes = {p.problem_id: build_witness_probe_set_v1(tmpl_by_id[p.problem_id], p,
              witness_seed=C1_WITNESS_SEED, timeout_s=WITNESS_TIMEOUT_S) for p in run_problems}

    arm_reports, arm_gates, arm_traces = {}, {}, {}
    for arm in arms:
        print(f"  [3] arm {arm} ...")
        outcomes, traces = _run_one_arm(arm, run_problems, probes, tmpl_by_id, gen, seed, args.max_tokens)
        rep = _arm_report(args.model, base_seed_rep, outcomes, arm, K=5)
        mlb = _mlb_rates(rep)
        arm_reports[arm] = rep
        arm_gates[arm] = {"mlb": mlb, "phase2": _evaluate_phase2_gates(report=rep, mlb=mlb),
                          "per_problem_passed": [bool(o.final_passed) for o in outcomes]}
        arm_traces[arm] = traces

    sidecar_f.close()
    wall = time.time() - t0
    qids = list(base_seed_rep.question_ids)
    a1 = list(base_seed_rep.per_problem_a1_passed)
    b0 = list(base_seed_rep.per_problem_b_passed)
    base_gates = _evaluate_phase2_gates(report=base, mlb=_mlb_rates(base))
    c1_pp = arm_gates[C1_ARM]["per_problem_passed"]
    c1_pct = arm_gates[C1_ARM]["phase2"]["b_pct"]

    arm_summ = {}
    for arm in arms:
        g = arm_gates[arm]["phase2"]
        bp = arm_gates[arm]["per_problem_passed"]
        r_b0 = _rescues_vs(bp, b0, qids)
        r_c1 = _rescues_vs(bp, c1_pp, qids)
        modes_c1 = sorted({mode_by_id.get(q, "?") for q in r_c1})
        fams_c1 = sorted({fams.get(q, "?") for q in r_c1})
        # structural-rescue audit (§7b cond 4): a vs-C1 rescue counts iff the structure witness fired
        # genuinely-new on that problem (a ladder or a new greedy datapoint), not a format-only fix.
        tr_by_q = {t["problem_id"]: t for t in arm_traces[arm]}
        structural = [q for q in r_c1 if tr_by_q.get(q, {}).get("any_genuinely_new", False)]
        formatting_only = bool(r_c1 and not structural and arm in STRUCT_ARMS)
        span_ok = bool(len(modes_c1) >= 2 or len(fams_c1) >= 3)
        is_struct = arm in STRUCT_ARMS
        arm_summ[arm] = {
            "arm_pct": g["b_pct"], "arm_minus_a1_pp": g["b_minus_a1_pp"],
            "arm_minus_b0_pp": round(g["b_pct"] - base_gates["b_pct"], 4),
            "arm_minus_c1_pp": round(g["b_pct"] - c1_pct, 4),
            "rescues_vs_b0": r_b0, "rescues_vs_c1": r_c1,
            "modes_rescued_vs_c1": modes_c1, "n_modes_rescued_vs_c1": len(modes_c1),
            "families_rescued_vs_c1": fams_c1, "n_families_rescued_vs_c1": len(fams_c1),
            "structural_rescues_vs_c1": structural, "formatting_only": formatting_only,
            "dev_gate_pass": bool(is_struct
                                  and round(g["b_pct"] - base_gates["b_pct"], 4) >= 3.33
                                  and round(g["b_pct"] - c1_pct, 4) >= 3.33 and span_ok),
            "eval_earn_pass": bool(is_struct and g["b_minus_a1_pp"] >= 0.0
                                   and round(g["b_pct"] - base_gates["b_pct"], 4) >= 5.0
                                   and round(g["b_pct"] - c1_pct, 4) >= 5.0 and span_ok
                                   and not formatting_only and len(structural) == len(r_c1)),
            "mlb": arm_gates[arm]["mlb"], "phase2": g}

    # pre-committed lead selection: argmax over struct arms of (arm - C1) among span-ok arms;
    # ties -> S4 > S2 > S3 > S1.
    order = {ARM_S4_CONTROLLER: 0, ARM_S2_SUBSTRUCTURE: 1, ARM_S3_SEARCH: 2, ARM_S1_GREEDY: 3}
    span_ok_arms = [a for a in arms if a in STRUCT_ARMS and arm_summ[a]["n_modes_rescued_vs_c1"] >= 2
                    or (a in STRUCT_ARMS and arm_summ[a]["n_families_rescued_vs_c1"] >= 3)]
    cand = span_ok_arms or [a for a in arms if a in STRUCT_ARMS]
    lead = sorted(cand, key=lambda a: (-arm_summ[a]["arm_minus_c1_pp"], order.get(a, 9)))[0] if cand else None

    payload = {"schema": "coordpy.w135_structure_witness_bench_v1", "mode": args.mode,
               "model_id": args.model, "seed": seed, "n_problems": len(run_problems),
               "arm_set": arm_set, "slice_cid": slice_cid, "corpus_cid": corpus.corpus_cid(),
               "by_family": by_fam, "by_mode": by_mode, "wall_s": round(wall, 2),
               "minted_date": MINTED_DATE, "question_ids": qids, "families": fams,
               "modes": {q: mode_by_id.get(q, "?") for q in qids},
               "base_A0_A1_B0": {"a0_pct": base_gates["a0_pct"], "a1_pct": base_gates["a1_pct"],
                                 "b0_pct": base_gates["b_pct"], "b0_minus_a1_pp": base_gates["b_minus_a1_pp"],
                                 "per_problem_a1": a1, "per_problem_b0": b0,
                                 "mlb": _mlb_rates(base), "phase2": base_gates},
               "c1_counterexample_pct": c1_pct, "lead_arm": lead, "arms": arm_summ,
               "arm_traces": arm_traces}
    (out_dir / f"w135_{args.mode}_report.json").write_text(json.dumps(payload, indent=2, default=str))
    (Path(args.out_dir) / args.mode / f"latest_{args.mode}.txt").write_text(out_dir.name + "\n")

    print(f"\n  WALL {wall:.1f}s  A0={base_gates['a0_pct']:.2f} A1={base_gates['a1_pct']:.2f} "
          f"B0={base_gates['b_pct']:.2f} (B0-A1={base_gates['b_minus_a1_pp']:+.2f}pp)  "
          f"C1={c1_pct:.2f} (C1-B0={c1_pct-base_gates['b_pct']:+.2f}pp)")
    for arm in arms:
        s = arm_summ[arm]
        print(f"  {arm}: {s['arm_pct']:.2f}%  (a-A1={s['arm_minus_a1_pp']:+.2f} "
              f"a-B0={s['arm_minus_b0_pp']:+.2f} a-C1={s['arm_minus_c1_pp']:+.2f})  "
              f"resc_vs_C1={len(s['rescues_vs_c1'])} ({s['n_modes_rescued_vs_c1']}m/{s['n_families_rescued_vs_c1']}f "
              f"struct={len(s['structural_rescues_vs_c1'])}) MLB-2 {s['mlb']['mlb2_rescue_rate']*100:.1f}%  "
              f"dev_gate={s['dev_gate_pass']} eval_earn={s['eval_earn_pass']}")
    print(f"  LEAD (pre-committed) = {lead}")
    print(f"  out_dir: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
