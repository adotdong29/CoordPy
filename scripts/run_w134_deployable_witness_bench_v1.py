#!/usr/bin/env python3
"""W134-beta — held-out COMPLEXITY-ONLY deployable-witness mechanism bench (dev / eval / frontier).

Arms (RUNBOOK_W134 §6), all same-budget K=5, each scored in the "B" slot so "arm - A1" is
byte-identical to "B - A1" (the verbatim W108 `_mlb_rates` / `_evaluate_phase2_gates`):

  A0/A1/B0  — the validated blind W120/W132 stack (`run_icpc_reflexion_bench_v1`)
  C0        — the exact-oracle EW2 complexity witness (W133 `ARM_C2_COMPLEXITY`); the UPPER BOUND
  D1/D2/D3  — the DEPLOYABLE complexity witness arms (`run_deployable_witness_arm_v1`); D3 = lead

Modes::
    python scripts/run_w134_deployable_witness_bench_v1.py --mode dev --dry-run     # 0 NIM
    python scripts/run_w134_deployable_witness_bench_v1.py --mode dev               # full slate on DEV
    python scripts/run_w134_deployable_witness_bench_v1.py --mode eval --lead D3    # lead+C0+baselines
    python scripts/run_w134_deployable_witness_bench_v1.py --mode frontier --lead D3

Spend discipline (RUNBOOK §7/§9): dev is the go/no-go; eval runs only if the dev gate clears;
frontier runs only if the eval earn rule passes.  The eval split CID and the frontier 30-slice CID
are asserted before eval / frontier NIM.  Requires `NVIDIA_API_KEY`.
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

from coordpy.complexity_only_corpus_v1 import (  # noqa: E402
    MIN_FRONTIER, MINTED_DATE, load_or_build_corpus_v1, select_dev_bench_slice_v1,
    select_frontier_slice_v1,
)
from coordpy.deployable_complexity_witness_v1 import (  # noqa: E402
    ACTION_REWRITE, ARM_D1_REWRITE, ARM_D2_GATED, ARM_D3_CONTROLLER, LADDER_VALUE_SEED,
    run_deployable_witness_arm_v1,
)
from coordpy.exact_oracle_witness_v1 import (  # noqa: E402
    ARM_C2_COMPLEXITY, build_witness_probe_set_v1, run_witness_arm_v1,
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
C0_WITNESS_SEED = 999_134
# LOCKED before any eval/frontier spend (from results/w134/corpus/corpus_build_v1.json, the $0
# Lane-alpha build at timeout_s=8.0 that predates all beta NIM). Asserted on eval/frontier.
LOCKED_CORPUS_CID = "191d995487d6cb09db6dba7683413661c69b1cefa82036a3fc339d5b0bb54a55"
LOCKED_EVAL_SPLIT_CID_PREFIX = "748dd6faa8a82b80"
LOCKED_FRONTIER_SLICE_CID_PREFIX = "31a813041e333dc8"
CORPUS_CACHE = ROOT / "results" / "w134" / "corpus" / "corpus_cache.pkl"
SPLIT_SEEDS = {"dev": 134_201, "eval": 134_202, "frontier": 134_203}
DEPLOYABLE_ARMS = (ARM_D1_REWRITE, ARM_D2_GATED, ARM_D3_CONTROLLER)
C0_ARM = "C0"
LATENCY_THROTTLE_S = 12.0   # RUNBOOK §9: median > this => dev_bench_per_family drops to 1


def _arm_report(model_id, base_seed_rep, arm_outcomes, arm_id, K):
    """Wrap an arm's outcomes into an IcpcBenchReportV1 with A0/A1 from the base bench and
    B := the arm, so the W108 evaluator scores 'arm - A1' identically to 'B - A1'."""
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
        question_ids=qids, seed_merkle_root=f"w134_arm_{arm_id}_{base_seed_rep.seed}")
    return IcpcBenchReportV1(
        schema=W120_ICPC_REFLEXION_BENCH_V1_SCHEMA_VERSION, model_id=str(model_id),
        n_problems=base_seed_rep.n_problems, n_seeds=1, K_multi_sample=int(K),
        per_seed=(seed_rep,), a0_mean_pass_at_1=base_seed_rep.a0_pass_at_1,
        a1_mean_pass_at_1=base_seed_rep.a1_pass_at_1, b_mean_pass_at_1=float(b_acc),
        b_mean_minus_a1_mean_pp=float((b_acc - base_seed_rep.a1_pass_at_1) * 100.0),
        bench_merkle_root=f"w134_arm_{arm_id}")


def _rescues_vs(per_b, per_ref, qids, fams):
    return [(q, fams.get(q, "?")) for i, q in enumerate(qids) if per_b[i] and not per_ref[i]]


def _select(mode, per_family):
    corpus = load_or_build_corpus_v1(CORPUS_CACHE, expect_corpus_cid=LOCKED_CORPUS_CID)
    if corpus.corpus_cid() != LOCKED_CORPUS_CID:
        raise SystemExit(f"corpus CID {corpus.corpus_cid()[:16]} != locked {LOCKED_CORPUS_CID[:16]}; refusing NIM.")
    if not corpus.meets_floors:
        raise SystemExit("corpus does not meet the 36/36/36/30 floors; refusing NIM.")
    if mode == "dev":
        probs = list(select_dev_bench_slice_v1(corpus.dev, per_family=per_family))
        return corpus, probs, "dev_bench", {p.problem_id: p.family for p in probs}
    if mode == "eval":
        if not corpus.eval.split_cid.startswith(LOCKED_EVAL_SPLIT_CID_PREFIX):
            raise SystemExit(f"eval split CID {corpus.eval.split_cid[:16]} != locked "
                             f"{LOCKED_EVAL_SPLIT_CID_PREFIX}; refusing eval NIM.")
        probs = list(select_frontier_slice_v1(corpus.eval, n=MIN_FRONTIER))  # family-balanced 30
        return corpus, probs, corpus.eval.split_cid, {p.problem_id: p.family for p in probs}
    # frontier
    fslice = select_frontier_slice_v1(corpus.frontier, n=MIN_FRONTIER)
    if not corpus.frontier_slice_cid.startswith(LOCKED_FRONTIER_SLICE_CID_PREFIX):
        raise SystemExit(f"frontier slice CID {corpus.frontier_slice_cid[:16]} != locked "
                         f"{LOCKED_FRONTIER_SLICE_CID_PREFIX}; refusing frontier NIM.")
    probs = list(fslice)
    return corpus, probs, corpus.frontier_slice_cid, {p.problem_id: p.family for p in probs}


def main() -> int:
    ap = argparse.ArgumentParser(description="W134 deployable-witness complexity bench")
    ap.add_argument("--mode", choices=["dev", "eval", "frontier"], required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--lead", choices=list(DEPLOYABLE_ARMS), default=ARM_D3_CONTROLLER)
    ap.add_argument("--per-family", type=int, default=2, help="dev bench instances/family (§9)")
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "w134"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    corpus, run_problems, slice_cid, fams = _select(args.mode, args.per_family)
    tmpl_by_id = corpus.template_by_problem_id()
    by_fam = {}
    for p in run_problems:
        by_fam[p.family] = by_fam.get(p.family, 0) + 1
    print(f"  mode={args.mode} n={len(run_problems)} families={len(by_fam)} "
          f"slice_cid={str(slice_cid)[:16]} model={args.model}")
    if args.dry_run:
        print(f"  --dry-run: validated {len(run_problems)} problems "
              f"(corpus_cid={corpus.corpus_cid()[:16]}); stopping before NIM.")
        return 0

    arms = ([C0_ARM] + list(DEPLOYABLE_ARMS)) if args.mode == "dev" else [C0_ARM, args.lead]
    seed = SPLIT_SEEDS[args.mode]
    run_id = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_model = str(args.model).replace("/", "_")
    out_dir = Path(args.out_dir) / args.mode / f"w134_{args.mode}_{safe_model}_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    sidecar_f = open(out_dir / "calls.jsonl", "w")

    def sidecar_writer(rec):
        sidecar_f.write(json.dumps(rec, separators=(",", ":")) + "\n")
        sidecar_f.flush()

    gen = _build_nim_gen(model=str(args.model), sidecar_writer=sidecar_writer)
    cfg = IcpcBenchConfigV1(K_multi_sample=5, seeds=(int(seed),), sampling_temperature=0.7,
                            max_tokens_per_call=int(args.max_tokens), executor_timeout_s=EXEC_TIMEOUT_S)
    pilot_subset = [p.to_pilot_problem(minted_date=MINTED_DATE) for p in run_problems]

    t0 = time.time()
    print("  [1] A0/A1/B0 (validated blind stack) ...")
    base = run_icpc_reflexion_bench_v1(
        gen=gen, model_id=str(args.model), subset=pilot_subset, config=cfg,
        on_problem_start=lambda s, i, q: print(f"    base {i+1}/{len(pilot_subset)} {q}", flush=True))
    base_seed_rep = base.per_seed[0]

    print("  [2] C0 exact-oracle EW2 probe sets ($0 NIM) ...")
    probes = {p.problem_id: build_witness_probe_set_v1(tmpl_by_id[p.problem_id], p,
              witness_seed=C0_WITNESS_SEED, timeout_s=WITNESS_TIMEOUT_S) for p in run_problems}

    arm_reports, arm_gates, arm_traces = {}, {}, {}
    for arm in arms:
        print(f"  [3] arm {arm} ...")
        outcomes, traces = [], []
        for i, p in enumerate(run_problems):
            if arm == C0_ARM:
                o, tr = run_witness_arm_v1(
                    seed=int(seed), template=tmpl_by_id[p.problem_id], problem=p,
                    probe=probes[p.problem_id], gen=gen, K=5, temperature=0.7,
                    max_tokens=int(args.max_tokens), timeout_s=EXEC_TIMEOUT_S,
                    arm=ARM_C2_COMPLEXITY, minted_date=MINTED_DATE, witness_timeout_s=WITNESS_TIMEOUT_S)
            else:
                o, tr = run_deployable_witness_arm_v1(
                    seed=int(seed), pilot=p.to_pilot_problem(minted_date=MINTED_DATE), gen=gen, K=5,
                    temperature=0.7, max_tokens=int(args.max_tokens), timeout_s=EXEC_TIMEOUT_S,
                    arm=arm, ladder_seed=LADDER_VALUE_SEED, witness_timeout_s=WITNESS_TIMEOUT_S)
            outcomes.append(o)
            traces.append(tr.to_dict())
            print(f"    {arm} {i+1}/{len(run_problems)} {p.problem_id} passed={o.final_passed}", flush=True)
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
    c0_pct = arm_gates[C0_ARM]["phase2"]["b_pct"]

    arm_summ = {}
    for arm in arms:
        g = arm_gates[arm]["phase2"]
        bp = arm_gates[arm]["per_problem_passed"]
        r_a1 = _rescues_vs(bp, a1, qids, fams)
        r_b0 = _rescues_vs(bp, b0, qids, fams)
        fams_b0 = sorted({f for _, f in r_b0})
        # §7b algorithmic-rescue audit: a D-arm rescue counts iff that attempt emitted a REWRITE
        tr_by_q = {t["problem_id"]: t for t in arm_traces[arm]}
        algo = [q for q, _ in r_b0 if (ACTION_REWRITE in tr_by_q.get(q, {}).get("actions", [])
                                       or arm == C0_ARM)]
        formatting_only = bool(r_b0 and not algo)
        arm_summ[arm] = {
            "arm_pct": g["b_pct"], "arm_minus_a1_pp": g["b_minus_a1_pp"],
            "arm_minus_b0_pp": round(g["b_pct"] - base_gates["b_pct"], 4),
            "c0_minus_arm_pp": round(c0_pct - g["b_pct"], 4),
            "rescues_vs_b0": r_b0, "n_families_rescued_vs_b0": len(fams_b0),
            "families_rescued_vs_b0": fams_b0, "algorithmic_rescues_vs_b0": algo,
            "formatting_only": formatting_only,
            "dev_gate_pass": bool(arm != C0_ARM
                                  and round(g["b_pct"] - base_gates["b_pct"], 4) >= 3.33
                                  and len(fams_b0) >= 2 and (c0_pct - g["b_pct"]) <= 3.33),
            "eval_earn_pass": bool(arm != C0_ARM and g["b_minus_a1_pp"] >= 0.0
                                   and round(g["b_pct"] - base_gates["b_pct"], 4) >= 5.0
                                   and (c0_pct - g["b_pct"]) <= 2.0 and len(fams_b0) >= 3
                                   and not formatting_only),
            "mlb": arm_gates[arm]["mlb"], "phase2": g}

    payload = {"schema": "coordpy.w134_deployable_witness_bench_v1", "mode": args.mode,
               "model_id": args.model, "seed": seed, "n_problems": len(run_problems),
               "slice_cid": slice_cid, "corpus_cid": corpus.corpus_cid(), "by_family": by_fam,
               "wall_s": round(wall, 2), "minted_date": MINTED_DATE, "question_ids": qids,
               "families": fams,
               "base_A0_A1_B0": {"a0_pct": base_gates["a0_pct"], "a1_pct": base_gates["a1_pct"],
                                 "b0_pct": base_gates["b_pct"], "b0_minus_a1_pp": base_gates["b_minus_a1_pp"],
                                 "per_problem_a1": a1, "per_problem_b0": b0,
                                 "mlb": _mlb_rates(base), "phase2": base_gates},
               "c0_exact_oracle_pct": c0_pct, "arms": arm_summ, "arm_traces": arm_traces}
    (out_dir / f"w134_{args.mode}_report.json").write_text(json.dumps(payload, indent=2, default=str))
    (Path(args.out_dir) / args.mode / f"latest_{args.mode}.txt").write_text(out_dir.name + "\n")

    print(f"\n  WALL {wall:.1f}s  A0={base_gates['a0_pct']:.2f} A1={base_gates['a1_pct']:.2f} "
          f"B0={base_gates['b_pct']:.2f} (B0-A1={base_gates['b_minus_a1_pp']:+.2f}pp)  "
          f"C0={c0_pct:.2f} (C0-B0={c0_pct-base_gates['b_pct']:+.2f}pp)")
    for arm in arms:
        s = arm_summ[arm]
        print(f"  {arm}: {s['arm_pct']:.2f}%  (arm-A1={s['arm_minus_a1_pp']:+.2f} "
              f"arm-B0={s['arm_minus_b0_pp']:+.2f} C0-arm={s['c0_minus_arm_pp']:+.2f})  "
              f"rescues_vs_B0={len(s['rescues_vs_b0'])} in {s['n_families_rescued_vs_b0']} fams "
              f"MLB-2 {s['mlb']['mlb2_rescue_rate']*100:.1f}%  dev_gate={s['dev_gate_pass']} "
              f"eval_earn={s['eval_earn_pass']}")
    print(f"  out_dir: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
