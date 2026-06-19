#!/usr/bin/env python3
"""W136-beta — held-out MACHINE-STRUCTURED-TRACE mechanism bench (dev / eval / frontier).

Arms (RUNBOOK_W136 §6), all same-budget K=5, each scored in the "B" slot so "arm - A1" is byte-identical
to "B - A1" (the verbatim W108 ``_mlb_rates`` / ``_evaluate_phase2_gates`` that scored W89/W105/W120/
W132/W133/W134/W135):

  A0/A1/B0   — the validated blind W120/W132 stack (``run_icpc_reflexion_bench_v1``)
  C1         — the exact-oracle EW1 COUNTEREXAMPLE witness (W133)
  S4         — the W135 PROSE solution-structure controller (the flat optimal-only ladder); the lever
               T must beat by >= +5pp (the W136 thesis: machine-structured state > prose structure)
  T1         — the machine-structured algorithm-state TRACE rewrite (LEAD; full dual-trajectory capsule)
  T2         — the forward-only trace-conditioned CONTROLLER (staged; routes capsule vs counterexample)

DEV spend discipline (RUNBOOK §7a/§9): the dev bench REUSES the W135 dev problems + their already-paid
A0/A1/B0/C1/S4 baselines (all 81.25%, the SAME 3 capability-bound traps), so NIM is spent ONLY on the
genuinely-new trace arms (T1 lead; T2 staged on a T1 crack) — the sharpest, cheapest "machine-structured
state vs prose" contrast.  EVAL / FRONTIER (on the fresh, locked W136 slices) run the FULL fresh baseline
stack (airtight) and run ONLY if the prior gate cleared; the locked W136 eval/frontier CIDs are asserted
before spend.  Requires ``NVIDIA_API_KEY``.

    python scripts/run_w136_trace_bench_v1.py --mode dev --dry-run        # 0 NIM (validate reuse + probes)
    python scripts/run_w136_trace_bench_v1.py --mode dev                  # T1 (+T2 if --arms full)
    python scripts/run_w136_trace_bench_v1.py --mode eval                 # fresh A0/A1/B0/C1/S4/T1
    python scripts/run_w136_trace_bench_v1.py --mode frontier
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import sys  # noqa: E402
sys.path.insert(0, str(ROOT))

from coordpy.algorithm_state_trace_corpus_v1 import (  # noqa: E402
    MIN_FRONTIER, MINTED_DATE as W136_MINTED_DATE, load_or_build_algorithm_state_corpus_v1,
    select_dev_bench_slice_v1, select_frontier_slice_v1,
)
from coordpy.noncomplexity_structure_corpus_v1 import (  # noqa: E402
    MINTED_DATE as W135_MINTED_DATE, load_corpus_v1,
    select_dev_bench_slice_v1 as w135_dev_slice,
)
from coordpy.algorithm_state_trace_v1 import (  # noqa: E402
    ARM_T1_TRACE_REWRITE, ARM_T2_TRACE_CONTROLLER, run_trace_arm_v1,
)
from coordpy.solution_structure_witness_v1 import ARM_S4_CONTROLLER, run_structure_witness_arm_v1  # noqa: E402
from coordpy.exact_oracle_witness_v1 import (  # noqa: E402
    ARM_C1_COUNTEREXAMPLE, build_witness_probe_set_v1, run_witness_arm_v1,
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
C1_WITNESS_SEED = 999_135          # same probe seed as W135 (clean reuse of the W135 baselines on dev)
T_WITNESS_SEED = 999_136
SPLIT_SEEDS = {"dev": 136_201, "eval": 136_202, "frontier": 136_203}
LATENCY_THROTTLE_S = 12.0

# LOCKED before any eval/frontier spend (filled from results/w136/corpus/corpus_build_selftest_v1.json,
# the $0 Lane-α build predating all β NIM). Asserted on eval/frontier.
LOCKED_CORPUS_CID = "ce1a6bc6541250ee98dd97be631c02da957734844c06c53c67412ad68b31a68a"
LOCKED_EVAL_SPLIT_CID_PREFIX = "135193533d6b5733"
LOCKED_FRONTIER_SLICE_CID_PREFIX = "3f75b3020271ada8"
W136_CACHE = ROOT / "results" / "w136" / "corpus" / "corpus_cache.pkl"
W135_CACHE = ROOT / "results" / "w135" / "corpus" / "corpus_cache.pkl"


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
        question_ids=qids, seed_merkle_root=f"w136_arm_{arm_id}_{base_seed_rep.seed}")
    return IcpcBenchReportV1(
        schema=W120_ICPC_REFLEXION_BENCH_V1_SCHEMA_VERSION, model_id=str(model_id),
        n_problems=base_seed_rep.n_problems, n_seeds=1, K_multi_sample=int(K),
        per_seed=(seed_rep,), a0_mean_pass_at_1=base_seed_rep.a0_pass_at_1,
        a1_mean_pass_at_1=base_seed_rep.a1_pass_at_1, b_mean_pass_at_1=float(b_acc),
        b_mean_minus_a1_mean_pp=float((b_acc - base_seed_rep.a1_pass_at_1) * 100.0),
        bench_merkle_root=f"w136_arm_{arm_id}")


def _rescues_vs(per_b, per_ref, qids):
    return [q for i, q in enumerate(qids) if per_b[i] and not per_ref[i]]


def _load_w135_dev_baselines():
    """Reuse the W135 dev problems + their already-paid A0/A1/B0/C1/S4 baselines (all 81.25%, the SAME
    3 capability-bound traps). S4=C1=B0=A1 pass-set on this field (S4-C1=+0.00, 0 rescues), so the
    per-problem b0 vector IS the S4/C1 reference."""
    rep_paths = sorted(glob.glob(str(ROOT / "results" / "w135" / "dev" / "**" / "w135_dev_report.json"),
                                 recursive=True))
    if not rep_paths:
        raise SystemExit("W135 dev report not found; cannot reuse baselines.")
    d = json.load(open(rep_paths[-1]))
    base = d["base_A0_A1_B0"]
    return d, base


def _select(mode):
    if mode == "dev":
        w135 = load_corpus_v1(W135_CACHE)
        if w135 is None:
            raise SystemExit("W135 corpus cache missing; cannot reuse the W135 dev problems.")
        probs = list(w135_dev_slice(w135.dev, per_family=1))
        tmpl = w135.template_by_problem_id()
        mode_by_id = w135.mode_by_problem_id()
        return probs, {p.problem_id: p.family for p in probs}, mode_by_id, tmpl, "w135_dev_bench", W135_MINTED_DATE
    corpus = load_or_build_algorithm_state_corpus_v1(W136_CACHE, expect_corpus_cid=None)
    if LOCKED_CORPUS_CID != "__FILL_FROM_ALPHA_BUILD__" and corpus.corpus_cid() != LOCKED_CORPUS_CID:
        raise SystemExit(f"corpus CID {corpus.corpus_cid()[:16]} != locked; refusing NIM.")
    if not corpus.meets_floors:
        raise SystemExit("W136 corpus does not meet the 36/36/36/30 floors; refusing NIM.")
    mode_by_id = corpus.mode_by_problem_id()
    tmpl = corpus.template_by_problem_id()
    if mode == "eval":
        if LOCKED_EVAL_SPLIT_CID_PREFIX != "__FILL__" and not corpus.eval.split_cid.startswith(LOCKED_EVAL_SPLIT_CID_PREFIX):
            raise SystemExit("eval split CID != locked; refusing eval NIM.")
        probs = list(select_frontier_slice_v1(corpus.eval, n=MIN_FRONTIER))
        return probs, {p.problem_id: p.family for p in probs}, mode_by_id, tmpl, corpus.eval.split_cid, W136_MINTED_DATE
    fslice = select_frontier_slice_v1(corpus.frontier, n=MIN_FRONTIER)
    if LOCKED_FRONTIER_SLICE_CID_PREFIX != "__FILL__" and not corpus.frontier_slice_cid.startswith(LOCKED_FRONTIER_SLICE_CID_PREFIX):
        raise SystemExit("frontier slice CID != locked; refusing frontier NIM.")
    return list(fslice), {p.problem_id: p.family for p in fslice}, mode_by_id, tmpl, corpus.frontier_slice_cid, W136_MINTED_DATE


def _run_trace(arm, probs, probes, tmpl, gen, seed, minted_date, max_tokens):
    outs, trs = [], []
    for i, p in enumerate(probs):
        o, tr = run_trace_arm_v1(
            seed=int(seed), template=tmpl[p.problem_id], problem=p, probe=probes[p.problem_id], gen=gen,
            K=5, temperature=0.7, max_tokens=int(max_tokens), timeout_s=EXEC_TIMEOUT_S, arm=arm,
            minted_date=minted_date, witness_timeout_s=WITNESS_TIMEOUT_S)
        outs.append(o)
        trs.append(tr.to_dict())
        print(f"    {arm} {i+1}/{len(probs)} {p.problem_id} passed={o.final_passed}", flush=True)
    return outs, trs


def main() -> int:
    ap = argparse.ArgumentParser(description="W136 algorithm-state-trace bench")
    ap.add_argument("--mode", choices=["dev", "eval", "frontier"], required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--arms", choices=["lead", "full"], default="lead",
                    help="dev: lead=T1 only (decisive); full=T1+T2")
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "w136"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    probs, fams, mode_by_id, tmpl, slice_cid, minted_date = _select(args.mode)
    by_mode = {}
    for p in probs:
        by_mode[p.mode] = by_mode.get(p.mode, 0) + 1
    print(f"  mode={args.mode} n={len(probs)} modes={by_mode} slice={str(slice_cid)[:24]} model={args.model}")
    if args.dry_run:
        print(f"  --dry-run: validated {len(probs)} problems; stopping before NIM.")
        return 0

    run_id = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_model = str(args.model).replace("/", "_")
    out_dir = Path(args.out_dir) / args.mode / f"w136_{args.mode}_{safe_model}_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    sidecar_f = open(out_dir / "calls.jsonl", "w")

    def sidecar_writer(rec):
        sidecar_f.write(json.dumps(rec, separators=(",", ":")) + "\n")
        sidecar_f.flush()

    gen = _build_nim_gen(model=str(args.model), sidecar_writer=sidecar_writer)
    seed = SPLIT_SEEDS[args.mode]
    t0 = time.time()

    print("  [1] witness/trace probe sets ($0 NIM) ...")
    probes = {p.problem_id: build_witness_probe_set_v1(tmpl[p.problem_id], p,
              witness_seed=T_WITNESS_SEED, timeout_s=WITNESS_TIMEOUT_S) for p in probs}

    if args.mode == "dev":
        # reuse the W135 A0/A1/B0/C1/S4 baselines (same problems); spend ONLY on the trace arms
        wd, base = _load_w135_dev_baselines()
        qids = list(wd["question_ids"])
        a1 = list(base["per_problem_a1"]); b0 = list(base["per_problem_b0"])
        # S4 = C1 = B0 pass set on this field (S4-C1=+0.00, 0 rescues; all 81.25%): per_problem_b0 is the ref
        s4 = list(b0); c1 = list(b0)
        a0_pct, a1_pct, b0_pct = base["a0_pct"], base["a1_pct"], base["b0_pct"]
        s4_pct = float(wd["arms"]["S4"]["arm_pct"]); c1_pct = float(wd["c1_counterexample_pct"])
        # order probs to match the reused qids
        by_id = {p.problem_id: p for p in probs}
        probs = [by_id[q] for q in qids if q in by_id]
        trace_arms = [ARM_T1_TRACE_REWRITE] + ([ARM_T2_TRACE_CONTROLLER] if args.arms == "full" else [])
        arm_results, arm_traces = {}, {}
        for arm in trace_arms:
            print(f"  [2] trace arm {arm} (fresh; reusing baselines) ...")
            outs, trs = _run_trace(arm, probs, probes, tmpl, gen, seed, minted_date, args.max_tokens)
            tp = [bool(o.final_passed) for o in outs]
            pct = 100.0 * sum(tp) / (len(tp) or 1)
            tr_by_q = {t["problem_id"]: t for t in trs}
            r_b0 = _rescues_vs(tp, b0, qids); r_s4 = _rescues_vs(tp, s4, qids); r_c1 = _rescues_vs(tp, c1, qids)
            regress = [qids[i] for i in range(len(qids)) if b0[i] and not tp[i]]
            modes_b0 = sorted({mode_by_id.get(q, "?") for q in r_b0})
            fams_b0 = sorted({fams.get(q, "?") for q in r_b0})
            structural = [q for q in r_b0 if tr_by_q.get(q, {}).get("any_genuinely_new", False)]
            span_ok = bool(len(modes_b0) >= 2 or len(fams_b0) >= 3)
            arm_results[arm] = {
                "arm_pct": round(pct, 4), "arm_minus_a1_pp": round(pct - a1_pct, 4),
                "arm_minus_b0_pp": round(pct - b0_pct, 4), "arm_minus_s4_pp": round(pct - s4_pct, 4),
                "arm_minus_c1_pp": round(pct - c1_pct, 4),
                "rescues_vs_b0": r_b0, "rescues_vs_s4": r_s4, "rescues_vs_c1": r_c1,
                "regressions_vs_b0": regress, "modes_rescued_vs_b0": modes_b0,
                "n_modes_rescued_vs_b0": len(modes_b0), "families_rescued_vs_b0": fams_b0,
                "n_families_rescued_vs_b0": len(fams_b0), "structural_rescues": structural,
                "per_problem_passed": tp,
                # §7a DEV gate: T beats B0 >= +3.33 AND beats S4 >= +3.33 AND span >=2 modes / >=3 fams
                "dev_gate_pass": bool(round(pct - b0_pct, 4) >= 3.33 and round(pct - s4_pct, 4) >= 3.33
                                      and span_ok and len(structural) == len(r_b0) and r_b0),
            }
            arm_traces[arm] = trs
        sidecar_f.close()
        wall = time.time() - t0
        payload = {"schema": "coordpy.w136_trace_bench_v1", "mode": "dev", "model_id": args.model,
                   "seed": seed, "n_problems": len(probs), "slice_cid": slice_cid,
                   "reused_w135_baselines": True, "question_ids": qids, "families": fams,
                   "modes": {q: mode_by_id.get(q, "?") for q in qids}, "wall_s": round(wall, 2),
                   "baselines": {"a0_pct": a0_pct, "a1_pct": a1_pct, "b0_pct": b0_pct,
                                 "c1_pct": c1_pct, "s4_pct": s4_pct,
                                 "per_problem_a1": a1, "per_problem_b0": b0},
                   "arms": arm_results, "arm_traces": arm_traces}
        (out_dir / "w136_dev_report.json").write_text(json.dumps(payload, indent=2, default=str))
        (Path(args.out_dir) / "dev" / "latest_dev.txt").write_text(out_dir.name + "\n")
        print(f"\n  WALL {wall:.1f}s  (reused) A0={a0_pct:.2f} A1={a1_pct:.2f} B0={b0_pct:.2f} "
              f"C1={c1_pct:.2f} S4={s4_pct:.2f}")
        for arm in trace_arms:
            s = arm_results[arm]
            print(f"  {arm}: {s['arm_pct']:.2f}% (a-A1={s['arm_minus_a1_pp']:+.2f} a-B0={s['arm_minus_b0_pp']:+.2f} "
                  f"a-S4={s['arm_minus_s4_pp']:+.2f})  resc_vs_B0={len(s['rescues_vs_b0'])} "
                  f"({s['n_modes_rescued_vs_b0']}m/{s['n_families_rescued_vs_b0']}f struct={len(s['structural_rescues'])}) "
                  f"regress={len(s['regressions_vs_b0'])}  DEV_GATE={s['dev_gate_pass']}")
        print(f"  out_dir: {out_dir}")
        return 0

    # ---- eval / frontier: full fresh baseline stack + C1 + S4 + T1 (airtight) ----
    cfg = IcpcBenchConfigV1(K_multi_sample=5, seeds=(int(seed),), sampling_temperature=0.7,
                            max_tokens_per_call=int(args.max_tokens), executor_timeout_s=EXEC_TIMEOUT_S)
    pilot_subset = [p.to_pilot_problem(minted_date=minted_date) for p in probs]
    print("  [2] A0/A1/B0 (validated blind stack) ...")
    base = run_icpc_reflexion_bench_v1(gen=gen, model_id=str(args.model), subset=pilot_subset, config=cfg,
                                       on_problem_start=lambda s, i, q: print(f"    base {i+1}/{len(pilot_subset)} {q}", flush=True))
    bsr = base.per_seed[0]
    base_gates = _evaluate_phase2_gates(report=base, mlb=_mlb_rates(base))
    qids = list(bsr.question_ids); a1 = list(bsr.per_problem_a1_passed); b0 = list(bsr.per_problem_b_passed)

    arm_results, arm_traces = {}, {}
    # C1
    print("  [3] C1 ..."); c1_o = [run_witness_arm_v1(seed=seed, template=tmpl[p.problem_id], problem=p,
            probe=probes[p.problem_id], gen=gen, K=5, temperature=0.7, max_tokens=args.max_tokens,
            timeout_s=EXEC_TIMEOUT_S, arm=ARM_C1_COUNTEREXAMPLE, minted_date=minted_date,
            witness_timeout_s=WITNESS_TIMEOUT_S)[0] for p in probs]
    c1 = [bool(o.final_passed) for o in c1_o]; c1_pct = 100.0 * sum(c1) / (len(c1) or 1)
    # S4
    print("  [4] S4 ..."); s4_o = [run_structure_witness_arm_v1(seed=seed, template=tmpl[p.problem_id], problem=p,
            probe=probes[p.problem_id], gen=gen, K=5, temperature=0.7, max_tokens=args.max_tokens,
            timeout_s=EXEC_TIMEOUT_S, arm=ARM_S4_CONTROLLER, minted_date=minted_date,
            witness_timeout_s=WITNESS_TIMEOUT_S)[0] for p in probs]
    s4 = [bool(o.final_passed) for o in s4_o]; s4_pct = 100.0 * sum(s4) / (len(s4) or 1)
    # T1 (lead)
    print("  [5] T1 (lead) ..."); t1_o, t1_tr = _run_trace(ARM_T1_TRACE_REWRITE, probs, probes, tmpl, gen, seed, minted_date, args.max_tokens)
    tp = [bool(o.final_passed) for o in t1_o]; t1_pct = 100.0 * sum(tp) / (len(tp) or 1)
    tr_by_q = {t["problem_id"]: t for t in t1_tr}
    r_b0 = _rescues_vs(tp, b0, qids); r_s4 = _rescues_vs(tp, s4, qids)
    modes_b0 = sorted({mode_by_id.get(q, "?") for q in r_b0}); fams_b0 = sorted({fams.get(q, "?") for q in r_b0})
    structural = [q for q in r_b0 if tr_by_q.get(q, {}).get("any_genuinely_new", False)]
    span_ok = bool(len(modes_b0) >= 2 or len(fams_b0) >= 3)
    earn = bool(round(t1_pct - base_gates["b_pct"], 4) >= 5.0 and round(t1_pct - s4_pct, 4) >= 5.0
                and span_ok and len(structural) == len(r_b0) and r_b0)
    sidecar_f.close(); wall = time.time() - t0
    payload = {"schema": "coordpy.w136_trace_bench_v1", "mode": args.mode, "model_id": args.model,
               "seed": seed, "n_problems": len(probs), "slice_cid": slice_cid, "wall_s": round(wall, 2),
               "question_ids": qids, "families": fams, "modes": {q: mode_by_id.get(q, "?") for q in qids},
               "baselines": {"a0_pct": base_gates["a0_pct"], "a1_pct": base_gates["a1_pct"],
                             "b0_pct": base_gates["b_pct"], "c1_pct": c1_pct, "s4_pct": s4_pct},
               "lead_T1": {"arm_pct": round(t1_pct, 4), "arm_minus_a1_pp": round(t1_pct - base_gates["a1_pct"], 4),
                           "arm_minus_b0_pp": round(t1_pct - base_gates["b_pct"], 4),
                           "arm_minus_s4_pp": round(t1_pct - s4_pct, 4),
                           "rescues_vs_b0": r_b0, "rescues_vs_s4": r_s4, "structural_rescues": structural,
                           "n_modes_rescued_vs_b0": len(modes_b0), "n_families_rescued_vs_b0": len(fams_b0),
                           "per_problem_passed": tp,
                           "earn_pass": earn if args.mode == "eval" else None,
                           "frontier_pass": (bool(round(t1_pct - base_gates["a1_pct"], 4) >= 5.0)
                                             if args.mode == "frontier" else None)},
               "arm_traces": {"T1": t1_tr}}
    (out_dir / f"w136_{args.mode}_report.json").write_text(json.dumps(payload, indent=2, default=str))
    (Path(args.out_dir) / args.mode / f"latest_{args.mode}.txt").write_text(out_dir.name + "\n")
    print(f"\n  WALL {wall:.1f}s A0={base_gates['a0_pct']:.2f} A1={base_gates['a1_pct']:.2f} "
          f"B0={base_gates['b_pct']:.2f} C1={c1_pct:.2f} S4={s4_pct:.2f}  T1={t1_pct:.2f} "
          f"(T1-B0={t1_pct-base_gates['b_pct']:+.2f} T1-S4={t1_pct-s4_pct:+.2f})  "
          f"earn={payload['lead_T1']['earn_pass']} frontier={payload['lead_T1']['frontier_pass']}")
    print(f"  out_dir: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
