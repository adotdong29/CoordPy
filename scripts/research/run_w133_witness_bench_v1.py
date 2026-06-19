#!/usr/bin/env python3
"""W133-beta — held-out witness-guided mechanism bench (dev / eval / frontier).

Reuses the *already-validated* W120 reflexion bench for A0 / A1 / B0 (the blind W132 stack)
and the verbatim W108 evaluator (`_mlb_rates` / `_evaluate_phase2_gates` — the SAME code that
scored W89 / W105 / W120 / W132).  The witness arms C1 / C2 / C3 are run by
`run_witness_arm_v1` (a strict same-budget swap of the B-arm feedback object) and scored by
placing each arm in the "B" slot, so "C - A1" is computed byte-identically to "B - A1".

Modes (RUNBOOK W133 § 1 branch order)::

    python scripts/run_w133_witness_bench_v1.py --mode dev   --dry-run     # 0 NIM
    python scripts/run_w133_witness_bench_v1.py --mode dev                 # full slate on DEV
    python scripts/run_w133_witness_bench_v1.py --mode eval --lead C3      # lead+baselines on EVAL
    python scripts/run_w133_witness_bench_v1.py --mode frontier --lead C3  # lead+baselines, W132 anchor

Requires `NVIDIA_API_KEY`.  Spend discipline (RUNBOOK § 7/§ 8): dev is the go/no-go; eval runs
only if the dev gate clears; frontier runs only if the eval earn rule passes.  The eval slice
CID is asserted before eval spend; the frontier core-slice CID is asserted before frontier spend.
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

from coordpy.resistant_by_construction_slate_v1 import RBC_SLATE_V1  # noqa: E402
from coordpy.resistant_by_construction_battlefield_v1 import (  # noqa: E402
    core_slice_cid_v1, mint_battlefield_v1, select_core_slice_v1,
)
from coordpy.witness_curriculum_corpus_v1 import (  # noqa: E402
    DEV_SEED, EVAL_SEED, MINTED_DATE, TRAIN_SEED, build_curriculum_v1,
)
from coordpy.exact_oracle_witness_v1 import (  # noqa: E402
    ARM_C1_COUNTEREXAMPLE, ARM_C2_COMPLEXITY, ARM_C3_CONTROLLER,
    OBS_RUNTIME_ERROR, OBS_TIMEOUT, OBS_WRONG_ANSWER,
    build_witness_probe_set_v1, run_witness_arm_v1,
)
from coordpy.icpc_reflexion_bench_v1 import (  # noqa: E402
    IcpcBenchConfigV1, IcpcBenchReportV1, IcpcSeedReportV1,
    W120_ICPC_REFLEXION_BENCH_V1_SCHEMA_VERSION, run_icpc_reflexion_bench_v1,
)
from coordpy.coordpy_icpc_battlefield_v1 import (  # noqa: E402
    ICPC_BATTLEFIELD_LISTING_SNAPSHOT_V1,
)
from scripts.run_w108_livecodebench_pilot import (  # noqa: E402
    _build_nim_gen, _evaluate_phase2_gates, _mlb_rates,
)

OFFICIAL_IDENTITIES = tuple(sorted({row[1] for row in ICPC_BATTLEFIELD_LISTING_SNAPSHOT_V1}))
DEFAULT_MODEL = "meta/llama-3.3-70b-instruct"
WITNESS_SEED = 999_133
EXEC_TIMEOUT_S = 8.0
WITNESS_TIMEOUT_S = 2.0
W132_ANCHOR_SEED = 132
W132_CORE_SLICE_CID_PREFIX = "f6a2ebed3da2f13b"
# LOCKED before any eval spend (from results/w133/curriculum/curriculum_build_v1.json, the $0
# Lane-alpha build that predates all beta NIM): the eval split is held out from mechanism design.
LOCKED_EVAL_SPLIT_CID_PREFIX = "88b9b79a7389711b"
SPLIT_SEEDS = {"dev": 133_101, "eval": 133_102, "frontier": 132_001}
WITNESS_ARMS = (ARM_C1_COUNTEREXAMPLE, ARM_C2_COMPLEXITY, ARM_C3_CONTROLLER)


def _arm_report(model_id, base_seed_rep, arm_outcomes, arm_id, K):
    """Build an IcpcBenchReportV1 with A0/A1 from the base bench and B := the witness arm,
    so the W108 evaluator scores 'arm - A1' identically to 'B - A1'."""
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
        question_ids=qids, seed_merkle_root=f"w133_arm_{arm_id}_{base_seed_rep.seed}")
    return IcpcBenchReportV1(
        schema=W120_ICPC_REFLEXION_BENCH_V1_SCHEMA_VERSION, model_id=str(model_id),
        n_problems=base_seed_rep.n_problems, n_seeds=1, K_multi_sample=int(K),
        per_seed=(seed_rep,), a0_mean_pass_at_1=base_seed_rep.a0_pass_at_1,
        a1_mean_pass_at_1=base_seed_rep.a1_pass_at_1, b_mean_pass_at_1=float(b_acc),
        b_mean_minus_a1_mean_pp=float((b_acc - base_seed_rep.a1_pass_at_1) * 100.0),
        bench_merkle_root=f"w133_arm_{arm_id}")


def _rescues_vs(per_b, per_ref, qids, modes):
    """problems the arm passes that the reference (A1 or B0) fails, with their modes."""
    out = []
    for i, q in enumerate(qids):
        if per_b[i] and not per_ref[i]:
            out.append((q, modes.get(q, "?")))
    return out


def _select_problems(mode, model):
    if mode == "frontier":
        bf = mint_battlefield_v1(RBC_SLATE_V1, global_seed=W132_ANCHOR_SEED,
                                 minted_date=MINTED_DATE, timeout_s=EXEC_TIMEOUT_S,
                                 official_identities=OFFICIAL_IDENTITIES)
        core = select_core_slice_v1(bf, n_problems=30)
        cid = core_slice_cid_v1(core)
        if not cid.startswith(W132_CORE_SLICE_CID_PREFIX):
            raise SystemExit(f"frontier core slice CID {cid[:16]} != {W132_CORE_SLICE_CID_PREFIX}; refusing.")
        return list(core), cid, {p.problem_id: p.mode for p in core}
    cur = build_curriculum_v1(RBC_SLATE_V1, minted_date=MINTED_DATE, train_seed=TRAIN_SEED,
                              dev_seed=DEV_SEED, eval_seed=EVAL_SEED,
                              official_identities=OFFICIAL_IDENTITIES, timeout_s=EXEC_TIMEOUT_S)
    split = cur.dev if mode == "dev" else cur.eval
    if not cur.meets_min_per_split:
        raise SystemExit("curriculum does not meet >=32/split; refusing NIM.")
    if mode == "eval" and not split.split_cid.startswith(LOCKED_EVAL_SPLIT_CID_PREFIX):
        raise SystemExit(f"eval split CID {split.split_cid[:16]} != locked "
                         f"{LOCKED_EVAL_SPLIT_CID_PREFIX} (drift); refusing eval NIM.")
    probs = list(split.problems())
    return probs, split.split_cid, {p.problem_id: p.mode for p in probs}


def main() -> int:
    ap = argparse.ArgumentParser(description="W133 witness-guided mechanism bench")
    ap.add_argument("--mode", choices=["dev", "eval", "frontier"], required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--lead", choices=list(WITNESS_ARMS), default=ARM_C3_CONTROLLER,
                    help="witness arm to run on eval/frontier (the dev-selected lead)")
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "w133"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    run_problems, slice_cid, modes = _select_problems(args.mode, args.model)
    arms = list(WITNESS_ARMS) if args.mode == "dev" else [args.lead]
    seed = SPLIT_SEEDS[args.mode]
    by_mode = {}
    for p in run_problems:
        by_mode[p.mode] = by_mode.get(p.mode, 0) + 1
    print(f"  mode={args.mode} n={len(run_problems)} modes={by_mode} slice_cid={slice_cid[:16]} "
          f"witness_arms={arms} model={args.model}")
    if args.dry_run:
        print(f"  --dry-run: validated {len(run_problems)} problems; stopping before NIM.")
        return 0

    templates_by_id = {f"rbc_{t.name}": t for t in RBC_SLATE_V1}
    pilot_subset = [p.to_pilot_problem(minted_date=MINTED_DATE) for p in run_problems]

    run_id = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_model = str(args.model).replace("/", "_")
    out_dir = Path(args.out_dir) / args.mode / f"w133_{args.mode}_{safe_model}_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    sidecar_f = open(out_dir / "calls.jsonl", "w")

    def sidecar_writer(rec):
        sidecar_f.write(json.dumps(rec, separators=(",", ":")) + "\n")
        sidecar_f.flush()

    gen = _build_nim_gen(model=str(args.model), sidecar_writer=sidecar_writer)
    cfg = IcpcBenchConfigV1(K_multi_sample=5, seeds=(int(seed),), sampling_temperature=0.7,
                            max_tokens_per_call=int(args.max_tokens),
                            executor_timeout_s=EXEC_TIMEOUT_S)
    t0 = time.time()
    print("  [1] A0/A1/B0 (validated blind stack) ...")
    base = run_icpc_reflexion_bench_v1(
        gen=gen, model_id=str(args.model), subset=pilot_subset, config=cfg,
        on_problem_start=lambda s, i, q: print(f"    base seed={s} {i+1}/{len(pilot_subset)} {q}", flush=True))
    base_seed_rep = base.per_seed[0]

    print("  [2] building witness probe sets (deterministic, $0 NIM) ...")
    probes = {}
    for p in run_problems:
        probes[p.problem_id] = build_witness_probe_set_v1(
            templates_by_id[p.problem_id], p, witness_seed=WITNESS_SEED, timeout_s=WITNESS_TIMEOUT_S)

    arm_reports = {}
    arm_gates = {}
    arm_traces = {}
    for arm in arms:
        print(f"  [3] witness arm {arm} ...")
        outcomes = []
        traces = []
        for i, p in enumerate(run_problems):
            o, tr = run_witness_arm_v1(
                seed=int(seed), template=templates_by_id[p.problem_id], problem=p,
                probe=probes[p.problem_id], gen=gen, K=5, temperature=0.7,
                max_tokens=int(args.max_tokens), timeout_s=EXEC_TIMEOUT_S, arm=arm,
                minted_date=MINTED_DATE, witness_timeout_s=WITNESS_TIMEOUT_S)
            outcomes.append(o)
            traces.append(tr)
            print(f"    {arm} {i+1}/{len(run_problems)} {p.problem_id} passed={o.final_passed} "
                  f"first_idx={o.first_pass_attempt_idx}", flush=True)
        rep = _arm_report(args.model, base_seed_rep, outcomes, arm, K=5)
        mlb = _mlb_rates(rep)
        gates = _evaluate_phase2_gates(report=rep, mlb=mlb)
        arm_reports[arm] = rep
        arm_gates[arm] = {"mlb": mlb, "phase2": gates,
                          "per_problem_passed": [bool(o.final_passed) for o in outcomes],
                          "per_problem_first_idx": [int(o.first_pass_attempt_idx) for o in outcomes]}
        arm_traces[arm] = [t.to_dict() for t in traces]

    sidecar_f.close()
    wall_s = float(time.time() - t0)

    qids = list(base_seed_rep.question_ids)
    a0 = list(base_seed_rep.per_problem_a0_passed)
    a1 = list(base_seed_rep.per_problem_a1_passed)
    b0 = list(base_seed_rep.per_problem_b_passed)
    base_mlb = _mlb_rates(base)
    base_gates = _evaluate_phase2_gates(report=base, mlb=base_mlb)

    arm_summ = {}
    for arm in arms:
        g = arm_gates[arm]["phase2"]
        bp = arm_gates[arm]["per_problem_passed"]
        rescues_a1 = _rescues_vs(bp, a1, qids, modes)
        rescues_b0 = _rescues_vs(bp, b0, qids, modes)
        modes_vs_a1 = sorted({m for _, m in rescues_a1})
        modes_vs_b0 = sorted({m for _, m in rescues_b0})
        # §7b formatting-vs-algorithmic rescue audit (from the witness traces)
        tr_by_qid = {t["problem_id"]: t for t in arm_traces[arm]}
        algo_rescues = [q for q, _ in rescues_a1
                        if any(ok in (OBS_WRONG_ANSWER, OBS_TIMEOUT)
                               for ok in tr_by_qid.get(q, {}).get("witness_observed_kinds", []))]
        formatting_only = bool(rescues_a1 and not algo_rescues)
        # pre-committed §7b earn rule (eval) — recorded on every arm for transparency
        earn = bool(g["b_minus_a1_pp"] >= 5.0
                    and float(g["b_pct"] - base_gates["b_pct"]) > 3.33
                    and len(modes_vs_a1) >= 2 and not formatting_only)
        arm_summ[arm] = {
            "arm_pct": g["b_pct"], "arm_minus_a1_pp": g["b_minus_a1_pp"],
            "arm_minus_b0_pp": float(round(g["b_pct"] - base_gates["b_pct"], 4)),
            "rescues_vs_a1": rescues_a1, "rescues_vs_b0": rescues_b0,
            "n_modes_rescued_vs_a1": len(modes_vs_a1), "n_modes_rescued_vs_b0": len(modes_vs_b0),
            "modes_rescued_vs_a1": modes_vs_a1, "modes_rescued_vs_b0": modes_vs_b0,
            "algorithmic_rescues_vs_a1": algo_rescues, "formatting_only": formatting_only,
            "dev_gate_pass": bool(float(g["b_pct"] - base_gates["b_pct"]) >= 3.33
                                  and len(modes_vs_b0) >= 2),
            "eval_earn_rule_pass": earn,
            "mlb": arm_gates[arm]["mlb"], "phase2": g}

    payload = {
        "schema": "coordpy.w133_witness_bench_v1", "mode": args.mode, "model_id": args.model,
        "seed": seed, "n_problems": len(run_problems), "slice_cid": slice_cid,
        "by_mode": by_mode, "wall_s": round(wall_s, 2), "minted_date": MINTED_DATE,
        "question_ids": qids, "modes": modes,
        "base_A0_A1_B0": {"a0_pct": base_gates["a0_pct"], "a1_pct": base_gates["a1_pct"],
                          "b0_pct": base_gates["b_pct"], "b0_minus_a1_pp": base_gates["b_minus_a1_pp"],
                          "per_problem_a0": a0, "per_problem_a1": a1, "per_problem_b0": b0,
                          "mlb": base_mlb, "phase2": base_gates},
        "witness_arms": arm_summ, "witness_traces": arm_traces,
    }
    (out_dir / f"w133_{args.mode}_report.json").write_text(json.dumps(payload, indent=2, default=str))
    (Path(args.out_dir) / args.mode / f"latest_{args.mode}.txt").write_text(out_dir.name + "\n")

    print()
    print(f"  WALL {wall_s:.1f}s  A0={base_gates['a0_pct']:.2f}% A1={base_gates['a1_pct']:.2f}% "
          f"B0={base_gates['b_pct']:.2f}% (B0-A1={base_gates['b_minus_a1_pp']:+.2f}pp; "
          f"MLB-1 {base_mlb['mlb1_invocation_rate']*100:.1f}% MLB-2 {base_mlb['mlb2_rescue_rate']*100:.1f}%)")
    for arm in arms:
        s = arm_summ[arm]
        print(f"  {arm}: {s['arm_pct']:.2f}%  (arm-A1={s['arm_minus_a1_pp']:+.2f}pp  "
              f"arm-B0={s['arm_minus_b0_pp']:+.2f}pp)  rescues_vs_A1={len(s['rescues_vs_a1'])} "
              f"in {s['n_modes_rescued_vs_a1']} modes {s['modes_rescued_vs_a1']}  "
              f"MLB-2 {s['mlb']['mlb2_rescue_rate']*100:.1f}%")
    print(f"  out_dir: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
