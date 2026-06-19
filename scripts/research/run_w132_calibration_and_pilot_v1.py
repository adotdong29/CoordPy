#!/usr/bin/env python3
"""W132-β — Maverick calibration + earned pilot on the resistant-by-construction battlefield.

Reuses the *already-validated* W120 reflexion bench (A0 / A1 / B sequential reflexion,
``run_icpc_reflexion_bench_v1``) and the verbatim W108 evaluator (``_mlb_rates`` /
``_evaluate_phase2_gates``) — the SAME code that scored W89/W105/W120 — on the deterministic
core 30-slice of the CoordPy-minted battlefield.  The ONLY change vs the W120 pilot is the
corpus: minted resistant-by-construction problems instead of fetched official ICPC packages.

Modes::

    python scripts/run_w132_calibration_and_pilot_v1.py --dry-run        # 0 NIM
    python scripts/run_w132_calibration_and_pilot_v1.py --mode calibration   # ~66 NIM
    python scripts/run_w132_calibration_and_pilot_v1.py --mode pilot         # ~330 NIM

Requires ``NVIDIA_API_KEY``.  The pilot REFUSES to spend unless the battlefield earned
(>=30 admitted, Maverick resistance-certified, core slice CID matches) and — for the full
pilot — the calibration was non-degenerate-cleared.
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

from coordpy.resistant_by_construction_battlefield_v1 import (  # noqa: E402
    certify_resistance_v1,
    core_slice_cid_v1,
    mint_battlefield_v1,
    select_core_slice_v1,
)
from coordpy.resistant_by_construction_slate_v1 import RBC_SLATE_V1  # noqa: E402
from coordpy.icpc_reflexion_bench_v1 import (  # noqa: E402
    IcpcBenchConfigV1,
    run_icpc_reflexion_bench_v1,
)
from coordpy.coordpy_icpc_battlefield_v1 import (  # noqa: E402
    ICPC_BATTLEFIELD_LISTING_SNAPSHOT_V1,
)
from scripts.run_w108_livecodebench_pilot import (  # noqa: E402
    _build_nim_gen,
    _evaluate_phase2_gates,
    _mlb_rates,
)

W132_TARGET_MODEL = "meta/llama-4-maverick-17b-128e-instruct"
MINTED_DATE = "2026-06-02"
GLOBAL_SEED = 132
EXPECTED_CORE_SLICE_CID = "f6a2ebed3da2f13b"   # prefix; full asserted below
EXEC_TIMEOUT_S = 8.0
OFFICIAL_IDENTITIES = tuple(sorted({row[1] for row in ICPC_BATTLEFIELD_LISTING_SNAPSHOT_V1}))
MODE_ORDER = ("COMPLEXITY_BLIND", "HIDDEN_EDGE_STATE_MISS", "SEARCH_ENUM",
              "WRONG_ALGORITHM_ADMISSIBLE")


def _calibration_slice(core, n=6):
    """Mode-spanning round-robin pick from the sorted core slice (deterministic)."""
    by_mode = {m: [p for p in core if p.mode == m] for m in MODE_ORDER}
    picks, i = [], 0
    while len(picks) < n and any(by_mode.values()):
        m = MODE_ORDER[i % len(MODE_ORDER)]
        if by_mode[m]:
            picks.append(by_mode[m].pop(0))
        i += 1
    return picks


def _interpret_w132(verdict_label, b_minus_a1_pp, mlb2):
    if verdict_label == "PASS_MECHANISM_DRIVEN":
        return {"outcome": "RESISTANT_BY_CONSTRUCTION_SUPERIORITY_SINGLE_SEED",
                "verdict_label": verdict_label, "b_minus_a1_pp": float(b_minus_a1_pp),
                "mlb2_rescue_rate": float(mlb2),
                "w133_branch": ("CLEAN resistant-by-construction superiority (single seed) "
                                "=> W133 = multi-seed same-budget confirmation toward "
                                "W89/W105 retirement-grade.")}
    if verdict_label == "PASS_NON_MECHANISM_DRIVEN":
        return {"outcome": "MARGIN_WITHOUT_MECHANISM_LOAD_BEARING",
                "verdict_label": verdict_label, "b_minus_a1_pp": float(b_minus_a1_pp),
                "mlb2_rescue_rate": float(mlb2),
                "w133_branch": ("Margin present but reflexion not load-bearing (MLB fail) "
                                "=> NOT a clean mechanism win; register bounded.")}
    return {"outcome": "RESISTANT_BY_CONSTRUCTION_PILOT_CAP",
            "verdict_label": verdict_label, "b_minus_a1_pp": float(b_minus_a1_pp),
            "mlb2_rescue_rate": float(mlb2),
            "w133_branch": ("FAIL on the minted resistant-by-construction field => the "
                            "bounded contamination-EXPOSED-HumanEval-family-at-70B ceiling "
                            "STANDS; resistant superiority still 0 clean; the "
                            "'wrong-test' escape is closed. W133 = different axis / "
                            "DEV_ONLY characterization on the minted field.")}


def main() -> int:
    ap = argparse.ArgumentParser(description="W132 Maverick calibration + pilot")
    ap.add_argument("--model", default=W132_TARGET_MODEL)
    ap.add_argument("--mode", choices=["calibration", "pilot"], default="calibration")
    ap.add_argument("--seed", type=int, default=132_001)
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--timeout-s", type=float, default=EXEC_TIMEOUT_S)
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "w132" / "pilot"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print("  minting battlefield (deterministic) ...")
    bf = mint_battlefield_v1(RBC_SLATE_V1, global_seed=GLOBAL_SEED,
                             minted_date=MINTED_DATE, timeout_s=float(args.timeout_s),
                             official_identities=OFFICIAL_IDENTITIES)
    cert = certify_resistance_v1(model_id=str(args.model), minted_date=MINTED_DATE,
                                 n_core=bf.manifest.n_admitted, raw_cid=bf.manifest.raw_cid)
    core = select_core_slice_v1(bf, n_problems=30)
    cid = core_slice_cid_v1(core)
    earned = bool(bf.meets_min_slice and cert.resistant and len(core) == 30)
    print(f"  n_admitted={bf.manifest.n_admitted} meets_min_slice={bf.meets_min_slice} "
          f"resistant={cert.resistant} core_cid={cid[:16]}…")
    if not earned:
        raise SystemExit("battlefield not pilot-earned (<30 or not resistant); refusing NIM.")
    if not cid.startswith(EXPECTED_CORE_SLICE_CID):
        raise SystemExit(f"core slice CID {cid[:16]} != {EXPECTED_CORE_SLICE_CID} (drift); refusing.")

    if args.mode == "calibration":
        run_problems = _calibration_slice(core, n=6)
    else:
        run_problems = list(core)
    subset = [p.to_pilot_problem(minted_date=MINTED_DATE) for p in run_problems]
    modes = {}
    for p in run_problems:
        modes[p.mode] = modes.get(p.mode, 0) + 1
    print(f"  mode={args.mode} running {len(subset)} problems modes={modes}")
    for p in run_problems:
        print(f"    {p.problem_id:40s} samples={len(p.samples)} secret={len(p.secret_cases)}")

    if args.dry_run:
        print(f"  --dry-run: validated {len(subset)} minted problems; stopping before NIM.")
        print("  --- statement preview (problem 1) ---")
        print("\n".join(subset[0].statement.splitlines()[:10]))
        return 0

    run_id = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_model = str(args.model).replace("/", "_")
    out_dir = Path(args.out_dir) / f"w132_{args.mode}_{safe_model}_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    sidecar_f = open(out_dir / "reflexion_calls.jsonl", "w")

    def sidecar_writer(rec):
        sidecar_f.write(json.dumps(rec, separators=(",", ":")) + "\n")
        sidecar_f.flush()

    gen = _build_nim_gen(model=str(args.model), sidecar_writer=sidecar_writer)
    cfg = IcpcBenchConfigV1(K_multi_sample=5, seeds=(int(args.seed),),
                            sampling_temperature=0.7,
                            max_tokens_per_call=int(args.max_tokens),
                            executor_timeout_s=float(args.timeout_s))
    print(f"  bench config = {cfg}")
    t0 = time.time()
    report = run_icpc_reflexion_bench_v1(
        gen=gen, model_id=str(args.model), subset=subset, config=cfg,
        on_problem_start=lambda s, i, t: print(
            f"  seed={s} p_idx={i+1}/{len(subset)} qid={t}", flush=True))
    sidecar_f.close()
    wall_s = float(time.time() - t0)
    mlb = _mlb_rates(report)
    gates = _evaluate_phase2_gates(report=report, mlb=mlb)
    interp = _interpret_w132(gates["verdict_label"], gates["b_minus_a1_pp"],
                             float(mlb["mlb2_rescue_rate"]))

    a1 = float(report.a1_mean_pass_at_1)
    non_degenerate = bool(0.0 < a1 < 0.90 and mlb["n_b_invoked_reflexion"] >= 1)
    rep = report.to_dict()
    rep.update({
        "mode": args.mode, "wall_s": round(wall_s, 2),
        "battlefield_manifest_cid": bf.manifest.manifest_cid(),
        "core_slice_cid": cid, "minted_date": MINTED_DATE,
        "resistance_cert": cert.to_dict(),
        "slice_problem_ids": [p.problem_id for p in run_problems],
        "slice_modes": modes,
        "mlb": mlb, "phase2_evaluation": gates, "w132_interpretation": interp,
        "calibration_non_degenerate": non_degenerate,
        "calibration_rule": "0 < A1 < 0.90 and at least one attempt-0 failure (MLB invocable)",
    })
    (out_dir / f"w132_{args.mode}_report.json").write_text(
        json.dumps(rep, indent=2, default=str))
    (Path(args.out_dir) / f"latest_{args.mode}.txt").write_text(out_dir.name + "\n")

    print()
    print(f"  WALL {wall_s:.1f}s; A0={report.a0_mean_pass_at_1*100:.2f}% "
          f"A1={report.a1_mean_pass_at_1*100:.2f}% B={report.b_mean_pass_at_1*100:.2f}% "
          f"B-A1={report.b_mean_minus_a1_mean_pp:+.2f}pp")
    print(f"  MLB-1 {mlb['mlb1_invocation_rate']*100:.2f}% "
          f"({mlb['n_b_invoked_reflexion']}/{mlb['n_problems_total']}) "
          f"{'PASS' if mlb['mlb1_passes'] else 'FAIL'}; "
          f"MLB-2 {mlb['mlb2_rescue_rate']*100:.2f}% "
          f"({mlb['n_b_rescued_via_reflexion']}/{mlb['n_b_invoked_reflexion']}) "
          f"{'PASS' if mlb['mlb2_passes'] else 'FAIL'}")
    if args.mode == "calibration":
        print(f"  calibration_non_degenerate = {non_degenerate}  (A1={a1*100:.1f}%, "
              f"invoked={mlb['n_b_invoked_reflexion']})")
        print("  => " + ("CLEARED: full pilot may run." if non_degenerate
                         else "DEGENERATE: do NOT force the full pilot (RUNBOOK § 8a)."))
    else:
        print(f"  Phase-2 {gates['n_phase2_passed_of_9']}/9; verdict {gates['verdict_label']}")
        print(f"  W132 outcome: {interp['outcome']}")
    print(f"  out_dir: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
