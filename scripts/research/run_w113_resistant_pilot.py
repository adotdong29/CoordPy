#!/usr/bin/env python3
"""W113-α — Maverick × RESISTANT-LiveCodeBench cheap-pilot driver (main lane).

CONDITIONAL ON the W113 resistant-slice preflight
(``results/w113/resistant_slice_preflight/preflight_verdict.json``) having
``overall_pass=true`` AND ``docs/RUNBOOK_W113.md`` being locked.

Runs the W89 sequential-reflexion B-pipeline + A0 + A1 baselines against the
date-filtered RESISTANT-for-Llama-4 LiveCodeBench slice (every problem dated
strictly after August 2024) at the stronger model
``meta/llama-4-maverick-17b-128e-instruct``, 1 seed × N × K=5 (default 30 ⇒ 330
NIM calls).

This isolates MODEL SCALE as the only variable vs W108: it reuses the EXACT
W108 30-slice (CID ``2afc318c…``) — refusing to run if the resistant partition
+ deterministic selection do not reproduce it — so the comparison is
Maverick-vs-70B on identical resistant problems, mirroring how W112 reused the
EXACT W110 slice to compare Maverick-vs-70B on exposed BigCodeBench.

Reuse (no duplication): the NIM generator + the MLB / Phase-2 gate evaluator are
imported VERBATIM from the W108 pilot driver (the exact code that scored W108
and — via the W110 driver twin — W112), so W113's gate logic is byte-identical
to the exposed-twin it is compared against. The resistant-slice assertion uses
``livecodebench_resistant_slice_v1``; the verdict is mapped by the
pre-committed ``cross_scale_resistant_interpretation_v1`` rule.

Requires ``NVIDIA_API_KEY``.

Usage::

    # canary (2 problems ~ 22 calls) then full pilot (30 ~ 330 calls):
    python scripts/run_w113_resistant_pilot.py --n-problems 2 --label canary
    python scripts/run_w113_resistant_pilot.py --n-problems 30
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import sys  # noqa: E402
sys.path.insert(0, str(ROOT))

from coordpy.livecodebench_loader_v1 import (  # noqa: E402
    load_livecodebench_functional_v1,
)
from coordpy.livecodebench_reflexion_bench_v1 import (  # noqa: E402
    LCBBenchConfigV1,
    run_livecodebench_reflexion_bench_v1,
    select_livecodebench_functional_slice_v1,
)
from coordpy.livecodebench_resistant_slice_v1 import (  # noqa: E402
    cutoff_boundary_for_model_v1,
    resistant_partition_for_model_v1,
)
from coordpy.cross_scale_resistant_interpretation_v1 import (  # noqa: E402
    W108_RESISTANT_LCB_SLICE_CID,
    interpret_cross_scale_resistant_result_v1,
)
# Reuse the proven W108 NIM generator + gate evaluator verbatim (the exact code
# that scored W108 / W112) — namespace import, no duplication.
from scripts.run_w108_livecodebench_pilot import (  # noqa: E402
    _build_nim_gen,
    _evaluate_phase2_gates,
    _file_sha256,
    _mlb_rates,
    _sha256_hex,
)

W113_TARGET_MODEL = "meta/llama-4-maverick-17b-128e-instruct"
W113_LIVECODEBENCH_RELEASE = "release_v6"
W113_LIVECODEBENCH_CACHE_PATH = (
    "~/.cache/coordpy/livecodebench-test6.jsonl")
W113_LIVECODEBENCH_RELEASE_V6_SHA256 = (
    "bb4c364f71921c4495a6ad15abe1a927350b720009f4933e2e71f8af0f6fd1f5")
W113_PREFLIGHT_VERDICT_CID = (
    "6f30990c042593cd6c26290f54ec254472a369d7887b21ca86fed04c797f6ac8")
W113_RESISTANT_BENCHMARK = (
    "LiveCodeBench release_v6 functional, date-filtered RESISTANT-for-Llama-4 "
    "(contest_date > 2024-08-31; all 2025-01..04)")


def main() -> int:
    import os
    ap = argparse.ArgumentParser(
        description="W113 Maverick × resistant-LiveCodeBench cheap-pilot driver")
    ap.add_argument("--model", default=W113_TARGET_MODEL)
    ap.add_argument("--release", default=W113_LIVECODEBENCH_RELEASE)
    ap.add_argument(
        "--cache-path",
        default=os.path.expanduser(W113_LIVECODEBENCH_CACHE_PATH))
    ap.add_argument(
        "--expected-sha256", default=W113_LIVECODEBENCH_RELEASE_V6_SHA256)
    ap.add_argument("--n-problems", type=int, default=30)
    ap.add_argument("--seed", type=int, default=113_001)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument(
        "--out-dir",
        default=str(ROOT / "results" / "w113" / "resistant_pilot"))
    ap.add_argument("--label", default="")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print(f"  loading LiveCodeBench {args.release} (SHA-pinned) ...")
    full_subset = load_livecodebench_functional_v1(
        release=str(args.release), cache_path=str(args.cache_path),
        expected_sha256=str(args.expected_sha256))
    print(f"  functional subset = {len(full_subset)} problems")

    # RESISTANCE GATE — refuse to spend on a non-resistant / perturbed slice.
    cutoff = cutoff_boundary_for_model_v1(str(args.model))
    if not cutoff.is_resistant_grade():
        raise SystemExit(
            f"model {args.model} cutoff is {cutoff.confidence}, not KNOWN — "
            "cannot certify a resistant slice; refusing to spend NIM "
            "(W113 § 2 / § 6).")
    part = resistant_partition_for_model_v1(full_subset, model_id=str(args.model))
    print(f"  resistant partition (boundary {cutoff.boundary_date} "
          f"[{cutoff.confidence}]): {part.n_resistant}/{part.n_total} "
          f"resistant; dates {part.resistant_date_min}..{part.resistant_date_max}; "
          f"excluded miss={len(part.excluded_missing_date)} "
          f"unparse={len(part.excluded_unparseable_date)} "
          f"not-after={len(part.excluded_not_after_cutoff)}")
    rset = set(part.resistant_question_ids)
    resistant_subset = tuple(p for p in full_subset if p.question_id in rset)

    pilot_slice = select_livecodebench_functional_slice_v1(
        resistant_subset, n_problems=int(args.n_problems))
    slice_qids = [p.question_id for p in pilot_slice]
    slice_cid = _sha256_hex({"kind": "w108_lcb_pilot_slice_v1",
                             "question_ids": slice_qids})
    mix = Counter(p.difficulty for p in pilot_slice)
    dates = sorted(str(p.contest_date)[:10] for p in pilot_slice)
    print(f"  resistant pilot slice = {len(pilot_slice)} problems; "
          f"difficulty {dict(mix)}; dates {dates[0]}..{dates[-1]}")
    print(f"  resistant_slice_cid = {slice_cid}")
    equals_w108 = bool(
        int(args.n_problems) == 30 and slice_cid == W108_RESISTANT_LCB_SLICE_CID)
    if int(args.n_problems) == 30 and not equals_w108:
        raise SystemExit(
            f"resistant 30-slice CID {slice_cid} != W108 slice "
            f"{W108_RESISTANT_LCB_SLICE_CID}; the date filter perturbed the "
            "slice — refusing the clean-cross-scale claim (W113 § 4 P6).")
    print(f"  == W108 slice CID: {equals_w108} (clean cross-scale)")

    corpus_sha = _file_sha256(Path(str(args.cache_path)))
    if corpus_sha.lower() != str(args.expected_sha256).lower():
        raise SystemExit("corpus SHA drift; refusing to spend NIM")

    if args.dry_run:
        print("  --dry-run: validated resistant slice + corpus; "
              "stopping before NIM")
        return 0

    run_id = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_model = str(args.model).replace("/", "_")
    lbl = (f"_{args.label}" if args.label else "")
    out_dir = (Path(args.out_dir)
               / f"w113_resistant_pilot_{safe_model}_{run_id}{lbl}")
    out_dir.mkdir(parents=True, exist_ok=True)
    sidecar_path = out_dir / "livecodebench_reflexion_calls.jsonl"
    sidecar_f = open(sidecar_path, "w")

    def sidecar_writer(rec):
        sidecar_f.write(json.dumps(rec, separators=(",", ":")) + "\n")
        sidecar_f.flush()

    provenance = {
        "schema": "coordpy.w113_resistant_pilot.v1",
        "milestone": "W113-alpha",
        "model_id": str(args.model), "seed": int(args.seed),
        "release": str(args.release),
        "n_problems": int(len(pilot_slice)), "K_multi_sample": 5,
        "corpus_path": str(args.cache_path),
        "corpus_sha256": str(corpus_sha),
        "dataset": "livecodebench/code_generation_lite",
        "preflight_verdict_cid": str(W113_PREFLIGHT_VERDICT_CID),
        "resistant_slice_cid": str(slice_cid),
        "equals_w108_slice_cid": bool(equals_w108),
        "w108_slice_cid": str(W108_RESISTANT_LCB_SLICE_CID),
        "slice_question_ids": list(slice_qids),
        "slice_difficulty_mix": dict(mix),
        "slice_contest_date_min": dates[0], "slice_contest_date_max": dates[-1],
        "resistant_benchmark": W113_RESISTANT_BENCHMARK,
        "cutoff_boundary": cutoff.boundary_date,
        "cutoff_confidence": cutoff.confidence,
        "contamination_window": (
            "RESISTANT for Llama-4-Maverick: every problem dated strictly after "
            "the August-2024 cutoff (all 2025-01..04)"),
        "fixed_priors": {
            "w108_70b_resistant_lcb_b_minus_a1_pp": -3.33,
            "w112_maverick_exposed_bcb_b_minus_a1_pp": 10.00,
        },
        "max_tokens_per_call": int(args.max_tokens),
        "clean_reopening_bar": "verdict_label == PASS_MECHANISM_DRIVEN",
        "label": str(args.label),
    }
    with open(out_dir / "provenance.json", "w") as f:
        json.dump(provenance, f, indent=2, default=str)
    print(f"  output: {out_dir}")

    gen = _build_nim_gen(model=str(args.model), sidecar_writer=sidecar_writer)
    cfg = LCBBenchConfigV1(
        K_multi_sample=5, seeds=(int(args.seed),),
        sampling_temperature=0.7, max_tokens_per_call=int(args.max_tokens))
    print(f"  bench config = {cfg}")
    t0 = time.time()
    report = run_livecodebench_reflexion_bench_v1(
        gen=gen, model_id=str(args.model), subset=pilot_slice, config=cfg,
        on_problem_start=lambda s, i, t: print(
            f"  seed={s} p_idx={i+1}/{len(pilot_slice)} qid={t}", flush=True))
    sidecar_f.close()
    wall_s = float(time.time() - t0)
    mlb = _mlb_rates(report)
    gates = _evaluate_phase2_gates(report=report, mlb=mlb)

    interp = interpret_cross_scale_resistant_result_v1(
        model_id=str(args.model),
        resistant_benchmark=W113_RESISTANT_BENCHMARK,
        verdict_label=gates["verdict_label"],
        b_minus_a1_pp=gates["b_minus_a1_pp"],
        mlb2_rescue_rate=float(mlb["mlb2_rescue_rate"]))

    rep = report.to_dict()
    rep["wall_s"] = float(round(wall_s, 2))
    rep["provenance"] = provenance
    rep["mlb"] = mlb
    rep["phase2_evaluation"] = gates
    rep["cross_scale_interpretation"] = interp.to_dict()
    rep["cross_scale_interpretation_cid"] = interp.cid()
    with open(out_dir / "livecodebench_reflexion_bench_report.json", "w") as f:
        json.dump(rep, f, indent=2, default=str)
    with open(out_dir.parent / "latest_run.txt", "w") as f:
        f.write(out_dir.name + "\n")

    print()
    print(f"  WALL: {wall_s:.1f} s; "
          f"A0={report.a0_mean_pass_at_1*100:.2f}% "
          f"A1={report.a1_mean_pass_at_1*100:.2f}% "
          f"B={report.b_mean_pass_at_1*100:.2f}% "
          f"B-A1={report.b_mean_minus_a1_mean_pp:+.2f}pp")
    print(f"  MLB-1 invocation: {mlb['mlb1_invocation_rate']*100:.2f}% "
          f"({mlb['n_b_invoked_reflexion']}/{mlb['n_problems_total']}) "
          f"-> {'PASS' if mlb['mlb1_passes'] else 'FAIL'}")
    print(f"  MLB-2 rescue: {mlb['mlb2_rescue_rate']*100:.2f}% "
          f"({mlb['n_b_rescued_via_reflexion']}/"
          f"{mlb['n_b_invoked_reflexion']}) "
          f"-> {'PASS' if mlb['mlb2_passes'] else 'FAIL'}")
    print(f"  Phase 2 gates: {gates['n_phase2_passed_of_9']}/9")
    print(f"  Verdict: {gates['verdict_label']}")
    print(f"  Cross-scale outcome: {interp.outcome} "
          f"(clean_reopening={interp.clean_resistant_reopening})")
    print(f"  -> W114: {interp.w114_branch[:90]}")
    print(f"  out_dir: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
