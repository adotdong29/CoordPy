#!/usr/bin/env python3
"""W121-β — Maverick × matched-EXPOSED official-ICPC control cheap-pilot driver.

CONDITIONAL ON the W121 exposed-control construction
(``results/w121/exposed_control/exposed_control_verdict.json``) having
``exposed_pilot_earned=true`` AND ``docs/RUNBOOK_W121.md`` being locked.

Runs the SAME W89 sequential-reflexion bench (``icpc_reflexion_bench_v1``) + the SAME
gate evaluator (verbatim W108 ``_evaluate_phase2_gates`` / ``_mlb_rates``) that W120 ran
on the RESISTANT ICPC battlefield — but on the matched EXPOSED control 30-slice (the
immediately-preceding PRE-cutoff year-drops of the SAME official ICPC org families;
every problem dated AT OR BEFORE Maverick's Aug-2024 cutoff ⇒ EXPOSED), at
``meta/llama-4-maverick-17b-128e-instruct``, 1 seed × N × K=5 (default 30 ⇒ 330 calls).

The ONLY systematic difference vs the W120 pilot is the contest date relative to the
cutoff (EXPOSED here vs RESISTANT there): same model, same family, same grader, same
evaluator, same K/budget discipline.  The exposed margin is then contrasted with the
LOCKED W120 resistant result (B−A1 = +0.00) via the pre-committed three-branch
``interpret_exposed_vs_resistant_v1`` (loophole-closed / confound-weakens / ambiguous).

Reuses the W120 pilot's package loaders + statement cleaner (no duplication).  Requires
``NVIDIA_API_KEY``.

Usage::

    python scripts/run_w121_exposed_pilot.py --dry-run                 # 0 NIM
    python scripts/run_w121_exposed_pilot.py --n-problems 2 --label canary
    python scripts/run_w121_exposed_pilot.py --n-problems 30
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

from coordpy.coordpy_icpc_exposed_control_v1 import (  # noqa: E402
    classify_exposed_listing_v1,
    interpret_exposed_vs_resistant_v1,
    run_exposed_control_construction_v1,
    select_exposed_core_slice_v1,
)
from coordpy.coordpy_icpc_battlefield_v1 import KIND_PASSFAIL, core_slice_cid_v1  # noqa: E402
from coordpy.icpc_reflexion_bench_v1 import (  # noqa: E402
    IcpcBenchConfigV1,
    IcpcPilotProblemV1,
    run_icpc_reflexion_bench_v1,
)
# Reuse the proven W120 ICPC package loaders + the verbatim W108 NIM gen + gate evaluator.
import scripts.run_w120_icpc_pilot as w120pilot  # noqa: E402
from scripts.run_w120_icpc_pilot import (  # noqa: E402
    _ensure_ecna_zip,
    _ensure_rmrc_tarball,
    _find_pkg_dir,
    _read_cases,
    _read_statement,
)
from scripts.run_w108_livecodebench_pilot import (  # noqa: E402
    _build_nim_gen,
    _evaluate_phase2_gates,
    _mlb_rates,
)

W121_TARGET_MODEL = "meta/llama-4-maverick-17b-128e-instruct"
W121_EXPECTED_CORE_SLICE_CID_30 = "32d15db5b66e1e12"  # prefix; full asserted in main()
# the W120 RESISTANT result this exposed control is contrasted against (LOCKED)
W120_RESISTANT_B_MINUS_A1_PP = 0.0
# exposed ECNA year-folder mapping (pre-cutoff)
W121_ECNA_YEAR_FOLDER = {"2022": "2022-2023", "2023": "2023-2024"}
# redirect the reused W120 package cache to a W121-specific dir
w120pilot.PKG_CACHE = Path("/tmp/w121_icpc/pkgcache")


def load_pilot_problems(slice_records) -> list[IcpcPilotProblemV1]:
    w120pilot.PKG_CACHE.mkdir(parents=True, exist_ok=True)
    rmrc_roots: dict[str, Path] = {}
    out: list[IcpcPilotProblemV1] = []
    for rec in slice_records:
        repo = rec.source_repo
        if rec.surface == "ECNA":
            yf = W121_ECNA_YEAR_FOLDER[rec.contest_date[:4]]
            pkg = _ensure_ecna_zip(rec.short_name, yf)
        else:
            if repo not in rmrc_roots:
                rmrc_roots[repo] = _ensure_rmrc_tarball(repo)
            pkg = _find_pkg_dir(rmrc_roots[repo], rec.short_name)
        out.append(IcpcPilotProblemV1(
            problem_id=rec.problem_id, short_name=rec.short_name,
            source_repo=repo, contest_date=rec.contest_date,
            statement=_read_statement(pkg), kind=KIND_PASSFAIL, float_tol=1e-6,
            samples=tuple(_read_cases(pkg, "sample")),
            secret_cases=tuple(_read_cases(pkg, "secret"))))
    return out


def _interpret_w121(b_minus_a1_pp: float, verdict_label: str, mlb2: float) -> dict:
    """Pre-committed W121 outcome mapping (RUNBOOK_W121 § 3): the EXPOSED margin vs the
    LOCKED W120 RESISTANT +0.00, through the three-branch interpreter."""
    o = interpret_exposed_vs_resistant_v1(
        exposed_b_minus_a1=float(b_minus_a1_pp),
        resistant_b_minus_a1=W120_RESISTANT_B_MINUS_A1_PP)
    return {"outcome": o.outcome, "verdict_label": verdict_label,
            "exposed_b_minus_a1_pp": float(b_minus_a1_pp),
            "resistant_b_minus_a1_pp": W120_RESISTANT_B_MINUS_A1_PP,
            "mlb2_rescue_rate": float(mlb2),
            "paired_seed_earned": bool(o.paired_seed_earned),
            "interpretation": o.interpretation, "detail": o.to_dict()}


def main() -> int:
    ap = argparse.ArgumentParser(description="W121 Maverick × exposed-ICPC control pilot")
    ap.add_argument("--model", default=W121_TARGET_MODEL)
    ap.add_argument("--n-problems", type=int, default=30)
    ap.add_argument("--seed", type=int, default=120_001)
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--timeout-s", type=float, default=15.0)
    ap.add_argument(
        "--out-dir", default=str(ROOT / "results" / "w121" / "exposed_pilot"))
    ap.add_argument("--label", default="")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # EXPOSED-PILOT-EARNED GATE — refuse to spend unless the exposed control certifies.
    res = run_exposed_control_construction_v1(verified_on=_dt.date.today().isoformat())
    if not res.exposed_pilot_earned:
        raise SystemExit("exposed control not pilot-admissible (core <30 or no "
                         "exposed-certifiable model); refusing NIM (W121 § 7).")
    pilot_models = [m.model_id for m in res.per_model if m.pilot_admissible]
    if str(args.model) not in pilot_models:
        raise SystemExit(f"{args.model} not exposed-pilot-admissible {pilot_models}.")
    print(f"  exposed control: core={res.manifest.n_core_passfail} "
          f"exposed_pilot_earned=True certifiable={pilot_models}")

    full = classify_exposed_listing_v1()
    slice30 = select_exposed_core_slice_v1(full, n_problems=30)
    slice_cid = core_slice_cid_v1(slice30)
    if not slice_cid.startswith(W121_EXPECTED_CORE_SLICE_CID_30):
        raise SystemExit(f"exposed 30-slice CID {slice_cid[:16]} != expected "
                         f"{W121_EXPECTED_CORE_SLICE_CID_30}; refusing (slice drift).")
    run_slice = list(slice30)[:int(args.n_problems)]
    print(f"  exposed 30-slice cid={slice_cid[:16]}…; running {len(run_slice)} problems")

    print("  fetching official ICPC packages (cached) ...")
    problems = load_pilot_problems(run_slice)
    for p in problems:
        print(f"    {p.problem_id:56s} samples={len(p.samples)} "
              f"secret={len(p.secret_cases)} stmt={len(p.statement)}b")
    bad = [p.problem_id for p in problems if not p.secret_cases or not p.statement
           or p.statement.startswith("(statement")]
    if bad:
        raise SystemExit(f"package load failed for {bad}; refusing to spend NIM.")

    run_id = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_model = str(args.model).replace("/", "_")
    lbl = (f"_{args.label}" if args.label else "")
    out_dir = Path(args.out_dir) / f"w121_exposed_pilot_{safe_model}_{run_id}{lbl}"
    out_dir.mkdir(parents=True, exist_ok=True)

    provenance = {
        "schema": "coordpy.w121_exposed_pilot.v1", "milestone": "W121-beta",
        "model_id": str(args.model), "seed": int(args.seed),
        "n_problems": len(run_slice), "K_multi_sample": 5,
        "exposed_instrument": "coordpy_icpc_exposed_control_v1",
        "exposed_manifest_cid": res.manifest.manifest_cid(),
        "exposed_surfaces": list(res.manifest.surfaces),
        "core_slice_cid": slice_cid,
        "slice_problem_ids": [p.problem_id for p in problems],
        "slice_contest_date_min": min(p.contest_date for p in problems),
        "slice_contest_date_max": max(p.contest_date for p in problems),
        "cutoff_boundary": "2024-08-31", "cutoff_confidence": "KNOWN",
        "contamination_window": ("EXPOSED for Maverick: every problem dated AT OR BEFORE "
                                 "the Aug-2024 cutoff (RMRC 2021 + 2022-2023 + ECNA "
                                 "2022-2023 + 2023-2024) — matched-family control to the "
                                 "W120 resistant battlefield (RMRC/ECNA 2024-26)"),
        "matched_resistant_instrument": "coordpy_icpc_battlefield_v1 (W120)",
        "resistant_b_minus_a1_pp": W120_RESISTANT_B_MINUS_A1_PP,
        "grader": "official secret cases (data/secret); token-diff oracle; NO LLM judge",
        "reflexion_feedback": "public samples + judge verdict + stderr ONLY (no secret leak)",
        "max_tokens_per_call": int(args.max_tokens),
        "executor_timeout_s": float(args.timeout_s),
        "clean_exposure_margin_bar": "exposed B-A1 >= +5.0 pp (W89/W105 grade)",
        "label": str(args.label),
    }
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, default=str))

    if args.dry_run:
        print(f"  --dry-run: validated slice + {len(problems)} packages; "
              f"stopping before NIM. ({out_dir})")
        print("  --- statement preview (problem 1) ---")
        print("\n".join(problems[0].statement.splitlines()[:12]))
        return 0

    sidecar_f = open(out_dir / "exposed_reflexion_calls.jsonl", "w")

    def sidecar_writer(rec):
        sidecar_f.write(json.dumps(rec, separators=(",", ":")) + "\n")
        sidecar_f.flush()

    gen = _build_nim_gen(model=str(args.model), sidecar_writer=sidecar_writer)
    cfg = IcpcBenchConfigV1(
        K_multi_sample=5, seeds=(int(args.seed),), sampling_temperature=0.7,
        max_tokens_per_call=int(args.max_tokens),
        executor_timeout_s=float(args.timeout_s))
    print(f"  bench config = {cfg}")
    t0 = time.time()
    report = run_icpc_reflexion_bench_v1(
        gen=gen, model_id=str(args.model), subset=problems, config=cfg,
        on_problem_start=lambda s, i, t: print(
            f"  seed={s} p_idx={i+1}/{len(problems)} qid={t}", flush=True))
    sidecar_f.close()
    wall_s = float(time.time() - t0)
    mlb = _mlb_rates(report)
    gates = _evaluate_phase2_gates(report=report, mlb=mlb)
    interp = _interpret_w121(gates["b_minus_a1_pp"], gates["verdict_label"],
                             float(mlb["mlb2_rescue_rate"]))

    rep = report.to_dict()
    rep.update({"wall_s": round(wall_s, 2), "provenance": provenance, "mlb": mlb,
                "phase2_evaluation": gates, "w121_interpretation": interp})
    (out_dir / "exposed_reflexion_bench_report.json").write_text(
        json.dumps(rep, indent=2, default=str))
    (Path(args.out_dir) / "latest_run.txt").write_text(out_dir.name + "\n")

    print()
    print(f"  WALL {wall_s:.1f}s; A0={report.a0_mean_pass_at_1*100:.2f}% "
          f"A1={report.a1_mean_pass_at_1*100:.2f}% B={report.b_mean_pass_at_1*100:.2f}% "
          f"B-A1={report.b_mean_minus_a1_mean_pp:+.2f}pp")
    print(f"  MLB-1 {mlb['mlb1_invocation_rate']*100:.2f}% "
          f"{'PASS' if mlb['mlb1_passes'] else 'FAIL'}; "
          f"MLB-2 {mlb['mlb2_rescue_rate']*100:.2f}% "
          f"{'PASS' if mlb['mlb2_passes'] else 'FAIL'}")
    print(f"  Phase-2 {gates['n_phase2_passed_of_9']}/9; verdict {gates['verdict_label']}")
    print(f"  EXPOSED B-A1={interp['exposed_b_minus_a1_pp']:+.2f}pp  vs  RESISTANT "
          f"{interp['resistant_b_minus_a1_pp']:+.2f}pp (W120)")
    print(f"  W121 outcome: {interp['outcome']}")
    print(f"  paired_seed_earned: {interp['paired_seed_earned']}")
    print(f"  out_dir: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
