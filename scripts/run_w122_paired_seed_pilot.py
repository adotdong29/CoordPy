#!/usr/bin/env python3
"""W122-α — paired-seed CLOSURE pilot: ONE new seed on BOTH the W120 RESISTANT and the
W121 EXPOSED official-ICPC battlefields, then the 2-seed aggregate closure verdict.

This is the main empirical lane of W122 (RUNBOOK_W122 §§ 1-2-6).  It removes the ONE live
caveat on the matched-family contrast: single seed each side.  It reuses the EXACT W120
resistant 30-slice (CID 01bf9ef8...) and W121 exposed 30-slice (CID 32d15db5...), the EXACT
W120/W121 package loaders, the EXACT W120 stdin/stdout reflexion bench, and the VERBATIM
W108 gate evaluator — the ONLY thing that changes vs W120/W121 is the seed (default 120002).

For each field it runs A0 + A1 + B (sequential reflexion, K=5) on the 30-slice at
``meta/llama-4-maverick-17b-128e-instruct`` (1 new seed x 30 x K=5 = 330 calls/field; both
fields = 660 calls).  It then loads the EXISTING seed-120001 results (W120 + W121) and
computes the 2-seed aggregate verdict via ``interpret_paired_closure_v1`` (the pre-committed
symmetric closure rule).  If the 2-seed aggregate is B4 AMBIGUOUS, the runbook earns a 3rd
paired seed (run this script again with --seed 120003 --include-prior-seeds 120001,120002).

Grader = official secret cases (token-diff oracle; NO LLM judge).  Reflexion feedback =
public samples + judge verdict bit + stderr ONLY (never secret data).  Requires
``NVIDIA_API_KEY``.

Usage::

    # validate slices + packages on BOTH fields, NO NIM:
    python scripts/run_w122_paired_seed_pilot.py --dry-run
    # canary (2 problems/field, ~44 calls):
    python scripts/run_w122_paired_seed_pilot.py --n-problems 2 --label canary
    # full paired seed (660 calls) + 2-seed closure verdict:
    python scripts/run_w122_paired_seed_pilot.py --n-problems 30
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

from coordpy.coordpy_icpc_battlefield_v1 import (  # noqa: E402
    KIND_PASSFAIL, classify_battlefield_listing_v1, core_slice_cid_v1,
    run_battlefield_construction_v1, select_battlefield_core_slice_v1)
from coordpy.coordpy_icpc_exposed_control_v1 import (  # noqa: E402
    classify_exposed_listing_v1, run_exposed_control_construction_v1,
    select_exposed_core_slice_v1)
from coordpy.icpc_reflexion_bench_v1 import (  # noqa: E402
    IcpcBenchConfigV1, IcpcPilotProblemV1, run_icpc_reflexion_bench_v1)
from coordpy.coordpy_icpc_paired_seed_closure_v1 import (  # noqa: E402
    FieldSeedResultV1, W122_EXPOSED_FIELD, W122_RESISTANT_FIELD,
    aggregate_field_seeds_v1, interpret_paired_closure_v1, w123_fire_condition_v1)
# reuse the W120 + W121 package loaders + the verbatim W108 NIM gen + gate evaluator
import scripts.run_w120_icpc_pilot as w120pilot  # noqa: E402
import scripts.run_w121_exposed_pilot as w121pilot  # noqa: E402
from scripts.run_w108_livecodebench_pilot import (  # noqa: E402
    _build_nim_gen, _evaluate_phase2_gates, _mlb_rates)

W122_TARGET_MODEL = "meta/llama-4-maverick-17b-128e-instruct"
RES_CID = "01bf9ef869a56e20"   # W120 resistant 30-slice CID prefix (verified)
EXP_CID = "32d15db5b66e1e12"   # W121 exposed   30-slice CID prefix (verified)
# isolate this run's package cache so it never clobbers the W120/W121 caches
w120pilot.PKG_CACHE = Path("/tmp/w122_icpc/pkgcache_resistant")


def _resistant_problems(n: int):
    full = classify_battlefield_listing_v1()
    sl = select_battlefield_core_slice_v1(full, n_problems=30)
    cid = core_slice_cid_v1(sl)
    if not cid.startswith(RES_CID):
        raise SystemExit(f"RESISTANT slice CID {cid[:16]} != {RES_CID}; refusing (drift).")
    return list(sl)[:n], cid


def _exposed_problems(n: int):
    full = classify_exposed_listing_v1()
    sl = select_exposed_core_slice_v1(full, n_problems=30)
    cid = core_slice_cid_v1(sl)
    if not cid.startswith(EXP_CID):
        raise SystemExit(f"EXPOSED slice CID {cid[:16]} != {EXP_CID}; refusing (drift).")
    return list(sl)[:n], cid


def _load_resistant_pkgs(slice_records):
    w120pilot.PKG_CACHE.mkdir(parents=True, exist_ok=True)
    return w120pilot.load_pilot_problems(slice_records)


def _load_exposed_pkgs(slice_records):
    w121pilot.w120pilot.PKG_CACHE = Path("/tmp/w122_icpc/pkgcache_exposed")
    w121pilot.w120pilot.PKG_CACHE.mkdir(parents=True, exist_ok=True)
    return w121pilot.load_pilot_problems(slice_records)


def _run_field(*, field, problems, gen, seed, max_tokens, timeout_s, on_start):
    cfg = IcpcBenchConfigV1(
        K_multi_sample=5, seeds=(int(seed),), sampling_temperature=0.7,
        max_tokens_per_call=int(max_tokens), executor_timeout_s=float(timeout_s))
    report = run_icpc_reflexion_bench_v1(
        gen=gen, model_id=W122_TARGET_MODEL, subset=problems, config=cfg,
        on_problem_start=on_start)
    mlb = _mlb_rates(report)
    gates = _evaluate_phase2_gates(report=report, mlb=mlb)
    return report, mlb, gates


def _report_to_field_seed(field: str, path: str) -> FieldSeedResultV1:
    d = json.loads(Path(path).read_text())
    gates = d.get("phase2_evaluation", {})
    mlb = d.get("mlb", {})
    return FieldSeedResultV1(
        field=field, seed=int(d["per_seed"][0]["seed"]),
        b_minus_a1_pp=float(d["b_mean_minus_a1_mean_pp"]),
        verdict_label=str(gates.get("verdict_label", "FAIL")),
        a0_pass_at_1_pct=float(d["a0_mean_pass_at_1"] * 100),
        mlb2_rescue_rate=float(mlb.get("mlb2_rescue_rate", 0.0)))


def _collect_prior_field_seeds(field: str, *, current_run_dir: Path,
                               n_problems_required: int = 30) -> dict:
    """Gather ALL prior seed results for ``field``, keyed by seed (deduped) so a 3rd-seed
    run aggregates {120001, 120002, 120003}, NOT just {120001, 120003}. Sources:

    * the CANONICAL seed-120001 run — W120 (resistant) / W121 (exposed);
    * EVERY prior W122 paired-seed run dir (except ``current_run_dir``), full (n=30) only.

    Re-runs of the same seed keep the LATEST-written (largest mtime) report for that seed.
    """
    if field == W122_RESISTANT_FIELD:
        canon_dir = ROOT / "results" / "w120" / "icpc_pilot"
        canon_report = "icpc_reflexion_bench_report.json"
        canon_glob = "results/w120/icpc_pilot/w120_icpc_pilot_*_2026*/icpc_reflexion_bench_report.json"
        w122_report = "resistant_reflexion_bench_report.json"
    else:
        canon_dir = ROOT / "results" / "w121" / "exposed_pilot"
        canon_report = "exposed_reflexion_bench_report.json"
        canon_glob = "results/w121/exposed_pilot/w121_exposed_pilot_*_2026*/exposed_reflexion_bench_report.json"
        w122_report = "exposed_reflexion_bench_report.json"

    candidates: list[str] = []
    # canonical seed-120001 via latest_run.txt (fallback: largest non-canary report)
    lr = canon_dir / "latest_run.txt"
    canon_path = None
    if lr.is_file():
        c = lr.parent / lr.read_text().strip() / canon_report
        if c.exists():
            canon_path = str(c)
    if canon_path is None:
        gp = [p for p in glob.glob(str(ROOT / canon_glob)) if "canary" not in p]
        if not gp:
            raise SystemExit(f"no canonical seed-120001 report for {field}")
        canon_path = max(gp, key=lambda p: Path(p).stat().st_size)
    candidates.append(canon_path)
    # every prior W122 paired-seed run dir (full only), excluding the current one
    for dd in sorted(glob.glob(str(ROOT / "results" / "w122" / "paired_seed"
                                    / "w122_paired_seed_*"))):
        if Path(dd).resolve() == current_run_dir.resolve() or "canary" in dd:
            continue
        rp = Path(dd) / w122_report
        if rp.exists():
            candidates.append(str(rp))

    by_seed: dict[int, tuple[float, FieldSeedResultV1]] = {}
    for p in candidates:
        d = json.loads(Path(p).read_text())
        if int(d.get("n_problems", 0)) < n_problems_required:
            continue
        fs = _report_to_field_seed(field, p)
        mtime = Path(p).stat().st_mtime
        if fs.seed not in by_seed or mtime > by_seed[fs.seed][0]:
            by_seed[fs.seed] = (mtime, fs)
    return {seed: v[1] for seed, v in by_seed.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description="W122 paired-seed closure pilot (both fields)")
    ap.add_argument("--model", default=W122_TARGET_MODEL)
    ap.add_argument("--n-problems", type=int, default=30)
    ap.add_argument("--seed", type=int, default=120_002)
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--timeout-s", type=float, default=15.0)
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "w122" / "paired_seed"))
    ap.add_argument("--label", default="")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # EARNED GATE — both battlefields must still certify Maverick (Lane γ gate closed =>
    # Maverick is the unique target; this re-affirms pilot-admissibility before spend).
    rres = run_battlefield_construction_v1(verified_on=_dt.date.today().isoformat())
    eres = run_exposed_control_construction_v1(verified_on=_dt.date.today().isoformat())
    if not rres.pilot_earned:
        raise SystemExit("resistant battlefield not pilot-admissible; refusing NIM.")
    if not eres.exposed_pilot_earned:
        raise SystemExit("exposed control not pilot-admissible; refusing NIM.")
    if args.model not in [m.model_id for m in rres.per_model if m.pilot_admissible]:
        raise SystemExit(f"{args.model} not resistant-pilot-admissible; refusing.")
    if args.model not in [m.model_id for m in eres.per_model if m.pilot_admissible]:
        raise SystemExit(f"{args.model} not exposed-pilot-admissible; refusing.")

    n = int(args.n_problems)
    r_slice, r_cid = _resistant_problems(n)
    e_slice, e_cid = _exposed_problems(n)
    print(f"  RESISTANT 30-slice cid={r_cid[:16]}…  running {len(r_slice)} problems")
    print(f"  EXPOSED   30-slice cid={e_cid[:16]}…  running {len(e_slice)} problems")
    print(f"  seed={args.seed} (paired NEW seed; existing seed 120001 already on both fields)")

    print("  fetching official ICPC packages (cached) ...")
    r_problems = _load_resistant_pkgs(r_slice)
    e_problems = _load_exposed_pkgs(e_slice)
    for tag, ps in (("RES", r_problems), ("EXP", e_problems)):
        bad = [p.problem_id for p in ps if not p.secret_cases or not p.statement
               or p.statement.startswith("(statement")]
        if bad:
            raise SystemExit(f"{tag} package load failed for {bad}; refusing NIM.")
        print(f"    {tag}: {len(ps)} packages OK "
              f"(secret cases {min(len(p.secret_cases) for p in ps)}–"
              f"{max(len(p.secret_cases) for p in ps)})")

    run_id = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    lbl = (f"_{args.label}" if args.label else "")
    out_dir = Path(args.out_dir) / f"w122_paired_seed_{args.seed}_{run_id}{lbl}"
    out_dir.mkdir(parents=True, exist_ok=True)

    provenance = {
        "schema": "coordpy.w122_paired_seed_pilot.v1", "milestone": "W122-alpha",
        "model_id": str(args.model), "paired_new_seed": int(args.seed),
        "existing_seed": 120_001, "n_problems": n, "K_multi_sample": 5,
        "resistant_slice_cid": r_cid, "exposed_slice_cid": e_cid,
        "resistant_instrument": "coordpy_icpc_battlefield_v1 (W120)",
        "exposed_instrument": "coordpy_icpc_exposed_control_v1 (W121)",
        "grader": "official secret cases (data/secret); token-diff oracle; NO LLM judge",
        "reflexion_feedback": "public samples + judge verdict + stderr ONLY (no secret leak)",
        "max_tokens_per_call": int(args.max_tokens),
        "executor_timeout_s": float(args.timeout_s), "label": str(args.label),
        "closure_rule": "RUNBOOK_W122 §2 (symmetric; B1..B4; null band 3.34; margin 5.00)",
    }
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, default=str))

    if args.dry_run:
        print(f"  --dry-run: validated BOTH slices + {len(r_problems)}+{len(e_problems)} "
              f"packages; stopping before NIM. ({out_dir})")
        return 0

    results = {}
    for field, problems, sidecar_name in (
            (W122_RESISTANT_FIELD, r_problems, "resistant_reflexion_calls.jsonl"),
            (W122_EXPOSED_FIELD, e_problems, "exposed_reflexion_calls.jsonl")):
        print(f"\n  ===== FIELD: {field} ({len(problems)} problems, seed {args.seed}) =====")
        sidecar_f = open(out_dir / sidecar_name, "w")
        gen = _build_nim_gen(
            model=str(args.model),
            sidecar_writer=lambda rec: (sidecar_f.write(
                json.dumps(rec, separators=(",", ":")) + "\n"), sidecar_f.flush()))
        t0 = time.time()
        report, mlb, gates = _run_field(
            field=field, problems=problems, gen=gen, seed=int(args.seed),
            max_tokens=int(args.max_tokens), timeout_s=float(args.timeout_s),
            on_start=lambda s, i, t: print(
                f"    {field} seed={s} p_idx={i+1}/{len(problems)} qid={t}", flush=True))
        sidecar_f.close()
        wall = time.time() - t0
        rep = report.to_dict()
        rep.update({"wall_s": round(wall, 2), "provenance": provenance, "mlb": mlb,
                    "phase2_evaluation": gates, "field": field})
        (out_dir / f"{field}_reflexion_bench_report.json").write_text(
            json.dumps(rep, indent=2, default=str))
        results[field] = (report, mlb, gates)
        print(f"    {field}: A0={report.a0_mean_pass_at_1*100:.2f} "
              f"A1={report.a1_mean_pass_at_1*100:.2f} B={report.b_mean_pass_at_1*100:.2f} "
              f"B-A1={report.b_mean_minus_a1_mean_pp:+.2f}pp; "
              f"MLB-1 {mlb['mlb1_invocation_rate']*100:.1f}%/{('P' if mlb['mlb1_passes'] else 'F')} "
              f"MLB-2 {mlb['mlb2_rescue_rate']*100:.1f}%/{('P' if mlb['mlb2_passes'] else 'F')} "
              f"{gates['n_phase2_passed_of_9']}/9 {gates['verdict_label']}")

    # ---- 2-seed aggregate closure verdict (only meaningful at full n=30) ----
    if n >= 30:
        new_seed_results = {}
        for field in (W122_RESISTANT_FIELD, W122_EXPOSED_FIELD):
            report, mlb, gates = results[field]
            new_seed_results[field] = FieldSeedResultV1(
                field=field, seed=int(args.seed),
                b_minus_a1_pp=float(report.b_mean_minus_a1_mean_pp),
                verdict_label=str(gates["verdict_label"]),
                a0_pass_at_1_pct=float(report.a0_mean_pass_at_1 * 100),
                mlb2_rescue_rate=float(mlb["mlb2_rescue_rate"]))
        # gather ALL prior seeds (W120/W121 seed-120001 + every prior W122 seed), keyed by
        # seed; the current run's new seed overwrites any same-seed prior (a re-run).
        prior_res = _collect_prior_field_seeds(
            W122_RESISTANT_FIELD, current_run_dir=out_dir)
        prior_exp = _collect_prior_field_seeds(
            W122_EXPOSED_FIELD, current_run_dir=out_dir)
        prior_res[int(args.seed)] = new_seed_results[W122_RESISTANT_FIELD]
        prior_exp[int(args.seed)] = new_seed_results[W122_EXPOSED_FIELD]
        r_agg = aggregate_field_seeds_v1(list(prior_res.values()))
        e_agg = aggregate_field_seeds_v1(list(prior_exp.values()))
        print(f"  aggregating RESISTANT seeds {sorted(prior_res)} | "
              f"EXPOSED seeds {sorted(prior_exp)}")
        verdict = interpret_paired_closure_v1(resistant=r_agg, exposed=e_agg)
        fire = w123_fire_condition_v1(verdict.branch)
        payload = {"schema": "coordpy.w122_paired_seed_closure_verdict.v1",
                   "provenance": provenance,
                   "resistant_aggregate": r_agg.to_dict(),
                   "exposed_aggregate": e_agg.to_dict(),
                   "closure_verdict": verdict.to_dict(),
                   "w123_fire_condition": fire.to_dict(),
                   "ts_utc": _dt.datetime.now(_dt.timezone.utc).isoformat()}
        (out_dir / "paired_seed_closure_verdict.json").write_text(
            json.dumps(payload, indent=2, default=str))
        (Path(args.out_dir) / "latest_run.txt").write_text(out_dir.name + "\n")
        print("\n  ===== 2-SEED AGGREGATE CLOSURE VERDICT =====")
        print(f"    RESISTANT seeds {r_agg.seeds} margins {r_agg.per_seed_b_minus_a1_pp} "
              f"=> mean {r_agg.mean_b_minus_a1_pp:+.2f}pp (null_band={verdict.null_band_pp})")
        print(f"    EXPOSED   seeds {e_agg.seeds} margins {e_agg.per_seed_b_minus_a1_pp} "
              f"=> mean {e_agg.mean_b_minus_a1_pp:+.2f}pp")
        print(f"    BRANCH: {verdict.branch}")
        print(f"    caveat_closed={verdict.caveat_closed} third_seed_earned={verdict.third_seed_earned}")
        print(f"    {verdict.interpretation}")
        print(f"    W121 delta: {verdict.w121_delta}")
        print(f"    W123 fires on: {fire.fires_on}")
    print(f"\n  out_dir: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
