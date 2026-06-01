"""W127 Lane γ — targeted resistant fresh-generation probe (gated by R1 ∧ R2).

This driver ENCODES the RUNBOOK_W127 § 6 earn gate in executable form.  Fresh resistant
hosted spend is earned ONLY iff BOTH:

* **R1** — the EXPOSED dev bench (Lane β) verdict is EARNED
  (results/w127/dev_bench/exposed_dev_bench_verdict.json), AND
* **R2** — the capability atlas (Lane α) has a dominant scaffoldable resistant cluster
  (>= R2_MIN_CLUSTER problems, scaffoldable) that the dev-bench earned families target.

If R1 ∧ R2 hold, it runs the SMALLEST honest cluster-matched probe (the scaffoldable
resistant problems in the earned cluster, K=5, <= 1 seed) with G3 scaffolded fresh
generation, grades on the OFFICIAL secret cases, and reports ``targeted_new_solves`` vs the
old 11-generation pool (which was 0 on these).  If R1 ∧ R2 do NOT both hold ⇒ **$0 resistant
NIM**, registers the exact blocker.  Emits
results/w127/targeted_probe/targeted_resistant_probe_verdict.json.
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

from coordpy.coordpy_icpc_battlefield_v1 import (  # noqa: E402
    classify_battlefield_listing_v1, core_slice_cid_v1, select_battlefield_core_slice_v1)
import coordpy.controller_native_code_mechanism_v1 as M  # noqa: E402
import coordpy.family_scaffold_generation_v1 as G  # noqa: E402
from coordpy.family_adapted_repair_synthesis_v1 import SynthesisLeakageGuardV1  # noqa: E402
from coordpy.icpc_reflexion_bench_v1 import (  # noqa: E402
    grade_icpc_candidate_case_v1, grade_on_secret_v1)
from scripts.run_w120_icpc_pilot import load_pilot_problems  # noqa: E402
from scripts.run_w127_exposed_dev_bench_v1 import _build_local_nim_gen  # noqa: E402

EXPECTED_SLICE_CID_30 = "01bf9ef869a56e20"
CORPUS_DIR = os.path.join(ROOT, "results", "w120", "icpc_pilot")
CACHE = os.path.join(ROOT, "results", "w126", "cache", "grade_cache_v1.json")
ATLAS = os.path.join(ROOT, "results", "w127", "atlas", "capability_atlas_v1.json")
DEV = os.path.join(ROOT, "results", "w127", "dev_bench", "exposed_dev_bench_verdict.json")
OUT_DIR = os.path.join(ROOT, "results", "w127", "targeted_probe")
W127_TARGET_MODEL = "meta/llama-4-maverick-17b-128e-instruct"
R2_MIN_CLUSTER = 3


RESISTANT_PKG_ROOTS = ("/tmp/w120_icpc/pkgcache", "/tmp/w122_icpc/pkgcache_resistant")


def _latest_corpus() -> str:
    name = open(os.path.join(CORPUS_DIR, "latest_run.txt")).read().strip()
    return os.path.join(CORPUS_DIR, name, "icpc_reflexion_calls.jsonl")


def _resistant_accepted_texts(short_name: str, *, max_files: int = 8) -> list:
    """Load a resistant target's OWN accepted-solution texts for the leakage BLOCK CHECK
    only (RUNBOOK_W127 § 3) — NEVER passed to the generator."""
    import glob
    out: list = []
    for root in RESISTANT_PKG_ROOTS:
        for p in sorted(glob.glob(os.path.join(
                root, "**", short_name, "submissions", "accepted", "*"), recursive=True)):
            if os.path.isfile(p):
                try:
                    out.append(open(p, encoding="utf-8", errors="replace").read())
                except OSError:
                    pass
            if len(out) >= max_files:
                return out
    return out


def _passes_all_samples(problem, code, *, timeout_s=5.0):
    for inp, exp in problem.samples:
        r = grade_icpc_candidate_case_v1(
            candidate_code=code, stdin_text=inp, expected_stdout=exp,
            kind=problem.kind, float_tol=problem.float_tol, timeout_s=timeout_s)
        if not r.passed:
            return False
    return True


def _pass_at_k(problem, codes, *, timeout_s):
    """Prescreen on public samples (cheap) then grade only sample-passers on secret."""
    for code in codes:
        if not code.strip() or not _passes_all_samples(problem, code):
            continue
        passed, _t, _n = grade_on_secret_v1(problem, code, timeout_s=timeout_s)
        if passed:
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exposed-root", default="/tmp/w121_icpc")
    ap.add_argument("--model", default=W127_TARGET_MODEL)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--R", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--timeout-s", type=float, default=8.0)
    ap.add_argument("--force", action="store_true",
                    help="ignore the R1∧R2 gate (NEVER use for a real verdict)")
    ap.add_argument("--gate-only", action="store_true",
                    help="print the R1∧R2 decision + cluster subset, spend nothing")
    args = ap.parse_args()

    atlas = json.load(open(ATLAS))["atlas"]
    dev = json.load(open(DEV))
    dev_earned = bool(dev["earn_gate"]["earned"])
    earned_families = set(dev["earn_gate"]["gain_families"])
    # R2 — scaffoldable resistant clusters >= R2_MIN_CLUSTER that the dev line targets
    scaff_by_fam = atlas.get("scaffoldable_by_family", {})
    dominant_scaffoldable = {f: c for f, c in scaff_by_fam.items() if c >= R2_MIN_CLUSTER}
    r2_families = set(dominant_scaffoldable) & earned_families
    r2 = bool(r2_families)
    r1 = dev_earned

    blocker = None
    if not r1:
        blocker = (f"R1 FAIL — EXPOSED scaffold dev bench NOT earned "
                   f"({dev['earn_gate']['verdict_label']}, net "
                   f"{dev['earn_gate']['net_scaffold_gain']:+d})")
    elif not r2:
        blocker = (f"R2 FAIL — no scaffoldable resistant cluster >= {R2_MIN_CLUSTER} "
                   f"intersecting the dev-earned families {sorted(earned_families)} "
                   f"(scaffoldable resistant clusters: {scaff_by_fam})")

    if args.gate_only:
        subset = [e["short_name"] for e in atlas["entries"]
                  if e["dominant_algorithm_family"] in r2_families
                  and e.get("scaffoldable_flag")]
        print(f"  R1(dev earned)={r1}  R2(cluster match)={r2}  "
              f"r2_families={sorted(r2_families)}")
        print(f"  cluster-matched resistant subset ({len(subset)}): {subset}")
        print(f"  blocker={blocker}")
        return 0

    verdict = {
        "schema": "coordpy.w127_targeted_resistant_probe.v1", "lane": "gamma_targeted_probe",
        "verified_on": _dt.date.today().isoformat(),
        "r1_dev_earned": r1, "r2_cluster_match": r2,
        "earned_families": sorted(earned_families),
        "scaffoldable_resistant_clusters": scaff_by_fam,
        "r2_target_families": sorted(r2_families),
        "r2_min_cluster": R2_MIN_CLUSTER,
    }

    if not (r1 and r2) and not args.force:
        verdict.update({"probe_launched": False, "nim_spend": 0, "blocker": blocker,
                        "carry_forward": "W127-L-EXPOSED-SCAFFOLD-DEV-BENCH-CAP"
                        if not r1 else "W127-L-RESISTANT-CLUSTER-NOT-SCAFFOLD-MATCHED",
                        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat()})
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(os.path.join(OUT_DIR, "targeted_resistant_probe_verdict.json"), "w") as f:
            json.dump(verdict, f, indent=2, default=str)
        print(f"  R1(dev earned)={r1}  R2(cluster match)={r2}  => NOT EARNED")
        print(f"  blocker: {blocker}")
        print(f"  $0 resistant NIM. wrote {OUT_DIR}/targeted_resistant_probe_verdict.json")
        return 0

    # ---- R1 ∧ R2 hold: run the smallest cluster-matched targeted probe ----
    print(f"  R1∧R2 EARNED (target families {sorted(r2_families)}); running targeted probe")
    full = classify_battlefield_listing_v1()
    slice30 = select_battlefield_core_slice_v1(full, n_problems=30)
    slice_cid = core_slice_cid_v1(slice30)
    if not slice_cid.startswith(EXPECTED_SLICE_CID_30):
        raise SystemExit("slice drift; refusing.")
    problems = load_pilot_problems(list(slice30))
    recs = [json.loads(l) for l in open(_latest_corpus())]
    pools = {problems[i].problem_id:
             M.build_pool_from_records(recs[i * 11:i * 11 + 11], problems[i].problem_id)
             for i in range(len(problems))}
    cache = json.load(open(CACHE))
    unsolved = set(cache["summary"]["unsolved_ids"])
    atlas_by_short = {e["short_name"]: e for e in atlas["entries"]}

    # cluster-matched subset: unsolved resistant problems whose atlas family is in r2_families
    subset = [p for p in problems if p.problem_id in unsolved
              and atlas_by_short.get(p.short_name, {}).get("dominant_algorithm_family")
              in r2_families
              and atlas_by_short.get(p.short_name, {}).get("scaffoldable_flag")]
    print(f"  cluster-matched resistant subset: {[p.short_name for p in subset]}")

    teachers = G.load_exposed_problems_v1(args.exposed_root)
    lib = G.build_scaffold_library_v1(teachers)
    run_id = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_run = os.path.join(OUT_DIR, f"w127_probe_{run_id}")
    os.makedirs(out_run, exist_ok=True)
    sidecar = open(os.path.join(out_run, "probe_calls.jsonl"), "w")
    gen = _build_local_nim_gen(model=str(args.model), read_timeout_s=110.0,
                               sidecar_writer=lambda r: sidecar.write(
                                   json.dumps(r, separators=(",", ":")) + "\n"))

    t0 = time.time()
    per = []
    n_calls = 0
    for p in subset:
        pool = pools[p.problem_id]
        prior = [pool.a0_code, *pool.a1_codes, *pool.b_codes]
        cls = G.target_family_ranking_v1(p.statement, p.samples, prior_generations=prior)
        prio = G.prioritized_families_v1(cls)
        rr = G.retrieve_scaffolds_v1(target_short=p.short_name, target_statement=p.statement,
                                     target_family=cls.family, library=lib, R=args.R,
                                     candidate_families=prio)
        codes = G.scaffolded_generate_v1(gen, p, rr.scaffolds, K=args.K,
                                         temperature=args.temperature,
                                         max_tokens=args.max_tokens)
        n_calls += args.K
        prov = "\n".join([sc.skeleton for sc in rr.scaffolds] + [p.statement]
                         + [i + o for i, o in p.samples])
        guard = SynthesisLeakageGuardV1(p, target_accepted_texts=(),
                                        provenance_texts=[prov])
        clean, reason = G.assert_scaffold_pipeline_clean_v1(
            target_short=p.short_name, scaffolds=rr.scaffolds, candidate_texts=codes,
            guard=guard, target_accepted_texts=_resistant_accepted_texts(p.short_name),
            provenance=prov)
        solved = bool(clean and _pass_at_k(p, codes, timeout_s=args.timeout_s))
        per.append({"short_name": p.short_name, "family": cls.family,
                    "n_scaffolds": len(rr.scaffolds), "leakage_clean": clean,
                    "new_solve": solved})
        print(f"   {p.short_name:28s} fam={cls.family:16s} nscaf={len(rr.scaffolds)} "
              f"clean={int(clean)} new_solve={int(solved)}"
              f"{'  *** NEW RESISTANT SOLVE ***' if solved else ''}", flush=True)
    sidecar.close()
    wall = time.time() - t0
    new_solves = sum(1 for r in per if r["new_solve"])
    verdict.update({
        "probe_launched": True, "nim_spend": n_calls, "wall_s": round(wall, 1),
        "slice_cid": slice_cid, "subset": [p.short_name for p in subset],
        "teacher_library": lib.summary(), "per_problem": per,
        "targeted_new_solves": new_solves,
        "all_leakage_clean": all(r["leakage_clean"] for r in per),
        "verdict_label": ("RESISTANT_SCAFFOLD_NEW_SOLVE" if new_solves >= 1 else
                          "RESISTANT_SCAFFOLD_FRESH_GEN_CAP"),
        "carry_forward": (None if new_solves >= 1 else
                          "W127-L-RESISTANT-SCAFFOLD-FRESH-GEN-CAP"),
        "broader_pilot_decision": ("EARNED — define broader cluster-matched pilot"
                                   if new_solves >= 1 else "NOT earned"),
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    })
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "targeted_resistant_probe_verdict.json"), "w") as f:
        json.dump(verdict, f, indent=2, default=str)
    print(f"\n  targeted_new_solves={new_solves}/{len(subset)}  nim={n_calls}  "
          f"wall={wall:.0f}s  verdict={verdict['verdict_label']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
