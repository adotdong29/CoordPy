"""W128 Lane γ — targeted resistant role-diverse-search probe (gated by T1 ∧ T2).

ENCODES the RUNBOOK_W128 § 6 earn gate in executable form.  Fresh resistant hosted spend is
earned ONLY iff BOTH:

* **T1** — the EXPOSED hard-cluster dev bench (Lane β) verdict is EARNED
  (results/w128/dev_bench/hard_cluster_dev_bench_verdict.json), AND
* **T2** — a NAMED hard family ({graph_flow, simulation_grid}) intersects the dev-earned
  families (graph_flow has 0 EXPOSED supply ⇒ in practice T2 = simulation_grid earned).

If T1 ∧ T2 hold, it runs the SMALLEST honest cluster-matched probe (the resistant hard
problems in the earned named family, e.g. the 4 simulation_grid resistant problems) with the
RDA4 role-diverse mechanism (5 calls/problem), grades committed + pool on the OFFICIAL secret
cases, and reports ``targeted_new_solves`` (RDA4 committed) vs the old 11-generation pool
(0 on these).  If T1 ∧ T2 do NOT both hold ⇒ **$0 resistant NIM**, registers the exact
blocker.  Emits results/w128/targeted_probe/targeted_resistant_probe_verdict.json.
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
import coordpy.family_scaffold_generation_v1 as G  # noqa: E402
import coordpy.role_diverse_algorithm_search_v1 as R  # noqa: E402
from coordpy.family_adapted_repair_synthesis_v1 import SynthesisLeakageGuardV1  # noqa: E402
from scripts.run_w120_icpc_pilot import load_pilot_problems  # noqa: E402
from scripts.run_w127_exposed_dev_bench_v1 import _build_local_nim_gen  # noqa: E402
from scripts.run_w127_targeted_resistant_probe_v1 import _resistant_accepted_texts  # noqa: E402

EXPECTED_SLICE_CID_30 = "01bf9ef869a56e20"
CORPUS_DIR = os.path.join(ROOT, "results", "w120", "icpc_pilot")
CACHE = os.path.join(ROOT, "results", "w126", "cache", "grade_cache_v1.json")
ATLAS = os.path.join(ROOT, "results", "w127", "atlas", "capability_atlas_v1.json")
DEV = os.path.join(ROOT, "results", "w128", "dev_bench", "hard_cluster_dev_bench_verdict.json")
OUT_DIR = os.path.join(ROOT, "results", "w128", "targeted_probe")
W128_TARGET_MODEL = "meta/llama-4-maverick-17b-128e-instruct"
NAMED_HARD = ("graph_flow", "simulation_grid")


def _make_leak_check(short_name, prob):
    accepted = _resistant_accepted_texts(short_name)
    prov = prob.statement + "\n" + "\n".join(i + o for i, o in prob.samples)
    guard = SynthesisLeakageGuardV1(prob, target_accepted_texts=(), provenance_texts=[prov])

    def check(code: str) -> bool:
        if G.reproduces_accepted_block_v1(code, accepted, provenance=prov):
            return False
        res = guard.check(code)
        clean = res.clean if hasattr(res, "clean") else bool(res)
        return bool(clean)
    return check


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=W128_TARGET_MODEL)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--n-sketches", type=int, default=4)
    ap.add_argument("--analyze-temp", type=float, default=0.5)
    ap.add_argument("--impl-temp", type=float, default=0.2)
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--timeout-s", type=float, default=8.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--gate-only", action="store_true")
    ap.add_argument("--force", action="store_true", help="ignore the gate (NEVER for a verdict)")
    args = ap.parse_args()

    atlas = json.load(open(ATLAS))["atlas"]
    atlas_by_short = {e["short_name"]: e for e in atlas["entries"]}
    if not os.path.exists(DEV):
        raise SystemExit("dev-bench verdict missing; run Lane β first.")
    dev = json.load(open(DEV))
    t1 = bool(dev["earn_gate"]["earned"])
    earned_families = set(dev["earn_gate"]["gain_families"])
    t2_families = earned_families & set(NAMED_HARD)
    t2 = bool(t2_families)

    blocker = None
    if not t1:
        blocker = (f"T1 FAIL — EXPOSED hard-cluster dev bench NOT earned "
                   f"({dev['earn_gate']['verdict_label']}, net "
                   f"{dev['earn_gate']['net_rda_gain']:+d})")
    elif not t2:
        blocker = (f"T2 FAIL — no NAMED hard family {list(NAMED_HARD)} in the dev-earned "
                   f"families {sorted(earned_families)} (graph_flow EXPOSED supply=0)")

    cache = json.load(open(CACHE))
    unsolved = set(cache["summary"]["unsolved_ids"])

    verdict = {
        "schema": "coordpy.w128_targeted_resistant_probe.v1", "lane": "gamma_targeted_probe",
        "verified_on": _dt.date.today().isoformat(),
        "t1_dev_earned": t1, "t2_named_hard_match": t2,
        "earned_families": sorted(earned_families), "t2_target_families": sorted(t2_families),
        "named_hard_families": list(NAMED_HARD),
    }

    if args.gate_only:
        subset_names = sorted(
            s for s, e in atlas_by_short.items()
            if e["dominant_algorithm_family"] in t2_families)
        print(f"  T1(dev earned)={t1}  T2(named-hard match)={t2}  "
              f"t2_families={sorted(t2_families)}")
        print(f"  cluster-matched resistant subset ({len(subset_names)}): {subset_names}")
        print(f"  blocker={blocker}")
        return 0

    if not (t1 and t2) and not args.force:
        verdict.update({
            "probe_launched": False, "nim_spend": 0, "blocker": blocker,
            "carry_forward": ("W128-L-ROLE-DIVERSE-HARD-CLUSTER-DEV-BENCH-CAP" if not t1
                              else "W128-L-RESISTANT-CLUSTER-NOT-HARD-MATCHED"),
            "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat()})
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(os.path.join(OUT_DIR, "targeted_resistant_probe_verdict.json"), "w") as f:
            json.dump(verdict, f, indent=2, default=str)
        print(f"  T1={t1}  T2={t2}  => NOT EARNED")
        print(f"  blocker: {blocker}")
        print(f"  $0 resistant NIM. wrote {OUT_DIR}/targeted_resistant_probe_verdict.json")
        return 0

    # ---- T1 ∧ T2 hold: smallest cluster-matched targeted probe with the RDA4 mechanism ----
    print(f"  T1∧T2 EARNED (named-hard families {sorted(t2_families)}); running RDA probe")
    full = classify_battlefield_listing_v1()
    slice30 = select_battlefield_core_slice_v1(full, n_problems=30)
    slice_cid = core_slice_cid_v1(slice30)
    if not slice_cid.startswith(EXPECTED_SLICE_CID_30):
        raise SystemExit("slice drift; refusing.")
    problems = load_pilot_problems(list(slice30))
    subset = [p for p in problems if p.problem_id in unsolved
              and atlas_by_short.get(p.short_name, {}).get("dominant_algorithm_family")
              in t2_families]
    print(f"  cluster-matched resistant subset ({len(subset)}): "
          f"{[p.short_name for p in subset]}")

    run_id = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_run = os.path.join(OUT_DIR, f"w128_probe_{run_id}")
    os.makedirs(out_run, exist_ok=True)
    import threading
    sc = open(os.path.join(out_run, "probe_calls.jsonl"), "w")
    _lock = threading.Lock()
    gen = _build_local_nim_gen(model=str(args.model), read_timeout_s=110.0,
                               sidecar_writer=lambda r: (_lock.acquire(), sc.write(
                                   json.dumps(r, separators=(",", ":")) + "\n"),
                                   sc.flush(), _lock.release()))

    t0 = time.time()
    per = []
    n_calls = [0]
    import concurrent.futures
    plock = threading.Lock()

    def _run(p):
        fam = atlas_by_short.get(p.short_name, {}).get("dominant_algorithm_family", "")
        leak = _make_leak_check(p.short_name, p)
        o = R.run_role_diverse_search_v1(
            gen, p, K=args.K, n_sketches=args.n_sketches, analyze_temp=args.analyze_temp,
            impl_temp=args.impl_temp, max_tokens=args.max_tokens, timeout_s=args.timeout_s,
            family=fam, grade_secret=True, leakage_check=leak)
        committed = bool(o.committed_pass["RDA4"]) and o.leakage_clean
        rec = {"short_name": p.short_name, "family": fam,
               "diversity": o.diversity["classify"], "leakage_clean": o.leakage_clean,
               "rda4_committed_pass": o.committed_pass["RDA4"],
               "pool_pass": o.pool_pass, "rda4_abstained": o.abstained["RDA4"],
               "new_solve": committed, "pool_new_solve": bool(o.pool_pass and o.leakage_clean)}
        with plock:
            per.append(rec)
            n_calls[0] += o.n_calls
            print(f"   {p.short_name:24s} fam={fam:16s} div={o.diversity['classify']:11s} "
                  f"rda4={int(o.committed_pass['RDA4'])} pool={int(o.pool_pass)} "
                  f"abstain={int(o.abstained['RDA4'])} clean={int(o.leakage_clean)}"
                  f"{'  *** NEW RESISTANT SOLVE ***' if committed else ''}", flush=True)
        return rec

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(_run, subset))
    sc.close()
    wall = time.time() - t0
    new_solves = sum(1 for r in per if r["new_solve"])
    pool_new = sum(1 for r in per if r["pool_new_solve"] and r["pool_pass"])
    verdict.update({
        "probe_launched": True, "nim_spend": n_calls[0], "wall_s": round(wall, 1),
        "slice_cid": slice_cid, "subset": [p.short_name for p in subset],
        "per_problem": per, "targeted_new_solves": new_solves,
        "targeted_pool_new_solves": pool_new,
        "all_leakage_clean": all(r["leakage_clean"] for r in per) if per else True,
        "verdict_label": ("RESISTANT_ROLE_DIVERSE_NEW_SOLVE" if new_solves >= 1 else
                          "RESISTANT_ROLE_DIVERSE_SEARCH_CAP"),
        "carry_forward": (None if new_solves >= 1 else
                          "W128-L-RESISTANT-ROLE-DIVERSE-SEARCH-CAP"),
        "broader_pilot_decision": ("EARNED — define broader cluster-matched pilot"
                                   if new_solves >= 1 else "NOT earned"),
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    })
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "targeted_resistant_probe_verdict.json"), "w") as f:
        json.dump(verdict, f, indent=2, default=str)
    print(f"\n  targeted_new_solves={new_solves}/{len(subset)} (pool ceiling {pool_new}) "
          f"nim={n_calls[0]} wall={wall:.0f}s verdict={verdict['verdict_label']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
