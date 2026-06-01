"""W128 Lane β — EXPOSED hard-cluster dev bench (NIM; dev spend ALLOWED for mechanism
validation only — RUNBOOK_W128 § 5).

Validates the role-diverse algorithm-SEARCH line on a disjoint same-family EXPOSED
hard-cluster dev bench BEFORE any resistant spend.  Dev targets = EXPOSED problems whose
public family is in NON_SCAFFOLDABLE_FAMILIES (simulation_grid PRIORITY; graph_flow EXPOSED
supply = 0 ⇒ resistant-probe-only).  Three arms at MATCHED budget:

* ``plain``    — K=5 i.i.d. plain generation @ T=0.7  (== W120/W121/W127 A1).
* ``scaffold`` — W127 G2->G3 scaffolded generation, K=5 @ T=0.7  (reference; weak by design).
* ``rda``      — role-diverse search: 5 calls = 1 ANALYZE + 4 IMPLEMENT (RDA4 = earn arm),
                 with NIM-free RDA1..RDA4 selection variants + pool ceiling + diversity.

Applies the R1' earn gate (net RDA4 gain >= +2, spans clusters / includes simulation_grid,
winners diversity-REAL + leakage-clean, nontrivial, beats scaffold).  Emits
results/w128/dev_bench/hard_cluster_dev_bench_verdict.json.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime as _dt
import hashlib
import json
import os
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import coordpy.family_scaffold_generation_v1 as G  # noqa: E402
import coordpy.role_diverse_algorithm_search_v1 as R  # noqa: E402
from coordpy.family_adapted_repair_synthesis_v1 import SynthesisLeakageGuardV1  # noqa: E402
from coordpy.icpc_reflexion_bench_v1 import extract_candidate_code_v1  # noqa: E402
# reuse the W127 NIM client + pass@k helpers (DRY)
from scripts.run_w127_exposed_dev_bench_v1 import (  # noqa: E402
    _build_local_nim_gen, _pass_at_k)

W128_TARGET_MODEL = "meta/llama-4-maverick-17b-128e-instruct"
OUT_DIR = os.path.join(ROOT, "results", "w128", "dev_bench")


def _make_leak_check(ep, prob):
    """§3 leakage closure for the RDA arm: a candidate that reproduces a contiguous accepted
    block (W127 corrected tripwire) OR trips the secret guard is NOT clean."""
    accepted = list(ep.accepted_codes)
    prov = ep.statement + "\n" + "\n".join(i + o for i, o in ep.samples)
    guard = SynthesisLeakageGuardV1(prob, target_accepted_texts=(), provenance_texts=[prov])

    def check(code: str) -> bool:
        if G.reproduces_accepted_block_v1(code, accepted, provenance=prov):
            return False
        res = guard.check(code)  # LeakageVerdictV1(clean, reason)
        clean = (res.clean if hasattr(res, "clean")
                 else (res if isinstance(res, bool) else bool(res[0])))
        return bool(clean)
    return check


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exposed-root", default="/tmp/w121_icpc")
    ap.add_argument("--model", default=W128_TARGET_MODEL)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--R", type=int, default=2, help="scaffolds retrieved per target")
    ap.add_argument("--n-sketches", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--analyze-temp", type=float, default=0.5)
    ap.add_argument("--impl-temp", type=float, default=0.2)
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--timeout-s", type=float, default=10.0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--read-timeout-s", type=float, default=120.0)
    ap.add_argument("--families", default=",".join(R.NON_SCAFFOLDABLE_FAMILIES))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-canary", action="store_true")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    hard_families = tuple(f.strip() for f in args.families.split(",") if f.strip())
    probs = G.load_exposed_problems_v1(args.exposed_root)
    fam_of = {p.short_name: G.target_family_ranking_v1(p.statement, p.samples).family
              for p in probs}
    dev = sorted([p for p in probs if fam_of[p.short_name] in hard_families],
                 key=lambda p: p.short_name)
    if args.limit:
        dev = dev[:args.limit]
    teacher = [p for p in probs if p.short_name not in {d.short_name for d in dev}]
    lib = G.build_scaffold_library_v1(teacher)
    dev_cid = hashlib.sha256(json.dumps(sorted(d.short_name for d in dev)).encode()).hexdigest()
    teacher_cid = hashlib.sha256(
        json.dumps(sorted(t.short_name for t in teacher)).encode()).hexdigest()
    assert {d.short_name for d in dev}.isdisjoint({t.short_name for t in teacher})
    fam_counts = {}
    for d in dev:
        fam_counts[fam_of[d.short_name]] = fam_counts.get(fam_of[d.short_name], 0) + 1
    print(f"  exposed={len(probs)} hard-dev={len(dev)} teacher={len(teacher)} (disjoint)")
    print(f"  dev families: {fam_counts}")
    print(f"  hard_dev_target_cid={dev_cid[:16]}…  teacher_cid={teacher_cid[:16]}…")

    run_id = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    lbl = (f"_{args.label}" if args.label else "")
    out_run = os.path.join(OUT_DIR, f"w128_dev_bench_{run_id}{lbl}")
    os.makedirs(out_run, exist_ok=True)
    sidecar = open(os.path.join(out_run, "dev_bench_calls.jsonl"), "w")
    _slock = threading.Lock()

    def sidecar_writer(rec):
        with _slock:
            sidecar.write(json.dumps(rec, separators=(",", ":")) + "\n")
            sidecar.flush()

    gen = _build_local_nim_gen(model=str(args.model), sidecar_writer=sidecar_writer,
                               read_timeout_s=args.read_timeout_s)

    if not args.no_canary:
        print("  canary: 1 plain gen on the first dev target …", flush=True)
        ctext, _w = gen(G.build_plain_prompt_v1(dev[0].as_pilot_problem()), args.max_tokens, 0.0)
        ccode = extract_candidate_code_v1(response_text=ctext)
        print(f"    canary {len(ctext)}b -> code {len(ccode)}b (parses={bool(ccode.strip())})")
        if not ccode.strip():
            raise SystemExit("canary produced no code; aborting before bench spend.")

    # precompute plain + scaffold prompts (deterministic, no NIM)
    plan = []
    for ep in dev:
        prob = ep.as_pilot_problem()
        cls = G.target_family_ranking_v1(ep.statement, ep.samples)
        prio = G.prioritized_families_v1(cls)
        rr = G.retrieve_scaffolds_v1(target_short=ep.short_name, target_statement=ep.statement,
                                     target_family=cls.family, library=lib, R=args.R,
                                     candidate_families=prio)
        plan.append({"ep": ep, "prob": prob, "family": cls.family, "rr": rr,
                     "base_prompt": G.build_plain_prompt_v1(prob),
                     "scaf_prompt": G.build_scaffolded_prompt_v1(prob, rr.scaffolds)})

    t0 = time.time()
    # ----- Phase A: plain + scaffold i.i.d. generations (concurrent) -----
    jobs = []
    for ti, pl in enumerate(plan):
        for _s in range(args.K):
            jobs.append((ti, "base", pl["base_prompt"]))
            jobs.append((ti, "scaf", pl["scaf_prompt"]))

    def _run(job):
        ti, arm, prompt = job
        try:
            text, _w = gen(prompt, args.max_tokens, args.temperature)
            return ti, arm, extract_candidate_code_v1(response_text=text)
        except Exception as e:  # noqa: BLE001
            print(f"    [gen-fail t{ti} {arm}] {type(e).__name__}: {e}", flush=True)
            return ti, arm, ""

    gen_out: dict = {}
    n_calls = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for ti, arm, code in ex.map(_run, jobs):
            gen_out.setdefault((ti, arm), []).append(code)
            n_calls += 1
            if n_calls % 20 == 0:
                print(f"    … phaseA {n_calls}/{len(jobs)} ({time.time()-t0:.0f}s)", flush=True)
    print(f"  phaseA done: {n_calls} calls in {time.time()-t0:.0f}s", flush=True)

    # ----- Phase B: role-diverse search per target (concurrent over targets) -----
    rda_lock = threading.Lock()
    rda_out: dict = {}
    rda_calls = [0]

    def _run_rda(ti):
        pl = plan[ti]
        leak = _make_leak_check(pl["ep"], pl["prob"])
        o = R.run_role_diverse_search_v1(
            gen, pl["prob"], K=args.K, n_sketches=args.n_sketches,
            analyze_temp=args.analyze_temp, impl_temp=args.impl_temp,
            max_tokens=args.max_tokens, timeout_s=args.timeout_s, family=pl["family"],
            grade_secret=True, leakage_check=leak)
        with rda_lock:
            rda_out[ti] = o
            rda_calls[0] += o.n_calls
            print(f"    [rda {len(rda_out)}/{len(plan)}] {pl['ep'].short_name:24s} "
                  f"fam={pl['family']:16s} div={o.diversity['classify']:11s} "
                  f"rda4={int(o.committed_pass['RDA4'])} pool={int(o.pool_pass)} "
                  f"abstain4={int(o.abstained['RDA4'])} clean={int(o.leakage_clean)}", flush=True)
        return ti

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(_run_rda, range(len(plan))))
    n_calls += rda_calls[0]

    # ----- grade plain + scaffold; assemble per-target results -----
    results = []
    for ti, pl in enumerate(plan):
        ep, prob, rr = pl["ep"], pl["prob"], pl["rr"]
        base_pass, _bk, base_nparse = _pass_at_k(prob, gen_out.get((ti, "base"), []),
                                                 timeout_s=args.timeout_s)
        scaf_pass, _sk, _sn = _pass_at_k(prob, gen_out.get((ti, "scaf"), []),
                                         timeout_s=args.timeout_s)
        o = rda_out[ti]
        results.append(R.RdaDevBenchTargetResultV1(
            short_name=ep.short_name, family=pl["family"], baseline_pass=base_pass,
            scaffold_pass=scaf_pass, rda_committed_pass=o.committed_pass["RDA4"],
            rda_pool_pass=o.pool_pass, rda_abstained=o.abstained["RDA4"],
            diversity_real=(o.diversity["classify"] == "REAL"),
            leakage_clean=o.leakage_clean, failure_was_trivial=bool(base_nparse == 0)))
    wall = time.time() - t0
    sidecar.close()

    gate = R.apply_rda_dev_bench_earn_gate_v1(results)
    # per-variant load-bearing tally (which RDA stage carried wins / abstains)
    variant_pass = {v: sum(1 for ti in rda_out if rda_out[ti].committed_pass[v])
                    for v in R.RDA_VARIANTS}
    variant_abstain = {v: sum(1 for ti in rda_out if rda_out[ti].abstained[v])
                       for v in R.RDA_VARIANTS}
    verdict = {
        "schema": "coordpy.w128_hard_cluster_dev_bench.v1", "lane": "beta_hard_cluster_dev_bench",
        "verified_on": _dt.date.today().isoformat(), "model_id": str(args.model),
        "K": args.K, "n_sketches": args.n_sketches, "temperature": args.temperature,
        "analyze_temp": args.analyze_temp, "impl_temp": args.impl_temp,
        "timeout_s": args.timeout_s, "nim_calls": n_calls, "wall_s": round(wall, 1),
        "exposed_root": args.exposed_root, "hard_families": list(hard_families),
        "dev_family_counts": fam_counts, "hard_dev_target_cid": dev_cid,
        "teacher_corpus_cid": teacher_cid, "library": lib.summary(),
        "variant_pass": variant_pass, "variant_abstain": variant_abstain,
        "rda_pool_total_pass": gate.rda_pool_total_pass,
        "per_target": [r.to_dict() for r in results],
        "per_target_detail": {rda_out[ti].short_name: rda_out[ti].to_dict() for ti in rda_out},
        "earn_gate": gate.to_dict(), "dev_min_net_gain": R.DEV_MIN_NET_GAIN,
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    with open(os.path.join(out_run, "hard_cluster_dev_bench_verdict.json"), "w") as f:
        json.dump(verdict, f, indent=2, default=str)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "hard_cluster_dev_bench_verdict.json"), "w") as f:
        json.dump(verdict, f, indent=2, default=str)
    with open(os.path.join(OUT_DIR, "latest_run.txt"), "w") as f:
        f.write(os.path.basename(out_run))

    print()
    print(f"  baseline {gate.baseline_total_pass}/{gate.n_targets} | scaffold "
          f"{gate.scaffold_total_pass}/{gate.n_targets} | RDA4 {gate.rda_total_pass}/{gate.n_targets} "
          f"| pool {gate.rda_pool_total_pass}/{gate.n_targets}")
    print(f"  net_rda_gain={gate.net_rda_gain:+d} (unique={gate.rda_unique_solves} "
          f"regr={gate.rda_regressions}) net_vs_scaffold={gate.net_vs_scaffold:+d} "
          f"gain_families={list(gate.gain_families)} includes_named={gate.gain_includes_named_cluster}")
    print(f"  variant_pass={variant_pass} variant_abstain={variant_abstain}")
    print(f"  EARN GATE: {gate.verdict_label} (earned={gate.earned})")
    print(f"    {gate.rationale}")
    print(f"  nim_calls={n_calls} wall={wall:.0f}s -> {out_run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
