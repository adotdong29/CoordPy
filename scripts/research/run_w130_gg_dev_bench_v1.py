"""W130 Lane β — stronger same-budget GENERATOR dev bench (NIM; EXPOSED dev spend ALLOWED for
generator validation — RUNBOOK_W130 § 7).

Runs the GG1-GG4 + GGLEAD generator slate on the SAME 11 hard-cluster EXPOSED dev targets as
W128/W129, at MATCHED K=5 budget, with the W129 selector held FIXED downstream
(``public_signal_selection_oracle_v1.select_so_v1`` SOLEAD, NIM-free).  Applies the R2W earn gate
(>= 2 NEW pool solves absent from the old W128/W129 pool, spanning >= 2 families/atlas-modes,
realness-REAL + leakage-clean).  Runs the stored-regression trio guard (blueberrywaffle / pawnshop
/ sunandmoon).  Emits results/w130/dev_bench/gg_dev_bench_verdict.json.
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
import coordpy.stronger_generator_slate_v1 as S  # noqa: E402
from coordpy.icpc_reflexion_bench_v1 import extract_candidate_code_v1  # noqa: E402
from scripts.run_w127_exposed_dev_bench_v1 import _build_local_nim_gen  # noqa: E402

W130_TARGET_MODEL = "meta/llama-4-maverick-17b-128e-instruct"
OUT_DIR = os.path.join(ROOT, "results", "w130", "dev_bench")
ATLAS = "results/w130/atlas/generator_failure_atlas_v1.json"
OLD_POOL_SOLVED = ("pawnshop", "sunandmoon", "blueberrywaffle")  # W128/W129 pool secret-solvers
TRIO = ("blueberrywaffle", "pawnshop", "sunandmoon")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exposed-root", default="/tmp/w121_icpc")
    ap.add_argument("--model", default=W130_TARGET_MODEL)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--n-sketches", type=int, default=4)
    ap.add_argument("--analyze-temp", type=float, default=0.5)
    ap.add_argument("--impl-temp", type=float, default=0.2)
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--timeout-s", type=float, default=5.0, help="public/derived exec cap")
    ap.add_argument("--secret-timeout-s", type=float, default=8.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--read-timeout-s", type=float, default=90.0)
    ap.add_argument("--families", default=",".join(R.NON_SCAFFOLDABLE_FAMILIES))
    ap.add_argument("--arms", default=",".join(S.GG_VARIANTS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default="", help="comma-separated short-names to restrict to")
    ap.add_argument("--no-canary", action="store_true")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    hard_families = tuple(f.strip() for f in args.families.split(",") if f.strip())
    probs = G.load_exposed_problems_v1(args.exposed_root)
    fam_of = {p.short_name: G.target_family_ranking_v1(p.statement, p.samples).family
              for p in probs}
    dev = sorted([p for p in probs if fam_of[p.short_name] in hard_families],
                 key=lambda p: p.short_name)
    if args.only:
        keep = {s.strip() for s in args.only.split(",")}
        dev = [p for p in dev if p.short_name in keep]
    if args.limit:
        dev = dev[:args.limit]
    teacher = [p for p in probs if p.short_name not in {d.short_name for d in dev}]
    lib = G.build_scaffold_library_v1(teacher)
    dev_cid = hashlib.sha256(json.dumps(sorted(d.short_name for d in dev)).encode()).hexdigest()
    teacher_cid = hashlib.sha256(
        json.dumps(sorted(t.short_name for t in teacher)).encode()).hexdigest()
    assert {d.short_name for d in dev}.isdisjoint({t.short_name for t in teacher})

    # atlas modes (for the earn gate's span check)
    atlas_mode_of = {}
    try:
        ad = json.load(open(os.path.join(ROOT, ATLAS)))
        atlas_mode_of = {r["short_name"]: r["dominant_mode"] for r in ad["atlas"]["records"]}
    except Exception:  # noqa: BLE001
        pass

    print(f"  exposed={len(probs)} hard-dev={len(dev)} teacher={len(teacher)} arms={arms}")
    print(f"  hard_dev_target_cid={dev_cid[:16]}…  teacher_cid={teacher_cid[:16]}…")

    run_id = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    lbl = (f"_{args.label}" if args.label else "")
    out_run = os.path.join(OUT_DIR, f"w130_gg_dev_bench_{run_id}{lbl}")
    os.makedirs(out_run, exist_ok=True)
    sidecar = open(os.path.join(out_run, "gg_dev_bench_calls.jsonl"), "w")
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

    t0 = time.time()
    jobs = [(ep, arm) for ep in dev for arm in arms]
    outcomes_by_problem: dict = {ep.short_name: {} for ep in dev}
    counts = {"done": 0, "calls": 0}
    clock = threading.Lock()

    def _run(job):
        ep, arm = job
        prob = ep.as_pilot_problem()
        kw = dict(K=args.K, n_sketches=args.n_sketches, analyze_temp=args.analyze_temp,
                  impl_temp=args.impl_temp, max_tokens=args.max_tokens, timeout_s=args.timeout_s,
                  accepted_codes=tuple(ep.accepted_codes))
        try:
            if arm == "GG3":
                kw.pop("timeout_s", None)
                o = S.run_gg3_v1(gen, prob, family=fam_of[ep.short_name], library=lib, **kw)
            else:
                o = S.ARM_RUNNERS[arm](gen, prob, **kw)
        except Exception as e:  # noqa: BLE001
            print(f"    [FAIL {ep.short_name} {arm}] {type(e).__name__}: {e}", flush=True)
            return ep.short_name, arm, None
        return ep.short_name, arm, o

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for short, arm, o in ex.map(_run, jobs):
            if o is None:
                continue
            outcomes_by_problem[short][arm] = o
            with clock:
                counts["done"] += 1
                counts["calls"] += o.n_calls
            newp = (o.pool_pass and short not in OLD_POOL_SOLVED)
            print(f"    [{counts['done']}/{len(jobs)}] {short:20s} {arm:7s} "
                  f"calls={o.n_calls} pool={int(o.pool_pass)} commit={int(o.committed_pass)} "
                  f"{'NEW!' if newp else ''} sel={o.selector_branch[:22]} "
                  f"clean={int(o.leakage_clean)} ({time.time()-t0:.0f}s)", flush=True)

    wall = time.time() - t0
    sidecar.close()

    gate = S.apply_gg_dev_bench_earn_gate_v1(
        outcomes_by_problem, old_pool_solved=OLD_POOL_SOLVED, family_of=fam_of,
        atlas_mode_of=atlas_mode_of)

    # stored-regression trio guard (the fixed selector must not regress)
    trio = {}
    for name in TRIO:
        d = outcomes_by_problem.get(name, {})
        trio[name] = {arm: {"committed_label": o.committed_label, "committed_pass": o.committed_pass,
                            "selector_branch": o.selector_branch, "pool_pass": o.pool_pass}
                      for arm, o in d.items()}

    verdict = {
        "schema": "coordpy.w130_gg_dev_bench.v1", "lane": "beta_gg_generator_dev_bench",
        "verified_on": _dt.date.today().isoformat(), "model_id": str(args.model), "K": args.K,
        "n_sketches": args.n_sketches, "analyze_temp": args.analyze_temp,
        "impl_temp": args.impl_temp, "arms": arms, "nim_calls": counts["calls"],
        "wall_s": round(wall, 1), "exposed_root": args.exposed_root,
        "hard_families": list(hard_families), "hard_dev_target_cid": dev_cid,
        "teacher_corpus_cid": teacher_cid, "old_pool_solved": list(OLD_POOL_SOLVED),
        "controls": {"gg1_gate": S.gg1_gate_control_v1(), "gg2_rewrite": S.gg2_rewrite_control_v1(),
                     "hosted_controller_mining": S.examine_hosted_controller_applicability_v1()},
        "stored_regression_trio": trio,
        "per_problem": {p: {a: o.to_dict() for a, o in d.items()}
                        for p, d in outcomes_by_problem.items()},
        "earn_gate": gate.to_dict(),
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    with open(os.path.join(out_run, "gg_dev_bench_verdict.json"), "w") as f:
        json.dump(verdict, f, indent=2, default=str)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "gg_dev_bench_verdict.json"), "w") as f:
        json.dump(verdict, f, indent=2, default=str)
    with open(os.path.join(OUT_DIR, "latest_run.txt"), "w") as f:
        f.write(os.path.basename(out_run))

    print()
    print(f"  === GG DEV BENCH ({len(dev)} targets × {len(arms)} arms) ===")
    for arm in arms:
        nps = gate.per_arm_new_pool_solves.get(arm, [])
        nc = gate.per_arm_new_committed.get(arm, [])
        print(f"    {arm:7s} new_pool_solves={len(nps)} {nps}  new_committed={len(nc)} {nc}")
    print(f"  best_arm={gate.best_arm} best_new={gate.best_new_count} "
          f"families={gate.new_solve_families} modes={gate.new_solve_modes}")
    print(f"  EARN GATE: {gate.verdict_label} (earned={gate.earned})")
    print(f"    {gate.rationale}")
    print(f"  controls: gg1_gate={verdict['controls']['gg1_gate']['passes']} "
          f"gg2_rewrite={verdict['controls']['gg2_rewrite']['passes']}")
    print(f"  nim_calls={counts['calls']} wall={wall:.0f}s -> {out_run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
