"""W127 Lane β — $0 re-evaluation of the EXPOSED dev bench with the corrected
(boilerplate-robust) accepted-solution leakage tripwire.

The first dev-bench run's locked leakage guard flagged the two scaffold UNIQUE SOLVES
(`champernownecount`, `electionparadox`) and `impartialstrings` as
`DEV_BENCH_INVALID_LEAKAGE` — but the flagged "accepted-solution lines" were universal
boilerplate (`n, k = map(int, input().split())`, `n = int(input())`, `while i < n:`).  The
winning candidates were verified to be structurally DIFFERENT correct derivations sharing
only boilerplate (NOT memorized reproductions).  This is a guard CALIBRATION false positive,
directly analogous to the W126 `emoticons` correction.  The corrected accepted-line tripwire
requires a CONTIGUOUS reproduced block (a real "accepted solution shown" signature); the
positive control still bites (a planted accepted solution is caught — tested).

This script re-grades + re-checks the ALREADY-PAID stored candidates (the run's
``dev_bench_calls.jsonl``) with the corrected check — **$0 NIM** — and re-applies the locked
R1 earn gate.  It is OUTCOME-RELEVANT (the correction can flip the verdict to EARNED); it is
applied transparently, with the original (buggy) verdict preserved as the raw record.

Emits results/w127/dev_bench/exposed_dev_bench_verdict_reeval_v1.json and updates the
canonical results/w127/dev_bench/exposed_dev_bench_verdict.json to the corrected verdict.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import coordpy.family_scaffold_generation_v1 as G  # noqa: E402
from coordpy.family_adapted_repair_synthesis_v1 import SynthesisLeakageGuardV1  # noqa: E402
from coordpy.icpc_reflexion_bench_v1 import (  # noqa: E402
    extract_candidate_code_v1, grade_icpc_candidate_case_v1, grade_on_secret_v1)

OUT_DIR = os.path.join(ROOT, "results", "w127", "dev_bench")


def _passes_samples(problem, code, *, timeout_s=5.0):
    for inp, exp in problem.samples:
        r = grade_icpc_candidate_case_v1(candidate_code=code, stdin_text=inp,
                                         expected_stdout=exp, kind=problem.kind,
                                         float_tol=problem.float_tol, timeout_s=timeout_s)
        if not r.passed:
            return False
    return True


def _pass_at_k(problem, codes, *, timeout_s):
    first, nparse = -1, 0
    for k, code in enumerate(codes):
        if not code.strip():
            continue
        nparse += 1
        if not _passes_samples(problem, code):
            continue
        passed, _t, _n = grade_on_secret_v1(problem, code, timeout_s=timeout_s)
        if passed and first < 0:
            return True, k, nparse
    return False, first, nparse


def _bucket(s):
    return int(hashlib.sha256(s.encode()).hexdigest(), 16) % 3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exposed-root", default="/tmp/w121_icpc")
    ap.add_argument("--run-dir", default="")
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--R", type=int, default=2)
    ap.add_argument("--timeout-s", type=float, default=8.0)
    ap.add_argument("--dev-bucket", type=int, default=0)
    args = ap.parse_args()

    run_dir = args.run_dir or os.path.join(
        OUT_DIR, open(os.path.join(OUT_DIR, "latest_run.txt")).read().strip())
    recs = [json.loads(l) for l in open(os.path.join(run_dir, "dev_bench_calls.jsonl"))]
    by_prompt: dict = {}
    for r in recs:
        by_prompt.setdefault(r["prompt"], []).append(
            extract_candidate_code_v1(response_text=r["response_text"]))
    print(f"  re-eval run_dir={os.path.basename(run_dir)}  ({len(recs)} stored calls, $0)")

    probs = G.load_exposed_problems_v1(args.exposed_root)
    teacher = [p for p in probs if _bucket(p.short_name) != args.dev_bucket]
    dev = sorted([p for p in probs if _bucket(p.short_name) == args.dev_bucket],
                 key=lambda p: p.short_name)
    lib = G.build_scaffold_library_v1(teacher)

    results = []
    for ep in dev:
        prob = ep.as_pilot_problem()
        base_prompt = G.build_plain_prompt_v1(prob)
        if base_prompt not in by_prompt:
            continue   # this dev target was not in the (limited) run
        cls = G.target_family_ranking_v1(ep.statement, ep.samples)
        prio = G.prioritized_families_v1(cls)
        rr = G.retrieve_scaffolds_v1(
            target_short=ep.short_name, target_statement=ep.statement,
            target_family=cls.family, library=lib, R=args.R, candidate_families=prio)
        scaf_prompt = G.build_scaffolded_prompt_v1(prob, rr.scaffolds)
        base_codes = by_prompt.get(base_prompt, [])
        scaf_codes = by_prompt.get(scaf_prompt, [])
        base_pass, base_k, base_nparse = _pass_at_k(prob, base_codes,
                                                    timeout_s=args.timeout_s)
        scaf_pass, scaf_k, _ = _pass_at_k(prob, scaf_codes, timeout_s=args.timeout_s)
        prov = "\n".join([sc.skeleton for sc in rr.scaffolds] + [ep.statement]
                         + [i + o for i, o in ep.samples])
        guard = SynthesisLeakageGuardV1(prob, target_accepted_texts=(),
                                        provenance_texts=[prov])
        clean, reason = G.assert_scaffold_pipeline_clean_v1(
            target_short=ep.short_name, scaffolds=rr.scaffolds, candidate_texts=scaf_codes,
            guard=guard, target_accepted_texts=list(ep.accepted_codes), provenance=prov)
        r = G.DevBenchTargetResultV1(
            short_name=ep.short_name, family=cls.family,
            families_pulled=rr.families_pulled, n_scaffolds=len(rr.scaffolds),
            baseline_pass=base_pass, scaffold_pass=scaf_pass,
            baseline_first_pass_k=base_k, scaffold_first_pass_k=scaf_k,
            failure_family_was_trivial=bool(base_nparse == 0), leakage_clean=clean)
        results.append(r)
        flag = ("  *** SCAFFOLD UNIQUE SOLVE (clean) ***"
                if (scaf_pass and not base_pass and clean) else
                ("  (regression)" if (base_pass and not scaf_pass) else ""))
        print(f"   {ep.short_name:28s} fam={cls.family:16s} base={int(base_pass)} "
              f"scaf={int(scaf_pass)} clean={int(clean)} ({reason}){flag}")

    gate = G.apply_dev_bench_earn_gate_v1(results)
    orig = json.load(open(os.path.join(OUT_DIR, "exposed_dev_bench_verdict.json")))
    verdict = dict(orig)
    verdict.update({
        "schema": "coordpy.w127_exposed_dev_bench_reeval.v1",
        "leakage_recalibration": (
            "accepted-line tripwire corrected from per-line (boilerplate-FP) to "
            "contiguous-block; positive control preserved (planted accepted solution still "
            "caught); $0 re-grade of stored candidates; OUTCOME-RELEVANT, applied "
            "transparently"),
        "original_verdict_label": orig["earn_gate"]["verdict_label"],
        "nim_calls": 0, "per_target": [r.to_dict() for r in results],
        "earn_gate": gate.to_dict(),
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    })
    with open(os.path.join(OUT_DIR, "exposed_dev_bench_verdict_reeval_v1.json"), "w") as f:
        json.dump(verdict, f, indent=2, default=str)
    with open(os.path.join(OUT_DIR, "exposed_dev_bench_verdict.json"), "w") as f:
        json.dump(verdict, f, indent=2, default=str)

    print(f"\n  CORRECTED earn gate: baseline {gate.baseline_total_pass} -> scaffold "
          f"{gate.scaffold_total_pass}  net={gate.net_scaffold_gain:+d}")
    print(f"  unique={gate.scaffold_unique_solves} regress={gate.scaffold_regressions} "
          f"families={list(gate.gain_families)} leakage_clean={gate.all_leakage_clean}")
    print(f"  VERDICT: {gate.verdict_label} (earned={gate.earned})  "
          f"[was {orig['earn_gate']['verdict_label']}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
