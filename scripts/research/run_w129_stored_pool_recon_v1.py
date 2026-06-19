"""W129 Lane α RECON — $0-NIM reconstruction + per-candidate grading of the STORED W128
hard-cluster dev-bench pools.

The W128 selection cap was localized to the verification-based SELECTION layer: the pool
reached 3/11 but RDA4 committed only 2/11 (net +0 = +1 ``blueberrywaffle`` − 1 ``pawnshop``).
Before building (or buying) anything, W129 must answer a sharp, falsifiable, $0 question:

    For the W128 miss problems — especially ``pawnshop`` where the pool HAD a passing program
    (``pool_first_label``) but RDA4 committed a FAILING one — are the hidden-correct and the
    hidden-wrong public-survivors SEPARABLE by ANY public-derivable signal (invariants,
    differential testing vs a trusted/brute-force sketch, harder derived cases)?  Or do they
    agree on every public-derivable input (selection genuinely blind)?

This script reconstructs the EXACT W128 candidate pools by REPLAYING the stored generations
(keyed by ``prompt_sha256`` — the W128 sidecar stored full prompt+response for every call),
re-parses the role artifacts, extracts every candidate (A0/B1/C2/D3), and grades each on the
PUBLIC samples AND the official SECRET cases.  $0 NIM.  No model is called.  The accepted
solution is never shown to any model path (it is not used at all here — this is pure grading).

Outputs ``results/w129/recon/stored_pool_recon_v1.json`` + a per-target table.
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
import coordpy.role_diverse_algorithm_search_v1 as R  # noqa: E402
from coordpy.coordpy_icpc_battlefield_v1 import grade_icpc_candidate_case_v1  # noqa: E402
from coordpy.icpc_reflexion_bench_v1 import (  # noqa: E402
    extract_candidate_code_v1, grade_on_secret_v1)

OUT_DIR = os.path.join(ROOT, "results", "w129", "recon")


def make_replay_gen(calls_paths):
    """Replay gen: sha256(prompt) -> stored response_text. $0 NIM, deterministic."""
    by_sha: dict[str, str] = {}
    n = 0
    for p in calls_paths:
        if not os.path.exists(p):
            continue
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                by_sha[rec["prompt_sha256"]] = rec["response_text"]
                n += 1
    misses = {"n": 0}

    def gen(prompt, max_tokens, temperature):  # noqa: ARG001
        key = hashlib.sha256(prompt.encode()).hexdigest()
        if key not in by_sha:
            misses["n"] += 1
            raise KeyError(f"REPLAY_MISS sha={key[:16]} len={len(prompt)}")
        return by_sha[key], 0
    gen._n_loaded = n  # type: ignore[attr-defined]
    gen._by_sha = by_sha  # type: ignore[attr-defined]
    gen._misses = misses  # type: ignore[attr-defined]
    return gen


def _public_pass(prob, code, *, timeout_s=5.0) -> bool:
    if not code or not code.strip():
        return False
    for inp, exp in prob.samples:
        r = grade_icpc_candidate_case_v1(
            candidate_code=code, stdin_text=inp, expected_stdout=exp,
            kind=prob.kind, float_tol=prob.float_tol, timeout_s=timeout_s)
        if not r.passed:
            return False
    return True


def _secret_pass(prob, code, *, timeout_s=10.0) -> bool:
    if not code or not code.strip():
        return False
    try:
        ok, _tail, _n = grade_on_secret_v1(prob, code, timeout_s=timeout_s)
        return bool(ok)
    except Exception:  # noqa: BLE001
        return False


def reconstruct_target(gen, ep, *, K=5, n_sketches=4, max_tokens=1536,
                       analyze_temp=0.5, impl_temp=0.2):
    """Replay ANALYZE + IMPLEMENT for one target -> (artifacts, [CandidateImplV1])."""
    prob = ep.as_pilot_problem()
    n_impl = max(1, K - 1)
    n_sketches = min(n_sketches, n_impl)
    a_text, _ = gen(R.build_analyze_prompt_v1(prob, n_sketches=n_sketches),
                    max_tokens, analyze_temp)
    arts = R.parse_role_artifacts_v1(a_text, n_sketches=n_sketches)
    sketches = list(arts.sketches)
    if not sketches:
        sketches = [R.SketchV1("A", "direct", "Implement the most direct correct algorithm.")]
    while len(sketches) < n_impl:
        sketches.append(sketches[len(sketches) % len(arts.sketches or sketches)])
    impls = []
    for i in range(n_impl):
        sk = sketches[i]
        text, _ = gen(R.build_implement_prompt_v1(prob, arts.spec, sk), max_tokens, impl_temp)
        code = extract_candidate_code_v1(response_text=text)
        impls.append(R.CandidateImplV1(f"{sk.label}{i}", code, R._parses(code)))
    return prob, arts, impls


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exposed-root", default="/tmp/w121_icpc")
    ap.add_argument("--calls", default="results/w128/dev_bench/"
                    "w128_dev_bench_20260601T185815Z_fixed/dev_bench_calls.jsonl")
    ap.add_argument("--families", default=",".join(R.NON_SCAFFOLDABLE_FAMILIES))
    ap.add_argument("--timeout-s", type=float, default=10.0)
    args = ap.parse_args()

    hard_families = tuple(f.strip() for f in args.families.split(",") if f.strip())
    probs = G.load_exposed_problems_v1(args.exposed_root)
    fam_of = {p.short_name: G.target_family_ranking_v1(p.statement, p.samples).family
              for p in probs}
    dev = sorted([p for p in probs if fam_of[p.short_name] in hard_families],
                 key=lambda p: p.short_name)
    print(f"  exposed={len(probs)} hard-dev={len(dev)}")

    calls_paths = [os.path.join(ROOT, args.calls)] if not os.path.isabs(args.calls) else [args.calls]
    gen = make_replay_gen(calls_paths)
    print(f"  replay calls loaded: {gen._n_loaded}")

    per_target = []
    sep_summary = {"pawnshop": None, "blueberrywaffle": None, "sunandmoon": None}
    for ep in dev:
        try:
            prob, arts, impls = reconstruct_target(gen, ep)
        except KeyError as e:
            print(f"    [{ep.short_name}] REPLAY MISS: {e}")
            per_target.append({"short_name": ep.short_name, "family": fam_of[ep.short_name],
                               "replay_miss": True})
            continue
        derived = [inp for inp, _e in arts.counterexamples]
        # grade each candidate
        cands = []
        for im in impls:
            pub = _public_pass(prob, im.code, timeout_s=5.0) if im.parses else False
            sec = _secret_pass(prob, im.code, timeout_s=args.timeout_s) if im.parses else False
            # behavior signature on derived cases (candidate-vs-candidate agreement axis)
            sig = []
            for inp in derived:
                out, _dig = R._run_capture_stdout_v1(im.code, inp, timeout_s=5.0)
                sig.append(out)
            cands.append({"label": im.label, "parses": im.parses, "public_pass": pub,
                          "secret_pass": sec, "code_sha": R._sha(im.code)[:16],
                          "sig": sig, "code_len": len(im.code)})
        pub_survivors = [c for c in cands if c["public_pass"]]
        pub_correct = [c for c in pub_survivors if c["secret_pass"]]
        pub_wrong = [c for c in pub_survivors if not c["secret_pass"]]
        pool_pass = any(c["secret_pass"] for c in cands)
        # SEPARABILITY: among public-survivors, is there >=1 correct AND >=1 wrong, and do
        # they DIFFER on any derived case? (a public-derivable separating signal exists)
        sep_on_derived = False
        if pub_correct and pub_wrong:
            for cc in pub_correct:
                for cw in pub_wrong:
                    if cc["sig"] != cw["sig"]:
                        sep_on_derived = True
        # invariant signal availability (artifacts produced checkable invariants?)
        n_inv = len(arts.invariants)
        # brute-force / reference sketch present? (a sketch whose name/outline names brute force)
        bf_sketch = any(("brute" in (s.approach_name + s.outline).lower()
                         or "exhaust" in (s.approach_name + s.outline).lower()
                         or "naive" in (s.approach_name + s.outline).lower())
                        for s in arts.sketches)
        rec = {
            "short_name": ep.short_name, "family": fam_of[ep.short_name],
            "n_impls": len(impls), "n_parse": sum(1 for c in cands if c["parses"]),
            "n_public_survivors": len(pub_survivors),
            "n_pub_correct": len(pub_correct), "n_pub_wrong": len(pub_wrong),
            "pool_pass": pool_pass,
            "pool_secret_labels": [c["label"] for c in cands if c["secret_pass"]],
            "public_survivor_labels": [c["label"] for c in pub_survivors],
            "n_derived": len(derived), "n_invariants": n_inv, "bf_sketch": bf_sketch,
            "separable_on_derived": sep_on_derived,
            "candidates": cands,
        }
        per_target.append(rec)
        if ep.short_name in sep_summary:
            sep_summary[ep.short_name] = {
                "pub_correct": [c["label"] for c in pub_correct],
                "pub_wrong": [c["label"] for c in pub_wrong],
                "separable_on_derived": sep_on_derived,
                "n_derived": len(derived), "n_invariants": n_inv, "bf_sketch": bf_sketch,
                "pool_secret_labels": rec["pool_secret_labels"]}
        print(f"    [{ep.short_name:22s}] fam={fam_of[ep.short_name]:15s} "
              f"parse={rec['n_parse']} pub_surv={rec['n_public_survivors']} "
              f"(correct={rec['n_pub_correct']} wrong={rec['n_pub_wrong']}) "
              f"pool={int(pool_pass)}{rec['pool_secret_labels']} "
              f"sep_derived={int(sep_on_derived)} inv={n_inv} bf={int(bf_sketch)}")

    os.makedirs(OUT_DIR, exist_ok=True)
    out = {
        "schema": "coordpy.w129_stored_pool_recon.v1", "lane": "alpha_recon", "nim_spend": 0,
        "verified_on": _dt.date.today().isoformat(),
        "exposed_root": args.exposed_root, "calls": args.calls,
        "n_targets": len([r for r in per_target if not r.get("replay_miss")]),
        "miss_pattern_summary": sep_summary,
        "per_target": per_target,
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    with open(os.path.join(OUT_DIR, "stored_pool_recon_v1.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print()
    print("  === MISS-PATTERN SEPARABILITY ===")
    for k, v in sep_summary.items():
        print(f"    {k:18s}: {v}")
    print(f"  -> {os.path.join(OUT_DIR, 'stored_pool_recon_v1.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
