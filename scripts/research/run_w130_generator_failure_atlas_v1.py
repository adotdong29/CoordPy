"""W130 Lane α — generator-failure atlas builder ($0 NIM).

Reconstructs the FULL old W128/W129 candidate pool (plain U scaffold U rda) per hard-cluster
EXPOSED dev target from the stored W128 sidecar, grades every candidate with a mechanical
failure signature, cross-checks the OFFLINE accepted-algorithm reference (never model-facing),
and classifies each problem's dominant generator-failure mode (schema LOCKED in
``coordpy.generator_failure_atlas_v1`` before any result is interpreted).

The plain/scaffold arms issue K i.i.d. calls with an IDENTICAL prompt (same sha), so the
sidecar is bucketed into a LIST of responses per prompt sha (not last-wins) to recover all K
generations.  The rda arm's impl prompts are distinct per sketch (distinct sha).

$0 NIM — no model is called.  Emits results/w130/atlas/generator_failure_atlas_v1.json.
"""
from __future__ import annotations

import argparse
import collections
import datetime as _dt
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import coordpy.family_scaffold_generation_v1 as G  # noqa: E402
import coordpy.generator_failure_atlas_v1 as A  # noqa: E402
import coordpy.role_diverse_algorithm_search_v1 as R  # noqa: E402
from coordpy.icpc_reflexion_bench_v1 import extract_candidate_code_v1  # noqa: E402

OUT_DIR = os.path.join(ROOT, "results", "w130", "atlas")
DEFAULT_CALLS = ("results/w128/dev_bench/"
                 "w128_dev_bench_20260601T185815Z_fixed/dev_bench_calls.jsonl")
DEV_VERDICT = "results/w128/dev_bench/hard_cluster_dev_bench_verdict.json"


def _sha(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()


def load_sidecar_by_sha(path: str) -> dict:
    """sha256(prompt) -> [response_text, ...] (all K i.i.d. responses for that prompt)."""
    by_sha: dict[str, list] = collections.defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            by_sha[rec["prompt_sha256"]].append(rec["response_text"])
    return by_sha


def _surface_of(pkg_dir: str) -> str:
    # /tmp/w121_icpc/<surface>/<problem>
    parts = os.path.normpath(pkg_dir).split(os.sep)
    return parts[-2] if len(parts) >= 2 else "?"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exposed-root", default="/tmp/w121_icpc")
    ap.add_argument("--calls", default=DEFAULT_CALLS)
    ap.add_argument("--families", default=",".join(R.NON_SCAFFOLDABLE_FAMILIES))
    ap.add_argument("--R", type=int, default=2)
    ap.add_argument("--n-sketches", type=int, default=4)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--timeout-s", type=float, default=6.0, help="secret-grading per-case cap")
    args = ap.parse_args()

    hard_families = tuple(f.strip() for f in args.families.split(",") if f.strip())
    probs = G.load_exposed_problems_v1(args.exposed_root)
    fam_of = {p.short_name: G.target_family_ranking_v1(p.statement, p.samples).family
              for p in probs}
    dev = sorted([p for p in probs if fam_of[p.short_name] in hard_families],
                 key=lambda p: p.short_name)
    teacher = [p for p in probs if p.short_name not in {d.short_name for d in dev}]
    lib = G.build_scaffold_library_v1(teacher)
    calls_path = args.calls if os.path.isabs(args.calls) else os.path.join(ROOT, args.calls)
    by_sha = load_sidecar_by_sha(calls_path)
    print(f"  exposed={len(probs)} hard-dev={len(dev)} sidecar_prompts={len(by_sha)} "
          f"sidecar_calls={sum(len(v) for v in by_sha.values())}")

    # committed status under W128 (rda_committed_pass) for cross-reference
    committed = {}
    try:
        dv = json.load(open(os.path.join(ROOT, DEV_VERDICT)))
        committed = {t["short_name"]: bool(t["rda_committed_pass"]) for t in dv["per_target"]}
    except Exception:  # noqa: BLE001
        pass

    records = []
    n_replay_miss = 0
    for ep in dev:
        prob = ep.as_pilot_problem()
        cls = G.target_family_ranking_v1(ep.statement, ep.samples)
        prio = G.prioritized_families_v1(cls)
        rr = G.retrieve_scaffolds_v1(target_short=ep.short_name, target_statement=ep.statement,
                                     target_family=cls.family, library=lib, R=args.R,
                                     candidate_families=prio)
        plain_prompt = G.build_plain_prompt_v1(prob)
        scaf_prompt = G.build_scaffolded_prompt_v1(prob, rr.scaffolds)
        analyze_prompt = R.build_analyze_prompt_v1(prob, n_sketches=args.n_sketches)

        cands = []
        # plain arm
        for i, resp in enumerate(by_sha.get(_sha(plain_prompt), [])):
            code = extract_candidate_code_v1(response_text=resp)
            cands.append(A.classify_candidate_v1(prob, code, label=f"plain{i}", arm="plain",
                                                 secret_timeout_s=args.timeout_s))
        # scaffold arm
        for i, resp in enumerate(by_sha.get(_sha(scaf_prompt), [])):
            code = extract_candidate_code_v1(response_text=resp)
            cands.append(A.classify_candidate_v1(prob, code, label=f"scaf{i}", arm="scaffold",
                                                 secret_timeout_s=args.timeout_s))
        # rda arm: replay analyze -> parse sketches -> replay each impl
        sketches = []
        a_list = by_sha.get(_sha(analyze_prompt), [])
        if a_list:
            arts = R.parse_role_artifacts_v1(a_list[0], n_sketches=args.n_sketches)
            sketches = list(arts.sketches)
            n_impl = max(1, args.K - 1)
            sk = sketches or [R.SketchV1("A", "direct", "Implement the direct algorithm.")]
            while len(sk) < n_impl:
                sk.append(sk[len(sk) % len(sketches or sk)])
            for i in range(n_impl):
                imp_prompt = R.build_implement_prompt_v1(prob, arts.spec, sk[i])
                rl = by_sha.get(_sha(imp_prompt), [])
                if not rl:
                    n_replay_miss += 1
                    continue
                code = extract_candidate_code_v1(response_text=rl[0])
                cands.append(A.classify_candidate_v1(prob, code, label=f"{sk[i].label}{i}",
                                                     arm="rda", secret_timeout_s=args.timeout_s))
        rec = A.build_problem_record_v1(
            short_name=ep.short_name, family=fam_of[ep.short_name],
            surface=_surface_of(ep.pkg_dir), date=prob.contest_date,
            candidates=cands, sketches=sketches, accepted_codes=list(ep.accepted_codes),
            committed_w128=committed.get(ep.short_name))
        records.append(rec)
        c = rec.counts
        print(f"    [{ep.short_name:22s}] {fam_of[ep.short_name]:15s} "
              f"cands={rec.n_candidates:2d} pool={int(rec.pool_bearing)} "
              f"pubSurv={rec.n_public_survivors}(c{rec.n_pub_correct}/w{rec.n_pub_wrong}) "
              f"WA={c['WRONG_ANSWER']} TLE={c['TLE']} CR={c['CRASH']} PE={c['PARSE_ERR']} "
              f"HF={c['HIDDEN_FAIL']} adm={int(rec.sketch_admissible)} "
              f"-> {rec.dominant_mode}")

    atlas = A.build_atlas_v1(records)
    os.makedirs(OUT_DIR, exist_ok=True)
    out = {
        "schema": "coordpy.w130_generator_failure_atlas.v1", "lane": "alpha_generator_atlas",
        "nim_spend": 0, "verified_on": _dt.date.today().isoformat(),
        "exposed_root": args.exposed_root, "calls": args.calls,
        "replay_misses": n_replay_miss,
        "taxonomy": list(A.GENERATOR_FAILURE_MODES),
        "generator_fixable_modes": sorted(A.GENERATOR_FIXABLE_MODES),
        "atlas": atlas.to_dict(),
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    with open(os.path.join(OUT_DIR, "generator_failure_atlas_v1.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)

    print()
    print(f"  === GENERATOR-FAILURE ATLAS (n={atlas.n_problems}) ===")
    print(f"  pool_bearing={len(atlas.pool_bearing)} {atlas.pool_bearing}")
    print(f"  pool_dead={len(atlas.pool_dead)} {atlas.pool_dead}")
    print(f"  mode_histogram={atlas.mode_histogram}")
    print(f"  selector_fixable (W129 domain)={atlas.selector_fixable}")
    print(f"  GENERATOR-fixable pool-dead={atlas.generator_fixable}")
    print(f"  capability_failures (NO_SKETCH)={atlas.capability_failures}")
    print(f"  top generator modes (pool-dead)={atlas.top_generator_modes}")
    print(f"  replay_misses={n_replay_miss}")
    print(f"  -> {os.path.join(OUT_DIR, 'generator_failure_atlas_v1.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
