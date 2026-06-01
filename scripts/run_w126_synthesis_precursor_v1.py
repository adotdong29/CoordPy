"""W126 Lane α/β — $0 family-adapted repair-synthesis precursor on the resistant ICPC field.

Reuses the EXACT W125 chain (W120 resistant 30-slice, CID ``01bf9ef869a56e20``; 330
already-paid Maverick generations) + the W126 grade cache to target the UNIFORMLY-UNSOLVED
resistant problems (the 22 with ZERO secret-passing generations across all 11 — the sharp
precursor slice, RUNBOOK_W126 § 8).

For each unsolved problem it runs the new-trajectory synthesis slate (S1 splice / S2
digest-repair / S3 motif-harden / S-CONS output-consensus dispatcher) under the strict
no-leakage guard, grades candidates on the OFFICIAL secret + sample cases, and measures:

* oracle ceiling  — does ANY synthesized trajectory pass secret (latent headroom)?
* blind P1         — does a HIDDEN-TEST-BLIND committed trajectory pass secret (the
                     pilot-earning metric)?
* P2               — do the blind wins span >= 2 distinct families?

Then applies the locked precursor earn gate.  $0 NIM.  Emits
results/w126/lane_alpha_beta/synthesis_precursor_verdict.json.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from coordpy.coordpy_icpc_battlefield_v1 import (  # noqa: E402
    classify_battlefield_listing_v1, core_slice_cid_v1, select_battlefield_core_slice_v1)
import coordpy.controller_native_code_mechanism_v1 as M  # noqa: E402
import coordpy.family_adapted_repair_synthesis_v1 as S  # noqa: E402
from scripts.run_w120_icpc_pilot import load_pilot_problems  # noqa: E402

EXPECTED_SLICE_CID_30 = "01bf9ef869a56e20"
CORPUS_DIR = os.path.join(ROOT, "results", "w120", "icpc_pilot")
CACHE = os.path.join(ROOT, "results", "w126", "cache", "grade_cache_v1.json")
OUT_DIR = os.path.join(ROOT, "results", "w126", "lane_alpha_beta")


def _latest_corpus() -> str:
    name = open(os.path.join(CORPUS_DIR, "latest_run.txt")).read().strip()
    return os.path.join(CORPUS_DIR, name, "icpc_reflexion_calls.jsonl")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout-s", type=float, default=8.0)
    ap.add_argument("--b-syn", type=int, default=5, help="blind committed synth budget")
    ap.add_argument("--consensus-max-cases", type=int, default=0,
                    help="0 = ALL secret cases (honest all-or-nothing); >0 caps for speed")
    ap.add_argument("--limit", type=int, default=0, help="debug: limit #unsolved probed")
    args = ap.parse_args()

    full = classify_battlefield_listing_v1()
    slice30 = select_battlefield_core_slice_v1(full, n_problems=30)
    slice_cid = core_slice_cid_v1(slice30)
    if not slice_cid.startswith(EXPECTED_SLICE_CID_30):
        raise SystemExit(f"slice CID {slice_cid[:16]} != {EXPECTED_SLICE_CID_30}; refusing.")
    problems = load_pilot_problems(list(slice30))
    recs = [json.loads(l) for l in open(_latest_corpus())]
    if len(recs) != 11 * len(problems):
        raise SystemExit(f"corpus {len(recs)} != {11*len(problems)}; refusing.")
    pools = [M.build_pool_from_records(recs[i * 11:i * 11 + 11], problems[i].problem_id)
             for i in range(len(problems))]

    cache = json.load(open(CACHE))
    unsolved_ids = set(cache["summary"]["unsolved_ids"])
    print(f"  slice cid={slice_cid[:16]}…  unsolved={len(unsolved_ids)}/30 "
          f"(the sharp precursor slice)")

    # teacher corpus (EXPOSED-side, problem-disjoint) + family motifs
    target_shorts = [p.short_name for p in problems]
    corpus = S.load_exposed_teacher_corpus_v1(target_short_names=target_shorts)
    motifs = S.derive_family_motifs_v1(corpus)
    print(f"  teacher corpus: {motifs.n_solutions} exposed accepted .py / "
          f"{motifs.n_problems} problems (disjoint); corpus_cid={motifs.corpus_cid[:16]}…")
    print(f"  family idioms: {motifs.idiom_freq}")

    targets = [(p, pool) for p, pool in zip(problems, pools)
               if p.problem_id in unsolved_ids]
    if args.limit:
        targets = targets[:args.limit]

    t0 = _dt.datetime.now(_dt.timezone.utc)
    results = []
    for i, (p, pool) in enumerate(targets):
        guard = S.SynthesisLeakageGuardV1(p)  # target accepted solution NEVER loaded
        r = S.synthesize_and_measure_problem_v1(
            p, pool, motifs, guard, was_unsolved=True, B_syn=args.b_syn,
            timeout_s=args.timeout_s, consensus_max_cases=args.consensus_max_cases)
        results.append(r)
        flag = ("  *** BLIND NEW PASS ***" if r.blind_new_pass else
                ("  (oracle-only new pass)" if r.oracle_new_pass else ""))
        print(f"   [{i+1:2d}/{len(targets)}] {p.short_name:28s} "
              f"cands={r.n_synth_candidates:3d} oracle={int(r.oracle_program_pass)} "
              f"blind={int(r.blind_program_pass)} cons={int(r.consensus_pass)}"
              f"({r.consensus_variant or '-'}) leak_clean={int(r.leakage_clean)}{flag}",
              flush=True)
    wall = (_dt.datetime.now(_dt.timezone.utc) - t0).total_seconds()

    gate = S.apply_synthesis_earn_gate_v1(results)
    verdict = {
        "schema": "coordpy.w126_synthesis_precursor.v1", "lane": "alpha_beta_resistant_first",
        "verified_on": "2026-05-31", "field": "W120 resistant official-ICPC 30-slice",
        "slice_cid": slice_cid, "n_unsolved_probed": len(targets),
        "nim_spend": 0, "wall_s": round(wall, 1), "b_syn": args.b_syn,
        "teacher_corpus": {"n_solutions": motifs.n_solutions,
                           "n_problems": motifs.n_problems,
                           "corpus_cid": motifs.corpus_cid, "idiom_freq": motifs.idiom_freq},
        "per_problem": [r.to_dict() for r in results],
        "earn_gate": gate.to_dict(),
        "null_band_pp": 3.34, "retirement_bar_pp": 5.00,
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "synthesis_precursor_verdict.json")
    with open(out, "w") as f:
        json.dump(verdict, f, indent=2, default=str)

    print()
    print(f"  ORACLE new-solved={gate.oracle_new_solved}  BLIND new-solved="
          f"{gate.blind_new_solved}  families={list(gate.distinct_families)}")
    print(f"  leakage_all_clean={gate.leakage_all_clean}  P1(>=2 new)={gate.p1_two_distinct_new}"
          f"  P2(>=2 families)={gate.p2_two_distinct_families}")
    print(f"  EARN GATE: {gate.verdict_label} (earned={gate.earned})")
    print(f"    {gate.rationale}")
    print(f"  wrote {out}  (wall {wall:.0f}s, $0 NIM)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
