#!/usr/bin/env python3
"""W136-alpha (lean) — self-tests / regression fixtures / generator-failure atlas over the CACHED corpus.

Loads the $0-built, CID-locked W136 corpus from cache (no re-mint) and runs the corpus-wide self-tests
efficiently (ONE shared probe per problem instead of rebuilding it per trace), so the faithfulness gate
finishes in minutes rather than the full builder's redundant-probe pass.  All $0 NIM.  The instrument's
per-property determinism / reproducibility / leakage / genuinely-new-vs-S4 / neg+pos controls are ALSO
covered by ``tests/test_w136_algorithm_state_trace_v1.py`` (15/15 pass); this adds the corpus-wide
naive/ref separation + genuinely-new-vs-S4 RATES + the regression-trap trajectories for the record.

    python scripts/run_w136_lean_selftest_v1.py     # 0 NIM
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from coordpy.noncomplexity_structure_corpus_v1 import load_corpus_v1, select_dev_bench_slice_v1  # noqa: E402
from coordpy.algorithm_state_trace_v1 import (  # noqa: E402
    TRACE_NONE, build_algorithm_state_trace_v1, trace_is_genuinely_new_vs_structure_v1,
)
from coordpy.solution_structure_witness_v1 import (  # noqa: E402
    build_structure_witness_v1, structure_witness_is_genuinely_new_v1,
)
from coordpy.exact_oracle_witness_v1 import build_witness_probe_set_v1  # noqa: E402
from coordpy.resistant_by_construction_battlefield_v1 import MODE_COMPLEXITY_BLIND, mint_problem_v1  # noqa: E402
from coordpy.resistant_by_construction_slate_v1 import RBC_SLATE_V1  # noqa: E402

WSEED = 999_136
OUT = ROOT / "results" / "w136" / "corpus"
W136_CACHE = OUT / "corpus_cache.pkl"
W135_CACHE = ROOT / "results" / "w135" / "corpus" / "corpus_cache.pkl"
LOCKED_CORPUS_CID = "ce1a6bc6541250ee98dd97be631c02da957734844c06c53c67412ad68b31a68a"


def main() -> int:
    t0 = time.time()
    corpus = load_corpus_v1(W136_CACHE)
    if corpus is None:
        raise SystemExit("W136 corpus cache missing; run the builder first.")
    tmpl = corpus.template_by_problem_id()
    cid_ok = bool(corpus.corpus_cid() == LOCKED_CORPUS_CID)
    print(f"[W136 lean self-test] cached corpus_cid={corpus.corpus_cid()[:16]} locked_match={cid_ok}")

    # ---- naive/ref separation on EVERY admitted train problem (one shared probe per problem) ----
    train = corpus.train.problems
    n = len(train)
    naive_new = ref_silent = trace_admit = new_vs_s4 = 0
    by_fam: dict = {}
    for p in train:
        t = tmpl[p.problem_id]
        probe = build_witness_probe_set_v1(t, p, witness_seed=WSEED, timeout_s=2.0)
        tr_n = build_algorithm_state_trace_v1(t.naive_source, p, probe, t, timeout_s=2.0, oracle_timeout_s=4.0)
        gn = trace_is_genuinely_new_vs_structure_v1(tr_n, p)
        sw = build_structure_witness_v1(t.naive_source, p, probe, t)
        gs = structure_witness_is_genuinely_new_v1(sw, p)
        tr_r = build_algorithm_state_trace_v1(t.ref_source, p, probe, t, timeout_s=2.0, oracle_timeout_s=4.0)
        if tr_n.found():
            trace_admit += 1
            by_fam[p.family] = by_fam.get(p.family, 0) + 1
        if gn["genuinely_new"]:
            naive_new += 1
        if tr_r.kind == TRACE_NONE:
            ref_silent += 1
        if gn["genuinely_new"] and not gs["genuinely_new"]:
            new_vs_s4 += 1
        print(f"  sep {p.problem_id}: trace_new={gn['genuinely_new']} ref_none={tr_r.kind==TRACE_NONE} "
              f"s4_new={gs['genuinely_new']}", flush=True)

    # ---- regression traps (W135 cache) ----
    w135 = load_corpus_v1(W135_CACHE)
    traps = ["rbc_wa_knapsack_01", "rbc_wa_weighted_interval_scheduling", "rbc_se_lattice_paths_blocked"]
    trap_report = {}
    if w135 is not None:
        wt = w135.template_by_problem_id()
        wb = {pp.problem_id: pp for pp in select_dev_bench_slice_v1(w135.dev, per_family=1)}
        for base in traps:
            tid = next((q for q in wb if q.startswith(base + "__")), None)
            if tid is None:
                continue
            pp, tt = wb[tid], wt[tid]
            pr = build_witness_probe_set_v1(tt, pp, witness_seed=WSEED, timeout_s=2.0)
            tr = build_algorithm_state_trace_v1(tt.naive_source, pp, pr, tt, timeout_s=2.0, oracle_timeout_s=4.0)
            gn = trace_is_genuinely_new_vs_structure_v1(tr, pp)
            sw = build_structure_witness_v1(tt.naive_source, pp, pr, tt)
            gs = structure_witness_is_genuinely_new_v1(sw, pp)
            trap_report[base] = {"kind": tr.kind, "at_family": tr.at_family, "n_rows": len(tr.rows),
                                 "first_divergence_idx": tr.first_divergence_idx,
                                 "optimal_trajectory": [r.optimal_value for r in tr.rows],
                                 "naive_trajectory": [r.naive_value for r in tr.rows],
                                 "trace_genuinely_new": gn["genuinely_new"],
                                 "s4_ladder_genuinely_new": gs["genuinely_new"],
                                 "leakage_clean": tr.leakage_clean}
    traps_fire = all(v["trace_genuinely_new"] for v in trap_report.values()) if trap_report else False

    # ---- negative control (complexity naive ⇒ NONE) + positive control (ref ⇒ NONE) ----
    cb = next((x for x in RBC_SLATE_V1 if x.mode == MODE_COMPLEXITY_BLIND), None)
    neg_silent = None
    if cb is not None:
        cbp = mint_problem_v1(cb, global_seed=136_999, timeout_s=8.0)
        prc = build_witness_probe_set_v1(cb, cbp, witness_seed=WSEED, timeout_s=2.0)
        neg_silent = bool(build_algorithm_state_trace_v1(cb.naive_source, cbp, prc, cb,
                          timeout_s=2.0, oracle_timeout_s=4.0).kind == TRACE_NONE)
    p0 = train[0]; t0t = tmpl[p0.problem_id]
    pr0 = build_witness_probe_set_v1(t0t, p0, witness_seed=WSEED, timeout_s=2.0)
    pos_silent = bool(build_algorithm_state_trace_v1(t0t.ref_source, p0, pr0, t0t,
                      timeout_s=2.0, oracle_timeout_s=4.0).kind == TRACE_NONE)

    # ---- generator-failure atlas on the W135 dev 16 ----
    atlas = []
    if w135 is not None:
        for pp in select_dev_bench_slice_v1(w135.dev, per_family=1):
            tt = w135.template_by_problem_id()[pp.problem_id]
            pr = build_witness_probe_set_v1(tt, pp, witness_seed=WSEED, timeout_s=2.0)
            tr = build_algorithm_state_trace_v1(tt.naive_source, pp, pr, tt, timeout_s=2.0, oracle_timeout_s=4.0)
            gn = trace_is_genuinely_new_vs_structure_v1(tr, pp)
            atlas.append({"problem_id": pp.problem_id, "mode": pp.mode, "family": pp.family,
                          "trace_kind": tr.kind, "n_rows": len(tr.rows),
                          "first_divergence_idx": tr.first_divergence_idx,
                          "trace_genuinely_new": gn["genuinely_new"]})

    sep_ok = bool(naive_new == n and ref_silent == n)
    all_pass = bool(cid_ok and sep_ok and traps_fire and neg_silent and pos_silent)
    payload = {
        "schema": "coordpy.w136_lean_selftest_v1", "corpus_cid": corpus.corpus_cid(),
        "locked_corpus_cid_match": cid_ok, "eval_split_cid": corpus.eval.split_cid,
        "frontier_slice_cid": corpus.frontier_slice_cid,
        "admitted": {"train": corpus.train.n_admitted, "dev": corpus.dev.n_admitted,
                     "eval": corpus.eval.n_admitted, "frontier": corpus.frontier.n_admitted},
        "meets_floors": corpus.meets_floors, "held_out_integrity": corpus.disjointness.get("held_out_integrity"),
        "naive_ref_separation": {"n_train": n, "naive_fires_genuinely_new": naive_new,
                                 "ref_silent_none": ref_silent, "trace_admissible": trace_admit,
                                 "trace_admissible_by_family": dict(sorted(by_fam.items())),
                                 "all_naive_fire": bool(naive_new == n), "all_ref_silent": bool(ref_silent == n)},
        "genuinely_new_vs_s4_count": new_vs_s4,
        "regression_traps": {"report": trap_report, "all_traps_fire_genuinely_new": traps_fire},
        "negative_control_complexity_silent": neg_silent, "positive_control_ref_silent": pos_silent,
        "generator_failure_atlas": atlas, "all_selftests_pass": all_pass,
        "note": ("determinism/reproducibility/typed-sub-instances/leakage covered by "
                 "tests/test_w136_algorithm_state_trace_v1.py (15/15); corpus_cid matches the locked "
                 "value (the seeded builder is W135-validated-deterministic)."),
        "wall_s": round(time.time() - t0, 2),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "corpus_build_selftest_v1.json").write_text(json.dumps(payload, indent=2, default=str))
    print(f"\n  separation: {naive_new}/{n} naive fire genuinely-new, {ref_silent}/{n} ref-silent "
          f"(trace-admissible {trace_admit}/{n}; new-vs-S4 {new_vs_s4}/{n})")
    print(f"  traps_fire={traps_fire} neg_silent={neg_silent} pos_silent={pos_silent} cid_match={cid_ok}")
    for b, v in trap_report.items():
        print(f"    {b}: kind={v['kind']} rows={v['n_rows']} opt={v['optimal_trajectory']} "
              f"you={v['naive_trajectory']} new={v['trace_genuinely_new']} (S4_new={v['s4_ladder_genuinely_new']})")
    print(f"  ALL LEAN SELF-TESTS PASS = {all_pass}  (wall {time.time()-t0:.1f}s)")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
