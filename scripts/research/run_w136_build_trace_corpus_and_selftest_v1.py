#!/usr/bin/env python3
"""W136-alpha — build the algorithm-state-trace corpus + run the $0 self-tests / regression fixtures /
generator-failure atlas (all NIM-free).

Lane-α (RUNBOOK_W136 §3/§8): mint the fresh-seed W136 corpus (16 wa_*/se_* templates × 136_0xx seeds),
LOCK the corpus / eval-split / frontier-slice CIDs, and prove the trace instrument is REAL + leakage-clean
before any NIM:

  1. trace reproducibility           — same (code, problem, seed) ⇒ byte-identical capsule + verdict
  2. deterministic typed sub-instances — same input ⇒ same ladder (bytes)
  3. naive/ref separation            — trace fires genuinely-new on every admitted train naive_source,
                                        is NONE on every ref_source (faithfulness); records the
                                        trace-admissibility + genuinely-new-vs-S4 rates
  4. genuinely-new-vs-S4             — the trace carries the dual trajectory + transition S4's flat
                                        optimal-only ladder lacks
  5. deterministic split regeneration — re-mint ⇒ same corpus_cid

Regression fixtures: the 3 W135 capability-bound traps (trace must fire genuinely-new where S4 was flat);
the W134 COMPLEXITY negative control (trace NONE on a value-correct-but-slow naive ⇒ no counterexample);
the positive control (ref_source ⇒ NONE).  Generator-failure atlas: characterize the W135 dev traps.

    python scripts/run_w136_build_trace_corpus_and_selftest_v1.py     # 0 NIM
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from coordpy.algorithm_state_trace_corpus_v1 import (  # noqa: E402
    MIN_FRONTIER, MIN_PER_SPLIT, MINTED_DATE, build_algorithm_state_corpus_v1,
    select_dev_bench_slice_v1, select_frontier_slice_v1,
)
from coordpy.noncomplexity_structure_corpus_v1 import save_corpus_v1  # noqa: E402
from coordpy.algorithm_state_trace_v1 import (  # noqa: E402
    TRACE_NONE, build_algorithm_state_trace_v1, trace_is_genuinely_new_vs_structure_v1,
)
from coordpy.solution_structure_witness_v1 import (  # noqa: E402
    build_structure_witness_v1, structure_witness_is_genuinely_new_v1,
)
from coordpy.exact_oracle_witness_v1 import build_witness_probe_set_v1  # noqa: E402
from coordpy.resistant_by_construction_battlefield_v1 import (  # noqa: E402
    MODE_COMPLEXITY_BLIND, mint_problem_v1,
)
from coordpy.resistant_by_construction_slate_v1 import RBC_SLATE_V1  # noqa: E402
from coordpy.noncomplexity_structure_corpus_v1 import load_corpus_v1  # noqa: E402

WITNESS_SEED = 999_136
OUT = ROOT / "results" / "w136" / "corpus"
W135_CACHE = ROOT / "results" / "w135" / "corpus" / "corpus_cache.pkl"


def _trace_on(code, problem, template):
    probe = build_witness_probe_set_v1(template, problem, witness_seed=WITNESS_SEED, timeout_s=2.0)
    return build_algorithm_state_trace_v1(code, problem, probe, template, timeout_s=2.0,
                                          oracle_timeout_s=4.0)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print("[W136-alpha] building the fresh-seed algorithm-state-trace corpus ($0 NIM) ...")
    corpus = build_algorithm_state_corpus_v1()
    save_corpus_v1(corpus, OUT / "corpus_cache.pkl")
    dev_bench = select_dev_bench_slice_v1(corpus.dev, per_family=1)
    fslice = select_frontier_slice_v1(corpus.frontier, n=MIN_FRONTIER)
    tmpl_by_id = corpus.template_by_problem_id()

    admitted = {"train": corpus.train.n_admitted, "dev": corpus.dev.n_admitted,
                "eval": corpus.eval.n_admitted, "frontier": corpus.frontier.n_admitted}
    meets = corpus.meets_floors
    print(f"  corpus_cid          = {corpus.corpus_cid()}")
    print(f"  eval_split_cid      = {corpus.eval.split_cid}")
    print(f"  frontier_slice_cid  = {corpus.frontier_slice_cid}")
    print(f"  admitted train/dev/eval/frontier = {admitted}  (floors {MIN_PER_SPLIT}/"
          f"{MIN_PER_SPLIT}/{MIN_PER_SPLIT}/{MIN_FRONTIER}; meets={meets})")
    print(f"  held_out_integrity  = {corpus.disjointness.get('held_out_integrity')}")

    # ---------- determinism: re-mint ⇒ same corpus_cid ----------
    corpus2 = build_algorithm_state_corpus_v1()
    det_corpus = bool(corpus2.corpus_cid() == corpus.corpus_cid())

    # ---------- naive/ref separation + genuinely-new-vs-S4 (faithfulness; on admitted TRAIN) ----------
    train = corpus.train.problems
    n_train = len(train)
    naive_fires_new = 0          # trace genuinely-new on naive_source
    ref_silent = 0               # trace NONE on ref_source
    trace_admissible = 0         # trace fires (found) on naive_source
    new_vs_s4 = 0                # trace genuinely-new where S4 ladder was NOT genuinely-new
    per_family_admit: dict = {}
    for p in train:
        t = tmpl_by_id[p.problem_id]
        tr_n = _trace_on(t.naive_source, p, t)
        gn = trace_is_genuinely_new_vs_structure_v1(tr_n, p)
        sw = build_structure_witness_v1(
            t.naive_source, p, build_witness_probe_set_v1(t, p, witness_seed=WITNESS_SEED, timeout_s=2.0), t)
        gs = structure_witness_is_genuinely_new_v1(sw, p)
        tr_r = _trace_on(t.ref_source, p, t)
        if tr_n.found():
            trace_admissible += 1
            per_family_admit[p.family] = per_family_admit.get(p.family, 0) + 1
        if gn["genuinely_new"]:
            naive_fires_new += 1
        if tr_r.kind == TRACE_NONE:
            ref_silent += 1
        if gn["genuinely_new"] and not gs["genuinely_new"]:
            new_vs_s4 += 1

    # ---------- reproducibility + deterministic sub-instances ----------
    p0 = train[0]
    t0t = tmpl_by_id[p0.problem_id]
    a = _trace_on(t0t.naive_source, p0, t0t)
    b = _trace_on(t0t.naive_source, p0, t0t)
    repro = bool(a.to_capsule_block("T1") == b.to_capsule_block("T1") and a.cid() == b.cid())

    # ---------- regression fixtures ----------
    # (a) the 3 W135 capability-bound traps: trace fires genuinely-new where S4 was flat
    w135 = load_corpus_v1(W135_CACHE)
    traps = ["rbc_wa_knapsack_01", "rbc_wa_weighted_interval_scheduling", "rbc_se_lattice_paths_blocked"]
    trap_report = {}
    if w135 is not None:
        w135_tmpl = w135.template_by_problem_id()
        w135_by_id = {pp.problem_id: pp for pp in select_dev_bench_slice_v1(w135.dev, per_family=1)}
        for base in traps:
            tid = next((q for q in w135_by_id if q.startswith(base + "__")), None)
            if tid is None:
                continue
            pp, tt = w135_by_id[tid], w135_tmpl[tid]
            tr = _trace_on(tt.naive_source, pp, tt)
            gn = trace_is_genuinely_new_vs_structure_v1(tr, pp)
            sw = build_structure_witness_v1(
                tt.naive_source, pp, build_witness_probe_set_v1(tt, pp, witness_seed=WITNESS_SEED, timeout_s=2.0), tt)
            gs = structure_witness_is_genuinely_new_v1(sw, pp)
            trap_report[base] = {"kind": tr.kind, "at_family": tr.at_family, "n_rows": len(tr.rows),
                                 "first_divergence_idx": tr.first_divergence_idx,
                                 "optimal_trajectory": [r.optimal_value for r in tr.rows],
                                 "naive_trajectory": [r.naive_value for r in tr.rows],
                                 "trace_genuinely_new": gn["genuinely_new"],
                                 "s4_ladder_genuinely_new": gs["genuinely_new"], "leakage_clean": tr.leakage_clean}
    traps_fire = all(v["trace_genuinely_new"] for v in trap_report.values()) if trap_report else False

    # (b) negative control: a COMPLEXITY_BLIND naive is value-correct ⇒ NO counterexample ⇒ trace NONE
    cb_tmpl = next((t for t in RBC_SLATE_V1 if t.mode == MODE_COMPLEXITY_BLIND), None)
    neg_silent = None
    if cb_tmpl is not None:
        cbp = mint_problem_v1(cb_tmpl, global_seed=136_999, timeout_s=8.0)
        tr_cb = _trace_on(cb_tmpl.naive_source, cbp, cb_tmpl)
        neg_silent = bool(tr_cb.kind == TRACE_NONE)

    # (c) positive control: ref_source ⇒ NONE (already counted in ref_silent; spot-check)
    pos_silent = bool(_trace_on(t0t.ref_source, p0, t0t).kind == TRACE_NONE)

    # ---------- generator-failure atlas on the W135 dev 16 (characterize the field) ----------
    atlas = []
    if w135 is not None:
        for pp in select_dev_bench_slice_v1(w135.dev, per_family=1):
            tt = w135.template_by_problem_id()[pp.problem_id]
            tr = _trace_on(tt.naive_source, pp, tt)
            gn = trace_is_genuinely_new_vs_structure_v1(tr, pp)
            atlas.append({"problem_id": pp.problem_id, "mode": pp.mode, "family": pp.family,
                          "trace_kind": tr.kind, "n_rows": len(tr.rows),
                          "first_divergence_idx": tr.first_divergence_idx,
                          "trace_genuinely_new": gn["genuinely_new"]})

    selftests = {
        "trace_reproducibility": repro,
        "deterministic_split_regeneration": det_corpus,
        "naive_ref_separation": {"n_train": n_train, "naive_fires_genuinely_new": naive_fires_new,
                                 "ref_silent_none": ref_silent, "trace_admissible": trace_admissible,
                                 "trace_admissible_by_family": dict(sorted(per_family_admit.items())),
                                 "all_naive_fire": bool(naive_fires_new == n_train),
                                 "all_ref_silent": bool(ref_silent == n_train)},
        "genuinely_new_vs_s4_count": new_vs_s4,
        "regression_traps": {"report": trap_report, "all_traps_fire_genuinely_new": traps_fire},
        "negative_control_complexity_silent": neg_silent,
        "positive_control_ref_silent": pos_silent,
    }
    all_pass = bool(repro and det_corpus and meets
                    and selftests["naive_ref_separation"]["all_naive_fire"]
                    and selftests["naive_ref_separation"]["all_ref_silent"]
                    and traps_fire and neg_silent and pos_silent)

    payload = {
        "schema": "coordpy.w136_trace_corpus_selftest_v1", "minted_date": MINTED_DATE,
        "corpus_cid": corpus.corpus_cid(), "eval_split_cid": corpus.eval.split_cid,
        "frontier_slice_cid": corpus.frontier_slice_cid, "admitted": admitted, "meets_floors": meets,
        "held_out_integrity": corpus.disjointness.get("held_out_integrity"),
        "dev_bench_ids": [p.problem_id for p in dev_bench],
        "frontier_slice_ids": [p.problem_id for p in fslice],
        "selftests": selftests, "all_selftests_pass": all_pass,
        "generator_failure_atlas": atlas, "wall_s": round(time.time() - t0, 2),
    }
    (OUT / "corpus_build_selftest_v1.json").write_text(json.dumps(payload, indent=2, default=str))

    print(f"\n  self-tests: repro={repro} det_corpus={det_corpus} "
          f"naive/ref sep={naive_fires_new}/{n_train} fire, {ref_silent}/{n_train} ref-silent "
          f"(trace-admissible {trace_admissible}/{n_train}; new-vs-S4 {new_vs_s4}/{n_train})")
    print(f"  regression: traps_fire={traps_fire}  neg_control_silent={neg_silent}  "
          f"pos_control_silent={pos_silent}")
    for base, v in trap_report.items():
        print(f"    {base}: kind={v['kind']} rows={v['n_rows']} opt={v['optimal_trajectory']} "
              f"you={v['naive_trajectory']} new={v['trace_genuinely_new']} (S4_new={v['s4_ladder_genuinely_new']})")
    print(f"  ALL SELF-TESTS PASS = {all_pass}")
    print(f"  out: {OUT / 'corpus_build_selftest_v1.json'}  (wall {time.time()-t0:.1f}s)")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
