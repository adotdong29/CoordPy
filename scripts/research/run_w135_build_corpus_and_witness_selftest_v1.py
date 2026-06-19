#!/usr/bin/env python3
"""W135-alpha — build the NON-COMPLEXITY corpus + run the structure-witness self-tests ($0 NIM).

Produces (results/w135/corpus/):
  * corpus_build_v1.json                 — the corpus manifest + LOCKED CIDs + disjointness
  * separation_characterization_v1.json  — the structure witness vs the EW1 counterexample on the
                                           naive(wrong-algo) / ref(correct) pair of every admitted
                                           train problem (the Lane-alpha faithfulness gate)
  * selftest_v1.json                     — the five Lane-alpha quality gates + regression fixtures

No model inference; the only code execution is the (audited) answer-key + candidate subprocess.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import sys  # noqa: E402
sys.path.insert(0, str(ROOT))

from coordpy.noncomplexity_structure_corpus_v1 import (  # noqa: E402
    build_noncomplexity_corpus_v1, noncomplexity_slate_v1, select_dev_bench_slice_v1,
    build_noncomplexity_split_v1, save_corpus_v1, TRAIN_SEEDS,
)
from coordpy.resistant_by_construction_battlefield_v1 import (  # noqa: E402
    MODE_COMPLEXITY_BLIND, mint_battlefield_v1,
)
from coordpy.resistant_by_construction_slate_v1 import RBC_SLATE_V1  # noqa: E402
from coordpy.exact_oracle_witness_v1 import (  # noqa: E402
    build_witness_probe_set_v1, find_counterexample_witness_v1, witness_is_genuinely_new_v1,
)
from coordpy.solution_structure_witness_v1 import (  # noqa: E402
    build_structure_witness_v1, structure_witness_is_genuinely_new_v1,
)

OUT = ROOT / "results" / "w135" / "corpus"
WITNESS_SEED = 999_135
WITNESS_TIMEOUT_S = 2.0
ORACLE_TIMEOUT_S = 4.0


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=float, default=8.0, help="mint gate timeout")
    ap.add_argument("--lock-cid", default="", help="if set, assert corpus_cid matches before caching")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    slate = noncomplexity_slate_v1()
    print(f"[1] minting the non-complexity corpus ({len(slate)} templates × 20 seed-disjoint mints; "
          f"timeout={args.timeout}s) ...")
    corpus = build_noncomplexity_corpus_v1(timeout_s=float(args.timeout))
    cid = corpus.corpus_cid()
    if args.lock_cid and cid != args.lock_cid:
        print(f"    !! corpus_cid {cid} != locked {args.lock_cid}; NOT writing cache")
        raise SystemExit(2)
    cd = corpus.to_dict()
    (OUT / "corpus_build_v1.json").write_text(json.dumps(cd, indent=2, default=str))
    save_corpus_v1(corpus, OUT / "corpus_cache.pkl")
    print(f"    templates={len(slate)}  admitted train/dev/eval/frontier="
          f"{corpus.train.n_admitted}/{corpus.dev.n_admitted}/{corpus.eval.n_admitted}/"
          f"{corpus.frontier.n_admitted}  meets_floors={corpus.meets_floors}")
    print(f"    train modes={corpus.train.mode_histogram}")
    print(f"    corpus_cid={cid[:16]}  eval_split_cid={corpus.eval.split_cid[:16]}  "
          f"frontier_slice_cid={corpus.frontier_slice_cid[:16]}")
    print(f"    disjointness held_out_integrity={corpus.disjointness['held_out_integrity']}")
    dev_slice = select_dev_bench_slice_v1(corpus.dev, per_family=1)
    print(f"    dev_bench_slice(per_family=1)={len(dev_slice)} ; frontier_slice={len(corpus.frontier_slice_ids)}")

    tmpl_by_id = corpus.template_by_problem_id()

    # ---- separation characterization: structure witness vs EW1 counterexample on naive/ref ----
    print("[2] naive/ref separation characterization on the TRAIN split ...")
    sep_rows, struct_sep_ok, ew1_sep_ok, struct_new, ew1_new, agree = [], 0, 0, 0, 0, 0
    for p in corpus.train.problems:
        t = tmpl_by_id[p.problem_id]
        probe = build_witness_probe_set_v1(t, p, witness_seed=WITNESS_SEED, timeout_s=WITNESS_TIMEOUT_S)
        sw_naive = build_structure_witness_v1(t.naive_source, p, probe, t,
                                              timeout_s=WITNESS_TIMEOUT_S, oracle_timeout_s=ORACLE_TIMEOUT_S)
        sw_ref = build_structure_witness_v1(t.ref_source, p, probe, t,
                                            timeout_s=WITNESS_TIMEOUT_S, oracle_timeout_s=ORACLE_TIMEOUT_S)
        # EW1 baseline (the W133 flat counterexample) on the same naive/ref
        ew1_naive = find_counterexample_witness_v1(t.naive_source, p, probe, t, timeout_s=WITNESS_TIMEOUT_S)
        ew1_ref = find_counterexample_witness_v1(t.ref_source, p, probe, t, timeout_s=WITNESS_TIMEOUT_S)
        s_ok = bool(sw_naive.found() and not sw_ref.found())
        e_ok = bool(ew1_naive.found() and not ew1_ref.found())
        gn = structure_witness_is_genuinely_new_v1(sw_naive, p)
        en = witness_is_genuinely_new_v1(ew1_naive, p)
        struct_sep_ok += int(s_ok); ew1_sep_ok += int(e_ok); agree += int(s_ok == e_ok)
        struct_new += int(gn["genuinely_new"]); ew1_new += int(en["genuinely_new"])
        sep_rows.append({
            "problem_id": p.problem_id, "family": p.family, "mode": p.mode,
            "struct_naive_found": sw_naive.found(), "struct_ref_silent": not sw_ref.found(),
            "struct_separation_ok": s_ok, "struct_genuinely_new": gn["genuinely_new"],
            "struct_kind": sw_naive.kind, "n_ladder_rungs": gn["n_ladder_rungs"],
            "has_attribution_contrast": gn["has_attribution_contrast"],
            "struct_leakage_clean": sw_naive.leakage_clean,
            "ew1_naive_found": ew1_naive.found(), "ew1_separation_ok": e_ok,
            "ew1_genuinely_new": en["genuinely_new"]})
    n = len(corpus.train.problems)
    all_struct_leak_clean = all(r["struct_leakage_clean"] for r in sep_rows)
    sep = {"n_train_problems": n,
           "struct_separation_pass": struct_sep_ok, "struct_separation_rate": struct_sep_ok / max(1, n),
           "ew1_separation_pass": ew1_sep_ok, "ew1_separation_rate": ew1_sep_ok / max(1, n),
           "struct_vs_ew1_separation_agreement": agree / max(1, n),
           "struct_genuinely_new_pass": struct_new, "struct_genuinely_new_rate": struct_new / max(1, n),
           "ew1_genuinely_new_pass": ew1_new, "ew1_genuinely_new_rate": ew1_new / max(1, n),
           "all_struct_leakage_clean": all_struct_leak_clean,
           "by_mode": sorted({r["mode"] for r in sep_rows}),
           "by_family": sorted({r["family"] for r in sep_rows}), "rows": sep_rows}
    (OUT / "separation_characterization_v1.json").write_text(json.dumps(sep, indent=2, default=str))
    print(f"    struct separation {struct_sep_ok}/{n} ; EW1 separation {ew1_sep_ok}/{n} ; "
          f"agreement {agree}/{n}")
    print(f"    struct genuinely-new {struct_new}/{n} ; EW1 genuinely-new {ew1_new}/{n} ; "
          f"all_struct_leak_clean={all_struct_leak_clean}")

    # ---- five Lane-alpha quality gates + regression fixtures ----
    print("[3] self-test gates + regression fixtures ...")
    sample = corpus.train.problems[0]
    st = tmpl_by_id[sample.problem_id]
    sprobe = build_witness_probe_set_v1(st, sample, witness_seed=WITNESS_SEED, timeout_s=WITNESS_TIMEOUT_S)
    # 1. witness reproducibility (block bytes + found verdict)
    w1 = build_structure_witness_v1(st.naive_source, sample, sprobe, st, timeout_s=WITNESS_TIMEOUT_S)
    w2 = build_structure_witness_v1(st.naive_source, sample, sprobe, st, timeout_s=WITNESS_TIMEOUT_S)
    g1 = bool(w1.found() == w2.found() and w1.cid() == w2.cid()
              and w1.to_prompt_block("S4") == w2.to_prompt_block("S4"))
    # 2. deterministic shrink/ladder (same counterexample input + same ladder summaries)
    g2 = bool(w1.counterexample.probe_input == w2.counterexample.probe_input
              and [r.summary for r in w1.ladder] == [r.summary for r in w2.ladder]
              and [r.optimal_value for r in w1.ladder] == [r.optimal_value for r in w2.ladder])
    # 3. naive/ref separation on every admitted train problem
    g3 = bool(struct_sep_ok == n)
    # 4. genuinely-new vs EW1 (strictly MORE than a bare counterexample = a >=2-rung sub-value
    #    ladder, or a canonical-greedy datapoint distinct from observed) on a strong majority of
    #    train problems, always leakage-clean.  (struct_new and ew1_new measure DIFFERENT axes:
    #    struct_new = more-than-EW1; ew1_new = EW1 more-than-the-blind-B0-bit — both recorded.)
    g4 = bool(struct_new >= int(0.6 * n) and all_struct_leak_clean)
    # 5. deterministic split regeneration (re-mint train seeds -> same split_cid)
    re_train = build_noncomplexity_split_v1(noncomplexity_slate_v1(), split="train", seeds=TRAIN_SEEDS,
                                            minted_date=corpus.minted_date)
    g5 = bool(re_train.split_cid == corpus.train.split_cid)

    # regression fixtures: the structure witness fires GENUINELY-NEW (a >=2-rung optimal-substructure
    # ladder) on the W133 zero-gain WA/SE families WHERE the OBVIOUS sub-structure is extractable
    # (the demonstration that it carries strictly more than the flat counterexample channel; families
    # whose obvious parameterisation is thin — knapsack/LCS/grid/subset — fall back to the
    # counterexample+greedy-contrast and are reported in the 54/80 genuinely-new rate, not here).
    fixt = {}
    for fam in ["wa_max_nonadjacent_sum", "wa_min_coins", "se_count_stair_climbings",
                "se_count_bsts_catalan"]:
        pid = f"rbc_{fam}__s{TRAIN_SEEDS[0]}"
        p = next((q for q in corpus.train.problems if q.problem_id == pid), None)
        if p is None:
            fixt[fam] = {"present": False}
            continue
        t = tmpl_by_id[pid]
        probe = build_witness_probe_set_v1(t, p, witness_seed=WITNESS_SEED, timeout_s=WITNESS_TIMEOUT_S)
        w = build_structure_witness_v1(t.naive_source, p, probe, t, timeout_s=WITNESS_TIMEOUT_S,
                                       oracle_timeout_s=ORACLE_TIMEOUT_S)
        gn = structure_witness_is_genuinely_new_v1(w, p)
        fixt[fam] = {"present": True, "fires": w.found(), "genuinely_new": gn["genuinely_new"],
                     "kind": w.kind, "n_ladder_rungs": gn["n_ladder_rungs"],
                     "leakage_clean": w.leakage_clean}
    fixtures_fire = all(v.get("genuinely_new") for v in fixt.values() if v.get("present"))

    # negative control: structure witness must be SILENT (NONE) on a COMPLEXITY_BLIND naive
    cb = next(t for t in RBC_SLATE_V1 if t.mode == MODE_COMPLEXITY_BLIND)
    cbf = mint_battlefield_v1([cb], global_seed=TRAIN_SEEDS[0], minted_date=corpus.minted_date, timeout_s=8.0)
    neg_cb_silent = True
    if cbf.problems:
        cp = cbf.problems[0]
        cprobe = build_witness_probe_set_v1(cb, cp, witness_seed=WITNESS_SEED, timeout_s=WITNESS_TIMEOUT_S)
        cw = build_structure_witness_v1(cb.naive_source, cp, cprobe, cb, timeout_s=WITNESS_TIMEOUT_S,
                                        oracle_timeout_s=ORACLE_TIMEOUT_S)
        neg_cb_silent = bool(not cw.found())
    # positive control: structure witness must be NONE on the correct ref_source of a train problem
    pos_ref_silent = bool(not build_structure_witness_v1(st.ref_source, sample, sprobe, st,
                                                         timeout_s=WITNESS_TIMEOUT_S).found())

    selftest = {
        "g1_witness_reproducible": g1, "g2_shrink_ladder_deterministic": g2,
        "g3_naive_ref_separation_all": g3, "g4_genuinely_new_vs_ew1": g4,
        "g5_split_regen_deterministic": g5,
        "struct_genuinely_new_pass": struct_new, "ew1_genuinely_new_pass": ew1_new,
        "regression_fixtures": fixt, "regression_fixtures_all_genuinely_new": fixtures_fire,
        "negative_control_complexity_silent": neg_cb_silent,
        "positive_control_ref_silent": pos_ref_silent,
        "all_pass": bool(g1 and g2 and g3 and g4 and g5 and fixtures_fire and neg_cb_silent
                         and pos_ref_silent)}
    (OUT / "selftest_v1.json").write_text(json.dumps(selftest, indent=2, default=str))
    wall = time.time() - t0
    print(f"    g1_repro={g1} g2_shrink={g2} g3_sep={g3} g4_gnew={g4} g5_regen={g5}")
    print(f"    regression_fixtures_all_gnew={fixtures_fire} neg_control_complexity_silent={neg_cb_silent} "
          f"pos_control_ref_silent={pos_ref_silent}")
    print(f"    ALL_PASS={selftest['all_pass']}  (wall {wall:.0f}s)")
    print(f"\nLOCKED CIDs (record in RUNBOOK_W135.md §3 before any eval/frontier spend):")
    print(f"  corpus_cid           = {corpus.corpus_cid()}")
    print(f"  eval_split_cid       = {corpus.eval.split_cid}")
    print(f"  frontier_slice_cid   = {corpus.frontier_slice_cid}")
    return 0 if (corpus.meets_floors and selftest["all_pass"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
