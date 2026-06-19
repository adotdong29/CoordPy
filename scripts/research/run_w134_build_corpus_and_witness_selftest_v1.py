#!/usr/bin/env python3
"""W134-alpha — build the complexity-only corpus + run the deployable-witness self-tests ($0 NIM).

Produces (results/w134/corpus/):
  * corpus_build_v1.json                 — the corpus manifest + LOCKED CIDs + disjointness
  * separation_characterization_v1.json  — the deployable witness vs the exact-oracle EW2 on the
                                           naive(O(N^2)) / ref(O(N log N)) pair of every admitted
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

from coordpy.complexity_only_corpus_v1 import (  # noqa: E402
    build_complexity_corpus_v1, complexity_slate_v1, select_dev_bench_slice_v1,
    select_frontier_slice_v1, build_complexity_split_v1, save_corpus_v1, TRAIN_SEEDS,
)
from coordpy.deployable_complexity_witness_v1 import (  # noqa: E402
    build_deployable_witness_v1, deployable_witness_is_genuinely_new_v1, build_ladder_v1,
)
from coordpy.exact_oracle_witness_v1 import (  # noqa: E402
    build_witness_probe_set_v1, find_complexity_witness_v1,
)

OUT = ROOT / "results" / "w134" / "corpus"
WITNESS_TIMEOUT_S = 2.0
# The corpus CID is timeout-invariant for these definitively-O(N^2) naives (they TLE at any
# reasonable gate timeout; the reference + brute finish fast at any timeout), so the canonical CID
# locked at the 8.0 build must be reproduced by a faster 3.0 build — asserted below.
LOCKED_CORPUS_CID = "191d995487d6cb09db6dba7683413661c69b1cefa82036a3fc339d5b0bb54a55"


def _spec_consistent(statement, samples) -> bool:
    lad = build_ladder_v1(statement, samples)
    if not lad.parseable:
        return False
    for sz, shps in lad.rungs:
        for _k, inp in shps:
            lines = inp.split("\n")
            if lines[0].split()[0] != str(sz) or len(lines[1].split()) != sz:
                return False
    return True


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=float, default=8.0,
                    help="mint gate timeout (CID-invariant; 3.0 builds faster, asserted equal)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"[1] minting the complexity-only corpus (20 seed-disjoint mints; timeout={args.timeout}s) ...")
    corpus = build_complexity_corpus_v1(timeout_s=float(args.timeout))
    cid = corpus.corpus_cid()
    if cid != LOCKED_CORPUS_CID:
        print(f"    !! corpus_cid {cid} != locked {LOCKED_CORPUS_CID} (timeout-variance?); NOT writing cache")
        raise SystemExit(2)
    print(f"    corpus_cid MATCHES locked {LOCKED_CORPUS_CID[:16]} (timeout-invariant confirmed)")
    cd = corpus.to_dict()
    (OUT / "corpus_build_v1.json").write_text(json.dumps(cd, indent=2, default=str))
    save_corpus_v1(corpus, OUT / "corpus_cache.pkl")   # deterministic cache so the bench never re-mints
    print(f"    templates={len(complexity_slate_v1())}  admitted="
          f"{cd['n_admitted']}  meets_floors={corpus.meets_floors}")
    print(f"    corpus_cid={corpus.corpus_cid()[:16]}  eval_split_cid={corpus.eval.split_cid[:16]}  "
          f"frontier_slice_cid={corpus.frontier_slice_cid[:16]}")
    print(f"    disjointness held_out_integrity={corpus.disjointness['held_out_integrity']}")
    dev_slice = select_dev_bench_slice_v1(corpus.dev, per_family=2)
    print(f"    dev_bench_slice(per_family=2)={len(dev_slice)} ; frontier_slice={len(corpus.frontier_slice_ids)}")

    tmpl_by_id = corpus.template_by_problem_id()

    # ---- separation characterization: deployable witness vs exact-oracle EW2 on naive/ref ----
    print("[2] naive/ref separation characterization on the TRAIN split ...")
    sep_rows = []
    dep_sep_ok = 0
    ew2_sep_ok = 0
    agree = 0
    for p in corpus.train.problems:
        t = tmpl_by_id[p.problem_id]
        w_naive = build_deployable_witness_v1(t.naive_source, statement=p.statement,
                                              samples=p.samples, timeout_s=WITNESS_TIMEOUT_S)
        w_ref = build_deployable_witness_v1(t.ref_source, statement=p.statement,
                                            samples=p.samples, timeout_s=WITNESS_TIMEOUT_S)
        dep_ok = bool(w_naive.found() and not w_ref.found())
        # exact-oracle EW2 (the C0 upper-bound witness) on the same naive/ref, for comparison
        probe = build_witness_probe_set_v1(t, p, witness_seed=999_134, timeout_s=WITNESS_TIMEOUT_S)
        e_naive = find_complexity_witness_v1(t.naive_source, p, probe, timeout_s=WITNESS_TIMEOUT_S)
        e_ref = find_complexity_witness_v1(t.ref_source, p, probe, timeout_s=WITNESS_TIMEOUT_S)
        ew2_ok = bool(e_naive.found() and not e_ref.found())
        dep_sep_ok += int(dep_ok)
        ew2_sep_ok += int(ew2_ok)
        agree += int(dep_ok == ew2_ok)
        gn = deployable_witness_is_genuinely_new_v1(w_naive)
        sep_rows.append({
            "problem_id": p.problem_id, "family": p.family,
            "deployable_naive_fired": w_naive.found(), "deployable_naive_reason": w_naive.reason,
            "deployable_naive_expo": w_naive.growth.fitted_exponent,
            "deployable_ref_fired": w_ref.found(), "deployable_ref_reason": w_ref.reason,
            "deployable_separation_ok": dep_ok, "deployable_genuinely_new": gn["genuinely_new"],
            "ew2_naive_fired": e_naive.found(), "ew2_ref_fired": e_ref.found(),
            "ew2_separation_ok": ew2_ok})
    n = len(corpus.train.problems)
    sep = {"n_train_problems": n,
           "deployable_separation_pass": dep_sep_ok, "deployable_separation_rate": dep_sep_ok / max(1, n),
           "ew2_separation_pass": ew2_sep_ok, "ew2_separation_rate": ew2_sep_ok / max(1, n),
           "deployable_vs_ew2_agreement": agree / max(1, n),
           "all_deployable_genuinely_new": all(r["deployable_genuinely_new"] for r in sep_rows),
           "by_family": sorted({r["family"] for r in sep_rows}), "rows": sep_rows}
    (OUT / "separation_characterization_v1.json").write_text(json.dumps(sep, indent=2, default=str))
    print(f"    deployable separation {dep_sep_ok}/{n} ; EW2 separation {ew2_sep_ok}/{n} ; "
          f"agreement {agree}/{n} ; all_genuinely_new={sep['all_deployable_genuinely_new']}")

    # ---- five Lane-alpha quality gates + regression fixtures ----
    print("[3] self-test gates + regression fixtures ...")
    sample = corpus.train.problems[0]
    st = tmpl_by_id[sample.problem_id]
    # 1. witness reproducibility (verdict): same naive -> same fired verdict
    w1 = build_deployable_witness_v1(st.naive_source, statement=sample.statement,
                                     samples=sample.samples, timeout_s=WITNESS_TIMEOUT_S)
    w2 = build_deployable_witness_v1(st.naive_source, statement=sample.statement,
                                     samples=sample.samples, timeout_s=WITNESS_TIMEOUT_S)
    g1_reproducible = bool(w1.found() == w2.found() == True)
    # 2. deterministic ladder (bytes)
    l1 = build_ladder_v1(sample.statement, sample.samples)
    l2 = build_ladder_v1(sample.statement, sample.samples)
    g2_ladder_det = bool(l1.cid() == l2.cid())
    # 3. public-spec-consistent stress
    g3_spec = all(_spec_consistent(p.statement, p.samples) for p in corpus.train.problems)
    # 4. naive/ref separation on every admitted train problem
    g4_separation = bool(dep_sep_ok == n)
    # 5. deterministic split regeneration (re-mint train seeds -> same split_cid)
    re_train = build_complexity_split_v1(complexity_slate_v1(), split="train", seeds=TRAIN_SEEDS,
                                         minted_date=corpus.minted_date)
    g5_regen = bool(re_train.split_cid == corpus.train.split_cid)
    # regression fixtures: deployable witness FIRES on the W132/W133 rescued complexity families
    rescue_families = ["cb_pairs_absdiff_le_d", "cb_distinct_in_windows", "cb_pairs_sum_eq_t",
                       "cb_subarrays_sum_eq_k"]
    fixt = {}
    for fam in rescue_families:
        pid = f"rbc_{fam}__s{TRAIN_SEEDS[0]}"
        p = next((q for q in corpus.train.problems if q.problem_id == pid), None)
        if p is None:
            fixt[fam] = {"present": False}
            continue
        t = tmpl_by_id[pid]
        w = build_deployable_witness_v1(t.naive_source, statement=p.statement, samples=p.samples,
                                        timeout_s=WITNESS_TIMEOUT_S)
        fixt[fam] = {"present": True, "naive_fires": w.found(), "reason": w.reason}
    fixtures_fire = all(v.get("naive_fires") for v in fixt.values() if v.get("present"))
    # negative control: a FAST value-wrong program must NOT fire the complexity witness
    fast_wrong = "import sys\nd=sys.stdin.buffer.read().split()\nprint(0)\n"
    neg = build_deployable_witness_v1(fast_wrong, statement=sample.statement,
                                      samples=sample.samples, timeout_s=WITNESS_TIMEOUT_S)
    neg_control_ok = bool(not neg.found())
    selftest = {
        "g1_witness_reproducible": g1_reproducible, "g2_ladder_deterministic": g2_ladder_det,
        "g3_public_spec_consistent": g3_spec, "g4_naive_ref_separation_all": g4_separation,
        "g5_split_regen_deterministic": g5_regen,
        "regression_fixtures": fixt, "regression_fixtures_all_fire": fixtures_fire,
        "negative_control_fast_wrong_silent": neg_control_ok,
        "all_pass": bool(g1_reproducible and g2_ladder_det and g3_spec and g4_separation
                         and g5_regen and fixtures_fire and neg_control_ok)}
    (OUT / "selftest_v1.json").write_text(json.dumps(selftest, indent=2, default=str))
    wall = time.time() - t0
    print(f"    g1_repro={g1_reproducible} g2_ladder={g2_ladder_det} g3_spec={g3_spec} "
          f"g4_separation={g4_separation} g5_regen={g5_regen}")
    print(f"    regression_fixtures_all_fire={fixtures_fire} negative_control_silent={neg_control_ok}")
    print(f"    ALL_PASS={selftest['all_pass']}  (wall {wall:.0f}s)")
    print(f"\nLOCKED CIDs (record in RUNBOOK_W134.md before any eval/frontier spend):")
    print(f"  corpus_cid           = {corpus.corpus_cid()}")
    print(f"  eval_split_cid       = {corpus.eval.split_cid}")
    print(f"  frontier_slice_cid   = {corpus.frontier_slice_cid}")
    return 0 if (corpus.meets_floors and selftest["all_pass"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
