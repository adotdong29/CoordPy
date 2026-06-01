"""W126 Lane α recon — $0 grade-cache + failure characterization of the resistant pool.

Reuses the EXACT W125 chain to load the deterministic W120 resistant 30-slice
(CID prefix ``01bf9ef869a56e20``) and the 330 already-paid Maverick generations
(11 per problem = [A0 | A1x5 | Bx5]).  Grades every generation on the OFFICIAL secret +
sample cases and writes a persistent cache so the synthesis precursor iterates fast.

This is the precursor diagnostic: which problems are uniformly unsolved, which failures
are VISIBLE (fail public samples — addressable by blind synthesis) vs HIDDEN
(``looks_right_fails_hidden`` — pass all samples, fail secret — non-discriminating), and
what typed digests the failures carry.  $0 NIM.

Emits results/w126/cache/grade_cache_v1.json and prints the split.
"""
from __future__ import annotations

import argparse
import collections
import datetime as _dt
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from coordpy.coordpy_icpc_battlefield_v1 import (  # noqa: E402
    classify_battlefield_listing_v1, core_slice_cid_v1, select_battlefield_core_slice_v1)
import coordpy.controller_native_code_mechanism_v1 as M  # noqa: E402
from coordpy.icpc_reflexion_bench_v1 import grade_on_secret_v1  # noqa: E402
from scripts.run_w120_icpc_pilot import load_pilot_problems  # noqa: E402

EXPECTED_SLICE_CID_30 = "01bf9ef869a56e20"
CORPUS_DIR = os.path.join(ROOT, "results", "w120", "icpc_pilot")
OUT_DIR = os.path.join(ROOT, "results", "w126", "cache")
ARM_LABELS = ["A0"] + [f"A1.{i}" for i in range(5)] + [f"B.{i}" for i in range(5)]


def _latest_corpus() -> str:
    name = open(os.path.join(CORPUS_DIR, "latest_run.txt")).read().strip()
    return os.path.join(CORPUS_DIR, name, "icpc_reflexion_calls.jsonl")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--timeout-s", type=float, default=15.0)
    args = ap.parse_args()

    full = classify_battlefield_listing_v1()
    slice30 = select_battlefield_core_slice_v1(full, n_problems=30)
    slice_cid = core_slice_cid_v1(slice30)
    if not slice_cid.startswith(EXPECTED_SLICE_CID_30):
        raise SystemExit(f"slice CID {slice_cid[:16]} != {EXPECTED_SLICE_CID_30}; refusing.")
    problems = load_pilot_problems(list(slice30))
    corpus = _latest_corpus()
    recs = [json.loads(l) for l in open(corpus)]
    if len(recs) != 11 * len(problems):
        raise SystemExit(f"corpus {len(recs)} != {11*len(problems)}; refusing.")
    pools = [M.build_pool_from_records(recs[i * 11:i * 11 + 11], problems[i].problem_id)
             for i in range(len(problems))]
    n = max(1, min(int(args.n), len(problems)))
    print(f"  slice cid={slice_cid[:16]}…  problems={n}  corpus={corpus.split('/')[-2]}")

    t0 = _dt.datetime.now(_dt.timezone.utc)
    cache: dict = {"schema": "coordpy.w126_grade_cache.v1", "slice_cid": slice_cid,
                   "corpus": corpus, "timeout_s": args.timeout_s, "problems": {}}
    digest_counter: collections.Counter = collections.Counter()
    n_solved = 0
    n_unsolved = 0
    n_unsolved_visible = 0   # no gen passes all samples -> blind-addressable
    n_unsolved_hidden = 0    # >=1 gen passes all samples but fails secret -> non-discriminating
    unsolved_ids: list[str] = []
    for pi in range(n):
        prob, pool = problems[pi], pools[pi]
        plane = M.AuditedGraderPlaneV1(prob, caller_agent_id="w126recon",
                                       timeout_s=float(args.timeout_s))
        codes = [pool.a0_code, *pool.a1_codes, *pool.b_codes]
        gens = []
        any_secret = False
        any_all_sample = False
        for arm, code in zip(ARM_LABELS, codes):
            sec, tail, ncase = grade_on_secret_v1(prob, code, timeout_s=float(args.timeout_s))
            sg = plane.grade_samples(code)
            dk = M._digest_key(sg.digest)
            digest_counter[sg.digest.exception_type or ("wrong" if not sg.all_pass else "ok")] += 1
            gens.append({
                "arm": arm, "code_sha": M._code_norm_sha(code), "parses": M._parses(code),
                "code_len": len(code), "secret_pass": bool(sec),
                "sample_pass": sg.n_pass, "sample_total": sg.n_total,
                "all_sample_pass": sg.all_pass, "secret_fail_tail": tail[:120],
                "digest_exc": sg.digest.exception_type, "digest_key": dk[:16]})
            any_secret = any_secret or sec
            any_all_sample = any_all_sample or sg.all_pass
        distinct_codes = len({g["code_sha"] for g in gens})
        distinct_digests = len({g["digest_key"] for g in gens})
        rec = {"short_name": prob.short_name, "n_samples": len(prob.samples),
               "n_secret": len(prob.secret_cases), "solved": any_secret,
               "any_all_sample_pass": any_all_sample,
               "distinct_codes": distinct_codes, "distinct_digests": distinct_digests,
               "gens": gens}
        cache["problems"][prob.problem_id] = rec
        if any_secret:
            n_solved += 1
        else:
            n_unsolved += 1
            unsolved_ids.append(prob.problem_id)
            if any_all_sample:
                n_unsolved_hidden += 1
            else:
                n_unsolved_visible += 1
        flag = "SOLVED" if any_secret else ("UNSOLVED/HIDDEN" if any_all_sample
                                            else "UNSOLVED/VISIBLE")
        print(f"   [{pi+1:2d}/{n}] {prob.short_name:28s} {flag:16s} "
              f"codes={distinct_codes:2d} digests={distinct_digests:2d} "
              f"n_secret={len(prob.secret_cases)}", flush=True)
    wall = (_dt.datetime.now(_dt.timezone.utc) - t0).total_seconds()
    cache["summary"] = {
        "n_problems": n, "n_solved": n_solved, "n_unsolved": n_unsolved,
        "n_unsolved_visible": n_unsolved_visible, "n_unsolved_hidden": n_unsolved_hidden,
        "unsolved_ids": unsolved_ids, "digest_dist": dict(digest_counter),
        "wall_s": round(wall, 1)}
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "grade_cache_v1.json")
    with open(out, "w") as f:
        json.dump(cache, f, indent=2, default=str)
    print()
    print(f"  SUMMARY: solved={n_solved}/{n}  unsolved={n_unsolved} "
          f"(VISIBLE/blind-addressable={n_unsolved_visible}, "
          f"HIDDEN/non-discriminating={n_unsolved_hidden})")
    print(f"  digest distribution: {dict(digest_counter)}")
    print(f"  wrote {out}  (wall {wall:.0f}s, $0 NIM)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
