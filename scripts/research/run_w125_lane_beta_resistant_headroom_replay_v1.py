"""W125 Lane β — $0 resistant headroom replay over the already-paid Maverick corpus.

Reloads the deterministic W120 resistant 30-slice (CID prefix ``01bf9ef869a56e20``;
packages cached offline), reads the 330 already-paid real Maverick generations from
``results/w120/icpc_pilot/<latest_run>/icpc_reflexion_calls.jsonl`` (11 per problem =
[A0 | A1×5 | B×5]), re-grades every generation on the OFFICIAL secret + sample cases, and
runs the controller-native headroom probe + the pilot earn gate (RUNBOOK § 6).

This spends $0 NIM — it re-routes generations Maverick already produced and grades with the
local official oracle. A FRESH hosted pilot is gated on the emitted earn verdict.

Emits results/w125/lane_beta/{headroom_verdict.json}.
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
from scripts.run_w120_icpc_pilot import load_pilot_problems  # noqa: E402

EXPECTED_SLICE_CID_30 = "01bf9ef869a56e20"
CORPUS_DIR = os.path.join(ROOT, "results", "w120", "icpc_pilot")
OUT_DIR = os.path.join(ROOT, "results", "w125", "lane_beta")


def _latest_corpus() -> str:
    name = open(os.path.join(CORPUS_DIR, "latest_run.txt")).read().strip()
    return os.path.join(CORPUS_DIR, name, "icpc_reflexion_calls.jsonl")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="limit #problems (timing/debug)")
    ap.add_argument("--timeout-s", type=float, default=15.0)
    args = ap.parse_args()

    full = classify_battlefield_listing_v1()
    slice30 = select_battlefield_core_slice_v1(full, n_problems=30)
    slice_cid = core_slice_cid_v1(slice30)
    if not slice_cid.startswith(EXPECTED_SLICE_CID_30):
        raise SystemExit(f"slice CID {slice_cid[:16]} != {EXPECTED_SLICE_CID_30}; refusing.")
    run_slice = list(slice30)
    problems = load_pilot_problems(run_slice)
    print(f"  loaded {len(problems)} resistant problems; slice cid={slice_cid[:16]}…")

    corpus = _latest_corpus()
    recs = [json.loads(l) for l in open(corpus)]
    if len(recs) != 11 * len(problems):
        raise SystemExit(f"corpus has {len(recs)} records, expected {11*len(problems)} "
                         f"(11 per problem); refusing.")
    pools = [M.build_pool_from_records(recs[i * 11:i * 11 + 11], problems[i].problem_id)
             for i in range(len(problems))]
    print(f"  built {len(pools)} generation pools from {len(recs)} real Maverick calls "
          f"({corpus.split('/')[-2]})")

    n = max(1, min(int(args.n), len(problems)))
    probs = problems[:n]
    pls = pools[:n]

    def on_problem(i, qid, tally):
        print(f"    [{i+1}/{n}] {qid[:46]:46s} a1={tally['a1_pass']} "
              f"union={tally['pool_union']} c3={tally['c3_pass']} "
              f"blind_headroom={tally['blind_headroom']} diverge={tally['divergence']}",
              flush=True)

    t0 = _dt.datetime.now(_dt.timezone.utc)
    hr = M.headroom_probe(probs, pls, field="w120_resistant_icpc",
                          K=5, timeout_s=float(args.timeout_s), on_problem=on_problem)
    wall_s = (_dt.datetime.now(_dt.timezone.utc) - t0).total_seconds()

    # structural slate + contract on the controlled synthetic substrate (the contract
    # checks verify the controller's INTRINSIC properties — audit chain, determinism,
    # never-reads-secret, same-budget — best done on a deterministic input; the resistant
    # headroom above is the real-field measurement).
    slate = M.evaluate_mechanism_slate()
    sc_prob, sc_pool = M.synthetic_contract_problem()
    contract = M.run_contract_checks(sc_pool, sc_prob, K=5)
    gate = M.apply_pilot_earn_gate(contract, slate, hr)

    verdict = {
        "schema": "coordpy.w125_lane_beta_headroom.v1", "lane": "beta_resistant_first",
        "verified_on": "2026-05-31", "field": "W120 resistant official-ICPC 30-slice",
        "slice_cid": slice_cid, "n_problems_probed": n, "corpus": corpus,
        "nim_spend": 0, "wall_s": round(wall_s, 1),
        "w120_baseline": {"a0_pct": 20.00, "a1_pct": 23.33, "b_pct": 23.33,
                          "b_minus_a1_pp": 0.00, "mlb1": 0.8333, "mlb2": 0.08,
                          "verdict": "FAIL/BOUNDED_CEILING_HOLDS_ON_RESISTANT_ICPC"},
        "headroom": hr.to_dict(),
        "slate": {"lead": slate.lead, "real": list(slate.real_candidates),
                  "fake": list(slate.fake_candidates)},
        "contract": contract.to_dict(),
        "earn_gate": gate.to_dict(),
        "null_band_pp": 3.34, "retirement_bar_pp": 5.00,
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "headroom_verdict.json")
    with open(out, "w") as f:
        json.dump(verdict, f, indent=2, default=str)

    print()
    print(f"  HEADROOM: a1_pass={hr.a1_pass_count}/{n} pool_union={hr.pool_union_secret_count} "
          f"oracle_headroom={hr.oracle_pool_headroom} blind_selection_headroom="
          f"{hr.blind_selection_headroom} reflexion_divergence={hr.reflexion_divergence} "
          f"looks_right_fails_hidden={hr.looks_right_fails_hidden}")
    print(f"  CONTRACT all_pass={contract.all_pass} | LEAD={slate.lead}")
    print(f"  EARN GATE: {gate.verdict_label} (earned={gate.earned})")
    print(f"    {gate.rationale}")
    print(f"  wrote {out}  (wall {wall_s:.0f}s, $0 NIM)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
