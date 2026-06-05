#!/usr/bin/env python3
"""W137 Lane α — $0 build + self-test of the parser-neutral hard battlefield v2.

Asserts the locked slate fingerprint (drift guard), runs HC1 (parser-neutrality), HC2 (exact-oracle
discrimination), HC5 (template diversity), the W136 I/O-confound REGRESSION fixture (a flattened
input must FAIL HC1; the normal-form input must PASS), and deterministic regeneration + mint
timeout-invariance.  Writes ``results/w137/w137_build_selftest_v1.json``.  NO NIM.

Run:  .venv/bin/python scripts/run_w137_build_v2_battlefield_and_selftest_v1.py
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coordpy.hard_battlefield_slate_v2 import (  # noqa: E402
    build_hard_slate_v2, slate_fingerprint_cid_v1)
from coordpy.hard_battlefield_corpus_v2 import template_diversity_v1  # noqa: E402
from coordpy.parser_neutral_io_v1 import (  # noqa: E402
    io_shape, scalar_line, rows, render_normal_form_v1, parser_neutrality_gate_v1)
from coordpy.resistant_by_construction_battlefield_v1 import mint_problem_v1  # noqa: E402

LOCKED_SLATE_CID = "2ce207c567324e4322f308e58a1fc2c88a8d4bdd0e340d2ec8a1b867d82b3f70"
SELFTEST_SEED = 137_001
OUT = "results/w137/w137_build_selftest_v1.json"


def main() -> int:
    t0 = time.time()
    report: dict = {"schema": "w137_build_selftest_v1"}

    # --- drift guard ----------------------------------------------------------------
    fp = slate_fingerprint_cid_v1()
    report["slate_fingerprint_cid"] = fp
    report["slate_cid_locked_match"] = bool(fp == LOCKED_SLATE_CID)

    slate = build_hard_slate_v2()
    report["n_templates"] = len(slate)
    from collections import Counter
    report["mode_histogram"] = dict(Counter(t.minted.mode for t in slate))

    # --- HC5 diversity --------------------------------------------------------------
    div = template_diversity_v1()
    report["HC5_diversity"] = div.to_dict()

    # --- HC1 + HC2 on every template (mint timeout 3.0; timeout-invariant) ----------
    per_tpl = []
    n_ok = 0
    for t in slate:
        p = mint_problem_v1(t.minted, global_seed=SELFTEST_SEED, timeout_s=3.0)
        hc1 = parser_neutrality_gate_v1([i for i, _ in p.secret_cases], t.io_shape)
        ok = bool(p.gates.admitted and hc1.is_parser_neutral)
        n_ok += ok
        per_tpl.append({"name": t.minted.name, "mode": t.minted.mode,
                        "HC2_admitted": bool(p.gates.admitted), "HC2_reason": p.gates.reason,
                        "HC1_parser_neutral": bool(hc1.is_parser_neutral),
                        "n_brute_checked": p.gates.n_brute_checked,
                        "n_naive_secret_fail": p.gates.n_naive_secret_fail,
                        "naive_fail_kinds": sorted(set(p.gates.naive_fail_kinds))})
    report["per_template"] = per_tpl
    report["HC1_HC2_admitted"] = n_ok
    report["HC1_HC2_all_pass"] = bool(n_ok == len(slate))

    # --- W136 confound REGRESSION fixture -------------------------------------------
    # The exact W132 confound: a knapsack body flattened onto one line must FAIL HC1; the
    # normal-form (one item per line) must PASS.  This proves HC1 catches the W136 bug.
    knap_shape = io_shape(scalar_line("N", "W"), rows("ITEMS", "N", "w", "v"))
    nf_input = render_normal_form_v1(
        knap_shape, {"N": 3, "W": 50, "ITEMS": [(10, 60), (20, 100), (30, 120)]})
    flat_input = "3 50\n10 60 20 100 30 120\n"   # W132-style whitespace-flattened body
    hc1_nf = parser_neutrality_gate_v1([nf_input], knap_shape)
    hc1_flat = parser_neutrality_gate_v1([flat_input], knap_shape)
    report["confound_regression"] = {
        "nf_input_HC1_pass": bool(hc1_nf.is_parser_neutral),
        "flattened_input_HC1_pass": bool(hc1_flat.is_parser_neutral),
        "flattened_first_failure": hc1_flat.first_failure,
        "fixture_ok": bool(hc1_nf.is_parser_neutral and not hc1_flat.is_parser_neutral)}

    # --- deterministic regen + mint timeout-invariance ------------------------------
    t = slate[0]  # cb_count_inversions_v2 (a TIMEOUT template)
    p_a = mint_problem_v1(t.minted, global_seed=SELFTEST_SEED, timeout_s=3.0)
    p_b = mint_problem_v1(t.minted, global_seed=SELFTEST_SEED, timeout_s=3.0)
    p_c = mint_problem_v1(t.minted, global_seed=SELFTEST_SEED, timeout_s=8.0)
    report["determinism"] = {
        "regen_same_cid": bool(p_a.content_cid() == p_b.content_cid()),
        "timeout_invariant_cid": bool(p_a.content_cid() == p_c.content_cid()),
        "content_cid": p_a.content_cid()[:16]}

    report["all_selftests_pass"] = bool(
        report["slate_cid_locked_match"] and report["HC1_HC2_all_pass"]
        and div.all_distinct and report["confound_regression"]["fixture_ok"]
        and report["determinism"]["regen_same_cid"]
        and report["determinism"]["timeout_invariant_cid"])
    report["wall_s"] = round(time.time() - t0, 1)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(report, f, indent=2)

    print(f"slate_cid_locked_match : {report['slate_cid_locked_match']}")
    print(f"HC1+HC2 admitted       : {n_ok}/{len(slate)}")
    print(f"HC5 all_distinct       : {div.all_distinct} (max_jac={div.max_pairwise_jaccard:.4f}, modes={div.n_modes})")
    print(f"confound fixture ok    : {report['confound_regression']['fixture_ok']} "
          f"(nf_pass={hc1_nf.is_parser_neutral}, flat_pass={hc1_flat.is_parser_neutral})")
    print(f"determinism            : regen={report['determinism']['regen_same_cid']} "
          f"timeout_inv={report['determinism']['timeout_invariant_cid']}")
    print(f"ALL SELFTESTS PASS     : {report['all_selftests_pass']}  ({report['wall_s']}s)")
    print(f"wrote {OUT}")
    return 0 if report["all_selftests_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
