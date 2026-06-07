#!/usr/bin/env python3
"""W139 Lane β — capability-matched cross-tier mechanism bench (RUNBOOK_W139 §7a/§7b/§8).

For each ladder tier, mints THAT TIER'S OWN per-tier band slice (from the per-tier calibration) and
runs the same-budget arms.  The LEAD is ``Cm`` (the capability-matched controller): on a tier that is
NOT witness-eligible it KEEPs (plain self-consistency, ``Cm ≡ A1``, never hurts); on an eligible tier
it APPLYs the witness with a per-problem revert.  Diagnostic arms at the anchor: ``C0`` (blind-apply
complexity witness — reproduces the W138 +40 anchor win and the 8B −25 harm it caused) and ``Nb`` (the
large-probe counterexample — the 2nd-mode revival).  The §7a/§7b gate is computed at the anchor with
the cross-tier same-sign + non-negativity condition supplied as ``two_tier_same_sign``.

Same-budget: every arm K=5, attempt-0 standard, one model call/attempt, no early stop, graded on
secret; witness generation is $0 (owned-oracle on FRESH probes), outside K.

Run:  .venv/bin/python scripts/run_w139_cross_tier_bench_v1.py --mode dev
        [--calibration results/w139/w139_per_tier_calibration_v1.json] [--n-per-cell 2]
        [--diagnostics] [--tiers small,mid,strong] [--dry-run]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from coordpy.headroom_band_slate_v3 import (  # noqa: E402
    CX_FACTORIES, EXTRA_CX_FACTORIES, FUNC_FACTORIES, band_slate_fingerprint_cid_v1)
from coordpy.parser_neutral_io_v1 import parser_neutrality_gate_v1  # noqa: E402
from coordpy.resistant_by_construction_battlefield_v1 import mint_problem_v1  # noqa: E402
from coordpy.headroom_band_corpus_v3 import CorpusProblemV3, DEFAULT_MINTED_DATE  # noqa: E402
from coordpy.icpc_reflexion_bench_v1 import (  # noqa: E402
    IcpcArmOutcomeV1, IcpcBenchConfigV1, IcpcBenchReportV1, IcpcSeedReportV1,
    W120_ICPC_REFLEXION_BENCH_V1_SCHEMA_VERSION, run_icpc_reflexion_bench_v1)
from coordpy.exact_oracle_witness_v1 import (  # noqa: E402
    ARM_C1_COUNTEREXAMPLE, ARM_C2_COMPLEXITY, ARM_C3_CONTROLLER,
    build_witness_probe_set_v1, run_witness_arm_v1)
from coordpy.band_mechanism_bench_v1 import (  # noqa: E402
    arm_scored_on_problem_v1, evaluate_gate_v1, fake_different_report_v1)
from coordpy.per_tier_band_calibration_v1 import LADDER_V2  # noqa: E402
from coordpy.capability_matched_witness_compiler_v1 import (  # noqa: E402
    CM_ARM, NB_ARM, build_combined_probe_set_v1, build_large_probe_set_v1,
    run_capability_matched_arm_v1)
from scripts.run_w108_livecodebench_pilot import (  # noqa: E402
    _build_nim_gen, _evaluate_phase2_gates, _mlb_rates)

EXEC_TIMEOUT_S = 8.0
WITNESS_TIMEOUT_S = 2.0
WITNESS_SEED = 999_139
PROBE_KNOB_HIDDEN_EDGE = 1_500
NONNEG_TOL_PP = 1.0   # a tier counts as "non-negative" if Cm-A1 >= -1.0pp (one-draw noise)
W139_SPLIT_SEED_BASE = {"dev": 139_200_000, "eval": 139_300_000, "frontier": 139_400_000}


def _factory_for(fam: str):
    return CX_FACTORIES.get(fam) or FUNC_FACTORIES.get(fam) or EXTRA_CX_FACTORIES.get(fam)


def _probe_template_for(fam: str, mode: str, graded_knob: int):
    fac = _factory_for(fam)
    if mode == "HIDDEN_EDGE_STATE_MISS":
        return fac(min(PROBE_KNOB_HIDDEN_EDGE, int(graded_knob)))
    return fac(int(graded_knob))


def _mint_split_w139(split, *, templates, n_replicas, timeout_s, mint_timeout_s):
    """W139 mint honoring the locked 139_2/3/4 split bases (dev/eval/frontier disjoint)."""
    base = W139_SPLIT_SEED_BASE[split]
    seeds = [base + r for r in range(int(n_replicas))]
    out = []
    for t in templates:
        for s in seeds:
            p = mint_problem_v1(t.minted, global_seed=s, timeout_s=float(mint_timeout_s))
            hc1 = parser_neutrality_gate_v1([i for i, _ in p.secret_cases], t.io_shape)
            out.append(CorpusProblemV3(
                split=split, seed=s, cell_id=t.minted.name, family=t.minted.family,
                mode=t.minted.mode, minted=p, hc1_parser_neutral=bool(hc1.is_parser_neutral),
                gate_admitted=bool(p.gates.admitted)))
    return out


def _arm_report(model_id, base_seed_rep, arm_passed, arm_id, K):
    qids = base_seed_rep.question_ids
    b_passed = tuple(bool(x) for x in arm_passed)
    n = float(len(qids)) or 1.0
    b_acc = sum(1 for x in b_passed if x) / n
    seed_rep = IcpcSeedReportV1(
        schema=W120_ICPC_REFLEXION_BENCH_V1_SCHEMA_VERSION, seed=base_seed_rep.seed,
        n_problems=base_seed_rep.n_problems, a0_pass_at_1=base_seed_rep.a0_pass_at_1,
        a1_pass_at_1=base_seed_rep.a1_pass_at_1, b_pass_at_1=float(b_acc),
        per_problem_a0_passed=base_seed_rep.per_problem_a0_passed,
        per_problem_a1_passed=base_seed_rep.per_problem_a1_passed,
        per_problem_b_passed=b_passed, per_problem_b_first_pass_idx=tuple(0 for _ in b_passed),
        question_ids=qids, seed_merkle_root=f"w139_arm_{arm_id}_{base_seed_rep.seed}")
    return IcpcBenchReportV1(
        schema=W120_ICPC_REFLEXION_BENCH_V1_SCHEMA_VERSION, model_id=str(model_id),
        n_problems=base_seed_rep.n_problems, n_seeds=1, K_multi_sample=int(K),
        per_seed=(seed_rep,), a0_mean_pass_at_1=base_seed_rep.a0_pass_at_1,
        a1_mean_pass_at_1=base_seed_rep.a1_pass_at_1, b_mean_pass_at_1=float(b_acc),
        b_mean_minus_a1_mean_pp=float((b_acc - base_seed_rep.a1_pass_at_1) * 100.0),
        bench_merkle_root=f"w139_arm_{arm_id}")


def _run_diag_witness(arm_oracle, run_problems, tmpl_by_cell, probes_std, gen, seed, a1_passed,
                      max_tokens):
    """A blind-apply diagnostic witness arm (C0=complexity / Nb=large-probe counterexample) scored on
    its mode; inherits A1 where not scored (same-budget no-op)."""
    passed = []
    algo = []
    for i, p in enumerate(run_problems):
        scored = (arm_oracle == ARM_C2_COMPLEXITY and p.mode == "COMPLEXITY_BLIND") or \
                 (arm_oracle == ARM_C1_COUNTEREXAMPLE and p.mode != "COMPLEXITY_BLIND")
        if not scored:
            passed.append(bool(a1_passed[i]))
            algo.append(True)
            continue
        o, tr = run_witness_arm_v1(
            seed=seed, template=tmpl_by_cell[p.cell_id].minted, problem=p.minted,
            probe=probes_std[p.minted.problem_id], gen=gen, K=5, temperature=0.7,
            max_tokens=int(max_tokens), timeout_s=EXEC_TIMEOUT_S, arm=arm_oracle,
            minted_date=DEFAULT_MINTED_DATE, witness_timeout_s=WITNESS_TIMEOUT_S)
        passed.append(bool(o.final_passed))
        algo.append(bool(tr.rescue_is_algorithmic()))
    return passed, algo


def run_tier(*, tier, model_id, witness_eligible, band_cells, mode, n_per_cell, max_tokens,
             mint_timeout, out_dir, diagnostics):
    """Run A0/A1/B0 + Cm (+ C0/Nb diagnostics) on one tier's OWN per-tier band slice."""
    templates = []
    for fam, (cell_id, knob) in sorted(band_cells.items()):
        fac = _factory_for(fam)
        if fac is None:
            continue
        templates.append(fac(int(knob)))
    if not templates:
        return {"tier": tier, "model": model_id, "n_problems": 0, "skipped": "no band cells"}
    tmpl_by_cell = {t.minted.name: t for t in templates}
    knob_by_family = {fam: int(knob) for fam, (cid, knob) in band_cells.items()}
    seed = W139_SPLIT_SEED_BASE[mode]
    minted = _mint_split_w139(mode, templates=templates, n_replicas=n_per_cell,
                              timeout_s=EXEC_TIMEOUT_S, mint_timeout_s=mint_timeout)
    run_problems = [p for p in minted if p.admitted]
    if not run_problems:
        return {"tier": tier, "model": model_id, "n_problems": 0, "skipped": "no admitted instances"}

    sc = open(os.path.join(out_dir, f"calls_{tier}.jsonl"), "w")
    gen = _build_nim_gen(model=model_id,
                         sidecar_writer=lambda r: (sc.write(json.dumps(r) + "\n"), sc.flush()))
    pilots = [p.to_pilot(minted_date=DEFAULT_MINTED_DATE) for p in run_problems]
    cfg = IcpcBenchConfigV1(K_multi_sample=5, seeds=(seed,), sampling_temperature=0.7,
                            max_tokens_per_call=int(max_tokens), executor_timeout_s=EXEC_TIMEOUT_S)
    print(f"  [{tier}] baseline A0/A1/B0 on {len(pilots)} problems ({model_id}) "
          f"eligible={witness_eligible}...", flush=True)
    base = run_icpc_reflexion_bench_v1(gen=gen, model_id=model_id, subset=pilots, config=cfg)
    bsr = base.per_seed[0]
    per_a1 = list(bsr.per_problem_a1_passed)
    per_b0 = list(bsr.per_problem_b_passed)

    # Cm: the capability-matched controller (combined large-probe + complexity stress; C3 routing)
    cm_pass = []
    cm_algo = []
    cm_traces = []
    for i, p in enumerate(run_problems):
        ptmpl = _probe_template_for(p.family, p.mode, knob_by_family[p.family])
        combined = build_combined_probe_set_v1(
            graded_template=tmpl_by_cell[p.cell_id].minted, probe_template=ptmpl.minted,
            problem=p.minted, witness_seed=WITNESS_SEED)
        o, tr = run_capability_matched_arm_v1(
            seed=seed, template=tmpl_by_cell[p.cell_id].minted, problem=p.minted, probe=combined,
            gen=gen, K=5, temperature=0.7, max_tokens=int(max_tokens), timeout_s=EXEC_TIMEOUT_S,
            minted_date=DEFAULT_MINTED_DATE, witness_eligible=witness_eligible,
            witness_arm=ARM_C3_CONTROLLER, witness_timeout_s=WITNESS_TIMEOUT_S)
        cm_pass.append(bool(o.final_passed))
        cm_algo.append(bool(tr.rescue_is_algorithmic()))
        cm_traces.append(tr.to_dict())
        print(f"    [{tier}] Cm {i+1}/{len(run_problems)} {p.cell_id} pass={o.final_passed} "
              f"act={tr.actions[:2]}..", flush=True)

    diag = {}
    if diagnostics:
        probes_std = {p.minted.problem_id: build_witness_probe_set_v1(
            tmpl_by_cell[p.cell_id].minted, p.minted, witness_seed=WITNESS_SEED,
            timeout_s=EXEC_TIMEOUT_S) for p in run_problems}
        c0_pass, c0_algo = _run_diag_witness(ARM_C2_COMPLEXITY, run_problems, tmpl_by_cell,
                                             probes_std, gen, seed, per_a1, max_tokens)
        # Nb uses the LARGE probe set for the counterexample search
        probes_lg = {p.minted.problem_id: build_large_probe_set_v1(
            _probe_template_for(p.family, p.mode, knob_by_family[p.family]).minted, p.minted,
            witness_seed=WITNESS_SEED) for p in run_problems}
        nb_pass = []
        nb_algo = []
        for i, p in enumerate(run_problems):
            if p.mode == "COMPLEXITY_BLIND":
                nb_pass.append(bool(per_a1[i]))
                nb_algo.append(True)
                continue
            o, tr = run_witness_arm_v1(
                seed=seed, template=_probe_template_for(p.family, p.mode, knob_by_family[p.family]).minted,
                problem=p.minted, probe=probes_lg[p.minted.problem_id], gen=gen, K=5,
                temperature=0.7, max_tokens=int(max_tokens), timeout_s=EXEC_TIMEOUT_S,
                arm=ARM_C1_COUNTEREXAMPLE, minted_date=DEFAULT_MINTED_DATE,
                witness_timeout_s=WITNESS_TIMEOUT_S)
            nb_pass.append(bool(o.final_passed))
            nb_algo.append(bool(tr.rescue_is_algorithmic()))
        diag = {"C0": {"passed": c0_pass, "algo": c0_algo, "acc": sum(c0_pass) / len(c0_pass)},
                "Nb": {"passed": nb_pass, "algo": nb_algo, "acc": sum(nb_pass) / len(nb_pass)}}
    sc.close()

    n = float(len(run_problems))
    cm_acc = sum(cm_pass) / n
    return {
        "tier": tier, "model": model_id, "witness_eligible": bool(witness_eligible),
        "n_problems": len(run_problems),
        "cell": [p.cell_id for p in run_problems], "mode": [p.mode for p in run_problems],
        "family": [p.family for p in run_problems],
        "a0": base.a0_mean_pass_at_1, "a1": base.a1_mean_pass_at_1, "b0": base.b_mean_pass_at_1,
        "cm_acc": cm_acc, "cm_minus_a1_pp": round((cm_acc - base.a1_mean_pass_at_1) * 100, 2),
        "cm_minus_b0_pp": round((cm_acc - base.b_mean_pass_at_1) * 100, 2),
        "per_a1": [bool(x) for x in per_a1], "per_b0": [bool(x) for x in per_b0],
        "cm_passed": cm_pass, "cm_algo": cm_algo, "cm_traces": cm_traces, "diag": diag,
        "base_seed_rep_a1": list(per_a1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dev", "eval", "frontier"], default="dev")
    ap.add_argument("--calibration", default="results/w139/w139_per_tier_calibration_v1.json")
    ap.add_argument("--n-per-cell", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--mint-timeout", type=float, default=1.0)
    ap.add_argument("--tiers", default="small,mid,strong")
    ap.add_argument("--diagnostics", action="store_true",
                    help="also run C0 (blind complexity witness) + Nb (large-probe counterexample) "
                         "at EVERY tier — shows the 8B -25 harm Cm avoids and whether the large-probe "
                         "2nd mode gives a second positive tier on the mid")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    with open(args.calibration) as f:
        cal = json.load(f)
    assert cal.get("slate_fingerprint_cid") == band_slate_fingerprint_cid_v1(
        cx_knobs=tuple(cal.get("cx_knobs", (6000, 20000, 50000))),
        func_knobs=tuple(cal.get("func_knobs", (1500, 4000, 30000)))), "SLATE DRIFT — refusing NIM"
    band_for_tier = cal.get("band_for_tier", {})
    eligible_tiers = set(cal.get("witness_eligible_tiers", []))
    want_tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]
    ladder = [m for m in LADDER_V2 if m.tier in want_tiers]
    anchor_tier = cal.get("anchor_tier", "strong")

    print(f"per_tier_calibration_cid: {cal.get('per_tier_calibration_cid')}")
    print(f"witness-eligible tiers: {sorted(eligible_tiers)}")
    for m in ladder:
        cells = band_for_tier.get(m.tier, {})
        print(f"  {m.tier:6s} band: " + ", ".join(f"{f}@{c['knob_value']}" for f, c in cells.items()))
    fd = fake_different_report_v1(real_arm_ids=("Cm", "Nb", "C0")).to_dict()
    print(f"fake-different (Cm/Nb/C0 REAL, M3/B0 FAKE): bites={fd['bites']}")
    if args.dry_run:
        print("--dry-run: stopping before NIM.")
        return 0

    run_id = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = os.path.join(ROOT, "results", "w139", args.mode, f"w139_{args.mode}_{run_id}")
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    tier_results = {}
    for m in ladder:
        cells = {f: (c["cell_id"], c["knob_value"]) for f, c in band_for_tier.get(m.tier, {}).items()}
        tier_results[m.tier] = run_tier(
            tier=m.tier, model_id=m.model_id, witness_eligible=(m.tier in eligible_tiers),
            band_cells=cells, mode=args.mode, n_per_cell=args.n_per_cell, max_tokens=args.max_tokens,
            mint_timeout=args.mint_timeout, out_dir=out_dir, diagnostics=args.diagnostics)

    # cross-tier verdict
    deltas = {t: r.get("cm_minus_a1_pp", 0.0) for t, r in tier_results.items() if r.get("n_problems")}
    anchor = tier_results.get(anchor_tier, {})
    anchor_pos = bool(anchor.get("cm_minus_a1_pp", 0.0) > 0)
    n_positive = sum(1 for d in deltas.values() if d > 0)
    all_nonneg = all(d >= -NONNEG_TOL_PP for d in deltas.values())
    two_tier_same_sign = bool(anchor_pos and n_positive >= 2 and all_nonneg)

    gate = None
    if anchor.get("n_problems"):
        gate = evaluate_gate_v1(
            name=("dev_gate" if args.mode == "dev" else "eval_earn"),
            per_lead=anchor["cm_passed"], per_a1=anchor["per_a1"], per_b0=anchor["per_b0"],
            modes=anchor["mode"], families=anchor["family"], rescue_is_structural=anchor["cm_algo"],
            margin_pp=(3.33 if args.mode == "dev" else 5.00),
            two_tier_same_sign=two_tier_same_sign).to_dict()

    wall = time.time() - t0
    report = {"schema": "w139_cross_tier_bench_v1", "mode": args.mode,
              "per_tier_calibration_cid": cal.get("per_tier_calibration_cid"),
              "slate_fingerprint_cid": cal.get("slate_fingerprint_cid"),
              "anchor_tier": anchor_tier, "eligible_tiers": sorted(eligible_tiers),
              "tier_results": tier_results, "cross_tier_deltas_pp": deltas,
              "anchor_positive": anchor_pos, "n_positive_tiers": n_positive,
              "all_tiers_nonneg": all_nonneg, "two_tier_same_sign": two_tier_same_sign,
              "fake_different": fd, "gate": gate, "wall_s": round(wall, 1)}
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)

    print("\n=== W139 CROSS-TIER BENCH ===")
    for t, r in tier_results.items():
        if not r.get("n_problems"):
            print(f"  {t:6s}: SKIP ({r.get('skipped')})")
            continue
        line = (f"  {t:6s} (elig={r['witness_eligible']}): A0={r['a0']*100:.1f} A1={r['a1']*100:.1f} "
                f"B0={r['b0']*100:.1f} Cm={r['cm_acc']*100:.1f}  "
                f"Cm-A1={r['cm_minus_a1_pp']:+.2f} Cm-B0={r['cm_minus_b0_pp']:+.2f}")
        if r.get("diag"):
            line += (f"  [C0={r['diag']['C0']['acc']*100:.1f} Nb={r['diag']['Nb']['acc']*100:.1f}]")
        print(line)
    print(f"cross-tier: anchor_positive={anchor_pos} n_positive={n_positive} "
          f"all_nonneg={all_nonneg} -> two_tier_same_sign={two_tier_same_sign}")
    if gate:
        print(f"§7a/§7b gate (lead=Cm): passed={gate['passed']} reason={gate['reason']}")
        print(f"  Cm-A1={gate['lead_minus_a1_pp']:+.2f} Cm-B0={gate['lead_minus_b0_pp']:+.2f} "
              f"span_ok={gate['span_ok']} modes={gate['rescue_modes']} fams={gate['rescue_families']}")
    print(f"wall {wall:.0f}s -> {out_dir}/report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
