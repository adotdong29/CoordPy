#!/usr/bin/env python3
"""W103 — emit RESULTS_W103_HUMANEVAL_PLUS_PHASE2_70B_V1.md
from the cheap-pilot bench report.

Reads the latest
`results/w103/humaneval_plus_pilot/<RUN>/humaneval_plus_reflexion_bench_report.json`
and produces a populated Markdown verdict matching the W103
RUNBOOK template (helper-anchored slice + provenance + 9
Phase 2 gates + MLB-1 + MLB-2 sub-gates + branch decision
logic).

Usage::

    python scripts/_w103_emit_pilot_result_doc.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _latest_pilot_report() -> Path:
    latest_ptr = (
        ROOT / "results" / "w103" / "humaneval_plus_pilot"
        / "latest_run.txt")
    if not latest_ptr.exists():
        raise FileNotFoundError(
            f"pilot latest_run.txt not found at {latest_ptr}")
    pointer = latest_ptr.read_text().strip()
    return (
        latest_ptr.parent / pointer
        / "humaneval_plus_reflexion_bench_report.json")


def _label_pass(b: bool) -> str:
    return "PASS" if b else "FAIL"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--bench-report", default=None,
        help="Path to bench report JSON (defaults to latest)")
    ap.add_argument(
        "--out",
        default=str(
            ROOT / "docs"
            / "RESULTS_W103_HUMANEVAL_PLUS_PHASE2_70B_V1.md"))
    args = ap.parse_args()
    report_path = (
        Path(args.bench_report)
        if args.bench_report else _latest_pilot_report())
    if not report_path.exists():
        print(
            f"ERROR: bench report not found at {report_path}",
            file=sys.stderr)
        return 2
    rep = json.loads(report_path.read_text())
    seeds = rep.get("per_seed") or []
    if not seeds:
        print("ERROR: no per_seed in bench report",
              file=sys.stderr)
        return 2
    s = seeds[0]
    gates = (rep.get("phase2_evaluation") or {})
    mlb = rep.get("mlb") or {}
    prov = rep.get("provenance") or {}
    a0 = float(rep["a0_mean_pass_at_1"]) * 100
    a1 = float(rep["a1_mean_pass_at_1"]) * 100
    b = float(rep["b_mean_pass_at_1"]) * 100
    margin = float(rep["b_mean_minus_a1_mean_pp"])
    margin_a0 = b - a0
    n_problems = int(s.get("n_problems", 30))
    wall_s = float(rep.get("wall_s", 0.0))
    bench_merkle = str(rep.get("bench_merkle_root", ""))
    gates_dict = gates.get("phase2_gates") or {}
    n_phase2_passed = int(gates.get("n_phase2_passed_of_9", 0))
    verdict_label = str(gates.get("verdict_label", "UNKNOWN"))
    mlb1_passes = bool(mlb.get("mlb1_passes", False))
    mlb2_passes = bool(mlb.get("mlb2_passes", False))
    mlb1_rate = float(
        mlb.get("mlb1_invocation_rate", 0)) * 100
    mlb2_rate = float(
        mlb.get("mlb2_rescue_rate", 0)) * 100
    n_invoked = int(mlb.get("n_b_invoked_reflexion", 0))
    n_rescued = int(mlb.get("n_b_rescued_via_reflexion", 0))
    n_problems_total = int(mlb.get("n_problems_total", 0))
    per_problem_a0 = list(s.get("per_problem_a0_passed", []))
    per_problem_a1 = list(s.get("per_problem_a1_passed", []))
    per_problem_b = list(s.get("per_problem_b_passed", []))
    n_b_ge_a1 = sum(
        1 for i in range(len(per_problem_a1))
        if not (per_problem_a1[i] and not per_problem_b[i]))
    n_a1_only = sum(
        1 for i in range(len(per_problem_a1))
        if per_problem_a1[i] and not per_problem_b[i])
    n_b_only = sum(
        1 for i in range(len(per_problem_a1))
        if per_problem_b[i] and not per_problem_a1[i])
    n_shared_wins = sum(
        1 for i in range(len(per_problem_a1))
        if per_problem_a1[i] and per_problem_b[i])
    n_shared_fails = sum(
        1 for i in range(len(per_problem_a1))
        if not per_problem_a1[i] and not per_problem_b[i])
    bench_task_ids = list(prov.get(
        "bench_iteration_task_ids", []))
    md: list[str] = []
    md.append(
        "# W103 — HumanEval+ cheap pilot Phase 2 70B V1")
    md.append("")
    md.append(
        "> **2026-05-25.  Cheap HumanEval+ pilot verdict at "
        "Llama-3.3-70B-Instruct on 1 seed × 30 problems × "
        "K=5 = 330 NIM calls.  Slice helper-anchored "
        "(`coordpy.code_slice_selector_v1` consumption; COO-14 "
        "downstream).  Pre-committed W95 9-gate Phase 2 + W101 "
        "MLB-1 + MLB-2 sub-gate evaluation.  Pilot CONDITIONAL "
        "on the W102 HumanEval+ preflight re-confirmation "
        "(7/7 PASS, verdict cid `4f57a2cf...`) + helper-"
        "anchored slice (CID "
        f"`{prov.get('slice_cid_helper_priority', '')}`).**")
    md.append("")
    md.append("## Inputs (provenance)")
    md.append("")
    md.append("| Field | Value |")
    md.append("|---|---|")
    md.append(
        "| Candidate mechanism | B (W89 sequential reflexion "
        "on HumanEval+ at K=5) |")
    md.append(
        "| Target model | `meta/llama-3.3-70b-instruct` |")
    md.append(
        f"| HumanEval+ corpus SHA-256 | "
        f"`{prov.get('corpus_sha256', '')}` |")
    md.append(
        f"| Preflight verdict cid (re-confirmed W103) | "
        f"`{prov.get('preflight_verdict_cid', '')}` |")
    md.append(
        f"| Helper proposal CID (humaneval_plus) | "
        f"`{prov.get('helper_proposal_cid_humaneval_plus', '')}` |")
    md.append(
        f"| Helper proposal CID (humaneval top-up) | "
        f"`{prov.get('helper_proposal_cid_humaneval_topup', '')}` |")
    md.append(
        f"| Mining report CID | "
        f"`{prov.get('mining_report_cid', '')}` |")
    md.append(
        f"| Slice CID (helper-priority order) | "
        f"`{prov.get('slice_cid_helper_priority', '')}` |")
    md.append(
        f"| Slice CID (bench iteration order) | "
        f"`{prov.get('slice_cid_bench_order', '')}` |")
    md.append(
        f"| Bench Merkle root | `{bench_merkle}` |")
    md.append(
        f"| Pilot wall | {wall_s:.1f} s |")
    md.append(
        f"| Seed (candidate sampling RNG) | "
        f"{prov.get('seed', 103001)} |")
    md.append(
        f"| n_problems | {n_problems} |")
    md.append(
        "| Arsenal-mining prior (RECORDED; NOT a Phase 2 gate "
        "input) | "
        f"B−A1 = +{float(prov.get('arsenal_mining_prior_humaneval_plus', {}).get('b_minus_a1_pp', 0)):.2f} pp; "
        f"rescue = {float(prov.get('arsenal_mining_prior_humaneval_plus', {}).get('rescue_fraction', 0))*100:.2f} % "
        "(W102 cross-bench extension) |")
    md.append("")
    md.append("## Headline numbers")
    md.append("")
    md.append("| Arm | Pass rate (n / N) |")
    md.append("|---|---:|")
    md.append(
        f"| A0 | {a0:.2f}% ({sum(per_problem_a0)} / {n_problems}) |")
    md.append(
        f"| A1 @ K=5 | {a1:.2f}% ({sum(per_problem_a1)} / {n_problems}) |")
    md.append(
        f"| B (sequential reflexion K=5) | {b:.2f}% "
        f"({sum(per_problem_b)} / {n_problems}) |")
    md.append(
        f"| **B − A1** | **{margin:+.2f} pp** |")
    md.append(
        f"| **B − A0** | **{margin_a0:+.2f} pp** |")
    md.append("")
    md.append("## Per-problem cluster surface")
    md.append("")
    md.append("| Cluster | Count |")
    md.append("|---|---:|")
    md.append(f"| a1_only_wins (B regression) | {n_a1_only} |")
    md.append(f"| b_only_wins (B rescue) | {n_b_only} |")
    md.append(f"| shared_wins | {n_shared_wins} |")
    md.append(f"| shared_fails (hard cluster) | {n_shared_fails} |")
    md.append("")
    md.append("## Phase 2 gates")
    md.append("")
    md.append("| # | Gate | Verdict |")
    md.append("|---|---|---|")
    for idx, (key, label) in enumerate([
        ("G1_slice_pre_committed",
         "Slice pre-committed (helper-anchored; 30 problems; "
         "CID locked BEFORE NIM call)"),
        ("G2_a1_lt_90pct", "A1 @ K=5 < 90 %"),
        ("G3_b_gt_a1", "B > A1"),
        ("G4_margin_geq_5pp", "B − A1 ≥ +5 pp"),
        ("G5_b_gt_a0_by_geq_5pp", "B > A0 by ≥ +5 pp"),
        ("G6_per_problem_majority",
         f"Per-problem majority B ≥ A1 (≥ 16/30; observed "
         f"{n_b_ge_a1}/{n_problems})"),
        ("G7_budget_exact",
         "Budget exact (1 + 5 + 5 = 11 calls / problem)"),
        ("G8_audit_chain_re_derives",
         "Audit chain re-derives offline"),
        ("G9_executor_clean", "Executor stays clean"),
    ], start=1):
        v = gates_dict.get(key)
        md.append(
            f"| {idx} | {label} | "
            f"{'PASS' if bool(v) else 'FAIL'} |")
    md.append("")
    md.append("## MLB sub-gates (mechanism-load-bearingness)")
    md.append("")
    md.append("| Sub-gate | Threshold | Observed | Verdict |")
    md.append("|---|---:|---:|---|")
    md.append(
        f"| **MLB-1** reflexion-cycle invocation rate | "
        f"≥ 33 % | {mlb1_rate:.2f} % "
        f"({n_invoked}/{n_problems_total}) | "
        f"{_label_pass(mlb1_passes)} |")
    md.append(
        f"| **MLB-2** reflexion rescue rate | ≥ 33 % | "
        f"{mlb2_rate:.2f} % "
        f"({n_rescued}/{n_invoked if n_invoked else 0}) | "
        f"{_label_pass(mlb2_passes)} |")
    md.append("")
    md.append("## Verdict")
    md.append("")
    md.append(
        f"**{n_phase2_passed} of 9 Phase 2 gates PASS.**  "
        f"MLB sub-gates: MLB-1 = {_label_pass(mlb1_passes)}, "
        f"MLB-2 = {_label_pass(mlb2_passes)}.")
    md.append("")
    md.append(f"**Verdict label: `{verdict_label}`**.")
    md.append("")
    md.append(
        "### Decision applied per the pre-committed W103 "
        "RUNBOOK")
    md.append("")
    if verdict_label == "PASS_MECHANISM_DRIVEN":
        md.append(
            "* Cheap pilot PASSes 9/9 + MLB sub-gates clearing "
            "→ **Branch A** of the W103 RUNBOOK § Planning lane.")
        md.append(
            "* **W104 = HumanEval+ cross-scale confirmation** "
            "at a SECOND model class (per the W96-C / W100 "
            "cross-scale discipline).  W104 RUNBOOK locks the "
            "exact target model.")
        md.append(
            "* W105+ = HumanEval+ Phase 3 retirement bench (3 "
            "seeds × 100 problems × K=5) IF W104 cross-scale "
            "PASSes.")
        md.append(
            "* Carry-forward registered: "
            "`W103-L-HUMANEVAL-PLUS-REFLEXION-PHASE2-70B-PASS` "
            "(Phase 2 cheap-pilot single-seed PASS; NOT a "
            "retirement).")
        md.append(
            "* `COO-9` remains the lead path — the W89 mechanism "
            "extends to a SECOND EvalPlus benchmark family at "
            "the cheap-pilot scale.")
        md.append(
            "* Programme entitlement: the W89 sequential-"
            "reflexion mechanism extends to a SECOND published "
            "code benchmark family (HumanEval+ EvalPlus-hardened) "
            "at the cheap-pilot scale.  This is stronger than "
            "W102 alone (which FAILed on MBPP+ V2) but weaker "
            "than a multi-seed Phase 3 retirement.")
    elif verdict_label == "PASS_NON_MECHANISM_DRIVEN":
        md.append(
            "* Cheap pilot PASSes 9/9 BUT MLB-2 FAILs → "
            "`PASS_NON_MECHANISM_DRIVEN`; **Branch B** of the "
            "W103 RUNBOOK § Planning lane.")
        md.append(
            "* **W104 = HumanEval+ mechanism-variation slate** "
            "(parallel B-style variants: B1 = enforced-"
            "reflexion-on-attempt-0; B2 = test-aware "
            "decomposition reader+solver on the EvalPlus extra-"
            "test surface; B3 = sidecar-driven failure-cluster "
            "targeting).  Pre-commit cheap-pilot earning rule: "
            "at least one B-variant must lift MLB-2 ≥ 33 % AND "
            "keep margin ≥ +5 pp.")
        md.append(
            "* Carry-forwards added: "
            "`W103-L-HUMANEVAL-PLUS-REFLEXION-PHASE2-70B-"
            "PASS-NON-MECHANISM-DRIVEN-CAP` + "
            "`W103-L-HUMANEVAL-PLUS-MECHANISM-LOAD-BEARINGNESS-"
            "WEAK-AT-70B-CAP`.")
        md.append(
            "* Cross-scale W104 NOT entitled per W96-C / W100 / "
            "W101 / W102 precedent.")
        md.append(
            "* `COO-9` remains the lead path; W104 attacks "
            "mechanism load-bearingness directly.")
    else:  # FAIL
        md.append(
            "* Cheap pilot FAILs → **Branch C** of the W103 "
            "RUNBOOK § Planning lane.")
        md.append(
            f"* Carry-forward added: "
            "`W103-L-HUMANEVAL-PLUS-REFLEXION-PHASE2-70B-CAP` "
            f"(B − A1 = {margin:+.2f} pp; "
            f"MLB-1 = {mlb1_rate:.2f} % "
            f"{_label_pass(mlb1_passes)}; "
            f"MLB-2 = {mlb2_rate:.2f} % "
            f"{_label_pass(mlb2_passes)}).")
        md.append(
            "* **W104 = code-line ranking refresh** with the "
            "pre-committed triage logic from the W103 RUNBOOK § "
            "Planning lane.  Dispatch decision recorded in the "
            "milestone summary based on the FAIL signature.")
        md.append(
            "* `COO-9` remains the lead path; the EvalPlus-"
            "family attack continues, the question shifts to "
            "WHICH benchmark.")
        md.append(
            "* If the FAIL signature is similar to W102's "
            "(margin < 0 pp + MLB-2 < 33 %), the W89 mechanism "
            "cross-bench generalisation is structurally bounded "
            "at the base-HumanEval template at 70B.  W104 "
            "evaluates whether the mechanism distribution or "
            "the benchmark ceiling is the binding constraint.")
    md.append("")
    md.append("## Honest framing")
    md.append("")
    md.append(
        "* This is a 1-seed × 30-problem cheap pilot AT THE "
        "Phase 2 SIZE.  It is NOT retirement evidence — that "
        "requires W105+ Phase 3 multi-seed (3 seeds × 100 "
        "problems × K=5).")
    md.append(
        "* The W89 70B HumanEval K=5 multi-seed retirement "
        "remains the only confirmed multi-seed same-budget "
        "multi-agent superiority retirement in the programme.")
    md.append(
        "* The W102 arsenal-mining HumanEval+ re-grade prior "
        "predicted B − A1 ≈ +5.56 pp on W88 historical responses "
        "re-graded against HumanEval+ extra tests.  This pilot "
        "is the fresh-K=5 sampling ground truth.  Per the W102 "
        "anti-pattern carry-forward, the prior is RECORDED but "
        "is NOT a Phase 2 gate input.")
    md.append(
        "* The slice is helper-anchored (first NIM-spending "
        "pilot to consume `coordpy.code_slice_selector_v1` as a "
        "load-bearing input).  COO-14 transitions from "
        "\"shipped helper with worked example\" to \"shipped "
        "helper as load-bearing pilot input\".")
    md.append("")
    md.append("## Anchors")
    md.append("")
    md.append(
        f"* `{report_path.relative_to(ROOT)}` — bench report.")
    md.append(
        f"* `{report_path.parent.relative_to(ROOT)}"
        "/humaneval_plus_reflexion_calls.jsonl` — per-call "
        "sidecar.")
    md.append(
        f"* `{report_path.parent.relative_to(ROOT)}"
        "/provenance.json` — explicit provenance fields.")
    md.append("* `docs/RUNBOOK_W103.md` — pre-commit contract.")
    md.append(
        "* `docs/RESULTS_W103_HELPER_CONSUMPTION_V1.md` — slice "
        "attestation.")
    md.append(
        "* `docs/RESULTS_W103_HUMANEVAL_PLUS_PREFLIGHT_RECONFIRM_V1.md` "
        "— preflight re-confirmation.")
    md.append(
        "* `docs/RESULTS_W103_MILESTONE_SUMMARY_V1.md` — "
        "milestone summary.")
    md.append(
        "* `docs/FRONTIER_RELEVANCE_AUDIT_W103_V1.md` — "
        "frontier audit supplement.")
    md.append("")
    out_path = Path(args.out)
    out_path.write_text("\n".join(md))
    print(f"  wrote {out_path}")
    print(
        f"  verdict label: {verdict_label}; "
        f"Phase 2 = {n_phase2_passed}/9; "
        f"MLB-1 = {_label_pass(mlb1_passes)}; "
        f"MLB-2 = {_label_pass(mlb2_passes)}")
    print(
        f"  A0 = {a0:.2f}% / A1 = {a1:.2f}% / B = {b:.2f}% / "
        f"B-A1 = {margin:+.2f} pp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
