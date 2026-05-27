#!/usr/bin/env python3
"""W104 — emit RESULTS_W104_HUMANEVAL_PLUS_PHASE2_405B_V1.md
+ RESULTS_W104_CROSS_SCALE_COMPARATOR_V1.md from the cross-
scale pilot bench report + cross-scale comparator report.

Reads the latest
`results/w104/humaneval_plus_cross_scale_pilot/<RUN>/
humaneval_plus_reflexion_bench_report.json` and
`cross_scale_comparator_report.json` and produces populated
Markdown matching the W104 RUNBOOK template (byte-equal slice
reuse + cross-generation/cross-scale target + 9 Phase 2 gates
+ MLB-1 + MLB-2 sub-gates + cross-scale comparator block +
branch decision logic).

Usage::

    python scripts/_w104_emit_pilot_result_doc.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _latest_pilot_dir() -> Path:
    latest_ptr = (
        ROOT / "results" / "w104"
        / "humaneval_plus_cross_scale_pilot"
        / "latest_run.txt")
    if not latest_ptr.exists():
        raise FileNotFoundError(
            f"pilot latest_run.txt not found at {latest_ptr}")
    pointer = latest_ptr.read_text().strip()
    return latest_ptr.parent / pointer


def _label_pass(b: bool) -> str:
    return "PASS" if b else "FAIL"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pilot-dir", default=None,
        help="Path to pilot output dir (defaults to latest)")
    ap.add_argument(
        "--verdict-out",
        default=str(
            ROOT / "docs"
            / "RESULTS_W104_HUMANEVAL_PLUS_PHASE2_405B_V1.md"))
    ap.add_argument(
        "--comparator-out",
        default=str(
            ROOT / "docs"
            / "RESULTS_W104_CROSS_SCALE_COMPARATOR_V1.md"))
    args = ap.parse_args()
    pilot_dir = (
        Path(args.pilot_dir)
        if args.pilot_dir else _latest_pilot_dir())
    report_path = (
        pilot_dir / "humaneval_plus_reflexion_bench_report.json")
    comp_path = pilot_dir / "cross_scale_comparator_report.json"
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
    target_used = str(prov.get("model_id", ""))
    primary = str(prov.get("primary_target", ""))
    backup = str(prov.get("backup_target", ""))
    used_backup = (target_used == backup)
    cross_scale_label = (
        "cross-generation (Llama-3.1 vs Llama-3.3 at 70B)"
        if used_backup
        else "cross-scale-UP (70B → 405B)")
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

    # Cross-scale comparator block
    comp_md_block = ""
    comp_rep = None
    if comp_path.exists():
        comp_rep = json.loads(comp_path.read_text())
    md: list[str] = []
    md.append(
        "# W104 — HumanEval+ cross-scale Phase 2 pilot V1")
    md.append("")
    md.append(
        f"> **2026-05-26.  Cross-scale HumanEval+ Phase 2 cheap "
        f"pilot verdict at `{target_used}` on the W103 helper-"
        "anchored 30-problem slice (BYTE-FOR-BYTE reuse; slice "
        f"CID `{prov.get('slice_cid_helper_priority', '')}`) "
        "× K=5 = 330 NIM calls.  Cross-scale form actually "
        f"achieved: {cross_scale_label}.**")
    md.append("")
    if used_backup:
        md.append(
            f"> **Reachability event**: pre-locked primary "
            f"target `{primary}` returned HTTP 404 on the "
            "reachability smoke probe; the pre-locked backup "
            f"`{backup}` was applied per the W104 RUNBOOK § "
            "Target-model selection rule.  The cross-scale "
            "shape achieved is therefore cross-generation at "
            "the same parameter scale (Llama 3.1 vs Llama 3.3 "
            "at 70B), NOT cross-scale-UP.  This is a weaker "
            "form of cross-scale than the primary target would "
            "have produced; the verdict reads against the W89 / "
            "W103 same-scale base-HumanEval template, NOT "
            "against the W96-A / W100 cross-scale-UP precedent.")
        md.append("")
    md.append("## Inputs (provenance)")
    md.append("")
    md.append("| Field | Value |")
    md.append("|---|---|")
    md.append(
        "| Candidate mechanism | B (W89 sequential reflexion "
        "on HumanEval+ at K=5) |")
    md.append(f"| Target model (used) | `{target_used}` |")
    md.append(
        f"| Pre-locked primary target | `{primary}` "
        f"({'unreachable (HTTP 404)' if used_backup else 'reachable; used'}) |")
    md.append(
        f"| Pre-locked backup target | `{backup}` "
        f"({'used' if used_backup else 'unused'}) |")
    md.append(
        f"| Cross-scale form achieved | {cross_scale_label} |")
    md.append(
        f"| HumanEval+ corpus SHA-256 | "
        f"`{prov.get('corpus_sha256', '')}` |")
    md.append(
        f"| Slice CID (helper-priority order; W103 reused) | "
        f"`{prov.get('slice_cid_helper_priority', '')}` |")
    md.append(
        f"| Slice CID (bench iteration order; W103 reused) | "
        f"`{prov.get('slice_cid_bench_order', '')}` |")
    md.append(
        f"| Preflight verdict cid (W102/W103 re-used) | "
        f"`{prov.get('preflight_verdict_cid', '')}` |")
    md.append(
        f"| Helper proposal CID (humaneval_plus; W103 reused) | "
        f"`{prov.get('helper_proposal_cid_humaneval_plus', '')}` |")
    md.append(
        f"| Mining report CID | "
        f"`{prov.get('mining_report_cid', '')}` |")
    md.append(
        f"| Bench Merkle root | `{bench_merkle}` |")
    md.append(
        f"| Pilot wall | {wall_s:.1f} s |")
    md.append(
        f"| Seed (candidate sampling RNG) | "
        f"{prov.get('seed', 104001)} |")
    md.append(f"| n_problems | {n_problems} |")
    md.append(
        f"| Target-selection-rule version | "
        f"`{prov.get('target_selection_rule_version', '')}` |")
    md.append(
        "| Arsenal-mining prior (RECORDED; NOT a Phase 2 gate "
        "input) | "
        f"B−A1 = +{float(prov.get('arsenal_mining_prior_humaneval_plus', {}).get('b_minus_a1_pp', 0)):.2f} pp; "
        f"rescue = {float(prov.get('arsenal_mining_prior_humaneval_plus', {}).get('rescue_fraction', 0))*100:.2f} % "
        "(W102 cross-bench extension) |")
    md.append(
        "| W103 70B empirical anchor (RECORDED; NOT a Phase 2 "
        "gate input) | "
        f"B−A1 = +{float(prov.get('w103_70b_empirical_anchor', {}).get('b_minus_a1_pp', 0)):.2f} pp; "
        f"MLB-2 = {float(prov.get('w103_70b_empirical_anchor', {}).get('mlb2_rescue_rate', 0))*100:.2f} % |")
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
    md.append(
        f"| shared_fails (hard cluster) | {n_shared_fails} |")
    md.append("")
    md.append("## Phase 2 gates")
    md.append("")
    md.append("| # | Gate | Verdict |")
    md.append("|---|---|---|")
    for idx, (key, label) in enumerate([
        ("G1_slice_pre_committed",
         "Slice pre-committed (W103 byte-equal reuse; 30 "
         "problems; CID locked BEFORE NIM call)"),
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
    if comp_rep is not None:
        md.append("## Cross-scale comparator block (vs W103 70B)")
        md.append("")
        md.append(
            f"| Field | Scale A (W103 70B) | "
            f"Scale B (W104 {target_used}) |")
        md.append("|---|---|---|")
        md.append(
            f"| Model | `{comp_rep['scale_a_model_id']}` | "
            f"`{comp_rep['scale_b_model_id']}` |")
        md.append(
            f"| Bench Merkle | "
            f"`{comp_rep['scale_a_bench_merkle'][:16]}...` | "
            f"`{comp_rep['scale_b_bench_merkle'][:16]}...` |")
        md.append(
            f"| MLB-1 invocation | "
            f"{comp_rep['aggregate_mlb1_invocation_rate_at_scale_a']*100:.2f}% | "
            f"{comp_rep['aggregate_mlb1_invocation_rate_at_scale_b']*100:.2f}% |")
        md.append(
            f"| MLB-2 rescue | "
            f"{comp_rep['aggregate_mlb2_rescue_rate_at_scale_a']*100:.2f}% | "
            f"{comp_rep['aggregate_mlb2_rescue_rate_at_scale_b']*100:.2f}% |")
        md.append(
            f"| B − A1 (pp) | "
            f"{comp_rep['aggregate_b_minus_a1_pp_at_scale_a']:+.2f} | "
            f"{comp_rep['aggregate_b_minus_a1_pp_at_scale_b']:+.2f} |")
        md.append("")
        md.append(
            f"* **Cross-scale shift on B − A1**: "
            f"{comp_rep['cross_scale_shift_on_b_minus_a1_pp']:+.2f} pp")
        md.append(
            f"* **Cross-scale shift on MLB-2**: "
            f"{comp_rep['cross_scale_shift_on_mlb2_pp']:+.2f} pp")
        md.append("")
        md.append("### Cluster-shift aggregate")
        md.append("")
        md.append("| Shift | Count |")
        md.append("|---|---:|")
        for k in ("stayed", "improved", "regressed", "flipped"):
            md.append(
                f"| `{k}` | "
                f"{comp_rep['aggregate_cluster_shift_counts'].get(k, 0)} |")
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
        "### Decision applied per the pre-committed W104 "
        "RUNBOOK")
    md.append("")
    if verdict_label == "PASS_MECHANISM_DRIVEN":
        md.append(
            "* Cheap pilot PASSes 9/9 + MLB sub-gates clearing "
            "→ **Branch A** of the W104 RUNBOOK § Planning lane.")
        md.append(
            f"* Carry-forward registered: "
            "`W104-L-HUMANEVAL-PLUS-REFLEXION-PHASE2-405B-PASS` "
            f"(single-seed cross-scale cheap-pilot PASS at "
            f"`{target_used}`; NOT a multi-scale retirement).")
        md.append(
            "* **W105 = HumanEval+ Phase 3 retirement bench** "
            "(3 seeds × 100 problems × K=5 × 2 scales = 6 600 "
            "NIM calls) is ENTITLED.  Slice pack pre-built in "
            "`data/w105/phase3_slice_pack/<RUN>/slice_pack.json` "
            "(pack CID `8be55f3bf1650df3...`).")
        md.append(
            "* `COO-9` remains the lead path.  Programme "
            f"entitlement: the W89 sequential-reflexion mechanism "
            f"extends to HumanEval+ across TWO model classes "
            f"({cross_scale_label}) at Phase 2 cheap-pilot "
            "quality.  Retirement-grade generalisation still "
            "requires W105 Phase 3 multi-seed.")
    elif verdict_label == "PASS_NON_MECHANISM_DRIVEN":
        md.append(
            "* Cheap pilot PASSes 9/9 BUT MLB-2 FAILs → "
            "`PASS_NON_MECHANISM_DRIVEN`; **Branch B** of the "
            "W104 RUNBOOK § Planning lane.")
        md.append(
            "* **W105 = HumanEval+ mechanism-variation slate** "
            f"at `{target_used}` (B1 = enforced-reflexion-on-"
            "attempt-0; B2 = test-aware decomposition reader+"
            "solver on the EvalPlus extra-test surface; B3 = "
            "sidecar-driven failure-cluster targeting at "
            "the cross-scale target).  Cheap-pilot earning "
            "rule: at least one B-variant must lift MLB-2 ≥ "
            "33 % AND keep margin ≥ +5 pp.")
        md.append(
            "* Carry-forward added: "
            f"`W104-L-HUMANEVAL-PLUS-MECHANISM-LOAD-BEARINGNESS-"
            f"WEAK-AT-405B-CAP` "
            f"(margin {margin:+.2f} pp PASS; MLB-2 "
            f"{mlb2_rate:.2f} % FAIL).")
        md.append(
            "* Phase 3 NOT entitled.  Cross-scale W105 path is "
            "mechanism variation, not retirement.")
        md.append(
            "* `COO-9` remains the lead path; W105 attacks "
            "mechanism load-bearingness directly at the "
            "cross-scale target.")
    else:  # FAIL
        md.append(
            "* Cheap pilot FAILs → **Branch C** of the W104 "
            "RUNBOOK § Planning lane.")
        md.append(
            f"* Carry-forward added: "
            "`W104-L-HUMANEVAL-PLUS-REFLEXION-PHASE2-405B-CAP` "
            f"(B − A1 = {margin:+.2f} pp; "
            f"MLB-1 = {mlb1_rate:.2f} % "
            f"{_label_pass(mlb1_passes)}; "
            f"MLB-2 = {mlb2_rate:.2f} % "
            f"{_label_pass(mlb2_passes)}; "
            f"A1 = {a1:.2f} %).")
        md.append(
            "* The Branch C triage table in "
            "`docs/RESULTS_W104_HELPER_W105_PLANNING_V1.md` "
            "applies; the dispatch decision based on the FAIL "
            "signature is recorded in "
            "`docs/RESULTS_W104_MILESTONE_SUMMARY_V1.md`.")
        md.append(
            "* `COO-9` remains the lead path unless the FAIL "
            "signature triggers a structural pivot to "
            "LiveCodeBench / APPS via the Branch C dispatch.")
        md.append(
            "* The W103 70B PASS is the only confirmed code-"
            "benchmark cross-bench Phase 2 anchor; W104 caps "
            "the W103 mechanism at single-scale at 70B at the "
            "cheap-pilot size.")
    md.append("")
    md.append("## Honest framing")
    md.append("")
    if used_backup:
        md.append(
            f"* **Backup-target reality**: pre-locked primary "
            f"`{primary}` was unreachable (HTTP 404 on NIM); "
            f"the pre-locked backup `{backup}` was used.  The "
            "cross-scale form actually achieved is "
            f"{cross_scale_label} — WEAKER than the primary "
            "target's cross-scale-UP form would have been.")
        md.append(
            "* Honest framing: this is NOT a 70B → 405B cross-"
            "scale test.  It is a 70B-Llama-3.3 → 70B-Llama-3.1 "
            "cross-generation test at the same parameter "
            "scale.")
    md.append(
        "* This is a 1-seed × 30-problem cheap pilot AT THE "
        "Phase 2 SIZE.  It is NOT retirement evidence — "
        "retirement requires W105 Phase 3 multi-seed (3 seeds "
        "× 100 problems × K=5 × 2 scales).")
    md.append(
        "* The W89 70B HumanEval K=5 multi-seed retirement "
        "remains the only confirmed multi-seed same-budget "
        "multi-agent superiority retirement in the programme.")
    md.append(
        "* The slice is BYTE-EQUAL to W103 (same 30 task_ids "
        "in the same bench-iteration order).  Cross-scale "
        "comparator catches mix-ups at write time.")
    md.append(
        "* The W104 arsenal-mining prior + W103 70B empirical "
        "anchor are RECORDED in provenance but are NOT Phase "
        "2 gate inputs.  Per the W102 anti-pattern carry-"
        "forward, cross-bench / cross-scale priors are UPPER "
        "BOUNDS only; fresh-K=5 sampling at the cross-scale "
        "target is the ground truth.")
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
    if comp_path.exists():
        md.append(
            f"* `{comp_path.relative_to(ROOT)}` — cross-scale "
            "comparator JSON.")
        md.append(
            f"* `{comp_path.parent.relative_to(ROOT)}"
            "/cross_scale_comparator_report.md` — comparator "
            "markdown.")
    md.append("* `docs/RUNBOOK_W104.md` — pre-commit contract.")
    md.append(
        "* `docs/RESULTS_W104_CROSS_SCALE_COMPARATOR_V1.md` — "
        "cross-scale comparator narrative.")
    md.append(
        "* `docs/RESULTS_W104_HELPER_W105_PLANNING_V1.md` — "
        "W105 Phase 3 slice pack + Branch C dispatch.")
    md.append(
        "* `docs/RESULTS_W104_MILESTONE_SUMMARY_V1.md` — "
        "milestone summary.")
    md.append(
        "* `docs/FRONTIER_RELEVANCE_AUDIT_W104_V1.md` — "
        "frontier audit supplement.")
    md.append("")
    out_path = Path(args.verdict_out)
    out_path.write_text("\n".join(md))
    print(f"  wrote {out_path}")

    # Also emit the standalone cross-scale comparator narrative
    if comp_rep is not None:
        cm: list[str] = []
        cm.append(
            "# W104 — Cross-scale comparator V1")
        cm.append("")
        cm.append(
            f"> **2026-05-26.  Cross-scale comparison of the "
            f"W103 70B HumanEval+ cheap-pilot result vs the "
            f"W104 cross-scale pilot at `{target_used}` on the "
            "BYTE-EQUAL 30-problem helper-anchored slice.**")
        cm.append("")
        cm.append(
            f"* Cross-scale form achieved: {cross_scale_label}.")
        cm.append(
            f"* Slice CID (bench-iteration order; identical at "
            f"both scales): `{comp_rep['slice_cid']}`.")
        cm.append(
            f"* HumanEval+ corpus SHA-256 (identical at both "
            f"scales): `{comp_rep['corpus_sha256']}`.")
        cm.append(
            f"* Comparator schema: `{comp_rep['schema']}`.")
        cm.append("")
        cm.append("## Aggregate cross-scale numbers")
        cm.append("")
        cm.append(
            f"| Field | Scale A (W103 70B Llama-3.3) | "
            f"Scale B (W104 {target_used}) |")
        cm.append("|---|---|---|")
        cm.append(
            f"| Model | `{comp_rep['scale_a_model_id']}` | "
            f"`{comp_rep['scale_b_model_id']}` |")
        cm.append(
            f"| Bench Merkle | "
            f"`{comp_rep['scale_a_bench_merkle']}` | "
            f"`{comp_rep['scale_b_bench_merkle']}` |")
        cm.append(
            f"| A0 mean pass-rate | "
            f"{(comp_rep['aggregate_b_minus_a1_pp_at_scale_a'] + comp_rep.get('a1_pp_a', 0)):+.2f} pp (see below) | "
            "see verdict doc |")
        cm.append(
            f"| MLB-1 invocation | "
            f"{comp_rep['aggregate_mlb1_invocation_rate_at_scale_a']*100:.2f}% | "
            f"{comp_rep['aggregate_mlb1_invocation_rate_at_scale_b']*100:.2f}% |")
        cm.append(
            f"| MLB-2 rescue | "
            f"{comp_rep['aggregate_mlb2_rescue_rate_at_scale_a']*100:.2f}% | "
            f"{comp_rep['aggregate_mlb2_rescue_rate_at_scale_b']*100:.2f}% |")
        cm.append(
            f"| B − A1 (pp) | "
            f"{comp_rep['aggregate_b_minus_a1_pp_at_scale_a']:+.2f} | "
            f"{comp_rep['aggregate_b_minus_a1_pp_at_scale_b']:+.2f} |")
        cm.append("")
        cm.append(
            f"**Cross-scale shift on B − A1**: "
            f"{comp_rep['cross_scale_shift_on_b_minus_a1_pp']:+.2f} pp")
        cm.append(
            f"**Cross-scale shift on MLB-2 rescue rate**: "
            f"{comp_rep['cross_scale_shift_on_mlb2_pp']:+.2f} pp")
        cm.append("")
        cm.append("## Aggregate arm deltas (B − A)")
        cm.append("")
        for arm, key in (("A0", "a0_pp"), ("A1", "a1_pp"),
                          ("B", "b_pp")):
            v = comp_rep['aggregate_arm_deltas_pp'].get(key, 0)
            cm.append(f"* {arm}: {v:+.2f} pp")
        cm.append("")
        cm.append("## Cluster-shift aggregate")
        cm.append("")
        cm.append("| Shift | Count |")
        cm.append("|---|---:|")
        for k in ("stayed", "improved", "regressed", "flipped"):
            cm.append(
                f"| `{k}` | "
                f"{comp_rep['aggregate_cluster_shift_counts'].get(k, 0)} |")
        cm.append("")
        cm.append("## Per-problem rows")
        cm.append("")
        cm.append(
            "| idx | task_id | A0 A→B | A1 A→B | B A→B | "
            "bidx A→B | shift |")
        cm.append("|---|---|---|---|---|---|---|")
        for i, row in enumerate(comp_rep["per_problem"]):
            cm.append(
                f"| {i+1} | {row['task_id']} | "
                f"{int(row['a0_at_scale_a'])}→{int(row['a0_at_scale_b'])} | "
                f"{int(row['a1_at_scale_a'])}→{int(row['a1_at_scale_b'])} | "
                f"{int(row['b_at_scale_a'])}→{int(row['b_at_scale_b'])} | "
                f"{int(row['b_first_pass_idx_at_scale_a'])}→"
                f"{int(row['b_first_pass_idx_at_scale_b'])} | "
                f"`{row['cluster_shift']}` |")
        cm.append("")
        cm.append("## Honest framing")
        cm.append("")
        if used_backup:
            cm.append(
                f"* The cross-scale form achieved is "
                f"{cross_scale_label}.  Pre-locked primary "
                f"`{primary}` was unreachable; pre-locked "
                f"backup `{backup}` was applied per the W104 "
                "RUNBOOK § Target-model selection rule.")
            cm.append(
                "* This comparator output reflects "
                "cross-GENERATION at the same parameter scale, "
                "NOT cross-scale-UP.  Future W104.x or W105 "
                "milestones must use the proper cross-scale-UP "
                "target if 405B becomes reachable on NIM.")
        cm.append(
            "* The slice + corpus + schema are byte-equal at "
            "both scales (the comparator REFUSES to run "
            "otherwise per the W104 hardening lane).  The "
            "per-problem rows are apples-to-apples.")
        cm.append("")
        cm.append("## Anchors")
        cm.append("")
        cm.append(
            f"* `{comp_path.relative_to(ROOT)}` — comparator "
            "JSON.")
        cm.append(
            f"* `{comp_path.parent.relative_to(ROOT)}"
            "/cross_scale_comparator_report.md` — comparator "
            "markdown emitted by the driver.")
        cm.append(
            "* `docs/RESULTS_W104_HUMANEVAL_PLUS_PHASE2_405B_"
            "V1.md` — W104 verdict.")
        cm.append(
            "* `docs/RUNBOOK_W104.md` — pre-commit contract.")
        cm_path = Path(args.comparator_out)
        cm_path.write_text("\n".join(cm))
        print(f"  wrote {cm_path}")

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
