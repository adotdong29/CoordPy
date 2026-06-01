"""W124 Lane α — local transformer-native code-intervention mechanism bench.

Builds the matched-ICPC labeled code dataset from the REAL Maverick generations
already on disk (W120 resistant + W121 exposed + W122 paired seeds), reads REAL
distilgpt2 hidden state at AST boundaries, and runs the M4 separability probe vs
a surface baseline under problem-disjoint grouped CV. Emits the Lane-α verdict
JSON. NIM-free. No network in the default (offline-label) mode.

Run under the only torch+transformers env on this host:
    PYTHONPATH=. /Users/qdong/opt/anaconda3/bin/python \
        scripts/run_w124_lane_alpha_mechanism_bench_v1.py

Labels (default offline mode) are the OFFICIAL grader verdicts recorded by the
original pilots (``grade_on_secret_v1`` ran during W120-W122); we read them from
the per-problem bench reports. Per-arm call order is deterministic
(A0[temp0] + A1[5×temp0.7 base] + B[1 base + 4 reflexion]); see
``icpc_reflexion_bench_v1._run_{a0,a1,b}``.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from coordpy.icpc_reflexion_bench_v1 import extract_candidate_code_v1  # noqa: E402
from coordpy.transformer_native_code_intervention_v1 import (  # noqa: E402
    LabeledCodeRowV1, AstBoundaryHiddenEncoderV1, run_m4_separability_probe_v1,
    m4_gate_v1, run_m5_projector_if_earned_v1, M6ToolSubstrateCodeControllerV1,
    assemble_lane_alpha_verdict_v1,
)

# (calls.jsonl, bench_report.json, field) for the 6 full non-canary runs.
RUN_SPECS = [
    ("results/w120/icpc_pilot/w120_icpc_pilot_meta_llama-4-maverick-17b-128e-instruct_20260531T030343Z",
     "icpc_reflexion_calls.jsonl", "icpc_reflexion_bench_report.json", "resistant"),
    ("results/w121/exposed_pilot/w121_exposed_pilot_meta_llama-4-maverick-17b-128e-instruct_20260531T132231Z",
     "exposed_reflexion_calls.jsonl", "exposed_reflexion_bench_report.json", "exposed"),
    ("results/w122/paired_seed/w122_paired_seed_120002_20260531T155909Z",
     "resistant_reflexion_calls.jsonl", "resistant_reflexion_bench_report.json", "resistant"),
    ("results/w122/paired_seed/w122_paired_seed_120002_20260531T155909Z",
     "exposed_reflexion_calls.jsonl", "exposed_reflexion_bench_report.json", "exposed"),
    ("results/w122/paired_seed/w122_paired_seed_120003_20260531T184401Z",
     "resistant_reflexion_calls.jsonl", "resistant_reflexion_bench_report.json", "resistant"),
    ("results/w122/paired_seed/w122_paired_seed_120003_20260531T184401Z",
     "exposed_reflexion_calls.jsonl", "exposed_reflexion_bench_report.json", "exposed"),
]


def _load(spec):
    d, cj, rj, field = spec
    calls = [json.loads(l) for l in (ROOT / d / cj).read_text().splitlines() if l.strip()]
    report = json.loads((ROOT / d / rj).read_text())
    return calls, report, field, d


def build_offline_rows(calls, report, field, run_tag):
    """Deterministic offline labelling from the per-problem bench report.

    Per problem the gen-call order is fixed (no early stop):
      [0]=A0(temp0)  [1..5]=A1 base(temp0.7)  [6]=B k0 base  [7..10]=B reflexion k1..4
    Clean labels: A0 (a0_passed); failed whole arms (negatives); B reflexion
    first-pass round (positive) + earlier rounds (negative). Ambiguous samples
    (a base sample of a PASSING A1, or post-first-pass reflexion rounds) skipped.
    """
    seed_rep = report["per_seed"][0]
    qids = seed_rep["question_ids"]
    a0p = seed_rep["per_problem_a0_passed"]
    a1p = seed_rep["per_problem_a1_passed"]
    bp = seed_rep["per_problem_b_passed"]
    bfp = seed_rep["per_problem_b_first_pass_idx"]
    n = len(qids)
    if len(calls) != 11 * n:
        return []  # unexpected shape -> skip this run for offline mode
    rows = []
    for j in range(n):
        chunk = calls[11 * j: 11 * j + 11]
        pid = f"{field}:{qids[j]}"
        codes = [extract_candidate_code_v1(response_text=c.get("response_text", "")) for c in chunk]
        # A0
        rows.append(LabeledCodeRowV1(problem_id=pid, field=field, arm="A0",
                                     source=codes[0], passed=bool(a0p[j]),
                                     label_source="report_a0"))
        a1_pass = bool(a1p[j])
        b_pass = bool(bp[j])
        fp = int(bfp[j])
        # A1 base [1..5] + B k0 base [6]: clean negatives only if BOTH arms failed
        # on the base samples (A1 all-fail AND B did not pass at k0).
        if (not a1_pass) and (fp != 0):
            for idx in list(range(1, 6)) + [6]:
                rows.append(LabeledCodeRowV1(problem_id=pid, field=field, arm="A1",
                                             source=codes[idx], passed=False,
                                             label_source="report_failed_arm"))
        # B reflexion rounds k=1..4 -> chunk positions 7..10
        for pos, kk in zip(range(7, 11), range(1, 5)):
            if not b_pass or fp == -1:
                rows.append(LabeledCodeRowV1(problem_id=pid, field=field, arm="B",
                                             source=codes[pos], passed=False,
                                             label_source="report_failed_arm"))
            elif kk < fp:
                rows.append(LabeledCodeRowV1(problem_id=pid, field=field, arm="B",
                                             source=codes[pos], passed=False,
                                             label_source="report_b_pre_firstpass"))
            elif kk == fp:
                rows.append(LabeledCodeRowV1(problem_id=pid, field=field, arm="B",
                                             source=codes[pos], passed=True,
                                             label_source="report_b_firstpass"))
            # kk > fp : ambiguous (no early stop) -> skip
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="W124 Lane α mechanism bench")
    ap.add_argument("--model", default="distilbert/distilgpt2")
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "w124" / "lane_alpha"))
    ap.add_argument("--seed", type=int, default=124_000)
    args = ap.parse_args()

    print("== W124 Lane α — local transformer-native code-intervention bench ==")
    rows = []
    per_run = {}
    for spec in RUN_SPECS:
        calls, report, field, tag = _load(spec)
        r = build_offline_rows(calls, report, field, tag)
        per_run[tag + ":" + field] = {"n_rows": len(r),
                                       "n_pos": sum(x.passed for x in r)}
        rows.extend(r)
        print(f"  loaded {tag.split('/')[-1]} [{field}]: {len(r)} labelled rows "
              f"({sum(x.passed for x in r)} pos)")
    n_pos_all = sum(r.passed for r in rows)
    n_neg_all = len(rows) - n_pos_all
    # Keep ALL positives (scarce class); deterministically subsample negatives to
    # a generous cap per field so the probe stays well-powered but the CPU encode
    # is bounded. AUC is robust to class balance; manifest records the seed+counts.
    import random as _random
    NEG_PER_FIELD = 130
    rng = _random.Random(args.seed)
    kept = [r for r in rows if r.passed]
    for field in ("resistant", "exposed"):
        negs = [r for r in rows if (not r.passed) and r.field == field]
        rng.shuffle(negs)
        kept.extend(negs[:NEG_PER_FIELD])
    rng.shuffle(kept)
    subsample_info = {"neg_per_field_cap": NEG_PER_FIELD,
                      "n_before": len(rows), "n_pos_all": n_pos_all,
                      "n_neg_all": n_neg_all, "n_after": len(kept),
                      "subsample_seed": args.seed}
    rows = kept
    n_pos = sum(r.passed for r in rows)
    print(f"  TOTAL labelled: {subsample_info['n_before']} ({n_pos_all} pos / {n_neg_all} neg); "
          f"subsampled negatives -> {len(rows)} rows ({n_pos} pos), "
          f"{len(set(r.problem_id for r in rows))} problems")

    outdir0 = Path(args.out_dir)
    outdir0.mkdir(parents=True, exist_ok=True)
    cache_path = str(outdir0 / "w124_feature_cache.npz")
    print(f"  building encoder ({args.model}) — reading REAL hidden state "
          f"(cache: {cache_path}) ...")
    enc = AstBoundaryHiddenEncoderV1(model_name=args.model, require_real_model=True,
                                     cache_path=cache_path)

    def _prog(i, n):
        print(f"    encoded {i}/{n}", flush=True)

    print("  running M4 separability probe (problem-disjoint grouped CV) ...")
    m4 = run_m4_separability_probe_v1(rows, encoder=enc, seed=args.seed, progress=_prog)
    enc.flush()
    gate = m4_gate_v1(m4)
    print(f"  M4: AUC_hidden={m4.auc_hidden:.4f} AUC_surface={m4.auc_surface:.4f} "
          f"margin={gate['margin_over_surface']:.4f} -> {gate['verdict']}")
    print(f"      resistant AUC_h={m4.auc_hidden_resistant:.4f} "
          f"exposed AUC_h={m4.auc_hidden_exposed:.4f}")

    m5 = run_m5_projector_if_earned_v1(rows, encoder=enc, m4_gate=gate)
    ctrl = M6ToolSubstrateCodeControllerV1(k_budget=5)
    # M6 contract self-checks (deterministic; never reads hidden tests)
    demo = [
        ctrl.route(stderr_tail="SyntaxError: invalid syntax", timed_out=False, attempt_idx=0),
        ctrl.route(stderr_tail="", timed_out=True, attempt_idx=1),
        ctrl.route(stderr_tail="IndexError: list index out of range", timed_out=False, attempt_idx=2),
        ctrl.route(stderr_tail="", timed_out=False, attempt_idx=4),
    ]
    m6_contract = {
        "controller": "M6ToolSubstrateCodeControllerV1",
        "hosted_translatable": ctrl.is_hosted_translatable(),
        "local_gain_demonstrated": False,  # contract-only; no competent local generator
        "deterministic_routes": [d.route for d in demo],
        "route_reasons": [d.reason for d in demo],
        "materially_different_from_reflexion": True,
        "reads_hidden_test_source": False,
    }

    manifest = {
        "matched_core_slices": {"resistant_30slice_cid": "01bf9ef8",
                                "exposed_30slice_cid": "32d15db5"},
        "runs": per_run, "label_mode": "offline_official_grader_reports",
        "model": args.model, "n_rows": len(rows), "n_pos": n_pos,
        "subsample": subsample_info,
    }
    verdict = assemble_lane_alpha_verdict_v1(
        m4=m4, m4_gate=gate, m5=m5, m6_contract=m6_contract,
        dataset_manifest=manifest)
    verdict["generated_at"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
    verdict["per_run"] = per_run

    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / "w124_lane_alpha_verdict.json"
    out.write_text(json.dumps(verdict, indent=2, default=str))
    print(f"  wrote {out}")
    print(f"  HOSTED MAVERICK PROBE EARNED: {verdict['hosted_maverick_probe_earned']}")
    print("  >>> " + verdict["interpretation"][:200])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
