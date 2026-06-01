"""W124 Lane β — learned-memory / controller relevance probe (NIM-free).

Mandatory lane. Question: is there a learned-controller line in the existing
arsenal (``differentiable_memory_substrate_v1`` / ``composed_learned_memory_v1``)
that is ACTUALLY relevant to ICPC code repair / planning here, or are those
memory lines still too synthetic to matter for the current code-closure problem?

Honest controller decision the arsenal would have to win: given the trace of a
problem the self-consistency arm (A1) FAILED, decide whether to invest the
sequential-reflexion budget (B) — i.e. predict "reflexion will rescue this".
That is the only place a learned controller could add value over A1 here.

This probe builds the trace-decision dataset from the SAME 6 on-disk runs as
Lane α and applies the RUNBOOK_W124 §5 earn rule: a learned controller earns a
"real line" verdict only if it beats the base-rate baseline by >= +0.05 balanced
accuracy under problem-disjoint CV. Emits results/w124/lane_beta/.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# reuse the exact run set Lane α used
from scripts.run_w124_lane_alpha_mechanism_bench_v1 import RUN_SPECS, _load  # noqa: E402

BAL_ACC_MARGIN = 0.05  # RUNBOOK_W124 §5


def build_trace_decision_rows():
    """One row per (run, problem) the A1 arm FAILED. Features known BEFORE the
    reflexion decision; label = did B (reflexion) rescue it."""
    rows = []
    for spec in RUN_SPECS:
        calls, report, field, tag = _load(spec)
        s = report["per_seed"][0]
        qids = s["question_ids"]
        a0p, a1p, bp = s["per_problem_a0_passed"], s["per_problem_a1_passed"], s["per_problem_b_passed"]
        bfp = s["per_problem_b_first_pass_idx"]
        for j, qid in enumerate(qids):
            if a1p[j]:
                continue  # controller only matters when A1 (self-consistency) failed
            feats = [
                1.0 if a0p[j] else 0.0,                 # did the greedy single-shot pass
                float(field == "exposed"),              # field (contamination side)
                float(j) / max(1, len(qids)),           # position (difficulty proxy by slice order)
            ]
            label = 1 if (bp[j] and not a1p[j]) else 0   # reflexion rescued an A1 failure
            rows.append({"problem_id": f"{field}:{qid}", "field": field,
                         "feats": feats, "label": int(label), "run": tag})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="W124 Lane β controller relevance probe")
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "w124" / "lane_beta"))
    args = ap.parse_args()
    import numpy as np

    rows = build_trace_decision_rows()
    y = np.array([r["label"] for r in rows])
    X = np.array([r["feats"] for r in rows], dtype=float)
    groups = np.array([r["problem_id"] for r in rows])
    n, npos = len(y), int(y.sum())
    base_rate = float(y.mean()) if n else 0.0
    # base-rate (majority) balanced accuracy is 0.5 by construction; a useful
    # controller must beat that.
    majority_bal_acc = 0.5

    learned_bal_acc = float("nan")
    note = ""
    if n >= 12 and npos >= 3 and (n - npos) >= 3:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GroupKFold
        from sklearn.metrics import balanced_accuracy_score
        from sklearn.preprocessing import StandardScaler
        n_groups = len(set(groups.tolist()))
        sp = max(2, min(5, n_groups))
        oof = np.full(n, -1, dtype=int)
        gkf = GroupKFold(n_splits=sp)
        for tr, te in gkf.split(X, y, groups):
            if len(np.unique(y[tr])) < 2:
                continue
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(max_iter=2000, class_weight="balanced")
            clf.fit(sc.transform(X[tr]), y[tr])
            oof[te] = clf.predict(sc.transform(X[te]))
        m = oof >= 0
        if m.sum() >= 4 and len(np.unique(y[m])) >= 2:
            learned_bal_acc = float(balanced_accuracy_score(y[m], oof[m]))
        else:
            note = "CV folds collapsed (single-class) — underpowered."
    else:
        note = (f"underpowered: n={n} A1-failed decisions, {npos} rescue events "
                f"(need >=3 per class) — too few to fit/evaluate a controller.")

    earns = (not np.isnan(learned_bal_acc)) and (learned_bal_acc - majority_bal_acc) >= BAL_ACC_MARGIN
    verdict = {
        "schema": "coordpy.w124_lane_beta_controller_probe.v1",
        "lane": "beta_learned_memory_controller",
        "question": ("Is there a learned-controller line in differentiable_memory_substrate_v1 / "
                     "composed_learned_memory_v1 actually relevant to ICPC code repair here?"),
        "decision_problem": "predict reflexion-rescue of an A1-failed problem from pre-decision trace features",
        "n_a1_failed_decisions": n, "n_rescue_events": npos, "rescue_base_rate": base_rate,
        "majority_balanced_accuracy": majority_bal_acc,
        "learned_balanced_accuracy_cv": learned_bal_acc,
        "earn_margin_required": BAL_ACC_MARGIN,
        "controller_line_earned": bool(earns),
        "structural_assessment": (
            "differentiable_memory_substrate_v1 + composed_learned_memory_v1 are sequence-memory "
            "models trained on SYNTHETIC content-addressed-recall / long-horizon datasets "
            "(build_content_addressed_recall_dataset_v1 / build_composed_long_horizon_dataset_v1). "
            "The ICPC reflexion-rescue decision is a small tabular classification "
            f"(n={n} A1-failed problems, {npos} rescue events across 6 runs) — architecturally "
            "mismatched and data-starved; a heavyweight learned-memory controller is not warranted "
            "even before training it."),
        "verdict": ("LEARNED_CONTROLLER_LINE_EARNED" if earns
                    else "TOO_SYNTHETIC_NOT_WARRANTED"),
        "note": note,
        "nim_spend": 0,
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "w124_lane_beta_verdict.json").write_text(json.dumps(verdict, indent=2, default=str))
    print(json.dumps({k: verdict[k] for k in
                      ["n_a1_failed_decisions", "n_rescue_events", "rescue_base_rate",
                       "learned_balanced_accuracy_cv", "controller_line_earned", "verdict", "note"]},
                     indent=2, default=str))
    print(f"  wrote {outdir / 'w124_lane_beta_verdict.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
