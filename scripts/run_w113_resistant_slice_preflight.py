#!/usr/bin/env python3
"""W113-α — RESISTANT-for-Llama-4 LiveCodeBench slice preflight (NIM-free).

W112 reopened a +10.00pp Maverick reflexion margin on BigCodeBench, but
BigCodeBench 2024-06 is contamination-EXPOSED for Maverick (Aug-2024 cutoff >
release).  W113 builds the CLEAN instrument: a date-filtered LiveCodeBench slice
verifiably RESISTANT for Llama-4-Maverick (every problem dated STRICTLY after
August 2024).

This preflight is the EARNING artifact for the W113 Maverick cheap pilot.  It
runs ALL probes against the REAL SHA-pinned ``release_v6`` corpus with ZERO NIM
calls:

* P1 — corpus integrity: load via the SHA-pinned loader (refuses on missing
  cache / SHA mismatch / schema mismatch).  [reuses the W108 loader]
* P2 — executor_V2 self-test (synthetic gold/wrong/loop + REAL gold zigzag).
  [reuses ``run_w108_livecodebench_preflight._run_executor_v2_self_test``]
* P3 — loader real-data self-test (all func_names resolved, plain-arg mix,
  difficulty mix).  [reuses ``..._run_loader_realdata_self_test``]
* P5 — RESISTANT-PARTITION integrity (the W113 load-bearing check):
  partition the functional subset by the Llama-4-Maverick cutoff boundary
  (2024-08-31 -> resistant iff date >= 2024-09-01); report the EXCLUSION
  breakdown (missing / unparseable / not-after-cutoff); require the resistant
  subset >= MIN_RESISTANT_SLICE (30).  Prints the resistant date range so the
  "strictly after August 2024" claim is auditable.
* P6 — RESISTANT-SLICE selection: select the deterministic 30-slice FROM the
  resistant subset, pin its CID, and ASSERT it equals the W108 resistant slice
  CID (``2afc318c…``).  Equivalence proves the date filter did not perturb the
  problem set, so the Maverick pilot runs on the EXACT problems 70B ran in W108
  (clean cross-scale: model scale is the ONLY variable).

Lane-β readiness: also reports tier-2 applicability of the resistant slice
(``tier2_readiness_v1``) so the no-spend / spend rule is auditable here.

``overall_pass`` is True iff P1∧P2∧P3∧P5∧P6 hold — and ONLY THEN is the W113
Maverick cheap pilot earned (RUNBOOK_W113 § 4).

Usage::

    python scripts/run_w113_resistant_slice_preflight.py
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import sys  # noqa: E402
sys.path.insert(0, str(ROOT))

from coordpy.livecodebench_loader_v1 import (  # noqa: E402
    LiveCodeBenchCorpusError,
    load_livecodebench_functional_v1,
)
from coordpy.livecodebench_reflexion_bench_v1 import (  # noqa: E402
    select_livecodebench_functional_slice_v1,
)
from coordpy.livecodebench_resistant_slice_v1 import (  # noqa: E402
    MIN_RESISTANT_SLICE,
    cutoff_boundary_for_model_v1,
    resistant_partition_for_model_v1,
)
from coordpy.cross_scale_resistant_interpretation_v1 import (  # noqa: E402
    W108_RESISTANT_LCB_SLICE_CID,
    interpret_cross_scale_resistant_result_v1,
)
from coordpy.tier2_readiness_v1 import (  # noqa: E402
    assess_tier2_applicability_v1,
    decide_tier2_spend_v1,
)
# Reuse the W108 preflight probes verbatim (namespace import; no duplication).
from scripts.run_w108_livecodebench_preflight import (  # noqa: E402
    _run_executor_v2_self_test,
    _run_loader_realdata_self_test,
)

OUT_ROOT = ROOT / "results" / "w113" / "resistant_slice_preflight"

# Same operator-fetched + W108-verified pin (real release_v6 test6.jsonl).
W113_LIVECODEBENCH_RELEASE = "release_v6"
W113_LIVECODEBENCH_CACHE_PATH = os.path.expanduser(
    "~/.cache/coordpy/livecodebench-test6.jsonl")
W113_LIVECODEBENCH_RELEASE_V6_SHA256 = (
    "bb4c364f71921c4495a6ad15abe1a927350b720009f4933e2e71f8af0f6fd1f5")
W113_LIVECODEBENCH_DATASET_COMMIT = "0fe84c39"

# The W113 main-lane target.
W113_TARGET_MODEL = "meta/llama-4-maverick-17b-128e-instruct"

# A0/A1/B contract — byte-identical mechanism to W89/W103/W105/W108/W110/W112.
A0A1B_SPEC = {
    "A0": "stock single-shot at T=0.0 (1 model call / problem)",
    "A1": "first-pass-among-K=5 self-consistency at T=0.7 (5 calls)",
    "B": ("sequential-reflexion-K=5 at T=0.7, each turn conditioned on the "
          "cumulative (candidate, executor stderr) history (5 calls)"),
    "budget": "A1 and B are byte-exact K=5; no early-stop; same model on all arms",
    "executor_truth": ("functional public-test block; subprocess exit 0 iff "
                       "every case matches; NO LLM-as-judge"),
}

PHASE2_GATES = {
    "G1_slice_pre_committed": "slice CID pinned before the run",
    "G2_a1_lt_90pct": "A1@K=5 < 90% (non-saturated; real headroom)",
    "G3_b_gt_a1": "B > A1 (strict)",
    "G4_margin_geq_5pp": "(B - A1) >= +5.0 pp",
    "G5_b_gt_a0_by_geq_5pp": "(B - A0) >= +5.0 pp",
    "G6_per_problem_majority": "B did not regress vs A1 on >= 16 of 30",
    "G7_budget_exact": "A1/B budget byte-exact",
    "G8_audit_chain_re_derives": "per-call CIDs + per-seed/bench Merkle re-derive",
    "G9_executor_clean": "no-LLM-judge subprocess executor",
    "MLB1_invocation_rate_geq_33pct": "reflexion invoked on >= 33% of problems",
    "MLB2_rescue_rate_geq_33pct": "of invocations, >= 33% rescued by reflexion",
}

# § 1α bar: a CLEAN resistant reopening requires PASS_MECHANISM_DRIVEN; a
# margin-without-mechanism (PASS_NON_MECHANISM_DRIVEN) is NOT enough.
CLEAN_REOPENING_BAR = "verdict_label == PASS_MECHANISM_DRIVEN"


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _cid(payload: object) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _run_resistant_partition_self_test(subset, *, model_id: str) -> dict:
    """P5 — resistant-partition integrity for the target model."""
    cutoff = cutoff_boundary_for_model_v1(model_id)
    part = resistant_partition_for_model_v1(subset, model_id=model_id)
    checks = {
        "cutoff_is_known_grade": bool(cutoff.is_resistant_grade()),
        "resistant_subset_geq_min": bool(part.n_resistant >= MIN_RESISTANT_SLICE),
        "no_unparseable_dates": bool(
            len(part.excluded_unparseable_date) == 0),
        "no_missing_dates": bool(len(part.excluded_missing_date) == 0),
        "resistant_min_strictly_after_boundary": bool(
            part.resistant_date_min > cutoff.boundary_date
            if part.resistant_date_min else False),
    }
    return {
        "checks": checks,
        "all_pass": all(checks.values()),
        "model_id": str(model_id),
        "cutoff_boundary": cutoff.boundary_date,
        "cutoff_confidence": cutoff.confidence,
        "partition": part.to_dict(),
        "partition_cid": part.partition_cid(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="W113-α RESISTANT-for-Llama-4 LiveCodeBench preflight")
    ap.add_argument("--release", default=W113_LIVECODEBENCH_RELEASE)
    ap.add_argument("--cache-path", default=W113_LIVECODEBENCH_CACHE_PATH)
    ap.add_argument(
        "--expected-sha256", default=W113_LIVECODEBENCH_RELEASE_V6_SHA256)
    ap.add_argument("--model", default=W113_TARGET_MODEL)
    ap.add_argument("--n-problems", type=int, default=30)
    ap.add_argument("--out-root", default=str(OUT_ROOT))
    args = ap.parse_args()

    # P1 — corpus integrity via the SHA-pinned loader (refuses on mismatch).
    try:
        subset = load_livecodebench_functional_v1(
            release=str(args.release), cache_path=str(args.cache_path),
            expected_sha256=str(args.expected_sha256))
        p1 = {"loaded": True, "n_functional": len(subset),
              "release": str(args.release),
              "sha256_pin": str(args.expected_sha256),
              "dataset_commit": W113_LIVECODEBENCH_DATASET_COMMIT,
              "all_pass": bool(len(subset) > 0)}
    except LiveCodeBenchCorpusError as e:
        verdict = {
            "schema": "coordpy.w113_resistant_slice_preflight.v1",
            "milestone": "W113-alpha",
            "P1_corpus_integrity": {"loaded": False, "error": str(e),
                                    "all_pass": False},
            "overall_pass": False,
            "ts_utc": _dt.datetime.now(_dt.timezone.utc).isoformat()}
        out_dir = Path(args.out_root); out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "preflight_verdict.json").write_text(
            json.dumps(verdict, indent=2, default=str))
        print(f"  P1 corpus integrity: FAIL ({e})")
        return 1

    p2 = _run_executor_v2_self_test(subset)
    p3 = _run_loader_realdata_self_test(subset)
    p5 = _run_resistant_partition_self_test(subset, model_id=str(args.model))

    # P6 — resistant-slice selection + equivalence to the W108 slice.
    # Build the resistant subset (preserve loader order), then select the
    # deterministic 30-slice from it.
    resistant_ids = set(p5["partition"]["resistant_question_ids"])
    resistant_subset = tuple(
        p for p in subset if p.question_id in resistant_ids)
    pilot_slice = select_livecodebench_functional_slice_v1(
        resistant_subset, n_problems=int(args.n_problems))
    slice_qids = [p.question_id for p in pilot_slice]
    slice_cid = _cid({"kind": "w108_lcb_pilot_slice_v1",
                      "question_ids": slice_qids})
    slice_diffs = Counter(str(p.difficulty) for p in pilot_slice)
    slice_dates = sorted(str(p.contest_date)[:10] for p in pilot_slice)
    equals_w108 = bool(slice_cid == W108_RESISTANT_LCB_SLICE_CID)
    p6 = {
        "n_problems": len(pilot_slice),
        "resistant_slice_cid": slice_cid,
        "equals_w108_slice_cid": equals_w108,
        "w108_slice_cid": W108_RESISTANT_LCB_SLICE_CID,
        "question_ids": slice_qids,
        "difficulty_mix": dict(slice_diffs),
        "contest_date_min": slice_dates[0] if slice_dates else "",
        "contest_date_max": slice_dates[-1] if slice_dates else "",
        "selection_rule": (
            "difficulty-stratified largest-remainder over the RESISTANT subset, "
            "(contest_date, question_id)-ordered, OUTCOME-BLIND"),
        "a0_a1_b_spec": A0A1B_SPEC,
        "phase2_gates": PHASE2_GATES,
        "clean_reopening_bar": CLEAN_REOPENING_BAR,
        "all_pass": bool(
            len(pilot_slice) == int(args.n_problems) and equals_w108),
    }

    # Lane-β readiness: tier-2 applicability of THIS resistant slice.
    tier2_applic = assess_tier2_applicability_v1(
        slice_date_min=p6["contest_date_min"])
    tier2_preview = {
        "applicability": [a.to_dict() for a in tier2_applic],
        "spend_if_reopens": decide_tier2_spend_v1(
            main_lane_outcome="RESISTANT_SUPERIORITY_REOPENS",
            slice_date_min=p6["contest_date_min"]).to_dict(),
        "spend_if_exposure": decide_tier2_spend_v1(
            main_lane_outcome="EXPOSURE_CONFIRMED",
            slice_date_min=p6["contest_date_min"]).to_dict(),
    }

    # Pre-commit the W113 interpretation branches (purely illustrative here;
    # the live branch is selected by the pilot's actual verdict_label).
    interp_precommit = {
        label: interpret_cross_scale_resistant_result_v1(
            model_id=str(args.model),
            resistant_benchmark="LiveCodeBench release_v6 (resistant-for-Llama4)",
            verdict_label=label, b_minus_a1_pp=0.0, mlb2_rescue_rate=0.0,
        ).to_dict()["outcome"]
        for label in ("PASS_MECHANISM_DRIVEN", "PASS_NON_MECHANISM_DRIVEN",
                      "FAIL")
    }

    overall = bool(
        p1["all_pass"] and p2["all_pass"] and p3["all_pass"]
        and p5["all_pass"] and p6["all_pass"])

    verdict = {
        "schema": "coordpy.w113_resistant_slice_preflight.v1",
        "milestone": "W113-alpha",
        "target_model": str(args.model),
        "resistant_for": "meta/llama-4-maverick-17b-128e-instruct (Aug-2024 cutoff)",
        "P1_corpus_integrity": p1,
        "P2_executor_v2_self_test": p2,
        "P3_loader_realdata_self_test": p3,
        "P5_resistant_partition": p5,
        "P6_resistant_slice": p6,
        "tier2_readiness_preview": tier2_preview,
        "interpretation_branches_precommit": interp_precommit,
        "overall_pass": overall,
        "pilot_earned": overall,
        "ts_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    verdict["verdict_cid"] = _cid(
        {k: v for k, v in verdict.items() if k != "ts_utc"})

    out_dir = Path(args.out_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "preflight_verdict.json").write_text(
        json.dumps(verdict, indent=2, default=str))

    part = p5["partition"]
    print("=== W113-α RESISTANT-for-Llama-4 LiveCodeBench preflight ===")
    print(f"  P1 corpus integrity  : {'PASS' if p1['all_pass'] else 'FAIL'} "
          f"({p1.get('n_functional')} functional; SHA pinned)")
    print(f"  P2 executor_V2 test  : {'PASS' if p2['all_pass'] else 'FAIL'} "
          f"{p2['checks']}")
    print(f"  P3 loader real-data  : {'PASS' if p3['all_pass'] else 'FAIL'} "
          f"(functional={p3['n_functional']}; "
          f"dates {p3['contest_date_min']}..{p3['contest_date_max']})")
    print(f"  P5 resistant-partn   : {'PASS' if p5['all_pass'] else 'FAIL'} "
          f"(model={p5['model_id']}; boundary={p5['cutoff_boundary']} "
          f"[{p5['cutoff_confidence']}]; resistant={part['n_resistant']}/"
          f"{part['n_total']}; dates {part['resistant_date_min']}.."
          f"{part['resistant_date_max']}; excluded "
          f"miss={part['n_excluded_missing_date']} "
          f"unparse={part['n_excluded_unparseable_date']} "
          f"not-after={part['n_excluded_not_after_cutoff']})")
    print(f"  P6 resistant-slice   : {'PASS' if p6['all_pass'] else 'FAIL'} "
          f"(n={p6['n_problems']}; diff={p6['difficulty_mix']}; "
          f"dates {p6['contest_date_min']}..{p6['contest_date_max']})")
    print(f"     resistant_slice_cid: {p6['resistant_slice_cid']}")
    print(f"     == W108 slice CID  : {p6['equals_w108_slice_cid']} "
          f"(clean cross-scale: only model scale varies vs W108)")
    print("  Lane-β tier-2 applicability (this slice):")
    for a in tier2_applic:
        print(f"     {a.model_id:42s} cutoff={a.cutoff_boundary}"
              f"[{a.cutoff_confidence}] resistant={a.slice_certifiably_resistant}"
              f" — {a.applicability_reason[:60]}")
    print(f"  OVERALL              : "
          f"{'PASS — Maverick pilot EARNED' if overall else 'FAIL — NOT earned'}")
    print(f"  verdict_cid          : {verdict['verdict_cid']}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
