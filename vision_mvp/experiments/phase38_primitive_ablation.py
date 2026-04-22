"""Phase 38 Part B — minimum dynamic primitive ablation.

Runs feature-ablated threads on the Phase-35 contested bank
and the Phase-37 nested bank, plus the Phase-36 bounded
adaptive-subscription primitive as a sister point in the
design space. Reports per-feature collapse rates, so the
reader can tell which features are load-bearing on which
task family.

The purpose is to empirically evaluate Phase-37 Conjecture
C37-4's five-feature candidate set:

  1. bounded typed reply-kind enum;
  2. bounded witness-token cap;
  3. terminating resolution rule;
  4. round-aware reply state;
  5. bounded-context invariant (proxied by the ``bounded_
     witness`` flag — on deterministic tasks the Phase-35
     P35-2 invariant is equivalent to the cap).

The ablation table pivots on removing ONE feature at a time
(``only_missing(f)``) plus two composite controls (full, no).

Reproducible commands:

    python3 -m vision_mvp.experiments.phase38_primitive_ablation \\
        --seeds 35 36 --distractor-counts 4 6 \\
        --out vision_mvp/results_phase38_primitive_ablation.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

from vision_mvp.core.primitive_ablation import (
    AblationResult, AblatedFeatures, FEATURES,
    full_features, only_missing,
    run_ablated_thread_contested,
    run_ablated_thread_nested,
)
from vision_mvp.tasks.contested_incident import (
    build_contested_bank, decoder_from_handoffs_phase35,
)
from vision_mvp.tasks.nested_contested_incident import (
    build_nested_bank, grade_nested,
)


def _grade_contested(scenario, handoffs) -> bool:
    cue = decoder_from_handoffs_phase35(handoffs)
    return (cue["root_cause"] == scenario.gold_root_cause
            and tuple(sorted(cue["services"])) == tuple(
                sorted(scenario.gold_services))
            and cue["remediation"] == scenario.gold_remediation)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int,
                      default=[35, 36])
    ap.add_argument("--distractor-counts", nargs="+", type=int,
                      default=[4, 6])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    t0 = time.time()

    # Build the ablation grid: full + one-feature-missing per
    # feature + no-features control.
    configs: list[tuple[str, AblatedFeatures]] = []
    configs.append(("full", full_features()))
    for f in FEATURES:
        configs.append((f"no_{f}", only_missing(f)))
    configs.append(("no_all", AblatedFeatures(
        typed_vocab=False, bounded_witness=False,
        terminating_resolution=False,
        round_aware_state=False, frozen_membership=False)))

    contested_results: list[AblationResult] = []
    nested_results: list[AblationResult] = []

    print(f"\n[phase38-B] running {len(configs)} configs on "
          f"contested + nested banks", flush=True)

    for k in args.distractor_counts:
        for seed in args.seeds:
            contested_bank = build_contested_bank(
                seed=seed, distractors_per_role=k)
            nested_bank = build_nested_bank(
                seed=seed, distractors_per_role=k)
            # Contested bank
            for (label, feats) in configs:
                for scenario in contested_bank:
                    router, handoffs, dbg = \
                        run_ablated_thread_contested(
                            scenario, feats)
                    ok = _grade_contested(scenario, handoffs)
                    contested_results.append(AblationResult(
                        features=feats, family="contested",
                        scenario_id=scenario.scenario_id,
                        full_correct=ok,
                        resolution_kind=dbg.get(
                            "resolution_kind"),
                        debug=dbg,
                    ))
                for scenario in nested_bank:
                    router, handoffs, dbg = \
                        run_ablated_thread_nested(
                            scenario, feats)
                    grading = grade_nested(scenario, handoffs)
                    ok = grading["full_correct"]
                    nested_results.append(AblationResult(
                        features=feats, family="nested",
                        scenario_id=scenario.scenario_id,
                        full_correct=ok,
                        resolution_kind=dbg.get(
                            "resolution_kind"),
                        debug=dbg,
                    ))

    wall = time.time() - t0
    print(f"\n[phase38-B] wall = {wall:.1f}s")

    # Pool per (config, family).
    def _pool(results: list[AblationResult]) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for r in results:
            key = r.features.label()
            acc = out.setdefault(key, {
                "n": 0, "n_correct": 0,
                "features": r.features.as_dict(),
                "label": r.features.label(),
                "resolution_hist": {}})
            acc["n"] += 1
            if r.full_correct:
                acc["n_correct"] += 1
            rk = r.resolution_kind or "NONE"
            acc["resolution_hist"][rk] = \
                acc["resolution_hist"].get(rk, 0) + 1
        for key, acc in out.items():
            n = max(1, acc["n"])
            acc["accuracy"] = round(acc["n_correct"] / n, 4)
        return out

    contested_pool = _pool(contested_results)
    nested_pool = _pool(nested_results)

    print()
    print("=" * 78)
    print("PHASE 38 PART B — minimum-primitive ablation (pooled)")
    print("=" * 78)
    print(f"{'config':36} {'contested':>12} {'nested':>12}")
    for (label, feats) in configs:
        key = feats.label()
        c = contested_pool.get(key, {}).get("accuracy", 0.0)
        n = nested_pool.get(key, {}).get("accuracy", 0.0)
        print(f"{label:36} {c:12.3f} {n:12.3f}")

    payload = {
        "config": vars(args),
        "configs": [
            {"label": lbl, "features": f.as_dict()}
            for (lbl, f) in configs],
        "contested_results": [r.as_dict()
                                for r in contested_results],
        "nested_results": [r.as_dict()
                            for r in nested_results],
        "contested_pool": contested_pool,
        "nested_pool": nested_pool,
        "wall_seconds": round(wall, 2),
    }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
