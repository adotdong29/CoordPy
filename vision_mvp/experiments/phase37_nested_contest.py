"""Phase 37 Part C — nested-contest thread-vs-adaptive comparison.

Phase 36 Part C Theorem P36-4 showed dynamic threads and
bounded adaptive subscriptions are empirically equivalent on
the Phase-35 contested bank — every noise cell, 0 pp gap.
Conjecture C36-5 extended the equivalence to a broader task
family, but flagged candidates (nested contests, role-local
reply memory across rounds, authenticated provenance) where
the thread might dominate.

This driver evaluates the equivalence on a task family that
sits *outside* Phase-36's one-round window: nested-contest
scenarios where producer-local evidence is insufficient
without at least one peer's round-1 reply. Four strategies:

  * ``static_handoff``      — Phase-31 baseline.
  * ``adaptive_sub_1r``     — Phase-36 single-round bounded
    adaptive subscriptions. Dominated baseline on nested.
  * ``adaptive_sub_2r``     — two-round adaptive sub with an
    inter-round auditor-to-producer briefing edge
    (``CLAIM_COORDINATION_BRIEFING``). See
    ``tasks/nested_contested_incident.run_nested_two_round_
    adaptive_sub``.
  * ``dynamic_nested_2r``   — Phase-35 escalation thread with
    ``max_rounds=2``; round-2 producers read the thread's
    ``replies`` list directly.

The headline diagnostic is the per-strategy accuracy on the
nested bank plus the messaging-budget counters (edges
installed, replies posted, briefings sent). The two 2-round
strategies can either (a) match in accuracy, reinforcing
C36-5 at the *accuracy* level while displaying a structural
differential in the protocol complexity, or (b) diverge,
locating a real primitive separation.

Reproducible commands:

    python3 -m vision_mvp.experiments.phase37_nested_contest \\
        --seeds 37 38 --distractor-counts 4 6 \\
        --out vision_mvp/results_phase37_nested_contest.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

from vision_mvp.tasks.nested_contested_incident import (
    NESTED_ALL_STRATEGIES, STRATEGY_NESTED_ADAPTIVE_1R,
    STRATEGY_NESTED_ADAPTIVE_2R, STRATEGY_NESTED_DYNAMIC,
    STRATEGY_NESTED_STATIC, build_nested_bank, run_nested_bank,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[37, 38])
    ap.add_argument("--distractor-counts", nargs="+", type=int,
                      default=[4, 6])
    ap.add_argument("--strategies", nargs="+",
                      default=list(NESTED_ALL_STRATEGIES))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    t0 = time.time()
    per_cell: list[dict] = []
    per_strat_acc: dict[str, list[float]] = {}
    per_strat_log_lengths: dict[str, list[int]] = {}
    per_strat_n_edges_installed: dict[str, list[int]] = {}
    per_strat_n_briefings: dict[str, list[int]] = {}

    for k in args.distractor_counts:
        for seed in args.seeds:
            bank = build_nested_bank(
                seed=seed, distractors_per_role=k)
            ms = run_nested_bank(
                bank, strategies=tuple(args.strategies),
                max_events_per_role=200, inbox_capacity=32,
                witness_token_cap=12)
            # Per-strategy accuracy + messaging accounting.
            by_strat: dict[str, list] = {}
            for m in ms:
                by_strat.setdefault(m.strategy, []).append(m)
            print(f"\n[phase37-C] k={k} seed={seed}", flush=True)
            for strat in args.strategies:
                msr = by_strat.get(strat, [])
                if not msr:
                    continue
                acc = sum(1 for m in msr
                            if m.grading["full_correct"]) / len(msr)
                log_len = sum(m.handoff_log_length for m in msr)
                n_edges = sum(
                    m.debug.get("n_hypothesis_edges_installed", 0)
                    for m in msr)
                n_brief = sum(
                    m.debug.get("n_briefings_installed", 0)
                    for m in msr)
                print(f"    {strat:>20}  "
                      f"acc={acc:.3f}  log_len={log_len}  "
                      f"edges={n_edges}  briefings={n_brief}",
                      flush=True)
                per_strat_acc.setdefault(strat, []).append(acc)
                per_strat_log_lengths.setdefault(strat, []).append(log_len)
                per_strat_n_edges_installed.setdefault(
                    strat, []).append(n_edges)
                per_strat_n_briefings.setdefault(strat, []).append(n_brief)
            per_cell.append({
                "distractors_per_role": k, "seed": seed,
                "measurements": [m.as_dict() for m in ms],
            })

    wall = time.time() - t0
    print(f"\n[phase37-C] overall wall = {wall:.1f}s")

    # Pooled across cells.
    print()
    print("=" * 72)
    print("PHASE 37 PART C — pooled nested comparison")
    print("=" * 72)
    pooled: dict[str, dict] = {}
    for strat in args.strategies:
        accs = per_strat_acc.get(strat, [])
        if not accs:
            continue
        n = len(accs)
        mean_acc = sum(accs) / n
        logs = per_strat_log_lengths.get(strat, [])
        edges = per_strat_n_edges_installed.get(strat, [])
        briefs = per_strat_n_briefings.get(strat, [])
        pooled[strat] = {
            "n_cells": n,
            "accuracy_full_mean": round(mean_acc, 4),
            "log_length_total": sum(logs),
            "n_edges_installed_total": sum(edges),
            "n_briefings_total": sum(briefs),
        }
        print(f"  {strat:>20}  acc={mean_acc:.3f}  "
              f"log_len_total={sum(logs)}  "
              f"edges_total={sum(edges)}  "
              f"briefings_total={sum(briefs)}")

    # Equivalence diagnostic (dynamic_nested_2r vs adaptive_sub_2r).
    if (STRATEGY_NESTED_DYNAMIC in pooled
            and STRATEGY_NESTED_ADAPTIVE_2R in pooled):
        dyn = pooled[STRATEGY_NESTED_DYNAMIC]["accuracy_full_mean"]
        adp2 = pooled[STRATEGY_NESTED_ADAPTIVE_2R][
            "accuracy_full_mean"]
        dyn_log = pooled[STRATEGY_NESTED_DYNAMIC]["log_length_total"]
        adp_log = pooled[STRATEGY_NESTED_ADAPTIVE_2R][
            "log_length_total"]
        gap = dyn - adp2
        print()
        print("  gap(dyn vs adaptive_sub_2r) = "
              f"{gap:+.4f} acc; {dyn_log} vs {adp_log} log entries")

    payload = {
        "config": vars(args),
        "per_cell": per_cell,
        "pooled": pooled,
        "wall_seconds": round(wall, 2),
    }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
