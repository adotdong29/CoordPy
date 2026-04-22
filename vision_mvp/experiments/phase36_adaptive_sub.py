"""Phase 36 Part C — adaptive-subscription vs dynamic-thread comparison.

Conjecture C35-5 named the design-space alternative to escalation
threads: **bounded adaptive subscriptions** — instead of opening a
short-lived thread object, the auditor temporarily edits the
subscription table to route a new claim kind from each producer
for a bounded number of rounds, and then removes the edit.

Phase 36 Part C ships this alternative as ``core/adaptive_sub``
and compares it head-to-head with the Phase-35 dynamic-thread
path on:

  * the clean contested bank
  * the noisy-reply bank (Phase-36 Part A grid)
  * an LLM-reply bank (Phase-36 Part B, scenario-aware mock)

The driver produces per-strategy accuracy and per-primitive
messaging budget (edges installed / thread replies / witness
tokens per scenario) so the empirical equivalence / separation
of C35-5 can be quantified on a single artifact.

Headline result (see RESULTS_PHASE36.md § D.3):

  * On the clean bank: dynamic = adaptive_sub = 100 %; both
    dominate static at 33 %. Thread path costs 10 replies /
    49 witness tokens total; adaptive_sub costs 10 hypothesis
    handoffs / ~43 witness tokens total — within 12 %.
  * On the noisy-reply grid (drop_prob sweep): dynamic and
    adaptive_sub track within 0–1 pp across every cell. The
    primitive choice does not change the degradation curve.
  * Under adversarial drop_root (budget=1): both collapse to
    the static baseline (33 %). The primitive choice is
    irrelevant when the producer-local reflection is
    unrecoverably wrong.

Reproducible commands:

    # Clean bank + matched noise grid, 2 seeds, k=6.
    python3 -m vision_mvp.experiments.phase36_adaptive_sub \\
        --mock --seeds 35 36 --distractor-counts 6 \\
        --drop-probs 0.0 0.25 0.5 1.0 \\
        --out vision_mvp/results_phase36_adaptive_sub.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Callable

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

from vision_mvp.core.llm_client import LLMClient
from vision_mvp.core.reply_noise import (
    ReplyNoiseConfig, ReplyCorruptionReport,
    noisy_causality_extractor,
)
from vision_mvp.tasks.contested_incident import (
    STRATEGY_STATIC_HANDOFF, STRATEGY_DYNAMIC,
    STRATEGY_ADAPTIVE_SUB, MockContestedAuditor,
    build_contested_bank, run_contested_loop,
    infer_causality_hypothesis,
)


def _make_auditor(model: str, mock: bool
                   ) -> tuple[Callable[[str], str], object]:
    if mock:
        m = MockContestedAuditor()
        return m, m
    client = LLMClient(model=model, timeout=300.0)

    def _call(prompt: str) -> str:
        return client.generate(prompt, max_tokens=80,
                                temperature=0.0)
    return _call, client.stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:0.5b")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--seeds", nargs="+", type=int, default=[35, 36])
    ap.add_argument("--distractor-counts", nargs="+", type=int,
                      default=[6, 20])
    ap.add_argument("--drop-probs", nargs="+", type=float,
                      default=[0.0, 0.25, 0.5, 1.0])
    ap.add_argument("--mislabel-probs", nargs="+", type=float,
                      default=[0.0])
    ap.add_argument("--strategies", nargs="+",
                      default=[STRATEGY_STATIC_HANDOFF,
                                STRATEGY_DYNAMIC,
                                STRATEGY_ADAPTIVE_SUB])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    auditor, stats = _make_auditor(args.model, args.mock)

    t0 = time.time()
    per_config: list[dict] = []
    for k in args.distractor_counts:
        for seed in args.seeds:
            bank = build_contested_bank(
                seed=seed, distractors_per_role=k)
            for dp in args.drop_probs:
                for mp in args.mislabel_probs:
                    pooled_accum: dict[str, dict] = {}
                    per_strategy_noise: dict[str, dict] = {}
                    for strat in args.strategies:
                        rep_rep = ReplyCorruptionReport()
                        cfg = ReplyNoiseConfig(
                            drop_prob=dp,
                            mislabel_prob=mp,
                            seed=seed)
                        extractor = noisy_causality_extractor(
                            infer_causality_hypothesis, cfg,
                            report=rep_rep)
                        rep_single = run_contested_loop(
                            bank, auditor, strategies=(strat,),
                            seed=seed,
                            max_events_in_prompt=200,
                            inbox_capacity=32,
                            causality_extractor=extractor,
                        )
                        pooled_accum.update(rep_single.pooled())
                        per_strategy_noise[strat] = \
                            rep_rep.as_dict()
                    print(f"\n[phase36-C] k={k} seed={seed} "
                          f"drop={dp} mis={mp}", flush=True)
                    for s in args.strategies:
                        p = pooled_accum.get(s, {})
                        if not p:
                            continue
                        print(
                            f"    {s:>18}  "
                            f"acc_full={p['accuracy_full']:.3f}  "
                            f"contest={p['contested_accuracy_full']:.3f}  "
                            f"tok={p['mean_prompt_tokens']:.0f}  "
                            f"threads={p['n_threads_opened']}  "
                            f"replies={p['n_thread_replies_total']}  "
                            f"wtok={p['n_thread_witness_tokens_total']}  "
                            f"fhist={p['failure_hist']}",
                            flush=True)
                    per_config.append({
                        "distractors_per_role": k,
                        "seed": seed,
                        "drop_prob": dp,
                        "mislabel_prob": mp,
                        "noise_counters": per_strategy_noise,
                        "pooled": pooled_accum,
                    })

    wall = time.time() - t0
    print(f"\n[phase36-C] overall wall = {wall:.1f}s")

    # Aggregate per-cell equivalence diagnostic:
    #     gap = acc(dynamic) - acc(adaptive_sub)
    cells_by_noise: dict[str, list[dict]] = {}
    for entry in per_config:
        key = f"drop={entry['drop_prob']}:mis={entry['mislabel_prob']}"
        cells_by_noise.setdefault(key, []).append(entry)
    print()
    print("=" * 72)
    print("PHASE 36 PART C — POOLED per-noise-cell "
          "equivalence (dynamic vs adaptive_sub)")
    print("=" * 72)
    for key in sorted(cells_by_noise.keys()):
        entries = cells_by_noise[key]
        dyn_accs: list[float] = []
        adp_accs: list[float] = []
        sta_accs: list[float] = []
        for e in entries:
            if STRATEGY_DYNAMIC in e["pooled"]:
                dyn_accs.append(
                    e["pooled"][STRATEGY_DYNAMIC]["accuracy_full"])
            if STRATEGY_ADAPTIVE_SUB in e["pooled"]:
                adp_accs.append(
                    e["pooled"][STRATEGY_ADAPTIVE_SUB]["accuracy_full"])
            if STRATEGY_STATIC_HANDOFF in e["pooled"]:
                sta_accs.append(
                    e["pooled"][STRATEGY_STATIC_HANDOFF][
                        "accuracy_full"])
        if dyn_accs and adp_accs:
            dyn_mean = sum(dyn_accs) / len(dyn_accs)
            adp_mean = sum(adp_accs) / len(adp_accs)
            sta_mean = (sum(sta_accs) / len(sta_accs)
                         if sta_accs else 0.0)
            print(f"  {key:>28}  "
                  f"dyn={dyn_mean:.3f}  adp={adp_mean:.3f}  "
                  f"static={sta_mean:.3f}  "
                  f"gap_dyn_vs_adp={dyn_mean - adp_mean:+.3f}")

    llm_stats: dict | None = None
    if hasattr(stats, "total_tokens"):
        llm_stats = {
            "prompt_tokens": stats.prompt_tokens,
            "output_tokens": stats.output_tokens,
            "n_generate_calls": stats.n_generate_calls,
            "total_wall": round(stats.total_wall, 2),
        }

    payload = {
        "config": vars(args),
        "per_config": per_config,
        "wall_seconds": round(wall, 2),
        "llm_stats": llm_stats,
    }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
