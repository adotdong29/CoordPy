"""Phase 36 Part A — dynamic coordination under noisy producer-local replies.

Sweeps ``core/reply_noise.ReplyNoiseConfig`` across a Bernoulli
drop/mislabel grid on the Phase-35 contested bank, running the
dynamic, adaptive-sub, and static-handoff strategies side-by-side
under a matched noise budget. Produces the Phase-36 Part A
headline: does the Phase-35 dynamic-coordination benefit survive
imperfect producer reflection?

Reproducible commands (mock auditor, sub-second wall):

    python3 -m vision_mvp.experiments.phase36_noisy_dynamic \\
        --mock --seeds 35 36 \\
        --drop-probs 0.0 0.1 0.25 0.5 0.75 1.0 \\
        --mislabel-probs 0.0 0.1 0.25 \\
        --distractor-counts 6 \\
        --out vision_mvp/results_phase36_noisy_dynamic.json

    # Adversarial reply path (matched nominal budget vs i.i.d.)
    python3 -m vision_mvp.experiments.phase36_noisy_dynamic \\
        --mock --seeds 35 \\
        --adversarial drop_root \\
        --out vision_mvp/results_phase36_adversarial_reply.json

Scope: reuses the Phase-35 harness + scenario bank. The
contested decoder is unchanged; only the producer-local
causality extractor is perturbed at the reply boundary.
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
    AdversarialReplyConfig, adversarial_reply_extractor,
    ADVERSARIAL_REPLY_MODE_DROP_ROOT,
    ADVERSARIAL_REPLY_MODE_FLIP_ROOT_TO_SYMPTOM,
    ADVERSARIAL_REPLY_MODE_INJECT_ROOT_ON_SYMPTOM,
    ADVERSARIAL_REPLY_MODE_COMBINED,
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
        return client.generate(prompt, max_tokens=80, temperature=0.0)
    return _call, client.stats


_ADV_MODES = {
    "drop_root": ADVERSARIAL_REPLY_MODE_DROP_ROOT,
    "flip_root_to_symptom": ADVERSARIAL_REPLY_MODE_FLIP_ROOT_TO_SYMPTOM,
    "inject_root_on_symptom": ADVERSARIAL_REPLY_MODE_INJECT_ROOT_ON_SYMPTOM,
    "combined": ADVERSARIAL_REPLY_MODE_COMBINED,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:0.5b")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--seeds", nargs="+", type=int, default=[35, 36])
    ap.add_argument("--distractor-counts", nargs="+", type=int,
                      default=[6])
    ap.add_argument("--drop-probs", nargs="+", type=float,
                      default=[0.0, 0.1, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--mislabel-probs", nargs="+", type=float,
                      default=[0.0])
    ap.add_argument("--adversarial", choices=list(_ADV_MODES.keys()),
                      default=None,
                      help="run adversarial reply wrapper instead of "
                           "i.i.d. Bernoulli noise")
    ap.add_argument("--adv-budget", type=int, default=1)
    ap.add_argument("--strategies", nargs="+",
                      default=[STRATEGY_STATIC_HANDOFF,
                                STRATEGY_DYNAMIC,
                                STRATEGY_ADAPTIVE_SUB])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    auditor, stats = _make_auditor(args.model, args.mock)

    overall_start = time.time()
    per_config: list[dict] = []

    grid: list[tuple[float, float]] = []
    if args.adversarial:
        # Adversarial sweep is just one cell (no i.i.d. grid).
        grid = [(-1.0, -1.0)]
    else:
        for dp in args.drop_probs:
            for mp in args.mislabel_probs:
                grid.append((dp, mp))

    for k in args.distractor_counts:
        for seed in args.seeds:
            bank = build_contested_bank(
                seed=seed, distractors_per_role=k)
            for (dp, mp) in grid:
                # Build a fresh extractor per strategy so stateful
                # wrappers (adversarial budgets, deterministic RNG)
                # are isolated across strategies. Otherwise a
                # single extractor's internal budget is depleted
                # by the first strategy and the second/third
                # strategy sees a clean oracle — a driver artifact.
                pooled_accum: dict[str, dict] = {}
                measurements_accum: list = []
                strategy_noise_counters: dict[str, dict] = {}
                for strat in args.strategies:
                    if strat not in ("dynamic", "dynamic_wrap",
                                       "adaptive_sub"):
                        # Strategies without a replier see no
                        # noise at this boundary.
                        rep_rep = ReplyCorruptionReport()
                        extractor = None
                    else:
                        rep_rep = ReplyCorruptionReport()
                        if args.adversarial:
                            adv_mode = _ADV_MODES[args.adversarial]
                            adv_cfg = AdversarialReplyConfig(
                                target_mode=adv_mode,
                                budget=args.adv_budget,
                            )
                            extractor = adversarial_reply_extractor(
                                infer_causality_hypothesis, adv_cfg,
                                report=rep_rep)
                        else:
                            cfg = ReplyNoiseConfig(
                                drop_prob=dp, mislabel_prob=mp,
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
                    measurements_accum.extend(
                        rep_single.measurements)
                    strategy_noise_counters[strat] = \
                        rep_rep.as_dict()
                noise_dict = ({
                    "adversarial": args.adversarial,
                    "budget": args.adv_budget,
                } if args.adversarial else ReplyNoiseConfig(
                    drop_prob=dp, mislabel_prob=mp, seed=seed
                ).as_dict())
                print(
                    f"\n[phase36-A] k={k} seed={seed} "
                    f"noise={noise_dict}", flush=True)
                for s in args.strategies:
                    p = pooled_accum.get(s, {})
                    if not p:
                        continue
                    print(
                        f"    {s:>18}  "
                        f"acc_full={p['accuracy_full']:.3f}  "
                        f"contest={p['contested_accuracy_full']:.3f}  "
                        f"tok={p['mean_prompt_tokens']:.0f}  "
                        f"fhist={p['failure_hist']}  "
                        f"noise={strategy_noise_counters.get(s, {})}",
                        flush=True)
                per_config.append({
                    "distractors_per_role": k,
                    "seed": seed,
                    "noise": noise_dict,
                    "noise_counters_by_strategy":
                        strategy_noise_counters,
                    "report": {
                        "pooled": pooled_accum,
                        "measurements": [m.as_dict()
                                          for m in measurements_accum],
                    },
                })

    wall = time.time() - overall_start
    print(f"\n[phase36-A] overall wall = {wall:.1f}s", flush=True)

    # Pool across seeds per (k, strategy, drop_p, mislabel_p).
    pooled_out: dict[str, dict] = {}
    by_cell: dict[tuple, list[dict]] = {}
    for entry in per_config:
        k = entry["distractors_per_role"]
        nd = entry["noise"]
        dp = nd.get("drop_prob", -1.0) if "drop_prob" in nd else \
            f"adv/{nd.get('adversarial')}/b={nd.get('budget')}"
        mp = nd.get("mislabel_prob", -1.0) \
            if "mislabel_prob" in nd else 0.0
        for strat, row in entry["report"]["pooled"].items():
            key = (k, strat, dp, mp)
            by_cell.setdefault(key, []).append(row)
    for (k, strat, dp, mp), rows in by_cell.items():
        n = len(rows)
        mean_acc = sum(r["accuracy_full"] for r in rows) / n
        mean_contest = sum(
            r["contested_accuracy_full"] for r in rows) / n
        mean_tok = sum(r["mean_prompt_tokens"] for r in rows) / n
        fhist: dict[str, int] = {}
        for r in rows:
            for kk, vv in r["failure_hist"].items():
                fhist[kk] = fhist.get(kk, 0) + vv
        pooled_out[f"k={k}:{strat}:drop={dp}:mis={mp}"] = {
            "n": n,
            "accuracy_full_mean": round(mean_acc, 4),
            "contested_accuracy_full_mean": round(mean_contest, 4),
            "mean_prompt_tokens": round(mean_tok, 2),
            "failure_hist": fhist,
        }

    print()
    print("=" * 72)
    print("PHASE 36 PART A — POOLED (k, strategy, drop, mislabel)")
    print("=" * 72)
    for key in sorted(pooled_out.keys()):
        row = pooled_out[key]
        print(
            f"  {key:>55}  acc={row['accuracy_full_mean']:.3f}  "
            f"cont={row['contested_accuracy_full_mean']:.3f}  "
            f"tok={row['mean_prompt_tokens']:.0f}  "
            f"fhist={row['failure_hist']}")

    llm_stats: dict | None = None
    if hasattr(stats, "total_tokens"):
        llm_stats = {
            "prompt_tokens": stats.prompt_tokens,
            "output_tokens": stats.output_tokens,
            "n_generate_calls": stats.n_generate_calls,
            "total_wall": round(stats.total_wall, 2),
        }

    payload = {
        "config": {
            "model": args.model, "mock": args.mock,
            "seeds": args.seeds,
            "distractor_counts": args.distractor_counts,
            "drop_probs": args.drop_probs,
            "mislabel_probs": args.mislabel_probs,
            "adversarial": args.adversarial,
            "adv_budget": args.adv_budget,
            "strategies": list(args.strategies),
        },
        "per_config": per_config,
        "pooled": pooled_out,
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
