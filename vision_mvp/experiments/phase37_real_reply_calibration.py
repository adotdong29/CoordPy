"""Phase 37 Part A — real-LLM reply calibration.

Runs ``LLMThreadReplier`` wrapped in ``CalibratingReplier``
against a real Ollama model on the Phase-35 contested bank.
Buckets every call against the Phase-35 oracle
(``infer_causality_hypothesis``) into the Phase-37 calibration
taxonomy:

  * CAL_CORRECT              — well-formed, correct reply_kind.
  * CAL_MALFORMED            — no JSON recovered.
  * CAL_OUT_OF_VOCAB         — JSON recovered, unknown reply_kind.
  * CAL_SEM_*                — semantic error buckets (6 pairs).
  * CAL_WITNESS_TRUNCATED    — orthogonal budget counter.

The driver also records the dynamic-coordination accuracy under
the real-LLM replier for comparison to the synthetic
``malformed_prob`` curve from Phase-36 Part B.

Headline output (``--out <path>``):

    {
      "config": {...},
      "per_model": [
        {"model": "qwen2.5:0.5b",
         "calibration_rates": {...},
         "pooled": {...},
         "llm_stats": {...}},
        ...
      ],
      "synthetic_anchor": {...},   # reproducible anchor
    }

Reproducible commands:

    # Calibrate the two stock local models.
    python3 -m vision_mvp.experiments.phase37_real_reply_calibration \\
        --models qwen2.5:0.5b qwen2.5-coder:7b --seeds 35 \\
        --distractor-counts 4 \\
        --out vision_mvp/results_phase37_real_reply_calibration.json

    # Single-model quick run (sub-minute on 0.5b).
    python3 -m vision_mvp.experiments.phase37_real_reply_calibration \\
        --models qwen2.5:0.5b --seeds 35 --distractor-counts 4 \\
        --out vision_mvp/results_phase37_real_reply_calibration_0p5b.json

The auditor is always the deterministic ``MockContestedAuditor``
so the reply axis is isolated from auditor synthesis noise.
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
from vision_mvp.core.llm_thread_replier import (
    LLMReplyConfig, LLMReplierStats, LLMThreadReplier,
)
from vision_mvp.core.reply_calibration import (
    CalibratingReplier, ReplyCalibrationReport,
    causality_extractor_from_calibrating_replier,
)
from vision_mvp.tasks.contested_incident import (
    STRATEGY_DYNAMIC, STRATEGY_ADAPTIVE_SUB,
    STRATEGY_STATIC_HANDOFF, MockContestedAuditor,
    build_contested_bank, run_contested_loop,
    infer_causality_hypothesis,
)


def _build_real_replier(model: str, timeout: float = 300.0
                          ) -> tuple[LLMThreadReplier, LLMClient]:
    client = LLMClient(model=model, timeout=timeout)

    def _call(prompt: str) -> str:
        return client.generate(prompt, max_tokens=60,
                                temperature=0.0)
    replier = LLMThreadReplier(
        llm_call=_call,
        config=LLMReplyConfig(witness_token_cap=12),
        cache={},
    )
    return replier, client


def run_model_calibration(model: str,
                           seeds: list[int],
                           distractor_counts: list[int],
                           strategies: tuple[str, ...] = (
                               STRATEGY_DYNAMIC,
                               STRATEGY_ADAPTIVE_SUB,
                               STRATEGY_STATIC_HANDOFF),
                           max_events_in_prompt: int = 200,
                           timeout: float = 300.0,
                           ) -> dict:
    """Run the Phase-36 contested loop under a real-LLM replier
    wrapped in CalibratingReplier. Returns a dict with calibration
    rates, pooled per-strategy accuracy, and LLM client stats.
    """
    auditor = MockContestedAuditor()
    replier, client = _build_real_replier(model, timeout=timeout)
    report = ReplyCalibrationReport()
    wrapper = CalibratingReplier(
        inner=replier, oracle=infer_causality_hypothesis,
        report=report,
        witness_token_cap=replier.config.witness_token_cap,
    )
    extractor = causality_extractor_from_calibrating_replier(wrapper)

    t0 = time.time()
    per_config: list[dict] = []
    pooled_by_strat_cells: dict[str, list[dict]] = {}
    for k in distractor_counts:
        for seed in seeds:
            bank = build_contested_bank(
                seed=seed, distractors_per_role=k)
            pooled_accum: dict[str, dict] = {}
            for strat in strategies:
                # Reset per-strategy counters so the calibration
                # report accumulates calls across strategies (the
                # dynamic / adaptive_sub replier is called per
                # contested scenario; static never triggers it).
                rep_single = run_contested_loop(
                    bank, auditor, strategies=(strat,),
                    seed=seed,
                    max_events_in_prompt=max_events_in_prompt,
                    inbox_capacity=32,
                    causality_extractor=extractor,
                )
                pooled_accum.update(rep_single.pooled())
                pooled_by_strat_cells.setdefault(strat, []).append(
                    rep_single.pooled().get(strat, {}))
            per_config.append({
                "distractors_per_role": k, "seed": seed,
                "pooled": pooled_accum,
            })
    wall = time.time() - t0

    # Aggregate per-strategy accuracy over cells.
    pooled_means: dict[str, dict] = {}
    for strat, rows in pooled_by_strat_cells.items():
        if not rows:
            continue
        n = len(rows)
        mean_acc = sum(r.get("accuracy_full", 0.0)
                        for r in rows) / max(1, n)
        mean_cont = sum(r.get("contested_accuracy_full", 0.0)
                         for r in rows) / max(1, n)
        pooled_means[strat] = {
            "n_cells": n,
            "accuracy_full_mean": round(mean_acc, 4),
            "contested_accuracy_full_mean": round(mean_cont, 4),
        }

    return {
        "model": model,
        "calibration": report.as_dict(),
        "calibration_rates": report.rates(),
        "per_config": per_config,
        "pooled_mean_over_cells": pooled_means,
        "inner_replier_stats": replier.stats.as_dict(),
        "llm_client_stats": {
            "prompt_tokens": client.stats.prompt_tokens,
            "output_tokens": client.stats.output_tokens,
            "n_generate_calls": client.stats.n_generate_calls,
            "total_wall": round(client.stats.total_wall, 2),
        },
        "wall_seconds": round(wall, 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                      default=["qwen2.5:0.5b"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[35])
    ap.add_argument("--distractor-counts", nargs="+", type=int,
                      default=[4])
    ap.add_argument("--strategies", nargs="+",
                      default=[STRATEGY_DYNAMIC,
                                STRATEGY_ADAPTIVE_SUB,
                                STRATEGY_STATIC_HANDOFF])
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    t0 = time.time()
    per_model: list[dict] = []
    for model in args.models:
        print(f"\n[phase37-A] calibrating model={model!r}",
              flush=True)
        try:
            row = run_model_calibration(
                model=model,
                seeds=args.seeds,
                distractor_counts=args.distractor_counts,
                strategies=tuple(args.strategies),
                timeout=args.timeout,
            )
        except Exception as ex:  # don't blow up the whole sweep
            row = {"model": model, "error": str(ex)}
        per_model.append(row)
        if "calibration_rates" in row:
            rates = row["calibration_rates"]
            print(f"  n_calls={rates['n_calls']}  "
                  f"correct={rates['correct_rate']:.3f}  "
                  f"malformed={rates['malformed_rate']:.3f}  "
                  f"oov={rates['out_of_vocab_rate']:.3f}  "
                  f"sem_wrong={rates['semantic_wrong_rate']:.3f}  "
                  f"wtrunc={rates['witness_truncation_rate']:.3f}")
            for strat, p in row["pooled_mean_over_cells"].items():
                print(f"    {strat:>20}  "
                      f"acc_full={p['accuracy_full_mean']:.3f}  "
                      f"contest={p['contested_accuracy_full_mean']:.3f}")
            print(f"  wall={row['wall_seconds']:.1f}s "
                  f"prompt_tokens={row['llm_client_stats']['prompt_tokens']} "
                  f"output_tokens={row['llm_client_stats']['output_tokens']}")

    payload = {
        "config": {
            "models": args.models,
            "seeds": args.seeds,
            "distractor_counts": args.distractor_counts,
            "strategies": list(args.strategies),
            "timeout": args.timeout,
        },
        "per_model": per_model,
        "wall_seconds": round(time.time() - t0, 2),
    }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
