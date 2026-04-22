"""Phase 35 — contested-incident benchmark driver.

Runs the ``vision_mvp.tasks.contested_incident`` harness across the
six-scenario bank (4 genuine contests + 2 controls) under four
delivery strategies:

  * ``naive``            — raw broadcast to the auditor.
  * ``static_handoff``   — Phase-31 typed handoffs + static
                            priority decoder.
  * ``dynamic``          — Phase-35 dynamic coordination: typed
                            handoffs + single escalation thread
                            per contested scenario + thread-
                            resolution-aware decoder.
  * ``dynamic_wrap``     — dynamic strategy + explicit "copy
                            verbatim" instruction for the wrap
                            path (for LLM-in-loop runs).

Headline claim (mock reader ceiling, seed=35, k ∈ {6, 20, 60, 120}):

  * ``naive``            acc_full = 0.33,  mean tokens ≈ 600 @ k=6
  * ``static_handoff``   acc_full = 0.33,  mean tokens ≈ 215 @ k=6
  * ``dynamic``          acc_full = 1.00,  mean tokens ≈ 246 @ k=6
                         (+ 5 threads, 10 replies, ≤ 12 tokens each)
  * ``dynamic_wrap``     acc_full = 1.00,  mean tokens ≈ 297 @ k=6

See RESULTS_PHASE35.md § D.

Reproducible commands:

    # Mock auditor, full distractor sweep, two seeds
    python3 -m vision_mvp.experiments.phase35_contested_incident \\
        --mock --distractor-counts 6 20 60 120 --seeds 35 36 \\
        --out vision_mvp/results_phase35_mock.json

    # Real Ollama auditor
    python3 -m vision_mvp.experiments.phase35_contested_incident \\
        --model qwen2.5:0.5b --distractor-counts 6 \\
        --out vision_mvp/results_phase35_llm_0p5b.json

Scope: same as Phase 31 — deterministic grader, no model-judged
grading, per-scenario seeds are independent.
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
from vision_mvp.tasks.contested_incident import (
    ALL_STRATEGIES, MockContestedAuditor,
    build_contested_bank, run_contested_loop,
)


def _make_auditor(model: str, mock: bool = False,
                   max_answer_tokens: int = 80
                   ) -> tuple[Callable[[str], str], object]:
    if mock:
        m = MockContestedAuditor()
        return m, m
    client = LLMClient(model=model, timeout=300.0)

    def _call(prompt: str) -> str:
        return client.generate(prompt, max_tokens=max_answer_tokens,
                                temperature=0.0)
    return _call, client.stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:0.5b")
    ap.add_argument("--mock", action="store_true",
                      help="deterministic mock auditor (no LLM calls)")
    ap.add_argument("--distractor-counts", nargs="+", type=int,
                      default=[6])
    ap.add_argument("--seeds", nargs="+", type=int, default=[35])
    ap.add_argument("--max-events-in-prompt", type=int, default=200)
    ap.add_argument("--inbox-capacity", type=int, default=32)
    ap.add_argument("--strategies", nargs="+",
                      default=list(ALL_STRATEGIES))
    ap.add_argument("--max-answer-tokens", type=int, default=80)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    auditor, stats = _make_auditor(
        args.model, mock=args.mock,
        max_answer_tokens=args.max_answer_tokens)

    overall_start = time.time()
    per_config: list[dict] = []

    for k in args.distractor_counts:
        for seed in args.seeds:
            print(f"\n[phase35] distractors_per_role={k} seed={seed} "
                  f"mock={args.mock}", flush=True)
            bank = build_contested_bank(
                seed=seed, distractors_per_role=k)
            print(f"  n_scenarios={len(bank)} "
                  f"n_contested={sum(1 for s in bank if s.contested)}",
                  flush=True)
            rep = run_contested_loop(
                bank, auditor,
                strategies=tuple(args.strategies),
                seed=seed,
                max_events_in_prompt=args.max_events_in_prompt,
                inbox_capacity=args.inbox_capacity,
            )
            pooled = rep.pooled()
            for s in args.strategies:
                p = pooled.get(s, {})
                if not p:
                    continue
                print(
                    f"    {s:>18}  acc_full={p['accuracy_full']:.3f}  "
                    f"contested_acc={p['contested_accuracy_full']:.3f}  "
                    f"tok={p['mean_prompt_tokens']:>6.0f}  "
                    f"threads={p['n_threads_opened']}  "
                    f"replies={p['n_thread_replies_total']}  "
                    f"wtok={p['n_thread_witness_tokens_total']}  "
                    f"trunc={p['truncated_count']}  "
                    f"fhist={p['failure_hist']}",
                    flush=True)
            per_config.append({
                "distractors_per_role": k,
                "seed": seed,
                "report": rep.as_dict(),
            })

    overall = time.time() - overall_start
    print(f"\n[phase35] overall wall = {overall:.1f}s", flush=True)

    # Pooled across seeds, per (strategy, k).
    by_k_strategy: dict[tuple[int, str], list[dict]] = {}
    for entry in per_config:
        for strat, row in entry["report"]["pooled"].items():
            key = (entry["distractors_per_role"], strat)
            by_k_strategy.setdefault(key, []).append(row)
    pooled_out: dict[str, dict] = {}
    for (k, strat), rows in by_k_strategy.items():
        n = len(rows)
        mean_acc = sum(r["accuracy_full"] for r in rows) / n
        mean_contest_acc = sum(
            r["contested_accuracy_full"] for r in rows) / n
        mean_rc_acc = sum(r["accuracy_root_cause"] for r in rows) / n
        mean_tok = sum(r["mean_prompt_tokens"] for r in rows) / n
        threads = sum(r["n_threads_opened"] for r in rows)
        replies = sum(r["n_thread_replies_total"] for r in rows)
        wtok = sum(r["n_thread_witness_tokens_total"] for r in rows)
        trunc = sum(r["truncated_count"] for r in rows)
        fhist: dict[str, int] = {}
        for r in rows:
            for kk, vv in r["failure_hist"].items():
                fhist[kk] = fhist.get(kk, 0) + vv
        pooled_out[f"k={k}:{strat}"] = {
            "n_reports": n,
            "accuracy_full_mean": round(mean_acc, 4),
            "accuracy_root_cause_mean": round(mean_rc_acc, 4),
            "contested_accuracy_full_mean": round(mean_contest_acc, 4),
            "mean_prompt_tokens": round(mean_tok, 2),
            "n_threads_opened_total": threads,
            "n_thread_replies_total": replies,
            "n_thread_witness_tokens_total": wtok,
            "truncated_count_total": trunc,
            "failure_hist": fhist,
        }
    print()
    print("=" * 72)
    print("PHASE 35 POOLED — per (distractor_count, strategy)")
    print("=" * 72)
    for key in sorted(pooled_out.keys()):
        row = pooled_out[key]
        print(f"  {key:>35}  acc={row['accuracy_full_mean']:.3f}  "
              f"rc_acc={row['accuracy_root_cause_mean']:.3f}  "
              f"cont_acc={row['contested_accuracy_full_mean']:.3f}  "
              f"tok={row['mean_prompt_tokens']:>6.0f}  "
              f"th={row['n_threads_opened_total']}  "
              f"rep={row['n_thread_replies_total']}  "
              f"wtok={row['n_thread_witness_tokens_total']}  "
              f"trunc={row['truncated_count_total']}  "
              f"fhist={row['failure_hist']}")

    llm_stats: dict | None = None
    if hasattr(stats, "total_tokens"):
        llm_stats = {
            "prompt_tokens": stats.prompt_tokens,
            "output_tokens": stats.output_tokens,
            "n_generate_calls": stats.n_generate_calls,
            "total_wall": round(stats.total_wall, 2),
        }
        print(f"  llm: calls={stats.n_generate_calls} "
              f"prompt_toks={stats.prompt_tokens} "
              f"wall={stats.total_wall:.1f}s")
    elif hasattr(stats, "n_calls"):
        llm_stats = {
            "n_calls": stats.n_calls,
            "total_prompt_chars": stats.total_prompt_chars,
        }
        print(f"  mock: calls={stats.n_calls} "
              f"chars={stats.total_prompt_chars}")

    payload = {
        "config": {
            "model": args.model, "mock": args.mock,
            "distractor_counts": args.distractor_counts,
            "seeds": args.seeds,
            "max_events_in_prompt": args.max_events_in_prompt,
            "inbox_capacity": args.inbox_capacity,
            "strategies": list(args.strategies),
            "max_answer_tokens": args.max_answer_tokens,
        },
        "per_config": per_config,
        "pooled": pooled_out,
        "wall_seconds": round(overall, 2),
        "llm_stats": llm_stats,
    }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
