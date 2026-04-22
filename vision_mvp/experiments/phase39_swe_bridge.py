"""Phase 39 — substrate-driven mini SWE-bench bridge driver.

Runs the Phase-39 SWE-bench-style bridge
(``vision_mvp/tasks/swe_bench_bridge``) on its 4-instance mini
bank under three strategies (naive / routing / substrate), with
either:

  * a deterministic oracle patch generator (the substrate's
    *correctness ceiling* — every strategy passes; the report
    is then about the team substrate's bounded-context
    invariant on the patch_generator role); or
  * a real LLM patch generator (Ollama-driven), which surfaces
    the substrate's *correctness benefit* — the LLM only sees
    the typed-handoff cue under substrate and the raw event
    stream under naive.

The goal is NOT to beat SWE-bench. The goal is to ship a
runnable artifact whose schema matches SWE-bench's, whose
multi-role team substrate composes cleanly with the existing
Phase 31..38 communication primitives (typed handoffs,
hash-chained log, role-keyed inboxes), and whose
``--mode real`` measurement gives the programme its first
*team-shaped* SWE-style benchmark headline.

Reproducible commands:

    # Mock — deterministic oracle generator (sub-second).
    python3 -m vision_mvp.experiments.phase39_swe_bridge \\
        --mode mock --n-distractors 0 6 12 \\
        --out vision_mvp/results_phase39_swe_bridge_mock.json

    # Real — qwen2.5:0.5b patch generator on 4 instances.
    python3 -m vision_mvp.experiments.phase39_swe_bridge \\
        --mode real --model qwen2.5:0.5b --n-distractors 6 \\
        --out vision_mvp/results_phase39_swe_bridge_0p5b.json

    # Real — qwen2.5-coder:7b patch generator.
    python3 -m vision_mvp.experiments.phase39_swe_bridge \\
        --mode real --model qwen2.5-coder:7b --n-distractors 6 \\
        --out vision_mvp/results_phase39_swe_bridge_7b.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

from vision_mvp.tasks.swe_bench_bridge import (
    ALL_SWE_STRATEGIES, STRATEGY_NAIVE, STRATEGY_ROUTING,
    STRATEGY_SUBSTRATE, build_mini_swe_bank,
    deterministic_oracle_generator, llm_patch_generator,
    run_swe_loop,
)


def _make_generator(mode: str, model: str | None,
                     timeout: float = 300.0):
    if mode == "mock":
        return deterministic_oracle_generator, None
    from vision_mvp.core.llm_client import LLMClient
    client = LLMClient(model=model, timeout=timeout)

    def _call(prompt: str) -> str:
        return client.generate(prompt, max_tokens=180,
                                temperature=0.0)
    return llm_patch_generator(_call), client


def _pretty(pooled: dict) -> str:
    rows = []
    rows.append(f"{'strategy':>14} {'pass@1':>8} "
                 f"{'apply':>7} {'tok≈':>8} {'evts':>5} "
                 f"{'hand':>5}")
    for strat, p in pooled.items():
        rows.append(f"{strat:>14} {p['pass_at_1']:>8.3f} "
                    f"{p['patch_applied_rate']:>7.3f} "
                    f"{p['mean_patch_gen_prompt_tokens_approx']:>8.1f} "
                    f"{p['mean_events_to_patch_gen']:>5.1f} "
                    f"{p['mean_handoffs']:>5.1f}")
    return "\n".join(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("mock", "real"),
                      default="mock")
    ap.add_argument("--model", default="qwen2.5:0.5b")
    ap.add_argument("--n-distractors", nargs="+", type=int,
                      default=[6])
    ap.add_argument("--strategies", nargs="+",
                      default=list(ALL_SWE_STRATEGIES))
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    overall_start = time.time()
    generator, client = _make_generator(
        args.mode, args.model, timeout=args.timeout)

    per_distractor: list[dict] = []
    for n_distractors in args.n_distractors:
        tasks, repo_files = build_mini_swe_bank(
            n_distractors=n_distractors)
        print(f"\n[phase39] mode={args.mode} "
              f"model={args.model if args.mode == 'real' else '-'} "
              f"n_distractors={n_distractors} n_tasks={len(tasks)}",
              flush=True)
        rep = run_swe_loop(
            bank=tasks, repo_files=repo_files,
            generator=generator,
            strategies=tuple(args.strategies))
        pooled = rep.pooled_summary()
        print(_pretty(pooled))
        # Per-instance breakdown for the substrate strategy.
        sub_rows = [m for m in rep.measurements
                    if m.strategy == STRATEGY_SUBSTRATE]
        if sub_rows:
            print("  per-instance (substrate):")
            for m in sub_rows:
                print(f"    {m.instance_id:>14}  "
                      f"pass={int(m.test_passed)}  "
                      f"apply={int(m.patch_applied)}  "
                      f"err={m.error_kind}")
        per_distractor.append({
            "n_distractors": n_distractors,
            "report": rep.as_dict(),
        })

    overall = time.time() - overall_start

    # Cross-distractor pooled summary.
    pooled_by_strat: dict[str, list[dict]] = {}
    for row in per_distractor:
        for strat, p in row["report"]["pooled"].items():
            pooled_by_strat.setdefault(strat, []).append(p)
    pooled_summary: dict[str, dict] = {}
    for strat, rows in pooled_by_strat.items():
        n = len(rows)
        pooled_summary[strat] = {
            "n_cells": n,
            "pass_at_1_mean": round(sum(
                r["pass_at_1"] for r in rows) / n, 4),
            "tokens_mean": round(sum(
                r["mean_patch_gen_prompt_tokens_approx"]
                for r in rows) / n, 1),
            "events_mean": round(sum(
                r["mean_events_to_patch_gen"]
                for r in rows) / n, 2),
        }

    print()
    print("=" * 72)
    print("PHASE 39 — SWE-bench-style bridge pooled summary")
    print("=" * 72)
    for strat, p in pooled_summary.items():
        print(f"  {strat:>12}  pass@1={p['pass_at_1_mean']:.3f}  "
              f"tok≈{p['tokens_mean']:.1f}  "
              f"events={p['events_mean']:.2f}")
    print(f"  wall: {overall:.1f}s")

    payload = {
        "config": vars(args),
        "per_distractor": per_distractor,
        "pooled_summary": pooled_summary,
        "wall_seconds": round(overall, 2),
    }
    if client is not None and hasattr(client, "stats"):
        payload["llm_client_stats"] = {
            "prompt_tokens": client.stats.prompt_tokens,
            "output_tokens": client.stats.output_tokens,
            "n_generate_calls": client.stats.n_generate_calls,
            "total_wall": round(client.stats.total_wall, 2),
        }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
