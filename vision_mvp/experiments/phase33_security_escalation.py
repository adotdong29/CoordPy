"""Phase 33 Part C — third non-code domain benchmark.

Drives the security-audit-escalation harness
(``vision_mvp.tasks.security_escalation``) across the five-scenario
catalogue under the four delivery strategies (naive / routing /
substrate / substrate_wrap). The benchmark is the programme's third
non-code task-scale benchmark after Phase 31 (incident triage) and
Phase 32 (compliance review). Three distinct domains with the same
substrate primitive give us a K = 3 empirical data point on
Conjecture C32-4 (role-lattice stability across domains).

Reproducible commands:

    # Mock, all scenarios, 2-seed distractor sweep.
    python3 -m vision_mvp.experiments.phase33_security_escalation --mock \\
        --distractor-counts 6 20 60 120 --seeds 33 34 \\
        --out vision_mvp/results_phase33_security_mock.json

    # Real LLM via Ollama (optional; costs real time).
    python3 -m vision_mvp.experiments.phase33_security_escalation \\
        --model qwen2.5-coder:7b --distractor-counts 6 \\
        --out vision_mvp/results_phase33_security_llm_7b.json
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
from vision_mvp.tasks.security_escalation import (
    ALL_STRATEGIES, MockSecurityAuditor,
    build_scenario_bank, run_security_loop,
)


def _make_auditor(model: str, mock: bool = False,
                   max_answer_tokens: int = 96
                   ) -> tuple[Callable[[str], str], object]:
    if mock:
        m = MockSecurityAuditor()
        return m, m
    client = LLMClient(model=model, timeout=600.0)

    def _call(prompt: str) -> str:
        return client.generate(prompt, max_tokens=max_answer_tokens,
                                temperature=0.0)
    return _call, client.stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:0.5b")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--distractor-counts", nargs="+", type=int,
                      default=[20])
    ap.add_argument("--seeds", nargs="+", type=int, default=[33])
    ap.add_argument("--max-events-in-prompt", type=int, default=200)
    ap.add_argument("--inbox-capacity", type=int, default=64)
    ap.add_argument("--strategies", nargs="+",
                      default=list(ALL_STRATEGIES))
    ap.add_argument("--max-answer-tokens", type=int, default=96)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    auditor, stats = _make_auditor(
        args.model, mock=args.mock,
        max_answer_tokens=args.max_answer_tokens)

    overall_start = time.time()
    per_config: list[dict] = []

    for k in args.distractor_counts:
        for seed in args.seeds:
            print(f"\n[phase33/C] distractors_per_role={k} seed={seed} "
                  f"mock={args.mock}", flush=True)
            bank = build_scenario_bank(
                seed=seed, distractors_per_role=k)
            print(f"  n_scenarios={len(bank)}", flush=True)
            rep = run_security_loop(
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
                    f"sev={p['accuracy_severity']:.3f}  "
                    f"cls={p['accuracy_classification']:.3f}  "
                    f"tok={p['mean_prompt_tokens']:>6.0f}  "
                    f"rel={p['mean_aggregator_relevance_fraction']:.3f}  "
                    f"recall={p['mean_handoff_recall']:.3f}  "
                    f"prec={p['mean_handoff_precision']:.3f}  "
                    f"trunc={p['truncated_count']}  "
                    f"fhist={p['failure_hist']}",
                    flush=True)
            per_config.append({
                "distractors_per_role": k,
                "seed": seed,
                "report": rep.as_dict(),
            })

    overall = time.time() - overall_start
    print(f"\n[phase33/C] overall wall = {overall:.1f}s", flush=True)

    by_k_strategy: dict[tuple[int, str], list[dict]] = {}
    for entry in per_config:
        for strat, row in entry["report"]["pooled"].items():
            key = (entry["distractors_per_role"], strat)
            by_k_strategy.setdefault(key, []).append(row)
    pooled_out: dict[str, dict] = {}
    for (k, strat), rows in by_k_strategy.items():
        n = len(rows)
        mean_acc = sum(r["accuracy_full"] for r in rows) / n
        mean_sev = sum(r["accuracy_severity"] for r in rows) / n
        mean_cls = sum(r["accuracy_classification"] for r in rows) / n
        mean_tok = sum(r["mean_prompt_tokens"] for r in rows) / n
        mean_rel = sum(r["mean_aggregator_relevance_fraction"]
                        for r in rows) / n
        mean_recall = sum(r["mean_handoff_recall"] for r in rows) / n
        mean_prec = sum(r["mean_handoff_precision"] for r in rows) / n
        trunc = sum(r["truncated_count"] for r in rows)
        fhist: dict[str, int] = {}
        for r in rows:
            for kk, vv in r["failure_hist"].items():
                fhist[kk] = fhist.get(kk, 0) + vv
        pooled_out[f"k={k}:{strat}"] = {
            "n_reports": n,
            "accuracy_full_mean": round(mean_acc, 4),
            "accuracy_severity_mean": round(mean_sev, 4),
            "accuracy_classification_mean": round(mean_cls, 4),
            "mean_prompt_tokens": round(mean_tok, 2),
            "mean_aggregator_relevance_fraction": round(mean_rel, 4),
            "mean_handoff_recall": round(mean_recall, 4),
            "mean_handoff_precision": round(mean_prec, 4),
            "truncated_count_total": trunc,
            "failure_hist": fhist,
        }
    print()
    print("=" * 80)
    print("PHASE 33 / PART C POOLED — per (distractor_count, strategy)")
    print("=" * 80)
    for key in sorted(pooled_out.keys()):
        row = pooled_out[key]
        print(f"  {key:>35}  acc={row['accuracy_full_mean']:.3f}  "
              f"sev={row['accuracy_severity_mean']:.3f}  "
              f"cls={row['accuracy_classification_mean']:.3f}  "
              f"tok={row['mean_prompt_tokens']:>6.0f}  "
              f"recall={row['mean_handoff_recall']:.3f}  "
              f"prec={row['mean_handoff_precision']:.3f}  "
              f"trunc={row['truncated_count_total']}  "
              f"fhist={row['failure_hist']}")

    llm_stats = None
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
