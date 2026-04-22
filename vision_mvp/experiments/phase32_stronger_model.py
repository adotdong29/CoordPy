"""Phase 32 Part C — stronger-model spot check on the typed-handoff
substrate.

The Phase-31 LLM-in-loop headline was transcription-bounded at 40 % on
``qwen2.5:0.5b``. This script runs a stronger local model
(``qwen2.5-coder:7b`` by default) against the Phase-31 incident-triage
and Phase-32 compliance-review harnesses to show whether the substrate
slice accuracy rises toward the mock ceiling (Theorem P31-4 upper
bound) when the model's transcription fidelity is no longer the
bottleneck.

Disciplined: one seed, one small distractor count (k=6 by default —
the substrate's bounded-context claim is flat in k, so k=6 suffices
to measure whether the model can transcribe the substrate cue). We
spot-check the small-k regime where naive is *also* viable under the
model's context budget, so the substrate vs naive comparison is
apples-to-apples.

Reproducible command:

    python3 -m vision_mvp.experiments.phase32_stronger_model \\
        --model qwen2.5-coder:7b --seeds 32 \\
        --distractor-counts 6 \\
        --out vision_mvp/results_phase32_llm_7b_spot.json
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
from vision_mvp.tasks.incident_triage import (
    MockIncidentAuditor,
    build_scenario_bank as build_incident_bank,
    run_incident_loop,
    ALL_STRATEGIES as INCIDENT_STRATEGIES,
)
from vision_mvp.tasks.compliance_review import (
    MockComplianceAuditor,
    build_scenario_bank as build_compliance_bank,
    run_compliance_loop,
    ALL_STRATEGIES as COMPLIANCE_STRATEGIES,
)


def _make(model: str, mock: bool,
           mock_factory: Callable[[], object],
           max_answer_tokens: int = 80,
           ) -> tuple[Callable[[str], str], object]:
    if mock:
        m = mock_factory()
        return m, m
    client = LLMClient(model=model, timeout=600.0)

    def _call(prompt: str) -> str:
        return client.generate(prompt, max_tokens=max_answer_tokens,
                                temperature=0.0)
    return _call, client.stats


def _pooled_dict(pooled: dict) -> dict:
    return {k: v for k, v in pooled.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5-coder:7b")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--seeds", nargs="+", type=int, default=[32])
    ap.add_argument("--distractor-counts", nargs="+", type=int,
                      default=[6])
    ap.add_argument("--max-events-in-prompt", type=int, default=200)
    ap.add_argument("--inbox-capacity", type=int, default=64)
    ap.add_argument("--max-answer-tokens", type=int, default=80)
    ap.add_argument("--strategies", nargs="+",
                      default=["naive", "substrate", "substrate_wrap"])
    ap.add_argument("--domains", nargs="+",
                      default=["incident", "compliance"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    overall_start = time.time()
    results: dict = {"config": {
        "model": args.model, "mock": args.mock,
        "seeds": args.seeds,
        "distractor_counts": args.distractor_counts,
        "max_events_in_prompt": args.max_events_in_prompt,
        "inbox_capacity": args.inbox_capacity,
        "strategies": args.strategies,
        "domains": args.domains,
        "max_answer_tokens": args.max_answer_tokens,
    }, "domains": {}}

    for domain in args.domains:
        print(f"\n[phase32/C] domain={domain}  model={args.model}  "
              f"mock={args.mock}", flush=True)
        if domain == "incident":
            aud, stats = _make(args.model, args.mock,
                                lambda: MockIncidentAuditor(),
                                max_answer_tokens=args.max_answer_tokens)
        else:
            aud, stats = _make(args.model, args.mock,
                                lambda: MockComplianceAuditor(),
                                max_answer_tokens=args.max_answer_tokens)
        per_config: list[dict] = []
        for k in args.distractor_counts:
            for seed in args.seeds:
                print(f"    k={k} seed={seed}", flush=True)
                if domain == "incident":
                    bank = build_incident_bank(
                        seed=seed, distractors_per_role=k)
                    rep = run_incident_loop(
                        bank, aud, strategies=tuple(args.strategies),
                        seed=seed,
                        max_events_in_prompt=args.max_events_in_prompt,
                        inbox_capacity=args.inbox_capacity)
                else:
                    bank = build_compliance_bank(
                        seed=seed, distractors_per_role=k)
                    rep = run_compliance_loop(
                        bank, aud, strategies=tuple(args.strategies),
                        seed=seed,
                        max_docs_in_prompt=args.max_events_in_prompt,
                        inbox_capacity=args.inbox_capacity)
                pooled = rep.pooled()
                for s in args.strategies:
                    p = pooled.get(s, {})
                    if not p:
                        continue
                    print(
                        f"      {s:>18}  acc_full={p['accuracy_full']:.3f}"
                        f"  tok={p['mean_prompt_tokens']:>6.0f}  "
                        f"fhist={p['failure_hist']}",
                        flush=True)
                per_config.append({
                    "distractors_per_role": k,
                    "seed": seed, "report": rep.as_dict(),
                })
        llm_stats = None
        if hasattr(stats, "total_tokens"):
            llm_stats = {
                "prompt_tokens": stats.prompt_tokens,
                "output_tokens": stats.output_tokens,
                "n_generate_calls": stats.n_generate_calls,
                "total_wall": round(stats.total_wall, 2),
            }
            print(f"    llm({domain}): calls={stats.n_generate_calls} "
                  f"prompt_toks={stats.prompt_tokens} "
                  f"wall={stats.total_wall:.1f}s")
        elif hasattr(stats, "n_calls"):
            llm_stats = {
                "n_calls": stats.n_calls,
                "total_prompt_chars": stats.total_prompt_chars,
            }
            print(f"    mock({domain}): calls={stats.n_calls}")
        results["domains"][domain] = {
            "per_config": per_config,
            "llm_stats": llm_stats,
        }

    overall = time.time() - overall_start
    results["wall_seconds"] = round(overall, 2)
    print(f"\n[phase32/C] overall wall = {overall:.1f}s", flush=True)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
