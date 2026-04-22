"""Phase 39 — frontier-model bounded sweep on the team substrate.

Phase 32/C ran a single 7B spot check on incident triage and
compliance review. Phase 33 added a single 7B extractor run on
all three non-code domains. Phase 39 broadens the spot check
into a *bounded but cross-family* sweep so the substrate's
correctness-preservation claim has more than one model behind
it.

Discipline: bounded breadth, not unbounded scale.
  * 2–4 models, mixed families if available.
  * 1–2 seeds, 1 representative k per domain.
  * 1 representative non-code domain (incident triage)
    plus the contested bank if requested.
  * The pooled report makes naive vs substrate
    vs substrate_wrap comparable across models.

The script always runs the mock auditor first as a *strategy
ceiling reference*, so a measured shortfall is attributable
to LLM transcription / reasoning fidelity, not to any
substrate gap.

Reproducible commands:

    # Mock reference (sub-second).
    python3 -m vision_mvp.experiments.phase39_frontier_substrate \\
        --mock --domains incident --distractor-counts 6 \\
        --out vision_mvp/results_phase39_frontier_mock.json

    # Real frontier sweep (2 models on 1 domain at k=6).
    python3 -m vision_mvp.experiments.phase39_frontier_substrate \\
        --models llama3.1:8b gemma2:9b qwen2.5-coder:7b \\
        --domains incident --distractor-counts 6 --seeds 31 \\
        --out vision_mvp/results_phase39_frontier_substrate.json
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

from vision_mvp.tasks.swe_loop_harness import MockAnswerLLM


# Each domain registers a (build_bank, run_loop, ALL_STRATEGIES)
# triple plus a one-line report formatter.
def _domain_incident():
    from vision_mvp.tasks.incident_triage import (
        ALL_STRATEGIES, build_scenario_bank, run_incident_loop,
    )

    def build(seed: int, k: int):
        return build_scenario_bank(seed=seed,
                                     distractors_per_role=k)

    def run(bank, auditor, seed, k, strategies):
        return run_incident_loop(
            bank, auditor, strategies=tuple(strategies),
            seed=seed, max_events_in_prompt=200,
            inbox_capacity=32)

    def fmt(pooled, strat):
        p = pooled.get(strat, {})
        return (f"acc_full={p.get('accuracy_full', 0):.3f}  "
                 f"rc={p.get('accuracy_root_cause', 0):.3f}  "
                 f"tok≈{p.get('mean_prompt_tokens', 0):.0f}")

    return ("incident", build, run, ALL_STRATEGIES, fmt)


def _domain_compliance():
    from vision_mvp.tasks.compliance_review import (
        ALL_STRATEGIES, build_compliance_scenario_bank,
        run_compliance_loop,
    )

    def build(seed: int, k: int):
        return build_compliance_scenario_bank(
            seed=seed, distractors_per_role=k)

    def run(bank, auditor, seed, k, strategies):
        return run_compliance_loop(
            bank, auditor, strategies=tuple(strategies),
            seed=seed, max_events_in_prompt=200,
            inbox_capacity=32)

    def fmt(pooled, strat):
        p = pooled.get(strat, {})
        return (f"verdict={p.get('accuracy_verdict', 0):.3f}  "
                 f"flags={p.get('accuracy_flags', 0):.3f}  "
                 f"tok≈{p.get('mean_prompt_tokens', 0):.0f}")

    return ("compliance", build, run, ALL_STRATEGIES, fmt)


_DOMAIN_REGISTRY = {
    "incident": _domain_incident,
    "compliance": _domain_compliance,
}


def _make_auditor(model: str | None, mock: bool, max_tokens: int):
    if mock:
        m = MockAnswerLLM()
        return m, m
    from vision_mvp.core.llm_client import LLMClient
    client = LLMClient(model=model, timeout=300.0)

    def _call(prompt: str) -> str:
        return client.generate(prompt, max_tokens=max_tokens,
                                temperature=0.0)
    return _call, client


def _run_one(domain_key: str,
              auditor_factory: Callable,
              models: list[str | None],
              distractor_counts: list[int],
              seeds: list[int],
              max_tokens: int) -> list[dict]:
    """Run one domain across (models × distractor_counts × seeds).

    Returns one row per (model, k, seed) cell. ``models`` may
    contain ``None`` for the mock baseline.
    """
    domain_key, build, run_loop, ALL_STRATEGIES, fmt = \
        _DOMAIN_REGISTRY[domain_key]()
    rows: list[dict] = []
    for model in models:
        is_mock = model is None
        auditor, client = _make_auditor(
            model=model if not is_mock else None,
            mock=is_mock, max_tokens=max_tokens)
        for k in distractor_counts:
            for seed in seeds:
                bank = build(seed=seed, k=k)
                t0 = time.time()
                rep = run_loop(bank, auditor, seed=seed, k=k,
                                strategies=ALL_STRATEGIES)
                wall = time.time() - t0
                pooled = rep.pooled()
                row = {
                    "domain": domain_key,
                    "model": model if not is_mock else "mock",
                    "k": k, "seed": seed,
                    "wall_seconds": round(wall, 2),
                    "pooled": pooled,
                }
                if client is not None and hasattr(
                        client, "stats"):
                    row["llm_stats"] = {
                        "prompt_tokens": getattr(
                            client.stats, "prompt_tokens", 0),
                        "output_tokens": getattr(
                            client.stats, "output_tokens", 0),
                        "n_generate_calls": getattr(
                            client.stats, "n_generate_calls", 0),
                    }
                rows.append(row)
                tag = (f"  [{domain_key}] model={model or 'mock':>20} "
                        f"k={k:<3} seed={seed:<3} "
                        f"wall={wall:>6.1f}s")
                print(tag, flush=True)
                for strat in ALL_STRATEGIES:
                    print(f"      {strat:>16}  {fmt(pooled, strat)}",
                          flush=True)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true",
                      help="Run only the mock auditor (no Ollama).")
    ap.add_argument("--models", nargs="+",
                      default=["llama3.1:8b", "gemma2:9b"])
    ap.add_argument("--domains", nargs="+", default=["incident"])
    ap.add_argument("--distractor-counts", nargs="+", type=int,
                      default=[6])
    ap.add_argument("--seeds", nargs="+", type=int, default=[31])
    ap.add_argument("--max-answer-tokens", type=int, default=80)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    models: list[str | None]
    if args.mock:
        models = [None]
    else:
        # Always include the mock as the strategy-ceiling reference.
        models = [None] + list(args.models)

    overall_start = time.time()
    all_rows: list[dict] = []
    for d in args.domains:
        if d not in _DOMAIN_REGISTRY:
            print(f"[phase39-frontier] unknown domain {d!r}; "
                  f"available: {sorted(_DOMAIN_REGISTRY.keys())}")
            continue
        print(f"\n[phase39-frontier] domain={d}", flush=True)
        rows = _run_one(
            domain_key=d, auditor_factory=_make_auditor,
            models=models,
            distractor_counts=args.distractor_counts,
            seeds=args.seeds,
            max_tokens=args.max_answer_tokens)
        all_rows.extend(rows)

    overall = time.time() - overall_start
    payload = {
        "config": {
            "mock": args.mock, "models": args.models,
            "domains": args.domains,
            "distractor_counts": args.distractor_counts,
            "seeds": args.seeds,
        },
        "rows": all_rows,
        "wall_seconds": round(overall, 2),
    }
    print()
    print("=" * 72)
    print("PHASE 39 — frontier-model substrate sweep")
    print("=" * 72)
    print(f"  total wall: {overall:.1f}s   "
          f"n_cells: {len(all_rows)}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
