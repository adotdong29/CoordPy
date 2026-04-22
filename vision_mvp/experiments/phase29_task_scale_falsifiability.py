"""Phase 29 — task-scale causal-relevance falsifiability benchmark.

Runs the Phase-29 harness (``vision_mvp.tasks.task_scale_swe``) across
the four local Phase-23 corpora, computes the aggregator-role causal-
relevance fraction under three delivery strategies (naive / routing /
substrate), and applies the ROADMAP-specified falsifiability gate to
the naive baseline.

The benchmark is **deterministic**: the event stream, task bank, and
oracle are all seeded from the same constant (``--seed``, default 29),
and every strategy is a pure function of the event stream + task +
role. No LLM calls; no retrieval training loop; the substrate path
returns the planner-matched answer verbatim.

Reproduce (default):

    python -m vision_mvp.experiments.phase29_task_scale_falsifiability \\
        --out vision_mvp/results_phase29_taskscale.json

Subset a single corpus (faster CI):

    python -m vision_mvp.experiments.phase29_task_scale_falsifiability \\
        --corpora vision-core \\
        --out vision_mvp/results_phase29_taskscale_core.json

See ``vision_mvp/RESULTS_PHASE29.md`` for the research framing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

from vision_mvp.tasks.python_corpus import PythonCorpus
from vision_mvp.tasks.task_scale_swe import (
    ALL_ROLES,
    decide_falsifiability,
    run_corpus_bench,
)


def _default_corpora() -> list[tuple[str, str]]:
    """Per-corpus (name, root) pairs for Phase-29, all local."""
    repo = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", ".."))
    return [
        ("vision-core",        os.path.join(repo, "vision_mvp/core")),
        ("vision-tasks",       os.path.join(repo, "vision_mvp/tasks")),
        ("vision-tests",       os.path.join(repo, "vision_mvp/tests")),
        ("vision-experiments", os.path.join(repo, "vision_mvp/experiments")),
    ]


def _print_headline(name: str, rep, decision) -> None:
    p = rep.pooled["per_strategy"]
    n_tok_naive = p["naive"]["mean_delivered_tokens_per_role"]
    n_tok_rout = p["routing"]["mean_delivered_tokens_per_role"]
    n_tok_sub = p["substrate"]["mean_delivered_tokens_per_role"]
    print()
    print(f"=== {name} ===")
    print(f"  n_events={rep.n_events}  n_tasks={rep.n_tasks}")
    print(f"  aggregator causal-relevance fraction:")
    print(f"    naive     = {p['naive']['mean_relevance_fraction_aggregator']:.4f}")
    print(f"    routing   = {p['routing']['mean_relevance_fraction_aggregator']:.4f}")
    print(f"    substrate = {p['substrate']['mean_relevance_fraction_aggregator']:.4f}")
    print(f"  aggregator answer correctness:")
    print(f"    naive     = {p['naive']['answer_correct_rate_aggregator']:.4f}")
    print(f"    routing   = {p['routing']['answer_correct_rate_aggregator']:.4f}")
    print(f"    substrate = {p['substrate']['answer_correct_rate_aggregator']:.4f}")
    print(f"  substrate_match_rate = {p['substrate']['substrate_match_rate']:.4f}")
    print(f"  mean delivered tokens per role:")
    head = (f"    {'role':>18} | {'naive':>10} | {'routing':>10} | "
            f"{'substrate':>10} | {'naive/sub':>10}")
    print(head)
    print("    " + "-" * (len(head) - 4))
    for role in ALL_ROLES:
        nt = n_tok_naive.get(role, 0.0)
        rt = n_tok_rout.get(role, 0.0)
        st = n_tok_sub.get(role, 0.0)
        ratio = nt / st if st > 0 else float("inf")
        print(f"    {role:>18} | {nt:>10.1f} | {rt:>10.1f} | "
              f"{st:>10.2f} | {ratio:>10.1f}x")
    print(f"  falsifiability decision: {decision.decision.upper()}")
    print(f"    reasoning: {decision.reasoning}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=29)
    ap.add_argument("--corpora", nargs="*", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-agent-comments", type=int, default=6)
    ap.add_argument("--gate-lower", type=float, default=0.50)
    ap.add_argument("--gate-upper", type=float, default=0.80)
    args = ap.parse_args()

    corpora_cfg = _default_corpora()
    if args.corpora is not None:
        want = set(args.corpora)
        corpora_cfg = [(n, r) for (n, r) in corpora_cfg if n in want]
    if not corpora_cfg:
        print("No runnable corpora.", flush=True)
        return 1

    t0 = time.time()
    corpus_reports: list[dict] = []
    corpus_decisions: list[dict] = []
    pooled_agg_rel_values: list[float] = []
    pooled_tokens: dict[str, dict[str, float]] = {
        s: {role: 0.0 for role in ALL_ROLES}
        for s in ("naive", "routing", "substrate")
    }
    pooled_correct: dict[str, tuple[int, int]] = {
        "naive": (0, 0), "routing": (0, 0), "substrate": (0, 0),
    }
    pooled_substrate_match: tuple[int, int] = (0, 0)
    total_tasks = 0
    total_events = 0

    for name, root in corpora_cfg:
        if not os.path.isdir(root):
            print(f"[{name}] skipping — {root} not a directory", flush=True)
            continue
        print(f"[{name}] building corpus at {root}...", flush=True)
        t_c = time.time()
        c = PythonCorpus(root=root, seed=args.seed)
        c.build()
        print(f"[{name}] corpus built in {time.time() - t_c:.2f}s "
              f"(n_files={c.n_files} n_functions={c.n_functions_total})",
              flush=True)
        rep = run_corpus_bench(name, c, seed=args.seed,
                                n_agent_comments=args.n_agent_comments)
        decision = decide_falsifiability(
            rep.pooled["per_strategy"]["naive"][
                "mean_relevance_fraction_aggregator"],
            gate_lower=args.gate_lower, gate_upper=args.gate_upper,
        )
        _print_headline(name, rep, decision)
        corpus_reports.append(rep.as_dict())
        corpus_decisions.append({
            "corpus_name": name,
            "decision": decision.decision,
            "reasoning": decision.reasoning,
            "naive_relevance": decision.naive_aggregator_relevance,
        })

        # Accumulate for pooled.
        for strat, info in rep.pooled["per_strategy"].items():
            for role in ALL_ROLES:
                pooled_tokens[strat][role] += (
                    info["mean_delivered_tokens_per_role"].get(role, 0.0)
                    * info["n_aggregator_tasks"])
            ok, tot = pooled_correct[strat]
            pooled_correct[strat] = (
                ok + int(info["answer_correct_rate_aggregator"]
                          * info["n_aggregator_tasks"]),
                tot + info["n_aggregator_tasks"])
        ms, ts = pooled_substrate_match
        pooled_substrate_match = (
            ms + int(rep.pooled["per_strategy"]["substrate"]
                      ["substrate_match_rate"]
                      * rep.pooled["per_strategy"]["substrate"]
                      ["n_aggregator_tasks"]),
            ts + rep.pooled["per_strategy"]["substrate"][
                "n_aggregator_tasks"])
        pooled_agg_rel_values.extend(
            [rep.pooled["per_strategy"]["naive"][
                "mean_relevance_fraction_aggregator"]]
            * rep.pooled["per_strategy"]["naive"]["n_aggregator_tasks"]
        )
        total_tasks += rep.n_tasks
        total_events += rep.n_events

    pooled_tokens_mean = {
        strat: {role: round(pooled_tokens[strat][role]
                              / max(1, pooled_correct[strat][1]), 2)
                 for role in ALL_ROLES}
        for strat in pooled_tokens
    }
    pooled_correct_rate = {
        strat: round(ok / max(1, tot), 4)
        for strat, (ok, tot) in pooled_correct.items()
    }
    pooled_substrate_rate = round(
        pooled_substrate_match[0] / max(1, pooled_substrate_match[1]), 4)
    pooled_naive_rel = round(
        sum(pooled_agg_rel_values) / max(1, len(pooled_agg_rel_values)), 4)

    pooled_decision = decide_falsifiability(pooled_naive_rel,
                                              gate_lower=args.gate_lower,
                                              gate_upper=args.gate_upper)

    print()
    print("=" * 72)
    print("PHASE 29 POOLED — across all runnable corpora")
    print("=" * 72)
    print(f"  n_corpora={len(corpus_reports)}  "
          f"n_tasks_total={total_tasks}  n_events_total={total_events}")
    print(f"  pooled naive aggregator relevance "
          f"= {pooled_naive_rel:.4f}")
    print(f"  pooled substrate_match_rate  "
          f"= {pooled_substrate_rate:.4f}")
    print(f"  pooled answer correctness:")
    for strat, rate in pooled_correct_rate.items():
        print(f"    {strat:>10} = {rate:.4f}")
    print(f"  pooled mean delivered tokens per role:")
    head = (f"    {'role':>18} | {'naive':>12} | {'routing':>12} | "
            f"{'substrate':>12} | {'naive/sub':>10}")
    print(head)
    print("    " + "-" * (len(head) - 4))
    for role in ALL_ROLES:
        n = pooled_tokens_mean["naive"][role]
        r = pooled_tokens_mean["routing"][role]
        s = pooled_tokens_mean["substrate"][role]
        ratio = n / s if s > 0 else float("inf")
        print(f"    {role:>18} | {n:>12.1f} | {r:>12.1f} | "
              f"{s:>12.2f} | {ratio:>10.1f}x")
    print()
    print(f"  POOLED FALSIFIABILITY DECISION: "
          f"{pooled_decision.decision.upper()}")
    print(f"  reasoning: {pooled_decision.reasoning}")
    print(f"  wall-time: {time.time() - t0:.2f}s")

    payload = {
        "config": {
            "seed": args.seed,
            "n_agent_comments": args.n_agent_comments,
            "gate_lower": args.gate_lower,
            "gate_upper": args.gate_upper,
        },
        "corpora": corpus_reports,
        "corpus_decisions": corpus_decisions,
        "pooled": {
            "n_corpora": len(corpus_reports),
            "n_tasks": total_tasks,
            "n_events": total_events,
            "naive_aggregator_relevance": pooled_naive_rel,
            "answer_correct_rate": pooled_correct_rate,
            "substrate_match_rate": pooled_substrate_rate,
            "mean_delivered_tokens_per_role": pooled_tokens_mean,
            "decision": pooled_decision.decision,
            "reasoning": pooled_decision.reasoning,
        },
    }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
