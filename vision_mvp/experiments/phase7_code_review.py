"""Phase 7: N real LLM agents perform an actual code review.

Each agent is a specialist reviewer (security, concurrency, perf, …) looking
at a piece of real code with a known critical issue. The team converges on
the critical issue via the CASR protocol, and a final synthesis step produces
a CEO-readable structured review.

Unlike Phase 6 (single-word classification), this tests whether the protocol
preserves *reasoning quality* — not just majority-vote on a trivia question.

Usage:
    python -m vision_mvp.experiments.phase7_code_review
    python -m vision_mvp.experiments.phase7_code_review --n 200 --task sql
    python -m vision_mvp.experiments.phase7_code_review --task race --model llama3.1:8b
"""

from __future__ import annotations
import sys, os, json, time, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from collections import Counter
import numpy as np

from vision_mvp.core.llm_team import LLMTeam
from vision_mvp.tasks.code_review import (
    DEFAULT_TASKS, SQL_INJECTION, RACE_CONDITION, MEMORY_LEAK,
    assign_reviewer_personas, CodeReviewTask,
)


TASK_LOOKUP = {
    "sql": SQL_INJECTION,
    "race": RACE_CONDITION,
    "memory": MEMORY_LEAK,
}


def build_question(task: CodeReviewTask) -> str:
    return (
        f"{task.description}\n\n"
        f"Code:\n```python\n{task.code.strip()}\n```\n\n"
        f"In one short sentence, name the single most important issue."
    )


def progress(msg: str) -> None:
    print(msg, flush=True)


def run(task: CodeReviewTask, n: int, rounds: int, sample: int,
        model: str, seed: int, parallel: int, synthesize: bool) -> dict:
    question = build_question(task)
    personas = assign_reviewer_personas(n)
    team = LLMTeam(n_agents=n, personas=personas, question=question,
                   model=model, seed=seed, n_parallel_llm=parallel)

    print("=" * 78, flush=True)
    print(f"{n} LLM reviewers examining code on a {task.critical_issue.split('—')[0].strip()} issue.", flush=True)
    print(f"Model: {model}", flush=True)
    print(f"Workspace ⌈log₂ N⌉ = {team.workspace.capacity()}", flush=True)
    print("=" * 78, flush=True)

    print("\n[init] seeding archetype analyses…", flush=True)
    t0 = time.time()
    team.initialize(progress_cb=progress)
    print(f"  init wall: {time.time()-t0:.1f}s  "
          f"(generate={team.client.stats.n_generate_calls}, "
          f"embed={team.client.stats.n_embed_calls})", flush=True)

    for r in range(rounds):
        print(f"\n[round {r+1}]", flush=True)
        t0 = time.time()
        info = team.step(progress_cb=progress)
        print(f"  wall: {time.time()-t0:.1f}s", flush=True)
        print(f"  admitted: {info['admitted']}", flush=True)
        print(f"  consensus text: {info['consensus_text']!r}", flush=True)

    # Score per-agent: how many of the sampled/admitted agents flagged the
    # critical issue?
    print(f"\n[final-sample] polling {sample} random agents…", flush=True)
    t0 = time.time()
    final_sample = team.finalize_sample(sample_size=sample, progress_cb=progress)
    print(f"  wall: {time.time()-t0:.1f}s", flush=True)

    sample_scores = [task.scores(ans) for _, ans in final_sample]
    sample_critical = sum(1 for s in sample_scores if s["critical_found"])
    sample_critical_rate = sample_critical / max(len(sample_scores), 1)

    # Nearest-neighbor for all N (cheap, no LLM)
    nn_texts = team.nearest_neighbor_texts()
    nn_scores = [task.scores(t) for t in nn_texts]
    nn_critical = sum(1 for s in nn_scores if s["critical_found"])
    nn_critical_rate = nn_critical / max(len(nn_scores), 1)

    # Synthesis: one final LLM call combines top-k reviewer texts
    synthesis_text = ""
    if synthesize:
        print("\n[synthesis] combining top-k reviewer outputs…", flush=True)
        framing = (
            "You are aggregating multiple expert code reviews into one final "
            "report. Produce EXACTLY three sections, each one sentence:\n"
            "  CRITICAL ISSUE:\n"
            "  MINOR ISSUES:\n"
            "  RECOMMENDATION:"
        )
        t0 = time.time()
        synthesis_text = team.synthesize(framing, max_tokens=250,
                                         top_k_from_admitted=team.workspace.capacity())
        print(f"  synthesis wall: {time.time()-t0:.1f}s", flush=True)
        print("\n" + "-" * 78, flush=True)
        print(synthesis_text, flush=True)
        print("-" * 78, flush=True)

    synthesis_scores = task.scores(synthesis_text) if synthesis_text else None

    stats = team.stats()

    print("\n" + "=" * 78, flush=True)
    print("RESULTS", flush=True)
    print("=" * 78, flush=True)
    print(f"  N = {n}, rounds = {rounds}, sample = {sample}", flush=True)
    print(f"  workspace = {stats['workspace_size']}  manifold_dim = {stats['manifold_dim']}", flush=True)
    print(f"  LLM generate calls = {stats['llm_generate_calls']}", flush=True)
    print(f"  LLM tokens total   = {stats['llm_total_tokens']:,}", flush=True)
    print(f"  wall time          = {stats['wall_llm_seconds']} s", flush=True)
    print(f"\n  Sample ({sample} agents) — critical issue flagged: "
          f"{sample_critical}/{len(sample_scores)} = {sample_critical_rate:.2%}", flush=True)
    print(f"  All {n} agents (nearest-neighbor) — critical issue flagged: "
          f"{nn_critical}/{len(nn_scores)} = {nn_critical_rate:.2%}", flush=True)
    if synthesis_scores:
        print(f"\n  SYNTHESIS critical flagged: "
              f"{'YES' if synthesis_scores['critical_found'] else 'NO'}  "
              f"(minor mentions: {synthesis_scores['n_minor_mentioned']}/{len(task.minor_issues)})",
              flush=True)

    return {
        "task": task.critical_issue,
        "n_agents": n,
        "rounds": rounds,
        "sample_size": sample,
        "sample_critical_rate": sample_critical_rate,
        "nn_critical_rate": nn_critical_rate,
        "synthesis_text": synthesis_text,
        "synthesis_scores": synthesis_scores,
        "per_round_consensus": team.per_round_consensus_text,
        "per_round_admitted": team.per_round_admitted,
        "stats": stats,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--sample", type=int, default=20)
    ap.add_argument("--task", choices=list(TASK_LOOKUP.keys()), default="sql")
    ap.add_argument("--model", default="llama3.1:8b")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--no-synth", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    task = TASK_LOOKUP[args.task]
    result = run(task, n=args.n, rounds=args.rounds, sample=args.sample,
                 model=args.model, seed=args.seed, parallel=args.parallel,
                 synthesize=not args.no_synth)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nWrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
