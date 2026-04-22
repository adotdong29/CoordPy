"""Example 5 — 100 LLM agents review real code, produce a CEO-ready report.

This is the 'actual task' demo. The team reviews a Python function with an
SQL-injection vulnerability and outputs a structured security report.

Requires a local Ollama with a code-capable model (default qwen2.5-coder:7b):
    ollama pull qwen2.5-coder:7b
    ollama serve &

Run:
    python3 examples/05_real_code_review.py
    python3 examples/05_real_code_review.py --n 50 --task race
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import argparse

from vision_mvp.core.llm_team import LLMTeam
from vision_mvp.tasks.code_review import (
    SQL_INJECTION, RACE_CONDITION, MEMORY_LEAK, assign_reviewer_personas,
)


TASKS = {"sql": SQL_INJECTION, "race": RACE_CONDITION, "memory": MEMORY_LEAK}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--sample", type=int, default=10)
    ap.add_argument("--task", default="sql", choices=list(TASKS))
    ap.add_argument("--model", default="qwen2.5-coder:7b")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    task = TASKS[args.task]
    question = (
        f"{task.description}\n\nCode:\n```python\n{task.code.strip()}\n```\n\n"
        "In one short sentence, name the single most important issue."
    )

    personas = assign_reviewer_personas(args.n)
    team = LLMTeam(n_agents=args.n, personas=personas, question=question,
                   model=args.model, seed=args.seed)

    print(f"\n{args.n} AI reviewers examining code. "
          f"Workspace = {team.workspace.capacity()} agents speak per round.\n",
          flush=True)

    t0 = time.time()
    team.initialize(progress_cb=lambda m: None)   # quiet init
    print(f"  init  done in {time.time()-t0:.0f}s "
          f"({team.client.stats.n_generate_calls} LLM calls)", flush=True)

    for r in range(args.rounds):
        t0 = time.time()
        info = team.step()
        print(f"  round {r+1} done in {time.time()-t0:.0f}s — "
              f"consensus: {info['consensus_text']!r}", flush=True)

    # Synthesis
    framing = (
        "You are aggregating expert code reviews into one final report. "
        "Produce EXACTLY three sections, each one sentence:\n"
        "  CRITICAL ISSUE:\n"
        "  MINOR ISSUES:\n"
        "  RECOMMENDATION:"
    )
    t0 = time.time()
    final = team.synthesize(framing, max_tokens=250)
    print(f"  synthesis done in {time.time()-t0:.0f}s\n", flush=True)

    print("=" * 70)
    print("CODE UNDER REVIEW")
    print("=" * 70)
    print(task.code)
    print("=" * 70)
    print(f"FINAL REPORT (from {args.n} reviewer agents)")
    print("=" * 70)
    print(final)
    print("=" * 70)
    print(f"\nstats: {team.stats()}\n")
    scores = task.scores(final)
    print(f"Critical issue correctly identified: {scores['critical_found']}")


if __name__ == "__main__":
    main()
