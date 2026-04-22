"""Phase 6: 1 000 real LLM agents, coordinated on this laptop.

Run locally with Ollama + qwen2.5:0.5b. The key trick is that ONLY
⌈log₂ N⌉ ≈ 10 of the 1000 agents do an LLM call per round; the other
990 carry embeddings and drift through the shared manifold.

Cost budget (qwen2.5:0.5b, M3 Pro):
    init:         (# unique personas)  LLM calls                ≈ 20 × 1 s = 20 s
    per round:    ~10 generate + 1 batch embed                   ≈ 15 s
    final sample: ~30–50 generate calls                          ≈ 45 s
  Total for 5 rounds + final sample:                              ≈ 2–3 minutes

If we tried naive broadcast (every agent sees every other each round)
at 1000 agents × 5 rounds, that's 5000 LLM calls ≈ 2 hours. Can't do it
in a session.

Usage:
    python -m vision_mvp.experiments.phase6_llm_1000
    python -m vision_mvp.experiments.phase6_llm_1000 --n 500 --rounds 3
"""

from __future__ import annotations
import sys, os, json, time, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
from collections import Counter

from vision_mvp.core.llm_team import LLMTeam
from vision_mvp.tasks.llm_consensus import (
    DEFAULT_PERSONAS, DEFAULT_QUESTIONS, assign_personas, LLMQuestion)


def progress(msg: str) -> None:
    print(msg, flush=True)


def run_one(question: LLMQuestion, n: int, rounds: int, sample_size: int,
            model: str, seed: int = 0, **kwargs) -> dict:
    personas = assign_personas(n)
    team = LLMTeam(n_agents=n, personas=personas, question=question.question,
                   model=model, seed=seed,
                   n_parallel_llm=kwargs.get("n_parallel_llm", 1))

    print("=" * 78, flush=True)
    print(f"Coordinating {n} LLM agents on:\n  {question.question}", flush=True)
    print(f"Ground truth: {question.ground_truth}", flush=True)
    print(f"Unique personas: {len(set(personas))}", flush=True)
    print(f"Workspace size (⌈log₂ N⌉): {team.workspace.capacity()}", flush=True)
    print("=" * 78, flush=True)

    # Init
    print("\n[init] generating archetype answers…", flush=True)
    t0 = time.time()
    team.initialize(progress_cb=progress)
    print(f"  archetype init wall: {time.time()-t0:.1f}s,  "
          f"calls so far: {team.client.stats.n_generate_calls}g + "
          f"{team.client.stats.n_embed_calls}e", flush=True)

    # A few unique initial opinions
    unique_init = set(team.text_state)
    print(f"  unique initial opinions: {len(unique_init)}", flush=True)

    # Rounds
    for r in range(rounds):
        print(f"\n[round {r+1}] admitting {team.workspace.capacity()} agents…",
              flush=True)
        t0 = time.time()
        info = team.step(progress_cb=progress)
        print(f"  round wall: {time.time()-t0:.1f}s", flush=True)
        print(f"  admitted agent ids: {info['admitted']}", flush=True)
        print(f"  consensus text: {info['consensus_text']!r}", flush=True)
        print(f"  cumulative LLM tokens: {info['total_llm_tokens']}  "
              f"broadcast tokens: {info['ctx_tokens_broadcast']}", flush=True)

    # Final sample
    print(f"\n[final] polling {sample_size} randomly-sampled agents for their "
          "final answer given the consensus context…", flush=True)
    t0 = time.time()
    final_sample = team.finalize_sample(sample_size=sample_size,
                                        progress_cb=progress)
    print(f"  final-sample wall: {time.time()-t0:.1f}s", flush=True)

    # Tally accuracy from sample
    normalized = [question.normalize(ans) for _, ans in final_sample]
    counts = Counter(normalized)
    majority = counts.most_common(1)[0][0] if counts else ""
    accuracy = sum(1 for v in normalized if v == question.ground_truth) / \
               max(len(normalized), 1)

    # For the remaining 950 agents we use the nearest-neighbor-to-admitted
    # heuristic — each agent's "final opinion" = text of nearest admitted
    # agent by embedding cosine similarity. No LLM call needed.
    nn_texts = team.nearest_neighbor_texts()
    nn_norm = [question.normalize(t) for t in nn_texts]
    nn_majority = Counter(nn_norm).most_common(1)[0][0]
    nn_accuracy = sum(1 for v in nn_norm if v == question.ground_truth) / n

    stats = team.stats()

    # Naive-broadcast extrapolation: what WOULD it have cost?
    avg_prompt = (stats["llm_prompt_tokens"]
                  / max(stats["llm_generate_calls"], 1))
    avg_output = (stats["llm_output_tokens"]
                  / max(stats["llm_generate_calls"], 1))
    naive_gen_calls = n * rounds + sample_size
    naive_avg_prompt = avg_prompt * (n / 10)   # naive context is O(N), ours is O(1)
    naive_tokens_extrap = int(naive_gen_calls * (naive_avg_prompt + avg_output))

    print("\n" + "=" * 78, flush=True)
    print("RESULTS", flush=True)
    print("=" * 78, flush=True)
    print(f"  N = {n}, rounds = {rounds}, sample = {sample_size}", flush=True)
    print(f"  workspace size = {stats['workspace_size']} (= ⌈log₂ N⌉)", flush=True)
    print(f"  manifold dim   = {stats['manifold_dim']}", flush=True)
    print(f"  LLM generate calls = {stats['llm_generate_calls']}", flush=True)
    print(f"  LLM embed calls    = {stats['llm_embed_calls']}", flush=True)
    print(f"  LLM tokens total   = {stats['llm_total_tokens']:,}", flush=True)
    print(f"  broadcast tokens   = {stats['ctx_tokens_broadcast']:,}", flush=True)
    print(f"  wall time          = {stats['wall_llm_seconds']} s", flush=True)
    print(f"\n  Sampled final agents ({sample_size}):", flush=True)
    print(f"    accuracy vs ground truth: {accuracy:.2f}", flush=True)
    print(f"    majority answer: {majority!r}", flush=True)
    print(f"\n  All {n} agents (via nearest-neighbor to admitted):", flush=True)
    print(f"    accuracy vs ground truth: {nn_accuracy:.2f}", flush=True)
    print(f"    majority answer: {nn_majority!r}", flush=True)
    print(f"\n  Naive broadcast extrapolation at this N and rounds:", flush=True)
    print(f"    estimated LLM tokens:  {naive_tokens_extrap:,}", flush=True)
    if stats['llm_total_tokens'] > 0:
        ratio = naive_tokens_extrap / stats['llm_total_tokens']
        print(f"    naive / vision ratio: {ratio:.1f}×", flush=True)

    return {
        "question": question.question,
        "ground_truth": question.ground_truth,
        "n_agents": n,
        "rounds": rounds,
        "sample_size": sample_size,
        "sample_accuracy": accuracy,
        "sample_majority": majority,
        "sample_answers": final_sample,
        "nn_accuracy": nn_accuracy,
        "nn_majority": nn_majority,
        "stats": stats,
        "naive_tokens_extrap": naive_tokens_extrap,
        "per_round_admitted": team.per_round_admitted,
        "per_round_consensus": team.per_round_consensus_text,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000,
                    help="number of agents (default 1000)")
    ap.add_argument("--rounds", type=int, default=5,
                    help="coordination rounds (default 5)")
    ap.add_argument("--sample", type=int, default=30,
                    help="final-answer sample size (default 30)")
    ap.add_argument("--question", type=int, default=0,
                    help="DEFAULT_QUESTIONS index")
    ap.add_argument("--model", default="qwen2.5:0.5b")
    ap.add_argument("--out", default=None,
                    help="dump results to JSON")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--parallel", type=int, default=1,
                    help="concurrent LLM calls per round (1 = serial)")
    args = ap.parse_args()

    q = DEFAULT_QUESTIONS[args.question]
    result = run_one(
        question=q,
        n=args.n,
        rounds=args.rounds,
        sample_size=args.sample,
        model=args.model,
        seed=args.seed,
        n_parallel_llm=args.parallel,
    )
    if args.out:
        # final_sample list of (int, str) tuples — serialize as lists
        result["sample_answers"] = [[i, s] for i, s in result["sample_answers"]]
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nWrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
