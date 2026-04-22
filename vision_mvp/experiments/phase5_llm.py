"""Phase 5 experiments — real LLM agents using the vision stack.

Run qwen2.5:0.5b agents (local via Ollama) on a small battery of factual
questions. Compare naive-broadcast routing against the vision stack.

We report, for each (question, N, protocol):
  - Accuracy of final answers vs ground truth
  - Agreement (plurality share) — 1.0 = unanimous
  - Total LLM tokens used (prompt + output + embeddings)
  - Number of LLM calls
  - Wall-clock time

Since real LLM calls cost real wall time, we keep N modest (up to ~15-20)
and rounds small (2-3). The point is qualitative: does vision-routing
produce equivalent accuracy with fewer tokens?
"""

from __future__ import annotations
import sys, os, json, time, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from vision_mvp.protocols.llm_protocols import run_llm_naive, run_llm_vision
from vision_mvp.tasks.llm_consensus import DEFAULT_QUESTIONS


def run_case(q, n_agents: int, rounds: int, model: str):
    """Run both protocols on one (question, N) cell."""
    __import__("sys").stdout.flush(); print(f"\n  [naive]   N={n_agents}  Q={q.question[:50]!r}")
    t0 = time.time()
    rn = run_llm_naive(q, n_agents=n_agents, rounds=rounds, model=model)
    wn = time.time() - t0
    __import__("sys").stdout.flush(); print(f"    acc={rn.accuracy:.2f}  agree={rn.agreement:.2f}  "
          f"tokens={rn.llm_stats.total_tokens()}  "
          f"calls={rn.llm_stats.n_generate_calls}g+{rn.llm_stats.n_embed_calls}e  "
          f"wall={wn:.1f}s")

    __import__("sys").stdout.flush(); print(f"  [vision]  N={n_agents}  Q={q.question[:50]!r}")
    t0 = time.time()
    rv = run_llm_vision(q, n_agents=n_agents, rounds=rounds, model=model)
    wv = time.time() - t0
    __import__("sys").stdout.flush(); print(f"    acc={rv.accuracy:.2f}  agree={rv.agreement:.2f}  "
          f"tokens={rv.llm_stats.total_tokens()}  "
          f"calls={rv.llm_stats.n_generate_calls}g+{rv.llm_stats.n_embed_calls}e  "
          f"wall={wv:.1f}s")

    return {
        "question": q.question,
        "ground_truth": q.ground_truth,
        "n_agents": n_agents,
        "rounds": rounds,
        "naive": {
            "accuracy": rn.accuracy, "agreement": rn.agreement,
            "total_tokens": rn.llm_stats.total_tokens(),
            "prompt_tokens": rn.llm_stats.prompt_tokens,
            "output_tokens": rn.llm_stats.output_tokens,
            "embed_tokens": rn.llm_stats.embed_tokens,
            "n_generate": rn.llm_stats.n_generate_calls,
            "n_embed": rn.llm_stats.n_embed_calls,
            "wall_s": round(wn, 2),
            "per_round_accuracy": rn.per_round_accuracy,
        },
        "vision": {
            "accuracy": rv.accuracy, "agreement": rv.agreement,
            "total_tokens": rv.llm_stats.total_tokens(),
            "prompt_tokens": rv.llm_stats.prompt_tokens,
            "output_tokens": rv.llm_stats.output_tokens,
            "embed_tokens": rv.llm_stats.embed_tokens,
            "n_generate": rv.llm_stats.n_generate_calls,
            "n_embed": rv.llm_stats.n_embed_calls,
            "wall_s": round(wv, 2),
            "per_round_accuracy": rv.per_round_accuracy,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:0.5b")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--n-values", type=int, nargs="+", default=[5, 10])
    ap.add_argument("--out", default=None)
    ap.add_argument("--question-idx", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()

    results = []
    for qi in args.question_idx:
        q = DEFAULT_QUESTIONS[qi]
        __import__("sys").stdout.flush(); print(f"\n{'='*70}\nQuestion #{qi}: {q.question}\n"
              f"Ground truth: {q.ground_truth}\n{'='*70}")
        for n in args.n_values:
            row = run_case(q, n_agents=n, rounds=args.rounds, model=args.model)
            row["question_idx"] = qi
            results.append(row)

    # Summary
    __import__("sys").stdout.flush(); print("\n" + "=" * 78)
    __import__("sys").stdout.flush(); print("SUMMARY")
    __import__("sys").stdout.flush(); print("=" * 78)
    hdr = f"{'Q':>2} | {'N':>3} | {'R':>2} | {'protocol':>8} | {'acc':>5} | {'agree':>5} | {'tokens':>7} | {'calls':>7}"
    __import__("sys").stdout.flush(); print(hdr)
    __import__("sys").stdout.flush(); print("-" * len(hdr))
    for r in results:
        for p in ("naive", "vision"):
            d = r[p]
            __import__("sys").stdout.flush(); print(f"{r['question_idx']:>2} | {r['n_agents']:>3} | {r['rounds']:>2} | "
                  f"{p:>8} | {d['accuracy']:>5.2f} | {d['agreement']:>5.2f} | "
                  f"{d['total_tokens']:>7} | "
                  f"{d['n_generate']:>3}g+{d['n_embed']:>3}e")

    # Aggregate ratios
    total_naive = sum(r["naive"]["total_tokens"] for r in results)
    total_vision = sum(r["vision"]["total_tokens"] for r in results)
    if total_vision > 0:
        __import__("sys").stdout.flush(); print(f"\nOverall token ratio (naive / vision): {total_naive/total_vision:.2f}x")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        __import__("sys").stdout.flush(); print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
