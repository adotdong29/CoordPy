"""Example 4 — real LLM agents (requires local Ollama).

Needs a running Ollama with qwen2.5:0.5b (or change the model argument).

    ollama pull qwen2.5:0.5b
    ollama serve &

Then:
    python3 examples/04_local_llm.py

Both naive-broadcast and vision-stack protocols are run on the same
classification question. Vision typically saves 20–40% of tokens with no
accuracy loss — and that gap grows with N.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vision_mvp.tasks.llm_consensus import DEFAULT_QUESTIONS
from vision_mvp.protocols.llm_protocols import run_llm_naive, run_llm_vision


def main(n_agents: int = 10, model: str = "qwen2.5:0.5b"):
    q = DEFAULT_QUESTIONS[0]
    print(f"Question: {q.question}\nGround truth: {q.ground_truth}\n")

    print("Running naive-broadcast protocol …")
    rn = run_llm_naive(q, n_agents=n_agents, rounds=2, model=model)
    print(f"  accuracy: {rn.accuracy:.2f}  agreement: {rn.agreement:.2f}  "
          f"tokens: {rn.llm_stats.total_tokens()}")

    print("Running vision-stack protocol …")
    rv = run_llm_vision(q, n_agents=n_agents, rounds=2, model=model)
    print(f"  accuracy: {rv.accuracy:.2f}  agreement: {rv.agreement:.2f}  "
          f"tokens: {rv.llm_stats.total_tokens()}")

    if rn.llm_stats.total_tokens() > 0:
        ratio = rn.llm_stats.total_tokens() / max(rv.llm_stats.total_tokens(), 1)
        print(f"\nNaive / vision token ratio: {ratio:.2f}× "
              f"({100 * (1 - 1/ratio):.1f}% saved by vision)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--model", default="qwen2.5:0.5b")
    args = ap.parse_args()
    main(n_agents=args.n, model=args.model)
