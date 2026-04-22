"""Phase 8: GENUINE collaboration.

Each of N agents sees ONE chunk of a document. No agent alone can see the
whole thing. The team must answer cross-chunk questions (the top-3 systemic
risks of the company the document is about).

This is fundamentally different from Phase 6/7, where every agent already
knew the answer. Here, every agent has partial information, and the team
must pool it to answer.

Three runs are compared:
  1. **ISOLATED single agent**: the same LLM, sees only 1 chunk, tries to
     answer the full question. Expected to miss most risks.
  2. **DISTRIBUTED team with naive-read**: one single agent reads the ENTIRE
     document and answers. Expected to score best; shows what an unbounded
     context could in principle do — the 'oracle' baseline.
  3. **CASR team**: N agents, each with 1 chunk, collaborating via the
     vision stack. Expected to match or approach the oracle — with a
     tiny fraction of its wall cost at scale.

Run:
    python -m vision_mvp.experiments.phase8_distributed --n 16 --model qwen2.5-coder:7b
"""

from __future__ import annotations
import sys, os, time, json, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from vision_mvp.core.llm_client import LLMClient
from vision_mvp.core.llm_team import LLMTeam
from vision_mvp.tasks.distributed_summary import DistributedSummaryTask
from vision_mvp.tasks.llm_consensus import assign_personas


def progress(msg: str) -> None:
    print(msg, flush=True)


def run_isolated(task: DistributedSummaryTask, chunks: list[str],
                 model: str, agent_idx: int = 0) -> dict:
    """One agent sees ONLY one chunk, asked the global question."""
    client = LLMClient(model=model)
    chunk = chunks[agent_idx]
    prompt = (
        "You have access ONLY to this chunk of a larger document:\n"
        f"---\n{chunk}\n---\n\n"
        f"Question: {task.question}\n\n"
        "Answer based only on what you can see in the chunk:"
    )
    t0 = time.time()
    answer = client.generate(prompt, max_tokens=200, temperature=0.2)
    wall = time.time() - t0
    score = task.score(answer)
    return {"mode": "isolated", "agent_idx": agent_idx,
            "answer": answer, "score": score,
            "wall_s": wall, "tokens": client.stats.total_tokens()}


def run_oracle(task: DistributedSummaryTask, model: str) -> dict:
    """A single-agent 'oracle' with the ENTIRE document in its context.

    This is what a big-context model would do: ingest everything, answer.
    The token cost is linear in document length (not team size), but so
    is wall time — for a 100k-word doc this is unworkable.
    """
    client = LLMClient(model=model)
    prompt = (
        "You are a senior risk analyst reviewing a company's internal Q3 "
        "incident review. Read the whole document and answer:\n\n"
        f"---\n{task.document}\n---\n\n"
        f"Question: {task.question}"
    )
    t0 = time.time()
    answer = client.generate(prompt, max_tokens=350, temperature=0.2)
    wall = time.time() - t0
    score = task.score(answer)
    return {"mode": "oracle", "answer": answer, "score": score,
            "wall_s": wall, "tokens": client.stats.total_tokens()}


def run_team(task: DistributedSummaryTask, chunks: list[str],
             model: str, rounds: int, sample: int, seed: int) -> dict:
    """CASR team: each of N agents sees one chunk, team synthesizes."""
    n = len(chunks)
    personas = assign_personas(n)
    team = LLMTeam(
        n_agents=n,
        personas=personas,
        question=task.question,
        model=model,
        seed=seed,
        per_agent_context=chunks,
    )
    progress(f"[team init] {n} agents, each with their own chunk")
    t0 = time.time()
    team.initialize(progress_cb=progress, max_tokens=100)
    progress(f"  init wall: {time.time()-t0:.1f}s "
             f"({team.client.stats.n_generate_calls} generate, "
             f"{team.client.stats.n_embed_calls} embed)")

    for r in range(rounds):
        progress(f"[team round {r+1}]")
        t0 = time.time()
        info = team.step()
        progress(f"  wall {time.time()-t0:.1f}s  "
                 f"admitted: {info['admitted']}  "
                 f"consensus: {info['consensus_text'][:80]!r}")

    # Sample final answers
    progress(f"[team final-sample] {sample} agents")
    team.finalize_sample(sample_size=sample)

    # Synthesis: broad mode — see every chunk's observation, not just
    # top-k near consensus. Essential for cross-chunk pattern detection.
    framing = (
        "You are the lead risk analyst synthesizing partial views from "
        "team members. Each team member has seen ONLY a small chunk of "
        "the source document. They each report what their chunk says. "
        "YOUR job is to spot CROSS-CHUNK PATTERNS — if multiple team "
        "members mention the same vendor, same problem type, or same "
        "failure mode, that is a SYSTEMIC risk.\n\n"
        "Produce exactly 3 top systemic risks. For each, list the team "
        "members / incidents that support it. Prefer risks that recur "
        "across multiple chunks over risks from a single chunk."
    )
    progress("[team synthesize — broad mode, all chunks]")
    t0 = time.time()
    final = team.synthesize(framing, max_tokens=500, include_all_chunks=True)
    progress(f"  synth wall: {time.time()-t0:.1f}s")

    score = task.score(final)
    stats = team.stats()
    return {
        "mode": "team",
        "synthesis": final,
        "score": score,
        "stats": stats,
        "consensus_per_round": team.per_round_consensus_text,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16,
                    help="number of agents / chunks")
    ap.add_argument("--model", default="qwen2.5-coder:7b")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--sample", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--skip-oracle", action="store_true",
                    help="skip the oracle run (useful if the doc is very long)")
    ap.add_argument("--skip-isolated", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    task = DistributedSummaryTask()
    chunks = task.make_chunks(args.n)
    print(f"Document: {len(task.document.split())} words across {args.n} chunks", flush=True)
    print(f"Expected systemic risks: {list(task.systemic_risks.keys())}", flush=True)
    print(f"Model: {args.model}\n", flush=True)

    results = {}

    if not args.skip_isolated:
        print("=" * 78)
        print("RUN 1: isolated single agent (sees only one chunk)")
        print("=" * 78, flush=True)
        iso = run_isolated(task, chunks, args.model, agent_idx=0)
        print(f"\nAnswer:\n{iso['answer']}\n")
        print(f"Risks identified: {iso['score']['n_risks_identified']}/3  "
              f"(wall {iso['wall_s']:.1f}s, tokens {iso['tokens']})",
              flush=True)
        results["isolated"] = iso

    if not args.skip_oracle:
        print("\n" + "=" * 78)
        print("RUN 2: single-agent oracle (reads ENTIRE document)")
        print("=" * 78, flush=True)
        orc = run_oracle(task, args.model)
        print(f"\nAnswer:\n{orc['answer']}\n")
        print(f"Risks identified: {orc['score']['n_risks_identified']}/3  "
              f"(wall {orc['wall_s']:.1f}s, tokens {orc['tokens']})",
              flush=True)
        results["oracle"] = orc

    print("\n" + "=" * 78)
    print(f"RUN 3: CASR team (N={args.n} agents, each with ONE chunk)")
    print("=" * 78, flush=True)
    team = run_team(task, chunks, args.model, rounds=args.rounds,
                    sample=args.sample, seed=args.seed)
    print(f"\nTeam synthesis:\n{team['synthesis']}\n")
    print(f"Risks identified: {team['score']['n_risks_identified']}/3", flush=True)
    print(f"stats: {team['stats']}", flush=True)
    results["team"] = team

    # Scoreboard
    print("\n" + "=" * 78)
    print("SCOREBOARD")
    print("=" * 78, flush=True)
    print(f"{'mode':>12} | {'risks_found':>11} | {'tokens':>8} | {'wall_s':>6}")
    if "isolated" in results:
        r = results["isolated"]
        print(f"{'isolated':>12} | {r['score']['n_risks_identified']:>11}/3 | "
              f"{r['tokens']:>8} | {r['wall_s']:>6.1f}")
    if "oracle" in results:
        r = results["oracle"]
        print(f"{'oracle':>12} | {r['score']['n_risks_identified']:>11}/3 | "
              f"{r['tokens']:>8} | {r['wall_s']:>6.1f}")
    r = results["team"]
    print(f"{'casr_team':>12} | {r['score']['n_risks_identified']:>11}/3 | "
          f"{r['stats']['llm_total_tokens']:>8} | {r['stats']['wall_llm_seconds']:>6.1f}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nWrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
