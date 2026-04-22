"""Phase 8 variant — map-reduce style (no rounds, just synthesis).

For purely distributed-observation tasks (each agent has independent info
and the answer needs cross-chunk pattern detection), the iterative CASR
protocol is overkill. The simplest form is:

    map:    each agent reads its chunk, emits a one-sentence observation
    reduce: one synthesis LLM call combines all observations

This is the classical map-reduce pattern. No rounds, no consensus voting,
no workspace. Just pool all observations and let the synthesizer see them.

The CASR machinery (manifold, workspace) is valuable when the task needs
*convergent* answers (everyone agreeing on one answer). For *pooling*
partial independent observations, map-reduce is simpler and cheaper.

This experiment runs map-reduce on the same Orion Systems task and
compares against the full-CASR team.

Usage:
    python -m vision_mvp.experiments.phase8_mapreduce --n 16
"""

from __future__ import annotations
import sys, os, time, argparse, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from vision_mvp.core.llm_client import LLMClient
from vision_mvp.tasks.distributed_summary import DistributedSummaryTask


def run_map_reduce(task: DistributedSummaryTask, n: int,
                   model: str, parallel: int = 1) -> dict:
    """Map: each agent reads its chunk. Reduce: one synthesis."""
    chunks = task.make_chunks(n)
    client = LLMClient(model=model)

    print(f"[map] {n} agents, each reads ONE chunk", flush=True)
    observations = []
    t_map_start = time.time()
    for i, chunk in enumerate(chunks):
        print(f"  agent {i+1}/{n}", flush=True)
        prompt = (
            "You have access ONLY to this chunk of an internal company "
            "incident review. Read it and list the concrete facts you "
            "observe (incident id, vendor if any, detection-to-mitigation "
            "time if any, root-cause hints, runbook mentions). Keep it to "
            "3–4 short sentences.\n\n"
            f"---\n{chunk}\n---\n\n"
            "Your observations:"
        )
        obs = client.generate(prompt, max_tokens=150, temperature=0.2)
        observations.append(obs)
    wall_map = time.time() - t_map_start

    print(f"\n[reduce] single synthesis call over all {n} observations",
          flush=True)
    framing = (
        "You are the lead risk analyst. Below are per-chunk observations "
        "from team members who each saw ONE small chunk of a company's "
        "Q3 incident review. YOUR job: identify the top 3 systemic risks. "
        "A systemic risk is one that RECURS across multiple team members' "
        "observations — same vendor, same failure mode, same root-cause "
        "family. Do NOT promote a one-incident issue to a systemic risk.\n\n"
        "Output format: 3 bullets, each naming the risk and listing the "
        "team members / incidents that support it."
    )
    bullet = "\n".join(f"Team member {i+1}: {o}"
                       for i, o in enumerate(observations))
    prompt = f"{framing}\n\nQuestion: {task.question}\n\n{bullet}\n\nSynthesis:"
    t_red_start = time.time()
    final = client.generate(prompt, max_tokens=600, temperature=0.2)
    wall_red = time.time() - t_red_start

    score = task.score(final)
    return {
        "mode": "map_reduce",
        "n_agents": n,
        "per_chunk_observations": observations,
        "synthesis": final,
        "score": score,
        "wall_map_s": round(wall_map, 1),
        "wall_reduce_s": round(wall_red, 1),
        "total_llm_tokens": client.stats.total_tokens(),
        "n_generate_calls": client.stats.n_generate_calls,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--model", default="qwen2.5-coder:7b")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    task = DistributedSummaryTask()
    print(f"Document: {len(task.document.split())} words across {args.n} chunks",
          flush=True)
    print(f"Ground-truth risks: {list(task.systemic_risks.keys())}\n",
          flush=True)

    result = run_map_reduce(task, args.n, args.model)

    print("\n" + "=" * 78)
    print("SYNTHESIS")
    print("=" * 78)
    print(result["synthesis"])

    print("\n" + "=" * 78)
    print("SCORE")
    print("=" * 78)
    print(f"Risks identified: {result['score']['n_risks_identified']}/3")
    for risk, found in result["score"].items():
        if risk in task.systemic_risks:
            print(f"  {risk}: {'✓' if found else '✗'}")

    print(f"\nMap wall: {result['wall_map_s']}s, reduce wall: "
          f"{result['wall_reduce_s']}s, total LLM tokens: "
          f"{result['total_llm_tokens']}", flush=True)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nWrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
