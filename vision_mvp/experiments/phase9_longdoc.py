"""Phase 9B — longer-than-context document + map-reduce team.

The corpus is ~11 000 words (≈ 14 500 tokens), roughly 3.5× Ollama's
default 4 k context window. A single agent attempting the full document
will be truncated silently to ~2 800 words and miss most of it.

The map-reduce team, where each agent sees 1/40th of the document, has
NO such limitation — each chunk is tiny. The synthesis over all chunk
observations is the team's path to the full answer.

Three runs:
  1. ORACLE (attempts full doc) — shows what default-context truncation
     looks like (or, if the model has ≥ 16k context, shows that an
     unbounded-context oracle still can match).
  2. Map-reduce team (40 agents, one chunk each) — the distributed
     solution that is INDEPENDENT of per-agent context size.
  3. Smaller-sample oracle (last chunk only) — sanity that default
     context works when input is small.

Run:
    python -m vision_mvp.experiments.phase9_longdoc --n 40
"""

from __future__ import annotations
import sys, os, argparse, json, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from vision_mvp.core.llm_client import LLMClient
from vision_mvp.tasks.long_corpus import LongCorpusTask


def run_oracle(task: LongCorpusTask, model: str) -> dict:
    client = LLMClient(model=model)
    prompt = (
        "You are a senior risk analyst. Read this internal incident review "
        "and answer the question. If any part of the document seems "
        "truncated or missing, still give your best answer.\n\n"
        f"---\n{task.document}\n---\n\n"
        "Question: Name the top 3 systemic risks facing this company. "
        "For each risk, explain which incidents support it."
    )
    t0 = time.time()
    ans = client.generate(prompt, max_tokens=400, temperature=0.2)
    wall = time.time() - t0
    s = task.score(ans)
    return {"mode": "oracle", "answer": ans, "score": s,
            "wall_s": round(wall, 1),
            "tokens": client.stats.total_tokens()}


def run_map_reduce(task: LongCorpusTask, n_chunks: int,
                    model: str) -> dict:
    chunks = task.chunk(n_chunks)
    client = LLMClient(model=model)

    observations = []
    t_map0 = time.time()
    print(f"[map] {n_chunks} agents, each reads ONE chunk…", flush=True)
    for i, chunk in enumerate(chunks):
        if (i + 1) % 5 == 0 or i == 0:
            print(f"  agent {i+1}/{n_chunks}", flush=True)
        prompt = (
            "You have access ONLY to this chunk of an internal company "
            "incident review. Read it and list concrete facts — incident "
            "IDs, vendor names, detection-to-mitigation times, "
            "runbook/doc issues, root causes. 3–4 short sentences.\n\n"
            f"---\n{chunk}\n---\n\n"
            "Your observations:"
        )
        obs = client.generate(prompt, max_tokens=180, temperature=0.2)
        observations.append(obs)
    wall_map = time.time() - t_map0

    # Batched synthesis over all observations
    print("\n[reduce] synthesis over all observations…", flush=True)
    bullets = "\n".join(f"Member {i+1}: {o}"
                        for i, o in enumerate(observations))
    prompt = (
        "You are the lead risk analyst. Below are per-chunk observations "
        "from team members who each saw ONE small chunk of a company's "
        "multi-quarter incident review. Identify the top 3 systemic risks. "
        "A systemic risk must recur across multiple members' observations "
        "— same vendor, same failure mode, same root-cause family. Do NOT "
        "promote a one-incident issue to a systemic risk.\n\n"
        "Output three numbered risks. For each, name 2-4 supporting "
        "incidents / members.\n\n"
        f"{bullets}\n\nSynthesis:"
    )
    t_r0 = time.time()
    final = client.generate(prompt, max_tokens=600, temperature=0.2)
    wall_reduce = time.time() - t_r0

    s = task.score(final)
    return {
        "mode": "map_reduce",
        "n_agents": n_chunks,
        "synthesis": final,
        "score": s,
        "wall_map_s": round(wall_map, 1),
        "wall_reduce_s": round(wall_reduce, 1),
        "tokens": client.stats.total_tokens(),
        "n_generate_calls": client.stats.n_generate_calls,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5-coder:7b")
    ap.add_argument("--sections", type=int, default=40)
    ap.add_argument("--n", type=int, default=40,
                    help="number of chunks / agents in map-reduce")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--skip-oracle", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    task = LongCorpusTask(n_sections=args.sections, seed=args.seed)
    task.generate()
    print(f"Document: {task.word_count} words (~{task.word_count*4//3} tokens)",
          flush=True)
    print(f"Risks being scored: {list(task.systemic_risks.keys())}\n",
          flush=True)

    results = {}

    if not args.skip_oracle:
        print("=" * 78)
        print("RUN 1: oracle (single agent, attempts full doc)")
        print("=" * 78, flush=True)
        try:
            orc = run_oracle(task, args.model)
            print(f"\nAnswer:\n{orc['answer']}\n")
            print(f"Score: {orc['score']['n_risks_identified']}/3 "
                  f"({orc['wall_s']}s, {orc['tokens']} tokens)", flush=True)
            results["oracle"] = orc
        except Exception as e:
            print(f"Oracle failed: {e}", flush=True)
            results["oracle"] = {"failed": str(e)}

    print("\n" + "=" * 78)
    print(f"RUN 2: map-reduce team (N={args.n} agents, 1 chunk each)")
    print("=" * 78, flush=True)
    mr = run_map_reduce(task, n_chunks=args.n, model=args.model)
    print(f"\nSynthesis:\n{mr['synthesis']}\n")
    print(f"Score: {mr['score']['n_risks_identified']}/3 "
          f"(map {mr['wall_map_s']}s, reduce {mr['wall_reduce_s']}s, "
          f"{mr['tokens']} tokens)", flush=True)
    results["map_reduce"] = mr

    print("\n" + "=" * 78)
    print("SCOREBOARD")
    print("=" * 78)
    print(f"{'mode':>14} | {'risks':>5} | {'tokens':>8} | {'wall_s':>8}")
    if "oracle" in results and "score" in results["oracle"]:
        r = results["oracle"]
        print(f"{'oracle':>14} | {r['score']['n_risks_identified']:>5}/3 | "
              f"{r['tokens']:>8} | {r['wall_s']:>8}")
    r = results["map_reduce"]
    total_mr_wall = r["wall_map_s"] + r["wall_reduce_s"]
    print(f"{'map_reduce':>14} | {r['score']['n_risks_identified']:>5}/3 | "
          f"{r['tokens']:>8} | {total_mr_wall:>8}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nWrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
