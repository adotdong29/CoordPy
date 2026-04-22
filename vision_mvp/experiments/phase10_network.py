"""Phase 10 — Agent Network collaborative build.

Hundreds of agents, thirteen specialties, forty interconnected subtasks
forming a DAG. Each agent self-claims ready tasks matching its specialty
key. Completions are routed to downstream agents via MoE routing.

Per-agent context stays bounded regardless of team size — an agent only
sees messages the router delivers to it (capacity-limited inbox), plus
its currently-claimed subtask and that task's direct dependencies.

Usage:
    # Mock-LLM run, fast, for smoke
    python -m vision_mvp.experiments.phase10_network --mock --n 30

    # Real LLM (local Ollama)
    python -m vision_mvp.experiments.phase10_network --n 30 --model qwen2.5-coder:7b
"""

from __future__ import annotations
import sys, os, argparse, json, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np

from vision_mvp.core.agent_network import AgentNetwork, NetworkAgent
from vision_mvp.core.llm_client import LLMClient
from vision_mvp.core.sheaf_monitor import SheafMonitor
from vision_mvp.tasks.collaborative_build import (
    SPECIALTIES, SPECIALTY_DESCRIPTIONS, make_subtasks,
    assign_agent_specialties, score_build,
)
from vision_mvp.core.agent_keys import l2_normalize


def make_real_llm_call(client: LLMClient, specialty_persona: str):
    """Build an llm_call(persona, inbox, ctx) closure for the real LLM."""
    def llm_call(persona: str, inbox, ctx: str) -> str:
        prompt = (
            f"{persona}\n\n"
            f"{ctx}\n\n"
            "Produce your contribution in 3-5 concise sentences. "
            "Reference specific outputs from your dependencies when relevant. "
            "Focus on concrete engineering content, not marketing language."
        )
        return client.generate(prompt, max_tokens=220, temperature=0.2)
    return llm_call


def make_mock_llm_call(specialty: str):
    """Fake llm_call that returns a templated string — for fast smoke tests."""
    def llm_call(persona: str, inbox, ctx: str) -> str:
        ctx_digest = ctx[:80].replace("\n", " | ")
        return (f"[{specialty}] Contribution based on {len(inbox)} messages and "
                f"context snippet '{ctx_digest}'. This deliverable references "
                f"schema, endpoints, services, deployment, and observability.")
    return llm_call


def make_real_embed(client: LLMClient):
    """Embedding closure that caches by text for speed."""
    cache = {}
    def embed(text: str) -> np.ndarray:
        key = text.strip()[:256]
        if key in cache:
            return cache[key]
        emb = np.asarray(client.embed(key), dtype=np.float64)
        cache[key] = emb
        return emb
    return embed


def make_mock_embed(dim: int = 64):
    """Fake embedding: hash text into a stable dim-dimensional unit vector."""
    def embed(text: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(text)) % (2**31))
        return l2_normalize(rng.standard_normal(dim))
    return embed


def build_network(n_agents: int, mock: bool, model: str,
                  progress_cb=None) -> tuple[AgentNetwork, LLMClient | None, int]:
    # Determine embedding dimension
    if mock:
        embed = make_mock_embed(dim=64)
        client = None
        dim_keys = 64
    else:
        client = LLMClient(model=model)
        embed = make_real_embed(client)
        # Probe embedding dim by embedding one short phrase
        if progress_cb:
            progress_cb("probing LLM embedding dimension …")
        probe = embed("probe")
        dim_keys = len(probe)

    net = AgentNetwork(n_agents=n_agents, dim_keys=dim_keys)

    # Assign specialties round-robin
    specialties = assign_agent_specialties(n_agents)
    for aid in range(n_agents):
        spec = specialties[aid]
        persona_text = f"You are agent {aid}. {SPECIALTY_DESCRIPTIONS[spec]}"
        if mock:
            llm_call = make_mock_llm_call(spec)
        else:
            llm_call = make_real_llm_call(client, persona_text)
        agent = NetworkAgent(
            agent_id=aid, persona=persona_text, role=spec,
            llm_call=llm_call, embed_fn=embed,
        )
        # Init key from specialty description — so key ≈ what I do
        net.register_agent(agent, key_init_text=f"{spec}: {SPECIALTY_DESCRIPTIONS[spec]}")

    # Populate task board
    subs = make_subtasks(embed)
    for s in subs:
        net.board.add(s)

    return net, client, dim_keys


def progress(msg):
    print(msg, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="number of agents")
    ap.add_argument("--mock", action="store_true",
                    help="use mock LLM (instant, good for smoke)")
    ap.add_argument("--model", default="qwen2.5-coder:7b")
    ap.add_argument("--max-rounds", type=int, default=20)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    t0 = time.time()
    print(f"Building network: n={args.n} mock={args.mock} model={args.model}",
          flush=True)
    net, client, dim_keys = build_network(
        n_agents=args.n, mock=args.mock, model=args.model,
        progress_cb=progress,
    )
    print(f"  dim_keys={dim_keys} n_clusters={net.keys.n_clusters}",
          flush=True)
    print(f"  task board: {net.board.total_count()} subtasks, "
          f"{len(net.board.ready_tasks())} ready now", flush=True)
    net.router.top_k = args.top_k

    print(f"\nRunning up to {args.max_rounds} rounds …", flush=True)
    per_round_stats = []
    for r in range(args.max_rounds):
        if net.board.all_done():
            print(f"  all done at round {r}", flush=True)
            break
        tr = time.time()
        info = net.run_round(progress_cb=None)
        wall = time.time() - tr
        ready_after = len(net.board.ready_tasks())
        print(f"  round {info['round']}: "
              f"done {info['board']['counts'].get('done', 0)}/"
              f"{info['board']['total']}  "
              f"msgs={info['messages_posted']}  "
              f"ready_next={ready_after}  "
              f"router_mean_load={info['router']['mean_load']} "
              f"max_load={info['router']['max_load']}  "
              f"wall={wall:.1f}s", flush=True)
        per_round_stats.append({"round": info["round"], **info, "wall": wall})

    wall_total = time.time() - t0

    # Scoring
    score = score_build(net.board)
    mstats = net.message_stats()

    # Sheaf consistency (light version): put an edge between every pair of agents
    # that collaborated on a dep-chain, with a very simple stalk/interface
    # dimension. This is a DIAGNOSTIC — we don't auto-reconcile.
    sheaf = SheafMonitor(stalk_dim=dim_keys, interface_dim=min(dim_keys, 8))
    # Use one belief vector per agent = agent's key (proxy)
    beliefs = {aid: net.keys.get_key(aid) for aid in range(args.n)}
    # Edges: every pair of agents who worked on adjacent subtasks (parent→child)
    seen_edges = set()
    for sub in net.board.subtasks.values():
        if sub.assignee is None:
            continue
        for dep in sub.deps:
            dep_sub = net.board.subtasks.get(dep)
            if dep_sub is None or dep_sub.assignee is None:
                continue
            pair = tuple(sorted([sub.assignee, dep_sub.assignee]))
            if pair[0] == pair[1] or pair in seen_edges:
                continue
            seen_edges.add(pair)
            sheaf.add_edge(pair[0], pair[1])
    top_discord = []
    if sheaf.edges:
        top_discord = sheaf.top_disagreements(beliefs, k=5)

    print("\n" + "=" * 78, flush=True)
    print("RESULT", flush=True)
    print("=" * 78, flush=True)
    print(f"  agents: {args.n}  total subtasks: {net.board.total_count()}", flush=True)
    print(f"  completion: {score['completion_rate']:.2%}  "
          f"done {score['n_done']}/{score['n_total']}", flush=True)
    print(f"  integration score: {score['integration_score']:.2f} "
          f"({score['integrations_checked']} cross-dep tasks checked)",
          flush=True)
    print(f"  total wall: {wall_total:.1f}s  "
          f"rounds: {net._round}", flush=True)
    if client:
        print(f"  LLM total tokens: {client.stats.total_tokens():,}  "
              f"generate calls: {client.stats.n_generate_calls}  "
              f"embed calls: {client.stats.n_embed_calls}", flush=True)
    print(f"\n  message stats: {mstats}", flush=True)
    if top_discord:
        print(f"\n  top sheaf-discord edges (agents who disagree most):",
              flush=True)
        for d in top_discord[:3]:
            print(f"    agents {d['u']} <-> {d['v']}: discord {d['discord']:.3f}",
                  flush=True)

    if args.out:
        out_data = {
            "n_agents": args.n,
            "mock": args.mock,
            "model": args.model if not args.mock else None,
            "score": score,
            "message_stats": mstats,
            "per_round_stats": per_round_stats,
            "wall_total": wall_total,
            "llm_generate_calls": client.stats.n_generate_calls if client else 0,
            "llm_total_tokens": client.stats.total_tokens() if client else 0,
            "top_sheaf_discord": top_discord[:5],
        }
        with open(args.out, "w") as f:
            json.dump(out_data, f, indent=2, default=str)
        print(f"\nWrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
