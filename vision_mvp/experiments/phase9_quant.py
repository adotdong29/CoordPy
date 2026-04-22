"""Phase 9 — multi-role team completing a quant strategy task.

Four distinct roles doing DIFFERENT work, end-to-end, producing a
portfolio that is then scored against known ground-truth signals.

Roles:
  - RESEARCH (10 agents): each reads 3 research notes → distills signal/noise
  - MARKET  (20 agents):   each reads ONE asset's time series → directional view
  - STRATEGY (5 agents):   each reads all research digests + market views →
                           proposes per-asset trades
  - PM      (1 agent):     synthesizes strategy proposals → final portfolio

Scoring:
  - hit_rate: fraction of correct directions
  - gross_return_pct: sum of team's direction × next-day return
  - compare vs random baseline (~50% hit rate) and optimal (100%)

Usage:
    python -m vision_mvp.experiments.phase9_quant --model qwen2.5-coder:7b
"""

from __future__ import annotations
import sys, os, argparse, json, re, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from vision_mvp.core.llm_client import LLMClient
from vision_mvp.core.multi_role_team import MultiRoleTeam, Role
from vision_mvp.tasks.quant_strategy import QuantTask


# ---- Prompt templates per role ----

def research_prompt(inputs: list[str], persona: str) -> str:
    return (
        f"{persona}\n\n"
        "Below are 2-3 research notes from our trading desk. For EACH note, "
        "classify as [INFORMATIVE] or [NOISE], then extract the concrete "
        "signal (ticker + direction long/short if any). Be terse.\n\n"
        f"{inputs[0]}\n\n"
        "Your digest (one line per note):"
    )


def market_prompt(inputs: list[str], persona: str) -> str:
    return (
        f"{persona}\n\n"
        "You see ONE asset's last-10-day returns. Decide a directional view "
        "for the NEXT day: LONG (expect up), SHORT (expect down), or FLAT. "
        "State your decision and a one-sentence reason.\n\n"
        f"{inputs[0]}\n\n"
        "Your decision:"
    )


def strategy_prompt(inputs: list[str], persona: str) -> str:
    return (
        f"{persona}\n\n"
        "You have access to: (a) research digests, (b) per-asset market views. "
        "For each ticker you see mentioned, propose FINAL direction long / "
        "short / flat and a one-line rationale. Prefer LONG when both "
        "research and market agree positive; SHORT when both agree negative; "
        "FLAT when they conflict or signal is weak.\n\n"
        f"{inputs[0]}\n\n"
        "Your proposals (one per ticker):"
    )


def pm_prompt(inputs: list[str], persona: str) -> str:
    return (
        f"{persona}\n\n"
        "You are the portfolio manager. Below are your strategy desk's "
        "proposals. Produce the FINAL portfolio: for each ticker, output a "
        "single line in EXACTLY this format:\n"
        "  TICKER: LONG|SHORT|FLAT  — rationale\n"
        "Include every ticker mentioned. Output nothing else.\n\n"
        f"{inputs[0]}\n\n"
        "Final portfolio:"
    )


# ---- Parse PM output → dict[ticker → direction] ----

TICKER_LINE = re.compile(r"(SYN\d{2})\s*:?\s*(LONG|SHORT|FLAT)",
                         re.IGNORECASE)


def parse_portfolio(pm_text: str) -> dict[str, str]:
    directions = {}
    for match in TICKER_LINE.finditer(pm_text):
        ticker = match.group(1).upper()
        direction = match.group(2).lower()
        directions[ticker] = direction
    return directions


def progress(msg: str) -> None:
    print(msg, flush=True)


def run(n_research_agents: int, n_market_agents: int,
        n_strategy_agents: int, model: str, seed: int) -> dict:
    task = QuantTask(n_assets=n_market_agents,
                     n_research_notes=n_research_agents * 3,
                     seed=seed)
    task.generate()

    print(f"Task: {len(task.assets)} assets, "
          f"{len(task.research_notes)} research notes, "
          f"{len(set(a.regime for a in task.assets))} regimes",
          flush=True)
    print(f"Optimal score: {task.optimal_score()}", flush=True)
    print(f"Random baseline: {task.random_baseline_score(seed=0)}\n", flush=True)

    # --- Split research notes across research agents ---
    research_inputs = []
    notes_per_agent = len(task.research_notes) // n_research_agents
    for i in range(n_research_agents):
        slice_notes = task.research_notes[i * notes_per_agent:(i + 1) * notes_per_agent]
        if not slice_notes:
            slice_notes = [task.research_notes[0]]
        bullet = "\n".join(task.note_text(n) for n in slice_notes)
        research_inputs.append(bullet)

    # --- Market inputs: one asset per agent ---
    market_inputs = [task.asset_text(a) for a in task.assets]

    # Strategy agents read nothing raw — they read upstream outputs
    # (we handle that by making strategy role read from 'research' —
    # but it actually needs BOTH research and market.
    # Workaround: create a tiny "merge" role between that just copies
    # both upstream into its agent's context.)

    client = LLMClient(model=model)

    roles = [
        Role(name="research",
             persona="You are a sell-side research analyst. Distill signal from noise.",
             prompt_template=research_prompt,
             n_agents=n_research_agents,
             reads_from="raw",
             max_tokens=150),
        Role(name="market",
             persona="You are a technical analyst specializing in daily return patterns.",
             prompt_template=market_prompt,
             n_agents=n_market_agents,
             reads_from="raw",
             max_tokens=60),
        # Strategy agents read research; market views are spliced in below
        # via a manual merge step.
        Role(name="strategy",
             persona="You are a strategy quant synthesizing research and market views.",
             prompt_template=strategy_prompt,
             n_agents=n_strategy_agents,
             reads_from="research",
             max_tokens=300),
        Role(name="pm",
             persona="You are the portfolio manager producing the final book.",
             prompt_template=pm_prompt,
             n_agents=1,
             reads_from="strategy",
             max_tokens=500),
    ]

    team = MultiRoleTeam(
        roles=roles,
        raw_inputs_by_role={"research": research_inputs, "market": market_inputs},
        client=client,
    )

    # Patch the strategy layer to include BOTH research digests AND market
    # views. We do this by running research + market layers first, then
    # manually constructing the strategy layer's inputs.
    # Run research + market manually:
    t0 = time.time()
    progress(f"[research] {n_research_agents} agents distilling notes …")
    for i, inp in enumerate(research_inputs):
        progress(f"  research agent {i+1}/{n_research_agents}")
        out = client.generate(research_prompt([inp], roles[0].persona),
                              max_tokens=roles[0].max_tokens, temperature=0.2)
        team._outputs_by_role.setdefault("research", []).append(out)
    team._wall_by_role["research"] = time.time() - t0

    t0 = time.time()
    progress(f"[market] {n_market_agents} agents analyzing assets …")
    for i, inp in enumerate(market_inputs):
        progress(f"  market agent {i+1}/{n_market_agents}")
        out = client.generate(market_prompt([inp], roles[1].persona),
                              max_tokens=roles[1].max_tokens, temperature=0.2)
        team._outputs_by_role.setdefault("market", []).append(out)
    team._wall_by_role["market"] = time.time() - t0

    # Strategy reads BOTH — construct input as concatenation
    research_block = "\n".join(f"- {o}" for o in team._outputs_by_role["research"])
    market_block = "\n".join(f"- {o}" for o in team._outputs_by_role["market"])
    combined = (f"Research digests:\n{research_block}\n\n"
                f"Per-asset market views:\n{market_block}")

    t0 = time.time()
    progress(f"[strategy] {n_strategy_agents} agents combining signals …")
    for i in range(n_strategy_agents):
        progress(f"  strategy agent {i+1}/{n_strategy_agents}")
        out = client.generate(
            strategy_prompt([combined], roles[2].persona),
            max_tokens=roles[2].max_tokens, temperature=0.2,
        )
        team._outputs_by_role.setdefault("strategy", []).append(out)
    team._wall_by_role["strategy"] = time.time() - t0

    # PM reads all strategy proposals
    strategy_block = "\n".join(f"=== Strategy proposal {i+1} ===\n{o}"
                                for i, o in enumerate(team._outputs_by_role["strategy"]))
    t0 = time.time()
    progress("[pm] synthesizing final portfolio …")
    pm_out = client.generate(pm_prompt([strategy_block], roles[3].persona),
                             max_tokens=roles[3].max_tokens, temperature=0.2)
    team._outputs_by_role["pm"] = [pm_out]
    team._wall_by_role["pm"] = time.time() - t0

    print("\n" + "=" * 78)
    print("FINAL PORTFOLIO (PM output)")
    print("=" * 78, flush=True)
    print(pm_out)

    directions = parse_portfolio(pm_out)
    print(f"\nParsed {len(directions)} ticker decisions", flush=True)

    score = task.score_portfolio(directions)
    print("\n" + "=" * 78)
    print("SCORE")
    print("=" * 78)
    for k, v in score.items():
        print(f"  {k}: {v}")

    print(f"\nRandom baseline: hit_rate {task.random_baseline_score()['hit_rate']}  "
          f"Optimal: hit_rate {task.optimal_score()['hit_rate']}", flush=True)

    stats = team.stats
    stats.update({
        "task_score": score,
        "directions": directions,
        "final_portfolio_text": pm_out,
        "optimal_score": task.optimal_score(),
        "random_baseline_score": task.random_baseline_score(),
    })
    print(f"\n{stats}", flush=True)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5-coder:7b")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--research", type=int, default=6)
    ap.add_argument("--market", type=int, default=16)
    ap.add_argument("--strategy", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    result = run(n_research_agents=args.research,
                 n_market_agents=args.market,
                 n_strategy_agents=args.strategy,
                 model=args.model, seed=args.seed)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nWrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
