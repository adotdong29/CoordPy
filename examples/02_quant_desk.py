"""Example 02 — small quant trading desk as a CoordPy AgentTeam.

A four-role team driven entirely off the stable public ``presets``
surface:

    market_watcher  ->  signal_researcher  ->  risk_manager
                                                       \\
                                                        ->  portfolio_synthesizer

The synthesizer emits an ``ACTION:`` line in
``{EXECUTE, EXECUTE-WITH-MODS, NO-ACTION}`` that
:meth:`coordpy.TeamResult.parse_action` extracts as a structured
:class:`~coordpy.ActionDecision`.

Why this example exists
-----------------------

The README pitches CoordPy as the right runtime for "AI agent teams"
and recommends ``presets.quant_desk_team(...)`` as the one-line
preset path. This file actually runs that path against a real local
LLM and demonstrates the bounded-context handoff story: the writer
never receives the full transcript, only the latest N
``TEAM_HANDOFF`` payloads that the runtime threads forward.

How to run
----------

Local Ollama::

    COORDPY_BACKEND=ollama \\
    COORDPY_MODEL=qwen2.5:14b \\
    COORDPY_OLLAMA_URL=http://localhost:11434 \\
        python3 examples/02_quant_desk.py --out-dir /tmp/desk-run

OpenAI-compatible provider::

    COORDPY_BACKEND=openai \\
    COORDPY_MODEL=gpt-4o-mini \\
    COORDPY_API_KEY=... \\
        python3 examples/02_quant_desk.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from coordpy import AgentTurn, presets


SCENARIO_BULLISH = """\
Date: 2026-05-03 (Friday close).
Universe: US large-cap equities; focus on AAPL, NVDA, MSFT.
Market context:
  - SPX +0.4% on the week, breadth narrow (top-10 names = 62% of return).
  - 10y yield 4.31%, +6 bp on the week; VIX 14.2.
  - NVDA: beat consensus on data-center revenue, guidance modestly raised.
  - AAPL: services strong, hardware unit growth flat; supply chain stable.
  - MSFT: Azure growth re-accelerating, capex guidance up.
News tape (last 24h):
  - Fed minutes: dot plot unchanged; one fewer cut priced for 2026 H2.
  - Treasury auction: 7y solid bid-to-cover, no concession.
  - Geopolitics: Taiwan Strait quiet; Red Sea shipping incidents flat WoW.
Risk constraints:
  - Single-name max position 4% of NAV.
  - Sector concentration cap: tech at most 30% gross exposure.
  - No new positions in names with earnings inside 5 trading days.
"""

SCENARIO_RISK_OFF = """\
Date: 2026-05-03 (Friday close).
Universe: US large-cap equities; focus on AAPL, NVDA, MSFT.
Market context:
  - SPX -2.1% on the week, breadth deteriorating; advance/decline -3:1.
  - 10y yield 4.62%, +18 bp on the week; VIX 22.8 (last 30d high).
  - NVDA: missed data-center guide; whispered slowdown in hyperscaler capex.
  - AAPL: China-revenue warning; services growth decelerating.
  - MSFT: Azure flat sequentially; capex guidance reduced.
News tape (last 24h):
  - Fed minutes: hawkish surprise, dot plot lifted, no cuts priced for 2026.
  - Treasury auction: 30y tailed 3 bp, weak bid-to-cover.
  - Geopolitics: tariff escalation rumours on semis; risk premium widening.
Risk constraints:
  - Single-name max position 4% of NAV.
  - Sector concentration cap: tech at most 30% gross exposure.
  - No new positions in names with earnings inside 5 trading days.
"""

SCENARIO_AMBIGUOUS = """\
Date: 2026-05-03 (Friday close).
Universe: US large-cap equities; focus on AAPL, NVDA, MSFT.
Market context:
  - SPX flat on the week, breadth mixed; equal-weight outperforming cap-weight.
  - 10y yield 4.40%, unchanged WoW; VIX 17.5 (mid-range).
  - NVDA: in-line print, guide steady; consensus modelling already
    priced in modest beat.
  - AAPL: services strong, hardware growth modestly negative; mixed
    read across geographies.
  - MSFT: Azure stable, capex guidance held.
News tape (last 24h):
  - Fed minutes: no surprise; market pricing 1 cut H2 2026.
  - Treasury auction: 5y on-screen, no signal.
  - Geopolitics: nothing material.
Risk constraints:
  - Single-name max position 4% of NAV.
  - Sector concentration cap: tech at most 30% gross exposure.
  - No new positions in names with earnings inside 5 trading days.
"""

SCENARIOS = {
    "bullish": SCENARIO_BULLISH,
    "risk_off": SCENARIO_RISK_OFF,
    "ambiguous": SCENARIO_AMBIGUOUS,
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument(
        "--scenario", default="bullish",
        choices=list(SCENARIOS) + ["custom"],
        help="Built-in scenario name, or 'custom' with --scenario-file.")
    ap.add_argument(
        "--scenario-file", default=None,
        help="Path to a custom scenario.txt; only used with --scenario custom.")
    ap.add_argument(
        "--max-visible-handoffs", type=int, default=2,
        help="Bounded context: last-N handoffs threaded forward. "
             "Default 2 -- demonstrates real bounded context vs. cramming. "
             "Pass a very large number (e.g. 99) for the unbounded baseline.")
    ap.add_argument(
        "--max-tokens", type=int, default=360,
        help="Per-agent generation cap.")
    ap.add_argument(
        "--temperature", type=float, default=0.0,
        help="Per-agent sampling temperature.")
    ap.add_argument(
        "--out-dir", default=None,
        help="If set, dump the team capsule view + desk note + manifest "
             "via TeamResult.dump(out_dir).")
    ap.add_argument(
        "--quiet", action="store_true",
        help="Skip the live per-turn streaming display.")
    return ap.parse_args()


def load_scenario(name: str, path: str | None) -> str:
    if name != "custom":
        return SCENARIOS[name]
    if path is None:
        raise SystemExit(
            "--scenario custom requires --scenario-file PATH")
    return Path(path).read_text(encoding="utf-8")


def stream_turn(turn: AgentTurn) -> None:
    """Plain-text per-turn progress callback; no terminal helpers."""
    cid = (turn.capsule_cid or "")[:12] or "-"
    head = (turn.output or "").strip().splitlines()[0] if turn.output else ""
    head = head[:96] + ("..." if len(head) > 96 else "")
    print(
        f"  {turn.role:<22s} capsule={cid}  "
        f"in={turn.prompt_tokens:>5d}  out={turn.output_tokens:>5d}  "
        f"wall={turn.wall_ms / 1000:6.2f}s  "
        f"vis={turn.visible_handoffs}",
        flush=True,
    )
    if head:
        print(f"    -> {head}", flush=True)


def main() -> int:
    args = parse_args()
    scenario = load_scenario(args.scenario, args.scenario_file)

    # One-line preset construction. presets.quant_desk_team picks up
    # the backend from COORDPY_BACKEND / COORDPY_MODEL / COORDPY_*
    # env vars and applies the curated four-role role instructions
    # tuned for the canonical pattern.
    try:
        team = presets.quant_desk_team(
            max_visible_handoffs=args.max_visible_handoffs,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
    except Exception as exc:
        print(
            "could not build team backend.\n"
            "  reason: " + repr(exc) + "\n"
            "  expected env: COORDPY_BACKEND, COORDPY_MODEL, and either\n"
            "                COORDPY_OLLAMA_URL (ollama) or\n"
            "                COORDPY_API_KEY (openai-compatible).",
            file=sys.stderr,
        )
        return 2

    backend = team.backend
    print(
        f"=== coordpy quant desk - scenario={args.scenario}, "
        f"max_visible_handoffs={args.max_visible_handoffs} ===\n"
        f"  backend     {type(backend).__name__}\n"
        f"  model       {backend.model}\n"
        f"  base_url    {backend.base_url or '(default)'}\n"
        f"  max_tokens  {args.max_tokens}\n"
        f"  temperature {args.temperature}\n"
    )

    task = (
        "Produce a single desk note for the following scenario.\n"
        f"\n=== SCENARIO ===\n{scenario}\n=== END SCENARIO ==="
    )

    print("=== LIVE TURNS ===")
    result = team.run(
        task,
        progress=None if args.quiet else stream_turn,
    )

    print()
    print("=== FINAL DESK NOTE ===")
    print(result.final_output.strip())

    cramming = result.cramming_estimate()
    action = result.parse_action()
    print()
    print("=== TELEMETRY ===")
    print(f"  turns          {len(result.turns)}")
    print(
        f"  total_tokens   {result.total_tokens}  "
        f"(in={result.total_prompt_tokens}, "
        f"out={result.total_output_tokens})"
    )
    print(f"  total_wall     {result.total_wall_ms / 1000.0:.2f} s")
    print(f"  calls          {result.total_calls}")
    print(f"  bounded_words  {cramming['bounded_words']}")
    print(f"  naive_words    {cramming['naive_words']}")
    print(
        f"  savings        {cramming['saved_words']} words "
        f"({cramming['savings_pct']:.1f}%)  "
        f"~{cramming['estimated_tokens_saved']} tokens"
    )
    if action is not None:
        print(f"  parsed_action  {action.action}")

    print()
    print("=== AUDIT ===")
    print(f"  capsule_root   {result.root_cid or '-'}")
    if result.capsule_view is not None:
        cv = result.capsule_view
        stats = cv.get("stats") or {}
        capsules = cv.get("capsules") or []
        print(f"  capsule_schema {cv.get('schema')}")
        print(
            f"  chain_ok       "
            f"{'yes' if cv.get('chain_ok') else 'NO'}")
        head = cv.get("chain_head") or ""
        print(f"  chain_head     {head[:32]}...")
        print(
            f"  n_capsules     "
            f"{stats.get('n_entries') or len(capsules)}")
        print(f"  by_kind        {stats.get('by_kind') or {}}")

    if args.out_dir:
        paths = result.dump(args.out_dir)
        print()
        print("=== ARTEFACTS ===")
        for label, path in paths.items():
            print(f"  {label:<14s} {path}")
        print()
        print("Re-verify the sealed chain from disk:")
        print(
            f"  coordpy-capsule verify-view "
            f"--view {paths['capsule_view']}"
        )
        print()
        print("Replay the same prompts on another model:")
        print(
            f"  coordpy-team replay --result {paths['team_result']} \\"
        )
        print("    --backend ollama --model gemma2:9b \\")
        print("    --out-dir /tmp/desk-replay")

    return 0


if __name__ == "__main__":
    sys.exit(main())
