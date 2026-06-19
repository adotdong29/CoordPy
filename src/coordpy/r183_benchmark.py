"""W75 R-183 benchmark family — Multi-agent task success across
fifteen regimes (Plane B).

The primary scoreboard for W75. Runs MASC V11 across all 15
regimes and checks:

* V20 strictly beats V19 on ≥ 50 % of seeds for each regime
  (the load-bearing W75 win bar).
* TSC V20 strictly beats TSC V19 on ≥ 50 % of seeds for each
  regime.
* team_success_per_visible_token is non-trivial for V20 + TSC V20.
* Visible-token savings vs transcript ≥ 50 % for V20 / TSC V20.

H1010..H1039 cell families (30 H-bars; two per regime — V20 beat +
TSC V20 beat).
"""

from __future__ import annotations

from typing import Any, Sequence

from coordpy.multi_agent_substrate_coordinator_v11 import (
    MultiAgentSubstrateCoordinatorV11,
    W75_MASC_V11_REGIMES,
    W75_MASC_V11_REGIME_COMPOUND_CHAIN,
)


R183_SCHEMA_VERSION: str = "coordpy.r183_benchmark.v1"


def run_r183(*, seeds: Sequence[int]) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    masc = MultiAgentSubstrateCoordinatorV11()
    per_regime: dict[str, Any] = {}
    for regime in W75_MASC_V11_REGIMES:
        _, agg = masc.run_batch(
            seeds=list(seeds), regime=regime,
            n_agents=5, n_turns=12,
            budget_tokens_per_turn=64,
            target_tolerance=0.1)
        per_regime[regime] = agg
        idx = int(W75_MASC_V11_REGIMES.index(regime))
        bar_v20 = f"H{1010 + 2 * idx}"
        bar_tsc = f"H{1010 + 2 * idx + 1}"
        cells[bar_v20] = bool(agg.v20_beats_v19_rate >= 0.5)
        cells[bar_tsc] = bool(
            agg.tsc_v20_beats_tsc_v19_rate >= 0.5)
    # Extra summary bars beyond the 30:
    chain_agg = per_regime[W75_MASC_V11_REGIME_COMPOUND_CHAIN]
    cells["H1040"] = bool(
        chain_agg.v20_beats_v19_rate >= 0.5)  # Compound-chain.
    cells["H1041"] = bool(
        chain_agg.team_success_per_visible_token_v20 > 0.0)
    return {
        "schema": R183_SCHEMA_VERSION,
        "n_seeds": int(len(seeds)),
        "n_regimes": int(len(W75_MASC_V11_REGIMES)),
        "cells": cells,
        "per_regime_v20_beats": {
            r: float(a.v20_beats_v19_rate)
            for r, a in per_regime.items()},
        "per_regime_tsc_v20_beats": {
            r: float(a.tsc_v20_beats_tsc_v19_rate)
            for r, a in per_regime.items()},
        "per_regime_team_success_per_visible_token_v20": {
            r: float(a.team_success_per_visible_token_v20)
            for r, a in per_regime.items()},
        "all_pass": bool(all(cells.values())),
    }


__all__ = [
    "R183_SCHEMA_VERSION",
    "run_r183",
]
