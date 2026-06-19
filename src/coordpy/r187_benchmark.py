"""W76 R-187 benchmark family — Multi-agent task success across
sixteen regimes (Plane B).

The primary scoreboard for W76. Runs MASC V12 across all 16
regimes and checks:

* V21 strictly beats V20 on ≥ 50 % of seeds for each regime
  (the load-bearing W76 win bar).
* TSC V21 strictly beats TSC V20 on ≥ 50 % of seeds for each
  regime.
* team_success_per_visible_token is non-trivial for V21 + TSC V21.

H1110..H1141 cell families (32 H-bars; two per regime — V21 beat
+ TSC V21 beat). Plus extra summary bars.
"""

from __future__ import annotations

from typing import Any, Sequence

from coordpy.multi_agent_substrate_coordinator_v12 import (
    MultiAgentSubstrateCoordinatorV12,
    W76_MASC_V12_REGIMES,
    W76_MASC_V12_REGIME_CHAIN_THEN_RESTART,
)


R187_SCHEMA_VERSION: str = "coordpy.r187_benchmark.v1"


def run_r187(*, seeds: Sequence[int]) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    masc = MultiAgentSubstrateCoordinatorV12()
    per_regime: dict[str, Any] = {}
    for regime in W76_MASC_V12_REGIMES:
        _, agg = masc.run_batch(
            seeds=list(seeds), regime=regime,
            n_agents=5, n_turns=12,
            budget_tokens_per_turn=64,
            target_tolerance=0.1)
        per_regime[regime] = agg
        idx = int(W76_MASC_V12_REGIMES.index(regime))
        bar_v21 = f"H{1110 + 2 * idx}"
        bar_tsc = f"H{1110 + 2 * idx + 1}"
        cells[bar_v21] = bool(agg.v21_beats_v20_rate >= 0.5)
        cells[bar_tsc] = bool(
            agg.tsc_v21_beats_tsc_v20_rate >= 0.5)
    # Extra summary bars beyond the 32:
    ctr_agg = per_regime[W76_MASC_V12_REGIME_CHAIN_THEN_RESTART]
    cells["H1142"] = bool(
        ctr_agg.v21_beats_v20_rate >= 0.5)
    cells["H1143"] = bool(
        ctr_agg.team_success_per_visible_token_v21 > 0.0)
    return {
        "schema": R187_SCHEMA_VERSION,
        "n_seeds": int(len(seeds)),
        "n_regimes": int(len(W76_MASC_V12_REGIMES)),
        "cells": cells,
        "per_regime_v21_beats": {
            r: float(a.v21_beats_v20_rate)
            for r, a in per_regime.items()},
        "per_regime_tsc_v21_beats": {
            r: float(a.tsc_v21_beats_tsc_v20_rate)
            for r, a in per_regime.items()},
        "per_regime_team_success_per_visible_token_v21": {
            r: float(a.team_success_per_visible_token_v21)
            for r, a in per_regime.items()},
        "all_pass": bool(all(cells.values())),
    }


__all__ = [
    "R187_SCHEMA_VERSION",
    "run_r187",
]
