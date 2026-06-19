"""W77 R-191 benchmark family — Multi-agent task success across
seventeen regimes (Plane B).

The primary scoreboard for W77. Runs MASC V13 across all 17
regimes and checks:

* V22 strictly beats V21 on ≥ 50 % of seeds for each regime
  (the load-bearing W77 win bar).
* TSC V22 strictly beats TSC V21 on ≥ 50 % of seeds for each
  regime.
* team_success_per_visible_token is non-trivial for V22 + TSC V22.

H1200..H1235 cell families (34 H-bars; two per regime — V22 beat
+ TSC V22 beat). Plus extra summary bars.
"""

from __future__ import annotations

from typing import Any, Sequence

from coordpy.multi_agent_substrate_coordinator_v13 import (
    MultiAgentSubstrateCoordinatorV13,
    W77_MASC_V13_REGIMES,
    W77_MASC_V13_REGIME_POST_RESTART_REPLACEMENT,
)


R191_SCHEMA_VERSION: str = "coordpy.r191_benchmark.v1"


def run_r191(*, seeds: Sequence[int]) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    masc = MultiAgentSubstrateCoordinatorV13()
    per_regime: dict[str, Any] = {}
    for regime in W77_MASC_V13_REGIMES:
        _, agg = masc.run_batch(
            seeds=list(seeds), regime=regime,
            n_agents=5, n_turns=12,
            budget_tokens_per_turn=64,
            target_tolerance=0.1)
        per_regime[regime] = agg
        idx = int(W77_MASC_V13_REGIMES.index(regime))
        bar_v22 = f"H{1200 + 2 * idx}"
        bar_tsc = f"H{1200 + 2 * idx + 1}"
        cells[bar_v22] = bool(agg.v22_beats_v21_rate >= 0.5)
        cells[bar_tsc] = bool(
            agg.tsc_v22_beats_tsc_v21_rate >= 0.5)
    # Extra summary bars beyond the 34:
    pcr_agg = per_regime[
        W77_MASC_V13_REGIME_POST_RESTART_REPLACEMENT]
    cells["H1236"] = bool(pcr_agg.v22_beats_v21_rate >= 0.5)
    cells["H1237"] = bool(
        pcr_agg.team_success_per_visible_token_v22 > 0.0)
    return {
        "schema": R191_SCHEMA_VERSION,
        "n_seeds": int(len(seeds)),
        "n_regimes": int(len(W77_MASC_V13_REGIMES)),
        "cells": cells,
        "per_regime_v22_beats": {
            r: float(a.v22_beats_v21_rate)
            for r, a in per_regime.items()},
        "per_regime_tsc_v22_beats": {
            r: float(a.tsc_v22_beats_tsc_v21_rate)
            for r, a in per_regime.items()},
        "per_regime_team_success_per_visible_token_v22": {
            r: float(a.team_success_per_visible_token_v22)
            for r, a in per_regime.items()},
        "all_pass": bool(all(cells.values())),
    }


__all__ = ["R191_SCHEMA_VERSION", "run_r191"]
