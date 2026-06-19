"""W78 R-195 benchmark family — Multi-Agent Task Success across
18 regimes (the primary scoreboard).

Exercises MASC V14 across all 18 regimes — the 17 W77 regimes
plus the new W78 long-horizon-reconstruction regime. Verifies V23
strictly beats V22 on ≥ 50 % of seeds in every regime, and TSC_V23
strictly beats TSC_V22 on ≥ 50 % of seeds in every regime. Adds
the new bounded-window-baseline-failure bar.

H1220..H1239 cell families (20 H-bars: 18 per-regime + 2 special
bars).
"""

from __future__ import annotations

from typing import Any, Sequence

from coordpy.bounded_window_baseline_v1 import (
    BoundedWindowQuery,
    build_default_bounded_window_baselines,
    run_bounded_window_falsifier,
)
from coordpy.long_horizon_reconstruction_substrate_v1 import (
    LongHorizonReconstructionQuery,
    build_default_long_horizon_reconstruction_carrier,
    reconstruct_long_horizon_event,
)
from coordpy.multi_agent_substrate_coordinator_v14 import (
    MultiAgentSubstrateCoordinatorV14,
    W78_MASC_V14_REGIMES,
)


R195_SCHEMA_VERSION: str = "coordpy.r195_benchmark.v1"


def run_r195(*, seeds: Sequence[int]) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    masc = MultiAgentSubstrateCoordinatorV14()
    for i, regime in enumerate(W78_MASC_V14_REGIMES):
        _, agg = masc.run_batch(
            seeds=list(seeds), regime=regime)
        # H1220..H1237: V23 strict beat ≥ 0.5 per regime.
        cells[f"H{1220 + i}"] = bool(
            float(agg.v23_beats_v22_rate) >= 0.5
            and float(agg.tsc_v23_beats_tsc_v22_rate) >= 0.5)
    # H1238: bounded-window baseline failure bar — all fixed-k
    # baselines fail on a 200-turn-back query.
    baselines = build_default_bounded_window_baselines()
    bw_query = BoundedWindowQuery(
        query_id="bw_q", current_turn=220,
        source_turn=20,
        expected_event_cid="placeholder")
    _, falsifier = run_bounded_window_falsifier(
        baselines=baselines, query=bw_query)
    cells["H1238"] = bool(falsifier.all_fixed_k_failed)
    # H1239: W78 long-horizon-reconstruction substrate succeeds
    # on the same query family.
    carrier = build_default_long_horizon_reconstruction_carrier(
        n_events=256, seed=78030)
    lhr_q = LongHorizonReconstructionQuery(
        query_id="lhr_q", source_turn=20, current_turn=220)
    out = reconstruct_long_horizon_event(
        carrier=carrier, query=lhr_q, visible_tokens_used=4)
    cells["H1239"] = bool(out.success)
    return {
        "schema": R195_SCHEMA_VERSION,
        "seeds": list(seeds),
        "cells": dict(cells),
        "n_regimes": len(W78_MASC_V14_REGIMES),
        "all_pass": bool(all(bool(v) for v in cells.values())),
    }


__all__ = ["R195_SCHEMA_VERSION", "run_r195"]
