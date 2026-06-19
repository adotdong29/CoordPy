"""W79 R-199 benchmark family — Multi-Agent Task Success across
19 regimes.

Exercises MASC V15 across all 19 regimes — the 18 W78 regimes
plus the new W79 regime. Verifies V24 strictly beats V23 on ≥ 50 %
of seeds in every regime, and TSC_V24 strictly beats TSC_V23 on
≥ 50 % of seeds in every regime. Plus a bounded-window-baseline
V2 failure bar and a long-horizon-reconstruction V2 success bar
on the new W79 regime.

H1330..H1351 cell families (22 H-bars: 19 per-regime + 3 special).
"""

from __future__ import annotations

from typing import Any, Sequence

from coordpy.bounded_window_baseline_v1 import (
    BoundedWindowQuery,
)
from coordpy.bounded_window_baseline_v2 import (
    build_default_bounded_window_baselines_v2,
    prove_bounded_window_insufficient_v2,
    run_bounded_window_falsifier_v2,
)
from coordpy.learned_consolidation_v1 import (
    build_learned_consolidation_head_v1,
)
from coordpy.long_horizon_reconstruction_substrate_v1 import (
    LongHorizonReconstructionQuery,
)
from coordpy.long_horizon_reconstruction_substrate_v2 import (
    build_default_long_horizon_reconstruction_carrier_v2,
    reconstruct_long_horizon_event_v2,
)
from coordpy.multi_agent_substrate_coordinator_v15 import (
    MultiAgentSubstrateCoordinatorV15,
    W79_MASC_V15_REGIMES,
)


R199_SCHEMA_VERSION: str = "coordpy.r199_benchmark.v1"


def run_r199(*, seeds: Sequence[int]) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    masc = MultiAgentSubstrateCoordinatorV15()
    for i, regime in enumerate(W79_MASC_V15_REGIMES):
        _, agg = masc.run_batch(
            seeds=list(seeds), regime=regime)
        # H1330..H1348: V24 strict beat ≥ 0.5 per regime.
        cells[f"H{1330 + i}"] = bool(
            float(agg.v24_beats_v23_rate) >= 0.5
            and float(agg.tsc_v24_beats_tsc_v23_rate) >= 0.5)
    # H1349: bounded-window V2 baseline failure on the W79
    # source-event horizon.
    baselines = build_default_bounded_window_baselines_v2()
    bw_q = BoundedWindowQuery(
        query_id="bw_q199",
        current_turn=250, source_turn=10,
        expected_event_cid="ev_x")
    _, fals = run_bounded_window_falsifier_v2(
        baselines_v2=baselines, query=bw_q)
    cells["H1349"] = bool(fals.all_fixed_k_failed_v2)
    # H1350: W79 long-horizon-reconstruction substrate V2 succeeds.
    head = build_learned_consolidation_head_v1(seed=199)
    carrier = build_default_long_horizon_reconstruction_carrier_v2(
        n_events=256, seed=199, head=head)
    lhr_q = LongHorizonReconstructionQuery(
        query_id="lhr_q199",
        source_turn=10, current_turn=250)
    out = reconstruct_long_horizon_event_v2(
        carrier_v2=carrier, query=lhr_q,
        visible_tokens_used=4)
    cells["H1350"] = bool(out.success)
    # H1351: V2 insufficiency proof.
    proof = prove_bounded_window_insufficient_v2(
        query_horizon_turns=240, baselines_v2=baselines)
    cells["H1351"] = bool(proof.proven)
    return {
        "schema": R199_SCHEMA_VERSION,
        "seeds": list(seeds),
        "cells": dict(cells),
        "n_regimes": len(W79_MASC_V15_REGIMES),
        "all_pass": bool(all(bool(v) for v in cells.values())),
    }


__all__ = ["R199_SCHEMA_VERSION", "run_r199"]
