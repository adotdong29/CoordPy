"""W76 R-188 benchmark family — Handoff V8 + Falsifier + Limitation.

The W76 falsifier-and-limitation-reproduction family + Plane A↔B
chain-then-restart-aware handoff family.

H1150..H1163 cell families (14 H-bars):

* H1150  Handoff V8 envelope is content-addressed
* H1151  Handoff V8 text-only routes to plane A
* H1152  Handoff V8 substrate-only routes to plane B
* H1153  Handoff V8 chain_then_restart_pressure ≥ floor +
         substrate_trust ≥ floor promotes to real_substrate_only
         with chain_then_restart_alignment = 1.0
* H1154  Handoff V8 chain_then_restart_trajectory_cid +
         post_compound_chain_restart_window > 0 falls back to
         chain_then_restart_fallback
* H1155  Handoff V8 cross-plane savings ≥ 85 %
* H1156  Handoff V8 chain-then-restart falsifier honest=0,
         dishonest=1
* H1157  Boundary V9 falsifier honest=0 / dishonest=1
* H1158  KV V21 chain-then-restart-pressure falsifier honest = 0
* H1159  MASC V12 chain-then-restart regime
         (restart_after_compound_chain_repair_under_budget) ≥
         50 % strict-beat
* H1160  W76 substrate is in-repo NumPy (limitation reproduction;
         23 layers / 259 vocab)
* H1161  W76 hosted control plane V9 does NOT pierce wall
         (limitation reproduction; ≥ 40 blocked axes)
* H1162  No-version-bump invariant: ``coordpy.__version__ ==
         "0.5.20"`` and ``SDK_VERSION == "coordpy.sdk.v3.43"``
* H1163  Frontier substrate access still blocked (V9 frontier-
         blocked axes set non-empty and unchanged from W70)
"""

from __future__ import annotations

from typing import Any, Sequence

from coordpy import SDK_VERSION, __version__
from coordpy.hosted_real_handoff_coordinator import HandoffRequest
from coordpy.hosted_real_handoff_coordinator_v2 import (
    HandoffRequestV2,
)
from coordpy.hosted_real_handoff_coordinator_v3 import (
    HandoffRequestV3,
)
from coordpy.hosted_real_handoff_coordinator_v4 import (
    HandoffRequestV4,
)
from coordpy.hosted_real_handoff_coordinator import (
    W69_HANDOFF_DECISION_HOSTED_ONLY,
    W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY,
)
from coordpy.hosted_real_handoff_coordinator_v5 import (
    HandoffRequestV5,
)
from coordpy.hosted_real_handoff_coordinator_v6 import (
    HandoffRequestV6,
)
from coordpy.hosted_real_handoff_coordinator_v7 import (
    HandoffRequestV7,
)
from coordpy.hosted_real_handoff_coordinator_v8 import (
    HandoffRequestV8, HostedRealHandoffCoordinatorV8,
    W76_HANDOFF_DECISION_CHAIN_THEN_RESTART_FALLBACK,
    hosted_real_handoff_v8_chain_then_restart_aware_savings,
    probe_hosted_real_handoff_v8_chain_then_restart_falsifier,
)
from coordpy.hosted_real_substrate_boundary_v3 import (
    W70_FRONTIER_BLOCKED_AXES,
)
from coordpy.hosted_real_substrate_boundary_v9 import (
    W76_FRONTIER_BLOCKED_AXES,
    build_default_hosted_real_substrate_boundary_v9,
    probe_hosted_real_substrate_boundary_v9_falsifier,
)
from coordpy.kv_bridge_v21 import (
    probe_kv_bridge_v21_chain_then_restart_falsifier,
)
from coordpy.multi_agent_substrate_coordinator_v12 import (
    MultiAgentSubstrateCoordinatorV12,
    W76_MASC_V12_REGIME_CHAIN_THEN_RESTART,
)
from coordpy.tiny_substrate_v21 import (
    W76_DEFAULT_V21_N_LAYERS, W76_TINY_V21_VOCAB_SIZE,
)


R188_SCHEMA_VERSION: str = "coordpy.r188_benchmark.v1"


def _req_v8(*, rc: str,
            compound_chain_then_restart_pressure: float = 0.0,
            compound_chain_then_restart_trajectory_cid: str = "",
            post_compound_chain_restart_window_turns: int = 0,
            needs_text_only: bool = True,
            needs_substrate_state_access: bool = False,
            visible_token_budget: int = 256,
            baseline_token_cost: int = 512,
            ) -> HandoffRequestV8:
    return HandoffRequestV8(
        inner_v7=HandoffRequestV7(
            inner_v6=HandoffRequestV6(
                inner_v5=HandoffRequestV5(
                    inner_v4=HandoffRequestV4(
                        inner_v3=HandoffRequestV3(
                            inner_v2=HandoffRequestV2(
                                inner_v1=HandoffRequest(
                                    request_cid=str(rc),
                                    needs_text_only=bool(
                                        needs_text_only),
                                    needs_substrate_state_access=bool(
                                        needs_substrate_state_access)),
                                visible_token_budget=int(
                                    visible_token_budget),
                                baseline_token_cost=int(
                                    baseline_token_cost),
                                dominant_repair_label=0),
                            restart_pressure=0.0,
                            delayed_repair_trajectory_cid="",
                            delay_turns=0,
                            expected_substrate_trust=0.7),
                        rejoin_pressure=0.0,
                        restart_repair_trajectory_cid="",
                        rejoin_lag_turns=0,
                        expected_substrate_trust_v4=0.7),
                    replacement_pressure=0.0,
                    replacement_repair_trajectory_cid="",
                    replacement_lag_turns=0,
                    expected_substrate_trust_v5=0.7),
                compound_pressure=0.0,
                compound_repair_trajectory_cid="",
                compound_window_turns=0,
                expected_substrate_trust_v6=0.7),
            compound_chain_pressure=0.0,
            compound_chain_repair_trajectory_cid="",
            compound_chain_window_turns=0,
            expected_substrate_trust_v7=0.7),
        compound_chain_then_restart_pressure=float(
            compound_chain_then_restart_pressure),
        compound_chain_then_restart_trajectory_cid=str(
            compound_chain_then_restart_trajectory_cid),
        post_compound_chain_restart_window_turns=int(
            post_compound_chain_restart_window_turns),
        expected_substrate_trust_v8=0.7)


def run_r188(*, seeds: Sequence[int]) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    coord = HostedRealHandoffCoordinatorV8()
    # H1150: envelope content-addressed.
    env_a = coord.decide_v8(req_v8=_req_v8(rc="a"))
    env_b = coord.decide_v8(req_v8=_req_v8(rc="b"))
    cells["H1150"] = bool(env_a.cid() != env_b.cid())
    # H1151: text-only routes to plane A.
    env_text = coord.decide_v8(req_v8=_req_v8(rc="text"))
    cells["H1151"] = bool(
        str(env_text.decision_v8)
        == W69_HANDOFF_DECISION_HOSTED_ONLY)
    # H1152: substrate-only routes to plane B.
    env_sub = coord.decide_v8(
        req_v8=_req_v8(
            rc="sub",
            needs_text_only=False,
            needs_substrate_state_access=True))
    cells["H1152"] = bool(
        str(env_sub.decision_v8)
        == W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY)
    # H1153: chain-then-restart promotion with alignment 1.0.
    env_ctr = coord.decide_v8(
        req_v8=_req_v8(
            rc="ctr",
            compound_chain_then_restart_pressure=0.9,
            compound_chain_then_restart_trajectory_cid="cid",
            post_compound_chain_restart_window_turns=4))
    cells["H1153"] = bool(
        str(env_ctr.decision_v8)
        == W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY
        and float(env_ctr.compound_chain_then_restart_alignment)
        == 1.0)
    # H1154: chain-then-restart-fallback fires.
    env_ctr_fb = coord.decide_v8(
        req_v8=_req_v8(
            rc="ctr-fb",
            compound_chain_then_restart_pressure=0.0,
            compound_chain_then_restart_trajectory_cid="cid",
            post_compound_chain_restart_window_turns=4))
    cells["H1154"] = bool(
        str(env_ctr_fb.decision_v8)
        == W76_HANDOFF_DECISION_CHAIN_THEN_RESTART_FALLBACK
        and bool(
            env_ctr_fb.chain_then_restart_fallback_active))
    # H1155: cross-plane savings ≥ 85 %.
    sav = hosted_real_handoff_v8_chain_then_restart_aware_savings(
        n_turns=100)
    cells["H1155"] = bool(float(sav["saving_ratio"]) >= 0.85)
    # H1156: chain-then-restart falsifier honest=0, dishonest=1.
    f_honest = (
        probe_hosted_real_handoff_v8_chain_then_restart_falsifier(
            envelope_v8=env_ctr,
            claim_satisfied=True))
    f_dishonest = (
        probe_hosted_real_handoff_v8_chain_then_restart_falsifier(
            envelope_v8=env_text,
            claim_satisfied=True))
    cells["H1156"] = bool(
        float(f_honest.falsifier_score) == 0.0
        and float(f_dishonest.falsifier_score) == 1.0)
    # H1157: boundary V9 falsifier.
    boundary = build_default_hosted_real_substrate_boundary_v9()
    f_b_honest = (
        probe_hosted_real_substrate_boundary_v9_falsifier(
            boundary=boundary,
            claimed_axis=(
                "compound_chain_then_restart_trajectory_cid"),
            claim_satisfied_at_hosted=False))
    f_b_dishonest = (
        probe_hosted_real_substrate_boundary_v9_falsifier(
            boundary=boundary,
            claimed_axis=(
                "compound_chain_then_restart_trajectory_cid"),
            claim_satisfied_at_hosted=True))
    cells["H1157"] = bool(
        float(f_b_honest.falsifier_score) == 0.0
        and float(f_b_dishonest.falsifier_score) == 1.0)
    # H1158: KV V21 chain-then-restart-pressure falsifier honest.
    fk = probe_kv_bridge_v21_chain_then_restart_falsifier(
        chain_then_restart_pressure_flag=1)
    cells["H1158"] = bool(float(fk.falsifier_score) == 0.0)
    # H1159: MASC V12 chain-then-restart regime ≥ 50 % strict beat.
    masc = MultiAgentSubstrateCoordinatorV12()
    _, agg = masc.run_batch(
        seeds=list(seeds),
        regime=W76_MASC_V12_REGIME_CHAIN_THEN_RESTART)
    cells["H1159"] = bool(agg.v21_beats_v20_rate >= 0.5)
    # H1160: W76 substrate is in-repo NumPy.
    cells["H1160"] = bool(
        int(W76_DEFAULT_V21_N_LAYERS) == 23
        and int(W76_TINY_V21_VOCAB_SIZE) == 259)
    # H1161: Hosted V9 does NOT pierce wall.
    cells["H1161"] = bool(len(boundary.blocked_axes) >= 40)
    # H1162: no-version-bump invariant.
    cells["H1162"] = bool(
        __version__ == "1.2.1"
        and SDK_VERSION == "coordpy.sdk.v3.43")
    # H1163: frontier-blocked axes unchanged from W70.
    cells["H1163"] = bool(
        len(W76_FRONTIER_BLOCKED_AXES) > 0
        and tuple(W76_FRONTIER_BLOCKED_AXES)
        == tuple(W70_FRONTIER_BLOCKED_AXES))
    return {
        "schema": R188_SCHEMA_VERSION,
        "n_seeds": int(len(seeds)),
        "cells": cells,
        "all_pass": bool(all(cells.values())),
    }


__all__ = [
    "R188_SCHEMA_VERSION",
    "run_r188",
]
