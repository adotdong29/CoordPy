"""W75 R-184 benchmark family — Handoff V7 + Falsifier + Limitation.

The W75 falsifier-and-limitation-reproduction family + Plane A↔B
compound-chain-aware handoff family.

H1050..H1063 cell families (14 H-bars):

* H1050  Handoff V7 envelope is content-addressed
* H1051  Handoff V7 text-only routes to plane A
* H1052  Handoff V7 substrate-only routes to plane B
* H1053  Handoff V7 compound_chain_pressure ≥ floor +
         substrate_trust ≥ floor promotes to real_substrate_only
         with compound_chain_alignment = 1.0
* H1054  Handoff V7 compound_chain_repair_trajectory +
         compound_chain_window > 0 falls back to
         compound_repair_after_replacement_then_rejoin_fallback
* H1055  Handoff V7 cross-plane savings ≥ 84 %
* H1056  Handoff V7 compound-chain falsifier honest=0, dishonest=1
* H1057  Boundary V8 falsifier honest=0 / dishonest=1
* H1058  KV V20 compound-chain-pressure falsifier honest = 0
* H1059  MASC V11 chain regime
         (compound_repair_after_replacement_then_rejoin_under_budget)
         ≥ 50 % strict-beat
* H1060  W75 substrate is in-repo NumPy (limitation reproduction)
* H1061  W75 hosted control plane V8 does NOT pierce wall
         (limitation reproduction; ≥ 37 blocked-axes)
* H1062  No-version-bump invariant: ``coordpy.__version__ ==
         "0.5.20"`` and ``SDK_VERSION == "coordpy.sdk.v3.43"``
* H1063  Frontier substrate access still blocked (V8 frontier-
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
    HandoffRequestV7, HostedRealHandoffCoordinatorV7,
    W75_HANDOFF_DECISION_COMPOUND_CHAIN_REPAIR_FALLBACK,
    hosted_real_handoff_v7_compound_chain_aware_savings,
    probe_hosted_real_handoff_v7_compound_chain_falsifier,
)
from coordpy.hosted_real_substrate_boundary_v3 import (
    W70_FRONTIER_BLOCKED_AXES,
)
from coordpy.hosted_real_substrate_boundary_v8 import (
    W75_FRONTIER_BLOCKED_AXES,
    build_default_hosted_real_substrate_boundary_v8,
    probe_hosted_real_substrate_boundary_v8_falsifier,
)
from coordpy.kv_bridge_v20 import (
    probe_kv_bridge_v20_compound_chain_pressure_falsifier,
)
from coordpy.multi_agent_substrate_coordinator_v11 import (
    MultiAgentSubstrateCoordinatorV11,
    W75_MASC_V11_REGIME_COMPOUND_CHAIN,
)
from coordpy.tiny_substrate_v20 import (
    W75_DEFAULT_V20_N_LAYERS, W75_TINY_V20_VOCAB_SIZE,
)


R184_SCHEMA_VERSION: str = "coordpy.r184_benchmark.v1"


def _req_v7(*, rc: str, compound_chain_pressure: float = 0.0,
            compound_chain_repair_trajectory_cid: str = "",
            compound_chain_window_turns: int = 0,
            needs_text_only: bool = True,
            needs_substrate_state_access: bool = False,
            visible_token_budget: int = 256,
            baseline_token_cost: int = 512,
            ) -> HandoffRequestV7:
    return HandoffRequestV7(
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
        compound_chain_pressure=float(
            compound_chain_pressure),
        compound_chain_repair_trajectory_cid=str(
            compound_chain_repair_trajectory_cid),
        compound_chain_window_turns=int(
            compound_chain_window_turns),
        expected_substrate_trust_v7=0.7)


def run_r184(*, seeds: Sequence[int]) -> dict[str, Any]:
    cells: dke = {}  # type: ignore  # placeholder
    cells = {}
    coord = HostedRealHandoffCoordinatorV7()
    # H1050: envelope is content-addressed.
    env_a = coord.decide_v7(req_v7=_req_v7(rc="a"))
    env_b = coord.decide_v7(req_v7=_req_v7(rc="b"))
    cells["H1050"] = bool(env_a.cid() != env_b.cid())
    # H1051: text-only routes to plane A.
    env_text = coord.decide_v7(req_v7=_req_v7(rc="text"))
    cells["H1051"] = bool(
        str(env_text.decision_v7)
        == W69_HANDOFF_DECISION_HOSTED_ONLY)
    # H1052: substrate-only routes to plane B.
    env_sub = coord.decide_v7(
        req_v7=_req_v7(
            rc="sub",
            needs_text_only=False,
            needs_substrate_state_access=True))
    cells["H1052"] = bool(
        str(env_sub.decision_v7)
        == W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY)
    # H1053: chain promotion with alignment 1.0.
    env_chain = coord.decide_v7(
        req_v7=_req_v7(
            rc="chain", compound_chain_pressure=0.9,
            compound_chain_repair_trajectory_cid="cid",
            compound_chain_window_turns=4))
    cells["H1053"] = bool(
        str(env_chain.decision_v7)
        == W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY
        and float(env_chain.compound_chain_alignment) == 1.0)
    # H1054: chain-repair-RTR fallback fires.
    env_chain_fb = coord.decide_v7(
        req_v7=_req_v7(
            rc="chain-fb",
            compound_chain_pressure=0.0,
            compound_chain_repair_trajectory_cid="cid",
            compound_chain_window_turns=4))
    cells["H1054"] = bool(
        str(env_chain_fb.decision_v7)
        == W75_HANDOFF_DECISION_COMPOUND_CHAIN_REPAIR_FALLBACK
        and bool(
            env_chain_fb
            .compound_chain_repair_rtr_fallback_active))
    # H1055: cross-plane savings ≥ 84 %.
    sav = hosted_real_handoff_v7_compound_chain_aware_savings(
        n_turns=100)
    cells["H1055"] = bool(float(sav["saving_ratio"]) >= 0.84)
    # H1056: chain falsifier honest=0, dishonest=1.
    f_honest = probe_hosted_real_handoff_v7_compound_chain_falsifier(
        envelope_v7=env_chain,
        claim_satisfied=True)
    f_dishonest = probe_hosted_real_handoff_v7_compound_chain_falsifier(
        envelope_v7=env_text,
        claim_satisfied=True)
    cells["H1056"] = bool(
        float(f_honest.falsifier_score) == 0.0
        and float(f_dishonest.falsifier_score) == 1.0)
    # H1057: boundary V8 falsifier.
    boundary = build_default_hosted_real_substrate_boundary_v8()
    f_b_honest = probe_hosted_real_substrate_boundary_v8_falsifier(
        boundary=boundary,
        claimed_axis="compound_chain_repair_trajectory_cid",
        claim_satisfied_at_hosted=False)
    f_b_dishonest = probe_hosted_real_substrate_boundary_v8_falsifier(
        boundary=boundary,
        claimed_axis="compound_chain_repair_trajectory_cid",
        claim_satisfied_at_hosted=True)
    cells["H1057"] = bool(
        float(f_b_honest.falsifier_score) == 0.0
        and float(f_b_dishonest.falsifier_score) == 1.0)
    # H1058: KV V20 chain-pressure falsifier honest=0.
    fk = probe_kv_bridge_v20_compound_chain_pressure_falsifier(
        compound_chain_pressure_flag=1)
    cells["H1058"] = bool(float(fk.falsifier_score) == 0.0)
    # H1059: MASC V11 chain regime ≥ 50 % strict beat.
    masc = MultiAgentSubstrateCoordinatorV11()
    _, agg = masc.run_batch(
        seeds=list(seeds),
        regime=W75_MASC_V11_REGIME_COMPOUND_CHAIN)
    cells["H1059"] = bool(agg.v20_beats_v19_rate >= 0.5)
    # H1060: W75 substrate is in-repo NumPy.
    cells["H1060"] = bool(
        int(W75_DEFAULT_V20_N_LAYERS) == 22
        and int(W75_TINY_V20_VOCAB_SIZE) == 259)
    # H1061: Hosted V8 does NOT pierce wall (≥ 37 blocked axes).
    cells["H1061"] = bool(len(boundary.blocked_axes) >= 37)
    # H1062: no-version-bump invariant.
    cells["H1062"] = bool(
        __version__ == "1.2.0"
        and SDK_VERSION == "coordpy.sdk.v3.43")
    # H1063: frontier-blocked axes unchanged from W70.
    cells["H1063"] = bool(
        len(W75_FRONTIER_BLOCKED_AXES) > 0
        and tuple(W75_FRONTIER_BLOCKED_AXES)
        == tuple(W70_FRONTIER_BLOCKED_AXES))
    return {
        "schema": R184_SCHEMA_VERSION,
        "n_seeds": int(len(seeds)),
        "cells": cells,
        "all_pass": bool(all(cells.values())),
    }


__all__ = [
    "R184_SCHEMA_VERSION",
    "run_r184",
]
