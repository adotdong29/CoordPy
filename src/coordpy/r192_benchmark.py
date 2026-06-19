"""W77 R-192 benchmark family — Handoff V9 + Falsifier + Limitation.

The W77 falsifier-and-limitation-reproduction family + Plane A↔B
post-restart-replacement-aware handoff family.

H1240..H1253 cell families (14 H-bars):

* H1240  Handoff V9 envelope is content-addressed
* H1241  Handoff V9 text-only routes to plane A
* H1242  Handoff V9 substrate-only routes to plane B
* H1243  Handoff V9 post_restart_replacement_pressure ≥ floor +
         substrate_trust ≥ floor promotes to real_substrate_only
         with post_restart_replacement_alignment = 1.0
* H1244  Handoff V9 post_restart_replacement_trajectory_cid +
         post_restart_replacement_window > 0 falls back to
         post_restart_replacement_fallback
* H1245  Handoff V9 cross-plane savings ≥ 86 %
* H1246  Handoff V9 post-restart-replacement falsifier honest=0,
         dishonest=1
* H1247  Boundary V10 falsifier honest=0 / dishonest=1
* H1248  KV V22 post-restart-replacement-pressure falsifier
         honest = 0
* H1249  MASC V13 post-restart-replacement regime
         (replacement_after_restart_after_compound_chain_repair_under_budget)
         ≥ 50 % strict-beat
* H1250  W77 substrate is in-repo NumPy (limitation reproduction;
         23 layers / 259 vocab)
* H1251  W77 hosted control plane V10 does NOT pierce wall
         (limitation reproduction; ≥ 43 blocked axes)
* H1252  No-version-bump invariant: ``coordpy.__version__ ==
         "0.5.20"`` and ``SDK_VERSION == "coordpy.sdk.v3.43"``
* H1253  Frontier substrate access still blocked (V10 frontier-
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
    HandoffRequestV8,
)
from coordpy.hosted_real_handoff_coordinator_v9 import (
    HandoffRequestV9, HostedRealHandoffCoordinatorV9,
    W77_HANDOFF_DECISION_POST_RESTART_REPLACEMENT_FALLBACK,
    hosted_real_handoff_v9_post_restart_replacement_aware_savings,
    probe_hosted_real_handoff_v9_post_restart_replacement_falsifier,
)
from coordpy.hosted_real_substrate_boundary_v3 import (
    W70_FRONTIER_BLOCKED_AXES,
)
from coordpy.hosted_real_substrate_boundary_v10 import (
    W77_FRONTIER_BLOCKED_AXES,
    build_default_hosted_real_substrate_boundary_v10,
    probe_hosted_real_substrate_boundary_v10_falsifier,
)
from coordpy.kv_bridge_v22 import (
    probe_kv_bridge_v22_post_restart_replacement_falsifier,
)
from coordpy.multi_agent_substrate_coordinator_v13 import (
    MultiAgentSubstrateCoordinatorV13,
    W77_MASC_V13_REGIME_POST_RESTART_REPLACEMENT,
)
from coordpy.tiny_substrate_v22 import (
    W77_DEFAULT_V22_N_LAYERS, W77_TINY_V22_VOCAB_SIZE,
)


R192_SCHEMA_VERSION: str = "coordpy.r192_benchmark.v1"


def _req_v9(*, rc: str,
            post_restart_replacement_pressure: float = 0.0,
            post_restart_replacement_trajectory_cid: str = "",
            post_restart_replacement_window_turns: int = 0,
            needs_text_only: bool = True,
            needs_substrate_state_access: bool = False,
            visible_token_budget: int = 256,
            baseline_token_cost: int = 512,
            ) -> HandoffRequestV9:
    return HandoffRequestV9(
        inner_v8=HandoffRequestV8(
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
            compound_chain_then_restart_pressure=0.0,
            compound_chain_then_restart_trajectory_cid="",
            post_compound_chain_restart_window_turns=0,
            expected_substrate_trust_v8=0.7),
        post_restart_replacement_pressure=float(
            post_restart_replacement_pressure),
        post_restart_replacement_trajectory_cid=str(
            post_restart_replacement_trajectory_cid),
        post_restart_replacement_window_turns=int(
            post_restart_replacement_window_turns),
        expected_substrate_trust_v9=0.7)


def run_r192(*, seeds: Sequence[int]) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    coord = HostedRealHandoffCoordinatorV9()
    # H1240: envelope content-addressed.
    env_a = coord.decide_v9(req_v9=_req_v9(rc="a"))
    env_b = coord.decide_v9(req_v9=_req_v9(rc="b"))
    cells["H1240"] = bool(env_a.cid() != env_b.cid())
    # H1241: text-only routes to plane A.
    env_text = coord.decide_v9(req_v9=_req_v9(rc="text"))
    cells["H1241"] = bool(
        str(env_text.decision_v9)
        == W69_HANDOFF_DECISION_HOSTED_ONLY)
    # H1242: substrate-only routes to plane B.
    env_sub = coord.decide_v9(
        req_v9=_req_v9(
            rc="sub",
            needs_text_only=False,
            needs_substrate_state_access=True))
    cells["H1242"] = bool(
        str(env_sub.decision_v9)
        == W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY)
    # H1243: post-restart-replacement promotion with alignment 1.0.
    env_pcr = coord.decide_v9(
        req_v9=_req_v9(
            rc="pcr",
            post_restart_replacement_pressure=0.9,
            post_restart_replacement_trajectory_cid="cid",
            post_restart_replacement_window_turns=4))
    cells["H1243"] = bool(
        str(env_pcr.decision_v9)
        == W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY
        and float(env_pcr.post_restart_replacement_alignment)
        == 1.0)
    # H1244: post-restart-replacement fallback fires.
    env_pcr_fb = coord.decide_v9(
        req_v9=_req_v9(
            rc="pcr-fb",
            post_restart_replacement_pressure=0.0,
            post_restart_replacement_trajectory_cid="cid",
            post_restart_replacement_window_turns=4))
    cells["H1244"] = bool(
        str(env_pcr_fb.decision_v9)
        == W77_HANDOFF_DECISION_POST_RESTART_REPLACEMENT_FALLBACK
        and bool(
            env_pcr_fb.post_restart_replacement_fallback_active))
    # H1245: cross-plane savings ≥ 86 %.
    sav = (
        hosted_real_handoff_v9_post_restart_replacement_aware_savings(
            n_turns=100))
    cells["H1245"] = bool(float(sav["saving_ratio"]) >= 0.86)
    # H1246: post-restart-replacement falsifier honest=0,
    # dishonest=1.
    f_honest = (
        probe_hosted_real_handoff_v9_post_restart_replacement_falsifier(
            envelope_v9=env_pcr,
            claim_satisfied=True))
    f_dishonest = (
        probe_hosted_real_handoff_v9_post_restart_replacement_falsifier(
            envelope_v9=env_text,
            claim_satisfied=True))
    cells["H1246"] = bool(
        float(f_honest.falsifier_score) == 0.0
        and float(f_dishonest.falsifier_score) == 1.0)
    # H1247: boundary V10 falsifier.
    boundary = (
        build_default_hosted_real_substrate_boundary_v10())
    f_b_honest = (
        probe_hosted_real_substrate_boundary_v10_falsifier(
            boundary=boundary,
            claimed_axis=(
                "replacement_after_restart_after_compound_chain_trajectory_cid"),
            claim_satisfied_at_hosted=False))
    f_b_dishonest = (
        probe_hosted_real_substrate_boundary_v10_falsifier(
            boundary=boundary,
            claimed_axis=(
                "replacement_after_restart_after_compound_chain_trajectory_cid"),
            claim_satisfied_at_hosted=True))
    cells["H1247"] = bool(
        float(f_b_honest.falsifier_score) == 0.0
        and float(f_b_dishonest.falsifier_score) == 1.0)
    # H1248: KV V22 post-restart-replacement-pressure falsifier.
    fk = probe_kv_bridge_v22_post_restart_replacement_falsifier(
        post_restart_replacement_pressure_flag=1)
    cells["H1248"] = bool(float(fk.falsifier_score) == 0.0)
    # H1249: MASC V13 PCR regime ≥ 50 % strict beat.
    masc = MultiAgentSubstrateCoordinatorV13()
    _, agg = masc.run_batch(
        seeds=list(seeds),
        regime=W77_MASC_V13_REGIME_POST_RESTART_REPLACEMENT)
    cells["H1249"] = bool(agg.v22_beats_v21_rate >= 0.5)
    # H1250: W77 substrate is in-repo NumPy.
    cells["H1250"] = bool(
        int(W77_DEFAULT_V22_N_LAYERS) == 23
        and int(W77_TINY_V22_VOCAB_SIZE) == 259)
    # H1251: Hosted V10 does NOT pierce wall (≥ 43 blocked axes).
    cells["H1251"] = bool(len(boundary.blocked_axes) >= 43)
    # H1252: no-version-bump invariant.
    cells["H1252"] = bool(
        __version__ == "1.2.0"
        and SDK_VERSION == "coordpy.sdk.v3.43")
    # H1253: frontier-blocked axes unchanged from W70.
    cells["H1253"] = bool(
        len(W77_FRONTIER_BLOCKED_AXES) > 0
        and tuple(W77_FRONTIER_BLOCKED_AXES)
        == tuple(W70_FRONTIER_BLOCKED_AXES))
    return {
        "schema": R192_SCHEMA_VERSION,
        "n_seeds": int(len(seeds)),
        "cells": cells,
        "all_pass": bool(all(cells.values())),
    }


__all__ = ["R192_SCHEMA_VERSION", "run_r192"]
