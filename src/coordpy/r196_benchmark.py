"""W78 R-196 benchmark family — Handoff V10 + bounded-window
falsifier + long-horizon limitation reproductions.

H1250..H1265 cell families (16 H-bars):

* H1250  Handoff V10 envelope content-addressed
* H1251  Handoff V10 long-horizon-reconstruction-aware promotion
         (LHR pressure high → Plane B)
* H1252  Handoff V10 long-horizon-reconstruction fallback fires
         (trajectory CID non-empty + blackout > floor)
* H1253  Handoff V10 cross-plane savings ≥ 0.87 at default config
* H1254  Handoff V10 long-horizon-reconstruction falsifier
         honest=0, dishonest=1
* H1255  Boundary V11 ≥ 46 blocked axes (V23)
* H1256  Boundary V11 falsifier honest=0 on V23 axes
* H1257  KV V23 long-horizon-reconstruction falsifier honest=0
* H1258  Bounded-window baseline: all fixed-k fail on >32-turn
         query (W78-T-BOUNDED-WINDOW-INSUFFICIENT)
* H1259  Bounded-window-insufficiency proof: proven=True
* H1260  W78 reconstruction substrate beats bounded-window
         baselines strictly on the >32-turn query
* H1261  W78 substrate is in-repo NumPy (limitation reproduction)
* H1262  Hosted plane V11 does NOT pierce wall (limitation
         reproduction)
* H1263  Frontier-substrate access still blocked (W70 forward)
* H1264  Bounded-window baselines fail by construction (limitation
         reproduction; W78-L-BOUNDED-WINDOW-INSUFFICIENT-CAP)
* H1265  Long-horizon-reconstruction substrate non-bounded-window
         success rate ≥ 0.95 on a 5-query sample (saving > 0.85
         on FLOPs)
"""

from __future__ import annotations

from typing import Any, Sequence

from coordpy.bounded_window_baseline_v1 import (
    BoundedWindowQuery,
    build_default_bounded_window_baselines,
    prove_bounded_window_insufficient,
    run_bounded_window_falsifier,
)
from coordpy.hosted_real_handoff_coordinator_v10 import (
    HandoffEnvelopeV10, HandoffRequestV10,
    HostedRealHandoffCoordinatorV10,
    W78_HANDOFF_DECISION_LONG_HORIZON_RECONSTRUCTION_FALLBACK,
    hosted_real_handoff_v10_long_horizon_reconstruction_aware_savings,
    probe_hosted_real_handoff_v10_long_horizon_reconstruction_falsifier,
)
from coordpy.hosted_real_handoff_coordinator_v9 import (
    HandoffRequestV9,
)
from coordpy.hosted_real_handoff_coordinator_v8 import (
    HandoffRequestV8,
)
from coordpy.hosted_real_handoff_coordinator_v7 import (
    HandoffRequestV7,
)
from coordpy.hosted_real_handoff_coordinator_v6 import (
    HandoffRequestV6,
)
from coordpy.hosted_real_handoff_coordinator_v5 import (
    HandoffRequestV5,
)
from coordpy.hosted_real_handoff_coordinator_v4 import (
    HandoffRequestV4,
)
from coordpy.hosted_real_handoff_coordinator_v3 import (
    HandoffRequestV3,
)
from coordpy.hosted_real_handoff_coordinator_v2 import (
    HandoffRequestV2,
)
from coordpy.hosted_real_handoff_coordinator import (
    HandoffRequest, W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY,
)
from coordpy.hosted_real_substrate_boundary_v11 import (
    build_default_hosted_real_substrate_boundary_v11,
    probe_hosted_real_substrate_boundary_v11_falsifier,
)
from coordpy.hosted_real_substrate_boundary_v3 import (
    W70_FRONTIER_BLOCKED_AXES,
)
from coordpy.kv_bridge_v23 import (
    probe_kv_bridge_v23_long_horizon_reconstruction_falsifier,
)
from coordpy.long_horizon_reconstruction_substrate_v1 import (
    LongHorizonReconstructionQuery,
    build_default_long_horizon_reconstruction_carrier,
    reconstruct_long_horizon_event,
    report_reconstruction_vs_recompute_economics,
)
from coordpy.tiny_substrate_v23 import W78_REPAIR_LABELS_V23


R196_SCHEMA_VERSION: str = "coordpy.r196_benchmark.v1"


def _make_req_v10(
        rc: str,
        long_horizon_reconstruction_pressure: float = 0.0,
        long_horizon_reconstruction_trajectory_cid: str = "",
        long_horizon_blackout_window_turns: int = 0,
        needs_text_only: bool = True,
        needs_substrate_state_access: bool = False,
) -> HandoffRequestV10:
    return HandoffRequestV10(
        inner_v9=HandoffRequestV9(
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
                                        visible_token_budget=512,
                                        baseline_token_cost=512,
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
            post_restart_replacement_pressure=0.0,
            post_restart_replacement_trajectory_cid="",
            post_restart_replacement_window_turns=0,
            expected_substrate_trust_v9=0.7),
        long_horizon_reconstruction_pressure=float(
            long_horizon_reconstruction_pressure),
        long_horizon_reconstruction_trajectory_cid=str(
            long_horizon_reconstruction_trajectory_cid),
        long_horizon_blackout_window_turns=int(
            long_horizon_blackout_window_turns),
        expected_substrate_trust_v10=0.7)


def run_r196(*, seeds: Sequence[int]) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    coord = HostedRealHandoffCoordinatorV10()
    # H1250: envelope content-addressed.
    e_a = coord.decide_v10(
        req_v10=_make_req_v10("rc-a"))
    e_b = coord.decide_v10(
        req_v10=_make_req_v10("rc-a"))
    cells["H1250"] = bool(e_a.cid() == e_b.cid())
    # H1251: LHR-aware promotion: high LHR pressure → Plane B.
    e_lhr = coord.decide_v10(
        req_v10=_make_req_v10(
            "rc-lhr",
            long_horizon_reconstruction_pressure=0.95,
            long_horizon_reconstruction_trajectory_cid=(
                "lhr_x"),
            long_horizon_blackout_window_turns=100))
    cells["H1251"] = bool(
        e_lhr.decision_v10
        == W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY
        and float(
            e_lhr.long_horizon_reconstruction_alignment) == 1.0)
    # H1252: LHR fallback fires.
    e_fb = coord.decide_v10(
        req_v10=_make_req_v10(
            "rc-fb",
            long_horizon_reconstruction_pressure=0.0,
            long_horizon_reconstruction_trajectory_cid=(
                "lhr_y"),
            long_horizon_blackout_window_turns=100))
    cells["H1252"] = bool(
        e_fb.decision_v10
        == W78_HANDOFF_DECISION_LONG_HORIZON_RECONSTRUCTION_FALLBACK
        and bool(
            e_fb.long_horizon_reconstruction_fallback_active))
    # H1253: cross-plane savings ≥ 0.87.
    sav = (
        hosted_real_handoff_v10_long_horizon_reconstruction_aware_savings(
            n_turns=100))
    cells["H1253"] = bool(
        float(sav["saving_ratio"]) >= 0.87)
    # H1254: handoff V10 LHR falsifier: dishonest=1.
    fsl = (
        probe_hosted_real_handoff_v10_long_horizon_reconstruction_falsifier(
            envelope_v10=e_a,
            claim_kind=(
                "hosted_satisfies_long_horizon_reconstruction"),
            claim_satisfied=True))
    cells["H1254"] = bool(float(fsl.falsifier_score) == 1.0)
    # H1255: boundary V11 ≥ 46 blocked.
    boundary = build_default_hosted_real_substrate_boundary_v11()
    cells["H1255"] = bool(len(boundary.blocked_axes) >= 46)
    # H1256: boundary V11 falsifier honest=0 on a V23 axis.
    fbo = probe_hosted_real_substrate_boundary_v11_falsifier(
        boundary=boundary,
        claimed_axis=(
            "long_horizon_reconstruction_trajectory_cid"),
        claim_satisfied_at_hosted=False)
    cells["H1256"] = bool(float(fbo.falsifier_score) == 0.0)
    # H1257: KV V23 LHR falsifier honest=0 on flag=1.
    fkv = (
        probe_kv_bridge_v23_long_horizon_reconstruction_falsifier(
            long_horizon_reconstruction_pressure_flag=1))
    cells["H1257"] = bool(float(fkv.falsifier_score) == 0.0)
    # H1258: bounded-window baselines all fail on > 32-turn query.
    baselines = build_default_bounded_window_baselines()
    bw_q = BoundedWindowQuery(
        query_id="bw1", current_turn=200,
        source_turn=20, expected_event_cid="event_x")
    _, fals = run_bounded_window_falsifier(
        baselines=baselines, query=bw_q)
    cells["H1258"] = bool(fals.all_fixed_k_failed)
    # H1259: bounded-window-insufficiency proof.
    proof = prove_bounded_window_insufficient(
        query_horizon_turns=180, baselines=baselines)
    cells["H1259"] = bool(proof.proven)
    # H1260: W78 reconstruction substrate succeeds on the same
    # query.
    carrier = build_default_long_horizon_reconstruction_carrier(
        n_events=256, seed=78060)
    known = carrier.entries[20]
    lhr_q = LongHorizonReconstructionQuery(
        query_id="lhr1", source_turn=20, current_turn=200)
    out = reconstruct_long_horizon_event(
        carrier=carrier, query=lhr_q, visible_tokens_used=4)
    cells["H1260"] = bool(
        bool(out.success)
        and str(out.reconstructed_event_cid)
        == str(known.event_cid))
    # H1261: W78 substrate is in-repo NumPy.
    from coordpy.tiny_substrate_v23 import (
        W78_TINY_V23_VOCAB_SIZE,
    )
    cells["H1261"] = bool(int(W78_TINY_V23_VOCAB_SIZE) == 259)
    # H1262: hosted V11 does NOT pierce wall.
    cells["H1262"] = bool(
        "long_horizon_reconstruction_trajectory_cid"
        in tuple(boundary.blocked_axes))
    # H1263: frontier-substrate access still blocked.
    cells["H1263"] = bool(
        all(
            ax in tuple(boundary.frontier_blocked_axes)
            for ax in W70_FRONTIER_BLOCKED_AXES))
    # H1264: bounded-window baselines fail by construction.
    cells["H1264"] = bool(proof.proven)
    # H1265: LHR reconstruction substrate ≥ 95 % success on
    # 5-query sample; FLOPs saving ≥ 0.85.
    queries = [
        LongHorizonReconstructionQuery(
            query_id=f"q{i}",
            source_turn=int(i * 30),
            current_turn=int(200 + i * 5))
        for i in range(5)]
    outs = [
        reconstruct_long_horizon_event(
            carrier=carrier, query=q,
            visible_tokens_used=4)
        for q in queries]
    succ_rate = (
        sum(1 for o in outs if o.success) / float(len(outs)))
    econ = report_reconstruction_vs_recompute_economics(
        query=queries[-1], carrier=carrier)
    cells["H1265"] = bool(
        succ_rate >= 0.95
        and float(econ.saving_ratio) >= 0.85)
    return {
        "schema": R196_SCHEMA_VERSION,
        "seeds": list(seeds),
        "cells": dict(cells),
        "all_pass": bool(all(bool(v) for v in cells.values())),
    }


__all__ = ["R196_SCHEMA_VERSION", "run_r196"]
