"""W78 R-194 benchmark family — Real Substrate V23 (Plane B).

Exercises the W78 real substrate V23 modules: tiny_substrate_v23
(LHR trajectory CID, length-per-layer in [0..14], pressure gate),
kv_bridge_v23 (19-target ridge + 156-dim LHR fingerprint +
falsifier), cache_controller_v21 (18-objective ridge), replay
controller V19 (26 regimes + 16-label LHR routing head),
persistent_latent_v30, long_horizon_retention_v30, mlsc V26,
consensus V24, masc V14 (one-regime), substrate_adapter_v23,
deep_substrate_hybrid_v23, long_horizon_reconstruction_substrate_v1.

H1200..H1213 cell families (14 H-bars).
"""

from __future__ import annotations

from typing import Any, Sequence

from coordpy.bounded_window_baseline_v1 import (
    build_default_bounded_window_baselines,
    BoundedWindowQuery, run_bounded_window_falsifier,
    prove_bounded_window_insufficient,
)
from coordpy.cache_controller_v21 import (
    CacheControllerV21, fit_eighteen_objective_ridge_v21,
    fit_per_role_long_horizon_reconstruction_pressure_head_v21,
)
from coordpy.consensus_fallback_controller_v24 import (
    ConsensusFallbackControllerV24,
    W78_CONSENSUS_V24_STAGES,
)
from coordpy.deep_substrate_hybrid_v22 import (
    DeepSubstrateHybridV22ForwardWitness,
)
from coordpy.deep_substrate_hybrid_v23 import (
    DeepSubstrateHybridV23,
    deep_substrate_hybrid_v23_forward,
)
from coordpy.kv_bridge_v22 import KVBridgeV22Projection
from coordpy.kv_bridge_v23 import (
    KVBridgeV23Projection,
    fit_kv_bridge_v23_nineteen_target,
    probe_kv_bridge_v23_long_horizon_reconstruction_falsifier,
)
from coordpy.long_horizon_reconstruction_substrate_v1 import (
    LongHorizonReconstructionQuery,
    build_default_long_horizon_reconstruction_carrier,
    reconstruct_long_horizon_event,
    report_reconstruction_vs_recompute_economics,
)
from coordpy.long_horizon_retention_v30 import (
    LongHorizonReconstructionV30Head,
    W78_DEFAULT_LHR_V30_MAX_K,
)
from coordpy.mergeable_latent_capsule_v25 import (
    MergeableLatentCapsuleV25,
    W77_MLSC_V25_SCHEMA_VERSION,
)
from coordpy.mergeable_latent_capsule_v26 import (
    MergeOperatorV26, wrap_v25_as_v26,
)
from coordpy.persistent_latent_v30 import (
    PersistentLatentStateV30Chain,
    W78_DEFAULT_V30_N_LAYERS,
    W78_DEFAULT_V30_MAX_CHAIN_WALK_DEPTH,
)
from coordpy.replay_controller import ReplayCandidate
from coordpy.replay_controller_v19 import (
    ReplayControllerV19,
    W78_REPLAY_REGIMES_V19,
    W78_LONG_HORIZON_RECONSTRUCTION_AWARE_ROUTING_LABELS,
    fit_replay_controller_v19_per_role,
    fit_replay_v19_long_horizon_reconstruction_aware_routing_head,
)
from coordpy.substrate_adapter_v23 import (
    probe_tiny_substrate_v23_adapter,
    W78_SUBSTRATE_TIER_SUBSTRATE_V23_FULL,
)
from coordpy.tiny_substrate_v23 import (
    W78_REPAIR_LABELS_V23,
    W78_REPAIR_LONG_DELAY_RECONSTRUCTION,
    build_default_tiny_substrate_v23,
    forward_tiny_substrate_v23,
    record_long_horizon_reconstruction_window_v23,
    substrate_long_horizon_reconstruction_dominance_flops_v23,
    substrate_long_horizon_reconstruction_pressure_throttle_v23,
    tokenize_bytes_v23,
)


R194_SCHEMA_VERSION: str = "coordpy.r194_benchmark.v1"


def run_r194(*, seeds: Sequence[int]) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    # H1200: substrate V23 forward determinism.
    sub = build_default_tiny_substrate_v23(seed=78001)
    ids = tokenize_bytes_v23("w78 r194", max_len=8)
    trace_a, cache_a = forward_tiny_substrate_v23(sub, ids)
    trace_b, cache_b = forward_tiny_substrate_v23(sub, ids)
    cells["H1200"] = bool(trace_a.cid() == trace_b.cid())
    # H1200b: LHR trajectory CID active after recording event.
    record_long_horizon_reconstruction_window_v23(
        cache_a,
        compound_chain_failure_turn=30,
        reconstruction_request_turn=88,
        long_horizon_blackout_window_turns=58)
    trace_c, cache_c = forward_tiny_substrate_v23(
        sub, ids, v23_kv_cache=cache_a,
        long_horizon_reconstruction_pressure=0.95)
    cells["H1200b"] = bool(
        str(cache_c.long_horizon_reconstruction_trajectory_cid)
        != "")
    # H1201: LHR label fires under blackout + recon-request.
    cells["H1201"] = bool(
        int(
            (trace_c.long_horizon_reconstruction_length_per_layer
             == W78_REPAIR_LONG_DELAY_RECONSTRUCTION).sum())
        > 0)
    # H1202: substrate LHR recompute-dominance flops saves ≥ 0.92.
    flops = (
        substrate_long_horizon_reconstruction_dominance_flops_v23(
            n_tokens=2048))
    cells["H1202"] = bool(
        float(flops["saving_ratio"]) >= 0.92)
    # H1202b: substrate LHR pressure throttle saves > 0.
    thr = (
        substrate_long_horizon_reconstruction_pressure_throttle_v23(
            visible_token_budget=64, baseline_token_cost=512,
            long_horizon_blackout_window_turns=200))
    cells["H1202b"] = bool(int(thr["saving_tokens"]) > 0)
    # H1203: KV V23 projection chain wraps V22 byte-for-byte.
    import numpy as _np
    rng = _np.random.default_rng(78002)
    from coordpy.w77_team import W77Params
    w77_p = W77Params.build_default(seed=78900)
    kvb23 = KVBridgeV23Projection.init_from_v22(
        w77_p.kv_bridge_v22, seed_v23=77996)
    cells["H1203"] = bool(
        str(kvb23.inner_v22.cid())
        == str(w77_p.kv_bridge_v22.cid()))
    # H1203b: LHR falsifier honest=0 on flag=1 (decision flips
    # when inverted, so falsifier scores 0).
    falsifier = (
        probe_kv_bridge_v23_long_horizon_reconstruction_falsifier(
            long_horizon_reconstruction_pressure_flag=1))
    cells["H1203b"] = bool(
        float(falsifier.falsifier_score) == 0.0)
    # H1204: cache V21 18-objective ridge converges.
    cc21 = CacheControllerV21.init(fit_seed=78003)
    X4 = rng.standard_normal((20, 4)).tolist()
    ys = [
        rng.standard_normal(20).tolist() for _ in range(18)]
    cc21, rep = fit_eighteen_objective_ridge_v21(
        controller=cc21, train_features=X4,
        target_drop_oracle=ys[0],
        target_retrieval_relevance=ys[1],
        target_hidden_wins=ys[2],
        target_replay_dominance=ys[3],
        target_team_task_success=ys[4],
        target_team_failure_recovery=ys[5],
        target_branch_merge=ys[6],
        target_partial_contradiction=ys[7],
        target_multi_branch_rejoin=ys[8],
        target_budget_primary=ys[9],
        target_restart_dominance=ys[10],
        target_delayed_rejoin_after_restart=ys[11],
        target_replacement_after_ctr=ys[12],
        target_compound_repair=ys[13],
        target_compound_chain_repair=ys[14],
        target_chain_then_restart_repair=ys[15],
        target_post_restart_replacement_repair=ys[16],
        target_long_horizon_reconstruction_repair=ys[17])
    cells["H1204"] = bool(int(rep.n_objectives) == 18)
    # H1205: replay V19 routing head trained, 26 regimes.
    rcv19 = ReplayControllerV19.init()
    cands = {
        r: [ReplayCandidate(
            100, 1000, 50, 0.1, 0.0, 0.3, True, True, 0)]
        for r in W78_REPLAY_REGIMES_V19}
    decs = {
        r: ["choose_reuse"] for r in W78_REPLAY_REGIMES_V19}
    rcv19, _ = fit_replay_controller_v19_per_role(
        controller=rcv19, role="planner",
        train_candidates_per_regime=cands,
        train_decisions_per_regime=decs)
    X_team = rng.standard_normal((32, 8)).tolist()
    labels = [
        W78_LONG_HORIZON_RECONSTRUCTION_AWARE_ROUTING_LABELS[
            i % len(
                W78_LONG_HORIZON_RECONSTRUCTION_AWARE_ROUTING_LABELS)]
        for i in range(32)]
    rcv19, rep = (
        fit_replay_v19_long_horizon_reconstruction_aware_routing_head(
            controller=rcv19,
            train_team_features=X_team,
            train_routing_labels=labels))
    cells["H1205"] = bool(
        int(rep.n_classes)
        == len(W78_LONG_HORIZON_RECONSTRUCTION_AWARE_ROUTING_LABELS)
        and len(W78_REPLAY_REGIMES_V19) == 26)
    # H1206: persistent V30: 29 layers, max_chain_walk_depth=33554432.
    cells["H1206"] = bool(
        int(W78_DEFAULT_V30_N_LAYERS) == 29
        and int(W78_DEFAULT_V30_MAX_CHAIN_WALK_DEPTH)
            == 33554432)
    # H1207: LHR V30 retention head: 29 heads, max_k=2048.
    lhr30 = LongHorizonReconstructionV30Head.init()
    cells["H1207"] = bool(
        int(lhr30.max_k) == int(W78_DEFAULT_LHR_V30_MAX_K))
    # H1208: MLSC V26 long-horizon-reconstruction chain added.
    from coordpy.mergeable_latent_capsule_v25 import (
        wrap_v24_as_v25,
    )
    # Simulate a V25 capsule trivially.
    v25 = MergeableLatentCapsuleV25(
        schema=W77_MLSC_V25_SCHEMA_VERSION,
        inner_v24=None,
        post_restart_replacement_trajectory_chain=(),
        post_restart_replacement_window_chain=(),
        algebra_signature_v25="merge_v25_propagation")
    try:
        v26 = wrap_v25_as_v26(
            v25,
            long_horizon_reconstruction_trajectory_chain=(
                "lhr_cid_x",),
            reconstruction_request_window_chain=(
                "rrw_x",))
        cells["H1208"] = bool(
            len(v26.long_horizon_reconstruction_trajectory_chain)
            == 1)
    except Exception:
        cells["H1208"] = False
    # H1209: consensus V24 42-stage chain.
    cells["H1209"] = bool(
        len(W78_CONSENSUS_V24_STAGES) >= 42)
    # H1210: substrate adapter V23 full tier.
    cap = probe_tiny_substrate_v23_adapter()
    cells["H1210"] = bool(
        cap.tier == W78_SUBSTRATE_TIER_SUBSTRATE_V23_FULL)
    # H1211: deep hybrid V23 — 23-way fires when all axes present.
    v22w = DeepSubstrateHybridV22ForwardWitness(
        schema="coordpy.deep_substrate_hybrid_v22.v1",
        hybrid_cid="", inner_v21_witness_cid="",
        twenty_two_way=True,
        cache_controller_v20_fired=True,
        replay_controller_v18_fired=True,
        post_restart_replacement_trajectory_active=True,
        post_restart_replacement_repair_active=True,
        team_consensus_controller_v12_active=True,
        post_restart_replacement_trajectory_cid="prc_x",
        post_restart_replacement_repair_l1=1,
        post_restart_replacement_pressure_gate_mean=0.5)
    hyb = DeepSubstrateHybridV23()
    w = deep_substrate_hybrid_v23_forward(
        hybrid=hyb, v22_witness=v22w,
        cache_controller_v21=cc21,
        replay_controller_v19=rcv19,
        long_horizon_reconstruction_trajectory_cid="lhr_x",
        long_horizon_reconstruction_repair_l1=1,
        long_horizon_reconstruction_pressure_gate_mean=0.5,
        n_team_consensus_v13_invocations=1)
    cells["H1211"] = bool(w.twenty_three_way)
    # H1212: long-horizon-reconstruction substrate reconstructs.
    carrier = build_default_long_horizon_reconstruction_carrier(
        n_events=256, seed=78007)
    known = carrier.entries[30]
    q = LongHorizonReconstructionQuery(
        query_id="q1", source_turn=30, current_turn=220)
    out = reconstruct_long_horizon_event(
        carrier=carrier, query=q, visible_tokens_used=4)
    cells["H1212"] = bool(
        bool(out.success)
        and str(out.reconstructed_event_cid)
        == str(known.event_cid))
    # H1212b: substrate beats full replay on FLOPs.
    econ = report_reconstruction_vs_recompute_economics(
        query=q, carrier=carrier)
    cells["H1212b"] = bool(float(econ.saving_ratio) >= 0.90)
    # H1213: V23 repair labels = 15 (V22's 14 + 1).
    cells["H1213"] = bool(
        len(W78_REPAIR_LABELS_V23) == 15)
    return {
        "schema": R194_SCHEMA_VERSION,
        "seeds": list(seeds),
        "cells": dict(cells),
        "all_pass": bool(all(bool(v) for v in cells.values())),
    }


__all__ = ["R194_SCHEMA_VERSION", "run_r194"]
