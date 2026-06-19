"""W79 R-198 benchmark family — Real-substrate plane V24 + learned
consolidation + long-horizon reconstruction V2.

H1300..H1319 cell families (20 H-bars):

* H1300  V24 substrate exposes new W79 axis
* H1301  V24 substrate forward deterministic on identical inputs
* H1302  KV V24 20-target ridge
* H1303  KV V24 168-dim fingerprint deterministic
* H1304  KV V24 RTRLD falsifier honest=0
* H1305  Cache V22 19-objective ridge fits
* H1306  Replay V20 27 regimes
* H1307  Replay V20 17-label routing head
* H1308  Persistent V31 30 layers
* H1309  LHR V31 30 heads, max_k=4096
* H1310  MLSC V27 RTRLD trajectory chain populated
* H1311  Consensus V25 44 stages
* H1312  Substrate adapter V24 has v24_full tier
* H1313  Substrate adapter V24 has controlled_runtime tier
* H1314  Deep substrate hybrid V24 twenty-four-way active
* H1315  LHR substrate V2 carrier has learned slots
* H1316  LHR substrate V2 reconstruction succeeds on > 32-turn
         queries
* H1317  Replay-vs-recompute V2 arbiter picks substrate_replay
         when carrier is cheap
* H1318  Bounded-window baseline V2 k64 fails on > 64-turn query
* H1319  Bounded-window baseline V2 cross-prompt fails on > 64
"""

from __future__ import annotations

from typing import Any, Sequence

from coordpy.bounded_window_baseline_v1 import BoundedWindowQuery
from coordpy.bounded_window_baseline_v2 import (
    build_default_bounded_window_baselines_v2,
    run_bounded_window_falsifier_v2,
)
from coordpy.cache_controller_v22 import (
    CacheControllerV22, W79_CACHE_CONTROLLER_V22_N_OBJECTIVES,
    fit_nineteen_objective_ridge_v22,
)
from coordpy.consensus_fallback_controller_v25 import (
    W79_CONSENSUS_V25_N_STAGES,
)
from coordpy.deep_substrate_hybrid_v23 import (
    DeepSubstrateHybridV23ForwardWitness,
)
from coordpy.deep_substrate_hybrid_v24 import (
    DeepSubstrateHybridV24, deep_substrate_hybrid_v24_forward,
)
from coordpy.kv_bridge_v24 import (
    W79_KV_BRIDGE_V24_FINGERPRINT_DIM,
    W79_KV_BRIDGE_V24_N_TARGETS,
    compute_replacement_then_restart_after_long_delay_fingerprint_v24,
    probe_kv_bridge_v24_replacement_then_restart_after_long_delay_falsifier,
)
from coordpy.learned_consolidation_v1 import (
    build_learned_consolidation_head_v1,
    build_nonlinear_consolidation_dataset,
    train_learned_consolidation_head,
)
from coordpy.long_horizon_reconstruction_substrate_v1 import (
    LongHorizonReconstructionQuery,
)
from coordpy.long_horizon_reconstruction_substrate_v2 import (
    W79_REPLAY_VS_RECOMPUTE_DECISION_SUBSTRATE_REPLAY,
    arbitrate_replay_vs_recompute_v2,
    build_default_long_horizon_reconstruction_carrier_v2,
    reconstruct_long_horizon_event_v2,
)
from coordpy.long_horizon_retention_v31 import (
    LongHorizonReconstructionV31Head,
    W79_DEFAULT_LHR_V31_MAX_K,
)
from coordpy.persistent_latent_v31 import (
    W79_DEFAULT_V31_N_LAYERS,
)
from coordpy.replay_controller_v20 import (
    W79_REPLACEMENT_THEN_RESTART_AFTER_LONG_DELAY_AWARE_ROUTING_LABELS,
    W79_REPLAY_REGIMES_V20,
)
from coordpy.substrate_adapter_v24 import (
    W79_SUBSTRATE_TIER_CONTROLLED_RUNTIME_V1,
    W79_SUBSTRATE_TIER_SUBSTRATE_V24_FULL,
    probe_all_v24_adapters,
)
from coordpy.tiny_substrate_v24 import (
    W79_REPAIR_LABELS_V24,
    build_default_tiny_substrate_v24,
    emit_tiny_substrate_v24_forward_witness,
    forward_tiny_substrate_v24,
    record_replacement_then_restart_after_long_delay_window_v24,
    tokenize_bytes_v24,
)
from coordpy.mergeable_latent_capsule_v3 import (
    make_root_capsule_v3,
)
from coordpy.mergeable_latent_capsule_v4 import wrap_v3_as_v4
from coordpy.mergeable_latent_capsule_v5 import wrap_v4_as_v5
from coordpy.mergeable_latent_capsule_v6 import wrap_v5_as_v6
from coordpy.mergeable_latent_capsule_v7 import wrap_v6_as_v7
from coordpy.mergeable_latent_capsule_v8 import wrap_v7_as_v8
from coordpy.mergeable_latent_capsule_v9 import wrap_v8_as_v9
from coordpy.mergeable_latent_capsule_v10 import wrap_v9_as_v10
from coordpy.mergeable_latent_capsule_v11 import wrap_v10_as_v11
from coordpy.mergeable_latent_capsule_v12 import wrap_v11_as_v12
from coordpy.mergeable_latent_capsule_v13 import wrap_v12_as_v13
from coordpy.mergeable_latent_capsule_v14 import wrap_v13_as_v14
from coordpy.mergeable_latent_capsule_v15 import wrap_v14_as_v15
from coordpy.mergeable_latent_capsule_v16 import wrap_v15_as_v16
from coordpy.mergeable_latent_capsule_v17 import wrap_v16_as_v17
from coordpy.mergeable_latent_capsule_v18 import wrap_v17_as_v18
from coordpy.mergeable_latent_capsule_v19 import wrap_v18_as_v19
from coordpy.mergeable_latent_capsule_v20 import wrap_v19_as_v20
from coordpy.mergeable_latent_capsule_v21 import wrap_v20_as_v21
from coordpy.mergeable_latent_capsule_v22 import wrap_v21_as_v22
from coordpy.mergeable_latent_capsule_v23 import wrap_v22_as_v23
from coordpy.mergeable_latent_capsule_v24 import wrap_v23_as_v24
from coordpy.mergeable_latent_capsule_v25 import wrap_v24_as_v25
from coordpy.mergeable_latent_capsule_v26 import wrap_v25_as_v26
from coordpy.mergeable_latent_capsule_v27 import wrap_v26_as_v27


R198_SCHEMA_VERSION: str = "coordpy.r198_benchmark.v1"


def _build_v27_capsule(rtrld_traj: str) -> Any:
    v3 = make_root_capsule_v3(
        branch_id="r198", payload=(0.1,) * 6,
        fact_tags=("r198",), confidence=0.9, trust=0.9,
        turn_index=0)
    v = v3
    for wrap in (
            wrap_v3_as_v4, wrap_v4_as_v5, wrap_v5_as_v6,
            wrap_v6_as_v7, wrap_v7_as_v8, wrap_v8_as_v9,
            wrap_v9_as_v10, wrap_v10_as_v11,
            wrap_v11_as_v12, wrap_v12_as_v13,
            wrap_v13_as_v14, wrap_v14_as_v15,
            wrap_v15_as_v16, wrap_v16_as_v17,
            wrap_v17_as_v18, wrap_v18_as_v19):
        v = wrap(v)
    v = wrap_v19_as_v20(
        v, restart_repair_trajectory_chain=(),
        rejoin_pressure_chain=())
    v = wrap_v20_as_v21(
        v, replacement_repair_trajectory_chain=(),
        contradiction_chain=())
    v = wrap_v21_as_v22(
        v, compound_repair_trajectory_chain=(),
        delayed_repair_chain=())
    v = wrap_v22_as_v23(
        v, compound_chain_repair_trajectory_chain=(),
        replacement_then_rejoin_chain=())
    v = wrap_v23_as_v24(
        v, chain_then_restart_trajectory_chain=(),
        post_compound_chain_restart_chain=())
    v = wrap_v24_as_v25(
        v, post_restart_replacement_trajectory_chain=(),
        post_restart_replacement_window_chain=())
    v = wrap_v25_as_v26(
        v, long_horizon_reconstruction_trajectory_chain=(),
        reconstruction_request_window_chain=())
    return wrap_v26_as_v27(
        v,
        replacement_then_restart_after_long_delay_trajectory_chain=(
            rtrld_traj,),
        replacement_then_restart_after_long_delay_window_chain=(
            "rtw_1",))


def run_r198(*, seeds: Sequence[int]) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    p = build_default_tiny_substrate_v24()
    ids = tokenize_bytes_v24("r198 substrate", max_len=16)
    t1, c1 = forward_tiny_substrate_v24(
        p, ids,
        replacement_then_restart_after_long_delay_pressure=0.95)
    record_replacement_then_restart_after_long_delay_window_v24(
        c1, long_delay_blackout_start_turn=30,
        replacement_turn=45, restart_turn=70,
        reconstruction_request_turn=95)
    t2, c2 = forward_tiny_substrate_v24(
        p, ids, v24_kv_cache=c1,
        replacement_then_restart_after_long_delay_pressure=0.95)
    w = emit_tiny_substrate_v24_forward_witness(p, t2, c2)
    cells["H1300"] = bool(
        len(W79_REPAIR_LABELS_V24) == 16
        and "replacement_then_restart_after_long_delay_reconstruction"
        in W79_REPAIR_LABELS_V24)
    # H1301 determinism: rebuild and re-run, compare witness.
    p2 = build_default_tiny_substrate_v24()
    t3, c3 = forward_tiny_substrate_v24(
        p2, ids,
        replacement_then_restart_after_long_delay_pressure=0.95)
    record_replacement_then_restart_after_long_delay_window_v24(
        c3, long_delay_blackout_start_turn=30,
        replacement_turn=45, restart_turn=70,
        reconstruction_request_turn=95)
    t4, c4 = forward_tiny_substrate_v24(
        p2, ids, v24_kv_cache=c3,
        replacement_then_restart_after_long_delay_pressure=0.95)
    w_b = emit_tiny_substrate_v24_forward_witness(p2, t4, c4)
    cells["H1301"] = bool(w.cid() == w_b.cid())
    cells["H1302"] = bool(
        int(W79_KV_BRIDGE_V24_N_TARGETS) == 20)
    fp_a = (
        compute_replacement_then_restart_after_long_delay_fingerprint_v24(
            role="r",
            long_horizon_reconstruction_trajectory_cid="lhr",
            replacement_then_restart_after_long_delay_trajectory_cid="rtrld",
            replacement_then_restart_after_long_delay_count=2,
            long_delay_blackout_window_turns=60))
    fp_b = (
        compute_replacement_then_restart_after_long_delay_fingerprint_v24(
            role="r",
            long_horizon_reconstruction_trajectory_cid="lhr",
            replacement_then_restart_after_long_delay_trajectory_cid="rtrld",
            replacement_then_restart_after_long_delay_count=2,
            long_delay_blackout_window_turns=60))
    cells["H1303"] = bool(
        len(fp_a) == int(W79_KV_BRIDGE_V24_FINGERPRINT_DIM)
        and tuple(fp_a) == tuple(fp_b))
    fals = (
        probe_kv_bridge_v24_replacement_then_restart_after_long_delay_falsifier(
            replacement_then_restart_after_long_delay_pressure_flag=1))
    cells["H1304"] = bool(float(fals.falsifier_score) == 0.0)
    import numpy as np
    cc22 = CacheControllerV22.init(fit_seed=42)
    rng = np.random.default_rng(42)
    X = rng.standard_normal((20, 4)).tolist()
    targets = [
        rng.standard_normal(20).tolist() for _ in range(
            int(W79_CACHE_CONTROLLER_V22_N_OBJECTIVES))]
    cc22, rep = fit_nineteen_objective_ridge_v22(
        controller=cc22, train_features=X,
        targets_per_objective=targets)
    cells["H1305"] = bool(
        cc22.nineteen_objective_head is not None
        and int(rep.n_objectives) == 19)
    cells["H1306"] = bool(len(W79_REPLAY_REGIMES_V20) == 27)
    cells["H1307"] = bool(
        len(
            W79_REPLACEMENT_THEN_RESTART_AFTER_LONG_DELAY_AWARE_ROUTING_LABELS)
        == 17)
    cells["H1308"] = bool(int(W79_DEFAULT_V31_N_LAYERS) == 30)
    head31 = LongHorizonReconstructionV31Head.init(seed=42)
    cells["H1309"] = bool(
        int(W79_DEFAULT_LHR_V31_MAX_K) == 4096
        and int(head31.replacement_then_restart_after_long_delay_dim)
        == 8)
    v27_c = _build_v27_capsule(rtrld_traj="rtrld_x")
    cells["H1310"] = bool(
        len(
            v27_c
            .replacement_then_restart_after_long_delay_trajectory_chain)
        >= 1)
    cells["H1311"] = bool(int(W79_CONSENSUS_V25_N_STAGES) == 44)
    adapters = probe_all_v24_adapters()
    cells["H1312"] = bool(adapters.has_v24_full())
    cells["H1313"] = bool(adapters.has_controlled_runtime())
    v23_w = DeepSubstrateHybridV23ForwardWitness(
        schema="coordpy.deep_substrate_hybrid_v23.v1",
        hybrid_cid="", inner_v22_witness_cid="",
        twenty_three_way=True,
        cache_controller_v21_fired=True,
        replay_controller_v19_fired=True,
        long_horizon_reconstruction_trajectory_active=True,
        long_horizon_reconstruction_repair_active=True,
        team_consensus_controller_v13_active=True,
        long_horizon_reconstruction_trajectory_cid="lhr",
        long_horizon_reconstruction_repair_l1=1,
        long_horizon_reconstruction_pressure_gate_mean=0.5)
    dhv24 = DeepSubstrateHybridV24()
    dh_w = deep_substrate_hybrid_v24_forward(
        hybrid=dhv24, v23_witness=v23_w,
        replacement_then_restart_after_long_delay_trajectory_cid="rtrld",
        replacement_then_restart_after_long_delay_repair_l1=1,
        replacement_then_restart_after_long_delay_pressure_gate_mean=0.5,
        n_team_consensus_v14_invocations=1)
    cells["H1314"] = bool(dh_w.twenty_four_way)
    head = build_learned_consolidation_head_v1(seed=99)
    X_, Y_ = build_nonlinear_consolidation_dataset(
        n_samples=80, seed=99)
    head, _ = train_learned_consolidation_head(
        head=head, train_features=X_.tolist(),
        train_targets=Y_.tolist(),
        n_iters=30)
    carrier_v2 = (
        build_default_long_horizon_reconstruction_carrier_v2(
            n_events=128, seed=42, head=head))
    cells["H1315"] = bool(len(carrier_v2.learned_slots) >= 64)
    q = LongHorizonReconstructionQuery(
        query_id="q", source_turn=10, current_turn=200)
    out = reconstruct_long_horizon_event_v2(
        carrier_v2=carrier_v2, query=q,
        visible_tokens_used=4)
    cells["H1316"] = bool(out.success)
    arb = arbitrate_replay_vs_recompute_v2(
        carrier_v2=carrier_v2, query=q,
        controlled_runtime_params_cid="fake")
    cells["H1317"] = bool(
        arb.chosen
        == W79_REPLAY_VS_RECOMPUTE_DECISION_SUBSTRATE_REPLAY)
    baselines = build_default_bounded_window_baselines_v2()
    bw_q = BoundedWindowQuery(
        query_id="bq", current_turn=200, source_turn=10,
        expected_event_cid="ev")
    _, fals_v2 = run_bounded_window_falsifier_v2(
        baselines_v2=baselines, query=bw_q)
    cells["H1318"] = bool(fals_v2.k64_failed)
    cells["H1319"] = bool(fals_v2.cross_prompt_summary_failed)
    return {
        "schema": R198_SCHEMA_VERSION,
        "seeds": list(seeds),
        "cells": dict(cells),
        "all_pass": bool(all(bool(v) for v in cells.values())),
    }


__all__ = ["R198_SCHEMA_VERSION", "run_r198"]
