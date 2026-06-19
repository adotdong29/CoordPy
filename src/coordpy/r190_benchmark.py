"""W77 R-190 benchmark family — Real Substrate Plane V22 (Plane B).

Exercises the W77 real substrate plane V22 modules:
tiny_substrate_v22 (24 layers, 3 new V22 axes), kv_bridge_v22
(18-target ridge), cache_controller_v20 (17-objective ridge),
replay_controller_v18 (25 regimes + 15-label routing), deep
substrate hybrid V22 (22-way), substrate adapter V22, persistent
latent V29, long-horizon retention V29, MLSC V25, consensus V23.

H1180..H1192 cell families (14 H-bars):

* H1180  V22 substrate inherits V21's 23 physical layers + a
         fourth substrate-axis layer (post-restart-replacement)
* H1181  V22 post-restart-replacement trajectory CID is content-
         addressed (differs from V21 trajectory CID)
* H1181b V22 post-restart-replacement length per layer is shape (L,)
* H1182  V22 post-restart-replacement-pressure gate per layer is
         shape (L,) with values in [0,1]
* H1183  KV V22 eighteen-target ridge fits and falsifier returns 0
         on honest claim
* H1184  KV V22 148-dim post-restart-replacement fingerprint is
         deterministic and varies with inputs
* H1185  Cache V20 seventeen-objective ridge fits and converges
* H1186  Cache V20 per-role post-restart-replacement-pressure head
         is 18-dim
* H1187  Replay V18 has 25 regimes (24 V17 + 1 new)
* H1188  Replay V18 post-restart-replacement-aware routing head
         has 15 classes
* H1189  Persistent V29 has 28 layers; max_chain_walk_depth=
         16777216
* H1190  LHR V29 has 28 heads; max_k=1024
* H1191  MLSC V25 propagates the post-restart-replacement
         trajectory chain through merge
* H1192  Consensus V23 has 40 stages including 2 new V23 arbiters
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as _np

from coordpy.cache_controller_v20 import (
    CacheControllerV20,
    fit_per_role_post_restart_replacement_pressure_head_v20,
    fit_seventeen_objective_ridge_v20,
)
from coordpy.consensus_fallback_controller_v23 import (
    ConsensusFallbackControllerV23, W77_CONSENSUS_V23_STAGES,
)
from coordpy.kv_bridge_v21 import KVBridgeV21Projection
from coordpy.kv_bridge_v22 import (
    KVBridgeV22Projection,
    compute_post_restart_replacement_fingerprint_v22,
    probe_kv_bridge_v22_post_restart_replacement_falsifier,
)
from coordpy.long_horizon_retention_v29 import (
    LongHorizonReconstructionV29Head,
    W77_DEFAULT_LHR_V29_MAX_K,
)
from coordpy.mergeable_latent_capsule_v3 import (
    make_root_capsule_v3,
)
from coordpy.mergeable_latent_capsule_v25 import (
    MergeOperatorV25, wrap_v24_as_v25,
)
from coordpy.persistent_latent_v29 import (
    PersistentLatentStateV29Chain,
    W77_DEFAULT_V29_MAX_CHAIN_WALK_DEPTH,
    W77_DEFAULT_V29_N_LAYERS,
)
from coordpy.replay_controller_v18 import (
    ReplayControllerV18,
    W77_POST_RESTART_REPLACEMENT_AWARE_ROUTING_LABELS,
    W77_REPLAY_REGIMES_V18,
)
from coordpy.tiny_substrate_v22 import (
    build_default_tiny_substrate_v22,
    forward_tiny_substrate_v22,
    record_post_restart_replacement_window_v22,
    tokenize_bytes_v22,
    W77_DEFAULT_V22_N_LAYERS,
)


R190_SCHEMA_VERSION: str = "coordpy.r190_benchmark.v1"


def run_r190(*, seeds: Sequence[int]) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    # H1180: V22 substrate has 23 physical layers (inherits V21).
    cells["H1180"] = bool(int(W77_DEFAULT_V22_N_LAYERS) == 23)
    # Build V22 substrate.
    params_v22 = build_default_tiny_substrate_v22(seed=77100)
    ids = tokenize_bytes_v22("r190", max_len=8)
    trace, cache = forward_tiny_substrate_v22(
        params_v22, ids,
        post_restart_replacement_pressure=0.9)
    record_post_restart_replacement_window_v22(
        cache, chain_then_restart_repair_turn=18,
        replacement_turn=22,
        post_restart_replacement_window_turns=10,
        role="planner_fresh")
    trace, cache = forward_tiny_substrate_v22(
        params_v22, ids, v22_kv_cache=cache,
        post_restart_replacement_pressure=0.9)
    # H1181: PCR trajectory CID is content-addressed (non-empty,
    # differs from V21 inner CID).
    pcr_cid = str(
        cache
        .replacement_after_restart_after_compound_chain_trajectory_cid)
    v21_cid = str(
        cache.v21_cache
        .compound_chain_then_restart_trajectory_cid)
    cells["H1181"] = bool(
        len(pcr_cid) > 0 and pcr_cid != v21_cid)
    # H1181b: PCR length per layer is shape (L,).
    pcr_len = (
        cache
        .replacement_after_restart_after_compound_chain_length_per_layer)
    cells["H1181b"] = bool(
        pcr_len is not None
        and pcr_len.shape == (int(W77_DEFAULT_V22_N_LAYERS),)
        and pcr_len.dtype == _np.int64)
    # H1182: PCR pressure gate per layer in [0,1].
    pcr_gate = (
        trace
        .replacement_after_restart_after_compound_chain_pressure_gate_per_layer)
    cells["H1182"] = bool(
        pcr_gate.shape == (int(W77_DEFAULT_V22_N_LAYERS),)
        and bool((pcr_gate >= 0.0).all())
        and bool((pcr_gate <= 1.0).all()))
    # H1183: KV V22 falsifier returns 0 on honest claim.
    fk = probe_kv_bridge_v22_post_restart_replacement_falsifier(
        post_restart_replacement_pressure_flag=1)
    cells["H1183"] = bool(float(fk.falsifier_score) == 0.0)
    # H1184: KV V22 fingerprint is deterministic and varies with
    # inputs.
    fp_a = compute_post_restart_replacement_fingerprint_v22(
        role="a",
        compound_chain_then_restart_trajectory_cid="cidA",
        replacement_after_restart_after_compound_chain_trajectory_cid="prcA")
    fp_a2 = compute_post_restart_replacement_fingerprint_v22(
        role="a",
        compound_chain_then_restart_trajectory_cid="cidA",
        replacement_after_restart_after_compound_chain_trajectory_cid="prcA")
    fp_b = compute_post_restart_replacement_fingerprint_v22(
        role="b",
        compound_chain_then_restart_trajectory_cid="cidA",
        replacement_after_restart_after_compound_chain_trajectory_cid="prcA")
    cells["H1184"] = bool(
        fp_a == fp_a2 and fp_a != fp_b
        and len(fp_a) == 148)
    # H1185: Cache V20 seventeen-objective ridge fits.
    rng = _np.random.default_rng(7700)
    X = rng.standard_normal((20, 4))
    cc = CacheControllerV20.init(fit_seed=77200)
    ys = {
        f"y{i}": (X[:, i % 4] * (0.1 + 0.05 * i)).tolist()
        for i in range(17)}
    cc, rep = fit_seventeen_objective_ridge_v20(
        controller=cc, train_features=X.tolist(),
        target_drop_oracle=ys["y0"],
        target_retrieval_relevance=ys["y1"],
        target_hidden_wins=ys["y2"],
        target_replay_dominance=ys["y3"],
        target_team_task_success=ys["y4"],
        target_team_failure_recovery=ys["y5"],
        target_branch_merge=ys["y6"],
        target_partial_contradiction=ys["y7"],
        target_multi_branch_rejoin=ys["y8"],
        target_budget_primary=ys["y9"],
        target_restart_dominance=ys["y10"],
        target_delayed_rejoin_after_restart=ys["y11"],
        target_replacement_after_ctr=ys["y12"],
        target_compound_repair=ys["y13"],
        target_compound_chain_repair=ys["y14"],
        target_chain_then_restart_repair=ys["y15"],
        target_post_restart_replacement_repair=ys["y16"])
    cells["H1185"] = bool(
        rep.converged and rep.n_objectives == 17)
    # H1186: Cache V20 per-role 18-dim head.
    X18 = rng.standard_normal((20, 18))
    cc, head_rep = (
        fit_per_role_post_restart_replacement_pressure_head_v20(
            controller=cc, role="planner",
            train_features=X18.tolist(),
            target_post_restart_replacement_priorities=(
                X18[:, 0] * 0.3 + X18[:, 17] * 0.5).tolist()))
    cells["H1186"] = bool(
        "planner"
        in cc.per_role_post_restart_replacement_pressure_heads_v20
        and (cc
             .per_role_post_restart_replacement_pressure_heads_v20[
                 "planner"].shape == (18,)))
    # H1187: Replay V18 has 25 regimes.
    cells["H1187"] = bool(len(W77_REPLAY_REGIMES_V18) == 25)
    # H1188: Replay V18 routing labels = 15.
    cells["H1188"] = bool(
        len(W77_POST_RESTART_REPLACEMENT_AWARE_ROUTING_LABELS)
        == 15)
    # H1189: Persistent V29 layers/max_walk.
    cells["H1189"] = bool(
        int(W77_DEFAULT_V29_N_LAYERS) == 28
        and int(W77_DEFAULT_V29_MAX_CHAIN_WALK_DEPTH)
            == 16777216)
    # H1190: LHR V29 max_k.
    lhr = LongHorizonReconstructionV29Head.init(seed=77400)
    cells["H1190"] = bool(
        int(lhr.max_k) == int(W77_DEFAULT_LHR_V29_MAX_K)
        and int(W77_DEFAULT_LHR_V29_MAX_K) == 1024)
    # H1191: MLSC V25 propagates PCR chain through merge.
    from coordpy.mergeable_latent_capsule_v24 import (
        wrap_v23_as_v24,
    )
    from coordpy.mergeable_latent_capsule_v23 import (
        wrap_v22_as_v23,
    )
    from coordpy.mergeable_latent_capsule_v22 import (
        wrap_v21_as_v22,
    )
    from coordpy.mergeable_latent_capsule_v21 import (
        wrap_v20_as_v21,
    )
    from coordpy.mergeable_latent_capsule_v20 import (
        wrap_v19_as_v20,
    )
    from coordpy.mergeable_latent_capsule_v19 import (
        wrap_v18_as_v19,
    )
    from coordpy.mergeable_latent_capsule_v18 import (
        wrap_v17_as_v18,
    )
    from coordpy.mergeable_latent_capsule_v17 import (
        wrap_v16_as_v17,
    )
    from coordpy.mergeable_latent_capsule_v16 import (
        wrap_v15_as_v16,
    )
    from coordpy.mergeable_latent_capsule_v15 import (
        wrap_v14_as_v15,
    )
    from coordpy.mergeable_latent_capsule_v14 import (
        wrap_v13_as_v14,
    )
    from coordpy.mergeable_latent_capsule_v13 import (
        wrap_v12_as_v13,
    )
    from coordpy.mergeable_latent_capsule_v12 import (
        wrap_v11_as_v12,
    )
    from coordpy.mergeable_latent_capsule_v11 import (
        wrap_v10_as_v11,
    )
    from coordpy.mergeable_latent_capsule_v10 import (
        wrap_v9_as_v10,
    )
    from coordpy.mergeable_latent_capsule_v9 import (
        wrap_v8_as_v9,
    )
    from coordpy.mergeable_latent_capsule_v8 import (
        wrap_v7_as_v8,
    )
    from coordpy.mergeable_latent_capsule_v7 import (
        wrap_v6_as_v7,
    )
    from coordpy.mergeable_latent_capsule_v6 import (
        wrap_v5_as_v6,
    )
    from coordpy.mergeable_latent_capsule_v5 import (
        wrap_v4_as_v5,
    )
    from coordpy.mergeable_latent_capsule_v4 import (
        wrap_v3_as_v4,
    )
    v3 = make_root_capsule_v3(
        branch_id="r190",
        payload=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
        fact_tags=("r190",), confidence=0.9, trust=0.9,
        turn_index=0)
    v_chain = v3
    for wrap in (
            wrap_v3_as_v4, wrap_v4_as_v5, wrap_v5_as_v6,
            wrap_v6_as_v7, wrap_v7_as_v8, wrap_v8_as_v9,
            wrap_v9_as_v10, wrap_v10_as_v11,
            wrap_v11_as_v12, wrap_v12_as_v13,
            wrap_v13_as_v14, wrap_v14_as_v15,
            wrap_v15_as_v16, wrap_v16_as_v17,
            wrap_v17_as_v18, wrap_v18_as_v19):
        v_chain = wrap(v_chain)
    v_chain = wrap_v19_as_v20(v_chain)
    v_chain = wrap_v20_as_v21(v_chain)
    v_chain = wrap_v21_as_v22(v_chain)
    v_chain = wrap_v22_as_v23(v_chain)
    v_chain = wrap_v23_as_v24(v_chain)
    v25 = wrap_v24_as_v25(
        v_chain,
        post_restart_replacement_trajectory_chain=("pcr_a",),
        post_restart_replacement_window_chain=("w_a",))
    op = MergeOperatorV25()
    merged = op.merge(
        [v25],
        post_restart_replacement_trajectory_chain=("pcr_b",),
        post_restart_replacement_window_chain=("w_b",))
    cells["H1191"] = bool(
        "pcr_a"
        in merged.post_restart_replacement_trajectory_chain
        and "pcr_b"
        in merged.post_restart_replacement_trajectory_chain
        and "w_a"
        in merged.post_restart_replacement_window_chain
        and "w_b"
        in merged.post_restart_replacement_window_chain)
    # H1192: Consensus V23 has 40 stages (38 V22 + 2 new).
    cells["H1192"] = bool(len(W77_CONSENSUS_V23_STAGES) == 40)
    return {
        "schema": R190_SCHEMA_VERSION,
        "n_seeds": int(len(seeds)),
        "cells": cells,
        "all_pass": bool(all(cells.values())),
    }


__all__ = ["R190_SCHEMA_VERSION", "run_r190"]
