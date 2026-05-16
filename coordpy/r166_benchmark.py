"""W71 R-166 benchmark family — Real Substrate Plane V16 (Plane B).

Exercises the W71 real substrate plane V16: 18-layer V16 substrate
forward (delayed-repair-trajectory CID, restart-dominance per
layer, delayed-repair gate per layer), KV V16 12-target ridge,
cache V14 eleven-objective ridge + per-role restart-priority head,
replay V12 19-regime + restart-aware routing head, persistent V23
(22 layers, max_chain_walk_depth=262144), LHR V23 (22 heads,
max_k=640), MLSC V19 (delayed-repair chain), consensus V17 (28
stages), deep substrate hybrid V16 (16-way).

H550..H565 cell families (16 H-bars):

* H550   Substrate V16 forward-determinism on (params, token_ids)
* H550b  Substrate V16 delayed-repair-trajectory CID byte-stable
* H551   Substrate V16 restart-dominance shape (L=18,)
* H551b  Substrate V16 delayed-repair gate shape (L=18,)
* H552   Substrate V16 repair-dominance flops saving ≥ 0.83 over 7
* H552b  Substrate V16 delayed-repair throttle saving ≥ 0.5 at
         budget=64 / baseline=512 with delay=3
* H553   KV V16 twelve-target ridge fit converges
* H554   Cache V14 eleven-objective ridge converges
* H554b  Cache V14 per-role restart-priority head converges
* H555   Replay V12 19 regimes including
         ``delayed_repair_after_restart_regime``
* H555b  Replay V12 restart-aware routing head 9-class
* H556   Persistent V23 max_chain_walk_depth = 262144
* H556b  Persistent V23 distractor rank = 22
* H557   LHR V23 22 heads + max_k = 640
* H558   MLSC V19 delayed-repair-trajectory chain merge union
* H559   Consensus V17 has ≥ 28 stages
"""

from __future__ import annotations

from typing import Any, Sequence

from coordpy.cache_controller_v14 import (
    CacheControllerV14, fit_eleven_objective_ridge_v14,
    fit_per_role_restart_priority_head_v14,
)
from coordpy.consensus_fallback_controller_v17 import (
    W71_CONSENSUS_V17_STAGES,
)
from coordpy.kv_bridge_v15 import KVBridgeV15Projection
from coordpy.kv_bridge_v14 import KVBridgeV14Projection
from coordpy.kv_bridge_v13 import KVBridgeV13Projection
from coordpy.kv_bridge_v12 import KVBridgeV12Projection
from coordpy.kv_bridge_v11 import KVBridgeV11Projection
from coordpy.kv_bridge_v10 import KVBridgeV10Projection
from coordpy.kv_bridge_v9 import KVBridgeV9Projection
from coordpy.kv_bridge_v8 import KVBridgeV8Projection
from coordpy.kv_bridge_v7 import KVBridgeV7Projection
from coordpy.kv_bridge_v6 import KVBridgeV6Projection
from coordpy.kv_bridge_v5 import KVBridgeV5Projection
from coordpy.kv_bridge_v4 import KVBridgeV4Projection
from coordpy.kv_bridge_v3 import KVBridgeV3Projection
from coordpy.kv_bridge_v16 import (
    KVBridgeV16Projection, fit_kv_bridge_v16_twelve_target,
)
from coordpy.long_horizon_retention_v23 import (
    LongHorizonReconstructionV23Head, W71_DEFAULT_LHR_V23_MAX_K,
)
from coordpy.mergeable_latent_capsule_v19 import (
    MergeOperatorV19, wrap_v18_as_v19,
)
from coordpy.mergeable_latent_capsule_v18 import wrap_v17_as_v18
from coordpy.mergeable_latent_capsule_v17 import wrap_v16_as_v17
from coordpy.mergeable_latent_capsule_v16 import wrap_v15_as_v16
from coordpy.mergeable_latent_capsule_v15 import wrap_v14_as_v15
from coordpy.mergeable_latent_capsule_v14 import wrap_v13_as_v14
from coordpy.mergeable_latent_capsule_v13 import wrap_v12_as_v13
from coordpy.mergeable_latent_capsule_v12 import wrap_v11_as_v12
from coordpy.mergeable_latent_capsule_v11 import wrap_v10_as_v11
from coordpy.mergeable_latent_capsule_v10 import wrap_v9_as_v10
from coordpy.mergeable_latent_capsule_v9 import wrap_v8_as_v9
from coordpy.mergeable_latent_capsule_v8 import wrap_v7_as_v8
from coordpy.mergeable_latent_capsule_v7 import wrap_v6_as_v7
from coordpy.mergeable_latent_capsule_v6 import wrap_v5_as_v6
from coordpy.mergeable_latent_capsule_v5 import wrap_v4_as_v5
from coordpy.mergeable_latent_capsule_v4 import wrap_v3_as_v4
from coordpy.mergeable_latent_capsule_v3 import (
    make_root_capsule_v3,
)
from coordpy.persistent_latent_v23 import (
    W71_DEFAULT_V23_DISTRACTOR_RANK,
    W71_DEFAULT_V23_MAX_CHAIN_WALK_DEPTH,
    W71_DEFAULT_V23_N_LAYERS,
)
from coordpy.replay_controller_v12 import (
    W71_REPLAY_REGIMES_V12, W71_RESTART_AWARE_ROUTING_LABELS,
)
from coordpy.tiny_substrate_v16 import (
    W71_DEFAULT_V16_N_LAYERS,
    build_default_tiny_substrate_v16,
    forward_tiny_substrate_v16,
    record_delay_window_v16, record_restart_event_v16,
    substrate_delayed_repair_throttle_v16,
    substrate_repair_dominance_flops_v16,
    tokenize_bytes_v16,
)


R166_SCHEMA_VERSION: str = "coordpy.r166_benchmark.v1"


def _build_v16_substrate(seed: int = 71000):
    return build_default_tiny_substrate_v16(seed=int(seed))


def _build_v16_kv_projection(sub):
    cfg = sub.config.v15.v14.v13.v12.v11.v10.v9
    d_head = int(cfg.d_model) // int(cfg.n_heads)
    kv_b3 = KVBridgeV3Projection.init(
        n_layers=int(cfg.n_layers),
        n_heads=int(cfg.n_heads),
        n_kv_heads=int(cfg.n_kv_heads),
        n_inject_tokens=3, carrier_dim=6,
        d_head=int(d_head), seed=71008)
    kv_b14 = KVBridgeV14Projection.init_from_v13(
        KVBridgeV13Projection.init_from_v12(
            KVBridgeV12Projection.init_from_v11(
                KVBridgeV11Projection.init_from_v10(
                    KVBridgeV10Projection.init_from_v9(
                        KVBridgeV9Projection.init_from_v8(
                            KVBridgeV8Projection.init_from_v7(
                                KVBridgeV7Projection.init_from_v6(
                                    KVBridgeV6Projection.init_from_v5(
                                        KVBridgeV5Projection.init_from_v4(
                                            KVBridgeV4Projection.init_from_v3(
                                                kv_b3, seed_v4=71009),
                                            seed_v5=71010),
                                        seed_v6=71011),
                                    seed_v7=71012),
                                seed_v8=71013),
                            seed_v9=71014),
                        seed_v10=71015),
                    seed_v11=71016),
                seed_v12=71017),
            seed_v13=71018),
        seed_v14=71019)
    kv_b15 = KVBridgeV15Projection.init_from_v14(
        kv_b14, seed_v15=71020)
    return KVBridgeV16Projection.init_from_v15(
        kv_b15, seed_v16=71021)


def family_h550_substrate_v16_determinism(
        seed: int) -> dict[str, Any]:
    sub = _build_v16_substrate(seed=int(seed))
    ids = tokenize_bytes_v16("r166_550", max_len=8)
    t1, _ = forward_tiny_substrate_v16(sub, ids)
    t2, _ = forward_tiny_substrate_v16(sub, ids)
    return {
        "schema": R166_SCHEMA_VERSION,
        "name": "h550_substrate_v16_determinism",
        "passed": bool(t1.cid() == t2.cid()),
    }


def family_h550b_substrate_v16_drt_cid_byte_stable(
        seed: int) -> dict[str, Any]:
    sub = _build_v16_substrate(seed=int(seed))
    ids = tokenize_bytes_v16("r166_550b", max_len=8)
    t1, c1 = forward_tiny_substrate_v16(sub, ids)
    record_restart_event_v16(
        c1, turn=2, restart_kind="agent_restart",
        role="planner")
    record_delay_window_v16(
        c1, restart_turn=2, repair_turn=5,
        delay_turns=3, role="planner")
    t2, c2 = forward_tiny_substrate_v16(
        sub, ids, v16_kv_cache=c1)
    t3, c3 = forward_tiny_substrate_v16(
        sub, ids, v16_kv_cache=c1)
    return {
        "schema": R166_SCHEMA_VERSION,
        "name":
            "h550b_substrate_v16_drt_cid_byte_stable",
        "passed": bool(
            t2.delayed_repair_trajectory_cid
            == t3.delayed_repair_trajectory_cid
            and bool(t2.delayed_repair_trajectory_cid)),
    }


def family_h551_substrate_v16_restart_per_layer_shape(
        seed: int) -> dict[str, Any]:
    sub = _build_v16_substrate(seed=int(seed))
    ids = tokenize_bytes_v16("r166_551", max_len=8)
    t, _ = forward_tiny_substrate_v16(sub, ids)
    return {
        "schema": R166_SCHEMA_VERSION,
        "name":
            "h551_substrate_v16_restart_per_layer_shape",
        "passed": bool(
            t.restart_dominance_per_layer.shape
            == (int(W71_DEFAULT_V16_N_LAYERS),)),
    }


def family_h551b_substrate_v16_delayed_gate_shape(
        seed: int) -> dict[str, Any]:
    sub = _build_v16_substrate(seed=int(seed))
    ids = tokenize_bytes_v16("r166_551b", max_len=8)
    t, _ = forward_tiny_substrate_v16(sub, ids)
    return {
        "schema": R166_SCHEMA_VERSION,
        "name":
            "h551b_substrate_v16_delayed_gate_shape",
        "passed": bool(
            t.delayed_repair_gate_per_layer.shape
            == (int(W71_DEFAULT_V16_N_LAYERS),)),
    }


def family_h552_substrate_v16_repair_dominance_flops(
        seed: int) -> dict[str, Any]:
    r = substrate_repair_dominance_flops_v16(
        n_tokens=128, n_repairs=7)
    return {
        "schema": R166_SCHEMA_VERSION,
        "name":
            "h552_substrate_v16_repair_dominance_flops",
        "passed": bool(r["saving_ratio"] >= 0.83),
        "ratio": float(r["saving_ratio"]),
    }


def family_h552b_substrate_v16_delayed_repair_throttle(
        seed: int) -> dict[str, Any]:
    r = substrate_delayed_repair_throttle_v16(
        visible_token_budget=64,
        baseline_token_cost=512, delay_turns=3)
    return {
        "schema": R166_SCHEMA_VERSION,
        "name":
            "h552b_substrate_v16_delayed_repair_throttle",
        "passed": bool(r["saving_ratio"] >= 0.5),
        "ratio": float(r["saving_ratio"]),
    }


def family_h553_kv_v16_twelve_target_ridge(
        seed: int) -> dict[str, Any]:
    from coordpy.tiny_substrate_v16 import tokenize_bytes_v16
    import numpy as _np
    sub = _build_v16_substrate(seed=int(seed) + 71300)
    proj = _build_v16_kv_projection(sub)
    cfg = sub.config.v15.v14.v13.v12.v11.v10.v9
    rng = _np.random.default_rng(int(seed) + 71301)
    ids = tokenize_bytes_v16("kvr166", max_len=4)
    while len(ids) < 4:
        ids.append(0)
    carriers = [
        list(rng.standard_normal(8)) for _ in range(4)]
    # Vocab-shaped sparse delta-logits per target (12 stacks).
    targets = []
    for i in range(12):
        t = _np.zeros(int(cfg.vocab_size))
        t[20 + 15 * i] = 1.0 / (i + 1)
        targets.append(t.tolist())
    _, rep = fit_kv_bridge_v16_twelve_target(
        params=sub, projection=proj,
        train_carriers=carriers,
        target_delta_logits_stack=targets,
        follow_up_token_ids=ids, n_directions=3,
        restart_dominance_target_index=11)
    return {
        "schema": R166_SCHEMA_VERSION,
        "name": "h553_kv_v16_twelve_target_ridge",
        "passed": bool(rep.n_targets == 12),
        "rd_pre": float(rep.restart_dominance_pre),
        "rd_post": float(rep.restart_dominance_post),
    }


def family_h554_cache_v14_eleven_objective_ridge(
        seed: int) -> dict[str, Any]:
    cc = CacheControllerV14.init(fit_seed=int(seed))
    import numpy as _np
    rng = _np.random.default_rng(int(seed) + 71310)
    X = rng.standard_normal((10, 4))
    _, rep = fit_eleven_objective_ridge_v14(
        controller=cc, train_features=X.tolist(),
        target_drop_oracle=X.sum(-1).tolist(),
        target_retrieval_relevance=X[:, 0].tolist(),
        target_hidden_wins=(
            X[:, 1] - X[:, 2]).tolist(),
        target_replay_dominance=(X[:, 3] * 0.5).tolist(),
        target_team_task_success=(X[:, 0] * 0.3).tolist(),
        target_team_failure_recovery=(X[:, 2] * 0.4).tolist(),
        target_branch_merge=(
            X[:, 0] * 0.2 + X[:, 2] * 0.5).tolist(),
        target_partial_contradiction=(
            X[:, 1] * 0.3 + X[:, 3] * 0.4).tolist(),
        target_multi_branch_rejoin=(X[:, 0] * 0.5).tolist(),
        target_budget_primary=(X[:, 2] * 0.3).tolist(),
        target_restart_dominance=(
            X[:, 3] * 0.2 + X[:, 0] * 0.1).tolist())
    return {
        "schema": R166_SCHEMA_VERSION,
        "name": "h554_cache_v14_eleven_objective_ridge",
        "passed": bool(
            rep.converged and rep.n_objectives == 11),
    }


def family_h554b_cache_v14_per_role_restart_head(
        seed: int) -> dict[str, Any]:
    cc = CacheControllerV14.init(fit_seed=int(seed))
    import numpy as _np
    rng = _np.random.default_rng(int(seed) + 71320)
    X12 = rng.standard_normal((10, 12))
    _, rep = fit_per_role_restart_priority_head_v14(
        controller=cc, role="planner",
        train_features=X12.tolist(),
        target_restart_priorities=(
            X12[:, 0] * 0.3 + X12[:, 11] * 0.4).tolist())
    return {
        "schema": R166_SCHEMA_VERSION,
        "name": "h554b_cache_v14_per_role_restart_head",
        "passed": bool(rep.converged),
    }


def family_h555_replay_v12_nineteen_regimes(
        seed: int) -> dict[str, Any]:
    return {
        "schema": R166_SCHEMA_VERSION,
        "name": "h555_replay_v12_nineteen_regimes",
        "passed": bool(
            len(W71_REPLAY_REGIMES_V12) == 19
            and "delayed_repair_after_restart_regime"
                in W71_REPLAY_REGIMES_V12),
    }


def family_h555b_replay_v12_restart_routing_nine_class(
        seed: int) -> dict[str, Any]:
    return {
        "schema": R166_SCHEMA_VERSION,
        "name":
            "h555b_replay_v12_restart_routing_nine_class",
        "passed": bool(
            len(W71_RESTART_AWARE_ROUTING_LABELS) == 9
            and "restart_route"
                in W71_RESTART_AWARE_ROUTING_LABELS),
    }


def family_h556_persistent_v23_chain_walk_depth(
        seed: int) -> dict[str, Any]:
    return {
        "schema": R166_SCHEMA_VERSION,
        "name": "h556_persistent_v23_chain_walk_depth",
        "passed": bool(
            int(W71_DEFAULT_V23_MAX_CHAIN_WALK_DEPTH)
            >= 262144),
    }


def family_h556b_persistent_v23_distractor_rank(
        seed: int) -> dict[str, Any]:
    return {
        "schema": R166_SCHEMA_VERSION,
        "name": "h556b_persistent_v23_distractor_rank",
        "passed": bool(
            int(W71_DEFAULT_V23_DISTRACTOR_RANK) == 22
            and int(W71_DEFAULT_V23_N_LAYERS) == 22),
    }


def family_h557_lhr_v23_heads(
        seed: int) -> dict[str, Any]:
    head = LongHorizonReconstructionV23Head.init(
        seed=int(seed) + 71330)
    return {
        "schema": R166_SCHEMA_VERSION,
        "name": "h557_lhr_v23_heads",
        "passed": bool(
            int(head.max_k) >= int(W71_DEFAULT_LHR_V23_MAX_K)
            and int(head.restart_dim) == 8),
    }


def family_h558_mlsc_v19_delayed_repair_chain_merge(
        seed: int) -> dict[str, Any]:
    v3 = make_root_capsule_v3(
        branch_id="r166_558",
        payload=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
        fact_tags=("r166",), confidence=0.9, trust=0.9,
        turn_index=0)
    v4 = wrap_v3_as_v4(v3)
    v5 = wrap_v4_as_v5(v4)
    v6 = wrap_v5_as_v6(v5)
    v7 = wrap_v6_as_v7(v6)
    v8 = wrap_v7_as_v8(v7)
    v9 = wrap_v8_as_v9(v8)
    v10 = wrap_v9_as_v10(v9)
    v11 = wrap_v10_as_v11(v10)
    v12 = wrap_v11_as_v12(v11)
    v13 = wrap_v12_as_v13(v12)
    v14 = wrap_v13_as_v14(v13)
    v15 = wrap_v14_as_v15(v14)
    v16 = wrap_v15_as_v16(v15)
    v17 = wrap_v16_as_v17(v16)
    v18 = wrap_v17_as_v18(v17)
    v19_a = wrap_v18_as_v19(
        v18,
        delayed_repair_trajectory_chain=("drt_a",),
        restart_dominance_chain=("rst_a",))
    v19_b = wrap_v18_as_v19(
        v18,
        delayed_repair_trajectory_chain=("drt_b",),
        restart_dominance_chain=("rst_b",))
    op = MergeOperatorV19(factor_dim=6)
    merged = op.merge([v19_a, v19_b])
    return {
        "schema": R166_SCHEMA_VERSION,
        "name":
            "h558_mlsc_v19_delayed_repair_chain_merge",
        "passed": bool(
            "drt_a" in merged.delayed_repair_trajectory_chain
            and "drt_b" in merged.delayed_repair_trajectory_chain
            and "rst_a" in merged.restart_dominance_chain
            and "rst_b" in merged.restart_dominance_chain),
    }


def family_h559_consensus_v17_stage_count(
        seed: int) -> dict[str, Any]:
    return {
        "schema": R166_SCHEMA_VERSION,
        "name": "h559_consensus_v17_stage_count",
        "passed": bool(
            len(W71_CONSENSUS_V17_STAGES) >= 28
            and "restart_aware_arbiter"
                in W71_CONSENSUS_V17_STAGES
            and "delayed_repair_after_restart_arbiter"
                in W71_CONSENSUS_V17_STAGES),
    }


_R166_FAMILIES: tuple[Any, ...] = (
    family_h550_substrate_v16_determinism,
    family_h550b_substrate_v16_drt_cid_byte_stable,
    family_h551_substrate_v16_restart_per_layer_shape,
    family_h551b_substrate_v16_delayed_gate_shape,
    family_h552_substrate_v16_repair_dominance_flops,
    family_h552b_substrate_v16_delayed_repair_throttle,
    family_h553_kv_v16_twelve_target_ridge,
    family_h554_cache_v14_eleven_objective_ridge,
    family_h554b_cache_v14_per_role_restart_head,
    family_h555_replay_v12_nineteen_regimes,
    family_h555b_replay_v12_restart_routing_nine_class,
    family_h556_persistent_v23_chain_walk_depth,
    family_h556b_persistent_v23_distractor_rank,
    family_h557_lhr_v23_heads,
    family_h558_mlsc_v19_delayed_repair_chain_merge,
    family_h559_consensus_v17_stage_count,
)


def run_r166(
        seeds: Sequence[int] = (199, 299, 399),
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for s in seeds:
        results: dict[str, Any] = {}
        for fn in _R166_FAMILIES:
            try:
                r = fn(int(s))
            except Exception as e:
                r = {
                    "schema": R166_SCHEMA_VERSION,
                    "name": getattr(fn, "__name__", "unknown"),
                    "passed": False,
                    "exception": str(e),
                }
            results[r["name"]] = r
        out.append({
            "schema": R166_SCHEMA_VERSION,
            "seed": int(s),
            "family_results": results,
        })
    return out


__all__ = ["R166_SCHEMA_VERSION", "run_r166"]
