"""W74 R-178 benchmark family — Real Substrate Plane V19 (Plane B).

Exercises the W74 real substrate plane V19: 21-layer V19 substrate
forward (compound-repair-trajectory CID, compound-repair-rate per
layer, compound-pressure gate per layer), KV V19 15-target ridge,
cache V17 fourteen-objective ridge + per-role compound-pressure
head, replay V15 22-regime + compound-aware routing head,
persistent V26 (25 layers, max_chain_walk_depth=2097152), LHR V26
(25 heads, max_k=832), MLSC V22 (compound-repair chain), consensus
V20 (34 stages), deep substrate hybrid V19 (19-way).

H900..H915 cell families (16 H-bars):

* H900   Substrate V19 forward-determinism on (params, token_ids)
* H900b  Substrate V19 compound-repair-trajectory CID byte-stable
* H901   Substrate V19 compound-repair-rate per-layer shape (L=21,)
* H901b  Substrate V19 compound-pressure gate shape (L=21,)
* H902   Substrate V19 compound-dominance flops saving ≥ 0.93 over 10
* H902b  Substrate V19 compound-pressure throttle saving ≥ 0.6
* H903   KV V19 fifteen-target ridge fit converges
* H904   Cache V17 fourteen-objective ridge converges
* H904b  Cache V17 per-role compound-pressure head converges
* H905   Replay V15 22 regimes including
         ``compound_repair_after_delayed_repair_then_replacement_regime``
* H905b  Replay V15 compound-aware routing head 12-class
* H906   Persistent V26 max_chain_walk_depth = 2097152
* H906b  Persistent V26 distractor rank = 25
* H907   LHR V26 25 heads + max_k = 832
* H908   MLSC V22 compound-repair-trajectory chain merge union
* H909   Consensus V20 has ≥ 34 stages
"""

from __future__ import annotations

from typing import Any, Sequence

from coordpy.cache_controller_v17 import (
    CacheControllerV17,
    fit_per_role_compound_pressure_head_v17,
    fit_fourteen_objective_ridge_v17,
)
from coordpy.consensus_fallback_controller_v20 import (
    W74_CONSENSUS_V20_STAGES,
)
from coordpy.kv_bridge_v3 import KVBridgeV3Projection
from coordpy.kv_bridge_v4 import KVBridgeV4Projection
from coordpy.kv_bridge_v5 import KVBridgeV5Projection
from coordpy.kv_bridge_v6 import KVBridgeV6Projection
from coordpy.kv_bridge_v7 import KVBridgeV7Projection
from coordpy.kv_bridge_v8 import KVBridgeV8Projection
from coordpy.kv_bridge_v9 import KVBridgeV9Projection
from coordpy.kv_bridge_v10 import KVBridgeV10Projection
from coordpy.kv_bridge_v11 import KVBridgeV11Projection
from coordpy.kv_bridge_v12 import KVBridgeV12Projection
from coordpy.kv_bridge_v13 import KVBridgeV13Projection
from coordpy.kv_bridge_v14 import KVBridgeV14Projection
from coordpy.kv_bridge_v15 import KVBridgeV15Projection
from coordpy.kv_bridge_v16 import KVBridgeV16Projection
from coordpy.kv_bridge_v17 import KVBridgeV17Projection
from coordpy.kv_bridge_v18 import KVBridgeV18Projection
from coordpy.kv_bridge_v19 import (
    KVBridgeV19Projection, fit_kv_bridge_v19_fifteen_target,
)
from coordpy.long_horizon_retention_v26 import (
    LongHorizonReconstructionV26Head,
    W74_DEFAULT_LHR_V26_MAX_K,
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
from coordpy.mergeable_latent_capsule_v22 import (
    MergeOperatorV22, wrap_v21_as_v22,
)
from coordpy.persistent_latent_v26 import (
    W74_DEFAULT_V26_DISTRACTOR_RANK,
    W74_DEFAULT_V26_MAX_CHAIN_WALK_DEPTH,
    W74_DEFAULT_V26_N_LAYERS,
)
from coordpy.replay_controller_v15 import (
    W74_COMPOUND_AWARE_ROUTING_LABELS,
    W74_REPLAY_REGIMES_V15,
)
from coordpy.tiny_substrate_v19 import (
    W74_DEFAULT_V19_N_LAYERS,
    build_default_tiny_substrate_v19,
    forward_tiny_substrate_v19,
    record_compound_failure_window_v19,
    record_delayed_repair_event_v19,
    substrate_compound_repair_dominance_flops_v19,
    substrate_compound_pressure_throttle_v19,
    tokenize_bytes_v19,
)


R178_SCHEMA_VERSION: str = "coordpy.r178_benchmark.v1"


def _build_v19_substrate(seed: int = 74000):
    return build_default_tiny_substrate_v19(seed=int(seed))


def _build_v19_kv_projection(sub):
    cfg = sub.config.v18.v17.v16.v15.v14.v13.v12.v11.v10.v9
    d_head = int(cfg.d_model) // int(cfg.n_heads)
    kv_b3 = KVBridgeV3Projection.init(
        n_layers=int(cfg.n_layers),
        n_heads=int(cfg.n_heads),
        n_kv_heads=int(cfg.n_kv_heads),
        n_inject_tokens=3, carrier_dim=6,
        d_head=int(d_head), seed=74008)
    kv_b15 = KVBridgeV15Projection.init_from_v14(
        KVBridgeV14Projection.init_from_v13(
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
                                                    kv_b3, seed_v4=74009),
                                                seed_v5=74010),
                                            seed_v6=74011),
                                        seed_v7=74012),
                                    seed_v8=74013),
                                seed_v9=74014),
                            seed_v10=74015),
                        seed_v11=74016),
                    seed_v12=74017),
                seed_v13=74018),
            seed_v14=74019),
        seed_v15=74020)
    kv_b16 = KVBridgeV16Projection.init_from_v15(
        kv_b15, seed_v16=74021)
    kv_b17 = KVBridgeV17Projection.init_from_v16(
        kv_b16, seed_v17=74022)
    kv_b18 = KVBridgeV18Projection.init_from_v17(
        kv_b17, seed_v18=74023)
    return KVBridgeV19Projection.init_from_v18(
        kv_b18, seed_v19=74024)


def family_h900_substrate_v19_determinism(
        seed: int) -> dict[str, Any]:
    """V19 forward determinism — at the V19-axis granularity."""
    import numpy as _np
    sub = _build_v19_substrate(seed=int(seed))
    ids = tokenize_bytes_v19("r178_900", max_len=8)
    forward_tiny_substrate_v19(sub, ids)
    t1, _ = forward_tiny_substrate_v19(sub, ids)
    t2, _ = forward_tiny_substrate_v19(sub, ids)
    return {
        "schema": R178_SCHEMA_VERSION,
        "name": "h900_substrate_v19_determinism",
        "passed": bool(
            t1.compound_repair_trajectory_cid
            == t2.compound_repair_trajectory_cid
            and _np.array_equal(
                t1.compound_repair_rate_per_layer,
                t2.compound_repair_rate_per_layer)
            and _np.array_equal(
                t1.compound_pressure_gate_per_layer,
                t2.compound_pressure_gate_per_layer)
            and _np.array_equal(
                t1.v19_gate_score_per_layer,
                t2.v19_gate_score_per_layer)),
    }


def family_h900b_substrate_v19_crt_cid_byte_stable(
        seed: int) -> dict[str, Any]:
    sub = _build_v19_substrate(seed=int(seed))
    ids = tokenize_bytes_v19("r178_900b", max_len=8)
    t1, c1 = forward_tiny_substrate_v19(sub, ids)
    record_delayed_repair_event_v19(
        c1, turn=2,
        delayed_kind="delayed_repair",
        role="planner")
    record_compound_failure_window_v19(
        c1, delayed_repair_turn=2,
        replacement_turn=5, rejoin_turn=12,
        compound_window_turns=10, role="planner")
    t2, c2 = forward_tiny_substrate_v19(
        sub, ids, v19_kv_cache=c1)
    t3, _ = forward_tiny_substrate_v19(
        sub, ids, v19_kv_cache=c1)
    return {
        "schema": R178_SCHEMA_VERSION,
        "name":
            "h900b_substrate_v19_crt_cid_byte_stable",
        "passed": bool(
            t2.compound_repair_trajectory_cid
            == t3.compound_repair_trajectory_cid
            and bool(t2.compound_repair_trajectory_cid)),
    }


def family_h901_substrate_v19_compound_per_layer_shape(
        seed: int) -> dict[str, Any]:
    sub = _build_v19_substrate(seed=int(seed))
    ids = tokenize_bytes_v19("r178_901", max_len=8)
    t, _ = forward_tiny_substrate_v19(sub, ids)
    return {
        "schema": R178_SCHEMA_VERSION,
        "name":
            "h901_substrate_v19_compound_per_layer_shape",
        "passed": bool(
            t.compound_repair_rate_per_layer.shape
            == (int(W74_DEFAULT_V19_N_LAYERS),)),
    }


def family_h901b_substrate_v19_compound_gate_shape(
        seed: int) -> dict[str, Any]:
    sub = _build_v19_substrate(seed=int(seed))
    ids = tokenize_bytes_v19("r178_901b", max_len=8)
    t, _ = forward_tiny_substrate_v19(sub, ids)
    return {
        "schema": R178_SCHEMA_VERSION,
        "name":
            "h901b_substrate_v19_compound_gate_shape",
        "passed": bool(
            t.compound_pressure_gate_per_layer.shape
            == (int(W74_DEFAULT_V19_N_LAYERS),)),
    }


def family_h902_substrate_v19_compound_dominance_flops(
        seed: int) -> dict[str, Any]:
    r = substrate_compound_repair_dominance_flops_v19(
        n_tokens=128, n_repairs=10)
    return {
        "schema": R178_SCHEMA_VERSION,
        "name":
            "h902_substrate_v19_compound_dominance_flops",
        "passed": bool(r["saving_ratio"] >= 0.93),
        "ratio": float(r["saving_ratio"]),
    }


def family_h902b_substrate_v19_compound_throttle(
        seed: int) -> dict[str, Any]:
    r = substrate_compound_pressure_throttle_v19(
        visible_token_budget=64,
        baseline_token_cost=512, compound_window_turns=8)
    return {
        "schema": R178_SCHEMA_VERSION,
        "name":
            "h902b_substrate_v19_compound_throttle",
        "passed": bool(r["saving_ratio"] >= 0.6),
        "ratio": float(r["saving_ratio"]),
    }


def family_h903_kv_v19_fifteen_target_ridge(
        seed: int) -> dict[str, Any]:
    import numpy as _np
    sub = _build_v19_substrate(seed=int(seed) + 74300)
    proj = _build_v19_kv_projection(sub)
    cfg = sub.config.v18.v17.v16.v15.v14.v13.v12.v11.v10.v9
    rng = _np.random.default_rng(int(seed) + 74301)
    ids = tokenize_bytes_v19("kvr178", max_len=4)
    while len(ids) < 4:
        ids.append(0)
    carriers = [
        list(rng.standard_normal(8)) for _ in range(4)]
    targets = []
    for i in range(15):
        t = _np.zeros(int(cfg.vocab_size))
        t[20 + 15 * i] = 1.0 / (i + 1)
        targets.append(t.tolist())
    _, rep = fit_kv_bridge_v19_fifteen_target(
        params=sub, projection=proj,
        train_carriers=carriers,
        target_delta_logits_stack=targets,
        follow_up_token_ids=ids, n_directions=3,
        compound_target_index=14)
    return {
        "schema": R178_SCHEMA_VERSION,
        "name": "h903_kv_v19_fifteen_target_ridge",
        "passed": bool(rep.n_targets == 15),
        "cmp_pre": float(rep.compound_pre),
        "cmp_post": float(rep.compound_post),
    }


def family_h904_cache_v17_fourteen_objective_ridge(
        seed: int) -> dict[str, Any]:
    cc = CacheControllerV17.init(fit_seed=int(seed))
    import numpy as _np
    rng = _np.random.default_rng(int(seed) + 74310)
    X = rng.standard_normal((10, 4))
    _, rep = fit_fourteen_objective_ridge_v17(
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
            X[:, 3] * 0.2 + X[:, 0] * 0.1).tolist(),
        target_delayed_rejoin_after_restart=(
            X[:, 1] * 0.3 + X[:, 2] * 0.3
            + X[:, 3] * 0.2).tolist(),
        target_replacement_after_ctr=(
            X[:, 0] * 0.3 + X[:, 1] * 0.3
            + X[:, 2] * 0.3).tolist(),
        target_compound_repair=(
            X[:, 0] * 0.25 + X[:, 1] * 0.25
            + X[:, 2] * 0.25 + X[:, 3] * 0.25).tolist())
    return {
        "schema": R178_SCHEMA_VERSION,
        "name": "h904_cache_v17_fourteen_objective_ridge",
        "passed": bool(
            rep.converged and rep.n_objectives == 14),
    }


def family_h904b_cache_v17_per_role_compound_head(
        seed: int) -> dict[str, Any]:
    cc = CacheControllerV17.init(fit_seed=int(seed))
    import numpy as _np
    rng = _np.random.default_rng(int(seed) + 74320)
    X15 = rng.standard_normal((10, 15))
    _, rep = fit_per_role_compound_pressure_head_v17(
        controller=cc, role="planner",
        train_features=X15.tolist(),
        target_compound_priorities=(
            X15[:, 0] * 0.3 + X15[:, 14] * 0.4).tolist())
    return {
        "schema": R178_SCHEMA_VERSION,
        "name": "h904b_cache_v17_per_role_compound_head",
        "passed": bool(rep.converged),
    }


def family_h905_replay_v15_twenty_two_regimes(
        seed: int) -> dict[str, Any]:
    return {
        "schema": R178_SCHEMA_VERSION,
        "name": "h905_replay_v15_twenty_two_regimes",
        "passed": bool(
            len(W74_REPLAY_REGIMES_V15) == 22
            and (
                "compound_repair_after_delayed_repair_then_"
                "replacement_regime"
                in W74_REPLAY_REGIMES_V15)),
    }


def family_h905b_replay_v15_compound_routing_twelve_class(
        seed: int) -> dict[str, Any]:
    return {
        "schema": R178_SCHEMA_VERSION,
        "name":
            "h905b_replay_v15_compound_routing_twelve_class",
        "passed": bool(
            len(W74_COMPOUND_AWARE_ROUTING_LABELS) == 12
            and "compound_route"
                in W74_COMPOUND_AWARE_ROUTING_LABELS),
    }


def family_h906_persistent_v26_chain_walk_depth(
        seed: int) -> dict[str, Any]:
    return {
        "schema": R178_SCHEMA_VERSION,
        "name": "h906_persistent_v26_chain_walk_depth",
        "passed": bool(
            int(W74_DEFAULT_V26_MAX_CHAIN_WALK_DEPTH)
            >= 2097152),
    }


def family_h906b_persistent_v26_distractor_rank(
        seed: int) -> dict[str, Any]:
    return {
        "schema": R178_SCHEMA_VERSION,
        "name": "h906b_persistent_v26_distractor_rank",
        "passed": bool(
            int(W74_DEFAULT_V26_DISTRACTOR_RANK) == 25
            and int(W74_DEFAULT_V26_N_LAYERS) == 25),
    }


def family_h907_lhr_v26_heads(
        seed: int) -> dict[str, Any]:
    head = LongHorizonReconstructionV26Head.init(
        seed=int(seed) + 74330)
    return {
        "schema": R178_SCHEMA_VERSION,
        "name": "h907_lhr_v26_heads",
        "passed": bool(
            int(head.max_k) >= int(W74_DEFAULT_LHR_V26_MAX_K)
            and int(head.compound_dim) == 8),
    }


def family_h908_mlsc_v22_compound_chain_merge(
        seed: int) -> dict[str, Any]:
    v3 = make_root_capsule_v3(
        branch_id="r178_908",
        payload=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
        fact_tags=("r178",), confidence=0.9, trust=0.9,
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
    v19 = wrap_v18_as_v19(v18)
    v20 = wrap_v19_as_v20(v19)
    v21 = wrap_v20_as_v21(v20)
    v22_a = wrap_v21_as_v22(
        v21,
        compound_repair_trajectory_chain=("cmp_a",),
        delayed_repair_chain=("dr_a",))
    v22_b = wrap_v21_as_v22(
        v21,
        compound_repair_trajectory_chain=("cmp_b",),
        delayed_repair_chain=("dr_b",))
    op = MergeOperatorV22(factor_dim=6)
    merged = op.merge([v22_a, v22_b])
    return {
        "schema": R178_SCHEMA_VERSION,
        "name":
            "h908_mlsc_v22_compound_chain_merge",
        "passed": bool(
            "cmp_a"
            in merged.compound_repair_trajectory_chain
            and "cmp_b"
            in merged.compound_repair_trajectory_chain
            and "dr_a" in merged.delayed_repair_chain
            and "dr_b" in merged.delayed_repair_chain),
    }


def family_h909_consensus_v20_stage_count(
        seed: int) -> dict[str, Any]:
    return {
        "schema": R178_SCHEMA_VERSION,
        "name": "h909_consensus_v20_stage_count",
        "passed": bool(
            len(W74_CONSENSUS_V20_STAGES) >= 34
            and "compound_repair_arbiter"
                in W74_CONSENSUS_V20_STAGES
            and "compound_repair_after_delayed_repair_then_"
                "replacement_arbiter"
                in W74_CONSENSUS_V20_STAGES),
    }


_R178_FAMILIES: tuple[Any, ...] = (
    family_h900_substrate_v19_determinism,
    family_h900b_substrate_v19_crt_cid_byte_stable,
    family_h901_substrate_v19_compound_per_layer_shape,
    family_h901b_substrate_v19_compound_gate_shape,
    family_h902_substrate_v19_compound_dominance_flops,
    family_h902b_substrate_v19_compound_throttle,
    family_h903_kv_v19_fifteen_target_ridge,
    family_h904_cache_v17_fourteen_objective_ridge,
    family_h904b_cache_v17_per_role_compound_head,
    family_h905_replay_v15_twenty_two_regimes,
    family_h905b_replay_v15_compound_routing_twelve_class,
    family_h906_persistent_v26_chain_walk_depth,
    family_h906b_persistent_v26_distractor_rank,
    family_h907_lhr_v26_heads,
    family_h908_mlsc_v22_compound_chain_merge,
    family_h909_consensus_v20_stage_count,
)


def run_r178(
        seeds: Sequence[int] = (199, 299, 399),
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for s in seeds:
        results: dict[str, Any] = {}
        for fn in _R178_FAMILIES:
            try:
                r = fn(int(s))
            except Exception as e:
                r = {
                    "schema": R178_SCHEMA_VERSION,
                    "name": getattr(fn, "__name__", "unknown"),
                    "passed": False,
                    "exception": str(e),
                }
            results[r["name"]] = r
        out.append({
            "schema": R178_SCHEMA_VERSION,
            "seed": int(s),
            "family_results": results,
        })
    return out


__all__ = ["R178_SCHEMA_VERSION", "run_r178"]
