"""W73 R-174 benchmark family — Real Substrate Plane V18 (Plane B).

Exercises the W73 real substrate plane V18: 20-layer V18 substrate
forward (replacement-repair-trajectory CID, replacement-after-CTR
per layer, replacement-pressure gate per layer), KV V18 14-target
ridge, cache V16 thirteen-objective ridge + per-role replacement-
pressure head, replay V14 21-regime + replacement-aware routing
head, persistent V25 (24 layers, max_chain_walk_depth=1048576),
LHR V25 (24 heads, max_k=768), MLSC V21 (replacement-repair
chain), consensus V19 (32 stages), deep substrate hybrid V18 (18-
way).

H820..H835 cell families (16 H-bars):

* H820   Substrate V18 forward-determinism on (params, token_ids)
* H820b  Substrate V18 replacement-repair-trajectory CID byte-stable
* H821   Substrate V18 replacement-after-CTR per-layer shape (L=20,)
* H821b  Substrate V18 replacement-pressure gate shape (L=20,)
* H822   Substrate V18 replacement-dominance flops saving ≥ 0.92 over 9
* H822b  Substrate V18 replacement-pressure throttle saving ≥ 0.5
* H823   KV V18 fourteen-target ridge fit converges
* H824   Cache V16 thirteen-objective ridge converges
* H824b  Cache V16 per-role replacement-pressure head converges
* H825   Replay V14 21 regimes including
         ``replacement_after_contradiction_then_rejoin_regime``
* H825b  Replay V14 replacement-aware routing head 11-class
* H826   Persistent V25 max_chain_walk_depth = 1048576
* H826b  Persistent V25 distractor rank = 24
* H827   LHR V25 24 heads + max_k = 768
* H828   MLSC V21 replacement-repair-trajectory chain merge union
* H829   Consensus V19 has ≥ 32 stages
"""

from __future__ import annotations

from typing import Any, Sequence

from coordpy.cache_controller_v16 import (
    CacheControllerV16,
    fit_per_role_replacement_pressure_head_v16,
    fit_thirteen_objective_ridge_v16,
)
from coordpy.consensus_fallback_controller_v19 import (
    W73_CONSENSUS_V19_STAGES,
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
from coordpy.kv_bridge_v18 import (
    KVBridgeV18Projection, fit_kv_bridge_v18_fourteen_target,
)
from coordpy.long_horizon_retention_v25 import (
    LongHorizonReconstructionV25Head,
    W73_DEFAULT_LHR_V25_MAX_K,
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
from coordpy.mergeable_latent_capsule_v21 import (
    MergeOperatorV21, wrap_v20_as_v21,
)
from coordpy.persistent_latent_v25 import (
    W73_DEFAULT_V25_DISTRACTOR_RANK,
    W73_DEFAULT_V25_MAX_CHAIN_WALK_DEPTH,
    W73_DEFAULT_V25_N_LAYERS,
)
from coordpy.replay_controller_v14 import (
    W73_REPLACEMENT_AWARE_ROUTING_LABELS,
    W73_REPLAY_REGIMES_V14,
)
from coordpy.tiny_substrate_v18 import (
    W73_DEFAULT_V18_N_LAYERS,
    build_default_tiny_substrate_v18,
    forward_tiny_substrate_v18,
    record_contradiction_event_v18,
    record_replacement_event_v18,
    record_replacement_window_v18,
    substrate_replacement_dominance_flops_v18,
    substrate_replacement_pressure_throttle_v18,
    tokenize_bytes_v18,
)


R174_SCHEMA_VERSION: str = "coordpy.r174_benchmark.v1"


def _build_v18_substrate(seed: int = 73000):
    return build_default_tiny_substrate_v18(seed=int(seed))


def _build_v18_kv_projection(sub):
    cfg = sub.config.v17.v16.v15.v14.v13.v12.v11.v10.v9
    d_head = int(cfg.d_model) // int(cfg.n_heads)
    kv_b3 = KVBridgeV3Projection.init(
        n_layers=int(cfg.n_layers),
        n_heads=int(cfg.n_heads),
        n_kv_heads=int(cfg.n_kv_heads),
        n_inject_tokens=3, carrier_dim=6,
        d_head=int(d_head), seed=73008)
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
                                                    kv_b3, seed_v4=73009),
                                                seed_v5=73010),
                                            seed_v6=73011),
                                        seed_v7=73012),
                                    seed_v8=73013),
                                seed_v9=73014),
                            seed_v10=73015),
                        seed_v11=73016),
                    seed_v12=73017),
                seed_v13=73018),
            seed_v14=73019),
        seed_v15=73020)
    kv_b16 = KVBridgeV16Projection.init_from_v15(
        kv_b15, seed_v16=73021)
    kv_b17 = KVBridgeV17Projection.init_from_v16(
        kv_b16, seed_v17=73022)
    return KVBridgeV18Projection.init_from_v17(
        kv_b17, seed_v18=73023)


def family_h820_substrate_v18_determinism(
        seed: int) -> dict[str, Any]:
    """V18 forward determinism — at the V18-axis granularity."""
    import numpy as _np
    sub = _build_v18_substrate(seed=int(seed))
    ids = tokenize_bytes_v18("r174_820", max_len=8)
    forward_tiny_substrate_v18(sub, ids)
    t1, _ = forward_tiny_substrate_v18(sub, ids)
    t2, _ = forward_tiny_substrate_v18(sub, ids)
    return {
        "schema": R174_SCHEMA_VERSION,
        "name": "h820_substrate_v18_determinism",
        "passed": bool(
            t1.replacement_repair_trajectory_cid
            == t2.replacement_repair_trajectory_cid
            and _np.array_equal(
                t1.replacement_after_ctr_per_layer,
                t2.replacement_after_ctr_per_layer)
            and _np.array_equal(
                t1.replacement_pressure_gate_per_layer,
                t2.replacement_pressure_gate_per_layer)
            and _np.array_equal(
                t1.v18_gate_score_per_layer,
                t2.v18_gate_score_per_layer)),
    }


def family_h820b_substrate_v18_rrt_cid_byte_stable(
        seed: int) -> dict[str, Any]:
    sub = _build_v18_substrate(seed=int(seed))
    ids = tokenize_bytes_v18("r174_820b", max_len=8)
    t1, c1 = forward_tiny_substrate_v18(sub, ids)
    record_contradiction_event_v18(
        c1, turn=2,
        contradiction_kind="fact_contradiction",
        role="planner")
    record_replacement_event_v18(
        c1, turn=3, role="planner")
    record_replacement_window_v18(
        c1, contradiction_turn=2,
        replacement_turn=3, rejoin_turn=8,
        replacement_lag_turns=5, role="planner")
    t2, c2 = forward_tiny_substrate_v18(
        sub, ids, v18_kv_cache=c1)
    t3, _ = forward_tiny_substrate_v18(
        sub, ids, v18_kv_cache=c1)
    return {
        "schema": R174_SCHEMA_VERSION,
        "name":
            "h820b_substrate_v18_rrt_cid_byte_stable",
        "passed": bool(
            t2.replacement_repair_trajectory_cid
            == t3.replacement_repair_trajectory_cid
            and bool(t2.replacement_repair_trajectory_cid)),
    }


def family_h821_substrate_v18_replacement_per_layer_shape(
        seed: int) -> dict[str, Any]:
    sub = _build_v18_substrate(seed=int(seed))
    ids = tokenize_bytes_v18("r174_821", max_len=8)
    t, _ = forward_tiny_substrate_v18(sub, ids)
    return {
        "schema": R174_SCHEMA_VERSION,
        "name":
            "h821_substrate_v18_replacement_per_layer_shape",
        "passed": bool(
            t.replacement_after_ctr_per_layer.shape
            == (int(W73_DEFAULT_V18_N_LAYERS),)),
    }


def family_h821b_substrate_v18_replacement_gate_shape(
        seed: int) -> dict[str, Any]:
    sub = _build_v18_substrate(seed=int(seed))
    ids = tokenize_bytes_v18("r174_821b", max_len=8)
    t, _ = forward_tiny_substrate_v18(sub, ids)
    return {
        "schema": R174_SCHEMA_VERSION,
        "name":
            "h821b_substrate_v18_replacement_gate_shape",
        "passed": bool(
            t.replacement_pressure_gate_per_layer.shape
            == (int(W73_DEFAULT_V18_N_LAYERS),)),
    }


def family_h822_substrate_v18_replacement_dominance_flops(
        seed: int) -> dict[str, Any]:
    r = substrate_replacement_dominance_flops_v18(
        n_tokens=128, n_repairs=9)
    return {
        "schema": R174_SCHEMA_VERSION,
        "name":
            "h822_substrate_v18_replacement_dominance_flops",
        "passed": bool(r["saving_ratio"] >= 0.92),
        "ratio": float(r["saving_ratio"]),
    }


def family_h822b_substrate_v18_replacement_throttle(
        seed: int) -> dict[str, Any]:
    r = substrate_replacement_pressure_throttle_v18(
        visible_token_budget=64,
        baseline_token_cost=512, replacement_lag_turns=5)
    return {
        "schema": R174_SCHEMA_VERSION,
        "name":
            "h822b_substrate_v18_replacement_throttle",
        "passed": bool(r["saving_ratio"] >= 0.5),
        "ratio": float(r["saving_ratio"]),
    }


def family_h823_kv_v18_fourteen_target_ridge(
        seed: int) -> dict[str, Any]:
    import numpy as _np
    sub = _build_v18_substrate(seed=int(seed) + 73300)
    proj = _build_v18_kv_projection(sub)
    cfg = sub.config.v17.v16.v15.v14.v13.v12.v11.v10.v9
    rng = _np.random.default_rng(int(seed) + 73301)
    ids = tokenize_bytes_v18("kvr174", max_len=4)
    while len(ids) < 4:
        ids.append(0)
    carriers = [
        list(rng.standard_normal(8)) for _ in range(4)]
    targets = []
    for i in range(14):
        t = _np.zeros(int(cfg.vocab_size))
        t[20 + 15 * i] = 1.0 / (i + 1)
        targets.append(t.tolist())
    _, rep = fit_kv_bridge_v18_fourteen_target(
        params=sub, projection=proj,
        train_carriers=carriers,
        target_delta_logits_stack=targets,
        follow_up_token_ids=ids, n_directions=3,
        replacement_target_index=13)
    return {
        "schema": R174_SCHEMA_VERSION,
        "name": "h823_kv_v18_fourteen_target_ridge",
        "passed": bool(rep.n_targets == 14),
        "rep_pre": float(rep.replacement_pre),
        "rep_post": float(rep.replacement_post),
    }


def family_h824_cache_v16_thirteen_objective_ridge(
        seed: int) -> dict[str, Any]:
    cc = CacheControllerV16.init(fit_seed=int(seed))
    import numpy as _np
    rng = _np.random.default_rng(int(seed) + 73310)
    X = rng.standard_normal((10, 4))
    _, rep = fit_thirteen_objective_ridge_v16(
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
            + X[:, 2] * 0.3).tolist())
    return {
        "schema": R174_SCHEMA_VERSION,
        "name": "h824_cache_v16_thirteen_objective_ridge",
        "passed": bool(
            rep.converged and rep.n_objectives == 13),
    }


def family_h824b_cache_v16_per_role_replacement_head(
        seed: int) -> dict[str, Any]:
    cc = CacheControllerV16.init(fit_seed=int(seed))
    import numpy as _np
    rng = _np.random.default_rng(int(seed) + 73320)
    X14 = rng.standard_normal((10, 14))
    _, rep = fit_per_role_replacement_pressure_head_v16(
        controller=cc, role="planner",
        train_features=X14.tolist(),
        target_replacement_priorities=(
            X14[:, 0] * 0.3 + X14[:, 13] * 0.4).tolist())
    return {
        "schema": R174_SCHEMA_VERSION,
        "name": "h824b_cache_v16_per_role_replacement_head",
        "passed": bool(rep.converged),
    }


def family_h825_replay_v14_twenty_one_regimes(
        seed: int) -> dict[str, Any]:
    return {
        "schema": R174_SCHEMA_VERSION,
        "name": "h825_replay_v14_twenty_one_regimes",
        "passed": bool(
            len(W73_REPLAY_REGIMES_V14) == 21
            and (
                "replacement_after_contradiction_then_rejoin_regime"
                in W73_REPLAY_REGIMES_V14)),
    }


def family_h825b_replay_v14_replacement_routing_eleven_class(
        seed: int) -> dict[str, Any]:
    return {
        "schema": R174_SCHEMA_VERSION,
        "name":
            "h825b_replay_v14_replacement_routing_eleven_class",
        "passed": bool(
            len(W73_REPLACEMENT_AWARE_ROUTING_LABELS) == 11
            and "replacement_route"
                in W73_REPLACEMENT_AWARE_ROUTING_LABELS),
    }


def family_h826_persistent_v25_chain_walk_depth(
        seed: int) -> dict[str, Any]:
    return {
        "schema": R174_SCHEMA_VERSION,
        "name": "h826_persistent_v25_chain_walk_depth",
        "passed": bool(
            int(W73_DEFAULT_V25_MAX_CHAIN_WALK_DEPTH)
            >= 1048576),
    }


def family_h826b_persistent_v25_distractor_rank(
        seed: int) -> dict[str, Any]:
    return {
        "schema": R174_SCHEMA_VERSION,
        "name": "h826b_persistent_v25_distractor_rank",
        "passed": bool(
            int(W73_DEFAULT_V25_DISTRACTOR_RANK) == 24
            and int(W73_DEFAULT_V25_N_LAYERS) == 24),
    }


def family_h827_lhr_v25_heads(
        seed: int) -> dict[str, Any]:
    head = LongHorizonReconstructionV25Head.init(
        seed=int(seed) + 73330)
    return {
        "schema": R174_SCHEMA_VERSION,
        "name": "h827_lhr_v25_heads",
        "passed": bool(
            int(head.max_k) >= int(W73_DEFAULT_LHR_V25_MAX_K)
            and int(head.replacement_dim) == 8),
    }


def family_h828_mlsc_v21_replacement_chain_merge(
        seed: int) -> dict[str, Any]:
    v3 = make_root_capsule_v3(
        branch_id="r174_828",
        payload=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
        fact_tags=("r174",), confidence=0.9, trust=0.9,
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
    v21_a = wrap_v20_as_v21(
        v20,
        replacement_repair_trajectory_chain=("rep_a",),
        contradiction_chain=("ct_a",))
    v21_b = wrap_v20_as_v21(
        v20,
        replacement_repair_trajectory_chain=("rep_b",),
        contradiction_chain=("ct_b",))
    op = MergeOperatorV21(factor_dim=6)
    merged = op.merge([v21_a, v21_b])
    return {
        "schema": R174_SCHEMA_VERSION,
        "name":
            "h828_mlsc_v21_replacement_chain_merge",
        "passed": bool(
            "rep_a"
            in merged.replacement_repair_trajectory_chain
            and "rep_b"
            in merged.replacement_repair_trajectory_chain
            and "ct_a" in merged.contradiction_chain
            and "ct_b" in merged.contradiction_chain),
    }


def family_h829_consensus_v19_stage_count(
        seed: int) -> dict[str, Any]:
    return {
        "schema": R174_SCHEMA_VERSION,
        "name": "h829_consensus_v19_stage_count",
        "passed": bool(
            len(W73_CONSENSUS_V19_STAGES) >= 32
            and "replacement_pressure_arbiter"
                in W73_CONSENSUS_V19_STAGES
            and "replacement_after_contradiction_then_rejoin_arbiter"
                in W73_CONSENSUS_V19_STAGES),
    }


_R174_FAMILIES: tuple[Any, ...] = (
    family_h820_substrate_v18_determinism,
    family_h820b_substrate_v18_rrt_cid_byte_stable,
    family_h821_substrate_v18_replacement_per_layer_shape,
    family_h821b_substrate_v18_replacement_gate_shape,
    family_h822_substrate_v18_replacement_dominance_flops,
    family_h822b_substrate_v18_replacement_throttle,
    family_h823_kv_v18_fourteen_target_ridge,
    family_h824_cache_v16_thirteen_objective_ridge,
    family_h824b_cache_v16_per_role_replacement_head,
    family_h825_replay_v14_twenty_one_regimes,
    family_h825b_replay_v14_replacement_routing_eleven_class,
    family_h826_persistent_v25_chain_walk_depth,
    family_h826b_persistent_v25_distractor_rank,
    family_h827_lhr_v25_heads,
    family_h828_mlsc_v21_replacement_chain_merge,
    family_h829_consensus_v19_stage_count,
)


def run_r174(
        seeds: Sequence[int] = (199, 299, 399),
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for s in seeds:
        results: dict[str, Any] = {}
        for fn in _R174_FAMILIES:
            try:
                r = fn(int(s))
            except Exception as e:
                r = {
                    "schema": R174_SCHEMA_VERSION,
                    "name": getattr(fn, "__name__", "unknown"),
                    "passed": False,
                    "exception": str(e),
                }
            results[r["name"]] = r
        out.append({
            "schema": R174_SCHEMA_VERSION,
            "seed": int(s),
            "family_results": results,
        })
    return out


__all__ = ["R174_SCHEMA_VERSION", "run_r174"]
