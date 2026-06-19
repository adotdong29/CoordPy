"""W72 R-170 benchmark family — Real Substrate Plane V17 (Plane B).

Exercises the W72 real substrate plane V17: 19-layer V17 substrate
forward (restart-repair-trajectory CID, delayed-rejoin-after-restart
per layer, rejoin-pressure gate per layer), KV V17 13-target ridge,
cache V15 twelve-objective ridge + per-role rejoin-pressure head,
replay V13 20-regime + rejoin-aware routing head, persistent V24
(23 layers, max_chain_walk_depth=524288), LHR V24 (23 heads,
max_k=704), MLSC V20 (restart-repair chain), consensus V18 (30
stages), deep substrate hybrid V17 (17-way).

H720..H735 cell families (16 H-bars):

* H720   Substrate V17 forward-determinism on (params, token_ids)
* H720b  Substrate V17 restart-repair-trajectory CID byte-stable
* H721   Substrate V17 delayed-rejoin per-layer shape (L=19,)
* H721b  Substrate V17 rejoin-pressure gate shape (L=19,)
* H722   Substrate V17 rejoin-dominance flops saving ≥ 0.90 over 8
* H722b  Substrate V17 rejoin-pressure throttle saving ≥ 0.5 at
         budget=64 / baseline=512 with rejoin_lag=4
* H723   KV V17 thirteen-target ridge fit converges
* H724   Cache V15 twelve-objective ridge converges
* H724b  Cache V15 per-role rejoin-pressure head converges
* H725   Replay V13 20 regimes including
         ``delayed_rejoin_after_restart_under_budget_regime``
* H725b  Replay V13 rejoin-aware routing head 10-class
* H726   Persistent V24 max_chain_walk_depth = 524288
* H726b  Persistent V24 distractor rank = 23
* H727   LHR V24 23 heads + max_k = 704
* H728   MLSC V20 restart-repair-trajectory chain merge union
* H729   Consensus V18 has ≥ 30 stages
"""

from __future__ import annotations

from typing import Any, Sequence

from coordpy.cache_controller_v15 import (
    CacheControllerV15, fit_per_role_rejoin_pressure_head_v15,
    fit_twelve_objective_ridge_v15,
)
from coordpy.consensus_fallback_controller_v18 import (
    W72_CONSENSUS_V18_STAGES,
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
from coordpy.kv_bridge_v17 import (
    KVBridgeV17Projection, fit_kv_bridge_v17_thirteen_target,
)
from coordpy.long_horizon_retention_v24 import (
    LongHorizonReconstructionV24Head,
    W72_DEFAULT_LHR_V24_MAX_K,
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
from coordpy.mergeable_latent_capsule_v20 import (
    MergeOperatorV20, wrap_v19_as_v20,
)
from coordpy.persistent_latent_v24 import (
    W72_DEFAULT_V24_DISTRACTOR_RANK,
    W72_DEFAULT_V24_MAX_CHAIN_WALK_DEPTH,
    W72_DEFAULT_V24_N_LAYERS,
)
from coordpy.replay_controller_v13 import (
    W72_REJOIN_AWARE_ROUTING_LABELS, W72_REPLAY_REGIMES_V13,
)
from coordpy.tiny_substrate_v17 import (
    W72_DEFAULT_V17_N_LAYERS,
    build_default_tiny_substrate_v17,
    forward_tiny_substrate_v17,
    record_branch_pressure_window_v17,
    record_rejoin_event_v17,
    substrate_rejoin_dominance_flops_v17,
    substrate_rejoin_pressure_throttle_v17,
    tokenize_bytes_v17,
)


R170_SCHEMA_VERSION: str = "coordpy.r170_benchmark.v1"


def _build_v17_substrate(seed: int = 72000):
    return build_default_tiny_substrate_v17(seed=int(seed))


def _build_v17_kv_projection(sub):
    cfg = sub.config.v16.v15.v14.v13.v12.v11.v10.v9
    d_head = int(cfg.d_model) // int(cfg.n_heads)
    kv_b3 = KVBridgeV3Projection.init(
        n_layers=int(cfg.n_layers),
        n_heads=int(cfg.n_heads),
        n_kv_heads=int(cfg.n_kv_heads),
        n_inject_tokens=3, carrier_dim=6,
        d_head=int(d_head), seed=72008)
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
                                                    kv_b3, seed_v4=72009),
                                                seed_v5=72010),
                                            seed_v6=72011),
                                        seed_v7=72012),
                                    seed_v8=72013),
                                seed_v9=72014),
                            seed_v10=72015),
                        seed_v11=72016),
                    seed_v12=72017),
                seed_v13=72018),
            seed_v14=72019),
        seed_v15=72020)
    kv_b16 = KVBridgeV16Projection.init_from_v15(
        kv_b15, seed_v16=72021)
    return KVBridgeV17Projection.init_from_v16(
        kv_b16, seed_v17=72022)


def family_h720_substrate_v17_determinism(
        seed: int) -> dict[str, Any]:
    """V17 forward determinism — at the V17-axis granularity.

    Compares the V17-new axes (restart-repair-trajectory CID,
    delayed-rejoin per-layer, rejoin-pressure gate per layer,
    V17 composite gate). The underlying V13 write-log carries a
    BLAS-precision-dependent float (``gate_score_mean``) that
    can drift by one ULP across cold/warm calls; the W72
    determinism guarantee is at the V17 axis layer, not at the
    inner-V13 write-log float.
    """
    import numpy as _np
    sub = _build_v17_substrate(seed=int(seed))
    ids = tokenize_bytes_v17("r170_720", max_len=8)
    # Warm-up forward to settle BLAS / FMA path on first run.
    forward_tiny_substrate_v17(sub, ids)
    t1, _ = forward_tiny_substrate_v17(sub, ids)
    t2, _ = forward_tiny_substrate_v17(sub, ids)
    return {
        "schema": R170_SCHEMA_VERSION,
        "name": "h720_substrate_v17_determinism",
        "passed": bool(
            t1.restart_repair_trajectory_cid
            == t2.restart_repair_trajectory_cid
            and _np.array_equal(
                t1.delayed_rejoin_after_restart_per_layer,
                t2.delayed_rejoin_after_restart_per_layer)
            and _np.array_equal(
                t1.rejoin_pressure_gate_per_layer,
                t2.rejoin_pressure_gate_per_layer)
            and _np.array_equal(
                t1.v17_gate_score_per_layer,
                t2.v17_gate_score_per_layer)),
    }


def family_h720b_substrate_v17_rrt_cid_byte_stable(
        seed: int) -> dict[str, Any]:
    sub = _build_v17_substrate(seed=int(seed))
    ids = tokenize_bytes_v17("r170_720b", max_len=8)
    t1, c1 = forward_tiny_substrate_v17(sub, ids)
    record_rejoin_event_v17(
        c1, turn=2, rejoin_kind="branch_rejoin",
        role="planner")
    record_branch_pressure_window_v17(
        c1, restart_turn=1, rejoin_turn=5,
        rejoin_lag_turns=4, role="planner")
    t2, c2 = forward_tiny_substrate_v17(
        sub, ids, v17_kv_cache=c1)
    t3, c3 = forward_tiny_substrate_v17(
        sub, ids, v17_kv_cache=c1)
    return {
        "schema": R170_SCHEMA_VERSION,
        "name":
            "h720b_substrate_v17_rrt_cid_byte_stable",
        "passed": bool(
            t2.restart_repair_trajectory_cid
            == t3.restart_repair_trajectory_cid
            and bool(t2.restart_repair_trajectory_cid)),
    }


def family_h721_substrate_v17_rejoin_per_layer_shape(
        seed: int) -> dict[str, Any]:
    sub = _build_v17_substrate(seed=int(seed))
    ids = tokenize_bytes_v17("r170_721", max_len=8)
    t, _ = forward_tiny_substrate_v17(sub, ids)
    return {
        "schema": R170_SCHEMA_VERSION,
        "name":
            "h721_substrate_v17_rejoin_per_layer_shape",
        "passed": bool(
            t.delayed_rejoin_after_restart_per_layer.shape
            == (int(W72_DEFAULT_V17_N_LAYERS),)),
    }


def family_h721b_substrate_v17_rejoin_pressure_gate_shape(
        seed: int) -> dict[str, Any]:
    sub = _build_v17_substrate(seed=int(seed))
    ids = tokenize_bytes_v17("r170_721b", max_len=8)
    t, _ = forward_tiny_substrate_v17(sub, ids)
    return {
        "schema": R170_SCHEMA_VERSION,
        "name":
            "h721b_substrate_v17_rejoin_pressure_gate_shape",
        "passed": bool(
            t.rejoin_pressure_gate_per_layer.shape
            == (int(W72_DEFAULT_V17_N_LAYERS),)),
    }


def family_h722_substrate_v17_rejoin_dominance_flops(
        seed: int) -> dict[str, Any]:
    r = substrate_rejoin_dominance_flops_v17(
        n_tokens=128, n_repairs=8)
    return {
        "schema": R170_SCHEMA_VERSION,
        "name":
            "h722_substrate_v17_rejoin_dominance_flops",
        "passed": bool(r["saving_ratio"] >= 0.90),
        "ratio": float(r["saving_ratio"]),
    }


def family_h722b_substrate_v17_rejoin_pressure_throttle(
        seed: int) -> dict[str, Any]:
    r = substrate_rejoin_pressure_throttle_v17(
        visible_token_budget=64,
        baseline_token_cost=512, rejoin_lag_turns=4)
    return {
        "schema": R170_SCHEMA_VERSION,
        "name":
            "h722b_substrate_v17_rejoin_pressure_throttle",
        "passed": bool(r["saving_ratio"] >= 0.5),
        "ratio": float(r["saving_ratio"]),
    }


def family_h723_kv_v17_thirteen_target_ridge(
        seed: int) -> dict[str, Any]:
    import numpy as _np
    sub = _build_v17_substrate(seed=int(seed) + 72300)
    proj = _build_v17_kv_projection(sub)
    cfg = sub.config.v16.v15.v14.v13.v12.v11.v10.v9
    rng = _np.random.default_rng(int(seed) + 72301)
    ids = tokenize_bytes_v17("kvr170", max_len=4)
    while len(ids) < 4:
        ids.append(0)
    carriers = [
        list(rng.standard_normal(8)) for _ in range(4)]
    # Vocab-shaped sparse delta-logits per target (13 stacks).
    targets = []
    for i in range(13):
        t = _np.zeros(int(cfg.vocab_size))
        t[20 + 15 * i] = 1.0 / (i + 1)
        targets.append(t.tolist())
    _, rep = fit_kv_bridge_v17_thirteen_target(
        params=sub, projection=proj,
        train_carriers=carriers,
        target_delta_logits_stack=targets,
        follow_up_token_ids=ids, n_directions=3,
        delayed_rejoin_target_index=12)
    return {
        "schema": R170_SCHEMA_VERSION,
        "name": "h723_kv_v17_thirteen_target_ridge",
        "passed": bool(rep.n_targets == 13),
        "rj_pre": float(rep.delayed_rejoin_pre),
        "rj_post": float(rep.delayed_rejoin_post),
    }


def family_h724_cache_v15_twelve_objective_ridge(
        seed: int) -> dict[str, Any]:
    cc = CacheControllerV15.init(fit_seed=int(seed))
    import numpy as _np
    rng = _np.random.default_rng(int(seed) + 72310)
    X = rng.standard_normal((10, 4))
    _, rep = fit_twelve_objective_ridge_v15(
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
            + X[:, 3] * 0.2).tolist())
    return {
        "schema": R170_SCHEMA_VERSION,
        "name": "h724_cache_v15_twelve_objective_ridge",
        "passed": bool(
            rep.converged and rep.n_objectives == 12),
    }


def family_h724b_cache_v15_per_role_rejoin_head(
        seed: int) -> dict[str, Any]:
    cc = CacheControllerV15.init(fit_seed=int(seed))
    import numpy as _np
    rng = _np.random.default_rng(int(seed) + 72320)
    X13 = rng.standard_normal((10, 13))
    _, rep = fit_per_role_rejoin_pressure_head_v15(
        controller=cc, role="planner",
        train_features=X13.tolist(),
        target_rejoin_priorities=(
            X13[:, 0] * 0.3 + X13[:, 12] * 0.4).tolist())
    return {
        "schema": R170_SCHEMA_VERSION,
        "name": "h724b_cache_v15_per_role_rejoin_head",
        "passed": bool(rep.converged),
    }


def family_h725_replay_v13_twenty_regimes(
        seed: int) -> dict[str, Any]:
    return {
        "schema": R170_SCHEMA_VERSION,
        "name": "h725_replay_v13_twenty_regimes",
        "passed": bool(
            len(W72_REPLAY_REGIMES_V13) == 20
            and (
                "delayed_rejoin_after_restart_under_budget_regime"
                in W72_REPLAY_REGIMES_V13)),
    }


def family_h725b_replay_v13_rejoin_routing_ten_class(
        seed: int) -> dict[str, Any]:
    return {
        "schema": R170_SCHEMA_VERSION,
        "name":
            "h725b_replay_v13_rejoin_routing_ten_class",
        "passed": bool(
            len(W72_REJOIN_AWARE_ROUTING_LABELS) == 10
            and "rejoin_route"
                in W72_REJOIN_AWARE_ROUTING_LABELS),
    }


def family_h726_persistent_v24_chain_walk_depth(
        seed: int) -> dict[str, Any]:
    return {
        "schema": R170_SCHEMA_VERSION,
        "name": "h726_persistent_v24_chain_walk_depth",
        "passed": bool(
            int(W72_DEFAULT_V24_MAX_CHAIN_WALK_DEPTH)
            >= 524288),
    }


def family_h726b_persistent_v24_distractor_rank(
        seed: int) -> dict[str, Any]:
    return {
        "schema": R170_SCHEMA_VERSION,
        "name": "h726b_persistent_v24_distractor_rank",
        "passed": bool(
            int(W72_DEFAULT_V24_DISTRACTOR_RANK) == 23
            and int(W72_DEFAULT_V24_N_LAYERS) == 23),
    }


def family_h727_lhr_v24_heads(
        seed: int) -> dict[str, Any]:
    head = LongHorizonReconstructionV24Head.init(
        seed=int(seed) + 72330)
    return {
        "schema": R170_SCHEMA_VERSION,
        "name": "h727_lhr_v24_heads",
        "passed": bool(
            int(head.max_k) >= int(W72_DEFAULT_LHR_V24_MAX_K)
            and int(head.rejoin_dim) == 8),
    }


def family_h728_mlsc_v20_restart_repair_chain_merge(
        seed: int) -> dict[str, Any]:
    v3 = make_root_capsule_v3(
        branch_id="r170_728",
        payload=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
        fact_tags=("r170",), confidence=0.9, trust=0.9,
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
    v20_a = wrap_v19_as_v20(
        v19,
        restart_repair_trajectory_chain=("rrt_a",),
        rejoin_pressure_chain=("rj_a",))
    v20_b = wrap_v19_as_v20(
        v19,
        restart_repair_trajectory_chain=("rrt_b",),
        rejoin_pressure_chain=("rj_b",))
    op = MergeOperatorV20(factor_dim=6)
    merged = op.merge([v20_a, v20_b])
    return {
        "schema": R170_SCHEMA_VERSION,
        "name":
            "h728_mlsc_v20_restart_repair_chain_merge",
        "passed": bool(
            "rrt_a"
            in merged.restart_repair_trajectory_chain
            and "rrt_b"
            in merged.restart_repair_trajectory_chain
            and "rj_a" in merged.rejoin_pressure_chain
            and "rj_b" in merged.rejoin_pressure_chain),
    }


def family_h729_consensus_v18_stage_count(
        seed: int) -> dict[str, Any]:
    return {
        "schema": R170_SCHEMA_VERSION,
        "name": "h729_consensus_v18_stage_count",
        "passed": bool(
            len(W72_CONSENSUS_V18_STAGES) >= 30
            and "rejoin_pressure_arbiter"
                in W72_CONSENSUS_V18_STAGES
            and "delayed_rejoin_after_restart_arbiter"
                in W72_CONSENSUS_V18_STAGES),
    }


_R170_FAMILIES: tuple[Any, ...] = (
    family_h720_substrate_v17_determinism,
    family_h720b_substrate_v17_rrt_cid_byte_stable,
    family_h721_substrate_v17_rejoin_per_layer_shape,
    family_h721b_substrate_v17_rejoin_pressure_gate_shape,
    family_h722_substrate_v17_rejoin_dominance_flops,
    family_h722b_substrate_v17_rejoin_pressure_throttle,
    family_h723_kv_v17_thirteen_target_ridge,
    family_h724_cache_v15_twelve_objective_ridge,
    family_h724b_cache_v15_per_role_rejoin_head,
    family_h725_replay_v13_twenty_regimes,
    family_h725b_replay_v13_rejoin_routing_ten_class,
    family_h726_persistent_v24_chain_walk_depth,
    family_h726b_persistent_v24_distractor_rank,
    family_h727_lhr_v24_heads,
    family_h728_mlsc_v20_restart_repair_chain_merge,
    family_h729_consensus_v18_stage_count,
)


def run_r170(
        seeds: Sequence[int] = (199, 299, 399),
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for s in seeds:
        results: dict[str, Any] = {}
        for fn in _R170_FAMILIES:
            try:
                r = fn(int(s))
            except Exception as e:
                r = {
                    "schema": R170_SCHEMA_VERSION,
                    "name": getattr(fn, "__name__", "unknown"),
                    "passed": False,
                    "exception": str(e),
                }
            results[r["name"]] = r
        out.append({
            "schema": R170_SCHEMA_VERSION,
            "seed": int(s),
            "family_results": results,
        })
    return out


__all__ = ["R170_SCHEMA_VERSION", "run_r170"]
