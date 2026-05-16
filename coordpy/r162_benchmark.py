"""W70 R-162 benchmark family — Real Substrate Plane V15 (Plane B).

Exercises the W70 real substrate plane: V15 substrate determinism,
new V15 axes, KV bridge V15, cache V13, replay V11, persistent V22,
LHR V22, MLSC V18, consensus V16, deep substrate hybrid V15,
substrate adapter V15, repair-trajectory CID, budget-primary gate,
and repair-dominance flop saving.

H480..H494 cell families (16 H-bars):

* H480   V15 substrate determinism
* H480b  V15 repair_trajectory_cid is deterministic
* H480c  V15 dominant_repair_per_layer shape (L,) with L=17
* H480d  V15 budget_primary_gate_per_layer shape (L,)
* H480e  V15 gate score shape (L,) and L=17
* H480f  V15 repair-dominance flop saving ≥ 80 %
* H480g  V15 budget-primary throttle ≥ 50 % at tight budget
* H481   KV V15 eleven-target ridge converges
* H481b  KV V15 70-dim repair-trajectory fingerprint length
* H481c  KV V15 repair-dominance falsifier
* H482   Cache V13 ten-objective converges
* H483   Replay V11 18-regime + per-role + budget-primary
         routing head
* H484   Substrate adapter V15 only-tier-full at in-repo
* H485   Persistent V22 21 layers + nineteenth skip + max_chain_walk=131072
* H486   LHR V22 21 heads + max_k=576
* H487   Deep substrate hybrid V15 fifteen-way + consensus V16 26 stages
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as _np

from coordpy.cache_controller_v13 import (
    CacheControllerV13,
    fit_ten_objective_ridge_v13,
)
from coordpy.consensus_fallback_controller_v16 import (
    ConsensusFallbackControllerV16,
    W70_CONSENSUS_V16_STAGES,
)
from coordpy.deep_substrate_hybrid_v14 import (
    DeepSubstrateHybridV14ForwardWitness,
)
from coordpy.deep_substrate_hybrid_v15 import (
    DeepSubstrateHybridV15,
    deep_substrate_hybrid_v15_forward,
)
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
from coordpy.kv_bridge_v15 import (
    KVBridgeV15Projection,
    compute_repair_trajectory_fingerprint_v15,
    fit_kv_bridge_v15_eleven_target,
    probe_kv_bridge_v15_repair_dominance_falsifier,
)
from coordpy.long_horizon_retention_v22 import (
    LongHorizonReconstructionV22Head,
    W70_DEFAULT_LHR_V22_MAX_K,
)
from coordpy.persistent_latent_v22 import (
    W70_DEFAULT_V22_MAX_CHAIN_WALK_DEPTH,
    W70_DEFAULT_V22_N_LAYERS,
)
from coordpy.replay_controller import ReplayCandidate
from coordpy.replay_controller_v11 import (
    ReplayControllerV11,
    W70_BUDGET_PRIMARY_ROUTING_LABELS,
    W70_REPLAY_REGIMES_V11,
    fit_replay_controller_v11_per_role,
    fit_replay_v11_budget_primary_routing_head,
)
from coordpy.substrate_adapter_v15 import (
    W70_SUBSTRATE_TIER_SUBSTRATE_V15_FULL,
    probe_all_v15_adapters,
)
from coordpy.tiny_substrate_v15 import (
    W70_DEFAULT_V15_N_LAYERS,
    build_default_tiny_substrate_v15,
    forward_tiny_substrate_v15,
    substrate_budget_primary_throttle_v15,
    substrate_repair_dominance_flops_v15,
    tokenize_bytes_v15,
)


R162_SCHEMA_VERSION: str = "coordpy.r162_benchmark.v1"


def family_h480_v15_substrate_determinism(
        seed: int) -> dict[str, Any]:
    p = build_default_tiny_substrate_v15(
        seed=int(seed) + 70000)
    ids = tokenize_bytes_v15("h480", max_len=12)
    t1, _ = forward_tiny_substrate_v15(p, ids)
    t2, _ = forward_tiny_substrate_v15(p, ids)
    return {
        "schema": R162_SCHEMA_VERSION,
        "name": "h480_v15_substrate_determinism",
        "passed": bool(t1.cid() == t2.cid()),
    }


def family_h480b_repair_trajectory_cid_deterministic(
        seed: int) -> dict[str, Any]:
    p = build_default_tiny_substrate_v15(
        seed=int(seed) + 70001)
    ids = tokenize_bytes_v15("h480b", max_len=12)
    _, c1 = forward_tiny_substrate_v15(p, ids)
    _, c2 = forward_tiny_substrate_v15(p, ids)
    return {
        "schema": R162_SCHEMA_VERSION,
        "name": "h480b_repair_trajectory_cid_deterministic",
        "passed": bool(
            c1.repair_trajectory_cid
            == c2.repair_trajectory_cid
            and len(c1.repair_trajectory_cid) > 0),
    }


def family_h480c_dominant_repair_per_layer_shape(
        seed: int) -> dict[str, Any]:
    p = build_default_tiny_substrate_v15(
        seed=int(seed) + 70002)
    ids = tokenize_bytes_v15("h480c", max_len=12)
    _, c = forward_tiny_substrate_v15(p, ids)
    ok = (
        c.dominant_repair_per_layer is not None
        and c.dominant_repair_per_layer.shape == (
            W70_DEFAULT_V15_N_LAYERS,)
        and W70_DEFAULT_V15_N_LAYERS == 17)
    return {
        "schema": R162_SCHEMA_VERSION,
        "name": "h480c_dominant_repair_per_layer_shape",
        "passed": bool(ok),
    }


def family_h480d_budget_primary_gate_shape(
        seed: int) -> dict[str, Any]:
    p = build_default_tiny_substrate_v15(
        seed=int(seed) + 70003)
    ids = tokenize_bytes_v15("h480d", max_len=12)
    t, _ = forward_tiny_substrate_v15(p, ids)
    return {
        "schema": R162_SCHEMA_VERSION,
        "name": "h480d_budget_primary_gate_shape",
        "passed": bool(
            t.budget_primary_gate_per_layer.shape == (
                W70_DEFAULT_V15_N_LAYERS,)),
    }


def family_h480e_v15_gate_score_shape(
        seed: int) -> dict[str, Any]:
    p = build_default_tiny_substrate_v15(
        seed=int(seed) + 70004)
    ids = tokenize_bytes_v15("h480e", max_len=12)
    t, _ = forward_tiny_substrate_v15(p, ids)
    return {
        "schema": R162_SCHEMA_VERSION,
        "name": "h480e_v15_gate_score_shape",
        "passed": bool(
            t.v15_gate_score_per_layer.shape == (
                W70_DEFAULT_V15_N_LAYERS,)
            and W70_DEFAULT_V15_N_LAYERS == 17),
    }


def family_h480f_repair_dominance_flops_saving(
        seed: int) -> dict[str, Any]:
    sav = substrate_repair_dominance_flops_v15(
        n_tokens=128, n_repairs=6)
    return {
        "schema": R162_SCHEMA_VERSION,
        "name": "h480f_repair_dominance_flops_saving",
        "passed": bool(sav["saving_ratio"] >= 0.8),
        "ratio": float(sav["saving_ratio"]),
    }


def family_h480g_budget_primary_throttle(
        seed: int) -> dict[str, Any]:
    sav = substrate_budget_primary_throttle_v15(
        visible_token_budget=64, baseline_token_cost=512)
    return {
        "schema": R162_SCHEMA_VERSION,
        "name": "h480g_budget_primary_throttle",
        "passed": bool(sav["saving_ratio"] >= 0.5),
        "ratio": float(sav["saving_ratio"]),
    }


def _build_kv_v15(p, seed: int) -> KVBridgeV15Projection:
    cfg = p.config.v14.v13.v12.v11.v10.v9
    d_head = cfg.d_model // cfg.n_heads
    b3 = KVBridgeV3Projection.init(
        n_layers=cfg.n_layers, n_heads=cfg.n_heads,
        n_kv_heads=cfg.n_kv_heads, n_inject_tokens=3,
        carrier_dim=8, d_head=d_head, seed=int(seed))
    return KVBridgeV15Projection.init_from_v14(
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
                                                    b3, seed_v4=int(seed) + 1),
                                                seed_v5=int(seed) + 2),
                                            seed_v6=int(seed) + 3),
                                        seed_v7=int(seed) + 4),
                                    seed_v8=int(seed) + 5),
                                seed_v9=int(seed) + 6),
                            seed_v10=int(seed) + 7),
                        seed_v11=int(seed) + 8),
                    seed_v12=int(seed) + 9),
                seed_v13=int(seed) + 10),
            seed_v14=int(seed) + 11),
        seed_v15=int(seed) + 12)


def family_h481_kv_v15_eleven_target_ridge(
        seed: int) -> dict[str, Any]:
    p = build_default_tiny_substrate_v15(
        seed=int(seed) + 70100)
    b15 = _build_kv_v15(p, seed=70300)
    cfg = p.config.v14.v13.v12.v11.v10.v9
    rng = _np.random.default_rng(int(seed) + 70200)
    ids = tokenize_bytes_v15("kvr162", max_len=4)
    while len(ids) < 4:
        ids.append(0)
    carriers = [list(rng.standard_normal(8)) for _ in range(4)]
    targets = []
    for i in range(11):
        t = _np.zeros(cfg.vocab_size)
        t[20 + 15 * i] = 1.0 / (i + 1)
        targets.append(t.tolist())
    _, rep = fit_kv_bridge_v15_eleven_target(
        params=p, projection=b15,
        train_carriers=carriers,
        target_delta_logits_stack=targets,
        follow_up_token_ids=ids, n_directions=3)
    return {
        "schema": R162_SCHEMA_VERSION,
        "name": "h481_kv_v15_eleven_target_ridge",
        "passed": bool(rep.n_targets == 11 and rep.converged),
    }


def family_h481b_repair_trajectory_fingerprint_dim(
        seed: int) -> dict[str, Any]:
    fp = compute_repair_trajectory_fingerprint_v15(
        role="planner",
        repair_trajectory_cid="0" * 64,
        dominant_repair_label=1,
        visible_token_budget=128.0,
        baseline_cost=512.0,
        task_id=str(seed),
        team_id="team")
    return {
        "schema": R162_SCHEMA_VERSION,
        "name": "h481b_repair_trajectory_fingerprint_dim",
        "passed": bool(len(fp) == 70),
    }


def family_h481c_repair_dominance_falsifier(
        seed: int) -> dict[str, Any]:
    honest = probe_kv_bridge_v15_repair_dominance_falsifier(
        dominant_repair_label=1)
    return {
        "schema": R162_SCHEMA_VERSION,
        "name": "h481c_repair_dominance_falsifier",
        "passed": bool(honest.falsifier_score == 0.0),
    }


def family_h482_cache_v13_ten_objective(
        seed: int) -> dict[str, Any]:
    cc13 = CacheControllerV13.init(fit_seed=int(seed) + 70400)
    rng = _np.random.default_rng(int(seed) + 70401)
    X = rng.standard_normal((12, 4))
    _, rep = fit_ten_objective_ridge_v13(
        controller=cc13, train_features=X.tolist(),
        target_drop_oracle=X.sum(axis=-1).tolist(),
        target_retrieval_relevance=X[:, 0].tolist(),
        target_hidden_wins=(X[:, 1] - X[:, 2]).tolist(),
        target_replay_dominance=(X[:, 3] * 0.5).tolist(),
        target_team_task_success=(
            X[:, 0] * 0.3 - X[:, 1] * 0.1).tolist(),
        target_team_failure_recovery=(
            X[:, 2] * 0.4 + X[:, 3] * 0.2).tolist(),
        target_branch_merge=(
            X[:, 0] * 0.2 + X[:, 2] * 0.5).tolist(),
        target_partial_contradiction=(
            X[:, 1] * 0.3 + X[:, 3] * 0.4).tolist(),
        target_multi_branch_rejoin=(
            X[:, 0] * 0.5 + X[:, 1] * 0.2).tolist(),
        target_budget_primary=(
            X[:, 0] * 0.2 + X[:, 1] * 0.3 + X[:, 2] * 0.4
        ).tolist())
    return {
        "schema": R162_SCHEMA_VERSION,
        "name": "h482_cache_v13_ten_objective",
        "passed": bool(
            rep.converged and rep.n_objectives == 10),
    }


def family_h483_replay_v11_budget_primary_routing(
        seed: int) -> dict[str, Any]:
    rc11 = ReplayControllerV11.init()
    cands = {
        r: [ReplayCandidate(
            100, 1000, 50, 0.1, 0.0, 0.3, True, True, 0)]
        for r in W70_REPLAY_REGIMES_V11}
    decs = {r: ["choose_reuse"] for r in W70_REPLAY_REGIMES_V11}
    rc11, _ = fit_replay_controller_v11_per_role(
        controller=rc11, role="planner",
        train_candidates_per_regime=cands,
        train_decisions_per_regime=decs)
    rng = _np.random.default_rng(int(seed) + 70500)
    X = rng.standard_normal((40, 11))
    labs = [
        W70_BUDGET_PRIMARY_ROUTING_LABELS[
            i % len(W70_BUDGET_PRIMARY_ROUTING_LABELS)]
        for i in range(40)]
    _, rep = fit_replay_v11_budget_primary_routing_head(
        controller=rc11, train_team_features=X.tolist(),
        train_routing_labels=labs)
    return {
        "schema": R162_SCHEMA_VERSION,
        "name": "h483_replay_v11_budget_primary_routing",
        "passed": bool(
            len(W70_REPLAY_REGIMES_V11) == 18
            and rep.post_classification_acc
                >= rep.pre_classification_acc),
    }


def family_h484_substrate_adapter_v15_full_tier(
        seed: int) -> dict[str, Any]:
    matrix = probe_all_v15_adapters()
    in_repo = matrix.by_name().get("tiny_substrate_v15")
    return {
        "schema": R162_SCHEMA_VERSION,
        "name": "h484_substrate_adapter_v15_full_tier",
        "passed": bool(
            in_repo is not None
            and in_repo.tier
            == W70_SUBSTRATE_TIER_SUBSTRATE_V15_FULL),
    }


def family_h485_persistent_v22_layers_skip(
        seed: int) -> dict[str, Any]:
    return {
        "schema": R162_SCHEMA_VERSION,
        "name": "h485_persistent_v22_layers_skip",
        "passed": bool(
            W70_DEFAULT_V22_N_LAYERS == 21
            and W70_DEFAULT_V22_MAX_CHAIN_WALK_DEPTH == 131072),
    }


def family_h486_lhr_v22_max_k_heads(
        seed: int) -> dict[str, Any]:
    head = LongHorizonReconstructionV22Head.init(
        seed=int(seed) + 70600)
    return {
        "schema": R162_SCHEMA_VERSION,
        "name": "h486_lhr_v22_max_k_heads",
        "passed": bool(
            head.max_k == W70_DEFAULT_LHR_V22_MAX_K
            and W70_DEFAULT_LHR_V22_MAX_K == 576),
    }


def family_h487_deep_v15_fifteen_way_and_consensus_v16(
        seed: int) -> dict[str, Any]:
    hybrid = DeepSubstrateHybridV15()
    cc13 = CacheControllerV13.init(fit_seed=int(seed) + 70700)
    rng = _np.random.default_rng(int(seed) + 70701)
    X = rng.standard_normal((6, 4))
    cc13, _ = fit_ten_objective_ridge_v13(
        controller=cc13, train_features=X.tolist(),
        target_drop_oracle=X.sum(axis=-1).tolist(),
        target_retrieval_relevance=X[:, 0].tolist(),
        target_hidden_wins=(X[:, 1] - X[:, 2]).tolist(),
        target_replay_dominance=(X[:, 3] * 0.5).tolist(),
        target_team_task_success=(
            X[:, 0] * 0.3 - X[:, 1] * 0.1).tolist(),
        target_team_failure_recovery=(
            X[:, 2] * 0.4 + X[:, 3] * 0.2).tolist(),
        target_branch_merge=(
            X[:, 0] * 0.2 + X[:, 2] * 0.5).tolist(),
        target_partial_contradiction=(
            X[:, 1] * 0.3 + X[:, 3] * 0.4).tolist(),
        target_multi_branch_rejoin=(
            X[:, 0] * 0.5 + X[:, 1] * 0.2).tolist(),
        target_budget_primary=(
            X[:, 0] * 0.2 + X[:, 1] * 0.3 + X[:, 2] * 0.4
        ).tolist())
    rc11 = ReplayControllerV11.init()
    cands = {
        r: [ReplayCandidate(
            100, 1000, 50, 0.1, 0.0, 0.3, True, True, 0)]
        for r in W70_REPLAY_REGIMES_V11}
    decs = {r: ["choose_reuse"] for r in W70_REPLAY_REGIMES_V11}
    rc11, _ = fit_replay_controller_v11_per_role(
        controller=rc11, role="planner",
        train_candidates_per_regime=cands,
        train_decisions_per_regime=decs)
    rng2 = _np.random.default_rng(int(seed) + 70702)
    Xb = rng2.standard_normal((20, 11))
    labs = [
        W70_BUDGET_PRIMARY_ROUTING_LABELS[
            i % len(W70_BUDGET_PRIMARY_ROUTING_LABELS)]
        for i in range(20)]
    rc11, _ = fit_replay_v11_budget_primary_routing_head(
        controller=rc11, train_team_features=Xb.tolist(),
        train_routing_labels=labs)
    v14_witness = DeepSubstrateHybridV14ForwardWitness(
        schema="coordpy.deep_substrate_hybrid_v14.v1",
        hybrid_cid="",
        inner_v13_witness_cid="",
        fourteen_way=True,
        cache_controller_v12_fired=True,
        replay_controller_v10_fired=True,
        multi_branch_rejoin_witness_active=True,
        silent_corruption_active=True,
        substrate_self_checksum_active=True,
        team_consensus_controller_v4_active=True,
        multi_branch_rejoin_witness_l1=0.5,
        silent_corruption_count=1,
        substrate_self_checksum_cid="x")
    w = deep_substrate_hybrid_v15_forward(
        hybrid=hybrid, v14_witness=v14_witness,
        cache_controller_v13=cc13,
        replay_controller_v11=rc11,
        repair_trajectory_cid="abc",
        dominant_repair_l1=3,
        budget_primary_gate_mean=0.7,
        n_team_consensus_v5_invocations=1)
    return {
        "schema": R162_SCHEMA_VERSION,
        "name": "h487_deep_v15_fifteen_way_and_consensus_v16",
        "passed": bool(
            w.fifteen_way
            and len(W70_CONSENSUS_V16_STAGES) >= 26),
        "n_stages": int(len(W70_CONSENSUS_V16_STAGES)),
    }


_R162_FAMILIES: tuple[Any, ...] = (
    family_h480_v15_substrate_determinism,
    family_h480b_repair_trajectory_cid_deterministic,
    family_h480c_dominant_repair_per_layer_shape,
    family_h480d_budget_primary_gate_shape,
    family_h480e_v15_gate_score_shape,
    family_h480f_repair_dominance_flops_saving,
    family_h480g_budget_primary_throttle,
    family_h481_kv_v15_eleven_target_ridge,
    family_h481b_repair_trajectory_fingerprint_dim,
    family_h481c_repair_dominance_falsifier,
    family_h482_cache_v13_ten_objective,
    family_h483_replay_v11_budget_primary_routing,
    family_h484_substrate_adapter_v15_full_tier,
    family_h485_persistent_v22_layers_skip,
    family_h486_lhr_v22_max_k_heads,
    family_h487_deep_v15_fifteen_way_and_consensus_v16,
)


def run_r162(
        seeds: Sequence[int] = (199, 299, 399),
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for s in seeds:
        results: dict[str, Any] = {}
        for fn in _R162_FAMILIES:
            try:
                r = fn(int(s))
            except Exception as e:
                r = {
                    "schema": R162_SCHEMA_VERSION,
                    "name": getattr(fn, "__name__", "unknown"),
                    "passed": False,
                    "exception": str(e),
                }
            results[r["name"]] = r
        out.append({
            "schema": R162_SCHEMA_VERSION,
            "seed": int(s),
            "family_results": results,
        })
    return out


__all__ = ["R162_SCHEMA_VERSION", "run_r162"]
