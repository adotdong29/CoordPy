"""W67 R-149 benchmark family — V12 substrate / twelve-way hybrid /
multi-agent task-success V3 / team-coordinator-V3 / team-consensus-
controller V2.

H285..H292 cell families (24 H-bars):

* H285   V12 substrate determinism
* H285b  V12 branch_merge_witness tensor shape (L, H, T)
* H285c  V12 role_dropout_recovery_flag semantics
* H285d  V12 v12_gate_score_per_layer shape (L,)
* H285e  V12 substrate snapshot-fork content-addressed
* H285f  V12 substrate branch-merge primitive
* H285g  KV bridge V12 eight-target ridge converges
* H286   KV bridge V12 branch_merge falsifier returns 0 under
         inversion
* H286b  KV bridge V12 role-pair fingerprint dim = 40
* H287   HSB V11 eight-target ridge converges
* H287b  HSB V11 hidden-vs-branch-merge probe in [0, 1]
* H288   prefix V11 K=128 drift curve has correct length
* H288b  prefix V11 six-way decision is valid
* H289   attention V11 seven-stage clamp keeps delta_l2 <= cap
* H289b  cache controller V10 seven-objective ridge converges
* H289c  cache controller V10 per-role 8-dim eviction head
         converges
* H290   replay controller V8 per-role per-regime head converges
         (12 regimes)
* H290b  replay controller V8 branch_merge_routing head converges
* H290c  replay controller V8 includes twelve regimes
* H291   substrate branch-merge flops saving >= 0.6
* H291b  MASC V3 V12 strictly beats V11 on >= 50 % of seeds
         (baseline regime)
* H291c  MASC V3 TSC_V12 strictly beats TSC_V11 on >= 50 % of seeds
         (baseline regime)
* H291d  substrate adapter V12 has matrix.has_v12_full=True
* H291e  team-consensus controller V2 fires branch_merge arbiter
         on conflicting payloads
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as _np

from coordpy.attention_steering_bridge_v2 import (
    AttentionSteeringV2Projection,
)
from coordpy.attention_steering_bridge_v11 import (
    steer_attention_and_measure_v11,
)
from coordpy.cache_controller_v10 import (
    CacheControllerV10,
    fit_seven_objective_ridge_v10,
    fit_per_role_eviction_head_v10,
)
from coordpy.hidden_state_bridge_v2 import (
    HiddenStateBridgeV2Projection,
)
from coordpy.hidden_state_bridge_v3 import (
    HiddenStateBridgeV3Projection,
)
from coordpy.hidden_state_bridge_v4 import (
    HiddenStateBridgeV4Projection,
)
from coordpy.hidden_state_bridge_v5 import (
    HiddenStateBridgeV5Projection,
)
from coordpy.hidden_state_bridge_v6 import (
    HiddenStateBridgeV6Projection,
)
from coordpy.hidden_state_bridge_v7 import (
    HiddenStateBridgeV7Projection,
)
from coordpy.hidden_state_bridge_v8 import (
    HiddenStateBridgeV8Projection,
)
from coordpy.hidden_state_bridge_v9 import (
    HiddenStateBridgeV9Projection,
)
from coordpy.hidden_state_bridge_v10 import (
    HiddenStateBridgeV10Projection,
)
from coordpy.hidden_state_bridge_v11 import (
    HiddenStateBridgeV11Projection,
    fit_hsb_v11_eight_target,
    probe_hsb_v11_hidden_vs_branch_merge,
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
from coordpy.kv_bridge_v12 import (
    KVBridgeV12Projection,
    compute_role_pair_fingerprint_v12,
    fit_kv_bridge_v12_eight_target,
    probe_kv_bridge_v12_branch_merge_falsifier,
)
from coordpy.multi_agent_substrate_coordinator_v2 import (
    W66_MASC_V2_REGIME_BASELINE,
)
from coordpy.multi_agent_substrate_coordinator_v3 import (
    MultiAgentSubstrateCoordinatorV3,
)
from coordpy.prefix_state_bridge_v11 import (
    W67_DEFAULT_PREFIX_V11_K_STEPS,
    compare_prefix_vs_hidden_vs_replay_vs_team_vs_recover_vs_branch_v11,
)
from coordpy.replay_controller import ReplayCandidate
from coordpy.replay_controller_v5 import ReplayControllerV5
from coordpy.replay_controller_v6 import ReplayControllerV6
from coordpy.replay_controller_v7 import ReplayControllerV7
from coordpy.replay_controller_v8 import (
    ReplayControllerV8,
    W67_BRANCH_MERGE_ROUTING_LABELS,
    W67_REPLAY_REGIMES_V8,
    fit_replay_controller_v8_per_role,
    fit_replay_v8_branch_merge_routing_head,
)
from coordpy.substrate_adapter_v12 import (
    W67_SUBSTRATE_TIER_SUBSTRATE_V12_FULL,
    probe_all_v12_adapters,
    probe_tiny_substrate_v12_adapter,
)
from coordpy.team_consensus_controller_v2 import (
    TeamConsensusControllerV2,
    W67_TC_V2_DECISION_BRANCH_MERGE_RECONCILIATION,
)
from coordpy.tiny_substrate_v12 import (
    build_default_tiny_substrate_v12,
    forward_tiny_substrate_v12,
    record_branch_merge_witness_v12,
    substrate_branch_merge_flops_v12,
    substrate_branch_merge_v12,
    substrate_snapshot_fork_v12,
    tokenize_bytes_v12,
    trigger_role_dropout_recovery_v12,
)


R149_SCHEMA_VERSION: str = "coordpy.r149_benchmark.v1"


def family_h285_v12_substrate_determinism(
        seed: int) -> dict[str, Any]:
    p = build_default_tiny_substrate_v12(seed=int(seed) + 49000)
    ids = tokenize_bytes_v12("h285", max_len=12)
    t1, _ = forward_tiny_substrate_v12(p, ids)
    t2, _ = forward_tiny_substrate_v12(p, ids)
    return {
        "schema": R149_SCHEMA_VERSION,
        "name": "h285_v12_substrate_determinism",
        "passed": bool(t1.cid() == t2.cid()),
    }


def family_h285b_v12_branch_merge_witness_shape(
        seed: int) -> dict[str, Any]:
    p = build_default_tiny_substrate_v12(seed=int(seed) + 49001)
    ids = tokenize_bytes_v12("h285b", max_len=12)
    _, c = forward_tiny_substrate_v12(p, ids)
    return {
        "schema": R149_SCHEMA_VERSION,
        "name": "h285b_v12_branch_merge_witness_shape",
        "passed": bool(c.branch_merge_witness.shape == (
            p.config.v11.v10.v9.n_layers,
            p.config.v11.v10.v9.n_heads,
            p.config.v11.v10.v9.max_len)),
        "shape": list(c.branch_merge_witness.shape),
    }


def family_h285c_v12_role_dropout_recovery_flag(
        seed: int) -> dict[str, Any]:
    p = build_default_tiny_substrate_v12(seed=int(seed) + 49002)
    ids = tokenize_bytes_v12("h285c", max_len=12)
    _, c = forward_tiny_substrate_v12(p, ids)
    trigger_role_dropout_recovery_v12(
        c, absent_role="r0", covering_role="r1", window=2)
    return {
        "schema": R149_SCHEMA_VERSION,
        "name": "h285c_v12_role_dropout_recovery_flag",
        "passed": bool(
            ("r0", "r1") in c.role_dropout_recovery_flag
            and c.role_dropout_recovery_flag[
                ("r0", "r1")]["triggered"]
            and c.role_dropout_recovery_flag[
                ("r0", "r1")]["window"] == 2),
    }


def family_h285d_v12_gate_score_shape(
        seed: int) -> dict[str, Any]:
    p = build_default_tiny_substrate_v12(seed=int(seed) + 49003)
    ids = tokenize_bytes_v12("h285d", max_len=12)
    t, _ = forward_tiny_substrate_v12(p, ids)
    return {
        "schema": R149_SCHEMA_VERSION,
        "name": "h285d_v12_gate_score_shape",
        "passed": bool(t.v12_gate_score_per_layer.shape == (
            p.config.v11.v10.v9.n_layers,)),
    }


def family_h285e_v12_substrate_snapshot_fork(
        seed: int) -> dict[str, Any]:
    p = build_default_tiny_substrate_v12(seed=int(seed) + 49004)
    ids = tokenize_bytes_v12("h285e", max_len=12)
    _, c = forward_tiny_substrate_v12(p, ids)
    fork = substrate_snapshot_fork_v12(
        c, branch_ids=["b0", "b1", "b2"],
        fork_label="r149_fork")
    fork2 = substrate_snapshot_fork_v12(
        c, branch_ids=["b0", "b1", "b2"],
        fork_label="r149_fork")
    return {
        "schema": R149_SCHEMA_VERSION,
        "name": "h285e_v12_substrate_snapshot_fork",
        "passed": bool(
            fork.cid() == fork2.cid()
            and len(fork.per_branch_cid) == 3),
    }


def family_h285f_v12_substrate_branch_merge(
        seed: int) -> dict[str, Any]:
    p = build_default_tiny_substrate_v12(seed=int(seed) + 49005)
    ids = tokenize_bytes_v12("h285f", max_len=12)
    _, c = forward_tiny_substrate_v12(p, ids)
    fork = substrate_snapshot_fork_v12(
        c, branch_ids=["b0", "b1"],
        fork_label="r149_merge_fork")
    merge = substrate_branch_merge_v12(
        c, fork=fork,
        branch_payloads={
            "b0": [1.0, 2.0, 3.0], "b1": [1.1, 2.1, 3.1]})
    return {
        "schema": R149_SCHEMA_VERSION,
        "name": "h285f_v12_substrate_branch_merge",
        "passed": bool(
            merge.n_branches_merged == 2
            and merge.reconciliation_delta_l1 > 0.0),
        "delta_l1": float(merge.reconciliation_delta_l1),
    }


def family_h285g_kv_bridge_v12_eight_target(
        seed: int) -> dict[str, Any]:
    p = build_default_tiny_substrate_v12(seed=int(seed) + 49006)
    cfg = p.config.v11.v10.v9
    d_head = cfg.d_model // cfg.n_heads
    proj = KVBridgeV12Projection.init_from_v11(
        KVBridgeV11Projection.init_from_v10(
            KVBridgeV10Projection.init_from_v9(
                KVBridgeV9Projection.init_from_v8(
                    KVBridgeV8Projection.init_from_v7(
                        KVBridgeV7Projection.init_from_v6(
                            KVBridgeV6Projection.init_from_v5(
                                KVBridgeV5Projection.init_from_v4(
                                    KVBridgeV4Projection.init_from_v3(
                                        KVBridgeV3Projection.init(
                                            n_layers=cfg.n_layers,
                                            n_heads=cfg.n_heads,
                                            n_kv_heads=cfg.n_kv_heads,
                                            n_inject_tokens=3,
                                            carrier_dim=8,
                                            d_head=d_head,
                                            seed=49500),
                                        seed_v4=49501),
                                    seed_v5=49502),
                                seed_v6=49503),
                            seed_v7=49504),
                        seed_v8=49505),
                    seed_v9=49506),
                seed_v10=49507),
            seed_v11=49508),
        seed_v12=49509)
    rng = _np.random.default_rng(int(seed))
    ids = tokenize_bytes_v12("kv12", max_len=4)
    while len(ids) < 4:
        ids.append(0)
    carriers = [list(rng.standard_normal(8)) for _ in range(4)]
    targets = []
    for i in range(8):
        t = _np.zeros(cfg.vocab_size)
        t[100 + 20 * i] = 1.0 / (i + 1)
        targets.append(t.tolist())
    _, report = fit_kv_bridge_v12_eight_target(
        params=p, projection=proj,
        train_carriers=carriers,
        target_delta_logits_stack=targets,
        follow_up_token_ids=ids, n_directions=3)
    return {
        "schema": R149_SCHEMA_VERSION,
        "name": "h285g_kv_bridge_v12_eight_target",
        "passed": bool(report.n_targets == 8 and (
            report.branch_merge_post
            <= report.branch_merge_pre + 1e-3)),
    }


def family_h286_kv_bridge_v12_branch_merge_falsifier(
        seed: int) -> dict[str, Any]:
    fals = probe_kv_bridge_v12_branch_merge_falsifier(
        branch_merge_flag=0.5)
    return {
        "schema": R149_SCHEMA_VERSION,
        "name": "h286_kv_bridge_v12_branch_merge_falsifier",
        "passed": bool(fals.falsifier_score == 0.0),
    }


def family_h286b_role_pair_fingerprint_dim(
        seed: int) -> dict[str, Any]:
    fp = compute_role_pair_fingerprint_v12(
        role_a="planner", role_b="analyst",
        task_id="t", team_id="x", branch_id="b1")
    return {
        "schema": R149_SCHEMA_VERSION,
        "name": "h286b_role_pair_fingerprint_dim",
        "passed": bool(len(fp) == 40),
    }


def family_h287_hsb_v11_eight_target(
        seed: int) -> dict[str, Any]:
    from coordpy.tiny_substrate_v11 import (
        build_default_tiny_substrate_v11,
    )
    p = build_default_tiny_substrate_v11(seed=int(seed) + 49007)
    cfg = p.config.v10.v9
    proj = HiddenStateBridgeV11Projection.init_from_v10(
        HiddenStateBridgeV10Projection.init_from_v9(
            HiddenStateBridgeV9Projection.init_from_v8(
                HiddenStateBridgeV8Projection.init_from_v7(
                    HiddenStateBridgeV7Projection.init_from_v6(
                        HiddenStateBridgeV6Projection.init_from_v5(
                            HiddenStateBridgeV5Projection.init_from_v4(
                                HiddenStateBridgeV4Projection.init_from_v3(
                                    HiddenStateBridgeV3Projection.init_from_v2(
                                        HiddenStateBridgeV2Projection.init(
                                            target_layers=(1, 3),
                                            n_tokens=6,
                                            carrier_dim=6,
                                            d_model=cfg.d_model,
                                            seed=49600),
                                        n_heads=cfg.n_heads,
                                        seed_v3=49601),
                                    seed_v4=49602),
                                n_positions=3, seed_v5=49603),
                            seed_v6=49604),
                        seed_v7=49605),
                    seed_v8=49606),
                seed_v9=49607),
            seed_v10=49608),
        seed_v11=49609)
    rng = _np.random.default_rng(int(seed))
    ids = list(range(1, 5))
    carriers = [list(rng.standard_normal(6)) for _ in range(4)]
    targets = []
    for i in range(8):
        t = _np.zeros(cfg.vocab_size)
        t[100 + 20 * i] = 1.0 / (i + 1)
        targets.append(t.tolist())
    _, report = fit_hsb_v11_eight_target(
        params=p.v3_params, projection=proj,
        train_carriers=carriers,
        target_delta_logits_stack=targets,
        follow_up_token_ids=ids, n_directions=3)
    return {
        "schema": R149_SCHEMA_VERSION,
        "name": "h287_hsb_v11_eight_target",
        "passed": bool(report.n_targets == 8 and (
            report.role_dropout_post
            <= report.role_dropout_pre + 1e-3)),
    }


def family_h287b_hsb_v11_hidden_vs_branch_merge(
        seed: int) -> dict[str, Any]:
    L, H = 4, 4
    h = _np.full((L, H), 0.1)
    b = _np.full((L, H), 0.5)
    res = probe_hsb_v11_hidden_vs_branch_merge(
        n_layers=L, n_heads=H,
        hidden_residual_l2_grid=h.tolist(),
        branch_merge_residual_l2_grid=b.tolist())
    return {
        "schema": R149_SCHEMA_VERSION,
        "name": "h287b_hsb_v11_hidden_vs_branch_merge",
        "passed": bool(
            0.0 <= res["win_rate_mean"] <= 1.0
            and res["win_rate_mean"] >= 0.99),
        "win_rate_mean": float(res["win_rate_mean"]),
    }


def family_h288_prefix_v11_k128(seed: int) -> dict[str, Any]:
    return {
        "schema": R149_SCHEMA_VERSION,
        "name": "h288_prefix_v11_k128",
        "passed": bool(W67_DEFAULT_PREFIX_V11_K_STEPS == 128),
    }


def family_h288b_prefix_v11_six_way_decision(
        seed: int) -> dict[str, Any]:
    res = compare_prefix_vs_hidden_vs_replay_vs_team_vs_recover_vs_branch_v11(
        prefix_drift_curve=[0.1, 0.2, 0.3],
        hidden_drift_curve=[0.1, 0.2, 0.3],
        replay_drift_curve=[0.1, 0.2, 0.3],
        team_drift_curve=[0.1, 0.2, 0.3],
        recover_drift_curve=[0.1, 0.2, 0.3],
        branch_drift_curve=[0.05, 0.05, 0.05])
    return {
        "schema": R149_SCHEMA_VERSION,
        "name": "h288b_prefix_v11_six_way_decision",
        "passed": bool(res.decision in (
            "prefix_wins", "hidden_wins", "replay_wins",
            "team_wins", "recover_wins", "branch_wins", "tie")),
    }


def family_h289_attn_v11_seven_stage(
        seed: int) -> dict[str, Any]:
    from coordpy.tiny_substrate_v3 import TinyV3SubstrateParams
    sp = TinyV3SubstrateParams.init()
    proj = AttentionSteeringV2Projection.init(
        n_layers=sp.config.n_layers,
        n_heads=sp.config.n_heads,
        n_query=6, n_key=6,
        carrier_dim=6, seed=int(seed) + 49710)
    ids = tokenize_bytes_v12("h289", max_len=6)
    w = steer_attention_and_measure_v11(
        params=sp, carrier=[0.1] * 6,
        projection=proj, token_ids=ids,
        hellinger_budget=0.5, attention_trust_ema=0.7,
        branch_merge_ema=0.6,
        branch_id="r149_branch", team_id="r149_team")
    return {
        "schema": R149_SCHEMA_VERSION,
        "name": "h289_attn_v11_seven_stage",
        "passed": bool(
            w.seven_stage_used
            and w.attention_delta_l2_after_seven_stage
            <= float(w.branch_merge_cap) + 1e-9),
        "delta": float(
            w.attention_delta_l2_after_seven_stage),
    }


def family_h289b_cache_v10_seven_objective(
        seed: int) -> dict[str, Any]:
    cc = CacheControllerV10.init()
    rng = _np.random.default_rng(int(seed) + 49800)
    X = rng.standard_normal((10, 4))
    cc2, report = fit_seven_objective_ridge_v10(
        controller=cc, train_features=X.tolist(),
        target_drop_oracle=X.sum(axis=-1).tolist(),
        target_retrieval_relevance=X[:, 0].tolist(),
        target_hidden_wins=(X[:, 1] - X[:, 2]).tolist(),
        target_replay_dominance=(X[:, 3] * 0.5).tolist(),
        target_team_task_success=(
            X[:, 0] * 0.3 - X[:, 1] * 0.1).tolist(),
        target_team_failure_recovery=(
            X[:, 2] * 0.4 + X[:, 3] * 0.2).tolist(),
        target_branch_merge=(
            X[:, 0] * 0.2 + X[:, 2] * 0.5).tolist())
    return {
        "schema": R149_SCHEMA_VERSION,
        "name": "h289b_cache_v10_seven_objective",
        "passed": bool(
            report.converged and report.n_objectives == 7),
    }


def family_h289c_cache_v10_per_role_eviction(
        seed: int) -> dict[str, Any]:
    cc = CacheControllerV10.init()
    rng = _np.random.default_rng(int(seed) + 49810)
    X = rng.standard_normal((8, 8))
    y = X[:, 0] * 0.4 + X[:, 7] * 0.3
    _, report = fit_per_role_eviction_head_v10(
        controller=cc, role="planner",
        train_features=X.tolist(),
        target_eviction_priorities=y.tolist())
    return {
        "schema": R149_SCHEMA_VERSION,
        "name": "h289c_cache_v10_per_role_eviction",
        "passed": bool(report.converged),
    }


def family_h290_replay_v8_per_role_per_regime(
        seed: int) -> dict[str, Any]:
    rcv5 = ReplayControllerV5.init()
    rcv6 = ReplayControllerV6.init(inner_v5=rcv5)
    rcv7 = ReplayControllerV7.init(inner_v6=rcv6)
    ctrl = ReplayControllerV8.init(inner_v7=rcv7)
    cands = {
        r: [ReplayCandidate(
            100, 1000, 50, 0.1, 0.0, 0.3, True, True, 0)]
        for r in W67_REPLAY_REGIMES_V8}
    decs = {
        r: ["choose_reuse"] for r in W67_REPLAY_REGIMES_V8}
    _, report = fit_replay_controller_v8_per_role(
        controller=ctrl, role="planner",
        train_candidates_per_regime=cands,
        train_decisions_per_regime=decs)
    return {
        "schema": R149_SCHEMA_VERSION,
        "name": "h290_replay_v8_per_role_per_regime",
        "passed": bool(
            report.n_role_regime_heads_v8 == 12
            and report.converged),
        "n_heads": int(report.n_role_regime_heads_v8),
    }


def family_h290b_replay_v8_branch_merge_routing(
        seed: int) -> dict[str, Any]:
    rcv5 = ReplayControllerV5.init()
    rcv6 = ReplayControllerV6.init(inner_v5=rcv5)
    rcv7 = ReplayControllerV7.init(inner_v6=rcv6)
    ctrl = ReplayControllerV8.init(inner_v7=rcv7)
    rng = _np.random.default_rng(int(seed) + 49820)
    X = rng.standard_normal((30, 10))
    decs: list[str] = []
    for i in range(30):
        if X[i, 0] > 0.5:
            decs.append(W67_BRANCH_MERGE_ROUTING_LABELS[0])
        elif X[i, 1] > 0.0:
            decs.append(W67_BRANCH_MERGE_ROUTING_LABELS[1])
        elif X[i, 2] > 0.0:
            decs.append(W67_BRANCH_MERGE_ROUTING_LABELS[2])
        else:
            decs.append(W67_BRANCH_MERGE_ROUTING_LABELS[3])
    _, report = fit_replay_v8_branch_merge_routing_head(
        controller=ctrl, train_team_features=X.tolist(),
        train_routing_labels=decs)
    return {
        "schema": R149_SCHEMA_VERSION,
        "name": "h290b_replay_v8_branch_merge_routing",
        "passed": bool(
            report.branch_merge_routing_trained
            and report.converged),
        "residual": float(
            report.branch_merge_routing_train_residual),
    }


def family_h290c_replay_v8_twelve_regimes(
        seed: int) -> dict[str, Any]:
    return {
        "schema": R149_SCHEMA_VERSION,
        "name": "h290c_replay_v8_twelve_regimes",
        "passed": bool(len(W67_REPLAY_REGIMES_V8) == 12),
        "n_regimes": int(len(W67_REPLAY_REGIMES_V8)),
    }


def family_h291_substrate_branch_merge_flops_saving(
        seed: int) -> dict[str, Any]:
    saving = substrate_branch_merge_flops_v12(
        n_tokens=128, n_branches=4)
    return {
        "schema": R149_SCHEMA_VERSION,
        "name": "h291_substrate_branch_merge_flops_saving",
        "passed": bool(saving["saving_ratio"] >= 0.6),
        "ratio": float(saving["saving_ratio"]),
    }


def family_h291b_masc_v3_v12_beats_v11(
        seed: int) -> dict[str, Any]:
    masc = MultiAgentSubstrateCoordinatorV3()
    _, agg = masc.run_batch(
        seeds=list(range(15)),
        regime=W66_MASC_V2_REGIME_BASELINE)
    return {
        "schema": R149_SCHEMA_VERSION,
        "name": "h291b_masc_v3_v12_beats_v11",
        "passed": bool(agg.v12_beats_v11_rate >= 0.5),
        "rate": float(agg.v12_beats_v11_rate),
    }


def family_h291c_masc_v3_tsc_v12_beats_tsc_v11(
        seed: int) -> dict[str, Any]:
    masc = MultiAgentSubstrateCoordinatorV3()
    _, agg = masc.run_batch(
        seeds=list(range(15)),
        regime=W66_MASC_V2_REGIME_BASELINE)
    return {
        "schema": R149_SCHEMA_VERSION,
        "name": "h291c_masc_v3_tsc_v12_beats_tsc_v11",
        "passed": bool(agg.tsc_v12_beats_tsc_v11_rate >= 0.5),
        "rate": float(agg.tsc_v12_beats_tsc_v11_rate),
    }


def family_h291d_substrate_adapter_v12_full_tier(
        seed: int) -> dict[str, Any]:
    cap = probe_tiny_substrate_v12_adapter()
    matrix = probe_all_v12_adapters()
    return {
        "schema": R149_SCHEMA_VERSION,
        "name": "h291d_substrate_adapter_v12_full_tier",
        "passed": bool(
            cap.tier == W67_SUBSTRATE_TIER_SUBSTRATE_V12_FULL
            and matrix.has_v12_full()),
        "tier": str(cap.tier),
    }


def family_h291e_team_consensus_v2_branch_merge(
        seed: int) -> dict[str, Any]:
    tc2 = TeamConsensusControllerV2()
    res = tc2.decide_v2(
        regime="branch_merge_reconciliation",
        agent_guesses=[0.5, 0.6, 0.55],
        agent_confidences=[0.7, 0.8, 0.65],
        substrate_replay_trust=0.7,
        branch_payloads_per_branch={
            "b0": [0.5, 0.6], "b1": [0.45, 0.55]},
        branch_trusts={"b0": 0.6, "b1": 0.4})
    return {
        "schema": R149_SCHEMA_VERSION,
        "name": "h291e_team_consensus_v2_branch_merge",
        "passed": bool(
            res["decision"]
            == W67_TC_V2_DECISION_BRANCH_MERGE_RECONCILIATION
            and res.get("payload") is not None),
    }


_R149_FAMILIES: tuple[Any, ...] = (
    family_h285_v12_substrate_determinism,
    family_h285b_v12_branch_merge_witness_shape,
    family_h285c_v12_role_dropout_recovery_flag,
    family_h285d_v12_gate_score_shape,
    family_h285e_v12_substrate_snapshot_fork,
    family_h285f_v12_substrate_branch_merge,
    family_h285g_kv_bridge_v12_eight_target,
    family_h286_kv_bridge_v12_branch_merge_falsifier,
    family_h286b_role_pair_fingerprint_dim,
    family_h287_hsb_v11_eight_target,
    family_h287b_hsb_v11_hidden_vs_branch_merge,
    family_h288_prefix_v11_k128,
    family_h288b_prefix_v11_six_way_decision,
    family_h289_attn_v11_seven_stage,
    family_h289b_cache_v10_seven_objective,
    family_h289c_cache_v10_per_role_eviction,
    family_h290_replay_v8_per_role_per_regime,
    family_h290b_replay_v8_branch_merge_routing,
    family_h290c_replay_v8_twelve_regimes,
    family_h291_substrate_branch_merge_flops_saving,
    family_h291b_masc_v3_v12_beats_v11,
    family_h291c_masc_v3_tsc_v12_beats_tsc_v11,
    family_h291d_substrate_adapter_v12_full_tier,
    family_h291e_team_consensus_v2_branch_merge,
)


def run_r149(
        seeds: Sequence[int] = (199, 299, 399),
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for s in seeds:
        results: dict[str, Any] = {}
        for fn in _R149_FAMILIES:
            try:
                r = fn(int(s))
            except Exception as e:
                r = {
                    "schema": R149_SCHEMA_VERSION,
                    "name": getattr(fn, "__name__", "unknown"),
                    "passed": False,
                    "exception": str(e),
                }
            results[r["name"]] = r
        out.append({
            "schema": R149_SCHEMA_VERSION,
            "seed": int(s),
            "family_results": results,
        })
    return out


__all__ = ["R149_SCHEMA_VERSION", "run_r149"]
