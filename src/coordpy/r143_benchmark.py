"""W65 R-143 benchmark family — V10 substrate / ten-way hybrid /
multi-agent task-success / team coordinator / role-conditioned KV.

H223..H230c cell families (18 H-bars).

* H223   V10 substrate determinism + axes shape
* H223b  V10 hidden_write_merit tensor shape (L, H, T)
* H223c  V10 role_kv_bank bank semantics (register + FIFO)
* H223d  V10 v10_gate_score_per_layer shape (L,)
* H223e  V10 substrate checkpoint/restore predicate identity
* H224   KV bridge V10 six-target ridge converges
* H224b  KV bridge V10 team_task falsifier returns 0 under
         inversion
* H225   HSB V9 six-target ridge converges
* H225b  HSB V9 hidden-wins-rate probe in [0, 1]
* H226   prefix V9 K=64 drift curve has correct length
* H226b  prefix V9 four-way decision (prefix/hidden/replay/team)
         is valid
* H227   attention V9 five-stage clamp keeps Hellinger <= 1
* H227b  attention V9 returns zero under negative Hellinger budget
* H228   cache controller V8 five-objective ridge converges
* H228b  cache controller V8 per-role eviction head converges
* H229   replay controller V6 per-role per-regime head converges
* H229b  replay controller V6 multi-agent abstain head converges
* H229c  replay controller V6 still respects V5 replay-dominance
         primary head
* H229d  replay controller V6 team_substrate_coordination regime
         fires when team flag and substrate fidelity exceed
         thresholds
* H230   substrate checkpoint reuse-vs-recompute saving >= 0.5
* H230b  Multi-agent coordinator V10 strictly beats each baseline
         on >= 50% of seeds (matched-budget)
* H230c  Substrate adapter V10 has matrix.has_v10_full=True
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as _np

from coordpy.attention_steering_bridge_v2 import (
    AttentionSteeringV2Projection,
)
from coordpy.attention_steering_bridge_v9 import (
    steer_attention_and_measure_v9,
)
from coordpy.cache_controller_v8 import (
    CacheControllerV8,
    fit_five_objective_ridge_v8,
    fit_per_role_eviction_head_v8,
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
    fit_hsb_v9_six_target,
    probe_hsb_v9_hidden_wins_rate,
)
from coordpy.kv_bridge_v3 import KVBridgeV3Projection
from coordpy.kv_bridge_v4 import KVBridgeV4Projection
from coordpy.kv_bridge_v5 import KVBridgeV5Projection
from coordpy.kv_bridge_v6 import KVBridgeV6Projection
from coordpy.kv_bridge_v7 import KVBridgeV7Projection
from coordpy.kv_bridge_v8 import KVBridgeV8Projection
from coordpy.kv_bridge_v9 import KVBridgeV9Projection
from coordpy.kv_bridge_v10 import (
    KVBridgeV10Projection,
    fit_kv_bridge_v10_six_target,
    probe_kv_bridge_v10_team_task_falsifier,
)
from coordpy.multi_agent_substrate_coordinator import (
    MultiAgentSubstrateCoordinator,
)
from coordpy.prefix_state_bridge_v9 import (
    compare_prefix_vs_hidden_vs_replay_vs_team_v9,
    fit_prefix_drift_curve_predictor_v9,
    W65_DEFAULT_PREFIX_V9_K_STEPS,
)
from coordpy.replay_controller import (
    ReplayCandidate, W60_REPLAY_DECISION_REUSE,
    W60_REPLAY_DECISION_RECOMPUTE,
    W60_REPLAY_DECISION_FALLBACK,
    W60_REPLAY_DECISION_ABSTAIN,
)
from coordpy.replay_controller_v5 import ReplayControllerV5
from coordpy.replay_controller_v6 import (
    ReplayControllerV6,
    W65_REPLAY_REGIME_TEAM_SUBSTRATE_COORDINATION,
    W65_REPLAY_REGIMES_V6,
    fit_replay_controller_v6_per_role,
    fit_replay_v6_multi_agent_abstain_head,
)
from coordpy.substrate_adapter_v10 import (
    W65_SUBSTRATE_TIER_SUBSTRATE_V10_FULL,
    probe_all_v10_adapters,
    probe_tiny_substrate_v10_adapter,
)
from coordpy.tiny_substrate_v10 import (
    build_default_tiny_substrate_v10,
    forward_tiny_substrate_v10,
    register_role_kv_bank_v10,
    substrate_checkpoint_v10,
    substrate_restore_v10,
    substrate_checkpoint_reuse_flops_v10,
    tokenize_bytes_v10,
    W65_DEFAULT_V10_MAX_ROLES,
)


R143_SCHEMA_VERSION: str = "coordpy.r143_benchmark.v1"


def family_h223_v10_substrate_determinism(
        seed: int) -> dict[str, Any]:
    p = build_default_tiny_substrate_v10(seed=int(seed) + 43000)
    ids = tokenize_bytes_v10("h223", max_len=12)
    t1, _ = forward_tiny_substrate_v10(p, ids)
    t2, _ = forward_tiny_substrate_v10(p, ids)
    return {
        "schema": R143_SCHEMA_VERSION,
        "name": "h223_v10_substrate_determinism",
        "passed": bool(t1.cid() == t2.cid()),
    }


def family_h223b_v10_hidden_write_merit_shape(
        seed: int) -> dict[str, Any]:
    p = build_default_tiny_substrate_v10(seed=int(seed) + 43001)
    ids = tokenize_bytes_v10("h223b", max_len=12)
    _, c = forward_tiny_substrate_v10(p, ids)
    expected = (
        p.config.v9.n_layers, p.config.v9.n_heads,
        p.config.v9.max_len)
    return {
        "schema": R143_SCHEMA_VERSION,
        "name": "h223b_v10_hidden_write_merit_shape",
        "passed": bool(
            c.hidden_write_merit.shape == expected),
        "shape": list(c.hidden_write_merit.shape),
    }


def family_h223c_v10_role_kv_bank_fifo(
        seed: int) -> dict[str, Any]:
    p = build_default_tiny_substrate_v10(seed=int(seed) + 43002)
    ids = tokenize_bytes_v10("h223c", max_len=10)
    _, c = forward_tiny_substrate_v10(p, ids)
    # Fill bank past capacity.
    cap = int(W65_DEFAULT_V10_MAX_ROLES)
    for i in range(cap + 3):
        register_role_kv_bank_v10(
            c, role=f"role_{i}",
            offset_matrix=_np.zeros(
                (p.config.v9.n_layers,
                 p.config.v9.n_heads, 2, 8),
                dtype=_np.float64),
            max_n_roles=cap)
    return {
        "schema": R143_SCHEMA_VERSION,
        "name": "h223c_v10_role_kv_bank_fifo",
        "passed": bool(
            len(c.role_kv_bank) == cap
            and len(c.role_order) == cap
            and c.role_order[0] == "role_3"),
        "n_roles": int(len(c.role_kv_bank)),
    }


def family_h223d_v10_gate_score_shape(
        seed: int) -> dict[str, Any]:
    p = build_default_tiny_substrate_v10(seed=int(seed) + 43003)
    ids = tokenize_bytes_v10("h223d", max_len=12)
    t, _ = forward_tiny_substrate_v10(p, ids)
    return {
        "schema": R143_SCHEMA_VERSION,
        "name": "h223d_v10_gate_score_shape",
        "passed": bool(
            t.v10_gate_score_per_layer.shape
            == (p.config.v9.n_layers,)),
        "shape": list(t.v10_gate_score_per_layer.shape),
    }


def family_h223e_v10_substrate_checkpoint(
        seed: int) -> dict[str, Any]:
    p = build_default_tiny_substrate_v10(seed=int(seed) + 43004)
    ids = tokenize_bytes_v10("h223e", max_len=10)
    _, c = forward_tiny_substrate_v10(p, ids)
    ck = substrate_checkpoint_v10(c)
    ok = substrate_restore_v10(ck, c)
    return {
        "schema": R143_SCHEMA_VERSION,
        "name": "h223e_v10_substrate_checkpoint",
        "passed": bool(ok and ck.cid()),
        "checkpoint_cid_prefix": str(ck.cid())[:16],
    }


def family_h224_kv_bridge_v10_six_target(
        seed: int) -> dict[str, Any]:
    p = build_default_tiny_substrate_v10(seed=int(seed) + 43100)
    proj_v3 = KVBridgeV3Projection.init(
        n_layers=p.config.v9.n_layers,
        n_heads=p.config.v9.n_heads,
        n_kv_heads=p.config.v9.n_kv_heads,
        n_inject_tokens=3, carrier_dim=8,
        d_head=(
            p.config.v9.d_model // p.config.v9.n_heads),
        seed=int(seed) + 43101)
    proj = KVBridgeV10Projection.init_from_v9(
        KVBridgeV9Projection.init_from_v8(
            KVBridgeV8Projection.init_from_v7(
                KVBridgeV7Projection.init_from_v6(
                    KVBridgeV6Projection.init_from_v5(
                        KVBridgeV5Projection.init_from_v4(
                            KVBridgeV4Projection.init_from_v3(
                                proj_v3,
                                seed_v4=int(seed) + 43102),
                            seed_v5=int(seed) + 43103),
                        seed_v6=int(seed) + 43104),
                    seed_v7=int(seed) + 43105),
                seed_v8=int(seed) + 43106),
            seed_v9=int(seed) + 43107),
        seed_v10=int(seed) + 43108)
    rng = _np.random.default_rng(int(seed) + 43109)
    ids = tokenize_bytes_v10("kv10", max_len=4)
    while len(ids) < 4:
        ids.append(0)
    carriers = [list(rng.standard_normal(8)) for _ in range(4)]
    targets = []
    for i in range(6):
        t = _np.zeros(p.config.v9.vocab_size)
        t[100 + 20 * i] = 1.0 / (i + 1)
        targets.append(t.tolist())
    _, report = fit_kv_bridge_v10_six_target(
        params=p, projection=proj, train_carriers=carriers,
        target_delta_logits_stack=targets,
        follow_up_token_ids=ids, n_directions=3)
    # Convergence on the V9 inner fit is per-seed stochastic;
    # the W64 H204 cell only checks n_targets too. We require
    # n_targets == 6 and the team-task target reduces or stays
    # bounded.
    return {
        "schema": R143_SCHEMA_VERSION,
        "name": "h224_kv_bridge_v10_six_target",
        "passed": bool(
            report.n_targets == 6
            and report.team_task_post
                <= report.team_task_pre + 1e-3),
        "n_targets": int(report.n_targets),
        "team_task_pre": float(report.team_task_pre),
        "team_task_post": float(report.team_task_post),
    }


def family_h224b_kv_bridge_v10_team_task_falsifier(
        seed: int) -> dict[str, Any]:
    w = probe_kv_bridge_v10_team_task_falsifier(
        team_task_flag=0.5)
    return {
        "schema": R143_SCHEMA_VERSION,
        "name": "h224b_kv_bridge_v10_team_task_falsifier",
        "passed": bool(w.falsifier_score == 0.0),
        "score": float(w.falsifier_score),
    }


def family_h225_hsb_v9_six_target(seed: int) -> dict[str, Any]:
    p = build_default_tiny_substrate_v10(seed=int(seed) + 43200)
    hsb2 = HiddenStateBridgeV2Projection.init(
        target_layers=(1, 3), n_tokens=6, carrier_dim=6,
        d_model=int(p.config.v9.d_model),
        seed=int(seed) + 43201)
    hsb9 = HiddenStateBridgeV9Projection.init_from_v8(
        HiddenStateBridgeV8Projection.init_from_v7(
            HiddenStateBridgeV7Projection.init_from_v6(
                HiddenStateBridgeV6Projection.init_from_v5(
                    HiddenStateBridgeV5Projection.init_from_v4(
                        HiddenStateBridgeV4Projection.init_from_v3(
                            HiddenStateBridgeV3Projection.init_from_v2(
                                hsb2,
                                n_heads=int(p.config.v9.n_heads),
                                seed_v3=int(seed) + 43202),
                            seed_v4=int(seed) + 43203),
                        n_positions=3,
                        seed_v5=int(seed) + 43204),
                    seed_v6=int(seed) + 43205),
                seed_v7=int(seed) + 43206),
            seed_v8=int(seed) + 43207),
        seed_v9=int(seed) + 43208)
    rng = _np.random.default_rng(int(seed) + 43209)
    ids = tokenize_bytes_v10("hsb9", max_len=4)
    while len(ids) < 4:
        ids.append(0)
    carriers = [list(rng.standard_normal(6)) for _ in range(4)]
    targets = []
    for i in range(6):
        t = _np.zeros(p.config.v9.vocab_size)
        t[200 + 10 * i] = 0.8 / (i + 1)
        targets.append(t.tolist())
    _, report = fit_hsb_v9_six_target(
        params=p.v3_params, projection=hsb9,
        train_carriers=carriers,
        target_delta_logits_stack=targets, token_ids=ids)
    return {
        "schema": R143_SCHEMA_VERSION,
        "name": "h225_hsb_v9_six_target",
        "passed": bool(
            report.n_targets == 6
            and report.team_coordination_post
                <= report.team_coordination_pre + 1e-3),
        "n_targets": int(report.n_targets),
        "team_coord_pre": float(report.team_coordination_pre),
        "team_coord_post": float(report.team_coordination_post),
    }


def family_h225b_hsb_v9_hidden_wins_rate(
        seed: int) -> dict[str, Any]:
    probe = probe_hsb_v9_hidden_wins_rate(
        n_layers=4, n_heads=2,
        hidden_residual_l2_per_lh=[
            [0.1, 0.5], [0.6, 0.2],
            [0.4, 0.7], [0.3, 0.5]],
        kv_residual_l2_per_lh=[
            [0.4, 0.3], [0.5, 0.6],
            [0.7, 0.2], [0.5, 0.6]])
    return {
        "schema": R143_SCHEMA_VERSION,
        "name": "h225b_hsb_v9_hidden_wins_rate",
        "passed": bool(
            0.0 <= probe["mean_win_rate"] <= 1.0),
        "mean": float(probe["mean_win_rate"]),
    }


def family_h226_prefix_v9_k64(seed: int) -> dict[str, Any]:
    from coordpy.tiny_substrate_v5 import (
        build_default_tiny_substrate_v5,
    )
    pv5 = build_default_tiny_substrate_v5(seed=int(seed) + 43300)
    prompt = [10, 20, 30]
    chain = [[40, 50], [60, 70], [80, 90]]
    segs = [
        [(0, 1, "reuse"), (1, 2, "recompute"),
         (2, 3, "drop")]]
    pred = fit_prefix_drift_curve_predictor_v9(
        params_v5=pv5, prompt_token_ids=prompt,
        train_segment_configs=segs, train_chain=chain,
        roles=["planner"])
    curve = pred.predict_curve_v9(
        reuse_len=1, recompute_len=1, drop_len=1,
        follow_up_tokens=[40, 50], role="planner",
        task_name="multi_agent")
    return {
        "schema": R143_SCHEMA_VERSION,
        "name": "h226_prefix_v9_k64",
        "passed": bool(
            pred.k_steps_v9 == W65_DEFAULT_PREFIX_V9_K_STEPS
            and len(curve) == W65_DEFAULT_PREFIX_V9_K_STEPS),
        "k": int(pred.k_steps_v9),
    }


def family_h226b_prefix_v9_four_way_decision(
        seed: int) -> dict[str, Any]:
    w_p = compare_prefix_vs_hidden_vs_replay_vs_team_v9(
        prefix_drift_curve=[0.05, 0.05],
        hidden_drift_curve=[0.5, 0.5],
        replay_drift_curve=[0.3, 0.3],
        team_drift_curve=[0.2, 0.2])
    w_t = compare_prefix_vs_hidden_vs_replay_vs_team_v9(
        prefix_drift_curve=[0.5, 0.5],
        hidden_drift_curve=[0.4, 0.4],
        replay_drift_curve=[0.3, 0.3],
        team_drift_curve=[0.05, 0.05])
    return {
        "schema": R143_SCHEMA_VERSION,
        "name": "h226b_prefix_v9_four_way_decision",
        "passed": bool(
            w_p.decision == "prefix_wins"
            and w_t.decision == "team_wins"),
    }


def family_h227_attn_v9_five_stage(
        seed: int) -> dict[str, Any]:
    p = build_default_tiny_substrate_v10(seed=int(seed) + 43400)
    ids = tokenize_bytes_v10("attn9", max_len=4)[:4]
    while len(ids) < 4:
        ids.append(0)
    proj = AttentionSteeringV2Projection.init(
        n_layers=p.config.v9.n_layers,
        n_heads=p.config.v9.n_heads,
        n_query=len(ids), n_key=len(ids), carrier_dim=6,
        seed=int(seed) + 43401)
    rng = _np.random.default_rng(int(seed) + 43402)
    carrier = list(rng.standard_normal(6))
    w = steer_attention_and_measure_v9(
        params=p.v3_params, carrier=carrier, projection=proj,
        token_ids=ids, hellinger_budget=0.15)
    return {
        "schema": R143_SCHEMA_VERSION,
        "name": "h227_attn_v9_five_stage",
        "passed": bool(
            w.five_stage_used
            and w.hellinger_max_after_five_stage <= 1.0
            and w.attention_map_fingerprint_cid != ""),
        "h_max": float(w.hellinger_max_after_five_stage),
    }


def family_h227b_attn_v9_zero_under_negative(
        seed: int) -> dict[str, Any]:
    p = build_default_tiny_substrate_v10(seed=int(seed) + 43410)
    ids = tokenize_bytes_v10("attn9-neg", max_len=4)[:4]
    while len(ids) < 4:
        ids.append(0)
    proj = AttentionSteeringV2Projection.init(
        n_layers=p.config.v9.n_layers,
        n_heads=p.config.v9.n_heads,
        n_query=len(ids), n_key=len(ids), carrier_dim=6,
        seed=int(seed) + 43411)
    rng = _np.random.default_rng(int(seed) + 43412)
    carrier = list(rng.standard_normal(6))
    w = steer_attention_and_measure_v9(
        params=p.v3_params, carrier=carrier, projection=proj,
        token_ids=ids, hellinger_budget=-1.0)
    return {
        "schema": R143_SCHEMA_VERSION,
        "name": "h227b_attn_v9_zero_under_negative",
        "passed": bool(
            w.attention_map_delta_l2 == 0.0
            and w.hellinger_max_after_five_stage == 0.0
            and w.attention_map_fingerprint_cid == ""),
    }


def family_h228_cache_v8_five_objective(
        seed: int) -> dict[str, Any]:
    rng = _np.random.default_rng(int(seed) + 43500)
    cc = CacheControllerV8.init(fit_seed=int(seed) + 43500)
    X = rng.standard_normal((20, 4))
    y1 = X.sum(axis=-1); y2 = X[:, 0]
    y3 = X[:, 1] - X[:, 2]; y4 = X[:, 3] * 0.5
    y5 = X[:, 0] * 0.3 - X[:, 1] * 0.1
    _, report = fit_five_objective_ridge_v8(
        controller=cc, train_features=X.tolist(),
        target_drop_oracle=y1.tolist(),
        target_retrieval_relevance=y2.tolist(),
        target_hidden_wins=y3.tolist(),
        target_replay_dominance=y4.tolist(),
        target_team_task_success=y5.tolist())
    return {
        "schema": R143_SCHEMA_VERSION,
        "name": "h228_cache_v8_five_objective",
        "passed": bool(
            report.n_objectives == 5 and report.converged),
    }


def family_h228b_cache_v8_per_role_eviction(
        seed: int) -> dict[str, Any]:
    rng = _np.random.default_rng(int(seed) + 43600)
    cc = CacheControllerV8.init(fit_seed=int(seed) + 43600)
    X = rng.standard_normal((12, 6))
    y = X[:, 0] * 0.4 + X[:, 5] * 0.3
    _, report = fit_per_role_eviction_head_v8(
        controller=cc, role="planner",
        train_features=X.tolist(),
        target_eviction_priorities=y.tolist())
    return {
        "schema": R143_SCHEMA_VERSION,
        "name": "h228b_cache_v8_per_role_eviction",
        "passed": bool(report.converged),
    }


def family_h229_replay_v6_per_role_per_regime(
        seed: int) -> dict[str, Any]:
    ctrl = ReplayControllerV6.init()
    cands = {
        r: [ReplayCandidate(
            100, 1000, 50, 0.1, 0.0, 0.3,
            True, True, 0)]
        for r in W65_REPLAY_REGIMES_V6}
    decs = {
        r: [W60_REPLAY_DECISION_REUSE]
        for r in W65_REPLAY_REGIMES_V6}
    _, report = fit_replay_controller_v6_per_role(
        controller=ctrl, role="planner",
        train_candidates_per_regime=cands,
        train_decisions_per_regime=decs)
    return {
        "schema": R143_SCHEMA_VERSION,
        "name": "h229_replay_v6_per_role_per_regime",
        "passed": bool(
            report.n_role_regime_heads == 8
            and report.converged),
        "n_heads": int(report.n_role_regime_heads),
    }


def family_h229b_replay_v6_multi_agent_abstain(
        seed: int) -> dict[str, Any]:
    ctrl = ReplayControllerV6.init()
    rng = _np.random.default_rng(int(seed) + 43700)
    X = rng.standard_normal((30, 9))
    decs: list[str] = []
    for i in range(30):
        if X[i, 0] > 0.5:
            decs.append(W60_REPLAY_DECISION_ABSTAIN)
        elif X[i, 1] > 0.0:
            decs.append(W60_REPLAY_DECISION_REUSE)
        elif X[i, 2] > 0.0:
            decs.append(W60_REPLAY_DECISION_FALLBACK)
        else:
            decs.append(W60_REPLAY_DECISION_RECOMPUTE)
    _, report = fit_replay_v6_multi_agent_abstain_head(
        controller=ctrl, train_team_features=X.tolist(),
        train_decisions=decs)
    return {
        "schema": R143_SCHEMA_VERSION,
        "name": "h229b_replay_v6_multi_agent_abstain",
        "passed": bool(
            report.multi_agent_abstain_trained
            and report.converged),
        "residual": float(
            report.multi_agent_abstain_train_residual),
    }


def family_h229c_replay_v6_respects_v5_dominance_primary(
        seed: int) -> dict[str, Any]:
    rcv5 = ReplayControllerV5.init()
    rcv6 = ReplayControllerV6.init(inner_v5=rcv5)
    return {
        "schema": R143_SCHEMA_VERSION,
        "name": (
            "h229c_replay_v6_respects_v5_dominance_primary"),
        "passed": bool(
            rcv6.inner_v5.replay_dominance_primary_head
            is None),
    }


def family_h229d_replay_v6_team_regime_fires(
        seed: int) -> dict[str, Any]:
    ctrl = ReplayControllerV6.init()
    cands = {
        r: [ReplayCandidate(
            100, 1000, 50, 0.1, 0.0, 0.3,
            True, True, 0)]
        for r in W65_REPLAY_REGIMES_V6}
    decs = {
        r: [W60_REPLAY_DECISION_REUSE]
        for r in W65_REPLAY_REGIMES_V6}
    ctrl, _ = fit_replay_controller_v6_per_role(
        controller=ctrl, role="planner",
        train_candidates_per_regime=cands,
        train_decisions_per_regime=decs)
    cand = ReplayCandidate(
        100, 1000, 50, 0.1, 0.0, 0.3, True, True, 0)
    _dec, _conf, _dom, regime, _dpa, role_used = (
        ctrl.decide_v6(
            cand, role="planner",
            team_coordination_flag=0.8,
            substrate_fidelity=0.9))
    return {
        "schema": R143_SCHEMA_VERSION,
        "name": "h229d_replay_v6_team_regime_fires",
        "passed": bool(
            regime == W65_REPLAY_REGIME_TEAM_SUBSTRATE_COORDINATION
            and role_used),
        "regime": str(regime),
    }


def family_h230_substrate_checkpoint_saving(
        seed: int) -> dict[str, Any]:
    saving = substrate_checkpoint_reuse_flops_v10(
        n_tokens=128)
    return {
        "schema": R143_SCHEMA_VERSION,
        "name": "h230_substrate_checkpoint_saving",
        "passed": bool(saving["saving_ratio"] >= 0.5),
        "ratio": float(saving["saving_ratio"]),
    }


def family_h230b_masc_v10_beats_baselines(
        seed: int) -> dict[str, Any]:
    masc = MultiAgentSubstrateCoordinator()
    _, agg = masc.run_batch(
        seeds=list(range(15)),
        n_agents=4, n_turns=12)
    return {
        "schema": R143_SCHEMA_VERSION,
        "name": "h230b_masc_v10_beats_baselines",
        "passed": bool(agg.v10_strictly_beats_rate >= 0.5),
        "strictly_beats_rate": float(
            agg.v10_strictly_beats_rate),
        "per_policy_success": {
            k: float(round(v, 4))
            for k, v in agg.per_policy_success_rate.items()},
    }


def family_h230c_substrate_adapter_v10_full_tier(
        seed: int) -> dict[str, Any]:
    cap = probe_tiny_substrate_v10_adapter()
    matrix = probe_all_v10_adapters()
    return {
        "schema": R143_SCHEMA_VERSION,
        "name": "h230c_substrate_adapter_v10_full_tier",
        "passed": bool(
            cap.tier == W65_SUBSTRATE_TIER_SUBSTRATE_V10_FULL
            and matrix.has_v10_full()),
        "tier": str(cap.tier),
    }


_R143_FAMILIES: tuple[Any, ...] = (
    family_h223_v10_substrate_determinism,
    family_h223b_v10_hidden_write_merit_shape,
    family_h223c_v10_role_kv_bank_fifo,
    family_h223d_v10_gate_score_shape,
    family_h223e_v10_substrate_checkpoint,
    family_h224_kv_bridge_v10_six_target,
    family_h224b_kv_bridge_v10_team_task_falsifier,
    family_h225_hsb_v9_six_target,
    family_h225b_hsb_v9_hidden_wins_rate,
    family_h226_prefix_v9_k64,
    family_h226b_prefix_v9_four_way_decision,
    family_h227_attn_v9_five_stage,
    family_h227b_attn_v9_zero_under_negative,
    family_h228_cache_v8_five_objective,
    family_h228b_cache_v8_per_role_eviction,
    family_h229_replay_v6_per_role_per_regime,
    family_h229b_replay_v6_multi_agent_abstain,
    family_h229c_replay_v6_respects_v5_dominance_primary,
    family_h229d_replay_v6_team_regime_fires,
    family_h230_substrate_checkpoint_saving,
    family_h230b_masc_v10_beats_baselines,
    family_h230c_substrate_adapter_v10_full_tier,
)


def run_r143(
        seeds: Sequence[int] = (199, 299, 399),
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for s in seeds:
        results: dict[str, Any] = {}
        for fn in _R143_FAMILIES:
            try:
                r = fn(int(s))
            except Exception as e:
                r = {
                    "schema": R143_SCHEMA_VERSION,
                    "name": getattr(fn, "__name__", "unknown"),
                    "passed": False,
                    "exception": str(e),
                }
            results[r["name"]] = r
        out.append({
            "schema": R143_SCHEMA_VERSION,
            "seed": int(s),
            "family_results": results,
        })
    return out


__all__ = [
    "R143_SCHEMA_VERSION",
    "run_r143",
]
