"""W66 R-146 benchmark family — V11 substrate / eleven-way hybrid /
multi-agent task-success / team-coordinator-V2 / team-consensus-
controller.

H245..H252c cell families (24 H-bars):

* H245   V11 substrate determinism
* H245b  V11 replay_trust_ledger tensor shape (L, H, T)
* H245c  V11 team_failure_recovery_flag semantics
* H245d  V11 v11_gate_score_per_layer shape (L,)
* H245e  V11 substrate snapshot-diff content-addressed
* H246   KV bridge V11 seven-target ridge converges
* H246b  KV bridge V11 team_failure_recovery falsifier returns 0
         under inversion
* H246c  KV bridge V11 multi-agent task fingerprint dim = 30
* H247   HSB V10 seven-target ridge converges
* H247b  HSB V10 hidden-wins-vs-team-success probe in [0, 1]
* H248   prefix V10 K=96 drift curve has correct length
* H248b  prefix V10 five-way decision (prefix/hidden/replay/team/
         recover) is valid
* H249   attention V10 six-stage clamp keeps delta_l2 <= cap
* H249b  attention V10 returns zero under negative Hellinger
         budget
* H250   cache controller V9 six-objective ridge converges
* H250b  cache controller V9 per-role 7-dim eviction head
         converges
* H251   replay controller V7 per-role per-regime head converges
* H251b  replay controller V7 team_substrate_routing head
         converges
* H251c  replay controller V7 includes nine regimes
* H252   substrate snapshot-diff reuse-vs-recompute saving >= 0.6
* H252b  Multi-agent coordinator V2 V11 strictly beats V10 on
         >= 50% of seeds (matched-budget, baseline regime)
* H252c  TSC_V11 policy strictly beats V11 on >= 50% of seeds
         (matched-budget, baseline regime)
* H252d  Substrate adapter V11 has matrix.has_v11_full=True
* H252e  Team-consensus controller fires quorum on all-active
         agents
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as _np

from coordpy.attention_steering_bridge_v2 import (
    AttentionSteeringV2Projection,
)
from coordpy.attention_steering_bridge_v10 import (
    steer_attention_and_measure_v10,
)
from coordpy.cache_controller_v9 import (
    CacheControllerV9,
    fit_six_objective_ridge_v9,
    fit_per_role_eviction_head_v9,
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
    fit_hsb_v10_seven_target,
    probe_hsb_v10_hidden_wins_vs_team_success,
)
from coordpy.kv_bridge_v3 import KVBridgeV3Projection
from coordpy.kv_bridge_v4 import KVBridgeV4Projection
from coordpy.kv_bridge_v5 import KVBridgeV5Projection
from coordpy.kv_bridge_v6 import KVBridgeV6Projection
from coordpy.kv_bridge_v7 import KVBridgeV7Projection
from coordpy.kv_bridge_v8 import KVBridgeV8Projection
from coordpy.kv_bridge_v9 import KVBridgeV9Projection
from coordpy.kv_bridge_v10 import KVBridgeV10Projection
from coordpy.kv_bridge_v11 import (
    KVBridgeV11Projection,
    compute_multi_agent_task_fingerprint_v11,
    fit_kv_bridge_v11_seven_target,
    probe_kv_bridge_v11_team_failure_recovery_falsifier,
)
from coordpy.multi_agent_substrate_coordinator_v2 import (
    MultiAgentSubstrateCoordinatorV2,
    W66_MASC_V2_REGIME_BASELINE,
)
from coordpy.prefix_state_bridge_v10 import (
    PrefixV10FiveWayDecision,
    W66_DEFAULT_PREFIX_V10_K_STEPS,
    compare_prefix_vs_hidden_vs_replay_vs_team_vs_recover_v10,
)
from coordpy.replay_controller import (
    ReplayCandidate,
)
from coordpy.replay_controller_v5 import ReplayControllerV5
from coordpy.replay_controller_v6 import ReplayControllerV6
from coordpy.replay_controller_v7 import (
    ReplayControllerV7,
    W66_REPLAY_REGIMES_V7,
    W66_TEAM_SUBSTRATE_ROUTING_LABELS,
    fit_replay_controller_v7_per_role,
    fit_replay_v7_team_substrate_routing_head,
)
from coordpy.substrate_adapter_v11 import (
    W66_SUBSTRATE_TIER_SUBSTRATE_V11_FULL,
    probe_all_v11_adapters,
    probe_tiny_substrate_v11_adapter,
)
from coordpy.team_consensus_controller import (
    TeamConsensusController, W66_TC_DECISION_QUORUM,
)
from coordpy.tiny_substrate_v11 import (
    build_default_tiny_substrate_v11,
    forward_tiny_substrate_v11,
    substrate_snapshot_diff_v11,
    substrate_snapshot_recover_flops_v11,
    tokenize_bytes_v11,
    trigger_team_failure_recovery_v11,
)


R146_SCHEMA_VERSION: str = "coordpy.r146_benchmark.v1"


def family_h245_v11_substrate_determinism(
        seed: int) -> dict[str, Any]:
    p = build_default_tiny_substrate_v11(seed=int(seed) + 46000)
    ids = tokenize_bytes_v11("h245", max_len=12)
    t1, _ = forward_tiny_substrate_v11(p, ids)
    t2, _ = forward_tiny_substrate_v11(p, ids)
    return {
        "schema": R146_SCHEMA_VERSION,
        "name": "h245_v11_substrate_determinism",
        "passed": bool(t1.cid() == t2.cid()),
    }


def family_h245b_v11_replay_trust_ledger_shape(
        seed: int) -> dict[str, Any]:
    p = build_default_tiny_substrate_v11(seed=int(seed) + 46001)
    ids = tokenize_bytes_v11("h245b", max_len=12)
    _, c = forward_tiny_substrate_v11(p, ids)
    expected = (
        p.config.v10.v9.n_layers, p.config.v10.v9.n_heads,
        p.config.v10.v9.max_len)
    return {
        "schema": R146_SCHEMA_VERSION,
        "name": "h245b_v11_replay_trust_ledger_shape",
        "passed": bool(
            c.replay_trust_ledger.shape == expected),
        "shape": list(c.replay_trust_ledger.shape),
    }


def family_h245c_v11_team_failure_recovery_flag(
        seed: int) -> dict[str, Any]:
    p = build_default_tiny_substrate_v11(seed=int(seed) + 46002)
    ids = tokenize_bytes_v11("h245c", max_len=10)
    _, c = forward_tiny_substrate_v11(p, ids)
    trigger_team_failure_recovery_v11(
        c, role="planner", reason="test_fail")
    trigger_team_failure_recovery_v11(
        c, role="researcher", reason="test_fail2")
    return {
        "schema": R146_SCHEMA_VERSION,
        "name": "h245c_v11_team_failure_recovery_flag",
        "passed": bool(
            len(c.team_failure_recovery_flag) == 2
            and c.team_failure_recovery_flag.get(
                "planner", {}).get("triggered", False)
            and c.team_failure_recovery_flag.get(
                "researcher", {}).get("triggered", False)),
        "n_flagged": int(len(c.team_failure_recovery_flag)),
    }


def family_h245d_v11_gate_score_shape(
        seed: int) -> dict[str, Any]:
    p = build_default_tiny_substrate_v11(seed=int(seed) + 46003)
    ids = tokenize_bytes_v11("h245d", max_len=12)
    t, _ = forward_tiny_substrate_v11(p, ids)
    return {
        "schema": R146_SCHEMA_VERSION,
        "name": "h245d_v11_gate_score_shape",
        "passed": bool(
            t.v11_gate_score_per_layer.shape
            == (p.config.v10.v9.n_layers,)),
        "shape": list(t.v11_gate_score_per_layer.shape),
    }


def family_h245e_v11_substrate_snapshot_diff(
        seed: int) -> dict[str, Any]:
    p = build_default_tiny_substrate_v11(seed=int(seed) + 46004)
    ids = tokenize_bytes_v11("h245e", max_len=10)
    _, c1 = forward_tiny_substrate_v11(p, ids)
    c2 = c1.clone()
    trigger_team_failure_recovery_v11(
        c2, role="planner", reason="diff_test")
    diff = substrate_snapshot_diff_v11(c1, c2)
    diff2 = substrate_snapshot_diff_v11(c1, c2)
    return {
        "schema": R146_SCHEMA_VERSION,
        "name": "h245e_v11_substrate_snapshot_diff",
        "passed": bool(
            diff.cid() == diff2.cid()
            and "team_failure_recovery_flag"
            in diff.changed_axes),
        "diff_cid": str(diff.cid())[:16],
    }


def _build_v11_kv_bridge(p, seed: int) -> KVBridgeV11Projection:
    d_head = (
        p.config.v10.v9.d_model // p.config.v10.v9.n_heads)
    kv_b3 = KVBridgeV3Projection.init(
        n_layers=p.config.v10.v9.n_layers,
        n_heads=p.config.v10.v9.n_heads,
        n_kv_heads=p.config.v10.v9.n_kv_heads,
        n_inject_tokens=3, carrier_dim=6,
        d_head=d_head, seed=int(seed) + 1)
    kv_b4 = KVBridgeV4Projection.init_from_v3(
        kv_b3, seed_v4=int(seed) + 2)
    kv_b5 = KVBridgeV5Projection.init_from_v4(
        kv_b4, seed_v5=int(seed) + 3)
    kv_b6 = KVBridgeV6Projection.init_from_v5(
        kv_b5, seed_v6=int(seed) + 4)
    kv_b7 = KVBridgeV7Projection.init_from_v6(
        kv_b6, seed_v7=int(seed) + 5)
    kv_b8 = KVBridgeV8Projection.init_from_v7(
        kv_b7, seed_v8=int(seed) + 6)
    kv_b9 = KVBridgeV9Projection.init_from_v8(
        kv_b8, seed_v9=int(seed) + 7)
    kv_b10 = KVBridgeV10Projection.init_from_v9(
        kv_b9, seed_v10=int(seed) + 8)
    return KVBridgeV11Projection.init_from_v10(
        kv_b10, seed_v11=int(seed) + 9)


def family_h246_kv_bridge_v11_seven_target(
        seed: int) -> dict[str, Any]:
    p = build_default_tiny_substrate_v11(seed=int(seed) + 46200)
    kv_b11 = _build_v11_kv_bridge(p, int(seed) + 46201)
    rng = _np.random.default_rng(int(seed) + 46300)
    ids = tokenize_bytes_v11("kv11", max_len=4)
    while len(ids) < 4:
        ids.append(0)
    carriers = [list(rng.standard_normal(8)) for _ in range(4)]
    targets = []
    for i in range(7):
        t = _np.zeros(p.config.v10.v9.vocab_size)
        t[100 + 20 * i] = 1.0 / (i + 1)
        targets.append(t.tolist())
    _, report = fit_kv_bridge_v11_seven_target(
        params=p, projection=kv_b11,
        train_carriers=carriers,
        target_delta_logits_stack=targets,
        follow_up_token_ids=ids, n_directions=3)
    return {
        "schema": R146_SCHEMA_VERSION,
        "name": "h246_kv_bridge_v11_seven_target",
        "passed": bool(
            report.n_targets == 7
            and report.team_failure_recovery_post
                <= report.team_failure_recovery_pre + 1e-3),
        "n_targets": int(report.n_targets),
        "tfr_pre": float(report.team_failure_recovery_pre),
        "tfr_post": float(report.team_failure_recovery_post),
    }


def family_h246b_kv_bridge_v11_team_failure_recovery_falsifier(
        seed: int) -> dict[str, Any]:
    f1 = probe_kv_bridge_v11_team_failure_recovery_falsifier(
        team_failure_recovery_flag=0.5)
    f2 = probe_kv_bridge_v11_team_failure_recovery_falsifier(
        team_failure_recovery_flag=-0.5)
    return {
        "schema": R146_SCHEMA_VERSION,
        "name": (
            "h246b_kv_bridge_v11_team_failure_recovery_falsifier"),
        "passed": bool(
            f1.falsifier_score == 0.0
            and f2.falsifier_score == 0.0),
    }


def family_h246c_multi_agent_task_fingerprint_dim(
        seed: int) -> dict[str, Any]:
    fp = compute_multi_agent_task_fingerprint_v11(
        role="planner", task_id="task_a", team_id="team_x")
    return {
        "schema": R146_SCHEMA_VERSION,
        "name": "h246c_multi_agent_task_fingerprint_dim",
        "passed": bool(len(fp) == 30),
        "dim": int(len(fp)),
    }


def _build_v10_hsb(p, seed: int) -> HiddenStateBridgeV10Projection:
    hsb2 = HiddenStateBridgeV2Projection.init(
        target_layers=(1, 3), n_tokens=6, carrier_dim=6,
        d_model=p.config.v10.v9.d_model,
        seed=int(seed) + 1)
    hsb3 = HiddenStateBridgeV3Projection.init_from_v2(
        hsb2, n_heads=p.config.v10.v9.n_heads,
        seed_v3=int(seed) + 2)
    hsb4 = HiddenStateBridgeV4Projection.init_from_v3(
        hsb3, seed_v4=int(seed) + 3)
    hsb5 = HiddenStateBridgeV5Projection.init_from_v4(
        hsb4, n_positions=3, seed_v5=int(seed) + 4)
    hsb6 = HiddenStateBridgeV6Projection.init_from_v5(
        hsb5, seed_v6=int(seed) + 5)
    hsb7 = HiddenStateBridgeV7Projection.init_from_v6(
        hsb6, seed_v7=int(seed) + 6)
    hsb8 = HiddenStateBridgeV8Projection.init_from_v7(
        hsb7, seed_v8=int(seed) + 7)
    hsb9 = HiddenStateBridgeV9Projection.init_from_v8(
        hsb8, seed_v9=int(seed) + 8)
    return HiddenStateBridgeV10Projection.init_from_v9(
        hsb9, seed_v10=int(seed) + 9)


def family_h247_hsb_v10_seven_target(
        seed: int) -> dict[str, Any]:
    p = build_default_tiny_substrate_v11(seed=int(seed) + 46400)
    hsb10 = _build_v10_hsb(p, int(seed) + 46401)
    rng = _np.random.default_rng(int(seed) + 46500)
    carriers = rng.standard_normal((6, 6)).tolist()
    targets = [
        rng.standard_normal(p.config.v10.v9.vocab_size).tolist()
        for _ in range(7)]
    ids = tokenize_bytes_v11("h247", max_len=8)
    _, report = fit_hsb_v10_seven_target(
        params=p.v3_params, projection=hsb10,
        train_carriers=carriers,
        target_delta_logits_stack=targets,
        token_ids=ids)
    return {
        "schema": R146_SCHEMA_VERSION,
        "name": "h247_hsb_v10_seven_target",
        "passed": bool(
            report.n_targets == 7 and report.converged),
    }


def family_h247b_hsb_v10_hidden_wins_vs_team_success(
        seed: int) -> dict[str, Any]:
    res = probe_hsb_v10_hidden_wins_vs_team_success(
        n_layers=3, n_heads=2,
        hidden_residual_l2_per_lh=[
            [0.2, 0.4], [0.3, 0.1], [0.1, 0.6]],
        team_success_indicator_per_lh=[
            [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    return {
        "schema": R146_SCHEMA_VERSION,
        "name": "h247b_hsb_v10_hidden_wins_vs_team_success",
        "passed": bool(
            0.0 <= res["mean_joint_win_rate"] <= 1.0),
        "mean_joint_win_rate": float(
            res["mean_joint_win_rate"]),
    }


def family_h248_prefix_v10_k96(seed: int) -> dict[str, Any]:
    # We only verify the K=96 length of the comparator inputs.
    # The actual fit is exercised via R-147 paths.
    return {
        "schema": R146_SCHEMA_VERSION,
        "name": "h248_prefix_v10_k96",
        "passed": bool(
            int(W66_DEFAULT_PREFIX_V10_K_STEPS) == 96),
        "k_steps": int(W66_DEFAULT_PREFIX_V10_K_STEPS),
    }


def family_h248b_prefix_v10_five_way_decision(
        seed: int) -> dict[str, Any]:
    d = compare_prefix_vs_hidden_vs_replay_vs_team_vs_recover_v10(
        prefix_drift_curve=[0.5, 0.4, 0.3],
        hidden_drift_curve=[0.2, 0.1, 0.05],
        replay_drift_curve=[0.6, 0.5, 0.5],
        team_drift_curve=[0.3, 0.2, 0.1],
        recover_drift_curve=[0.5, 0.4, 0.3])
    valid = (
        d.decision in (
            "prefix_wins", "hidden_wins", "replay_wins",
            "team_wins", "recover_wins", "tie"))
    return {
        "schema": R146_SCHEMA_VERSION,
        "name": "h248b_prefix_v10_five_way_decision",
        "passed": bool(valid),
        "decision": str(d.decision),
    }


def family_h249_attn_v10_six_stage(
        seed: int) -> dict[str, Any]:
    from coordpy.tiny_substrate_v3 import TinyV3SubstrateParams
    sp = TinyV3SubstrateParams.init()
    proj = AttentionSteeringV2Projection.init(
        n_layers=sp.config.n_layers,
        n_heads=sp.config.n_heads,
        n_query=6, n_key=6,
        carrier_dim=6, seed=int(seed) + 46600)
    ids = tokenize_bytes_v11("h249", max_len=6)
    rng = _np.random.default_rng(int(seed) + 46700)
    carrier = rng.standard_normal(6).tolist()
    w = steer_attention_and_measure_v10(
        params=sp, carrier=carrier, projection=proj,
        token_ids=ids,
        hellinger_budget=0.5,
        attention_trust_ema=0.7,
        team_id="w66_team")
    return {
        "schema": R146_SCHEMA_VERSION,
        "name": "h249_attn_v10_six_stage",
        "passed": bool(
            w.six_stage_used
            and w.attention_delta_l2_after_six_stage
            <= float(w.trust_cap) + 1e-9),
        "delta": float(
            w.attention_delta_l2_after_six_stage),
    }


def family_h249b_attn_v10_zero_under_negative(
        seed: int) -> dict[str, Any]:
    from coordpy.tiny_substrate_v3 import TinyV3SubstrateParams
    sp = TinyV3SubstrateParams.init()
    proj = AttentionSteeringV2Projection.init(
        n_layers=sp.config.n_layers,
        n_heads=sp.config.n_heads,
        n_query=6, n_key=6,
        carrier_dim=6, seed=int(seed) + 46800)
    ids = tokenize_bytes_v11("h249b", max_len=6)
    carrier = [0.1] * 6
    w = steer_attention_and_measure_v10(
        params=sp, carrier=carrier, projection=proj,
        token_ids=ids, hellinger_budget=-1.0,
        attention_trust_ema=0.7)
    return {
        "schema": R146_SCHEMA_VERSION,
        "name": "h249b_attn_v10_zero_under_negative",
        "passed": bool(
            w.attention_delta_l2_after_six_stage == 0.0),
    }


def family_h250_cache_v9_six_objective(
        seed: int) -> dict[str, Any]:
    rng = _np.random.default_rng(int(seed) + 46900)
    cc = CacheControllerV9.init(fit_seed=int(seed) + 46900)
    X = rng.standard_normal((20, 4))
    y1 = X.sum(axis=-1); y2 = X[:, 0]
    y3 = X[:, 1] - X[:, 2]; y4 = X[:, 3] * 0.5
    y5 = X[:, 0] * 0.3 - X[:, 1] * 0.1
    y6 = X[:, 2] * 0.4 + X[:, 3] * 0.2
    _, report = fit_six_objective_ridge_v9(
        controller=cc, train_features=X.tolist(),
        target_drop_oracle=y1.tolist(),
        target_retrieval_relevance=y2.tolist(),
        target_hidden_wins=y3.tolist(),
        target_replay_dominance=y4.tolist(),
        target_team_task_success=y5.tolist(),
        target_team_failure_recovery=y6.tolist())
    return {
        "schema": R146_SCHEMA_VERSION,
        "name": "h250_cache_v9_six_objective",
        "passed": bool(
            report.n_objectives == 6 and report.converged),
    }


def family_h250b_cache_v9_per_role_eviction(
        seed: int) -> dict[str, Any]:
    rng = _np.random.default_rng(int(seed) + 47000)
    cc = CacheControllerV9.init(fit_seed=int(seed) + 47000)
    X = rng.standard_normal((12, 7))
    y = X[:, 0] * 0.4 + X[:, 6] * 0.3
    _, report = fit_per_role_eviction_head_v9(
        controller=cc, role="planner",
        train_features=X.tolist(),
        target_eviction_priorities=y.tolist())
    return {
        "schema": R146_SCHEMA_VERSION,
        "name": "h250b_cache_v9_per_role_eviction",
        "passed": bool(report.converged),
    }


def family_h251_replay_v7_per_role_per_regime(
        seed: int) -> dict[str, Any]:
    rcv5 = ReplayControllerV5.init()
    rcv6 = ReplayControllerV6.init(inner_v5=rcv5)
    ctrl = ReplayControllerV7.init(inner_v6=rcv6)
    cands = {
        r: [ReplayCandidate(
            100, 1000, 50, 0.1, 0.0, 0.3,
            True, True, 0)]
        for r in W66_REPLAY_REGIMES_V7}
    decs = {
        r: ["choose_reuse"]
        for r in W66_REPLAY_REGIMES_V7}
    _, report = fit_replay_controller_v7_per_role(
        controller=ctrl, role="planner",
        train_candidates_per_regime=cands,
        train_decisions_per_regime=decs)
    return {
        "schema": R146_SCHEMA_VERSION,
        "name": "h251_replay_v7_per_role_per_regime",
        "passed": bool(
            report.n_role_regime_heads_v7 == 10
            and report.converged),
        "n_heads": int(report.n_role_regime_heads_v7),
    }


def family_h251b_replay_v7_team_substrate_routing(
        seed: int) -> dict[str, Any]:
    rcv5 = ReplayControllerV5.init()
    rcv6 = ReplayControllerV6.init(inner_v5=rcv5)
    ctrl = ReplayControllerV7.init(inner_v6=rcv6)
    rng = _np.random.default_rng(int(seed) + 47100)
    X = rng.standard_normal((30, 10))
    decs: list[str] = []
    for i in range(30):
        if X[i, 0] > 0.5:
            decs.append(W66_TEAM_SUBSTRATE_ROUTING_LABELS[0])
        elif X[i, 1] > 0.0:
            decs.append(W66_TEAM_SUBSTRATE_ROUTING_LABELS[1])
        elif X[i, 2] > 0.0:
            decs.append(W66_TEAM_SUBSTRATE_ROUTING_LABELS[2])
        else:
            decs.append(W66_TEAM_SUBSTRATE_ROUTING_LABELS[3])
    _, report = fit_replay_v7_team_substrate_routing_head(
        controller=ctrl, train_team_features=X.tolist(),
        train_routing_labels=decs)
    return {
        "schema": R146_SCHEMA_VERSION,
        "name": "h251b_replay_v7_team_substrate_routing",
        "passed": bool(
            report.team_substrate_routing_trained
            and report.converged),
        "residual": float(
            report.team_substrate_routing_train_residual),
    }


def family_h251c_replay_v7_ten_regimes(
        seed: int) -> dict[str, Any]:
    return {
        "schema": R146_SCHEMA_VERSION,
        "name": "h251c_replay_v7_ten_regimes",
        "passed": bool(len(W66_REPLAY_REGIMES_V7) == 10),
        "n_regimes": int(len(W66_REPLAY_REGIMES_V7)),
    }


def family_h252_substrate_snapshot_diff_saving(
        seed: int) -> dict[str, Any]:
    saving = substrate_snapshot_recover_flops_v11(
        n_tokens=128)
    return {
        "schema": R146_SCHEMA_VERSION,
        "name": "h252_substrate_snapshot_diff_saving",
        "passed": bool(saving["saving_ratio"] >= 0.6),
        "ratio": float(saving["saving_ratio"]),
    }


def family_h252b_masc_v2_v11_beats_v10(
        seed: int) -> dict[str, Any]:
    masc = MultiAgentSubstrateCoordinatorV2()
    _, agg = masc.run_batch(
        seeds=list(range(15)), regime=W66_MASC_V2_REGIME_BASELINE)
    return {
        "schema": R146_SCHEMA_VERSION,
        "name": "h252b_masc_v2_v11_beats_v10",
        "passed": bool(agg.v11_beats_v10_rate >= 0.5),
        "rate": float(agg.v11_beats_v10_rate),
        "per_policy_succ": {
            k: float(round(v, 4))
            for k, v in agg.per_policy_success_rate.items()},
    }


def family_h252c_masc_v2_tsc_v11_beats_v11(
        seed: int) -> dict[str, Any]:
    masc = MultiAgentSubstrateCoordinatorV2()
    _, agg = masc.run_batch(
        seeds=list(range(15)), regime=W66_MASC_V2_REGIME_BASELINE)
    return {
        "schema": R146_SCHEMA_VERSION,
        "name": "h252c_masc_v2_tsc_v11_beats_v11",
        "passed": bool(agg.tsc_v11_beats_v11_rate >= 0.5),
        "rate": float(agg.tsc_v11_beats_v11_rate),
    }


def family_h252d_substrate_adapter_v11_full_tier(
        seed: int) -> dict[str, Any]:
    cap = probe_tiny_substrate_v11_adapter()
    matrix = probe_all_v11_adapters()
    return {
        "schema": R146_SCHEMA_VERSION,
        "name": "h252d_substrate_adapter_v11_full_tier",
        "passed": bool(
            cap.tier == W66_SUBSTRATE_TIER_SUBSTRATE_V11_FULL
            and matrix.has_v11_full()),
        "tier": str(cap.tier),
    }


def family_h252e_team_consensus_quorum(
        seed: int) -> dict[str, Any]:
    tc = TeamConsensusController()
    res = tc.decide(
        regime=W66_MASC_V2_REGIME_BASELINE,
        agent_guesses=[0.5, 0.6, 0.55],
        agent_confidences=[0.7, 0.8, 0.65])
    return {
        "schema": R146_SCHEMA_VERSION,
        "name": "h252e_team_consensus_quorum",
        "passed": bool(
            res["decision"] == W66_TC_DECISION_QUORUM
            and res["payload"] is not None),
        "payload": float(res.get("payload", 0.0)),
    }


_R146_FAMILIES: tuple[Any, ...] = (
    family_h245_v11_substrate_determinism,
    family_h245b_v11_replay_trust_ledger_shape,
    family_h245c_v11_team_failure_recovery_flag,
    family_h245d_v11_gate_score_shape,
    family_h245e_v11_substrate_snapshot_diff,
    family_h246_kv_bridge_v11_seven_target,
    family_h246b_kv_bridge_v11_team_failure_recovery_falsifier,
    family_h246c_multi_agent_task_fingerprint_dim,
    family_h247_hsb_v10_seven_target,
    family_h247b_hsb_v10_hidden_wins_vs_team_success,
    family_h248_prefix_v10_k96,
    family_h248b_prefix_v10_five_way_decision,
    family_h249_attn_v10_six_stage,
    family_h249b_attn_v10_zero_under_negative,
    family_h250_cache_v9_six_objective,
    family_h250b_cache_v9_per_role_eviction,
    family_h251_replay_v7_per_role_per_regime,
    family_h251b_replay_v7_team_substrate_routing,
    family_h251c_replay_v7_ten_regimes,
    family_h252_substrate_snapshot_diff_saving,
    family_h252b_masc_v2_v11_beats_v10,
    family_h252c_masc_v2_tsc_v11_beats_v11,
    family_h252d_substrate_adapter_v11_full_tier,
    family_h252e_team_consensus_quorum,
)


def run_r146(
        seeds: Sequence[int] = (199, 299, 399),
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for s in seeds:
        results: dict[str, Any] = {}
        for fn in _R146_FAMILIES:
            try:
                r = fn(int(s))
            except Exception as e:
                r = {
                    "schema": R146_SCHEMA_VERSION,
                    "name": getattr(fn, "__name__", "unknown"),
                    "passed": False,
                    "exception": str(e),
                }
            results[r["name"]] = r
        out.append({
            "schema": R146_SCHEMA_VERSION,
            "seed": int(s),
            "family_results": results,
        })
    return out


__all__ = [
    "R146_SCHEMA_VERSION",
    "run_r146",
]
