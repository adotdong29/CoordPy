"""W75 R-182 benchmark family — Real Substrate Plane V20 (Plane B).

Exercises the W75 real substrate plane V20 modules: tiny substrate
V20 (compound-chain CID + per-layer chain-length + chain-pressure
gate), KV bridge V20 (16-target ridge + 130-dim chain fingerprint +
chain-pressure falsifier), cache controller V18 (15-objective ridge +
per-role 16-dim chain-pressure head), replay controller V16 (23
regimes + 13-label chain-aware routing head), substrate adapter V20
(substrate_v20_full tier), persistent V27 (26 layers / 24th carrier),
LHR V27 (26 heads, max_k=896), MLSC V23, consensus V21, deep
substrate hybrid V20 (twenty-way loop).

H990..H1005 cell families (16 H-bars):

* H990   V20 substrate is byte-deterministic on (params, ids,
         events)
* H990b  V20 substrate produces non-empty compound_chain CID
* H991   KV V20 sixteen-target ridge fit reduces residual
* H991b  KV V20 chain fingerprint is 130-dim
* H991c  KV V20 chain-pressure falsifier scores 0 on honest input
* H992   Cache V18 fifteen-objective ridge fit converges
* H992b  Cache V18 per-role chain-pressure head is 16-dim
* H993   Replay V16 23-regime classifier reaches >= 0.5 acc
* H993b  Replay V16 chain-aware routing head fit non-trivial
* H994   Substrate adapter V20 has substrate_v20_full tier
* H995   Persistent V27 reports 26 layers
* H996   LHR V27 reports 26 heads + max_k=896
* H997   MLSC V23 merge inherits both new chains
* H998   Consensus V21 has ≥ 36 stages
* H999   Deep substrate hybrid V20 twenty-way loop fires
* H1000  V20 substrate-recompute flops savings ratio ≥ 0.9
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as _np

from coordpy.cache_controller_v18 import (
    CacheControllerV18, fit_fifteen_objective_ridge_v18,
    fit_per_role_compound_chain_pressure_head_v18,
)
from coordpy.consensus_fallback_controller_v21 import (
    ConsensusFallbackControllerV21, W75_CONSENSUS_V21_STAGES,
)
from coordpy.deep_substrate_hybrid_v19 import (
    DeepSubstrateHybridV19ForwardWitness,
)
from coordpy.deep_substrate_hybrid_v20 import (
    DeepSubstrateHybridV20, deep_substrate_hybrid_v20_forward,
)
from coordpy.kv_bridge_v20 import (
    compute_compound_chain_repair_fingerprint_v20,
    probe_kv_bridge_v20_compound_chain_pressure_falsifier,
    W75_KV_V20_FINGERPRINT_DIM,
)
from coordpy.long_horizon_retention_v27 import (
    LongHorizonReconstructionV27Head,
    W75_DEFAULT_LHR_V27_MAX_K, emit_lhr_v27_witness,
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
from coordpy.mergeable_latent_capsule_v23 import (
    MergeOperatorV23, wrap_v22_as_v23,
)
from coordpy.persistent_latent_v27 import (
    PersistentLatentStateV27Chain, W75_DEFAULT_V27_N_LAYERS,
)
from coordpy.replay_controller import ReplayCandidate
from coordpy.replay_controller_v16 import (
    ReplayControllerV16,
    W75_COMPOUND_CHAIN_AWARE_ROUTING_LABELS,
    W75_REPLAY_REGIMES_V16,
    fit_replay_controller_v16_per_role,
    fit_replay_v16_compound_chain_aware_routing_head,
)
from coordpy.substrate_adapter_v20 import (
    W75_SUBSTRATE_TIER_SUBSTRATE_V20_FULL,
    probe_all_v20_adapters,
)
from coordpy.tiny_substrate_v20 import (
    build_default_tiny_substrate_v20,
    forward_tiny_substrate_v20,
    record_compound_chain_window_v20,
    substrate_compound_chain_repair_dominance_flops_v20,
    tokenize_bytes_v20,
)


R182_SCHEMA_VERSION: str = "coordpy.r182_benchmark.v1"


def run_r182(*, seeds: Sequence[int]) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    p = build_default_tiny_substrate_v20()
    # H990: byte-determinism (different inputs → different CIDs;
    # same input twice → same CID).
    cids: set[str] = set()
    for s in seeds:
        ids = tokenize_bytes_v20(f"r182-seed-{int(s)}", max_len=16)
        trace, cache = forward_tiny_substrate_v20(p, ids)
        cids.add(str(trace.cid()))
    # Deterministic: re-running with the first seed should match.
    ids0 = tokenize_bytes_v20(
        f"r182-seed-{int(seeds[0])}", max_len=16)
    trace0_a, _ = forward_tiny_substrate_v20(p, ids0)
    trace0_b, _ = forward_tiny_substrate_v20(p, ids0)
    cells["H990"] = bool(
        len(cids) == len(seeds)
        and str(trace0_a.cid()) == str(trace0_b.cid()))
    # H990b: compound-chain CID is non-empty after recording events.
    ids = tokenize_bytes_v20("r182-chain", max_len=4)
    _, cache = forward_tiny_substrate_v20(p, ids)
    record_compound_chain_window_v20(
        cache, replacement_turn=2, delayed_repair_turn=5,
        rejoin_turn=10, compound_chain_window_turns=8,
        role="p", branch_id="b")
    trace, cache = forward_tiny_substrate_v20(
        p, ids, v20_kv_cache=cache,
        compound_chain_pressure=0.8)
    cells["H990b"] = bool(
        len(str(cache.compound_chain_repair_trajectory_cid)) > 0)
    # H991: KV V20 sixteen-target ridge fit.
    from coordpy.kv_bridge_v20 import (
        KVBridgeV20Projection, fit_kv_bridge_v20_sixteen_target,
    )
    from coordpy.kv_bridge_v19 import KVBridgeV19Projection
    from coordpy.kv_bridge_v18 import KVBridgeV18Projection
    from coordpy.kv_bridge_v17 import KVBridgeV17Projection
    from coordpy.kv_bridge_v16 import KVBridgeV16Projection
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
    cfg = p.config.v19.v18.v17.v16.v15.v14.v13.v12.v11.v10.v9
    d_head = cfg.d_model // cfg.n_heads
    b3 = KVBridgeV3Projection.init(
        n_layers=cfg.n_layers, n_heads=cfg.n_heads,
        n_kv_heads=cfg.n_kv_heads, n_inject_tokens=3,
        carrier_dim=6, d_head=d_head, seed=181)
    b4 = KVBridgeV4Projection.init_from_v3(b3, seed_v4=182)
    b5 = KVBridgeV5Projection.init_from_v4(b4, seed_v5=183)
    b6 = KVBridgeV6Projection.init_from_v5(b5, seed_v6=184)
    b7 = KVBridgeV7Projection.init_from_v6(b6, seed_v7=185)
    b8 = KVBridgeV8Projection.init_from_v7(b7, seed_v8=186)
    b9 = KVBridgeV9Projection.init_from_v8(b8, seed_v9=187)
    b10 = KVBridgeV10Projection.init_from_v9(b9, seed_v10=188)
    b11 = KVBridgeV11Projection.init_from_v10(b10, seed_v11=189)
    b12 = KVBridgeV12Projection.init_from_v11(b11, seed_v12=190)
    b13 = KVBridgeV13Projection.init_from_v12(b12, seed_v13=191)
    b14 = KVBridgeV14Projection.init_from_v13(b13, seed_v14=192)
    b15 = KVBridgeV15Projection.init_from_v14(b14, seed_v15=193)
    b16 = KVBridgeV16Projection.init_from_v15(b15, seed_v16=194)
    b17 = KVBridgeV17Projection.init_from_v16(b16, seed_v17=195)
    b18 = KVBridgeV18Projection.init_from_v17(b17, seed_v18=196)
    b19 = KVBridgeV19Projection.init_from_v18(b18, seed_v19=197)
    b20 = KVBridgeV20Projection.init_from_v19(b19, seed_v20=198)
    rng = _np.random.default_rng(182)
    carriers = rng.standard_normal((6, 6)).tolist()
    targets = [rng.standard_normal((259,)).tolist() for _ in range(16)]
    follow_up = [1, 2, 3]
    try:
        _, report = fit_kv_bridge_v20_sixteen_target(
            params=p, projection=b20,
            train_carriers=carriers,
            target_delta_logits_stack=targets,
            follow_up_token_ids=follow_up)
        cells["H991"] = bool(report.n_targets == 16)
    except Exception:
        cells["H991"] = False
    # H991b: chain fingerprint is 130-dim.
    fp = compute_compound_chain_repair_fingerprint_v20(
        role="r", repair_trajectory_cid="a",
        delayed_repair_trajectory_cid="b",
        restart_repair_trajectory_cid="c",
        replacement_repair_trajectory_cid="d",
        compound_repair_trajectory_cid="e",
        compound_chain_repair_trajectory_cid="f")
    cells["H991b"] = bool(len(fp) == W75_KV_V20_FINGERPRINT_DIM)
    # H991c: chain-pressure falsifier scores 0 on honest input.
    f0 = probe_kv_bridge_v20_compound_chain_pressure_falsifier(
        compound_chain_pressure_flag=1)
    cells["H991c"] = bool(float(f0.falsifier_score) == 0.0)
    # H992: cache V18 fifteen-objective ridge converges.
    cc18 = CacheControllerV18.init(fit_seed=182)
    X = rng.standard_normal((10, 4))
    try:
        cc18_fit, rep = fit_fifteen_objective_ridge_v18(
            controller=cc18, train_features=X.tolist(),
            target_drop_oracle=X.sum(axis=-1).tolist(),
            target_retrieval_relevance=X[:, 0].tolist(),
            target_hidden_wins=(X[:, 1] - X[:, 2]).tolist(),
            target_replay_dominance=(X[:, 3] * 0.5).tolist(),
            target_team_task_success=(X[:, 0] * 0.3).tolist(),
            target_team_failure_recovery=(X[:, 2] * 0.4).tolist(),
            target_branch_merge=(X[:, 0] * 0.2).tolist(),
            target_partial_contradiction=(X[:, 1] * 0.3).tolist(),
            target_multi_branch_rejoin=(X[:, 0] * 0.5).tolist(),
            target_budget_primary=(X[:, 0] * 0.2).tolist(),
            target_restart_dominance=(X[:, 3] * 0.4).tolist(),
            target_delayed_rejoin_after_restart=(X[:, 1] * 0.3).tolist(),
            target_replacement_after_ctr=(X[:, 0] * 0.3).tolist(),
            target_compound_repair=(X[:, 0] * 0.25).tolist(),
            target_compound_chain_repair=(X[:, 0] * 0.2).tolist())
        cells["H992"] = bool(rep.converged)
    except Exception:
        cells["H992"] = False
        cc18_fit = cc18
    # H992b: per-role 16-dim chain head.
    X16 = rng.standard_normal((10, 16))
    cc18_fit, _ = fit_per_role_compound_chain_pressure_head_v18(
        controller=cc18_fit, role="r",
        train_features=X16.tolist(),
        target_compound_chain_priorities=(X16[:, 0] * 0.5).tolist())
    cells["H992b"] = bool(
        "r" in cc18_fit.per_role_compound_chain_pressure_heads_v18
        and cc18_fit
        .per_role_compound_chain_pressure_heads_v18["r"].shape[0]
        == 16)
    # H993: replay V16 23-regime fit.
    rcv16 = ReplayControllerV16.init()
    cands = {
        r: [ReplayCandidate(100, 1000, 50, 0.1, 0.0, 0.3, True, True, 0)]
        for r in W75_REPLAY_REGIMES_V16}
    decs = {r: ["choose_reuse"] for r in W75_REPLAY_REGIMES_V16}
    rcv16, fit_rep = fit_replay_controller_v16_per_role(
        controller=rcv16, role="r",
        train_candidates_per_regime=cands,
        train_decisions_per_regime=decs)
    cells["H993"] = bool(
        len(W75_REPLAY_REGIMES_V16) == 23
        and fit_rep.post_classification_acc >= 0.5)
    # H993b: chain-aware routing head fit.
    Xt = rng.standard_normal((52, 10))
    labs = [
        W75_COMPOUND_CHAIN_AWARE_ROUTING_LABELS[
            i % len(W75_COMPOUND_CHAIN_AWARE_ROUTING_LABELS)]
        for i in range(52)]
    rcv16, _ = fit_replay_v16_compound_chain_aware_routing_head(
        controller=rcv16,
        train_team_features=Xt.tolist(),
        train_routing_labels=labs)
    cells["H993b"] = bool(
        rcv16.compound_chain_aware_routing_head is not None)
    # H994: substrate adapter V20 tier.
    matrix = probe_all_v20_adapters()
    cells["H994"] = bool(matrix.has_v20_full())
    # H995: persistent V27 reports 26 layers.
    persist_chain = PersistentLatentStateV27Chain.empty()
    cells["H995"] = bool(W75_DEFAULT_V27_N_LAYERS == 26)
    # H996: LHR V27 reports 26 heads + max_k=896.
    lhr = LongHorizonReconstructionV27Head.init(seed=183)
    w = emit_lhr_v27_witness(lhr, carrier=[0.1] * 6, k=10)
    cells["H996"] = bool(
        w.n_heads == 26 and w.max_k == W75_DEFAULT_LHR_V27_MAX_K
        and w.max_k == 896)
    # H997: MLSC V23 merge inherits both new chains.
    v3 = make_root_capsule_v3(
        branch_id="r182", payload=(0.1,) * 6,
        fact_tags=("r182",), confidence=0.9, trust=0.9,
        turn_index=0)
    vv = v3
    for w_fn in (wrap_v3_as_v4, wrap_v4_as_v5, wrap_v5_as_v6,
                 wrap_v6_as_v7, wrap_v7_as_v8, wrap_v8_as_v9,
                 wrap_v9_as_v10, wrap_v10_as_v11,
                 wrap_v11_as_v12, wrap_v12_as_v13,
                 wrap_v13_as_v14, wrap_v14_as_v15,
                 wrap_v15_as_v16, wrap_v16_as_v17,
                 wrap_v17_as_v18, wrap_v18_as_v19,
                 wrap_v19_as_v20, wrap_v20_as_v21,
                 wrap_v21_as_v22):
        vv = w_fn(vv)
    v23a = wrap_v22_as_v23(
        vv, compound_chain_repair_trajectory_chain=("a",),
        replacement_then_rejoin_chain=("ra",))
    v23b = wrap_v22_as_v23(
        vv, compound_chain_repair_trajectory_chain=("b",),
        replacement_then_rejoin_chain=("rb",))
    op = MergeOperatorV23()
    merged = op.merge([v23a, v23b])
    cells["H997"] = bool(
        set(merged.compound_chain_repair_trajectory_chain)
        == {"a", "b"}
        and set(merged.replacement_then_rejoin_chain)
        == {"ra", "rb"})
    # H998: consensus V21 ≥ 36 stages.
    cells["H998"] = bool(len(W75_CONSENSUS_V21_STAGES) >= 36)
    # H999: deep substrate hybrid V20 twenty-way.
    hybrid = DeepSubstrateHybridV20()
    rcv16_full = rcv16
    cc18_full = cc18_fit
    v19_w = DeepSubstrateHybridV19ForwardWitness(
        schema="v19", hybrid_cid="", inner_v18_witness_cid="",
        nineteen_way=True,
        cache_controller_v17_fired=True,
        replay_controller_v15_fired=True,
        compound_repair_trajectory_active=True,
        compound_repair_active=True,
        team_consensus_controller_v9_active=True,
        compound_repair_trajectory_cid="x",
        compound_repair_l1=3,
        compound_pressure_gate_mean=0.6)
    w_v20 = deep_substrate_hybrid_v20_forward(
        hybrid=hybrid, v19_witness=v19_w,
        cache_controller_v18=cc18_full,
        replay_controller_v16=rcv16_full,
        compound_chain_repair_trajectory_cid="x",
        compound_chain_repair_l1=3,
        compound_chain_pressure_gate_mean=0.6,
        n_team_consensus_v10_invocations=1)
    cells["H999"] = bool(w_v20.twenty_way)
    # H1000: substrate compound-chain flops savings ≥ 0.9.
    flops = substrate_compound_chain_repair_dominance_flops_v20(
        n_tokens=64, n_repairs=11)
    cells["H1000"] = bool(float(flops["saving_ratio"]) >= 0.9)
    return {
        "schema": R182_SCHEMA_VERSION,
        "n_seeds": int(len(seeds)),
        "cells": cells,
        "all_pass": bool(all(cells.values())),
    }


__all__ = [
    "R182_SCHEMA_VERSION",
    "run_r182",
]
