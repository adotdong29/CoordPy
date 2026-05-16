"""W69 R-157 benchmark family — Real Substrate Plane V14 (Plane B).

Exercises the W69 real substrate plane: V14 substrate determinism,
new V14 axes, KV bridge V14, HSB V13, prefix V13, cache V12,
replay V10, persistent V21, LHR V21, ECC V21, MH V19, MLSC V17,
consensus V15, CRC V17, multi-branch-rejoin and silent-corruption
primitives.

H410..H425 cell families (16 H-bars):

* H410   V14 substrate determinism
* H410b  V14 multi-branch-rejoin-witness tensor shape (L, H, T)
* H410c  V14 silent-corruption-witness semantics + member-replace
* H410d  V14 substrate self-checksum CID is deterministic
* H410e  V14 gate score shape (L,) and L=16
* H410f  V14 multi-branch-rejoin flops saving ≥ 80 %
* H410g  V14 substrate self-checksum 1-byte detect rate ≥ 95 %
* H411   KV V14 ten-target ridge converges
* H411b  KV V14 60-dim silent-corruption fingerprint length
* H411c  KV V14 multi-branch-rejoin falsifier
* H412   HSB V13 hidden-vs-multi-branch-rejoin probe in [0, 1]
* H413   Prefix V13 K-steps = 256 + 8-way comparator
* H414   Cache V12 nine-objective converges
* H415   Replay V10 16-regime + per-role + multi-branch-rejoin
         routing head
* H416   Substrate adapter V14 only-tier-full at in-repo
* H417   ECC V21 ≥ 2^35 codes
* H418   CRC V17 131072-bucket detect rate ≥ 0.95
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as _np

from coordpy.hidden_state_bridge_v13 import (
    compute_hsb_v13_multi_branch_rejoin_margin,
    probe_hsb_v13_hidden_vs_multi_branch_rejoin,
)
from coordpy.kv_bridge_v14 import (
    KVBridgeV14Projection,
    compute_silent_corruption_fingerprint_v14,
    fit_kv_bridge_v14_ten_target,
    probe_kv_bridge_v14_multi_branch_rejoin_falsifier,
)
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
from coordpy.cache_controller_v12 import (
    CacheControllerV12,
    fit_nine_objective_ridge_v12,
    fit_per_role_silent_corruption_head_v12,
)
from coordpy.prefix_state_bridge_v13 import (
    W69_DEFAULT_PREFIX_V13_K_STEPS,
    compare_prefix_v13_eight_way,
)
from coordpy.replay_controller import ReplayCandidate
from coordpy.replay_controller_v10 import (
    ReplayControllerV10,
    W69_MULTI_BRANCH_REJOIN_ROUTING_LABELS,
    W69_REPLAY_REGIMES_V10,
    fit_replay_controller_v10_per_role,
    fit_replay_v10_multi_branch_rejoin_routing_head,
)
from coordpy.substrate_adapter_v14 import (
    W69_SUBSTRATE_TIER_SUBSTRATE_V14_FULL,
    probe_all_v14_adapters,
)
from coordpy.tiny_substrate_v14 import (
    build_default_tiny_substrate_v14,
    forward_tiny_substrate_v14,
    record_multi_branch_rejoin_witness_v14,
    repair_silent_corruption_v14,
    substrate_multi_branch_rejoin_flops_v14,
    substrate_self_checksum_v14,
    substrate_silent_corruption_detect_v14,
    tokenize_bytes_v14,
    trigger_silent_corruption_v14,
)
from coordpy.ecc_codebook_v21 import (
    ECCCodebookV21, probe_ecc_v21_rate_floor_falsifier,
)
from coordpy.corruption_robust_carrier_v17 import (
    CorruptionRobustCarrierV17,
    emit_corruption_robustness_v17_witness,
)


R157_SCHEMA_VERSION: str = "coordpy.r157_benchmark.v1"


def family_h410_v14_substrate_determinism(
        seed: int) -> dict[str, Any]:
    p = build_default_tiny_substrate_v14(seed=int(seed) + 57000)
    ids = tokenize_bytes_v14("h410", max_len=12)
    t1, _ = forward_tiny_substrate_v14(p, ids)
    t2, _ = forward_tiny_substrate_v14(p, ids)
    return {
        "schema": R157_SCHEMA_VERSION,
        "name": "h410_v14_substrate_determinism",
        "passed": bool(t1.cid() == t2.cid()),
    }


def family_h410b_multi_branch_rejoin_shape(
        seed: int) -> dict[str, Any]:
    p = build_default_tiny_substrate_v14(seed=int(seed) + 57001)
    ids = tokenize_bytes_v14("h410b", max_len=12)
    _, cache = forward_tiny_substrate_v14(p, ids)
    cfg = p.config.v13.v12.v11.v10.v9
    ok = (
        cache.multi_branch_rejoin_witness.shape == (
            cfg.n_layers, cfg.n_heads, cfg.max_len))
    return {
        "schema": R157_SCHEMA_VERSION,
        "name": "h410b_multi_branch_rejoin_shape",
        "passed": bool(ok),
    }


def family_h410c_silent_corruption_member_replace(
        seed: int) -> dict[str, Any]:
    p = build_default_tiny_substrate_v14(seed=int(seed) + 57002)
    ids = tokenize_bytes_v14("h410c", max_len=12)
    _, cache = forward_tiny_substrate_v14(p, ids)
    trigger_silent_corruption_v14(
        cache, role="planner", corrupted_bytes=2,
        member_replaced=True, detect_turn=3, repair_turn=-1)
    repaired = repair_silent_corruption_v14(
        cache, role="planner", repair_turn=5)
    entry = cache.silent_corruption_witness.get("planner")
    return {
        "schema": R157_SCHEMA_VERSION,
        "name": "h410c_silent_corruption_member_replace",
        "passed": bool(
            repaired and entry is not None
            and entry["member_replaced"]
            and entry["repair_turn"] == 5),
    }


def family_h410d_self_checksum_deterministic(
        seed: int) -> dict[str, Any]:
    p = build_default_tiny_substrate_v14(seed=int(seed) + 57003)
    ids = tokenize_bytes_v14("h410d", max_len=12)
    _, c1 = forward_tiny_substrate_v14(p, ids)
    _, c2 = forward_tiny_substrate_v14(p, ids)
    cid1 = substrate_self_checksum_v14(c1)
    cid2 = substrate_self_checksum_v14(c2)
    return {
        "schema": R157_SCHEMA_VERSION,
        "name": "h410d_self_checksum_deterministic",
        "passed": bool(cid1 == cid2),
    }


def family_h410e_v14_gate_score_shape(
        seed: int) -> dict[str, Any]:
    p = build_default_tiny_substrate_v14(seed=int(seed) + 57004)
    ids = tokenize_bytes_v14("h410e", max_len=12)
    t, _ = forward_tiny_substrate_v14(p, ids)
    cfg = p.config.v13.v12.v11.v10.v9
    return {
        "schema": R157_SCHEMA_VERSION,
        "name": "h410e_v14_gate_score_shape",
        "passed": bool(
            t.v14_gate_score_per_layer.shape == (cfg.n_layers,)
            and cfg.n_layers == 16),
    }


def family_h410f_multi_branch_rejoin_flops_saving(
        seed: int) -> dict[str, Any]:
    sav = substrate_multi_branch_rejoin_flops_v14(
        n_tokens=128, n_branches=4)
    return {
        "schema": R157_SCHEMA_VERSION,
        "name": "h410f_multi_branch_rejoin_flops_saving",
        "passed": bool(sav["saving_ratio"] >= 0.8),
        "ratio": float(sav["saving_ratio"]),
    }


def family_h410g_substrate_self_checksum_detect_rate(
        seed: int) -> dict[str, Any]:
    det = substrate_silent_corruption_detect_v14(
        single_byte_corruption_per_role=1, n_roles=6)
    return {
        "schema": R157_SCHEMA_VERSION,
        "name": "h410g_substrate_self_checksum_detect_rate",
        "passed": bool(det["passes_95pct_floor"]),
        "rate": float(det["single_byte_detect_rate"]),
    }


def family_h411_kv_v14_ten_target_ridge(
        seed: int) -> dict[str, Any]:
    p = build_default_tiny_substrate_v14(seed=int(seed) + 57100)
    cfg = p.config.v13.v12.v11.v10.v9
    d_head = cfg.d_model // cfg.n_heads
    b3 = KVBridgeV3Projection.init(
        n_layers=cfg.n_layers, n_heads=cfg.n_heads,
        n_kv_heads=cfg.n_kv_heads, n_inject_tokens=3,
        carrier_dim=8, d_head=d_head, seed=69300)
    b14 = KVBridgeV14Projection.init_from_v13(
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
                                                b3, seed_v4=69301),
                                            seed_v5=69302),
                                        seed_v6=69303),
                                    seed_v7=69304),
                                seed_v8=69305),
                            seed_v9=69306),
                        seed_v10=69307),
                    seed_v11=69308),
                seed_v12=69309),
            seed_v13=69310),
        seed_v14=69311)
    rng = _np.random.default_rng(int(seed) + 57200)
    ids = tokenize_bytes_v14("kvr157", max_len=4)
    while len(ids) < 4:
        ids.append(0)
    carriers = [list(rng.standard_normal(8)) for _ in range(4)]
    targets = []
    for i in range(10):
        t = _np.zeros(cfg.vocab_size)
        t[20 + 15 * i] = 1.0 / (i + 1)
        targets.append(t.tolist())
    _, rep = fit_kv_bridge_v14_ten_target(
        params=p, projection=b14,
        train_carriers=carriers,
        target_delta_logits_stack=targets,
        follow_up_token_ids=ids, n_directions=3)
    return {
        "schema": R157_SCHEMA_VERSION,
        "name": "h411_kv_v14_ten_target_ridge",
        "passed": bool(rep.n_targets == 10 and rep.converged),
    }


def family_h411b_silent_corruption_fingerprint_dim(
        seed: int) -> dict[str, Any]:
    fp = compute_silent_corruption_fingerprint_v14(
        role="planner", corrupted_bytes=2,
        member_replaced=True, task_id=str(seed),
        team_id="team")
    return {
        "schema": R157_SCHEMA_VERSION,
        "name": "h411b_silent_corruption_fingerprint_dim",
        "passed": bool(len(fp) == 60),
    }


def family_h411c_multi_branch_rejoin_falsifier(
        seed: int) -> dict[str, Any]:
    honest = (
        probe_kv_bridge_v14_multi_branch_rejoin_falsifier(
            multi_branch_rejoin_flag=0.5))
    return {
        "schema": R157_SCHEMA_VERSION,
        "name": "h411c_multi_branch_rejoin_falsifier",
        "passed": bool(honest.falsifier_score == 0.0),
    }


def family_h412_hsb_v13_hidden_vs_mbr(
        seed: int) -> dict[str, Any]:
    val = probe_hsb_v13_hidden_vs_multi_branch_rejoin(
        hidden_residual_l2=0.01,
        kv_residual_l2=0.2,
        prefix_residual_l2=0.2,
        replay_residual_l2=0.2,
        recover_residual_l2=0.2,
        branch_merge_residual_l2=0.2,
        agent_replacement_residual_l2=0.2,
        multi_branch_rejoin_residual_l2=0.5)
    return {
        "schema": R157_SCHEMA_VERSION,
        "name": "h412_hsb_v13_hidden_vs_mbr",
        "passed": bool(0.0 <= val <= 1.0 and val == 1.0),
    }


def family_h413_prefix_v13_k256_eight_way(
        seed: int) -> dict[str, Any]:
    comp = compare_prefix_v13_eight_way(
        prefix_l1_area=0.1,
        hidden_l1_area=0.2,
        replay_l1_area=0.3,
        team_l1_area=0.4,
        recover_l1_area=0.5,
        branch_l1_area=0.6,
        contradict_l1_area=0.7,
        multi_branch_rejoin_l1_area=0.05)
    return {
        "schema": R157_SCHEMA_VERSION,
        "name": "h413_prefix_v13_k256_eight_way",
        "passed": bool(
            comp["winner"] == "multi_branch_rejoin"
            and W69_DEFAULT_PREFIX_V13_K_STEPS == 256),
    }


def family_h414_cache_v12_nine_objective(
        seed: int) -> dict[str, Any]:
    cc12 = CacheControllerV12.init(fit_seed=int(seed) + 57400)
    rng = _np.random.default_rng(int(seed) + 57401)
    X = rng.standard_normal((12, 4))
    _, rep = fit_nine_objective_ridge_v12(
        controller=cc12, train_features=X.tolist(),
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
            X[:, 0] * 0.5 + X[:, 1] * 0.2).tolist())
    return {
        "schema": R157_SCHEMA_VERSION,
        "name": "h414_cache_v12_nine_objective",
        "passed": bool(rep.converged and rep.n_objectives == 9),
    }


def family_h415_replay_v10_routing_head(
        seed: int) -> dict[str, Any]:
    rc10 = ReplayControllerV10.init()
    cands = {
        r: [ReplayCandidate(
            100, 1000, 50, 0.1, 0.0, 0.3, True, True, 0)]
        for r in W69_REPLAY_REGIMES_V10}
    decs = {r: ["choose_reuse"] for r in W69_REPLAY_REGIMES_V10}
    rc10, _ = fit_replay_controller_v10_per_role(
        controller=rc10, role="planner",
        train_candidates_per_regime=cands,
        train_decisions_per_regime=decs)
    rng = _np.random.default_rng(int(seed) + 57500)
    X = rng.standard_normal((40, 11))
    labs = [
        W69_MULTI_BRANCH_REJOIN_ROUTING_LABELS[i % 7]
        for i in range(40)]
    _, rep = fit_replay_v10_multi_branch_rejoin_routing_head(
        controller=rc10, train_team_features=X.tolist(),
        train_routing_labels=labs)
    return {
        "schema": R157_SCHEMA_VERSION,
        "name": "h415_replay_v10_routing_head",
        "passed": bool(
            len(W69_REPLAY_REGIMES_V10) == 16
            and rep.post_classification_acc
                >= rep.pre_classification_acc),
    }


def family_h416_substrate_adapter_v14_full_tier(
        seed: int) -> dict[str, Any]:
    matrix = probe_all_v14_adapters()
    in_repo = matrix.by_name().get("tiny_substrate_v14")
    return {
        "schema": R157_SCHEMA_VERSION,
        "name": "h416_substrate_adapter_v14_full_tier",
        "passed": bool(
            in_repo is not None
            and in_repo.tier == W69_SUBSTRATE_TIER_SUBSTRATE_V14_FULL),
    }


def family_h417_ecc_v21_total_codes(
        seed: int) -> dict[str, Any]:
    cb = ECCCodebookV21.init(seed=int(seed) + 57600)
    fal = probe_ecc_v21_rate_floor_falsifier(codebook=cb)
    return {
        "schema": R157_SCHEMA_VERSION,
        "name": "h417_ecc_v21_total_codes",
        "passed": bool(
            cb.total_codes >= 2 ** 35
            and fal["target_exceeds_ceiling"]),
        "total_codes": int(cb.total_codes),
    }


def family_h418_crc_v17_detect_rate(
        seed: int) -> dict[str, Any]:
    crc = CorruptionRobustCarrierV17()
    w = emit_corruption_robustness_v17_witness(
        crc_v17=crc, n_probes=12, seed=int(seed) + 57700)
    return {
        "schema": R157_SCHEMA_VERSION,
        "name": "h418_crc_v17_detect_rate",
        "passed": bool(
            w.kv131072_corruption_detect_rate >= 0.95),
        "rate": float(w.kv131072_corruption_detect_rate),
    }


_R157_FAMILIES: tuple[Any, ...] = (
    family_h410_v14_substrate_determinism,
    family_h410b_multi_branch_rejoin_shape,
    family_h410c_silent_corruption_member_replace,
    family_h410d_self_checksum_deterministic,
    family_h410e_v14_gate_score_shape,
    family_h410f_multi_branch_rejoin_flops_saving,
    family_h410g_substrate_self_checksum_detect_rate,
    family_h411_kv_v14_ten_target_ridge,
    family_h411b_silent_corruption_fingerprint_dim,
    family_h411c_multi_branch_rejoin_falsifier,
    family_h412_hsb_v13_hidden_vs_mbr,
    family_h413_prefix_v13_k256_eight_way,
    family_h414_cache_v12_nine_objective,
    family_h415_replay_v10_routing_head,
    family_h416_substrate_adapter_v14_full_tier,
    family_h417_ecc_v21_total_codes,
    family_h418_crc_v17_detect_rate,
)


def run_r157(
        seeds: Sequence[int] = (199, 299, 399),
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for s in seeds:
        results: dict[str, Any] = {}
        for fn in _R157_FAMILIES:
            try:
                r = fn(int(s))
            except Exception as e:
                r = {
                    "schema": R157_SCHEMA_VERSION,
                    "name": getattr(fn, "__name__", "unknown"),
                    "passed": False,
                    "exception": str(e),
                }
            results[r["name"]] = r
        out.append({
            "schema": R157_SCHEMA_VERSION,
            "seed": int(s),
            "family_results": results,
        })
    return out


__all__ = ["R157_SCHEMA_VERSION", "run_r157"]
