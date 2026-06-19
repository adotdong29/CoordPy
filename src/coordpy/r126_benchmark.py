"""R-126 — W60 long-horizon retention / reconstruction / aggressive
cramming family.

H134..H145 cell families.

* H134  persistent V12 replay-EMA propagates through 64 turns
* H134b persistent V12 chain walk depth >= 64
* H134c persistent V12 distractor basis is non-degenerate (rank=4)
* H135  LHR V12 six-way runs without crashing
* H135b LHR V12 two-layer scorer fit residual ≤ pre-fit
* H135c LHR V12 max_k = 96
* H136  ECC V12 bits/visible-token ≥ 23
* H136b ECC V12 total codes = 2^21
* H136c ECC V12 2048-bit/token falsifier reproduces
* H137  multi-hop V10 chain-length 15 over 16 backends
* H137b multi-hop V10 replay axis used + compromise threshold > 0
* H138  TVS V9 ten arms sum to one
* H138b TVS V9 replay_controller_choice arm dominates when rp strict highest
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.r126_benchmark requires numpy") from exc

from .ecc_codebook_v12 import (
    ECCCodebookV12,
    compress_carrier_ecc_v12,
    probe_ecc_v12_rate_floor_falsifier,
)
from .ecc_codebook_v5 import (
    W53_DEFAULT_ECC_CODE_DIM,
    W53_DEFAULT_ECC_EMIT_MASK_LEN,
)
from .long_horizon_retention_v12 import (
    LongHorizonReconstructionV12Head,
    evaluate_lhr_v12_six_way,
    fit_lhr_v12_two_layer_retention_scorer,
)
from .multi_hop_translator_v10 import (
    W60_DEFAULT_MH_V10_BACKENDS,
    W60_DEFAULT_MH_V10_CHAIN_LEN,
    emit_multi_hop_v10_witness,
)
from .persistent_latent_v12 import (
    PersistentLatentStateV12Chain,
    V12StackedCell,
    emit_persistent_v12_witness,
    step_persistent_state_v12,
)
from .quantised_compression import QuantisedBudgetGate
from .transcript_vs_shared_arbiter_v9 import (
    emit_tvs_arbiter_v9_witness, ten_arm_compare,
)


R126_SCHEMA_VERSION: str = "coordpy.r126_benchmark.v1"


def family_h134_persistent_v12_replay_ema_propagates(
        seed: int) -> dict[str, Any]:
    cell = V12StackedCell.init(seed=int(seed) + 26000)
    chain = PersistentLatentStateV12Chain.empty()
    prev = None
    for t in range(64):
        rng = _np.random.default_rng(int(seed) + 26001 + t)
        carrier = list(rng.standard_normal(
            cell.state_dim).tolist())
        rep = list(rng.standard_normal(
            cell.state_dim).tolist())
        prev = step_persistent_state_v12(
            cell=cell, prev_state=prev,
            carrier_values=carrier,
            turn_index=t, role="r",
            replay_skip=rep, replay_fidelity=0.85)
        chain.add(prev)
    w = emit_persistent_v12_witness(
        cell=cell, chain=chain, leaf_cid=prev.cid())
    return {
        "schema": R126_SCHEMA_VERSION,
        "name": "h134_persistent_v12_replay_ema_propagates",
        "passed": bool(
            len(prev.replay_carrier)
            == cell.state_dim
            and any(abs(float(x)) > 0.0
                     for x in prev.replay_carrier)),
        "replay_carrier_l2": float(
            sum(float(x) ** 2
                for x in prev.replay_carrier) ** 0.5),
        "n_layers": int(prev.n_layers),
        "achieved_walk": int(w.achieved_chain_walk_depth),
    }


def family_h134b_persistent_v12_chain_walk_depth(
        seed: int) -> dict[str, Any]:
    cell = V12StackedCell.init(seed=int(seed) + 26100)
    chain = PersistentLatentStateV12Chain.empty()
    prev = None
    for t in range(64):
        carrier = [0.1 * (t + 1)] * cell.state_dim
        prev = step_persistent_state_v12(
            cell=cell, prev_state=prev,
            carrier_values=carrier,
            turn_index=t, role="r")
        chain.add(prev)
    w = emit_persistent_v12_witness(
        cell=cell, chain=chain, leaf_cid=prev.cid())
    return {
        "schema": R126_SCHEMA_VERSION,
        "name": "h134b_persistent_v12_chain_walk_depth",
        "passed": bool(w.achieved_chain_walk_depth >= 64),
        "achieved": int(w.achieved_chain_walk_depth),
        "n_layers": int(w.n_layers),
    }


def family_h134c_persistent_v12_distractor_basis(
        seed: int) -> dict[str, Any]:
    cell = V12StackedCell.init(seed=int(seed) + 26150)
    chain = PersistentLatentStateV12Chain.empty()
    prev = step_persistent_state_v12(
        cell=cell, prev_state=None,
        carrier_values=[0.1] * cell.state_dim,
        turn_index=0, role="r")
    chain.add(prev)
    w = emit_persistent_v12_witness(
        cell=cell, chain=chain, leaf_cid=prev.cid())
    return {
        "schema": R126_SCHEMA_VERSION,
        "name": "h134c_persistent_v12_distractor_basis",
        "passed": bool(w.distractor_basis_rank == 4),
        "distractor_rank": int(w.distractor_basis_rank),
    }


def family_h135_lhr_v12_six_way_runs(
        seed: int) -> dict[str, Any]:
    head = LongHorizonReconstructionV12Head.init(
        seed=int(seed) + 26200)
    rng = _np.random.default_rng(int(seed) + 26201)
    n = 12
    carriers = [list(rng.standard_normal(8))
                 for _ in range(n)]
    targets = [list(rng.standard_normal(head.out_dim))
                for _ in range(n)]
    subs = [list(rng.standard_normal(32))
             for _ in range(n)]
    hids = [list(rng.standard_normal(32))
             for _ in range(n)]
    atts = [list(rng.standard_normal(32))
             for _ in range(n)]
    rets = [list(rng.standard_normal(32))
             for _ in range(n)]
    reps = [list(rng.standard_normal(16))
             for _ in range(n)]
    rep = evaluate_lhr_v12_six_way(
        head, carrier_examples=carriers,
        target_examples=targets,
        substrate_states=subs,
        hidden_states=hids,
        attention_states=atts,
        retrieval_states=rets,
        replay_states=reps, k=16)
    return {
        "schema": R126_SCHEMA_VERSION,
        "name": "h135_lhr_v12_six_way_runs",
        "passed": bool(rep["n"] == n
                        and rep["replay_mse"] >= 0.0),
        "proxy_mse": float(rep["proxy_mse"]),
        "substrate_mse": float(rep["substrate_mse"]),
        "hidden_mse": float(rep["hidden_state_mse"]),
        "attention_mse": float(rep["attention_mse"]),
        "retrieval_mse": float(rep["retrieval_mse"]),
        "replay_mse": float(rep["replay_mse"]),
        "n": int(rep["n"]),
    }


def family_h135b_lhr_v12_two_layer_scorer_fit(
        seed: int) -> dict[str, Any]:
    head = LongHorizonReconstructionV12Head.init(
        seed=int(seed) + 26300)
    rng = _np.random.default_rng(int(seed) + 26301)
    n = 32
    d = 8
    X = rng.standard_normal((n, d))
    true_w = rng.standard_normal(d)
    y = X @ true_w + 0.1 * rng.standard_normal(n)
    head, post_residual = fit_lhr_v12_two_layer_retention_scorer(
        head, train_carriers=X.tolist(),
        train_targets=y.tolist(), seed=int(seed) + 26302)
    pre_residual = float(_np.mean(_np.abs(y)))
    return {
        "schema": R126_SCHEMA_VERSION,
        "name": "h135b_lhr_v12_two_layer_scorer_fit",
        "passed": bool(post_residual < pre_residual + 1e-9),
        "pre_residual": float(pre_residual),
        "post_residual": float(post_residual),
        "scorer_fitted": bool(
            head.scorer_layer2 is not None),
    }


def family_h135c_lhr_v12_max_k(
        seed: int) -> dict[str, Any]:
    head = LongHorizonReconstructionV12Head.init()
    return {
        "schema": R126_SCHEMA_VERSION,
        "name": "h135c_lhr_v12_max_k",
        "passed": bool(int(head.max_k) == 96),
        "max_k": int(head.max_k),
    }


def family_h136_ecc_v12_bits_per_token(
        seed: int) -> dict[str, Any]:
    cb = ECCCodebookV12.init(seed=int(seed) + 26400)
    gate = QuantisedBudgetGate.init(
        in_dim=W53_DEFAULT_ECC_CODE_DIM,
        emit_mask_len=W53_DEFAULT_ECC_EMIT_MASK_LEN,
        seed=int(seed) + 26401)
    gate.importance_threshold = 0.0
    gate.w_emit.values = [1.0] * len(gate.w_emit.values)
    rng = _np.random.default_rng(int(seed) + 26402)
    carrier = list(rng.standard_normal(
        W53_DEFAULT_ECC_CODE_DIM).tolist())
    comp = compress_carrier_ecc_v12(
        carrier, codebook=cb, gate=gate)
    return {
        "schema": R126_SCHEMA_VERSION,
        "name": "h136_ecc_v12_bits_per_token",
        "passed": bool(
            float(comp["bits_per_visible_token"]) >= 23.0),
        "bits_per_visible_token": float(
            comp["bits_per_visible_token"]),
        "visible_tokens": int(comp["visible_tokens"]),
    }


def family_h136b_ecc_v12_total_codes(
        seed: int) -> dict[str, Any]:
    cb = ECCCodebookV12.init()
    return {
        "schema": R126_SCHEMA_VERSION,
        "name": "h136b_ecc_v12_total_codes",
        "passed": bool(int(cb.total_codes) == 2097152),
        "total_codes": int(cb.total_codes),
        "target": 2097152,
    }


def family_h136c_ecc_v12_falsifier(
        seed: int) -> dict[str, Any]:
    cb = ECCCodebookV12.init()
    fal = probe_ecc_v12_rate_floor_falsifier(codebook=cb)
    return {
        "schema": R126_SCHEMA_VERSION,
        "name": "h136c_ecc_v12_falsifier",
        "passed": bool(fal["reproduces_cap"]),
        "target_bits_per_token": float(
            fal["target_bits_per_token"]),
        "info_bound": float(fal["info_bound"]),
        "above_bound": bool(
            fal["target_above_info_bound"]),
    }


def family_h137_multi_hop_v10_chain(
        seed: int) -> dict[str, Any]:
    w = emit_multi_hop_v10_witness(
        backends=W60_DEFAULT_MH_V10_BACKENDS,
        chain_length=W60_DEFAULT_MH_V10_CHAIN_LEN,
        seed=int(seed) + 26500)
    return {
        "schema": R126_SCHEMA_VERSION,
        "name": "h137_multi_hop_v10_chain",
        "passed": bool(
            w.chain_length == 15
            and w.n_edges == 16 * 15
            and len(w.backends) == 16),
        "chain_length": int(w.chain_length),
        "n_edges": int(w.n_edges),
        "n_backends": int(len(w.backends)),
        "arbitration_kind": str(w.arbitration_kind),
    }


def family_h137b_multi_hop_v10_replay_axis(
        seed: int) -> dict[str, Any]:
    w = emit_multi_hop_v10_witness(seed=int(seed) + 26600)
    return {
        "schema": R126_SCHEMA_VERSION,
        "name": "h137b_multi_hop_v10_replay_axis",
        "passed": bool(
            w.replay_axis_used
            and w.compromise_threshold >= 1),
        "replay_axis_used": bool(w.replay_axis_used),
        "retrieval_axis_used": bool(
            w.retrieval_axis_used),
        "composite_trust_used": bool(
            w.composite_trust_used),
        "compromise_threshold": int(
            w.compromise_threshold),
    }


def family_h138_tvs_v9_ten_arms_sum_to_one(
        seed: int) -> dict[str, Any]:
    rng = _np.random.default_rng(int(seed) + 26700)
    n = 20
    res = ten_arm_compare(
        per_turn_confidences=rng.uniform(
            0.0, 1.0, n).tolist(),
        per_turn_trust_scores=rng.uniform(
            0.0, 1.0, n).tolist(),
        per_turn_merge_retentions=rng.uniform(
            0.0, 1.0, n).tolist(),
        per_turn_tw_retentions=rng.uniform(
            0.0, 1.0, n).tolist(),
        per_turn_substrate_fidelities=rng.uniform(
            0.0, 1.0, n).tolist(),
        per_turn_hidden_fidelities=rng.uniform(
            0.0, 1.0, n).tolist(),
        per_turn_cache_fidelities=rng.uniform(
            0.0, 1.0, n).tolist(),
        per_turn_retrieval_fidelities=rng.uniform(
            0.0, 1.0, n).tolist(),
        per_turn_replay_fidelities=rng.uniform(
            0.0, 1.0, n).tolist())
    s = float(sum(res.pick_rates.values()))
    w = emit_tvs_arbiter_v9_witness(result=res)
    return {
        "schema": R126_SCHEMA_VERSION,
        "name": "h138_tvs_v9_ten_arms_sum_to_one",
        "passed": bool(
            w.pick_rates_sum_to_one
            and w.n_arms == 10),
        "n_arms": int(w.n_arms),
        "pick_rate_sum": float(s),
        "replay_used": bool(w.replay_controller_used),
    }


def family_h138b_tvs_v9_replay_dominates(
        seed: int) -> dict[str, Any]:
    n = 16
    res = ten_arm_compare(
        per_turn_confidences=[0.3] * n,
        per_turn_trust_scores=[0.3] * n,
        per_turn_merge_retentions=[0.3] * n,
        per_turn_tw_retentions=[0.3] * n,
        per_turn_substrate_fidelities=[0.4] * n,
        per_turn_hidden_fidelities=[0.4] * n,
        per_turn_cache_fidelities=[0.4] * n,
        per_turn_retrieval_fidelities=[0.4] * n,
        per_turn_replay_fidelities=[0.95] * n)
    rate_replay = float(
        res.pick_rates["replay_controller_choice"])
    return {
        "schema": R126_SCHEMA_VERSION,
        "name": "h138b_tvs_v9_replay_dominates",
        "passed": bool(rate_replay >= 0.9),
        "replay_rate": float(rate_replay),
        "all_rates": {
            k: float(v) for k, v in res.pick_rates.items()},
    }


def run_r126(seeds: Sequence[int] = (194, 294, 394)
              ) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    families = [
        family_h134_persistent_v12_replay_ema_propagates,
        family_h134b_persistent_v12_chain_walk_depth,
        family_h134c_persistent_v12_distractor_basis,
        family_h135_lhr_v12_six_way_runs,
        family_h135b_lhr_v12_two_layer_scorer_fit,
        family_h135c_lhr_v12_max_k,
        family_h136_ecc_v12_bits_per_token,
        family_h136b_ecc_v12_total_codes,
        family_h136c_ecc_v12_falsifier,
        family_h137_multi_hop_v10_chain,
        family_h137b_multi_hop_v10_replay_axis,
        family_h138_tvs_v9_ten_arms_sum_to_one,
        family_h138b_tvs_v9_replay_dominates,
    ]
    for s in seeds:
        per_family: dict[str, dict[str, Any]] = {}
        for fam in families:
            res = fam(int(s))
            per_family[res["name"]] = res
        out.append({"seed": int(s),
                     "family_results": per_family})
    return out


__all__ = [
    "R126_SCHEMA_VERSION",
    "family_h134_persistent_v12_replay_ema_propagates",
    "family_h134b_persistent_v12_chain_walk_depth",
    "family_h134c_persistent_v12_distractor_basis",
    "family_h135_lhr_v12_six_way_runs",
    "family_h135b_lhr_v12_two_layer_scorer_fit",
    "family_h135c_lhr_v12_max_k",
    "family_h136_ecc_v12_bits_per_token",
    "family_h136b_ecc_v12_total_codes",
    "family_h136c_ecc_v12_falsifier",
    "family_h137_multi_hop_v10_chain",
    "family_h137b_multi_hop_v10_replay_axis",
    "family_h138_tvs_v9_ten_arms_sum_to_one",
    "family_h138b_tvs_v9_replay_dominates",
    "run_r126",
]
