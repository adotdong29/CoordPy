"""R-123 — W59 long-horizon retention / reconstruction / aggressive
cramming family.

H115..H125 cell families. R-123 is the W59 *retention/cramming*
family:

* H115  persistent V11 retrieval-EMA propagates through 32 turns
* H115b persistent V11 chain walk depth ≥ 32
* H116  LHR V11 five-way runs without crashing
* H116b LHR V11 retention scorer fit residual ≤ pre-fit
* H116c LHR V11 max_k = 80
* H117  ECC V11 bits/visible-token ≥ 22
* H117b ECC V11 total codes = 2^20
* H117c ECC V11 1024-bit/token falsifier reproduces
* H118  multi-hop V9 chain-length 13 over 14 backends
* H118b multi-hop V9 retrieval axis used
* H119  TVS V8 nine arms sum to one
* H119b TVS V8 retrieval_replay arm dominates when rf strict highest
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.r123_benchmark requires numpy") from exc

from .ecc_codebook_v11 import (
    ECCCodebookV11,
    compress_carrier_ecc_v11,
    probe_ecc_v11_rate_floor_falsifier,
)
from .ecc_codebook_v5 import (
    W53_DEFAULT_ECC_CODE_DIM, W53_DEFAULT_ECC_EMIT_MASK_LEN,
)
from .long_horizon_retention_v11 import (
    LongHorizonReconstructionV11Head,
    evaluate_lhr_v11_five_way,
    fit_lhr_v11_retention_scorer,
)
from .multi_hop_translator_v9 import (
    W59_DEFAULT_MH_V9_BACKENDS,
    W59_DEFAULT_MH_V9_CHAIN_LEN,
    emit_multi_hop_v9_witness,
)
from .persistent_latent_v11 import (
    PersistentLatentStateV11Chain,
    V11StackedCell,
    emit_persistent_v11_witness,
    step_persistent_state_v11,
)
from .quantised_compression import QuantisedBudgetGate
from .transcript_vs_shared_arbiter_v8 import (
    nine_arm_compare, emit_tvs_arbiter_v8_witness,
)


R123_SCHEMA_VERSION: str = "coordpy.r123_benchmark.v1"


def family_h115_persistent_v11_retrieval_ema_propagates(
        seed: int) -> dict[str, Any]:
    cell = V11StackedCell.init(seed=int(seed) + 23000)
    chain = PersistentLatentStateV11Chain.empty()
    prev = None
    for t in range(32):
        rng = _np.random.default_rng(int(seed) + 23001 + t)
        carrier = list(rng.standard_normal(
            cell.state_dim).tolist())
        ret = list(rng.standard_normal(
            cell.state_dim).tolist())
        state = step_persistent_state_v11(
            cell=cell, prev_state=prev,
            carrier_values=carrier,
            turn_index=t, role="r",
            retrieval_skip=ret,
            retrieval_fidelity=0.85)
        chain.add(state)
        prev = state
    w = emit_persistent_v11_witness(
        cell=cell, chain=chain, leaf_cid=prev.cid())
    return {
        "schema": R123_SCHEMA_VERSION,
        "name": "h115_persistent_v11_retrieval_ema_propagates",
        "passed": bool(
            len(prev.retrieval_carrier)
            == cell.state_dim
            and any(abs(float(x)) > 0.0
                     for x in prev.retrieval_carrier)),
        "retrieval_carrier_l2": float(
            sum(float(x) ** 2
                for x in prev.retrieval_carrier) ** 0.5),
        "n_layers": int(prev.n_layers),
        "achieved_walk": int(w.achieved_chain_walk_depth),
    }


def family_h115b_persistent_v11_chain_walk_depth(
        seed: int) -> dict[str, Any]:
    cell = V11StackedCell.init(seed=int(seed) + 23100)
    chain = PersistentLatentStateV11Chain.empty()
    prev = None
    for t in range(32):
        carrier = [0.1 * (t + 1)] * cell.state_dim
        prev = step_persistent_state_v11(
            cell=cell, prev_state=prev,
            carrier_values=carrier,
            turn_index=t, role="r")
        chain.add(prev)
    w = emit_persistent_v11_witness(
        cell=cell, chain=chain, leaf_cid=prev.cid())
    return {
        "schema": R123_SCHEMA_VERSION,
        "name": "h115b_persistent_v11_chain_walk_depth",
        "passed": bool(w.achieved_chain_walk_depth >= 32),
        "achieved": int(w.achieved_chain_walk_depth),
        "n_layers": int(w.n_layers),
    }


def family_h116_lhr_v11_five_way_runs(
        seed: int) -> dict[str, Any]:
    head = LongHorizonReconstructionV11Head.init(
        seed=int(seed) + 23200)
    rng = _np.random.default_rng(int(seed) + 23201)
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
    rep = evaluate_lhr_v11_five_way(
        head, carrier_examples=carriers,
        target_examples=targets,
        substrate_states=subs,
        hidden_states=hids,
        attention_states=atts,
        retrieval_states=rets, k=16)
    return {
        "schema": R123_SCHEMA_VERSION,
        "name": "h116_lhr_v11_five_way_runs",
        "passed": bool(rep["n"] == n
                        and rep["retrieval_mse"] >= 0.0),
        "proxy_mse": float(rep["proxy_mse"]),
        "substrate_mse": float(rep["substrate_mse"]),
        "hidden_mse": float(rep["hidden_state_mse"]),
        "attention_mse": float(rep["attention_mse"]),
        "retrieval_mse": float(rep["retrieval_mse"]),
        "n": int(rep["n"]),
    }


def family_h116b_lhr_v11_retention_scorer_fit(
        seed: int) -> dict[str, Any]:
    head = LongHorizonReconstructionV11Head.init(
        seed=int(seed) + 23300)
    rng = _np.random.default_rng(int(seed) + 23301)
    n = 32
    d = 8
    X = rng.standard_normal((n, d))
    true_w = rng.standard_normal(d)
    y = X @ true_w + 0.1 * rng.standard_normal(n)
    head, post_residual = fit_lhr_v11_retention_scorer(
        head, train_carriers=X.tolist(),
        train_targets=y.tolist())
    pre_residual = float(_np.mean(_np.abs(y)))
    return {
        "schema": R123_SCHEMA_VERSION,
        "name": "h116b_lhr_v11_retention_scorer_fit",
        "passed": bool(post_residual < pre_residual + 1e-9),
        "pre_residual": float(pre_residual),
        "post_residual": float(post_residual),
        "scorer_fitted": bool(
            head.retention_scorer is not None),
    }


def family_h116c_lhr_v11_max_k(
        seed: int) -> dict[str, Any]:
    head = LongHorizonReconstructionV11Head.init()
    return {
        "schema": R123_SCHEMA_VERSION,
        "name": "h116c_lhr_v11_max_k",
        "passed": bool(int(head.max_k) == 80),
        "max_k": int(head.max_k),
    }


def family_h117_ecc_v11_bits_per_token(
        seed: int) -> dict[str, Any]:
    cb = ECCCodebookV11.init(seed=int(seed) + 23400)
    gate = QuantisedBudgetGate.init(
        in_dim=W53_DEFAULT_ECC_CODE_DIM,
        emit_mask_len=W53_DEFAULT_ECC_EMIT_MASK_LEN,
        seed=int(seed) + 23401)
    gate.importance_threshold = 0.0
    gate.w_emit.values = [1.0] * len(gate.w_emit.values)
    rng = _np.random.default_rng(int(seed) + 23402)
    carrier = list(rng.standard_normal(
        W53_DEFAULT_ECC_CODE_DIM).tolist())
    comp = compress_carrier_ecc_v11(
        carrier, codebook=cb, gate=gate)
    return {
        "schema": R123_SCHEMA_VERSION,
        "name": "h117_ecc_v11_bits_per_token",
        "passed": bool(
            float(comp["bits_per_visible_token"]) >= 22.0),
        "bits_per_visible_token": float(
            comp["bits_per_visible_token"]),
        "visible_tokens": int(comp["visible_tokens"]),
    }


def family_h117b_ecc_v11_total_codes(
        seed: int) -> dict[str, Any]:
    cb = ECCCodebookV11.init()
    return {
        "schema": R123_SCHEMA_VERSION,
        "name": "h117b_ecc_v11_total_codes",
        "passed": bool(int(cb.total_codes) == 1048576),
        "total_codes": int(cb.total_codes),
        "target": 1048576,
    }


def family_h117c_ecc_v11_falsifier(
        seed: int) -> dict[str, Any]:
    cb = ECCCodebookV11.init()
    fal = probe_ecc_v11_rate_floor_falsifier(codebook=cb)
    return {
        "schema": R123_SCHEMA_VERSION,
        "name": "h117c_ecc_v11_falsifier",
        "passed": bool(fal["reproduces_cap"]),
        "target_bits_per_token": float(
            fal["target_bits_per_token"]),
        "info_bound": float(fal["info_bound"]),
        "above_bound": bool(fal["target_above_info_bound"]),
    }


def family_h118_multi_hop_v9_chain(
        seed: int) -> dict[str, Any]:
    w = emit_multi_hop_v9_witness(
        backends=W59_DEFAULT_MH_V9_BACKENDS,
        chain_length=W59_DEFAULT_MH_V9_CHAIN_LEN,
        seed=int(seed) + 23500)
    return {
        "schema": R123_SCHEMA_VERSION,
        "name": "h118_multi_hop_v9_chain",
        "passed": bool(
            w.chain_length == 13
            and w.n_edges == 14 * 13
            and len(w.backends) == 14),
        "chain_length": int(w.chain_length),
        "n_edges": int(w.n_edges),
        "n_backends": int(len(w.backends)),
        "arbitration_kind": str(w.arbitration_kind),
    }


def family_h118b_multi_hop_v9_retrieval_axis(
        seed: int) -> dict[str, Any]:
    w = emit_multi_hop_v9_witness(seed=int(seed) + 23600)
    return {
        "schema": R123_SCHEMA_VERSION,
        "name": "h118b_multi_hop_v9_retrieval_axis",
        "passed": bool(w.retrieval_axis_used),
        "retrieval_axis_used": bool(w.retrieval_axis_used),
        "composite_trust_used": bool(w.composite_trust_used),
    }


def family_h119_tvs_v8_nine_arms_sum_to_one(
        seed: int) -> dict[str, Any]:
    rng = _np.random.default_rng(int(seed) + 23700)
    n_turns = 20
    res = nine_arm_compare(
        per_turn_confidences=rng.uniform(
            0.0, 1.0, n_turns).tolist(),
        per_turn_trust_scores=rng.uniform(
            0.0, 1.0, n_turns).tolist(),
        per_turn_merge_retentions=rng.uniform(
            0.0, 1.0, n_turns).tolist(),
        per_turn_tw_retentions=rng.uniform(
            0.0, 1.0, n_turns).tolist(),
        per_turn_substrate_fidelities=rng.uniform(
            0.0, 1.0, n_turns).tolist(),
        per_turn_hidden_fidelities=rng.uniform(
            0.0, 1.0, n_turns).tolist(),
        per_turn_cache_fidelities=rng.uniform(
            0.0, 1.0, n_turns).tolist(),
        per_turn_retrieval_fidelities=rng.uniform(
            0.0, 1.0, n_turns).tolist())
    s = float(sum(res.pick_rates.values()))
    w = emit_tvs_arbiter_v8_witness(result=res)
    return {
        "schema": R123_SCHEMA_VERSION,
        "name": "h119_tvs_v8_nine_arms_sum_to_one",
        "passed": bool(
            w.pick_rates_sum_to_one
            and w.n_arms == 9),
        "n_arms": int(w.n_arms),
        "pick_rate_sum": float(s),
        "retrieval_used": bool(w.retrieval_used),
    }


def family_h119b_tvs_v8_retrieval_dominates(
        seed: int) -> dict[str, Any]:
    n_turns = 16
    # Retrieval fidelity strictly highest each turn.
    rng = _np.random.default_rng(int(seed) + 23800)
    res = nine_arm_compare(
        per_turn_confidences=[0.3] * n_turns,
        per_turn_trust_scores=[0.3] * n_turns,
        per_turn_merge_retentions=[0.3] * n_turns,
        per_turn_tw_retentions=[0.3] * n_turns,
        per_turn_substrate_fidelities=[0.4] * n_turns,
        per_turn_hidden_fidelities=[0.4] * n_turns,
        per_turn_cache_fidelities=[0.4] * n_turns,
        per_turn_retrieval_fidelities=[0.9] * n_turns)
    rate_retrieval = float(
        res.pick_rates["retrieval_replay"])
    return {
        "schema": R123_SCHEMA_VERSION,
        "name": "h119b_tvs_v8_retrieval_dominates",
        "passed": bool(rate_retrieval >= 0.9),
        "retrieval_rate": float(rate_retrieval),
        "all_rates": {
            k: float(v) for k, v in res.pick_rates.items()},
    }


def run_r123(seeds: Sequence[int] = (191, 291, 391)
              ) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    families = [
        family_h115_persistent_v11_retrieval_ema_propagates,
        family_h115b_persistent_v11_chain_walk_depth,
        family_h116_lhr_v11_five_way_runs,
        family_h116b_lhr_v11_retention_scorer_fit,
        family_h116c_lhr_v11_max_k,
        family_h117_ecc_v11_bits_per_token,
        family_h117b_ecc_v11_total_codes,
        family_h117c_ecc_v11_falsifier,
        family_h118_multi_hop_v9_chain,
        family_h118b_multi_hop_v9_retrieval_axis,
        family_h119_tvs_v8_nine_arms_sum_to_one,
        family_h119b_tvs_v8_retrieval_dominates,
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
    "R123_SCHEMA_VERSION",
    "family_h115_persistent_v11_retrieval_ema_propagates",
    "family_h115b_persistent_v11_chain_walk_depth",
    "family_h116_lhr_v11_five_way_runs",
    "family_h116b_lhr_v11_retention_scorer_fit",
    "family_h116c_lhr_v11_max_k",
    "family_h117_ecc_v11_bits_per_token",
    "family_h117b_ecc_v11_total_codes",
    "family_h117c_ecc_v11_falsifier",
    "family_h118_multi_hop_v9_chain",
    "family_h118b_multi_hop_v9_retrieval_axis",
    "family_h119_tvs_v8_nine_arms_sum_to_one",
    "family_h119b_tvs_v8_retrieval_dominates",
    "run_r123",
]
