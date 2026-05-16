"""W66 R-147 benchmark family — 8192-turn retention / V18 LHR /
V16 multi-hop / V18 ECC / persistent V18.

H253..H259a cell families (14 H-bars):

* H253   persistent V18 fifteenth-skip carrier present
* H253b  persistent V18 max_chain_walk_depth >= 8192
* H253c  persistent V18 17 layers
* H253d  persistent V18 chain walk preserves CIDs
* H254   LHR V18 17 heads + max_k=320
* H254b  LHR V18 seventeen_way_value runs on team-failure-
         recovery indicator
* H254c  LHR V18 eight-layer scorer ridge fit converges
* H255   multi-hop V16 36 backends + chain-length 26
* H255b  multi-hop V16 eleven-axis arbitration kind correct
* H255c  multi-hop V16 compromise threshold in [1, 11]
* H256   ECC V18 2^29 codes
* H256b  ECC V18 bits/visible-token >= 31.0 at full emit
* H256c  ECC V18 rate-floor falsifier triggers
* H259a  ECC V18 K17 meta15 index well-formed
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as _np

from coordpy.ecc_codebook_v18 import (
    ECCCodebookV18,
    compress_carrier_ecc_v18,
    probe_ecc_v18_rate_floor_falsifier,
)
from coordpy.ecc_codebook_v5 import (
    W53_DEFAULT_ECC_CODE_DIM, W53_DEFAULT_ECC_EMIT_MASK_LEN,
)
from coordpy.long_horizon_retention_v18 import (
    LongHorizonReconstructionV18Head,
    W66_DEFAULT_LHR_V18_MAX_K,
    W66_DEFAULT_LHR_V18_SWISH_PROJ_DIM,
    fit_lhr_v18_eight_layer_scorer,
)
from coordpy.multi_hop_translator_v16 import (
    W66_DEFAULT_MH_V16_BACKENDS,
    W66_DEFAULT_MH_V16_CHAIN_LEN,
    evaluate_dec_chain_len26_fidelity,
    emit_multi_hop_v16_witness,
)
from coordpy.persistent_latent_v12 import V12StackedCell
from coordpy.persistent_latent_v18 import (
    PersistentLatentStateV18Chain,
    W66_DEFAULT_V18_MAX_CHAIN_WALK_DEPTH,
    W66_DEFAULT_V18_N_LAYERS,
    emit_persistent_v18_witness,
    step_persistent_state_v18,
)
from coordpy.quantised_compression import QuantisedBudgetGate


R147_SCHEMA_VERSION: str = "coordpy.r147_benchmark.v1"


def family_h253_persistent_v18_fifteenth_skip(
        seed: int) -> dict[str, Any]:
    cell = V12StackedCell.init(seed=int(seed) + 47000)
    chain = PersistentLatentStateV18Chain.empty()
    sd = int(cell.state_dim)
    state = step_persistent_state_v18(
        cell=cell, prev_state=None,
        carrier_values=[0.1] * sd, turn_index=0, role="r",
        team_failure_recovery_skip_v18=[0.5] * sd,
        substrate_fidelity=0.9, attention_fidelity=0.9,
        retrieval_fidelity=0.9, replay_fidelity=0.9)
    chain.add(state)
    w = emit_persistent_v18_witness(chain, state.cid())
    return {
        "schema": R147_SCHEMA_VERSION,
        "name": "h253_persistent_v18_fifteenth_skip",
        "passed": bool(w.fifteenth_skip_present),
        "carrier_l1": float(
            w.team_failure_recovery_carrier_l1_sum),
    }


def family_h253b_persistent_v18_chain_walk_depth(
        seed: int) -> dict[str, Any]:
    return {
        "schema": R147_SCHEMA_VERSION,
        "name": "h253b_persistent_v18_chain_walk_depth",
        "passed": bool(
            int(W66_DEFAULT_V18_MAX_CHAIN_WALK_DEPTH) >= 8192),
        "depth": int(W66_DEFAULT_V18_MAX_CHAIN_WALK_DEPTH),
    }


def family_h253c_persistent_v18_seventeen_layers(
        seed: int) -> dict[str, Any]:
    return {
        "schema": R147_SCHEMA_VERSION,
        "name": "h253c_persistent_v18_seventeen_layers",
        "passed": bool(int(W66_DEFAULT_V18_N_LAYERS) == 17),
        "n_layers": int(W66_DEFAULT_V18_N_LAYERS),
    }


def family_h253d_persistent_v18_chain_preserves_cids(
        seed: int) -> dict[str, Any]:
    cell = V12StackedCell.init(seed=int(seed) + 47100)
    chain = PersistentLatentStateV18Chain.empty()
    sd = int(cell.state_dim)
    state_a = step_persistent_state_v18(
        cell=cell, prev_state=None,
        carrier_values=[0.1] * sd, turn_index=0, role="r",
        team_failure_recovery_skip_v18=[0.4] * sd,
        substrate_fidelity=0.9, attention_fidelity=0.9,
        retrieval_fidelity=0.9, replay_fidelity=0.9)
    chain.add(state_a)
    state_b = step_persistent_state_v18(
        cell=cell, prev_state=state_a,
        carrier_values=[0.2] * sd, turn_index=1, role="r",
        team_failure_recovery_skip_v18=[0.5] * sd,
        substrate_fidelity=0.9, attention_fidelity=0.9,
        retrieval_fidelity=0.9, replay_fidelity=0.9)
    chain.add(state_b)
    return {
        "schema": R147_SCHEMA_VERSION,
        "name": "h253d_persistent_v18_chain_preserves_cids",
        "passed": bool(
            state_a.cid() != state_b.cid()
            and state_a.cid() in chain.states
            and state_b.cid() in chain.states),
        "n_states": int(len(chain.states)),
    }


def family_h254_lhr_v18_heads_and_max_k(
        seed: int) -> dict[str, Any]:
    head = LongHorizonReconstructionV18Head.init(
        seed=int(seed) + 47200)
    return {
        "schema": R147_SCHEMA_VERSION,
        "name": "h254_lhr_v18_heads_and_max_k",
        "passed": bool(
            int(head.max_k) == int(W66_DEFAULT_LHR_V18_MAX_K)
            and int(W66_DEFAULT_LHR_V18_MAX_K) == 320),
        "max_k": int(head.max_k),
    }


def family_h254b_lhr_v18_seventeen_way_value(
        seed: int) -> dict[str, Any]:
    head = LongHorizonReconstructionV18Head.init(
        seed=int(seed) + 47300)
    out = head.seventeen_way_value(
        carrier=[0.1] * 8, k=4,
        team_failure_recovery_indicator=[0.5] * 8,
        team_task_success_indicator=[0.5] * 8,
        replay_dominance_indicator=[0.5] * 8,
        hidden_wins_indicator=[0.5] * 8,
        replay_dominance_primary_indicator=[0.6] * 8)
    out_arr = _np.asarray(out, dtype=_np.float64)
    return {
        "schema": R147_SCHEMA_VERSION,
        "name": "h254b_lhr_v18_seventeen_way_value",
        "passed": bool(
            out_arr.size == head.out_dim
            and bool(_np.all(_np.isfinite(out_arr)))),
        "out_dim": int(out_arr.size),
    }


def family_h254c_lhr_v18_eight_layer_scorer(
        seed: int) -> dict[str, Any]:
    head = LongHorizonReconstructionV18Head.init(
        seed=int(seed) + 47400)
    rng = _np.random.default_rng(int(seed) + 47500)
    n = 30
    X = rng.standard_normal(
        (n, int(W66_DEFAULT_LHR_V18_SWISH_PROJ_DIM)))
    y = X.sum(axis=-1).tolist()
    _, audit = fit_lhr_v18_eight_layer_scorer(
        head=head, train_features=X.tolist(),
        train_targets=y)
    return {
        "schema": R147_SCHEMA_VERSION,
        "name": "h254c_lhr_v18_eight_layer_scorer",
        "passed": bool(audit.get("converged", False)),
        "post": float(audit.get("post_fit_residual", -1.0)),
    }


def family_h255_multi_hop_v16_backends(
        seed: int) -> dict[str, Any]:
    return {
        "schema": R147_SCHEMA_VERSION,
        "name": "h255_multi_hop_v16_backends",
        "passed": bool(
            int(len(W66_DEFAULT_MH_V16_BACKENDS)) == 36
            and int(W66_DEFAULT_MH_V16_CHAIN_LEN) == 26),
        "n_backends": int(len(W66_DEFAULT_MH_V16_BACKENDS)),
        "chain_len": int(W66_DEFAULT_MH_V16_CHAIN_LEN),
    }


def family_h255b_multi_hop_v16_eleven_axis(
        seed: int) -> dict[str, Any]:
    w = emit_multi_hop_v16_witness(
        seed=int(seed) + 47600)
    return {
        "schema": R147_SCHEMA_VERSION,
        "name": "h255b_multi_hop_v16_eleven_axis",
        "passed": bool(
            w.eleven_axis_used
            and "eleven_axis" in w.arbitration_kind),
        "kind": str(w.arbitration_kind),
    }


def family_h255c_multi_hop_v16_compromise_threshold(
        seed: int) -> dict[str, Any]:
    res = evaluate_dec_chain_len26_fidelity(
        seed=int(seed) + 47700)
    thr = int(res["compromise_threshold"])
    return {
        "schema": R147_SCHEMA_VERSION,
        "name": "h255c_multi_hop_v16_compromise_threshold",
        "passed": bool(1 <= thr <= 11),
        "threshold": int(thr),
    }


def family_h256_ecc_v18_total_codes(
        seed: int) -> dict[str, Any]:
    cb = ECCCodebookV18.init(seed=int(seed) + 47800)
    return {
        "schema": R147_SCHEMA_VERSION,
        "name": "h256_ecc_v18_total_codes",
        "passed": bool(int(cb.total_codes) == 2 ** 29),
        "total_codes": int(cb.total_codes),
    }


def family_h256b_ecc_v18_bits_per_token(
        seed: int) -> dict[str, Any]:
    cb = ECCCodebookV18.init(seed=int(seed) + 47900)
    gate = QuantisedBudgetGate.init(
        in_dim=W53_DEFAULT_ECC_CODE_DIM,
        emit_mask_len=W53_DEFAULT_ECC_EMIT_MASK_LEN,
        seed=int(seed) + 47950)
    gate.importance_threshold = 0.0
    gate.w_emit.values = [1.0] * len(gate.w_emit.values)
    carrier = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    comp = compress_carrier_ecc_v18(
        carrier, codebook=cb, gate=gate)
    return {
        "schema": R147_SCHEMA_VERSION,
        "name": "h256b_ecc_v18_bits_per_token",
        "passed": bool(
            float(comp["bits_per_visible_token"]) >= 31.0),
        "bits_per": float(comp["bits_per_visible_token"]),
    }


def family_h256c_ecc_v18_rate_floor_falsifier(
        seed: int) -> dict[str, Any]:
    cb = ECCCodebookV18.init(seed=int(seed) + 48000)
    f = probe_ecc_v18_rate_floor_falsifier(codebook=cb)
    return {
        "schema": R147_SCHEMA_VERSION,
        "name": "h256c_ecc_v18_rate_floor_falsifier",
        "passed": bool(f["target_exceeds_ceiling"]),
        "ceiling_bits": float(f["ceiling_bits"]),
    }


def family_h259a_ecc_v18_k17_meta15_index(
        seed: int) -> dict[str, Any]:
    cb = ECCCodebookV18.init(seed=int(seed) + 48100)
    gate = QuantisedBudgetGate.init(
        in_dim=W53_DEFAULT_ECC_CODE_DIM,
        emit_mask_len=W53_DEFAULT_ECC_EMIT_MASK_LEN,
        seed=int(seed) + 48150)
    gate.importance_threshold = 0.0
    gate.w_emit.values = [1.0] * len(gate.w_emit.values)
    carrier = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    comp = compress_carrier_ecc_v18(
        carrier, codebook=cb, gate=gate)
    idx = int(comp["meta15_index"])
    return {
        "schema": R147_SCHEMA_VERSION,
        "name": "h259a_ecc_v18_k17_meta15_index",
        "passed": bool(0 <= idx < cb.n_meta15),
        "meta15_index": int(idx),
    }


_R147_FAMILIES: tuple[Any, ...] = (
    family_h253_persistent_v18_fifteenth_skip,
    family_h253b_persistent_v18_chain_walk_depth,
    family_h253c_persistent_v18_seventeen_layers,
    family_h253d_persistent_v18_chain_preserves_cids,
    family_h254_lhr_v18_heads_and_max_k,
    family_h254b_lhr_v18_seventeen_way_value,
    family_h254c_lhr_v18_eight_layer_scorer,
    family_h255_multi_hop_v16_backends,
    family_h255b_multi_hop_v16_eleven_axis,
    family_h255c_multi_hop_v16_compromise_threshold,
    family_h256_ecc_v18_total_codes,
    family_h256b_ecc_v18_bits_per_token,
    family_h256c_ecc_v18_rate_floor_falsifier,
    family_h259a_ecc_v18_k17_meta15_index,
)


def run_r147(
        seeds: Sequence[int] = (199, 299, 399),
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for s in seeds:
        results: dict[str, Any] = {}
        for fn in _R147_FAMILIES:
            try:
                r = fn(int(s))
            except Exception as e:
                r = {
                    "schema": R147_SCHEMA_VERSION,
                    "name": getattr(fn, "__name__", "unknown"),
                    "passed": False,
                    "exception": str(e),
                }
            results[r["name"]] = r
        out.append({
            "schema": R147_SCHEMA_VERSION,
            "seed": int(s),
            "family_results": results,
        })
    return out


__all__ = [
    "R147_SCHEMA_VERSION",
    "run_r147",
]
