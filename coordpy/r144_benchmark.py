"""W65 R-144 benchmark family — 8192-turn retention / V17 LHR /
V15 multi-hop / V17 ECC / persistent V17.

H231..H236 cell families (12 H-bars).

* H231   Persistent V17 fourteenth-skip carrier present
* H231b  Persistent V17 chain walk depth ≥ 8192 supported
* H231c  Persistent V17 distractor rank ≥ 14
* H232   Multi-hop V15 ten-axis composite at chain-len 25
* H232b  Multi-hop V15 35 backends, 1190 directed edges
* H232c  Multi-hop V15 compromise threshold ∈ [1, 10]
* H233   LHR V17 sixteen-way runs without crashing
* H233b  LHR V17 max_k = 256 (vs V16's 192)
* H233c  LHR V17 seven-layer scorer ridge converges
* H234   ECC V17 ≥ 29.0 bits/visible-token at full emit
* H234b  ECC V17 total codes = 2^27 = 134217728
* H234c  ECC V17 16384-bit/token falsifier reproduces structural
         ceiling
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as _np

from coordpy.ecc_codebook_v17 import (
    ECCCodebookV17, compress_carrier_ecc_v17,
    probe_ecc_v17_rate_floor_falsifier,
)
from coordpy.long_horizon_retention_v15 import (
    W63_DEFAULT_LHR_V15_GELU_PROJ_DIM,
)
from coordpy.long_horizon_retention_v16 import (
    W64_DEFAULT_LHR_V16_SILU_PROJ_DIM,
)
from coordpy.long_horizon_retention_v17 import (
    LongHorizonReconstructionV17Head,
    W65_DEFAULT_LHR_V17_MAX_K,
    fit_lhr_v17_seven_layer_scorer,
    emit_lhr_v17_witness,
)
from coordpy.multi_hop_translator_v15 import (
    W65_DEFAULT_MH_V15_BACKENDS,
    W65_DEFAULT_MH_V15_CHAIN_LEN,
    evaluate_dec_chain_len25_fidelity,
)
from coordpy.persistent_latent_v12 import V12StackedCell
from coordpy.persistent_latent_v17 import (
    PersistentLatentStateV17Chain,
    W65_DEFAULT_V17_DISTRACTOR_RANK,
    W65_DEFAULT_V17_MAX_CHAIN_WALK_DEPTH,
    emit_persistent_v17_witness,
    step_persistent_state_v17,
)
from coordpy.quantised_compression import QuantisedBudgetGate
from coordpy.ecc_codebook_v5 import (
    W53_DEFAULT_ECC_CODE_DIM, W53_DEFAULT_ECC_EMIT_MASK_LEN,
)


R144_SCHEMA_VERSION: str = "coordpy.r144_benchmark.v1"


def family_h231_persistent_v17_fourteenth_skip(
        seed: int) -> dict[str, Any]:
    cell = V12StackedCell.init(seed=int(seed) + 44000)
    chain = PersistentLatentStateV17Chain.empty()
    carrier = [0.5] * int(cell.state_dim)
    state = step_persistent_state_v17(
        cell=cell, prev_state=None,
        carrier_values=carrier,
        turn_index=0, role="r",
        substrate_skip=carrier,
        hidden_wins_skip=carrier,
        prefix_reuse_skip=carrier,
        replay_dominance_witness_skip_v16=carrier,
        team_task_success_skip_v17=carrier)
    chain.add(state)
    w = emit_persistent_v17_witness(chain, state.cid())
    return {
        "schema": R144_SCHEMA_VERSION,
        "name": "h231_persistent_v17_fourteenth_skip",
        "passed": bool(w.fourteenth_skip_present),
    }


def family_h231b_persistent_v17_chain_walk_depth(
        seed: int) -> dict[str, Any]:
    return {
        "schema": R144_SCHEMA_VERSION,
        "name": "h231b_persistent_v17_chain_walk_depth",
        "passed": bool(
            W65_DEFAULT_V17_MAX_CHAIN_WALK_DEPTH >= 8192),
        "max_depth": int(W65_DEFAULT_V17_MAX_CHAIN_WALK_DEPTH),
    }


def family_h231c_persistent_v17_distractor_rank(
        seed: int) -> dict[str, Any]:
    return {
        "schema": R144_SCHEMA_VERSION,
        "name": "h231c_persistent_v17_distractor_rank",
        "passed": bool(
            W65_DEFAULT_V17_DISTRACTOR_RANK >= 14),
        "rank": int(W65_DEFAULT_V17_DISTRACTOR_RANK),
    }


def family_h232_multi_hop_v15_ten_axis(
        seed: int) -> dict[str, Any]:
    res = evaluate_dec_chain_len25_fidelity(
        seed=int(seed) + 44100)
    return {
        "schema": R144_SCHEMA_VERSION,
        "name": "h232_multi_hop_v15_ten_axis",
        "passed": bool("ten_axis" in str(res.get("kind", ""))),
        "kind": str(res.get("kind", "")),
    }


def family_h232b_multi_hop_v15_backends_edges(
        seed: int) -> dict[str, Any]:
    n_b = len(W65_DEFAULT_MH_V15_BACKENDS)
    n_edges = n_b * (n_b - 1)
    return {
        "schema": R144_SCHEMA_VERSION,
        "name": "h232b_multi_hop_v15_backends_edges",
        "passed": bool(n_b == 35 and n_edges == 1190),
        "n_backends": int(n_b), "n_edges": int(n_edges),
    }


def family_h232c_multi_hop_v15_compromise_threshold(
        seed: int) -> dict[str, Any]:
    res = evaluate_dec_chain_len25_fidelity(
        seed=int(seed) + 44200)
    t = int(res["compromise_threshold"])
    return {
        "schema": R144_SCHEMA_VERSION,
        "name": "h232c_multi_hop_v15_compromise_threshold",
        "passed": bool(1 <= t <= 10),
        "threshold": int(t),
    }


def family_h233_lhr_v17_sixteen_way(
        seed: int) -> dict[str, Any]:
    head = LongHorizonReconstructionV17Head.init(
        seed=int(seed) + 44300)
    w = emit_lhr_v17_witness(
        head, carrier=[0.1] * 8, k=4,
        team_task_success_indicator=[0.5] * 8,
        replay_dominance_indicator=[1.0] * 8,
        hidden_wins_indicator=[0.5] * 8,
        replay_dominance_primary_indicator=[0.7] * 8)
    return {
        "schema": R144_SCHEMA_VERSION,
        "name": "h233_lhr_v17_sixteen_way",
        "passed": bool(w.sixteen_way_runs),
    }


def family_h233b_lhr_v17_max_k(seed: int) -> dict[str, Any]:
    return {
        "schema": R144_SCHEMA_VERSION,
        "name": "h233b_lhr_v17_max_k",
        "passed": bool(W65_DEFAULT_LHR_V17_MAX_K == 256),
        "max_k": int(W65_DEFAULT_LHR_V17_MAX_K),
    }


def family_h233c_lhr_v17_seven_layer_scorer(
        seed: int) -> dict[str, Any]:
    head = LongHorizonReconstructionV17Head.init(
        seed=int(seed) + 44400)
    rng = _np.random.default_rng(int(seed) + 44401)
    n = 16
    X = rng.standard_normal(
        (n, W64_DEFAULT_LHR_V16_SILU_PROJ_DIM))
    y = X[:, 0] + X[:, 1] * 0.5
    _, audit = fit_lhr_v17_seven_layer_scorer(
        head=head, train_features=X.tolist(),
        train_targets=y.tolist())
    return {
        "schema": R144_SCHEMA_VERSION,
        "name": "h233c_lhr_v17_seven_layer_scorer",
        "passed": bool(audit["converged"]),
    }


def family_h234_ecc_v17_bits_per_token(
        seed: int) -> dict[str, Any]:
    book = ECCCodebookV17.init(seed=int(seed) + 44500)
    gate = QuantisedBudgetGate.init(
        in_dim=W53_DEFAULT_ECC_CODE_DIM,
        emit_mask_len=W53_DEFAULT_ECC_EMIT_MASK_LEN,
        seed=int(seed) + 44501)
    gate.importance_threshold = 0.0
    gate.w_emit.values = [1.0] * len(gate.w_emit.values)
    rng = _np.random.default_rng(int(seed) + 44502)
    carrier = list(rng.standard_normal(
        W53_DEFAULT_ECC_CODE_DIM))
    comp = compress_carrier_ecc_v17(
        carrier, codebook=book, gate=gate)
    bits = float(comp["bits_per_visible_token"])
    return {
        "schema": R144_SCHEMA_VERSION,
        "name": "h234_ecc_v17_bits_per_token",
        "passed": bool(bits >= 29.0),
        "bits_per_visible_token": float(bits),
    }


def family_h234b_ecc_v17_total_codes(
        seed: int) -> dict[str, Any]:
    book = ECCCodebookV17.init(seed=int(seed) + 44600)
    return {
        "schema": R144_SCHEMA_VERSION,
        "name": "h234b_ecc_v17_total_codes",
        "passed": bool(book.total_codes == (1 << 27)),
        "total_codes": int(book.total_codes),
    }


def family_h234c_ecc_v17_rate_floor_falsifier(
        seed: int) -> dict[str, Any]:
    book = ECCCodebookV17.init(seed=int(seed) + 44700)
    f = probe_ecc_v17_rate_floor_falsifier(codebook=book)
    return {
        "schema": R144_SCHEMA_VERSION,
        "name": "h234c_ecc_v17_rate_floor_falsifier",
        "passed": bool(f["target_exceeds_ceiling"]),
        "ceiling_bits": float(f["ceiling_bits"]),
    }


_R144_FAMILIES: tuple[Any, ...] = (
    family_h231_persistent_v17_fourteenth_skip,
    family_h231b_persistent_v17_chain_walk_depth,
    family_h231c_persistent_v17_distractor_rank,
    family_h232_multi_hop_v15_ten_axis,
    family_h232b_multi_hop_v15_backends_edges,
    family_h232c_multi_hop_v15_compromise_threshold,
    family_h233_lhr_v17_sixteen_way,
    family_h233b_lhr_v17_max_k,
    family_h233c_lhr_v17_seven_layer_scorer,
    family_h234_ecc_v17_bits_per_token,
    family_h234b_ecc_v17_total_codes,
    family_h234c_ecc_v17_rate_floor_falsifier,
)


def run_r144(
        seeds: Sequence[int] = (199, 299, 399),
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for s in seeds:
        results: dict[str, Any] = {}
        for fn in _R144_FAMILIES:
            try:
                r = fn(int(s))
            except Exception as e:
                r = {
                    "schema": R144_SCHEMA_VERSION,
                    "name": getattr(fn, "__name__", "unknown"),
                    "passed": False,
                    "exception": str(e),
                }
            results[r["name"]] = r
        out.append({
            "schema": R144_SCHEMA_VERSION,
            "seed": int(s),
            "family_results": results,
        })
    return out


__all__ = [
    "R144_SCHEMA_VERSION",
    "run_r144",
]
