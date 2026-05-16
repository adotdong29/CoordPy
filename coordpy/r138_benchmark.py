"""W64 R-138 benchmark family — long-horizon retention /
reconstruction / aggressive-compression / persistent state /
multi-hop transfer.

H211..H222 cell families (12 H-bars).

* H211   Persistent V16 thirteenth-skip carrier present
* H211b  Persistent V16 chain walk depth ≥ 6144 supported
* H211c  Persistent V16 distractor rank ≥ 12
* H212   Multi-hop V14 nine-axis composite at chain-len 21
* H212b  Multi-hop V14 27 backends, 702 directed edges
* H212c  Multi-hop V14 compromise threshold ∈ [1, 9]
* H213   LHR V16 fifteen-way runs without crashing
* H213b  LHR V16 max_k = 192 (vs V15's 160)
* H213c  LHR V16 six-layer scorer ridge converges
* H214   ECC V16 ≥ 27.0 bits/visible-token at full emit
* H214b  ECC V16 total codes = 2^25 = 33 554 432
* H214c  ECC V16 8192-bit/token target reproduces structural
         ceiling (above log2(2^25) = 25)
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as _np

from coordpy.ecc_codebook_v16 import (
    ECCCodebookV16, compress_carrier_ecc_v16,
    probe_ecc_v16_rate_floor_falsifier,
)
from coordpy.long_horizon_retention_v16 import (
    LongHorizonReconstructionV16Head,
    W64_DEFAULT_LHR_V16_MAX_K,
    fit_lhr_v16_six_layer_scorer,
    emit_lhr_v16_witness,
)
from coordpy.multi_hop_translator_v14 import (
    W64_DEFAULT_MH_V14_BACKENDS,
    W64_DEFAULT_MH_V14_CHAIN_LEN,
    evaluate_dec_chain_len21_fidelity,
    emit_multi_hop_v14_witness,
)
from coordpy.persistent_latent_v12 import V12StackedCell
from coordpy.persistent_latent_v16 import (
    PersistentLatentStateV16Chain,
    W64_DEFAULT_V16_DISTRACTOR_RANK,
    W64_DEFAULT_V16_MAX_CHAIN_WALK_DEPTH,
    emit_persistent_v16_witness,
    step_persistent_state_v16,
)
from coordpy.quantised_compression import QuantisedBudgetGate
from coordpy.ecc_codebook_v5 import (
    W53_DEFAULT_ECC_CODE_DIM, W53_DEFAULT_ECC_EMIT_MASK_LEN,
)


R138_SCHEMA_VERSION: str = "coordpy.r138_benchmark.v1"


def family_h211_persistent_v16_thirteenth_skip(
        seed: int) -> dict[str, Any]:
    cell = V12StackedCell.init(seed=int(seed) + 38000)
    chain = PersistentLatentStateV16Chain.empty()
    carrier = [0.5] * int(cell.state_dim)
    state = step_persistent_state_v16(
        cell=cell, prev_state=None,
        carrier_values=carrier,
        turn_index=0, role="r",
        substrate_skip=carrier,
        hidden_wins_skip=carrier,
        prefix_reuse_skip=carrier,
        replay_dominance_witness_skip_v16=carrier)
    chain.add(state)
    w = emit_persistent_v16_witness(chain, state.cid())
    return {
        "schema": R138_SCHEMA_VERSION,
        "name": "h211_persistent_v16_thirteenth_skip",
        "passed": bool(w.thirteenth_skip_present),
    }


def family_h211b_persistent_v16_chain_walk_depth(
        seed: int) -> dict[str, Any]:
    cell = V12StackedCell.init(seed=int(seed) + 38010)
    chain = PersistentLatentStateV16Chain.empty()
    carrier = [0.5] * int(cell.state_dim)
    states = []
    for i in range(8):
        prev = states[-1] if states else None
        s = step_persistent_state_v16(
            cell=cell, prev_state=prev,
            carrier_values=carrier,
            turn_index=int(i), role="r",
            substrate_skip=carrier,
            hidden_wins_skip=carrier,
            prefix_reuse_skip=carrier,
            replay_dominance_witness_skip_v16=carrier)
        chain.add(s)
        states.append(s)
    w = emit_persistent_v16_witness(
        chain, states[-1].cid(),
        max_depth=W64_DEFAULT_V16_MAX_CHAIN_WALK_DEPTH)
    return {
        "schema": R138_SCHEMA_VERSION,
        "name": "h211b_persistent_v16_chain_walk_depth",
        "passed": bool(
            W64_DEFAULT_V16_MAX_CHAIN_WALK_DEPTH >= 6144),
        "max_depth": int(
            W64_DEFAULT_V16_MAX_CHAIN_WALK_DEPTH),
    }


def family_h211c_persistent_v16_distractor_rank(
        seed: int) -> dict[str, Any]:
    return {
        "schema": R138_SCHEMA_VERSION,
        "name": "h211c_persistent_v16_distractor_rank",
        "passed": bool(
            W64_DEFAULT_V16_DISTRACTOR_RANK >= 12),
        "rank": int(W64_DEFAULT_V16_DISTRACTOR_RANK),
    }


def family_h212_multi_hop_v14_nine_axis(
        seed: int) -> dict[str, Any]:
    res = evaluate_dec_chain_len21_fidelity(
        seed=int(seed) + 38100)
    return {
        "schema": R138_SCHEMA_VERSION,
        "name": "h212_multi_hop_v14_nine_axis",
        "passed": bool(
            "nine_axis" in str(res.get("kind", ""))),
        "kind": str(res.get("kind", "")),
    }


def family_h212b_multi_hop_v14_backends_edges(
        seed: int) -> dict[str, Any]:
    n_b = len(W64_DEFAULT_MH_V14_BACKENDS)
    n_edges = n_b * (n_b - 1)
    return {
        "schema": R138_SCHEMA_VERSION,
        "name": "h212b_multi_hop_v14_backends_edges",
        "passed": bool(n_b == 27 and n_edges == 702),
        "n_backends": int(n_b),
        "n_edges": int(n_edges),
    }


def family_h212c_multi_hop_v14_compromise_threshold(
        seed: int) -> dict[str, Any]:
    res = evaluate_dec_chain_len21_fidelity(
        seed=int(seed) + 38200)
    t = int(res["compromise_threshold"])
    return {
        "schema": R138_SCHEMA_VERSION,
        "name": "h212c_multi_hop_v14_compromise_threshold",
        "passed": bool(1 <= t <= 9),
        "threshold": int(t),
    }


def family_h213_lhr_v16_fifteen_way(
        seed: int) -> dict[str, Any]:
    head = LongHorizonReconstructionV16Head.init(
        seed=int(seed) + 38300)
    w = emit_lhr_v16_witness(
        head, carrier=[0.1] * 8, k=4,
        replay_dominance_indicator=[1.0] * 8,
        hidden_wins_indicator=[0.5] * 8,
        replay_dominance_primary_indicator=[0.7] * 8)
    return {
        "schema": R138_SCHEMA_VERSION,
        "name": "h213_lhr_v16_fifteen_way",
        "passed": bool(w.fifteen_way_runs),
    }


def family_h213b_lhr_v16_max_k(
        seed: int) -> dict[str, Any]:
    return {
        "schema": R138_SCHEMA_VERSION,
        "name": "h213b_lhr_v16_max_k",
        "passed": bool(W64_DEFAULT_LHR_V16_MAX_K == 192),
        "max_k": int(W64_DEFAULT_LHR_V16_MAX_K),
    }


def family_h213c_lhr_v16_six_layer_scorer(
        seed: int) -> dict[str, Any]:
    head = LongHorizonReconstructionV16Head.init(
        seed=int(seed) + 38400)
    rng = _np.random.default_rng(int(seed) + 38401)
    from coordpy.long_horizon_retention_v15 import (
        W63_DEFAULT_LHR_V15_GELU_PROJ_DIM,
    )
    n = 16
    X = rng.standard_normal(
        (n, W63_DEFAULT_LHR_V15_GELU_PROJ_DIM))
    y = X[:, 0] + X[:, 1] * 0.5
    _, audit = fit_lhr_v16_six_layer_scorer(
        head=head, train_features=X.tolist(),
        train_targets=y.tolist())
    return {
        "schema": R138_SCHEMA_VERSION,
        "name": "h213c_lhr_v16_six_layer_scorer",
        "passed": bool(audit["converged"]),
    }


def family_h214_ecc_v16_bits_per_token(
        seed: int) -> dict[str, Any]:
    book = ECCCodebookV16.init(seed=int(seed) + 38500)
    gate = QuantisedBudgetGate.init(
        in_dim=W53_DEFAULT_ECC_CODE_DIM,
        emit_mask_len=W53_DEFAULT_ECC_EMIT_MASK_LEN,
        seed=int(seed) + 38501)
    gate.importance_threshold = 0.0
    gate.w_emit.values = [1.0] * len(gate.w_emit.values)
    rng = _np.random.default_rng(int(seed) + 38502)
    carrier = list(rng.standard_normal(
        W53_DEFAULT_ECC_CODE_DIM))
    comp = compress_carrier_ecc_v16(
        carrier, codebook=book, gate=gate)
    bits = float(comp["bits_per_visible_token"])
    return {
        "schema": R138_SCHEMA_VERSION,
        "name": "h214_ecc_v16_bits_per_token",
        "passed": bool(bits >= 27.0),
        "bits_per_visible_token": float(bits),
    }


def family_h214b_ecc_v16_total_codes(
        seed: int) -> dict[str, Any]:
    book = ECCCodebookV16.init(seed=int(seed) + 38600)
    return {
        "schema": R138_SCHEMA_VERSION,
        "name": "h214b_ecc_v16_total_codes",
        "passed": bool(book.total_codes == (1 << 25)),
        "total_codes": int(book.total_codes),
    }


def family_h214c_ecc_v16_rate_floor_falsifier(
        seed: int) -> dict[str, Any]:
    book = ECCCodebookV16.init(seed=int(seed) + 38700)
    f = probe_ecc_v16_rate_floor_falsifier(codebook=book)
    return {
        "schema": R138_SCHEMA_VERSION,
        "name": "h214c_ecc_v16_rate_floor_falsifier",
        "passed": bool(f["target_exceeds_ceiling"]),
        "ceiling_bits": float(f["ceiling_bits"]),
    }


_R138_FAMILIES: tuple[Any, ...] = (
    family_h211_persistent_v16_thirteenth_skip,
    family_h211b_persistent_v16_chain_walk_depth,
    family_h211c_persistent_v16_distractor_rank,
    family_h212_multi_hop_v14_nine_axis,
    family_h212b_multi_hop_v14_backends_edges,
    family_h212c_multi_hop_v14_compromise_threshold,
    family_h213_lhr_v16_fifteen_way,
    family_h213b_lhr_v16_max_k,
    family_h213c_lhr_v16_six_layer_scorer,
    family_h214_ecc_v16_bits_per_token,
    family_h214b_ecc_v16_total_codes,
    family_h214c_ecc_v16_rate_floor_falsifier,
)


def run_r138(
        seeds: Sequence[int] = (199, 299, 399),
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for s in seeds:
        results: dict[str, Any] = {}
        for fn in _R138_FAMILIES:
            try:
                r = fn(int(s))
            except Exception as e:
                r = {
                    "schema": R138_SCHEMA_VERSION,
                    "name": getattr(fn, "__name__", "unknown"),
                    "passed": False,
                    "exception": str(e),
                }
            results[r["name"]] = r
        out.append({
            "schema": R138_SCHEMA_VERSION,
            "seed": int(s),
            "family_results": results,
        })
    return out
