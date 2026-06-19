"""W62 R-132 benchmark family — long-horizon retention /
reconstruction / aggressive-compression.

H168..H171c cell families.

* H168   persistent V14 chain walk depth ≥ 2048
* H168b  persistent V14 decuple skip carries rd EMA
* H168c  persistent V14 distractor rank ≥ 8
* H169   LHR V14 13-way reconstruction head runs
* H169b  LHR V14 replay-dominance head produces non-trivial output
* H169c  LHR V14 four-layer scorer fits
* H170   ECC V14 bits/visible-token ≥ 25.0
* H170b  ECC V14 total codes = 2^23
* H170c  ECC V14 rate floor falsifier reproduces ceiling
* H171   multi-hop V12 chain-length 17 over 20 backends
* H171b  multi-hop V12 seven-axis composite used
* H171c  multi-hop V12 compromise threshold in [1, 7]
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as _np

from .ecc_codebook_v14 import (
    ECCCodebookV14, compress_carrier_ecc_v14,
    emit_ecc_v14_compression_witness,
    probe_ecc_v14_rate_floor_falsifier,
)
from .ecc_codebook_v5 import (
    W53_DEFAULT_ECC_CODE_DIM, W53_DEFAULT_ECC_EMIT_MASK_LEN,
)
from .long_horizon_retention_v14 import (
    LongHorizonReconstructionV14Head,
    emit_lhr_v14_witness,
    fit_lhr_v14_four_layer_scorer,
)
from .multi_hop_translator_v12 import (
    W62_DEFAULT_MH_V12_BACKENDS,
    W62_DEFAULT_MH_V12_CHAIN_LEN,
    emit_multi_hop_v12_witness,
)
from .persistent_latent_v12 import V12StackedCell
from .persistent_latent_v14 import (
    PersistentLatentStateV14Chain,
    W62_DEFAULT_V14_DISTRACTOR_RANK,
    W62_DEFAULT_V14_MAX_CHAIN_WALK_DEPTH,
    emit_persistent_v14_witness,
    step_persistent_state_v14,
)
from .quantised_compression import QuantisedBudgetGate


R132_SCHEMA_VERSION: str = "coordpy.r132_benchmark.v1"


def family_h168_persistent_v14_chain_walk_depth(
        seed: int) -> dict[str, Any]:
    return {
        "schema": R132_SCHEMA_VERSION,
        "name": "h168_persistent_v14_chain_walk_depth",
        "passed": bool(
            W62_DEFAULT_V14_MAX_CHAIN_WALK_DEPTH >= 2048),
        "depth": int(
            W62_DEFAULT_V14_MAX_CHAIN_WALK_DEPTH),
    }


def family_h168b_persistent_v14_decuple_skip(
        seed: int) -> dict[str, Any]:
    cell = V12StackedCell.init(seed=int(seed) + 31800)
    sd = int(cell.state_dim)
    rd_skip = [1.0] * sd
    state = step_persistent_state_v14(
        cell=cell, prev_state=None,
        carrier_values=[0.1] * sd,
        turn_index=0, role="r",
        replay_dominance_skip=rd_skip)
    chain = PersistentLatentStateV14Chain.empty()
    chain.add(state)
    w = emit_persistent_v14_witness(chain, state.cid())
    return {
        "schema": R132_SCHEMA_VERSION,
        "name": "h168b_persistent_v14_decuple_skip",
        "passed": bool(w.decuple_skip_present),
        "rd_l1_sum": float(
            w.replay_dominance_carrier_l1_sum),
    }


def family_h168c_persistent_v14_distractor_rank(
        seed: int) -> dict[str, Any]:
    return {
        "schema": R132_SCHEMA_VERSION,
        "name": "h168c_persistent_v14_distractor_rank",
        "passed": bool(W62_DEFAULT_V14_DISTRACTOR_RANK >= 8),
        "rank": int(W62_DEFAULT_V14_DISTRACTOR_RANK),
    }


def family_h169_lhr_v14_thirteen_way(
        seed: int) -> dict[str, Any]:
    head = LongHorizonReconstructionV14Head.init(
        seed=int(seed) + 31900)
    runs = True
    try:
        head.thirteen_way_value(
            carrier=[0.1] * 8, k=4,
            replay_dominance_indicator=[1.0] * 8)
    except Exception:
        runs = False
    return {
        "schema": R132_SCHEMA_VERSION,
        "name": "h169_lhr_v14_thirteen_way",
        "passed": bool(runs),
    }


def family_h169b_lhr_v14_replay_dominance_head(
        seed: int) -> dict[str, Any]:
    head = LongHorizonReconstructionV14Head.init(
        seed=int(seed) + 32000)
    zero = head.thirteen_way_value(
        carrier=[0.1] * 8, k=4,
        replay_dominance_indicator=[0.0] * 8)
    one = head.thirteen_way_value(
        carrier=[0.1] * 8, k=4,
        replay_dominance_indicator=[1.0] * 8)
    delta = float(_np.linalg.norm(
        _np.asarray(zero, dtype=_np.float64)
        - _np.asarray(one, dtype=_np.float64)))
    return {
        "schema": R132_SCHEMA_VERSION,
        "name": "h169b_lhr_v14_replay_dominance_head",
        "passed": bool(delta > 1e-6),
        "delta": float(delta),
    }


def family_h169c_lhr_v14_four_layer_scorer(
        seed: int) -> dict[str, Any]:
    head = LongHorizonReconstructionV14Head.init(
        seed=int(seed) + 32100)
    rng = _np.random.default_rng(int(seed) + 32101)
    # Use the V13 tanh_proj_dim so the V14 tanh2 layer applies.
    feat_dim = int(head.inner_v13.tanh_proj_dim)
    n = 20
    X = rng.standard_normal((n, feat_dim))
    y = X.sum(axis=-1)
    _, audit = fit_lhr_v14_four_layer_scorer(
        head=head, train_features=X.tolist(),
        train_targets=y.tolist())
    return {
        "schema": R132_SCHEMA_VERSION,
        "name": "h169c_lhr_v14_four_layer_scorer",
        "passed": bool(audit.get("converged", False)),
        "post_residual": float(audit.get("post_fit_residual", 0.0)),
    }


def family_h170_ecc_v14_bits_per_token(
        seed: int) -> dict[str, Any]:
    ecc = ECCCodebookV14.init(seed=int(seed) + 32200)
    gate = QuantisedBudgetGate.init(
        in_dim=W53_DEFAULT_ECC_CODE_DIM,
        emit_mask_len=W53_DEFAULT_ECC_EMIT_MASK_LEN,
        seed=int(seed) + 32201)
    gate.importance_threshold = 0.0
    gate.w_emit.values = [1.0] * len(gate.w_emit.values)
    rng = _np.random.default_rng(int(seed) + 32202)
    carrier = list(rng.standard_normal(
        W53_DEFAULT_ECC_CODE_DIM))
    comp = compress_carrier_ecc_v14(
        carrier, codebook=ecc, gate=gate)
    bits_per = float(comp["bits_per_visible_token"])
    return {
        "schema": R132_SCHEMA_VERSION,
        "name": "h170_ecc_v14_bits_per_token",
        "passed": bool(bits_per >= 25.0),
        "bits_per_token": float(bits_per),
    }


def family_h170b_ecc_v14_total_codes(
        seed: int) -> dict[str, Any]:
    ecc = ECCCodebookV14.init(seed=int(seed) + 32300)
    return {
        "schema": R132_SCHEMA_VERSION,
        "name": "h170b_ecc_v14_total_codes",
        "passed": bool(ecc.total_codes == 2 ** 23),
        "total_codes": int(ecc.total_codes),
    }


def family_h170c_ecc_v14_rate_floor_falsifier(
        seed: int) -> dict[str, Any]:
    ecc = ECCCodebookV14.init(seed=int(seed) + 32400)
    out = probe_ecc_v14_rate_floor_falsifier(codebook=ecc)
    return {
        "schema": R132_SCHEMA_VERSION,
        "name": "h170c_ecc_v14_rate_floor_falsifier",
        "passed": bool(out["target_exceeds_ceiling"]),
        "ceiling_bits": float(out["ceiling_bits"]),
        "target_bits": float(out["target_bits"]),
    }


def family_h171_multi_hop_v12_chain_length(
        seed: int) -> dict[str, Any]:
    w = emit_multi_hop_v12_witness(
        backends=W62_DEFAULT_MH_V12_BACKENDS,
        chain_length=W62_DEFAULT_MH_V12_CHAIN_LEN,
        seed=int(seed) + 32500)
    return {
        "schema": R132_SCHEMA_VERSION,
        "name": "h171_multi_hop_v12_chain_length",
        "passed": bool(
            w.chain_length == 17
            and w.n_backends == 20
            and w.n_edges == 20 * 19),
        "chain_length": int(w.chain_length),
        "n_backends": int(w.n_backends),
        "n_edges": int(w.n_edges),
    }


def family_h171b_multi_hop_v12_seven_axis(
        seed: int) -> dict[str, Any]:
    w = emit_multi_hop_v12_witness(
        backends=W62_DEFAULT_MH_V12_BACKENDS,
        chain_length=W62_DEFAULT_MH_V12_CHAIN_LEN,
        seed=int(seed) + 32600)
    return {
        "schema": R132_SCHEMA_VERSION,
        "name": "h171b_multi_hop_v12_seven_axis",
        "passed": bool(w.seven_axis_used),
    }


def family_h171c_multi_hop_v12_compromise_threshold(
        seed: int) -> dict[str, Any]:
    w = emit_multi_hop_v12_witness(
        backends=W62_DEFAULT_MH_V12_BACKENDS,
        chain_length=W62_DEFAULT_MH_V12_CHAIN_LEN,
        seed=int(seed) + 32700)
    t = int(w.compromise_threshold)
    return {
        "schema": R132_SCHEMA_VERSION,
        "name": "h171c_multi_hop_v12_compromise_threshold",
        "passed": bool(1 <= t <= 7),
        "compromise_threshold": int(t),
    }


_R132_FAMILIES: tuple[Any, ...] = (
    family_h168_persistent_v14_chain_walk_depth,
    family_h168b_persistent_v14_decuple_skip,
    family_h168c_persistent_v14_distractor_rank,
    family_h169_lhr_v14_thirteen_way,
    family_h169b_lhr_v14_replay_dominance_head,
    family_h169c_lhr_v14_four_layer_scorer,
    family_h170_ecc_v14_bits_per_token,
    family_h170b_ecc_v14_total_codes,
    family_h170c_ecc_v14_rate_floor_falsifier,
    family_h171_multi_hop_v12_chain_length,
    family_h171b_multi_hop_v12_seven_axis,
    family_h171c_multi_hop_v12_compromise_threshold,
)


def run_r132(
        seeds: Sequence[int] = (200, 300, 400),
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for s in seeds:
        results = {}
        for fn in _R132_FAMILIES:
            try:
                r = fn(int(s))
            except Exception as e:
                r = {
                    "schema": R132_SCHEMA_VERSION,
                    "name": getattr(fn, "__name__", "unknown"),
                    "passed": False,
                    "exception": str(e),
                }
            results[r["name"]] = r
        out.append({
            "schema": R132_SCHEMA_VERSION,
            "seed": int(s),
            "family_results": results,
        })
    return out


__all__ = ["R132_SCHEMA_VERSION", "run_r132"]
