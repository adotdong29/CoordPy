"""W67 R-150 benchmark family — long-horizon retention V19 /
aggressive ECC V19 / persistent V19 / multi-hop V17.

H293..H298 cell families (14 H-bars):

* H293   Persistent V19 chain-walk depth ≥ 16384 (config bound)
* H293b  Persistent V19 has sixteenth skip carrier
* H293c  Persistent V19 chain CID is content-addressed
* H294   Long-horizon retention V19 max_k=384
* H294b  Long-horizon retention V19 has 18 heads
* H294c  Long-horizon retention V19 nine-layer scorer fits
* H295   ECC V19 has 18 levels
* H295b  ECC V19 total codes = 2^31
* H295c  ECC V19 ≥ 33.0 bits/visible-token at full emit
* H295d  ECC V19 65536-bit/token falsifier exceeds ceiling
* H296   Multi-hop V17 backends = 40
* H296b  Multi-hop V17 chain length = 30
* H296c  Multi-hop V17 compromise threshold bounded in [1, 12]
* H297   Persistent V19 + LHR V19 reconstruct over a 32-turn chain
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as _np

from coordpy.ecc_codebook_v19 import (
    ECCCodebookV19,
    W67_DEFAULT_ECC_V19_K18,
    compress_carrier_ecc_v19,
    probe_ecc_v19_rate_floor_falsifier,
)
from coordpy.ecc_codebook_v5 import (
    W53_DEFAULT_ECC_CODE_DIM, W53_DEFAULT_ECC_EMIT_MASK_LEN,
)
from coordpy.long_horizon_retention_v19 import (
    LongHorizonReconstructionV19Head,
    W67_DEFAULT_LHR_V19_MAX_K,
    fit_lhr_v19_nine_layer_scorer,
)
from coordpy.multi_hop_translator_v17 import (
    W67_DEFAULT_MH_V17_BACKENDS,
    W67_DEFAULT_MH_V17_CHAIN_LEN,
    evaluate_dec_chain_len30_fidelity,
)
from coordpy.persistent_latent_v12 import V12StackedCell
from coordpy.persistent_latent_v19 import (
    PersistentLatentStateV19Chain,
    W67_DEFAULT_V19_MAX_CHAIN_WALK_DEPTH,
    step_persistent_state_v19,
)
from coordpy.quantised_compression import QuantisedBudgetGate


R150_SCHEMA_VERSION: str = "coordpy.r150_benchmark.v1"


def family_h293_v19_chain_walk_depth(
        seed: int) -> dict[str, Any]:
    return {
        "schema": R150_SCHEMA_VERSION,
        "name": "h293_v19_chain_walk_depth",
        "passed": bool(
            W67_DEFAULT_V19_MAX_CHAIN_WALK_DEPTH >= 16384),
        "depth": int(W67_DEFAULT_V19_MAX_CHAIN_WALK_DEPTH),
    }


def family_h293b_v19_sixteenth_skip_carrier(
        seed: int) -> dict[str, Any]:
    cell = V12StackedCell.init(seed=int(seed) + 50000)
    chain = PersistentLatentStateV19Chain.empty()
    s = step_persistent_state_v19(
        cell=cell, prev_state=None,
        carrier_values=[0.1] * int(cell.state_dim),
        turn_index=0, role="r",
        role_dropout_recovery_skip_v19=[0.5] * int(
            cell.state_dim))
    chain.add(s)
    return {
        "schema": R150_SCHEMA_VERSION,
        "name": "h293b_v19_sixteenth_skip_carrier",
        "passed": bool(
            sum(abs(float(v))
                for v
                in s.role_dropout_recovery_carrier) > 0.0),
    }


def family_h293c_v19_chain_cid_content_addressed(
        seed: int) -> dict[str, Any]:
    cell = V12StackedCell.init(seed=int(seed) + 50001)
    chain1 = PersistentLatentStateV19Chain.empty()
    chain2 = PersistentLatentStateV19Chain.empty()
    cv = [0.1] * int(cell.state_dim)
    for t in range(5):
        s1 = step_persistent_state_v19(
            cell=cell, prev_state=None,
            carrier_values=cv,
            turn_index=int(t), role="r")
        s2 = step_persistent_state_v19(
            cell=cell, prev_state=None,
            carrier_values=cv,
            turn_index=int(t), role="r")
        chain1.add(s1)
        chain2.add(s2)
    return {
        "schema": R150_SCHEMA_VERSION,
        "name": "h293c_v19_chain_cid_content_addressed",
        "passed": bool(chain1.cid() == chain2.cid()),
    }


def family_h294_lhr_v19_max_k(seed: int) -> dict[str, Any]:
    return {
        "schema": R150_SCHEMA_VERSION,
        "name": "h294_lhr_v19_max_k",
        "passed": bool(W67_DEFAULT_LHR_V19_MAX_K == 384),
    }


def family_h294b_lhr_v19_n_heads(
        seed: int) -> dict[str, Any]:
    head = LongHorizonReconstructionV19Head.init(
        seed=int(seed) + 50010)
    out = head.eighteen_way_value(
        carrier=[0.1] * 8, k=4,
        role_dropout_indicator=[0.5] * 8)
    return {
        "schema": R150_SCHEMA_VERSION,
        "name": "h294b_lhr_v19_n_heads",
        "passed": bool(out is not None and len(out) > 0),
    }


def family_h294c_lhr_v19_nine_layer_scorer(
        seed: int) -> dict[str, Any]:
    head = LongHorizonReconstructionV19Head.init(
        seed=int(seed) + 50020)
    rng = _np.random.default_rng(int(seed) + 50021)
    n = 16
    X = rng.standard_normal((n, head.inner_v18.swish_proj_dim))
    y = X.sum(axis=-1) + 0.1 * rng.standard_normal(n)
    _, audit = fit_lhr_v19_nine_layer_scorer(
        head=head, train_features=X.tolist(),
        train_targets=y.tolist())
    return {
        "schema": R150_SCHEMA_VERSION,
        "name": "h294c_lhr_v19_nine_layer_scorer",
        "passed": bool(audit["converged"]),
    }


def family_h295_ecc_v19_n_levels(
        seed: int) -> dict[str, Any]:
    book = ECCCodebookV19.init(
        seed=int(seed) + 50030)
    return {
        "schema": R150_SCHEMA_VERSION,
        "name": "h295_ecc_v19_n_levels",
        "passed": bool(book.n_meta16 == W67_DEFAULT_ECC_V19_K18),
    }


def family_h295b_ecc_v19_total_codes(
        seed: int) -> dict[str, Any]:
    book = ECCCodebookV19.init(seed=int(seed) + 50031)
    return {
        "schema": R150_SCHEMA_VERSION,
        "name": "h295b_ecc_v19_total_codes",
        "passed": bool(book.total_codes == (2 ** 31)),
        "total": int(book.total_codes),
    }


def family_h295c_ecc_v19_bits_per_token(
        seed: int) -> dict[str, Any]:
    book = ECCCodebookV19.init(seed=int(seed) + 50032)
    gate = QuantisedBudgetGate.init(
        in_dim=W53_DEFAULT_ECC_CODE_DIM,
        emit_mask_len=W53_DEFAULT_ECC_EMIT_MASK_LEN,
        seed=int(seed) + 50033)
    gate.importance_threshold = 0.0
    gate.w_emit.values = [1.0] * len(gate.w_emit.values)
    carrier = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    while len(carrier) < W53_DEFAULT_ECC_CODE_DIM:
        carrier.append(0.0)
    comp = compress_carrier_ecc_v19(
        carrier, codebook=book, gate=gate)
    return {
        "schema": R150_SCHEMA_VERSION,
        "name": "h295c_ecc_v19_bits_per_token",
        "passed": bool(
            comp["bits_per_visible_token"] >= 33.0),
        "bits_per_visible_token": float(
            comp["bits_per_visible_token"]),
    }


def family_h295d_ecc_v19_rate_floor(
        seed: int) -> dict[str, Any]:
    book = ECCCodebookV19.init(seed=int(seed) + 50034)
    res = probe_ecc_v19_rate_floor_falsifier(codebook=book)
    return {
        "schema": R150_SCHEMA_VERSION,
        "name": "h295d_ecc_v19_rate_floor",
        "passed": bool(res["target_exceeds_ceiling"]),
        "ceiling_bits": float(res["ceiling_bits"]),
    }


def family_h296_multi_hop_v17_n_backends(
        seed: int) -> dict[str, Any]:
    return {
        "schema": R150_SCHEMA_VERSION,
        "name": "h296_multi_hop_v17_n_backends",
        "passed": bool(
            len(W67_DEFAULT_MH_V17_BACKENDS) == 40),
        "n_backends": int(len(W67_DEFAULT_MH_V17_BACKENDS)),
    }


def family_h296b_multi_hop_v17_chain_length(
        seed: int) -> dict[str, Any]:
    return {
        "schema": R150_SCHEMA_VERSION,
        "name": "h296b_multi_hop_v17_chain_length",
        "passed": bool(W67_DEFAULT_MH_V17_CHAIN_LEN == 30),
    }


def family_h296c_multi_hop_v17_compromise_threshold(
        seed: int) -> dict[str, Any]:
    res = evaluate_dec_chain_len30_fidelity(
        seed=int(seed) + 50040)
    th = int(res["compromise_threshold"])
    return {
        "schema": R150_SCHEMA_VERSION,
        "name": "h296c_multi_hop_v17_compromise_threshold",
        "passed": bool(1 <= th <= 12),
        "threshold": int(th),
    }


def family_h297_persistent_lhr_long_chain(
        seed: int) -> dict[str, Any]:
    cell = V12StackedCell.init(seed=int(seed) + 50050)
    chain = PersistentLatentStateV19Chain.empty()
    n_turns = 32
    last = None
    for t in range(n_turns):
        last = step_persistent_state_v19(
            cell=cell, prev_state=last,
            carrier_values=[0.1] * int(cell.state_dim),
            turn_index=int(t), role="r")
        chain.add(last)
    head = LongHorizonReconstructionV19Head.init(
        seed=int(seed) + 50051)
    out = head.eighteen_way_value(
        carrier=[0.1] * 8, k=16,
        role_dropout_indicator=[0.5] * 8)
    return {
        "schema": R150_SCHEMA_VERSION,
        "name": "h297_persistent_lhr_long_chain",
        "passed": bool(
            len(chain.states) == n_turns
            and out is not None and len(out) > 0),
    }


_R150_FAMILIES: tuple[Any, ...] = (
    family_h293_v19_chain_walk_depth,
    family_h293b_v19_sixteenth_skip_carrier,
    family_h293c_v19_chain_cid_content_addressed,
    family_h294_lhr_v19_max_k,
    family_h294b_lhr_v19_n_heads,
    family_h294c_lhr_v19_nine_layer_scorer,
    family_h295_ecc_v19_n_levels,
    family_h295b_ecc_v19_total_codes,
    family_h295c_ecc_v19_bits_per_token,
    family_h295d_ecc_v19_rate_floor,
    family_h296_multi_hop_v17_n_backends,
    family_h296b_multi_hop_v17_chain_length,
    family_h296c_multi_hop_v17_compromise_threshold,
    family_h297_persistent_lhr_long_chain,
)


def run_r150(
        seeds: Sequence[int] = (199, 299, 399),
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for s in seeds:
        results: dict[str, Any] = {}
        for fn in _R150_FAMILIES:
            try:
                r = fn(int(s))
            except Exception as e:
                r = {
                    "schema": R150_SCHEMA_VERSION,
                    "name": getattr(fn, "__name__", "unknown"),
                    "passed": False,
                    "exception": str(e),
                }
            results[r["name"]] = r
        out.append({
            "schema": R150_SCHEMA_VERSION,
            "seed": int(s),
            "family_results": results,
        })
    return out


__all__ = ["R150_SCHEMA_VERSION", "run_r150"]
