"""W61 R-129 benchmark family — long-horizon retention / persistent
latent / ECC / multi-hop / mergeable capsule.

H153..H157b cell families.

* H153   persistent_v13 9th carrier (replay confidence) propagates
* H153b  persistent_v13 chain walk depth ≥ 1024
* H153c  persistent_v13 distractor rank ≥ 6
* H154   multi_hop_v11 chain length 16 with 306 edges
* H154b  multi_hop_v11 six-axis trust composite used
* H154c  multi_hop_v11 compromise threshold detected
* H155   lhr_v13 twelve_way_value runs
* H155b  lhr_v13 attention_pattern head non-trivial
* H155c  lhr_v13 three-layer scorer fits
* H156   ecc_v13 bits/visible-token ≥ 24.0 at full emit
* H156b  ecc_v13 total_codes = 2^22 = 4_194_304
* H156c  ecc_v13 falsifier reproduces structural cap
* H157   mlsc_v9 attention_pattern_witness_chain inherits via merge
* H157b  mlsc_v9 cache_retrieval_witness_chain inherits via merge
* H157c  mlsc_v9 per_layer_head_trust_matrix merges by max
"""

from __future__ import annotations

import math
from typing import Any, Sequence

from .ecc_codebook_v13 import (
    ECCCodebookV13, compress_carrier_ecc_v13,
    probe_ecc_v13_rate_floor_falsifier,
)
from .long_horizon_retention_v13 import (
    LongHorizonReconstructionV13Head,
    fit_lhr_v13_three_layer_scorer,
)
from .mergeable_latent_capsule_v3 import make_root_capsule_v3
from .mergeable_latent_capsule_v4 import wrap_v3_as_v4
from .mergeable_latent_capsule_v5 import wrap_v4_as_v5
from .mergeable_latent_capsule_v6 import wrap_v5_as_v6
from .mergeable_latent_capsule_v7 import wrap_v6_as_v7
from .mergeable_latent_capsule_v8 import wrap_v7_as_v8
from .mergeable_latent_capsule_v9 import (
    MergeOperatorV9, W61_MLSC_V9_ALGEBRA_ATTENTION_PATTERN_STEER,
    W61_MLSC_V9_ALGEBRA_CACHE_RETRIEVAL_QUERY,
    wrap_v8_as_v9,
)
from .multi_hop_translator_v11 import (
    W61_DEFAULT_MH_V11_BACKENDS,
    W61_DEFAULT_MH_V11_CHAIN_LEN,
    evaluate_dec_chain_len16_fidelity,
)
from .persistent_latent_v12 import V12StackedCell
from .persistent_latent_v13 import (
    PersistentLatentStateV13Chain,
    W61_DEFAULT_V13_DISTRACTOR_RANK,
    W61_DEFAULT_V13_MAX_CHAIN_WALK_DEPTH,
    emit_persistent_v13_witness,
    step_persistent_state_v13,
)
from .ecc_codebook_v5 import (
    W53_DEFAULT_ECC_CODE_DIM, W53_DEFAULT_ECC_EMIT_MASK_LEN,
)
from .quantised_compression import QuantisedBudgetGate


R129_SCHEMA_VERSION: str = "coordpy.r129_benchmark.v1"


def family_h153_persistent_v13_ninth_carrier(seed: int) -> dict[str, Any]:
    cell = V12StackedCell.init(seed=int(seed) + 29300)
    chain = PersistentLatentStateV13Chain.empty()
    s0 = step_persistent_state_v13(
        cell=cell, prev_state=None,
        carrier_values=[0.1] * 16, turn_index=0, role="a",
        replay_confidence_skip=[0.5] * 16)
    chain.add(s0)
    s1 = step_persistent_state_v13(
        cell=cell, prev_state=s0,
        carrier_values=[0.2] * 16, turn_index=1, role="a",
        replay_confidence_skip=[0.8] * 16)
    chain.add(s1)
    rc_total = float(
        sum(abs(v) for v in s1.replay_confidence_carrier))
    w = emit_persistent_v13_witness(chain, s1.cid())
    return {
        "schema": R129_SCHEMA_VERSION,
        "name": "h153_persistent_v13_ninth_carrier",
        "passed": bool(
            rc_total > 0.0 and w.nine_skip_present),
        "rc_total": float(rc_total),
    }


def family_h153b_persistent_v13_chain_walk_depth(seed: int) -> dict[str, Any]:
    cell = V12StackedCell.init(seed=int(seed) + 29400)
    chain = PersistentLatentStateV13Chain.empty()
    prev = None
    for i in range(8):
        s = step_persistent_state_v13(
            cell=cell, prev_state=prev,
            carrier_values=[float(i) * 0.1] * 16,
            turn_index=i, role="a",
            replay_confidence_skip=[float(i) * 0.05] * 16)
        chain.add(s)
        prev = s
    return {
        "schema": R129_SCHEMA_VERSION,
        "name": "h153b_persistent_v13_chain_walk_depth",
        "passed": bool(
            W61_DEFAULT_V13_MAX_CHAIN_WALK_DEPTH >= 1024
            and len(chain.states) == 8),
        "n_states": int(len(chain.states)),
    }


def family_h153c_persistent_v13_distractor_rank(seed: int) -> dict[str, Any]:
    return {
        "schema": R129_SCHEMA_VERSION,
        "name": "h153c_persistent_v13_distractor_rank",
        "passed": bool(W61_DEFAULT_V13_DISTRACTOR_RANK >= 6),
        "rank": int(W61_DEFAULT_V13_DISTRACTOR_RANK),
    }


def family_h154_multi_hop_v11_chain_length(seed: int) -> dict[str, Any]:
    w = evaluate_dec_chain_len16_fidelity(
        seed=int(seed) + 29500)
    n_back = len(W61_DEFAULT_MH_V11_BACKENDS)
    return {
        "schema": R129_SCHEMA_VERSION,
        "name": "h154_multi_hop_v11_chain_length",
        "passed": bool(
            w.chain_length == 16
            and w.n_edges == n_back * (n_back - 1)
            and n_back == 18),
        "chain_length": int(w.chain_length),
        "n_edges": int(w.n_edges),
    }


def family_h154b_multi_hop_v11_six_axis(seed: int) -> dict[str, Any]:
    w = evaluate_dec_chain_len16_fidelity(
        seed=int(seed) + 29600)
    return {
        "schema": R129_SCHEMA_VERSION,
        "name": "h154b_multi_hop_v11_six_axis_trust",
        "passed": bool(
            w.composite_trust_used
            and w.attention_pattern_axis_used),
        "kind": str(w.arbitration_kind),
    }


def family_h154c_multi_hop_v11_compromise_threshold(seed: int) -> dict[str, Any]:
    w = evaluate_dec_chain_len16_fidelity(
        seed=int(seed) + 29700)
    return {
        "schema": R129_SCHEMA_VERSION,
        "name": "h154c_multi_hop_v11_compromise_threshold",
        "passed": bool(1 <= w.compromise_threshold <= 6),
        "threshold": int(w.compromise_threshold),
    }


def family_h155_lhr_v13_twelve_way(seed: int) -> dict[str, Any]:
    head = LongHorizonReconstructionV13Head.init(
        seed=int(seed) + 29800)
    out = head.twelve_way_value(
        carrier=[0.1] * 8, k=4,
        replay_state=[0.5] * 4, retrieval_state=[0.3] * 4,
        attention_state=[0.2] * 4, hidden_state=[0.4] * 4,
        attention_top_k_indicator=[0.7] * 8,
        substrate_state=[0.3] * 4)
    import numpy as np
    return {
        "schema": R129_SCHEMA_VERSION,
        "name": "h155_lhr_v13_twelve_way",
        "passed": bool(
            np.asarray(out).size == head.out_dim
            and head.attention_pattern_W is not None),
    }


def family_h155b_lhr_v13_attention_head_nontrivial(seed: int) -> dict[str, Any]:
    head = LongHorizonReconstructionV13Head.init(
        seed=int(seed) + 29900)
    out_with = head.attention_pattern_value(
        attention_top_k_indicator=[1.0] * 8)
    out_zero = head.attention_pattern_value(
        attention_top_k_indicator=[0.0] * 8)
    import numpy as np
    diff = float(np.linalg.norm(out_with - out_zero))
    return {
        "schema": R129_SCHEMA_VERSION,
        "name": "h155b_lhr_v13_attention_head_nontrivial",
        "passed": bool(diff > 1e-6),
        "delta_l2": float(diff),
    }


def family_h155c_lhr_v13_three_layer_scorer(seed: int) -> dict[str, Any]:
    import numpy as np
    head = LongHorizonReconstructionV13Head.init(
        seed=int(seed) + 30000)
    rng = np.random.default_rng(int(seed) + 30001)
    carriers = [list(rng.standard_normal(16))
                 for _ in range(16)]
    targets = [float(np.linalg.norm(c)) for c in carriers]
    head, residual = fit_lhr_v13_three_layer_scorer(
        head, train_carriers=carriers, train_targets=targets)
    return {
        "schema": R129_SCHEMA_VERSION,
        "name": "h155c_lhr_v13_three_layer_scorer",
        "passed": bool(
            head.scorer_layer3 is not None
            and head.scorer_layer3.size > 0),
        "residual": float(residual),
    }


def family_h156_ecc_v13_bits_per_token(seed: int) -> dict[str, Any]:
    import numpy as np
    cb = ECCCodebookV13.init(seed=int(seed) + 30100)
    gate = QuantisedBudgetGate.init(
        in_dim=W53_DEFAULT_ECC_CODE_DIM,
        emit_mask_len=W53_DEFAULT_ECC_EMIT_MASK_LEN,
        seed=int(seed) + 30101)
    gate.importance_threshold = 0.0
    gate.w_emit.values = [1.0] * len(gate.w_emit.values)
    rng = np.random.default_rng(int(seed) + 30102)
    carrier = list(rng.standard_normal(
        W53_DEFAULT_ECC_CODE_DIM).tolist())
    out = compress_carrier_ecc_v13(
        carrier=carrier, codebook=cb, gate=gate)
    return {
        "schema": R129_SCHEMA_VERSION,
        "name": "h156_ecc_v13_bits_per_token",
        "passed": bool(
            float(out["bits_per_visible_token"]) >= 24.0),
        "bits_per_token": float(out["bits_per_visible_token"]),
    }


def family_h156b_ecc_v13_total_codes(seed: int) -> dict[str, Any]:
    cb = ECCCodebookV13.init(seed=int(seed) + 30200)
    return {
        "schema": R129_SCHEMA_VERSION,
        "name": "h156b_ecc_v13_total_codes",
        "passed": bool(cb.total_codes == (1 << 22)),
        "total_codes": int(cb.total_codes),
    }


def family_h156c_ecc_v13_falsifier(seed: int) -> dict[str, Any]:
    cb = ECCCodebookV13.init(seed=int(seed) + 30300)
    r = probe_ecc_v13_rate_floor_falsifier(
        codebook=cb, target_bits_per_token=2048.0)
    return {
        "schema": R129_SCHEMA_VERSION,
        "name": "h156c_ecc_v13_falsifier",
        "passed": bool(
            r["target_above_info_bound"]
            and r["reproduces_cap"]),
        "info_bound": float(r["info_bound"]),
    }


def _make_v9_capsule(
        seed: int, payload: list[float], *,
        attention_pattern_witness: tuple[str, ...] = (),
        cache_retrieval_witness: tuple[str, ...] = (),
        trust_matrix: tuple[tuple[int, int, float], ...] = (),
):
    v3 = make_root_capsule_v3(
        branch_id="r129",
        payload=tuple(payload),
        trust=1.0, confidence=1.0)
    v4 = wrap_v3_as_v4(v3)
    v5 = wrap_v4_as_v5(v4)
    v6 = wrap_v5_as_v6(v5)
    v7 = wrap_v6_as_v7(v6)
    v8 = wrap_v7_as_v8(v7)
    return wrap_v8_as_v9(
        v8,
        attention_pattern_witness_chain=attention_pattern_witness,
        cache_retrieval_witness_chain=cache_retrieval_witness,
        per_layer_head_trust_matrix=trust_matrix,
        algebra_signature_v7=(
            W61_MLSC_V9_ALGEBRA_ATTENTION_PATTERN_STEER))


def family_h157_mlsc_v9_attention_pattern_chain(seed: int) -> dict[str, Any]:
    cap_a = _make_v9_capsule(
        seed, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        attention_pattern_witness=("attn_a", "attn_b"))
    cap_b = _make_v9_capsule(
        seed, [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        attention_pattern_witness=("attn_c",))
    op = MergeOperatorV9()
    merged = op.merge(
        [cap_a, cap_b],
        attention_pattern_witness_chain=("attn_d",),
        algebra_signature_v7=(
            W61_MLSC_V9_ALGEBRA_ATTENTION_PATTERN_STEER))
    expected = {"attn_a", "attn_b", "attn_c", "attn_d"}
    actual = set(merged.attention_pattern_witness_chain)
    return {
        "schema": R129_SCHEMA_VERSION,
        "name": "h157_mlsc_v9_attention_pattern_chain",
        "passed": bool(expected.issubset(actual)),
        "chain": sorted(actual),
    }


def family_h157b_mlsc_v9_cache_retrieval_chain(seed: int) -> dict[str, Any]:
    cap_a = _make_v9_capsule(
        seed, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        cache_retrieval_witness=("cr_a",))
    cap_b = _make_v9_capsule(
        seed, [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        cache_retrieval_witness=("cr_b", "cr_c"))
    op = MergeOperatorV9()
    merged = op.merge(
        [cap_a, cap_b],
        cache_retrieval_witness_chain=("cr_d",),
        algebra_signature_v7=(
            W61_MLSC_V9_ALGEBRA_CACHE_RETRIEVAL_QUERY))
    expected = {"cr_a", "cr_b", "cr_c", "cr_d"}
    actual = set(merged.cache_retrieval_witness_chain)
    return {
        "schema": R129_SCHEMA_VERSION,
        "name": "h157b_mlsc_v9_cache_retrieval_chain",
        "passed": bool(expected.issubset(actual)),
        "chain": sorted(actual),
    }


def family_h157c_mlsc_v9_trust_matrix_merge(seed: int) -> dict[str, Any]:
    cap_a = _make_v9_capsule(
        seed, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        trust_matrix=((0, 0, 0.5), (1, 1, 0.6)))
    cap_b = _make_v9_capsule(
        seed, [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        trust_matrix=((0, 0, 0.9), (2, 2, 0.4)))
    op = MergeOperatorV9()
    merged = op.merge(
        [cap_a, cap_b],
        algebra_signature_v7=(
            W61_MLSC_V9_ALGEBRA_ATTENTION_PATTERN_STEER))
    table = dict(((int(l), int(h)), float(t))
                  for l, h, t in
                  merged.per_layer_head_trust_matrix)
    # max(0.5, 0.9) = 0.9 at (0,0); (1,1)=0.6; (2,2)=0.4
    return {
        "schema": R129_SCHEMA_VERSION,
        "name": "h157c_mlsc_v9_trust_matrix_merge",
        "passed": bool(
            table.get((0, 0), 0.0) == 0.9
            and table.get((1, 1), 0.0) == 0.6
            and table.get((2, 2), 0.0) == 0.4),
        "table": table,
    }


_R129_FAMILIES: tuple[Any, ...] = (
    family_h153_persistent_v13_ninth_carrier,
    family_h153b_persistent_v13_chain_walk_depth,
    family_h153c_persistent_v13_distractor_rank,
    family_h154_multi_hop_v11_chain_length,
    family_h154b_multi_hop_v11_six_axis,
    family_h154c_multi_hop_v11_compromise_threshold,
    family_h155_lhr_v13_twelve_way,
    family_h155b_lhr_v13_attention_head_nontrivial,
    family_h155c_lhr_v13_three_layer_scorer,
    family_h156_ecc_v13_bits_per_token,
    family_h156b_ecc_v13_total_codes,
    family_h156c_ecc_v13_falsifier,
    family_h157_mlsc_v9_attention_pattern_chain,
    family_h157b_mlsc_v9_cache_retrieval_chain,
    family_h157c_mlsc_v9_trust_matrix_merge,
)


def run_r129(
        seeds: Sequence[int] = (197, 297, 397),
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for s in seeds:
        results = {}
        for fn in _R129_FAMILIES:
            try:
                r = fn(int(s))
            except Exception as e:
                r = {
                    "schema": R129_SCHEMA_VERSION,
                    "name": getattr(fn, "__name__", "unknown"),
                    "passed": False,
                    "exception": str(e),
                }
            results[r["name"]] = r
        out.append({
            "schema": R129_SCHEMA_VERSION,
            "seed": int(s),
            "family_results": results,
        })
    return out


__all__ = [
    "R129_SCHEMA_VERSION",
    "run_r129",
]
