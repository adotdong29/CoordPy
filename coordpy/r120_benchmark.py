"""R-120 — W58 long-horizon / reconstruction / cramming family.

Families:

* H90  LHR V10 four-way runs without crash (9 heads, max_k=72)
* H90b LHR V10 attention head beats substrate head on attention-aligned
* H91  MLSC V6 attention_witness_chain inheritance
* H91b MLSC V6 cache_reuse_witness_cid propagation
* H97a ECC V10 ≥ 21.0 bits/visible-token at full emit
* H97b ECC V10 total_codes = 524288
* H95  ECC V10 1024-bit/token falsifier reproduces
* H101 V10 chain walk depth = 512
* H101b V10 8-layer cell
* H101c V10 attention-fidelity damping changes top state
* H101d Persistent V10 carrier-round-trip-deterministic
* H102 multi-hop V8 chain-length-11 over 12 backends
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.r120_benchmark requires numpy") from exc

from .ecc_codebook_v10 import (
    ECCCodebookV10, compress_carrier_ecc_v10,
    probe_ecc_v10_rate_floor_falsifier,
)
from .ecc_codebook_v5 import W53_DEFAULT_ECC_CODE_DIM
from .long_horizon_retention_v10 import (
    LongHorizonReconstructionV10Head,
    evaluate_lhr_v10_four_way,
)
from .mergeable_latent_capsule_v3 import make_root_capsule_v3
from .mergeable_latent_capsule_v4 import wrap_v3_as_v4
from .mergeable_latent_capsule_v5 import wrap_v4_as_v5
from .mergeable_latent_capsule_v6 import (
    MergeOperatorV6, wrap_v5_as_v6,
)
from .multi_hop_translator_v8 import (
    W58_DEFAULT_MH_V8_BACKENDS,
    W58_DEFAULT_MH_V8_CHAIN_LEN,
    evaluate_dec_chain_len11_fidelity,
)
from .persistent_latent_v10 import (
    V10StackedCell, PersistentLatentStateV10Chain,
    step_persistent_state_v10,
    W58_DEFAULT_V10_MAX_CHAIN_WALK_DEPTH,
)
from .quantised_compression import QuantisedBudgetGate


R120_SCHEMA_VERSION: str = "coordpy.r120_benchmark.v1"


@dataclasses.dataclass(frozen=True)
class R120SeedResult:
    seed: int
    family_results: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": R120_SCHEMA_VERSION,
            "seed": int(self.seed),
            "family_results": dict(self.family_results),
        }


def family_h90_lhr_v10_four_way_runs(seed: int) -> dict[str, Any]:
    head = LongHorizonReconstructionV10Head.init(
        seed=int(seed) + 20100)
    rng = _np.random.default_rng(int(seed) + 20101)
    carriers = [
        rng.standard_normal(8).tolist() for _ in range(4)]
    targets = [
        rng.standard_normal(head.out_dim).tolist() for _ in range(4)]
    subs = [rng.standard_normal(8).tolist() for _ in range(4)]
    hids = [rng.standard_normal(8).tolist() for _ in range(4)]
    atts = [rng.standard_normal(8).tolist() for _ in range(4)]
    res = evaluate_lhr_v10_four_way(
        head, carrier_examples=carriers,
        target_examples=targets,
        substrate_states=subs, hidden_states=hids,
        attention_states=atts, k=4)
    return {
        "schema": R120_SCHEMA_VERSION,
        "name": "h90_lhr_v10_four_way_runs",
        "passed": bool(
            res["n"] == 4 and head.max_k == 72),
        "max_k": int(head.max_k),
        "proxy_mse": float(res["proxy_mse"]),
        "substrate_mse": float(res["substrate_mse"]),
        "hidden_state_mse": float(res["hidden_state_mse"]),
        "attention_mse": float(res["attention_mse"]),
    }


def family_h90b_lhr_v10_attention_beats_substrate_on_aligned(
        seed: int) -> dict[str, Any]:
    """Constructive: when the target is the V10 attention head's
    output, the attention head has lower MSE than the substrate
    head (because the substrate head doesn't see the attention
    contribution)."""
    head = LongHorizonReconstructionV10Head.init(
        seed=int(seed) + 20200)
    rng = _np.random.default_rng(int(seed) + 20201)
    carriers: list[list[float]] = []
    targets: list[list[float]] = []
    subs: list[list[float]] = []
    hids: list[list[float]] = []
    atts: list[list[float]] = []
    for _ in range(4):
        c = rng.standard_normal(8).tolist()
        s = rng.standard_normal(8).tolist()
        h = rng.standard_normal(8).tolist()
        a = rng.standard_normal(8).tolist()
        # Target is exactly the attention head's prediction.
        tgt = head.attention_conditioned_value(
            carrier=c, k=4, attention_state=a,
            hidden_state=h, substrate_state=s)
        carriers.append(c)
        targets.append(tgt)
        subs.append(s)
        hids.append(h)
        atts.append(a)
    res = evaluate_lhr_v10_four_way(
        head, carrier_examples=carriers,
        target_examples=targets,
        substrate_states=subs, hidden_states=hids,
        attention_states=atts, k=4)
    return {
        "schema": R120_SCHEMA_VERSION,
        "name": "h90b_lhr_v10_attention_beats_substrate_on_aligned",
        "passed": bool(
            res["attention_mse"] <= res["substrate_mse"] + 1e-9),
        "attention_mse": float(res["attention_mse"]),
        "substrate_mse": float(res["substrate_mse"]),
    }


def family_h91_mlsc_v6_attention_chain_inheritance(
        seed: int) -> dict[str, Any]:
    op = MergeOperatorV6(factor_dim=6)
    c1_v3 = make_root_capsule_v3(
        branch_id="b1", payload=(0.1,) * 6,
        fact_tags=("t",), confidence=0.9, trust=0.9,
        turn_index=0)
    c2_v3 = make_root_capsule_v3(
        branch_id="b2", payload=(0.2,) * 6,
        fact_tags=("t",), confidence=0.85, trust=0.85,
        turn_index=0)
    v4a = wrap_v3_as_v4(c1_v3, substrate_witness_cid="sa")
    v4b = wrap_v3_as_v4(c2_v3, substrate_witness_cid="sb")
    v5a = wrap_v4_as_v5(
        v4a, hidden_state_witness_chain=("h1",),
        attention_witness_cid="ax1")
    v5b = wrap_v4_as_v5(
        v4b, hidden_state_witness_chain=("h2",),
        attention_witness_cid="ax2")
    v6a = wrap_v5_as_v6(
        v5a, attention_witness_chain=("a1", "a2"),
        cache_reuse_witness_cid="c1")
    v6b = wrap_v5_as_v6(
        v5b, attention_witness_chain=("a2", "a3"),
        cache_reuse_witness_cid="c2")
    merged = op.merge([v6a, v6b])
    chain = list(merged.attention_witness_chain)
    return {
        "schema": R120_SCHEMA_VERSION,
        "name": "h91_mlsc_v6_attention_chain_inheritance",
        "passed": bool(
            "a1" in chain and "a2" in chain and "a3" in chain),
        "chain": list(chain),
    }


def family_h91b_mlsc_v6_cache_reuse_witness(
        seed: int) -> dict[str, Any]:
    op = MergeOperatorV6(factor_dim=6)
    c1_v3 = make_root_capsule_v3(
        branch_id="b1", payload=(0.1,) * 6,
        fact_tags=("t",), confidence=0.9, trust=0.9,
        turn_index=0)
    v4a = wrap_v3_as_v4(c1_v3)
    v5a = wrap_v4_as_v5(v4a)
    v6a = wrap_v5_as_v6(
        v5a, cache_reuse_witness_cid="cache_xyz")
    merged = op.merge(
        [v6a], cache_reuse_witness_cid="cache_merge")
    return {
        "schema": R120_SCHEMA_VERSION,
        "name": "h91b_mlsc_v6_cache_reuse_witness",
        "passed": bool(
            merged.cache_reuse_witness_cid == "cache_merge"),
        "actual": str(merged.cache_reuse_witness_cid),
    }


def family_h97a_ecc_v10_bits_per_token(
        seed: int) -> dict[str, Any]:
    cb = ECCCodebookV10.init(seed=int(seed) + 20300)
    gate = QuantisedBudgetGate.init()
    gate.importance_threshold = 0.0
    gate.w_emit.values = [1.0] * len(gate.w_emit.values)
    rng = _np.random.default_rng(int(seed) + 20301)
    carrier = rng.standard_normal(W53_DEFAULT_ECC_CODE_DIM).tolist()
    comp = compress_carrier_ecc_v10(
        carrier, codebook=cb, gate=gate)
    return {
        "schema": R120_SCHEMA_VERSION,
        "name": "h97a_ecc_v10_bits_per_token",
        "passed": bool(
            float(comp["bits_per_visible_token"]) >= 21.0),
        "bits_per_visible_token": float(
            comp["bits_per_visible_token"]),
    }


def family_h97b_ecc_v10_total_codes(
        seed: int) -> dict[str, Any]:
    cb = ECCCodebookV10.init(seed=int(seed) + 20310)
    return {
        "schema": R120_SCHEMA_VERSION,
        "name": "h97b_ecc_v10_total_codes",
        "passed": bool(cb.total_codes == 524288),
        "total_codes": int(cb.total_codes),
    }


def family_h95_ecc_v10_rate_floor_falsifier(
        seed: int) -> dict[str, Any]:
    cb = ECCCodebookV10.init(seed=int(seed) + 20320)
    res = probe_ecc_v10_rate_floor_falsifier(
        codebook=cb, target_bits_per_token=1024.0)
    return {
        "schema": R120_SCHEMA_VERSION,
        "name": "h95_ecc_v10_rate_floor_falsifier",
        "passed": bool(res["target_above_info_bound"]),
        "info_bound": float(res["info_bound"]),
    }


def family_h101_v10_chain_walk_512(
        seed: int) -> dict[str, Any]:
    cell = V10StackedCell.init(
        state_dim=8, input_dim=8, n_layers=8,
        seed=int(seed) + 20400)
    chain = PersistentLatentStateV10Chain.empty()
    prev = None
    for i in range(512):
        state = step_persistent_state_v10(
            cell=cell, prev_state=prev,
            carrier_values=[float(i % 7 - 3)] * 8,
            turn_index=i, role="r")
        chain.add(state)
        prev = state
    walked = chain.walk_from(prev.cid())
    return {
        "schema": R120_SCHEMA_VERSION,
        "name": "h101_v10_chain_walk_512",
        "passed": bool(
            len(walked) == 512 and
            cell.n_layers == 8),
        "achieved_depth": int(len(walked)),
        "n_layers": int(cell.n_layers),
    }


def family_h101b_v10_8_layer_cell(
        seed: int) -> dict[str, Any]:
    cell = V10StackedCell.init(
        state_dim=8, input_dim=8, n_layers=8,
        seed=int(seed) + 20410)
    return {
        "schema": R120_SCHEMA_VERSION,
        "name": "h101b_v10_8_layer_cell",
        "passed": bool(cell.n_layers == 8),
        "n_layers": int(cell.n_layers),
    }


def family_h101c_v10_attention_fidelity_damps(
        seed: int) -> dict[str, Any]:
    """attention_fidelity=0 with attention_skip=non-zero must
    produce a different top state than attention_fidelity=1."""
    cell = V10StackedCell.init(
        state_dim=8, input_dim=8, n_layers=8,
        seed=int(seed) + 20420)
    state_full = step_persistent_state_v10(
        cell=cell, prev_state=None,
        carrier_values=[0.3] * 8,
        turn_index=0, role="r",
        attention_skip=[1.0] * 8,
        attention_fidelity=1.0)
    state_zero = step_persistent_state_v10(
        cell=cell, prev_state=None,
        carrier_values=[0.3] * 8,
        turn_index=0, role="r",
        attention_skip=[1.0] * 8,
        attention_fidelity=0.0)
    return {
        "schema": R120_SCHEMA_VERSION,
        "name": "h101c_v10_attention_fidelity_damps",
        "passed": bool(
            state_full.top_state != state_zero.top_state),
        "max_abs_diff": float(max(
            abs(a - b) for a, b in zip(
                state_full.top_state, state_zero.top_state))),
    }


def family_h101d_v10_carrier_round_trip_deterministic(
        seed: int) -> dict[str, Any]:
    """Same inputs to step_persistent_state_v10 produce byte-
    identical CIDs."""
    cell = V10StackedCell.init(
        state_dim=8, input_dim=8, n_layers=8,
        seed=int(seed) + 20430)
    s1 = step_persistent_state_v10(
        cell=cell, prev_state=None,
        carrier_values=[0.2] * 8,
        turn_index=0, role="r",
        substrate_skip=[0.1] * 8,
        hidden_state_skip=[0.05] * 8,
        attention_skip=[0.02] * 8)
    s2 = step_persistent_state_v10(
        cell=cell, prev_state=None,
        carrier_values=[0.2] * 8,
        turn_index=0, role="r",
        substrate_skip=[0.1] * 8,
        hidden_state_skip=[0.05] * 8,
        attention_skip=[0.02] * 8)
    return {
        "schema": R120_SCHEMA_VERSION,
        "name": "h101d_v10_carrier_round_trip_deterministic",
        "passed": bool(s1.cid() == s2.cid()),
        "cid_a": str(s1.cid()),
        "cid_b": str(s2.cid()),
    }


def family_h102_multi_hop_v8_chain_len_11(
        seed: int) -> dict[str, Any]:
    w = evaluate_dec_chain_len11_fidelity(
        seed=int(seed) + 20500)
    return {
        "schema": R120_SCHEMA_VERSION,
        "name": "h102_multi_hop_v8_chain_len_11",
        "passed": bool(
            w.chain_length == 11
            and w.n_edges == 132
            and len(w.backends) == 12),
        "chain_length": int(w.chain_length),
        "n_edges": int(w.n_edges),
        "n_backends": int(len(w.backends)),
    }


R120_FAMILIES: tuple[tuple[str, Any], ...] = (
    ("h90_lhr_v10_four_way_runs",
     family_h90_lhr_v10_four_way_runs),
    ("h90b_lhr_v10_attention_beats_substrate_on_aligned",
     family_h90b_lhr_v10_attention_beats_substrate_on_aligned),
    ("h91_mlsc_v6_attention_chain_inheritance",
     family_h91_mlsc_v6_attention_chain_inheritance),
    ("h91b_mlsc_v6_cache_reuse_witness",
     family_h91b_mlsc_v6_cache_reuse_witness),
    ("h97a_ecc_v10_bits_per_token",
     family_h97a_ecc_v10_bits_per_token),
    ("h97b_ecc_v10_total_codes",
     family_h97b_ecc_v10_total_codes),
    ("h95_ecc_v10_rate_floor_falsifier",
     family_h95_ecc_v10_rate_floor_falsifier),
    ("h101_v10_chain_walk_512",
     family_h101_v10_chain_walk_512),
    ("h101b_v10_8_layer_cell",
     family_h101b_v10_8_layer_cell),
    ("h101c_v10_attention_fidelity_damps",
     family_h101c_v10_attention_fidelity_damps),
    ("h101d_v10_carrier_round_trip_deterministic",
     family_h101d_v10_carrier_round_trip_deterministic),
    ("h102_multi_hop_v8_chain_len_11",
     family_h102_multi_hop_v8_chain_len_11),
)


def run_r120(*, seeds: Sequence[int] = (0, 1, 2)) -> dict[str, Any]:
    rows: list[R120SeedResult] = []
    for s in seeds:
        results: dict[str, dict[str, Any]] = {}
        for name, fn in R120_FAMILIES:
            results[name] = fn(int(s))
        rows.append(R120SeedResult(
            seed=int(s), family_results=results))
    summary = {
        "schema": R120_SCHEMA_VERSION,
        "n_seeds": int(len(seeds)),
        "seeds": [r.to_dict() for r in rows],
    }
    pass_counts: dict[str, int] = {}
    for r in rows:
        for k, v in r.family_results.items():
            if bool(v.get("passed", False)):
                pass_counts[k] = pass_counts.get(k, 0) + 1
    summary["pass_counts"] = pass_counts
    summary["all_passed"] = bool(all(
        pass_counts.get(name, 0) == len(seeds)
        for name, _ in R120_FAMILIES))
    return summary


__all__ = [
    "R120_SCHEMA_VERSION",
    "R120_FAMILIES",
    "R120SeedResult",
    "run_r120",
]
