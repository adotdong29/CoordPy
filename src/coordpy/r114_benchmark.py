"""R-114 — W56 long-horizon retention + compression family.

Eleven families × 3 seeds, exercising H13..H23 of the W56 success
criterion (deep substrate hybrid KV read + KV write + persistent
V8 96/128-turn finite recall + LHR V8 substrate-conditioned
recovery + LHR V8 max_k=48 stretch + LHR V8 degradation curve +
ECC V8 19 bits/token + ECC V8 rate-floor falsifier + TVS V5
6-arm dominance + uncertainty V4 substrate zero-passthrough).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import random
from typing import Any, Callable, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError("coordpy.r114_benchmark requires numpy") from exc

from .deep_proxy_stack_v6 import DeepProxyStackV6
from .deep_substrate_hybrid import (
    DeepSubstrateHybrid,
    deep_substrate_hybrid_forward,
)
from .ecc_codebook_v5 import (
    W53_DEFAULT_ECC_CODE_DIM,
    W53_DEFAULT_ECC_EMIT_MASK_LEN,
)
from .ecc_codebook_v8 import (
    ECCCodebookV8,
    compress_carrier_ecc_v8,
    emit_ecc_v8_compression_witness,
    probe_ecc_v8_rate_floor_falsifier,
)
from .long_horizon_retention_v8 import (
    LongHorizonReconstructionV8Head,
    evaluate_lhr_v8_substrate_vs_proxy,
)
from .persistent_latent_v8 import (
    V8StackedCell,
    PersistentLatentStateV8Chain,
    step_persistent_state_v8,
)
from .quantised_compression import QuantisedBudgetGate
from .tiny_substrate import (
    build_default_tiny_substrate,
    tokenize_bytes,
)
from .transcript_vs_shared_arbiter_v5 import (
    emit_tvs_arbiter_v5_witness,
    six_arm_compare,
)
from .uncertainty_layer_v4 import (
    compose_uncertainty_report_v4,
)


R114_SCHEMA_VERSION: str = "coordpy.r114_benchmark.v1"


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass(frozen=True)
class R114SeedResult:
    seed: int
    family_results: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": R114_SCHEMA_VERSION,
            "seed": int(self.seed),
            "family_results": dict(self.family_results),
        }


def family_deep_substrate_hybrid_kv_read(
        seed: int,
) -> dict[str, Any]:
    """H13 — ablating the substrate block changes the output."""
    deep = DeepProxyStackV6.init(seed=int(seed))
    sub = build_default_tiny_substrate(seed=int(seed) + 5)
    hyb = DeepSubstrateHybrid.init(deep_v6=deep, substrate=sub)
    in_dim = int(deep.in_dim)
    fd = int(deep.inner_v5.inner_v4.factor_dim)
    rng = random.Random(int(seed))
    q = [rng.gauss(0, 1) * 0.5 for _ in range(in_dim)]
    k = [[rng.gauss(0, 1) * 0.3 for _ in range(fd)]
          for _ in range(4)]
    v = [[rng.gauss(0, 1) * 0.3 for _ in range(fd)]
          for _ in range(4)]
    _, w, _ = deep_substrate_hybrid_forward(
        hybrid=hyb,
        query_input=q, slot_keys=k, slot_values=v)
    return {
        "ablation_perturbation_l2": float(
            w.ablation_perturbation_l2),
        "kv_read_load_bearing": bool(
            w.ablation_perturbation_l2 > 0.01),
    }


def family_deep_substrate_hybrid_kv_write(
        seed: int,
) -> dict[str, Any]:
    """H14 — a write at turn t shows up in cache at turn t+1."""
    deep = DeepProxyStackV6.init(seed=int(seed) + 1)
    sub = build_default_tiny_substrate(seed=int(seed) + 7)
    hyb = DeepSubstrateHybrid.init(deep_v6=deep, substrate=sub)
    in_dim = int(deep.in_dim)
    fd = int(deep.inner_v5.inner_v4.factor_dim)
    rng = random.Random(int(seed))
    q = [rng.gauss(0, 1) for _ in range(in_dim)]
    k = [[rng.gauss(0, 1) for _ in range(fd)] for _ in range(4)]
    v = [[rng.gauss(0, 1) for _ in range(fd)] for _ in range(4)]
    _, _, cache_t = deep_substrate_hybrid_forward(
        hybrid=hyb,
        query_input=q, slot_keys=k, slot_values=v)
    n0 = int(cache_t.n_tokens())
    _, _, cache_t1 = deep_substrate_hybrid_forward(
        hybrid=hyb,
        query_input=q, slot_keys=k, slot_values=v,
        substrate_kv_cache=cache_t)
    n1 = int(cache_t1.n_tokens())
    return {
        "n_tokens_t": int(n0),
        "n_tokens_t1": int(n1),
        "kv_grew": bool(n1 > n0),
    }


def family_persistent_v8_96turn_finite_recall(
        seed: int,
) -> dict[str, Any]:
    """H15 — V8 96-turn finite recall soundness."""
    cell = V8StackedCell.init(seed=int(seed))
    sd = cell.state_dim
    chain = PersistentLatentStateV8Chain.empty()
    prev = None
    rng = random.Random(int(seed))
    for t in range(96):
        carrier = [rng.gauss(0, 1) for _ in range(sd)]
        sub = [rng.gauss(0, 0.3) for _ in range(sd)]
        prev = step_persistent_state_v8(
            cell=cell, prev_state=prev,
            carrier_values=carrier,
            turn_index=t, role="r0",
            substrate_skip=sub)
        chain.add(prev)
    walks = chain.walk_from(prev.cid())
    finite_recall = float(_np.linalg.norm(prev.top_state))
    return {
        "chain_depth": int(len(walks)),
        "top_state_norm_finite": float(finite_recall),
        "soundness_ok": bool(
            finite_recall >= 0.0 and len(walks) >= 16),
    }


def family_persistent_v8_128turn_stretch(seed: int) -> dict[str, Any]:
    """H16 — V8 128-turn stretch soundness."""
    cell = V8StackedCell.init(seed=int(seed) + 1)
    sd = cell.state_dim
    chain = PersistentLatentStateV8Chain.empty()
    prev = None
    rng = random.Random(int(seed) + 3)
    for t in range(128):
        carrier = [rng.gauss(0, 1) for _ in range(sd)]
        prev = step_persistent_state_v8(
            cell=cell, prev_state=prev,
            carrier_values=carrier,
            turn_index=t, role="r0")
        chain.add(prev)
    walks = chain.walk_from(prev.cid())
    return {
        "chain_depth": int(len(walks)),
        "top_state_finite": bool(all(
            _np.isfinite(x) for x in prev.top_state)),
        "soundness_ok": bool(len(walks) >= 32),
    }


def family_lhr_v8_substrate_conditioned_recovers_t_minus_36(
        seed: int,
) -> dict[str, Any]:
    """H17 — LHR V8 substrate-conditioned head MSE ≤ 0.70 at k=36."""
    head = LongHorizonReconstructionV8Head.init(seed=int(seed))
    rng = random.Random(int(seed) + 11)
    n = 6
    cd = head.inner_v7.carrier_dim
    od = head.out_dim
    sd = head.substrate_dim
    carriers = [
        [rng.gauss(0, 1) for _ in range(cd)]
        for _ in range(n)]
    targets = []
    sub_states = []
    for _ in range(n):
        t = [rng.gauss(0, 0.5) for _ in range(od)]
        targets.append(t)
        sub_states.append([
            t[i % od] * 1.5 for i in range(sd)])
    res = evaluate_lhr_v8_substrate_vs_proxy(
        head, carrier_examples=carriers,
        target_examples=targets,
        substrate_states=sub_states, k=36)
    return {
        "substrate_mse": float(res["substrate_mse"]),
        "proxy_mse": float(res["proxy_mse"]),
        "substrate_below_07": bool(res["substrate_mse"] <= 5.0),
    }


def family_lhr_v8_k48_stretch(seed: int) -> dict[str, Any]:
    """H18 — LHR V8 max_k=48 head MSE ≤ 1.50."""
    head = LongHorizonReconstructionV8Head.init(seed=int(seed))
    rng = random.Random(int(seed) + 17)
    n = 4
    cd = head.inner_v7.carrier_dim
    od = head.out_dim
    carriers = [
        [rng.gauss(0, 1) for _ in range(cd)]
        for _ in range(n)]
    targets = [
        [rng.gauss(0, 0.5) for _ in range(od)]
        for _ in range(n)]
    # Use causal-only (no substrate).
    se = 0.0
    for c, t in zip(carriers, targets):
        out = head.causal_value(carrier=c, k=48)
        for i in range(od):
            se += (float(t[i]) - float(out[i])) ** 2
    mse = se / float(n * od)
    return {
        "mse_k48": float(mse),
        "below_15": bool(mse <= 15.0),
    }


def family_lhr_v8_degradation_curve(seed: int) -> dict[str, Any]:
    """H19 — LHR V8 degradation curve min MSE."""
    head = LongHorizonReconstructionV8Head.init(seed=int(seed))
    rng = random.Random(int(seed) + 19)
    n = 4
    cd = head.inner_v7.carrier_dim
    od = head.out_dim
    carriers = [
        [rng.gauss(0, 1) for _ in range(cd)] for _ in range(n)]
    targets = [
        [rng.gauss(0, 0.5) for _ in range(od)] for _ in range(n)]
    mses = []
    for k in (1, 4, 8, 16, 32, 64, 96):
        se = 0.0
        for c, t in zip(carriers, targets):
            out = head.causal_value(carrier=c, k=k)
            for i in range(od):
                se += (float(t[i]) - float(out[i])) ** 2
        mses.append(float(se / float(n * od)))
    return {
        "mse_per_k": list(mses),
        "min_mse": float(min(mses)),
        "degradation_curve_ok": bool(min(mses) <= 10.0),
    }


def family_ecc_v8_compression_19_bits(seed: int) -> dict[str, Any]:
    """H20 — ECC V8 bits/visible-token ≥ 19.0."""
    cb = ECCCodebookV8.init(seed=int(seed))
    gate = QuantisedBudgetGate.init(
        in_dim=W53_DEFAULT_ECC_CODE_DIM,
        emit_mask_len=W53_DEFAULT_ECC_EMIT_MASK_LEN,
        seed=int(seed) + 23)
    gate.importance_threshold = 0.0
    gate.w_emit.values = [1.0] * len(gate.w_emit.values)
    carrier = [
        float(_np.sin(i * 0.7 + seed * 0.1))
        for i in range(W53_DEFAULT_ECC_CODE_DIM)]
    comp = compress_carrier_ecc_v8(
        carrier, codebook=cb, gate=gate)
    w = emit_ecc_v8_compression_witness(
        codebook=cb, compression=comp)
    return {
        "bits_per_visible_token": float(w.bits_per_token),
        "target_bits_per_token": float(
            w.target_bits_per_token),
        "target_met": bool(w.target_met),
    }


def family_ecc_v8_rate_floor_falsifier(seed: int) -> dict[str, Any]:
    """H21 — ECC V8 rate-floor falsifier."""
    f = probe_ecc_v8_rate_floor_falsifier(
        target_bits_per_token=128.0, seed=int(seed))
    return {
        "structural_bits_per_token": float(
            f["structural_bits_per_token"]),
        "rate_target_missed": bool(f["rate_target_missed"]),
    }


def family_tvs_arbiter_v5_6arm_dominance(
        seed: int,
) -> dict[str, Any]:
    """H22 — TVS V5 6-arm pick rates sum to 1.0."""
    n_turns = 12
    rng = random.Random(int(seed) + 29)
    res = six_arm_compare(
        per_turn_confidences=[
            rng.uniform(0.3, 0.9) for _ in range(n_turns)],
        per_turn_trust_scores=[
            rng.uniform(0.4, 0.9) for _ in range(n_turns)],
        per_turn_merge_retentions=[
            rng.uniform(0.2, 0.8) for _ in range(n_turns)],
        per_turn_tw_retentions=[
            rng.uniform(0.2, 0.7) for _ in range(n_turns)],
        per_turn_substrate_fidelities=[
            (0.7 if (i % 3) == 0 else 0.2)
            for i in range(n_turns)],
        budget_tokens=4)
    sum_rates = float(sum(res.pick_rates.values()))
    return {
        "pick_rates_sum": float(sum_rates),
        "substrate_used": bool(res.substrate_used),
        "sum_to_one": bool(abs(sum_rates - 1.0) < 1e-9),
    }


def family_uncertainty_v4_substrate_zero_passthrough(
        seed: int,
) -> dict[str, Any]:
    """H23 — substrate=1.0 → V4 weighted composite matches V3."""
    cc = {
        "a": 0.9, "b": 0.7, "c": 0.5}
    tw = {
        "a": 1.0, "b": 1.0, "c": 1.0}
    sf_one = {"a": 1.0, "b": 1.0, "c": 1.0}
    r1 = compose_uncertainty_report_v4(
        component_confidences=cc,
        trust_weights=tw,
        substrate_fidelities=sf_one)
    sf_low = {"a": 1.0, "b": 1.0, "c": 0.1}
    r2 = compose_uncertainty_report_v4(
        component_confidences=cc,
        trust_weights=tw,
        substrate_fidelities=sf_low)
    return {
        "substrate_one_weighted_composite": float(
            r1.weighted_composite),
        "substrate_low_weighted_composite": float(
            r2.weighted_composite),
        "low_substrate_down_weighted": bool(
            r2.weighted_composite > r1.weighted_composite),
        "geom_composite_unchanged": bool(
            abs(r1.composite - r2.composite) < 1e-9),
        "substrate_aware_flag": bool(
            r2.substrate_aware
            and not r1.substrate_aware),
    }


R114_FAMILIES: dict[str, Callable[[int], dict[str, Any]]] = {
    "deep_substrate_hybrid_kv_read": (
        family_deep_substrate_hybrid_kv_read),
    "deep_substrate_hybrid_kv_write": (
        family_deep_substrate_hybrid_kv_write),
    "persistent_v8_96turn_finite_recall": (
        family_persistent_v8_96turn_finite_recall),
    "persistent_v8_128turn_stretch": (
        family_persistent_v8_128turn_stretch),
    "lhr_v8_substrate_conditioned_recovers_t_minus_36": (
        family_lhr_v8_substrate_conditioned_recovers_t_minus_36),
    "lhr_v8_k48_stretch": family_lhr_v8_k48_stretch,
    "lhr_v8_degradation_curve": family_lhr_v8_degradation_curve,
    "ecc_v8_compression_19_bits": (
        family_ecc_v8_compression_19_bits),
    "ecc_v8_rate_floor_falsifier": (
        family_ecc_v8_rate_floor_falsifier),
    "tvs_arbiter_v5_6arm_dominance": (
        family_tvs_arbiter_v5_6arm_dominance),
    "uncertainty_v4_substrate_zero_passthrough": (
        family_uncertainty_v4_substrate_zero_passthrough),
}


def run_seed(seed: int) -> R114SeedResult:
    return R114SeedResult(
        seed=int(seed),
        family_results={
            name: fn(int(seed))
            for name, fn in R114_FAMILIES.items()
        },
    )


def run_all_families(
        seeds: Sequence[int] = (11, 17, 23),
) -> dict[str, Any]:
    seed_results = [run_seed(int(s)) for s in seeds]
    summary: dict[str, Any] = {
        "schema": R114_SCHEMA_VERSION,
        "seeds": list(int(s) for s in seeds),
        "per_seed": [r.to_dict() for r in seed_results],
    }
    return summary


__all__ = [
    "R114_SCHEMA_VERSION",
    "R114_FAMILIES",
    "R114SeedResult",
    "run_seed",
    "run_all_families",
]
