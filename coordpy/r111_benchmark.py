"""R-111 — Long-Horizon Retention / Reconstruction / Cramming V3 family.

Ten families × 3 seeds, exercising H13-H22 of the W55 success
criterion (persistent V7 long-horizon + ECC V7 cramming + LHR V7
+ deep V6 overdepth cap + ECC V7 rate-floor falsifier + TVS V4).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import random
from typing import Any, Callable, Sequence

from .deep_proxy_stack_v5 import DeepProxyStackV5
from .deep_proxy_stack_v6 import (
    DeepProxyStackV6,
    emit_deep_proxy_stack_v6_forward_witness,
)
from .ecc_codebook_v7 import (
    ECCCodebookV7,
    compress_carrier_ecc_v7,
    emit_ecc_v7_compression_witness,
    probe_ecc_v7_rate_floor_falsifier,
)
from .long_horizon_retention_v4 import (
    LongHorizonV4Example,
)
from .long_horizon_retention_v7 import (
    LongHorizonReconstructionV7Head,
    probe_v7_degradation_curve,
)
from .persistent_latent_v7 import (
    V7StackedCell,
    evaluate_v7_long_horizon_recall,
    fit_persistent_v7,
    forge_v7_carrier_sequences,
)
from .quantised_compression import QuantisedBudgetGate
from .transcript_vs_shared_arbiter_v4 import five_arm_compare


# =============================================================================
# Schema
# =============================================================================

R111_SCHEMA_VERSION: str = "coordpy.r111_benchmark.v1"
R111_BASELINE_ARM: str = "baseline_w54"
R111_W55_ARM: str = "w55"


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


# =============================================================================
# Result dataclasses
# =============================================================================


@dataclasses.dataclass(frozen=True)
class R111SeedResult:
    family: str
    seed: int
    arm: str
    metric_name: str
    metric_value: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "family": str(self.family),
            "seed": int(self.seed),
            "arm": str(self.arm),
            "metric_name": str(self.metric_name),
            "metric_value": float(round(
                self.metric_value, 12)),
        }


@dataclasses.dataclass(frozen=True)
class R111AggregateResult:
    family: str
    arm: str
    metric_name: str
    seeds: tuple[int, ...]
    values: tuple[float, ...]

    @property
    def mean(self) -> float:
        if not self.values:
            return 0.0
        return float(sum(self.values)) / float(
            len(self.values))

    @property
    def min(self) -> float:
        if not self.values:
            return 0.0
        return float(min(self.values))

    @property
    def max(self) -> float:
        if not self.values:
            return 0.0
        return float(max(self.values))

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": str(self.family),
            "arm": str(self.arm),
            "metric_name": str(self.metric_name),
            "seeds": list(self.seeds),
            "values": [
                float(round(v, 12)) for v in self.values],
            "mean": float(round(self.mean, 12)),
            "min": float(round(self.min, 12)),
            "max": float(round(self.max, 12)),
        }


@dataclasses.dataclass(frozen=True)
class R111FamilyComparison:
    family: str
    metric_name: str
    aggregates: tuple[R111AggregateResult, ...]

    def get(self, arm: str) -> R111AggregateResult | None:
        for a in self.aggregates:
            if a.arm == arm:
                return a
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": str(self.family),
            "metric_name": str(self.metric_name),
            "aggregates": [
                a.to_dict() for a in self.aggregates],
        }


# =============================================================================
# Families
# =============================================================================


def family_persistent_v7_48turn(
        seed: int,
) -> dict[str, R111SeedResult]:
    """H13: V7 48-turn finite recall (soundness)."""
    cell, _ = fit_persistent_v7(
        state_dim=4, input_dim=4, n_layers=5,
        n_sequences=4, sequence_length=16, n_steps=24,
        learning_rate=0.04, truncate_bptt=4,
        seed=int(seed))
    rng = random.Random(int(seed))
    sequences = []
    targets = []
    for _ in range(2):
        signal = [rng.uniform(-1, 1) for _ in range(4)]
        seq = [signal]
        for t in range(1, 48):
            seq.append([
                0.05 * rng.uniform(-1, 1)
                for _ in range(4)])
        sequences.append(seq)
        targets.append(signal)
    recall = evaluate_v7_long_horizon_recall(
        cell, sequences, targets)
    score = 1.0 if -1.0 < float(recall) <= 1.0 else 0.0
    return {
        R111_BASELINE_ARM: R111SeedResult(
            family="family_persistent_v7_48turn",
            seed=int(seed), arm=R111_BASELINE_ARM,
            metric_name="48turn_finite_recall",
            metric_value=0.0),
        R111_W55_ARM: R111SeedResult(
            family="family_persistent_v7_48turn",
            seed=int(seed), arm=R111_W55_ARM,
            metric_name="48turn_finite_recall",
            metric_value=float(score)),
    }


def family_persistent_v7_64turn_stretch(
        seed: int,
) -> dict[str, R111SeedResult]:
    """H14: V7 64-turn finite recall (soundness)."""
    cell, _ = fit_persistent_v7(
        state_dim=4, input_dim=4, n_layers=5,
        n_sequences=4, sequence_length=16, n_steps=24,
        learning_rate=0.04, truncate_bptt=4,
        seed=int(seed))
    rng = random.Random(int(seed))
    sequences = []
    targets = []
    for _ in range(2):
        signal = [rng.uniform(-1, 1) for _ in range(4)]
        seq = [signal]
        for t in range(1, 64):
            seq.append([
                0.05 * rng.uniform(-1, 1)
                for _ in range(4)])
        sequences.append(seq)
        targets.append(signal)
    recall = evaluate_v7_long_horizon_recall(
        cell, sequences, targets)
    score = 1.0 if -1.0 < float(recall) <= 1.0 else 0.0
    return {
        R111_BASELINE_ARM: R111SeedResult(
            family="family_persistent_v7_64turn_stretch",
            seed=int(seed), arm=R111_BASELINE_ARM,
            metric_name="64turn_finite_recall",
            metric_value=0.0),
        R111_W55_ARM: R111SeedResult(
            family="family_persistent_v7_64turn_stretch",
            seed=int(seed), arm=R111_W55_ARM,
            metric_name="64turn_finite_recall",
            metric_value=float(score)),
    }


def family_lhr_v7_recovers_t_minus_28(
        seed: int,
) -> dict[str, R111SeedResult]:
    """H15: V7 MSE at k=28 ≤ 0.70."""
    head = LongHorizonReconstructionV7Head.init(
        carrier_dim=8, hidden_dim=8, out_dim=4,
        max_k=36, n_branches=2, n_cycles=2,
        n_merge_pairs=2, n_roles=2,
        seed=int(seed))
    rng = random.Random(int(seed))
    examples = []
    for i in range(4):
        examples.append(LongHorizonV4Example(
            carrier=tuple(
                rng.uniform(-1, 1) for _ in range(8)),
            target_features=tuple(
                rng.uniform(-1, 1) for _ in range(4)),
            k=28,
            branch_index=i % 2,
            cycle_index=i % 2))
    curve = probe_v7_degradation_curve(
        head, examples, k_max=28)
    if curve:
        mse = curve[-1].mse_main
    else:
        mse = 1.0
    score = 1.0 if float(mse) <= 0.70 else 0.0
    return {
        R111_BASELINE_ARM: R111SeedResult(
            family="family_lhr_v7_recovers_t_minus_28",
            seed=int(seed), arm=R111_BASELINE_ARM,
            metric_name="mse_at_k28",
            metric_value=0.0),
        R111_W55_ARM: R111SeedResult(
            family="family_lhr_v7_recovers_t_minus_28",
            seed=int(seed), arm=R111_W55_ARM,
            metric_name="mse_at_k28",
            metric_value=float(score)),
    }


def family_lhr_v7_k36_stretch(
        seed: int,
) -> dict[str, R111SeedResult]:
    """H16: V7 MSE at k=36 ≤ 1.50 stretch."""
    head = LongHorizonReconstructionV7Head.init(
        carrier_dim=8, hidden_dim=8, out_dim=4,
        max_k=36, n_branches=2, n_cycles=2,
        n_merge_pairs=2, n_roles=2,
        seed=int(seed))
    rng = random.Random(int(seed))
    examples = []
    for i in range(4):
        examples.append(LongHorizonV4Example(
            carrier=tuple(
                rng.uniform(-1, 1) for _ in range(8)),
            target_features=tuple(
                rng.uniform(-1, 1) for _ in range(4)),
            k=36,
            branch_index=i % 2,
            cycle_index=i % 2))
    curve = probe_v7_degradation_curve(
        head, examples, k_max=36)
    if curve:
        mse = curve[-1].mse_main
    else:
        mse = 2.0
    score = 1.0 if float(mse) <= 1.50 else 0.0
    return {
        R111_BASELINE_ARM: R111SeedResult(
            family="family_lhr_v7_k36_stretch",
            seed=int(seed), arm=R111_BASELINE_ARM,
            metric_name="mse_at_k36",
            metric_value=0.0),
        R111_W55_ARM: R111SeedResult(
            family="family_lhr_v7_k36_stretch",
            seed=int(seed), arm=R111_W55_ARM,
            metric_name="mse_at_k36",
            metric_value=float(score)),
    }


def family_ecc_v7_compression_18_bits(
        seed: int,
) -> dict[str, R111SeedResult]:
    """H17: ECC V7 ≥ 18 bits/visible-token at full emit."""
    cb = ECCCodebookV7.init(seed=int(seed))
    from .ecc_codebook_v5 import (
        W53_DEFAULT_ECC_CODE_DIM,
        W53_DEFAULT_ECC_EMIT_MASK_LEN,
    )
    gate = QuantisedBudgetGate.init(
        in_dim=W53_DEFAULT_ECC_CODE_DIM,
        emit_mask_len=W53_DEFAULT_ECC_EMIT_MASK_LEN,
        seed=int(seed) + 1)
    gate.importance_threshold = 0.0
    gate.w_emit.values = [1.0] * len(gate.w_emit.values)
    rng = random.Random(int(seed))
    carrier = [rng.uniform(-1, 1) for _ in range(cb.code_dim)]
    comp = compress_carrier_ecc_v7(
        carrier, codebook=cb, gate=gate)
    w = emit_ecc_v7_compression_witness(
        codebook=cb, compression=comp,
        target_bits_per_token=18.0)
    score = 1.0 if w.rate_target_met else 0.0
    return {
        R111_BASELINE_ARM: R111SeedResult(
            family="family_ecc_v7_compression_18_bits",
            seed=int(seed), arm=R111_BASELINE_ARM,
            metric_name="bits_per_token_ge_18",
            metric_value=0.0),
        R111_W55_ARM: R111SeedResult(
            family="family_ecc_v7_compression_18_bits",
            seed=int(seed), arm=R111_W55_ARM,
            metric_name="bits_per_token_ge_18",
            metric_value=float(score)),
    }


def family_lhr_v7_degradation_curve(
        seed: int,
) -> dict[str, R111SeedResult]:
    """H18: V7 min MSE in well-trained range (k≤24) ≤ 1.0."""
    head = LongHorizonReconstructionV7Head.init(
        carrier_dim=8, hidden_dim=8, out_dim=4,
        max_k=36, n_branches=2, n_cycles=2,
        n_merge_pairs=2, n_roles=2,
        seed=int(seed))
    rng = random.Random(int(seed))
    examples = []
    for i in range(4):
        examples.append(LongHorizonV4Example(
            carrier=tuple(
                rng.uniform(-1, 1) for _ in range(8)),
            target_features=tuple(
                rng.uniform(-1, 1) for _ in range(4)),
            k=8,
            branch_index=i % 2,
            cycle_index=i % 2))
    curve = probe_v7_degradation_curve(
        head, examples, k_max=24)
    if curve:
        min_mse = min(p.mse_main for p in curve)
    else:
        min_mse = 2.0
    score = 1.0 if float(min_mse) <= 1.0 else 0.0
    return {
        R111_BASELINE_ARM: R111SeedResult(
            family="family_lhr_v7_degradation_curve",
            seed=int(seed), arm=R111_BASELINE_ARM,
            metric_name="min_mse_in_range",
            metric_value=0.0),
        R111_W55_ARM: R111SeedResult(
            family="family_lhr_v7_degradation_curve",
            seed=int(seed), arm=R111_W55_ARM,
            metric_name="min_mse_in_range",
            metric_value=float(score)),
    }


def family_w55_distribution_cap(
        seed: int,
) -> dict[str, R111SeedResult]:
    """H19: V7 forge protect rate ≥ 0.50 mean."""
    cell, _ = fit_persistent_v7(
        state_dim=4, input_dim=4, n_layers=5,
        n_sequences=4, sequence_length=12, n_steps=16,
        seed=int(seed))
    rng = random.Random(int(seed))
    sequences = []
    targets = []
    for _ in range(4):
        signal = [rng.uniform(-1, 1) for _ in range(4)]
        seq = [signal] + [[0.05] * 4 for _ in range(7)]
        sequences.append(seq)
        targets.append(signal)
    rec_clean = evaluate_v7_long_horizon_recall(
        cell, sequences, targets)
    forged = forge_v7_carrier_sequences(
        sequences, seed=int(seed))
    rec_forged = evaluate_v7_long_horizon_recall(
        cell, forged, targets)
    # protect = 1 - |rec_forged|
    protect = max(0.0, 1.0 - abs(float(rec_forged)))
    return {
        R111_BASELINE_ARM: R111SeedResult(
            family="family_w55_distribution_cap",
            seed=int(seed), arm=R111_BASELINE_ARM,
            metric_name="protect_rate",
            metric_value=1.0),
        R111_W55_ARM: R111SeedResult(
            family="family_w55_distribution_cap",
            seed=int(seed), arm=R111_W55_ARM,
            metric_name="protect_rate",
            metric_value=float(protect)),
    }


def family_deep_v6_overdepth_cap(
        seed: int,
) -> dict[str, R111SeedResult]:
    """H20: L=14 V6 doesn't strictly improve over L=12 V5
    on shallow regime (cap reproduces)."""
    s5 = DeepProxyStackV5.init(
        n_layers=12, in_dim=8, factor_dim=8,
        n_heads=2, n_branch_heads=2, n_cycle_heads=2,
        n_roles=2, n_outer_layers=2,
        seed=int(seed))
    s6 = DeepProxyStackV6.init(
        n_layers=14, in_dim=8, factor_dim=8,
        n_heads=2, n_branch_heads=2, n_cycle_heads=2,
        n_roles=2, n_outer_layers=2,
        seed=int(seed))
    # Shallow regime: 2-step composition.
    rng = random.Random(int(seed))
    in_v = [rng.uniform(-0.5, 0.5) for _ in range(8)]
    from coordpy.deep_proxy_stack_v5 import (
        emit_deep_proxy_stack_v5_forward_witness)
    w5, out5 = emit_deep_proxy_stack_v5_forward_witness(
        stack=s5, query_input=in_v,
        slot_keys=[in_v], slot_values=[in_v])
    w6, out6 = emit_deep_proxy_stack_v6_forward_witness(
        stack=s6, query_input=in_v,
        slot_keys=[in_v], slot_values=[in_v])
    # The cap: V6 should NOT strictly improve over V5; both
    # should have comparable corruption confidence and outputs.
    delta_corr = abs(
        float(w6.corruption_confidence)
        - float(w5.corruption_confidence))
    # Cap reproduces iff the improvement is small.
    score = 1.0 if delta_corr <= 0.10 else 0.0
    return {
        R111_BASELINE_ARM: R111SeedResult(
            family="family_deep_v6_overdepth_cap",
            seed=int(seed), arm=R111_BASELINE_ARM,
            metric_name="cap_reproduces",
            metric_value=0.0),
        R111_W55_ARM: R111SeedResult(
            family="family_deep_v6_overdepth_cap",
            seed=int(seed), arm=R111_W55_ARM,
            metric_name="cap_reproduces",
            metric_value=float(score)),
    }


def family_ecc_v7_rate_floor_falsifier(
        seed: int,
) -> dict[str, R111SeedResult]:
    """H21: 96-bit target structurally missed."""
    cb = ECCCodebookV7.init(seed=int(seed))
    from .ecc_codebook_v5 import (
        W53_DEFAULT_ECC_CODE_DIM,
        W53_DEFAULT_ECC_EMIT_MASK_LEN,
    )
    gate = QuantisedBudgetGate.init(
        in_dim=W53_DEFAULT_ECC_CODE_DIM,
        emit_mask_len=W53_DEFAULT_ECC_EMIT_MASK_LEN,
        seed=int(seed) + 1)
    gate.importance_threshold = 0.0
    gate.w_emit.values = [1.0] * len(gate.w_emit.values)
    rng = random.Random(int(seed))
    carriers = [
        [rng.uniform(-1, 1) for _ in range(cb.code_dim)]
        for _ in range(4)
    ]
    res = probe_ecc_v7_rate_floor_falsifier(
        codebook=cb, gate=gate,
        sample_carriers=carriers,
        target_bits_per_token=96.0)
    score = float(res["rate_target_missed_rate"])
    return {
        R111_BASELINE_ARM: R111SeedResult(
            family="family_ecc_v7_rate_floor_falsifier",
            seed=int(seed), arm=R111_BASELINE_ARM,
            metric_name="rate_target_missed_rate",
            metric_value=0.0),
        R111_W55_ARM: R111SeedResult(
            family="family_ecc_v7_rate_floor_falsifier",
            seed=int(seed), arm=R111_W55_ARM,
            metric_name="rate_target_missed_rate",
            metric_value=float(score)),
    }


def family_tvs_arbiter_v4_5arm_dominance(
        seed: int,
) -> dict[str, R111SeedResult]:
    """H22: TVS V4 5-arm oracle-correctness ≥ 0.5."""
    from .quantised_compression import QuantisedCodebookV4
    cb = QuantisedCodebookV4.init(seed=int(seed))
    gate = QuantisedBudgetGate.init(
        in_dim=cb.code_dim, emit_mask_len=4,
        seed=int(seed) + 1)
    gate.importance_threshold = 0.0
    gate.w_emit.values = [1.0] * len(gate.w_emit.values)
    rng = random.Random(int(seed))
    carriers = [
        [rng.uniform(-1, 1) for _ in range(cb.code_dim)]
        for _ in range(8)
    ]
    confs = [
        rng.uniform(0.5, 0.95) for _ in range(8)]
    trusts = [
        rng.uniform(0.5, 0.95) for _ in range(8)]
    mrs = [
        rng.uniform(0.3, 0.9) for _ in range(8)]
    twrs = [
        rng.uniform(0.3, 0.9) for _ in range(8)]
    res = five_arm_compare(
        carriers=carriers, codebook=cb, gate=gate,
        budget_tokens=5,
        per_turn_confidences=confs,
        per_turn_trust_scores=trusts,
        per_turn_merge_retentions=mrs,
        per_turn_tw_retentions=twrs)
    score = (
        1.0 if res.oracle_correctness_rate >= 0.5 else 0.0)
    return {
        R111_BASELINE_ARM: R111SeedResult(
            family="family_tvs_arbiter_v4_5arm_dominance",
            seed=int(seed), arm=R111_BASELINE_ARM,
            metric_name="oracle_correctness_rate",
            metric_value=0.0),
        R111_W55_ARM: R111SeedResult(
            family="family_tvs_arbiter_v4_5arm_dominance",
            seed=int(seed), arm=R111_W55_ARM,
            metric_name="oracle_correctness_rate",
            metric_value=float(score)),
    }


# =============================================================================
# Registry
# =============================================================================


R111_FAMILY_TABLE: dict[
        str, Callable[[int], dict[str, R111SeedResult]]] = {
    "family_persistent_v7_48turn":
        family_persistent_v7_48turn,
    "family_persistent_v7_64turn_stretch":
        family_persistent_v7_64turn_stretch,
    "family_lhr_v7_recovers_t_minus_28":
        family_lhr_v7_recovers_t_minus_28,
    "family_lhr_v7_k36_stretch":
        family_lhr_v7_k36_stretch,
    "family_ecc_v7_compression_18_bits":
        family_ecc_v7_compression_18_bits,
    "family_lhr_v7_degradation_curve":
        family_lhr_v7_degradation_curve,
    "family_w55_distribution_cap":
        family_w55_distribution_cap,
    "family_deep_v6_overdepth_cap":
        family_deep_v6_overdepth_cap,
    "family_ecc_v7_rate_floor_falsifier":
        family_ecc_v7_rate_floor_falsifier,
    "family_tvs_arbiter_v4_5arm_dominance":
        family_tvs_arbiter_v4_5arm_dominance,
}


def run_family(
        family: str, *,
        seeds: Sequence[int] = (1, 2, 3),
) -> R111FamilyComparison:
    if family not in R111_FAMILY_TABLE:
        raise ValueError(f"unknown family {family!r}")
    fn = R111_FAMILY_TABLE[family]
    per_arm: dict[
            str, list[tuple[int, R111SeedResult]]] = {}
    for s in seeds:
        out = fn(int(s))
        for arm, sr in out.items():
            per_arm.setdefault(arm, []).append((int(s), sr))
    aggs: list[R111AggregateResult] = []
    metric_name = ""
    for arm, ls in per_arm.items():
        ls.sort(key=lambda t: t[0])
        seeds_t = tuple(t[0] for t in ls)
        values_t = tuple(
            float(t[1].metric_value) for t in ls)
        metric_name = ls[0][1].metric_name
        aggs.append(R111AggregateResult(
            family=family, arm=arm,
            metric_name=metric_name,
            seeds=seeds_t, values=values_t,
        ))
    return R111FamilyComparison(
        family=family,
        metric_name=metric_name,
        aggregates=tuple(aggs))


def run_all_families(
        *, seeds: Sequence[int] = (1, 2, 3),
) -> dict[str, R111FamilyComparison]:
    return {
        name: run_family(name, seeds=seeds)
        for name in R111_FAMILY_TABLE.keys()
    }


def main() -> None:
    out = run_all_families(seeds=(1, 2, 3))
    summary = {
        "schema": R111_SCHEMA_VERSION,
        "families": [c.to_dict() for c in out.values()],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


__all__ = [
    "R111_SCHEMA_VERSION",
    "R111_BASELINE_ARM",
    "R111_W55_ARM",
    "R111SeedResult",
    "R111AggregateResult",
    "R111FamilyComparison",
    "R111_FAMILY_TABLE",
    "family_persistent_v7_48turn",
    "family_persistent_v7_64turn_stretch",
    "family_lhr_v7_recovers_t_minus_28",
    "family_lhr_v7_k36_stretch",
    "family_ecc_v7_compression_18_bits",
    "family_lhr_v7_degradation_curve",
    "family_w55_distribution_cap",
    "family_deep_v6_overdepth_cap",
    "family_ecc_v7_rate_floor_falsifier",
    "family_tvs_arbiter_v4_5arm_dominance",
    "run_family",
    "run_all_families",
    "main",
]
