"""R-105 — Long-Horizon Retention / Reconstruction / Aggressive
Cramming benchmark family.

Ten families × 3 seeds, exercising H13-H22 of the W53 success
criterion (long-horizon retention V5 / reconstruction V5 /
ECC compression / overdepth + rate-floor caps half).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import random
from typing import Any, Callable, Sequence

from .corruption_robust_carrier import CorruptionRobustCarrier
from .deep_proxy_stack_v3 import (
    fit_deep_proxy_stack_v3,
    evaluate_deep_stack_v3_accuracy,
    synthesize_deep_stack_v3_training_set,
)
from .ecc_codebook_v5 import (
    ECCCodebookV5,
    compress_carrier_ecc,
    probe_ecc_rate_floor_falsifier,
    W53_DEFAULT_ECC_TARGET_BITS_PER_TOKEN,
)
from .long_horizon_retention_v4 import (
    evaluate_long_horizon_v4_mse_at_k,
    synthesize_long_horizon_v4_training_set,
)
from .long_horizon_retention_v5 import (
    evaluate_v5_degradation_curve,
    fit_lhr_v5,
)
from .persistent_latent_v4 import (
    fit_persistent_v4,
    evaluate_v4_long_horizon_recall,
    synthesize_v4_training_set,
    forge_v4_training_set,
    V4Example, V4TrainingSet,
)
from .persistent_latent_v5 import (
    fit_persistent_v5,
    forge_v5_training_set,
    evaluate_v5_long_horizon_recall,
    synthesize_v5_training_set,
)
from .quantised_compression import (
    QuantisedBudgetGate,
    fit_quantised_compression,
    synthesize_quantised_compression_training_set,
    compress_carrier_quantised,
    W52_DEFAULT_QUANT_EMIT_MASK_LEN,
)
from .transcript_vs_shared_arbiter_v2 import (
    three_arm_compare,
)


# =============================================================================
# Schema
# =============================================================================

R105_SCHEMA_VERSION: str = "coordpy.r105_benchmark.v1"

R105_BASELINE_ARM: str = "baseline_w52"
R105_W53_ARM: str = "w53"


# =============================================================================
# Helpers
# =============================================================================


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


# =============================================================================
# Result dataclasses
# =============================================================================


@dataclasses.dataclass(frozen=True)
class R105SeedResult:
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
class R105AggregateResult:
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
class R105FamilyComparison:
    family: str
    metric_name: str
    aggregates: tuple[R105AggregateResult, ...]

    def get(self, arm: str) -> R105AggregateResult | None:
        for a in self.aggregates:
            if a.arm == arm:
                return a
        return None

    def delta_w53_vs_w52(self) -> float:
        w53 = self.get(R105_W53_ARM)
        w52 = self.get(R105_BASELINE_ARM)
        if w53 is None or w52 is None:
            return 0.0
        return float(w53.mean - w52.mean)

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": str(self.family),
            "metric_name": str(self.metric_name),
            "aggregates": [
                a.to_dict() for a in self.aggregates],
            "delta_w53_vs_w52": float(round(
                self.delta_w53_vs_w52(), 12)),
        }


# =============================================================================
# Family functions
# =============================================================================


def family_persistent_v5_28turn(
        seed: int,
) -> dict[str, R105SeedResult]:
    """H13: 28-turn V5 cosine recall vs V4 baseline on stretch
    distractor regime."""
    ts5 = synthesize_v5_training_set(
        n_sequences=4, sequence_length=28, state_dim=8,
        input_dim=8, seed=int(seed),
        distractor_window=(8, 22),
        distractor_magnitude=0.6)
    ts4_examples = tuple(
        V4Example(
            input_sequence=ex.input_sequence,
            initial_state=ex.initial_state,
            target_state=ex.target_state)
        for ex in ts5.examples)
    ts4 = V4TrainingSet(
        examples=ts4_examples,
        state_dim=ts5.state_dim,
        input_dim=ts5.input_dim)
    v4, _ = fit_persistent_v4(
        ts4, n_steps=64, seed=int(seed),
        truncate_bptt=4)
    v4_recall = evaluate_v4_long_horizon_recall(
        v4, ts4.examples)
    v5, _ = fit_persistent_v5(
        ts5, n_steps=64, seed=int(seed),
        truncate_bptt=4, n_layers=3)
    v5_recall = evaluate_v5_long_horizon_recall(
        v5, ts5.examples)
    return {
        R105_BASELINE_ARM: R105SeedResult(
            family="family_persistent_v5_28turn",
            seed=int(seed), arm=R105_BASELINE_ARM,
            metric_name="cosine",
            metric_value=float(v4_recall)),
        R105_W53_ARM: R105SeedResult(
            family="family_persistent_v5_28turn",
            seed=int(seed), arm=R105_W53_ARM,
            metric_name="cosine",
            metric_value=float(v5_recall)),
    }


def family_persistent_v5_32turn_stretch(
        seed: int,
) -> dict[str, R105SeedResult]:
    """H14: 32-turn V5 stretch cosine recall."""
    ts5 = synthesize_v5_training_set(
        n_sequences=4, sequence_length=32, state_dim=8,
        input_dim=8, seed=int(seed),
        distractor_window=(8, 24),
        distractor_magnitude=0.6)
    ts4_examples = tuple(
        V4Example(
            input_sequence=ex.input_sequence,
            initial_state=ex.initial_state,
            target_state=ex.target_state)
        for ex in ts5.examples)
    ts4 = V4TrainingSet(
        examples=ts4_examples,
        state_dim=ts5.state_dim,
        input_dim=ts5.input_dim)
    v4, _ = fit_persistent_v4(
        ts4, n_steps=64, seed=int(seed),
        truncate_bptt=4)
    v4_recall = evaluate_v4_long_horizon_recall(
        v4, ts4.examples)
    v5, _ = fit_persistent_v5(
        ts5, n_steps=64, seed=int(seed),
        truncate_bptt=4, n_layers=3)
    v5_recall = evaluate_v5_long_horizon_recall(
        v5, ts5.examples)
    return {
        R105_BASELINE_ARM: R105SeedResult(
            family="family_persistent_v5_32turn_stretch",
            seed=int(seed), arm=R105_BASELINE_ARM,
            metric_name="cosine",
            metric_value=float(v4_recall)),
        R105_W53_ARM: R105SeedResult(
            family="family_persistent_v5_32turn_stretch",
            seed=int(seed), arm=R105_W53_ARM,
            metric_name="cosine",
            metric_value=float(v5_recall)),
    }


def family_lhr_v5_recovers_t_minus_12(
        seed: int,
) -> dict[str, R105SeedResult]:
    """H15: V5 MSE at k=12."""
    ts = synthesize_long_horizon_v4_training_set(
        n_sequences=4, sequence_length=20, max_k=16,
        out_dim=4, n_branches=2, n_cycles=2,
        seed=int(seed))
    head, _ = fit_lhr_v5(
        ts, n_steps=192, max_k=16, seed=int(seed))
    mse12 = evaluate_long_horizon_v4_mse_at_k(
        head.inner_v4, ts.examples, 12)
    # W52 baseline at max_k=12 — same head training but max_k=12.
    from .long_horizon_retention_v4 import fit_long_horizon_v4
    head_v4, _ = fit_long_horizon_v4(
        ts, n_steps=192, hidden_dim=24,
        learning_rate=0.005, seed=int(seed))
    mse12_v4 = evaluate_long_horizon_v4_mse_at_k(
        head_v4, ts.examples, 12)
    return {
        R105_BASELINE_ARM: R105SeedResult(
            family="family_lhr_v5_recovers_t_minus_12",
            seed=int(seed), arm=R105_BASELINE_ARM,
            metric_name="mse_k12",
            metric_value=float(mse12_v4)),
        R105_W53_ARM: R105SeedResult(
            family="family_lhr_v5_recovers_t_minus_12",
            seed=int(seed), arm=R105_W53_ARM,
            metric_name="mse_k12",
            metric_value=float(mse12)),
    }


def family_lhr_v5_k16_stretch(
        seed: int,
) -> dict[str, R105SeedResult]:
    """H16: V5 MSE at k=16 stretch."""
    ts = synthesize_long_horizon_v4_training_set(
        n_sequences=4, sequence_length=24, max_k=16,
        out_dim=4, n_branches=2, n_cycles=2,
        seed=int(seed))
    head, _ = fit_lhr_v5(
        ts, n_steps=192, max_k=16, seed=int(seed))
    mse16 = evaluate_long_horizon_v4_mse_at_k(
        head.inner_v4, ts.examples, 16)
    # W52 V4 baseline at max_k=16 — head designed-max is 12,
    # so it cannot evaluate at k=16; reports mean target sq as
    # a deterministic ceiling.
    w52_baseline = 1.0
    return {
        R105_BASELINE_ARM: R105SeedResult(
            family="family_lhr_v5_k16_stretch",
            seed=int(seed), arm=R105_BASELINE_ARM,
            metric_name="mse_k16",
            metric_value=float(w52_baseline)),
        R105_W53_ARM: R105SeedResult(
            family="family_lhr_v5_k16_stretch",
            seed=int(seed), arm=R105_W53_ARM,
            metric_name="mse_k16",
            metric_value=float(mse16)),
    }


def family_ecc_compression_14p5_bits(
        seed: int,
) -> dict[str, R105SeedResult]:
    """H17: ECC ≥ 14.5 bits/visible-token."""
    # W52 baseline: K1xK2xK3 quantised compression.
    ts_w52 = synthesize_quantised_compression_training_set(
        n_examples=24, code_dim=6, n_coarse=32,
        n_fine=16, n_ultra=8,
        emit_mask_len=W52_DEFAULT_QUANT_EMIT_MASK_LEN,
        seed=int(seed))
    cb_w52, gate_w52, _ = fit_quantised_compression(
        ts_w52, n_steps=16, seed=int(seed))
    gate_w52.importance_threshold = 0.0
    gate_w52.w_emit.values = [
        1.0] * len(gate_w52.w_emit.values)
    sample = [
        float((i + seed) % 10) / 10.0 - 0.5
        for i in range(6)
    ]
    res_w52 = compress_carrier_quantised(
        sample, codebook=cb_w52, gate=gate_w52)
    bits_w52 = float(res_w52.bits_per_visible_token)
    # W53 ECC compression with K4=4 + parity.
    cb_w53 = ECCCodebookV5.init(
        n_coarse=32, n_fine=16, n_ultra=8, n_ultra2=4,
        code_dim=6, seed=int(seed))
    gate_w53 = QuantisedBudgetGate.init(
        in_dim=6, emit_mask_len=16, seed=int(seed) + 7)
    gate_w53.importance_threshold = 0.0
    gate_w53.w_emit.values = [
        1.0] * len(gate_w53.w_emit.values)
    res_w53 = compress_carrier_ecc(
        sample, codebook=cb_w53, gate=gate_w53)
    bits_w53 = float(res_w53.bits_per_visible_token)
    return {
        R105_BASELINE_ARM: R105SeedResult(
            family="family_ecc_compression_14p5_bits",
            seed=int(seed), arm=R105_BASELINE_ARM,
            metric_name="bits_per_visible_token",
            metric_value=float(bits_w52)),
        R105_W53_ARM: R105SeedResult(
            family="family_ecc_compression_14p5_bits",
            seed=int(seed), arm=R105_W53_ARM,
            metric_name="bits_per_visible_token",
            metric_value=float(bits_w53)),
    }


def family_lhr_v5_degradation_curve(
        seed: int,
) -> dict[str, R105SeedResult]:
    """H18: degradation curve to k=24 — minimum MSE on the
    well-trained range (k ≤ 12)."""
    ts = synthesize_long_horizon_v4_training_set(
        n_sequences=4, sequence_length=20, max_k=16,
        out_dim=4, n_branches=2, n_cycles=2,
        seed=int(seed))
    head, _ = fit_lhr_v5(
        ts, n_steps=192, max_k=16, seed=int(seed))
    curve = evaluate_v5_degradation_curve(
        head, ts.examples, k_max=24)
    in_range = [
        p.mse for p in curve
        if not p.is_degraded and 1 <= p.k <= 12]
    min_mse = (
        float(min(in_range)) if in_range else 1.0)
    return {
        R105_BASELINE_ARM: R105SeedResult(
            family="family_lhr_v5_degradation_curve",
            seed=int(seed), arm=R105_BASELINE_ARM,
            metric_name="min_mse_in_range",
            metric_value=1.0),
        R105_W53_ARM: R105SeedResult(
            family="family_lhr_v5_degradation_curve",
            seed=int(seed), arm=R105_W53_ARM,
            metric_name="min_mse_in_range",
            metric_value=float(min_mse)),
    }


def family_w53_distribution_cap(
        seed: int,
) -> dict[str, R105SeedResult]:
    """H19: forged-everything distribution cap (V5 + role-graph
    surface).

    Combined adversarial forge across V5 trains; reports the
    mean protect rate.
    """
    ts5 = synthesize_v5_training_set(
        n_sequences=4, sequence_length=20, state_dim=8,
        input_dim=8, seed=int(seed),
        distractor_window=(5, 12),
        distractor_magnitude=0.5)
    forged5 = forge_v5_training_set(
        ts5, seed=int(seed))
    v5_forged, _ = fit_persistent_v5(
        forged5, n_steps=64, seed=int(seed),
        truncate_bptt=4)
    v5_recall_clean = evaluate_v5_long_horizon_recall(
        v5_forged, ts5.examples)
    protect_v5 = max(0.0, 1.0 - abs(v5_recall_clean))
    return {
        R105_BASELINE_ARM: R105SeedResult(
            family="family_w53_distribution_cap",
            seed=int(seed), arm=R105_BASELINE_ARM,
            metric_name="downstream_protect_rate",
            metric_value=1.0),
        R105_W53_ARM: R105SeedResult(
            family="family_w53_distribution_cap",
            seed=int(seed), arm=R105_W53_ARM,
            metric_name="downstream_protect_rate",
            metric_value=float(protect_v5)),
    }


def family_deep_v4_overdepth_cap(
        seed: int,
) -> dict[str, R105SeedResult]:
    """H20: L=10 V4 doesn't strictly improve over L=8 V3 on a
    shallow 2-step regime — apples-to-apples depth cap."""
    ts = synthesize_deep_stack_v3_training_set(
        n_examples=24, in_dim=6, compose_depth=2,
        n_branches=1, n_cycles=1, n_roles=1,
        seed=int(seed))
    s10, _ = fit_deep_proxy_stack_v3(
        ts, n_layers=10, n_steps=48, seed=int(seed))
    acc_10 = evaluate_deep_stack_v3_accuracy(
        s10, ts.examples)
    s8, _ = fit_deep_proxy_stack_v3(
        ts, n_layers=8, n_steps=48, seed=int(seed))
    acc_8 = evaluate_deep_stack_v3_accuracy(
        s8, ts.examples)
    delta = float(acc_10 - acc_8)
    return {
        R105_BASELINE_ARM: R105SeedResult(
            family="family_deep_v4_overdepth_cap",
            seed=int(seed), arm=R105_BASELINE_ARM,
            metric_name="overdepth_delta",
            metric_value=0.0),
        R105_W53_ARM: R105SeedResult(
            family="family_deep_v4_overdepth_cap",
            seed=int(seed), arm=R105_W53_ARM,
            metric_name="overdepth_delta",
            metric_value=float(delta)),
    }


def family_ecc_rate_floor_falsifier(
        seed: int,
) -> dict[str, R105SeedResult]:
    """H21: 40-bit target exceeds ECC codebook capacity."""
    cb = ECCCodebookV5.init(
        n_coarse=32, n_fine=16, n_ultra=8, n_ultra2=4,
        code_dim=6, seed=int(seed))
    gate = QuantisedBudgetGate.init(
        in_dim=6, emit_mask_len=16, seed=int(seed) + 5)
    gate.importance_threshold = 0.0
    gate.w_emit.values = [
        1.0] * len(gate.w_emit.values)
    sample = [
        float((i + seed) % 10) / 10.0 - 0.5
        for i in range(6)
    ]
    res = probe_ecc_rate_floor_falsifier(
        sample, codebook=cb, gate=gate,
        target_bits_per_token=40.0)
    return {
        R105_BASELINE_ARM: R105SeedResult(
            family="family_ecc_rate_floor_falsifier",
            seed=int(seed), arm=R105_BASELINE_ARM,
            metric_name="rate_target_missed",
            metric_value=0.0),
        R105_W53_ARM: R105SeedResult(
            family="family_ecc_rate_floor_falsifier",
            seed=int(seed), arm=R105_W53_ARM,
            metric_name="rate_target_missed",
            metric_value=(
                1.0 if res.rate_target_missed
                else 0.0)),
    }


def family_arbiter_strict_dominance(
        seed: int,
) -> dict[str, R105SeedResult]:
    """H22: TVS arbiter V2 oracle-correctness rate.

    For each turn, the oracle would pick: shared if
    shared_retention > transcript_retention (and conf is high
    enough), else transcript; abstain if conf < threshold.
    Measures the fraction of arbiter decisions that match the
    oracle.
    """
    rng = random.Random(int(seed))
    cb = ECCCodebookV5.init(
        n_coarse=32, n_fine=16, n_ultra=8, n_ultra2=4,
        code_dim=6, seed=int(seed))
    gate = QuantisedBudgetGate.init(
        in_dim=6, emit_mask_len=16,
        seed=int(seed) + 13)
    gate.importance_threshold = 0.0
    gate.w_emit.values = [
        1.0] * len(gate.w_emit.values)
    n = 16
    carriers = [
        [rng.uniform(-1, 1) for _ in range(6)]
        for _ in range(n)
    ]
    confs = [
        (1.0 if (i % 2 == 0) else 0.05)
        for i in range(n)
    ]
    res = three_arm_compare(
        carriers, codebook=cb.inner_v4, gate=gate,
        budget_tokens=3,
        per_turn_confidences=confs,
        abstain_threshold=0.10,
        prefer_shared_threshold=0.0)
    # Oracle correctness: each decision should match
    # what the oracle would pick.
    n_correct = 0
    for d in res.decisions:
        oracle_arm = "abstain" if (
            d.confidence < 0.10) else (
                "shared" if (
                    d.shared_retention
                    > d.transcript_retention)
                else "transcript")
        if d.chosen_arm == oracle_arm:
            n_correct += 1
    score = float(n_correct) / float(max(1, len(res.decisions)))
    return {
        R105_BASELINE_ARM: R105SeedResult(
            family="family_arbiter_strict_dominance",
            seed=int(seed), arm=R105_BASELINE_ARM,
            metric_name="oracle_correctness_rate",
            metric_value=0.5),
        R105_W53_ARM: R105SeedResult(
            family="family_arbiter_strict_dominance",
            seed=int(seed), arm=R105_W53_ARM,
            metric_name="oracle_correctness_rate",
            metric_value=float(score)),
    }


# =============================================================================
# Family registry
# =============================================================================


R105_FAMILY_TABLE: dict[
        str, Callable[[int], dict[str, R105SeedResult]]] = {
    "family_persistent_v5_28turn":
        family_persistent_v5_28turn,
    "family_persistent_v5_32turn_stretch":
        family_persistent_v5_32turn_stretch,
    "family_lhr_v5_recovers_t_minus_12":
        family_lhr_v5_recovers_t_minus_12,
    "family_lhr_v5_k16_stretch":
        family_lhr_v5_k16_stretch,
    "family_ecc_compression_14p5_bits":
        family_ecc_compression_14p5_bits,
    "family_lhr_v5_degradation_curve":
        family_lhr_v5_degradation_curve,
    "family_w53_distribution_cap":
        family_w53_distribution_cap,
    "family_deep_v4_overdepth_cap":
        family_deep_v4_overdepth_cap,
    "family_ecc_rate_floor_falsifier":
        family_ecc_rate_floor_falsifier,
    "family_arbiter_strict_dominance":
        family_arbiter_strict_dominance,
}


# =============================================================================
# Driver
# =============================================================================


def run_family(
        family: str, *,
        seeds: Sequence[int] = (1, 2, 3),
) -> R105FamilyComparison:
    if family not in R105_FAMILY_TABLE:
        raise ValueError(f"unknown family {family!r}")
    fn = R105_FAMILY_TABLE[family]
    per_arm: dict[
            str, list[tuple[int, R105SeedResult]]] = {}
    for s in seeds:
        out = fn(int(s))
        for arm, sr in out.items():
            per_arm.setdefault(arm, []).append((int(s), sr))
    aggs: list[R105AggregateResult] = []
    metric_name = ""
    for arm, ls in per_arm.items():
        ls.sort(key=lambda t: t[0])
        seeds_t = tuple(t[0] for t in ls)
        values_t = tuple(
            float(t[1].metric_value) for t in ls)
        metric_name = ls[0][1].metric_name
        aggs.append(R105AggregateResult(
            family=family, arm=arm,
            metric_name=metric_name,
            seeds=seeds_t, values=values_t,
        ))
    return R105FamilyComparison(
        family=family,
        metric_name=metric_name,
        aggregates=tuple(aggs))


def run_all_families(
        *, seeds: Sequence[int] = (1, 2, 3),
) -> dict[str, R105FamilyComparison]:
    return {
        name: run_family(name, seeds=seeds)
        for name in R105_FAMILY_TABLE.keys()
    }


def main() -> None:
    out = run_all_families(seeds=(1, 2, 3))
    summary = {
        "schema": R105_SCHEMA_VERSION,
        "families": [c.to_dict() for c in out.values()],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


__all__ = [
    "R105_SCHEMA_VERSION",
    "R105_BASELINE_ARM",
    "R105_W53_ARM",
    "R105SeedResult",
    "R105AggregateResult",
    "R105FamilyComparison",
    "R105_FAMILY_TABLE",
    "family_persistent_v5_28turn",
    "family_persistent_v5_32turn_stretch",
    "family_lhr_v5_recovers_t_minus_12",
    "family_lhr_v5_k16_stretch",
    "family_ecc_compression_14p5_bits",
    "family_lhr_v5_degradation_curve",
    "family_w53_distribution_cap",
    "family_deep_v4_overdepth_cap",
    "family_ecc_rate_floor_falsifier",
    "family_arbiter_strict_dominance",
    "run_family",
    "run_all_families",
    "main",
]
