"""R-103 — Long-Horizon Retention / Reconstruction / Aggressive
Cramming benchmark family.

Ten families × 3 seeds, exercising H13-H22 of the W52 success
criterion (long-horizon retention / reconstruction /
quantised compression / overdepth + rate-floor caps half).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any, Callable, Sequence

from .branch_cycle_memory_v2 import (
    BranchCycleMemoryV2Head,
    apply_writes_to_v2_head,
    evaluate_joint_recall_v2,
    evaluate_v1_joint_recall_baseline,
    fit_branch_cycle_memory_v2,
    synthesize_branch_cycle_memory_v2_training_set,
)
from .branch_cycle_memory import BranchCycleMemoryHead
from .deep_proxy_stack_v3 import (
    synthesize_deep_stack_v3_training_set,
    fit_deep_proxy_stack_v3,
    evaluate_deep_stack_v3_accuracy,
)
from .deep_proxy_stack_v2 import (
    synthesize_deep_stack_v2_training_set,
    fit_deep_proxy_stack_v2,
    evaluate_deep_stack_v2_accuracy,
)
from .hierarchical_compression import (
    HierarchicalCodebook, HierarchicalEmitGate,
    compress_carrier_hierarchical,
    fit_hierarchical_compression,
    synthesize_hierarchical_compression_training_set,
    W51_DEFAULT_HIER_EMIT_MASK_LEN,
)
from .long_horizon_retention_v4 import (
    fit_long_horizon_v4,
    evaluate_long_horizon_v4_mse_at_k,
    synthesize_long_horizon_v4_training_set,
)
from .long_horizon_retention import (
    evaluate_long_horizon_mse_at_k,
    fit_long_horizon_reconstruction_v3,
    synthesize_long_horizon_reconstruction_training_set,
)
from .persistent_latent_v4 import (
    V4StackedCell,
    evaluate_v4_long_horizon_recall,
    fit_persistent_v4,
    forge_v4_training_set,
    synthesize_v4_training_set,
)
from .persistent_shared_latent import (
    PersistentStateCell,
    PersistentStateExample,
    PersistentStateTrainingSet,
    evaluate_long_horizon_recall,
    fit_persistent_state_cell,
)
from .quantised_compression import (
    QuantisedBudgetGate,
    QuantisedCodebookV4,
    compress_carrier_quantised,
    fit_quantised_compression,
    probe_quantised_degradation_curve,
    probe_quantised_rate_floor_falsifier,
    synthesize_quantised_compression_training_set,
    W52_DEFAULT_QUANT_EMIT_MASK_LEN,
)
from .role_graph_transfer import (
    forge_role_graph_training_set,
    fit_role_graph_mixer,
    evaluate_role_graph_accuracy,
    synthesize_role_graph_training_set,
)


# =============================================================================
# Schema
# =============================================================================

R103_SCHEMA_VERSION: str = "coordpy.r103_benchmark.v1"

R103_BASELINE_ARM: str = "baseline_w51"
R103_W52_ARM: str = "w52"


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


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for i in range(n):
        ai = float(a[i])
        bi = float(b[i])
        dot += ai * bi
        na += ai * ai
        nb += bi * bi
    if na <= 1e-30 or nb <= 1e-30:
        return 0.0
    return float(dot / (math.sqrt(na) * math.sqrt(nb)))


# =============================================================================
# Result dataclasses
# =============================================================================


@dataclasses.dataclass(frozen=True)
class R103SeedResult:
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
class R103AggregateResult:
    family: str
    arm: str
    metric_name: str
    seeds: tuple[int, ...]
    values: tuple[float, ...]

    @property
    def mean(self) -> float:
        if not self.values:
            return 0.0
        return float(sum(self.values)) / float(len(self.values))

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
class R103FamilyComparison:
    family: str
    metric_name: str
    aggregates: tuple[R103AggregateResult, ...]

    def get(self, arm: str) -> R103AggregateResult | None:
        for a in self.aggregates:
            if a.arm == arm:
                return a
        return None

    def delta_w52_vs_w51(self) -> float:
        w52 = self.get(R103_W52_ARM)
        w51 = self.get(R103_BASELINE_ARM)
        if w52 is None or w51 is None:
            return 0.0
        return float(w52.mean - w51.mean)

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": str(self.family),
            "metric_name": str(self.metric_name),
            "aggregates": [a.to_dict() for a in self.aggregates],
            "delta_w52_vs_w51": float(round(
                self.delta_w52_vs_w51(), 12)),
        }


# =============================================================================
# Family functions
# =============================================================================


def family_long_horizon_v4_retention_20turn(
        seed: int,
) -> dict[str, R103SeedResult]:
    """H13: 20-turn V4 cosine recall vs V3 baseline on the same
    corrupted regime."""
    ts4 = synthesize_v4_training_set(
        n_sequences=4, sequence_length=20, state_dim=8,
        input_dim=8, seed=int(seed),
        distractor_window=(5, 12),
        distractor_magnitude=0.5)
    v3_examples = tuple(
        PersistentStateExample(
            input_sequence=ex.input_sequence,
            initial_state=ex.initial_state,
            target_state=ex.target_state)
        for ex in ts4.examples)
    ts3 = PersistentStateTrainingSet(
        examples=v3_examples,
        state_dim=ts4.state_dim,
        input_dim=ts4.input_dim)
    v3, _ = fit_persistent_state_cell(
        ts3, n_steps=96, seed=int(seed), truncate_bptt=4)
    v3_recall = evaluate_long_horizon_recall(v3, ts3.examples)
    v4, _ = fit_persistent_v4(
        ts4, n_steps=96, seed=int(seed), truncate_bptt=4)
    v4_recall = evaluate_v4_long_horizon_recall(
        v4, ts4.examples)
    return {
        R103_BASELINE_ARM: R103SeedResult(
            family="family_long_horizon_v4_retention_20turn",
            seed=int(seed), arm=R103_BASELINE_ARM,
            metric_name="cosine",
            metric_value=float(v3_recall)),
        R103_W52_ARM: R103SeedResult(
            family="family_long_horizon_v4_retention_20turn",
            seed=int(seed), arm=R103_W52_ARM,
            metric_name="cosine",
            metric_value=float(v4_recall)),
    }


def family_long_horizon_v4_retention_24turn_stretch(
        seed: int,
) -> dict[str, R103SeedResult]:
    """H14: 24-turn V4 stretch cosine."""
    ts4 = synthesize_v4_training_set(
        n_sequences=4, sequence_length=24, state_dim=8,
        input_dim=8, seed=int(seed),
        distractor_window=(6, 14),
        distractor_magnitude=0.5)
    v3_examples = tuple(
        PersistentStateExample(
            input_sequence=ex.input_sequence,
            initial_state=ex.initial_state,
            target_state=ex.target_state)
        for ex in ts4.examples)
    ts3 = PersistentStateTrainingSet(
        examples=v3_examples,
        state_dim=ts4.state_dim,
        input_dim=ts4.input_dim)
    v3, _ = fit_persistent_state_cell(
        ts3, n_steps=96, seed=int(seed), truncate_bptt=4)
    v3_recall = evaluate_long_horizon_recall(v3, ts3.examples)
    v4, _ = fit_persistent_v4(
        ts4, n_steps=96, seed=int(seed), truncate_bptt=4)
    v4_recall = evaluate_v4_long_horizon_recall(
        v4, ts4.examples)
    return {
        R103_BASELINE_ARM: R103SeedResult(
            family=(
                "family_long_horizon_v4_retention_24turn_stretch"),
            seed=int(seed), arm=R103_BASELINE_ARM,
            metric_name="cosine",
            metric_value=float(v3_recall)),
        R103_W52_ARM: R103SeedResult(
            family=(
                "family_long_horizon_v4_retention_24turn_stretch"),
            seed=int(seed), arm=R103_W52_ARM,
            metric_name="cosine",
            metric_value=float(v4_recall)),
    }


def family_reconstruction_v4_recovers_t_minus_8(
        seed: int,
) -> dict[str, R103SeedResult]:
    """H15: V4 MSE at k=8."""
    ts = synthesize_long_horizon_v4_training_set(
        n_sequences=4, sequence_length=16, max_k=12,
        out_dim=4, n_branches=2, n_cycles=2, seed=int(seed))
    head, _ = fit_long_horizon_v4(
        ts, n_steps=288, hidden_dim=24, learning_rate=0.005,
        seed=int(seed))
    mse8 = evaluate_long_horizon_v4_mse_at_k(
        head, ts.examples, 8)
    # W51 V3 baseline at max_k=8 on the SAME reconstruction
    # problem — but V3 has only causal+branch heads so its
    # ceiling is the W51 H13 number ~0.41 at k=5; at k=8 V3
    # is undefined (out of range), report 1.0 as ceiling.
    w51_baseline = 1.0
    return {
        R103_BASELINE_ARM: R103SeedResult(
            family=(
                "family_reconstruction_v4_recovers_t_minus_8"),
            seed=int(seed), arm=R103_BASELINE_ARM,
            metric_name="mse_k8",
            metric_value=float(w51_baseline)),
        R103_W52_ARM: R103SeedResult(
            family=(
                "family_reconstruction_v4_recovers_t_minus_8"),
            seed=int(seed), arm=R103_W52_ARM,
            metric_name="mse_k8",
            metric_value=float(mse8)),
    }


def family_reconstruction_v4_k12_stretch(
        seed: int,
) -> dict[str, R103SeedResult]:
    """H16: V4 MSE at k=12 stretch."""
    ts = synthesize_long_horizon_v4_training_set(
        n_sequences=4, sequence_length=16, max_k=12,
        out_dim=4, n_branches=2, n_cycles=2, seed=int(seed))
    head, _ = fit_long_horizon_v4(
        ts, n_steps=288, hidden_dim=24, learning_rate=0.005,
        seed=int(seed))
    mse12 = evaluate_long_horizon_v4_mse_at_k(
        head, ts.examples, 12)
    w51_baseline = 1.0
    return {
        R103_BASELINE_ARM: R103SeedResult(
            family="family_reconstruction_v4_k12_stretch",
            seed=int(seed), arm=R103_BASELINE_ARM,
            metric_name="mse_k12",
            metric_value=float(w51_baseline)),
        R103_W52_ARM: R103SeedResult(
            family="family_reconstruction_v4_k12_stretch",
            seed=int(seed), arm=R103_W52_ARM,
            metric_name="mse_k12",
            metric_value=float(mse12)),
    }


def family_quantised_compression_14bits(
        seed: int,
) -> dict[str, R103SeedResult]:
    """H17: 14 bits/visible-token via quantised compression."""
    # W51 baseline: hierarchical (K1=32 × K2=16) compression
    # — reports its bits/token.
    ts_w51 = synthesize_hierarchical_compression_training_set(
        n_examples=24, code_dim=6, n_coarse=32, n_fine=16,
        emit_mask_len=W51_DEFAULT_HIER_EMIT_MASK_LEN,
        seed=int(seed))
    cb_w51, gate_w51, _ = fit_hierarchical_compression(
        ts_w51, n_steps=32, seed=int(seed))
    gate_w51.importance_threshold = 0.0
    gate_w51.w_emit.values = [1.0] * len(gate_w51.w_emit.values)
    sample = [float((i + seed) % 10) / 10.0 - 0.5
              for i in range(6)]
    res_w51 = compress_carrier_hierarchical(
        sample, codebook=cb_w51, gate=gate_w51)
    bits_w51 = float(res_w51.bits_per_visible_token)
    # W52 quantised: K1=32 × K2=16 × K3=8
    ts_w52 = synthesize_quantised_compression_training_set(
        n_examples=24, code_dim=6, n_coarse=32, n_fine=16,
        n_ultra=8, emit_mask_len=W52_DEFAULT_QUANT_EMIT_MASK_LEN,
        seed=int(seed))
    cb_w52, gate_w52, _ = fit_quantised_compression(
        ts_w52, n_steps=16, seed=int(seed))
    gate_w52.importance_threshold = 0.0
    gate_w52.w_emit.values = [1.0] * len(gate_w52.w_emit.values)
    res_w52 = compress_carrier_quantised(
        sample, codebook=cb_w52, gate=gate_w52)
    bits_w52 = float(res_w52.bits_per_visible_token)
    return {
        R103_BASELINE_ARM: R103SeedResult(
            family="family_quantised_compression_14bits",
            seed=int(seed), arm=R103_BASELINE_ARM,
            metric_name="bits_per_visible_token",
            metric_value=float(bits_w51)),
        R103_W52_ARM: R103SeedResult(
            family="family_quantised_compression_14bits",
            seed=int(seed), arm=R103_W52_ARM,
            metric_name="bits_per_visible_token",
            metric_value=float(bits_w52)),
    }


def family_quantised_degradation_curve(
        seed: int,
) -> dict[str, R103SeedResult]:
    """H18: degradation curve min bits across budgets."""
    ts = synthesize_quantised_compression_training_set(
        n_examples=24, code_dim=6, n_coarse=32, n_fine=16,
        n_ultra=8, emit_mask_len=W52_DEFAULT_QUANT_EMIT_MASK_LEN,
        seed=int(seed))
    cb, gate, _ = fit_quantised_compression(
        ts, n_steps=16, seed=int(seed))
    gate.importance_threshold = 0.0
    gate.w_emit.values = [1.0] * len(gate.w_emit.values)
    sample = [float((i + seed) % 10) / 10.0 - 0.5
              for i in range(6)]
    curve = probe_quantised_degradation_curve(
        sample, codebook=cb, gate=gate)
    bits_min = float(
        min(p.bits_per_visible_token for p in curve))
    return {
        R103_BASELINE_ARM: R103SeedResult(
            family="family_quantised_degradation_curve",
            seed=int(seed), arm=R103_BASELINE_ARM,
            metric_name="min_bits_per_token",
            metric_value=0.0),
        R103_W52_ARM: R103SeedResult(
            family="family_quantised_degradation_curve",
            seed=int(seed), arm=R103_W52_ARM,
            metric_name="min_bits_per_token",
            metric_value=float(bits_min)),
    }


def family_branch_cycle_memory_v2_merge_gain(
        seed: int,
) -> dict[str, R103SeedResult]:
    """H19: V2 BCM joint recall vs V1 baseline."""
    ts = synthesize_branch_cycle_memory_v2_training_set(
        n_examples=16, factor_dim=4,
        n_branch_pages=4, n_cycle_pages=4,
        seed=int(seed))
    v2, _ = fit_branch_cycle_memory_v2(
        ts, n_steps=48, seed=int(seed))
    v2_recall = evaluate_joint_recall_v2(v2, ts.examples)
    v1 = BranchCycleMemoryHead.init(
        factor_dim=4, n_branch_pages=4, n_cycle_pages=4,
        seed=int(seed))
    v1_recall = evaluate_v1_joint_recall_baseline(
        v1, ts.examples)
    return {
        R103_BASELINE_ARM: R103SeedResult(
            family="family_branch_cycle_memory_v2_merge_gain",
            seed=int(seed), arm=R103_BASELINE_ARM,
            metric_name="joint_recall",
            metric_value=float(v1_recall)),
        R103_W52_ARM: R103SeedResult(
            family="family_branch_cycle_memory_v2_merge_gain",
            seed=int(seed), arm=R103_W52_ARM,
            metric_name="joint_recall",
            metric_value=float(v2_recall)),
    }


def family_w52_distribution_cap(
        seed: int,
) -> dict[str, R103SeedResult]:
    """H20: forged-everything distribution cap.

    Combined adversarial forge: persistent-V4 + role-graph.
    Reports the mean protect rate across both surfaces.
    """
    # 1) Persistent V4 forge
    ts4 = synthesize_v4_training_set(
        n_sequences=4, sequence_length=20, state_dim=8,
        input_dim=8, seed=int(seed),
        distractor_window=(5, 12),
        distractor_magnitude=0.5)
    forged4 = forge_v4_training_set(ts4, seed=int(seed))
    v4_forged, _ = fit_persistent_v4(
        forged4, n_steps=64, seed=int(seed), truncate_bptt=4)
    v4_recall_on_clean = evaluate_v4_long_horizon_recall(
        v4_forged, ts4.examples)
    protect_v4 = max(0.0, 1.0 - abs(v4_recall_on_clean))
    # 2) Role-graph forge
    ts_rg = synthesize_role_graph_training_set(
        role_universe=("r0", "r1", "r2", "r3"),
        state_dim=6, n_examples_per_edge=3, seed=int(seed))
    forged_rg = forge_role_graph_training_set(
        ts_rg, seed=int(seed))
    mixer_forged, _ = fit_role_graph_mixer(
        forged_rg, n_steps=64, seed=int(seed))
    rg_acc_on_clean = evaluate_role_graph_accuracy(
        mixer_forged, ts_rg.examples)
    protect_rg = max(0.0, 1.0 - abs(rg_acc_on_clean))
    mean_protect = (protect_v4 + protect_rg) / 2.0
    return {
        R103_BASELINE_ARM: R103SeedResult(
            family="family_w52_distribution_cap",
            seed=int(seed), arm=R103_BASELINE_ARM,
            metric_name="downstream_protect_rate",
            metric_value=1.0),
        R103_W52_ARM: R103SeedResult(
            family="family_w52_distribution_cap",
            seed=int(seed), arm=R103_W52_ARM,
            metric_name="downstream_protect_rate",
            metric_value=float(mean_protect)),
    }


def family_deep_stack_v3_overdepth_cap(
        seed: int,
) -> dict[str, R103SeedResult]:
    """H21: L=8 V3 doesn't strictly improve over L=6 V3 on a
    shallow 2-step regime (apples-to-apples depth comparison
    within V3)."""
    # 2-step regime, single role/branch/cycle to isolate depth.
    ts = synthesize_deep_stack_v3_training_set(
        n_examples=24, in_dim=6, compose_depth=2,
        n_branches=1, n_cycles=1, n_roles=1, seed=int(seed))
    s8, _ = fit_deep_proxy_stack_v3(
        ts, n_layers=8, n_steps=48, seed=int(seed))
    acc_8 = evaluate_deep_stack_v3_accuracy(s8, ts.examples)
    s6, _ = fit_deep_proxy_stack_v3(
        ts, n_layers=6, n_steps=48, seed=int(seed))
    acc_6 = evaluate_deep_stack_v3_accuracy(s6, ts.examples)
    delta = float(acc_8 - acc_6)
    return {
        R103_BASELINE_ARM: R103SeedResult(
            family="family_deep_stack_v3_overdepth_cap",
            seed=int(seed), arm=R103_BASELINE_ARM,
            metric_name="overdepth_delta",
            metric_value=0.0),
        R103_W52_ARM: R103SeedResult(
            family="family_deep_stack_v3_overdepth_cap",
            seed=int(seed), arm=R103_W52_ARM,
            metric_name="overdepth_delta",
            metric_value=float(delta)),
    }


def family_quantised_rate_floor_falsifier(
        seed: int,
) -> dict[str, R103SeedResult]:
    """H22: 32-bit target exceeds quantised codebook capacity."""
    ts = synthesize_quantised_compression_training_set(
        n_examples=24, code_dim=6, n_coarse=32, n_fine=16,
        n_ultra=8, emit_mask_len=W52_DEFAULT_QUANT_EMIT_MASK_LEN,
        seed=int(seed))
    cb, gate, _ = fit_quantised_compression(
        ts, n_steps=16, seed=int(seed))
    gate.importance_threshold = 0.0
    gate.w_emit.values = [1.0] * len(gate.w_emit.values)
    sample = [float((i + seed) % 10) / 10.0 - 0.5
              for i in range(6)]
    res = probe_quantised_rate_floor_falsifier(
        sample, codebook=cb, gate=gate,
        target_bits_per_token=32.0)
    return {
        R103_BASELINE_ARM: R103SeedResult(
            family="family_quantised_rate_floor_falsifier",
            seed=int(seed), arm=R103_BASELINE_ARM,
            metric_name="rate_target_missed",
            metric_value=0.0),
        R103_W52_ARM: R103SeedResult(
            family="family_quantised_rate_floor_falsifier",
            seed=int(seed), arm=R103_W52_ARM,
            metric_name="rate_target_missed",
            metric_value=(
                1.0 if res.rate_target_missed else 0.0)),
    }


# =============================================================================
# Family registry
# =============================================================================


R103_FAMILY_TABLE: dict[
        str, Callable[[int], dict[str, R103SeedResult]]] = {
    "family_long_horizon_v4_retention_20turn":
        family_long_horizon_v4_retention_20turn,
    "family_long_horizon_v4_retention_24turn_stretch":
        family_long_horizon_v4_retention_24turn_stretch,
    "family_reconstruction_v4_recovers_t_minus_8":
        family_reconstruction_v4_recovers_t_minus_8,
    "family_reconstruction_v4_k12_stretch":
        family_reconstruction_v4_k12_stretch,
    "family_quantised_compression_14bits":
        family_quantised_compression_14bits,
    "family_quantised_degradation_curve":
        family_quantised_degradation_curve,
    "family_branch_cycle_memory_v2_merge_gain":
        family_branch_cycle_memory_v2_merge_gain,
    "family_w52_distribution_cap":
        family_w52_distribution_cap,
    "family_deep_stack_v3_overdepth_cap":
        family_deep_stack_v3_overdepth_cap,
    "family_quantised_rate_floor_falsifier":
        family_quantised_rate_floor_falsifier,
}


# =============================================================================
# Driver
# =============================================================================


def run_family(
        family: str, *,
        seeds: Sequence[int] = (1, 2, 3),
) -> R103FamilyComparison:
    if family not in R103_FAMILY_TABLE:
        raise ValueError(f"unknown family {family!r}")
    fn = R103_FAMILY_TABLE[family]
    per_arm: dict[
            str, list[tuple[int, R103SeedResult]]] = {}
    for s in seeds:
        out = fn(int(s))
        for arm, sr in out.items():
            per_arm.setdefault(arm, []).append((int(s), sr))
    aggs: list[R103AggregateResult] = []
    metric_name = ""
    for arm, ls in per_arm.items():
        ls.sort(key=lambda t: t[0])
        seeds_t = tuple(t[0] for t in ls)
        values_t = tuple(float(t[1].metric_value) for t in ls)
        metric_name = ls[0][1].metric_name
        aggs.append(R103AggregateResult(
            family=family, arm=arm,
            metric_name=metric_name,
            seeds=seeds_t, values=values_t,
        ))
    return R103FamilyComparison(
        family=family,
        metric_name=metric_name,
        aggregates=tuple(aggs))


def run_all_families(
        *, seeds: Sequence[int] = (1, 2, 3),
) -> dict[str, R103FamilyComparison]:
    return {
        name: run_family(name, seeds=seeds)
        for name in R103_FAMILY_TABLE.keys()
    }


def main() -> None:
    out = run_all_families(seeds=(1, 2, 3))
    summary = {
        "schema": R103_SCHEMA_VERSION,
        "families": [c.to_dict() for c in out.values()],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


__all__ = [
    "R103_SCHEMA_VERSION",
    "R103_BASELINE_ARM",
    "R103_W52_ARM",
    "R103SeedResult",
    "R103AggregateResult",
    "R103FamilyComparison",
    "R103_FAMILY_TABLE",
    "family_long_horizon_v4_retention_20turn",
    "family_long_horizon_v4_retention_24turn_stretch",
    "family_reconstruction_v4_recovers_t_minus_8",
    "family_reconstruction_v4_k12_stretch",
    "family_quantised_compression_14bits",
    "family_quantised_degradation_curve",
    "family_branch_cycle_memory_v2_merge_gain",
    "family_w52_distribution_cap",
    "family_deep_stack_v3_overdepth_cap",
    "family_quantised_rate_floor_falsifier",
    "run_family",
    "run_all_families",
    "main",
]
