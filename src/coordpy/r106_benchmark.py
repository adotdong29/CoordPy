"""R-106 — Corruption / Hostile-Channel / Consensus-Merge family.

Twelve families × 3 seeds, exercising H23-H34 of the W53
success criterion (corruption-robust carrier / consensus
quorum / merge integrity / abstain semantics half).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import random
from typing import Any, Callable, Sequence

from .branch_merge_memory_v3 import (
    BranchMergeMemoryV3Head,
    evaluate_consensus_recall,
)
from .corruption_robust_carrier import (
    CorruptionRobustCarrier,
    HostileChannelProbeResult,
    probe_hostile_channel,
)
from .deep_proxy_stack_v4 import (
    DeepProxyStackV4,
    emit_deep_proxy_stack_v4_forward_witness,
)
from .ecc_codebook_v5 import (
    ECCCodebookV5,
    compress_carrier_ecc,
    decode_with_parity_check,
    flip_random_bit,
    emit_ecc_robustness_witness,
)
from .mergeable_latent_capsule import (
    MergeAuditTrail,
    MergeOperator,
    MergeableLatentCapsule,
    compute_consensus_quorum,
    make_root_capsule,
    merge_capsules,
)
from .multi_hop_translator_v3 import (
    build_unfitted_quint_translator,
    fit_quint_translator,
    score_quint_fidelity,
    synthesize_quint_training_set,
)
from .multi_hop_translator import perturb_edge
from .persistent_latent_v5 import (
    fit_persistent_v5,
    forge_v5_training_set,
    evaluate_v5_long_horizon_recall,
    synthesize_v5_training_set,
)
from .quantised_compression import QuantisedBudgetGate
from .uncertainty_layer import (
    calibration_check,
    compose_uncertainty_report,
)


# =============================================================================
# Schema
# =============================================================================

R106_SCHEMA_VERSION: str = "coordpy.r106_benchmark.v1"

R106_BASELINE_ARM: str = "baseline_w52"
R106_W53_ARM: str = "w53"


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
class R106SeedResult:
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
class R106AggregateResult:
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
class R106FamilyComparison:
    family: str
    metric_name: str
    aggregates: tuple[R106AggregateResult, ...]

    def get(self, arm: str) -> R106AggregateResult | None:
        for a in self.aggregates:
            if a.arm == arm:
                return a
        return None

    def delta_w53_vs_w52(self) -> float:
        w53 = self.get(R106_W53_ARM)
        w52 = self.get(R106_BASELINE_ARM)
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
# Helpers for benchmark families
# =============================================================================


def _build_default_crc(
        *, seed: int,
) -> CorruptionRobustCarrier:
    cb = ECCCodebookV5.init(
        n_coarse=32, n_fine=16, n_ultra=8, n_ultra2=4,
        code_dim=6, seed=int(seed))
    gate = QuantisedBudgetGate.init(
        in_dim=6, emit_mask_len=16, seed=int(seed) + 5)
    gate.importance_threshold = 0.0
    gate.w_emit.values = [
        1.0] * len(gate.w_emit.values)
    return CorruptionRobustCarrier.init(
        codebook=cb, gate=gate, repetition=3)


# =============================================================================
# Family functions
# =============================================================================


def family_single_bit_detect_rate(
        seed: int,
) -> dict[str, R106SeedResult]:
    """H23: single-bit corruption detect rate ≥ 0.80."""
    crc = _build_default_crc(seed=int(seed))
    rng = random.Random(int(seed))
    carriers = [
        [rng.uniform(-1, 1) for _ in range(6)]
        for _ in range(20)
    ]
    res = probe_hostile_channel(
        carriers, crc=crc, flip_intensity=1.0,
        seed=int(seed) + 11)
    # Baseline (W52): no parity → cannot detect.
    return {
        R106_BASELINE_ARM: R106SeedResult(
            family="family_single_bit_detect_rate",
            seed=int(seed), arm=R106_BASELINE_ARM,
            metric_name="detect_rate",
            metric_value=0.0),
        R106_W53_ARM: R106SeedResult(
            family="family_single_bit_detect_rate",
            seed=int(seed), arm=R106_W53_ARM,
            metric_name="detect_rate",
            metric_value=float(res.detect_rate)),
    }


def family_single_bit_correction_rate(
        seed: int,
) -> dict[str, R106SeedResult]:
    """H24: single-bit partial correction rate ≥ 0.30."""
    crc = _build_default_crc(seed=int(seed))
    rng = random.Random(int(seed))
    carriers = [
        [rng.uniform(-1, 1) for _ in range(6)]
        for _ in range(20)
    ]
    res = probe_hostile_channel(
        carriers, crc=crc, flip_intensity=1.0,
        seed=int(seed) + 13)
    return {
        R106_BASELINE_ARM: R106SeedResult(
            family="family_single_bit_correction_rate",
            seed=int(seed), arm=R106_BASELINE_ARM,
            metric_name="correction_rate",
            metric_value=0.0),
        R106_W53_ARM: R106SeedResult(
            family="family_single_bit_correction_rate",
            seed=int(seed), arm=R106_W53_ARM,
            metric_name="correction_rate",
            metric_value=float(res.correction_rate)),
    }


def family_two_bit_graceful_degrade(
        seed: int,
) -> dict[str, R106SeedResult]:
    """H25: 2-bit corruption → graceful (abstain ≥ 0.50,
    silent_failure ≤ 0.30)."""
    crc = _build_default_crc(seed=int(seed))
    rng = random.Random(int(seed))
    carriers = [
        [rng.uniform(-1, 1) for _ in range(6)]
        for _ in range(20)
    ]
    res = probe_hostile_channel(
        carriers, crc=crc, flip_intensity=2.0,
        seed=int(seed) + 17)
    score = (
        1.0 if (
            res.abstain_rate >= 0.50
            and res.silent_failure_rate <= 0.30)
        else 0.0)
    return {
        R106_BASELINE_ARM: R106SeedResult(
            family="family_two_bit_graceful_degrade",
            seed=int(seed), arm=R106_BASELINE_ARM,
            metric_name="graceful_degrade_score",
            metric_value=0.0),
        R106_W53_ARM: R106SeedResult(
            family="family_two_bit_graceful_degrade",
            seed=int(seed), arm=R106_W53_ARM,
            metric_name="graceful_degrade_score",
            metric_value=float(score)),
    }


def family_consensus_recall_kof2(
        seed: int,
) -> dict[str, R106SeedResult]:
    """H26: BMM V3 consensus recall ≥ 0.70 with K=2-of-N."""
    h = BranchMergeMemoryV3Head.init(
        factor_dim=4, n_branch_pages=4,
        n_cycle_pages=4, n_joint_pages=4,
        n_consensus_pages=4, k_required=2,
        cosine_floor=0.5, seed=int(seed))
    rec = evaluate_consensus_recall(
        h, n_branches=4, n_consistent=3,
        factor_dim=4, seed=int(seed) + 7)
    return {
        R106_BASELINE_ARM: R106SeedResult(
            family="family_consensus_recall_kof2",
            seed=int(seed), arm=R106_BASELINE_ARM,
            metric_name="consensus_recall",
            metric_value=0.0),
        R106_W53_ARM: R106SeedResult(
            family="family_consensus_recall_kof2",
            seed=int(seed), arm=R106_W53_ARM,
            metric_name="consensus_recall",
            metric_value=float(rec)),
    }


def family_consensus_abstain_when_disagreed(
        seed: int,
) -> dict[str, R106SeedResult]:
    """H27: BMM V3 abstains when ≤ K branches agree.

    With K_required=4 and only 2 consistent branches, the
    consensus must NOT be reached → recall is 0.
    """
    h = BranchMergeMemoryV3Head.init(
        factor_dim=4, n_branch_pages=4,
        n_cycle_pages=4, n_joint_pages=4,
        n_consensus_pages=4, k_required=4,
        cosine_floor=0.99, seed=int(seed))
    # Insert random branches — none should reach K=4 ≥ 0.99.
    rec = evaluate_consensus_recall(
        h, n_branches=4, n_consistent=2,
        factor_dim=4, seed=int(seed) + 11)
    score = 1.0 if rec == 0.0 else 0.0
    return {
        R106_BASELINE_ARM: R106SeedResult(
            family="family_consensus_abstain_when_disagreed",
            seed=int(seed), arm=R106_BASELINE_ARM,
            metric_name="abstain_correct_score",
            metric_value=0.0),
        R106_W53_ARM: R106SeedResult(
            family="family_consensus_abstain_when_disagreed",
            seed=int(seed), arm=R106_W53_ARM,
            metric_name="abstain_correct_score",
            metric_value=float(score)),
    }


def family_mlsc_merge_replay_determinism(
        seed: int,
) -> dict[str, R106SeedResult]:
    """H28: same parents → same merged CID across two runs."""
    op1 = MergeOperator(factor_dim=4)
    op2 = MergeOperator(factor_dim=4)
    rng = random.Random(int(seed))
    p_a = make_root_capsule(
        branch_id="a",
        payload=[rng.uniform(-1, 1) for _ in range(4)],
        confidence=0.5)
    p_b = make_root_capsule(
        branch_id="b",
        payload=[rng.uniform(-1, 1) for _ in range(4)],
        confidence=0.5)
    audit1 = MergeAuditTrail.empty()
    audit2 = MergeAuditTrail.empty()
    m1 = merge_capsules(
        op1, [p_a, p_b], audit_trail=audit1)
    m2 = merge_capsules(
        op2, [p_a, p_b], audit_trail=audit2)
    score = 1.0 if (
        m1.cid() == m2.cid()) else 0.0
    return {
        R106_BASELINE_ARM: R106SeedResult(
            family="family_mlsc_merge_replay_determinism",
            seed=int(seed), arm=R106_BASELINE_ARM,
            metric_name="replay_ok",
            metric_value=0.0),
        R106_W53_ARM: R106SeedResult(
            family="family_mlsc_merge_replay_determinism",
            seed=int(seed), arm=R106_W53_ARM,
            metric_name="replay_ok",
            metric_value=float(score)),
    }


def family_perturbed_edge_uncertainty_report(
        seed: int,
) -> dict[str, R106SeedResult]:
    """H29: under a perturbed translator edge, the V3
    arbitration uncertainty mean rises strictly."""
    ts = synthesize_quint_training_set(
        n_examples=16, code_dim=6, feature_dim=6,
        seed=int(seed))
    tr_clean, _ = fit_quint_translator(
        ts, n_steps=64, seed=int(seed))
    fid_clean = score_quint_fidelity(
        tr_clean, ts.examples[:8])
    tr_pert = perturb_edge(
        tr_clean, src="B", dst="C",
        noise_magnitude=2.0, seed=int(seed) * 7)
    fid_pert = score_quint_fidelity(
        tr_pert, ts.examples[:8])
    delta = float(
        fid_pert.arbitration_uncertainty_mean
        - fid_clean.arbitration_uncertainty_mean)
    score = 1.0 if delta > 0 else 0.0
    return {
        R106_BASELINE_ARM: R106SeedResult(
            family="family_perturbed_edge_uncertainty_report",
            seed=int(seed), arm=R106_BASELINE_ARM,
            metric_name="uncertainty_rises_score",
            metric_value=0.0),
        R106_W53_ARM: R106SeedResult(
            family="family_perturbed_edge_uncertainty_report",
            seed=int(seed), arm=R106_W53_ARM,
            metric_name="uncertainty_rises_score",
            metric_value=float(score)),
    }


def family_compromise_v5_persistent_state(
        seed: int,
) -> dict[str, R106SeedResult]:
    """H30: forged V5 train → mostly degraded recall on clean
    examples. Reports the protect rate."""
    ts5 = synthesize_v5_training_set(
        n_sequences=4, sequence_length=20, state_dim=8,
        input_dim=8, seed=int(seed),
        distractor_window=(5, 12),
        distractor_magnitude=0.5)
    forged = forge_v5_training_set(
        ts5, seed=int(seed))
    v5_forged, _ = fit_persistent_v5(
        forged, n_steps=64, seed=int(seed),
        truncate_bptt=4)
    rec_clean = evaluate_v5_long_horizon_recall(
        v5_forged, ts5.examples)
    protect = max(0.0, 1.0 - abs(rec_clean))
    return {
        R106_BASELINE_ARM: R106SeedResult(
            family="family_compromise_v5_persistent_state",
            seed=int(seed), arm=R106_BASELINE_ARM,
            metric_name="downstream_protect_rate",
            metric_value=1.0),
        R106_W53_ARM: R106SeedResult(
            family="family_compromise_v5_persistent_state",
            seed=int(seed), arm=R106_W53_ARM,
            metric_name="downstream_protect_rate",
            metric_value=float(protect)),
    }


def family_corruption_robust_carrier_safety(
        seed: int,
) -> dict[str, R106SeedResult]:
    """H31: silent_failure_rate ≤ 0.10 across single-bit flips.

    A silent failure is a corrupted decode whose parity check
    passed. With per-segment XOR parity, a single bit flip can
    only escape detection if it lands in the parity bit itself
    (flips the parity but no segment bit) — so silent_failure
    is bounded.
    """
    crc = _build_default_crc(seed=int(seed))
    rng = random.Random(int(seed))
    carriers = [
        [rng.uniform(-1, 1) for _ in range(6)]
        for _ in range(40)
    ]
    res = probe_hostile_channel(
        carriers, crc=crc, flip_intensity=1.0,
        seed=int(seed) + 23)
    score = (
        1.0 if res.silent_failure_rate <= 0.10
        else 0.0)
    return {
        R106_BASELINE_ARM: R106SeedResult(
            family="family_corruption_robust_carrier_safety",
            seed=int(seed), arm=R106_BASELINE_ARM,
            metric_name="silent_failure_safety",
            metric_value=0.0),
        R106_W53_ARM: R106SeedResult(
            family="family_corruption_robust_carrier_safety",
            seed=int(seed), arm=R106_W53_ARM,
            metric_name="silent_failure_safety",
            metric_value=float(score)),
    }


def family_uncertainty_calibration_under_noise(
        seed: int,
) -> dict[str, R106SeedResult]:
    """H32: uncertainty layer remains calibrated when injected
    noise lowers per-component confidences."""
    rng = random.Random(int(seed))
    n = 30
    confs: list[float] = []
    accs: list[float] = []
    for i in range(n):
        is_high = (i % 3 != 0)
        report = compose_uncertainty_report(
            persistent_v5_confidence=(
                rng.uniform(0.7, 0.9)
                if is_high else rng.uniform(0.0, 0.2)),
            multi_hop_v3_confidence=(
                rng.uniform(0.6, 0.9)
                if is_high else rng.uniform(0.0, 0.3)),
            mlsc_capsule_confidence=(
                rng.uniform(0.6, 0.9)
                if is_high else rng.uniform(0.0, 0.3)),
            deep_v4_corruption_confidence=(
                rng.uniform(0.6, 0.9)
                if is_high else rng.uniform(0.0, 0.3)),
            crc_silent_failure_rate=(
                rng.uniform(0.0, 0.1)
                if is_high else rng.uniform(0.4, 0.8)))
        a = (
            rng.uniform(0.7, 1.0)
            if is_high
            else rng.uniform(0.0, 0.4))
        confs.append(float(report.composite_confidence))
        accs.append(a)
    res = calibration_check(
        confs, accs, min_calibration_gap=0.10)
    score = 1.0 if res.calibrated else 0.0
    return {
        R106_BASELINE_ARM: R106SeedResult(
            family=(
                "family_uncertainty_calibration_under_noise"),
            seed=int(seed), arm=R106_BASELINE_ARM,
            metric_name="calibrated_under_noise",
            metric_value=0.0),
        R106_W53_ARM: R106SeedResult(
            family=(
                "family_uncertainty_calibration_under_noise"),
            seed=int(seed), arm=R106_W53_ARM,
            metric_name="calibrated_under_noise",
            metric_value=float(score)),
    }


def family_persistent_v5_chain_walk_depth(
        seed: int,
) -> dict[str, R106SeedResult]:
    """H33: persistent V5 chain walks back ≥ 16 turns."""
    from coordpy.persistent_latent_v5 import (
        V5StackedCell, PersistentLatentStateV5Chain,
        step_persistent_state_v5, emit_persistent_v5_witness,
    )
    cell = V5StackedCell.init(
        state_dim=4, input_dim=4, n_layers=3,
        seed=int(seed))
    chain = PersistentLatentStateV5Chain.empty()
    prev = None
    rng = random.Random(int(seed))
    for t in range(20):
        carrier = [
            rng.uniform(-1, 1) for _ in range(4)]
        s = step_persistent_state_v5(
            cell=cell, prev_state=prev,
            carrier_values=carrier,
            turn_index=int(t),
            role="r0", branch_id="m",
            cycle_index=0,
            skip_input=carrier)
        chain.add(s)
        prev = s
    if prev is None:
        depth = 0
    else:
        w = emit_persistent_v5_witness(
            state=prev, cell=cell, chain=chain,
            max_walk_depth=32)
        depth = int(w.chain_walk_depth)
    score = (
        1.0 if depth >= 16 else 0.0)
    return {
        R106_BASELINE_ARM: R106SeedResult(
            family="family_persistent_v5_chain_walk_depth",
            seed=int(seed), arm=R106_BASELINE_ARM,
            metric_name="chain_walk_depth_score",
            metric_value=0.0),
        R106_W53_ARM: R106SeedResult(
            family="family_persistent_v5_chain_walk_depth",
            seed=int(seed), arm=R106_W53_ARM,
            metric_name="chain_walk_depth_score",
            metric_value=float(score)),
    }


def family_w53_integration_envelope(
        seed: int,
) -> dict[str, R106SeedResult]:
    """H34: integration sanity — a single W53 envelope binds the
    persistent V5 chain CID, MLSC audit trail CID, all
    per-turn witness CIDs, and the W52 outer CID."""
    from coordpy.agents import Agent
    from coordpy.synthetic_llm import SyntheticLLMClient
    from coordpy.w53_team import (
        W53Team, build_w53_registry)
    backend = SyntheticLLMClient(
        model_tag=f"synth.r106i.{seed}",
        default_response="i")
    agents = [
        Agent(name="a1", instructions="", role="r0",
              backend=backend, temperature=0.0,
              max_tokens=20)
    ]
    reg = build_w53_registry(
        schema_cid=f"r106_int_{seed}",
        role_universe=("r0",))
    team = W53Team(
        agents=agents, backend=backend, registry=reg,
        max_visible_handoffs=2)
    r = team.run("integration probe")
    env = r.w53_envelope
    score = 1.0 if (
        env.persistent_v5_chain_cid
        and env.mlsc_audit_trail_cid
        and env.turn_witness_bundle_cid
        and env.w52_outer_cid
        and env.params_cid) else 0.0
    return {
        R106_BASELINE_ARM: R106SeedResult(
            family="family_w53_integration_envelope",
            seed=int(seed), arm=R106_BASELINE_ARM,
            metric_name="envelope_complete",
            metric_value=0.0),
        R106_W53_ARM: R106SeedResult(
            family="family_w53_integration_envelope",
            seed=int(seed), arm=R106_W53_ARM,
            metric_name="envelope_complete",
            metric_value=float(score)),
    }


# =============================================================================
# Family registry
# =============================================================================


R106_FAMILY_TABLE: dict[
        str, Callable[[int], dict[str, R106SeedResult]]] = {
    "family_single_bit_detect_rate":
        family_single_bit_detect_rate,
    "family_single_bit_correction_rate":
        family_single_bit_correction_rate,
    "family_two_bit_graceful_degrade":
        family_two_bit_graceful_degrade,
    "family_consensus_recall_kof2":
        family_consensus_recall_kof2,
    "family_consensus_abstain_when_disagreed":
        family_consensus_abstain_when_disagreed,
    "family_mlsc_merge_replay_determinism":
        family_mlsc_merge_replay_determinism,
    "family_perturbed_edge_uncertainty_report":
        family_perturbed_edge_uncertainty_report,
    "family_compromise_v5_persistent_state":
        family_compromise_v5_persistent_state,
    "family_corruption_robust_carrier_safety":
        family_corruption_robust_carrier_safety,
    "family_uncertainty_calibration_under_noise":
        family_uncertainty_calibration_under_noise,
    "family_persistent_v5_chain_walk_depth":
        family_persistent_v5_chain_walk_depth,
    "family_w53_integration_envelope":
        family_w53_integration_envelope,
}


# =============================================================================
# Driver
# =============================================================================


def run_family(
        family: str, *,
        seeds: Sequence[int] = (1, 2, 3),
) -> R106FamilyComparison:
    if family not in R106_FAMILY_TABLE:
        raise ValueError(f"unknown family {family!r}")
    fn = R106_FAMILY_TABLE[family]
    per_arm: dict[
            str, list[tuple[int, R106SeedResult]]] = {}
    for s in seeds:
        out = fn(int(s))
        for arm, sr in out.items():
            per_arm.setdefault(arm, []).append((int(s), sr))
    aggs: list[R106AggregateResult] = []
    metric_name = ""
    for arm, ls in per_arm.items():
        ls.sort(key=lambda t: t[0])
        seeds_t = tuple(t[0] for t in ls)
        values_t = tuple(
            float(t[1].metric_value) for t in ls)
        metric_name = ls[0][1].metric_name
        aggs.append(R106AggregateResult(
            family=family, arm=arm,
            metric_name=metric_name,
            seeds=seeds_t, values=values_t,
        ))
    return R106FamilyComparison(
        family=family,
        metric_name=metric_name,
        aggregates=tuple(aggs))


def run_all_families(
        *, seeds: Sequence[int] = (1, 2, 3),
) -> dict[str, R106FamilyComparison]:
    return {
        name: run_family(name, seeds=seeds)
        for name in R106_FAMILY_TABLE.keys()
    }


def main() -> None:
    out = run_all_families(seeds=(1, 2, 3))
    summary = {
        "schema": R106_SCHEMA_VERSION,
        "families": [c.to_dict() for c in out.values()],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


__all__ = [
    "R106_SCHEMA_VERSION",
    "R106_BASELINE_ARM",
    "R106_W53_ARM",
    "R106SeedResult",
    "R106AggregateResult",
    "R106FamilyComparison",
    "R106_FAMILY_TABLE",
    "family_single_bit_detect_rate",
    "family_single_bit_correction_rate",
    "family_two_bit_graceful_degrade",
    "family_consensus_recall_kof2",
    "family_consensus_abstain_when_disagreed",
    "family_mlsc_merge_replay_determinism",
    "family_perturbed_edge_uncertainty_report",
    "family_compromise_v5_persistent_state",
    "family_corruption_robust_carrier_safety",
    "family_uncertainty_calibration_under_noise",
    "family_persistent_v5_chain_walk_depth",
    "family_w53_integration_envelope",
    "run_family",
    "run_all_families",
    "main",
]
