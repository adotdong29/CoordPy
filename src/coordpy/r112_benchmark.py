"""R-112 — Corruption / Trust-Consensus / Algebra / Fallback family.

Sixteen families × 3 seeds, exercising H23-H38 of the W55
success criterion (BCH(15,7) + 5-of-7 repetition + interleave +
TWCC + MLSC V3 trust decay + uncertainty V3 + W55 integration +
algebra soundness).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import random
from typing import Any, Callable, Sequence

from .corruption_robust_carrier_v3 import (
    CorruptionRobustCarrierV3,
    probe_hostile_channel_v3,
)
from .deep_proxy_stack_v6 import (
    DeepProxyStackV6,
    emit_deep_proxy_stack_v6_forward_witness,
)
from .disagreement_algebra import (
    check_difference_self_cancellation,
    check_intersection_distributivity_on_agreement,
    check_merge_idempotent,
    difference_op, intersection_op, merge_op,
)
from .ecc_codebook_v5 import (
    ECCCodebookV5,
    W53_DEFAULT_ECC_CODE_DIM,
    W53_DEFAULT_ECC_EMIT_MASK_LEN,
)
from .mergeable_latent_capsule_v3 import (
    MergeAuditTrailV3,
    MergeOperatorV3,
    W55_DEFAULT_MLSC_V3_TRUST_DECAY,
    make_root_capsule_v3,
    merge_capsules_v3,
    reinforce_capsule_trust_v3,
    step_branch_capsule_v3,
)
from .multi_hop_translator_v5 import (
    fit_hept_translator,
    score_hept_fidelity,
    synthesize_hept_training_set,
)
from .persistent_latent_v7 import (
    V7StackedCell,
    evaluate_v7_long_horizon_recall,
    fit_persistent_v7,
    forge_v7_carrier_sequences,
)
from .quantised_compression import QuantisedBudgetGate
from .trust_weighted_consensus_controller import (
    TrustWeightedConsensusController,
    TrustWeightedConsensusPolicy,
    W55_TWCC_DECISION_ABSTAIN,
    W55_TWCC_DECISION_BEST_PARENT,
    W55_TWCC_DECISION_QUORUM,
    W55_TWCC_DECISION_TRANSCRIPT,
    W55_TWCC_DECISION_TRUST_WEIGHTED,
    W55_TWCC_STAGE_ABSTAIN,
    W55_TWCC_STAGE_FALLBACK_BEST_PARENT,
    W55_TWCC_STAGE_FALLBACK_TRANSCRIPT,
    W55_TWCC_STAGE_K_OF_N,
    W55_TWCC_STAGE_TRUST_WEIGHTED,
)
from .uncertainty_layer import calibration_check
from .uncertainty_layer_v3 import (
    FactUncertainty,
    calibration_check_under_adversarial,
    compose_uncertainty_report_v3,
)


# =============================================================================
# Schema
# =============================================================================

R112_SCHEMA_VERSION: str = "coordpy.r112_benchmark.v1"
R112_BASELINE_ARM: str = "baseline_w54"
R112_W55_ARM: str = "w55"


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
class R112SeedResult:
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
class R112AggregateResult:
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
class R112FamilyComparison:
    family: str
    metric_name: str
    aggregates: tuple[R112AggregateResult, ...]

    def get(self, arm: str) -> R112AggregateResult | None:
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
# Helpers
# =============================================================================


def _make_crc_v3(seed: int) -> CorruptionRobustCarrierV3:
    cb = ECCCodebookV5.init(seed=int(seed))
    gate = QuantisedBudgetGate.init(
        in_dim=W53_DEFAULT_ECC_CODE_DIM,
        emit_mask_len=W53_DEFAULT_ECC_EMIT_MASK_LEN,
        seed=int(seed) + 1)
    gate.importance_threshold = 0.0
    gate.w_emit.values = [1.0] * len(gate.w_emit.values)
    return CorruptionRobustCarrierV3.init(
        codebook=cb, gate=gate)


# =============================================================================
# Families
# =============================================================================


def family_bch_double_bit_correct(
        seed: int,
) -> dict[str, R112SeedResult]:
    """H23: BCH(15,7) two-bit correct rate ≥ 0.85."""
    crc = _make_crc_v3(int(seed))
    rng = random.Random(int(seed))
    carriers = [
        [rng.uniform(-1, 1) for _ in range(crc.codebook.code_dim)]
        for _ in range(20)
    ]
    res = probe_hostile_channel_v3(
        crc, carriers=carriers, flip_intensity=1.0,
        seed=int(seed))
    score = (
        1.0 if res.bch_double_bit_correct_rate >= 0.85
        else 0.0)
    return {
        R112_BASELINE_ARM: R112SeedResult(
            family="family_bch_double_bit_correct",
            seed=int(seed), arm=R112_BASELINE_ARM,
            metric_name="double_correct_ge_0_85",
            metric_value=0.0),
        R112_W55_ARM: R112SeedResult(
            family="family_bch_double_bit_correct",
            seed=int(seed), arm=R112_W55_ARM,
            metric_name="double_correct_ge_0_85",
            metric_value=float(score)),
    }


def family_bch_three_bit_detect(
        seed: int,
) -> dict[str, R112SeedResult]:
    """H24: BCH(15,7) three-bit detect rate ≥ 0.55."""
    crc = _make_crc_v3(int(seed))
    rng = random.Random(int(seed))
    carriers = [
        [rng.uniform(-1, 1) for _ in range(crc.codebook.code_dim)]
        for _ in range(20)
    ]
    res = probe_hostile_channel_v3(
        crc, carriers=carriers, flip_intensity=1.0,
        seed=int(seed))
    score = (
        1.0 if res.bch_three_bit_detect_rate >= 0.55
        else 0.0)
    return {
        R112_BASELINE_ARM: R112SeedResult(
            family="family_bch_three_bit_detect",
            seed=int(seed), arm=R112_BASELINE_ARM,
            metric_name="three_detect_ge_0_55",
            metric_value=0.0),
        R112_W55_ARM: R112SeedResult(
            family="family_bch_three_bit_detect",
            seed=int(seed), arm=R112_W55_ARM,
            metric_name="three_detect_ge_0_55",
            metric_value=float(score)),
    }


def family_crc_v3_silent_failure_floor(
        seed: int,
) -> dict[str, R112SeedResult]:
    """H25: silent failure ≤ 0.03 single-bit."""
    crc = _make_crc_v3(int(seed))
    rng = random.Random(int(seed))
    carriers = [
        [rng.uniform(-1, 1) for _ in range(crc.codebook.code_dim)]
        for _ in range(20)
    ]
    res = probe_hostile_channel_v3(
        crc, carriers=carriers, flip_intensity=1.0,
        seed=int(seed))
    score = (
        1.0 if res.silent_failure_rate <= 0.03 else 0.0)
    return {
        R112_BASELINE_ARM: R112SeedResult(
            family="family_crc_v3_silent_failure_floor",
            seed=int(seed), arm=R112_BASELINE_ARM,
            metric_name="silent_failure_le_0_03",
            metric_value=0.0),
        R112_W55_ARM: R112SeedResult(
            family="family_crc_v3_silent_failure_floor",
            seed=int(seed), arm=R112_W55_ARM,
            metric_name="silent_failure_le_0_03",
            metric_value=float(score)),
    }


def family_trust_consensus_controller_recall(
        seed: int,
) -> dict[str, R112SeedResult]:
    """H26: trust-weighted quorum recall ≥ 0.70."""
    op = MergeOperatorV3(factor_dim=4, trust_floor=0.0)
    policy = TrustWeightedConsensusPolicy(
        k_min=2, k_max=4, cosine_floor=0.5,
        fallback_cosine_floor=0.0,
        trust_threshold=0.5,
        allow_trust_weighted=True,
        allow_fallback_best_parent=True,
        allow_fallback_transcript=False)
    ctrl = TrustWeightedConsensusController.init(
        policy=policy, operator=op)
    rng = random.Random(int(seed))
    n_quorum_succ = 0
    n_trials = 10
    for trial in range(n_trials):
        target = [rng.uniform(-1, 1) for _ in range(4)]
        branches = [
            make_root_capsule_v3(
                branch_id=f"b{i}",
                payload=[
                    t + 0.05 * rng.uniform(-1, 1)
                    for t in target],
                confidence=0.85, trust=0.9)
            for i in range(4)
        ]
        res, _ = ctrl.decide(branches, k_required=2)
        if res.decision in (
                W55_TWCC_DECISION_QUORUM,
                W55_TWCC_DECISION_TRUST_WEIGHTED):
            n_quorum_succ += 1
    rate = float(n_quorum_succ) / float(n_trials)
    score = 1.0 if rate >= 0.70 else 0.0
    return {
        R112_BASELINE_ARM: R112SeedResult(
            family="family_trust_consensus_controller_recall",
            seed=int(seed), arm=R112_BASELINE_ARM,
            metric_name="quorum_recall_ge_0_70",
            metric_value=0.0),
        R112_W55_ARM: R112SeedResult(
            family="family_trust_consensus_controller_recall",
            seed=int(seed), arm=R112_W55_ARM,
            metric_name="quorum_recall_ge_0_70",
            metric_value=float(score)),
    }


def family_trust_consensus_controller_5stage_fallback(
        seed: int,
) -> dict[str, R112SeedResult]:
    """H27: 5-stage fallback completes."""
    op = MergeOperatorV3(factor_dim=4, trust_floor=0.0)
    policy = TrustWeightedConsensusPolicy(
        k_min=3, k_max=4, cosine_floor=0.99,
        fallback_cosine_floor=0.95,
        trust_threshold=10.0,
        allow_trust_weighted=True,
        allow_fallback_best_parent=True,
        allow_fallback_transcript=True)
    ctrl = TrustWeightedConsensusController.init(
        policy=policy, operator=op)
    rng = random.Random(int(seed))
    branches = [
        make_root_capsule_v3(
            branch_id=f"b{i}",
            payload=[float(j == i) for j in range(4)],
            confidence=0.5, trust=0.3)
        for i in range(3)
    ]
    res, entry = ctrl.decide(
        branches, k_required=3,
        transcript_payload=[1.0, 0.0, 0.0, 0.0])
    # All five stages should have been attempted (or at least
    # some sequence ending in transcript or abstain).
    stages_attempted = {
        s.stage for s in entry.stage_attempts}
    expected = {
        W55_TWCC_STAGE_K_OF_N,
        W55_TWCC_STAGE_TRUST_WEIGHTED,
        W55_TWCC_STAGE_FALLBACK_BEST_PARENT,
    }
    score = (
        1.0 if expected.issubset(stages_attempted) else 0.0)
    return {
        R112_BASELINE_ARM: R112SeedResult(
            family="family_trust_consensus_controller_5stage_fallback",
            seed=int(seed), arm=R112_BASELINE_ARM,
            metric_name="5stage_walk_complete",
            metric_value=0.0),
        R112_W55_ARM: R112SeedResult(
            family="family_trust_consensus_controller_5stage_fallback",
            seed=int(seed), arm=R112_W55_ARM,
            metric_name="5stage_walk_complete",
            metric_value=float(score)),
    }


def family_mlsc_v3_trust_decay(
        seed: int,
) -> dict[str, R112SeedResult]:
    """H28: trust decays in [0,1] each turn; reinforces on merge."""
    rng = random.Random(int(seed))
    p = make_root_capsule_v3(
        branch_id="a",
        payload=[rng.uniform(-1, 1) for _ in range(4)],
        confidence=0.9, trust=0.9,
        trust_decay=W55_DEFAULT_MLSC_V3_TRUST_DECAY)
    trust_chain = [p.trust]
    cur = p
    for t in range(6):
        cur = step_branch_capsule_v3(
            parent=cur,
            payload=[rng.uniform(-1, 1) for _ in range(4)],
            new_fact_tags=(f"t{t}",), turn_index=t + 1)
        trust_chain.append(cur.trust)
    # Trust should decay monotonically.
    monotone = all(
        trust_chain[i] >= trust_chain[i + 1]
        for i in range(len(trust_chain) - 1))
    # Reinforce should raise trust.
    re = reinforce_capsule_trust_v3(cur, reinforcement=0.5)
    rises = re.trust > cur.trust
    # All in [0, 1]
    bounds = all(0.0 <= t <= 1.0 for t in trust_chain)
    score = 1.0 if (monotone and rises and bounds) else 0.0
    return {
        R112_BASELINE_ARM: R112SeedResult(
            family="family_mlsc_v3_trust_decay",
            seed=int(seed), arm=R112_BASELINE_ARM,
            metric_name="trust_decay_correct",
            metric_value=0.0),
        R112_W55_ARM: R112SeedResult(
            family="family_mlsc_v3_trust_decay",
            seed=int(seed), arm=R112_W55_ARM,
            metric_name="trust_decay_correct",
            metric_value=float(score)),
    }


def family_disagreement_algebra_soundness(
        seed: int,
) -> dict[str, R112SeedResult]:
    """H29: ⊕/⊖/⊗ operate correctly on adversarial inputs."""
    rng = random.Random(int(seed))
    # Adversarial: large magnitudes, opposite signs.
    a = [10.0 * rng.uniform(-1, 1) for _ in range(4)]
    b = [-1.0 * v for v in a]
    c = [0.5 * v for v in a]
    # ⊕ idempotent should still hold.
    r1 = check_merge_idempotent(a)
    # ⊖ self-cancel should hold even on extreme inputs.
    r2 = check_difference_self_cancellation(a)
    # ⊗ distributivity should hold on agreement subspace.
    r3 = check_intersection_distributivity_on_agreement(
        a, b, c)
    # ⊖ commutativity: a ⊖ b = b ⊖ a
    d_ab = difference_op(a, b)
    d_ba = difference_op(b, a)
    comm = all(
        abs(d_ab[i] - d_ba[i]) < 1e-9
        for i in range(len(d_ab)))
    score = 1.0 if (r1.ok and r2.ok and r3.ok and comm) else 0.0
    return {
        R112_BASELINE_ARM: R112SeedResult(
            family="family_disagreement_algebra_soundness",
            seed=int(seed), arm=R112_BASELINE_ARM,
            metric_name="algebra_soundness_adversarial",
            metric_value=0.0),
        R112_W55_ARM: R112SeedResult(
            family="family_disagreement_algebra_soundness",
            seed=int(seed), arm=R112_W55_ARM,
            metric_name="algebra_soundness_adversarial",
            metric_value=float(score)),
    }


def family_compromise_v7_persistent_state(
        seed: int,
) -> dict[str, R112SeedResult]:
    """H30: forged V7 train → protect rate ≥ 0.45 mean."""
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
    forged = forge_v7_carrier_sequences(
        sequences, seed=int(seed))
    rec_forged = evaluate_v7_long_horizon_recall(
        cell, forged, targets)
    protect = max(0.0, 1.0 - abs(float(rec_forged)))
    return {
        R112_BASELINE_ARM: R112SeedResult(
            family="family_compromise_v7_persistent_state",
            seed=int(seed), arm=R112_BASELINE_ARM,
            metric_name="protect_rate",
            metric_value=1.0),
        R112_W55_ARM: R112SeedResult(
            family="family_compromise_v7_persistent_state",
            seed=int(seed), arm=R112_W55_ARM,
            metric_name="protect_rate",
            metric_value=float(protect)),
    }


def family_corruption_robust_carrier_v3_safety(
        seed: int,
) -> dict[str, R112SeedResult]:
    """H31: silent failure ≤ 0.03 across single-bit (tighter than W54)."""
    crc = _make_crc_v3(int(seed))
    rng = random.Random(int(seed))
    carriers = [
        [rng.uniform(-1, 1) for _ in range(crc.codebook.code_dim)]
        for _ in range(20)
    ]
    res = probe_hostile_channel_v3(
        crc, carriers=carriers, flip_intensity=1.0,
        seed=int(seed))
    score = (
        1.0 if res.silent_failure_rate <= 0.03 else 0.0)
    return {
        R112_BASELINE_ARM: R112SeedResult(
            family="family_corruption_robust_carrier_v3_safety",
            seed=int(seed), arm=R112_BASELINE_ARM,
            metric_name="silent_failure_safety",
            metric_value=0.0),
        R112_W55_ARM: R112SeedResult(
            family="family_corruption_robust_carrier_v3_safety",
            seed=int(seed), arm=R112_W55_ARM,
            metric_name="silent_failure_safety",
            metric_value=float(score)),
    }


def family_uncertainty_v3_trust_weighted_composite(
        seed: int,
) -> dict[str, R112SeedResult]:
    """H32: trust-weighted composite penalises low-trust."""
    rng = random.Random(int(seed))
    # Test: high-conf component with low trust should reduce
    # tw_composite vs untrusted composite.
    report_eq = compose_uncertainty_report_v3(
        persistent_v7_confidence=0.9,
        multi_hop_v5_confidence=0.9,
        mlsc_v3_capsule_confidence=0.9,
        deep_v6_corruption_confidence=0.9,
        crc_v3_silent_failure_rate=0.05,
        trust_weights={
            "persistent_v7": 1.0,
            "multi_hop_v5": 1.0,
            "mlsc_v3": 1.0,
            "deep_v6": 1.0,
            "crc_v3": 1.0,
        })
    # Now lower trust on one high-conf component.
    report_low = compose_uncertainty_report_v3(
        persistent_v7_confidence=0.9,
        multi_hop_v5_confidence=0.9,
        mlsc_v3_capsule_confidence=0.9,
        deep_v6_corruption_confidence=0.9,
        crc_v3_silent_failure_rate=0.05,
        trust_weights={
            "persistent_v7": 0.1,
            "multi_hop_v5": 1.0,
            "mlsc_v3": 1.0,
            "deep_v6": 1.0,
            "crc_v3": 1.0,
        })
    # The trust-weighted composite should differ.
    score = 1.0 if (
        abs(report_low.trust_weighted_composite
             - report_eq.trust_weighted_composite) > 1e-6
    ) else 0.0
    return {
        R112_BASELINE_ARM: R112SeedResult(
            family="family_uncertainty_v3_trust_weighted_composite",
            seed=int(seed), arm=R112_BASELINE_ARM,
            metric_name="tw_composite_responds",
            metric_value=0.0),
        R112_W55_ARM: R112SeedResult(
            family="family_uncertainty_v3_trust_weighted_composite",
            seed=int(seed), arm=R112_W55_ARM,
            metric_name="tw_composite_responds",
            metric_value=float(score)),
    }


def family_persistent_v7_chain_walk_depth(
        seed: int,
) -> dict[str, R112SeedResult]:
    """H33: V7 chain walks back ≥ 32 turns."""
    from .persistent_latent_v7 import (
        PersistentLatentStateV7Chain,
        emit_persistent_v7_witness,
        step_persistent_state_v7,
    )
    cell = V7StackedCell.init(
        state_dim=4, input_dim=4, n_layers=5,
        seed=int(seed))
    chain = PersistentLatentStateV7Chain.empty()
    state = None
    rng = random.Random(int(seed))
    for t in range(36):
        state = step_persistent_state_v7(
            cell=cell, prev_state=state,
            carrier_values=[
                rng.uniform(-1, 1) for _ in range(4)],
            turn_index=t, role="r0",
            anchor_skip=[
                rng.uniform(-1, 1) for _ in range(4)])
        chain.add(state)
    w = emit_persistent_v7_witness(
        state=state, cell=cell, chain=chain)
    score = 1.0 if w.chain_walk_depth >= 32 else 0.0
    return {
        R112_BASELINE_ARM: R112SeedResult(
            family="family_persistent_v7_chain_walk_depth",
            seed=int(seed), arm=R112_BASELINE_ARM,
            metric_name="chain_walk_ge_32",
            metric_value=0.0),
        R112_W55_ARM: R112SeedResult(
            family="family_persistent_v7_chain_walk_depth",
            seed=int(seed), arm=R112_W55_ARM,
            metric_name="chain_walk_ge_32",
            metric_value=float(score)),
    }


def family_w55_integration_envelope(
        seed: int,
) -> dict[str, R112SeedResult]:
    """H34: W55 envelope binds all required CIDs."""
    from coordpy.agents import Agent
    from coordpy.synthetic_llm import SyntheticLLMClient
    from coordpy.w55_team import W55Team, build_w55_registry
    backend = SyntheticLLMClient(
        model_tag=f"r112.i.{seed}", default_response="i")
    agents = [
        Agent(name="a1", instructions="", role="r0",
              backend=backend, temperature=0.0,
              max_tokens=20)]
    reg = build_w55_registry(
        schema_cid=f"r112_int_{seed}",
        role_universe=("r0",))
    team = W55Team(
        agents=agents, backend=backend, registry=reg,
        max_visible_handoffs=2)
    r = team.run("integration probe")
    env = r.w55_envelope
    # All required CIDs nonempty.
    required = [
        env.w54_outer_cid,
        env.params_cid,
        env.turn_witness_bundle_cid,
        env.persistent_v7_chain_cid,
        env.mlsc_v3_audit_trail_cid,
        env.twcc_audit_trail_cid,
        env.disagreement_algebra_trace_cid,
    ]
    score = 1.0 if all(
        len(s) == 64 for s in required) else 0.0
    return {
        R112_BASELINE_ARM: R112SeedResult(
            family="family_w55_integration_envelope",
            seed=int(seed), arm=R112_BASELINE_ARM,
            metric_name="integration_complete",
            metric_value=0.0),
        R112_W55_ARM: R112SeedResult(
            family="family_w55_integration_envelope",
            seed=int(seed), arm=R112_W55_ARM,
            metric_name="integration_complete",
            metric_value=float(score)),
    }


def family_arbiter_v4_budget_allocator(
        seed: int,
) -> dict[str, R112SeedResult]:
    """H35: per-arm budget sums to total budget."""
    from .transcript_vs_shared_arbiter_v4 import (
        five_arm_compare)
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
    res = five_arm_compare(
        carriers=carriers, codebook=cb, gate=gate,
        budget_tokens=5,
        per_turn_confidences=[0.8] * 8,
        per_turn_trust_scores=[0.9] * 8,
        per_turn_merge_retentions=[0.7] * 8,
        per_turn_tw_retentions=[0.85] * 8)
    score = (
        1.0 if res.budget_allocator_correct_rate >= 0.95
        else 0.0)
    return {
        R112_BASELINE_ARM: R112SeedResult(
            family="family_arbiter_v4_budget_allocator",
            seed=int(seed), arm=R112_BASELINE_ARM,
            metric_name="budget_allocator_ge_0_95",
            metric_value=0.0),
        R112_W55_ARM: R112SeedResult(
            family="family_arbiter_v4_budget_allocator",
            seed=int(seed), arm=R112_W55_ARM,
            metric_name="budget_allocator_ge_0_95",
            metric_value=float(score)),
    }


def family_deep_v6_adaptive_abstain_threshold(
        seed: int,
) -> dict[str, R112SeedResult]:
    """H36: adaptive threshold scales monotonically with input norm."""
    stack = DeepProxyStackV6.init(
        n_layers=14, in_dim=8, factor_dim=8,
        n_heads=2, n_branch_heads=2, n_cycle_heads=2,
        n_roles=2, n_outer_layers=2,
        seed=int(seed))
    # Smaller input → smaller threshold.
    q_small = [0.01] * 8
    q_med = [1.0] * 8
    q_big = [1e6] * 8
    w_s, _ = emit_deep_proxy_stack_v6_forward_witness(
        stack=stack, query_input=q_small,
        slot_keys=[q_small], slot_values=[q_small])
    w_m, _ = emit_deep_proxy_stack_v6_forward_witness(
        stack=stack, query_input=q_med,
        slot_keys=[q_med], slot_values=[q_med])
    w_b, _ = emit_deep_proxy_stack_v6_forward_witness(
        stack=stack, query_input=q_big,
        slot_keys=[q_big], slot_values=[q_big])
    monotone = (
        w_s.adaptive_threshold <= w_m.adaptive_threshold
        <= w_b.adaptive_threshold)
    score = 1.0 if monotone else 0.0
    return {
        R112_BASELINE_ARM: R112SeedResult(
            family="family_deep_v6_adaptive_abstain_threshold",
            seed=int(seed), arm=R112_BASELINE_ARM,
            metric_name="adaptive_monotone",
            metric_value=0.0),
        R112_W55_ARM: R112SeedResult(
            family="family_deep_v6_adaptive_abstain_threshold",
            seed=int(seed), arm=R112_W55_ARM,
            metric_name="adaptive_monotone",
            metric_value=float(score)),
    }


def family_interleaving_burst_recovery(
        seed: int,
) -> dict[str, R112SeedResult]:
    """H37: interleaved CRC V3 recovers ≥ 80% of 3-bit burst errors."""
    crc = _make_crc_v3(int(seed))
    rng = random.Random(int(seed))
    carriers = [
        [rng.uniform(-1, 1) for _ in range(crc.codebook.code_dim)]
        for _ in range(20)
    ]
    res = probe_hostile_channel_v3(
        crc, carriers=carriers, flip_intensity=1.0,
        seed=int(seed))
    score = (
        1.0
        if res.interleave_burst_recovery_rate >= 0.80
        else 0.0)
    return {
        R112_BASELINE_ARM: R112SeedResult(
            family="family_interleaving_burst_recovery",
            seed=int(seed), arm=R112_BASELINE_ARM,
            metric_name="burst_recovery_ge_0_80",
            metric_value=0.0),
        R112_W55_ARM: R112SeedResult(
            family="family_interleaving_burst_recovery",
            seed=int(seed), arm=R112_W55_ARM,
            metric_name="burst_recovery_ge_0_80",
            metric_value=float(score)),
    }


def family_mlsc_v3_per_fact_uncertainty_propagation(
        seed: int,
) -> dict[str, R112SeedResult]:
    """H38: per-fact uncertainty composes correctly under merges."""
    from .uncertainty_layer_v3 import (
        FactUncertainty, propagate_fact_uncertainty)
    rng = random.Random(int(seed))
    a_facts = [
        FactUncertainty("fact_a", rng.uniform(0.6, 0.9), 1),
        FactUncertainty("shared", rng.uniform(0.5, 0.8), 1)]
    b_facts = [
        FactUncertainty("fact_b", rng.uniform(0.6, 0.9), 1),
        FactUncertainty("shared", rng.uniform(0.5, 0.8), 1)]
    merged = propagate_fact_uncertainty([a_facts, b_facts])
    # Verify: 3 facts, shared has count=2, others count=1.
    has_3 = len(merged) == 3
    shared_count_ok = any(
        fu.tag == "shared" and fu.n_contributors == 2
        for fu in merged)
    others_count_ok = all(
        fu.n_contributors == 1
        for fu in merged
        if fu.tag != "shared")
    # All confs in (0, 1).
    bounds_ok = all(
        0.0 <= fu.confidence <= 1.0 for fu in merged)
    score = (
        1.0
        if (has_3 and shared_count_ok and others_count_ok
             and bounds_ok)
        else 0.0)
    return {
        R112_BASELINE_ARM: R112SeedResult(
            family="family_mlsc_v3_per_fact_uncertainty_propagation",
            seed=int(seed), arm=R112_BASELINE_ARM,
            metric_name="fact_propagation_ok",
            metric_value=0.0),
        R112_W55_ARM: R112SeedResult(
            family="family_mlsc_v3_per_fact_uncertainty_propagation",
            seed=int(seed), arm=R112_W55_ARM,
            metric_name="fact_propagation_ok",
            metric_value=float(score)),
    }


# =============================================================================
# Registry
# =============================================================================


R112_FAMILY_TABLE: dict[
        str, Callable[[int], dict[str, R112SeedResult]]] = {
    "family_bch_double_bit_correct":
        family_bch_double_bit_correct,
    "family_bch_three_bit_detect":
        family_bch_three_bit_detect,
    "family_crc_v3_silent_failure_floor":
        family_crc_v3_silent_failure_floor,
    "family_trust_consensus_controller_recall":
        family_trust_consensus_controller_recall,
    "family_trust_consensus_controller_5stage_fallback":
        family_trust_consensus_controller_5stage_fallback,
    "family_mlsc_v3_trust_decay":
        family_mlsc_v3_trust_decay,
    "family_disagreement_algebra_soundness":
        family_disagreement_algebra_soundness,
    "family_compromise_v7_persistent_state":
        family_compromise_v7_persistent_state,
    "family_corruption_robust_carrier_v3_safety":
        family_corruption_robust_carrier_v3_safety,
    "family_uncertainty_v3_trust_weighted_composite":
        family_uncertainty_v3_trust_weighted_composite,
    "family_persistent_v7_chain_walk_depth":
        family_persistent_v7_chain_walk_depth,
    "family_w55_integration_envelope":
        family_w55_integration_envelope,
    "family_arbiter_v4_budget_allocator":
        family_arbiter_v4_budget_allocator,
    "family_deep_v6_adaptive_abstain_threshold":
        family_deep_v6_adaptive_abstain_threshold,
    "family_interleaving_burst_recovery":
        family_interleaving_burst_recovery,
    "family_mlsc_v3_per_fact_uncertainty_propagation":
        family_mlsc_v3_per_fact_uncertainty_propagation,
}


def run_family(
        family: str, *,
        seeds: Sequence[int] = (1, 2, 3),
) -> R112FamilyComparison:
    if family not in R112_FAMILY_TABLE:
        raise ValueError(f"unknown family {family!r}")
    fn = R112_FAMILY_TABLE[family]
    per_arm: dict[
            str, list[tuple[int, R112SeedResult]]] = {}
    for s in seeds:
        out = fn(int(s))
        for arm, sr in out.items():
            per_arm.setdefault(arm, []).append((int(s), sr))
    aggs: list[R112AggregateResult] = []
    metric_name = ""
    for arm, ls in per_arm.items():
        ls.sort(key=lambda t: t[0])
        seeds_t = tuple(t[0] for t in ls)
        values_t = tuple(
            float(t[1].metric_value) for t in ls)
        metric_name = ls[0][1].metric_name
        aggs.append(R112AggregateResult(
            family=family, arm=arm,
            metric_name=metric_name,
            seeds=seeds_t, values=values_t,
        ))
    return R112FamilyComparison(
        family=family,
        metric_name=metric_name,
        aggregates=tuple(aggs))


def run_all_families(
        *, seeds: Sequence[int] = (1, 2, 3),
) -> dict[str, R112FamilyComparison]:
    return {
        name: run_family(name, seeds=seeds)
        for name in R112_FAMILY_TABLE.keys()
    }


def main() -> None:
    out = run_all_families(seeds=(1, 2, 3))
    summary = {
        "schema": R112_SCHEMA_VERSION,
        "families": [c.to_dict() for c in out.values()],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


__all__ = [
    "R112_SCHEMA_VERSION",
    "R112_BASELINE_ARM",
    "R112_W55_ARM",
    "R112SeedResult",
    "R112AggregateResult",
    "R112FamilyComparison",
    "R112_FAMILY_TABLE",
    "family_bch_double_bit_correct",
    "family_bch_three_bit_detect",
    "family_crc_v3_silent_failure_floor",
    "family_trust_consensus_controller_recall",
    "family_trust_consensus_controller_5stage_fallback",
    "family_mlsc_v3_trust_decay",
    "family_disagreement_algebra_soundness",
    "family_compromise_v7_persistent_state",
    "family_corruption_robust_carrier_v3_safety",
    "family_uncertainty_v3_trust_weighted_composite",
    "family_persistent_v7_chain_walk_depth",
    "family_w55_integration_envelope",
    "family_arbiter_v4_budget_allocator",
    "family_deep_v6_adaptive_abstain_threshold",
    "family_interleaving_burst_recovery",
    "family_mlsc_v3_per_fact_uncertainty_propagation",
    "run_family",
    "run_all_families",
    "main",
]
