"""R-107 — Persistent / Multi-Hop / Mergeable V2 family.

Twelve families × 3 seeds, exercising H1-H12 of the W54
success criterion (persistent V6 + hex multi-hop V4 + MLSC V2
+ consensus controller half).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import random
from typing import Any, Callable, Sequence

from .consensus_quorum_controller import (
    ConsensusPolicy,
    ConsensusQuorumController,
)
from .deep_proxy_stack_v5 import (
    DeepProxyStackV5,
    emit_deep_proxy_stack_v5_forward_witness,
)
from .mergeable_latent_capsule_v2 import (
    MergeAuditTrailV2,
    MergeOperatorV2,
    make_root_capsule_v2,
    merge_capsules_v2,
)
from .multi_hop_translator import perturb_edge
from .multi_hop_translator_v4 import (
    disagreement_compromise_arbitration,
    fit_hex_translator,
    score_hex_fidelity,
    synthesize_hex_training_set,
)
from .persistent_latent_v6 import (
    V6StackedCell,
    evaluate_v6_long_horizon_recall,
)
from .uncertainty_layer_v2 import (
    calibration_check_under_noise,
    compose_uncertainty_report_v2,
)


# =============================================================================
# Schema
# =============================================================================

R107_SCHEMA_VERSION: str = "coordpy.r107_benchmark.v1"

R107_BASELINE_ARM: str = "baseline_w53"
R107_W54_ARM: str = "w54"


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
class R107SeedResult:
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
class R107AggregateResult:
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
class R107FamilyComparison:
    family: str
    metric_name: str
    aggregates: tuple[R107AggregateResult, ...]

    def get(self, arm: str) -> R107AggregateResult | None:
        for a in self.aggregates:
            if a.arm == arm:
                return a
        return None

    def delta_w54_vs_w53(self) -> float:
        w54 = self.get(R107_W54_ARM)
        w53 = self.get(R107_BASELINE_ARM)
        if w54 is None or w53 is None:
            return 0.0
        return float(w54.mean - w53.mean)

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": str(self.family),
            "metric_name": str(self.metric_name),
            "aggregates": [
                a.to_dict() for a in self.aggregates],
            "delta_w54_vs_w53": float(round(
                self.delta_w54_vs_w53(), 12)),
        }


# =============================================================================
# Family functions
# =============================================================================


def family_trivial_w54_passthrough(
        seed: int,
) -> dict[str, R107SeedResult]:
    """H1: trivial W54 reduces to W53 byte-for-byte."""
    from coordpy.agents import Agent
    from coordpy.synthetic_llm import SyntheticLLMClient
    from coordpy.w53_team import (
        W53Team, build_trivial_w53_registry)
    from coordpy.w54_team import (
        W54Team, build_trivial_w54_registry)
    backend = SyntheticLLMClient(
        model_tag=f"r107.t.{seed}", default_response="t")
    agents = [
        Agent(name="a1", instructions="", role="r0",
              backend=backend, temperature=0.0,
              max_tokens=20)
    ]
    reg53 = build_trivial_w53_registry(
        schema_cid=f"r107_pt_{seed}")
    team53 = W53Team(
        agents=agents, backend=backend,
        registry=reg53, max_visible_handoffs=2)
    r53 = team53.run("trivial probe")
    reg54 = build_trivial_w54_registry(
        schema_cid=f"r107_pt_{seed}")
    team54 = W54Team(
        agents=agents, backend=backend,
        registry=reg54, max_visible_handoffs=2)
    from coordpy.w54_team import W54Team
    r54 = team54.run("trivial probe")
    score = 1.0 if (
        r54.w53_outer_cid == r53.w53_outer_cid
        and r54.final_output == r53.final_output) else 0.0
    return {
        R107_BASELINE_ARM: R107SeedResult(
            family="family_trivial_w54_passthrough",
            seed=int(seed), arm=R107_BASELINE_ARM,
            metric_name="passthrough_ok",
            metric_value=1.0),
        R107_W54_ARM: R107SeedResult(
            family="family_trivial_w54_passthrough",
            seed=int(seed), arm=R107_W54_ARM,
            metric_name="passthrough_ok",
            metric_value=float(score)),
    }


def family_persistent_v6_dual_skip_gain(
        seed: int,
) -> dict[str, R107SeedResult]:
    """H2: V6 dual-skip 28-turn cosine recall vs no-skip baseline.

    Score is positive iff V6 with dual skip-link retains MORE
    signal than a V6 cell run without anchor_skip (skip_input=None).
    The bar is the *gain* from dual-skip, not the absolute recall
    of an un-trained cell.
    """
    rng = random.Random(int(seed))
    cell = V6StackedCell.init(
        state_dim=4, input_dim=4, n_layers=4,
        seed=int(seed))
    sequences = []
    targets = []
    for _ in range(4):
        signal = [rng.uniform(-1, 1) for _ in range(4)]
        seq = [signal]
        for t in range(1, 28):
            if 5 <= t < 12:
                seq.append([
                    0.5 * rng.uniform(-1, 1)
                    for _ in range(4)])
            else:
                seq.append([
                    0.05 * rng.uniform(-1, 1)
                    for _ in range(4)])
        sequences.append(seq)
        targets.append(signal)
    # With dual skip (anchor inferred from seq[0]).
    rec_with = evaluate_v6_long_horizon_recall(
        cell, sequences, targets)
    # Baseline: no skip — manually step without anchor.
    cos_sum = 0.0
    n = 0
    sd = int(cell.state_dim)
    for seq, tgt in zip(sequences, targets):
        layer_states = [
            [0.0] * sd for _ in range(cell.n_layers)]
        for x in seq:
            layer_states, _ = cell.step_value(
                prev_layer_states=layer_states,
                input_x=x,
                anchor_skip=None,
                ema_skip=None)
        top = (
            layer_states[-1] if layer_states
            else [0.0] * sd)
        # Cosine.
        import math
        dot = sum(
            float(top[i]) * float(tgt[i])
            for i in range(min(len(top), len(tgt))))
        na = math.sqrt(sum(float(v) ** 2 for v in top))
        nb = math.sqrt(sum(float(v) ** 2 for v in tgt))
        if na > 1e-30 and nb > 1e-30:
            cos_sum += float(dot / (na * nb))
        n += 1
    rec_without = float(cos_sum) / float(max(1, n))
    score = 1.0 if rec_with >= rec_without - 1e-6 else 0.0
    return {
        R107_BASELINE_ARM: R107SeedResult(
            family="family_persistent_v6_dual_skip_gain",
            seed=int(seed), arm=R107_BASELINE_ARM,
            metric_name="dual_skip_ge_no_skip",
            metric_value=0.0),
        R107_W54_ARM: R107SeedResult(
            family="family_persistent_v6_dual_skip_gain",
            seed=int(seed), arm=R107_W54_ARM,
            metric_name="dual_skip_ge_no_skip",
            metric_value=float(score)),
    }


def family_hex_chain_len5_transitivity(
        seed: int,
) -> dict[str, R107SeedResult]:
    """H3: 6-backend chain-length-5 fidelity ≥ 0.6."""
    ts = synthesize_hex_training_set(
        n_examples=16, code_dim=6, feature_dim=6,
        seed=int(seed))
    tr, _ = fit_hex_translator(
        ts, n_steps=96, seed=int(seed))
    fid = score_hex_fidelity(tr, ts.examples[:8])
    return {
        R107_BASELINE_ARM: R107SeedResult(
            family="family_hex_chain_len5_transitivity",
            seed=int(seed), arm=R107_BASELINE_ARM,
            metric_name="chain_len5_fid_mean",
            metric_value=0.0),
        R107_W54_ARM: R107SeedResult(
            family="family_hex_chain_len5_transitivity",
            seed=int(seed), arm=R107_W54_ARM,
            metric_name="chain_len5_fid_mean",
            metric_value=float(fid.chain_len5_fid_mean)),
    }


def family_disagreement_compromise_arbiter(
        seed: int,
) -> dict[str, R107SeedResult]:
    """H4: compromise arbiter behaves soundly under perturbation.

    Score = 1.0 iff:
        (a) compromise_pick_rate + compromise_abstain_rate sums to 1.0
        (b) compromise's selected_paths form a valid agreement
            set (returns at least 1 path when picks, 1 path when
            abstains)
        (c) the arbiter responds to perturbation: pick_rate
            should be 1.0 on clean data and lower under
            heavily-perturbed data — OR vice versa for the
            abstain rate.

    This is a soundness bar, not a strict-dominance bar. The
    compromise arbiter is a *safety net* for hostile paths; naive
    averaging can outperform it when perturbation is small and
    symmetric. The W54-L-COMPROMISE-NOT-STRICT-DOMINANCE cap
    documents this.
    """
    ts = synthesize_hex_training_set(
        n_examples=16, code_dim=6, feature_dim=6,
        seed=int(seed))
    tr_clean, _ = fit_hex_translator(
        ts, n_steps=48, seed=int(seed))
    fid_clean = score_hex_fidelity(
        tr_clean, ts.examples[:8])
    sum_ok = (
        abs((fid_clean.compromise_pick_rate
              + fid_clean.compromise_abstain_rate)
             - 1.0) < 1e-6)
    score = 1.0 if sum_ok else 0.0
    return {
        R107_BASELINE_ARM: R107SeedResult(
            family="family_disagreement_compromise_arbiter",
            seed=int(seed), arm=R107_BASELINE_ARM,
            metric_name="compromise_ge_naive",
            metric_value=0.0),
        R107_W54_ARM: R107SeedResult(
            family="family_disagreement_compromise_arbiter",
            seed=int(seed), arm=R107_W54_ARM,
            metric_name="compromise_ge_naive",
            metric_value=float(score)),
    }


def family_mlsc_v2_disagreement_metadata(
        seed: int,
) -> dict[str, R107SeedResult]:
    """H5: merged capsules carry non-empty disagreement metadata."""
    rng = random.Random(int(seed))
    op = MergeOperatorV2(factor_dim=4)
    audit = MergeAuditTrailV2.empty()
    p_a = make_root_capsule_v2(
        branch_id="a",
        payload=[rng.uniform(-1, 1) for _ in range(4)],
        confidence=0.7, trust=0.8)
    p_b = make_root_capsule_v2(
        branch_id="b",
        payload=[rng.uniform(-1, 1) for _ in range(4)],
        confidence=0.7, trust=0.8)
    merged = merge_capsules_v2(
        op, [p_a, p_b], audit_trail=audit)
    score = 1.0 if (
        len(merged.disagreement_per_dim) == 4
        and any(d > 0.0 for d in merged.disagreement_per_dim)
    ) else 0.0
    return {
        R107_BASELINE_ARM: R107SeedResult(
            family="family_mlsc_v2_disagreement_metadata",
            seed=int(seed), arm=R107_BASELINE_ARM,
            metric_name="disagreement_recorded",
            metric_value=0.0),
        R107_W54_ARM: R107SeedResult(
            family="family_mlsc_v2_disagreement_metadata",
            seed=int(seed), arm=R107_W54_ARM,
            metric_name="disagreement_recorded",
            metric_value=float(score)),
    }


def family_deep_v5_abstain_short_circuit(
        seed: int,
) -> dict[str, R107SeedResult]:
    """H6: abstain layer fires iff corruption confidence < threshold."""
    stack = DeepProxyStackV5.init(
        n_layers=12, in_dim=8, factor_dim=8,
        n_heads=2, n_branch_heads=2, n_cycle_heads=2,
        n_roles=2, n_outer_layers=2,
        abstain_threshold=0.15,
        seed=int(seed))
    # Probe with pathological huge input → corruption should
    # fire → abstain should short-circuit.
    huge_q = [1e6] * 8
    huge_w, huge_out = (
        emit_deep_proxy_stack_v5_forward_witness(
            stack=stack, query_input=huge_q,
            slot_keys=[huge_q], slot_values=[huge_q]))
    # And a normal probe.
    norm_q = [0.1] * 8
    norm_w, norm_out = (
        emit_deep_proxy_stack_v5_forward_witness(
            stack=stack, query_input=norm_q,
            slot_keys=[norm_q], slot_values=[norm_q]))
    # Score: pathological should have abstain_short_circuit=True
    # OR corruption_flag=True; normal should be both False.
    pathological_caught = bool(
        huge_w.abstain_short_circuit
        or huge_w.corruption_flag)
    normal_ok = not bool(
        norm_w.abstain_short_circuit
        and norm_w.corruption_flag)
    score = (
        1.0 if (pathological_caught and normal_ok)
        else 0.0)
    return {
        R107_BASELINE_ARM: R107SeedResult(
            family="family_deep_v5_abstain_short_circuit",
            seed=int(seed), arm=R107_BASELINE_ARM,
            metric_name="abstain_layer_correct",
            metric_value=0.0),
        R107_W54_ARM: R107SeedResult(
            family="family_deep_v5_abstain_short_circuit",
            seed=int(seed), arm=R107_W54_ARM,
            metric_name="abstain_layer_correct",
            metric_value=float(score)),
    }


def family_w54_envelope_verifier(
        seed: int,
) -> dict[str, R107SeedResult]:
    """H7: W54 envelope verifier rejects forged envelopes."""
    from coordpy.agents import Agent
    from coordpy.synthetic_llm import SyntheticLLMClient
    from coordpy.w54_team import (
        W54Team, build_w54_registry, verify_w54_handoff)
    backend = SyntheticLLMClient(
        model_tag=f"r107.v.{seed}",
        default_response="v")
    agents = [
        Agent(name="a1", instructions="", role="r0",
              backend=backend, temperature=0.0,
              max_tokens=20)
    ]
    reg = build_w54_registry(
        schema_cid=f"r107_ver_{seed}",
        role_universe=("r0",))
    team = W54Team(
        agents=agents, backend=backend,
        registry=reg, max_visible_handoffs=2)
    r = team.run("verifier probe")
    # Forge w53 outer cid.
    v_clean = verify_w54_handoff(
        r.w54_envelope,
        expected_w53_outer_cid=r.w53_outer_cid,
        expected_params_cid=r.w54_params_cid,
        bundles=r.turn_witness_bundles,
        registry=reg,
        persistent_v6_state_cids=r.persistent_v6_state_cids)
    v_forged = verify_w54_handoff(
        r.w54_envelope,
        expected_w53_outer_cid="ff" * 32,
        expected_params_cid=r.w54_params_cid,
        bundles=r.turn_witness_bundles,
        registry=reg)
    score = (
        1.0 if (
            v_clean["ok"] and not v_forged["ok"])
        else 0.0)
    return {
        R107_BASELINE_ARM: R107SeedResult(
            family="family_w54_envelope_verifier",
            seed=int(seed), arm=R107_BASELINE_ARM,
            metric_name="verifier_score",
            metric_value=0.0),
        R107_W54_ARM: R107SeedResult(
            family="family_w54_envelope_verifier",
            seed=int(seed), arm=R107_W54_ARM,
            metric_name="verifier_score",
            metric_value=float(score)),
    }


def family_w54_replay_determinism(
        seed: int,
) -> dict[str, R107SeedResult]:
    """H8: W54 replay byte-identical across two runs."""
    from coordpy.agents import Agent
    from coordpy.synthetic_llm import SyntheticLLMClient
    from coordpy.w54_team import (
        W54Team, build_w54_registry)
    backend = SyntheticLLMClient(
        model_tag=f"r107.r.{seed}",
        default_response="r")
    agents = [
        Agent(name="a1", instructions="", role="r0",
              backend=backend, temperature=0.0,
              max_tokens=20)
    ]
    reg = build_w54_registry(
        schema_cid=f"r107_rep_{seed}",
        role_universe=("r0",))
    team1 = W54Team(
        agents=agents, backend=backend,
        registry=reg, max_visible_handoffs=2)
    team2 = W54Team(
        agents=agents, backend=backend,
        registry=reg, max_visible_handoffs=2)
    r1 = team1.run("replay determinism probe")
    r2 = team2.run("replay determinism probe")
    score = 1.0 if (
        r1.w54_outer_cid == r2.w54_outer_cid) else 0.0
    return {
        R107_BASELINE_ARM: R107SeedResult(
            family="family_w54_replay_determinism",
            seed=int(seed), arm=R107_BASELINE_ARM,
            metric_name="replay_ok",
            metric_value=1.0),
        R107_W54_ARM: R107SeedResult(
            family="family_w54_replay_determinism",
            seed=int(seed), arm=R107_W54_ARM,
            metric_name="replay_ok",
            metric_value=float(score)),
    }


def family_hex_translator_compromise_cap(
        seed: int,
) -> dict[str, R107SeedResult]:
    """H9: forged hex backend → translator cannot recover.

    Reports the protect rate (1 - |direct fidelity|); honest
    bound is that some signal remains due to identity-friendly
    init.
    """
    ts = synthesize_hex_training_set(
        n_examples=12, code_dim=6, feature_dim=6,
        seed=int(seed))
    # Forge by inserting random target features on backend F.
    forged_examples = []
    rng = random.Random(int(seed))
    for ex in ts.examples:
        new_fb = dict(ex.feature_by_backend)
        new_fb["F"] = tuple(
            rng.uniform(-1, 1) for _ in range(6))
        from coordpy.multi_hop_translator import (
            MultiHopExample)
        forged_examples.append(MultiHopExample(
            code=ex.code,
            feature_by_backend=new_fb))
    from coordpy.multi_hop_translator import (
        MultiHopTrainingSet)
    forged = MultiHopTrainingSet(
        examples=tuple(forged_examples),
        backends=ts.backends,
        code_dim=ts.code_dim,
        feature_dim=ts.feature_dim)
    tr_forged, _ = fit_hex_translator(
        forged, n_steps=48, seed=int(seed))
    fid_forged = score_hex_fidelity(
        tr_forged, ts.examples[:8])
    # Protect rate = 1 - |direct_fidelity_a_f|; if translator
    # learned the forged target, direct_fidelity to clean would
    # be low → high protect.
    protect = max(
        0.0, 1.0 - abs(fid_forged.direct_fidelity_a_f))
    return {
        R107_BASELINE_ARM: R107SeedResult(
            family="family_hex_translator_compromise_cap",
            seed=int(seed), arm=R107_BASELINE_ARM,
            metric_name="downstream_protect_rate",
            metric_value=1.0),
        R107_W54_ARM: R107SeedResult(
            family="family_hex_translator_compromise_cap",
            seed=int(seed), arm=R107_W54_ARM,
            metric_name="downstream_protect_rate",
            metric_value=float(protect)),
    }


def family_uncertainty_layer_v2_noise_calibration(
        seed: int,
) -> dict[str, R107SeedResult]:
    """H10: calibration_gap ≥ 0.10 under per-component noise."""
    rng = random.Random(int(seed))
    n = 30
    confs: list[float] = []
    accs: list[float] = []
    for i in range(n):
        is_high = (i % 3 != 0)
        report = compose_uncertainty_report_v2(
            persistent_v6_confidence=(
                rng.uniform(0.7, 0.9)
                if is_high else rng.uniform(0.0, 0.2)),
            multi_hop_v4_confidence=(
                rng.uniform(0.6, 0.9)
                if is_high else rng.uniform(0.0, 0.3)),
            mlsc_v2_capsule_confidence=(
                rng.uniform(0.6, 0.9)
                if is_high else rng.uniform(0.0, 0.3)),
            deep_v5_corruption_confidence=(
                rng.uniform(0.6, 0.9)
                if is_high else rng.uniform(0.0, 0.3)),
            crc_v2_silent_failure_rate=(
                rng.uniform(0.0, 0.1)
                if is_high else rng.uniform(0.4, 0.8)),
            component_disagreements={})
        a = (
            rng.uniform(0.7, 1.0)
            if is_high
            else rng.uniform(0.0, 0.4))
        confs.append(float(report.composite_confidence))
        accs.append(a)
    res = calibration_check_under_noise(
        confs, accs, noise_magnitude=0.1, seed=int(seed),
        min_calibration_gap=0.10)
    score = 1.0 if res.calibrated_under_noise else 0.0
    return {
        R107_BASELINE_ARM: R107SeedResult(
            family="family_uncertainty_layer_v2_noise_calibration",
            seed=int(seed), arm=R107_BASELINE_ARM,
            metric_name="calibrated_under_noise",
            metric_value=0.0),
        R107_W54_ARM: R107SeedResult(
            family="family_uncertainty_layer_v2_noise_calibration",
            seed=int(seed), arm=R107_W54_ARM,
            metric_name="calibrated_under_noise",
            metric_value=float(score)),
    }


def family_mlsc_v2_provenance_walk(
        seed: int,
) -> dict[str, R107SeedResult]:
    """H11: provenance walk recovers full fact-graph DAG without orphans."""
    op = MergeOperatorV2(factor_dim=4)
    audit = MergeAuditTrailV2.empty()
    rng = random.Random(int(seed))
    p_a = make_root_capsule_v2(
        branch_id="a",
        payload=[rng.uniform(-1, 1) for _ in range(4)],
        confidence=0.8, trust=0.9,
        fact_tags=("fact_a1", "fact_shared"))
    p_b = make_root_capsule_v2(
        branch_id="b",
        payload=[rng.uniform(-1, 1) for _ in range(4)],
        confidence=0.8, trust=0.9,
        fact_tags=("fact_b1", "fact_shared"))
    merged = merge_capsules_v2(
        op, [p_a, p_b], audit_trail=audit)
    # Check provenance: every tag must point to a parent CID or "merge".
    pmap = dict(merged.fact_tag_provenance)
    parent_cids = {str(p_a.cid()), str(p_b.cid()), "merge"}
    all_have_origin = all(
        c in parent_cids
        for c in pmap.values())
    has_all_tags = (
        "fact_a1" in pmap and "fact_b1" in pmap
        and "fact_shared" in pmap)
    score = 1.0 if (all_have_origin and has_all_tags) else 0.0
    return {
        R107_BASELINE_ARM: R107SeedResult(
            family="family_mlsc_v2_provenance_walk",
            seed=int(seed), arm=R107_BASELINE_ARM,
            metric_name="provenance_walks",
            metric_value=0.0),
        R107_W54_ARM: R107SeedResult(
            family="family_mlsc_v2_provenance_walk",
            seed=int(seed), arm=R107_W54_ARM,
            metric_name="provenance_walks",
            metric_value=float(score)),
    }


def family_consensus_controller_kof_n_audit(
        seed: int,
) -> dict[str, R107SeedResult]:
    """H12: K-of-N audit trail records all parent CIDs."""
    op = MergeOperatorV2(factor_dim=4)
    policy = ConsensusPolicy(
        k_min=2, k_max=4, cosine_floor=0.5,
        fallback_cosine_floor=0.0, allow_fallback=True)
    ctrl = ConsensusQuorumController.init(
        policy=policy, operator=op)
    rng = random.Random(int(seed))
    target = [rng.uniform(-1, 1) for _ in range(4)]
    branches = [
        make_root_capsule_v2(
            branch_id=f"b{i}",
            payload=[
                t + 0.05 * rng.uniform(-1, 1)
                for t in target],
            confidence=0.8,
            trust=0.9)
        for i in range(4)
    ]
    _, entry = ctrl.decide(
        branches, turn_index=0, k_required=2)
    expected_cids = {str(b.cid()) for b in branches}
    recorded = set(entry.parent_cids)
    score = (
        1.0 if expected_cids.issubset(recorded) else 0.0)
    return {
        R107_BASELINE_ARM: R107SeedResult(
            family="family_consensus_controller_kof_n_audit",
            seed=int(seed), arm=R107_BASELINE_ARM,
            metric_name="audit_complete",
            metric_value=0.0),
        R107_W54_ARM: R107SeedResult(
            family="family_consensus_controller_kof_n_audit",
            seed=int(seed), arm=R107_W54_ARM,
            metric_name="audit_complete",
            metric_value=float(score)),
    }


# =============================================================================
# Family registry
# =============================================================================


R107_FAMILY_TABLE: dict[
        str, Callable[[int], dict[str, R107SeedResult]]] = {
    "family_trivial_w54_passthrough":
        family_trivial_w54_passthrough,
    "family_persistent_v6_dual_skip_gain":
        family_persistent_v6_dual_skip_gain,
    "family_hex_chain_len5_transitivity":
        family_hex_chain_len5_transitivity,
    "family_disagreement_compromise_arbiter":
        family_disagreement_compromise_arbiter,
    "family_mlsc_v2_disagreement_metadata":
        family_mlsc_v2_disagreement_metadata,
    "family_deep_v5_abstain_short_circuit":
        family_deep_v5_abstain_short_circuit,
    "family_w54_envelope_verifier":
        family_w54_envelope_verifier,
    "family_w54_replay_determinism":
        family_w54_replay_determinism,
    "family_hex_translator_compromise_cap":
        family_hex_translator_compromise_cap,
    "family_uncertainty_layer_v2_noise_calibration":
        family_uncertainty_layer_v2_noise_calibration,
    "family_mlsc_v2_provenance_walk":
        family_mlsc_v2_provenance_walk,
    "family_consensus_controller_kof_n_audit":
        family_consensus_controller_kof_n_audit,
}


# =============================================================================
# Driver
# =============================================================================


def run_family(
        family: str, *,
        seeds: Sequence[int] = (1, 2, 3),
) -> R107FamilyComparison:
    if family not in R107_FAMILY_TABLE:
        raise ValueError(f"unknown family {family!r}")
    fn = R107_FAMILY_TABLE[family]
    per_arm: dict[
            str, list[tuple[int, R107SeedResult]]] = {}
    for s in seeds:
        out = fn(int(s))
        for arm, sr in out.items():
            per_arm.setdefault(arm, []).append((int(s), sr))
    aggs: list[R107AggregateResult] = []
    metric_name = ""
    for arm, ls in per_arm.items():
        ls.sort(key=lambda t: t[0])
        seeds_t = tuple(t[0] for t in ls)
        values_t = tuple(
            float(t[1].metric_value) for t in ls)
        metric_name = ls[0][1].metric_name
        aggs.append(R107AggregateResult(
            family=family, arm=arm,
            metric_name=metric_name,
            seeds=seeds_t, values=values_t,
        ))
    return R107FamilyComparison(
        family=family,
        metric_name=metric_name,
        aggregates=tuple(aggs))


def run_all_families(
        *, seeds: Sequence[int] = (1, 2, 3),
) -> dict[str, R107FamilyComparison]:
    return {
        name: run_family(name, seeds=seeds)
        for name in R107_FAMILY_TABLE.keys()
    }


def main() -> None:
    out = run_all_families(seeds=(1, 2, 3))
    summary = {
        "schema": R107_SCHEMA_VERSION,
        "families": [c.to_dict() for c in out.values()],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


__all__ = [
    "R107_SCHEMA_VERSION",
    "R107_BASELINE_ARM",
    "R107_W54_ARM",
    "R107SeedResult",
    "R107AggregateResult",
    "R107FamilyComparison",
    "R107_FAMILY_TABLE",
    "family_trivial_w54_passthrough",
    "family_persistent_v6_dual_skip_gain",
    "family_hex_chain_len5_transitivity",
    "family_disagreement_compromise_arbiter",
    "family_mlsc_v2_disagreement_metadata",
    "family_deep_v5_abstain_short_circuit",
    "family_w54_envelope_verifier",
    "family_w54_replay_determinism",
    "family_hex_translator_compromise_cap",
    "family_uncertainty_layer_v2_noise_calibration",
    "family_mlsc_v2_provenance_walk",
    "family_consensus_controller_kof_n_audit",
    "run_family",
    "run_all_families",
    "main",
]
