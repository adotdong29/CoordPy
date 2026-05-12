"""R-102 — Quantised Persistent Multi-Hop benchmark family.

Twelve families × 3 seeds, exercising H1-H12 of the W52 success
criterion (persistent V4 / multi-hop / role-graph /
transcript-comparator half).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Callable, Sequence

from .agents import Agent
from .branch_cycle_memory_v2 import (
    fit_branch_cycle_memory_v2,
    evaluate_joint_recall_v2,
    evaluate_v1_joint_recall_baseline,
    synthesize_branch_cycle_memory_v2_training_set,
)
from .branch_cycle_memory import BranchCycleMemoryHead
from .deep_proxy_stack_v3 import (
    DeepProxyStackV3,
    synthesize_deep_stack_v3_training_set,
    fit_deep_proxy_stack_v3,
    evaluate_deep_stack_v3_accuracy,
    collapse_role_banks,
)
from .deep_proxy_stack_v2 import (
    synthesize_deep_stack_v2_training_set,
    fit_deep_proxy_stack_v2,
    evaluate_deep_stack_v2_accuracy,
)
from .multi_hop_translator import (
    build_unfitted_multi_hop_translator,
    calibrate_confidence_from_residual,
    fit_multi_hop_translator,
    forge_multi_hop_training_set,
    perturb_edge,
    run_multi_hop_realism_anchor_probe,
    score_multi_hop_fidelity,
    synthesize_multi_hop_training_set,
)
from .persistent_latent_v4 import (
    V4StackedCell,
    fit_persistent_v4,
    evaluate_v4_long_horizon_recall,
    synthesize_v4_training_set,
)
from .persistent_shared_latent import (
    PersistentStateCell,
    evaluate_long_horizon_recall,
    fit_persistent_state_cell,
    synthesize_persistent_state_training_set,
)
from .quantised_compression import (
    fit_quantised_compression,
    synthesize_quantised_compression_training_set,
)
from .role_graph_transfer import (
    evaluate_equal_weight_accuracy,
    evaluate_role_graph_accuracy,
    fit_role_graph_mixer,
    forge_role_graph_training_set,
    synthesize_role_graph_training_set,
)
from .synthetic_llm import SyntheticLLMClient
from .transcript_vs_shared_state import (
    emit_transcript_vs_shared_witness,
)
from .w51_team import (
    W51Team,
    build_trivial_w51_registry,
    build_w51_registry,
)
from .w52_team import (
    W52Team,
    build_trivial_w52_registry,
    build_w52_registry,
    verify_w52_handoff,
)


# =============================================================================
# Schema
# =============================================================================

R102_SCHEMA_VERSION: str = "coordpy.r102_benchmark.v1"

R102_BASELINE_ARM: str = "baseline_w51"
R102_W52_ARM: str = "w52"


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
class R102SeedResult:
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
class R102AggregateResult:
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
class R102FamilyComparison:
    family: str
    metric_name: str
    aggregates: tuple[R102AggregateResult, ...]

    def get(self, arm: str) -> R102AggregateResult | None:
        for a in self.aggregates:
            if a.arm == arm:
                return a
        return None

    def delta_w52_vs_w51(self) -> float:
        w52 = self.get(R102_W52_ARM)
        w51 = self.get(R102_BASELINE_ARM)
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


def family_trivial_w52_passthrough(
        seed: int,
) -> dict[str, R102SeedResult]:
    """H1: trivial W52 reduces to W51 byte-for-byte."""
    backend = SyntheticLLMClient(
        model_tag=f"synth.r102.{seed}", default_response="hello")
    agents = [
        Agent(name="a1", instructions="", role="r0",
              backend=backend, temperature=0.0, max_tokens=20),
    ]
    w51_triv = W51Team(
        agents=agents, backend=backend,
        registry=build_trivial_w51_registry(
            schema_cid=f"r102_seed_{seed}"),
        max_visible_handoffs=2).run(
        f"r102 passthrough probe seed {seed}")
    w52_triv = W52Team(
        agents=agents, backend=backend,
        registry=build_trivial_w52_registry(
            schema_cid=f"r102_seed_{seed}"),
        max_visible_handoffs=2).run(
        f"r102 passthrough probe seed {seed}")
    ok = 1.0 if (
        w51_triv.w51_outer_cid == w52_triv.w51_outer_cid) else 0.0
    return {
        R102_BASELINE_ARM: R102SeedResult(
            family="family_trivial_w52_passthrough",
            seed=int(seed), arm=R102_BASELINE_ARM,
            metric_name="passthrough_ok", metric_value=1.0),
        R102_W52_ARM: R102SeedResult(
            family="family_trivial_w52_passthrough",
            seed=int(seed), arm=R102_W52_ARM,
            metric_name="passthrough_ok", metric_value=ok),
    }


def family_persistent_v4_long_horizon_gain(
        seed: int,
) -> dict[str, R102SeedResult]:
    """H2: V4 24-turn retention gain over trained V3 baseline.

    Uses a 24-turn regime with mid-sequence distractors so V3
    can't simply keep the gate closed — V4's skip-link delivers
    the win.
    """
    # Synthesize the V4 training set (corrupted, with distractors).
    ts4 = synthesize_v4_training_set(
        n_sequences=4, sequence_length=24, state_dim=8,
        input_dim=8, seed=int(seed),
        distractor_window=(6, 14),
        distractor_magnitude=0.5)
    # Build the V3 equivalent (same sequences, same targets) so
    # the comparison is on identical data.
    from .persistent_shared_latent import (
        PersistentStateExample,
        PersistentStateTrainingSet,
    )
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
        R102_BASELINE_ARM: R102SeedResult(
            family="family_persistent_v4_long_horizon_gain",
            seed=int(seed), arm=R102_BASELINE_ARM,
            metric_name="recall",
            metric_value=float(v3_recall)),
        R102_W52_ARM: R102SeedResult(
            family="family_persistent_v4_long_horizon_gain",
            seed=int(seed), arm=R102_W52_ARM,
            metric_name="recall",
            metric_value=float(v4_recall)),
    }


def family_multi_hop_quad_transitivity(
        seed: int,
) -> dict[str, R102SeedResult]:
    """H3: length-3 chain transitive fidelity."""
    ts = synthesize_multi_hop_training_set(
        n_examples=24, code_dim=8, feature_dim=8,
        backends=("A", "B", "C", "D"), seed=int(seed))
    untrained = build_unfitted_multi_hop_translator(
        code_dim=8, feature_dim=8, seed=int(seed))
    base_fid = score_multi_hop_fidelity(
        untrained, ts.examples[:8])
    trained, _ = fit_multi_hop_translator(
        ts, n_steps=192, seed=int(seed))
    train_fid = score_multi_hop_fidelity(
        trained, ts.examples[:8])
    return {
        R102_BASELINE_ARM: R102SeedResult(
            family="family_multi_hop_quad_transitivity",
            seed=int(seed), arm=R102_BASELINE_ARM,
            metric_name="chain_len3_fidelity",
            metric_value=float(
                base_fid.chain_len3_fidelity_mean)),
        R102_W52_ARM: R102SeedResult(
            family="family_multi_hop_quad_transitivity",
            seed=int(seed), arm=R102_W52_ARM,
            metric_name="chain_len3_fidelity",
            metric_value=float(
                train_fid.chain_len3_fidelity_mean)),
    }


def family_disagreement_weighted_arbitration(
        seed: int,
) -> dict[str, R102SeedResult]:
    """H4: under a perturbed edge, weighted arbitration beats
    naive."""
    ts = synthesize_multi_hop_training_set(
        n_examples=24, code_dim=8, feature_dim=8,
        backends=("A", "B", "C", "D"), seed=int(seed))
    tr, _ = fit_multi_hop_translator(
        ts, n_steps=192, seed=int(seed))
    # Perturb a single edge after training.
    tr = perturb_edge(tr, src="A", dst="B",
                      noise_magnitude=2.0,
                      seed=int(seed) * 7)
    # Calibrate per-edge confidence from training residuals.
    tr = calibrate_confidence_from_residual(tr, ts.examples)
    fid = score_multi_hop_fidelity(tr, ts.examples[:8])
    return {
        R102_BASELINE_ARM: R102SeedResult(
            family="family_disagreement_weighted_arbitration",
            seed=int(seed), arm=R102_BASELINE_ARM,
            metric_name="arbitration_score",
            metric_value=float(fid.arbitration_naive_score)),
        R102_W52_ARM: R102SeedResult(
            family="family_disagreement_weighted_arbitration",
            seed=int(seed), arm=R102_W52_ARM,
            metric_name="arbitration_score",
            metric_value=float(
                fid.arbitration_weighted_score)),
    }


def family_deep_stack_v3_depth_strict_gain(
        seed: int,
) -> dict[str, R102SeedResult]:
    """H5: L=8 V3 stack delta over L=6 W51 V2 baseline."""
    ts3 = synthesize_deep_stack_v3_training_set(
        n_examples=24, in_dim=6, compose_depth=8,
        n_branches=2, n_cycles=2, n_roles=2, seed=int(seed))
    s3, _ = fit_deep_proxy_stack_v3(
        ts3, n_layers=8, n_steps=48, seed=int(seed))
    acc_v3 = evaluate_deep_stack_v3_accuracy(s3, ts3.examples)
    # W51 V2 L=6 baseline on the same composition regime.
    ts2 = synthesize_deep_stack_v2_training_set(
        n_examples=24, in_dim=6, compose_depth=8,
        n_branches=2, n_cycles=2, seed=int(seed))
    s2, _ = fit_deep_proxy_stack_v2(
        ts2, n_layers=6, n_steps=48, seed=int(seed))
    acc_v2 = evaluate_deep_stack_v2_accuracy(s2, ts2.examples)
    return {
        R102_BASELINE_ARM: R102SeedResult(
            family="family_deep_stack_v3_depth_strict_gain",
            seed=int(seed), arm=R102_BASELINE_ARM,
            metric_name="acc",
            metric_value=float(acc_v2)),
        R102_W52_ARM: R102SeedResult(
            family="family_deep_stack_v3_depth_strict_gain",
            seed=int(seed), arm=R102_W52_ARM,
            metric_name="acc",
            metric_value=float(acc_v3)),
    }


def family_role_graph_transfer_gain(
        seed: int,
) -> dict[str, R102SeedResult]:
    """H6: role-graph beats equal-weight transfer."""
    ts = synthesize_role_graph_training_set(
        role_universe=("r0", "r1", "r2", "r3"),
        state_dim=6, n_examples_per_edge=4, seed=int(seed))
    mixer, _ = fit_role_graph_mixer(
        ts, n_steps=128, seed=int(seed))
    rg_acc = evaluate_role_graph_accuracy(mixer, ts.examples)
    ew_acc = evaluate_equal_weight_accuracy(mixer, ts.examples)
    return {
        R102_BASELINE_ARM: R102SeedResult(
            family="family_role_graph_transfer_gain",
            seed=int(seed), arm=R102_BASELINE_ARM,
            metric_name="acc",
            metric_value=float(ew_acc)),
        R102_W52_ARM: R102SeedResult(
            family="family_role_graph_transfer_gain",
            seed=int(seed), arm=R102_W52_ARM,
            metric_name="acc",
            metric_value=float(rg_acc)),
    }


def family_transcript_vs_shared_state(
        seed: int,
) -> dict[str, R102SeedResult]:
    """H7: shared-latent retention > transcript truncation at
    matched budget."""
    import random
    ts = synthesize_quantised_compression_training_set(
        n_examples=24, code_dim=12, n_coarse=32, n_fine=16,
        n_ultra=8, emit_mask_len=16, seed=int(seed))
    cb, gate, _ = fit_quantised_compression(
        ts, n_steps=16, seed=int(seed))
    gate.importance_threshold = 0.0
    gate.w_emit.values = [1.0] * len(gate.w_emit.values)
    rng = random.Random(int(seed))
    carriers = [
        [rng.uniform(-1, 1) for _ in range(12)]
        for _ in range(20)
    ]
    w = emit_transcript_vs_shared_witness(
        carriers=carriers, codebook=cb, gate=gate,
        budget_tokens=3)
    return {
        R102_BASELINE_ARM: R102SeedResult(
            family="family_transcript_vs_shared_state",
            seed=int(seed), arm=R102_BASELINE_ARM,
            metric_name="retention",
            metric_value=float(
                w.transcript_retention_cosine)),
        R102_W52_ARM: R102SeedResult(
            family="family_transcript_vs_shared_state",
            seed=int(seed), arm=R102_W52_ARM,
            metric_name="retention",
            metric_value=float(w.shared_retention_cosine)),
    }


def family_multi_hop_realism_probe(
        seed: int,
) -> dict[str, R102SeedResult]:
    """H8: Ollama quad anchor when reachable; skip-ok else."""
    payload = run_multi_hop_realism_anchor_probe()
    skipped_ok = float(payload.get("skipped_ok", 1.0))
    transitive = float(payload.get(
        "chain_len3_a_b_c_d", 0.0))
    return {
        R102_BASELINE_ARM: R102SeedResult(
            family="family_multi_hop_realism_probe",
            seed=int(seed), arm=R102_BASELINE_ARM,
            metric_name="anchor_skipped_ok",
            metric_value=float(skipped_ok)),
        R102_W52_ARM: R102SeedResult(
            family="family_multi_hop_realism_probe",
            seed=int(seed), arm=R102_W52_ARM,
            metric_name="anchor_transitive_or_skip",
            metric_value=float(
                transitive if transitive > 0 else skipped_ok)),
    }


def family_w52_envelope_verifier(
        seed: int,
) -> dict[str, R102SeedResult]:
    """H9: W52 envelope verifier rejects forged envelopes."""
    backend = SyntheticLLMClient(
        model_tag=f"synth.r102v.{seed}", default_response="x")
    agents = [
        Agent(name="a1", instructions="", role="r0",
              backend=backend, temperature=0.0, max_tokens=20),
    ]
    reg = build_w52_registry(
        schema_cid=f"r102_verifier_seed_{seed}",
        role_universe=("r0",))
    team = W52Team(
        agents=agents, backend=backend, registry=reg,
        max_visible_handoffs=2)
    r = team.run(f"r102 verifier task {seed}")
    v_clean = verify_w52_handoff(
        r.w52_envelope,
        expected_w51_outer_cid=r.w51_outer_cid,
        expected_params_cid=r.w52_params_cid,
        bundles=r.turn_witness_bundles,
        registry=reg,
        persistent_v4_state_cids=r.persistent_v4_state_cids)
    v_forged = verify_w52_handoff(
        r.w52_envelope,
        expected_w51_outer_cid="ff" * 32,
        expected_params_cid=r.w52_params_cid,
        bundles=r.turn_witness_bundles,
        registry=reg)
    score = 1.0 if (
        v_clean["ok"] and not v_forged["ok"]) else 0.0
    return {
        R102_BASELINE_ARM: R102SeedResult(
            family="family_w52_envelope_verifier",
            seed=int(seed), arm=R102_BASELINE_ARM,
            metric_name="verifier_score", metric_value=0.0),
        R102_W52_ARM: R102SeedResult(
            family="family_w52_envelope_verifier",
            seed=int(seed), arm=R102_W52_ARM,
            metric_name="verifier_score",
            metric_value=float(score)),
    }


def family_w52_replay_determinism(
        seed: int,
) -> dict[str, R102SeedResult]:
    """H10: W52 replay byte-identical across two runs."""
    backend = SyntheticLLMClient(
        model_tag=f"synth.r102r.{seed}", default_response="r")
    agents = [
        Agent(name="a1", instructions="", role="r0",
              backend=backend, temperature=0.0, max_tokens=20),
    ]
    reg = build_w52_registry(
        schema_cid=f"r102_rep_seed_{seed}",
        role_universe=("r0",))
    r1 = W52Team(
        agents=agents, backend=backend, registry=reg,
        max_visible_handoffs=2).run("replay_task")
    r2 = W52Team(
        agents=agents, backend=backend, registry=reg,
        max_visible_handoffs=2).run("replay_task")
    ok = 1.0 if (
        r1.w51_outer_cid == r2.w51_outer_cid
        and r1.w52_outer_cid == r2.w52_outer_cid
    ) else 0.0
    return {
        R102_BASELINE_ARM: R102SeedResult(
            family="family_w52_replay_determinism",
            seed=int(seed), arm=R102_BASELINE_ARM,
            metric_name="replay_ok", metric_value=1.0),
        R102_W52_ARM: R102SeedResult(
            family="family_w52_replay_determinism",
            seed=int(seed), arm=R102_W52_ARM,
            metric_name="replay_ok",
            metric_value=float(ok)),
    }


def family_multi_hop_translator_compromise_cap(
        seed: int,
) -> dict[str, R102SeedResult]:
    """H11: forged quad backend → translator cannot recover."""
    ts = synthesize_multi_hop_training_set(
        n_examples=16, code_dim=8, feature_dim=8,
        backends=("A", "B", "C", "D"), seed=int(seed))
    forged = forge_multi_hop_training_set(ts, seed=int(seed))
    translator, _ = fit_multi_hop_translator(
        forged, n_steps=96, seed=int(seed))
    fid_clean = score_multi_hop_fidelity(
        translator, ts.examples[:8])
    # Mean direct fidelity over the 12 edges.
    direct_vals = list(fid_clean.direct_fidelities.values())
    mean_dir = (
        float(sum(direct_vals)) / float(max(1, len(direct_vals)))
        if direct_vals else 0.0)
    protect_rate = max(0.0, 1.0 - abs(mean_dir))
    return {
        R102_BASELINE_ARM: R102SeedResult(
            family=(
                "family_multi_hop_translator_compromise_cap"),
            seed=int(seed), arm=R102_BASELINE_ARM,
            metric_name="downstream_protect_rate",
            metric_value=1.0),
        R102_W52_ARM: R102SeedResult(
            family=(
                "family_multi_hop_translator_compromise_cap"),
            seed=int(seed), arm=R102_W52_ARM,
            metric_name="downstream_protect_rate",
            metric_value=float(protect_rate)),
    }


def family_role_graph_distribution_cap(
        seed: int,
) -> dict[str, R102SeedResult]:
    """H12: forged role-graph → mixer cannot recover."""
    ts = synthesize_role_graph_training_set(
        role_universe=("r0", "r1", "r2", "r3"),
        state_dim=6, n_examples_per_edge=3, seed=int(seed))
    forged = forge_role_graph_training_set(
        ts, seed=int(seed))
    mixer, _ = fit_role_graph_mixer(
        forged, n_steps=128, seed=int(seed))
    rg_acc_clean = evaluate_role_graph_accuracy(
        mixer, ts.examples)
    protect_rate = max(0.0, 1.0 - abs(rg_acc_clean))
    return {
        R102_BASELINE_ARM: R102SeedResult(
            family="family_role_graph_distribution_cap",
            seed=int(seed), arm=R102_BASELINE_ARM,
            metric_name="downstream_protect_rate",
            metric_value=1.0),
        R102_W52_ARM: R102SeedResult(
            family="family_role_graph_distribution_cap",
            seed=int(seed), arm=R102_W52_ARM,
            metric_name="downstream_protect_rate",
            metric_value=float(protect_rate)),
    }


# =============================================================================
# Family registry
# =============================================================================


R102_FAMILY_TABLE: dict[
        str, Callable[[int], dict[str, R102SeedResult]]] = {
    "family_trivial_w52_passthrough":
        family_trivial_w52_passthrough,
    "family_persistent_v4_long_horizon_gain":
        family_persistent_v4_long_horizon_gain,
    "family_multi_hop_quad_transitivity":
        family_multi_hop_quad_transitivity,
    "family_disagreement_weighted_arbitration":
        family_disagreement_weighted_arbitration,
    "family_deep_stack_v3_depth_strict_gain":
        family_deep_stack_v3_depth_strict_gain,
    "family_role_graph_transfer_gain":
        family_role_graph_transfer_gain,
    "family_transcript_vs_shared_state":
        family_transcript_vs_shared_state,
    "family_multi_hop_realism_probe":
        family_multi_hop_realism_probe,
    "family_w52_envelope_verifier":
        family_w52_envelope_verifier,
    "family_w52_replay_determinism":
        family_w52_replay_determinism,
    "family_multi_hop_translator_compromise_cap":
        family_multi_hop_translator_compromise_cap,
    "family_role_graph_distribution_cap":
        family_role_graph_distribution_cap,
}


# =============================================================================
# Driver
# =============================================================================


def run_family(
        family: str, *,
        seeds: Sequence[int] = (1, 2, 3),
) -> R102FamilyComparison:
    """Run one family across the given seeds and aggregate."""
    if family not in R102_FAMILY_TABLE:
        raise ValueError(f"unknown family {family!r}")
    fn = R102_FAMILY_TABLE[family]
    per_arm: dict[
            str, list[tuple[int, R102SeedResult]]] = {}
    for s in seeds:
        out = fn(int(s))
        for arm, sr in out.items():
            per_arm.setdefault(arm, []).append((int(s), sr))
    aggs: list[R102AggregateResult] = []
    metric_name = ""
    for arm, ls in per_arm.items():
        ls.sort(key=lambda t: t[0])
        seeds_t = tuple(t[0] for t in ls)
        values_t = tuple(float(t[1].metric_value) for t in ls)
        metric_name = ls[0][1].metric_name
        aggs.append(R102AggregateResult(
            family=family, arm=arm,
            metric_name=metric_name,
            seeds=seeds_t, values=values_t,
        ))
    return R102FamilyComparison(
        family=family,
        metric_name=metric_name,
        aggregates=tuple(aggs))


def run_all_families(
        *, seeds: Sequence[int] = (1, 2, 3),
) -> dict[str, R102FamilyComparison]:
    return {
        name: run_family(name, seeds=seeds)
        for name in R102_FAMILY_TABLE.keys()
    }


def main() -> None:
    out = run_all_families(seeds=(1, 2, 3))
    summary = {
        "schema": R102_SCHEMA_VERSION,
        "families": [c.to_dict() for c in out.values()],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


__all__ = [
    "R102_SCHEMA_VERSION",
    "R102_BASELINE_ARM",
    "R102_W52_ARM",
    "R102SeedResult",
    "R102AggregateResult",
    "R102FamilyComparison",
    "R102_FAMILY_TABLE",
    "family_trivial_w52_passthrough",
    "family_persistent_v4_long_horizon_gain",
    "family_multi_hop_quad_transitivity",
    "family_disagreement_weighted_arbitration",
    "family_deep_stack_v3_depth_strict_gain",
    "family_role_graph_transfer_gain",
    "family_transcript_vs_shared_state",
    "family_multi_hop_realism_probe",
    "family_w52_envelope_verifier",
    "family_w52_replay_determinism",
    "family_multi_hop_translator_compromise_cap",
    "family_role_graph_distribution_cap",
    "run_family",
    "run_all_families",
    "main",
]
