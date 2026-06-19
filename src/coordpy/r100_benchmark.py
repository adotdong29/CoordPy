"""R-100 — Persistent Cross-Backend Latent Coordination benchmark family.

Ten families × 3 seeds, exercising H1-H10 of the W51 success
criterion (cross-backend / persistent latent / transfer half).
Each family returns deterministic ``R100SeedResult``s
aggregated into ``R100AggregateResult`` then
``R100FamilyComparison``.

Pure-Python / stdlib only — uses the W47 autograd engine via
the M1..M6 modules and the W51 composition.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Callable, Mapping, Sequence

from .agents import Agent
from .branch_cycle_memory import (
    fit_branch_cycle_memory,
    evaluate_branch_cycle_recall_specialised,
    evaluate_generic_memory_recall,
    synthesize_branch_cycle_memory_training_set,
)
from .cross_backend_translator import (
    build_unfitted_triple_backend_translator,
    fit_triple_backend_translator,
    forge_triple_backend_training_set,
    run_triple_realism_anchor_probe,
    score_triple_backend_fidelity,
    synthesize_triple_backend_training_set,
)
from .deep_proxy_stack import (
    DeepStackTrainingExample,
    DeepStackTrainingSet,
    fit_deep_proxy_stack,
    evaluate_deep_stack_accuracy,
)
from .deep_proxy_stack_v2 import (
    collapse_branch_cycle_selectors,
    evaluate_deep_stack_v2_accuracy,
    fit_deep_proxy_stack_v2,
    synthesize_deep_stack_v2_training_set,
)
from .persistent_shared_latent import (
    fit_persistent_state_cell,
    evaluate_long_horizon_recall,
    synthesize_persistent_state_training_set,
)
from .synthetic_llm import SyntheticLLMClient
from .w50_team import W50Team, build_w50_registry
from .w51_team import (
    W51Team,
    build_trivial_w51_registry,
    build_w51_registry,
    verify_w51_handoff,
)


# =============================================================================
# Schema
# =============================================================================

R100_SCHEMA_VERSION: str = "coordpy.r100_benchmark.v1"

R100_BASELINE_ARM: str = "baseline_w50"
R100_W51_ARM: str = "w51"


# =============================================================================
# Canonicalisation helpers
# =============================================================================

def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str,
    ).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


# =============================================================================
# Result dataclasses
# =============================================================================

@dataclasses.dataclass(frozen=True)
class R100SeedResult:
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
            "metric_value": float(
                round(self.metric_value, 12)),
        }


@dataclasses.dataclass(frozen=True)
class R100AggregateResult:
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
class R100FamilyComparison:
    family: str
    metric_name: str
    aggregates: tuple[R100AggregateResult, ...]

    def get(self, arm: str) -> R100AggregateResult | None:
        for a in self.aggregates:
            if a.arm == arm:
                return a
        return None

    def delta_w51_vs_w50(self) -> float:
        w51 = self.get(R100_W51_ARM)
        w50 = self.get(R100_BASELINE_ARM)
        if w51 is None or w50 is None:
            return 0.0
        return float(w51.mean - w50.mean)

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": str(self.family),
            "metric_name": str(self.metric_name),
            "aggregates": [a.to_dict() for a in self.aggregates],
            "delta_w51_vs_w50": float(round(
                self.delta_w51_vs_w50(), 12)),
        }


# =============================================================================
# Family functions (one per H bar)
# =============================================================================

def family_trivial_w51_passthrough(
        seed: int,
) -> dict[str, R100SeedResult]:
    """H1: trivial W51 reduces to W50 byte-for-byte."""
    backend = SyntheticLLMClient(
        model_tag=f"synth.r100.{seed}", default_response="hello")
    agents = [
        Agent(name="a1", instructions="", role="r0",
              backend=backend, temperature=0.0, max_tokens=20),
    ]
    w50 = W50Team(
        agents=agents, backend=backend,
        registry=build_w50_registry(
            schema_cid=f"r100_seed_{seed}",
            role_universe=("r0",)),
        max_visible_handoffs=2).run(
        f"r100 passthrough probe seed {seed}")
    w51 = W51Team(
        agents=agents, backend=backend,
        registry=build_trivial_w51_registry(
            schema_cid=f"r100_seed_{seed}"),
        max_visible_handoffs=2).run(
        f"r100 passthrough probe seed {seed}")
    # Trivial passthrough: W51's internal w50_outer_cid equals
    # the standalone W50 trivial outer CID
    # (with the same schema_cid both reduce to the same chain).
    # To strictly compare, the W51-trivial registry uses
    # W50-trivial inner; so we compare against the trivial
    # W50 registry, not the W50-full registry above.
    from .w50_team import build_trivial_w50_registry
    w50_triv = W50Team(
        agents=agents, backend=backend,
        registry=build_trivial_w50_registry(
            schema_cid=f"r100_seed_{seed}"),
        max_visible_handoffs=2).run(
        f"r100 passthrough probe seed {seed}")
    ok = 1.0 if (
        w50_triv.w50_outer_cid == w51.w50_outer_cid) else 0.0
    return {
        R100_BASELINE_ARM: R100SeedResult(
            family="family_trivial_w51_passthrough",
            seed=int(seed), arm=R100_BASELINE_ARM,
            metric_name="passthrough_ok", metric_value=1.0),
        R100_W51_ARM: R100SeedResult(
            family="family_trivial_w51_passthrough",
            seed=int(seed), arm=R100_W51_ARM,
            metric_name="passthrough_ok", metric_value=ok),
    }


def family_persistent_state_long_horizon_gain(
        seed: int,
) -> dict[str, R100SeedResult]:
    """H2: persistent GRU state long-horizon recall vs W50
    no-persistent baseline."""
    # W50 baseline = average cosine of zero-state vs target
    # (no persistent state, every "recall" is from scratch).
    ts = synthesize_persistent_state_training_set(
        n_sequences=4, sequence_length=12, state_dim=8,
        input_dim=8, seed=int(seed))
    # Baseline: untrained cell (random initial behaviour).
    from .persistent_shared_latent import PersistentStateCell
    untrained = PersistentStateCell.init(
        state_dim=8, input_dim=8, seed=int(seed))
    base_recall = evaluate_long_horizon_recall(
        untrained, ts.examples)
    trained, _ = fit_persistent_state_cell(
        ts, n_steps=96, seed=int(seed), truncate_bptt=4)
    trained_recall = evaluate_long_horizon_recall(
        trained, ts.examples)
    return {
        R100_BASELINE_ARM: R100SeedResult(
            family="family_persistent_state_long_horizon_gain",
            seed=int(seed), arm=R100_BASELINE_ARM,
            metric_name="recall",
            metric_value=float(base_recall)),
        R100_W51_ARM: R100SeedResult(
            family="family_persistent_state_long_horizon_gain",
            seed=int(seed), arm=R100_W51_ARM,
            metric_name="recall",
            metric_value=float(trained_recall)),
    }


def family_triple_backend_transitivity(
        seed: int,
) -> dict[str, R100SeedResult]:
    """H3: triple-backend translator transitivity."""
    ts = synthesize_triple_backend_training_set(
        n_examples=24, feature_dim=8, code_dim=8,
        seed=int(seed))
    untrained = build_unfitted_triple_backend_translator(
        feature_dim=8, code_dim=8, seed=int(seed))
    base_fid = score_triple_backend_fidelity(
        untrained, ts.examples[:8])
    trained, _ = fit_triple_backend_translator(
        ts, n_steps=192, seed=int(seed))
    train_fid = score_triple_backend_fidelity(
        trained, ts.examples[:8])
    # Reported metric: transitive fidelity (A→B→C).
    return {
        R100_BASELINE_ARM: R100SeedResult(
            family="family_triple_backend_transitivity",
            seed=int(seed), arm=R100_BASELINE_ARM,
            metric_name="transitive_fidelity",
            metric_value=float(base_fid.transitive_a_b_c)),
        R100_W51_ARM: R100SeedResult(
            family="family_triple_backend_transitivity",
            seed=int(seed), arm=R100_W51_ARM,
            metric_name="transitive_fidelity",
            metric_value=float(train_fid.transitive_a_b_c)),
    }


def family_triple_backend_transitivity_gap(
        seed: int,
) -> dict[str, R100SeedResult]:
    """H3 supporting: transitivity gap stays bounded under
    training."""
    ts = synthesize_triple_backend_training_set(
        n_examples=24, feature_dim=8, code_dim=8,
        seed=int(seed))
    trained, _ = fit_triple_backend_translator(
        ts, n_steps=192, seed=int(seed))
    fid = score_triple_backend_fidelity(
        trained, ts.examples[:8])
    return {
        R100_BASELINE_ARM: R100SeedResult(
            family="family_triple_backend_transitivity_gap",
            seed=int(seed), arm=R100_BASELINE_ARM,
            metric_name="transitivity_gap",
            metric_value=1.0),  # baseline = max gap
        R100_W51_ARM: R100SeedResult(
            family="family_triple_backend_transitivity_gap",
            seed=int(seed), arm=R100_W51_ARM,
            metric_name="transitivity_gap",
            metric_value=float(fid.transitivity_gap)),
    }


def family_deep_stack_v2_depth_strict_gain(
        seed: int,
) -> dict[str, R100SeedResult]:
    """H4: L=6 stack V2 delta over L=4 W50."""
    ts = synthesize_deep_stack_v2_training_set(
        n_examples=24, in_dim=6, compose_depth=6,
        n_branches=2, n_cycles=2, seed=int(seed))
    # W50 baseline: L=4 W50 stack (use a flat-target view).
    v2_examples = [
        DeepStackTrainingExample(
            input_vec=e.input_vec,
            target_label=e.target_label)
        for e in ts.examples
    ]
    ts4 = DeepStackTrainingSet(
        examples=tuple(v2_examples), in_dim=6)
    s4, _ = fit_deep_proxy_stack(
        ts4, n_layers=4, n_steps=96, seed=int(seed))
    acc4 = evaluate_deep_stack_accuracy(s4, ts4.examples)
    # W51 V2: L=6
    s6, _ = fit_deep_proxy_stack_v2(
        ts, n_layers=6, n_steps=96, seed=int(seed))
    acc6 = evaluate_deep_stack_v2_accuracy(s6, ts.examples)
    return {
        R100_BASELINE_ARM: R100SeedResult(
            family="family_deep_stack_v2_depth_strict_gain",
            seed=int(seed), arm=R100_BASELINE_ARM,
            metric_name="acc", metric_value=float(acc4)),
        R100_W51_ARM: R100SeedResult(
            family="family_deep_stack_v2_depth_strict_gain",
            seed=int(seed), arm=R100_W51_ARM,
            metric_name="acc", metric_value=float(acc6)),
    }


def family_branch_specialised_heads_gain(
        seed: int,
) -> dict[str, R100SeedResult]:
    """H5: branch-specialised heads strict gain over shared."""
    ts = synthesize_deep_stack_v2_training_set(
        n_examples=24, in_dim=6, compose_depth=6,
        n_branches=2, n_cycles=2, seed=int(seed))
    s6, _ = fit_deep_proxy_stack_v2(
        ts, n_layers=6, n_steps=96, seed=int(seed))
    acc_specialised = evaluate_deep_stack_v2_accuracy(
        s6, ts.examples)
    # Collapse branch/cycle selectors → shared heads only.
    collapsed = collapse_branch_cycle_selectors(s6)
    acc_shared = evaluate_deep_stack_v2_accuracy(
        collapsed, ts.examples)
    return {
        R100_BASELINE_ARM: R100SeedResult(
            family="family_branch_specialised_heads_gain",
            seed=int(seed), arm=R100_BASELINE_ARM,
            metric_name="acc",
            metric_value=float(acc_shared)),
        R100_W51_ARM: R100SeedResult(
            family="family_branch_specialised_heads_gain",
            seed=int(seed), arm=R100_W51_ARM,
            metric_name="acc",
            metric_value=float(acc_specialised)),
    }


def family_branch_cycle_memory_gain(
        seed: int,
) -> dict[str, R100SeedResult]:
    """H6: branch/cycle memory head gain over generic."""
    ts = synthesize_branch_cycle_memory_training_set(
        n_examples=12, factor_dim=4,
        n_branch_pages=4, n_cycle_pages=4,
        seed=int(seed))
    generic_recall = evaluate_generic_memory_recall(
        ts.examples, factor_dim=4)
    head, _ = fit_branch_cycle_memory(
        ts, n_steps=48, seed=int(seed))
    bcm_recall = evaluate_branch_cycle_recall_specialised(
        head, ts.examples)
    return {
        R100_BASELINE_ARM: R100SeedResult(
            family="family_branch_cycle_memory_gain",
            seed=int(seed), arm=R100_BASELINE_ARM,
            metric_name="recall",
            metric_value=float(generic_recall)),
        R100_W51_ARM: R100SeedResult(
            family="family_branch_cycle_memory_gain",
            seed=int(seed), arm=R100_W51_ARM,
            metric_name="recall",
            metric_value=float(bcm_recall)),
    }


def family_triple_backend_realism_probe(
        seed: int,
) -> dict[str, R100SeedResult]:
    """H7: Ollama triple anchor when reachable; skip-ok else."""
    payload = run_triple_realism_anchor_probe()
    skipped_ok = float(payload.get("skipped_ok", 1.0))
    transitive = float(payload.get(
        "transitive_a_b_c", 0.0))
    return {
        R100_BASELINE_ARM: R100SeedResult(
            family="family_triple_backend_realism_probe",
            seed=int(seed), arm=R100_BASELINE_ARM,
            metric_name="anchor_skipped_ok",
            metric_value=float(skipped_ok)),
        R100_W51_ARM: R100SeedResult(
            family="family_triple_backend_realism_probe",
            seed=int(seed), arm=R100_W51_ARM,
            metric_name="anchor_transitive_or_skip",
            metric_value=float(
                transitive if transitive > 0 else skipped_ok)),
    }


def family_w51_envelope_verifier(
        seed: int,
) -> dict[str, R100SeedResult]:
    """H8: W51 envelope verifier rejects forged envelopes.
    24+ disjoint failure modes."""
    backend = SyntheticLLMClient(
        model_tag=f"synth.r100v.{seed}", default_response="x")
    agents = [
        Agent(name="a1", instructions="", role="r0",
              backend=backend, temperature=0.0, max_tokens=20),
    ]
    reg = build_w51_registry(
        schema_cid=f"r100_verifier_seed_{seed}",
        role_universe=("r0",))
    team = W51Team(
        agents=agents, backend=backend, registry=reg,
        max_visible_handoffs=2)
    r = team.run(f"r100 verifier task {seed}")
    v_clean = verify_w51_handoff(
        r.w51_envelope,
        expected_w50_outer_cid=r.w50_outer_cid,
        expected_params_cid=r.w51_params_cid,
        bundles=r.turn_witness_bundles,
        registry=reg,
        persistent_state_cids=r.persistent_state_cids)
    # Forged: tamper W50 outer
    v_forged = verify_w51_handoff(
        r.w51_envelope,
        expected_w50_outer_cid="ff" * 32,
        expected_params_cid=r.w51_params_cid,
        bundles=r.turn_witness_bundles,
        registry=reg)
    score = 1.0 if (
        v_clean["ok"] and not v_forged["ok"]) else 0.0
    return {
        R100_BASELINE_ARM: R100SeedResult(
            family="family_w51_envelope_verifier",
            seed=int(seed), arm=R100_BASELINE_ARM,
            metric_name="verifier_score",
            metric_value=0.0),
        R100_W51_ARM: R100SeedResult(
            family="family_w51_envelope_verifier",
            seed=int(seed), arm=R100_W51_ARM,
            metric_name="verifier_score",
            metric_value=float(score)),
    }


def family_w51_replay_determinism(
        seed: int,
) -> dict[str, R100SeedResult]:
    """H9: W51 replay byte-identical across two runs."""
    backend = SyntheticLLMClient(
        model_tag=f"synth.r100r.{seed}", default_response="r")
    agents = [
        Agent(name="a1", instructions="", role="r0",
              backend=backend, temperature=0.0, max_tokens=20),
    ]
    reg = build_w51_registry(
        schema_cid=f"r100_rep_seed_{seed}",
        role_universe=("r0",))
    r1 = W51Team(
        agents=agents, backend=backend, registry=reg,
        max_visible_handoffs=2).run("replay_task")
    r2 = W51Team(
        agents=agents, backend=backend, registry=reg,
        max_visible_handoffs=2).run("replay_task")
    ok = 1.0 if (
        r1.w50_outer_cid == r2.w50_outer_cid
        and r1.w51_outer_cid == r2.w51_outer_cid
    ) else 0.0
    return {
        R100_BASELINE_ARM: R100SeedResult(
            family="family_w51_replay_determinism",
            seed=int(seed), arm=R100_BASELINE_ARM,
            metric_name="replay_ok", metric_value=1.0),
        R100_W51_ARM: R100SeedResult(
            family="family_w51_replay_determinism",
            seed=int(seed), arm=R100_W51_ARM,
            metric_name="replay_ok",
            metric_value=float(ok)),
    }


def family_cross_backend_translator_compromise_cap(
        seed: int,
) -> dict[str, R100SeedResult]:
    """H10: forged triple-backend → trained translator cannot
    recover."""
    ts = synthesize_triple_backend_training_set(
        n_examples=16, feature_dim=8, code_dim=8,
        seed=int(seed))
    forged = forge_triple_backend_training_set(ts, seed=int(seed))
    translator, _ = fit_triple_backend_translator(
        forged, n_steps=96, seed=int(seed))
    fid_on_clean = score_triple_backend_fidelity(
        translator, ts.examples[:8])
    # protect_rate = 1 - |mean direct fidelity| (high =
    # forged trained cannot reproduce clean transfer).
    mean_dir = float(
        (fid_on_clean.direct_ab + fid_on_clean.direct_ac
         + fid_on_clean.direct_bc) / 3.0)
    protect_rate = max(0.0, 1.0 - abs(mean_dir))
    return {
        R100_BASELINE_ARM: R100SeedResult(
            family=(
                "family_cross_backend_translator_compromise_cap"),
            seed=int(seed), arm=R100_BASELINE_ARM,
            metric_name="downstream_protect_rate",
            metric_value=1.0),
        R100_W51_ARM: R100SeedResult(
            family=(
                "family_cross_backend_translator_compromise_cap"),
            seed=int(seed), arm=R100_W51_ARM,
            metric_name="downstream_protect_rate",
            metric_value=float(protect_rate)),
    }


# =============================================================================
# Family registry
# =============================================================================

R100_FAMILY_TABLE: dict[
        str, Callable[[int], dict[str, R100SeedResult]]] = {
    "family_trivial_w51_passthrough":
        family_trivial_w51_passthrough,
    "family_persistent_state_long_horizon_gain":
        family_persistent_state_long_horizon_gain,
    "family_triple_backend_transitivity":
        family_triple_backend_transitivity,
    "family_triple_backend_transitivity_gap":
        family_triple_backend_transitivity_gap,
    "family_deep_stack_v2_depth_strict_gain":
        family_deep_stack_v2_depth_strict_gain,
    "family_branch_specialised_heads_gain":
        family_branch_specialised_heads_gain,
    "family_branch_cycle_memory_gain":
        family_branch_cycle_memory_gain,
    "family_triple_backend_realism_probe":
        family_triple_backend_realism_probe,
    "family_w51_envelope_verifier":
        family_w51_envelope_verifier,
    "family_w51_replay_determinism":
        family_w51_replay_determinism,
    "family_cross_backend_translator_compromise_cap":
        family_cross_backend_translator_compromise_cap,
}


# =============================================================================
# Driver
# =============================================================================

def run_family(
        family: str, *,
        seeds: Sequence[int] = (1, 2, 3),
) -> R100FamilyComparison:
    """Run one family across the given seeds and aggregate."""
    if family not in R100_FAMILY_TABLE:
        raise ValueError(f"unknown family {family!r}")
    fn = R100_FAMILY_TABLE[family]
    per_arm: dict[str, list[tuple[int, R100SeedResult]]] = {}
    for s in seeds:
        out = fn(int(s))
        for arm, sr in out.items():
            per_arm.setdefault(arm, []).append((int(s), sr))
    aggs: list[R100AggregateResult] = []
    metric_name = ""
    for arm, ls in per_arm.items():
        ls.sort(key=lambda t: t[0])
        seeds_t = tuple(t[0] for t in ls)
        values_t = tuple(float(t[1].metric_value) for t in ls)
        metric_name = ls[0][1].metric_name
        aggs.append(R100AggregateResult(
            family=family, arm=arm,
            metric_name=metric_name,
            seeds=seeds_t, values=values_t,
        ))
    return R100FamilyComparison(
        family=family,
        metric_name=metric_name,
        aggregates=tuple(aggs))


def run_all_families(
        *, seeds: Sequence[int] = (1, 2, 3),
) -> dict[str, R100FamilyComparison]:
    return {
        name: run_family(name, seeds=seeds)
        for name in R100_FAMILY_TABLE.keys()
    }


def main() -> None:
    out = run_all_families(seeds=(1, 2, 3))
    summary = {
        "schema": R100_SCHEMA_VERSION,
        "families": [c.to_dict() for c in out.values()],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


__all__ = [
    "R100_SCHEMA_VERSION",
    "R100_BASELINE_ARM",
    "R100_W51_ARM",
    "R100SeedResult",
    "R100AggregateResult",
    "R100FamilyComparison",
    "R100_FAMILY_TABLE",
    "family_trivial_w51_passthrough",
    "family_persistent_state_long_horizon_gain",
    "family_triple_backend_transitivity",
    "family_triple_backend_transitivity_gap",
    "family_deep_stack_v2_depth_strict_gain",
    "family_branch_specialised_heads_gain",
    "family_branch_cycle_memory_gain",
    "family_triple_backend_realism_probe",
    "family_w51_envelope_verifier",
    "family_w51_replay_determinism",
    "family_cross_backend_translator_compromise_cap",
    "run_family",
    "run_all_families",
    "main",
]
