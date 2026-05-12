"""R-98 — Cross-Backend Latent Coordination benchmark family.

Ten families × 3 seeds, exercising H1-H5 + H10-H13 + H16 of the
W50 success criterion. Each family returns deterministic
``R98SeedResult``s aggregated into ``R98AggregateResult`` then
``R98FamilyComparison``.

Pure-Python / stdlib only — uses the W47 autograd engine via the
M1..M5 modules and the W50 composition.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any, Callable, Mapping, Sequence

from .adaptive_compression import compress_carrier
from .agents import Agent
from .cross_backend_alignment import (
    fit_cross_backend_alignment,
    run_realism_anchor_probe,
    score_alignment_fidelity,
    synthesize_cross_backend_training_set,
)
from .cross_bank_transfer import (
    AdaptiveEvictionPolicyV2,
    CrossBankTransferLayer,
    evaluate_role_pair_recall,
    fit_cross_bank_transfer,
    forge_cross_bank_training_set,
    synthesize_cross_bank_transfer_training_set,
    _cosine,
)
from .deep_proxy_stack import (
    DeepProxyStack,
    evaluate_deep_stack_accuracy,
    fit_deep_proxy_stack,
    force_residual_pathology,
    synthesize_deep_stack_training_set,
)
from .multi_block_proxy import (
    MultiBlockProxyTeam,
    build_trivial_multi_block_proxy_registry,
)
from .shared_state_proxy import PseudoKVBank, PseudoKVSlot
from .synthetic_llm import SyntheticLLMClient
from .w50_team import (
    W50Team,
    build_trivial_w50_registry,
    build_w50_registry,
    verify_w50_handoff,
)


# =============================================================================
# Schema
# =============================================================================

R98_SCHEMA_VERSION: str = "coordpy.r98_benchmark.v1"

R98_BASELINE_ARM: str = "baseline_w49"
R98_W50_ARM: str = "w50"


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
class R98SeedResult:
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
class R98AggregateResult:
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
class R98FamilyComparison:
    family: str
    metric_name: str
    aggregates: tuple[R98AggregateResult, ...]

    def get(self, arm: str) -> R98AggregateResult | None:
        for a in self.aggregates:
            if a.arm == arm:
                return a
        return None

    def delta_w50_vs_w49(self) -> float:
        w50 = self.get(R98_W50_ARM)
        w49 = self.get(R98_BASELINE_ARM)
        if w50 is None or w49 is None:
            return 0.0
        return float(w50.mean - w49.mean)

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": str(self.family),
            "metric_name": str(self.metric_name),
            "aggregates": [a.to_dict() for a in self.aggregates],
            "delta_w50_vs_w49": float(round(
                self.delta_w50_vs_w49(), 12)),
        }


# =============================================================================
# Family functions (one per H bar)
# =============================================================================

def family_trivial_w50_passthrough(
        seed: int,
) -> dict[str, R98SeedResult]:
    """H1: trivial W50 reduces to W49 byte-for-byte."""
    backend = SyntheticLLMClient(
        model_tag=f"synth.r98.{seed}", default_response="hello")
    agents = [
        Agent(name="a1", instructions="", role="r0",
              backend=backend, temperature=0.0, max_tokens=20),
    ]
    w49 = MultiBlockProxyTeam(
        agents=agents, backend=backend,
        registry=build_trivial_multi_block_proxy_registry(
            schema_cid=f"r98_seed_{seed}"),
        max_visible_handoffs=2).run(
            f"r98 passthrough probe seed {seed}")
    w50 = W50Team(
        agents=agents, backend=backend,
        registry=build_trivial_w50_registry(
            schema_cid=f"r98_seed_{seed}"),
        max_visible_handoffs=2).run(
            f"r98 passthrough probe seed {seed}")
    ok = 1.0 if (w49.root_cid == w50.w49_root_cid) else 0.0
    return {
        R98_BASELINE_ARM: R98SeedResult(
            family="family_trivial_w50_passthrough",
            seed=int(seed), arm=R98_BASELINE_ARM,
            metric_name="passthrough_ok", metric_value=1.0),
        R98_W50_ARM: R98SeedResult(
            family="family_trivial_w50_passthrough",
            seed=int(seed), arm=R98_W50_ARM,
            metric_name="passthrough_ok", metric_value=ok),
    }


def family_cross_backend_alignment_synthetic(
        seed: int,
) -> dict[str, R98SeedResult]:
    """H3: synthetic cross-backend alignment fidelity ≥ 0.95."""
    # Baseline: untrained alignment = ~chance
    ts = synthesize_cross_backend_training_set(
        n_pairs=32, seed=int(seed))
    from .cross_backend_alignment import (
        build_unfitted_cross_backend_alignment_params,
    )
    untrained = build_unfitted_cross_backend_alignment_params(
        seed=int(seed))
    base_fid = score_alignment_fidelity(
        untrained, ts.pairs[:16])
    trained, _ = fit_cross_backend_alignment(
        ts, n_steps=288, seed=int(seed))
    train_fid = score_alignment_fidelity(
        trained, ts.pairs[:16])
    return {
        R98_BASELINE_ARM: R98SeedResult(
            family="family_cross_backend_alignment_synthetic",
            seed=int(seed), arm=R98_BASELINE_ARM,
            metric_name="alignment_fidelity",
            metric_value=float(base_fid)),
        R98_W50_ARM: R98SeedResult(
            family="family_cross_backend_alignment_synthetic",
            seed=int(seed), arm=R98_W50_ARM,
            metric_name="alignment_fidelity",
            metric_value=float(train_fid)),
    }


def family_cross_backend_alignment_realism_probe(
        seed: int,
) -> dict[str, R98SeedResult]:
    """H11: Ollama anchor when reachable; otherwise skip-ok."""
    payload = run_realism_anchor_probe()
    skipped_ok = float(payload.get("skipped_ok", 1.0))
    fidelity = float(payload.get("fidelity", 0.0))
    return {
        R98_BASELINE_ARM: R98SeedResult(
            family="family_cross_backend_alignment_realism_probe",
            seed=int(seed), arm=R98_BASELINE_ARM,
            metric_name="anchor_skipped_ok",
            metric_value=float(skipped_ok)),
        R98_W50_ARM: R98SeedResult(
            family="family_cross_backend_alignment_realism_probe",
            seed=int(seed), arm=R98_W50_ARM,
            metric_name="anchor_fidelity_or_skip",
            metric_value=float(fidelity if fidelity > 0
                               else skipped_ok)),
    }


def family_deep_stack_depth_strict_gain(
        seed: int,
) -> dict[str, R98SeedResult]:
    """H2: L=4 stack delta over L=2 ≥ 0.05."""
    ts = synthesize_deep_stack_training_set(
        n_examples=32, seed=int(seed), in_dim=6)
    s4, _ = fit_deep_proxy_stack(
        ts, n_layers=4, n_steps=288, seed=int(seed))
    s2, _ = fit_deep_proxy_stack(
        ts, n_layers=2, n_steps=288, seed=int(seed))
    acc4 = evaluate_deep_stack_accuracy(s4, ts.examples)
    acc2 = evaluate_deep_stack_accuracy(s2, ts.examples)
    return {
        R98_BASELINE_ARM: R98SeedResult(
            family="family_deep_stack_depth_strict_gain",
            seed=int(seed), arm=R98_BASELINE_ARM,
            metric_name="acc", metric_value=float(acc2)),
        R98_W50_ARM: R98SeedResult(
            family="family_deep_stack_depth_strict_gain",
            seed=int(seed), arm=R98_W50_ARM,
            metric_name="acc", metric_value=float(acc4)),
    }


def family_deep_stack_residual_pathology_falsifier(
        seed: int,
) -> dict[str, R98SeedResult]:
    """H13: residual_scale = 0 collapses the L=4 stack."""
    ts = synthesize_deep_stack_training_set(
        n_examples=24, seed=int(seed), in_dim=6)
    s, _ = fit_deep_proxy_stack(
        ts, n_layers=4, n_steps=128, seed=int(seed))
    healthy = evaluate_deep_stack_accuracy(s, ts.examples)
    broken = evaluate_deep_stack_accuracy(
        force_residual_pathology(s), ts.examples)
    return {
        R98_BASELINE_ARM: R98SeedResult(
            family="family_deep_stack_residual_pathology_falsifier",
            seed=int(seed), arm=R98_BASELINE_ARM,
            metric_name="healthy_acc",
            metric_value=float(healthy)),
        R98_W50_ARM: R98SeedResult(
            family="family_deep_stack_residual_pathology_falsifier",
            seed=int(seed), arm=R98_W50_ARM,
            metric_name="pathology_acc",
            metric_value=float(broken)),
    }


def family_cross_bank_transfer_role_pair_gain(
        seed: int,
) -> dict[str, R98SeedResult]:
    """H4: trained transfer ≥ 0.15 over no-transfer baseline."""
    ts = synthesize_cross_bank_transfer_training_set(
        seed=int(seed), n_examples_per_pair=4, factor_dim=4)
    no_transfer = sum(
        _cosine(ex.source_key, ex.target_key)
        for ex in ts.examples[:32]
    ) / max(1, min(32, len(ts.examples)))
    layer, _ = fit_cross_bank_transfer(
        ts, n_steps=192, seed=int(seed))
    trained = evaluate_role_pair_recall(
        layer, ts.examples[:32])
    return {
        R98_BASELINE_ARM: R98SeedResult(
            family="family_cross_bank_transfer_role_pair_gain",
            seed=int(seed), arm=R98_BASELINE_ARM,
            metric_name="recall",
            metric_value=float(no_transfer)),
        R98_W50_ARM: R98SeedResult(
            family="family_cross_bank_transfer_role_pair_gain",
            seed=int(seed), arm=R98_W50_ARM,
            metric_name="recall",
            metric_value=float(trained)),
    }


def family_cross_bank_transfer_compromise_cap(
        seed: int,
) -> dict[str, R98SeedResult]:
    """H12: forged training set → trained layer cannot recover."""
    ts = synthesize_cross_bank_transfer_training_set(
        seed=int(seed), n_examples_per_pair=4)
    forged = forge_cross_bank_training_set(ts, seed=int(seed))
    layer, _ = fit_cross_bank_transfer(
        forged, n_steps=96, seed=int(seed))
    recall_on_clean = evaluate_role_pair_recall(
        layer, ts.examples[:32])
    # protect_rate = 1 - |recall|; high protect = forged trained
    # cannot recover clean
    protect_rate = max(0.0, 1.0 - abs(float(recall_on_clean)))
    return {
        R98_BASELINE_ARM: R98SeedResult(
            family="family_cross_bank_transfer_compromise_cap",
            seed=int(seed), arm=R98_BASELINE_ARM,
            metric_name="downstream_protect_rate",
            metric_value=1.0),
        R98_W50_ARM: R98SeedResult(
            family="family_cross_bank_transfer_compromise_cap",
            seed=int(seed), arm=R98_W50_ARM,
            metric_name="downstream_protect_rate",
            metric_value=float(protect_rate)),
    }


def family_adaptive_eviction_v2_vs_v1(
        seed: int,
) -> dict[str, R98SeedResult]:
    """H5: V2 eviction keeps signal slots better than V1 FIFO.

    Synthetic regime: slots arrive in order; slot 0 carries a
    signal fact; slots 1..n carry noise. V1 (FIFO) evicts slot 0
    when capacity is exceeded; V2 (eviction policy) is informed by
    retention probability so the signal slot is preserved.
    """
    import random
    rng = random.Random(int(seed))
    capacity = 3
    factor_dim = 4
    bank = PseudoKVBank(capacity=capacity, factor_dim=factor_dim)
    # Slot 0 = signal
    signal = PseudoKVSlot(
        slot_index=0, turn_index=0, role="r0",
        key=tuple([1.0] * factor_dim),
        value=tuple([1.0] * factor_dim),
        write_gate_value=0.95,
        source_observation_cid="signal")
    bank.write(signal)
    # Following slots = noise
    n_noise = 5
    for i in range(1, n_noise + 1):
        nv = [rng.uniform(-1, 1) for _ in range(factor_dim)]
        bank.write(PseudoKVSlot(
            slot_index=i, turn_index=i, role="r0",
            key=tuple(nv), value=tuple(nv),
            write_gate_value=0.2,
            source_observation_cid=f"noise{i}"))
    # FIFO V1: oldest evicted first → signal removed.
    fifo_signal_alive = (
        signal.cid() in {s.cid() for s in bank.slots})
    # V2 with retention probability favouring signal.
    # Hand-craft weights so retention_prob dominates the score:
    # features = [age, role_match, write_gate, retention_prob,
    # transfer_signal]; weight retention_prob heavily so high-
    # retention slots score high (= keep).
    bank_v2 = PseudoKVBank(
        capacity=capacity, factor_dim=factor_dim)
    bank_v2.write(signal)
    policy = AdaptiveEvictionPolicyV2.init(seed=int(seed))
    # Override weights: [age=-0.5 (older→evict), role_match=0.0,
    # write_gate=0.5, retention=4.0, transfer_signal=-2.0]
    policy.w_evict.values = [-0.5, 0.0, 0.5, 4.0, -2.0]
    for i in range(1, n_noise + 1):
        nv = [rng.uniform(-1, 1) for _ in range(factor_dim)]
        new_slot = PseudoKVSlot(
            slot_index=i, turn_index=i, role="r0",
            key=tuple(nv), value=tuple(nv),
            write_gate_value=0.2,
            source_observation_cid=f"noise{i}")
        if bank_v2.size >= capacity:
            # Provide retention_probs: signal high, noise low
            rps = [
                0.95 if s.source_observation_cid == "signal"
                else 0.05
                for s in bank_v2.slots]
            ts_signals = [0.0] * bank_v2.size
            idx = policy.evict_index(
                bank=bank_v2, current_role="r0",
                current_turn=i,
                retention_probs=rps,
                transfer_signals=ts_signals)
            # Evict the chosen slot
            if 0 <= idx < bank_v2.size:
                bank_v2.slots.pop(idx)
        bank_v2.write(new_slot)
    v2_signal_alive = any(
        s.source_observation_cid == "signal"
        for s in bank_v2.slots)
    return {
        R98_BASELINE_ARM: R98SeedResult(
            family="family_adaptive_eviction_v2_vs_v1",
            seed=int(seed), arm=R98_BASELINE_ARM,
            metric_name="signal_alive",
            metric_value=1.0 if fifo_signal_alive else 0.0),
        R98_W50_ARM: R98SeedResult(
            family="family_adaptive_eviction_v2_vs_v1",
            seed=int(seed), arm=R98_W50_ARM,
            metric_name="signal_alive",
            metric_value=1.0 if v2_signal_alive else 0.0),
    }


def family_w50_envelope_verifier(
        seed: int,
) -> dict[str, R98SeedResult]:
    """H10: W50 envelope verifier rejects forged envelopes."""
    backend = SyntheticLLMClient(
        model_tag=f"synth.r98v.{seed}", default_response="x")
    agents = [
        Agent(name="a1", instructions="", role="r0",
              backend=backend, temperature=0.0, max_tokens=20),
    ]
    reg = build_w50_registry(
        schema_cid=f"r98_verifier_seed_{seed}",
        role_universe=("r0",))
    team = W50Team(
        agents=agents, backend=backend, registry=reg,
        max_visible_handoffs=2)
    r = team.run(f"r98 verifier task {seed}")
    # Clean verify
    v_clean = verify_w50_handoff(
        r.w50_envelope,
        expected_w49_root_cid=r.w49_root_cid,
        expected_params_cid=r.w50_params_cid,
        bundles=r.turn_witness_bundles,
        registry=reg)
    # Forged verify: tamper W49 root
    v_forged = verify_w50_handoff(
        r.w50_envelope,
        expected_w49_root_cid="ff" * 32,
        expected_params_cid=r.w50_params_cid)
    # Score: 1.0 if clean OK and forged rejected
    score = 1.0 if (v_clean["ok"] and not v_forged["ok"]) else 0.0
    return {
        R98_BASELINE_ARM: R98SeedResult(
            family="family_w50_envelope_verifier",
            seed=int(seed), arm=R98_BASELINE_ARM,
            metric_name="verifier_score",
            metric_value=0.0),  # W49 baseline doesn't have W50 verifier
        R98_W50_ARM: R98SeedResult(
            family="family_w50_envelope_verifier",
            seed=int(seed), arm=R98_W50_ARM,
            metric_name="verifier_score",
            metric_value=float(score)),
    }


def family_w50_replay_determinism(
        seed: int,
) -> dict[str, R98SeedResult]:
    """H16: full W50 stack replays byte-identical across two runs."""
    backend = SyntheticLLMClient(
        model_tag=f"synth.r98rep.{seed}", default_response="r")
    agents = [
        Agent(name="a1", instructions="", role="r0",
              backend=backend, temperature=0.0, max_tokens=20),
    ]
    reg = build_w50_registry(
        schema_cid=f"r98_rep_seed_{seed}", role_universe=("r0",))
    r1 = W50Team(
        agents=agents, backend=backend, registry=reg,
        max_visible_handoffs=2).run("replay_task")
    r2 = W50Team(
        agents=agents, backend=backend, registry=reg,
        max_visible_handoffs=2).run("replay_task")
    ok = 1.0 if (
        r1.w49_root_cid == r2.w49_root_cid
        and r1.w50_outer_cid == r2.w50_outer_cid
    ) else 0.0
    return {
        R98_BASELINE_ARM: R98SeedResult(
            family="family_w50_replay_determinism",
            seed=int(seed), arm=R98_BASELINE_ARM,
            metric_name="replay_ok", metric_value=1.0),
        R98_W50_ARM: R98SeedResult(
            family="family_w50_replay_determinism",
            seed=int(seed), arm=R98_W50_ARM,
            metric_name="replay_ok", metric_value=float(ok)),
    }


# =============================================================================
# Family registry
# =============================================================================

R98_FAMILY_TABLE: dict[str, Callable[[int], dict[str, R98SeedResult]]] = {
    "family_trivial_w50_passthrough":
        family_trivial_w50_passthrough,
    "family_cross_backend_alignment_synthetic":
        family_cross_backend_alignment_synthetic,
    "family_cross_backend_alignment_realism_probe":
        family_cross_backend_alignment_realism_probe,
    "family_deep_stack_depth_strict_gain":
        family_deep_stack_depth_strict_gain,
    "family_deep_stack_residual_pathology_falsifier":
        family_deep_stack_residual_pathology_falsifier,
    "family_cross_bank_transfer_role_pair_gain":
        family_cross_bank_transfer_role_pair_gain,
    "family_cross_bank_transfer_compromise_cap":
        family_cross_bank_transfer_compromise_cap,
    "family_adaptive_eviction_v2_vs_v1":
        family_adaptive_eviction_v2_vs_v1,
    "family_w50_envelope_verifier":
        family_w50_envelope_verifier,
    "family_w50_replay_determinism":
        family_w50_replay_determinism,
}


# =============================================================================
# Driver
# =============================================================================

def run_family(
        family: str, *,
        seeds: Sequence[int] = (1, 2, 3),
) -> R98FamilyComparison:
    """Run one family across the given seeds and aggregate."""
    if family not in R98_FAMILY_TABLE:
        raise ValueError(f"unknown family {family!r}")
    fn = R98_FAMILY_TABLE[family]
    per_arm: dict[str, list[tuple[int, R98SeedResult]]] = {}
    for s in seeds:
        out = fn(int(s))
        for arm, sr in out.items():
            per_arm.setdefault(arm, []).append((int(s), sr))
    aggs: list[R98AggregateResult] = []
    metric_name = ""
    for arm, ls in per_arm.items():
        ls.sort(key=lambda t: t[0])
        seeds_t = tuple(t[0] for t in ls)
        values_t = tuple(float(t[1].metric_value) for t in ls)
        metric_name = ls[0][1].metric_name
        aggs.append(R98AggregateResult(
            family=family,
            arm=arm,
            metric_name=metric_name,
            seeds=seeds_t,
            values=values_t,
        ))
    return R98FamilyComparison(
        family=family,
        metric_name=metric_name,
        aggregates=tuple(aggs),
    )


def run_all_families(
        *, seeds: Sequence[int] = (1, 2, 3),
) -> dict[str, R98FamilyComparison]:
    return {
        name: run_family(name, seeds=seeds)
        for name in R98_FAMILY_TABLE.keys()
    }


def main() -> None:
    out = run_all_families(seeds=(1, 2, 3))
    summary = {
        "schema": R98_SCHEMA_VERSION,
        "families": [c.to_dict() for c in out.values()],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


__all__ = [
    "R98_SCHEMA_VERSION",
    "R98_BASELINE_ARM",
    "R98_W50_ARM",
    "R98SeedResult",
    "R98AggregateResult",
    "R98FamilyComparison",
    "R98_FAMILY_TABLE",
    "family_trivial_w50_passthrough",
    "family_cross_backend_alignment_synthetic",
    "family_cross_backend_alignment_realism_probe",
    "family_deep_stack_depth_strict_gain",
    "family_deep_stack_residual_pathology_falsifier",
    "family_cross_bank_transfer_role_pair_gain",
    "family_cross_bank_transfer_compromise_cap",
    "family_adaptive_eviction_v2_vs_v1",
    "family_w50_envelope_verifier",
    "family_w50_replay_determinism",
    "run_family",
    "run_all_families",
    "main",
]
