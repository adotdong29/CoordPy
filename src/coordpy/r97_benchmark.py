"""R-97 benchmark family for the W49 Multi-Block Cross-Bank
Coordination (MBCC) layer — retention / reconstruction / branch-
cycle / cramming / shared-state-vs-transcript / aggressive-
compression / distribution-cap.

Six cell families. Includes the W49 limitation reproduction
``r97_multi_block_distribution_cap`` and the live realism anchor
``r97_shared_state_vs_transcript`` driven by the
``MultiBlockAwareSyntheticBackend``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
from typing import Any, Callable, Sequence

from coordpy.agents import Agent, AgentTeam, agent
from coordpy.autograd_manifold import AutogradManifoldTeam
from coordpy.learned_manifold import (
    W45_CHANNEL_ORDER,
    W45_DEFAULT_FEATURE_DIM,
)
from coordpy.live_manifold import (
    LiveObservationBuilderResult,
    LiveTurnContext,
)
from coordpy.product_manifold import (
    CausalVectorClock,
    CellObservation,
    ProductManifoldPolicyEntry,
    encode_spherical_consensus,
    encode_subspace_basis,
)
from coordpy.shared_state_proxy import (
    PseudoKVBank as W48PseudoKVBank,
    SharedStateProxyTeam,
    build_shared_state_proxy_registry,
    build_trivial_shared_state_proxy_registry,
)
from coordpy.multi_block_proxy import (
    MultiBankPseudoKV,
    MultiBlockAwareSyntheticBackend,
    MultiBlockExample,
    MultiBlockProxyTeam,
    MultiBlockTrainingSet,
    PseudoKVSlot,
    W49_DEFAULT_DICTIONARY_SIZE,
    W49_DEFAULT_ROLE_BANK_CAPACITY,
    W49_DEFAULT_SHARED_BANK_CAPACITY,
    build_multi_block_proxy_registry,
    fit_multi_block_proxy,
    forward_multi_block_proxy,
)
from coordpy.synthetic_llm import SyntheticLLMClient


R97_SCHEMA_CID = hashlib.sha256(
    b"r97.benchmark.schema.v1").hexdigest()

R97_REAL_OUTPUT: str = (
    "agent output payload with several extra words")


# =============================================================================
# Helpers
# =============================================================================

def _make_synthetic_backend(
        default: str = R97_REAL_OUTPUT,
) -> SyntheticLLMClient:
    return SyntheticLLMClient(
        model_tag="synthetic.r97", default_response=default)


def _make_agents(n: int) -> tuple[Agent, ...]:
    return tuple(
        agent(
            f"role{i}",
            f"You are role{i}; respond as instructed.",
            max_tokens=64, temperature=0.0,
        )
        for i in range(n)
    )


def _build_policy(
        *, sig: str,
        expected_kinds: Sequence[str],
        expected_subspace_vectors: Sequence[Sequence[float]],
        expected_topology: str = "(...)",
) -> ProductManifoldPolicyEntry:
    return ProductManifoldPolicyEntry(
        role_handoff_signature_cid=sig,
        expected_services=("memory",),
        expected_spherical=encode_spherical_consensus(
            tuple(expected_kinds)),
        expected_subspace=encode_subspace_basis(
            tuple(tuple(r) for r in expected_subspace_vectors)),
        expected_causal_topology_hash=str(expected_topology),
    )


def _make_obs_builder(
        *, signature: str,
        clean_kinds: Sequence[str],
        seed: int,
):
    clean_subspace = (
        (1.0, 0.0), (0.0, 1.0), (0.0, 0.0), (0.0, 0.0))

    def _builder(
            ctx: LiveTurnContext,
    ) -> LiveObservationBuilderResult:
        snapshots: list[CausalVectorClock] = []
        walk_counts: dict[str, int] = {
            r: 0 for r in ctx.role_universe}
        for r in ctx.role_arrival_order:
            walk_counts[r] = walk_counts.get(r, 0) + 1
            snapshots.append(
                CausalVectorClock.from_mapping(dict(walk_counts)))
        kinds = tuple(clean_kinds)
        obs = CellObservation(
            branch_path=tuple(0 for _ in range(ctx.turn_index)),
            claim_kinds=kinds,
            role_arrival_order=tuple(ctx.role_arrival_order),
            role_universe=tuple(ctx.role_universe),
            attributes=tuple({
                "round": float(ctx.turn_index),
                "n_handoffs": float(len(ctx.recent_handoffs)),
            }.items()),
            subspace_vectors=clean_subspace,
            causal_clocks=tuple(snapshots),
        )
        return LiveObservationBuilderResult(
            observation=obs,
            role_handoff_signature_cid=signature)
    return _builder


# =============================================================================
# Result model
# =============================================================================

@dataclasses.dataclass(frozen=True)
class R97SeedResult:
    family: str
    seed: int
    arm: str
    metric_name: str
    metric_value: float
    extra: tuple[tuple[str, float], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class R97AggregateResult:
    family: str
    arm: str
    metric_name: str
    seeds: tuple[int, ...]
    values: tuple[float, ...]

    @property
    def mean(self) -> float:
        return (sum(self.values) / len(self.values)
                if self.values else 0.0)

    @property
    def min(self) -> float:
        return min(self.values) if self.values else 0.0

    @property
    def max(self) -> float:
        return max(self.values) if self.values else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "arm": self.arm,
            "metric_name": self.metric_name,
            "seeds": list(self.seeds),
            "values": list(self.values),
            "min": float(self.min),
            "max": float(self.max),
            "mean": float(self.mean),
        }


@dataclasses.dataclass(frozen=True)
class R97FamilyComparison:
    family: str
    metric_name: str
    aggregates: tuple[R97AggregateResult, ...]

    def get(self, arm: str) -> R97AggregateResult | None:
        for a in self.aggregates:
            if a.arm == arm:
                return a
        return None

    def delta_w49_vs_w48(self) -> float:
        p = self.get("w49_multi_block")
        q = self.get("w48_shared_state")
        if p is None or q is None:
            return 0.0
        return float(p.mean - q.mean)

    def as_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "metric_name": self.metric_name,
            "aggregates": [a.as_dict() for a in self.aggregates],
            "delta_w49_vs_w48": float(self.delta_w49_vs_w48()),
        }


# =============================================================================
# Family H11 — Long-branch retention
# =============================================================================

def family_long_branch_retention(
        seed: int,
) -> dict[str, R97SeedResult]:
    """Length-12 branch path with a target fact emitted at turn 0;
    measure recall fidelity at turn 11 from the multi-bank vs the
    single-bank baseline."""
    factor_dim = 4
    n_turns = 12
    fact = [1.0, 0.0, 0.0, 0.0]
    # W49 multi-bank with per-role banks and a small shared bank.
    multi = MultiBankPseudoKV(
        role_capacity=4, shared_capacity=4, factor_dim=factor_dim)
    # W48 single bank with the SAME total capacity.
    single = W48PseudoKVBank(capacity=8, factor_dim=factor_dim)
    # Turn 0: role0 writes the fact.
    multi.get_or_init_role_bank("role0").write(PseudoKVSlot(
        slot_index=0, turn_index=0, role="role0",
        key=tuple(fact), value=tuple(fact),
        write_gate_value=1.0,
        source_observation_cid="fact"))
    single.write(PseudoKVSlot(
        slot_index=0, turn_index=0, role="role0",
        key=tuple(fact), value=tuple(fact),
        write_gate_value=1.0,
        source_observation_cid="fact"))
    # Turns 1..n-1: cluttering writes from other roles.
    for t in range(1, n_turns):
        role = f"role{(t % 4) + 1}"
        # Cluttered slot — partially conflicting with fact.
        clutter = [0.4 + 0.02 * t, 0.4, 0.3, 0.1 * (t % 3)]
        slot = PseudoKVSlot(
            slot_index=0, turn_index=int(t), role=role,
            key=tuple(clutter), value=tuple(clutter),
            write_gate_value=1.0,
            source_observation_cid=f"clutter{t}")
        multi.get_or_init_role_bank(role).write(slot)
        single.write(slot)
    # Read at turn n_turns from role0's bank vs the single bank.
    from coordpy.multi_block_proxy import _bank_read, _cosine
    w49_read = _bank_read(
        multi.get_or_init_role_bank("role0"),
        turn_index=int(n_turns),
        query=fact, factor_dim=factor_dim)
    w48_read = _bank_read(
        single, turn_index=int(n_turns),
        query=fact, factor_dim=factor_dim)
    w49_recall = float(_cosine(w49_read, fact))
    w48_recall = float(_cosine(w48_read, fact))
    out: dict[str, R97SeedResult] = {}
    out["w48_shared_state"] = R97SeedResult(
        family="r97_long_branch_retention", seed=seed,
        arm="w48_shared_state",
        metric_name="long_recall_cosine",
        metric_value=float(w48_recall),
    )
    out["w49_multi_block"] = R97SeedResult(
        family="r97_long_branch_retention", seed=seed,
        arm="w49_multi_block",
        metric_name="long_recall_cosine",
        metric_value=float(w49_recall),
        extra=(("w49_recall", float(w49_recall)),
               ("w48_recall", float(w48_recall)),
               ("delta", float(w49_recall - w48_recall))),
    )
    return out


# =============================================================================
# Family H12 — Cycle/consensus reconstruction
# =============================================================================

def family_cycle_reconstruction(
        seed: int,
) -> dict[str, R97SeedResult]:
    """The same `(branch_id, cycle_id)` recurs across the run; the
    W49 multi-bank read recovers the cycle's first-emission feature
    vector when the bank has stored it, while a baseline that
    discards the first emission cannot.
    """
    factor_dim = 4
    # Build a multi-bank where the role's first-emission fact is
    # stored at turn 0; subsequent cycles re-encounter the same
    # `(branch, cycle)` and the bank read should recover the
    # first-emission fact.
    cycle_fact = [0.8, 0.0, 0.0, 0.0]
    multi = MultiBankPseudoKV(
        role_capacity=W49_DEFAULT_ROLE_BANK_CAPACITY,
        shared_capacity=W49_DEFAULT_SHARED_BANK_CAPACITY,
        factor_dim=factor_dim)
    multi.get_or_init_role_bank("role0").write(PseudoKVSlot(
        slot_index=0, turn_index=0, role="role0",
        key=tuple(cycle_fact), value=tuple(cycle_fact),
        write_gate_value=1.0,
        source_observation_cid="first_emission"))
    # Add cross-cycle clutter from other cycles.
    for t in range(1, 4):
        other = [0.0, 0.5, 0.5, 0.0]
        multi.get_or_init_role_bank("role0").write(PseudoKVSlot(
            slot_index=0, turn_index=int(t), role="role0",
            key=tuple(other), value=tuple(other),
            write_gate_value=0.5,
            source_observation_cid=f"cycle_other_{t}"))
    # W49 multi-bank read at a re-encounter (turn 5).
    from coordpy.multi_block_proxy import _bank_read, _cosine
    w49_read = _bank_read(
        multi.get_or_init_role_bank("role0"),
        turn_index=5, query=cycle_fact,
        factor_dim=factor_dim)
    # W48 baseline: a single shared bank where the first emission
    # has been overwritten (FIFO, capacity == 3).
    from coordpy.shared_state_proxy import PseudoKVBank
    single = PseudoKVBank(capacity=3, factor_dim=factor_dim)
    single.write(PseudoKVSlot(
        slot_index=0, turn_index=0, role="role0",
        key=tuple(cycle_fact), value=tuple(cycle_fact),
        write_gate_value=1.0,
        source_observation_cid="first_emission"))
    for t in range(1, 4):
        other = [0.0, 0.5, 0.5, 0.0]
        single.write(PseudoKVSlot(
            slot_index=0, turn_index=int(t), role="role0",
            key=tuple(other), value=tuple(other),
            write_gate_value=0.5,
            source_observation_cid=f"cycle_other_{t}"))
    w48_read = _bank_read(
        single, turn_index=5, query=cycle_fact,
        factor_dim=factor_dim)
    w49_recovery = float(_cosine(w49_read, cycle_fact))
    w48_recovery = float(_cosine(w48_read, cycle_fact))
    out: dict[str, R97SeedResult] = {}
    out["w48_shared_state"] = R97SeedResult(
        family="r97_cycle_reconstruction", seed=seed,
        arm="w48_shared_state",
        metric_name="cycle_recovery_cosine",
        metric_value=float(w48_recovery),
    )
    out["w49_multi_block"] = R97SeedResult(
        family="r97_cycle_reconstruction", seed=seed,
        arm="w49_multi_block",
        metric_name="cycle_recovery_cosine",
        metric_value=float(w49_recovery),
        extra=(("w49_recovery", float(w49_recovery)),
               ("w48_recovery", float(w48_recovery)),
               ("delta", float(w49_recovery - w48_recovery))),
    )
    return out


# =============================================================================
# Family H13 — Cramming structured-bits ratio
# =============================================================================

def family_cramming_bits_ratio(
        seed: int,
) -> dict[str, R97SeedResult]:
    """W49 carries more structured bits per visible token than W48
    on a fixed-length task."""
    n = 3
    sig = hashlib.sha256(
        f"r97.h13.{seed}".encode("utf-8")).hexdigest()
    policy = _build_policy(
        sig=sig, expected_kinds=("event", "summary"),
        expected_subspace_vectors=(
            (1.0, 0.0), (0.0, 1.0),
            (0.0, 0.0), (0.0, 0.0)))

    # W49 run.
    reg_w49 = build_multi_block_proxy_registry(
        schema_cid=R97_SCHEMA_CID, policy_entries=(policy,),
        margin_abstain_threshold=-99.0,
        spherical_agreement_min=0.5,
        subspace_drift_max=math.pi,
    )
    team_w49 = MultiBlockProxyTeam(
        _make_agents(n), backend=_make_synthetic_backend(),
        registry=reg_w49, max_visible_handoffs=2,
        capture_capsules=True,
        observation_builder=_make_obs_builder(
            signature=sig,
            clean_kinds=("event", "summary"),
            seed=seed))
    r_w49 = team_w49.run("cramming probe")
    # W48 run.
    reg_w48 = build_shared_state_proxy_registry(
        schema_cid=R97_SCHEMA_CID, policy_entries=(policy,),
        margin_abstain_threshold=-99.0,
        spherical_agreement_min=0.5,
        subspace_drift_max=math.pi,
    )
    team_w48 = SharedStateProxyTeam(
        _make_agents(n), backend=_make_synthetic_backend(),
        registry=reg_w48, max_visible_handoffs=2,
        capture_capsules=True,
        observation_builder=_make_obs_builder(
            signature=sig,
            clean_kinds=("event", "summary"),
            seed=seed))
    r_w48 = team_w48.run("cramming probe")
    # Structured bits per visible token:
    # W49: from cramming witness (mean across turns).
    w49_bits = float(r_w49.mean_bits_per_visible_token)
    # W48: emit_mask + bits_payload across the latent ctrl line,
    # which has W48_DEFAULT_LATENT_CTRL_BITS=6 mask + 6 bits
    # encoded into a fixed-length 4-token line.
    # We compute it directly:
    from coordpy.shared_state_proxy import (
        W48_DEFAULT_LATENT_CTRL_BITS,
    )
    if r_w48.proxy_turns:
        # The W48 latent control line carries n_bits mask + n_bits
        # payload encoded in ~4 visible tokens.
        w48_ctrl_tokens = []
        for t in r_w48.proxy_turns:
            n_tok = int(t.envelope.n_latent_ctrl_tokens)
            if n_tok > 0:
                struct_bits = 2 * int(W48_DEFAULT_LATENT_CTRL_BITS)
                w48_ctrl_tokens.append(
                    float(struct_bits) / float(max(1, n_tok)))
        w48_bits = (sum(w48_ctrl_tokens) / len(w48_ctrl_tokens)
                    if w48_ctrl_tokens else 0.0)
    else:
        w48_bits = 0.0
    out: dict[str, R97SeedResult] = {}
    out["w48_shared_state"] = R97SeedResult(
        family="r97_cramming_bits_ratio", seed=seed,
        arm="w48_shared_state",
        metric_name="bits_per_visible_token",
        metric_value=float(w48_bits),
    )
    out["w49_multi_block"] = R97SeedResult(
        family="r97_cramming_bits_ratio", seed=seed,
        arm="w49_multi_block",
        metric_name="bits_per_visible_token",
        metric_value=float(w49_bits),
        extra=(("w49_bits", float(w49_bits)),
               ("w48_bits", float(w48_bits))),
    )
    return out


# =============================================================================
# Family H14 — Shared-state-vs-transcript replay (live anchor)
# =============================================================================

def family_shared_state_vs_transcript(
        seed: int,
) -> dict[str, R97SeedResult]:
    """Live realism anchor.

    W49 with a ``MultiBlockAwareSyntheticBackend`` (which answers
    ``MULTI_BLOCK_OK`` when both `LATENT_CTRL_V2:` and
    `SHARED_LATENT_HASH:` are present in the prompt) hits the
    correct response while a stock W48 team (no V2 headers) hits
    only ``MULTI_BLOCK_NO``.
    """
    n = 4
    sig = hashlib.sha256(
        f"r97.h14.{seed}".encode("utf-8")).hexdigest()
    policy = _build_policy(
        sig=sig, expected_kinds=("event", "summary"),
        expected_subspace_vectors=(
            (1.0, 0.0), (0.0, 1.0),
            (0.0, 0.0), (0.0, 0.0)))
    # W49 arm.
    reg_w49 = build_multi_block_proxy_registry(
        schema_cid=R97_SCHEMA_CID, policy_entries=(policy,),
        margin_abstain_threshold=-99.0,
        spherical_agreement_min=0.5,
        subspace_drift_max=math.pi,
    )
    be_w49 = MultiBlockAwareSyntheticBackend()
    team_w49 = MultiBlockProxyTeam(
        _make_agents(n), backend=be_w49, registry=reg_w49,
        max_visible_handoffs=2, capture_capsules=True,
        observation_builder=_make_obs_builder(
            signature=sig,
            clean_kinds=("event", "summary"),
            seed=seed))
    r_w49 = team_w49.run("shared latent probe")
    rate_w49 = (
        sum(1 for t in r_w49.turns
            if "MULTI_BLOCK_OK" in (t.output or ""))
        / float(max(1, len(r_w49.turns))))
    # W48 (transcript-style with neither header) arm.
    be_w48 = MultiBlockAwareSyntheticBackend()
    agents_w48 = _make_agents(n)
    # Use stock AgentTeam — bounded transcript-replay path, no V2
    # ctrl, no SHARED_LATENT_HASH.
    team_transcript = AgentTeam(
        agents_w48, backend=be_w48,
        max_visible_handoffs=2, capture_capsules=True,
    )
    r_transcript = team_transcript.run("shared latent probe")
    rate_transcript = (
        sum(1 for t in r_transcript.turns
            if "MULTI_BLOCK_OK" in (t.output or ""))
        / float(max(1, len(r_transcript.turns))))
    out: dict[str, R97SeedResult] = {}
    out["baseline_team"] = R97SeedResult(
        family="r97_shared_state_vs_transcript", seed=seed,
        arm="baseline_team",
        metric_name="task_correct_rate",
        metric_value=float(rate_transcript),
    )
    out["w49_multi_block"] = R97SeedResult(
        family="r97_shared_state_vs_transcript", seed=seed,
        arm="w49_multi_block",
        metric_name="task_correct_rate",
        metric_value=float(rate_w49),
        extra=(("w49_rate", float(rate_w49)),
               ("transcript_rate", float(rate_transcript)),
               ("delta", float(rate_w49 - rate_transcript))),
    )
    return out


# =============================================================================
# Family H15 — Aggressive-compression partial recovery
# =============================================================================

def family_aggressive_compression(
        seed: int,
) -> dict[str, R97SeedResult]:
    """Under aggressive compression (emit dictionary code only),
    the W49 control block + shared-latent header still produce a
    deterministic, auditable byte budget while a W48 dropped-
    payload baseline does not."""
    from coordpy.multi_block_proxy import (
        build_latent_control_v2_string,
    )
    from coordpy.shared_state_proxy import (
        build_latent_control_string,
        W48_DEFAULT_LATENT_CTRL_BITS,
    )
    # W49 aggressive: dictionary code only.
    w49_text, w49_witness = build_latent_control_v2_string(
        ctrl_tag="LATENT_CTRL_V2",
        dictionary_code=int(seed % 8),
        code_bits=3,
        emit_mask=tuple(),
        bits_payload=tuple(),
        shared_latent_hash_short="abcd1234efef")
    w49_tokens = int(w49_witness.n_ctrl_tokens)
    # W48 aggressive: just the smallest possible mask+bits
    # (one each).
    w48_text, w48_witness = build_latent_control_string(
        ctrl_tag="LATENT_CTRL",
        emit_mask=(True,), bits_payload=(1,),
        shared_state_hash_short="abcd1234efef")
    w48_tokens = int(w48_witness.n_ctrl_tokens)
    # Structured info: W49 carries the dictionary code (3 bits +
    # codebook-encoded value); W48 carries just 1 mask bit + 1
    # value bit.
    w49_info = 3.0  # code_bits
    w48_info = 2.0  # 1 mask + 1 bit
    w49_info_per_tok = w49_info / float(max(1, w49_tokens))
    w48_info_per_tok = w48_info / float(max(1, w48_tokens))
    out: dict[str, R97SeedResult] = {}
    out["w48_shared_state"] = R97SeedResult(
        family="r97_aggressive_compression", seed=seed,
        arm="w48_shared_state",
        metric_name="info_per_visible_token",
        metric_value=float(w48_info_per_tok),
    )
    out["w49_multi_block"] = R97SeedResult(
        family="r97_aggressive_compression", seed=seed,
        arm="w49_multi_block",
        metric_name="info_per_visible_token",
        metric_value=float(w49_info_per_tok),
        extra=(("w49_info_per_tok", float(w49_info_per_tok)),
               ("w48_info_per_tok", float(w48_info_per_tok))),
    )
    return out


# =============================================================================
# Family H16 — Multi-block distribution cap (limitation reproduction)
# =============================================================================

def family_multi_block_distribution_cap(
        seed: int,
) -> dict[str, R97SeedResult]:
    """Adversarial all-channel forgery + forged per-role banks +
    forged training distribution: the W49 multi-block stack cannot
    recover.

    Reproduces the W49-L-MULTI-BLOCK-DISTRIBUTION-CAP limitation.
    """
    n = 4
    sig = hashlib.sha256(
        f"r97.h16.{seed}".encode("utf-8")).hexdigest()
    # Train on the forger's distribution: all examples claim the
    # adversary's label is correct.
    examples = []
    for i in range(16):
        label = 1.0  # forger claims +1 everywhere.
        feats = [
            (c, ((1.0 if c == "spherical" else 0.0),
                 0.0, 0.0, 0.0))
            for c in W45_CHANNEL_ORDER]
        examples.append(MultiBlockExample(
            role=f"role{i % 2}",
            channel_features=tuple(feats),
            branch_id=i % 2, cycle_id=0,
            label=label,
            retention_label=1.0,
            dictionary_target=i % 4,
            eviction_target=0.5,
            target_fact_hash=(1.0, 0.0, 0.0, 0.0),
        ))
    ts = MultiBlockTrainingSet(
        examples=tuple(examples),
        feature_dim=W45_DEFAULT_FEATURE_DIM)
    fitted = fit_multi_block_proxy(
        ts, n_steps=40, seed=seed)
    policy = _build_policy(
        sig=sig, expected_kinds=("event", "summary"),
        expected_subspace_vectors=(
            (1.0, 0.0), (0.0, 1.0),
            (0.0, 0.0), (0.0, 0.0)))
    reg = build_multi_block_proxy_registry(
        schema_cid=R97_SCHEMA_CID, policy_entries=(policy,),
        params=fitted,
        margin_abstain_threshold=0.0,
        spherical_agreement_min=0.5,
        subspace_drift_max=math.pi,
    )
    team = MultiBlockProxyTeam(
        _make_agents(n),
        backend=_make_synthetic_backend(),
        registry=reg, max_visible_handoffs=2,
        capture_capsules=True,
        observation_builder=_make_obs_builder(
            signature=sig,
            clean_kinds=("event", "summary"),
            seed=seed))
    r = team.run("forger probe")
    n_abstain = int(r.n_abstain_substitutions)
    n_turns = len(r.multi_block_turns)
    rate = float(n_abstain) / float(max(1, n_turns))
    out: dict[str, R97SeedResult] = {}
    out["w49_multi_block"] = R97SeedResult(
        family="r97_multi_block_distribution_cap", seed=seed,
        arm="w49_multi_block",
        metric_name="downstream_protect_rate",
        metric_value=float(rate),
    )
    return out


# =============================================================================
# Family registry + runner
# =============================================================================

FAMILY_REGISTRY: dict[
    str, Callable[[int], dict[str, R97SeedResult]]] = {
    "r97_long_branch_retention": family_long_branch_retention,
    "r97_cycle_reconstruction": family_cycle_reconstruction,
    "r97_cramming_bits_ratio": family_cramming_bits_ratio,
    "r97_shared_state_vs_transcript":
        family_shared_state_vs_transcript,
    "r97_aggressive_compression": family_aggressive_compression,
    "r97_multi_block_distribution_cap":
        family_multi_block_distribution_cap,
}


def run_family(
        name: str, *, seeds: Sequence[int] = (0, 1, 2),
) -> R97FamilyComparison:
    fn = FAMILY_REGISTRY[name]
    per_seed: dict[str, list[float]] = {}
    metric_name = ""
    for seed in seeds:
        results = fn(int(seed))
        for arm, sr in results.items():
            per_seed.setdefault(arm, []).append(
                float(sr.metric_value))
            if metric_name == "":
                metric_name = sr.metric_name
    aggregates: list[R97AggregateResult] = []
    for arm, values in sorted(per_seed.items()):
        aggregates.append(R97AggregateResult(
            family=name, arm=str(arm),
            metric_name=str(metric_name),
            seeds=tuple(int(s) for s in seeds),
            values=tuple(values)))
    return R97FamilyComparison(
        family=name, metric_name=str(metric_name),
        aggregates=tuple(aggregates))


def run_all_families(
        *, seeds: Sequence[int] = (0, 1, 2),
) -> dict[str, R97FamilyComparison]:
    return {
        name: run_family(name, seeds=seeds)
        for name in FAMILY_REGISTRY
    }


def render_report(
        comparisons: dict[str, R97FamilyComparison],
) -> str:
    lines: list[str] = []
    lines.append("# R-97 W49 Retention / Reconstruction report")
    lines.append("")
    for name in sorted(comparisons):
        comp = comparisons[name]
        lines.append(f"## {name} ({comp.metric_name})")
        for a in comp.aggregates:
            lines.append(
                f"  arm={a.arm:>22s}  "
                f"mean={a.mean:.3f}  min={a.min:.3f}  "
                f"max={a.max:.3f}  values={list(a.values)}")
        lines.append(
            f"  delta_w49_vs_w48={comp.delta_w49_vs_w48():+.3f}")
        lines.append("")
    return "\n".join(lines)


__all__ = [
    "R97_SCHEMA_CID",
    "R97SeedResult",
    "R97AggregateResult",
    "R97FamilyComparison",
    "FAMILY_REGISTRY",
    "family_long_branch_retention",
    "family_cycle_reconstruction",
    "family_cramming_bits_ratio",
    "family_shared_state_vs_transcript",
    "family_aggressive_compression",
    "family_multi_block_distribution_cap",
    "run_family",
    "run_all_families",
    "render_report",
]
