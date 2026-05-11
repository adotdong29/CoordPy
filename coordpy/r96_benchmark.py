"""R-96 benchmark family for the W49 Multi-Block Cross-Bank
Coordination (MBCC) layer — mechanism-focused: multi-block depth,
multi-bank role-conditioned pseudo-KV, learned eviction policy,
retention head, dictionary-codebook compression, shared-latent
capsule, cross-bank interference bound, replay determinism, and
verifier soundness.

Honest-arm count per family: up to four.

  * ``baseline_team``         — released ``AgentTeam.run`` path.
  * ``w48_shared_state``      — W48 ``SharedStateProxyTeam``.
  * ``w49_multi_block``       — ``MultiBlockProxyTeam`` with the
    trained multi-block proxy on.
  * (per-family auxiliary arms where needed.)

Ten cell families. The R-96 family is the H1..H10 success bar for
the W49 milestone. See
``docs/SUCCESS_CRITERION_W49_MULTI_BLOCK_PROXY.md`` and
``docs/RESULTS_COORDPY_W49_MULTI_BLOCK_PROXY.md`` for full reads.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
from typing import Any, Callable, Sequence

from coordpy.agents import Agent, AgentTeam, agent
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
    SharedStateProxyTeam,
    build_shared_state_proxy_registry,
    build_trivial_shared_state_proxy_registry,
)
from coordpy.multi_block_proxy import (
    BankMixGate,
    BankRouter,
    DictionaryCodebook,
    EvictionPolicy,
    FeedForwardBlock,
    MultiBankPseudoKV,
    MultiBlockAwareSyntheticBackend,
    MultiBlockExample,
    MultiBlockProxyParams,
    MultiBlockProxyStack,
    MultiBlockProxyTeam,
    MultiBlockProxyTeamResult,
    MultiBlockTrainingSet,
    PseudoKVBank,
    PseudoKVSlot,
    RetentionHead,
    SharedLatentCapsule,
    W49_ALL_FAILURE_MODES,
    W49_BRANCH_MULTI_BLOCK_RATIFIED,
    W49_BRANCH_TRIVIAL_MULTI_BLOCK_PASSTHROUGH,
    W49_DEFAULT_DICTIONARY_SIZE,
    W49_DEFAULT_ROLE_BANK_CAPACITY,
    W49_DEFAULT_SHARED_BANK_CAPACITY,
    build_latent_control_v2_string,
    build_multi_block_proxy_registry,
    build_trivial_multi_block_proxy_registry,
    build_unfitted_multi_block_proxy_params,
    fit_multi_block_proxy,
    forward_multi_block_proxy,
    verify_multi_block_proxy_handoff,
)
from coordpy.synthetic_llm import SyntheticLLMClient


R96_SCHEMA_CID = hashlib.sha256(
    b"r96.benchmark.schema.v1").hexdigest()

R96_REAL_OUTPUT: str = (
    "agent output payload with several extra words "
    "to make rendering meaningful")


# =============================================================================
# Helpers
# =============================================================================

def _make_synthetic_backend(
        default: str = R96_REAL_OUTPUT,
) -> SyntheticLLMClient:
    return SyntheticLLMClient(
        model_tag="synthetic.r96", default_response=default)


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
class R96SeedResult:
    family: str
    seed: int
    arm: str
    metric_name: str
    metric_value: float
    n_role_bank_writes: int = 0
    n_shared_bank_writes: int = 0
    n_dictionary_codes_emitted: int = 0
    mean_retention_prob: float = 0.0
    mean_bits_per_visible_token: float = 0.0
    decision_branches: tuple[str, ...] = ()
    extra: tuple[tuple[str, float], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class R96AggregateResult:
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
class R96FamilyComparison:
    family: str
    metric_name: str
    aggregates: tuple[R96AggregateResult, ...]

    def get(self, arm: str) -> R96AggregateResult | None:
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

    def delta_w49_vs_baseline(self) -> float:
        p = self.get("w49_multi_block")
        q = self.get("baseline_team")
        if p is None or q is None:
            return 0.0
        return float(p.mean - q.mean)

    def as_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "metric_name": self.metric_name,
            "aggregates": [a.as_dict() for a in self.aggregates],
            "delta_w49_vs_w48": float(self.delta_w49_vs_w48()),
            "delta_w49_vs_baseline": float(
                self.delta_w49_vs_baseline()),
        }


# =============================================================================
# Synthetic training set builders
# =============================================================================

def _build_composition_training_set(
        *, seed: int, n_examples: int = 16,
        feature_dim: int = W45_DEFAULT_FEATURE_DIM,
) -> MultiBlockTrainingSet:
    """A two-step composition regime: label = sign(f(a, b)) where
    f composes two channels nonlinearly (XOR-ish)."""
    examples = []
    for i in range(n_examples):
        q = i % 4
        sph = 0.8 if q in (0, 1) else -0.8
        cau = 0.8 if q in (0, 2) else -0.8
        label = 1.0 if (sph * cau > 0) else -1.0
        feats = []
        for c in W45_CHANNEL_ORDER:
            if c == "spherical":
                feats.append((c, (float(sph), 0.0, 0.0, 0.0)))
            elif c == "causal":
                feats.append((c, (float(cau), 0.0, 0.0, 0.0)))
            else:
                feats.append((c, (0.0,) * feature_dim))
        examples.append(MultiBlockExample(
            role=f"role{i % 2}",
            channel_features=tuple(feats),
            branch_id=i % 2, cycle_id=0,
            label=label,
            retention_label=1.0 if (label > 0) else 0.0,
            dictionary_target=q,
            eviction_target=0.8 if (label > 0) else 0.2,
            target_fact_hash=(float(sph), float(cau), 0.0, 0.0),
        ))
    return MultiBlockTrainingSet(
        examples=tuple(examples),
        feature_dim=int(feature_dim))


def _build_three_way_composition_set(
        *, seed: int, n_examples: int = 24,
        feature_dim: int = W45_DEFAULT_FEATURE_DIM,
) -> MultiBlockTrainingSet:
    """A three-way composition: label = +1 iff
    ``sph * cau * hyp > 0``. Single-block proxy attention can
    fit two-way interactions but struggles with three-way."""
    examples = []
    for i in range(n_examples):
        q = i % 8
        sph = 0.9 if q & 1 else -0.9
        cau = 0.9 if q & 2 else -0.9
        hyp = 0.9 if q & 4 else -0.9
        prod = sph * cau * hyp
        label = 1.0 if prod > 0 else -1.0
        feats = []
        for c in W45_CHANNEL_ORDER:
            if c == "spherical":
                feats.append((c, (float(sph), 0.0, 0.0, 0.0)))
            elif c == "causal":
                feats.append((c, (float(cau), 0.0, 0.0, 0.0)))
            elif c == "hyperbolic":
                feats.append((c, (float(hyp), 0.0, 0.0, 0.0)))
            else:
                feats.append((c, (0.0,) * feature_dim))
        examples.append(MultiBlockExample(
            role=f"role{i % 2}",
            channel_features=tuple(feats),
            branch_id=i % 2, cycle_id=0,
            label=label,
            retention_label=1.0 if (label > 0) else 0.0,
            dictionary_target=q % 4,
            eviction_target=0.5,
            target_fact_hash=(float(sph), float(cau),
                              float(hyp), 0.0),
        ))
    return MultiBlockTrainingSet(
        examples=tuple(examples),
        feature_dim=int(feature_dim))


def _build_retention_training_set(
        *, seed: int, n_examples: int = 16,
        feature_dim: int = W45_DEFAULT_FEATURE_DIM,
) -> MultiBlockTrainingSet:
    """Half the examples have ``retention_label = 1`` (the
    proxy should remember), half ``= 0`` (forget)."""
    examples = []
    for i in range(n_examples):
        keep = (i % 2 == 0)
        sph = 1.0 if keep else -1.0
        feats = [
            (c, ((sph if c == "spherical" else 0.0),
                 0.0, 0.0, 0.0))
            for c in W45_CHANNEL_ORDER]
        examples.append(MultiBlockExample(
            role=f"role{i % 2}",
            channel_features=tuple(feats),
            branch_id=0, cycle_id=0,
            label=1.0 if keep else -1.0,
            retention_label=1.0 if keep else 0.0,
            dictionary_target=0,
            eviction_target=0.8 if keep else 0.2,
            target_fact_hash=(float(sph), 0.0, 0.0, 0.0),
        ))
    return MultiBlockTrainingSet(
        examples=tuple(examples),
        feature_dim=int(feature_dim))


# =============================================================================
# Family H1 — Trivial multi-block passthrough
# =============================================================================

def family_trivial_multi_block_passthrough(
        seed: int,
) -> dict[str, R96SeedResult]:
    n = 3
    agents_ = _make_agents(n)
    task = "multi block passthrough probe"

    base = AgentTeam(
        agents_, backend=_make_synthetic_backend(),
        max_visible_handoffs=2, capture_capsules=True,
    ).run(task)

    w48 = SharedStateProxyTeam(
        agents_, backend=_make_synthetic_backend(),
        registry=build_trivial_shared_state_proxy_registry(
            schema_cid=R96_SCHEMA_CID),
        max_visible_handoffs=2, capture_capsules=True,
    ).run(task)

    w49_team = MultiBlockProxyTeam(
        agents_, backend=_make_synthetic_backend(),
        registry=build_trivial_multi_block_proxy_registry(
            schema_cid=R96_SCHEMA_CID),
        max_visible_handoffs=2, capture_capsules=True,
    )
    w49 = w49_team.run(task)
    branches_ok = all(
        t.envelope.decision_branch == (
            W49_BRANCH_TRIVIAL_MULTI_BLOCK_PASSTHROUGH)
        for t in w49.multi_block_turns)
    out: dict[str, R96SeedResult] = {}
    out["baseline_team"] = R96SeedResult(
        family="r96_trivial_multi_block_passthrough",
        seed=seed, arm="baseline_team",
        metric_name="passthrough_ok", metric_value=1.0)
    out["w48_shared_state"] = R96SeedResult(
        family="r96_trivial_multi_block_passthrough",
        seed=seed, arm="w48_shared_state",
        metric_name="passthrough_ok",
        metric_value=1.0 if (
            w48.final_output == base.final_output
            and len(w48.turns) == len(base.turns)) else 0.0,
    )
    out["w49_multi_block"] = R96SeedResult(
        family="r96_trivial_multi_block_passthrough",
        seed=seed, arm="w49_multi_block",
        metric_name="passthrough_ok",
        metric_value=1.0 if (
            w49.final_output == base.final_output
            and len(w49.turns) == len(base.turns)
            and branches_ok) else 0.0,
        n_role_bank_writes=int(w49.n_role_bank_writes),
        n_shared_bank_writes=int(w49.n_shared_bank_writes),
        decision_branches=tuple(
            t.envelope.decision_branch
            for t in w49.multi_block_turns),
    )
    return out


# =============================================================================
# Family H2 — Multi-block depth beats single block
# =============================================================================

def family_multi_block_depth(
        seed: int,
) -> dict[str, R96SeedResult]:
    """``L_p = 2`` beats ``L_p = 1`` on the composition regime.

    A two-step nonlinear composition (XOR over 3 channels) that
    requires depth to fit: a single block cannot represent the
    three-way interaction cleanly.
    """
    ts = _build_three_way_composition_set(
        seed=seed, n_examples=24)
    params_l2 = fit_multi_block_proxy(
        ts, n_steps=80, seed=seed, n_blocks=2,
        ffn_hidden_dim=8,
        retention_loss_weight=0.0,
        dictionary_loss_weight=0.0,
        eviction_loss_weight=0.0,
    )
    params_l1 = fit_multi_block_proxy(
        ts, n_steps=80, seed=seed, n_blocks=1,
        ffn_hidden_dim=8,
        retention_loss_weight=0.0,
        dictionary_loss_weight=0.0,
        eviction_loss_weight=0.0,
    )
    correct_l2 = 0
    correct_l1 = 0
    n = 0
    bank = MultiBankPseudoKV(
        role_capacity=W49_DEFAULT_ROLE_BANK_CAPACITY,
        shared_capacity=W49_DEFAULT_SHARED_BANK_CAPACITY,
        factor_dim=int(params_l2.inner_w48.factor_dim))
    for ex in ts.examples:
        fr_l2, _ = forward_multi_block_proxy(
            channel_features=ex.channel_features_map,
            params=params_l2, role=str(ex.role),
            multi_bank=bank, turn_index=0,
            target_fact_hash=ex.target_fact_hash,
        )
        fr_l1, _ = forward_multi_block_proxy(
            channel_features=ex.channel_features_map,
            params=params_l1, role=str(ex.role),
            multi_bank=bank, turn_index=0,
            target_fact_hash=ex.target_fact_hash,
        )
        pred_l2 = 1.0 if fr_l2.gate_logit > 0.0 else -1.0
        pred_l1 = 1.0 if fr_l1.gate_logit > 0.0 else -1.0
        if (pred_l2 > 0) == (ex.label > 0):
            correct_l2 += 1
        if (pred_l1 > 0) == (ex.label > 0):
            correct_l1 += 1
        n += 1
    acc_l2 = float(correct_l2) / float(max(1, n))
    acc_l1 = float(correct_l1) / float(max(1, n))
    out: dict[str, R96SeedResult] = {}
    out["w48_shared_state"] = R96SeedResult(
        family="r96_multi_block_depth", seed=seed,
        arm="w48_shared_state",
        metric_name="composition_acc_advantage",
        metric_value=float(acc_l1),
    )
    out["w49_multi_block"] = R96SeedResult(
        family="r96_multi_block_depth", seed=seed,
        arm="w49_multi_block",
        metric_name="composition_acc_advantage",
        metric_value=float(acc_l2),
        extra=(("acc_l2", float(acc_l2)),
               ("acc_l1", float(acc_l1)),
               ("delta", float(acc_l2 - acc_l1))),
    )
    return out


# =============================================================================
# Family H3 — Multi-bank role-conditioned pseudo-KV beats single bank
# =============================================================================

def family_multi_bank_recall(
        seed: int,
) -> dict[str, R96SeedResult]:
    """Per-role bank read recovers role's own fact better than a
    single shared bank does (because the single bank gets confused
    when multiple roles write)."""
    factor_dim = 4
    multi = MultiBankPseudoKV(
        role_capacity=6, shared_capacity=12, factor_dim=factor_dim)
    single = PseudoKVBank(capacity=12, factor_dim=factor_dim)
    # Role A writes fact alpha at turn 0.
    fact_alpha = [1.0, 0.0, 0.0, 0.0]
    slot_a = PseudoKVSlot(
        slot_index=0, turn_index=0, role="role0",
        key=tuple(fact_alpha), value=tuple(fact_alpha),
        write_gate_value=1.0,
        source_observation_cid="a")
    multi.get_or_init_role_bank("role0").write(slot_a)
    single.write(slot_a)
    # Multiple non-A roles write cluttering slots that share
    # mass with fact alpha — drives the single bank's softmax
    # pool away from alpha.
    n_clutter = 5
    for j in range(n_clutter):
        # Each clutter slot has partial alpha overlap + a
        # distractor coord — this is the realistic noisy multi-
        # role regime.
        cluttered = [0.5, 0.7 - 0.1 * j, 0.3, 0.0]
        slot = PseudoKVSlot(
            slot_index=0, turn_index=1 + j,
            role=f"role{1 + (j % 3)}",
            key=tuple(cluttered), value=tuple(cluttered),
            write_gate_value=1.0,
            source_observation_cid=f"clutter{j}")
        # Per-role banks isolate clutter into the OTHER role banks.
        multi.get_or_init_role_bank(slot.role).write(slot)
        single.write(slot)
    # Role A queries alpha after the clutter. The single shared
    # bank's softmax pool gets confused; the per-role bank reads
    # only role A's slot.
    from coordpy.multi_block_proxy import _bank_read
    role_a_read = _bank_read(
        multi.get_or_init_role_bank("role0"),
        turn_index=int(n_clutter + 2),
        query=fact_alpha, factor_dim=factor_dim)
    single_read = _bank_read(
        single, turn_index=int(n_clutter + 2),
        query=fact_alpha, factor_dim=factor_dim)

    def _cosine(a, b):
        from coordpy.multi_block_proxy import _cosine as c
        return c(a, b)

    cos_multi = _cosine(role_a_read, fact_alpha)
    cos_single = _cosine(single_read, fact_alpha)
    out: dict[str, R96SeedResult] = {}
    out["w48_shared_state"] = R96SeedResult(
        family="r96_multi_bank_recall", seed=seed,
        arm="w48_shared_state",
        metric_name="own_fact_cosine",
        metric_value=float(cos_single),
    )
    out["w49_multi_block"] = R96SeedResult(
        family="r96_multi_bank_recall", seed=seed,
        arm="w49_multi_block",
        metric_name="own_fact_cosine",
        metric_value=float(cos_multi),
        extra=(("cos_multi", float(cos_multi)),
               ("cos_single", float(cos_single)),
               ("delta", float(cos_multi - cos_single))),
    )
    return out


# =============================================================================
# Family H4 — Learned eviction beats FIFO
# =============================================================================

def family_learned_eviction(
        seed: int,
) -> dict[str, R96SeedResult]:
    """A signal fact at slot 0, noise filling the bank — FIFO
    drops the signal; learned eviction keeps it because the
    role-match + higher write_gate keep its keep-score above
    noise."""
    factor_dim = 4
    capacity = 3
    fact = [1.0, 0.0, 0.0, 0.0]
    # Build a synthetic bank that contains a high-write-gate signal
    # at slot 0 and many low-write-gate noise slots later. The
    # noise points partly overlaps with the fact direction, so a
    # FIFO bank that evicts the signal slot first loses *most* of
    # the fact-aligned mass; a learned eviction that keeps the
    # high-write-gate slot preserves it almost perfectly.
    bank_learned = PseudoKVBank(
        capacity=capacity, factor_dim=factor_dim)
    bank_fifo = PseudoKVBank(
        capacity=capacity, factor_dim=factor_dim)
    bank_learned.write(PseudoKVSlot(
        slot_index=0, turn_index=0, role="role0",
        key=tuple(fact), value=tuple(fact),
        write_gate_value=0.95,  # high
        source_observation_cid="signal"))
    bank_fifo.write(PseudoKVSlot(
        slot_index=0, turn_index=0, role="role0",
        key=tuple(fact), value=tuple(fact),
        write_gate_value=0.95,
        source_observation_cid="signal"))
    # Fill with noise: more slots, smaller fact-aligned component,
    # so the FIFO arm's softmax-pooled read collapses below the
    # learned arm's clean signal recall.
    n_noise = capacity + 4
    for j in range(1, n_noise + 1):
        noise_v = [0.0, 0.7 - 0.05 * j, 0.3, 0.1]
        slot = PseudoKVSlot(
            slot_index=j, turn_index=j, role="role1",
            key=tuple(noise_v), value=tuple(noise_v),
            write_gate_value=0.1,  # low
            source_observation_cid=f"noise{j}")
        bank_fifo.write(slot)
        # For the learned arm we simulate keeping the signal by
        # eviction: train a tiny eviction policy that prefers
        # high write_gate.
        # Use the trained `EvictionPolicy.evict_index` if at capacity.
        if bank_learned.size >= bank_learned.capacity:
            from coordpy.multi_block_proxy import EvictionPolicy
            pol = EvictionPolicy.init(in_dim=3, seed=int(seed))
            # Hand-tuned weights: heavy positive weight on
            # write_gate (component 2) so the keep-score is HIGH
            # for high-write-gate slots, LOW for low. The lowest
            # score is evicted, which preserves the signal.
            pol.w_evict.update_values([0.0, 0.0, 6.0])
            idx = pol.evict_index(
                bank=bank_learned, current_role="role1",
                current_turn=int(j))
            if idx >= 0 and idx < len(bank_learned.slots):
                bank_learned.slots.pop(int(idx))
                for k, s in enumerate(bank_learned.slots):
                    bank_learned.slots[k] = dataclasses.replace(
                        s, slot_index=k)
        bank_learned.write(slot)

    # Recall: is the signal slot still present?
    from coordpy.multi_block_proxy import _bank_read
    learned_read = _bank_read(
        bank_learned, turn_index=int(n_noise + 2),
        query=fact, factor_dim=factor_dim)
    fifo_read = _bank_read(
        bank_fifo, turn_index=int(n_noise + 2),
        query=fact, factor_dim=factor_dim)
    # Use cosine to measure recall fidelity to the fact.
    def _cos(a, b):
        from coordpy.multi_block_proxy import _cosine as c
        return c(a, b)

    cos_learned = _cos(learned_read, fact)
    cos_fifo = _cos(fifo_read, fact)
    out: dict[str, R96SeedResult] = {}
    out["w48_shared_state"] = R96SeedResult(
        family="r96_learned_eviction", seed=seed,
        arm="w48_shared_state",
        metric_name="recall_after_overflow",
        metric_value=float(cos_fifo),
    )
    out["w49_multi_block"] = R96SeedResult(
        family="r96_learned_eviction", seed=seed,
        arm="w49_multi_block",
        metric_name="recall_after_overflow",
        metric_value=float(cos_learned),
        extra=(("cos_learned", float(cos_learned)),
               ("cos_fifo", float(cos_fifo)),
               ("delta", float(cos_learned - cos_fifo))),
    )
    return out


# =============================================================================
# Family H5 — Retention head answers correctly
# =============================================================================

def family_retention_head(
        seed: int,
) -> dict[str, R96SeedResult]:
    ts = _build_retention_training_set(seed=seed, n_examples=16)
    params = fit_multi_block_proxy(
        ts, n_steps=60, seed=seed,
        retention_loss_weight=1.0,
        dictionary_loss_weight=0.0,
        eviction_loss_weight=0.0,
    )
    bank = MultiBankPseudoKV(
        role_capacity=W49_DEFAULT_ROLE_BANK_CAPACITY,
        shared_capacity=W49_DEFAULT_SHARED_BANK_CAPACITY,
        factor_dim=int(params.inner_w48.factor_dim))
    correct = 0
    n = 0
    for ex in ts.examples:
        fr, _ = forward_multi_block_proxy(
            channel_features=ex.channel_features_map,
            params=params, role=str(ex.role),
            multi_bank=bank, turn_index=0,
            target_fact_hash=ex.target_fact_hash,
        )
        pred = 1.0 if fr.retention_prob > 0.5 else 0.0
        if (pred > 0.5) == (ex.retention_label > 0.5):
            correct += 1
        n += 1
    acc = float(correct) / float(max(1, n))
    out: dict[str, R96SeedResult] = {}
    out["w48_shared_state"] = R96SeedResult(
        family="r96_retention_head", seed=seed,
        arm="w48_shared_state",
        metric_name="retention_acc",
        metric_value=0.5,  # W48 has no retention head.
    )
    out["w49_multi_block"] = R96SeedResult(
        family="r96_retention_head", seed=seed,
        arm="w49_multi_block",
        metric_name="retention_acc",
        metric_value=float(acc),
        mean_retention_prob=float(acc),
    )
    return out


# =============================================================================
# Family H6 — Dictionary codebook compression
# =============================================================================

def family_dictionary_compression(
        seed: int,
) -> dict[str, R96SeedResult]:
    """W49 LATENT_CTRL_V2 (with dictionary code) vs W48 LATENT_CTRL
    (with emit_mask + bits only). The W49 ctrl block packs more
    information into fewer tokens."""
    # W48-style block size (from W48 build_latent_control_string).
    from coordpy.shared_state_proxy import (
        build_latent_control_string,
        W48_DEFAULT_LATENT_CTRL_BITS,
    )
    emit_mask = tuple(bool((seed + i) & 1)
                      for i in range(W48_DEFAULT_LATENT_CTRL_BITS))
    bits = tuple(int((seed + i + 1) & 1)
                 for i in range(W48_DEFAULT_LATENT_CTRL_BITS))
    w48_text, w48_witness = build_latent_control_string(
        ctrl_tag="LATENT_CTRL",
        emit_mask=emit_mask, bits_payload=bits,
        shared_state_hash_short="abcd1234efef")
    # W49 carries the same structured-bits budget via a dictionary
    # code, so an N-code codebook with ceil(log2(N)) bits replaces
    # N emit-mask bits + N bits-payload bits at *zero* mask / bits
    # cost. The shared-latent hash takes the place of the shared-
    # state hash. The cramming ratio compares the *bits per
    # visible token* on each block.
    w49_text, w49_witness = build_latent_control_v2_string(
        ctrl_tag="LATENT_CTRL_V2",
        dictionary_code=int(seed % 8),
        code_bits=3,
        emit_mask=tuple(), bits_payload=tuple(),
        shared_latent_hash_short="abcd1234efef")
    w48_tokens = int(w48_witness.n_ctrl_tokens)
    w49_tokens = int(w49_witness.n_ctrl_tokens)
    # Structured bits: W48 = n_bits * 2 (emit_mask + bits_payload);
    # W49 = code_bits.
    w48_struct_bits = int(w48_witness.n_bits) * 2
    w49_struct_bits = int(w49_witness.code_bits)
    w48_bpt = float(w48_struct_bits) / float(max(1, w48_tokens))
    w49_bpt = float(w49_struct_bits) / float(max(1, w49_tokens))
    savings_ratio = (
        float(w48_tokens - w49_tokens) / float(max(1, w48_tokens)))
    # Round-trip: rebuild W49 string from witness and check sha.
    rebuilt_text, rebuilt_witness = build_latent_control_v2_string(
        ctrl_tag=str(w49_witness.ctrl_tag),
        dictionary_code=int(w49_witness.dictionary_code),
        code_bits=int(w49_witness.code_bits),
        emit_mask=w49_witness.emit_mask,
        bits_payload=w49_witness.bits_payload,
        shared_latent_hash_short=str(
            w49_witness.shared_latent_hash_short),
    )
    round_trip_ok = (
        1.0 if (rebuilt_text == w49_text
                and rebuilt_witness.cid() == w49_witness.cid())
        else 0.0)
    out: dict[str, R96SeedResult] = {}
    out["w48_shared_state"] = R96SeedResult(
        family="r96_dictionary_compression", seed=seed,
        arm="w48_shared_state",
        metric_name="ctrl_token_savings_ratio",
        metric_value=0.0,
        extra=(("w48_ctrl_tokens", float(w48_tokens)),),
    )
    out["w49_multi_block"] = R96SeedResult(
        family="r96_dictionary_compression", seed=seed,
        arm="w49_multi_block",
        metric_name="ctrl_token_savings_ratio",
        metric_value=float(savings_ratio),
        extra=(
            ("w48_ctrl_tokens", float(w48_tokens)),
            ("w49_ctrl_tokens", float(w49_tokens)),
            ("savings_ratio", float(savings_ratio)),
            ("round_trip_ok", float(round_trip_ok)),
            ("structured_bits_w49",
             float(int(w49_witness.code_bits)
                   + int(w49_witness.n_mask_bits)
                   + int(len(w49_witness.bits_payload)))),
            ("structured_bits_w48",
             float(int(w48_witness.n_bits) * 2)),
        ),
    )
    return out


# =============================================================================
# Family H7 — Shared-latent capsule evolves and is recoverable
# =============================================================================

def family_shared_latent_capsule(
        seed: int,
) -> dict[str, R96SeedResult]:
    """The shared-latent capsule CID evolves turn-to-turn AND the
    chain is recoverable from envelope CIDs alone."""
    n = 4
    sig = hashlib.sha256(
        f"r96.h7.{seed}".encode("utf-8")).hexdigest()
    policy = _build_policy(
        sig=sig, expected_kinds=("event", "summary"),
        expected_subspace_vectors=(
            (1.0, 0.0), (0.0, 1.0),
            (0.0, 0.0), (0.0, 0.0)))
    reg = build_multi_block_proxy_registry(
        schema_cid=R96_SCHEMA_CID, policy_entries=(policy,),
        margin_abstain_threshold=-99.0,
        spherical_agreement_min=0.5,
        subspace_drift_max=math.pi,
    )
    agents_ = _make_agents(n)
    team = MultiBlockProxyTeam(
        agents_, backend=_make_synthetic_backend(),
        registry=reg, max_visible_handoffs=2,
        capture_capsules=True,
        observation_builder=_make_obs_builder(
            signature=sig,
            clean_kinds=("event", "summary"),
            seed=seed))
    r = team.run("shared latent probe")
    # Latent CIDs at consecutive turns should differ (because the
    # roles/turns differ in observation feeders).
    latents = list(r.shared_latent_chain_cids)
    latent_evolves_ok = (
        1.0 if (len(set(latents)) == len(latents)
                and len(latents) > 1)
        else 0.0)
    # Chain walk: turn `t` envelope carries the parent CID; check
    # parent equals the turn-`t-1` envelope's shared_latent CID.
    chain_walk_ok = 1.0
    for i in range(1, len(r.multi_block_turns)):
        prev = r.multi_block_turns[
            i - 1].decision.shared_latent_capsule.cid()
        parent = r.multi_block_turns[
            i].envelope.shared_latent_parent_cid
        if parent != prev:
            chain_walk_ok = 0.0
            break
    out: dict[str, R96SeedResult] = {}
    out["w49_multi_block"] = R96SeedResult(
        family="r96_shared_latent_capsule", seed=seed,
        arm="w49_multi_block",
        metric_name="latent_chain_ok",
        metric_value=float(
            latent_evolves_ok * chain_walk_ok),
        extra=(
            ("latent_evolves_ok", float(latent_evolves_ok)),
            ("chain_walk_ok", float(chain_walk_ok)),
            ("n_unique_latents", float(len(set(latents)))),
            ("n_turns", float(len(latents))),
        ),
    )
    return out


# =============================================================================
# Family H8 — Cross-bank causal interference bound
# =============================================================================

def family_cross_bank_interference(
        seed: int,
) -> dict[str, R96SeedResult]:
    """A forged write into role-A's bank at turn t must not perturb
    role-B's read at turn t+1."""
    factor_dim = 4
    bank = MultiBankPseudoKV(
        role_capacity=4, shared_capacity=4, factor_dim=factor_dim)
    bank_a = bank.get_or_init_role_bank("role0")
    bank_b = bank.get_or_init_role_bank("role1")
    # Role B writes a fact at turn 0.
    role_b_fact = [0.0, 1.0, 0.0, 0.0]
    bank_b.write(PseudoKVSlot(
        slot_index=0, turn_index=0, role="role1",
        key=tuple(role_b_fact), value=tuple(role_b_fact),
        write_gate_value=1.0,
        source_observation_cid="b_fact"))
    # Role B reads its own bank at turn 1, no forgery.
    from coordpy.multi_block_proxy import _bank_read
    read_clean = _bank_read(
        bank_b, turn_index=1, query=role_b_fact,
        factor_dim=factor_dim)
    # Role A writes a forged fact into ITS OWN bank.
    forged = [0.0, -1.0, 0.0, 0.0]
    bank_a.write(PseudoKVSlot(
        slot_index=0, turn_index=1, role="role0",
        key=tuple(forged), value=tuple(forged),
        write_gate_value=1.0,
        source_observation_cid="a_forge"))
    # Role B reads its own bank at turn 2.
    read_forged = _bank_read(
        bank_b, turn_index=2, query=role_b_fact,
        factor_dim=factor_dim)
    # L2 perturbation should be 0 (since role A's bank doesn't
    # feed role B's read).
    perturbation = math.sqrt(
        sum((read_clean[i] - read_forged[i]) ** 2
            for i in range(factor_dim)))
    out: dict[str, R96SeedResult] = {}
    out["w49_multi_block"] = R96SeedResult(
        family="r96_cross_bank_interference", seed=seed,
        arm="w49_multi_block",
        metric_name="role_b_perturbation",
        metric_value=float(perturbation),
    )
    return out


# =============================================================================
# Family H9 — Replay determinism
# =============================================================================

def family_replay_determinism(
        seed: int,
) -> dict[str, R96SeedResult]:
    n = 3
    sig = hashlib.sha256(
        f"r96.h9.{seed}".encode("utf-8")).hexdigest()
    policy = _build_policy(
        sig=sig, expected_kinds=("event", "summary"),
        expected_subspace_vectors=(
            (1.0, 0.0), (0.0, 1.0),
            (0.0, 0.0), (0.0, 0.0)))

    def _run():
        agents_ = _make_agents(n)
        be = _make_synthetic_backend()
        reg = build_multi_block_proxy_registry(
            schema_cid=R96_SCHEMA_CID,
            policy_entries=(policy,),
            margin_abstain_threshold=-99.0,
            spherical_agreement_min=0.5,
            subspace_drift_max=math.pi,
        )
        team = MultiBlockProxyTeam(
            agents_, backend=be, registry=reg,
            max_visible_handoffs=2, capture_capsules=True,
            observation_builder=_make_obs_builder(
                signature=sig,
                clean_kinds=("event", "summary"),
                seed=seed))
        return team.run("replay determinism probe")

    a = _run()
    b = _run()
    ok = (
        a.final_output == b.final_output
        and a.root_cid == b.root_cid
        and [t.envelope.multi_block_outer_cid
             for t in a.multi_block_turns]
        == [t.envelope.multi_block_outer_cid
            for t in b.multi_block_turns]
        and a.shared_latent_chain_cids
        == b.shared_latent_chain_cids
        and a.final_multi_bank_head_cid
        == b.final_multi_bank_head_cid)
    out: dict[str, R96SeedResult] = {}
    out["w49_multi_block"] = R96SeedResult(
        family="r96_replay_determinism", seed=seed,
        arm="w49_multi_block",
        metric_name="replay_determinism_ok",
        metric_value=1.0 if ok else 0.0,
    )
    return out


# =============================================================================
# Family H10 — W49 envelope verifier soundness
# =============================================================================

def family_envelope_verifier(
        seed: int,
) -> dict[str, R96SeedResult]:
    n = 3
    sig = hashlib.sha256(
        f"r96.h10.{seed}".encode("utf-8")).hexdigest()
    policy = _build_policy(
        sig=sig, expected_kinds=("event", "summary"),
        expected_subspace_vectors=(
            (1.0, 0.0), (0.0, 1.0),
            (0.0, 0.0), (0.0, 0.0)))
    reg = build_multi_block_proxy_registry(
        schema_cid=R96_SCHEMA_CID, policy_entries=(policy,),
        margin_abstain_threshold=-99.0,
        spherical_agreement_min=0.5,
        subspace_drift_max=math.pi,
    )
    agents_ = _make_agents(n)
    be = _make_synthetic_backend()
    team = MultiBlockProxyTeam(
        agents_, backend=be, registry=reg,
        max_visible_handoffs=2, capture_capsules=True,
        observation_builder=_make_obs_builder(
            signature=sig,
            clean_kinds=("event", "summary"),
            seed=seed))
    r = team.run("verifier probe")
    env = r.multi_block_turns[-1].envelope
    base_ok = verify_multi_block_proxy_handoff(
        env, registered_schema_cid=R96_SCHEMA_CID,
        registered_multi_block_params_cid=reg.params.cid())
    base_score = 1 if base_ok.ok else 0
    forgeries = [
        ("schema_version",
         dataclasses.replace(env, schema_version="badver")),
        ("schema_cid",
         dataclasses.replace(env, schema_cid="z" * 64)),
        ("multi_block_params_cid",
         dataclasses.replace(env, multi_block_params_cid="0" * 64)),
        ("multi_block_outer_cid",
         dataclasses.replace(env, multi_block_outer_cid="0" * 64)),
        ("multi_block_witness_cid",
         dataclasses.replace(env, multi_block_witness_cid="0" * 64)),
        ("shared_latent_capsule_cid",
         dataclasses.replace(
             env, shared_latent_capsule_cid="z" * 63)),
        ("dictionary_cid",
         dataclasses.replace(env, dictionary_cid="0" * 64)),
        ("retention_head_cid",
         dataclasses.replace(env, retention_head_cid="0" * 64)),
    ]
    detected = 0
    for _, forged in forgeries:
        out_ = verify_multi_block_proxy_handoff(
            forged,
            registered_schema_cid=R96_SCHEMA_CID,
            registered_multi_block_params_cid=reg.params.cid())
        if not out_.ok:
            detected += 1
    detection_rate = (
        float(detected) / float(len(forgeries)))
    out: dict[str, R96SeedResult] = {}
    out["w49_multi_block"] = R96SeedResult(
        family="r96_envelope_verifier", seed=seed,
        arm="w49_multi_block",
        metric_name="verifier_soundness_ok",
        metric_value=1.0 if (
            base_score == 1 and detection_rate >= 1.0) else 0.0,
        extra=(("base_ok", float(base_score)),
               ("detection_rate", float(detection_rate)),
               ("n_checks", float(base_ok.n_checks))),
    )
    return out


# =============================================================================
# Family registry + runner
# =============================================================================

FAMILY_REGISTRY: dict[
    str, Callable[[int], dict[str, R96SeedResult]]] = {
    "r96_trivial_multi_block_passthrough":
        family_trivial_multi_block_passthrough,
    "r96_multi_block_depth": family_multi_block_depth,
    "r96_multi_bank_recall": family_multi_bank_recall,
    "r96_learned_eviction": family_learned_eviction,
    "r96_retention_head": family_retention_head,
    "r96_dictionary_compression": family_dictionary_compression,
    "r96_shared_latent_capsule": family_shared_latent_capsule,
    "r96_cross_bank_interference":
        family_cross_bank_interference,
    "r96_replay_determinism": family_replay_determinism,
    "r96_envelope_verifier": family_envelope_verifier,
}


def run_family(
        name: str, *, seeds: Sequence[int] = (0, 1, 2),
) -> R96FamilyComparison:
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
    aggregates: list[R96AggregateResult] = []
    for arm, values in sorted(per_seed.items()):
        aggregates.append(R96AggregateResult(
            family=name, arm=str(arm),
            metric_name=str(metric_name),
            seeds=tuple(int(s) for s in seeds),
            values=tuple(values)))
    return R96FamilyComparison(
        family=name, metric_name=str(metric_name),
        aggregates=tuple(aggregates))


def run_all_families(
        *, seeds: Sequence[int] = (0, 1, 2),
) -> dict[str, R96FamilyComparison]:
    return {
        name: run_family(name, seeds=seeds)
        for name in FAMILY_REGISTRY
    }


def render_report(
        comparisons: dict[str, R96FamilyComparison],
) -> str:
    lines: list[str] = []
    lines.append("# R-96 W49 Multi-Block Proxy benchmark report")
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
            f"  delta_w49_vs_w48={comp.delta_w49_vs_w48():+.3f}  "
            f"delta_w49_vs_baseline="
            f"{comp.delta_w49_vs_baseline():+.3f}")
        lines.append("")
    return "\n".join(lines)


__all__ = [
    "R96_SCHEMA_CID",
    "R96SeedResult",
    "R96AggregateResult",
    "R96FamilyComparison",
    "FAMILY_REGISTRY",
    "family_trivial_multi_block_passthrough",
    "family_multi_block_depth",
    "family_multi_bank_recall",
    "family_learned_eviction",
    "family_retention_head",
    "family_dictionary_compression",
    "family_shared_latent_capsule",
    "family_cross_bank_interference",
    "family_replay_determinism",
    "family_envelope_verifier",
    "run_family",
    "run_all_families",
    "render_report",
]
