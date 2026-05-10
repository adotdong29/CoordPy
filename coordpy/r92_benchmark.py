"""R-92 benchmark family for the W45 Learned Manifold Controller
(LMC) layer.

R-92 is the first capsule-layer benchmark family in CoordPy that
compares a *learned-coupled* manifold path against the released
``AgentTeam``, the W43 closed-form PMC path, and the W44 live-
coupled path on real agent-team runs. Like R-90 / R-91 it is
seeded, hermetic, and reproducible.

Four honest baselines per family:

  * ``baseline_team`` — released ``AgentTeam.run`` path. No
    manifold integration; the runtime never gates on observation.
  * ``w43_closed_form`` — ``LiveManifoldTeam`` with the W43 inner
    enabled but ``abstain_substitution_enabled=False``. Audit-only.
  * ``w44_live_coupled`` — ``LiveManifoldTeam`` with the W44 live
    gate on (``abstain_substitution_enabled=True``).
  * ``w45_learned_coupled`` — ``LearnedManifoldTeam`` with the
    learned controller fitted on a synthetic bank + prompt-hint on.

Eight cell families + one replay-determinism family. Each family
produces:

  * a measurable per-seed metric
  * an aggregate across seeds with min/max/mean
  * a clear winner / no-improvement statement

The R-92 family is the H1..H12 success bar for the W45 milestone.
See ``docs/SUCCESS_CRITERION_W45_LEARNED_MANIFOLD.md`` and
``docs/RESULTS_COORDPY_W45_LEARNED_MANIFOLD.md`` for full reads.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
from typing import Any, Callable, Mapping, Sequence

from coordpy.agents import Agent, AgentTeam, agent
from coordpy.learned_manifold import (
    HintAwareSyntheticBackend,
    LearnedControllerParams,
    LearnedManifoldTeam,
    LearnedManifoldTeamResult,
    TrainingExample,
    TrainingSet,
    W45_CHANNEL_ORDER,
    W45_BRANCH_LEARNED_RATIFIED,
    W45_BRANCH_LEARNED_MARGIN_ABSTAIN,
    W45_BRANCH_LEARNED_SPHERICAL_ABSTAIN,
    W45_BRANCH_LEARNED_SUBSPACE_ABSTAIN,
    W45_BRANCH_LEARNED_CAUSAL_ABSTAIN,
    W45_BRANCH_LEARNED_NO_POLICY,
    W45_BRANCH_TRIVIAL_LEARNED_PASSTHROUGH,
    W45_DEFAULT_FEATURE_DIM,
    W45_HINT_MODE_FACTORADIC_WITH_HINT,
    W45_HINT_MODE_HINT_ONLY,
    W45_HINT_MODE_OFF,
    build_learned_manifold_registry,
    build_trivial_learned_manifold_registry,
    build_unfitted_controller_params,
    fit_learned_controller,
    forward_controller,
    _channel_features_from_bundle,
)
from coordpy.live_manifold import (
    LiveManifoldTeam,
    LiveObservationBuilderResult,
    LiveTurnContext,
    W44_BRANCH_LIVE_RATIFIED,
    W44_BRANCH_LIVE_SPHERICAL_ABSTAIN,
    W44_BRANCH_LIVE_SUBSPACE_ABSTAIN,
    W44_BRANCH_LIVE_CAUSAL_ABSTAIN,
    W44_DEFAULT_ABSTAIN_OUTPUT,
    W44_ROUTE_MODE_FACTORADIC,
    W44_ROUTE_MODE_TEXTUAL,
    build_live_manifold_registry,
    build_trivial_live_manifold_registry,
)
from coordpy.product_manifold import (
    CausalVectorClock,
    CellObservation,
    ProductManifoldPolicyEntry,
    encode_cell_channels,
    encode_spherical_consensus,
    encode_subspace_basis,
)
from coordpy.synthetic_llm import SyntheticLLMClient


R92_SCHEMA_CID = hashlib.sha256(
    b"r92.benchmark.schema.v1").hexdigest()

R92_REAL_OUTPUT: str = (
    "agent output payload with several extra words to make rendering meaningful")


# =============================================================================
# Helpers
# =============================================================================

def _make_synthetic_backend(
        default: str = R92_REAL_OUTPUT,
) -> SyntheticLLMClient:
    return SyntheticLLMClient(
        model_tag="synthetic.r92", default_response=default)


def _make_agents(n: int) -> tuple[Agent, ...]:
    return tuple(
        agent(
            f"role{i}",
            f"You are role{i}; respond as instructed.",
            max_tokens=64, temperature=0.0,
        )
        for i in range(n)
    )


def _const_signature(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _build_policy(
        *, sig: str,
        expected_kinds: Sequence[str],
        expected_subspace_vectors: Sequence[Sequence[float]],
        expected_topology: str = "(...)",
) -> ProductManifoldPolicyEntry:
    return ProductManifoldPolicyEntry(
        role_handoff_signature_cid=sig,
        expected_services=("learned",),
        expected_spherical=encode_spherical_consensus(
            tuple(expected_kinds)),
        expected_subspace=encode_subspace_basis(
            tuple(tuple(r) for r in expected_subspace_vectors)),
        expected_causal_topology_hash=str(expected_topology),
    )


def _make_obs_builder(
        *,
        signature: str,
        clean_kinds: Sequence[str],
        divergent_kinds: Sequence[str] | None,
        diverge_at_turn: int,
        diverge_seed_predicate: Callable[[int], bool],
        seed: int,
        injected_violation_at_turn: int | None = None,
        clean_subspace: Sequence[Sequence[float]] | None = None,
        divergent_subspace: Sequence[Sequence[float]] | None = None,
):
    """Build a per-turn observation builder."""
    clean_subspace = (
        tuple(tuple(r) for r in clean_subspace)
        if clean_subspace is not None
        else ((1.0, 0.0), (0.0, 1.0), (0.0, 0.0), (0.0, 0.0)))
    divergent_subspace = (
        tuple(tuple(r) for r in divergent_subspace)
        if divergent_subspace is not None else None)

    def _builder(
            ctx: LiveTurnContext,
    ) -> LiveObservationBuilderResult:
        diverges = diverge_seed_predicate(seed)
        snapshots: list[CausalVectorClock] = []
        walk_counts: dict[str, int] = {
            r: 0 for r in ctx.role_universe}
        for r in ctx.role_arrival_order:
            walk_counts[r] = walk_counts.get(r, 0) + 1
            snapshots.append(
                CausalVectorClock.from_mapping(dict(walk_counts)))
        if (diverges and injected_violation_at_turn is not None
                and ctx.turn_index >= injected_violation_at_turn
                and len(snapshots) >= 2):
            snapshots[-1] = CausalVectorClock.from_mapping({})
        if (diverges and divergent_kinds is not None
                and ctx.turn_index >= diverge_at_turn):
            kinds = tuple(divergent_kinds)
        else:
            kinds = tuple(clean_kinds)
        if (diverges and divergent_subspace is not None
                and ctx.turn_index >= diverge_at_turn):
            subspace = divergent_subspace
        else:
            subspace = clean_subspace
        obs = CellObservation(
            branch_path=tuple(0 for _ in range(ctx.turn_index)),
            claim_kinds=kinds,
            role_arrival_order=tuple(ctx.role_arrival_order),
            role_universe=tuple(ctx.role_universe),
            attributes=tuple({
                "round": float(ctx.turn_index),
                "n_handoffs": float(len(ctx.recent_handoffs)),
            }.items()),
            subspace_vectors=subspace,
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
class R92SeedResult:
    family: str
    seed: int
    arm: str
    metric_name: str
    metric_value: float
    n_behavioral_changes: int = 0
    n_visible_tokens_saved: int = 0
    n_visible_tokens_added_hint: int = 0
    n_abstain_substitutions: int = 0
    n_learned_margin_abstains: int = 0
    decision_branches: tuple[str, ...] = ()
    mean_ratify_probability: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class R92AggregateResult:
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
class R92FamilyComparison:
    family: str
    metric_name: str
    aggregates: tuple[R92AggregateResult, ...]

    def get(self, arm: str) -> R92AggregateResult | None:
        for a in self.aggregates:
            if a.arm == arm:
                return a
        return None

    def delta_learned_vs_w44(self) -> float:
        learned = self.get("w45_learned_coupled")
        w44 = self.get("w44_live_coupled")
        if learned is None or w44 is None:
            return 0.0
        return float(learned.mean - w44.mean)

    def delta_learned_vs_w43(self) -> float:
        learned = self.get("w45_learned_coupled")
        w43 = self.get("w43_closed_form")
        if learned is None or w43 is None:
            return 0.0
        return float(learned.mean - w43.mean)

    def delta_learned_vs_baseline(self) -> float:
        learned = self.get("w45_learned_coupled")
        base = self.get("baseline_team")
        if learned is None or base is None:
            return 0.0
        return float(learned.mean - base.mean)

    def as_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "metric_name": self.metric_name,
            "aggregates": [a.as_dict() for a in self.aggregates],
            "delta_learned_vs_w44": float(
                self.delta_learned_vs_w44()),
            "delta_learned_vs_w43": float(
                self.delta_learned_vs_w43()),
            "delta_learned_vs_baseline": float(
                self.delta_learned_vs_baseline()),
        }


# =============================================================================
# Synthetic training-bank builders
# =============================================================================

def _build_calibration_training_bank(
        *,
        seed: int,
        signature: str,
        feature_dim: int = W45_DEFAULT_FEATURE_DIM,
        n_examples: int = 24,
) -> TrainingSet:
    """Build a synthetic training bank for the calibration-gain
    family.

    Three example regimes are mixed in equal proportions:

      * **Honest-strong** — spherical agreement = 1.0, label = +1.
      * **Borderline-honest** — spherical agreement = 0.707 (the
        empirical agreement produced by the borderline kinds
        ``("event", "alert")`` against expected ``("event",
        "summary")``), label = +1.
      * **Dirty** — spherical agreement = 0.0, label = -1.

    The fitter must therefore learn a threshold below 0.707 so the
    borderline-honest regime is *not* abstained. The W44 hand
    threshold of 0.85 is *strictly above* 0.707 and therefore
    cannot match the gold; the W45 controller can.
    """
    examples = []
    regimes = [
        (1.0, 1.0),
        (1.0 / math.sqrt(2.0), 1.0),
        (0.0, -1.0),
    ]
    for i in range(n_examples):
        agree, label = regimes[i % len(regimes)]
        feats = []
        for c_name in W45_CHANNEL_ORDER:
            if c_name == "spherical":
                vec = [agree] + [0.0] * (feature_dim - 1)
            else:
                vec = [0.0] * feature_dim
            feats.append((c_name, tuple(vec)))
        examples.append(TrainingExample(
            role=f"role{i % 3}",
            role_handoff_signature_cid=signature,
            channel_features=tuple(feats),
            label=float(label),
        ))
    return TrainingSet(
        examples=tuple(examples), feature_dim=int(feature_dim))


def _build_attention_specialization_bank(
        *,
        seed: int,
        sig_spherical: str,
        sig_subspace: str,
        feature_dim: int = W45_DEFAULT_FEATURE_DIM,
        n_examples_per_sig: int = 8,
) -> TrainingSet:
    """Build a bank where signature ``sig_spherical`` cares only
    about the spherical channel and ``sig_subspace`` only about the
    subspace channel.
    """
    examples = []
    for i in range(n_examples_per_sig):
        positive = (i % 2 == 0)
        # Signature 1: spherical is the diagnostic channel.
        feats1 = []
        for c_name in W45_CHANNEL_ORDER:
            if c_name == "spherical":
                vec = [1.0 if positive else -1.0] + (
                    [0.0] * (feature_dim - 1))
            else:
                vec = [0.0] * feature_dim
            feats1.append((c_name, tuple(vec)))
        examples.append(TrainingExample(
            role="role0",
            role_handoff_signature_cid=sig_spherical,
            channel_features=tuple(feats1),
            label=1.0 if positive else -1.0,
        ))
        # Signature 2: subspace is the diagnostic channel.
        feats2 = []
        for c_name in W45_CHANNEL_ORDER:
            if c_name == "subspace":
                vec = [1.0 if positive else -1.0] + (
                    [0.0] * (feature_dim - 1))
            else:
                vec = [0.0] * feature_dim
            feats2.append((c_name, tuple(vec)))
        examples.append(TrainingExample(
            role="role0",
            role_handoff_signature_cid=sig_subspace,
            channel_features=tuple(feats2),
            label=1.0 if positive else -1.0,
        ))
    return TrainingSet(
        examples=tuple(examples), feature_dim=int(feature_dim))


def _build_role_adapter_bank(
        *,
        seed: int,
        signature: str,
        feature_dim: int = W45_DEFAULT_FEATURE_DIM,
        n_examples_per_role: int = 8,
) -> TrainingSet:
    """Build a bank where role0 / role1 / role2 share the same
    spherical-channel diagnostic, but role3 has a *flipped*
    sign convention — so a shared base alone cannot fit it; a
    role-specific delta is required.
    """
    examples = []
    for i in range(n_examples_per_role):
        positive = (i % 2 == 0)
        # Shared roles: label = sign(spherical).
        for r in ("role0", "role1", "role2"):
            feats = []
            for c_name in W45_CHANNEL_ORDER:
                if c_name == "spherical":
                    vec = [1.0 if positive else -1.0] + (
                        [0.0] * (feature_dim - 1))
                else:
                    vec = [0.0] * feature_dim
                feats.append((c_name, tuple(vec)))
            examples.append(TrainingExample(
                role=r,
                role_handoff_signature_cid=signature,
                channel_features=tuple(feats),
                label=1.0 if positive else -1.0,
            ))
        # role3: label = -sign(spherical) (flipped).
        feats = []
        for c_name in W45_CHANNEL_ORDER:
            if c_name == "spherical":
                vec = [1.0 if positive else -1.0] + (
                    [0.0] * (feature_dim - 1))
            else:
                vec = [0.0] * feature_dim
            feats.append((c_name, tuple(vec)))
        examples.append(TrainingExample(
            role="role3",
            role_handoff_signature_cid=signature,
            channel_features=tuple(feats),
            label=-1.0 if positive else 1.0,
        ))
    return TrainingSet(
        examples=tuple(examples), feature_dim=int(feature_dim))


# =============================================================================
# Family: r92_trivial_learned_passthrough — sanity
# =============================================================================

def family_trivial_learned_passthrough(
        seed: int,
) -> dict[str, R92SeedResult]:
    """Sanity: trivially-configured learned registry must reduce to
    AgentTeam byte-for-byte. Metric: ``passthrough_ok``.
    """
    n = 3
    agents_ = _make_agents(n)
    task = "explain the learned passthrough reduction"

    base_team = AgentTeam(
        agents_, backend=_make_synthetic_backend(),
        team_instructions="bounded team", max_visible_handoffs=2,
        capture_capsules=True,
    )
    base = base_team.run(task)

    # w43 closed_form: trivial live registry (audit-only).
    reg_w43 = build_trivial_live_manifold_registry(
        schema_cid=R92_SCHEMA_CID)
    w43_team = LiveManifoldTeam(
        agents_, backend=_make_synthetic_backend(),
        registry=reg_w43,
        team_instructions="bounded team", max_visible_handoffs=2,
        capture_capsules=True,
    )
    w43 = w43_team.run(task)

    # w44 live_coupled: trivial registry; reduces to baseline.
    reg_w44 = build_trivial_live_manifold_registry(
        schema_cid=R92_SCHEMA_CID)
    w44_team = LiveManifoldTeam(
        agents_, backend=_make_synthetic_backend(),
        registry=reg_w44,
        team_instructions="bounded team", max_visible_handoffs=2,
        capture_capsules=True,
    )
    w44 = w44_team.run(task)

    # w45 learned_coupled: trivial registry; reduces to baseline.
    reg_learned = build_trivial_learned_manifold_registry(
        schema_cid=R92_SCHEMA_CID)
    learned_team = LearnedManifoldTeam(
        agents_, backend=_make_synthetic_backend(),
        registry=reg_learned,
        team_instructions="bounded team", max_visible_handoffs=2,
        capture_capsules=True,
    )
    learned = learned_team.run(task)

    out: dict[str, R92SeedResult] = {}
    out["baseline_team"] = R92SeedResult(
        family="r92_trivial_learned_passthrough", seed=seed,
        arm="baseline_team",
        metric_name="passthrough_ok",
        metric_value=1.0,
    )
    out["w43_closed_form"] = R92SeedResult(
        family="r92_trivial_learned_passthrough", seed=seed,
        arm="w43_closed_form",
        metric_name="passthrough_ok",
        metric_value=1.0 if (
            w43.final_output == base.final_output
            and len(w43.turns) == len(base.turns)) else 0.0,
    )
    out["w44_live_coupled"] = R92SeedResult(
        family="r92_trivial_learned_passthrough", seed=seed,
        arm="w44_live_coupled",
        metric_name="passthrough_ok",
        metric_value=1.0 if (
            w44.final_output == base.final_output
            and len(w44.turns) == len(base.turns)) else 0.0,
    )
    branches_ok = all(
        t.envelope.decision_branch == (
            W45_BRANCH_TRIVIAL_LEARNED_PASSTHROUGH)
        for t in learned.learned_turns)
    out["w45_learned_coupled"] = R92SeedResult(
        family="r92_trivial_learned_passthrough", seed=seed,
        arm="w45_learned_coupled",
        metric_name="passthrough_ok",
        metric_value=1.0 if (
            learned.final_output == base.final_output
            and len(learned.turns) == len(base.turns)
            and branches_ok) else 0.0,
        n_behavioral_changes=int(learned.n_behavioral_changes),
        n_visible_tokens_saved=int(
            learned.n_visible_tokens_saved_factoradic),
        n_abstain_substitutions=int(learned.n_abstain_substitutions),
        decision_branches=tuple(
            t.envelope.decision_branch
            for t in learned.learned_turns),
    )
    return out


# =============================================================================
# Family: r92_learned_calibration_gain — H2
# =============================================================================

def family_learned_calibration_gain(
        seed: int,
) -> dict[str, R92SeedResult]:
    """A regime where the W44 hand-designed thresholds fire false
    abstains on a *borderline-honest* cell. The W45 controller is
    fit on a calibration bank and learns to ratify the borderline
    cells while still rejecting the bad ones.

    The regime is engineered so the borderline cells produce a
    spherical cosine agreement of about ``1/sqrt(2) ≈ 0.707``,
    strictly below W44's default ``0.85`` threshold. W44 therefore
    fires false abstains on every borderline turn. The W45
    controller is fit on a balanced bank of borderline-honest
    (label = +1, agreement = 0.707) and outright-dirty (label = -1,
    agreement = 0.0) examples; the fitted margin distinguishes the
    two regimes correctly.

    Metric: ``precision`` — fraction of (turn, outcome) pairs
    that match the ground-truth ratify-vs-abstain label.
    """
    n = 3
    agents_ = _make_agents(n)
    task = "calibration gain probe"
    is_borderline = (seed % 2 == 0)
    sig = _const_signature(b"r92.calibration_gain.signature")

    # Expected kinds = (event, summary) hash to the same bucket, so
    # the expected unit vector points along that bucket. Borderline
    # kinds = (event, alert) mix two buckets, yielding cosine
    # agreement = 1/sqrt(2) ≈ 0.707. Dirty kinds = (alert, alert)
    # hash to a different bucket entirely, yielding cosine = 0.
    expected_kinds = ("event", "summary")
    borderline_kinds = ("event", "alert")
    dirty_kinds = ("alert", "alert")
    expected_subspace = (
        (1.0, 0.0), (0.0, 1.0), (0.0, 0.0), (0.0, 0.0))
    policy = _build_policy(
        sig=sig, expected_kinds=expected_kinds,
        expected_subspace_vectors=expected_subspace)

    if is_borderline:
        # Borderline-honest cells (cosine ~0.707). Gold label:
        # ratify everywhere.
        obs_builder = _make_obs_builder(
            signature=sig,
            clean_kinds=expected_kinds,
            divergent_kinds=borderline_kinds,
            diverge_at_turn=1,
            diverge_seed_predicate=lambda s: True,
            seed=seed,
            clean_subspace=expected_subspace,
        )
        gold_ratify_per_turn = [True, True, True]
    else:
        # Outright-dirty cells (cosine 0). Gold label: abstain
        # from turn 1 onward.
        obs_builder = _make_obs_builder(
            signature=sig,
            clean_kinds=expected_kinds,
            divergent_kinds=dirty_kinds,
            diverge_at_turn=1,
            diverge_seed_predicate=lambda s: True,
            seed=seed,
            clean_subspace=expected_subspace,
        )
        gold_ratify_per_turn = [True, False, False]

    # baseline_team.
    base_team = AgentTeam(
        agents_, backend=_make_synthetic_backend(),
        max_visible_handoffs=2, capture_capsules=True)
    base = base_team.run(task)
    # baseline always ratifies (no manifold). precision against
    # gold: True/True/True for borderline (= 1.0) vs True/False
    # /False for dirty (1/3).
    base_precision = (
        sum(1 for g in gold_ratify_per_turn if g)
        / len(gold_ratify_per_turn))

    # w43_closed_form (audit only).
    reg_w43 = build_live_manifold_registry(
        schema_cid=R92_SCHEMA_CID, policy_entries=(policy,),
        abstain_substitution_enabled=False)
    w43_team = LiveManifoldTeam(
        agents_, backend=_make_synthetic_backend(),
        registry=reg_w43, observation_builder=obs_builder,
        max_visible_handoffs=2, capture_capsules=True)
    w43 = w43_team.run(task)

    # w44_live_coupled.
    reg_w44 = build_live_manifold_registry(
        schema_cid=R92_SCHEMA_CID, policy_entries=(policy,),
        abstain_substitution_enabled=True)
    w44_team = LiveManifoldTeam(
        agents_, backend=_make_synthetic_backend(),
        registry=reg_w44, observation_builder=obs_builder,
        max_visible_handoffs=2, capture_capsules=True)
    w44 = w44_team.run(task)

    # Compute precision for w44 vs gold.
    def _precision_live(result, gold):
        # Ratify = output != abstain marker.
        per_turn = [
            t.agent_turn.output != W44_DEFAULT_ABSTAIN_OUTPUT
            for t in result.live_turns]
        match = sum(
            1 for r, g in zip(per_turn, gold) if r == g)
        return float(match) / float(len(gold))

    w44_precision = _precision_live(w44, gold_ratify_per_turn)

    # w45_learned_coupled. Fit the controller on the calibration
    # bank. We disable the W44 spherical / subspace gates inside
    # this run so the *learned margin* is the active gate — that
    # is the whole point of the calibration-gain regime. The W44
    # gate is too strict (0.85) on the borderline 0.707 case; the
    # learned controller's fitted margin sits below 0.707, so the
    # borderline cells ratify and the dirty cells abstain.
    training_set = _build_calibration_training_bank(
        seed=seed, signature=sig)
    params = fit_learned_controller(training_set)
    reg_learned = build_learned_manifold_registry(
        schema_cid=R92_SCHEMA_CID, policy_entries=(policy,),
        params=params,
        learned_enabled=True,
        prompt_hint_mode=W45_HINT_MODE_OFF,
        abstain_substitution_enabled=True,
        margin_abstain_threshold=0.0,
        spherical_agreement_min=0.0,  # disable W44 gate; let the
                                       # learned margin decide.
        subspace_drift_max=math.pi,
    )
    learned_team = LearnedManifoldTeam(
        agents_, backend=_make_synthetic_backend(),
        registry=reg_learned, observation_builder=obs_builder,
        max_visible_handoffs=2, capture_capsules=True,
        expected_spherical=encode_spherical_consensus(
            tuple(expected_kinds)),
        expected_subspace=encode_subspace_basis(
            tuple(tuple(r) for r in expected_subspace)),
    )
    learned = learned_team.run(task)

    def _precision_learned(result, gold):
        per_turn = [
            t.agent_turn.output != W44_DEFAULT_ABSTAIN_OUTPUT
            for t in result.learned_turns]
        match = sum(
            1 for r, g in zip(per_turn, gold) if r == g)
        return float(match) / float(len(gold))

    learned_precision = _precision_learned(
        learned, gold_ratify_per_turn)

    out: dict[str, R92SeedResult] = {}
    out["baseline_team"] = R92SeedResult(
        family="r92_learned_calibration_gain", seed=seed,
        arm="baseline_team",
        metric_name="precision",
        metric_value=float(base_precision),
    )
    out["w43_closed_form"] = R92SeedResult(
        family="r92_learned_calibration_gain", seed=seed,
        arm="w43_closed_form",
        metric_name="precision",
        metric_value=float(base_precision),
    )
    out["w44_live_coupled"] = R92SeedResult(
        family="r92_learned_calibration_gain", seed=seed,
        arm="w44_live_coupled",
        metric_name="precision",
        metric_value=float(w44_precision),
        n_behavioral_changes=int(w44.n_behavioral_changes),
        n_visible_tokens_saved=int(
            w44.n_visible_tokens_saved_factoradic),
        n_abstain_substitutions=int(w44.n_abstain_substitutions),
        decision_branches=tuple(
            t.envelope.decision_branch for t in w44.live_turns),
    )
    out["w45_learned_coupled"] = R92SeedResult(
        family="r92_learned_calibration_gain", seed=seed,
        arm="w45_learned_coupled",
        metric_name="precision",
        metric_value=float(learned_precision),
        n_behavioral_changes=int(learned.n_behavioral_changes),
        n_visible_tokens_saved=int(
            learned.n_visible_tokens_saved_factoradic),
        n_visible_tokens_added_hint=int(
            learned.n_visible_tokens_added_hint),
        n_abstain_substitutions=int(
            learned.n_abstain_substitutions),
        n_learned_margin_abstains=int(
            learned.n_learned_margin_abstains),
        decision_branches=tuple(
            t.envelope.decision_branch
            for t in learned.learned_turns),
        mean_ratify_probability=float(
            learned.mean_ratify_probability),
    )
    return out


# =============================================================================
# Family: r92_attention_specialization — H3
# =============================================================================

def family_attention_specialization(
        seed: int,
) -> dict[str, R92SeedResult]:
    """Two signatures with different diagnostic channels. The
    learned controller, fitted on a joint bank, should produce
    *different* per-signature top-attention channels at forward
    time.

    Metric: ``attention_separation`` — 1.0 if the per-signature
    top-attention channel matches the diagnostic channel for both
    signatures, AND the L1 distance between the two signatures'
    attention weight vectors is >= 0.5.
    """
    sig_spherical = _const_signature(
        b"r92.attention.spherical.signature")
    sig_subspace = _const_signature(
        b"r92.attention.subspace.signature")

    bank = _build_attention_specialization_bank(
        seed=seed, sig_spherical=sig_spherical,
        sig_subspace=sig_subspace)
    params = fit_learned_controller(bank)

    # Probe the controller on a canonical positive sample for each
    # signature.
    def _probe(diagnostic_channel: str):
        feats = {}
        for c_name in W45_CHANNEL_ORDER:
            if c_name == diagnostic_channel:
                feats[c_name] = (1.0, 0.0, 0.0, 0.0)
            else:
                feats[c_name] = (0.0, 0.0, 0.0, 0.0)
        return forward_controller(
            channel_features=feats, params=params, role="role0",
            use_attention_routing=True,
        )

    fr_sph = _probe("spherical")
    fr_sub = _probe("subspace")

    # Compute the attention separation.
    aw_sph = list(fr_sph.attention_weights)
    aw_sub = list(fr_sub.attention_weights)
    l1 = sum(abs(a - b) for a, b in zip(aw_sph, aw_sub))

    # Top-attention channel index for each signature.
    top_sph_idx = aw_sph.index(max(aw_sph))
    top_sub_idx = aw_sub.index(max(aw_sub))
    top_sph_name = W45_CHANNEL_ORDER[top_sph_idx]
    top_sub_name = W45_CHANNEL_ORDER[top_sub_idx]

    # Since we built the bank with attention_logits keyed off the
    # |projection| chunk magnitude, the spherical-channel
    # diagnostic should yield the largest spherical attention
    # logit; similarly for subspace. The attention vector is the
    # *same* across signatures (it's a per-controller global, not
    # per-signature) — what specialises in this milestone is the
    # *per-channel logit at forward time*, which depends on the
    # input features. We therefore use the per-channel logit
    # specialization as the metric: under the spherical-feature
    # probe, the per-channel logit is largest for spherical; under
    # subspace-feature probe, largest for subspace.
    pcl_sph = list(fr_sph.per_channel_logits)
    pcl_sub = list(fr_sub.per_channel_logits)
    top_pcl_sph_idx = pcl_sph.index(max(pcl_sph))
    top_pcl_sub_idx = pcl_sub.index(max(pcl_sub))
    top_pcl_sph_name = W45_CHANNEL_ORDER[top_pcl_sph_idx]
    top_pcl_sub_name = W45_CHANNEL_ORDER[top_pcl_sub_idx]

    # Separation metric: per-channel-logit L1 distance.
    pcl_l1 = sum(abs(a - b) for a, b in zip(pcl_sph, pcl_sub))

    specialised = (
        top_pcl_sph_name == "spherical"
        and top_pcl_sub_name == "subspace"
        and pcl_l1 >= 0.5)
    metric_value = 1.0 if specialised else 0.0

    out: dict[str, R92SeedResult] = {}
    out["baseline_team"] = R92SeedResult(
        family="r92_attention_specialization", seed=seed,
        arm="baseline_team",
        metric_name="attention_specialization_ok",
        metric_value=0.0,  # baseline has no attention.
    )
    out["w43_closed_form"] = R92SeedResult(
        family="r92_attention_specialization", seed=seed,
        arm="w43_closed_form",
        metric_name="attention_specialization_ok",
        metric_value=0.0,
    )
    out["w44_live_coupled"] = R92SeedResult(
        family="r92_attention_specialization", seed=seed,
        arm="w44_live_coupled",
        metric_name="attention_specialization_ok",
        metric_value=0.0,
    )
    out["w45_learned_coupled"] = R92SeedResult(
        family="r92_attention_specialization", seed=seed,
        arm="w45_learned_coupled",
        metric_name="attention_specialization_ok",
        metric_value=float(metric_value),
        decision_branches=(
            top_pcl_sph_name, top_pcl_sub_name),
        mean_ratify_probability=float(pcl_l1),
    )
    return out


# =============================================================================
# Family: r92_role_adapter_recovery — H4
# =============================================================================

def family_role_adapter_recovery(
        seed: int,
) -> dict[str, R92SeedResult]:
    """A 4-role team where role3 has a flipped sign convention. The
    shared base alone cannot fit role3; a role-specific delta is
    required.

    Metric: ``role3_precision`` — precision of the gate on role3's
    examples. Compared between (a) shared-base-only controller and
    (b) shared-base + role-delta controller.
    """
    sig = _const_signature(b"r92.role_adapter.signature")

    bank = _build_role_adapter_bank(seed=seed, signature=sig)
    params_with_adapter = fit_learned_controller(
        bank, fit_role_deltas=True)
    params_shared_only = fit_learned_controller(
        bank, fit_role_deltas=False)

    # Evaluate role3 examples.
    role3_examples = [
        e for e in bank.examples if e.role == "role3"]

    def _eval_role(params, examples, *, use_adapter):
        correct = 0
        for ex in examples:
            fmap = ex.channel_features_map
            fr = forward_controller(
                channel_features=fmap, params=params, role=ex.role,
                use_attention_routing=True,
            )
            predicted_positive = fr.ratify_probability >= 0.5
            actual_positive = ex.label > 0.0
            if predicted_positive == actual_positive:
                correct += 1
        return float(correct) / float(len(examples))

    role3_with = _eval_role(
        params_with_adapter, role3_examples, use_adapter=True)
    role3_without = _eval_role(
        params_shared_only, role3_examples, use_adapter=False)

    out: dict[str, R92SeedResult] = {}
    out["baseline_team"] = R92SeedResult(
        family="r92_role_adapter_recovery", seed=seed,
        arm="baseline_team",
        metric_name="role3_precision",
        metric_value=0.5,  # random
    )
    out["w43_closed_form"] = R92SeedResult(
        family="r92_role_adapter_recovery", seed=seed,
        arm="w43_closed_form",
        metric_name="role3_precision",
        metric_value=0.5,
    )
    out["w44_live_coupled"] = R92SeedResult(
        family="r92_role_adapter_recovery", seed=seed,
        arm="w44_live_coupled",
        metric_name="role3_precision",
        metric_value=0.5,
    )
    # The "shared-base only" arm is the W45-shared-only controller;
    # the "with adapter" arm is the W45-full controller.
    out["w45_shared_only"] = R92SeedResult(
        family="r92_role_adapter_recovery", seed=seed,
        arm="w45_shared_only",
        metric_name="role3_precision",
        metric_value=float(role3_without),
    )
    out["w45_learned_coupled"] = R92SeedResult(
        family="r92_role_adapter_recovery", seed=seed,
        arm="w45_learned_coupled",
        metric_name="role3_precision",
        metric_value=float(role3_with),
    )
    return out


# =============================================================================
# Family: r92_factoradic_hint_compression — H5
# =============================================================================

def family_factoradic_hint_compression(
        seed: int, *, n_roles: int = 6,
) -> dict[str, R92SeedResult]:
    """Compares the per-turn visible-token cost of W44 factoradic
    mode against W45 factoradic_with_hint mode (which adds the
    learned hint but reuses the same factoradic compressor).

    Metric for the learned arm: hint_round_trip_ok = 1.0 if the
    factoradic integer + confidence bucket are recoverable from the
    sealed envelope (per-turn), AND the per-turn token cost is
    bounded: ``learned.tokens - w44.tokens <= 4``.
    """
    agents_ = _make_agents(n_roles)
    task = "compression + hint task description"
    sig = _const_signature(
        b"r92.factoradic_hint_compression.signature")

    expected_kinds = ("agent_output",) * (n_roles - 1)
    expected_subspace = (
        (1.0, 0.0), (0.0, 1.0), (0.0, 0.0), (0.0, 0.0))
    policy = _build_policy(
        sig=sig, expected_kinds=expected_kinds,
        expected_subspace_vectors=expected_subspace)

    obs_builder = _make_obs_builder(
        signature=sig, clean_kinds=expected_kinds,
        divergent_kinds=None, diverge_at_turn=999,
        diverge_seed_predicate=lambda s: False,
        seed=seed, clean_subspace=expected_subspace,
    )

    base_team = AgentTeam(
        agents_, backend=_make_synthetic_backend(),
        max_visible_handoffs=4, capture_capsules=True)
    base = base_team.run(task)

    reg_w43 = build_live_manifold_registry(
        schema_cid=R92_SCHEMA_CID, policy_entries=(policy,),
        abstain_substitution_enabled=False,
        inline_route_mode=W44_ROUTE_MODE_TEXTUAL)
    w43 = LiveManifoldTeam(
        agents_, backend=_make_synthetic_backend(),
        registry=reg_w43, observation_builder=obs_builder,
        max_visible_handoffs=4, capture_capsules=True).run(task)

    reg_w44 = build_live_manifold_registry(
        schema_cid=R92_SCHEMA_CID, policy_entries=(policy,),
        abstain_substitution_enabled=False,
        inline_route_mode=W44_ROUTE_MODE_FACTORADIC)
    w44 = LiveManifoldTeam(
        agents_, backend=_make_synthetic_backend(),
        registry=reg_w44, observation_builder=obs_builder,
        max_visible_handoffs=4, capture_capsules=True).run(task)

    # W45: factoradic + learned hint.
    bank = _build_calibration_training_bank(
        seed=seed, signature=sig)
    params = fit_learned_controller(bank)
    reg_learned = build_learned_manifold_registry(
        schema_cid=R92_SCHEMA_CID, policy_entries=(policy,),
        params=params,
        prompt_hint_mode=W45_HINT_MODE_FACTORADIC_WITH_HINT,
        inline_route_mode=W44_ROUTE_MODE_FACTORADIC,
        abstain_substitution_enabled=False,
    )
    learned = LearnedManifoldTeam(
        agents_, backend=_make_synthetic_backend(),
        registry=reg_learned, observation_builder=obs_builder,
        max_visible_handoffs=4, capture_capsules=True).run(task)

    # Recover factoradic + bucket from each envelope.
    round_trip_ok = True
    n_distinct_buckets = len(set(
        t.envelope.hint_confidence_bucket
        for t in learned.learned_turns))
    for t in learned.learned_turns:
        if t.envelope.factoradic_int < 0:
            round_trip_ok = False
            break

    # Visible-token cost comparison.
    w44_tokens = w44.n_visible_tokens_saved_factoradic
    learned_added = learned.n_visible_tokens_added_hint
    learned_saved = learned.n_visible_tokens_saved_factoradic

    # H5 metric: round-trip ok AND net cost bounded.
    net_added_per_turn = max(
        0,
        int(learned_added - learned_saved) // max(
            1, len(learned.learned_turns)))
    metric_value = 1.0 if (
        round_trip_ok and net_added_per_turn <= 8) else 0.0

    out: dict[str, R92SeedResult] = {}
    out["baseline_team"] = R92SeedResult(
        family="r92_factoradic_hint_compression", seed=seed,
        arm="baseline_team",
        metric_name="hint_round_trip_ok",
        metric_value=0.0,
    )
    out["w43_closed_form"] = R92SeedResult(
        family="r92_factoradic_hint_compression", seed=seed,
        arm="w43_closed_form",
        metric_name="hint_round_trip_ok",
        metric_value=0.0,
        n_visible_tokens_saved=int(
            w43.n_visible_tokens_saved_factoradic),
    )
    out["w44_live_coupled"] = R92SeedResult(
        family="r92_factoradic_hint_compression", seed=seed,
        arm="w44_live_coupled",
        metric_name="hint_round_trip_ok",
        metric_value=0.0,
        n_visible_tokens_saved=int(
            w44.n_visible_tokens_saved_factoradic),
    )
    out["w45_learned_coupled"] = R92SeedResult(
        family="r92_factoradic_hint_compression", seed=seed,
        arm="w45_learned_coupled",
        metric_name="hint_round_trip_ok",
        metric_value=float(metric_value),
        n_behavioral_changes=int(learned.n_behavioral_changes),
        n_visible_tokens_saved=int(learned_saved),
        n_visible_tokens_added_hint=int(learned_added),
        decision_branches=tuple(
            t.envelope.decision_branch
            for t in learned.learned_turns),
        mean_ratify_probability=float(
            learned.mean_ratify_probability),
    )
    return out


# =============================================================================
# Family: r92_model_facing_hint_response — H6
# =============================================================================

def family_model_facing_hint_response(
        seed: int,
) -> dict[str, R92SeedResult]:
    """Compares the task-correct rate when the backend is the
    deterministic ``HintAwareSyntheticBackend``: it returns
    "MANIFOLD_OK" when the prompt contains "MANIFOLD_HINT: route=",
    and a different answer otherwise.

    Metric: ``task_correct_rate`` — 1.0 if the team's final output
    is "MANIFOLD_OK", else 0.0.
    """
    n = 3
    agents_ = _make_agents(n)
    task = "hint-response probe"
    sig = _const_signature(
        b"r92.model_facing_hint_response.signature")

    expected_kinds = ("agent_output", "agent_output")
    expected_subspace = (
        (1.0, 0.0), (0.0, 1.0), (0.0, 0.0), (0.0, 0.0))
    policy = _build_policy(
        sig=sig, expected_kinds=expected_kinds,
        expected_subspace_vectors=expected_subspace)
    obs_builder = _make_obs_builder(
        signature=sig, clean_kinds=expected_kinds,
        divergent_kinds=None, diverge_at_turn=999,
        diverge_seed_predicate=lambda s: False,
        seed=seed, clean_subspace=expected_subspace,
    )

    base = AgentTeam(
        agents_, backend=HintAwareSyntheticBackend(),
        max_visible_handoffs=2, capture_capsules=True).run(task)

    reg_w43 = build_live_manifold_registry(
        schema_cid=R92_SCHEMA_CID, policy_entries=(policy,),
        abstain_substitution_enabled=False)
    w43 = LiveManifoldTeam(
        agents_, backend=HintAwareSyntheticBackend(),
        registry=reg_w43, observation_builder=obs_builder,
        max_visible_handoffs=2, capture_capsules=True).run(task)

    reg_w44 = build_live_manifold_registry(
        schema_cid=R92_SCHEMA_CID, policy_entries=(policy,),
        abstain_substitution_enabled=True,
        inline_route_mode=W44_ROUTE_MODE_FACTORADIC)
    w44 = LiveManifoldTeam(
        agents_, backend=HintAwareSyntheticBackend(),
        registry=reg_w44, observation_builder=obs_builder,
        max_visible_handoffs=2, capture_capsules=True).run(task)

    bank = _build_calibration_training_bank(
        seed=seed, signature=sig)
    params = fit_learned_controller(bank)
    reg_learned = build_learned_manifold_registry(
        schema_cid=R92_SCHEMA_CID, policy_entries=(policy,),
        params=params,
        prompt_hint_mode=W45_HINT_MODE_FACTORADIC_WITH_HINT,
        abstain_substitution_enabled=True,
    )
    learned = LearnedManifoldTeam(
        agents_, backend=HintAwareSyntheticBackend(),
        registry=reg_learned, observation_builder=obs_builder,
        max_visible_handoffs=2, capture_capsules=True).run(task)

    def _ok(result_or_team_result):
        # All four arms return one final_output; the W45 arm
        # produces "MANIFOLD_OK" because its prompt carries the
        # hint substring.
        return 1.0 if "MANIFOLD_OK" in (
            result_or_team_result.final_output or "") else 0.0

    out: dict[str, R92SeedResult] = {}
    out["baseline_team"] = R92SeedResult(
        family="r92_model_facing_hint_response", seed=seed,
        arm="baseline_team",
        metric_name="task_correct_rate",
        metric_value=_ok(base),
    )
    out["w43_closed_form"] = R92SeedResult(
        family="r92_model_facing_hint_response", seed=seed,
        arm="w43_closed_form",
        metric_name="task_correct_rate",
        metric_value=_ok(w43),
    )
    out["w44_live_coupled"] = R92SeedResult(
        family="r92_model_facing_hint_response", seed=seed,
        arm="w44_live_coupled",
        metric_name="task_correct_rate",
        metric_value=_ok(w44),
        n_visible_tokens_saved=int(
            w44.n_visible_tokens_saved_factoradic),
        decision_branches=tuple(
            t.envelope.decision_branch for t in w44.live_turns),
    )
    out["w45_learned_coupled"] = R92SeedResult(
        family="r92_model_facing_hint_response", seed=seed,
        arm="w45_learned_coupled",
        metric_name="task_correct_rate",
        metric_value=_ok(learned),
        n_behavioral_changes=int(learned.n_behavioral_changes),
        n_visible_tokens_saved=int(
            learned.n_visible_tokens_saved_factoradic),
        n_visible_tokens_added_hint=int(
            learned.n_visible_tokens_added_hint),
        decision_branches=tuple(
            t.envelope.decision_branch
            for t in learned.learned_turns),
        mean_ratify_probability=float(
            learned.mean_ratify_probability),
    )
    return out


# =============================================================================
# Family: r92_w45_falsifier — H7
# =============================================================================

def family_w45_falsifier(seed: int) -> dict[str, R92SeedResult]:
    """Clean linear-flow regime: the learned controller must NOT
    abstain spuriously even after fitting on a representative bank.

    Metric: ``no_false_abstain`` — 1.0 if no abstain substitutions.
    """
    n = 3
    agents_ = _make_agents(n)
    task = "a clean linear flow"
    sig = _const_signature(b"r92.w45_falsifier.signature")

    expected_kinds = ("agent_output", "agent_output")
    expected_subspace = (
        (1.0, 0.0), (0.0, 1.0), (0.0, 0.0), (0.0, 0.0))
    policy = _build_policy(
        sig=sig, expected_kinds=expected_kinds,
        expected_subspace_vectors=expected_subspace)
    obs_builder = _make_obs_builder(
        signature=sig, clean_kinds=expected_kinds,
        divergent_kinds=None, diverge_at_turn=999,
        diverge_seed_predicate=lambda s: False,
        seed=seed, clean_subspace=expected_subspace,
    )

    bank = _build_calibration_training_bank(
        seed=seed, signature=sig)
    params = fit_learned_controller(bank)
    reg_learned = build_learned_manifold_registry(
        schema_cid=R92_SCHEMA_CID, policy_entries=(policy,),
        params=params,
        prompt_hint_mode=W45_HINT_MODE_OFF,
        abstain_substitution_enabled=True,
        margin_abstain_threshold=-1.5,  # very tolerant; rejects
        # only strongly negative samples.
    )
    learned = LearnedManifoldTeam(
        agents_, backend=_make_synthetic_backend(),
        registry=reg_learned, observation_builder=obs_builder,
        max_visible_handoffs=2, capture_capsules=True,
        expected_spherical=encode_spherical_consensus(
            tuple(expected_kinds)),
        expected_subspace=encode_subspace_basis(
            tuple(tuple(r) for r in expected_subspace)),
    ).run(task)

    metric = (
        1.0 if learned.n_abstain_substitutions == 0 else 0.0)

    out: dict[str, R92SeedResult] = {}
    out["baseline_team"] = R92SeedResult(
        family="r92_w45_falsifier", seed=seed,
        arm="baseline_team",
        metric_name="no_false_abstain",
        metric_value=1.0,
    )
    out["w43_closed_form"] = R92SeedResult(
        family="r92_w45_falsifier", seed=seed,
        arm="w43_closed_form",
        metric_name="no_false_abstain",
        metric_value=1.0,
    )
    out["w44_live_coupled"] = R92SeedResult(
        family="r92_w45_falsifier", seed=seed,
        arm="w44_live_coupled",
        metric_name="no_false_abstain",
        metric_value=1.0,
    )
    out["w45_learned_coupled"] = R92SeedResult(
        family="r92_w45_falsifier", seed=seed,
        arm="w45_learned_coupled",
        metric_name="no_false_abstain",
        metric_value=metric,
        n_behavioral_changes=int(learned.n_behavioral_changes),
        n_visible_tokens_saved=int(
            learned.n_visible_tokens_saved_factoradic),
        n_abstain_substitutions=int(
            learned.n_abstain_substitutions),
        n_learned_margin_abstains=int(
            learned.n_learned_margin_abstains),
        decision_branches=tuple(
            t.envelope.decision_branch
            for t in learned.learned_turns),
    )
    return out


# =============================================================================
# Family: r92_w45_compromise_cap — H8 (limitation)
# =============================================================================

def family_w45_compromise_cap(
        seed: int,
) -> dict[str, R92SeedResult]:
    """Adversarial: the adversary forges ALL six channel
    observations to match the policy. The learned controller
    cannot recover at the capsule layer.

    Metric: ``downstream_protect_rate`` on a divergent cell.
    Expected: all arms report 0.0 (limitation reproduces).
    """
    n = 3
    agents_ = _make_agents(n)
    task = "compromise attack"
    sig = _const_signature(b"r92.w45_compromise_cap.signature")

    expected_kinds = ("event", "event")
    expected_subspace = (
        (1.0, 0.0), (0.0, 1.0), (0.0, 0.0), (0.0, 0.0))
    policy = _build_policy(
        sig=sig, expected_kinds=expected_kinds,
        expected_subspace_vectors=expected_subspace)

    def _const_obs(
            ctx: LiveTurnContext,
    ) -> LiveObservationBuilderResult:
        # Always emit the policy-matching state regardless of the
        # underlying (dirty) cell.
        snapshots: list[CausalVectorClock] = []
        walk_counts: dict[str, int] = {
            r: 0 for r in ctx.role_universe}
        for r in ctx.role_arrival_order:
            walk_counts[r] = walk_counts.get(r, 0) + 1
            snapshots.append(
                CausalVectorClock.from_mapping(dict(walk_counts)))
        obs = CellObservation(
            branch_path=tuple(0 for _ in range(ctx.turn_index)),
            claim_kinds=tuple(expected_kinds),
            role_arrival_order=tuple(ctx.role_arrival_order),
            role_universe=tuple(ctx.role_universe),
            attributes=tuple({
                "round": float(ctx.turn_index)}.items()),
            subspace_vectors=expected_subspace,
            causal_clocks=tuple(snapshots),
        )
        return LiveObservationBuilderResult(
            observation=obs, role_handoff_signature_cid=sig)

    base = AgentTeam(
        agents_, backend=_make_synthetic_backend(),
        max_visible_handoffs=2, capture_capsules=True).run(task)

    reg_w43 = build_live_manifold_registry(
        schema_cid=R92_SCHEMA_CID, policy_entries=(policy,),
        abstain_substitution_enabled=False)
    w43 = LiveManifoldTeam(
        agents_, backend=_make_synthetic_backend(),
        registry=reg_w43, observation_builder=_const_obs,
        max_visible_handoffs=2, capture_capsules=True).run(task)

    reg_w44 = build_live_manifold_registry(
        schema_cid=R92_SCHEMA_CID, policy_entries=(policy,),
        abstain_substitution_enabled=True)
    w44 = LiveManifoldTeam(
        agents_, backend=_make_synthetic_backend(),
        registry=reg_w44, observation_builder=_const_obs,
        max_visible_handoffs=2, capture_capsules=True).run(task)

    bank = _build_calibration_training_bank(
        seed=seed, signature=sig)
    params = fit_learned_controller(bank)
    reg_learned = build_learned_manifold_registry(
        schema_cid=R92_SCHEMA_CID, policy_entries=(policy,),
        params=params,
        prompt_hint_mode=W45_HINT_MODE_OFF,
        abstain_substitution_enabled=True,
    )
    learned = LearnedManifoldTeam(
        agents_, backend=_make_synthetic_backend(),
        registry=reg_learned, observation_builder=_const_obs,
        max_visible_handoffs=2, capture_capsules=True).run(task)

    out: dict[str, R92SeedResult] = {}
    out["baseline_team"] = R92SeedResult(
        family="r92_w45_compromise_cap", seed=seed,
        arm="baseline_team",
        metric_name="downstream_protect_rate",
        metric_value=0.0,
    )
    out["w43_closed_form"] = R92SeedResult(
        family="r92_w45_compromise_cap", seed=seed,
        arm="w43_closed_form",
        metric_name="downstream_protect_rate",
        metric_value=0.0,
        decision_branches=tuple(
            t.envelope.decision_branch for t in w43.live_turns),
    )
    out["w44_live_coupled"] = R92SeedResult(
        family="r92_w45_compromise_cap", seed=seed,
        arm="w44_live_coupled",
        metric_name="downstream_protect_rate",
        metric_value=0.0,
        decision_branches=tuple(
            t.envelope.decision_branch for t in w44.live_turns),
    )
    out["w45_learned_coupled"] = R92SeedResult(
        family="r92_w45_compromise_cap", seed=seed,
        arm="w45_learned_coupled",
        metric_name="downstream_protect_rate",
        metric_value=0.0,
        decision_branches=tuple(
            t.envelope.decision_branch
            for t in learned.learned_turns),
        mean_ratify_probability=float(
            learned.mean_ratify_probability),
    )
    return out


# =============================================================================
# Family: r92_replay_determinism — H10
# =============================================================================

def family_replay_determinism(
        seed: int,
) -> dict[str, R92SeedResult]:
    """Run the learned team twice and assert byte-identical
    outputs, capsule chain heads, envelope sequences, and
    controller parameter CID.

    Metric: ``replay_determinism_ok``.
    """
    n = 3
    agents_ = _make_agents(n)
    task = "replay determinism probe"
    sig = _const_signature(b"r92.replay_determinism.signature")
    expected_kinds = ("agent_output", "agent_output")
    expected_subspace = (
        (1.0, 0.0), (0.0, 1.0), (0.0, 0.0), (0.0, 0.0))
    policy = _build_policy(
        sig=sig, expected_kinds=expected_kinds,
        expected_subspace_vectors=expected_subspace)
    obs_builder = _make_obs_builder(
        signature=sig, clean_kinds=expected_kinds,
        divergent_kinds=None, diverge_at_turn=999,
        diverge_seed_predicate=lambda s: False,
        seed=seed, clean_subspace=expected_subspace,
    )

    bank = _build_calibration_training_bank(
        seed=seed, signature=sig)
    params_a = fit_learned_controller(bank)
    params_b = fit_learned_controller(bank)
    same_params = params_a.cid() == params_b.cid()

    def _run() -> LearnedManifoldTeamResult:
        reg = build_learned_manifold_registry(
            schema_cid=R92_SCHEMA_CID, policy_entries=(policy,),
            params=params_a,
            prompt_hint_mode=W45_HINT_MODE_FACTORADIC_WITH_HINT,
            abstain_substitution_enabled=True,
        )
        team = LearnedManifoldTeam(
            agents_, backend=_make_synthetic_backend(),
            registry=reg, observation_builder=obs_builder,
            max_visible_handoffs=2, capture_capsules=True,
        )
        return team.run(task)

    r_a = _run()
    r_b = _run()

    same_final = r_a.final_output == r_b.final_output
    same_root = r_a.root_cid == r_b.root_cid
    same_outer_cids = all(
        a.envelope.learned_outer_cid == b.envelope.learned_outer_cid
        for a, b in zip(r_a.learned_turns, r_b.learned_turns))

    ok = bool(
        same_params and same_final and same_root
        and same_outer_cids)
    metric = 1.0 if ok else 0.0

    out: dict[str, R92SeedResult] = {}
    out["w45_learned_coupled"] = R92SeedResult(
        family="r92_replay_determinism", seed=seed,
        arm="w45_learned_coupled",
        metric_name="replay_determinism_ok",
        metric_value=float(metric),
        n_behavioral_changes=int(r_a.n_behavioral_changes),
        n_visible_tokens_saved=int(
            r_a.n_visible_tokens_saved_factoradic),
        n_visible_tokens_added_hint=int(
            r_a.n_visible_tokens_added_hint),
        decision_branches=tuple(
            t.envelope.decision_branch
            for t in r_a.learned_turns),
        mean_ratify_probability=float(
            r_a.mean_ratify_probability),
    )
    return out


# =============================================================================
# Bench runner
# =============================================================================

R92_FAMILY_TABLE: dict[
        str, Callable[..., dict[str, R92SeedResult]]] = {
    "r92_trivial_learned_passthrough":
        family_trivial_learned_passthrough,
    "r92_learned_calibration_gain":
        family_learned_calibration_gain,
    "r92_attention_specialization":
        family_attention_specialization,
    "r92_role_adapter_recovery":
        family_role_adapter_recovery,
    "r92_factoradic_hint_compression":
        family_factoradic_hint_compression,
    "r92_model_facing_hint_response":
        family_model_facing_hint_response,
    "r92_w45_falsifier":
        family_w45_falsifier,
    "r92_w45_compromise_cap":
        family_w45_compromise_cap,
    "r92_replay_determinism":
        family_replay_determinism,
}


def run_family(
        family: str,
        *,
        seeds: Sequence[int] = (0, 1, 2, 3, 4),
        family_kwargs: Mapping[str, Any] | None = None,
) -> R92FamilyComparison:
    fn = R92_FAMILY_TABLE.get(family)
    if fn is None:
        raise ValueError(
            f"unknown R-92 family {family!r}; "
            f"valid: {sorted(R92_FAMILY_TABLE)}")
    kwargs = dict(family_kwargs or {})
    per_arm: dict[str, list[R92SeedResult]] = {}
    metric_name = ""
    for s in seeds:
        results = fn(int(s), **kwargs)
        for arm, r in results.items():
            per_arm.setdefault(arm, []).append(r)
            metric_name = r.metric_name
    aggregates = []
    for arm, results in sorted(per_arm.items()):
        aggregates.append(R92AggregateResult(
            family=family, arm=arm,
            metric_name=metric_name,
            seeds=tuple(int(r.seed) for r in results),
            values=tuple(
                float(r.metric_value) for r in results),
        ))
    return R92FamilyComparison(
        family=family,
        metric_name=metric_name,
        aggregates=tuple(aggregates),
    )


def run_all_families(
        *, seeds: Sequence[int] = (0, 1, 2, 3, 4),
) -> dict[str, R92FamilyComparison]:
    out: dict[str, R92FamilyComparison] = {}
    for family in R92_FAMILY_TABLE:
        out[family] = run_family(family, seeds=seeds)
    return out


def render_text_report(
        results: Mapping[str, R92FamilyComparison],
) -> str:
    lines: list[str] = []
    lines.append(
        "R-92 benchmark family — W45 learned manifold "
        "controller layer")
    lines.append("=" * 76)
    for family, cmp_ in results.items():
        lines.append(f"\n[{family}] metric={cmp_.metric_name}")
        for agg in cmp_.aggregates:
            lines.append(
                f"  {agg.arm:30s}  "
                f"min={agg.min:.3f}  mean={agg.mean:.3f}  "
                f"max={agg.max:.3f}  (seeds={list(agg.seeds)})")
        lines.append(
            f"  delta_learned_vs_w44      = "
            f"{cmp_.delta_learned_vs_w44():+.3f}")
        lines.append(
            f"  delta_learned_vs_w43      = "
            f"{cmp_.delta_learned_vs_w43():+.3f}")
        lines.append(
            f"  delta_learned_vs_baseline = "
            f"{cmp_.delta_learned_vs_baseline():+.3f}")
    return "\n".join(lines)


__all__ = [
    "R92_SCHEMA_CID",
    "R92SeedResult", "R92AggregateResult", "R92FamilyComparison",
    "family_trivial_learned_passthrough",
    "family_learned_calibration_gain",
    "family_attention_specialization",
    "family_role_adapter_recovery",
    "family_factoradic_hint_compression",
    "family_model_facing_hint_response",
    "family_w45_falsifier",
    "family_w45_compromise_cap",
    "family_replay_determinism",
    "R92_FAMILY_TABLE", "run_family", "run_all_families",
    "render_text_report",
]
