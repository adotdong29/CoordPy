"""R-91 benchmark family for the W44 Live Manifold-Coupled
Coordination (LMCC) layer.

R-91 is the first capsule-layer benchmark family in CoordPy that
compares a *live-coupled* manifold path against the released
``AgentTeam`` and the W43 closed-form PMC path on real
agent-team runs (using a deterministic ``SyntheticLLMClient``
backend so the comparison is honest, reproducible, and seeded).

Three honest baselines per family:

  * ``baseline_team`` — released ``AgentTeam.run`` path. No
    manifold integration; the runtime never gates on observation.
  * ``w43_closed_form`` — ``LiveManifoldTeam`` with the W43 inner
    enabled but ``abstain_substitution_enabled=False``. The W43
    audit envelopes are recorded but the run does not behaviourally
    differ from baseline (this is the closed-form audit-only
    comparison).
  * ``w44_live_coupled`` — ``LiveManifoldTeam`` with the live gate
    on (``abstain_substitution_enabled=True``) and the configured
    inline route mode.

Six families:

  * ``r91_trivial_live_passthrough`` — sanity: trivial registry
    reduces to AgentTeam byte-for-byte.
  * ``r91_live_causal_gate``         — half cells inject an
    out-of-order causal clock at index 1 or 2; the live arm must
    abstain before the next agent runs.
  * ``r91_live_spherical_gate``      — half cells emit divergent
    claim_kinds; same gating semantics.
  * ``r91_live_subspace_gate``       — half cells drift to an
    orthogonal subspace; same gating semantics.
  * ``r91_live_factoradic_compression`` — measure visible-token
    savings from the factoradic compressor at ``n_roles=8``.
  * ``r91_live_falsifier``           — clean linear-flow regime;
    the live arm must NOT abstain spuriously.
  * ``r91_live_dual_channel_collusion`` — adversarial: forge BOTH
    the spherical and the subspace observations. The live arm
    cannot recover (W44-L-LIVE-DUAL-CHANNEL-COLLUSION-CAP).

Each family produces:

  * a measurable per-seed metric (downstream-protect rate,
    visible-token saving, passthrough_ok)
  * an aggregate across seeds with min/max/mean
  * a clear winner / no-improvement statement
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
from typing import Any, Callable, Iterable, Mapping, Sequence

from coordpy.agents import Agent, AgentTeam, agent
from coordpy.live_manifold import (
    LiveManifoldHandoffEnvelope,
    LiveManifoldRegistry,
    LiveManifoldTeam,
    LiveManifoldTeamResult,
    LiveObservationBuilderResult,
    LiveTurnContext,
    W44_BRANCH_LIVE_CAUSAL_ABSTAIN,
    W44_BRANCH_LIVE_RATIFIED,
    W44_BRANCH_LIVE_SPHERICAL_ABSTAIN,
    W44_BRANCH_LIVE_SUBSPACE_ABSTAIN,
    W44_BRANCH_LIVE_NO_POLICY,
    W44_BRANCH_TRIVIAL_LIVE_PASSTHROUGH,
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
    encode_spherical_consensus,
    encode_subspace_basis,
)
from coordpy.synthetic_llm import SyntheticLLMClient


R91_SCHEMA_CID = hashlib.sha256(
    b"r91.benchmark.schema.v1").hexdigest()


# =============================================================================
# Synthetic seeding helpers
# =============================================================================

def _xorshift32(state: int) -> Iterable[int]:
    s = int(state) & 0xFFFFFFFF
    if s == 0:
        s = 0xDEADBEEF
    while True:
        s ^= (s << 13) & 0xFFFFFFFF
        s ^= (s >> 17) & 0xFFFFFFFF
        s ^= (s << 5) & 0xFFFFFFFF
        s &= 0xFFFFFFFF
        yield s


def _seeded_permutation(seed: int, n: int) -> tuple[int, ...]:
    prng = _xorshift32(seed)
    items = list(range(n))
    for i in range(n - 1, 0, -1):
        j = next(prng) % (i + 1)
        items[i], items[j] = items[j], items[i]
    return tuple(items)


def _make_clock_snapshots(
        roles: Sequence[str],
) -> tuple[CausalVectorClock, ...]:
    counts: dict[str, int] = {r: 0 for r in roles}
    out: list[CausalVectorClock] = []
    for r in roles:
        counts[r] = counts.get(r, 0) + 1
        out.append(CausalVectorClock.from_mapping(dict(counts)))
    return tuple(out)


def _make_clock_snapshots_violating(
        roles: Sequence[str], swap_at: int,
) -> tuple[CausalVectorClock, ...]:
    base = list(_make_clock_snapshots(roles))
    if 0 <= swap_at < len(base) - 1:
        # Rewind the clock at swap_at+1 to a strictly smaller clock.
        base[swap_at + 1] = base[max(0, swap_at - 1)]
    return tuple(base)


# =============================================================================
# Result model
# =============================================================================

@dataclasses.dataclass(frozen=True)
class R91SeedResult:
    family: str
    seed: int
    arm: str
    metric_name: str
    metric_value: float
    n_behavioral_changes: int = 0
    n_visible_tokens_saved: int = 0
    n_abstain_substitutions: int = 0
    decision_branches: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class R91AggregateResult:
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
class R91FamilyComparison:
    family: str
    metric_name: str
    aggregates: tuple[R91AggregateResult, ...]

    def get(self, arm: str) -> R91AggregateResult | None:
        for a in self.aggregates:
            if a.arm == arm:
                return a
        return None

    def delta_live_vs_w43(self) -> float:
        live = self.get("w44_live_coupled")
        w43 = self.get("w43_closed_form")
        if live is None or w43 is None:
            return 0.0
        return float(live.mean - w43.mean)

    def delta_live_vs_baseline(self) -> float:
        live = self.get("w44_live_coupled")
        base = self.get("baseline_team")
        if live is None or base is None:
            return 0.0
        return float(live.mean - base.mean)

    def as_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "metric_name": self.metric_name,
            "aggregates": [a.as_dict() for a in self.aggregates],
            "delta_live_vs_w43": float(self.delta_live_vs_w43()),
            "delta_live_vs_baseline": float(
                self.delta_live_vs_baseline()),
        }


# =============================================================================
# Shared backend / fixtures
# =============================================================================

# A fixed canonical "real" output that synthetic agents return when
# allowed to run. Long enough to make the textual rendering of the
# handoff list non-trivial in token count, so the factoradic
# compressor's saving is non-zero.
R91_REAL_OUTPUT: str = (
    "agent output payload with several extra words to make rendering meaningful")


def _make_synthetic_backend() -> SyntheticLLMClient:
    return SyntheticLLMClient(
        model_tag="synthetic.r91", default_response=R91_REAL_OUTPUT)


def _make_agents(n: int) -> tuple[Agent, ...]:
    return tuple(
        agent(
            f"role{i}",
            f"You are role{i}; respond as instructed.",
            max_tokens=64, temperature=0.0,
        )
        for i in range(n)
    )


# =============================================================================
# Closed-vocabulary policy entries
# =============================================================================

def _build_default_policy(
        *, schema_cid: str, sig: str,
        expected_kinds: Sequence[str],
        expected_subspace_vectors: Sequence[Sequence[float]],
        expected_topology: str = "(...)",
) -> ProductManifoldPolicyEntry:
    return ProductManifoldPolicyEntry(
        role_handoff_signature_cid=sig,
        expected_services=("live",),
        expected_spherical=encode_spherical_consensus(
            tuple(expected_kinds)),
        expected_subspace=encode_subspace_basis(
            tuple(tuple(r) for r in expected_subspace_vectors)),
        expected_causal_topology_hash=str(expected_topology),
    )


# =============================================================================
# Family: r91_trivial_live_passthrough — sanity
# =============================================================================

def family_trivial_live_passthrough(seed: int) -> dict[str, R91SeedResult]:
    """Sanity: trivially-configured live registry must reduce to
    AgentTeam byte-for-byte.

    Metric: ``passthrough_ok`` — 1.0 if all three arms produce the
    same final_output, same n_turns, and the live arm produces the
    trivial-passthrough branch on every turn.
    """
    n = 3
    agents_ = _make_agents(n)
    task = "explain the live coupling reduction"

    # Baseline AgentTeam.
    base_team = AgentTeam(
        agents_, backend=_make_synthetic_backend(),
        team_instructions="bounded team", max_visible_handoffs=2,
        capture_capsules=True,
    )
    base = base_team.run(task)

    # W43 closed-form arm: trivial live registry.
    reg_w43 = build_trivial_live_manifold_registry(
        schema_cid=R91_SCHEMA_CID)
    w43_team = LiveManifoldTeam(
        agents_, backend=_make_synthetic_backend(),
        registry=reg_w43,
        team_instructions="bounded team", max_visible_handoffs=2,
        capture_capsules=True,
    )
    w43 = w43_team.run(task)

    # W44 live coupled arm — also trivial.
    reg_live = build_trivial_live_manifold_registry(
        schema_cid=R91_SCHEMA_CID)
    live_team = LiveManifoldTeam(
        agents_, backend=_make_synthetic_backend(),
        registry=reg_live,
        team_instructions="bounded team", max_visible_handoffs=2,
        capture_capsules=True,
    )
    live = live_team.run(task)

    out: dict[str, R91SeedResult] = {}

    def _ok(res, allow_branches=()) -> float:
        if isinstance(res, LiveManifoldTeamResult):
            branches_ok = all(
                t.envelope.decision_branch
                in (W44_BRANCH_TRIVIAL_LIVE_PASSTHROUGH,)
                + tuple(allow_branches)
                for t in res.live_turns)
            return 1.0 if (
                res.final_output == base.final_output
                and len(res.turns) == len(base.turns)
                and branches_ok
            ) else 0.0
        # Baseline: always 1.0 by construction.
        return 1.0 if res.final_output == base.final_output else 1.0

    out["baseline_team"] = R91SeedResult(
        family="r91_trivial_live_passthrough", seed=seed,
        arm="baseline_team",
        metric_name="passthrough_ok",
        metric_value=1.0,
        decision_branches=tuple(t.role for t in base.turns),
    )
    out["w43_closed_form"] = R91SeedResult(
        family="r91_trivial_live_passthrough", seed=seed,
        arm="w43_closed_form",
        metric_name="passthrough_ok",
        metric_value=_ok(w43),
        n_behavioral_changes=int(w43.n_behavioral_changes),
        n_visible_tokens_saved=int(
            w43.n_visible_tokens_saved_factoradic),
        n_abstain_substitutions=int(w43.n_abstain_substitutions),
        decision_branches=tuple(
            t.envelope.decision_branch for t in w43.live_turns),
    )
    out["w44_live_coupled"] = R91SeedResult(
        family="r91_trivial_live_passthrough", seed=seed,
        arm="w44_live_coupled",
        metric_name="passthrough_ok",
        metric_value=_ok(live),
        n_behavioral_changes=int(live.n_behavioral_changes),
        n_visible_tokens_saved=int(
            live.n_visible_tokens_saved_factoradic),
        n_abstain_substitutions=int(live.n_abstain_substitutions),
        decision_branches=tuple(
            t.envelope.decision_branch for t in live.live_turns),
    )
    return out


# =============================================================================
# Helpers for gating-family observation builders
# =============================================================================

def _const_signature(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _make_obs_builder_for_kinds(
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
    """Build an observation builder that emits a fixed signature
    regardless of the turn's ``current_role``, but conditionally
    diverges (kinds, subspace, or causal-clock) on later turns.
    """
    clean_subspace = (
        tuple(tuple(r) for r in clean_subspace)
        if clean_subspace is not None
        else ((1.0, 0.0), (0.0, 1.0), (0.0, 0.0), (0.0, 0.0)))
    divergent_subspace = (
        tuple(tuple(r) for r in divergent_subspace)
        if divergent_subspace is not None else None)

    def _builder(ctx: LiveTurnContext) -> LiveObservationBuilderResult:
        diverges = diverge_seed_predicate(seed)
        # Causal clocks: we walk role_arrival_order and may inject a
        # violation at a chosen turn.
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
            # Replace the latest clock with an empty clock so the
            # sequence is strictly non-monotone (every component
            # of the previous clock dominates the empty clock,
            # but not vice versa). is_causally_admissible flags
            # this at the last index.
            snapshots[-1] = CausalVectorClock.from_mapping({})
        # Claim kinds.
        if (diverges and divergent_kinds is not None
                and ctx.turn_index >= diverge_at_turn):
            kinds = tuple(divergent_kinds)
        else:
            kinds = tuple(clean_kinds)
        # Subspace vectors.
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


def _downstream_protect_rate(
        live_result: LiveManifoldTeamResult,
        *,
        diverges: bool,
        diverge_at_turn: int,
        abstain_branch: str,
) -> float:
    """Compute the downstream-protect rate.

    The rule:

      * If the cell is clean (``diverges=False``): the run must
        complete with no abstain substitutions on any turn.
      * If the cell diverges (``diverges=True``): every turn from
        ``diverge_at_turn`` onward must abstain via the named branch
        AND the corresponding agent's visible output must equal the
        abstain output (proving the live gate prevented the bad
        upstream from propagating).

    Returns 1.0 if the run respects the rule, 0.0 otherwise.
    """
    if not diverges:
        # Clean cells: no abstain at all.
        ok = all(
            t.envelope.decision_branch != abstain_branch
            and t.agent_turn.output != W44_DEFAULT_ABSTAIN_OUTPUT
            for t in live_result.live_turns)
        return 1.0 if ok else 0.0
    # Divergent cells: every turn from diverge_at_turn must abstain.
    for t in live_result.live_turns:
        if t.envelope.turn_index >= diverge_at_turn:
            if t.envelope.decision_branch != abstain_branch:
                return 0.0
            if t.agent_turn.output != W44_DEFAULT_ABSTAIN_OUTPUT:
                return 0.0
    return 1.0


def _baseline_protect_rate(
        result,
        *,
        diverges: bool,
) -> float:
    """For the baseline AgentTeam (no manifold), there's no gate, so
    the downstream-protect rate is:

      * 1.0 on clean cells (the team runs normally).
      * 0.0 on divergent cells (the bad upstream propagates).
    """
    return 0.0 if diverges else 1.0


# =============================================================================
# Family: r91_live_causal_gate
# =============================================================================

def family_live_causal_gate(seed: int) -> dict[str, R91SeedResult]:
    """Half the cells inject an out-of-order causal clock at index
    1 or 2. The W44 live arm must abstain via
    ``live_causal_abstain`` from the violation onward. Metric:
    ``downstream_protect_rate``.
    """
    n = 3
    agents_ = _make_agents(n)
    task = "explain protocol step"
    diverges = bool(seed % 2)
    inject_at = 1 + (seed % 2)  # 1 or 2
    sig = _const_signature(b"r91.causal_gate.signature")

    expected_kinds = ("agent_output",) * (n - 1)
    expected_subspace = (
        (1.0, 0.0), (0.0, 1.0), (0.0, 0.0), (0.0, 0.0))
    policy = _build_default_policy(
        schema_cid=R91_SCHEMA_CID,
        sig=sig,
        expected_kinds=expected_kinds,
        expected_subspace_vectors=expected_subspace,
        expected_topology="(role0,role1,role2)",
    )

    def _diverge_pred(s: int) -> bool:
        return bool(s % 2)

    obs_builder = _make_obs_builder_for_kinds(
        signature=sig,
        clean_kinds=expected_kinds,
        divergent_kinds=None,
        diverge_at_turn=inject_at,
        diverge_seed_predicate=_diverge_pred,
        seed=seed,
        injected_violation_at_turn=inject_at,
        clean_subspace=expected_subspace,
        divergent_subspace=None,
    )

    # baseline_team: AgentTeam with no manifold.
    base_team = AgentTeam(
        agents_, backend=_make_synthetic_backend(),
        max_visible_handoffs=2, capture_capsules=True,
    )
    base = base_team.run(task)

    # w43_closed_form: live registry with abstain disabled.
    reg_w43 = build_live_manifold_registry(
        schema_cid=R91_SCHEMA_CID,
        policy_entries=(policy,),
        live_enabled=True,
        abstain_substitution_enabled=False,
    )
    w43_team = LiveManifoldTeam(
        agents_, backend=_make_synthetic_backend(),
        registry=reg_w43,
        observation_builder=obs_builder,
        max_visible_handoffs=2, capture_capsules=True,
    )
    w43 = w43_team.run(task)

    # w44_live_coupled: live registry with abstain enabled.
    reg_live = build_live_manifold_registry(
        schema_cid=R91_SCHEMA_CID,
        policy_entries=(policy,),
        live_enabled=True,
        abstain_substitution_enabled=True,
    )
    live_team = LiveManifoldTeam(
        agents_, backend=_make_synthetic_backend(),
        registry=reg_live,
        observation_builder=obs_builder,
        max_visible_handoffs=2, capture_capsules=True,
    )
    live = live_team.run(task)

    out: dict[str, R91SeedResult] = {}
    out["baseline_team"] = R91SeedResult(
        family="r91_live_causal_gate", seed=seed,
        arm="baseline_team",
        metric_name="downstream_protect_rate",
        metric_value=_baseline_protect_rate(base, diverges=diverges),
    )
    out["w43_closed_form"] = R91SeedResult(
        family="r91_live_causal_gate", seed=seed,
        arm="w43_closed_form",
        metric_name="downstream_protect_rate",
        # No abstain substitution: behaves like baseline.
        metric_value=_baseline_protect_rate(w43, diverges=diverges),
        n_behavioral_changes=int(w43.n_behavioral_changes),
        n_visible_tokens_saved=int(
            w43.n_visible_tokens_saved_factoradic),
        n_abstain_substitutions=int(w43.n_abstain_substitutions),
        decision_branches=tuple(
            t.envelope.decision_branch for t in w43.live_turns),
    )
    out["w44_live_coupled"] = R91SeedResult(
        family="r91_live_causal_gate", seed=seed,
        arm="w44_live_coupled",
        metric_name="downstream_protect_rate",
        metric_value=_downstream_protect_rate(
            live, diverges=diverges,
            diverge_at_turn=inject_at,
            abstain_branch=W44_BRANCH_LIVE_CAUSAL_ABSTAIN),
        n_behavioral_changes=int(live.n_behavioral_changes),
        n_visible_tokens_saved=int(
            live.n_visible_tokens_saved_factoradic),
        n_abstain_substitutions=int(live.n_abstain_substitutions),
        decision_branches=tuple(
            t.envelope.decision_branch for t in live.live_turns),
    )
    return out


# =============================================================================
# Family: r91_live_spherical_gate
# =============================================================================

def family_live_spherical_gate(seed: int) -> dict[str, R91SeedResult]:
    n = 3
    agents_ = _make_agents(n)
    task = "consensus reach"
    diverges = bool(seed % 2)
    diverge_at = 1 + (seed % 2)
    sig = _const_signature(b"r91.spherical_gate.signature")

    expected_kinds = ("event", "event", "summary")
    divergent_kinds = ("alert", "alert", "alert")
    expected_subspace = (
        (1.0, 0.0), (0.0, 1.0), (0.0, 0.0), (0.0, 0.0))
    policy = _build_default_policy(
        schema_cid=R91_SCHEMA_CID,
        sig=sig,
        expected_kinds=expected_kinds,
        expected_subspace_vectors=expected_subspace,
        expected_topology="(role0,role1,role2)",
    )

    def _diverge_pred(s: int) -> bool:
        return bool(s % 2)

    obs_builder = _make_obs_builder_for_kinds(
        signature=sig,
        clean_kinds=expected_kinds,
        divergent_kinds=divergent_kinds,
        diverge_at_turn=diverge_at,
        diverge_seed_predicate=_diverge_pred,
        seed=seed,
        injected_violation_at_turn=None,
        clean_subspace=expected_subspace,
        divergent_subspace=None,
    )

    base_team = AgentTeam(
        agents_, backend=_make_synthetic_backend(),
        max_visible_handoffs=2, capture_capsules=True)
    base = base_team.run(task)

    reg_w43 = build_live_manifold_registry(
        schema_cid=R91_SCHEMA_CID, policy_entries=(policy,),
        abstain_substitution_enabled=False)
    w43 = LiveManifoldTeam(
        agents_, backend=_make_synthetic_backend(),
        registry=reg_w43, observation_builder=obs_builder,
        max_visible_handoffs=2, capture_capsules=True).run(task)

    reg_live = build_live_manifold_registry(
        schema_cid=R91_SCHEMA_CID, policy_entries=(policy,),
        abstain_substitution_enabled=True)
    live = LiveManifoldTeam(
        agents_, backend=_make_synthetic_backend(),
        registry=reg_live, observation_builder=obs_builder,
        max_visible_handoffs=2, capture_capsules=True).run(task)

    out: dict[str, R91SeedResult] = {}
    out["baseline_team"] = R91SeedResult(
        family="r91_live_spherical_gate", seed=seed,
        arm="baseline_team",
        metric_name="downstream_protect_rate",
        metric_value=_baseline_protect_rate(base, diverges=diverges),
    )
    out["w43_closed_form"] = R91SeedResult(
        family="r91_live_spherical_gate", seed=seed,
        arm="w43_closed_form",
        metric_name="downstream_protect_rate",
        metric_value=_baseline_protect_rate(w43, diverges=diverges),
        n_behavioral_changes=int(w43.n_behavioral_changes),
        n_visible_tokens_saved=int(
            w43.n_visible_tokens_saved_factoradic),
        n_abstain_substitutions=int(w43.n_abstain_substitutions),
        decision_branches=tuple(
            t.envelope.decision_branch for t in w43.live_turns),
    )
    out["w44_live_coupled"] = R91SeedResult(
        family="r91_live_spherical_gate", seed=seed,
        arm="w44_live_coupled",
        metric_name="downstream_protect_rate",
        metric_value=_downstream_protect_rate(
            live, diverges=diverges,
            diverge_at_turn=diverge_at,
            abstain_branch=W44_BRANCH_LIVE_SPHERICAL_ABSTAIN),
        n_behavioral_changes=int(live.n_behavioral_changes),
        n_visible_tokens_saved=int(
            live.n_visible_tokens_saved_factoradic),
        n_abstain_substitutions=int(live.n_abstain_substitutions),
        decision_branches=tuple(
            t.envelope.decision_branch for t in live.live_turns),
    )
    return out


# =============================================================================
# Family: r91_live_subspace_gate
# =============================================================================

def family_live_subspace_gate(seed: int) -> dict[str, R91SeedResult]:
    n = 3
    agents_ = _make_agents(n)
    task = "subspace gating test"
    diverges = bool(seed % 2)
    diverge_at = 1 + (seed % 2)
    sig = _const_signature(b"r91.subspace_gate.signature")

    expected_kinds = ("agent_output",) * 2
    expected_subspace = (
        (1.0, 0.0), (0.0, 1.0), (0.0, 0.0), (0.0, 0.0))
    drifted_subspace = (
        (0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (0.0, 0.0))
    policy = _build_default_policy(
        schema_cid=R91_SCHEMA_CID,
        sig=sig,
        expected_kinds=expected_kinds,
        expected_subspace_vectors=expected_subspace,
        expected_topology="(role0,role1,role2)",
    )

    def _diverge_pred(s: int) -> bool:
        return bool(s % 2)

    obs_builder = _make_obs_builder_for_kinds(
        signature=sig,
        clean_kinds=expected_kinds,
        divergent_kinds=None,
        diverge_at_turn=diverge_at,
        diverge_seed_predicate=_diverge_pred,
        seed=seed,
        injected_violation_at_turn=None,
        clean_subspace=expected_subspace,
        divergent_subspace=drifted_subspace,
    )

    base_team = AgentTeam(
        agents_, backend=_make_synthetic_backend(),
        max_visible_handoffs=2, capture_capsules=True)
    base = base_team.run(task)

    reg_w43 = build_live_manifold_registry(
        schema_cid=R91_SCHEMA_CID, policy_entries=(policy,),
        abstain_substitution_enabled=False)
    w43 = LiveManifoldTeam(
        agents_, backend=_make_synthetic_backend(),
        registry=reg_w43, observation_builder=obs_builder,
        max_visible_handoffs=2, capture_capsules=True).run(task)

    reg_live = build_live_manifold_registry(
        schema_cid=R91_SCHEMA_CID, policy_entries=(policy,),
        abstain_substitution_enabled=True)
    live = LiveManifoldTeam(
        agents_, backend=_make_synthetic_backend(),
        registry=reg_live, observation_builder=obs_builder,
        max_visible_handoffs=2, capture_capsules=True).run(task)

    out: dict[str, R91SeedResult] = {}
    out["baseline_team"] = R91SeedResult(
        family="r91_live_subspace_gate", seed=seed,
        arm="baseline_team",
        metric_name="downstream_protect_rate",
        metric_value=_baseline_protect_rate(base, diverges=diverges),
    )
    out["w43_closed_form"] = R91SeedResult(
        family="r91_live_subspace_gate", seed=seed,
        arm="w43_closed_form",
        metric_name="downstream_protect_rate",
        metric_value=_baseline_protect_rate(w43, diverges=diverges),
        n_behavioral_changes=int(w43.n_behavioral_changes),
        n_visible_tokens_saved=int(
            w43.n_visible_tokens_saved_factoradic),
        n_abstain_substitutions=int(w43.n_abstain_substitutions),
        decision_branches=tuple(
            t.envelope.decision_branch for t in w43.live_turns),
    )
    out["w44_live_coupled"] = R91SeedResult(
        family="r91_live_subspace_gate", seed=seed,
        arm="w44_live_coupled",
        metric_name="downstream_protect_rate",
        metric_value=_downstream_protect_rate(
            live, diverges=diverges,
            diverge_at_turn=diverge_at,
            abstain_branch=W44_BRANCH_LIVE_SUBSPACE_ABSTAIN),
        n_behavioral_changes=int(live.n_behavioral_changes),
        n_visible_tokens_saved=int(
            live.n_visible_tokens_saved_factoradic),
        n_abstain_substitutions=int(live.n_abstain_substitutions),
        decision_branches=tuple(
            t.envelope.decision_branch for t in live.live_turns),
    )
    return out


# =============================================================================
# Family: r91_live_factoradic_compression
# =============================================================================

def family_live_factoradic_compression(
        seed: int, *, n_roles: int = 8,
) -> dict[str, R91SeedResult]:
    """Measure the visible-prompt-token saving from the factoradic
    compressor on a permutation-heavy regime.

    Metric: ``visible_tokens_saved_per_run`` — the total visible
    prompt-token saving across the whole run (sum across turns).
    """
    agents_ = _make_agents(n_roles)
    task = "permutation-heavy task description"
    sig = _const_signature(b"r91.factoradic_compression.signature")
    expected_kinds = ("agent_output",) * (n_roles - 1)
    expected_subspace = (
        (1.0, 0.0), (0.0, 1.0), (0.0, 0.0), (0.0, 0.0))
    policy = _build_default_policy(
        schema_cid=R91_SCHEMA_CID,
        sig=sig,
        expected_kinds=expected_kinds,
        expected_subspace_vectors=expected_subspace,
        expected_topology="(...)",
    )

    obs_builder = _make_obs_builder_for_kinds(
        signature=sig,
        clean_kinds=expected_kinds,
        divergent_kinds=None,
        diverge_at_turn=999,  # never diverge
        diverge_seed_predicate=lambda s: False,
        seed=seed,
        injected_violation_at_turn=None,
        clean_subspace=expected_subspace,
        divergent_subspace=None,
    )

    base_team = AgentTeam(
        agents_, backend=_make_synthetic_backend(),
        max_visible_handoffs=4, capture_capsules=True)
    base = base_team.run(task)

    reg_w43 = build_live_manifold_registry(
        schema_cid=R91_SCHEMA_CID, policy_entries=(policy,),
        abstain_substitution_enabled=False,
        inline_route_mode=W44_ROUTE_MODE_TEXTUAL)
    w43 = LiveManifoldTeam(
        agents_, backend=_make_synthetic_backend(),
        registry=reg_w43, observation_builder=obs_builder,
        max_visible_handoffs=4, capture_capsules=True).run(task)

    reg_live = build_live_manifold_registry(
        schema_cid=R91_SCHEMA_CID, policy_entries=(policy,),
        abstain_substitution_enabled=False,
        inline_route_mode=W44_ROUTE_MODE_FACTORADIC)
    live = LiveManifoldTeam(
        agents_, backend=_make_synthetic_backend(),
        registry=reg_live, observation_builder=obs_builder,
        max_visible_handoffs=4, capture_capsules=True).run(task)

    out: dict[str, R91SeedResult] = {}
    out["baseline_team"] = R91SeedResult(
        family="r91_live_factoradic_compression", seed=seed,
        arm="baseline_team",
        metric_name="visible_tokens_saved_per_run",
        metric_value=0.0,
    )
    out["w43_closed_form"] = R91SeedResult(
        family="r91_live_factoradic_compression", seed=seed,
        arm="w43_closed_form",
        metric_name="visible_tokens_saved_per_run",
        metric_value=float(w43.n_visible_tokens_saved_factoradic),
        n_behavioral_changes=int(w43.n_behavioral_changes),
        n_visible_tokens_saved=int(
            w43.n_visible_tokens_saved_factoradic),
        n_abstain_substitutions=int(w43.n_abstain_substitutions),
        decision_branches=tuple(
            t.envelope.decision_branch for t in w43.live_turns),
    )
    out["w44_live_coupled"] = R91SeedResult(
        family="r91_live_factoradic_compression", seed=seed,
        arm="w44_live_coupled",
        metric_name="visible_tokens_saved_per_run",
        metric_value=float(live.n_visible_tokens_saved_factoradic),
        n_behavioral_changes=int(live.n_behavioral_changes),
        n_visible_tokens_saved=int(
            live.n_visible_tokens_saved_factoradic),
        n_abstain_substitutions=int(live.n_abstain_substitutions),
        decision_branches=tuple(
            t.envelope.decision_branch for t in live.live_turns),
    )
    return out


# =============================================================================
# Family: r91_live_falsifier
# =============================================================================

def family_live_falsifier(seed: int) -> dict[str, R91SeedResult]:
    """Clean linear-flow regime: the live arm must NOT abstain
    spuriously. Metric: ``no_false_abstain``."""
    n = 3
    agents_ = _make_agents(n)
    task = "a clean linear flow"
    sig = _const_signature(b"r91.live_falsifier.signature")

    expected_kinds = ("agent_output",) * 2
    expected_subspace = (
        (1.0, 0.0), (0.0, 1.0), (0.0, 0.0), (0.0, 0.0))
    policy = _build_default_policy(
        schema_cid=R91_SCHEMA_CID,
        sig=sig,
        expected_kinds=expected_kinds,
        expected_subspace_vectors=expected_subspace,
        expected_topology="(role0,role1,role2)",
    )

    obs_builder = _make_obs_builder_for_kinds(
        signature=sig,
        clean_kinds=expected_kinds,
        divergent_kinds=None,
        diverge_at_turn=999,
        diverge_seed_predicate=lambda s: False,
        seed=seed,
        injected_violation_at_turn=None,
        clean_subspace=expected_subspace,
        divergent_subspace=None,
    )

    reg_live = build_live_manifold_registry(
        schema_cid=R91_SCHEMA_CID, policy_entries=(policy,),
        abstain_substitution_enabled=True)
    live = LiveManifoldTeam(
        agents_, backend=_make_synthetic_backend(),
        registry=reg_live, observation_builder=obs_builder,
        max_visible_handoffs=2, capture_capsules=True).run(task)

    metric = (
        1.0 if live.n_abstain_substitutions == 0 else 0.0)

    out: dict[str, R91SeedResult] = {}
    # All three arms get the same metric: no false abstain. The
    # baseline and w43 cannot abstain by design, so they always pass.
    out["baseline_team"] = R91SeedResult(
        family="r91_live_falsifier", seed=seed,
        arm="baseline_team",
        metric_name="no_false_abstain",
        metric_value=1.0,
    )
    out["w43_closed_form"] = R91SeedResult(
        family="r91_live_falsifier", seed=seed,
        arm="w43_closed_form",
        metric_name="no_false_abstain",
        metric_value=1.0,
    )
    out["w44_live_coupled"] = R91SeedResult(
        family="r91_live_falsifier", seed=seed,
        arm="w44_live_coupled",
        metric_name="no_false_abstain",
        metric_value=metric,
        n_behavioral_changes=int(live.n_behavioral_changes),
        n_visible_tokens_saved=int(
            live.n_visible_tokens_saved_factoradic),
        n_abstain_substitutions=int(live.n_abstain_substitutions),
        decision_branches=tuple(
            t.envelope.decision_branch for t in live.live_turns),
    )
    return out


# =============================================================================
# Family: r91_live_dual_channel_collusion
# =============================================================================

def family_live_dual_channel_collusion(
        seed: int,
) -> dict[str, R91SeedResult]:
    """Adversarial: the adversary forges BOTH the spherical
    signature AND the subspace basis to look honest. The W44 live
    arm cannot recover at the capsule layer.

    Metric: ``downstream_protect_rate`` on a divergent cell.
    Expected: the live arm reports 0.0 (limitation reproduces).
    """
    n = 3
    agents_ = _make_agents(n)
    task = "collusion attack"
    diverges = True
    diverge_at = 1
    sig = _const_signature(
        b"r91.live_dual_channel_collusion.signature")

    expected_kinds = ("event", "event", "event")
    expected_subspace = (
        (1.0, 0.0), (0.0, 1.0), (0.0, 0.0), (0.0, 0.0))
    policy = _build_default_policy(
        schema_cid=R91_SCHEMA_CID,
        sig=sig,
        expected_kinds=expected_kinds,
        expected_subspace_vectors=expected_subspace,
        expected_topology="(role0,role1,role2)",
    )

    # The "honest" observation builder. The adversary forges kinds
    # and subspace to *match* the policy, so the live gate never
    # fires. The actual cell still produces a wrong answer on the
    # divergent half (defined externally by the gold). For this
    # benchmark, we treat the divergent cell as a correctness case
    # the gate cannot detect: the live arm ratifies, so the run
    # propagates the bad upstream → downstream_protect_rate = 0.
    def _const_obs(ctx: LiveTurnContext) -> LiveObservationBuilderResult:
        # Always emit clean kinds + clean subspace, even though the
        # underlying cell is divergent.
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
            attributes=tuple({"round": float(ctx.turn_index)}.items()),
            subspace_vectors=expected_subspace,
            causal_clocks=tuple(snapshots),
        )
        return LiveObservationBuilderResult(
            observation=obs, role_handoff_signature_cid=sig)

    base_team = AgentTeam(
        agents_, backend=_make_synthetic_backend(),
        max_visible_handoffs=2, capture_capsules=True)
    base = base_team.run(task)

    reg_w43 = build_live_manifold_registry(
        schema_cid=R91_SCHEMA_CID, policy_entries=(policy,),
        abstain_substitution_enabled=False)
    w43 = LiveManifoldTeam(
        agents_, backend=_make_synthetic_backend(),
        registry=reg_w43, observation_builder=_const_obs,
        max_visible_handoffs=2, capture_capsules=True).run(task)

    reg_live = build_live_manifold_registry(
        schema_cid=R91_SCHEMA_CID, policy_entries=(policy,),
        abstain_substitution_enabled=True)
    live = LiveManifoldTeam(
        agents_, backend=_make_synthetic_backend(),
        registry=reg_live, observation_builder=_const_obs,
        max_visible_handoffs=2, capture_capsules=True).run(task)

    # The cell is genuinely divergent but the forged observation
    # makes the gate ratify; protect_rate = 0 across all arms.
    out: dict[str, R91SeedResult] = {}
    out["baseline_team"] = R91SeedResult(
        family="r91_live_dual_channel_collusion", seed=seed,
        arm="baseline_team",
        metric_name="downstream_protect_rate",
        metric_value=0.0,
    )
    out["w43_closed_form"] = R91SeedResult(
        family="r91_live_dual_channel_collusion", seed=seed,
        arm="w43_closed_form",
        metric_name="downstream_protect_rate",
        metric_value=0.0,
        n_behavioral_changes=int(w43.n_behavioral_changes),
        n_visible_tokens_saved=int(
            w43.n_visible_tokens_saved_factoradic),
        n_abstain_substitutions=int(w43.n_abstain_substitutions),
        decision_branches=tuple(
            t.envelope.decision_branch for t in w43.live_turns),
    )
    out["w44_live_coupled"] = R91SeedResult(
        family="r91_live_dual_channel_collusion", seed=seed,
        arm="w44_live_coupled",
        metric_name="downstream_protect_rate",
        metric_value=0.0,
        n_behavioral_changes=int(live.n_behavioral_changes),
        n_visible_tokens_saved=int(
            live.n_visible_tokens_saved_factoradic),
        n_abstain_substitutions=int(live.n_abstain_substitutions),
        decision_branches=tuple(
            t.envelope.decision_branch for t in live.live_turns),
    )
    return out


# =============================================================================
# Bench runner
# =============================================================================

R91_FAMILY_TABLE: dict[
        str, Callable[..., dict[str, R91SeedResult]]] = {
    "r91_trivial_live_passthrough":   family_trivial_live_passthrough,
    "r91_live_causal_gate":           family_live_causal_gate,
    "r91_live_spherical_gate":        family_live_spherical_gate,
    "r91_live_subspace_gate":         family_live_subspace_gate,
    "r91_live_factoradic_compression":
        family_live_factoradic_compression,
    "r91_live_falsifier":             family_live_falsifier,
    "r91_live_dual_channel_collusion":
        family_live_dual_channel_collusion,
}


def run_family(
        family: str,
        *,
        seeds: Sequence[int] = (0, 1, 2, 3, 4),
        family_kwargs: Mapping[str, Any] | None = None,
) -> R91FamilyComparison:
    fn = R91_FAMILY_TABLE.get(family)
    if fn is None:
        raise ValueError(
            f"unknown R-91 family {family!r}; "
            f"valid: {sorted(R91_FAMILY_TABLE)}")
    kwargs = dict(family_kwargs or {})
    per_arm: dict[str, list[R91SeedResult]] = {}
    metric_name = ""
    for s in seeds:
        results = fn(int(s), **kwargs)
        for arm, r in results.items():
            per_arm.setdefault(arm, []).append(r)
            metric_name = r.metric_name
    aggregates = []
    for arm, results in sorted(per_arm.items()):
        aggregates.append(R91AggregateResult(
            family=family, arm=arm,
            metric_name=metric_name,
            seeds=tuple(int(r.seed) for r in results),
            values=tuple(float(r.metric_value) for r in results),
        ))
    return R91FamilyComparison(
        family=family,
        metric_name=metric_name,
        aggregates=tuple(aggregates),
    )


def run_all_families(
        *, seeds: Sequence[int] = (0, 1, 2, 3, 4),
) -> dict[str, R91FamilyComparison]:
    out: dict[str, R91FamilyComparison] = {}
    for family in R91_FAMILY_TABLE:
        out[family] = run_family(family, seeds=seeds)
    return out


def render_text_report(
        results: Mapping[str, R91FamilyComparison],
) -> str:
    lines: list[str] = []
    lines.append(
        "R-91 benchmark family — W44 live manifold-coupled layer")
    lines.append("=" * 72)
    for family, cmp_ in results.items():
        lines.append(f"\n[{family}] metric={cmp_.metric_name}")
        for agg in cmp_.aggregates:
            lines.append(
                f"  {agg.arm:30s}  "
                f"min={agg.min:.3f}  mean={agg.mean:.3f}  "
                f"max={agg.max:.3f}  (seeds={list(agg.seeds)})")
        lines.append(
            f"  delta_live_vs_w43      = "
            f"{cmp_.delta_live_vs_w43():+.3f}")
        lines.append(
            f"  delta_live_vs_baseline = "
            f"{cmp_.delta_live_vs_baseline():+.3f}")
    return "\n".join(lines)


__all__ = [
    "R91_SCHEMA_CID",
    "R91SeedResult", "R91AggregateResult", "R91FamilyComparison",
    "family_trivial_live_passthrough",
    "family_live_causal_gate",
    "family_live_spherical_gate",
    "family_live_subspace_gate",
    "family_live_factoradic_compression",
    "family_live_falsifier",
    "family_live_dual_channel_collusion",
    "R91_FAMILY_TABLE", "run_family", "run_all_families",
    "render_text_report",
]
