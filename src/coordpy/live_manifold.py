"""W44 Live Manifold-Coupled Coordination (LMCC) — capsule-native
live-coupled orchestrator that couples W43 product-manifold state
to actual agent-team run behaviour.

W44 is the first capsule-native CoordPy layer that lets the W43
product-manifold channels actually *change run behaviour* in a
sequential agent team. Three channels (spherical / subspace /
causal) drive per-turn *gating decisions*: when a registered policy
flags a violation, the live orchestrator substitutes a deterministic
abstain output for the agent's ``generate()`` call so the next
agent's prompt never sees the bad upstream handoff. A fourth channel
(factoradic route) can replace the textual rendering of the role
arrival order with a single integer header, reducing the visible
prompt-token cost while preserving the full route in the audit
envelope. The remaining two channels (hyperbolic, euclidean) are
audit-only at the live layer.

W44 is strictly additive on top of W43 and on the released v3.43
``AgentTeam`` surface:

  * the released ``coordpy.AgentTeam.run`` path is byte-for-byte
    unchanged
  * the W43 surface (``coordpy.product_manifold``,
    ``coordpy.r90_benchmark``) is unchanged
  * with the trivial live registry (``live_enabled=False``,
    ``inline_route_mode='textual'``,
    ``abstain_substitution_enabled=False``), the W44 orchestrator
    reduces to ``AgentTeam.run`` byte-for-byte (the
    ``W44-L-TRIVIAL-LIVE-PASSTHROUGH`` falsifier).

W44 is held outside the stable SDK contract at this milestone:
the module ships at ``coordpy.live_manifold`` and is reachable
only through an explicit import.

Honest scope (do-not-overstate)
-------------------------------

W44 does NOT close any of the W43 conjectures
(``W43-C-MIXED-CURVATURE-LATENT``,
``W43-C-COLLECTIVE-KV-POOLING``,
``W43-C-FULL-GRASSMANNIAN-HOMOTOPY``). These remain substrate-
blocked.

W44 does NOT claim that real LLMs decode the factoradic header.
The factoradic compressor's behavioural effect is measured on a
deterministic ``SyntheticLLMClient`` testbed; on real LLMs the
saving is a visible-token saving without a guaranteed behavioural-
decoding gain.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import time
from typing import Any, Callable, Mapping, Sequence

from .agents import (
    Agent,
    AgentTurn,
    TEAM_RESULT_SCHEMA,
    _safe_usage_snapshot,
    _sha256_str,
)
from .capsule import CapsuleBudget, CapsuleLedger, render_view
from .llm_backend import LLMBackend
from .product_manifold import (
    CausalVectorClock,
    CellObservation,
    ProductManifoldOrchestrator,
    ProductManifoldPolicyEntry,
    ProductManifoldPolicyRegistry,
    ProductManifoldRegistry,
    SphericalConsensusSignature,
    SubspaceBasis,
    W43_BRANCH_PMC_CAUSAL_VIOLATION_ABSTAINED,
    W43_BRANCH_PMC_NO_POLICY,
    W43_BRANCH_PMC_RATIFIED,
    W43_BRANCH_PMC_REJECTED,
    W43_BRANCH_PMC_SPHERICAL_DIVERGENCE_ABSTAINED,
    W43_BRANCH_PMC_SUBSPACE_DRIFT_ABSTAINED,
    W43_BRANCH_TRIVIAL_PMC_PASSTHROUGH,
    W43ProductManifoldResult,
    build_product_manifold_registry,
    build_trivial_product_manifold_registry,
    encode_cell_channels,
    encode_factoradic_route,
    encode_spherical_consensus,
    encode_subspace_basis,
)
from .team_coord import capsule_team_handoff


# =============================================================================
# Schema, branches, defaults
# =============================================================================

W44_LIVE_MANIFOLD_SCHEMA_VERSION: str = "coordpy.live_manifold.v1"
W44_TEAM_RESULT_SCHEMA: str = "coordpy.live_manifold_team_result.v1"

# Decision branches.
W44_BRANCH_TRIVIAL_LIVE_PASSTHROUGH: str = "live_trivial_passthrough"
W44_BRANCH_LIVE_DISABLED: str = "live_disabled"
W44_BRANCH_LIVE_RATIFIED: str = "live_ratified"
W44_BRANCH_LIVE_NO_POLICY: str = "live_no_policy"
W44_BRANCH_LIVE_CAUSAL_ABSTAIN: str = "live_causal_abstain"
W44_BRANCH_LIVE_SPHERICAL_ABSTAIN: str = "live_spherical_abstain"
W44_BRANCH_LIVE_SUBSPACE_ABSTAIN: str = "live_subspace_abstain"
W44_BRANCH_LIVE_REJECTED: str = "live_rejected"

W44_ALL_BRANCHES: tuple[str, ...] = (
    W44_BRANCH_TRIVIAL_LIVE_PASSTHROUGH,
    W44_BRANCH_LIVE_DISABLED,
    W44_BRANCH_LIVE_RATIFIED,
    W44_BRANCH_LIVE_NO_POLICY,
    W44_BRANCH_LIVE_CAUSAL_ABSTAIN,
    W44_BRANCH_LIVE_SPHERICAL_ABSTAIN,
    W44_BRANCH_LIVE_SUBSPACE_ABSTAIN,
    W44_BRANCH_LIVE_REJECTED,
)

W44_ABSTAIN_BRANCHES: frozenset[str] = frozenset({
    W44_BRANCH_LIVE_CAUSAL_ABSTAIN,
    W44_BRANCH_LIVE_SPHERICAL_ABSTAIN,
    W44_BRANCH_LIVE_SUBSPACE_ABSTAIN,
})

# Inline route modes.
W44_ROUTE_MODE_TEXTUAL: str = "textual"
W44_ROUTE_MODE_FACTORADIC: str = "factoradic"
W44_ROUTE_MODE_FACTORADIC_WITH_TEXTUAL: str = "factoradic_with_textual"

W44_ALL_ROUTE_MODES: tuple[str, ...] = (
    W44_ROUTE_MODE_TEXTUAL,
    W44_ROUTE_MODE_FACTORADIC,
    W44_ROUTE_MODE_FACTORADIC_WITH_TEXTUAL,
)

W44_DEFAULT_ABSTAIN_OUTPUT: str = "[live_manifold_abstain]"
W44_DEFAULT_PARENT_W42_CID: str = hashlib.sha256(
    b"w44.parent_w42_cid.placeholder").hexdigest()


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
# Observation builder
# =============================================================================

@dataclasses.dataclass(frozen=True)
class LiveObservationBuilderResult:
    """Bundle of (observation, signature_cid) returned by an
    observation builder for a single turn."""

    observation: CellObservation
    role_handoff_signature_cid: str


# An observation builder is a callable that takes the per-turn
# state and returns an observation + signature CID. Callers can
# customise this for their domain; the default implementation below
# covers the sequential-agent-team case.
LiveObservationBuilder = Callable[
    ["LiveTurnContext"], LiveObservationBuilderResult
]


@dataclasses.dataclass(frozen=True)
class LiveTurnContext:
    """Per-turn context handed to an observation builder.

    Fields:

    * ``turn_index`` — 0-based index of the turn about to run.
    * ``role_universe`` — canonical sorted role names of the team.
    * ``role_arrival_order`` — roles that have already authored a
      handoff, in arrival order.
    * ``current_role`` — role of the agent about to run.
    * ``recent_handoffs`` — bounded list of (role, payload) the
      agent will see (post-bounding).
    * ``all_prior_outputs`` — full list of (role, payload) for
      every prior turn (used by the factoradic compressor and the
      naive-token counterfactual).
    * ``causal_counts`` — per-role Lamport counters as observed so
      far. Mutating this dict is allowed (it is a fresh copy).
    * ``injected_clock_violation`` — optional override for the
      clock at this turn (used by R-91 fixtures to inject
      violations without changing the team contract).
    * ``observed_claim_kinds_override`` — optional override for the
      observed claim_kinds at this turn (used by R-91 fixtures to
      simulate spherical divergence).
    * ``observed_subspace_override`` — optional override for the
      observed subspace vectors at this turn (used by R-91
      fixtures to simulate subspace drift).
    """

    turn_index: int
    role_universe: tuple[str, ...]
    role_arrival_order: tuple[str, ...]
    current_role: str
    recent_handoffs: tuple[tuple[str, str], ...]
    all_prior_outputs: tuple[tuple[str, str], ...]
    causal_counts: dict[str, int]
    injected_clock_violation: bool = False
    observed_claim_kinds_override: tuple[str, ...] | None = None
    observed_subspace_override: tuple[tuple[float, ...], ...] | None = None


def default_live_observation_builder(
        ctx: LiveTurnContext,
) -> LiveObservationBuilderResult:
    """Default observation builder for the sequential-agent-team
    case.

    Builds:

      * ``branch_path`` from the (linear) turn index — sequential
        teams have no branching, so the path is all-zeros of length
        ``turn_index``.
      * ``claim_kinds`` from the recent handoffs (using
        ``"agent_output"`` as the canonical claim_kind unless an
        override is supplied).
      * ``attributes`` from ``round`` and ``n_handoffs``.
      * ``role_arrival_order`` and ``role_universe`` from the
        context.
      * ``subspace_vectors`` from a small constant basis unless an
        override is supplied (the released team contract does not
        carry subspace state in the handoff payload; an external
        observation builder is the integration point for callers
        that do).
      * ``causal_clocks`` from a Lamport vector clock that
        increments the role's component on each turn. If
        ``injected_clock_violation`` is True, the last clock is
        rewound by one slot to simulate an out-of-order arrival.

    The role_handoff signature CID is derived from the sorted tuple
    ``(claim_kinds, roles, branch_depth)`` — stable under
    permutations of those, so it acts like the W42 role-handoff
    signature for the live-coupling fixture.
    """
    # Branch path: sequential teams have no branching. We use
    # ``turn_index`` zeros as a deterministic placeholder.
    branch_path = tuple(0 for _ in range(ctx.turn_index))
    # Claim kinds: prefer the override, else derive from the
    # recent_handoffs.
    if ctx.observed_claim_kinds_override is not None:
        claim_kinds = tuple(ctx.observed_claim_kinds_override)
    else:
        claim_kinds = tuple(
            "agent_output" for _ in ctx.recent_handoffs)
    # Causal clock: increment the *previous* role's component on
    # each turn (the upstream agent that just produced a handoff is
    # the one that increments).
    counts = dict(ctx.causal_counts)
    if ctx.role_arrival_order:
        last_role = ctx.role_arrival_order[-1]
        counts[last_role] = counts.get(last_role, 0) + 1
    elif ctx.current_role:
        counts[ctx.current_role] = counts.get(ctx.current_role, 0)
    # Build per-arrival snapshot. We re-walk the role_arrival_order
    # to materialise the clock at every step.
    snapshots: list[CausalVectorClock] = []
    walk_counts: dict[str, int] = {r: 0 for r in ctx.role_universe}
    for r in ctx.role_arrival_order:
        walk_counts[r] = walk_counts.get(r, 0) + 1
        snapshots.append(
            CausalVectorClock.from_mapping(dict(walk_counts)))
    if ctx.injected_clock_violation and len(snapshots) >= 2:
        # Rewind the last clock to a strictly smaller one.
        snapshots[-1] = snapshots[-min(3, len(snapshots))]
    if ctx.observed_subspace_override is not None:
        subspace_vectors = ctx.observed_subspace_override
    else:
        # Default: a clean diagonal basis. The released team
        # contract does not surface a subspace; this is the safe
        # fixture default.
        subspace_vectors = (
            (1.0, 0.0), (0.0, 1.0), (0.0, 0.0), (0.0, 0.0))
    obs = CellObservation(
        branch_path=branch_path,
        claim_kinds=claim_kinds,
        role_arrival_order=tuple(ctx.role_arrival_order),
        role_universe=tuple(ctx.role_universe),
        attributes=tuple({
            "round": float(ctx.turn_index),
            "n_handoffs": float(len(ctx.recent_handoffs)),
        }.items()),
        subspace_vectors=subspace_vectors,
        causal_clocks=tuple(snapshots),
    )
    sig = _sha256_hex({
        "kind": "w44_default_live_signature",
        "claim_kinds": sorted(claim_kinds),
        "roles": sorted(ctx.role_arrival_order),
        "branch_depth": len(branch_path),
    })
    return LiveObservationBuilderResult(
        observation=obs,
        role_handoff_signature_cid=sig,
    )


# =============================================================================
# Gating decision
# =============================================================================

@dataclasses.dataclass(frozen=True)
class LiveGatingDecision:
    """Result of running the live gate on one turn.

    Fields:

    * ``branch`` — one of :data:`W44_ALL_BRANCHES`.
    * ``pmc_branch`` — the underlying W43 branch that drove the
      decision (recorded for audit).
    * ``spherical_agreement`` / ``subspace_drift`` /
      ``causal_admissible`` — channel-level evidence from W43.
    * ``factoradic_int`` — the route channel's factoradic integer
      (for compression and audit); 0 when the route is empty.
    * ``factoradic_n_bits`` — the route channel's information
      capacity in bits.
    * ``role_handoff_signature_cid`` — the registered signature CID
      for the cell.
    * ``policy_entry_cid`` — the policy entry CID, or empty when
      the cell hit the no-policy branch.
    * ``pmc_envelope_cid`` — the W43 manifest-v13 outer CID, or
      empty for the trivial-passthrough branch.
    * ``abstain_reason`` — short reason string (empty when not
      abstaining).
    """

    branch: str
    pmc_branch: str
    spherical_agreement: float
    subspace_drift: float
    causal_admissible: bool
    factoradic_int: int
    factoradic_n_bits: int
    role_handoff_signature_cid: str
    policy_entry_cid: str
    pmc_envelope_cid: str
    abstain_reason: str

    def is_abstain(self) -> bool:
        return self.branch in W44_ABSTAIN_BRANCHES


# =============================================================================
# Live registry + orchestrator
# =============================================================================

@dataclasses.dataclass
class LiveManifoldRegistry:
    """Controller-side configuration for the live coupling.

    Wraps a :class:`ProductManifoldRegistry` (the W43 inner) and
    adds three live-layer toggles:

      * ``live_enabled`` — master switch for the live coupling.
        When False, the orchestrator falls through to the W43
        passthrough (W44 disabled).
      * ``inline_route_mode`` — one of
        :data:`W44_ROUTE_MODE_TEXTUAL`,
        :data:`W44_ROUTE_MODE_FACTORADIC`,
        :data:`W44_ROUTE_MODE_FACTORADIC_WITH_TEXTUAL`. Controls
        whether the prompt builder replaces the textual role-
        arrival ordering with a factoradic header.
      * ``abstain_substitution_enabled`` — when True, abstain
        decisions substitute a deterministic abstain output for
        the agent's ``generate()`` call. When False, the live
        layer records the gating decision for audit but does not
        change run behaviour (research / measurement only).
    """

    schema_cid: str
    pmc_registry: ProductManifoldRegistry
    live_enabled: bool = True
    inline_route_mode: str = W44_ROUTE_MODE_TEXTUAL
    abstain_substitution_enabled: bool = True
    abstain_output: str = W44_DEFAULT_ABSTAIN_OUTPUT

    @property
    def is_trivial(self) -> bool:
        """The live registry is trivial iff the W43 inner is
        trivial AND the live-layer toggles are off / textual."""
        return (
            self.pmc_registry.is_trivial
            and not self.live_enabled
            and self.inline_route_mode == W44_ROUTE_MODE_TEXTUAL
            and not self.abstain_substitution_enabled
        )


def _classify_pmc_branch_to_live(
        pmc_branch: str,
) -> str:
    """Map a W43 PMC decision branch to the corresponding W44
    live-layer branch."""
    if pmc_branch == W43_BRANCH_PMC_RATIFIED:
        return W44_BRANCH_LIVE_RATIFIED
    if pmc_branch == W43_BRANCH_PMC_NO_POLICY:
        return W44_BRANCH_LIVE_NO_POLICY
    if pmc_branch == W43_BRANCH_PMC_CAUSAL_VIOLATION_ABSTAINED:
        return W44_BRANCH_LIVE_CAUSAL_ABSTAIN
    if pmc_branch == W43_BRANCH_PMC_SPHERICAL_DIVERGENCE_ABSTAINED:
        return W44_BRANCH_LIVE_SPHERICAL_ABSTAIN
    if pmc_branch == W43_BRANCH_PMC_SUBSPACE_DRIFT_ABSTAINED:
        return W44_BRANCH_LIVE_SUBSPACE_ABSTAIN
    if pmc_branch == W43_BRANCH_TRIVIAL_PMC_PASSTHROUGH:
        return W44_BRANCH_TRIVIAL_LIVE_PASSTHROUGH
    if pmc_branch == W43_BRANCH_PMC_REJECTED:
        return W44_BRANCH_LIVE_REJECTED
    # Default safety: treat unknown as no-policy (no behavioural
    # change).
    return W44_BRANCH_LIVE_NO_POLICY


class LiveManifoldOrchestrator:
    """Per-turn live gating + prompt-construction witness.

    Wraps a :class:`ProductManifoldOrchestrator` (the W43 inner) and
    a :class:`LiveManifoldRegistry`. Stateless across cells; each
    call to :meth:`gate` is a fresh gating decision.
    """

    def __init__(self, registry: LiveManifoldRegistry) -> None:
        self.registry = registry
        self._pmc = ProductManifoldOrchestrator(
            registry=registry.pmc_registry,
            require_w43_verification=True,
        )

    @property
    def schema_cid(self) -> str:
        return str(self.registry.schema_cid)

    def reset_session(self) -> None:
        self._pmc.reset_session()

    def gate(
            self,
            *,
            observation: CellObservation,
            role_handoff_signature_cid: str,
            parent_w42_cid: str,
            n_w42_visible_tokens: int,
    ) -> tuple[LiveGatingDecision, W43ProductManifoldResult]:
        """Run the live gate for one turn.

        Returns ``(decision, w43_result)`` where ``decision`` is the
        live-layer classification and ``w43_result`` is the W43
        envelope for downstream audit.
        """
        # Defer to the W43 inner for the closed-form decision +
        # envelope. This guarantees the W44 layer never duplicates
        # W43 logic; it only re-classifies the W43 branch and adds
        # the live-coupling decision.
        w43 = self._pmc.decode(
            observation=observation,
            role_handoff_signature_cid=role_handoff_signature_cid,
            parent_w42_cid=parent_w42_cid,
            n_w42_visible_tokens=int(n_w42_visible_tokens),
        )
        if not self.registry.live_enabled:
            branch = (
                W44_BRANCH_TRIVIAL_LIVE_PASSTHROUGH
                if self.registry.is_trivial
                else W44_BRANCH_LIVE_DISABLED
            )
        else:
            branch = _classify_pmc_branch_to_live(w43.decision_branch)

        abstain_reason = ""
        if branch == W44_BRANCH_LIVE_CAUSAL_ABSTAIN:
            abstain_reason = "causal_violation"
        elif branch == W44_BRANCH_LIVE_SPHERICAL_ABSTAIN:
            abstain_reason = "spherical_divergence"
        elif branch == W44_BRANCH_LIVE_SUBSPACE_ABSTAIN:
            abstain_reason = "subspace_drift"

        decision = LiveGatingDecision(
            branch=branch,
            pmc_branch=w43.decision_branch,
            spherical_agreement=float(w43.spherical_agreement),
            subspace_drift=float(w43.subspace_drift),
            causal_admissible=bool(w43.causal_admissible),
            factoradic_int=int(self._extract_factoradic_int(observation)),
            factoradic_n_bits=int(self._factoradic_n_bits(observation)),
            role_handoff_signature_cid=str(role_handoff_signature_cid),
            policy_entry_cid=str(w43.policy_entry_cid or ""),
            pmc_envelope_cid=str(w43.w43_cid or ""),
            abstain_reason=abstain_reason,
        )
        return decision, w43

    @staticmethod
    def _extract_factoradic_int(obs: CellObservation) -> int:
        """Compute the factoradic integer for the observation's
        role arrival order against role universe."""
        if not obs.role_universe:
            return 0
        index = {r: i for i, r in enumerate(obs.role_universe)}
        try:
            perm = [index[r] for r in obs.role_arrival_order]
        except KeyError:
            return 0
        seen = set(perm)
        for i in range(len(obs.role_universe)):
            if i not in seen:
                perm.append(i)
        return int(encode_factoradic_route(perm).factoradic_int)

    @staticmethod
    def _factoradic_n_bits(obs: CellObservation) -> int:
        if not obs.role_universe:
            return 0
        n = len(obs.role_universe)
        if n < 2:
            return 0
        return int(math.ceil(math.log2(math.factorial(n))))


# =============================================================================
# Live envelope
# =============================================================================

@dataclasses.dataclass(frozen=True)
class LiveManifoldHandoffEnvelope:
    """Sealed live-manifold envelope for one turn of the W44 layer.

    Records:

    * the underlying ``TEAM_HANDOFF`` capsule CID
      (``parent_team_handoff_cid``)
    * the W43 manifest-v13 envelope CID
      (``parent_w43_envelope_cid``)
    * the live-layer decision branch + abstain reason
    * the prompt-construction witness fields
    * the visible-token saving from the factoradic compressor
    * the live witness CID and outer CID

    The outer CID is content-addressed by every other field. The
    verifier (:func:`verify_live_manifold_handoff`) re-derives the
    outer CID from the bytes alone and detects tampering with any
    subfield through one of the disjoint named failure modes.
    """

    schema_version: str
    schema_cid: str
    turn_index: int
    role: str

    parent_team_handoff_cid: str
    parent_w43_envelope_cid: str
    parent_w42_cid: str

    decision_branch: str
    pmc_branch: str
    abstain_reason: str
    role_handoff_signature_cid: str
    policy_entry_cid: str

    inline_route_mode: str
    factoradic_int: int
    factoradic_n_bits: int

    # Prompt-construction witness.
    prompt_sha256: str
    prompt_construction_witness_cid: str
    output_sha256: str

    # Token accounting.
    n_visible_prompt_tokens_textual: int
    n_visible_prompt_tokens_actual: int
    n_visible_prompt_tokens_saved: int
    n_overhead_tokens: int

    behavioral_change: bool

    live_witness_cid: str
    live_outer_cid: str

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def recompute_outer_cid(self) -> str:
        return _compute_w44_outer_cid(
            schema_cid=self.schema_cid,
            parent_team_handoff_cid=self.parent_team_handoff_cid,
            parent_w43_envelope_cid=self.parent_w43_envelope_cid,
            live_witness_cid=self.live_witness_cid,
            turn_index=int(self.turn_index),
        )


def _compute_prompt_construction_witness_cid(
        *,
        turn_index: int,
        role: str,
        prompt_sha256: str,
        inline_route_mode: str,
        factoradic_int: int,
        factoradic_n_bits: int,
        n_visible_prompt_tokens_textual: int,
        n_visible_prompt_tokens_actual: int,
) -> str:
    return _sha256_hex({
        "kind": "w44_prompt_construction_witness",
        "turn_index": int(turn_index),
        "role": str(role),
        "prompt_sha256": str(prompt_sha256),
        "inline_route_mode": str(inline_route_mode),
        "factoradic_int": int(factoradic_int),
        "factoradic_n_bits": int(factoradic_n_bits),
        "n_visible_prompt_tokens_textual": int(
            n_visible_prompt_tokens_textual),
        "n_visible_prompt_tokens_actual": int(
            n_visible_prompt_tokens_actual),
    })


def _compute_w44_live_witness_cid(
        *,
        decision_branch: str,
        pmc_branch: str,
        abstain_reason: str,
        role_handoff_signature_cid: str,
        policy_entry_cid: str,
        prompt_construction_witness_cid: str,
        output_sha256: str,
        behavioral_change: bool,
) -> str:
    return _sha256_hex({
        "kind": "w44_live_witness",
        "decision_branch": str(decision_branch),
        "pmc_branch": str(pmc_branch),
        "abstain_reason": str(abstain_reason),
        "role_handoff_signature_cid": str(role_handoff_signature_cid),
        "policy_entry_cid": str(policy_entry_cid),
        "prompt_construction_witness_cid": str(
            prompt_construction_witness_cid),
        "output_sha256": str(output_sha256),
        "behavioral_change": bool(behavioral_change),
    })


def _compute_w44_outer_cid(
        *,
        schema_cid: str,
        parent_team_handoff_cid: str,
        parent_w43_envelope_cid: str,
        live_witness_cid: str,
        turn_index: int,
) -> str:
    return _sha256_hex({
        "kind": "w44_live_outer",
        "schema_cid": str(schema_cid),
        "parent_team_handoff_cid": str(parent_team_handoff_cid),
        "parent_w43_envelope_cid": str(parent_w43_envelope_cid),
        "live_witness_cid": str(live_witness_cid),
        "turn_index": int(turn_index),
    })


# =============================================================================
# Verifier (12 enumerated W44 failure modes)
# =============================================================================

@dataclasses.dataclass(frozen=True)
class LiveManifoldVerificationOutcome:
    ok: bool
    reason: str
    n_checks: int


def verify_live_manifold_handoff(
        env: "LiveManifoldHandoffEnvelope | None",
        *,
        registered_schema_cid: str,
) -> LiveManifoldVerificationOutcome:
    """Pure-function verifier for the W44 live envelope.

    Enumerates 12 disjoint W44 failure modes:

    1.  ``empty_w44_envelope``
    2.  ``w44_schema_version_unknown``
    3.  ``w44_schema_cid_mismatch``
    4.  ``w44_decision_branch_unknown``
    5.  ``w44_route_mode_unknown``
    6.  ``w44_role_handoff_signature_cid_invalid``
    7.  ``w44_prompt_sha256_invalid``
    8.  ``w44_token_accounting_invalid``
    9.  ``w44_factoradic_bits_invalid``
    10. ``w44_prompt_construction_witness_cid_mismatch``
    11. ``w44_live_witness_cid_mismatch``
    12. ``w44_outer_cid_mismatch``

    Structural field-validity checks (8, 9) precede the derived-CID
    checks (10, 11, 12) so a forged envelope that underflows a
    field is detected by name even when the derived CIDs were
    re-computed from the bad value.
    """
    n = 0
    if env is None:
        return LiveManifoldVerificationOutcome(
            ok=False, reason="empty_w44_envelope", n_checks=0)
    n += 1
    if env.schema_version != W44_LIVE_MANIFOLD_SCHEMA_VERSION:
        return LiveManifoldVerificationOutcome(
            ok=False, reason="w44_schema_version_unknown",
            n_checks=n)
    n += 1
    if env.schema_cid != str(registered_schema_cid):
        return LiveManifoldVerificationOutcome(
            ok=False, reason="w44_schema_cid_mismatch",
            n_checks=n)
    n += 1
    if env.decision_branch not in W44_ALL_BRANCHES:
        return LiveManifoldVerificationOutcome(
            ok=False, reason="w44_decision_branch_unknown",
            n_checks=n)
    n += 1
    if env.inline_route_mode not in W44_ALL_ROUTE_MODES:
        return LiveManifoldVerificationOutcome(
            ok=False, reason="w44_route_mode_unknown",
            n_checks=n)
    n += 1
    if env.decision_branch != W44_BRANCH_TRIVIAL_LIVE_PASSTHROUGH:
        if (not env.role_handoff_signature_cid
                or len(env.role_handoff_signature_cid) != 64):
            return LiveManifoldVerificationOutcome(
                ok=False,
                reason="w44_role_handoff_signature_cid_invalid",
                n_checks=n)
    n += 1
    if (env.prompt_sha256 is None
            or (env.prompt_sha256
                and len(env.prompt_sha256) not in (0, 64))):
        return LiveManifoldVerificationOutcome(
            ok=False, reason="w44_prompt_sha256_invalid",
            n_checks=n)
    n += 1
    if (env.n_visible_prompt_tokens_textual < 0
            or env.n_visible_prompt_tokens_actual < 0
            or env.n_overhead_tokens < 0
            or env.n_visible_prompt_tokens_saved
            != (int(env.n_visible_prompt_tokens_textual)
                - int(env.n_visible_prompt_tokens_actual))):
        return LiveManifoldVerificationOutcome(
            ok=False, reason="w44_token_accounting_invalid",
            n_checks=n)
    n += 1
    if (env.factoradic_n_bits < 0 or env.factoradic_int < 0):
        return LiveManifoldVerificationOutcome(
            ok=False, reason="w44_factoradic_bits_invalid",
            n_checks=n)
    n += 1
    expected_construction = _compute_prompt_construction_witness_cid(
        turn_index=int(env.turn_index),
        role=env.role,
        prompt_sha256=env.prompt_sha256,
        inline_route_mode=env.inline_route_mode,
        factoradic_int=int(env.factoradic_int),
        factoradic_n_bits=int(env.factoradic_n_bits),
        n_visible_prompt_tokens_textual=int(
            env.n_visible_prompt_tokens_textual),
        n_visible_prompt_tokens_actual=int(
            env.n_visible_prompt_tokens_actual),
    )
    if expected_construction != env.prompt_construction_witness_cid:
        return LiveManifoldVerificationOutcome(
            ok=False,
            reason="w44_prompt_construction_witness_cid_mismatch",
            n_checks=n)
    n += 1
    expected_witness = _compute_w44_live_witness_cid(
        decision_branch=env.decision_branch,
        pmc_branch=env.pmc_branch,
        abstain_reason=env.abstain_reason,
        role_handoff_signature_cid=env.role_handoff_signature_cid,
        policy_entry_cid=env.policy_entry_cid,
        prompt_construction_witness_cid=(
            env.prompt_construction_witness_cid),
        output_sha256=env.output_sha256,
        behavioral_change=bool(env.behavioral_change),
    )
    if expected_witness != env.live_witness_cid:
        return LiveManifoldVerificationOutcome(
            ok=False, reason="w44_live_witness_cid_mismatch",
            n_checks=n)
    n += 1
    if env.recompute_outer_cid() != env.live_outer_cid:
        return LiveManifoldVerificationOutcome(
            ok=False, reason="w44_outer_cid_mismatch",
            n_checks=n)
    n += 1
    return LiveManifoldVerificationOutcome(
        ok=True, reason="ok", n_checks=n)


# =============================================================================
# Live team result
# =============================================================================

@dataclasses.dataclass(frozen=True)
class LiveManifoldTurn:
    """One turn of a :class:`LiveManifoldTeam` run, augmenting
    :class:`AgentTurn` with live-layer metadata.

    Carries the underlying agent turn plus the live decision and
    the sealed live envelope. The agent turn's ``output`` is the
    actual handoff payload (potentially the abstain output when
    the live gate substituted it).
    """

    agent_turn: AgentTurn
    decision: LiveGatingDecision
    envelope: LiveManifoldHandoffEnvelope


@dataclasses.dataclass(frozen=True)
class LiveManifoldTeamResult:
    """Result of a :class:`LiveManifoldTeam` run.

    ``base_result`` mirrors the released :class:`coordpy.TeamResult`
    contract for replay compatibility: the captured capsule chain,
    final output, totals, and per-turn agent turns. The W44 layer
    adds:

      * ``live_turns`` — per-turn ``LiveManifoldTurn`` records
        (decisions + envelopes).
      * ``n_behavioral_changes`` — number of turns where the live
        gate or factoradic compressor materially changed the run.
      * ``n_visible_tokens_saved_factoradic`` — total visible
        prompt-token saving from the factoradic compressor.
      * ``n_abstain_substitutions`` — number of turns where an
        abstain output was substituted for a real ``generate()``
        call.
    """

    task: str
    final_output: str
    turns: tuple[AgentTurn, ...]
    live_turns: tuple[LiveManifoldTurn, ...]
    capsule_view: dict[str, Any] | None = None
    root_cid: str | None = None
    total_prompt_tokens: int = 0
    total_output_tokens: int = 0
    total_wall_ms: float = 0.0
    total_calls: int = 0
    backend_model: str = ""
    backend_base_url: str | None = None
    team_instructions: str = ""
    task_summary: str | None = None
    max_visible_handoffs: int = 0
    stopped_early: bool = False
    n_behavioral_changes: int = 0
    n_visible_tokens_saved_factoradic: int = 0
    n_abstain_substitutions: int = 0
    schema: str = W44_TEAM_RESULT_SCHEMA

    @property
    def total_tokens(self) -> int:
        return int(self.total_prompt_tokens
                   + self.total_output_tokens)


# =============================================================================
# Live manifold team
# =============================================================================

class LiveManifoldTeam:
    """W44 live-coupled agent team.

    Wraps the released :class:`coordpy.AgentTeam` contract with the
    live manifold gate + factoradic compressor. With the trivial
    live registry, this team reduces to ``AgentTeam.run`` byte-for-
    byte (the W44-L-TRIVIAL-LIVE-PASSTHROUGH falsifier).
    """

    def __init__(
            self,
            agents: Sequence[Agent],
            *,
            backend: Any | None = None,
            registry: LiveManifoldRegistry,
            observation_builder: LiveObservationBuilder | None = None,
            team_instructions: str = "",
            max_visible_handoffs: int = 4,
            capture_capsules: bool = True,
            task_summary: str | None = None,
            handoff_budget: "CapsuleBudget | None" = None,
            parent_w42_cid: str = W44_DEFAULT_PARENT_W42_CID,
    ) -> None:
        if not agents:
            raise ValueError(
                "LiveManifoldTeam requires at least one agent")
        if max_visible_handoffs <= 0:
            raise ValueError("max_visible_handoffs must be > 0")
        self.agents = tuple(agents)
        self.backend = backend
        self.registry = registry
        self.orchestrator = LiveManifoldOrchestrator(registry)
        self.observation_builder = (
            observation_builder or default_live_observation_builder)
        self.team_instructions = team_instructions.strip()
        self.max_visible_handoffs = int(max_visible_handoffs)
        self.capture_capsules = bool(capture_capsules)
        self.task_summary = (
            task_summary.strip() if task_summary else None)
        self.handoff_budget = handoff_budget
        self.parent_w42_cid = str(parent_w42_cid)

    @property
    def schema_cid(self) -> str:
        return self.orchestrator.schema_cid

    def _resolve_backend(self, member: Agent) -> LLMBackend:
        backend = member.backend or self.backend
        if backend is None:
            raise ValueError(
                "no backend configured; pass backend=... to "
                "LiveManifoldTeam")
        if not isinstance(backend, LLMBackend):
            raise TypeError(
                "backend must satisfy the LLMBackend protocol")
        return backend

    def _build_prompt(
            self,
            *,
            member: Agent,
            task: str,
            turn_index: int,
            recent_handoffs: Sequence[tuple[str, str]],
            all_prior_outputs: Sequence[tuple[str, str]],
            decision: LiveGatingDecision,
            role_universe: Sequence[str],
            role_arrival_order: Sequence[str],
    ) -> tuple[str, str, int, int]:
        """Build the bounded prompt + a textual-shadow prompt for
        token-accounting purposes.

        Returns ``(bounded_prompt, textual_shadow_prompt,
        n_textual_tokens, n_actual_tokens)``.

        When ``inline_route_mode`` is :data:`W44_ROUTE_MODE_TEXTUAL`,
        the bounded prompt and the textual shadow are identical. When
        it is :data:`W44_ROUTE_MODE_FACTORADIC` (or the
        ``_with_textual`` variant), the bounded prompt's role-
        arrival rendering is replaced with a single
        ``FACTORADIC_ROUTE`` header; the textual shadow is left as
        the AgentTeam-equivalent textual rendering, so the saving
        is computable as
        ``len(textual.split()) - len(bounded.split())``.
        """
        # Standard parts (shared by both shadow and bounded).
        common_parts: list[str] = []
        if self.team_instructions:
            common_parts.append(self.team_instructions)
        common_parts.append(f"Agent: {member.name}")
        common_parts.append(f"Role: {member.effective_role}")
        common_parts.append(member.instructions.strip())
        if turn_index == 0 or self.task_summary is None:
            common_parts.append(f"Task: {task.strip()}")
        else:
            common_parts.append(
                f"Task summary: {self.task_summary.strip()}")
        # Textual shadow — the AgentTeam-equivalent rendering.
        textual_parts = list(common_parts)
        if recent_handoffs:
            rendered = "\n".join(
                f"- {role}: {text}"
                for role, text in recent_handoffs[
                    -self.max_visible_handoffs:])
            textual_parts.append(
                "Visible team handoffs (bounded to avoid token "
                f"cramming):\n{rendered}")
        textual_parts.append(
            "Reply with your contribution for the next team "
            "member.")
        textual_prompt = "\n\n".join(textual_parts)
        # Bounded prompt — substitute the textual rendering with a
        # factoradic header when configured.
        bounded_parts = list(common_parts)
        mode = self.registry.inline_route_mode
        if (mode in (W44_ROUTE_MODE_FACTORADIC,
                       W44_ROUTE_MODE_FACTORADIC_WITH_TEXTUAL)
                and decision.factoradic_n_bits > 0
                and recent_handoffs):
            header = (
                f"FACTORADIC_ROUTE: {decision.factoradic_int} "
                f"over {','.join(role_universe)}")
            if mode == W44_ROUTE_MODE_FACTORADIC_WITH_TEXTUAL:
                rendered = "\n".join(
                    f"- {role}: {text}"
                    for role, text in recent_handoffs[
                        -self.max_visible_handoffs:])
                bounded_parts.append(
                    f"{header}\nVisible team handoffs:\n{rendered}")
            else:
                bounded_parts.append(header)
        elif recent_handoffs:
            rendered = "\n".join(
                f"- {role}: {text}"
                for role, text in recent_handoffs[
                    -self.max_visible_handoffs:])
            bounded_parts.append(
                "Visible team handoffs (bounded to avoid token "
                f"cramming):\n{rendered}")
        bounded_parts.append(
            "Reply with your contribution for the next team "
            "member.")
        bounded_prompt = "\n\n".join(bounded_parts)
        n_textual = len(textual_prompt.split())
        n_actual = len(bounded_prompt.split())
        return bounded_prompt, textual_prompt, n_textual, n_actual

    def run(
            self,
            task: str,
            *,
            progress: Callable[[LiveManifoldTurn], None] | None = None,
    ) -> LiveManifoldTeamResult:
        """Run the live-coupled team once over ``task``."""
        ledger = (
            CapsuleLedger() if self.capture_capsules else None)
        agent_turns: list[AgentTurn] = []
        live_turns: list[LiveManifoldTurn] = []
        recent_handoffs: list[tuple[str, str]] = []
        all_prior_outputs: list[tuple[str, str]] = []
        role_arrival_order: list[str] = []
        causal_counts: dict[str, int] = {
            a.effective_role: 0 for a in self.agents}
        parent_cid: str | None = None
        total_prompt_tokens = 0
        total_output_tokens = 0
        total_wall_ms = 0.0
        total_calls = 0
        n_behavioral_changes = 0
        n_visible_tokens_saved_factoradic = 0
        n_abstain_substitutions = 0
        head_backend = self.backend
        head_model = (
            getattr(head_backend, "model", "") or "")
        head_base = getattr(head_backend, "base_url", None)
        role_universe = tuple(sorted(
            {a.effective_role for a in self.agents}))
        n_w42_visible_tokens = 0

        self.orchestrator.reset_session()

        for idx, member in enumerate(self.agents):
            backend = self._resolve_backend(member)
            role = member.effective_role
            ctx = LiveTurnContext(
                turn_index=int(idx),
                role_universe=role_universe,
                role_arrival_order=tuple(role_arrival_order),
                current_role=str(role),
                recent_handoffs=tuple(recent_handoffs),
                all_prior_outputs=tuple(all_prior_outputs),
                causal_counts=dict(causal_counts),
                injected_clock_violation=False,
            )
            obs_result = self.observation_builder(ctx)
            decision, w43 = self.orchestrator.gate(
                observation=obs_result.observation,
                role_handoff_signature_cid=(
                    obs_result.role_handoff_signature_cid),
                parent_w42_cid=self.parent_w42_cid,
                n_w42_visible_tokens=n_w42_visible_tokens,
            )
            visible_count = min(
                len(recent_handoffs), self.max_visible_handoffs)
            (bounded_prompt, textual_prompt,
             n_textual_tokens, n_actual_tokens) = self._build_prompt(
                member=member,
                task=task,
                turn_index=idx,
                recent_handoffs=recent_handoffs,
                all_prior_outputs=all_prior_outputs,
                decision=decision,
                role_universe=role_universe,
                role_arrival_order=role_arrival_order,
            )
            # Decide what to actually run.
            do_substitute = (
                decision.is_abstain()
                and self.registry.abstain_substitution_enabled)
            if do_substitute:
                output = str(self.registry.abstain_output)
                wall_ms = 0.0
                d_prompt = 0
                d_output = 0
                d_calls = 0
                actual_prompt = ""
                n_abstain_substitutions += 1
                n_behavioral_changes += 1
            else:
                actual_prompt = bounded_prompt
                prompt_sha_before = _sha256_str(actual_prompt)  # noqa: F841
                usage_before = _safe_usage_snapshot(backend)
                t0 = time.time()
                output = backend.generate(
                    actual_prompt,
                    max_tokens=member.max_tokens,
                    temperature=member.temperature,
                )
                wall_ms = (time.time() - t0) * 1000.0
                usage_after = _safe_usage_snapshot(backend)
                d_prompt = max(
                    0,
                    int(usage_after["prompt_tokens"])
                    - int(usage_before["prompt_tokens"]),
                )
                d_output = max(
                    0,
                    int(usage_after["output_tokens"])
                    - int(usage_before["output_tokens"]),
                )
                d_calls = max(
                    0,
                    int(usage_after["n_calls"])
                    - int(usage_before["n_calls"]),
                )

            # Token-saving accounting (factoradic compressor).
            n_saved = max(0, int(n_textual_tokens) - int(n_actual_tokens))
            if n_saved > 0 and not do_substitute:
                # Only count saving when the agent actually ran
                # (substitution doesn't send any prompt at all).
                n_visible_tokens_saved_factoradic += int(n_saved)
                # The compressor itself counts as a behavioural change
                # because the prompt bytes that the model saw differ
                # from the textual baseline.
                n_behavioral_changes += 1

            # Seal the underlying TEAM_HANDOFF capsule (preserves the
            # released AgentTeam audit story).
            prompt_sha = _sha256_str(actual_prompt)
            output_sha = _sha256_str(output)
            backend_model = getattr(backend, "model", "") or ""
            capsule_cid: str | None = None
            if ledger is not None:
                next_role = (
                    self.agents[idx + 1].effective_role
                    if idx + 1 < len(self.agents)
                    else "team_output"
                )
                payload_words = max(1, len((output or "").split()))
                if self.handoff_budget is not None:
                    handoff_budget = self.handoff_budget
                else:
                    handoff_max_tokens = max(
                        member.max_tokens,
                        payload_words + 32, 128)
                    handoff_budget = CapsuleBudget(
                        max_bytes=1 << 14,
                        max_tokens=handoff_max_tokens,
                        max_parents=8,
                    )
                claim_kind = (
                    "agent_output_abstain"
                    if do_substitute else "agent_output")
                handoff = capsule_team_handoff(
                    source_role=role,
                    to_role=next_role,
                    claim_kind=claim_kind,
                    payload=output,
                    round=0,
                    parents=(parent_cid,) if parent_cid else (),
                    n_tokens=payload_words,
                    budget=handoff_budget,
                    prompt_sha256=prompt_sha,
                    prompt_bytes=len(
                        actual_prompt.encode("utf-8")),
                    model_tag=backend_model,
                )
                sealed = ledger.admit_and_seal(handoff)
                capsule_cid = sealed.cid
                parent_cid = sealed.cid

            backend_base = getattr(backend, "base_url", None)
            agent_turn = AgentTurn(
                agent_name=member.name,
                role=role,
                prompt=actual_prompt,
                output=output,
                capsule_cid=capsule_cid,
                prompt_tokens=d_prompt,
                output_tokens=d_output,
                wall_ms=wall_ms,
                visible_handoffs=visible_count,
                prompt_sha256=prompt_sha,
                model_tag=backend_model,
                prompt_words=int(n_actual_tokens),
                naive_prompt_words=int(n_textual_tokens),
                temperature=float(member.temperature),
                max_tokens=int(member.max_tokens),
                backend_base_url=backend_base,
            )
            agent_turns.append(agent_turn)

            # Build live envelope.
            mode = self.registry.inline_route_mode
            n_overhead = int(w43.n_w43_overhead_tokens)
            construction_cid = (
                _compute_prompt_construction_witness_cid(
                    turn_index=int(idx),
                    role=str(role),
                    prompt_sha256=prompt_sha,
                    inline_route_mode=str(mode),
                    factoradic_int=int(decision.factoradic_int),
                    factoradic_n_bits=int(decision.factoradic_n_bits),
                    n_visible_prompt_tokens_textual=int(
                        n_textual_tokens),
                    n_visible_prompt_tokens_actual=int(
                        n_actual_tokens),
                ))
            behavioral_change = bool(
                do_substitute or n_saved > 0)
            witness_cid = _compute_w44_live_witness_cid(
                decision_branch=decision.branch,
                pmc_branch=decision.pmc_branch,
                abstain_reason=decision.abstain_reason,
                role_handoff_signature_cid=(
                    decision.role_handoff_signature_cid),
                policy_entry_cid=decision.policy_entry_cid,
                prompt_construction_witness_cid=construction_cid,
                output_sha256=output_sha,
                behavioral_change=behavioral_change,
            )
            outer_cid = _compute_w44_outer_cid(
                schema_cid=self.schema_cid,
                parent_team_handoff_cid=str(capsule_cid or ""),
                parent_w43_envelope_cid=str(decision.pmc_envelope_cid),
                live_witness_cid=witness_cid,
                turn_index=int(idx),
            )
            envelope = LiveManifoldHandoffEnvelope(
                schema_version=W44_LIVE_MANIFOLD_SCHEMA_VERSION,
                schema_cid=self.schema_cid,
                turn_index=int(idx),
                role=str(role),
                parent_team_handoff_cid=str(capsule_cid or ""),
                parent_w43_envelope_cid=str(decision.pmc_envelope_cid),
                parent_w42_cid=str(self.parent_w42_cid),
                decision_branch=decision.branch,
                pmc_branch=decision.pmc_branch,
                abstain_reason=decision.abstain_reason,
                role_handoff_signature_cid=(
                    decision.role_handoff_signature_cid),
                policy_entry_cid=decision.policy_entry_cid,
                inline_route_mode=str(mode),
                factoradic_int=int(decision.factoradic_int),
                factoradic_n_bits=int(decision.factoradic_n_bits),
                prompt_sha256=prompt_sha,
                prompt_construction_witness_cid=construction_cid,
                output_sha256=output_sha,
                n_visible_prompt_tokens_textual=int(n_textual_tokens),
                n_visible_prompt_tokens_actual=int(n_actual_tokens),
                n_visible_prompt_tokens_saved=int(n_saved),
                n_overhead_tokens=int(n_overhead),
                behavioral_change=bool(behavioral_change),
                live_witness_cid=witness_cid,
                live_outer_cid=outer_cid,
            )
            live_turn = LiveManifoldTurn(
                agent_turn=agent_turn,
                decision=decision,
                envelope=envelope,
            )
            live_turns.append(live_turn)

            total_prompt_tokens += int(d_prompt)
            total_output_tokens += int(d_output)
            total_wall_ms += float(wall_ms)
            total_calls += int(d_calls or (0 if do_substitute else 1))

            # Update state for next turn.
            recent_handoffs.append((role, output))
            all_prior_outputs.append((role, output))
            role_arrival_order.append(role)
            if len(recent_handoffs) > self.max_visible_handoffs:
                recent_handoffs = recent_handoffs[
                    -self.max_visible_handoffs:]
            n_w42_visible_tokens = int(visible_count)

            if progress is not None:
                try:
                    progress(live_turn)
                except Exception:
                    import sys as _sys
                    import traceback as _tb
                    print(
                        "[LiveManifoldTeam] progress callback "
                        "raised; continuing run:",
                        file=_sys.stderr)
                    _tb.print_exc()

        view = (
            render_view(
                ledger, root_cid=parent_cid,
                include_payload=True,
            ).as_dict()
            if ledger is not None else None
        )
        final_output = (
            agent_turns[-1].output if agent_turns else "")
        root_cid = (
            view.get("root_cid") if view is not None else None
        ) or parent_cid
        return LiveManifoldTeamResult(
            task=task,
            final_output=final_output,
            turns=tuple(agent_turns),
            live_turns=tuple(live_turns),
            capsule_view=view,
            root_cid=root_cid,
            total_prompt_tokens=int(total_prompt_tokens),
            total_output_tokens=int(total_output_tokens),
            total_wall_ms=float(total_wall_ms),
            total_calls=int(total_calls),
            backend_model=str(head_model),
            backend_base_url=head_base,
            team_instructions=self.team_instructions,
            task_summary=self.task_summary,
            max_visible_handoffs=int(self.max_visible_handoffs),
            stopped_early=False,
            n_behavioral_changes=int(n_behavioral_changes),
            n_visible_tokens_saved_factoradic=int(
                n_visible_tokens_saved_factoradic),
            n_abstain_substitutions=int(n_abstain_substitutions),
        )


# =============================================================================
# Builders
# =============================================================================

def build_trivial_live_manifold_registry(
        *, schema_cid: str | None = None,
) -> LiveManifoldRegistry:
    """Build a registry whose orchestrator reduces to AgentTeam
    byte-for-byte (the W44-L-TRIVIAL-LIVE-PASSTHROUGH falsifier)."""
    cid = schema_cid or _sha256_hex({"kind": "w44_trivial_schema"})
    return LiveManifoldRegistry(
        schema_cid=str(cid),
        pmc_registry=build_trivial_product_manifold_registry(
            schema_cid=str(cid)),
        live_enabled=False,
        inline_route_mode=W44_ROUTE_MODE_TEXTUAL,
        abstain_substitution_enabled=False,
    )


def build_live_manifold_registry(
        *,
        schema_cid: str,
        policy_entries: Sequence[ProductManifoldPolicyEntry] = (),
        live_enabled: bool = True,
        inline_route_mode: str = W44_ROUTE_MODE_TEXTUAL,
        abstain_substitution_enabled: bool = True,
        abstain_output: str = W44_DEFAULT_ABSTAIN_OUTPUT,
        spherical_agreement_min: float = 0.85,
        subspace_drift_max: float = 0.25,
        pmc_enabled: bool = True,
        manifest_v13_disabled: bool = False,
        abstain_on_causal_violation: bool = True,
        abstain_on_subspace_drift: bool = True,
        abstain_on_spherical_divergence: bool = True,
) -> LiveManifoldRegistry:
    """Build a fully configured live registry on top of a W43
    product-manifold registry.

    The W43 inner is constructed from the same ``schema_cid`` so the
    audit chain is binding across both layers.
    """
    if inline_route_mode not in W44_ALL_ROUTE_MODES:
        raise ValueError(
            f"inline_route_mode={inline_route_mode!r} not in "
            f"{W44_ALL_ROUTE_MODES}")
    pmc_inner = build_product_manifold_registry(
        schema_cid=str(schema_cid),
        policy_entries=policy_entries,
        pmc_enabled=bool(pmc_enabled),
        manifest_v13_disabled=bool(manifest_v13_disabled),
        abstain_on_causal_violation=bool(
            abstain_on_causal_violation),
        abstain_on_subspace_drift=bool(abstain_on_subspace_drift),
        abstain_on_spherical_divergence=bool(
            abstain_on_spherical_divergence),
        spherical_agreement_min=float(spherical_agreement_min),
        subspace_drift_max=float(subspace_drift_max),
    )
    return LiveManifoldRegistry(
        schema_cid=str(schema_cid),
        pmc_registry=pmc_inner,
        live_enabled=bool(live_enabled),
        inline_route_mode=str(inline_route_mode),
        abstain_substitution_enabled=bool(
            abstain_substitution_enabled),
        abstain_output=str(abstain_output),
    )


__all__ = [
    # Schema, branches, defaults
    "W44_LIVE_MANIFOLD_SCHEMA_VERSION",
    "W44_TEAM_RESULT_SCHEMA",
    "W44_BRANCH_TRIVIAL_LIVE_PASSTHROUGH",
    "W44_BRANCH_LIVE_DISABLED",
    "W44_BRANCH_LIVE_RATIFIED",
    "W44_BRANCH_LIVE_NO_POLICY",
    "W44_BRANCH_LIVE_CAUSAL_ABSTAIN",
    "W44_BRANCH_LIVE_SPHERICAL_ABSTAIN",
    "W44_BRANCH_LIVE_SUBSPACE_ABSTAIN",
    "W44_BRANCH_LIVE_REJECTED",
    "W44_ALL_BRANCHES",
    "W44_ABSTAIN_BRANCHES",
    "W44_ROUTE_MODE_TEXTUAL",
    "W44_ROUTE_MODE_FACTORADIC",
    "W44_ROUTE_MODE_FACTORADIC_WITH_TEXTUAL",
    "W44_ALL_ROUTE_MODES",
    "W44_DEFAULT_ABSTAIN_OUTPUT",
    "W44_DEFAULT_PARENT_W42_CID",
    # Observation builder
    "LiveTurnContext",
    "LiveObservationBuilder",
    "LiveObservationBuilderResult",
    "default_live_observation_builder",
    # Decision + envelope
    "LiveGatingDecision",
    "LiveManifoldHandoffEnvelope",
    # Orchestrator + team
    "LiveManifoldRegistry",
    "LiveManifoldOrchestrator",
    "LiveManifoldTurn",
    "LiveManifoldTeamResult",
    "LiveManifoldTeam",
    # Verifier
    "LiveManifoldVerificationOutcome",
    "verify_live_manifold_handoff",
    # Builders
    "build_trivial_live_manifold_registry",
    "build_live_manifold_registry",
]
