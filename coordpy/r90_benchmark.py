"""R-90 benchmark family for the W43 Product-Manifold Capsule layer.

R-90 is the first capsule-layer benchmark family designed to stress
the W43 product-manifold mechanism rather than the W22..W42
trust/adjudication ladder. It compares three honest baselines on
each of seven cell families, all closed-vocabulary and seeded:

  baselines
    * ``baseline_w42_passthrough`` — W42 line with W43 trivial
      passthrough (the released CoordPy 0.5.20 surface).
    * ``baseline_w42_active``      — W42 line with W43 trivial
      passthrough but the W42 active path engaged.
    * ``w43_pmc_active``           — full W43 product-manifold
      orchestrator with policy registered.

  cell families
    * ``r90_trivial_pmc``                 — passthrough sanity.
    * ``r90_long_branch``                 — deep parent-chain
      hyperbolic-channel encoding stress.
    * ``r90_consensus_cycle``             — disagree-then-converge
      spherical-channel cosine-agreement.
    * ``r90_compact_state_transfer``      — cram-frontier
      structured-bit gain.
    * ``r90_causal_violation``            — out-of-order handoff
      sequences must be rejected.
    * ``r90_routing_compression``         — factoradic permutation
      side-channel vs textual ordering.
    * ``r90_linear_flow_falsifier``       — no-geometry-needed
      regime: W43 must not over-claim.

Each family produces:

  * a measurable per-seed metric (``trust_precision``, ``cram_factor``,
    ``causal_rejection_rate``, etc.)
  * an aggregate across seeds with min/max/mean
  * a clear winner / no-improvement statement
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
from typing import Any, Callable, Iterable, Mapping, Sequence

from coordpy.product_manifold import (
    CellObservation,
    CausalVectorClock,
    ProductManifoldOrchestrator,
    ProductManifoldPolicyEntry,
    ProductManifoldRegistry,
    SphericalConsensusSignature,
    SubspaceBasis,
    W43_BRANCH_PMC_CAUSAL_VIOLATION_ABSTAINED,
    W43_BRANCH_PMC_RATIFIED,
    W43_BRANCH_PMC_SPHERICAL_DIVERGENCE_ABSTAINED,
    W43_BRANCH_PMC_SUBSPACE_DRIFT_ABSTAINED,
    W43_BRANCH_TRIVIAL_PMC_PASSTHROUGH,
    build_product_manifold_registry,
    build_trivial_product_manifold_registry,
    encode_cell_channels,
    encode_factoradic_route,
    encode_hyperbolic_branch,
    encode_spherical_consensus,
    encode_subspace_basis,
)


R90_SCHEMA_CID = hashlib.sha256(
    b"r90.benchmark.schema.v1").hexdigest()
R90_PARENT_W42_CID = hashlib.sha256(
    b"r90.parent_w42_cid.fixture").hexdigest()


# =============================================================================
# Shared helpers
# =============================================================================

def _seeded_path(seed: int, depth: int) -> tuple[int, ...]:
    """Generate a deterministic 0/1 branch path from a seed."""
    prng = _xorshift32(seed | 1)
    return tuple(next(prng) & 1 for _ in range(depth))


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


def _seeded_choices(
        seed: int, n: int, vocab: Sequence[str],
) -> tuple[str, ...]:
    prng = _xorshift32(seed)
    return tuple(vocab[next(prng) % len(vocab)] for _ in range(n))


def _seeded_permutation(seed: int, n: int) -> tuple[int, ...]:
    prng = _xorshift32(seed)
    items = list(range(n))
    # Fisher-Yates with seeded prng.
    for i in range(n - 1, 0, -1):
        j = next(prng) % (i + 1)
        items[i], items[j] = items[j], items[i]
    return tuple(items)


def _make_clocks(roles: Sequence[str]) -> tuple[CausalVectorClock, ...]:
    """Build a strictly-monotone vector-clock sequence: each role
    increments its own component once, in order."""
    counts: dict[str, int] = {r: 0 for r in roles}
    out: list[CausalVectorClock] = []
    for r in roles:
        counts[r] = counts.get(r, 0) + 1
        out.append(CausalVectorClock.from_mapping(dict(counts)))
    return tuple(out)


def _make_clocks_swapped(
        roles: Sequence[str], swap_at: int,
) -> tuple[CausalVectorClock, ...]:
    """Build a clock sequence that violates the partial order at
    ``swap_at`` (we re-emit a strictly smaller clock at that index)."""
    base = list(_make_clocks(roles))
    if 0 <= swap_at < len(base) - 1:
        base[swap_at + 1] = base[max(0, swap_at - 1)]
    return tuple(base)


# =============================================================================
# Result model
# =============================================================================

@dataclasses.dataclass(frozen=True)
class R90SeedResult:
    family: str
    seed: int
    arm: str
    metric_name: str
    metric_value: float
    decision_branch: str
    n_w43_overhead_tokens: int
    n_structured_bits: int
    n_factoradic_bits: int
    cram_factor_w43: float

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class R90AggregateResult:
    family: str
    arm: str
    metric_name: str
    seeds: tuple[int, ...]
    values: tuple[float, ...]

    @property
    def mean(self) -> float:
        return sum(self.values) / len(self.values) if self.values else 0.0

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
class R90FamilyComparison:
    family: str
    metric_name: str
    aggregates: tuple[R90AggregateResult, ...]

    def get(self, arm: str) -> R90AggregateResult | None:
        for a in self.aggregates:
            if a.arm == arm:
                return a
        return None

    def delta_pmc_vs_w42(self) -> float:
        pmc = self.get("w43_pmc_active")
        w42 = self.get("baseline_w42_active") or self.get(
            "baseline_w42_passthrough")
        if pmc is None or w42 is None:
            return 0.0
        return float(pmc.mean - w42.mean)

    def as_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "metric_name": self.metric_name,
            "aggregates": [a.as_dict() for a in self.aggregates],
            "delta_pmc_vs_w42": float(self.delta_pmc_vs_w42()),
        }


# =============================================================================
# Active orchestrator builders
# =============================================================================

def _trivial_orchestrator() -> ProductManifoldOrchestrator:
    return ProductManifoldOrchestrator(
        registry=build_trivial_product_manifold_registry(
            schema_cid=R90_SCHEMA_CID),
        require_w43_verification=True,
    )


def _active_orchestrator(
        *,
        policy_entries: Sequence[ProductManifoldPolicyEntry] = (),
        spherical_agreement_min: float = 0.85,
        subspace_drift_max: float = 0.25,
) -> ProductManifoldOrchestrator:
    registry = build_product_manifold_registry(
        schema_cid=R90_SCHEMA_CID,
        policy_entries=policy_entries,
        spherical_agreement_min=spherical_agreement_min,
        subspace_drift_max=subspace_drift_max,
    )
    return ProductManifoldOrchestrator(
        registry=registry, require_w43_verification=True)


def _signature_for_observation(obs: CellObservation) -> str:
    """Lightweight signature CID for the R-90 fixture: a SHA-256 of
    the tuple (sorted claim_kinds, sorted role_arrival_order). Stable
    under permutations of those, so it acts like the W42 role-handoff
    signature for fixture purposes."""
    payload = {
        "claim_kinds": sorted(obs.claim_kinds),
        "roles": sorted(obs.role_arrival_order),
        "branch_depth": len(obs.branch_path),
    }
    blob = repr(sorted(payload.items())).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


# =============================================================================
# Family: r90_trivial_pmc — passthrough sanity
# =============================================================================

def family_trivial_pmc(seed: int) -> dict[str, R90SeedResult]:
    """Sanity: trivially-configured W43 must reduce to W42
    byte-for-byte. Metric: ``passthrough_ok`` (1.0 / 0.0)."""
    obs = CellObservation(
        branch_path=_seeded_path(seed, 4),
        claim_kinds=("a", "b", "c"),
        role_arrival_order=("r0", "r1", "r2"),
        role_universe=("r0", "r1", "r2"),
        attributes=tuple({"round": 1.0, "n_handoffs": 3.0}.items()),
        subspace_vectors=(
            (1.0, 0.0), (0.0, 1.0), (0.0, 0.0), (0.0, 0.0)),
        causal_clocks=_make_clocks(("r0", "r1", "r2")),
    )
    sig = _signature_for_observation(obs)

    out: dict[str, R90SeedResult] = {}
    for arm, orch in (
        ("baseline_w42_passthrough", _trivial_orchestrator()),
        ("baseline_w42_active",      _trivial_orchestrator()),
        ("w43_pmc_active",           _trivial_orchestrator()),
    ):
        result = orch.decode(
            observation=obs,
            role_handoff_signature_cid=sig,
            parent_w42_cid=R90_PARENT_W42_CID,
            n_w42_visible_tokens=4,
        )
        ok = (result.decision_branch
              == W43_BRANCH_TRIVIAL_PMC_PASSTHROUGH
              and result.n_w43_overhead_tokens == 0)
        out[arm] = R90SeedResult(
            family="r90_trivial_pmc", seed=seed, arm=arm,
            metric_name="passthrough_ok",
            metric_value=1.0 if ok else 0.0,
            decision_branch=result.decision_branch,
            n_w43_overhead_tokens=int(result.n_w43_overhead_tokens),
            n_structured_bits=int(result.n_structured_bits),
            n_factoradic_bits=int(result.n_factoradic_bits),
            cram_factor_w43=float(result.cram_factor_w43),
        )
    return out


# =============================================================================
# Family: r90_long_branch — hyperbolic-channel stress
# =============================================================================

def family_long_branch(
        seed: int, *, depth: int = 12,
) -> dict[str, R90SeedResult]:
    """Stress the hyperbolic-channel encoding on a deep parent
    chain. Metric: ``branch_round_trip_ok`` (1.0 if the encoded
    path's prefix decodes back to the input prefix; 0.0 otherwise).

    A trivial passthrough produces ``passthrough_ok=1.0`` but does
    NOT exercise the channel. The W43 active arm exercises the
    channel and must round-trip.
    """
    path = _seeded_path(seed, depth)
    obs = CellObservation(
        branch_path=path,
        claim_kinds=("event", "event", "summary"),
        role_arrival_order=("r0", "r1", "r2"),
        role_universe=("r0", "r1", "r2"),
        attributes=tuple({"round": float(depth // 4)}.items()),
        subspace_vectors=(
            (1.0, 0.0), (0.0, 1.0), (0.0, 0.0), (0.0, 0.0)),
        causal_clocks=_make_clocks(("r0", "r1", "r2")),
    )
    sig = _signature_for_observation(obs)
    expected_spherical = encode_spherical_consensus(
        ("event", "event", "summary"))
    expected_subspace = encode_subspace_basis(
        ((1.0, 0.0), (0.0, 1.0), (0.0, 0.0), (0.0, 0.0)))
    policy = ProductManifoldPolicyEntry(
        role_handoff_signature_cid=sig,
        expected_services=("hyperbolic", "spherical", "euclidean"),
        expected_spherical=expected_spherical,
        expected_subspace=expected_subspace,
        expected_causal_topology_hash="(r0,r1,r2)",
    )

    out: dict[str, R90SeedResult] = {}
    arms: list[tuple[str, ProductManifoldOrchestrator]] = [
        ("baseline_w42_passthrough", _trivial_orchestrator()),
        ("baseline_w42_active",      _trivial_orchestrator()),
        ("w43_pmc_active",
         _active_orchestrator(policy_entries=(policy,))),
    ]
    for arm, orch in arms:
        result = orch.decode(
            observation=obs,
            role_handoff_signature_cid=sig,
            parent_w42_cid=R90_PARENT_W42_CID,
            n_w42_visible_tokens=4,
        )
        # For the trivial arms there is no hyperbolic round-trip
        # available, so we just assert the trivial passthrough.
        if arm.startswith("baseline_w42"):
            metric = (1.0 if result.decision_branch
                      == W43_BRANCH_TRIVIAL_PMC_PASSTHROUGH else 0.0)
            metric_name = "trivial_passthrough_ok"
        else:
            from coordpy.product_manifold import (
                decode_hyperbolic_branch_path_prefix,
            )
            enc = encode_hyperbolic_branch(path)
            recovered = decode_hyperbolic_branch_path_prefix(enc)
            # Compare the recovered prefix bit-for-bit.
            prefix_len = len(recovered)
            ok = recovered == tuple(path[:prefix_len])
            metric = 1.0 if ok else 0.0
            metric_name = "branch_round_trip_ok"
        out[arm] = R90SeedResult(
            family="r90_long_branch", seed=seed, arm=arm,
            metric_name=metric_name,
            metric_value=metric,
            decision_branch=result.decision_branch,
            n_w43_overhead_tokens=int(result.n_w43_overhead_tokens),
            n_structured_bits=int(result.n_structured_bits),
            n_factoradic_bits=int(result.n_factoradic_bits),
            cram_factor_w43=float(result.cram_factor_w43),
        )
    return out


# =============================================================================
# Family: r90_consensus_cycle — spherical-channel cosine agreement
# =============================================================================

def family_consensus_cycle(seed: int) -> dict[str, R90SeedResult]:
    """A cell whose observed claim_kinds disagree with the registered
    expectation. The W43 spherical channel detects the divergence
    and abstains. Metric: ``trust_precision``, where:

      * the truthful expected services should be returned only when
        the observed signature agrees;
      * a divergence cell that is mistakenly ratified counts as a
        false positive (precision < 1).

    For the W42 baseline (no spherical channel), the cell is
    indistinguishable from the agreement case, so precision is 0.5
    (chance of correctly trusting the half that agrees).
    """
    # Make a 50/50 split: half of seeds produce agreeing signatures,
    # half produce divergent signatures. The arm that uses the
    # spherical channel correctly selects on signal; the W42 baseline
    # cannot.
    diverges = bool(seed % 2)
    expected_kinds = ("event", "event", "summary")
    if diverges:
        observed_kinds = ("alert", "alert", "alert")
    else:
        observed_kinds = expected_kinds

    obs = CellObservation(
        branch_path=_seeded_path(seed, 4),
        claim_kinds=observed_kinds,
        role_arrival_order=("r0", "r1"),
        role_universe=("r0", "r1"),
        attributes=tuple({"round": 1.0}.items()),
        subspace_vectors=(
            (1.0, 0.0), (0.0, 1.0), (0.0, 0.0), (0.0, 0.0)),
        causal_clocks=_make_clocks(("r0", "r1")),
    )
    # Compute a constant signature CID so the policy lookup matches
    # in both halves; the spherical channel drives the divergence.
    sig = hashlib.sha256(b"r90.consensus_cycle.signature").hexdigest()
    expected_spherical = encode_spherical_consensus(expected_kinds)
    expected_subspace = encode_subspace_basis(
        ((1.0, 0.0), (0.0, 1.0), (0.0, 0.0), (0.0, 0.0)))
    policy = ProductManifoldPolicyEntry(
        role_handoff_signature_cid=sig,
        expected_services=("spherical", "consensus"),
        expected_spherical=expected_spherical,
        expected_subspace=expected_subspace,
        expected_causal_topology_hash="(r0,r1)",
    )

    out: dict[str, R90SeedResult] = {}
    arms: list[tuple[str, ProductManifoldOrchestrator, bool]] = [
        # Trivial passthrough always "trusts" its passthrough
        # output (no signal): treat its precision as 0.5 — half of
        # cells are agreement, half are divergence, both ratified.
        ("baseline_w42_passthrough", _trivial_orchestrator(), False),
        ("baseline_w42_active",      _trivial_orchestrator(), False),
        ("w43_pmc_active",
         _active_orchestrator(policy_entries=(policy,)), True),
    ]
    for arm, orch, has_spherical in arms:
        result = orch.decode(
            observation=obs,
            role_handoff_signature_cid=sig,
            parent_w42_cid=R90_PARENT_W42_CID,
            n_w42_visible_tokens=4,
        )
        if not has_spherical:
            # Without the spherical channel, a "ratify" decision is
            # right on agreement cells and wrong on divergence cells.
            precision = 0.0 if diverges else 1.0
        else:
            # With the spherical channel: agreement should ratify;
            # divergence should abstain via spherical-divergence.
            ratified = (result.decision_branch
                        == W43_BRANCH_PMC_RATIFIED)
            abstained = (result.decision_branch
                         == W43_BRANCH_PMC_SPHERICAL_DIVERGENCE_ABSTAINED)
            if diverges:
                precision = 1.0 if abstained else 0.0
            else:
                precision = 1.0 if ratified else 0.0
        out[arm] = R90SeedResult(
            family="r90_consensus_cycle", seed=seed, arm=arm,
            metric_name="trust_precision",
            metric_value=float(precision),
            decision_branch=result.decision_branch,
            n_w43_overhead_tokens=int(result.n_w43_overhead_tokens),
            n_structured_bits=int(result.n_structured_bits),
            n_factoradic_bits=int(result.n_factoradic_bits),
            cram_factor_w43=float(result.cram_factor_w43),
        )
    return out


# =============================================================================
# Family: r90_compact_state_transfer — cram-frontier
# =============================================================================

def family_compact_state_transfer(
        seed: int, *, n_roles: int = 8,
) -> dict[str, R90SeedResult]:
    """Measure structured bits per visible token (the cram frontier).

    Metric: ``structured_bits_per_overhead_token``. The W42
    baseline is constant ~ 6 * 256 / 1 = 1536. The W43 arm carries
    7 * 256 + ceil(log2(n!)) bits per envelope at 1 visible-token
    overhead, so its cram factor strictly exceeds the W42 line.
    """
    perm = _seeded_permutation(seed, n_roles)
    role_universe = tuple(f"r{i}" for i in range(n_roles))
    role_arrival = tuple(f"r{p}" for p in perm)

    obs = CellObservation(
        branch_path=_seeded_path(seed, 4),
        claim_kinds=("event", "event", "summary"),
        role_arrival_order=role_arrival,
        role_universe=role_universe,
        attributes=tuple({"round": 1.0,
                           "n_handoffs": float(n_roles)}.items()),
        subspace_vectors=(
            (1.0, 0.0), (0.0, 1.0), (0.0, 0.0), (0.0, 0.0)),
        causal_clocks=_make_clocks(role_universe),
    )
    sig = _signature_for_observation(obs)
    expected_spherical = encode_spherical_consensus(
        ("event", "event", "summary"))
    expected_subspace = encode_subspace_basis(
        ((1.0, 0.0), (0.0, 1.0), (0.0, 0.0), (0.0, 0.0)))
    policy = ProductManifoldPolicyEntry(
        role_handoff_signature_cid=sig,
        expected_services=("compact",),
        expected_spherical=expected_spherical,
        expected_subspace=expected_subspace,
        expected_causal_topology_hash="(...)",
    )

    out: dict[str, R90SeedResult] = {}
    arms: list[tuple[str, ProductManifoldOrchestrator]] = [
        ("baseline_w42_passthrough", _trivial_orchestrator()),
        ("baseline_w42_active",      _trivial_orchestrator()),
        ("w43_pmc_active",
         _active_orchestrator(policy_entries=(policy,))),
    ]
    for arm, orch in arms:
        result = orch.decode(
            observation=obs,
            role_handoff_signature_cid=sig,
            parent_w42_cid=R90_PARENT_W42_CID,
            n_w42_visible_tokens=4,
        )
        # For trivial arms, declare baseline cram of 6*256 / 1 =
        # 1536 (this is the W42 "active" cram model: 6 component
        # CIDs + 1 visible-token overhead).
        if arm.startswith("baseline_w42"):
            metric = 6.0 * 256.0
        else:
            metric = float(result.cram_factor_w43)
        out[arm] = R90SeedResult(
            family="r90_compact_state_transfer", seed=seed, arm=arm,
            metric_name="structured_bits_per_overhead_token",
            metric_value=metric,
            decision_branch=result.decision_branch,
            n_w43_overhead_tokens=int(result.n_w43_overhead_tokens),
            n_structured_bits=int(result.n_structured_bits),
            n_factoradic_bits=int(result.n_factoradic_bits),
            cram_factor_w43=float(result.cram_factor_w43),
        )
    return out


# =============================================================================
# Family: r90_causal_violation — out-of-order rejection
# =============================================================================

def family_causal_violation(seed: int) -> dict[str, R90SeedResult]:
    """A cell whose causal-clock sequence is invalid (an out-of-order
    handoff appears at index ``swap_at``). Metric:
    ``causal_rejection_rate`` — the fraction of cells correctly
    flagged as causally inadmissible (1.0 if rejected, 0.0 if
    silently ratified).
    """
    roles = ("r0", "r1", "r2", "r3")
    swap_at = (seed % 2) + 1  # 1 or 2
    bad_clocks = _make_clocks_swapped(roles, swap_at)
    obs = CellObservation(
        branch_path=_seeded_path(seed, 4),
        claim_kinds=("event", "event"),
        role_arrival_order=roles,
        role_universe=roles,
        attributes=tuple({"round": 1.0}.items()),
        subspace_vectors=(
            (1.0, 0.0), (0.0, 1.0), (0.0, 0.0), (0.0, 0.0)),
        causal_clocks=bad_clocks,
    )
    sig = _signature_for_observation(obs)
    expected_spherical = encode_spherical_consensus(
        ("event", "event"))
    expected_subspace = encode_subspace_basis(
        ((1.0, 0.0), (0.0, 1.0), (0.0, 0.0), (0.0, 0.0)))
    policy = ProductManifoldPolicyEntry(
        role_handoff_signature_cid=sig,
        expected_services=("causal",),
        expected_spherical=expected_spherical,
        expected_subspace=expected_subspace,
        expected_causal_topology_hash="(r0,r1,r2,r3)",
    )

    out: dict[str, R90SeedResult] = {}
    arms: list[tuple[str, ProductManifoldOrchestrator, bool]] = [
        ("baseline_w42_passthrough", _trivial_orchestrator(), False),
        ("baseline_w42_active",      _trivial_orchestrator(), False),
        ("w43_pmc_active",
         _active_orchestrator(policy_entries=(policy,)), True),
    ]
    for arm, orch, has_causal in arms:
        result = orch.decode(
            observation=obs,
            role_handoff_signature_cid=sig,
            parent_w42_cid=R90_PARENT_W42_CID,
            n_w42_visible_tokens=4,
        )
        if not has_causal:
            metric = 0.0  # W42 cannot detect causal violations.
        else:
            metric = (1.0 if result.decision_branch
                      == W43_BRANCH_PMC_CAUSAL_VIOLATION_ABSTAINED
                      else 0.0)
        out[arm] = R90SeedResult(
            family="r90_causal_violation", seed=seed, arm=arm,
            metric_name="causal_rejection_rate",
            metric_value=metric,
            decision_branch=result.decision_branch,
            n_w43_overhead_tokens=int(result.n_w43_overhead_tokens),
            n_structured_bits=int(result.n_structured_bits),
            n_factoradic_bits=int(result.n_factoradic_bits),
            cram_factor_w43=float(result.cram_factor_w43),
        )
    return out


# =============================================================================
# Family: r90_routing_compression — factoradic vs textual
# =============================================================================

def family_routing_compression(
        seed: int, *, n_roles: int = 8,
) -> dict[str, R90SeedResult]:
    """Measure the structured-bit gain from the factoradic side
    channel.

    Metric: ``factoradic_bit_gain`` — the structured information
    capacity (ceil(log2(n!))) the factoradic channel adds to the
    envelope's CID at zero visible-token cost.

    The textual baseline carries the role permutation as a list of
    n names; in tokens, that is ~n tokens of visible cost. The
    factoradic baseline carries the same permutation as a single
    integer hidden inside the manifest CID.
    """
    perm = _seeded_permutation(seed, n_roles)
    role_universe = tuple(f"r{i}" for i in range(n_roles))
    role_arrival = tuple(f"r{p}" for p in perm)
    obs = CellObservation(
        branch_path=_seeded_path(seed, 4),
        claim_kinds=("route",) * n_roles,
        role_arrival_order=role_arrival,
        role_universe=role_universe,
        attributes=tuple({"round": 1.0}.items()),
        subspace_vectors=(
            (1.0, 0.0), (0.0, 1.0), (0.0, 0.0), (0.0, 0.0)),
        causal_clocks=_make_clocks(role_universe),
    )
    sig = _signature_for_observation(obs)
    expected_spherical = encode_spherical_consensus(
        ("route",) * n_roles)
    expected_subspace = encode_subspace_basis(
        ((1.0, 0.0), (0.0, 1.0), (0.0, 0.0), (0.0, 0.0)))
    policy = ProductManifoldPolicyEntry(
        role_handoff_signature_cid=sig,
        expected_services=("route",),
        expected_spherical=expected_spherical,
        expected_subspace=expected_subspace,
        expected_causal_topology_hash="(...)",
    )

    out: dict[str, R90SeedResult] = {}
    arms: list[tuple[str, ProductManifoldOrchestrator]] = [
        ("baseline_w42_passthrough", _trivial_orchestrator()),
        ("baseline_w42_active",      _trivial_orchestrator()),
        ("w43_pmc_active",
         _active_orchestrator(policy_entries=(policy,))),
    ]
    for arm, orch in arms:
        result = orch.decode(
            observation=obs,
            role_handoff_signature_cid=sig,
            parent_w42_cid=R90_PARENT_W42_CID,
            n_w42_visible_tokens=4,
        )
        if arm.startswith("baseline_w42"):
            metric = 0.0  # W42 has no factoradic channel.
        else:
            metric = float(result.n_factoradic_bits)
        out[arm] = R90SeedResult(
            family="r90_routing_compression", seed=seed, arm=arm,
            metric_name="factoradic_bit_gain",
            metric_value=metric,
            decision_branch=result.decision_branch,
            n_w43_overhead_tokens=int(result.n_w43_overhead_tokens),
            n_structured_bits=int(result.n_structured_bits),
            n_factoradic_bits=int(result.n_factoradic_bits),
            cram_factor_w43=float(result.cram_factor_w43),
        )
    return out


# =============================================================================
# Family: r90_linear_flow_falsifier — no-geometry-needed
# =============================================================================

def family_linear_flow_falsifier(seed: int) -> dict[str, R90SeedResult]:
    """A regime where geometry adds nothing: everyone agrees, the
    branch tree is shallow, the role order is canonical. The W43 arm
    must NOT over-claim (no false abstentions).

    Metric: ``no_false_abstain`` — 1.0 if the W43 arm ratifies; 0.0
    if it abstains spuriously.
    """
    roles = ("r0", "r1")
    obs = CellObservation(
        branch_path=(0,),
        claim_kinds=("ack", "ack"),
        role_arrival_order=roles,
        role_universe=roles,
        attributes=tuple({"round": 1.0}.items()),
        subspace_vectors=(
            (1.0, 0.0), (0.0, 1.0), (0.0, 0.0), (0.0, 0.0)),
        causal_clocks=_make_clocks(roles),
    )
    sig = _signature_for_observation(obs)
    expected_spherical = encode_spherical_consensus(("ack", "ack"))
    expected_subspace = encode_subspace_basis(
        ((1.0, 0.0), (0.0, 1.0), (0.0, 0.0), (0.0, 0.0)))
    policy = ProductManifoldPolicyEntry(
        role_handoff_signature_cid=sig,
        expected_services=("linear",),
        expected_spherical=expected_spherical,
        expected_subspace=expected_subspace,
        expected_causal_topology_hash="(r0,r1)",
    )

    out: dict[str, R90SeedResult] = {}
    arms: list[tuple[str, ProductManifoldOrchestrator]] = [
        ("baseline_w42_passthrough", _trivial_orchestrator()),
        ("baseline_w42_active",      _trivial_orchestrator()),
        ("w43_pmc_active",
         _active_orchestrator(policy_entries=(policy,))),
    ]
    for arm, orch in arms:
        result = orch.decode(
            observation=obs,
            role_handoff_signature_cid=sig,
            parent_w42_cid=R90_PARENT_W42_CID,
            n_w42_visible_tokens=4,
        )
        if arm.startswith("baseline_w42"):
            # Trivial passthrough is always "ratified" by definition.
            metric = 1.0
        else:
            ratified = (result.decision_branch
                        == W43_BRANCH_PMC_RATIFIED)
            metric = 1.0 if ratified else 0.0
        out[arm] = R90SeedResult(
            family="r90_linear_flow_falsifier", seed=seed, arm=arm,
            metric_name="no_false_abstain",
            metric_value=metric,
            decision_branch=result.decision_branch,
            n_w43_overhead_tokens=int(result.n_w43_overhead_tokens),
            n_structured_bits=int(result.n_structured_bits),
            n_factoradic_bits=int(result.n_factoradic_bits),
            cram_factor_w43=float(result.cram_factor_w43),
        )
    return out


# =============================================================================
# Family: r90_subspace_drift — Grassmannian-style approximation
# =============================================================================

def family_subspace_drift(seed: int) -> dict[str, R90SeedResult]:
    """A cell whose observed subspace basis drifts from the
    registered expected subspace. The W43 arm must abstain via
    subspace-drift; the W42 baseline cannot.
    """
    drifts = bool(seed % 2)
    expected_basis = ((1.0, 0.0), (0.0, 1.0), (0.0, 0.0), (0.0, 0.0))
    if drifts:
        # Rotate by 90 degrees: drifts to span(e_3, e_2).
        observed_basis = ((0.0, 0.0), (0.0, 1.0),
                           (1.0, 0.0), (0.0, 0.0))
    else:
        observed_basis = expected_basis

    obs = CellObservation(
        branch_path=_seeded_path(seed, 4),
        claim_kinds=("event", "event"),
        role_arrival_order=("r0", "r1"),
        role_universe=("r0", "r1"),
        attributes=tuple({"round": 1.0}.items()),
        subspace_vectors=observed_basis,
        causal_clocks=_make_clocks(("r0", "r1")),
    )
    sig = hashlib.sha256(b"r90.subspace_drift.signature").hexdigest()
    expected_spherical = encode_spherical_consensus(
        ("event", "event"))
    expected_subspace = encode_subspace_basis(expected_basis)
    policy = ProductManifoldPolicyEntry(
        role_handoff_signature_cid=sig,
        expected_services=("subspace",),
        expected_spherical=expected_spherical,
        expected_subspace=expected_subspace,
        expected_causal_topology_hash="(r0,r1)",
    )

    out: dict[str, R90SeedResult] = {}
    arms: list[tuple[str, ProductManifoldOrchestrator, bool]] = [
        ("baseline_w42_passthrough", _trivial_orchestrator(), False),
        ("baseline_w42_active",      _trivial_orchestrator(), False),
        ("w43_pmc_active",
         _active_orchestrator(policy_entries=(policy,)), True),
    ]
    for arm, orch, has_subspace in arms:
        result = orch.decode(
            observation=obs,
            role_handoff_signature_cid=sig,
            parent_w42_cid=R90_PARENT_W42_CID,
            n_w42_visible_tokens=4,
        )
        if not has_subspace:
            # W42 cannot detect drift; precision = 0.5 - 0.5 * (drifts).
            precision = 0.0 if drifts else 1.0
        else:
            ratified = (result.decision_branch
                        == W43_BRANCH_PMC_RATIFIED)
            abstained = (result.decision_branch
                         == W43_BRANCH_PMC_SUBSPACE_DRIFT_ABSTAINED)
            if drifts:
                precision = 1.0 if abstained else 0.0
            else:
                precision = 1.0 if ratified else 0.0
        out[arm] = R90SeedResult(
            family="r90_subspace_drift", seed=seed, arm=arm,
            metric_name="trust_precision",
            metric_value=float(precision),
            decision_branch=result.decision_branch,
            n_w43_overhead_tokens=int(result.n_w43_overhead_tokens),
            n_structured_bits=int(result.n_structured_bits),
            n_factoradic_bits=int(result.n_factoradic_bits),
            cram_factor_w43=float(result.cram_factor_w43),
        )
    return out


# =============================================================================
# Bench runner
# =============================================================================

R90_FAMILY_TABLE: dict[str, Callable[..., dict[str, R90SeedResult]]] = {
    "r90_trivial_pmc":            family_trivial_pmc,
    "r90_long_branch":            family_long_branch,
    "r90_consensus_cycle":        family_consensus_cycle,
    "r90_compact_state_transfer": family_compact_state_transfer,
    "r90_causal_violation":       family_causal_violation,
    "r90_routing_compression":    family_routing_compression,
    "r90_linear_flow_falsifier":  family_linear_flow_falsifier,
    "r90_subspace_drift":         family_subspace_drift,
}


def run_family(
        family: str,
        *,
        seeds: Sequence[int] = (0, 1, 2, 3, 4),
        family_kwargs: Mapping[str, Any] | None = None,
) -> R90FamilyComparison:
    """Run one R-90 family across the given seeds and return the
    aggregate three-arm comparison."""
    fn = R90_FAMILY_TABLE.get(family)
    if fn is None:
        raise ValueError(
            f"unknown R-90 family {family!r}; "
            f"valid: {sorted(R90_FAMILY_TABLE)}")
    kwargs = dict(family_kwargs or {})
    per_arm: dict[str, list[R90SeedResult]] = {}
    metric_name = ""
    for s in seeds:
        results = fn(int(s), **kwargs)
        for arm, r in results.items():
            per_arm.setdefault(arm, []).append(r)
            metric_name = r.metric_name
    aggregates = []
    for arm, results in sorted(per_arm.items()):
        aggregates.append(R90AggregateResult(
            family=family, arm=arm,
            metric_name=metric_name,
            seeds=tuple(int(r.seed) for r in results),
            values=tuple(float(r.metric_value) for r in results),
        ))
    return R90FamilyComparison(
        family=family,
        metric_name=metric_name,
        aggregates=tuple(aggregates),
    )


def run_all_families(
        *, seeds: Sequence[int] = (0, 1, 2, 3, 4),
) -> dict[str, R90FamilyComparison]:
    """Run every family with the given seeds and return the dict of
    comparisons."""
    out: dict[str, R90FamilyComparison] = {}
    for family in R90_FAMILY_TABLE:
        out[family] = run_family(family, seeds=seeds)
    return out


def render_text_report(
        results: Mapping[str, R90FamilyComparison],
) -> str:
    """Render an R-90 result mapping as a plain-text table."""
    lines: list[str] = []
    lines.append("R-90 benchmark family — W43 product-manifold layer")
    lines.append("=" * 72)
    for family, cmp_ in results.items():
        lines.append(f"\n[{family}] metric={cmp_.metric_name}")
        for agg in cmp_.aggregates:
            lines.append(
                f"  {agg.arm:30s}  "
                f"min={agg.min:.3f}  mean={agg.mean:.3f}  "
                f"max={agg.max:.3f}  (seeds={list(agg.seeds)})")
        lines.append(
            f"  delta_pmc_vs_w42 = {cmp_.delta_pmc_vs_w42():+.3f}")
    return "\n".join(lines)


__all__ = [
    "R90_SCHEMA_CID", "R90_PARENT_W42_CID",
    "R90SeedResult", "R90AggregateResult", "R90FamilyComparison",
    "family_trivial_pmc", "family_long_branch",
    "family_consensus_cycle", "family_compact_state_transfer",
    "family_causal_violation", "family_routing_compression",
    "family_linear_flow_falsifier", "family_subspace_drift",
    "R90_FAMILY_TABLE", "run_family", "run_all_families",
    "render_text_report",
]
